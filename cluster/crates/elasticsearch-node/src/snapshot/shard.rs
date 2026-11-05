//! Data-node side of snapshots: freezing a primary's files (uploaded by
//! the node itself, [`super::upload`], or served to a master that pulls
//! them from a node without that capability), plus repository verification.
//!
//! **Freezing a shard copy.** The node holding the primary first ships
//! every op to its replicas, then takes the primary's write gate (no write
//! can run), flushes — only when no copy still needs translog ops, exactly
//! as the gated `_flush` path does, saving the vector graphs first when the
//! commit's graph checkpoint lags it by more than a file recovery accepts
//! (`indices.recovery.graph_checkpoint_lag_ops`), so a restore adopting the
//! image opens the graphs as they were — and hard-links the committed engine
//! files (`index_meta.json`, `shard-0/shard_state.json` and the files of
//! `shard-0/core-<gen>/` the core's commit point references, as file
//! recovery ships them) into `snapshot-staging/<uuid>/<index>/`. The
//! engine only ever writes files through tmp + rename, so a hard link is an
//! immutable image: later merges / refreshes / flushes cannot change it.
//! When the flush has to be skipped (a recovery is still replaying the
//! translog) the translog files are *copied* instead, and opening the image
//! replays them. The gate is released as soon as the links exist; the
//! (possibly slow) upload reads the staging copy.

use super::common::{dir_name, repo_context, safe_rel, staging_root};
use crate::node::{ClusterNode, TransportNet};
use crate::front::SharedText;
use crate::shared_sequence::SharedSequence;
use elasticsearch_engine::error::EsError;
use elasticsearch_snapshots::{Repository, RepositoryMetadata};
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const ACTION_PREPARE: &str = "internal:admin/snapshot/shard/prepare";
pub const ACTION_READ: &str = "internal:admin/snapshot/shard/read";
pub const ACTION_RELEASE: &str = "internal:admin/snapshot/shard/release";
pub const ACTION_VERIFY: &str = "internal:admin/repository/verify";

/// Bytes per file chunk pulled by the master (nodes that cannot upload).
pub const CHUNK: u64 = 4 << 20;
/// Longest a shard freeze waits for a tracked copy that is not in sync
/// (recovering) before staging an unflushed image (see [`prepare`]).
const LAGGING_COPY_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Test seam (hidden): the next `n` shard freezes stage an unflushed image
/// (commit plus translog) as when a copy is still recovering.
#[doc(hidden)]
pub static SKIP_FLUSHES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrepareReq {
    pub index: SharedText,
    pub shard: u32,
    pub snapshot_uuid: SharedText,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrepareResp {
    /// Absolute staging directory on the preparing node.
    pub dir: SharedText,
    /// (relative path, length).
    pub files: SharedSequence<(SharedText, u64)>,
    pub flushed: bool,
    pub max_seq_no: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadReq {
    pub snapshot_uuid: SharedText,
    pub index: SharedText,
    /// Shard number (absent from pre-multi-shard senders: shard 0).
    #[serde(default)]
    pub shard: u32,
    pub path: SharedText,
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseReq {
    pub snapshot_uuid: SharedText,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyReq {
    pub repository: SharedText,
    pub repo_type: SharedText,
    pub settings_json: SharedText,
    pub seed: SharedText,
}

fn remote(kind: &str, reason: impl Into<String>, status: u16) -> RemoteError {
    RemoteError::new(kind, reason.into(), status)
}

pub fn register(node: &Arc<ClusterNode>) {
    // Leftovers of snapshots / restores interrupted by a restart.
    let _ = std::fs::remove_dir_all(staging_root(node));
    let t = &node.transport;
    let n = node.clone();
    t.register_handler(ACTION_PREPARE, move |r: PrepareReq| {
        let n = n.clone();
        async move { prepare(&n, &r).await.map_err(|e|RemoteError::new(&e.kind,e.reason,e.status)) }
    });
    let n = node.clone();
    t.register_handler(ACTION_READ, move |r: ReadReq| {
        let n = n.clone();
        async move {
            let root = OwnedPath::join(&n.config.data_path,std::ffi::OsStr::new("snapshot-staging")).map_err(|e|RemoteError::new(&e.kind,e.reason,e.status))?;
            tokio::task::spawn_blocking(move || read_chunk(root.as_path(), &r)).await.map_err(|e| remote("exception", e.to_string(), 500))?.map_err(|e|RemoteError::new(&e.kind,e.reason,e.status))
        }
    });
    let n = node.clone();
    t.register_handler(ACTION_RELEASE, move |r: ReleaseReq| {
        let n = n.clone();
        async move {
            release(&n, &r.snapshot_uuid);
            Ok(())
        }
    });
    let n = node.clone();
    t.register_handler(ACTION_VERIFY, move |r: VerifyReq| {
        let n = n.clone();
        async move { verify_local(&n, &r).await }
    });
}

/// A path copy whose storage is admitted before construction and lives through
/// the filesystem operation (including blocking-worker handoff).
struct OwnedPath { value:PathBuf, _owner:elasticsearch_engine::memory::Reservation }
impl OwnedPath {
    fn as_path(&self)->&Path{&self.value}
    fn join(base:&Path, rel:&std::ffi::OsStr)->Result<Self,EsError>{
        let bytes=base.as_os_str().as_encoded_bytes().len().checked_add(rel.as_encoded_bytes().len()).and_then(|n|n.checked_add(1)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","snapshot path overflow"))?;
        let owner=elasticsearch_engine::memory::breakers().reserve_request(bytes as u64,"snapshot path")?.into_result_owner();
        let mut value=PathBuf::new();value.try_reserve_exact(bytes).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        value.push(base);value.push(rel);Ok(Self{value,_owner:owner})
    }
}
struct SafeComponent<'a>(&'a str);
impl std::fmt::Display for SafeComponent<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{use std::fmt::Write;for c in self.0.chars(){f.write_char(if c.is_ascii_alphanumeric()||matches!(c,'-'|'_'|'.'){c}else{'_'})?;}Ok(())}}
fn owned_staging_path(root:&Path,uuid:&str,index:&str,shard:u32)->Result<OwnedPath,EsError>{
    let rel=SharedText::try_format(format_args!("{}/{}/{}",SafeComponent(uuid),SafeComponent(index),shard))?;
    OwnedPath::join(root,std::ffi::OsStr::new(rel.as_str()))
}

pub fn staging_dir(node: &ClusterNode, snapshot_uuid: &str, index: &str, shard: u32) -> PathBuf {
    staging_path(&staging_root(node), snapshot_uuid, index, shard)
}

/// `<root>/<uuid>/<index>/<shard>`: one image per shard copy, so a node
/// holding several primaries of the same index stages them side by side.
pub fn staging_path(root: &Path, snapshot_uuid: &str, index: &str, shard: u32) -> PathBuf {
    root.join(dir_name(snapshot_uuid)).join(dir_name(index)).join(shard.to_string())
}

pub fn release(node: &ClusterNode, snapshot_uuid: &str) {
    super::upload::forget(node, snapshot_uuid);
    let _ = std::fs::remove_dir_all(staging_root(node).join(dir_name(snapshot_uuid)));
}

/// Freeze the local primary of `r.index` into its staging directory.
pub async fn prepare(node: &Arc<ClusterNode>, r: &PrepareReq) -> Result<PrepareResp, EsError> {
    let p = node.shards.primary(&r.index, r.shard).ok_or_else(|| EsError::new(500,"snapshot_exception",format!("[{}][{}] primary shard is not on node [{}]", r.index, r.shard, node.local.name)))?;
    let net = TransportNet(node.clone());
    if let Err(e) = p.replicate_all(&net).await {
        tracing::warn!(index = %r.index, "replication before snapshot failed: {e}");
    }
    // ES always snapshots a flushed commit, so the image keeps the shard's
    // segments (a timestamp-sorted index's leaf order depends on them). We
    // may flush only once no tracked copy still needs translog ops. In-sync
    // copies are caught up with the write gate held (under continuous
    // writes they would otherwise always look behind); a copy still
    // recovering (a replica allocated just after the index was created)
    // usually finishes within moments, so wait briefly for it, with the gate
    // released, before falling back to an unflushed image whose translog
    // replays into a single segment.
    let (_gate, max, lagging) = p.freeze_for_snapshot(&net, LAGGING_COPY_WAIT).await;
    let skip = SKIP_FLUSHES.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |n| n.checked_sub(1)).is_ok();
    // The flush also brings the commit's graph checkpoint within the lag a
    // file recovery accepts (what a file restore inserts at open), so a
    // restored copy opens the snapshotted graphs as they were.
    let graph_lag = node.effective_setting(crate::replication::file_recovery::GRAPH_LAG_OPS).and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(50_000);
    let copy = p.copy();
    let root=OwnedPath::join(&node.config.data_path,std::ffi::OsStr::new("snapshot-staging"))?;
    let dest = owned_staging_path(&root.value, &r.snapshot_uuid, &r.index, r.shard)?;
    let res = tokio::task::spawn_blocking(move || stage(&copy, dest.as_path(), !lagging && !skip, graph_lag)).await.map_err(|e| EsError::internal(e.to_string()))?;
    let (dir, files, flushed) = res?;
    Ok(PrepareResp { dir, files, flushed, max_seq_no: max })
}

fn link_or_copy(src: &Path, dst: &Path) -> std::io::Result<()> {
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p)?;
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => std::fs::copy(src, dst).map(|_| ()),
    }
}

fn snapshot_io(error:std::io::Error)->EsError{EsError::new(500,"i_o_exception",error.to_string())}
fn entry_name(entry:&std::fs::DirEntry)->Result<(std::ffi::OsString,elasticsearch_engine::memory::Reservation),EsError>{
    let owner=elasticsearch_engine::memory::breakers().reserve_request(256,"snapshot directory entry name")?;
    let name=entry.file_name();if name.as_os_str().as_encoded_bytes().len()>255{return Err(EsError::internal("snapshot directory entry exceeds supported NAME_MAX"));}Ok((name,owner))
}
fn copy_tree(src:&Path,dst:&Path,link:bool)->Result<(),EsError>{
    if !src.exists(){return Ok(())}
    for entry in std::fs::read_dir(src).map_err(snapshot_io)? {
        let entry=entry.map_err(snapshot_io)?;let(name,_name_owner)=entry_name(&entry)?;
        if name.as_os_str().as_encoded_bytes().ends_with(b".tmp"){continue}
        let source=OwnedPath::join(src,&name)?;let destination=OwnedPath::join(dst,&name)?;
        if entry.file_type().map_err(snapshot_io)?.is_dir(){copy_tree(source.as_path(),destination.as_path(),link)?;}
        else {
            let result=if link{link_or_copy(source.as_path(),destination.as_path())}else{std::fs::create_dir_all(dst).map_err(snapshot_io)?;std::fs::copy(source.as_path(),destination.as_path()).map(|_|())};
            match result{Ok(())=>{},Err(e)if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(snapshot_io(e))}
        }
    }Ok(())
}

/// Flush (optionally) and link the committed files of shard copy `copy`
/// (with its index's `index_meta.json`) into `dest`.
fn stage(copy: &elasticsearch_engine::index::IndexShard, dest: &Path, may_flush: bool, graph_lag: i64) -> Result<(SharedText, SharedSequence<(SharedText, u64)>, bool), EsError> {
    // Nothing uncommitted (and the graph checkpoint within the lag): the
    // commit point is already the full image (and an unchanged shard keeps
    // byte-identical files, so the next snapshot is fully incremental).
    let dirty = copy.shard.uncommitted_ops() > 0;
    let flush = may_flush && crate::shards::flush_within_graph_lag(copy, graph_lag).map_err(|e| EsError::new(500, "snapshot_exception", e))?;
    let need_translog = dirty && !may_flush;
    let _ = std::fs::remove_dir_all(dest);
    std::fs::create_dir_all(dest).map_err(|e|EsError::new(500,"snapshot_exception",e.to_string()))?;
    let r: Result<(),EsError> = (|| {
        let src=copy.path();let _files=copy.shard.files_pinned();
        let meta_source=OwnedPath::join(src,std::ffi::OsStr::new("index_meta.json"))?;
        let meta_dest=OwnedPath::join(dest,std::ffi::OsStr::new("index_meta.json"))?;
        link_or_copy(meta_source.as_path(),meta_dest.as_path()).map_err(snapshot_io)?;
        let shard_name=SharedText::try_format(format_args!("shard-{}",copy.id))?;
        let shard=OwnedPath::join(src,std::ffi::OsStr::new(shard_name.as_str()))?;
        let state_path=OwnedPath::join(shard.as_path(),std::ffi::OsStr::new("shard_state.json"))?;
        let state_file=std::fs::File::open(state_path.as_path()).map_err(snapshot_io)?;
        let state_len=usize::try_from(state_file.metadata().map_err(snapshot_io)?.len()).map_err(|_|EsError::new(429,"circuit_breaking_exception","snapshot shard state length overflow"))?;
        let state_raw=SharedSequence::try_read_prefix(state_file,state_len)?;
        let state_text=std::str::from_utf8(&state_raw).map_err(|e|EsError::new(500,"i_o_exception",e.to_string()))?;
        let state=crate::owned_json::OwnedJson::parse(state_text)?;
        let core_gen=state["core_generation"].as_u64().unwrap_or(0);
        let shard_dest=OwnedPath::join(dest,std::ffi::OsStr::new("shard-0"))?;
        std::fs::create_dir_all(shard_dest.as_path()).map_err(snapshot_io)?;
        let state_dest=OwnedPath::join(shard_dest.as_path(),std::ffi::OsStr::new("shard_state.json"))?;
        std::fs::write(state_dest.as_path(),state_raw.as_ref()).map_err(snapshot_io)?;
        let core=SharedText::try_format(format_args!("core-{core_gen}"))?;
        let core_source=OwnedPath::join(shard.as_path(),std::ffi::OsStr::new(core.as_str()))?;
        let core_dest=OwnedPath::join(shard_dest.as_path(),std::ffi::OsStr::new(core.as_str()))?;
        // Only the files the commit references (as file recovery ships):
        // never a merge's or seal's output still being written, files a
        // commit retired but did not delete yet, or crash leftovers.
        let committed=copy.shard.committed_core_files()?;
        std::fs::create_dir_all(core_dest.as_path()).map_err(snapshot_io)?;
        for name in &committed {
            let source=OwnedPath::join(core_source.as_path(),std::ffi::OsStr::new(name))?;
            let destination=OwnedPath::join(core_dest.as_path(),std::ffi::OsStr::new(name))?;
            link_or_copy(source.as_path(),destination.as_path()).map_err(snapshot_io)?;
        }
        if need_translog {
            let source=OwnedPath::join(shard.as_path(),std::ffi::OsStr::new("translog"))?;
            let destination=OwnedPath::join(shard_dest.as_path(),std::ffi::OsStr::new("translog"))?;
            copy_tree(source.as_path(),destination.as_path(),false)?;
        }
        Ok(())
    })();
    if let Err(e)=r {
        let _=std::fs::remove_dir_all(dest);
        if e.status==429{return Err(e)}
        return Err(EsError::new(500,"snapshot_exception",format!("failed to stage shard files: {e}")));
    }
    let files = admitted_files(dest)?;
    Ok((SharedText::try_format(format_args!("{}",dest.display()))?, files, flush))
}

fn read_chunk(root: &Path, r: &ReadReq) -> Result<SharedSequence<u8>, EsError> {
    let uuid=SharedText::try_format(format_args!("{}",SafeComponent(&r.snapshot_uuid)))?;
    if !safe_rel(&r.path) || !safe_rel(&uuid) {
        return Err(EsError::new(500,"i_o_exception",format!("invalid path [{}]",r.path)));
    }
    let dir=owned_staging_path(root,&r.snapshot_uuid,&r.index,r.shard)?;
    let p=OwnedPath::join(&dir.value,std::ffi::OsStr::new(r.path.as_str()))?;
    let mut f = std::fs::File::open(&p.value).map_err(|e|EsError::new(500,"i_o_exception",format!("{}: {e}",p.value.display())))?;
    f.seek(SeekFrom::Start(r.offset)).map_err(|e|EsError::new(500,"i_o_exception",e.to_string()))?;
    SharedSequence::try_read_prefix(f,r.len.min(CHUNK) as usize)
}

fn admitted_files(base:&Path)->Result<SharedSequence<(SharedText,u64)>,EsError>{
    fn walk(base:&Path,dir:&Path,push:&mut dyn FnMut((SharedText,u64))->Result<(),EsError>)->Result<(),EsError>{
        let io=|e:std::io::Error|EsError::new(500,"i_o_exception",e.to_string());
        for entry in std::fs::read_dir(dir).map_err(io)? {
            let entry=entry.map_err(io)?;
            // Linux directory-entry names are bounded by NAME_MAX (255 bytes).
            // Charge the name copy before std::fs materializes it; the full
            // joined path has a separate exact owner retained through recursion.
            let (name,_name_owner)=entry_name(&entry)?;
            let path=OwnedPath::join(dir,&name)?;let kind=entry.file_type().map_err(io)?;
            if kind.is_dir(){walk(base,&path.value,push)?;}
            else if kind.is_file(){
                let rel=path.value.strip_prefix(base).map_err(|e|EsError::internal(e.to_string()))?;
                let name=SharedText::try_format(format_args!("{}",rel.display()))?;
                push((name,entry.metadata().map_err(io)?.len()))?;
            }
        }
        Ok(())
    }
    SharedSequence::try_generate_sorted(|push|walk(base,base,push))
}

#[derive(Debug)]
struct SnapshotNames { files:SharedSequence<(SharedText,u64)>, _owner:elasticsearch_engine::memory::Reservation }
struct SnapshotWorkspace { owner:elasticsearch_engine::memory::Reservation }
fn workspace_error(error:EsError)->elasticsearch_snapshots::SnapshotError{
    elasticsearch_snapshots::SnapshotError::new(if error.status==429{"circuit_breaking_exception"}else{"snapshot_exception"},error.status,error.reason.to_string())
}
impl elasticsearch_snapshots::service::SnapshotWorkspace for SnapshotWorkspace {
 fn grow(&mut self,bytes:usize,label:&'static str)->elasticsearch_snapshots::error::Result<()>{self.owner.grow(bytes as u64,label).map_err(workspace_error)}
}
impl elasticsearch_snapshots::service::SnapshotFileSource for SnapshotNames {
 fn workspace(&self)->elasticsearch_snapshots::error::Result<Box<dyn elasticsearch_snapshots::service::SnapshotWorkspace>>{
  let owner=elasticsearch_engine::memory::breakers().reserve_request(std::mem::size_of::<SnapshotWorkspace>()as u64,"snapshot repository workspace owner").map_err(workspace_error)?.into_result_owner();
  Ok(Box::new(SnapshotWorkspace{owner}))
 }
 fn len(&self)->usize{self.files.len()}
 fn get(&self,i:usize)->Option<&str>{self.files.get(i).map(|(name,_)|name.as_str())}
}
pub fn snapshot_names(files:SharedSequence<(SharedText,u64)>)->Result<elasticsearch_snapshots::service::SnapshotFiles,EsError>{
 let memory=elasticsearch_engine::memory::breakers().reserve_request((std::mem::size_of::<SnapshotNames>()+2*std::mem::size_of::<usize>())as u64,"snapshot name owner")?;
 Ok(elasticsearch_snapshots::service::SnapshotFiles(Arc::new(SnapshotNames{files,_owner:memory.into_result_owner()})))
}

/// This node's part of `_verify`: open the repository with the node's own
/// settings (`path.repo`, S3 clients), read the master's seed blob and
/// write our own. Every failure is a `repository_verification_exception`
/// (500) caused by the underlying error, whatever it is (ES
/// `BlobStoreRepository.verify` wraps them all), so what the master reports
/// never depends on which error a node hit or on either node's version.
pub async fn verify_local(node: &ClusterNode, r: &VerifyReq) -> Result<(), RemoteError> {
    verify_local_steps(node, r).await.map_err(|(message, cause)| verification_failure(message, cause))
}

fn verification_failure(message: String, cause: Option<EsError>) -> RemoteError {
    let failure = RemoteError::new("repository_verification_exception", message, 500);
    match cause {
        Some(e) => failure.with_cause(RemoteError::new(&e.kind, e.reason, e.status)),
        None => failure,
    }
}

/// [`verify_local`]: a failure is its message and its cause.
async fn verify_local_steps(node: &ClusterNode, r: &VerifyReq) -> Result<(), (String, Option<EsError>)> {
    let failed = |e: EsError| ("Failed to verify repository".to_string(), Some(e));
    let mut owner=crate::owned_json::OwnedJson::parse(&r.settings_json).map_err(failed)?;
    fn copy(owner:&mut crate::owned_json::OwnedJson,text:&str)->Result<String,EsError>{
        owner.reserve_extra(text.len(),"repository verification metadata text")?;
        let mut out=String::new();out.try_reserve_exact(text.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;out.push_str(text);Ok(out)
    }
    let name=copy(&mut owner,&r.repository).map_err(failed)?;let kind=copy(&mut owner,&r.repo_type).map_err(failed)?;
    owner.reserve_extra(std::mem::size_of::<crate::owned_json::OwnedJson>(),"repository verification input owner").map_err(failed)?;
    let mut settings=serde_json::Value::Null;
    owner.transform(|value,_|{settings=std::mem::take(value);Ok(())}).map_err(failed)?;
    let meta = RepositoryMetadata::new(name,kind,settings);
    let repo = Repository::open_owned(meta, &repo_context(node),Box::new(owner)).await.map_err(|e|failed(EsError::new(e.status,e.kind,e.reason)))?;
    let inaccessible = || format!("store location is not accessible on the node [{}]", node.local.name);
    if repo.is_readonly() {
        repo.latest_generation().await.map_err(|e|(inaccessible(), Some(EsError::new(e.status,e.kind,e.reason))))?;
        return Ok(());
    }
    let c = repo.root().child(&format!("tests-{}", r.seed));
    let back = c.read_blob("master.dat").await.map_err(|e|(format!("store location is not shared between this node [{}] and the master", node.local.name), Some(EsError::new(e.status,e.kind,e.reason))))?;
    if back.as_ref() != r.seed.as_bytes() {
        return Err((format!("store location is not shared between node [{}] and the master: unexpected seed", node.local.name), None));
    }
    c.write_blob(&format!("data-{}.dat", node.local.id), bytes::Bytes::from_owner(SeedBytes(r.seed.clone())), false)
        .await
        .map_err(|e|(inaccessible(), Some(EsError::new(e.status,e.kind,e.reason))))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node's verify failure is a repository_verification_exception (500)
    /// whatever caused it; the cause stays visible under it.
    #[test]
    fn verify_failures_are_repository_verification_exceptions() {
        for cause in [EsError::new(500, "i_o_exception", "disk gone"), EsError::new(429, "circuit_breaking_exception", "[request] Data too large")] {
            let (kind, reason) = (cause.kind.clone(), cause.reason.clone());
            let e = verification_failure("store location is not accessible on the node [n1]".into(), Some(cause));
            assert_eq!((e.error_type.as_str(), e.status), ("repository_verification_exception", 500));
            let json = e.to_json();
            assert_eq!(json["reason"], "store location is not accessible on the node [n1]");
            assert_eq!(json["caused_by"]["type"], kind.as_str());
            assert_eq!(json["caused_by"]["reason"], reason.as_str());
        }
        let e = verification_failure("unexpected seed".into(), None);
        assert_eq!((e.error_type.as_str(), e.status), ("repository_verification_exception", 500));
        assert!(e.caused_by.is_none());
    }

    #[test]
    fn staging_path_is_per_shard() {
        let a = staging_path(Path::new("/r"), "u", "idx", 0);
        let b = staging_path(Path::new("/r"), "u", "idx", 3);
        assert_ne!(a, b);
        assert!(b.ends_with("3"));
        assert_eq!(a.parent(), b.parent());
    }

    #[test]
    fn read_req_shard_defaults_to_zero() {
        let r: ReadReq = serde_json::from_value(serde_json::json!({
            "snapshot_uuid": "u", "index": "i", "path": "p", "offset": 0, "len": 1
        }))
        .unwrap();
        assert_eq!(r.shard, 0);
    }
}

#[derive(Debug)]
struct SeedBytes(SharedText);
impl AsRef<[u8]> for SeedBytes{fn as_ref(&self)->&[u8]{self.0.as_bytes()}}
