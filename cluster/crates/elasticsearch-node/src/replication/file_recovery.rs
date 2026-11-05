//! File-based peer recovery (ES `RecoverySourceHandler.phase1`,
//! `PeerRecoveryTargetService`, `RecoveryTarget`, `MultiFileWriter`).
//!
//! An operations-based recovery re-indexes every document and vector of
//! the shard through one writer: graphs are rebuilt from scratch (hours at
//! 10M+ vectors). This ships the primary's last commit as files instead,
//! then replays only the operations above it ([`super::recovery`]).
//!
//! **Source** (`shards::start_file_recovery_source`, `shards::file_chunk`):
//! flushes (asking for a graph save first when the commit's graph
//! checkpoint lags far behind it, B9), pins that commit
//! (`IndexShard::snapshot_files`: mapped files, a retention lock on the
//! history above the commit's local checkpoint, checksums computed off the
//! shard locks) and registers a [`RecoverySource::start_from`] whose
//! translog phase starts above that local checkpoint. Chunks are read at
//! their offset from the pinned mappings, each request fenced by the
//! source's allocation id and primary term, and paced by the node's
//! limiter (`replication::throttle`, `indices.recovery.max_bytes_per_sec`:
//! file data only, as ES; the translog phase is not paced).
//!
//! **Target** ([`plan`], [`PartWriter`], [`install`]):
//! 1. the copy's old store was set aside (`shard-<n>.reuse/`) and an empty
//!    placeholder copy holds the shard slot; the staging directory
//!    (`shard-<n>.recovery/`) survives retries;
//! 2. [`plan`] diffs the source's manifest against what is already there
//!    by name, length and CRC-32 (ES `Store.MetadataSnapshot.recoveryDiff`):
//!    files an interrupted attempt completed are kept, partial ones resume
//!    from their last verified chunk, identical files of the old store are
//!    hard-linked (reused in place), the rest is fetched; then the disk must
//!    have room for what is fetched plus the high watermark's free space;
//! 3. each fetched file is written to `<name>.part` chunk by chunk
//!    (positional writes; each chunk's checksum verified; a running CRC-32
//!    and an fsynced progress sidecar every [`SYNC_BYTES`]), then renamed
//!    into place only once the whole-file CRC-32 matches the manifest
//!    (fsync, rename, directory fsync). A mismatch discards the file, which
//!    is fetched again;
//! 4. the staged copy replaces the placeholder in three steps, so the
//!    node's apply lock (which the cluster applier takes) is never held
//!    while it opens: [`release_placeholder`] (apply lock, short) retires
//!    the placeholder and its empty store; the index takes the field ids the
//!    received files were written with ([`staged_core_ids`],
//!    `IndexService::adopt_core_ids`: ids follow the order fields first
//!    appeared, so a shell built later from the final mapping numbers them
//!    otherwise); [`open_recovered`] (no lock)
//!    opens the staged copy as an unpublished prepared shard and its
//!    replica applier (`RecoverySource::FromRecoveredFiles`: graph
//!    checkpoint reconciliation, possibly a long catch-up); [`activate`]
//!    (apply lock, short, after re-checking that the recovery is current)
//!    publishes it in the slot. The staged files stay until it is active: a
//!    failed open is retried (and then reported) without fetching them
//!    again. Phase 2 is the ordinary translog phase.
//!
//! **Graph checkpoints.** A commit references one `.rmvc` checkpoint per
//! mutable vector field, saved on its own schedule, so it can be older than
//! the commit (B9). The source ships exactly the checkpoint the commit
//! references: the pair is what a crash restart of the source itself
//! opens, and the target's open inserts the committed points the
//! checkpoint lacks (bounded batches, `insert_committed_points`). The
//! source bounds that catch-up: when the lag exceeds the graph-lag limit it
//! asks its pre-recovery flush to save the graphs
//! (`Shard::request_graph_save`), which captures them at the flush's
//! commit point outside the shard locks (chunked, as PIT capture is).
//! Capturing a fresh checkpoint apart from a commit was rejected: no
//! commit would reference it, so the target could not open it consistently
//! with the text segments, and making one reference it is that flush.
//!
//! Nothing here holds a shard or core lock while files move: pinning takes
//! the write state and core read lock only to map the files, checksums and
//! chunk reads use the mappings, and every blocking step runs on the
//! recovery pool, never on the cluster applier or a transport worker.

#[path="file_control.rs"]
mod file_control;
pub use file_control::{FileRecoveryStartReq,FileChunkReq};

use super::recovery::FileRecoveryStats;
use super::replica::ReplicaApplier;
use elasticsearch_engine::hooks::files::{file_crc32, note_checksum, IndexFileEntry, IndexManifest};
use elasticsearch_engine::index::{IndexService, IndexShard, PreparedShard, RecoverySource as EngineRecovery, ShardOpenSpec};
use elasticsearch_engine::rustic_core::codec::{crc32, crc32_update};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

type R<T> = std::result::Result<T, String>;

pub const ACTION_FILE_RECOVERY_START: &str = "internal:index/shard/recovery/start_file_recovery";
pub const ACTION_FILE_CHUNK: &str = "internal:index/shard/recovery/file_chunk";

/// Node setting (not ES): `false` keeps the operations-based document
/// snapshot recovery for every copy.
pub const FILE_BASED: &str = "indices.recovery.file_based";
/// Node setting (not ES): ops a commit may hold above its graph checkpoint
/// before a file recovery source saves the graphs first (default 50,000).
pub const GRAPH_LAG_OPS: &str = "indices.recovery.graph_checkpoint_lag_ops";

/// A partial file's data is fsynced and its progress recorded this often.
pub const SYNC_BYTES: u64 = 16 << 20;
/// The file transfer's bookkeeping in the staging directory (removed
/// before installation): every completed file with its length and CRC-32.
pub const PROGRESS_FILE: &str = "recovery-files.json";
/// Marks the empty copy holding the shard slot during a file recovery (so
/// a retry does not mistake it for a store worth setting aside).
pub const PLACEHOLDER_MARKER: &str = "recovery-placeholder";
const PART_SUFFIX: &str = ".part";
const SIDECAR_SUFFIX: &str = ".part.json";

/// Error texts the node's retry logic matches.
pub const CHUNK_CORRUPT: &str = "file chunk checksum mismatch";
pub const FILE_CORRUPT: &str = "recovered file checksum mismatch";
pub const INSUFFICIENT_DISK: &str = "insufficient disk space for recovery";
pub const STAGING_BUSY: &str = "recovery staging directory is in use";

/// Test seams (hidden; benchmarks and the cluster tests use them): the
/// chunk request after the next `n` fails once as a dropped connection
/// would (`-1`: off), and the next `n` received chunks arrive damaged.
#[doc(hidden)]
pub static INTERRUPT_AFTER_CHUNKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);
#[doc(hidden)]
pub static CORRUPT_CHUNKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub const INJECTED_INTERRUPTION: &str = "injected recovery interruption (connection ] disconnected)";

/// Test seams (hidden): the next `n` [`open_recovered`] calls fail as an
/// open of the received files would, and a hook runs once where the
/// received copy opens (off the apply lock; a test makes it slow).
#[doc(hidden)]
pub static FAIL_INSTALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[doc(hidden)]
pub static INSTALL_OPEN_HOOK: LazyLock<Mutex<Option<Box<dyn FnOnce() + Send>>>> = LazyLock::new(Default::default);
pub const INJECTED_INSTALL_FAILURE: &str = "injected failure opening the recovered files";

/// Opens of the received copy a recovery attempts (with a backoff in
/// between) before it fails the copy (ES: the target fails the shard and
/// the master reallocates it). The received files are kept throughout.
pub const INSTALL_ATTEMPTS: u32 = 3;
/// Error text of a failed open of the received copy (never retried as a
/// whole recovery: the copy is reported failed).
pub const INSTALL_FAILED: &str = "cannot open the recovered files";

#[doc(hidden)]
pub fn inject_interruption() -> R<()> {
    use std::sync::atomic::Ordering::SeqCst;
    match INTERRUPT_AFTER_CHUNKS.fetch_update(SeqCst, SeqCst, |n| (n >= 0).then(|| n - 1)) {
        Ok(0) => Err(INJECTED_INTERRUPTION.into()),
        _ => Ok(()),
    }
}

#[doc(hidden)]
pub fn inject_corruption(data: &mut [u8]) {
    use std::sync::atomic::Ordering::SeqCst;
    if !data.is_empty() && CORRUPT_CHUNKS.fetch_update(SeqCst, SeqCst, |n| (n > 0).then(|| n - 1)).is_ok() {
        data[data.len() / 2] ^= 0x5a;
    }
}

/// Chunk bytes on the wire: one byte string (bincode writes length + raw
/// bytes, not one element per byte).
#[path="file_bytes.rs"]
mod file_bytes;
pub use file_bytes::ChunkBytes;
#[path="file_manifest.rs"]
mod file_manifest;
pub use file_manifest::AdmittedManifest;

/// Start a file-based recovery of `target_allocation_id` (ES
/// `StartRecoveryRequest`).


/// What the source pinned: the commit's files and the phase-2 start.
#[path="file_response.rs"]
mod file_response;
pub use file_response::FileRecoveryStartResp;

/// One chunk of one file of the pinned commit.


#[derive(Debug, Serialize, Deserialize)]
pub struct FileChunkResp {
    pub offset: u64,
    pub data: ChunkBytes,
    /// CRC-32 of `data` as read on the source.
    pub crc32: u32,
    /// The source's throttle time for this recovery so far.
    pub source_throttle_nanos: u64,
}

/// The staging directory of shard `shard`'s file recovery.
pub fn staging_dir(svc: &IndexService, shard: u32) -> PathBuf {
    svc.path().join(format!("shard-{shard}.recovery"))
}

/// Paths owned by an active recovery. Allocate the full backing before joining
/// the numeric component; the returned owner can follow a cancelled task.
pub fn admitted_recovery_dir(svc:&IndexService,shard:u32,reuse:bool)->Result<elasticsearch_engine::rustic_core::admission::Allocation<PathBuf>,String>{
    let suffix=if reuse{"reuse"}else{"recovery"};
    let size=svc.path().as_os_str().as_encoded_bytes().len().checked_add(64).ok_or("recovery path overflow")?;
    let memory=elasticsearch_engine::rustic_core::admission::shared().reserve((size as u64).saturating_add(64),"recovery directory path").map_err(|e|e.to_string())?;
    // The numeric component is bounded (u32 decimal plus the fixed suffix).
    let component=format!("shard-{shard}.{suffix}");
    let mut path=PathBuf::new();path.try_reserve_exact(size).map_err(|e|format!("memory admission refused: recovery path: {e}"))?;
    path.push(svc.path());path.push(component);
    Ok(elasticsearch_engine::rustic_core::admission::Allocation::new(path,memory))
}

/// Where a recovering copy's previous store waits to be reused.
pub fn reuse_dir(svc: &IndexService, shard: u32) -> PathBuf {
    svc.path().join(format!("shard-{shard}.reuse"))
}

/// The hard-linked image of the staged files [`open_recovered`] hands to
/// the engine (which moves it into the shard directory, or deletes it when
/// the open fails), so the staged files survive a failed open.
pub fn install_dir(svc: &IndexService, shard: u32) -> PathBuf {
    svc.path().join(format!("shard-{shard}.install"))
}

/// Delete the staging and set-aside directories of shard `shard` (the copy
/// left the node, or its data is stale), unless a recovery is using them.
pub fn remove_recovery_dirs(svc: &IndexService, shard: u32) {
    let staging = staging_dir(svc, shard);
    let Ok(Some(_lock)) = StagingLock::try_acquire(&staging) else { return };
    for dir in [staging.clone(), reuse_dir(svc, shard), install_dir(svc, shard)] {
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => tracing::info!(dir = %dir.display(), "removed leftover recovery files"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(dir = %dir.display(), "cannot remove leftover recovery files: {e}"),
        }
    }
}

/// One recovery at a time uses a staging directory (a superseded task
/// stops at its next chunk; its successor waits for it, retrying).
pub struct StagingLock(PathBuf, elasticsearch_engine::rustic_core::admission::Reservation);

static STAGING: LazyLock<Mutex<BTreeSet<PathBuf>>> = LazyLock::new(Default::default);

impl StagingLock {
    pub fn try_acquire(dir: &Path) -> R<Option<StagingLock>> {
        let mut entries=STAGING.lock();
        if entries.contains(dir){return Ok(None);}
        let bytes=(dir.as_os_str().as_encoded_bytes().len() as u64).checked_mul(2).and_then(|n|n.checked_add(2048)).ok_or("recovery staging lock metadata overflow")?;
        let memory=elasticsearch_engine::rustic_core::admission::shared().reserve(bytes,"recovery staging ownership registry").map_err(|e|e.to_string())?;
        entries.insert(dir.to_path_buf());
        Ok(Some(StagingLock(dir.to_path_buf(),memory)))
    }
}

impl Drop for StagingLock {
    fn drop(&mut self) {
        let mut entries=STAGING.lock();
        entries.remove(&self.0);
        // BTreeSet may retain an empty root after removing its final entry.
        // Destroy that backing while this final lock's reservation is live.
        if entries.is_empty(){drop(std::mem::take(&mut *entries));}
    }
}

fn io(what: &str, path: &Path, e: impl std::fmt::Display) -> String {
    format!("{what} [{}]: {e}", path.display())
}

/// `rel` under `dir`, refusing anything that could leave it.
fn safe_join(dir: &Path, rel: &str) -> R<PathBuf> {
    if rel.is_empty() || rel.starts_with('/') || rel.split('/').any(|s| s == ".." || s == "." || s.is_empty()) {
        return Err(format!("bad file path [{rel}] in a recovery manifest"));
    }
    let mut joined=PathBuf::with_capacity(dir.as_os_str().as_encoded_bytes().len().saturating_add(rel.len()).saturating_add(1));
    joined.push(dir);
    joined.push(rel);
    Ok(joined)
}

fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut s = std::ffi::OsString::with_capacity(path.as_os_str().as_encoded_bytes().len().saturating_add(suffix.len()));
    s.push(path.as_os_str());
    s.push(suffix);
    PathBuf::from(s)
}

fn sync_dir(dir: Option<&Path>) {
    if let Some(d) = dir {
        if let Ok(f) = std::fs::File::open(d) {
            let _ = f.sync_all();
        }
    }
}

/// Write `bytes` to `path` atomically and durably.
fn write_durable(path: &Path, bytes: &[u8]) -> R<()> {
    let tmp = suffixed(path, ".tmp~");
    std::fs::write(&tmp, bytes).map_err(|e| io("cannot write", &tmp, e))?;
    elasticsearch_engine::rustic_core::platform::sync_file(&tmp).map_err(|e| io("cannot fsync", &tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| io("cannot rename", &tmp, e))?;
    sync_dir(path.parent());
    Ok(())
}

/// Stream JSON directly into the durable temporary file. The serializer does
/// not first materialize a second copy of the progress map as encoded bytes.
fn write_durable_json(path:&Path,value:&impl Serialize)->R<()> {
    let bytes=path.as_os_str().as_encoded_bytes().len().checked_add(".tmp~".len()).ok_or("recovery sidecar path overflow")?;
    let _memory=elasticsearch_engine::rustic_core::admission::shared().reserve(bytes as u64,"recovery sidecar temporary path").map_err(|e|e.to_string())?;
    let tmp=suffixed(path,".tmp~");
    let mut file=std::fs::File::create(&tmp).map_err(|e|io("cannot write",&tmp,e))?;
    serde_json::to_writer(&mut file,value).map_err(|e|e.to_string())?;
    file.sync_all().map_err(|e|io("cannot fsync",&tmp,e))?;
    std::fs::rename(&tmp,path).map_err(|e|io("cannot rename",&tmp,e))?;
    sync_dir(path.parent());
    Ok(())
}

/// Every regular file under `dir`, as `/`-separated relative paths.
fn list_files(dir: &Path) -> R<elasticsearch_engine::rustic_core::admission::Allocation<Vec<String>>> {
    use elasticsearch_engine::rustic_core::admission::{self,Allocation,Reservation};
    fn push(values:&mut Vec<String>,value:String,memory:&mut Reservation)->R<()> {
        if values.len()==values.capacity(){
            let old=values.capacity().saturating_mul(std::mem::size_of::<String>());
            let capacity=values.capacity().max(4).checked_mul(2).ok_or("recovery path list overflow")?;
            let requested=capacity.checked_mul(std::mem::size_of::<String>()).ok_or("recovery path list overflow")?;
            memory.grow(requested as u64,"recovery path list replacement").map_err(|e|e.to_string())?;
            values.try_reserve_exact(capacity-values.len()).map_err(|e|e.to_string())?;
            memory.shrink(old as u64);
        }
        values.push(value);Ok(())
    }
    let mut memory=admission::shared().reserve(0,"recovery directory listing").map_err(|e|e.to_string())?;
    let mut out=Vec::new();let mut stack=Vec::new();
    push(&mut stack,String::new(),&mut memory)?;
    while let Some(prefix)=stack.pop(){
        let path_len=dir.as_os_str().as_encoded_bytes().len().checked_add(prefix.len()+1).ok_or("recovery directory path overflow")?;
        let _path=admission::shared().reserve(path_len as u64,"recovery directory traversal path").map_err(|e|e.to_string())?;
        let d=if prefix.is_empty(){dir.to_path_buf()}else{safe_join(dir,&prefix)?};
        let rd=match std::fs::read_dir(&d){Ok(rd)=>rd,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>continue,Err(e)=>return Err(io("cannot list",&d,e))};
        for entry in rd.flatten(){
            // DirEntry's OS-owned name is the input; admit UTF-8 expansion and
            // destination text before copying it into retained traversal data.
            let raw=entry.file_name();
            let name_bound=raw.as_encoded_bytes().len().checked_mul(3).ok_or("recovery filename overflow")?;
            let text_bound=prefix.len().checked_add(name_bound).and_then(|n|n.checked_add(1)).ok_or("recovery relative path overflow")?;
            let _name=admission::shared().reserve(name_bound as u64,"recovery filename UTF-8 scratch").map_err(|e|e.to_string())?;
            memory.grow(text_bound as u64,"recovery directory entry text").map_err(|e|e.to_string())?;
            let name=raw.to_string_lossy();
            let mut rel=String::new();rel.try_reserve_exact(text_bound).map_err(|e|e.to_string())?;
            if !prefix.is_empty(){rel.push_str(&prefix);rel.push('/');}rel.push_str(&name);
            match entry.file_type(){Ok(t) if t.is_dir()=>push(&mut stack,rel,&mut memory)?,Ok(t) if t.is_file()=>push(&mut out,rel,&mut memory)?,_=>{}}
        }
    }
    drop(stack);
    Ok(Allocation::new(out,memory))
}

/// The completed files of a staging directory (`PROGRESS_FILE`).
#[derive(Debug, Default, Serialize)]
pub struct Progress {
    /// Relative path → (length, CRC-32).
    pub done: BTreeMap<String, (u64, u32)>,
    #[serde(skip)]
    memory: Option<elasticsearch_engine::rustic_core::admission::Reservation>,
}

impl Progress {
    pub fn load(staging: &Path) -> R<Progress> {
        use std::io::Read;
        // Unreadable/corrupt bookkeeping costs a re-fetch. Admission refusal
        // must propagate, rather than silently retrying an unbudgeted decode.
        let budget=elasticsearch_engine::rustic_core::admission::shared();
        let path_bytes=staging.as_os_str().as_encoded_bytes().len().checked_add(PROGRESS_FILE.len()+1).ok_or("recovery progress path overflow")?;
        let _path_memory=budget.reserve(path_bytes as u64,"recovery progress path").map_err(|e|e.to_string())?;
        let path=safe_join(staging,PROGRESS_FILE)?;
        let Ok(mut file)=std::fs::File::open(path) else{return Ok(Self::default())};
        let Ok(metadata)=file.metadata() else{return Ok(Self::default())};
        let len=usize::try_from(metadata.len()).map_err(|_|"recovery progress file too large")?;
        // Input bytes plus JSON parser's geometric escaped-string scratch.
        let scratch=(len as u64).checked_mul(3).ok_or("recovery progress scratch overflow")?;
        let _scratch=budget.reserve(scratch,"recovery progress JSON scratch").map_err(|e|e.to_string())?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(len).map_err(|e|e.to_string())?;bytes.resize(len,0);
        if file.read_exact(&mut bytes).is_err(){return Ok(Self::default());}
        let mut extra=[0u8;1];if file.read(&mut extra).map_err(|e|e.to_string())?!=0{return Err("recovery progress changed while reading".into());}
        // Every BTree entry has a JSON colon outside strings. Count without
        // decoding/allocating; malformed input can only over-reserve here.
        let (mut quoted,mut escaped,mut entries)=(false,false,0u64);
        for byte in &bytes {
            if quoted {if escaped{escaped=false;}else if *byte==b'\\'{escaped=true;}else if *byte==b'"'{quoted=false;}}
            else if *byte==b'"'{quoted=true;}else if *byte==b':'{entries=entries.checked_add(1).ok_or("recovery progress entry overflow")?;}
        }
        let retained=entries.checked_mul(2048).and_then(|n|n.checked_add((len as u64).checked_mul(2)?)).ok_or("recovery progress map overflow")?;
        let memory=budget.reserve(retained,"recovery progress decoded map").map_err(|e|e.to_string())?;
        #[derive(Deserialize)] struct Decoded {done:BTreeMap<String,(u64,u32)>}
        match serde_json::from_slice::<Decoded>(&bytes) {
            Ok(decoded)=>Ok(Self{done:decoded.done,memory:Some(memory)}),
            Err(_)=>Ok(Self::default()),
        }
    }

    fn save(&self, staging: &Path) -> R<()> {
        let bytes=staging.as_os_str().as_encoded_bytes().len().saturating_add(PROGRESS_FILE.len()+1);
        let _path=elasticsearch_engine::rustic_core::admission::shared().reserve(bytes as u64,"recovery progress output path").map_err(|e|e.to_string())?;
        write_durable_json(&safe_join(staging,PROGRESS_FILE)?, self)
    }

    /// Record `entry` complete (durably: a retry keeps it).
    pub fn mark(&mut self, staging: &Path, entry: &IndexFileEntry) -> R<()> {
        self.record(entry)?;
        self.save(staging)
    }

    /// Record `entry` complete in memory only (see [`ProgressBatch`]).
    fn record(&mut self, entry: &IndexFileEntry) -> R<()> {
        if !self.done.contains_key(&entry.path) {
            let memory=self.memory.get_or_insert(elasticsearch_engine::rustic_core::admission::shared().reserve(0,"recovery progress map").map_err(|e|e.to_string())?);
            memory.grow((entry.path.len() as u64).saturating_add(2048),"recovery completed file entry").map_err(|e|e.to_string())?;
            self.done.insert(entry.path.clone(),(entry.len,entry.crc32));
        } else if let Some(value)=self.done.get_mut(&entry.path){*value=(entry.len,entry.crc32); }
        Ok(())
    }
}

/// Files received since the progress file was last written: saved every
/// [`PROGRESS_BATCH_FILES`] files, [`PROGRESS_BATCH_BYTES`] bytes or
/// [`PROGRESS_BATCH_INTERVAL`], and when the transfer ends (an error
/// included, so an interrupted recovery resumes after them). Each save rewrites (and fsyncs) the whole progress map, so saving it
/// per file cost O(files^2): a 17,846-file recovery (chaos seed 3, candidate
/// 27) received about 18 files a second and slowed as the map grew. A
/// received file not yet recorded is fetched again by a retry (`plan`
/// removes it), so batching costs at most a batch of re-fetching. A file is
/// recorded only once its directory entry is durable: the directories of the
/// batch are fsynced before the save.
pub struct ProgressBatch {
    progress: Progress,
    dirs: std::collections::BTreeSet<PathBuf>,
    pending: usize,
    pending_bytes: u64,
    saved: std::time::Instant,
    interval: std::time::Duration,
}

/// See [`ProgressBatch`].
pub const PROGRESS_BATCH_FILES: usize = 256;
/// See [`ProgressBatch`].
pub const PROGRESS_BATCH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
/// Unsaved received bytes that make a save due (re-fetching them would cost
/// more than the save): a large file is recorded as soon as it completes.
pub const PROGRESS_BATCH_BYTES: u64 = 1 << 20;

impl ProgressBatch {
    pub fn load(staging: &Path) -> R<ProgressBatch> {
        Ok(ProgressBatch { progress: Progress::load(staging)?, dirs: Default::default(), pending: 0, pending_bytes: 0, saved: std::time::Instant::now(), interval: PROGRESS_BATCH_INTERVAL })
    }

    /// Save at least this often (tests).
    pub fn with_interval(mut self, interval: std::time::Duration) -> ProgressBatch {
        self.interval = interval;
        self
    }

    /// A file [`PartWriter::finish_deferred`] completed (`dir`: its parent).
    pub fn record(&mut self, staging: &Path, entry: &IndexFileEntry, dir: Option<PathBuf>) -> R<()> {
        self.progress.record(entry)?;
        self.dirs.extend(dir);
        self.pending += 1;
        self.pending_bytes = self.pending_bytes.saturating_add(entry.len);
        if self.pending >= PROGRESS_BATCH_FILES || self.pending_bytes >= PROGRESS_BATCH_BYTES || self.saved.elapsed() >= self.interval {
            self.save(staging)?;
        }
        Ok(())
    }

    /// Make every recorded file's directory entry durable, then the progress.
    pub fn save(&mut self, staging: &Path) -> R<()> {
        if self.pending == 0 {
            return Ok(());
        }
        for d in std::mem::take(&mut self.dirs) {
            sync_dir(Some(&d));
        }
        self.progress.save(staging)?;
        self.pending = 0;
        self.pending_bytes = 0;
        self.saved = std::time::Instant::now();
        Ok(())
    }

    /// Recorded files not saved yet.
    pub fn pending(&self) -> usize {
        self.pending
    }
}

/// A partial file's progress (`<name>.part.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Sidecar {
    len: u64,
    crc32: u32,
    /// Bytes written, fsynced and covered by `running`.
    done: u64,
    /// Running CRC-32 state (not finalized) over the first `done` bytes.
    running: u32,
}

/// What phase 1 does with one file of the manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileAction {
    /// Completed by an earlier attempt of this recovery.
    Staged,
    /// Identical to this file of the copy's old store: hard-linked.
    Reuse(PathBuf),
    /// Partly received: continue at `done` (`running`: the CRC-32 state).
    Resume { done: u64, running: u32 },
    Fetch,
}

#[derive(Clone, Debug)]
pub struct PlannedFile {
    pub entry: IndexFileEntry,
    pub action: FileAction,
}

/// Phase 1 of one recovery attempt (see [`plan`]).
#[derive(Debug, Default)]
pub struct Plan {
    pub files: Vec<PlannedFile>,
    // Must follow files: the allocation remains charged through their drop.
    _memory: Option<elasticsearch_engine::rustic_core::admission::Reservation>,
}

impl Plan {
    /// Bytes still to receive from the source.
    pub fn bytes_to_fetch(&self) -> u64 {
        self.files.iter().map(|f| match f.action {
            FileAction::Fetch => f.entry.len,
            FileAction::Resume { done, .. } => f.entry.len.saturating_sub(done),
            FileAction::Staged | FileAction::Reuse(_) => 0,
        }).sum()
    }

    /// The `_recovery` file stats at the start of the transfer: reused
    /// files count as reused (ES), files and bytes an interrupted attempt
    /// already received as recovered.
    pub fn stats(&self) -> R<FileRecoveryStats> {
        let mut s = FileRecoveryStats::default();
        for f in &self.files {
            s.files_total += 1;
            s.bytes_total += f.entry.len;
            let (recovered, reused) = match f.action {
                FileAction::Reuse(_) => (0, true),
                FileAction::Staged => (f.entry.len, false),
                FileAction::Resume { done, .. } => (done, false),
                FileAction::Fetch => (0, false),
            };
            if reused {
                s.files_reused += 1;
                s.bytes_reused += f.entry.len;
            } else {
                s.bytes_recovered += recovered;
                if f.action == FileAction::Staged {
                    s.files_recovered += 1;
                }
            }
            s.details.push(&f.entry.path, f.entry.len, recovered, reused)?;
        }
        Ok(s)
    }
}

/// Old-store files by name, for reuse: every file under a `core-<g>/`
/// directory of `reuse` (the commit's generation may differ).
fn reuse_candidates(reuse: &Path) -> R<elasticsearch_engine::rustic_core::admission::Allocation<BTreeMap<String, Vec<PathBuf>>>> {
    use elasticsearch_engine::rustic_core::admission::{self,Allocation};
    let mut memory=admission::shared().reserve(0,"recovery reuse candidates").map_err(|e|e.to_string())?;
    let mut out:BTreeMap<String,Vec<PathBuf>>=BTreeMap::new();
    let listed=list_files(reuse)?;
    for rel in listed.iter(){
        let mut parts=rel.split('/');
        let (Some(dir),Some(name),None)=(parts.next(),parts.next(),parts.next())else{continue};
        if !dir.starts_with("core-"){continue;}
        if !out.contains_key(name){
            memory.grow((name.len() as u64).saturating_add(2048),"recovery reuse candidate key").map_err(|e|e.to_string())?;
            out.insert(name.to_owned(),Vec::new());
        }
        let paths=out.get_mut(name).expect("inserted above");
        if paths.len()==paths.capacity(){
            let old=paths.capacity().saturating_mul(std::mem::size_of::<PathBuf>());
            let capacity=paths.capacity().max(4).checked_mul(2).ok_or("recovery reuse paths overflow")?;
            let next=capacity.checked_mul(std::mem::size_of::<PathBuf>()).ok_or("recovery reuse paths overflow")?;
            memory.grow(next as u64,"recovery reuse path slots").map_err(|e|e.to_string())?;
            paths.try_reserve_exact(capacity-paths.len()).map_err(|e|e.to_string())?;
            memory.shrink(old as u64);
        }
        let path_bytes=reuse.as_os_str().as_encoded_bytes().len().checked_add(rel.len()+1).ok_or("recovery reuse path overflow")?;
        memory.grow(path_bytes as u64,"recovery reuse path").map_err(|e|e.to_string())?;
        paths.push(safe_join(reuse,rel)?);
    }
    Ok(Allocation::new(out,memory))
}

/// The core file name of a manifest path (`core-<g>/<name>`), if it is one.
fn core_file_name(path: &str) -> Option<&str> {
    let (dir, name) = path.split_once('/')?;
    (dir.starts_with("core-") && !name.contains('/')).then_some(name)
}

fn read_sidecar(path: &Path, entry: &IndexFileEntry) -> Option<(u64, u32)> {
    // Four numeric fields fit well below 256 bytes. Read a bounded stack
    // buffer instead of allocating according to a corrupt sidecar's size.
    use std::io::Read;
    let mut file=std::fs::File::open(suffixed(path,SIDECAR_SUFFIX)).ok()?;
    let mut bytes=[0u8;256];let mut used=0;
    while used<bytes.len(){let n=file.read(&mut bytes[used..]).ok()?;if n==0{break;}used+=n;}
    let mut extra=[0u8;1];if file.read(&mut extra).ok()?!=0{return None;}
    let side: Sidecar = serde_json::from_slice(&bytes[..used]).ok()?;
    let part_len = std::fs::metadata(suffixed(path, PART_SUFFIX)).ok()?.len();
    (side.len == entry.len && side.crc32 == entry.crc32 && side.done <= entry.len && part_len >= side.done).then_some((side.done, side.running))
}

/// Phase 1 plan of one attempt (see the module docs, step 2): sweeps
/// `staging` of everything this manifest does not use, then decides each
/// file. Reads (and checksums, cached) the files an earlier attempt
/// completed and old-store files with the same name and length as a
/// manifest file; a resumed prefix is read back by [`PartWriter::open`].
pub fn plan(staging: &Path, reuse: &Path, manifest: &IndexManifest) -> R<Plan> {
    std::fs::create_dir_all(staging).map_err(|e| io("cannot create", staging, e))?;
    let mut progress = Progress::load(staging)?;
    let wanted_bytes=(manifest.files.len() as u64).checked_mul(2048).ok_or("recovery wanted map overflow")?;
    let _wanted_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(wanted_bytes,"recovery wanted file map").map_err(|e|e.to_string())?;
    let wanted: BTreeMap<&str, &IndexFileEntry> = manifest.files.iter().map(|f| (f.path.as_str(), f)).collect();
    progress.done.retain(|p, (len, crc)| wanted.get(p.as_str()).is_some_and(|e| e.len == *len && e.crc32 == *crc));
    let listed=list_files(staging)?;
    for rel in listed.iter() {
        if rel == PROGRESS_FILE {
            continue;
        }
        let keep = match rel.strip_suffix(SIDECAR_SUFFIX).or_else(|| rel.strip_suffix(PART_SUFFIX)) {
            Some(base) => wanted.contains_key(base) && !progress.done.contains_key(base),
            None => progress.done.contains_key(rel.as_str()),
        };
        if !keep {
            let p = staging.join(&rel);
            std::fs::remove_file(&p).map_err(|e| io("cannot remove", &p, e))?;
        }
    }
    progress.save(staging)?;
    let old = reuse_candidates(reuse)?;
    let bytes=(manifest.files.len() as u64).checked_mul(std::mem::size_of::<PlannedFile>() as u64)
        .and_then(|bytes|manifest.files.iter().try_fold(bytes,|total,file|total.checked_add(file.path.len() as u64)))
        .ok_or("recovery file plan size overflow")?;
    let mut memory=elasticsearch_engine::rustic_core::admission::shared().reserve(bytes,"recovery file plan").map_err(|e|e.to_string())?;
    let mut files = Vec::new();
    files.try_reserve_exact(manifest.files.len()).map_err(|e|e.to_string())?;
    for e in &manifest.files {
        let path = safe_join(staging, &e.path)?;
        // Read back (cached by file identity in this process: a file this
        // process received and renamed into place is not read again).
        let staged = progress.done.contains_key(&e.path) && file_crc32(&path).is_ok_and(|(len, crc)| len == e.len && crc == e.crc32);
        let action = if staged {
            FileAction::Staged
        } else if let Some((done, running)) = read_sidecar(&path, e) {
            FileAction::Resume { done, running }
        } else {
            let same = core_file_name(&e.path).and_then(|name| old.get(name)).and_then(|candidates| candidates.iter().find(|c| {
                std::fs::metadata(c).is_ok_and(|m| m.len() == e.len) && file_crc32(c).is_ok_and(|(len, crc)| len == e.len && crc == e.crc32)
            }));
            match same {
                Some(src) => {
                    memory.grow(src.as_os_str().as_encoded_bytes().len() as u64,"recovery reuse path").map_err(|e|e.to_string())?;
                    FileAction::Reuse(src.clone())
                },
                None => FileAction::Fetch,
            }
        };
        files.push(PlannedFile { entry: e.clone(), action });
    }
    Ok(Plan { files, _memory:Some(memory) })
}

/// Bytes at the end of a file compared with the source's before an old
/// store's file (or one an earlier attempt staged) stands in for it. The
/// manifest's whole-file CRC-32 identifies nothing for a file that ends
/// with a CRC-32 of its own preceding bytes (plus a fixed magic): every
/// such file has the same whole-file CRC (the CRC residue; 0x0ecce128 for
/// every `.rvec` file). Vector payload segments (`vec_*.rvec`: content,
/// CRC-32 of it, `RVEC`) are written that way, and their names come from
/// a per-copy counter, so a stale copy's `vec_00000260.rvec` of the same
/// length as the source's (one or two rows: tiny, equal-sized files) passed
/// as identical. The recovered copy then held the old copy's vectors under
/// the source's points: documents paired with other documents' vectors,
/// rows a graph checkpoint references missing, reconciliation failing on
/// out-of-order keys. The tail holds such a file's content CRC (and, for
/// other formats, their footer), so equal tails make the match as strong
/// as the CRC-32 of the content.
pub const TAIL_CHECK_BYTES: u64 = 64;

/// The tail [`TAIL_CHECK_BYTES`] of a file of `len` bytes: `(offset, len)`.
pub fn tail_range(len: u64) -> (u64, u64) {
    let n = len.min(TAIL_CHECK_BYTES);
    (len - n, n)
}

/// The local file a planned file would be taken from without fetching it
/// (the old store's file it is reused from, or the file an earlier attempt
/// staged), if any: these are compared with the source's tail first.
pub fn local_candidate(staging: &Path, f: &PlannedFile) -> R<Option<PathBuf>> {
    match &f.action {
        FileAction::Reuse(src) => Ok(Some(src.clone())),
        FileAction::Staged => safe_join(staging, &f.entry.path).map(Some),
        FileAction::Resume { .. } | FileAction::Fetch => Ok(None),
    }
}

/// The tail ([`tail_range`]) of the local file at `path`, which must have
/// `len` bytes (anything else does not match).
pub fn local_tail(path: &Path, len: u64) -> R<Option<Vec<u8>>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = match std::fs::File::open(path) { Ok(file) => file, Err(_) => return Ok(None) };
    if file.metadata().map_err(|e| io("cannot stat", path, e))?.len() != len {
        return Ok(None);
    }
    let (offset, n) = tail_range(len);
    file.seek(SeekFrom::Start(offset)).map_err(|e| io("cannot seek", path, e))?;
    let mut tail = vec![0u8; n as usize];
    file.read_exact(&mut tail).map_err(|e| io("cannot read", path, e))?;
    Ok(Some(tail))
}

/// A reused or staged file whose tail differs from the source's: fetched
/// after all. Returns the bytes it adds to the fetch.
pub fn demote_to_fetch(f: &mut PlannedFile) -> u64 {
    f.action = FileAction::Fetch;
    f.entry.len
}

/// Refuse a recovery whose files do not fit (ES keeps shards off a node
/// above the high disk watermark): `needed` more bytes on the filesystem of
/// `dir` while keeping `reserve` free. Unknown disk statistics pass.
pub fn check_disk(dir: &Path, needed: u64, reserve: u64) -> R<()> {
    let du = crate::stats::disk_usage(dir);
    if du.total_bytes == 0 || du.available_bytes >= needed.saturating_add(reserve) {
        return Ok(());
    }
    Err(format!("{INSUFFICIENT_DISK}: the shard's files need [{needed}] more bytes, [{}] has [{}] available and must keep [{reserve}] free (cluster.routing.allocation.disk.watermark.high)",
        dir.display(), du.available_bytes))
}

/// Bytes the file recoveries in flight on this node still have to write,
/// by filesystem (device): [`reserve_disk`] counts them as used, since the
/// disk statistics do not show them yet.
static RESERVATIONS: LazyLock<Mutex<BTreeMap<usize,(u64, Arc<std::sync::atomic::AtomicU64>)>>> = LazyLock::new(Default::default);

/// One recovery's share of [`RESERVATIONS`]: what it still has to write.
/// Released when dropped.
pub struct DiskReservation(Arc<std::sync::atomic::AtomicU64>,elasticsearch_engine::rustic_core::admission::Reservation);

impl DiskReservation {
    /// `n` more bytes were written (the disk statistics count them now).
    pub fn written(&self, n: u64) {
        use std::sync::atomic::Ordering::SeqCst;
        let _ = self.0.fetch_update(SeqCst, SeqCst, |left| Some(left.saturating_sub(n)));
    }

    /// Bytes still reserved.
    pub fn left(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Drop for DiskReservation {
    fn drop(&mut self) {
        let mut all=RESERVATIONS.lock();
        all.remove(&(Arc::as_ptr(&self.0) as usize));
        if all.is_empty(){drop(std::mem::take(&mut *all));}
    }
}

/// [`check_disk`] for a recovery of `needed` bytes into `dir`, counting
/// what the other recoveries in flight on the same filesystem still have
/// to write (two 40 GB recoveries must not both pass with 60 GB free); the
/// bytes are reserved until the returned guard is dropped. Checks and
/// reservations are serialized.
pub fn reserve_disk(dir: &Path, needed: u64, reserve: u64) -> R<DiskReservation> {
    use std::sync::atomic::Ordering::SeqCst;
    let dev = device_of(dir).map_err(|e| io("cannot stat", dir, e))?;
    let mut all = RESERVATIONS.lock();
    let others: u64 = all.values().filter(|(d, _)| *d == dev).map(|(_, left)| left.load(SeqCst)).fold(0,u64::saturating_add);
    check_disk(dir, needed, reserve.saturating_add(others))
        .map_err(|e| if others > 0 { format!("{e}, plus [{others}] bytes other recoveries in flight still write") } else { e })?;
    let memory=elasticsearch_engine::rustic_core::admission::shared().reserve(2048+std::mem::size_of::<std::sync::atomic::AtomicU64>() as u64+2*std::mem::size_of::<usize>() as u64,"recovery disk reservation registry").map_err(|e|e.to_string())?;
    let left = Arc::new(std::sync::atomic::AtomicU64::new(needed));
    all.insert(Arc::as_ptr(&left) as usize,(dev,left.clone()));
    Ok(DiskReservation(left,memory))
}

/// The filesystem `path` is on, to group reservations by. Without device
/// numbers (Windows) every path counts as one filesystem: reservations
/// then over-count, never under-count.
#[cfg(unix)]
fn device_of(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(std::fs::metadata(path)?.dev())
}

#[cfg(not(unix))]
fn device_of(path: &Path) -> std::io::Result<u64> {
    std::fs::metadata(path).map(|_| 0)
}

/// `pread` of exactly `buf.len()` bytes at `at`.
#[cfg(unix)]
fn read_exact_at(f: &std::fs::File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, at)
}

/// Positional read of exactly `buf.len()` bytes at `at` (`seek_read` also
/// moves the cursor; nothing here reads or writes through it).
#[cfg(windows)]
fn read_exact_at(f: &std::fs::File, mut buf: &mut [u8], mut at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, at)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                at += n as u64;
            }
        }
    }
    Ok(())
}

/// `pwrite` of all of `data` at `at`.
#[cfg(unix)]
fn write_all_at(f: &std::fs::File, data: &[u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(data, at)
}

/// Positional write of all of `data` at `at` (see [`read_exact_at`]).
#[cfg(windows)]
fn write_all_at(f: &std::fs::File, mut data: &[u8], mut at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        match f.seek_write(data, at)? {
            0 => return Err(std::io::ErrorKind::WriteZero.into()),
            n => {
                data = &data[n..];
                at += n as u64;
            }
        }
    }
    Ok(())
}

/// Install a reused old-store file into the staging directory (a hard
/// link: same filesystem, no copy; a copy otherwise) and record it.
pub fn link_reused(staging: &Path, entry: &IndexFileEntry, src: &Path, progress: &mut Progress) -> R<()> {
    let path = safe_join(staging, &entry.path)?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| io("cannot create", d, e))?;
    }
    let _ = std::fs::remove_file(&path);
    if std::fs::hard_link(src, &path).is_err() {
        let tmp = suffixed(&path, PART_SUFFIX);
        std::fs::copy(src, &tmp).map_err(|e| io("cannot copy", src, e))?;
        elasticsearch_engine::rustic_core::platform::sync_file(&tmp).map_err(|e| io("cannot fsync", &tmp, e))?;
        let (len, crc) = file_crc32(&tmp).map_err(|e| io("cannot read", &tmp, e.reason))?;
        if len != entry.len || crc != entry.crc32 {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("{FILE_CORRUPT}: [{}] changed while it was reused", entry.path));
        }
        std::fs::rename(&tmp, &path).map_err(|e| io("cannot rename", &tmp, e))?;
    }
    sync_dir(path.parent());
    progress.mark(staging, entry)
}

/// One file being received (see the module docs, step 3).
pub struct PartWriter {
    path: PathBuf,
    part: PathBuf,
    sidecar: PathBuf,
    file: std::fs::File,
    entry: IndexFileEntry,
    done: u64,
    running: u32,
    unsynced: u64,
    _memory: elasticsearch_engine::rustic_core::admission::Reservation,
}

impl PartWriter {
    /// Start (or with `resume`, continue) receiving `entry` into `staging`.
    /// A resumed prefix is read back from disk and must match the CRC-32
    /// state its sidecar recorded (it was written by an earlier attempt,
    /// possibly before a crash): otherwise everything received for the
    /// file is discarded and this fails with [`FILE_CORRUPT`].
    pub fn open(staging: &Path, entry: &IndexFileEntry, resume: Option<(u64, u32)>) -> R<PartWriter> {
        // Three independently, exactly preallocated owned paths and one
        // entry path. safe_join/suffixed never clone then grow a path buffer.
        let path_bytes=(staging.as_os_str().as_encoded_bytes().len() as u64)
            .checked_add(entry.path.len() as u64).and_then(|n|n.checked_add(1))
            .and_then(|n|n.checked_mul(3))
            .and_then(|n|n.checked_add((PART_SUFFIX.len()+SIDECAR_SUFFIX.len()) as u64))
            .and_then(|n|n.checked_add(entry.path.len() as u64)).ok_or("recovery writer metadata overflow")?;
        let memory=elasticsearch_engine::rustic_core::admission::shared().reserve(path_bytes,"recovery file writer paths").map_err(|e|e.to_string())?;
        let path = safe_join(staging, &entry.path)?;
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).map_err(|e| io("cannot create", d, e))?;
        }
        let (part, sidecar) = (suffixed(&path, PART_SUFFIX), suffixed(&path, SIDECAR_SUFFIX));
        let file = std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&part).map_err(|e| io("cannot open", &part, e))?;
        let (done, running) = resume.unwrap_or((0, 0xFFFF_FFFF));
        // Anything past the last verified chunk was never checkpointed.
        file.set_len(done).map_err(|e| io("cannot truncate", &part, e))?;
        let mut w = PartWriter { path, part, sidecar, file, entry: entry.clone(), done, running, unsynced: 0, _memory:memory };
        if resume.is_none() {
            w.checkpoint()?;
        } else if w.prefix_crc_state()? != running {
            let e = format!("{FILE_CORRUPT}: the first [{done}] bytes of [{}] received earlier changed on disk", entry.path);
            w.discard();
            return Err(e);
        }
        Ok(w)
    }

    /// The running CRC-32 state of the first `done` bytes as they are on
    /// disk (read in 1 MiB steps).
    fn prefix_crc_state(&self) -> R<u32> {
        let _memory=elasticsearch_engine::rustic_core::admission::shared().reserve(1<<20,"recovery resumed-prefix checksum buffer").map_err(|e|e.to_string())?;
        let mut buf=Vec::new();
        buf.try_reserve_exact(1<<20).map_err(|e|e.to_string())?;
        buf.resize(1<<20,0u8);
        let (mut at, mut crc) = (0u64, 0xFFFF_FFFFu32);
        while at < self.done {
            let n = (self.done - at).min(buf.len() as u64) as usize;
            read_exact_at(&self.file, &mut buf[..n], at).map_err(|e| io("cannot read", &self.part, e))?;
            crc = crc32_update(crc, &buf[..n]);
            at += n as u64;
        }
        Ok(crc)
    }

    pub fn entry(&self) -> &IndexFileEntry {
        &self.entry
    }

    /// Bytes received and verified so far.
    pub fn done(&self) -> u64 {
        self.done
    }

    /// Append the chunk at `offset` (the next one: chunks arrive in order),
    /// after checking it against the CRC-32 the source computed.
    pub fn write_chunk(&mut self, offset: u64, data: &[u8], chunk_crc: u32) -> R<()> {
        if offset != self.done || offset + data.len() as u64 > self.entry.len {
            return Err(format!("chunk [{offset}+{}] of [{}] does not continue at [{}] of [{}] bytes", data.len(), self.entry.path, self.done, self.entry.len));
        }
        if crc32(data) != chunk_crc {
            return Err(format!("{CHUNK_CORRUPT}: [{}] at offset [{offset}]", self.entry.path));
        }
        write_all_at(&self.file, data, offset).map_err(|e| io("cannot write", &self.part, e))?;
        self.running = crc32_update(self.running, data);
        self.done += data.len() as u64;
        self.unsynced += data.len() as u64;
        if self.unsynced >= SYNC_BYTES {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// Make what was received durable and record it (a retry resumes here).
    fn checkpoint(&mut self) -> R<()> {
        self.file.sync_data().map_err(|e| io("cannot fsync", &self.part, e))?;
        let side = Sidecar { len: self.entry.len, crc32: self.entry.crc32, done: self.done, running: self.running };
        write_durable_json(&self.sidecar, &side)?;
        self.unsynced = 0;
        Ok(())
    }

    /// Persist progress without finishing (the transfer is interrupted).
    pub fn suspend(mut self) -> R<()> {
        self.checkpoint()
    }

    /// The file is complete: verify the whole-file CRC-32, then fsync,
    /// rename into place, fsync the directory and record it. A mismatch
    /// discards everything received for it.
    pub fn finish(self, staging: &Path, progress: &mut Progress) -> R<()> {
        let (entry, dir) = self.finish_deferred()?;
        sync_dir(dir.as_deref());
        progress.mark(staging, &entry)
    }

    /// [`PartWriter::finish`] up to the rename: the file's data is durable
    /// and in place; its directory's fsync and the progress record are the
    /// caller's ([`ProgressBatch::record`]). Returns the entry and the
    /// file's directory.
    pub fn finish_deferred(mut self) -> R<(IndexFileEntry, Option<PathBuf>)> {
        if self.done != self.entry.len {
            return Err(format!("[{}] ended at [{}] of [{}] bytes", self.entry.path, self.done, self.entry.len));
        }
        let crc = !self.running;
        if crc != self.entry.crc32 {
            let e = format!("{FILE_CORRUPT}: [{}] has crc32 [{crc:08x}], the source's is [{:08x}]", self.entry.path, self.entry.crc32);
            self.discard();
            return Err(e);
        }
        self.file.sync_all().map_err(|e| io("cannot fsync", &self.part, e))?;
        std::fs::rename(&self.part, &self.path).map_err(|e| io("cannot rename", &self.part, e))?;
        let _ = std::fs::remove_file(&self.sidecar);
        note_checksum(&self.path, crc);
        let dir = self.path.parent().map(Path::to_path_buf);
        Ok((self.entry.clone(), dir))
    }

    /// Drop everything received for this file.
    pub fn discard(self) {
        let _ = std::fs::remove_file(&self.part);
        let _ = std::fs::remove_file(&self.sidecar);
    }
}

/// Install step 1 (the caller holds the node's apply lock and checked that
/// the recovery is current): retire the placeholder copy and its empty
/// store, freeing the slot and the shard directory for the received copy.
/// Again after a failed open: the placeholder is gone, and what the failed
/// open left in the shard directory (never the staged files) is removed.
pub fn release_placeholder(svc: &IndexService, shard: u32, placeholder: &Arc<IndexShard>) -> R<()> {
    if svc.shard(shard).is_some_and(|open| !Arc::ptr_eq(&open, placeholder)) {
        return Err("recovery target copy was replaced".into());
    }
    svc.close_shard_if_same(placeholder).map_err(|e| format!("cannot close the placeholder copy: {}", e.reason))?;
    svc.remove_shard(shard, true).map_err(|e| format!("cannot remove the placeholder store: {}", e.reason))
}

/// Inspect the actual pinned/staged files, not the current writer setting:
/// switching the writer back to v11 does not rewrite existing v12 files.
/// DiskBBQ IVF files and code log chunks are file kinds of v13 themselves.
/// Reads only fixed headers and the checkpoint's rows header (never its graph).
pub(crate) fn validate_file_formats<B:std::ops::Deref>(
    manifest: &IndexManifest, reader_max: u32,
    mut read: impl FnMut(&str, u64, u64) -> R<B>,
) -> R<()> where B::Target:AsRef<[u8]> {
    for file in &manifest.files {
        let name = file.path.rsplit(['/', '\\']).next().unwrap_or(&file.path);
        let required = if file.path.ends_with(".rstc") {
            let header = read(&file.path, 0, 8)?;
            let header=(*header).as_ref();
            if header.len() != 8 || &header[..4] != b"RSTC" { return Err(format!("invalid segment header [{}]", file.path)); }
            let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
            if version < elasticsearch_engine::rustic_core::codec::MIN_READ_VERSION {
                return Err(format!("unsupported segment version {version} [{}]", file.path));
            }
            version
        } else if file.path.ends_with(".rmvc") {
            let header = read(&file.path, 0, 16)?;
            let header=(*header).as_ref();
            let format2 = header.len() == 16 && &header[..8] == b"RMVC\x02\0\0\0";
            if header.len() != 16 || (&header[..8] != b"RMVC\x01\0\0\0" && !format2) { return Err(format!("unsupported checkpoint header [{}]", file.path)); }
            let rows = u64::from_le_bytes(header[8..16].try_into().unwrap()).checked_add(16)
                .filter(|offset| *offset <= file.len.saturating_sub(8)).ok_or_else(|| format!("invalid checkpoint graph length [{}]", file.path))?;
            let header = read(&file.path, rows, 8)?;
            let header=(*header).as_ref();
            if header.len() != 8 || &header[..4] != b"RMVR" { return Err(format!("invalid checkpoint rows [{}]", file.path)); }
            match u32::from_le_bytes(header[4..8].try_into().unwrap()) {
                // Format 1 files carry v1-v3 rows; format 2 (graph + code
                // log references, segment format v13) only v4 rows.
                1 | 2 if !format2 => 11, 3 if !format2 => 12,
                4 if format2 => elasticsearch_engine::rustic_core::codec::RMVC2_SINCE,
                version => return Err(format!("unsupported checkpoint rows version {version} [{}]", file.path)),
            }
        } else if elasticsearch_engine::rustic_core::vector::ivf_field::is_ivf_file(name)
            || elasticsearch_engine::rustic_core::vector::store::is_code_log_file(name) {
            // Unknown to every build reading less (no header to inspect: the
            // file kind itself is new), see `codec::VECTOR_FILES_SINCE`.
            elasticsearch_engine::rustic_core::codec::VECTOR_FILES_SINCE
        } else { continue };
        if required > reader_max {
            return Err(elasticsearch_cluster::compatibility::segment_format_refusal(required,
                &format!("file [{}] requires segment format v{required}, target reads through v{reader_max}", file.path)));
        }
    }
    Ok(())
}

pub(crate) fn validate_staged_formats(staging: &Path, manifest: &IndexManifest) -> R<()> {
    validate_staged_formats_upto(staging, manifest, elasticsearch_engine::rustic_core::codec::VERSION)
}

/// [`validate_staged_formats`] against `reader_max` (a snapshot restore:
/// the cluster's segment format floor, which every member reads).
pub(crate) fn validate_staged_formats_upto(staging: &Path, manifest: &IndexManifest, reader_max: u32) -> R<()> {
    use std::io::{Read, Seek, SeekFrom};
    validate_file_formats(manifest, reader_max, |path, offset, len| {
        let path = safe_join(staging, path)?;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        let mut bytes = vec![0; len as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    })
}

/// The field ids a staged copy's files were written with: the schema of its
/// core commit (`core-<g>/meta.json`, `g` from the `shard_state.json` under
/// `shard_root`), user fields only, by core field key. The receiving index
/// takes them (`IndexService::adopt_core_ids`) before it opens the files: a
/// shell whose mapping arrived in other steps (a node that joined later, a
/// snapshot restored into a new index) numbers the same fields otherwise.
pub(crate) fn staged_core_ids(shard_root: &Path) -> R<BTreeMap<String, u16>> {
    let read = |p: PathBuf| -> R<serde_json::Value> {
        let bytes = std::fs::read(&p).map_err(|e| io("cannot read", &p, e))?;
        serde_json::from_slice(&bytes).map_err(|e| io("cannot parse", &p, e))
    };
    let state = read(shard_root.join("shard_state.json"))?;
    let generation = state["core_generation"].as_u64().unwrap_or(0);
    let commit = read(shard_root.join(format!("core-{generation}")).join("meta.json"))?;
    let mut ids = BTreeMap::new();
    for f in commit["schema"].as_array().ok_or("the core commit has no schema")? {
        let (Some(id), Some(name)) = (f["id"].as_u64().and_then(|id| u16::try_from(id).ok()), f["name"].as_str()) else {
            return Err("invalid field in the core commit's schema".into());
        };
        if id >= elasticsearch_engine::mapping::FIRST_USER_FIELD {
            ids.insert(name.to_string(), id);
        }
    }
    Ok(ids)
}

/// Install step 2 (no lock held: the open reconciles each graph checkpoint
/// with the commit, which can take long): open the staged copy as an
/// unpublished prepared shard (`RecoverySource::FromRecoveredFiles`) and
/// its replica applier. The engine gets a hard-linked image of the staged
/// files ([`install_dir`]): the staged files and their progress stay, so a
/// failed open is retried without fetching anything again. Dropping the
/// result closes the copy.
pub fn open_recovered(svc: &Arc<IndexService>, shard: u32, allocation_id: &str, staging: &Path, manifest: &IndexManifest) -> R<(PreparedShard, ReplicaApplier)> {
    let hook = INSTALL_OPEN_HOOK.lock().take();
    if let Some(hook) = hook {
        hook();
    }
    let failed = |e: String| format!("{INSTALL_FAILED}: {e}");
    if FAIL_INSTALLS.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() {
        return Err(failed(INJECTED_INSTALL_FAILURE.into()));
    }
    let image = install_dir(svc, shard);
    link_image(staging, &image, manifest).map_err(failed)?;
    let recovery = EngineRecovery::FromRecoveredFiles { staged: image.clone(), manifest: manifest.clone() };
    let started = std::time::Instant::now();
    let prepared = svc.prepare_shard(shard, ShardOpenSpec { allocation_id: allocation_id.to_string(), primary: true, primary_term: 1, recovery });
    // Moved into the shard directory, or (failed) left over.
    let _ = std::fs::remove_dir_all(&image);
    let prepared = prepared.map_err(|e| failed(format!("[{}] {}", e.kind, e.reason)))?;
    let applier = ReplicaApplier::open_prepared(svc.clone(), &prepared).map_err(|e| failed(format!("replica applier: {e}")))?;
    tracing::info!(index = %svc.name, shard, millis = started.elapsed().as_millis() as u64, "opened the recovered copy");
    Ok((prepared, applier))
}

/// Hard-link every manifest file of `staging` under `image` (a fresh
/// directory; a copy where a link fails). The engine's writes to a shard
/// are tmp + rename, so a link shared with the staged file is never
/// written through.
fn link_image(staging: &Path, image: &Path, manifest: &IndexManifest) -> R<()> {
    match std::fs::remove_dir_all(image) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io("cannot remove", image, e)),
    }
    for f in &manifest.files {
        let (from, to) = (safe_join(staging, &f.path)?, safe_join(image, &f.path)?);
        if let Some(d) = to.parent() {
            std::fs::create_dir_all(d).map_err(|e| io("cannot create", d, e))?;
        }
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to).map_err(|e| io("cannot copy", &from, e))?;
        }
    }
    sync_dir(image.parent());
    Ok(())
}

/// Install step 3 (the caller holds the node's apply lock and checked
/// again that the recovery is current): publish the prepared copy in the
/// slot (refused when anything closed or reopened the slot since step 1).
pub fn activate(svc: &IndexService, prepared: &mut PreparedShard) -> R<Arc<IndexShard>> {
    svc.try_install_prepared_shard(prepared).map_err(|e| format!("cannot publish the recovered copy: [{}] {}", e.kind, e.reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, data: &[u8]) -> IndexFileEntry {
        IndexFileEntry { path: path.into(), len: data.len() as u64, crc32: crc32(data) }
    }

    fn manifest(files: Vec<IndexFileEntry>) -> IndexManifest {
        IndexManifest { index: "i".into(), uuid: "u".into(), files, committed_seq_no: 0, local_checkpoint: 0, max_seq_no: 0, primary_term: 1, includes_translog: false }
    }

    fn data(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u32).wrapping_mul(2654435761).rotate_left(seed as u32) as u8).collect()
    }

    fn receive(staging: &Path, e: &IndexFileEntry, bytes: &[u8], resume: Option<(u64, u32)>, chunk: usize, stop_after: Option<usize>) -> R<bool> {
        let mut w = PartWriter::open(staging, e, resume)?;
        let mut n = 0;
        while (w.done() as usize) < bytes.len() {
            if stop_after.is_some_and(|s| n == s) {
                w.suspend()?;
                return Ok(false);
            }
            let off = w.done() as usize;
            let c = &bytes[off..(off + chunk).min(bytes.len())];
            w.write_chunk(off as u64, c, crc32(c))?;
            n += 1;
        }
        let mut progress = Progress::load(staging)?;
        w.finish(staging, &mut progress)?;
        Ok(true)
    }

    /// chaos seed 3: a 17,846-file recovery rewrote the whole
    /// progress map per file. Files are now recorded in batches; a file
    /// received but not yet saved is planned for fetching again (never
    /// trusted), and the final save records everything.
    #[test]
    fn progress_is_saved_in_batches_and_unsaved_files_are_fetched_again() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("staging");
        std::fs::create_dir_all(staging.join("core-0")).unwrap();
        let mut files = Vec::new();
        let mut batch = None;
        for k in 0..(PROGRESS_BATCH_FILES + 10) {
            let bytes = data(16 + k % 5, k as u8);
            let e = entry(&format!("core-0/f{k:05}.rstc"), &bytes);
            let mut w = PartWriter::open(&staging, &e, None).unwrap();
            w.write_chunk(0, &bytes, crc32(&bytes)).unwrap();
            let b = batch.get_or_insert_with(|| ProgressBatch::load(&staging).unwrap().with_interval(std::time::Duration::from_secs(3600)));
            let (done, dir) = w.finish_deferred().unwrap();
            assert_eq!(done, e);
            b.record(&staging, &done, dir).unwrap();
            files.push(e);
        }
        let mut batch = batch.unwrap();
        // The first batch is on disk; the last ten are recorded in memory only.
        assert_eq!(batch.pending(), 10);
        assert_eq!(Progress::load(&staging).unwrap().done.len(), PROGRESS_BATCH_FILES);
        let unsaved = plan(&staging, &tmp.path().join("reuse"), &manifest(files.clone())).unwrap();
        assert_eq!(unsaved.files.iter().filter(|f| f.action == FileAction::Staged).count(), PROGRESS_BATCH_FILES);
        assert_eq!(unsaved.files.iter().filter(|f| f.action == FileAction::Fetch).count(), 10, "unrecorded files are fetched again");
        // A fresh attempt: receive the ten again, then the final save.
        let mut batch2 = ProgressBatch::load(&staging).unwrap().with_interval(std::time::Duration::from_secs(3600));
        for (k, e) in files.iter().enumerate().skip(PROGRESS_BATCH_FILES) {
            let bytes = data(16 + k % 5, k as u8);
            let mut w = PartWriter::open(&staging, e, None).unwrap();
            w.write_chunk(0, &bytes, crc32(&bytes)).unwrap();
            let (done, dir) = w.finish_deferred().unwrap();
            batch2.record(&staging, &done, dir).unwrap();
        }
        batch2.save(&staging).unwrap();
        assert_eq!(batch2.pending(), 0);
        let all = plan(&staging, &tmp.path().join("reuse"), &manifest(files)).unwrap();
        assert!(all.files.iter().all(|f| f.action == FileAction::Staged));
        batch.save(&staging).unwrap();
        // A large file is saved as soon as it completes.
        let big = data(PROGRESS_BATCH_BYTES as usize + 1, 7);
        let e = entry("core-0/big.rstc", &big);
        let mut w = PartWriter::open(&staging, &e, None).unwrap();
        w.write_chunk(0, &big, crc32(&big)).unwrap();
        let mut batch3 = ProgressBatch::load(&staging).unwrap().with_interval(std::time::Duration::from_secs(3600));
        let (done, dir) = w.finish_deferred().unwrap();
        batch3.record(&staging, &done, dir).unwrap();
        assert_eq!(batch3.pending(), 0, "a file of at least PROGRESS_BATCH_BYTES is saved at once");
        assert!(Progress::load(&staging).unwrap().done.contains_key("core-0/big.rstc"));
    }

    /// A file laid out as a vector segment ends: its content, the CRC-32 of
    /// the content, then the `RVEC` magic.
    fn self_checksummed(n: usize, seed: u8) -> Vec<u8> {
        let mut bytes = data(n, seed);
        let crc = crc32(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(b"RVEC");
        bytes
    }

    /// C20 root cause: two different self-checksummed files of one length
    /// have the same whole-file CRC-32, so name + length + CRC planned a
    /// stale old-store file as identical to the source's. Their tails (the
    /// content CRC) differ, which is what the target now compares.
    #[test]
    fn self_checksummed_files_of_one_length_differ_only_in_their_tails() {
        let (old, new) = (self_checksummed(248, 1), self_checksummed(248, 7));
        assert_ne!(old, new);
        assert_eq!(crc32(&old), crc32(&new), "the whole-file CRC-32 is the residue for both");
        let dir = tempfile::tempdir().unwrap();
        let (staging, reuse) = (dir.path().join("staging"), dir.path().join("reuse"));
        std::fs::create_dir_all(reuse.join("core-0")).unwrap();
        std::fs::write(reuse.join("core-0/vec_00000260.rvec"), &old).unwrap();
        let source = entry("core-0/vec_00000260.rvec", &new);
        let mut plan = plan(&staging, &reuse, &manifest(vec![source.clone()])).unwrap();
        // The manifest check alone takes the stale file.
        let candidate = local_candidate(&staging, &plan.files[0]).unwrap().expect("planned as reused");
        assert!(matches!(plan.files[0].action, FileAction::Reuse(_)));
        let (offset, n) = tail_range(source.len);
        let source_tail = &new[offset as usize..][..n as usize];
        let local = local_tail(&candidate, source.len).unwrap().unwrap();
        assert_ne!(local.as_slice(), source_tail, "the tails tell them apart");
        assert_eq!(demote_to_fetch(&mut plan.files[0]), source.len);
        assert_eq!(plan.bytes_to_fetch(), source.len);
        // An identical file still matches by its tail too.
        std::fs::write(&candidate, &new).unwrap();
        assert_eq!(local_tail(&candidate, source.len).unwrap().unwrap().as_slice(), source_tail);
        // A file of another length never matches.
        assert!(local_tail(&candidate, source.len + 1).unwrap().is_none());
        assert_eq!(tail_range(10), (0, 10));
    }

    #[test]
    fn chunk_bytes_cross_bincode_as_one_byte_string() {
        let b = ChunkBytes::copy_from(&(0..=255u8).collect::<Vec<_>>()).unwrap();
        let wire = bincode::serialize(&b).unwrap();
        assert_eq!(wire.len(), 8 + 256);
        assert_eq!(bincode::deserialize::<ChunkBytes>(&wire).unwrap(), b);
    }

    #[test]
    fn interrupted_file_resumes_from_its_last_checkpoint() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("shard-0.recovery");
        let bytes = data(3 * SYNC_BYTES as usize + 12345, 3);
        let e = entry("core-1/vec_00000001.rvec", &bytes);
        let m = manifest(vec![e.clone()]);
        std::fs::create_dir_all(&staging).unwrap();
        // 1 MiB chunks: stops after 40 chunks; the sidecar holds 32 MiB.
        assert!(!receive(&staging, &e, &bytes, None, 1 << 20, Some(40)).unwrap());
        let p = plan(&staging, &tmp.path().join("none"), &m).unwrap();
        let FileAction::Resume { done, running } = p.files[0].action.clone() else { panic!("{:?}", p.files[0].action) };
        assert_eq!(done, 40 << 20, "suspend checkpoints everything verified");
        assert_eq!(p.bytes_to_fetch(), e.len - done);
        assert!(receive(&staging, &e, &bytes, Some((done, running)), 1 << 20, None).unwrap());
        assert_eq!(std::fs::read(staging.join(&e.path)).unwrap(), bytes);
        assert!(!suffixed(&staging.join(&e.path), PART_SUFFIX).exists() && !suffixed(&staging.join(&e.path), SIDECAR_SUFFIX).exists());
        // A retry against the same manifest keeps it.
        assert_eq!(plan(&staging, &tmp.path().join("none"), &m).unwrap().files[0].action, FileAction::Staged);
    }

    #[test]
    fn corrupt_chunks_and_files_are_rejected_and_refetched() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("s");
        let bytes = data(300_000, 5);
        let e = entry("core-1/seg_00000001.rstc", &bytes);
        let mut w = PartWriter::open(&staging, &e, None).unwrap();
        let mut bad = bytes[..100_000].to_vec();
        bad[7] ^= 1;
        // In flight: the chunk's checksum is the source's.
        let err = w.write_chunk(0, &bad, crc32(&bytes[..100_000])).unwrap_err();
        assert!(err.starts_with(CHUNK_CORRUPT), "{err}");
        assert_eq!(w.done(), 0, "nothing written");
        // On disk after a crash: the whole-file checksum catches it and the
        // file starts over.
        w.write_chunk(0, &bytes[..100_000], crc32(&bytes[..100_000])).unwrap();
        w.suspend().unwrap();
        {
            let part = std::fs::OpenOptions::new().write(true).open(suffixed(&staging.join(&e.path), PART_SUFFIX)).unwrap();
            write_all_at(&part, &[bytes[50] ^ 0xff], 50).unwrap();
        }
        let m = manifest(vec![e.clone()]);
        let FileAction::Resume { done, running } = plan(&staging, tmp.path(), &m).unwrap().files[0].action.clone() else { panic!() };
        let err = receive(&staging, &e, &bytes, Some((done, running)), 64 << 10, None).unwrap_err();
        assert!(err.starts_with(FILE_CORRUPT), "{err}");
        assert!(!staging.join(&e.path).exists(), "never renamed into place");
        assert_eq!(plan(&staging, tmp.path(), &m).unwrap().files[0].action, FileAction::Fetch, "re-sent from zero");
        assert!(receive(&staging, &e, &bytes, None, 64 << 10, None).unwrap());
        assert_eq!(std::fs::read(staging.join(&e.path)).unwrap(), bytes);
    }

    #[test]
    fn identical_old_store_files_are_reused_and_others_fetched() {
        let tmp = tempfile::tempdir().unwrap();
        let (staging, reuse) = (tmp.path().join("shard-0.recovery"), tmp.path().join("shard-0.reuse"));
        let same = data(70_000, 1);
        let other = data(70_000, 2);
        std::fs::create_dir_all(reuse.join("core-4")).unwrap();
        std::fs::write(reuse.join("core-4/seg_00000001.rstc"), &same).unwrap();
        // Same name and length, other content: not reused.
        std::fs::write(reuse.join("core-4/seg_00000002.rstc"), &other).unwrap();
        std::fs::write(reuse.join("shard_state.json"), b"{}").unwrap();
        let state = b"{\"committed_seq_no\":3}".to_vec();
        let m = manifest(vec![entry("shard_state.json", &state), entry("core-9/seg_00000001.rstc", &same), entry("core-9/seg_00000002.rstc", &data(70_000, 3))]);
        let p = plan(&staging, &reuse, &m).unwrap();
        assert_eq!(p.files[0].action, FileAction::Fetch, "only core files are reused");
        assert_eq!(p.files[1].action, FileAction::Reuse(reuse.join("core-4/seg_00000001.rstc")));
        assert_eq!(p.files[2].action, FileAction::Fetch);
        let s = p.stats().unwrap();
        assert_eq!((s.files_total, s.files_reused, s.bytes_reused), (3, 1, 70_000));
        assert_eq!(s.files_percent(), "0.0%");
        let mut progress = Progress::load(&staging).unwrap();
        link_reused(&staging, &m.files[1], &reuse.join("core-4/seg_00000001.rstc"), &mut progress).unwrap();
        assert_eq!(std::fs::read(staging.join("core-9/seg_00000001.rstc")).unwrap(), same);
        // Leftovers of another manifest are swept; the reused file is kept.
        std::fs::write(staging.join("core-9/stale.rvec"), b"x").unwrap();
        let p = plan(&staging, &reuse, &m).unwrap();
        assert_eq!(p.files[1].action, FileAction::Staged);
        assert!(!staging.join("core-9/stale.rvec").exists());
    }

    #[test]
    fn a_full_disk_refuses_the_transfer() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(check_disk(tmp.path(), 1, 0).is_ok());
        let err = check_disk(tmp.path(), u64::MAX / 2, 0).unwrap_err();
        assert!(err.starts_with(INSUFFICIENT_DISK), "{err}");
    }

    #[test]
    fn concurrent_recoveries_reserve_their_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let available = crate::stats::disk_usage(tmp.path()).available_bytes;
        if crate::stats::disk_usage(tmp.path()).total_bytes == 0 {
            return; // no disk statistics: every check passes
        }
        let share = available / 5 * 3;
        let first = reserve_disk(tmp.path(), share, 0).unwrap();
        let err = reserve_disk(tmp.path(), share, 0).err().expect("the first recovery's bytes are counted");
        assert!(err.starts_with(INSUFFICIENT_DISK), "{err}");
        // As the first one writes, the disk statistics count its bytes
        // instead (not here: nothing is written), and its reservation shrinks.
        first.written(share);
        assert_eq!(first.left(), 0);
        let second = reserve_disk(tmp.path(), share, 0).unwrap();
        drop(first);
        drop(second);
        assert!(reserve_disk(tmp.path(), share, 0).is_ok(), "released when dropped");
    }

    #[test]
    fn staging_lock_is_exclusive_in_process() {
        let p = PathBuf::from("/nonexistent/shard-0.recovery");
        let a = StagingLock::try_acquire(&p).unwrap().unwrap();
        assert!(StagingLock::try_acquire(&p).unwrap().is_none());
        drop(a);
        assert!(StagingLock::try_acquire(&p).unwrap().is_some());
    }
    #[test]
    fn recovery_format_checks_actual_files_and_checkpoint_rows() {
        let segment = |version: u32| [b"RSTC".as_slice(), &version.to_le_bytes()].concat();
        let checkpoint = |version: u32| [b"RMVC\x01\0\0\0".as_slice(), &3u64.to_le_bytes(), b"abc", b"RMVR", &version.to_le_bytes()].concat();
        // Format 2 (code log references): readable from segment format v13.
        let checkpoint2 = |version: u32| [b"RMVC\x02\0\0\0".as_slice(), &3u64.to_le_bytes(), b"abc", b"RMVR", &version.to_le_bytes()].concat();
        for (name, data, reader, expected) in [
            ("core-1/seg_1.rstc", segment(11), 11, true),
            ("core-1/seg_1.rstc", segment(12), 11, false),
            ("core-1/seg_1.rstc", segment(12), 12, true),
            ("core-1/field.rmvc", checkpoint(2), 11, true),
            ("core-1/field.rmvc", checkpoint(3), 11, false),
            ("core-1/field.rmvc", checkpoint(3), 12, true),
            ("core-1/field.rmvc", checkpoint(4), 12, false),
            ("core-1/field.rmvc", checkpoint(4), 13, false),
            ("core-1/field.rmvc", checkpoint2(4), 12, false),
            ("core-1/field.rmvc", checkpoint2(4), 13, true),
            ("core-1/field.rmvc", checkpoint2(3), 13, false),
            // DiskBBQ IVF files and code log chunks: new file kinds of v13.
            ("core-1/ivf_0001_0000000000000002.rivc", b"RIVC".to_vec(), 12, false),
            ("core-1/ivf_0001_0000000000000002.rivp", b"RIVP".to_vec(), 11, false),
            ("core-1/ivf_0001_0000000000000002.rivp", b"RIVP".to_vec(), 13, true),
            ("core-1/mvq_0000000000000001_00000000_00000010.rmvq", vec![0; 4], 12, false),
            ("core-1/mvq_0000000000000001_00000000_00000010.rmvq", vec![0; 4], 13, true),
            ("core-1/vec_1.rvec", vec![0; 4], 11, true),
        ] {
            assert_eq!(elasticsearch_cluster::compatibility::VECTOR_FILES_SINCE, elasticsearch_engine::rustic_core::codec::VECTOR_FILES_SINCE);
            let m = manifest(vec![entry(name, &data)]);
            let result = validate_file_formats(&m, reader, |path, offset, len| {
                assert_eq!(path, name);
                data.get(offset as usize..(offset + len) as usize).map(|s| s.to_vec()).ok_or_else(|| "short read".into())
            });
            assert_eq!(result.is_ok(), expected, "{name}: {result:?}");
            if let Some(reason) = result.as_ref().map_err(String::as_str).err().filter(|r| r.contains("target reads through")) {
                // Tagged: the master does not count it toward max_retry.
                assert!(elasticsearch_cluster::compatibility::refused_segment_format(reason).is_some_and(|v| v > reader), "{reason}");
            }
        }
    }

    #[test]
    fn recovery_rejects_torn_and_overflowed_checkpoint_before_install() {
        let data = [b"RMVC\x01\0\0\0".as_slice(), &u64::MAX.to_le_bytes()].concat();
        let m = manifest(vec![entry("core-1/field.rmvc", &data)]);
        assert!(validate_file_formats(&m, 12, |_, _, _| Ok(data.clone())).is_err());
        assert!(validate_file_formats(&m, 12, |_, _, _| Ok(vec![0; 7])).is_err());
    }

}
