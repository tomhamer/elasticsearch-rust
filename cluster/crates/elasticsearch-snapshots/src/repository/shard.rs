//! Shard-level snapshot formats and the incremental shard snapshot / restore.

use super::{DATA_BLOB_PREFIX, INDEX_FILE_PREFIX, Repository, RepositoryData, snapshot_blob_name};
use crate::blobstore::ByteStream;
use crate::error::{Result, SnapshotError};
use crate::snapshot::{ShardStage, ShardStatus, SnapshotStats};
use crate::util::{CancelToken, RateLimiter, now_millis, random_base64_uuid};
use bytes::Bytes;
use futures::StreamExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

fn workspace_size(parts: impl IntoIterator<Item=usize>) -> Result<usize> {
    parts.into_iter().try_fold(0usize, |n, part| n.checked_add(part))
        .ok_or_else(|| SnapshotError::new("circuit_breaking_exception",429,"snapshot workspace overflow"))
}
fn file_info_bytes(file:&FileInfo)->Result<usize> {
    workspace_size([file.name.len(),file.physical_name.len(),file.checksum.len(),file.written_by.len()])
}
fn reserve_slots<T>(values:&mut Vec<T>,count:usize,memory:&mut dyn crate::service::SnapshotWorkspace)->Result<()> {
    if count>values.capacity() {
        let bytes=count.checked_mul(std::mem::size_of::<T>()).ok_or_else(||SnapshotError::new("circuit_breaking_exception",429,"snapshot vector overflow"))?;
        // The old allocation can coexist during reallocation. Retain both
        // charges until this operation's metadata and upload consumers drop.
        memory.grow(bytes,"snapshot metadata vector")?;
        values.try_reserve_exact(count-values.len()).map_err(|e|SnapshotError::new("circuit_breaking_exception",429,e.to_string()))?;
    }
    Ok(())
}

/// ES `BlobStoreIndexShardSnapshot.FileInfo`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileInfo {
    /// Blob name (`__{uuid}`).
    pub name: String,
    /// Path of the file relative to the shard directory.
    pub physical_name: String,
    pub length: u64,
    /// Hex SHA-256 of the file content.
    pub checksum: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_size: Option<u64>,
    #[serde(default)]
    pub written_by: String,
}

impl FileInfo {
    pub fn number_of_parts(&self) -> u64 {
        match self.part_size {
            Some(ps) if ps > 0 && self.length > 0 => self.length.div_ceil(ps),
            _ => 1,
        }
    }

    /// ES `partName`: the bare name when there is a single part.
    pub fn part_name(&self, part: u64) -> String {
        if self.number_of_parts() > 1 { format!("{}.part{}", self.name, part) } else { self.name.clone() }
    }

    pub fn part_bytes(&self, part: u64) -> u64 {
        match self.part_size {
            Some(ps) if self.number_of_parts() > 1 => {
                let start = part * ps;
                ps.min(self.length - start)
            }
            _ => self.length,
        }
    }

    pub fn part_names(&self) -> Vec<String> {
        (0..self.number_of_parts()).map(|p| self.part_name(p)).collect()
    }

    /// Same physical file as `other` (name + length + checksum).
    pub fn is_same(&self, physical_name: &str, length: u64, checksum: &str) -> bool {
        self.physical_name == physical_name && self.length == length && self.checksum == checksum
    }
}

/// One snapshot's entry in the shard-level index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShardSnapshotFiles {
    #[serde(default)]
    pub uuid: String,
    /// Blob names (`__...`) of the files in this snapshot of the shard.
    pub files: Vec<String>,
}

/// ES `BlobStoreIndexShardSnapshots` (`indices/{id}/{shard}/index-{gen}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShardSnapshots {
    #[serde(default)]
    pub files: Vec<FileInfo>,
    /// snapshot name → files.
    #[serde(default)]
    pub snapshots: BTreeMap<String, ShardSnapshotFiles>,
}

impl ShardSnapshots {
    pub fn find_file(&self, physical_name: &str, length: u64, checksum: &str) -> Option<&FileInfo> {
        self.files.iter().find(|f| f.is_same(physical_name, length, checksum))
    }

    /// Rebuild `files` to contain exactly the files referenced by `snapshots`.
    pub fn compact(&mut self) {
        let referenced: BTreeSet<&String> = self.snapshots.values().flat_map(|s| s.files.iter()).collect();
        let mut seen = BTreeSet::new();
        self.files.retain(|f| referenced.contains(&f.name) && seen.insert(f.name.clone()));
    }

    /// All blob names (including parts) referenced.
    pub fn referenced_blobs(&self) -> BTreeSet<String> {
        self.files.iter().flat_map(|f| f.part_names()).collect()
    }
}

/// ES `BlobStoreIndexShardSnapshot` (`indices/{id}/{shard}/snap-{uuid}.dat`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardSnapshotMeta {
    pub name: String,
    #[serde(default)]
    pub index_version: i64,
    pub start_time: i64,
    pub time: i64,
    /// Incremental file count (ES naming).
    pub number_of_files: u64,
    /// Incremental size (ES naming).
    pub total_size: u64,
    pub files: Vec<FileInfo>,
}

impl ShardSnapshotMeta {
    pub fn stats(&self) -> SnapshotStats {
        SnapshotStats {
            incremental_file_count: self.number_of_files,
            incremental_size: self.total_size,
            processed_file_count: self.number_of_files,
            processed_size: self.total_size,
            total_file_count: self.files.len() as u64,
            total_size: self.files.iter().map(|f| f.length).sum(),
            start_time: self.start_time,
            time: self.time,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSnapshotResult {
    pub index_id: String,
    pub shard: u32,
    pub new_gen: String,
    pub old_gen: Option<String>,
    pub stats: SnapshotStats,
    pub total_size: u64,
    pub segment_count: u32,
}

fn validate_rel_path(p: &str) -> Result<()> {
    let path = Path::new(p);
    if p.is_empty()
        || path.is_absolute()
        || path.components().any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(SnapshotError::illegal_argument(format!("invalid shard file path [{p}]")));
    }
    Ok(())
}

/// Length + SHA-256 of a local file (on a blocking thread).
pub async fn hash_file(path: PathBuf) -> Result<(u64, String)> {
    hash_file_owned(AdmittedPath{path,_memory:None}).await
}
async fn hash_snapshot_file(files:&crate::service::SnapshotFiles,dir:&Path,rel:&str)->Result<(u64,String)> {
    let mut memory=files.workspace()?;
    memory.grow(workspace_size([dir.as_os_str().len(),rel.len(),1,1<<20,64])?,"snapshot hashing buffer and path")?;
    hash_file_owned(AdmittedPath{path:joined_path(dir,rel)?,_memory:Some(memory)}).await
}
struct AdmittedPath { path:PathBuf, _memory:Option<Box<dyn crate::service::SnapshotWorkspace>> }
impl AdmittedPath {fn as_path(&self)->&Path{&self.path}}
fn reserved_text(bytes:usize)->Result<String>{
    let mut text=String::new();
    text.try_reserve_exact(bytes).map_err(|e|SnapshotError::new("circuit_breaking_exception",429,e.to_string()))?;
    Ok(text)
}
fn joined_path(dir:&Path,rel:&str)->Result<PathBuf>{
    let bytes=workspace_size([dir.as_os_str().len(),rel.len(),1])?;
    let mut path=PathBuf::new();path.try_reserve_exact(bytes).map_err(|e|SnapshotError::new("circuit_breaking_exception",429,e.to_string()))?;
    path.push(dir);path.push(rel);Ok(path)
}
async fn hash_file_owned(owned:AdmittedPath)->Result<(u64,String)> {
    tokio::task::spawn_blocking(move || -> Result<(u64, String)> {
        // The blocking operation can outlive cancellation of its awaiting task.
        let path=owned.as_path();
        use std::io::Read;
        let mut f = std::fs::File::open(&path).map_err(|e| SnapshotError::io(format!("reading [{}]", path.display()), e))?;
        let mut h = sha2::Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut len = 0u64;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            len += n as u64;
            h.update(&buf[..n]);
        }
        Ok((len, hex::encode(h.finalize())))
    })
    .await
    .map_err(|e| SnapshotError::illegal_state(format!("hashing task failed: {e}")))?
}

/// Length, SHA-256 and CRC-32 of a local file, in one read (on a blocking
/// thread): a file a restore keeps from an earlier attempt.
async fn hash_file_with_crc(path: PathBuf) -> Result<(u64, String, u32)> {
    tokio::task::spawn_blocking(move || -> Result<(u64, String, u32)> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path).map_err(|e| SnapshotError::io(format!("reading [{}]", path.display()), e))?;
        let mut h = sha2::Sha256::new();
        let mut crc = 0xFFFF_FFFFu32;
        let mut buf = vec![0u8; 1 << 20];
        let mut len = 0u64;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            len += n as u64;
            h.update(&buf[..n]);
            crc = crate::util::crc32_update(crc, &buf[..n]);
        }
        Ok((len, hex::encode(h.finalize()), !crc))
    })
    .await
    .map_err(|e| SnapshotError::illegal_state(format!("hashing task failed: {e}")))?
}

/// Stream `len` bytes of `path` from `offset`, rate limited and cancellable.
fn file_range_stream(
    path: PathBuf,
    path_owner:Box<dyn crate::service::SnapshotWorkspace>,
    source:crate::service::SnapshotFiles,
    offset: u64,
    len: u64,
    buf_size: usize,
    rate: Arc<RateLimiter>,
    cancel: CancelToken,
    processed: Arc<Mutex<ShardStatus>>,
) -> ByteStream {
    struct St {
        file: Option<std::fs::File>,
        remaining: u64,
        path:PathBuf,
        _path_owner:Box<dyn crate::service::SnapshotWorkspace>,
    }
    struct Chunk {bytes:Vec<u8>,_memory:Box<dyn crate::service::SnapshotWorkspace>}
    impl AsRef<[u8]> for Chunk {fn as_ref(&self)->&[u8]{&self.bytes}}
    futures::stream::try_unfold(St {file:None,remaining:len,path,_path_owner:path_owner},move |st| {
        let source=source.clone();let rate=rate.clone();let cancel=cancel.clone();let processed=processed.clone();
        async move {
            if st.remaining==0{return Ok(None)}
            // The worker owns the file, path and allocation reservations. If
            // this future is cancelled, none can be released before its read.
            let (mut st,chunk)=tokio::task::spawn_blocking(move || -> Result<(St,Chunk)> {
                use std::io::{Read,Seek};
                let mut st=st;cancel.check()?;
                if st.file.is_none(){let mut file=std::fs::File::open(&st.path)?;file.seek(std::io::SeekFrom::Start(offset))?;st.file=Some(file);}
                let n=(buf_size as u64).min(st.remaining)as usize;
                let mut memory=source.workspace()?;
                memory.grow(workspace_size([n,std::mem::size_of::<Chunk>(),4*std::mem::size_of::<usize>()])?,"snapshot upload chunk")?;
                let mut buf=Vec::new();buf.try_reserve_exact(n).map_err(|e|SnapshotError::new("circuit_breaking_exception",429,e.to_string()))?;buf.resize(n,0);
                st.file.as_mut().unwrap().read_exact(&mut buf)?;
                Ok((st,Chunk{bytes:buf,_memory:memory}))
            }).await.map_err(|e|SnapshotError::illegal_state(format!("snapshot read task failed: {e}")))??;
            let n=chunk.bytes.len();rate.acquire(n as u64).await;
            processed.lock().stats.processed_size+=n as u64;st.remaining-=n as u64;
            Ok(Some((Bytes::from_owner(chunk),st)))
        }
    }).boxed()
}

/// Inputs of one shard snapshot.
pub struct ShardSnapshotTask {
    pub snapshot_name: String,
    pub snapshot_uuid: String,
    pub index_id: String,
    pub shard: u32,
    pub dir: PathBuf,
    pub files: crate::service::SnapshotFiles,
    pub status: Arc<Mutex<ShardStatus>>,
    pub cancel: CancelToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreShardResult {
    pub files: u64,
    pub bytes: u64,
    pub reused_files: u64,
    /// Every file of the shard snapshot as written (or kept) in the target
    /// directory, each verified against the snapshot's length and SHA-256.
    pub restored: Vec<RestoredFile>,
}

/// One file a [`Repository::restore_shard`] wrote (or kept): its CRC-32
/// (IEEE) was computed over the very bytes that matched the snapshot's
/// SHA-256, so a caller can hand the file to a CRC-checked consumer (the
/// engine's file adoption) without reading it again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredFile {
    pub physical_name: String,
    pub length: u64,
    pub crc32: u32,
}

impl Repository {
    /// Incrementally snapshot one shard: files already in the repository with
    /// the same name, length and SHA-256 are referenced, the rest uploaded.
    pub async fn snapshot_shard(&self, data: &RepositoryData, task: ShardSnapshotTask) -> Result<ShardSnapshotResult> {
        self.ensure_writable("create snapshot")?;
        let lock = self.shard_lock(&task.index_id, task.shard);
        let _guard = lock.lock().await;
        let old_gen = self.current_shard_gen(data, &task.index_id, task.shard);
        let res = self.snapshot_shard_at(old_gen, task).await?;
        self.pending_shard_gens.lock().insert((res.index_id.clone(), res.shard), res.new_gen.clone());
        Ok(res)
    }

    /// Snapshot a shard through `run`, which writes the shard's blobs
    /// (usually on the node holding its primary, ES `SnapshotShardsService`)
    /// on top of the shard generation it is given and returns the new one.
    /// Holds this process's shard lock meanwhile, so shard operations of the
    /// coordinating node (other snapshots, deletes) see the generations in
    /// order. The result is accepted (and its generation recorded for the
    /// finalize) only once [`Self::verify_shard_snapshot`] finds the shard
    /// snapshot complete in the repository under this `index_id`: a shard
    /// is never reported successful from another node's word alone.
    pub async fn snapshot_shard_with<F, Fut>(&self, data: &RepositoryData, index_id: &str, shard: u32, snapshot_uuid: &str, run: F) -> Result<ShardSnapshotResult>
    where
        F: FnOnce(Option<String>) -> Fut,
        Fut: std::future::Future<Output = Result<ShardSnapshotResult>>,
    {
        self.ensure_writable("create snapshot")?;
        let lock = self.shard_lock(index_id, shard);
        let _guard = lock.lock().await;
        let old_gen = self.current_shard_gen(data, index_id, shard);
        let mut res = run(old_gen.clone()).await?;
        self.verify_shard_snapshot(index_id, shard, snapshot_uuid, &res.new_gen).await?;
        res.index_id = index_id.to_string();
        res.shard = shard;
        res.old_gen = old_gen;
        self.pending_shard_gens.lock().insert((index_id.to_string(), shard), res.new_gen.clone());
        Ok(res)
    }

    /// Whether snapshot `snapshot_uuid` of `index_id`/`shard` is durably in
    /// the repository: its `snap-{uuid}.dat`, the shard index
    /// `index-{gen_}` listing it, and every data blob (part) it references.
    pub async fn verify_shard_snapshot(&self, index_id: &str, shard: u32, snapshot_uuid: &str, gen_: &str) -> Result<()> {
        let fail = |m: String| SnapshotError::new("snapshot_exception", 500, format!("[{index_id}][{shard}] {m}"));
        let meta = self
            .read_shard_snapshot(index_id, shard, snapshot_uuid)
            .await
            .map_err(|e| fail(format!("shard snapshot [indices/{index_id}/{shard}/{}] is not in the repository: {e}", snapshot_blob_name(snapshot_uuid))))?;
        let index = self
            .read_shard_snapshots(index_id, shard, Some(gen_))
            .await
            .map_err(|e| fail(format!("shard generation [{gen_}] is not readable: {e}")))?;
        if !index.snapshots.values().any(|s| s.uuid == snapshot_uuid) {
            return Err(fail(format!("shard generation [{gen_}] does not list snapshot [{snapshot_uuid}]")));
        }
        let container = self.shard_container(index_id, shard);
        let listed = match container.list_blobs_by_prefix(DATA_BLOB_PREFIX).await {
            Ok(l) => Some(l),
            Err(e) if e.kind == "unsupported_operation_exception" => None,
            Err(e) => return Err(e),
        };
        for f in &meta.files {
            for part in f.part_names() {
                let present = match &listed {
                    Some(l) => l.contains_key(&part),
                    None => container.blob_exists(&part).await?,
                };
                if !present {
                    return Err(fail(format!("data blob [{part}] of [{}] is not in the repository", f.physical_name)));
                }
            }
        }
        Ok(())
    }

    /// [`Self::snapshot_shard`] on top of shard generation `old_gen` (`None`:
    /// the shard has none yet), without the shard lock and without recording
    /// the new generation: the caller coordinating the snapshot does both
    /// ([`Self::snapshot_shard_with`]). Used directly by a data node writing
    /// its own primary's blobs.
    pub async fn snapshot_shard_at(&self, old_gen: Option<String>, task: ShardSnapshotTask) -> Result<ShardSnapshotResult> {
        // Declared before all derived metadata, so their buffers drop first.
        let mut workspace=task.files.workspace()?;
        self.ensure_writable("create snapshot")?;
        let start = now_millis();
        {
            let mut st = task.status.lock();
            st.stage = ShardStage::Started;
            st.stats.start_time = start;
        }
        let mut shard_index = self.read_shard_snapshots(&task.index_id, task.shard, old_gen.as_deref()).await?;
        let container = self.shard_container(&task.index_id, task.shard);

        let mut files: Vec<FileInfo> = Vec::new();
        reserve_slots(&mut files,task.files.len(),workspace.as_mut())?;
        // Upload entries borrow the final file metadata through stable indices.
        let mut to_upload: Vec<usize> = Vec::new();
        reserve_slots(&mut to_upload,task.files.len(),workspace.as_mut())?;
        let mut seen = BTreeSet::new();
        for rel in task.files.iter() {
            task.cancel.check()?;
            let tree_node=2*(11*std::mem::size_of::<String>()+20*std::mem::size_of::<usize>());
            workspace.grow(workspace_size([rel.len(),rel.len(),tree_node])?,"snapshot normalized path and deduplication")?;
            let mut normalized = reserved_text(rel.len())?;
            for c in rel.chars() { normalized.push(if c == '\\' { '/' } else { c }); }
            let rel = normalized;
            validate_rel_path(&rel)?;
            if !seen.insert(rel.clone()) {
                continue;
            }
            workspace.grow(64,"snapshot checksum text")?;
            let (len, checksum) = hash_snapshot_file(&task.files,&task.dir,&rel).await?;
            if let Some(existing) = shard_index.find_file(&rel, len, &checksum) {
                workspace.grow(file_info_bytes(existing)?,"snapshot reused file metadata")?;
                files.push(existing.clone());
            } else {
                workspace.grow(workspace_size([DATA_BLOB_PREFIX.len(),22,rel.len(),crate::snapshot::LUCENE_VERSION.len(),22])?,"snapshot new file metadata")?;
                let mut name = reserved_text(workspace_size([DATA_BLOB_PREFIX.len(),22])?)?;
                name.push_str(DATA_BLOB_PREFIX);
                name.push_str(&random_base64_uuid());
                let fi = FileInfo {
                    name,
                    physical_name: rel.clone(),
                    length: len,
                    checksum,
                    part_size: self.settings.chunk_size,
                    written_by: crate::snapshot::LUCENE_VERSION.to_string(),
                };
                to_upload.push(files.len());
                files.push(fi);
            }
        }
        let incremental_count = to_upload.len() as u64;
        let incremental_size: u64 = to_upload.iter().map(|&i| files[i].length).sum();
        {
            let mut st = task.status.lock();
            st.stats.incremental_file_count = incremental_count;
            st.stats.incremental_size = incremental_size;
            st.stats.total_file_count = files.len() as u64;
            st.stats.total_size = files.iter().map(|f| f.length).sum();
        }

        // Small reads when throttled so progress and cancellation stay responsive.
        let read_chunk = match self.snapshot_rate.bytes_per_sec() {
            Some(rate) => (rate / 10).clamp(8 * 1024, self.settings.io_buffer) as usize,
            None => self.settings.io_buffer as usize,
        };
        for &index in &to_upload {
            let fi=&files[index];
            for part in 0..fi.number_of_parts() {
                task.cancel.check()?;
                let offset = part * fi.part_size.unwrap_or(0);
                let len = fi.part_bytes(part);
                let path_bytes=workspace_size([task.dir.as_os_str().len(),fi.physical_name.len(),1])?;
                let mut path_owner=task.files.workspace()?;
                path_owner.grow(path_bytes,"snapshot upload path")?;
                let mut part_owner=task.files.workspace()?;
                part_owner.grow(workspace_size([fi.name.len(),25])?,"snapshot part name")?;
                let stream = file_range_stream(
                    joined_path(&task.dir,&fi.physical_name)?,path_owner,task.files.clone(),
                    offset,
                    len,
                    read_chunk,
                    self.snapshot_rate.clone(),
                    task.cancel.clone(),
                    task.status.clone(),
                );
                let mut part_name = reserved_text(workspace_size([fi.name.len(),25])?)?;
                part_name.push_str(&fi.name);
                if fi.number_of_parts() > 1 {
                    use std::fmt::Write;
                    write!(&mut part_name, ".part{part}").expect("writing to reserved String");
                }
                container.write_blob_stream(&part_name, len, stream, true).await?;
            }
            task.status.lock().stats.processed_file_count += 1;
        }

        task.status.lock().stage = ShardStage::Finalize;
        let time = now_millis() - start;
        let mut meta_files=Vec::new();reserve_slots(&mut meta_files,files.len(),workspace.as_mut())?;
        for file in &files {workspace.grow(file_info_bytes(file)?,"snapshot final file metadata")?;meta_files.push(file.clone());}
        workspace.grow(task.snapshot_name.len(),"snapshot metadata name")?;
        let meta = ShardSnapshotMeta {
            name: task.snapshot_name.clone(),
            index_version: 0,
            start_time: start,
            time,
            number_of_files: incremental_count,
            total_size: incremental_size,
            files: meta_files,
        };
        self.write_meta(&container, &snapshot_blob_name(&task.snapshot_uuid), &meta, false).await?;

        // New cumulative shard index.
        let count=shard_index.files.len().checked_add(to_upload.len()).ok_or_else(||SnapshotError::new("circuit_breaking_exception",429,"snapshot file count overflow"))?;
        reserve_slots(&mut shard_index.files,count,workspace.as_mut())?;
        for &index in &to_upload {
            let f=&files[index];workspace.grow(file_info_bytes(f)?,"snapshot cumulative file metadata")?;
            shard_index.files.push(f.clone());
        }
        let mut names=Vec::new();reserve_slots(&mut names,files.len(),workspace.as_mut())?;
        for file in &files {workspace.grow(file.name.len(),"snapshot referenced name")?;names.push(file.name.clone());}
        let entry_nodes=2*(11*std::mem::size_of::<(String,ShardSnapshotFiles)>()+20*std::mem::size_of::<usize>());
        workspace.grow(workspace_size([task.snapshot_name.len(),task.snapshot_uuid.len(),entry_nodes])?,"snapshot cumulative entry")?;
        shard_index.snapshots.insert(
            task.snapshot_name.clone(),
            ShardSnapshotFiles { uuid: task.snapshot_uuid.clone(), files: names },
        );
        // compact's temporary borrowed-reference and owned-name sets are
        // charged separately from the persisted metadata they index.
        let node=2*(11*std::mem::size_of::<String>()+20*std::mem::size_of::<usize>());
        for snapshot in shard_index.snapshots.values(){for _ in &snapshot.files {workspace.grow(node,"snapshot compact references")?;}}
        for file in &shard_index.files {workspace.grow(workspace_size([node,file.name.len()])?,"snapshot compact names")?;}
        shard_index.compact();
        let new_gen = random_base64_uuid();
        self.write_meta(&container, &format!("{INDEX_FILE_PREFIX}{new_gen}"), &shard_index, true).await?;

        let total_size = files.iter().map(|f| f.length).sum();
        let segment_count = files.iter().filter(|f| f.physical_name.ends_with(".si")).count() as u32;
        let stats = {
            let mut st = task.status.lock();
            st.stats.time = now_millis() - start;
            st.stats.processed_file_count = incremental_count;
            st.stats.processed_size = incremental_size;
            st.stage = ShardStage::Done;
            st.stats
        };
        Ok(ShardSnapshotResult {
            index_id: task.index_id,
            shard: task.shard,
            new_gen,
            old_gen,
            stats,
            total_size,
            segment_count,
        })
    }

    /// Read the shard-level snapshot metadata (`snap-{uuid}.dat`) of a shard.
    pub async fn read_shard_snapshot(&self, index_id: &str, shard: u32, snapshot_uuid: &str) -> Result<ShardSnapshotMeta> {
        self.read_meta(&self.shard_container(index_id, shard), &snapshot_blob_name(snapshot_uuid)).await
    }

    /// Write the files of a snapshotted shard into `target_dir`, verifying
    /// length and SHA-256. Files already present in `target_dir` with the same
    /// length and checksum are kept (incremental restore).
    pub async fn restore_shard(
        &self,
        snapshot_uuid: &str,
        index_id: &str,
        shard: u32,
        target_dir: &Path,
        cancel: &CancelToken,
    ) -> Result<RestoreShardResult> {
        let meta = self.read_shard_snapshot(index_id, shard, snapshot_uuid).await?;
        let container = self.shard_container(index_id, shard);
        tokio::fs::create_dir_all(target_dir).await?;
        let mut res = RestoreShardResult { files: 0, bytes: 0, reused_files: 0, restored: Vec::with_capacity(meta.files.len()) };
        for fi in &meta.files {
            cancel.check()?;
            validate_rel_path(&fi.physical_name)?;
            let dest = target_dir.join(&fi.physical_name);
            if let Ok(md) = tokio::fs::metadata(&dest).await {
                if md.len() == fi.length {
                    if let Ok((len, sha, crc)) = hash_file_with_crc(dest.clone()).await {
                        if len == fi.length && sha == fi.checksum {
                            res.reused_files += 1;
                            res.files += 1;
                            res.restored.push(RestoredFile { physical_name: fi.physical_name.clone(), length: len, crc32: crc });
                            continue;
                        }
                    }
                }
            }
            if let Some(p) = dest.parent() {
                tokio::fs::create_dir_all(p).await?;
            }
            let tmp = dest.with_file_name(format!(
                "restore-{}-{}",
                dest.file_name().unwrap_or_default().to_string_lossy(),
                random_base64_uuid()
            ));
            let r: Result<u32> = async {
                let mut out = tokio::io::BufWriter::new(tokio::fs::File::create(&tmp).await?);
                let mut h = sha2::Sha256::new();
                let mut crc = 0xFFFF_FFFFu32;
                let mut written = 0u64;
                for part in 0..fi.number_of_parts() {
                    let plen = fi.part_bytes(part);
                    let mut s = container.read_blob_stream(&fi.part_name(part), plen, self.settings.io_buffer);
                    while let Some(chunk) = s.next().await {
                        let chunk = chunk?;
                        cancel.check()?;
                        self.restore_rate.acquire(chunk.len() as u64).await;
                        h.update(&chunk);
                        crc = crate::util::crc32_update(crc, &chunk);
                        written += chunk.len() as u64;
                        out.write_all(&chunk).await?;
                    }
                }
                out.flush().await?;
                out.get_mut().sync_all().await?;
                let checksum = hex::encode(h.finalize());
                if written != fi.length || checksum != fi.checksum {
                    return Err(SnapshotError::new(
                        "corrupt_index_exception",
                        500,
                        format!(
                            "checksum failed (hardware problem?) : expected={} actual={} length={} written={} (resource={})",
                            fi.checksum, checksum, fi.length, written, fi.physical_name
                        ),
                    ));
                }
                tokio::fs::rename(&tmp, &dest).await?;
                Ok(!crc)
            }
            .await;
            let crc = match r {
                Ok(crc) => crc,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(e);
                }
            };
            res.files += 1;
            res.bytes += fi.length;
            res.restored.push(RestoredFile { physical_name: fi.physical_name.clone(), length: fi.length, crc32: crc });
        }
        if let Ok(d) = tokio::fs::File::open(target_dir).await {
            let _ = d.sync_all().await;
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn upload_chunks_retain_independent_admission_and_refuse_before_read() {
        use std::sync::atomic::{AtomicUsize,Ordering};
        use crate::service::{SnapshotFiles,SnapshotFileSource,SnapshotWorkspace};
        #[derive(Debug)]
        struct Source { used:Arc<AtomicUsize>,limit:usize }
        struct Owner { used:Arc<AtomicUsize>,limit:usize,held:usize }
        impl SnapshotWorkspace for Owner {
            fn grow(&mut self,bytes:usize,_:&'static str)->Result<()> {
                self.used.fetch_update(Ordering::SeqCst,Ordering::SeqCst,|n|n.checked_add(bytes).filter(|n|*n<=self.limit))
                    .map_err(|_|SnapshotError::new("circuit_breaking_exception",429,"fixture capacity"))?;
                self.held+=bytes;Ok(())
            }
        }
        impl Drop for Owner {fn drop(&mut self){self.used.fetch_sub(self.held,Ordering::SeqCst);}}
        impl SnapshotFileSource for Source {
            fn len(&self)->usize{0}
            fn get(&self,_:usize)->Option<&str>{None}
            fn workspace(&self)->Result<Box<dyn SnapshotWorkspace>>{Ok(Box::new(Owner{used:self.used.clone(),limit:self.limit,held:0}))}
        }
        let dir=tempfile::tempdir().unwrap();let path=dir.path().join("chunk");
        std::fs::write(&path,b"abcd").unwrap();
        let used=Arc::new(AtomicUsize::new(0));
        let source=SnapshotFiles(Arc::new(Source{used:used.clone(),limit:4096}));
        let mut stream=file_range_stream(path.clone(),source.workspace().unwrap(),source,0,4,2,
            Arc::new(RateLimiter::new(None)),CancelToken::default(),Arc::new(Mutex::new(ShardStatus::new(ShardStage::Started))));
        let first=stream.next().await.unwrap().unwrap();assert_eq!(&first[..],b"ab");
        let copy=first.clone();drop(first);drop(stream);
        assert!(used.load(Ordering::SeqCst)>0);drop(copy);assert_eq!(used.load(Ordering::SeqCst),0);
        let source=SnapshotFiles(Arc::new(Source{used:used.clone(),limit:1}));
        let mut stream=file_range_stream(path,source.workspace().unwrap(),source,0,4,2,
            Arc::new(RateLimiter::new(None)),CancelToken::default(),Arc::new(Mutex::new(ShardStatus::new(ShardStage::Started))));
        assert_eq!(stream.next().await.unwrap().unwrap_err().status,429);
        drop(stream);assert_eq!(used.load(Ordering::SeqCst),0);
    }

    #[test]
    fn parts() {
        let mut f = FileInfo {
            name: "__a".into(),
            physical_name: "x".into(),
            length: 25,
            checksum: String::new(),
            part_size: Some(10),
            written_by: String::new(),
        };
        assert_eq!(f.number_of_parts(), 3);
        assert_eq!(f.part_names(), vec!["__a.part0", "__a.part1", "__a.part2"]);
        assert_eq!((f.part_bytes(0), f.part_bytes(2)), (10, 5));
        f.part_size = Some(100);
        assert_eq!(f.part_names(), vec!["__a"]);
        f.length = 0;
        assert_eq!(f.number_of_parts(), 1);
        f.part_size = None;
        assert_eq!(f.part_bytes(0), 0);
    }
}
