//! Index-level file sets (hook 2, sharding phase 1): every file a node
//! needs to bring up a copy of an index from files, with checksums.
//!
//! Layout (relative to the index directory, as on disk):
//!
//! ```text
//! index_meta.json                      name, uuid, settings, mappings, aliases
//! shard-0/shard_state.json             commit point: committed seq_no, generations, seq_no state
//! shard-0/primary_terms.json           term history (if any term was ever bumped)
//! shard-0/core-<g>/meta.json, seg_*.rstc, seg_*.del, vec_*.rvec   the core commit
//! shard-0/translog/translog-*.tlog     (with include_translog) ops above the commit
//! shard-0/history/side-*.log           (with include_translog) no-ops / per-op terms
//! transfer_manifest.json               written last by the receiver
//! ```
//!
//! Retention leases and anything else under the shard directory (e.g. the
//! cluster crate's `cluster-replica-wal/`) are per copy and not included.

use crate::error::{EsError, Result};
use crate::hooks::retention::RetentionLock;
use rustic_core::codec::crc32;
use rustic_core::codec::directory::{FileIdentity, FileSlice};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "transfer_manifest.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFileEntry {
    /// Relative to the index directory, `/`-separated.
    pub path: String,
    pub len: u64,
    pub crc32: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexManifest {
    pub index: String,
    pub uuid: String,
    pub files: Vec<IndexFileEntry>,
    /// Seq_no of the core commit (`shard_state.json`).
    pub committed_seq_no: i64,
    /// Local checkpoint and max seq_no of the source when captured.
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    pub primary_term: u64,
    pub includes_translog: bool,
}

impl IndexManifest {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.len).sum()
    }
}

pub(crate) enum FileSource {
    #[allow(dead_code)]
    Mapped(FileSlice),
    Bytes(rustic_core::admission::Allocation<Vec<u8>>),
    /// An immutable file of the shard (core commit) that is also on disk at
    /// this path, so a receiver on the same filesystem can hard-link it,
    /// with the identity of the file the slice maps (from the mapped
    /// descriptor; the path may name another file by the time it is read).
    MappedAt(FileSlice, PathBuf, Option<FileIdentity>),
    /// Core pin and converted path/key retain admission until the source drops.
    PinnedAt(rustic_core::transfer::PinnedFile, PathBuf, rustic_core::admission::Reservation),
    /// An append-only file (translog / sidecar): its first `len` bytes.
    Disk(PathBuf, u64),
}

impl FileSource {
    fn read_range(&self, off:u64, len:u64)->Result<rustic_core::admission::Allocation<Vec<u8>>> {
        use rustic_core::admission::{self,Allocation};
        let total=self.len();let start=off.min(total);let end=off.saturating_add(len).min(total);
        let size=usize::try_from(end-start).map_err(|_|EsError::new(429,"circuit_breaking_exception","file range exceeds address space"))?;
        let memory=admission::shared().reserve(size as u64,"owned file source range")?;
        let mut out=Vec::new();out.try_reserve_exact(size).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        match self {
            FileSource::Mapped(s)|FileSource::MappedAt(s,..)=>out.extend_from_slice(&s.as_slice()[start as usize..end as usize]),
            FileSource::PinnedAt(pin,..)=>out.extend_from_slice(&pin.data.as_slice()[start as usize..end as usize]),
            FileSource::Bytes(v)=>out.extend_from_slice(&v[start as usize..end as usize]),
            FileSource::Disk(path,_)=>{use std::io::{Read,Seek,SeekFrom};let mut file=std::fs::File::open(path)?;file.seek(SeekFrom::Start(start))?;out.resize(size,0);file.read_exact(&mut out)?;}
        }
        Ok(Allocation::new(out,memory))
    }

    fn len(&self) -> u64 {
        match self {
            FileSource::Mapped(s) | FileSource::MappedAt(s, ..) => s.len() as u64,
            FileSource::PinnedAt(pin, ..) => pin.data.len() as u64,
            FileSource::Bytes(v) => v.len() as u64,
            FileSource::Disk(_, n) => *n,
        }
    }
}

/// Source registry ownership is admitted before keys/nodes are constructed.
/// Its guard is separate from value owners and therefore outlives tree nodes.
pub(crate) struct CapturedSources {
    pub(crate) values:BTreeMap<String,FileSource>,
    memory:rustic_core::admission::Reservation,
}
impl CapturedSources {
    pub(crate) fn new()->Result<Self>{Ok(Self{values:BTreeMap::new(),memory:rustic_core::admission::shared().reserve(0,"captured file source registry")?})}
    pub(crate) fn reserve_entry(&mut self,key_bytes:usize,extra_bytes:usize)->Result<()> {
        let bytes=(key_bytes as u64).checked_add(extra_bytes as u64).and_then(|n|n.checked_add(2048)).ok_or_else(||EsError::internal("file source registry size overflow"))?;
        self.memory.grow(bytes,"captured file source entry")?;Ok(())
    }
    #[cfg(test)]
    fn from_fixture(values:BTreeMap<String,FileSource>)->Self {
        let bytes=values.keys().map(|key|key.len() as u64+2048).sum();
        Self{values,memory:rustic_core::admission::shared().reserve(bytes,"file source fixture").unwrap()}
    }
}
/// Read exactly the current metadata file length under admission. A concurrent
/// replacement cannot expand the allocation beyond the captured descriptor.
pub(crate) fn read_metadata_file(path:&Path)->Result<rustic_core::admission::Allocation<Vec<u8>>> {
    read_metadata_descriptor(std::fs::File::open(path)?)
}
/// Optional metadata may be absent, but unreadable or changing metadata must
/// fail capture rather than publish a snapshot without its history identity.
pub(crate) fn read_optional_metadata_file(path:&Path)->Result<Option<rustic_core::admission::Allocation<Vec<u8>>>> {
    match std::fs::File::open(path) {
        Ok(file)=>read_metadata_descriptor(file).map(Some),
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>Ok(None),
        Err(error)=>Err(error.into()),
    }
}
fn read_metadata_descriptor(mut file:std::fs::File)->Result<rustic_core::admission::Allocation<Vec<u8>>> {
    use std::io::Read;
    let len=usize::try_from(file.metadata()?.len()).map_err(|_|EsError::internal("metadata file too large"))?;
    let memory=rustic_core::admission::shared().reserve(len as u64,"captured metadata file bytes")?;
    let mut bytes=Vec::new();bytes.try_reserve_exact(len).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;bytes.resize(len,0);
    file.read_exact(&mut bytes)?;
    let mut extra=[0u8;1];if file.read(&mut extra)?!=0{return Err(EsError::internal("metadata file changed during capture"));}
    Ok(rustic_core::admission::Allocation::new(bytes,memory))
}

/// A consistent, pinned set of an index's files (see
/// [`IndexService::snapshot_files`](crate::index::IndexService::snapshot_files)).
/// While it lives it holds a retention lock on the source shard (history
/// above the commit stays available for the ops-replay phase that follows
/// a file copy).
pub struct IndexFileSet {
    pub manifest: IndexManifest,
    pub(crate) sources: BTreeMap<String, FileSource>,
    pub(crate) lock: Option<RetentionLock>,
    _manifest_memory: rustic_core::admission::Reservation,
    _source_memory: rustic_core::admission::Reservation,
}

impl std::fmt::Debug for IndexFileSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IndexFileSet({} files, {} bytes)", self.manifest.files.len(), self.manifest.total_bytes())
    }
}

/// Checksums of immutable files by identity (device, inode, length,
/// mtime of an open descriptor, [`FileIdentity`]): a 30 GB graph checkpoint
/// is hashed once per process, not once per recovery, and a hard link
/// shares its entry. Never keyed on a name: a name (`seg_N.del`) can be
/// replaced by tmp + rename with the same length. Bounded: cleared when it
/// grows past `CHECKSUM_CACHE_MAX` entries.
type ChecksumKey=(u64,u64,u64,i128);
#[derive(Default)]
struct ChecksumCache {
    values:BTreeMap<ChecksumKey,u32>,
    // The whole tree, including any empty root, drops before this owner.
    memory:Option<rustic_core::admission::Reservation>,
}
static CHECKSUMS: std::sync::LazyLock<parking_lot::Mutex<ChecksumCache>> = std::sync::LazyLock::new(Default::default);
const CHECKSUM_CACHE_MAX: usize = 1 << 16;
fn checksum_key(id:&FileIdentity)->ChecksumKey{(id.dev,id.ino,id.len,id.mtime_nanos)}
fn cached_checksum(id: &FileIdentity) -> Option<u32> {
    CHECKSUMS.lock().values.get(&checksum_key(id)).copied()
}
fn cache_checksum(id: FileIdentity, crc: u32) {
    let mut cache=CHECKSUMS.lock();
    let key=checksum_key(&id);
    if let Some(value)=cache.values.get_mut(&key){*value=crc;return;}
    if cache.values.len()>=CHECKSUM_CACHE_MAX {
        drop(std::mem::take(&mut cache.values));cache.memory=None;
    }
    // Optional cache: inability to admit a node changes only caching, never
    // the checksum result. A fresh whole-tree owner avoids binding future
    // global cache growth to an expired recovery task's resident reservation.
    let Some(bytes)=(cache.values.len() as u64).checked_add(1).and_then(|n|n.checked_mul(2048))else{return};
    let Ok(memory)=rustic_core::admission::shared().reserve(bytes,"immutable-file checksum cache")else{return};
    cache.values.insert(key,crc);
    cache.memory=Some(memory);
}

/// Remember the CRC-32 of the file at `path` (verified by a writer that
/// just produced it and that alone writes that name), so a later
/// [`file_crc32`] does not re-read it.
pub fn note_checksum(path: &Path, crc: u32) {
    if let Some(id) = std::fs::metadata(path).ok().as_ref().and_then(FileIdentity::of) {
        cache_checksum(id, crc);
    }
}

/// CRC-32 of `len` bytes read from `reader` in 1 MiB steps (never the
/// whole file in memory).
fn stream_crc(mut reader: impl std::io::Read, len: u64) -> Result<u32> {
    let _memory=crate::memory::breakers().reserve_request(1<<20,"recovery file checksum buffer")?;
    let mut buf=Vec::new();buf.try_reserve_exact(1<<20).map_err(|error|EsError::new(429,"circuit_breaking_exception",format!("recovery workspace allocation failed: {error}")))?;buf.resize(1<<20,0u8);
    let (mut left, mut crc) = (len, 0xFFFF_FFFFu32);
    while left > 0 {
        let step = left.min(buf.len() as u64) as usize;
        let n = reader.read(&mut buf[..step])?;
        if n == 0 {
            return Err(EsError::new(500, "corrupt_index_exception", format!("file ended {left} bytes early")));
        }
        crc = rustic_core::codec::crc32_update(crc, &buf[..n]);
        left -= n as u64;
    }
    Ok(!crc)
}

/// Length and CRC-32 of the file at `path`, streamed (and cached by file
/// identity, see `CHECKSUMS`).
pub fn file_crc32(path: &Path) -> Result<(u64, u32)> {
    let f = std::fs::File::open(path)?;
    let meta = f.metadata()?;
    // The open descriptor's identity: what is read below.
    let id = FileIdentity::of(&meta);
    if let Some(crc) = id.as_ref().and_then(cached_checksum) {
        return Ok((meta.len(), crc));
    }
    let crc = stream_crc(f, meta.len())?;
    if let Some(id) = id {
        cache_checksum(id, crc);
    }
    Ok((meta.len(), crc))
}

impl FileSource {
    /// The CRC-32 of the whole source, computed without holding any shard
    /// lock (mapped commit files through the cache, by the identity of the
    /// file they map, taken when they were pinned: the path is never
    /// looked up again, it may name a replacement by now).
    fn checksum(&self) -> Result<u32> {
        match self {
            FileSource::MappedAt(s, _, id) => {
                let id = id.filter(|id| id.len == s.len() as u64);
                if let Some(crc) = id.as_ref().and_then(cached_checksum) {
                    return Ok(crc);
                }
                let crc = crc32(s.as_slice());
                if let Some(id) = id {
                    cache_checksum(id, crc);
                }
                Ok(crc)
            }
            FileSource::PinnedAt(pin, ..) => {
                let id = pin.identity.filter(|id| id.len == pin.data.len() as u64);
                if let Some(crc) = id.as_ref().and_then(cached_checksum) { return Ok(crc); }
                let crc = crc32(pin.data.as_slice());
                if let Some(id) = id { cache_checksum(id, crc); }
                Ok(crc)
            }
            FileSource::Mapped(s) => Ok(crc32(s.as_slice())),
            FileSource::Bytes(v) => Ok(crc32(v)),
            FileSource::Disk(p, len) => stream_crc(std::fs::File::open(p)?, *len),
        }
    }
}

impl IndexFileSet {
    /// Checksums every source here, after the caller released the shard
    /// locks it pinned them under.
    pub(crate) fn build(input_manifest: IndexManifest, sources: CapturedSources, lock: Option<RetentionLock>) -> Result<IndexFileSet> {
        let source_memory = sources.memory;
        let sources = sources.values;
        let manifest_bytes = sources.keys().try_fold(
            (sources.len() as u64).saturating_mul(std::mem::size_of::<IndexFileEntry>() as u64)
                .saturating_add(input_manifest.index.capacity() as u64)
                .saturating_add(input_manifest.uuid.capacity() as u64),
            |n, key| n.checked_add(key.len() as u64),
        ).ok_or_else(||EsError::new(429,"circuit_breaking_exception","manifest entry size overflow"))?;
        let memory = rustic_core::admission::shared().reserve(manifest_bytes, "captured manifest entries")?;
        // Buffers declared after their guards drop first on any checksum or
        // allocation refusal, as well as in the returned field order.
        let mut manifest = input_manifest;
        manifest.files = Vec::new();
        manifest.files.try_reserve_exact(sources.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        for (path, src) in &sources {
            manifest.files.push(IndexFileEntry { path: path.clone(), len: src.len(), crc32: src.checksum()? });
        }
        Ok(IndexFileSet { manifest, sources, lock, _manifest_memory: memory, _source_memory: source_memory })
    }

    /// The manifest entry of `path`.
    pub fn entry(&self, path: &str) -> Option<&IndexFileEntry> {
        self.manifest.files.iter().find(|f| f.path == path)
    }

    /// One whole file.
    pub fn read(&self, path: &str) -> Result<rustic_core::admission::Allocation<Vec<u8>>> {
        self.read_range(path, 0, u64::MAX)
    }

    /// Part of a file (chunked network transfers).
    pub fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<rustic_core::admission::Allocation<Vec<u8>>> {
        self.sources.get(path).ok_or_else(|| EsError::resource_not_found(format!("no file [{path}] in the file set")))?.read_range(offset, len)
    }

    /// The retention lock held on the source (history `> committed_seq_no`).
    pub fn retention_lock(&self) -> Option<&RetentionLock> {
        self.lock.as_ref()
    }

    /// Give up the retention lock early (e.g. once the target took over).
    pub fn release_retention(&mut self) {
        self.lock = None;
    }

    /// Write a complete copy under `dest` (created; must not hold another
    /// index), every file fsynced, manifest last. `dest` can then be passed
    /// to [`Node::adopt_index`](crate::node::Node::adopt_index).
    ///
    /// With `hard_link`, immutable commit files are hard-linked from the
    /// source instead of copied (same filesystem only; each link is verified
    /// against the manifest and anything that cannot be linked falls back to
    /// a copy). Translog / sidecar files are always copied.
    pub fn write_to_dir(&self, dest: &Path, hard_link: bool) -> Result<()> {
        for f in &self.manifest.files {
            if hard_link {
                if let Some(FileSource::PinnedAt(pin, src, _)) = self.sources.get(&f.path) {
                    match link_file(dest, f, pin.data.as_slice(), src) {
                        Ok(()) => continue,
                        Err(error) if error.kind == "circuit_breaking_exception" => return Err(error),
                        Err(_) => {},
                    }
                }
                if let Some(FileSource::MappedAt(slice, src, _)) = self.sources.get(&f.path) {
                    match link_file(dest,f,slice.as_slice(),src) {
                        Ok(())=>continue,
                        Err(error) if error.kind=="circuit_breaking_exception"=>return Err(error),
                        Err(_)=>{},
                    }
                }
            }
            let source=self.sources.get(&f.path).ok_or_else(||EsError::resource_not_found(format!("no file [{}] in the file set",f.path)))?;
            match source {
                FileSource::Mapped(slice)|FileSource::MappedAt(slice,..)=>receive_file_from(dest,f,std::io::Cursor::new(slice.as_slice()))?,
                FileSource::PinnedAt(pin,..)=>receive_file_from(dest,f,std::io::Cursor::new(pin.data.as_slice()))?,
                FileSource::Bytes(bytes)=>receive_file_from(dest,f,std::io::Cursor::new(bytes.as_slice()))?,
                FileSource::Disk(path,len)=>{
                    use std::io::Read;
                    receive_file_from(dest,f,std::fs::File::open(path)?.take(*len))?;
                }
            }
        }
        finalize_received(dest, &self.manifest)
    }
}

struct AdmittedPath {value:PathBuf,_memory:rustic_core::admission::Reservation}
impl std::ops::Deref for AdmittedPath {type Target=Path;fn deref(&self)->&Path{&self.value}}
impl AsRef<Path> for AdmittedPath {fn as_ref(&self)->&Path{&self.value}}
fn safe_join(dest: &Path, rel: &str) -> Result<AdmittedPath> {
    if rel.is_empty() || rel.starts_with('/') || rel.split('/').any(|s| s == ".." || s == "." || s.is_empty()) {
        return Err(EsError::illegal_argument(format!("bad file path [{rel}] in transfer")));
    }
    let size=dest.as_os_str().as_encoded_bytes().len().checked_add(rel.len()).and_then(|n|n.checked_add(1)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","recovery path overflow"))?;
    let memory=rustic_core::admission::shared().reserve(size as u64,"recovery destination path")?;
    let mut value=PathBuf::new();value.try_reserve_exact(size).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;value.push(dest);value.push(rel);
    Ok(AdmittedPath{value,_memory:memory})
}

/// Receiver side: verify one file against its manifest entry and write it
/// (atomically, fsynced) under `dest`.
pub fn receive_file(dest: &Path, entry: &IndexFileEntry, data: &[u8]) -> Result<()> {
    if data.len() as u64 != entry.len || crc32(data) != entry.crc32 {
        return Err(EsError::new(500, "corrupt_index_exception", format!("checksum failed for [{}]: length {} (expected {}), crc32 {:08x} (expected {:08x})", entry.path, data.len(), entry.len, crc32(data), entry.crc32)));
    }
    let p = safe_join(dest, &entry.path)?;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    super::write_durable(&p, data)?;
    Ok(())
}

/// Whether the file at `path` holds exactly `pinned`, compared in 1 MiB
/// steps (never the whole file in memory).
fn same_bytes(path: &Path, pinned: &[u8]) -> Result<bool> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return Ok(false) };
    if f.metadata().map(|m| m.len()).ok() != Some(pinned.len() as u64) {
        return Ok(false);
    }
    let _memory=crate::memory::breakers().reserve_request(1<<20,"recovery pinned-file comparison buffer")?;
    let mut buf=Vec::new();buf.try_reserve_exact(1<<20).map_err(|error|EsError::new(429,"circuit_breaking_exception",format!("recovery workspace allocation failed: {error}")))?;buf.resize(1<<20,0u8);
    let mut at = 0usize;
    while at < pinned.len() {
        let want = (pinned.len() - at).min(buf.len());
        if f.read_exact(&mut buf[..want]).is_err() || buf[..want] != pinned[at..at + want] {
            return Ok(false);
        }
        at += want;
    }
    Ok(true)
}

/// Hard-link `src` as `entry` under `dest` and verify it holds the pinned
/// bytes (compared byte for byte: the name may be a newer file with the
/// same length, e.g. a `.del` replaced since the pin, whose cached
/// checksum would not tell); on any failure the link is removed.
fn link_file(dest: &Path, entry: &IndexFileEntry, pinned: &[u8], src: &Path) -> Result<()> {
    let p = safe_join(dest, &entry.path)?;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let _ = std::fs::remove_file(&p);
    std::fs::hard_link(src, &p)?;
    let checked = if pinned.len() as u64 == entry.len {same_bytes(&p,pinned)}else{Ok(false)};
    let ok=match checked{Ok(ok)=>ok,Err(error)=>{let _=std::fs::remove_file(&p);return Err(error);}};
    if !ok {
        let _ = std::fs::remove_file(&p);
        return Err(EsError::internal("hard-linked file does not match the pinned bytes"));
    }
    super::sync_dir(p.parent());
    Ok(())
}

/// Streaming [`receive_file`]: copy `reader` to a temp file in chunks
/// (never holding the whole file), checking length and CRC-32 against
/// `entry` as it goes, then fsync and rename into place. On any mismatch or
/// I/O error nothing is left under the final name.
pub fn receive_file_from(dest: &Path, entry: &IndexFileEntry, mut reader: impl std::io::Read) -> Result<()> {
    use std::io::Write;
    let p = safe_join(dest, &entry.path)?;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let _temp_memory=rustic_core::admission::shared().reserve((p.as_os_str().as_encoded_bytes().len() as u64).saturating_mul(3).saturating_add(32),"recovery temporary path")?;
    let tmp = p.with_extension("tmp~");
    let res = (|| -> Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        let _memory=crate::memory::breakers().reserve_request(256*1024,"recovery file receive buffer")?;
        let mut buf=Vec::new();buf.try_reserve_exact(256*1024).map_err(|error|EsError::new(429,"circuit_breaking_exception",format!("recovery workspace allocation failed: {error}")))?;buf.resize(256*1024,0u8);
        let (mut total, mut crc) = (0u64, 0xFFFF_FFFFu32);
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n as u64;
            if total > entry.len {
                break;
            }
            crc = rustic_core::codec::crc32_update(crc, &buf[..n]);
            f.write_all(&buf[..n])?;
        }
        let got = !crc;
        if total != entry.len || got != entry.crc32 {
            return Err(EsError::new(500, "corrupt_index_exception", format!("checksum failed for [{}]: length {}{} (expected {}), crc32 {:08x} (expected {:08x})", entry.path, total, if total > entry.len { "+" } else { "" }, entry.len, got, entry.crc32)));
        }
        f.sync_all()?;
        Ok(())
    })();
    match res {
        Ok(()) => {
            std::fs::rename(&tmp, &p)?;
            super::sync_dir(p.parent());
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Receiver side: verify every file is present and intact, then write the
/// manifest (marks the copy complete).
pub fn finalize_received(dest: &Path, manifest: &IndexManifest) -> Result<()> {
    verify_index_dir(dest, manifest)?;
    struct Count(u64);
    impl std::io::Write for Count {fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{self.0=self.0.checked_add(bytes.len() as u64).ok_or_else(||std::io::Error::other("manifest size overflow"))?;Ok(bytes.len())}fn flush(&mut self)->std::io::Result<()>{Ok(())}}
    let mut count=Count(0);serde_json::to_writer_pretty(&mut count,manifest)?;
    let size=usize::try_from(count.0).map_err(|_|EsError::new(429,"circuit_breaking_exception","manifest size overflow"))?;
    let _output=rustic_core::admission::shared().reserve(count.0,"received manifest encoded bytes")?;
    let mut bytes=Vec::new();bytes.try_reserve_exact(size).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;bytes.resize(size,0);
    let mut cursor=std::io::Cursor::new(bytes.as_mut_slice());serde_json::to_writer_pretty(&mut cursor,manifest)?;
    if cursor.position()!=count.0{return Err(EsError::internal("manifest encoding changed size"));}
    let path=safe_join(dest,MANIFEST_FILE)?;
    let _temp=rustic_core::admission::shared().reserve((path.as_os_str().as_encoded_bytes().len() as u64).saturating_mul(3).saturating_add(32),"durable manifest temporary path")?;
    super::write_durable(&path, &bytes)?;
    Ok(())
}

/// Check every manifest file under `dir` (length + CRC-32, streamed).
pub fn verify_index_dir(dir: &Path, manifest: &IndexManifest) -> Result<()> {
    for f in &manifest.files {
        let p = safe_join(dir, &f.path)?;
        let (len, crc) = file_crc32(&p).map_err(|e| if e.kind=="circuit_breaking_exception" {e} else {EsError::new(500, "corrupt_index_exception", format!("missing file [{}]: {}", f.path, e.reason))})?;
        if len != f.len || crc != f.crc32 {
            return Err(EsError::new(500, "corrupt_index_exception", format!("checksum failed for [{}]", f.path)));
        }
    }
    Ok(())
}

/// Check that every manifest file is present under `dir` with its length:
/// for a peer recovery's staged copy, whose files were each verified
/// (chunk checksums plus the whole-file CRC-32) before they were renamed
/// into place, so re-reading tens of GB is not needed.
pub fn verify_index_dir_lengths(dir: &Path, manifest: &IndexManifest) -> Result<()> {
    for f in &manifest.files {
        let p = safe_join(dir, &f.path)?;
        let len = std::fs::metadata(&p).map_err(|e| EsError::new(500, "corrupt_index_exception", format!("missing file [{}]: {e}", f.path)))?.len();
        if len != f.len {
            return Err(EsError::new(500, "corrupt_index_exception", format!("length of [{}] is [{len}], expected [{}]", f.path, f.len)));
        }
    }
    Ok(())
}

/// The transfer manifest of a received copy, if it has one.
pub fn read_manifest(dir: &Path) -> Result<Option<rustic_core::admission::Allocation<IndexManifest>>> {
    let path=safe_join(dir,MANIFEST_FILE)?;
    let file=match std::fs::File::open(&path) {
        Ok(file)=>file,
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(None),
        Err(error)=>return Err(error.into()),
    };
    let bytes=read_metadata_descriptor(file)?;
    // Accepted schema has only strings and file-entry structs. Eight bytes
    // per input byte cover geometrically grown strings/entry slots; a second
    // such allowance covers decoder scratch while retained values are built.
    let memory=rustic_core::admission::shared().reserve((bytes.len() as u64).saturating_mul(16).saturating_add(4096),"transfer manifest decoded metadata")?;
    let manifest=serde_json::from_slice(&bytes)?;
    Ok(Some(rustic_core::admission::Allocation::new(manifest,memory)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(data: &[u8]) -> IndexFileEntry {
        IndexFileEntry { path: "a/b.bin".into(), len: data.len() as u64, crc32: crc32(data) }
    }

    #[test]
    fn streaming_receive_verifies_and_cleans_up() {
        let tmp = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..700_000u32).map(|i| (i % 251) as u8).collect();
        let e = entry(&data);
        receive_file_from(tmp.path(), &e, std::io::Cursor::new(&data)).unwrap();
        assert_eq!(std::fs::read(tmp.path().join("a/b.bin")).unwrap(), data);
        // Corrupt / short / long streams are rejected and leave nothing behind.
        let other = tempfile::tempdir().unwrap();
        let mut bad = data.clone();
        bad[10] ^= 1;
        for stream in [bad, data[..1000].to_vec(), [data.clone(), vec![0]].concat()] {
            let err = receive_file_from(other.path(), &e, std::io::Cursor::new(stream)).unwrap_err();
            assert_eq!(err.kind, "corrupt_index_exception");
        }
        assert!(!other.path().join("a/b.bin").exists() && !other.path().join("a/b.tmp~").exists());
    }

    /// A commit replaces `seg_N.del` (same name, same length, tmp +
    /// rename) between the pin and the checksums: each pinned mapping gets
    /// its own bytes' checksum, the cache is not poisoned for the new
    /// file, and a hard link to the name (now the new file) is refused.
    #[test]
    fn a_replaced_del_file_between_pin_and_build_keeps_each_checksum() {
        use rustic_core::codec::directory::{Directory, MmapDirectory};
        let tmp = tempfile::tempdir().unwrap();
        let core = tmp.path().join("core-1");
        let dir = MmapDirectory::open(&core).unwrap();
        let (old, new) = (vec![0x0fu8; 4096], vec![0xf0u8; 4096]);
        dir.write("seg_00000001.del", &old).unwrap();
        let path = core.join("seg_00000001.del");
        let manifest = || IndexManifest { index: "i".into(), uuid: "u".into(), files: Vec::new(), committed_seq_no: 0, local_checkpoint: 0, max_seq_no: 0, primary_term: 1, includes_translog: false };
        let pin = || {
            let (slice, id) = dir.open_read_identified("seg_00000001.del").unwrap();
            BTreeMap::from([("core-1/seg_00000001.del".to_string(), FileSource::MappedAt(slice, path.clone(), id))])
        };
        let first = pin();
        // The commit lands after the pin, before the checksums.
        dir.write("seg_00000001.del", &new).unwrap();
        let first = IndexFileSet::build(manifest(), CapturedSources::from_fixture(first), None).unwrap();
        assert_eq!(first.manifest.files[0].crc32, crc32(&old), "the pinned (old) bytes' checksum");
        let second = IndexFileSet::build(manifest(), CapturedSources::from_fixture(pin()), None).unwrap();
        assert_eq!(second.manifest.files[0].crc32, crc32(&new), "the new file is not served the old file's cached checksum");
        assert_eq!(file_crc32(&path).unwrap(), (new.len() as u64, crc32(&new)));
        // A hard-linked copy of the first set would link the new file under
        // the old manifest entry: refused, the pinned bytes are copied.
        let dest = tmp.path().join("copy");
        first.write_to_dir(&dest, true).unwrap();
        assert_eq!(std::fs::read(dest.join("core-1/seg_00000001.del")).unwrap(), old);
        let second_dest = tmp.path().join("copy2");
        second.write_to_dir(&second_dest, true).unwrap();
        assert_eq!(std::fs::read(second_dest.join("core-1/seg_00000001.del")).unwrap(), new);
    }
}

#[cfg(test)]
mod optional_metadata_tests {
    use super::*;
    #[test]
    fn absent_metadata_is_optional_but_unreadable_metadata_fails() {
        let dir=tempfile::tempdir().unwrap();
        assert!(read_optional_metadata_file(&dir.path().join("absent")).unwrap().is_none());
        assert!(read_optional_metadata_file(dir.path()).is_err());
        let path=dir.path().join("terms");
        std::fs::write(&path,b"term history").unwrap();
        assert_eq!(&**read_optional_metadata_file(&path).unwrap().unwrap(),b"term history");
    }
}
