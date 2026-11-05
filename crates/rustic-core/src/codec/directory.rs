//! Storage abstraction: a `Directory` of named files, read as `FileSlice`s.
//!
//! `RamDirectory` keeps files in memory (tests, ephemeral indexes).
//! `MmapDirectory` maps files from disk; reads are zero-copy and the OS
//! page cache does the caching, which is how Lucene's `MMapDirectory` works.
//! With mapping disallowed (ES `node.store.allow_mmap: false`, see
//! [`set_mmap_allowed`]) it reads each file into memory instead and maps
//! nothing.

use crate::error::{Result, RusticError};
use parking_lot::RwLock;
use std::fmt;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Bytes = Arc<dyn Deref<Target = [u8]> + Send + Sync>;

/// ES `node.store.allow_mmap` for this process (default true): whether a
/// [`MmapDirectory`] opened from now on maps its files.
static MMAP_ALLOWED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Set from the node's `node.store.allow_mmap` at start, before any index
/// opens. With `false`, no store file is memory-mapped.
pub fn set_mmap_allowed(allowed: bool) {
    MMAP_ALLOWED.store(allowed, std::sync::atomic::Ordering::Relaxed);
}

/// See [`set_mmap_allowed`].
pub fn mmap_allowed() -> bool {
    MMAP_ALLOWED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Why a store file could not be charged to the node's memory accounting.
#[derive(Debug, Clone)]
pub struct StoreFileRefusal {
    /// The limit that refused it, in bytes.
    pub limit: u64,
    /// The accounting's own message (ES's breaker text on a node).
    pub reason: String,
}

/// Charges `bytes` of store file `file` read into memory to the node's
/// memory accounting. The returned guard is held with the buffer and
/// releases the charge when dropped.
pub type StoreFileAdmission = fn(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal>;

static STORE_FILE_ADMISSION: std::sync::OnceLock<StoreFileAdmission> = std::sync::OnceLock::new();

/// Installs the node's store file accounting (once, at start, before any
/// index opens). Without one, the core's shared admission budget is
/// charged.
pub fn set_store_file_admission(admission: StoreFileAdmission) {
    let _ = STORE_FILE_ADMISSION.set(admission);
}

fn store_file_admission() -> StoreFileAdmission {
    *STORE_FILE_ADMISSION.get_or_init(|| shared_admission)
}

/// The default [`StoreFileAdmission`]: the core's shared admission budget.
fn shared_admission(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal> {
    let budget = crate::admission::shared();
    budget
        .reserve(bytes, &format!("store file [{file}]"))
        .map(|r| Box::new(r) as Box<dyn Send>)
        .map_err(|e| StoreFileRefusal { limit: budget.limit(), reason: e.to_string() })
}

/// A store file's bytes held in memory with their accounting charge, which
/// is released when the last slice of the file drops.
struct HeldFile {
    bytes: Vec<u8>,
    /// `Mutex`: makes the guard (only `Send`) shareable; never locked.
    _charge: std::sync::Mutex<Box<dyn Send>>,
}

impl Deref for HeldFile {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

/// The message of a refused store file read: ES-style, naming the file,
/// its size and the limit, and the way out. Ends with the core's
/// "needs N bytes, limit L" so the node reports `bytes_wanted` /
/// `bytes_limit`.
fn refused(name: &str, bytes: u64, refusal: &StoreFileRefusal) -> RusticError {
    RusticError::Memory(format!(
        "cannot read store file [{name}] of [{bytes}] bytes into memory: [node.store.allow_mmap] is false, so store files are held in memory and \
         counted against the node's memory, and this one would go over the limit of [{limit}] bytes ({reason}). Set [node.store.allow_mmap] to \
         [true] (with [vm.max_map_count] at least [262144]) or give the node more memory; needs {bytes} bytes, limit {limit}",
        limit = refusal.limit,
        reason = refusal.reason,
    ))
}

/// The whole of an open file, read into memory: the store's read path when
/// mapping is disallowed. The bytes are charged to `admission` before they
/// are allocated and for as long as any slice of them lives; a refusal is
/// an error naming the file, its size and the limit. The bytes are owned,
/// so a later replace or delete of the name cannot change them (as a
/// mapping pins its inode).
pub(crate) fn read_whole(file: &std::fs::File, name: &str, len: u64, admission: StoreFileAdmission) -> Result<FileSlice> {
    let charge = admission(len, name).map_err(|refusal| refused(name, len, &refusal))?;
    let size = usize::try_from(len).map_err(|_| RusticError::Memory(format!("store file [{name}] of [{len}] bytes does not fit in memory")))?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(size).map_err(|e| RusticError::Memory(format!("store file [{name}] of [{len}] bytes: {e}")))?;
    buf.resize(size, 0);
    let mut done = 0;
    while done < size {
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, &mut buf[done..], done as u64)?;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, &mut buf[done..], done as u64)?;
        if n == 0 {
            return Err(RusticError::Corrupted(format!("store file [{name}] shrank while read: {done} of {size} bytes")));
        }
        done += n;
    }
    // The bytes now live in memory; keeping them in the page cache as well
    // would hold the file twice.
    #[cfg(target_os = "linux")]
    // SAFETY: advice on a valid open descriptor.
    unsafe {
        libc::posix_fadvise(std::os::unix::io::AsRawFd::as_raw_fd(file), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
    Ok(FileSlice::new(Arc::new(HeldFile { bytes: buf, _charge: std::sync::Mutex::new(charge) })))
}

/// [`read_whole`] of an open file with the process's store file accounting
/// (for stores other than [`MmapDirectory`] that would otherwise map).
pub(crate) fn read_unmapped(file: &std::fs::File, name: &str) -> Result<FileSlice> {
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(FileSlice::empty());
    }
    read_whole(file, name, len, store_file_admission())
}

/// A cheaply clonable view over (part of) a file's bytes.
#[derive(Clone)]
pub struct FileSlice {
    data: Bytes,
    start: usize,
    end: usize,
}

impl FileSlice {
    pub fn new(data: Bytes) -> Self {
        let end = data.len();
        Self { data, start: 0, end }
    }

    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::new(Arc::new(v))
    }

    pub fn empty() -> Self {
        Self::from_vec(Vec::new())
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &(**self.data)[self.start..self.end]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A sub-view. Panics if the range is out of bounds.
    pub fn slice(&self, range: Range<usize>) -> FileSlice {
        assert!(range.end <= self.len(), "FileSlice::slice out of bounds");
        FileSlice {
            data: self.data.clone(),
            start: self.start + range.start,
            end: self.start + range.end,
        }
    }
}

impl fmt::Debug for FileSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FileSlice({} bytes)", self.len())
    }
}

/// Identity of an on-disk file: device, inode, length and mtime (ns) of
/// the open file a mapping was made from. Files are replaced by tmp +
/// rename, so a name can stand for another inode later; a cache keyed by
/// file contents keys on this, taken from the mapped descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileIdentity {
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_nanos: i128,
}

impl FileIdentity {
    /// The identity of an open file's metadata.
    #[cfg(unix)]
    pub fn of(meta: &std::fs::Metadata) -> Option<FileIdentity> {
        use std::os::unix::fs::MetadataExt;
        let mtime_nanos = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(-1, |d| d.as_nanos() as i128);
        Some(FileIdentity { dev: meta.dev(), ino: meta.ino(), len: meta.len(), mtime_nanos })
    }

    /// `None`: no device / inode numbers on this platform.
    #[cfg(not(unix))]
    pub fn of(_meta: &std::fs::Metadata) -> Option<FileIdentity> {
        None
    }
}

/// A filename list whose strings and replacement array remain admitted.
#[derive(Debug)]
pub struct AdmittedNames { names: Vec<String>, memory: crate::admission::Reservation }
impl std::ops::Deref for AdmittedNames { type Target = [String]; fn deref(&self) -> &[String] { &self.names } }
impl AdmittedNames {
    pub fn new() -> Result<Self> { Self::with_budget(&crate::admission::shared()) }
    fn with_budget(budget:&Arc<crate::admission::Budget>)->Result<Self> { Ok(Self { names: Vec::new(), memory: budget.reserve(0, "directory filename list")? }) }
    pub fn push(&mut self, name: &str) -> Result<()> {
        if self.names.len() == self.names.capacity() {
            let old = self.names.capacity();
            let cap = old.max(4).checked_mul(2).ok_or_else(|| RusticError::Memory("filename list capacity overflow".into()))?;
            let bytes = cap.checked_mul(std::mem::size_of::<String>()).ok_or_else(|| RusticError::Memory("filename list capacity overflow".into()))?;
            self.memory.grow(bytes as u64, "directory filename replacement array")?;
            self.names.try_reserve_exact(cap-self.names.len()).map_err(|e| RusticError::Memory(e.to_string()))?;
            self.memory.shrink((old * std::mem::size_of::<String>()) as u64);
        }
        self.memory.grow(name.len() as u64, "directory filename bytes")?;
        let mut text=String::new();text.try_reserve_exact(name.len()).map_err(|e|RusticError::Memory(e.to_string()))?;text.push_str(name);
        self.names.push(text); Ok(())
    }
    pub fn sort(&mut self) { self.names.sort_unstable(); }
}

/// A named-file store.
pub trait Directory: Send + Sync + fmt::Debug {
    fn open_read(&self, name: &str) -> Result<FileSlice>;
    /// [`Directory::open_read`] plus the identity of the file the slice
    /// was read or mapped from (`None`: not a plain on-disk file, or not
    /// known). The default knows none.
    fn open_read_identified(&self, name: &str) -> Result<(FileSlice, Option<FileIdentity>)> {
        Ok((self.open_read(name)?, None))
    }
    /// Atomically replace `name` with `data`.
    fn write(&self, name: &str, data: &[u8]) -> Result<()>;
    /// Atomically replace `name` with the concatenation of `parts` (lets a
    /// file be streamed from its pieces without assembling one buffer).
    fn write_parts(&self, name: &str, parts: &[&[u8]]) -> Result<()> {
        self.write(name, &parts.concat())
    }
    fn exists(&self, name: &str) -> bool;
    fn delete(&self, name: &str) -> Result<()>;
    fn list(&self) -> Result<Vec<String>>;
    /// Owned listing used by recovery and cleanup. Include unfinished writes
    /// when requested; callers retain this owner through all borrowed names.
    fn list_admitted(&self, temporary: bool) -> Result<AdmittedNames>;
    /// Temporary files of writes not finished (`.tmp`), which
    /// [`Directory::list`] leaves out where it hides them. Only safe to
    /// delete while nothing writes to the directory (an index opening).
    fn list_temporary(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    /// Retire this handle (its index is closed or failed and another may
    /// open the same files): once it returns, no write, rename or delete
    /// through it (detached seal / merge jobs still holding it included)
    /// changes the directory any more; each fails instead. Files already
    /// opened stay readable, but new opens fail too: the retired pathname
    /// may belong to another store. Waits for a rename in progress. The
    /// default does nothing.
    fn close(&self) {}
    /// Warm the OS cache for a file about to be read in bulk (a merge
    /// input). Advisory: the default does nothing, errors are ignored.
    fn prefetch(&self, _name: &str) {}
    /// Open `name` for streaming writes; the file appears atomically (and
    /// durably, where the directory syncs) on [`OutputFile::finish`]. The
    /// default buffers in memory and calls [`Directory::write`].
    fn create_output<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        Ok(Box::new(BufferedOutput { dir: self, name: name.to_string(), buf: Vec::new() }))
    }
    /// [`Directory::write_parts`] of a segment file that only a later
    /// commit point references (Lucene's refresh): the file appears
    /// atomically, but is made durable by [`Directory::sync_unsynced`]
    /// before that commit, not now. The default syncs now.
    fn write_parts_unsynced(&self, name: &str, parts: &[&[u8]]) -> Result<()> {
        self.write_parts(name, parts)
    }
    /// [`Directory::create_output`] of a segment file that only a later
    /// commit point references (a merge's output), synced like
    /// [`Directory::write_parts_unsynced`]. The default syncs on finish.
    fn create_output_unsynced<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        self.create_output(name)
    }
    /// [`Directory::create_output`] of a scratch file (deleted once its
    /// writer is done, never referenced by a commit): never synced where
    /// the directory can skip it.
    fn create_scratch_output<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        self.create_output(name)
    }
    /// Make durable every file written unsynced so far, and the directory
    /// entries naming them: called before a commit point that may
    /// reference them is written. Files deleted meanwhile are skipped.
    fn sync_unsynced(&self) -> Result<()> {
        Ok(())
    }
}

/// A file being written sequentially (see [`Directory::create_output`]).
pub trait OutputFile: Send {
    /// This output is a recovery checkpoint, not a query input. On Linux,
    /// streaming implementations can retire synced output from page cache
    /// instead of displacing the mapped source rows used by live indexing.
    /// Advisory only; implementations without a file descriptor do nothing.
    fn cold_checkpoint(&mut self) {}
    fn write_all(&mut self, data: &[u8]) -> Result<()>;
    /// Make the file visible under its name (atomically).
    fn finish(self: Box<Self>) -> Result<()>;
}

struct BufferedOutput<'a, D: Directory + ?Sized> {
    dir: &'a D,
    name: String,
    buf: Vec<u8>,
}

impl<D: Directory + ?Sized> OutputFile for BufferedOutput<'_, D> {
    fn write_all(&mut self, data: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(data);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.dir.write(&self.name, &self.buf)
    }
}

/// Streaming output of [`MmapDirectory`]: a buffered temp file, synced and
/// renamed into place on `finish`.
struct MmapOutput {
    /// `None` once finished.
    file: Option<std::io::BufWriter<std::fs::File>>,
    tmp: PathBuf,
    path: PathBuf,
    /// The directory handle's [`Directory::close`] state.
    closed: Arc<RwLock<bool>>,
    cold: bool,
    /// Bytes handed to `write_all`.
    written: u64,
    /// Bytes retired from the cache (cold) or queued for writeback (not cold).
    released: u64,
    /// When the file is made durable.
    sync: OutputSync,
}

/// When a [`MmapOutput`] is made durable.
enum OutputSync {
    /// On `finish` (file and directory entry).
    Now,
    /// By the next [`Directory::sync_unsynced`]: `finish` records the name.
    Deferred(Arc<parking_lot::Mutex<std::collections::HashSet<String>>>),
    /// Never (a scratch file).
    Never,
}

// Large enough to amortize writeback, small compared with a graph checkpoint.
const CHECKPOINT_WRITE_WINDOW: u64 = 32 << 20;

#[cfg(target_os = "linux")]
fn release_file_cache(file: &std::fs::File, offset: u64, length: u64) {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(length)) = (libc::off_t::try_from(offset), libc::off_t::try_from(length)) else { return };
    // SAFETY: the descriptor remains open for this call. This only advises
    // cache replacement; it neither unmaps readers nor changes file bytes.
    // Filesystems may ignore advice. Failure cannot weaken durability.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, length, libc::POSIX_FADV_DONTNEED); }
}

/// Written bytes between [`start_writeback`] calls of a streaming output.
#[cfg(target_os = "linux")]
const WRITEBACK_WINDOW: u64 = 8 << 20;

/// Start (not wait for) writeback of a just-written range. Advisory: the
/// final `sync_all` still makes the file durable.
#[cfg(target_os = "linux")]
fn start_writeback(file: &std::fs::File, offset: u64, length: u64) {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(length)) = (libc::off64_t::try_from(offset), libc::off64_t::try_from(length)) else { return };
    // SAFETY: the descriptor remains open for this call, which only queues
    // dirty pages of the range for writeback.
    unsafe { libc::sync_file_range(file.as_raw_fd(), offset, length, libc::SYNC_FILE_RANGE_WRITE); }
}

fn closed_error() -> RusticError {
    RusticError::Io(std::io::Error::other("the directory handle is closed (its index was closed or failed)"))
}

impl OutputFile for MmapOutput {
    fn cold_checkpoint(&mut self) {
        self.cold = cfg!(target_os = "linux");
    }

    fn write_all(&mut self, data: &[u8]) -> Result<()> {
        use std::io::Write;
        let file = self.file.as_mut().expect("output finished");
        if !self.cold {
            file.write_all(data)?;
            self.written += data.len() as u64;
            // Start writeback of each full window as it is written (RocksDB's
            // `bytes_per_sync`), keeping the pages cached: a large output (a
            // merge) then reaches the disk at its own, throttled, pace rather
            // than as gigabytes of dirty pages flushed in bursts by the kernel
            // or the final sync, which stalled search page faults behind them.
            #[cfg(target_os = "linux")]
            if self.written - self.released >= WRITEBACK_WINDOW {
                file.flush()?;
                start_writeback(file.get_ref(), self.released, self.written - self.released);
                self.released = self.written;
            }
            return Ok(());
        }
        let mut remaining = data;
        while !remaining.is_empty() {
            let available = (CHECKPOINT_WRITE_WINDOW - (self.written - self.released)) as usize;
            let n = remaining.len().min(available);
            file.write_all(&remaining[..n])?;
            self.written += n as u64;
            remaining = &remaining[n..];
            if self.written - self.released == CHECKPOINT_WRITE_WINDOW {
                // Dirty pages cannot reliably be retired. Keep the normal
                // final fsync/rename/directory-fsync publication protocol too.
                file.flush()?;
                file.get_ref().sync_data()?;
                #[cfg(target_os = "linux")]
                release_file_cache(file.get_ref(), self.released, CHECKPOINT_WRITE_WINDOW);
                self.released = self.written;
            }
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        let f = self.file.take().expect("output finished").into_inner().map_err(|e| e.into_error())?;
        if matches!(self.sync, OutputSync::Now) {
            f.sync_all()?;
        }
        #[cfg(target_os = "linux")]
        if self.cold { release_file_cache(&f, 0, 0); }
        drop(f);
        // Held across the rename: a close waits for it, and no rename
        // starts after one.
        let closed = self.closed.clone();
        let closed = closed.read();
        if *closed {
            return Err(closed_error());
        }
        std::fs::rename(&self.tmp, &self.path)?;
        match &self.sync {
            // The file contents and the directory entry must both survive restart.
            OutputSync::Now => if let Some(parent) = self.path.parent() {
                crate::platform::sync_dir(parent)?;
            },
            OutputSync::Deferred(unsynced) => if let Some(name) = self.path.file_name().and_then(|n| n.to_str()) {
                unsynced.lock().insert(name.to_string());
            },
            OutputSync::Never => {}
        }
        Ok(())
    }
}

impl Drop for MmapOutput {
    fn drop(&mut self) {
        // Abandoned before `finish`: remove the partial temp file.
        self.file.take();
        let _ = std::fs::remove_file(&self.tmp);
    }
}

struct RamFile {
    bytes: Vec<u8>,
    _memory: crate::admission::Reservation,
}
impl Deref for RamFile { type Target = [u8]; fn deref(&self) -> &[u8] { &self.bytes } }

/// In-memory directory.
#[derive(Default)]
pub struct RamDirectory {
    files: RwLock<std::collections::BTreeMap<String, Arc<RamFile>>>,
}

impl RamDirectory {
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Debug for RamDirectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RamDirectory({} files)", self.files.read().len())
    }
}

impl Directory for RamDirectory {
    fn open_read(&self, name: &str) -> Result<FileSlice> {
        let files = self.files.read();
        let data = files
            .get(name)
            .ok_or_else(|| RusticError::IndexNotFound(format!("file {name}")))?;
        Ok(FileSlice::new(data.clone()))
    }

    fn write(&self, name: &str, data: &[u8]) -> Result<()> {
        let memory = crate::admission::shared().reserve((data.len() as u64).saturating_add(name.len() as u64).saturating_add(2048), "RAM directory file and registry")?;
        self.files
            .write()
            .insert(name.to_string(), Arc::new(RamFile { bytes: data.to_vec(), _memory: memory }));
        Ok(())
    }

    fn exists(&self, name: &str) -> bool {
        self.files.read().contains_key(name)
    }

    fn delete(&self, name: &str) -> Result<()> {
        let mut files = self.files.write();
        let removed = files.remove(name);
        if files.is_empty() { *files = std::collections::BTreeMap::new(); }
        drop(removed);
        Ok(())
    }

    fn list_admitted(&self, _temporary: bool) -> Result<AdmittedNames> {
        let mut names=AdmittedNames::new()?;
        for name in self.files.read().keys() { names.push(name)?; }
        names.sort(); Ok(names)
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self.files.read().keys().cloned().collect();
        names.sort();
        Ok(names)
    }
}

/// Memory-mapped on-disk directory.
#[derive(Debug)]
pub struct MmapDirectory {
    root: PathBuf,
    /// [`Directory::close`]d: no more writes through this handle.
    closed: Arc<RwLock<bool>>,
    /// Files written unsynced, not yet [`Directory::sync_unsynced`].
    unsynced: Arc<parking_lot::Mutex<std::collections::HashSet<String>>>,
    /// `None`: map files on read (the default). `Some`: read them into
    /// memory, charged to this accounting ([`mmap_allowed`]).
    in_memory: Option<StoreFileAdmission>,
}

impl MmapDirectory {
    /// Opens `root`, mapping files unless [`set_mmap_allowed`]`(false)`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(root, mmap_allowed())
    }

    /// [`MmapDirectory::open`] with mapping on or off for this directory
    /// (off: files are charged to the node's store file accounting).
    pub fn open_with(root: impl AsRef<Path>, mmap: bool) -> Result<Self> {
        Self::open_in(root, if mmap { None } else { Some(store_file_admission()) })
    }

    /// A directory that reads files into memory, charged to `admission`.
    pub fn open_in_memory(root: impl AsRef<Path>, admission: StoreFileAdmission) -> Result<Self> {
        Self::open_in(root, Some(admission))
    }

    fn open_in(root: impl AsRef<Path>, in_memory: Option<StoreFileAdmission>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root, closed: Arc::new(RwLock::new(false)), unsynced: Default::default(), in_memory })
    }

    fn write_parts_open(&self, name: &str, parts: &[&[u8]], sync: bool) -> Result<()> {
        let final_path = self.file_path(name);
        let tmp_path = self.file_path(&format!("{name}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path)?;
            for p in parts {
                f.write_all(p)?;
            }
            if sync {
                f.sync_all()?;
            }
        }
        std::fs::rename(&tmp_path, &final_path)?;
        if !sync {
            self.unsynced.lock().insert(name.to_string());
        }
        Ok(())
    }

    fn write_open(&self, name: &str, data: &[u8]) -> Result<()> {
        let final_path = self.file_path(name);
        let tmp_path = self.file_path(&format!("{name}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path)?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Run a change to the directory unless the handle is closed (held
    /// against a concurrent close for its duration).
    fn changing<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let closed = self.closed.read();
        if *closed {
            return Err(closed_error());
        }
        f()
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    fn file_path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn create_streaming_output<'a>(&'a self, name: &str, next: &std::sync::atomic::AtomicU64, sync: OutputSync) -> Result<Box<dyn OutputFile + 'a>> {
        loop {
            let sequence = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = self.file_path(&format!("{name}.{}.{}.tmp", std::process::id(), sequence));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(file) => return Ok(Box::new(MmapOutput {
                    file: Some(std::io::BufWriter::with_capacity(1 << 20, file)),
                    tmp, path: self.file_path(name), closed: self.closed.clone(),
                    cold: false, written: 0, released: 0, sync,
                })),
                // A crashed writer may have had the same PID. Never truncate
                // its file, or a live writer's file on a shared filesystem.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Directory for MmapDirectory {
    fn create_output<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        static NEXT_OUTPUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if *self.closed.read() {
            return Err(closed_error());
        }
        self.create_streaming_output(name, &NEXT_OUTPUT, OutputSync::Now)
    }

    fn close(&self) {
        *self.closed.write() = true;
    }

    /// Reads the file once with large `read(2)` calls. Besides warming the
    /// page cache, this matters on macOS: the first mmap fault on pages
    /// that were just written through `write(2)` is very slow (a merge
    /// right after a flush spent seconds in faults), and after a `read`
    /// pass the same faults are cheap.
    fn prefetch(&self, name: &str) {
        use std::io::Read;
        let Ok(mut f) = std::fs::File::open(self.file_path(name)) else { return };
        let mut buf = vec![0u8; 4 << 20];
        while let Ok(n) = f.read(&mut buf) {
            if n == 0 {
                break;
            }
        }
    }

    fn open_read(&self, name: &str) -> Result<FileSlice> {
        self.open_read_identified(name).map(|(slice, _)| slice)
    }

    /// The identity is the mapped descriptor's (`fstat`), never a second
    /// lookup of the name, which a concurrent replace could move on.
    fn open_read_identified(&self, name: &str) -> Result<(FileSlice, Option<FileIdentity>)> {
        // A retired shard pathname may already belong to another store.
        // Existing FileSlice mappings pin their inode; new opens must never
        // cross that generation boundary. Hold the fence through mmap.
        let closed = self.closed.read();
        if *closed { return Err(closed_error()); }
        let path = self.file_path(name);
        let file = std::fs::File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                RusticError::IndexNotFound(format!("file {name}"))
            } else {
                e.into()
            }
        })?;
        let meta = file.metadata()?;
        let identity = FileIdentity::of(&meta);
        if meta.len() == 0 {
            return Ok((FileSlice::empty(), identity));
        }
        if let Some(admission) = self.in_memory {
            return Ok((read_whole(&file, name, meta.len(), admission)?, identity));
        }
        // SAFETY: files are written atomically (tmp + rename) and never
        // modified in place afterwards, so the mapping cannot observe a
        // partial write.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok((FileSlice::new(Arc::new(mmap)), identity))
    }

    fn write_parts(&self, name: &str, parts: &[&[u8]]) -> Result<()> {
        self.changing(|| self.write_parts_open(name, parts, true))
    }

    fn write_parts_unsynced(&self, name: &str, parts: &[&[u8]]) -> Result<()> {
        self.changing(|| self.write_parts_open(name, parts, false))
    }

    fn create_output_unsynced<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        static NEXT_OUTPUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if *self.closed.read() {
            return Err(closed_error());
        }
        self.create_streaming_output(name, &NEXT_OUTPUT, OutputSync::Deferred(self.unsynced.clone()))
    }

    fn create_scratch_output<'a>(&'a self, name: &str) -> Result<Box<dyn OutputFile + 'a>> {
        static NEXT_OUTPUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if *self.closed.read() {
            return Err(closed_error());
        }
        self.create_streaming_output(name, &NEXT_OUTPUT, OutputSync::Never)
    }

    fn sync_unsynced(&self) -> Result<()> {
        let names: Vec<String> = std::mem::take(&mut *self.unsynced.lock()).into_iter().collect();
        if names.is_empty() {
            return Ok(());
        }
        // In parallel: each fsync waits on the device, not the CPU.
        let failed = parking_lot::Mutex::new(None);
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..names.len().min(8) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(name) = names.get(i) else { break };
                    let synced = match std::fs::File::open(self.file_path(name)) {
                        Ok(f) => f.sync_all(),
                        // Deleted since (merged away): nothing to make durable.
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                        Err(e) => Err(e),
                    };
                    if let Err(e) = synced {
                        failed.lock().get_or_insert(e);
                    }
                });
            }
        });
        let synced: Result<()> = match failed.into_inner() {
            Some(e) => Err(RusticError::from(e)),
            None => crate::platform::sync_dir(&self.root).map_err(RusticError::from),
        };
        if synced.is_err() {
            // Not durable: a commit must not reference them. A later
            // retry syncs them again (the caller decides whether a failed
            // fsync is final).
            self.unsynced.lock().extend(names);
        }
        synced
    }

    fn write(&self, name: &str, data: &[u8]) -> Result<()> {
        self.changing(|| self.write_open(name, data))
    }

    fn delete(&self, name: &str) -> Result<()> {
        self.changing(|| match std::fs::remove_file(self.file_path(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        })
    }

    fn exists(&self, name: &str) -> bool {
        self.file_path(name).is_file()
    }

    fn list_admitted(&self, temporary: bool) -> Result<AdmittedNames> {
        let mut names=AdmittedNames::new()?;
        for entry in std::fs::read_dir(&self.root)? {
            let entry=entry?;
            if entry.file_type()?.is_file() {
                let os_name=entry.file_name();
                if let Some(name)=os_name.to_str() {
                    if temporary || !name.ends_with(".tmp") { names.push(name)?; }
                }
            }
        }
        names.sort(); Ok(names)
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if !name.ends_with(".tmp") {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names.sort();
        Ok(names)
    }

    fn list_temporary(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".tmp") {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unsynced_segment_files_are_visible_now_and_synced_by_sync_unsynced() {
        use super::{Directory, OutputFile};
        let tmp = tempfile::tempdir().unwrap();
        let dir = super::MmapDirectory::open(tmp.path()).unwrap();
        dir.write_parts_unsynced("a.seg", &[b"ab", b"cd"]).unwrap();
        let mut out = dir.create_output_unsynced("b.seg").unwrap();
        out.write_all(b"merged").unwrap();
        out.finish().unwrap();
        let mut scratch = dir.create_scratch_output(".merge-x.tmp").unwrap();
        scratch.write_all(b"scratch").unwrap();
        scratch.finish().unwrap();
        // Visible (atomically renamed) before any sync.
        assert_eq!(dir.open_read("a.seg").unwrap().as_slice(), b"abcd");
        assert_eq!(dir.open_read("b.seg").unwrap().as_slice(), b"merged");
        let pending = |d: &super::MmapDirectory| { let mut v: Vec<String> = d.unsynced.lock().iter().cloned().collect(); v.sort(); v };
        // Scratch files are never recorded; synced writes neither.
        dir.write("c.seg", b"synced").unwrap();
        assert_eq!(pending(&dir), vec!["a.seg".to_string(), "b.seg".to_string()]);
        // A file deleted before the commit (merged away) is skipped.
        dir.delete("a.seg").unwrap();
        dir.sync_unsynced().unwrap();
        assert!(pending(&dir).is_empty());
        dir.sync_unsynced().unwrap();
    }

    #[test]
    fn listing_denial_preserves_existing_names_and_retained_owner() {
        let budget=crate::admission::Budget::new(4096);
        let mut names=AdmittedNames::with_budget(&budget).unwrap();
        for _ in 0..8 { names.push("x").unwrap(); }
        let retained=budget.used();
        budget.set_limit(retained);
        assert!(matches!(names.push("denied"),Err(RusticError::Memory(_))));
        assert_eq!(names.len(),8);
        assert!(names.iter().all(|name|name=="x"));
        assert_eq!(budget.used(),retained);
        drop(names);
        assert_eq!(budget.used(),0);
    }
    use super::*;

    #[test]
    fn retired_directory_keeps_mapped_bytes_but_cannot_open_replacement_files() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("shard");
        let old = MmapDirectory::open(&path).unwrap();
        old.write("segment", b"old generation").unwrap();
        let retained = old.open_read("segment").unwrap();
        old.close();
        std::fs::rename(&path, tmp.path().join("retired")).unwrap();
        let new = MmapDirectory::open(&path).unwrap();
        new.write("segment", b"new generation").unwrap();
        std::fs::remove_dir_all(tmp.path().join("retired")).unwrap();
        assert_eq!(retained.as_slice(), b"old generation");
        assert!(old.open_read("segment").is_err(), "closed reader must not map replacement data");
        assert_eq!(new.open_read("segment").unwrap().as_slice(), b"new generation");
    }

    fn exercise(dir: &dyn Directory) {
        assert!(!dir.exists("a"));
        dir.write("a", b"hello world").unwrap();
        assert!(dir.exists("a"));
        let f = dir.open_read("a").unwrap();
        assert_eq!(f.as_slice(), b"hello world");
        assert_eq!(f.slice(6..11).as_slice(), b"world");
        dir.write("a", b"replaced").unwrap();
        assert_eq!(dir.open_read("a").unwrap().as_slice(), b"replaced");
        // The old mapping is still valid.
        assert_eq!(f.as_slice(), b"hello world");
        dir.write("empty", b"").unwrap();
        assert!(dir.open_read("empty").unwrap().is_empty());
        assert_eq!(dir.list().unwrap(), vec!["a".to_string(), "empty".to_string()]);
        dir.delete("a").unwrap();
        assert!(!dir.exists("a"));
        assert!(dir.open_read("a").is_err());
    }

    #[test]
    fn streamed_outputs_publish_atomically_and_clean_abandoned_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        dir.write("target", b"original").unwrap();
        let old = dir.open_read("target").unwrap();
        let mut first = dir.create_output("target").unwrap();
        let mut second = dir.create_output("target").unwrap();
        first.write_all(b"first").unwrap();
        second.write_all(b"second").unwrap();
        assert_eq!(dir.open_read("target").unwrap().as_slice(), b"original");
        first.finish().unwrap();
        assert_eq!(dir.open_read("target").unwrap().as_slice(), b"first");
        second.finish().unwrap();
        assert_eq!(dir.open_read("target").unwrap().as_slice(), b"second");
        assert_eq!(old.as_slice(), b"original");
        let mut abandoned = dir.create_output("target").unwrap();
        abandoned.write_all(b"partial").unwrap();
        drop(abandoned);
        assert_eq!(dir.open_read("target").unwrap().as_slice(), b"second");
        assert_eq!(dir.list().unwrap(), vec!["target"]);
    }

    #[test]
    fn streaming_output_skips_stale_temporary_names_after_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        let stale = format!("target.{}.0.tmp", std::process::id());
        std::fs::write(tmp.path().join(&stale), b"unowned bytes").unwrap();
        let next = std::sync::atomic::AtomicU64::new(0);
        let mut output = dir.create_streaming_output("target", &next, OutputSync::Now).unwrap();
        output.write_all(b"new bytes").unwrap();
        output.finish().unwrap();
        assert_eq!(dir.open_read("target").unwrap().as_slice(), b"new bytes");
        assert_eq!(std::fs::read(tmp.path().join(stale)).unwrap(), b"unowned bytes");
        assert_eq!(next.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn cold_checkpoint_writeback_preserves_bytes_and_atomic_publication() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        dir.write("checkpoint", b"previous commit").unwrap();
        let previous = dir.open_read("checkpoint").unwrap();
        let mut output = dir.create_output("checkpoint").unwrap();
        output.cold_checkpoint();
        let block: Vec<u8> = (0..65536).map(|i| (i % 251) as u8).collect();
        // Cross a writeback boundary, then leave a non-page-aligned tail.
        let count = CHECKPOINT_WRITE_WINDOW as usize / block.len() + 1;
        for _ in 0..count { output.write_all(&block).unwrap(); }
        output.write_all(b"tail").unwrap();
        assert_eq!(dir.open_read("checkpoint").unwrap().as_slice(), b"previous commit");
        output.finish().unwrap();
        let result = dir.open_read("checkpoint").unwrap();
        assert_eq!(result.len(), count * block.len() + 4);
        for chunk in result.as_slice()[..count * block.len()].chunks_exact(block.len()) {
            assert_eq!(chunk, block.as_slice());
        }
        assert_eq!(&result.as_slice()[count * block.len()..], b"tail");
        assert_eq!(previous.as_slice(), b"previous commit");
        assert!(dir.list_temporary().unwrap().is_empty());

        // Intermediate writeback never publishes. A retired directory must
        // still reject finish and clean up the already-synced temporary file.
        let mut rejected = dir.create_output("checkpoint").unwrap();
        rejected.cold_checkpoint();
        for _ in 0..count { rejected.write_all(&block).unwrap(); }
        dir.close();
        assert!(rejected.finish().is_err());
        // A closed directory refuses opens, so check the files themselves.
        assert_eq!(std::fs::read(tmp.path().join("checkpoint")).unwrap(), result.as_slice());
        let temporary: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).collect();
        assert!(temporary.is_empty(), "{temporary:?}");
    }

    #[test]
    fn failed_stream_publication_removes_temporary_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        let mut output = dir.create_output("blocked").unwrap();
        output.write_all(b"contents").unwrap();
        std::fs::create_dir(tmp.path().join("blocked")).unwrap();
        assert!(output.finish().is_err());
        assert!(dir.list().unwrap().is_empty());
        assert!(tmp.path().join("blocked").is_dir());
    }

    #[test]
    fn ram_directory() {
        exercise(&RamDirectory::new());
    }

    /// A streaming output larger than several writeback windows, written
    /// in uneven pieces, publishes exactly its bytes.
    #[test]
    fn streaming_output_with_incremental_writeback_preserves_bytes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        let bytes: Vec<u8> = (0..(27u64 << 20) + 12345).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let mut out = dir.create_output("big").unwrap();
        let mut at = 0;
        for (i, n) in std::iter::repeat([1usize, 3 << 20, 777, 9 << 20, 65536]).flatten().enumerate() {
            if at == bytes.len() { break; }
            let n = n.min(bytes.len() - at) + (i % 2);
            let n = n.min(bytes.len() - at);
            out.write_all(&bytes[at..at + n]).unwrap();
            at += n;
        }
        out.finish().unwrap();
        assert_eq!(dir.open_read("big").unwrap().as_slice(), &bytes[..]);
        assert!(std::fs::read_dir(tmp.path()).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".tmp")));
    }

    #[test]
    fn mmap_directory() {
        let tmp = tempfile::TempDir::new().unwrap();
        exercise(&MmapDirectory::open(tmp.path()).unwrap());
        assert!(!tmp.path().join("a.tmp").exists());
    }

    /// `node.store.allow_mmap: false`: the same contract, with every file
    /// read into memory and none of them mapped.
    #[test]
    fn unmapped_directory() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = MmapDirectory::open_in_memory(tmp.path(), roomy).unwrap();
        exercise(&dir);
        let bytes: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        dir.write("big", &bytes).unwrap();
        let (slice, identity) = dir.open_read_identified("big").unwrap();
        assert_eq!(slice.as_slice(), &bytes[..]);
        assert_eq!(identity.map(|i| i.len), FileIdentity::of(&std::fs::metadata(tmp.path().join("big")).unwrap()).map(|i| i.len));
        // The bytes are owned: replacing the file leaves an open slice as it was.
        dir.write("big", b"replaced").unwrap();
        assert_eq!(slice.as_slice(), &bytes[..]);
        #[cfg(target_os = "linux")]
        {
            assert!(super::mapped_paths_under(tmp.path()).is_empty());
            // The check sees mappings: a mapping directory's open shows up.
            let mapping = MmapDirectory::open_with(tmp.path(), true).unwrap();
            let mapped = mapping.open_read("big").unwrap();
            assert_eq!(super::mapped_paths_under(tmp.path()).len(), 1);
            drop(mapped);
        }
    }

    /// A budget of its own for each admission test (a `fn` pointer cannot
    /// capture one).
    fn budget(slot: &'static std::sync::OnceLock<Arc<crate::admission::Budget>>, limit: u64) -> Arc<crate::admission::Budget> {
        slot.get_or_init(|| crate::admission::Budget::new(limit)).clone()
    }
    static CHARGED: std::sync::OnceLock<Arc<crate::admission::Budget>> = std::sync::OnceLock::new();
    fn charged(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal> {
        let b = budget(&CHARGED, 64 << 20);
        b.reserve(bytes, file).map(|r| Box::new(r) as Box<dyn Send>).map_err(|e| StoreFileRefusal { limit: b.limit(), reason: e.to_string() })
    }

    /// Every in-memory store file is charged exactly its size while any
    /// slice of it lives, and released when the last one drops.
    #[test]
    fn in_memory_files_are_charged_while_held() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = MmapDirectory::open_in_memory(tmp.path(), charged).unwrap();
        let used = || budget(&CHARGED, 64 << 20).used();
        dir.write("a", &vec![1u8; 1_000_000]).unwrap();
        dir.write("b", &vec![2u8; 250_000]).unwrap();
        dir.write("empty", b"").unwrap();
        assert_eq!(used(), 0, "writing charges nothing");
        let a = dir.open_read("a").unwrap();
        assert_eq!(used(), 1_000_000);
        let b = dir.open_read("b").unwrap();
        assert_eq!(used(), 1_250_000);
        let _empty = dir.open_read("empty").unwrap();
        assert_eq!(used(), 1_250_000, "an empty file holds no buffer");
        // A sub-slice keeps the whole file (and its charge) alive.
        let part = a.slice(10..20);
        drop(a);
        assert_eq!(used(), 1_250_000);
        assert_eq!(part.as_slice(), &[1u8; 10]);
        drop(part);
        assert_eq!(used(), 250_000);
        drop(b);
        assert_eq!(used(), 0);
    }

    static ROOMY: std::sync::OnceLock<Arc<crate::admission::Budget>> = std::sync::OnceLock::new();
    fn roomy(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal> {
        let b = budget(&ROOMY, 1 << 30);
        b.reserve(bytes, file).map(|r| Box::new(r) as Box<dyn Send>).map_err(|e| StoreFileRefusal { limit: b.limit(), reason: e.to_string() })
    }

    static SMALL: std::sync::OnceLock<Arc<crate::admission::Budget>> = std::sync::OnceLock::new();
    fn small(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal> {
        let b = budget(&SMALL, 100_000);
        b.reserve(bytes, file).map(|r| Box::new(r) as Box<dyn Send>).map_err(|e| StoreFileRefusal { limit: b.limit(), reason: e.to_string() })
    }

    /// A file over the limit is refused before anything is allocated, with
    /// an error naming the file, its size, the limit and the way out; the
    /// files that fit still open.
    #[test]
    fn in_memory_file_over_the_limit_is_refused() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = MmapDirectory::open_in_memory(tmp.path(), small).unwrap();
        dir.write("fits", &vec![1u8; 60_000]).unwrap();
        dir.write("big", &vec![2u8; 150_000]).unwrap();
        let fits = dir.open_read("fits").unwrap();
        let err = dir.open_read("big").unwrap_err();
        let RusticError::Memory(message) = &err else { panic!("{err:?}") };
        for part in ["[big]", "[150000] bytes", "[100000] bytes", "[node.store.allow_mmap]", "[vm.max_map_count]", "needs 150000 bytes, limit 100000"] {
            assert!(message.contains(part), "[{part}] missing from: {message}");
        }
        assert!(message.ends_with("needs 150000 bytes, limit 100000"), "{message}");
        assert_eq!(budget(&SMALL, 100_000).used(), 60_000, "the refusal holds nothing");
        // Room for it once the other file is released.
        drop(fits);
        dir.write("big", &vec![2u8; 90_000]).unwrap();
        assert_eq!(dir.open_read("big").unwrap().len(), 90_000);
    }
}

/// Test support: the files under `root` this process maps now
/// (`/proc/self/maps`).
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn mapped_paths_under(root: &Path) -> Vec<String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let root = root.to_string_lossy().into_owned();
    let mut paths: Vec<String> = std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter_map(|line| line.split_whitespace().nth(5).filter(|p| p.starts_with(root.as_str())).map(str::to_string))
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
mod admitted_ram_tests {
    use super::*;
    #[test]
    fn deleted_ram_file_stays_charged_through_last_slice() {
        let owner = crate::admission::Owner::new();
        let _scope = owner.enter();
        let dir = RamDirectory::new();
        dir.write("file", &[1, 2, 3]).unwrap();
        let retained = dir.open_read("file").unwrap();
        let charged = owner.bytes().current;
        assert!(charged >= 3);
        dir.delete("file").unwrap();
        drop(dir);
        assert_eq!(owner.bytes().current, charged);
        assert_eq!(retained.as_slice(), &[1, 2, 3]);
        drop(retained);
        assert_eq!(owner.bytes().current, 0);
    }
}
