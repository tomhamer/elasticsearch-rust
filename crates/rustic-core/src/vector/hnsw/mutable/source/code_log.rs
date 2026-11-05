//! Append-only, file-backed code-only quantized rows: the mutable graph's
//! counterpart of Lucene's `.veq` file.
//!
//! A quantized row whose f32 is durable in the vector store keeps only its
//! identity header, codes and corrections ([`super::code_only_row`]). Those
//! rows are immutable, so instead of living in the owner's anonymous
//! [`super::ArrayStore`] they are appended here, and a tagged source
//! directory slot ([`super::LOG_TAG`]) addresses them by row index.
//!
//! Rows are grouped in fixed-size chunks. The newest chunk is an anonymous
//! buffer (bounded by [`CHUNK_BYTES`]); a full chunk is written once to an
//! immutable `mvq_*` file and from then on read through the directory's
//! mapping, so the page cache holds it and the kernel may evict it, as for
//! Lucene's mapped vector files. Durable files are never rewritten: a graph
//! checkpoint references the files covering the rows it uses (the newest,
//! partially filled chunk as a prefix file), so checkpoint I/O is the new
//! rows only, never the whole field.
//!
//! Readers address rows without locks: chunk base pointers are published
//! before any slot tagging a row in them, and a buffer replaced by its
//! mapping is retired through the graph's generation domain (freed only once
//! no reader older than the swap remains).
//!
//! A log is append-only, so every update or delete of a shed row leaves a
//! dead row behind. A log's chunk directory is two-level (pages of
//! [`PAGE_CHUNKS`] chunk pointers allocated on demand), so a mostly-live log
//! grows to the full 30-bit row space instead of stopping at a fixed chunk
//! count.
//! [`CodeLogs`] therefore holds two log slots: when the active log's dead
//! fraction passes a threshold, a fresh log (new id) becomes active and the
//! live rows are copied into it in bounded batches, each source slot retagged
//! to the new log. The old log stays readable until no reader older than the
//! switch remains, then is dropped; its files go once no commit references
//! them. A full log is backpressure (rows keep their f32), never an error.
use super::*;
use std::sync::atomic::{AtomicPtr, AtomicUsize};

/// Target bytes of one chunk: the anonymous buffer of the newest rows, and
/// each durable file. Bounded resident overhead per field, large writes.
const CHUNK_BYTES: usize = 16 << 20;
/// Chunk pointers per directory page (allocated when first needed).
const PAGE_CHUNKS: usize = 4096;
/// Directory pages one log addresses.
const MAX_PAGES: usize = 1024;
/// Chunks one log addresses (with 16 MiB chunks, past the 30-bit row space
/// for every row width up to 16 KiB).
const MAX_CHUNKS: usize = PAGE_CHUNKS * MAX_PAGES;

/// One page of a log's chunk directory: chunk base pointers. Never freed
/// before the log, so a reader holding the log may follow it lock-free.
struct DirectoryPage {
    slots: Box<[AtomicPtr<u32>]>,
    _memory: crate::admission::Reservation,
}
/// Rows a tagged source slot can address (30 bits; the next bit names the
/// [`CodeLogs`] slot).
pub(super) const MAX_ROWS: u64 = (1 << 30) - 1;

/// File name of chunk `chunk` of log `log` holding its first `rows` rows.
/// A prefix file is immutable: a later prefix of the same chunk is a new
/// name, and the older one goes away with the last commit referencing it.
pub(super) fn chunk_file_name(log: u64, chunk: u32, rows: u32) -> String {
    format!("mvq_{log:016x}_{chunk:08x}_{rows:08x}.rmvq")
}

#[cfg(test)]
use crate::vector::store::is_code_log_file;

/// A fresh log identity: unique per process and across restarts (a log
/// restored from a checkpoint keeps its identity).
fn new_log_id() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let n = COUNTER.fetch_add(1, Relaxed);
    let mut x = nanos ^ (u64::from(std::process::id()) << 32) ^ n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 31; x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9); x ^= x >> 29;
    x.max(1)
}

/// An anonymous, zero-initialised (lazily committed) chunk buffer.
struct Buffer {
    ptr: std::ptr::NonNull<u32>,
    words: usize,
    _memory: crate::admission::Reservation,
}
// SAFETY: the buffer is plain memory; writers are serialised by the log
// (single appender under the graph writer) and never touch published rows.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
impl Buffer {
    fn layout(words: usize) -> Result<std::alloc::Layout> {
        let bytes = words.checked_mul(4).filter(|b| *b > 0).ok_or_else(|| failure("code log chunk size"))?;
        std::alloc::Layout::from_size_align(bytes, 64).map_err(|_| failure("code log chunk layout"))
    }
    fn new(words: usize) -> Result<Self> {
        let layout = Self::layout(words)?;
        let memory = crate::admission::shared().reserve(layout.size() as u64, "mutable code log active chunk")?;
        // SAFETY: non-zero size (checked in `layout`).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<u32>();
        let ptr = std::ptr::NonNull::new(ptr).ok_or_else(|| RusticError::Memory("code log chunk allocation failed".into()))?;
        crate::accounting::CODE_LOG_ANON_BYTES.fetch_add(layout.size() as u64, Relaxed);
        Ok(Self { ptr, words, _memory: memory })
    }
    fn bytes(&self, words: usize) -> &[u8] {
        // SAFETY: `words` never exceeds the allocation; rows below it are
        // fully written (callers only pass published row counts).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<u8>(), words.min(self.words) * 4) }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Ok(layout) = Self::layout(self.words) {
            crate::accounting::CODE_LOG_ANON_BYTES.fetch_sub(layout.size() as u64, Relaxed);
            // SAFETY: allocated in `new` with this layout.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), layout) };
        }
    }
}

enum Backing {
    Anon(Buffer),
    /// The chunk's durable file, mapped (keeps the mapping alive).
    Mapped(crate::codec::directory::FileSlice),
}

#[derive(Default)]
struct State {
    rows: u64,
    backing: Vec<Option<Backing>>,
    /// Per chunk, the newest durable file: (rows it holds, name).
    persisted: Vec<Option<(u32, String)>>,
    /// Buffers replaced by their mapping, with the generation they were
    /// retired at (freed once every older reader is gone).
    retired: Vec<(u64, Buffer)>,
}

/// One durable chunk a checkpoint references: its index and row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChunkRef { pub chunk: u32, pub rows: u32 }

/// A full chunk written to its file, waiting to replace its buffer.
pub(super) struct SealedChunk { chunk: usize, file: crate::codec::directory::FileSlice }

pub(super) struct CodeLog {
    id: u64,
    /// Words per row (identity header + codes + corrections); 0 until the
    /// first row fixes it.
    row_words: AtomicUsize,
    rows_per_chunk: AtomicUsize,
    /// Two-level chunk directory: [`MAX_PAGES`] page pointers.
    chunks: Box<[AtomicPtr<DirectoryPage>]>,
    state: parking_lot::Mutex<State>,
    /// Serialises file writes (never held with the graph writer).
    io: parking_lot::Mutex<()>,
    /// Source slots currently tagged with a row of this log (maintained by
    /// the owner under the graph writer); `rows - live` rows are dead.
    live: std::sync::atomic::AtomicU64,
    /// Test seam: a smaller capacity (0: none).
    #[cfg(test)]
    test_capacity: std::sync::atomic::AtomicU64,
    _directory_memory: crate::admission::Reservation,
}

impl CodeLog {
    pub fn new() -> Result<Self> {
        Self::with_id(new_log_id())
    }

    fn with_id(id: u64) -> Result<Self> {
        let directory_memory = crate::admission::shared().reserve((MAX_PAGES * std::mem::size_of::<AtomicPtr<DirectoryPage>>()) as u64, "code log chunk directory")?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(MAX_PAGES).map_err(|_| failure("code log chunk directory allocation"))?;
        chunks.extend((0..MAX_PAGES).map(|_| AtomicPtr::new(std::ptr::null_mut())));
        Ok(Self { id, row_words: AtomicUsize::new(0), rows_per_chunk: AtomicUsize::new(0),
            chunks: chunks.into_boxed_slice(), state: parking_lot::Mutex::new(State::default()), io: parking_lot::Mutex::new(()),
            live: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            test_capacity: std::sync::atomic::AtomicU64::new(0),
            _directory_memory: directory_memory })
    }

    /// The directory entry of `chunk`, if its page exists.
    #[inline]
    fn chunk_slot(&self, chunk: usize) -> Option<&AtomicPtr<u32>> {
        let page = self.chunks.get(chunk / PAGE_CHUNKS)?.load(Acquire);
        if page.is_null() { return None; }
        // SAFETY: pages are published fully initialised (Release) and freed
        // only when the log drops.
        unsafe { &*page }.slots.get(chunk % PAGE_CHUNKS)
    }

    /// The directory entry of `chunk`, allocating its page. Only the single
    /// appender (or restore, before the log is shared) calls this.
    fn ensure_chunk_slot(&self, chunk: usize) -> Result<&AtomicPtr<u32>> {
        let page_index = chunk / PAGE_CHUNKS;
        let top = self.chunks.get(page_index).ok_or_else(|| failure("code log chunk directory exhausted"))?;
        if top.load(Acquire).is_null() {
            let memory = crate::admission::shared().reserve((PAGE_CHUNKS * std::mem::size_of::<AtomicPtr<u32>>()) as u64, "code log chunk directory page")?;
            let mut slots = Vec::new();
            slots.try_reserve_exact(PAGE_CHUNKS).map_err(|_| failure("code log chunk directory page allocation"))?;
            slots.extend((0..PAGE_CHUNKS).map(|_| AtomicPtr::new(std::ptr::null_mut())));
            let page = Box::new(DirectoryPage { slots: slots.into_boxed_slice(), _memory: memory });
            top.store(Box::into_raw(page), Release);
        }
        self.chunk_slot(chunk).ok_or_else(|| failure("code log chunk directory page"))
    }

    /// A source slot now tags one of this log's rows.
    pub fn tagged(&self) { self.live.fetch_add(1, Relaxed); }
    /// A source slot stopped tagging one of this log's rows.
    pub fn untagged(&self) {
        let _ = self.live.fetch_update(Relaxed, Relaxed, |live| Some(live.saturating_sub(1)));
    }
    /// Rows some source slot still tags.
    pub fn live_rows(&self) -> u64 { self.live.load(Relaxed) }
    /// Rows this log can ever hold (`None` until the first row fixes the
    /// shape).
    pub fn capacity(&self) -> Option<u64> {
        let per = self.rows_per_chunk() as u64;
        #[cfg(test)]
        {
            let cap = self.test_capacity.load(Relaxed);
            if per > 0 && cap > 0 { return Some(cap); }
        }
        (per > 0).then(|| per.saturating_mul(MAX_CHUNKS as u64).min(MAX_ROWS + 1))
    }
    /// Test seam: pretend the log holds at most `rows` rows.
    #[cfg(test)]
    pub fn set_test_capacity(&self, rows: u64) { self.test_capacity.store(rows, Relaxed); }
    /// No further row fits: appending would fail.
    pub fn is_full(&self) -> bool {
        self.capacity().is_some_and(|capacity| self.rows() >= capacity)
    }
    /// Whether the dead rows justify copying the live ones into a fresh
    /// log: at least `min_rows` rows and half of them dead, or the log at
    /// three quarters of its capacity with at most half of the capacity
    /// live. Either way the copy lands at most half full, so it does not
    /// re-trigger soon and leaves room for shedding meanwhile.
    pub fn wants_compaction(&self, min_rows: u64) -> bool {
        let rows = self.rows();
        if rows == 0 || rows < min_rows { return false; }
        let live = self.live_rows();
        let dead = rows.saturating_sub(live);
        let near_full = self.capacity().is_some_and(|capacity| rows.saturating_mul(4) >= capacity.saturating_mul(3)
            && live.saturating_mul(2) <= capacity);
        dead.saturating_mul(2) >= rows || near_full
    }

    pub fn id(&self) -> u64 { self.id }
    pub fn row_words(&self) -> usize { self.row_words.load(Acquire) }
    pub fn rows_per_chunk(&self) -> usize { self.rows_per_chunk.load(Acquire) }
    /// Rows appended so far.
    pub fn rows(&self) -> u64 { self.state.lock().rows }

    fn fix_shape(&self, row_words: usize) -> Result<()> {
        if row_words == 0 { return Err(failure("empty code log row")); }
        let rows_per_chunk = (CHUNK_BYTES / (row_words * 4)).max(1);
        self.rows_per_chunk.store(rows_per_chunk, Release);
        self.row_words.store(row_words, Release);
        Ok(())
    }

    /// Row `index` (identity header, codes, corrections). The caller holds a
    /// generation guard of the owning graph's domain taken before the slot
    /// tagging this row was read; the row stays valid while it lives.
    #[inline]
    pub fn row(&self, index: u64) -> Option<&[f32]> {
        let per = self.rows_per_chunk.load(Acquire);
        let words = self.row_words.load(Acquire);
        if per == 0 || words == 0 { return None; }
        let chunk = usize::try_from(index / per as u64).ok()?;
        let base = self.chunk_slot(chunk)?.load(Acquire);
        if base.is_null() { return None; }
        let offset = (index % per as u64) as usize * words;
        // SAFETY: the chunk holds `per * words` words; rows are written
        // before any slot tags them (Release/Acquire on the slot), and the
        // guard keeps a retired buffer alive (see the module docs).
        Some(unsafe { std::slice::from_raw_parts(base.add(offset).cast::<f32>(), words) })
    }

    /// Append one row; returns its index. The caller is the single
    /// appender (it holds the owning graph's writer).
    pub fn append(&self, words: &[u32]) -> Result<u64> {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        if self.row_words.load(Acquire) == 0 { self.fix_shape(words.len())?; }
        let row_words = self.row_words.load(Acquire);
        if words.len() != row_words { return Err(failure("code log row width changed")); }
        let per = self.rows_per_chunk.load(Acquire);
        let index = state.rows;
        if index > MAX_ROWS { return Err(failure("code log row index exhausted")); }
        if per == 0 { return Err(failure("code log chunk shape")); }
        let chunk = usize::try_from(index / per as u64).map_err(|_| failure("code log chunk index"))?;
        if chunk >= MAX_CHUNKS { return Err(failure("code log chunk directory exhausted")); }
        if state.backing.len() <= chunk {
            let (backing, persisted) = (state.backing.len(), state.persisted.len());
            state.backing.try_reserve(chunk + 1 - backing).map_err(|_| failure("code log chunk table allocation"))?;
            state.persisted.try_reserve((chunk + 1).saturating_sub(persisted)).map_err(|_| failure("code log chunk table allocation"))?;
            while state.backing.len() <= chunk { state.backing.push(None); }
            while state.persisted.len() <= chunk { state.persisted.push(None); }
        }
        if state.backing[chunk].is_none() {
            let buffer = Buffer::new(per.checked_mul(row_words).ok_or_else(|| failure("code log chunk size"))?)?;
            self.ensure_chunk_slot(chunk)?.store(buffer.ptr.as_ptr(), Release);
            state.backing[chunk] = Some(Backing::Anon(buffer));
        }
        let base = match &state.backing[chunk] {
            Some(Backing::Anon(buffer)) => buffer.ptr.as_ptr(),
            _ => return Err(failure("code log append into a sealed chunk")),
        };
        let offset = (index % per as u64) as usize * row_words;
        // SAFETY: within this chunk's allocation; no reader sees this row
        // before its slot is published, and the appender is unique.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), base.add(offset), row_words) };
        state.rows = index + 1;
        Ok(index)
    }

    /// Write every full chunk that has no file yet (off every graph lock).
    /// Each returned chunk replaces its buffer with its mapping once the
    /// caller installs it ([`Self::install_sealed`]) under the graph writer.
    pub fn seal_full(&self, directory: &dyn crate::codec::directory::Directory) -> Result<Vec<SealedChunk>> {
        let _io = self.io.lock();
        let per = self.rows_per_chunk();
        let words = self.row_words();
        if per == 0 { return Ok(Vec::new()); }
        let pending: Vec<usize> = {
            let state = self.state.lock();
            let full = usize::try_from(state.rows / per as u64).unwrap_or(usize::MAX);
            (0..full.min(state.backing.len())).filter(|&chunk| matches!(state.backing[chunk], Some(Backing::Anon(_)))
                && state.persisted[chunk].as_ref().is_none_or(|(rows, _)| (*rows as usize) < per)).collect()
        };
        let mut sealed = Vec::with_capacity(pending.len());
        for chunk in pending {
            let rows = u32::try_from(per).map_err(|_| failure("code log chunk rows"))?;
            let name = self.write_prefix(directory, chunk, per * words, rows)?;
            let file = directory.open_read(&name)?;
            self.state.lock().persisted[chunk] = Some((rows, name));
            if file.len() == per * words * 4 && file.as_slice().as_ptr().align_offset(4) == 0 {
                sealed.push(SealedChunk { chunk, file });
            }
        }
        Ok(sealed)
    }

    /// Replace sealed chunks' buffers with their mappings. The caller holds
    /// the graph writer (no generation advance between recording the
    /// retirement at `generation` and publishing the new base).
    pub fn install_sealed(&self, sealed: Vec<SealedChunk>, generation: u64) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        for SealedChunk { chunk, file } in sealed {
            let Some(slot) = state.backing.get_mut(chunk) else { continue };
            if !matches!(*slot, Some(Backing::Anon(_))) { continue; }
            let base = file.as_slice().as_ptr().cast::<u32>().cast_mut();
            crate::accounting::CODE_LOG_MAPPED_BYTES.fetch_add(file.len() as u64, Relaxed);
            if let Some(Backing::Anon(buffer)) = slot.replace(Backing::Mapped(file)) {
                // Mapped bytes are read-only: never written through `base`.
                if let Some(entry) = self.chunk_slot(chunk) { entry.store(base, Release); }
                state.retired.push((generation, buffer));
            }
        }
    }

    /// Free retired buffers no reader can still hold.
    pub fn reclaim(&self, oldest: u64) {
        // A file write may still be reading a retired buffer (it holds `io`).
        let Some(_io) = self.io.try_lock() else { return };
        self.state.lock().retired.retain(|(generation, _)| *generation >= oldest);
    }

    fn write_prefix(&self, directory: &dyn crate::codec::directory::Directory, chunk: usize, words: usize, rows: u32) -> Result<String> {
        enum Source { Anon(*const u8, usize), Mapped(crate::codec::directory::FileSlice) }
        // Resolve the bytes under the state lock, write them without it (the
        // appender takes it per row under the graph writer). An anonymous
        // buffer stays allocated meanwhile: the caller holds `io`, and
        // retired buffers are only freed with `io` free ([`Self::reclaim`]).
        let source = {
            let state = self.state.lock();
            match state.backing.get(chunk) {
                Some(Some(Backing::Anon(buffer))) => { let bytes = buffer.bytes(words); Source::Anon(bytes.as_ptr(), bytes.len()) }
                Some(Some(Backing::Mapped(file))) => Source::Mapped(file.clone()),
                _ => return Err(failure("code log chunk absent")),
            }
        };
        let name = chunk_file_name(self.id, chunk as u32, rows);
        let mut output = directory.create_output(&name)?;
        match &source {
            // SAFETY: see above; these rows are published and immutable.
            Source::Anon(ptr, len) => output.write_all(unsafe { std::slice::from_raw_parts(*ptr, *len) })?,
            Source::Mapped(file) => output.write_all(&file.as_slice()[..(words * 4).min(file.len())])?,
        }
        output.finish()?;
        Ok(name)
    }

    /// Make the rows below `upto` durable: every chunk they span has a file
    /// holding at least its rows below `upto` (existing files are reused;
    /// only the newest chunk's prefix is written). Returns the chunks a
    /// checkpoint references for those rows.
    pub fn persist(&self, directory: &dyn crate::codec::directory::Directory, upto: u64) -> Result<Vec<ChunkRef>> {
        let _io = self.io.lock();
        let per = self.rows_per_chunk() as u64;
        let words = self.row_words();
        if upto == 0 || per == 0 { return Ok(Vec::new()); }
        if upto > self.rows() { return Err(failure("code log persistence beyond appended rows")); }
        let chunks = upto.div_ceil(per);
        let mut out = Vec::with_capacity(chunks as usize);
        for chunk in 0..chunks {
            let needed = (upto - chunk * per).min(per) as u32;
            let chunk = chunk as usize;
            let existing = self.state.lock().persisted.get(chunk).cloned().flatten();
            let rows = match existing {
                Some((rows, _)) if rows >= needed => rows,
                _ => {
                    let name = self.write_prefix(directory, chunk, needed as usize * words, needed)?;
                    self.state.lock().persisted[chunk] = Some((needed, name));
                    needed
                }
            };
            out.push(ChunkRef { chunk: chunk as u32, rows });
        }
        Ok(out)
    }

    /// Names of every chunk file this log wrote or restored.
    pub fn persisted_files(&self) -> Vec<String> {
        self.state.lock().persisted.iter().flatten().map(|(_, name)| name.clone()).collect()
    }

    /// File names of `chunks` of this log.
    pub fn file_names(&self, chunks: &[ChunkRef]) -> Vec<String> {
        chunks.iter().map(|c| chunk_file_name(self.id, c.chunk, c.rows)).collect()
    }

    /// Reopen a log from the chunk files a checkpoint references. Full
    /// chunks are mapped; a final prefix is copied into a fresh buffer that
    /// further rows are appended to (its file stays as it is).
    pub fn restore(directory: &dyn crate::codec::directory::Directory, id: u64, row_words: usize, rows_per_chunk: usize, chunks: &[ChunkRef]) -> Result<Self> {
        let log = Self::with_id(id)?;
        if chunks.is_empty() { return Ok(log); }
        log.fix_shape(row_words)?;
        if log.rows_per_chunk() != rows_per_chunk { return Err(corrupt("code log chunk shape")); }
        if chunks.len() > MAX_CHUNKS { return Err(corrupt("code log chunk count")); }
        let mut state = log.state.lock();
        for (position, chunk) in chunks.iter().enumerate() {
            if chunk.chunk as usize != position || chunk.rows == 0 || chunk.rows as usize > rows_per_chunk {
                return Err(corrupt("code log chunk sequence"));
            }
            let last = position + 1 == chunks.len();
            if !last && chunk.rows as usize != rows_per_chunk { return Err(corrupt("code log interior prefix chunk")); }
            let name = chunk_file_name(id, chunk.chunk, chunk.rows);
            let file = directory.open_read(&name)?;
            let words = chunk.rows as usize * row_words;
            if file.len() != words * 4 { return Err(corrupt("code log chunk length")); }
            let full = chunk.rows as usize == rows_per_chunk;
            let backing = if full && file.as_slice().as_ptr().align_offset(4) == 0 {
                log.ensure_chunk_slot(position)?.store(file.as_slice().as_ptr().cast::<u32>().cast_mut(), Release);
                crate::accounting::CODE_LOG_MAPPED_BYTES.fetch_add(file.len() as u64, Relaxed);
                Backing::Mapped(file)
            } else {
                let buffer = Buffer::new(rows_per_chunk * row_words)?;
                // SAFETY: the buffer holds `rows_per_chunk * row_words` words.
                unsafe { std::ptr::copy_nonoverlapping(file.as_slice().as_ptr(), buffer.ptr.as_ptr().cast::<u8>(), words * 4) };
                log.ensure_chunk_slot(position)?.store(buffer.ptr.as_ptr(), Release);
                Backing::Anon(buffer)
            };
            state.backing.push(Some(backing));
            state.persisted.push(Some((chunk.rows, name)));
            state.rows += u64::from(chunk.rows);
        }
        drop(state);
        Ok(log)
    }

    /// (anonymous buffer bytes, mapped file bytes).
    pub fn memory_bytes(&self) -> (usize, usize) {
        let state = self.state.lock();
        let mut anon = state.retired.iter().map(|(_, b)| b.words * 4).sum::<usize>();
        let mut mapped = 0;
        for backing in state.backing.iter().flatten() {
            match backing {
                Backing::Anon(buffer) => anon += buffer.words * 4,
                Backing::Mapped(file) => mapped += file.len(),
            }
        }
        (anon, mapped)
    }
}

/// The owner's code logs: the active one new rows are appended to, and
/// during or just after a compaction the previous one (see the module
/// docs). A source slot's tag names its log slot, so a reader resolves
/// either without locks; a switched-out log is freed only once no reader
/// older than the switch remains (the same generation rule as buffers).
pub(super) struct CodeLogs {
    slots: [AtomicPtr<CodeLog>; 2],
    active: AtomicUsize,
    state: parking_lot::Mutex<LogsState>,
}

#[derive(Default)]
struct LogsState {
    /// Compaction in progress: the next source slot to migrate. Every slot
    /// below it no longer tags the inactive log.
    cursor: Option<usize>,
    /// The switched-out log's slot and the generation it was retired at.
    retiring: Option<(usize, u64)>,
}

impl CodeLogs {
    pub fn new(log: CodeLog) -> Self {
        Self {
            slots: [AtomicPtr::new(Box::into_raw(Box::new(log))), AtomicPtr::new(std::ptr::null_mut())],
            active: AtomicUsize::new(0),
            state: parking_lot::Mutex::new(LogsState::default()),
        }
    }

    /// Slot new rows are appended to.
    pub fn active_slot(&self) -> usize { self.active.load(Acquire) }

    /// The log new rows are appended to.
    pub fn active(&self) -> &CodeLog {
        self.get(self.active_slot()).expect("the active code log slot is always populated")
    }

    /// The log in `slot`. Callers either hold the graph writer or a
    /// generation guard taken before they read a tag naming this slot; a
    /// log is freed only after both are gone ([`Self::reclaim`]).
    #[inline]
    pub fn get(&self, slot: usize) -> Option<&CodeLog> {
        let pointer = self.slots.get(slot)?.load(Acquire);
        // SAFETY: see above; the box is only freed by `reclaim`/`drop`.
        (!pointer.is_null()).then(|| unsafe { &*pointer })
    }

    /// Both logs, active first.
    pub fn each(&self) -> impl Iterator<Item = &CodeLog> {
        let active = self.active_slot();
        [active, 1 - active].into_iter().filter_map(move |slot| self.get(slot))
    }

    /// The log being compacted into the active one (`None` when idle).
    pub fn compacting(&self) -> Option<(&CodeLog, usize)> {
        let cursor = self.state.lock().cursor?;
        Some((self.get(1 - self.active_slot())?, cursor))
    }

    /// Whether one more shed row may be appended to the active log. While a
    /// compaction copies into it, room for every row still to be copied
    /// from the old log stays reserved (copying moves a row from the old
    /// log's live count to the target's rows, so the sum never grows), so
    /// ordinary shedding can never fill the target and stall the copy.
    pub fn shed_room(&self) -> bool {
        let active = self.active();
        let Some(capacity) = active.capacity() else { return true };
        let reserved = self.compacting().map_or(0, |(old, _)| old.live_rows());
        active.rows().saturating_add(reserved) < capacity
    }

    /// Whether a new compaction may start: none running and the previous
    /// log fully reclaimed (its slot is reused by the next target).
    pub fn idle(&self) -> bool {
        let state = self.state.lock();
        state.cursor.is_none() && state.retiring.is_none() && self.get(1 - self.active_slot()).is_none()
    }

    /// Make `target` the active log; the current one becomes the
    /// compaction source. Caller holds the graph writer and checked
    /// [`Self::idle`].
    pub fn begin_compaction(&self, target: CodeLog) -> Result<()> {
        let mut state = self.state.lock();
        let next = 1 - self.active_slot();
        if state.cursor.is_some() || state.retiring.is_some() || !self.slots[next].load(Acquire).is_null() {
            return Err(failure("code log compaction already in progress"));
        }
        self.slots[next].store(Box::into_raw(Box::new(target)), Release);
        self.active.store(next, Release);
        state.cursor = Some(0);
        Ok(())
    }

    /// Record migration progress (caller holds the graph writer).
    pub fn advance(&self, cursor: usize) {
        if let Some(current) = self.state.lock().cursor.as_mut() { *current = cursor; }
    }

    /// Every slot was migrated: retire the previous log at `generation`.
    pub fn finish_compaction(&self, generation: u64) {
        let mut state = self.state.lock();
        if state.cursor.take().is_some() {
            state.retiring = Some((1 - self.active_slot(), generation));
        }
    }

    /// Free the switched-out log once no reader older than its retirement
    /// remains, and reclaim each log's retired buffers.
    pub fn reclaim(&self, oldest: u64) {
        let mut state = self.state.lock();
        if let Some((slot, _)) = state.retiring.filter(|(_, generation)| *generation < oldest) {
            state.retiring = None;
            let pointer = self.slots[slot].swap(std::ptr::null_mut(), std::sync::atomic::Ordering::AcqRel);
            // SAFETY: allocated by `Box::into_raw`; no reader older than
            // the retirement remains, newer ones never see a tag naming this
            // slot until the next compaction, and unguarded diagnostics
            // ([`Self::persisted_files`], [`Self::memory_bytes`]) hold this
            // state lock, as here, while they touch a log.
            if !pointer.is_null() { drop(unsafe { Box::from_raw(pointer) }); }
        }
        for log in self.each() { log.reclaim(oldest); }
    }

    /// Chunk files of both logs (a retiring log's files may still be
    /// referenced by a checkpoint being written).
    pub fn persisted_files(&self) -> Vec<String> {
        let _state = self.state.lock();
        self.each().flat_map(CodeLog::persisted_files).collect()
    }

    /// (anonymous buffer bytes, mapped file bytes) of both logs.
    pub fn memory_bytes(&self) -> (usize, usize) {
        let _state = self.state.lock();
        self.each().map(CodeLog::memory_bytes).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

impl Drop for CodeLogs {
    fn drop(&mut self) {
        for slot in &self.slots {
            let pointer = slot.swap(std::ptr::null_mut(), std::sync::atomic::Ordering::AcqRel);
            // SAFETY: allocated by `Box::into_raw`; `&mut self` excludes readers.
            if !pointer.is_null() { drop(unsafe { Box::from_raw(pointer) }); }
        }
    }
}

// SAFETY: the raw pointers own `CodeLog`s (Send + Sync) exactly like boxes.
unsafe impl Send for CodeLogs {}
unsafe impl Sync for CodeLogs {}

impl Drop for CodeLog {
    fn drop(&mut self) {
        let mapped: usize = self.state.get_mut().backing.iter().flatten()
            .map(|b| if let Backing::Mapped(file) = b { file.len() } else { 0 }).sum();
        crate::accounting::CODE_LOG_MAPPED_BYTES.fetch_sub(mapped as u64, Relaxed);
        for page in self.chunks.iter() {
            let page = page.swap(std::ptr::null_mut(), Relaxed);
            // SAFETY: allocated by `Box::into_raw` in `ensure_chunk_slot`;
            // `&mut self` excludes readers.
            if !page.is_null() { drop(unsafe { Box::from_raw(page) }); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::{Directory, RamDirectory};

    fn row(i: u32, words: usize) -> Vec<u32> { (0..words as u32).map(|w| i.wrapping_mul(31).wrapping_add(w)).collect() }

    #[test]
    fn appends_read_back_across_chunks_seal_and_persist_only_new_rows() {
        let _budget = crate::admission::private_shared(256 << 20);
        let dir = RamDirectory::new();
        let log = CodeLog::new().unwrap();
        // ~16 MiB / (1024 * 4) = 4096 rows per chunk.
        let words = 1024;
        let per = CHUNK_BYTES / (words * 4);
        let total = per * 2 + 17;
        for i in 0..total as u32 { assert_eq!(log.append(&row(i, words)).unwrap(), u64::from(i)); }
        assert_eq!(log.rows_per_chunk(), per);
        for i in [0u32, 1, per as u32 - 1, per as u32, total as u32 - 1] {
            let got: Vec<u32> = log.row(u64::from(i)).unwrap().iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, row(i, words));
        }
        assert!(log.row(total as u64 * 10).is_none());
        // Two full chunks seal into files; the newest stays anonymous.
        let sealed = log.seal_full(&dir).unwrap();
        assert_eq!(sealed.len(), 2);
        log.install_sealed(sealed, 1);
        let (anon, mapped) = log.memory_bytes();
        assert_eq!(mapped, 2 * per * words * 4);
        assert!(anon >= per * words * 4);
        log.reclaim(2);
        assert_eq!(log.memory_bytes().0, per * words * 4, "retired buffers freed after the generation passes");
        let first: Vec<u32> = log.row(1).unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(first, row(1, words), "rows read through the mapping");
        // A checkpoint reuses the sealed files and writes only the prefix.
        let refs = log.persist(&dir, total as u64).unwrap();
        assert_eq!(refs, vec![ChunkRef { chunk: 0, rows: per as u32 }, ChunkRef { chunk: 1, rows: per as u32 }, ChunkRef { chunk: 2, rows: 17 }]);
        let names = log.file_names(&refs);
        assert!(names.iter().all(|n| is_code_log_file(n) && dir.exists(n)));
        // More rows: the next persistence adds one new prefix file only.
        for i in total as u32..total as u32 + 5 { log.append(&row(i, words)).unwrap(); }
        let before = dir.list().unwrap();
        let refs2 = log.persist(&dir, total as u64 + 5).unwrap();
        assert_eq!(refs2[..2], refs[..2]);
        assert_eq!(refs2[2], ChunkRef { chunk: 2, rows: 22 });
        let added: Vec<String> = dir.list().unwrap().into_iter().filter(|n| !before.contains(n)).collect();
        assert_eq!(added, vec![chunk_file_name(log.id(), 2, 22)]);
        // Rows already covered by a file are never rewritten.
        assert_eq!(log.persist(&dir, 3).unwrap(), vec![ChunkRef { chunk: 0, rows: per as u32 }]);

        let restored = CodeLog::restore(&dir, log.id(), words, per, &refs2).unwrap();
        assert_eq!(restored.rows(), total as u64 + 5);
        for i in [0u32, per as u32 + 3, total as u32 + 4] {
            let got: Vec<u32> = restored.row(u64::from(i)).unwrap().iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, row(i, words));
        }
        // Appending continues in the restored prefix chunk.
        assert_eq!(restored.append(&row(99, words)).unwrap(), total as u64 + 5);
        assert!(restored.append(&row(1, words + 1)).is_err(), "row width is fixed");
        // A missing file or a bad sequence is corruption, not an empty log.
        assert!(CodeLog::restore(&dir, log.id(), words, per, &[ChunkRef { chunk: 0, rows: 3 }, ChunkRef { chunk: 1, rows: 3 }]).is_err());
        assert!(CodeLog::restore(&dir, log.id() ^ 1, words, per, &refs2).is_err());
    }

    #[test]
    fn dead_rows_and_capacity_drive_compaction_and_fullness() {
        let _budget = crate::admission::private_shared(256 << 20);
        let log = CodeLog::new().unwrap();
        assert!(log.capacity().is_none() && !log.is_full() && !log.wants_compaction(0));
        let words = 1024;
        for i in 0..100 { log.append(&row(i, words)).unwrap(); log.tagged(); }
        let per = (CHUNK_BYTES / (words * 4)) as u64;
        assert_eq!(log.capacity(), Some((per * MAX_CHUNKS as u64).min(MAX_ROWS + 1)));
        assert!(!log.wants_compaction(10), "all live");
        for _ in 0..49 { log.untagged(); }
        assert!(!log.wants_compaction(10), "49% dead");
        log.untagged();
        assert!(log.wants_compaction(10), "half dead");
        assert!(!log.wants_compaction(101), "below the minimum size");
        // Untagging never underflows.
        for _ in 0..200 { log.untagged(); }
        assert_eq!(log.live_rows(), 0);
    }

    #[test]
    fn a_near_full_log_compacts_only_when_the_copy_lands_at_most_half_full() {
        let _budget = crate::admission::private_shared(256 << 20);
        let log = CodeLog::new().unwrap();
        for i in 0..80 { log.append(&row(i, 8)).unwrap(); log.tagged(); }
        log.set_test_capacity(100);
        // 80% full, 30 dead: the old rule (an eighth dead) compacted into a
        // log 50% full that soon re-triggered; 50 live is now the most.
        for _ in 0..30 { log.untagged(); }
        assert_eq!(log.live_rows(), 50);
        assert!(log.wants_compaction(1), "near full, the copy lands at half capacity");
        log.tagged();
        assert!(!log.wants_compaction(1), "51 live of 100: the copy would land over half full");
        // Below three quarters the half-dead rule alone applies.
        let small = CodeLog::new().unwrap();
        for i in 0..70 { small.append(&row(i, 8)).unwrap(); small.tagged(); }
        small.set_test_capacity(100);
        for _ in 0..30 { small.untagged(); }
        assert!(!small.wants_compaction(1));
    }

    #[test]
    fn shedding_during_compaction_keeps_room_for_the_rows_still_to_copy() {
        let _budget = crate::admission::private_shared(256 << 20);
        let old = CodeLog::new().unwrap();
        for i in 0..8 { old.append(&row(i, 8)).unwrap(); old.tagged(); }
        let logs = CodeLogs::new(old);
        assert!(logs.shed_room());
        let target = CodeLog::new().unwrap();
        target.append(&row(0, 8)).unwrap(); // fixes the shape
        target.set_test_capacity(10);
        logs.begin_compaction(target).unwrap();
        // 1 row in the target + 8 still to copy < 10: one shed row fits.
        assert!(logs.shed_room());
        logs.active().append(&row(1, 8)).unwrap();
        assert!(!logs.shed_room(), "the last free row is the copy's");
        // Copying a row moves it from the old log's live count to the target.
        let (old, _) = logs.compacting().unwrap();
        logs.active().append(&row(2, 8)).unwrap();
        old.untagged();
        assert!(!logs.shed_room(), "copying never frees shedding room");
        // Without a compaction only fullness counts.
        let idle = CodeLogs::new(CodeLog::new().unwrap());
        idle.active().append(&row(0, 8)).unwrap();
        idle.active().set_test_capacity(2);
        assert!(idle.shed_room());
        idle.active().append(&row(1, 8)).unwrap();
        assert!(!idle.shed_room());
    }

    #[test]
    fn the_chunk_directory_grows_past_one_page_to_the_full_row_space() {
        let _budget = crate::admission::private_shared(256 << 20);
        let log = CodeLog::new().unwrap();
        log.append(&row(0, 1024)).unwrap();
        // 4096 rows per chunk: one page (4096 chunks) used to be the limit,
        // 16M rows; the paged directory reaches the 30-bit row space.
        assert_eq!(log.capacity(), Some(MAX_ROWS + 1));
        assert!(log.chunk_slot(PAGE_CHUNKS + 5).is_none(), "pages are allocated on demand");
        let slot = log.ensure_chunk_slot(PAGE_CHUNKS + 5).unwrap() as *const _;
        assert_eq!(log.chunk_slot(PAGE_CHUNKS + 5).unwrap() as *const _, slot);
        assert!(log.chunk_slot(PAGE_CHUNKS * 2).is_none());
        assert!(log.ensure_chunk_slot(MAX_CHUNKS).is_err());
        // The first page still serves the rows appended before.
        let got: Vec<u32> = log.row(0).unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, row(0, 1024));
    }

    #[test]
    fn code_logs_switch_keeps_the_old_log_readable_until_reclaimed() {
        let _budget = crate::admission::private_shared(256 << 20);
        let words = 8;
        let first = CodeLog::new().unwrap();
        let first_id = first.id();
        for i in 0..10 { first.append(&row(i, words)).unwrap(); first.tagged(); }
        let logs = CodeLogs::new(first);
        assert_eq!(logs.active_slot(), 0);
        assert!(logs.idle() && logs.compacting().is_none());
        logs.begin_compaction(CodeLog::new().unwrap()).unwrap();
        assert_eq!(logs.active_slot(), 1);
        assert_ne!(logs.active().id(), first_id, "the target is a fresh log id");
        assert!(!logs.idle());
        assert!(logs.begin_compaction(CodeLog::new().unwrap()).is_err(), "one compaction at a time");
        let (old, cursor) = logs.compacting().unwrap();
        assert_eq!((old.id(), cursor), (first_id, 0));
        // Copy a row; the old one stays readable for readers holding its tag.
        let copied: Vec<u32> = old.row(3).unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(logs.active().append(&copied).unwrap(), 0);
        logs.advance(10);
        assert_eq!(logs.compacting().unwrap().1, 10);
        logs.finish_compaction(5);
        assert!(logs.compacting().is_none());
        logs.reclaim(5);
        assert!(logs.get(0).is_some(), "a reader at the retirement generation may still hold old tags");
        let read: Vec<u32> = logs.get(0).unwrap().row(3).unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(read, row(3, words));
        assert!(!logs.idle(), "the next compaction waits for reclamation");
        logs.reclaim(6);
        assert!(logs.get(0).is_none(), "freed once every older reader is gone");
        assert!(logs.idle());
        assert_eq!(logs.each().count(), 1);
        // The next compaction reuses the freed slot.
        logs.begin_compaction(CodeLog::new().unwrap()).unwrap();
        assert_eq!(logs.active_slot(), 0);
    }
}
