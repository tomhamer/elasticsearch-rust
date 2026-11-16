//! Zeroed allocations for the traversal's dense arrays (row and code
//! planes): at least 2 MiB is mapped anonymously and advised onto
//! transparent huge pages, so a traversal's random reads across a large
//! field take few TLB misses; smaller sizes use the global allocator.
//! `RUSTIC_KNN_THP=0` keeps every allocation on the global allocator.
//!
//! [`ZeroedBlock::file_backed`] maps an unlinked temporary file in the
//! node's spill directory ([`set_spill_dir`]) instead: page cache, which
//! the kernel writes back and evicts under memory pressure, for arrays
//! that are rebuilt rather than recovered (never part of a commit) and
//! that must not count against the resident budget.
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

/// Where file-backed blocks are mapped from (unset: none are).
static SPILL_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
/// Bytes mapped from spill files now.
static SPILLED: AtomicU64 = AtomicU64::new(0);

/// Let file-backed blocks map unlinked files in `dir` (created if
/// missing). The first call of the process wins (its nodes share the data
/// disk); leftovers of a previous process are unlinked files, gone with it.
pub fn set_spill_dir(dir: std::path::PathBuf) -> std::io::Result<()> {
    std::fs::create_dir_all(&dir)?;
    let _ = SPILL_DIR.set(dir);
    Ok(())
}

/// Bytes of blocks mapped from spill files now.
pub fn spilled_bytes() -> u64 {
    SPILLED.load(Ordering::Relaxed)
}

const HUGE: usize = 2 << 20;

fn huge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cfg!(target_os = "linux") && std::env::var("RUSTIC_KNN_THP").map_or(true, |v| v != "0"))
}

// Tests share one process: only a thread that opts in maps spill files
// (others keep testing refusals under small budgets).
#[cfg(test)]
thread_local! { static TEST_SPILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// Test seam: let this thread map spill files.
#[cfg(test)]
pub(super) fn spill_on_this_thread(on: bool) {
    TEST_SPILL.with(|flag| flag.set(on));
}
#[cfg(test)]
thread_local! { static TEST_NO_SPACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// Test seam: reserving a spill file's blocks fails on this thread (as on
/// a full disk or exhausted quota).
#[cfg(test)]
pub(super) fn spill_space_fails_on_this_thread(on: bool) {
    TEST_NO_SPACE.with(|flag| flag.set(on));
}

/// Smallest plane chunk that spills past the budget threshold.
const SPILL_MIN: usize = if cfg!(test) { 4096 } else { HUGE };

/// Percent of the admission limit above which large plane chunks map spill
/// files instead of being charged (`RUSTIC_VECTOR_SPILL_PERCENT`, 85).
fn spill_percent() -> u64 {
    static PCT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PCT.get_or_init(|| std::env::var("RUSTIC_VECTOR_SPILL_PERCENT").ok().and_then(|v| v.parse().ok()).filter(|&v: &u64| v <= 100).unwrap_or(85))
}

/// A zeroed chunk of a vector plane (codes, rows, maintenance rows) and
/// its admission charge. Below the spill threshold of the shared budget it
/// is resident and charged; past it (and with a spill directory) it maps an
/// unlinked spill file and is not charged: the kernel writes back and evicts
/// its cold pages, so a node past its memory budget keeps indexing and
/// searching, slower (page faults on cold entries), instead of refusing
/// writes. The planes are rebuilt from the source rows and code log, never
/// recovered from these files.
pub(super) fn plane_chunk(bytes: usize, label: &str) -> crate::error::Result<(ZeroedBlock, Option<crate::admission::Reservation>)> {
    let budget = crate::admission::shared();
    if let Some(block) = spill_block(bytes, &budget) { return Ok((block, None)); }
    let memory = budget.reserve(bytes as u64, label)?;
    let block = ZeroedBlock::new(bytes).ok_or_else(|| crate::error::RusticError::Memory(format!("{label} allocation")))?;
    Ok((block, Some(memory)))
}

/// `bytes` zeroed bytes mapped from a spill file, if an array of that size
/// charged to `budget` should spill now (large, and past the spill
/// threshold; see [`plane_chunk`]); `None`: allocate and charge it.
pub(super) fn spill_block(bytes: usize, budget: &crate::admission::Budget) -> Option<ZeroedBlock> {
    if !spill_due(bytes, budget) { return None; }
    spilled(bytes)
}

/// Whether an array of `bytes` charged to `budget` should spill now: large,
/// and the budget would pass its spill threshold with it.
pub(super) fn spill_due(bytes: usize, budget: &crate::admission::Budget) -> bool {
    bytes >= SPILL_MIN && budget.used().saturating_add(bytes as u64) > budget.limit() / 100 * spill_percent()
}

/// `bytes` zeroed bytes mapped from a spill file (logged once per process).
pub(super) fn spilled(bytes: usize) -> Option<ZeroedBlock> {
    let block = ZeroedBlock::file_backed(bytes)?;
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::warn!(spilled_bytes = spilled_bytes(), "vector memory budget nearly used: large vector arrays now map disk-backed spill files (slower, not refused)");
    }
    Some(block)
}

/// A zeroed, 64-byte aligned allocation, freed on drop.
pub(super) struct ZeroedBlock {
    ptr: *mut u8,
    kind: Kind,
}
enum Kind {
    Heap(Layout),
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Mapped { base: *mut u8, len: usize },
    /// A shared mapping of an unlinked spill file.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Spilled { base: *mut u8, len: usize },
}
// SAFETY: plain owned memory.
unsafe impl Send for ZeroedBlock {}
unsafe impl Sync for ZeroedBlock {}

impl ZeroedBlock {
    /// `bytes` zeroed bytes (nonzero), or `None` if the memory is unavailable.
    pub(super) fn new(bytes: usize) -> Option<Self> {
        if bytes == 0 { return None; }
        #[cfg(target_os = "linux")]
        if bytes >= HUGE && huge_enabled() {
            // Room to align the start to a huge page.
            let len = bytes.div_ceil(HUGE) * HUGE + HUGE;
            // SAFETY: a fresh private anonymous mapping (zero-filled on
            // first touch); nothing else refers to it.
            let base = unsafe {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
            };
            if base != libc::MAP_FAILED {
                let base = base.cast::<u8>();
                let ptr = base.wrapping_add((HUGE - base as usize % HUGE) % HUGE);
                // Advice only: without huge pages the mapping still works.
                // SAFETY: inside the mapping.
                unsafe { libc::madvise(ptr.cast(), len - (ptr as usize - base as usize), libc::MADV_HUGEPAGE) };
                return Some(Self { ptr, kind: Kind::Mapped { base, len } });
            }
        }
        let layout = Layout::from_size_align(bytes, 64).ok()?;
        // SAFETY: nonzero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        (!ptr.is_null()).then_some(Self { ptr, kind: Kind::Heap(layout) })
    }
    /// `bytes` zeroed bytes mapped from a new unlinked file in the spill
    /// directory (`None`: no directory set, or the file or mapping could
    /// not be made; the caller then allocates as usual).
    #[cfg(target_os = "linux")]
    pub(super) fn file_backed(bytes: usize) -> Option<Self> {
        use std::os::unix::ffi::OsStrExt;
        if bytes == 0 { return None; }
        #[cfg(test)]
        if !TEST_SPILL.with(std::cell::Cell::get) { return None; }
        let dir = std::ffi::CString::new(SPILL_DIR.get()?.as_os_str().as_bytes()).ok()?;
        let len = bytes.div_ceil(4096) * 4096;
        // SAFETY: plain syscalls on a descriptor this function owns; the
        // mapping keeps the file alive after the descriptor closes.
        unsafe {
            let fd = libc::open(dir.as_ptr(), libc::O_TMPFILE | libc::O_RDWR | libc::O_CLOEXEC, 0o600);
            if fd < 0 { return None; }
            if libc::ftruncate(fd, len as libc::off_t) != 0 { libc::close(fd); return None; }
            // The file's blocks are reserved before the mapping is used: a
            // write fault on a sparse page the filesystem cannot back
            // (ENOSPC, EDQUOT) would SIGBUS the node. No space: no spill,
            // the caller allocates admitted memory (or is refused).
            #[cfg(test)]
            let reserved = if TEST_NO_SPACE.with(std::cell::Cell::get) { libc::ENOSPC } else { libc::posix_fallocate(fd, 0, len as libc::off_t) };
            #[cfg(not(test))]
            let reserved = libc::posix_fallocate(fd, 0, len as libc::off_t);
            if reserved != 0 { libc::close(fd); return None; }
            let base = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
            libc::close(fd);
            if base == libc::MAP_FAILED { return None; }
            // Random single-row reads: no readahead around them.
            libc::madvise(base, len, libc::MADV_RANDOM);
            SPILLED.fetch_add(len as u64, Ordering::Relaxed);
            Some(Self { ptr: base.cast(), kind: Kind::Spilled { base: base.cast(), len } })
        }
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn file_backed(_bytes: usize) -> Option<Self> { None }

    /// Whether this block maps a spill file (not resident memory).
    pub(super) fn is_file_backed(&self) -> bool {
        matches!(self.kind, Kind::Spilled { .. })
    }

    #[inline]
    pub(super) fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}
impl Drop for ZeroedBlock {
    fn drop(&mut self) {
        match self.kind {
            // SAFETY: allocated with this layout.
            Kind::Heap(layout) => unsafe { dealloc(self.ptr, layout) },
            // SAFETY: the whole mapping, unmapped once.
            #[cfg(target_os = "linux")]
            Kind::Mapped { base, len } => unsafe { libc::munmap(base.cast(), len); },
            // SAFETY: as `Mapped`; the unlinked file goes with its mapping.
            #[cfg(target_os = "linux")]
            Kind::Spilled { base, len } => unsafe {
                libc::munmap(base.cast(), len);
                SPILLED.fetch_sub(len as u64, Ordering::Relaxed);
            },
            #[cfg(not(target_os = "linux"))]
            Kind::Mapped { .. } | Kind::Spilled { .. } => {}
        }
    }
}

/// `len` zeroed `T`s in a [`ZeroedBlock`]: a fixed-size array whose
/// untouched pages are not resident (a large one is a `MAP_NORESERVE`
/// mapping, so its capacity costs address space until written). Only for
/// types whose all-zero bytes are a valid value (atomics, plain integers)
/// and that need no drop.
pub(super) struct Slab<T> {
    block: Option<ZeroedBlock>,
    len: usize,
    _t: std::marker::PhantomData<T>,
}
// SAFETY: owns its elements as a `Box<[T]>` would.
unsafe impl<T: Send> Send for Slab<T> {}
unsafe impl<T: Sync> Sync for Slab<T> {}
impl<T> Default for Slab<T> {
    fn default() -> Self { Self { block: None, len: 0, _t: std::marker::PhantomData } }
}
impl<T> Slab<T> {
    /// `len` zeroed elements (`None`: the memory is unavailable).
    pub(super) fn zeroed(len: usize) -> Option<Self> {
        const { assert!(!std::mem::needs_drop::<T>() && std::mem::align_of::<T>() <= 64) };
        let bytes = len.checked_mul(std::mem::size_of::<T>())?;
        if bytes == 0 { return Some(Self { len, ..Self::default() }); }
        Some(Self { block: Some(ZeroedBlock::new(bytes)?), len, _t: std::marker::PhantomData })
    }
    /// `len` zeroed elements as reserved address space only: `None` unless
    /// the array is a lazily backed mapping (Linux), so a caller reserving
    /// far ahead never takes heap memory for it.
    pub(super) fn reserved(len: usize) -> Option<Self> {
        let bytes = len.checked_mul(std::mem::size_of::<T>())?;
        if !cfg!(target_os = "linux") || bytes < HUGE || !huge_enabled() { return None; }
        let slab = Self::zeroed(len)?;
        slab.block.as_ref().is_some_and(|block| matches!(block.kind, Kind::Mapped { .. })).then_some(slab)
    }
    /// `len` elements in `block` (zeroed, at least that large).
    pub(super) fn from_block(block: ZeroedBlock, len: usize) -> Self {
        const { assert!(!std::mem::needs_drop::<T>() && std::mem::align_of::<T>() <= 64) };
        Self { block: Some(block), len, _t: std::marker::PhantomData }
    }
    /// Elements (fixed at creation).
    pub(super) fn capacity(&self) -> usize { self.len }
}
impl<T> std::ops::Deref for Slab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.block {
            // SAFETY: `len` zero-initialized (valid) elements, aligned.
            Some(block) => unsafe { std::slice::from_raw_parts(block.ptr().cast::<T>(), self.len) },
            None => &[],
        }
    }
}
impl<T> std::ops::DerefMut for Slab<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &self.block {
            // SAFETY: as `deref`; `&mut self` is the only borrow.
            Some(block) => unsafe { std::slice::from_raw_parts_mut(block.ptr().cast::<T>(), self.len) },
            None => &mut [],
        }
    }
}
impl<'a, T> IntoIterator for &'a Slab<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_and_huge_blocks_are_zeroed_aligned_and_writable() {
        for bytes in [64usize, 4096, 3 << 20, (5 << 20) + 192] {
            let block = ZeroedBlock::new(bytes).unwrap();
            assert_eq!(block.ptr() as usize % 64, 0);
            // SAFETY: inside the block.
            let s = unsafe { std::slice::from_raw_parts_mut(block.ptr(), bytes) };
            assert!(s.iter().step_by(4093).all(|&b| b == 0));
            s[bytes - 1] = 7;
            s[0] = 9;
            assert_eq!((s[0], s[bytes - 1]), (9, 7));
        }
        assert!(ZeroedBlock::new(0).is_none());
    }
}

#[cfg(all(test, target_os = "linux"))]
mod space_tests {
    /// #302: a spill file whose blocks cannot be reserved is not mapped:
    /// the chunk is allocated and charged instead (refused past the budget,
    /// a typed error), never a sparse mapping a write could SIGBUS on.
    #[test]
    fn no_space_for_a_spill_file_falls_back_to_admitted_memory() {
        super::set_spill_dir(std::env::temp_dir().join("rustic-vector-spill-space-test")).unwrap();
        super::spill_on_this_thread(true);
        let block = super::ZeroedBlock::file_backed(1 << 20).expect("a reserved spill file");
        assert!(block.is_file_backed());
        drop(block);
        super::spill_space_fails_on_this_thread(true);
        assert!(super::ZeroedBlock::file_backed(1 << 20).is_none(), "no unreserved mapping");
        {
            // Past the spill threshold with no space: charged, then refused.
            let _budget = crate::admission::private_shared(64 << 10);
            match super::plane_chunk(1 << 20, "spill test") {
                Err(crate::error::RusticError::Memory(_)) => {}
                Err(other) => panic!("{other:?}"),
                Ok((block, memory)) => panic!("allocated {} (file backed {}) with charge {:?}", 1 << 20, block.is_file_backed(), memory.map(|m| m.bytes())),
            }
        }
        super::spill_space_fails_on_this_thread(false);
        super::spill_on_this_thread(false);
    }
}
