//! Zeroed allocations for the traversal's dense arrays (row and code
//! planes): at least 2 MiB is mapped anonymously and advised onto
//! transparent huge pages, so a traversal's random reads across a large
//! field take few TLB misses; smaller sizes use the global allocator.
//! `RUSTIC_KNN_THP=0` keeps every allocation on the global allocator.
use std::alloc::{alloc_zeroed, dealloc, Layout};

const HUGE: usize = 2 << 20;

fn huge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cfg!(target_os = "linux") && std::env::var("RUSTIC_KNN_THP").map_or(true, |v| v != "0"))
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
            #[cfg(not(target_os = "linux"))]
            Kind::Mapped { .. } => {}
        }
    }
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
