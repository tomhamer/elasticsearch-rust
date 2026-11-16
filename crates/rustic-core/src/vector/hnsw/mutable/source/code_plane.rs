//! A quantized field's codes by graph node id, at a fixed stride: what the
//! traversal scores, one computed address per neighbour (no source
//! directory or row lookup). Vespa keeps a tensor attribute's vectors dense
//! by docid beside its HNSW graph; this is the same layout for the codes.
//!
//! The source rows and the code log stay the identity and durable form of a
//! node's codes (checkpoints, recovery, captures and maintenance read them);
//! the plane is the in-memory copy searches read. An entry is written once,
//! when a node's row is installed (a new node or a restored one), by the
//! graph's single writer and before the node is published, and is never
//! rewritten while the node can be reached: a slot is reused only after every
//! reader that could see its previous node has drained (the generation
//! domain's reclamation). A reader that acquired a node therefore reads its
//! entry through that publication's happens-before edge. Entries are never
//! freed until the owner is: chunks only grow.
//!
//! Chunks double from [`FIRST`] slots up to [`BIG`] slots and are then all
//! [`BIG`] slots (as the row plane's), reached through a two-level table:
//! a large field allocates, and is charged for, at most one chunk ahead.
//! (Chunks that kept doubling charged half the field's codes in one step,
//! e.g. ~12 GB at 64M 128-d int8 nodes, on every shard at about the same
//! time under uniform routing.)
//!
//! Entry layout (words): `[epoch lo, epoch hi, corrections and codes...]`,
//! padded to whole 64-byte lines. Epoch 0 (never a real epoch) marks an
//! entry that was not written; a reader falls back to the source row.

use super::super::huge::ZeroedBlock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

/// Slots of the first two chunks; chunk `k` in `1..SMALL` holds
/// `FIRST << (k - 1)` slots.
const FIRST: u32 = 1 << 10;
/// Slots of every chunk from id `BIG` on.
const BIG: u32 = 1 << 16;
/// Doubling chunks below `BIG`: 0 and 1 hold `FIRST`, then 2^11..2^15.
const SMALL: usize = 7;
/// Big chunks per second-level table.
const LEAF: usize = 256;
/// Second-level tables covering every `u32` id from `BIG` on.
const ROOT: usize = ((1usize << 32) / BIG as usize).div_ceil(LEAF);

/// Node `id`'s chunk (`< SMALL`: a doubling chunk; else `SMALL + big`
/// index) and its index in the chunk.
#[inline]
fn locate(id: u32) -> (usize, usize) {
    if id >= BIG {
        let big = (id / BIG - 1) as usize;
        return (SMALL + big, (id % BIG) as usize);
    }
    let q = id >> FIRST.trailing_zeros();
    if q == 0 { return (0, id as usize); }
    let k = (32 - q.leading_zeros()) as usize;
    (k, (id - (FIRST << (k - 1))) as usize)
}
/// Slots of chunk `k`.
#[inline]
fn chunk_slots(k: usize) -> usize {
    match k {
        0 => FIRST as usize,
        k if k < SMALL => (FIRST as usize) << (k - 1),
        _ => BIG as usize,
    }
}

struct Chunk {
    words: *mut AtomicU32,
    /// Owns `words` (huge pages when large).
    _block: ZeroedBlock,
    _memory: Option<crate::admission::Reservation>,
}
type Leaf = [AtomicPtr<Chunk>; LEAF];

pub(super) struct CodePlane {
    /// Words per entry (0 until the first write fixes it).
    stride: AtomicUsize,
    /// Chunk `k` covers the ids [`locate`] maps to it; chunks and leaves
    /// are published with Release once allocated, never replaced.
    small: [AtomicPtr<Chunk>; SMALL],
    root: Box<[AtomicPtr<Leaf>]>,
    bytes: AtomicUsize,
}

// SAFETY: chunks are only written by the graph's single writer (see the
// module notes) and published through atomics; readers only read entries
// whose writes happen before the node publication they acquired.
unsafe impl Send for CodePlane {}
unsafe impl Sync for CodePlane {}

impl Default for CodePlane {
    fn default() -> Self {
        let mut root = Vec::with_capacity(ROOT);
        root.resize_with(ROOT, || AtomicPtr::new(std::ptr::null_mut()));
        Self { stride: AtomicUsize::new(0), small: Default::default(), root: root.into_boxed_slice(), bytes: AtomicUsize::new(0) }
    }
}

impl Drop for CodePlane {
    fn drop(&mut self) {
        let free = |entry: &AtomicPtr<Chunk>| {
            let chunk = entry.swap(std::ptr::null_mut(), Ordering::AcqRel);
            // SAFETY: created by `Box::into_raw` in `chunk`, dropped once.
            if !chunk.is_null() { drop(unsafe { Box::from_raw(chunk) }); }
        };
        self.small.iter().for_each(free);
        for leaf in self.root.iter() {
            let leaf = leaf.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if leaf.is_null() { continue; }
            // SAFETY: created by `Box::into_raw` in `chunk`, dropped once.
            let leaf = unsafe { Box::from_raw(leaf) };
            leaf.iter().for_each(free);
        }
    }
}

impl CodePlane {
    /// Heap bytes of the allocated chunks (and the chunk table).
    pub(super) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed) + self.root.len() * std::mem::size_of::<AtomicPtr<Leaf>>()
    }

    /// Chunk `k` (null: never allocated).
    #[inline]
    fn chunk_ptr(&self, k: usize) -> *const Chunk {
        if k < SMALL { return self.small[k].load(Ordering::Acquire); }
        let big = k - SMALL;
        let Some(leaf) = self.root.get(big / LEAF) else { return std::ptr::null() };
        let leaf = leaf.load(Ordering::Acquire);
        if leaf.is_null() { return std::ptr::null(); }
        // SAFETY: a published leaf lives as long as the plane.
        unsafe { (*leaf)[big % LEAF].load(Ordering::Acquire) }
    }

    /// Words per entry, `0` before the first write.
    #[inline]
    pub(super) fn stride(&self) -> usize {
        self.stride.load(Ordering::Acquire)
    }

    /// Write node `id`'s entry: its codes' `epoch` and `words` (corrections
    /// and codes, as a source row holds them after its header). Only the
    /// graph's writer calls this, before `id` is published.
    pub(super) fn write(&self, id: u32, epoch: u64, words: &[u32]) -> crate::error::Result<()> {
        let width = (words.len() + 2).div_ceil(16) * 16;
        let stride = match self.stride.load(Ordering::Acquire) {
            0 => { self.stride.store(width, Ordering::Release); width }
            s => s,
        };
        if words.len() + 2 > stride || epoch == 0 {
            return Err(super::failure("code plane entry does not fit its stride"));
        }
        let (k, index) = locate(id);
        let chunk = self.chunk(k, stride)?;
        let base = index * stride;
        // SAFETY: `base + stride` is inside the chunk (`chunk_slots(k) * stride`
        // words); only the single writer stores, readers of this entry are
        // ordered after the node's publication.
        unsafe {
            let entry = (*chunk).words.add(base);
            for (i, &w) in words.iter().enumerate() {
                (*entry.add(2 + i)).store(w, Ordering::Relaxed);
            }
            (*entry).store(epoch as u32, Ordering::Relaxed);
            (*entry.add(1)).store((epoch >> 32) as u32, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Mark node `id`'s entry unwritten (no allocation). Only the writer
    /// calls this, for a slot no reader can reach.
    pub(super) fn clear(&self, id: u32) {
        let stride = self.stride();
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if stride == 0 || chunk.is_null() { return; }
        // SAFETY: inside the chunk; single writer.
        unsafe {
            let entry = (*chunk).words.add(index * stride);
            (*entry).store(0, Ordering::Relaxed);
            (*entry.add(1)).store(0, Ordering::Relaxed);
        }
    }

    fn chunk(&self, index: usize, stride: usize) -> crate::error::Result<*const Chunk> {
        let slot: &AtomicPtr<Chunk> = if index < SMALL { &self.small[index] } else {
            let big = index - SMALL;
            let root = self.root.get(big / LEAF).ok_or_else(|| super::failure("code plane id out of range"))?;
            let mut leaf = root.load(Ordering::Acquire);
            if leaf.is_null() {
                let fresh: Box<Leaf> = Box::new(std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())));
                leaf = Box::into_raw(fresh);
                // Single writer: nothing else installs this leaf.
                root.store(leaf, Ordering::Release);
                self.bytes.fetch_add(std::mem::size_of::<Leaf>(), Ordering::Relaxed);
            }
            // SAFETY: published leaf, lives as long as the plane.
            unsafe { &(*leaf)[big % LEAF] }
        };
        let current = slot.load(Ordering::Acquire);
        if !current.is_null() { return Ok(current); }
        let bytes = chunk_slots(index) * stride * 4;
        // Zeroed memory is a valid `AtomicU32` array and marks every entry
        // unwritten (epoch 0). Past the budget's spill threshold, a spill
        // file (see `huge::plane_chunk`).
        let (block, memory) = super::super::huge::plane_chunk(bytes, "mutable code plane")?;
        let words = block.ptr().cast::<AtomicU32>();
        let chunk = Box::into_raw(Box::new(Chunk { words, _block: block, _memory: memory }));
        // Single writer: nothing else installs this chunk.
        slot.store(chunk, Ordering::Release);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(chunk)
    }

    /// Node `id`'s entry (`stride` words), if its chunk exists.
    #[inline]
    pub(super) fn entry(&self, id: u32, stride: usize) -> Option<&[f32]> {
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if chunk.is_null() { return None; }
        let base = index * stride;
        // SAFETY: inside the chunk; the entry's writes happen before the
        // caller's acquisition of node `id` (module notes). `AtomicU32` and
        // `f32` share size and alignment.
        Some(unsafe { std::slice::from_raw_parts((*chunk).words.add(base).cast::<f32>(), stride) })
    }

    /// Warm node `id`'s entry (no read).
    #[inline]
    pub(super) fn prefetch(&self, id: u32, stride: usize) {
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if chunk.is_null() { return; }
        // SAFETY: address arithmetic inside the chunk; prefetch never faults.
        let address = unsafe { (*chunk).words.add(index * stride) };
        crate::vector::distance::prefetch(address.cast::<u8>(), stride * 4);
    }
}

impl CodePlane {
    /// The traversal's view (fixed stride; call once the stride is set).
    #[inline]
    pub(super) fn view(&self) -> CodePlaneRef<'_> {
        CodePlaneRef { plane: self, stride: self.stride() }
    }
}

/// A [`CodePlane`] as traversal prefetches it (see
/// [`MutableVectorSource::dense_layout`](crate::vector::hnsw::MutableVectorSource::dense_layout));
/// the bound query scorer reads the entries.
#[derive(Clone, Copy)]
pub struct CodePlaneRef<'a> {
    plane: &'a CodePlane,
    stride: usize,
}
impl CodePlaneRef<'_> {
    /// Warm node `id`'s entry, reading nothing.
    #[inline]
    pub(crate) fn prefetch(self, id: u32) {
        self.plane.prefetch(id, self.stride);
    }
    /// Address of node `id`'s entry (0: its chunk was never allocated),
    /// warmed: what the bound scorer's `score_entries` reads.
    #[inline]
    pub(crate) fn entry_address(self, id: u32) -> usize {
        let (k, index) = locate(id);
        let chunk = self.plane.chunk_ptr(k);
        if chunk.is_null() { return 0; }
        // SAFETY: inside the chunk; prefetch never faults.
        let address = unsafe { (*chunk).words.add(index * self.stride) };
        crate::vector::distance::prefetch(address.cast::<u8>(), self.stride * 4);
        address as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip_across_chunks_and_unwritten_entries_read_as_empty() {
        let _budget = crate::admission::private_shared(256 << 20);
        let plane = CodePlane::default();
        assert_eq!(plane.stride(), 0);
        let words: Vec<u32> = (0..28).collect();
        for (id, chunk) in [(1u32, 0), (2, 0), (FIRST + 5, 1), (4 * FIRST, 3), (BIG, SMALL), (3 * BIG + 7, SMALL + 2), (u32::MAX, SMALL + (u32::MAX / BIG - 1) as usize)] {
            assert_eq!(locate(id).0, chunk);
            assert!(locate(id).1 < chunk_slots(chunk));
            if id != u32::MAX { plane.write(id, 7 + u64::from(id) << 32 | 3, &words).unwrap(); }
        }
        let stride = plane.stride();
        assert_eq!(stride, 32);
        let entry = plane.entry(FIRST + 5, stride).unwrap();
        assert_eq!(entry[2..30].iter().map(|w| w.to_bits()).collect::<Vec<_>>(), words);
        assert_eq!(u64::from(entry[0].to_bits()) | u64::from(entry[1].to_bits()) << 32, 7 + u64::from(FIRST + 5) << 32 | 3);
        // An unwritten entry of an allocated chunk: epoch 0.
        assert_eq!(plane.entry(3, stride).unwrap()[0].to_bits(), 0);
        // A chunk never allocated: none.
        assert!(plane.entry(2 * FIRST, stride).is_none());
        assert_eq!(locate(2 * FIRST).0, 2);
        assert!(plane.write(9, 0, &words).is_err(), "epoch 0 marks an empty entry");
        assert!(plane.write(9, 1, &[0; 40]).is_err(), "a wider entry than the stride");
        // Past the doubling chunks every chunk is BIG slots: one write far
        // out allocates (and charges) one bounded chunk.
        let before = plane.bytes();
        plane.write(200 * BIG + 1, 5, &words).unwrap();
        assert!(plane.bytes() - before <= BIG as usize * stride * 4 + std::mem::size_of::<Leaf>());
        assert_eq!(plane.entry(200 * BIG + 1, stride).unwrap()[2].to_bits(), 0);
        assert!(plane.entry(201 * BIG, stride).is_none());
    }
}
