//! An unquantized field's rows by graph node id, at a fixed stride: the
//! field's only in-memory copy of each row (it replaces the row's
//! [`ArrayStore`](super::ArrayStore) allocation), so a traversal step reads
//! one computed address instead of the source directory, the store's
//! descriptor and then the row. Vespa keeps a tensor attribute's vectors
//! dense by docid beside its HNSW graph; this is that layout.
//!
//! An entry holds the row exactly as the store would (the identity header,
//! then the padded f32 cells), placed so that the cells start on a 64-byte
//! line: a 96-d row's cells are exactly six lines, and the header sits in
//! the line before them, which traversal never reads. The source directory
//! slot of a plane row holds [`PLANE_REF`] so every other reader
//! (checkpoints, captures, maintenance, the identity checks) finds the same
//! encoded row as before.
//!
//! An entry is written once, when a node's row is installed, by the graph's
//! single writer and before the node is published, and is never rewritten
//! while the node can be reached: a slot is reused only after every reader
//! that could see its previous node has drained (the generation domain's
//! reclamation, which also holds back captures). A reader that acquired a
//! node therefore reads its entry through that publication's happens-before
//! edge. Chunks are never freed until the plane is.
//!
//! Chunks double from [`FIRST`] slots up to [`BIG`] slots and are then all
//! [`BIG`] slots, reached through a two-level table allocated on demand, so
//! a large field over-allocates at most one chunk (its admission charge
//! included) and a small one allocates little.

use super::super::huge::ZeroedBlock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

/// A source directory slot whose row lives in the owner's [`RowPlane`].
/// Below the code-log tag and outside every ArrayStore reference (the
/// source store is configured with at most 511 buffers, so buffer 511 is
/// never allocated).
pub(super) const PLANE_REF: u32 = (1 << 31) - 1;

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
#[inline]
fn chunk_slots(k: usize) -> usize {
    match k {
        0 => FIRST as usize,
        k if k < SMALL => (FIRST as usize) << (k - 1),
        _ => BIG as usize,
    }
}

struct Chunk {
    bytes: *mut u8,
    /// Owns `bytes` (huge pages when large).
    _block: ZeroedBlock,
    _memory: crate::admission::Reservation,
    /// Node memory stats: source rows, as the row store charges them.
    _charge: crate::accounting::Charge,
}
type Leaf = [AtomicPtr<Chunk>; LEAF];

pub(super) struct RowPlane {
    /// Words of an encoded row (header and padded cells).
    width: usize,
    /// Bytes per entry: a whole number of lines holding the cells (from a
    /// line start) and the 32-byte header just before them.
    stride: usize,
    small: [AtomicPtr<Chunk>; SMALL],
    root: Box<[AtomicPtr<Leaf>]>,
    bytes: AtomicUsize,
}

// SAFETY: chunks are only written by the graph's single writer (module
// notes) and published through atomics; readers only read entries whose
// writes happen before the node publication they acquired.
unsafe impl Send for RowPlane {}
unsafe impl Sync for RowPlane {}

impl Drop for RowPlane {
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
            // SAFETY: created by `Box::into_raw` in `leaf`, dropped once.
            let leaf = unsafe { Box::from_raw(leaf) };
            leaf.iter().for_each(free);
        }
    }
}

impl RowPlane {
    /// A plane of `dim`-cell rows (the source's encoded row width).
    pub(super) fn new(dim: usize) -> Self {
        let width = dim.div_ceil(8) * 8 + 8;
        let stride = (32 + 4 * width).div_ceil(64) * 64;
        let mut root = Vec::with_capacity(ROOT);
        root.resize_with(ROOT, || AtomicPtr::new(std::ptr::null_mut()));
        Self { width, stride, small: Default::default(), root: root.into_boxed_slice(), bytes: AtomicUsize::new(0) }
    }

    /// Heap bytes of the allocated chunks and tables.
    pub(super) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed) + self.root.len() * std::mem::size_of::<AtomicPtr<Leaf>>()
    }

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

    /// Byte address of node `id`'s cells (`None`: its chunk was never
    /// allocated). The header is the 32 bytes before it.
    #[inline]
    fn cells(&self, id: u32) -> Option<*mut u8> {
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if chunk.is_null() { return None; }
        // SAFETY: inside the chunk (`chunk_slots(k) * stride` bytes).
        Some(unsafe { (*chunk).bytes.add(index * self.stride + 64) })
    }

    /// Write node `id`'s encoded row (`width` words: the identity header,
    /// then the padded cells). Only the graph's writer calls this, before
    /// `id` is published (or while no reader can reach it).
    pub(super) fn write(&self, id: u32, encoded: &[u32]) -> crate::error::Result<()> {
        if encoded.len() != self.width {
            return Err(super::failure("row plane entry width"));
        }
        let (k, index) = locate(id);
        let chunk = self.chunk(k)?;
        // SAFETY: inside the chunk; single writer; readers of this entry are
        // ordered after the node's publication.
        unsafe {
            let words = (*chunk).bytes.add(index * self.stride + 32).cast::<AtomicU32>();
            for (i, &w) in encoded.iter().enumerate() {
                (*words.add(i)).store(w, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn chunk(&self, k: usize) -> crate::error::Result<*const Chunk> {
        let slot: &AtomicPtr<Chunk> = if k < SMALL { &self.small[k] } else {
            let big = k - SMALL;
            let root = self.root.get(big / LEAF).ok_or_else(|| super::failure("row plane id out of range"))?;
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
        let bytes = chunk_slots(k) * self.stride;
        let memory = crate::admission::shared().reserve(bytes as u64, "mutable row plane")?;
        // Zeroed memory is a valid `AtomicU32` array and an entry that is
        // never read before it is written.
        let block = ZeroedBlock::new(bytes).ok_or_else(|| super::failure("row plane allocation"))?;
        let raw = block.ptr();
        let charge = crate::accounting::Charge::new(crate::accounting::Category::VectorRows, bytes as u64);
        let chunk = Box::into_raw(Box::new(Chunk { bytes: raw, _block: block, _memory: memory, _charge: charge }));
        slot.store(chunk, Ordering::Release);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(chunk)
    }

    /// Node `id`'s encoded row (`width` words), if its chunk exists. Read it
    /// only for a node acquired under a guard (module notes).
    #[inline]
    pub(super) fn entry(&self, id: u32) -> Option<&[f32]> {
        let cells = self.cells(id)?;
        // SAFETY: the header and cells lie inside the chunk; `AtomicU32` and
        // `f32` share size and alignment; the entry's writes happen before
        // the caller's acquisition of node `id`.
        Some(unsafe { std::slice::from_raw_parts(cells.sub(32).cast::<f32>(), self.width) })
    }

    /// The traversal's view: cells by id, no header.
    #[inline]
    pub(super) fn view(&self) -> RowPlaneRef<'_> {
        RowPlaneRef { plane: self }
    }
}

/// A [`RowPlane`] as traversal reads it (see
/// [`MutableVectorSource::dense_layout`](crate::vector::hnsw::MutableVectorSource::dense_layout)).
#[derive(Clone, Copy)]
pub struct RowPlaneRef<'a> {
    plane: &'a RowPlane,
}
impl<'a> RowPlaneRef<'a> {
    /// Node `id`'s cells: the f32 row of an acquired, visible node whose
    /// source this plane is (`None`: never written).
    #[inline]
    pub(crate) fn cells(self, id: u32, dim: usize) -> Option<&'a [f32]> {
        let cells = self.plane.cells(id)?;
        // SAFETY: as `RowPlane::entry`.
        Some(unsafe { std::slice::from_raw_parts(cells.cast::<f32>(), dim) })
    }
    /// Address of node `id`'s cells (null: never written), for
    /// [`cells`](Self::cells)'s caller that reads them later.
    #[inline]
    pub(crate) fn cells_ptr(self, id: u32) -> *const f32 {
        self.plane.cells(id).map_or(std::ptr::null(), |cells| cells.cast::<f32>().cast_const())
    }
    /// Warm node `id`'s cells (`bytes` of them), reading nothing.
    #[inline]
    pub(crate) fn prefetch(self, id: u32, bytes: usize) {
        if let Some(cells) = self.plane.cells(id) {
            crate::vector::distance::prefetch(cells, bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip_across_small_and_big_chunks() {
        let _budget = crate::admission::private_shared(256 << 20);
        for dim in [1usize, 7, 96, 100, 768] {
            let plane = RowPlane::new(dim);
            let width = dim.div_ceil(8) * 8 + 8;
            assert_eq!(plane.stride % 64, 0);
            assert!(plane.stride >= 32 + 4 * width);
            // Big chunks of wide rows exceed the test memory budget.
            let all = [1u32, 2, FIRST - 1, FIRST, FIRST + 5, 4 * FIRST, BIG - 1, BIG, BIG + 1, 3 * BIG + 7, 300 * BIG + 1, u32::MAX];
            let ids: Vec<u32> = all.iter().copied().filter(|&id| dim <= 100 || id < BIG || id == u32::MAX).collect();
            for &id in &ids {
                let (k, index) = locate(id);
                assert!(index < chunk_slots(k), "id {id}");
                if k >= SMALL { assert!((k - SMALL) / LEAF < ROOT); }
            }
            for &id in &ids[..ids.len() - 1] {
                let row: Vec<u32> = (0..width as u32).map(|w| w ^ id.wrapping_mul(2654435761)).collect();
                plane.write(id, &row).unwrap();
            }
            for &id in &ids[..ids.len() - 1] {
                let row: Vec<u32> = (0..width as u32).map(|w| w ^ id.wrapping_mul(2654435761)).collect();
                let entry = plane.entry(id).unwrap();
                assert_eq!(entry.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), row, "dim {dim} id {id}");
                let cells = plane.view().cells(id, dim).unwrap();
                assert_eq!(cells.as_ptr() as usize % 64, 0, "cells start on a line");
                assert_eq!(cells.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), row[8..8 + dim]);
            }
            // A chunk never allocated: none.
            assert!(plane.entry(200 * BIG).is_none());
            assert!(plane.view().cells(u32::MAX, dim).is_none());
            assert!(plane.write(9, &vec![0; width + 1]).is_err());
        }
    }
}
