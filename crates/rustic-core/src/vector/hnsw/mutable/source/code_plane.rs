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
//! Entry layout (words): `[epoch lo, epoch hi, corrections and codes...]`,
//! padded to whole 64-byte lines. Epoch 0 (never a real epoch) marks an
//! entry that was not written; a reader falls back to the source row.

use super::super::huge::ZeroedBlock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

/// Slots of the first two chunks; chunk `k >= 1` holds `FIRST << (k - 1)`
/// slots, so a small field allocates little and a large one few chunks.
const FIRST: u32 = 1 << 10;
/// Chunks covering every `u32` id.
const CHUNKS: usize = 33 - FIRST.trailing_zeros() as usize;

/// Node `id`'s chunk and its index in it.
#[inline]
fn locate(id: u32) -> (usize, usize) {
    let q = id >> FIRST.trailing_zeros();
    if q == 0 { return (0, id as usize); }
    let k = (32 - q.leading_zeros()) as usize;
    (k, (id - (FIRST << (k - 1))) as usize)
}
/// Slots of chunk `k`.
#[inline]
fn chunk_slots(k: usize) -> usize {
    if k == 0 { FIRST as usize } else { (FIRST as usize) << (k - 1) }
}

struct Chunk {
    words: *mut AtomicU32,
    /// Owns `words` (huge pages when large).
    _block: ZeroedBlock,
    _memory: crate::admission::Reservation,
}

pub(super) struct CodePlane {
    /// Words per entry (0 until the first write fixes it).
    stride: AtomicUsize,
    /// Chunk `k` covers the ids [`locate`] maps to it; published with Release
    /// once allocated, never replaced.
    table: Box<[AtomicPtr<Chunk>]>,
    bytes: AtomicUsize,
}

// SAFETY: chunks are only written by the graph's single writer (see the
// module notes) and published through atomics; readers only read entries
// whose writes happen before the node publication they acquired.
unsafe impl Send for CodePlane {}
unsafe impl Sync for CodePlane {}

impl Default for CodePlane {
    fn default() -> Self {
        let mut table = Vec::with_capacity(CHUNKS);
        table.resize_with(CHUNKS, || AtomicPtr::new(std::ptr::null_mut()));
        Self { stride: AtomicUsize::new(0), table: table.into_boxed_slice(), bytes: AtomicUsize::new(0) }
    }
}

impl Drop for CodePlane {
    fn drop(&mut self) {
        for entry in self.table.iter() {
            let chunk = entry.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !chunk.is_null() {
                // SAFETY: created by `Box::into_raw` in `chunk`, dropped once.
                drop(unsafe { Box::from_raw(chunk) });
            }
        }
    }
}

impl CodePlane {
    /// Heap bytes of the allocated chunks (and the chunk table).
    pub(super) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed) + self.table.len() * std::mem::size_of::<AtomicPtr<Chunk>>()
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
        let Some(entry) = self.table.get(k) else { return };
        let chunk = entry.load(Ordering::Acquire);
        if stride == 0 || chunk.is_null() { return; }
        // SAFETY: inside the chunk; single writer.
        unsafe {
            let entry = (*chunk).words.add(index * stride);
            (*entry).store(0, Ordering::Relaxed);
            (*entry.add(1)).store(0, Ordering::Relaxed);
        }
    }

    fn chunk(&self, index: usize, stride: usize) -> crate::error::Result<*const Chunk> {
        let slot = self.table.get(index).ok_or_else(|| super::failure("code plane id out of range"))?;
        let current = slot.load(Ordering::Acquire);
        if !current.is_null() { return Ok(current); }
        let bytes = chunk_slots(index) * stride * 4;
        let memory = crate::admission::shared().reserve(bytes as u64, "mutable code plane")?;
        // Zeroed memory is a valid `AtomicU32` array and marks every entry
        // unwritten (epoch 0).
        let block = ZeroedBlock::new(bytes).ok_or_else(|| super::failure("code plane allocation"))?;
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
        let chunk = self.table.get(k)?.load(Ordering::Acquire);
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
        let Some(entry) = self.table.get(k) else { return };
        let chunk = entry.load(Ordering::Acquire);
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
        let Some(entry) = self.plane.table.get(k) else { return 0 };
        let chunk = entry.load(Ordering::Acquire);
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
        for (id, chunk) in [(1u32, 0), (2, 0), (FIRST + 5, 1), (4 * FIRST, 3), (u32::MAX, 22)] {
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
    }
}
