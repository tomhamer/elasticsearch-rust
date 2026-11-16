//! A 1-bit (bbq) field's graph maintenance rows by node id: each node's
//! vector as 4-bit OSQ codes around its calibration epoch's centroid,
//! which pruning and diversity selection decode and compare
//! ([`MutableVectorSource::get_node_approx`](crate::vector::hnsw::MutableVectorSource::get_node_approx)).
//!
//! 1-bit codes decode too coarsely for diversity selection (nodes lost
//! their links), so maintenance used to read every row's f32 from the
//! vector store: a random 4 KiB read (at 1024 dimensions) for each row of
//! every pruned neighbour list, i.e. hundreds per insert. Once the raw
//! vectors outgrew the page cache those were disk reads and ingest
//! collapsed. ES builds a BBQ graph at merge from quantized vectors only
//! (4-bit query codes against 1-bit doc codes, the query codes in a
//! temporary file); this keeps such a side per node instead, at
//! [`bits`] bits per dimension (2 by default: a sixteenth of the f32, 272
//! bytes at 1024-d; 4, ES's query width, on request), written once when the
//! node is installed (its f32 still in memory). On msmarco-v2 (2M 1024-d
//! vectors, 1000 queries) recall@10 against exact search was 0.920 / 0.931
//! at num_candidates 100 / 1000 with 2-bit rows, 0.926 / 0.931 with 4-bit
//! ones and 0.917 / 0.926 comparing f32 rows.
//!
//! The rows are rebuilt, never recovered: they are not part of any commit.
//! They are resident and charged like the code plane (and spill past the
//! budget like it, `huge::plane_chunk`): pruning reads them at random, and
//! in the page cache they were evicted by the streaming writes of ingestion
//! (store, translog, raw vectors) and read back a page per row. A node
//! without a row (restored from a checkpoint, or a row that could not be
//! written) falls back to its f32.
//!
//! Writes follow the code plane's rules (under the graph's writer, before
//! the node is published; a slot is reused only after its readers
//! drained), except a backfill of a restored node's row, written under the
//! writer after its publication: an entry's epoch is stored last with
//! Release and read first with Acquire, so a reader sees no entry or a
//! complete one. Chunks are installed by compare-and-swap.
//! Entry layout (words): `[epoch lo, epoch hi, lower, upper, additional,
//! component sum, packed 4-bit codes...]`; epoch 0 marks an unwritten entry.

use super::super::huge::ZeroedBlock;
use crate::vector::osq::OsqQuantizer;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

/// Bits per dimension of maintenance rows (`RUSTIC_BBQ_MAINTENANCE_BITS`:
/// 2, the default, or 4, ES's BBQ query width, for twice the memory).
pub(super) fn bits() -> u8 {
    static BITS: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| match std::env::var("RUSTIC_BBQ_MAINTENANCE_BITS").ok().as_deref() { Some("4") => 4, _ => 2 })
}

/// Slots of the first two chunks; chunk `k` in `1..SMALL` holds
/// `FIRST << (k - 1)` slots.
const FIRST: u32 = 1 << 10;
/// Slots of every chunk from id `BIG` on (bounded chunks: growth never
/// allocates more than one chunk ahead).
const BIG: u32 = 1 << 16;
const SMALL: usize = 7;
const LEAF: usize = 256;
const ROOT: usize = ((1usize << 32) / BIG as usize).div_ceil(LEAF);

#[inline]
fn locate(id: u32) -> (usize, usize) {
    if id >= BIG {
        return (SMALL + (id / BIG - 1) as usize, (id % BIG) as usize);
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
    words: *mut AtomicU32,
    _block: ZeroedBlock,
    /// `None` for a file-backed chunk (page cache, not charged).
    _memory: Option<crate::admission::Reservation>,
}
type Leaf = [AtomicPtr<Chunk>; LEAF];

pub(super) struct MutationPlane {
    dim: usize,
    /// Bits per dimension of its rows.
    bits: u8,
    /// Words per entry.
    stride: usize,
    small: [AtomicPtr<Chunk>; SMALL],
    root: Box<[AtomicPtr<Leaf>]>,
    /// Resident (charged) bytes and file-backed bytes of the chunks.
    resident: AtomicUsize,
    mapped: AtomicUsize,
}

// SAFETY: as the code plane: chunks are written by the graph's single
// writer and published through atomics; readers read entries whose writes
// happen before the node publication they acquired.
unsafe impl Send for MutationPlane {}
unsafe impl Sync for MutationPlane {}

impl Drop for MutationPlane {
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

/// Words of a `bits`-bit maintenance row after the epoch header. 2-bit
/// rows (the default, compared on their codes, see [`CodeSpace`]): lower,
/// step, `centroid . codes`, the code sum, the squared norm of the vector
/// they reconstruct, a pad word, then the codes as two bit planes (low
/// bits, high bits). 4-bit rows: four OSQ corrections and the packed codes
/// (decoded to be compared).
pub(super) fn row_words(dim: usize, bits: u8) -> usize {
    if bits == 2 { 6 + 2 * dim.div_ceil(32) } else { 4 + (dim * usize::from(bits)).div_ceil(32) }
}

/// The calibration id of rows quantized around a field's frozen
/// maintenance centroid ([`Maintenance`]); real epoch ids count from 1.
pub(super) const MAINTENANCE_EPOCH: u64 = 1 << 62;
/// Admissions after which a field's maintenance centroid is frozen: rows
/// compared on their codes need one centroid, and the mean of this many
/// rows is a stable estimate of the field's. (Small under test, so the unit
/// tests' graphs mostly compare on codes.)
pub(super) const MAINTENANCE_FREEZE_ROWS: u64 = if cfg!(test) { 512 } else { 32_768 };

/// A field's frozen maintenance centroid: every 2-bit row encoded after the
/// freeze is quantized around it (calibration [`MAINTENANCE_EPOCH`]), so any
/// two such rows compare on their codes. Rows from before (around their own
/// epoch's centroid) are decoded to be compared.
pub(super) struct Maintenance {
    /// The OSQ quantizer around the frozen centroid.
    pub(super) osq: OsqQuantizer,
}

/// Pairwise distances of 2-bit maintenance rows quantized around one
/// centroid `c`, computed on their codes: each row reconstructs
/// `c + lower + step * q`, and the dot product of two reconstructions is
/// `c.c + (la + lb) sum(c) + d la lb + la sb sum(qb) + lb sa sum(qa)
/// + sa c.qa + sb c.qb + sa sb qa.qb`, where `qa.qb` takes four popcounts
/// over the bit planes. The same values (up to rounding) as decoding both
/// rows and comparing them, without the decode.
pub(super) struct CodeSpace {
    cc: f32,
    csum: f32,
    dim: f32,
    similarity: crate::vector::Similarity,
    words: usize,
}

impl CodeSpace {
    pub(super) fn new(centroid: &[f32], similarity: crate::vector::Similarity) -> Self {
        Self { cc: centroid.iter().map(|c| c * c).sum(), csum: centroid.iter().sum(), dim: centroid.len() as f32,
            similarity, words: centroid.len().div_ceil(32) }
    }

    /// The dot product of the vectors rows `a` and `b` reconstruct.
    #[inline]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        let (la, sa, cqa, qa) = (a[0], a[1], a[2], a[3].to_bits() as f32);
        let (lb, sb, cqb, qb) = (b[0], b[1], b[2], b[3].to_bits() as f32);
        let w = self.words;
        let (a0, a1) = (&a[6..6 + w], &a[6 + w..6 + 2 * w]);
        let (b0, b1) = (&b[6..6 + w], &b[6 + w..6 + 2 * w]);
        let (mut p00, mut p01, mut p10, mut p11) = (0u32, 0u32, 0u32, 0u32);
        for i in 0..w {
            let (x0, x1, y0, y1) = (a0[i].to_bits(), a1[i].to_bits(), b0[i].to_bits(), b1[i].to_bits());
            p00 += (x0 & y0).count_ones();
            p01 += (x0 & y1).count_ones();
            p10 += (x1 & y0).count_ones();
            p11 += (x1 & y1).count_ones();
        }
        let qq = (p00 + 2 * (p01 + p10) + 4 * p11) as f32;
        self.cc + (la + lb) * self.csum + self.dim * la * lb + la * sb * qb + lb * sa * qa + sa * cqa + sb * cqb + sa * sb * qq
    }

    /// The maintenance distance (lower is closer) of rows `a` and `b`, as
    /// the similarity's mutation distance of their reconstructions.
    #[inline]
    pub(super) fn distance(&self, a: &[f32], b: &[f32]) -> f64 {
        use crate::vector::Similarity;
        let dot = self.dot(a, b);
        f64::from(match self.similarity {
            Similarity::L2 => a[4] + b[4] - 2.0 * dot,
            Similarity::Cosine => {
                let norms = a[4].sqrt() * b[4].sqrt();
                if norms == 0.0 { 1.0 } else { 1.0 - dot / norms }
            }
            _ => -dot,
        })
    }
}

impl MutationPlane {
    pub(super) fn new(dim: usize, bits: u8) -> Self {
        let stride = (2 + row_words(dim, bits)).div_ceil(4) * 4;
        let mut root = Vec::with_capacity(ROOT);
        root.resize_with(ROOT, || AtomicPtr::new(std::ptr::null_mut()));
        Self { dim, bits, stride, small: Default::default(), root: root.into_boxed_slice(),
            resident: AtomicUsize::new(0), mapped: AtomicUsize::new(0) }
    }

    /// Resident (admission-charged) and file-backed bytes of the chunks.
    pub(super) fn bytes(&self) -> (usize, usize) {
        (self.resident.load(Ordering::Relaxed) + self.root.len() * std::mem::size_of::<AtomicPtr<Leaf>>(),
         self.mapped.load(Ordering::Relaxed))
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

    fn chunk(&self, k: usize) -> crate::error::Result<*const Chunk> {
        let slot: &AtomicPtr<Chunk> = if k < SMALL { &self.small[k] } else {
            let big = k - SMALL;
            let root = self.root.get(big / LEAF).ok_or_else(|| super::failure("mutation plane id out of range"))?;
            let mut leaf = root.load(Ordering::Acquire);
            if leaf.is_null() {
                let fresh: Box<Leaf> = Box::new(std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())));
                let fresh = Box::into_raw(fresh);
                match root.compare_exchange(std::ptr::null_mut(), fresh, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => { leaf = fresh; self.resident.fetch_add(std::mem::size_of::<Leaf>(), Ordering::Relaxed); }
                    // SAFETY: never published; the winner's leaf is used.
                    Err(winner) => { drop(unsafe { Box::from_raw(fresh) }); leaf = winner; }
                }
            }
            // SAFETY: published leaf, lives as long as the plane.
            unsafe { &(*leaf)[big % LEAF] }
        };
        let current = slot.load(Ordering::Acquire);
        if !current.is_null() { return Ok(current); }
        let bytes = chunk_slots(k) * self.stride * 4;
        // Resident and charged, past the budget's spill threshold a spill
        // file (see the module notes).
        let (block, memory) = super::super::huge::plane_chunk(bytes, "mutable maintenance rows")?;
        let file_backed = block.is_file_backed();
        let words = block.ptr().cast::<AtomicU32>();
        let chunk = Box::into_raw(Box::new(Chunk { words, _block: block, _memory: memory }));
        match slot.compare_exchange(std::ptr::null_mut(), chunk, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                if file_backed { self.mapped.fetch_add(bytes, Ordering::Relaxed); } else { self.resident.fetch_add(bytes, Ordering::Relaxed); }
                Ok(chunk)
            }
            // SAFETY: never published; the winner's chunk is used.
            Err(winner) => { drop(unsafe { Box::from_raw(chunk) }); Ok(winner) }
        }
    }

    /// Write node `id`'s row: `epoch` and `words` ([`row_words`] long).
    /// Only the graph's writer calls this, before `id` is published.
    pub(super) fn write(&self, id: u32, epoch: u64, words: &[u32]) -> crate::error::Result<()> {
        if words.len() != row_words(self.dim, self.bits) || epoch == 0 {
            return Err(super::failure("mutation plane entry width"));
        }
        let (k, index) = locate(id);
        let chunk = self.chunk(k)?;
        // SAFETY: inside the chunk; single writer; readers of this entry are
        // ordered after the node's publication.
        unsafe {
            let entry = (*chunk).words.add(index * self.stride);
            for (i, &w) in words.iter().enumerate() {
                (*entry.add(2 + i)).store(w, Ordering::Relaxed);
            }
            (*entry.add(1)).store((epoch >> 32) as u32, Ordering::Relaxed);
            (*entry).store(epoch as u32, Ordering::Release);
        }
        Ok(())
    }

    /// Mark node `id`'s entry unwritten (no allocation). Only the writer
    /// calls this, for a slot no reader can reach.
    pub(super) fn clear(&self, id: u32) {
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if chunk.is_null() { return; }
        // SAFETY: inside the chunk; single writer.
        unsafe {
            let entry = (*chunk).words.add(index * self.stride);
            (*entry).store(0, Ordering::Relaxed);
            (*entry.add(1)).store(0, Ordering::Relaxed);
        }
    }

    /// Node `id`'s row: its epoch and words (`None`: never written). Read it
    /// only for a node acquired under a guard (module notes).
    #[inline]
    pub(super) fn entry(&self, id: u32) -> Option<(u64, &[f32])> {
        let (k, index) = locate(id);
        let chunk = self.chunk_ptr(k);
        if chunk.is_null() { return None; }
        // SAFETY: inside the chunk; `AtomicU32` and `f32` share size and
        // alignment; the entry's writes happen before the caller's
        // acquisition of node `id`.
        let words = unsafe { (*chunk).words.add(index * self.stride) };
        // SAFETY: inside the chunk. The low epoch word is stored last
        // (Release): acquired first, the rest of the entry is complete.
        let low = unsafe { (*words).load(Ordering::Acquire) };
        if low == 0 && unsafe { (*words.add(1)).load(Ordering::Relaxed) } == 0 { return None; }
        let entry = unsafe { std::slice::from_raw_parts(words.cast::<f32>(), 2 + row_words(self.dim, self.bits)) };
        let epoch = u64::from(low) | (u64::from(entry[1].to_bits()) << 32);
        (epoch != 0).then(|| (epoch, &entry[2..]))
    }

    /// Bits per dimension of its rows.
    pub(super) fn bits(&self) -> u8 { self.bits }
}

/// `vector`'s `bits`-bit maintenance row for an epoch quantizing around
/// `q`'s centroid (`unit`: the vector as quantized, unit length for
/// cosine): corrections, then the codes, 4-bit ones packed as ES packs int4
/// (byte `i`: dimension `i` high, `i + half` low), 2-bit ones four per byte
/// (dimension `4i + j` in bits `2j`).
pub(super) fn encode(q: &OsqQuantizer, unit: &[f32], bits: u8) -> Vec<u32> {
    if bits == 2 { return encode2(q, unit); }
    let mut codes = Vec::with_capacity(unit.len());
    let c = q.quantize(unit, bits, &mut codes);
    let mut words = Vec::with_capacity(row_words(unit.len(), bits));
    words.extend([c.lower.to_bits(), c.upper.to_bits(), c.additional.to_bits(), c.component_sum as u32]);
    let packed: Vec<u8> = if bits == 4 {
        let half = unit.len().div_ceil(2);
        (0..half).map(|i| (codes[i] << 4) | codes.get(i + half).copied().unwrap_or(0)).collect()
    } else {
        codes.chunks(4).map(|four| four.iter().enumerate().fold(0u8, |b, (j, &c)| b | ((c & 3) << (2 * j)))).collect()
    };
    let mut bytes = packed.into_iter();
    loop {
        let mut word = [0u8; 4];
        let mut n = 0;
        for b in word.iter_mut() { match bytes.next() { Some(v) => { *b = v; n += 1; } None => break } }
        if n == 0 { break; }
        words.push(u32::from_le_bytes(word));
    }
    words
}

/// `unit`'s 2-bit maintenance row around `q`'s centroid (see
/// [`row_words`]), quantized as ES's OSQ quantizes (the interval refined to
/// the vector): a one-pass interval left nodes with fewer links.
pub(super) fn encode2(q: &OsqQuantizer, unit: &[f32]) -> Vec<u32> {
    let dim = unit.len();
    let mut codes = Vec::with_capacity(dim);
    let corrections = q.quantize(unit, 2, &mut codes);
    let (lower, step) = (corrections.lower, (corrections.upper - corrections.lower) / 3.0);
    let w = dim.div_ceil(32);
    let mut words = vec![0u32; 6 + 2 * w];
    let (mut qsum, mut cq, mut norm2) = (0u32, 0f32, 0f32);
    for (d, (&q, &c)) in codes.iter().zip(&q.centroid).enumerate() {
        let q = u32::from(q.min(3));
        if q & 1 != 0 { words[6 + d / 32] |= 1 << (d % 32); }
        if q & 2 != 0 { words[6 + w + d / 32] |= 1 << (d % 32); }
        qsum += q;
        cq = c.mul_add(q as f32, cq);
        let r = (q as f32).mul_add(step, lower + c);
        norm2 = r.mul_add(r, norm2);
    }
    words[0] = lower.to_bits();
    words[1] = step.to_bits();
    words[2] = cq.to_bits();
    words[3] = qsum;
    words[4] = norm2.to_bits();
    words
}

/// The vector a maintenance row approximates, in its quantizer's space
/// (unit length for cosine): `centroid + lower + code * step` per dimension.
pub(super) fn decode(centroid: &[f32], words: &[f32], dim: usize, bits: u8, out: &mut Vec<f32>) -> bool {
    if words.len() != row_words(dim, bits) || centroid.len() != dim { return false; }
    if bits == 2 {
        let (lower, step, w) = (words[0], words[1], dim.div_ceil(32));
        out.clear();
        out.extend(centroid.iter().enumerate().map(|(d, &c)| {
            let bit = |plane: usize| (words[6 + plane * w + d / 32].to_bits() >> (d % 32)) & 1;
            ((bit(0) | (bit(1) << 1)) as f32).mul_add(step, lower + c)
        }));
        return out.iter().all(|x| x.is_finite());
    }
    let (lower, upper) = (words[0], words[1]);
    let step = (upper - lower) / f32::from((1u8 << bits) - 1);
    let byte = |i: usize| (words[4 + i / 4].to_bits() >> ((i % 4) * 8)) as u8;
    out.clear();
    out.resize(dim, 0.0);
    if bits == 4 {
        let half = dim.div_ceil(2);
        let (head, tail) = out.split_at_mut(half);
        let (c_head, c_tail) = centroid.split_at(half);
        for (i, (o, &m)) in head.iter_mut().zip(c_head).enumerate() { *o = f32::from(byte(i) >> 4).mul_add(step, lower + m); }
        for (i, (o, &m)) in tail.iter_mut().zip(c_tail).enumerate() { *o = f32::from(byte(i) & 15).mul_add(step, lower + m); }
    } else {
        for (d, (o, &m)) in out.iter_mut().zip(centroid).enumerate() { *o = f32::from((byte(d / 4) >> (2 * (d % 4))) & 3).mul_add(step, lower + m); }
    }
    out.iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::osq::OsqSim;

    #[test]
    fn rows_round_trip_close_to_the_vector_and_chunks_stay_bounded() {
        let dim = 100;
        let rows: Vec<Vec<f32>> = (0..64).map(|i| (0..dim).map(|d| ((i * 31 + d * 7) % 23) as f32 / 23.0 - 0.5).collect()).collect();
        let q = OsqQuantizer::fit(OsqSim::Dot, rows.len(), dim, 1, 4, |i| rows[i].as_slice());
        for (bits, bound) in [(4u8, 0.1f32), (2, 0.35)] {
        let plane = MutationPlane::new(dim, bits);
        let mut out = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let id = [1u32, 1023, 1024, BIG - 1, BIG, 3 * BIG + 5][i % 6] + i as u32;
            let words = encode(&q, row, bits);
            plane.write(id, 7, &words).unwrap();
            let (epoch, stored) = plane.entry(id).unwrap();
            assert_eq!(epoch, 7);
            assert!(decode(&q.centroid, stored, dim, bits, &mut out));
            let err: f32 = row.iter().zip(&out).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            let norm: f32 = row.iter().map(|a| a * a).sum::<f32>().sqrt();
            assert!(err < bound * norm, "{bits} bits, row {i}: error {err} of {norm}");
            plane.clear(id);
            assert!(plane.entry(id).is_none());
        }
        assert!(plane.entry(9 * BIG).is_none());
        }
        // Every chunk past the doubling ones is BIG slots: the largest
        // allocation is bounded whatever the id.
        assert_eq!(chunk_slots(SMALL + 1000), BIG as usize);
        assert_eq!(locate(BIG), (SMALL, 0));
        assert_eq!(locate(BIG - 1), (SMALL - 1, (BIG - 1 - (FIRST << (SMALL - 2))) as usize));
    }

    /// 2-bit rows compared on their codes give the distances their decoded
    /// rows give, for every similarity.
    #[test]
    fn code_space_distances_match_decoded_rows() {
        use crate::vector::Similarity;
        let dim = 100;
        let rows: Vec<Vec<f32>> = (0..40).map(|i| {
            let v: Vec<f32> = (0..dim).map(|d| ((i * 37 + d * 11) % 29) as f32 / 29.0 - 0.4).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect()
        }).collect();
        let centroid: Vec<f32> = (0..dim).map(|d| rows.iter().map(|r| r[d]).sum::<f32>() / rows.len() as f32).collect();
        let q = OsqQuantizer::from_centroid(crate::vector::osq::OsqSim::Dot, centroid.clone(), 1, 4);
        let encoded: Vec<Vec<f32>> = rows.iter().map(|r| encode2(&q, r).into_iter().map(f32::from_bits).collect()).collect();
        let decoded: Vec<Vec<f32>> = encoded.iter().map(|w| { let mut out = Vec::new(); assert!(decode(&centroid, w, dim, 2, &mut out)); out }).collect();
        for similarity in [Similarity::DotProduct, Similarity::L2, Similarity::Cosine, Similarity::MaxInnerProduct] {
            let space = CodeSpace::new(&centroid, similarity);
            for a in 0..rows.len() {
                for b in 0..rows.len() {
                    let exact = match similarity {
                        Similarity::Cosine => {
                            let dot: f32 = decoded[a].iter().zip(&decoded[b]).map(|(x, y)| x * y).sum();
                            let na = decoded[a].iter().map(|x| x * x).sum::<f32>().sqrt();
                            let nb = decoded[b].iter().map(|x| x * x).sum::<f32>().sqrt();
                            1.0 - dot / (na * nb)
                        }
                        _ => similarity.distance(&decoded[a], &decoded[b]),
                    };
                    let fast = space.distance(&encoded[a], &encoded[b]) as f32;
                    assert!((fast - exact).abs() < 1e-3, "{similarity:?} {a} {b}: {fast} vs {exact}");
                }
            }
        }
        // And the reconstructions stay close to their vectors.
        for (row, out) in rows.iter().zip(&decoded) {
            let err: f32 = row.iter().zip(out).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
            assert!(err < 0.6, "2-bit reconstruction error {err}");
        }
    }
}
