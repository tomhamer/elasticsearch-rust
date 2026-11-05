//! On-disk flat (brute-force) vector column.
//!
//! Wire format (all integers little-endian):
//!
//! ```text
//! u32 dim
//! u32 num_docs
//! u8  has_values_flag        // 1 if a bitmap follows, 0 if all docs have vectors
//! [u64 bitmap words]         // only when has_values_flag == 1; ceil(num_docs/64) words
//! [pad to a multiple of 4 bytes from the column start]
//! [f32 LE x dim x num_present]  // vectors in ascending doc-id order
//! ```
//!
//! [`FlatVectors::get`] is zero-copy when the vector section happens to be
//! 4-byte aligned (it returns `Cow::Borrowed`), and falls back to a copy
//! (`Cow::Owned`) otherwise.

use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::DocId;
use crate::error::{Result, RusticError};

use super::Similarity;

/// Reader over an encoded flat vector column.
pub struct FlatVectors<'a> {
    dim: u32,
    num_docs: u32,
    /// Per-doc presence bitmap bytes (only present when the column is sparse).
    bitmap: Option<&'a [u8]>,
    /// For sparse columns: number of present docs before each 64-doc word,
    /// so a doc's slot in the packed section is O(1) (rank query).
    rank: Vec<u32>,
    /// Raw bytes of the vector section.
    vecs: &'a [u8],
    squared_norms: std::sync::OnceLock<Vec<f32>>,
}

/// A vector returned by [`FlatVectors::get`]: borrowed when the on-disk
/// section is 4-byte aligned, otherwise a copy.
pub type VectorRef<'a> = Cow<'a, [f32]>;

impl<'a> FlatVectors<'a> {
    /// Parse an encoded column. Returns an error if the data is truncated or
    /// internally inconsistent.
    pub fn open(data: &'a [u8]) -> Result<FlatVectors<'a>> {
        let mut p = 0usize;
        let rd_u32 = |p: &mut usize, data: &'a [u8]| -> Result<u32> {
            if *p + 4 > data.len() {
                return Err(RusticError::Corrupted("flat vectors: truncated header".into()));
            }
            let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
            *p += 4;
            Ok(v)
        };
        let dim = rd_u32(&mut p, data)?;
        let num_docs = rd_u32(&mut p, data)?;
        if p + 1 > data.len() {
            return Err(RusticError::Corrupted("flat vectors: truncated flag".into()));
        }
        let has_bitmap = data[p] != 0;
        p += 1;
        // Borrowed byte view of the presence bitmap (only when sparse).
        let bitmap: Option<&'a [u8]> = if has_bitmap {
            let words = (num_docs as usize + 63) / 64;
            if p + words * 8 > data.len() {
                return Err(RusticError::Corrupted("flat vectors: truncated bitmap".into()));
            }
            let words_slice = &data[p..p + words * 8];
            p += words * 8;
            Some(words_slice)
        } else {
            None
        };
        // Rank table + present count (sparse) or all docs present (dense).
        let mut rank = Vec::new();
        let num_present = match bitmap {
            Some(words) => {
                let mut acc = 0u32;
                rank.reserve(words.len() / 8);
                for w in words.chunks_exact(8) {
                    rank.push(acc);
                    acc += u64::from_le_bytes(w.try_into().unwrap()).count_ones();
                }
                acc as usize
            }
            None => num_docs as usize,
        };
        // Vectors start 4-aligned relative to the column so that a 4-aligned
        // column (the segment codec guarantees this) yields zero-copy reads.
        p = (p + 3) & !3;
        let vecs_start = p;
        let expected = (dim as usize) * num_present * 4;
        if data.len() < vecs_start + expected {
            return Err(RusticError::Corrupted(
                "flat vectors: truncated vector data".into(),
            ));
        }
        let vecs = &data[vecs_start..vecs_start + expected];
        Ok(FlatVectors {
            dim,
            num_docs,
            bitmap,
            rank,
            vecs,
            squared_norms: std::sync::OnceLock::new(),
        })
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Number of docs in the column (including docs without vectors).
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Whether `doc` has a vector.
    pub fn has(&self, doc: DocId) -> bool {
        if doc >= self.num_docs {
            return false;
        }
        match self.bitmap {
            None => true,
            Some(words) => {
                let w = doc as usize / 64;
                let bit = (doc as usize) % 64;
                let word = u64::from_le_bytes(words[w * 8..w * 8 + 8].try_into().unwrap());
                word >> bit & 1 == 1
            }
        }
    }

    /// The vector for `doc`, if present.
    ///
    /// Zero-copy (`Cow::Borrowed`) when the vector section is 4-byte aligned;
    /// otherwise a copy (`Cow::Owned`).
    pub fn get(&self, doc: DocId) -> Option<VectorRef<'a>> {
        if !self.has(doc) {
            return None;
        }
        // Slot of `doc` in the packed section: rank(doc) = present docs before it.
        let present = match self.bitmap {
            None => doc as usize,
            Some(words) => {
                let w = doc as usize / 64;
                let word = u64::from_le_bytes(words[w * 8..w * 8 + 8].try_into().unwrap());
                let below = word & ((1u64 << (doc as usize % 64)) - 1);
                self.rank[w] as usize + below.count_ones() as usize
            }
        };
        let dim = self.dim as usize;
        let byte_off = present * dim * 4;
        let slice = &self.vecs[byte_off..byte_off + dim * 4];
        let ptr = slice.as_ptr();
        // Zero-copy path: only valid when the bytes are 4-byte aligned and the
        // platform is little-endian (the on-disk format is LE f32).
        if cfg!(target_endian = "little") && ptr.align_offset(4) == 0 {
            // SAFETY: `slice` is `dim * 4` bytes, 4-byte aligned, and lives for
            // `'a`. Every bit pattern is a valid `f32`.
            let floats = unsafe { std::slice::from_raw_parts(ptr as *const f32, dim) };
            Some(Cow::Borrowed(floats))
        } else {
            let out: Vec<f32> = slice
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Some(Cow::Owned(out))
        }
    }

    /// Brute-force k-NN over all accepted docs that have vectors.
    ///
    /// Returns up to `k` `(doc, score)` pairs sorted by score descending,
    /// where `score` is [`Similarity::score`] (higher is better). Docs for
    /// which `accept(doc)` is false are skipped entirely.
    pub fn brute_force_knn(
        &self,
        query: &[f32],
        k: usize,
        similarity: Similarity,
        accept: impl Fn(DocId) -> bool,
    ) -> Vec<(DocId, f32)> {
        assert_eq!(query.len(), self.dim as usize, "query dim mismatch");
        let dim = self.dim as usize;
        if let Some(all) = super::hnsw::VectorSource::as_contiguous(self) {
            let rows = all.chunks_exact(dim.max(1)).enumerate().map(|(d, v)| (d as DocId, v)).filter(|(d, _)| accept(*d));
            return exact_top_k(query, similarity, k, rows);
        }
        // Sparse or misaligned column: go through `get`.
        let owned: Vec<(DocId, VectorRef<'a>)> =
            (0..self.num_docs).filter(|&d| accept(d)).filter_map(|d| self.get(d).map(|v| (d, v))).collect();
        exact_top_k(query, similarity, k, owned.iter().map(|(d, v)| (*d, v.as_ref())))
    }
}

/// Exact top-`k` of `rows` by [`Similarity::score`] (desc, ties by id asc).
///
/// Ranks with the SIMD distance kernels (one fused pass for cosine, query
/// norm hoisted) through a bounded max-heap of packed `(distance, id)`
/// keys, then computes the ES score only for the survivors (plus a small
/// margin so float ties between distance and score can't change the top k).
pub(crate) fn exact_top_k<'v>(
    query: &[f32],
    similarity: Similarity,
    k: usize,
    rows: impl Iterator<Item = (u32, &'v [f32])>,
) -> Vec<(u32, f32)> {
    let (hits, scanned) = exact_top_k_uncounted(query, similarity, k, rows, None);
    super::hnsw::add_vector_ops(scanned);
    hits
}

/// Native POC access to the production exact-scan implementation. Keep the
/// optimized cached-norm/prefetch path identical in baseline comparisons.
/// Row ids index `squared_norms`; callers must retain rows through this call.
#[cfg(feature = "mutable-hnsw-poc")]
pub fn poc_exact_top_k<'v>(
    query: &[f32], similarity: Similarity, k: usize,
    rows: impl Iterator<Item = (u32, &'v [f32])>,
    squared_norms: Option<&[f32]>,
) -> (Vec<(u32, f32)>, u64) {
    exact_top_k_uncounted(query, similarity, k, rows, squared_norms)
}

pub(crate) fn exact_top_k_uncounted<'v>(
    query: &[f32], similarity: Similarity, k: usize,
    rows: impl Iterator<Item = (u32, &'v [f32])>,
    squared_norms: Option<&[f32]>,
) -> (Vec<(u32, f32)>, u64) {
    use super::distance::{dot, dot_and_sq_norm, l2_sq, norm};
    use super::hnsw::pack;
    use super::Metric;

    /// Heap entry ordered by its packed `(distance, id)` key only.
    struct E<'v>(u64, &'v [f32]);
    impl PartialEq for E<'_> {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for E<'_> {}
    impl PartialOrd for E<'_> {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for E<'_> {
        fn cmp(&self, o: &Self) -> Ordering {
            self.0.cmp(&o.0)
        }
    }

    if k == 0 {
        return (Vec::new(), 0);
    }
    let keep = k + k / 2 + 4;
    let metric = similarity.metric();
    let qn = if metric == Metric::Cos { norm(query) } else { 0.0 };
    let inv_qn = if qn > 0.0 { 1.0 / qn } else { 0.0 };
    let mut heap: BinaryHeap<E<'v>> = BinaryHeap::with_capacity(keep + 1);
    let mut scanned = 0u64;
    // Sparse filters can jump across a page for every vector. Issue reads
    // for a small batch before computing distances to overlap memory latency.
    for (id, v) in prefetch_rows(rows) {
        scanned += 1;
        let d = match metric {
            Metric::L2 => l2_sq(query, v),
            Metric::Dot => -dot(query, v),
            Metric::Cos => {
                let (x, vv) = match squared_norms {
                    Some(norms) => (dot(query, v), norms[id as usize]),
                    None => dot_and_sq_norm(query, v),
                };
                if vv == 0.0 || inv_qn == 0.0 { 1.0 } else { 1.0 - x * inv_qn / vv.sqrt() }
            }
            Metric::Packed => similarity.distance(query, v),
        };
        if d.is_nan() {
            continue;
        }
        let key = pack(d, id);
        if heap.len() < keep {
            heap.push(E(key, v));
        } else if key < heap.peek().unwrap().0 {
            heap.pop();
            heap.push(E(key, v));
        }
    }
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|E(key, v)| (key as u32, similarity.score(query, v))).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    (out, scanned)
}

fn prefetch_rows<'a>(mut rows: impl Iterator<Item = (u32, &'a [f32])>) -> impl Iterator<Item = (u32, &'a [f32])> {
    let mut batch = [None; 16];
    let mut position = batch.len();
    std::iter::from_fn(move || {
        if position == batch.len() {
            for entry in &mut batch { *entry = rows.next(); }
            for (_, vector) in batch.iter().flatten() {
                for offset in (0..vector.len()).step_by(16) {
                    // SAFETY: every address is inside the borrowed vector;
                    // prefetch neither reads a Rust value nor modifies memory.
                    #[cfg(target_arch = "aarch64")]
                    unsafe { std::arch::asm!("prfm pldl1keep, [{address}]", address = in(reg) vector.as_ptr().add(offset), options(readonly, nostack, preserves_flags)); }
                    #[cfg(target_arch = "x86_64")]
                    unsafe { std::arch::x86_64::_mm_prefetch(vector.as_ptr().add(offset) as *const i8, std::arch::x86_64::_MM_HINT_T0); }
                }
            }
            position = 0;
        }
        let entry = batch[position].take();
        position += 1;
        entry
    })
}

impl<'a> super::hnsw::VectorSource for FlatVectors<'a> {
    fn squared_norms(&self) -> Option<&[f32]> {
        let data = <Self as super::hnsw::VectorSource>::as_contiguous(self)?;
        Some(self.squared_norms.get_or_init(|| if self.dim == 0 { vec![0.0; self.num_docs as usize] } else {
            data.chunks_exact(self.dim as usize).map(|v|super::distance::dot_and_sq_norm(v, v).1).collect()
        }))
    }
    fn dim(&self) -> u32 {
        self.dim
    }
    fn num_docs(&self) -> u32 {
        self.num_docs
    }
    /// Zero-copy read; the segment codec 4-aligns the column so this never
    /// needs the copying path. Panics if the column is misaligned, which
    /// indicates a codec bug rather than bad data.
    fn get(&self, doc: DocId) -> Option<&[f32]> {
        match self.get(doc)? {
            Cow::Borrowed(v) => Some(v),
            Cow::Owned(_) => panic!("flat vector column is not 4-byte aligned"),
        }
    }
    fn as_contiguous(&self) -> Option<&[f32]> {
        if self.bitmap.is_some() || !cfg!(target_endian = "little") || self.vecs.as_ptr().align_offset(4) != 0 {
            return None;
        }
        // SAFETY: dense column (every doc present, in doc order), 4-byte
        // aligned, `num_docs * dim` LE floats; any bit pattern is a valid f32.
        Some(unsafe { std::slice::from_raw_parts(self.vecs.as_ptr() as *const f32, self.vecs.len() / 4) })
    }
}

/// Builder for a flat vector column.
pub struct FlatVectorsBuilder {
    dim: u32,
    /// doc -> vector, in insertion order of doc ids.
    entries: Vec<(DocId, Vec<f32>)>,
}

impl FlatVectorsBuilder {
    /// Start a builder for vectors of `dim` dimensions.
    pub fn new(dim: u32) -> Self {
        FlatVectorsBuilder {
            dim,
            entries: Vec::new(),
        }
    }

    /// Record the vector for `doc`. Panics if `vec.len() != dim` or the doc
    /// was already set.
    pub fn set(&mut self, doc: DocId, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim as usize, "vector dim mismatch");
        // Docs normally arrive in ascending order; only the last entry can
        // collide. Out-of-order inserts are sorted (and checked) at encode.
        debug_assert!(self.entries.last().map_or(true, |(d, _)| *d != doc), "doc {doc} already set");
        self.entries.push((doc, vec.to_vec()));
    }

    /// Encode the column for `num_docs` total docs. Docs not set via
    /// [`set`](Self::set) are marked absent in the presence bitmap.
    pub fn encode(&self, num_docs: u32) -> Vec<u8> {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|(d, _)| *d);
        assert!(entries.windows(2).all(|w| w[0].0 != w[1].0), "duplicate doc in vector column");
        let num_present = entries.len() as u32;
        let sparse = num_present < num_docs;
        let mut out = Vec::with_capacity(
            9 + (num_docs as usize / 64 + 1) * 8 + num_present as usize * self.dim as usize * 4,
        );
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&num_docs.to_le_bytes());
        out.push(sparse as u8);
        if sparse {
            let words = (num_docs as usize + 63) / 64;
            let mut bitmap = vec![0u64; words];
            for (doc, _) in &entries {
                bitmap[*doc as usize / 64] |= 1u64 << (*doc as usize % 64);
            }
            for w in bitmap {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        while out.len() % 4 != 0 {
            out.push(0);
        }
        for (_, v) in &entries {
            for f in v {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_vec(seed: u32, dim: u32) -> Vec<f32> {
        let mut x = seed as u64 | 1;
        (0..dim)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let v = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
                v * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn dense_roundtrip() {
        let dim = 8u32;
        let num_docs = 5u32;
        let mut b = FlatVectorsBuilder::new(dim);
        let mut expected = Vec::new();
        for d in 0..num_docs {
            let v = rng_vec(d + 1, dim);
            expected.push(v.clone());
            b.set(d, &v);
        }
        let data = b.encode(num_docs);
        let f = FlatVectors::open(&data).unwrap();
        assert_eq!(f.dim(), dim);
        assert_eq!(f.num_docs(), num_docs);
        for d in 0..num_docs {
            assert!(f.has(d));
            assert_eq!(f.get(d).unwrap().as_ref(), expected[d as usize].as_slice());
        }
    }

    #[test]
    fn sparse_roundtrip() {
        let dim = 4u32;
        let num_docs = 130u32; // > 64 to exercise multi-word bitmap
        let mut b = FlatVectorsBuilder::new(dim);
        let mut present = Vec::new();
        for d in 0..num_docs {
            if d % 3 != 0 {
                let v = rng_vec(d + 7, dim);
                b.set(d, &v);
                present.push((d, v));
            }
        }
        let data = b.encode(num_docs);
        let f = FlatVectors::open(&data).unwrap();
        for d in 0..num_docs {
            if d % 3 == 0 {
                assert!(!f.has(d));
                assert!(f.get(d).is_none());
            } else {
                assert!(f.has(d));
                let (doc, v) = present.iter().find(|(dd, _)| *dd == d).unwrap();
                assert_eq!(f.get(*doc).unwrap().as_ref(), v.as_slice());
            }
        }
    }

    #[test]
    fn truncated_data_is_corrupted() {
        let mut b = FlatVectorsBuilder::new(4);
        b.set(0, &[1.0, 2.0, 3.0, 4.0]);
        let data = b.encode(1);
        assert!(FlatVectors::open(&data[..data.len() - 1]).is_err());
        assert!(FlatVectors::open(&data[..4]).is_err());
    }

    #[test]
    fn brute_force_matches_sorted_scores() {
        let dim = 16u32;
        let num_docs = 50u32;
        let mut b = FlatVectorsBuilder::new(dim);
        for d in 0..num_docs {
            b.set(d, &rng_vec(d + 100, dim));
        }
        let data = b.encode(num_docs);
        let f = FlatVectors::open(&data).unwrap();
        let query = rng_vec(999, dim);
        let k = 7;
        let got = f.brute_force_knn(&query, k, Similarity::Cosine, |_| true);
        // Reference: score everything, sort, take k.
        let mut all: Vec<(DocId, f32)> = (0..num_docs)
            .map(|d| (d, Similarity::Cosine.score(&query, f.get(d).unwrap().as_ref())))
            .collect();
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let want: Vec<DocId> = all.iter().take(k).map(|(d, _)| *d).collect();
        let got_docs: Vec<DocId> = got.iter().map(|(d, _)| *d).collect();
        assert_eq!(got_docs, want);
        // Scores sorted descending.
        for w in got.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn brute_force_respects_accept_and_sparse() {
        let dim = 8u32;
        let num_docs = 20u32;
        let mut b = FlatVectorsBuilder::new(dim);
        for d in 0..num_docs {
            if d % 2 == 0 {
                b.set(d, &rng_vec(d + 1, dim));
            }
        }
        let data = b.encode(num_docs);
        let f = FlatVectors::open(&data).unwrap();
        let query = rng_vec(5, dim);
        // Accept only odd docs: none of them have vectors -> empty result.
        let r = f.brute_force_knn(&query, 5, Similarity::L2, |d| d % 2 == 1);
        assert!(r.is_empty());
        // Accept only even docs: exactly the docs with vectors.
        let r = f.brute_force_knn(&query, 5, Similarity::L2, |d| d % 2 == 0);
        assert_eq!(r.len(), 5);
        assert!(r.iter().all(|(d, _)| *d % 2 == 0));
    }
}
