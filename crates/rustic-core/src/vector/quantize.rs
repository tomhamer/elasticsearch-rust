//! Int8 scalar quantization (Elasticsearch `index_options.type: int8_hnsw`).
//!
//! A [`ScalarQuantizer`] is fitted on a sample of vector components by
//! percentile clipping at a confidence interval (default 0.99, i.e. the
//! central 99% of component values map onto the int8 range and the tails are
//! clamped). Each component is then mapped linearly from `[min, max]` to
//! `[-127, 127]`.
//!
//! Wire format of an [`Int8Vectors`] column (all integers little-endian):
//!
//! ```text
//! u32 dim
//! u32 num_docs
//! f32 min
//! f32 max
//! [i8 x dim x num_docs]   // one row per doc, no presence bitmap
//! ```
//!
//! [`Int8Vectors::get`] is alignment-free (`i8` has alignment 1).
//! [`Int8VectorSource`] implements [`VectorSource`] by dequantizing on the
//! fly; it allocates one `Vec<f32>` per call (v1 simplicity — a reusable
//! scratch buffer would need interior mutability behind the `&self` trait).
//!
//! # Quantization-aware search
//!
//! [`QuantizedVectors`] is an in-memory scoring index used *during* HNSW
//! traversal (the graph is walked with cheap approximate distances, then the
//! best `k * oversample` candidates are rescored with exact `f32` — ES
//! `int8_hnsw` / `bbq_hnsw` with `rescore_vector`, Qdrant `rescore`):
//!
//! * **int8** — symmetric: the query is quantized with the same
//!   [`ScalarQuantizer`] and scored with an `i8` dot product (NEON
//!   `smull`/`sadalp`), plus per-vector correction terms (`|c|^2` for L2,
//!   `sum(c)` for dot / cosine) so every similarity is estimated, not just L2.
//! * **int4** — Lucene's 4-bit scalar quantization ([`super::int4`]): codes
//!   `0..=15` packed two per byte, the query quantized with the same
//!   quantizer, Lucene's corrective offsets for the dot-product family. The
//!   estimates (and hence the scores returned without rescoring) are the
//!   ones ES returns for `int4_hnsw` / `int4_flat`.
//! * **binary** (BBQ / RaBitQ-style) — each vector is centered on the
//!   segment centroid and stored as 1 sign bit per dimension plus two
//!   floats (`|r|^2`, `|r|^2 / |r|_1`). The query is quantized to 4 bits
//!   per dimension and laid out as 4 bit planes, so `<q, sign(r)>` is 4
//!   AND+popcount passes; the RaBitQ estimator
//!   `<q - c, r> ~= <q - c, sign(r)> * |r|^2 / |r|_1` turns that into an
//!   L2 / dot / cosine estimate.

use std::cell::RefCell;

use crate::DocId;
use crate::error::{Result, RusticError};

use super::hnsw::VectorSource;
use super::int4::{self, Int4Vectors, UniformQuantizer};
use super::{Metric, Similarity};

/// A fitted min/max scalar quantizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarQuantizer {
    /// Lower clip bound; maps to `-127`.
    pub min: f32,
    /// Upper clip bound; maps to `127`.
    pub max: f32,
}

impl ScalarQuantizer {
    /// Default confidence interval used by ES for int8 quantization.
    pub const DEFAULT_CONFIDENCE_INTERVAL: f32 = 0.99;

    /// Fit on a sample of vectors by percentile clipping.
    ///
    /// `confidence_interval` in `(0, 1]`: the central fraction of component
    /// values kept inside `[min, max]`. `1.0` means plain min/max. With an
    /// empty sample the quantizer is `[-1, 1]`.
    pub fn fit<'a, I>(vectors: I, confidence_interval: f32) -> ScalarQuantizer
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        let ci = if confidence_interval.is_finite() && confidence_interval > 0.0 {
            confidence_interval.min(1.0)
        } else {
            Self::DEFAULT_CONFIDENCE_INTERVAL
        };
        let mut sample: Vec<f32> = Vec::new();
        for v in vectors {
            sample.extend(v.iter().copied().filter(|x| x.is_finite()));
        }
        if sample.is_empty() {
            return ScalarQuantizer {
                min: -1.0,
                max: 1.0,
            };
        }
        let n = sample.len();
        let tail = (1.0 - ci) / 2.0;
        let lo_idx = (((n as f32 - 1.0) * tail).round() as usize).min(n - 1);
        let hi_idx = (((n as f32 - 1.0) * (1.0 - tail)).round() as usize).min(n - 1);
        // Two O(n) selections instead of a full sort.
        let mut min = *sample.select_nth_unstable_by(lo_idx, f32::total_cmp).1;
        let mut max = *sample.select_nth_unstable_by(hi_idx, f32::total_cmp).1;
        if max <= min {
            // Degenerate (constant) data: widen slightly to avoid div-by-zero.
            let c = min;
            min = c - 1.0;
            max = c + 1.0;
        }
        ScalarQuantizer { min, max }
    }

    /// Scale factor `127 / ((max - min) / 2)` per unit of input.
    #[inline]
    fn scale(&self) -> f32 {
        254.0 / (self.max - self.min)
    }

    /// Quantize one vector. Components outside `[min, max]` are clamped.
    pub fn quantize(&self, v: &[f32]) -> Vec<i8> {
        let mut out = Vec::with_capacity(v.len());
        self.quantize_into(v, &mut out);
        out
    }

    /// Quantize into an existing buffer (cleared first).
    pub fn quantize_into(&self, v: &[f32], out: &mut Vec<i8>) {
        out.clear();
        let s = self.scale();
        for &x in v {
            let c = x.clamp(self.min, self.max);
            let q = ((c - self.min) * s - 127.0).round();
            out.push(q.clamp(-127.0, 127.0) as i8);
        }
    }

    /// Dequantize one vector.
    pub fn dequantize(&self, q: &[i8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(q.len());
        self.dequantize_into(q, &mut out);
        out
    }

    /// Dequantize into an existing buffer (cleared first).
    pub fn dequantize_into(&self, q: &[i8], out: &mut Vec<f32>) {
        out.clear();
        let inv = (self.max - self.min) / 254.0;
        for &x in q {
            out.push((x as f32 + 127.0) * inv + self.min);
        }
    }
}

/// Integer dot product of two int8 vectors, accumulated in `i32`.
///
/// Uses ARM dot-product instructions when available, otherwise mandatory
/// NEON (`smull` + `sadalp`) on aarch64 and a chunked autovectorized loop
/// elsewhere. Panics if the lengths differ.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: extension checked above; slices have equal lengths.
        return unsafe { dot_i8_dotprod(a.as_ptr(), b.as_ptr(), a.len()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { dot_i8_neon(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(a, b)
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut s = [0i32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..16 {
            s[i] += x[i] as i16 as i32 * y[i] as i16 as i32;
        }
    }
    let mut t: i32 = s.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        t += *x as i32 * *y as i32;
    }
    t
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_i8_neon(a: *const i8, b: *const i8, n: usize) -> i32 {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(a.add(i));
            let y = vld1q_s8(b.add(i));
            // |x*y| <= 127*128 fits i16; pairwise-add-accumulate into i32.
            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
            acc1 = vpadalq_s16(acc1, vmull_high_s8(x, y));
            i += 16;
        }
        let mut s = vaddvq_s32(vaddq_s32(acc0, acc1));
        while i < n {
            s += *a.add(i) as i32 * *b.add(i) as i32;
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_dotprod(a: *const i8, b: *const i8, n: usize) -> i32 {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = acc0;
        let mut acc2 = acc0;
        let mut acc3 = acc0;
        let mut i = 0;
        while i + 64 <= n {
            let x = vld1q_s8(a.add(i));
            let y = vld1q_s8(b.add(i));
            // The stable toolchain lacks the dot-product intrinsic. The
            // feature-gated instruction reads only these SIMD registers.
            core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc0, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_s8(a.add(i + 16));
            let y = vld1q_s8(b.add(i + 16));
            core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc1, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_s8(a.add(i + 32));
            let y = vld1q_s8(b.add(i + 32));
            core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc2, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_s8(a.add(i + 48));
            let y = vld1q_s8(b.add(i + 48));
            core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc3, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            i += 64;
        }
        while i + 16 <= n {
            let x = vld1q_s8(a.add(i));
            let y = vld1q_s8(b.add(i));
            core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc0, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            i += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        while i < n {
            sum += *a.add(i) as i32 * *b.add(i) as i32;
            i += 1;
        }
        sum
    }
}

/// Reader over an encoded int8 vector column.
pub struct Int8Vectors<'a> {
    dim: u32,
    num_docs: u32,
    quantizer: ScalarQuantizer,
    data: &'a [u8],
}

impl<'a> Int8Vectors<'a> {
    /// Header size in bytes.
    const HEADER: usize = 16;

    /// Encode a dense int8 column. `rows` must have exactly `num_docs`
    /// entries of `dim` bytes each (one per doc, in doc order).
    pub fn encode(dim: u32, quantizer: ScalarQuantizer, rows: &[Vec<i8>]) -> Vec<u8> {
        let num_docs = rows.len() as u32;
        let mut out = Vec::with_capacity(Self::HEADER + rows.len() * dim as usize);
        out.extend_from_slice(&dim.to_le_bytes());
        out.extend_from_slice(&num_docs.to_le_bytes());
        out.extend_from_slice(&quantizer.min.to_le_bytes());
        out.extend_from_slice(&quantizer.max.to_le_bytes());
        for r in rows {
            assert_eq!(r.len(), dim as usize, "int8 row dim mismatch");
            out.extend(r.iter().map(|&x| x as u8));
        }
        out
    }

    /// Parse an encoded int8 column.
    pub fn open(data: &'a [u8]) -> Result<Int8Vectors<'a>> {
        if data.len() < Self::HEADER {
            return Err(RusticError::Corrupted("int8 vectors: truncated header".into()));
        }
        let dim = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let num_docs = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let min = f32::from_le_bytes(data[8..12].try_into().unwrap());
        let max = f32::from_le_bytes(data[12..16].try_into().unwrap());
        let expected = Self::HEADER + dim as usize * num_docs as usize;
        if data.len() < expected {
            return Err(RusticError::Corrupted("int8 vectors: truncated data".into()));
        }
        if !(min < max) {
            return Err(RusticError::Corrupted("int8 vectors: invalid min/max".into()));
        }
        Ok(Int8Vectors {
            dim,
            num_docs,
            quantizer: ScalarQuantizer { min, max },
            data: &data[..expected],
        })
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Number of docs.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// The quantizer this column was encoded with.
    pub fn quantizer(&self) -> ScalarQuantizer {
        self.quantizer
    }

    /// The int8 row for `doc`, if `doc < num_docs`.
    pub fn get(&self, doc: DocId) -> Option<&'a [i8]> {
        if doc >= self.num_docs {
            return None;
        }
        let dim = self.dim as usize;
        let start = Self::HEADER + doc as usize * dim;
        let bytes = &self.data[start..start + dim];
        // SAFETY: `i8` and `u8` have identical size and alignment (1); every
        // bit pattern is valid for both.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, dim) })
    }

    /// Wrap as a [`VectorSource`] that dequantizes on the fly.
    pub fn as_source(&self) -> Int8VectorSource<'_, 'a> {
        Int8VectorSource {
            inner: self,
            scratch: RefCell::new(Vec::with_capacity(self.dim as usize)),
        }
    }
}

/// [`VectorSource`] adapter over [`Int8Vectors`] that dequantizes on the fly.
///
/// Each call to [`VectorSource::get`] dequantizes into an internal scratch
/// buffer and returns a slice into it. Because the trait returns a borrow
/// tied to `&self`, the scratch is reused across calls and the returned slice
/// is only valid until the next `get`. A fresh `Vec` is allocated per call
/// only when the buffer needs to grow (i.e. essentially never after the
/// first). Callers that hold two results at once should copy.
pub struct Int8VectorSource<'s, 'a> {
    inner: &'s Int8Vectors<'a>,
    scratch: RefCell<Vec<f32>>,
}

impl<'s, 'a> VectorSource for Int8VectorSource<'s, 'a> {
    fn dim(&self) -> u32 {
        self.inner.dim
    }

    fn num_docs(&self) -> u32 {
        self.inner.num_docs
    }

    fn get(&self, doc: DocId) -> Option<&[f32]> {
        let q = self.inner.get(doc)?;
        let mut buf = self.scratch.borrow_mut();
        self.inner.quantizer.dequantize_into(q, &mut buf);
        // Leak the borrow into a slice tied to `&self`. This is sound only
        // because the buffer is never reallocated while a slice is live for
        // callers that copy before the next `get`; see the type docs.
        let ptr = buf.as_ptr();
        let len = buf.len();
        drop(buf);
        // SAFETY: the Vec is owned by `self` and outlives the returned borrow;
        // its capacity was reserved for `dim` so `dequantize_into` of a
        // `dim`-length row never reallocates after construction.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}


// ---------------------------------------------------------------------------
// Quantization-aware search.

/// Which (if any) quantized representation a graph search scores with.
///
/// In a [`PlannerConfig`](super::store::PlannerConfig) `Auto` means "the
/// field's configured quantization" ([`VectorFieldOptions`](super::store::VectorFieldOptions));
/// in field options and on a bare [`VectorSegment`](super::segment::VectorSegment)
/// it means "whatever quantized copy the segment stores" (int4, int8 or
/// nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Quantization {
    /// Resolve from the field / segment (see above).
    #[default]
    Auto,
    /// Full-precision `f32` distances.
    None,
    /// Int8 scalar quantization (+ optional `f32` rescoring).
    Int8,
    /// 4-bit scalar quantization, two codes per byte (ES `int4_*`).
    Int4,
    /// 1-bit (BBQ / RaBitQ-style) quantization (+ optional `f32` rescoring).
    Binary,
}

impl Quantization {
    /// Parse `none` / `int8` / `int4` / `binary` (`bbq`) / `auto`.
    pub fn parse(s: &str) -> Option<Quantization> {
        Some(match s {
            "auto" => Quantization::Auto,
            "none" | "f32" | "float" => Quantization::None,
            "int8" | "int8_hnsw" | "int8_flat" => Quantization::Int8,
            "int4" | "int4_hnsw" | "int4_flat" => Quantization::Int4,
            "binary" | "bbq" | "bbq_hnsw" | "bbq_flat" | "bit" => Quantization::Binary,
            _ => return None,
        })
    }

    /// Whether this is a real quantized representation (`Int8`, `Int4`, `Binary`).
    pub fn is_quantized(self) -> bool {
        matches!(self, Quantization::Int8 | Quantization::Int4 | Quantization::Binary)
    }
}

/// Max vectors sampled to fit an int8 quantizer (Lucene samples 25k too).
const FIT_SAMPLE: usize = 25_000;

enum QKind {
    Int8 {
        q: ScalarQuantizer,
        codes: Vec<i8>,
        /// L2: `|c|^2`; dot / cosine: `sum(c)`.
        corr: Vec<f32>,
    },
    /// Lucene-style 4-bit codes, packed two per byte (`int4::pack_nibbles`).
    Int4 {
        q: UniformQuantizer,
        /// Bytes per row (`ceil(dim / 2)`).
        row: usize,
        codes: Vec<u8>,
        /// L2: `sum(c^2)`; dot / mip / cosine: Lucene's corrective offset.
        corr: Vec<f32>,
        /// Cosine: codes are of the unit-normalised vectors.
        normalised: bool,
    },
    Binary {
        centroid: Vec<f32>,
        words: usize,
        /// One row per vector, `words + 2` words: the sign codes, then the
        /// row's correction terms packed as f32 pairs, `(scale, rn2)` and
        /// `(cr, inv_norm)`: `scale = |r|^2 / |r|_1` (the RaBitQ rescale
        /// factor), `rn2 = |r|^2` with `r = v - centroid`, `cr = <centroid,
        /// r>` (dot / cosine), `inv_norm` (cosine). One row is one
        /// contiguous read (32 bytes at 128 dims), where separate arrays
        /// cost a cache miss each for a scan of scattered rows (a filter).
        rows: Vec<u64>,
    },
}

/// An in-memory quantized copy of a vector column, indexed by doc/ordinal,
/// used to score candidates during graph traversal. See the module docs.
pub struct QuantizedVectors {
    dim: usize,
    n: usize,
    metric: Metric,
    /// `1 / |v|` per vector (cosine only; `0` for zero vectors).
    inv_norm: Vec<f32>,
    kind: QKind,
}

/// A query prepared for [`QuantizedVectors::distance`].
pub struct QuantQuery {
    /// int8: the quantized query. binary: unused.
    codes: Vec<i8>,
    /// int4: the (unpacked) 4-bit query codes.
    u: Vec<u8>,
    /// binary: 4 bit planes of the 4-bit query, `4 * words` words.
    planes: Vec<u64>,
    /// int8 L2: `|qc|^2`; int8 dot: `sum(qc)`.
    a: f32,
    /// binary: `lo` of the 4-bit query; int8: unused.
    lo: f32,
    /// binary: step of the 4-bit query.
    step: f32,
    /// binary: `sum(quantized q - c)`.
    sum_rq: f32,
    /// binary: `|q - c|^2`.
    rq2: f32,
    /// binary: `<q, c>`.
    qc: f32,
    /// cosine: `1 / |q|`.
    inv_qn: f32,
}

fn rows<'v>(vectors: &'v dyn VectorSource) -> impl Iterator<Item = Option<&'v [f32]>> + 'v {
    (0..vectors.num_docs()).map(move |d| vectors.get(d))
}

fn inv_norms(vectors: &dyn VectorSource) -> Vec<f32> {
    rows(vectors)
        .map(|v| {
            let n = v.map_or(0.0, super::distance::norm);
            if n > 0.0 { 1.0 / n } else { 0.0 }
        })
        .collect()
}

impl QuantizedVectors {
    /// Number of vectors (docs).
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes held by the quantized codes (not counting per-vector floats).
    pub fn code_bytes(&self) -> usize {
        match &self.kind {
            QKind::Int8 { codes, .. } => codes.len(),
            QKind::Int4 { codes, .. } => codes.len(),
            QKind::Binary { words, rows, .. } => rows.len() / (words + 2) * words * 8,
        }
    }

    /// The kind of this index (`Int8`, `Int4` or `Binary`).
    pub fn kind(&self) -> Quantization {
        match self.kind {
            QKind::Int8 { .. } => Quantization::Int8,
            QKind::Int4 { .. } => Quantization::Int4,
            QKind::Binary { .. } => Quantization::Binary,
        }
    }

    /// Build a packed int4 index over every doc of `vectors` (Lucene
    /// semantics: `confidence_interval` `None` = Lucene's default for the
    /// dimension, `Some(0.0)` = the dynamic interval; cosine quantizes the
    /// unit-normalised vectors).
    pub fn int4(vectors: &dyn VectorSource, similarity: Similarity, confidence_interval: Option<f32>) -> QuantizedVectors {
        let n = vectors.num_docs() as usize;
        let dim = vectors.dim() as usize;
        let zero = vec![0.0f32; dim];
        let (q, normalised, rows, offsets) = int4::build_int4(n, |d| vectors.get(d as DocId).unwrap_or(&zero), similarity, confidence_interval);
        let mut codes = Vec::with_capacity(n * int4::packed_len(dim));
        for r in &rows {
            codes.extend_from_slice(r);
        }
        Self::from_int4_codes(vectors, similarity, q, normalised, codes, offsets)
    }

    /// Wrap a stored int4 column (the segment's `int4` section) for search.
    pub fn from_int4_column(col: &Int4Vectors<'_>, vectors: &dyn VectorSource, similarity: Similarity) -> QuantizedVectors {
        Self::from_int4_codes(vectors, similarity, col.quantizer(), col.normalised(), col.rows().to_vec(), col.offsets())
    }

    fn from_int4_codes(vectors: &dyn VectorSource, similarity: Similarity, q: UniformQuantizer, normalised: bool, codes: Vec<u8>, offsets: Vec<f32>) -> QuantizedVectors {
        let dim = vectors.dim() as usize;
        let row = int4::packed_len(dim).max(1);
        let n = codes.len() / row;
        let metric = similarity.metric();
        let corr: Vec<f32> = match metric {
            Metric::L2 => codes.chunks_exact(row).map(|c| int4::sqsum_packed_u4(c) as f32).collect(),
            _ => (0..n).map(|i| offsets.get(i).copied().unwrap_or(0.0)).collect(),
        };
        // Cosine codes are of unit vectors: the estimate is the cosine itself.
        QuantizedVectors { dim, n, metric, inv_norm: Vec::new(), kind: QKind::Int4 { q, row, codes, corr, normalised } }
    }

    /// Build an int8 index over every doc of `vectors` (docs without a
    /// vector get zero codes; they are never graph nodes). The quantizer is
    /// fitted on an evenly strided sample of at most 25k vectors.
    pub fn int8(vectors: &dyn VectorSource, similarity: Similarity) -> QuantizedVectors {
        let n = vectors.num_docs() as usize;
        let stride = n.div_ceil(FIT_SAMPLE).max(1);
        let q = ScalarQuantizer::fit(
            (0..n).step_by(stride).filter_map(|d| vectors.get(d as DocId)),
            ScalarQuantizer::DEFAULT_CONFIDENCE_INTERVAL,
        );
        let dim = vectors.dim() as usize;
        let mut codes = Vec::with_capacity(n * dim);
        let mut buf = Vec::with_capacity(dim);
        for v in rows(vectors) {
            match v {
                Some(v) => {
                    q.quantize_into(v, &mut buf);
                    codes.extend_from_slice(&buf);
                }
                None => codes.extend(std::iter::repeat_n(0i8, dim)),
            }
        }
        Self::from_int8_codes(vectors, similarity, q, codes)
    }

    /// Wrap a stored int8 column (the segment's `int8` section) for search.
    pub fn from_int8_column(col: &Int8Vectors<'_>, vectors: &dyn VectorSource, similarity: Similarity) -> QuantizedVectors {
        let dim = col.dim() as usize;
        let mut codes = Vec::with_capacity(col.num_docs() as usize * dim);
        for d in 0..col.num_docs() {
            codes.extend_from_slice(col.get(d).unwrap());
        }
        Self::from_int8_codes(vectors, similarity, col.quantizer(), codes)
    }

    fn from_int8_codes(vectors: &dyn VectorSource, similarity: Similarity, q: ScalarQuantizer, codes: Vec<i8>) -> QuantizedVectors {
        let dim = vectors.dim() as usize;
        let n = codes.len() / dim.max(1);
        let metric = similarity.metric();
        let corr = codes
            .chunks_exact(dim.max(1))
            .map(|c| match metric {
                Metric::L2 => dot_i8(c, c) as f32,
                _ => c.iter().map(|&x| x as i32).sum::<i32>() as f32,
            })
            .collect();
        let inv_norm = if metric == Metric::Cos { inv_norms(vectors) } else { Vec::new() };
        QuantizedVectors { dim, n, metric, inv_norm, kind: QKind::Int8 { q, codes, corr } }
    }

    /// Build a 1-bit (BBQ / RaBitQ-style) index over every doc of `vectors`.
    pub fn binary(vectors: &dyn VectorSource, similarity: Similarity) -> QuantizedVectors {
        let dim = vectors.dim() as usize;
        let n = vectors.num_docs() as usize;
        let metric = similarity.metric();
        // Centroid of the present vectors.
        let inv_norm = if metric == Metric::Cos { inv_norms(vectors) } else { Vec::new() };
        let mut centroid = vec![0.0f64; dim];
        let mut count = 0usize;
        for v in rows(vectors).flatten() {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += *x as f64;
            }
            count += 1;
        }
        let centroid: Vec<f32> = centroid.iter().map(|c| (c / count.max(1) as f64) as f32).collect();
        let words = dim.div_ceil(64);
        let stride = words + 2;
        let pair = |a: f32, b: f32| u64::from(a.to_bits()) | (u64::from(b.to_bits()) << 32);
        let mut packed = vec![0u64; n * stride];
        let mut r = vec![0.0f32; dim];
        for (d, v) in rows(vectors).enumerate() {
            let Some(v) = v else { continue };
            for i in 0..dim {
                r[i] = v[i] - centroid[i];
            }
            let row = &mut packed[d * stride..(d + 1) * stride];
            let mut l1 = 0.0f32;
            for (i, &x) in r.iter().enumerate() {
                if x > 0.0 {
                    row[i / 64] |= 1 << (i % 64);
                }
                l1 += x.abs();
            }
            let n2 = super::distance::dot(&r, &r);
            let scale = if l1 > 0.0 { n2 / l1 } else { 0.0 };
            let cr = if metric == Metric::L2 { 0.0 } else { super::distance::dot(&centroid, &r) };
            row[words] = pair(scale, n2);
            row[words + 1] = pair(cr, inv_norm.get(d).copied().unwrap_or(0.0));
        }
        QuantizedVectors {
            dim,
            n,
            metric,
            inv_norm,
            kind: QKind::Binary { centroid, words, rows: packed },
        }
    }

    /// Prepare `query` for repeated [`distance`](Self::distance) calls.
    pub fn prepare(&self, query: &[f32]) -> QuantQuery {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let qn = super::distance::norm(query);
        let inv_qn = if qn > 0.0 { 1.0 / qn } else { 0.0 };
        let mut out = QuantQuery {
            codes: Vec::new(),
            u: Vec::new(),
            planes: Vec::new(),
            a: 0.0,
            lo: 0.0,
            step: 0.0,
            sum_rq: 0.0,
            rq2: 0.0,
            qc: 0.0,
            inv_qn,
        };
        match &self.kind {
            QKind::Int8 { q, .. } => {
                q.quantize_into(query, &mut out.codes);
                out.a = match self.metric {
                    Metric::L2 => dot_i8(&out.codes, &out.codes) as f32,
                    _ => out.codes.iter().map(|&x| x as i32).sum::<i32>() as f32,
                };
            }
            QKind::Int4 { q, normalised, .. } => {
                let unit: Vec<f32>;
                let qv: &[f32] = if *normalised && inv_qn > 0.0 {
                    unit = query.iter().map(|x| x * inv_qn).collect();
                    &unit
                } else {
                    query
                };
                let off = q.quantize_into(qv, &mut out.u);
                out.a = match self.metric {
                    Metric::L2 => int4::dot_u8(&out.u, &out.u) as f32,
                    _ => off,
                };
            }
            QKind::Binary { centroid, words, .. } => {
                let rq: Vec<f32> = query.iter().zip(centroid).map(|(x, c)| x - c).collect();
                let lo = rq.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = rq.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
                let mut planes = vec![0u64; 4 * words];
                let mut usum = 0u32;
                for (i, &x) in rq.iter().enumerate() {
                    let u = (((x - lo) / step).round() as i32).clamp(0, 15) as u32;
                    usum += u;
                    for (j, plane) in planes.chunks_exact_mut(*words).enumerate() {
                        if u >> j & 1 == 1 {
                            plane[i / 64] |= 1 << (i % 64);
                        }
                    }
                }
                out.planes = planes;
                out.lo = lo;
                out.step = step;
                out.sum_rq = lo * self.dim as f32 + step * usum as f32;
                out.rq2 = super::distance::dot(&rq, &rq);
                out.qc = super::distance::dot(query, centroid);
            }
        }
        out
    }

    /// Approximate distance (lower is better, same scale as
    /// [`Similarity::distance`]) between the prepared query and vector `ord`.
    #[inline]
    pub fn distance(&self, pq: &QuantQuery, ord: u32) -> f32 {
        let o = ord as usize;
        match &self.kind {
            QKind::Int8 { q, codes, corr } => {
                let dim = self.dim;
                let c = &codes[o * dim..(o + 1) * dim];
                let d = dot_i8(&pq.codes, c) as f32;
                let s = (q.max - q.min) / 254.0;
                match self.metric {
                    Metric::L2 => s * s * (pq.a + corr[o] - 2.0 * d),
                    Metric::Dot | Metric::Cos => {
                        let b = q.min + 127.0 * s;
                        let dotf = dim as f32 * b * b + b * s * (pq.a + corr[o]) + s * s * d;
                        if self.metric == Metric::Dot {
                            -dotf
                        } else {
                            1.0 - dotf * pq.inv_qn * self.inv_norm[o]
                        }
                    }
                    Metric::Packed => unreachable!("packed rows are never quantized"),
                }
            }
            QKind::Int4 { q, row, codes, corr, .. } => {
                let c = &codes[o * row..(o + 1) * row];
                let d = int4::dot_packed_u4(c, &pq.u) as f32;
                let m = q.const_multiplier();
                match self.metric {
                    // alpha^2 * sum((qx - qy)^2)
                    Metric::L2 => m * (pq.a + corr[o] - 2.0 * d),
                    // Lucene: adj = alpha^2 * dot + off_q + off_v
                    Metric::Dot => -(m * d + pq.a + corr[o]),
                    Metric::Cos => 1.0 - (m * d + pq.a + corr[o]),
                    Metric::Packed => unreachable!("packed rows are never quantized"),
                }
            }
            QKind::Binary { words, rows, .. } => {
                let w = *words;
                let row = &rows[o * (w + 2)..(o + 1) * (w + 2)];
                let code = &row[..w];
                let (t0, t1) = (row[w], row[w + 1]);
                let (scale, rn2) = (f32::from_bits(t0 as u32), f32::from_bits((t0 >> 32) as u32));
                let (cr, inv_norm) = (f32::from_bits(t1 as u32), f32::from_bits((t1 >> 32) as u32));
                let mut pop = 0u32;
                let mut weighted = 0u32;
                for i in 0..w {
                    let b = code[i];
                    pop += b.count_ones();
                    weighted += (pq.planes[i] & b).count_ones()
                        + 2 * (pq.planes[w + i] & b).count_ones()
                        + 4 * (pq.planes[2 * w + i] & b).count_ones()
                        + 8 * (pq.planes[3 * w + i] & b).count_ones();
                }
                // <q - c, sign(r)> = 2 * sum_{bit set}(q - c) - sum(q - c)
                let s1 = pq.lo * pop as f32 + pq.step * weighted as f32;
                let qs = 2.0 * s1 - pq.sum_rq;
                let est = qs * scale; // ~ <q - c, r>
                match self.metric {
                    Metric::L2 => pq.rq2 + rn2 - 2.0 * est,
                    Metric::Dot => -(est + cr + pq.qc),
                    // `r` was built from raw (not unit) vectors for cosine
                    // too, so normalize the dot estimate.
                    Metric::Cos => 1.0 - (est + cr + pq.qc) * pq.inv_qn * inv_norm,
                    Metric::Packed => unreachable!("packed rows are never quantized"),
                }
            }
        }
    }

    /// Prefetch the codes of `ord` (graph traversal hint).
    #[inline(always)]
    pub(crate) fn prefetch(&self, ord: u32) {
        let o = ord as usize;
        match &self.kind {
            QKind::Int8 { codes, .. } => {
                if let Some(c) = codes.get(o * self.dim) {
                    super::distance::prefetch(c as *const i8 as *const u8, self.dim);
                }
            }
            QKind::Int4 { row, codes, .. } => {
                if let Some(c) = codes.get(o * row) {
                    super::distance::prefetch(c as *const u8, *row);
                }
            }
            QKind::Binary { words, rows, .. } => {
                if let Some(c) = rows.get(o * (words + 2)) {
                    super::distance::prefetch(c as *const u64 as *const u8, (words + 2) * 8);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut x = seed | 1;
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
    fn fit_quantize_dequantize_roundtrip_is_close() {
        let vecs: Vec<Vec<f32>> = (0..50).map(|i| rng_vec(i + 1, 32)).collect();
        let q = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 1.0);
        assert!(q.min <= -0.9 && q.max >= 0.9, "{q:?}");
        let step = (q.max - q.min) / 254.0;
        for v in &vecs {
            let qi = q.quantize(v);
            let back = q.dequantize(&qi);
            for (a, b) in v.iter().zip(&back) {
                assert!((a - b).abs() <= step * 0.51 + 1e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn fit_clips_tails_with_confidence_interval() {
        // Mostly in [-1,1] with a few huge outliers.
        let mut vecs: Vec<Vec<f32>> = (0..100).map(|i| rng_vec(i + 1, 16)).collect();
        vecs[0][0] = 1000.0;
        vecs[1][0] = -1000.0;
        let q99 = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 0.99);
        let q100 = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 1.0);
        assert!(q100.max >= 1000.0 && q100.min <= -1000.0);
        assert!(q99.max < 2.0 && q99.min > -2.0, "{q99:?}");
        // Outliers clamp to the extremes.
        assert_eq!(q99.quantize(&[1000.0])[0], 127);
        assert_eq!(q99.quantize(&[-1000.0])[0], -127);
    }

    #[test]
    fn fit_handles_empty_and_constant() {
        let q = ScalarQuantizer::fit(std::iter::empty::<&[f32]>(), 0.99);
        assert_eq!(q, ScalarQuantizer { min: -1.0, max: 1.0 });
        let c = [0.5f32; 8];
        let q = ScalarQuantizer::fit([c.as_slice()], 0.99);
        assert!(q.min < 0.5 && q.max > 0.5);
        let back = q.dequantize(&q.quantize(&c));
        for b in back {
            assert!((b - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn dot_i8_full_range_and_tails() {
        for len in (0usize..=135).chain([256, 384, 768, 1536, 16384]) {
            let a: Vec<i8> = (0..len).map(|i| (i.wrapping_mul(71).wrapping_add(128)) as i8).collect();
            let b: Vec<i8> = (0..len).map(|i| (i.wrapping_mul(137).wrapping_add(255)) as i8).collect();
            let expected: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
            assert_eq!(dot_i8(&a, &b), expected, "len={len}");
            let a = vec![i8::MIN; len];
            let b = vec![i8::MAX; len];
            assert_eq!(dot_i8(&a, &b), (len as i32) * i8::MIN as i32 * i8::MAX as i32);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: checked the extension and both lengths.
                assert_eq!(unsafe { dot_i8_dotprod(a.as_ptr(), b.as_ptr(), len) }, dot_i8(&a, &b));
            }
        }
    }

    #[test]
    fn dot_i8_matches_naive() {
        let mut x = 7u64;
        for len in 1..=70usize {
            let mut next = || {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u8 as i8
            };
            let a: Vec<i8> = (0..len).map(|_| next()).collect();
            let b: Vec<i8> = (0..len).map(|_| next()).collect();
            let naive: i32 = a.iter().zip(&b).map(|(p, q)| *p as i32 * *q as i32).sum();
            assert_eq!(dot_i8(&a, &b), naive, "len={len}");
        }
    }

    #[test]
    fn int8_column_roundtrip_and_source() {
        let dim = 12u32;
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| rng_vec(i + 3, dim as usize)).collect();
        let q = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 0.99);
        let rows: Vec<Vec<i8>> = vecs.iter().map(|v| q.quantize(v)).collect();
        let data = Int8Vectors::encode(dim, q, &rows);
        let col = Int8Vectors::open(&data).unwrap();
        assert_eq!(col.dim(), dim);
        assert_eq!(col.num_docs(), 20);
        assert_eq!(col.quantizer(), q);
        for (d, r) in rows.iter().enumerate() {
            assert_eq!(col.get(d as u32).unwrap(), r.as_slice());
        }
        assert!(col.get(20).is_none());
        let src = col.as_source();
        assert_eq!(src.dim(), dim);
        assert_eq!(src.num_docs(), 20);
        let step = (q.max - q.min) / 254.0;
        for (d, v) in vecs.iter().enumerate() {
            let got = src.get(d as u32).unwrap().to_vec();
            for (a, b) in v.iter().zip(&got) {
                assert!((a - b).abs() <= step * 0.51 + 1e-6);
            }
        }
        assert!(src.get(99).is_none());
        // Truncation is detected.
        assert!(Int8Vectors::open(&data[..data.len() - 1]).is_err());
        assert!(Int8Vectors::open(&data[..8]).is_err());
    }
}
