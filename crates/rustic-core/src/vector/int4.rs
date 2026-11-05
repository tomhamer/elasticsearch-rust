//! Lucene-style scalar quantization (`Lucene99ScalarQuantizedVectorsFormat`)
//! and the packed 4-bit column it produces (ES `int4_hnsw` / `int4_flat`).
//!
//! # Quantizer
//!
//! [`UniformQuantizer`] is Lucene's `ScalarQuantizer`: a component `x` is
//! clamped to `[min, max]` and mapped to the unsigned code
//! `round((x - min) * levels / (max - min))` with `levels = 2^bits - 1`
//! (`15` for int4, `127` for Lucene's 7-bit "int8"). `alpha = (max - min) /
//! levels` maps a code back. Two fits are provided, both ported from Lucene:
//!
//! * [`UniformQuantizer::fit`] — a fixed *confidence interval*: per batch of
//!   20 sampled vectors the central `ci` fraction of component values gives
//!   a `(lower, upper)` pair, averaged over batches (`fromVectors`).
//! * [`UniformQuantizer::fit_auto`] — Lucene's *dynamic* interval
//!   (`confidence_interval: 0`, `fromVectorsAutoInterval`; ES 9 `int4_hnsw`
//!   mappings select OSQ instead, see [`super::osq`]): quantiles at two
//!   confidence intervals bound a 16 x 16 grid of `(lower, upper)`
//!   candidates; each candidate is scored by how well the quantized scores
//!   of every sampled vector against its 10 nearest sampled neighbours
//!   correlate with the exact scores, and the best pair wins (coarse 4 x 4
//!   pass, then the winning quadrant).
//!
//! The batch quantile selector, the candidate grid, the score-error
//! correlation and the two-pass grid search follow Lucene line by line (see
//! the module tests). The reservoir sampler is a bit-exact port of
//! `java.util.Random` seeded with `Random(42)` per fit.
//!
//! # Scores
//!
//! [`UniformQuantizer::score`] is `ScalarQuantizedVectorSimilarity`: the
//! `_score` ES returns from a quantized index that is not rescored.
//! `l2_norm`: `1 / (1 + alpha^2 * sqdist(codes))`. Dot-product family:
//! `adj = alpha^2 * dot(codes) + off_q + off_v` where each vector's `off`
//! (Lucene's corrective offset, computed from the floats at quantization
//! time) is `sum(min * (x - min / 2) + (dx - dxq) * dxq)`; `dot_product` /
//! `cosine` score `max((1 + adj) / 2, 0)` (cosine quantizes unit vectors),
//! `max_inner_product` scores `adj + 1` or `1 / (1 - adj)`.
//!
//! # Packed column ([`Int4Vectors`])
//!
//! Codes are packed two per byte exactly like Lucene's `compress=true`
//! layout: byte `i` holds code `i` in its high nibble and code `i + dim/2`
//! in its low nibble (ES requires even `dims`; an odd `dim` pads the last
//! low nibble with 0). Wire format (little-endian):
//!
//! ```text
//! u32 dim, u32 num_docs, f32 min, f32 max, u8 bits (4), u8 flags, u16 0, u32 0   // 24 bytes
//! [f32 offset] x num_docs            // Lucene corrective offsets (0 for l2_norm)
//! [u8 x ceil(dim / 2)] x num_docs    // packed codes, one row per ordinal
//! ```
//!
//! `flags` bit 0: the stored codes are of the *unit-normalised* vectors
//! (cosine similarity, as Lucene).

use crate::DocId;
use crate::error::{Result, RusticError};

use super::distance;
use super::{Metric, Similarity};

/// Lucene's minimum confidence interval (`Lucene99ScalarQuantizedVectorsFormat`).
pub const MIN_CONFIDENCE_INTERVAL: f32 = 0.9;
/// The dynamic-interval marker (ES `confidence_interval: 0`).
pub const DYNAMIC_CONFIDENCE_INTERVAL: f32 = 0.0;
/// Vectors Lucene samples for a fixed-interval fit.
pub const QUANTIZATION_SAMPLE_SIZE: usize = 25_000;
/// Vectors Lucene samples for the dynamic fit.
const AUTO_SAMPLE_SIZE: usize = 1000;
/// Quantiles are taken per batch of this many sampled vectors.
const SCRATCH_SIZE: usize = 20;

/// Lucene's default confidence interval for `dim`: `max(0.9, 1 - 1 / (dim + 1))`.
pub fn default_confidence_interval(dim: usize) -> f32 {
    (1.0 - 1.0 / (dim as f32 + 1.0)).max(MIN_CONFIDENCE_INTERVAL)
}

/// A bit-exact port of `java.util.Random` (48-bit LCG), used for Lucene's
/// reservoir sampling.
#[derive(Clone, Debug)]
pub struct JavaRandom(u64);

impl JavaRandom {
    const MULT: u64 = 0x5DEECE66D;
    const MASK: u64 = (1 << 48) - 1;

    pub fn new(seed: i64) -> Self {
        Self(((seed as u64) ^ Self::MULT) & Self::MASK)
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.0 = (self.0.wrapping_mul(Self::MULT).wrapping_add(0xB)) & Self::MASK;
        (self.0 >> (48 - bits)) as i32
    }

    /// `Random.nextInt()`.
    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    /// `Random.nextInt(bound)` for `bound > 0`.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0);
        if bound & bound.wrapping_neg() == bound {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
                return val;
            }
        }
    }
}

/// Lucene `reservoirSampleIndices`: `sample` ascending indices out of `n`.
pub fn reservoir_sample_indices(n: usize, sample: usize) -> Vec<usize> {
    let mut rng = JavaRandom::new(42);
    let mut take: Vec<usize> = (0..sample.min(n)).collect();
    for i in sample..n {
        let j = rng.next_int_bounded((i + 1) as i32) as usize;
        if j < sample {
            take[j] = i;
        }
    }
    take.sort_unstable();
    take
}

/// Lucene `getUpperAndLowerQuantile`: `(lower, upper)` bounds of the central
/// `ci` fraction of `values` (the whole range when there are at most two).
pub fn quantile_bounds(values: &mut [f32], ci: f32) -> (f32, f32) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    values.sort_unstable_by(f32::total_cmp);
    if n <= 2 {
        return (values[0], values[n - 1]);
    }
    let sel = (n as f32 * (1.0 - ci) / 2.0 + 0.5) as usize;
    let sel = sel.min((n - 1) / 2);
    (values[sel], values[n - 1 - sel])
}

/// A fitted Lucene-style `bits`-bit scalar quantizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformQuantizer {
    /// Lower clip bound (code 0).
    pub min: f32,
    /// Upper clip bound (code `2^bits - 1`).
    pub max: f32,
    /// Bits per code (`1..=8`).
    pub bits: u8,
}

impl UniformQuantizer {
    pub fn new(min: f32, max: f32, bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        let (min, max) = if min.is_finite() && max.is_finite() { (min, max) } else { (-1.0, 1.0) };
        // Degenerate (constant) data: a hair of width keeps `scale` finite.
        let max = if max > min { max } else { min + 1e-6 };
        Self { min, max, bits }
    }

    /// Highest code.
    #[inline]
    pub fn levels(&self) -> u32 {
        (1u32 << self.bits) - 1
    }

    /// Lucene `alpha`: the value of one code step.
    #[inline]
    pub fn alpha(&self) -> f32 {
        (self.max - self.min) / self.levels() as f32
    }

    /// Lucene `scale`: codes per unit of input.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.levels() as f32 / (self.max - self.min)
    }

    /// Lucene `getConstantMultiplier`: `alpha^2`.
    #[inline]
    pub fn const_multiplier(&self) -> f32 {
        let a = self.alpha();
        a * a
    }

    /// Quantize `v` into `out` (cleared first) and return Lucene's
    /// corrective offset (meaningful for the dot-product family; ignore it
    /// for `l2_norm`). Each code is `0..=levels`.
    pub fn quantize_into(&self, v: &[f32], out: &mut Vec<u8>) -> f32 {
        out.clear();
        out.reserve(v.len());
        let scale = self.scale();
        let alpha = self.alpha();
        let min = self.min;
        let mut corr = 0.0f32;
        for &x in v {
            let dx = x - min;
            let dxc = x.clamp(min, self.max) - min;
            let dxs = scale * dxc;
            // Java `Math.round`: floor(x + 0.5).
            let q = (dxs + 0.5).floor();
            let dxq = q * alpha;
            out.push(q as u8);
            corr += min * (x - min / 2.0) + (dx - dxq) * dxq;
        }
        corr
    }

    /// Quantize one vector.
    pub fn quantize(&self, v: &[f32]) -> (Vec<u8>, f32) {
        let mut out = Vec::with_capacity(v.len());
        let corr = self.quantize_into(v, &mut out);
        (out, corr)
    }

    /// Dequantize codes (`alpha * code + min`).
    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        let a = self.alpha();
        codes.iter().map(|&c| a * c as f32 + self.min).collect()
    }

    /// Lucene's quantized score of two code vectors with their offsets (the
    /// `_score` ES returns without rescoring). `cosine` expects codes of
    /// unit-normalised vectors.
    pub fn score(&self, sim: Similarity, q: &[u8], q_off: f32, v: &[u8], v_off: f32) -> f32 {
        match sim {
            Similarity::L2 => 1.0 / (1.0 + sqdist_u8(q, v) as f32 * self.const_multiplier()),
            Similarity::DotProduct | Similarity::Cosine => {
                let adj = dot_u8(q, v) as f32 * self.const_multiplier() + q_off + v_off;
                ((1.0 + adj) / 2.0).max(0.0)
            }
            Similarity::MaxInnerProduct => {
                let adj = dot_u8(q, v) as f32 * self.const_multiplier() + q_off + v_off;
                scale_mip(adj)
            }
            packed => unreachable!("packed similarity {packed:?} is never quantized"),
        }
    }

    /// Lucene `fromVectors`: fit with a fixed confidence interval `ci` in
    /// `[0.9, 1]` (`1` = plain min/max; anything below 0.9 is raised to
    /// 0.9). `n` vectors are read through `row`; at most 25k are sampled.
    pub fn fit<'a>(n: usize, row: impl Fn(usize) -> &'a [f32], ci: f32, bits: u8) -> Self {
        Self::fit_sampled(n, row, ci, bits, QUANTIZATION_SAMPLE_SIZE)
    }

    fn fit_sampled<'a>(n: usize, row: impl Fn(usize) -> &'a [f32], ci: f32, bits: u8, sample: usize) -> Self {
        let ci = if ci.is_finite() { ci.clamp(MIN_CONFIDENCE_INTERVAL, 1.0) } else { 1.0 };
        if n == 0 {
            return Self::new(0.0, 0.0, bits);
        }
        if ci == 1.0 {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..n {
                for &x in row(i) {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
            return Self::new(lo, hi, bits);
        }
        let (lo_sum, hi_sum, count) = if n <= sample {
            let batch = SCRATCH_SIZE.min(n);
            batched_quantiles((0..n).map(&row), batch, &[ci])
        } else {
            let idx = reservoir_sample_indices(n, sample);
            batched_quantiles(idx.into_iter().map(&row), SCRATCH_SIZE, &[ci])
        };
        let c = count.max(1) as f64;
        Self::new((lo_sum[0] / c) as f32, (hi_sum[0] / c) as f32, bits)
    }

    /// Lucene `fromVectorsAutoInterval`: the dynamic confidence interval (ES
    /// `confidence_interval: 0`). `sim` picks the score the candidate
    /// intervals are correlated against; `cosine` normalises the vectors and
    /// correlates dot products, as Lucene does.
    pub fn fit_auto<'a>(n: usize, row: impl Fn(usize) -> &'a [f32], sim: Similarity, bits: u8) -> Self {
        if n == 0 {
            return Self::new(0.0, 0.0, bits);
        }
        let dim = row(0).len();
        let normalise = sim == Similarity::Cosine;
        let sim = if normalise { Similarity::DotProduct } else { sim };
        let norm = |v: &[f32]| -> Vec<f32> {
            if normalise {
                let nn = distance::norm(v);
                if nn > 0.0 { v.iter().map(|x| x / nn).collect() } else { v.to_vec() }
            } else {
                v.to_vec()
            }
        };
        let sampled: Vec<Vec<f32>> = if n <= AUTO_SAMPLE_SIZE {
            (0..n).map(|i| norm(row(i))).collect()
        } else {
            reservoir_sample_indices(n, AUTO_SAMPLE_SIZE).into_iter().map(|i| norm(row(i))).collect()
        };
        let cis = [1.0 - (dim as f32 / 10.0).min(32.0) / (dim as f32 + 1.0), 1.0 - 1.0 / (dim as f32 + 1.0)];
        let batch = if n <= AUTO_SAMPLE_SIZE { SCRATCH_SIZE.min(n) } else { SCRATCH_SIZE };
        let (lo_sum, hi_sum, count) = batched_quantiles(sampled.iter().map(|v| v.as_slice()), batch, &cis);
        let c = count.max(1) as f64;
        // Grid bounds (Lucene's naming): lower candidates run from `al`
        // (lower quantile at the wide interval) to `au` (at the narrow one),
        // upper candidates from `bl` to `bu`.
        let al = (lo_sum[1] / c) as f32;
        let bu = (hi_sum[1] / c) as f32;
        let au = (lo_sum[0] / c) as f32;
        let bl = (hi_sum[0] / c) as f32;
        if ![al, au, bl, bu].iter().all(|x| x.is_finite()) {
            return Self::new(al.min(au), bl.max(bu), bits);
        }
        let mut lower = [0.0f32; 16];
        let mut upper = [0.0f32; 16];
        for (idx, i) in (0..16).map(|i| (i, (2 * i) as f32)) {
            lower[idx] = al + i * (au - al) / 32.0;
            upper[idx] = bl + i * (bu - bl) / 32.0;
        }
        let neighbours = nearest_neighbours(&sampled, sim);
        let (lo, hi) = grid_search(&neighbours, &sampled, &lower, &upper, sim, bits);
        Self::new(lo, hi, bits)
    }
}

/// `Math.max((1 + x) ...)` helper for max-inner-product (Lucene
/// `scaleMaxInnerProductScore`).
#[inline]
pub fn scale_mip(adj: f32) -> f32 {
    if adj < 0.0 { 1.0 / (1.0 + -adj) } else { adj + 1.0 }
}

/// Per-batch quantiles of the component values of `rows`, summed over the
/// full batches (a trailing partial batch is ignored, like Lucene).
/// Returns `(lower sums, upper sums, batches)` per confidence interval.
fn batched_quantiles<'a>(rows: impl Iterator<Item = &'a [f32]>, batch: usize, cis: &[f32]) -> (Vec<f64>, Vec<f64>, usize) {
    let mut lo = vec![0.0f64; cis.len()];
    let mut hi = vec![0.0f64; cis.len()];
    let mut count = 0usize;
    let mut scratch: Vec<f32> = Vec::new();
    let mut in_batch = 0usize;
    let mut sorted: Vec<f32> = Vec::new();
    for v in rows {
        scratch.extend_from_slice(v);
        in_batch += 1;
        if in_batch == batch {
            for (k, &ci) in cis.iter().enumerate() {
                sorted.clear();
                sorted.extend_from_slice(&scratch);
                let (l, u) = quantile_bounds(&mut sorted, ci);
                lo[k] += l as f64;
                hi[k] += u as f64;
            }
            count += 1;
            in_batch = 0;
            scratch.clear();
        }
    }
    (lo, hi, count)
}

/// Lucene `findNearestNeighbors`: per sampled vector, its 10 nearest
/// sampled neighbours `(index, exact score)` and the variance of those
/// scores (sample variance, `n - 1`).
fn nearest_neighbours(vectors: &[Vec<f32>], sim: Similarity) -> Vec<(Vec<(usize, f32)>, f32)> {
    let n = vectors.len();
    let mut queues: Vec<Vec<(usize, f32)>> = vec![Vec::with_capacity(11); n];
    let push = |q: &mut Vec<(usize, f32)>, item: (usize, f32)| {
        // Keep the 10 best (highest score); ties keep the earliest.
        if q.len() < 10 {
            q.push(item);
        } else if let Some((wi, _)) = q.iter().enumerate().min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)) {
            if item.1 > q[wi].1 {
                q[wi] = item;
            }
        }
    };
    for i in 0..n {
        for j in i + 1..n {
            let s = sim.score(&vectors[i], &vectors[j]);
            push(&mut queues[i], (j, s));
            push(&mut queues[j], (i, s));
        }
    }
    queues
        .into_iter()
        .map(|q| {
            let mut mv = OnlineMeanVar::default();
            for &(_, s) in &q {
                mv.add(s as f64);
            }
            (q, mv.var())
        })
        .collect()
}

#[derive(Default)]
struct OnlineMeanVar {
    mean: f64,
    m2: f64,
    n: usize,
}

impl OnlineMeanVar {
    fn add(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }
    /// Sample variance (`n - 1`); NaN with fewer than two values, as Java.
    fn var(&self) -> f32 {
        (self.m2 / (self.n as f64 - 1.0)) as f32
    }
}

/// Lucene `ScoreErrorCorrelator.scoreErrorCorrelation` for one candidate.
fn score_error_correlation(
    neighbours: &[(Vec<(usize, f32)>, f32)],
    vectors: &[Vec<f32>],
    lo: f32,
    hi: f32,
    sim: Similarity,
    bits: u8,
    qbuf: &mut Vec<u8>,
    vbuf: &mut Vec<u8>,
) -> f64 {
    let q = UniformQuantizer::new(lo, hi, bits);
    let mut corr = OnlineMeanVar::default();
    for (i, (nn, score_var)) in neighbours.iter().enumerate() {
        let q_off = q.quantize_into(&vectors[i], qbuf);
        let mut errors = OnlineMeanVar::default();
        for &(j, exact) in nn {
            let v_off = q.quantize_into(&vectors[j], vbuf);
            let qs = q.score(sim, qbuf, q_off, vbuf, v_off);
            errors.add((qs - exact) as f64);
        }
        corr.add(1.0 - (errors.var() / score_var) as f64);
    }
    if corr.mean.is_nan() { 0.0 } else { corr.mean }
}

/// Lucene `candidateGridSearch`: coarse 4 x 4 pass over the 16 x 16 grid,
/// then the 3 x 3 remainder of the winning quadrant.
fn grid_search(
    neighbours: &[(Vec<(usize, f32)>, f32)],
    vectors: &[Vec<f32>],
    lower: &[f32; 16],
    upper: &[f32; 16],
    sim: Similarity,
    bits: u8,
) -> (f32, f32) {
    let mut best = f64::NEG_INFINITY;
    let (mut bl, mut bu) = (0.0f32, 0.0f32);
    let (mut qi, mut qj) = (0usize, 0usize);
    let mut qbuf = Vec::new();
    let mut vbuf = Vec::new();
    let mut try_pair = |i: usize, j: usize, best: &mut f64, bl: &mut f32, bu: &mut f32| -> bool {
        let (lo, hi) = (lower[i], upper[j]);
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return false;
        }
        let m = score_error_correlation(neighbours, vectors, lo, hi, sim, bits, &mut qbuf, &mut vbuf);
        if m > *best {
            *best = m;
            *bl = lo;
            *bu = hi;
            return true;
        }
        false
    };
    for i in (0..16).step_by(4) {
        for j in (0..16).step_by(4) {
            if try_pair(i, j, &mut best, &mut bl, &mut bu) {
                qi = i;
                qj = j;
            }
        }
    }
    for i in qi + 1..qi + 4 {
        for j in qj + 1..qj + 4 {
            try_pair(i, j, &mut best, &mut bl, &mut bu);
        }
    }
    if best == f64::NEG_INFINITY {
        // Every pair degenerate: fall back to the widest interval.
        return (lower.iter().copied().fold(f32::INFINITY, f32::min), upper.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    }
    (bl, bu)
}

// ---------------------------------------------------------------------------
// Integer kernels.

/// Dot product of two unsigned code vectors (codes `<= 255`, so `u32` holds
/// any dimension up to 65k).
#[inline]
pub fn dot_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "dot_u8: length mismatch");
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: extension checked above; slices have equal lengths.
        return unsafe { dot_u8_dotprod(a.as_ptr(), b.as_ptr(), a.len()) };
    }
    let mut s = [0u32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..16 {
            s[i] += x[i] as u32 * y[i] as u32;
        }
    }
    let mut t: u32 = s.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        t += *x as u32 * *y as u32;
    }
    t
}

/// [`dot_u8`] of `q` with four rows at once (the query loaded once per
/// 16 bytes): a traversal scores a neighbourhood's rows in groups of four.
/// Each result equals `dot_u8(q, row)` (exact integer sums).
pub fn dot_u8_four(q: &[u8], rows: [&[u8]; 4]) -> [u32; 4] {
    assert!(rows.iter().all(|r| r.len() == q.len()), "dot_u8_four: length mismatch");
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: extension checked above; every row has the query's length.
        return unsafe { dot_u8_four_dotprod(q.as_ptr(), rows.map(<[u8]>::as_ptr), q.len()) };
    }
    rows.map(|row| dot_u8(q, row))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_u8_four_dotprod(q: *const u8, rows: [*const u8; 4], n: usize) -> [u32; 4] {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_u32(0); 4];
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_u8(q.add(i));
            for (r, a) in rows.iter().zip(acc.iter_mut()) {
                let y = vld1q_u8(r.add(i));
                core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) *a, x = in(vreg) x, y = in(vreg) y,
                    options(pure, nomem, nostack, preserves_flags));
            }
            i += 16;
        }
        let mut out = [vaddvq_u32(acc[0]), vaddvq_u32(acc[1]), vaddvq_u32(acc[2]), vaddvq_u32(acc[3])];
        while i < n {
            let x = u32::from(*q.add(i));
            for (o, r) in out.iter_mut().zip(rows.iter()) { *o += x * u32::from(*r.add(i)); }
            i += 1;
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_u8_dotprod(a: *const u8, b: *const u8, n: usize) -> u32 {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = acc0;
        let mut acc2 = acc0;
        let mut acc3 = acc0;
        let mut i = 0;
        while i + 64 <= n {
            let x = vld1q_u8(a.add(i));
            let y = vld1q_u8(b.add(i));
            // The stable toolchain lacks the dot-product intrinsic. The
            // feature-gated instruction reads only these SIMD registers.
            core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc0, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_u8(a.add(i + 16));
            let y = vld1q_u8(b.add(i + 16));
            core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc1, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_u8(a.add(i + 32));
            let y = vld1q_u8(b.add(i + 32));
            core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc2, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            let x = vld1q_u8(a.add(i + 48));
            let y = vld1q_u8(b.add(i + 48));
            core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc3, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            i += 64;
        }
        while i + 16 <= n {
            let x = vld1q_u8(a.add(i));
            let y = vld1q_u8(b.add(i));
            core::arch::asm!("udot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc0, x = in(vreg) x, y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags));
            i += 16;
        }
        let mut sum = vaddvq_u32(vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3)));
        while i < n {
            sum += *a.add(i) as u32 * *b.add(i) as u32;
            i += 1;
        }
        sum
    }
}

/// Squared Euclidean distance of two unsigned code vectors.
#[inline]
pub fn sqdist_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "sqdist_u8: length mismatch");
    let mut s = [0u32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..16 {
            let d = x[i] as i32 - y[i] as i32;
            s[i] += (d * d) as u32;
        }
    }
    let mut t: u32 = s.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = *x as i32 - *y as i32;
        t += (d * d) as u32;
    }
    t
}

/// Bytes per packed row of `dim` 4-bit codes.
#[inline]
pub fn packed_len(dim: usize) -> usize {
    dim.div_ceil(2)
}

/// Pack 4-bit codes Lucene-style: byte `i` = `codes[i] << 4 | codes[i + half]`.
pub fn pack_nibbles(codes: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let half = packed_len(codes.len());
    for i in 0..half {
        let hi = codes[i] & 15;
        let lo = codes.get(i + half).copied().unwrap_or(0) & 15;
        out.push(hi << 4 | lo);
    }
}

/// Inverse of [`pack_nibbles`] for a row of `dim` codes.
pub fn unpack_nibbles(packed: &[u8], dim: usize, out: &mut Vec<u8>) {
    out.clear();
    out.resize(dim, 0);
    let half = packed_len(dim);
    for (i, &b) in packed.iter().enumerate().take(half) {
        out[i] = b >> 4;
        if i + half < dim {
            out[i + half] = b & 15;
        }
    }
}

/// Dot product of a packed row with an unpacked query of `dim` codes.
#[inline]
pub fn dot_packed_u4(packed: &[u8], query: &[u8]) -> u32 {
    let dim = query.len();
    let half = packed_len(dim);
    debug_assert_eq!(packed.len(), half);
    let (q_hi, q_lo) = query.split_at(half.min(dim));
    let mut s_hi = [0u32; 16];
    let mut s_lo = [0u32; 16];
    let mut cp = packed.chunks_exact(16);
    let mut ch = q_hi.chunks_exact(16);
    let mut cl = q_lo.chunks_exact(16);
    // Even dims: the three streams are the same length.
    if q_lo.len() == half {
        for ((p, h), l) in (&mut cp).zip(&mut ch).zip(&mut cl) {
            for i in 0..16 {
                s_hi[i] += (p[i] >> 4) as u32 * h[i] as u32;
                s_lo[i] += (p[i] & 15) as u32 * l[i] as u32;
            }
        }
        let mut t: u32 = s_hi.iter().sum::<u32>() + s_lo.iter().sum::<u32>();
        for ((p, h), l) in cp.remainder().iter().zip(ch.remainder()).zip(cl.remainder()) {
            t += (*p >> 4) as u32 * *h as u32 + (*p & 15) as u32 * *l as u32;
        }
        return t;
    }
    let mut t = 0u32;
    for (i, &p) in packed.iter().enumerate() {
        t += (p >> 4) as u32 * q_hi[i] as u32;
        if let Some(&l) = q_lo.get(i) {
            t += (p & 15) as u32 * l as u32;
        }
    }
    t
}

/// `sum(code^2)` of a packed row.
#[inline]
pub fn sqsum_packed_u4(packed: &[u8]) -> u32 {
    packed.iter().map(|&p| ((p >> 4) as u32).pow(2) + ((p & 15) as u32).pow(2)).sum()
}

// ---------------------------------------------------------------------------
// Stored column.

/// Reader over an encoded packed int4 column (see the module docs).
pub struct Int4Vectors<'a> {
    dim: u32,
    num_docs: u32,
    quantizer: UniformQuantizer,
    normalised: bool,
    data: &'a [u8],
}

impl<'a> Int4Vectors<'a> {
    const HEADER: usize = 24;
    const FLAG_NORMALISED: u8 = 1;

    /// Encode a column: `rows` are packed code rows (one per ordinal, in
    /// order) and `offsets` their Lucene corrective offsets (any length is
    /// accepted; missing offsets are 0).
    pub fn encode(dim: u32, quantizer: UniformQuantizer, normalised: bool, rows: &[Vec<u8>], offsets: &[f32]) -> Vec<u8> {
        let n = rows.len();
        let pl = packed_len(dim as usize);
        let mut out = Vec::with_capacity(Self::HEADER + n * (4 + pl));
        out.extend_from_slice(&dim.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&quantizer.min.to_le_bytes());
        out.extend_from_slice(&quantizer.max.to_le_bytes());
        out.push(quantizer.bits);
        out.push(if normalised { Self::FLAG_NORMALISED } else { 0 });
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for i in 0..n {
            out.extend_from_slice(&offsets.get(i).copied().unwrap_or(0.0).to_le_bytes());
        }
        for r in rows {
            assert_eq!(r.len(), pl, "int4 row length mismatch");
            out.extend_from_slice(r);
        }
        out
    }

    /// Parse an encoded column.
    pub fn open(data: &'a [u8]) -> Result<Int4Vectors<'a>> {
        if data.len() < Self::HEADER {
            return Err(RusticError::Corrupted("int4 vectors: truncated header".into()));
        }
        let rd = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let dim = rd(0);
        let num_docs = rd(4);
        let min = f32::from_bits(rd(8));
        let max = f32::from_bits(rd(12));
        let bits = data[16];
        let flags = data[17];
        if bits != 4 {
            return Err(RusticError::Corrupted(format!("int4 vectors: unsupported bits {bits}")));
        }
        if !(min < max) || !min.is_finite() || !max.is_finite() {
            return Err(RusticError::Corrupted("int4 vectors: invalid min/max".into()));
        }
        let n = num_docs as usize;
        let expected = Self::HEADER + n * 4 + n * packed_len(dim as usize);
        if data.len() < expected {
            return Err(RusticError::Corrupted("int4 vectors: truncated data".into()));
        }
        Ok(Int4Vectors {
            dim,
            num_docs,
            quantizer: UniformQuantizer { min, max, bits },
            normalised: flags & Self::FLAG_NORMALISED != 0,
            data: &data[..expected],
        })
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn quantizer(&self) -> UniformQuantizer {
        self.quantizer
    }

    /// Whether the codes are of unit-normalised vectors (cosine).
    pub fn normalised(&self) -> bool {
        self.normalised
    }

    /// Lucene corrective offset of `doc`.
    pub fn offset(&self, doc: DocId) -> Option<f32> {
        if doc >= self.num_docs {
            return None;
        }
        let at = Self::HEADER + doc as usize * 4;
        Some(f32::from_le_bytes(self.data[at..at + 4].try_into().unwrap()))
    }

    /// The packed code row of `doc`.
    pub fn get(&self, doc: DocId) -> Option<&'a [u8]> {
        if doc >= self.num_docs {
            return None;
        }
        let pl = packed_len(self.dim as usize);
        let start = Self::HEADER + self.num_docs as usize * 4 + doc as usize * pl;
        Some(&self.data[start..start + pl])
    }

    /// Every packed row, contiguous.
    pub fn rows(&self) -> &'a [u8] {
        let start = Self::HEADER + self.num_docs as usize * 4;
        &self.data[start..]
    }

    /// Every offset, in order.
    pub fn offsets(&self) -> Vec<f32> {
        (0..self.num_docs).filter_map(|d| self.offset(d)).collect()
    }

    /// Dequantized vector of `doc` (unit-normalised for cosine columns).
    pub fn dequantize(&self, doc: DocId) -> Option<Vec<f32>> {
        let mut codes = Vec::new();
        unpack_nibbles(self.get(doc)?, self.dim as usize, &mut codes);
        Some(self.quantizer.dequantize(&codes))
    }
}

/// Quantize every row of a float column into packed int4 rows + offsets,
/// normalising first for cosine. Returns `(quantizer, normalised, rows, offsets)`.
pub fn build_int4<'a>(n: usize, row: impl Fn(usize) -> &'a [f32], sim: Similarity, confidence_interval: Option<f32>) -> (UniformQuantizer, bool, Vec<Vec<u8>>, Vec<f32>) {
    let normalised = sim == Similarity::Cosine;
    let dim = if n > 0 { row(0).len() } else { 0 };
    let unit_into = |v: &[f32], out: &mut Vec<f32>| {
        let nn = distance::norm(v);
        out.clear();
        if nn > 0.0 { out.extend(v.iter().map(|x| x / nn)); } else { out.extend_from_slice(v); }
    };
    let unit = |v: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(v.len());
        unit_into(v, &mut out);
        out
    };
    // Fit on the (normalised) rows. Cosine normalises only the rows the fit
    // samples (Lucene samples at most 25k), never a copy of every row; the
    // dynamic interval normalises its own sample.
    let q = match confidence_interval {
        Some(ci) if ci == DYNAMIC_CONFIDENCE_INTERVAL => UniformQuantizer::fit_auto(n, &row, sim, 4),
        ci => {
            let ci = ci.unwrap_or_else(|| default_confidence_interval(dim));
            if normalised {
                let picked: Vec<usize> = if n <= QUANTIZATION_SAMPLE_SIZE { (0..n).collect() } else { reservoir_sample_indices(n, QUANTIZATION_SAMPLE_SIZE) };
                let sample: Vec<Vec<f32>> = picked.into_iter().map(|i| unit(row(i))).collect();
                UniformQuantizer::fit(sample.len(), |i| sample[i].as_slice(), ci, 4)
            } else {
                UniformQuantizer::fit(n, &row, ci, 4)
            }
        }
    };
    let mut rows = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n);
    let mut codes = Vec::with_capacity(dim);
    let mut packed = Vec::with_capacity(packed_len(dim));
    let mut scratch = Vec::with_capacity(dim);
    for i in 0..n {
        let v = if normalised { unit_into(row(i), &mut scratch); scratch.as_slice() } else { row(i) };
        let off = q.quantize_into(v, &mut codes);
        pack_nibbles(&codes, &mut packed);
        rows.push(packed.clone());
        offsets.push(if sim.metric() == Metric::L2 { 0.0 } else { off });
    }
    (q, normalised, rows, offsets)
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
    fn java_random_is_bit_exact() {
        // `new java.util.Random(42).nextInt()` and the first bounded draws
        // (verified with the JDK bundled in ES 8.19.22).
        let mut r = JavaRandom::new(42);
        assert_eq!(r.next_int(), -1170105035);
        let mut r = JavaRandom::new(42);
        let v: Vec<i32> = (0..5).map(|_| r.next_int_bounded(100)).collect();
        assert_eq!(v, vec![30, 63, 48, 84, 70]);
        let mut r = JavaRandom::new(42);
        let v: Vec<i32> = (0..4).map(|_| r.next_int_bounded(16)).collect();
        assert_eq!(v, vec![11, 0, 10, 0]);
        let idx = reservoir_sample_indices(50_000, 1000);
        assert_eq!(idx.len(), 1000);
        assert!(idx.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(reservoir_sample_indices(5, 10), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn quantile_bounds_match_lucene_selector() {
        let mut v: Vec<f32> = vec![230.0, 300.33, -34.8988, 15.555, -0.5, 100.0, -13.0, 14.8, 0.5, 111.3, -13.0, 14.8];
        // 12 values at ci 0.92: selectorIndex = (int)(12 * 0.08 / 2 + 0.5) = 0 -> min/max.
        assert_eq!(quantile_bounds(&mut v.clone(), 0.92), (-34.8988, 300.33));
        // ci 0.8: selectorIndex = (int)(1.2 + 0.5) = 1 -> drop one from each tail.
        assert_eq!(quantile_bounds(&mut v, 0.8), (-13.0, 230.0));
        assert_eq!(quantile_bounds(&mut [3.0, 1.0], 0.5), (1.0, 3.0));
        assert_eq!(default_confidence_interval(4), 0.9);
        assert!((default_confidence_interval(127) - (1.0 - 1.0 / 128.0)).abs() < 1e-6);
    }

    #[test]
    fn quantize_roundtrip_within_half_step_and_codes_in_range() {
        for bits in [4u8, 7] {
            let vecs: Vec<Vec<f32>> = (0..40).map(|i| rng_vec(i + 1, 32)).collect();
            let q = UniformQuantizer::fit(vecs.len(), |i| vecs[i].as_slice(), 1.0, bits);
            assert_eq!(q.levels(), (1 << bits) - 1);
            for v in &vecs {
                let (codes, _) = q.quantize(v);
                assert!(codes.iter().all(|&c| c as u32 <= q.levels()));
                let back = q.dequantize(&codes);
                for (a, b) in v.iter().zip(&back) {
                    assert!((a - b).abs() <= q.alpha() * 0.5 + 1e-5, "bits {bits}: {a} vs {b}");
                }
            }
            // Clamping.
            assert_eq!(q.quantize(&[1e9]).0[0] as u32, q.levels());
            assert_eq!(q.quantize(&[-1e9]).0[0], 0);
        }
    }

    #[test]
    fn dot_u8_full_range_and_tails() {
        for len in (0usize..=135).chain([256, 384, 768, 1536, 16384]) {
            let a: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(71).wrapping_add(128)) as u8).collect();
            let b: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(137).wrapping_add(255)) as u8).collect();
            let expected: u32 = a.iter().zip(&b).map(|(&x, &y)| x as u32 * y as u32).sum();
            assert_eq!(dot_u8(&a, &b), expected, "len={len}");
            let a = vec![u8::MIN; len];
            let b = vec![u8::MAX; len];
            assert_eq!(dot_u8(&a, &b), (len as u32) * u8::MIN as u32 * u8::MAX as u32);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: checked the extension and both lengths.
                assert_eq!(unsafe { dot_u8_dotprod(a.as_ptr(), b.as_ptr(), len) }, dot_u8(&a, &b));
            }
        }
    }

    #[test]
    fn nibble_packing_roundtrip_and_dot() {
        for dim in [1usize, 2, 3, 4, 7, 8, 31, 32, 33, 64, 128] {
            let codes: Vec<u8> = (0..dim).map(|i| (i * 7 % 16) as u8).collect();
            let query: Vec<u8> = (0..dim).map(|i| (i * 5 % 16) as u8).collect();
            let mut packed = Vec::new();
            pack_nibbles(&codes, &mut packed);
            assert_eq!(packed.len(), packed_len(dim));
            if dim % 2 == 0 {
                // Lucene layout: byte i = codes[i] << 4 | codes[i + dim/2].
                for i in 0..dim / 2 {
                    assert_eq!(packed[i], codes[i] << 4 | codes[i + dim / 2]);
                }
            }
            let mut back = Vec::new();
            unpack_nibbles(&packed, dim, &mut back);
            assert_eq!(back, codes);
            assert_eq!(dot_packed_u4(&packed, &query), dot_u8(&codes, &query));
            assert_eq!(sqsum_packed_u4(&packed), dot_u8(&codes, &codes));
            let naive_sq: u32 = codes.iter().zip(&query).map(|(a, b)| (*a as i32 - *b as i32).pow(2) as u32).sum();
            assert_eq!(sqdist_u8(&codes, &query), naive_sq);
        }
    }

    #[test]
    fn int4_column_roundtrip() {
        let dim = 6u32;
        let vecs: Vec<Vec<f32>> = (0..17).map(|i| rng_vec(i + 3, dim as usize)).collect();
        for sim in [Similarity::L2, Similarity::DotProduct, Similarity::Cosine] {
            let (q, normalised, rows, offsets) = build_int4(vecs.len(), |i| vecs[i].as_slice(), sim, Some(0.0));
            assert_eq!(normalised, sim == Similarity::Cosine);
            let bytes = Int4Vectors::encode(dim, q, normalised, &rows, &offsets);
            let col = Int4Vectors::open(&bytes).unwrap();
            assert_eq!((col.dim(), col.num_docs(), col.quantizer(), col.normalised()), (dim, 17, q, normalised));
            for (d, r) in rows.iter().enumerate() {
                assert_eq!(col.get(d as u32).unwrap(), r.as_slice());
                assert_eq!(col.offset(d as u32).unwrap(), offsets[d]);
                let back = col.dequantize(d as u32).unwrap();
                let src: Vec<f32> = if normalised {
                    let nn = distance::norm(&vecs[d]);
                    vecs[d].iter().map(|x| x / nn).collect()
                } else {
                    vecs[d].clone()
                };
                for (a, b) in src.iter().zip(&back) {
                    assert!((a - b).abs() <= q.alpha() * 0.5 + 1e-5 || *a < q.min || *a > q.max);
                }
            }
            if sim == Similarity::L2 {
                assert!(offsets.iter().all(|o| *o == 0.0));
            }
            assert!(col.get(17).is_none());
            assert!(Int4Vectors::open(&bytes[..bytes.len() - 1]).is_err());
            assert!(Int4Vectors::open(&bytes[..10]).is_err());
        }
    }

    /// The exact numbers ES 8.19.22 returns for the `42_knn_search_int4_flat`
    /// documents (verified against a reference node): the dynamic interval
    /// puts the query and doc 2 on identical codes (score 1.0) and keeps doc
    /// 3 about one step away.
    #[test]
    fn dynamic_interval_reproduces_es_int4_scores() {
        let docs: Vec<Vec<f32>> = vec![vec![230.0, 300.33, -34.8988, 15.555], vec![-0.5, 100.0, -13.0, 14.8], vec![0.5, 111.3, -13.0, 14.8]];
        let q = UniformQuantizer::fit_auto(3, |i| docs[i].as_slice(), Similarity::L2, 4);
        let query = [-0.5f32, 90.0, -10.0, 14.8];
        let (qc, qo) = q.quantize(&query);
        let scores: Vec<f32> = docs
            .iter()
            .map(|d| {
                let (dc, doff) = q.quantize(d);
                q.score(Similarity::L2, &qc, qo, &dc, doff)
            })
            .collect();
        assert_eq!(scores[1], 1.0, "{q:?} {scores:?}");
        assert!((scores[2] - 0.002684576).abs() < 1e-6, "{q:?} {scores:?}");
        assert!((scores[0] - 1.1910488e-05).abs() < 1e-8, "{q:?} {scores:?}");
    }

    #[test]
    fn score_estimates_are_close_to_exact() {
        let dim = 64;
        let vecs: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                let v = rng_vec(i + 11, dim);
                let n = distance::norm(&v);
                v.iter().map(|x| x / n).collect()
            })
            .collect();
        for sim in [Similarity::L2, Similarity::DotProduct, Similarity::MaxInnerProduct, Similarity::Cosine] {
            let q4 = UniformQuantizer::fit_auto(vecs.len(), |i| vecs[i].as_slice(), sim, 4);
            let q7 = UniformQuantizer::fit(vecs.len(), |i| vecs[i].as_slice(), default_confidence_interval(dim), 7);
            let mut err4 = 0.0f64;
            let mut err7 = 0.0f64;
            let mut n = 0usize;
            for i in 0..20 {
                let (qc4, qo4) = q4.quantize(&vecs[i]);
                let (qc7, qo7) = q7.quantize(&vecs[i]);
                for j in 20..120 {
                    let exact = sim.score(&vecs[i], &vecs[j]);
                    let (c4, o4) = q4.quantize(&vecs[j]);
                    let (c7, o7) = q7.quantize(&vecs[j]);
                    err4 += (q4.score(sim, &qc4, qo4, &c4, o4) - exact).abs() as f64;
                    err7 += (q7.score(sim, &qc7, qo7, &c7, o7) - exact).abs() as f64;
                    n += 1;
                }
            }
            let (e4, e7) = (err4 / n as f64, err7 / n as f64);
            // Unit vectors: scores live in ~[0, 1]; int4 within a few
            // hundredths, 7-bit an order of magnitude tighter.
            assert!(e4 < 0.03, "{sim:?}: int4 mean abs error {e4}");
            assert!(e7 < 0.005, "{sim:?}: int7 mean abs error {e7}");
            assert!(e7 < e4, "{sim:?}: {e7} vs {e4}");
        }
    }
}
