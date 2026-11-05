//! The `_score` Elasticsearch returns for scalar-quantized `dense_vector`
//! fields searched without rescoring (`int8_hnsw` / `int8_flat`: Lucene's
//! 7-bit quantization; `int4_hnsw` / `int4_flat`: 4 bits).
//!
//! ES (Lucene `Lucene99ScalarQuantizedVectorsFormat`) quantizes each Lucene
//! segment with quantiles fitted on that segment's vectors, and a kNN
//! search over the segment scores candidates with the quantized query and
//! vectors (`ScalarQuantizedVectorSimilarity` / ES `Int7SQVectorScorer`):
//! the scores are estimates, not the exact similarity. To return the same
//! scores, every group of vectors that ES would hold in one segment gets
//! its own quantizer, fitted exactly as Lucene fits it:
//!
//! * a memtable / an appendable vector chunk: the vectors one refresh (or
//!   flush) wrote, deleted ones included (Lucene flushes them too);
//! * a sealed vector segment: the vectors it holds (a merge fits anew).
//!
//! These estimate codes are built in memory per group and cached; they
//! are never written to disk.
//!
//! Searches find candidates as usual (graph or brute force, `f32` or the
//! segment's own quantized copy), then score the best `max(k,
//! num_candidates)` of each group with the ES estimate and keep the top
//! `k` by it.
//!
//! `cosine` vectors are normalised the way ES normalises them at index
//! and query time (`DenseVectorFieldMapper`: divided by the `float`
//! magnitude unless within `1e-3` of unit length) and scored as Lucene's
//! `DOT_PRODUCT`.

mod checkpoint;

use std::borrow::Cow;
use std::sync::Arc;

use super::int4::{self, UniformQuantizer, DYNAMIC_CONFIDENCE_INTERVAL};
use super::osq::{OsqCorrections, OsqQuantizer, OsqSim};
use super::quantize::Quantization;
use super::store::VectorFieldOptions;
use super::Similarity;

/// How a field's searches are scored when ES would return quantized
/// estimates: Lucene bits per component and the confidence interval of
/// the quantile fit (`None`: Lucene's default for the dimension; `0`: the
/// dynamic interval), or ES 9's optimized scalar quantization (`osq`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EsQuant {
    pub bits: u8,
    pub confidence_interval: Option<f32>,
    /// ES 9 (`ES94ScalarQuantizedVectorsFormat`): per-vector OSQ intervals
    /// around the segment centroid instead of Lucene 9's segment quantiles.
    pub osq: bool,
}

impl EsQuant {
    /// Stored mutable codes remain enabled independently of response rescoring.
    pub(crate) fn for_mutable(options: &VectorFieldOptions) -> crate::error::Result<Option<Self>> {
        let bits = match options.quantization {
            Quantization::None => return Ok(None),
            Quantization::Auto | Quantization::Int8 => match options.traversal_bits {
                None | Some(7) => 7,
                Some(4) => 4,
                Some(_) => return Err(crate::error::RusticError::InvalidQuery("unsupported int8 traversal code width".into())),
            },
            Quantization::Int4 => 4,
            Quantization::Binary => 1,
        };
        if options.confidence_interval.is_some_and(|ci| !ci.is_finite() || (ci != 0.0 && !(0.9..=1.0).contains(&ci))) {
            return Err(crate::error::RusticError::InvalidQuery("invalid mutable confidence interval".into()));
        }
        Ok(Some(Self { bits, confidence_interval: options.confidence_interval, osq: options.osq || bits == 1 }))
    }

    /// The ES scoring of a field with `options` searched with rescore
    /// `oversample` (`0` = no rescoring): `None` when ES returns exact
    /// similarities (unquantized, BBQ-rescored or rescored fields).
    pub fn of(quantization: Quantization, options: &VectorFieldOptions, oversample: f32) -> Option<EsQuant> {
        if oversample > 0.0 {
            return None;
        }
        // DiskBBQ rows not in the field's index yet (a memtable, a fresh
        // chunk) are what ES holds as a flat IVF segment: `bits`-bit OSQ
        // codes around the segment's centroid.
        if let Some(ivf) = options.ivf.filter(|_| quantization != Quantization::None) {
            return Some(EsQuant { bits: ivf.doc_bits(), confidence_interval: None, osq: true });
        }
        let bits = match quantization {
            Quantization::Int8 => 7,
            Quantization::Int4 => 4,
            // BBQ (`ES93BinaryQuantizedVectorsFormat`): 1-bit OSQ codes.
            Quantization::Binary if options.osq => 1,
            _ => return None,
        };
        Some(EsQuant { bits, confidence_interval: options.confidence_interval, osq: options.osq })
    }

    /// Cache key of a quantizer fitted for this scoring.
    pub fn key(&self) -> (u8, u32) {
        (self.bits | if self.osq { 0x80 } else { 0 }, self.confidence_interval.map_or(u32::MAX, f32::to_bits))
    }

    /// Fit the quantizer of one segment's `n` vectors (read through `row`,
    /// raw as indexed) like Lucene's flush / merge does.
    pub fn fit<'a>(&self, sim: Similarity, n: usize, row: impl Fn(usize) -> &'a [f32]) -> EsQuantizer {
        let unit: Option<Vec<Vec<f32>>> = (sim == Similarity::Cosine).then(|| (0..n).map(|i| es_unit(row(i)).into_owned()).collect());
        let get = |i: usize| match &unit {
            Some(v) => v[i].as_slice(),
            None => row(i),
        };
        // Lucene sees ES's normalised cosine vectors as `DOT_PRODUCT`.
        let fit_sim = if sim == Similarity::Cosine { Similarity::DotProduct } else { sim };
        let dim = if n > 0 { get(0).len() } else { 0 };
        if self.osq {
            // Queries: 4 bits against 1-bit (BBQ) or 2-bit codes, else the
            // codes' own width (Lucene `ScalarEncoding.getQueryBits`).
            let query_bits = if self.bits <= 4 { 4 } else { self.bits };
            return EsQuantizer::Osq(Arc::new(OsqQuantizer::fit(OsqSim::of(sim), n, dim, self.bits, query_bits, get)));
        }
        EsQuantizer::Lucene99(match self.confidence_interval {
            Some(ci) if ci == DYNAMIC_CONFIDENCE_INTERVAL => UniformQuantizer::fit_auto(n, get, fit_sim, self.bits),
            Some(ci) => UniformQuantizer::fit(n, get, ci, self.bits),
            None => UniformQuantizer::fit(n, get, int4::default_confidence_interval(dim), self.bits),
        })
    }
}

/// A fitted segment quantizer: Lucene 9 scalar quantiles, or ES 9 OSQ.
#[derive(Clone, Debug)]
pub enum EsQuantizer {
    Lucene99(UniformQuantizer),
    Osq(Arc<OsqQuantizer>),
}

/// ES's normalisation of a `cosine` vector: divided by its `float`
/// magnitude (accumulated in `float`, as ES does) unless the squared
/// magnitude is within `1e-3` of 1.
pub fn es_unit(v: &[f32]) -> Cow<'_, [f32]> {
    let sq: f32 = v.iter().map(|x| x * x).sum();
    if (sq - 1.0).abs() <= 1e-3 || sq == 0.0 {
        return Cow::Borrowed(v);
    }
    let len = (sq as f64).sqrt() as f32;
    Cow::Owned(v.iter().map(|x| x / len).collect())
}

/// A segment's rows quantized with its ES quantizer, as Lucene stores them
/// when it writes the segment, so queries only compare codes.
pub struct EsCodes {
    pub q: EsQuantizer,
    dims: usize,
    codes: Vec<u8>,
    /// Lucene 9: the per-row corrective offset.
    offsets: Vec<f32>,
    /// OSQ: the per-row corrective terms.
    corrections: Vec<OsqCorrections>,
    // Keep the reservation until packed payloads and calibration references drop.
    admission: Option<crate::admission::Reservation>,
}

impl EsCodes {
    pub(crate) fn retain_admission(&mut self, memory: crate::admission::Reservation) { self.admission=Some(memory); }
    /// Quantize rows `0..n` (`row(i)`, as indexed) with `q`.
    pub fn build<'a>(q: EsQuantizer, sim: Similarity, n: usize, dims: usize, row: impl Fn(usize) -> &'a [f32]) -> EsCodes {
        let width = Self::row_bytes_for(&q, dims);
        let mut codes = Vec::with_capacity(n * width);
        let mut offsets = Vec::new();
        let mut corrections = Vec::new();
        let mut buf = Vec::with_capacity(if n==0 {0} else {dims});
        for i in 0..n {
            let v = row(i);
            let vv = if sim == Similarity::Cosine { es_unit(v) } else { Cow::Borrowed(v) };
            match &q {
                EsQuantizer::Lucene99(u) => offsets.push(u.quantize_into(&vv, &mut buf)),
                EsQuantizer::Osq(o) => corrections.push(o.quantize_doc(&vv, &mut buf)),
            }
            Self::append_codes(&q, &buf, &mut codes);
        }
        EsCodes { q, dims, codes, offsets, corrections, admission: None }
    }

    pub(crate) fn is_calibration_for(&self, spec: EsQuant, dims: usize, sim: Similarity) -> bool {
        self.dims == dims && self.codes.is_empty() && self.offsets.is_empty() && self.corrections.is_empty()
            && Self::bits(&self.q) == spec.bits && match &self.q {
                EsQuantizer::Lucene99(_) => !spec.osq,
                EsQuantizer::Osq(q) => spec.osq && q.sim == OsqSim::of(sim),
            }
    }

    /// A self-contained packed row, suitable for the mutable owner's
    /// generation-managed word store. Calibration is retained separately.
    /// Corrections occupy four words for both formats, followed by packed codes.
    pub(crate) fn row_words(&self, row: usize) -> Vec<u32> {
        let mut words = match &self.q {
            EsQuantizer::Lucene99(_) => vec![self.offsets[row].to_bits(), 0, 0, 0],
            EsQuantizer::Osq(_) => {
                let c = self.corrections[row];
                vec![c.lower.to_bits(), c.upper.to_bits(), c.additional.to_bits(), c.component_sum as u32]
            }
        };
        for bytes in self.codes(row).chunks(4) {
            let mut word = [0; 4]; word[..bytes.len()].copy_from_slice(bytes);
            words.push(u32::from_le_bytes(word));
        }
        words
    }

    /// Heap storage of the cached row codes and corrections (excluding the
    /// shared fitted quantizer). Used by memtable flush accounting.
    pub fn heap_bytes(&self) -> usize {
        self.codes.capacity()
            + self.offsets.capacity() * std::mem::size_of::<f32>()
            + self.corrections.capacity() * std::mem::size_of::<OsqCorrections>()
    }

    fn bits(q: &EsQuantizer) -> u8 {
        match q { EsQuantizer::Lucene99(q) => q.bits, EsQuantizer::Osq(q) => q.doc_bits }
    }

    fn row_bytes_for(q: &EsQuantizer, dims: usize) -> usize {
        match Self::bits(q) { 1 => dims.div_ceil(8), 4 => dims.div_ceil(2), _ => dims }
    }

    // Seven-bit codes retain byte alignment for the existing SIMD dot kernel.
    // Int4 and BBQ stay packed in RAM as well as in their durable checkpoint.
    fn append_codes(q: &EsQuantizer, row: &[u8], out: &mut Vec<u8>) {
        match Self::bits(q) {
            1 => for chunk in row.chunks(8) {
                out.push(chunk.iter().enumerate().fold(0, |byte, (bit, code)| byte | ((code & 1) << bit)));
            },
            4 => {
                let half = row.len().div_ceil(2);
                for i in 0..half { out.push((row[i] << 4) | row.get(i + half).copied().unwrap_or(0)); }
            },
            _ => out.extend_from_slice(row),
        }
    }

    fn codes(&self, i: usize) -> &[u8] {
        let width = Self::row_bytes_for(&self.q, self.dims);
        &self.codes[i * width..(i + 1) * width]
    }

    fn code_at(&self, row: usize, dimension: usize) -> u8 {
        let codes = self.codes(row);
        match Self::bits(&self.q) {
            1 => (codes[dimension / 8] >> (dimension % 8)) & 1,
            4 => {
                let half = self.dims.div_ceil(2);
                if dimension < half { codes[dimension] >> 4 } else { codes[dimension - half] & 15 }
            },
            _ => codes[dimension],
        }
    }

    fn dot(&self, row: usize, query: &[u8]) -> u32 {
        match Self::bits(&self.q) {
            4 => int4::dot_packed_u4(self.codes(row), query),
            1 => {
                let bits = query.iter().copied().max().map_or(1, |m| (8 - m.leading_zeros()).max(1) as usize);
                dot_bits1(self.codes(row), &bit_planes(query, bits))
            }
            _ => int4::dot_u8(self.codes(row), query),
        }
    }

}


/// Query codes (each `< 2^planes`) as bit planes in the packed 1-bit doc
/// layout (dimension `d` is bit `d % 64` of word `d / 64`): plane `j` holds
/// bit `j` of every code. Lucene's `ipByteBinByte` layout.
pub(crate) fn bit_planes(codes: &[u8], planes: usize) -> Vec<u64> {
    let words = codes.len().div_ceil(64);
    let mut out = vec![0u64; planes * words];
    for (w, chunk) in codes.chunks(64).enumerate() {
        for j in 0..planes {
            out[j * words + w] = chunk.iter().enumerate().fold(0u64, |acc, (bit, &q)| acc | (u64::from((q >> j) & 1) << bit));
        }
    }
    out
}

/// `sum_d query[d] * bit_d(packed)` for packed 1-bit doc codes (LSB-first
/// bytes) against [`bit_planes`] of the query: one AND + popcount per 64
/// dimensions and plane instead of a multiply per dimension.
#[inline]
pub(crate) fn dot_bits1(packed: &[u8], planes: &[u64]) -> u32 {
    let words = packed.len().div_ceil(8);
    if words == 0 { return 0; }
    let n = planes.len() / words;
    #[cfg(target_arch = "aarch64")]
    if n <= 4 && words >= 2 {
        // SAFETY: NEON is part of the aarch64 baseline; the reads stay within
        // `packed` (whole 16-byte pairs of its full words) and `planes`
        // (`n * words` words).
        return unsafe { dot_bits1_neon(packed, planes, words, n) };
    }
    dot_bits1_scalar(packed, planes, words, n)
}

/// [`dot_bits1`] with NEON: 16 bytes of the doc against each plane per step
/// (AND, byte popcount, pairwise accumulate into 16-bit lanes), the odd tail
/// word and a partial last word scalar. Up to 4 planes (4-bit queries).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_bits1_neon(packed: &[u8], planes: &[u64], words: usize, n: usize) -> u32 {
    use std::arch::aarch64::*;
    let full = packed.len() / 8;
    let pairs = full / 2;
    let mut acc = [vdupq_n_u16(0); 4];
    let doc = packed.as_ptr();
    let plane = planes.as_ptr().cast::<u8>();
    for p in 0..pairs {
        let d = vld1q_u8(doc.add(p * 16));
        for (j, a) in acc.iter_mut().enumerate().take(n) {
            let q = vld1q_u8(plane.add((j * words + p * 2) * 8));
            *a = vpadalq_u8(*a, vcntq_u8(vandq_u8(d, q)));
        }
    }
    let mut total = 0u32;
    for (j, a) in acc.iter().enumerate().take(n) { total += vaddlvq_u16(*a) << j; }
    for w in pairs * 2..words {
        let doc = if w < full {
            u64::from_le_bytes(packed[w * 8..w * 8 + 8].try_into().expect("8 bytes"))
        } else {
            let mut b = [0u8; 8]; let tail = &packed[w * 8..]; b[..tail.len()].copy_from_slice(tail); u64::from_le_bytes(b)
        };
        for j in 0..n { total += ((planes[j * words + w] & doc).count_ones()) << j; }
    }
    total
}

/// [`dot_bits1`] of four doc rows of the same length against one query:
/// each 16-byte step loads the query planes once for the four rows.
pub(crate) fn dot_bits1_four(rows: [&[u8]; 4], planes: &[u64]) -> [u32; 4] {
    let len = rows[0].len();
    let words = len.div_ceil(8);
    #[cfg(target_arch = "aarch64")]
    if words >= 2 && rows.iter().all(|r| r.len() == len) {
        // SAFETY: as `dot_bits1_neon`, for each row. The plane count is a
        // constant so the 4 x N accumulators stay in registers.
        match planes.len() / words {
            4 => return unsafe { dot_bits1_four_neon::<4>(rows, planes, words) },
            3 => return unsafe { dot_bits1_four_neon::<3>(rows, planes, words) },
            2 => return unsafe { dot_bits1_four_neon::<2>(rows, planes, words) },
            1 => return unsafe { dot_bits1_four_neon::<1>(rows, planes, words) },
            _ => {}
        }
    }
    rows.map(|row| dot_bits1(row, planes))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_bits1_four_neon<const N: usize>(rows: [&[u8]; 4], planes: &[u64], words: usize) -> [u32; 4] {
    use std::arch::aarch64::*;
    let n = N;
    let full = rows[0].len() / 8;
    let pairs = full / 2;
    let mut acc = [[vdupq_n_u16(0); N]; 4];
    let plane = planes.as_ptr().cast::<u8>();
    let (r0, r1, r2, r3) = (rows[0].as_ptr(), rows[1].as_ptr(), rows[2].as_ptr(), rows[3].as_ptr());
    for p in 0..pairs {
        let d = [vld1q_u8(r0.add(p * 16)), vld1q_u8(r1.add(p * 16)), vld1q_u8(r2.add(p * 16)), vld1q_u8(r3.add(p * 16))];
        for j in 0..N {
            let q = vld1q_u8(plane.add((j * words + p * 2) * 8));
            acc[0][j] = vpadalq_u8(acc[0][j], vcntq_u8(vandq_u8(d[0], q)));
            acc[1][j] = vpadalq_u8(acc[1][j], vcntq_u8(vandq_u8(d[1], q)));
            acc[2][j] = vpadalq_u8(acc[2][j], vcntq_u8(vandq_u8(d[2], q)));
            acc[3][j] = vpadalq_u8(acc[3][j], vcntq_u8(vandq_u8(d[3], q)));
        }
    }
    let mut out = [0u32; 4];
    for r in 0..4 {
        for j in 0..N { out[r] += vaddlvq_u16(acc[r][j]) << j; }
        let packed = rows[r];
        for w in pairs * 2..words {
            let doc = if w < full {
                u64::from_le_bytes(packed[w * 8..w * 8 + 8].try_into().expect("8 bytes"))
            } else {
                let mut b = [0u8; 8]; let tail = &packed[w * 8..]; b[..tail.len()].copy_from_slice(tail); u64::from_le_bytes(b)
            };
            for j in 0..n { out[r] += ((planes[j * words + w] & doc).count_ones()) << j; }
        }
    }
    out
}

fn dot_bits1_scalar(packed: &[u8], planes: &[u64], words: usize, n: usize) -> u32 {
    let mut total = 0u32;
    let full = packed.len() / 8;
    for w in 0..words {
        let doc = if w < full {
            u64::from_le_bytes(packed[w * 8..w * 8 + 8].try_into().expect("8 bytes"))
        } else {
            let mut b = [0u8; 8]; let tail = &packed[w * 8..]; b[..tail.len()].copy_from_slice(tail); u64::from_le_bytes(b)
        };
        for j in 0..n { total += ((planes[j * words + w] & doc).count_ones()) << j; }
    }
    total
}

impl EsQuantizer {
    /// The vector a stored code row approximates (`words`: four correction
    /// words, then the packed codes, as [`EsQuantScorer::score_words`]
    /// reads them), in this quantizer's space (unit length for cosine).
    /// Graph maintenance compares rows with these instead of reading their
    /// f32, which for shed rows lives on disk.
    pub(crate) fn decode_words(&self, words: &[f32], dims: usize, out: &mut Vec<f32>) -> bool {
        let bits = EsCodes::bits(self);
        let bytes = EsCodes::row_bytes_for(self, dims);
        if words.len() != 4 + bytes.div_ceil(4) { return false; }
        let code_byte = |i: usize| ((words[4 + i / 4].to_bits() >> ((i % 4) * 8)) & 255) as u8;
        let code = |d: usize| match bits {
            1 => (code_byte(d / 8) >> (d % 8)) & 1,
            4 => { let half = dims.div_ceil(2); if d < half { code_byte(d) >> 4 } else { code_byte(d - half) & 15 } },
            _ => code_byte(d),
        };
        out.clear();
        out.reserve(dims);
        // Byte codes (7 bits): a straight loop over the packed bytes, which
        // the compiler vectorizes; packed 1/4-bit codes unpack per cell.
        #[cfg(target_endian = "little")]
        let bytes_codes: Option<&[u8]> = (bits == 7).then(|| {
            // SAFETY: initialized word payloads; on little-endian targets their
            // bytes are exactly the packed codes (as `score_words` reads them).
            unsafe { std::slice::from_raw_parts(words[4..].as_ptr().cast::<u8>(), dims) }
        });
        #[cfg(not(target_endian = "little"))]
        let bytes_codes: Option<&[u8]> = None;
        match self {
            EsQuantizer::Lucene99(u) => {
                let (a, min) = (u.alpha(), u.min);
                match bytes_codes {
                    Some(codes) => out.extend(codes.iter().map(|&c| a * f32::from(c) + min)),
                    None => out.extend((0..dims).map(|d| a * f32::from(code(d)) + min)),
                }
            }
            EsQuantizer::Osq(q) => {
                // Doc codes span [lower, upper] of the centered vector.
                let (lower, upper) = (words[0], words[1]);
                let levels = ((1u32 << bits) - 1).max(1) as f32;
                let step = (upper - lower) / levels;
                #[cfg(target_endian = "little")]
                if bits == 4 && q.centroid.len() == dims {
                    // Packed 4-bit codes: byte i holds dimension i (high
                    // nibble) and dimension i + half (low nibble).
                    let half = dims.div_ceil(2);
                    // SAFETY: as `bytes_codes`: the packed 4-bit codes.
                    let packed = unsafe { std::slice::from_raw_parts(words[4..].as_ptr().cast::<u8>(), half) };
                    out.resize(dims, 0.0);
                    let (head, tail) = out.split_at_mut(half);
                    let (c_head, c_tail) = q.centroid.split_at(half);
                    for ((o, &b), &m) in head.iter_mut().zip(packed).zip(c_head) { *o = f32::from(b >> 4).mul_add(step, lower + m); }
                    for ((o, &b), &m) in tail.iter_mut().zip(packed).zip(c_tail) { *o = f32::from(b & 15).mul_add(step, lower + m); }
                    return out.iter().all(|x| x.is_finite());
                }
                #[cfg(target_endian = "little")]
                if bits == 1 && q.centroid.len() == dims {
                    // SAFETY: as `bytes_codes`: the packed 1-bit codes.
                    let packed = unsafe { std::slice::from_raw_parts(words[4..].as_ptr().cast::<u8>(), dims.div_ceil(8)) };
                    let high = lower + step;
                    out.extend(q.centroid.iter().enumerate().map(|(d, &m)| m + if (packed[d >> 3] >> (d & 7)) & 1 == 1 { high } else { lower }));
                    return out.iter().all(|x| x.is_finite());
                }
                match bytes_codes {
                    Some(codes) if q.centroid.len() == dims =>
                        out.extend(codes.iter().zip(&q.centroid).map(|(&c, &m)| f32::from(c).mul_add(step, lower + m))),
                    _ => out.extend((0..dims).map(|d| lower + f32::from(code(d)) * step + q.centroid.get(d).copied().unwrap_or(0.0))),
                }
            }
        }
        out.iter().all(|x| x.is_finite())
    }
}

/// Scores vectors of one segment against a query with the segment's
/// quantizer, as ES does without rescoring.
#[derive(Clone)]
pub struct EsQuantScorer {
    q: EsQuantizer,
    sim: Similarity,
    codes: Vec<u8>,
    offset: f32,
    corrections: OsqCorrections,
    buf: Vec<u8>,
    /// 1-bit doc codes: the query codes as bit planes ([`dot_bits1`]).
    planes: Vec<u64>,
}

impl EsQuantScorer {
    pub fn new(q: EsQuantizer, sim: Similarity, query: &[f32]) -> Self {
        let qv = if sim == Similarity::Cosine { es_unit(query) } else { Cow::Borrowed(query) };
        let mut codes = Vec::with_capacity(query.len());
        let (offset, corrections) = match &q {
            EsQuantizer::Lucene99(u) => (u.quantize_into(&qv, &mut codes), OsqCorrections::default()),
            EsQuantizer::Osq(o) => (0.0, o.quantize_query(&qv, &mut codes)),
        };
        let planes = if EsCodes::bits(&q) == 1 {
            let bits = match &q { EsQuantizer::Osq(o) => o.query_bits as usize, EsQuantizer::Lucene99(_) => 8 };
            bit_planes(&codes, bits.clamp(1, 8))
        } else { Vec::new() };
        EsQuantScorer { q, sim, codes, offset, corrections, buf: Vec::with_capacity(query.len()), planes }
    }

    /// The ES score of raw (as indexed) vector `v`.
    pub fn score(&mut self, v: &[f32]) -> f32 {
        let vv = if self.sim == Similarity::Cosine { es_unit(v) } else { Cow::Borrowed(v) };
        match &self.q {
            EsQuantizer::Lucene99(u) => {
                let off = u.quantize_into(&vv, &mut self.buf);
                u.score(self.sim, &self.codes, self.offset, &self.buf, off)
            }
            EsQuantizer::Osq(o) => {
                let c = o.quantize_doc(&vv, &mut self.buf);
                o.score(&self.codes, &self.corrections, &self.buf, &c)
            }
        }
    }

    /// Score a row borrowed directly from mutable source storage. No row
    /// allocation, quantizer refit or f32 fallback occurs on this path.
    pub(crate) fn score_words(&self, words: &[f32], dims: usize) -> Option<f32> {
        let bits = EsCodes::bits(&self.q);
        let bytes = EsCodes::row_bytes_for(&self.q, dims);
        if words.len() != 4 + bytes.div_ceil(4) || self.codes.len() != dims { return None; }
        let code_byte = |i: usize| ((words[4 + i / 4].to_bits() >> ((i % 4) * 8)) & 255) as u8;
        let code = |d: usize| match bits {
            1 => (code_byte(d / 8) >> (d % 8)) & 1,
            4 => { let half = dims.div_ceil(2); if d < half { code_byte(d) >> 4 } else { code_byte(d - half) & 15 } },
            _ => code_byte(d),
        };
        #[cfg(target_endian = "little")]
        // Word payloads are initialized bit patterns. On little-endian targets
        // their byte representation is exactly the packed-code format.
        let packed = unsafe { std::slice::from_raw_parts(words[4..].as_ptr().cast::<u8>(), bytes) };
        let dot = || {
            #[cfg(target_endian = "little")]
            if bits == 4 { return int4::dot_packed_u4(packed, &self.codes) as f32; }
            #[cfg(target_endian = "little")]
            if bits == 7 { return int4::dot_u8(packed, &self.codes) as f32; }
            #[cfg(target_endian = "little")]
            if bits == 1 { return dot_bits1(packed, &self.planes) as f32; }
            self.codes.iter().enumerate().map(|(d, &q)| u32::from(q) * u32::from(code(d))).sum::<u32>() as f32
        };
        Some(match &self.q {
            EsQuantizer::Lucene99(u) => match self.sim {
                Similarity::L2 => {
                    let squared: u32 = self.codes.iter().enumerate().map(|(d, &q)| {
                        let delta = i32::from(q) - i32::from(code(d)); (delta * delta) as u32
                    }).sum();
                    1.0 / (1.0 + squared as f32 * u.const_multiplier())
                }
                Similarity::Cosine | Similarity::DotProduct => {
                    let adjusted = dot() * u.const_multiplier() + self.offset + words[0];
                    ((1.0 + adjusted) / 2.0).max(0.0)
                },
                Similarity::MaxInnerProduct => int4::scale_mip(dot() * u.const_multiplier() + self.offset + words[0]),
                packed => unreachable!("packed similarity {packed:?} is never quantized"),
            },
            EsQuantizer::Osq(q) => q.score_raw(dot(), &self.corrections, &OsqCorrections {
                lower: words[0], upper: words[1], additional: words[2], component_sum: words[3].to_bits() as i32,
            }),
        })
    }

    /// Words of a code row ([`score_words`](Self::score_words)'s input) of
    /// this scorer's quantizer at `dims`.
    #[inline]
    pub(crate) fn words_len(&self, dims: usize) -> usize {
        4 + EsCodes::row_bytes_for(&self.q, dims).div_ceil(4)
    }

    /// [`score_words`](Self::score_words) of four rows, the 7-bit OSQ dot
    /// products computed together ([`int4::dot_u8_four`]); each score equals
    /// the row's own `score_words`.
    pub(crate) fn score_words_four(&self, rows: [&[f32]; 4], dims: usize) -> [Option<f32>; 4] {
        let EsQuantizer::Osq(q) = &self.q else { return rows.map(|row| self.score_words(row, dims)) };
        let bytes = EsCodes::row_bytes_for(&self.q, dims);
        let bits = EsCodes::bits(&self.q);
        if !matches!(bits, 1 | 7) || (bits == 1 && self.planes.is_empty()) || !cfg!(target_endian = "little") || self.codes.len() != dims
            || rows.iter().any(|row| row.len() != 4 + bytes.div_ceil(4)) {
            return rows.map(|row| self.score_words(row, dims));
        }
        // SAFETY: as `score_words`: each row's words after the corrections
        // are its packed codes, `bytes` long, on a little-endian target.
        let packed = rows.map(|row| unsafe { std::slice::from_raw_parts(row[4..].as_ptr().cast::<u8>(), bytes) });
        let dots = if bits == 1 { dot_bits1_four(packed, &self.planes) } else { int4::dot_u8_four(&self.codes, packed) };
        let mut out = [None; 4];
        for ((o, row), dot) in out.iter_mut().zip(rows).zip(dots) {
            *o = Some(q.score_raw(dot as f32, &self.corrections, &OsqCorrections {
                lower: row[0], upper: row[1], additional: row[2], component_sum: row[3].to_bits() as i32,
            }));
        }
        out
    }

    /// The ES score of row `i` of a segment's precomputed codes.
    pub fn score_row(&self, coded: &EsCodes, i: usize) -> f32 {
        match &self.q {
            EsQuantizer::Lucene99(u) if !matches!(EsCodes::bits(&coded.q), 1 | 4) =>
                u.score(self.sim, &self.codes, self.offset, coded.codes(i), coded.offsets[i]),
            EsQuantizer::Lucene99(u) => match self.sim {
                Similarity::L2 => {
                    let squared: u32 = self.codes.iter().enumerate().map(|(j, &q)| {
                        let delta = i32::from(q) - i32::from(coded.code_at(i, j));
                        (delta * delta) as u32
                    }).sum();
                    1.0 / (1.0 + squared as f32 * u.const_multiplier())
                }
                Similarity::DotProduct | Similarity::Cosine => {
                    let adj = coded.dot(i, &self.codes) as f32 * u.const_multiplier() + self.offset + coded.offsets[i];
                    ((1.0 + adj) / 2.0).max(0.0)
                }
                Similarity::MaxInnerProduct => {
                    let adj = coded.dot(i, &self.codes) as f32 * u.const_multiplier() + self.offset + coded.offsets[i];
                    int4::scale_mip(adj)
                }
                packed => unreachable!("packed similarity {packed:?} is never quantized"),
            },
            EsQuantizer::Osq(o) => {
                let dot = if EsCodes::bits(&coded.q) == 1 && !self.planes.is_empty() { dot_bits1(coded.codes(i), &self.planes) }
                    else { coded.dot(i, &self.codes) };
                o.score_raw(dot as f32, &self.corrections, &coded.corrections[i])
            }
        }
    }

    /// [`EsQuantScorer::top_k`] over a segment's precomputed codes
    /// (`cands` are its ordinals).
    pub fn top_k_coded(&self, cands: Vec<(u32, f32)>, k: usize, coded: &EsCodes) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = cands.into_iter().map(|(o, _)| (o, self.score_row(coded, o as usize))).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// Rescore `cands` (`(id, _)` with `vector(id)`) and keep the best `k`
    /// by the ES estimate (ties: lower id first).
    pub fn top_k<'a, I: Copy + Ord>(&mut self, cands: Vec<(I, f32)>, k: usize, vector: impl Fn(I) -> &'a [f32]) -> Vec<(I, f32)> {
        let mut out: Vec<(I, f32)> = cands.into_iter().map(|(i, _)| (i, self.score(vector(i)))).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_plane_dot_matches_per_dimension_dot() {
        for dims in [1usize, 7, 8, 63, 64, 65, 130, 384, 1536] {
            let codes: Vec<u8> = (0..dims).map(|d| ((d * 7 + 3) % 16) as u8).collect();
            let packed: Vec<u8> = (0..dims.div_ceil(8)).map(|b| (0..8).filter(|bit| b * 8 + bit < dims && (b * 13 + bit * 5) % 3 == 0).fold(0u8, |a, bit| a | (1 << bit))).collect();
            let expected: u32 = (0..dims).map(|d| u32::from(codes[d]) * u32::from((packed[d / 8] >> (d % 8)) & 1)).sum();
            assert_eq!(dot_bits1(&packed, &bit_planes(&codes, 4)), expected, "dims={dims}");
            assert_eq!(dot_bits1_scalar(&packed, &bit_planes(&codes, 4), packed.len().div_ceil(8), 4), expected, "dims={dims}");
            let rows = [&packed[..], &packed[..], &packed[..], &packed[..]];
            assert_eq!(dot_bits1_four(rows, &bit_planes(&codes, 4)), [expected; 4], "dims={dims}");
        }
    }

    #[test]
    fn packed_resident_rows_match_raw_quantized_scores_and_bound_storage() {
        for dims in [1, 7, 8, 13, 32, 127, 384] {
            let rows: Vec<Vec<f32>> = (0..9).map(|r| (0..dims).map(|d|
                ((r * 29 + d * 7) % 101) as f32 / 31.0 - 1.5).collect()).collect();
            for sim in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct, Similarity::MaxInnerProduct] {
                for (bits, osq) in [(7, false), (4, false), (7, true), (4, true), (1, true)] {
                    let spec = EsQuant { bits, osq, confidence_interval: None };
                    let q = spec.fit(sim, rows.len(), |i| &rows[i]);
                    let coded = EsCodes::build(q.clone(), sim, rows.len(), dims, |i| &rows[i]);
                    let width = match bits { 1 => dims.div_ceil(8), 4 => dims.div_ceil(2), _ => dims };
                    assert_eq!(coded.codes.len(), rows.len() * width);
                    assert_eq!(coded.codes.capacity(), rows.len() * width);
                    let restored = EsCodes::from_checkpoint(&coded.checkpoint_bytes().unwrap()).unwrap();
                    assert_eq!(restored.codes, coded.codes);
                    for query in &rows {
                        let mut scorer = EsQuantScorer::new(q.clone(), sim, query);
                        for (i, row) in rows.iter().enumerate() {
                            let expected = scorer.score(row).to_bits();
                            assert_eq!(scorer.score_row(&coded, i).to_bits(), expected,
                                "dims={dims} bits={bits} osq={osq} sim={sim:?}");
                            assert_eq!(scorer.score_row(&restored, i).to_bits(), expected);
                            let words: Vec<f32> = coded.row_words(i).into_iter().map(f32::from_bits).collect();
                            assert_eq!(scorer.score_words(&words, dims).unwrap().to_bits(), expected);
                            assert!(scorer.score_words(&words[..words.len()-1], dims).is_none());
                        }
                    }
                }
            }
        }
    }

    fn emb(hash: i32, dims: usize) -> Vec<f32> {
        let b = hash.to_string().into_bytes();
        (0..dims).map(|i| b[i % b.len()] as f32).collect()
    }

    /// Values computed with Lucene's `ScalarQuantizer` (the scores ES
    /// 8.19 returns in `inference/47_semantic_text_knn.yml`).
    #[test]
    fn matches_lucene_int7_scores() {
        let spec = EsQuant { bits: 7, confidence_interval: None, osq: false };
        // One vector per segment: quantiles over its own components.
        let doc: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let q = spec.fit(Similarity::Cosine, 1, |_| doc.as_slice());
        // `"test".hashCode()` = 3556498.
        let query = emb(3556498, 10);
        let mut s = EsQuantScorer::new(q, Similarity::Cosine, &query);
        assert_eq!(s.score(&doc), 0.9439374);
        // Two vectors in one segment ("inference test", "another inference test").
        let rows = [emb("inference test".chars().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)), 10), emb("another inference test".chars().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)), 10)];
        let q = spec.fit(Similarity::Cosine, 2, |i| rows[i].as_slice());
        let mut s = EsQuantScorer::new(q, Similarity::Cosine, &query);
        assert_eq!(s.score(&rows[0]), 0.99875855);
        assert_eq!(s.score(&rows[1]), 0.9990483);
    }
}
