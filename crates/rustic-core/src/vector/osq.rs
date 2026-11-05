//! Optimized scalar quantization (OSQ), as Elasticsearch 9 scores
//! `int8_*` / `int4_*` fields (`ES94ScalarQuantizedVectorsFormat`, Lucene
//! 10.4 `Lucene104ScalarQuantizedVectorsFormat`).
//!
//! Each Lucene segment has a centroid (the mean of its vectors). Every
//! vector is centered on it and quantized to `bits` with its own interval,
//! chosen by a few rounds of coordinate descent on the quantization loss
//! (`OptimizedScalarQuantizer`), and keeps four corrective terms: the
//! interval, an additional correction (dot product with the centroid, or
//! the squared norm of the residual for euclidean) and the sum of its
//! codes. The query is quantized the same way, and the score estimate is
//! the corrected integer dot product of the codes
//! (`Int7uOSQVectorScorer.applyCorrections`).
//!
//! The arithmetic mirrors ES's scalar implementation
//! (`DefaultESVectorUtilSupport`), including its fused multiply-adds.

use super::Similarity;

/// ES `OptimizedScalarQuantizer.MINIMUM_MSE_GRID`: the initial interval per
/// bit count, in standard deviations around the mean.
const MINIMUM_MSE_GRID: [[f32; 2]; 8] = [
    [-0.798, 0.798],
    [-1.493, 1.493],
    [-2.051, 2.051],
    [-2.514, 2.514],
    [-2.916, 2.916],
    [-3.278, 3.278],
    [-3.611, 3.611],
    [-3.922, 3.922],
];
const LAMBDA: f32 = 0.1;
const ITERS: usize = 5;

/// The corrective terms of one quantized vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OsqCorrections {
    pub lower: f32,
    pub upper: f32,
    pub additional: f32,
    pub component_sum: i32,
}

/// How a vector field is scored: euclidean, or (unit / raw) inner product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OsqSim {
    Euclidean,
    Dot,
    MaxInnerProduct,
}

impl OsqSim {
    /// Lucene's similarity function for a field of `sim` (cosine vectors are
    /// stored unit length and scored as dot products).
    pub fn of(sim: Similarity) -> OsqSim {
        match sim {
            Similarity::L2 => OsqSim::Euclidean,
            Similarity::Cosine | Similarity::DotProduct => OsqSim::Dot,
            Similarity::MaxInnerProduct => OsqSim::MaxInnerProduct,
            packed => unreachable!("packed similarity {packed:?} is never quantized"),
        }
    }
}

#[inline]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Java `Math.round(float)`: floor(x + 0.5).
#[inline]
fn jround(x: f32) -> i32 {
    (x + 0.5).floor() as i32
}

/// A segment's OSQ quantizer: its centroid and the code widths of stored
/// vectors and of queries.
#[derive(Clone, Debug, PartialEq)]
pub struct OsqQuantizer {
    pub sim: OsqSim,
    pub centroid: Vec<f32>,
    /// `dot(centroid, centroid)`.
    pub centroid_dp: f32,
    pub doc_bits: u8,
    pub query_bits: u8,
}

impl OsqQuantizer {
    /// Fit on `n` vectors (read through `row`, already unit length for
    /// cosine fields): the centroid is their mean.
    pub fn fit<'a>(sim: OsqSim, n: usize, dims: usize, doc_bits: u8, query_bits: u8, row: impl Fn(usize) -> &'a [f32]) -> OsqQuantizer {
        let mut c = vec![0f32; dims];
        for i in 0..n {
            for (a, x) in c.iter_mut().zip(row(i)) {
                *a += *x;
            }
        }
        if n > 0 {
            for a in c.iter_mut() {
                *a /= n as f32;
            }
        }
        Self::from_centroid(sim, c, doc_bits, query_bits)
    }

    /// A quantizer around a known `centroid` (e.g. the running mean of every
    /// vector of a field, as one merged ES segment would be fitted).
    pub fn from_centroid(sim: OsqSim, centroid: Vec<f32>, doc_bits: u8, query_bits: u8) -> OsqQuantizer {
        let mut dp = 0f32;
        for x in &centroid {
            dp += x * x;
        }
        OsqQuantizer { sim, centroid, centroid_dp: dp, doc_bits, query_bits }
    }

    /// Quantize a stored vector into `out` (one code per dimension).
    pub fn quantize_doc(&self, v: &[f32], out: &mut Vec<u8>) -> OsqCorrections {
        self.quantize(v, self.doc_bits, out)
    }

    /// Quantize a query vector into `out`.
    pub fn quantize_query(&self, v: &[f32], out: &mut Vec<u8>) -> OsqCorrections {
        self.quantize(v, self.query_bits, out)
    }

    /// A query quantized against this centroid for one posting list scan:
    /// ES's initial MSE-grid interval (`mean + grid * std`, clamped to the
    /// range) without the iterative refinement, in two passes and no
    /// allocation. The `query_bits`-bit query is the finer side of the
    /// estimate, so the refinement barely moves it, while a scan quantizes
    /// the query again for every list it visits. Used only when the
    /// candidates are rescored (the estimates rank, exact scores are
    /// returned); unrescored searches use [`Self::quantize_query`], so
    /// their estimate scores are ES's.
    pub fn quantize_query_fast(&self, v: &[f32], out: &mut Vec<u8>) -> OsqCorrections {
        let bits = self.query_bits;
        let dims = v.len();
        let c = &self.centroid[..dims.min(self.centroid.len())];
        let dot_like = self.sim != OsqSim::Euclidean;
        // Independent lanes (vectorized): sums, squares, range, `v . c`.
        const L: usize = 8;
        let (mut sum, mut sq, mut cd) = ([0f32; L], [0f32; L], [0f32; L]);
        let (mut lo, mut hi) = ([f32::MAX; L], [-f32::MAX; L]);
        let full = c.len() / L * L;
        for (vs, cs) in v[..full].chunks_exact(L).zip(c[..full].chunks_exact(L)) {
            for l in 0..L {
                let x = vs[l] - cs[l];
                sum[l] += x;
                sq[l] = x.mul_add(x, sq[l]);
                cd[l] = vs[l].mul_add(cs[l], cd[l]);
                lo[l] = lo[l].min(x);
                hi[l] = hi[l].max(x);
            }
        }
        let (mut s1, mut s2, mut cdot) = (sum.iter().sum::<f32>(), sq.iter().sum::<f32>(), cd.iter().sum::<f32>());
        let (mut min, mut max) = (lo.iter().copied().fold(f32::MAX, f32::min), hi.iter().copied().fold(-f32::MAX, f32::max));
        for i in full..dims {
            let ci = c.get(i).copied().unwrap_or(0.0);
            let x = v[i] - ci;
            s1 += x;
            s2 = x.mul_add(x, s2);
            cdot = v[i].mul_add(ci, cdot);
            min = min.min(x);
            max = max.max(x);
        }
        let n = dims.max(1) as f32;
        let mean = s1 / n;
        let var = (s2 / n - mean * mean).max(0.0);
        let std = var.sqrt();
        let grid = MINIMUM_MSE_GRID[(bits.clamp(1, 8) - 1) as usize];
        let clamp = |x: f64| x.max(min as f64).min(max as f64);
        let (lo, hi) = (clamp((grid[0] * std + mean) as f64) as f32, clamp((grid[1] * std + mean) as f64) as f32);
        let steps = ((1i32 << bits) - 1) as f32;
        let inv = steps / (hi - lo);
        out.clear();
        out.resize(dims, 0);
        let mut total = 0u32;
        for (i, o) in out.iter_mut().enumerate() {
            let x = (v[i] - c.get(i).copied().unwrap_or(0.0)).max(lo).min(hi);
            // `floor(x + 0.5)` of a value in `[0, steps]` (Java `Math.round`).
            let code = ((x - lo) * inv + 0.5).max(0.0).min(steps) as u32;
            total += code;
            *o = code as u8;
        }
        OsqCorrections { lower: lo, upper: hi, additional: if dot_like { cdot } else { s2 }, component_sum: total as i32 }
    }

    /// ES `OptimizedScalarQuantizer.scalarQuantize`.
    pub fn quantize(&self, v: &[f32], bits: u8, out: &mut Vec<u8>) -> OsqCorrections {
        let dims = v.len();
        let mut centered = vec![0f32; dims];
        // centerAndCalculateOSQStats{Dp,Euclidean}.
        let (mut mean, mut var, mut norm2, mut cdot) = (0f32, 0f32, 0f32, 0f32);
        let (mut min, mut max) = (f32::MAX, -f32::MAX);
        for i in 0..dims {
            let c = self.centroid.get(i).copied().unwrap_or(0.0);
            if self.sim != OsqSim::Euclidean {
                cdot = fma(v[i], c, cdot);
            }
            centered[i] = v[i] - c;
            min = min.min(centered[i]);
            max = max.max(centered[i]);
            norm2 = fma(centered[i], centered[i], norm2);
            let delta = centered[i] - mean;
            mean += delta / (i + 1) as f32;
            let delta2 = centered[i] - mean;
            var = fma(delta, delta2, var);
        }
        if dims > 0 {
            var /= dims as f32;
        }
        let std = (var as f64).sqrt() as f32;
        let grid = MINIMUM_MSE_GRID[(bits.clamp(1, 8) - 1) as usize];
        let clamp = |x: f64| x.max(min as f64).min(max as f64);
        let mut interval = [clamp((grid[0] * std + mean) as f64) as f32, clamp((grid[1] * std + mean) as f64) as f32];
        let points = 1i32 << bits;
        let mut codes = vec![0i32; dims];
        let have = optimize_intervals(&mut interval, &mut codes, &centered, norm2, points);
        let sum = if have {
            codes.iter().sum()
        } else {
            quantize_with_intervals(&centered, &mut codes, interval[0], interval[1], bits)
        };
        out.clear();
        out.extend(codes.iter().map(|c| *c as u8));
        OsqCorrections { lower: interval[0], upper: interval[1], additional: if self.sim == OsqSim::Euclidean { norm2 } else { cdot }, component_sum: sum }
    }

    /// The ES score estimate of a stored vector (`d`, `dc`) for a quantized
    /// query (`q`, `qc`).
    pub fn score(&self, q: &[u8], qc: &OsqCorrections, d: &[u8], dc: &OsqCorrections) -> f32 {
        let len = q.len().min(d.len());
        let raw = super::int4::dot_u8(&q[..len], &d[..len]);
        self.score_raw(raw as f32, qc, dc)
    }

    /// `applyCorrections` of an integer code dot product.
    pub fn score_raw(&self, raw: f32, qc: &OsqCorrections, dc: &OsqCorrections) -> f32 {
        score_of_estimate(self.sim, self.estimate(raw, qc, dc))
    }

    /// The corrected estimate [`score_raw`](Self::score_raw) maps to a
    /// score: the squared distance (euclidean) or the inner product of the
    /// query and the stored vector.
    pub fn estimate(&self, raw: f32, qc: &OsqCorrections, dc: &OsqCorrections) -> f32 {
        let dims = self.centroid.len() as f32;
        let x1 = dc.component_sum as f32;
        let ax = dc.lower;
        let lx = (dc.upper - ax) * (1.0 / ((1i32 << self.doc_bits) - 1) as f32);
        let ay = qc.lower;
        let ly = (qc.upper - ay) * (1.0 / ((1i32 << self.query_bits) - 1) as f32);
        let y1 = qc.component_sum as f32;
        let score = ax * ay * dims + ay * lx * x1 + ax * ly * y1 + lx * ly * raw;
        match self.sim {
            OsqSim::Euclidean => qc.additional + dc.additional - 2.0 * score,
            // Preserve Lucene104's f32 correction association.
            OsqSim::Dot | OsqSim::MaxInnerProduct => score + (qc.additional + dc.additional - self.centroid_dp),
        }
    }
}

/// The ES score of an [`OsqQuantizer::estimate`] (a squared distance or an
/// inner product): increasing in the inner product, decreasing in the
/// distance.
pub fn score_of_estimate(sim: OsqSim, e: f32) -> f32 {
    match sim {
        OsqSim::Euclidean => 1.0 / (1.0 + e.max(0.0)),
        OsqSim::Dot => (1.0 + e.clamp(-1.0, 1.0)) / 2.0,
        OsqSim::MaxInnerProduct => {
            if e < 0.0 {
                1.0 / (1.0 - e)
            } else {
                e + 1.0
            }
        }
    }
}

/// ES `ESVectorUtil.calculateOSQLoss`.
fn osq_loss(v: &[f32], a: f32, b: f32, points: i32, norm2: f32, codes: &mut [i32]) -> f32 {
    let step = (b - a) / (points as f32 - 1.0);
    let inv = 1.0 / step;
    let (mut xe, mut e) = (0f32, 0f32);
    for (i, xi) in v.iter().enumerate() {
        codes[i] = jround((xi.max(a).min(b) - a) * inv);
        let xiq = fma(step, codes[i] as f32, a);
        let d = xi - xiq;
        e = fma(d, d, e);
        xe = fma(*xi, d, xe);
    }
    (1.0 - LAMBDA) * xe * xe / norm2 + LAMBDA * e
}

/// ES `OptimizedScalarQuantizer.optimizeIntervals`: `true` when `codes`
/// already holds the quantized vector.
fn optimize_intervals(iv: &mut [f32; 2], codes: &mut [i32], v: &[f32], norm2: f32, points: i32) -> bool {
    let mut loss = osq_loss(v, iv[0], iv[1], points, norm2, codes) as f64;
    let scale = (1.0 - LAMBDA) / norm2;
    if !scale.is_finite() {
        return true;
    }
    for _ in 0..ITERS {
        // calculateOSQGridPoints.
        let (mut daa, mut dab, mut dbb, mut dax, mut dbx) = (0f32, 0f32, 0f32, 0f32, 0f32);
        let inv_pm = 1.0 / (points as f32 - 1.0);
        for (i, x) in v.iter().enumerate() {
            let s = codes[i] as f32 * inv_pm;
            let ms = 1.0 - s;
            daa = fma(ms, ms, daa);
            dab = fma(ms, s, dab);
            dbb = fma(s, s, dbb);
            dax = fma(ms, *x, dax);
            dbx = fma(s, *x, dbx);
        }
        // Java evaluates these in `float` before widening to `double`.
        let m0 = (scale * dax * dax + LAMBDA * daa) as f64;
        let m1 = (scale * dax * dbx + LAMBDA * dab) as f64;
        let m2 = (scale * dbx * dbx + LAMBDA * dbb) as f64;
        let det = m0 * m2 - m1 * m1;
        if det == 0.0 {
            return true;
        }
        let a = ((m2 * dax as f64 - m1 * dbx as f64) / det) as f32;
        let b = ((m0 * dbx as f64 - m1 * dax as f64) / det) as f32;
        if (iv[0] - a).abs() < 1e-8 && (iv[1] - b).abs() < 1e-8 {
            return true;
        }
        if b < a {
            return true;
        }
        let new_loss = osq_loss(v, a, b, points, norm2, codes) as f64;
        if new_loss > loss {
            return false;
        }
        iv[0] = a;
        iv[1] = b;
        loss = new_loss;
    }
    true
}

/// ES `quantizeVectorWithIntervals`: the codes and their sum.
fn quantize_with_intervals(v: &[f32], codes: &mut [i32], lo: f32, hi: f32, bits: u8) -> i32 {
    let steps = ((1i32 << bits) - 1) as f32;
    let inv = steps / (hi - lo);
    let mut sum = 0;
    for (i, x) in v.iter().enumerate() {
        let xi = x.max(lo).min(hi);
        let c = jround((xi - lo) * inv);
        sum += c;
        codes[i] = c;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_track_exact_scores() {
        let rows: Vec<Vec<f32>> = (0..50).map(|i| (0..16).map(|j| ((i * 7 + j * 3) % 11) as f32 / 11.0 - 0.4).collect()).collect();
        for sim in [OsqSim::Euclidean, OsqSim::MaxInnerProduct] {
            let q = OsqQuantizer::fit(sim, rows.len(), 16, 7, 7, |i| rows[i].as_slice());
            let query: Vec<f32> = (0..16).map(|j| (j as f32 / 16.0) - 0.3).collect();
            let (mut qc, mut dc) = (Vec::new(), Vec::new());
            let qcorr = q.quantize_query(&query, &mut qc);
            for r in &rows {
                let dcorr = q.quantize_doc(r, &mut dc);
                let est = q.score(&qc, &qcorr, &dc, &dcorr);
                let exact = match sim {
                    OsqSim::Euclidean => 1.0 / (1.0 + r.iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()),
                    _ => {
                        let d: f32 = r.iter().zip(&query).map(|(a, b)| a * b).sum();
                        if d < 0.0 { 1.0 / (1.0 - d) } else { d + 1.0 }
                    }
                };
                assert!((est - exact).abs() < 0.02, "{sim:?}: {est} vs {exact}");
            }
        }
    }
}

#[cfg(test)]
mod upstream_encoded_oracle {
    use super::*;
    #[test]
    fn upstream_dot_correction_preserves_lucene_addition_order() {
        // Actual Lucene 10.5.1 oracle27 case19 (7-bit, 96 dimensions).
        // The prior left-associated expression produced bits1063352658.
        let q = OsqQuantizer { sim: OsqSim::Dot, centroid: vec![0.0;96],
            centroid_dp: 0.030792475, doc_bits:7, query_bits:7 };
        let qc = OsqCorrections { lower:-0.16737431, upper:0.18195963,
            additional:-0.078334145, component_sum:5905 };
        let dc = OsqCorrections { lower:-0.17048097, upper:0.17678557,
            additional:-0.08506857, component_sum:5980 };
        assert_eq!(q.score_raw(494902.0, &qc, &dc).to_bits(), 1063352659);
    }

    /// Explicit EC2 gate: the golden file must come from the pinned Java generator.
    /// Ignored in ordinary unit runs because generation requires upstream jars.
    #[test]
    #[ignore = "requires externally generated Lucene 10.5.1 oracle; run explicit EC2 oracle gate"]
    fn lucene_1051_encoded_score_oracle() {
        let path = std::env::var("PR3_ENCODED_SCORE_ORACLE").expect("required upstream golden JSON path");
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(value["lucene"], "10.5.1");
        assert_eq!(value["oracle"], "Lucene104ScalarQuantizedVectorScorer");
        let cases = value["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 192);
        let floats = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect::<Vec<_>>();
        let correction = |v: &serde_json::Value| {
            let a = floats(v); assert_eq!(a.len(), 4);
            OsqCorrections { lower: a[0], upper: a[1], additional: a[2], component_sum: a[3] as i32 }
        };
        for (i, c) in cases.iter().enumerate() {
            let bits = c["bits"].as_u64().unwrap() as u8;
            assert!([7,4,1].contains(&bits));
            let sim = match c["sim"].as_str().unwrap() { "l2" => OsqSim::Euclidean, "dot"|"cosine" => OsqSim::Dot, "mip" => OsqSim::MaxInnerProduct, _ => panic!("unexpected similarity") };
            let mut q = OsqQuantizer::from_centroid(sim, floats(&c["centroid"]), bits, if bits == 1 {4} else {bits});
            // Keep calibration out of this oracle: compare identical upstream
            // encoded inputs and correction terms, not differently fitted graphs.
            q.centroid_dp = c["centroid_dp"].as_f64().unwrap() as f32;
            let codes = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|n| u8::try_from(n.as_u64().unwrap()).unwrap()).collect::<Vec<_>>();
            let query = codes(&c["query_codes"]); let doc = codes(&c["doc_codes"]);
            assert_eq!(query.len(), q.centroid.len()); assert_eq!(doc.len(), query.len());
            let actual = q.score(&query, &correction(&c["query_correction"]), &doc, &correction(&c["doc_correction"]));
            let expected = c["score"].as_f64().unwrap() as f32;
            // Start strict. Any arithmetic difference must be examined against
            // upstream operation ordering, not hidden by the old 25%/infinite bounds.
            assert_eq!(actual.to_bits(), expected.to_bits(), "case {i}: {actual} vs {expected}");
        }
    }
}
