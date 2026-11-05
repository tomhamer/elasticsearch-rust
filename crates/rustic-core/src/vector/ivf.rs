//! DiskBBQ (`index_options.type: bbq_disk`) building blocks, after ES
//! 9.5's `ES950DiskBBQVectorsFormat` (`IVFVectorsWriter`,
//! `ES950DiskBBQVectorsWriter`, `IVFVectorsReader`): clustering, spill and
//! codes. The index itself, one per field of a shard shared by all its
//! vector segments, is [`super::ivf_field`].
//!
//! * Hierarchical k-means (`HierarchicalKMeans`): `clamp(n / cluster_size,
//!   2, 128)` clusters fitted on a sample of 64 vectors per cluster, every
//!   cluster above `cluster_size` split recursively, then one
//!   neighbourhood-restricted refinement (`KMeansLocal`, 128 neighbouring
//!   centroids) over a sample of 32 vectors per centroid and a final pass
//!   over every vector.
//! * SOAR spill (`Soar`, lambda 1): each vector's second centroid
//!   minimizes `|x - c'|^2 + lambda ((x - c') . r)^2 / |r|^2`, `r = x - c`
//!   the residual to its own centroid.
//! * Codes: ES's optimized scalar quantizer *relative to a centroid*
//!   (`OptimizedScalarQuantizer.scalarQuantize(.., centroid)`) at `bits`
//!   1, 2, 4 or 7 (`QuantEncoding`) with the four corrective terms; queries
//!   are quantized to 4 bits (7 for 7-bit codes). A code is `bits` bit
//!   planes of `ceil(dim / 64)` words (bit `d` of plane `p` is bit `p` of
//!   component `d`), or for 7 bits one byte per component padded to 8.
//! * `cosine` vectors are clustered and quantized unit length (ES stores
//!   them normalized); `precondition` applies ES's random block-orthogonal
//!   rotation (`Preconditioner`, 32-dim blocks) first.
//! * The visit budget: ES's dynamic ratio (`computeDynamicVisitRatio`,
//!   capped by `computeSegmentSizeCap`) and collector size
//!   (`IvfSegmentConfig.leafCollectorBudget`).

use rayon::prelude::*;

use super::distance::{dot, l2_sq};
use super::osq::OsqSim;
use crate::error::{Result, RusticError};

/// ES `ES950DiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER`.
pub const DEFAULT_VECTORS_PER_CLUSTER: u32 = 384;
/// ES `MIN_VECTORS_PER_CLUSTER`.
pub const MIN_VECTORS_PER_CLUSTER: u32 = 64;
/// ES `MAX_VECTORS_PER_CLUSTER`.
pub const MAX_VECTORS_PER_CLUSTER: u32 = 1 << 16;
/// ES `DEFAULT_FLAT_VECTOR_THRESHOLD_MULTIPLIER`.
const FLAT_THRESHOLD_MULTIPLIER: usize = 3;
/// `HierarchicalKMeans.MAXK`.
const MAXK: usize = 128;
/// `HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT`.
const SAMPLES_PER_CLUSTER: usize = 64;
/// `HierarchicalKMeans` Lloyd iterations (unbalanced).
const MAX_ITERATIONS: usize = 6;
/// `HierarchicalKMeans.MAXK` as the refinement neighbourhood size.
pub(crate) const CLUSTERS_PER_NEIGHBORHOOD: usize = 128;
/// `HierarchicalKMeans.DEFAULT_SOAR_LAMBDA`.
pub(crate) const SOAR_LAMBDA: f32 = 1.0;
/// `Soar.SOAR_MIN_DISTANCE`.
pub(crate) const SOAR_MIN_DISTANCE: f32 = 1e-16;
/// ES `DEFAULT_PRECONDITIONING_BLOCK_DIMENSION`.
const PRECONDITION_BLOCK: usize = 32;
pub(crate) const NO_SPILL: u32 = u32::MAX;
/// ES seeds its k-means samples and preconditioner with 42.
const SEED: u64 = 42;
/// Vectors per parallel work item.
const CHUNK: usize = 1024;

/// Bytes of one entry's corrective terms.
pub(crate) const CORRECTION_BYTES: usize = 16;

/// A `bbq_disk` field's IVF parameters (its `index_options`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IvfOptions {
    /// Target vectors per posting list (`cluster_size`).
    pub cluster_size: u32,
    /// `flat_index_threshold`: `-1` dynamic (three times `cluster_size`),
    /// `0` never flat, else segments of at most this many vectors get one
    /// centroid.
    pub flat_index_threshold: i32,
    /// `default_visit_percentage` (`0`: ES's dynamic visit ratio).
    pub default_visit_percentage: f32,
    /// Code width: 1, 2, 4 or 7 (`bits`).
    pub bits: u8,
    /// `on_disk_rescore`: ES reads rescoring vectors with direct IO. Our
    /// rescoring always reads the mapped `f32` column, never a heap copy.
    #[serde(default)]
    pub on_disk_rescore: bool,
    /// `precondition`: rotate vectors before clustering and quantization.
    #[serde(default)]
    pub precondition: bool,
    /// The oversample the collector is sized for (ES
    /// `IvfQueryConfigResolver`'s mapping oversample:
    /// `rescore_vector.oversample`, else `DEFAULT_OVERSAMPLE` 3, even when
    /// nothing is rescored).
    #[serde(default = "default_budget_oversample")]
    pub budget_oversample: f32,
    /// The mapping sets `rescore_vector.oversample`: rescoring takes ES's
    /// `ceil(k * oversample)` best estimates, not every collected one.
    #[serde(default)]
    pub explicit_oversample: bool,
}

fn default_budget_oversample() -> f32 {
    3.0
}

impl Default for IvfOptions {
    fn default() -> Self {
        Self { cluster_size: DEFAULT_VECTORS_PER_CLUSTER, flat_index_threshold: -1, default_visit_percentage: 0.0, bits: 1, on_disk_rescore: false, precondition: false, budget_oversample: 3.0, explicit_oversample: false }
    }
}

impl IvfOptions {
    /// `cluster_size` within ES's bounds.
    pub fn target_cluster_size(&self) -> usize {
        self.cluster_size.clamp(MIN_VECTORS_PER_CLUSTER, MAX_VECTORS_PER_CLUSTER) as usize
    }

    /// Segments of at most this many vectors are one flat posting list
    /// (`0`: none are).
    pub fn flat_threshold(&self) -> usize {
        match self.flat_index_threshold {
            t if t < 0 => self.target_cluster_size() * FLAT_THRESHOLD_MULTIPLIER,
            t => t as usize,
        }
    }

    /// Stored code width (`QuantEncoding.fromBits`).
    pub fn doc_bits(&self) -> u8 {
        match self.bits {
            1 | 2 | 4 | 7 => self.bits,
            _ => 1,
        }
    }

    /// Query code width (`QuantEncoding.queryBits`).
    pub fn query_bits(&self) -> u8 {
        if self.doc_bits() == 7 { 7 } else { 4 }
    }

    /// The mapping's visit ratio (`0`: dynamic).
    pub fn default_visit_ratio(&self) -> f32 {
        (self.default_visit_percentage / 100.0).clamp(0.0, 1.0)
    }
}

/// ES `IVFVectorsReader.computeDynamicVisitRatio`: a visit ratio from the
/// `num_candidates / k` ratio and `k` (the two-signal model).
pub fn dynamic_visit_ratio(num_candidates: usize, k: usize) -> f32 {
    const V_MIN: f64 = 0.003;
    const V_MAX: f64 = 0.04;
    const RATIO_WEIGHT: f64 = 0.85;
    const K_WEIGHT: f64 = 0.15;
    let log_scale = |v: f64, max: f64| (v.ln_1p() / max.ln_1p()).clamp(0.0, 1.0);
    let r = num_candidates as f64 / k.max(1) as f64;
    let z = RATIO_WEIGHT * log_scale(r - 1.0, 10.0) + K_WEIGHT * log_scale(k as f64, 10_000.0);
    (V_MIN + (V_MAX - V_MIN) * z) as f32
}

/// ES `IVFVectorsReader.computeSegmentSizeCap`: larger segments have better
/// formed clusters and need a lower ratio.
pub fn segment_size_cap(num_vectors: usize) -> f32 {
    const CAP_COEFFICIENT: f64 = 0.045;
    const CAP_REF_SIZE: f64 = 1_000_000.0;
    const CAP_EXPONENT: f64 = 0.35;
    const TARGET_RECALL: f64 = 0.9;
    if num_vectors == 0 {
        return 0.04;
    }
    let size_scale = (CAP_REF_SIZE / num_vectors as f64).powf(CAP_EXPONENT);
    let recall_scale = 0.1 / (1.0 - TARGET_RECALL);
    (CAP_COEFFICIENT * size_scale * recall_scale).min(1.0) as f32
}

/// ES `IvfSegmentConfig.leafCollectorBudget`: candidates one segment
/// collects for a result of `k` (twice the oversampled count: spilled
/// duplicates take collector slots).
pub fn leaf_budget(k: usize, oversample: f32) -> usize {
    (2.0 * k as f32 * oversample.max(1.0)).round() as usize
}

// ---------------------------------------------------------------------------
// Deterministic randomness (k-means samples, preconditioner).

/// SplitMix64.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Uniform in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Standard normal (Box-Muller).
    pub(crate) fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Preconditioner.

/// ES `Preconditioner`: a random rotation made of orthogonal blocks over a
/// random partition of the dimensions. Distances and inner products are
/// unchanged, but the quantized coordinates are decorrelated.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Preconditioner {
    dim: usize,
    /// `(source dimensions, row-major orthogonal matrix)` per block; block
    /// outputs are laid out in order.
    blocks: Vec<(Vec<u32>, Vec<f32>)>,
}

impl Preconditioner {
    /// ES `createPreconditioner(dim, 32)`, with our generator.
    pub(crate) fn new(dim: usize) -> Self {
        let mut rng = Rng::new(SEED);
        let block = PRECONDITION_BLOCK.min(dim).max(1);
        let mut sizes = vec![block; dim / block];
        if dim % block != 0 {
            sizes.push(dim % block);
        }
        let matrices: Vec<Vec<f32>> = sizes.iter().map(|&s| {
            let mut m: Vec<f32> = (0..s * s).map(|_| rng.gaussian() as f32).collect();
            gram_schmidt(&mut m, s);
            m
        }).collect();
        let mut order: Vec<u32> = (0..dim as u32).collect();
        for i in (1..dim).rev() {
            let j = rng.below(i + 1);
            order.swap(i, j);
        }
        let mut at = 0;
        let blocks = sizes.iter().zip(matrices).map(|(&s, m)| {
            let mut dims = order[at..at + s].to_vec();
            dims.sort_unstable();
            at += s;
            (dims, m)
        }).collect();
        Preconditioner { dim, blocks }
    }

    /// `out = R v`.
    pub(crate) fn apply(&self, v: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(v.len(), self.dim);
        out.clear();
        let mut x = Vec::with_capacity(PRECONDITION_BLOCK);
        for (dims, m) in &self.blocks {
            let s = dims.len();
            x.clear();
            x.extend(dims.iter().map(|&d| v[d as usize]));
            for r in 0..s {
                out.push(dot(&m[r * s..(r + 1) * s], &x));
            }
        }
    }

    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for (dims, m) in &self.blocks {
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            for x in m {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }

    pub(crate) fn decode(b: &[u8], dim: usize) -> Result<Self> {
        let bad = || RusticError::Corrupted("bad IVF preconditioner".into());
        let mut at = 0;
        let u32_at = |at: &mut usize| -> Result<u32> {
            let v = b.get(*at..*at + 4).ok_or_else(bad)?;
            *at += 4;
            Ok(u32::from_le_bytes(v.try_into().unwrap()))
        };
        let n = u32_at(&mut at)? as usize;
        // Every dimension needs an index and at least one matrix float;
        // every nonempty block also needs its length. Reject forged counts
        // before allocating arrays from persisted metadata.
        if dim == 0 || n == 0 || n > dim
            || dim > b.len().saturating_sub(4) / 8
            || n > b.len().saturating_sub(4) / 12 {
            return Err(bad());
        }
        let mut seen = vec![false; dim];
        let mut blocks = Vec::with_capacity(n.min(dim));
        for _ in 0..n {
            let s = u32_at(&mut at)? as usize;
            if s == 0 || s > PRECONDITION_BLOCK || s > dim {
                return Err(bad());
            }
            let mut dims = Vec::with_capacity(s);
            for _ in 0..s {
                let d = u32_at(&mut at)? as usize;
                if d >= dim || std::mem::replace(&mut seen[d], true) {
                    return Err(bad());
                }
                dims.push(d as u32);
            }
            let bytes = b.get(at..at + s * s * 4).ok_or_else(bad)?;
            at += s * s * 4;
            let m: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            if m.iter().any(|value| !value.is_finite()) {
                return Err(bad());
            }
            blocks.push((dims, m));
        }
        if at != b.len() || seen.iter().any(|s| !s) {
            return Err(bad());
        }
        Ok(Preconditioner { dim, blocks })
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.blocks.iter().map(|(d, m)| d.len() * 4 + m.len() * 4).sum()
    }
}

/// ES `Preconditioner.modifiedGramSchmidt` over the rows of `m` (`s x s`).
fn gram_schmidt(m: &mut [f32], s: usize) {
    for i in 0..s {
        let norm = m[i * s..(i + 1) * s].iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        for j in 0..s {
            m[i * s + j] = (m[i * s + j] as f64 / norm) as f32;
        }
        for k in i + 1..s {
            let d: f64 = (0..s).map(|j| m[i * s + j] as f64 * m[k * s + j] as f64).sum();
            for j in 0..s {
                m[k * s + j] -= (d * m[i * s + j] as f64) as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The vectors as clustered and quantized.

/// Per-thread buffers of [`Points::get`].
#[derive(Default)]
pub(crate) struct Scratch {
    unit: Vec<f32>,
    rotated: Vec<f32>,
}

/// ES's normalization of a `cosine` vector (`es_quant::es_unit`) into `out`.
fn unit_into(v: &[f32], out: &mut Vec<f32>) {
    out.clear();
    let sq: f32 = v.iter().map(|x| x * x).sum();
    if (sq - 1.0).abs() <= 1e-3 || sq == 0.0 {
        out.extend_from_slice(v);
    } else {
        let len = (sq as f64).sqrt() as f32;
        out.extend(v.iter().map(|x| x / len));
    }
}

/// A segment's vectors as the index sees them: unit length for `cosine`,
/// rotated when preconditioned. Nothing is copied up front.
pub(crate) struct Points<'a> {
    rows: &'a [&'a [f32]],
    unit: bool,
    pre: Option<&'a Preconditioner>,
    dim: usize,
}

impl<'a> Points<'a> {
    pub(crate) fn new(rows: &'a [&'a [f32]], unit: bool, pre: Option<&'a Preconditioner>, dim: usize) -> Self {
        Points { rows, unit, pre, dim }
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn get<'b>(&'b self, i: usize, s: &'b mut Scratch) -> &'b [f32] {
        prepare(self.rows[i], self.unit, self.pre, s)
    }
}

/// `v` as an index sees it: unit length (`unit`, ES's `cosine`), then
/// rotated (`pre`); `v` itself when neither applies.
pub(crate) fn prepare<'b>(v: &'b [f32], unit: bool, pre: Option<&Preconditioner>, s: &'b mut Scratch) -> &'b [f32] {
    let Scratch { unit: normalized, rotated } = s;
    let v: &'b [f32] = if unit {
        unit_into(v, normalized);
        let normalized: &'b Vec<f32> = normalized;
        normalized.as_slice()
    } else {
        v
    };
    match pre {
        Some(p) => {
            p.apply(v, rotated);
            let rotated: &'b Vec<f32> = rotated;
            rotated.as_slice()
        }
        None => v,
    }
}

// ---------------------------------------------------------------------------
// Hierarchical k-means with SOAR spill.

/// A clustering: `centroids` (`k x dim`), each vector's own centroid and
/// its spilled one ([`NO_SPILL`]: none).
pub(crate) struct Clusters {
    pub dim: usize,
    pub centroids: Vec<f32>,
    pub assign: Vec<u32>,
    pub spill: Vec<u32>,
}

impl Clusters {
    pub fn len(&self) -> usize {
        if self.dim == 0 { 0 } else { self.centroids.len() / self.dim }
    }
}

pub(crate) fn centroid(centroids: &[f32], dim: usize, c: usize) -> &[f32] {
    &centroids[c * dim..(c + 1) * dim]
}

/// The mean of `ids`.
fn mean(points: &Points, ids: &[u32]) -> Vec<f32> {
    let dim = points.dim;
    let partials: Vec<Vec<f64>> = ids.par_chunks(CHUNK).map_init(Scratch::default, |s, chunk| {
        let mut acc = vec![0f64; dim];
        for &i in chunk {
            for (a, x) in acc.iter_mut().zip(points.get(i as usize, s)) {
                *a += *x as f64;
            }
        }
        acc
    }).collect();
    let mut acc = vec![0f64; dim];
    for p in partials {
        for (a, x) in acc.iter_mut().zip(p) {
            *a += x;
        }
    }
    let inv = 1.0 / ids.len().max(1) as f64;
    acc.into_iter().map(|a| (a * inv) as f32).collect()
}

/// Assign each of `ids` (positions of `out`) to its nearest centroid; with
/// `neighbours`, only among its current centroid and that centroid's
/// neighbourhood. Whether any assignment changed.
fn assign_nearest(points: &Points, ids: &[u32], centroids: &[f32], neighbours: Option<&[Vec<u32>]>, out: &mut [u32]) -> bool {
    let dim = points.dim;
    let k = centroids.len() / dim;
    ids.par_chunks(CHUNK).zip(out.par_chunks_mut(CHUNK)).map_init(Scratch::default, |s, (chunk, out)| {
        let mut changed = false;
        for (&i, slot) in chunk.iter().zip(out.iter_mut()) {
            let v = points.get(i as usize, s);
            let cur = *slot as usize;
            let best = match neighbours {
                Some(nb) if cur < k => {
                    let mut best = cur;
                    let mut bd = l2_sq(v, centroid(centroids, dim, cur));
                    for &o in &nb[cur] {
                        let d = l2_sq(v, centroid(centroids, dim, o as usize));
                        if d < bd {
                            bd = d;
                            best = o as usize;
                        }
                    }
                    best
                }
                _ => {
                    let mut best = 0;
                    let mut bd = f32::INFINITY;
                    for c in 0..k {
                        let d = l2_sq(v, centroid(centroids, dim, c));
                        if d < bd {
                            bd = d;
                            best = c;
                        }
                    }
                    best
                }
            };
            if best != cur {
                *slot = best as u32;
                changed = true;
            }
        }
        changed
    }).reduce(|| false, |a, b| a | b)
}

/// Move every centroid to the mean of its members (`assign` gives the
/// centroid of each of `ids`); empty centroids stay. Member counts.
fn update_centroids(points: &Points, ids: &[u32], assign: &[u32], centroids: &mut [f32]) -> Vec<u32> {
    let dim = points.dim;
    let k = centroids.len() / dim;
    let mut counts = vec![0u32; k];
    for &a in assign {
        counts[a as usize] += 1;
    }
    let mut starts = vec![0usize; k + 1];
    for c in 0..k {
        starts[c + 1] = starts[c] + counts[c] as usize;
    }
    let mut fill = starts.clone();
    let mut members = vec![0u32; ids.len()];
    for (pos, &a) in assign.iter().enumerate() {
        members[fill[a as usize]] = ids[pos];
        fill[a as usize] += 1;
    }
    centroids.par_chunks_mut(dim).enumerate().for_each_init(Scratch::default, |s, (c, out)| {
        let m = &members[starts[c]..starts[c + 1]];
        if m.is_empty() {
            return;
        }
        let mut acc = vec![0f64; dim];
        for &i in m {
            for (a, x) in acc.iter_mut().zip(points.get(i as usize, s)) {
                *a += *x as f64;
            }
        }
        let inv = 1.0 / m.len() as f64;
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = (a * inv) as f32;
        }
    });
    counts
}

/// Lloyd's iterations over `ids` (ES `LloydKMeansLocal`): stops early
/// once no assignment changes.
fn lloyd(points: &Points, ids: &[u32], centroids: &mut [f32], neighbours: Option<&[Vec<u32>]>, assign: &mut [u32], iterations: usize, work: &mut dyn FnMut(usize)) {
    for _ in 0..iterations {
        let changed = assign_nearest(points, ids, centroids, neighbours, assign);
        work(ids.len());
        if !changed {
            break;
        }
        update_centroids(points, ids, assign, centroids);
    }
}

/// `m` distinct positions of `0..n`, ascending (a partial Fisher-Yates
/// shuffle).
fn sample_positions(n: usize, m: usize, rng: &mut Rng) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    for i in 0..m.min(n) {
        let j = i + rng.below(n - i);
        all.swap(i, j);
    }
    all.truncate(m.min(n));
    all.sort_unstable();
    all
}

/// ES `KMeansLocal.pickInitialCentroids`: reservoir sampling of `k` of `ids`.
fn initial_centroids(points: &Points, ids: &[u32], k: usize) -> Vec<f32> {
    let mut rng = Rng::new(SEED);
    let k = k.min(ids.len());
    let mut chosen: Vec<u32> = ids[..k].to_vec();
    for (i, &id) in ids.iter().enumerate().skip(k) {
        if rng.next_f64() < k as f64 / i as f64 {
            chosen[rng.below(k)] = id;
        }
    }
    let mut s = Scratch::default();
    let mut out = Vec::with_capacity(k * points.dim);
    for &id in &chosen {
        out.extend_from_slice(points.get(id as usize, &mut s));
    }
    out
}

/// ES `HierarchicalKMeans.clusterAndSplit` over `ids`: centroids and the
/// centroid of each of `ids`.
fn cluster_and_split(points: &Points, ids: &[u32], target: usize, work: &mut dyn FnMut(usize)) -> (Vec<f32>, Vec<u32>) {
    let dim = points.dim;
    let n = ids.len();
    if n <= target {
        return (mean(points, ids), vec![0; n]);
    }
    let k = (((n as f32 + target as f32 / 2.0) / target as f32) as usize).clamp(2, MAXK).min(n);
    let m = (k * SAMPLES_PER_CLUSTER).min(n);
    let mut centroids = initial_centroids(points, ids, k);
    let mut assign = vec![u32::MAX; n];
    if m < n {
        let mut rng = Rng::new(SEED);
        let sample: Vec<u32> = sample_positions(n, m, &mut rng).into_iter().map(|p| ids[p]).collect();
        let mut sample_assign = vec![u32::MAX; m];
        lloyd(points, &sample, &mut centroids, None, &mut sample_assign, MAX_ITERATIONS, work);
        // A final pass over every vector.
        assign_nearest(points, ids, &centroids, None, &mut assign);
        work(n);
        update_centroids(points, ids, &assign, &mut centroids);
    } else {
        lloyd(points, ids, &mut centroids, None, &mut assign, MAX_ITERATIONS, work);
    }
    let mut counts = vec![0usize; k];
    for &a in &assign {
        counts[a as usize] += 1;
    }
    // Split every cluster above the target (one that could not be split,
    // all of `ids` in one cluster, is kept).
    let mut next = k as u32;
    for c in 0..k {
        if counts[c] <= target || counts[c] >= n {
            continue;
        }
        let positions: Vec<usize> = (0..n).filter(|&p| assign[p] == c as u32).collect();
        let sub_ids: Vec<u32> = positions.iter().map(|&p| ids[p]).collect();
        let (sub_centroids, sub_assign) = cluster_and_split(points, &sub_ids, target, work);
        let sub_k = sub_centroids.len() / dim;
        if sub_k <= 1 {
            continue;
        }
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&sub_centroids[..dim]);
        let base = next;
        centroids.extend_from_slice(&sub_centroids[dim..]);
        next += (sub_k - 1) as u32;
        for (&p, &a) in positions.iter().zip(&sub_assign) {
            assign[p] = if a == 0 { c as u32 } else { base + a - 1 };
        }
    }
    (centroids, assign)
}

/// Drop centroids without members (ES `removeEmptyClusters`), renumbering
/// the assignments.
fn remove_empty(centroids: &mut Vec<f32>, assign: &mut [u32], dim: usize) {
    let k = centroids.len() / dim;
    let mut counts = vec![0u32; k];
    for &a in assign.iter() {
        counts[a as usize] += 1;
    }
    if counts.iter().all(|&c| c > 0) {
        return;
    }
    let mut remap = vec![u32::MAX; k];
    let mut kept = Vec::with_capacity(centroids.len());
    for c in 0..k {
        if counts[c] > 0 {
            remap[c] = (kept.len() / dim) as u32;
            kept.extend_from_slice(centroid(centroids, dim, c));
        }
    }
    *centroids = kept;
    for a in assign.iter_mut() {
        *a = remap[*a as usize];
    }
}

/// The `count` nearest other centroids of every centroid
/// (`NeighborHood.computeNeighborhoodsBruteForce`).
fn neighbourhoods(centroids: &[f32], dim: usize, count: usize) -> Vec<Vec<u32>> {
    let k = centroids.len() / dim;
    let cmp = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    (0..k).into_par_iter().map(|c| {
        let me = centroid(centroids, dim, c);
        let mut d: Vec<(f32, u32)> = (0..k).filter(|&o| o != c).map(|o| (l2_sq(me, centroid(centroids, dim, o)), o as u32)).collect();
        let take = count.min(d.len());
        if take > 0 && take < d.len() {
            d.select_nth_unstable_by(take - 1, cmp);
            d.truncate(take);
        }
        d.sort_by(cmp);
        d.into_iter().map(|x| x.1).collect()
    }).collect()
}

/// ES `Soar.assignSpilled`: each vector's second centroid, minimizing the
/// SOAR loss among its own centroid's neighbourhood (all others when there
/// are few centroids).
fn soar(points: &Points, centroids: &[f32], assign: &[u32], neighbours: Option<&[Vec<u32>]>) -> Vec<u32> {
    let dim = points.dim;
    let k = centroids.len() / dim;
    (0..points.len()).into_par_iter().map_init(|| (Scratch::default(), Vec::with_capacity(dim)), |state, i| {
        let (s, r) = state;
        let v = points.get(i, s);
        let c = assign[i] as usize;
        let own = centroid(centroids, dim, c);
        let dist = l2_sq(v, own);
        if dist <= SOAR_MIN_DISTANCE {
            return NO_SPILL;
        }
        r.clear();
        r.extend(v.iter().zip(own).map(|(a, b)| a - b));
        let r: &[f32] = r;
        let vr = dot(v, r);
        let mut best = NO_SPILL;
        let mut min = f32::MAX;
        let mut consider = |o: usize| {
            let co = centroid(centroids, dim, o);
            let proj = vr - dot(co, r);
            let loss = l2_sq(v, co) + SOAR_LAMBDA * proj * proj / dist;
            if loss < min {
                min = loss;
                best = o as u32;
            }
        };
        match neighbours {
            Some(nb) => {
                for &o in &nb[c] {
                    consider(o as usize);
                }
            }
            None => {
                for o in (0..k).filter(|&o| o != c) {
                    consider(o);
                }
            }
        }
        best
    }).collect()
}

/// One centroid, the mean, holding everything
/// (`buildFlatCentroidAssignments`).
pub(crate) fn single(points: &Points) -> Clusters {
    let ids: Vec<u32> = (0..points.len() as u32).collect();
    Clusters { dim: points.dim, centroids: mean(points, &ids), assign: vec![0; points.len()], spill: vec![NO_SPILL; points.len()] }
}

/// ES `HierarchicalKMeans.cluster` with SOAR: `work(vectors)` is called
/// after every pass over that many vectors (the caller's throttle).
pub(crate) fn cluster(points: &Points, target: usize, work: &mut dyn FnMut(usize)) -> Clusters {
    let n = points.len();
    let dim = points.dim;
    if n == 0 {
        return Clusters { dim, centroids: Vec::new(), assign: Vec::new(), spill: Vec::new() };
    }
    if n <= target {
        return single(points);
    }
    let ids: Vec<u32> = (0..n as u32).collect();
    let (mut centroids, mut assign) = cluster_and_split(points, &ids, target, work);
    remove_empty(&mut centroids, &mut assign, dim);
    let k = centroids.len() / dim;
    if k <= 1 || k >= n {
        return Clusters { dim, centroids, assign, spill: vec![NO_SPILL; n] };
    }
    // Neighbourhood-aware refinement (`KMeansLocal.cluster` with
    // `clustersPerNeighborhood`), on a sample, then every vector.
    let neighbours = (k > CLUSTERS_PER_NEIGHBORHOOD).then(|| neighbourhoods(&centroids, dim, CLUSTERS_PER_NEIGHBORHOOD));
    let local = (k * SAMPLES_PER_CLUSTER / 2).min(n);
    if local < n {
        let mut rng = Rng::new(SEED);
        let positions = sample_positions(n, local, &mut rng);
        let sample: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let mut sample_assign: Vec<u32> = positions.iter().map(|&p| assign[p]).collect();
        lloyd(points, &sample, &mut centroids, neighbours.as_deref(), &mut sample_assign, MAX_ITERATIONS, work);
        assign_nearest(points, &ids, &centroids, neighbours.as_deref(), &mut assign);
        work(n);
        update_centroids(points, &ids, &assign, &mut centroids);
    } else {
        lloyd(points, &ids, &mut centroids, neighbours.as_deref(), &mut assign, MAX_ITERATIONS, work);
    }
    remove_empty(&mut centroids, &mut assign, dim);
    let k = centroids.len() / dim;
    if k <= 1 {
        return Clusters { dim, centroids, assign, spill: vec![NO_SPILL; n] };
    }
    let neighbours = (k > CLUSTERS_PER_NEIGHBORHOOD).then(|| neighbourhoods(&centroids, dim, CLUSTERS_PER_NEIGHBORHOOD));
    let spill = soar(points, &centroids, &assign, neighbours.as_deref());
    work(n);
    Clusters { dim, centroids, assign, spill }
}

/// Split a cluster's vectors (`points`, all of them) as
/// `HierarchicalKMeans.clusterAndSplit` does: centroids and each vector's
/// centroid among them (one centroid when `points` fits `target`).
pub(crate) fn split(points: &Points, target: usize, work: &mut dyn FnMut(usize)) -> (Vec<f32>, Vec<u32>) {
    let ids: Vec<u32> = (0..points.len() as u32).collect();
    let (mut centroids, mut assign) = cluster_and_split(points, &ids, target, work);
    remove_empty(&mut centroids, &mut assign, points.dim);
    (centroids, assign)
}

/// Incremental assignment of new vectors to existing centroids (the
/// shared index's inserts): each vector's nearest `active` centroid and its
/// SOAR spill among the 128 next nearest ([`NO_SPILL`]: none).
pub(crate) fn assign(points: &Points, centroids: &[f32], active: &[bool]) -> Vec<(u32, u32)> {
    let dim = points.dim;
    let k = centroids.len() / dim.max(1);
    (0..points.len()).into_par_iter().map_init(|| (Scratch::default(), Vec::with_capacity(k), Vec::with_capacity(dim)), |state, i| {
        let (s, dists, r) = state;
        let v = points.get(i, s);
        dists.clear();
        dists.extend((0..k).filter(|&c| active[c]).map(|c| (l2_sq(v, centroid(centroids, dim, c)), c as u32)));
        let cmp = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        let Some(&(dist, primary)) = dists.iter().min_by(|a, b| cmp(*a, *b)) else { return (u32::MAX, NO_SPILL) };
        if dist <= SOAR_MIN_DISTANCE || dists.len() < 2 {
            return (primary, NO_SPILL);
        }
        dists.retain(|x| x.1 != primary);
        let take = CLUSTERS_PER_NEIGHBORHOOD.min(dists.len());
        if take < dists.len() {
            dists.select_nth_unstable_by(take - 1, cmp);
            dists.truncate(take);
        }
        let own = centroid(centroids, dim, primary as usize);
        r.clear();
        r.extend(v.iter().zip(own).map(|(a, b)| a - b));
        let r: &[f32] = r;
        let vr = dot(v, r);
        let mut best = NO_SPILL;
        let mut min = f32::MAX;
        for &(dsq, o) in dists.iter() {
            let proj = vr - dot(centroid(centroids, dim, o as usize), r);
            let loss = dsq + SOAR_LAMBDA * proj * proj / dist;
            if loss < min || (loss == min && o < best) {
                min = loss;
                best = o;
            }
        }
        (primary, best)
    }).collect()
}

// ---------------------------------------------------------------------------
// Codes.

fn code_words(dim: usize) -> usize {
    dim.div_ceil(64)
}

/// Bytes of one stored code.
pub(crate) fn code_bytes(bits: u8, dim: usize) -> usize {
    match bits {
        7 => dim.div_ceil(8) * 8,
        b => b as usize * code_words(dim) * 8,
    }
}

/// Bit planes of `codes` (one value per component, `< 2^bits`), appended
/// to `out` as little-endian words; 7-bit codes are bytes.
pub(crate) fn pack_codes(bits: u8, codes: &[u8], dim: usize, out: &mut Vec<u8>) {
    if bits == 7 {
        out.extend_from_slice(&codes[..dim]);
        out.resize(out.len() + code_bytes(7, dim) - dim, 0);
        return;
    }
    for p in 0..bits {
        for w in 0..code_words(dim) {
            let mut x = 0u64;
            for b in 0..64 {
                let d = w * 64 + b;
                if d < dim && (codes[d] >> p) & 1 == 1 {
                    x |= 1 << b;
                }
            }
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

/// The four bit planes of a 4-bit query code.
pub(crate) fn query_planes(codes: &[u8], dim: usize, out: &mut Vec<u64>) {
    let w = code_words(dim);
    out.clear();
    out.resize(4 * w, 0);
    for (d, &c) in codes[..dim].iter().enumerate() {
        let (word, bit) = (d / 64, d % 64);
        let c = u64::from(c);
        out[word] |= (c & 1) << bit;
        out[w + word] |= (c >> 1 & 1) << bit;
        out[2 * w + word] |= (c >> 2 & 1) << bit;
        out[3 * w + word] |= (c >> 3 & 1) << bit;
    }
}

/// `sum_d q_d x_d` of a stored code and a query code.
#[inline]
pub(crate) fn raw_dot(bits: u8, code: &[u8], planes: &[u64], query: &[u8], dim: usize) -> u32 {
    if bits == 7 {
        return super::int4::dot_u8(&code[..dim], &query[..dim]);
    }
    let w = code_words(dim);
    let mut total = 0u32;
    if bits == 1 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline; the reads stay
        // within `code[..w * 8]` and `planes[..4 * w]` (checked below).
        unsafe {
            assert!(code.len() >= w * 8 && planes.len() >= 4 * w);
            return raw_dot_1bit_neon(code.as_ptr(), planes.as_ptr(), w);
        }
        // One doc plane: each doc word read once against the 4 query planes.
        #[allow(unreachable_code)]
        for (j, d) in code[..w * 8].chunks_exact(8).enumerate() {
            let d = u64::from_le_bytes(d.try_into().unwrap());
            total += (d & planes[j]).count_ones()
                + ((d & planes[w + j]).count_ones() << 1)
                + ((d & planes[2 * w + j]).count_ones() << 2)
                + ((d & planes[3 * w + j]).count_ones() << 3);
        }
        return total;
    }
    for t in 0..bits as usize {
        let doc = &code[t * w * 8..(t + 1) * w * 8];
        for p in 0..4 {
            let q = &planes[p * w..(p + 1) * w];
            let mut c = 0u32;
            for (j, qw) in q.iter().enumerate() {
                let d = u64::from_le_bytes(doc[j * 8..j * 8 + 8].try_into().unwrap());
                c += (d & qw).count_ones();
            }
            total += c << (p + t);
        }
    }
    total
}

/// [`raw_dot`] of a 1-bit code (`w` words at `code`) and the four query
/// planes (`4 w` words at `planes`): 128 bits at a time, the plane-weighted
/// per-byte popcounts (`cnt0 + 2 cnt1 + 4 cnt2 + 8 cnt3`, at most 120 a
/// byte) widened into 16-bit lanes, summed once at the end.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn raw_dot_1bit_neon(code: *const u8, planes: *const u64, w: usize) -> u32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_u16(0);
        let mut total = 0u32;
        let mut j = 0;
        let mut n = 0;
        while j + 1 < w {
            let d = vld1q_u8(code.add(j * 8));
            let c0 = vcntq_u8(vandq_u8(d, vld1q_u8(planes.add(j) as *const u8)));
            let c1 = vcntq_u8(vandq_u8(d, vld1q_u8(planes.add(w + j) as *const u8)));
            let c2 = vcntq_u8(vandq_u8(d, vld1q_u8(planes.add(2 * w + j) as *const u8)));
            let c3 = vcntq_u8(vandq_u8(d, vld1q_u8(planes.add(3 * w + j) as *const u8)));
            let weighted = vaddq_u8(vaddq_u8(c0, vshlq_n_u8::<1>(c1)), vaddq_u8(vshlq_n_u8::<2>(c2), vshlq_n_u8::<3>(c3)));
            acc = vpadalq_u8(acc, weighted);
            j += 2;
            n += 1;
            // Each 16-bit lane gains at most 240 a chunk: flush well before
            // it could wrap.
            if n == 256 {
                total += vaddlvq_u16(acc);
                acc = vdupq_n_u16(0);
                n = 0;
            }
        }
        total += vaddlvq_u16(acc);
        if j < w {
            let d = (code.add(j * 8) as *const u64).read_unaligned();
            total += (d & *planes.add(j)).count_ones()
                + ((d & *planes.add(w + j)).count_ones() << 1)
                + ((d & *planes.add(2 * w + j)).count_ones() << 2)
                + ((d & *planes.add(3 * w + j)).count_ones() << 3);
        }
        total
    }
}

pub(crate) fn sim_tag(s: OsqSim) -> u8 {
    match s {
        OsqSim::Euclidean => 0,
        OsqSim::Dot => 1,
        OsqSim::MaxInnerProduct => 2,
    }
}

pub(crate) fn sim_from_tag(t: u8) -> Option<OsqSim> {
    Some(match t {
        0 => OsqSim::Euclidean,
        1 => OsqSim::Dot,
        2 => OsqSim::MaxInnerProduct,
        _ => return None,
    })
}

pub(crate) fn pad8(out: &mut Vec<u8>) {
    while out.len() % 8 != 0 {
        out.push(0);
    }
}

pub(crate) fn align8(x: usize) -> usize {
    (x + 7) & !7
}

#[cfg(test)]
/// Embedding-like data: `centres` cluster centres, each vector its
/// centre plus a low-rank (8 latent dimensions) variation and a little
/// isotropic noise, so neighbourhoods are not drowned by concentration
/// of measure as in isotropic high-dimensional noise.
pub(crate) fn test_corpus(seed: u64, n: usize, dim: usize, centres: usize) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    let c: Vec<Vec<f32>> = (0..centres).map(|_| (0..dim).map(|_| (r.next_f64() * 2.0 - 1.0) as f32).collect()).collect();
    let basis: Vec<Vec<f32>> = (0..centres * 8).map(|_| (0..dim).map(|_| r.gaussian() as f32 / (dim as f32).sqrt()).collect()).collect();
    (0..n).map(|_| {
        let ci = r.below(centres);
        let mut v = c[ci].clone();
        for b in &basis[ci * 8..ci * 8 + 8] {
            let w = r.gaussian() as f32 * 0.6;
            for (x, y) in v.iter_mut().zip(b) {
                *x += w * y;
            }
        }
        for x in v.iter_mut() {
            *x += r.gaussian() as f32 * 0.02;
        }
        v
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchical_kmeans_targets_the_cluster_size_and_assigns_nearest() {
        let rows = test_corpus(1, 6000, 32, 24);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let points = Points { rows: &refs, unit: false, pre: None, dim: 32 };
        let mut passes = 0;
        let c = cluster(&points, 128, &mut |_| passes += 1);
        assert!(passes > 0, "the work hook sees every pass");
        let k = c.len();
        // ES: top level clamp(n / target, 2, 128), split above the target.
        assert!((6000 / 128 / 2..=6000 / 128 * 3).contains(&k), "{k} centroids for 6000 vectors at 128");
        let mut sizes = vec![0usize; k];
        let mut nearest = 0;
        for (i, &a) in c.assign.iter().enumerate() {
            sizes[a as usize] += 1;
            // The final pass assigns every vector its nearest centroid; the
            // centroids then move to their members' means.
            let own = l2_sq(&rows[i], centroid(&c.centroids, 32, a as usize));
            let best = (0..k).map(|o| l2_sq(&rows[i], centroid(&c.centroids, 32, o))).fold(f32::INFINITY, f32::min);
            nearest += usize::from(own <= best * 1.0001 + 1e-6);
        }
        assert!(nearest * 100 >= rows.len() * 95, "{nearest} of {} in their nearest centroid", rows.len());
        assert!(sizes.iter().all(|&s| s > 0), "no empty clusters");
        assert!(*sizes.iter().max().unwrap() <= 128 * 4, "oversized cluster {:?}", sizes.iter().max());
        // Deterministic.
        let again = cluster(&points, 128, &mut |_| {});
        assert_eq!(again.assign, c.assign);
        assert_eq!(again.centroids, c.centroids);
    }

    #[test]
    fn small_inputs_get_one_mean_centroid_and_no_spill() {
        let rows = test_corpus(2, 100, 16, 4);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let points = Points { rows: &refs, unit: false, pre: None, dim: 16 };
        let c = cluster(&points, 384, &mut |_| {});
        assert_eq!(c.len(), 1);
        assert!(c.spill.iter().all(|&s| s == NO_SPILL));
        for d in 0..16 {
            let mean = rows.iter().map(|r| r[d] as f64).sum::<f64>() / 100.0;
            assert!((c.centroids[d] as f64 - mean).abs() < 1e-5);
        }
        assert!(cluster(&Points { rows: &[], unit: false, pre: None, dim: 16 }, 384, &mut |_| {}).centroids.is_empty());
    }

    #[test]
    fn soar_spills_to_a_different_centroid_minimizing_the_soar_loss() {
        let rows = test_corpus(3, 4000, 24, 16);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let points = Points { rows: &refs, unit: false, pre: None, dim: 24 };
        let c = cluster(&points, 100, &mut |_| {});
        let k = c.len();
        assert!(k > 1 && k <= CLUSTERS_PER_NEIGHBORHOOD);
        let mut spilled = 0;
        for i in 0..rows.len() {
            let (a, s) = (c.assign[i] as usize, c.spill[i]);
            if s == NO_SPILL {
                continue;
            }
            spilled += 1;
            assert_ne!(s as usize, a, "spill to the own centroid");
            let v = &rows[i];
            let own = centroid(&c.centroids, 24, a);
            let r: Vec<f32> = v.iter().zip(own).map(|(x, y)| x - y).collect();
            let rn = l2_sq(v, own);
            let loss = |o: usize| {
                let co = centroid(&c.centroids, 24, o);
                let proj: f32 = v.iter().zip(co).zip(&r).map(|((x, y), z)| (x - y) * z).sum();
                l2_sq(v, co) + SOAR_LAMBDA * proj * proj / rn
            };
            let best = (0..k).filter(|&o| o != a).map(loss).fold(f32::INFINITY, f32::min);
            assert!(loss(s as usize) <= best * 1.001 + 1e-5, "vector {i}: spill loss {} > best {best}", loss(s as usize));
        }
        assert!(spilled > rows.len() * 9 / 10, "nearly every vector spills ({spilled})");
    }

    #[test]
    fn preconditioner_is_an_orthogonal_rotation() {
        let p = Preconditioner::new(70);
        let mut r = Rng::new(9);
        let a: Vec<f32> = (0..70).map(|_| r.gaussian() as f32).collect();
        let b: Vec<f32> = (0..70).map(|_| r.gaussian() as f32).collect();
        let (mut ra, mut rb) = (Vec::new(), Vec::new());
        p.apply(&a, &mut ra);
        p.apply(&b, &mut rb);
        assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-3 * dot(&a, &a).sqrt() * dot(&b, &b).sqrt());
        assert!((l2_sq(&a, &b) - l2_sq(&ra, &rb)).abs() < 1e-3 * l2_sq(&a, &b));
        let mut bytes = Vec::new();
        p.encode(&mut bytes);
        assert_eq!(Preconditioner::decode(&bytes, 70).unwrap(), p);
        assert!(Preconditioner::decode(&bytes, 71).is_err());
    }



    #[test]
    fn preconditioner_rejects_forged_allocation_counts_and_nonfinite_matrix() {
        let one_block = 1u32.to_le_bytes();
        assert!(matches!(Preconditioner::decode(&one_block, u32::MAX as usize), Err(RusticError::Corrupted(_))));
        let mut bytes = Vec::new();
        Preconditioner::new(1).encode(&mut bytes);
        let matrix_at = 12;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            bytes[matrix_at..matrix_at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(matches!(Preconditioner::decode(&bytes, 1), Err(RusticError::Corrupted(_))));
        }
        bytes[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(Preconditioner::decode(&bytes, 1), Err(RusticError::Corrupted(_))));
    }

    #[test]
    fn one_bit_raw_dot_is_the_code_dot_product() {
        let mut r = Rng::new(11);
        for dim in [1usize, 63, 64, 100, 128, 200, 768, 1536, 4096] {
            for _ in 0..20 {
                let doc: Vec<u8> = (0..dim).map(|_| r.below(2) as u8).collect();
                let query: Vec<u8> = (0..dim).map(|_| r.below(16) as u8).collect();
                let mut code = Vec::new();
                pack_codes(1, &doc, dim, &mut code);
                let mut planes = Vec::new();
                query_planes(&query, dim, &mut planes);
                let want: u32 = doc.iter().zip(&query).map(|(&d, &q)| u32::from(d) * u32::from(q)).sum();
                assert_eq!(raw_dot(1, &code, &planes, &query, dim), want, "dim {dim}");
            }
        }
    }

    #[test]
    fn dynamic_visit_ratio_matches_es() {
        // `computeDynamicVisitRatio(numCands, k)` values.
        assert!((dynamic_visit_ratio(10, 10) - (0.003 + 0.037 * 0.15 * (11f64.ln() / 10_001f64.ln())) as f32).abs() < 1e-7);
        assert!(dynamic_visit_ratio(10_000, 1) <= 0.04 + 1e-7);
        assert!(dynamic_visit_ratio(100, 10) > dynamic_visit_ratio(15, 10));
        assert!((segment_size_cap(1_000_000) - 0.045).abs() < 1e-6);
        assert_eq!(segment_size_cap(10), 1.0);
        assert_eq!(leaf_budget(3, 1.5), 9);
        assert_eq!(leaf_budget(10, 0.0), 20);
    }
}

/// The IVF correctness fixtures share the production process-wide finite
/// admission budget. Serialize independent fixture lifetimes so simultaneous
/// 20k-row training workspaces do not test accidental harness contention.
/// Worker threads inside a fixture (including swap/search races) still run.
#[cfg(test)]
pub(crate) fn fixture_admission_lock() -> std::sync::MutexGuard<'static, ()> {
    static FIXTURES: std::sync::Mutex<()> = std::sync::Mutex::new(());
    FIXTURES.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}
