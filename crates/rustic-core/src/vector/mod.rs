//! Dense-vector search for Rustic.
//!
//! This module implements the pieces behind an Elasticsearch-style
//! `dense_vector` / `knn` field:
//!
//! * [`distance`] — SIMD-friendly dot / L2 / cosine primitives.
//! * [`flat`] — an on-disk flat (brute-force) vector column.
//! * [`hnsw`] — shared graph algorithms and search kernels.
//! * `mutable_ingest` — the mutable field graph and stable vector identities.
//! * [`quantize`] — scalar and binary scoring primitives.
//! * [`ivf`] / [`ivf_field`] — DiskBBQ (`bbq_disk`): the shared per-field IVF index.
//! * [`segment`] — durable vector payload chunks; these never build ANN graphs.
//! * [`store`] — point ownership, payload compaction and query dispatch.
//! * [`fusion`] — RRF and linear fusion for hybrid search.
//!
//! Scores follow Elasticsearch conventions: **higher is better**.
//! Distances (used internally by the graph) are **lower is better**.

pub mod distance;
pub mod es_quant;
pub mod exact_schedule;
pub mod exact_shared;
pub mod vespa_policy;
pub mod flat;
pub mod fusion;
pub mod gpu;
pub mod hnsw;
pub mod ivf;
pub mod ivf_field;
mod memory;
pub mod int4;
pub mod osq;
pub mod packed;
pub mod quantize;
pub mod segment;
pub mod store;
#[cfg(feature = "mutable-hnsw-poc")]
pub mod mutable_ingest;
// Index-level vector tests exercise the mutable HNSW owner, the only
// supported ANN backend (a default feature).
#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod tests;
#[cfg(test)]
mod ivf_index_tests;

pub use distance::{cosine, dot, l2_sq, norm};

/// Bytes of the optional vector column copies held now (bounded by
/// `RUSTIC_VECTOR_COLUMN_CACHE_BYTES`).
pub fn column_cache_bytes() -> usize {
    memory::column_budget().used()
}

/// Whether `error` is the mutable graph's retirement (or retired-adjacency)
/// budget being exhausted: old graph generations pinned by long-lived
/// readers (PITs) hold reclamation back. Only reachable with explicit caps
/// (the default limits are uncapped); background repair defers on it.
pub fn is_retirement_backpressure(error: &crate::error::RusticError) -> bool {
    matches!(error, crate::error::RusticError::Segment(message)
        if message.contains("retirement budget exhausted") || message.contains("retired adjacency budget exhausted"))
}
pub use flat::{FlatVectors, FlatVectorsBuilder};

/// Retained views (PIT, scroll) whose mutable-graph pin was released to
/// bound retired memory; they serve kNN by exact search since.
pub fn degraded_retained_views() -> u64 {
    #[cfg(feature = "mutable-hnsw-poc")]
    { mutable_ingest::degraded_retained_views() }
    #[cfg(not(feature = "mutable-hnsw-poc"))]
    { 0 }
}
pub use hnsw::{FilterMode, HnswBuilder, HnswGraph, HnswParams, VectorSource};
pub use int4::{Int4Vectors, UniformQuantizer};
pub use quantize::{Int8Vectors, QuantQuery, Quantization, QuantizedVectors, ScalarQuantizer};
pub use segment::VectorSegment;
pub use store::{DocFilter, OptimizeReport, OptimizerBacklog, OptimizerPolicy, PlannerConfig, PointBits, PointFilter, PointHit, PointId, SearchStats, VectorStore};

/// Vector similarity metric, matching Elasticsearch `dense_vector.similarity`.
///
/// [`Similarity::score`] returns an ES-style score where **higher is better**;
/// [`Similarity::distance`] returns the corresponding distance where
/// **lower is better** (this is what the HNSW graph minimizes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Similarity {
    /// Cosine similarity. Score: `(1 + cos) / 2`. Distance: `1 - cos`.
    Cosine,
    /// Dot product. Score: `(1 + dot) / 2` (ES does not clamp the raw dot).
    /// Distance: `-dot`.
    DotProduct,
    /// Squared Euclidean distance. Score: `1 / (1 + l2²)`. Distance: `l2²`.
    L2,
    /// Lucene `MAXIMUM_INNER_PRODUCT`: unnormalised dot product. Score:
    /// `dot >= 0 ? dot + 1 : 1 / (1 - dot)`. Distance: `-dot`.
    MaxInnerProduct,
    // Packed rows ([`packed`]): ES `element_type: bit` / `byte` vectors
    // stored as their bytes (four per `f32` lane, never read as floats).
    // Scores and distances equal the unpacked forms' (a bit vector as one
    // 0/1 value per bit under `L2`, a byte vector as its integer values).
    /// `bit` rows: Hamming distance. Score: `1 / (1 + hamming)` (`L2` of
    /// the 0/1 expansion).
    BitHamming,
    /// `byte` rows, [`Cosine`](Self::Cosine) of their integer values.
    ByteCosine,
    /// `byte` rows, [`DotProduct`](Self::DotProduct) of their integer values.
    ByteDotProduct,
    /// `byte` rows, [`L2`](Self::L2) of their integer values.
    ByteL2,
    /// `byte` rows, [`MaxInnerProduct`](Self::MaxInnerProduct) of their
    /// integer values.
    ByteMaxInnerProduct,
}

/// Internal distance family of a [`Similarity`]: what the kernels compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    /// `l2_sq(a, b)`.
    L2,
    /// `-dot(a, b)`.
    Dot,
    /// `1 - cos(a, b)`.
    Cos,
    /// Packed rows: only [`Similarity::distance`] / [`Similarity::score`]
    /// read them (no float kernel may).
    Packed,
}

impl Metric {
    /// The distance is derived from an `f32` dot product (`dot` / `dot_four`).
    #[inline]
    pub(crate) fn uses_dot(self) -> bool {
        matches!(self, Metric::Dot | Metric::Cos)
    }
}

impl Similarity {
    pub(crate) fn metric(self) -> Metric {
        match self {
            Similarity::Cosine => Metric::Cos,
            Similarity::L2 => Metric::L2,
            Similarity::DotProduct | Similarity::MaxInnerProduct => Metric::Dot,
            _ => Metric::Packed,
        }
    }

    /// Rows of this similarity are packed ([`packed`]): `bit` or `byte`.
    #[inline]
    pub fn is_packed(self) -> bool {
        matches!(self, Similarity::BitHamming | Similarity::ByteCosine | Similarity::ByteDotProduct
            | Similarity::ByteL2 | Similarity::ByteMaxInnerProduct)
    }

    /// Whether `row` holds values this similarity can score: finite floats,
    /// or (packed rows) any bytes.
    #[inline]
    pub fn valid_row(self, row: &[f32]) -> bool {
        self.is_packed() || row.iter().all(|x| x.is_finite())
    }

    /// The float similarity whose score / distance formulas this one uses
    /// (itself for float similarities; `L2` for Hamming).
    #[inline]
    pub fn unpacked(self) -> Similarity {
        match self {
            Similarity::BitHamming | Similarity::ByteL2 => Similarity::L2,
            Similarity::ByteCosine => Similarity::Cosine,
            Similarity::ByteDotProduct => Similarity::DotProduct,
            Similarity::ByteMaxInnerProduct => Similarity::MaxInnerProduct,
            s => s,
        }
    }

    /// ES-style score for the pair `a` / `b` (higher is better).
    ///
    /// * Cosine: `(1 + cos(a, b)) / 2`
    /// * DotProduct: `(1 + dot(a, b)) / 2`
    /// * L2: `1 / (1 + l2_sq(a, b))`
    ///
    /// `a` and `b` must have the same length.
    pub fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.is_packed() {
            return packed::score(*self, a, b);
        }
        match self {
            Similarity::Cosine => (1.0 + distance::cosine(a, b)) / 2.0,
            Similarity::DotProduct => (1.0 + distance::dot(a, b)) / 2.0,
            Similarity::L2 => 1.0 / (1.0 + distance::l2_sq(a, b)),
            Similarity::MaxInnerProduct => {
                let d = distance::dot(a, b);
                if d >= 0.0 { d + 1.0 } else { 1.0 / (1.0 - d) }
            }
            _ => unreachable!("packed similarity"),
        }
    }

    /// Internal distance for the pair `a` / `b` (lower is better).
    ///
    /// * Cosine: `1 - cos(a, b)`
    /// * DotProduct: `-dot(a, b)`
    /// * L2: `l2_sq(a, b)`
    ///
    /// `a` and `b` must have the same length.
    /// The score of a [`distance`](Self::distance) value (inverse of the
    /// `score` / `distance` relation), used to turn a quantized distance
    /// estimate into the score a non-rescored quantized search returns.
    pub fn score_of_distance(&self, d: f32) -> f32 {
        match self.unpacked() {
            Similarity::Cosine => (2.0 - d) / 2.0,
            Similarity::DotProduct => (1.0 - d) / 2.0,
            Similarity::L2 => 1.0 / (1.0 + d),
            Similarity::MaxInnerProduct => {
                let dot = -d;
                if dot >= 0.0 { dot + 1.0 } else { 1.0 / (1.0 - dot) }
            }
            _ => unreachable!("unpacked similarity"),
        }
    }

    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Similarity::Cosine => 1.0 - distance::cosine(a, b),
            Similarity::DotProduct => -distance::dot(a, b),
            Similarity::L2 => distance::l2_sq(a, b),
            Similarity::MaxInnerProduct => -distance::dot(a, b),
            _ => packed::distance(*self, a, b),
        }
    }
}
