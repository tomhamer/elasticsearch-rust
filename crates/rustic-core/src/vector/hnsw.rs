//! HNSW (Hierarchical Navigable Small World) graph, per Malkov & Yashunin.
//!
//! # Build
//!
//! The graph is built once over a [`VectorSource`] at segment seal and is then
//! immutable. The builder works on dense ordinals with hnswlib-style flat
//! storage: fixed-capacity neighbor arrays (`2m` on layer 0, `m` above) held
//! in `AtomicU32` slots with the neighbor's distance kept next to it, a
//! per-node `parking_lot::Mutex` serializing writers, a reusable
//! generation-stamped visited array and reusable heaps (thread-local), and
//! SIMD distances straight off the contiguous `f32` column (cosine uses
//! precomputed inverse norms). With `HnswParams::threads != 1` the
//! insertions run in parallel on rayon (hnswlib / Lucene concurrent-merge
//! style: readers see neighbor lists lock-free, writers lock the node they
//! modify, the entry point is updated under a global lock); `threads == 1`
//! is fully deterministic for a seed.
//!
//! Neighbor selection: a new node keeps the diverse candidates of its
//! `ef_construction` beam (the HNSW heuristic, `extendCandidates = false`,
//! as in Lucene / hnswlib). When a back-link overflows a full list, the
//! stored distances are reused and Lucene's `findWorstNonDiverse` drops one
//! neighbor (the farthest one that is closer to another kept neighbor than
//! to the node, else the farthest), so lists stay full instead of being
//! re-pruned from scratch.
//!
//! # Search
//!
//! A greedy descent through the upper layers followed by an `ef`-bounded beam
//! search on layer 0, with Lucene-style filtered search: docs rejected by
//! `accept` are still traversed for connectivity but never appear in the
//! results. Candidates are packed `(ordered distance bits, doc)` `u64`s so
//! every heap comparison is one integer compare. Search can also score with
//! a [`QuantizedVectors`] index (int8 or 1-bit) and rescore the best
//! `k * oversample` with exact `f32` ([`HnswGraph::search_quantized`]).
//!
//! Wire format (unchanged since v1; all integers little-endian):
//!
//! ```text
//! u32 num_layers
//! u32 entry            // entry-point doc id, or u32::MAX if the graph is empty
//! per layer:
//!   u32 num_nodes
//!   u32 max_m
//!   [u32 node ids present, ascending]
//!   [u32 csr offsets, length num_nodes + 1]
//!   [u32 neighbors]
//! ```

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering::{Acquire, Relaxed, Release}};

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::DocId;
use crate::error::{Result, RusticError};

use super::distance::{dot, dot_four, dot_and_sq_norm, l2_sq, norm, prefetch};
use super::quantize::{QuantQuery, QuantizedVectors};
use super::{Metric, Similarity};

#[cfg(feature = "mutable-hnsw-poc")]
mod mutable;
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) use mutable::{beam_workers, SharedQuantized};
pub use mutable::{PreparedNodeFilter, PendingMutableCheckpoint, RawRows, RawRowsSnapshot, CapturedMutableVectorRead, MutableVectorCheckpoint, MutableVectorOwner, MutableVectorRead, MutableCommitBatch, InsertReservation, MutableArrayMemory, MutableStorageMemory, MutableHnsw, MutableHnswLimits, MutableHnswSnapshot, MutableHnswStats, MutableSearchMode, MutableSearchResult, MutableQueryScorer, MutableVectorSource, NodeHandle, PreparedMutableInsert, ReclaimedNode, VespaFilterEstimate, VespaFilterFirstParams, VespaPlainParams};

/// Native POC instrumentation for the existing immutable search baseline.
#[cfg(feature = "mutable-hnsw-poc")]
pub fn poc_vector_ops() -> u64 { vector_ops() }

/// Source of vectors for building and searching an HNSW graph.
///
/// Docs without a vector return `None` from [`get`](VectorSource::get) and
/// are absent from the graph.
pub trait VectorSource {
    /// Vector dimensionality.
    fn dim(&self) -> u32;
    /// Total number of docs (including docs without vectors).
    fn num_docs(&self) -> u32;
    /// The vector for `doc`, if it has one.
    fn get(&self, doc: DocId) -> Option<&[f32]>;
    /// Every doc's vector as one dense row-major slice of
    /// `num_docs * dim` floats, when the source is laid out that way and
    /// every doc has a vector. Enables the zero-dispatch SIMD fast paths;
    /// the default (`None`) falls back to [`get`](VectorSource::get).
    fn as_contiguous(&self) -> Option<&[f32]> {
        None
    }
    /// Cached squared norms for a dense immutable column. Optional: sources
    /// without a cache retain the fused dot/norm distance kernel.
    fn squared_norms(&self) -> Option<&[f32]> { None }
    /// Optional unit-length copy of the contiguous column for cosine
    /// traversal. Zero-norm rows must be all zero. Original rows returned
    /// by `get` remain the source of final scores and fetched vectors.
    fn normalized_contiguous(&self) -> Option<&[f32]> { None }
}

/// Build parameters for [`HnswBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HnswParams {
    /// Max neighbors per node on upper levels (layer 0 uses `2 * m`).
    pub m: usize,
    /// Candidate list size during construction.
    pub ef_construction: usize,
    /// Seed for the deterministic level assignment.
    pub seed: u64,
    /// Insertion threads: `1` = sequential and deterministic (default),
    /// `0` = the current rayon pool, `n` = a dedicated pool of `n` threads.
    pub threads: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            ef_construction: 100,
            seed: 0x9E3779B97F4A7C15,
            threads: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Packed (distance, id) keys: one u64 compare per heap operation.

/// Order-preserving map of an `f32` to `u32` (total order, NaN last-ish).
#[inline(always)]
fn fkey(d: f32) -> u32 {
    let b = d.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

#[inline(always)]
fn funkey(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7fff_ffff } else { !k })
}

#[inline(always)]
pub(crate) fn pack(d: f32, id: u32) -> u64 {
    ((fkey(d) as u64) << 32) | id as u64
}

#[inline(always)]
pub(crate) fn unpack(x: u64) -> (f32, u32) {
    (funkey((x >> 32) as u32), x as u32)
}

/// How the layer-0 beam of a filtered search treats nodes the filter
/// rejects.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FilterMode {
    /// Rejected nodes are scored and traversed for connectivity, just not
    /// returned (broad filters).
    #[default]
    Plain,
    /// A segment-owned payload graph navigates accepted whole-value groups.
    Payload,
    /// Two-hop (ACORN-1, Lucene's `FilteredHnswGraphSearcher`): rejected
    /// nodes are not scored; their own neighbours are examined instead, so
    /// the distance computations stay close to an unfiltered search.
    /// `memo`: remember each rejection in the visited marks so a node is
    /// filter-checked at most once. Worth it when the filter check is more
    /// expensive than a visited lookup (owner-doc lookups); for a bitset
    /// over node ids the check comes first and nothing is memoised.
    TwoHop { memo: bool },
}

impl FilterMode {
    #[inline(always)]
    fn two_hop(self) -> bool {
        matches!(self, FilterMode::TwoHop { .. })
    }
}

// ---------------------------------------------------------------------------
// Reusable per-thread search state.

/// Generation-stamped visited set: reset is O(1) (bump the generation).
#[derive(Default)]
struct Visited {
    marks: Vec<u16>,
    generation: u16,
}

impl Visited {
    fn reset(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        // Generations are even; `generation + 1` marks a node the filter
        // rejected (two-hop search), so one load answers both questions.
        self.generation = self.generation.wrapping_add(2);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 2;
        }
    }

    /// The mark of `i`: 0 = unseen, 1 = visited, 2 = rejected (seen by the
    /// filter but not visited); out of range reads as visited.
    #[inline(always)]
    fn state(&self, i: u32) -> u8 {
        match self.marks.get(i as usize) {
            Some(&m) if m == self.generation => 1,
            Some(&m) if m == self.generation.wrapping_add(1) => 2,
            Some(_) => 0,
            None => 1,
        }
    }

    /// Mark `i` rejected (in range only).
    #[inline(always)]
    fn reject(&mut self, i: u32) {
        let g = self.generation.wrapping_add(1);
        if let Some(m) = self.marks.get_mut(i as usize) {
            *m = g;
        }
    }

    /// Mark `i`; returns `false` if it was already marked.
    #[inline(always)]
    fn insert(&mut self, i: u32) -> bool {
        match self.marks.get_mut(i as usize) {
            Some(m) if *m != self.generation => {
                *m = self.generation;
                true
            }
            _ => false,
        }
    }
}

#[derive(Default)]
struct Scratch {
    visited: Visited,
    cand: BinaryHeap<Reverse<u64>>,
    top: BinaryHeap<u64>,
    out: Vec<u64>,
    sel: Vec<u64>,
    tmp: Vec<u64>,
    nbuf: Vec<u32>,
    /// Rejected neighbours awaiting two-hop expansion.
    rbuf: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
    /// Vector distance / similarity computations made by searches on this
    /// thread (monotonic; see [`vector_ops`]).
    static VECTOR_OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Vector distance / similarity computations searches have made on this
/// thread so far (graph traversal, brute force, rescoring). Monotonic: a
/// search reads it before and after to count its own work (ES profile
/// `vector_operations_count`).
pub(crate) fn vector_ops() -> u64 {
    VECTOR_OPS.with(|c| c.get())
}

/// Count `n` vector computations on this thread (see [`vector_ops`]).
#[inline]
pub(crate) fn add_vector_ops(n: u64) {
    VECTOR_OPS.with(|c| c.set(c.get().wrapping_add(n)));
}

#[cfg(test)]
thread_local! {
    /// Test-only search counters: (nodes visited, neighbours rejected by the
    /// filter or accept predicate), and the route of the last mutable kNN.
    static SEARCH_COUNTS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    static LAST_ROUTE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Count `n` nodes a graph search visited (test builds only; free otherwise).
#[inline(always)]
pub(crate) fn note_visited(n: u64) {
    #[cfg(test)]
    SEARCH_COUNTS.with(|c| { let (v, r) = c.get(); c.set((v + n, r)); });
    #[cfg(not(test))]
    let _ = n;
}

/// Count `n` neighbours a graph search rejected by its filter (test builds only).
#[inline(always)]
pub(crate) fn note_rejected(n: u64) {
    #[cfg(test)]
    SEARCH_COUNTS.with(|c| { let (v, r) = c.get(); c.set((v, r + n)); });
    #[cfg(not(test))]
    let _ = n;
}

/// (visited, rejected) on this thread so far (monotonic; test builds).
#[cfg(test)]
pub(crate) fn search_counts() -> (u64, u64) {
    SEARCH_COUNTS.with(|c| c.get())
}

/// Record the route the last mutable kNN on this thread took (test builds).
#[inline(always)]
pub(crate) fn note_route(route: &'static str) {
    #[cfg(test)]
    LAST_ROUTE.with(|c| c.set(route));
    #[cfg(not(test))]
    let _ = route;
}

/// The route of the last mutable kNN on this thread (test builds).
#[cfg(test)]
pub(crate) fn last_route() -> &'static str {
    LAST_ROUTE.with(|c| c.get())
}

/// Run `f` with this thread's scratch (a fresh one if it is already in use).
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|s| match s.try_borrow_mut() {
        Ok(mut s) => f(&mut s),
        Err(_) => f(&mut Scratch::default()),
    })
}

// ---------------------------------------------------------------------------
// Distance providers for search.

/// Distance from a fixed query to graph node `id` (lower is better).
trait Dist {
    fn dist(&self, id: u32) -> f32;
    #[inline(always)]
    fn four(&self, ids: [u32; 4]) -> [f32; 4] { ids.map(|id| self.dist(id)) }
    #[inline(always)]
    fn prefetch(&self, _id: u32) {}
}

/// Contiguous `f32` column, SIMD kernels, no dynamic dispatch.
struct ContigDist<'a> {
    data: &'a [f32],
    dim: usize,
    q: &'a [f32],
    metric: Metric,
    /// The field's similarity (packed rows score through it).
    similarity: Similarity,
    inv_qn: f32,
    squared_norms: Option<&'a [f32]>,
}

impl Dist for ContigDist<'_> {
    #[inline(always)]
    fn four(&self, ids: [u32; 4]) -> [f32; 4] {
        if self.metric != Metric::Dot { return ids.map(|id| self.dist(id)); }
        let rows = ids.map(|id| {
            let start = id as usize * self.dim;
            self.data.get(start..start + self.dim)
        });
        let [Some(a), Some(b), Some(c), Some(d)] = rows else { return ids.map(|id| self.dist(id)); };
        dot_four(self.q, [a, b, c, d]).map(|v| -v)
    }
    #[inline(always)]
    fn dist(&self, id: u32) -> f32 {
        let s = id as usize * self.dim;
        let Some(v) = self.data.get(s..s + self.dim) else {
            return f32::INFINITY;
        };
        match self.metric {
            Metric::L2 => l2_sq(self.q, v),
            Metric::Dot => -dot(self.q, v),
            Metric::Cos => {
                let (d, vv) = match self.squared_norms {
                    Some(norms) => (dot(self.q, v), norms[id as usize]),
                    None => dot_and_sq_norm(self.q, v),
                };
                if vv == 0.0 || self.inv_qn == 0.0 { 1.0 } else { 1.0 - d * self.inv_qn / vv.sqrt() }
            }
            Metric::Packed => self.similarity.distance(self.q, v),
        }
    }
    #[inline(always)]
    fn prefetch(&self, id: u32) {
        let s = id as usize * self.dim;
        if s < self.data.len() {
            prefetch(self.data[s..].as_ptr() as *const u8, self.dim * 4);
        }
    }
}

/// Any [`VectorSource`] through `get`.
struct DynDist<'a> {
    vectors: &'a dyn VectorSource,
    q: &'a [f32],
    similarity: Similarity,
}

impl Dist for DynDist<'_> {
    #[inline]
    fn dist(&self, id: u32) -> f32 {
        match self.vectors.get(id) {
            Some(v) => self.similarity.distance(self.q, v),
            None => f32::INFINITY,
        }
    }
}

/// Quantized scoring.
struct QuantDist<'a> {
    qv: &'a QuantizedVectors,
    pq: &'a QuantQuery,
}

impl Dist for QuantDist<'_> {
    #[inline(always)]
    fn dist(&self, id: u32) -> f32 {
        if (id as usize) < self.qv.len() { self.qv.distance(self.pq, id) } else { f32::INFINITY }
    }
    #[inline(always)]
    fn prefetch(&self, id: u32) {
        if (id as usize) < self.qv.len() {
            self.qv.prefetch(id);
        }
    }
}

// ---------------------------------------------------------------------------
// Immutable graph.

/// One CSR layer of the graph. Node ids are doc ids; only docs with vectors
/// are present.
#[derive(Clone, Debug)]
struct Layer {
    /// Ascending doc ids present in this layer.
    nodes: Vec<DocId>,
    /// CSR offsets into `neighbors`, length `nodes.len() + 1`.
    offsets: Vec<u32>,
    /// Neighbor doc ids.
    neighbors: Vec<DocId>,
    /// Max neighbors for this layer (`2 * m` on layer 0, `m` above).
    max_m: usize,
    /// `nodes == 0..nodes.len()`: a node's row is its id (no binary search).
    dense: bool,
}

impl Layer {
    fn new(nodes: Vec<DocId>, offsets: Vec<u32>, neighbors: Vec<DocId>, max_m: usize) -> Layer {
        let dense = nodes.last().is_none_or(|&l| l as usize + 1 == nodes.len());
        Layer { nodes, offsets, neighbors, max_m, dense }
    }

    /// Row index of `node`, if present.
    #[inline(always)]
    fn row(&self, node: DocId) -> Option<usize> {
        if self.dense {
            ((node as usize) < self.nodes.len()).then_some(node as usize)
        } else {
            self.nodes.binary_search(&node).ok()
        }
    }

    /// Prefetch `node`'s CSR offsets (the first dependent load of
    /// [`Layer::neighbors_of`]).
    #[inline(always)]
    fn prefetch_offsets(&self, node: DocId) {
        if let Some(i) = self.row(node) {
            prefetch(self.offsets[i..].as_ptr() as *const u8, 8);
        }
    }

    /// Prefetch `node`'s neighbour list (its offsets should be cached).
    #[inline(always)]
    fn prefetch_neighbors(&self, node: DocId) {
        if let Some(i) = self.row(node) {
            let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
            if a < b && b <= self.neighbors.len() {
                prefetch(self.neighbors[a..].as_ptr() as *const u8, (b - a) * 4);
            }
        }
    }

    #[inline(always)]
    fn neighbors_of(&self, node: DocId) -> &[DocId] {
        let i = if self.dense {
            if node as usize >= self.nodes.len() {
                return &[];
            }
            node as usize
        } else {
            match self.nodes.binary_search(&node) {
                Ok(i) => i,
                Err(_) => return &[],
            }
        };
        &self.neighbors[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

/// An immutable HNSW graph.
#[derive(Clone)]
pub struct HnswGraph {
    layers: Vec<Layer>,
    entry: Option<DocId>,
    /// One past the largest doc id referenced (visited-set size).
    id_bound: u32,
    payload: Option<Box<HnswGraph>>,
    payload_parts: Vec<HnswGraph>,
    payload_masks: std::sync::OnceLock<Vec<Vec<u64>>>,
}

fn id_bound(layers: &[Layer]) -> u32 {
    layers
        .iter()
        .flat_map(|l| l.nodes.last().into_iter().chain(l.neighbors.iter().max()))
        .map(|&x| x.saturating_add(1))
        .max()
        .unwrap_or(0)
}

#[cfg(feature = "payload-hnsw-poc")]
fn payload_seed_graphs() -> &'static [HnswGraph] {
    static PARTS: std::sync::OnceLock<Vec<HnswGraph>> = std::sync::OnceLock::new();
    PARTS.get_or_init(|| {
        let path = format!("{}.parts.json", std::env::var("RUSTIC_POC_PAYLOAD_GRAPH").unwrap());
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        manifest.as_array().unwrap().iter().map(|part| {
            HnswGraph::open(&std::fs::read(part["graph"].as_str().unwrap()).unwrap()).unwrap()
        }).collect()
    })
}

impl HnswGraph {
    fn payload_seed_parts(&self) -> &[HnswGraph] {
        if let Some(overlay) = &self.payload { return &overlay.payload_parts; }
        if !self.payload_parts.is_empty() { return &self.payload_parts; }
        #[cfg(feature = "payload-hnsw-poc")]
        if std::env::var_os("RUSTIC_POC_PAYLOAD_GRAPH").is_some() { return payload_seed_graphs(); }
        &[]
    }

    /// Attach immutable navigation built from this segment's own ordinals.
    pub fn with_payload_partitions(mut self, partitions: &[HnswGraph]) -> Self {
        if !partitions.is_empty() {
            let mut overlay = self.with_payload_edges(partitions);
            overlay.payload = None;
            overlay.payload_masks = std::sync::OnceLock::new();
            overlay.strip_base_edges(&self);
            overlay.payload_parts = partitions.iter().map(|p| {
                let mut upper = p.clone();
                upper.payload = None;
                upper.payload_parts.clear();
                upper.layers[0].neighbors = Vec::new();
                upper.layers[0].offsets.fill(0);
                upper
            }).collect();
            overlay.payload_word_masks();
            self.payload = Some(Box::new(overlay));
        }
        self
    }

    /// The base adjacency is shared by both search modes. Persist only the
    /// extra layer-zero edges; payload seeds carry their own upper navigation.
    fn strip_base_edges(&mut self, base: &Self) {
        let layer = &self.layers[0];
        let mut offsets = Vec::with_capacity(layer.nodes.len() + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for &node in &layer.nodes {
            let existing = base.layers[0].neighbors_of(node);
            neighbors.extend(layer.neighbors_of(node).iter().copied().filter(|nb| !existing.contains(nb)));
            offsets.push(neighbors.len() as u32);
        }
        let max_degree = offsets.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
        self.layers = vec![Layer::new(layer.nodes.clone(), offsets, neighbors, max_degree)];
    }

    fn payload_word_masks(&self) -> &[Vec<u64>] {
        self.payload_masks.get_or_init(|| self.payload_parts.iter().map(|part| {
            let mut words = vec![0; (self.id_bound as usize).div_ceil(64)];
            for &id in &part.layers[0].nodes { words[id as usize / 64] |= 1u64 << (id % 64); }
            words
        }).collect())
    }

    pub fn payload_covers_words(&self, accepted: &[u64]) -> bool {
        let Some(overlay) = &self.payload else { return false };
        let masks = overlay.payload_word_masks();
        let mut covered = vec![0u64; (self.id_bound as usize).div_ceil(64)];
        for members in masks {
            if members.iter().enumerate().all(|(i, &m)| m & !accepted.get(i).copied().unwrap_or(0) == 0) {
                for (c, &m) in covered.iter_mut().zip(members) { *c |= m; }
            }
        }
        accepted.iter().enumerate().all(|(i, &w)| w & !covered.get(i).copied().unwrap_or(0) == 0)
    }

    /// Whole-value filters only. Partial groups (including deletes and
    /// additional security constraints) use the ordinary graph instead.
    pub fn payload_covers(&self, accept: &impl Fn(DocId) -> bool) -> bool {
        let Some(overlay) = &self.payload else { return false };
        let mut covered = vec![false; self.id_bound as usize];
        for part in &overlay.payload_parts {
            let nodes = &part.layers[0].nodes;
            if nodes.iter().all(|&id| accept(id)) {
                for &id in nodes { covered[id as usize] = true; }
            }
        }
        self.layers[0].nodes.iter().all(|&id| !accept(id) || covered[id as usize])
    }

    fn payload_overlay(&self, mode: FilterMode) -> &Self {
        if mode != FilterMode::Payload { return self }
        if self.payload.is_some() || !self.payload_parts.is_empty() { return self }
        #[cfg(not(feature = "payload-hnsw-poc"))]
        return self;
        #[cfg(feature = "payload-hnsw-poc")]
        {
        static OVERLAY: std::sync::OnceLock<HnswGraph> = std::sync::OnceLock::new();
        OVERLAY.get_or_init(|| {
            let path = std::env::var("RUSTIC_POC_PAYLOAD_GRAPH").expect("POC graph file required");
            let overlay = HnswGraph::open(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(overlay.layers[0].nodes, self.layers[0].nodes, "POC overlay node mismatch");
            overlay
        })
        }
    }

    /// Entry routing uses only upper layers, not a second copy of every
    /// payload value's layer-zero edges (already merged into the overlay).
    pub fn encode_upper_navigation(&self) -> Vec<u8> {
        let mut out = self.clone();
        if let Some(layer) = out.layers.first_mut() {
            layer.neighbors.clear(); layer.offsets.fill(0);
        }
        out.encode()
    }

    /// Merge payload-value graphs at every navigation layer.
    pub fn with_payload_edges(&self, partitions: &[HnswGraph]) -> Self {
        let mut out = self.clone();
        let base = &self.layers[0];
        let levels = partitions.iter().map(|g| g.layers.len()).max().unwrap_or(0).max(self.layers.len());
        out.layers.clear();
        for level in 0..levels {
            let source = self.layers.get(level);
            let mut nodes: std::collections::BTreeSet<u32> = source.into_iter().flat_map(|l| l.nodes.iter().copied()).collect();
            for partition in partitions {
                if let Some(layer) = partition.layers.get(level) { nodes.extend(&layer.nodes); }
            }
            let nodes: Vec<u32> = nodes.into_iter().collect();
            let mut offsets = Vec::with_capacity(nodes.len() + 1);
            let mut neighbors = Vec::new();
            let mut max_degree = source.map_or(0, |l| l.max_m);
            offsets.push(0);
            for &node in &nodes {
                assert!(base.row(node).is_some(), "payload node outside base graph");
                let mut row = source.map_or_else(Vec::new, |l| l.neighbors_of(node).to_vec());
                for partition in partitions {
                    let Some(layer) = partition.layers.get(level) else { continue };
                    for &nb in layer.neighbors_of(node) {
                        assert!(base.row(nb).is_some(), "payload edge outside base graph");
                        if nb != node && !row.contains(&nb) { row.push(nb); }
                    }
                }
                max_degree = max_degree.max(row.len());
                neighbors.extend(row);
                offsets.push(u32::try_from(neighbors.len()).expect("payload graph too large"));
            }
            out.layers.push(Layer::new(nodes, offsets, neighbors, max_degree));
        }
        if levels > self.layers.len() {
            out.entry = partitions.iter().find(|g| g.layers.len() == levels).and_then(|g| g.entry);
        }
        out
    }

    /// Number of layers (0 for an empty graph).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Entry point (the node with the highest level), if any.
    pub fn entry(&self) -> Option<DocId> {
        self.entry
    }

    /// Number of nodes on layer 0 (= vectors in the graph).
    pub fn num_nodes(&self) -> usize {
        self.layers.first().map_or(0, |l| l.nodes.len())
    }

    /// Insert GPU batch anchors before their lower-level neighbors. This is
    /// only a construction order; stable source/point IDs keep original order.
    #[cfg(feature = "gpu")]
    pub(crate) fn mutable_insertion_order(&self) -> Vec<u32> {
        let mut order = self.layers.first().map_or_else(Vec::new, |layer| layer.nodes.clone());
        order.sort_by_key(|&id| std::cmp::Reverse(self.layers.iter().rposition(|layer| layer.row(id).is_some()).unwrap_or(0)));
        order
    }

    /// Mean out-degree on layer 0.
    pub fn mean_degree(&self) -> f64 {
        self.layers.first().map_or(0.0, |l| l.neighbors.len() as f64 / l.nodes.len().max(1) as f64)
    }

    /// Max neighbours per node on `level` (`2m` on layer 0, `m` above), if
    /// the graph has that layer.
    pub fn max_neighbors(&self, level: usize) -> Option<usize> {
        self.layers.get(level).map(|l| l.max_m)
    }

    /// Whether this graph was built with HNSW `m` (so it can seed
    /// [`HnswBuilder::build_from`] with the same parameter).
    pub fn built_with_m(&self, m: usize) -> bool {
        self.layers.iter().enumerate().all(|(l, layer)| layer.max_m == if l == 0 { 2 * m } else { m })
    }

    /// Serialize the graph.
    pub fn encode(&self) -> Vec<u8> {
        let words: usize = self.layers.iter().map(|l| 2 + 2 * l.nodes.len() + 1 + l.neighbors.len()).sum();
        let mut out = Vec::with_capacity(8 + 4 * words);
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.entry.unwrap_or(u32::MAX).to_le_bytes());
        for layer in &self.layers {
            out.extend_from_slice(&(layer.nodes.len() as u32).to_le_bytes());
            out.extend_from_slice(&(layer.max_m as u32).to_le_bytes());
            for &n in &layer.nodes {
                out.extend_from_slice(&n.to_le_bytes());
            }
            for &o in &layer.offsets {
                out.extend_from_slice(&o.to_le_bytes());
            }
            for &n in &layer.neighbors {
                out.extend_from_slice(&n.to_le_bytes());
            }
        }
        if self.payload.is_some() || !self.payload_parts.is_empty() {
            out.extend_from_slice(b"RPLD");
            let blobs: Vec<Vec<u8>> = self.payload.iter().map(|g| g.encode()).chain(self.payload_parts.iter().map(|g| g.encode())).collect();
            out.extend_from_slice(&(self.payload.is_some() as u32).to_le_bytes());
            out.extend_from_slice(&(self.payload_parts.len() as u32).to_le_bytes());
            for blob in blobs {
                out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
                out.extend_from_slice(&blob);
            }
        }
        out
    }

    /// Parse a graph produced by [`encode`](Self::encode).
    pub fn open(data: &[u8]) -> Result<HnswGraph> {
        Self::open_nested(data, 0)
    }

    fn open_nested(data: &[u8], depth: usize) -> Result<HnswGraph> {
        if depth > 2 { return Err(RusticError::Corrupted("hnsw: nested payload".into())); }
        let mut p = 0usize;
        let read_u32s = |p: &mut usize, n: usize| -> Result<Vec<u32>> {
            let end = n.checked_mul(4).and_then(|b| b.checked_add(*p)).filter(|&e| e <= data.len());
            let Some(end) = end else {
                return Err(RusticError::Corrupted("hnsw: truncated".into()));
            };
            let v = data[*p..end].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            *p = end;
            Ok(v)
        };
        let hdr = read_u32s(&mut p, 2)?;
        let (num_layers, entry_raw) = (hdr[0] as usize, hdr[1]);
        let entry = (entry_raw != u32::MAX).then_some(entry_raw);
        let mut layers = Vec::with_capacity(num_layers.min(64));
        for _ in 0..num_layers {
            let h = read_u32s(&mut p, 2)?;
            let (num_nodes, max_m) = (h[0] as usize, h[1] as usize);
            let nodes = read_u32s(&mut p, num_nodes)?;
            let offsets = read_u32s(&mut p, num_nodes + 1)?;
            let num_neighbors = *offsets.last().unwrap() as usize;
            let neighbors = read_u32s(&mut p, num_neighbors)?;
            if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] > w[1]) || nodes.windows(2).any(|w| w[0] >= w[1]) {
                return Err(RusticError::Corrupted("hnsw: inconsistent CSR".into()));
            }
            layers.push(Layer::new(nodes, offsets, neighbors, max_m));
        }
        let id_bound = id_bound(&layers);
        let mut graph = HnswGraph { layers, entry, id_bound, payload: None, payload_parts: Vec::new(), payload_masks: std::sync::OnceLock::new() };
        if data.get(p..p + 4) == Some(b"RPLD") {
            p += 4;
            let header = read_u32s(&mut p, 2)?;
            if header[0] > 1 || header[1] > 64 { return Err(RusticError::Corrupted("hnsw: payload header".into())); }
            for i in 0..header[0] + header[1] {
                let len = read_u32s(&mut p, 1)?[0] as usize;
                let end = p.checked_add(len).filter(|&e| e <= data.len()).ok_or_else(|| RusticError::Corrupted("hnsw: payload truncated".into()))?;
                let mut child = Self::open_nested(&data[p..end], depth + 1)?;
                p = end;
                if child.id_bound > id_bound { return Err(RusticError::Corrupted("hnsw: payload ordinals".into())); }
                if child.layers.is_empty() || child.entry.is_none()
                    || child.layers.iter().any(|layer| layer.nodes.iter().chain(&layer.neighbors)
                        .any(|&node| graph.layers.first().is_none_or(|base| base.row(node).is_none()))) {
                    return Err(RusticError::Corrupted("hnsw: payload membership".into()));
                }
                if i == 0 && header[0] == 1 {
                    if child.layers.first().map(|l| &l.nodes) != graph.layers.first().map(|l| &l.nodes) { return Err(RusticError::Corrupted("hnsw: payload nodes".into())); }
                    // Also accepts early integration files that redundantly
                    // stored the base links alongside the extra links.
                    child.strip_base_edges(&graph);
                    graph.payload = Some(Box::new(child));
                } else { graph.payload_parts.push(child); }
            }
            if p != data.len() { return Err(RusticError::Corrupted("hnsw: payload trailing bytes".into())); }
        }
        graph.payload_word_masks();
        Ok(graph)
    }

    /// Greedy descent + layer-0 beam. Returns up to `ef` accepted
    /// `(distance, doc)` packed keys, ascending.
    /// Layer-0 beam search. With `two_hop` (a selective filter), only
    /// accepted nodes are scored and queued: a rejected neighbour is not
    /// scored but its own neighbours are examined instead (ACORN-1, as
    /// Lucene's `FilteredHnswGraphSearcher`), so the number of distance
    /// computations stays close to an unfiltered search instead of growing
    /// with `1 / selectivity`.
    fn search_impl_mode<D: Dist>(&self, d: &D, ef: usize, num_docs: u32, accept: &impl Fn(DocId) -> bool, mode: FilterMode, s: &mut Scratch) {
        if mode == FilterMode::Payload {
            self.search_impl_mode_inner::<D, true>(d, ef, num_docs, accept, mode, s);
        } else {
            self.search_impl_mode_inner::<D, false>(d, ef, num_docs, accept, mode, s);
        }
    }

    // Specialize the hot loop: ordinary traversal must not pay a payload
    // branch for every neighbor, even when the caller's mode is dynamic.
    fn search_impl_mode_inner<D: Dist, const PAYLOAD: bool>(&self, d: &D, ef: usize, num_docs: u32, accept: &impl Fn(DocId) -> bool, mode: FilterMode, s: &mut Scratch) {
        let two_hop = mode.two_hop();
        let memo = mode == FilterMode::TwoHop { memo: true };
        let payload = PAYLOAD;
        s.out.clear();
        let Some(mut cur) = self.entry else { return };
        let mut start_level = self.layers.len() - 1;
        if payload {
            if let Some((level, entry)) = self.layers.iter().enumerate().rev().find_map(|(l, layer)| {
                layer.nodes.iter().copied().find(|&id| accept(id)).map(|id| (l, id))
            }) { cur = entry; start_level = level; }
        }
        let mut cur_d = d.dist(cur);
        let mut ops = 1u64;
        let mut accepted_entry = if payload && accept(cur) { Some((cur, cur_d)) } else { None };
        let mut seeds: Vec<(u32, f32)> = Vec::new();
        if payload {
            // POC supports whole values of one keyword field. Seed every
            // eligible value so an OR cannot strand the walk in one value's
            // connected component. Each seed navigates that value's own
            // upper layers, then all seeds share the same layer-zero beam.
            for part in self.payload_seed_parts() {
                let Some(mut point) = part.entry.filter(|&id| accept(id)) else { continue };
                let mut distance = d.dist(point); ops += 1;
                for level in (1..part.layers.len()).rev() {
                    loop {
                        let mut improved = false;
                        for &nb in part.layers[level].neighbors_of(point) {
                            if !accept(nb) { continue }
                            let next = d.dist(nb); ops += 1;
                            if next < distance { point = nb; distance = next; improved = true; }
                        }
                        if !improved { break; }
                    }
                }
                seeds.push((point, distance));
            }
            if let Some(&(point, distance)) = seeds.iter().min_by(|a,b| a.1.total_cmp(&b.1)) { cur = point; cur_d = distance; }
        }
        if seeds.is_empty() {
        for level in (1..=start_level).rev() {
            let layer = &self.layers[level];
            loop {
                let mut improved = false;
                for &nb in layer.neighbors_of(cur) {
                    if payload && !accept(nb) { continue; }
                    let dd = d.dist(nb);
                    ops += 1;
                    if payload && accept(nb) && accepted_entry.is_none_or(|(_, best)| dd < best) {
                        accepted_entry = Some((nb, dd));
                    }
                    if dd < cur_d {
                        cur = nb;
                        cur_d = dd;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        }
        if payload && !accept(cur) {
            if let Some((entry, distance)) = accepted_entry { cur = entry; cur_d = distance; }
        }
        let layer = &self.layers[0];
        s.visited.reset(num_docs.max(self.id_bound) as usize);
        s.cand.clear();
        s.top.clear();
        s.visited.insert(cur);
        let k0 = pack(cur_d, cur);
        s.cand.push(Reverse(k0));
        if accept(cur) {
            s.top.push(k0);
        }
        for (entry, distance) in seeds {
            if s.visited.insert(entry) {
                let key = pack(distance, entry);
                s.cand.push(Reverse(key));
                s.top.push(key);
                if s.top.len() > ef { s.top.pop(); }
            }
        }
        while let Some(Reverse(c)) = s.cand.pop() {
            if s.top.len() >= ef && (c >> 32) > (*s.top.peek().unwrap() >> 32) {
                break;
            }
            let nbs = layer.neighbors_of(c as u32);
            s.nbuf.clear();
            if two_hop {
                // Accepted unvisited neighbours are scored; rejected ones
                // are not, but their unvisited accepted neighbours are (up
                // to the node's degree). Every node is filter-checked at
                // most once per search (refusals are remembered in the
                // visited marks).
                let cap = nbs.len().max(1);
                s.rbuf.clear();
                for &nb in nbs {
                    let st = s.visited.state(nb);
                    if st == 1 {
                        continue;
                    }
                    s.visited.insert(nb);
                    if st == 0 && accept(nb) {
                        d.prefetch(nb);
                        s.nbuf.push(nb);
                    } else {
                        // Its adjacency row is read below: start the two
                        // dependent loads (offsets, then neighbours) now.
                        layer.prefetch_offsets(nb);
                        s.rbuf.push(nb);
                    }
                }
                // Rejected rows are expanded in order; keep the next few
                // rows' neighbour lists in flight (random rows of a large
                // CSR array are memory-latency bound).
                const AHEAD: usize = 6;
                if s.nbuf.len() < cap {
                    for &r in s.rbuf.iter().take(AHEAD) {
                        layer.prefetch_neighbors(r);
                    }
                }
                let mut ri = 0;
                while ri < s.rbuf.len() && s.nbuf.len() < cap {
                    let r = s.rbuf[ri];
                    ri += 1;
                    if let Some(&nx) = s.rbuf.get(ri + AHEAD - 1) {
                        layer.prefetch_neighbors(nx);
                    }
                    if memo {
                        for &nb2 in layer.neighbors_of(r) {
                            if s.nbuf.len() >= cap {
                                break;
                            }
                            // A rejected two-hop node stays unvisited (it
                            // can still bridge as a one-hop neighbour
                            // later); its rejection is remembered so it is
                            // checked once.
                            if s.visited.state(nb2) != 0 {
                                continue;
                            }
                            if accept(nb2) {
                                s.visited.insert(nb2);
                                d.prefetch(nb2);
                                s.nbuf.push(nb2);
                            } else {
                                s.visited.reject(nb2);
                            }
                        }
                    } else {
                        // Cheap filter: test it first. The whole row is
                        // tested branch-free into a bitmask (most two-hop
                        // nodes fail, so a data-dependent branch per node
                        // mispredicts); only the accepted ones, in row
                        // order, touch the visited marks.
                        for row in layer.neighbors_of(r).chunks(64) {
                            let mut m = 0u64;
                            for (j, &nb2) in row.iter().enumerate() {
                                m |= (accept(nb2) as u64) << j;
                            }
                            while m != 0 && s.nbuf.len() < cap {
                                let nb2 = row[m.trailing_zeros() as usize];
                                m &= m - 1;
                                if s.visited.insert(nb2) {
                                    d.prefetch(nb2);
                                    s.nbuf.push(nb2);
                                }
                            }
                            if s.nbuf.len() >= cap {
                                break;
                            }
                        }
                    }
                }
            } else {
                for &nb in nbs {
                    if s.visited.insert(nb) && (!payload || accept(nb)) {
                        d.prefetch(nb);
                        s.nbuf.push(nb);
                    }
                }
                if payload {
                    if let Some(overlay) = &self.payload {
                        for &nb in overlay.layers[0].neighbors_of(c as u32) {
                            if s.visited.insert(nb) && accept(nb) {
                                d.prefetch(nb);
                                s.nbuf.push(nb);
                            }
                        }
                    }
                }
            }
            ops += s.nbuf.len() as u64;
            for group in s.nbuf.chunks(4) {
                let distances = if group.len() == 4 {
                    d.four([group[0], group[1], group[2], group[3]])
                } else {
                    let mut out = [0.0; 4];
                    for (i, &nb) in group.iter().enumerate() { out[i] = d.dist(nb); }
                    out
                };
                for (&nb, dd) in group.iter().zip(distances) {
                    if dd == f32::INFINITY || dd.is_nan() {
                        continue;
                    }
                    if s.top.len() < ef || dd < funkey((*s.top.peek().unwrap() >> 32) as u32) {
                        let key = pack(dd, nb);
                        s.cand.push(Reverse(key));
                        // Two-hop queues only accepted nodes: no second check.
                        if two_hop || payload || accept(nb) {
                            s.top.push(key);
                            if s.top.len() > ef {
                                s.top.pop();
                            }
                        }
                    }
                }
            }
        }
        s.out.extend(s.top.drain());
        s.out.sort_unstable();
        add_vector_ops(ops);
    }

    /// Exact-score the first `n` candidates of `s.out` and return the top `k`
    /// as `(doc, score)`, score descending, ties by doc.
    fn finish(
        cands: &[u64],
        n: usize,
        k: usize,
        query: &[f32],
        vectors: &dyn VectorSource,
        similarity: Similarity,
    ) -> Vec<(DocId, f32)> {
        let mut out: Vec<(DocId, f32)> = cands[..n.min(cands.len())]
            .iter()
            .filter_map(|&c| {
                let doc = c as u32;
                vectors.get(doc).map(|v| (doc, similarity.score(query, v)))
            })
            .collect();
        add_vector_ops(out.len() as u64);
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// k-NN search.
    ///
    /// Greedy descent from the entry point through the upper layers, then an
    /// `ef`-bounded beam search on layer 0. Docs for which `accept(doc)` is
    /// false are traversed (for connectivity) but excluded from the results.
    /// Returns up to `k` `(doc, score)` pairs sorted by score descending,
    /// where `score` is [`Similarity::score`].
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        accept: impl Fn(DocId) -> bool,
    ) -> Vec<(DocId, f32)> {
        self.search_mode(query, k, ef, vectors, similarity, accept, FilterMode::Plain)
    }

    /// [`HnswGraph::search`] for a selective filter: traverses with the
    /// two-hop expansion of rejected nodes (see `search_impl_mode`), which
    /// keeps the cost close to an unfiltered search.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        accept: impl Fn(DocId) -> bool,
    ) -> Vec<(DocId, f32)> {
        self.search_mode(query, k, ef, vectors, similarity, accept, FilterMode::TwoHop { memo: true })
    }

    /// [`HnswGraph::search`] with an explicit [`FilterMode`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        accept: impl Fn(DocId) -> bool,
        mode: FilterMode,
    ) -> Vec<(DocId, f32)> {
        assert_eq!(query.len(), vectors.dim() as usize, "query dim mismatch");
        if self.layers.is_empty() || k == 0 {
            return Vec::new();
        }
        let graph = self.payload_overlay(mode);
        let ef = ef.max(k);
        let num_docs = vectors.num_docs();
        with_scratch(|s| {
            match vectors.as_contiguous() {
                Some(data) => {
                    let qn = norm(query);
                    let normalized = if similarity.metric() == Metric::Cos && qn.is_finite() {
                        vectors.normalized_contiguous().filter(|v| v.len() == data.len())
                    } else { None };
                    let normalized_query: Vec<f32> = if normalized.is_some() {
                        query.iter().map(|v| if qn > 0.0 { *v / qn } else { 0.0 }).collect()
                    } else { Vec::new() };
                    let metric = if normalized.is_some() { Metric::Dot } else { similarity.metric() };
                    let d = ContigDist {
                        data: normalized.unwrap_or(data),
                        dim: query.len(),
                        q: if normalized.is_some() { &normalized_query } else { query },
                        metric,
                        similarity,
                        inv_qn: if qn > 0.0 { 1.0 / qn } else { 0.0 },
                        squared_norms: if metric == Metric::Cos { vectors.squared_norms() } else { None },
                    };
                    graph.search_impl_mode(&d, ef, num_docs, &accept, mode, s);
                }
                None => {
                    let d = DynDist { vectors, q: query, similarity };
                    graph.search_impl_mode(&d, ef, num_docs, &accept, mode, s);
                }
            }
            // Candidates are already in exact distance order; score a
            // margin past k so float ties between distance and score can't
            // change the top k.
            Self::finish(&s.out, k + k / 2 + 4, k, query, vectors, similarity)
        })
    }

    /// Quantization-aware k-NN: traverse the graph scoring with `quant`
    /// (int8 or 1-bit), then rescore the best `ceil(k * oversample)`
    /// candidates with exact `f32` [`Similarity::score`] and return the top
    /// `k` (ES `rescore_vector.oversample`, Qdrant `rescore`). The beam is
    /// `max(ef, k * oversample)`. `quant` must be indexed by doc id.
    #[allow(clippy::too_many_arguments)]
    pub fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        oversample: f32,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        quant: &QuantizedVectors,
        accept: impl Fn(DocId) -> bool,
    ) -> Vec<(DocId, f32)> {
        self.search_quantized_mode(query, k, ef, oversample, vectors, similarity, quant, accept, FilterMode::Plain)
    }

    /// [`HnswGraph::search_quantized`] with the two-hop filtered traversal
    /// of [`HnswGraph::search_filtered`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_quantized_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        oversample: f32,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        quant: &QuantizedVectors,
        accept: impl Fn(DocId) -> bool,
    ) -> Vec<(DocId, f32)> {
        self.search_quantized_mode(query, k, ef, oversample, vectors, similarity, quant, accept, FilterMode::TwoHop { memo: true })
    }

    /// [`HnswGraph::search_quantized`] with an explicit [`FilterMode`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_quantized_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        oversample: f32,
        vectors: &dyn VectorSource,
        similarity: Similarity,
        quant: &QuantizedVectors,
        accept: impl Fn(DocId) -> bool,
        mode: FilterMode,
    ) -> Vec<(DocId, f32)> {
        assert_eq!(query.len(), vectors.dim() as usize, "query dim mismatch");
        if self.layers.is_empty() || k == 0 {
            return Vec::new();
        }
        let graph = self.payload_overlay(mode);
        let pq = quant.prepare(query);
        let num_docs = vectors.num_docs();
        if oversample <= 0.0 {
            // No rescoring (ES `rescore_vector.oversample: 0` / absent on
            // int8 / int4): the quantized estimates are the scores.
            return with_scratch(|s| {
                graph.search_impl_mode(&QuantDist { qv: quant, pq: &pq }, ef.max(k), num_docs, &accept, mode, s);
                Self::finish_quantized(&s.out, k, similarity)
            });
        }
        let rescore = ((k as f32 * oversample.max(1.0)).ceil() as usize).max(k);
        with_scratch(|s| {
            graph.search_impl_mode(&QuantDist { qv: quant, pq: &pq }, ef.max(rescore), num_docs, &accept, mode, s);
            Self::finish(&s.out, rescore, k, query, vectors, similarity)
        })
    }

    /// The top `k` of `s.out` (ascending packed `(distance, doc)`) scored by
    /// their quantized distance estimates ([`Similarity::score_of_distance`]).
    fn finish_quantized(cands: &[u64], k: usize, similarity: Similarity) -> Vec<(DocId, f32)> {
        cands
            .iter()
            .take(k)
            .map(|&c| {
                let (d, doc) = unpack(c);
                (doc, similarity.score_of_distance(d).max(0.0))
            })
            .collect()
    }
}

/// One level of a graph assembled from fixed-degree k-NN rows (see
/// [`HnswGraph::from_levels`]).
pub struct KnnLevel {
    /// The level's nodes, ascending (all ordinals `0..n` on level 0).
    pub nodes: Vec<DocId>,
    /// Row width of `rows`.
    pub degree: usize,
    /// `nodes.len() * degree` neighbour ids (ordinals), row `i` belonging
    /// to `nodes[i]`.
    pub rows: Vec<u32>,
}

impl HnswGraph {
    /// A single-layer graph over ordinals `0..n` from a fixed-degree k-NN
    /// adjacency (`knn[i * degree..][..degree]` = node `i`'s neighbours),
    /// searched from `entry`.
    pub fn from_knn_rows(n: usize, degree: usize, knn: &[u32], entry: DocId) -> HnswGraph {
        if n == 0 {
            return Self::from_levels(Vec::new(), None, entry);
        }
        assert!(knn.len() >= n * degree, "hnsw: knn rows shorter than n * degree");
        Self::from_levels(vec![KnnLevel { nodes: (0..n as u32).collect(), degree, rows: knn[..n * degree].to_vec() }], None, entry)
    }

    /// A graph whose lower levels are fixed-degree k-NN graphs (e.g. CAGRA
    /// graphs built on the GPU over nested node sets, level 0 = every
    /// ordinal) and whose top levels are `top`, an HNSW graph over the
    /// last level's subset (or nothing: one level searched from `entry`).
    /// In each row, ids not on that level, self-loops and repeats are
    /// dropped; the order of the rest is kept (CAGRA rows are closest
    /// first). Each level's nodes must include the next level's.
    pub fn from_levels(levels: Vec<KnnLevel>, top: Option<HnswGraph>, entry: DocId) -> HnswGraph {
        let mut layers = Vec::with_capacity(levels.len() + top.as_ref().map_or(0, |t| t.layers.len()));
        for level in levels {
            let KnnLevel { nodes, degree, rows } = level;
            assert!(rows.len() >= nodes.len() * degree, "hnsw: knn rows shorter than nodes * degree");
            let dense = nodes.last().is_none_or(|&l| l as usize + 1 == nodes.len());
            let on_level = |v: u32| if dense { (v as usize) < nodes.len() } else { nodes.binary_search(&v).is_ok() };
            let mut offsets = Vec::with_capacity(nodes.len() + 1);
            offsets.push(0u32);
            let mut neighbors: Vec<DocId> = Vec::with_capacity(nodes.len() * degree);
            for (i, &node) in nodes.iter().enumerate() {
                let start = neighbors.len();
                for &v in &rows[i * degree..(i + 1) * degree] {
                    if v == node || !on_level(v) || neighbors[start..].contains(&v) {
                        continue;
                    }
                    neighbors.push(v);
                }
                offsets.push(neighbors.len() as u32);
            }
            layers.push(Layer::new(nodes, offsets, neighbors, degree));
        }
        let entry = match top {
            Some(t) if t.entry.is_some() => {
                let e = t.entry;
                layers.extend(t.layers);
                e
            }
            _ => (!layers.is_empty()).then_some(entry),
        };
        let id_bound = id_bound(&layers);
        HnswGraph { layers, entry, id_bound, payload: None, payload_parts: Vec::new(), payload_masks: std::sync::OnceLock::new() }
    }
}

// ---------------------------------------------------------------------------
// Builder.

/// Builder for an HNSW graph.
pub struct HnswBuilder;

thread_local! {
    /// Graphs built on this thread (see [`HnswBuilder::builds_on_this_thread`]).
    static BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Vectors inserted into graphs on this thread.
    static INSERTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

const NO_NODE: u32 = u32::MAX;

/// Mutable graph over dense ordinals `0..n` (see the module docs).
struct Build<'a> {
    data: &'a [f32],
    dim: usize,
    metric: Metric,
    /// The field's similarity (packed rows score through it).
    similarity: Similarity,
    /// Cosine: `1 / |v|` per ordinal (0 for zero vectors).
    inv_norm: Vec<f32>,
    m: usize,
    m0: usize,
    ef_c: usize,
    levels: Vec<u8>,
    /// Layer 0: per node `[count, ids x m0]`.
    l0: Vec<AtomicU32>,
    /// Layer 0: per node `[dist bits x m0]`, parallel to `l0`'s ids.
    l0d: Vec<AtomicU32>,
    /// Per node: index of its layer-1 block in `up` / `upd` (layers
    /// `1..=level` are consecutive blocks), or `NO_NODE`.
    up_block: Vec<u32>,
    /// Upper layers: per block `[count, ids x m]`.
    up: Vec<AtomicU32>,
    /// Upper layers: per block `[dist bits x m]`.
    upd: Vec<AtomicU32>,
    locks: Vec<Mutex<()>>,
    /// `(entry ordinal, its level)`; `NO_NODE` before the first insert.
    entry: Mutex<(u32, usize)>,
}

fn atomics(n: usize) -> Vec<AtomicU32> {
    (0..n).map(|_| AtomicU32::new(0)).collect()
}

impl<'a> Build<'a> {
    #[inline(always)]
    fn row(&self, i: u32) -> &'a [f32] {
        let s = i as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    #[inline(always)]
    fn inv(&self, i: u32) -> f32 {
        if self.metric == Metric::Cos { self.inv_norm[i as usize] } else { 0.0 }
    }

    /// Distance from vector `q` (inverse norm `q_inv`, cosine only) to node `b`.
    #[inline(always)]
    fn dist(&self, q: &[f32], q_inv: f32, b: u32) -> f32 {
        let v = self.row(b);
        match self.metric {
            Metric::L2 => l2_sq(q, v),
            Metric::Dot => -dot(q, v),
            Metric::Cos => 1.0 - dot(q, v) * q_inv * self.inv_norm[b as usize],
            Metric::Packed => self.similarity.distance(q, v),
        }
    }

    /// `(count+ids slots, dist slots, capacity)` of `node` on `level`.
    #[inline(always)]
    fn links(&self, node: u32, level: usize) -> (&[AtomicU32], &[AtomicU32], usize) {
        if level == 0 {
            let n = node as usize;
            (&self.l0[n * (self.m0 + 1)..(n + 1) * (self.m0 + 1)], &self.l0d[n * self.m0..(n + 1) * self.m0], self.m0)
        } else {
            let b = self.up_block[node as usize] as usize + level - 1;
            (&self.up[b * (self.m + 1)..(b + 1) * (self.m + 1)], &self.upd[b * self.m..(b + 1) * self.m], self.m)
        }
    }

    fn greedy(&self, q: &[f32], q_inv: f32, mut cur: u32, mut cur_d: f32, level: usize) -> (u32, f32) {
        loop {
            let mut improved = false;
            let (lk, _, cap) = self.links(cur, level);
            let cnt = (lk[0].load(Acquire) as usize).min(cap);
            for slot in &lk[1..1 + cnt] {
                let nb = slot.load(Relaxed);
                let d = self.dist(q, q_inv, nb);
                if d < cur_d {
                    cur = nb;
                    cur_d = d;
                    improved = true;
                }
            }
            if !improved {
                return (cur, cur_d);
            }
        }
    }

    /// Beam search on `level` from `ep`; leaves the `ef` nearest in `s.out`
    /// (ascending packed keys).
    fn search_layer(&self, q: &[f32], q_inv: f32, ep: u32, ep_d: f32, level: usize, s: &mut Scratch) {
        let ef = self.ef_c;
        s.visited.reset(self.levels.len());
        s.cand.clear();
        s.top.clear();
        s.visited.insert(ep);
        let k0 = pack(ep_d, ep);
        s.cand.push(Reverse(k0));
        s.top.push(k0);
        while let Some(Reverse(c)) = s.cand.pop() {
            let worst = *s.top.peek().unwrap();
            if s.top.len() >= ef && (c >> 32) > (worst >> 32) {
                break;
            }
            let (lk, _, cap) = self.links(c as u32, level);
            let cnt = (lk[0].load(Acquire) as usize).min(cap);
            s.nbuf.clear();
            for slot in &lk[1..1 + cnt] {
                let nb = slot.load(Relaxed);
                if s.visited.insert(nb) {
                    prefetch(self.row(nb).as_ptr() as *const u8, self.dim * 4);
                    s.nbuf.push(nb);
                }
            }
            for &nb in &s.nbuf {
                let d = self.dist(q, q_inv, nb);
                if s.top.len() < ef || d < funkey((*s.top.peek().unwrap() >> 32) as u32) {
                    let key = pack(d, nb);
                    s.cand.push(Reverse(key));
                    s.top.push(key);
                    if s.top.len() > ef {
                        s.top.pop();
                    }
                }
            }
        }
        s.out.clear();
        s.out.extend(s.top.drain());
        s.out.sort_unstable();
    }

    /// HNSW heuristic: keep a candidate only if it is closer to the base
    /// than to every already-kept neighbor. `cands` ascending.
    fn select_diverse(&self, cands: &[u64], max: usize, out: &mut Vec<u64>) {
        out.clear();
        for &c in cands {
            if out.len() >= max {
                break;
            }
            let (dc, id) = unpack(c);
            let v = self.row(id);
            let vi = self.inv(id);
            if out.iter().all(|&s| self.dist(v, vi, s as u32) >= dc) {
                out.push(c);
            }
        }
    }

    /// Index (in ascending `list`) of the neighbor to drop on overflow:
    /// the farthest one that is closer to a nearer kept neighbor than to the
    /// base (non-diverse), else the farthest (Lucene `findWorstNonDiverse`).
    fn worst_non_diverse(&self, list: &[u64]) -> usize {
        for i in (1..list.len()).rev() {
            let (di, id) = unpack(list[i]);
            let v = self.row(id);
            let vi = self.inv(id);
            if list[..i].iter().any(|&s| self.dist(v, vi, s as u32) < di) {
                return i;
            }
        }
        list.len() - 1
    }

    /// Add `node` (at distance `d`) to `nb`'s list on `level`.
    fn add_link(&self, nb: u32, node: u32, d: f32, level: usize, tmp: &mut Vec<u64>) {
        let _g = self.locks[nb as usize].lock();
        let (lk, ld, cap) = self.links(nb, level);
        let cnt = lk[0].load(Relaxed) as usize;
        if cnt < cap {
            lk[1 + cnt].store(node, Relaxed);
            ld[cnt].store(d.to_bits(), Relaxed);
            lk[0].store(cnt as u32 + 1, Release);
            return;
        }
        tmp.clear();
        for j in 0..cnt {
            tmp.push(pack(f32::from_bits(ld[j].load(Relaxed)), lk[1 + j].load(Relaxed)));
        }
        tmp.push(pack(d, node));
        tmp.sort_unstable();
        let drop = self.worst_non_diverse(tmp);
        tmp.remove(drop);
        for (j, &x) in tmp.iter().enumerate() {
            let (dx, id) = unpack(x);
            lk[1 + j].store(id, Relaxed);
            ld[j].store(dx.to_bits(), Relaxed);
        }
    }

    fn insert(&self, node: u32, s: &mut Scratch) {
        let level = self.levels[node as usize] as usize;
        let mut guard = Some(self.entry.lock());
        let (ep, max_level) = **guard.as_ref().unwrap();
        if ep == NO_NODE {
            **guard.as_mut().unwrap() = (node, level);
            return;
        }
        // A node that raises the top level holds the entry lock for its
        // whole insertion (hnswlib); everyone else releases it at once.
        if level <= max_level {
            guard = None;
        }
        let q = self.row(node);
        let q_inv = self.inv(node);
        let mut cur = ep;
        let mut cur_d = self.dist(q, q_inv, cur);
        for l in ((level + 1)..=max_level).rev() {
            (cur, cur_d) = self.greedy(q, q_inv, cur, cur_d, l);
        }
        for l in (0..=level.min(max_level)).rev() {
            self.search_layer(q, q_inv, cur, cur_d, l, s);
            let cap = if l == 0 { self.m0 } else { self.m };
            let mut sel = std::mem::take(&mut s.sel);
            self.select_diverse(&s.out, cap, &mut sel);
            {
                let _g = self.locks[node as usize].lock();
                let (lk, ld, _) = self.links(node, l);
                for (j, &x) in sel.iter().enumerate() {
                    let (dx, id) = unpack(x);
                    lk[1 + j].store(id, Relaxed);
                    ld[j].store(dx.to_bits(), Relaxed);
                }
                lk[0].store(sel.len() as u32, Release);
            }
            for &x in &sel {
                let (dx, nb) = unpack(x);
                self.add_link(nb, node, dx, l, &mut s.tmp);
            }
            s.sel = sel;
            (cur_d, cur) = unpack(s.out[0]);
        }
        if let Some(mut g) = guard {
            *g = (node, level);
        }
    }

    /// Freeze into CSR layers, mapping ordinals to doc ids.
    fn finish(&self, docs: &[DocId]) -> HnswGraph {
        let entry = self.entry.lock().0;
        let max_level = self.levels.iter().copied().max().unwrap_or(0) as usize;
        let mut layers = Vec::with_capacity(max_level + 1);
        for l in 0..=max_level {
            let max_m = if l == 0 { self.m0 } else { self.m };
            let mut nodes = Vec::new();
            let mut offsets = vec![0u32];
            let mut neighbors = Vec::new();
            for (ord, &lv) in self.levels.iter().enumerate() {
                if (lv as usize) < l {
                    continue;
                }
                let (lk, _, cap) = self.links(ord as u32, l);
                let cnt = (lk[0].load(Acquire) as usize).min(cap);
                nodes.push(docs[ord]);
                neighbors.extend(lk[1..1 + cnt].iter().map(|x| docs[x.load(Relaxed) as usize]));
                offsets.push(neighbors.len() as u32);
            }
            layers.push(Layer::new(nodes, offsets, neighbors, max_m));
        }
        let id_bound = id_bound(&layers);
        HnswGraph { layers, entry: (entry != NO_NODE).then(|| docs[entry as usize]), id_bound, payload: None, payload_parts: Vec::new(), payload_masks: std::sync::OnceLock::new() }
    }
}

impl HnswBuilder {
    /// Number of graphs built on the calling thread so far. Thread-local so
    /// tests can assert "this operation built no graph" while other tests
    /// build graphs concurrently.
    pub fn builds_on_this_thread() -> u64 {
        BUILDS.with(|c| c.get())
    }

    /// Number of vectors inserted into graphs on the calling thread so far.
    pub fn inserts_on_this_thread() -> u64 {
        INSERTS.with(|c| c.get())
    }

    /// Count a graph of `n` vectors built elsewhere (on the GPU) in the
    /// per-thread build / insert counters.
    pub(crate) fn note_external_build(n: usize) {
        BUILDS.with(|c| c.set(c.get() + 1));
        INSERTS.with(|c| c.set(c.get() + n as u64));
    }

    /// Build a graph over the docs of `vectors` that have a vector.
    ///
    /// Deterministic for a given `params.seed` when `params.threads == 1`.
    pub fn build(
        dim: u32,
        vectors: &dyn VectorSource,
        num_docs: u32,
        similarity: Similarity,
        params: HnswParams,
    ) -> HnswGraph {
        assert_eq!(dim, vectors.dim(), "hnsw: dim mismatch with vector source");
        let m = params.m.max(1);
        let m0 = 2 * m;
        let ef_c = params.ef_construction.max(1);
        // Level multiplier 1/ln(M) (Malkov & Yashunin); M=1 degenerates to one layer.
        let ml = if m > 1 { 1.0 / (m as f64).ln() } else { 0.0 };
        let dim = dim as usize;

        // Docs with vectors, ascending; ordinal i <-> docs[i].
        let docs: Vec<DocId> = (0..num_docs).filter(|&d| vectors.get(d).is_some()).collect();
        BUILDS.with(|c| c.set(c.get() + 1));
        INSERTS.with(|c| c.set(c.get() + docs.len() as u64));
        if docs.is_empty() {
            return HnswGraph { layers: Vec::new(), entry: None, id_bound: 0, payload: None, payload_parts: Vec::new(), payload_masks: std::sync::OnceLock::new() };
        }
        let n = docs.len();

        // Contiguous ordinal-major column: borrowed when the source already
        // is one (every doc present), else copied once.
        let owned: Vec<f32>;
        let data: &[f32] = match vectors.as_contiguous() {
            Some(c) if n == num_docs as usize && c.len() >= n * dim => &c[..n * dim],
            _ => {
                let mut v = Vec::with_capacity(n * dim);
                for &d in &docs {
                    v.extend_from_slice(vectors.get(d).unwrap());
                }
                owned = v;
                &owned
            }
        };

        // Deterministic level assignment (xorshift64* PRNG).
        let mut rng = XorShift64Star::new(params.seed);
        let levels: Vec<u8> = (0..n)
            .map(|_| {
                let u = rng.next_f64(); // uniform in [0, 1)
                ((-u.ln() * ml).floor() as usize).min(u8::MAX as usize) as u8
            })
            .collect();
        let mut up_block = vec![NO_NODE; n];
        let mut blocks = 0usize;
        for (i, &lv) in levels.iter().enumerate() {
            if lv > 0 {
                up_block[i] = blocks as u32;
                blocks += lv as usize;
            }
        }
        let metric = similarity.metric();
        let inv_norm = if metric == Metric::Cos {
            data.chunks_exact(dim)
                .map(|v| {
                    let x = norm(v);
                    if x > 0.0 { 1.0 / x } else { 0.0 }
                })
                .collect()
        } else {
            Vec::new()
        };
        let b = Build {
            data,
            dim,
            metric,
            similarity,
            inv_norm,
            m,
            m0,
            ef_c,
            levels,
            l0: atomics(n * (m0 + 1)),
            l0d: atomics(n * m0),
            up_block,
            up: atomics(blocks * (m + 1)),
            upd: atomics(blocks * m),
            locks: (0..n).map(|_| Mutex::new(())).collect(),
            entry: Mutex::new((NO_NODE, 0)),
        };

        let insert = |i: usize| with_scratch(|s| b.insert(i as u32, s));
        if params.threads == 1 || n < 2048 {
            (0..n).for_each(insert);
        } else {
            // Seed the graph sequentially so early parallel inserts don't
            // all race on a near-empty graph.
            let seq = 1024.min(n);
            (0..seq).for_each(insert);
            let run = || (seq..n).into_par_iter().with_min_len(16).for_each(insert);
            if params.threads == 0 {
                run();
            } else {
                match rayon::ThreadPoolBuilder::new().num_threads(params.threads).build() {
                    Ok(pool) => pool.install(run),
                    Err(_) => run(),
                }
            }
        }
        b.finish(&docs)
    }

    /// Build a graph over the docs of `vectors` starting from an existing
    /// graph instead of from scratch (Lucene 9.10's
    /// `IncrementalHnswGraphMerger`; Qdrant's old-graph reuse on merge).
    ///
    /// `seed_map[s]` is the doc id (in `vectors`) of node `s` of `seed`, or
    /// `u32::MAX` when that node is dropped (dead). Every kept seed node
    /// keeps its level and its adjacency (mapped); nodes that lost
    /// neighbours to dropped nodes are repaired locally (FreshDiskANN-style
    /// consolidation: the dropped neighbours' own neighbours become
    /// candidates and the HNSW diversity heuristic picks the list). Only
    /// the docs not in the seed are then inserted, so the cost is
    /// proportional to the new vectors plus one distance per kept edge.
    ///
    /// Falls back to [`build`](Self::build) when the seed cannot be used
    /// (empty, nothing kept, or built with another `m`). Deterministic for a
    /// given `params.seed` when `params.threads == 1`.
    pub fn build_from(
        dim: u32,
        vectors: &dyn VectorSource,
        num_docs: u32,
        similarity: Similarity,
        params: HnswParams,
        seed: &HnswGraph,
        seed_map: &[u32],
    ) -> HnswGraph {
        assert_eq!(dim, vectors.dim(), "hnsw: dim mismatch with vector source");
        let m = params.m.max(1);
        if seed.layers.is_empty() || !seed.built_with_m(m) {
            return Self::build(dim, vectors, num_docs, similarity, params);
        }
        let m0 = 2 * m;
        let ef_c = params.ef_construction.max(1);
        let ml = if m > 1 { 1.0 / (m as f64).ln() } else { 0.0 };
        let dim = dim as usize;
        let docs: Vec<DocId> = (0..num_docs).filter(|&d| vectors.get(d).is_some()).collect();
        let n = docs.len();
        // Seed node -> build ordinal (`NO_NODE` = dropped).
        let ord_of = |d: u32| -> u32 {
            if d == NO_NODE {
                return NO_NODE;
            }
            match docs.binary_search(&d) {
                Ok(i) => i as u32,
                Err(_) => NO_NODE,
            }
        };
        let smap: Vec<u32> = (0..seed.id_bound as usize).map(|s| ord_of(seed_map.get(s).copied().unwrap_or(NO_NODE))).collect();
        let sm = |s: u32| -> u32 { smap.get(s as usize).copied().unwrap_or(NO_NODE) };
        // Seed level of each kept node.
        let mut seed_level: Vec<i16> = vec![-1; n];
        for (l, layer) in seed.layers.iter().enumerate() {
            for &s in &layer.nodes {
                let o = sm(s);
                if o != NO_NODE {
                    seed_level[o as usize] = l.min(u8::MAX as usize) as i16;
                }
            }
        }
        let kept = seed_level.iter().filter(|&&l| l >= 0).count();
        if kept == 0 {
            return Self::build(dim as u32, vectors, num_docs, similarity, params);
        }
        BUILDS.with(|c| c.set(c.get() + 1));
        INSERTS.with(|c| c.set(c.get() + (n - kept) as u64));

        let owned: Vec<f32>;
        let data: &[f32] = match vectors.as_contiguous() {
            Some(c) if n == num_docs as usize && c.len() >= n * dim => &c[..n * dim],
            _ => {
                let mut v = Vec::with_capacity(n * dim);
                for &d in &docs {
                    v.extend_from_slice(vectors.get(d).unwrap());
                }
                owned = v;
                &owned
            }
        };
        // Levels: kept nodes keep theirs; new ones draw from the PRNG in
        // ordinal order (one draw per ordinal, so the sequence does not
        // depend on which nodes were kept).
        let mut rng = XorShift64Star::new(params.seed);
        let levels: Vec<u8> = (0..n)
            .map(|i| {
                let u = rng.next_f64();
                if seed_level[i] >= 0 {
                    seed_level[i] as u8
                } else {
                    ((-u.ln() * ml).floor() as usize).min(u8::MAX as usize) as u8
                }
            })
            .collect();
        let mut up_block = vec![NO_NODE; n];
        let mut blocks = 0usize;
        for (i, &lv) in levels.iter().enumerate() {
            if lv > 0 {
                up_block[i] = blocks as u32;
                blocks += lv as usize;
            }
        }
        let metric = similarity.metric();
        let inv_norm = if metric == Metric::Cos {
            data.chunks_exact(dim)
                .map(|v| {
                    let x = norm(v);
                    if x > 0.0 { 1.0 / x } else { 0.0 }
                })
                .collect()
        } else {
            Vec::new()
        };
        // Entry point: the seed's when kept, else the kept node with the
        // highest level (lowest ordinal on ties).
        let entry = match seed.entry.map(sm) {
            Some(o) if o != NO_NODE => o,
            _ => {
                let top = seed_level.iter().copied().max().unwrap_or(0);
                seed_level.iter().position(|&l| l == top).unwrap() as u32
            }
        };
        let entry_level = levels[entry as usize] as usize;
        let b = Build {
            data,
            dim,
            metric,
            similarity,
            inv_norm,
            m,
            m0,
            ef_c,
            levels,
            l0: atomics(n * (m0 + 1)),
            l0d: atomics(n * m0),
            up_block,
            up: atomics(blocks * (m + 1)),
            upd: atomics(blocks * m),
            locks: (0..n).map(|_| Mutex::new(())).collect(),
            entry: Mutex::new((entry, entry_level)),
        };
        // Seed node of each kept ordinal.
        let mut seed_of = vec![NO_NODE; n];
        for (s, &o) in smap.iter().enumerate() {
            if o != NO_NODE {
                seed_of[o as usize] = s as u32;
            }
        }
        // Copy (and repair) the kept adjacency. Each ordinal only writes its
        // own lists and only reads the seed graph and the vectors, so this
        // is order-independent (deterministic in parallel too).
        let copy = |o: usize| {
            with_scratch(|s| {
                let so = seed_of[o];
                if so == NO_NODE {
                    return;
                }
                let q = b.row(o as u32);
                let qi = b.inv(o as u32);
                for l in 0..=b.levels[o] as usize {
                    let layer = &seed.layers[l];
                    let cap = if l == 0 { b.m0 } else { b.m };
                    s.tmp.clear();
                    let mut damaged = false;
                    for &nb in layer.neighbors_of(so) {
                        let x = sm(nb);
                        if x != NO_NODE {
                            s.tmp.push(pack(b.dist(q, qi, x), x));
                        } else {
                            damaged = true;
                        }
                    }
                    if damaged {
                        // Neighbours of the dropped neighbours are candidates.
                        for &nb in layer.neighbors_of(so) {
                            if sm(nb) != NO_NODE {
                                continue;
                            }
                            for &nb2 in layer.neighbors_of(nb) {
                                let x = sm(nb2);
                                if x != NO_NODE && x != o as u32 {
                                    s.tmp.push(pack(b.dist(q, qi, x), x));
                                }
                            }
                        }
                    }
                    // A candidate reached twice has the same packed key both
                    // times, so sorting makes duplicates adjacent.
                    s.tmp.sort_unstable();
                    s.tmp.dedup();
                    let mut sel = std::mem::take(&mut s.sel);
                    if damaged {
                        b.select_diverse(&s.tmp, cap, &mut sel);
                    } else {
                        sel.clear();
                        sel.extend(s.tmp.iter().take(cap));
                    }
                    let (lk, ld, _) = b.links(o as u32, l);
                    for (j, &x) in sel.iter().enumerate() {
                        let (dx, id) = unpack(x);
                        lk[1 + j].store(id, Relaxed);
                        ld[j].store(dx.to_bits(), Relaxed);
                    }
                    lk[0].store(sel.len() as u32, Release);
                    s.sel = sel;
                }
            })
        };
        let new: Vec<usize> = (0..n).filter(|&i| seed_of[i] == NO_NODE).collect();
        let insert = |i: usize| with_scratch(|s| b.insert(i as u32, s));
        let parallel = params.threads != 1 && n >= 2048;
        let run = || {
            if parallel {
                (0..n).into_par_iter().with_min_len(64).for_each(copy);
                // A tiny seed is grown sequentially first, as in `build`.
                let seq = new.len().min(1024usize.saturating_sub(kept));
                new[..seq].iter().for_each(|&i| insert(i));
                new[seq..].par_iter().with_min_len(16).for_each(|&i| insert(i));
            } else {
                (0..n).for_each(copy);
                new.iter().for_each(|&i| insert(i));
            }
        };
        if !parallel || params.threads == 0 {
            run();
        } else {
            match rayon::ThreadPoolBuilder::new().num_threads(params.threads).build() {
                Ok(pool) => pool.install(run),
                Err(_) => run(),
            }
        }
        b.finish(&docs)
    }
}

/// Tiny deterministic xorshift64* PRNG.
struct XorShift64Star(u64);

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        XorShift64Star(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "payload-hnsw-poc")]
    #[test]
    fn payload_overlay_preserves_base_and_partition_edges() {
        let src = random_source(128, 8, 13, 0);
        let subset = Mem { dim: 8, rows: src.rows.iter().enumerate().map(|(i, row)| if i % 2 == 0 { row.clone() } else { None }).collect() };
        let base = HnswBuilder::build(8, &src, 128, Similarity::Cosine, HnswParams::default());
        let partition = HnswBuilder::build(8, &subset, 128, Similarity::Cosine, HnswParams::default());
        let overlay = base.with_payload_edges(std::slice::from_ref(&partition));
        assert_eq!(base.entry, overlay.entry);
        assert_eq!(base.layers[0].nodes, overlay.layers[0].nodes);
        for node in 0..128 {
            let row = overlay.layers[0].neighbors_of(node);
            for &nb in base.layers[0].neighbors_of(node).iter().chain(partition.layers[0].neighbors_of(node)) {
                assert!(row.contains(&nb));
            }
            let mut unique = row.to_vec(); unique.sort_unstable(); unique.dedup();
            assert_eq!(unique.len(), row.len());
        }
        let reopened = HnswGraph::open(&overlay.encode()).unwrap();
        assert_eq!(reopened.layers[0].neighbors, overlay.layers[0].neighbors);
    }

    #[test]
    fn cached_cosine_norms_preserve_traversal_and_scores() {
        struct Dense<'a> { dim: u32, data: &'a [f32], norms: Option<&'a [f32]> }
        impl VectorSource for Dense<'_> {
            fn dim(&self) -> u32 { self.dim }
            fn num_docs(&self) -> u32 { (self.data.len() / self.dim as usize) as u32 }
            fn get(&self, doc: DocId) -> Option<&[f32]> {
                let start = doc as usize * self.dim as usize;
                self.data.get(start..start + self.dim as usize)
            }
            fn as_contiguous(&self) -> Option<&[f32]> { Some(self.data) }
            fn squared_norms(&self) -> Option<&[f32]> { self.norms }
        }
        for dim in [17, 31, 128, 384] {
            let source = random_source(600, dim, 42, 0);
            let mut data: Vec<f32> = source.rows.iter().flat_map(|v|v.as_ref().unwrap().iter().copied()).collect();
            data[..dim as usize].fill(0.0);
            let norms: Vec<f32> = data.chunks_exact(dim as usize).map(|v|dot_and_sq_norm(v, v).1).collect();
            let plain = Dense { dim, data: &data, norms: None };
            let cached = Dense { dim, data: &data, norms: Some(&norms) };
            let graph = HnswBuilder::build(dim, &plain, 600, Similarity::Cosine, HnswParams::default());
            for query in [plain.get(0).unwrap(), plain.get(13).unwrap(), plain.get(255).unwrap()] {
                for ef in [100, 500] {
                    for mode in [FilterMode::Plain, FilterMode::TwoHop { memo: false }, FilterMode::TwoHop { memo: true }] {
                        let accept = |doc|doc % 3 != 0;
                        let expected = graph.search_mode(query, 10, ef, &plain, Similarity::Cosine, accept, mode);
                        let actual = graph.search_mode(query, 10, ef, &cached, Similarity::Cosine, accept, mode);
                        assert_eq!(actual, expected, "dim={dim} ef={ef} mode={mode:?}");
                    }
                }
            }
        }
    }

    /// In-memory vector source; `None` entries are docs without a vector.
    struct Mem {
        dim: u32,
        rows: Vec<Option<Vec<f32>>>,
    }

    impl VectorSource for Mem {
        fn dim(&self) -> u32 {
            self.dim
        }
        fn num_docs(&self) -> u32 {
            self.rows.len() as u32
        }
        fn get(&self, doc: DocId) -> Option<&[f32]> {
            self.rows.get(doc as usize)?.as_deref()
        }
    }

    fn random_source(n: usize, dim: u32, seed: u64, missing_every: usize) -> Mem {
        let mut rng = XorShift64Star::new(seed);
        let rows = (0..n)
            .map(|i| {
                if missing_every > 0 && i % missing_every == 0 {
                    None
                } else {
                    Some((0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect())
                }
            })
            .collect();
        Mem { dim, rows }
    }

    fn brute(src: &Mem, q: &[f32], k: usize, sim: Similarity, accept: impl Fn(DocId) -> bool) -> Vec<DocId> {
        let mut all: Vec<(f32, DocId)> = (0..src.num_docs())
            .filter(|&d| accept(d))
            .filter_map(|d| src.get(d).map(|v| (sim.score(q, v), d)))
            .collect();
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        all.into_iter().take(k).map(|(_, d)| d).collect()
    }

    fn recall(got: &[(DocId, f32)], want: &[DocId]) -> f32 {
        let hits = got.iter().filter(|(d, _)| want.contains(d)).count();
        hits as f32 / want.len().max(1) as f32
    }

    fn queries(n: usize, dim: u32) -> Vec<Vec<f32>> {
        let mut rng = XorShift64Star::new(777);
        (0..n)
            .map(|_| (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect())
            .collect()
    }

    /// Debug builds are ~20x slower; keep the corpus small there.
    fn corpus_size() -> usize {
        if cfg!(debug_assertions) { 600 } else { 2000 }
    }

    #[test]
    fn batched_distances_preserve_graph_traversal_and_operation_counts() {
        struct Single<'a>(&'a ContigDist<'a>);
        impl Dist for Single<'_> {
            fn dist(&self, id: u32) -> f32 { self.0.dist(id) }
        }
        for dim in [7, 32, 128] {
            let src = random_source(600, dim, 19, 0);
            let data: Vec<f32> = src.rows.iter().flatten().flatten().copied().collect();
            for sim in [Similarity::DotProduct, Similarity::L2, Similarity::Cosine] {
                let graph = HnswBuilder::build(dim, &src, src.num_docs(), sim, HnswParams::default());
                for q in queries(4, dim) {
                    let qn = norm(&q);
                    let distance = ContigDist { data: &data, dim: dim as usize, q: &q,
                        metric: sim.metric(), similarity: sim, inv_qn: if qn > 0.0 { 1.0 / qn } else { 0.0 }, squared_norms: None };
                    for mode in [FilterMode::Plain, FilterMode::TwoHop { memo: true }, FilterMode::TwoHop { memo: false }] {
                        for ef in [10, 100, 500] {
                            let accept = |id: u32| id % 2 == 0;
                            let mut one = Scratch::default();
                            let mut batch = Scratch::default();
                            let before = vector_ops();
                            graph.search_impl_mode(&Single(&distance), ef, src.num_docs(), &accept, mode, &mut one);
                            let ops = vector_ops() - before;
                            let before = vector_ops();
                            graph.search_impl_mode(&distance, ef, src.num_docs(), &accept, mode, &mut batch);
                            assert_eq!(one.out, batch.out, "dim={dim} sim={sim:?} mode={mode:?} ef={ef}");
                            assert_eq!(ops, vector_ops() - before);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn recall_against_brute_force() {
        let src = random_source(corpus_size(), 32, 1, 0);
        let params = HnswParams::default();
        for sim in [Similarity::Cosine, Similarity::L2, Similarity::DotProduct] {
            let g = HnswBuilder::build(32, &src, src.num_docs(), sim, params);
            assert_eq!(g.num_layers() >= 1, true);
            let mut total = 0.0;
            for q in queries(20, 32) {
                let got = g.search(&q, 10, 100, &src, sim, |_| true);
                assert_eq!(got.len(), 10);
                for w in got.windows(2) {
                    assert!(w[0].1 >= w[1].1, "scores must be descending");
                }
                total += recall(&got, &brute(&src, &q, 10, sim, |_| true));
            }
            let avg = total / 20.0;
            assert!(avg >= 0.85, "{sim:?} recall@10 = {avg}");
        }
    }

    #[test]
    fn filtered_search_returns_only_accepted_and_keeps_recall() {
        let src = random_source(corpus_size(), 32, 2, 0);
        let g = HnswBuilder::build(32, &src, src.num_docs(), Similarity::Cosine, HnswParams::default());
        let accept = |d: DocId| d % 2 == 0;
        let mut total = 0.0;
        for q in queries(20, 32) {
            let got = g.search(&q, 10, 100, &src, Similarity::Cosine, accept);
            assert_eq!(got.len(), 10, "filtered search must still fill k");
            assert!(got.iter().all(|(d, _)| accept(*d)));
            total += recall(&got, &brute(&src, &q, 10, Similarity::Cosine, accept));
        }
        assert!(total / 20.0 >= 0.8, "filtered recall = {}", total / 20.0);
    }

    #[test]
    fn docs_without_vectors_are_skipped() {
        let src = random_source(500, 16, 3, 5);
        let g = HnswBuilder::build(16, &src, src.num_docs(), Similarity::L2, HnswParams::default());
        for q in queries(5, 16) {
            let got = g.search(&q, 10, 50, &src, Similarity::L2, |_| true);
            assert_eq!(got.len(), 10);
            assert!(got.iter().all(|(d, _)| d % 5 != 0));
        }
    }

    #[test]
    fn serialize_roundtrip_is_identical() {
        let src = random_source(1000, 16, 4, 0);
        let g = HnswBuilder::build(16, &src, src.num_docs(), Similarity::Cosine, HnswParams::default());
        let bytes = g.encode();
        let g2 = HnswGraph::open(&bytes).unwrap();
        assert_eq!(g.num_layers(), g2.num_layers());
        assert_eq!(g.entry(), g2.entry());
        for q in queries(10, 16) {
            let a = g.search(&q, 10, 64, &src, Similarity::Cosine, |_| true);
            let b = g2.search(&q, 10, 64, &src, Similarity::Cosine, |_| true);
            assert_eq!(a, b);
        }
        assert!(HnswGraph::open(&bytes[..bytes.len() / 2]).is_err());
    }

    #[test]
    fn deterministic_for_seed() {
        let src = random_source(600, 8, 5, 0);
        let a = HnswBuilder::build(8, &src, src.num_docs(), Similarity::L2, HnswParams::default()).encode();
        let b = HnswBuilder::build(8, &src, src.num_docs(), Similarity::L2, HnswParams::default()).encode();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_and_single_node_graphs() {
        let empty = Mem { dim: 4, rows: vec![None, None] };
        let g = HnswBuilder::build(4, &empty, 2, Similarity::L2, HnswParams::default());
        assert!(g.search(&[0.0; 4], 5, 10, &empty, Similarity::L2, |_| true).is_empty());
        let bytes = g.encode();
        let g2 = HnswGraph::open(&bytes).unwrap();
        assert!(g2.search(&[0.0; 4], 5, 10, &empty, Similarity::L2, |_| true).is_empty());

        let one = Mem { dim: 4, rows: vec![None, Some(vec![1.0, 0.0, 0.0, 0.0])] };
        let g = HnswBuilder::build(4, &one, 2, Similarity::Cosine, HnswParams::default());
        let got = g.search(&[1.0, 0.0, 0.0, 0.0], 5, 10, &one, Similarity::Cosine, |_| true);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 1);
        assert!((got[0].1 - 1.0).abs() < 1e-6);
        assert!(g.search(&[1.0, 0.0, 0.0, 0.0], 5, 10, &one, Similarity::Cosine, |_| false).is_empty());
    }

    #[test]
    fn parallel_build_keeps_recall() {
        let src = random_source(corpus_size(), 32, 11, 7);
        for sim in [Similarity::L2, Similarity::Cosine] {
            let params = HnswParams { threads: 4, ..HnswParams::default() };
            let g = HnswBuilder::build(32, &src, src.num_docs(), sim, params);
            assert_eq!(g.num_nodes(), src.rows.iter().filter(|r| r.is_some()).count());
            let mut total = 0.0;
            for q in queries(20, 32) {
                let got = g.search(&q, 10, 100, &src, sim, |_| true);
                assert_eq!(got.len(), 10);
                assert!(got.iter().all(|(d, _)| d % 7 != 0));
                total += recall(&got, &brute(&src, &q, 10, sim, |_| true));
            }
            assert!(total / 20.0 >= 0.85, "{sim:?} parallel recall@10 = {}", total / 20.0);
        }
    }

    #[test]
    fn quantized_search_with_rescore_keeps_recall() {
        use crate::vector::quantize::QuantizedVectors;
        let src = random_source(corpus_size(), 64, 12, 0);
        for sim in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct] {
            let g = HnswBuilder::build(64, &src, src.num_docs(), sim, HnswParams::default());
            for (name, qv, oversample, want) in [
                ("int8", QuantizedVectors::int8(&src, sim), 2.0, 0.85),
                ("binary", QuantizedVectors::binary(&src, sim), 5.0, 0.7),
            ] {
                let mut total = 0.0;
                for q in queries(20, 64) {
                    let got = g.search_quantized(&q, 10, 100, oversample, &src, sim, &qv, |_| true);
                    assert_eq!(got.len(), 10);
                    for w in got.windows(2) {
                        assert!(w[0].1 >= w[1].1);
                    }
                    // Rescored: scores are exact.
                    for (d, s) in &got {
                        assert_eq!(*s, sim.score(&q, src.get(*d).unwrap()));
                    }
                    total += recall(&got, &brute(&src, &q, 10, sim, |_| true));
                }
                assert!(total / 20.0 >= want, "{sim:?} {name} recall@10 = {}", total / 20.0);
            }
        }
    }

    /// `(seed rows, merged source, seed_map)`: the seed graph covers the
    /// first `n_seed` rows of `src`; every `drop_every`-th of them is dead
    /// and left out of the merged source, which appends the remaining rows.
    fn merge_fixture(src: &Mem, n_seed: usize, drop_every: usize) -> (Mem, Mem, Vec<u32>, Vec<usize>) {
        let seed_src = Mem { dim: src.dim, rows: src.rows[..n_seed].to_vec() };
        let mut rows = Vec::new();
        let mut map = vec![u32::MAX; n_seed];
        let mut orig = Vec::new(); // merged doc -> row of `src`
        for (i, r) in src.rows.iter().enumerate() {
            if i < n_seed && drop_every > 0 && i % drop_every == 0 {
                continue;
            }
            if i < n_seed {
                map[i] = rows.len() as u32;
            }
            orig.push(i);
            rows.push(r.clone());
        }
        (seed_src, Mem { dim: src.dim, rows }, map, orig)
    }

    fn avg_recall(g: &HnswGraph, src: &Mem, sim: Similarity, nq: usize) -> f32 {
        recall_on(g, src, sim, &queries(nq, src.dim), 100)
    }

    fn recall_on(g: &HnswGraph, src: &Mem, sim: Similarity, qs: &[Vec<f32>], ef: usize) -> f32 {
        let per: Vec<f32> = qs.par_iter().map(|q| recall(&g.search(q, 10, ef, src, sim, |_| true), &brute(src, q, 10, sim, |_| true))).collect();
        per.iter().sum::<f32>() / qs.len() as f32
    }

    /// Gaussian-ish clusters (closer to real embeddings than uniform noise,
    /// where HNSW recall is low and noisy): `(source, queries)`.
    fn clustered(n: usize, nq: usize, dim: u32, seed: u64) -> (Mem, Vec<Vec<f32>>) {
        let mut rng = XorShift64Star::new(seed);
        let centers: Vec<Vec<f32>> = (0..64).map(|_| (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect()).collect();
        let mut point = |rng: &mut XorShift64Star| -> Vec<f32> {
            let c = &centers[(rng.next_u64() % centers.len() as u64) as usize];
            c.iter().map(|x| x + ((rng.next_f64() + rng.next_f64() + rng.next_f64() - 1.5) * 0.35) as f32).collect()
        };
        let rows = (0..n).map(|_| Some(point(&mut rng))).collect();
        let qs = (0..nq).map(|_| point(&mut rng)).collect();
        (Mem { dim, rows }, qs)
    }

    /// Merging with the largest input's graph as the seed matches a
    /// from-scratch build's recall and is faster. Release: 50k x 128 (40k
    /// seeded + 10k inserted), recall within 0.5pp; run with `cargo test
    /// --release -p rustic-core graph_reuse -- --nocapture` for the timings.
    #[test]
    fn graph_reuse_merge_matches_scratch_recall_and_is_faster() {
        let (n, dim, nq) = if cfg!(debug_assertions) { (3000, 16, 200) } else { (50_000, 128, 1000) };
        let n_seed = n * 4 / 5;
        let (src, qs) = clustered(n, nq, dim, 21);
        let sim = Similarity::Cosine;
        let params = HnswParams { threads: 0, ..HnswParams::default() };
        let (seed_src, merged, map, _) = merge_fixture(&src, n_seed, 0);
        let seed = HnswBuilder::build(dim, &seed_src, seed_src.num_docs(), sim, params);
        let t = std::time::Instant::now();
        let scratch = HnswBuilder::build(dim, &merged, merged.num_docs(), sim, params);
        let t_scratch = t.elapsed();
        let ins = HnswBuilder::inserts_on_this_thread();
        let t = std::time::Instant::now();
        let reused = HnswBuilder::build_from(dim, &merged, merged.num_docs(), sim, params, &seed, &map);
        let t_reuse = t.elapsed();
        assert_eq!(HnswBuilder::inserts_on_this_thread() - ins, (n - n_seed) as u64, "only the unseeded rows are inserted");
        assert_eq!(reused.num_nodes(), n);
        // Single-threaded timings too (deterministic builds).
        let p1 = HnswParams { threads: 1, ..params };
        let t = std::time::Instant::now();
        let scratch1 = HnswBuilder::build(dim, &merged, merged.num_docs(), sim, p1);
        let t_scratch1 = t.elapsed();
        let seed1 = HnswBuilder::build(dim, &seed_src, seed_src.num_docs(), sim, p1);
        let t = std::time::Instant::now();
        let reused1 = HnswBuilder::build_from(dim, &merged, merged.num_docs(), sim, p1, &seed1, &map);
        let t_reuse1 = t.elapsed();
        // Single-threaded, a seeded build replays exactly what a scratch build
        // does after its first `n_seed` inserts, so recall matches; parallel
        // builds differ by insertion-race noise (a second scratch build with
        // another seed gives the noise floor).
        let scratch_b = HnswBuilder::build(dim, &merged, merged.num_docs(), sim, HnswParams { seed: 7, ..params });
        let mut checks = Vec::new();
        for ef in [32, 100] {
            let (r_s, r_r) = (recall_on(&scratch, &merged, sim, &qs, ef), recall_on(&reused, &merged, sim, &qs, ef));
            let r_sb = recall_on(&scratch_b, &merged, sim, &qs, ef);
            let (r_s1, r_r1) = (recall_on(&scratch1, &merged, sim, &qs, ef), recall_on(&reused1, &merged, sim, &qs, ef));
            eprintln!(
                "[graph-reuse] {n} x {dim}d ({n_seed} seeded), ef={ef}: recall parallel scratch {r_s:.4} (2nd seed {r_sb:.4}) reuse {r_r:.4} | 1 thread scratch {r_s1:.4} reuse {r_r1:.4}"
            );
            checks.push((ef, r_s.min(r_sb), (r_s - r_sb).abs(), r_r, r_s1, r_r1));
        }
        // Single-threaded builds are deterministic: the seeded build must
        // match scratch closely. Parallel builds carry insertion-race noise
        // (one run saw reuse 0.9497 vs scratch 0.9569 at ef=32 while the
        // previous run passed): the reuse build is held to the lower of two
        // scratch builds within a 1-point recall loss (2 in debug), or
        // twice the scratch builds' own spread when that is larger (at
        // most 3 points).
        let slack = if cfg!(debug_assertions) { 0.02 } else { 0.005 };
        let floor = if cfg!(debug_assertions) { 0.02 } else { 0.01 };
        for (ef, r_s, spread, r_r, r_s1, r_r1) in checks {
            let noise = (2.0 * spread).max(floor).min(0.03);
            assert!(r_r >= r_s - noise, "ef {ef}: parallel reuse recall {r_r} vs scratch {r_s} (scratch spread {spread}, tolerance {noise})");
            assert!(r_r1 >= r_s1 - slack, "ef {ef}: 1-thread reuse recall {r_r1} vs scratch {r_s1}");
        }
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "[graph-reuse] build time: parallel scratch {:.0} ms reuse {:.0} ms ({:.2}x) | 1 thread scratch {:.0} ms reuse {:.0} ms ({:.2}x)",
            ms(t_scratch),
            ms(t_reuse),
            ms(t_scratch) / ms(t_reuse),
            ms(t_scratch1),
            ms(t_reuse1),
            ms(t_scratch1) / ms(t_reuse1)
        );
        if !cfg!(debug_assertions) {
            assert!(t_reuse < t_scratch && t_reuse1 < t_scratch1, "reuse not faster than scratch");
        }
    }

    #[test]
    fn graph_reuse_drops_dead_seed_points_and_repairs_links() {
        let n = corpus_size() * 2;
        let src = random_source(n, 32, 22, 0);
        for sim in [Similarity::L2, Similarity::Cosine] {
            let n_seed = n * 3 / 4;
            // 10 % of the seed is dead.
            let (seed_src, merged, map, orig) = merge_fixture(&src, n_seed, 10);
            let seed = HnswBuilder::build(32, &seed_src, seed_src.num_docs(), sim, HnswParams::default());
            let g = HnswBuilder::build_from(32, &merged, merged.num_docs(), sim, HnswParams::default(), &seed, &map);
            assert_eq!(g.num_nodes(), merged.rows.len());
            // Every node keeps outgoing links on layer 0 and no dead row appears.
            let l0 = &g.layers[0];
            for &node in &l0.nodes {
                assert!(!l0.neighbors_of(node).is_empty(), "node {node} lost every link");
                assert!(l0.neighbors_of(node).iter().all(|&x| (x as usize) < merged.rows.len() && x != node));
            }
            assert!(orig.iter().all(|&i| !(i < n_seed && i % 10 == 0)));
            let r = avg_recall(&g, &merged, sim, 30);
            let scratch = HnswBuilder::build(32, &merged, merged.num_docs(), sim, HnswParams::default());
            let rs = avg_recall(&scratch, &merged, sim, 30);
            assert!(r >= rs - 0.03, "{sim:?}: reuse with deletes recall {r} vs scratch {rs}");
            // The seed's entry point dead: another top-level node takes over.
            let mut map2 = map.clone();
            if let Some(e) = seed.entry() {
                map2[e as usize] = u32::MAX;
            }
            let rows2: Vec<Option<Vec<f32>>> =
                merged.rows.iter().enumerate().filter(|(i, _)| seed.entry().is_none_or(|e| map[e as usize] != *i as u32)).map(|(_, r)| r.clone()).collect();
            // Re-number the map onto the rows without the entry.
            let gone = seed.entry().map(|e| map[e as usize]).unwrap();
            let map2: Vec<u32> = map2.iter().map(|&x| if x == u32::MAX { x } else if x > gone { x - 1 } else { x }).collect();
            let m2 = Mem { dim: 32, rows: rows2 };
            let g2 = HnswBuilder::build_from(32, &m2, m2.num_docs(), sim, HnswParams::default(), &seed, &map2);
            assert_eq!(g2.num_nodes(), m2.rows.len());
            assert!(avg_recall(&g2, &m2, sim, 20) >= rs - 0.05);
        }
    }

    #[test]
    fn graph_reuse_is_deterministic_single_threaded_and_falls_back() {
        let src = random_source(corpus_size(), 16, 23, 0);
        let n_seed = corpus_size() / 2;
        let (seed_src, merged, map, _) = merge_fixture(&src, n_seed, 7);
        let p = HnswParams::default();
        let seed = HnswBuilder::build(16, &seed_src, seed_src.num_docs(), Similarity::L2, p);
        let a = HnswBuilder::build_from(16, &merged, merged.num_docs(), Similarity::L2, p, &seed, &map).encode();
        let b = HnswBuilder::build_from(16, &merged, merged.num_docs(), Similarity::L2, p, &seed, &map).encode();
        assert_eq!(a, b);
        // A seed built with another `m` (or an all-dead seed) is a plain build.
        let other = HnswBuilder::build(16, &seed_src, seed_src.num_docs(), Similarity::L2, HnswParams { m: 8, ..p });
        let ins = HnswBuilder::inserts_on_this_thread();
        let c = HnswBuilder::build_from(16, &merged, merged.num_docs(), Similarity::L2, p, &other, &map).encode();
        assert_eq!(HnswBuilder::inserts_on_this_thread() - ins, merged.rows.len() as u64);
        assert_eq!(c, HnswBuilder::build(16, &merged, merged.num_docs(), Similarity::L2, p).encode());
        let dead = vec![u32::MAX; n_seed];
        let d = HnswBuilder::build_from(16, &merged, merged.num_docs(), Similarity::L2, p, &seed, &dead).encode();
        assert_eq!(d, c);
        // The seeded graph round-trips through the wire format.
        let g = HnswGraph::open(&a).unwrap();
        assert_eq!(g.encode(), a);
    }

    #[test]
    fn packed_keys_order_like_floats() {
        let xs = [-3.5f32, -1.0, -0.0, 0.0, 1e-30, 0.5, 2.0, f32::INFINITY];
        for w in xs.windows(2) {
            assert!(pack(w[0], 9) <= pack(w[1], 1), "{} vs {}", w[0], w[1]);
        }
        for &x in &xs {
            assert_eq!(unpack(pack(x, 7)), (x, 7));
        }
    }

    #[test]
    fn segment_owned_payload_round_trip_and_partial_filter_fallback() {
        let src = random_source(128, 8, 17, 0);
        let params = HnswParams::default();
        let base = HnswBuilder::build(8, &src, 128, Similarity::L2, params);
        let parts: Vec<_> = (0..2).map(|group| {
            let subset = Mem { dim: 8, rows: src.rows.iter().enumerate().map(|(i, v)| {
                if i % 2 == group { v.clone() } else { None }
            }).collect() };
            HnswBuilder::build(8, &subset, 128, Similarity::L2, params)
        }).collect();
        let base_bytes = base.encode();
        let graph = base.with_payload_partitions(&parts);
        let encoded = graph.encode();
        let reopened = HnswGraph::open(&encoded).unwrap();
        assert_eq!(reopened.encode(), encoded);
        // The original graph prefix remains readable by older implementations.
        assert_eq!(&encoded[..base_bytes.len()], &base_bytes);
        let even = [0x5555_5555_5555_5555u64; 2];
        assert!(reopened.payload_covers_words(&even));
        assert!(reopened.payload_covers(&|id| id % 2 == 0));
        assert!(!reopened.payload_covers_words(&[even[0] & !1, even[1]]));
        assert!(!reopened.payload_covers(&|id| id % 2 == 0 && id != 0));
        let query = src.rows[4].as_ref().unwrap();
        let hits = reopened.search_mode(query, 10, 100, &src, Similarity::L2, |id| id % 2 == 0, FilterMode::Payload);
        assert_eq!(hits.len(), 10);
        assert_eq!(hits[0].0, 4);
        assert!(hits.iter().all(|(id, _)| id % 2 == 0));
        let independent = HnswGraph::open(&base_bytes).unwrap();
        assert!(!independent.payload_covers_words(&even));
        // Compact extra edges must reproduce the fully augmented traversal.
        let mut full = independent.with_payload_edges(&parts);
        full.payload_parts = reopened.payload.as_ref().unwrap().payload_parts.clone();
        for ef in [10, 40, 100] {
            for q in [0, 4, 7] {
                let query = src.rows[q].as_ref().unwrap();
                for group in [0, 1] {
                    let accept = |id: u32| id % 2 == group;
                    assert_eq!(reopened.search_mode(query, 10, ef, &src, Similarity::L2, accept, FilterMode::Payload),
                        full.search_mode(query, 10, ef, &src, Similarity::L2, accept, FilterMode::Payload));
                }
                assert_eq!(reopened.search(query, 10, ef, &src, Similarity::L2, |_| true),
                    independent.search(query, 10, ef, &src, Similarity::L2, |_| true));
            }
        }
        // Early integration archives with duplicated base links are readable
        // and normalize to the same compact representation on reopen.
        let mut legacy = independent.clone();
        legacy.payload = Some(Box::new(full));
        let migrated = HnswGraph::open(&legacy.encode()).unwrap();
        assert_eq!(migrated.encode(), encoded);
        // A damaged payload footer cannot be silently used.
        assert!(HnswGraph::open(&encoded[..encoded.len() - 1]).is_err());
    }
}
