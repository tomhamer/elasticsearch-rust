//! Per-index vector ownership and durable payload storage.
//!
//! ANN topology belongs to one mutable HNSW owner per vector field in a
//! shard. Text segments carry stable point IDs; text merges remap document
//! ownership without rebuilding that graph. Point liveness and refresh
//! visibility remain separate from graph readiness.
//!
//! Flush writes graph-free vector payload chunks. Background maintenance and
//! force merge compact those payloads and drop dead rows, retaining stable
//! point IDs. Explicit flat fields retain their flat scoring representation.
//! No maintenance operation creates, seeds or merges immutable HNSW graphs.
//!
//! Mutable query dispatch uses the field owner's graph and filter policy.
//! Payload readers also support exact scans and decoding retained files;
//! their presence does not authorize a legacy ANN backend. Unsupported
//! segmented-graph mappings require explicit reindexing, not an automatic
//! fallback to a different score or search implementation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

use crate::codec::directory::Directory;
use crate::codec::parse_vector_segment_file_name;
use crate::codec::vectors::PointColumn as EncodedPointColumn;
use crate::error::Result;
use crate::segment::Segment;
use crate::types::DocId;

use super::hnsw::{FilterMode, HnswParams};
use super::quantize::Quantization;
use super::segment::{by_score, top_k, SealKind, SealSpec, VectorSegment};
use super::Similarity;

/// Stable identifier of one stored vector.
pub type PointId = u64;

const NONE: u64 = u64::MAX;

fn pack(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

fn unpack(x: u64) -> Option<(u32, u32)> {
    (x != NONE).then(|| ((x >> 32) as u32, x as u32))
}

/// Merge policy of sealed payload segments. Every field's ANN index is
/// one per shard (mutable graph, shared IVF), so payload segments only hold
/// durable rows (f32, codes) and their number does not drive search cost:
/// merging them for count only rewrites rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadMerge {
    /// At most [`OptimizerPolicy::max_segments`] sealed segments, merging
    /// the smallest under the doubling rule (the segmented-graph policy).
    Capped,
    /// Lucene `TieredMergePolicy`: sizes floored at [`PAYLOAD_FLOOR_BYTES`],
    /// [`PAYLOAD_PER_TIER`] similar-sized segments per tier, merges up to
    /// [`PAYLOAD_MAX_MERGED_BYTES`]; a vector is rewritten ~log10(n) times.
    Tiered,
    /// Merge for count only above [`OptimizerPolicy::max_segments`]:
    /// segments below [`PAYLOAD_FLOOR_BYTES`] are consolidated (into
    /// floor-size groups), and above the ceiling the smallest segments are
    /// merged (see [`plan_ceiling`]); dead rows are reclaimed by vacuum
    /// (default ratio [`PAYLOAD_DELETES_ONLY_VACUUM`]).
    DeletesOnly,
}

/// Payload segments smaller than this are merged together (both policies).
pub const PAYLOAD_FLOOR_BYTES: usize = 64 << 20;
/// Segments per tier ([`PayloadMerge::Tiered`]).
pub const PAYLOAD_PER_TIER: usize = 10;
/// Largest merged payload segment ([`PayloadMerge::Tiered`]).
pub const PAYLOAD_MAX_MERGED_BYTES: usize = 5 << 30;
/// Default dead fraction that vacuums a sealed segment under
/// [`PayloadMerge::DeletesOnly`] (an explicitly configured
/// [`OptimizerPolicy::vacuum_deleted_ratio`] is always respected).
pub const PAYLOAD_DELETES_ONLY_VACUUM: f32 = 0.3;

impl PayloadMerge {
    /// `RUSTIC_VECTOR_PAYLOAD_MERGE` (`tiered`, `deletes-only`, `capped`),
    /// else [`PayloadMerge::DeletesOnly`]: in an A/B on 258k 1536-d vectors
    /// it wrote and read the least, kept the segment count bounded, and kept
    /// kNN p99 steady (tiered's merges spiked it); each vector is rewritten
    /// about once (floor consolidation) instead of log10(n) times.
    pub fn from_env() -> Self {
        static MODE: std::sync::OnceLock<PayloadMerge> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("RUSTIC_VECTOR_PAYLOAD_MERGE").as_deref() {
            Ok("tiered") => PayloadMerge::Tiered,
            Ok("capped") => PayloadMerge::Capped,
            _ => PayloadMerge::DeletesOnly,
        })
    }
}

/// Optimizer policy (Qdrant's indexing / vacuum / merge optimizers).
#[derive(Clone, Debug)]
pub struct OptimizerPolicy {
    /// Seal a field's appendable chunks into an indexed segment once they
    /// hold more than this many vectors. An upper bound on the size-based
    /// threshold ([`indexing_threshold_kb`](Self::indexing_threshold_kb));
    /// `usize::MAX` (the default) leaves it to the byte size.
    pub indexing_threshold: usize,
    /// Seal a field's appendable chunks once their `f32` data exceeds this
    /// many KiB (Qdrant's `indexing_threshold_kb`, default 20,000: ~5k
    /// vectors at 1024 dims, ~40k at 128). See
    /// [`seal_threshold`](Self::seal_threshold).
    pub indexing_threshold_kb: usize,
    /// Also seal once the chunks hold more vectors than this: past the
    /// planner's [`PlannerConfig::full_scan_threshold`] an exact scan of
    /// the tail costs more than a graph search.
    pub seal_full_scan_threshold: usize,
    /// Seal a field's appendable chunks (any size) once no chunk has been
    /// added to the field for this long, so a leftover tail is never brute
    /// forced indefinitely. `None` disables it.
    pub seal_idle: Option<std::time::Duration>,
    /// Rebuild a sealed segment when more than this fraction of it is dead.
    pub vacuum_deleted_ratio: f32,
    /// Soft cap on sealed segments per field: the merge target of
    /// [`PayloadMerge::Capped`] and the ceiling of
    /// [`PayloadMerge::DeletesOnly`] (above it the smallest segments are
    /// merged even when larger than the floor; flat and exact scans loop per
    /// segment). Sealed segments below the seal threshold are combined
    /// regardless.
    pub max_segments: usize,
    /// How sealed payload segments are merged for their number (vacuum of
    /// dead rows and force merge are separate).
    pub payload_merge: PayloadMerge,
    /// HNSW build parameters (`m` / `ef_construction` are overridden per
    /// field by [`VectorFieldOptions`]).
    pub hnsw: HnswParams,
    /// Also store an int8 quantized copy in sealed segments of fields
    /// without their own [`VectorFieldOptions::quantization`].
    pub quantize: bool,
}

impl OptimizerPolicy {
    /// Vectors of `dim` dimensions above which a field's appendable chunks
    /// are sealed: the smallest of [`indexing_threshold`](Self::indexing_threshold),
    /// [`indexing_threshold_kb`](Self::indexing_threshold_kb) worth of `f32`
    /// data and [`seal_full_scan_threshold`](Self::seal_full_scan_threshold).
    /// The dead fraction above which a sealed segment is vacuumed.
    /// The configured ratio is used as is: the deletes-only default is
    /// applied by [`Default`], never forced over a lower configured value.
    pub fn vacuum_ratio(&self) -> f32 {
        self.vacuum_deleted_ratio
    }

    /// Sealed segments smaller than this are merge candidates.
    pub fn seal_threshold(&self, dim: u32) -> usize {
        let by_bytes = self.indexing_threshold_kb.saturating_mul(1024) / (dim.max(1) as usize * 4);
        self.indexing_threshold.min(by_bytes).min(self.seal_full_scan_threshold).max(1)
    }
}

/// Per-field vector indexing policy (ES `dense_vector.index_options` +
/// `rescore_vector`), registered on the store with
/// [`VectorStore::set_field_options`]. Drives what the optimizer stores
/// when it seals a segment of the field and how searches with
/// [`PlannerConfig`] `Auto` / `None` settings score it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VectorFieldOptions {
    /// Eligible for GPU-assisted mutable construction; gated by node and license.
    #[serde(default)]
    pub gpu: bool,
    /// The quantized copy stored at seal time and scored at search time:
    /// `Int8`, `Int4`, `Binary` (built in memory, nothing stored), `None`
    /// (`f32` only) or `Auto` (the store's [`OptimizerPolicy::quantize`]).
    pub quantization: Quantization,
    /// Lucene confidence interval of the quantizer fit: `None` = Lucene's
    /// default for the dimension (`max(0.9, 1 - 1 / (dim + 1))`),
    /// `Some(0.0)` = the dynamic interval, else a value in `[0.9, 1]`.
    #[serde(default)]
    pub confidence_interval: Option<f32>,
    /// Build an HNSW graph when sealing (`false` = ES `*_flat`: sealed
    /// segments are brute-force searched).
    pub graph: bool,
    /// HNSW `m` (`None` = the optimizer policy's).
    #[serde(default)]
    pub m: Option<usize>,
    /// HNSW `ef_construction` (`None` = the optimizer policy's).
    #[serde(default)]
    pub ef_construction: Option<usize>,
    /// Default rescoring of quantized searches: the best `k * oversample`
    /// candidates are rescored with `f32`; `0` returns the quantized
    /// estimates as scores (ES `rescore_vector.oversample`).
    pub oversample: f32,
    /// Quantized estimates are ES 9's optimized scalar quantization
    /// (`ES94ScalarQuantizedVectorsFormat`) rather than Lucene 9's.
    #[serde(default)]
    pub osq: bool,
    /// Rescored candidates per `k * oversample` (`1` = exactly ES's count).
    /// ES's DiskBBQ estimates residuals against IVF cluster centroids, so its
    /// 1-bit candidates are tighter than BBQ-HNSW's; our `bbq_disk` searches
    /// the BBQ graph and rescores proportionally more for the same recall.
    #[serde(default = "one")]
    pub rescore_factor: f32,
    /// DiskBBQ (`bbq_disk`): the field has no graph; its rows are indexed by
    /// one shared IVF index ([`super::ivf_field`]).
    #[serde(default)]
    pub ivf: Option<super::ivf::IvfOptions>,
    /// Code width of a quantized shard graph's construction and traversal
    /// codes when it differs from the quantization's own (`Some(4)`: an
    /// int8 field traverses 4-bit OSQ codes; its final scores come from
    /// the f32 rescore). `None`: the quantization's width.
    #[serde(default)]
    pub traversal_bits: Option<u8>,
}

fn one() -> f32 {
    1.0
}

impl Default for VectorFieldOptions {
    /// What an unconfigured field does: the policy decides the quantization,
    /// graphs are built, and quantized searches rescore 3x.
    fn default() -> Self {
        Self { gpu: false, quantization: Quantization::Auto, confidence_interval: None, graph: true, m: None, ef_construction: None, oversample: 3.0, osq: false, rescore_factor: 1.0, ivf: None, traversal_bits: None }
    }
}

impl VectorFieldOptions {
    pub fn with_gpu(self, gpu: bool) -> Self { Self { gpu, ..self } }
    /// Plain `f32` HNSW (ES `hnsw`).
    pub fn f32_hnsw() -> Self {
        Self { quantization: Quantization::None, oversample: 0.0, ..Self::default() }
    }

    /// Quantized HNSW with ES's defaults for `q` (no rescoring for int8 /
    /// int4, 3x for binary; int4 uses the dynamic interval).
    pub fn quantized_hnsw(q: Quantization) -> Self {
        Self {
            quantization: q,
            confidence_interval: (q == Quantization::Int4).then_some(super::int4::DYNAMIC_CONFIDENCE_INTERVAL),
            oversample: if q == Quantization::Binary { 3.0 } else { 0.0 },
            ..Self::default()
        }
    }

    /// Without a graph (ES `flat`, `int8_flat`, `int4_flat`, `bbq_flat`).
    pub fn flat(self) -> Self {
        Self { graph: false, ..self }
    }

    /// With this `rescore_vector.oversample`.
    pub fn with_oversample(self, oversample: f32) -> Self {
        Self { oversample, ..self }
    }

    /// With ES 9 (OSQ) quantized estimates.
    pub fn with_osq(self, osq: bool) -> Self {
        Self { osq, ..self }
    }

    /// With this [`VectorFieldOptions::rescore_factor`].
    pub fn with_rescore_factor(self, rescore_factor: f32) -> Self {
        Self { rescore_factor, ..self }
    }

    /// With [`VectorFieldOptions::traversal_bits`].
    pub fn with_traversal_bits(self, traversal_bits: Option<u8>) -> Self {
        Self { traversal_bits, ..self }
    }

    /// With these DiskBBQ parameters (`None`: no IVF index).
    pub fn with_ivf(self, ivf: Option<super::ivf::IvfOptions>) -> Self {
        Self { ivf, ..self }
    }

    /// With this `confidence_interval`.
    pub fn with_confidence_interval(self, ci: Option<f32>) -> Self {
        Self { confidence_interval: ci, ..self }
    }

    /// With these HNSW parameters.
    pub fn with_hnsw(self, m: usize, ef_construction: usize) -> Self {
        Self { m: Some(m), ef_construction: Some(ef_construction), ..self }
    }

    /// The quantization sealed segments of this field store, given the
    /// optimizer policy (resolves `Auto`; `Binary` stores nothing).
    pub fn stored_quantization(&self, policy: &OptimizerPolicy) -> Quantization {
        // DiskBBQ's codes live in its shared index.
        if self.ivf.is_some() {
            return Quantization::None;
        }
        match self.quantization {
            Quantization::Auto => {
                if policy.quantize {
                    Quantization::Int8
                } else {
                    Quantization::None
                }
            }
            Quantization::Binary => Quantization::None,
            q => q,
        }
    }


}

impl Default for OptimizerPolicy {
    fn default() -> Self {
        Self {
            indexing_threshold: usize::MAX,
            indexing_threshold_kb: 20_000,
            seal_full_scan_threshold: PlannerConfig::default().full_scan_threshold,
            seal_idle: Some(std::time::Duration::from_secs(5)),
            // Deletes-only never merges large segments for count, so its
            // dead rows wait for vacuum at a higher default ratio.
            vacuum_deleted_ratio: if PayloadMerge::from_env() == PayloadMerge::DeletesOnly { PAYLOAD_DELETES_ONLY_VACUUM } else { 0.2 },
            // Payload segments only hold durable rows: every field's ANN
            // index (mutable graph, shared IVF) is one per shard, so search
            // cost does not grow with their number, while every merge
            // rewrites f32 payload. A cap of the core count (segmented-graph
            // era) rewrote each 1536-d vector ~8 times while indexing a
            // 1.3M-vector shard (4x write amplification, disk-bound). The
            // cap is `RUSTIC_VECTOR_MAX_SEGMENTS`, else 256 (never below the
            // core count). (Cached: `available_parallelism` reads cgroup
            // files, and an index is created per percolate request.)
            max_segments: {
                static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                *CAP.get_or_init(|| {
                    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(2);
                    std::env::var("RUSTIC_VECTOR_MAX_SEGMENTS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n >= 1)
                        .unwrap_or(256).max(cores)
                })
            },
            payload_merge: PayloadMerge::from_env(),
            // Graph builds use every core of the rayon pool.
            hnsw: HnswParams { threads: 0, ..HnswParams::default() },
            quantize: false,
        }
    }
}

/// Retired IVF file names retain admission through commit and retry.
/// The fixed allowance also covers the linked retirement queue node; queue
/// transfer and rollback splice nodes without allocating an unowned Vec.
pub type RetiredIvfFiles = Arc<crate::admission::Allocation<Vec<String>>>;

pub(crate) fn retire_ivf_files(files: Vec<String>) -> Result<RetiredIvfFiles> {
    let bytes = files.iter().fold(256u64, |n, f| n.saturating_add(f.capacity() as u64).saturating_add(128));
    let memory = crate::admission::shared().reserve(bytes, "retired IVF file names")?;
    Ok(Arc::new(crate::admission::Allocation::new(files, memory)))
}

/// See [`VectorStore::commit_point`].
#[derive(Default)]
pub struct VectorCommitPoint {
    pub segments: Vec<u32>,
    pub obsolete: Vec<u32>,
    pub next_segment_id: u32,
    pub next_point_id: u64,
    /// The DiskBBQ indexes' files.
    pub ivf: Vec<super::ivf_field::IvfFieldManifest>,
    /// DiskBBQ files replaced since the last commit point.
    pub obsolete_files: std::collections::LinkedList<RetiredIvfFiles>,
    pub(crate) _manifest_memory: Option<crate::admission::Reservation>,
}

/// What one [`VectorStore::optimize`] call did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptimizeReport {
    /// Appendable chunk groups sealed into an indexed segment.
    pub sealed: usize,
    /// Sealed segments rebuilt without their dead points.
    pub vacuumed: usize,
    /// Merge-optimizer combines.
    pub merged: usize,
    /// Segments dropped because nothing in them was alive (no build).
    pub dropped: usize,
    /// Vectors inserted into new graphs (a merge seeded with an existing
    /// graph counts only the vectors it inserted; graph-less seals count
    /// every row).
    pub vectors_indexed: usize,
    /// DiskBBQ index steps (train, insert, split, drop, compact).
    pub ivf_steps: usize,
}

/// Cheap optimizer diagnostics. Legacy counts cover vector-store segments;
/// mutable owners are reported separately because they have no such segments.
/// Searchable legacy vectors still in document memtables are not included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptimizerBacklog {
    /// Registered mutable fields need periodic reclamation/compaction even
    /// with zero vectors or legacy segments (retired allocations may remain).
    pub mutable_fields: usize,
    pub unsealed_segments: usize,
    /// Stored rows, including hidden rows and rows awaiting deletion cleanup.
    pub unsealed_vectors: usize,
    pub searchable_unsealed_vectors: usize,
    pub sealed_segments: usize,
    pub graph_segments: usize,
}

/// Query planner thresholds for filtered search, and how graph searches
/// score.
#[derive(Clone, Copy, Debug)]
pub struct PlannerConfig {
    /// Exact search when a segment has at most this many matching points
    /// (capped by [`full_scan_threshold_kb`](Self::full_scan_threshold_kb),
    /// see [`full_scan_limit`](Self::full_scan_limit)).
    pub full_scan_threshold: usize,
    /// ...and at most this many KiB of their `f32` vectors (Qdrant's
    /// `full_scan_threshold`, default 10,000 KB): a brute-force scan costs
    /// per dimension, so 1024-dim segments go through the graph past ~2,500
    /// matching vectors instead of 10,000.
    pub full_scan_threshold_kb: usize,
    /// ...or when matching points are at most this fraction of the segment.
    pub full_scan_fraction: f32,
    /// Quantized scoring (graph traversal and, on sealed segments, brute
    /// force). Default `Auto`: the field's [`VectorFieldOptions`], which for
    /// an unconfigured field means the segment's stored copy (int4 / int8)
    /// or `f32`. `None` forces full precision everywhere.
    pub quantization: Quantization,
    /// With quantization: rescore the best `k * oversample` candidates with
    /// exact `f32`; `Some(0.0)` returns the quantized estimates as scores.
    /// `None` = the field's [`VectorFieldOptions::oversample`] (ES
    /// `rescore_vector.oversample` of the mapping, overridden per query).
    pub oversample: Option<f32>,
    /// Filtered graph searches whose matching fraction is below this use
    /// the two-hop traversal (only accepted nodes are scored).
    pub two_hop_fraction: f32,
    /// DiskBBQ: the fraction of the field's vectors to visit (the knn
    /// `visit_percentage / 100`; `Some(0)` = ES's dynamic ratio). `None` =
    /// the field's `default_visit_percentage`.
    pub visit_ratio: Option<f32>,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            full_scan_threshold: 10_000,
            full_scan_threshold_kb: 10_000,
            full_scan_fraction: 0.05,
            quantization: Quantization::Auto,
            oversample: None,
            two_hop_fraction: 0.3,
            visit_ratio: None,
        }
    }
}

impl PlannerConfig {
    /// Always exact (`f32` brute force, no quantization).
    pub fn exact() -> Self {
        Self { full_scan_threshold: usize::MAX, full_scan_threshold_kb: usize::MAX, full_scan_fraction: 1.0, quantization: Quantization::None, ..Self::default() }
    }

    /// Most matching points of a `dim`-dimensional segment searched
    /// exactly: [`full_scan_threshold`](Self::full_scan_threshold) points,
    /// capped at [`full_scan_threshold_kb`](Self::full_scan_threshold_kb)
    /// of `f32` vectors.
    pub fn full_scan_limit(&self, dim: u32) -> usize {
        // An unbounded byte limit (it overflows) does not cap the count.
        let by_bytes = self.full_scan_threshold_kb.checked_mul(1024).map_or(usize::MAX, |b| b / (dim.max(1) as usize * 4));
        self.full_scan_threshold.min(by_bytes)
    }

    /// With this rescore oversample (`0` = none) instead of the field's.
    pub fn with_oversample(self, oversample: f32) -> Self {
        Self { oversample: Some(oversample), ..self }
    }

    /// With this DiskBBQ visit ratio (`0` = dynamic) instead of the field's.
    pub fn with_visit_ratio(self, visit_ratio: f32) -> Self {
        Self { visit_ratio: Some(visit_ratio), ..self }
    }
}

/// How the planner executed one search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Segments searched exactly.
    pub exact_segments: usize,
    /// Segments searched through the graph.
    pub graph_segments: usize,
    /// Graph searches that returned < k and fell back to exact.
    pub fallbacks: usize,
    /// Vector distance / similarity computations (traversal, brute force
    /// and `f32` rescoring): ES profile `vector_operations_count`.
    pub vector_ops: u64,
}

impl std::ops::AddAssign for SearchStats {
    fn add_assign(&mut self, o: Self) {
        self.exact_segments += o.exact_segments;
        self.graph_segments += o.graph_segments;
        self.fallbacks += o.fallbacks;
        self.vector_ops += o.vector_ops;
    }
}

/// One search hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointHit {
    pub point: PointId,
    /// Owning text segment id.
    pub segment: u32,
    pub doc: DocId,
    pub score: f32,
}

/// Points an O(points) scan of the point table reads between fair hand-offs
/// of its read guard to waiting writers.
const TABLE_SCAN_BUMP: usize = 4096;

/// Points per [`OwnerColumn`] chunk.
const OWNER_CHUNK: usize = 1 << 10;

struct OwnerPage {
    values: Vec<AtomicU64>,
    // Covers the complete point page. Snapshot readers can outlive removal
    // of the other columns, so retaining this charge is conservative.
    _memory: Option<crate::admission::Reservation>,
}
impl std::ops::Deref for OwnerPage {
    type Target = [AtomicU64];
    fn deref(&self) -> &Self::Target { &self.values }
}

#[derive(Default)]
struct OwnerDirectory {
    pages: BTreeMap<usize, Arc<OwnerPage>>,
    // Each immutable directory generation owns its own metadata reservation.
    // Readers can retain arbitrarily many different generations.
    _memory: Option<crate::admission::Reservation>,
}
impl std::ops::Deref for OwnerDirectory {
    type Target = BTreeMap<usize, Arc<OwnerPage>>;
    fn deref(&self) -> &Self::Target { &self.pages }
}

/// The point-owner column: fixed-size atomic chunks that never move once
/// allocated. Writers store under the table write lock; a query takes an
/// [`OwnerSnapshot`] (an `Arc` clone of the chunk directory) once and then
/// resolves owners with no lock at all.
#[derive(Default)]
struct OwnerColumn {
    len: usize,
    chunks: Arc<OwnerDirectory>,
    /// Set when a page was retired while a query held the directory (it
    /// could not be trimmed in place), cleared once trimmed: a dropping
    /// query snapshot looks for pages to collect only then, not on every
    /// query (`OwnerSnapshot::drop`, `PointTable::collect_retired_pages`).
    stale: Arc<AtomicBool>,
}

impl OwnerColumn {
    fn len(&self) -> usize {
        self.len
    }
    /// Bytes of the allocated chunks.
    fn heap_bytes(&self) -> usize {
        self.chunks.len() * (OWNER_CHUNK * std::mem::size_of::<AtomicU64>() + 256)
    }
    /// Materialize only the supplied page, retaining guards with the
    /// immutable directory and atomic page independently.
    fn install_page(&mut self, n: usize, resident: &PointColumn<u64>, memory: Option<crate::admission::Reservation>, page_memory: Option<crate::admission::Reservation>) {
        let page = (n - 1) / OWNER_CHUNK;
        let chunks = self.chunks.iter().filter(|(id, _)| resident.pages.contains_key(id))
            .map(|(&id, value)| (id, value.clone())).collect::<BTreeMap<_,_>>();
        let mut directory = OwnerDirectory { pages: chunks, _memory: memory };
        directory.pages.insert(page, Arc::new(OwnerPage {
            values: (0..OWNER_CHUNK).map(|_| AtomicU64::new(NONE)).collect(), _memory: page_memory,
        }));
        self.chunks = Arc::new(directory);
        self.len = self.len.max((page + 1).saturating_mul(OWNER_CHUNK));
    }
    fn get(&self, p: usize) -> Option<u64> {
        self.chunks.get(&(p / OWNER_CHUNK)).map(|page| page[p % OWNER_CHUNK].load(AtomicOrdering::Acquire))
    }
    /// Callers hold the table write lock and `p < len`.
    fn set(&mut self, p: usize, owner: u64) {
        debug_assert!(p < self.len);
        self.chunks[&(p / OWNER_CHUNK)][p % OWNER_CHUNK].store(owner, AtomicOrdering::Release);
    }
    #[cfg(feature = "mutable-hnsw-poc")]
    fn snapshot(&self, memory: Arc<Mutex<Option<crate::admission::Reservation>>>, table: std::sync::Weak<RwLock<PointTable>>) -> OwnerSnapshot {
        OwnerSnapshot { len: self.len, chunks: Some(self.chunks.clone()), table, stale: self.stale.clone(), _memory: memory }
    }
}

impl Clone for OwnerColumn {
    /// A deep copy (a captured read store owns its table).
    fn clone(&self) -> Self {
        let chunks = self.chunks.iter().map(|(&id, chunk)| (id, Arc::new(OwnerPage { values: chunk.iter()
            .map(|owner| AtomicU64::new(owner.load(AtomicOrdering::Acquire))).collect(), _memory: None }))).collect();
        Self { len: self.len, chunks: Arc::new(OwnerDirectory { pages: chunks, _memory: None }), stale: Arc::default() }
    }
}

/// [`OwnerColumn`] as of one query: lock-free reads of the live owners of
/// the points that existed when it was taken.
#[cfg(feature = "mutable-hnsw-poc")]
struct OwnerSnapshot {
    len: usize,
    chunks: Option<Arc<OwnerDirectory>>,
    table: std::sync::Weak<RwLock<PointTable>>,
    /// The table's [`OwnerColumn::stale`].
    stale: Arc<AtomicBool>,
    _memory: Arc<Mutex<Option<crate::admission::Reservation>>>,
}

#[cfg(feature = "mutable-hnsw-poc")]
impl Drop for OwnerSnapshot {
    fn drop(&mut self) {
        // Release this directory first, then opportunistically collect pages
        // whose last reader just left. Never wait behind a writer on drop.
        drop(self.chunks.take());
        // Nothing to collect (the common case): no lock, no page scan.
        if !self.stale.load(AtomicOrdering::Acquire) { return; }
        if let Some(table) = self.table.upgrade() {
            if let Some(mut table) = table.try_write() { table.collect_retired_pages(); }
        }
    }
}

/// Stable-ID page directory: holes do not allocate intervening entries.
#[derive(Clone)]
struct PointColumn<T> {
    pages: BTreeMap<usize, Vec<T>>,
    empty: T,
    page_len: usize,
}
impl<T: Clone> PointColumn<T> {
    fn new(empty: T, page_len: usize) -> Self { Self { pages: BTreeMap::new(), empty, page_len } }
    fn ensure(&mut self, index: usize) {
        let page_len = self.page_len;
        let empty = &self.empty;
        self.pages.entry(index / page_len).or_insert_with(|| vec![empty.clone(); page_len]);
    }
    fn get(&self, index: usize) -> Option<&T> { self.pages.get(&(index / self.page_len)).map(|p| &p[index % self.page_len]) }
    fn capacity(&self) -> usize { self.pages.len() * self.page_len }
    fn is_empty(&self) -> bool { self.pages.is_empty() }
}
impl<T: Clone> std::ops::Index<usize> for PointColumn<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T { self.get(index).unwrap_or(&self.empty) }
}
impl<T: Clone> std::ops::IndexMut<usize> for PointColumn<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.pages.get_mut(&(index / self.page_len)).expect("point page admitted before publication")[index % self.page_len]
    }
}

/// Sparse, stable-ID per-point state.
#[derive(Clone)]
struct PointTable {
    owner: OwnerColumn,
    loc: PointColumn<u64>,
    alive: PointColumn<u64>,
    /// Exact live counts updated with liveness/location under the table lock.
    /// Stats must not scan every vector while writers wait for that lock.
    live_by_segment: BTreeMap<u32, usize>,
    /// Exact kept counts ([`PointTable::is_kept`]: alive or hidden) per
    /// segment of the point's location, kept with the bits like
    /// `live_by_segment`: the optimizer's census of a segment
    /// ([`VectorStore::alive_count`]) reads it instead of scanning the
    /// segment's points (every maintenance pass, every segment).
    kept_by_segment: BTreeMap<u32, usize>,
    /// Points of live docs in text segments not yet visible to searches
    /// (sealed before an ES refresh published them): kept by the optimizer
    /// like alive points, never returned by a search.
    hidden: PointColumn<u64>,
    pending: PointColumn<u64>,
    /// Reconstructed from owning document segments at open/flush/text merge.
    payload: PointColumn<Option<Arc<crate::segment::VectorPayloadRow>>>,
    /// The live table's dense arrays on node-wide accounting (a clone,
    /// a retained capture's copy, is charged by the capture instead).
    charged: TableCharge,
    admission: TableAdmission,
    /// Bumped whenever a point turns alive, dead, hidden or released: a
    /// maintenance census found nothing to do only for one version.
    kept_version: u64,
    owner_cache: OwnerCacheSlot,
}

impl Default for PointTable {
    fn default() -> Self {
        Self { owner: OwnerColumn::default(), loc: PointColumn::new(NONE, OWNER_CHUNK),
            alive: PointColumn::new(0, OWNER_CHUNK / 64), hidden: PointColumn::new(0, OWNER_CHUNK / 64), pending: PointColumn::new(0, OWNER_CHUNK),
            payload: PointColumn::new(None, OWNER_CHUNK), live_by_segment: BTreeMap::new(), kept_by_segment: BTreeMap::new(),
            charged: TableCharge::default(), admission: TableAdmission::default(), kept_version: 0, owner_cache: OwnerCacheSlot::default() }
    }
}

/// Shared with live owner snapshots: their atomic chunks may outlive the store.
struct TableAdmission(Arc<Mutex<Option<crate::admission::Reservation>>>);
impl Default for TableAdmission {
    fn default() -> Self { Self(Arc::new(Mutex::new(None))) }
}
impl Clone for TableAdmission {
    // PointTable::clone is a deep copy, admitted explicitly by capture_read_store.
    fn clone(&self) -> Self { Self::default() }
}

/// [`PointTable::charged`]: resized as the table grows, released on drop;
/// a clone starts uncharged.
#[derive(Default)]
struct TableCharge(u64);

impl TableCharge {
    fn set(&mut self, bytes: u64) {
        use crate::accounting::{add, sub, Category};
        if bytes > self.0 {
            add(Category::VectorPoints, bytes - self.0);
        } else {
            sub(Category::VectorPoints, self.0 - bytes);
        }
        self.0 = bytes;
    }
}

impl Clone for TableCharge {
    fn clone(&self) -> Self {
        TableCharge(0)
    }
}

impl Drop for TableCharge {
    fn drop(&mut self) {
        crate::accounting::sub(crate::accounting::Category::VectorPoints, self.0);
    }
}

impl PointTable {
    /// Bytes of the dense arrays, from their capacities (O(1): the payload
    /// rows' strings are not walked).
    fn dense_bytes(&self) -> usize {
        self.owner.heap_bytes() + self.owner.chunks.len() * 1024 + (self.loc.capacity() + self.alive.capacity() + self.hidden.capacity() + self.pending.capacity()) * 8
            + self.payload.capacity() * std::mem::size_of::<Option<Arc<crate::segment::VectorPayloadRow>>>()
    }

    /// Heap bytes a retained capture's copy of the table holds (a deep
    /// copy, owner column included).
    fn heap_bytes(&self) -> usize {
        let words = self.owner.heap_bytes() / 8 + self.loc.capacity() + self.alive.capacity() + self.hidden.capacity() + self.pending.capacity();
        // Payload rows are shared immutable owners, not copied by a PIT.
        words * 8 + self.owner.chunks.len() * 1024 + self.payload.capacity() * std::mem::size_of::<Option<Arc<crate::segment::VectorPayloadRow>>>() + (self.live_by_segment.len() + self.kept_by_segment.len()) * 32
    }

    /// Admit a missing page before point IDs/owners can publish. The bound
    /// includes all column tree nodes and the owner directory copy.
    fn ensure_admitted(&mut self, n: usize) -> Result<()> {
        if n == 0 || self.loc.pages.contains_key(&((n - 1) / OWNER_CHUNK)) { return Ok(()); }
        let memory = self.admission.0.clone();
        let mut memory = memory.lock();
        if memory.is_none() { *memory = Some(crate::admission::shared().reserve(0, "vector point table budget")?); }
        let budget = memory.as_ref().expect("point admission installed");
        let page = budget.reserve_related((OWNER_CHUNK as u64) * 64 + 4096, "vector point page")?;
        let directory = budget.reserve_related(
            ((self.loc.pages.len() + 1) as u64).saturating_mul(256).saturating_add(128), "vector owner directory generation")?;
        self.owner.install_page(n, &self.loc, Some(directory), Some(page));
        self.ensure_columns(n);
        Ok(())
    }

    #[cfg(test)]
    fn ensure(&mut self, n: usize) {
        if n == 0 { return; }
        if !self.loc.pages.contains_key(&((n - 1) / OWNER_CHUNK)) { self.owner.install_page(n, &self.loc, None, None); }
        self.ensure_columns(n);
    }
    fn ensure_columns(&mut self, n: usize) {
        self.loc.ensure(n - 1);
        self.payload.ensure(n - 1);
        self.pending.ensure(n - 1);
        self.alive.ensure((n - 1) / 64);
        self.hidden.ensure((n - 1) / 64);
        self.charged.set(self.dense_bytes() as u64);
    }
    fn release_pending(&mut self, first: u64, count: usize) {
        for p in first..first + count as u64 {
            let slot = &mut self.pending[p as usize];
            debug_assert!(*slot > 0); *slot -= 1;
        }
        if count != 0 {
            for page in first as usize / OWNER_CHUNK..=(first as usize + count - 1) / OWNER_CHUNK { self.retire_empty_page(page); }
        }
    }
    fn collect_retired_pages(&mut self) {
        // Pages retired while a query held the directory are the only ones
        // to collect (`OwnerColumn::stale`).
        if !self.owner.stale.load(AtomicOrdering::Acquire) { return; }
        if !self.owner.chunks.pages.keys().any(|page| !self.loc.pages.contains_key(page)) {
            self.owner.stale.store(false, AtomicOrdering::Release);
            self.charged.set(self.dense_bytes() as u64);
            return;
        }
        // The query owner cache holds the directory weakly; drop it so a
        // directory no query holds can be trimmed in place.
        self.owner_cache.clear();
        if let Some(directory) = Arc::get_mut(&mut self.owner.chunks) {
            directory.pages.retain(|page, _| self.loc.pages.contains_key(page));
            self.owner.stale.store(false, AtomicOrdering::Release);
        }
        self.charged.set(self.dense_bytes() as u64);
    }
    fn retire_empty_page(&mut self, page: usize) {
        let Some(locations) = self.loc.pages.get(&page) else { return };
        if locations.iter().any(|v| *v != NONE) || self.pending.pages[&page].iter().any(|v| *v != 0) { return; }
        // In-flight queries see retired owners disappear, and keep the page's
        // charge until their directory dies. No allocation in cancellation.
        if let Some(owners) = self.owner.chunks.get(&page) {
            for owner in owners.iter() { owner.store(NONE, AtomicOrdering::Release); }
        }
        self.loc.pages.remove(&page); self.payload.pages.remove(&page);
        self.pending.pages.remove(&page); self.alive.pages.remove(&page); self.hidden.pages.remove(&page);
        self.owner_cache.clear();
        match Arc::get_mut(&mut self.owner.chunks) {
            Some(directory) => { directory.pages.remove(&page); }
            // A query holds the directory: collected when it lets go.
            None => self.owner.stale.store(true, AtomicOrdering::Release),
        }
        self.charged.set(self.dense_bytes() as u64);
    }
    fn alive_word(&self, start: i64) -> u64 {
        if start < 0 { return (0..64).fold(0, |bits, i| bits | (u64::from(start + i >= 0 && self.is_alive((start + i) as u64)) << i)); }
        let word = start as usize / 64;
        let shift = start as usize % 64;
        let lo = self.alive[word] >> shift;
        if shift == 0 { lo } else { lo | (self.alive[word + 1] << (64 - shift)) }
    }
    #[inline]
    fn is_hidden(&self, p: PointId) -> bool {
        let i = p as usize;
        self.hidden[i / 64] >> (i % 64) & 1 == 1
    }
    fn set_hidden(&mut self, p: PointId, v: bool) {
        let was_hidden = self.is_hidden(p);
        if was_hidden != v {
            self.kept_version = self.kept_version.wrapping_add(1);
        }
        let alive = self.is_alive(p);
        let i = p as usize;
        if v {
            self.hidden[i / 64] |= 1 << (i % 64);
        } else {
            self.hidden[i / 64] &= !(1 << (i % 64));
        }
        if !alive && was_hidden != v {
            if let Some((segment, _)) = self.loc(p) { self.adjust_kept(segment, v); }
        }
    }
    /// A copy of the kept bits ([`PointTable::is_kept`]), for scans that
    /// must not hold the table.
    fn kept_words(&self) -> Result<crate::admission::Allocation<BTreeMap<usize, Vec<u64>>>> {
        let count = self.alive.pages.len().saturating_add(self.hidden.pages.len());
        let bytes = count.saturating_mul(self.alive.page_len.saturating_mul(8).saturating_add(128));
        let memory = crate::admission::shared().reserve(bytes as u64, "IVF sparse kept snapshot")?;
        let mut pages = self.alive.pages.clone();
        for (&id, hidden) in &self.hidden.pages {
            let page = pages.entry(id).or_insert_with(|| vec![0; self.alive.page_len]);
            for (a, h) in page.iter_mut().zip(hidden) { *a |= h; }
        }
        Ok(crate::admission::Allocation::new(pages, memory))
    }
    /// Alive or hidden: the optimizer must keep the point's vector.
    #[inline]
    fn is_kept(&self, p: PointId) -> bool {
        self.is_alive(p) || self.is_hidden(p)
    }
    #[inline]
    fn is_alive(&self, p: PointId) -> bool {
        let i = p as usize;
        self.alive[i / 64] >> (i % 64) & 1 == 1
    }
    fn set_alive(&mut self, p: PointId, v: bool) {
        let was_alive = self.is_alive(p);
        if was_alive == v { return; }
        self.kept_version = self.kept_version.wrapping_add(1);
        if let Some((segment, _)) = self.loc(p) {
            self.adjust_live(segment, v);
            // Kept changes only when the point is not hidden.
            if !self.is_hidden(p) { self.adjust_kept(segment, v); }
        }
        let i = p as usize;
        if v {
            self.alive[i / 64] |= 1 << (i % 64);
        } else {
            self.alive[i / 64] &= !(1 << (i % 64));
        }
    }
    fn adjust_kept(&mut self, segment: u32, add: bool) {
        if add {
            *self.kept_by_segment.entry(segment).or_default() += 1;
        } else {
            let count = self.kept_by_segment.get_mut(&segment).expect("kept vector count missing");
            *count -= 1;
            if *count == 0 { self.kept_by_segment.remove(&segment); }
        }
    }
    fn adjust_live(&mut self, segment: u32, add: bool) {
        if add {
            *self.live_by_segment.entry(segment).or_default() += 1;
        } else {
            let count = self.live_by_segment.get_mut(&segment).expect("live vector count missing");
            *count -= 1;
            if *count == 0 { self.live_by_segment.remove(&segment); }
        }
    }
    fn set_loc(&mut self, p: PointId, location: u64) {
        let old = self.loc[p as usize];
        if old == location { return; }
        if self.is_alive(p) {
            if let Some((segment, _)) = unpack(old) { self.adjust_live(segment, false); }
            if let Some((segment, _)) = unpack(location) { self.adjust_live(segment, true); }
        }
        if self.is_kept(p) {
            if let Some((segment, _)) = unpack(old) { self.adjust_kept(segment, false); }
            if let Some((segment, _)) = unpack(location) { self.adjust_kept(segment, true); }
        }
        self.loc[p as usize] = location;
    }
    fn owner(&self, p: PointId) -> Option<(u32, DocId)> {
        unpack(self.owner.get(p as usize)?)
    }
    fn loc(&self, p: PointId) -> Option<(u32, u32)> {
        unpack(*self.loc.get(p as usize)?)
    }
}

/// The points of a segment.
fn n_points(s: &VectorSegment) -> usize {
    s.points().len()
}

/// [`PointTable::is_kept`] along ascending points, each liveness page
/// looked up once.
struct KeptCursor<'t> {
    t: &'t PointTable,
    page: usize,
    alive: Option<&'t [u64]>,
    hidden: Option<&'t [u64]>,
}

impl<'t> KeptCursor<'t> {
    fn new(t: &'t PointTable) -> Self {
        Self { t, page: usize::MAX, alive: None, hidden: None }
    }

    #[inline]
    fn is_kept(&mut self, p: PointId) -> bool {
        let word = p as usize / 64;
        let len = self.t.alive.page_len;
        let page = word / len;
        if page != self.page {
            self.page = page;
            self.alive = self.t.alive.pages.get(&page).map(Vec::as_slice);
            self.hidden = self.t.hidden.pages.get(&page).map(Vec::as_slice);
        }
        let (at, bit) = (word % len, p % 64);
        self.alive.is_some_and(|w| w[at] >> bit & 1 == 1) || self.hidden.is_some_and(|w| w[at] >> bit & 1 == 1)
    }
}

/// Point-owner lookups for one mutable kNN query, from an owner-column
/// snapshot taken once: no lookup takes a lock, so nothing holds the point
/// table across the graph traversal or exact scan, whose row reads
/// ([`StoreRowsSnapshot`]) take `state` then `table` themselves. Holding the
/// table across them inverted that order (a writer holding `state` waits
/// for `table`) and re-entered a fair lock behind a queued writer, hanging
/// the query and the shard's writers and merges.
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) struct MutablePointOwners {
    owners: OwnerSnapshot,
}
#[cfg(feature = "mutable-hnsw-poc")]
impl MutablePointOwners {
    /// The point's owner now (entries are live: consecutive lookups may see
    /// newer owners); callers resolve against their reader view and treat a
    /// vanished owner as not matching.
    pub(crate) fn owner(&self, point: PointId) -> Option<(u32, DocId)> {
        let p = usize::try_from(point).ok().filter(|p| *p < self.owners.len)?;
        unpack(self.owners.chunks.as_ref()?.get(&(p / OWNER_CHUNK))?[p % OWNER_CHUNK].load(AtomicOrdering::Acquire))
    }
}

#[derive(Default)]
struct State {
    segments: BTreeMap<u32, Arc<VectorSegment>>,
    /// Replaced/dropped segment ids whose files can go after the next commit.
    obsolete: Vec<u32>,
    /// Per-field indexing policy (see [`VectorFieldOptions`]).
    field_options: BTreeMap<u16, VectorFieldOptions>,
    /// When a chunk was last appended per field (the idle-seal clock). A
    /// field without an entry (chunks found at open) counts as idle.
    last_append: BTreeMap<u16, std::time::Instant>,
    /// Each DiskBBQ field's current shared index.
    ivf: BTreeMap<u16, Arc<super::ivf_field::IvfState>>,
    /// Committed DiskBBQ indexes not opened for want of memory: their
    /// fields are searched flat, commits keep listing their files, and the
    /// next maintenance step retries the open.
    ivf_deferred: BTreeMap<u16, super::ivf_field::IvfFieldManifest>,
    /// Per deferred field, when maintenance may retry its open and the
    /// failed retries so far (the delay doubles, up to a minute).
    ivf_open_retry: BTreeMap<u16, (std::time::Instant, u32)>,
    /// DiskBBQ files replaced since the last commit point.
    obsolete_files: std::collections::LinkedList<RetiredIvfFiles>,
    ivf_memory: Option<crate::admission::Reservation>,
}

impl State {
    fn insert_ivf(&mut self, field: u16, state: Arc<super::ivf_field::IvfState>) -> Result<()> {
        let count = self.ivf.len().saturating_add(usize::from(!self.ivf.contains_key(&field)));
        let memory = crate::admission::shared().reserve((count as u64).saturating_mul(512).saturating_add(512), "IVF field registry")?;
        self.ivf.insert(field, state);
        self.ivf_memory = Some(memory);
        Ok(())
    }
}

/// The vector store as the durable f32 rows of its mutable fields
/// ([`super::hnsw::RawRows`]): a quantized field keeps only codes in memory
/// for rows sealed into the store's mapped segments.
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) struct StoreRows(pub(crate) std::sync::Weak<VectorStore>);

#[cfg(feature = "mutable-hnsw-poc")]
impl super::hnsw::RawRows for StoreRows {
    fn snapshot(&self) -> Result<Option<Box<dyn super::hnsw::RawRowsSnapshot>>> {
        if self.0.upgrade().is_none(){return Ok(None);}
        let memory=crate::admission::shared().reserve((std::mem::size_of::<StoreRowsSnapshot>()+32) as u64,"durable vector read view")?;
        Ok(Some(Box::new(StoreRowsSnapshot { store: self.0.clone(), held: Mutex::new(BTreeMap::new()), failed:AtomicBool::new(false), _memory:Some(memory) })))
    }
}

#[cfg(test)]
thread_local! {
    /// Test seam: runs once on this thread after the next DiskBBQ step
    /// is built, before it is installed.
    pub(crate) static AFTER_IVF_BUILD: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
    /// Test seam: IVF index opens on this thread fail for want of memory.
    pub(crate) static FAIL_IVF_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Test seam: retrains of this thread are too large for one step.
    pub(crate) static SKIP_IVF_RETRAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Test seam: retrains this thread considered.
    pub(crate) static IVF_RETRAIN_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Test seam: IVF index opens attempted on this thread.
    pub(crate) static IVF_OPEN_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Test seam: runs once on this thread right after the next segment
    /// swap of [`VectorStore::replace`].
    pub(crate) static AFTER_SWAP: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

/// A [`StoreRows`] read: the segments it returned rows from stay alive
/// (and mapped) as long as it does. Held once per segment: a retained
/// (PIT) view reads rows through one snapshot for its whole lifetime, so a
/// per-row entry grew without bound; per segment it is bounded by the
/// segments the view's store holds.
#[cfg(feature = "mutable-hnsw-poc")]
struct StoreRowsSnapshot {
    /// Weak: a retained (PIT) store's own captured graph reads through it
    /// without a reference cycle; the segments rows came from stay held.
    store: std::sync::Weak<VectorStore>,
    held: Mutex<BTreeMap<u32, crate::admission::Allocation<Arc<VectorSegment>>>>,
    failed: AtomicBool,
    _memory: Option<crate::admission::Reservation>,
}

#[cfg(feature = "mutable-hnsw-poc")]
impl super::hnsw::RawRowsSnapshot for StoreRowsSnapshot {
    fn execution_result(&self)->Result<()> {
        if self.failed.load(AtomicOrdering::Acquire) {Err(crate::error::RusticError::Memory("durable vector read pin admission refused".into()))}else{Ok(())}
    }
    fn row(&self, key: u64) -> Option<&[f32]> {
        let store = self.store.upgrade()?;
        // Lock-order seam: no caller may hold the point table while it
        // reads rows (the read below takes state, then table).
        #[cfg(test)]
        if store.row_lock_probe.load(AtomicOrdering::Acquire) {
            store.row_lock_probed.fetch_add(1, AtomicOrdering::AcqRel);
            if store.table.try_write().is_none() {
                store.row_lock_violations.fetch_add(1, AtomicOrdering::AcqRel);
            }
        }
        let segment = {
            // Same order as every reader: state, then table.
            let state = store.state.read();
            let table = store.table.read();
            let (id, ord) = table.loc(key)?;
            (state.segments.get(&id)?.clone(), ord)
        };
        let (segment, ord) = segment;
        let pointer: *const VectorSegment = Arc::as_ptr(&segment);
        // A segment id is never reused within a store: an entry already
        // held is this very segment (`pointer` stays valid either way).
        {
            let mut held=self.held.lock();
            if !held.contains_key(&segment.id) {
                // A BTree entry releases backing with its owner; unlike a
                // HashMap, it does not retain uncharged high-water capacity.
                let memory=match crate::admission::shared().reserve(2048,"durable vector segment pin") {
                    Ok(memory)=>memory,
                    Err(_)=>{self.failed.store(true,AtomicOrdering::Release);return None;}
                };
                held.insert(segment.id,crate::admission::Allocation::new(segment,memory));
            }
        }
        // SAFETY: the segment is kept alive in `held` until this snapshot
        // drops, so the row (in its mapping or owned column) outlives the
        // returned borrow of `self`.
        Some(unsafe { &*pointer }.vector(ord))
    }
}

/// [`VectorStore::mutable_pressure`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MutablePressure {
    pub pending_repairs: u64,
    pub retired_slots: u64,
    pub retired_adjacency_bytes: u64,
}

/// One field's checkpoint at its capture point.
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) enum PendingFieldCheckpoint {
    /// Field id, construction threads, the pending capture.
    Capture(u16, usize, super::hnsw::PendingMutableCheckpoint),
    /// Unchanged since the last commit: its entry and mutation generation.
    Unchanged(MutableFieldManifest, u64),
}

/// Copy checkpoints begun with [`VectorStore::begin_mutable_checkpoint`].
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) fn finish_mutable_checkpoint(pending: Vec<PendingFieldCheckpoint>) -> Result<Vec<CapturedFieldCheckpoint>> {
    pending.into_iter().map(|pending| Ok(match pending {
        PendingFieldCheckpoint::Capture(field, threads, capture) => CapturedFieldCheckpoint::Captured(field, threads, capture.finish()?),
        PendingFieldCheckpoint::Unchanged(entry, generation) => CapturedFieldCheckpoint::Unchanged(entry, generation),
    })).collect()
}

/// One field's checkpoint captured for streaming later: field id,
/// construction threads, the owned graph/rows snapshot.
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) enum CapturedFieldCheckpoint {
    /// Field id, construction threads, the owned graph/rows snapshot.
    Captured(u16, usize, super::hnsw::MutableVectorCheckpoint),
    /// Unchanged since the last commit: its entry and mutation generation.
    Unchanged(MutableFieldManifest, u64),
}

/// [`PayloadMerge::DeletesOnly`]: segments below the floor merged together
/// (smallest first, until the group reaches the floor); none above it.
fn plan_floor(mut sealed: Vec<(usize, Arc<VectorSegment>)>, row_bytes: usize) -> Option<Vec<Arc<VectorSegment>>> {
    sealed.retain(|(n, _)| n.saturating_mul(row_bytes) < PAYLOAD_FLOOR_BYTES);
    if sealed.len() < 2 { return None; }
    sealed.sort_by_key(|(n, s)| (*n, s.id));
    let total: usize = sealed.iter().map(|(n, _)| n.saturating_mul(row_bytes)).sum();
    // Wait for a floor's worth (or a tier's count) of small segments: a
    // merge per new seal would rewrite the growing group every time.
    if total < PAYLOAD_FLOOR_BYTES && sealed.len() < PAYLOAD_PER_TIER { return None; }
    let mut chosen = Vec::new();
    let mut bytes = 0usize;
    for (n, s) in sealed {
        chosen.push(s);
        bytes = bytes.saturating_add(n.saturating_mul(row_bytes));
        if bytes >= PAYLOAD_FLOOR_BYTES { break; }
    }
    (chosen.len() >= 2).then_some(chosen)
}

/// [`PayloadMerge::DeletesOnly`] above [`OptimizerPolicy::max_segments`]:
/// how many of the smallest segments to merge, given every sealed
/// segment's size in rows sorted ascending (0: none). The largest segment
/// is never an input. At least enough to get back to the ceiling (and at
/// least two) are taken, extended to [`PAYLOAD_PER_TIER`] segments while
/// the merged size stays within [`PAYLOAD_MAX_MERGED_BYTES`], so one merge
/// buys headroom for several seals instead of rewriting a growing group on
/// every seal at the ceiling.
fn plan_ceiling(sizes_sorted: &[usize], max: usize, row_bytes: usize) -> usize {
    let max = max.max(1);
    let count = sizes_sorted.len();
    if count <= max || count < 3 { return 0; }
    let candidates = &sizes_sorted[..count - 1];
    let need = (count - max + 1).max(2).min(candidates.len());
    let mut take = need;
    let mut bytes: usize = candidates[..take].iter().map(|n| n.saturating_mul(row_bytes)).sum();
    while take < candidates.len() && take < PAYLOAD_PER_TIER {
        let next = bytes.saturating_add(candidates[take].saturating_mul(row_bytes));
        if next > PAYLOAD_MAX_MERGED_BYTES { break; }
        bytes = next;
        take += 1;
    }
    take
}

/// [`PayloadMerge::Tiered`] (Lucene `TieredMergePolicy`, simplified): the
/// segment count a tiered index of this size allows (per tier
/// [`PAYLOAD_PER_TIER`] segments, tiers growing by that factor from
/// [`PAYLOAD_FLOOR_BYTES`]); over it, the [`PAYLOAD_PER_TIER`] smallest
/// similar-sized segments are merged (never into more than
/// [`PAYLOAD_MAX_MERGED_BYTES`]; larger segments are left alone).
fn plan_tiered(mut sealed: Vec<(usize, Arc<VectorSegment>)>, row_bytes: usize) -> Option<Vec<Arc<VectorSegment>>> {
    let size = |n: usize| n.saturating_mul(row_bytes).max(PAYLOAD_FLOOR_BYTES);
    sealed.retain(|(n, _)| n.saturating_mul(row_bytes) < PAYLOAD_MAX_MERGED_BYTES / 2);
    if sealed.len() < 2 { return None; }
    sealed.sort_by_key(|(n, s)| (*n, s.id));
    let total: usize = sealed.iter().map(|(n, _)| size(*n)).sum();
    let mut allowed = 0usize;
    let mut level = PAYLOAD_FLOOR_BYTES;
    let mut remaining = total;
    loop {
        let in_level = remaining.div_ceil(level);
        if in_level < PAYLOAD_PER_TIER { allowed += in_level.max(1); break; }
        allowed += PAYLOAD_PER_TIER;
        remaining -= PAYLOAD_PER_TIER * level;
        level = level.saturating_mul(PAYLOAD_PER_TIER);
    }
    if sealed.len() <= allowed { return None; }
    // The window of PER_TIER consecutive (by size) segments with the least
    // skew (largest / total), preferring smaller merges on ties.
    let width = PAYLOAD_PER_TIER.min(sealed.len());
    let mut best: Option<(f64, usize)> = None;
    for start in 0..=sealed.len() - width {
        let window = &sealed[start..start + width];
        let bytes: usize = window.iter().map(|(n, _)| n.saturating_mul(row_bytes)).sum();
        if bytes > PAYLOAD_MAX_MERGED_BYTES { break; }
        let skew = size(window[width - 1].0) as f64 / window.iter().map(|(n, _)| size(*n)).sum::<usize>() as f64;
        if best.is_none_or(|(b, _)| skew < b - 1e-9) { best = Some((skew, start)); }
    }
    let (_, start) = best?;
    Some(sealed[start..start + width].iter().map(|(_, s)| s.clone()).collect())
}

/// Whether `name` is a mutable graph code log chunk file (quantized
/// code-only rows of a mutable vector field, referenced by its graph
/// checkpoint through [`MutableFieldManifest::code_files`]).
pub fn is_code_log_file(name: &str) -> bool {
    name.starts_with("mvq_") && name.ends_with(".rmvq")
}

/// Durable reference to one shard-field graph/source pair. Present even in
/// builds without the backend so they can reject unsupported commits explicitly.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct MutableFieldManifest {
    pub field: u16,
    pub generation: u64,
    pub threads: usize,
    /// Code log chunk files the checkpoint's rows reference
    /// ([`is_code_log_file`]); empty for checkpoints that embed every row.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub code_files: Vec<String>,
}
impl MutableFieldManifest {
    pub(crate) fn filename(&self) -> String {
        format!("mutable_{:016x}_{:04x}.rmvc", self.generation, self.field)
    }
    /// Every file this entry references: the checkpoint and its code files.
    pub(crate) fn files(&self) -> Vec<String> {
        let mut files = Vec::with_capacity(1 + self.code_files.len());
        files.push(self.filename());
        files.extend(self.code_files.iter().cloned());
        files
    }
}
#[cfg(feature = "mutable-hnsw-poc")]
#[derive(Default)]
pub(crate) struct MutableCommitState {
    pub next_generation: u64,
    pub files: Vec<String>,
    /// Per field, the manifest entry of the last commit and the graph's
    /// mutation generation it is exact for: a field whose graph has not
    /// changed since is not checkpointed again (its file is reused).
    pub checkpointed: BTreeMap<u16, (MutableFieldManifest, u64)>,
    /// Fields whose checkpoint rows could not be restored at open: rebuilt
    /// from the committed vector rows by
    /// [`VectorStore::reconcile_mutable_checkpoint`] (field, dim,
    /// similarity, parameters).
    pub rebuilds: Vec<(u16, u32, Similarity, HnswParams)>,
    pub metadata_memory: Option<crate::admission::Reservation>,
}

/// A scheduled `index_options` migration ([`VectorStore::request_migration`]).
#[cfg(feature = "mutable-hnsw-poc")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Migration {
    dim: u32,
    similarity: Similarity,
    params: HnswParams,
    graph: bool,
}

/// One field's entry in a commit's manifest ([`VectorStore::write_captured_mutable_checkpoint`]).
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) struct WrittenField {
    pub entry: MutableFieldManifest,
    /// The graph mutation generation the file is exact for (`None`: unknown).
    pub mutation_generation: Option<u64>,
    /// Written by this commit (else reused from an earlier one).
    pub fresh: bool,
}

/// The vector store of one index. See the module docs.
pub struct VectorStore {
    gpu_gate: RwLock<super::gpu::GpuGate>,
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) mutable_commit: Mutex<MutableCommitState>,
    #[cfg(feature = "mutable-hnsw-poc")]
    mutable_fields: Mutex<BTreeMap<u16, Arc<super::mutable_ingest::MutableIngestField>>>,
    /// Sealed segments whose mutable node maps were built ahead of queries
    /// (`warm_mutable_projections`), by projection id.
    warmed_projections: Mutex<std::collections::HashSet<u64>>,
    /// `index_options` updates whose graph is (re)built in the background
    /// ([`VectorStore::run_migrations`]).
    #[cfg(feature = "mutable-hnsw-poc")]
    migrations: Mutex<BTreeMap<u16, Migration>>,
    #[cfg(feature = "mutable-hnsw-poc")]
    mutable_read_only: bool,
    #[cfg(feature = "mutable-hnsw-poc")]
    captured_mutable_fields: BTreeMap<u16, Arc<super::mutable_ingest::CapturedMutableField>>,
    state: RwLock<State>,
    table: Arc<RwLock<PointTable>>,
    next_point: Arc<AtomicU64>,
    next_segment: AtomicU32,
    optimize_lock: Mutex<()>,
    /// Last field to consume the bounded maintenance rebuild allowance.
    /// The next round starts after it, so a hot field cannot starve others.
    /// u32::MAX means no field has consumed the allowance yet.
    maintenance_after: AtomicU32,
    /// Per `bbq_disk` field, the index and liveness a maintenance census
    /// last found nothing to do for (`(state address, next file, point
    /// table kept version)`): an unchanged field skips the census on later
    /// rounds.
    ivf_quiet: Mutex<std::collections::HashMap<u16, (usize, u64, u64)>>,
    /// Segments were swapped or dropped since the last
    /// [`VectorStore::commit_point`] (a reopen would redo that work).
    uncommitted: std::sync::atomic::AtomicBool,
    /// The owning index's reader-view slot (weak: views hold segments that
    /// hold this store): stamps queued mutable-graph deletes with the view
    /// generation that first excludes them and gates their physical removal
    /// on older views being gone ([`VectorStore::delete_epoch`],
    /// [`VectorStore::min_live_view`]).
    view_clock: RwLock<std::sync::Weak<crate::reader_view::ViewSlot>>,
    /// Test seam ([`VectorStore::pause_next_mutable_stage`]).
    #[cfg(any(test, feature = "test-seams"))]
    stage_pause: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Test seam: count row reads made while the point table is locked
    /// (see [`StoreRowsSnapshot`]).
    #[cfg(test)]
    pub(crate) row_lock_probe: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    pub(crate) row_lock_violations: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(crate) row_lock_probed: std::sync::atomic::AtomicUsize,
    /// A retained capture's accounting of its copied point table.
    retained_charge: std::sync::OnceLock<crate::accounting::Charge>,
    /// The owner directory of the last owner snapshot a query took
    /// ([`VectorStore::mutable_point_owners`]): a query that finds the point
    /// table write-locked (a merged segment's O(docs) attach) reads owners
    /// through it instead of waiting. Shared with the point table's
    /// [`OwnerCacheSlot`] (the store never replaces its table).
    #[cfg_attr(not(feature = "mutable-hnsw-poc"), allow(dead_code))]
    owner_cache: Arc<Mutex<Option<CachedOwners>>>,
}

/// The owner directory of the last owner snapshot a query took, held
/// weakly: it never keeps a directory (its pages, metadata charge) alive
/// past its last real holder, and the table clears it before it needs
/// exclusive ownership of its current directory to retire a page.
#[derive(Clone)]
#[cfg_attr(not(feature = "mutable-hnsw-poc"), allow(dead_code))]
struct CachedOwners {
    len: usize,
    chunks: std::sync::Weak<OwnerDirectory>,
    stale: Arc<AtomicBool>,
    memory: Arc<Mutex<Option<crate::admission::Reservation>>>,
}

/// [`PointTable`]'s handle on the store's owner cache. A cloned table (a
/// retained capture's copy) gets its own empty slot.
#[derive(Default)]
struct OwnerCacheSlot(Arc<Mutex<Option<CachedOwners>>>);
impl Clone for OwnerCacheSlot {
    fn clone(&self) -> Self { Self::default() }
}
impl OwnerCacheSlot {
    fn clear(&self) { *self.0.lock() = None; }
}

/// A vector chunk written for a text segment that has not finished encoding.
/// It is deliberately absent from the store: optimization cannot mistake its
/// not-yet-attached points for deletions. Dropping an unsuccessful append
/// removes its provisional file; failed cleanup is retried by the next commit.
/// Stable point identities reserved before text/vector files are written.
/// The allocator identity prevents crossing stores (including captured PITs).
/// A reservation can be retried after failed output, but published only once.
pub(crate) struct PointReservation {
    inner: Arc<PointReservationState>,
}
struct PointReservationState {
    table: std::sync::Weak<RwLock<PointTable>>,
    allocator: Arc<AtomicU64>,
    first: u64,
    count: usize,
    // 0: reserved, 1: provisional chunk, 2: published.
    state: std::sync::atomic::AtomicU8,
    #[cfg(feature = "mutable-hnsw-poc")]
    abandon: Mutex<Option<(usize, Box<dyn FnOnce() + Send + Sync>)>>,
}
impl Drop for PointReservationState {
    fn drop(&mut self) {
        #[cfg(feature = "mutable-hnsw-poc")]
        if self.state.load(AtomicOrdering::Acquire) != 2 {
            if let Some((_, cleanup)) = self.abandon.get_mut().take() { cleanup(); }
        }
        if let Some(table) = self.table.upgrade() { table.write().release_pending(self.first, self.count); }
    }
}
impl PointReservation {
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn on_abandon(&self, owner: usize, cleanup: impl FnOnce() + Send + Sync + 'static) -> Result<()> {
        let mut hook = self.inner.abandon.lock();
        if hook.is_some() || self.inner.state.load(AtomicOrdering::Acquire) != 0 {
            return Err(crate::error::RusticError::Segment("reservation cleanup already bound or in use".into()));
        }
        *hook = Some((owner, Box::new(cleanup)));
        Ok(())
    }
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn belongs_to_mutable(&self, owner: usize) -> bool {
        self.inner.abandon.lock().as_ref().is_some_and(|(id, _)| *id == owner)
    }
    pub(crate) fn belongs_to(&self, store: &VectorStore) -> bool {
        Arc::ptr_eq(&self.inner.allocator, &store.next_point)
    }
    pub(crate) fn points(&self) -> std::ops::Range<u64> {
        self.inner.first..self.inner.first + self.inner.count as u64
    }
}

pub(crate) struct PreparedVectorAppend<'a> {
    store: &'a VectorStore,
    directory: &'a dyn Directory,
    id: u32,
    segment: Option<Arc<VectorSegment>>,
    document_points: Vec<PointId>,
    installed: bool,
    reservations: Vec<Arc<PointReservationState>>,
}

impl PreparedVectorAppend<'_> {
    pub(crate) fn points(&self) -> &[PointId] {
        &self.document_points
    }
}

impl Drop for PreparedVectorAppend<'_> {
    fn drop(&mut self) {
        if !self.installed {
            for reservation in &self.reservations { reservation.state.store(0, AtomicOrdering::Release); }
            // Release our mapping before unlinking the provisional file.
            drop(self.segment.take());
            let name = crate::codec::vector_segment_file_name(self.id);
            if self.directory.exists(&name) && self.directory.delete(&name).is_err() {
                self.store.restore_obsolete(vec![self.id]);
            }
        }
    }
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("segments", &self.segment_ids())
            .field("next_point", &self.next_point_id())
            .finish()
    }
}

/// Pin backend selection for one request across an ES mapping transition.
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) enum MutableQueryField {
    Live(Arc<super::mutable_ingest::MutableIngestField>),
    Captured(Arc<super::mutable_ingest::CapturedMutableField>),
}

#[cfg(feature = "mutable-hnsw-poc")]
impl MutableQueryField {
    pub(crate) fn similarity(&self) -> Similarity {
        match self { Self::Live(field) => field.similarity(), Self::Captured(field) => field.similarity() }
    }
    /// The quantizer unrescored scores are estimated with (`None`: f32 field,
    /// or a released capture).
    pub(crate) fn score_quantizer(&self) -> Option<super::es_quant::EsQuantizer> {
        match self { Self::Live(field) => field.score_quantizer(), Self::Captured(field) => field.score_quantizer() }
    }
    /// Each point's own code score (the unrescored graph score): the exact
    /// route returns the same scores as the graph route.
    pub(crate) fn code_scores(&self, query: &[f32], points: &[u64]) -> Result<Vec<Option<f32>>> {
        match self { Self::Live(field) => field.code_scores(query, points), Self::Captured(field) => Ok(field.code_scores(query, points)) }
    }
    pub(crate) fn supports_projection_cache(&self) -> bool { matches!(self, Self::Live(_)) }
    /// A retained capture whose graph pin was released: search exactly.
    pub(crate) fn is_released(&self) -> bool {
        match self { Self::Live(_) => false, Self::Captured(field) => field.is_released() }
    }
    pub(crate) fn prepare_projection<'a>(&self, key: Vec<(u64, Vec<u64>)>, key_memory:crate::admission::Reservation,
        columns: impl FnOnce() -> Vec<Option<crate::codec::vectors::PointColumn<'a>>>)
        -> Result<Option<super::mutable_ingest::PreparedMutableFieldSearch<'_>>> {
        match self {
            Self::Live(field) => field.prepare_ann_segments_admitted(key,key_memory,columns).map(Some),
            Self::Captured(_) => Ok(None),
        }
    }
    pub(crate) fn document_limit(&self) -> u32 {
        match self { Self::Live(field) => field.document_limit(), Self::Captured(field) => field.document_limit() }
    }
    /// Graph nodes a search can score at most once each (the profile's
    /// operation bound): the live graph's nodes, or a capture's node bound.
    pub(crate) fn scored_node_bound(&self) -> u32 {
        match self { Self::Live(field) => field.active_nodes(), Self::Captured(field) => field.document_limit().saturating_sub(1) }
    }
    pub(crate) fn search(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: impl Fn(u64) -> bool, deadline: Option<std::time::Instant>) -> Result<super::hnsw::MutableSearchResult> {
        match self {
            Self::Live(field) => field.search_ann(query, plan, accept, deadline),
            Self::Captured(field) => field.search_ann(query, plan, accept, deadline),
        }
    }
    /// [`search`](Self::search) with an acceptance the search pool can share
    /// (a live field may split its beam over idle workers).
    pub(crate) fn search_shared(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: &(impl Fn(u64) -> bool + Sync), deadline: Option<std::time::Instant>) -> Result<super::hnsw::MutableSearchResult> {
        match self {
            Self::Live(field) => field.search_ann_shared(query, plan, accept, deadline),
            Self::Captured(field) => field.search_ann(query, plan, accept, deadline),
        }
    }
    pub(crate) fn search_points(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        points: &[PointId], deadline: Option<std::time::Instant>) -> Result<super::hnsw::MutableSearchResult> {
        match self {
            Self::Live(field) => field.prepare_ann_points(points)?.search(query, plan, deadline),
            Self::Captured(field) => field.search_ann_points(query, plan, points, deadline),
        }
    }
}

impl VectorStore {
    /// A store captured for a retained reader view ([`VectorStore::capture_read_store`]).
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn is_read_capture(&self) -> bool {
        self.mutable_read_only
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn mutable_query_field(&self, field: u16) -> Option<MutableQueryField> {
        if let Some(captured) = self.captured_mutable_fields.get(&field) {
            return Some(MutableQueryField::Captured(captured.clone()));
        }
        self.registered_mutable_field(field).map(MutableQueryField::Live)
    }
    /// Return the one mutable owner for this shard field. Activation currently
    /// requires an empty field: existing segment points need checkpoint/replay
    /// migration before this backend can safely serve them.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn mutable_field(self: &Arc<Self>, field: u16, dim: u32, similarity: Similarity, params: HnswParams)
        -> Result<Arc<super::mutable_ingest::MutableIngestField>>
    {
        if self.mutable_read_only {
            return Err(crate::error::RusticError::Segment("cannot admit mutable writes into a retained read store".into()));
        }
        // `state` before the field registry, never under it (lock order:
        // see `rows_leaving`). Activation runs under the shard publication
        // fence, so no segment of the field appears in between.
        let populated = self.state.read().segments.values().any(|segment| segment.field == field);
        let mut fields = self.mutable_fields.lock();
        if let Some(owner) = fields.get(&field) {
            if !owner.matches_config(dim, similarity, params) {
                return Err(crate::error::RusticError::Segment("mutable field configuration changed".into()));
            }
            return Ok(owner.clone());
        }
        if populated {
            return Err(crate::error::RusticError::Segment("mutable field activation requires checkpoint migration of existing points".into()));
        }
        let owner = super::mutable_ingest::MutableIngestField::new(self.clone(), field, dim, similarity, params)?;
        fields.insert(field, owner.clone());
        Ok(owner)
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn mutable_point_is_live(&self, point: PointId) -> bool {
        let table = self.table.read();
        table.is_alive(point) || table.is_hidden(point)
    }

    /// The retained capture of `field` (retained read stores only).
    #[cfg(all(test, feature = "mutable-hnsw-poc"))]
    pub(crate) fn captured_mutable_field(&self, field: u16) -> Option<Arc<super::mutable_ingest::CapturedMutableField>> {
        self.captured_mutable_fields.get(&field).cloned()
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    /// A query's owner snapshot. Never waits behind a writer holding the
    /// point table (a merged segment's attach writes every doc's owner under
    /// it, tens of ms on a large segment, which every concurrent kNN waited
    /// out): it then reads through the last snapshot's owner directory. The
    /// owner column is a fast path, so a stale directory is safe: its
    /// entries are the same live atomics (a retired page's are cleared
    /// first), and a point it lacks, or whose owner is outside the query's
    /// view, resolves through the view's own point maps.
    pub(crate) fn mutable_point_owners(&self) -> MutablePointOwners {
        if let Some(table) = self.table.try_read() {
            let owners = table.owner.snapshot(table.admission.0.clone(), Arc::downgrade(&self.table));
            if let Some(chunks) = &owners.chunks {
                *self.owner_cache.lock() = Some(CachedOwners { len: owners.len, chunks: Arc::downgrade(chunks), stale: owners.stale.clone(), memory: table.admission.0.clone() });
            }
            return MutablePointOwners { owners };
        }
        // A writer holds the table: the last snapshot's directory, if the
        // table (or a query) still holds it.
        let cached = self.owner_cache.lock().clone();
        if let Some(chunks) = cached.as_ref().and_then(|cached| cached.chunks.upgrade()) {
            let cached = cached.expect("upgraded from it");
            return MutablePointOwners { owners: OwnerSnapshot { len: cached.len, chunks: Some(chunks), table: Arc::downgrade(&self.table), stale: cached.stale, _memory: cached.memory } };
        }
        let table = self.table.read();
        MutablePointOwners { owners: table.owner.snapshot(table.admission.0.clone(), Arc::downgrade(&self.table)) }
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn mutable_document_limit(&self, field: u16) -> Option<u32> {
        if let Some(captured) = self.captured_mutable_fields.get(&field) { return Some(captured.document_limit()); }
        self.mutable_fields.lock().get(&field).map(|owner| owner.document_limit())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn registered_mutable_field(&self, field: u16) -> Option<Arc<super::mutable_ingest::MutableIngestField>> {
        self.mutable_fields.lock().get(&field).cloned()
    }

    /// Metadata arrays for the captured, pending, and serialized field
    /// registries. Caller keeps the field publication fence while capturing.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn checkpoint_registry_memory(&self) -> Result<crate::admission::Reservation> {
        let count = self.mutable_fields.lock().len() as u64;
        let per = (std::mem::size_of::<PendingFieldCheckpoint>() + std::mem::size_of::<CapturedFieldCheckpoint>()
            + std::mem::size_of::<WrittenField>() + std::mem::size_of::<MutableFieldManifest>() + 128) as u64;
        crate::admission::shared().reserve(count.saturating_mul(per).saturating_mul(2).saturating_add(4096), "mutable checkpoint registry arrays")
    }

    /// Called under the index publication fence and mutable_commit lock.
    /// All files finish before meta.json can reference any of them.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn write_mutable_checkpoint(&self, dir: &dyn Directory, generation: u64)
        -> Result<Vec<MutableFieldManifest>>
    {
        if self.mutable_read_only { return Err(crate::error::RusticError::Segment("cannot checkpoint a retained read store".into())); }
        let fields: Vec<_> = self.mutable_fields.lock().iter().map(|(&id, owner)| (id, owner.clone())).collect();
        let mut manifest = Vec::with_capacity(fields.len());
        for (field, owner) in fields {
            let mut entry = MutableFieldManifest { field, generation, threads: owner.construction_threads(), code_files: Vec::new() };
            match owner.write_checkpoint(dir, &entry.filename()) {
                Ok(code_files) => entry.code_files = code_files,
                Err(error) => {
                    for prior in &manifest {
                        let prior: &MutableFieldManifest = prior;
                        let _ = dir.delete(&prior.filename());
                    }
                    let _ = dir.delete(&entry.filename());
                    return Err(error);
                }
            }
            manifest.push(entry);
        }
        Ok(manifest)
    }

    /// Capture every field's checkpoint now (see
    /// `MutableIngestField::capture_checkpoint`) for streaming later with
    /// [`VectorStore::write_captured_mutable_checkpoint`].
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn capture_mutable_checkpoint(&self) -> Result<Vec<CapturedFieldCheckpoint>> {
        if self.mutable_read_only { return Err(crate::error::RusticError::Segment("cannot checkpoint a retained read store".into())); }
        let fields: Vec<_> = self.mutable_fields.lock().iter().map(|(&id, owner)| (id, owner.clone())).collect();
        let mut captured = Vec::with_capacity(fields.len());
        for (field, owner) in fields {
            captured.push(CapturedFieldCheckpoint::Captured(field, owner.construction_threads(), owner.capture_checkpoint()?));
        }
        Ok(captured)
    }

    /// Code log files of every registered mutable graph (written or
    /// restored; see `MutableIngestField::code_log_files`).
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn code_log_files(&self) -> Vec<String> {
        let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
        fields.iter().flat_map(|field| field.code_log_files()).collect()
    }

    /// Some mutable field has no graph checkpoint in any commit yet (its
    /// first capture needs graph ingestion quiesced).
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn has_uncheckpointed_mutable_fields(&self) -> bool {
        let commit = self.mutable_commit.lock();
        self.mutable_fields.lock().keys().any(|field| !commit.checkpointed.contains_key(field))
    }

    /// The capture point of every field's checkpoint (see
    /// `MutableIngestField::begin_capture_checkpoint`), finished later with
    /// [`finish_mutable_checkpoint`].
    #[cfg(feature = "mutable-hnsw-poc")]
    /// `save_graphs == false`: every field with a committed checkpoint keeps
    /// it (scheduled graph saves; a new field is always captured).
    pub(crate) fn begin_mutable_checkpoint(&self, committed: &BTreeMap<u16, (MutableFieldManifest, u64)>, save_graphs: bool) -> Result<Vec<PendingFieldCheckpoint>> {
        if self.mutable_read_only { return Err(crate::error::RusticError::Segment("cannot checkpoint a retained read store".into())); }
        let fields: Vec<_> = self.mutable_fields.lock().iter().map(|(&id, owner)| (id, owner.clone())).collect();
        let mut pending = Vec::with_capacity(fields.len());
        for (field, owner) in fields {
            let threads = owner.construction_threads();
            // An unchanged graph: the last commit's file is still exact.
            if let Some((entry, generation)) = committed.get(&field) {
                if (!save_graphs || owner.mutation_generation() == *generation) && entry.threads == threads {
                    pending.push(PendingFieldCheckpoint::Unchanged(entry.clone(), *generation));
                    continue;
                }
            }
            pending.push(PendingFieldCheckpoint::Capture(field, threads, owner.begin_capture_checkpoint()?));
        }
        Ok(pending)
    }

    /// [`VectorStore::write_mutable_checkpoint`] from checkpoints captured
    /// earlier: streams them without any field or index lock.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn write_captured_mutable_checkpoint(&self, dir: &dyn Directory, generation: u64, captured: &[CapturedFieldCheckpoint])
        -> Result<Vec<WrittenField>>
    {
        // Full code log chunks move to their files (and out of anonymous
        // memory) at every commit, whether or not a graph save is due: the
        // log's resident buffers stay bounded by the commit interval.
        let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
        for field in fields { field.seal_code_log(dir)?; }
        let mut manifest: Vec<WrittenField> = Vec::with_capacity(captured.len());
        for capture in captured {
            match capture {
                CapturedFieldCheckpoint::Captured(field, threads, checkpoint) => {
                    let mut entry = MutableFieldManifest { field: *field, generation, threads: *threads, code_files: Vec::new() };
                    match checkpoint.write_file(dir, &entry.filename()) {
                        Ok(code_files) => entry.code_files = code_files,
                        Err(error) => {
                            for prior in manifest.iter().filter(|prior| prior.fresh) {
                                let _ = dir.delete(&prior.entry.filename());
                            }
                            let _ = dir.delete(&entry.filename());
                            return Err(error);
                        }
                    }
                    manifest.push(WrittenField { entry, mutation_generation: Some(checkpoint.mutation_generation()), fresh: true });
                }
                CapturedFieldCheckpoint::Unchanged(entry, mutation_generation) => {
                    manifest.push(WrittenField { entry: entry.clone(), mutation_generation: Some(*mutation_generation), fresh: false });
                }
            }
        }
        Ok(manifest)
    }

    /// Restore topology/source before text attachment, then reconcile ownership
    /// after every committed text segment has been attached by Index::open_in.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn restore_mutable_checkpoint(self: &Arc<Self>, dir: &dyn Directory,
        manifest: &[MutableFieldManifest], next_generation: u64) -> Result<()>
    {
        let mut fields = self.mutable_fields.lock();
        if !fields.is_empty() { return Err(crate::error::RusticError::Corrupted("mutable registry already populated".into())); }
        let metadata_memory = crate::admission::shared().reserve((manifest.len() as u64).saturating_mul(512).saturating_add(4096), "restored mutable checkpoint registry")?;
        let mut restored = BTreeMap::new();
        let mut rebuilds = Vec::new();
        for entry in manifest {
            if entry.generation >= next_generation {
                return Err(crate::error::RusticError::Corrupted("invalid mutable field manifest".into()));
            }
            if restored.contains_key(&entry.field) || rebuilds.iter().any(|r: &(_, _, _, _)| r.0 == entry.field) {
                return Err(crate::error::RusticError::Corrupted("invalid mutable field manifest".into()));
            }
            match super::mutable_ingest::MutableIngestField::open_checkpoint(self.clone(), entry.field, dir, &entry.filename(), entry.threads)? {
                super::mutable_ingest::OpenedCheckpoint::Restored(field) => {
                    // The file is durable (this commit point references it).
                    field.checkpoint_durable(field.mutation_generation());
                    restored.insert(entry.field, field);
                }
                super::mutable_ingest::OpenedCheckpoint::Rebuild { dim, similarity, params } =>
                    rebuilds.push((entry.field, dim, similarity, params)),
            }
        }
        // Each restored graph is exactly its file (reconciliation at open
        // bumps the generation when it changes the graph). A rebuilt field
        // has no reusable checkpoint: the next commit captures it.
        let checkpointed = manifest.iter().filter_map(|entry| restored.get(&entry.field)
            .map(|field| (entry.field, (entry.clone(), field.mutation_generation())))).collect();
        *fields = restored;
        drop(fields);
        *self.mutable_commit.lock() = MutableCommitState { next_generation,
            files: manifest.iter().flat_map(MutableFieldManifest::files).collect(), checkpointed, rebuilds, metadata_memory: Some(metadata_memory) };
        Ok(())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn incompatible_mutable_fields(&self, schema: &[crate::types::Field]) -> Vec<u16> {
        self.mutable_fields.lock().iter().filter_map(|(&id, owner)| {
            let compatible = schema.iter().find(|f| f.id == id).is_some_and(|field| {
                let crate::types::FieldType::Vector { dim } = field.field_type else { return false; };
                field.vector_options.is_some_and(|options| u32::from(dim) == owner.dimensions()
                    && crate::codec::vectors::similarity_by_name(&field.similarity) == Some(owner.similarity())
                    && options.graph
                    && owner.matches_index_options(options))
            });
            (!compatible).then_some(id)
        }).collect()
    }

    /// Check the restored graphs against the committed schema. Returns the
    /// fields whose graph options differ (an `index_options` update
    /// committed before its migration completed): the caller schedules
    /// their migration again; the restored graph serves them meanwhile.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn validate_mutable_schema(&self, schema: &[crate::types::Field]) -> Result<Vec<u16>> {
        let mut migrate = Vec::new();
        for (&id, owner) in self.mutable_fields.lock().iter() {
            let field = schema.iter().find(|f| f.id == id)
                .ok_or_else(|| crate::error::RusticError::Corrupted("mutable checkpoint field absent from schema".into()))?;
            let dim = match &field.field_type {
                crate::types::FieldType::Vector { dim } => u32::from(*dim),
                _ => return Err(crate::error::RusticError::Corrupted("mutable checkpoint field is not a vector".into())),
            };
            let options = field.vector_options.ok_or_else(|| crate::error::RusticError::Corrupted("mutable checkpoint lacks explicit vector mapping".into()))?;
            if dim != owner.dimensions() || crate::codec::vectors::similarity_by_name(&field.similarity) != Some(owner.similarity()) {
                return Err(crate::error::RusticError::Corrupted("mutable checkpoint schema mismatch".into()));
            }
            if !options.graph || !owner.matches_index_options(options) {
                migrate.push(id);
            }
        }
        Ok(migrate)
    }

    /// Bring every restored field up to the committed documents (after the
    /// committed text segments are attached and the field options synced).
    /// A field whose checkpoint cannot be restored or reconciled (its file
    /// references vector rows no longer stored) is rebuilt from the
    /// committed vector rows instead of failing the shard.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn reconcile_mutable_checkpoint(self: &Arc<Self>) -> Result<()> {
        let rebuilds = std::mem::take(&mut self.mutable_commit.lock().rebuilds);
        for (field, dim, similarity, params) in rebuilds {
            self.rebuild_mutable_field(field, dim, similarity, params)?;
        }
        let fields: Vec<_> = self.mutable_fields.lock().iter().map(|(&id, field)| (id, field.clone())).collect();
        for (id, field) in fields {
            if let Err(error) = field.reconcile_committed_points() {
                // A cancelled open is not a broken checkpoint: no rebuild.
                if matches!(error, crate::error::RusticError::Cancelled(_)) {
                    return Err(error);
                }
                tracing::warn!(field = id, "reconciling the vector graph checkpoint failed: {error}; rebuilding it from the committed vector rows");
                let (dim, similarity, params) = field.rebuild_config();
                drop(field);
                self.rebuild_mutable_field(id, dim, similarity, params)?;
            }
        }
        Ok(())
    }

    /// Schedule `field`'s graph to be (re)built for its current mapping
    /// (`graph: false`: dropped) in the background: nothing is built here.
    /// Until [`run_migrations`](Self::run_migrations) swaps it in, the
    /// current graph (if any) keeps serving searches and admitting rows.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn request_migration(&self, field: u16, dim: u32, similarity: Similarity, params: HnswParams, graph: bool) {
        self.migrations.lock().insert(field, Migration { dim, similarity, params, graph });
    }

    /// Whether `field` has a migration scheduled: admissions then go to the
    /// current graph whatever the mapping's new options say.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn migration_pending(&self, field: u16) -> bool {
        self.migrations.lock().contains_key(&field)
    }

    /// Migrations scheduled and not completed.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn migrations_pending(&self) -> usize {
        self.migrations.lock().len()
    }

    /// Run the scheduled `index_options` migrations (background maintenance,
    /// off every shard lock): build each field's new graph unregistered from
    /// the current one (or, `flat` → `hnsw`, from the stored rows), catch it
    /// up with what arrived meanwhile, retire the current owner into it and
    /// register it (a brief swap). A failure leaves the current graph in
    /// place and the migration scheduled: the next pass retries it.
    /// Returns the migrations completed.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn run_migrations(self: &Arc<Self>) -> Result<usize> {
        if self.mutable_read_only {
            return Ok(0);
        }
        let pending: Vec<(u16, Migration)> = self.migrations.lock().iter().map(|(field, m)| (*field, *m)).collect();
        let mut done = 0;
        for (field, migration) in pending {
            match self.migrate_field(field, migration) {
                Ok(()) => {
                    let mut migrations = self.migrations.lock();
                    // A newer request for the field runs next pass.
                    if migrations.get(&field) == Some(&migration) {
                        migrations.remove(&field);
                    }
                    done += 1;
                }
                Err(error) => tracing::warn!(field, "vector field migration failed; retried by the next maintenance pass: {error}"),
            }
        }
        Ok(done)
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn migrate_field(self: &Arc<Self>, field: u16, migration: Migration) -> Result<()> {
        let previous = self.registered_mutable_field(field);
        // DiskBBQ: a field no longer `bbq_disk` loses its shared index here;
        // new code parameters are rebuilt by the next maintenance step,
        // which trains a replacement off every lock while the current index
        // keeps serving, and installs it only if the mapping still asks for
        // it.
        if self.field_options(field).ivf.is_none() {
            self.reset_ivf_field(field)?;
        }
        if !migration.graph {
            self.retire_mutable_field(field);
            return Ok(());
        }
        let fresh = super::mutable_ingest::MutableIngestField::new(self.clone(), field, migration.dim, migration.similarity, migration.params)?;
        match previous {
            Some(previous) => {
                // Bounded catch-up passes while writes continue, then the
                // retirement copies what is left (one write request's worth).
                let mut copied = std::collections::HashSet::new();
                for _ in 0..4 {
                    if fresh.catch_up_from(&previous, &mut copied)? < 1024 { break; }
                }
                // Once retired, `previous` forwards every admission to
                // `fresh`: it is registered whatever the last copy returned,
                // and a failed copy is repaired from the committed rows.
                let retired = previous.retire_into(&fresh, &mut copied);
                self.mutable_commit.lock().checkpointed.remove(&field);
                self.mutable_fields.lock().insert(field, fresh.clone());
                if let Err(error) = retired {
                    tracing::warn!(field, "vector field migration's last copy failed; repairing from the committed rows: {error}");
                    fresh.insert_missing_committed_points()?;
                }
            }
            None => {
                // `flat` → `hnsw`: the rows stored so far, then registered;
                // rows sealed meanwhile are caught up right after, later ones
                // as each seal stores them (`insert_unreserved_rows`).
                fresh.insert_missing_committed_points()?;
                self.mutable_commit.lock().checkpointed.remove(&field);
                self.mutable_fields.lock().insert(field, fresh.clone());
                fresh.insert_missing_committed_points()?;
            }
        }
        Ok(())
    }

    /// Rows a seal stored without a graph reservation (written while the
    /// field had no graph): insert them into the field's graph, if it has
    /// one now.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn insert_unreserved_rows(&self, field: u16, points: std::ops::Range<u64>) {
        let Some(owner) = self.registered_mutable_field(field) else { return };
        let points: Vec<u64> = points.collect();
        if let Err(error) = owner.insert_committed_points(&points) {
            tracing::warn!(field, "vector rows stored without a graph reservation were not inserted: {error}");
        }
    }

    /// Drop `field`'s graph (its mapping is no longer a graph type): its
    /// rows stay in their vector segments and are searched exactly; the
    /// next commit's manifest no longer lists a checkpoint for it.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn retire_mutable_field(&self, field: u16) {
        self.mutable_commit.lock().checkpointed.remove(&field);
        self.mutable_fields.lock().remove(&field);
    }

    /// Replace `field` with a fresh graph filled from the stored vector
    /// rows of its live points (open time only: a checkpoint that cannot be
    /// reused, with no writer running; a mapping update migrates in the
    /// background instead, see [`run_migrations`](Self::run_migrations)).
    /// It has no reusable checkpoint until the next save.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn rebuild_mutable_field(self: &Arc<Self>, field: u16, dim: u32, similarity: Similarity, params: HnswParams) -> Result<()> {
        let fresh = super::mutable_ingest::MutableIngestField::new(self.clone(), field, dim, similarity, params)?;
        self.mutable_commit.lock().checkpointed.remove(&field);
        if let Some(previous) = self.mutable_fields.lock().insert(field, fresh.clone()) {
            fresh.adopt_leases(&previous);
        }
        fresh.reconcile_committed_points()
    }

    /// Search either the live mutable field or its explicitly retained PIT
    /// topology. None means the field has not enabled this backend.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn search_mutable_ann(&self, field: u16, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: impl Fn(u64) -> bool, deadline: Option<std::time::Instant>)
        -> Result<Option<super::hnsw::MutableSearchResult>>
    {
        if let Some(captured) = self.captured_mutable_fields.get(&field) {
            return captured.search_ann(query, plan, accept, deadline).map(Some);
        }
        let owner = self.mutable_fields.lock().get(&field).cloned();
        owner.map(|owner| owner.search_ann(query, plan, accept, deadline)).transpose()
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn search_mutable_ann_points(&self, field: u16, query: &[f32],
        plan: &super::vespa_policy::QueryPlan, points: &[PointId], deadline: Option<std::time::Instant>)
        -> Result<Option<super::hnsw::MutableSearchResult>> {
        if let Some(captured) = self.captured_mutable_fields.get(&field) {
            return captured.search_ann_points(query, plan, points, deadline).map(Some);
        }
        let owner = self.mutable_fields.lock().get(&field).cloned();
        owner.map(|owner| owner.prepare_ann_points(points)?.search(query, plan, deadline)).transpose()
    }

    /// Retain current point locations, liveness and owners for an explicit PIT.
    /// Vector payloads/graphs remain shared through immutable segment Arcs.
    /// Call under the shard publication fence with document-mask capture.
    /// This copies point metadata once per PIT, not per ordinary search.
    pub fn capture_read_store(&self) -> Result<Arc<Self>> {
        #[cfg(feature = "mutable-hnsw-poc")]
        let captured_mutable_fields = {
            let fields = self.mutable_fields.lock();
            let mut captured = self.captured_mutable_fields.clone();
            for (&id, field) in fields.iter() { captured.insert(id, field.capture()?); }
            captured
        };
        // A new retained view: bound what retained views pin (never waits).
        #[cfg(feature = "mutable-hnsw-poc")]
        super::mutable_ingest::enforce_retained_budget(super::mutable_ingest::retained_graph_budget());
        let state = self.state.read();
        let table = self.table.read();
        let copy_memory = crate::admission::shared().reserve((table.heap_bytes() as u64).saturating_mul(2).saturating_add((state.ivf.len() as u64).saturating_mul(128)).saturating_add(4096), "retained vector point table")?;
        let mut copied_table = table.clone();
        copied_table.admission = TableAdmission(Arc::new(Mutex::new(Some(copy_memory))));
        let captured = Arc::new(Self {
            gpu_gate: RwLock::new(super::gpu::GpuGate::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_fields: Mutex::new(BTreeMap::new()),
            warmed_projections: Mutex::new(Default::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            migrations: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_commit: Mutex::new(MutableCommitState::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_read_only: true,
            #[cfg(feature = "mutable-hnsw-poc")]
            captured_mutable_fields,
            state: RwLock::new(State {
                segments: state.segments.clone(), obsolete: Vec::new(),
                field_options: state.field_options.clone(), last_append: BTreeMap::new(),
                // Immutable index versions: the view keeps the one it saw.
                ivf_memory: Some(crate::admission::shared().reserve((state.ivf.len() as u64).saturating_mul(512).saturating_add(512), "captured IVF field registry")?),
                ivf: state.ivf.clone(), ivf_deferred: BTreeMap::new(), ivf_open_retry: BTreeMap::new(), obsolete_files: Default::default(),
            }),
            owner_cache: copied_table.owner_cache.0.clone(),
            table: Arc::new(RwLock::new(copied_table)),
            next_point: Arc::new(AtomicU64::new(self.next_point.load(AtomicOrdering::Acquire))),
            next_segment: AtomicU32::new(self.next_segment.load(AtomicOrdering::Acquire)),
            optimize_lock: Mutex::new(()), maintenance_after: AtomicU32::new(u32::MAX),
            ivf_quiet: Default::default(),
            uncommitted: Default::default(),
            view_clock: RwLock::new(std::sync::Weak::new()),
            #[cfg(any(test, feature = "test-seams"))]
            stage_pause: Mutex::new(None),
            #[cfg(test)]
            row_lock_probe: Default::default(),
            #[cfg(test)]
            row_lock_violations: Default::default(),
            #[cfg(test)]
            row_lock_probed: Default::default(),
            retained_charge: std::sync::OnceLock::new(),
        });
        // Measured off the live store's locks (the copy is not shared yet).
        drop(table);
        drop(state);
        let copied = captured.table.read().heap_bytes() as u64;
        let _ = captured.retained_charge.set(crate::accounting::Charge::new(crate::accounting::Category::RetainedCaptures, copied));
        // A retained graph reads the f32 of its code-only rows from the
        // retained store (its own segments), not the live one.
        #[cfg(feature = "mutable-hnsw-poc")]
        for field in captured.captured_mutable_fields.values() {
            field.bind_raw(&StoreRows(Arc::downgrade(&captured)));
        }
        Ok(captured)
    }

    pub fn set_gpu_gate(&self, gate: super::gpu::GpuGate) { *self.gpu_gate.write() = gate; }
    pub fn gpu_allowed(&self) -> bool { self.gpu_gate.read().allows() }

    /// An empty store.
    pub fn new() -> Arc<Self> {
        Self::with_ids(0, 0)
    }

    /// An empty store (same as [`new`](Self::new)); provided so the store
    /// can be built with `Default::default()` in tests and examples.
    pub fn default() -> Arc<Self> {
        Self::with_ids(0, 0)
    }

    fn with_ids(next_point: u64, next_segment: u32) -> Arc<Self> {
        let table = PointTable::default();
        Arc::new(Self {
            gpu_gate: RwLock::new(super::gpu::GpuGate::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_fields: Mutex::new(BTreeMap::new()),
            warmed_projections: Mutex::new(Default::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            migrations: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_commit: Mutex::new(MutableCommitState::default()),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_read_only: false,
            #[cfg(feature = "mutable-hnsw-poc")]
            captured_mutable_fields: BTreeMap::new(),
            state: RwLock::new(State::default()),
            owner_cache: table.owner_cache.0.clone(),
            table: Arc::new(RwLock::new(table)),
            next_point: Arc::new(AtomicU64::new(next_point)),
            next_segment: AtomicU32::new(next_segment),
            optimize_lock: Mutex::new(()),
            maintenance_after: AtomicU32::new(u32::MAX),
            ivf_quiet: Default::default(),
            uncommitted: Default::default(),
            view_clock: RwLock::new(std::sync::Weak::new()),
            #[cfg(any(test, feature = "test-seams"))]
            stage_pause: Mutex::new(None),
            #[cfg(test)]
            row_lock_probe: Default::default(),
            #[cfg(test)]
            row_lock_violations: Default::default(),
            #[cfg(test)]
            row_lock_probed: Default::default(),
            retained_charge: std::sync::OnceLock::new(),
        })
    }

    /// Open the store of a committed index: the listed vector segments, and
    /// the id counters from the commit point. Text segments must then be
    /// [`attach`](Self::attach)ed to make their points alive.
    pub fn open(dir: &dyn Directory, segment_ids: &[u32], next_point: u64, next_segment: u32) -> Result<Arc<Self>> {
        let store = Self::with_ids(next_point, next_segment);
        // The monotonic allocator is not the resident point set. Register only
        // pages backed by surviving payload segments below.
        for &id in segment_ids {
            store.register(Arc::new(VectorSegment::open(dir, id)?))?;
        }
        Ok(store)
    }

    /// A store over *every* vector segment file in `dir` (for segments used
    /// without an [`Index`](crate::Index)); id counters continue after the
    /// largest ids found. When files overlap (a sealed segment and the
    /// chunks it replaced), the newest segment owns the point.
    pub fn open_dir(dir: &dyn Directory) -> Result<Arc<Self>> {
        let names=dir.list_admitted(false)?;
        let _ids_memory=crate::admission::shared().reserve((names.len() as u64).saturating_mul(4),"vector directory segment IDs")?;
        let mut ids: Vec<u32> = Vec::new();
        ids.try_reserve_exact(names.len()).map_err(|error|crate::error::RusticError::Memory(format!("vector directory ID allocation refused: {error}")))?;
        ids.extend(names.iter().filter_map(|n| parse_vector_segment_file_name(n)));
        ids.sort_unstable();
        let next=ids.last().map(|id|id.checked_add(1).ok_or_else(||crate::error::RusticError::Corrupted("vector segment ID exhausted".into()))).transpose()?.unwrap_or(0);
        let store = Self::with_ids(0, next);
        for id in ids {
            let seg = Arc::new(VectorSegment::open(dir, id)?);
            let max = seg.points().iter().copied().max();
            store.register(seg)?;
            if let Some(m) = max {
                let next=m.checked_add(1).ok_or_else(||crate::error::RusticError::Corrupted("vector point ID exhausted".into()))?;
                store.next_point.fetch_max(next, AtomicOrdering::Relaxed);
            }
        }
        Ok(store)
    }

    fn register(&self, seg: Arc<VectorSegment>) -> Result<()> {
        let mut state = self.state.write();
        let mut t = self.table.write();
        for &point in seg.points() {
            let n = usize::try_from(point).ok().and_then(|n| n.checked_add(1)).ok_or_else(|| crate::error::RusticError::Memory("vector point table exceeds address space".into()))?;
            if let Err(error) = t.ensure_admitted(n) {
                for &point in seg.points() { t.retire_empty_page(point as usize / OWNER_CHUNK); }
                return Err(error);
            }
        }
        Self::register_locked(&mut state, &mut t, seg);
        Ok(())
    }

    fn register_locked(state: &mut State, t: &mut PointTable, seg: Arc<VectorSegment>) {
        for (ord, &p) in seg.points().iter().enumerate() {
            t.set_loc(p, pack(seg.id, ord as u32));
        }
        state.segments.insert(seg.id, seg);
    }

    /// Next point id to be assigned (persisted in the commit point).
    pub fn next_point_id(&self) -> u64 {
        self.next_point.load(AtomicOrdering::Relaxed)
    }

    /// Next vector segment id (persisted in the commit point).
    pub fn next_segment_id(&self) -> u32 {
        self.next_segment.load(AtomicOrdering::Relaxed)
    }

    /// Ids of the current vector segments, ascending.
    pub fn segment_ids(&self) -> Vec<u32> {
        self.state.read().segments.keys().copied().collect()
    }

    /// A vector segment by id.
    pub fn segment(&self, id: u32) -> Option<Arc<VectorSegment>> {
        self.state.read().segments.get(&id).cloned()
    }

    /// The current vector segments of `field`, ascending id.
    pub fn segments(&self, field: u16) -> Vec<Arc<VectorSegment>> {
        self.state.read().segments.values().filter(|s| s.field == field).cloned().collect()
    }

    /// Segments replaced or dropped since the last call; their files may be
    /// deleted once a commit point no longer references them.
    pub fn take_obsolete(&self) -> Vec<u32> {
        std::mem::take(&mut self.state.write().obsolete)
    }

    /// The vector half of a commit point, consistent with optimizer swaps
    /// running concurrently: the current segment ids and the segments
    /// replaced so far (whose files a commit point listing those ids no
    /// longer references) under one lock, then the next ids, which are
    /// read after the lock and so exceed every listed id. Give `obsolete`
    /// back with [`VectorStore::restore_obsolete`] if the commit fails.
    pub fn commit_point(&self) -> Result<VectorCommitPoint> {
        let (segments, obsolete, ivf, obsolete_files, memory) = {
            let mut st = self.state.write();
            let bytes = st.ivf.values().fold(4096u64, |n, state| n.saturating_add(128).saturating_add((state.num_batches() as u64).saturating_mul(8)))
                .saturating_add(st.ivf_deferred.values().fold(0u64, |n, m| n.saturating_add(128).saturating_add((m.batches.len() as u64).saturating_mul(8))))
                .saturating_add((st.segments.len() as u64).saturating_mul(4));
            let memory = crate::admission::shared().reserve(bytes, "vector commit manifest")?;
            self.uncommitted.store(false, AtomicOrdering::Release);
            // An index lists only points of the segments listed with it, all
            // allocated before `next_point_id` is read below.
            (st.segments.keys().copied().collect(), std::mem::take(&mut st.obsolete),
                st.ivf.values().map(|s| s.manifest()).chain(st.ivf_deferred.values().cloned()).collect(), std::mem::take(&mut st.obsolete_files), memory)
        };
        Ok(VectorCommitPoint { segments, obsolete, next_segment_id: self.next_segment_id(), next_point_id: self.next_point_id(), ivf, obsolete_files, _manifest_memory: Some(memory) })
    }

    /// Re-queue obsolete segment ids of a commit that did not happen.
    pub fn restore_obsolete(&self, ids: Vec<u32>) {
        self.state.write().obsolete.extend(ids);
        self.uncommitted.store(true, AtomicOrdering::Release);
    }

    /// Re-queue obsolete DiskBBQ files of a commit that did not happen.
    pub fn restore_obsolete_files(&self, mut files: std::collections::LinkedList<RetiredIvfFiles>) {
        self.state.write().obsolete_files.append(&mut files);
        self.uncommitted.store(true, AtomicOrdering::Release);
    }

    /// Whether segments were swapped or dropped (seals, merges, vacuums)
    /// since the last commit point: a commit would save redoing them on
    /// reopen.
    pub fn has_uncommitted_changes(&self) -> bool {
        self.uncommitted.load(AtomicOrdering::Acquire)
    }

    /// Register (or replace) the indexing policy of `field`. Takes effect
    /// on the next seal of the field (existing segments keep their stored
    /// copy; a search that asks for a representation they do not store
    /// builds it in memory) and on the next search (rescoring, quantization).
    pub fn set_field_options(&self, field: u16, options: VectorFieldOptions) {
        self.state.write().field_options.insert(field, options);
    }

    /// The registered policy of `field` ([`VectorFieldOptions::default`]
    /// when none was registered).
    pub fn field_options(&self, field: u16) -> VectorFieldOptions {
        self.state.read().field_options.get(&field).copied().unwrap_or_default()
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn mutable_field_options(&self, field: u16) -> VectorFieldOptions {
        self.state.read().field_options.get(&field).copied().unwrap_or_else(VectorFieldOptions::f32_hnsw)
    }

    /// Whether point `p` is alive.
    pub fn is_alive(&self, p: PointId) -> bool {
        self.table.read().is_alive(p)
    }

    /// Owner `(text segment id, doc)` of point `p`.
    pub fn owner(&self, p: PointId) -> Option<(u32, DocId)> {
        self.table.read().owner(p)
    }

    /// Number of alive points of `field`.
    pub fn num_alive(&self, field: u16) -> usize {
        // Match replace(): state before table. Taking table first lets a
        // concurrent optimizer hold state while waiting for this reader,
        // with stats waiting for state in turn.
        let state = self.state.read();
        let t = self.table.read();
        state.segments.values().filter(|s| s.field == field)
            .map(|s| t.live_by_segment.get(&s.id).copied().unwrap_or(0)).sum()
    }

    /// The stored vector of point `p` (alive or not, while still stored).
    pub fn vector(&self, p: PointId) -> Option<Vec<f32>> {
        self.with_vector(p, <[f32]>::to_vec)
    }

    /// Borrow a row while retaining its immutable segment. Resolve location
    /// and segment under the same state-before-table order as replace(), so
    /// an optimizer cannot remove the old segment between the two lookups.
    pub fn with_vector<T>(&self, p: PointId, f: impl FnOnce(&[f32]) -> T) -> Option<T> {
        let (segment, ord) = {
            let state = self.state.read();
            let (id, ord) = self.table.read().loc(p)?;
            (state.segments.get(&id)?.clone(), ord)
        };
        Some(f(segment.vector(ord)))
    }

    /// Visit immutable rows under one consistent location/segment snapshot.
    /// The visitor must not re-enter this store or schedule work on a pool
    /// containing writers. Returning false stops the scan (e.g. invalid data).
    pub fn visit_vectors(&self, points: impl IntoIterator<Item = (DocId, Option<PointId>)>, mut visit: impl FnMut(DocId, &[f32]) -> bool) -> bool {
        self.visit_vector_rows(points, |doc, segment, ord| visit(doc, segment.vector(ord)))
    }

    /// Visit original rows with interpreter-compatible magnitudes. The same
    /// snapshot/visitor restrictions as visit_vectors apply.
    pub fn visit_vectors_with_norm(&self, points: impl IntoIterator<Item = (DocId, Option<PointId>)>, mut visit: impl FnMut(DocId, &[f32], f32) -> bool) -> bool {
        self.visit_vector_rows(points, |doc, segment, ord| visit(doc, segment.vector(ord), segment.original_norm(ord)))
    }

    fn visit_vector_rows(&self, points: impl IntoIterator<Item = (DocId, Option<PointId>)>, mut visit: impl FnMut(DocId, &VectorSegment, u32) -> bool) -> bool {
        let state = self.state.read();
        let table = self.table.read();
        // Consecutive document rows usually share a vector segment. Keep its
        // reference under the snapshot lock rather than hashing its id for
        // every row. Re-resolve on every transition, including A -> B -> A.
        let mut cached: Option<(u32, &VectorSegment)> = None;
        // Document order is not row order, so each row is a cache miss the
        // hardware prefetcher cannot predict. Resolve rows AHEAD documents
        // early and prefetch them; visits stay in document order.
        const AHEAD: usize = 8;
        let mut ring: std::collections::VecDeque<(DocId, &VectorSegment, u32)> = std::collections::VecDeque::with_capacity(AHEAD + 1);
        for (doc, point) in points {
            let resolved = point.and_then(|p| table.loc(p)).and_then(|(id, ord)| {
                let segment = match cached {
                    Some((previous, segment)) if previous == id => segment,
                    _ => {
                        let segment = state.segments.get(&id)?.as_ref();
                        cached = Some((id, segment));
                        segment
                    }
                };
                Some((segment, ord))
            });
            let Some((segment, ord)) = resolved else {
                // Visit what precedes the missing row, as the unbuffered loop did.
                for (doc, segment, ord) in ring.drain(..) {
                    if !visit(doc, segment, ord) { return false; }
                }
                return false;
            };
            let row = segment.vector(ord);
            super::distance::prefetch(row.as_ptr().cast::<u8>(), row.len() * 4);
            ring.push_back((doc, segment, ord));
            if ring.len() > AHEAD {
                let (doc, segment, ord) = ring.pop_front().expect("non-empty ring");
                if !visit(doc, segment, ord) { return false; }
            }
        }
        for (doc, segment, ord) in ring.drain(..) {
            if !visit(doc, segment, ord) { return false; }
        }
        true
    }

    // ---- write path ---------------------------------------------------

    /// Flush: store `rows` (`(doc, vector)` of one field) as a new
    /// appendable chunk. Returns the new point id of each row. No graph is
    /// built. Points stay dead until a text segment pointing at them is
    /// [`attach`](Self::attach)ed. This low-level API requires callers to
    /// exclude optimization until attachment. Document flushing instead uses
    /// [`Self::prepare_append`] and [`Self::attach_prepared`] so encoding and
    /// optimizer graph builds can run concurrently.
    pub fn append(
        &self,
        dir: &dyn Directory,
        field: u16,
        similarity: Similarity,
        dim: u32,
        rows: &[(DocId, &[f32])],
    ) -> Result<Vec<PointId>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut prepared = self.prepare_append(dir, field, similarity, dim, rows)?;
        let points = prepared.points().to_vec();
        self.register(prepared.segment.as_ref().unwrap().clone())?;
        for reservation in &prepared.reservations { reservation.state.store(2, AtomicOrdering::Release); }
        prepared.installed = true;
        self.state.write().last_append.insert(field, std::time::Instant::now());
        Ok(points)
    }

    /// Reserve IDs and write a chunk without making it visible to queries,
    /// optimizer planning or commit manifests. The text writer owns this
    /// guard until all fallible encoding/deletion writes have succeeded.
    pub(crate) fn prepare_append<'a>(
        &'a self,
        dir: &'a dyn Directory,
        field: u16,
        similarity: Similarity,
        dim: u32,
        rows: &[(DocId, &[f32])],
    ) -> Result<PreparedVectorAppend<'a>> {
        let reservation = self.reserve_points(rows.len())?;
        self.prepare_reserved_append(dir, field, similarity, dim, rows, &reservation)
    }

    /// Reserve once at ingestion; abandoned IDs are never reused. No file or
    /// searchable point is published by reservation alone.
    pub(crate) fn reserve_points(&self, count: usize) -> Result<PointReservation> {
        let count64 = u64::try_from(count).map_err(|_| crate::error::RusticError::Memory("vector point count overflow".into()))?;
        // Serialize ID allocation with dense admission. A refusal neither consumes
        // IDs nor exposes a point to graph/source publication.
        let mut table = self.table.write();
        table.collect_retired_pages();
        let first = self.next_point.load(AtomicOrdering::Relaxed);
        let next = first.checked_add(count64).ok_or_else(|| crate::error::RusticError::Memory("vector point IDs exhausted".into()))?;
        let n = usize::try_from(next).map_err(|_| crate::error::RusticError::Memory("vector point table exceeds address space".into()))?;
        if count != 0 {
            let first = usize::try_from(first).map_err(|_| crate::error::RusticError::Memory("vector point table exceeds address space".into()))?;
            for page in first / OWNER_CHUNK..=(n - 1) / OWNER_CHUNK {
                if let Err(error) = table.ensure_admitted(page * OWNER_CHUNK + 1) {
                    for admitted in first / OWNER_CHUNK..=page { table.retire_empty_page(admitted); }
                    return Err(error);
                }
            }
        }
        for p in first..next { table.pending[p as usize] += 1; }
        self.next_point.store(next, AtomicOrdering::Relaxed);
        drop(table);
        Ok(PointReservation { inner: Arc::new(PointReservationState {
            table: Arc::downgrade(&self.table), allocator: self.next_point.clone(), first, count, state: std::sync::atomic::AtomicU8::new(0),
            #[cfg(feature = "mutable-hnsw-poc")]
            abandon: Mutex::new(None),
        }) })
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn reserve_existing_point(&self, point: PointId) -> Result<PointReservation> {
        if point >= self.next_point_id() { return Err(crate::error::RusticError::Segment("unknown reusable point".into())); }
        {
            let mut table = self.table.write();
            table.ensure_admitted(point as usize + 1)?;
            let pending = &mut table.pending[point as usize];
            *pending = pending.checked_add(1).ok_or_else(|| crate::error::RusticError::Memory("point reservation count overflow".into()))?;
        }
        Ok(PointReservation { inner: Arc::new(PointReservationState {
            table: Arc::downgrade(&self.table), allocator: self.next_point.clone(), first: point, count: 1,
            state: std::sync::atomic::AtomicU8::new(0), abandon: Mutex::new(None),
        }) })
    }

    /// Encode a chunk using previously reserved identities. The reservation
    /// remains unavailable while its provisional guard is alive, and becomes
    /// reusable only if that write is abandoned before publication.
    pub(crate) fn prepare_reserved_append<'a>(
        &'a self, dir: &'a dyn Directory, field: u16, similarity: Similarity, dim: u32,
        rows: &[(DocId, &[f32])], reservation: &PointReservation,
    ) -> Result<PreparedVectorAppend<'a>> {
        self.prepare_reserved_appends(dir, field, similarity, dim, rows, &[reservation])
    }

    /// Coalesce multiple ingestion batches into one flush chunk, preserving
    /// the complete row set used by quantizer fitting and optimizer thresholds.
    pub(crate) fn prepare_reserved_appends<'a>(
        &'a self, dir: &'a dyn Directory, field: u16, similarity: Similarity, dim: u32,
        rows: &[(DocId, &[f32])], reservations: &[&PointReservation],
    ) -> Result<PreparedVectorAppend<'a>> {
        let count = reservations.iter().try_fold(0usize, |n, r| n.checked_add(r.inner.count));
        if rows.is_empty() || count != Some(rows.len()) || reservations.iter().any(|r| !r.belongs_to(self)) {
            return Err(crate::error::RusticError::Segment("vector reservation shape/store mismatch".into()));
        }
        let id = self.next_segment.fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed,
            |id| id.checked_add(1)).map_err(|_| crate::error::RusticError::Memory("vector segment IDs exhausted".into()))?;
        let mut claimed: Vec<Arc<PointReservationState>> = Vec::with_capacity(reservations.len());
        for reservation in reservations {
            if reservation.inner.state.compare_exchange(0, 1, AtomicOrdering::AcqRel, AtomicOrdering::Acquire).is_err() {
                for previous in &claimed { previous.state.store(0, AtomicOrdering::Release); }
                return Err(crate::error::RusticError::Segment("vector reservation already in use or published".into()));
            }
            claimed.push(reservation.inner.clone());
        }
        let mut prepared = PreparedVectorAppend { store: self, directory: dir, id, segment: None, document_points: Vec::new(), installed: false,
            reservations: claimed };
        let points: Vec<PointId> = reservations.iter().flat_map(|r| r.points()).collect();
        let mut prows: Vec<(u64, &[f32])> = points.iter().zip(rows).map(|(p, (_, v))| (*p, *v)).collect();
        #[cfg(feature = "mutable-hnsw-poc")]
        if self.registered_mutable_field(field).is_some() {
            let mut seen = std::collections::HashMap::new();
            let mut unique = Vec::with_capacity(prows.len());
            for (point, values) in prows {
                if let Some(previous) = seen.insert(point, values) {
                    if previous.len() != values.len() || previous.iter().zip(values).any(|(a,b)| a.to_bits() != b.to_bits()) {
                        return Err(crate::error::RusticError::Segment("reused point changes vector geometry".into()));
                    }
                } else { unique.push((point, values)); }
            }
            unique.sort_unstable_by_key(|(point, _)| *point);
            prows = unique;
        }
        prepared.document_points = points;
        // Flush never quantizes or builds graphs: both happen at seal time.
        let seg = VectorSegment::write_with(dir, id, field, similarity, dim, &prows, SealSpec::chunk())?;
        prepared.segment = Some(Arc::new(seg));
        Ok(prepared)
    }

    /// Publish chunk registration and its owning document liveness together.
    /// No optimizer lock is held while the text segment is encoded; the short
    /// state -> table critical section prevents observing unowned live chunks.
    pub(crate) fn attach_prepared(self: &Arc<Self>, seg: &Segment, prepared: &mut [PreparedVectorAppend<'_>], hidden: bool) {
        let map = seg.point_map();
        let payload = seg.vector_payload_rows();
        let dels = seg.attachment_deletions(hidden);
        let mut state = self.state.write();
        let mut table = self.table.write();
        for chunk in prepared.iter() {
            debug_assert!(std::ptr::eq(self.as_ref(), chunk.store));
            let vector = chunk.segment.as_ref().expect("prepared vector segment");
            Self::register_locked(&mut state, &mut table, vector.clone());
            state.last_append.insert(vector.field, std::time::Instant::now());
        }
        for field in map.fields() {
            for (doc, point) in map.iter(field) {
                if table.loc(point).is_none() { continue; }
                let keep = !dels.get(doc) && table.loc(point).is_some();
                if table.is_alive(point) && table.owner(point) != Some((seg.id, doc)) && (hidden || !keep) { continue; }
                table.owner.set(point as usize, pack(seg.id, doc));
                table.payload[point as usize] = Some(payload[doc as usize].clone());
                table.set_alive(point, !hidden && keep);
                table.set_hidden(point, hidden && keep);
            }
        }
        if !prepared.is_empty() {
            self.uncommitted.store(true, AtomicOrdering::Release);
        }
        for chunk in prepared { for reservation in &chunk.reservations { reservation.state.store(2, AtomicOrdering::Release); } chunk.installed = true; }
        drop(table);
        drop(state);
        seg.set_vector_store(self.clone());
    }

    /// Make `seg` the owner of every point its point map references: owner
    /// becomes `(seg.id, doc)` and the point is alive iff the doc is live.
    /// Called at open, after flush and when a merged segment is swapped in
    /// (a text merge's only vector work: O(docs) table writes). Also
    /// installs this store as `seg`'s store.
    pub fn attach(self: &Arc<Self>, seg: &Segment) {
        let map = seg.point_map();
        if !map.is_empty() {
            let payload = seg.vector_payload_rows();
            let dels = seg.deletions();
            let mut t = self.table.write();
            for field in map.fields() {
                for (doc, p) in map.iter(field) {
                    if t.loc(p).is_none() { continue; }
                    let alive = !dels.get(doc) && t.loc(p).is_some();
                    if !alive && t.is_alive(p) && t.owner(p) != Some((seg.id, doc)) { continue; }
                    t.owner.set(p as usize, pack(seg.id, doc));
                    t.payload[p as usize] = Some(payload[doc as usize].clone());
                    t.set_alive(p, alive);
                }
            }
        }
        seg.set_vector_store(self.clone());
    }

    /// [`VectorStore::attach`] for a text segment searches do not see yet:
    /// its points of live docs are kept (hidden) but not searchable until
    /// [`VectorStore::reveal`].
    pub fn attach_hidden(self: &Arc<Self>, seg: &Segment) {
        let map = seg.point_map();
        if !map.is_empty() {
            let payload = seg.vector_payload_rows();
            let dels = seg.attachment_deletions(true);
            let mut t = self.table.write();
            for field in map.fields() {
                for (doc, p) in map.iter(field) {
                    if t.loc(p).is_none() { continue; }
                    if t.is_alive(p) && t.owner(p) != Some((seg.id, doc)) { continue; }
                    t.owner.set(p as usize, pack(seg.id, doc));
                    t.payload[p as usize] = Some(payload[doc as usize].clone());
                    let keep = !dels.get(doc) && t.loc(p).is_some();
                    t.set_alive(p, false);
                    t.set_hidden(p, keep);
                }
            }
        }
        seg.set_vector_store(self.clone());
    }

    /// [`VectorStore::attach`] (or, `hidden`, [`VectorStore::attach_hidden`])
    /// of a coalesced seal's segment: its points were owned by the
    /// memtables `from` (writer ids) until now, and every one of them moves
    /// to `seg` (point ids and graph nodes do not change), deleted docs'
    /// included, so no point keeps a memtable owner. Points another
    /// segment owns are left alone, as `attach` does.
    pub fn attach_coalesced(self: &Arc<Self>, seg: &Segment, from: &[u32], hidden: bool) {
        let map = seg.point_map();
        if !map.is_empty() {
            let payload = seg.vector_payload_rows();
            let dels = seg.attachment_deletions(hidden);
            let mut t = self.table.write();
            for field in map.fields() {
                for (doc, p) in map.iter(field) {
                    if t.loc(p).is_none() { continue; }
                    let ours = t.owner(p).is_none_or(|(owner, _)| owner == seg.id || from.contains(&owner));
                    let keep = !dels.get(doc) && t.loc(p).is_some();
                    if t.is_alive(p) && !ours { continue; }
                    t.owner.set(p as usize, pack(seg.id, doc));
                    t.payload[p as usize] = Some(payload[doc as usize].clone());
                    if hidden {
                        t.set_alive(p, false);
                        t.set_hidden(p, keep);
                    } else {
                        t.set_alive(p, keep);
                    }
                }
            }
        }
        seg.set_vector_store(self.clone());
    }

    /// Make the hidden points of `seg` searchable (the segment became
    /// visible); points of docs deleted meanwhile stay dead.
    pub fn reveal(&self, seg: &Segment) {
        let map = seg.point_map();
        if map.is_empty() {
            return;
        }
        let dels = seg.attachment_deletions(true);
        let payload = seg.vector_payload_rows();
        let mut t = self.table.write();
        for field in map.fields() {
            for (doc, p) in map.iter(field) {
                if t.loc(p).is_none() { continue; }
                let alive = !dels.get(doc) && t.loc(p).is_some();
                if !alive && t.is_alive(p) && t.owner(p) != Some((seg.id, doc)) { continue; }
                t.owner.set(p as usize, pack(seg.id, doc));
                t.payload[p as usize] = Some(payload[doc as usize].clone());
                t.set_hidden(p, false);
                t.set_alive(p, alive);
            }
        }
    }

    /// Release every point `seg` owns (e.g. when it is dropped without a
    /// merge). Points re-owned by another segment are untouched.
    pub fn detach(&self, seg: &Segment) {
        let map = seg.point_map();
        let mut t = self.table.write();
        for field in map.fields() {
            for (doc, p) in map.iter(field) {
                if t.owner(p) == Some((seg.id, doc)) {
                    t.owner.set(p as usize, NONE);
                    t.set_alive(p, false);
                    t.set_hidden(p, false);
                    t.payload[p as usize] = None;
                }
            }
        }
    }

    /// Mark point `p` dead if it is still owned by `(segment, doc)`.
    pub fn delete(&self, p: PointId, segment: u32, doc: DocId) {
        {
            let mut t = self.table.write();
            if t.owner(p) == Some((segment, doc)) {
                t.set_alive(p, false);
                t.set_hidden(p, false);
                t.payload[p as usize] = None;
            }
        }
        // A retired document releases its lease even when an unchanged
        // replacement already owns the point. Callers notify only on the
        // first visible deletion of that document.
        #[cfg(feature = "mutable-hnsw-poc")]
        self.queue_mutable_delete(p);
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn queue_mutable_delete(&self, point: u64) {
        for owner in self.mutable_fields.lock().values() { owner.queue_document_delete(point); }
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn mutable_ingestion_batch(&self) -> Result<Vec<super::hnsw::MutableCommitBatch>> {
        let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
        fields.iter().map(|field| field.commit_batch()).collect()
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn drain_mutable_deletes(&self) -> Result<()> {
        let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
        for owner in fields { owner.drain_abandoned()?; }
        Ok(())
    }

    /// Test seam: the next mutable-field batch staged into this store runs
    /// `hook` while its graph ingestion is in flight (rows reserved and
    /// installed, graph preparation not started; the field's in-flight
    /// count is held, so a graph capture is refused). Production never
    /// arms it.
    /// Compiled only for tests (`test-seams`: the engine's tests enable it).
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-seams"))]
    pub fn pause_next_mutable_stage(&self, hook: impl FnOnce() + Send + 'static) {
        *self.stage_pause.lock() = Some(Box::new(hook));
    }

    /// Takes the armed [`Self::pause_next_mutable_stage`] hook, if any.
    #[cfg(all(feature = "mutable-hnsw-poc", any(test, feature = "test-seams")))]
    pub(crate) fn take_stage_pause(&self) -> Option<Box<dyn FnOnce() + Send>> {
        self.stage_pause.lock().take()
    }

    /// Bind the reader-view slot whose views read this store (`None`:
    /// unbind; deletes are then removed without waiting for views).
    pub(crate) fn set_view_clock(&self, slot: Option<&Arc<crate::reader_view::ViewSlot>>) {
        *self.view_clock.write() = slot.map_or_else(std::sync::Weak::new, Arc::downgrade);
    }

    /// Stamp for a mutable-graph delete queued now: the generation of the
    /// next published view (the first that excludes it). `0` without a slot.
    ///
    /// The stamp is the generation of the view the same publish installs:
    /// in NRT mode (the only mode with a bound clock) a delete reaches this
    /// queue either inside `publish_deletions` of a published leaf, called
    /// by `Index::publish_writes` / `Index::publish_sealed` under `&mut
    /// Index` right before `publish_view` (no other publish can interleave),
    /// or immediately for a leaf no view lists yet (memtable being written,
    /// hidden or parked segment), whose first view already excludes the
    /// document. So no view stamped `>= e` counts a delete stamped `e` live.
    pub(crate) fn delete_epoch(&self) -> u64 {
        self.view_clock.read().upgrade().map_or(0, |slot| slot.next_generation())
    }

    /// The oldest published view still alive (`None`: no slot or no view):
    /// a delete stamped `e` may be removed from the graph once this is
    /// `>= e`.
    pub(crate) fn min_live_view(&self) -> Option<u64> {
        self.view_clock.read().upgrade().and_then(|slot| slot.min_live_generation())
    }

    /// Whether any field has a mutable graph (delete repair can be due).
    pub fn has_mutable_fields(&self) -> bool {
        #[cfg(feature = "mutable-hnsw-poc")]
        { !self.mutable_fields.lock().is_empty() }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { false }
    }

    /// Background, bounded mutable-graph delete repair: at most `max`
    /// queued deletes per field. Takes no index lock; call it outside the
    /// owner's locks. Returns whether deletes remain queued.
    pub fn drain_mutable_deletes_batch(&self, max: usize) -> Result<bool> {
        self.drain_mutable_deletes_batch_inner(max)
    }

    /// Real off-heap bytes of `field`'s shared mutable graph: its live
    /// adjacency plus a 4-byte slot entry per allocated node (ES `vex`;
    /// Lucene's graph file likewise holds per-node offsets, so a graph of
    /// one isolated node is not empty), and its code log (quantized codes,
    /// ES `veq` / `veb`). `None`: the field has no mutable graph.
    pub fn mutable_off_heap(&self, field: u16) -> Option<(u64, u64)> {
        #[cfg(feature = "mutable-hnsw-poc")]
        // Cached past a size (see `MutableIngestField::off_heap_bytes`):
        // index stats must not walk a large graph per request.
        { self.registered_mutable_field(field).map(|owner| owner.off_heap_bytes()) }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { let _ = field; None }
    }

    /// Live nodes of `field`'s mutable graph (`None`: no such graph).
    /// Build the mutable fields' node maps of newly published sealed
    /// `segments` on a background thread, off the query path (as a Lucene
    /// reader builds per-segment state when it opens, not on a search).
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn warm_mutable_projections(&self, segments: Vec<Arc<crate::segment::Segment>>) {
        let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
        if fields.is_empty() { return; }
        let segments: Vec<_> = {
            let mut warmed = self.warmed_projections.lock();
            segments.into_iter().filter(|segment| warmed.insert(segment.projection_id())).collect()
        };
        if segments.is_empty() { return; }
        let spawned = std::thread::Builder::new().name("vector-projection-warm".into()).spawn(move || {
            for segment in &segments {
                for field in &fields {
                    if let Some(column) = segment.point_map().column(field.field_id()) {
                        // Best effort: a refused charge leaves it to the query.
                        let _ = field.warm_segment_node_map(segment.projection_id(), &column);
                    }
                }
            }
        });
        drop(spawned);
    }
    pub fn mutable_active_nodes(&self, field: u16) -> Option<u32> {
        #[cfg(feature = "mutable-hnsw-poc")]
        { self.registered_mutable_field(field).map(|owner| owner.graph_stats().active_nodes) }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { let _ = field; None }
    }

    /// Graph insertions into field `field`'s mutable graph since it was
    /// opened (writes plus the open-time catch-up of a lagging checkpoint);
    /// `None` without a mutable graph. A file-recovered copy reports only
    /// the points its shipped checkpoint lacked.
    pub fn mutable_graph_inserts(&self, field: u16) -> Option<u64> {
        #[cfg(feature = "mutable-hnsw-poc")]
        { self.registered_mutable_field(field).map(|owner| owner.graph_inserts()) }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { let _ = field; None }
    }

    /// Reader-pressure gauges over the mutable graphs (zero without them):
    /// queued deletes awaiting physical repair, then retired graph slots and
    /// adjacency bytes held back for older readers. Takes each graph's
    /// writer lock only for its O(1) counters.
    pub fn mutable_pressure(&self) -> MutablePressure {
        #[allow(unused_mut)]
        let mut out = MutablePressure::default();
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
            for owner in fields {
                out.pending_repairs += owner.pending_delete_count() as u64;
                let stats = owner.graph_stats();
                out.retired_slots += stats.retired_slots as u64;
                out.retired_adjacency_bytes += stats.retired_adjacency_bytes as u64;
            }
        }
        out
    }

    fn drain_mutable_deletes_batch_inner(&self, max: usize) -> Result<bool> {
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            // Background maintenance also bounds what retained views pin.
            super::mutable_ingest::enforce_retained_budget(super::mutable_ingest::retained_graph_budget());
            let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
            let mut more = false;
            for owner in fields { more |= owner.drain_some(max)?; }
            Ok(more)
        }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        {
            let _ = max;
            Ok(false)
        }
    }

    // ---- search -------------------------------------------------------

    /// The points of the docs `docs` names (deleted docs already cleared),
    /// as one dense set over point ids: what a `bbq_disk` scan tests per
    /// posting entry. `column(text segment)`: its doc-to-point column and
    /// runs of consecutive docs with consecutive points, whose docs are
    /// copied a word at a time; other segments map each matching doc.
    pub fn point_bits<'c>(&self, docs: &DocFilter, column: impl Fn(u32) -> Option<(EncodedPointColumn<'c>, Option<Arc<[crate::segment::PointRun]>>)>) -> PointBits {
        let mut bits = PointBits::with_capacity(self.next_point_id());
        for (sg, words) in docs.segments() {
            let Some((col, runs)) = column(sg) else { continue };
            map_docs_to_points(&mut bits.words, words, &col, runs.as_deref());
        }
        bits.count = bits.words.iter().map(|w| w.count_ones() as usize).sum();
        bits
    }

    /// [`Self::point_bits`] of one text segment's doc bits, trimmed to the
    /// words holding points: `(first word, words)`.
    pub fn segment_point_words(&self, words: &[u64], col: &EncodedPointColumn<'_>, runs: Option<&[crate::segment::PointRun]>) -> (usize, Vec<u64>) {
        let mut out = Vec::new();
        map_docs_to_points(&mut out, words, col, runs);
        let first = out.iter().position(|&w| w != 0).unwrap_or(out.len());
        let end = out.iter().rposition(|&w| w != 0).map_or(first, |i| i + 1);
        (first, out[first..end].to_vec())
    }

    /// Resolve `docs` to an [`OrdFilter`] over the vector segments of
    /// `field`: `column(text segment id)` gives that segment's `doc ->
    /// PointId` column. One pass over the matching docs (sequential reads
    /// of the point map and point table); dead points are left out.
    pub fn ord_filter<'c>(&self, field: u16, docs: DocFilter, column: impl Fn(u32) -> Option<(EncodedPointColumn<'c>, Option<u64>)>) -> OrdFilter {
        let segs = self.segments(field);
        let mut out: Vec<(u32, Vec<u64>, usize)> =
            segs.iter().map(|s| (s.id, vec![0u64; (s.len() as usize).div_ceil(64)], 0usize)).collect();
        // Contiguous point ranges map `point -> ordinal` by subtraction.
        let ranges: Vec<(u64, u64, bool)> = segs
            .iter()
            .map(|s| {
                let p = s.points();
                match (p.first(), p.last()) {
                    (Some(&a), Some(&b)) => (a, b, b - a + 1 == p.len() as u64),
                    _ => (1, 0, false),
                }
            })
            .collect();
        let all_contiguous = ranges.iter().all(|r| r.2);
        let t = self.table.read();
        let mut recount = false;
        // Contiguous ranges by first point, and which segments are whole
        // (every point alive).
        let mut by_lo: Vec<(u64, u64, usize)> = ranges.iter().enumerate().map(|(i, r)| (r.0, r.1, i)).collect();
        by_lo.sort_unstable();
        let whole: Vec<bool> = segs.iter().map(|s| t.live_by_segment.get(&s.id).is_some_and(|&n| n == s.len() as usize)).collect();
        for (sg, words) in docs.segments() {
            let Some((col, identity)) = column(sg) else { continue };
            if let (Some(base), true) = (identity, all_contiguous) {
                // Point `base + d` of doc `d`; every segment owns the whole
                // point range `[lo, hi]`: ordinal = base + d - lo. Word-level
                // shift of the doc bits, masked by the alive bits.
                let n_col = col.num_docs() as i64;
                for (i, s) in segs.iter().enumerate() {
                    let (lo, _, _) = ranges[i];
                    let shift = base as i64 - lo as i64; // ordinal = doc + shift
                    let e = &mut out[i];
                    let n_ord = s.len() as i64;
                    // Only the ordinals this text segment's docs map to:
                    // `[shift, shift + n_col)` within the segment.
                    let first = shift.clamp(0, n_ord) as usize / 64;
                    let end = ((shift + n_col).clamp(0, n_ord) as usize).div_ceil(64);
                    if first >= end {
                        continue;
                    }
                    for (w, dst) in e.1.iter_mut().enumerate().take(end).skip(first) {
                        let o0 = (w * 64) as i64;
                        let d0 = o0 - shift;
                        let mut bits = shifted_word(words, d0) & t.alive_word(lo as i64 + o0);
                        // Docs past the column and ordinals past the segment.
                        if d0 + 64 > n_col {
                            bits &= low_mask(n_col - d0);
                        }
                        if o0 + 64 > n_ord {
                            bits &= low_mask(n_ord - o0);
                        }
                        e.2 += (bits & !*dst).count_ones() as usize;
                        *dst |= bits;
                    }
                }
                continue;
            }
            // Every vector segment owns a contiguous point range (the
            // usual layout: chunks and their merges keep point order): a
            // doc's point finds its segment in the few ranges (ordinal =
            // point - first point) instead of the point table's location
            // array (a cache miss per doc), and segments whose points are
            // all alive need no liveness test. Counted once below.
            if all_contiguous {
                for (wi, &w0) in words.iter().enumerate() {
                    let mut w = w0;
                    while w != 0 {
                        let d = (wi * 64) as u32 + w.trailing_zeros();
                        w &= w - 1;
                        let Some(p) = col.get(d) else { continue };
                        let at = by_lo.partition_point(|r| r.0 <= p);
                        let Some(&(lo, hi, i)) = at.checked_sub(1).map(|at| &by_lo[at]) else { continue };
                        if p > hi || (!whole[i] && !t.is_alive(p)) {
                            continue;
                        }
                        let o = (p - lo) as usize;
                        out[i].1[o >> 6] |= 1u64 << (o & 63);
                    }
                }
                recount = true;
                continue;
            }
            for (wi, &w0) in words.iter().enumerate() {
                let mut w = w0;
                while w != 0 {
                    let d = (wi * 64) as u32 + w.trailing_zeros();
                    w &= w - 1;
                    let Some(p) = col.get(d) else { continue };
                    if !t.is_alive(p) {
                        continue;
                    }
                    // Which segment: the single/contiguous fast path, else
                    // the point table's location.
                    let hit = if out.len() == 1 && ranges[0].2 && p >= ranges[0].0 && p <= ranges[0].1 {
                        Some((0usize, (p - ranges[0].0) as u32))
                    } else {
                        t.loc(p).and_then(|(sid, o)| out.binary_search_by_key(&sid, |(s, _, _)| *s).ok().map(|i| (i, o)))
                    };
                    if let Some((i, o)) = hit {
                        let e = &mut out[i];
                        let m = 1u64 << (o & 63);
                        if let Some(x) = e.1.get_mut(o as usize >> 6) {
                            if *x & m == 0 {
                                *x |= m;
                                e.2 += 1;
                            }
                        }
                    }
                }
            }
        }
        drop(t);
        if recount {
            for e in &mut out { e.2 = e.1.iter().map(|w| w.count_ones() as usize).sum(); }
        }
        OrdFilter { segs: out, docs }
    }

    /// Fallible index-wide search, including the mutable backend. The older
    /// infallible payload-search API below remains useful for flat columns;
    /// mutable graph/source failures must be propagated to callers.
    pub fn try_search(&self, field: u16, query: &[f32], k: usize, ef: usize,
        filter: Option<&[PointId]>, cfg: &PlannerConfig) -> Result<(Vec<PointHit>, SearchStats)> {
        let bits = filter.map(|points| PointBits::from_points(points, self.next_point_id()));
        self.try_search_filtered(field, query, k, ef, bits.as_ref().map(PointFilter::Points), cfg)
    }

    pub fn try_search_filtered(&self, field: u16, query: &[f32], k: usize, ef: usize,
        filter: Option<PointFilter<'_>>, cfg: &PlannerConfig) -> Result<(Vec<PointHit>, SearchStats)> {
        #[cfg(feature = "mutable-hnsw-poc")]
        if let Some(owner) = self.mutable_query_field(field) {
            use super::vespa_policy::*;
            if k == 0 { return Ok((Vec::new(), SearchStats::default())); }
            let segments = self.segments(field);
            let options = self.field_options(field);
            let table = self.table.read();
            let mut points: Vec<_> = segments.iter().flat_map(|s| s.points().iter().copied()).filter(|&point| {
                if !table.is_alive(point) { return false; }
                match filter {
                    None => true,
                    Some(PointFilter::Points(bits)) => bits.contains(point),
                    Some(PointFilter::Docs(docs)) => table.owner(point).is_some_and(|(s,d)| docs.contains(s,d)),
                    Some(PointFilter::Ords(ordinals)) => table.owner(point).is_some_and(|(s,d)| ordinals.docs.contains(s,d)),
                }
            }).collect();
            // Sort, plan and traverse without the table lock (writers
            // reserve points and flip liveness under it); owners of the hits
            // are resolved under a fresh short guard below.
            drop(table);
            points.sort_unstable(); points.dedup();
            let matched = u32::try_from(points.len()).map_err(|_| crate::error::RusticError::InvalidQuery("mutable match count overflow".into()))?;
            let exact = cfg.full_scan_threshold == usize::MAX || (filter.is_some() && points.len() <= ef);
            let (target, rescore) = crate::query::mutable_knn::rescore_budget(k, exact, cfg.oversample, options)?;
            // Never below the matched rows: during an `index_options`
            // migration's swap the successor admits rows this graph lacks,
            // and the graph is read after the points were collected.
            let limit = owner.document_limit().max(matched);
            let preparation = choose_filter_preparation(PreparationInput {
                approximate: !exact, has_index: true, estimated_query_ratio: 1.0,
            }, PolicyConfig::default()).map_err(|e| crate::error::RusticError::InvalidQuery(e.to_string()))?;
            let prepared = if exact { PreparedGlobalFilter::NotBuilt } else {
                PreparedGlobalFilter::Active(FilterStatistics { size: limit, count: matched })
            };
            let plan = preparation.finish(HitBudget {
                target_hits: u32::try_from(target).map_err(|_| crate::error::RusticError::InvalidQuery("mutable target overflow".into()))?,
                explore_additional_hits: u32::try_from(ef.max(target) - target).map_err(|_| crate::error::RusticError::InvalidQuery("mutable beam overflow".into()))?,
            }, limit, prepared).map_err(|e| crate::error::RusticError::InvalidQuery(e.to_string()))?;
            let result = owner.search_points(query, &plan, &points, None)?;
            let table = self.table.read();
            // A hit released by a concurrent writer since (detached or
            // deleted) no longer matches.
            let mut hits: Vec<_> = result.hits.into_iter().filter_map(|(point, score)| {
                if !table.is_alive(point) { return None; }
                let (segment, doc) = table.owner(point)?;
                Some(PointHit { point, segment, doc, score })
            }).collect();
            drop(table);
            let mut rescore_ops = 0;
            if rescore && plan.route != Route::Exact {
                for hit in &mut hits {
                    hit.score = self.with_vector(hit.point, |v| owner.similarity().score(query, v))
                        .ok_or_else(|| crate::error::RusticError::Segment("mutable rescore source absent".into()))?;
                    rescore_ops += 1;
                }
            }
            hits.sort_unstable_by(|a,b| b.score.total_cmp(&a.score).then(a.point.cmp(&b.point)));
            hits.truncate(k);
            return Ok((hits, SearchStats { graph_segments: usize::from(plan.route != Route::Exact),
                exact_segments: usize::from(plan.route == Route::Exact), vector_ops: result.vector_ops + rescore_ops,
                ..Default::default() }));
        }
        self.search_filtered(field, query, k, ef, filter, cfg)
    }

    /// Top-`k` over the alive points of `field` (restricted to `filter`
    /// when given), across every vector segment in parallel. `ef` is the
    /// HNSW beam width. Hits are score-descending, ties by point id.
    ///
    /// Convenience over [`search_filtered`](Self::search_filtered): the
    /// point ids are copied into a [`PointBits`] (one bit set per id). Hot
    /// paths should build the [`PointBits`] / [`DocFilter`] once and call
    /// `search_filtered` directly.
    pub fn search(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&[PointId]>,
        cfg: &PlannerConfig,
    ) -> Result<(Vec<PointHit>, SearchStats)> {
        let bits = filter.map(|f| PointBits::from_points(f, self.next_point_id()));
        self.search_filtered(field, query, k, ef, bits.as_ref().map(PointFilter::Points), cfg)
    }

    /// [`search`](Self::search) with a pre-built filter. No per-query work
    /// proportional to the number of matching points happens here: the
    /// filter is consulted per visited graph node (O(1) bit test), and only
    /// segments the planner searches exactly enumerate their matches.
    pub fn search_filtered(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<PointFilter<'_>>,
        cfg: &PlannerConfig,
    ) -> Result<(Vec<PointHit>, SearchStats)> {
        // Segments, index and point table as of one instant: the table is
        // taken before the state guard is released (state -> table, the
        // order of every swap), so a swap either happened before (and the
        // snapshot lists its segment) or waits for this search to release
        // the table (and every location still names a snapshot segment).
        let st = self.state.read();
        let _registry_memory = crate::admission::shared().reserve((st.segments.len() as u64).saturating_mul(128).saturating_add((k as u64).saturating_mul(st.segments.len() as u64).saturating_mul(32)).saturating_add(4096), "vector search registry")?;
        let segs: Vec<Arc<VectorSegment>> = st.segments.values().filter(|s| s.field == field).cloned().collect();
        let ivf = st.ivf.get(&field).cloned();
        let opts = st.field_options.get(&field).copied().unwrap_or_default();
        if segs.is_empty() || k == 0 || filter.is_some_and(|f| f.is_empty()) {
            return Ok((Vec::new(), SearchStats::default()));
        }
        let table = self.table.read();
        drop(st);
        let t: &PointTable = &table;
        let total: usize = segs.iter().map(|s| s.len() as usize).sum();
        // DiskBBQ: the field's shared index, unless the request asks for
        // full precision (`PlannerConfig::exact`).
        if let (Some(ivf_options), false) = (opts.ivf, cfg.quantization == Quantization::None) {
            let (mut all, stats) = match ivf.as_ref() {
                Some(state) => search_ivf_field(state, &segs, t, query, k, ef, filter, total, cfg, &opts, &ivf_options)?,
                // No index yet (below the flat threshold, or not trained):
                // flat, as ES searches a small segment, without building
                // in-memory codes.
                None => {
                    let ops0 = super::hnsw::vector_ops();
                    let oversample = cfg.oversample.unwrap_or(opts.oversample);
                    let estimate = (oversample <= 0.0).then_some(super::es_quant::EsQuant { bits: ivf_options.doc_bits(), confidence_interval: None, osq: true });
                    let mut all = Vec::new();
                    for s in &segs {
                        let pts = s.points();
                        all.extend(search_unindexed(s, t, query, k, filter, total, None, estimate)?.into_iter().map(|(o, score)| (pts[o as usize], score)));
                    }
                    (all, SearchStats { exact_segments: segs.len(), vector_ops: super::hnsw::vector_ops().wrapping_sub(ops0), ..Default::default() })
                }
            };
            all.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
            all.dedup_by_key(|hit| hit.0);
            top_k(&mut all, k);
            let hits = all.into_iter().filter_map(|(p, score)| t.owner(p).map(|(segment, doc)| PointHit { point: p, segment, doc, score })).collect();
            return Ok((hits, stats));
        }
        let run = |s: &Arc<VectorSegment>| -> Result<_> {
            let (hits, st) = search_segment(s, t, query, k, ef, filter, total, cfg, &opts)?;
            let pts = s.points();
            Ok((hits.into_iter().map(|(o, sc)| (pts[o as usize], sc)).collect::<Vec<_>>(), st))
        };
        // Search holds the point-table (and often its caller's shard) read
        // lock. The global Rayon pool also runs writes: joining there can
        // steal a writer waiting for our own read lock. Use the isolated
        // query pool only from a non-Rayon caller; cross-pool installs from
        // a Rayon worker can steal from that worker's original pool too.
        let per_seg: Result<Vec<(Vec<(PointId, f32)>, SearchStats)>> =
            if segs.len() <= 1 || rayon::current_thread_index().is_some() {
                segs.iter().map(run).collect()
            } else {
                crate::query::searcher::in_search_pool(|| segs.par_iter().map(run).collect())
            };
        let mut stats = SearchStats::default();
        let mut all: Vec<(PointId, f32)> = Vec::new();
        for (hits, st) in per_seg? {
            stats += st;
            all.extend(hits);
        }
        // An unchanged mutable replacement retains its point identity. During
        // backend retirement both source chunks can still contain that point.
        // Deduplicate before truncation so copies cannot consume result slots.
        all.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
        all.dedup_by_key(|hit| hit.0);
        top_k(&mut all, k);
        let hits = all
            .into_iter()
            .filter_map(|(p, score)| t.owner(p).map(|(segment, doc)| PointHit { point: p, segment, doc, score }))
            .collect();
        Ok((hits, stats))
    }

    /// Warm every sealed segment for the searches its field's options run
    /// ([`VectorSegment::warm`]): segments opened from disk are otherwise
    /// decoded, quantized and paged in by the first query that reaches them
    /// (tens of ms at 100k x 1024 dims). Segments built by the optimizer are
    /// warmed before they become visible. Idempotent and cheap once warm.
    pub fn warm(&self) {
        for field in self.fields() {
            let options = self.field_options(field);
            // DiskBBQ rows are read from the mapping only: nothing to load.
            if options.ivf.is_some() {
                continue;
            }
            // A field with a shard graph is searched through it (its codes
            // live in the graph's code log): its segments are payload only,
            // and a per-segment quantized index would duplicate the codes
            // in anonymous memory (1.2 GB of a node's 2.8 GB at 800k int8
            // 1536-d rows).
            #[cfg(feature = "mutable-hnsw-poc")]
            if self.mutable_query_field(field).is_some() {
                continue;
            }
            for s in self.segments(field) {
                s.note_int4_confidence_interval(options.confidence_interval);
                s.warm(options.quantization);
            }
        }
    }

    // ---- optimizer ----------------------------------------------------

    /// O(segments), without scanning point rows or loading graph data. Keep
    /// the same state -> table lock order as segment replacement.
    pub fn optimizer_backlog(&self) -> OptimizerBacklog {
        // Do not nest the mutable-owner lock under state/table locks: owner
        // registration can access the store while holding mutable_fields.
        let mut backlog = OptimizerBacklog::default();
        #[cfg(feature = "mutable-hnsw-poc")]
        { backlog.mutable_fields = self.mutable_fields.lock().len() + self.migrations.lock().len(); }
        let st = self.state.read();
        let table = self.table.read();
        for s in st.segments.values() {
            if s.is_sealed() {
                backlog.sealed_segments += 1;
                backlog.graph_segments += usize::from(s.has_graph());
            } else {
                backlog.unsealed_segments += 1;
                backlog.unsealed_vectors += s.len() as usize;
                backlog.searchable_unsealed_vectors += table.live_by_segment.get(&s.id).copied().unwrap_or(0);
            }
        }
        backlog
    }

    /// Offer fresh chunks of every field an indexing pass, without spending
    /// the pass on unrelated vacuum or merge work. The thresholds and small
    /// sealed inputs are identical to the indexing phase of [`Self::optimize`].
    /// A scheduler can call this on every shard before maintenance begins.
    pub fn index_pending(&self, dir: &dyn Directory, policy: &OptimizerPolicy) -> Result<OptimizeReport> {
        let _guard = self.optimize_lock.lock();
        let mut report = OptimizeReport::default();
        let mut deferred = None;
        for field in self.fields() {
            // Do not walk every indexed point twice per background round.
            // Dead sealed inputs belong to maintenance; indexing only needs
            // to discard dead chunks before deciding whether a seal is due.
            for s in self.segments(field).into_iter().filter(|s| !s.is_sealed()) {
                if self.alive_count(&s) == 0 {
                    self.replace(&[s.id], None);
                    report.dropped += 1;
                }
            }
            self.index_field(dir, policy, field, &mut report)?;
            // DiskBBQ: train, or add the rows not yet indexed (ES indexes
            // a segment at flush).
            report.ivf_steps += ivf_isolated(field, self.ivf_insert(dir, field, Rebuild::Seal), &mut deferred)?;
        }
        deferred.map_or(Ok(report), Err)
    }

    /// Perform at most one vacuum or merge rebuild across all fields. Fully
    /// dead segments can be dropped without consuming the rebuild allowance.
    /// Fresh chunks are left to [`Self::index_pending`], so the caller can
    /// prioritize indexing across shards before making another maintenance
    /// pass. A rebuild already started is not interrupted.
    pub fn maintain_once(&self, dir: &dyn Directory, policy: &OptimizerPolicy) -> Result<OptimizeReport> {
        let _guard = self.optimize_lock.lock();
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
            for field in fields { field.maintain()?; }
        }
        let mut report = OptimizeReport::default();
        let mut fields = self.fields();
        let after = self.maintenance_after.load(AtomicOrdering::Relaxed);
        let start = fields.partition_point(|field| u32::from(*field) <= after);
        fields.rotate_left(start);
        let mut deferred = None;
        // DiskBBQ: one bounded index step (split, drop, compact) for every
        // field; a field's failure is surfaced after the others ran.
        for &field in &fields {
            self.drop_dead(field, &mut report);
            if ivf_isolated(field, self.ivf_step(dir, field, IvfWork::Maintain { force: false }, Rebuild::Merge), &mut deferred)? {
                report.ivf_steps += 1;
            }
        }
        for field in fields {
            for s in self.segments(field).into_iter().filter(|s| s.is_sealed()) {
                let dead = s.len() as usize - self.alive_count(&s);
                if dead as f32 > policy.vacuum_ratio() * s.len() as f32
                    && self.reclaimable_count(&s) as f32 > policy.vacuum_ratio() * s.len() as f32 {
                    // Advance before the build too: a failing field must
                    // not consume every subsequent round's allowance.
                    self.maintenance_after.store(u32::from(field), AtomicOrdering::Relaxed);
                    report.vectors_indexed += self.rebuild(dir, std::slice::from_ref(&s), policy)?;
                    report.vacuumed += 1;
                    return deferred.map_or(Ok(report), Err);
                }
            }
            if let Some(plan) = self.plan_merge(field, policy) {
                self.maintenance_after.store(u32::from(field), AtomicOrdering::Relaxed);
                report.vectors_indexed += self.rebuild(dir, &plan, policy)?;
                report.merged += 1;
                return deferred.map_or(Ok(report), Err);
            }
        }
        deferred.map_or(Ok(report), Err)
    }

    fn index_field(&self, dir: &dyn Directory, policy: &OptimizerPolicy, field: u16, report: &mut OptimizeReport) -> Result<()> {
        let chunks: Vec<Arc<VectorSegment>> = self.segments(field).into_iter().filter(|s| !s.is_sealed()).collect();
        if let Some(first) = chunks.first() {
            let threshold = policy.seal_threshold(first.dim);
            let total: usize = chunks.iter().map(|s| s.len() as usize).sum();
            let idle = policy.seal_idle.is_some_and(|d| {
                self.state.read().last_append.get(&field).is_none_or(|t| t.elapsed() >= d)
            });
            if total > threshold || idle {
                let mut inputs = chunks;
                inputs.extend(self.small_sealed(field, threshold));
                report.vectors_indexed += self.rebuild(dir, &inputs, policy)?;
                report.sealed += 1;
            }
        }
        Ok(())
    }

    /// Run the indexing, vacuum and merge optimizers once over every field.
    /// Builds happen outside the store locks; searches keep running on the
    /// old segments until the swap. Replaced files are reported by
    /// [`take_obsolete`](Self::take_obsolete).
    ///
    /// Indexing seals a field's appendable chunks when they exceed
    /// [`OptimizerPolicy::seal_threshold`] or have been idle for
    /// [`OptimizerPolicy::seal_idle`]; sealed segments below the threshold
    /// are folded into the same graph-free payload compaction.
    pub fn optimize(&self, dir: &dyn Directory, policy: &OptimizerPolicy) -> Result<OptimizeReport> {
        let _guard = self.optimize_lock.lock();
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
            for field in fields { field.maintain()?; }
        }
        let mut report = OptimizeReport::default();
        let mut deferred = None;
        for field in self.fields() {
            self.drop_dead(field, &mut report);
            // Indexing optimizer: seal the appendable chunks.
            self.index_field(dir, policy, field, &mut report)?;
            // Vacuum optimizer.
            for s in self.segments(field).into_iter().filter(|s| s.is_sealed()) {
                let alive = self.alive_count(&s);
                let dead = s.len() as usize - alive;
                if dead as f32 > policy.vacuum_ratio() * s.len() as f32
                    && self.reclaimable_count(&s) as f32 > policy.vacuum_ratio() * s.len() as f32 {
                    report.vectors_indexed += self.rebuild(dir, std::slice::from_ref(&s), policy)?;
                    report.vacuumed += 1;
                }
            }
            // Merge optimizer.
            while let Some(plan) = self.plan_merge(field, policy) {
                report.vectors_indexed += self.rebuild(dir, &plan, policy)?;
                report.merged += 1;
            }
            report.ivf_steps += ivf_isolated(field, self.ivf_steps(dir, field, false), &mut deferred)?;
        }
        deferred.map_or(Ok(report), Err)
    }

    /// Force merge (ES `_forcemerge?max_num_segments=N`): seal every
    /// appendable chunk and combine each field's segments down to at most
    /// `max_segments` durable payload segments, bypassing the thresholds.
    /// The mutable graph topology is retained; only payload files are combined.
    /// Then vacuums as [`optimize`](Self::optimize) does.
    pub fn force_merge(&self, dir: &dyn Directory, policy: &OptimizerPolicy, max_segments: usize) -> Result<OptimizeReport> {
        let _guard = self.optimize_lock.lock();
        self.repair_before_compaction()?;
        let target = max_segments.max(1);
        let mut report = OptimizeReport::default();
        let mut deferred = None;
        for field in self.fields() {
            self.drop_dead(field, &mut report);
            let segs = self.segments(field);
            let chunks: Vec<Arc<VectorSegment>> = segs.iter().filter(|s| !s.is_sealed()).cloned().collect();
            let mut sealed: Vec<(usize, Arc<VectorSegment>)> =
                segs.iter().filter(|s| s.is_sealed()).map(|s| (self.alive_count(s), s.clone())).collect();
            sealed.sort_by_key(|(n, s)| (*n, s.id));
            let units = sealed.len() + usize::from(!chunks.is_empty());
            if units > target || !chunks.is_empty() {
                // Combine the chunk group and the smallest sealed segments
                // so exactly `target` segments remain.
                let take_sealed = (units.saturating_sub(target) + 1).saturating_sub(usize::from(!chunks.is_empty())).min(sealed.len());
                let mut inputs = chunks.clone();
                inputs.extend(sealed[..take_sealed].iter().map(|(_, s)| s.clone()));
                if inputs.len() >= 2 || !chunks.is_empty() {
                    report.vectors_indexed += self.rebuild(dir, &inputs, policy)?;
                    if !chunks.is_empty() {
                        report.sealed += 1;
                    }
                    if take_sealed > 0 {
                        report.merged += 1;
                    }
                }
            }
            // Lucene's forced merge treats a segment with deletions as not
            // merged (`findForcedMerges`): it is rewritten without them
            // whatever their share, so a force-merged index holds no dead
            // rows. A routine vacuum waits for `vacuum_deleted_ratio`; here
            // an update-heavy shard kept up to that share of dead f32 rows.
            for s in self.segments(field).into_iter().filter(|s| s.is_sealed()) {
                let dead = s.len() as usize - self.alive_count(&s);
                if dead > 0 {
                    report.vectors_indexed += self.rebuild(dir, std::slice::from_ref(&s), policy)?;
                    report.vacuumed += 1;
                }
            }
            report.ivf_steps += ivf_isolated(field, self.ivf_steps(dir, field, true), &mut deferred)?;
        }
        deferred.map_or(Ok(report), Err)
    }

    /// ES `_forcemerge?only_expunge_deletes=true`: rebuild every sealed
    /// payload segment holding a dead point without it, and drop fully dead
    /// payload segments. The mutable graph is not rebuilt.
    pub fn expunge_deletes(&self, dir: &dyn Directory, policy: &OptimizerPolicy) -> Result<OptimizeReport> {
        let _guard = self.optimize_lock.lock();
        self.repair_before_compaction()?;
        let mut report = OptimizeReport::default();
        let mut deferred = None;
        for field in self.fields() {
            self.drop_dead(field, &mut report);
            for s in self.segments(field).into_iter().filter(|s| s.is_sealed()) {
                if self.alive_count(&s) < s.len() as usize {
                    report.vectors_indexed += self.rebuild(dir, std::slice::from_ref(&s), policy)?;
                    report.vacuumed += 1;
                }
            }
            report.ivf_steps += ivf_isolated(field, self.ivf_steps(dir, field, true), &mut deferred)?;
        }
        deferred.map_or(Ok(report), Err)
    }

    /// Explicit compaction (force merge, expunge deletes) first repairs the
    /// mutable graphs' queued deletes no live reader view still needs: a
    /// dead point the graph still holds keeps its durable row (`rebuild`),
    /// so without the repair it could not be reclaimed. Deletes an older
    /// view still needs stay queued and their rows stay, as before.
    fn repair_before_compaction(&self) -> Result<()> {
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let fields: Vec<_> = self.mutable_fields.lock().values().cloned().collect();
            for field in fields { field.drain_abandoned()?; }
        }
        Ok(())
    }

    fn fields(&self) -> Vec<u16> {
        let mut fields: Vec<u16> = self.state.read().segments.values().map(|s| s.field).collect();
        fields.sort_unstable();
        fields.dedup();
        fields
    }

    /// Drop fully-dead segments of `field` (no build).
    fn drop_dead(&self, field: u16, report: &mut OptimizeReport) {
        for s in self.segments(field) {
            if self.alive_count(&s) == 0 {
                self.replace(&[s.id], None);
                report.dropped += 1;
            }
        }
    }

    /// Sealed segments of `field` with fewer than `threshold` alive points.
    fn small_sealed(&self, field: u16, threshold: usize) -> Vec<Arc<VectorSegment>> {
        self.segments(field).into_iter().filter(|s| s.is_sealed() && self.alive_count(s) < threshold).collect()
    }

    /// Pick sealed segments to combine: every segment below the seal
    /// threshold (when there are at least two), else the smallest ones
    /// (never the largest) under the doubling rule (see module docs).
    fn plan_merge(&self, field: u16, policy: &OptimizerPolicy) -> Option<Vec<Arc<VectorSegment>>> {
        let max = policy.max_segments.max(1);
        let mut sealed: Vec<(usize, Arc<VectorSegment>)> = self
            .segments(field)
            .into_iter()
            .filter(|s| s.is_sealed())
            .map(|s| (self.alive_count(&s), s))
            .collect();
        if let Some((_, first)) = sealed.first() {
            let threshold = policy.seal_threshold(first.dim);
            let small: Vec<Arc<VectorSegment>> = sealed.iter().filter(|(n, _)| *n < threshold).map(|(_, s)| s.clone()).collect();
            if small.len() >= 2 {
                return Some(small);
            }
        }
        if let Some((_, first)) = sealed.first() {
            let row_bytes = (first.dim as usize).max(1) * 4;
            match policy.payload_merge {
                PayloadMerge::Capped => {}
                PayloadMerge::Tiered => return plan_tiered(sealed, row_bytes),
                PayloadMerge::DeletesOnly => {
                    if let Some(plan) = plan_floor(sealed.clone(), row_bytes) { return Some(plan); }
                    sealed.sort_by_key(|(n, s)| (*n, s.id));
                    let sizes: Vec<usize> = sealed.iter().map(|(n, _)| *n).collect();
                    let take = plan_ceiling(&sizes, max, row_bytes);
                    return (take >= 2).then(|| sealed[..take].iter().map(|(_, s)| s.clone()).collect());
                }
            }
        }
        if sealed.len() <= max {
            return None;
        }
        sealed.sort_by_key(|(n, s)| (*n, s.id));
        let count = sealed.len();
        sealed.pop(); // the largest is never re-indexed
        let need = (count - max + 1).max(2).min(sealed.len());
        let mut chosen: Vec<(usize, Arc<VectorSegment>)> = sealed[..need].to_vec();
        let doubles = |c: &[(usize, Arc<VectorSegment>)]| {
            let sum: usize = c.iter().map(|x| x.0).sum();
            let largest = c.last().map_or(0, |x| x.0);
            largest <= sum - largest
        };
        while !doubles(&chosen) && chosen.len() < sealed.len() {
            chosen.push(sealed[chosen.len()].clone());
        }
        if !doubles(&chosen) {
            if count <= 2 * max {
                return None;
            }
            chosen = sealed[..2].to_vec(); // hard cap
        }
        (chosen.len() >= 2).then(|| chosen.into_iter().map(|(_, s)| s).collect())
    }

    #[cfg(test)]
    pub(crate) fn plan_merge_for_test(&self, field: u16, policy: &OptimizerPolicy) -> Option<Vec<u32>> {
        self.plan_merge(field, policy).map(|plan| plan.iter().map(|s| s.id).collect())
    }

    /// Points of `s` the optimizer must keep: alive or hidden
    /// ([`PointTable::is_kept`]), or dead but pinned by a durable graph
    /// checkpoint. O(1) from the table's kept count, plus O(pinned) when a
    /// checkpoint pins points (every maintenance pass asks this of every
    /// segment: a scan of each segment's points made the pass O(store)).
    fn alive_count(&self, s: &VectorSegment) -> usize {
        let pinned = self.checkpoint_pinned_points(s.field);
        let t = self.table.read();
        let counted = Self::kept_count(&t, s, pinned.as_deref());
        #[cfg(test)]
        assert_eq!(counted, Self::kept_scan(&t, s, pinned.as_deref()), "kept count of segment {} disagrees with its points", s.id);
        counted
    }

    fn kept_count(t: &PointTable, s: &VectorSegment, pinned: Option<&std::collections::HashSet<u64>>) -> usize {
        let n = n_points(s);
        let kept = t.kept_by_segment.get(&s.id).copied().unwrap_or(0);
        if kept >= n { return n; }
        match pinned {
            Some(pinned) if !pinned.is_empty() => kept + pinned.iter()
                .filter(|&&p| !t.is_kept(p) && t.loc(p).is_some_and(|(segment, _)| segment == s.id)).count(),
            _ => kept,
        }
    }

    /// The count [`Self::kept_count`] answers, by scanning the segment's
    /// points (the census before the kept counts; tests check the two agree).
    #[cfg(test)]
    fn kept_scan(t: &PointTable, s: &VectorSegment, pinned: Option<&std::collections::HashSet<u64>>) -> usize {
        let mut kept = KeptCursor::new(t);
        s.points().iter().filter(|&&p| kept.is_kept(p) || pinned.is_some_and(|pinned| pinned.contains(&p))).count()
    }

    /// Rows a rebuild of `s` would drop now: dead points no mutable graph
    /// still holds (repair pending) and no durable graph checkpoint pins.
    /// A segment whose dead rows are all held is not vacuumed: the rewrite
    /// would keep them, and run again every pass until the repair.
    fn reclaimable_count(&self, s: &VectorSegment) -> usize {
        let pinned = self.checkpoint_pinned_points(s.field);
        #[cfg(feature = "mutable-hnsw-poc")]
        let graph = self.registered_mutable_field(s.field);
        let dead: Vec<u64> = {
            let mut t = self.table.read();
            let mut dead = Vec::new();
            for chunk in s.points().chunks(TABLE_SCAN_BUMP) {
                let mut kept = KeptCursor::new(&t);
                dead.extend(chunk.iter().copied().filter(|&p| !kept.is_kept(p)));
                parking_lot::RwLockReadGuard::bump(&mut t);
            }
            dead
        };
        dead.into_iter().filter(|p| {
            if pinned.as_ref().is_some_and(|pinned| pinned.contains(p)) { return false; }
            #[cfg(feature = "mutable-hnsw-poc")]
            if graph.as_ref().is_some_and(|field| field.holds_point(*p)) { return false; }
            true
        }).count()
    }

    /// Dead points of `field` whose rows the newest durable graph
    /// checkpoint still references (see `MutableIngestField`'s checkpoint
    /// pins): kept like live rows until a newer checkpoint is durable.
    /// Taken before any store lock.
    fn checkpoint_pinned_points(&self, field: u16) -> Option<Arc<std::collections::HashSet<u64>>> {
        #[cfg(feature = "mutable-hnsw-poc")]
        { self.registered_mutable_field(field).and_then(|field| field.checkpoint_pinned_points()) }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { let _ = field; None }
    }

    /// Compact durable vector payloads without constructing an ANN graph.
    /// The mutable field owner retains topology and stable point identities.
    fn rebuild(&self, dir: &dyn Directory, sources: &[Arc<VectorSegment>], policy: &OptimizerPolicy) -> Result<usize> {
        if sources.is_empty() { return Ok(0); }
        #[cfg(feature = "mutable-hnsw-poc")]
        let mutable = self.registered_mutable_field(sources[0].field).is_some();
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let mutable = false;
        let options = self.field_options(sources[0].field);
        if !mutable && options.graph {
            return Err(crate::error::RusticError::Unsupported(
                "legacy segmented vector maintenance has been removed; reindex into a mutable HNSW field".into()));
        }
        // A mutable graph still holding a dead point's node (repair pending)
        // reads its f32 from here: keep it until the node is gone.
        #[cfg(feature = "mutable-hnsw-poc")]
        let graph = self.registered_mutable_field(sources[0].field);
        #[cfg(feature = "mutable-hnsw-poc")]
        let graph_holds = |p: u64| graph.as_ref().is_some_and(|field| field.holds_point(p));
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let graph_holds = |_: u64| false;
        // Reserve ordinal/point workspace before checkpoint sets and the
        // retained row list grow. The encoding peak has its own shared guard;
        // overlapping shard optimizers cannot each spend the entire cap.
        let points=sources.iter().fold(0u64, |n, source| n.saturating_add(source.len() as u64));
        let _workspace=crate::admission::shared().reserve(points.saturating_mul(128).saturating_add(4096), "vector optimizer rows")?;
        let pinned = self.checkpoint_pinned_points(sources[0].field);
        let pinned = |p: u64| pinned.as_ref().is_some_and(|pinned| pinned.contains(&p));
        let mut rows: Vec<(u64, &[f32])> = Vec::new();
        {
            let mut t = self.table.read();
            let mut scanned = 0usize;
            for s in sources {
                for (ord, &p) in s.points().iter().enumerate() {
                    scanned += 1;
                    if scanned % TABLE_SCAN_BUMP == 0 { parking_lot::RwLockReadGuard::bump(&mut t); }
                    if t.is_kept(p) || graph_holds(p) || pinned(p) {
                        rows.push((p, s.vector(ord as u32)));
                    }
                }
            }
        }
        let ids: Vec<u32> = sources.iter().map(|s| s.id).collect();
        if rows.is_empty() {
            self.replace(&ids, None);
            return Ok(0);
        }
        rows.sort_by_key(|(p, _)| *p);
        let f = &sources[0];
        let id = self.next_segment.fetch_add(1, AtomicOrdering::Relaxed);
        let spec = SealSpec { kind: SealKind::Flat,
            quantize: if mutable { Quantization::None } else { options.stored_quantization(policy) },
            confidence_interval: if mutable { None } else { options.confidence_interval } };
        let seg = VectorSegment::write_with(dir, id, f.field, f.similarity, f.dim, &rows, spec)?;
        let n = rows.len();
        self.replace(&ids, Some(Arc::new(seg)));
        Ok(n)
    }

    // ---- DiskBBQ ------------------------------------------------------

    /// Every file the current DiskBBQ indexes reference.
    pub fn ivf_files(&self) -> Result<crate::admission::Allocation<std::collections::HashSet<String>>> {
        let state = self.state.read();
        let count = state.ivf.values().fold(0u64, |n, index| n.saturating_add(index.num_batches() as u64).saturating_add(1))
            .saturating_add(state.ivf_deferred.values().fold(0u64, |n, m| n.saturating_add(m.batches.len() as u64).saturating_add(1)));
        let memory = crate::admission::shared().reserve(count.saturating_mul(256).saturating_add(256), "IVF live file names")?;
        let files = state.ivf.values().flat_map(|s| s.manifest().files()).chain(state.ivf_deferred.values().flat_map(|m| m.files())).collect();
        Ok(crate::admission::Allocation::new(files, memory))
    }

    /// `field`'s current shared DiskBBQ index, if it has one.
    pub fn ivf_state(&self, field: u16) -> Option<Arc<super::ivf_field::IvfState>> {
        self.state.read().ivf.get(&field).cloned()
    }

    /// Install the DiskBBQ indexes of a commit point (after the vector
    /// segments are registered). A field whose files cannot be read (or
    /// are of a version this build does not know) has no index: its rows
    /// are searched flat until maintenance trains a new one, and the next
    /// commit retires the unreadable files. One that cannot be opened for
    /// want of memory does not fail the open: it is deferred (see
    /// [`Self::open_ivf_field`]).
    pub fn open_ivf(&self, dir: &dyn Directory, manifests: &[super::ivf_field::IvfFieldManifest]) -> Result<()> {
        let count = self.state.read().segments.len().saturating_add(manifests.len());
        let _workspace = crate::admission::shared().reserve((count as u64).saturating_mul(128).saturating_add(4096), "IVF restore registries")?;
        for m in manifests {
            crate::cancel::check()?;
            self.open_ivf_field(dir, m)?;
        }
        Ok(())
    }

    /// Open and install `m`, a field's committed index; whether it was
    /// installed. Unreadable or unknown-version files are retired (the next
    /// commit stops listing them). An admission refusal defers the index:
    /// the field is searched flat, commits keep listing its files, and the
    /// next maintenance step of the field retries. Only cancellation fails.
    fn open_ivf_field(&self, dir: &dyn Directory, m: &super::ivf_field::IvfFieldManifest) -> Result<bool> {
        let opened = (|| -> Result<bool> {
            #[cfg(test)]
            IVF_OPEN_ATTEMPTS.with(|n| n.set(n.get() + 1));
            #[cfg(test)]
            if FAIL_IVF_OPEN.with(std::cell::Cell::get) {
                return Err(crate::error::RusticError::Memory("test: IVF open refused".into()));
            }
            let _manifest_workspace = crate::admission::shared().reserve((m.batches.len() as u64).saturating_mul(128).saturating_add(4096), "IVF restored manifest producer")?;
            let Some((centroids, batches, indexed, next, _open_memory)) = super::ivf_field::open(dir, m)? else { return Ok(false) };
            let segments: Vec<Arc<VectorSegment>> = self.segments(m.field);
            let covered = segments.iter().filter(|s| s.points().iter().all(|&p| indexed.contains(p))).map(|s| s.id).collect();
            let state = super::ivf_field::IvfState::assemble(m.field, centroids, batches, indexed, covered, next)?;
            let mut st = self.state.write();
            st.insert_ivf(m.field, Arc::new(state))?;
            st.ivf_deferred.remove(&m.field);
            Ok(true)
        })();
        let reason = match opened {
            Ok(true) => return Ok(true),
            Ok(false) => "of a newer version".to_string(),
            // A cancelled open fails the open; it judges nothing.
            Err(error @ crate::error::RusticError::Cancelled(_)) => return Err(error),
            Err(crate::error::RusticError::Memory(reason)) => {
                tracing::warn!(field = m.field, "DiskBBQ index not opened for want of memory ({reason}): searching the field flat until maintenance opens it");
                self.state.write().ivf_deferred.insert(m.field, m.clone());
                return Ok(false);
            }
            Err(error) => format!("unreadable ({error})"),
        };
        tracing::warn!(field = m.field, "DiskBBQ index files {reason}: searching the field flat until it is rebuilt");
        let retired = retire_ivf_files(m.files())?;
        let mut st = self.state.write();
        st.ivf_deferred.remove(&m.field);
        st.obsolete_files.push_back(retired);
        // The next commit stops listing them (then deletes them).
        self.uncommitted.store(true, AtomicOrdering::Release);
        Ok(false)
    }

    /// Drop `field`'s DiskBBQ index (its mapping no longer is `bbq_disk`,
    /// or its codes were built with other parameters): its files go at the
    /// next commit, its rows are searched flat until maintenance trains a
    /// new one. Searches and views holding the index finish on it.
    pub fn reset_ivf_field(&self, field: u16) -> Result<()> {
        let mut st = self.state.write();
        st.ivf_open_retry.remove(&field);
        if let Some(m) = st.ivf_deferred.get(&field) {
            let files = retire_ivf_files(m.files())?;
            st.ivf_deferred.remove(&field);
            st.obsolete_files.push_back(files);
            self.uncommitted.store(true, AtomicOrdering::Release);
        }
        if let Some(old) = st.ivf.get(&field) {
            let _workspace = crate::admission::shared().reserve((old.num_batches() as u64).saturating_add(1).saturating_mul(128), "IVF retirement producer")?;
            let files = retire_ivf_files(old.manifest().files())?;
            st.ivf.remove(&field);
            st.obsolete_files.push_back(files);
            self.uncommitted.store(true, AtomicOrdering::Release);
        }
        Ok(())
    }

    /// Whether `field`'s DiskBBQ index no longer fits `options` (a field of
    /// `similarity` and `dim`): none wanted, or other code parameters.
    pub fn ivf_field_stale(&self, field: u16, similarity: Similarity, dim: u32, options: &VectorFieldOptions) -> bool {
        let Some(state) = self.ivf_state(field) else { return false };
        options.ivf.is_none_or(|ivf| super::ivf_field::BuildParams::of(similarity, dim as usize, &ivf) != state.params())
    }

    /// Test seam: rewrite payload segment `id` (a vacuum) under the
    /// optimizer lock, as a concurrent swap would.
    #[cfg(test)]
    pub(crate) fn rewrite_segment_for_test(&self, dir: &dyn Directory, policy: &OptimizerPolicy, id: u32) -> Result<usize> {
        self.merge_segments_for_test(dir, policy, &[id])
    }

    /// Test seam: combine payload segments `ids` (a merge) under the
    /// optimizer lock, with no index step after it.
    #[cfg(test)]
    pub(crate) fn merge_segments_for_test(&self, dir: &dyn Directory, policy: &OptimizerPolicy, ids: &[u32]) -> Result<usize> {
        let _guard = self.optimize_lock.lock();
        let sources = ids.iter().map(|&id| self.segment(id).ok_or_else(|| crate::error::RusticError::Segment(format!("no vector segment {id}")))).collect::<Result<Vec<_>>>()?;
        self.rebuild(dir, &sources, policy)
    }

    /// Insert, then maintenance steps until none is due (bounded); with
    /// `force` (force merge, expunge deletes) every batch is compacted too.
    fn ivf_steps(&self, dir: &dyn Directory, field: u16, force: bool) -> Result<usize> {
        const MAX_STEPS: usize = 256;
        let mode = if force { Rebuild::Force } else { Rebuild::Merge };
        let mut n = self.ivf_insert(dir, field, if force { Rebuild::Force } else { Rebuild::Seal })?;
        // Splits, drops and due compactions (an unsplittable cluster is
        // marked, not retried, so this ends).
        for _ in 0..MAX_STEPS {
            if !self.ivf_step(dir, field, IvfWork::Maintain { force: false }, mode)? {
                break;
            }
            n += 1;
        }
        // A force merge then compacts everything once, against one
        // liveness snapshot (never once per step).
        if force && self.ivf_step(dir, field, IvfWork::Maintain { force: true }, mode)? {
            n += 1;
        }
        Ok(n)
    }

    /// Insert steps until every row of `field` is indexed: each takes a
    /// bounded share of the rows (see [`super::ivf_field::max_step_rows`]).
    /// The steps that changed the index. Callers hold `optimize_lock`.
    fn ivf_insert(&self, dir: &dyn Directory, field: u16, mode: Rebuild) -> Result<usize> {
        const MAX_STEPS: usize = 1024;
        let mut n = 0;
        for _ in 0..MAX_STEPS {
            let mut more = false;
            if !self.ivf_step_bounded(dir, field, IvfWork::Insert, mode, &mut more)? {
                break;
            }
            n += 1;
            if !more {
                break;
            }
        }
        Ok(n)
    }

    /// One DiskBBQ step of `field` (see [`super::ivf_field`]), built on the
    /// merge pool (`Force`: the force-merge pool); merges and force merges
    /// write through the merge throttle and pace their passes on it.
    /// Whether the index changed. Callers hold `optimize_lock`.
    fn ivf_step(&self, dir: &dyn Directory, field: u16, work: IvfWork, mode: Rebuild) -> Result<bool> {
        self.ivf_step_bounded(dir, field, work, mode, &mut false)
    }

    /// [`Self::ivf_step`]; `more`: set when an insert step left rows for
    /// the next one.
    fn ivf_step_bounded(&self, dir: &dyn Directory, field: u16, work: IvfWork, mode: Rebuild, more: &mut bool) -> Result<bool> {
        use super::ivf_field::{self as f, Action, BuildParams, IvfState, Step, Writer};
        let options = self.field_options(field);
        let Some(ivf) = options.ivf else {
            // No longer `bbq_disk` (a mapping update, perhaps migrating in
            // the background): its index goes.
            self.reset_ivf_field(field)?;
            return Ok(false);
        };
        // A graph left from before the mapping update: the migration retires it.
        #[cfg(feature = "mutable-hnsw-poc")]
        if self.registered_mutable_field(field).is_some() { return Ok(false); }
        // A committed index whose open was deferred (no memory then): open
        // it first. Until it opens the field is searched flat and nothing
        // replaces it; an unreadable one is retired and a new one trained.
        let deferred = self.state.read().ivf_deferred.get(&field).cloned();
        if let Some(m) = deferred {
            let now = std::time::Instant::now();
            if self.state.read().ivf_open_retry.get(&field).is_some_and(|(at, _)| now < *at) {
                return Ok(false);
            }
            // A probe for the open's memory first: a refusal costs no read
            // of the index.
            let fits = match ivf_open_estimate(dir, &m).and_then(|bytes| crate::admission::shared().reserve(bytes, "IVF deferred open probe")) {
                Ok(probe) => { drop(probe); true }
                Err(crate::error::RusticError::Memory(_)) => false,
                Err(error) => return Err(error),
            };
            let opened = fits && self.open_ivf_field(dir, &m)?;
            let mut st = self.state.write();
            if opened || !st.ivf_deferred.contains_key(&field) {
                st.ivf_open_retry.remove(&field);
            } else {
                let tries = st.ivf_open_retry.get(&field).map_or(0, |r| r.1).saturating_add(1);
                let delay = std::time::Duration::from_secs(1u64 << tries.min(6)).min(std::time::Duration::from_secs(60));
                st.ivf_open_retry.insert(field, (now + delay, tries));
                return Ok(false);
            }
        }
        let _registry_memory = crate::admission::shared().reserve((self.state.read().segments.len() as u64).saturating_mul(128).saturating_add(4096), "IVF maintenance registry")?;
        let (segs, current) = {
            let st = self.state.read();
            (st.segments.values().filter(|s| s.field == field).cloned().collect::<Vec<_>>(), st.ivf.get(&field).cloned())
        };
        let Some(first) = segs.first() else { return Ok(false) };
        let row_count = segs.iter().fold(0u64, |n, s| n.saturating_add(s.len() as u64));
        // An insert step lists the unindexed rows (at most all of them) as
        // `(point, row)`, 24 bytes each; maintenance steps list one
        // cluster's members, admitted with their workspace.
        let row_bytes = if work == IvfWork::Insert { row_count.saturating_mul(32) } else { 0 };
        let _row_memory = crate::admission::shared().reserve(row_bytes.saturating_add(4096), "IVF maintenance row references")?;
        let params = BuildParams::of(first.similarity, first.dim as usize, &ivf);
        // New code parameters (a mapping update, migrating in the
        // background): a replacement is trained from every row while the
        // stale index keeps serving, and takes its place at install.
        let (current, stale) = match current {
            Some(state) if state.params() != params => (None, Some(state)),
            other => (other, None),
        };
        let by_id: std::collections::HashMap<u32, &Arc<VectorSegment>> = segs.iter().map(|s| (s.id, s)).collect();
        let throttled = mode != Rebuild::Seal;
        let pool = if mode == Rebuild::Force { crate::force_merge_pool() } else { crate::merge_pool() };
        let code_bytes = params.dim * usize::from(params.bits) / 8 + 16;
        // Inputs: rows borrowed from the segments, liveness copied, so no
        // store lock is held while building.
        let built: Option<(Step, Option<Arc<IvfState>>)> = match work {
            IvfWork::Insert => {
                // Every row indexed (each segment covered): nothing to list.
                if stale.is_none() && current.as_ref().is_some_and(|c| segs.iter().all(|s| c.covered.contains(&s.id))) {
                    return Ok(false);
                }
                let mut rows: Vec<(u64, &[f32])> = segs.iter().flat_map(|s| {
                    let indexed = current.as_ref().map(|c| c.indexed.clone());
                    s.points().iter().enumerate().filter(move |(_, p)| indexed.as_ref().is_none_or(|i| !i.contains(**p)))
                        .map(move |(o, &p)| (p, s.vector(o as u32)))
                }).collect();
                rows.sort_by_key(|r| r.0);
                rows.dedup_by_key(|r| r.0);
                if rows.is_empty() {
                    // Every row is indexed, so every segment is covered: one
                    // whose coverage a swap could not carry over is marked now.
                    let Some(base) = current.as_ref().filter(|c| segs.iter().any(|s| !c.covered.contains(&s.id))) else { return Ok(false) };
                    let covered = base.with_covered(segs.iter().map(|s| s.id).collect())?;
                    let mut st = self.state.write();
                    if !st.ivf.get(&field).is_some_and(|now| Arc::ptr_eq(now, base)) {
                        return Ok(false);
                    }
                    st.ivf.insert(field, Arc::new(covered));
                    return Ok(true);
                }
                // Only live rows are trained on and inserted; dead ones (a
                // reopen finds every point without entries unindexed) are
                // marked indexed with no entries, so their segments stay
                // covered and nothing scans them.
                let dead: Vec<u64> = {
                    let t = self.table.read();
                    let dead = rows.iter().filter(|r| !t.is_kept(r.0)).map(|r| r.0).collect();
                    rows.retain(|r| t.is_kept(r.0));
                    dead
                };
                if current.is_none() {
                    // Below the flat threshold the field stays flat (ES
                    // `flat_index_threshold`); a stale index then just goes.
                    let flat = ivf.flat_threshold();
                    if rows.is_empty() || (flat > 0 && rows.len() <= flat) {
                        if stale.is_some() {
                            self.reset_ivf_field(field)?;
                            return Ok(true);
                        }
                        return Ok(false);
                    }
                }
                // A bounded step (its workspace within a quarter of the
                // admission limit): rows spread over the field are trained
                // on or inserted now, the rest by the next insert steps.
                let max_rows = f::max_step_rows(&params, crate::admission::shared().limit());
                // An index about to double is retrained on every live row
                // (within one step's bound) instead of stretching centroids
                // fitted to a fraction of the data.
                // The candidate rows are counted (the live rows plus the
                // unindexed ones, an upper bound on the kept rows) before
                // any is listed; a field too large for one step skips the
                // retrain, and that is recorded (`skipped_retrain`) so the
                // next steps do not attempt it again before the index
                // doubles once more.
                let mut skipped_retrain = false;
                let mut _retrain_memory = None;
                let retrain: Option<Vec<(u64, &[f32])>> = match &current {
                    Some(state) if stale.is_none() && state.wants_retrain(rows.len()) => {
                        let bound = {
                            let t = self.table.read();
                            segs.iter().map(|s| t.live_by_segment.get(&s.id).copied().unwrap_or(0)).sum::<usize>()
                        }.saturating_add(rows.len()).min(row_count as usize);
                        #[cfg(test)]
                        IVF_RETRAIN_ATTEMPTS.with(|n| n.set(n.get() + 1));
                        #[cfg(test)]
                        let max_rows = if SKIP_IVF_RETRAIN.with(std::cell::Cell::get) { 0 } else { max_rows };
                        let listed = (bound <= max_rows).then(|| {
                            crate::admission::shared().reserve((bound as u64).saturating_mul(std::mem::size_of::<(u64, &[f32])>() as u64).saturating_add(4096), "IVF retrain rows")
                        }).transpose();
                        match listed {
                            Ok(Some(memory)) => {
                                _retrain_memory = Some(memory);
                                let t = self.table.read();
                                let mut all: Vec<(u64, &[f32])> = Vec::with_capacity(bound);
                                // More kept rows than counted (points hidden
                                // under an open view): no retrain this time,
                                // never a partial one.
                                let mut fits = true;
                                'rows: for s in &segs {
                                    for (o, &p) in s.points().iter().enumerate() {
                                        if t.is_kept(p) {
                                            if all.len() == bound {
                                                fits = false;
                                                break 'rows;
                                            }
                                            all.push((p, s.vector(o as u32)));
                                        }
                                    }
                                }
                                drop(t);
                                if fits {
                                    all.sort_by_key(|r| r.0);
                                    all.dedup_by_key(|r| r.0);
                                    Some(all)
                                } else {
                                    skipped_retrain = true;
                                    None
                                }
                            }
                            Ok(None) | Err(crate::error::RusticError::Memory(_)) => {
                                skipped_retrain = true;
                                None
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    _ => None,
                };
                if retrain.is_none() && rows.len() > max_rows {
                    let stride = rows.len().div_ceil(max_rows);
                    let mut i = 0usize;
                    rows.retain(|_| { i += 1; (i - 1) % stride == 0 });
                    *more = true;
                }
                let field_id = field;
                let current_ref = current.clone();
                let stale_ref = stale.clone();
                let fresh_file = {
                    let locked = self.state.read();
                    dir.list_admitted(false)?.iter().chain(locked.obsolete_files.iter().flat_map(|group| group.iter())).filter_map(|n| super::ivf_field::parse_file_name(n))
                        .filter(|(f, _)| *f == field).map(|(_, id)| id).max().unwrap_or(0)
                        .checked_add(1).ok_or_else(|| crate::error::RusticError::Corrupted("IVF file sequence exhausted".into()))?
                };
                crate::on_pool(pool, || -> Result<Option<(Step, Option<Arc<IvfState>>)>> {
                    let mut pace = throttled.then(crate::merge::MergeWork::begin);
                    let mut work = |v: usize| if let Some(p) = pace.as_mut() { p.work(v * code_bytes) };
                    let mut w = Writer { dir, field: field_id, throttled, work: &mut work, written: Vec::new() };
                    let out = match (&current_ref, &retrain) {
                        (Some(state), Some(all)) => {
                            let next = fresh_file.max(state.next_file());
                            f::train(&mut w, params, &ivf, all, next).and_then(|(c, b, next)| {
                                let indexed = Arc::new(state.indexed.with(rows.iter().map(|r| r.0).chain(dead.iter().copied()))?);
                                Ok(Step { state: IvfState::assemble(field_id, c, vec![b], indexed, Default::default(), next)?.trained_on(all.len()), obsolete: state.manifest().files() })
                            })
                        }
                        (None, _) => {
                            // Past every file of the field on disk or awaiting
                            // deletion (a reset index's files stay until the
                            // commit that no longer references them).
                            let next = fresh_file;
                            f::train(&mut w, params, &ivf, &rows, next).and_then(|(c, b, next)| {
                                let indexed = f::indexed_of(rows.iter().map(|r| r.0).chain(dead.iter().copied()))?;
                                // A replacement retires the stale index's files.
                                let obsolete = stale_ref.as_ref().map(|old| old.manifest().files()).unwrap_or_default();
                                Ok(Step { state: IvfState::assemble(field_id, c, vec![b], indexed, Default::default(), next)?, obsolete })
                            })
                        }
                        // A skipped retrain counts as one at this size.
                        (Some(state), None) => f::insert(&mut w, state, &rows, &dead).map(|mut step| {
                            if skipped_retrain {
                                let indexed = step.state.indexed.count() as usize;
                                step.state = step.state.trained_on(indexed);
                            }
                            step
                        }),
                    };
                    match out {
                        // The base is the index this step replaces: the
                        // install checks that it is still the current one.
                        Ok(step) => Ok(Some((step, current_ref.clone().or_else(|| stale_ref.clone())))),
                        Err(error) => { w.discard(); Err(error) }
                    }
                })?
            }
            IvfWork::Maintain { force } => {
                let Some(state) = current.clone() else { return Ok(false) };
                // Any liveness change (a point released from a closed view
                // included) is a new version of the table.
                let kept_version = self.table.read().kept_version;
                let quiet_key = (Arc::as_ptr(&state) as usize, state.next_file(), kept_version);
                if !force && self.ivf_quiet.lock().get(&field) == Some(&quiet_key) {
                    return Ok(false);
                }
                let kept_words = self.table.read().kept_words()?;
                let kept = |p: u64| {
                    let word = (p >> 6) as usize;
                    kept_words.get(&(word / (OWNER_CHUNK / 64)))
                        .is_some_and(|page| page[word % (OWNER_CHUNK / 64)] >> (p & 63) & 1 == 1)
                };
                let (live, garbage) = state.census(&kept)?;
                let _action_memory = crate::admission::shared().reserve((state.num_batches() as u64).saturating_mul(32).saturating_add(256), "IVF maintenance action")?;
                let Some(action) = f::next_action(&state, &ivf, &live, garbage, force) else {
                    if !force {
                        self.ivf_quiet.lock().insert(field, quiet_key);
                    }
                    return Ok(false);
                };
                tracing::debug!(target: "ivf", field, ?action, centroids = state.num_centroids(), batches = state.num_batches(), garbage, "IVF maintenance action");
                let field_id = field;
                let state_ref = state.clone();
                let split = matches!(action, Action::Split(_));
                match action {
                    Action::Split(cs) | Action::Drop(cs) => {
                        // Rows of the clusters' live entries.
                        let mut groups: Vec<(usize, f::Members<'_>)> = Vec::with_capacity(cs.len());
                        {
                            let t = self.table.read();
                            for &c in &cs {
                                let mut m: f::Members<'_> = Vec::new();
                                for e in state.list_entries(c) {
                                    let p = e & !f::SPILL;
                                    if !t.is_kept(p) { continue; }
                                    let Some((sid, ord)) = t.loc(p) else { continue };
                                    if let Some(s) = by_id.get(&sid) { m.push((e, s.vector(ord))); }
                                }
                                groups.push((c, m));
                            }
                        }
                        crate::on_pool(pool, || -> Result<Option<(Step, Option<Arc<IvfState>>)>> {
                            let mut pace = throttled.then(crate::merge::MergeWork::begin);
                            let mut work = |v: usize| if let Some(p) = pace.as_mut() { p.work(v * code_bytes) };
                            let mut w = Writer { dir, field: field_id, throttled, work: &mut work, written: Vec::new() };
                            let out = if split { f::split_many(&mut w, &state_ref, &groups, &ivf).map(Some) } else { f::drop_many(&mut w, &state_ref, &groups) };
                            match out {
                                Ok(Some(step)) => Ok(Some((step, Some(state_ref.clone())))),
                                // Nothing to drop into: remembered for this
                                // generation, so the next steps fall through
                                // to compactions instead of retrying it.
                                Ok(None) => Ok(Some((Step { state: state_ref.marked_unsplittable(cs[0])?, obsolete: Vec::new() }, Some(state_ref.clone())))),
                                Err(error) => { w.discard(); Err(error) }
                            }
                        })?
                    }
                    Action::Compact(which) => {
                        crate::on_pool(pool, || -> Result<Option<(Step, Option<Arc<IvfState>>)>> {
                            let mut pace = throttled.then(crate::merge::MergeWork::begin);
                            let mut work = |v: usize| if let Some(p) = pace.as_mut() { p.work(v * code_bytes) };
                            let mut w = Writer { dir, field: field_id, throttled, work: &mut work, written: Vec::new() };
                            match f::compact(&mut w, &state_ref, &which, &kept) {
                                Ok(step) => Ok(Some((step, Some(state_ref.clone())))),
                                Err(error) => { w.discard(); Err(error) }
                            }
                        })?
                    }
                }
            }
        };
        let Some((step, base)) = built else { return Ok(false) };
        #[cfg(test)]
        AFTER_IVF_BUILD.with(|hook| if let Some(hook) = hook.borrow_mut().take() { hook() });
        let _publication_memory = crate::admission::shared().reserve((step.state.unsplittable().len() as u64).saturating_add(base.as_ref().map_or(0, |b| b.unsplittable().len() as u64)).saturating_mul(128)
            .saturating_add((step.state.num_batches() as u64).saturating_mul(256)).saturating_add(4096), "IVF publication producers")?;
        // Coverage of the segments known now (others: searched flat until
        // the next step or swap).
        let segments = self.segments(field);
        let covered: std::collections::HashSet<u32> = segments.iter().filter(|s| s.points().iter().all(|&p| step.state.indexed.contains(p))).map(|s| s.id).collect();
        let marks = step.state.unsplittable().clone();
        let mut marks: std::collections::HashSet<(u32, u32)> = marks
            .into_iter()
            // A replacement for other parameters inherits no marks.
            .chain(base.iter().filter(|b| b.params() == step.state.params()).flat_map(|b| b.unsplittable().iter().copied()))
            .collect();
        let state0 = step.state.with_covered(covered)?;
        marks.retain(|&(c, generation)| state0.generation(c as usize) == Some(generation));
        let state = Arc::new(state0.with_unsplittable(marks)?);
        let mut st = self.state.write();
        let unchanged = match (&base, st.ivf.get(&field)) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        // The mapping as of now (an update may have landed during the build,
        // a background migration included): a build for other code
        // parameters, or for a field no longer `bbq_disk`, is void.
        let now = st.field_options.get(&field).copied().unwrap_or_default();
        let current_params = now.ivf.map(|o| BuildParams::of(first.similarity, first.dim as usize, &o));
        if !unchanged || current_params != Some(state.params()) {
            // Reset meanwhile (a mapping update): this build is void. Its
            // file numbers are retired, not free: the index still current
            // (the mapping may change back) numbers its next files past
            // them, else a later step would rewrite a name queued for
            // deletion and the commit would delete a live file. (Without
            // an index, a training step numbers past every file on disk
            // or queued.)
            if let Some(now) = st.ivf.get(&field) {
                now.reserve_files_through(state.next_file());
            }
            st.obsolete_files.push_back(retire_ivf_files(state.manifest().files().into_iter().filter(|name| !base.as_ref().is_some_and(|b| b.manifest().files().contains(name))).collect())?);
            self.uncommitted.store(true, AtomicOrdering::Release);
            return Ok(false);
        }
        let obsolete = retire_ivf_files(step.obsolete)?;
        st.insert_ivf(field, state)?;
        st.obsolete_files.push_back(obsolete);
        self.uncommitted.store(true, AtomicOrdering::Release);
        Ok(true)
    }

    /// Swap `old` segments for `new` (if any): points of `new` relocate,
    /// points only in `old` lose their location.
    fn replace(&self, old: &[u32], new: Option<Arc<VectorSegment>>) {
        // Dropped points a mutable graph still holds (dead documents whose
        // node repair is pending) take their f32 back into memory first:
        // their durable row goes with this replacement.
        // Before the swap, so the row is readable throughout: restoring it
        // after left a window in which a query (or a view's exact scan) met
        // a code-only row with no durable f32 ("missing live source row").
        #[cfg(feature = "mutable-hnsw-poc")]
        for (field, point, row) in self.rows_leaving(old, new.as_deref()) {
            if let Some(field) = self.registered_mutable_field(field) {
                if let Err(error) = field.restore_raw(point, &row) {
                    tracing::warn!("restoring the f32 of mutable point {point} failed: {error}");
                }
            }
        }
        // New payload IDs need no IVF rewrite (entries are point ids), but
        // the replacement must stay covered: an uncovered segment is scanned
        // row by row for points the index lacks on every query.
        let coverage = new.as_deref().and_then(|seg| self.coverage_after_swap(old, seg));
        self.replace_locked(old, new, coverage);
        #[cfg(test)]
        AFTER_SWAP.with(|hook| if let Some(hook) = hook.borrow_mut().take() { hook() });
    }

    /// `(field, point, f32)` of the points `replace(old, new)` drops that a
    /// mutable graph still holds.
    #[cfg(feature = "mutable-hnsw-poc")]
    fn rows_leaving(&self, old: &[u32], new: Option<&VectorSegment>) -> Vec<(u16, u64, Vec<f32>)> {
        // The segments are taken under `state`, which is released before the
        // O(points) scan and before any field lookup: `mutable_field` takes
        // the field registry and then `state`, and the reverse order here
        // (state held, registry wanted) deadlocked with a queued state writer.
        let segments: Vec<Arc<VectorSegment>> = {
            let state = self.state.read();
            old.iter().filter_map(|id| state.segments.get(id).cloned()).collect()
        };
        let mut out = Vec::new();
        for segment in segments {
            let Some(field) = self.registered_mutable_field(segment.field) else { continue };
            for (ord, &p) in segment.points().iter().enumerate() {
                let moving = new.is_some_and(|n| n.points().binary_search(&p).is_ok());
                if !moving && field.holds_point(p) {
                    out.push((segment.field, p, segment.vector(ord as u32).to_vec()));
                }
            }
        }
        out
    }

    /// `new`'s field index with `new` covered in place of `old`, when every
    /// `old` segment was covered (a vacuum or merge: `new` holds only their
    /// points, all indexed). Built before the swap takes the write lock;
    /// `(base, replacement)`, installed only if `base` is still current.
    /// `None`: nothing to carry over, or no memory for the copy (`new` is
    /// then scanned for unindexed rows until the next insert step covers it).
    fn coverage_after_swap(&self, old: &[u32], new: &VectorSegment) -> Option<(Arc<super::ivf_field::IvfState>, Arc<super::ivf_field::IvfState>)> {
        let base = self.state.read().ivf.get(&new.field).cloned()?;
        if old.is_empty() || !old.iter().all(|id| base.covered.contains(id)) {
            return None;
        }
        let copy = || -> Result<super::ivf_field::IvfState> {
            let _covered = crate::admission::shared().reserve((base.covered.capacity() as u64).saturating_mul(16).saturating_add(256), "IVF covered copy producer")?;
            let mut covered = base.covered.clone();
            for id in old {
                covered.remove(id);
            }
            covered.insert(new.id);
            base.with_covered(covered)
        };
        match copy() {
            Ok(state) => Some((base, Arc::new(state))),
            Err(error) => {
                tracing::warn!(field = new.field, "DiskBBQ coverage of swapped segment {} not carried over ({error}): its rows are checked against the index until the next insert step", new.id);
                None
            }
        }
    }

    fn replace_locked(&self, old: &[u32], new: Option<Arc<VectorSegment>>, coverage: Option<(Arc<super::ivf_field::IvfState>, Arc<super::ivf_field::IvfState>)>) {
        let mut st = self.state.write();
        self.uncommitted.store(true, AtomicOrdering::Release);
        let mut t = self.table.write();
        // `new` holds its points sorted by id (see `rebuild`).
        let moving = |p: u64| new.as_ref().is_some_and(|n| n.points().binary_search(&p).is_ok());
        for id in old {
            if let Some(s) = st.segments.remove(id) {
                for &p in s.points() {
                    if !moving(p) && t.loc(p).is_some_and(|(sid, _)| sid == *id) {
                        // Dropped (dead) point: no longer stored anywhere.
                        t.set_loc(p, NONE);
                        t.set_alive(p, false);
                        t.set_hidden(p, false);
                        t.retire_empty_page(p as usize / OWNER_CHUNK);
                    }
                }
                st.obsolete.push(*id);
            }
        }
        if let Some(seg) = new {
            // Relocate; liveness is untouched (a point deleted during the
            // build stays dead).
            for (ord, &p) in seg.points().iter().enumerate() {
                t.set_loc(p, pack(seg.id, ord as u32));
            }
            // In the same critical section as the segment swap: no search
            // sees `new` without its coverage. A step installed meanwhile
            // computed coverage itself.
            if let Some((base, covered)) = coverage {
                if st.ivf.get(&seg.field).is_some_and(|now| Arc::ptr_eq(now, &base)) {
                    st.ivf.insert(seg.field, covered);
                }
            }
            st.segments.insert(seg.id, seg);
        }
    }
}

/// One field's DiskBBQ steps within a pass over every field: one field's
/// index must not keep the others from being sealed and maintained.
/// Cancellation ends the pass. An admission refusal leaves the field
/// searched as it is (its unindexed rows flat) until a later pass; another
/// error is kept in `deferred` (the first one), returned once the pass has
/// visited every field.
fn ivf_isolated<T: Default>(field: u16, result: Result<T>, deferred: &mut Option<crate::error::RusticError>) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error @ crate::error::RusticError::Cancelled(_)) => Err(error),
        Err(crate::error::RusticError::Memory(reason)) => {
            tracing::warn!(field, "DiskBBQ maintenance deferred ({reason}): the field's unindexed rows are searched flat meanwhile");
            Ok(T::default())
        }
        Err(error) => {
            tracing::warn!(field, "DiskBBQ maintenance failed ({error}); continuing with the other fields");
            deferred.get_or_insert(error);
            Ok(T::default())
        }
    }
}

/// Bytes opening a committed index takes, about: the centroid decode
/// (four times the file, then the table), the run tables and the indexed
/// set (from the posting files' sizes). The files are mapped, not read.
fn ivf_open_estimate(dir: &dyn Directory, m: &super::ivf_field::IvfFieldManifest) -> Result<u64> {
    let centroids = dir.open_read(&super::ivf_field::centroid_file_name(m.field, m.centroids))?.len() as u64;
    let mut postings = 0u64;
    for &b in &m.batches {
        postings = postings.saturating_add(dir.open_read(&super::ivf_field::batch_file_name(m.field, b))?.len() as u64);
    }
    Ok(centroids.saturating_mul(5).saturating_add(postings / 16).saturating_add((m.batches.len() as u64).saturating_mul(128)).saturating_add(4096))
}

/// How a [`VectorStore`] DiskBBQ step is scheduled (and throttled).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rebuild {
    /// Indexing fresh rows: ES's flush (merge pool, not throttled).
    Seal,
    /// A background maintenance step (merge pool, throttled).
    Merge,
    /// Force merge / expunge deletes (force-merge pool, IO limit).
    Force,
}

/// A DiskBBQ step's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvfWork {
    /// Train, or index the rows not yet indexed.
    Insert,
    /// Split, drop or compact (`force`: compact everything).
    Maintain { force: bool },
}

/// A DiskBBQ field's search (ES `IVFKnnFloatVectorQuery`, then
/// `RescoreKnnVectorQuery`): the shared index's best
/// `round(2 k max(1, oversample))` estimates, the `ceil(k * oversample)`
/// best rescored from the mapped `f32` column (`oversample` 0: the top `k`
/// estimates are the scores), plus the rows not indexed yet (fresh chunks)
/// scored flat. A filter selective enough that ES would skip most
/// centroids (at most 1.25 accepted vectors a centroid) scores its matches
/// exactly instead. `(point, score)`, unsorted.
#[allow(clippy::too_many_arguments)]
fn search_ivf_field(
    state: &super::ivf_field::IvfState,
    segs: &[Arc<VectorSegment>],
    t: &PointTable,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<PointFilter<'_>>,
    total: usize,
    cfg: &PlannerConfig,
    opts: &VectorFieldOptions,
    ivf: &super::ivf::IvfOptions,
) -> Result<(Vec<(PointId, f32)>, SearchStats)> {
    let budget = super::ivf::leaf_budget(k, cfg.oversample.unwrap_or(ivf.budget_oversample));
    let bytes = (budget as u64).saturating_mul(128).saturating_add((segs.len() as u64).saturating_mul(128))
        .saturating_add((k as u64).saturating_mul(segs.len() as u64).saturating_mul(32)).saturating_add(4096);
    let _workspace = crate::admission::shared().reserve(bytes, "IVF result merge and rescoring")?;
    let ops0 = super::hnsw::vector_ops();
    let mut st = SearchStats::default();
    let oversample = cfg.oversample.unwrap_or(opts.oversample);
    let estimate = (oversample <= 0.0).then_some(super::es_quant::EsQuant { bits: ivf.doc_bits(), confidence_interval: None, osq: true });
    let matching = match filter {
        None => usize::MAX,
        Some(PointFilter::Points(b)) => b.len(),
        Some(PointFilter::Docs(d)) => d.len(),
        Some(PointFilter::Ords(o)) => o.len().max(o.docs.len()),
    };
    let mut out: Vec<(PointId, f32)> = Vec::new();
    let selective = filter.is_some() && (matching as f64) <= 1.25 * state.centroids.active() as f64;
    // Rows the index does not hold (all of them for a selective filter).
    for s in segs {
        if !selective && state.covered.contains(&s.id) {
            continue;
        }
        let skip = (!selective).then_some(&*state.indexed);
        let hits = search_unindexed(s, t, query, k, filter, total, skip, estimate)?;
        if !hits.is_empty() || !selective {
            st.exact_segments += 1;
        }
        let pts = s.points();
        out.extend(hits.into_iter().map(|(o, score)| (pts[o as usize], score)));
    }
    if !selective {
        // Dense per-query accept bits over point ids (alive, and an ordinal
        // or point filter resolved to points), so each posting entry the
        // scan meets costs one bit test, as ES tests its accept-docs bits;
        // an owner-doc filter is tested per alive entry.
        let accept_bits = ivf_accept_words(segs, t, filter)?;
        let docs_only = match filter {
            Some(PointFilter::Docs(d)) => Some(d),
            _ => None,
        };
        let accept = |p: PointId| -> bool {
            let i = p as usize;
            accept_bits.get(i >> 6).is_some_and(|w| w >> (i & 63) & 1 == 1)
                && docs_only.is_none_or(|d| t.owner(p).is_some_and(|(sg, doc)| d.contains(sg, doc)))
        };
        let ratio = cfg.visit_ratio.unwrap_or_else(|| ivf.default_visit_ratio());
        // ES (`IvfQueryConfigResolver`): the collector is sized for the
        // query's oversample, else the mapping's, else 3, rescored or not.
        let budget = super::ivf::leaf_budget(k, cfg.oversample.unwrap_or(ivf.budget_oversample));
        // ES `DenseVectorFieldMapper.createKnnQuery`: a rescoring query asks
        // for at least `min(ceil(k * oversample), 10,000)` candidates; that
        // count reaches `computeDynamicVisitRatio`.
        let num_candidates = if oversample > 0.0 { ef.max(((k as f64 * f64::from(oversample)).ceil() as usize).min(10_000)) } else { ef };
        // The live vectors of the field, the `n` of the visit budget.
        let population: usize = segs.iter().map(|s| t.live_by_segment.get(&s.id).copied().unwrap_or(0)).sum();
        let started = std::time::Instant::now();
        let (cands, ivf_stats) = state.search(query, k, num_candidates, ratio, budget, Some(population), filter.map(|_| matching), oversample <= 0.0, accept)?;
        tracing::trace!(target: "ivf_query", ?ivf_stats, matching, population, search_us = started.elapsed().as_micros() as u64, "IVF search");
        st.graph_segments += 1;
        if oversample <= 0.0 {
            out.extend(cands.iter().copied().take(k));
        } else {
            // ES rescores its shard's best `ceil(k * oversample)` estimates
            // (`OVERSAMPLE_LIMIT`: at most 10,000). With the oversample
            // defaulted (neither the request nor the mapping sets it), every
            // collected candidate (twice that, `leaf_budget`) is rescored:
            // exact scores of a few dozen more rows cost microseconds and
            // lift recall past the 1-bit estimates' ranking errors. An explicit oversample keeps ES's
            // count.
            let es_keep = ((k as f64 * f64::from(oversample.max(1.0))).ceil().min(10_000.0) as usize).max(k);
            let keep = if cfg.oversample.is_none() && !ivf.explicit_oversample { es_keep.max(cands.len().min(10_000)) } else { es_keep };
            let by_id: std::collections::HashMap<u32, &Arc<VectorSegment>> = segs.iter().map(|s| (s.id, s)).collect();
            let mut rescored = 0u64;
            for (p, _) in cands.iter().copied().take(keep) {
                let Some((sid, ord)) = t.loc(p) else { continue };
                let Some(s) = by_id.get(&sid) else { continue };
                out.push((p, s.similarity.score(query, s.vector(ord))));
                rescored += 1;
            }
            super::hnsw::add_vector_ops(rescored);
        }
    }
    st.vector_ops = super::hnsw::vector_ops().wrapping_sub(ops0);
    Ok((out, st))
}

/// Set in `out` (dense words over point ids, grown as needed) the point
/// of every doc set in `words`: a word at a time along `runs`, else doc
/// by doc through `col`.
fn map_docs_to_points(out: &mut Vec<u64>, words: &[u64], col: &EncodedPointColumn<'_>, runs: Option<&[crate::segment::PointRun]>) {
    match runs {
        Some(runs) => {
            for r in runs {
                let (p0, d0, len) = (r.point as i64, i64::from(r.doc), i64::from(r.len));
                let first = (p0 / 64) as usize;
                let end = ((p0 + len) as usize).div_ceil(64);
                if out.len() < end {
                    out.resize(end, 0);
                }
                for w in first..end {
                    let lo = (w * 64) as i64;
                    // Doc of point `lo`, then the run's points only.
                    let mut x = shifted_word(words, lo - p0 + d0);
                    if lo < p0 {
                        x &= !low_mask(p0 - lo);
                    }
                    if lo + 64 > p0 + len {
                        x &= low_mask(p0 + len - lo);
                    }
                    out[w] |= x;
                }
            }
        }
        None => {
            for (wi, &w0) in words.iter().enumerate() {
                let mut w = w0;
                while w != 0 {
                    let d = (wi * 64) as u32 + w.trailing_zeros();
                    w &= w - 1;
                    if let Some(p) = col.get(d) {
                        let i = p as usize;
                        if out.len() <= i / 64 {
                            out.resize(i / 64 + 1, 0);
                        }
                        out[i / 64] |= 1 << (i % 64);
                    }
                }
            }
        }
    }
}

/// The alive points of the field (`PointTable::alive`), as one dense
/// bitset over point ids, restricted to `filter` when it names points or
/// per-segment ordinals (its matches only are walked); an owner-doc filter
/// is left to the caller.
fn ivf_accept_words(segs: &[Arc<VectorSegment>], t: &PointTable, filter: Option<PointFilter<'_>>) -> Result<crate::admission::Allocation<Vec<u64>>> {
    let page = t.alive.page_len;
    let n = t.alive.pages.last_key_value().map_or(0, |(&k, _)| (k + 1) * page);
    let ords = matches!(filter, Some(PointFilter::Ords(_)));
    let bytes = (n as u64).saturating_mul(if ords { 16 } else { 8 }).saturating_add(256);
    let memory = crate::admission::shared().reserve(bytes, "IVF query accept bits")?;
    let mut words = vec![0u64; n];
    for (&k, p) in &t.alive.pages {
        words[k * page..(k + 1) * page].copy_from_slice(p);
    }
    match filter {
        None | Some(PointFilter::Docs(_)) => {}
        Some(PointFilter::Points(b)) => {
            for (i, w) in words.iter_mut().enumerate() {
                *w &= b.words.get(i).copied().unwrap_or(0);
            }
        }
        Some(PointFilter::Ords(o)) => {
            let mut matched = vec![0u64; n];
            let mut set = |p: PointId| {
                let i = p as usize;
                if let Some(w) = matched.get_mut(i >> 6) {
                    *w |= 1 << (i & 63);
                }
            };
            for s in segs {
                let pts = s.points();
                match o.get(s.id) {
                    Some((w, _)) => {
                        for ord in SetBits::new(w) {
                            if let Some(&p) = pts.get(ord as usize) {
                                set(p);
                            }
                        }
                    }
                    // Swapped in after the filter was built: its docs.
                    None => {
                        for &p in pts {
                            if t.owner(p).is_some_and(|(sg, doc)| o.docs.contains(sg, doc)) {
                                set(p);
                            }
                        }
                    }
                }
            }
            for (w, m) in words.iter_mut().zip(&matched) {
                *w &= m;
            }
        }
    }
    Ok(crate::admission::Allocation::new(words, memory))
}

/// Flat top-`k` over segment `s`'s accepted ordinals whose point `skip`
/// does not hold: exact `f32`, or with `estimate` the ES estimates of a
/// flat segment of these rows (fitted per query: these are fresh rows
/// awaiting indexing, no codes are cached).
#[allow(clippy::too_many_arguments)]
fn search_unindexed(
    s: &VectorSegment,
    t: &PointTable,
    query: &[f32],
    k: usize,
    filter: Option<PointFilter<'_>>,
    total: usize,
    skip: Option<&super::ivf_field::Indexed>,
    estimate: Option<super::es_quant::EsQuant>,
) -> Result<Vec<(u32, f32)>> {
    let per_row = if estimate.is_some() && s.similarity == Similarity::Cosine { (s.dim as u64).saturating_mul(4).saturating_add(64) } else { 32 };
    let bytes = (s.len() as u64).saturating_mul(per_row).saturating_add((s.dim as u64).saturating_mul(64)).saturating_add(4096);
    let _workspace = crate::admission::shared().reserve(bytes, "IVF flat query workspace")?;
    let n = s.len() as usize;
    let pts = s.points();
    let (lo, hi) = match (pts.first(), pts.last()) {
        (Some(&a), Some(&b)) => (a, b),
        _ => return Ok(Vec::new()),
    };
    let ords: Option<(&[u64], usize)> = match filter {
        Some(PointFilter::Ords(o)) => o.get(s.id),
        _ => None,
    };
    let doc_filter: Option<&DocFilter> = match filter {
        Some(PointFilter::Docs(d)) => Some(d),
        Some(PointFilter::Ords(o)) if ords.is_none() => Some(&o.docs),
        _ => None,
    };
    let fresh = |o: u32| skip.is_none_or(|i| !i.contains(pts[o as usize]));
    let accept = |o: u32| -> bool {
        if !fresh(o) {
            return false;
        }
        if let Some((w, _)) = ords {
            return word_bit(w, o);
        }
        let p = pts[o as usize];
        t.is_alive(p)
            && match (filter, doc_filter) {
                (None, _) => true,
                (Some(PointFilter::Points(b)), _) => b.contains(p),
                (_, Some(d)) => t.owner(p).is_some_and(|(sg, doc)| d.contains(sg, doc)),
                _ => true,
            }
    };
    let chosen: Vec<u32> = match (filter, ords) {
        (Some(PointFilter::Ords(_)), Some((w, _))) => SetBits::new(w).filter(|&o| o < n as u32 && fresh(o)).collect(),
        (Some(PointFilter::Points(b)), _) if b.count_range(lo, hi) * 8 < n => {
            let mut v = Vec::new();
            b.for_each_in_range(lo, hi, |p| {
                if let Some((sid, o)) = t.loc(p) {
                    if sid == s.id && t.is_alive(p) && fresh(o) {
                        v.push(o);
                    }
                }
            });
            v.sort_unstable();
            v
        }
        _ => (0..n as u32).filter(|&o| accept(o)).collect(),
    };
    if chosen.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    Ok(match estimate {
        None => s.search_exact(query, k, chosen.into_iter()),
        Some(spec) => {
            let q = spec.fit(s.similarity, n, |i| s.vector(i as u32));
            let mut scorer = super::es_quant::EsQuantScorer::new(q, s.similarity, query);
            let mut hits: Vec<(u32, f32)> = chosen.iter().map(|&o| (o, scorer.score(s.vector(o)))).collect();
            super::hnsw::add_vector_ops(hits.len() as u64);
            top_k(&mut hits, k);
            hits
        }
    })
}

/// A dense bitset over [`PointId`]s: an explicit pre-filter for
/// [`VectorStore::search_filtered`].
#[derive(Clone, Debug, Default)]
pub struct PointBits {
    words: Vec<u64>,
    count: usize,
}

impl PointBits {
    /// An empty set sized for point ids `< num_points` (it grows on demand).
    pub fn with_capacity(num_points: u64) -> Self {
        Self { words: vec![0; (num_points as usize).div_ceil(64)], count: 0 }
    }

    /// Add the points of `words`, dense words starting at word `first`.
    pub fn or_words(&mut self, first: usize, words: &[u64]) {
        if self.words.len() < first + words.len() {
            self.words.resize(first + words.len(), 0);
        }
        for (dst, w) in self.words[first..].iter_mut().zip(words) {
            self.count += (w & !*dst).count_ones() as usize;
            *dst |= w;
        }
    }

    /// The set of `points` (duplicates allowed).
    pub fn from_points(points: &[PointId], num_points: u64) -> Self {
        let max = points.iter().copied().max().map_or(0, |m| m + 1);
        let mut words = vec![0u64; (num_points.max(max) as usize).div_ceil(64)];
        for &p in points {
            words[p as usize / 64] |= 1 << (p % 64);
        }
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Self { words, count }
    }

    /// Add `p`.
    #[inline]
    pub fn insert(&mut self, p: PointId) {
        let w = p as usize / 64;
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        let m = 1u64 << (p % 64);
        if self.words[w] & m == 0 {
            self.words[w] |= m;
            self.count += 1;
        }
    }

    /// Whether `p` is in the set.
    #[inline(always)]
    pub fn contains(&self, p: PointId) -> bool {
        self.words.get(p as usize / 64).is_some_and(|w| w >> (p % 64) & 1 == 1)
    }

    /// Number of points in the set.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Set bits in `lo..=hi` (popcount).
    fn count_range(&self, lo: u64, hi: u64) -> usize {
        let (lw, hw) = (lo as usize / 64, hi as usize / 64);
        if lw >= self.words.len() || lo > hi {
            return 0;
        }
        let hw = hw.min(self.words.len() - 1);
        let mut n: usize = self.words[lw..=hw].iter().map(|w| w.count_ones() as usize).sum();
        // Trim the partial edge words.
        n -= (self.words[lw] & ((1u64 << (lo % 64)) - 1)).count_ones() as usize;
        if hw == hi as usize / 64 && hi % 64 != 63 {
            n -= (self.words[hw] & !((2u64 << (hi % 64)) - 1)).count_ones() as usize;
        }
        n
    }

    /// Every set point in `lo..=hi`, ascending.
    fn for_each_in_range(&self, lo: u64, hi: u64, mut f: impl FnMut(PointId)) {
        let lw = lo as usize / 64;
        let hw = (hi as usize / 64).min(self.words.len().saturating_sub(1));
        if lw >= self.words.len() || lo > hi {
            return;
        }
        for (i, &w0) in self.words[lw..=hw].iter().enumerate() {
            let mut w = w0;
            while w != 0 {
                let p = ((lw + i) * 64) as u64 + w.trailing_zeros() as u64;
                w &= w - 1;
                if p >= lo && p <= hi {
                    f(p);
                }
            }
        }
    }
}

/// Per text segment doc bitsets (Lucene's filter `Bits`): a point matches
/// iff the doc that owns it is set. Built straight from a filter scorer
/// with no doc -> point mapping; the vector search resolves a point's owner
/// only for the graph nodes it actually visits.
#[derive(Clone, Debug, Default)]
pub struct DocFilter {
    /// `(text segment id, doc bitset words)`, ascending id.
    segs: Vec<(u32, Vec<u64>)>,
    count: usize,
}

impl DocFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the matching docs of text segment `segment` (bit `d` = doc `d`
    /// matches; deleted docs should already be cleared).
    pub fn add_segment(&mut self, segment: u32, words: Vec<u64>) {
        self.count += words.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        let i = self.segs.partition_point(|(s, _)| *s < segment);
        if self.segs.get(i).is_some_and(|(s, _)| *s == segment) {
            let old = std::mem::replace(&mut self.segs[i].1, words);
            self.count -= old.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        } else {
            self.segs.insert(i, (segment, words));
        }
    }

    /// Whether `doc` of text segment `segment` matches.
    #[inline(always)]
    pub fn contains(&self, segment: u32, doc: DocId) -> bool {
        let words = if self.segs.len() == 1 {
            if self.segs[0].0 != segment {
                return false;
            }
            &self.segs[0].1
        } else {
            match self.segs.binary_search_by_key(&segment, |(s, _)| *s) {
                Ok(i) => &self.segs[i].1,
                Err(_) => return false,
            }
        };
        words.get(doc as usize / 64).is_some_and(|w| w >> (doc % 64) & 1 == 1)
    }

    /// Matching docs over every segment (an upper bound on matching points
    /// of any one vector field).
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `(text segment id, doc bitset words)`, ascending id.
    pub fn segments(&self) -> impl Iterator<Item = (u32, &[u64])> {
        self.segs.iter().map(|(s, w)| (*s, w.as_slice()))
    }
}

/// A [`DocFilter`] resolved to the vector segments: per segment, a bitset
/// over its point *ordinals* (the graph's node ids) of the alive points
/// whose owner doc passes. Costs one pass over the matching docs to build
/// (see [`VectorStore::ord_filter`]) and then makes the per-node filter
/// check of the graph traversal a single L1-resident bit test instead of
/// three dependent lookups (`ordinal -> point -> owner -> doc bit`), which
/// is what dominates a selective filtered search (the two-hop traversal
/// filter-checks ~`1 / selectivity` nodes per scored one). The exact
/// (full-scan) path enumerates the set ordinals directly.
#[derive(Clone, Debug, Default)]
pub struct OrdFilter {
    /// `(vector segment id, ordinal bitset, count)`, ascending id.
    segs: Vec<(u32, Vec<u64>, usize)>,
    /// The doc filter it was built from (fallback for a vector segment
    /// that was swapped in after the build).
    docs: DocFilter,
}

impl OrdFilter {
    /// The ordinal bits and count of vector segment `sid`.
    #[inline]
    fn get(&self, sid: u32) -> Option<(&[u64], usize)> {
        let i = if self.segs.len() == 1 {
            if self.segs[0].0 != sid {
                return None;
            }
            0
        } else {
            self.segs.binary_search_by_key(&sid, |(s, _, _)| *s).ok()?
        };
        let (_, w, c) = &self.segs[i];
        Some((w.as_slice(), *c))
    }

    /// The doc filter this was built from.
    pub fn docs(&self) -> &DocFilter {
        &self.docs
    }

    /// Matching alive points over every segment.
    pub fn len(&self) -> usize {
        self.segs.iter().map(|(_, _, c)| *c).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[inline(always)]
fn word_bit(words: &[u64], i: u32) -> bool {
    words.get(i as usize >> 6).is_some_and(|w| w >> (i & 63) & 1 == 1)
}

/// A pre-filter for [`VectorStore::search_filtered`].
#[derive(Clone, Copy, Debug)]
pub enum PointFilter<'a> {
    /// Explicit point set: exact per-segment counts, exact searches
    /// enumerate just the matching points. Best for selective filters.
    Points(&'a PointBits),
    /// Owner-doc bitsets: zero per-match preprocessing. Best for broad
    /// filters, which the planner sends through the graph.
    Docs(&'a DocFilter),
    /// Owner-doc bitsets resolved to per-segment ordinal bitsets: O(1)
    /// per-node checks, exact counts. Best for selective filters that are
    /// still too broad for an exact scan (the two-hop graph range).
    Ords(&'a OrdFilter),
}

impl PointFilter<'_> {
    fn is_empty(self) -> bool {
        match self {
            PointFilter::Points(b) => b.is_empty(),
            PointFilter::Docs(d) => d.is_empty(),
            PointFilter::Ords(o) => o.is_empty() && o.docs.is_empty(),
        }
    }
}

/// Search one segment with the planner. `filter = None` searches every
/// alive point. `total` = vectors over all segments of the field (for the
/// [`PointFilter::Docs`] per-segment estimate). `opts` is the field's
/// policy: it resolves `cfg`'s `Auto` quantization and `None` oversample.
#[allow(clippy::too_many_arguments)]
fn search_segment(
    s: &VectorSegment,
    t: &PointTable,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<PointFilter<'_>>,
    total: usize,
    cfg: &PlannerConfig,
    opts: &VectorFieldOptions,
) -> Result<(Vec<(u32, f32)>, SearchStats)> {
    s.note_int4_confidence_interval(opts.confidence_interval);
    // The search runs on this thread, which counts its vector operations.
    let ops0 = super::hnsw::vector_ops();
    let quant = match cfg.quantization {
        Quantization::Auto => opts.quantization,
        q => q,
    };
    let esq = super::es_quant::EsQuant::of(quant, opts, cfg.oversample.unwrap_or(opts.oversample));
    let (hits, mut st) = match esq {
        // ES returns the segment-quantized estimates: score the best
        // `max(k, ef)` candidates that way and keep the top `k`.
        Some(spec) => {
            let kk = k.max(ef);
            let (cands, st) = plan_segment_search(s, t, query, kk, ef.max(kk), filter, total, cfg, opts);
            let coded = s.es_codes(&spec)?;
            let scorer = super::es_quant::EsQuantScorer::new(coded.q.clone(), s.similarity, query);
            (scorer.top_k_coded(cands, k, &coded), st)
        }
        None => plan_segment_search(s, t, query, k, ef, filter, total, cfg, opts),
    };
    st.vector_ops = super::hnsw::vector_ops().wrapping_sub(ops0);
    Ok((hits, st))
}

/// [`search_segment`] without the operation count.
#[allow(clippy::too_many_arguments)]
fn plan_segment_search(
    s: &VectorSegment,
    t: &PointTable,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<PointFilter<'_>>,
    total: usize,
    cfg: &PlannerConfig,
    opts: &VectorFieldOptions,
) -> (Vec<(u32, f32)>, SearchStats) {
    let mut st = SearchStats::default();
    let n = s.len() as usize;
    let pts = s.points();
    // Segment points are ascending (see `rebuild` / `append`).
    let (lo, hi) = match (pts.first(), pts.last()) {
        (Some(&a), Some(&b)) => (a, b),
        _ => return (Vec::new(), st),
    };
    // Quantized scoring: the request's, else the field's (`Auto` there
    // means whatever copy the segment stores).
    let quant = match cfg.quantization {
        Quantization::Auto => opts.quantization,
        q => q,
    };
    let oversample = match cfg.oversample.unwrap_or(opts.oversample) {
        o if o > 0.0 => o * opts.rescore_factor,
        o => o,
    };
    #[cfg(feature = "payload-hnsw-poc")]
    let internal_quant = if quant == Quantization::None {
        std::env::var("RUSTIC_POC_GRAPH_QUANT").ok().map(|s| s.parse::<f32>().expect("positive POC rescore factor"))
    } else { None };
    #[cfg(feature = "payload-hnsw-poc")]
    let (quant, oversample) = if let Some(factor) = internal_quant {
        assert!(factor.is_finite() && factor >= 1.0 && factor <= 100.0);
        (Quantization::Int8, factor)
    } else { (quant, oversample) };
    // An ordinal filter for this segment (alive points already excluded);
    // a segment it does not know falls back to its doc filter.
    let ords: Option<(&[u64], usize)> = match filter {
        Some(PointFilter::Ords(o)) => o.get(s.id),
        _ => None,
    };
    let doc_filter: Option<&DocFilter> = match filter {
        Some(PointFilter::Docs(d)) => Some(d),
        Some(PointFilter::Ords(o)) if ords.is_none() => Some(&o.docs),
        _ => None,
    };
    // Upper bound on the matching points of this segment.
    let matching = match (filter, ords, doc_filter) {
        (None, _, _) => n,
        (_, Some((_, c)), _) => c.min(n),
        (Some(PointFilter::Points(b)), _, _) => b.count_range(lo, hi).min(n),
        (_, _, Some(d)) => ((d.len() as u128 * n as u128).div_ceil(total.max(1) as u128) as usize).min(n),
        _ => n,
    };
    if matching == 0 {
        return (Vec::new(), st);
    }
    let accept = |o: u32| -> bool {
        if let Some((w, _)) = ords {
            return word_bit(w, o);
        }
        let p = pts[o as usize];
        t.is_alive(p)
            && match (filter, doc_filter) {
                (None, _) => true,
                (Some(PointFilter::Points(b)), _) => b.contains(p),
                (_, Some(d)) => t.owner(p).is_some_and(|(sg, doc)| d.contains(sg, doc)),
                _ => true,
            }
    };
    // Brute force over the matching ordinals. Sealed segments score their
    // quantized copy (then rescore) like ES's exact fallback on a quantized
    // index; appendable chunks have no copy and score `f32`.
    let flat = |st: &mut SearchStats| -> Vec<(u32, f32)> {
        st.exact_segments += 1;
        let q = if s.is_sealed() { quant } else { Quantization::None };
        match filter {
            // Ordinal bitset: enumerate just the set ordinals.
            Some(PointFilter::Ords(_)) if ords.is_some() => {
                let w = ords.unwrap().0;
                s.search_flat(query, k, q, oversample, SetBits::new(w))
            }
            // Sparse explicit set: enumerate just its points in range.
            Some(PointFilter::Points(b)) if matching * 8 < n => {
                let mut ords = Vec::with_capacity(matching);
                b.for_each_in_range(lo, hi, |p| {
                    if t.is_alive(p) {
                        if let Some((sid, o)) = t.loc(p) {
                            if sid == s.id {
                                ords.push(o);
                            }
                        }
                    }
                });
                s.search_flat(query, k, q, oversample, ords.into_iter())
            }
            _ => s.search_flat(query, k, q, oversample, (0..n as u32).filter(|&o| accept(o))),
        }
    };
    // Keep the exact-scan planner for selective filters. Payload navigation is
    // used only when this immutable segment covers the entire accepted set.
    let payload = quant == Quantization::None && filter.is_some()
        && matching > cfg.full_scan_limit(s.dim)
        && matching as f32 > cfg.full_scan_fraction * n as f32
        && ords.is_some_and(|(words, _)| s.graph().is_some_and(|g| g.payload_covers_words(words)));
    #[cfg(feature = "payload-hnsw-poc")]
    let payload = payload || (matching < n && filter.is_some() && (quant == Quantization::None || internal_quant.is_some()) && std::env::var_os("RUSTIC_POC_PAYLOAD_GRAPH").is_some());
    if !s.has_graph() || (!payload && (matching <= cfg.full_scan_limit(s.dim) || matching as f32 <= cfg.full_scan_fraction * n as f32)) {
        return (flat(&mut st), st);
    }
    st.graph_segments += 1;
    // Selective filters: two-hop traversal (score only accepted nodes). An
    // ordinal bitset is cheaper to test than a visited mark, so it is not
    // memoised.
    let mode = if filter.is_some() && (matching as f32) < cfg.two_hop_fraction * n as f32 {
        FilterMode::TwoHop { memo: ords.is_none() }
    } else {
        FilterMode::Plain
    };
    let mode = if payload { FilterMode::Payload } else { mode };
    let hits = s.search_graph_mode(query, k, ef, quant, oversample, accept, mode);
    if hits.len() < k.min(matching) {
        st.fallbacks += 1;
        st.graph_segments -= 1;
        return (flat(&mut st), st);
    }
    (hits, st)
}

/// The 64 bits of `words` starting at bit position `bit` (which may be
/// negative or past the end: bits outside the bitset read as 0).
#[inline(always)]
fn shifted_word(words: &[u64], bit: i64) -> u64 {
    let get = |w: i64| -> u64 {
        if w < 0 || w >= words.len() as i64 {
            0
        } else {
            words[w as usize]
        }
    };
    let w = bit.div_euclid(64);
    let r = bit.rem_euclid(64) as u32;
    if r == 0 {
        get(w)
    } else {
        (get(w) >> r) | (get(w + 1) << (64 - r))
    }
}

/// The lowest `n` bits set (`n <= 0` -> 0, `n >= 64` -> all).
#[inline(always)]
fn low_mask(n: i64) -> u64 {
    if n <= 0 {
        0
    } else if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// Iterator over the set bit positions of a bitset, ascending.
struct SetBits<'a> {
    words: &'a [u64],
    i: usize,
    cur: u64,
}

impl<'a> SetBits<'a> {
    fn new(words: &'a [u64]) -> Self {
        Self { words, i: 0, cur: words.first().copied().unwrap_or(0) }
    }
}

impl Iterator for SetBits<'_> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        while self.cur == 0 {
            self.i += 1;
            if self.i >= self.words.len() {
                return None;
            }
            self.cur = self.words[self.i];
        }
        let b = self.cur.trailing_zeros();
        self.cur &= self.cur - 1;
        Some((self.i * 64) as u32 + b)
    }
}

/// Merge per-source hit lists into one top-`k` (score desc, key asc).
pub fn merge_top_k<T: Ord + Copy>(lists: impl IntoIterator<Item = Vec<(T, f32)>>, k: usize) -> Vec<(T, f32)> {
    let mut all: Vec<(T, f32)> = lists.into_iter().flatten().collect();
    all.sort_by(by_score);
    all.truncate(k);
    all
}

#[cfg(test)]
mod bit_tests {
    use super::*;

    #[test]
    fn shifted_word_reads_across_words_and_edges() {
        let words = [0xF0F0_F0F0_F0F0_F0F0u64, 0x0F0F_0F0F_0F0F_0F0Fu64, 0xFFFF_FFFF_FFFF_FFFFu64];
        let bit = |b: i64| -> u64 {
            if b < 0 || b >= 192 {
                0
            } else {
                words[b as usize / 64] >> (b % 64) & 1
            }
        };
        for start in -70i64..200 {
            let got = shifted_word(&words, start);
            for i in 0..64 {
                assert_eq!(got >> i & 1, bit(start + i), "start {start} bit {i}");
            }
        }
        assert_eq!(low_mask(-3), 0);
        assert_eq!(low_mask(0), 0);
        assert_eq!(low_mask(1), 1);
        assert_eq!(low_mask(63), u64::MAX >> 1);
        assert_eq!(low_mask(64), u64::MAX);
        assert_eq!(low_mask(100), u64::MAX);
        assert_eq!(SetBits::new(&[0b1010, 0, 1 << 63]).collect::<Vec<_>>(), vec![1, 3, 191]);
        assert_eq!(SetBits::new(&[]).count(), 0);
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;

    #[test]
    fn live_point_tables_are_charged_and_copies_are_not() {
        let mut table = PointTable::default();
        table.ensure(1000);
        assert_eq!(table.charged.0, table.dense_bytes() as u64);
        assert!(table.charged.0 >= (OWNER_CHUNK * 8) as u64);
        let copy = table.clone();
        assert_eq!(copy.charged.0, 0, "a capture's copy is charged by the capture");
        assert!(copy.heap_bytes() >= OWNER_CHUNK * 8, "the copy owns its owner column");
        table.ensure(10);
        assert_eq!(table.charged.0, table.dense_bytes() as u64, "never shrinks below the arrays");
    }

    /// Owner lookups of a mutable kNN query take no lock: they resolve
    /// while a writer holds the point table, and a writer never waits for a
    /// query's lookups.
    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn point_owner_lookups_take_no_lock() {
        let store = VectorStore::new();
        {
            let mut table = store.table.write();
            table.ensure(3);
            table.owner.set(2, pack(7, 1));
        }
        let owners = store.mutable_point_owners();
        let held = store.table.write();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let owners = &owners;
            scope.spawn(move || tx.send((owners.owner(2), owners.owner(1), owners.owner(5))).unwrap());
            let got = rx.recv_timeout(std::time::Duration::from_secs(10));
            drop(held);
            assert_eq!(got.expect("an owner lookup waited for the table lock"), (Some((7, 1)), None, None));
        });
        // Live entries: an owner change after the snapshot is seen.
        store.table.write().owner.set(2, pack(8, 0));
        assert_eq!(owners.owner(2), Some((8, 0)));
    }

    /// R8a: a long-lived row snapshot (a retained view's) holds each
    /// segment once, however many rows it reads.
    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn row_snapshot_holds_each_segment_once() {
        use crate::codec::directory::RamDirectory;
        let dir = RamDirectory::new();
        let rows: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32, 1.0, 2.0]).collect();
        let pairs: Vec<(u64, &[f32])> = rows.iter().enumerate().map(|(i, row)| (i as u64, row.as_slice())).collect();
        let segment = Arc::new(VectorSegment::write_with(&dir, 0, 0, Similarity::Cosine, 3, &pairs, SealSpec::chunk()).unwrap());
        let store = VectorStore::new();
        store.register(segment).unwrap();
        let snapshot = StoreRowsSnapshot { store: Arc::downgrade(&store), held: Mutex::new(BTreeMap::new()), failed:AtomicBool::new(false), _memory:None };
        for round in 0..10_000u64 {
            let key = round % 8;
            assert_eq!(super::super::hnsw::RawRowsSnapshot::row(&snapshot, key), Some(rows[key as usize].as_slice()));
        }
        assert_eq!(snapshot.held.lock().len(), 1, "the snapshot grew per row read");
    }

    #[test]
    fn vector_reads_pin_rows_across_segment_replacement() {
        use crate::codec::directory::RamDirectory;
        let dir = RamDirectory::new();
        let row = [3.0f32, 4.0, 1.0];
        let a = Arc::new(VectorSegment::write_with(&dir, 0, 0, Similarity::Cosine, 3, &[(0, row.as_slice())], SealSpec::chunk()).unwrap());
        let b = Arc::new(VectorSegment::write_with(&dir, 1, 0, Similarity::Cosine, 3, &[(0, row.as_slice())], SealSpec::chunk()).unwrap());
        let store = VectorStore::new();
        store.register(a.clone()).unwrap();
        // The callback owns a pin, not a store lock: replacement may finish
        // while its borrowed row remains readable.
        assert_eq!(store.with_vector(0, |values| {
            store.replace(&[0], Some(b.clone()));
            values.to_vec()
        }), Some(row.to_vec()));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|scope| {
            let reader_barrier = barrier.clone();
            let reader_store = &store;
            let reader = scope.spawn(move || {
                reader_barrier.wait();
                for _ in 0..100_000 { assert_eq!(reader_store.vector(0).as_deref(), Some(row.as_slice())); }
            });
            barrier.wait();
            for i in 0..10_000 {
                let (old, next) = if i % 2 == 0 { (1, a.clone()) } else { (0, b.clone()) };
                store.replace(&[old], Some(next));
            }
            reader.join().unwrap();
        });
        assert!(store.with_vector(99, |v| v.len()).is_none());
        let mut visited = Vec::new();
        assert!(!store.visit_vectors([(42, Some(0)), (43, None)], |doc, values| {
            visited.push((doc, values.to_vec())); true
        }));
        assert_eq!(visited, vec![(42, row.to_vec())]);
        assert!(store.visit_vectors([(99, Some(0))], |doc, values| doc == 99 && values == row.as_slice()));
    }

    #[test]
    fn vector_batch_visits_resolve_non_monotonic_segment_transitions() {
        use crate::codec::directory::RamDirectory;
        let dir = RamDirectory::new();
        let store = VectorStore::new();
        for (id, point, row) in [(0, 0, [1.0f32, 2.0]), (1, 1, [3.0, 4.0])] {
            store.register(Arc::new(VectorSegment::write_with(&dir, id, 0, Similarity::Cosine, 2,
                &[(point, row.as_slice())], SealSpec::chunk()).unwrap())).unwrap();
        }
        let mut rows = Vec::new();
        assert!(store.visit_vectors([(10, Some(0)), (11, Some(0)), (12, Some(1)), (13, Some(0))],
            |doc, row| { rows.push((doc, row.to_vec())); true }));
        assert_eq!(rows, vec![(10, vec![1.0, 2.0]), (11, vec![1.0, 2.0]),
            (12, vec![3.0, 4.0]), (13, vec![1.0, 2.0])]);
        let mut visited = 0;
        assert!(!store.visit_vectors([(0, Some(0)), (1, Some(1))], |_, _| { visited += 1; false }));
        assert_eq!(visited, 1);
        assert!(!store.visit_vectors([(0, Some(0)), (1, Some(99))], |_, _| true));
    }

    #[test]
    fn vector_live_counts_follow_repeated_changes_relocation_and_removal() {
        let mut table = PointTable::default();
        table.ensure(257);
        let mut seed = 17u64;
        for step in 0..20_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let point = (seed >> 32) % 257;
            match step % 5 {
                0 => table.set_loc(point, pack((seed % 7) as u32, point as u32)),
                1 => table.set_alive(point, true),
                2 => table.set_alive(point, false),
                3 => table.set_loc(point, NONE),
                _ => { let alive = table.is_alive(point); table.set_alive(point, alive); }
            }
            // Hidden turns on and off independently of liveness.
            if step % 3 == 0 { table.set_hidden((seed >> 40) % 257, (seed >> 20) & 1 == 1); }
            let mut expected = BTreeMap::<u32, usize>::new();
            let mut kept = BTreeMap::<u32, usize>::new();
            for p in 0..257 {
                if let Some((segment, _)) = table.loc(p) {
                    if table.is_alive(p) { *expected.entry(segment).or_default() += 1; }
                    if table.is_kept(p) { *kept.entry(segment).or_default() += 1; }
                }
            }
            assert_eq!(table.live_by_segment, expected, "step {step}");
            assert_eq!(table.kept_by_segment, kept, "kept, step {step}");
        }
    }

    #[test]
    fn vector_live_count_component_probe() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut table = PointTable::default();
        let count = 1_000_000;
        for page in (0..count).step_by(OWNER_CHUNK) { table.ensure(page + 1); }
        for p in 0..count as u64 {
            table.set_loc(p, pack(1, p as u32));
            table.set_alive(p, p % 7 != 0);
        }
        let started = Instant::now();
        let mut expected = 0;
        for _ in 0..32 { expected = (0..count as u64).filter(|p| black_box(&table).is_alive(*p)).count(); black_box(expected); }
        let scan_ns = started.elapsed().as_nanos();
        let started = Instant::now();
        for _ in 0..32 {
            let actual = black_box(&table).live_by_segment.get(&black_box(1)).copied().unwrap_or(0);
            assert_eq!(actual,expected);
            black_box(actual);
        }
        eprintln!("VECTOR_LIVE_COUNT_POC points={count} repetitions=32 scan_ns={scan_ns} counted_ns={}", started.elapsed().as_nanos());
    }

    #[test]
    fn vector_stats_take_state_before_point_table() {
        use std::time::{Duration, Instant};
        let store = VectorStore::new();
        // Hold the second lock. The stats worker must acquire state before
        // waiting for this table; replace() uses the same order. Release the
        // table before asserting so a broken order fails rather than hangs.
        let table = store.table.write();
        let worker_store = store.clone();
        let (started, ready) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            started.send(()).unwrap();
            worker_store.num_alive(0)
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let ordered = loop {
            if store.state.try_write().is_none() { break true; }
            if Instant::now() >= deadline { break false; }
            std::thread::sleep(Duration::from_millis(1));
        };
        drop(table);
        assert_eq!(worker.join().unwrap(), 0);
        assert!(ordered, "vector stats must acquire state before waiting for the point table");
    }

    #[test]
    fn default_store_is_empty() {
        let store = VectorStore::default();
        assert_eq!(store.segment_ids().len(), 0);
        assert_eq!(store.next_point_id(), 0);
    }
}

#[cfg(test)]
mod point_table_admission_tests {
    use super::*;

    #[test]
    fn payload_compaction_retires_dead_point_page_but_not_captured_copy() {
        let dir = crate::codec::directory::RamDirectory::new();
        let segment = Arc::new(VectorSegment::write_with(&dir, 0, 0, Similarity::Cosine, 2,
            &[(0, &[1.0, 0.0])], SealSpec::chunk()).unwrap());
        let budget = crate::admission::Budget::new(4 << 20);
        let store = store_with_budget(&budget);
        store.register(segment).unwrap();
        let captured = {
            let mut table = store.table.write();
            table.owner.set(0, pack(7, 2)); table.set_alive(0, true);
            // Match capture_read_store's pre-copy ownership without touching
            // the process-global budget in this isolated regression.
            let memory=budget.reserve((table.heap_bytes() as u64).saturating_mul(2).saturating_add(4096), "captured point table").unwrap();
            crate::admission::Allocation::new(table.clone(),memory)
        };
        let captured_bytes=(captured.heap_bytes() as u64).saturating_mul(2).saturating_add(4096);
        store.delete(0, 7, 2);
        assert_eq!(store.table.read().loc(0), Some((0, 0)), "delete keeps durable location until compaction");
        store.replace_locked(&[0], None, None);
        assert!(store.table.read().loc.is_empty());
        assert!(store.table.read().owner.chunks.is_empty());
        assert_eq!(captured.loc(0), Some((0, 0)));
        assert_eq!(captured.owner(0), Some((7, 2)));
        assert!(captured.is_alive(0));
        // Retiring the page does not destroy the live directory's Arc/BTree
        // backing. Its independent generation owner remains until store drop.
        let directory_bytes=store.table.read().owner.chunks._memory.as_ref().unwrap().bytes();
        assert_eq!(directory_bytes,256+128,"single-page directory generation bound");
        assert_eq!(budget.used(),captured_bytes+directory_bytes,"only capture and live empty directory remain charged");
        drop(store);assert_eq!(budget.used(),captured_bytes);
        drop(captured);assert_eq!(budget.used(),0);
    }

    #[test]
    fn abandoned_point_ranges_reclaim_pages_without_reusing_ids() {
        let budget = crate::admission::Budget::new(4 << 20);
        let store = store_with_budget(&budget);
        for _ in 0..32 {
            let reservation = store.reserve_points(OWNER_CHUNK).unwrap();
            assert_eq!(store.table.read().loc.pages.len(), 1);
            drop(reservation);
            assert!(store.table.read().loc.is_empty());
            assert!(store.table.read().owner.chunks.is_empty());
            assert!(budget.used() < 1024, "only directory metadata remains");
        }
        assert_eq!(store.next_point_id(), (32 * OWNER_CHUNK) as u64);
        drop(store); assert_eq!(budget.used(), 0);
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn retired_page_waits_for_pending_writes_and_escaped_owner_readers() {
        let budget = crate::admission::Budget::new(4 << 20);
        let store = store_with_budget(&budget);
        let pending = store.reserve_points(1).unwrap();
        let reader = store.mutable_point_owners();
        let before = budget.used();
        store.table.write().retire_empty_page(0);
        assert_eq!(store.table.read().loc.pages.len(), 1, "pending write owns page");
        drop(pending);
        assert!(store.table.read().loc.is_empty());
        assert_eq!(budget.used(), before, "reader retains retired atomic page and directory");
        let replacement = store.reserve_points(1).unwrap();
        assert_eq!(replacement.points().next(), Some(1));
        store.table.write().owner.set(1, pack(7, 2));
        assert_eq!(reader.owner(1), None, "retired snapshot never aliases new page");
        drop(reader);
        assert_eq!(budget.used(), before, "only replacement page remains");
        drop(replacement); drop(store); assert_eq!(budget.used(), 0);
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn historical_owner_directories_keep_independent_admission() {
        let budget = crate::admission::Budget::new(4 << 20);
        let store = store_with_budget(&budget);
        let first = store.reserve_points(OWNER_CHUNK).unwrap();
        let old = store.mutable_point_owners();
        let second = store.reserve_points(OWNER_CHUNK).unwrap();
        let before = budget.used();
        drop(old);
        assert_eq!(before - budget.used(), 256 + 128, "old directory metadata released independently of current table");
        drop(first); drop(second); drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn sparse_historical_ids_allocate_only_resident_pages_and_reopen() {
        let dir = crate::codec::directory::RamDirectory::new();
        let point = 1u64 << 40;
        let segment = VectorSegment::write_with(&dir, 0, 0, Similarity::Cosine, 2,
            &[(point, &[1.0, 0.0])], SealSpec::chunk()).unwrap();
        drop(segment);
        let store = VectorStore::open(&dir, &[0], point + (1 << 30), 1).unwrap();
        let mut table = store.table.write();
        assert_eq!(table.owner.chunks.len(), 1);
        assert!(table.heap_bytes() < 128 * 1024);
        assert_eq!(table.loc(point), Some((0, 0)));
        assert_eq!(table.loc(point - OWNER_CHUNK as u64), None);
        table.owner.set(point as usize, pack(7, 2));
        table.set_alive(point, true);
        let captured = table.clone();
        table.owner.set(point as usize, pack(8, 3));
        table.set_alive(point, false);
        assert_eq!(captured.owner(point), Some((7, 2)));
        assert!(captured.is_alive(point));
        assert!(!table.is_alive(point));
        assert!(table.live_by_segment.is_empty());
        drop(table);
        let reservation = store.reserve_points(2).unwrap();
        assert_eq!(reservation.points().next(), Some(point + (1 << 30)));
        assert_eq!(store.table.read().owner.chunks.len(), 2);
        assert_eq!(store.next_point_id(), point + (1 << 30) + 2);
    }

    #[test]
    fn sparse_alive_words_cross_page_boundaries_and_holes() {
        let mut table = PointTable::default();
        let point = (1usize << 30) + OWNER_CHUNK - 7;
        table.ensure(point + 1);
        table.ensure(point + 65);
        for i in 0..64 { table.set_alive((point + i) as u64, i % 3 == 0); }
        let expected = (0..64).fold(0u64, |bits, i| bits | (u64::from(i % 3 == 0) << i));
        assert_eq!(table.alive_word(point as i64), expected);
        assert_eq!(table.alive_word((point - OWNER_CHUNK * 2) as i64), 0);
    }

    #[test]
    fn exhausted_point_allocator_refuses_without_wrapping_or_allocating() {
        let budget=crate::admission::Budget::new(4<<20);
        let store=store_with_budget(&budget);
        store.next_point.store(u64::MAX-1,AtomicOrdering::Relaxed);
        assert!(matches!(store.reserve_points(2),Err(crate::error::RusticError::Memory(_))));
        assert_eq!(store.next_point_id(),u64::MAX-1);
        assert!(store.table.read().loc.is_empty());
        assert_eq!(budget.used(),0,"overflow must refuse before page growth");
        store.next_point.store(u64::MAX,AtomicOrdering::Relaxed);
        let error = store.reserve_points(1).err().expect("exhausted allocator must reject");
        assert!(matches!(&error, crate::error::RusticError::Memory(reason) if reason == "vector point IDs exhausted"), "{error}");
        assert_eq!(store.next_point_id(), u64::MAX);
        assert!(store.segment_ids().is_empty());
        assert_eq!(budget.used(),0);
        drop(store);assert_eq!(budget.used(),0);
    }


    fn store_with_budget(budget: &Arc<crate::admission::Budget>) -> Arc<VectorStore> {
        let store = VectorStore::new();
        store.table.write().admission = TableAdmission(Arc::new(Mutex::new(Some(budget.reserve(0, "test").unwrap()))));
        store
    }

    #[test]
    fn dense_point_refusal_precedes_ids_and_table_mutation() {
        let budget = crate::admission::Budget::new(1);
        let store = store_with_budget(&budget);
        assert!(matches!(store.reserve_points(10), Err(crate::error::RusticError::Memory(_))));
        assert_eq!(store.next_point_id(), 0);
        let table = store.table.read();
        assert_eq!(table.owner.len(), 0);
        assert!(table.loc.is_empty() && table.payload.is_empty());
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn reopened_segment_refusal_precedes_registration() {
        let dir = crate::codec::directory::RamDirectory::new();
        let segment = Arc::new(VectorSegment::write_with(&dir, 0, 0, Similarity::Cosine, 2,
            &[(0, &[1.0, 0.0])], SealSpec::chunk()).unwrap());
        let budget = crate::admission::Budget::new(1);
        let store = store_with_budget(&budget);
        assert!(matches!(store.register(segment), Err(crate::error::RusticError::Memory(_))));
        assert!(store.state.read().segments.is_empty());
        assert!(store.table.read().loc.is_empty());
        assert_eq!(budget.used(), 0);
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn dense_point_admission_follows_escaped_owner_snapshot() {
        let budget = crate::admission::Budget::new(4 << 20);
        let store = store_with_budget(&budget);
        let points = store.reserve_points(2).unwrap();
        let owners = store.mutable_point_owners();
        assert_eq!(store.next_point_id(), 2);
        assert!(budget.used() > 0);
        drop(points);
        drop(store);
        assert!(budget.used() > 0, "owner chunks still retained by reader");
        drop(owners);
        assert_eq!(budget.used(), 0);
    }
}

#[cfg(test)]
mod payload_ceiling_tests {
    use super::*;

    #[test]
    fn deletes_only_ceiling_merges_only_the_smallest_above_max_segments() {
        let row = 1536 * 4;
        // At or below the ceiling: nothing, whatever the sizes.
        assert_eq!(plan_ceiling(&[1, 2, 3], 3, row), 0);
        assert_eq!(plan_ceiling(&[100_000; 200], 256, row), 0);
        // One over: the smallest are merged (never the largest), extended up
        // to a tier's worth of segments.
        assert_eq!(plan_ceiling(&[50, 200, 200, 200], 3, row), 3);
        let big = vec![20_000usize; 300]; // ~117 MiB each, above the floor
        assert_eq!(plan_ceiling(&big, 256, row), 45, "back to the ceiling in one merge");
        assert_eq!(plan_ceiling(&big[..257], 256, row), PAYLOAD_PER_TIER);
        // The merged size stays within the cap once the minimum is met.
        let huge = PAYLOAD_MAX_MERGED_BYTES / row / 2; // two fill the cap
        let sizes = vec![huge; 12];
        assert_eq!(plan_ceiling(&sizes, 11, row), 2);
    }

    #[test]
    fn configured_vacuum_ratio_is_respected_under_deletes_only() {
        let policy = OptimizerPolicy { vacuum_deleted_ratio: 0.1, payload_merge: PayloadMerge::DeletesOnly, ..OptimizerPolicy::default() };
        assert_eq!(policy.vacuum_ratio(), 0.1);
        let policy = OptimizerPolicy { vacuum_deleted_ratio: 0.5, payload_merge: PayloadMerge::DeletesOnly, ..OptimizerPolicy::default() };
        assert_eq!(policy.vacuum_ratio(), 0.5);
        if PayloadMerge::from_env() == PayloadMerge::DeletesOnly {
            assert_eq!(OptimizerPolicy::default().vacuum_ratio(), PAYLOAD_DELETES_ONLY_VACUUM);
        }
    }
}
