//! Shared mutable-HNSW lifecycle used by the ES vector store for empty
//! unquantized (`hnsw`, `byte`, `bit`) fields; quantized and pre-populated
//! fields stay on the segmented store.
//! The historical `mutable-hnsw-poc` Cargo feature name remains for library builds.
//!
//! Ordered reservations allocate generation-tagged slots before parallel
//! preparation. One topology writer publishes immutable neighbor arrays.
//! Readers pin the shared storage generation before capturing graph bounds; they never lock the
//! writer. Deletion repairs local links and returns a source-reclamation lease
//! only after old graph readers and prepared-operation leases drain. Dropping
//! that lease permits slot reuse. Persistent adjacency contains only u32 IDs;
//! transient candidates/proposals retain the acquired node incarnation.
//!
//! This is current topology, NOT historical PIT topology. Visibility-generation
//! retirement delays physical deletion but does not retain old adjacency roots.
//! One exception keeps live readers from losing hits during delete repair: a
//! removed slot remembers its last levels for readers admitted at or before
//! the removal (they may still hold a neighbour's pre-rewrite link list), so
//! such a reader can traverse through it, never returning it as a hit.
//! Source snapshots separately own their immutable rows. Slot reuse alone does
//! not establish bounded total memory if that source retains historical rows.
//!
//! Construction uses Vespa's shared fixed-seed level stream and prepare-time
//! bootstrap marker (default threshold 10,000). HnswParams.seed remains legacy
//! codec metadata; it does not seed this mutable stream. Completion preserves
//! prepared-neighbor order, reconnect compares numeric distances only, and
//! Plain tightens its cutoff only after result overflow, matching the pinned
//! Vespa loops. Architecture-specific SIMD reduction and Rust equal-distance
//! heap/sort ordering remain differences; this is not byte-identical Vespa ANN.
//! RMHG4 (with validated RMHG3 migration) reopens saved topology but starts a fresh fixed-seed generator; it does
//! not persist or claim to continue the previous in-memory random stream.

use super::*;
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

// Native packed storage shares one generation boundary with graph readers and
// prepared operations in the integrated ES vector-store path.
mod adaptive;
mod array_store;
mod compaction;
mod dense_beam;
mod parallel_beam;
pub(crate) use parallel_beam::beam_workers;
pub(crate) use source::SharedQuantized;
mod huge;
mod filter_first;
mod generation;
mod level_generator;
mod node_table;
mod storage;
mod source;
pub use source::{CapturedMutableVectorRead, MutableVectorCheckpoint, MutableVectorOwner, MutableVectorRead, PendingMutableCheckpoint, RawRows, RawRowsSnapshot};
mod visited;
use array_store::{LevelsRef, LinksRef};
pub use filter_first::{PreparedNodeFilter, MutableSearchMode, VespaFilterEstimate, VespaFilterFirstParams, VespaPlainParams};
use generation::GenerationGuard;
use node_table::{NodeIdentity, NodeMetadata, NodeRecord, NodeState};
use storage::{CaptureLog, GraphStorage as Control, Writer};
pub use storage::{InsertReservation, MutableArrayMemory, MutableStorageMemory, ReclaimedNode};
use visited::{FilterEstimate, VisitedTracker, estimate_visited_nodes};

const NO_ENTRY: u64 = 0;
// Pinned default_nearest_neighbor_index_factory.cpp:34.
const MIN_SIZE_BEFORE_TWO_PHASE: u32 = 10_000;

/// Slot generations protect against ABA; they are not refresh generations or
/// durable operation identities. Handles are scoped to one graph incarnation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeHandle(u64);
impl NodeHandle {
    fn new(slot: u32, generation: u32) -> Self {
        Self((u64::from(generation) << 32) | u64::from(slot))
    }
    pub fn slot(self) -> u32 {
        self.0 as u32
    }
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Query-bound encoded scoring. Missing codes are a missing source error, never
/// permission to fall back to raw arithmetic. Implementations must key codes by
/// the complete node identity and retain their storage for the query lifetime.
pub trait MutableQueryScorer {
    /// ES score (higher is better), before optional exact rescoring.
    fn score(&self, handle: NodeHandle, key: u64) -> Option<f32>;
    /// Scores of graph nodes `ids` (`slot + 1`) of nodes the caller acquired
    /// under its guard, read by id from a slot-addressed code store, into
    /// `out` (NaN: no entry, score that node with [`score`](Self::score)).
    /// `false`: no such store (nothing written).
    fn score_ids(&self, _ids: &[u32], _out: &mut [f32]) -> bool { false }
    /// Warm node `id`'s codes for [`score_ids`](Self::score_ids) (no read).
    fn prefetch_id(&self, _id: u32) {}
    /// [`score_ids`](Self::score_ids) of code plane entries given by
    /// address (`DenseLayout::Codes` entries of nodes the caller acquired;
    /// 0: no entry, NaN). `false`: no such store (nothing written).
    fn score_entries(&self, _entries: &[usize], _out: &mut [f32]) -> bool { false }
    /// The query as this scorer quantized it, per calibration epoch met so
    /// far, for [`MutableVectorSource::bind_query_seeded`] on another
    /// thread (`None`: nothing to share).
    fn quantized_state(&self) -> Option<Vec<(u64, crate::vector::es_quant::EsQuantScorer)>> { None }
}

/// Borrow through a caller-pinned source root. Production adapters should use
/// get_node's direct slot/generation lookup, not a hash/Arc clone per distance.
/// Install the reservation's row before prepare; remove the matching descriptor
/// when its ReclaimedNode arrives, retaining owners in older source roots.
/// For a growing COW directory, capture graph.snapshot() before pinning the
/// source root, then call snapshot.search. Passing an older captured root to
/// graph.search can include newer graph nodes absent from that source; the
/// missing_source_lookups diagnostic makes that incomplete result explicit.
/// A key/handle must always resolve to identical row values through preparation,
/// completion and search; relocation may change ownership, never those values.
/// Each preparation/completion source view must cover the published candidates
/// it can encounter (a stale immutable root is insufficient during insertion).
/// The caller must keep source ownership covering prepare through complete/drop.
/// Prepared graph leases withhold reclamation tokens for previously visible
/// source rows; they do not own an arbitrary borrowed source implementation.
pub trait MutableVectorSource {
    /// Deferred source ownership failures invalidate the whole operation.
    fn execution_result(&self)->Result<()> {Ok(())}
    /// How this source lays out what live traversal scores, when that is
    /// dense by graph node id (`slot + 1`) for every node a live view can
    /// acquire: then the level-0 beam reads it by id (`dense_beam`).
    fn dense_layout(&self) -> Option<DenseLayout<'_>> { None }
    /// Bind once per query. Mutation/graph construction always uses raw rows.
    fn bind_query(&self, _query: &[f32]) -> Option<Box<dyn MutableQueryScorer + '_>> { None }
    /// [`bind_query`](Self::bind_query) for one worker of a search whose
    /// workers share their quantizations of the query (`shared`, seeded
    /// with `MutableQueryScorer::quantized_state`): each calibration epoch
    /// is quantized once per search.
    fn bind_query_shared<'s>(&'s self, query: &[f32], _shared: &'s source::SharedQuantized) -> Option<Box<dyn MutableQueryScorer + 's>> {
        self.bind_query(query)
    }

    /// Warm the directory slot before resolving the vector, like Vespa prefetch_docid.
    fn prefetch_node(&self, _handle: NodeHandle, _key: u64) {}
    /// Warm the vector row itself (its directory entry should already be
    /// warm from `prefetch_node`). Called for a whole neighbourhood before
    /// any of it is scored, so the row misses overlap instead of serializing.
    fn prefetch_row(&self, _handle: NodeHandle, _key: u64) {}
    /// Warm what scoring graph node `id` (`slot + 1`) reads, knowing only its
    /// id: the traversal's look-ahead issues it for the next candidate's
    /// neighbours before their node records are read (a fixed-stride store
    /// computes the row's address; a directory warms its slot).
    fn prefetch_slot(&self, _id: u32) {}
    fn get(&self, key: u64) -> Option<&[f32]>;
    fn get_node(&self, _handle: NodeHandle, key: u64) -> Option<&[f32]> {
        self.get(key)
    }
    /// The row graph maintenance compares (diversity selection, neighbour
    /// pruning): the exact row, or for a quantized source the vector its
    /// codes approximate (no f32 read: a shed row's f32 is on disk, and
    /// maintenance touches rows at random). Same scale as quantized scores.
    fn get_node_approx(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        self.get_node(handle, key)
    }
    /// A quantized node's stored code row for graph maintenance (`None`
    /// when pruning compares rows instead): its calibration epoch and its
    /// code words, scored pairwise by [`code_scorer`](Self::code_scorer).
    fn code_row(&self, _handle: NodeHandle, _key: u64) -> Option<(u64, &[f32])> { None }
    /// The pairwise distance (lower is closer) of two code rows of `epoch`,
    /// on the stored codes, as Vespa's pruning computes distances on a
    /// tensor's stored cell type; `None` when the codes do not support it.
    fn code_scorer(&self, _epoch: u64) -> Option<Box<dyn Fn(&[f32], &[f32]) -> f64 + '_>> { None }
}

/// See [`MutableVectorSource::dense_layout`].
#[derive(Clone, Copy)]
pub enum DenseLayout<'a> {
    /// Unquantized rows (f32 cells), the source's only copy: traversal
    /// scores them itself.
    Rows(source::RowPlaneRef<'a>),
    /// Quantized codes, scored by the bound query scorer's `score_ids`;
    /// traversal only prefetches them.
    Codes(source::CodePlaneRef<'a>),
}

/// Batches storage maintenance at a caller's commit boundary. This does not
/// delay graph visibility or make the mutations transactionally atomic.
/// Dropping on success, error or unwind performs pending maintenance. Readers
/// and prepared operations retain their normal generation/source guards.
#[must_use]
pub struct MutableCommitBatch {
    control: Arc<Control>,
}
impl Drop for MutableCommitBatch {
    fn drop(&mut self) {
        self.control.end_maintenance_batch();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MutableHnswLimits {
    pub max_slots: u32,
    pub max_pending: usize,
    pub max_retired: usize,
    /// Checked before each serialized mutation; one bounded mutation may overshoot.
    pub max_retired_adjacency_bytes: usize,
}
impl Default for MutableHnswLimits {
    /// Readers never make writers wait or fail: retired nodes and adjacency
    /// held back by old generations (long PITs / scrolls) are not capped by
    /// default; they are reclaimed once those readers close, and their size
    /// is reported (`MutableHnswStats::retired_*`, `_nodes/stats`). Explicit
    /// caps remain available to embedders that prefer rejection.
    fn default() -> Self {
        Self {
            max_slots: u32::MAX - 1,
            max_pending: 65_536,
            max_retired: usize::MAX,
            max_retired_adjacency_bytes: usize::MAX,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct MutableHnswStats {
    pub allocated_slots: u32,
    pub active_nodes: u32,
    pub pending_reservations: usize,
    pub reusable_slots: usize,
    pub retired_slots: usize,
    pub exhausted_slots: usize,
    pub visibility_retained: usize,
    pub issued_key_watermark: Option<u64>,
    pub live_adjacency_bytes: usize,
    pub retired_adjacency_bytes: usize,
    pub neighbor_size: usize,
}
#[derive(Debug, Default)]
pub struct MutableSearchResult {
    pub hits: Vec<(u64, f32)>,
    pub vector_ops: u64,
    /// Final score evaluations, including candidates discarded by top-k.
    pub final_score_ops: u64,
    pub entry_fallback_scanned: u64,
    /// Live graph rows absent from the supplied source view: never count this
    /// result as a successful completeness/recall audit. Counts lookups.
    pub missing_source_lookups: u64,
}

/// Stored adjacency contains nonzero internal node IDs (public slot + 1).
/// Candidate ordering follows pinned Vespa numeric distance-only comparators.
/// Rust heap ordering among equal distances is not a C++ layout guarantee.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Candidate {
    distance: f64,
    id: u32,
    levels: LevelsRef,
}
impl Candidate {
    #[inline]
    fn new(distance: f64, node: NodeView<'_>) -> Self {
        Self {
            distance,
            id: node.id,
            levels: node.levels,
        }
    }
}
impl PartialEq for Candidate {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .expect("candidate distance must be finite")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SavedNeighbor {
    handle: NodeHandle,
    levels: LevelsRef,
}
#[derive(Default)]
struct MutableScratch {
    cand: BinaryHeap<Reverse<Candidate>>,
    top: BinaryHeap<Candidate>,
    out: Vec<Candidate>,
    /// The code score of each `out` candidate, when the beam that filled
    /// `out` scored codes and kept them (`dense_beam`; empty otherwise).
    out_scores: Vec<f32>,
    selected: Vec<Candidate>,
}
thread_local! { static MUTABLE_SCRATCH: RefCell<MutableScratch> = RefCell::new(MutableScratch::default()); }
fn with_mutable_scratch<R>(f: impl FnOnce(&mut MutableScratch) -> R) -> R {
    struct Reset<'a>(&'a mut MutableScratch);
    impl Drop for Reset<'_> {
        fn drop(&mut self) {
            self.0.cand.clear();
            self.0.top.clear();
            self.0.out.clear();
            self.0.out_scores.clear();
            self.0.selected.clear();
        }
    }
    MUTABLE_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        let mut reset = Reset(&mut scratch);
        // Saved refs never survive the lifetime of their query/preparation
        // guard, including unwinding user source/accept callbacks.
        f(&mut reset.0)
    })
}

/// A valid acquired level reference plus its guarded scalar record. Identity
/// loads are lazy: traversal visibility needs publication, filtering/source
/// access needs key/incarnation, and cosine distance alone needs cached norm.
/// Published scalars remain immutable through growth/removal while this same
/// continuously held generation guard prevents source/ID reuse.
#[derive(Clone, Copy)]
struct NodeView<'a> {
    view: &'a MutableHnswSnapshot,
    id: u32,
    levels: LevelsRef,
    record: &'a NodeRecord,
}
impl<'a> NodeView<'a> {
    #[inline]
    fn handle(self) -> NodeHandle {
        NodeHandle::new(self.id - 1, self.record.incarnation_after_acquire())
    }
    #[inline]
    fn key(self) -> u64 {
        self.record.source_key_after_acquire()
    }
    #[inline]
    fn inverse_norm(self) -> f32 {
        self.record.norm_after_acquire()
    }
    #[inline]
    fn level(self) -> usize {
        if let Some(nodes) = &self.view.captured {
            return nodes[self.id as usize].as_ref().expect("captured node").links.len() - 1;
        }
        self.level_refs().len() - 1
    }
    #[inline]
    fn level_refs(self) -> &'a [AtomicU32] {
        // SAFETY: NodeView only originates from current acquire or a candidate
        // captured under this same continuously held domain guard.
        unsafe {
            self.view
                .control
                .levels
                .view_prevalidated(&self.view.guard)
                .get(self.levels)
                .expect("valid acquired levels")
        }
    }
    #[inline]
    fn links(self, level: usize) -> Option<&'a [u32]> {
        if self.view.captured.is_some() { self.links_typed::<true>(level) }
        else { self.links_typed::<false>(level) }
    }
    #[inline]
    fn links_typed<const CAPTURED: bool>(self, level: usize) -> Option<&'a [u32]> {
        let reference = if CAPTURED {
            let nodes = self.view.captured.as_ref().expect("captured traversal");
            *nodes[self.id as usize].as_ref()?.links.get(level)?
        } else { LinksRef::from_raw(self.level_refs().get(level)?.load(Acquire)) };
        // SAFETY: link reference acquired transitively from protected levels.
        Some(unsafe {
            self.view
                .control
                .links
                .view_prevalidated(&self.view.guard)
                .get(reference)
                .expect("valid acquired links")
        })
    }
}
fn internal_id(handle: NodeHandle) -> u32 {
    handle.slot() + 1
}
fn packed_entry(id: u32, level: usize) -> u64 {
    (level as u64) << 32 | u64::from(id)
}
pub struct MutableHnsw {
    dim: u32,
    similarity: Similarity,
    params: HnswParams,
    level_generator: level_generator::VespaLevelGenerator,
    min_size_before_two_phase: u32,
    control: Arc<Control>,
}
/// Transferable, generation-pinned current-topology view. Holding this view delays
/// slot/node reclamation. It bounds later publications but is NOT an ES PIT.
struct CapturedNode {
    levels: LevelsRef,
    links: crate::admission::Allocation<Vec<LinksRef>>,
}
/// A capture begun at its capture point and not copied yet
/// (`MutableHnsw::begin_capture`); dropping it retires its pre-image log.
struct PendingCapture {
    view: Option<MutableHnswSnapshot>,
    logged: bool,
}
impl Drop for PendingCapture {
    fn drop(&mut self) {
        if self.logged {
            if let Some(view) = &self.view {
                view.control.writer.lock().capture_log = None;
            }
        }
    }
}
/// Nodes a chunked capture copies per topology-writer hold.
const CAPTURE_CHUNK_NODES: u32 = 4096;
pub struct MutableHnswSnapshot {
    captured: Option<crate::admission::Allocation<Vec<Option<CapturedNode>>>>,
    /// The topology sequence when this live view was taken with no
    /// publication in progress and every id below its bound a published
    /// node (`None` otherwise): while the sequence still reads this value,
    /// every node a link names is visible to this view (`dense_beam`).
    quiet: Option<u64>,
    guard: GenerationGuard,
    control: Arc<Control>,
    len: u32,
    entry: u64,
    entry_levels: LevelsRef,
    entry_identity: Option<NodeIdentity>,
    publication: u64,
    dim: u32,
    similarity: Similarity,
    m: usize,
}
pub struct PreparedMutableInsert {
    reservation: InsertReservation,
    inverse_norm: f32,
    preparation: PreparedConnections,
    /// Reverse-link pruning computed off the topology writer (see
    /// [`MutableHnsw::prepare_pruning`]); completion applies an entry only
    /// if the neighbour's list is still the one it was computed from.
    pruned: Vec<PrunedLinks>,
    _generation: GenerationGuard,
}
/// One neighbour's pruned link list at one level, computed in preparation:
/// `observed` is the list the new node's reverse link overflows (the
/// neighbour's links then, with the new node appended), `handles` their
/// owners then, `kept` what pruning keeps of it.
struct PrunedLinks {
    node: NodeHandle,
    level: usize,
    observed: Vec<u32>,
    handles: Vec<NodeHandle>,
    kept: Vec<u32>,
}
enum PreparedConnections {
    /// Captured before reading a level or traversing the graph. Completion
    /// performs preparation under the topology writer even if the graph has
    /// since grown beyond the threshold.
    Bootstrap,
    Neighbors(Vec<Vec<SavedNeighbor>>),
}
impl PreparedMutableInsert {
    pub fn key(&self) -> u64 {
        self.reservation.key
    }
    pub fn handle(&self) -> NodeHandle {
        self.reservation.handle
    }
    pub fn is_bootstrap(&self) -> bool {
        matches!(self.preparation, PreparedConnections::Bootstrap)
    }
    /// Owned proposal allocation estimate; excludes caller-owned vector rows.
    pub fn estimated_bytes(&self) -> usize {
        let extra = match &self.preparation {
            PreparedConnections::Bootstrap => 0,
            PreparedConnections::Neighbors(connections) => {
                connections.capacity() * std::mem::size_of::<Vec<SavedNeighbor>>()
                    + connections
                        .iter()
                        .map(|v| v.capacity() * std::mem::size_of::<SavedNeighbor>())
                        .sum::<usize>()
            }
        };
        std::mem::size_of::<Self>() + extra
    }
}

impl MutableHnsw {
    pub fn new(dim: u32, similarity: Similarity, params: HnswParams) -> Self {
        Self::with_limits(dim, similarity, params, MutableHnswLimits::default())
    }
    pub fn with_limits(dim: u32, similarity: Similarity, params: HnswParams, limits: MutableHnswLimits) -> Self {
        Self::try_with_limits(dim, similarity, params, limits)
            .expect("mutable graph construction failed")
    }
    /// Production creation and recovery propagate admission refusal before any
    /// graph becomes visible; the convenience constructors remain for callers
    /// explicitly choosing an infallible construction contract.
    pub fn try_new(dim: u32, similarity: Similarity, params: HnswParams) -> Result<Self> {
        Self::try_with_limits(dim, similarity, params, MutableHnswLimits::default())
    }
    pub fn try_with_limits(
        dim: u32,
        similarity: Similarity,
        mut params: HnswParams,
        limits: MutableHnswLimits,
    ) -> Result<Self> {
        if dim == 0 || limits.max_slots == u32::MAX || limits.max_slots == 0
            || limits.max_pending == 0 || limits.max_retired == 0
            || limits.max_retired_adjacency_bytes == 0 {
            return Err(failure("invalid mutable storage limits"));
        }
        params.m = params.m.max(1);
        if params.m > u32::MAX as usize / 2 {
            return Err(failure("invalid mutable fanout"));
        }
        params.ef_construction = params.ef_construction.max(1);
        Ok(Self {
            dim,
            similarity,
            level_generator: level_generator::VespaLevelGenerator::new(params.m as u32)
                .expect("validated mutable fanout"),
            min_size_before_two_phase: MIN_SIZE_BEFORE_TWO_PHASE,
            params,
            control: Arc::new(Control::new(limits)?),
        })
    }
    pub fn dim(&self) -> u32 {
        self.dim
    }
    pub fn similarity(&self) -> Similarity {
        self.similarity
    }
    pub fn params(&self) -> HnswParams {
        self.params
    }
    /// Peak allocated public slots, distinct from live traversal bound.
    pub fn num_nodes(&self) -> u32 {
        self.control.allocated.load(Acquire)
    }
    pub fn active_nodes(&self) -> u32 {
        self.control.active.load(Acquire)
    }
    /// Changes with every published topology change (insert, remove, link
    /// update, compaction): equal values mean an unchanged graph, so a
    /// checkpoint taken at one is still exact at the other.
    pub fn mutation_generation(&self) -> u64 {
        self.control.mutation_generation()
    }
    pub fn stats(&self) -> MutableHnswStats {
        self.control.stats()
    }
    pub fn storage_memory(&self) -> MutableStorageMemory {
        self.control.memory_usage()
    }
    /// Match attribute-style commit maintenance while preserving immediate
    /// publication. Keep batches bounded; pressure still triggers maintenance
    /// at write admission and pinned-reader budgets still reject excess work.
    pub fn commit_batch(&self) -> Result<MutableCommitBatch> {
        self.control.begin_maintenance_batch()?;
        Ok(MutableCommitBatch { control: self.control.clone() })
    }
    /// Cheap native drain predicate; detailed storage accounting is separate.
    pub fn reclamation_pending(&self) -> bool {
        self.control.reclamation_pending()
    }
    pub fn try_snapshot(&self) -> Result<MutableHnswSnapshot> {
        self.try_snapshot_with_hook(|_| {})
    }
    fn try_snapshot_with_hook(
        &self,
        mut after_entry: impl FnMut(u64),
    ) -> Result<MutableHnswSnapshot> {
        let guard = self
            .control
            .domain
            .take_guard()
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.control.check_readable()?;
        // Validate storage-domain ownership once at snapshot admission. Every
        // NodeView is derived from this snapshot and borrows its same guard.
        self.control.levels.view(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.control.links.view(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.control.nodes.read(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let seq = self.control.topology_seq();
        let (entry, entry_levels, entry_identity) = loop {
            let packed = self.control.entry.load(Acquire);
            after_entry(packed);
            if packed == NO_ENTRY {
                break (packed, LevelsRef::INVALID, None);
            }
            let row = self
                .control
                .nodes
                .read(&guard)
                .map_err(crate::vector::hnsw::mutable::failure_error)?
                .get(packed as u32);
            if let Some(row) = row {
                let levels = row.levels_ref();
                if levels.is_valid() {
                    let identity = row.identity_after_acquire();
                    if self.control.entry.load(Acquire) == packed {
                        break (packed, levels, Some(identity));
                    }
                }
            }
            self.control.check_readable()?;
            // The writer releases replacement entry before invalidating an old
            // node. Retry ID/level + acquired ref as one coherent tuple.
        };
        let publication = self.control.publication.load(Acquire);
        let live_bound = self.control.live_bound.load(Acquire);
        let active = self.control.active.load(Acquire);
        // Quiet: no publication stored while this view was taken, and every
        // id below the searchable bound a published node; links then name
        // only nodes this view sees (removal leaves in-links of other nodes
        // naming the removed slot, which is then not active).
        let quiet = (seq % 2 == 0 && active.checked_add(1) == Some(live_bound)
            && self.control.topology_seq_after_fence() == seq).then_some(seq);
        Ok(MutableHnswSnapshot {
            captured: None,
            quiet,
            guard,
            control: self.control.clone(),
            len: live_bound.max((entry as u32).saturating_add(1)),
            entry,
            entry_levels,
            entry_identity,
            publication,
            dim: self.dim,
            similarity: self.similarity,
            m: self.params.m,
        })
    }
    /// Saver-style retained topology: every link reference as of one point
    /// in time, holding payloads through the existing generation guard.
    /// O(nodes + levels) metadata is copied, in chunks of
    /// [`CAPTURE_CHUNK_NODES`] with the topology writer released between
    /// them: the capture installs a pre-image log at its start and every
    /// topology publication records a not-yet-copied node's levels/links
    /// before changing them (`publish_overlay`), so the copy stays exactly
    /// the graph at the start while graph writers proceed. Payloads and
    /// graph algorithms are unchanged. Ordinary queries use try_snapshot.
    pub fn capture_snapshot(&self) -> Result<MutableHnswSnapshot> {
        Ok(self.capture_snapshot_with(|_, _| Ok(()))?.0)
    }

    /// [`Self::capture_snapshot`], running `at_start` under the topology
    /// writer at the capture point (callers record writer state that must
    /// match the captured graph exactly).
    fn capture_snapshot_with<R>(&self, at_start: impl FnOnce(&mut Writer, &MutableHnswSnapshot) -> Result<R>)
        -> Result<(MutableHnswSnapshot, R)>
    {
        let (pending, start) = self.begin_capture(at_start)?;
        Ok((self.finish_capture(pending)?, start))
    }

    /// The capture point of [`Self::capture_snapshot_with`] (one short
    /// topology-writer hold); [`Self::finish_capture`] copies later, while
    /// graph writers proceed. Dropping the pending capture abandons it.
    fn begin_capture<R>(&self, at_start: impl FnOnce(&mut Writer, &MutableHnswSnapshot) -> Result<R>)
        -> Result<(PendingCapture, R)>
    {
        let mut writer = self.control.writer.lock();
        let view = self.try_snapshot()?;
        let start = at_start(&mut *writer, &view)?;
        if writer.capture_log.is_some() {
            // Another capture owns the log: copy under the writer now.
            let view = self.capture_snapshot_locked_from(view)?;
            return Ok((PendingCapture { view: Some(view), logged: false }, start));
        }
        writer.capture_log = Some(CaptureLog { cursor: 0, len: view.len, images: std::collections::HashMap::new(), memory:None });
        Ok((PendingCapture { view: Some(view), logged: true }, start))
    }

    /// Copy a capture begun with [`Self::begin_capture`] (chunked).
    fn finish_capture(&self, mut pending: PendingCapture) -> Result<MutableHnswSnapshot> {
        let mut view = pending.view.take().ok_or_else(|| failure("capture already finished"))?;
        if !pending.logged { return Ok(view); }
        pending.logged = false;
        let copied = self.capture_chunks(&view);
        // Always retire the log, whatever the outcome.
        self.control.writer.lock().capture_log = None;
        view.captured = Some(copied?);
        view.quiet = None;
        Ok(view)
    }

    fn capture_chunks(&self, view: &MutableHnswSnapshot) -> Result<crate::admission::Allocation<Vec<Option<CapturedNode>>>> {
        let memory=Self::capture_nodes_memory(view.len)?;
        let mut captured = Vec::new();
        captured.try_reserve_exact(view.len as usize).map_err(|_| failure("capture node allocation"))?;
        let mut id = 0u32;
        while id < view.len {
            let end = id.saturating_add(CAPTURE_CHUNK_NODES).min(view.len);
            let mut writer = self.control.writer.lock();
            let log = writer.capture_log.as_mut().ok_or_else(|| failure("capture log lost"))?;
            for slot in id..end {
                let value = match log.images.remove(&slot) {
                    // Changed since the capture point: its pre-image.
                    Some((levels, links)) => {
                        // SAFETY: snapshot admission validated this table/guard pair.
                        let visible = unsafe { self.control.nodes.read_prevalidated(&view.guard) }.get(slot)
                            .is_some_and(|row| row.identity_record().publication_after_acquire() <= view.publication);
                        (visible && levels.is_valid()).then_some(CapturedNode { levels, links })
                    }
                    None => Self::capture_node(view, slot)?,
                };
                captured.push(value);
            }
            log.cursor = end;
            drop(writer);
            id = end;
        }
        Ok(crate::admission::Allocation::new(captured,memory))
    }

    fn capture_nodes_memory(len:u32)->Result<crate::admission::Reservation> {
        let bytes=(len as usize).checked_mul(std::mem::size_of::<Option<CapturedNode>>()).ok_or_else(||failure("capture node bytes"))?;
        crate::admission::shared().reserve(bytes as u64,"captured graph nodes")
    }
    fn capture_links(refs:&[std::sync::atomic::AtomicU32])->Result<crate::admission::Allocation<Vec<LinksRef>>> {
        let bytes=refs.len().checked_mul(std::mem::size_of::<LinksRef>()).ok_or_else(||failure("capture link bytes"))?;
        let memory=crate::admission::shared().reserve(bytes as u64,"captured graph level references")?;
        let mut links=Vec::new();links.try_reserve_exact(refs.len()).map_err(|_|failure("capture links allocation"))?;
        links.extend(refs.iter().map(|r|LinksRef::from_raw(r.load(Acquire))));
        Ok(crate::admission::Allocation::new(links,memory))
    }

    /// The node at internal `id` as `view` (an uncaptured snapshot) sees it
    /// now, if visible to it.
    fn capture_node(view: &MutableHnswSnapshot, id: u32) -> Result<Option<CapturedNode>> {
        let Some(node) = view.current_node(id).filter(|node| view.visible(*node)) else { return Ok(None) };
        Ok(Some(CapturedNode {levels:node.levels,links:Self::capture_links(node.level_refs())?}))
    }

    // Caller holds the topology writer fence.
    fn capture_snapshot_locked_from(&self, mut view: MutableHnswSnapshot) -> Result<MutableHnswSnapshot> {
        let memory=Self::capture_nodes_memory(view.len)?;
        let mut captured = Vec::new();
        captured.try_reserve_exact(view.len as usize).map_err(|_| failure("capture node allocation"))?;
        for id in 0..view.len {
            captured.push(Self::capture_node(&view, id)?);
        }
        view.captured = Some(crate::admission::Allocation::new(captured,memory));
        view.quiet = None;
        Ok(view)
    }

    /// Compatibility convenience: admission failure fails loudly; it must never
    /// become a successful empty search. Fallible integrations use try_snapshot.
    pub fn snapshot(&self) -> MutableHnswSnapshot {
        self.try_snapshot()
            .expect("mutable HNSW snapshot admission failed")
    }
    pub fn owns_reclaimed(&self, token: &ReclaimedNode) -> bool {
        self.control.owns_reclaimed(token)
    }
    pub fn collect_reclaimed(&self) -> Vec<ReclaimedNode> {
        self.control.collect_reclaimed()
    }
    pub fn reserve_insert(&self, key: u64) -> Result<InsertReservation> {
        self.control.reserve_insert(key)
    }
    /// [`Self::reserve_insert`] of a key the graph does not hold, at or
    /// below the highest key issued (see `GraphStorage::reserve_insert_backfill`).
    pub(crate) fn reserve_insert_backfill(&self, key: u64) -> Result<InsertReservation> {
        self.control.reserve_insert_backfill(key)
    }
    pub fn prepare(
        &self,
        reservation: InsertReservation,
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<PreparedMutableInsert> {
        if !Arc::ptr_eq(&self.control, &reservation.control) {
            return Err(failure("reservation from another graph"));
        }
        let bootstrap = self.active_nodes() < self.min_size_before_two_phase;
        // Shared-domain guard precedes candidate acquisition and moves with the
        // prepared operation, retaining stores through the reservation Arc.
        let generation = self
            .control
            .domain
            .take_guard()
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let row = checked_row(source, reservation.handle, reservation.key, self.dim, self.similarity)?;
        let inverse_norm = if self.similarity == Similarity::Cosine {
            let n = norm(row);
            if n > 0.0 { 1.0 / n } else { 0.0 }
        } else {
            0.0
        };
        // Pinned prepare_add_document samples active size ONCE before choosing
        // PreparedFirstAddDoc versus two-phase preparation. A bootstrap marker
        // does no RNG draw or graph traversal here. Source validation and its
        // lifetime lease remain explicit obligations of this native adapter.
        let (preparation, pruned) = if bootstrap {
            (PreparedConnections::Bootstrap, Vec::new())
        } else {
            let connections = self.prepare_connections(row, (reservation.handle, reservation.key), source)?;
            let pruned = self.prepare_pruning(&connections, (reservation.handle, reservation.key), source)?;
            (PreparedConnections::Neighbors(connections), pruned)
        };
        Ok(PreparedMutableInsert {
            reservation,
            inverse_norm,
            preparation,
            pruned,
            _generation: generation,
        })
    }
    /// The reverse-link pruning completion would do for `connections` (the
    /// neighbours whose list the new node's link overflows), computed here,
    /// in parallel preparation, from a current snapshot instead of under the
    /// topology writer. Vespa's `complete_add_document` shrinks those lists
    /// in the write thread; this computes the same selection ahead of it,
    /// and completion uses it only for a list that has not changed since
    /// (otherwise it shrinks as Vespa does), so the graph is the same. A
    /// neighbour whose pruning cannot be computed here is left to completion.
    fn prepare_pruning(
        &self,
        connections: &[Vec<SavedNeighbor>],
        node: (NodeHandle, u64),
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<Vec<PrunedLinks>> {
        let view = self.try_snapshot()?;
        let level = connections.len().saturating_sub(1);
        let overlay = AdjacencyOverlay::new(&view, source, self.similarity, self.params.m, Some((node.0, node.1, level)));
        let id = internal_id(node.0);
        let mut pruned = Vec::new();
        for (l, selected) in connections.iter().enumerate() {
            let capacity = if l == 0 { 2 * self.params.m } else { self.params.m };
            for neighbor in selected {
                let Some(current) = view.node(neighbor.handle).filter(|n| n.levels == neighbor.levels) else { continue };
                let Some(links) = current.links(l) else { continue };
                if links.len() + usize::from(!links.contains(&id)) <= capacity { continue; }
                let mut observed = links.to_vec();
                if !observed.contains(&id) { observed.push(id); }
                let Ok(handles) = observed.iter().map(|&other| overlay.handle(other)).collect::<Result<Vec<_>>>() else { continue };
                // A failed row read is consumed here (`row` reports it) and
                // leaves this list to completion.
                let Ok(kept) = overlay.select_kept(neighbor.handle, &observed, capacity) else { continue };
                pruned.push(PrunedLinks { node: neighbor.handle, level: l, observed, handles, kept });
            }
        }
        Ok(pruned)
    }
    fn prepare_connections(
        &self,
        row: &[f32],
        node: (NodeHandle, u64),
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<Vec<Vec<SavedNeighbor>>> {
        let level = self
            .level_generator
            .next_level()
            .map_err(crate::vector::hnsw::mutable::failure_error)? as usize;
        self.prepare_connections_at_level(row, node, source, level)
    }
    /// `node`: the inserted node's handle and key (its row is installed).
    fn prepare_connections_at_level(
        &self,
        row: &[f32],
        node: (NodeHandle, u64),
        source: &(impl MutableVectorSource + ?Sized),
        level: usize,
    ) -> Result<Vec<Vec<SavedNeighbor>>> {
        let view = self.try_snapshot()?;
        // Quantized fields search for the new node's neighbours with the
        // field's quantized scorer (as Lucene builds a scalar-quantized HNSW
        // graph from its quantized vectors): it reads codes, not f32 rows.
        // Distances are on the same scale as the f32 ones that diversity
        // selection compares them with. f32 fields bind nothing.
        let distance = SourceDistance::for_query(&view, source, row);
        // Diversity selection compares the candidates with each other on
        // their approximate rows (decoded codes for a quantized field). The
        // candidate's distance to the new node it is weighed against must
        // be on that same scale: the new node's approximate row, not the
        // search's code estimate. With 1-bit codes the two scales differ
        // enough that almost every candidate after the first was rejected,
        // leaving nodes with one or two links and, after reverse-link
        // pruning, none (unreachable from the entry point).
        let anchor = if distance.encoded.is_some() { source.get_node_approx(node.0, node.1) } else { None };
        let mut connections = vec![Vec::new(); level + 1];
        if let Some(current) = view.available_entry(&distance) {
            let current_distance = distance.dist(current);
            let top = view.entry_level(current);
            with_mutable_scratch(|scratch| {
                let mut acquired = Candidate::new(
                    current_distance,
                    view.acquired_node(current).expect("captured entry"),
                );
                for l in ((level + 1)..=top).rev() {
                    acquired = view.greedy_acquired(&distance, acquired, l);
                }
                scratch.top.clear();
                scratch.top.push(acquired);
                // Vespa internal_prepare_add_node keeps best_neighbors intact
                // between levels. Diversity selection only chooses the edges
                // to publish; it must not discard the other lower-level seeds.
                for l in (0..=level.min(top)).rev() {
                    view.construction_beam(&distance, l, self.params.ef_construction, scratch);
                    select_diverse(
                        &view,
                        source,
                        self.similarity,
                        &scratch.out,
                        anchor,
                        self.params.m,
                        &mut scratch.selected,
                    );
                    connections[l].extend(
                        scratch
                            .selected
                            .iter()
                            .filter(|candidate| view.candidate_node(**candidate).level() >= l)
                            .map(|candidate| SavedNeighbor {
                                handle: view.candidate_node(*candidate).handle(),
                                levels: candidate.levels,
                            }),
                    );
                }
            });
        }
        source.execution_result()?;
        if distance.missing_source.get() != 0 {
            return Err(failure("incomplete preparation source view"));
        }
        Ok(connections)
    }
    /// Convert a GPU batch's directed neighbors into ordinary mutable insertion
    /// proposals. Completion still adds reciprocal edges, prunes with the same
    /// diversity rule, and publishes through the generation-fenced writer.
    /// Batch nodes must be completed from highest level to lowest. Missing
    /// earlier neighbors are bridged to the existing shard graph on the CPU.
    #[cfg(feature = "gpu")]
    pub(crate) fn prepare_gpu(
        &self,
        reservation: InsertReservation,
        source: &(impl MutableVectorSource + ?Sized),
        gpu: &HnswGraph,
        ordinal: u32,
        handles: &[NodeHandle],
    ) -> Result<PreparedMutableInsert> {
        if !Arc::ptr_eq(&self.control, &reservation.control) {
            return Err(failure("GPU reservation from another graph"));
        }
        if handles.get(ordinal as usize) != Some(&reservation.handle) {
            return Err(failure("GPU ordinal/reservation mismatch"));
        }
        let level = gpu.layers.iter().rposition(|layer| layer.row(ordinal).is_some())
            .ok_or_else(|| failure("GPU graph missing reserved row"))?;
        let generation = self.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let row = checked_row(source, reservation.handle, reservation.key, self.dim, self.similarity)?;
        let inverse_norm = if self.similarity == Similarity::Cosine {
            let n = norm(row); if n > 0.0 { 1.0 / n } else { 0.0 }
        } else { 0.0 };
        let view = self.try_snapshot()?;
        let distance = BoundMutationDistance::new(row, self.similarity);
        let mut connections = vec![Vec::new(); level + 1];
        for (l, selected) in connections.iter_mut().enumerate() {
            let mut candidates = Vec::new();
            for &neighbor in gpu.layers[l].neighbors_of(ordinal) {
                let handle = *handles.get(neighbor as usize)
                    .ok_or_else(|| failure("GPU neighbor outside reserved batch"))?;
                let Some(node) = view.node(handle).filter(|node| view.visible(*node) && node.level() >= l) else { continue };
                let other = checked_row(source, handle, node.key(), self.dim, self.similarity)?;
                candidates.push(Candidate::new(distance.distance(other)?, node));
            }
            candidates.sort_unstable();
            let mut diverse = Vec::new();
            select_diverse(&view, source, self.similarity, &candidates, None, self.params.m, &mut diverse);
            selected.extend(diverse.into_iter().map(|candidate| SavedNeighbor {
                handle: view.candidate_node(candidate).handle(), levels: candidate.levels,
            }));
        }
        // At least the first/top node in each GPU component needs a bridge to
        // the existing shard graph. Never publish a disconnected batch merely
        // because its nearest neighbors have not been completed yet.
        if connections.iter().any(Vec::is_empty) {
            let bridges = self.prepare_connections_at_level(row, (reservation.handle, reservation.key), source, level)?;
            for (selected, bridge) in connections.iter_mut().zip(bridges) {
                if selected.is_empty() { *selected = bridge; }
            }
        }
        Ok(PreparedMutableInsert { reservation, inverse_norm,
            preparation: PreparedConnections::Neighbors(connections), pruned: Vec::new(), _generation: generation })
    }

    fn check_retired_adjacency_budget(&self) -> Result<()> {
        self.control.check_retired_adjacency_budget()
    }
    pub fn complete(
        &self,
        mut prepared: PreparedMutableInsert,
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<NodeHandle> {
        if !Arc::ptr_eq(&self.control, &prepared.reservation.control) {
            return Err(failure("proposal from another graph"));
        }
        let diagnostic = crate::vector::mutable_ingest::diagnostics::writer_start();
        let mut writer = self.control.writer.lock();
        let acquired = diagnostic.map(|_| std::time::Instant::now());
        // `writer` is released before `prepared` drops (an uncompleted
        // reservation's drop takes the writer).
        let id = self.complete_locked(&mut prepared, source, &mut writer)?;
        let finished = diagnostic.map(|_| std::time::Instant::now());
        drop(writer);
        if let (Some(start), Some(acquired), Some(finished)) = (diagnostic, acquired, finished) {
            crate::vector::mutable_ingest::diagnostics::writer_finish(start, acquired, finished);
        }
        Ok(id)
    }

    /// [`Self::complete`] of `proposals` in order under ONE topology-writer
    /// hold, each published as it completes (searches never take the
    /// writer: they read generation-pinned snapshots). Completing each
    /// proposal under its own hold made every concurrent bulk of a shard
    /// hand the writer over per insert (a futex park and wake each time),
    /// so completion, not preparation, paced a shard's ingest. The hold
    /// lasts one staging slice (`STAGE_SLICE_DOCS` inserts, each a bounded
    /// reverse-link update), shorter than a shedding round's.
    ///
    /// Each completed proposal's handle is pushed to `done` (the caller
    /// owns rollback of published nodes on an error). Uncompleted
    /// proposals stay with the caller, which drops them after this
    /// returns, with the writer released.
    pub fn complete_batch(
        &self,
        proposals: &mut [PreparedMutableInsert],
        source: &(impl MutableVectorSource + ?Sized),
        done: &mut Vec<NodeHandle>,
    ) -> Result<()> {
        if proposals.iter().any(|p| !Arc::ptr_eq(&self.control, &p.reservation.control)) {
            return Err(failure("proposal from another graph"));
        }
        let diagnostic = crate::vector::mutable_ingest::diagnostics::writer_start();
        let mut writer = self.control.writer.lock();
        let acquired = diagnostic.map(|_| std::time::Instant::now());
        for prepared in proposals.iter_mut() {
            done.push(self.complete_locked(prepared, source, &mut writer)?);
        }
        let finished = diagnostic.map(|_| std::time::Instant::now());
        drop(writer);
        if let (Some(start), Some(acquired), Some(finished)) = (diagnostic, acquired, finished) {
            crate::vector::mutable_ingest::diagnostics::writer_finish(start, acquired, finished);
        }
        Ok(())
    }

    /// One completion under the held topology writer.
    fn complete_locked(
        &self,
        prepared: &mut PreparedMutableInsert,
        source: &(impl MutableVectorSource + ?Sized),
        writer: &mut Writer,
    ) -> Result<NodeHandle> {
        let id = prepared.handle();
        let key = prepared.key();
        let row = checked_row(source, id, key, self.dim, self.similarity)?;
        self.control.admit_write(writer)?;
        if writer.pending.get(&key) != Some(&id) {
            return Err(failure("reservation no longer pending"));
        }
        self.check_retired_adjacency_budget()?;
        let mut connections = match &mut prepared.preparation {
            PreparedConnections::Bootstrap => self.prepare_connections(row, (id, key), source)?,
            PreparedConnections::Neighbors(c) => std::mem::take(c),
        };
        let view = self.try_snapshot()?;
        self.filter_proposal(&view, source, &mut connections)?;
        let publication = self
            .control
            .publication
            .load(Relaxed)
            .checked_add(1)
            .ok_or_else(|| failure("publication sequence exhausted"))?;
        let level = connections.len() - 1;
        let mut overlay = AdjacencyOverlay::new(
            &view,
            source,
            self.similarity,
            self.params.m,
            Some((id, key, level)),
        );
        for (l, selected) in connections.iter().enumerate() {
            overlay.set(
                id,
                l,
                selected.iter().map(|n| internal_id(n.handle)).collect(),
            );
            for n in selected {
                overlay.add(n.handle, id, l)?;
            }
            for n in selected {
                let ahead = prepared.pruned.iter().find(|p| p.level == l && p.node == n.handle);
                if let Some(ahead) = ahead {
                    if overlay.apply_pruned(ahead)? { continue; }
                }
                overlay.shrink(n.handle, l)?;
            }
        }
        source.execution_result()?;
        self.control
            .nodes
            .set_reserved_norm(internal_id(id), prepared.inverse_norm)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.publish_overlay(
            &view,
            overlay,
            Publication::Insert {
                id,
                level,
                publication,
            },
            writer,
        )?;
        writer.pending.remove(&key);
        prepared.reservation.pending = false;
        INSERTS.with(|c| c.set(c.get().wrapping_add(1)));
        drop(view);
        self.control.finish_mutation(writer);
        Ok(id)
    }
    pub fn insert(
        &self,
        key: u64,
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<NodeHandle> {
        self.complete(self.prepare(self.reserve_insert(key)?, source)?, source)
    }
    fn filter_proposal(
        &self,
        view: &MutableHnswSnapshot,
        source: &(impl MutableVectorSource + ?Sized),
        connections: &mut [Vec<SavedNeighbor>],
    ) -> Result<()> {
        for (level, ids) in connections.iter_mut().enumerate() {
            let mut incomplete = false;
            ids.retain(|saved| {
                view.node(saved.handle).is_some_and(|node| {
                    if node.levels != saved.levels || !view.visible(node) || node.level() < level {
                        return false;
                    }
                    let valid = source
                        .get_node(saved.handle, node.key())
                        .is_some_and(|r| r.len() == self.dim as usize);
                    incomplete |= !valid;
                    valid
                })
            });
            if incomplete {
                return Err(failure("incomplete completion source view"));
            }
            // Pinned hnsw_index.cpp:752-768 filters in proposal order. The
            // visited/selection path already produces distinct node IDs; an
            // extra ID sort/dedup here changes reciprocal insertion/pruning.
            // retain above preserves survivor order and the saved-ref checks.
        }
        Ok(())
    }
    /// Schedule a physical removal for the first oldest-visible generation at
    /// or after retire_at. Caller supplies the independent visibility clock.
    pub fn retire_after(&self, id: NodeHandle, retire_at: u64) -> bool {
        let mut writer = self.control.writer.lock();
        if !self.snapshot().live(id) {
            return false;
        }
        writer
            .visibility_retained
            .entry(id)
            .and_modify(|g| *g = (*g).max(retire_at))
            .or_insert(retire_at);
        true
    }
    pub fn reclaim_through(
        &self,
        oldest_visible_generation: u64,
        max_nodes: usize,
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<usize> {
        let ids: Vec<_> = self
            .control
            .writer
            .lock()
            .visibility_retained
            .iter()
            .filter(|(_, generation)| **generation <= oldest_visible_generation)
            .map(|(&id, _)| id)
            .take(max_nodes)
            .collect();
        let mut removed = 0;
        for id in ids {
            let mut writer = self.control.writer.lock();
            if writer
                .visibility_retained
                .get(&id)
                .is_some_and(|g| *g <= oldest_visible_generation)
                && self.remove_locked(id, source, &mut writer)?
            {
                removed += 1;
            }
        }
        Ok(removed)
    }
    /// Immediate physical removal requires caller proof no old visibility lease
    /// needs this row. Concurrent live readers keep traversing through the
    /// removed node (`removed_bridge`) but never return it; use
    /// retire_after/reclaim_through for visibility-aware scheduling.
    pub fn remove(
        &self,
        id: NodeHandle,
        source: &(impl MutableVectorSource + ?Sized),
    ) -> Result<bool> {
        self.remove_locked(id, source, &mut self.control.writer.lock())
    }
    fn remove_locked(
        &self,
        id: NodeHandle,
        source: &(impl MutableVectorSource + ?Sized),
        writer: &mut Writer,
    ) -> Result<bool> {
        self.control.admit_write(writer)?;
        let view = self.try_snapshot()?;
        let Some(node) = view.node(id).filter(|n| view.visible(*n)) else {
            return Ok(false);
        };
        self.check_retired_adjacency_budget()?;
        self.control.reserve_retirement(writer)?;
        let mut overlay =
            AdjacencyOverlay::new(&view, source, self.similarity, self.params.m, None);
        let mut replacement = None;
        for level in (0..=node.level()).rev() {
            let cluster: Vec<_> = node
                .links(level)
                .unwrap()
                .iter()
                .filter_map(|&n| view.current_handle(n))
                .filter(|&h| view.live(h))
                .collect();
            for &n in &cluster {
                overlay.row(n)?;
            }
            if replacement.is_none() {
                replacement = cluster.first().map(|&h| (h, level));
            }
            for &neighbor in &cluster {
                overlay.remove(neighbor, id, level)?;
            }
            overlay.reconnect(&cluster, level)?;
        }
        self.publish_overlay(
            &view,
            overlay,
            Publication::Remove {
                id,
                key: node.key(),
                replacement,
            },
            writer,
        )?;
        drop(view);
        self.control.finish_mutation(writer);
        Ok(true)
    }
    /// Convenience for an immutable or fully covering source. For a growing
    /// source, capture snapshot() first, then its source root, and search that
    /// snapshot. Nonzero missing_source_lookups invalidates a completeness audit.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
    ) -> MutableSearchResult {
        self.snapshot().search(query, k, ef, source, accept)
    }
    pub fn search_with_handles(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
    ) -> MutableSearchResult {
        self.snapshot()
            .search_with_handles(query, k, ef, source, accept)
    }
    /// Plain traversal with precomputed statistics for the actual document
    /// filter. Statistics select visited storage only; they do not alter ANN
    /// cutoffs or replace the authoritative accept predicate. Capture a graph
    /// snapshot before a growing source root when using a concurrent source.
    pub fn search_with_filter_estimate(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
        estimate: VespaFilterEstimate,
    ) -> Result<MutableSearchResult> {
        self.try_snapshot()?
            .search_with_filter_estimate(query, k, ef, source, accept, estimate)
    }
    /// RMHG4 logical checkpoint: internal node IDs and a coherent entry ID/level,
    /// never arena references. RMHG3 is explicitly migrated from public slot
    /// edges and full-handle entry; reopening resets the fixed level stream.
    pub fn encode(&self) -> Vec<u8> {
        let writer = self.control.writer.lock();
        let view = self.snapshot();
        self.encode_view_locked(&writer, &view)
    }
    // Caller holds the topology writer and a successfully admitted read view.
    // Reuse that view when pairing graph/source; do not acquire an extra guard
    // through the infallible compatibility snapshot API.
    fn encode_view_locked(&self, writer: &Writer, view: &MutableHnswSnapshot) -> Vec<u8> {
        self.encode_retained_view(view, self.num_nodes(), writer.watermark, &writer.visibility_retained, None)
    }
    // A captured view owns immutable level/link references. Saved incarnations
    // also freeze empty slots, which may be allocated after the capture.
    fn encode_retained_view(&self, view: &MutableHnswSnapshot, allocated: u32,
        watermark: Option<u64>, retirements: &std::collections::HashMap<NodeHandle, u64>, incarnations: Option<&[u32]>) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.encode_retained_into(&mut bytes,view,allocated,watermark,retirements,incarnations);
        bytes
    }
    /// Bytes [`Self::encode_retained_into`] writes for `view`.
    fn retained_len(&self,view:&MutableHnswSnapshot,allocated:u32)->Result<usize> {
        let mut length=68usize.checked_add((allocated as usize).checked_mul(8).ok_or_else(||failure("graph checkpoint size"))?).ok_or_else(||failure("graph checkpoint size"))?;
        for slot in 0..allocated {
            if let Some(node)=view.current_node(slot+1) {
                length=length.checked_add(36).ok_or_else(||failure("graph checkpoint size"))?;
                for level in 0..=node.level() {
                    let words=node.links(level).unwrap().len().checked_add(1).ok_or_else(||failure("graph checkpoint links"))?;
                    length=length.checked_add(words.checked_mul(4).ok_or_else(||failure("graph checkpoint links"))?).ok_or_else(||failure("graph checkpoint size"))?;
                }
            }
        }
        Ok(length)
    }
    /// [`Self::encode_retained_admitted`]'s bytes handed to `out` in pieces
    /// of about `chunk` bytes, never all in memory at once (a checkpoint of a
    /// large graph is many GB: built whole, it was anonymous memory zeroed
    /// page by page, then copied to the page cache). Returns the length.
    #[allow(clippy::too_many_arguments)]
    fn stream_retained(&self,view:&MutableHnswSnapshot,allocated:u32,watermark:Option<u64>,retirements:&std::collections::HashMap<NodeHandle,u64>,incarnations:Option<&[u32]>,
        chunk:usize,out:&mut dyn FnMut(&[u8])->Result<()>)->Result<usize> {
        let chunk=chunk.max(64);
        let _memory=crate::admission::shared().reserve(chunk as u64 * 2,"streamed graph checkpoint buffer")?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(chunk * 2).map_err(|_|failure("graph checkpoint buffer"))?;
        let mut written=0usize;
        self.encode_retained_chunked(&mut bytes,view,allocated,watermark,retirements,incarnations,chunk,&mut |b: &mut Vec<u8>| {
            out(b.as_slice())?; written+=b.len(); b.clear(); Ok(())
        })?;
        if !bytes.is_empty() { out(&bytes)?; written+=bytes.len(); }
        Ok(written)
    }
    fn encode_retained_admitted(&self,view:&MutableHnswSnapshot,allocated:u32,watermark:Option<u64>,retirements:&std::collections::HashMap<NodeHandle,u64>,incarnations:Option<&[u32]>)->Result<crate::admission::Allocation<Vec<u8>>> {
        let length=self.retained_len(view,allocated)?;
        let memory=crate::admission::shared().reserve(length as u64,"serialized graph checkpoint")?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(length).map_err(|_|failure("graph checkpoint allocation"))?;
        self.encode_retained_into(&mut bytes,view,allocated,watermark,retirements,incarnations);
        debug_assert_eq!(bytes.len(),length);
        Ok(crate::admission::Allocation::new(bytes,memory))
    }
    fn encode_retained_into(&self,bytes:&mut Vec<u8>,view:&MutableHnswSnapshot,allocated:u32,watermark:Option<u64>,retirements:&std::collections::HashMap<NodeHandle,u64>,incarnations:Option<&[u32]>) {
        self.encode_retained_chunked(bytes,view,allocated,watermark,retirements,incarnations,usize::MAX,&mut |_| Ok(()))
            .expect("an unbounded encode never flushes");
    }
    /// [`Self::encode_retained_into`], calling `flush` with the bytes so far
    /// whenever they reach `chunk` (at a node boundary; `flush` empties them).
    #[allow(clippy::too_many_arguments)]
    fn encode_retained_chunked(&self,bytes:&mut Vec<u8>,view:&MutableHnswSnapshot,allocated:u32,watermark:Option<u64>,retirements:&std::collections::HashMap<NodeHandle,u64>,incarnations:Option<&[u32]>,
        chunk:usize,flush:&mut dyn FnMut(&mut Vec<u8>)->Result<()>)->Result<()> {
        bytes.extend_from_slice(b"RMHG\x04\0\0\0");
        put32(bytes, self.dim);
        put32(
            bytes,
            match self.similarity {
                Similarity::Cosine => 0,
                Similarity::DotProduct => 1,
                Similarity::L2 => 2,
                Similarity::MaxInnerProduct => 3,
                Similarity::BitHamming => 4,
                Similarity::ByteCosine => 5,
                Similarity::ByteDotProduct => 6,
                Similarity::ByteL2 => 7,
                Similarity::ByteMaxInnerProduct => 8,
            },
        );
        put32(bytes, self.params.m as u32);
        put64(bytes, self.params.ef_construction as u64);
        put64(bytes, self.params.seed);
        put64(bytes, view.entry);
        put32(bytes, allocated);
        put64(bytes, view.publication);
        put32(bytes, u32::from(watermark.is_some()));
        put64(bytes, watermark.unwrap_or(0));
        let table = self
            .control
            .nodes
            .read(&view.guard)
            .expect("matching domain");
        for slot in 0..allocated {
            if bytes.len() >= chunk { flush(bytes)?; }
            let incarnation = incarnations.map_or_else(
                || table.get(slot + 1).expect("allocated node row").metadata().incarnation,
                |saved| saved[slot as usize]);
            put32(bytes, incarnation);
            let node = view.current_node(slot + 1);
            put32(bytes, u32::from(node.is_some()));
            let Some(node) = node else {
                continue;
            };
            put64(bytes, node.key());
            put64(bytes, node.record.publication_after_acquire());
            put32(bytes, node.inverse_norm().to_bits());
            put32(bytes, node.level() as u32);
            let retirement = retirements.get(&node.handle());
            put32(bytes, u32::from(retirement.is_some()));
            put64(bytes, retirement.copied().unwrap_or(0));
            for level in 0..=node.level() {
                let links = node.links(level).unwrap();
                put32(bytes, links.len() as u32);
                for &id in links {
                    put32(bytes, id);
                }
            }
        }
        Ok(())
    }
    pub fn open(bytes: &[u8]) -> Result<Self> {
        let mut input = Decoder { bytes, offset: 0 };
        let header = input.take(8)?;
        let legacy = header == b"RMHG\x03\0\0\0";
        if !legacy && header != b"RMHG\x04\0\0\0" {
            return Err(corrupt("header/version (expected RMHG3 or RMHG4)"));
        }
        let dim = input.u32()?;
        let similarity = match input.u32()? {
            0 => Similarity::Cosine,
            1 => Similarity::DotProduct,
            2 => Similarity::L2,
            3 => Similarity::MaxInnerProduct,
            4 => Similarity::BitHamming,
            5 => Similarity::ByteCosine,
            6 => Similarity::ByteDotProduct,
            7 => Similarity::ByteL2,
            8 => Similarity::ByteMaxInnerProduct,
            _ => return Err(corrupt("similarity")),
        };
        let m = input.u32()? as usize;
        let ef_construction =
            usize::try_from(input.u64()?).map_err(|_| corrupt("construction width"))?;
        let seed = input.u64()?;
        let mut entry = input.u64()?;
        let len = input.u32()?;
        let publication = input.u64()?;
        let has_watermark = input.u32()?;
        let watermark_value = input.u64()?;
        if has_watermark > 1 || (has_watermark == 0 && watermark_value != 0) {
            return Err(corrupt("watermark"));
        }
        let watermark = (has_watermark == 1).then_some(watermark_value);
        if (len != 0 && watermark.is_none())
            || (len == 0 && (watermark.is_some() || publication != 0))
        {
            return Err(corrupt("allocator watermark/slot universe"));
        }
        if dim == 0
            || m == 0
            || m > u32::MAX as usize / 2
            || ef_construction == 0
            || len == u32::MAX
            || len as usize > input.remaining() / 8
        {
            return Err(corrupt("dimensions/count"));
        }
        let graph = Self::try_new(
            dim,
            similarity,
            HnswParams {
                m,
                ef_construction,
                seed,
                threads: 1,
            },
        )?;
        // Free slots and retirement membership remain owned by the writer.
        // HashMap load factor/power-of-two upper bound includes control bytes.
        let count=len as usize;
        let buckets=count.max(1).checked_next_power_of_two().and_then(|v|v.checked_mul(2)).ok_or_else(||failure("checkpoint registry capacity"))?;
        let key_bytes=buckets.checked_mul(std::mem::size_of::<u64>()+1).and_then(|v|v.checked_add(32)).ok_or_else(||failure("checkpoint duplicate key bytes"))?;
        let _key_memory=crate::admission::shared().reserve(key_bytes as u64,"checkpoint duplicate key workspace")?;
        let mut writer = graph.control.writer.lock();
        writer.reserve_slots(count)?;
        writer.watermark = watermark;

        graph
            .control
            .nodes
            .ensure_len(len + 1)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        graph.control.allocated.store(len, Release);
        let mut active = 0;
        let mut live_bound = 1;
        let mut keys = HashSet::new();
        keys.try_reserve(count).map_err(|_|failure("checkpoint duplicate key allocation"))?;
        for slot in 0..len {
            let generation = input.u32()?;
            let present = input.u32()?;
            if generation == 0 || present > 1 {
                return Err(corrupt("slot metadata"));
            }
            let id = NodeHandle::new(slot, generation);
            if present == 0 {
                // Private new graph: no reader/source can observe this temporary
                // reservation. Retain its incarnation while restoring vacancy.
                unsafe {
                    graph.control.nodes.initialize(
                        slot + 1,
                        NodeMetadata {
                            source_key: 0,
                            incarnation: generation,
                            publication: 0,
                            norm: 0.0,
                            state: NodeState::Reserved,
                        },
                    )
                }
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
                graph
                    .control
                    .nodes
                    .invalidate(slot + 1)
                    .map_err(crate::vector::hnsw::mutable::failure_error)?;
                unsafe { graph.control.nodes.mark_reclaimed(slot + 1) }
                    .map_err(crate::vector::hnsw::mutable::failure_error)?;
                if generation == u32::MAX {
                    writer.exhausted += 1;
                } else {
                    writer.free.push(slot);
                }
                continue;
            }
            let key = input.u64()?;
            let born = input.u64()?;
            let inverse_norm = f32::from_bits(input.u32()?);
            let level = input.u32()? as usize;
            let has_retirement = input.u32()?;
            let retirement = input.u64()?;
            if !inverse_norm.is_finite()
                || inverse_norm < 0.0
                || level > 63
                || born == 0
                || born > publication
                || !watermark.is_some_and(|last| key <= last)
                || !keys.insert(key)
                || has_retirement > 1
                || (has_retirement == 0 && retirement != 0)
            {
                return Err(corrupt("node metadata/key"));
            }
            let _reference_memory=crate::admission::shared().reserve(((level+1)*std::mem::size_of::<u32>()) as u64,"checkpoint level references")?;
            let mut references = Vec::with_capacity(level + 1);
            for l in 0..=level {
                let count = input.u32()? as usize;
                if count > if l == 0 { 2 * m } else { m } || count > input.remaining() / 4 {
                    return Err(corrupt("neighbor count"));
                }
                let _neighbor_memory=crate::admission::shared().reserve((count as u64)*4,"checkpoint neighbor decode")?;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    let encoded = input.u32()?;
                    let other = if legacy {
                        encoded
                            .checked_add(1)
                            .ok_or_else(|| corrupt("legacy neighbor overflow"))?
                    } else {
                        encoded
                    };
                    if other == 0 || other > len || neighbors.contains(&other) {
                        return Err(corrupt("neighbor"));
                    }
                    neighbors.push(other);
                }
                references.push(
                    graph
                        .control
                        .links
                        .writer()
                        .add(&neighbors)
                        .map_err(crate::vector::hnsw::mutable::failure_error)?
                        .raw(),
                );
            }
            let levels = graph
                .control
                .levels
                .writer()
                .add(&references)
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
            unsafe {
                graph.control.nodes.initialize(
                    slot + 1,
                    NodeMetadata {
                        source_key: key,
                        incarnation: generation,
                        publication: born,
                        norm: inverse_norm,
                        state: NodeState::Reserved,
                    },
                )
            }
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
            unsafe { graph.control.nodes.publish(slot + 1, levels, born) }
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
            if has_retirement == 1 {
                writer.visibility_retained.insert(id, retirement);
            }
            active += 1;
            live_bound = slot + 2;
        }
        if input.remaining() != 0 {
            return Err(corrupt("trailing bytes"));
        }
        graph.control.publication.store(publication, Release);
        graph.control.live_bound.store(live_bound, Release);
        let view = graph.snapshot();
        for id in view.handles() {
            let node = view.node(id).unwrap();
            for level in 0..=node.level() {
                for &neighbor in node.links(level).unwrap() {
                    let Some(other) = view.current_node(neighbor) else {
                        return Err(corrupt("absent neighbor"));
                    };
                    if neighbor == internal_id(id)
                        || !view.visible(other)
                        || other.level() < level
                        || !other.links(level).unwrap().contains(&internal_id(id))
                    {
                        return Err(corrupt("asymmetric neighbor/level"));
                    }
                }
            }
        }
        if legacy {
            entry = if entry == u64::MAX {
                if active != 0 {
                    return Err(corrupt("legacy missing entry"));
                }
                NO_ENTRY
            } else {
                let handle = NodeHandle(entry);
                let node = view
                    .node(handle)
                    .filter(|n| view.visible(*n))
                    .ok_or_else(|| corrupt("legacy entry incarnation"))?;
                packed_entry(internal_id(handle), node.level())
            };
        }
        if entry != NO_ENTRY {
            let id = entry as u32;
            let level = (entry >> 32) as usize;
            if !view
                .current_node(id)
                .is_some_and(|n| view.visible(n) && n.level() >= level)
            {
                return Err(corrupt("entry"));
            }
        }
        // A disconnected live graph can legitimately have no entry after its
        // isolated entry is removed, matching pinned remove_node behavior.
        graph.control.entry.store(entry, Release);
        graph.control.active.store(active, Release);
        drop(view);
        graph.control.finish_mutation(&mut writer);
        drop(writer);
        Ok(graph)
    }
}

/// Planning reads the same progressively updated adjacency as the previous
/// serialized implementation. Only final owner values are allocated/published.
///
/// Bind lhs cells and their squared norm once. Vespa's float access uses f32
/// SIMD products, promoted to double for angular normalization. Preserve ES
/// public scores separately from these internal traversal distances.
struct BoundMutationDistance<'a> {
    row: &'a [f32],
    similarity: Similarity,
    squared_norm: f64,
    legacy_norm: f32,
}
impl<'a> BoundMutationDistance<'a> {
    fn new(row: &'a [f32], similarity: Similarity) -> Self {
        Self { row, similarity,
            squared_norm: if cfg!(feature = "mutable-hnsw-vespa-angular") && similarity == Similarity::Cosine { f64::from(dot(row, row)) } else { 0.0 },
            legacy_norm: if !cfg!(feature = "mutable-hnsw-vespa-angular") && similarity == Similarity::Cosine { norm(row) } else { 0.0 } }
    }
    fn distance(&self, right: &[f32]) -> Result<f64> {
        let distance = self.distance_value(right);
        if !distance.is_finite() {
            return Err(failure("nonfinite mutation distance"));
        }
        Ok(distance)
    }
    fn distance_value(&self, right: &[f32]) -> f64 {
        if self.similarity == Similarity::Cosine {
            if cfg!(feature = "mutable-hnsw-vespa-angular") {
                angular_distance(self.squared_norm, right, dot(self.row, right))
            } else {
                let distance = if self.legacy_norm == 0.0 { 1.0 } else {
                    let (product, squared_norm) = crate::vector::distance::dot_and_sq_norm(self.row, right);
                    if squared_norm == 0.0 { 1.0 } else { 1.0 - product / (self.legacy_norm * squared_norm.sqrt()) }
                };
                f64::from(distance)
            }
        } else {
            f64::from(self.similarity.distance(self.row, right))
        }
    }
}

#[inline(always)]
fn angular_distance(lhs_norm_sq: f64, right: &[f32], product: f32) -> f64 {
    let squared_norms = lhs_norm_sq * f64::from(dot(right, right));
    let divisor = if squared_norms > 0.0 { squared_norms.sqrt() } else { 1.0 };
    1.0 - f64::from(product) / divisor
}

struct AdjacencyOverlay<'a, S: ?Sized> {
    view: &'a MutableHnswSnapshot,
    source: &'a S,
    similarity: Similarity,
    m: usize,
    new: Option<(NodeHandle, u64, usize)>,
    changed: BTreeMap<(NodeHandle, usize), Vec<u32>>,
}
impl<'a, S: MutableVectorSource + ?Sized> AdjacencyOverlay<'a, S> {
    fn new(
        view: &'a MutableHnswSnapshot,
        source: &'a S,
        similarity: Similarity,
        m: usize,
        new: Option<(NodeHandle, u64, usize)>,
    ) -> Self {
        Self {
            view,
            source,
            similarity,
            m,
            new,
            changed: BTreeMap::new(),
        }
    }
    fn row(&self, id: NodeHandle) -> Result<&'a [f32]> {
        let key = if let Some((new, key, _)) = self.new.filter(|n| n.0 == id) {
            let _ = new;
            key
        } else {
            self.view
                .node(id)
                .ok_or_else(|| failure("overlay owner disappeared"))?
                .key()
        };
        // Pruning compares rows pairwise: approximate rows for quantized
        // fields (see `MutableVectorSource::get_node_approx`).
        let row = self.source.get_node_approx(id, key);
        self.source.execution_result()?;
        let row = row.ok_or_else(|| failure("missing vector"))?;
        if row.len() != self.view.dim as usize || !self.similarity.valid_row(row) {
            return Err(failure("invalid vector"));
        }
        Ok(row)
    }
    fn handle(&self, id: u32) -> Result<NodeHandle> {
        if let Some((new, _, _)) = self.new.filter(|n| internal_id(n.0) == id) {
            return Ok(new);
        }
        self.view
            .current_handle(id)
            .ok_or_else(|| failure("overlay edge has no current owner"))
    }
    fn links(&mut self, id: NodeHandle, level: usize) -> Result<&mut Vec<u32>> {
        if !self.changed.contains_key(&(id, level)) {
            let links = if self.new.is_some_and(|n| n.0 == id) {
                Vec::new()
            } else {
                self.view
                    .node(id)
                    .and_then(|n| n.links(level))
                    .ok_or_else(|| failure("overlay missing level"))?
                    .to_vec()
            };
            self.changed.insert((id, level), links);
        }
        Ok(self.changed.get_mut(&(id, level)).unwrap())
    }
    fn set(&mut self, id: NodeHandle, level: usize, links: Vec<u32>) {
        self.changed.insert((id, level), links);
    }
    fn add(&mut self, from: NodeHandle, to: NodeHandle, level: usize) -> Result<()> {
        if from != to {
            let links = self.links(from, level)?;
            let id = internal_id(to);
            if !links.contains(&id) {
                links.push(id);
            }
        }
        Ok(())
    }
    fn remove(&mut self, from: NodeHandle, to: NodeHandle, level: usize) -> Result<()> {
        self.links(from, level)?.retain(|&id| id != internal_id(to));
        Ok(())
    }
    fn bind_distance(&self, id: NodeHandle) -> Result<BoundMutationDistance<'a>> {
        Ok(BoundMutationDistance::new(self.row(id)?, self.similarity))
    }
    fn shrink(&mut self, id: NodeHandle, level: usize) -> Result<()> {
        let capacity = if level == 0 { 2 * self.m } else { self.m };
        let links = self.links(id, level)?;
        if links.len() <= capacity {
            return Ok(());
        }
        let old = links.clone();
        let kept = self.select_kept(id, &old, capacity)?;
        self.replace_pruned(id, level, old, kept)
    }
    /// What pruning keeps of `old`, node `id`'s overflowing link list: the
    /// candidates by distance from `id`, each kept unless a kept one is
    /// closer to it (Vespa's heuristic select_neighbors), up to `capacity`.
    /// Reads rows only; changes nothing.
    fn select_kept(&self, id: NodeHandle, old: &[u32], capacity: usize) -> Result<Vec<u32>> {
        if let Some(kept) = self.select_kept_codes(id, old, capacity)? {
            return Ok(kept);
        }
        let mut candidates = Vec::with_capacity(old.len());
        let from_owner = self.bind_distance(id)?;
        for &other in old {
            let other = self.handle(other)?;
            candidates.push((from_owner.distance(self.row(other)?)?, other));
        }
        candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("finite distance"));
        // Rows are borrowed from the mutation's pinned source; retain the
        // checked slice while selecting instead of resolving its owner for
        // every pairwise comparison. Candidate order/distances are unchanged.
        let mut selected: Vec<(NodeHandle, &'a [f32])> = Vec::with_capacity(capacity);
        for (distance, candidate) in candidates {
            if selected.len() >= capacity {
                break;
            }
            let mut valid = true;
            let from_candidate = if selected.is_empty() { None } else { Some(self.bind_distance(candidate)?) };
            for &(_, other_row) in &selected {
                if from_candidate.as_ref().expect("nonempty selected set").distance(other_row)? < distance {
                    valid = false;
                    break;
                }
            }
            if valid {
                let row = match &from_candidate {
                    Some(bound) => bound.row,
                    None => self.row(candidate)?,
                };
                selected.push((candidate, row));
            }
        }
        Ok(selected.into_iter().map(|(id, _)| internal_id(id)).collect())
    }
    /// [`select_kept`](Self::select_kept) with every distance on the stored
    /// codes (a quantized field: Vespa computes pruning distances on the
    /// tensor's stored cell type), where the row path decodes each row of
    /// the list from its codes and compares the decoded rows. `None` when
    /// the source has no such codes or the list mixes calibration epochs.
    fn select_kept_codes(&self, id: NodeHandle, old: &[u32], capacity: usize) -> Result<Option<Vec<u32>>> {
        let key = |id: NodeHandle| -> Result<u64> {
            if let Some((_, key, _)) = self.new.filter(|n| n.0 == id) { return Ok(key); }
            Ok(self.view.node(id).ok_or_else(|| failure("overlay owner disappeared"))?.key())
        };
        let Some((epoch, owner)) = self.source.code_row(id, key(id)?) else { return Ok(None) };
        let Some(distance) = self.source.code_scorer(epoch) else { return Ok(None) };
        let mut candidates = Vec::with_capacity(old.len());
        for &other in old {
            let other = self.handle(other)?;
            match self.source.code_row(other, key(other)?) {
                Some((e, words)) if e == epoch => candidates.push((distance(owner, words), other, words)),
                _ => return Ok(None),
            }
        }
        if candidates.iter().any(|c| !c.0.is_finite()) {
            return Err(failure("nonfinite mutation distance"));
        }
        candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("finite distance"));
        let mut selected: Vec<(NodeHandle, &[f32])> = Vec::with_capacity(capacity);
        for (d, candidate, words) in candidates {
            if selected.len() >= capacity {
                break;
            }
            if selected.iter().all(|&(_, other)| distance(words, other) >= d) {
                selected.push((candidate, words));
            }
        }
        Ok(Some(selected.into_iter().map(|(id, _)| internal_id(id)).collect()))
    }
    /// Node `id`'s list at `level` becomes `kept` (of `old`); the nodes it
    /// dropped lose their link back to it, as Vespa's shrink_if_needed does.
    fn replace_pruned(&mut self, id: NodeHandle, level: usize, old: Vec<u32>, kept: Vec<u32>) -> Result<()> {
        let removed: Vec<_> = old.into_iter().filter(|id| !kept.contains(id)).collect();
        self.set(id, level, kept);
        for other in removed {
            let other = self.handle(other)?;
            self.remove(other, id, level)?;
        }
        Ok(())
    }
    /// Apply pruning computed in preparation if the neighbour's list (with
    /// this insert's reverse link) is still exactly the list, and its owners
    /// the nodes, it was computed from: the selection is then the one
    /// `shrink` would make. Otherwise changes nothing (`false`).
    fn apply_pruned(&mut self, ahead: &PrunedLinks) -> Result<bool> {
        let links = self.links(ahead.node, ahead.level)?;
        if *links != ahead.observed {
            return Ok(false);
        }
        for (&other, &handle) in ahead.observed.iter().zip(&ahead.handles) {
            if self.handle(other).ok() != Some(handle) {
                return Ok(false);
            }
        }
        let old = ahead.observed.clone();
        // Unit tests check every applied selection against a fresh one.
        #[cfg(test)]
        {
            let capacity = if ahead.level == 0 { 2 * self.m } else { self.m };
            assert_eq!(self.select_kept(ahead.node, &old, capacity)?, ahead.kept, "pruning computed in preparation differs");
        }
        self.replace_pruned(ahead.node, ahead.level, old, ahead.kept.clone())?;
        Ok(true)
    }
    fn reconnect(&mut self, cluster: &[NodeHandle], level: usize) -> Result<()> {
        let mut pairs = Vec::new();
        for (i, &a) in cluster.iter().enumerate() {
            let neighbors = self.links(a, level)?.clone();
            let mut from_a = None;
            for &b in &cluster[i + 1..] {
                if !neighbors.contains(&internal_id(b)) {
                    if from_a.is_none() { from_a = Some(self.bind_distance(a)?); }
                    pairs.push((from_a.as_ref().unwrap().distance(self.row(b)?)?, a, b));
                }
            }
        }
        // Pinned PairDist (hnsw_index.cpp:83-91) orders numeric distance only.
        // Equal distances, including +/-0, are equivalent; Rust's unstable
        // sort need not choose the same tied order as C++ std::sort.
        pairs.sort_unstable_by(reconnect_pair_cmp);
        let capacity = if level == 0 { 2 * self.m } else { self.m };
        let mut added = Vec::new();
        for &(_, a, b) in &pairs {
            let la = self.links(a, level)?.len();
            let lb = self.links(b, level)?.len();
            if la >= capacity || lb >= capacity {
                continue;
            }
            if la == 0 || lb == 0 || (!added.contains(&a) && !added.contains(&b)) {
                self.add(a, b, level)?;
                self.add(b, a, level)?;
                added.push(a);
                added.push(b);
            }
        }
        // The pinned pass gives each node at most one new edge, so the
        // removed node's neighbours can be left in separate groups: a pair
        // inside a group is closer than any pair across, takes both ends'
        // one edge, and nothing then links the groups that were connected
        // only through the removed node (a collinear field lost every live
        // document past a run of deletes; a 4-node group became an island).
        // Link the groups: the cluster's nodes by their direct links, then
        // the closest unlinked pairs across groups (Kruskal over `pairs`),
        // capacity permitting, until the cluster is one group.
        let mut group: Vec<usize> = (0..cluster.len()).collect();
        fn root(group: &mut [usize], mut i: usize) -> usize {
            while group[i] != i { group[i] = group[group[i]]; i = group[i]; }
            i
        }
        for (i, &a) in cluster.iter().enumerate() {
            let links = self.links(a, level)?.clone();
            for (j, &b) in cluster.iter().enumerate().skip(i + 1) {
                if links.contains(&internal_id(b)) {
                    let (ri, rj) = (root(&mut group, i), root(&mut group, j));
                    group[ri] = rj;
                }
            }
        }
        for &(_, a, b) in &pairs {
            let (Some(i), Some(j)) = (cluster.iter().position(|&c| c == a), cluster.iter().position(|&c| c == b)) else { continue };
            let (ri, rj) = (root(&mut group, i), root(&mut group, j));
            if ri == rj {
                continue;
            }
            if self.links(a, level)?.len() >= capacity || self.links(b, level)?.len() >= capacity {
                continue;
            }
            self.add(a, b, level)?;
            self.add(b, a, level)?;
            group[ri] = rj;
        }
        Ok(())
    }
}
type ReconnectPair = (f64, NodeHandle, NodeHandle);
fn reconnect_pair_cmp(a: &ReconnectPair, b: &ReconnectPair) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0).expect("finite reconnect distance")
}
#[derive(Clone, Copy)]
enum Publication {
    Insert {
        id: NodeHandle,
        level: usize,
        publication: u64,
    },
    Remove {
        id: NodeHandle,
        key: u64,
        replacement: Option<(NodeHandle, usize)>,
    },
    LinksOnly,
    RelocateLinks,
}
struct PoisonOnUnwind<'a> {
    control: &'a Control,
    armed: bool,
}
impl Drop for PoisonOnUnwind<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.control.poison();
        }
    }
}
impl MutableHnsw {
    fn publish_overlay<S: MutableVectorSource + ?Sized>(
        &self,
        view: &MutableHnswSnapshot,
        overlay: AdjacencyOverlay<'_, S>,
        publication: Publication,
        _writer: &mut Writer,
    ) -> Result<()> {
        self.control.mark_maintenance_pending(_writer);
        let links_view = self
            .control
            .links
            .view(&view.guard)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let levels_view = self
            .control
            .levels
            .view(&view.guard)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut changes = Vec::new();
        let mut old_links = Vec::new();
        let mut old_levels = Vec::new();
        for ((handle, level), values) in overlay.changed {
            let new = matches!(publication,Publication::Insert{id,..} if id==handle);
            if !new {
                let node = view
                    .node(handle)
                    .ok_or_else(|| failure("publication owner disappeared"))?;
                if node
                    .links(level)
                    .ok_or_else(|| failure("publication level absent"))?
                    == values && !matches!(publication, Publication::RelocateLinks)
                {
                    continue;
                }
                let old = LinksRef::from_raw(node.level_refs()[level].load(Acquire));
                if old.is_valid() {
                    old_links.push(
                        unsafe { links_view.retirement_token(old) }
                            .map_err(crate::vector::hnsw::mutable::failure_error)?,
                    );
                }
            }
            changes.push((handle, level, values));
        }
        // A chunked capture in progress keeps the graph as of its start:
        // record each node it has not copied yet before its first change.
        if _writer.capture_log.is_some() {
            let removed=match publication {Publication::Remove{id,..}=>Some(id),_=>None};
            let touched=changes.iter()
                .filter(|change| !matches!(publication, Publication::Insert{id,..} if id==change.0))
                .map(|change|change.0).chain(removed);
            if let Some(log) = _writer.capture_log.as_mut() {
                for handle in touched {
                    let slot = internal_id(handle);
                    if slot < log.cursor || slot >= log.len || log.images.contains_key(&slot) { continue; }
                    let Some(node) = view.node(handle) else { continue };
                    log.reserve_image()?;
                    let links=Self::capture_links(node.level_refs())?;
                    log.images.insert(slot,(node.levels,links));
                }
            }
        }
        if let Publication::Remove { id, .. } = publication {
            let node = view
                .node(id)
                .ok_or_else(|| failure("removal owner disappeared"))?;
            old_levels.push(
                unsafe { levels_view.retirement_token(node.levels) }
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            );
            for reference in node.level_refs() {
                let old = LinksRef::from_raw(reference.load(Acquire));
                if old.is_valid() {
                    old_links.push(
                        unsafe { links_view.retirement_token(old) }
                            .map_err(crate::vector::hnsw::mutable::failure_error)?,
                    );
                }
            }
        }
        let mut link_batch = self
            .control
            .links
            .writer()
            .stage_batch(changes.len(), &old_links)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut staged = Vec::with_capacity(changes.len());
        for (_, _, values) in &changes {
            staged.push(
                link_batch
                    .stage(values)
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            );
        }
        let link_batch = link_batch
            .finish_staging()
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut references = Vec::with_capacity(staged.len());
        for index in staged {
            references.push(
                link_batch
                    .resolve(index)
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            );
        }
        let mut level_batch = self
            .control
            .levels
            .writer()
            .stage_batch(
                usize::from(matches!(publication, Publication::Insert { .. })),
                &old_levels,
            )
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let new_levels = if let Publication::Insert { id, level, .. } = publication {
            let mut levels = vec![0; level + 1];
            for ((handle, l, _), reference) in changes.iter().zip(&references) {
                if *handle == id {
                    levels[*l] = reference.raw();
                }
            }
            Some(
                level_batch
                    .stage(&levels)
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            )
        } else {
            None
        };
        let level_batch = level_batch
            .finish_staging()
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let new_levels = new_levels
            .map(|index| level_batch.resolve(index))
            .transpose()
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        // The dense level-0 links words to record after publication (no
        // allocation once publishing starts).
        let mut links0 = Vec::with_capacity(changes.len());
        let mut inserted_links0 = 0;
        for ((handle, level, _), new) in changes.iter().zip(&references) {
            if *level != 0 { continue; }
            if matches!(publication, Publication::Insert{id,..} if id==*handle) { inserted_links0 = new.raw(); }
            else { links0.push((internal_id(*handle), new.raw())); }
        }
        // Resolve each atomic owner only AFTER both stores have finished growing.
        let mut replacements = Vec::with_capacity(changes.len());
        for ((handle, level, _), new) in changes.iter().zip(&references) {
            if matches!(publication,Publication::Insert{id,..} if id==*handle) {
                continue;
            }
            let node = view
                .node(*handle)
                .ok_or_else(|| failure("owner invalidated before preflight"))?;
            let old = node.level_refs()[*level].load(Acquire);
            let owner = unsafe { levels_view.retirement_token(node.levels) }
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
            replacements.push(
                level_batch
                    .preflight_replace(owner, *level, old, new.raw())
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            );
        }
        let transition = match publication {
            Publication::Insert {
                id, publication, ..
            } => Some(
                unsafe {
                    self.control.nodes.preflight_publish_with_links0(
                        internal_id(id),
                        new_levels.unwrap(),
                        publication,
                        inserted_links0,
                    )
                }
                .map_err(crate::vector::hnsw::mutable::failure_error)?,
            ),
            Publication::Remove { id, .. } => Some(
                self.control
                    .nodes
                    .preflight_invalidate(internal_id(id), view.node(id).unwrap().levels)
                    .map_err(crate::vector::hnsw::mutable::failure_error)?,
            ),
            Publication::LinksOnly | Publication::RelocateLinks => None,
        };
        let next_bound = if let Publication::Remove { id, .. } = publication {
            let mut bound = self.control.live_bound.load(Relaxed);
            while bound > 1
                && ((bound - 1) == internal_id(id) || view.current_node(bound - 1).is_none())
            {
                bound -= 1;
            }
            bound
        } else {
            self.control.live_bound.load(Relaxed)
        };
        // No fallible work, callbacks, distance calculations or allocation below.
        let mut poison = PoisonOnUnwind {
            control: &self.control,
            armed: true,
        };
        self.control.begin_topology_change();
        let link_batch = link_batch.begin_publish();
        let mut level_batch = level_batch.begin_publish();
        for replacement in replacements {
            unsafe { level_batch.replace_prevalidated(replacement) };
        }
        if let Publication::Remove {
            id, replacement, ..
        } = publication
        {
            if self.control.entry.load(Relaxed) as u32 == internal_id(id) {
                self.control.entry.store(
                    replacement.map_or(NO_ENTRY, |(h, l)| packed_entry(internal_id(h), l)),
                    Release,
                );
            }
        }
        if let Some(transition) = transition {
            transition.commit();
        }
        // The node table's mutex, released by the commit above. Until this
        // store a reader may still follow a node's previous level-0 links
        // (retired, held for it): the topology it read an instant earlier.
        self.control.nodes.set_links0(&links0);
        match publication {
            Publication::Insert {
                id,
                level,
                publication,
            } => {
                self.control
                    .live_bound
                    .fetch_max(internal_id(id) + 1, Release);
                self.control.active.fetch_add(1, Release);
                self.control.publication.store(publication, Release);
                let entry = self.control.entry.load(Relaxed);
                if entry == NO_ENTRY || level > (entry >> 32) as usize {
                    self.control
                        .entry
                        .store(packed_entry(internal_id(id), level), Release);
                }
            }
            Publication::Remove { .. } => {
                self.control.active.fetch_sub(1, Release);
                self.control.live_bound.store(next_bound, Release);
            }
            Publication::LinksOnly | Publication::RelocateLinks => {}
        }
        self.control.end_topology_change();
        unsafe {
            level_batch.finish_after_publish();
            link_batch.finish_after_publish();
        }
        if let Publication::Remove { id, key, .. } = publication {
            _writer.visibility_retained.remove(&id);
            self.control.retire_committed(_writer, id, key);
        }
        poison.armed = false;
        self.control.note_mutation();
        Ok(())
    }
    #[cfg(test)]
    fn test_set_vacant_incarnation(&self, slot: u32, generation: u32) -> Result<()> {
        let _writer = self.control.writer.lock();
        if generation == 0 || !_writer.free.contains(&slot) {
            return Err(failure("test slot must be vacant/free"));
        }
        // Test fixture owns the graph exclusively and has drained all readers
        // and source ACKs before modifying this deliberately terminal counter.
        unsafe {
            self.control.nodes.initialize(
                slot + 1,
                NodeMetadata {
                    source_key: 0,
                    incarnation: generation,
                    publication: 0,
                    norm: 0.0,
                    state: NodeState::Reserved,
                },
            )
        }
        .map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.control
            .nodes
            .invalidate(slot + 1)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        unsafe { self.control.nodes.mark_reclaimed(slot + 1) }
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        Ok(())
    }
    #[cfg(test)]
    fn test_install_node(
        &self,
        handle: NodeHandle,
        key: u64,
        norm: f32,
        links: Vec<Vec<u32>>,
    ) -> Result<()> {
        struct NoRows;
        impl MutableVectorSource for NoRows {
            fn get(&self, _: u64) -> Option<&[f32]> {
                None
            }
        }
        if links.is_empty() {
            return Err(failure("test node needs a level"));
        }
        let mut writer = self.control.writer.lock();
        self.control.admit_write(&mut writer)?;
        if writer.pending.get(&key) != Some(&handle) {
            return Err(failure("test node has no reservation"));
        }
        let view = self.try_snapshot()?;
        let level = links.len() - 1;
        let mut overlay = AdjacencyOverlay::new(
            &view,
            &NoRows,
            self.similarity,
            self.params.m,
            Some((handle, key, level)),
        );
        for (l, values) in links.into_iter().enumerate() {
            overlay.set(handle, l, values);
        }
        self.control
            .nodes
            .set_reserved_norm(internal_id(handle), norm)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let publication = self
            .control
            .publication
            .load(Relaxed)
            .checked_add(1)
            .ok_or_else(|| failure("publication exhausted"))?;
        self.publish_overlay(
            &view,
            overlay,
            Publication::Insert {
                id: handle,
                level,
                publication,
            },
            &mut writer,
        )?;
        writer.pending.remove(&key);
        drop(view);
        self.control.finish_mutation(&mut writer);
        Ok(())
    }
    #[cfg(test)]
    fn test_move_levels(&self, id: NodeHandle) -> Result<()> {
        let mut writer = self.control.writer.lock();
        self.control.admit_write(&mut writer)?;
        let view = self.try_snapshot()?;
        self.relocate_levels(&view, id, &mut writer)?;
        drop(view);
        self.control.finish_mutation(&mut writer);
        Ok(())
    }
    #[cfg(test)]
    fn test_replace_links(&self, id: NodeHandle, level: usize, links: &[u32]) -> Result<()> {
        struct NoRows;
        impl MutableVectorSource for NoRows {
            fn get(&self, _: u64) -> Option<&[f32]> {
                None
            }
        }
        let mut writer = self.control.writer.lock();
        self.control.admit_write(&mut writer)?;
        let view = self.try_snapshot()?;
        let mut overlay =
            AdjacencyOverlay::new(&view, &NoRows, self.similarity, self.params.m, None);
        overlay.set(id, level, links.to_vec());
        self.publish_overlay(&view, overlay, Publication::LinksOnly, &mut writer)?;
        drop(view);
        self.control.finish_mutation(&mut writer);
        Ok(())
    }
    #[cfg(test)]
    fn test_set_entry(&self, id: NodeHandle) -> Result<()> {
        let _writer = self.control.writer.lock();
        let view = self.try_snapshot()?;
        let node = view.node(id).ok_or_else(|| failure("test entry absent"))?;
        self.control
            .entry
            .store(packed_entry(internal_id(id), node.level()), Release);
        Ok(())
    }
}

impl MutableHnswSnapshot {
    /// Heap bytes of a retained capture's own copy of the topology
    /// (`None`: an ordinary snapshot, which copies nothing).
    pub fn captured_bytes(&self) -> usize {
        self.captured.as_ref().map_or(0, |nodes| nodes.capacity() * std::mem::size_of::<Option<CapturedNode>>()
            + nodes.iter().flatten().map(|n| n.links.capacity() * std::mem::size_of::<LinksRef>()).sum::<usize>())
    }
    pub fn dim(&self) -> u32 { self.dim }
    pub fn similarity(&self) -> Similarity { self.similarity }
    /// Identity of the live graph this snapshot reads (equal for snapshots
    /// of the same graph).
    pub fn graph_identity(&self) -> usize { Arc::as_ptr(&self.control) as usize }
    /// Retired adjacency bytes the live graph holds back for readers.
    pub fn retired_adjacency_bytes(&self) -> usize { self.control.stats().retired_adjacency_bytes }
    pub fn num_nodes(&self) -> u32 {
        self.len.saturating_sub(1)
    }
    /// Enumerate visible source identities under this view's generation guard.
    /// Current views retain the ordinary mutable-topology semantics; callers
    /// restoring a checkpoint must keep the newly decoded graph private.
    /// The visible source identity of graph node `id` (`slot + 1`), if any.
    pub(crate) fn visible_identity(&self, id: u32) -> Option<(u64, NodeHandle)> {
        self.current_node(id).filter(|node| self.visible(*node)).map(|node| (node.key(), node.handle()))
    }
    pub fn visible_identities(&self) -> impl Iterator<Item = (u64, NodeHandle)> + '_ {
        self.handles().filter_map(|handle| {
            self.node(handle).filter(|node| self.visible(*node))
                .map(|node| (node.key(), handle))
        })
    }
    fn node(&self, id: NodeHandle) -> Option<NodeView<'_>> {
        self.current_node(id.slot().checked_add(1)?)
            .filter(|n| n.handle() == id)
    }
    // The generic traversal is instantiated in callers' crates. Export these
    // small accessors for inlining even in the non-LTO iteration profile,
    // preserving every acquire, bound and publication check.
    #[inline]
    fn current_node(&self, id: u32) -> Option<NodeView<'_>> {
        if self.captured.is_some() { self.current_node_typed::<true>(id) }
        else { self.current_node_typed::<false>(id) }
    }
    #[inline]
    fn current_node_typed<const CAPTURED: bool>(&self, id: u32) -> Option<NodeView<'_>> {
        if id == 0 || id >= self.len {
            return None;
        }
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        let row = unsafe { self.control.nodes.read_prevalidated(&self.guard) }
            .get(id)?;
        let levels = if CAPTURED {
            let nodes = self.captured.as_ref().expect("captured traversal");
            nodes.get(id as usize)?.as_ref()?.levels
        } else { row.levels_ref() };
        if !levels.is_valid() {
            return None;
        }
        Some(NodeView {
            view: self,
            id,
            levels,
            record: row.identity_record(),
        })
    }
    fn current_handle(&self, id: u32) -> Option<NodeHandle> {
        self.current_node(id).map(NodeView::handle)
    }
    /// Live (uncaptured) traversal only: a node removed after this reader was
    /// admitted, reached through a neighbour link list read before removal's
    /// rewrite. Removal invalidates the slot right after rewriting the
    /// neighbours, so without this bridge such a reader loses every node it
    /// could only reach through the removed one. The node stays traversable
    /// (its retained levels/links are pinned by this guard) but callers must
    /// never return it as a hit; final hits re-resolve CURRENT nodes.
    #[inline]
    fn removed_bridge(&self, id: u32) -> Option<NodeView<'_>> {
        if self.captured.is_some() || id == 0 || id >= self.len {
            return None;
        }
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        let row = unsafe { self.control.nodes.read_prevalidated(&self.guard) }.get(id)?;
        if row.levels_ref().is_valid() {
            return None;
        }
        let levels = row.removed_levels_for(self.guard.generation())?;
        let node = NodeView { view: self, id, levels, record: row.identity_record() };
        self.visible(node).then_some(node)
    }
    /// Whether an acquired candidate's slot has since been removed (live
    /// traversal); such a candidate is a bridge, not a result.
    #[inline]
    fn candidate_removed(&self, candidate: Candidate) -> bool {
        if self.captured.is_some() {
            return false;
        }
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        unsafe { self.control.nodes.read_prevalidated(&self.guard) }
            .get(candidate.id)
            .is_none_or(|row| !row.levels_ref().is_valid())
    }
    #[inline]
    fn candidate_node(&self, candidate: Candidate) -> NodeView<'_> {
        if self.captured.is_some() { self.candidate_node_typed::<true>(candidate) }
        else { self.candidate_node_typed::<false>(candidate) }
    }
    #[inline]
    fn candidate_node_typed<const CAPTURED: bool>(&self, candidate: Candidate) -> NodeView<'_> {
        if CAPTURED { return self.current_node_typed::<true>(candidate.id).expect("captured candidate"); }
        // SAFETY contract: Candidate is private, acquired under this snapshot's
        // guard, and scratch containers are cleared before another query. ID
        // metadata cannot be reused until this guard and source ACK have drained.
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        let row = unsafe { self.control.nodes.read_prevalidated(&self.guard) }
            .get(candidate.id)
            .expect("acquired candidate ID");
        NodeView {
            view: self,
            id: candidate.id,
            levels: candidate.levels,
            record: row.identity_record(),
        }
    }
    fn handles(&self) -> impl Iterator<Item = NodeHandle> + '_ {
        (1..self.len).filter_map(|id| self.current_handle(id))
    }
    #[inline]
    fn visible(&self, node: NodeView<'_>) -> bool {
        node.record.publication_after_acquire() <= self.publication
    }
    /// `current_node_typed` filtered by visibility checked in the node
    /// table's dense copy of the publication (same value and contract): for
    /// the code plane path, which reads nothing else of the record. Paths
    /// that read the record anyway (its key) check it there (`visible`).
    #[inline]
    fn current_visible_dense<const CAPTURED: bool>(&self, id: u32) -> Option<NodeView<'_>> {
        if id == 0 || id >= self.len {
            return None;
        }
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        let (row, publication) = unsafe { self.control.nodes.read_prevalidated(&self.guard) }.get_with_publication(id)?;
        let levels = if CAPTURED {
            let nodes = self.captured.as_ref().expect("captured traversal");
            nodes.get(id as usize)?.as_ref()?.levels
        } else { row.levels_ref() };
        if !levels.is_valid() || publication.load(Relaxed) > self.publication {
            return None;
        }
        Some(NodeView { view: self, id, levels, record: row.identity_record() })
    }
    /// `visible` read from the node table's dense publication copy (live
    /// traversal of a quantized field: same value and contract).
    #[inline]
    fn visible_dense(&self, node: NodeView<'_>) -> bool {
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        unsafe { self.control.nodes.read_prevalidated(&self.guard) }
            .get_with_publication(node.id)
            .is_some_and(|(_, publication)| publication.load(Relaxed) <= self.publication)
    }
    fn live(&self, id: NodeHandle) -> bool {
        self.node(id).is_some_and(|n| self.visible(n))
    }
    fn entry_level(&self, handle: NodeHandle) -> usize {
        let captured = self.entry;
        if captured as u32 == internal_id(handle) {
            return (captured >> 32) as usize;
        }
        let current = self.control.entry.load(Acquire);
        if current as u32 == internal_id(handle) {
            (current >> 32) as usize
        } else {
            self.node(handle).map_or(0, NodeView::level)
        }
    }
    fn acquired_node(&self, handle: NodeHandle) -> Option<NodeView<'_>> {
        if self.captured.is_some() { return self.node(handle); }
        if let Some(identity) = self.entry_identity {
            if NodeHandle::new((self.entry as u32) - 1, identity.incarnation) == handle {
                // The coherent captured entry may now be invalidated. Its
                // source identity is still immutable in every table copy:
                // this guard prevents ID reuse. Acquire the current root for
                // the record, retaining the already-acquired entry LevelsRef.
                // SAFETY: same admission-validated pair, retained throughout traversal.
                let record = unsafe { self.control.nodes.read_prevalidated(&self.guard) }
                    .get(self.entry as u32)
                    .expect("captured entry ID");
                return Some(NodeView {
                    view: self,
                    id: self.entry as u32,
                    levels: self.entry_levels,
                    record: record.identity_record(),
                });
            }
        }
        self.node(handle)
    }
    fn available_entry<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
    ) -> Option<NodeHandle> {
        let identity = self.entry_identity?;
        let handle = NodeHandle::new((self.entry as u32) - 1, identity.incarnation);
        d.resolve(handle).map(|_| handle)
    }
    fn greedy<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        current: NodeHandle,
        distance: f64,
        level: usize,
    ) -> (NodeHandle, f64) {
        let Some(node) = self.acquired_node(current) else {
            return (current, distance);
        };
        let candidate = self.greedy_acquired(d, Candidate::new(distance, node), level);
        (self.candidate_node(candidate).handle(), candidate.distance)
    }
    fn greedy_acquired<S: MutableVectorSource + ?Sized>(
        &self, d: &SourceDistance<'_, S>, current: Candidate, level: usize,
    ) -> Candidate {
        if self.captured.is_some() { self.greedy_acquired_typed::<_, true, false>(d, current, level) }
        else { self.greedy_acquired_typed::<_, false, false>(d, current, level) }
    }
    /// Search-only upper-layer descent that, on a live view, also steps
    /// through nodes removed after this reader was admitted
    /// (`removed_bridge`), as the layer-0 beam does. The result may be such
    /// a node: the plain beam starts from an acquired candidate and never
    /// returns it. Construction keeps `greedy_acquired` (a removed node must
    /// not become an insert's neighbour seed).
    fn greedy_acquired_bridged<S: MutableVectorSource + ?Sized>(
        &self, d: &SourceDistance<'_, S>, current: Candidate, level: usize,
    ) -> Candidate {
        if self.captured.is_some() { self.greedy_acquired_typed::<_, true, false>(d, current, level) }
        else { self.greedy_acquired_typed::<_, false, true>(d, current, level) }
    }
    fn greedy_acquired_typed<S: MutableVectorSource + ?Sized, const CAPTURED: bool, const BRIDGE: bool>(
        &self, d: &SourceDistance<'_, S>, mut current: Candidate, level: usize,
    ) -> Candidate {
        loop {
            let mut improved = false;
            let Some(links) = self.candidate_node_typed::<CAPTURED>(current).links_typed::<CAPTURED>(level) else {
                break;
            };
            for &neighbor in links {
                let node = self.current_node_typed::<CAPTURED>(neighbor)
                    .or_else(|| if BRIDGE && !CAPTURED { self.removed_bridge(neighbor) } else { None });
                let Some(row) = node.and_then(|node| d.resolve_node(node)) else {
                    continue;
                };
                let distance = d.distance_to(&row);
                if distance < current.distance {
                    current = Candidate::new(distance, row.node);
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }
    /// Match Vespa's search_layer policy with the captured graph-node bound.
    /// Source/document IDs need not equal reusable graph IDs, so a document
    /// filter's size never truncates this bound (the MULTI wrapper behavior).
    fn layer_visited(
        &self,
        level: usize,
        ef: usize,
        filter: Option<FilterEstimate>,
    ) -> VisitedTracker {
        let degree = if level == 0 { 2 * self.m } else { self.m };
        // Graph bounds are u32. Larger public usize widths necessarily cover
        // that universe and may safely saturate before the estimate calculation.
        let width = u32::try_from(ef).unwrap_or(u32::MAX);
        let estimate = estimate_visited_nodes(degree as u32, self.len, width, filter);
        VisitedTracker::new(self.len, estimate)
    }
    /// Construction-only port of Vespa search_layer_helper (0521b006,
    /// hnsw_index.cpp:365-451). Keep the entire best-neighbor heap across
    /// descent levels, rebuild only candidates/visited, and tighten the
    /// distance bound only after overflow, as the Plain layer search does.
    fn construction_beam<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        level: usize,
        ef: usize,
        s: &mut MutableScratch,
    ) {
        // A retained per-thread bitmap (no per-layer allocation, zeroing or
        // hashing); Vespa sizes its per-layer tracker from this estimate.
        visited::with_layer_visited(self.len, |visited| self.construction_beam_typed(d, level, ef, s, visited));
    }
    fn construction_beam_typed<S: MutableVectorSource + ?Sized, V: visited::LayerVisited>(
        &self,
        d: &SourceDistance<'_, S>,
        level: usize,
        ef: usize,
        s: &mut MutableScratch,
        visited: &mut V,
    ) {
        s.cand.clear();
        for &entry in &s.top {
            if entry.id < self.len {
                s.cand.push(Reverse(entry));
                visited.insert(entry.id);
            }
        }
        let mut limit = f64::MAX;
        let mut neighbors = Vec::with_capacity(if level == 0 { 2 * self.m } else { self.m });
        // A code plane (live reads): score each neighbourhood by node id, as
        // plain search does, instead of resolving each neighbour's source row.
        let by_id = self.captured.is_none() && d.scores_by_id();
        let (mut ids, mut scores, mut distances) = (Vec::new(), Vec::new(), Vec::new());
        while let Some(Reverse(candidate)) = s.cand.pop() {
            if candidate.distance as f64 > limit {
                break;
            }
            let Some(links) = self.candidate_node(candidate).links(level) else {
                continue;
            };
            // Overlap the per-neighbour misses as in beam_acquired_typed.
            // SAFETY: snapshot admission validated this immutable table/guard pair.
            let table = unsafe { self.control.nodes.read_prevalidated(&self.guard) };
            for &neighbor in links {
                if by_id {
                    if visited.state(neighbor) == 0 { table.prefetch_hot(neighbor); d.prefetch_id(neighbor); }
                } else {
                    table.prefetch(neighbor);
                }
            }
            neighbors.clear();
            for &neighbor in links {
                let Some(node) = self.current_node(neighbor) else {
                    continue;
                };
                if !visited.insert(neighbor) {
                    continue;
                }
                if by_id {
                    // `resolve_node`'s visibility check, in the dense copy.
                    if self.visible_dense(node) { neighbors.push(node); }
                    continue;
                }
                d.source.prefetch_node(node.handle(), node.key());
                neighbors.push(node);
            }
            distances.clear();
            if by_id {
                d.distances_by_id(&neighbors, &mut ids, &mut scores, &mut distances);
            } else {
                for &node in &neighbors {
                    d.source.prefetch_row(node.handle(), node.key());
                }
                for &node in &neighbors {
                    let Some(row) = d.resolve_node(node) else {
                        continue;
                    };
                    distances.push((row.node, d.distance_to(&row)));
                }
            }
            for &(node, distance) in &distances {
                if !distance.is_finite() || distance as f64 >= limit {
                    continue;
                }
                let next = Candidate::new(distance, node);
                s.cand.push(Reverse(next));
                s.top.push(next);
                while s.top.len() > ef {
                    s.top.pop();
                    limit = s.top.peek().expect("positive construction width").distance as f64;
                }
            }
        }
        s.out.clear();
        s.out.extend(s.top.iter().copied());
        s.out.sort_unstable();
    }
    #[allow(clippy::too_many_arguments)]
    fn beam<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        entry: NodeHandle,
        distance: f64,
        level: usize,
        ef: usize,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
    ) {
        let Some(node) = self.acquired_node(entry) else {
            s.out.clear();
            return;
        };
        self.beam_acquired(
            d,
            Candidate::new(distance, node),
            level,
            ef,
            None,
            VespaPlainParams { prefetch_tensors: true, ..Default::default() },
            accept,
            s,
        )
    }
    fn beam_acquired<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        entry: Candidate,
        level: usize,
        ef: usize,
        filter: Option<FilterEstimate>,
        params: VespaPlainParams,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
    ) {
        // A retained per-thread bitmap (no per-layer allocation, zeroing of
        // `len / 8` bytes, or hashing). The filter estimate only sized
        // Vespa's per-layer tracker; marks and bounds are the same.
        let _ = filter;
        visited::with_layer_visited(self.len, |visited| if self.captured.is_some() {
            self.beam_acquired_typed::<_, _, true, false>(d, entry, level, ef, params, accept, s, visited)
        } else {
            if d.scores_by_id() {
                self.beam_acquired_typed::<_, _, false, true>(d, entry, level, ef, params, accept, s, visited)
            } else {
                self.beam_acquired_typed::<_, _, false, false>(d, entry, level, ef, params, accept, s, visited)
            }
        });
    }
    /// `BY_ID`: the bound scorer reads a code plane (live reads only): the
    /// neighbourhood is scored by node id. A separate instantiation, so the
    /// row path's code is unchanged.
    fn beam_acquired_typed<S: MutableVectorSource + ?Sized, V: visited::LayerVisited, const CAPTURED: bool, const BY_ID: bool>(
        &self,
        d: &SourceDistance<'_, S>,
        entry: Candidate,
        level: usize,
        ef: usize,
        params: VespaPlainParams,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
        visited: &mut V,
    ) {
        s.cand.clear();
        s.top.clear();
        if params.expired() { s.out.clear(); return; }
        let Some(first_row) = d.resolve_node(self.candidate_node_typed::<CAPTURED>(entry)) else {
            s.out.clear();
            return;
        };
        visited.insert(entry.id);
        let first = entry;
        s.cand.push(Reverse(first));
        if (CAPTURED || !self.candidate_removed(first)) && accept(first_row.id, first_row.node.key()) {
            s.top.push(first);
        }
        // Pinned search_layer_helper starts unbounded even if the seed fills
        // the result heap. Only an actual overflow establishes a cutoff.
        let mut limit = f64::MAX;
        let mut neighbors = Vec::with_capacity(if level == 0 { 2 * self.m } else { self.m });
        let mut rows = Vec::with_capacity(neighbors.capacity());
        // A code plane: score each neighbourhood by node id.
        let by_id = BY_ID;
        let (mut ids, mut scores) = (Vec::new(), Vec::new());
        while let Some(Reverse(candidate)) = s.cand.pop() {
            if candidate.distance > (1.0 + params.exploration_slack) * limit {
                break;
            }
            let Some(links) = self.candidate_node_typed::<CAPTURED>(candidate).links_typed::<CAPTURED>(level) else {
                continue;
            };
            // Each neighbour is three dependent misses (node table, source
            // directory, row). Issue every node-table line first, then every
            // directory entry, then every row, so a neighbourhood costs about
            // one miss latency per stage instead of one per neighbour.
            // SAFETY: snapshot admission validated this immutable table/guard pair.
            let table = unsafe { self.control.nodes.read_prevalidated(&self.guard) };
            for &neighbor in links {
                if visited.state(neighbor) == 0 {
                    // By id: the visibility check reads only the levels word
                    // and the dense publication, and the codes' address needs
                    // no node record, so the codes load with them.
                    if by_id { table.prefetch_hot(neighbor); d.prefetch_id(neighbor); }
                    else { table.prefetch(neighbor); }
                }
            }
            // Stage metadata and prefetch source-directory entries first.
            // Tensor prefetch remains an independent option.
            neighbors.clear();
            for &neighbor in links {
                if visited.state(neighbor) != 0 {
                    continue;
                }
                let found = if by_id { self.current_visible_dense::<CAPTURED>(neighbor) }
                    else { self.current_node_typed::<CAPTURED>(neighbor).filter(|n| self.visible(*n)) };
                let Some(node) = found else {
                    // A live reader may hold links read before a concurrent
                    // removal's rewrite: keep traversing through the removed
                    // node (frontier only, never the result heap).
                    if !CAPTURED {
                        if let Some(bridge) = self.removed_bridge(neighbor) {
                            if visited.insert(neighbor) {
                                if let Some(row) = d.resolve_node(bridge) {
                                    let distance = d.distance_to(&row);
                                    if distance.is_finite() && distance < (1.0 + params.exploration_slack) * limit {
                                        s.cand.push(Reverse(Candidate::new(distance, row.node)));
                                    }
                                }
                            }
                        }
                    }
                    continue;
                };
                if !visited.insert(neighbor) { continue; }
                note_visited(1);
                if !by_id { d.source.prefetch_node(node.handle(), node.key()); }
                neighbors.push(node);
            }
            if by_id {
                d.visit_ids(&neighbors, &mut ids, &mut scores, ef, &mut limit, params.exploration_slack, accept, s);
                if params.expired() { break; }
                continue;
            }
            for &node in &neighbors {
                d.source.prefetch_row(node.handle(), node.key());
            }
            rows.clear();
            for &node in &neighbors {
                if let Some(row) = d.resolve_node(node) { rows.push(row); }
            }
            if params.prefetch_tensors {
                for row in &rows {
                    prefetch(row.row.as_ptr() as *const u8, row.row.len() * 4);
                }
            }
            // Prefetch the full neighborhood before distance evaluation, as in
            // Vespa's layer helper; batching only changes distance arithmetic.
            if let Some(distances) = d.parallel_distances(&rows, true) {
                for (row, distance) in rows.iter().zip(distances) {
                    if !distance.is_finite() { continue; }
                    let candidate = Candidate::new(distance, row.node);
                    if distance < (1.0 + params.exploration_slack) * limit {
                        s.cand.push(Reverse(candidate));
                        if distance < limit && { let ok = accept(row.id, row.node.key()); if !ok { note_rejected(1); } ok } {
                            s.top.push(candidate);
                            while s.top.len() > ef {
                                s.top.pop();
                                limit = s.top.peek().expect("positive query width").distance;
                            }
                        }
                    }
                }
                if params.expired() { break; }
                continue;
            }
            let mut batch = [None; 4];
            let mut count = 0;
            for &row in &rows {
                batch[count] = Some(row);
                count += 1;
                if count == 4 {
                    d.visit_batch(&batch, count, ef, &mut limit, params.exploration_slack, accept, s);
                    count = 0;
                }
            }
            if count > 0 {
                d.visit_batch(&batch, count, ef, &mut limit, params.exploration_slack, accept, s);
            }
            if params.expired() { break; }
        }
        s.out.clear();
        s.out.extend(s.top.drain());
        s.out.sort_unstable();
    }
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
    ) -> MutableSearchResult {
        self.search_with_handles(query, k, ef, source, |_, key| accept(key))
    }
    /// Compatibility path without GlobalFilter statistics. The predicate stays
    /// authoritative, but its selectivity cannot be inferred from a closure.
    /// Filtered callers needing the pinned visited policy should use
    /// search_with_filter_estimate with their precomputed document size/count.
    pub fn search_with_handles(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
    ) -> MutableSearchResult {
        self.search_with_handles_and_filter(query, k, ef, source, accept, None)
    }
    /// Precomputed filter statistics must describe this exact accept predicate
    /// over the source document universe. No scan/count is performed per query.
    pub fn search_with_filter_estimate(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
        estimate: VespaFilterEstimate,
    ) -> Result<MutableSearchResult> {
        if estimate.count > estimate.size {
            return Err(failure("filter count exceeds document filter size"));
        }
        self.control.check_readable()?;
        self.search_with_execution(
            query,
            k,
            ef,
            source,
            |_, key| accept(key),
            Some(FilterEstimate {
                size: estimate.size,
                count: estimate.count,
            }),
            VespaPlainParams { prefetch_tensors: true, ..Default::default() },
            None,
        )
    }
    fn search_with_handles_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
        filter: Option<FilterEstimate>,
    ) -> MutableSearchResult {
        self.search_with_execution(query, k, ef, source, accept, filter,
            VespaPlainParams { prefetch_tensors: true, ..Default::default() }, None)
            .expect("infallible compatibility search requires a readable mutable graph")
    }
    fn search_with_execution(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
        filter: Option<FilterEstimate>,
        params: VespaPlainParams,
        // `accept` is exactly this bitmap of node ids (see `dense_beam`).
        node_filter: Option<&[u64]>,
    ) -> Result<MutableSearchResult> {
        self.control.check_readable()?;
        if k == 0 || query.len() != self.dim as usize || !self.similarity.valid_row(query) {
            return Ok(MutableSearchResult::default());
        }
        let d = SourceDistance::for_query(self, source, query);
        let Some(current) = self.available_entry(&d) else {
            self.control.check_readable()?;
            return Ok(MutableSearchResult {
                entry_fallback_scanned: d.entry_scanned.get(),
                missing_source_lookups: d.missing_source.get(),
                ..Default::default()
            });
        };
        let distance = d.dist(current);
        // Node may have been removed since resolving entry; the pinned memory
        // remains safe, but the slot pointer is allowed to be empty now.
        let top = self.entry_level(current);
        let mut acquired = Candidate::new(
            distance,
            self.acquired_node(current).expect("captured entry"),
        );
        for level in (1..=top).rev() {
            acquired = match self.dense_greedy(&d, acquired, level) {
                Some(candidate) => candidate,
                None => self.greedy_acquired_bridged(&d, acquired, level),
            };
        }
        let hits = with_mutable_scratch(|s| {
            if !self.dense_beam(&d, acquired, ef.max(k), params, &accept, node_filter, s) {
                self.beam_acquired(&d, acquired, 0, ef.max(k), filter, params, &accept, s);
            }
            // A code beam's own scores are the codes' result scores (the
            // same entries and scorer): no second read of each hit's row.
            let scores = (s.out_scores.len() == s.out.len()).then_some(&s.out_scores);
            let mut hits: Vec<_> = s
                .out
                .iter()
                .enumerate()
                .take(k.saturating_add(k / 2).saturating_add(4))
                .filter_map(|(i, candidate)| {
                    let row = d.resolve_id(candidate.id)?;
                    if !accept(row.id, row.node.key()) {
                        return None;
                    }
                    d.operations.set(d.operations.get() + 1);
                    d.final_score_ops.set(d.final_score_ops.get() + 1);
                    let score = match scores { Some(scores) => scores[i], None => d.result_score(&row)? };
                    Some((row.node.key(), score))
                })
                .collect();
            hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            hits.truncate(k);
            hits
        });
        self.control.check_readable()?;
        let vector_ops = d.operations.get();
        add_vector_ops(vector_ops);
        Ok(MutableSearchResult {
            hits,
            vector_ops,
            final_score_ops: d.final_score_ops.get(),
            entry_fallback_scanned: d.entry_scanned.get(),
            missing_source_lookups: d.missing_source.get(),
        })
    }
}
#[derive(Clone, Copy)]
struct Resolved<'a> {
    id: NodeHandle,
    node: NodeView<'a>,
    row: &'a [f32],
}
struct SourceDistance<'a, S: ?Sized> {
    view: &'a MutableHnswSnapshot,
    source: &'a S,
    query: &'a [f32],
    query_norm_sq: f64,
    inverse_query_norm: f32,
    metric: Metric,
    encoded: Option<Box<dyn MutableQueryScorer + 'a>>,
    _query_work: adaptive::QueryWork,
    operations: Cell<u64>,
    final_score_ops: Cell<u64>,
    entry_scanned: Cell<u64>,
    missing_source: Cell<u64>,
}
impl<'a, S: MutableVectorSource + ?Sized> SourceDistance<'a, S> {
    fn new(view: &'a MutableHnswSnapshot, source: &'a S, query: &'a [f32]) -> Self {
        let n = if cfg!(feature = "mutable-hnsw-vespa-angular") && view.similarity == Similarity::Cosine {
            f64::from(dot(query, query))
        } else {
            0.0
        };
        Self {
            view,
            source,
            query,
            query_norm_sq: n,
            inverse_query_norm: if !cfg!(feature = "mutable-hnsw-vespa-angular") && view.similarity == Similarity::Cosine {
                let norm = norm(query); if norm > 0.0 { 1.0 / norm } else { 0.0 }
            } else { 0.0 },
            metric: view.similarity.metric(),
            encoded: None,
            _query_work: adaptive::QueryWork::new(),
            operations: Cell::new(0),
            final_score_ops: Cell::new(0),
            entry_scanned: Cell::new(0),
            missing_source: Cell::new(0),
        }
    }
    fn for_query(view: &'a MutableHnswSnapshot, source: &'a S, query: &'a [f32]) -> Self {
        let mut distance = Self::new(view, source, query);
        distance.encoded = source.bind_query(query);
        distance
    }
    #[inline(always)]
    fn resolve(&self, id: NodeHandle) -> Option<Resolved<'a>> {
        self.resolve_node(self.view.acquired_node(id)?)
    }
    #[inline(always)]
    fn resolve_id(&self, slot: u32) -> Option<Resolved<'a>> {
        self.resolve_node(self.view.current_node(slot)?)
    }
    #[inline(always)]
    fn resolve_node(&self, node: NodeView<'a>) -> Option<Resolved<'a>> {
        if !self.view.visible(node) {
            return None;
        }
        let id = node.handle();
        // A bound quantized scorer reads the row's codes itself: fetching its
        // f32 (for a shed row, a lookup in the vector store) would be wasted.
        // A missing row then shows as a missing score (`encoded_score`).
        if self.encoded.is_some() {
            return Some(Resolved { id, node, row: &[] });
        }
        let row = self.source.get_node(id, node.key());
        let Some(row) = row.filter(|row| row.len() == self.view.dim as usize) else {
            self.missing_source.set(self.missing_source.get() + 1);
            return None;
        };
        Some(Resolved { id, node, row })
    }
    #[inline(always)]
    fn from_dot(&self, row: &Resolved<'_>, product: f32) -> f64 {
        match self.metric {
            Metric::Dot => -f64::from(product),
            Metric::Cos if cfg!(feature = "mutable-hnsw-vespa-angular") => angular_distance(self.query_norm_sq, row.row, product),
            Metric::Cos => {
                let inverse_norm = row.node.inverse_norm();
                f64::from(if inverse_norm == 0.0 || self.inverse_query_norm == 0.0 { 1.0 }
                    else { 1.0 - product * self.inverse_query_norm * inverse_norm })
            },
            Metric::L2 | Metric::Packed => unreachable!(),
        }
    }
    fn dist(&self, id: NodeHandle) -> f64 {
        let Some(row) = self.resolve(id) else {
            return f64::INFINITY;
        };
        self.distance_to(&row)
    }
    fn encoded_score(&self, row: &Resolved<'_>) -> Option<f32> {
        let score = self.encoded.as_ref()?.score(row.id, row.node.key());
        if score.is_none_or(|v| !v.is_finite() || v < 0.0) {
            self.missing_source.set(self.missing_source.get() + 1);
            return None;
        }
        score
    }
    fn result_score(&self, row: &Resolved<'_>) -> Option<f32> {
        if self.encoded.is_some() { self.encoded_score(row) }
        else { Some(self.view.similarity.score(self.query, row.row)) }
    }
    fn encoded_distance(&self, row: &Resolved<'_>) -> f64 {
        let Some(score) = self.encoded_score(row) else { return f64::INFINITY; };
        self.distance_of_score(score)
    }
    /// The traversal distance of an encoded score (see `encoded_distance`).
    #[inline]
    fn distance_of_score(&self, score: f32) -> f64 {
        let score = f64::from(score);
        match self.view.similarity {
            Similarity::Cosine => 2.0 - 2.0 * score,
            Similarity::DotProduct => 1.0 - 2.0 * score,
            Similarity::L2 => if score > 0.0 { score.recip() - 1.0 } else { f64::MAX },
            Similarity::MaxInnerProduct => if score >= 1.0 { 1.0 - score }
                else if score > 0.0 { score.recip() - 1.0 } else { f64::MAX },
            // Packed rows are never encoded.
            _ => f64::INFINITY,
        }
    }
    #[inline(always)]
    fn distance_to(&self, row: &Resolved<'_>) -> f64 {
        self.operations.set(self.operations.get() + 1);
        if self.encoded.is_some() {
            self.encoded_distance(row)
        } else {
            self.distance_value(row)
        }
    }
    /// The raw-row distance (not counted).
    #[inline(always)]
    fn distance_value(&self, row: &Resolved<'_>) -> f64 {
        match self.metric {
            Metric::L2 => f64::from(l2_sq(self.query, row.row)),
            Metric::Packed => f64::from(self.view.similarity.distance(self.query, row.row)),
            Metric::Dot | Metric::Cos => self.from_dot(row, dot(self.query, row.row)),
        }
    }
    fn parallel_distances(&self, rows: &[Resolved<'_>], batched: bool) -> Option<Vec<f64>> {
        // Only immutable tensor slices/scalars cross threads, never graph views,
        // source callbacks, counters, filters or the caller's frontier state.
        if self.encoded.is_some() || self.metric == Metric::Packed || !adaptive::eligible(rows.len(), self.query.len()) { return None; }
        let values: Vec<_> = rows.iter().map(|r| (r.row, r.node.inverse_norm())).collect();
        let query = self.query;
        let metric = self.metric;
        let inverse_query_norm = self.inverse_query_norm;
        let query_norm_sq = self.query_norm_sq;
        let result = adaptive::calculate(&values, query.len(), |chunk| {
            let convert = |row: (&[f32], f32), product: f32| match metric {
                Metric::Dot => -f64::from(product),
                Metric::Cos if cfg!(feature = "mutable-hnsw-vespa-angular") => angular_distance(query_norm_sq, row.0, product),
                Metric::Cos => f64::from(if row.1 == 0.0 || inverse_query_norm == 0.0 { 1.0 }
                    else { 1.0 - product * inverse_query_norm * row.1 }),
                Metric::L2 | Metric::Packed => unreachable!(),
            };
            let mut out = Vec::with_capacity(chunk.len());
            for group in chunk.chunks(4) {
                if batched && group.len() == 4 && metric != Metric::L2 {
                    let products = dot_four(query, [group[0].0, group[1].0, group[2].0, group[3].0]);
                    for i in 0..4 { out.push(convert(group[i], products[i])); }
                } else {
                    for &row in group { out.push(if metric == Metric::L2 { f64::from(l2_sq(query, row.0)) }
                        else { convert(row, dot(query, row.0)) }); }
                }
            }
            out
        })?;
        self.operations.set(self.operations.get() + rows.len() as u64);
        Some(result)
    }

    /// The bound scorer reads codes by node id (a code plane): the traversal
    /// scores a neighbourhood by id, no source row lookup per neighbour.
    fn scores_by_id(&self) -> bool {
        self.encoded.as_ref().is_some_and(|e| e.score_ids(&[], &mut []))
    }
    #[inline(always)]
    fn prefetch_id(&self, id: u32) {
        if let Some(e) = &self.encoded { e.prefetch_id(id); }
    }
    /// Distances of `nodes` (visible, newly visited) scored by id into
    /// `out`, as `distance_to` of each resolved node would give them: a node
    /// without a plane entry is scored from its row, a missing score is left
    /// out (counted as a missing source). `ids` and `scores` are scratch.
    fn distances_by_id<'n>(&self, nodes: &[NodeView<'n>], ids: &mut Vec<u32>, scores: &mut Vec<f32>, out: &mut Vec<(NodeView<'n>, f64)>) {
        let Some(encoded) = &self.encoded else { return };
        ids.clear();
        ids.extend(nodes.iter().map(|n| n.id));
        scores.clear();
        scores.resize(ids.len(), f32::NAN);
        encoded.score_ids(ids, scores);
        self.operations.set(self.operations.get() + nodes.len() as u64);
        for (&node, &score) in nodes.iter().zip(scores.iter()) {
            let distance = if score.is_nan() {
                let id = node.handle();
                self.encoded_distance(&Resolved { id, node, row: &[] })
            } else if score < 0.0 || !score.is_finite() {
                self.missing_source.set(self.missing_source.get() + 1);
                continue;
            } else {
                self.distance_of_score(score)
            };
            out.push((node, distance));
        }
    }
    /// Distances of nodes `nodes` (acquired, visible, newly visited) scored
    /// by id, `scores` the scratch; then each admitted as `visit_batch`
    /// admits a row. A node without a plane entry is scored from its row.
    #[allow(clippy::too_many_arguments)]
    fn visit_ids(
        &self,
        nodes: &[NodeView<'_>],
        ids: &mut Vec<u32>,
        scores: &mut Vec<f32>,
        ef: usize,
        limit: &mut f64,
        exploration_slack: f64,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
    ) {
        let Some(encoded) = &self.encoded else { return };
        ids.clear();
        ids.extend(nodes.iter().map(|n| n.id));
        scores.clear();
        scores.resize(ids.len(), f32::NAN);
        encoded.score_ids(ids, scores);
        self.operations.set(self.operations.get() + nodes.len() as u64);
        for (&node, &score) in nodes.iter().zip(scores.iter()) {
            let distance = if score.is_nan() {
                match self.resolve_node(node) { Some(row) => self.encoded_distance(&row), None => continue }
            } else if score < 0.0 || !score.is_finite() {
                self.missing_source.set(self.missing_source.get() + 1);
                continue;
            } else {
                self.distance_of_score(score)
            };
            self.admit(node, distance, ef, limit, exploration_slack, accept, s);
        }
    }
    /// `visit_batch`'s admission of one scored node.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn admit(
        &self,
        node: NodeView<'_>,
        distance: f64,
        ef: usize,
        limit: &mut f64,
        exploration_slack: f64,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
    ) {
        if !distance.is_finite() {
            return;
        }
        let candidate = Candidate::new(distance, node);
        // Exploration slack expands only the frontier, not result admission.
        // Pinned search_layer_helper cpp:435-443.
        if distance < (1.0 + exploration_slack) * *limit {
            s.cand.push(Reverse(candidate));
            if distance < *limit && { let ok = accept(node.handle(), node.key()); if !ok { note_rejected(1); } ok } {
                s.top.push(candidate);
                while s.top.len() > ef {
                    s.top.pop();
                    *limit = s.top.peek().expect("positive query width").distance;
                }
            }
        }
    }
    fn visit_batch(
        &self,
        rows: &[Option<Resolved<'_>>; 4],
        count: usize,
        ef: usize,
        limit: &mut f64,
        exploration_slack: f64,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        s: &mut MutableScratch,
    ) {
        let mut distances = [0.0; 4];
        if self.encoded.is_some() {
            for i in 0..count { distances[i] = self.encoded_distance(&rows[i].unwrap()); }
        } else if count == 4 && self.metric.uses_dot() {
            let products = dot_four(self.query, rows.map(|row| row.unwrap().row));
            for i in 0..4 {
                distances[i] = self.from_dot(&rows[i].unwrap(), products[i]);
            }
        } else {
            for i in 0..count {
                let row = rows[i].unwrap();
                distances[i] = self.distance_value(&row);
            }
        }
        self.operations.set(self.operations.get() + count as u64);
        for i in 0..count {
            let row = rows[i].unwrap();
            let distance = f64::from(distances[i]);
            if !distance.is_finite() {
                continue;
            }
            let candidate = Candidate::new(distance, row.node);
            // Exploration slack expands only the frontier, not result admission.
            // Pinned search_layer_helper cpp:435-443.
            if distance < (1.0 + exploration_slack) * *limit {
                s.cand.push(Reverse(candidate));
                if distance < *limit && { let ok = accept(row.id, row.node.key()); if !ok { note_rejected(1); } ok } {
                    s.top.push(candidate);
                    while s.top.len() > ef {
                        s.top.pop();
                        *limit = s.top.peek().expect("positive query width").distance;
                    }
                }
            }
        }
    }
}
fn checked_row(
    source: &(impl MutableVectorSource + ?Sized),
    handle: NodeHandle,
    key: u64,
    dim: u32,
    similarity: Similarity,
) -> Result<&[f32]> {
    let row=source.get_node(handle,key);
    source.execution_result()?;
    let row=row.ok_or_else(||failure("missing vector"))?;
    if row.len() != dim as usize || !similarity.valid_row(row) {
        return Err(failure("invalid vector"));
    }
    Ok(row)
}
fn pair_distance(
    view: &MutableHnswSnapshot,
    source: &(impl MutableVectorSource + ?Sized),
    similarity: Similarity,
    a: NodeHandle,
    b: NodeHandle,
) -> Option<f32> {
    let a = source.get_node(a, view.node(a)?.key())?;
    let b = source.get_node(b, view.node(b)?.key())?;
    if a.len() != view.dim as usize || b.len() != a.len() {
        return None;
    }
    let d = similarity.distance(a, b);
    d.is_finite().then_some(d)
}
/// `anchor`: the new node's approximate row, when the candidates'
/// distances are code estimates (see `prepare_connections_at_level`): each
/// candidate is then weighed by its approximate distance to it.
fn select_diverse<'s>(
    view: &MutableHnswSnapshot,
    source: &'s (impl MutableVectorSource + ?Sized),
    similarity: Similarity,
    candidates: &[Candidate],
    anchor: Option<&[f32]>,
    max: usize,
    selected: &mut Vec<Candidate>,
) {
    selected.clear();
    // The selected candidates' rows, looked up once each (a selected
    // candidate always had one): the diversity check compares every later
    // candidate against all of them.
    let mut rows: Vec<&'s [f32]> = Vec::with_capacity(max.min(candidates.len()));
    for &candidate in candidates {
        if selected.len() >= max {
            break;
        }
        let node = view.candidate_node(candidate);
        let Some(row) = source.get_node_approx(node.handle(), node.key()) else {
            continue;
        };
        // Vespa's have_closer_distance binds the candidate once for the
        // complete diversity check, including preparation (not only pruning).
        let bound = BoundMutationDistance::new(row, similarity);
        let own = match anchor {
            Some(anchor) if anchor.len() == row.len() => bound.distance_value(anchor),
            _ => candidate.distance,
        };
        if rows.iter().all(|v| row.len() != v.len() || bound.distance_value(v) >= own) {
            selected.push(candidate);
            rows.push(row);
        }
    }
}
fn put32(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn put64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
/// Preserve admission failures before generic mutation diagnostics stringify them.
fn failure_error<E: std::fmt::Display + 'static>(error: E) -> RusticError {
    let typed = &error as &dyn std::any::Any;
    if typed.downcast_ref::<array_store::StoreError>() == Some(&array_store::StoreError::MemoryLimit) {
        return RusticError::Memory("mutable vector resident allocation exceeds shared budget".into());
    }
    if typed.downcast_ref::<node_table::NodeTableError>() == Some(&node_table::NodeTableError::MemoryLimit) {
        return RusticError::Memory("mutable node table exceeds shared budget".into());
    }
    if let Some(RusticError::Memory(reason)) = typed.downcast_ref::<RusticError>() {
        return RusticError::Memory(reason.clone());
    }
    failure(&error.to_string())
}
fn failure(detail: &str) -> RusticError {
    RusticError::Segment(format!("mutable hnsw: {detail}"))
}
fn corrupt(detail: &str) -> RusticError {
    RusticError::Corrupted(format!("mutable hnsw: {detail}"))
}
struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Decoder<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .filter(|&n| n <= self.bytes.len())
            .ok_or_else(|| corrupt("truncated"))?;
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod adapter_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod fidelity_order_tests;
#[cfg(test)]
mod lazy_identity_tests;
#[cfg(test)]
mod visited_integration_tests;

#[cfg(test)]
mod quantized_tests;

#[cfg(test)]
mod kernel_tests;
