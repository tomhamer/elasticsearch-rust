//! Sequenced identity reservation and parallel preparation for a shard field.
//! This is an ingestion integration primitive, not ES document publication.
//! Callers must transfer a completed batch under their document/WAL fence.
use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use parking_lot::Mutex;
use rayon::prelude::*;
use crate::error::{Result, RusticError};
use super::{HnswParams, Similarity};
use super::hnsw::{MutableHnsw, MutableVectorOwner, NodeHandle};
use super::store::{PointReservation, VectorStore};

pub(crate) mod diagnostics;
mod point_handles;
mod projection_cache;
use projection_cache::{ProjectionCache, ProjectionKey};
use point_handles::PointHandles;

fn identity_map(graph: &super::hnsw::MutableHnswSnapshot) -> Result<PointHandles> {
    let mut identities = PointHandles::default();
    for identity in graph.visible_identities() { identities.extend(std::iter::once(identity))?; }
    Ok(identities)
}

/// A segment node map entry for a doc with no vector (never looked up).
const NO_VECTOR: u32 = u32::MAX;

fn empty_node_filter(nodes: u32) -> Result<crate::admission::Allocation<Vec<u64>>> {
    let words = (nodes as usize + 1).div_ceil(64);
    let memory=crate::admission::shared().reserve(words as u64*8+256,"mutable node filter")?;
    let mut filter = Vec::new();
    filter.try_reserve_exact(words).map_err(|_| RusticError::Memory("mutable node filter allocation".into()))?;
    filter.resize(words, 0);
    Ok(crate::admission::Allocation::new(filter,memory))
}

/// One sequencer/graph/source owner per vector field and shard copy. Text
/// segment IDs never enter this owner. Do not create one per memtable.
/// [`MutableIngestField::off_heap_bytes`]'s last measurement: when, of
/// which graph generation, and the bytes.
#[derive(Default)]
struct OffHeapCache {
    value: Option<(std::time::Instant, u64, (u64, u64))>,
    refreshing: bool,
}

/// Graphs up to this many slots are measured exactly on every change.
pub(crate) const OFF_HEAP_EXACT_SLOTS: u32 = 1 << 20;
/// A larger graph's stats are re-measured at most this often (in the background).
pub(crate) const OFF_HEAP_REFRESH: std::time::Duration = std::time::Duration::from_secs(300);

pub struct MutableIngestField {
    /// The last [`MutableIngestField::off_heap_bytes`] measurement.
    off_heap: parking_lot::Mutex<OffHeapCache>,
    store: std::sync::Weak<VectorStore>,
    params: HnswParams,
    field_id: u16,
    graph: Arc<MutableHnsw>,
    owner: Arc<MutableVectorOwner>,
    reserve: Mutex<()>,
    abandoned: Mutex<Vec<NodeHandle>>,
    inflight: AtomicUsize,
    /// First point of each admission reserved in this graph and not yet
    /// accepted or abandoned (graph keys are reserved in increasing order
    /// under `reserve`, but admissions are accepted in any order). A
    /// migration's catch-up copies only the points below the lowest one
    /// (see [`MutableIngestField::settled_frontier`]).
    admitting: Mutex<std::collections::BTreeSet<u64>>,
    identities: parking_lot::RwLock<PointHandles>,
    projection_cache: Mutex<ProjectionCache>,
    /// Bumped when a point leaves the graph while its document may still be
    /// live (an abandoned admission, a reconciliation, a migration's copy):
    /// cached segment projections are only invalidated by these, not by
    /// delete repair (a deleted document is masked in every view that could
    /// still map it).
    projection_epoch: AtomicU64,
    /// Queued document deletes: `(point, epoch)`, the epoch being the view
    /// generation that first excludes the document (see
    /// `VectorStore::delete_epoch`). FIFO: epochs are stamped from a
    /// monotonic clock, so the eligible deletes form a prefix.
    pending_deletes: Mutex<std::collections::VecDeque<(u64, u64)>>,
    extra_references: Mutex<std::collections::HashMap<u64, usize>>,
    /// Rows the newest durable graph checkpoint still references although
    /// the live graph removed their nodes (see [`CheckpointPins`]).
    checkpoint_pins: Mutex<CheckpointPins>,
    /// Graph insertions since this owner opened (staged writes and the
    /// open-time catch-up): a copy recovered from files inserts only what
    /// its shipped checkpoint lacked.
    graph_inserts: std::sync::atomic::AtomicU64,
    /// Set when an `index_options` migration retires this owner: from then
    /// on admissions, leases and deletes go to its successor (see
    /// [`MutableIngestField::retire_into`]).
    successor: parking_lot::RwLock<Option<Arc<MutableIngestField>>>,
    /// The owners this one replaced: writers still hold reservations they
    /// made, which this owner accepts as its own.
    predecessors: parking_lot::RwLock<Vec<usize>>,
}

/// Graph checkpoints are saved apart from text commits, and a commit may
/// reuse an older checkpoint file whose nodes reference durable f32 rows
/// (B9). A node the live graph removed after that checkpoint's capture is
/// still in the file, so the vector optimizer must keep its row until a
/// newer checkpoint (captured after the removal) is durable. Without any
/// checkpoint (durable or being captured) nothing references a row and
/// nothing is pinned.
#[derive(Default)]
struct CheckpointPins {
    /// Mutation generation of the newest durable checkpoint.
    durable: Option<u64>,
    /// Lowest generation of a checkpoint captured but not durable yet
    /// (cleared once a checkpoint at or above it is durable).
    capturing: Option<u64>,
    /// Removed point → the graph mutation generation that published its
    /// removal (in a checkpoint captured at generation `g` iff it is `> g`).
    removed: std::collections::HashMap<u64, u64>,
    /// The pinned points as handed to the optimizer, rebuilt only after
    /// `removed` changed (the optimizer asks once per segment per pass).
    snapshot: Option<Arc<std::collections::HashSet<u64>>>,
}

impl CheckpointPins {
    /// The oldest checkpoint generation some durable or pending file has.
    fn floor(&self) -> Option<u64> {
        match (self.durable, self.capturing) {
            (Some(durable), Some(capturing)) => Some(durable.min(capturing)),
            (durable, capturing) => durable.or(capturing),
        }
    }
}

/// Completed graph nodes which are still unpublished documents. Dropping the
/// batch queues their removal; cleanup errors are returned by the next admission
/// or explicit drain, never swallowed by a destructor.
pub(crate) struct StagedMutableRows {
    field: Arc<MutableIngestField>,
    reservation: Option<PointReservation>,
    handles: Vec<NodeHandle>,
    accepted: bool,
    /// This admission's first point while it is in `field.admitting`.
    first_point: Option<u64>,
}

impl Drop for StagedMutableRows {
    fn drop(&mut self) {
        if !self.accepted { self.field.abandoned.lock().append(&mut self.handles); }
        // Accepted (identities installed) or abandoned: settled.
        if let Some(point) = self.first_point.take() { self.field.admitting.lock().remove(&point); }
        self.field.inflight.fetch_sub(1, Ordering::AcqRel);
    }
}
impl StagedMutableRows {
    /// Only transfer after the caller has secured document ownership. This is
    /// not a WAL commit, refresh, or permission to expose rows to older views.
    pub(crate) fn accept(mut self) -> Result<(PointReservation, Vec<NodeHandle>)> {
        let field = self.field.clone();
        let mut handles = self.handles.clone();
        let points = self.points();
        let abandoned_points = points.clone();
        self.reservation.as_ref().expect("completed reservation").on_abandon(Arc::as_ptr(&field) as usize, move || {
            { let mut identities = field.identities.write();
              for point in abandoned_points.clone() { identities.remove(&point); } }
            field.projection_epoch.fetch_add(1, Ordering::AcqRel);
            field.abandoned.lock().append(&mut handles);
            // A migration copied these points into the successor's graph.
            if let Some(next) = field.successor() { next.forget_points(abandoned_points.clone()); }
        })?;
        self.field.identities.write().extend(points.zip(self.handles.iter().copied()))?;
        self.accepted = true;
        Ok((self.reservation.take().expect("completed reservation"), std::mem::take(&mut self.handles)))
    }
    pub(crate) fn points(&self) -> std::ops::Range<u64> {
        self.reservation.as_ref().expect("completed reservation").points()
    }
}

/// Query preparation pins graph/source together and translates the document
/// filter once. Reuse only while the caller retains the same document read
/// view. Preparing this filter is part of end-to-end query cost, not ANN time.
pub struct PreparedMutableFieldSearch<'a> {
    graph: super::hnsw::MutableHnswSnapshot,
    source: super::hnsw::MutableVectorRead<'a>,
    node_filter: Arc<crate::admission::Allocation<Vec<u64>>>,
    pub(crate) matched: usize,
}
impl PreparedMutableFieldSearch<'_> {
    pub fn search(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        deadline: Option<std::time::Instant>) -> Result<super::hnsw::MutableSearchResult>
    {
        if plan.route == super::vespa_policy::Route::Exact {
            return execute_prepared_exact(&self.graph, &self.source, &self.node_filter, query, plan.requested_target_hits as usize, deadline, plan.exact_codes);
        }
        // The exact document eligibility predicate was captured once during
        // preparation. Re-reading mutable ownership per candidate defeats the
        // GlobalFilter contract and contends across concurrent query workers.
        execute_ann(&self.graph, &self.source, query, plan, |handle, _| {
            let id = handle.slot() as usize + 1;
            self.node_filter.get(id / 64).is_some_and(|word| word & (1u64 << (id % 64)) != 0)
        }, deadline, Some(&self.node_filter))
    }
}

/// Exact execution shares the bounded search pool and competitive heap.
/// ES stable point keys, rather than recyclable graph slots, order score ties.
/// `codes`: score a quantized field's codes (its caller rescores the
/// gathered candidates with f32, as Lucene's exact search on a quantized
/// format and ES's rescore do) instead of reading every candidate's f32.
fn execute_prepared_exact(graph: &super::hnsw::MutableHnswSnapshot,
    source: &(impl super::hnsw::MutableVectorSource + Sync), node_filter: &[u64],
    query: &[f32], k: usize, deadline: Option<std::time::Instant>, codes: bool)
    -> Result<super::hnsw::MutableSearchResult>
{
    use super::exact_schedule::ExactTaskRanges;
    if query.len() != graph.dim() as usize || !graph.similarity().valid_row(query) {
        return Err(RusticError::InvalidQuery("invalid mutable exact query vector".into()));
    }
    if codes {
        if let Some(result) = execute_exact_codes_by_id(graph, source, node_filter, query, k, deadline)? {
            return Ok(result);
        }
    }
    // The filter's nodes only (a selective filter's few words of bits), not
    // every node of the graph.
    let mut exact_rows: Vec<(u64, NodeHandle)> = Vec::new();
    for (w, &word) in node_filter.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let id = w * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if let Some(row) = u32::try_from(id).ok().and_then(|id| graph.visible_identity(id)) { exact_rows.push(row); }
        }
    }
    exact_rows.sort_unstable_by_key(|row| row.0);
    if exact_rows.is_empty() || k == 0 { return Ok(Default::default()); }
    let end = usize::try_from(exact_rows.last().unwrap().0).ok()
        .and_then(|key| key.checked_add(1))
        .ok_or_else(|| RusticError::InvalidQuery("exact document range overflow".into()))?;
    let shared = super::exact_shared::SharedExactTopK::new(k);
    let execute = |parallel: bool| -> Result<u64> {
        let workers = if parallel { ExactTaskRanges::workers(rayon::current_num_threads(), exact_rows.len(), 1024) } else { 1 };
        let ranges = ExactTaskRanges::new(0..end, workers.saturating_mul(8));
        let admission = crate::admission::Context::capture();
        let run = |_| -> Result<u64> {
            let _admission = admission.enter();
            let mut operations = 0;
            // One bound scorer per worker (it caches per-epoch state).
            let scorer = if codes { source.bind_query(query) } else { None };
            while let Some(range) = ranges.claim() {
                let lo = exact_rows.partition_point(|row| row.0 < range.start as u64);
                let hi = exact_rows.partition_point(|row| row.0 < range.end as u64);
                if let Some(scorer) = &scorer {
                    // Rows are in key order, not memory order: each is a
                    // random miss. Prefetch the directory slot 16 rows ahead
                    // and the row itself 8 ahead (its slot is in by then).
                    for ordinal in lo..hi.min(lo + 16) { let (key, handle) = exact_rows[ordinal]; source.prefetch_node(handle, key); }
                    for ordinal in lo..hi.min(lo + 8) { let (key, handle) = exact_rows[ordinal]; source.prefetch_row(handle, key); }
                    for ordinal in lo..hi {
                        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
                            return Err(RusticError::InvalidQuery("mutable exact query deadline exceeded".into()));
                        }
                        if let Some(&(key, handle)) = exact_rows.get(ordinal + 16).filter(|_| ordinal + 16 < hi) { source.prefetch_node(handle, key); }
                        if let Some(&(key, handle)) = exact_rows.get(ordinal + 8).filter(|_| ordinal + 8 < hi) { source.prefetch_row(handle, key); }
                        let (key, handle) = exact_rows[ordinal];
                        let Some(score) = scorer.score(handle, key) else {
                            return Err(RusticError::Segment("mutable exact encountered a missing live source row".into()));
                        };
                        shared.offer(ordinal as u32, score);
                        operations += 1;
                    }
                    source.execution_result()?;
                    continue;
                }
                let mut failure = None;
                let rows = (lo..hi).map_while(|ordinal| {
                    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
                        failure = Some(RusticError::InvalidQuery("mutable exact query deadline exceeded".into()));
                        return None;
                    }
                    let (key, handle) = exact_rows[ordinal];
                    match source.get_node(handle, key) {
                        Some(row) => Some((ordinal as u32, row)),
                        None => {
                            failure = Some(RusticError::Segment("mutable exact encountered a missing live source row".into()));
                            None
                        }
                    }
                });
                operations += shared.scan(query, graph.similarity(), rows);
                source.execution_result()?;
                if let Some(error) = failure { return Err(error); }
            }
            Ok(operations)
        };
        if workers == 1 { run(0) } else {
            (0..workers).into_par_iter().map(run).try_reduce(|| 0, |a,b| Ok(a+b))
        }
    };
    // ES may hold shard/document read locks on a foreign Rayon worker.
    // Cross-pool install can steal a writer blocked on those same locks.
    let operations = if rayon::current_thread_index().is_some()
        && !crate::query::searcher::on_search_pool_worker() {
        execute(false)?
    } else {
        crate::query::searcher::in_search_pool(|| execute(true))?
    };
    Ok(super::hnsw::MutableSearchResult {
        hits: shared.finish().into_iter().map(|(ordinal, score)| (exact_rows[ordinal as usize].0, score)).collect(),
        vector_ops: operations, ..Default::default()
    })
}

/// Filtered nodes per worker below which an exact scan stays on fewer
/// threads (a worker's setup, a query quantization and a pool hop, costs
/// about as much as scoring this many codes).
const EXACT_MIN_PER_WORKER: usize = 8192;

/// Workers an exact scan of `matched` nodes borrows: its share of the cores
/// among the shard searches running now (`crate::search_load`: each shard
/// request of a search counts, this one's included), at most one per [`EXACT_MIN_PER_WORKER`]
/// nodes, and one while searches queue. A busy node scans on the calling
/// thread alone; an idle one spreads the scan over the idle cores.
pub(crate) fn exact_scan_workers(matched: usize) -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let max = *MAX.get_or_init(|| std::env::var("RUSTIC_KNN_EXACT_SCAN_WORKERS").ok().and_then(|v| v.parse().ok()).filter(|&v: &usize| v > 0).unwrap_or(usize::MAX));
    let cores = rayon::current_num_threads().max(1).min(max);
    let (queued, active) = crate::search_load();
    let share = if queued > 0 { 1 } else { (cores / active.max(1)).max(1) };
    share.min((matched / EXACT_MIN_PER_WORKER).max(1)).min(cores)
}

/// A worker's best `k` hits: (score, key), worst at the root; ties keep the
/// smaller key (ES orders equal scores by document).
struct ExactTop { k: usize, heap: std::collections::BinaryHeap<ExactHit> }
#[derive(Clone, Copy)]
struct ExactHit { score: f32, key: u64 }
impl PartialEq for ExactHit { fn eq(&self, other: &Self) -> bool { self.cmp(other) == std::cmp::Ordering::Equal } }
impl Eq for ExactHit {}
impl PartialOrd for ExactHit { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for ExactHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.total_cmp(&self.score).then(self.key.cmp(&other.key))
    }
}
impl ExactTop {
    fn new(k: usize) -> Self { Self { k, heap: std::collections::BinaryHeap::with_capacity(k + 1) } }
    #[inline]
    fn competitive(&self, score: f32) -> bool {
        self.heap.len() < self.k || self.heap.peek().is_some_and(|worst| score >= worst.score)
    }
    #[inline]
    fn offer(&mut self, score: f32, key: u64) {
        if self.k == 0 || score.is_nan() { return; }
        let hit = ExactHit { score, key };
        if self.heap.len() < self.k { self.heap.push(hit); }
        else if hit < *self.heap.peek().expect("full heap") { *self.heap.peek_mut().expect("full heap") = hit; }
    }
}

/// Exact search of a quantized live field by its code plane: the filter's
/// nodes in slot order (the plane's memory order), split over
/// [`exact_scan_workers`] workers by ranges of filter words, each binding the
/// query once, scoring visible nodes by id in batches (four rows per kernel
/// call where the codes allow) into its own top `k`; the tops merge at the
/// end. `None`: the source has no code plane (the caller's row path runs).
fn execute_exact_codes_by_id(graph: &super::hnsw::MutableHnswSnapshot,
    source: &(impl super::hnsw::MutableVectorSource + Sync), node_filter: &[u64],
    query: &[f32], k: usize, deadline: Option<std::time::Instant>)
    -> Result<Option<super::hnsw::MutableSearchResult>>
{
    use super::exact_schedule::ExactTaskRanges;
    let started = std::time::Instant::now();
    let Some(probe) = source.bind_query(query) else { return Ok(None) };
    if !probe.score_ids(&[], &mut []) { return Ok(None); }
    let matched: usize = node_filter.iter().map(|word| word.count_ones() as usize).sum();
    if matched == 0 || k == 0 || node_filter.is_empty() { return Ok(Some(Default::default())); }
    // Quantize the query once: score the first filtered node (its epoch's
    // quantization is then cached) and hand that state to every worker.
    if let Some(first) = node_filter.iter().enumerate().find(|(_, w)| **w != 0).map(|(i, w)| (i * 64) as u32 + w.trailing_zeros()) {
        if graph.visible_identity(first).is_some() { probe.score_ids(&[first], &mut [0.0]); }
    }
    let shared: super::hnsw::SharedQuantized = parking_lot::Mutex::new(probe.quantized_state().unwrap_or_default());
    drop(probe);
    let workers = exact_scan_workers(matched);
    // Ranges over the words that hold the filter's nodes, not the whole
    // node space: a filter correlated with insertion order (a range over
    // the oldest documents) sits in one end of it, and an even split of
    // all words gave nearly all of it to one task.
    let lo = node_filter.iter().position(|&w| w != 0).unwrap_or(0);
    let hi = node_filter.iter().rposition(|&w| w != 0).map_or(lo, |i| i + 1);
    let ranges = ExactTaskRanges::new(lo..hi, (workers * 16).min(hi - lo).max(1));
    let admission = crate::admission::Context::capture();
    const BATCH: usize = 256;
    let setup = started.elapsed();
    let first_worker = std::sync::OnceLock::<std::time::Duration>::new();
    let run = |_| -> Result<(ExactTop, u64)> {
        let _ = first_worker.set(started.elapsed());
        let _admission = admission.enter();
        let scorer = source.bind_query_shared(query, &shared)
            .ok_or_else(|| RusticError::Segment("mutable exact lost its code plane".into()))?;
        // Keyed by node id while scanning; the key (and the view's
        // visibility check) is read only for a competitive node.
        let mut top = ExactTop::new(k);
        let mut operations = 0u64;
        let mut ids: Vec<u32> = Vec::with_capacity(BATCH);
        let mut scores = vec![0f32; BATCH];
        let mut flush = |ids: &mut Vec<u32>, top: &mut ExactTop| -> Result<()> {
            let n = ids.len();
            if n == 0 { return Ok(()); }
            // Software-pipelined reads: each chunk's rows were asked for
            // `AHEAD` rows earlier, so their misses overlap the scoring of
            // the chunks before (a filter's rows are scattered over the
            // plane; asking for a whole batch at once overflows the
            // core's outstanding-miss slots and most requests are lost).
            const CHUNK: usize = 8;
            const AHEAD: usize = 24;
            for &id in &ids[..n.min(AHEAD)] { scorer.prefetch_id(id); }
            let mut c = 0;
            while c < n {
                let end = (c + CHUNK).min(n);
                for &id in &ids[(c + AHEAD).min(n)..(end + AHEAD).min(n)] { scorer.prefetch_id(id); }
                scorer.score_ids(&ids[c..end], &mut scores[c..end]);
                c = end;
            }
            for i in 0..n {
                let mut score = scores[i];
                if !score.is_nan() && !top.competitive(score) { continue; }
                // The visibility contract of the row path: only nodes this
                // view acquires become hits.
                let Some((key, handle)) = graph.visible_identity(ids[i]) else { continue };
                if score.is_nan() {
                    score = scorer.score(handle, key).ok_or_else(|| RusticError::Segment("mutable exact encountered a missing live source row".into()))?;
                }
                if top.competitive(score) { top.offer(score, key); }
            }
            operations += n as u64;
            ids.clear();
            Ok(())
        };
        while let Some(range) = ranges.claim() {
            if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
                return Err(RusticError::InvalidQuery("mutable exact query deadline exceeded".into()));
            }
            for w in range {
                let mut bits = node_filter[w];
                while bits != 0 {
                    let id = (w * 64) as u32 + bits.trailing_zeros();
                    bits &= bits - 1;
                    ids.push(id);
                    if ids.len() == BATCH { flush(&mut ids, &mut top)?; }
                }
            }
        }
        flush(&mut ids, &mut top)?;
        source.execution_result()?;
        Ok((top, operations))
    };
    let execute = |parallel: bool| -> Result<Vec<(ExactTop, u64)>> {
        if !parallel || workers == 1 { return Ok(vec![run(0)?]); }
        (0..workers).into_par_iter().map(run).collect()
    };
    // As the row path: a foreign Rayon worker scans alone (cross-pool install
    // can steal a writer blocked on locks the caller holds).
    let parts = if rayon::current_thread_index().is_some() && !crate::query::searcher::on_search_pool_worker() {
        execute(false)?
    } else {
        crate::query::searcher::in_search_pool(|| execute(true))?
    };
    let scanned = started.elapsed();
    let mut hits: Vec<ExactHit> = Vec::with_capacity(k * parts.len());
    let mut operations = 0;
    for (top, ops) in parts { hits.extend(top.heap.into_vec()); operations += ops; }
    hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.key.cmp(&b.key)));
    hits.truncate(k);
    EXACT_PHASES.with(|p| p.set([setup, first_worker.get().copied().unwrap_or_default(), scanned, started.elapsed()].map(|d| d.as_secs_f64() * 1e3)));
    Ok(Some(super::hnsw::MutableSearchResult {
        hits: hits.into_iter().map(|hit| (hit.key, hit.score)).collect(),
        vector_ops: operations, ..Default::default()
    }))
}

thread_local! {
    /// The calling thread's last exact code scan, in ms since its start:
    /// setup done (query bound and quantized), first worker running, scan
    /// done, merge done (`RUSTIC_KNN_SLOW_MS` reports).
    static EXACT_PHASES: std::cell::Cell<[f64; 4]> = const { std::cell::Cell::new([0.0; 4]) };
}
/// [`EXACT_PHASES`] of this thread, reset.
pub(crate) fn take_exact_phases() -> [f64; 4] { EXACT_PHASES.with(|p| p.replace([0.0; 4])) }

impl MutableIngestField {
    /// Closure adapter for GlobalFilter preparation. This evaluates document
    /// eligibility once per visible attribute ID, never once per graph edge.
    /// ES must retain its document-mask/publication fence while preparing and
    /// using this query. Do not cache it across refreshes or document updates.
    pub fn prepare_ann<F: Fn(u64) -> bool>(&self, accept: F)
        -> Result<PreparedMutableFieldSearch<'_>>
    {
        let (graph, source) = self.owner.read()?;
        let mut node_filter = empty_node_filter(graph.num_nodes())?;
        for (key, handle) in graph.visible_identities() {
            if accept(key) {
                let id = handle.slot() as usize + 1;
                if node_filter.len() <= id / 64 {return Err(RusticError::Memory("node filter bound changed".into()));}
                node_filter[id / 64] |= 1u64 << (id % 64);
            }
        }
        Ok(PreparedMutableFieldSearch { graph, source, matched: node_filter.iter().map(|w| w.count_ones() as usize).sum(), node_filter: Arc::new(node_filter) })
    }

    /// Cache only the ES document-to-node projection. Graph/source read views
    /// remain fresh; identity mutations invalidate all old slot translations.
    #[cfg(test)]
    pub(crate) fn prepare_ann_projection<I: IntoIterator<Item = u64>>(&self, key: ProjectionKey,
        build: impl FnOnce() -> I) -> Result<PreparedMutableFieldSearch<'_>> {
        let key_memory=crate::admission::shared().reserve(key.iter().map(|(_,w)|w.len() as u64*8+32).sum::<u64>()+256,"projection key")?;
        let (graph, source) = self.owner.read()?;
        let identities = self.identities.read();
        let revision = identities.revision();
        let nodes = graph.num_nodes();
        if let Some((node_filter, matched)) = self.projection_cache.lock().get(&key, revision, nodes) {
            return Ok(PreparedMutableFieldSearch { graph, source, node_filter, matched });
        }
        let mut words = empty_node_filter(nodes)?;
        identities.visit_nodes(build(), |node| {
            let id = node as usize;
            if let Some(word) = words.get_mut(id / 64) { *word |= 1u64 << (id % 64); }
        });
        let matched = words.iter().map(|word| word.count_ones() as usize).sum();
        let node_filter = Arc::new(words);
        self.projection_cache.lock().insert_admitted(key, key_memory, revision, nodes, false, node_filter.clone(), matched);
        Ok(PreparedMutableFieldSearch { graph, source, node_filter, matched })
    }

    /// Translate new masks using immutable segment document ordinals. This is
    /// needed because ES text segments do not use graph node IDs as document IDs.
    #[cfg(test)]
    fn prepare_ann_segments<'a>(&self,key:ProjectionKey,columns:impl FnOnce()->Vec<Option<crate::codec::vectors::PointColumn<'a>>>)->Result<PreparedMutableFieldSearch<'_>> {
        let memory=crate::admission::shared().reserve(key.iter().map(|(_,w)|w.len() as u64*8+32).sum::<u64>()+256,"test projection key")?;
        self.prepare_ann_segments_admitted(key,memory,columns)
    }
    /// Segment `segment`'s doc -> graph node map (`NO_VECTOR` for a doc
    /// without a vector, 0 for one with no node yet), cached per projection
    /// epoch; `None` if the segment is too large to cache one.
    fn segment_node_map(&self, segment: u64, epoch: u64, column: &crate::codec::vectors::PointColumn<'_>)
        -> Result<Option<Arc<crate::admission::Allocation<Vec<u32>>>>> {
        if let Some(cached) = self.projection_cache.lock().node_map(segment, epoch, 0) { return Ok(Some(cached)); }
        if !ProjectionCache::map_fits(column.num_docs() as usize) { return Ok(None); }
        let memory=crate::admission::shared().reserve(column.num_docs() as u64*4+256,"segment node projection")?;
        let mut mapping = Vec::new();
        mapping.try_reserve_exact(column.num_docs() as usize)
            .map_err(|_| RusticError::Memory("segment node projection".into()))?;
        {
            let identities = self.identities.read();
            mapping.extend((0..column.num_docs()).map(|doc| match column.get(doc) {
                None => NO_VECTOR,
                Some(point) => identities.get(&point).map_or(0, |h| h.slot() + 1),
            }));
        }
        let mapping = Arc::new(crate::admission::Allocation::new(mapping,memory));
        self.projection_cache.lock().insert_node_map_admitted(segment, epoch, 0, mapping.clone());
        Ok(Some(mapping))
    }

    /// Build a newly published sealed segment's node map ahead of the
    /// queries (see `VectorStore::warm_mutable_projections`): a large
    /// merged segment's map is a lookup per document (about 25 ms for 1M),
    /// which the first filtered or unfiltered query on it otherwise paid.
    pub(crate) fn warm_segment_node_map(&self, segment: u64, column: &crate::codec::vectors::PointColumn<'_>) -> Result<()> {
        let epoch = self.projection_epoch.load(Ordering::Acquire);
        self.segment_node_map(segment, epoch, column).map(|_| ())
    }

    pub(crate) fn prepare_ann_segments_admitted<'a>(&self, key: ProjectionKey, key_memory:crate::admission::Reservation,
        columns: impl FnOnce() -> Vec<Option<crate::codec::vectors::PointColumn<'a>>>)
        -> Result<PreparedMutableFieldSearch<'_>> {
        let (graph, source) = self.owner.read()?;
        let nodes = graph.num_nodes();
        let epoch = self.projection_epoch.load(Ordering::Acquire);
        // Keyed by `projection_epoch`, not the identity revision: every insert, update or delete
        // repair changes it, so under concurrent updates every query rebuilt
        // every segment's doc -> node map (an identity lookup per document,
        // millions per query) while holding the identity read lock that
        // every staging writer and delete repair waits on. A sealed
        // segment's live docs keep their nodes (a point's node changes only
        // by removal after its doc is deleted, which every view that still
        // sees the doc prevents, or in a migrated field, which has its own
        // cache), so the maps and filters stay valid: an entry for a doc
        // deleted since is masked out, and a doc with no node at build time
        // is looked up again (rows a graph gains later).
        if let Some((node_filter, matched)) = self.projection_cache.lock().get(&key, epoch, nodes) {
            return Ok(PreparedMutableFieldSearch { graph, source, node_filter, matched });
        }
        let mut words = empty_node_filter(nodes)?;
        // Every masked doc with a vector resolved to a node.
        let mut complete = true;
        let columns = columns();
        assert_eq!(key.len(), columns.len());
        for ((segment, mask), column) in key.iter().zip(columns) {
            let Some(column) = column else { continue; };
            let mapping = self.segment_node_map(*segment, epoch, &column)?;
            // Docs the map has no node for (and every doc without a map):
            // looked up now, under one short identity read.
            let mut lookups: Vec<u32> = Vec::new();
            // An empty mask is every doc of the leaf (unfiltered, no deletions).
            let all = mask.is_empty();
            let docs = column.num_docs() as usize;
            let word_count = if all { docs.div_ceil(64) } else { mask.len() };
            for wi in 0..word_count {
                let mut bits = if !all { mask[wi] }
                    else if wi + 1 == word_count && docs % 64 != 0 { (1u64 << (docs % 64)) - 1 } else { u64::MAX };
                while bits != 0 {
                    let doc = wi * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    match mapping.as_ref().map(|mapping| mapping.get(doc).copied().unwrap_or(NO_VECTOR)) {
                        Some(NO_VECTOR) => {}
                        Some(node) if node != 0 => {
                            if let Some(word) = words.get_mut(node as usize / 64) { *word |= 1 << (node % 64); }
                        }
                        _ => if let Ok(doc) = u32::try_from(doc) { lookups.push(doc); },
                    }
                }
            }
            if !lookups.is_empty() {
                let identities = self.identities.read();
                for doc in lookups {
                    let point = column.get(doc);
                    let node = point.and_then(|point| identities.get(&point)).map_or(0, |h| h.slot() + 1);
                    if node != 0 {
                        if let Some(word) = words.get_mut(node as usize / 64) { *word |= 1 << (node % 64); }
                    } else if point.is_some() { complete = false; }
                }
            }
        }
        let matched = words.iter().map(|word| word.count_ones() as usize).sum();
        let node_filter = Arc::new(words);
        self.projection_cache.lock().insert_admitted(key, key_memory, epoch, nodes, complete, node_filter.clone(), matched);
        Ok(PreparedMutableFieldSearch { graph, source, node_filter, matched })
    }

    pub(crate) fn prepare_ann_points(&self, points: &[u64]) -> Result<PreparedMutableFieldSearch<'_>> {
        let (graph, source) = self.owner.read()?;
        let mut node_filter = empty_node_filter(graph.num_nodes())?;
        let identities = self.identities.read();
        identities.visit(points, |handle| {
            let id = handle.slot() as usize + 1;
            if let Some(word) = node_filter.get_mut(id / 64) { *word |= 1u64 << (id % 64); }
        });
        Ok(PreparedMutableFieldSearch { graph, source, matched: node_filter.iter().map(|w| w.count_ones() as usize).sum(), node_filter: Arc::new(node_filter) })
    }

    pub(crate) fn new(store: Arc<VectorStore>, field_id: u16, dim: u32, similarity: Similarity, params: HnswParams) -> Result<Arc<Self>> {
        if dim == 0 || params.m == 0 || params.ef_construction == 0 {
            return Err(RusticError::InvalidQuery("invalid mutable field construction parameters".into()));
        }
        let graph = Arc::new(MutableHnsw::try_new(dim, similarity, params)?);
        let owner = Arc::new(MutableVectorOwner::new_quantized(graph.clone(),
            super::es_quant::EsQuant::for_mutable(&store.mutable_field_options(field_id))?)?);
        owner.set_raw_rows(Arc::new(super::store::StoreRows(Arc::downgrade(&store))));
        Ok(Arc::new(Self { store: Arc::downgrade(&store), params, field_id, graph, owner, reserve: Mutex::new(()),
            abandoned: Mutex::new(Vec::new()), inflight: AtomicUsize::new(0), admitting: Default::default(),
            identities: parking_lot::RwLock::new(Default::default()), projection_cache: Mutex::new(Default::default()), projection_epoch: AtomicU64::new(0), pending_deletes: Mutex::new(std::collections::VecDeque::new()), extra_references: Mutex::new(Default::default()),
            checkpoint_pins: Mutex::new(Default::default()), graph_inserts: Default::default(),
            successor: Default::default(), predecessors: Default::default(), off_heap: Default::default() }))
    }

    pub(crate) fn commit_batch(&self) -> Result<super::hnsw::MutableCommitBatch> {
        self.graph.commit_batch()
    }

    pub(crate) fn construction_threads(&self) -> usize { self.params.threads }

    /// Returns the code log files the checkpoint references.
    pub(crate) fn write_checkpoint(&self, dir: &dyn crate::codec::directory::Directory, name: &str) -> Result<Vec<String>> {
        // The snapshot owns references, so streaming requires no graph writer lock.
        self.capture_checkpoint()?.write_file(dir, name)
    }

    /// Move full code log chunks to their files (see
    /// [`MutableVectorOwner::seal_code_log`]).
    pub(crate) fn seal_code_log(&self, dir: &dyn crate::codec::directory::Directory) -> Result<()> {
        self.owner.seal_code_log(dir)
    }

    /// Code log files of this field's graph ([`MutableVectorOwner::code_log_files`]).
    pub(crate) fn code_log_files(&self) -> Vec<String> {
        self.owner.code_log_files()
    }

    /// Capture the checkpoint [`Self::write_checkpoint`] would write, for
    /// streaming later (outside the owner's write fence): the graph and rows
    /// as of now, owned by the returned snapshot.
    pub(crate) fn capture_checkpoint(&self) -> Result<super::hnsw::MutableVectorCheckpoint> {
        let _sequence = self.reserve.lock();
        if self.inflight.load(Ordering::Acquire) != 0 {
            return Err(RusticError::Segment("mutable checkpoint has in-flight ingestion".into()));
        }
        self.drain_locked()?;
        // Refuse an invalid manifest before publication if a caller omitted
        // mutable staging for an otherwise committed vector document.
        self.checkpoint_orphans()?;
        self.note_capture();
        self.owner.capture_checkpoint()
    }

    /// The capture point of a checkpoint taken under the owner's write
    /// fence (no ingestion in flight), copied later with
    /// [`super::hnsw::PendingMutableCheckpoint::finish`] off the fence.
    /// Unlike [`Self::capture_checkpoint`] it neither drains queued deletes
    /// nor validates against the vector store's sealed segments (memtables
    /// closed at this point are not sealed yet): reopening reconciles the
    /// committed points (`reconcile_committed_points`) and removes graph
    /// nodes of documents deleted since.
    ///
    /// Ingestion in flight does not block it (no write pause): batches not
    /// completed at the capture point are simply not in the captured graph
    /// (their pending reservations are invisible to it), and reopening
    /// reconciles them like any point committed after the graph checkpoint.
    pub(crate) fn begin_capture_checkpoint(&self) -> Result<super::hnsw::PendingMutableCheckpoint> {
        let _sequence = self.reserve.lock();
        self.note_capture();
        self.owner.begin_checkpoint()
    }

    /// Open a field from its checkpoint file: the field restored exactly as
    /// the file, or, when the file's rows cannot be restored (a referenced
    /// f32 row is gone from the vector store), the graph configuration to
    /// rebuild it with from the committed vector rows (the file must not be
    /// reused then).
    pub(crate) fn open_checkpoint(store: Arc<VectorStore>, field_id: u16,
        dir: &dyn crate::codec::directory::Directory, name: &str, threads: usize) -> Result<OpenedCheckpoint>
    {
        // Rows the checkpoint references (not embedded) are read from the
        // store's segments of the same commit.
        let raw = super::store::StoreRows(Arc::downgrade(&store));
        let snapshot = super::hnsw::RawRows::snapshot(&raw)?;
        let opened = MutableVectorOwner::open_file_with(dir, name, snapshot.as_deref());
        drop(snapshot);
        let (graph, owner) = match opened {
            Ok(opened) => opened,
            Err(error) if is_missing_referenced_row(&error) => {
                tracing::warn!(field = field_id, checkpoint = name, "{error}; rebuilding the vector graph from the committed vector rows");
                let graph = MutableVectorOwner::open_graph_file(dir, name)?;
                let mut params = graph.params();
                params.threads = threads;
                return Ok(OpenedCheckpoint::Rebuild { dim: graph.dim(), similarity: graph.similarity(), params });
            }
            Err(error) => return Err(error),
        };
        owner.set_raw_rows(Arc::new(raw));
        if graph.try_snapshot()?.visible_identities().any(|(key, _)| key >= store.next_point_id()) {
            return Err(RusticError::Corrupted("mutable checkpoint exceeds committed point allocator".into()));
        }
        let identities = identity_map(&graph.try_snapshot()?)?;
        let mut params = graph.params();
        params.threads = threads;
        Ok(OpenedCheckpoint::Restored(Arc::new(Self { store: Arc::downgrade(&store), params, field_id, graph,
            owner: Arc::new(owner), reserve: Mutex::new(()), abandoned: Mutex::new(Vec::new()), inflight: AtomicUsize::new(0), admitting: Default::default(),
            identities: parking_lot::RwLock::new(identities), projection_cache: Mutex::new(Default::default()), projection_epoch: AtomicU64::new(0), pending_deletes: Mutex::new(std::collections::VecDeque::new()), extra_references: Mutex::new(Default::default()),
            checkpoint_pins: Mutex::new(Default::default()), graph_inserts: Default::default(),
            successor: Default::default(), predecessors: Default::default(), off_heap: Default::default() })))
    }

    /// A graph snapshot can contain a staged-but-uncommitted point or a point
    /// deleted by the text commit. Restore the committed document universe
    /// without rebuilding adjacency for surviving nodes.
    fn checkpoint_orphans(&self) -> Result<Vec<NodeHandle>> {
        let (removed, missing) = self.checkpoint_differences()?;
        if !missing.is_empty() {
            return Err(RusticError::Corrupted("mutable checkpoint is missing committed vector points".into()));
        }
        Ok(removed.into_iter().map(|(_, handle)| handle).collect())
    }

    /// Graph nodes the committed documents no longer have (`removed`) and
    /// committed live points the graph lacks (`missing`: documents
    /// committed after the graph checkpoint, see
    /// [`Self::reconcile_committed_points`]).
    fn checkpoint_differences(&self) -> Result<(Vec<(u64, NodeHandle)>, Vec<u64>)> {
        let store = self.store.upgrade().ok_or_else(|| RusticError::Segment("mutable store dropped".into()))?;
        let expected: std::collections::HashSet<_> = store.segments(self.field_id).iter()
            .flat_map(|segment| segment.points().iter().copied())
            .filter(|key| store.mutable_point_is_live(*key)).collect();
        let view = self.graph.try_snapshot()?;
        let mut seen = std::collections::HashSet::with_capacity(expected.len());
        let mut removed = Vec::new();
        for (key, handle) in view.visible_identities() {
            if expected.contains(&key) { seen.insert(key); }
            else if store.mutable_point_is_live(key) {
                return Err(RusticError::Corrupted("mutable checkpoint belongs to another vector field".into()));
            } else { removed.push((key, handle)); }
        }
        let mut missing: Vec<u64> = expected.difference(&seen).copied().collect();
        missing.sort_unstable();
        drop(view);
        Ok((removed, missing))
    }

    /// Bring a graph restored from its checkpoint up to the committed
    /// documents: remove nodes of documents deleted since, and insert the
    /// committed points it lacks from the committed vector segments. Graph
    /// checkpoints are scheduled independently of text commits (Vespa
    /// style): a commit may reference an older graph checkpoint, and this
    /// catch-up is its replay (bounded by the save schedule).
    pub(crate) fn reconcile_committed_points(&self) -> Result<()> {
        let (removed, missing) = self.checkpoint_differences()?;
        for (point, handle) in removed {
            let source = self.owner.pin()?;
            self.graph.remove(handle, &source)?;
            self.pin_for_checkpoint(point);
        }
        if !missing.is_empty() {
            // Committed points whose admission was in flight at the
            // checkpoint's capture can lie below the keys it holds.
            let store = self.store()?;
            self.insert_points_backfill(&missing, |point| store.with_vector(point, <[f32]>::to_vec)
                .map(Some).ok_or_else(|| RusticError::Corrupted("committed vector point has no stored row".into())))?;
        }
        let store = self.store()?;
        // Lock order: the point table is never taken under the identity
        // directory (the vector optimizer holds the table and then reads
        // identities). Liveness is checked with no identity lock held.
        let points = self.identities.read().points();
        let dead: std::collections::HashSet<u64> = points.into_iter().filter(|key| !store.mutable_point_is_live(*key)).collect();
        if !dead.is_empty() {
            self.identities.write().retain(|key, _| !dead.contains(key));
            self.projection_epoch.fetch_add(1, Ordering::AcqRel);
        }
        self.owner.commit()
    }

    /// Insert the live committed points this graph lacks. Unlike
    /// [`reconcile_committed_points`](Self::reconcile_committed_points) it
    /// never removes a node: a registered graph holds admitted rows that are
    /// not sealed (or not live) yet. Returns how many it inserted.
    pub(crate) fn insert_missing_committed_points(&self) -> Result<usize> {
        let store = self.store()?;
        // Liveness first, with no identity lock held (see above).
        let live: Vec<u64> = store.segments(self.field_id).iter().flat_map(|segment| segment.points().iter().copied())
            .filter(|key| store.mutable_point_is_live(*key)).collect();
        let mut missing: Vec<u64> = { let identities = self.identities.read(); live.into_iter().filter(|key| identities.get(key).is_none()).collect() };
        missing.sort_unstable();
        self.insert_committed_points(&missing)?;
        Ok(missing.len())
    }

    /// Insert committed `points` (their rows read from the vector
    /// segments) into the graph under their existing ids (open time, a
    /// migration's catch-up, rows a seal stored without a reservation).
    pub(crate) fn insert_committed_points(&self, points: &[u64]) -> Result<()> {
        let store = self.store()?;
        self.insert_points_with(points, |point| store.with_vector(point, <[f32]>::to_vec)
            .map(Some).ok_or_else(|| RusticError::Corrupted("committed vector point has no stored row".into())))
    }

    /// Insert `points` with the rows `row` returns (`None`: skip the point)
    /// under their existing ids, in bounded batches; points already in the
    /// graph are skipped.
    fn insert_points_with(&self, points: &[u64], row: impl Fn(u64) -> Result<Option<Vec<f32>>>) -> Result<()> {
        self.insert_points_inner(points, row, false)
    }

    /// [`Self::insert_points_with`] at open: points absent from the restored
    /// graph may lie below its highest key (`MutableHnsw::reserve_insert_backfill`).
    fn insert_points_backfill(&self, points: &[u64], row: impl Fn(u64) -> Result<Option<Vec<f32>>>) -> Result<()> {
        self.insert_points_inner(points, row, true)
    }

    fn insert_points_inner(&self, points: &[u64], row: impl Fn(u64) -> Result<Option<Vec<f32>>>, backfill: bool) -> Result<()> {
        let sequence = self.reserve.lock();
        // A migration retired this owner (its last copy is done): the rows
        // belong in its successor.
        if let Some(next) = self.successor() {
            drop(sequence);
            return next.insert_points_inner(points, row, backfill);
        }
        let _sequence = sequence;
        // Streamed: only one batch of rows is buffered at a time (a large
        // catch-up or rebuild held every missing row in memory), and each
        // batch's graph preparation runs in parallel, as staged ingestion
        // does; completion stays serial (one topology writer).
        for batch in points.chunks(CATCH_UP_BATCH) {
            // A catch-up or rebuild can take hours: the opener may cancel it.
            crate::cancel::check()?;
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
            let mut keys: Vec<u64> = Vec::with_capacity(batch.len());
            for &point in batch {
                if self.identities.read().get(&point).is_some() { continue; }
                if let Some(values) = row(point)? { keys.push(point); rows.push(values); }
            }
            if keys.is_empty() { continue; }
            let batch = keys.as_slice();
            #[cfg(test)]
            CATCH_UP_MAX_BUFFERED.with(|max| max.set(max.get().max(rows.len())));
            let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
            let encoded = self.owner.prepare_codes(&slices)?;
            let commit_batch = self.graph.commit_batch()?;
            let mut reservations = Vec::with_capacity(batch.len());
            for ((point, values), encoded) in batch.iter().zip(&slices).zip(&encoded) {
                let reservation = if backfill { self.graph.reserve_insert_backfill(*point)? } else { self.graph.reserve_insert(*point)? };
                self.owner.install_prepared(&reservation, values, encoded.as_ref())?;
                reservations.push(reservation);
            }
            // Every task collected before an error propagates (no prepared
            // lease outlives the batch).
            let admission = crate::admission::Context::capture();
            let prepared: Vec<_> = reservations.into_par_iter().map(|reservation| {
                let _admission = admission.enter();
                let source = self.owner.pin()?;
                self.graph.prepare(reservation, &source)
            }).collect();
            for (point, proposal) in batch.iter().zip(prepared) {
                let source = self.owner.pin()?;
                let handle = self.graph.complete(proposal?, &source)?;
                self.identities.write().extend(std::iter::once((*point, handle)))?;
            }
            drop(commit_batch);
            self.owner.commit()?;
            self.graph_inserts.fetch_add(batch.len() as u64, Ordering::Relaxed);
            // Committed rows are durable in the vector segments: shed their
            // f32 now (best effort) rather than holding every caught-up
            // row's f32 in anonymous memory until the catch-up ends (1.15 GB
            // for 185k 1536-d rows at open).
            let _ = self.owner.shed_some(batch.len());
        }
        Ok(())
    }

    /// Graph insertions since this owner opened (see `graph_inserts`).
    pub(crate) fn graph_inserts(&self) -> u64 {
        self.graph_inserts.load(Ordering::Relaxed)
    }

    pub(crate) fn matches_config(&self, dim: u32, similarity: Similarity, params: HnswParams) -> bool {
        self.graph.dim() == dim && self.graph.similarity() == similarity && self.params == params
    }

    pub(crate) fn matches_index_options(&self, options: super::store::VectorFieldOptions) -> bool {
        options.m.is_none_or(|m| m == self.params.m)
            && options.ef_construction.is_none_or(|ef| ef == self.params.ef_construction)
            && super::es_quant::EsQuant::for_mutable(&options).is_ok_and(|spec| spec == self.owner.quantization())
    }

    /// Dimensions, similarity and parameters to rebuild this field with.
    pub(crate) fn rebuild_config(&self) -> (u32, Similarity, HnswParams) {
        (self.graph.dim(), self.graph.similarity(), self.params)
    }

    pub(crate) fn store(&self) -> Result<Arc<VectorStore>> {
        self.store.upgrade().ok_or_else(|| RusticError::Segment("mutable store dropped".into()))
    }
    pub(crate) fn document_limit(&self) -> u32 { self.graph.num_nodes().saturating_add(1) }
    /// Live graph nodes now (the profile's per-node operation bound).
    pub(crate) fn active_nodes(&self) -> u32 { self.graph.active_nodes() }
    pub(crate) fn field_id(&self) -> u16 { self.field_id }
    pub(crate) fn dimensions(&self) -> u32 { self.graph.dim() }
    pub(crate) fn similarity(&self) -> Similarity { self.graph.similarity() }
    /// The quantizer unrescored result scores are estimated with (B5).
    pub(crate) fn score_quantizer(&self) -> Option<super::es_quant::EsQuantizer> { self.owner.score_quantizer() }
    pub(crate) fn owns_reservation(self: &Arc<Self>, reservation: &PointReservation) -> bool {
        self.store.upgrade().is_some_and(|store| reservation.belongs_to(&store)) && (reservation.belongs_to_mutable(Arc::as_ptr(self) as usize)
            || self.predecessors.read().iter().any(|previous| reservation.belongs_to_mutable(*previous)))
    }

    /// Each point's own code score for `query`: its codes and their
    /// calibration epoch, what the graph route returns unrescored (`None` per
    /// point for an f32 field or a point not in the graph). Reads no f32 row
    /// and takes no lock but the source pin.
    pub(crate) fn code_scores(&self, query: &[f32], points: &[u64]) -> Result<Vec<Option<f32>>> {
        use super::hnsw::MutableVectorSource;
        let read = self.owner.pin()?;
        let Some(scorer) = read.bind_query(query) else { return Ok(vec![None; points.len()]) };
        let identities = self.identities.read();
        Ok(points.iter().map(|point| identities.get(point).and_then(|handle| scorer.score(*handle, *point))).collect())
    }

    /// The owner that replaced this one (a migration retired it).
    pub(crate) fn successor(&self) -> Option<Arc<MutableIngestField>> {
        self.successor.read().clone()
    }

    /// Drop `points` from this graph (their reservation was abandoned in a
    /// predecessor after a migration copied them).
    fn forget_points(&self, points: std::ops::Range<u64>) {
        let mut handles = Vec::new();
        {
            let mut identities = self.identities.write();
            for point in points.clone() {
                if let Some(handle) = identities.get(&point).copied() {
                    handles.push(handle);
                    identities.remove(&point);
                }
            }
            self.projection_epoch.fetch_add(1, Ordering::AcqRel);
        }
        self.abandoned.lock().append(&mut handles);
        if let Some(next) = self.successor() { next.forget_points(points); }
    }

    /// Bring this (unregistered) owner's graph to `previous`'s point set:
    /// insert the points it lacks (rows read through `previous`'s source,
    /// which holds every admitted row, durable or not) and remove the ones
    /// `previous` no longer has. `copied` is the points earlier passes took
    /// from `previous` (only those can be gone: once `previous` is retired
    /// this owner admits points of its own). Returns how many points
    /// changed. Runs off every shard lock; `previous` keeps serving writes
    /// and searches.
    pub(crate) fn catch_up_from(&self, previous: &MutableIngestField, copied: &mut std::collections::HashSet<u64>) -> Result<usize> {
        let frontier = { let _sequence = previous.reserve.lock(); previous.settled_frontier()? };
        self.catch_up_below(previous, copied, frontier)
    }

    /// The point below which this graph's admitted point set is final:
    /// the first point of the oldest admission still in flight, else the
    /// store's next point. Call under `reserve` (admissions reserve their
    /// points under it). A graph's keys must increase (`reserve_insert`):
    /// admissions reserve increasing points but are accepted in any order,
    /// so a catch-up that copied an accepted admission's points before an
    /// older in-flight one's were accepted could never insert the older
    /// ones ("mutable keys must increase monotonically"), failing every
    /// later pass and the retirement's last copy.
    fn settled_frontier(&self) -> Result<u64> {
        let next = self.store()?.next_point_id();
        Ok(self.admitting.lock().first().map_or(next, |&first| first.min(next)))
    }

    /// [`catch_up_from`](Self::catch_up_from) of `previous`'s points below
    /// `frontier` (see [`settled_frontier`](Self::settled_frontier)): every
    /// later pass copies only points above them.
    fn catch_up_below(&self, previous: &MutableIngestField, copied: &mut std::collections::HashSet<u64>, frontier: u64) -> Result<usize> {
        let wanted: std::collections::HashMap<u64, NodeHandle> = {
            let identities = previous.identities.read();
            identities.points().into_iter().filter(|&point| point < frontier)
                .filter_map(|point| identities.get(&point).map(|h| (point, *h))).collect()
        };
        let gone: Vec<u64> = copied.iter().copied().filter(|point| !wanted.contains_key(point)).collect();
        for &point in &gone {
            let handle = self.identities.read().get(&point).copied();
            if let Some(handle) = handle {
                let _sequence = self.reserve.lock();
                let source = self.owner.pin()?;
                self.graph.remove(handle, &source)?;
                self.identities.write().remove(&point);
                self.projection_epoch.fetch_add(1, Ordering::AcqRel);
            }
            copied.remove(&point);
        }
        let mut missing: Vec<u64> = wanted.keys().copied().filter(|point| self.identities.read().get(point).is_none()).collect();
        missing.sort_unstable();
        // A code-only node's f32 row is in `previous`'s raw rows until its
        // seal, in the vector segments after.
        use super::hnsw::MutableVectorSource;
        let store = self.store()?;
        let source = previous.owner.pin()?;
        self.insert_points_with(&missing, |point| Ok(wanted.get(&point).and_then(|handle| source.get_node(*handle, point)).map(<[f32]>::to_vec)
            .or_else(|| store.with_vector(point, <[f32]>::to_vec))))?;
        drop(source);
        copied.try_reserve(missing.len()).map_err(|_| RusticError::Memory("vector field migration point set".into()))?;
        copied.extend(missing.iter().copied());
        self.owner.commit()?;
        Ok(gone.len() + missing.len())
    }

    /// Retire this owner in favour of `fresh` (built by
    /// [`catch_up_from`](Self::catch_up_from) off every lock): under this
    /// owner's sequencer (new admissions wait at it), once the admissions
    /// already past it are accepted or abandoned, the last points are
    /// copied; then admissions, leases and deletes go to `fresh`, and this
    /// owner's leases and queued deletes move over (no drain runs between).
    /// The caller then registers `fresh`; searches use this owner's graph
    /// until then.
    ///
    /// The last copy must precede `fresh`'s own admissions: a graph's keys
    /// must increase, and `fresh`'s admissions take new (higher) points, so
    /// a point of this owner copied after one of them could never be
    /// inserted ("mutable keys must increase monotonically"). That failed
    /// every migration under concurrent writes.
    pub(crate) fn retire_into(&self, fresh: &Arc<MutableIngestField>, copied: &mut std::collections::HashSet<u64>) -> Result<()> {
        let _sequence = self.reserve.lock();
        // Admissions that reserved points before the sequencer was taken
        // complete their graph work and acceptance without it; they are
        // short (one write request). Ones still waiting for the sequencer
        // have reserved nothing: they are forwarded to `fresh` below.
        let started = std::time::Instant::now();
        while !self.admitting.lock().is_empty() {
            if started.elapsed() > std::time::Duration::from_secs(10) && started.elapsed().as_millis() % 10_000 < 2 {
                tracing::warn!(field = self.field_id, "vector field migration waits for in-flight admissions");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let caught_up = self.settled_frontier().and_then(|frontier| fresh.catch_up_below(self, copied, frontier));
        let mut predecessors = self.predecessors.read().clone();
        predecessors.push(self as *const Self as usize);
        fresh.predecessors.write().extend(predecessors);
        *self.successor.write() = Some(fresh.clone());
        // A row abandoned between the copy and the link above left this
        // graph without being forwarded: drop it from `fresh` too (nothing
        // new can be missing, this owner admits nothing now).
        let caught_up = caught_up.and_then(|_| self.settled_frontier())
            .and_then(|frontier| fresh.catch_up_below(self, copied, frontier));
        fresh.adopt_leases(self);
        caught_up.map(|_| ())
    }

    /// Atomically order point IDs and graph slots, then release the sequencer
    /// for parallel preparation. Completion uses the graph's topology writer;
    /// other batches may prepare concurrently without reversing source keys.
    pub(crate) fn stage(self: &Arc<Self>, rows: &[&[f32]]) -> Result<StagedMutableRows> {
        // A migration retired this owner: its successor admits the rows.
        if let Some(next) = self.successor() { return next.stage(rows); }
        // Validate the complete batch before consuming any identities.
        if rows.is_empty() || rows.iter().any(|v| v.len() != self.graph.dim() as usize || !self.graph.similarity().valid_row(v)) {
            return Err(RusticError::InvalidQuery("invalid mutable vector batch".into()));
        }
        let encoded = self.owner.prepare_codes(rows)?;
        self.inflight.fetch_add(1, Ordering::AcqRel);
        let mut staged = StagedMutableRows { field: self.clone(), reservation: None,
            handles: Vec::with_capacity(rows.len()), accepted: false, first_point: None };
        let reservations = {
            let sequence = self.reserve.lock();
            // Retired while this admission waited for the sequencer.
            if let Some(next) = self.successor() {
                drop(sequence);
                drop(staged);
                return next.stage(rows);
            }
            let _sequence = sequence;
            self.drain_on_write_locked()?;
            let store = self.store.upgrade().ok_or_else(|| RusticError::Segment("mutable vector store has been dropped".into()))?;
            let points = store.reserve_points(rows.len())?;
            self.admitting.lock().insert(points.points().start);
            staged.first_point = Some(points.points().start);
            let mut reservations = Vec::with_capacity(rows.len());
            for ((key, values), encoded) in points.points().zip(rows).zip(&encoded) {
                let reservation = self.graph.reserve_insert(key)?;
                self.owner.install_prepared(&reservation, values, encoded.as_ref())?;
                reservations.push(reservation);
            }
            staged.reservation = Some(points);
            reservations
        };
        // Test seam: graph ingestion in flight (reserve released).
        #[cfg(any(test, feature = "test-seams"))]
        if let Some(hook) = self.store.upgrade().and_then(|store| store.take_stage_pause()) { hook(); }
        let commit_batch = self.graph.commit_batch()?;
        // Collect every task before propagating an error, so no prepared lease
        // can outlive cancellation of this batch by the calling write path.
        let diagnostic = diagnostics::Batch::begin(self.field_id, rows.len());
        let admission = crate::admission::Context::capture();
        let prepared: Vec<_> = reservations.into_par_iter().map(|reservation| {
            let _admission = admission.enter();
            let prepare = || {
                let source = self.owner.pin()?;
                self.graph.prepare(reservation, &source)
            };
            match &diagnostic {
                Some(d) => d.task(prepare),
                None => prepare(),
            }
        }).collect();
        let completion = diagnostic.as_ref().map(|d| d.completion());
        for proposal in prepared {
            let source = self.owner.pin()?;
            staged.handles.push(self.graph.complete(proposal?, &source)?);
        }
        if let (Some(d), Some(t)) = (&diagnostic, completion) { d.finish(t); }
        drop(commit_batch);
        self.owner.commit()?;
        self.graph_inserts.fetch_add(staged.handles.len() as u64, Ordering::Relaxed);
        Ok(staged)
    }

    /// [`stage`](Self::stage) of a bulk's documents at once: their rows are
    /// coded together, reserved and installed under one sequencer hold,
    /// prepared in one parallel pass, completed in one loop and committed
    /// once, where staging each document by itself took the sequencer,
    /// prepared one row, and ran a source commit (generation scan and
    /// reclaim) per document. Each document keeps its own staged rows and
    /// so its own rollback; on an error every document is rolled back.
    pub(crate) fn stage_documents(self: &Arc<Self>, documents: &[Vec<&[f32]>]) -> Result<Vec<StagedMutableRows>> {
        if let Some(next) = self.successor() { return next.stage_documents(documents); }
        let dim = self.graph.dim() as usize;
        if documents.iter().any(|rows| rows.is_empty() || rows.iter().any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))) {
            return Err(RusticError::InvalidQuery("invalid mutable vector batch".into()));
        }
        // In slices: one slice's reservations hold the sequencer, and its
        // inserts one maintenance batch (generation advance and reclamation
        // wait for its end), so a whole bulk in one slice made searches and
        // other writers of this field wait behind every row of it (knn p99
        // during indexing). A slice still prepares its rows in parallel.
        // A failure drops the slices staged so far (their rows roll back).
        let mut staged = Vec::with_capacity(documents.len());
        for slice in documents.chunks(stage_slice_docs()) {
            staged.extend(self.stage_slice(slice)?);
        }
        Ok(staged)
    }

    /// [`stage_documents`](Self::stage_documents) of one slice (validated).
    fn stage_slice(self: &Arc<Self>, documents: &[Vec<&[f32]>]) -> Result<Vec<StagedMutableRows>> {
        let rows: Vec<&[f32]> = documents.iter().flatten().copied().collect();
        let encoded = self.owner.prepare_codes(&rows)?;
        let mut staged: Vec<StagedMutableRows> = Vec::with_capacity(documents.len());
        let mut reservations = Vec::with_capacity(rows.len());
        let mut owners = Vec::with_capacity(rows.len());
        let mut keys = Vec::with_capacity(rows.len());
        {
            let sequence = self.reserve.lock();
            // Retired while this admission waited for the sequencer.
            if let Some(next) = self.successor() {
                drop(sequence);
                drop(staged);
                return next.stage_slice(documents);
            }
            let _sequence = sequence;
            self.drain_on_write_locked()?;
            let store = self.store()?;
            for (doc, rows) in documents.iter().enumerate() {
                self.inflight.fetch_add(1, Ordering::AcqRel);
                staged.push(StagedMutableRows { field: self.clone(), reservation: None,
                    handles: Vec::with_capacity(rows.len()), accepted: false, first_point: None });
                let points = store.reserve_points(rows.len())?;
                self.admitting.lock().insert(points.points().start);
                staged[doc].first_point = Some(points.points().start);
                keys.extend(points.points().take(rows.len()));
                owners.extend(std::iter::repeat_n(doc, rows.len()));
                staged[doc].reservation = Some(points);
            }
            // Graph slots and source rows for the whole slice under one
            // topology-writer hold (see `reserve_install_batch`).
            self.owner.reserve_install_batch(&keys, &rows, &encoded, &mut reservations)?;
        }
        // Test seam: graph ingestion in flight (reserve released), as in `stage`.
        #[cfg(any(test, feature = "test-seams"))]
        if let Some(hook) = self.store.upgrade().and_then(|store| store.take_stage_pause()) { hook(); }
        let commit_batch = self.graph.commit_batch()?;
        let diagnostic = diagnostics::Batch::begin(self.field_id, rows.len());
        let admission = crate::admission::Context::capture();
        // This thread (a write thread) prepares and completes its slice, as
        // ES's write threads index their own bulks: concurrent bulks fill
        // the cores through the write pool. Only the share of the prepare
        // pool that other stagings leave free (`staging_workers`) prepares
        // in parallel beside it: all of it for a lone bulk, none once as
        // many bulks stage as the pool has threads (then no hand-off).
        let staging = Staging::enter();
        // While the node is searching, every slice prepares on the prepare
        // pool, whose threads run at a lower priority (see the engine's
        // `vector_prepare_pool`): searches take the cores they need first,
        // as before preparation moved to the write threads (on write
        // threads, at normal priority, it took text search p99 under vector
        // ingest from ~37 to ~53 ms). A slice there takes at most a share of
        // the pool (`PREPARE_SHARE_WHILE_SEARCHING`). Unprivileged threads
        // cannot raise their priority back, so the write thread itself is
        // not lowered.
        let searching = crate::searches_active();
        let workers = if searching {
            (staging.share() / prepare_share_while_searching()).max(1)
        } else {
            staging.workers()
        };
        let prepare_one = |reservation| {
            let _admission = admission.enter();
            let prepare = || {
                let source = self.owner.pin()?;
                self.graph.prepare(reservation, &source)
            };
            match &diagnostic {
                Some(d) => d.task(prepare),
                None => prepare(),
            }
        };
        // In waves: each wave's rows are prepared, then completed (under one
        // topology-writer hold, `MutableHnsw::complete_batch`) before the
        // next wave prepares, so a row is prepared against a graph holding
        // every earlier row of its slice but its own wave's. Rows a bulk
        // sends in document order are often each other's nearest neighbours:
        // preparing a whole 64-row slice before completing any left them
        // unable to link to one another (recall@10 0.80 -> 0.75 on Wikipedia
        // embeddings loaded in order). A wave is one row per parallel worker,
        // or `serial_prepare_wave()` rows prepared on this thread alone: about
        // as many rows in flight as preparing each row on its own pool
        // thread had.
        let wave = workers.max(serial_prepare_wave());
        let completion = diagnostic.as_ref().map(|d| d.completion());
        let mut pending = reservations.into_iter();
        let mut next_owner = 0;
        loop {
            let chunk: Vec<_> = pending.by_ref().take(wave).collect();
            if chunk.is_empty() { break; }
            let prepared: Vec<_> = if workers <= 1 && !searching {
                chunk.into_iter().map(&prepare_one).collect()
            } else {
                let min_len = chunk.len().div_ceil(workers).max(prepare_task_rows());
                on_prepare_pool(|| chunk.into_par_iter().with_min_len(min_len).map(&prepare_one).collect())
            };
            // Published nodes go to their documents' staged rows, which own
            // their rollback, also when a later completion fails.
            let mut proposals = Vec::with_capacity(prepared.len());
            for proposal in prepared { proposals.push(proposal?); }
            let source = self.owner.pin()?;
            let mut done = Vec::with_capacity(proposals.len());
            let completed = self.graph.complete_batch(&mut proposals, &source, &mut done);
            drop(source);
            for handle in done {
                staged[owners[next_owner]].handles.push(handle);
                next_owner += 1;
            }
            drop(proposals);
            completed?;
        }
        drop(staging);
        if let (Some(d), Some(t)) = (&diagnostic, completion) { d.finish(t); }
        drop(commit_batch);
        self.owner.commit()?;
        self.graph_inserts.fetch_add(rows.len() as u64, Ordering::Relaxed);
        Ok(staged)
    }

    /// GPU admission of multiple documents with separate rollback ownership.
    /// The GPU graph is temporary construction input, never a search segment.
    /// A failed GPU build reserves no identities and lets the caller use CPU.
    #[cfg(feature = "gpu")]
    pub(crate) fn stage_gpu_documents(self: &Arc<Self>, documents: &[Vec<&[f32]>])
        -> Result<Option<Vec<StagedMutableRows>>>
    {
        let count: usize = documents.iter().map(Vec::len).sum();
        if count < super::gpu::MIN_NUM_VECTORS_FOR_GPU_BUILD { return Ok(None); }
        let dim = self.graph.dim() as usize;
        if documents.iter().any(|rows| rows.is_empty() || rows.iter().any(|row|
            row.len() != dim || !self.graph.similarity().valid_row(row))) {
            return Err(RusticError::InvalidQuery("invalid GPU mutable document batch".into()));
        }
        let Some(mut lease) = super::gpu::try_lease() else { return Ok(None) };
        let mut data = Vec::new();
        data.try_reserve_exact(count.checked_mul(dim).ok_or_else(|| RusticError::Memory("GPU batch dimension overflow".into()))?)
            .map_err(|_| RusticError::Memory("GPU batch allocation".into()))?;
        for rows in documents { for row in rows { data.extend_from_slice(row); } }
        let gpu = match lease.build_graph(dim, &data, self.graph.similarity(), self.params.m, self.params.ef_construction) {
            Ok(graph) => graph,
            Err(error) => { tracing::warn!("GPU batch build failed, using CPU: {error}"); return Ok(None); }
        };
        drop(lease);
        drop(data);
        let order = gpu.mutable_insertion_order();
        if order.len() != count { return Err(RusticError::Segment("GPU batch row count mismatch".into())); }
        let rows: Vec<_> = documents.iter().flatten().copied().collect();
        let encoded = self.owner.prepare_codes(&rows)?;
        let mut encoded = encoded.iter();
        let mut staged = Vec::with_capacity(documents.len());
        let mut reservations = Vec::with_capacity(count);
        let mut owners = Vec::with_capacity(count);
        {
            let _sequence = self.reserve.lock();
            self.drain_on_write_locked()?;
            let store = self.store()?;
            for (doc, rows) in documents.iter().enumerate() {
                self.inflight.fetch_add(1, Ordering::AcqRel);
                staged.push(StagedMutableRows { field: self.clone(), reservation: None,
                    handles: Vec::with_capacity(rows.len()), accepted: false, first_point: None });
                let points = store.reserve_points(rows.len())?;
                self.admitting.lock().insert(points.points().start);
                staged[doc].first_point = Some(points.points().start);
                for (key, row) in points.points().zip(rows) {
                    let reservation = self.graph.reserve_insert(key)?;
                    self.owner.install_prepared(&reservation, row, encoded.next().expect("validated encoded GPU row").as_ref())?;
                    reservations.push(Some(reservation));
                    owners.push(doc);
                }
                staged[doc].reservation = Some(points);
            }
        }
        let handles: Vec<_> = reservations.iter().map(|r| r.as_ref().expect("reserved row").handle()).collect();
        let commit_batch = self.graph.commit_batch()?;
        for ordinal in order {
            let i = ordinal as usize;
            let reservation = reservations.get_mut(i).and_then(Option::take)
                .ok_or_else(|| RusticError::Segment("GPU batch duplicate/out-of-range row".into()))?;
            let source = self.owner.pin()?;
            let proposal = self.graph.prepare_gpu(reservation, &source, &gpu, ordinal, &handles)?;
            let handle = self.graph.complete(proposal, &source)?;
            staged[owners[i]].handles.push(handle);
        }
        drop(commit_batch);
        self.owner.commit()?;
        // Acceptance zips handles with point keys, so restore document order
        // after top-level-first construction. Failure cleanup above only needs
        // the successfully completed handle set, not its ordering.
        for batch in &mut staged { batch.handles.clear(); }
        for (handle, doc) in handles.into_iter().zip(owners) { staged[doc].handles.push(handle); }
        Ok(Some(staged))
    }

    /// Execute the shared Vespa ANN plan over this field. `accept` must include
    /// the caller's published document visibility as well as its query filter;
    /// graph readiness alone never makes a document searchable. Exact plans
    /// prepare eligibility before entering the document-range executor.
    /// No short-hit exact fill or silent missing-source recovery is performed.
    pub fn search_ann(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: impl Fn(u64) -> bool, deadline: Option<std::time::Instant>)
        -> Result<super::hnsw::MutableSearchResult>
    {
        let (graph, source) = self.owner.read()?;
        execute_ann(&graph, &source, query, plan, |_, key| accept(key), deadline, None)
    }

    /// [`search_ann`](Self::search_ann) with an acceptance the search pool's
    /// workers can share (the during-indexing view's visible points): a
    /// plain beam wide enough, on a node with idle cores, is split over
    /// them (`parallel_beam`), as the prepared node-filter search is.
    pub(crate) fn search_ann_shared(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: &(impl Fn(u64) -> bool + Sync), deadline: Option<std::time::Instant>)
        -> Result<super::hnsw::MutableSearchResult>
    {
        use super::vespa_policy::{Route, FilterApplication};
        let (graph, source) = self.owner.read()?;
        if plan.route == Route::Plain && plan.filter_application != FilterApplication::AnnUnfilteredThenOuterQuery {
            let width = plan.exploration_budget.unwrap_or(0) as usize;
            let workers = super::hnsw::beam_workers(width);
            if workers > 1 {
                let params = super::hnsw::VespaPlainParams {
                    filter_estimate: plan.filter_statistics.map(|s| super::hnsw::VespaFilterEstimate { size: s.size, count: s.count }),
                    deadline, prefetch_tensors: query_tensor_prefetch(), ..Default::default()
                };
                let parallel = crate::query::searcher::in_search_pool(|| graph.search_parallel(
                    query, plan.adjusted_target_hits as usize, width, &source, params, &|_, key| accept(key), None, workers))?;
                if let Some(result) = parallel {
                    super::hnsw::MutableVectorSource::execution_result(&source)?;
                    if result.missing_source_lookups != 0 {
                        return Err(RusticError::Segment("mutable ANN encountered a missing live source row".into()));
                    }
                    return Ok(result);
                }
            }
        }
        execute_ann(&graph, &source, query, plan, |_, key| accept(key), deadline, None)
    }

    /// Explicit retained view only: copies graph/source references under the
    /// caller's document publication fence. Ordinary search uses read(), above.
    pub(crate) fn capture(&self) -> Result<Arc<CapturedMutableField>> {
        let (graph, source) = self.owner.capture()?;
        Ok(CapturedMutableField::new(identity_map(&graph)?, graph, source))
    }

    /// Background attribute maintenance uses the native Vespa buffer-selection
    /// policy; moving storage references never rebuilds graph topology.
    pub(crate) fn maintain(&self) -> Result<()> {
        self.drain_released()?;
        let _sequence = self.reserve.lock();
        self.drain_abandoned_locked(usize::MAX)?;
        // Refresh repair may run before sealing makes raw rows durable and
        // then disarm itself. Retry from periodic maintenance even when no
        // further writes/refreshes arrive. Bound work per field and preserve
        // unsealed rows: shed_some checks durable identity before replacing.
        // Rounds of 4096 (the graph writer is released between them) so
        // shedding keeps up with sustained ingestion: an unshed row holds
        // its f32 in anonymous memory, a shed one only its codes in the
        // file-backed code log.
        for _ in 0..16 {
            if self.owner.shed_some(4096)? < 4096 { break; }
        }
        self.graph.compact_storage()?;
        self.owner.compact_storage()?;
        self.owner.commit()
    }

    /// Run before checkpoint admission and periodically after failed writes.
    /// A failed removal remains queued for retry and blocks further admission.
    pub fn drain_abandoned(&self) -> Result<()> {
        self.drain_released()?;
        let _sequence = self.reserve.lock();
        self.drain_abandoned_locked(usize::MAX)
    }

    /// Every eligible queued delete, in bounded batches with `reserve`
    /// released between them: a large backlog (a delete-by-query) never
    /// holds the field's writers for the whole repair.
    fn drain_released(&self) -> Result<()> {
        loop {
            let (processed, more) = {
                let _sequence = self.reserve.lock();
                self.drain_batch_locked(RELEASED_DRAIN_BATCH)?
            };
            if processed == 0 || !more { return Ok(()); }
        }
    }
    /// An unchanged vector can keep its topology/source identity while an ES
    /// replacement owns a second document lease until refresh retires the old.
    pub(crate) fn reuse_unchanged(self: &Arc<Self>, point: u64, values: &[f32]) -> Result<Option<PointReservation>> {
        let sequence = self.reserve.lock();
        // A migration retired this owner: the lease belongs to its successor.
        if let Some(next) = self.successor() {
            drop(sequence);
            return next.reuse_unchanged(point, values);
        }
        let _sequence = sequence;
        let Some(handle) = self.identities.read().get(&point).copied() else { return Ok(None); };
        if !self.owner.unchanged(handle, point, values)? { return Ok(None); }
        let reservation = self.store()?.reserve_existing_point(point)?;
        {
            let mut references = self.extra_references.lock();
            references.try_reserve(1).map_err(|_| RusticError::Memory("mutable document lease allocation".into()))?;
            let count = references.entry(point).or_default();
            *count = count.checked_add(1).ok_or_else(|| RusticError::Memory("mutable document lease overflow".into()))?;
        }
        let owner = self.clone();
        if let Err(error) = reservation.on_abandon(Arc::as_ptr(self) as usize, move || owner.queue_document_delete(point)) {
            let mut references = self.extra_references.lock();
            if let Some(count) = references.get_mut(&point) { *count -= 1; if *count == 0 { references.remove(&point); } }
            return Err(error);
        }
        Ok(Some(reservation))
    }

    /// Take over `previous`'s document leases and queued deletes (a graph
    /// rebuilt for a mapping update replaces `previous`): an unchanged
    /// replacement shares its point with the document it replaces, and the
    /// rebuilt graph must keep the node when that document's delete is
    /// repaired, exactly as `previous` would have.
    /// The caller holds `previous`'s sequencer (no drain moves a lease
    /// meanwhile).
    pub(crate) fn adopt_leases(&self, previous: &MutableIngestField) {
        let references = previous.extra_references.lock().clone();
        self.extra_references.lock().extend(references);
        let queued: Vec<(u64, u64)> = previous.pending_deletes.lock().iter().copied().collect();
        self.pending_deletes.lock().extend(queued);
    }

    /// Queued deletes not yet physically repaired.
    pub(crate) fn pending_delete_count(&self) -> usize { self.pending_deletes.lock().len() }
    pub(crate) fn graph_stats(&self) -> super::hnsw::MutableHnswStats { self.graph.stats() }

    /// `(adjacency, code log)` bytes for index stats (`dense_vector`
    /// `off_heap`). Measuring the adjacency walks every stored array under
    /// the graph's writer lock: on c22's 30M-vector graph, `_cat/indices`
    /// polled every 10 s spent 90% of the node's CPU there and held graph
    /// writes off meanwhile. ES reads these sizes from file lengths. A
    /// graph of at most [`OFF_HEAP_EXACT_SLOTS`] slots is measured whenever it
    /// changed. A larger one is measured once, then again in the
    /// background at most every [`OFF_HEAP_REFRESH`], while stats calls
    /// return the last measurement.
    pub(crate) fn off_heap_bytes(self: &Arc<Self>) -> (u64, u64) {
        let generation = self.graph.mutation_generation();
        let small = self.graph.num_nodes() <= OFF_HEAP_EXACT_SLOTS;
        {
            let mut cache = self.off_heap.lock();
            match cache.value {
                Some((_, measured_generation, value)) if measured_generation == generation => return value,
                Some((at, _, value)) if !small => {
                    if at.elapsed() >= OFF_HEAP_REFRESH && !cache.refreshing {
                        cache.refreshing = true;
                        let field = self.clone();
                        let spawned = std::thread::Builder::new().name("rustic-vector-stats".into()).spawn(move || {
                            let measured = field.measure_off_heap();
                            let mut cache = field.off_heap.lock();
                            cache.value = Some(measured);
                            cache.refreshing = false;
                        });
                        if spawned.is_err() { cache.refreshing = false; }
                    }
                    return value;
                }
                _ => {}
            }
        }
        let measured = self.measure_off_heap();
        self.off_heap.lock().value = Some(measured);
        measured.2
    }

    fn measure_off_heap(&self) -> (std::time::Instant, u64, (u64, u64)) {
        let generation = self.graph.mutation_generation();
        let stats = self.graph.stats();
        let graph = stats.live_adjacency_bytes as u64 + u64::from(stats.allocated_slots) * 4;
        (std::time::Instant::now(), generation, (graph, self.owner.code_log_bytes() as u64))
    }
    /// Bytes of the source's code log (see `MutableVectorOwner::code_log_bytes`).
    pub(crate) fn code_log_bytes(&self) -> usize { self.owner.code_log_bytes() }
    /// The graph's mutation generation (see `MutableHnsw::mutation_generation`).
    pub(crate) fn mutation_generation(&self) -> u64 { self.graph.mutation_generation() }

    pub(crate) fn queue_document_delete(&self, point: u64) {
        // A migration retired this owner: its successor repairs the delete.
        if let Some(next) = self.successor() { return next.queue_document_delete(point); }
        // Stamped under the queue lock: the queue stays in epoch order, so
        // a delete with a newer epoch never sits ahead of an eligible older
        // one (the drain stops at the first ineligible entry).
        let store = self.store.upgrade();
        let mut pending = self.pending_deletes.lock();
        let epoch = store.map_or(0, |store| store.delete_epoch());
        pending.push_back((point, epoch));
    }
    /// Bounded physical repair of queued document deletes (background
    /// maintenance, outside every shard lock): at most `max` eligible queued
    /// points, taken off the queue so publishers that queue deletes never
    /// wait on graph repair. Returns whether deletes remain queued. Logical
    /// visibility never depends on this: searches mask deleted documents
    /// through their reader view's deletions.
    pub(crate) fn drain_some(&self, max: usize) -> Result<bool> {
        let _sequence = self.reserve.lock();
        let (_, more) = self.drain_batch_locked(max)?;
        if !more {
            self.drain_abandoned_locked(max)?;
        }
        // Quantized rows whose f32 is durable keep only codes in memory.
        let shed = self.owner.shed_some(max)?;
        self.memory_diagnostic();
        Ok(more || shed == max)
    }

    /// Opt-in diagnostic (`RUSTIC_VECTOR_MEMORY_DIAGNOSTIC=<seconds>`, off by
    /// default): at most once per interval per field, log the graph's and
    /// the source's storage breakdown (resident bytes per vector).
    fn memory_diagnostic(&self) {
        static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        static LAST: Mutex<Vec<(usize, std::time::Instant)>> = parking_lot::const_mutex(Vec::new());
        let every = *EVERY.get_or_init(|| std::env::var("RUSTIC_VECTOR_MEMORY_DIAGNOSTIC").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        if every == 0 { return; }
        let me = self as *const Self as usize;
        {
            let mut last = LAST.lock();
            match last.iter_mut().find(|(field, _)| *field == me) {
                Some((_, at)) if at.elapsed().as_secs() < every => return,
                Some((_, at)) => *at = std::time::Instant::now(),
                None => last.push((me, std::time::Instant::now())),
            }
        }
        tracing::info!(target: "vector_memory_diagnostic", field = self.field_id, nodes = self.graph.active_nodes(),
            identity_heap_bytes = self.identities.read().heap_bytes(), graph = ?self.graph.storage_memory(),
            owner = %self.owner.memory_diagnostic(), "mutable vector memory diagnostic");
    }

    /// Test seam: a source commit boundary (reclaims retired rows).
    #[cfg(test)]
    pub(crate) fn commit_owner(&self) -> Result<()> {
        self.owner.commit()
    }

    /// Live source row bytes (see `MutableVectorOwner::row_bytes_used`).
    #[cfg(test)]
    pub(crate) fn source_row_bytes(&self) -> usize {
        self.owner.row_bytes_used()
    }

    /// Whether the graph still holds a node for `point`.
    pub(crate) fn holds_point(&self, point: u64) -> bool {
        self.identities.read().get(&point).is_some()
    }

    /// Bring `point`'s f32 back into memory (its durable row is going while
    /// the graph still holds the node).
    pub(crate) fn restore_raw(&self, point: u64, row: &[f32]) -> Result<()> {
        let Some(handle) = self.identities.read().get(&point).copied() else { return Ok(()) };
        self.owner.unshed(handle, point, row)
    }

    /// Test seam: see `MutableVectorOwner::set_code_log_compact_min_rows`.
    #[cfg(test)]
    pub(crate) fn set_code_log_compact_min_rows(&self, rows: u64) {
        self.owner.set_code_log_compact_min_rows(rows);
    }

    /// Test seam: see `MutableVectorOwner::set_code_log_capacity`.
    #[cfg(test)]
    pub(crate) fn set_code_log_capacity(&self, rows: u64) {
        self.owner.set_code_log_capacity(rows);
    }

    /// Test seam: see `MutableVectorOwner::code_log_state`.
    #[cfg(test)]
    pub(crate) fn code_log_state(&self) -> Option<(u64, u64, u64, bool)> {
        self.owner.code_log_state()
    }

    /// Test seam: see `MutableVectorOwner::unshed_newest_first`.
    #[cfg(test)]
    pub(crate) fn unshed_newest_first(&self) {
        self.owner.unshed_newest_first();
    }

    /// Whether `point`'s row is readable through the live source now.
    #[cfg(test)]
    pub(crate) fn row_readable(&self, point: u64) -> bool {
        use super::hnsw::MutableVectorSource;
        let Some(handle) = self.identities.read().get(&point).copied() else { return false };
        self.owner.pin().is_ok_and(|read| read.get_node(handle, point).is_some())
    }

    /// Shed the in-memory f32 of every quantized row whose f32 is durable
    /// (tests and maintenance).
    pub(crate) fn shed_all(&self) -> Result<usize> {
        let mut total = 0;
        loop {
            let shed = self.owner.shed_some(4096)?;
            total += shed;
            if shed < 4096 { return Ok(total); }
        }
    }

    /// Take at most `max` eligible queued deletes off the queue and remove
    /// their nodes without holding the queue lock. Eligible: no reader view
    /// older than the delete's epoch is alive (`VectorStore::min_live_view`),
    /// so an in-flight older view never loses a node it counts live; other
    /// deletes stay queued. A failure re-queues the unprocessed points
    /// (retried first); an exhausted retirement budget (old graph generations
    /// pinned by long-lived readers) only defers them. Returns how many were
    /// processed and whether deletes remain queued.
    fn drain_batch_locked(&self, max: usize) -> Result<(usize, bool)> {
        let min_live = self.store.upgrade().and_then(|store| store.min_live_view());
        let eligible = |epoch: u64| min_live.is_none_or(|min| min >= epoch);
        let batch: Vec<(u64, u64)> = {
            let mut pending = self.pending_deletes.lock();
            let mut taken = Vec::new();
            // Oldest first; stop at the first delete a live view still needs
            // (later ones carry the same or a newer epoch).
            while taken.len() < max {
                match pending.front() {
                    Some(&(_, epoch)) if eligible(epoch) => {
                        if let Some(entry) = pending.pop_front() { taken.push(entry); }
                    }
                    _ => break,
                }
            }
            taken
        };
        for i in 0..batch.len() {
            if let Err(error) = self.remove_deleted_point(batch[i].0) {
                let mut pending = self.pending_deletes.lock();
                for &entry in batch[i..].iter().rev() {
                    pending.push_front(entry);
                }
                drop(pending);
                if crate::vector::is_retirement_backpressure(&error) {
                    return Ok((i, true));
                }
                return Err(error);
            }
        }
        Ok((batch.len(), !self.pending_deletes.lock().is_empty()))
    }

    fn remove_deleted_point(&self, point: u64) -> Result<()> {
        let retained = {
            let mut references = self.extra_references.lock();
            if let Some(count) = references.get_mut(&point) {
                *count -= 1;
                if *count == 0 { references.remove(&point); }
                true
            } else { false }
        };
        if retained { return Ok(()); }
        let handle = self.identities.read().get(&point).copied();
        if let Some(handle) = handle {
            let source = self.owner.pin()?;
            self.graph.remove(handle, &source)?;
            self.pin_for_checkpoint(point);
            self.identities.write().remove(&point);
        }
        Ok(())
    }

    /// `point`'s node was just removed: an older durable checkpoint may
    /// still reference its row.
    fn pin_for_checkpoint(&self, point: u64) {
        let generation = self.graph.mutation_generation();
        let mut pins = self.checkpoint_pins.lock();
        if pins.floor().is_some_and(|floor| generation > floor) {
            pins.removed.insert(point, generation);
            pins.snapshot = None;
        }
    }

    /// A checkpoint capture begins (under `reserve`, so no removal runs
    /// between this and the capture point): removals from now on may be in
    /// a file that becomes durable.
    fn note_capture(&self) {
        let generation = self.graph.mutation_generation();
        let mut pins = self.checkpoint_pins.lock();
        pins.capturing = Some(pins.capturing.map_or(generation, |capturing| capturing.min(generation)));
    }

    /// A checkpoint of this field at mutation generation `generation` is
    /// durable (the commit point referencing it is published): rows of
    /// nodes removed before its capture are free.
    pub(crate) fn checkpoint_durable(&self, generation: u64) {
        let mut pins = self.checkpoint_pins.lock();
        pins.durable = Some(generation);
        if pins.capturing.is_some_and(|capturing| capturing <= generation) { pins.capturing = None; }
        let floor = pins.floor();
        let before = pins.removed.len();
        pins.removed.retain(|_, removed| floor.is_some_and(|floor| *removed > floor));
        if pins.removed.len() != before { pins.snapshot = None; }
    }

    /// Points whose durable rows the vector optimizer must keep for the
    /// newest durable checkpoint (`None`: none).
    pub(crate) fn checkpoint_pinned_points(&self) -> Option<Arc<std::collections::HashSet<u64>>> {
        let mut pins = self.checkpoint_pins.lock();
        if pins.removed.is_empty() { return None; }
        if pins.snapshot.is_none() {
            pins.snapshot = Some(Arc::new(pins.removed.keys().copied().collect()));
        }
        pins.snapshot.clone()
    }

    /// Write-path repair under `reserve`: a bounded slice of the queue so a
    /// writer never pays for a large backlog (background maintenance and
    /// checkpoints drain the rest), then abandoned reservations.
    ///
    /// Only past [`WRITE_PATH_REPAIR_BACKLOG`] queued deletes: this runs
    /// under the sequencer every concurrent writer of the field waits on, so
    /// repairing on every write serialized an update-heavy workload behind
    /// graph removals (each costs about an insert). Below the backlog the
    /// background repair (`repair_vectors`) keeps up; searches never depend
    /// on repair (reader views mask deleted docs). Above it the write path
    /// still helps, so the queue stays bounded.
    fn drain_on_write_locked(&self) -> Result<()> {
        if self.pending_deletes.lock().len() > WRITE_PATH_REPAIR_BACKLOG {
            self.drain_batch_locked(WRITE_PATH_DELETE_DRAIN)?;
        }
        self.drain_abandoned_locked(WRITE_PATH_DELETE_DRAIN)
    }

    fn drain_locked(&self) -> Result<()> {
        loop {
            let (processed, more) = self.drain_batch_locked(usize::MAX)?;
            if processed == 0 || !more {
                break;
            }
        }
        self.drain_abandoned_locked(usize::MAX)
    }

    /// `reclaim_max`: source rows of retired nodes released at most (the
    /// write path bounds it; maintenance releases the whole backlog).
    fn drain_abandoned_locked(&self, reclaim_max: usize) -> Result<()> {
        let mut abandoned = self.abandoned.lock();
        while let Some(&handle) = abandoned.last() {
            {
                let source = self.owner.pin()?;
                self.graph.remove(handle, &source)?;
            }
            abandoned.pop();
            self.owner.commit()?;
        }
        // Cancelled reservations can own source rows even without a completed
        // node; drain the graph's acknowledgment queue in that case too.
        self.owner.commit()?;
        // Release the source rows (ArrayStore rows, code log tags) of nodes
        // whose graph retirement no reader can still observe, then let the
        // graph reuse their slots. Nothing else collects these
        // acknowledgments, so without this every removed node kept its row
        // and slot, and the code log never saw its rows turn dead.
        self.owner.reclaim_some(reclaim_max)?;
        Ok(())
    }
}

/// Vespa's `filter_first_exploration` for ES filtered kNN, by beam width
/// `ef` (num_candidates): how small a 2-hop neighbourhood (relative to
/// `(2m)^2` queued ids) makes filter-first expand a third hop, which it then
/// does only until the neighbourhood holds `2m` accepted nodes. Vespa's
/// default, 0.01, left so_vector's vector-correlated `tags: java` filter
/// (7.8%) at recall@10 0.796 (ES 9.5.5: 0.811 force-merged, 0.876 with 17
/// segments). Exploration falls with the beam, `FILTER_FIRST_EXPLORATION_EF
/// / ef` between Vespa's default and 1: measured on that data (1M docs,
/// force-merged, HTTP p50), 80 gives recall@10 0.869 at 2.95 ms (ES 0.811 at
/// 2.84 ms, 0.876 at 4.18 ms with segments) and recall@100 0.906 at 8.17 ms
/// (ES 0.875 at 8.49 ms, 0.916 at 9.36 ms); 30 gives 0.870 at 2.86 ms but
/// 0.885 at k100, 50 sits between. Random-range filters are unchanged
/// (their 2-hop neighbourhoods fill with accepted nodes).
const FILTER_FIRST_EXPLORATION_EF: f64 = 80.0;
/// Filter hit ratios at or below which filter-first keeps Vespa's default
/// exploration, and from which it uses the full `ef`-scaled one (linear in
/// between). On a very sparse filter a third hop mostly queues rejected
/// nodes: at 2-5% (c31 vs the ef-scaled exploration, 200k x 96-d) k10 rows
/// cost 1.3-4.5x for recall that stays low either way, while so_vector's
/// 7.8% `tags: java` filter is where the third hop pays (recall@10 0.796 ->
/// 0.884, at ES's latency).
const FILTER_FIRST_SPARSE_RATIO: f64 = 0.05;
const FILTER_FIRST_EXPLORE_RATIO: f64 = 0.07;
fn filter_first_exploration(ef: usize, hit_ratio: Option<f64>) -> f64 {
    // Tuning seam for the so_vector comparison.
    static EF: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *EF.get_or_init(|| std::env::var("RUSTIC_FF_EXPLORATION_EF").ok().and_then(|v| v.parse().ok()).filter(|v: &f64| *v > 0.0).unwrap_or(FILTER_FIRST_EXPLORATION_EF));
    const VESPA_DEFAULT: f64 = 0.01;
    let wide = (scale / ef.max(1) as f64).clamp(VESPA_DEFAULT, 1.0);
    let weight = hit_ratio.map_or(1.0, |ratio| ((ratio - FILTER_FIRST_SPARSE_RATIO) / (FILTER_FIRST_EXPLORE_RATIO - FILTER_FIRST_SPARSE_RATIO)).clamp(0.0, 1.0));
    VESPA_DEFAULT + (wide - VESPA_DEFAULT) * weight
}
/// While searches run (`crate::searches_active`), a bulk slice prepares its
/// rows on 1/this of the preparation pool, in one chunk per worker. Measured
/// on the Rally-shaped concurrency bench (c31 / 1 / 2 / 3): single-client
/// append 935 / 2,090 / 1,377 / 1,375 docs/s, kNN p99 during indexing
/// 8.2 / 9.6 / 11.5 / 11.4 ms (free-running preparation: 2,640 docs/s at
/// 11.0-11.5 ms). The chunked full pool is the chosen trade-off; the seam
/// keeps the share tunable.
const PREPARE_SHARE_WHILE_SEARCHING: usize = 1;
fn prepare_share_while_searching() -> usize {
    // Tuning seam for the concurrency bench.
    static SHARE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| std::env::var("RUSTIC_PREPARE_SEARCH_SHARE").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(PREPARE_SHARE_WHILE_SEARCHING))
}
/// The pool staged graph preparation runs on ([`set_prepare_pool`]).
static PREPARE_POOL: std::sync::OnceLock<&'static rayon::ThreadPool> = std::sync::OnceLock::new();

/// Run staged graph preparation (`stage_documents`' parallel prepare
/// tasks) on `pool`. Only those tasks run there: the staging thread (a
/// write thread) reserves, waits for them and completes the slice itself,
/// where staging used to run whole on a pool thread. A pool thread then
/// held a slice's serial completion, and while it waited for its own
/// slice's tasks it took other bulks' whole stagings (rayon work stealing),
/// so few slices were in flight and the pool's cores sat idle (a 32-core
/// node indexing explicit-id int8 vectors at ~18 cores busy).
pub fn set_prepare_pool(pool: &'static rayon::ThreadPool) {
    let _ = PREPARE_POOL.set(pool);
}

/// `f` on the prepare pool ([`set_prepare_pool`]); inline without one or on
/// one of its threads. A thread of another rayon pool waits on a plain
/// scoped thread instead of installing directly: a cross-pool install would
/// run its own pool's jobs while it waits, which can re-enter a shard lock
/// the caller holds.
fn on_prepare_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match PREPARE_POOL.get() {
        Some(pool) if pool.current_thread_index().is_none() => {
            if rayon::current_thread_index().is_some() {
                std::thread::scope(|scope| scope.spawn(|| pool.install(f)).join().expect("prepare pool work panicked"))
            } else {
                pool.install(f)
            }
        }
        _ => f(),
    }
}

/// Rows a slice prepared on its own thread completes at a time (see
/// `stage_slice`'s waves): few, so rows of one bulk see each other.
const SERIAL_PREPARE_WAVE: usize = 2;
/// [`SERIAL_PREPARE_WAVE`], or `RUSTIC_PREPARE_WAVE` (diagnostics).
fn serial_prepare_wave() -> usize {
    static WAVE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WAVE.get_or_init(|| std::env::var("RUSTIC_PREPARE_WAVE").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(SERIAL_PREPARE_WAVE))
}

/// Fewest prepare pool threads a slice's parallel preparation takes (see
/// `Staging::workers`).
const MIN_PARALLEL_PREPARE: usize = 4;

/// Slices preparing now, node-wide (see `stage_slice`).
static STAGING: AtomicUsize = AtomicUsize::new(0);

/// One slice's preparation, counted in [`STAGING`] while alive.
struct Staging(usize);
impl Staging {
    fn enter() -> Self {
        Self(STAGING.fetch_add(1, Ordering::AcqRel) + 1)
    }
    /// Parallel preparation width for this slice: the prepare pool's
    /// threads (the caller's pool without one) shared among the slices
    /// preparing now; 1 = prepare on the calling thread alone. A share
    /// under [`MIN_PARALLEL_PREPARE`] threads is not worth the hand-off:
    /// with that many bulks staging the write threads already fill the
    /// cores, and a slice waiting for a few pool threads held its write
    /// thread idle (~5 of 32 write threads at 28 bulk clients).
    fn workers(&self) -> usize {
        let share = self.share();
        if share < MIN_PARALLEL_PREPARE { 1 } else { share }
    }
    /// This slice's share of the prepare pool's threads (at least 1).
    fn share(&self) -> usize {
        let threads = PREPARE_POOL.get().map_or_else(rayon::current_num_threads, |pool| pool.current_num_threads());
        (threads / self.0.max(1)).max(1)
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        STAGING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// [`STAGE_SLICE_DOCS`], or `RUSTIC_STAGE_SLICE_DOCS` (diagnostics).
fn stage_slice_docs() -> usize {
    static DOCS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DOCS.get_or_init(|| std::env::var("RUSTIC_STAGE_SLICE_DOCS").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(STAGE_SLICE_DOCS))
}
/// Fewest rows one parallel preparation task takes (`RUSTIC_PREPARE_TASK_ROWS`,
/// diagnostics; default 1).
fn prepare_task_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| std::env::var("RUSTIC_PREPARE_TASK_ROWS").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(1))
}
/// Documents a bulk stages per slice (see `stage_documents`).
const STAGE_SLICE_DOCS: usize = 64;
/// Committed points an open-time catch-up / rebuild inserts per batch.
const CATCH_UP_BATCH: usize = if cfg!(test) { 64 } else { 4096 };

#[cfg(test)]
thread_local! {
    /// Most rows one catch-up batch buffered on this thread (tests).
    pub(crate) static CATCH_UP_MAX_BUFFERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Queued deletes repaired per `reserve` hold by maintenance and
/// [`MutableIngestField::drain_abandoned`].
const RELEASED_DRAIN_BATCH: usize = 1024;

/// Queued deletes a staging writer repairs at most, under `reserve`.
const WRITE_PATH_DELETE_DRAIN: usize = 256;
/// Queued deletes above which a staging writer repairs some itself
/// ([`MutableIngestField::drain_on_write_locked`]).
const WRITE_PATH_REPAIR_BACKLOG: usize = 16_384;

/// Retained topology and source for an ES PIT; never a writable coordinator.
///
/// The graph pin can be released ([`CapturedMutableField::release`]) when
/// retained views hold too much retired graph memory
/// ([`enforce_retained_budget`]): the view then serves kNN by exact search
/// over its own vectors (its vector segments stay pinned by the view), so
/// results stay correct and only that view's latency degrades. Writers
/// never wait for or fail because of a retained view.
pub struct CapturedMutableField {
    graph: parking_lot::RwLock<Option<CapturedGraph>>,
    similarity: Similarity,
    document_limit: u32,
}

struct CapturedGraph {
    identities: PointHandles,
    graph: super::hnsw::MutableHnswSnapshot,
    source: super::hnsw::CapturedMutableVectorRead,
    /// The capture's own copies (identity directory, topology, row
    /// references) on node-wide accounting, released with the pin.
    _charge: crate::accounting::Charge,
}

impl CapturedMutableField {
    fn new(identities: PointHandles, graph: super::hnsw::MutableHnswSnapshot, source: super::hnsw::CapturedMutableVectorRead) -> Arc<Self> {
        let bytes = (identities.heap_bytes() + graph.captured_bytes() + source.captured_bytes()) as u64;
        let charge = crate::accounting::Charge::new(crate::accounting::Category::RetainedCaptures, bytes);
        let field = Arc::new(Self { similarity: graph.similarity(), document_limit: graph.num_nodes().saturating_add(1),
            graph: parking_lot::RwLock::new(Some(CapturedGraph { identities, graph, source, _charge: charge })) });
        register_retained(&field);
        field
    }

    fn released_error() -> RusticError {
        RusticError::Segment(RELEASED_CAPTURE.into())
    }

    /// Read the f32 of code-only rows from `raw` (the retained store).
    pub(crate) fn bind_raw(&self, raw: &dyn super::hnsw::RawRows) {
        if let Some(captured) = self.graph.read().as_ref() {
            captured.source.bind_raw(raw);
        }
    }

    /// Whether the graph pin was released (kNN must search exactly).
    pub(crate) fn is_released(&self) -> bool {
        self.graph.read().is_none()
    }

    /// Drop the graph pin unless a search is using it right now (never
    /// waits). Returns whether it is released now.
    pub(crate) fn release(&self) -> bool {
        let Some(mut graph) = self.graph.try_write() else { return false };
        drop(graph.take());
        true
    }

    /// Retired graph memory of the live graph this capture pins, that
    /// graph's identity, and the capture's own growing memory (rows it
    /// decoded from codes) (`None`: released).
    fn pinned_retired_bytes(&self) -> Option<(usize, usize, usize)> {
        let graph = self.graph.try_read()?;
        let captured = graph.as_ref()?;
        Some((captured.graph.graph_identity(), captured.graph.retired_adjacency_bytes(), captured.source.decoded_bytes()))
    }

    pub(crate) fn search_ann_points(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        points: &[u64], deadline: Option<std::time::Instant>) -> Result<super::hnsw::MutableSearchResult> {
        let guard = self.graph.read();
        let Some(captured) = guard.as_ref() else { return Err(Self::released_error()) };
        let mut filter = empty_node_filter(captured.graph.num_nodes())?;
        captured.identities.visit(points, |handle| {
            let id = handle.slot() as usize + 1;
            if let Some(word) = filter.get_mut(id / 64) { *word |= 1u64 << (id % 64); }
        });
        if plan.route == super::vespa_policy::Route::Exact {
            return execute_prepared_exact(&captured.graph, &captured.source, &filter, query,
                plan.requested_target_hits as usize, deadline, plan.exact_codes);
        }
        execute_ann(&captured.graph, &captured.source, query, plan, |handle, _| {
            let id = handle.slot() as usize + 1;
            filter.get(id / 64).is_some_and(|word| word & (1u64 << (id % 64)) != 0)
        }, deadline, Some(&filter))
    }

    pub(crate) fn similarity(&self) -> Similarity { self.similarity }
    /// Each point's own code score for `query` (see
    /// [`MutableIngestField::code_scores`]); `None` per point without codes
    /// (an f32 field, a released capture, a point outside the capture).
    pub(crate) fn code_scores(&self, query: &[f32], points: &[u64]) -> Vec<Option<f32>> {
        use super::hnsw::MutableVectorSource;
        let guard = self.graph.read();
        let Some(captured) = guard.as_ref() else { return vec![None; points.len()] };
        let Some(scorer) = captured.source.bind_query(query) else { return vec![None; points.len()] };
        points.iter().map(|point| captured.identities.get(point).and_then(|handle| scorer.score(*handle, *point))).collect()
    }
    /// The live owner's score quantizer; `None` once released (exact search).
    pub(crate) fn score_quantizer(&self) -> Option<super::es_quant::EsQuantizer> {
        self.graph.read().as_ref()?.source.score_quantizer()
    }
    pub(crate) fn document_limit(&self) -> u32 { self.document_limit }
    pub fn search_ann(&self, query: &[f32], plan: &super::vespa_policy::QueryPlan,
        accept: impl Fn(u64) -> bool, deadline: Option<std::time::Instant>)
        -> Result<super::hnsw::MutableSearchResult>
    {
        let guard = self.graph.read();
        let Some(captured) = guard.as_ref() else { return Err(Self::released_error()) };
        execute_ann(&captured.graph, &captured.source, query, plan, |_, key| accept(key), deadline, None)
    }
}

/// [`MutableIngestField::open_checkpoint`].
pub(crate) enum OpenedCheckpoint {
    Restored(Arc<MutableIngestField>),
    /// Rebuild the field from the committed vector rows, with this graph
    /// configuration (once the field's options are known).
    Rebuild { dim: u32, similarity: Similarity, params: HnswParams },
}

/// Whether `error` is a checkpoint row whose referenced f32 is gone from
/// the vector store (restore cannot rebuild it from the file).
fn is_missing_referenced_row(error: &RusticError) -> bool {
    matches!(error, RusticError::Corrupted(message) if message.contains("referenced source row missing"))
}

/// Error message of a search on a released capture (callers fall back to
/// exact search over the view's vectors).
pub(crate) const RELEASED_CAPTURE: &str = "retained mutable graph released";

/// Whether `error` is a search on a released capture.
pub(crate) fn is_released_capture(error: &RusticError) -> bool {
    matches!(error, RusticError::Segment(message) if message == RELEASED_CAPTURE)
}

static RETAINED: parking_lot::Mutex<Vec<std::sync::Weak<CapturedMutableField>>> = parking_lot::const_mutex(Vec::new());
static DEGRADED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn register_retained(field: &Arc<CapturedMutableField>) {
    let mut retained = RETAINED.lock();
    retained.retain(|weak| weak.strong_count() > 0);
    retained.push(Arc::downgrade(field));
}

/// Node-wide budget for retired graph memory held by retained views (PIT,
/// scroll, ES|QL captures): `RUSTIC_RETAINED_GRAPH_BUDGET_BYTES`, default
/// 1 GiB.
pub fn retained_graph_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| std::env::var("RUSTIC_RETAINED_GRAPH_BUDGET_BYTES").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(1 << 30))
}

/// Retired graph bytes pinned by unreleased retained captures now (each
/// graph counted once): what [`enforce_retained_budget`] bounds.
pub fn retained_graph_bytes() -> usize {
    // Upgraded outside the lock's scope: dropping the last strong reference
    // runs the capture's drop, which must not run under `RETAINED`.
    let live: Vec<Arc<CapturedMutableField>> = RETAINED.lock().iter().filter_map(std::sync::Weak::upgrade).collect();
    let mut graphs: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut own = 0usize;
    for field in &live {
        if let Some((graph, bytes, decoded)) = field.pinned_retired_bytes() {
            graphs.insert(graph, bytes);
            own = own.saturating_add(decoded);
        }
    }
    graphs.values().sum::<usize>().saturating_add(own)
}

/// Retained views whose graph pin was released to bound memory (they serve
/// kNN by exact search since).
pub fn degraded_retained_views() -> u64 {
    DEGRADED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bound retired graph memory held by retained views without ever blocking
/// a writer: while the retired adjacency bytes of the graphs pinned by
/// unreleased captures exceed `budget`, release the oldest capture (one
/// that is not searching right now). Retired memory is reclaimed by the
/// graph's normal maintenance once nothing older pins it, so at most one
/// capture is released per call (callers run it periodically, off every
/// write path). Returns how many captures were released.
pub fn enforce_retained_budget(budget: usize) -> usize {
    let live: Vec<Arc<CapturedMutableField>> = {
        let mut retained = RETAINED.lock();
        retained.retain(|weak| weak.strong_count() > 0);
        retained.iter().filter_map(std::sync::Weak::upgrade).collect()
    };
    let mut released = 0;
    // Oldest first (registration order).
    for (position, field) in live.iter().enumerate() {
        let mut graphs: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut own = 0usize;
        for other in &live[position..] {
            if let Some((graph, bytes, decoded)) = other.pinned_retired_bytes() {
                graphs.insert(graph, bytes);
                own = own.saturating_add(decoded);
            }
        }
        // A capture's decoded rows count as its own retained memory.
        if graphs.values().sum::<usize>().saturating_add(own) <= budget { break; }
        if field.is_released() { continue; }
        if field.release() {
            released += 1;
            DEGRADED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            break;
        }
    }
    released
}

// Diagnostic switch for Vespa's existing tensor-prefetch option. It does not
// change traversal order, admission, distances, or construction prefetch policy.
fn query_tensor_prefetch() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RUSTIC_HNSW_PREFETCH_TENSORS").is_ok_and(|v| v == "1"))
}

fn execute_ann(graph: &super::hnsw::MutableHnswSnapshot,
    source: &(impl super::hnsw::MutableVectorSource + Sync), query: &[f32],
    plan: &super::vespa_policy::QueryPlan, accept: impl Fn(NodeHandle, u64) -> bool,
    deadline: Option<std::time::Instant>, node_filter: Option<&[u64]>) -> Result<super::hnsw::MutableSearchResult>
{
        if query.len() != graph.dim() as usize || !graph.similarity().valid_row(query) {
            return Err(RusticError::InvalidQuery("invalid mutable ANN query vector".into()));
        }
        use super::vespa_policy::{Route, FilterApplication, Reason};
        use super::hnsw::{MutableSearchMode, VespaFilterEstimate, VespaFilterFirstParams, VespaPlainParams};
        if plan.route == Route::Exact {
            let mut node_filter = empty_node_filter(graph.num_nodes())?;
            for (key, handle) in graph.visible_identities() {
                if accept(handle, key) {
                    let id = handle.slot() as usize + 1;
                    if node_filter.len() <= id / 64 {return Err(RusticError::Memory("node filter bound changed".into()));}
                    node_filter[id / 64] |= 1u64 << (id % 64);
                }
            }
            return execute_prepared_exact(graph, source, &node_filter, query,
                plan.requested_target_hits as usize, deadline, plan.exact_codes);
        }
        if plan.filter_application == FilterApplication::AnnUnfilteredThenOuterQuery
            && plan.reason != Reason::ConstructedFilterMatchedAll {
            return Err(RusticError::Unsupported("ES knn prefilter cannot use Vespa postfilter execution".into()));
        }
        let width = plan.exploration_budget.ok_or_else(|| RusticError::InvalidQuery("missing ANN exploration budget".into()))? as usize;
        let estimate = plan.filter_statistics.map(|s| VespaFilterEstimate { size: s.size, count: s.count });
        let mode = match plan.route {
            Route::FilterFirst => MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                exploration: filter_first_exploration(width, estimate.filter(|e| e.size > 0).map(|e| f64::from(e.count) / f64::from(e.size))),
                filter_estimate: estimate, deadline, prefetch_tensors: query_tensor_prefetch(), ..Default::default()
            }),
            Route::Plain => MutableSearchMode::VespaPlain(VespaPlainParams {
                filter_estimate: estimate, deadline, prefetch_tensors: query_tensor_prefetch(), ..Default::default()
            }),
            Route::Exact => unreachable!(),
        };
        // Pin topology before source, as required by directory growth/reuse.
        // Every caller passing a node filter passes `accept` as exactly its
        // bitmap (prepared and point-set searches).
        // A plain beam over a prepared node filter on a node with idle
        // cores: split over the search pool (`parallel_beam`).
        if let (Route::Plain, Some(words), MutableSearchMode::VespaPlain(params)) = (plan.route, node_filter, mode) {
            let workers = super::hnsw::beam_workers(width);
            if workers > 1 {
                let parallel = crate::query::searcher::in_search_pool(|| graph.search_parallel_node_filter(
                    query, plan.adjusted_target_hits as usize, width, source, params, words, workers))?;
                if let Some(result) = parallel {
                    source.execution_result()?;
                    if result.missing_source_lookups != 0 {
                        return Err(RusticError::Segment("mutable ANN encountered a missing live source row".into()));
                    }
                    return Ok(result);
                }
            }
        }
        let result = match node_filter {
            Some(words) => graph.search_with_exact_node_filter(query, plan.adjusted_target_hits as usize,
                width, source, accept, mode, words)?,
            None => graph.search_with_handles_and_node_filter(query, plan.adjusted_target_hits as usize,
                width, source, accept, mode, None)?,
        };
        source.execution_result()?;
        if result.missing_source_lookups != 0 {
            return Err(RusticError::Segment("mutable ANN encountered a missing live source row".into()));
        }
        Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hnsw::MutableVectorSource;

    #[test]
    fn migration_copies_admissions_accepted_out_of_order_and_admits_after_retirement() {
        let store = VectorStore::new();
        let previous = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        // Two admissions reserve increasing points; the later one is
        // accepted first.
        let older = previous.stage(&[&[1.,2.], &[3.,4.]]).unwrap();
        let newer = previous.stage(&[&[5.,6.]]).unwrap();
        let (newer, _) = newer.accept().unwrap();
        let fresh = MutableIngestField::new(store.clone(), 1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let mut copied = std::collections::HashSet::new();
        // Only the points below the in-flight admission are settled.
        assert_eq!(fresh.catch_up_from(&previous, &mut copied).unwrap(), 0);
        let (older, _) = older.accept().unwrap();
        previous.retire_into(&fresh, &mut copied).unwrap();
        let held = |point| fresh.identities.read().get(&point).is_some();
        assert!(older.points().chain(newer.points()).all(held), "every admitted point copied");
        // Admissions after the retirement go to the successor's graph.
        let (after, _) = previous.stage(&[&[7.,8.]]).unwrap().accept().unwrap();
        assert!(after.points().all(held));
        assert!(fresh.admitting.lock().is_empty() && previous.admitting.lock().is_empty());
    }

    #[test]
    fn projection_reuses_only_matching_document_masks_and_identity_revision() {
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let lease = field.stage(&[&[1.,2.], &[3.,4.]]).unwrap().accept().unwrap().0;
        let points: Vec<_> = lease.points().collect();
        let key = vec![(1, vec![3])];
        let first = field.prepare_ann_projection(key.clone(), || points.clone()).unwrap();
        let bits = first.node_filter.clone();
        assert_eq!(first.matched, 2);
        drop(first);
        let cached = field.prepare_ann_projection(key.clone(), || -> Vec<u64> { panic!("same view must reuse projection") }).unwrap();
        assert!(Arc::ptr_eq(&bits, &cached.node_filter));
        drop(cached);
        let duplicates = field.prepare_ann_projection(vec![(2, vec![7])],
            || [points[0], points[0], u64::MAX]).unwrap();
        assert_eq!(duplicates.matched, 1, "count distinct resolved graph nodes only");
        drop(duplicates);
        let filtered = field.prepare_ann_projection(vec![(1, vec![1])], || vec![points[0]]).unwrap();
        assert_eq!(filtered.matched, 1);
        assert!(!Arc::ptr_eq(&bits, &filtered.node_filter));
        drop(filtered);
        drop(lease);
        field.drain_abandoned().unwrap();
        let replacement = field.stage(&[&[5.,6.]]).unwrap().accept().unwrap().0;
        let rebuilt = field.prepare_ann_projection(key, || replacement.points()).unwrap();
        assert_eq!(rebuilt.matched, 1);
        assert!(!Arc::ptr_eq(&bits, &rebuilt.node_filter));
    }

    /// An empty projection mask (an unfiltered leaf without deletions) is
    /// every doc of the leaf: the same filter as an explicit all-ones mask.
    #[test]
    fn empty_projection_mask_selects_every_doc() {
        use crate::codec::vectors::{encode_point_map, PointMapReader, NO_POINT};
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let lease = field.stage(&[&[1.,2.], &[3.,4.], &[5.,6.]]).unwrap().accept().unwrap().0;
        let points: Vec<_> = lease.points().collect();
        // 70 docs: a full word and a partial one, holes included.
        let column: Vec<u64> = (0..70).map(|d| if d % 3 == 1 { NO_POINT } else { points[d % 3] }).collect();
        let bytes = encode_point_map(&[(1, column)]);
        let reader = PointMapReader::open(&bytes).unwrap();
        let all = field.prepare_ann_segments(vec![(120, vec![])], || vec![reader.column(1)]).unwrap();
        let explicit = field.prepare_ann_segments(vec![(121, vec![u64::MAX, (1u64 << 6) - 1])], || vec![reader.column(1)]).unwrap();
        assert_eq!(all.node_filter.as_slice(), explicit.node_filter.as_slice());
        assert_eq!(all.matched, explicit.matched);
        assert_eq!(all.matched, 2);
    }

    #[test]
    fn segment_projection_preserves_holes_masks_and_identity_changes() {
        use crate::codec::vectors::{encode_point_map, PointMapReader, NO_POINT};
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let lease = field.stage(&[&[1.,2.], &[3.,4.]]).unwrap().accept().unwrap().0;
        let points: Vec<_> = lease.points().collect();
        let bytes = encode_point_map(&[(1, vec![points[1], NO_POINT, points[0], points[1], 999999])]);
        let reader = PointMapReader::open(&bytes).unwrap();
        for mask in [31u64, 1, 4, 8, 2, 16, 128, 5, 9, 12, 31] {
            let key = vec![(100, vec![mask])];
            let actual = field.prepare_ann_segments(key.clone(), || vec![reader.column(1)]).unwrap();
            // Different exact-cache key forces the independent point-based oracle.
            let expected = field.prepare_ann_projection(vec![(101, vec![mask])], ||
                (0..8).filter(move |doc| mask & (1 << doc) != 0).filter_map(|doc| reader.get(1,doc))).unwrap();
            assert_eq!(actual.node_filter.as_slice(), expected.node_filter.as_slice());
            assert_eq!(actual.matched, expected.matched);
        }
        drop(lease); field.drain_abandoned().unwrap();
        let replacement = field.stage(&[&[5.,6.]]).unwrap().accept().unwrap().0;
        let stale = field.prepare_ann_segments(vec![(100, vec![31])], || vec![reader.column(1)]).unwrap();
        assert_eq!(stale.matched, 0, "old document IDs must not expose reused graph slots");
        drop(stale);
        let bytes = encode_point_map(&[(1, replacement.points().collect())]);
        let reader = PointMapReader::open(&bytes).unwrap();
        assert_eq!(field.prepare_ann_segments(vec![(102, vec![1])], || vec![reader.column(1)]).unwrap().matched, 1);
    }

    #[test]
    fn delete_repair_keeps_cached_segment_projections() {
        use crate::codec::vectors::{encode_point_map, PointMapReader};
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let lease = field.stage(&[&[1.,2.], &[3.,4.], &[5.,6.]]).unwrap().accept().unwrap().0;
        let points: Vec<_> = lease.points().collect();
        let bytes = encode_point_map(&[(1, points.clone())]);
        let reader = PointMapReader::open(&bytes).unwrap();
        let first = field.prepare_ann_segments(vec![(100, vec![0b111])], || vec![reader.column(1)]).unwrap();
        assert_eq!(first.matched, 3);
        let bits = first.node_filter.clone();
        drop(first);
        // Delete repair of doc 2's point (a view that still saw doc 2 live
        // would have deferred it): the identity revision changes, the
        // cached projection of an unchanged mask stays valid and is reused.
        field.remove_deleted_point(points[2]).unwrap();
        let reused = field.prepare_ann_segments(vec![(100, vec![0b111])], || -> Vec<_> { panic!("the projection is cached") }).unwrap();
        assert!(Arc::ptr_eq(&bits, &reused.node_filter));
        drop(reused);
        // A view masking doc 2 builds from the cached doc -> node map.
        let masked = field.prepare_ann_segments(vec![(100, vec![0b011])], || vec![reader.column(1)]).unwrap();
        assert_eq!(masked.matched, 2);
    }

    #[test]
    fn abandoned_unchanged_document_lease_preserves_original_node() {
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let original = field.stage(&[&[1.,2.]]).unwrap().accept().unwrap().0;
        let point = original.points().start;
        let reused = field.reuse_unchanged(point, &[1.,2.]).unwrap().unwrap();
        drop(reused); field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 1);
        assert!(field.reuse_unchanged(point, &[2.,2.]).unwrap().is_none());
        drop(original); field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 0);
    }

    #[test]
    fn shared_field_search_obeys_document_visibility_and_vespa_route() {
        use super::super::vespa_policy::*;
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let values: Vec<_> = (0..100).map(|n| [n as f32, 1.]).collect();
        let rows: Vec<_> = values.iter().map(|v| v.as_slice()).collect();
        let batch = field.stage(&rows).unwrap();
        for count in [10, 50, 100] {
            let plan = choose_filter_preparation(PreparationInput {
                approximate: true, has_index: true, estimated_query_ratio: count as f64 / 100.,
            }, PolicyConfig::default()).unwrap().finish(HitBudget {
                target_hits: 3, explore_additional_hits: 97,
            }, 100, PreparedGlobalFilter::Active(FilterStatistics { size: 100, count })).unwrap();
            let result = field.search_ann(&[0., 1.], &plan, |key| key < count as u64, None).unwrap();
            assert_eq!(result.hits.len(), 3, "route {:?}, count {count}", plan.route);
            assert!(result.hits.iter().all(|(key, _)| *key < count as u64));
            assert_eq!(result.hits[0].0, 0);
            let predicate_calls = AtomicUsize::new(0);
            let prepared = field.prepare_ann(|key| {
                predicate_calls.fetch_add(1, Ordering::Relaxed);
                key < count as u64
            }).unwrap();
            let preparation_calls = predicate_calls.load(Ordering::Relaxed);
            assert_eq!(preparation_calls, 100);
            assert_eq!(prepared.search(&[0., 1.], &plan, None).unwrap().hits, result.hits);
            assert_eq!(predicate_calls.load(Ordering::Relaxed), preparation_calls,
                "query traversal must use captured eligibility, not lock live document owners");
            // An adversarial query outside a correlated matching range can have
            // short ANN results. Compare the integration with the direct native
            // traversal rather than silently demanding an exact-fill fallback.
            let distant = field.search_ann(&[99., 1.], &plan, |key| key < count as u64, None).unwrap();
            use super::super::hnsw::{MutableSearchMode, VespaFilterEstimate, VespaFilterFirstParams, VespaPlainParams};
            let estimate = Some(VespaFilterEstimate { size: 100, count });
            let mode = if plan.route == Route::FilterFirst {
                MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams { filter_estimate: estimate, ..Default::default() })
            } else {
                MutableSearchMode::VespaPlain(VespaPlainParams { filter_estimate: estimate, ..Default::default() })
            };
            let (graph, source) = field.owner.read().unwrap();
            let direct = graph.search_mode(&[99., 1.], 3, 100, &source, |key| key < count as u64, mode).unwrap();
            assert_eq!(distant.hits, direct.hits, "adapter must preserve native traversal behavior");
            eprintln!("correlated-filter diagnostic: count={count} route={:?} direct_hits={:?}", plan.route, direct.hits);

            assert!(field.search_ann(&[99., 1.], &plan, |_| false, None).unwrap().hits.is_empty(),
                "graph-ready rows must not escape a caller's unpublished-document mask");
        }
        let exact = choose_filter_preparation(PreparationInput {
            approximate: true, has_index: true, estimated_query_ratio: 0.01,
        }, PolicyConfig::default()).unwrap().finish(HitBudget {
            target_hits: 3, explore_additional_hits: 97,
        }, 100, PreparedGlobalFilter::NotBuilt).unwrap();
        assert_eq!(field.search_ann(&[99., 1.], &exact, |key| key < 10, None).unwrap()
            .hits.iter().map(|hit| hit.0).collect::<Vec<_>>(), vec![9, 8, 7]);
        let prepared = field.prepare_ann(|key| key < 10).unwrap();
        let result = prepared.search(&[99., 1.], &exact, None).unwrap();
        assert_eq!(result.hits.iter().map(|hit| hit.0).collect::<Vec<_>>(), vec![9, 8, 7]);
        assert_eq!(result.vector_ops, 10);
        assert!(prepared.search(&[1.], &exact, None).is_err());
        assert!(prepared.search(&[99., 1.], &exact, Some(std::time::Instant::now())).is_err());
        assert!(field.prepare_ann(|_| false).unwrap().search(&[99.,1.], &exact, None).unwrap().hits.is_empty());
        drop(batch);
        field.drain_abandoned().unwrap();
    }

    #[test]
    fn prepared_exact_parallel_ranges_preserve_sparse_scores_and_ties() {
        use super::super::vespa_policy::*;
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let values: Vec<_> = (0..4096).map(|n| [(n % 17) as f32, 1.]).collect();
        let rows: Vec<_> = values.iter().map(|v| v.as_slice()).collect();
        let batch = field.stage(&rows).unwrap();
        let prepared = field.prepare_ann(|key| key % 3 != 0).unwrap();
        let plan = choose_filter_preparation(PreparationInput {
            approximate: false, has_index: true, estimated_query_ratio: 0.7,
        }, PolicyConfig::default()).unwrap().finish(HitBudget {
            target_hits: 40, explore_additional_hits: 0,
        }, 4096, PreparedGlobalFilter::NotBuilt).unwrap();
        let query = [8., 1.];
        let mut expected: Vec<_> = values.iter().enumerate().filter(|(key, _)| key % 3 != 0)
            .map(|(key, row)| (key as u64, Similarity::L2.score(&query, row))).collect();
        let count = expected.len() as u64;
        expected.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        expected.truncate(40);
        for _ in 0..3 {
            let result = prepared.search(&query, &plan, None).unwrap();
            assert_eq!(result.hits, expected);
            assert_eq!(result.vector_ops, count);
        }
        drop(prepared);
        drop(batch);
        field.drain_abandoned().unwrap();
    }

    #[test]
    fn store_registry_owns_one_graph_per_field_and_rejects_configuration_drift() {
        let store = VectorStore::new();
        let params = HnswParams::default();
        let owners = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..8).map(|_| {
                let store = store.clone();
                scope.spawn(move || store.mutable_field(1, 2, Similarity::L2, params).unwrap())
            }).collect();
            jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(owners.iter().all(|owner| Arc::ptr_eq(owner, &owners[0])));
        let weak = Arc::downgrade(&owners[0]);
        drop(owners);
        assert!(weak.upgrade().is_some(), "store retains graph between text writers");
        assert!(store.mutable_field(1, 3, Similarity::L2, params).is_err());
        assert!(store.mutable_field(1, 2, Similarity::Cosine, params).is_err());
        assert!(store.mutable_field(1, 2, Similarity::L2, HnswParams { m: 32, ..params }).is_err());
        let other = store.mutable_field(2, 2, Similarity::L2, params).unwrap();
        assert!(!Arc::ptr_eq(&weak.upgrade().unwrap(), &other));
        assert!(store.capture_read_store().unwrap().mutable_field(1, 2, Similarity::L2, params).is_err());
        drop(other);
        drop(store);
        assert!(weak.upgrade().is_none(), "registry ownership must not leak via an Arc cycle");
    }

    #[test]
    fn graph_is_ready_before_identity_transfer_and_abandoned_batch_is_reclaimed() {
        let store = VectorStore::new();
        let field = MutableIngestField::new(store.clone(), 1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let batch = field.stage(&[&[1., 0.], &[0., 1.]]).unwrap();
        assert_eq!(batch.points(), 0..2);
        assert_eq!(field.graph.active_nodes(), 2);
        assert!(store.segment_ids().is_empty(), "no text/vector file publication");
        let source = field.owner.pin().unwrap();
        for (key, handle) in batch.points().zip(&batch.handles) { assert!(source.get_node(*handle, key).is_some()); }
        drop(source);
        drop(batch);
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 0);
        assert_eq!(field.inflight.load(Ordering::Acquire), 0);
        let batch = field.stage(&[&[2., 3.]]).unwrap();
        assert_eq!(batch.points(), 2..3, "abandoned point keys never repeat");
        let (reservation, handles) = batch.accept().unwrap();
        assert_eq!(reservation.points(), 2..3);
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 1);
        assert_eq!(field.owner.pin().unwrap().get_node(handles[0], 2), Some([2., 3.].as_slice()));
    }

    #[test]
    fn concurrent_batches_reserve_monotonic_keys_and_complete_before_transfer() {
        let store = VectorStore::new();
        let field = MutableIngestField::new(store.clone(), 1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let barrier = std::sync::Barrier::new(8);
        let batches = std::thread::scope(|scope| {
            let tasks: Vec<_> = (0..8).map(|n| {
                let field = field.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let values: Vec<_> = (0..32).map(|i| [n as f32, i as f32]).collect();
                    let rows: Vec<_> = values.iter().map(|v| v.as_slice()).collect();
                    let batch = field.stage(&rows).unwrap();
                    let points: Vec<_> = batch.points().collect();
                    let (reservation, _) = batch.accept().unwrap();
                    (points, reservation)
                })
            }).collect();
            tasks.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>()
        });
        let mut points: Vec<_> = batches.iter().flat_map(|(points, _)| points.iter().copied()).collect();
        points.sort_unstable();
        assert_eq!(points, (0..256).collect::<Vec<_>>());
        assert_eq!(field.graph.active_nodes(), 256);
        assert_eq!(field.inflight.load(Ordering::Acquire), 0);
        assert!(field.stage(&[&[f32::NAN, 0.]]).is_err());
        assert!(field.stage(&[&[1.]]).is_err());
        assert_eq!(store.next_point_id(), 256);
        drop(batches);
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 0, "discarded text reservations reclaim transferred nodes");
    }
}


#[cfg(test)]
mod writer_tests {
    use super::*;
    use crate::codec::directory::{Directory, FileSlice, RamDirectory};
    use crate::types::{Document, Field, FieldType};
    use crate::segment::SegmentWriter;
    use super::super::store::VectorFieldOptions;
    use std::sync::atomic::AtomicBool;

    #[derive(Debug, Default)]
    struct FailWrites { inner: RamDirectory, fail: AtomicBool }
    impl Directory for FailWrites {
        fn open_read(&self, n: &str) -> Result<FileSlice> { self.inner.open_read(n) }
        fn write(&self, n: &str, b: &[u8]) -> Result<()> {
            if self.fail.load(Ordering::Acquire) { return Err(RusticError::Io(std::io::Error::other("injected write failure"))); }
            self.inner.write(n, b)
        }
        fn list(&self) -> Result<Vec<String>> { self.inner.list() }
        fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        fn exists(&self, n: &str) -> bool { self.inner.exists(n) }
        fn delete(&self, n: &str) -> Result<()> { self.inner.delete(n) }
    }
    fn schema() -> Vec<Field> {
        vec![Field::new(1, "vector", FieldType::Vector { dim: 2 }).with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions::f32_hnsw()))]
    }
    fn document(x: f32) -> Document { let mut d = Document::new(); d.add_vector(1, vec![x, 1.]); d }

    #[test]
    fn retained_store_keeps_graph_source_and_document_owners_after_live_replacement() {
        use super::super::vespa_policy::*;
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let directory = RamDirectory::default();
        let mut writer = SegmentWriter::new(5);
        writer.add_document(&document(0.), &schema());
        writer.add_document(&document(10.), &schema());
        writer.stage_mutable_field(1, &field).unwrap();
        let segment = writer.write_to_store_ref(&directory, Some(&store), false).unwrap();
        let leaves = crate::segment::Segment::capture_read_view(&[Arc::new(segment)]).unwrap();
        let captured = leaves[0].vector_store().unwrap();
        let plan = choose_filter_preparation(PreparationInput {
            approximate: true, has_index: true, estimated_query_ratio: 1.,
        }, PolicyConfig::default()).unwrap().finish(HitBudget {
            target_hits: 1, explore_additional_hits: 19,
        }, 2, PreparedGlobalFilter::Inactive).unwrap();
        let before = captured.search_mutable_ann(1, &[0., 1.], &plan,
            |key| captured.owner(key).is_some(), None).unwrap().unwrap().hits;
        assert_eq!(before[0].0, 0);
        let handle = field.graph.snapshot().visible_identities().find(|(key, _)| *key == 0).unwrap().1;
        store.delete(0, 5, 0);
        {
            let source = field.owner.pin().unwrap();
            field.graph.remove(handle, &source).unwrap();
        }
        field.owner.commit().unwrap();
        let mut replacement = SegmentWriter::new(6);
        replacement.add_document(&document(0.), &schema());
        replacement.stage_mutable_field(1, &field).unwrap();
        let _segment = replacement.write_to_store_ref(&directory, Some(&store), false).unwrap();
        assert_eq!(store.search_mutable_ann(1, &[0., 1.], &plan,
            |key| store.owner(key).is_some(), None).unwrap().unwrap().hits[0].0, 2);
        assert_eq!(captured.search_mutable_ann(1, &[0., 1.], &plan,
            |key| captured.owner(key).is_some(), None).unwrap().unwrap().hits, before);
        let exact_plan = choose_filter_preparation(PreparationInput { approximate: false, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 1, explore_additional_hits: 0 }, 2, PreparedGlobalFilter::NotBuilt).unwrap();
        assert_eq!(captured.search_mutable_ann(1, &[0.,1.], &exact_plan,
            |key| captured.owner(key).is_some(), None).unwrap().unwrap().hits, before);
        let again = captured.capture_read_store().unwrap();
        drop(field);
        drop(store);
        assert_eq!(again.search_mutable_ann(1, &[0., 1.], &plan,
            |key| again.owner(key).is_some(), None).unwrap().unwrap().hits, before);
    }

    #[test]
    fn shared_graph_rows_survive_failed_flush_but_not_abandoned_writer() {
        let store = VectorStore::new();
        let field = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let mut writer = SegmentWriter::new(1);
        writer.add_document(&document(1.), &schema());
        assert_eq!(writer.stage_mutable_field(1, &field).unwrap(), 1);
        writer.add_document(&document(2.), &schema());
        assert_eq!(writer.stage_mutable_field(1, &field).unwrap(), 1);
        assert_eq!(writer.stage_mutable_field(1, &field).unwrap(), 0);
        assert_eq!(field.graph.active_nodes(), 2);
        assert_eq!(store.num_alive(1), 0, "graph readiness is not document publication");
        let identities = writer.reserved_vector_points(1);
        let directory = FailWrites::default();
        directory.fail.store(true, Ordering::Release);
        assert!(writer.write_to_store_ref(&directory, Some(&store), false).is_err());
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 2, "writer keeps retry ownership");
        directory.fail.store(false, Ordering::Release);
        writer.id = 17;
        let segment = writer.write_to_store_ref(&directory, Some(&store), false).unwrap();
        for (doc, point) in identities { assert_eq!(segment.point(1, doc), Some(point)); }
        drop(writer);
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 2, "published text now owns the points");
        assert_eq!(store.num_alive(1), 2);
        let shared = store.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        assert!(Arc::ptr_eq(&field, &shared), "new text writers reuse the existing graph");
        let mut abandoned = SegmentWriter::new(18);
        abandoned.add_document(&document(3.), &schema());
        abandoned.stage_mutable_field(1, &shared).unwrap();
        assert_eq!(field.graph.active_nodes(), 3);
        drop(abandoned);
        field.drain_abandoned().unwrap();
        assert_eq!(field.graph.active_nodes(), 2);
        let source = field.owner.pin().unwrap();
        let hits = field.graph.search(&[1., 1.], 3, 20, &source, |_| true).hits;
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|(key, _)| *key < 2));
    }

    #[test]
    fn writer_rejects_different_graph_owner_and_unsupported_mapping() {
        let store = VectorStore::new();
        let a = MutableIngestField::new(store.clone(), 1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let b = MutableIngestField::new(store.clone(), 1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let mut writer = SegmentWriter::new(0);
        writer.add_document(&document(1.), &schema());
        writer.stage_mutable_field(1, &a).unwrap();
        assert!(writer.stage_mutable_field(1, &b).is_err());
        let mut quantized = schema();
        quantized[0].vector_options = Some(VectorFieldOptions::quantized_hnsw(super::super::quantize::Quantization::Int8));
        let mut other = SegmentWriter::new(1);
        other.add_document(&document(2.), &quantized);
        assert!(other.stage_mutable_field(1, &a).is_err());
        let wrong_field = store.mutable_field(2, 2, Similarity::L2, HnswParams::default()).unwrap();
        let mut fresh = SegmentWriter::new(2);
        fresh.add_document(&document(2.), &schema());
        assert!(fresh.stage_mutable_field(1, &wrong_field).is_err());
        let mut custom = schema();
        custom[0].vector_options = Some(VectorFieldOptions::f32_hnsw().with_hnsw(32, 200));
        let mut mismatch = SegmentWriter::new(3);
        mismatch.add_document(&document(2.), &custom);
        assert!(mismatch.stage_mutable_field(1, &a).is_err());
        assert_eq!(store.next_point_id(), 1, "rejected mappings do not reserve IDs");
    }
}
