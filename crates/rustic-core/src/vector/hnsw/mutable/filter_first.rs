//! Opt-in port of Vespa's filter-first level-zero traversal, pinned to
//! 0521b00640ef934b359d2fc356f97d84499dd17d. This is not the immutable Rust
//! TwoHop strategy, and does not select a planner policy.
//!
//! Reference: searchlib/src/vespa/searchlib/tensor/hnsw_index.cpp,
//! search_layer_filter_first_helper / exploreNeighborhood / exploreNeighborhoodByOneHop.
//! https://github.com/vespa-engine/vespa/blob/0521b00640ef934b359d2fc356f97d84499dd17d/searchlib/src/vespa/searchlib/tensor/hnsw_index.cpp#L456-L617
//!
//! The graph's existing original-f32 kernels, score conversion, publication
//! bounds, generation-safe source ownership and final eligibility check remain
//! in use. This is current topology, not a historical PIT graph implementation.

use super::visited::{FilterEstimate, VisitedTracker, estimate_visited_nodes, LayerVisited, SparseVisited};
use super::*;
use std::collections::VecDeque;
use std::time::Instant;

/// Values match the pinned Vespa defaults in fef/indexproperties.cpp. The
/// deadline is optional for native callers; no planner thresholds live here.
/// Explicit filter-first calls require a precomputed filter_estimate.
#[derive(Clone, Copy, Debug)]
pub struct VespaFilterFirstParams {
    pub exploration: f64,
    pub exploration_slack: f64,
    pub prefetch_tensors: bool,
    pub deadline: Option<Instant>,
    pub filter_estimate: Option<VespaFilterEstimate>,
}
/// Document-domain filter metadata, not graph slot bounds or vector count.
/// The current POC tests one vector per document. Multi-vector callers must
/// supply document filter size/count independently of graph nodeid_limit.
#[derive(Clone, Copy, Debug)]
pub struct VespaFilterEstimate {
    pub size: u32,
    pub count: u32,
}

/// Plain search execution controls from the same pinned Vespa layer search.
/// Filter statistics remain document-domain metadata supplied by the planner.
#[derive(Clone, Copy, Debug, Default)]
pub struct VespaPlainParams {
    pub exploration_slack: f64,
    pub prefetch_tensors: bool,
    pub deadline: Option<Instant>,
    pub filter_estimate: Option<VespaFilterEstimate>,
}
impl VespaPlainParams {
    fn validate(self) -> Result<()> {
        if !self.exploration_slack.is_finite() || self.exploration_slack < 0.0 {
            return Err(failure("Plain exploration slack must be finite and nonnegative"));
        }
        if self.filter_estimate.is_some_and(|filter| filter.count > filter.size) {
            return Err(failure("filter count exceeds document filter size"));
        }
        Ok(())
    }
    pub(super) fn expired(self) -> bool {
        self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }
}
impl Default for VespaFilterFirstParams {
    fn default() -> Self {
        Self {
            exploration: 0.01,
            exploration_slack: 0.0,
            prefetch_tensors: false,
            deadline: None,
            filter_estimate: None,
        }
    }
}
impl VespaFilterFirstParams {
    fn validate(self) -> Result<()> {
        if !self.exploration.is_finite()
            || self.exploration < 0.0
            || !self.exploration_slack.is_finite()
            || self.exploration_slack < 0.0
        {
            return Err(failure(
                "filter-first exploration and slack must be finite and nonnegative",
            ));
        }
        let estimate = self
            .filter_estimate
            .ok_or_else(|| failure("filter-first requires precomputed filter size and count"))?;
        if estimate.count > estimate.size {
            return Err(failure("filter count exceeds document filter size"));
        }
        Ok(())
    }
    fn expired(self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

/// Plain preserves the existing search. Filter-first must be explicitly
/// requested; caller policy and filter evaluation are separate responsibilities.
#[derive(Clone, Copy, Debug, Default)]
pub enum MutableSearchMode {
    #[default]
    Plain,
    VespaPlain(VespaPlainParams),
    VespaFilterFirst(VespaFilterFirstParams),
}

#[derive(Default)]
struct FilterScratch<V: LayerVisited = VisitedTracker> {
    visited: V,
    candidates: BinaryHeap<Reverse<Candidate>>,
    best: BinaryHeap<Candidate>,
    todo: VecDeque<u32>,
    found: Vec<u32>,
    out: Vec<Candidate>,
}
thread_local! {
    // A retained visited bitmap (see `SparseVisited`), not one sized and
    // allocated per search as Vespa's bitmap / hash trackers are.
    static FILTER_SCRATCH: RefCell<FilterScratch<SparseVisited>> = RefCell::new(FilterScratch::default());
}

/// Exactly one queued wave. Queueing precedes visited/filter checks on purpose:
/// accepted and previously visited nodes are still bridges for the next hop.
/// The cap applies to accepted neighbors, not inspected links or queue entries.
/// Pinned filter-first reacquires CURRENT levels for every queued ID, including
/// the initial candidate. Plain's retained-candidate traversal is different.
fn one_hop<V: LayerVisited, const CAPTURED: bool>(
    view: &MutableHnswSnapshot,
    todo: &mut VecDeque<u32>,
    found: &mut Vec<u32>,
    visited: &mut V,
    cap: usize,
    accept: &impl Fn(NodeHandle, u64) -> bool,
    node_filter: Option<&[u64]>,
) {
    let wave = todo.len();
    // A live view's wave reads each queued node's words and level-0 links,
    // one dependent miss after another: warm them a few nodes ahead (the
    // dense beam's memory-level parallelism), without changing any read.
    let warm = if CAPTURED || wave < 4 { None } else { view.wave_warmer() };
    if let Some(warm) = &warm {
        for i in 0..WAVE_AHEAD.min(wave) { warm.words(todo[i]); }
        for i in 0..WAVE_LINKS_AHEAD.min(wave) { warm.links(todo[i]); }
    }
    for done in 0..wave {
        if found.len() >= cap {
            break;
        }
        if let Some(warm) = &warm {
            let left = wave - done;
            if WAVE_AHEAD < left { warm.words(todo[WAVE_AHEAD]); }
            if WAVE_LINKS_AHEAD < left { warm.links(todo[WAVE_LINKS_AHEAD]); }
        }
        let id = todo.pop_front().expect("current wave remains queued");
        // A live view under a prepared node filter (`accept` is exactly
        // that bitmap): visibility from the dense publication words and the
        // level-0 links from the hot slot, as the dense beam reads them, not
        // each node's record and levels (two more dependent misses a node).
        // Found nodes are checked again when scored (`visible_dense`).
        if let (Some(warm), Some(words)) = (&warm, node_filter) {
            match warm.visible_links(id, view.publication) {
                DenseNode::Invisible => continue,
                DenseNode::Links(links) => {
                    // Quiet (as the dense beam): these links were read while
                    // the topology sequence still held the view's quiet
                    // value, so every node they name is visible to it; no
                    // visibility word is read (each was a cache miss).
                    let quiet = view.quiet.is_some_and(|seq| view.control.topology_seq_after_fence() == seq);
                    for &neighbor_id in links {
                        if neighbor_id >= view.len { continue; }
                        todo.push_back(neighbor_id);
                        if visited.state(neighbor_id) != 0 { continue; }
                        if words.get(neighbor_id as usize / 64).is_none_or(|word| word & (1u64 << (neighbor_id % 64)) == 0) {
                            visited.insert(neighbor_id);
                            note_visited(1);
                            note_rejected(1);
                            continue;
                        }
                        if !(quiet || warm.visible(neighbor_id, view.publication)) || !visited.insert(neighbor_id) { continue; }
                        note_visited(1);
                        found.push(neighbor_id);
                        if found.len() >= cap { return; }
                    }
                    continue;
                }
                DenseNode::Generic => {}
            }
        }
        // A queued bridge removed concurrently (after this live reader read
        // the link naming it) is still expanded through its retained links;
        // it was never accepted, so it cannot become a hit.
        let Some(node) = view.current_node_typed::<CAPTURED>(id)
            .or_else(|| if CAPTURED { None } else { view.removed_bridge(id) }) else {
            continue;
        };
        if !view.visible(node) {
            continue;
        }
        let Some(links) = node.links_typed::<CAPTURED>(0) else {
            continue;
        };
        for &neighbor_id in links {
            if neighbor_id >= view.len {
                continue;
            }
            todo.push_back(neighbor_id);
            // Visited is local to this search and only transitions to marked.
            // Already marked IDs still bridge the next wave (queued above),
            // but their metadata cannot change this wave's result. Avoid
            // reloading current levels/publication for those duplicate edges.
            // Unmarked IDs retain the validity/publication-before-mark order.
            if visited.state(neighbor_id) != 0 {
                continue;
            }
            // Single-vector attribute IDs are graph node IDs. The caller
            // translates ES point/doc ownership into this prepared domain once.
            // A definitive rejection needs no node-table entry at all (no
            // levels, key, incarnation or publication), so it is decided
            // from the filter's bit before the node is looked up: on one
            // large graph, most neighbours of a selective filter are
            // rejected, and each lookup was a cache miss. Queued bridges
            // still undergo validity and visibility checks when expanded;
            // accepted nodes retain the full ES publication contract.
            if node_filter.is_some_and(|words| words.get(neighbor_id as usize / 64)
                .is_none_or(|word| word & (1u64 << (neighbor_id % 64)) == 0)) {
                visited.insert(neighbor_id);
                note_visited(1);
                note_rejected(1);
                continue;
            }
            let Some(neighbor_node) = view.current_node_typed::<CAPTURED>(neighbor_id) else {
                continue;
            };
            // Vespa checks acquired levels validity before marking. Here a
            // node's publication bound is also part of the read-view contract.
            if !view.visible(neighbor_node) || !visited.insert(neighbor_id) {
                continue;
            }
            note_visited(1);
            if accept(neighbor_node.handle(), neighbor_node.key()) {
                found.push(neighbor_id);
                if found.len() >= cap {
                    return;
                }
            } else {
                note_rejected(1);
            }
        }
    }
}

/// Queued nodes ahead whose words a wave warms (see `one_hop`).
const WAVE_AHEAD: usize = 16;
/// Queued nodes ahead whose level-0 links a wave warms (their words have
/// had `WAVE_AHEAD - WAVE_LINKS_AHEAD` nodes' time to arrive).
const WAVE_LINKS_AHEAD: usize = 8;

/// Prefetches for a live view's filter-first waves (addresses only).
struct WaveWarmer<'v> {
    hot: node_table::HotNodes<'v>,
    links: array_store::ArrayReadView<'v, u32, 20>,
}
/// A queued node as the dense words see it ([`WaveWarmer::visible_links`]).
enum DenseNode<'v> {
    /// Not visible to this view (published after it): skipped, as the
    /// record's publication check would.
    Invisible,
    /// Visible, with these level-0 links.
    Links(&'v [u32]),
    /// Removed, or links only reachable through its levels: the generic path.
    Generic,
}
impl<'v> WaveWarmer<'v> {
    #[inline(always)]
    fn visible(&self, id: u32, publication: u64) -> bool {
        self.hot.slots.get(id as usize).is_some_and(|slot| slot.vis.load(Acquire) <= publication)
    }
    #[inline(always)]
    fn visible_links(&self, id: u32, publication: u64) -> DenseNode<'v> {
        let Some(slot) = self.hot.slots.get(id as usize) else { return DenseNode::Generic };
        let word = slot.vis.load(Acquire);
        if word == u64::MAX { return DenseNode::Generic; }
        if word > publication { return DenseNode::Invisible; }
        let links0 = slot.links0.load(Acquire);
        if links0 == 0 { return DenseNode::Generic; }
        // SAFETY: a links reference current at some instant after this
        // reader's guard was taken (the writer updates the word before it
        // retires the links it replaced), so held for this guard.
        match unsafe { self.links.get(LinksRef::from_raw(links0)) } {
            Ok(links) => DenseNode::Links(links),
            Err(_) => DenseNode::Generic,
        }
    }
}
impl WaveWarmer<'_> {
    #[inline(always)]
    fn words(&self, id: u32) {
        if let Some(slot) = self.hot.slots.get(id as usize) {
            crate::vector::distance::prefetch((slot as *const node_table::HotSlot).cast(), 16);
        }
        if let Some(levels) = self.hot.levels.get(id as usize) {
            crate::vector::distance::prefetch((levels as *const std::sync::atomic::AtomicU32).cast(), 4);
        }
    }
    #[inline(always)]
    fn links(&self, id: u32) {
        let Some(slot) = self.hot.slots.get(id as usize) else { return };
        let links0 = slot.links0.load(Relaxed);
        if links0 == 0 { return; }
        // SAFETY: address only: a stale or racing reference is never
        // dereferenced here (the wave reads the links through the node).
        if let Ok(links) = unsafe { self.links.get(LinksRef::from_raw(links0)) } {
            crate::vector::distance::prefetch(links.as_ptr().cast(), links.len() * 4);
        }
    }
}
impl MutableHnswSnapshot {
    fn wave_warmer(&self) -> Option<WaveWarmer<'_>> {
        if self.captured.is_some() { return None; }
        // SAFETY: snapshot admission validated these stores/table with this
        // guard, held for the whole traversal.
        let table = unsafe { self.control.nodes.read_prevalidated(&self.guard) };
        let hot = table.hot()?;
        let links = unsafe { self.control.links.view_prevalidated(&self.guard) };
        Some(WaveWarmer { hot, links })
    }
}

fn neighborhood_typed<V: LayerVisited, const CAPTURED: bool>(
    view: &MutableHnswSnapshot,
    candidate: Candidate,
    params: VespaFilterFirstParams,
    cap: usize,
    accept: &impl Fn(NodeHandle, u64) -> bool,
    scratch: &mut FilterScratch<V>,
    node_filter: Option<&[u64]>,
) {
    scratch.todo.clear();
    scratch.found.clear();
    scratch.todo.push_back(candidate.id);
    one_hop::<_, CAPTURED>(
        view,
        &mut scratch.todo,
        &mut scratch.found,
        &mut scratch.visited,
        cap,
        accept,
        node_filter,
    );
    one_hop::<_, CAPTURED>(
        view,
        &mut scratch.todo,
        &mut scratch.found,
        &mut scratch.visited,
        cap,
        accept,
        node_filter,
    );
    // The residual FIFO includes duplicates and already visited nodes. Using
    // unique rejected nodes here would implement a different algorithm.
    if (scratch.todo.len() as f64) < params.exploration * (cap * cap) as f64 {
        one_hop::<_, CAPTURED>(
            view,
            &mut scratch.todo,
            &mut scratch.found,
            &mut scratch.visited,
            cap,
            accept,
            node_filter,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn beam<S: MutableVectorSource + ?Sized, V: LayerVisited, const CAPTURED: bool>(
    view: &MutableHnswSnapshot,
    d: &SourceDistance<'_, S>,
    entry: NodeHandle,
    distance: f64,
    ef: usize,
    params: VespaFilterFirstParams,
    accept: &impl Fn(NodeHandle, u64) -> bool,
    scratch: &mut FilterScratch<V>,
    node_filter: Option<&[u64]>,
) {
    let filter = params
        .filter_estimate
        .expect("validated filter-first parameters");
    let estimate = estimate_visited_nodes(
        (2 * view.m) as u32,
        view.len,
        u32::try_from(ef).unwrap_or(u32::MAX),
        Some(FilterEstimate {
            size: filter.size,
            count: filter.count,
        }),
    );
    scratch.visited.reset(view.len, estimate);
    scratch.candidates.clear();
    scratch.best.clear();
    scratch.out.clear();
    if params.expired() {
        scratch.visited.release();
        return;
    }
    let Some(entry_row) = d.resolve(entry) else {
        scratch.visited.release();
        return;
    };
    scratch.visited.insert(entry.slot() + 1);
    let first = Candidate::new(distance, entry_row.node);
    scratch.candidates.push(Reverse(first));
    if accept(entry, entry_row.node.key()) {
        scratch.best.push(first);
    }
    let cap = 2 * view.m;
    let mut rows = Vec::with_capacity(cap);
    let mut nodes = Vec::with_capacity(cap);
    // A code plane (live reads of a quantized field): score each found
    // neighbourhood by node id, as plain search does, with the visibility
    // check in the dense publication copy, instead of resolving each node's
    // source directory entry and row.
    let by_id = !CAPTURED && d.scores_by_id();
    let (mut ids, mut scores, mut scored) = (Vec::new(), Vec::new(), Vec::new());
    // Vespa updates this only after result-heap overflow, even when the seed
    // already fills ef. Tightening at size==ef changes equal-distance routing.
    let mut limit = f64::MAX;
    while let Some(Reverse(candidate)) = scratch.candidates.pop() {
        if candidate.distance > (1.0 + params.exploration_slack) * limit {
            break;
        }
        neighborhood_typed::<_, CAPTURED>(view, candidate, params, cap, accept, scratch, node_filter);
        nodes.clear();
        rows.clear();
        if by_id {
            // Quiet: every node the neighbourhood's links named is visible.
            let quiet = view.quiet.is_some_and(|seq| view.control.topology_seq_after_fence() == seq);
            for &id in &scratch.found {
                if let Some(node) = view.current_node_typed::<false>(id).filter(|node| quiet || view.visible_dense(*node)) {
                    d.prefetch_id(id);
                    nodes.push(node);
                }
            }
            scored.clear();
            d.distances_by_id(&nodes, &mut ids, &mut scores, &mut scored);
            for &(node, distance) in &scored {
                if !distance.is_finite() {
                    continue;
                }
                if distance < (1.0 + params.exploration_slack) * limit {
                    let candidate = Candidate::new(distance, node);
                    scratch.candidates.push(Reverse(candidate));
                    if distance < limit {
                        // Neighborhood already checked the filter.
                        scratch.best.push(candidate);
                        while scratch.best.len() > ef {
                            scratch.best.pop();
                            limit = scratch.best.peek().expect("positive ef").distance;
                        }
                    }
                }
            }
            if params.expired() {
                break;
            }
            continue;
        }
        for &id in &scratch.found {
            if let Some(node) = view.current_node_typed::<CAPTURED>(id).filter(|node| view.visible(*node)) {
                d.source.prefetch_node(node.handle(), node.key());
                nodes.push(node);
            }
        }
        for &node in &nodes {
            d.source.prefetch_row(node.handle(), node.key());
        }
        for &node in &nodes {
            if let Some(row) = d.resolve_node(node) { rows.push(row); }
        }
        if params.prefetch_tensors {
            for row in &rows {
                prefetch(row.row.as_ptr() as *const u8, row.row.len() * 4);
            }
        }
        // Resolve/filter the same neighborhood first, then evaluate four rows
        // with the existing bit-identical SIMD kernel. Candidate admission stays
        // in the original order; no speculative expansion or changed beam.
        let parallel = d.parallel_distances(&rows, true);
        for (batch_index, batch) in rows.chunks(4).enumerate() {
            let mut local = [0.0; 4];
            let distances = if let Some(distances) = &parallel {
                &distances[batch_index * 4..batch_index * 4 + batch.len()]
            } else {
                if d.encoded.is_none() && batch.len() == 4 && d.metric.uses_dot() {
                    let products = dot_four(d.query, [batch[0].row, batch[1].row, batch[2].row, batch[3].row]);
                    for i in 0..4 { local[i] = d.from_dot(&batch[i], products[i]); }
                    d.operations.set(d.operations.get() + 4);
                } else {
                    for (i, row) in batch.iter().enumerate() { local[i] = d.distance_to(row); }
                }
                &local[..batch.len()]
            };
            for (row, &distance) in batch.iter().zip(distances) {
                if !distance.is_finite() {
                    continue;
                }
                if distance < (1.0 + params.exploration_slack) * limit {
                    let candidate = Candidate::new(distance, row.node);
                    scratch.candidates.push(Reverse(candidate));
                    if distance < limit {
                        // Neighborhood already checked the filter.
                        scratch.best.push(candidate);
                        while scratch.best.len() > ef {
                            scratch.best.pop();
                            limit = scratch.best.peek().expect("positive ef").distance;
                        }
                    }
                }
            }
        }
        if params.expired() {
            break;
        }
    }
    scratch.out.extend(scratch.best.drain());
    scratch.out.sort_unstable();
    // The tracker is per-search in Vespa; a retained `SparseVisited` keeps
    // its map (bounded by the node count) and clears it on the next reset.
    scratch.visited.release();
}

impl MutableHnsw {
    /// Like search(), capture a snapshot before a growing source root instead
    /// of using this convenience wrapper with an older source snapshot.
    pub fn search_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
        mode: MutableSearchMode,
    ) -> Result<MutableSearchResult> {
        self.try_snapshot()?
            .search_mode(query, k, ef, source, accept, mode)
    }
    pub fn search_with_handles_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
        mode: MutableSearchMode,
    ) -> Result<MutableSearchResult> {
        self.try_snapshot()?
            .search_with_handles_mode(query, k, ef, source, accept, mode)
    }
}

/// A filter translated into this pinned graph's single-vector attribute IDs.
/// Keeping the snapshot borrowed prevents slot reuse while the filter is used.
/// Construct from the query's accepted handles, not a graph scan per neighbor.
/// This is an upper-bound filter: the final document predicate still applies.
pub struct PreparedNodeFilter<'a> {
    view: &'a MutableHnswSnapshot,
    words: Vec<u64>,
}
impl PreparedNodeFilter<'_> {
    pub fn search(
        &self, query: &[f32], k: usize, ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool, mode: MutableSearchMode,
    ) -> Result<MutableSearchResult> {
        self.view.search_with_handles_and_node_filter(query, k, ef, source, accept, mode, Some(&self.words))
    }
}

impl MutableHnswSnapshot {
    /// Translate accepted document handles once during GlobalFilter preparation.
    /// Stale incarnations and nodes outside this publication view are excluded.
    /// Callers must supply an upper bound of their final accepted document set.
    pub fn prepare_node_filter(
        &self, handles: impl IntoIterator<Item = NodeHandle>,
    ) -> Result<PreparedNodeFilter<'_>> {
        self.control.check_readable()?;
        let mut words = vec![0u64; (self.len as usize).div_ceil(64)];
        for handle in handles {
            if self.node(handle).is_some_and(|node| self.visible(node)) {
                let id = internal_id(handle) as usize;
                words[id / 64] |= 1u64 << (id % 64);
            }
        }
        self.control.check_readable()?;
        Ok(PreparedNodeFilter { view: self, words })
    }

    pub fn search_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(u64) -> bool,
        mode: MutableSearchMode,
    ) -> Result<MutableSearchResult> {
        self.search_with_handles_mode(query, k, ef, source, |_, key| accept(key), mode)
    }
    pub fn search_with_handles_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &(impl MutableVectorSource + ?Sized),
        accept: impl Fn(NodeHandle, u64) -> bool,
        mode: MutableSearchMode,
    ) -> Result<MutableSearchResult> {
        self.search_with_handles_and_node_filter(query, k, ef, source, accept, mode, None)
    }

    pub(crate) fn search_with_handles_and_node_filter(
        &self, query: &[f32], k: usize, ef: usize,
        source: &(impl MutableVectorSource + ?Sized), accept: impl Fn(NodeHandle, u64) -> bool,
        mode: MutableSearchMode, node_filter: Option<&[u64]>,
    ) -> Result<MutableSearchResult> {
        self.search_node_filtered(query, k, ef, source, accept, mode, node_filter, false)
    }
    /// [`Self::search_with_handles_and_node_filter`] where `accept` is
    /// exactly `node_filter`'s bitmap of node ids (not a stricter document
    /// predicate): Plain admission then tests the bit instead of resolving
    /// each admitted node's identity for `accept`.
    pub(crate) fn search_with_exact_node_filter(
        &self, query: &[f32], k: usize, ef: usize,
        source: &(impl MutableVectorSource + ?Sized), accept: impl Fn(NodeHandle, u64) -> bool,
        mode: MutableSearchMode, node_filter: &[u64],
    ) -> Result<MutableSearchResult> {
        self.search_node_filtered(query, k, ef, source, accept, mode, Some(node_filter), true)
    }
    #[allow(clippy::too_many_arguments)]
    fn search_node_filtered(
        &self, query: &[f32], k: usize, ef: usize,
        source: &(impl MutableVectorSource + ?Sized), accept: impl Fn(NodeHandle, u64) -> bool,
        mode: MutableSearchMode, node_filter: Option<&[u64]>, accept_is_filter: bool,
    ) -> Result<MutableSearchResult> {
        self.control.check_readable()?;
        if let MutableSearchMode::VespaPlain(params) = mode {
            params.validate()?;
            let filter = params.filter_estimate.map(|f| FilterEstimate { size: f.size, count: f.count });
            return self.search_with_execution(query, k, ef, source, accept, filter, params,
                node_filter.filter(|_| accept_is_filter));
        }
        let MutableSearchMode::VespaFilterFirst(params) = mode else {
            return self.search_with_execution(query, k, ef, source, accept, None,
                VespaPlainParams { prefetch_tensors: true, ..Default::default() }, node_filter.filter(|_| accept_is_filter));
        };
        params.validate()?;
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
        let mut acquired = Candidate::new(
            d.dist(current),
            self.acquired_node(current).expect("acquired entry"),
        );
        for level in (1..=self.entry_level(current)).rev() {
            acquired = self.greedy_acquired(&d, acquired, level);
        }
        let current = self.candidate_node(acquired).handle();
        let distance = acquired.distance;
        fn run<S: MutableVectorSource + ?Sized, V: LayerVisited, const CAPTURED: bool>(
            view: &MutableHnswSnapshot, d: &SourceDistance<'_, S>, current: NodeHandle,
            distance: f64, k: usize, ef: usize, params: VespaFilterFirstParams,
            accept: &impl Fn(NodeHandle, u64) -> bool, _query: &[f32], scratch: &mut FilterScratch<V>, node_filter: Option<&[u64]>,
        ) -> Vec<(u64, f32)> {
            beam::<_, _, CAPTURED>(
                view,
                d,
                current,
                distance,
                ef.max(k),
                params,
                accept,
                scratch,
                node_filter,
            );
            // Use the same bound scoring and eligibility validation as Plain.
            let mut hits: Vec<_> = scratch
                .out
                .iter()
                .take(k.saturating_add(k / 2).saturating_add(4))
                .filter_map(|candidate| {
                    let row = d.resolve_id(candidate.id)?;
                    if !accept(row.id, row.node.key()) {
                        return None;
                    }
                    d.operations.set(d.operations.get() + 1);
                    d.final_score_ops.set(d.final_score_ops.get() + 1);
                    Some((row.node.key(), d.result_score(&row)?))
                })
                .collect();
            hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            hits.truncate(k);
            hits
        }
        let hits = FILTER_SCRATCH.with(|s| if self.captured.is_some() { run::<_, _, true>(self, &d, current, distance, k, ef, params, &accept, query, &mut s.borrow_mut(), node_filter) } else { run::<_, _, false>(self, &d, current, distance, k, ef, params, &accept, query, &mut s.borrow_mut(), node_filter) });
        let vector_ops = d.operations.get();
        add_vector_ops(vector_ops);
        self.control.check_readable()?;
        Ok(MutableSearchResult {
            hits,
            vector_ops,
            final_score_ops: d.final_score_ops.get(),
            entry_fallback_scanned: d.entry_scanned.get(),
            missing_source_lookups: d.missing_source.get(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn prepared_attribute_filter_preserves_rejected_bridges_and_captured_results() {
        let (graph, rows, ids) = fixture(&[&[1], &[2], &[]], &[10., 20., 0.], 2);
        let view = graph.snapshot();
        let prepared = view.prepare_node_filter([ids[2]]).unwrap();
        let mode = MutableSearchMode::VespaFilterFirst(params(3, 1));
        let expected = view.search_with_handles_mode(&[0.], 1, 20, &rows,
            |_, key| key == 2, mode).unwrap();
        let actual = prepared.search(&[0.], 1, 20, &rows, |_, key| key == 2, mode).unwrap();
        assert_eq!(expected.hits, actual.hits);
        assert_eq!(actual.hits.len(), 1);
        assert_eq!(actual.vector_ops, expected.vector_ops);
        let captured = graph.capture_snapshot().unwrap();
        let retained_filter = captured.prepare_node_filter([ids[2]]).unwrap();
        assert_eq!(retained_filter.search(&[0.], 1, 20, &rows,
            |_, key| key == 2, mode).unwrap().hits, expected.hits);
        let empty = view.prepare_node_filter([]).unwrap();
        assert!(empty.search(&[0.], 1, 20, &rows, |_, key| key == 2, mode).unwrap().hits.is_empty());
    }

    #[test]
    fn fallible_search_returns_error_when_source_poisoning_happens_after_admission() {
        struct PoisonOnRead<'a>(&'a MutableHnsw);
        impl MutableVectorSource for PoisonOnRead<'_> {
            fn get(&self, _: u64) -> Option<&[f32]> {
                self.0.control.poison();
                Some(&[1.])
            }
        }
        for route in 0..4 {
            let rows = Rows(vec![vec![1.]]);
            let graph = MutableHnsw::new(1, Similarity::L2, HnswParams::default());
            graph.insert(0, &rows).unwrap();
            let view = graph.try_snapshot().unwrap();
            let source = PoisonOnRead(&graph);
            let result = if route == 3 {
                view.search_with_filter_estimate(&[1.], 1, 10, &source, |_| true,
                    VespaFilterEstimate { size: 1, count: 1 })
            } else {
                let mode = match route {
                    0 => MutableSearchMode::Plain,
                    1 => MutableSearchMode::VespaPlain(VespaPlainParams::default()),
                    _ => MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                        filter_estimate: Some(VespaFilterEstimate { size: 1, count: 1 }),
                        ..Default::default()
                    }),
                };
                view.search_mode(&[1.], 1, 10, &source, |_| true, mode)
            };
            assert!(result.is_err(), "route {route} must return an error, not panic or return hits");
        }
    }

    struct Rows(Vec<Vec<f32>>);
    impl MutableVectorSource for Rows {
        fn get(&self, key: u64) -> Option<&[f32]> {
            self.0
                .get(key as usize)
                .filter(|row| !row.is_empty())
                .map(Vec::as_slice)
        }
    }
    fn fixture(links: &[&[u32]], values: &[f32], m: usize) -> (MutableHnsw, Rows, Vec<NodeHandle>) {
        fixture_metric(links, values, m, Similarity::L2)
    }
    fn fixture_metric(
        links: &[&[u32]],
        values: &[f32],
        m: usize,
        similarity: Similarity,
    ) -> (MutableHnsw, Rows, Vec<NodeHandle>) {
        let rows = Rows(values.iter().map(|&v| vec![v]).collect());
        let graph = MutableHnsw::new(
            1,
            similarity,
            HnswParams {
                m,
                ..Default::default()
            },
        );
        let handles: Vec<_> = (0..values.len())
            .map(|key| graph.insert(key as u64, &rows).unwrap())
            .collect();
        let view = graph.snapshot();
        for (i, neighbors) in links.iter().enumerate() {
            let node = view.node(handles[i]).unwrap();
            for level in 0..=node.level() {
                graph.test_replace_links(handles[i], level, &[]).unwrap();
            }
            let neighbors: Vec<_> = neighbors
                .iter()
                .map(|&key| handles.get(key as usize).map_or(key, |h| h.slot() + 1))
                .collect();
            graph.test_replace_links(handles[i], 0, &neighbors).unwrap();
        }
        graph.test_set_entry(handles[0]).unwrap();
        drop(view);
        (graph, rows, handles)
    }
    fn node_id(handle: NodeHandle) -> u32 {
        handle.slot() + 1
    }
    fn candidate(view: &MutableHnswSnapshot, id: NodeHandle) -> Candidate {
        Candidate::new(0.0, view.node(id).unwrap())
    }
    fn params(size: u32, count: u32) -> VespaFilterFirstParams {
        VespaFilterFirstParams {
            filter_estimate: Some(VespaFilterEstimate { size, count }),
            ..Default::default()
        }
    }

    #[test]
    fn waves_expand_accepted_and_already_visited_bridges() {
        let (graph, _, ids) = fixture(&[&[1, 2], &[2, 3], &[3, 4], &[], &[]], &[0.; 5], 3);
        let view = graph.snapshot();
        let mut scratch: FilterScratch = FilterScratch::default();
        scratch.visited.reset(view.len, view.len);
        scratch.visited.insert(node_id(ids[0]));
        scratch.visited.insert(node_id(ids[2]));
        neighborhood_typed::<_, false>(
            &view,
            candidate(&view, ids[0]),
            VespaFilterFirstParams {
                exploration: 0.0,
                ..Default::default()
            },
            6,
            &|_, key| key != 0,
            &mut scratch,
            None,
        );
        assert_eq!(
            scratch.found,
            vec![node_id(ids[1]), node_id(ids[3]), node_id(ids[4])]
        );
        assert_eq!(
            scratch.todo,
            VecDeque::from([
                node_id(ids[2]),
                node_id(ids[3]),
                node_id(ids[3]),
                node_id(ids[4])
            ])
        );
    }

    #[test]
    fn third_wave_uses_strict_residual_queue_threshold() {
        let (graph, _, ids) = fixture(&[&[1], &[2], &[3], &[]], &[0.; 4], 1);
        let view = graph.snapshot();
        for (exploration, expected) in [(0.25, vec![]), (0.250001, vec![node_id(ids[3])])] {
            let mut scratch: FilterScratch = FilterScratch::default();
            scratch.visited.reset(view.len, view.len);
            scratch.visited.insert(node_id(ids[0]));
            neighborhood_typed::<_, false>(
                &view,
                candidate(&view, ids[0]),
                VespaFilterFirstParams {
                    exploration,
                    ..Default::default()
                },
                2,
                &|_, key| key == 3,
                &mut scratch,
                None,
            );
            assert_eq!(scratch.found, expected);
        }
    }

    #[test]
    fn neighborhood_caps_accepted_nodes_and_skips_out_of_bounds_links() {
        let (graph, _, ids) = fixture(&[&[999, 1, 2, 3], &[], &[], &[]], &[0.; 4], 1);
        let view = graph.snapshot();
        let mut scratch: FilterScratch = FilterScratch::default();
        scratch.visited.reset(view.len, view.len);
        scratch.visited.insert(node_id(ids[0]));
        neighborhood_typed::<_, false>(
            &view,
            candidate(&view, ids[0]),
            VespaFilterFirstParams::default(),
            2,
            &|_, _| true,
            &mut scratch,
            None,
        );
        assert_eq!(scratch.found, vec![node_id(ids[1]), node_id(ids[2])]);
        assert!(!scratch.todo.contains(&999));
        assert_eq!(scratch.visited.state(node_id(ids[3])), 0);
    }

    #[test]
    fn rejected_entry_is_a_bridge_without_scoring_rejected_neighbors() {
        let (graph, rows, _) = fixture(&[&[1], &[2], &[]], &[10., 20., 0.], 2);
        let result = graph
            .search_mode(
                &[0.],
                1,
                10,
                &rows,
                |key| key == 2,
                MutableSearchMode::VespaFilterFirst(params(3, 1)),
            )
            .unwrap();
        assert_eq!(result.hits, vec![(2, Similarity::L2.score(&[0.], &[0.]))]);
        assert_eq!(result.vector_ops, 3); // Entry, accepted node, final scoring.
        assert_eq!(result.missing_source_lookups, 0);
    }

    #[test]
    fn result_distance_limit_is_not_tightened_until_overflow() {
        let (graph, rows, _) = fixture(&[&[1], &[2], &[3], &[]], &[10., -10., 100., 0.], 2);
        let result = graph
            .search_mode(
                &[0.],
                1,
                1,
                &rows,
                |key| key != 2,
                MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                    exploration: 0.0,
                    ..params(4, 3)
                }),
            )
            .unwrap();
        // Entry and first neighbor have equal distance. The first neighbor
        // must reach the frontier before overflow establishes the limit.
        assert_eq!(result.hits[0].0, 3);
    }

    #[test]
    fn slack_uses_literal_multiplication_even_for_negative_dot_distances() {
        // Pinned Vespa mips_distance_transform.cpp::for_query_vector uses
        // extra_dim=false and calc returns -dot. Its slack expression is still
        // (1+slack)*limit, not limit+slack*abs(limit). Keep that behavior visible.
        let (graph, rows, _) = fixture_metric(
            &[&[1], &[2], &[3], &[]],
            &[1., 2., 0., 4.],
            2,
            Similarity::DotProduct,
        );
        for (exploration_slack, expected) in [(0.0, 3), (0.5, 1)] {
            let result = graph
                .search_mode(
                    &[1.],
                    1,
                    1,
                    &rows,
                    |key| key != 2,
                    MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                        exploration: 0.0,
                        exploration_slack,
                        ..params(4, 3)
                    }),
                )
                .unwrap();
            assert_eq!(result.hits[0].0, expected);
        }
    }

    #[test]
    fn newer_publications_are_not_used_as_bridges_by_an_older_view() {
        let rows = Rows(vec![vec![1.], vec![0.], vec![3.], vec![4.], vec![2.]]);
        let graph = MutableHnsw::new(1, Similarity::L2, HnswParams::default());
        let first = graph.insert(0, &rows).unwrap();
        let accepted = graph.insert(1, &rows).unwrap();
        let hole = graph.insert(2, &rows).unwrap();
        let high = graph.insert(3, &rows).unwrap();
        assert!(graph.remove(hole, &rows).unwrap());
        // Keep a higher published ID live, then reuse a lower hole. The old
        // view must reject the bridge's publication, not merely its ID bound.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut reclaimed = 0;
        while reclaimed == 0 && Instant::now() < deadline {
            let tokens = graph.collect_reclaimed();
            reclaimed += tokens.len();
            drop(tokens); // This fixture's immutable rows remain owned.
            std::thread::yield_now();
        }
        assert_eq!(reclaimed, 1);
        let reservation = graph.reserve_insert(4).unwrap();
        assert_eq!(reservation.handle().slot(), hole.slot());
        let old_view = graph.snapshot();
        assert!(node_id(high) < old_view.len);
        let bridge = graph
            .complete(graph.prepare(reservation, &rows).unwrap(), &rows)
            .unwrap();
        assert!(node_id(bridge) < old_view.len);
        assert!(!old_view.visible(old_view.node(bridge).unwrap()));
        graph
            .test_replace_links(first, 0, &[node_id(bridge)])
            .unwrap();
        graph
            .test_replace_links(bridge, 0, &[node_id(accepted)])
            .unwrap();
        let mut scratch: FilterScratch = FilterScratch::default();
        scratch.visited.reset(old_view.len, old_view.len);
        scratch.visited.insert(node_id(first));
        neighborhood_typed::<_, false>(
            &old_view,
            candidate(&old_view, first),
            VespaFilterFirstParams::default(),
            32,
            &|_, key| key == 1,
            &mut scratch,
            None,
        );
        assert!(scratch.found.is_empty());
        let new_view = graph.snapshot();
        scratch.visited.reset(new_view.len, new_view.len);
        scratch.visited.insert(node_id(first));
        neighborhood_typed::<_, false>(
            &new_view,
            candidate(&new_view, first),
            VespaFilterFirstParams::default(),
            32,
            &|_, key| key == 1,
            &mut scratch,
            None,
        );
        assert_eq!(scratch.found, vec![node_id(accepted)]);
    }

    #[test]
    fn first_hop_reacquires_current_levels_after_candidate_removal() {
        let (graph, rows, ids) = fixture(&[&[1], &[]], &[1., 0.], 2);
        let view = graph.snapshot();
        let retained = candidate(&view, ids[0]);
        assert!(graph.remove(ids[0], &rows).unwrap());
        // The acquired array is still alive under this view; the current
        // node ref filter-first reacquires is now invalid.
        assert_eq!(
            view.candidate_node(retained).links(0).unwrap(),
            &[node_id(ids[1])]
        );
        assert!(view.current_node(node_id(ids[0])).is_none());
        let run = |view: &MutableHnswSnapshot| {
            let mut scratch: FilterScratch = FilterScratch::default();
            scratch.visited.reset(view.len, view.len);
            neighborhood_typed::<_, false>(
                view,
                retained,
                VespaFilterFirstParams::default(),
                4,
                &|_, _| true,
                &mut scratch,
                None,
            );
            scratch.found
        };
        // A reader admitted before the removal may hold links naming the
        // removed node, so it still bridges through the node's retained
        // links (never accepting the node itself): its neighbour is found.
        assert_eq!(run(&view), vec![node_id(ids[1])]);
        // A reader admitted after the removal never sees those links: it
        // reacquires current levels only, so the removed node expands to
        // nothing, as in pinned filter-first.
        let after = graph.snapshot();
        assert!(after.current_node(node_id(ids[0])).is_none());
        assert!(run(&after).is_empty());
        drop(view);
    }

    #[test]
    fn missing_accepted_rows_remain_an_explicit_failure_diagnostic() {
        let (graph, mut rows, _) = fixture(&[&[1, 2], &[], &[]], &[3., 2., 1.], 2);
        rows.0[2].clear();
        let result = graph
            .search_mode(
                &[0.],
                2,
                10,
                &rows,
                |key| key != 0,
                MutableSearchMode::VespaFilterFirst(params(3, 2)),
            )
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].0, 1);
        assert!(result.missing_source_lookups > 0);
    }

    #[test]
    fn configured_plain_defaults_prefetch_deadline_and_validation() {
        let (graph, rows, _) = fixture(&[&[1, 2], &[0, 2], &[0, 1]], &[3., 2., 1.], 2);
        let baseline = graph.search(&[0.], 2, 10, &rows, |_| true);
        for prefetch_tensors in [false, true] {
            let result = graph.search_mode(&[0.], 2, 10, &rows, |_| true,
                MutableSearchMode::VespaPlain(VespaPlainParams { prefetch_tensors, ..Default::default() })).unwrap();
            assert_eq!(result.hits, baseline.hits);
            assert_eq!(result.vector_ops, baseline.vector_ops);
        }
        let expired = graph.search_mode(&[0.], 2, 10, &rows, |_| true,
            MutableSearchMode::VespaPlain(VespaPlainParams { deadline: Some(Instant::now()), ..Default::default() })).unwrap();
        assert!(expired.hits.is_empty());
        for exploration_slack in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(graph.search_mode(&[0.], 2, 10, &rows, |_| true,
                MutableSearchMode::VespaPlain(VespaPlainParams { exploration_slack, ..Default::default() })).is_err());
        }
        assert!(graph.search_mode(&[0.], 2, 10, &rows, |_| true,
            MutableSearchMode::VespaPlain(VespaPlainParams {
                filter_estimate: Some(VespaFilterEstimate { size: 2, count: 3 }), ..Default::default()
            })).is_err());
    }

    #[test]
    fn plain_slack_explores_a_distant_bridge_without_relaxing_result_limit() {
        let (graph, rows, _) = fixture(&[&[1], &[2], &[]], &[1., 2., 0.], 2);
        let baseline = graph.search(&[0.], 1, 1, &rows, |_| true);
        let expanded = graph.search_mode(&[0.], 1, 1, &rows, |_| true,
            MutableSearchMode::VespaPlain(VespaPlainParams { exploration_slack: 3.0, ..Default::default() })).unwrap();
        assert_eq!(baseline.hits[0].0, 0);
        assert_eq!(expanded.hits.len(), 1);
        assert_eq!(expanded.hits[0].0, 2);
    }

    #[test]
    fn explicit_plain_preserves_default_and_invalid_options_fail() {
        let (graph, rows, _) = fixture(&[&[1, 2], &[0, 2], &[0, 1]], &[3., 2., 1.], 2);
        let default = graph.search(&[0.], 2, 10, &rows, |_| true);
        let plain = graph
            .search_mode(&[0.], 2, 10, &rows, |_| true, MutableSearchMode::Plain)
            .unwrap();
        assert_eq!(default.hits, plain.hits);
        assert_eq!(default.vector_ops, plain.vector_ops);
        for (exploration, exploration_slack) in
            [(f64::NAN, 0.), (-1., 0.), (0., f64::INFINITY), (0., -1.)]
        {
            assert!(
                graph
                    .search_mode(
                        &[0.],
                        2,
                        10,
                        &rows,
                        |_| true,
                        MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                            exploration,
                            exploration_slack,
                            ..params(3, 3)
                        })
                    )
                    .is_err()
            );
        }
        for invalid in [VespaFilterFirstParams::default(), params(3, 4)] {
            assert!(
                graph
                    .search_mode(
                        &[0.],
                        2,
                        10,
                        &rows,
                        |_| true,
                        MutableSearchMode::VespaFilterFirst(invalid)
                    )
                    .is_err()
            );
        }
        let expired = graph
            .search_mode(
                &[0.],
                2,
                10,
                &rows,
                |_| true,
                MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                    deadline: Some(Instant::now() - Duration::from_secs(1)),
                    ..params(3, 3)
                }),
            )
            .unwrap();
        assert!(expired.hits.is_empty());
    }
}
