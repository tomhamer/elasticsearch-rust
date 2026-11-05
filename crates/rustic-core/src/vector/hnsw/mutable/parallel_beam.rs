//! Intra-query parallel level-0 search of the one shared graph.
//!
//! One query's beam split over several threads of the search pool, for a
//! query that has the cores to itself (`crate::search_load`): the upper
//! layers are descended once, a short beam from the layer-0 entry yields the
//! closest distinct nodes as seeds, and each worker runs the dense beam
//! (`dense_beam_in`, unchanged) from its own seed with a narrower width,
//! over one visited bitmap shared by all of them, so no node is scored
//! twice and the workers spread over the neighbourhood instead of
//! retracing each other. The workers' results merge into the query's top.
//!
//! The graph and its construction are untouched; this is a search-time
//! schedule over the same traversal (Vespa keeps one thread per query; the
//! partitioned beam follows the intra-query parallel graph search of iQAN,
//! PPoPP 2023, in its simplest form: independent beams, shared visited).
use super::*;
use super::dense_beam::{DenseScratch, Scoring};
use std::sync::atomic::AtomicU64;

/// A visited bitmap shared by the workers of one query (`fetch_or` marks).
/// `Default` is an empty map (the trait's requirement); the query's map is
/// built by [`SharedVisited::over`].
#[derive(Default)]
pub(super) struct SharedVisited { words: *const AtomicU64, len: usize }
// SAFETY: a view of `AtomicU64`s owned by the query for the workers' scope.
unsafe impl Send for SharedVisited {}
impl SharedVisited {
    fn over(words: &[AtomicU64]) -> Self { Self { words: words.as_ptr(), len: words.len() } }
    #[inline(always)]
    fn word(&self, id: u32) -> Option<&AtomicU64> {
        let w = id as usize / 64;
        // SAFETY: `w < len` of a slice the query keeps alive for the scope.
        (w < self.len).then(|| unsafe { &*self.words.add(w) })
    }
}
impl visited::LayerVisited for SharedVisited {
    fn reset(&mut self, _limit: u32, _estimate: u32) {}
    fn state(&self, id: u32) -> u8 {
        self.word(id).map_or(0, |w| u8::from(w.load(std::sync::atomic::Ordering::Relaxed) & (1u64 << (id % 64)) != 0))
    }
    #[inline(always)]
    fn insert(&mut self, id: u32) -> bool {
        let Some(w) = self.word(id) else { return false };
        let bit = 1u64 << (id % 64);
        // Load and store, no read-modify-write: two workers racing on a word
        // can lose a mark, which only lets a node be scored twice (the
        // merge drops the duplicate), never skips one.
        let word = w.load(std::sync::atomic::Ordering::Relaxed);
        if word & bit != 0 { return false; }
        w.store(word | bit, std::sync::atomic::Ordering::Relaxed);
        true
    }
    fn release(&mut self) {}
}

/// Most workers one query's beam uses (`RUSTIC_KNN_BEAM_WORKERS`, default
/// 4; 0 or 1 disables).
fn max_beam_workers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RUSTIC_KNN_BEAM_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4))
}
/// Each worker's beam width as a fraction of `ef / workers`
/// (`RUSTIC_KNN_BEAM_WIDTH`, default 1.5): the union of the workers' tops
/// holds `1.5 ef` candidates, so the split beams keep the serial one's recall.
fn beam_width_factor() -> f64 {
    static F: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("RUSTIC_KNN_BEAM_WIDTH").ok().and_then(|v| v.parse().ok())
        .filter(|v: &f64| *v > 0.0).unwrap_or(1.5))
}
/// Smallest beam worth splitting. Below it the split's fixed cost (the
/// seeds' beam, the shared map, the pool hop) is most of what it saves and
/// the narrow per-worker beams lose recall: dense_vector (2M x 96-d int8,
/// 4 workers) at num_candidates 1000 went 2.83 -> 2.21 ms p50 at recall@100
/// 0.979 -> 0.978, at 100 1.23 -> 1.18 ms but recall@10 0.957 -> 0.944;
/// so_vector (768-d bbq) at 300 was 0.2 ms slower.
const MIN_PARALLEL_EF: usize = 500;

/// Workers this query's beam may use now: its share of the cores among the
/// shard searches running (`crate::search_load`, this one's included), none
/// while searches queue, at most [`max_beam_workers`].
pub(crate) fn beam_workers(ef: usize) -> usize {
    let max = max_beam_workers();
    if max < 2 || ef < MIN_PARALLEL_EF { return 1; }
    let cores = rayon::current_num_threads().max(1);
    let (queued, active) = crate::search_load();
    if queued > 0 { return 1; }
    (cores / active.max(1)).clamp(1, max)
}

impl MutableHnswSnapshot {
    /// The plain level-0 search of a dense live view, split over `workers`
    /// threads of the caller's pool (see the module docs); `node_filter` is
    /// the query's acceptance (bitmap of node ids). `None`: this view or
    /// source does not qualify (the serial search runs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_parallel_node_filter<S: MutableVectorSource + Sync + ?Sized>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &S,
        params: VespaPlainParams,
        node_filter: &[u64],
        workers: usize,
    ) -> Result<Option<MutableSearchResult>> {
        let accept_bit = |id: u32| node_filter.get(id as usize / 64).is_some_and(|w| w & (1u64 << (id % 64)) != 0);
        self.search_parallel(query, k, ef, source, params, &|handle: NodeHandle, _key: u64| accept_bit(handle.slot() + 1), Some(node_filter), workers)
    }

    /// [`search_parallel_node_filter`](Self::search_parallel_node_filter)
    /// with any acceptance predicate the workers may share (`node_filter`:
    /// when `Some`, `accept` is exactly that bitmap of node ids). The
    /// during-indexing search uses this with its view's visible points.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_parallel<S: MutableVectorSource + Sync + ?Sized>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source: &S,
        params: VespaPlainParams,
        accept: &(impl Fn(NodeHandle, u64) -> bool + Sync),
        node_filter: Option<&[u64]>,
        workers: usize,
    ) -> Result<Option<MutableSearchResult>> {
        if workers < 2 || self.captured.is_some() || params.exploration_slack != 0.0 { return Ok(None); }
        self.control.check_readable()?;
        if k == 0 || query.len() != self.dim as usize || !self.similarity.valid_row(query) {
            return Ok(None);
        }
        let ef = ef.max(k);
        let d = SourceDistance::for_query(self, source, query);
        let Some(scoring) = self.dense_scoring(&d) else { return Ok(None) };
        let Some(current) = self.available_entry(&d) else { return Ok(None) };
        let distance = d.dist(current);
        let top = self.entry_level(current);
        let Some(node) = self.acquired_node(current) else { return Ok(None) };
        let mut acquired = Candidate::new(distance, node);
        for level in (1..=top).rev() {
            acquired = self.greedy_acquired_bridged(&d, acquired, level);
        }
        // Seeds: the closest nodes a short serial beam from the entry finds
        // (accepted or not: a seed is a place to start, not a hit).
        let seed_ef = (4 * workers).max(16);
        let seeds: Vec<Candidate> = with_mutable_scratch(|s| {
            let mut x = DenseScratch::default();
            visited::with_layer_visited(self.len, |visited| {
                self.dense_beam_in(&d, acquired, seed_ef, params, &|_, _| true, None, scoring, &mut x, visited, s);
            });
            s.out.iter().filter_map(|c| self.current_node(c.id).map(|node| Candidate::new(c.distance, node))).take(workers).collect()
        });
        if seeds.len() < 2 { return Ok(None); }
        let workers = seeds.len();
        let width = ((ef as f64 * beam_width_factor() / workers as f64).ceil() as usize).max(k).max(16);
        let words: Vec<AtomicU64> = (0..(self.len as usize).div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        // One worker's beam: its own scorer and scratch, the shared map.
        // The query as the main scorer quantized it (seeds' beam included),
        // so the workers do not quantize it again.
        let shared: source::SharedQuantized = parking_lot::Mutex::new(d.encoded.as_ref().and_then(|e| e.quantized_state()).unwrap_or_default());
        let run = |seed: Candidate| -> (Vec<(f64, u32, Option<f32>)>, u64, u64) {
            let mut dw = SourceDistance::new(self, source, query);
            dw.encoded = source.bind_query_shared(query, &shared);
            let Some(scoring) = self.dense_scoring(&dw) else { return (Vec::new(), 0, 0) };
            let mut shared = SharedVisited::over(&words);
            with_mutable_scratch(|s| {
                let mut x = DenseScratch::default();
                self.dense_beam_in(&dw, seed, width, params, accept, node_filter, scoring, &mut x, &mut shared, s);
                let scores = (s.out_scores.len() == s.out.len()).then_some(&s.out_scores);
                let out = s.out.iter().enumerate().map(|(i, c)| (c.distance, c.id, scores.map(|v| v[i]))).collect();
                (out, dw.operations.get(), dw.missing_source.get())
            })
        };
        let parts: Vec<_> = seeds.par_iter().map(|&seed| run(seed)).collect();
        let mut merged: Vec<(f64, u32, Option<f32>)> = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
        let (mut operations, mut missing) = (d.operations.get(), d.missing_source.get());
        for (out, ops, miss) in parts { merged.extend(out); operations += ops; missing += miss; }
        merged.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        merged.dedup_by_key(|c| c.1);
        d.operations.set(operations);
        d.missing_source.set(missing);
        // As the serial search: the best `k + k/2 + 4`, each resolved and
        // checked by the query's acceptance, scored, sorted, truncated.
        let mut hits: Vec<(u64, f32)> = merged.iter().take(k.saturating_add(k / 2).saturating_add(4))
            .filter_map(|&(_, id, score)| {
                let row = d.resolve_id(id)?;
                if !accept(row.id, row.node.key()) { return None; }
                d.operations.set(d.operations.get() + 1);
                d.final_score_ops.set(d.final_score_ops.get() + 1);
                let score = match score { Some(score) => score, None => d.result_score(&row)? };
                Some((row.node.key(), score))
            }).collect();
        hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.truncate(k);
        self.control.check_readable()?;
        let vector_ops = d.operations.get();
        add_vector_ops(vector_ops);
        Ok(Some(MutableSearchResult {
            hits,
            vector_ops,
            final_score_ops: d.final_score_ops.get(),
            entry_fallback_scanned: d.entry_scanned.get(),
            missing_source_lookups: d.missing_source.get(),
        }))
    }
}
