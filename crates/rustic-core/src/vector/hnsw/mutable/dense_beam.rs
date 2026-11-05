//! The live level-0 Plain beam over a source that keeps its rows or codes
//! dense by node id ([`DenseLayout`]): the same traversal as
//! `beam_acquired_typed` (pinned Vespa `search_layer_helper` with no
//! exploration slack), laid out for memory-level parallelism.
//!
//! Per expansion it reads the candidate's current levels and level-0 links;
//! per neighbour, one visibility word ([`HotNodes`]) and its row or codes at
//! a computed address, all prefetched for the whole neighbourhood before any
//! is read; a single acquire fence orders the neighbourhood's words before
//! its rows. Candidates are packed `u64` keys (ordered distance, id) in the
//! frontier and result heaps, so heap work compares integers and a node's
//! record (key, incarnation) is read only for nodes a caller predicate must
//! see. The result ordering is the old one's (distance only; equal
//! distances now order by id), with the same admission and stop rules.
//!
//! Packed rows (ES `byte` / `bit` vectors, four bytes per lane) are scored
//! one row at a time by the similarity's own packed kernel, the distance
//! the generic beam computes for them; only f32 rows use the four-row dot
//! batch.
//!
//! A removed neighbour (its word reset) is traversed through when this
//! reader may still hold links naming it (`removed_bridge`), never
//! admitted, as before. Captured views, exploration slack and sources
//! without a dense layout keep the generic beam.
use super::*;
use node_table::HotNodes;
use std::sync::atomic::{fence, AtomicU64, Ordering::Relaxed as Rx};

/// `RUSTIC_KNN_DENSE=0`: keep the generic beam (A/B diagnostics).
fn dense_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(test)]
    if GENERIC.with(|g| g.get()) { return false; }
    *ON.get_or_init(|| std::env::var("RUSTIC_KNN_DENSE").map_or(true, |v| v != "0"))
}
#[cfg(test)]
thread_local! { static GENERIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// Total order of finite f32 as u32 (smaller first).
#[inline(always)]
fn ordered(d: f32) -> u32 {
    let b = d.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}
#[inline(always)]
fn unordered(o: u32) -> f32 {
    f32::from_bits(if o & 0x8000_0000 != 0 { o & 0x7fff_ffff } else { !o })
}
#[inline(always)]
fn pack(d: f32, id: u32) -> u64 {
    (u64::from(ordered(d)) << 32) | u64::from(id)
}
#[inline(always)]
fn dist_of(key: u64) -> u32 {
    (key >> 32) as u32
}

#[derive(Default)]
pub(super) struct DenseScratch {
    cand: BinaryHeap<Reverse<u64>>,
    top: BinaryHeap<u64>,
    seen: Vec<(u32, usize)>,
    ok: Vec<u32>,
    /// Row addresses (`Rows`) of `ok`, 0 when never written.
    rows: Vec<usize>,
    scores: Vec<f32>,
}
thread_local! { static DENSE_SCRATCH: RefCell<DenseScratch> = RefCell::new(DenseScratch::default()); }

/// How a neighbourhood is scored: f32 cells by id, or the bound scorer's
/// codes by id.
#[derive(Clone, Copy)]
pub(super) enum Scoring<'a> {
    Rows(source::RowPlaneRef<'a>, Metric),
    Codes(source::CodePlaneRef<'a>),
}

impl MutableHnswSnapshot {
    /// Run the level-0 beam from `entry` into `s.out` (ascending distance)
    /// over a dense source, as `beam_acquired` would; `false` (nothing
    /// done) when this view or source does not qualify. `node_filter`: the
    /// caller's `accept` is exactly this bitmap of node ids (admission
    /// tests the bit; final hits still call `accept`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_beam<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        entry: Candidate,
        ef: usize,
        params: VespaPlainParams,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        node_filter: Option<&[u64]>,
        s: &mut MutableScratch,
    ) -> bool {
        if params.exploration_slack != 0.0 || ef == 0 {
            return false;
        }
        let Some(scoring) = self.dense_scoring(d) else { return false };
        DENSE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let x = &mut *scratch;
            x.cand.clear();
            x.top.clear();
            visited::with_layer_visited(self.len, |visited| {
                self.dense_beam_in(d, entry, ef, params, accept, node_filter, scoring, x, visited, s)
            });
            x.cand.clear();
            x.top.clear();
        });
        true
    }

    /// How this live view's dense beam scores `d`'s source (`None`: the
    /// dense beam does not apply: disabled, captured, no dense layout or
    /// hot node words).
    pub(super) fn dense_scoring<'s, S: MutableVectorSource + ?Sized>(&self, d: &'s SourceDistance<'_, S>) -> Option<Scoring<'s>> {
        if !dense_enabled() || self.captured.is_some() { return None; }
        let scoring = match d.source.dense_layout() {
            Some(DenseLayout::Rows(plane)) if d.encoded.is_none() => match self.similarity.metric() {
                metric @ (Metric::Dot | Metric::L2 | Metric::Packed) => Scoring::Rows(plane, metric),
                Metric::Cos if !cfg!(feature = "mutable-hnsw-vespa-angular") => Scoring::Rows(plane, Metric::Cos),
                _ => return None,
            },
            Some(DenseLayout::Codes(plane)) if d.scores_by_id() => Scoring::Codes(plane),
            _ => return None,
        };
        // SAFETY: snapshot admission validated this immutable table/guard pair.
        unsafe { self.control.nodes.read_prevalidated(&self.guard) }.hot()?;
        Some(scoring)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_beam_in<S: MutableVectorSource + ?Sized, V: visited::LayerVisited>(
        &self,
        d: &SourceDistance<'_, S>,
        entry: Candidate,
        ef: usize,
        params: VespaPlainParams,
        accept: &impl Fn(NodeHandle, u64) -> bool,
        node_filter: Option<&[u64]>,
        scoring: Scoring<'_>,
        x: &mut DenseScratch,
        visited: &mut V,
        s: &mut MutableScratch,
    ) {
        s.out.clear();
        if params.expired() { return; }
        let dim = self.dim as usize;
        let publication = self.publication;
        let accepts = |id: u32| -> bool {
            let ok = match node_filter {
                Some(words) => words.get(id as usize / 64).is_some_and(|w| w & (1u64 << (id % 64)) != 0),
                None => self.current_node(id).is_some_and(|node| accept(node.handle(), node.key())),
            };
            if !ok { note_rejected(1); }
            ok
        };
        // The entry, as the generic beam takes it: resolvable, then a
        // frontier seed, and a result if still current and accepted.
        let Some(first_row) = d.resolve_node(self.candidate_node(entry)) else { return };
        let entry_distance = match scoring {
            // The f32 distance the beam computed (exact in f32).
            Scoring::Rows(..) => entry.distance as f32,
            // Codes order by score: score the entry by id like its
            // neighbours (not counted: the descent already counted it).
            Scoring::Codes(_) => {
                let mut score = [f32::NAN];
                d.encoded.as_ref().expect("bound scorer").score_ids(&[entry.id], &mut score);
                let score = if score[0].is_nan() { d.encoded.as_ref().and_then(|e| e.score(first_row.id, first_row.node.key())) } else { Some(score[0]) };
                match score { Some(score) if score.is_finite() && score >= 0.0 => -score, _ => return }
            }
        };
        visited.insert(entry.id);
        let first = pack(entry_distance, entry.id);
        x.cand.push(Reverse(first));
        if !self.candidate_removed(entry) && accepts(entry.id) {
            x.top.push(first);
        }
        // Pinned search_layer_helper: unbounded until the results overflow.
        let mut limit = u64::MAX;
        let mut expansions = 0u32;
        // SAFETY (all views below): snapshot admission validated these
        // stores/table with this guard, held for the whole traversal.
        let levels_view = unsafe { self.control.levels.view_prevalidated(&self.guard) };
        let links_view = unsafe { self.control.links.view_prevalidated(&self.guard) };
        let rows_bytes = dim * 4;
        let mut products = [0f32; 4];
        // While the topology sequence holds the view's quiet value, every
        // node a link names is visible: no visibility words are read.
        let mut quiet = self.quiet;
        while let Some(Reverse(candidate)) = x.cand.pop() {
            if dist_of(candidate) > dist_of(limit) { break; }
            expansions += 1;
            if expansions % 16 == 0 && params.expired() { break; }
            let id = candidate as u32;
            let table = unsafe { self.control.nodes.read_prevalidated(&self.guard) };
            let Some(hot) = table.hot() else { break };
            let Some(links) = self.dense_links(&table, hot, &levels_view, &links_view, id) else { continue };
            // Stage 1: every unvisited neighbour's word and row or codes.
            x.seen.clear();
            for &n in links {
                // Marked once here: a neighbour found invisible below stays
                // invisible to this view (a bridge is decided below, once).
                if n == 0 || n >= self.len || !visited.insert(n) { continue; }
                if quiet.is_none() {
                    let word = &hot.slots[n as usize];
                    crate::vector::distance::prefetch((word as *const node_table::HotSlot).cast(), 8);
                }
                let row = match scoring {
                    Scoring::Rows(plane, _) => {
                        let row = plane.cells_ptr(n);
                        crate::vector::distance::prefetch(row.cast(), rows_bytes);
                        row as usize
                    }
                    Scoring::Codes(plane) => plane.entry_address(n),
                };
                x.seen.push((n, row));
            }
            // Look-ahead, part 1: the next candidate's words.
            let next = x.cand.peek().map(|Reverse(next)| *next as u32);
            if let Some(word) = next.and_then(|next| hot.slots.get(next as usize)) {
                crate::vector::distance::prefetch((word as *const node_table::HotSlot).cast(), 16);
            }
            // Stage 2: visibility. Words are read relaxed and ordered before
            // the rows by one acquire fence (each pairs with its Release
            // publication store).
            x.ok.clear();
            x.rows.clear();
            // Quiet: the links (and the words leading to them) were read
            // before any publication since the view (seqlock check).
            if quiet.is_some_and(|seq| self.control.topology_seq_after_fence() == seq) {
                for &(n, row) in &x.seen { x.ok.push(n); x.rows.push(row); }
                x.seen.clear();
            } else {
                quiet = None;
            }
            for &(n, row) in &x.seen {
                let word = hot.slots[n as usize].vis.load(Rx);
                if word <= publication {
                    x.ok.push(n); x.rows.push(row);
                } else if word == u64::MAX {
                    // Removed after this reader was admitted: a bridge only.
                    if let Some(bridge) = self.removed_bridge(n) {
                        {
                            if let Some(row) = d.resolve_node(bridge) {
                                let key = match scoring {
                                    Scoring::Rows(..) => {
                                        let distance = d.distance_to(&row);
                                        if !distance.is_finite() { continue; }
                                        pack(distance as f32, n)
                                    }
                                    Scoring::Codes(_) => {
                                        d.operations.set(d.operations.get() + 1);
                                        match d.encoded_score(&row) { Some(score) => pack(-score, n), None => continue }
                                    }
                                };
                                if dist_of(key) < dist_of(limit) { x.cand.push(Reverse(key)); }
                            }
                        }
                    }
                }
            }
            fence(Acquire);
            // Look-ahead, part 2: its level-0 links (its words have had
            // stage 2's misses to arrive; the links have the scoring).
            if let Some(slot) = next.and_then(|next| hot.slots.get(next as usize)) {
                let links0 = slot.links0.load(Rx);
                if links0 != 0 {
                    // SAFETY: address only (a stale or racing value is never
                    // dereferenced here).
                    if let Ok(links) = unsafe { links_view.get(LinksRef::from_raw(links0)) } {
                        crate::vector::distance::prefetch(links.as_ptr().cast(), links.len() * 4);
                    }
                }
            }
            note_visited(x.ok.len() as u64);
            if x.ok.is_empty() { continue; }
            // Stage 3: score and admit.
            match scoring {
                Scoring::Rows(plane, metric) => {
                    let _ = plane;
                    d.operations.set(d.operations.get() + x.ok.len() as u64);
                    let mut i = 0;
                    while i < x.ok.len() {
                        let count = (x.ok.len() - i).min(4);
                        let mut ids = [0u32; 4];
                        ids[..count].copy_from_slice(&x.ok[i..i + count]);
                        let group = &ids[..count];
                        // SAFETY: the address of a visible node's written
                        // cells (`dim` of them), read after the fence.
                        let rows: [Option<&[f32]>; 4] = std::array::from_fn(|j| (j < count && x.rows[i + j] != 0)
                            .then(|| unsafe { std::slice::from_raw_parts(x.rows[i + j] as *const f32, dim) }));
                        if metric.uses_dot() && rows.iter().all(Option::is_some) {
                            products = dot_four(d.query, rows.map(|r| r.unwrap()));
                        } else {
                            for (j, r) in rows.iter().enumerate() {
                                products[j] = match r {
                                    Some(r) if metric == Metric::L2 => l2_sq(d.query, r),
                                    // The packed distance itself (as the
                                    // generic beam's `distance_value`).
                                    Some(r) if metric == Metric::Packed => self.similarity.distance(d.query, r),
                                    Some(r) => dot(d.query, r),
                                    None => f32::NAN,
                                };
                            }
                        }
                        for (j, &n) in group.iter().enumerate() {
                            let distance = match metric {
                                Metric::L2 | Metric::Packed => products[j],
                                Metric::Dot => -products[j],
                                _ => {
                                    // Cosine: the record's cached inverse norm.
                                    let Some(node) = self.current_node(n) else { continue };
                                    let inverse = node.inverse_norm();
                                    if inverse == 0.0 || d.inverse_query_norm == 0.0 { 1.0 }
                                    else { 1.0 - products[j] * d.inverse_query_norm * inverse }
                                }
                            };
                            if rows[j].is_none() {
                                // Never written: the generic resolution
                                // (and its missing-source accounting).
                                let Some(node) = self.current_node(n) else { continue };
                                let Some(r) = d.resolve_node(node) else { continue };
                                let distance = d.distance_value(&r) as f32;
                                self.dense_admit(distance, n, ef, &mut limit, x, &accepts);
                                continue;
                            }
                            if !distance.is_finite() { continue; }
                            self.dense_admit(distance, n, ef, &mut limit, x, &accepts);
                        }
                        i += group.len();
                    }
                }
                Scoring::Codes(_) => {
                    let encoded = d.encoded.as_ref().expect("bound scorer");
                    x.scores.clear();
                    x.scores.resize(x.ok.len(), f32::NAN);
                    if !encoded.score_entries(&x.rows, &mut x.scores) {
                        encoded.score_ids(&x.ok, &mut x.scores);
                    }
                    d.operations.set(d.operations.get() + x.ok.len() as u64);
                    for j in 0..x.ok.len() {
                        let (n, mut score) = (x.ok[j], x.scores[j]);
                        if score.is_nan() {
                            // No plane entry: the row's codes.
                            let Some(node) = self.current_node(n) else { continue };
                            score = match encoded.score(node.handle(), node.key()) { Some(v) => v, None => f32::NAN };
                        }
                        if !score.is_finite() || score < 0.0 {
                            d.missing_source.set(d.missing_source.get() + 1);
                            continue;
                        }
                        self.dense_admit(-score, n, ef, &mut limit, x, &accepts);
                    }
                }
            }
        }
        // Ascending (distance, id), as the generic beam's sorted output.
        let mut out: Vec<u64> = x.top.drain().collect();
        out.sort_unstable();
        s.out_scores.clear();
        s.out.extend(out.into_iter().map(|key| {
            let value = unordered(dist_of(key));
            let distance = match scoring {
                Scoring::Rows(..) => f64::from(value),
                Scoring::Codes(_) => { s.out_scores.push(-value); d.distance_of_score(-value) }
            };
            Candidate { distance, id: key as u32, levels: LevelsRef::INVALID }
        }));
    }

    /// The upper-level greedy descent of a search (`greedy_acquired_bridged`)
    /// over a dense source: the same walk, distances and work, each
    /// neighbourhood's words and rows or codes warmed and scored together
    /// (codes through the plane, not each row through the source
    /// directory). `None` when the view or source does not qualify.
    pub(super) fn dense_greedy<S: MutableVectorSource + ?Sized>(
        &self,
        d: &SourceDistance<'_, S>,
        mut current: Candidate,
        level: usize,
    ) -> Option<Candidate> {
        if !dense_enabled() || self.captured.is_some() { return None; }
        let scoring = match d.source.dense_layout() {
            Some(DenseLayout::Rows(plane)) if d.encoded.is_none() => match self.similarity.metric() {
                metric @ (Metric::Dot | Metric::L2 | Metric::Packed) => Scoring::Rows(plane, metric),
                _ => return None,
            },
            Some(DenseLayout::Codes(plane)) if d.scores_by_id() => Scoring::Codes(plane),
            _ => return None,
        };
        let dim = self.dim as usize;
        let mut ids = [0u32; 64];
        let mut rows = [0usize; 64];
        let mut scores = [0f32; 64];
        loop {
            let mut improved = false;
            // SAFETY: snapshot admission validated this table/guard pair.
            let table = unsafe { self.control.nodes.read_prevalidated(&self.guard) };
            let hot = table.hot()?;
            let Some(links) = self.candidate_node_typed::<false>(current).links_typed::<false>(level) else { break };
            // Visible neighbours in link order (a removed one, as a bridge,
            // through the generic resolution below), warmed together.
            let mut n = 0;
            for &id in links {
                if n == ids.len() { break; }
                if id == 0 || id >= self.len { continue; }
                let word = hot.slots.get(id as usize)?.vis.load(Acquire);
                if word > self.publication { 
                    if word == u64::MAX && self.removed_bridge(id).is_some() { ids[n] = id; rows[n] = usize::MAX; n += 1; }
                    continue;
                }
                ids[n] = id;
                rows[n] = match scoring {
                    Scoring::Rows(plane, _) => { let p = plane.cells_ptr(id); crate::vector::distance::prefetch(p.cast(), dim * 4); p as usize }
                    Scoring::Codes(plane) => plane.entry_address(id),
                };
                n += 1;
            }
            if n < links.len() && n == ids.len() {
                // Wider than a level's links ever are: the generic walk.
                return None;
            }
            if let Scoring::Codes(_) = scoring {
                let encoded = d.encoded.as_ref().expect("bound scorer");
                let mut entries = [0usize; 64];
                for j in 0..n { entries[j] = if rows[j] == usize::MAX { 0 } else { rows[j] }; }
                scores[..n].fill(f32::NAN);
                if !encoded.score_entries(&entries[..n], &mut scores[..n]) { return None; }
            }
            for j in 0..n {
                let id = ids[j];
                let distance = if rows[j] == usize::MAX || (matches!(scoring, Scoring::Rows(..)) && rows[j] == 0)
                    || (matches!(scoring, Scoring::Codes(_)) && scores[j].is_nan()) {
                    // A bridge, or no dense entry: as `greedy_acquired_typed`.
                    let node = self.current_node(id).or_else(|| self.removed_bridge(id));
                    let Some(row) = node.and_then(|node| d.resolve_node(node)) else { continue };
                    d.distance_to(&row)
                } else {
                    d.operations.set(d.operations.get() + 1);
                    match scoring {
                        Scoring::Rows(_, metric) => {
                            // SAFETY: a visible node's written cells, read after
                            // the acquire of its word.
                            let row = unsafe { std::slice::from_raw_parts(rows[j] as *const f32, dim) };
                            f64::from(match metric {
                                Metric::L2 => l2_sq(d.query, row),
                                Metric::Packed => self.similarity.distance(d.query, row),
                                _ => -dot(d.query, row),
                            })
                        }
                        Scoring::Codes(_) => {
                            let score = scores[j];
                            if !score.is_finite() || score < 0.0 {
                                d.missing_source.set(d.missing_source.get() + 1);
                                f64::INFINITY
                            } else { d.distance_of_score(score) }
                        }
                    }
                };
                if distance < current.distance {
                    let node = self.current_node(id).or_else(|| self.removed_bridge(id))?;
                    current = Candidate::new(distance, node);
                    improved = true;
                }
            }
            if !improved { break; }
        }
        Some(current)
    }

    /// Admit one scored neighbour (`visit_batch`'s rule, slack 0).
    #[inline(always)]
    fn dense_admit(&self, distance: f32, id: u32, ef: usize, limit: &mut u64, x: &mut DenseScratch, accepts: &impl Fn(u32) -> bool) {
        let key = pack(distance, id);
        if dist_of(key) < dist_of(*limit) {
            x.cand.push(Reverse(key));
            if accepts(id) {
                if x.top.len() < ef {
                    x.top.push(key);
                } else {
                    // Full: push then pop the largest, in one sift (the
                    // largest is replaced unless `key` is it).
                    let mut largest = x.top.peek_mut().expect("positive query width");
                    if key < *largest { *largest = key; }
                    drop(largest);
                    *limit = *x.top.peek().expect("positive query width");
                }
            }
        }
    }

    /// Node `id`'s current level-0 links (a removed node's retained ones
    /// for this reader, as `removed_bridge`), `None` if it has none.
    #[inline(always)]
    fn dense_links<'v>(
        &'v self,
        table: &node_table::NodeTableRead<'v>,
        hot: HotNodes<'v>,
        levels_view: &array_store::ArrayReadView<'v, AtomicU32, 22>,
        links_view: &array_store::ArrayReadView<'v, u32, 20>,
        id: u32,
    ) -> Option<&'v [u32]> {
        // The dense word: the current level-0 links (0: read the levels).
        let links0 = hot.slots.get(id as usize)?.links0.load(Acquire);
        if links0 != 0 {
            // SAFETY: a links reference current at some instant after this
            // reader's guard was taken (the writer updates the word before
            // it retires the links it replaced), so held for this guard.
            return unsafe { links_view.get(LinksRef::from_raw(links0)) }.ok();
        }
        let raw = hot.levels.get(id as usize)?.load(Acquire);
        let levels = if raw != 0 { LevelsRef::from_raw(raw) } else {
            table.get(id)?.removed_levels_for(self.guard.generation())?
        };
        // SAFETY: a reference acquired from the current owner (or retained
        // for this guard) under this continuously held guard.
        let refs = unsafe { levels_view.get(levels) }.ok()?;
        let links = LinksRef::from_raw(refs.first()?.load(Acquire));
        unsafe { links_view.get(links) }.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_rows(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = XorShift64Star::new(seed);
        (0..n).map(|_| (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect()).collect()
    }

    /// The dense beam returns exactly the generic beam's hits (keys and
    /// scores) and work, for f32 rows (each metric) and int8 codes, on a
    /// graph with removed nodes (not quiet) and on a fresh one (quiet),
    /// unfiltered, with a node filter, and with a key predicate.
    #[test]
    fn dense_beam_matches_the_generic_beam() {
        for (similarity, quantized) in [(Similarity::DotProduct, false), (Similarity::Cosine, false), (Similarity::L2, false),
            (Similarity::MaxInnerProduct, false), (Similarity::DotProduct, true), (Similarity::L2, true),
            (Similarity::ByteMaxInnerProduct, false), (Similarity::ByteDotProduct, false), (Similarity::ByteL2, false),
            (Similarity::ByteCosine, false)] {
            let dim = 24;
            let packed = similarity.metric() == Metric::Packed;
            // Packed `byte` rows: 96 integer values, four per lane.
            let byte_rows = |n: usize, seed: u64| -> Vec<Vec<f32>> {
                let mut rng = XorShift64Star::new(seed);
                (0..n).map(|_| crate::vector::packed::pack(&(0..dim * 4).map(|_| (rng.next_f64() * 255.0) as u8).collect::<Vec<u8>>())).collect()
            };
            let mut rows = if packed { byte_rows(1500, 0x61) } else { unit_rows(1500, dim, 0x51 + quantized as u64) };
            if similarity == Similarity::DotProduct {
                for r in &mut rows { let n = r.iter().map(|x| x * x).sum::<f32>().sqrt(); r.iter_mut().for_each(|x| *x /= n); }
            }
            let graph = Arc::new(MutableHnsw::new(dim as u32, similarity, HnswParams { m: 8, ef_construction: 40, ..HnswParams::default() }));
            let owner = if quantized {
                MutableVectorOwner::new_quantized(graph.clone(), Some(crate::vector::es_quant::EsQuant { bits: 7, osq: true, confidence_interval: None })).unwrap()
            } else { MutableVectorOwner::new(graph.clone()).unwrap() };
            let mut handles = Vec::new();
            for (k, row) in rows.iter().enumerate() {
                let reservation = graph.reserve_insert(k as u64).unwrap();
                let code = owner.prepare_codes(&[row]).unwrap().pop().flatten();
                owner.install_prepared(&reservation, row, code.as_ref()).unwrap();
                let read = owner.pin().unwrap();
                let prepared = graph.prepare(reservation, &read).unwrap();
                handles.push(graph.complete(prepared, &read).unwrap());
            }
            owner.commit().unwrap();
            let mut queries = if packed { byte_rows(12, 0x79) } else { unit_rows(12, dim, 0x77) };
            // Unit queries: dot-product scores stay unsaturated (no ties).
            if !packed { for r in &mut queries { let n = r.iter().map(|x| x * x).sum::<f32>().sqrt(); r.iter_mut().for_each(|x| *x /= n); } }
            let compare = |label: &str, expect_quiet: bool| {
                let (view, read) = owner.read().unwrap();
                assert_eq!(view.quiet.is_some(), expect_quiet, "{label}");
                let all = vec![u64::MAX; (view.len as usize).div_ceil(64)];
                let some: Vec<u64> = (0..all.len()).map(|i| 0x5555_5555_5555_5555u64.rotate_left(i as u32)).collect();
                for q in &queries {
                    let run = |generic: bool| {
                        GENERIC.with(|g| g.set(generic));
                        let plain = MutableSearchMode::VespaPlain(VespaPlainParams::default());
                        let out = (view.search(q, 10, 60, &read, |key| key % 3 != 0),
                            view.search_with_exact_node_filter(q, 10, 60, &read, |h, _| { let id = h.slot() as usize + 1; all[id / 64] & (1 << (id % 64)) != 0 }, plain, &all).unwrap(),
                            view.search_with_exact_node_filter(q, 10, 60, &read, |h, _| { let id = h.slot() as usize + 1; some[id / 64] & (1 << (id % 64)) != 0 }, plain, &some).unwrap());
                        GENERIC.with(|g| g.set(false));
                        out
                    };
                    let (dense, generic) = (run(false), run(true));
                    for (d, g) in [(&dense.0, &generic.0), (&dense.1, &generic.1), (&dense.2, &generic.2)] {
                        assert_eq!(d.hits, g.hits, "{label} {similarity:?} quantized={quantized}");
                        assert_eq!(d.vector_ops, g.vector_ops, "{label} {similarity:?} quantized={quantized}");
                        assert_eq!(d.missing_source_lookups, 0);
                        assert!(!d.hits.is_empty());
                    }
                }
            };
            compare("fresh", true);
            for (k, h) in handles.iter().enumerate() {
                if k % 5 == 1 { let read = owner.pin().unwrap(); assert!(graph.remove(*h, &read).unwrap()); }
            }
            compare("after removals", false);
        }
    }
}
