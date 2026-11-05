//! Boolean composition of scorers: AND, OR, NOT, and minimum-should-match.

use crate::DocId;
use crate::query::bulk::Floor;
use crate::query::docset::{DocSet, EmptyDocSet, TERMINATED};
use crate::query::scorer::{ConstScorer, Scorer};

/// Intersection of two doc sets (AND).
///
/// Uses leapfrog: the side with the smaller current doc seeks to the other's
/// doc until both agree.
pub struct Intersection<L: DocSet, R: DocSet> {
    left: L,
    right: R,
    primed: bool,
    current: DocId,
}

impl<L: DocSet, R: DocSet> Intersection<L, R> {
    /// Build an intersection of `left` and `right`.
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            primed: false,
            current: TERMINATED,
        }
    }

    fn step(&mut self) -> DocId {
        loop {
            let l = self.left.doc();
            let r = self.right.doc();
            if l == TERMINATED || r == TERMINATED {
                self.current = TERMINATED;
                return TERMINATED;
            }
            if l == r {
                self.current = l;
                return l;
            }
            if l < r {
                self.left.seek(r);
            } else {
                self.right.seek(l);
            }
        }
    }
}

impl<L: DocSet, R: DocSet> DocSet for Intersection<L, R> {
    fn advance(&mut self) -> DocId {
        if !self.primed {
            self.left.advance();
            self.right.advance();
            self.primed = true;
        } else if self.current != TERMINATED {
            self.left.advance();
        } else {
            return TERMINATED;
        }
        self.step()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if !self.primed {
            self.left.seek(target);
            self.right.seek(target);
            self.primed = true;
        } else {
            if self.left.doc() < target {
                self.left.seek(target);
            }
            if self.right.doc() < target {
                self.right.seek(target);
            }
        }
        self.step()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.left.size_hint().min(self.right.size_hint())
    }
}

impl<L: Scorer, R: Scorer> Scorer for Intersection<L, R> {
    fn score_upper_bound(&self) -> Option<f32> {
        let bound = self.left.score_upper_bound()? + self.right.score_upper_bound()?;
        bound.is_finite().then_some(bound)
    }

    fn score(&mut self) -> f32 {
        self.left.score() + self.right.score()
    }
}

/// Intersection of any number of scorers (AND).
///
/// Prefer a posting iterator as driver; pure exact-lookup children verify only
/// its candidates. Among iterators of the same kind the smallest `size_hint`
/// drives. Score is the sum of the children in their original clause order.
pub struct IntersectionN {
    children: Vec<Box<dyn Scorer>>,
    driver: usize,
    primed: bool,
    current: DocId,
}

impl IntersectionN {
    /// Build an intersection of the given scorers. With no children the
    /// result is empty.
    pub fn new(mut children: Vec<Box<dyn Scorer>>) -> Self {
        // A clause whose exact seeks read a value (a doc values range)
        // leads over one of up to twice its size: testing the other's bits
        // per candidate is cheaper than reading its values.
        let driver = children
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| (c.has_fast_exact_seek(), if c.exact_seek_reads_value() { c.size_hint() / 2 } else { c.size_hint() }))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Verifiers learn the leading clause's cost (Lucene's leadCost).
        if let Some(lead) = children.get(driver).map(|c| c.size_hint()) {
            for (i, c) in children.iter_mut().enumerate() {
                if i != driver && c.has_fast_exact_seek() {
                    c.set_lead_cost(lead);
                }
            }
        }
        Self {
            children,
            driver,
            primed: false,
            current: TERMINATED,
        }
    }

    fn step(&mut self) -> DocId {
        loop {
            let d = self.children[self.driver].doc();
            if d == TERMINATED {
                self.current = TERMINATED;
                return TERMINATED;
            }
            let mut matched = true;
            for (i, c) in self.children.iter_mut().enumerate() {
                if i == self.driver {
                    continue;
                }
                if c.has_fast_exact_seek() {
                    if !c.seek_exact(d) { matched = false; break; }
                    continue;
                }
                let mut cd = c.doc();
                if cd < d {
                    cd = c.seek(d);
                }
                if cd == TERMINATED {
                    self.current = TERMINATED;
                    return TERMINATED;
                }
                if cd != d {
                    matched = false;
                    break;
                }
            }
            if matched {
                self.current = d;
                return d;
            }
            self.children[self.driver].advance();
        }
    }
}

impl DocSet for IntersectionN {
    fn advance(&mut self) -> DocId {
        if self.children.is_empty() {
            return TERMINATED;
        }
        if !self.primed {
            for (i, c) in self.children.iter_mut().enumerate() {
                if i == self.driver || !c.has_fast_exact_seek() { c.advance(); }
            }
            self.primed = true;
        } else if self.current != TERMINATED {
            self.children[self.driver].advance();
        } else {
            return TERMINATED;
        }
        self.step()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.children.is_empty() {
            return TERMINATED;
        }
        for (i, c) in self.children.iter_mut().enumerate() {
            if (i == self.driver || !c.has_fast_exact_seek()) && (!self.primed || c.doc() < target) {
                c.seek(target);
            }
        }
        self.primed = true;
        self.step()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.size_hint())
            .min()
            .unwrap_or(0)
    }
}

impl Scorer for IntersectionN {
    /// Each clause must score above the floor less the others' bounds.
    fn set_min_competitive_score(&mut self, floor: f32) {
        if !floor.is_finite() {
            return;
        }
        let Some(bounds) = self.children.iter().map(|c| c.score_upper_bound().filter(|b| b.is_finite() && *b >= 0.0)).collect::<Option<Vec<f32>>>() else { return };
        let total: f64 = bounds.iter().map(|&b| b as f64).sum();
        for (i, c) in self.children.iter_mut().enumerate() {
            // Slack keeps rounding from raising a clause's floor too far.
            let rest = (total - bounds[i] as f64) * (1.0 + 1e-5);
            let f = floor as f64 - rest;
            if f > 0.0 {
                c.set_min_competitive_score(f as f32);
            }
        }
    }

    fn score_upper_bound(&self) -> Option<f32> {
        // Sum in the same clause order as score(); nonnegative bounds
        // also bound every subset of optional clauses.
        let bound = self.children.iter().try_fold(0.0f32, |sum, c| Some(sum + c.score_upper_bound()?))?;
        bound.is_finite().then_some(bound)
    }

    fn score(&mut self) -> f32 {
        self.children.iter_mut().map(|c| c.score()).sum()
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.collect_top_k_bulk_min(k, deleted, f32::NEG_INFINITY)
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        let out = crate::query::bulk::try_conjunction_min(&mut self.children, k, deleted, floor)?;
        self.primed = true;
        self.current = TERMINATED;
        Some(out)
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        // A constant-scored doc set (a scored range: ES adds its constant to
        // every hit) AND a scored clause: the scored clause's block-max path
        // over the set's docs, each hit scoring the constant more (two
        // clauses: their f32 sum is the same in either order).
        if self.children.len() == 2 {
            for (f, r) in [(0usize, 1usize), (1, 0)] {
                if !self.children[r].can_collect_filtered() {
                    continue;
                }
                if let Some((keep, add)) = self.children[f].const_filter() {
                    if add.is_finite() && add >= 0.0 {
                        if let Some(out) = self.children[r].collect_top_k_filtered(k, deleted, &keep, add, count_limit, floor) {
                            self.primed = true;
                            self.current = TERMINATED;
                            return Some(out);
                        }
                    }
                    break;
                }
            }
        }
        let out = crate::query::bulk::try_conjunction_lim(&mut self.children, k, deleted, count_limit, floor)?;
        self.primed = true;
        self.current = TERMINATED;
        Some(out)
    }
}

/// A min-heap of child indices keyed on each child's current doc.
///
/// Children are stored in a parallel `docs` array so the heap can be
/// updated in place (Lucene's `DisiPriorityQueue`). All operations are
/// allocation-free after construction.
struct DocHeap {
    /// Current doc of each child, indexed by child id.
    docs: Vec<DocId>,
    /// Heap of child ids, ordered by `docs[id]`.
    heap: Vec<usize>,
}

impl DocHeap {
    fn new(docs: Vec<DocId>) -> Self {
        let heap: Vec<usize> = (0..docs.len()).collect();
        let mut h = Self { docs, heap };
        h.rebuild();
        h
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    fn less(&self, a: usize, b: usize) -> bool {
        let (da, db) = (self.docs[self.heap[a]], self.docs[self.heap[b]]);
        da < db || (da == db && self.heap[a] < self.heap[b])
    }

    /// Child id at the top (smallest doc).
    fn top(&self) -> usize {
        self.heap[0]
    }

    /// Doc of the top child.
    fn top_doc(&self) -> DocId {
        self.docs[self.heap[0]]
    }

    /// After changing `docs[top]`, restore the heap property.
    fn update_top(&mut self) {
        self.sift_down(0);
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            let mut m = i;
            if l < n && self.less(l, m) {
                m = l;
            }
            if r < n && self.less(r, m) {
                m = r;
            }
            if m == i {
                return;
            }
            self.heap.swap(i, m);
            i = m;
        }
    }

    /// Heapify from scratch (after every child moved).
    fn rebuild(&mut self) {
        let n = self.heap.len();
        for i in (0..n / 2).rev() {
            self.sift_down(i);
        }
    }
}

/// Union of any number of scorers (OR), including 0 and 1 children.
///
/// A heap keyed on each child's current doc yields the minimum; score is the
/// sum of all children currently on that doc.
pub struct Union {
    children: Vec<Box<dyn Scorer>>,
    heap: DocHeap,
    primed: bool,
    current: DocId,
    matched: Vec<usize>,
    matched_current: Option<DocId>,
    /// Clauses every competitive doc must match (see
    /// [`Scorer::set_min_competitive_score`]); empty: plain OR.
    required: Vec<usize>,
    min_competitive: f32,
}

impl Union {
    /// Build a union of the given scorers.
    pub fn new(children: Vec<Box<dyn Scorer>>) -> Self {
        let docs = children.iter().map(|c| c.doc()).collect();
        Self {
            children,
            heap: DocHeap::new(docs),
            primed: false,
            current: TERMINATED,
            matched: Vec::new(),
            matched_current: None,
            required: Vec::new(),
            min_competitive: f32::NEG_INFINITY,
        }
    }

    /// The next doc at or after `target` that every required clause
    /// matches (leapfrog), with the other clauses moved onto it so they
    /// score it when they match too.
    fn next_required(&mut self, mut target: DocId) -> DocId {
        'outer: loop {
            if target == TERMINATED {
                break TERMINATED;
            }
            for &i in &self.required {
                let c = &mut self.children[i];
                let d = if c.doc() < target { c.seek(target) } else { c.doc() };
                if d == TERMINATED {
                    break 'outer TERMINATED;
                }
                if d > target {
                    target = d;
                    continue 'outer;
                }
            }
            break target;
        }
    }

    fn advance_required(&mut self, target: DocId) -> DocId {
        self.matched_current = None;
        let d = self.next_required(target);
        if d != TERMINATED {
            for (i, c) in self.children.iter_mut().enumerate() {
                if c.doc() < d {
                    self.heap.docs[i] = c.seek(d);
                } else {
                    self.heap.docs[i] = c.doc();
                }
            }
        }
        self.primed = true;
        self.current = d;
        d
    }
}

impl DocSet for Union {
    fn advance(&mut self) -> DocId {
        if !self.required.is_empty() {
            if self.current == TERMINATED && self.primed {
                return TERMINATED;
            }
            let from = if self.primed { self.current.saturating_add(1) } else { 0 };
            return self.advance_required(from);
        }
        self.matched_current = None;
        if self.heap.len() == 0 {
            self.current = TERMINATED;
            return TERMINATED;
        }
        if !self.primed {
            for (i, c) in self.children.iter_mut().enumerate() {
                self.heap.docs[i] = c.advance();
            }
            self.heap.rebuild();
            self.primed = true;
        } else if self.current != TERMINATED {
            // Advance every child sitting on the current doc.
            while self.heap.top_doc() == self.current {
                let idx = self.heap.top();
                self.heap.docs[idx] = self.children[idx].advance();
                self.heap.update_top();
            }
        } else {
            return TERMINATED;
        }
        self.current = self.heap.top_doc();
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if !self.required.is_empty() {
            return self.advance_required(target);
        }
        self.matched_current = None;
        if self.heap.len() == 0 {
            self.current = TERMINATED;
            return TERMINATED;
        }
        for (i, c) in self.children.iter_mut().enumerate() {
            if !self.primed || (c.doc() < target) {
                self.heap.docs[i] = c.seek(target);
            }
        }
        self.heap.rebuild();
        self.primed = true;
        self.current = self.heap.top_doc();
        self.current
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.size_hint())
            .fold(0u32, |a, b| a.saturating_add(b))
    }
}

impl Union {
    /// Heap descendants cannot match when their parent is already beyond the
    /// current document. Avoid calling every child scorer's virtual doc() for
    /// sparse, many-clause disjunctions (e.g. query-rule pins plus organic hits).
    fn prepare_matches(&mut self) {
        if self.matched_current == Some(self.current) { return; }
        fn visit(heap: &DocHeap, at: usize, doc: DocId, out: &mut Vec<usize>) {
            let Some(&child) = heap.heap.get(at) else { return };
            if heap.docs[child] != doc { return; }
            out.push(child);
            visit(heap, at * 2 + 1, doc, out);
            visit(heap, at * 2 + 2, doc, out);
        }
        self.matched.clear();
        // Score summation must retain clause order, including rounding. A
        // few clauses (a multi-field word's dis_max, a short query): one
        // linear scan beats the heap walk plus a sort. (Walking as an AND,
        // the heap is not maintained.)
        if self.children.len() <= 16 || !self.required.is_empty() {
            let cur = self.current;
            self.matched.extend(self.heap.docs.iter().enumerate().filter_map(|(i, &doc)| (doc == cur).then_some(i)));
            self.matched_current = Some(self.current);
            return;
        }
        visit(&self.heap, 0, self.current, &mut self.matched);
        // For a dense match, a linear directory scan avoids sorting most
        // children.
        if self.matched.len() > self.children.len() / 4 {
            self.matched.clear();
            self.matched.extend(self.heap.docs.iter().enumerate().filter_map(|(i, &doc)| (doc == self.current).then_some(i)));
        } else {
            self.matched.sort_unstable();
        }
        self.matched_current = Some(self.current);
    }

    /// The children, while none has been advanced (bulk collection).
    pub(crate) fn children_mut(&mut self) -> Option<&mut [Box<dyn Scorer>]> {
        (!self.primed).then_some(self.children.as_mut_slice())
    }

    /// Every child's [`Scorer::max_score`], if all have one.
    pub(crate) fn child_max_scores(&self) -> Option<Vec<f32>> {
        self.children.iter().map(|c| c.max_score().filter(|b| b.is_finite() && *b >= 0.0)).collect()
    }

    /// Every child's [`Scorer::max_score_up_to`], if all have one.
    pub(crate) fn child_max_scores_up_to(&mut self, up_to: DocId) -> Option<Vec<f32>> {
        self.children.iter_mut().map(|c| c.max_score_up_to(up_to).filter(|b| b.is_finite() && *b >= 0.0)).collect()
    }

    /// MaxScore top-k (Lucene `MaxScoreBulkScorer`). Every match is counted
    /// until `count_limit`; after that, each 4096-doc window partitions the
    /// clauses by their bounds over the window ([`Scorer::max_score_up_to`]):
    /// the clauses whose bounds sum to at most the k-th best score are
    /// non-essential, only the other clauses' docs are candidates, and a
    /// non-essential clause is only probed while the candidate could still
    /// beat the threshold. Scores are summed in clause order, as
    /// [`Scorer::score`] does, so hits and ties match the per-doc loop.
    /// Hits must also beat `floor` (other leaves' hits).
    fn max_score_top_k(&mut self, k: usize, del: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.max_score_top_k_in(k, del, None, 0.0, count_limit, floor)
    }

    /// [`Union::max_score_top_k`] over the docs set in `keep` only (`None`:
    /// every doc), each hit scoring `add` more (a conjunction's constant
    /// clause; see [`Scorer::collect_top_k_filtered`]). Docs outside `keep`
    /// are skipped to the next doc it holds, uncounted and unscored.
    fn max_score_top_k_in(&mut self, k: usize, del: Option<&[u64]>, keep: Option<&[u64]>, add: f32, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        const WINDOW: DocId = 4096;
        if self.primed || self.children.len() < 2 || k == 0 {
            return None;
        }
        let global = self.child_max_scores()?;
        let n = self.children.len();
        // Equal bounds: the costliest clauses turn non-essential first, so
        // the rarest ones drive (Lucene `MaxScoreBulkScorer` orders ties by
        // descending cost), e.g. `query_string` words of constant score.
        let cost: Vec<u32> = self.children.iter().map(|c| c.size_hint()).collect();
        let mut docs: Vec<DocId> = self.children.iter_mut().map(|c| c.advance()).collect();
        self.primed = true;
        self.current = TERMINATED;
        let mut bounds = global.clone();
        let mut order: Vec<usize> = (0..n).collect();
        // f64 prefix sums, compared with slack, so f32 rounding of the real
        // sums can never prune a competitive document.
        let mut prefix = vec![0f64; n + 1];
        let below = |bound: f64, thr: f32| bound * (1.0 + 1e-5) < thr as f64;
        let mut top = crate::query::bulk::TopK::with_floor(k, floor.get());
        let mut total = 0u64;
        let mut published = 0u64;
        let mut ne = 0usize;
        let mut hit: Vec<(usize, f32)> = Vec::with_capacity(n);
        let mut steps = 0u32;
        // Docs below `pos` are done; the partition holds up to `win_end`.
        let mut pos: DocId = 0;
        let mut win_end: DocId = 0;
        // The best sum any doc can reach, added in clause order as the
        // scores are (f32 addition is monotonic, so no doc sums higher).
        let reachable = global.iter().fold(0f32, |s, &b| s + b);
        let reachable = if add == 0.0 { reachable } else { add + reachable };
        // The threshold the clauses' bounds are compared with: less `add`.
        let thr = |top: &crate::query::bulk::TopK| crate::query::bulk::shifted_threshold(top.threshold(), add);
        // The first doc at or after `d` that `keep` holds.
        let next_kept = |kp: &[u64], d: DocId| -> DocId {
            let mut w = (d >> 6) as usize;
            let Some(&first) = kp.get(w) else { return TERMINATED };
            let mut bits = first & (!0u64 << (d & 63));
            while bits == 0 {
                w += 1;
                match kp.get(w) {
                    Some(&x) => bits = x,
                    None => return TERMINATED,
                }
            }
            (w as DocId) * 64 + bits.trailing_zeros()
        };
        loop {
            steps = steps.wrapping_add(1);
            if steps % 1024 == 0 {
                top.sync(floor);
            }
            // Shard-wide, as the block paths count (Lucene's shared
            // hits threshold): other leaves' published matches count too.
            let counted = total >= count_limit || floor.hits_counted() >= count_limit;
            // Counted, and the k retained hits already score the most any
            // doc can: later docs at most tie, and ties go to earlier docs.
            if counted && top.threshold() >= reachable {
                break;
            }
            if counted && pos >= win_end {
                // A new window at `pos`: re-partition by the clauses' bounds
                // over it (each from its own current doc, which may lag).
                top.sync(floor);
                let end = (pos & !(WINDOW - 1)).saturating_add(WINDOW);
                for (i, c) in self.children.iter_mut().enumerate() {
                    bounds[i] = c.max_score_up_to(end - 1).filter(|b| b.is_finite() && *b >= 0.0).unwrap_or(global[i]);
                }
                order.sort_by(|&a, &b| bounds[a].total_cmp(&bounds[b]).then(cost[b].cmp(&cost[a])));
                for (j, &i) in order.iter().enumerate() {
                    prefix[j + 1] = prefix[j] + bounds[i] as f64;
                }
                ne = 0;
                while ne < n && below(prefix[ne + 1], thr(&top)) {
                    ne += 1;
                }
                // Clauses that turn essential skip what earlier windows
                // already ruled out.
                for &i in &order[ne..] {
                    if docs[i] < pos {
                        docs[i] = self.children[i].seek(pos);
                    }
                }
                win_end = end;
            } else if counted {
                while ne < n && below(prefix[ne + 1], thr(&top)) {
                    ne += 1;
                }
            }
            let d = order[ne..].iter().map(|&i| docs[i]).min().unwrap_or(TERMINATED);
            if counted && d >= win_end {
                // The window holds no more candidates. The next one starts
                // at the first doc any clause may still match: a clause
                // non-essential here may be essential (and competitive)
                // in a later window, so do not jump straight to `d`.
                // These clauses were non-competitive throughout this window.
                // Advance their cursors past it so exhausted clauses cannot
                // keep generating empty windows up to DocId::MAX.
                for &i in &order[..ne] {
                    if docs[i] < win_end {
                        docs[i] = self.children[i].seek(win_end);
                    }
                }
                let next = order[..ne].iter().map(|&i| docs[i]).min().unwrap_or(TERMINATED).min(d);
                if next == TERMINATED {
                    break;
                }
                pos = next;
                continue;
            }
            if d == TERMINATED {
                break;
            }
            if let Some(kp) = keep {
                let nd = next_kept(kp, d);
                if nd != d {
                    if nd == TERMINATED {
                        break;
                    }
                    for &i in &order[ne..] {
                        if docs[i] < nd {
                            docs[i] = self.children[i].seek(nd);
                        }
                    }
                    pos = nd;
                    continue;
                }
            }
            pos = d.saturating_add(1);
            hit.clear();
            let mut partial = 0f64;
            for &i in &order[ne..] {
                if docs[i] == d {
                    let s = self.children[i].score();
                    hit.push((i, s));
                    partial += s as f64;
                    docs[i] = self.children[i].advance();
                }
            }
            let mut pruned = false;
            for j in (0..ne).rev() {
                if below(partial + prefix[j + 1], thr(&top)) {
                    pruned = true;
                    break;
                }
                let i = order[j];
                if docs[i] < d {
                    docs[i] = self.children[i].seek(d);
                }
                if docs[i] == d {
                    let s = self.children[i].score();
                    hit.push((i, s));
                    partial += s as f64;
                }
            }
            if del.is_some_and(|w| w.get((d >> 6) as usize).is_some_and(|x| (x >> (d & 63)) & 1 == 1)) {
                continue;
            }
            total += 1;
            if total - published >= crate::query::bulk::COUNT_BATCH && total <= count_limit {
                floor.add_counted(total - published);
                published = total;
            }
            if pruned {
                continue;
            }
            hit.sort_unstable_by_key(|h| h.0);
            let s: f32 = hit.iter().map(|h| h.1).sum();
            top.offer(d, if add == 0.0 { s } else { add + s });
        }
        floor.add_counted(total - published);
        top.sync(floor);
        Some((top.into_sorted(), total))
    }

    /// Scores of every child positioned on the current doc (for dis_max).
    pub fn child_scores(&mut self, out: &mut Vec<f32>) {
        self.prepare_matches();
        for &i in &self.matched { out.push(self.children[i].score()); }
    }
}

impl Scorer for Union {
    fn max_score(&self) -> Option<f32> {
        let b = self.child_max_scores()?;
        let sum: f32 = b.iter().sum();
        sum.is_finite().then_some(sum)
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        let b = self.child_max_scores_up_to(up_to)?;
        let sum: f32 = b.iter().sum();
        sum.is_finite().then_some(sum)
    }

    fn can_collect_filtered(&mut self) -> bool {
        !self.primed && self.children.len() >= 2 && self.child_max_scores().is_some()
    }

    fn collect_top_k_filtered(&mut self, k: usize, deleted: Option<&[u64]>, keep: &[u64], add: f32, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        if let Some(out) = crate::query::bulk::try_term_groups_in(self.children.iter_mut(), k, deleted, Some(keep), add, count_limit, floor) {
            self.primed = true;
            self.current = TERMINATED;
            return Some(out);
        }
        // As the unfiltered path: an exact total visits every match anyway.
        if count_limit == u64::MAX {
            return None;
        }
        self.max_score_top_k_in(k, deleted, Some(keep), add, count_limit, floor)
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        // Block OR of BM25 terms / dis_max term groups first.
        if let Some(out) = crate::query::bulk::try_term_groups_lim(self.children.iter_mut(), k, deleted, count_limit, floor) {
            self.primed = true;
            self.current = TERMINATED;
            return Some(out);
        }
        // An exact total visits every match anyway: the per-doc loop.
        if count_limit == u64::MAX {
            return None;
        }
        // Many sparse token clauses (sparse_vector / text_expansion): the
        // window scan (in parallel parts on a large segment) beats MaxScore,
        // whose essential clauses each still walk a long list doc by doc and
        // whose candidates probe the others (msmarco on large merged
        // segments: 68 ms with MaxScore vs 32 ms exhaustive). Same top hits;
        // the total is exact, which the caller bounds as before.
        if self.children.len() >= FEATURE_WINDOW_MIN_CLAUSES {
            if let Some(out) = crate::query::feature::try_feature_window(&mut self.children, k, deleted, floor.get()) {
                self.primed = true;
                self.current = TERMINATED;
                return Some(out);
            }
        }
        self.max_score_top_k(k, deleted, count_limit, floor)
    }

    fn as_term_groups(&mut self) -> Option<Vec<(f32, Vec<&mut crate::query::bulk::Bm25TermScorer>)>> {
        // One level of nesting: each child must be a single group.
        if self.primed {
            return None;
        }
        let mut out = Vec::with_capacity(self.children.len());
        for c in self.children.iter_mut() {
            let mut g = c.as_term_groups()?;
            if g.len() != 1 {
                return None;
            }
            out.append(&mut g);
        }
        Some(out)
    }

    fn score_upper_bound(&self) -> Option<f32> {
        // Sum in the same clause order as score(); nonnegative bounds
        // also bound every subset of optional clauses.
        let bound = self.children.iter().try_fold(0.0f32, |sum, c| Some(sum + c.score_upper_bound()?))?;
        bound.is_finite().then_some(bound)
    }

    /// A clause whose absence leaves the others' bounds at or below the
    /// floor is required (Lucene `WANDScorer` / `MaxScore`): with all
    /// clauses required (`query_string` words once the best hits match
    /// every word) the OR walks as an AND.
    fn set_min_competitive_score(&mut self, floor: f32) {
        // Only a started union: every child then reports its position.
        if !self.primed || !floor.is_finite() || floor <= self.min_competitive || self.children.len() < 2 {
            return;
        }
        let Some(bounds) = self.children.iter().map(|c| c.score_upper_bound().filter(|b| b.is_finite() && *b >= 0.0)).collect::<Option<Vec<f32>>>() else { return };
        self.min_competitive = floor;
        let total: f64 = bounds.iter().map(|&b| b as f64).sum();
        // Slack: f32 sums of the real scores may round above the bounds' sum.
        let required: Vec<usize> = (0..bounds.len()).filter(|&i| (total - bounds[i] as f64) * (1.0 + 1e-5) < floor as f64).collect();
        if required.len() <= self.required.len() {
            return;
        }
        self.required = required;
        // Positioned on `current` already: the next advance() leaves it.
        for (i, c) in self.children.iter().enumerate() {
            self.heap.docs[i] = c.doc();
        }
    }

    fn score(&mut self) -> f32 {
        if !self.required.is_empty() {
            let d = self.current;
            return self.children.iter_mut().filter(|c| c.doc() == d).map(|c| c.score()).sum();
        }
        self.prepare_matches();
        self.matched.iter().map(|&i| self.children[i].score()).sum()
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.collect_top_k_bulk_min(k, deleted, f32::NEG_INFINITY)
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        // BM25 terms: MaxScore windows. Feature tokens (sparse_vector /
        // text_expansion clauses) whose every match must be counted: one
        // accumulate-and-collect pass per window, as Lucene's BooleanScorer.
        let out = match crate::query::bulk::try_disjunction_min(&mut self.children, k, deleted, floor) {
            Some(out) => out,
            None => crate::query::feature::try_feature_window(&mut self.children, k, deleted, floor)?,
        };
        self.primed = true;
        self.current = TERMINATED;
        Some(out)
    }
}

/// Fewest feature-token clauses a bounded-total top-k scores by window scan
/// rather than MaxScore ([`crate::query::feature::try_feature_window`]).
const FEATURE_WINDOW_MIN_CLAUSES: usize = 16;

/// Most docs (summed size hints) the pinned clauses of a [`PinnedUnion`]
/// may have.
const PINNED_DOCS: u64 = 1024;

/// An OR of one organic clause and clauses matching a handful of docs
/// (query rules' and `pinned` queries' boosted `ids`): per doc it is the
/// [`Union`] of them, but its bulk top-k scores the pinned docs on the side
/// and lets the organic clause run its own bulk path (MaxScore, block
/// OR) over the other docs, instead of driving the organic clause doc by
/// doc because a pinned clause's huge bound keeps it essential.
///
/// Exact: a pinned doc scores the sum of its matching clauses in clause
/// order (the organic one read from `probe`, a second scorer of the same
/// clause), as the union does; every other doc scores its organic score,
/// so the best `k` of those are among the organic clause's best `k`
/// outside the pinned docs. The total is the organic matches outside the
/// pinned docs plus the live pinned docs.
pub struct PinnedUnion {
    union: Union,
    organic: usize,
    probe: Box<dyn Scorer>,
    num_docs: u32,
    /// `Some(tie_breaker)`: the clauses combine as a dis_max
    /// ([`crate::query::query::DisMaxScorer`]: `max + tie * (sum - max)`,
    /// a `pinned` query), else as their sum (a bool's shoulds).
    tie: Option<f32>,
    scores: Vec<f32>,
}

impl PinnedUnion {
    /// The organic clause of `clauses` when every other one is small.
    pub fn organic_clause(clauses: &[Box<dyn Scorer>]) -> Option<usize> {
        let (big, _) = clauses.iter().enumerate().max_by_key(|(_, c)| c.size_hint())?;
        let small: u64 = clauses.iter().enumerate().filter(|(i, _)| *i != big).map(|(_, c)| c.size_hint() as u64).sum();
        (small <= PINNED_DOCS && clauses[big].size_hint() as u64 > PINNED_DOCS).then_some(big)
    }

    pub fn new(clauses: Vec<Box<dyn Scorer>>, organic: usize, probe: Box<dyn Scorer>, num_docs: u32, tie: Option<f32>) -> Self {
        Self { union: Union::new(clauses), organic, probe, num_docs, tie, scores: Vec::new() }
    }

    /// The clause scores of one doc (clause order) combined as the union
    /// (sum) or the dis_max does.
    fn combine(tie: Option<f32>, parts: &[f32]) -> f32 {
        match tie {
            None => parts.iter().sum(),
            Some(t) => {
                let max = parts.iter().copied().fold(0.0f32, f32::max);
                let sum: f32 = parts.iter().sum();
                max + t * (sum - max)
            }
        }
    }

    /// A bound of the combined score from the clauses' bounds.
    fn combine_bounds(&self, b: &[f32]) -> Option<f32> {
        let m = match self.tie {
            None => b.iter().sum(),
            Some(t) => {
                if !(t.is_finite() && t >= 0.0) {
                    return None;
                }
                let max = b.iter().copied().fold(0.0f32, f32::max);
                let sum: f32 = b.iter().sum();
                if t <= 1.0 { max + t * (sum - max) } else { max + t * sum }
            }
        };
        m.is_finite().then_some(m)
    }

    fn top_k(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: Option<&Floor<'_>>, min: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        use crate::query::collector::ScoredDoc;
        if self.union.primed {
            return None;
        }
        let n = self.union.children.len();
        let is_deleted = |d: DocId| deleted.is_some_and(|w| w.get(d as usize / 64).is_some_and(|x| (x >> (d % 64)) & 1 != 0));
        // The pinned clauses' docs, with each clause's score.
        let mut pins: Vec<(DocId, usize, f32)> = Vec::new();
        for (i, c) in self.union.children.iter_mut().enumerate() {
            if i == self.organic {
                continue;
            }
            loop {
                let d = c.advance();
                if d == TERMINATED {
                    break;
                }
                pins.push((d, i, c.score()));
            }
        }
        pins.sort_unstable_by_key(|p| (p.0, p.1));
        let words = (self.num_docs as usize).div_ceil(64);
        let mut mask: Vec<u64> = match deleted {
            Some(w) => w.to_vec(),
            None => Vec::new(),
        };
        mask.resize(words.max(mask.len()), 0);
        let mut pinned_hits: Vec<ScoredDoc> = Vec::new();
        let mut live_pins = 0u64;
        let mut parts: Vec<f32> = Vec::with_capacity(n);
        for group in pins.chunk_by(|a, b| a.0 == b.0) {
            let d = group[0].0;
            if let Some(w) = mask.get_mut(d as usize / 64) {
                *w |= 1u64 << (d % 64);
            }
            if is_deleted(d) {
                continue;
            }
            live_pins += 1;
            let organic = if self.probe.seek(d) == d { Some(self.probe.score()) } else { None };
            parts.clear();
            let mut g = group.iter().peekable();
            for i in 0..n {
                if i == self.organic {
                    parts.extend(organic);
                } else if g.peek().is_some_and(|p| p.1 == i) {
                    parts.push(g.next().unwrap().2);
                }
            }
            let score = Self::combine(self.tie, &parts);
            if score > min {
                pinned_hits.push(ScoredDoc { doc: d, score });
            }
        }
        // The organic clause over every other doc.
        let limit = if count_limit == u64::MAX { u64::MAX } else { count_limit.saturating_sub(live_pins).max(1) };
        let organic = &mut self.union.children[self.organic];
        let bulk = match floor {
            Some(f) => organic.collect_top_k_count_limited(k, Some(&mask), limit, f).or_else(|| organic.collect_top_k_bulk_min(k, Some(&mask), f.get())),
            None => organic.collect_top_k_bulk_min(k, Some(&mask), min),
        };
        let (mut hits, organic_total) = match bulk {
            Some(out) => out,
            None => {
                // No bulk path: the organic clause doc by doc.
                let mut top = crate::query::bulk::TopK::with_floor(k, min);
                let mut total = 0u64;
                loop {
                    let d = organic.advance();
                    if d == TERMINATED {
                        break;
                    }
                    if mask.get(d as usize / 64).is_some_and(|x| (x >> (d % 64)) & 1 != 0) {
                        continue;
                    }
                    total += 1;
                    top.offer(d, organic.score());
                }
                (top.into_sorted(), total)
            }
        };
        hits.extend(pinned_hits);
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        hits.truncate(k);
        self.union.primed = true;
        self.union.current = TERMINATED;
        Some((hits, organic_total + live_pins))
    }
}

impl DocSet for PinnedUnion {
    fn advance(&mut self) -> DocId {
        self.union.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.union.seek(target)
    }
    fn doc(&self) -> DocId {
        self.union.doc()
    }
    fn size_hint(&self) -> u32 {
        self.union.size_hint()
    }
}

impl Scorer for PinnedUnion {
    fn score(&mut self) -> f32 {
        if self.tie.is_none() {
            return self.union.score();
        }
        self.scores.clear();
        self.union.child_scores(&mut self.scores);
        Self::combine(self.tie, &self.scores)
    }
    fn max_score(&self) -> Option<f32> {
        let b = self.union.child_max_scores()?;
        self.combine_bounds(&b)
    }
    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        let b = self.union.child_max_scores_up_to(up_to)?;
        self.combine_bounds(&b)
    }
    fn score_upper_bound(&self) -> Option<f32> {
        match self.tie {
            None => self.union.score_upper_bound(),
            Some(_) => None,
        }
    }
    fn set_min_competitive_score(&mut self, floor: f32) {
        if self.tie.is_none() {
            self.union.set_min_competitive_score(floor);
        }
    }
    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.top_k(k, deleted, count_limit, Some(floor), floor.get())
    }
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.top_k(k, deleted, u64::MAX, None, f32::NEG_INFINITY)
    }
    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.top_k(k, deleted, u64::MAX, None, floor)
    }
}

/// Two-way disjunction without a heap: a plain merge of two sorted streams.
pub struct Disjunction2 {
    a: Box<dyn Scorer>,
    b: Box<dyn Scorer>,
    primed: bool,
    current: DocId,
    min_competitive: f32,
    required: u8,
    driver_a: bool,
}

impl Disjunction2 {
    pub fn new(a: Box<dyn Scorer>, b: Box<dyn Scorer>) -> Self {
        Self {
            a,
            b,
            primed: false,
            current: TERMINATED,
            min_competitive: f32::NEG_INFINITY,
            required: 0,
            driver_a: true,
        }
    }

    fn advance_competitive(&mut self) -> DocId {
        if self.primed && self.current == TERMINATED { return TERMINATED; }
        let (driver, other) = if self.driver_a { (&mut self.a, &mut self.b) } else { (&mut self.b, &mut self.a) };
        let d = if !self.primed { driver.advance() }
            else if driver.doc() > self.current { driver.doc() }
            else { driver.advance() };
        self.primed = true;
        self.current = Self::align_competitive(&mut **driver, &mut **other, d, self.required == 3);
        self.current
    }

    fn align_competitive(driver: &mut dyn Scorer, other: &mut dyn Scorer, mut d: DocId, both: bool) -> DocId {
        while d != TERMINATED {
            let matched = if other.doc() == d { true }
                else if other.doc() != TERMINATED && other.doc() > d { false }
                else if other.has_fast_exact_seek() { other.seek_exact(d) }
                else { other.seek(d) == d };
            if !both || matched { break; }
            d = driver.advance();
        }
        d
    }
}

impl DocSet for Disjunction2 {
    fn advance(&mut self) -> DocId {
        if self.required != 0 { return self.advance_competitive(); }
        if !self.primed {
            self.a.advance();
            self.b.advance();
            self.primed = true;
        } else if self.current == TERMINATED {
            return TERMINATED;
        } else {
            if self.a.doc() == self.current {
                self.a.advance();
            }
            if self.b.doc() == self.current {
                self.b.advance();
            }
        }
        self.current = self.a.doc().min(self.b.doc());
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        if self.required != 0 {
            let (driver, other) = if self.driver_a { (&mut self.a, &mut self.b) } else { (&mut self.b, &mut self.a) };
            let d = driver.seek(target);
            self.primed = true;
            self.current = Self::align_competitive(&mut **driver, &mut **other, d, self.required == 3);
            return self.current;
        }
        if !self.primed || self.a.doc() < target {
            self.a.seek(target);
        }
        if !self.primed || self.b.doc() < target {
            self.b.seek(target);
        }
        self.primed = true;
        self.current = self.a.doc().min(self.b.doc());
        self.current
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.a.size_hint().saturating_add(self.b.size_hint())
    }
}

impl Scorer for Disjunction2 {
    fn max_score(&self) -> Option<f32> {
        let a = self.a.max_score().filter(|b| b.is_finite() && *b >= 0.0)?;
        let b = self.b.max_score().filter(|b| b.is_finite() && *b >= 0.0)?;
        Some(a + b).filter(|s| s.is_finite())
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        let a = self.a.max_score_up_to(up_to).filter(|b| b.is_finite() && *b >= 0.0)?;
        let b = self.b.max_score_up_to(up_to).filter(|b| b.is_finite() && *b >= 0.0)?;
        Some(a + b).filter(|s| s.is_finite())
    }

    fn as_term_groups(&mut self) -> Option<Vec<(f32, Vec<&mut crate::query::bulk::Bm25TermScorer>)>> {
        if self.primed {
            return None;
        }
        let mut a = self.a.as_term_groups()?;
        let mut b = self.b.as_term_groups()?;
        if a.len() != 1 || b.len() != 1 {
            return None;
        }
        a.append(&mut b);
        Some(a)
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        let out = crate::query::bulk::try_term_groups_lim([&mut self.a, &mut self.b].into_iter(), k, deleted, count_limit, floor)?;
        self.primed = true;
        self.current = TERMINATED;
        Some(out)
    }

    fn set_min_competitive_score(&mut self, floor: f32) {
        if !floor.is_finite() || floor <= self.min_competitive { return; }
        let (Some(a), Some(b)) = (self.a.score_upper_bound(), self.b.score_upper_bound()) else { return; };
        if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 { return; }
        self.min_competitive = floor;
        // If one clause alone cannot compete, every competitive hit must
        // match the other. If neither can compete alone, OR becomes AND.
        let already_driven = self.required != 0;
        self.required = u8::from(b <= floor) | (u8::from(a <= floor) << 1);
        // Requirements only grow with the floor. Keep the existing driver:
        // the other child's failed exact seek may leave doc() at TERMINATED
        // without actually exhausting that child's future candidates.
        if already_driven { return; }
        self.driver_a = match self.required {
            1 => true,
            2 => false,
            3 => match (self.a.has_fast_exact_seek(), self.b.has_fast_exact_seek()) {
                (true, false) => false,
                (false, true) => true,
                _ => self.a.size_hint() <= self.b.size_hint(),
            },
            _ => true,
        };
    }

    fn score_upper_bound(&self) -> Option<f32> {
        let bound = self.a.score_upper_bound()? + self.b.score_upper_bound()?;
        bound.is_finite().then_some(bound)
    }

    fn score(&mut self) -> f32 {
        let d = self.current;
        let mut s = 0.0;
        if self.a.doc() == d {
            s += self.a.score();
        }
        if self.b.doc() == d {
            s += self.b.score();
        }
        s
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        self.collect_top_k_bulk_min(k, deleted, f32::NEG_INFINITY)
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        if self.primed {
            return None;
        }
        let a = self.a.as_bm25_term()? as *mut crate::query::bulk::Bm25TermScorer;
        let b = self.b.as_bm25_term()?;
        // SAFETY: `a` and `b` are distinct boxed scorers.
        let a = unsafe { &mut *a };
        if !a.is_unstarted() || !b.is_unstarted() || !a.norms().same_column(b.norms()) {
            return None;
        }
        let out = crate::query::bulk::disjunction_min(&[&*a, &*b], k, deleted, floor);
        a.advance_to_end();
        b.advance_to_end();
        self.primed = true;
        self.current = TERMINATED;
        Some(out)
    }
}

/// OR of any number of scorers, picking the cheapest implementation.
pub fn union_of(mut children: Vec<Box<dyn Scorer>>) -> Box<dyn Scorer> {
    match children.len() {
        0 => Box::new(ConstScorer::new(EmptyDocSet, 0.0)),
        1 => children.pop().unwrap(),
        2 => {
            let b = children.pop().unwrap();
            let a = children.pop().unwrap();
            Box::new(Disjunction2::new(a, b))
        }
        _ => Box::new(Union::new(children)),
    }
}

/// Docs in `inner` that are not in `excluded` (NOT).
pub struct Exclusion<I: Scorer, E: DocSet> {
    inner: I,
    excluded: E,
    excluded_primed: bool,
    current: DocId,
}

impl<I: Scorer, E: DocSet> Exclusion<I, E> {
    /// Build an exclusion of `excluded` from `inner`.
    pub fn new(inner: I, excluded: E) -> Self {
        Self {
            inner,
            excluded,
            excluded_primed: false,
            current: TERMINATED,
        }
    }

    fn step(&mut self) -> DocId {
        loop {
            let d = self.inner.doc();
            if d == TERMINATED {
                self.current = TERMINATED;
                return TERMINATED;
            }
            if !self.excluded_primed || self.excluded.doc() < d {
                self.excluded.seek(d);
                self.excluded_primed = true;
            }
            if self.excluded.doc() == d {
                self.inner.advance();
                continue;
            }
            self.current = d;
            return d;
        }
    }
}

impl<I: Scorer, E: DocSet> DocSet for Exclusion<I, E> {
    fn advance(&mut self) -> DocId {
        self.inner.advance();
        self.step()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target);
        self.step()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl<I: Scorer, E: DocSet> Scorer for Exclusion<I, E> {
    fn score(&mut self) -> f32 {
        self.inner.score()
    }
}

/// Iterates `required` and adds `optional`'s score when it lands on the same
/// doc (must + should).
pub struct RequiredOptionalScorer {
    required: Box<dyn Scorer>,
    optional: Box<dyn Scorer>,
    optional_primed: bool,
    current: DocId,
}

impl RequiredOptionalScorer {
    /// Build a must+should scorer.
    pub fn new(required: Box<dyn Scorer>, optional: Box<dyn Scorer>) -> Self {
        Self {
            required,
            optional,
            optional_primed: false,
            current: TERMINATED,
        }
    }

    fn align(&mut self) -> DocId {
        let d = self.required.doc();
        if d == TERMINATED {
            self.current = TERMINATED;
            return TERMINATED;
        }
        if !self.optional_primed || self.optional.doc() < d {
            self.optional.seek(d);
            self.optional_primed = true;
        }
        self.current = d;
        d
    }
}

impl DocSet for RequiredOptionalScorer {
    fn advance(&mut self) -> DocId {
        self.required.advance();
        self.align()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.required.seek(target);
        self.align()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.required.size_hint()
    }
}

impl Scorer for RequiredOptionalScorer {
    fn score(&mut self) -> f32 {
        let mut s = self.required.score();
        if self.optional.doc() == self.current {
            s += self.optional.score();
        }
        s
    }
}

/// Union that only yields docs matched by at least `k` children
/// (minimum-should-match).
///
/// Each step picks the k-th smallest current child doc as the candidate (no
/// smaller doc can have k children at or below it), seeks the lagging
/// children to it, and yields when at least k children land on it.
pub struct MinShouldMatchUnion {
    children: Vec<Box<dyn Scorer>>,
    docs: Vec<DocId>,
    scratch: Vec<DocId>,
    k: usize,
    primed: bool,
    current: DocId,
}

impl MinShouldMatchUnion {
    /// Build a minimum-should-match union. With `k == 0` it behaves like
    /// `k == 1`; with `k` greater than the number of children, or no
    /// children, the result is empty.
    pub fn new(children: Vec<Box<dyn Scorer>>, k: usize) -> Self {
        let n = children.len();
        let docs = children.iter().map(|c| c.doc()).collect();
        Self {
            children,
            docs,
            scratch: vec![TERMINATED; n],
            k: k.max(1),
            primed: false,
            current: TERMINATED,
        }
    }

    /// The k-th smallest current doc, or TERMINATED if fewer than k children
    /// remain.
    fn candidate(&mut self) -> DocId {
        if self.k > self.docs.len() {
            return TERMINATED;
        }
        self.scratch.copy_from_slice(&self.docs);
        let (_, kth, _) = self.scratch.select_nth_unstable(self.k - 1);
        *kth
    }

    fn step(&mut self) -> DocId {
        loop {
            let d = self.candidate();
            if d == TERMINATED {
                self.current = TERMINATED;
                return TERMINATED;
            }
            // Pull lagging children up to d.
            let mut count = 0usize;
            for (i, c) in self.children.iter_mut().enumerate() {
                if self.docs[i] < d {
                    self.docs[i] = c.seek(d);
                }
                if self.docs[i] == d {
                    count += 1;
                }
            }
            if count >= self.k {
                self.current = d;
                return d;
            }
            // Not enough children at d: move those at d past it and retry.
            for (i, c) in self.children.iter_mut().enumerate() {
                if self.docs[i] == d {
                    self.docs[i] = c.advance();
                }
            }
        }
    }
}

impl DocSet for MinShouldMatchUnion {
    fn advance(&mut self) -> DocId {
        if !self.primed {
            for (i, c) in self.children.iter_mut().enumerate() {
                self.docs[i] = c.advance();
            }
            self.primed = true;
        } else if self.current != TERMINATED {
            let d = self.current;
            for (i, c) in self.children.iter_mut().enumerate() {
                if self.docs[i] == d {
                    self.docs[i] = c.advance();
                }
            }
        } else {
            return TERMINATED;
        }
        self.step()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        for (i, c) in self.children.iter_mut().enumerate() {
            if !self.primed || self.docs[i] < target {
                self.docs[i] = c.seek(target);
            }
        }
        self.primed = true;
        self.step()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.size_hint())
            .fold(0u32, |a, b| a.saturating_add(b))
    }
}

impl Scorer for MinShouldMatchUnion {
    fn score(&mut self) -> f32 {
        let d = self.current;
        self.children
            .iter_mut()
            .filter(|c| c.doc() == d)
            .map(|c| c.score())
            .sum()
    }
}

/// `minimum_should_match` of a boolean query (ES / Solr spec), resolved
/// against the number of `should` clauses with [`MinShouldMatch::resolve`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MinShouldMatch {
    /// An absolute number of clauses (0 = default: one should is required
    /// only when there is no must/filter clause).
    Count(usize),
    /// A percentage of the should clauses, rounded down; negative values
    /// give the number that may be missing (`-25` = 75% rounded up).
    Percent(i32),
    /// A full spec string as Lucene/Solr/ES accept: `"3"`, `"-2"`, `"75%"`,
    /// `"-25%"`, `"3<90%"`, `"2<-25% 9<-3"`.
    Combination(String),
}

impl Default for MinShouldMatch {
    fn default() -> Self {
        MinShouldMatch::Count(0)
    }
}

impl From<usize> for MinShouldMatch {
    fn from(n: usize) -> Self {
        MinShouldMatch::Count(n)
    }
}

impl From<&str> for MinShouldMatch {
    fn from(s: &str) -> Self {
        MinShouldMatch::Combination(s.to_string())
    }
}

impl MinShouldMatch {
    /// The number of should clauses that must match out of `optional`
    /// (never negative; may exceed `optional`, which then matches nothing,
    /// as in ES). Unparseable specs resolve to 0.
    pub fn resolve(&self, optional: usize) -> usize {
        match self {
            MinShouldMatch::Count(n) => *n,
            MinShouldMatch::Percent(p) => percent(optional as i64, *p as i64).max(0) as usize,
            MinShouldMatch::Combination(s) => calculate_min_should_match(optional, s).unwrap_or(0),
        }
    }
}

/// `optional * percent / 100` with Lucene's float arithmetic: negative
/// results give `optional + calc` (truncated toward zero).
fn percent(optional: i64, pct: i64) -> i64 {
    let calc = ((optional * pct) as f32) * (1.0f32 / 100.0);
    if calc < 0.0 {
        optional + calc as i64
    } else {
        calc as i64
    }
}

/// Lucene / ES `Queries.calculateMinShouldMatch`: parse `spec` against
/// `optional` should clauses. `None` if the spec is malformed.
pub fn calculate_min_should_match(optional: usize, spec: &str) -> Option<usize> {
    let spec = spec.trim();
    let n = optional as i64;
    if spec.contains('<') {
        // Conditional specs: "a<x b<y" — x applies above a clauses, y above b.
        let normalised = spec.split('<').map(str::trim).collect::<Vec<_>>().join("<");
        let mut result = n;
        for part in normalised.split_whitespace() {
            let (bound, sub) = part.split_once('<')?;
            let upper: i64 = bound.parse().ok()?;
            if n <= upper {
                return Some(result.max(0) as usize);
            }
            result = calculate_min_should_match(optional, sub)? as i64;
        }
        return Some(result.max(0) as usize);
    }
    let result = if let Some(p) = spec.strip_suffix('%') {
        percent(n, p.trim().parse().ok()?)
    } else {
        let calc: i64 = spec.parse().ok()?;
        if calc < 0 { n + calc } else { calc }
    };
    Some(result.max(0) as usize)
}

/// Builds a boolean scorer tree from must/should/must_not clauses.
pub struct BooleanScorer;

impl BooleanScorer {
    /// Build the boolean tree:
    /// - must only: intersection (a single must is returned as is);
    /// - should only: union, or [`MinShouldMatchUnion`] when
    ///   `minimum_should_match` > 1;
    /// - must + should: [`RequiredOptionalScorer`] (shoulds combined as a
    ///   union);
    /// - must_not wraps the result in [`Exclusion`] (exclusions combined as
    ///   a union);
    /// - no must and no should: empty.
    pub fn build(
        must: Vec<Box<dyn Scorer>>,
        should: Vec<Box<dyn Scorer>>,
        must_not: Vec<Box<dyn DocSet>>,
        minimum_should_match: usize,
    ) -> Box<dyn Scorer> {
        let base: Box<dyn Scorer> = match (must.is_empty(), should.is_empty()) {
            (true, true) => Box::new(ConstScorer::new(EmptyDocSet, 0.0)),
            (false, true) => Self::all_of(must),
            (true, false) => {
                if minimum_should_match > 1 {
                    Box::new(MinShouldMatchUnion::new(should, minimum_should_match))
                } else {
                    union_of(should)
                }
            }
            (false, false) => {
                let optional = Self::any_of(should);
                let required = Self::all_of(must);
                Box::new(RequiredOptionalScorer::new(required, optional))
            }
        };
        if must_not.is_empty() {
            base
        } else {
            let excluded: Box<dyn DocSet> = if must_not.len() == 1 {
                must_not.into_iter().next().unwrap()
            } else {
                Box::new(Union::new(
                    must_not
                        .into_iter()
                        .map(|d| Box::new(ConstScorer::new(d, 0.0)) as Box<dyn Scorer>)
                        .collect(),
                ))
            };
            Box::new(Exclusion::new(base, excluded))
        }
    }

    fn all_of(mut v: Vec<Box<dyn Scorer>>) -> Box<dyn Scorer> {
        if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Box::new(IntersectionN::new(v))
        }
    }

    fn any_of(v: Vec<Box<dyn Scorer>>) -> Box<dyn Scorer> {
        union_of(v)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn max_score_top_k_matches_exhaustive_scoring() {
        use super::Union;
        use crate::query::docset::{DocSet, TERMINATED};
        use crate::query::scorer::Scorer;
        // A clause with per-doc scores and a (loose) max score.
        struct Var { docs: Vec<(u32, f32)>, pos: usize, bound: f32 }
        impl DocSet for Var {
            fn advance(&mut self) -> u32 { if self.pos <= self.docs.len() { self.pos += 1; } self.doc() }
            fn seek(&mut self, t: u32) -> u32 { while self.doc() < t { self.advance(); } self.doc() }
            fn doc(&self) -> u32 { if self.pos == 0 { TERMINATED } else { self.docs.get(self.pos - 1).map_or(TERMINATED, |d| d.0) } }
            fn size_hint(&self) -> u32 { self.docs.len() as u32 }
        }
        impl Scorer for Var {
            fn score(&mut self) -> f32 { self.docs[self.pos - 1].1 }
            fn max_score(&self) -> Option<f32> { Some(self.bound) }
        }
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for case in 0..40 {
            let n = 2 + case % 5;
            let num_docs = 3000u32;
            // Clause i matches about 1 doc in 2^i, with scores up to 1 + 2i:
            // frequent clauses score low, as BM25 does with idf.
            let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
            for (i, list) in lists.iter_mut().enumerate() {
                for d in 0..num_docs {
                    if rnd() % (1 << i) == 0 {
                        list.push((d, (rnd() % 1000) as f32 / 1000.0 * (1.0 + 2.0 * i as f32)));
                    }
                }
            }
            let del: Vec<u64> = (0..(num_docs as usize).div_ceil(64)).map(|_| if case % 3 == 0 { rnd() & rnd() } else { 0 }).collect();
            let deleted = |d: u32| (del[(d >> 6) as usize] >> (d & 63)) & 1 == 1;
            let make = || Union::new(lists.iter().enumerate().map(|(i, l)| {
                let bound = l.iter().map(|x| x.1).fold(0.0f32, f32::max) + if i % 2 == 0 { 0.0 } else { 0.5 };
                Box::new(Var { docs: l.clone(), pos: 0, bound }) as Box<dyn Scorer>
            }).collect());
            // Exhaustive: the per-doc union, clause-order sums.
            let mut all = make();
            let mut want: Vec<(u32, f32)> = Vec::new();
            let mut d = all.advance();
            while d != TERMINATED {
                if !deleted(d) { want.push((d, all.score())); }
                d = all.advance();
            }
            let exact_total = want.len() as u64;
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (k, limit) in [(10usize, 50u64), (1, 1), (20, 100_000), (5, 0)] {
                let (hits, total) = make().max_score_top_k(k, Some(&del), limit, &crate::query::bulk::Floor::none()).unwrap();
                let got: Vec<(u32, f32)> = hits.iter().map(|h| (h.doc, h.score)).collect();
                assert_eq!(got, want[..k.min(want.len())].to_vec(), "case {case} k {k} limit {limit}");
                assert!(total <= exact_total && total >= limit.min(exact_total), "case {case}: {total} of {exact_total}, limit {limit}");
                if limit >= exact_total { assert_eq!(total, exact_total); }
            }
            // Over a doc set (runs and scattered docs), each hit scoring a
            // constant clause more: the conjunction's per-doc scores.
            let keep: Vec<u64> = (0..(num_docs as usize).div_ceil(64)).map(|w| if w % 7 < 2 { 0 } else if w % 5 == 0 { !0 } else { rnd() | rnd() }).collect();
            let kept = |d: u32| (keep[(d >> 6) as usize] >> (d & 63)) & 1 == 1;
            for add in [1.0f32, 0.3] {
                let mut want: Vec<(u32, f32)> = Vec::new();
                let mut all = make();
                let mut d = all.advance();
                while d != TERMINATED {
                    if !deleted(d) && kept(d) { want.push((d, add + all.score())); }
                    d = all.advance();
                }
                let exact_total = want.len() as u64;
                want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for (k, limit) in [(10usize, 50u64), (1, 1), (20, 100_000), (5, 0)] {
                    let (hits, total) = make().max_score_top_k_in(k, Some(&del), Some(&keep), add, limit, &crate::query::bulk::Floor::none()).unwrap();
                    let got: Vec<(u32, f32)> = hits.iter().map(|h| (h.doc, h.score)).collect();
                    assert_eq!(got, want[..k.min(want.len())].to_vec(), "case {case} add {add} k {k} limit {limit}");
                    assert!(total <= exact_total && total >= limit.min(exact_total), "case {case}: {total} of {exact_total}, limit {limit}");
                    if limit >= exact_total { assert_eq!(total, exact_total); }
                }
            }
        }
    }

    #[test]
    fn windowed_max_score_top_k_matches_exhaustive_scoring() {
        use super::Union;
        use crate::query::docset::{DocSet, TERMINATED};
        use crate::query::query::DisMaxScorer;
        use crate::query::scorer::Scorer;
        // A clause with per-doc scores and range maxima (exact, or loose by
        // `slack`), as block-max impacts provide.
        struct Var { docs: Vec<(u32, f32)>, pos: usize, slack: f32 }
        impl DocSet for Var {
            fn advance(&mut self) -> u32 { if self.pos <= self.docs.len() { self.pos += 1; } self.doc() }
            fn seek(&mut self, t: u32) -> u32 { while self.doc() < t { self.advance(); } self.doc() }
            fn doc(&self) -> u32 { if self.pos == 0 { TERMINATED } else { self.docs.get(self.pos - 1).map_or(TERMINATED, |d| d.0) } }
            fn size_hint(&self) -> u32 { self.docs.len() as u32 }
        }
        impl Scorer for Var {
            fn score(&mut self) -> f32 { self.docs[self.pos - 1].1 }
            fn max_score(&self) -> Option<f32> { Some(self.docs.iter().map(|x| x.1).fold(0.0f32, f32::max) + self.slack) }
            fn max_score_up_to(&mut self, up_to: u32) -> Option<f32> {
                let rest = self.docs.get(self.pos.saturating_sub(1)..).unwrap_or(&[]);
                Some(rest.iter().take_while(|x| x.0 <= up_to).map(|x| x.1).fold(0.0f32, f32::max) + self.slack)
            }
        }
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for case in 0..30 {
            let n = 3 + case % 4;
            let num_docs = 20_000u32;
            // Clause i matches about 1 doc in 2^i, scoring up to 1 + 2i, with
            // a few high-scoring docs clustered in one region (so window
            // bounds differ from the global ones).
            let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n + 2];
            for (i, list) in lists.iter_mut().enumerate() {
                let i = i % n;
                for d in 0..num_docs {
                    if rnd() % (1 << i) == 0 {
                        let hot = (d / 4096) as usize % n == i;
                        let s = (rnd() % 1000) as f32 / 1000.0 * (1.0 + 2.0 * i as f32) * if hot { 3.0 } else { 1.0 };
                        list.push((d, s));
                    }
                }
            }
            let del: Vec<u64> = (0..(num_docs as usize).div_ceil(64)).map(|_| if case % 3 == 0 { rnd() & rnd() } else { 0 }).collect();
            let deleted = |d: u32| (del[(d >> 6) as usize] >> (d & 63)) & 1 == 1;
            let tie = if case % 2 == 0 { 0.0 } else { 0.4 };
            let make = || {
                let var = |i: usize| Box::new(Var { docs: lists[i].clone(), pos: 0, slack: if i % 2 == 0 { 0.0 } else { 0.25 } }) as Box<dyn Scorer>;
                let mut cs: Vec<Box<dyn Scorer>> = (0..n).map(&var).collect();
                // A dis_max over the last two lists.
                cs.push(Box::new(DisMaxScorer::new(vec![var(n), var(n + 1)], tie)));
                Union::new(cs)
            };
            let mut all = make();
            let mut want: Vec<(u32, f32)> = Vec::new();
            let mut d = all.advance();
            while d != TERMINATED {
                if !deleted(d) { want.push((d, all.score())); }
                d = all.advance();
            }
            let exact_total = want.len() as u64;
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (k, limit) in [(10usize, 50u64), (1, 0), (20, 100_000), (5, 3_000)] {
                let (hits, total) = make().max_score_top_k(k, Some(&del), limit, &crate::query::bulk::Floor::none()).unwrap();
                let got: Vec<(u32, u32)> = hits.iter().map(|h| (h.doc, h.score.to_bits())).collect();
                let exp: Vec<(u32, u32)> = want[..k.min(want.len())].iter().map(|h| (h.0, h.1.to_bits())).collect();
                assert_eq!(got, exp, "case {case} k {k} limit {limit}");
                assert!(total <= exact_total && total >= limit.min(exact_total), "case {case}: {total} of {exact_total}, limit {limit}");
                if limit >= exact_total { assert_eq!(total, exact_total); }
            }
        }
    }

    #[test]
    fn competitive_disjunction_uses_exact_membership_without_scanning() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        struct Lookup { doc: DocId, checks: Arc<AtomicUsize> }
        impl DocSet for Lookup {
            fn advance(&mut self) -> DocId { panic!("noncompetitive range must not scan") }
            fn doc(&self) -> DocId { self.doc }
            fn size_hint(&self) -> u32 { 0 }
            fn has_fast_exact_seek(&self) -> bool { true }
            fn seek_exact(&mut self, d: DocId) -> bool {
                self.checks.fetch_add(1, Ordering::Relaxed);
                let matched = d != TERMINATED && d % 7 == 0;
                self.doc = if matched { d } else { TERMINATED };
                matched
            }
        }
        for reverse in [false, true] { for seek_first in [false, true] { for floor in [2.0, 3.0] {
            let checks = Arc::new(AtomicUsize::new(0));
            let range: Box<dyn Scorer> = Box::new(ConstScorer::new(Lookup { doc: TERMINATED, checks: checks.clone() }, 2.0).with_safe_bound(true));
            let term: Box<dyn Scorer> = Box::new(ConstScorer::new(VecDocSet::new(vec![1, 7, 12, 14, 35, 100]), 3.0).with_safe_bound(true));
            let mut s = if reverse { Disjunction2::new(term, range) } else { Disjunction2::new(range, term) };
            s.set_min_competitive_score(floor);
            let mut d = if seek_first { s.seek(10) } else { s.advance() };
            let mut found = Vec::new();
            while d != TERMINATED {
                found.push((d, s.score()));
                d = s.advance();
            }
            let expected: Vec<_> = [1, 7, 12, 14, 35, 100].into_iter()
                .filter(|d| (!seek_first || *d >= 10) && (floor < 3.0 || d % 7 == 0))
                .map(|d| (d, if d % 7 == 0 { 5.0 } else { 3.0 })).collect();
            assert_eq!(found, expected);
            assert!(checks.load(Ordering::Relaxed) <= 6);
        }}}
    }

    #[test]
    fn competitive_disjunction_can_raise_floor_mid_stream() {
        let a = Box::new(ConstScorer::new(VecDocSet::new(vec![0, 3, 5]), 2.0).with_safe_bound(true));
        let b = Box::new(ConstScorer::new(VecDocSet::new(vec![0, 1, 5, 9]), 3.0).with_safe_bound(true));
        let mut s = Disjunction2::new(a, b);
        assert_eq!(s.advance(), 0);
        assert_eq!(s.score(), 5.0);
        s.set_min_competitive_score(2.0);
        assert_eq!(s.advance(), 1);
        assert_eq!(s.score(), 3.0);
        s.set_min_competitive_score(3.0);
        assert_eq!(s.advance(), 5);
        assert_eq!(s.score(), 5.0);
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.advance(), TERMINATED);
    }

    #[test]
    fn exact_lookup_intersection_visits_only_posting_candidates() {
        use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
        use crate::query::docset::{DocSet,VecDocSet,TERMINATED};
        use crate::query::scorer::{ConstScorer,Scorer};
        struct Lookup { doc:u32, checks:Arc<AtomicUsize> }
        impl DocSet for Lookup {
            fn advance(&mut self)->u32 {panic!("lookup child must not scan")}
            fn doc(&self)->u32 {self.doc}
            fn size_hint(&self)->u32 {0} // Unknown estimate must not displace the posting driver.
            fn has_fast_exact_seek(&self)->bool {true}
            fn seek_exact(&mut self,d:u32)->bool {
                self.checks.fetch_add(1,Ordering::Relaxed);
                let yes=d!=TERMINATED && d%7==0;
                self.doc=if yes {d} else {TERMINATED};yes
            }
        }
        for reverse in [false,true] {for seek_first in [false,true] {
            let checks=Arc::new(AtomicUsize::new(0));
            let mut children:Vec<Box<dyn Scorer>>=vec![
                Box::new(ConstScorer::new(Lookup{doc:TERMINATED,checks:checks.clone()},2.0)),
                Box::new(ConstScorer::new(VecDocSet::new(vec![1,7,12,14,35,100]),3.0)),
            ];
            if reverse {children.reverse();}
            let mut s=super::IntersectionN::new(children);
            let mut d=if seek_first {s.seek(10)} else {s.advance()};let mut hits=Vec::new();
            while d!=TERMINATED {hits.push(d);assert_eq!(s.score(),5.0);d=s.advance();}
            assert_eq!(hits,if seek_first {vec![14,35]} else {vec![7,14,35]});
            assert!(checks.load(Ordering::Relaxed)<=6);
        }}
    }

    use super::*;
    use crate::query::docset::VecDocSet;

    const LIMIT: u32 = 10_000;

    /// Once the floor needs every clause, the OR walks as an AND from the
    /// next doc on, with the same sums; inside an AND the floor reaches it
    /// less the other clauses' bounds.
    #[test]
    fn union_floor_requires_clauses() {
        let const_scorer = |docs: Vec<DocId>, score: f32| -> Box<dyn Scorer> { Box::new(ConstScorer::new(VecDocSet::new(docs), score).with_safe_bound(true)) };
        let mk = || Union::new(vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 1.0), const_scorer(multiples_of(5), 1.0)]);
        let mut u = mk();
        assert_eq!(u.advance(), 0);
        assert_eq!(u.advance(), 2);
        u.set_min_competitive_score(2.5);
        let mut got = Vec::new();
        loop {
            let d = u.advance();
            if d == TERMINATED || d > 200 { break; }
            got.push((d, u.score()));
        }
        let want: Vec<(DocId, f32)> = (3..=200).filter(|d| d % 30 == 0).map(|d| (d, 3.0)).collect();
        assert_eq!(got, want);
        // Two clauses needed: docs matching at least two of three (bounds 1, 1, 0.5).
        let mut u = Union::new(vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 1.0), const_scorer(multiples_of(5), 0.5)]);
        u.advance();
        u.set_min_competitive_score(1.6);
        let mut got = Vec::new();
        loop {
            let d = u.advance();
            if d == TERMINATED || d > 60 { break; }
            got.push((d, u.score()));
        }
        let want: Vec<(DocId, f32)> = (1..=60).filter(|d| d % 6 == 0).map(|d| (d, if d % 5 == 0 { 2.5 } else { 2.0 })).collect();
        assert_eq!(got, want);
        // Inside an AND with a 1.0 clause: floor 3.5 leaves 2.5 for the OR.
        let mut i = IntersectionN::new(vec![const_scorer((0..400).collect(), 1.0), Box::new(mk())]);
        assert_eq!(i.advance(), 0);
        i.set_min_competitive_score(3.5);
        let mut got = Vec::new();
        loop {
            let d = i.advance();
            if d == TERMINATED { break; }
            got.push((d, i.score()));
        }
        let want: Vec<(DocId, f32)> = (1..400).filter(|d| d % 30 == 0).map(|d| (d, 4.0)).collect();
        assert_eq!(got, want);
    }

    fn multiples_of(n: u32) -> Vec<DocId> {
        (0..LIMIT).step_by(n as usize).collect()
    }

    fn const_scorer(docs: Vec<DocId>, score: f32) -> Box<dyn Scorer> {
        Box::new(ConstScorer::new(VecDocSet::new(docs), score))
    }

    /// Drain a scorer from the start, returning (doc, score) pairs.
    fn drain(s: &mut dyn Scorer) -> Vec<(DocId, f32)> {
        let mut out = Vec::new();
        loop {
            let d = s.advance();
            if d == TERMINATED {
                break;
            }
            out.push((d, s.score()));
        }
        out
    }

    /// Seek to `target` first, then drain.
    fn drain_from(s: &mut dyn Scorer, target: DocId) -> Vec<(DocId, f32)> {
        let mut out = Vec::new();
        let d = s.seek(target);
        if d == TERMINATED {
            return out;
        }
        out.push((d, s.score()));
        out.extend(drain(s));
        out
    }

    fn docs_of(v: &[(DocId, f32)]) -> Vec<DocId> {
        v.iter().map(|(d, _)| *d).collect()
    }

    /// Expected score for the standard 2/3/5 test fixture (weights 1/2/4).
    fn fixture_score(d: DocId) -> f32 {
        let mut s = 0.0f32;
        if d % 2 == 0 {
            s += 1.0;
        }
        if d % 3 == 0 {
            s += 2.0;
        }
        if d % 5 == 0 {
            s += 4.0;
        }
        s
    }

    fn brute(pred: impl Fn(DocId) -> bool) -> Vec<DocId> {
        (0..LIMIT).filter(|&d| pred(d)).collect()
    }

    /// Check advance-from-start, seek-then-drain at several targets, and
    /// interleaved seek/advance against `want`, with scores from `score_of`.
    fn check_all(
        mk: impl Fn() -> Box<dyn Scorer>,
        want: &[DocId],
        score_of: impl Fn(DocId) -> f32,
    ) {
        let got = drain(&mut *mk());
        assert_eq!(docs_of(&got), want, "advance");
        for (d, sc) in &got {
            assert!((sc - score_of(*d)).abs() < 1e-6, "score doc {d}");
        }
        for target in [0u32, 1, 7, 30, 31, 100, 4999, 9999, 10_000, 20_000] {
            let got = drain_from(&mut *mk(), target);
            let w: Vec<DocId> = want.iter().copied().filter(|&d| d >= target).collect();
            assert_eq!(docs_of(&got), w, "seek {target}");
            for (d, sc) in &got {
                assert!((sc - score_of(*d)).abs() < 1e-6, "score doc {d} after seek {target}");
            }
        }
        // Interleaved: seek, then a few advances, repeat.
        let mut s = mk();
        let mut pos = 0usize;
        for target in [0u32, 3, 10, 50, 51, 200, 201, 2999, 9998, 9999] {
            while pos < want.len() && want[pos] < target {
                pos += 1;
            }
            let d = s.seek(target);
            let w = want.get(pos).copied().unwrap_or(TERMINATED);
            assert_eq!(d, w, "interleaved seek {target}");
            if d == TERMINATED {
                break;
            }
            assert!((s.score() - score_of(d)).abs() < 1e-6);
            for _ in 0..3 {
                pos += 1;
                let d = s.advance();
                let w = want.get(pos).copied().unwrap_or(TERMINATED);
                assert_eq!(d, w, "interleaved advance after seek {target}");
                if d == TERMINATED {
                    break;
                }
                assert!((s.score() - score_of(d)).abs() < 1e-6);
            }
            if pos >= want.len() {
                break;
            }
        }
    }

    #[test]
    fn intersection_two() {
        let want = brute(|d| d % 2 == 0 && d % 3 == 0);
        check_all(
            || {
                Box::new(Intersection::new(
                    ConstScorer::new(VecDocSet::new(multiples_of(2)), 1.0),
                    ConstScorer::new(VecDocSet::new(multiples_of(3)), 2.0),
                ))
            },
            &want,
            |_| 3.0,
        );
        // Empty side.
        let mut s = Intersection::new(VecDocSet::new(vec![]), VecDocSet::new(multiples_of(3)));
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.seek(5), TERMINATED);
    }

    #[test]
    fn intersection_n() {
        let want = brute(|d| d % 2 == 0 && d % 3 == 0 && d % 5 == 0);
        check_all(
            || {
                Box::new(IntersectionN::new(vec![
                    const_scorer(multiples_of(2), 1.0),
                    const_scorer(multiples_of(3), 2.0),
                    const_scorer(multiples_of(5), 4.0),
                ]))
            },
            &want,
            |_| 7.0,
        );
        // Single child.
        check_all(
            || Box::new(IntersectionN::new(vec![const_scorer(multiples_of(7), 1.0)])),
            &multiples_of(7),
            |_| 1.0,
        );
        // No children.
        let mut s = IntersectionN::new(Vec::new());
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.seek(3), TERMINATED);
    }

    #[test]
    fn disjunction2_matches_union() {
        let mk = || vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0)];
        let mut u = Union::new(mk());
        let mut d = Disjunction2::new(const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0));
        assert_eq!(drain(&mut u), drain(&mut d));
        let mut d = Disjunction2::new(const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0));
        assert_eq!(d.seek(7), 8);
        assert_eq!(d.score(), 1.0);
        assert_eq!(d.seek(9), 9);
        assert_eq!(d.score(), 2.0);
        assert_eq!(d.advance(), 10);
        assert_eq!(d.seek(12), 12);
        assert_eq!(d.score(), 3.0);
        assert_eq!(d.seek(LIMIT + 5), TERMINATED);
    }

    #[test]
    fn union_many_sparse_children_preserves_clause_order_and_seek() {
        let specs: Vec<_> = (0..513).map(|i| {
            let docs = if i == 0 { (0..1200).collect() } else { vec![i as u32, 1199] };
            let score = match i % 3 { 0 => 1.0e8, 1 => 1.0, _ => 3.25 };
            (docs, score)
        }).collect();
        let make = || Union::new(specs.iter().map(|(docs, score)| const_scorer(docs.clone(), *score)).collect());
        for seek in [false, true] {
            let mut union = make();
            for doc in (0..1200).filter(|d| !seek || d % 7 == 0 || *d == 1199) {
                assert_eq!(if seek { union.seek(doc) } else { union.advance() }, doc);
                let expected: Vec<f32> = specs.iter().filter(|(docs, _)| docs.contains(&doc)).map(|(_, score)| *score).collect();
                let sum: f32 = expected.iter().copied().sum();
                assert_eq!(union.score().to_bits(), sum.to_bits());
                let mut scores = Vec::new();
                union.child_scores(&mut scores);
                assert_eq!(scores, expected);
                assert_eq!(union.score().to_bits(), sum.to_bits());
            }
            assert_eq!(union.advance(), TERMINATED);
        }
    }

    #[test]
    fn union_three() {
        let want = brute(|d| d % 2 == 0 || d % 3 == 0 || d % 5 == 0);
        check_all(
            || {
                Box::new(Union::new(vec![
                    const_scorer(multiples_of(2), 1.0),
                    const_scorer(multiples_of(3), 2.0),
                    const_scorer(multiples_of(5), 4.0),
                ]))
            },
            &want,
            fixture_score,
        );
    }

    #[test]
    fn union_edge_cases() {
        let mut s = Union::new(Vec::new());
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.doc(), TERMINATED);
        assert_eq!(s.seek(10), TERMINATED);
        assert_eq!(s.size_hint(), 0);

        check_all(
            || Box::new(Union::new(vec![const_scorer(multiples_of(7), 3.0)])),
            &multiples_of(7),
            |_| 3.0,
        );

        // One empty child among others.
        let want = multiples_of(4);
        check_all(
            || {
                Box::new(Union::new(vec![
                    const_scorer(Vec::new(), 9.0),
                    const_scorer(multiples_of(4), 1.0),
                ]))
            },
            &want,
            |_| 1.0,
        );
    }

    #[test]
    fn exclusion() {
        let want = brute(|d| d % 2 == 0 && d % 3 != 0);
        check_all(
            || {
                Box::new(Exclusion::new(
                    ConstScorer::new(VecDocSet::new(multiples_of(2)), 1.5),
                    VecDocSet::new(multiples_of(3)),
                ))
            },
            &want,
            |_| 1.5,
        );
        // Exclude everything.
        let mut s = Exclusion::new(
            ConstScorer::new(VecDocSet::new(multiples_of(6)), 1.0),
            VecDocSet::new(multiples_of(3)),
        );
        assert_eq!(s.advance(), TERMINATED);
        // Exclude nothing.
        check_all(
            || {
                Box::new(Exclusion::new(
                    ConstScorer::new(VecDocSet::new(multiples_of(2)), 1.0),
                    VecDocSet::new(Vec::new()),
                ))
            },
            &multiples_of(2),
            |_| 1.0,
        );
    }

    #[test]
    fn required_optional() {
        check_all(
            || {
                Box::new(RequiredOptionalScorer::new(
                    const_scorer(multiples_of(2), 1.0),
                    const_scorer(multiples_of(3), 2.0),
                ))
            },
            &multiples_of(2),
            |d| if d % 3 == 0 { 3.0 } else { 1.0 },
        );
        // Optional exhausts early.
        check_all(
            || {
                Box::new(RequiredOptionalScorer::new(
                    const_scorer(multiples_of(2), 1.0),
                    const_scorer(vec![0, 4, 10], 2.0),
                ))
            },
            &multiples_of(2),
            |d| if d == 0 || d == 4 || d == 10 { 3.0 } else { 1.0 },
        );
    }

    #[test]
    fn min_should_match() {
        for k in [1usize, 2, 3] {
            let want = brute(|d| {
                [(d % 2 == 0), (d % 3 == 0), (d % 5 == 0)]
                    .iter()
                    .filter(|&&m| m)
                    .count()
                    >= k
            });
            check_all(
                || {
                    Box::new(MinShouldMatchUnion::new(
                        vec![
                            const_scorer(multiples_of(2), 1.0),
                            const_scorer(multiples_of(3), 2.0),
                            const_scorer(multiples_of(5), 4.0),
                        ],
                        k,
                    ))
                },
                &want,
                fixture_score,
            );
        }
        // k > children: empty.
        let mut s = MinShouldMatchUnion::new(vec![const_scorer(multiples_of(2), 1.0)], 2);
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.seek(4), TERMINATED);
        // No children: empty.
        let mut s = MinShouldMatchUnion::new(Vec::new(), 1);
        assert_eq!(s.advance(), TERMINATED);
        // k = 0 behaves as 1.
        check_all(
            || {
                Box::new(MinShouldMatchUnion::new(
                    vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0)],
                    0,
                ))
            },
            &brute(|d| d % 2 == 0 || d % 3 == 0),
            |d| fixture_score(d) - if d % 5 == 0 { 4.0 } else { 0.0 },
        );
    }

    #[test]
    fn min_should_match_spec_table() {
        // (optional clauses, spec, expected) — Solr's
        // TestSolrPluginUtils / ES QueriesTests examples.
        let table: &[(usize, &str, usize)] = &[
            (2, "0", 0),
            (1, "1", 1),
            (2, "1", 1),
            (3, "1", 1),
            (3, "2", 2),
            (5, "3", 3),
            (3, "-1", 2),
            (5, "-2", 3),
            (1, "-1", 0),
            (2, "-3", 0),
            (1, "100%", 1),
            (2, "100%", 2),
            (3, "75%", 2),
            (4, "75%", 3),
            (5, "75%", 3),
            (10, "75%", 7),
            (4, "-25%", 3),
            (5, "-25%", 4),
            (10, "-25%", 8),
            (3, "50%", 1),
            (1, "50%", 0),
            (1, "3<90%", 1),
            (3, "3<90%", 3),
            (4, "3<90%", 3),
            (10, "3<90%", 9),
            (1, "3<-25%", 1),
            (3, "3<-25%", 3),
            (4, "3<-25%", 3),
            (5, "3<-25%", 4),
            (1, "2<-25% 9<-3", 1),
            (2, "2<-25% 9<-3", 2),
            (3, "2<-25% 9<-3", 3),
            (8, "2<-25% 9<-3", 6),
            (9, "2<-25% 9<-3", 7),
            (10, "2<-25% 9<-3", 7),
            (20, "2<-25% 9<-3", 17),
            (10, " 2 < -25%  9 < -3 ", 7),
            (4, "2<2", 2),
            (5, "5", 5),
            (2, "5", 5),
        ];
        for &(n, spec, want) in table {
            assert_eq!(calculate_min_should_match(n, spec), Some(want), "calc({n}, {spec:?})");
            assert_eq!(MinShouldMatch::from(spec).resolve(n), want);
        }
        assert_eq!(calculate_min_should_match(3, "x%"), None);
        assert_eq!(MinShouldMatch::Combination("bogus".into()).resolve(3), 0);
        assert_eq!(MinShouldMatch::Count(2).resolve(9), 2);
        assert_eq!(MinShouldMatch::Percent(75).resolve(4), 3);
        assert_eq!(MinShouldMatch::Percent(-25).resolve(5), 4);
    }

    #[test]
    fn boolean_must_only() {
        check_all(
            || {
                BooleanScorer::build(
                    vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0)],
                    Vec::new(),
                    Vec::new(),
                    0,
                )
            },
            &brute(|d| d % 6 == 0),
            |_| 3.0,
        );
    }

    #[test]
    fn boolean_should_only() {
        check_all(
            || {
                BooleanScorer::build(
                    Vec::new(),
                    vec![const_scorer(multiples_of(2), 1.0), const_scorer(multiples_of(3), 2.0)],
                    Vec::new(),
                    1,
                )
            },
            &brute(|d| d % 2 == 0 || d % 3 == 0),
            |d| fixture_score(d) - if d % 5 == 0 { 4.0 } else { 0.0 },
        );
        check_all(
            || {
                BooleanScorer::build(
                    Vec::new(),
                    vec![
                        const_scorer(multiples_of(2), 1.0),
                        const_scorer(multiples_of(3), 2.0),
                        const_scorer(multiples_of(5), 4.0),
                    ],
                    Vec::new(),
                    2,
                )
            },
            &brute(|d| {
                [(d % 2 == 0), (d % 3 == 0), (d % 5 == 0)]
                    .iter()
                    .filter(|&&m| m)
                    .count()
                    >= 2
            }),
            fixture_score,
        );
    }

    #[test]
    fn boolean_must_and_should() {
        check_all(
            || {
                BooleanScorer::build(
                    vec![const_scorer(multiples_of(2), 1.0)],
                    vec![const_scorer(multiples_of(3), 2.0), const_scorer(multiples_of(5), 4.0)],
                    Vec::new(),
                    0,
                )
            },
            &multiples_of(2),
            fixture_score,
        );
    }

    #[test]
    fn boolean_must_not() {
        check_all(
            || {
                BooleanScorer::build(
                    vec![const_scorer(multiples_of(2), 1.0)],
                    Vec::new(),
                    vec![Box::new(VecDocSet::new(multiples_of(3)))],
                    0,
                )
            },
            &brute(|d| d % 2 == 0 && d % 3 != 0),
            |_| 1.0,
        );
        check_all(
            || {
                BooleanScorer::build(
                    vec![const_scorer(multiples_of(2), 1.0)],
                    Vec::new(),
                    vec![
                        Box::new(VecDocSet::new(multiples_of(3))),
                        Box::new(VecDocSet::new(multiples_of(5))),
                    ],
                    0,
                )
            },
            &brute(|d| d % 2 == 0 && d % 3 != 0 && d % 5 != 0),
            |_| 1.0,
        );
    }

    #[test]
    fn boolean_full_tree_and_empty() {
        check_all(
            || {
                BooleanScorer::build(
                    vec![const_scorer(multiples_of(2), 1.0)],
                    vec![const_scorer(multiples_of(3), 2.0)],
                    vec![Box::new(VecDocSet::new(multiples_of(7)))],
                    0,
                )
            },
            &brute(|d| d % 2 == 0 && d % 7 != 0),
            |d| if d % 3 == 0 { 3.0 } else { 1.0 },
        );
        let mut s = BooleanScorer::build(Vec::new(), Vec::new(), Vec::new(), 0);
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.seek(5), TERMINATED);
        let mut s = BooleanScorer::build(
            Vec::new(),
            Vec::new(),
            vec![Box::new(VecDocSet::new(multiples_of(2)))],
            0,
        );
        assert_eq!(s.advance(), TERMINATED);
    }
}

#[cfg(test)]
mod pinned_union_tests {
    use super::*;
    use crate::query::collector::collect_top_k;
    use crate::query::docset::VecDocSet;

    fn cs(docs: Vec<DocId>, score: f32) -> Box<dyn Scorer> {
        Box::new(ConstScorer::new(VecDocSet::new(docs), score))
    }

    fn organic() -> Box<dyn Scorer> {
        let every = |n: u32, s: f32| cs((0..20_000).step_by(n as usize).collect(), s);
        Box::new(Union::new(vec![every(2, 0.7), every(3, 1.3), every(7, 2.1), every(11, 0.45)]))
    }

    fn clauses(pinned_first: bool) -> Vec<Box<dyn Scorer>> {
        let pins = vec![cs(vec![5, 1234, 9999, 19_998], 2.0e7), cs(vec![17, 1234, 30_000], 1.0e7)];
        if pinned_first {
            pins.into_iter().chain([organic()]).collect()
        } else {
            [organic()].into_iter().chain(pins).collect()
        }
    }

    #[test]
    fn pinned_union_bulk_matches_the_per_doc_union() {
        let mut del = vec![0u64; 20_000usize.div_ceil(64) + 1];
        for d in [6u32, 9999, 42, 1000] {
            del[d as usize / 64] |= 1 << (d % 64);
        }
        for (pinned_first, tie) in [(true, None), (false, None), (true, Some(0.0)), (false, Some(0.0)), (true, Some(0.3))] {
            for deleted in [None, Some(del.as_slice())] {
                for k in [1usize, 3, 20, 500] {
                    let accept = |d: DocId| deleted.is_none_or(|w| w.get(d as usize / 64).is_none_or(|x| (x >> (d % 64)) & 1 == 0));
                    let mut per_doc: Box<dyn Scorer> = match tie {
                        None => Box::new(Union::new(clauses(pinned_first))),
                        Some(t) => Box::new(crate::query::query::DisMaxScorer::new(clauses(pinned_first), t)),
                    };
                    let want = collect_top_k(&mut *per_doc, k, accept);
                    let c = clauses(pinned_first);
                    let organic_at = PinnedUnion::organic_clause(&c).expect("pinned shape");
                    let mut p = PinnedUnion::new(c, organic_at, organic(), 20_000, tie);
                    let got = p.collect_top_k_count_limited(k, deleted, u64::MAX, &Floor::none()).expect("bulk");
                    assert_eq!(want.1, got.1, "total");
                    let bits = |v: &[crate::query::collector::ScoredDoc]| v.iter().map(|h| (h.doc, h.score.to_bits())).collect::<Vec<_>>();
                    assert_eq!(bits(&want.0), bits(&got.0), "pinned_first {pinned_first}, tie {tie:?}, k {k}");
                }
            }
        }
    }

    #[test]
    fn organic_clause_needs_one_large_clause() {
        assert!(PinnedUnion::organic_clause(&[cs(vec![1, 2], 1.0), cs(vec![3], 1.0)]).is_none());
        assert!(PinnedUnion::organic_clause(&[organic(), organic()]).is_none());
        assert_eq!(PinnedUnion::organic_clause(&clauses(true)), Some(2));
    }
}
