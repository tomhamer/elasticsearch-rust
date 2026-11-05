//! Scorers: doc sets that also produce a per-doc score.

use crate::DocId;
use crate::query::bulk::{Bm25TermScorer, Floor};
use crate::query::collector::ScoredDoc;
use crate::query::docset::DocSet;
use crate::query::segment_scorer::Norms;
use crate::query::similarity::SimWeight;
use std::sync::Arc;

/// A [`DocSet`] that also produces a score for the current doc.
pub trait Scorer: DocSet {
    /// Score of the current doc (valid after a successful `advance`/`seek`).
    fn score(&mut self) -> f32;

    /// Finite, nonnegative upper bound for every score this scorer can emit.
    /// Opting in also promises that skipping its remaining documents cannot
    /// suppress observable evaluation errors. Unknown/custom scorers decline.
    fn score_upper_bound(&self) -> Option<f32> { None }

    /// After hit counting is satisfied, omit future documents scoring at or
    /// below this monotonically increasing floor. The caller has already kept
    /// enough earlier documents, so equal-score later doc IDs cannot win ties.
    /// Implementations may ignore the hint; skipping must not hide errors.
    fn set_min_competitive_score(&mut self, _floor: f32) {}

    /// An upper bound for every score this scorer can emit (Lucene
    /// `Scorer.getMaxScore(NO_MORE_DOCS)`), used to skip clauses that cannot
    /// make a document competitive. Unlike [`Scorer::score_upper_bound`] it
    /// need not be reachable; `None` when unknown.
    fn max_score(&self) -> Option<f32> { self.score_upper_bound() }

    /// An upper bound for the scores of the docs from the current one (the
    /// start, when not yet advanced) up to `up_to` inclusive (Lucene
    /// `advanceShallow` + `getMaxScore(upTo)`), e.g. from the block-max
    /// impacts of the postings blocks in that range. Defaults to
    /// [`Scorer::max_score`].
    fn max_score_up_to(&mut self, _up_to: DocId) -> Option<f32> { self.max_score() }

    /// Top `k` hits (score desc, doc asc) of the whole, not yet advanced,
    /// scorer when the total only has to be exact up to `count_limit`
    /// (`track_total_hits`). Once `count_limit` live matches are counted,
    /// documents that cannot beat the k-th best score may be skipped, and
    /// the returned total is then a lower bound of at least `count_limit`.
    /// Only hits scoring strictly above `floor` matter (other leaves hold `k`
    /// hits that good); like [`Scorer::collect_top_k_bulk_min`], returned
    /// hits may still include worse ones. `None`: not supported here.
    fn collect_top_k_count_limited(&mut self, _k: usize, _deleted: Option<&[u64]>, _count_limit: u64, _floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
        None
    }

    /// [`Scorer::collect_top_k_count_limited`] over the docs set in `keep`
    /// only, each hit scoring `add` more: a conjunction of this scorer and
    /// a constant-scored doc set (see [`Scorer::const_filter`]). `None`
    /// when this scorer has no such path; nothing is consumed then.
    /// Whether [`Scorer::collect_top_k_filtered`] may take its path (a
    /// cheap check, before a conjunction builds the doc set's words).
    fn can_collect_filtered(&mut self) -> bool {
        false
    }

    fn collect_top_k_filtered(&mut self, _k: usize, _deleted: Option<&[u64]>, _keep: &[u64], _add: f32, _count_limit: u64, _floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
        None
    }

    /// An unstarted scorer that matches exactly the docs set in the
    /// returned words, each scoring the returned constant (a scored range
    /// or a cached filter bitset). `None` otherwise.
    fn const_filter(&mut self) -> Option<(std::sync::Arc<crate::admission::Allocation<Vec<u64>>>, f32)> {
        None
    }

    /// Bulk entry point (Lucene `BulkScorer`): collect the top `k` hits
    /// (score desc, doc asc) and the exact number of live matches of the
    /// whole, not yet advanced, scorer. `deleted` is the segment's deletion
    /// bitset words (bit set = deleted). `None` means "not supported here":
    /// the caller falls back to the per-doc loop.
    fn collect_top_k_bulk(&mut self, _k: usize, _deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        None
    }

    /// [`Scorer::collect_top_k_bulk`] for a caller to whom only hits scoring
    /// strictly above `min_competitive` matter (earlier leaves already hold
    /// `k` hits at least that good; Lucene's `MaxScoreAccumulator`). Blocks
    /// that cannot beat it may be skipped; the count stays exact. Returned
    /// hits may still include non-competitive ones.
    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, _min_competitive: f32) -> Option<(Vec<ScoredDoc>, u64)> {
        self.collect_top_k_bulk(k, deleted)
    }

    /// The concrete BM25 term scorer behind this scorer, if it is one
    /// (lets AND/OR bulk scorers drive term postings block by block).
    fn as_bm25_term(&mut self) -> Option<&mut Bm25TermScorer> {
        None
    }

    /// The concrete feature-token scorer behind this scorer, if it is one
    /// (lets the windowed feature OR read its postings block by block).
    fn as_feature_term(&mut self) -> Option<&mut crate::query::feature::FeatureTermScorer> {
        None
    }

    /// Append every remaining live doc (see [`DocSet::fill_all`]) to `out`
    /// and return the score every one of them has, for collectors that
    /// keep every match (aggregations); `None`: not a constant-score set
    /// that can do so, nothing consumed.
    fn collect_all(&mut self, _deleted: Option<&[u64]>, _out: &mut Vec<DocId>) -> Option<f32> {
        None
    }

    /// This unstarted scorer as a sum of dis_max groups of BM25 term
    /// scorers, `(tie_breaker, terms)` each scoring `max + tie * (sum - max)`
    /// of its terms' scores, summed in order exactly as [`Scorer::score`]
    /// sums them (a term is one group of one; a dis_max of terms one group;
    /// a disjunction of those its clauses' groups). Lets the block OR
    /// ([`crate::query::bulk::or_groups`]) drive multi-field disjunctions.
    fn as_term_groups(&mut self) -> Option<Vec<(f32, Vec<&mut Bm25TermScorer>)>> {
        self.as_bm25_term().map(|t| vec![(0.0, vec![t])])
    }
}

/// A scorer that returns a constant score for every doc in `inner`.
pub struct ConstScorer<D: DocSet> {
    inner: D,
    score: f32,
    safe_bound: bool,
}

impl<D: DocSet> ConstScorer<D> {
    /// Wrap `inner` with a constant score.
    pub fn new(inner: D, score: f32) -> Self {
        Self { inner, score, safe_bound: false }
    }

    /// Only opt in after the wrapped scorer has promised that skipping the
    /// remaining documents cannot suppress evaluation errors.
    pub(crate) fn with_safe_bound(mut self, safe: bool) -> Self {
        self.safe_bound = safe;
        self
    }
}

impl<D: DocSet> DocSet for ConstScorer<D> {
    fn has_fast_exact_seek(&self) -> bool { self.inner.has_fast_exact_seek() }

    fn exact_seek_reads_value(&self) -> bool { self.inner.exact_seek_reads_value() }

    fn set_lead_cost(&mut self, lead_cost: u32) { self.inner.set_lead_cost(lead_cost) }

    fn seek_exact(&mut self, target: DocId) -> bool { self.inner.seek_exact(target) }

    fn advance(&mut self) -> DocId {
        self.inner.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target)
    }

    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }

    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        self.inner.fill_all(deleted, out)
    }
}

impl<D: DocSet> Scorer for ConstScorer<D> {
    fn collect_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> Option<f32> {
        self.inner.fill_all(deleted, out).then_some(self.score)
    }

    fn score_upper_bound(&self) -> Option<f32> {
        (self.safe_bound && self.score.is_finite() && self.score >= 0.0).then_some(self.score)
    }

    fn score(&mut self) -> f32 {
        self.score
    }

    fn const_filter(&mut self) -> Option<(std::sync::Arc<crate::admission::Allocation<Vec<u64>>>, f32)> {
        Some((self.inner.filter_words()?, self.score))
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        // Uniform score: the best hits are the first k live docs.
        let (docs, n) = self.inner.collect_first_k_bulk(k, deleted)?;
        Some((docs.into_iter().map(|doc| ScoredDoc { doc, score: self.score }).collect(), n))
    }
}

/// Supplies the term frequency of the current doc for a [`TermScorer`].
///
/// The parent plugs in its postings reader: after the scorer's `advance`/
/// `seek` lands on a doc, `term_freq` returns that doc's frequency for the
/// scored term.
pub trait TermFreqSource {
    /// Term frequency of the current doc.
    fn term_freq(&self) -> u32;
}

/// A term scorer over a doc set that also knows each doc's term frequency,
/// scoring through a [`SimWeight`] (any [`Similarity`](crate::query::similarity::Similarity)).
///
/// `W` defaults to a boxed weight; the BM25 fast path instantiates it with
/// the concrete [`Bm25Weight`](crate::query::bm25::Bm25Weight) so the hot
/// loop is statically dispatched.
///
/// Field lengths (norms) are looked up through a closure, so the same scorer
/// can be used with any norms column.
pub struct TermScorer<D: DocSet + TermFreqSource, W: SimWeight = Box<dyn SimWeight>> {
    inner: D,
    weight: W,
    norms: NormSource,
}

enum NormSource {
    /// Field length per doc through a closure.
    Len(Box<dyn Fn(DocId) -> u32 + Send + Sync>),
    /// One norm byte per doc ([`SimWeight::score_norm`]).
    Bytes(Arc<Norms>),
}

impl<D: DocSet + TermFreqSource, W: SimWeight> TermScorer<D, W> {
    /// Build a term scorer. `field_len` maps a doc id to its field length
    /// (norm) for the scored field.
    pub fn new(inner: D, weight: W, field_len: Box<dyn Fn(DocId) -> u32 + Send + Sync>) -> Self {
        Self {
            inner,
            weight,
            norms: NormSource::Len(field_len),
        }
    }

    /// Build a term scorer over a field's 1-byte norms.
    pub fn with_norms(inner: D, weight: W, norms: Arc<Norms>) -> Self {
        Self {
            inner,
            weight,
            norms: NormSource::Bytes(norms),
        }
    }
}

impl<D: DocSet + TermFreqSource, W: SimWeight> DocSet for TermScorer<D, W> {
    fn advance(&mut self) -> DocId {
        self.inner.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target)
    }

    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl<D: DocSet + TermFreqSource, W: SimWeight> Scorer for TermScorer<D, W> {
    fn score(&mut self) -> f32 {
        let d = self.inner.doc();
        match &self.norms {
            NormSource::Len(f) => self.weight.score(self.inner.term_freq(), f(d)),
            NormSource::Bytes(n) => self.weight.score_norm(self.inner.term_freq(), n.byte(d)),
        }
    }
}

impl Scorer for Box<dyn Scorer> {
    fn collect_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> Option<f32> { (**self).collect_all(deleted, out) }
    fn score_upper_bound(&self) -> Option<f32> { (**self).score_upper_bound() }
    fn set_min_competitive_score(&mut self, floor: f32) { (**self).set_min_competitive_score(floor) }
    fn max_score(&self) -> Option<f32> { (**self).max_score() }
    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> { (**self).max_score_up_to(up_to) }
    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
        (**self).collect_top_k_count_limited(k, deleted, count_limit, floor)
    }
    fn collect_top_k_filtered(&mut self, k: usize, deleted: Option<&[u64]>, keep: &[u64], add: f32, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
        (**self).collect_top_k_filtered(k, deleted, keep, add, count_limit, floor)
    }
    fn can_collect_filtered(&mut self) -> bool {
        (**self).can_collect_filtered()
    }
    fn const_filter(&mut self) -> Option<(std::sync::Arc<crate::admission::Allocation<Vec<u64>>>, f32)> {
        (**self).const_filter()
    }
    #[inline]
    fn score(&mut self) -> f32 {
        (**self).score()
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        (**self).collect_top_k_bulk(k, deleted)
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, min_competitive: f32) -> Option<(Vec<ScoredDoc>, u64)> {
        (**self).collect_top_k_bulk_min(k, deleted, min_competitive)
    }

    fn as_bm25_term(&mut self) -> Option<&mut Bm25TermScorer> {
        (**self).as_bm25_term()
    }

    fn as_feature_term(&mut self) -> Option<&mut crate::query::feature::FeatureTermScorer> {
        (**self).as_feature_term()
    }

    fn as_term_groups(&mut self) -> Option<Vec<(f32, Vec<&mut Bm25TermScorer>)>> {
        (**self).as_term_groups()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::bm25::{idf, Bm25Params, Bm25Weight};
    use crate::query::docset::VecDocSet;

    /// Test doc set that also reports a fixed term frequency.
    struct TfDocSet {
        inner: VecDocSet,
        tf: u32,
    }

    impl DocSet for TfDocSet {
        fn advance(&mut self) -> DocId {
            self.inner.advance()
        }
        fn seek(&mut self, target: DocId) -> DocId {
            self.inner.seek(target)
        }
        fn doc(&self) -> DocId {
            self.inner.doc()
        }
        fn size_hint(&self) -> u32 {
            self.inner.size_hint()
        }
    }

    impl TermFreqSource for TfDocSet {
        fn term_freq(&self) -> u32 {
            self.tf
        }
    }

    #[test]
    fn const_scorer() {
        let docs: Vec<DocId> = (0..10).step_by(2).collect();
        let mut s = ConstScorer::new(VecDocSet::new(docs), 2.5);
        assert_eq!(s.advance(), 0);
        assert_eq!(s.score(), 2.5);
        assert_eq!(s.seek(5), 6);
        assert_eq!(s.score(), 2.5);
        assert_eq!(s.size_hint(), 5);
    }

    #[test]
    fn term_scorer_matches_bm25() {
        let docs: Vec<DocId> = (0..100).step_by(3).collect();
        let weight = Bm25Weight::new(17, 1000, Bm25Params::default(), 12.0);
        let mut s = TermScorer::new(
            TfDocSet { inner: VecDocSet::new(docs), tf: 4 },
            weight,
            Box::new(|d| 10 + d % 5),
        );
        let check = Bm25Weight::new(17, 1000, Bm25Params::default(), 12.0);
        while s.doc() != crate::query::docset::TERMINATED {
            let d = s.advance();
            if d == crate::query::docset::TERMINATED {
                break;
            }
            let len = 10 + d % 5;
            let want = check.score(4, len);
            assert!((s.score() - want).abs() < 1e-6, "doc {d}");
        }
        // Sanity: at average length the closed form holds.
        let idf = idf(17, 1000);
        let want = idf * 4.0 / (4.0 + 1.2);
        assert!((check.score(4, 12) - want).abs() < 1e-5);
    }

    #[test]
    fn boxed_scorer_delegates() {
        let docs: Vec<DocId> = (0..10).step_by(2).collect();
        let mut s: Box<dyn Scorer> = Box::new(ConstScorer::new(VecDocSet::new(docs), 1.5));
        assert_eq!(s.advance(), 0);
        assert_eq!(s.score(), 1.5);
        assert_eq!(s.seek(7), 8);
        assert_eq!(s.doc(), 8);
    }
}

/// Fewest docs of a leaf before [`doc_range_parts`] splits it (tests
/// lower it to split small leaves).
pub static PART_SPLIT_MIN_DOCS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1 << 20);

/// [`PART_SPLIT_MIN_DOCS`] now.
pub fn part_split_min_docs() -> u32 {
    PART_SPLIT_MIN_DOCS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Contiguous doc ranges covering `0..num_docs` for a scorer that costs a
/// lot per doc (scripts): one range for a small leaf, else up to two per
/// search thread, at least 256k docs each.
pub fn doc_range_parts(num_docs: u32) -> Vec<(DocId, DocId)> {
    let min = part_split_min_docs();
    if num_docs < min {
        return vec![(0, num_docs)];
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
    let parts = (threads * 2).min((num_docs / (min / 4).max(1)) as usize).max(1);
    let step = num_docs.div_ceil(parts as u32);
    (0..parts as u32).map(|i| (i * step, ((i + 1) * step).min(num_docs))).filter(|(a, b)| a < b).collect()
}

/// A scorer's docs restricted to `[lo, hi)`, with its scores.
pub struct DocRangeScorer {
    inner: Box<dyn Scorer>,
    lo: DocId,
    hi: DocId,
    cur: DocId,
    started: bool,
}

impl DocRangeScorer {
    pub fn new(inner: Box<dyn Scorer>, lo: DocId, hi: DocId) -> Self {
        Self { inner, lo, hi, cur: crate::query::docset::TERMINATED, started: false }
    }
    /// The first doc at or after `target`: an unstarted scorer advances
    /// first (a default `seek` starts from its current doc).
    fn start(&mut self, target: DocId) -> DocId {
        self.started = true;
        let d = self.inner.advance();
        if d < target { self.inner.seek(target) } else { d }
    }

    fn cap(&mut self, d: DocId) -> DocId {
        self.cur = if d >= self.hi { crate::query::docset::TERMINATED } else { d };
        self.cur
    }
}

impl DocSet for DocRangeScorer {
    fn advance(&mut self) -> DocId {
        if self.started && self.cur == crate::query::docset::TERMINATED {
            return self.cur;
        }
        let d = if self.started { self.inner.advance() } else { self.start(self.lo) };
        self.cap(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.started && (self.cur == crate::query::docset::TERMINATED || self.cur >= target) {
            return self.cur;
        }
        let d = if self.started { self.inner.seek(target.max(self.lo)) } else { self.start(target.max(self.lo)) };
        self.cap(d)
    }
    fn doc(&self) -> DocId {
        self.cur
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint().min(self.hi - self.lo)
    }
}

impl Scorer for DocRangeScorer {
    fn score(&mut self) -> f32 {
        self.inner.score()
    }
}

/// One leaf's scorer as parts over contiguous doc ranges, in doc order
/// (each its own scorer, e.g. its own script executor): iterated, the
/// parts' docs one after another; collected top-k, the parts on the search
/// pool, each into its own queue and live count, merged as one queue (same
/// hits, same order: score, then doc). A large merged segment's script
/// scoring otherwise runs on one thread.
pub struct PartedScorer {
    parts: Vec<std::sync::Mutex<Box<dyn Scorer>>>,
    cur: usize,
    doc: DocId,
}

impl PartedScorer {
    pub fn new(parts: Vec<Box<dyn Scorer>>) -> Self {
        Self { parts: parts.into_iter().map(std::sync::Mutex::new).collect(), cur: 0, doc: crate::query::docset::TERMINATED }
    }
}

impl DocSet for PartedScorer {
    fn advance(&mut self) -> DocId {
        while self.cur < self.parts.len() {
            let d = self.parts[self.cur].get_mut().expect("part lock").advance();
            if d != crate::query::docset::TERMINATED {
                self.doc = d;
                return d;
            }
            self.cur += 1;
        }
        self.doc = crate::query::docset::TERMINATED;
        self.doc
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc != crate::query::docset::TERMINATED && self.doc >= target {
            return self.doc;
        }
        while self.cur < self.parts.len() {
            let d = self.parts[self.cur].get_mut().expect("part lock").seek(target);
            if d != crate::query::docset::TERMINATED {
                self.doc = d;
                return d;
            }
            self.cur += 1;
        }
        self.doc = crate::query::docset::TERMINATED;
        self.doc
    }
    fn doc(&self) -> DocId {
        self.doc
    }
    fn size_hint(&self) -> u32 {
        self.parts.iter().map(|p| p.lock().map(|p| p.size_hint()).unwrap_or(0)).fold(0u32, u32::saturating_add)
    }
}

impl Scorer for PartedScorer {
    fn score(&mut self) -> f32 {
        self.parts[self.cur].get_mut().expect("part lock").score()
    }
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        if k == 0 || self.doc != crate::query::docset::TERMINATED || self.cur != 0 {
            return None;
        }
        let live = |d: DocId| deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
        let parts = &self.parts;
        let outs = crate::query::searcher::par_map_search(parts.len(), |i| {
            let mut p = parts[i].lock().expect("part lock");
            if let Some(r) = p.collect_top_k_bulk(k, deleted) {
                return r;
            }
            let mut top = crate::query::collector::TopKCollector::new(k);
            let mut n = 0u64;
            let mut d = p.advance();
            while d != crate::query::docset::TERMINATED {
                if live(d) {
                    n += 1;
                    let s = p.score();
                    top.collect(d, s);
                }
                d = p.advance();
            }
            (top.into_sorted_vec(), n)
        });
        let mut top = crate::query::collector::TopKCollector::new(k);
        let mut total = 0u64;
        for (hits, n) in outs {
            total += n;
            for h in hits {
                top.collect(h.doc, h.score);
            }
        }
        self.cur = self.parts.len();
        Some((top.into_sorted_vec(), total))
    }
}

#[cfg(test)]
mod parted_tests {
    use super::*;
    use crate::query::docset::TERMINATED;

    /// Docs `0..n` step 3, scoring `(doc * 7919) % 101`.
    struct Fake { docs: Vec<DocId>, at: usize, cur: DocId }
    impl DocSet for Fake {
        fn advance(&mut self) -> DocId {
            self.cur = self.docs.get(self.at).copied().unwrap_or(TERMINATED);
            self.at += 1;
            self.cur
        }
        fn doc(&self) -> DocId { self.cur }
        fn size_hint(&self) -> u32 { self.docs.len() as u32 }
    }
    impl Scorer for Fake {
        fn score(&mut self) -> f32 { ((self.cur as u64 * 7919) % 101) as f32 }
    }
    fn fake(n: u32) -> Box<dyn Scorer> { Box::new(Fake { docs: (0..n).step_by(3).collect(), at: 0, cur: TERMINATED }) }

    /// Parts over doc ranges give one scorer's docs in order and its top-k
    /// (ties broken by doc), with deleted docs skipped and counted out.
    #[test]
    fn parted_scorer_matches_one_pass() {
        let n = 10_000u32;
        let ranges = [(0, 1234), (1234, 1235), (1235, 7000), (7000, n)];
        let parted = || PartedScorer::new(ranges.iter().map(|&(lo, hi)| Box::new(DocRangeScorer::new(fake(n), lo, hi)) as Box<dyn Scorer>).collect());
        let mut all = Vec::new();
        let mut p = parted();
        let mut d = p.advance();
        while d != TERMINATED { all.push((d, p.score())); d = p.advance(); }
        let mut one = fake(n);
        let mut want = Vec::new();
        let mut d = one.advance();
        while d != TERMINATED { want.push((d, one.score())); d = one.advance(); }
        assert_eq!(all, want);
        let mut p = parted();
        assert_eq!(p.seek(4000), 4002);
        assert_eq!(p.seek(4001), 4002);
        assert_eq!(p.seek(9999), 9999);
        assert_eq!(p.advance(), TERMINATED);
        let mut deleted = vec![0u64; (n as usize).div_ceil(64)];
        for d in (0..n).step_by(7) { deleted[d as usize / 64] |= 1 << (d % 64); }
        for k in [1, 10, 50] {
            let (hits, total) = parted().collect_top_k_bulk(k, Some(&deleted)).unwrap();
            let mut top = crate::query::collector::TopKCollector::new(k);
            let mut count = 0;
            for &(d, s) in &want {
                if deleted[d as usize / 64] >> (d % 64) & 1 == 0 { count += 1; top.collect(d, s); }
            }
            let expect = top.into_sorted_vec();
            assert_eq!(total, count);
            assert_eq!(hits.iter().map(|h| (h.doc, h.score)).collect::<Vec<_>>(), expect.iter().map(|h| (h.doc, h.score)).collect::<Vec<_>>(), "k {k}");
        }
    }
}
