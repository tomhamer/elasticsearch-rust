//! Weighted feature queries (ES `sparse_vector` / `text_expansion` /
//! `weighted_tokens`, `rank_features` linear scoring; Lucene
//! `FeatureQuery` under a `BooleanQuery` of SHOULD clauses).
//!
//! A feature field ([`crate::types::Field::features`]) indexes each token
//! as a term whose frequency holds the weight's top 16 bits, so a token's
//! postings carry every doc's weight and each full block's skip entry its
//! largest one (`max_tf`). A doc scores `Σ query_weight × doc_weight` over
//! the query's tokens. Each token's scorer bounds a block range by
//! `query_weight × max weight`, which lets the OR of them run as MaxScore
//! top-k ([`crate::query::boolean::Union`]): blocks that cannot beat the
//! k-th best score are skipped instead of every matching doc being scored.

use crate::codec::postings::{PostingsReader, BLOCK_DOCS};
use crate::fieldtypes::features::freq_to_weight;
use crate::query::boolean::Union;
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::explain::Explanation;
use crate::query::query::{CustomQuery, SearchContext};
use crate::query::reader::LeafReader;
use crate::query::scorer::Scorer;
use crate::types::{DocId, Term};
use std::cell::Cell;

/// One token of a feature field scored `weight × doc weight`.
pub struct FeatureTermScorer {
    reader: PostingsReader,
    weight: f32,
    current: DocId,
    started: bool,
    /// Largest weight of the tail block's docs (NaN until computed).
    tail_max: Cell<f32>,
}

impl FeatureTermScorer {
    pub fn new(reader: PostingsReader, weight: f32) -> Self {
        Self { reader, weight, current: TERMINATED, started: false, tail_max: Cell::new(f32::NAN) }
    }

    /// Whether block bounds hold (a finite, nonnegative query weight).
    fn prunable(&self) -> bool {
        self.weight.is_finite() && self.weight >= 0.0
    }

    #[inline]
    fn doc_weight(tf: u32) -> f32 {
        freq_to_weight(tf)
    }

    /// Largest doc weight in the tail (partial) block, which has no skip
    /// entry; 0 without a tail.
    fn tail_max_weight(&self) -> f32 {
        let m = self.tail_max.get();
        if !m.is_nan() {
            return m;
        }
        let r = &self.reader;
        let mut m = 0f32;
        let n = r.tail_len();
        if n > 0 {
            let mut docs = [0u32; BLOCK_DOCS];
            let mut tfs = [0u32; BLOCK_DOCS];
            r.decode_tail(&mut docs, &mut tfs);
            for &tf in &tfs[..n] {
                m = m.max(Self::doc_weight(tf));
            }
        }
        self.tail_max.set(m);
        m
    }

    /// Largest doc weight of blocks `b..` up to the first block whose last
    /// doc is at least `hi`.
    fn max_weight_through(&self, mut b: usize, hi: DocId) -> f32 {
        let r = &self.reader;
        let (nf, nb) = (r.num_full_blocks(), r.num_blocks());
        let mut m = 0f32;
        while b < nb {
            if b < nf {
                let max_tf = r.skip(b).max_tf;
                if max_tf == u32::MAX {
                    return f32::INFINITY;
                }
                m = m.max(Self::doc_weight(max_tf));
                if r.block_last_doc(b) >= hi {
                    break;
                }
            } else {
                m = m.max(self.tail_max_weight());
            }
            b += 1;
        }
        m
    }

    /// The first full block at or after `from` that may hold docs >=
    /// `target`, else the tail's index.
    fn first_block_reaching(&self, target: DocId) -> usize {
        let r = &self.reader;
        let (mut lo, mut hi) = (0, r.num_full_blocks());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if r.block_last_doc(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn bound(&self, w: f32) -> Option<f32> {
        let b = self.weight * w;
        (b.is_finite() && b >= 0.0).then_some(b)
    }
}

impl DocSet for FeatureTermScorer {
    #[inline]
    fn advance(&mut self) -> DocId {
        self.started = true;
        self.current = self.reader.next().unwrap_or(TERMINATED);
        self.current
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        if self.started && (self.current == TERMINATED || self.current >= target) {
            return self.current;
        }
        self.started = true;
        self.current = self.reader.advance(target).unwrap_or(TERMINATED);
        self.current
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.reader.doc_freq()
    }
}

impl Scorer for FeatureTermScorer {
    #[inline]
    fn score(&mut self) -> f32 {
        self.weight * Self::doc_weight(self.reader.term_freq().unwrap_or(1))
    }

    fn as_feature_term(&mut self) -> Option<&mut FeatureTermScorer> {
        Some(self)
    }

    fn max_score(&self) -> Option<f32> {
        if !self.prunable() {
            return None;
        }
        self.bound(self.max_weight_through(0, TERMINATED))
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        if !self.prunable() {
            return None;
        }
        let from = if self.started { self.current } else { 0 };
        if from == TERMINATED || from > up_to {
            return Some(0.0);
        }
        let b = self.first_block_reaching(from);
        self.bound(self.max_weight_through(b, up_to))
    }
}

/// Docs per window of [`try_feature_window`] (Lucene `BooleanScorer`: 2048).
const FEATURE_WINDOW: usize = 4096;

/// One clause's position in [`try_feature_window`]: its decoded block.
struct WindowCursor {
    block: usize,
    at: usize,
    len: usize,
    docs: [u32; BLOCK_DOCS],
    tfs: [u32; BLOCK_DOCS],
}

impl WindowCursor {
    /// Decode block `b` (full or tail) of `r`; 0 docs past the last block.
    fn load(&mut self, r: &PostingsReader, b: usize) {
        self.block = b;
        self.at = 0;
        self.len = if b < r.num_full_blocks() {
            if r.decode_block(b, &mut self.docs, &mut self.tfs) {
                self.tfs.fill(1);
            }
            BLOCK_DOCS
        } else if b < r.num_blocks() {
            r.decode_tail(&mut self.docs, &mut self.tfs)
        } else {
            0
        };
    }
}

/// Top `k` (above `floor`) and the live match count of an OR of unstarted
/// feature-token clauses that must visit every match (an exact total:
/// `track_total_hits: true`), Lucene `BooleanScorer`-style: per window of
/// [`FEATURE_WINDOW`] docs, each clause adds its block-decoded postings
/// into a score array and a match bitset, then the window's matches are
/// collected in doc order. No per-doc heap over the (often 100+) clauses.
/// A doc's score is the sum of its clauses' scores in clause order, as
/// [`Union`] sums them, so hits and ties are the same. `None` (nothing
/// consumed) unless every clause is an unstarted [`FeatureTermScorer`].
pub(crate) fn try_feature_window(children: &mut [Box<dyn Scorer>], k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
    use rayon::prelude::*;
    if children.len() < 2 {
        return None;
    }
    let mut terms: Vec<&mut FeatureTermScorer> = Vec::with_capacity(children.len());
    for c in children.iter_mut() {
        let t = c.as_feature_term()?;
        if t.started {
            return None;
        }
        terms.push(t);
    }
    let (hits, total) = {
        let lists: Vec<(&PostingsReader, f32)> = terms.iter().map(|t| (&t.reader, t.weight)).collect();
        // A large segment's doc range splits into parts scored in parallel
        // (each a window scan of its own range), merged by score.
        let postings: u64 = lists.iter().map(|(r, _)| r.doc_freq() as u64).sum();
        let end = lists.iter().filter_map(|(r, _)| last_doc(r)).max().map_or(0, |d| d as u64 + 1);
        let parts = if postings >= FEATURE_PARALLEL_POSTINGS {
            (end / FEATURE_PART_MIN_DOCS).clamp(1, rayon::current_num_threads().max(1) as u64)
        } else {
            1
        };
        if parts <= 1 {
            window_range(&lists, 0, u32::MAX, k, deleted, floor)
        } else {
            let span = end.div_ceil(parts).next_multiple_of(FEATURE_WINDOW as u64);
            let ranges: Vec<(u32, u32)> = (0..parts).map(|p| ((p * span).min(end) as u32, ((p + 1) * span).min(end) as u32)).filter(|(a, b)| a < b).collect();
            let outs: Vec<(Vec<crate::query::collector::ScoredDoc>, u64)> = ranges.par_iter().map(|&(lo, hi)| window_range(&lists, lo, hi, k, deleted, floor)).collect();
            let total = outs.iter().map(|o| o.1).sum();
            let mut hits: Vec<crate::query::collector::ScoredDoc> = outs.into_iter().flat_map(|o| o.0).collect();
            // The order the sequential top-k keeps: score, then the earlier doc.
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
            hits.truncate(k);
            (hits, total)
        }
    };
    for t in terms {
        t.started = true;
        t.current = TERMINATED;
    }
    Some((hits, total))
}

/// Total postings above which [`try_feature_window`] scores a segment's
/// doc range in parallel parts, and the fewest docs a part covers.
/// (Small in tests, so the bit-exact tests run the parallel parts.)
const FEATURE_PARALLEL_POSTINGS: u64 = if cfg!(test) { 1 << 10 } else { 1 << 21 };
const FEATURE_PART_MIN_DOCS: u64 = if cfg!(test) { 1 << 14 } else { 1 << 18 };

/// The last doc of a postings list.
fn last_doc(r: &PostingsReader) -> Option<DocId> {
    if r.tail_len() > 0 {
        let (mut docs, mut tfs) = ([0u32; BLOCK_DOCS], [0u32; BLOCK_DOCS]);
        let n = r.decode_tail(&mut docs, &mut tfs);
        return n.checked_sub(1).map(|i| docs[i]);
    }
    r.num_full_blocks().checked_sub(1).map(|b| r.block_last_doc(b))
}

/// The window scan of [`try_feature_window`] over docs `lo..hi` (`lo` a
/// multiple of [`FEATURE_WINDOW`]): top `k` above `floor` and the live
/// match count.
fn window_range(lists: &[(&PostingsReader, f32)], lo: DocId, hi: DocId, k: usize, deleted: Option<&[u64]>, floor: f32) -> (Vec<crate::query::collector::ScoredDoc>, u64) {
    let mut cursors: Vec<WindowCursor> = lists
        .iter()
        .map(|(r, _)| {
            let mut c = WindowCursor { block: 0, at: 0, len: 0, docs: [0; BLOCK_DOCS], tfs: [0; BLOCK_DOCS] };
            // The first block that may hold a doc at or past `lo`.
            let (mut a, mut b) = (0usize, r.num_full_blocks());
            while a < b {
                let mid = (a + b) / 2;
                if r.block_last_doc(mid) < lo { a = mid + 1 } else { b = mid }
            }
            c.load(r, a);
            while c.len > 0 {
                while c.at < c.len && c.docs[c.at] < lo {
                    c.at += 1;
                }
                if c.at < c.len {
                    break;
                }
                let next = c.block + 1;
                c.load(r, next);
            }
            c
        })
        .collect();
    let mut top = crate::query::bulk::TopK::with_floor(k, floor);
    let mut total = 0u64;
    let mut scores = vec![0f32; FEATURE_WINDOW];
    let mut bits = vec![0u64; FEATURE_WINDOW / 64];
    loop {
        let mut min = TERMINATED;
        for c in &cursors {
            if c.at < c.len {
                min = min.min(c.docs[c.at]);
            }
        }
        if min == TERMINATED || min >= hi {
            break;
        }
        let base = min & !(FEATURE_WINDOW as u32 - 1);
        let end = (base as u64 + FEATURE_WINDOW as u64).min(hi as u64);
        for ((r, w), c) in lists.iter().zip(cursors.iter_mut()) {
            while c.len > 0 {
                while c.at < c.len {
                    let d = c.docs[c.at];
                    if d as u64 >= end {
                        break;
                    }
                    let off = (d - base) as usize;
                    let s = w * freq_to_weight(c.tfs[c.at]);
                    let (word, bit) = (off >> 6, 1u64 << (off & 63));
                    if bits[word] & bit == 0 {
                        bits[word] |= bit;
                        scores[off] = s;
                    } else {
                        scores[off] += s;
                    }
                    c.at += 1;
                }
                if c.at < c.len {
                    break;
                }
                let next = c.block + 1;
                c.load(r, next);
            }
        }
        for (wi, word) in bits.iter_mut().enumerate() {
            let mut w = *word;
            *word = 0;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                let off = wi * 64 + b;
                let doc = base + off as u32;
                if deleted.is_some_and(|del| del.get(doc as usize / 64).is_some_and(|x| (x >> (doc % 64)) & 1 != 0)) {
                    continue;
                }
                total += 1;
                top.offer(doc, scores[off]);
            }
        }
    }
    (top.into_sorted(), total)
}

/// `Σ weight × doc weight` over `tokens` of the feature field `field`
/// (an OR: a doc matches when it has any of the tokens).
#[derive(Debug)]
pub struct FeatureQuery {
    pub field: u16,
    /// `(token, query weight)`; a token listed twice keeps its largest weight.
    pub tokens: Vec<(String, f32)>,
    /// Query name (explanations).
    pub name: &'static str,
}

impl FeatureQuery {
    pub fn new(field: u16, mut tokens: Vec<(String, f32)>, name: &'static str) -> Self {
        tokens.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        tokens.dedup_by(|a, b| a.0 == b.0);
        Self { field, tokens, name }
    }

    fn scorers(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Vec<Box<dyn Scorer>> {
        let mut out: Vec<Box<dyn Scorer>> = Vec::with_capacity(self.tokens.len());
        for (token, w) in &self.tokens {
            let term = Term { field: self.field, text: token.as_bytes().to_vec() };
            if let Some(r) = ctx.postings(leaf, &term) {
                out.push(Box::new(FeatureTermScorer::new(r, w * boost)));
            }
        }
        out
    }
}

impl CustomQuery for FeatureQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        self.scorer_b(leaf, ctx, 1.0)
    }

    fn absorbs_boost(&self) -> bool {
        true
    }

    fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Option<Box<dyn Scorer>> {
        let mut s = self.scorers(leaf, ctx, boost);
        match s.len() {
            0 => None,
            1 => s.pop(),
            // `Union` (not `union_of`'s two-child `Disjunction2`) runs
            // MaxScore top-k for any number of clauses.
            _ => Some(Box::new(Union::new(s))),
        }
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        self.explain_b(leaf, ctx, doc, score, 1.0)
    }

    fn explain_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, _score: f32, boost: f32) -> Explanation {
        let mut parts = Vec::new();
        let mut total = 0f32;
        for (token, w) in &self.tokens {
            let term = Term { field: self.field, text: token.as_bytes().to_vec() };
            let Some(mut r) = ctx.postings(leaf, &term) else { continue };
            if r.advance(doc) != Some(doc) {
                continue;
            }
            let dw = freq_to_weight(r.term_freq().unwrap_or(1));
            let s = w * boost * dw;
            total += s;
            parts.push(Explanation::leaf(s, format!("Linear function on the {token} feature, computed as w * S from:")));
        }
        Explanation::new(total, format!("{} query, sum of:", self.name), parts)
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::codec::postings::write_postings;
    use crate::query::collector::collect_top_k;

    /// Clause `i` of a pseudo-random fixture: ascending docs below `n`
    /// (some lists denser than others, some single docs) with feature
    /// weights encoded as term freqs.
    fn lists(n: u32, clauses: usize, seed: u64) -> Vec<(Vec<u32>, Vec<u32>, f32)> {
        let mut x = seed;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..clauses)
            .map(|i| {
                let step = [1u64, 3, 17, 250, 4000, 60_000][i % 6];
                let mut docs = Vec::new();
                let mut d = rnd() % step;
                while d < n as u64 {
                    docs.push(d as u32);
                    d += 1 + rnd() % (2 * step);
                }
                // Weights as stored: the top 16 bits of an f32 (freq << 15).
                let tfs: Vec<u32> = docs.iter().map(|_| (1.0f32 + (rnd() % 1000) as f32 / 7.0).to_bits() >> 15).collect();
                (docs, tfs, 0.05 + (rnd() % 100) as f32 / 37.0)
            })
            .collect()
    }

    fn scorers(ls: &[(Vec<u32>, Vec<u32>, f32)]) -> Vec<Box<dyn Scorer>> {
        ls.iter()
            .map(|(docs, tfs, w)| {
                let mut buf = Vec::new();
                write_postings(docs, tfs, &mut buf);
                Box::new(FeatureTermScorer::new(PostingsReader::from_bytes(buf).unwrap(), *w)) as Box<dyn Scorer>
            })
            .collect()
    }

    #[test]
    fn windowed_feature_or_matches_the_per_doc_union_bit_for_bit() {
        let n = 200_000;
        for (clauses, seed) in [(2usize, 1u64), (5, 7), (40, 11), (130, 3)] {
            let ls = lists(n, clauses, seed);
            let mut del = vec![0u64; (n as usize).div_ceil(64)];
            for d in (0..n).step_by(89) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            for deleted in [None, Some(del.as_slice())] {
                for k in [1usize, 10, 1000] {
                    let mut union = Union::new(scorers(&ls));
                    let want = collect_top_k(&mut union, k, |d| deleted.is_none_or(|w| (w[d as usize / 64] >> (d % 64)) & 1 == 0));
                    let mut children = scorers(&ls);
                    let got = try_feature_window(&mut children, k, deleted, f32::NEG_INFINITY).expect("feature clauses");
                    assert_eq!(want.1, got.1, "{clauses} clauses: count");
                    assert_eq!(want.0.len(), got.0.len());
                    for (a, b) in want.0.iter().zip(&got.0) {
                        assert_eq!((a.doc, a.score.to_bits()), (b.doc, b.score.to_bits()), "{clauses} clauses, k {k}");
                    }
                    assert!(children.iter().all(|c| c.doc() == TERMINATED), "consumed");
                }
            }
        }
    }

    /// A bounded-total top-k of many token clauses takes the window scan:
    /// the same hits as the per-doc union, the exact count.
    #[test]
    fn bounded_union_of_many_tokens_matches_the_per_doc_union() {
        let n = 200_000;
        let ls = lists(n, 40, 9);
        for k in [1usize, 10, 100] {
            let mut union = Union::new(scorers(&ls));
            let want = collect_top_k(&mut union, k, |_| true);
            let mut u = Union::new(scorers(&ls));
            let got = u.collect_top_k_count_limited(k, None, 10_000, &crate::query::bulk::Floor::none()).expect("bulk");
            assert_eq!(got.1, want.1, "exact total");
            let bits = |v: &[crate::query::collector::ScoredDoc]| v.iter().map(|h| (h.doc, h.score.to_bits())).collect::<Vec<_>>();
            assert_eq!(bits(&got.0), bits(&want.0), "k {k}");
        }
    }

    #[test]
    fn windowed_feature_or_declines_other_clauses() {
        let ls = lists(10_000, 3, 5);
        let mut children = scorers(&ls);
        children[1].advance();
        assert!(try_feature_window(&mut children, 10, None, f32::NEG_INFINITY).is_none(), "a started clause");
    }
}
