//! Block-at-a-time ("bulk") top-k + exact-count execution for term, AND and
//! OR queries over BM25 term postings (Lucene's `BulkScorer`s).
//!
//! Everything here works on decoded 128-doc blocks instead of per-doc
//! virtual `advance`/`score` calls:
//!
//! * **term**: decode a block, score all of it in a vectorizable loop, and
//!   only scan for hits when the block max beats the top-k threshold. Blocks
//!   whose skip-table bound `score(max_tf, min_norm)` cannot beat the
//!   threshold are not decoded at all (still counted exactly; with
//!   deletions only their doc ids are decoded).
//! * **keyword / constant score** (no norms, every tf = 1): every hit scores
//!   the same, so top-k = the first k live docs and count = df − deleted.
//! * **AND** ([`conjunction`]): the rarest list's block is the candidate set,
//!   filtered through the other lists in df order; other lists skip blocks
//!   via `last_doc` and locate candidates by galloping; tfs and norms are
//!   fetched (random access) only for final matches.
//! * **OR** ([`disjunction`]): MaxScore windows (non-competitive clauses only
//!   set count bits and add their score to docs an essential clause hit);
//!   two clauses without deletions use inclusion–exclusion
//!   (`|A ∪ B| = dfA + dfB − |A ∩ B|`) and never walk the frequent list
//!   when it cannot compete.
//!
//! Scorers expose this through [`Scorer::collect_top_k_bulk`]; anything that
//! does not qualify falls back to the per-doc loop.

use crate::codec::postings::{PostingsReader, BLOCK_DOCS};
use crate::query::bm25::Bm25Weight;
use crate::query::collector::ScoredDoc;
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::scorer::Scorer;
use crate::query::segment_scorer::Norms;
use crate::types::DocId;
use std::sync::Arc;

#[inline(always)]
fn is_deleted(del: &[u64], d: u32) -> bool {
    match del.get((d >> 6) as usize) {
        Some(w) => (w >> (d & 63)) & 1 == 1,
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Top-k
// ---------------------------------------------------------------------------

/// Top-k kept as a sorted array (score desc, doc asc) with an explicit
/// threshold. [`TopK::offer`] assumes docs arrive in ascending order (a tie
/// with the threshold then never enters, which is the doc-asc tie-break);
/// [`TopK::offer_any`] handles arbitrary order.
pub struct TopK {
    k: usize,
    hits: Vec<ScoredDoc>,
    thr: f32,
    /// Hits must score strictly above this (other leaves' hits).
    floor: f32,
    /// The k-th best score last published to a [`Floor`].
    published: f32,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self::with_floor(k, f32::NEG_INFINITY)
    }

    /// Top-k that only admits hits scoring strictly above `floor` (the
    /// k-th best score of earlier leaves).
    pub fn with_floor(k: usize, floor: f32) -> Self {
        Self { k, hits: Vec::with_capacity(k.min(1024)), thr: if k == 0 { f32::INFINITY } else { floor }, floor, published: f32::NEG_INFINITY }
    }

    /// Exchange bounds with the other leaves of a parallel search: raise
    /// this top-k's floor to what they already hold, and publish this
    /// leaf's k-th best score once `k` hits are held.
    #[inline]
    pub fn sync(&mut self, f: &Floor<'_>) {
        if self.k == 0 || f.shared.is_none() {
            return;
        }
        let v = f.get();
        if v > self.floor {
            self.floor = v;
            if v > self.thr {
                self.thr = v;
            }
        }
        if self.hits.len() == self.k {
            let kth = self.hits[self.k - 1].score;
            if kth > self.published {
                self.published = kth;
                f.publish(kth);
            }
        }
    }

    /// Score a new (ascending-doc) hit must exceed to enter.
    #[inline(always)]
    pub fn threshold(&self) -> f32 {
        self.thr
    }

    /// Offer a hit whose doc is larger than every doc offered before.
    #[inline(always)]
    pub fn offer(&mut self, doc: DocId, score: f32) {
        if score > self.thr {
            self.insert(doc, score);
        }
    }

    /// Offer a hit in any doc order.
    #[inline]
    pub fn offer_any(&mut self, doc: DocId, score: f32) {
        if self.k == 0 {
            return;
        }
        if self.hits.len() < self.k {
            if score <= self.thr {
                return;
            }
        } else {
            let w = self.hits[self.k - 1];
            if !(score > w.score || (score == w.score && doc < w.doc)) {
                return;
            }
        }
        self.insert(doc, score);
    }

    #[inline(never)]
    fn insert(&mut self, doc: DocId, score: f32) {
        let mut i = self.hits.len();
        while i > 0 {
            let h = self.hits[i - 1];
            if h.score > score || (h.score == score && h.doc < doc) {
                break;
            }
            i -= 1;
        }
        if self.hits.len() == self.k {
            self.hits.pop();
        }
        self.hits.insert(i, ScoredDoc { doc, score });
        if self.hits.len() == self.k {
            // The floor only exceeds the k-th score once raised by `sync`.
            self.thr = self.hits[self.k - 1].score.max(self.floor);
        }
    }

    /// Hits sorted by score desc, doc asc.
    pub fn into_sorted(self) -> Vec<ScoredDoc> {
        self.hits
    }
}

/// The k-th best score some leaf of a parallel search already holds
/// (Lucene's `MaxScoreAccumulator`), raised monotonically, with the leaf
/// that holds it (the earliest of equal scores). Leaves read it as a floor
/// their hits must reach.
#[derive(Debug, Default)]
pub struct SharedFloor(std::sync::atomic::AtomicU64);

impl SharedFloor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish `kth`: some leaf holds `k` live hits scoring at least `kth`.
    #[inline]
    pub fn raise(&self, kth: f32) {
        self.raise_from(kth, u32::MAX);
    }

    /// Publish `kth` held by leaf `leaf` (`u32::MAX`: unknown).
    #[inline]
    pub fn raise_from(&self, kth: f32, leaf: u32) {
        // Bits of nonnegative floats order like the floats (0 = unset); of
        // equal scores the earliest leaf's is kept.
        if kth.is_finite() && kth > 0.0 {
            let v = ((kth.to_bits() as u64) << 32) | (u32::MAX - leaf) as u64;
            self.0.fetch_max(v, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Hits must score strictly above this. A hit tying the published
    /// score may still win on the (leaf, doc) tie-break, so the floor sits
    /// just below it.
    #[inline]
    pub fn get(&self) -> f32 {
        self.get_for(None)
    }

    /// [`Self::get`] for leaf `leaf`: a hit of a later leaf than the one
    /// holding the published score loses that tie (ties go to the earlier
    /// leaf), so the floor is the score itself.
    #[inline]
    pub fn get_for(&self, leaf: Option<u32>) -> f32 {
        match self.0.load(std::sync::atomic::Ordering::Relaxed) {
            0 => f32::NEG_INFINITY,
            v => {
                let score = f32::from_bits((v >> 32) as u32);
                let holder = u32::MAX - v as u32;
                if leaf.is_some_and(|l| holder < l) { score } else { score.next_down() }
            }
        }
    }
}

/// What a leaf's hits must beat: the k-th best of earlier leaves (`fixed`,
/// strictly: they win ties) and, in a parallel search, the scores other
/// leaves publish meanwhile.
#[derive(Clone, Copy, Debug)]
pub struct Floor<'a> {
    fixed: f32,
    shared: Option<&'a SharedFloor>,
    /// The leaf searching (its hits lose ties to earlier leaves'); `None`:
    /// unknown.
    leaf: Option<u32>,
    /// Matches counted by all leaves of the shard so far (published in
    /// batches): the `track_total_hits` limit is shard-wide, as Lucene's
    /// shared `HitsThresholdChecker`.
    hits: Option<&'a std::sync::atomic::AtomicU64>,
}

impl<'a> Floor<'a> {
    pub fn new(fixed: f32, shared: Option<&'a SharedFloor>) -> Self {
        Self { fixed, shared, hits: None, leaf: None }
    }

    /// No floor.
    pub fn none() -> Floor<'static> {
        Floor { fixed: f32::NEG_INFINITY, shared: None, hits: None, leaf: None }
    }

    /// Count against the shard-wide hit counter `hits`.
    pub fn with_hits<'b>(self, hits: &'b std::sync::atomic::AtomicU64) -> Floor<'b>
    where
        'a: 'b,
    {
        Floor { fixed: self.fixed, shared: self.shared, hits: Some(hits), leaf: self.leaf }
    }

    /// The floor of leaf `leaf` (see [`SharedFloor::get_for`]).
    pub fn at_leaf(self, leaf: usize) -> Self {
        Floor { leaf: u32::try_from(leaf).ok(), ..self }
    }

    /// Matches the shard's leaves published so far (0 without a counter).
    #[inline]
    pub fn hits_counted(&self) -> u64 {
        self.hits.map_or(0, |h| h.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Publish `n` more matches this leaf counted.
    #[inline]
    pub fn add_counted(&self, n: u64) {
        if let (Some(h), true) = (self.hits, n > 0) {
            h.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Hits must score strictly above this.
    #[inline]
    pub fn get(&self) -> f32 {
        match self.shared {
            Some(s) => self.fixed.max(s.get_for(self.leaf)),
            None => self.fixed,
        }
    }

    /// Publish this leaf's k-th best score to the other leaves.
    #[inline]
    pub fn publish(&self, kth: f32) {
        if let Some(s) = self.shared {
            s.raise_from(kth, self.leaf.unwrap_or(u32::MAX));
        }
    }
}

// ---------------------------------------------------------------------------
// The concrete BM25 term scorer
// ---------------------------------------------------------------------------

/// A BM25 term scorer over one segment's postings with 1-byte norms: a
/// statically dispatched per-doc [`Scorer`] plus the bulk paths.
pub struct Bm25TermScorer {
    reader: PostingsReader,
    weight: Bm25Weight,
    norms: Arc<Norms>,
    current: DocId,
    started: bool,
    /// Best score of the tail block's docs (NaN until computed).
    tail_max: std::cell::Cell<f32>,
}

impl Bm25TermScorer {
    #[inline]
    pub fn new(reader: PostingsReader, weight: Bm25Weight, norms: Arc<Norms>) -> Self {
        Self { reader, weight, norms, current: TERMINATED, started: false, tail_max: std::cell::Cell::new(f32::NAN) }
    }

    /// Exact best score of the docs in the tail (partial) block, which has
    /// no skip entry; 0 without a tail.
    pub fn tail_max_score(&self) -> f32 {
        let m = self.tail_max.get();
        if !m.is_nan() {
            return m;
        }
        let r = &self.reader;
        let n = r.tail_len();
        let mut m = 0f32;
        if n > 0 {
            let mut docs = [0u32; BLOCK_DOCS];
            let mut tfs = [0u32; BLOCK_DOCS];
            r.decode_tail(&mut docs, &mut tfs);
            for j in 0..n {
                m = m.max(self.weight.score_norm(tfs[j], self.norm_byte(docs[j])));
            }
        }
        self.tail_max.set(m);
        m
    }

    /// Upper bound of the scores of the docs in blocks `b..` up to the first
    /// block whose last doc is at least `hi` (block-max impacts; the tail
    /// scored exactly).
    #[inline]
    pub(crate) fn bound_through(&self, mut b: usize, hi: DocId) -> f32 {
        let r = &self.reader;
        let (nf, nb) = (r.num_full_blocks(), r.num_blocks());
        let mut m = 0f32;
        while b < nb {
            if b < nf {
                let s = r.skip(b);
                m = m.max(self.weight.block_bound(&s));
                if r.block_last_doc(b) >= hi {
                    break;
                }
            } else {
                m = m.max(self.tail_max_score());
            }
            b += 1;
        }
        m
    }

    pub fn reader(&self) -> &PostingsReader {
        &self.reader
    }

    pub fn weight(&self) -> &Bm25Weight {
        &self.weight
    }

    pub fn norms(&self) -> &Norms {
        &self.norms
    }

    /// True before the first `advance`/`seek`.
    pub fn is_unstarted(&self) -> bool {
        !self.started
    }

    /// Mark as consumed (after a bulk collection drove the postings).
    pub fn advance_to_end(&mut self) {
        self.started = true;
        self.current = TERMINATED;
    }

    /// Every block, window and staircase bound, and every skip, assumes a
    /// finite weight >= 0 and a score monotonic in tf and length. Parsing
    /// rejects negative boosts and settings out-of-range `k1` / `b`, but a
    /// weight can still get here (e.g. from a function's factor): such a
    /// scorer is scored exhaustively.
    pub fn prunable(&self) -> bool {
        let w = self.weight.weight();
        w.is_finite() && w >= 0.0 && self.weight.monotonic()
    }

    /// Every hit scores the same (no norms, every tf = 1).
    pub fn is_constant(&self) -> bool {
        self.norms.is_unit() && self.reader.total_term_freq() == self.reader.doc_freq() as u64
    }

    #[inline(always)]
    fn norm_byte(&self, d: DocId) -> u8 {
        self.norms.byte(d)
    }
}

impl DocSet for Bm25TermScorer {
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

    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        if self.started {
            return None;
        }
        self.started = true;
        self.current = TERMINATED;
        Some(first_k(&self.reader, k, deleted))
    }
}

impl Scorer for Bm25TermScorer {
    fn max_score(&self) -> Option<f32> {
        // No bound holds for a negative weight or non-monotonic BM25
        // parameters (`prunable`): the MaxScore and dis_max bounds of the
        // clauses above then stay unknown too.
        if !self.prunable() {
            return None;
        }
        // The largest block-max impact bound, and the tail scored exactly.
        let m = self.bound_through(0, TERMINATED);
        (m.is_finite() && m >= 0.0).then_some(m)
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        if !self.prunable() {
            return None;
        }
        let from = if self.started { self.current } else { 0 };
        if from == TERMINATED || from > up_to {
            return Some(0.0);
        }
        // The blocks from the one holding the current doc.
        let b = first_block_reaching(&self.reader, 0, from);
        let m = self.bound_through(b, up_to);
        (m.is_finite() && m >= 0.0).then_some(m)
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
        // Constant scores: the first live docs (`collect_top_k_bulk`).
        if self.started || self.is_constant() || !self.prunable() {
            return None;
        }
        self.started = true;
        self.current = TERMINATED;
        Some(term_top_k_lim(&self.reader, &self.weight, &self.norms, k, deleted, count_limit, floor))
    }

    fn score_upper_bound(&self) -> Option<f32> {
        // No frequency or norm variation: the bound is the exact score.
        if !self.is_constant() { return None; }
        let score = self.weight.score_norm(1, 1);
        (score.is_finite() && score >= 0.0).then_some(score)
    }

    #[inline]
    fn score(&mut self) -> f32 {
        let tf = self.reader.term_freq().unwrap_or(1);
        self.weight.score_norm(tf, self.norm_byte(self.current))
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        if self.started || (!self.prunable() && !self.is_constant()) {
            return None;
        }
        self.started = true;
        self.current = TERMINATED;
        if self.is_constant() {
            let score = self.weight.score_norm(1, 1);
            let (docs, n) = first_k(&self.reader, k, deleted);
            return Some((docs.into_iter().map(|doc| ScoredDoc { doc, score }).collect(), n));
        }
        Some(term_top_k(&self.reader, &self.weight, &self.norms, k, deleted))
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, min_competitive: f32) -> Option<(Vec<ScoredDoc>, u64)> {
        if self.started {
            return None;
        }
        if self.is_constant() {
            return self.collect_top_k_bulk(k, deleted);
        }
        if !self.prunable() {
            return None;
        }
        self.started = true;
        self.current = TERMINATED;
        Some(term_top_k_min(&self.reader, &self.weight, &self.norms, k, deleted, min_competitive))
    }

    fn as_bm25_term(&mut self) -> Option<&mut Bm25TermScorer> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Single term
// ---------------------------------------------------------------------------

/// First `k` live docs of a list and the number of live docs.
pub fn first_k(r: &PostingsReader, k: usize, deleted: Option<&[u64]>) -> (Vec<DocId>, u64) {
    let mut docs = [0u32; BLOCK_DOCS];
    let mut tfs = [0u32; BLOCK_DOCS];
    let mut out = Vec::with_capacity(k.min(BLOCK_DOCS));
    match deleted {
        None => {
            let mut b = 0;
            while out.len() < k && b < r.num_blocks() {
                let n = if b < r.num_full_blocks() {
                    r.decode_docs(b, &mut docs);
                    BLOCK_DOCS
                } else {
                    r.decode_tail(&mut docs, &mut tfs)
                };
                let take = n.min(k - out.len());
                out.extend_from_slice(&docs[..take]);
                b += 1;
            }
            (out, r.doc_freq() as u64)
        }
        Some(del) => {
            let mut count = 0u64;
            for b in 0..r.num_blocks() {
                let n = if b < r.num_full_blocks() {
                    r.decode_docs(b, &mut docs);
                    BLOCK_DOCS
                } else {
                    r.decode_tail(&mut docs, &mut tfs)
                };
                for &d in &docs[..n] {
                    if !is_deleted(del, d) {
                        count += 1;
                        if out.len() < k {
                            out.push(d);
                        }
                    }
                }
            }
            (out, count)
        }
    }
}

/// Score `n` docs of a decoded block into `scores`.
#[inline(always)]
fn score_block(
    w: &Bm25Weight,
    norms: &[u8],
    docs: &[u32; BLOCK_DOCS],
    tfs: &[u32; BLOCK_DOCS],
    tf_one: bool,
    n: usize,
    scores: &mut [f32; BLOCK_DOCS],
) {
    let cache = w.norm_cache();
    let mut c = [0f32; BLOCK_DOCS];
    if norms.is_empty() {
        c[..n].fill(cache[1]);
    } else if n > 0 && (docs[n - 1] as usize) < norms.len() {
        for i in 0..n {
            // SAFETY: docs ascending and docs[n-1] < norms.len().
            let nb = unsafe { *norms.get_unchecked(docs[i] as usize) };
            c[i] = cache[nb as usize];
        }
    } else {
        for i in 0..n {
            c[i] = cache[norms.get(docs[i] as usize).copied().unwrap_or(1) as usize];
        }
    }
    // Lucene's BM25Scorer: weight - weight / (1 + tf * inv), inv = cache[norm].
    let wt = w.weight();
    if tf_one {
        for i in 0..n {
            scores[i] = wt - wt / (1.0 + c[i]);
        }
    } else {
        for i in 0..n {
            let tf = tfs[i] as f32;
            scores[i] = wt - wt / (1.0 + tf * c[i]);
        }
    }
}

/// Max of `scores[..n]` (4 lanes so it vectorizes without fast-math).
#[inline(always)]
fn block_max(scores: &[f32; BLOCK_DOCS], n: usize) -> f32 {
    let mut m = [f32::NEG_INFINITY; 4];
    let full = n / 4 * 4;
    for i in (0..full).step_by(4) {
        for l in 0..4 {
            m[l] = m[l].max(scores[i + l]);
        }
    }
    let mut r = m[0].max(m[1]).max(m[2].max(m[3]));
    for &s in &scores[full..n] {
        r = r.max(s);
    }
    r
}

#[inline(always)]
fn collect_block(top: &mut TopK, docs: &[u32; BLOCK_DOCS], scores: &[f32; BLOCK_DOCS], n: usize, del: Option<&[u64]>) -> u64 {
    match del {
        None => {
            if block_max(scores, n) <= top.threshold() {
                return n as u64;
            }
            for i in 0..n {
                top.offer(docs[i], scores[i]);
            }
            n as u64
        }
        Some(del) => {
            let mut live = 0u64;
            for i in 0..n {
                if !is_deleted(del, docs[i]) {
                    live += 1;
                    top.offer(docs[i], scores[i]);
                }
            }
            live
        }
    }
}

/// Block-at-a-time single-term top-k + exact count with block-max skipping.
pub fn term_top_k(r: &PostingsReader, w: &Bm25Weight, norms: &Norms, k: usize, del: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
    term_top_k_min(r, w, norms, k, del, f32::NEG_INFINITY)
}

/// [`term_top_k`] admitting only hits scoring above `floor`.
pub fn term_top_k_min(r: &PostingsReader, w: &Bm25Weight, norms: &Norms, k: usize, del: Option<&[u64]>, floor: f32) -> (Vec<ScoredDoc>, u64) {
    term_top_k_lim(r, w, norms, k, del, u64::MAX, &Floor::new(floor, None))
}

/// [`term_top_k_min`] whose count only has to be exact up to `count_limit`
/// (then it is a lower bound): past it, non-competitive blocks of a leaf
/// with deletions are no longer decoded just to count their live docs.
pub fn term_top_k_lim(r: &PostingsReader, w: &Bm25Weight, norms: &Norms, k: usize, del: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> (Vec<ScoredDoc>, u64) {
    let nb = norms.bytes();
    let mut top = TopK::with_floor(k, floor.get());
    let mut docs = [0u32; BLOCK_DOCS];
    let mut tfs = [0u32; BLOCK_DOCS];
    let mut scores = [0f32; BLOCK_DOCS];
    let mut count = 0u64;
    for b in 0..r.num_full_blocks() {
        if b % 16 == 0 {
            top.sync(floor);
        }
        let s = r.skip(b);
        if w.block_bound(&s) <= top.threshold() {
            // Non-competitive block: count only.
            match del {
                None => count += BLOCK_DOCS as u64,
                Some(del) if count < count_limit => {
                    r.decode_docs(b, &mut docs);
                    count += docs.iter().filter(|&&d| !is_deleted(del, d)).count() as u64;
                }
                Some(_) => {}
            }
            continue;
        }
        let tf_one = r.decode_block(b, &mut docs, &mut tfs);
        score_block(w, nb, &docs, &tfs, tf_one, BLOCK_DOCS, &mut scores);
        count += collect_block(&mut top, &docs, &scores, BLOCK_DOCS, del);
    }
    let n = r.tail_len();
    if n > 0 {
        r.decode_tail(&mut docs, &mut tfs);
        score_block(w, nb, &docs, &tfs, false, n, &mut scores);
        count += collect_block(&mut top, &docs, &scores, n, del);
    }
    top.sync(floor);
    (top.into_sorted(), count)
}

// ---------------------------------------------------------------------------
// Block cursor (AND / OR)
// ---------------------------------------------------------------------------

const NO_BLOCK: usize = usize::MAX;
/// `docs` has 16 sentinel slots past the block.
const BUF: usize = BLOCK_DOCS + 16;

struct BlockCursor<'a> {
    r: &'a PostingsReader,
    blk: usize,
    n: usize,
    pos: usize,
    docs: [u32; BUF],
    /// Tail tfs (full-block tfs are read by random access).
    tail_tfs: [u32; BLOCK_DOCS],
}

impl<'a> BlockCursor<'a> {
    fn new(r: &'a PostingsReader) -> Self {
        Self { r, blk: NO_BLOCK, n: 0, pos: 0, docs: [u32::MAX; BUF], tail_tfs: [1; BLOCK_DOCS] }
    }

    #[inline(always)]
    fn num_blocks(&self) -> usize {
        self.r.num_blocks()
    }

    #[inline(always)]
    fn last_doc(&self, b: usize) -> u32 {
        self.r.block_last_doc(b)
    }

    #[inline]
    fn load(&mut self, b: usize) {
        // Decode straight into the cursor's buffer (no staging copy).
        let d: &mut [u32; BLOCK_DOCS] = (&mut self.docs[..BLOCK_DOCS]).try_into().unwrap();
        if b < self.r.num_full_blocks() {
            self.r.decode_docs(b, d);
            self.n = BLOCK_DOCS;
        } else {
            self.n = self.r.decode_tail(d, &mut self.tail_tfs);
        }
        self.docs[self.n..].fill(u32::MAX);
        self.blk = b;
        self.pos = 0;
    }

    /// tf of doc `i` of block `b` (b must be the current block if it is the
    /// tail).
    #[inline(always)]
    fn tf(&self, b: usize, i: usize) -> u32 {
        if b < self.r.num_full_blocks() { self.r.tf_at(b, i) } else { self.tail_tfs[i] }
    }
}

#[inline(always)]
fn lower_bound(docs: &[u32; BUF], mut lo: usize, n: usize, target: u32) -> usize {
    let mut len = n - lo;
    while len > 1 {
        let half = len / 2;
        lo = if docs[lo + half - 1] < target { lo + half } else { lo };
        len -= half;
    }
    if lo < n && docs[lo] < target {
        lo += 1;
    }
    lo
}

#[inline(always)]
fn gallop(docs: &[u32; BUF], lo: usize, n: usize, target: u32) -> usize {
    let mut step = 1;
    let mut hi = lo;
    let mut prev = lo;
    while hi < n && docs[hi] < target {
        prev = hi + 1;
        hi += step;
        step <<= 1;
    }
    lower_bound(docs, prev, hi.min(n), target)
}

// ---------------------------------------------------------------------------
// AND
// ---------------------------------------------------------------------------

/// Max clauses of the block conjunction.
pub const MAX_CONJ: usize = 8;

thread_local! {
    static CANDS: std::cell::RefCell<Option<Box<Cands>>> = const { std::cell::RefCell::new(None) };
}

struct Cands {
    doc: [u32; BLOCK_DOCS],
    /// Per list: block and index of the candidate (for the tf lookup).
    blk: [[u32; BLOCK_DOCS]; MAX_CONJ],
    idx: [[u8; BLOCK_DOCS]; MAX_CONJ],
    n: usize,
}

/// Filter candidates through list `j`. Returns false when the list is
/// exhausted (nothing after can match).
#[inline]
fn filter(c: &mut Cands, o: &mut BlockCursor<'_>, j: usize) -> bool {
    let nb = o.num_blocks();
    let mut out = 0usize;
    let mut i = 0usize;
    let n = c.n;
    while i < n {
        let cd = c.doc[i];
        if o.blk == NO_BLOCK || cd > o.docs[o.n - 1] {
            let mut b = if o.blk == NO_BLOCK { 0 } else { o.blk + 1 };
            while b < nb && o.last_doc(b) < cd {
                b += 1;
            }
            if b >= nb {
                c.n = out;
                return false;
            }
            o.load(b);
            if cd > o.docs[o.n - 1] {
                // Only possible for the tail (sentinel last_doc): no doc of
                // this list is >= cd.
                c.n = out;
                return false;
            }
        }
        let last = o.docs[o.n - 1];
        let mut e = i + 1;
        while e < n && c.doc[e] <= last {
            e += 1;
        }
        let bn = o.n;
        let blk = o.blk as u32;
        let mut p = o.pos;
        for k in i..e {
            let a = c.doc[k];
            p = gallop(&o.docs, p, bn, a);
            let hit = o.docs[p] == a;
            c.doc[out] = a;
            for jj in 0..j {
                c.blk[jj][out] = c.blk[jj][k];
                c.idx[jj][out] = c.idx[jj][k];
            }
            c.blk[j][out] = blk;
            c.idx[j][out] = p as u8;
            out += hit as usize;
        }
        o.pos = p.min(bn - 1);
        i = e;
    }
    c.n = out;
    true
}

/// Block-at-a-time conjunction of BM25 term scorers. Scores are summed in
/// clause order (as the per-doc intersection does).
pub fn conjunction(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
    conjunction_min(terms, k, del, f32::NEG_INFINITY)
}

/// [`conjunction`] admitting only hits scoring above `floor`.
pub fn conjunction_min(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>, floor: f32) -> (Vec<ScoredDoc>, u64) {
    conjunction_lim(terms, k, del, u64::MAX, &Floor::new(floor, None))
}

/// [`conjunction_min`] whose count only has to be exact up to
/// `count_limit` (then it is a lower bound): past it, a lead block is
/// skipped when the clauses' block-max bounds over its doc range, summed in
/// clause order, cannot beat the threshold (Lucene's
/// `BlockMaxConjunctionScorer`).
pub fn conjunction_lim(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> (Vec<ScoredDoc>, u64) {
    let nt = terms.len();
    assert!((1..=MAX_CONJ).contains(&nt));
    let mut order: Vec<usize> = (0..nt).collect();
    order.sort_by_key(|&i| terms[i].reader.doc_freq());
    let mut cursors: Vec<BlockCursor<'_>> = order.iter().map(|&i| BlockCursor::new(&terms[i].reader)).collect();
    // list position (in `order`) of each clause
    let mut slot = vec![0usize; nt];
    for (p, &i) in order.iter().enumerate() {
        slot[i] = p;
    }
    let norms = &terms[0].norms;
    let mut top = TopK::with_floor(k, floor.get());
    let mut total = 0u64;
    let lead_r = &terms[order[0]].reader;
    // Per clause: the first block that may hold docs of the current lead
    // block (only advanced once counting is done).
    let mut from = vec![0usize; nt];
    // Candidate scratch (~5.6 KB) is reused per thread instead of being
    // allocated and zeroed for every leaf.
    let mut c = CANDS.with(|x| x.borrow_mut().take()).unwrap_or_else(|| {
        Box::new(Cands { doc: [0; BLOCK_DOCS], blk: [[0; BLOCK_DOCS]; MAX_CONJ], idx: [[0; BLOCK_DOCS]; MAX_CONJ], n: 0 })
    });
    let (lead, others) = cursors.split_first_mut().unwrap();
    let mut tf = [0u32; MAX_CONJ];
    'blocks: for lb in 0..lead.num_blocks() {
        if lb % 16 == 0 {
            top.sync(floor);
        }
        if total >= count_limit && top.threshold() > f32::NEG_INFINITY {
            let lo = if lb == 0 { 0 } else { lead_r.block_last_doc(lb - 1).saturating_add(1) };
            let hi = lead_r.block_last_doc(lb);
            // f32 sums are monotone, so the clause-order sum of per-clause
            // bounds bounds every doc's clause-order score.
            let mut bound = 0f32;
            for (i, t) in terms.iter().enumerate() {
                from[i] = first_block_reaching(&t.reader, from[i], lo);
                bound += t.bound_through(from[i], hi);
            }
            if bound <= top.threshold() {
                continue;
            }
        }
        lead.load(lb);
        c.n = lead.n;
        c.doc.copy_from_slice(&lead.docs[..BLOCK_DOCS]);
        for kk in 0..c.n {
            c.blk[0][kk] = lb as u32;
            c.idx[0][kk] = kk as u8;
        }
        let mut more = true;
        for (j, o) in others.iter_mut().enumerate() {
            more &= filter(&mut c, o, j + 1);
            if c.n == 0 {
                break;
            }
        }
        for kk in 0..c.n {
            let d = c.doc[kk];
            if let Some(del) = del {
                if is_deleted(del, d) {
                    continue;
                }
            }
            total += 1;
            for (p, t) in tf.iter_mut().enumerate().take(nt) {
                let cur = if p == 0 { &*lead } else { &others[p - 1] };
                *t = cur.tf(c.blk[p][kk] as usize, c.idx[p][kk] as usize);
            }
            let nb = norms.byte(d);
            let mut s = 0f32;
            for (i, t) in terms.iter().enumerate() {
                s += t.weight.score_norm(tf[slot[i]], nb);
            }
            top.offer(d, s);
        }
        if !more {
            break 'blocks;
        }
    }
    CANDS.with(|x| *x.borrow_mut() = Some(c));
    top.sync(floor);
    (top.into_sorted(), total)
}

/// The first block at or after `from` that may hold docs >= `target`: the
/// first full block whose last doc reaches it, else the tail's index.
#[inline]
fn first_block_reaching(r: &PostingsReader, from: usize, target: DocId) -> usize {
    let (mut lo, mut hi) = (from, r.num_full_blocks());
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

// ---------------------------------------------------------------------------
// OR
// ---------------------------------------------------------------------------

const W: usize = 4096;
const WW: usize = W / 64;

struct Clause<'a> {
    cur: BlockCursor<'a>,
    /// Term index (sum order).
    id: usize,
    sc: [f32; BLOCK_DOCS],
    sc_ok: bool,
    done: bool,
}

impl Clause<'_> {
    /// Move to the first doc >= `target`, forward only, decoding only the
    /// block that holds it. False once exhausted.
    fn seek(&mut self, target: DocId) -> bool {
        if self.done {
            return false;
        }
        let c = &mut self.cur;
        if c.docs[c.n - 1] < target {
            let b = first_block_reaching(c.r, c.blk + 1, target);
            if b >= c.num_blocks() {
                self.done = true;
                return false;
            }
            c.load(b);
            self.sc_ok = false;
            if c.docs[c.n - 1] < target {
                self.done = true;
                return false;
            }
        }
        c.pos = gallop(&c.docs, c.pos, c.n, target);
        true
    }
}

/// Norm byte of `d` in a norms column (1 for unit norms or docs past it,
/// as [`Norms::byte`]).
#[inline(always)]
fn norm_of(nb: &[u8], d: DocId) -> u8 {
    nb.get(d as usize).copied().unwrap_or(1)
}

/// A dis_max's score from its clauses' scores (0 for clauses that do not
/// match), computed exactly as [`crate::query::query::DisMaxScorer`] does.
#[inline]
fn group_score(parts: &[f32], tie: f32) -> f32 {
    let max = parts.iter().copied().fold(0.0f32, f32::max);
    let sum: f32 = parts.iter().sum();
    max + tie * (sum - max)
}

/// Disjunction of BM25 term scorers: inclusion–exclusion for two clauses
/// without deletions, MaxScore windows otherwise.
pub fn disjunction(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
    disjunction_min(terms, k, del, f32::NEG_INFINITY)
}

/// [`disjunction`] admitting only hits scoring above `floor`.
pub fn disjunction_min(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>, floor: f32) -> (Vec<ScoredDoc>, u64) {
    if terms.len() == 2 && del.is_none() {
        return or2_inclusion_exclusion_min(terms[0], terms[1], k, floor);
    }
    or_maxscore_min(terms, k, del, floor)
}

/// MaxScore-window disjunction (Lucene `MaxScoreBulkScorer`-like).
pub fn or_maxscore(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
    or_maxscore_min(terms, k, del, f32::NEG_INFINITY)
}

/// [`or_maxscore`] admitting only hits scoring above `floor`.
pub fn or_maxscore_min(terms: &[&Bm25TermScorer], k: usize, del: Option<&[u64]>, floor: f32) -> (Vec<ScoredDoc>, u64) {
    let items: Vec<Vec<TermGroup<'_>>> = terms.iter().map(|&t| vec![(0.0, vec![t])]).collect();
    or_groups(&items, k, del, u64::MAX, &Floor::new(floor, None))
}

/// A dis_max of BM25 terms (`max + tie * (sum - max)` of their scores); a
/// lone term is a group of one.
pub type TermGroup<'a> = (f32, Vec<&'a Bm25TermScorer>);

/// MaxScore-window disjunction (Lucene `MaxScoreBulkScorer`) of `items`,
/// each the sum of its term groups, e.g. a multi-field `query_string`: one
/// item per word, a dis_max of the word's term in every field, and nested
/// groups as items of several groups. Terms may read different norms.
/// Scores are summed exactly as the per-doc [`crate::query::boolean::Union`]
/// of those scorers does (item order, group order, term order), so hits
/// and ties match it bit for bit.
///
/// Windows partition the groups by their block-max bounds over the window:
/// groups whose bounds sum below the threshold are non-essential, so a doc
/// only they match cannot compete. Until `count_limit` live matches are
/// counted (by this leaf, or by the shard's leaves together, see
/// [`Floor::with_hits`]), windows span 4096 docs and non-essential terms
/// are still walked to count their docs. After that, outer windows end
/// with the current postings block of a clause that could compete in the
/// previous one (super-block maxima bound the others over long windows); a
/// group the other groups' bounds cannot make up for is required, and then
/// only the cheapest required group's docs are candidates. Inner windows
/// of up to 4096 docs start at the next walked doc, so non-essential
/// clauses are never walked: they are only probed (block skip, then
/// gallop) on candidates, required ones first, and only while the doc's
/// bound can still beat the threshold. The total is then a lower bound
/// past `count_limit`. Per candidate, the matched term slots are tracked,
/// so scoring costs O(matches), not O(terms).
pub fn or_groups(items: &[Vec<TermGroup<'_>>], k: usize, del: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> (Vec<ScoredDoc>, u64) {
    or_groups_in(items, k, del, None, 0.0, count_limit, floor)
}

/// The largest `f32` at or below `thr - add` (exact in `f64`): a bound
/// below it leaves `add + score` below `thr` (see [`or_groups_in`]).
pub(crate) fn shifted_threshold(thr: f32, add: f32) -> f32 {
    if add == 0.0 || !thr.is_finite() {
        return thr;
    }
    let d = thr as f64 - add as f64;
    let mut f = d as f32;
    if f as f64 > d {
        f = if f > 0.0 { f32::from_bits(f.to_bits() - 1) } else if f == 0.0 { -f32::from_bits(1) } else { f32::from_bits(f.to_bits() + 1) };
    }
    f
}

/// [`or_groups`] over the docs set in `keep` only (`None`: every doc), each
/// hit scoring `add + score` (`add`: a conjunction's constant clause, e.g.
/// a scored range in `bool.must`; the f32 sum the per-doc conjunction
/// takes). Counts, the top-k and its threshold are in those scores; the
/// clauses' bounds are compared with the threshold less `add`.
pub(crate) fn or_groups_in(items: &[Vec<TermGroup<'_>>], k: usize, del: Option<&[u64]>, keep: Option<&[u64]>, add: f32, count_limit: u64, floor: &Floor<'_>) -> (Vec<ScoredDoc>, u64) {
    let mut terms: Vec<&Bm25TermScorer> = Vec::new();
    let mut group_of: Vec<usize> = Vec::new();
    let mut item_of: Vec<usize> = Vec::new();
    let mut tie: Vec<f32> = Vec::new();
    let mut gstart: Vec<usize> = vec![0];
    for (it, item) in items.iter().enumerate() {
        for (t, ts) in item {
            for &x in ts {
                terms.push(x);
                group_of.push(tie.len());
            }
            tie.push(*t);
            item_of.push(it);
            gstart.push(terms.len());
        }
    }
    let (nt, ng) = (terms.len(), tie.len());
    let mut top = TopK::with_floor(k, floor.get());
    if nt == 0 {
        return (top.into_sorted(), 0);
    }
    // One term per group and one group per item: a plain sum.
    let flat = items.iter().all(|it| it.len() == 1 && it[0].1.len() == 1);
    let single: Vec<bool> = (0..ng).map(|g| gstart[g + 1] - gstart[g] == 1).collect();
    // Postings of each group: the cheapest required group leads.
    let gcost: Vec<u64> = (0..ng).map(|g| (gstart[g]..gstart[g + 1]).map(|j| terms[j].reader.doc_freq() as u64).sum()).collect();
    let nbs: Vec<&[u8]> = terms.iter().map(|t| t.norms.bytes()).collect();
    let mut cls: Vec<Clause<'_>> = terms
        .iter()
        .enumerate()
        .map(|(i, t)| Clause { cur: BlockCursor::new(&t.reader), id: i, sc: [0.0; BLOCK_DOCS], sc_ok: false, done: t.reader.num_blocks() == 0 })
        .collect();
    for c in cls.iter_mut() {
        if !c.done {
            c.cur.load(0);
        }
    }
    // Per-doc, per-term partial scores (0 = no match) and, per doc, the
    // terms it matched (`tw` words), so per-candidate work is O(matches).
    // Both live in a per-thread scratch that is all zeros between calls.
    let tw = nt.div_ceil(64);
    let (mut parts_buf, mut touched_buf) = OR_SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if parts_buf.len() < W * nt {
        parts_buf.resize(W * nt, 0.0);
    }
    if touched_buf.len() < W * tw {
        touched_buf.resize(W * tw, 0);
    }
    let parts = &mut parts_buf[..W * nt];
    let touched = &mut touched_buf[..W * tw];
    let mut mask = [0u64; WW];
    let mut mask_all = [0u64; WW];
    let mut total = 0u64;
    let mut published = 0u64;
    let mut tfs = [0u32; BLOCK_DOCS];
    let mut wb = vec![0f32; nt];
    let mut gb = vec![0f64; ng];
    let mut gord: Vec<usize> = (0..ng).collect();
    let mut prefix = vec![0f64; ng + 1];
    let mut walk = vec![true; ng];
    let mut req = vec![false; ng];
    let mut probe: Vec<usize> = Vec::with_capacity(ng);
    let mut sup: Vec<Vec<f32>> = vec![Vec::new(); nt];
    // Pruning (counting done, threshold known) runs Lucene's outer windows:
    // `[pos, oend)` ends with the current block of a clause that was
    // essential before, bounds and partition hold for the whole window,
    // and only walked clauses' docs start inner windows, so stretches only
    // non-essential (frequent) clauses match are skipped without being
    // decoded. The partition is renewed when the threshold reaches `renew`.
    let mut pos: DocId = 0;
    let mut oend: DocId = 0;
    let mut renew = f64::INFINITY;
    let mut ess = vec![true; ng];
    let mut rest = 0f64;
    let mut lead = usize::MAX;
    let mut ne = 0;
    // Clause-level partition (see below): walked / window-sizing clauses,
    // the clauses probed on candidates, and the greedy split's state.
    let mut walk_c = vec![true; nt];
    let mut ess_c = vec![true; nt];
    let mut probe_c: Vec<usize> = Vec::with_capacity(nt);
    let mut clause_mode = false;
    let mut in_n = vec![false; nt];
    let (mut gmx, mut gsm) = (vec![0f64; ng], vec![0f64; ng]);
    let mut corder: Vec<usize> = (0..nt).collect();
    let df: Vec<u64> = terms.iter().map(|t| (t.reader.doc_freq() as u64).max(1)).collect();
    let clause_split = single.iter().any(|s| !s) && tie.iter().all(|&t| t <= 1.0);
    // One tie-0 dis_max: a doc competes only through one field's score.
    let lone_dismax = ng == 1 && tie[0] == 0.0;
    // The summed window bounds of every group (an upper bound of any doc's
    // score in the outer window).
    let mut s_all = f64::INFINITY;
    // Diagnosis counters ([`crate::query::stats`]), added once at the end.
    let started = crate::query::stats::start();
    // count windows, partitions, essential groups, required, skipped, inner windows, blocks, probes, scored, clause partitions, essential clauses
    let mut st = [0u64; 12];
    loop {
        top.sync(floor);
        let thr = shifted_threshold(top.threshold(), add);
        let counted = total >= count_limit || floor.hits_counted() >= count_limit;
        // Nothing to collect (`size: 0`): once counting is done, the rest
        // of the walk could only count past the limit.
        if k == 0 && counted {
            break;
        }
        let prune = counted && thr > f32::NEG_INFINITY;
        let (base, end);
        if prune {
            if pos == TERMINATED {
                break;
            }
            if pos >= oend || below(renew, thr) {
                if cls.iter().all(|c| c.done) {
                    break;
                }
                oend = outer_end(&cls, &ess_c, pos);
                for (j, c) in cls.iter().enumerate() {
                    let b = block_reaching(c, pos);
                    wb[j] = window_bound(terms[j], &mut sup[j], b, oend - 1);
                }
                ne = partition(&wb, &gstart, &tie, &mut gb, &mut gord, &mut prefix, thr);
                s_all = prefix[ng];
                st[1] += 1;
                st[2] += (ng - ne) as u64;
                // A group the others' bounds cannot compensate is
                // required; the cheapest leads and only its docs are
                // candidates (MaxScoreBulkScorer's conjunction mode).
                req.fill(false);
                lead = usize::MAX;
                renew = if ne < ng { prefix[ne + 1] } else { f64::INFINITY };
                for &g in &gord[ne..] {
                    if below(prefix[ng] - gb[g], thr) {
                        req[g] = true;
                        if lead == usize::MAX || gcost[g] < gcost[lead] {
                            lead = g;
                        }
                    } else {
                        renew = renew.min(prefix[ng] - gb[g]);
                    }
                }
                // Size the next window by the groups that can compete
                // (at least the best one).
                ess.fill(false);
                for &g in &gord[ne.min(ng - 1)..] {
                    ess[g] = true;
                }
                // Clause-level split: with dis_max groups over several
                // fields, a group's bound is its best field's (a short
                // title field's, for a common word), though most of its
                // docs match only weak fields (the word in long articles).
                // Non-essential clauses are picked greedily, by ascending
                // bound per posting (the most postings left unwalked per
                // unit of the bound budget first), while the summed group
                // bounds of the docs they alone match stay below the
                // threshold, if that leaves more postings unwalked than the
                // group split; the strong clauses of a common
                // word are then walked and its weak ones only probed. Lucene
                // splits at the clause (here: group) level only.
                clause_mode = false;
                if clause_split && ne < ng {
                    corder.sort_unstable_by(|&a, &b| (wb[a] as f64 * df[b] as f64).total_cmp(&(wb[b] as f64 * df[a] as f64)).then(wb[a].total_cmp(&wb[b])));
                    gmx.fill(0.0);
                    gsm.fill(0.0);
                    let gbound = |mx: f64, sm: f64, t: f64| mx + t * (sm - mx);
                    let mut sum_n = 0f64;
                    for &j in &corder {
                        let g = group_of[j];
                        let (w, t) = (wb[j] as f64, tie[g] as f64);
                        let (mx, sm) = (gmx[g].max(w), gsm[g] + w);
                        let d = gbound(mx, sm, t) - gbound(gmx[g], gsm[g], t);
                        in_n[j] = below(sum_n + d, thr);
                        if in_n[j] {
                            sum_n += d;
                            (gmx[g], gsm[g]) = (mx, sm);
                        }
                    }
                    // Postings the group split would walk: the lead group's
                    // (conjunction mode), else the essential groups'.
                    let walked: u64 = (0..nt).filter(|&j| !in_n[j]).map(|j| df[j]).sum();
                    let walked_groups: u64 = if lead != usize::MAX {
                        (gstart[lead]..gstart[lead + 1]).map(|j| df[j]).sum()
                    } else {
                        gord[ne..].iter().flat_map(|&g| gstart[g]..gstart[g + 1]).map(|j| df[j]).sum()
                    };
                    if walked < walked_groups {
                        clause_mode = true;
                        st[9] += 1;
                        for j in 0..nt {
                            walk_c[j] = !in_n[j];
                            ess_c[j] = walk_c[j];
                        }
                        // The next split: the cheapest walked clause joins.
                        if let Some(m) = (0..nt).filter(|&j| !in_n[j]).map(|j| wb[j] as f64).min_by(f64::total_cmp) {
                            renew = renew.min(sum_n + m);
                        }
                        // Probed by descending bound.
                        probe_c.clear();
                        probe_c.extend((0..nt).filter(|&j| in_n[j]));
                        probe_c.sort_unstable_by(|&a, &b| wb[b].total_cmp(&wb[a]));
                        rest = probe_c.iter().map(|&j| wb[j] as f64).sum();
                    }
                }
                if !clause_mode {
                    for j in 0..nt {
                        ess_c[j] = ess[group_of[j]];
                    }
                }
                // Bounds over a wider window also bound a narrower one.
                oend = oend.min(outer_end(&cls, &ess_c, pos));
                st[3] += (lead != usize::MAX) as u64;
                if ne == ng {
                    // Nothing in the window can beat the threshold.
                    st[4] += 1;
                    pos = oend;
                    oend = 0;
                    continue;
                }
                if !clause_mode {
                    walk.fill(lead == usize::MAX);
                    if lead == usize::MAX {
                        for &g in &gord[..ne] {
                            walk[g] = false;
                        }
                    } else {
                        walk[lead] = true;
                    }
                    for j in 0..nt {
                        walk_c[j] = walk[group_of[j]];
                    }
                    // Probed on candidates: the other required groups first
                    // (cheapest first; a doc missing one cannot compete),
                    // then the rest by descending bound.
                    probe.clear();
                    probe.extend((0..ng).filter(|&g| req[g] && g != lead));
                    probe.sort_unstable_by_key(|&g| gcost[g]);
                    probe.extend(gord.iter().rev().copied().filter(|&g| !walk[g] && !req[g]));
                    rest = probe.iter().map(|&g| gb[g]).sum();
                }
                st[10] += walk_c.iter().filter(|&&x| x).count() as u64;
                // Clauses not walked so far may lag behind the window.
                for c in cls.iter_mut() {
                    if walk_c[c.id] {
                        c.seek(pos);
                    }
                }
            }
            let mut min = TERMINATED;
            for c in &cls {
                if !c.done && walk_c[c.id] {
                    min = min.min(c.cur.docs[c.cur.pos]);
                }
            }
            if min >= oend {
                pos = oend;
                continue;
            }
            base = min & !63;
            end = oend.min(base.saturating_add(W as u32));
            st[5] += 1;
        } else {
            let mut min = TERMINATED;
            for c in &cls {
                if !c.done {
                    min = min.min(c.cur.docs[c.cur.pos]);
                }
            }
            if min == TERMINATED {
                break;
            }
            base = min & !63;
            end = base.saturating_add(W as u32);
            st[0] += 1;
            // While counting, every clause is walked: the first `ne`
            // groups of `gord` are non-essential in this window and their
            // terms only set count bits (and add to docs others hit).
            ne = 0;
            if thr > f32::NEG_INFINITY {
                for (j, c) in cls.iter().enumerate() {
                    wb[j] = if c.done { 0.0 } else { terms[j].bound_through(c.cur.blk, end - 1) };
                }
                ne = partition(&wb, &gstart, &tie, &mut gb, &mut gord, &mut prefix, thr);
            }
            walk.fill(true);
            for &g in &gord[..ne] {
                walk[g] = false;
            }
            for j in 0..nt {
                walk_c[j] = walk[group_of[j]];
            }
        }
        let nw = ((end - base) as usize).div_ceil(64);
        // Walked terms first (they set `mask`), then, while still
        // counting, the non-essential ones for the count.
        for pass in 0..2 {
            for c in cls.iter_mut() {
                let walked = walk_c[c.id];
                if c.done || (pass == 0) != walked || (!walked && counted) {
                    continue;
                }
                let id = c.id;
                let (tword, tbit) = (id >> 6, 1u64 << (id & 63));
                let w = &terms[id].weight;
                let nb = nbs[id];
                // Block-max skipping of a walked clause while pruning: a
                // block whose bound, with every other clause at its window
                // bound, cannot beat the threshold holds no competitive doc
                // through this clause (Lucene's impacts-driven skipping).
                // A doc of it that another walked clause makes a candidate
                // misses this clause's part, but cannot compete either. For
                // a lone tie-0 dis_max a doc competes only through a field
                // scoring above the threshold on its own, so the block's
                // own bound decides, beyond the window too.
                let skip_cut = if prune && walked {
                    let g = group_of[id];
                    let (a, e) = (gstart[g], gstart[g + 1]);
                    let t = tie[g] as f64;
                    let others = s_all - gb[g];
                    Some(move |bb: f32, wb: &[f32]| -> f64 {
                        if lone_dismax {
                            return bb as f64;
                        }
                        let (mut mx, mut sm) = (0f64, 0f64);
                        for j in a..e {
                            let v = if j == id { bb as f64 } else { wb[j] as f64 };
                            mx = mx.max(v);
                            sm += v;
                        }
                        // The group bound of `partition`.
                        others + if t <= 1.0 { mx + t * (sm - mx) } else { mx + t * sm }
                    })
                } else {
                    None
                };
                loop {
                    if let Some(cut) = &skip_cut {
                        let r = c.cur.r;
                        let (nf, nbk) = (r.num_full_blocks(), r.num_blocks());
                        let mut b = c.cur.blk;
                        while b < nbk {
                            let bb = if b < nf { w.block_bound(&r.skip(b)) } else { terms[id].tail_max_score() };
                            // Window bounds hold inside the outer window only.
                            let inside = lone_dismax || (b < nf && r.block_last_doc(b) < oend);
                            if !inside || !below(cut(bb, &wb), thr) {
                                break;
                            }
                            b += 1;
                        }
                        if b != c.cur.blk {
                            st[11] += (b - c.cur.blk) as u64;
                            if b >= nbk {
                                c.done = true;
                                break;
                            }
                            c.cur.load(b);
                            c.sc_ok = false;
                        }
                    }
                    let n = c.cur.n;
                    let mut p = c.cur.pos;
                    let stop = if c.cur.docs[n - 1] < end { n } else { p + c.cur.docs[p..n].partition_point(|&d| d < end) };
                    let b = c.cur.blk;
                    if walked {
                        if !c.sc_ok {
                            let tf_one = if b < c.cur.r.num_full_blocks() {
                                c.cur.r.decode_tfs(b, &mut tfs)
                            } else {
                                tfs[..n].copy_from_slice(&c.cur.tail_tfs[..n]);
                                false
                            };
                            let mut d = [0u32; BLOCK_DOCS];
                            d.copy_from_slice(&c.cur.docs[..BLOCK_DOCS]);
                            score_block(w, nb, &d, &tfs, tf_one, n, &mut c.sc);
                            c.sc_ok = true;
                        }
                        while p < stop {
                            let o = (c.cur.docs[p] - base) as usize;
                            parts[o * nt + id] = c.sc[p];
                            touched[o * tw + tword] |= tbit;
                            mask[o >> 6] |= 1u64 << (o & 63);
                            p += 1;
                        }
                    } else {
                        while p < stop {
                            let d = c.cur.docs[p];
                            let o = (d - base) as usize;
                            let bit = 1u64 << (o & 63);
                            mask_all[o >> 6] |= bit;
                            if mask[o >> 6] & bit != 0 {
                                parts[o * nt + id] = w.score_norm(c.cur.tf(b, p), norm_of(nb, d));
                                touched[o * tw + tword] |= tbit;
                            }
                            p += 1;
                        }
                    }
                    c.cur.pos = p;
                    if p < n {
                        break;
                    }
                    if c.cur.blk + 1 >= c.cur.num_blocks() {
                        c.done = true;
                        break;
                    }
                    let nxt = c.cur.blk + 1;
                    c.cur.load(nxt);
                    c.sc_ok = false;
                    st[6] += 1;
                }
            }
        }
        if prune && clause_mode {
            // Probe the clauses not walked on the candidates while the doc
            // can still compete: each unprobed clause adds at most its
            // bound to its group's score (tie <= 1).
            for wi in 0..nw {
                let mut m = mask[wi];
                while m != 0 {
                    let o = wi * 64 + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = base + o as u32;
                    let row = &mut parts[o * nt..(o + 1) * nt];
                    let tch = &mut touched[o * tw..(o + 1) * tw];
                    let mut bound = rest;
                    touched_groups(tch, row, &group_of, &single, &tie, |_, s| bound += s as f64);
                    for &j in &probe_c {
                        if below(bound, thr) {
                            break;
                        }
                        st[7] += 1;
                        bound -= wb[j] as f64;
                        let c = &mut cls[j];
                        if c.seek(d) && c.cur.docs[c.cur.pos] == d {
                            let g = group_of[j];
                            let (a, e) = (gstart[g], gstart[g + 1]);
                            let old = group_score(&row[a..e], tie[g]) as f64;
                            let (b, p) = (c.cur.blk, c.cur.pos);
                            row[j] = terms[j].weight.score_norm(c.cur.tf(b, p), norm_of(nbs[j], d));
                            tch[j >> 6] |= 1u64 << (j & 63);
                            bound += group_score(&row[a..e], tie[g]) as f64 - old;
                        }
                    }
                }
            }
        } else if prune && (ne > 0 || lead != usize::MAX) {
            // Probe the groups not walked on the candidates while the doc
            // can still compete. A doc abandoned early keeps a partial
            // score below the threshold, which the top-k then rejects.
            for wi in 0..nw {
                let mut m = mask[wi];
                while m != 0 {
                    let o = wi * 64 + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = base + o as u32;
                    let row = &mut parts[o * nt..(o + 1) * nt];
                    let tch = &mut touched[o * tw..(o + 1) * tw];
                    let mut bound = rest;
                    touched_groups(tch, row, &group_of, &single, &tie, |_, s| bound += s as f64);
                    for &g in &probe {
                        if below(bound, thr) {
                            break;
                        }
                        let (a, e) = (gstart[g], gstart[g + 1]);
                        let mut hit = false;
                        for id in a..e {
                            let c = &mut cls[id];
                            st[7] += 1;
                            if c.seek(d) && c.cur.docs[c.cur.pos] == d {
                                let (b, p) = (c.cur.blk, c.cur.pos);
                                row[id] = terms[id].weight.score_norm(c.cur.tf(b, p), norm_of(nbs[id], d));
                                tch[id >> 6] |= 1u64 << (id & 63);
                                hit = true;
                            }
                        }
                        if !hit {
                            if req[g] {
                                break;
                            }
                            bound -= gb[g];
                            continue;
                        }
                        bound += group_score(&row[a..e], tie[g]) as f64 - gb[g];
                    }
                }
            }
        }
        for wi in 0..nw {
            let me = mask[wi];
            let ma = mask_all[wi] | me;
            if ma == 0 {
                continue;
            }
            mask[wi] = 0;
            mask_all[wi] = 0;
            let w = (base as usize >> 6) + wi;
            let dead = del.map_or(0, |dl| dl.get(w).copied().unwrap_or(0)) | keep.map_or(0, |kp| !kp.get(w).copied().unwrap_or(0));
            total += (ma & !dead).count_ones() as u64;
            // Docs only non-essential terms match were never scored.
            let mut m = me;
            while m != 0 {
                let o = wi * 64 + m.trailing_zeros() as usize;
                m &= m - 1;
                let row = &mut parts[o * nt..(o + 1) * nt];
                let tch = &mut touched[o * tw..(o + 1) * tw];
                // The clause-order sum of the per-doc scorers: terms that
                // did not match add exact zeros, so they are skipped.
                let s = if flat {
                    let mut s = 0f32;
                    for_touched(tch, |j| s += row[j]);
                    s
                } else {
                    let (mut s, mut si, mut cur) = (0f32, 0f32, usize::MAX);
                    touched_groups(tch, row, &group_of, &single, &tie, |g, gs| {
                        if item_of[g] != cur {
                            s += si;
                            si = 0.0;
                            cur = item_of[g];
                        }
                        si += gs;
                    });
                    s + si
                };
                for_touched(tch, |j| row[j] = 0.0);
                tch.fill(0);
                if (dead >> (o & 63)) & 1 == 0 {
                    st[8] += 1;
                    top.offer(base + o as u32, if add == 0.0 { s } else { add + s });
                }
            }
        }
        if !counted && total - published >= COUNT_BATCH {
            floor.add_counted(total - published);
            published = total;
        }
        pos = end;
    }
    floor.add_counted(total - published);
    if crate::query::stats::ENABLED {
        use crate::query::stats::{add, C};
        let names = [
            C::or_count_windows, C::or_partitions, C::or_essential, C::or_required, C::or_skipped_windows, C::or_inner_windows, C::or_blocks,
            C::or_probes, C::or_scored, C::or_clause_partitions, C::or_essential_clauses, C::or_blocks_skipped,
        ];
        for (c, v) in names.into_iter().zip(st) {
            add(c, v);
        }
        add(C::or_calls, 1);
        crate::query::stats::add_since(C::or_ns, started);
        add(C::or_terms, nt as u64);
        add(C::or_groups, ng as u64);
        add(C::or_postings, terms.iter().map(|t| t.reader.doc_freq() as u64).sum());
        add(C::or_counted, total);
    }
    debug_assert!(parts.iter().all(|&x| x == 0.0) && touched.iter().all(|&x| x == 0));
    if parts_buf.len() <= SCRATCH_KEEP {
        OR_SCRATCH.with(|s| *s.borrow_mut() = (parts_buf, touched_buf));
    }
    top.sync(floor);
    (top.into_sorted(), total)
}

/// Leaves publish their counts to the other leaves in batches of this.
pub(crate) const COUNT_BATCH: u64 = 1024;

/// Largest [`or_groups`] score scratch (floats) kept per thread.
const SCRATCH_KEEP: usize = 1 << 20;

thread_local! {
    /// [`or_groups`]' per-doc partial scores and matched-term bits, zero
    /// between calls, reused so each leaf does not allocate and clear them.
    static OR_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// f64 bounds compared with slack, so f32 rounding of real scores can never
/// prune a competitive document.
#[inline(always)]
fn below(bound: f64, thr: f32) -> bool {
    bound * (1.0 + 1e-5) < thr as f64
}

/// Group bounds `gb` from the term bounds `wb` (a dis_max bound is
/// `(1 - t) * max + t * sum`, growing with every clause for t <= 1), `gord`
/// sorted by them ascending and their prefix sums; returns how many leading
/// groups of `gord` cannot beat `thr` together (non-essential).
fn partition(wb: &[f32], gstart: &[usize], tie: &[f32], gb: &mut [f64], gord: &mut [usize], prefix: &mut [f64], thr: f32) -> usize {
    let ng = tie.len();
    for g in 0..ng {
        let m = &wb[gstart[g]..gstart[g + 1]];
        let mx = m.iter().copied().fold(0.0f32, f32::max) as f64;
        let sm: f64 = m.iter().map(|&x| x as f64).sum();
        let t = tie[g] as f64;
        gb[g] = if t <= 1.0 { mx + t * (sm - mx) } else { mx + t * sm };
    }
    gord.sort_unstable_by(|&a, &b| gb[a].total_cmp(&gb[b]));
    for j in 0..ng {
        prefix[j + 1] = prefix[j] + gb[gord[j]];
    }
    let mut ne = 0;
    while ne < ng && below(prefix[ne + 1], thr) {
        ne += 1;
    }
    ne
}

/// The block of `c` holding its first doc >= `target` (without loading
/// it); the block count when there is none.
#[inline]
fn block_reaching(c: &Clause<'_>, target: DocId) -> usize {
    let r = c.cur.r;
    if c.done {
        return r.num_blocks();
    }
    if c.cur.docs[c.cur.n - 1] >= target {
        return c.cur.blk;
    }
    first_block_reaching(r, c.cur.blk + 1, target)
}

/// End (exclusive) of an outer pruning window starting at `pos`: the
/// nearest end of the postings block holding the next doc of an `ess`
/// clause (Lucene's `computeOuterWindowMax`), at least an inner
/// window when several clauses decide it.
fn outer_end(cls: &[Clause<'_>], ess: &[bool], pos: DocId) -> DocId {
    let (mut end, mut n) = (TERMINATED, 0);
    for c in cls {
        if c.done || !ess[c.id] {
            continue;
        }
        let r = c.cur.r;
        let b = block_reaching(c, c.cur.docs[c.cur.pos].max(pos));
        if b >= r.num_blocks() {
            continue;
        }
        n += 1;
        if b < r.num_full_blocks() {
            end = end.min(r.block_last_doc(b).saturating_add(1));
        }
    }
    if n > 1 {
        end = end.max(pos.saturating_add(W as u32));
    }
    end
}

/// Postings blocks per super-block bound of [`window_bound`].
const SUP: usize = 32;

/// [`Bm25TermScorer::bound_through`] for long windows: whole super-blocks
/// of `SUP` blocks inside the window read their precomputed maximum
/// (`sup`, built on first use; Lucene's level-1 impacts).
fn window_bound(t: &Bm25TermScorer, sup: &mut Vec<f32>, mut b: usize, hi: DocId) -> f32 {
    let r = &t.reader;
    let (nf, nb) = (r.num_full_blocks(), r.num_blocks());
    let block = |x: usize| {
        let s = r.skip(x);
        t.weight.block_bound(&s)
    };
    let mut m = 0f32;
    while b < nb {
        if b < nf {
            if b % SUP == 0 && b + SUP <= nf && r.block_last_doc(b + SUP - 1) < hi {
                if sup.is_empty() {
                    *sup = (0..nf / SUP).map(|s| (s * SUP..(s + 1) * SUP).map(&block).fold(0f32, f32::max)).collect();
                }
                m = m.max(sup[b / SUP]);
                b += SUP;
                continue;
            }
            m = m.max(block(b));
            if r.block_last_doc(b) >= hi {
                break;
            }
        } else {
            m = m.max(t.tail_max_score());
        }
        b += 1;
    }
    m
}

/// Calls `f` with each set bit (term slot) of `tch`, ascending.
#[inline(always)]
fn for_touched(tch: &[u64], mut f: impl FnMut(usize)) {
    for (w, &x) in tch.iter().enumerate() {
        let mut m = x;
        while m != 0 {
            f(w * 64 + m.trailing_zeros() as usize);
            m &= m - 1;
        }
    }
}

/// Calls `f(g, score)` for each group with a matched term slot in `tch`, in
/// group order, with the group's score as [`group_score`] computes it over
/// all its slots: unmatched slots hold exact zeros, which neither change
/// the max nor the slot-order sum.
#[inline]
fn touched_groups(tch: &[u64], row: &[f32], group_of: &[usize], single: &[bool], tie: &[f32], mut f: impl FnMut(usize, f32)) {
    let close = |g: usize, max: f32, sum: f32| if single[g] { sum } else { max + tie[g] * (sum - max) };
    let (mut cur, mut max, mut sum) = (usize::MAX, 0f32, 0f32);
    for_touched(tch, |j| {
        let g = group_of[j];
        if g != cur {
            if cur != usize::MAX {
                f(cur, close(cur, max, sum));
            }
            (cur, max, sum) = (g, 0.0, 0.0);
        }
        let v = row[j];
        max = max.max(v);
        sum += v;
    });
    if cur != usize::MAX {
        f(cur, close(cur, max, sum));
    }
}

/// Two-clause OR with an exact count by inclusion–exclusion (no deletions).
/// The rarer clause A is walked fully, probing B by block skip + gallop on
/// B's packed docs (random access); B is walked only if a B-only doc could
/// still enter the top-k.
pub fn or2_inclusion_exclusion(t0: &Bm25TermScorer, t1: &Bm25TermScorer, k: usize) -> (Vec<ScoredDoc>, u64) {
    or2_inclusion_exclusion_min(t0, t1, k, f32::NEG_INFINITY)
}

/// [`or2_inclusion_exclusion`] admitting only hits scoring above `floor`.
pub fn or2_inclusion_exclusion_min(t0: &Bm25TermScorer, t1: &Bm25TermScorer, k: usize, floor: f32) -> (Vec<ScoredDoc>, u64) {
    let a_first = t0.reader.doc_freq() <= t1.reader.doc_freq();
    let (ta, tb) = if a_first { (t0, t1) } else { (t1, t0) };
    let (ra, rb) = (&ta.reader, &tb.reader);
    let norms = &t0.norms;
    let mut a = BlockCursor::new(ra);
    let mut top = TopK::with_floor(k, floor);
    let mut inter = 0u64;
    let mut tfa = [0u32; BLOCK_DOCS];
    let mut bcur = BlockCursor::new(rb);
    let nbb = rb.num_blocks();
    for lb in 0..a.num_blocks() {
        a.load(lb);
        if lb < ra.num_full_blocks() {
            if ra.decode_tfs(lb, &mut tfa) {
                tfa = [1; BLOCK_DOCS];
            }
        } else {
            tfa = a.tail_tfs;
        }
        for i in 0..a.n {
            let d = a.docs[i];
            let nb = norms.byte(d);
            let sa = ta.weight.score_norm(tfa[i], nb);
            // Probe B.
            let mut sb = None;
            if bcur.blk == NO_BLOCK || d > bcur.docs[bcur.n - 1] {
                let mut b = if bcur.blk == NO_BLOCK { 0 } else { bcur.blk + 1 };
                while b < nbb && rb.block_last_doc(b) < d {
                    b += 1;
                }
                if b < nbb {
                    bcur.load(b);
                }
            }
            if bcur.blk != NO_BLOCK && bcur.n > 0 && d <= bcur.docs[bcur.n - 1] {
                let p = gallop(&bcur.docs, bcur.pos, bcur.n, d);
                bcur.pos = p.min(bcur.n - 1);
                if bcur.docs[p] == d {
                    sb = Some(tb.weight.score_norm(bcur.tf(bcur.blk, p), nb));
                }
            }
            let s = match sb {
                Some(sb) => {
                    inter += 1;
                    if a_first { sa + sb } else { sb + sa }
                }
                None => sa,
            };
            top.offer(d, s);
        }
    }
    if tb.weight.max_score() < top.threshold() {
        let total = ra.doc_freq() as u64 + rb.doc_freq() as u64 - inter;
        return (top.into_sorted(), total);
    }
    // Walk B for B-only docs that may compete.
    let mut b = BlockCursor::new(rb);
    let mut ac = BlockCursor::new(ra);
    let mut tfb = [0u32; BLOCK_DOCS];
    for blk in 0..b.num_blocks() {
        b.load(blk);
        if blk < rb.num_full_blocks() {
            if rb.decode_tfs(blk, &mut tfb) {
                tfb = [1; BLOCK_DOCS];
            }
        } else {
            tfb = b.tail_tfs;
        }
        for i in 0..b.n {
            let d = b.docs[i];
            // In A?
            if ac.blk == NO_BLOCK || d > ac.docs[ac.n - 1] {
                let mut x = if ac.blk == NO_BLOCK { 0 } else { ac.blk + 1 };
                while x < ra.num_blocks() && ra.block_last_doc(x) < d {
                    x += 1;
                }
                if x < ra.num_blocks() {
                    ac.load(x);
                }
            }
            if ac.blk != NO_BLOCK && ac.n > 0 && d <= ac.docs[ac.n - 1] {
                let p = gallop(&ac.docs, ac.pos, ac.n, d);
                ac.pos = p.min(ac.n - 1);
                if ac.docs[p] == d {
                    continue;
                }
            }
            top.offer_any(d, tb.weight.score_norm(tfb[i], norms.byte(d)));
        }
    }
    let total = ra.doc_freq() as u64 + rb.doc_freq() as u64 - inter;
    (top.into_sorted(), total)
}

/// Collect `children` (all BM25 term scorers, unstarted) as a conjunction.
pub fn try_conjunction(children: &mut [Box<dyn Scorer>], k: usize, del: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
    try_conjunction_min(children, k, del, f32::NEG_INFINITY)
}

/// [`try_conjunction`] admitting only hits scoring above `floor`.
pub fn try_conjunction_min(children: &mut [Box<dyn Scorer>], k: usize, del: Option<&[u64]>, floor: f32) -> Option<(Vec<ScoredDoc>, u64)> {
    try_conjunction_lim(children, k, del, u64::MAX, &Floor::new(floor, None))
}

/// [`try_conjunction_min`] with [`conjunction_lim`]'s bounded count.
pub fn try_conjunction_lim(children: &mut [Box<dyn Scorer>], k: usize, del: Option<&[u64]>, count_limit: u64, floor: &Floor<'_>) -> Option<(Vec<ScoredDoc>, u64)> {
    let terms = bm25_terms(children)?;
    if terms.len() > MAX_CONJ {
        return None;
    }
    let refs: Vec<&Bm25TermScorer> = terms.iter().map(|t| &**t).collect();
    if !same_norms(&refs) {
        return None;
    }
    let out = conjunction_lim(&refs, k, del, count_limit, floor);
    drop(refs);
    for t in terms {
        t.advance_to_end();
    }
    Some(out)
}

/// Collect `children` (all BM25 term scorers, unstarted) as a disjunction.
pub fn try_disjunction(children: &mut [Box<dyn Scorer>], k: usize, del: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
    try_disjunction_min(children, k, del, f32::NEG_INFINITY)
}

/// [`try_disjunction`] admitting only hits scoring above `floor`.
pub fn try_disjunction_min(children: &mut [Box<dyn Scorer>], k: usize, del: Option<&[u64]>, floor: f32) -> Option<(Vec<ScoredDoc>, u64)> {
    let terms = bm25_terms(children)?;
    let refs: Vec<&Bm25TermScorer> = terms.iter().map(|t| &**t).collect();
    if !same_norms(&refs) {
        return None;
    }
    let out = disjunction_min(&refs, k, del, floor);
    drop(refs);
    for t in terms {
        t.advance_to_end();
    }
    Some(out)
}

/// Collect a disjunction whose clauses are all BM25 term groups (see
/// [`Scorer::as_term_groups`]: terms, dis_max of terms, and disjunctions of
/// those), unstarted, when its total only has to be exact up to
/// `count_limit`: [`or2_inclusion_exclusion_min`] for two plain terms whose
/// count must be exact, else [`or_groups`]. `None` when a clause is
/// anything else; nothing is consumed then.
pub fn try_term_groups_lim<'s, S: Scorer + ?Sized + 's>(
    clauses: impl Iterator<Item = &'s mut S>,
    k: usize,
    del: Option<&[u64]>,
    count_limit: u64,
    floor: &Floor<'_>,
) -> Option<(Vec<ScoredDoc>, u64)> {
    try_term_groups_in(clauses, k, del, None, 0.0, count_limit, floor)
}

/// [`try_term_groups_lim`] over the docs set in `keep`, each hit scoring
/// `add` more (see [`or_groups_in`]).
pub(crate) fn try_term_groups_in<'s, S: Scorer + ?Sized + 's>(
    clauses: impl Iterator<Item = &'s mut S>,
    k: usize,
    del: Option<&[u64]>,
    keep: Option<&[u64]>,
    add: f32,
    count_limit: u64,
    floor: &Floor<'_>,
) -> Option<(Vec<ScoredDoc>, u64)> {
    let mut items: Vec<Vec<(f32, Vec<&'s mut Bm25TermScorer>)>> = Vec::new();
    for c in clauses {
        items.push(c.as_term_groups()?);
    }
    if items.is_empty() {
        return None;
    }
    for (tie, ts) in items.iter().flatten() {
        if ts.is_empty() || !(tie.is_finite() && *tie >= 0.0) || ts.iter().any(|t| !t.is_unstarted() || !t.prunable()) {
            return None;
        }
    }
    let refs: Vec<Vec<TermGroup<'_>>> = items.iter().map(|it| it.iter().map(|(tie, ts)| (*tie, ts.iter().map(|t| &**t).collect())).collect()).collect();
    let flat: Option<Vec<&Bm25TermScorer>> = refs
        .iter()
        .map(|it| match it.as_slice() {
            [(_, ts)] if ts.len() == 1 => Some(ts[0]),
            _ => None,
        })
        .collect();
    let out = match flat {
        // Inclusion–exclusion walks the rarer list fully: worth it only
        // while the union may still be below the count limit.
        Some(ts) if ts.len() == 2 && del.is_none() && keep.is_none() && add == 0.0 && same_norms(&ts)
            && (count_limit == u64::MAX || ts.iter().map(|t| t.reader.doc_freq() as u64).max().unwrap_or(0) < count_limit) =>
        {
            or2_inclusion_exclusion_min(ts[0], ts[1], k, floor.get())
        }
        _ => or_groups_in(&refs, k, del, keep, add, count_limit, floor),
    };
    drop(refs);
    for t in items.iter_mut().flatten().flat_map(|(_, ts)| ts.iter_mut()) {
        t.advance_to_end();
    }
    Some(out)
}

fn bm25_terms(children: &mut [Box<dyn Scorer>]) -> Option<Vec<&mut Bm25TermScorer>> {
    if children.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(children.len());
    for c in children.iter_mut() {
        let t = c.as_bm25_term()?;
        if !t.is_unstarted() || !t.prunable() {
            return None;
        }
        out.push(t);
    }
    Some(out)
}

/// The bulk AND/OR read norms from the first clause: all clauses must share
/// the field's norms.
fn same_norms(terms: &[&Bm25TermScorer]) -> bool {
    let first = &terms[0].norms;
    terms.iter().all(|t| t.norms.same_column(first))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::postings::{write_postings_with_norms, BlockNorms};
    use crate::query::bm25::{int_to_byte4, Bm25Params};
    use crate::query::collector::collect_top_k;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    struct Fx {
        norms: Arc<Norms>,
        lists: Vec<(Vec<u32>, Vec<u32>)>,
        n: u32,
    }

    fn fixture(n: u32, densities: &[u64], seed: u64) -> Fx {
        let mut s = seed;
        let lens: Vec<u32> = (0..n).map(|_| 1 + (lcg(&mut s) % 300) as u32).collect();
        let bytes: Vec<u8> = lens.iter().map(|&l| int_to_byte4(l)).collect();
        let total: u64 = lens.iter().map(|&l| l as u64).sum();
        let norms = Arc::new(Norms::from_bytes(bytes, n, total));
        let mut lists = Vec::new();
        for &den in densities {
            let (mut d, mut t) = (Vec::new(), Vec::new());
            for doc in 0..n {
                if lcg(&mut s) % den == 0 {
                    d.push(doc);
                    t.push(1 + (lcg(&mut s) % 4) as u32 * (lcg(&mut s) % 2) as u32);
                }
            }
            lists.push((d, t));
        }
        Fx { norms, lists, n }
    }

    thread_local! {
        /// Whether [`scorer`] writes postings with impact staircases (v12).
        static IMPACTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Run `f` over v11 postings (no staircase) and v12 ones (staircases:
    /// tighter block bounds, more blocks skipped and clauses split).
    fn v11_and_v12(f: impl Fn()) {
        for impacts in [false, true] {
            IMPACTS.with(|x| x.set(impacts));
            f();
        }
        IMPACTS.with(|x| x.set(false));
    }

    fn scorer(fx: &Fx, i: usize) -> Bm25TermScorer {
        let (d, t) = &fx.lists[i];
        let mut buf = Vec::new();
        crate::codec::postings::write_postings_impl(d, t, None, BlockNorms::Bytes(fx.norms.bytes()), IMPACTS.with(|x| x.get()), &mut buf);
        let r = PostingsReader::from_bytes(buf).unwrap();
        let w = Bm25Weight::new(d.len() as u64, fx.n as u64, Bm25Params::default(), fx.norms.avg()).with_boost(1.0 + i as f32 * 0.25);
        Bm25TermScorer::new(r, w, fx.norms.clone())
    }

    fn per_doc(mut s: Box<dyn Scorer>, k: usize, del: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
        collect_top_k(&mut *s, k, |d| del.is_none_or(|w| !is_deleted(w, d)))
    }

    fn check(a: (Vec<ScoredDoc>, u64), b: (Vec<ScoredDoc>, u64), what: &str) {
        assert_eq!(a.1, b.1, "{what}: count");
        assert_eq!(a.0.len(), b.0.len(), "{what}: len");
        for (x, y) in a.0.iter().zip(&b.0) {
            assert_eq!(x.doc, y.doc, "{what}: {a:?} vs {b:?}", a = a.0, b = b.0);
            assert!((x.score - y.score).abs() <= 1e-5 * x.score.abs().max(1.0), "{what}: score");
        }
    }

    #[test]
    fn bulk_paths_match_per_doc() {
        v11_and_v12(|| {
            let fx = fixture(30_000, &[3, 7, 40, 500], 42);
            let mut del = vec![0u64; (fx.n as usize).div_ceil(64)];
            for d in (0..fx.n).step_by(97) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            for dl in [None, Some(&del[..])] {
                for k in [1usize, 10, 37] {
                    for i in 0..4 {
                        let mut s = scorer(&fx, i);
                        let got = s.collect_top_k_bulk(k, dl).unwrap();
                        check(got, per_doc(Box::new(scorer(&fx, i)), k, dl), "term");
                    }
                    for set in [vec![0usize, 1], vec![1, 2], vec![2, 3, 0], vec![3, 0, 1, 2]] {
                        let mk = || set.iter().map(|&i| Box::new(scorer(&fx, i)) as Box<dyn Scorer>).collect::<Vec<_>>();
                        let mut c = mk();
                        let got = try_conjunction(&mut c, k, dl).unwrap();
                        let want = per_doc(Box::new(crate::query::boolean::IntersectionN::new(mk())), k, dl);
                        check(got, want, "and");
                        let mut c = mk();
                        let got = try_disjunction(&mut c, k, dl).unwrap();
                        let want = per_doc(crate::query::boolean::union_of(mk()), k, dl);
                        check(got, want, "or");
                    }
                }
            }
        });
    }

    /// A negative or non-finite weight breaks every bound: no bulk path
    /// takes it, the caller scores it exhaustively.
    #[test]
    fn negative_and_non_finite_weights_are_never_pruned() {
        let fx = fixture(5_000, &[3, 7], 5);
        for b in [-2.0f32, f32::NAN, f32::INFINITY] {
            let bad = || {
                let mut s = scorer(&fx, 0);
                s.weight = s.weight.clone().with_boost(b);
                s
            };
            assert!(!bad().prunable());
            assert!(bad().collect_top_k_bulk(10, None).is_none());
            assert!(bad().collect_top_k_bulk_min(10, None, 0.5).is_none());
            assert!(bad().collect_top_k_count_limited(10, None, 100, &Floor::none()).is_none());
            let mk = || vec![Box::new(bad()) as Box<dyn Scorer>, Box::new(scorer(&fx, 1)) as Box<dyn Scorer>];
            assert!(try_conjunction_lim(&mut mk(), 10, None, 100, &Floor::none()).is_none());
            assert!(try_disjunction_min(&mut mk(), 10, None, 0.5).is_none());
            let mut c = mk();
            assert!(try_term_groups_lim(c.iter_mut().map(|s| &mut **s), 10, None, 100, &Floor::none()).is_none());
            // Nothing was consumed: the per-doc path still sees every doc.
            assert!(c.iter_mut().all(|s| s.as_bm25_term().is_some_and(|t| t.is_unstarted())));
        }
        assert!(scorer(&fx, 0).prunable());
        // BM25 parameters out of Lucene's ranges (an index an older build
        // created with b: 2 or k1: -0.5): no bound either, so a should
        // query's MaxScore top-k is the exhaustive one.
        for (k1, b) in [(1.2f32, 2.0f32), (-0.5, 0.75)] {
            let sc = |i: usize| {
                let mut s = scorer(&fx, i);
                s.weight = Bm25Weight::new(fx.lists[i].0.len() as u64, fx.n as u64, Bm25Params { k1, b }, fx.norms.avg()).with_boost(1.0 + i as f32 * 0.25);
                s
            };
            assert!(!sc(0).prunable() && sc(0).max_score().is_none() && sc(0).max_score_up_to(100).is_none(), "k1 {k1} b {b}");
            let mk = || (0..2).map(|i| Box::new(sc(i)) as Box<dyn Scorer>).collect::<Vec<_>>();
            let want = per_doc(crate::query::boolean::union_of(mk()), 10, None);
            for limit in [0u64, 50, u64::MAX] {
                let got = crate::query::boolean::union_of(mk()).collect_top_k_count_limited(10, None, limit, &Floor::none()).unwrap_or_else(|| per_doc(crate::query::boolean::union_of(mk()), 10, None));
                check_limited(got, &want, limit, "non-monotonic should");
            }
        }
    }

    #[test]
    fn constant_score_keyword_path() {
        let docs: Vec<u32> = (0..1000u32).map(|i| i * 3).collect();
        let tfs = vec![1u32; docs.len()];
        let mut buf = Vec::new();
        write_postings_with_norms(&docs, &tfs, None, BlockNorms::Absent, &mut buf);
        let r = PostingsReader::from_bytes(buf).unwrap();
        let w = Bm25Weight::new(1000, 3000, Bm25Params::default(), 1.0);
        let mut s = Bm25TermScorer::new(r, w.clone(), Arc::new(Norms::unit()));
        assert!(s.is_constant());
        let (hits, n) = s.collect_top_k_bulk(10, None).unwrap();
        assert_eq!(n, 1000);
        assert_eq!(hits.iter().map(|h| h.doc).collect::<Vec<_>>(), (0..10u32).map(|i| i * 3).collect::<Vec<_>>());
        assert_eq!(hits[0].score, w.score_norm(1, 1));
        let del = vec![0b1001u64; 50];
        let r = s.reader().clone();
        let (hits, n) = first_k(&r, 3, Some(&del));
        assert_eq!(hits, vec![6, 9, 12]);
        let dead = docs.iter().filter(|&&d| is_deleted(&del, d)).count() as u64;
        assert_eq!(n, 1000 - dead);
    }

    /// Count-limited results against the exhaustive ones: identical hits
    /// (bit-for-bit scores), and a total that is exact up to the limit and
    /// a lower bound past it.
    fn check_limited(got: (Vec<ScoredDoc>, u64), want: &(Vec<ScoredDoc>, u64), limit: u64, what: &str) {
        let g: Vec<(u32, u32)> = got.0.iter().map(|h| (h.doc, h.score.to_bits())).collect();
        let w: Vec<(u32, u32)> = want.0.iter().map(|h| (h.doc, h.score.to_bits())).collect();
        assert_eq!(g, w, "{what}: hits (limit {limit})");
        assert!(got.1 <= want.1 && got.1 >= limit.min(want.1), "{what}: total {} of {} (limit {limit})", got.1, want.1);
        if limit >= want.1 {
            assert_eq!(got.1, want.1, "{what}: exact total (limit {limit})");
        }
    }

    #[test]
    fn count_limited_term_and_conjunction_match_exhaustive() {
        v11_and_v12(|| {
            let fx = fixture(30_000, &[3, 7, 40, 500], 7);
            let mut del = vec![0u64; (fx.n as usize).div_ceil(64)];
            for d in (0..fx.n).step_by(13) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            for dl in [None, Some(&del[..])] {
                for k in [1usize, 10] {
                    for limit in [0u64, 50, 1_000, u64::MAX] {
                        for i in 0..4 {
                            let want = per_doc(Box::new(scorer(&fx, i)), k, dl);
                            let got = scorer(&fx, i).collect_top_k_count_limited(k, dl, limit, &Floor::none()).unwrap();
                            check_limited(got, &want, limit, "term");
                        }
                        for set in [vec![0usize, 1], vec![1, 2, 0], vec![0, 1, 2, 3]] {
                            let mk = || set.iter().map(|&i| Box::new(scorer(&fx, i)) as Box<dyn Scorer>).collect::<Vec<_>>();
                            let want = per_doc(Box::new(crate::query::boolean::IntersectionN::new(mk())), k, dl);
                            let mut c = mk();
                            let got = try_conjunction_lim(&mut c, k, dl, limit, &Floor::none()).unwrap();
                            check_limited(got, &want, limit, "and");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn grouped_block_or_matches_exhaustive() {
        v11_and_v12(|| {
            use crate::query::boolean::{union_of, Union};
            use crate::query::query::DisMaxScorer;
            // Two "fields" with their own norms.
            let fa = fixture(20_000, &[3, 9, 60, 400], 11);
            let fb = fixture(20_000, &[5, 40, 300, 2], 12);
            let mut del = vec![0u64; 20_000usize.div_ceil(64)];
            for d in (0..20_000u32).step_by(11) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            let t = |fx: &Fx, i: usize| Box::new(scorer(fx, i)) as Box<dyn Scorer>;
            for tie in [0.0f32, 0.3] {
                let dm = |x: Box<dyn Scorer>, y: Box<dyn Scorer>| Box::new(DisMaxScorer::new(vec![x, y], tie)) as Box<dyn Scorer>;
                // Words: two dis_max over both fields, a lone term, and a
                // nested group (a two-clause disjunction of a dis_max and a term).
                let mk = || vec![dm(t(&fa, 0), t(&fb, 0)), dm(t(&fa, 1), t(&fb, 1)), t(&fa, 2), union_of(vec![dm(t(&fa, 3), t(&fb, 2)), t(&fb, 3)])];
                for dl in [None, Some(&del[..])] {
                    for k in [1usize, 10, 40] {
                        for limit in [0u64, 100, 5_000, u64::MAX] {
                            let want = per_doc(Box::new(Union::new(mk())), k, dl);
                            let got = Union::new(mk()).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("block OR");
                            check_limited(got, &want, limit, "dis_max OR");
                            let want = per_doc(dm(t(&fa, 1), t(&fb, 3)), k, dl);
                            let got = dm(t(&fa, 1), t(&fb, 3)).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("dis_max");
                            check_limited(got, &want, limit, "lone dis_max");
                            let want = per_doc(union_of(vec![dm(t(&fa, 0), t(&fb, 1)), t(&fb, 3)]), k, dl);
                            let got = union_of(vec![dm(t(&fa, 0), t(&fb, 1)), t(&fb, 3)]).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("pair");
                            check_limited(got, &want, limit, "two clauses");
                            // Plain terms over different norms.
                            let want = per_doc(Box::new(Union::new(vec![t(&fa, 0), t(&fb, 1), t(&fa, 3)])), k, dl);
                            let got = Union::new(vec![t(&fa, 0), t(&fb, 1), t(&fa, 3)]).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("terms");
                            check_limited(got, &want, limit, "terms");
                        }
                    }
                }
            }
        });
    }

    /// [`fixture`] whose terms are dense, high-tf "hot" in every third
    /// 4096-doc window (shifted per term) and sparse elsewhere, so the
    /// per-window block-max split changes from window to window.
    fn fixture_regions(n: u32, densities: &[u64], seed: u64) -> Fx {
        let mut fx = fixture(n, &[], seed);
        let mut s = seed ^ 0x5DEECE66D;
        for (j, &den) in densities.iter().enumerate() {
            let (mut d, mut t) = (Vec::new(), Vec::new());
            for doc in 0..n {
                let hot = ((doc / 4096) as usize + j) % 3 == 0;
                let m = if hot { 2 } else { den };
                if lcg(&mut s) % m == 0 {
                    d.push(doc);
                    t.push(if hot { 1 + (lcg(&mut s) % 8) as u32 } else { 1 });
                }
            }
            fx.lists.push((d, t));
        }
        fx
    }

    #[test]
    fn grouped_block_or_across_many_windows_matches_exhaustive() {
        v11_and_v12(|| {
            use crate::query::boolean::{union_of, Union};
            use crate::query::query::DisMaxScorer;
            let n = 70_000u32;
            let fa = fixture_regions(n, &[50, 200, 30, 900], 21);
            let fb = fixture_regions(n, &[400, 60, 5000, 25], 22);
            let mut del = vec![0u64; (n as usize).div_ceil(64)];
            for d in (0..n).step_by(17) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            let t = |fx: &Fx, i: usize| Box::new(scorer(fx, i)) as Box<dyn Scorer>;
            for tie in [0.0f32, 0.5] {
                let dm = |x: Box<dyn Scorer>, y: Box<dyn Scorer>| Box::new(DisMaxScorer::new(vec![x, y], tie)) as Box<dyn Scorer>;
                let mk = || vec![dm(t(&fa, 0), t(&fb, 0)), dm(t(&fa, 1), t(&fb, 1)), t(&fa, 2), union_of(vec![dm(t(&fa, 3), t(&fb, 2)), t(&fb, 3)])];
                for dl in [None, Some(&del[..])] {
                    for k in [1usize, 10, 30] {
                        for limit in [0u64, 100, 20_000, u64::MAX] {
                            let want = per_doc(Box::new(Union::new(mk())), k, dl);
                            let got = Union::new(mk()).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("block OR");
                            check_limited(got, &want, limit, "regions dis_max OR");
                            let want = per_doc(Box::new(Union::new(vec![t(&fa, 0), t(&fa, 1), t(&fa, 2), t(&fa, 3)])), k, dl);
                            let got = Union::new(vec![t(&fa, 0), t(&fa, 1), t(&fa, 2), t(&fa, 3)]).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("terms");
                            check_limited(got, &want, limit, "regions terms");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn long_multi_field_queries_match_exhaustive() {
        v11_and_v12(|| {
            use crate::query::boolean::Union;
            use crate::query::query::DisMaxScorer;
            // Mostly common words plus a few rare ones, over 2-4 fields with
            // their own norms: the shape where groups turn required.
            let n = 30_000u32;
            let dens: [u64; 14] = [2, 3, 2, 5, 4, 700, 3, 2, 60, 6, 2000, 3, 9, 2];
            let fxs: Vec<Fx> = (0..4).map(|c| fixture(n, &dens, 100 + c as u64)).collect();
            let mut del = vec![0u64; (n as usize).div_ceil(64)];
            for d in (0..n).step_by(23) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            let t = |c: usize, i: usize| Box::new(scorer(&fxs[c], i)) as Box<dyn Scorer>;
            for (case, words) in [8usize, 11, 14].into_iter().enumerate() {
                let ncols = 2 + case % 3;
                for tie in [0.0f32, 0.3] {
                    let mk = || -> Vec<Box<dyn Scorer>> {
                        (0..words)
                            .map(|w| if w % 5 == 4 { t(w % ncols, w) } else { Box::new(DisMaxScorer::new((0..ncols).map(|c| t(c, w)).collect(), tie)) as Box<dyn Scorer> })
                            .collect()
                    };
                    let flat = || -> Vec<Box<dyn Scorer>> { (0..words).map(|w| t(case % 4, w)).collect() };
                    for dl in [None, Some(&del[..])] {
                        for k in [1usize, 10, 20] {
                            let want = per_doc(Box::new(Union::new(mk())), k, dl);
                            let want_flat = per_doc(Box::new(Union::new(flat())), k, dl);
                            for limit in [0u64, 50, 2_000, u64::MAX] {
                                let what = format!("{words} words, {ncols} fields, tie {tie}, k {k}");
                                let got = Union::new(mk()).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("block OR");
                                check_limited(got, &want, limit, &what);
                                let got = Union::new(flat()).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("flat block OR");
                                check_limited(got, &want_flat, limit, &format!("flat {what}"));
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn wiki_shaped_queries_prune_and_match_exhaustive() {
        v11_and_v12(|| {
            use crate::query::boolean::Union;
            use crate::query::query::DisMaxScorer;
            use std::sync::atomic::{AtomicU64, Ordering};
            // Title-like queries over three fields with their own norms: a word
            // in every doc, common ones and a rare one, over hundreds of blocks,
            // so outer windows grow past the frequent terms' blocks and their
            // bounds come from super-blocks.
            let n = 200_000u32;
            let dens: [u64; 5] = [1, 2, 25, 90, 40_000];
            let fxs: Vec<Fx> = (0..3).map(|c| fixture(n, &dens, 300 + c as u64)).collect();
            let mut del = vec![0u64; (n as usize).div_ceil(64)];
            for d in (0..n).step_by(29) {
                del[d as usize / 64] |= 1 << (d % 64);
            }
            let t = |c: usize, i: usize| Box::new(scorer(&fxs[c], i)) as Box<dyn Scorer>;
            // Single words too: a lone dis_max prunes only by block bounds.
            for words in [vec![0usize, 4], vec![0, 1, 2, 4], vec![1, 3], vec![0, 1, 2, 3, 4], vec![1], vec![3], vec![0]] {
                for tie in [0.0f32, 0.3] {
                    let mk = || -> Vec<Box<dyn Scorer>> {
                        words.iter().map(|&w| Box::new(DisMaxScorer::new((0..3).map(|c| t(c, w)).collect(), tie)) as Box<dyn Scorer>).collect()
                    };
                    for dl in [None, Some(&del[..])] {
                        for k in [1usize, 20] {
                            let want = per_doc(Box::new(Union::new(mk())), k, dl);
                            for limit in [0u64, 10_001, u64::MAX] {
                                let what = format!("words {words:?}, tie {tie}, k {k}, deletions {}", dl.is_some());
                                let got = Union::new(mk()).collect_top_k_count_limited(k, dl, limit, &Floor::none()).expect("block OR");
                                check_limited(got, &want, limit, &what);
                                // Other leaves already reached the shard-wide
                                // limit: no exact counting, the same hits, and
                                // this leaf's count is published.
                                if limit != u64::MAX {
                                    let hits = AtomicU64::new(limit);
                                    let got = Union::new(mk()).collect_top_k_count_limited(k, dl, limit, &Floor::none().with_hits(&hits)).expect("block OR");
                                    assert!(got.1 <= want.1, "{what}: shared total");
                                    assert_eq!(hits.load(Ordering::Relaxed), limit + got.1, "{what}: published");
                                    check_limited((got.0, want.1), &want, u64::MAX, &format!("shared {what}"));
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn shared_floor_is_monotonic_and_admits_ties() {
        let f = SharedFloor::new();
        assert_eq!(f.get(), f32::NEG_INFINITY);
        f.raise(2.0);
        f.raise(1.0);
        f.raise(f32::NAN);
        f.raise(f32::INFINITY);
        assert_eq!(f.get(), 2.0f32.next_down());
        // A hit tying another leaf's k-th score may win the tie-break.
        let mut top = TopK::with_floor(2, Floor::new(1.5, Some(&f)).get());
        top.offer(1, 2.0);
        assert_eq!(top.into_sorted().len(), 1);
        // A full top-k publishes its k-th score; a raised floor raises the
        // threshold.
        let g = SharedFloor::new();
        let fl = Floor::new(f32::NEG_INFINITY, Some(&g));
        let mut t = TopK::with_floor(2, fl.get());
        t.offer(0, 3.0);
        t.sync(&fl);
        assert_eq!(g.get(), f32::NEG_INFINITY);
        t.offer(1, 4.0);
        t.sync(&fl);
        assert_eq!(g.get(), 3.0f32.next_down());
        g.raise(5.0);
        t.sync(&fl);
        assert_eq!(t.threshold(), 5.0f32.next_down());
        // A score held by an earlier leaf: a later leaf's tie loses, so its
        // floor is the score itself; the holder and earlier leaves may tie.
        let h = SharedFloor::new();
        h.raise_from(2.0, 3);
        h.raise_from(2.0, 5);
        assert_eq!(h.get_for(Some(4)), 2.0);
        assert_eq!(h.get_for(Some(3)), 2.0f32.next_down());
        assert_eq!(h.get_for(Some(1)), 2.0f32.next_down());
        assert_eq!(h.get_for(None), 2.0f32.next_down());
        t.offer(2, 4.5);
        assert_eq!(t.into_sorted().iter().map(|h| h.doc).collect::<Vec<_>>(), vec![1, 0]);
    }
}
