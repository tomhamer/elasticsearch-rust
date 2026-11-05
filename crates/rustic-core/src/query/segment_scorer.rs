//! Adapters that turn a segment's on-disk postings into [`DocSet`]s and
//! [`Scorer`]s, plus the position-aware phrase scorer.
//!
//! This is the bridge between the codec layer and the boolean/collector
//! layer: everything above here works on `Box<dyn Scorer>` trees and never
//! sees a segment.

use crate::codec::fastfields::NormsReader;
use crate::codec::postings::PostingsReader;
use crate::query::bm25::{Bm25Params, Bm25Weight};
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::bulk::Bm25TermScorer;
use crate::query::collector::ScoredDoc;
use crate::query::scorer::{Scorer, TermFreqSource};
use crate::query::similarity::SimWeight;
use crate::segment::Segment;
use crate::types::{DocId, Term};
use std::sync::Arc;

/// A [`DocSet`] over one term's postings in one segment.
pub struct PostingsDocSet {
    reader: PostingsReader,
    current: DocId,
}

impl PostingsDocSet {
    pub fn new(reader: PostingsReader) -> Self {
        Self {
            reader,
            current: TERMINATED,
        }
    }

    /// Positions of the current doc (empty unless indexed with positions).
    pub fn positions(&mut self) -> &[u32] {
        self.reader.positions()
    }

    /// The current doc's positions, each plus `shift`, into `out`.
    pub fn positions_into(&mut self, out: &mut Vec<u32>, shift: u32) {
        self.reader.positions_into(out, shift)
    }

    /// Term frequency of the current doc.
    pub fn freq(&mut self) -> u32 {
        self.reader.freq()
    }

    /// Positions `p` of the current doc with `p + shift` in sorted `targets`.
    pub fn count_shifted_matches(&mut self, targets: &[u32], shift: u32) -> u32 {
        self.reader.count_shifted_matches(targets, shift)
    }
}

impl DocSet for PostingsDocSet {
    fn advance(&mut self) -> DocId {
        self.current = self.reader.next().unwrap_or(TERMINATED);
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        self.current = self.reader.seek_forward(target).unwrap_or(TERMINATED);
        self.current
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.reader.doc_freq()
    }
}

impl TermFreqSource for PostingsDocSet {
    fn term_freq(&self) -> u32 {
        self.reader.term_freq().unwrap_or(1)
    }
}

/// A field's norms as a scorer can hold them: one `SmallFloat` byte per
/// doc, either a zero-copy view of the mapped segment file (`Arc` clone, no
/// per-query copy) or an owned vector (memtable path). A field without
/// norms scores every doc with norm byte 1 (length 1, average 1).
pub struct Norms {
    data: NormsData,
    /// Cached `bytes` view (points into `data`).
    ptr: *const u8,
    len: usize,
    avg: f32,
}

// SAFETY: `ptr`/`len` describe bytes owned by `data` (a `Vec` or an `Arc`
// of `Send + Sync` bytes) and are only read.
unsafe impl Send for Norms {}
unsafe impl Sync for Norms {}

enum NormsData {
    Unit,
    Owned(Vec<u8>),
    /// One byte per doc, straight from the segment mapping.
    Mapped(crate::codec::directory::FileSlice),
    /// A constant column's bytes: a process-wide run of its byte.
    Static(&'static [u8]),
}

impl Norms {
    fn build(data: NormsData, avg: f32) -> Self {
        let (ptr, len) = match &data {
            NormsData::Unit => (std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8, 0),
            NormsData::Owned(v) => (v.as_ptr(), v.len()),
            NormsData::Mapped(f) => {
                let s = f.as_slice();
                (s.as_ptr(), s.len())
            }
            NormsData::Static(s) => (s.as_ptr(), s.len()),
        };
        Self { data, ptr, len, avg }
    }

    /// Owned copy (prefer [`Norms::mapped`] for sealed segments).
    pub fn from_reader(r: &NormsReader<'_>) -> Self {
        Self::build(NormsData::Owned(r.bytes().to_vec()), r.avg())
    }

    /// Zero-copy norms over `bytes` (one norm byte per doc).
    pub fn mapped(bytes: crate::codec::directory::FileSlice, avg: f32) -> Self {
        Self::build(NormsData::Mapped(bytes), avg)
    }

    /// Norms of a constant column (one byte for every doc, see
    /// [`crate::codec::fastfields::NormsReader::open_constant`]).
    pub fn constant(bytes: &'static [u8], avg: f32) -> Self {
        Self::build(NormsData::Static(bytes), avg)
    }

    /// Norms from raw per-doc lengths (encoded to norm bytes).
    pub fn from_lengths(lengths: Vec<u32>, num_docs: u32, total: u64) -> Self {
        let bytes = lengths.iter().map(|&l| crate::query::bm25::int_to_byte4(l)).collect();
        Self::from_bytes(bytes, num_docs, total)
    }

    /// Norms from per-doc norm bytes (memtable path).
    pub fn from_bytes(bytes: Vec<u8>, num_docs: u32, total: u64) -> Self {
        let avg = if num_docs == 0 { 1.0 } else { (total as f64 / num_docs as f64).max(1.0) as f32 };
        Self::build(NormsData::Owned(bytes), avg)
    }

    /// Norms for a field with no stored lengths (every doc has length 1).
    pub fn unit() -> Self {
        Self::build(NormsData::Unit, 1.0)
    }

    /// True for a field without norms.
    pub fn is_unit(&self) -> bool {
        matches!(self.data, NormsData::Unit)
    }

    /// The norm bytes (empty for unit norms).
    #[inline(always)]
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: see the `Send` impl.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Norm byte of `doc` (1 for unit norms or docs past the column).
    #[inline(always)]
    pub fn byte(&self, doc: DocId) -> u8 {
        match self.bytes().get(doc as usize) {
            Some(&b) => b,
            None => 1,
        }
    }

    /// Field length of `doc` as scoring sees it (decoded norm byte).
    #[inline]
    pub fn len(&self, doc: DocId) -> u32 {
        crate::query::bm25::byte4_to_int(self.byte(doc))
    }

    pub fn avg(&self) -> f32 {
        self.avg
    }

    /// Same norm bytes (same column, or both unit).
    pub fn same_column(&self, other: &Norms) -> bool {
        self.is_unit() == other.is_unit() && self.ptr == other.ptr && self.len == other.len
    }
}

/// Build a BM25 [`TermScorer`] for `term` in `segment`.
///
/// `num_docs_global`/`doc_freq_global` let the caller supply index-wide
/// statistics so the same term scores identically in every segment (ES
/// uses per-shard stats; Lucene per-segment). Pass the segment's own
/// numbers to get Lucene behaviour.
pub fn term_scorer(
    segment: &Segment,
    term: &Term,
    params: Bm25Params,
    doc_freq_global: u64,
    num_docs_global: u64,
    norms: Arc<Norms>,
) -> Option<Box<dyn Scorer>> {
    let postings = segment.postings(term)?;
    let weight = Bm25Weight::new(doc_freq_global, num_docs_global, params, norms.avg());
    Some(Box::new(Bm25TermScorer::new(postings, weight, norms)))
}

/// Convenience: a term scorer using this segment's own statistics.
pub fn segment_term_scorer(segment: &Segment, term: &Term, params: Bm25Params) -> Option<Box<dyn Scorer>> {
    let df = segment.term_dict().doc_freq(term) as u64;
    if df == 0 {
        return None;
    }
    let norms = Arc::new(segment.norms_shared(term.field));
    term_scorer(segment, term, params, df, segment.num_docs as u64, norms)
}

/// One phrase position: the union of the postings of every alternative
/// term allowed there (a single term for a plain phrase; several for
/// Lucene's `MultiPhraseQuery`, e.g. synonyms).
pub struct PositionGroup {
    members: Vec<PostingsDocSet>,
    current: DocId,
    primed: bool,
}

impl PositionGroup {
    fn new(members: Vec<PostingsReader>) -> Self {
        Self {
            members: members.into_iter().map(PostingsDocSet::new).collect(),
            current: TERMINATED,
            primed: false,
        }
    }

    fn update(&mut self) -> DocId {
        self.current = self.members.iter().map(|m| m.doc()).min().unwrap_or(TERMINATED);
        self.current
    }

    /// Sorted, deduplicated positions (each plus `shift`) of every member on
    /// the current doc, into `out`.
    fn positions_into(&mut self, out: &mut Vec<u32>, shift: u32) {
        if self.members.len() == 1 {
            self.members[0].positions_into(out, shift);
            return;
        }
        out.clear();
        let d = self.current;
        for m in &mut self.members {
            if m.doc() == d {
                out.extend_from_slice(m.positions());
            }
        }
        out.sort_unstable();
        out.dedup();
        if shift != 0 {
            for p in out.iter_mut() {
                *p += shift;
            }
        }
    }

    /// Upper bound of the number of positions on the current doc.
    fn freq(&mut self) -> u32 {
        if self.members.len() == 1 {
            return self.members[0].freq();
        }
        let d = self.current;
        self.members.iter_mut().filter(|m| m.doc() == d).map(|m| m.freq()).sum()
    }
}

impl DocSet for PositionGroup {
    fn advance(&mut self) -> DocId {
        if self.members.len() == 1 {
            self.primed = true;
            self.current = self.members[0].advance();
            return self.current;
        }
        if !self.primed {
            self.primed = true;
            for m in &mut self.members {
                m.advance();
            }
        } else if self.current != TERMINATED {
            let d = self.current;
            for m in &mut self.members {
                if m.doc() == d {
                    m.advance();
                }
            }
        }
        self.update()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        if self.members.len() == 1 {
            self.primed = true;
            self.current = self.members[0].seek(target);
            return self.current;
        }
        for m in &mut self.members {
            if !self.primed || (m.doc() != TERMINATED && m.doc() < target) {
                m.seek(target);
            }
        }
        self.primed = true;
        self.update()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.members.iter().map(|m| m.size_hint()).fold(0u32, |a, b| a.saturating_add(b))
    }
}

/// Exact or sloppy phrase matching over N positional postings lists.
///
/// A doc matches when there exist positions `p_0..p_n` with
/// `p_i - p_0 == offset_i` (exact), or when the positions can be aligned
/// with at most `slop` total edit moves (ES `match_phrase` with `slop`,
/// approximated as Lucene's sloppy phrase: max distance between aligned
/// positions after subtracting the query offsets <= slop).
///
/// Each query position may accept several alternative terms
/// ([`PhraseScorer::new_multi`], Lucene's `MultiPhraseQuery`).
///
/// Two-phase execution: an approximation ANDs the terms' doc streams led by
/// the rarest term (doc ids only), then only surviving docs decode
/// positions, rarest term first, normalised by `max_offset - offset`
/// (never underflows) and merged in place in scorer-owned buffers (no
/// allocation per candidate). Two single-term positions stream the second
/// term's packed positions against the first's.
///
/// The phrase frequency is scored through a [`SimWeight`] as the term
/// frequency (for BM25 that is the summed IDF of the terms times the
/// BM25 tf part, as Lucene's `PhraseWeight` does).
pub struct PhraseScorer {
    /// Sorted by `size_hint` ascending (rarest first).
    terms: Vec<PositionGroup>,
    /// Normalisation shift per (sorted) term: `max_offset - offset`.
    shift: Vec<u32>,
    /// Index (in `terms`) of the query's first position (sloppy anchor).
    anchor: usize,
    slop: u32,
    weight: Box<dyn SimWeight>,
    norms: Arc<Norms>,
    current: DocId,
    phrase_freq: u32,
    primed: bool,
    left: Vec<u32>,
    right: Vec<u32>,
}

impl PhraseScorer {
    /// Build a phrase scorer. `terms` are `(postings, offset)` pairs in
    /// query order; offsets are relative to the first term (so the first
    /// is 0). Returns `None` if any postings list lacks positions.
    pub fn new(
        terms: Vec<(PostingsReader, u32)>,
        slop: u32,
        weight: impl SimWeight + 'static,
        norms: Arc<Norms>,
    ) -> Option<Self> {
        let groups = terms.into_iter().map(|(p, o)| (vec![p], o)).collect();
        Self::new_multi(groups, slop, Box::new(weight), norms)
    }

    /// Multi-phrase: each entry is the set of alternative postings allowed
    /// at that query offset (any of them satisfies the position). Returns
    /// `None` if a position has no alternatives or any list lacks positions.
    pub fn new_multi(
        groups: Vec<(Vec<PostingsReader>, u32)>,
        slop: u32,
        weight: Box<dyn SimWeight>,
        norms: Arc<Norms>,
    ) -> Option<Self> {
        if groups.is_empty()
            || groups.iter().any(|(g, _)| g.is_empty() || g.iter().any(|p| !p.has_positions()))
        {
            return None;
        }
        let max_off = groups.iter().map(|(_, o)| *o).max().unwrap_or(0);
        let mut v: Vec<(PositionGroup, u32, bool)> = groups
            .into_iter()
            .enumerate()
            .map(|(i, (g, o))| (PositionGroup::new(g), max_off - o, i == 0))
            .collect();
        v.sort_by_key(|(g, _, _)| g.size_hint());
        let anchor = v.iter().position(|x| x.2).unwrap_or(0);
        let shift = v.iter().map(|x| x.1).collect();
        Some(Self {
            terms: v.into_iter().map(|x| x.0).collect(),
            shift,
            anchor,
            slop,
            weight,
            norms,
            current: TERMINATED,
            phrase_freq: 0,
            primed: false,
            left: Vec::with_capacity(64),
            right: Vec::with_capacity(64),
        })
    }

    /// Phrase frequency in the current doc.
    pub fn freq(&self) -> u32 {
        self.phrase_freq
    }

    /// Field length of the current doc.
    pub fn field_len(&self) -> u32 {
        self.norms.len(self.current)
    }

    /// The weight this scorer scores through.
    pub fn weight(&self) -> &dyn SimWeight {
        &*self.weight
    }

    /// Next doc (>= `target`, a doc of the rarest term) that contains every
    /// position group (doc ids only).
    #[inline]
    fn approx(&mut self, mut target: DocId) -> DocId {
        let k = self.terms.len();
        'outer: loop {
            if target == TERMINATED {
                return TERMINATED;
            }
            for i in 1..k {
                let d = self.terms[i].seek(target);
                if d != target {
                    target = self.terms[0].seek(d);
                    continue 'outer;
                }
            }
            return target;
        }
    }

    #[inline]
    fn load(&mut self, i: usize, into_left: bool) {
        let s = self.shift[i];
        let buf = if into_left { &mut self.left } else { &mut self.right };
        self.terms[i].positions_into(buf, s);
    }

    /// Phrase frequency of the current (approximated) doc.
    fn confirm(&mut self) -> u32 {
        let k = self.terms.len();
        if k == 1 {
            self.load(0, true);
            return self.left.len() as u32;
        }
        if self.slop == 0 && k == 2 && self.terms[0].members.len() == 1 && self.terms[1].members.len() == 1 {
            // Decode the term with fewer positions here, stream the other's
            // packed positions against it.
            let (a, b) = if self.terms[0].freq() <= self.terms[1].freq() { (0, 1) } else { (1, 0) };
            self.load(a, true);
            let sb = self.shift[b];
            return self.terms[b].members[0].count_shifted_matches(&self.left, sb);
        }
        if self.slop == 0 {
            self.load(0, true);
            for i in 1..k {
                self.load(i, false);
                if i + 1 == k {
                    return count_intersection(&self.left, &self.right);
                }
                intersect_in_place(&mut self.left, &self.right);
                if self.left.is_empty() {
                    return 0;
                }
            }
            self.left.len() as u32
        } else {
            // Anchor (query position 0) positions with every other position
            // group within `slop` after normalisation.
            let a = self.anchor;
            self.load(a, true);
            for i in 0..k {
                if i == a {
                    continue;
                }
                self.load(i, false);
                filter_within(&mut self.left, &self.right, self.slop);
                if self.left.is_empty() {
                    return 0;
                }
            }
            self.left.len() as u32
        }
    }

    /// From a candidate doc of the rarest term, find the next match.
    fn find_match(&mut self, mut d: DocId) -> DocId {
        loop {
            let doc = self.approx(d);
            if doc == TERMINATED {
                self.current = TERMINATED;
                return TERMINATED;
            }
            let f = self.confirm();
            if f > 0 {
                self.current = doc;
                self.phrase_freq = f;
                return doc;
            }
            d = self.terms[0].advance();
        }
    }
}

/// |a ∩ b| for sorted, deduplicated inputs.
#[inline]
fn count_intersection(a: &[u32], b: &[u32]) -> u32 {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        c += (x == y) as u32;
        i += (x <= y) as usize;
        j += (y <= x) as usize;
    }
    c
}

#[inline]
fn intersect_in_place(a: &mut Vec<u32>, b: &[u32]) {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        if x == y {
            a[c] = x;
            c += 1;
        }
        i += (x <= y) as usize;
        j += (y <= x) as usize;
    }
    a.truncate(c);
}

/// Keep anchors `x` in `a` with some `y` in `b`, |x - y| <= slop.
#[inline]
fn filter_within(a: &mut Vec<u32>, b: &[u32], slop: u32) {
    let (mut j, mut c) = (0, 0);
    for i in 0..a.len() {
        let x = a[i];
        while j < b.len() && b[j].saturating_add(slop) < x {
            j += 1;
        }
        if j == b.len() {
            break;
        }
        if b[j] <= x.saturating_add(slop) {
            a[c] = x;
            c += 1;
        }
    }
    a.truncate(c);
}

impl DocSet for PhraseScorer {
    fn advance(&mut self) -> DocId {
        let d = if !self.primed {
            self.primed = true;
            self.terms[0].advance()
        } else if self.current == TERMINATED {
            return TERMINATED;
        } else {
            self.terms[0].advance()
        };
        self.find_match(d)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        if self.primed && self.current == TERMINATED {
            return TERMINATED;
        }
        self.primed = true;
        let d = self.terms[0].seek(target);
        self.find_match(d)
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.terms.iter().map(|t| t.size_hint()).min().unwrap_or(0)
    }
}

impl Scorer for PhraseScorer {
    fn score(&mut self) -> f32 {
        self.weight.score_norm(self.phrase_freq, self.norms.byte(self.current))
    }
}

/// Build a phrase scorer for consecutive `terms` in `segment`.
pub fn segment_phrase_scorer(
    segment: &Segment,
    terms: &[Term],
    slop: u32,
    params: Bm25Params,
) -> Option<Box<dyn Scorer>> {
    let field = terms.first()?.field;
    let mut readers = Vec::with_capacity(terms.len());
    for (i, t) in terms.iter().enumerate() {
        readers.push((segment.postings(t)?, i as u32));
    }
    let norms = Arc::new(segment.norms_shared(field));
    // Rarest term drives IDF, as Lucene's PhraseWeight sums per-term idfs;
    // we sum idf over terms by building a weight with the min doc_freq and
    // scaling is left to the caller's boost. Summing idfs:
    let num_docs = segment.num_docs as u64;
    let idf_sum: f32 = terms
        .iter()
        .map(|t| crate::query::bm25::idf(segment.term_dict().doc_freq(t) as u64, num_docs))
        .sum();
    // Bm25Weight computes idf from df; pick a df that reproduces idf_sum.
    // Simpler: construct with df=1 then scale. We expose scaling via a
    // wrapper scorer instead.
    let base = Bm25Weight::new(1, num_docs, params, norms.avg());
    let base_idf = crate::query::bm25::idf(1, num_docs);
    let scale = if base_idf > 0.0 { idf_sum / base_idf } else { 1.0 };
    let inner = PhraseScorer::new(readers, slop, base, norms)?;
    Some(Box::new(Scaled { inner, scale }))
}

/// Multiplies an inner scorer's score by a constant (query boost).
pub struct Scaled<S: Scorer> {
    inner: S,
    scale: f32,
}

impl<S: Scorer> Scaled<S> {
    pub fn new(inner: S, scale: f32) -> Self {
        Self { inner, scale }
    }
}

impl<S: Scorer> DocSet for Scaled<S> {
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
    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        self.inner.collect_first_k_bulk(k, deleted)
    }
}

impl<S: Scorer> Scorer for Scaled<S> {
    fn score(&mut self) -> f32 {
        self.inner.score() * self.scale
    }

    fn collect_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> Option<f32> {
        self.inner.collect_all(deleted, out).map(|s| s * self.scale)
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        // A positive scale keeps the order; scores are scaled afterwards,
        // exactly as the per-doc path computes them.
        if !(self.scale > 0.0) {
            return None;
        }
        let (mut hits, n) = self.inner.collect_top_k_bulk(k, deleted)?;
        for h in &mut hits {
            h.score *= self.scale;
        }
        Some((hits, n))
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, min_competitive: f32) -> Option<(Vec<ScoredDoc>, u64)> {
        if !(self.scale > 0.0) {
            return None;
        }
        // Conservative inner floor: rounding of `s * scale` must not push a
        // competitive hit below it.
        let floor = if min_competitive.is_finite() {
            let x = min_competitive / self.scale;
            x - x.abs() * 1e-5 - f32::EPSILON
        } else {
            min_competitive
        };
        let (mut hits, n) = self.inner.collect_top_k_bulk_min(k, deleted, floor)?;
        for h in &mut hits {
            h.score *= self.scale;
        }
        Some((hits, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::RamDirectory;
    use crate::query::collector::collect_top_k;
    use crate::segment::SegmentWriter;
    use crate::types::{Document, Field, FieldType};

    fn build() -> Segment {
        let dir = RamDirectory::new();
        let schema = vec![Field::new(0, "body", FieldType::Text)];
        let mut w = SegmentWriter::new(0);
        let texts = [
            "the quick brown fox",             // 0
            "quick brown dogs and a brown fox", // 1
            "fox quick brown",                 // 2: "quick brown" present, "brown fox" not
            "the quick red fox",               // 3: slop 1 matches "quick fox"? quick(1) fox(3)
            "brown",                           // 4
            "quick brown quick brown",         // 5: phrase twice
        ];
        for t in texts {
            let mut d = Document::new();
            d.add_text(0, t);
            w.add_document(&d, &schema);
        }
        w.write_to(&dir).unwrap()
    }

    fn docs_of(mut s: Box<dyn Scorer>) -> Vec<(DocId, u32)> {
        let mut out = Vec::new();
        loop {
            let d = s.advance();
            if d == TERMINATED {
                break;
            }
            out.push((d, 0));
        }
        out
    }

    #[test]
    fn term_scorer_ranks_by_tf_and_length() {
        let seg = build();
        let mut s = segment_term_scorer(&seg, &Term::new_text(0, "brown"), Bm25Params::default()).unwrap();
        let (hits, total) = collect_top_k(&mut *s, 10, |_| true);
        assert_eq!(total, 5);
        // doc 4 ("brown", length 1) and doc 5 (tf 2) should outrank doc 1 (tf 2, length 7).
        let top: Vec<DocId> = hits.iter().map(|h| h.doc).collect();
        assert!(top[0] == 4 || top[0] == 5, "{top:?}");
        let pos1 = top.iter().position(|&d| d == 1).unwrap();
        let pos0 = top.iter().position(|&d| d == 0).unwrap();
        assert!(pos1 < pos0, "tf=2 in len 7 should beat tf=1 in len 4: {top:?}");
    }

    #[test]
    fn exact_phrase() {
        let seg = build();
        let terms = [Term::new_text(0, "quick"), Term::new_text(0, "brown")];
        let s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        let docs: Vec<DocId> = docs_of(s).into_iter().map(|(d, _)| d).collect();
        assert_eq!(docs, vec![0, 1, 2, 5]);

        let terms = [Term::new_text(0, "brown"), Term::new_text(0, "fox")];
        let s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        let docs: Vec<DocId> = docs_of(s).into_iter().map(|(d, _)| d).collect();
        assert_eq!(docs, vec![0, 1]);

        // Three-term phrase.
        let terms = [Term::new_text(0, "quick"), Term::new_text(0, "brown"), Term::new_text(0, "fox")];
        let s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        let docs: Vec<DocId> = docs_of(s).into_iter().map(|(d, _)| d).collect();
        assert_eq!(docs, vec![0]);

        // Missing term → None.
        assert!(segment_phrase_scorer(&seg, &[Term::new_text(0, "zzz")], 0, Bm25Params::default()).is_none());
    }

    #[test]
    fn sloppy_phrase_and_phrase_freq_scoring() {
        let seg = build();
        let terms = [Term::new_text(0, "quick"), Term::new_text(0, "fox")];
        // exact: none ("quick fox" never adjacent)
        let s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        assert!(docs_of(s).is_empty());
        // slop 1: docs 0 ("quick brown fox"), 3 ("quick red fox")
        let s = segment_phrase_scorer(&seg, &terms, 1, Bm25Params::default()).unwrap();
        let docs: Vec<DocId> = docs_of(s).into_iter().map(|(d, _)| d).collect();
        assert_eq!(docs, vec![0, 3]);
        // slop 2 also admits doc 2 ("fox quick brown": fox(0) quick(1) → reversed, distance 2)
        let s = segment_phrase_scorer(&seg, &terms, 2, Bm25Params::default()).unwrap();
        let docs: Vec<DocId> = docs_of(s).into_iter().map(|(d, _)| d).collect();
        assert_eq!(docs, vec![0, 2, 3]);

        // Phrase freq: doc 5 has "quick brown" twice and should outscore doc 0.
        let terms = [Term::new_text(0, "quick"), Term::new_text(0, "brown")];
        let mut s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        let (hits, _) = collect_top_k(&mut *s, 10, |_| true);
        assert_eq!(hits[0].doc, 5);
    }

    #[test]
    fn phrase_seek() {
        let seg = build();
        let terms = [Term::new_text(0, "quick"), Term::new_text(0, "brown")];
        let mut s = segment_phrase_scorer(&seg, &terms, 0, Bm25Params::default()).unwrap();
        assert_eq!(s.seek(1), 1);
        assert_eq!(s.seek(2), 2);
        assert_eq!(s.seek(3), 5);
        assert_eq!(s.advance(), TERMINATED);
    }
}
