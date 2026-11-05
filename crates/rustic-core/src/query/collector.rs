//! Result collection: top-k gathering and counting.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::DocId;
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::scorer::Scorer;

/// A doc id with its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDoc {
    /// The doc id.
    pub doc: DocId,
    /// Its score.
    pub score: f32,
}

/// Heap entry ordered so that the *worst* hit (lowest score, then highest
/// doc id) is at the top of a max-heap.
#[derive(PartialEq)]
struct Worst(ScoredDoc);

impl Eq for Worst {}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or(Ordering::Equal)
            .then(self.0.doc.cmp(&other.0.doc))
    }
}

/// Collects the top-k docs by score.
///
/// A min-heap on score keeps only the k best seen so far; ties are broken by
/// lower doc id first (a doc with the same score but a higher id is evicted).
pub struct TopKCollector {
    k: usize,
    heap: BinaryHeap<Worst>,
}

impl TopKCollector {
    /// Collect up to `k` docs.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    /// Offer a doc and its score.
    pub fn collect(&mut self, doc: DocId, score: f32) {
        if self.k == 0 {
            return;
        }
        let entry = Worst(ScoredDoc { doc, score });
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            // Replace the worst (lowest score, or highest doc id on ties)
            // when the new hit beats it.
            let w = worst.0;
            if score > w.score || (score == w.score && doc < w.doc) {
                *worst = entry;
            }
        }
    }

    /// Lowest retained score once k hits are held. Equal scores may still
    /// compete through their doc-id tie break, so pruning must be strict.
    pub fn min_score(&self) -> Option<f32> {
        (self.heap.len() == self.k).then(|| self.heap.peek().map(|h| h.0.score)).flatten()
    }

    /// The collected docs, sorted by score descending, then doc id ascending.
    pub fn into_sorted_vec(self) -> Vec<ScoredDoc> {
        let mut v: Vec<ScoredDoc> = self.heap.into_iter().map(|w| w.0).collect();
        v.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.doc.cmp(&b.doc))
        });
        v
    }
}

impl Default for TopKCollector {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Counts the docs it is offered.
pub struct CountCollector {
    count: u64,
}

impl CountCollector {
    /// A fresh counter.
    pub fn new() -> Self {
        Self { count: 0 }
    }

    /// Count one doc.
    pub fn collect(&mut self) {
        self.count += 1;
    }

    /// The number of docs collected.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for CountCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Drive a scorer to completion, collecting the top-k docs for which
/// `accept` returns true (e.g. not deleted).
///
/// Returns the top-k results (score desc, doc asc) and the total number of
/// accepted docs.
pub fn collect_top_k(
    scorer: &mut dyn Scorer,
    k: usize,
    mut accept: impl FnMut(DocId) -> bool,
) -> (Vec<ScoredDoc>, u64) {
    let mut collector = TopKCollector::new(k);
    let mut total = 0u64;
    loop {
        let d = scorer.advance();
        if d == TERMINATED {
            break;
        }
        if accept(d) {
            total += 1;
            collector.collect(d, scorer.score());
        }
    }
    (collector.into_sorted_vec(), total)
}

/// [`collect_top_k`] for a segment's live docs given as its deletion bitset
/// words (`None` = no deletions). Uses the scorer's bulk path when it has
/// one (block-at-a-time term/AND/OR scoring), else the per-doc loop.
pub fn collect_top_k_live(scorer: &mut dyn Scorer, k: usize, deleted: Option<&[u64]>) -> (Vec<ScoredDoc>, u64) {
    if let Some(r) = scorer.collect_top_k_bulk(k, deleted) {
        return r;
    }
    match deleted {
        None => collect_top_k(scorer, k, |_| true),
        Some(del) => collect_top_k(scorer, k, |d| del.get(d as usize >> 6).is_none_or(|w| (w >> (d & 63)) & 1 == 0)),
    }
}

/// A sortable value for field sorting (Lucene `SortField` types). Ordering
/// of `Missing` is decided by the sort's `missing_first`.
#[derive(Clone, Debug, PartialEq)]
pub enum SortValue {
    I64(i64),
    F64(f64),
    Str(String),
    Missing,
}

impl SortValue {
    /// Whether `self` is the value a missing one sorts as (ES substitutes
    /// it: Long.MIN / MAX, -/+Infinity): a real value equal to it ties with
    /// a missing one, and the tiebreaker decides.
    pub(crate) fn is_missing_sentinel(&self, missing_first: bool) -> bool {
        match self {
            SortValue::I64(v) => *v == if missing_first { i64::MIN } else { i64::MAX },
            SortValue::F64(v) => *v == if missing_first { f64::NEG_INFINITY } else { f64::INFINITY },
            _ => false,
        }
    }

    fn cmp_asc(&self, other: &SortValue, missing_first: bool) -> Ordering {
        use SortValue::*;
        match (self, other) {
            (Missing, Missing) => Ordering::Equal,
            (Missing, v) | (v, Missing) if v.is_missing_sentinel(missing_first) => Ordering::Equal,
            (Missing, _) => if missing_first { Ordering::Less } else { Ordering::Greater },
            (_, Missing) => if missing_first { Ordering::Greater } else { Ordering::Less },
            (I64(a), I64(b)) => a.cmp(b),
            (F64(a), F64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Str(a), Str(b)) => a.cmp(b),
            (I64(a), F64(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
            (F64(a), I64(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
            (Str(_), _) => Ordering::Greater,
            (_, Str(_)) => Ordering::Less,
        }
    }
}

/// One sort criterion.
#[derive(Clone, Debug)]
pub struct SortSpec {
    pub descending: bool,
    pub missing_first: bool,
}

/// A hit with its sort key (Lucene `FieldDoc`).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldDoc {
    pub doc: DocId,
    pub score: f32,
    pub sort: Vec<SortValue>,
}

/// Compare two sort keys under `specs`; score desc then doc asc break ties.
pub fn compare_field_docs(a: &FieldDoc, b: &FieldDoc, specs: &[SortSpec]) -> Ordering {
    compare_sort_keys(&a.sort, a.score, a.doc, &b.sort, b.score, b.doc, specs)
}

/// [`compare_field_docs`] over borrowed parts (no `FieldDoc` needed).
pub fn compare_sort_keys(
    a_sort: &[SortValue],
    a_score: f32,
    a_doc: DocId,
    b_sort: &[SortValue],
    b_score: f32,
    b_doc: DocId,
    specs: &[SortSpec],
) -> Ordering {
    for (i, spec) in specs.iter().enumerate() {
        let (x, y) = (
            a_sort.get(i).unwrap_or(&SortValue::Missing),
            b_sort.get(i).unwrap_or(&SortValue::Missing),
        );
        let o = x.cmp_asc(y, spec.missing_first);
        let o = if spec.descending { o.reverse() } else { o };
        if o != Ordering::Equal {
            return o;
        }
    }
    // Lucene's field comparators break ties by doc id only (never by score).
    let _ = (a_score, b_score);
    a_doc.cmp(&b_doc)
}

/// Top-k by arbitrary sort keys (Lucene `TopFieldCollector`), with
/// `search_after` support: hits that sort at or before `after` are skipped.
pub struct TopFieldCollector {
    k: usize,
    specs: std::sync::Arc<Vec<SortSpec>>,
    after: Option<FieldDoc>,
    /// Hits tied with `after` on every sort value are after it whatever
    /// their doc id (`after` comes from an earlier leaf).
    after_ties: bool,
    heap: BinaryHeap<FieldWorst>,
}

struct FieldWorst {
    doc: FieldDoc,
    specs: std::sync::Arc<Vec<SortSpec>>,
}

impl PartialEq for FieldWorst {
    fn eq(&self, other: &Self) -> bool {
        compare_field_docs(&self.doc, &other.doc, &self.specs) == Ordering::Equal
    }
}
impl Eq for FieldWorst {}
impl PartialOrd for FieldWorst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FieldWorst {
    // Max-heap: the "greatest" is the worst hit (sorts last).
    fn cmp(&self, other: &Self) -> Ordering {
        compare_field_docs(&self.doc, &other.doc, &self.specs)
    }
}

impl TopFieldCollector {
    pub fn new(k: usize, specs: Vec<SortSpec>, after: Option<FieldDoc>) -> Self {
        Self {
            k,
            specs: std::sync::Arc::new(specs),
            after,
            after_ties: false,
            heap: BinaryHeap::with_capacity(k.min(1 << 16)),
        }
    }

    /// Treat hits tied with `after` on every sort value as after it (the
    /// `after` hit lives on an earlier leaf, whose doc ids all precede).
    pub fn with_ties_after(mut self) -> Self {
        self.after_ties = true;
        self
    }

    /// Offer a hit with its materialized sort values.
    pub fn collect(&mut self, doc: FieldDoc) {
        if self.k == 0 {
            return;
        }
        if let Some(after) = &self.after {
            match compare_field_docs(&doc, after, &self.specs) {
                Ordering::Greater => {}
                _ if self.after_ties && compare_sort_keys(&doc.sort, 0.0, 0, &after.sort, 0.0, 0, &self.specs) == Ordering::Equal => {}
                _ => return,
            }
        }
        if self.heap.len() < self.k {
            self.heap.push(FieldWorst { doc, specs: self.specs.clone() });
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if compare_field_docs(&doc, &worst.doc, &self.specs) == Ordering::Less {
                worst.doc = doc;
            }
        }
    }

    /// A worse primary key cannot enter a full heap. Equal primary keys
    /// still need the remaining fields and doc-id tiebreaker.
    pub(crate) fn primary_competitive(&self, value: &SortValue) -> bool {
        if self.k == 0 { return false }
        if self.heap.len() < self.k { return true }
        let Some(spec) = self.specs.first() else { return true };
        let Some(worst) = self.heap.peek().and_then(|w| w.doc.sort.first()) else { return true };
        let order = value.cmp_asc(worst, spec.missing_first);
        let order = if spec.descending { order.reverse() } else { order };
        order != Ordering::Greater
    }

    /// Hits in sort order.
    pub fn into_sorted_vec(self) -> Vec<FieldDoc> {
        let specs = self.specs;
        let mut v: Vec<FieldDoc> = self.heap.into_iter().map(|w| w.doc).collect();
        v.sort_by(|a, b| compare_field_docs(a, b, &specs));
        v
    }
}

/// Drive a scorer, materializing sort keys with `key_of(doc)` and keeping the
/// top-k under `specs`. Returns (hits, total accepted).
pub fn collect_top_k_sorted(
    scorer: &mut dyn Scorer,
    k: usize,
    specs: Vec<SortSpec>,
    after: Option<FieldDoc>,
    mut accept: impl FnMut(DocId) -> bool,
    mut key_of: impl FnMut(DocId) -> Vec<SortValue>,
) -> (Vec<FieldDoc>, u64) {
    let mut c = TopFieldCollector::new(k, specs, after);
    let mut total = 0u64;
    loop {
        let d = scorer.advance();
        if d == TERMINATED {
            break;
        }
        if accept(d) {
            total += 1;
            c.collect(FieldDoc {
                doc: d,
                score: scorer.score(),
                sort: key_of(d),
            });
        }
    }
    (c.into_sorted_vec(), total)
}

/// Count the docs in a doc set for which `accept` returns true.
pub fn count(docset: &mut dyn DocSet, mut accept: impl FnMut(DocId) -> bool) -> u64 {
    let mut n = 0u64;
    loop {
        let d = docset.advance();
        if d == TERMINATED {
            break;
        }
        if accept(d) {
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::docset::VecDocSet;
    use crate::query::scorer::ConstScorer;

    #[test]
    fn field_collector_sorts_and_pages() {
        // docs 0..50, key = (doc % 7) asc, then doc desc via second key
        let specs = vec![
            SortSpec { descending: false, missing_first: false },
            SortSpec { descending: true, missing_first: false },
        ];
        let key = |d: DocId| vec![SortValue::I64((d % 7) as i64), SortValue::I64(d as i64)];
        let mut s = ConstScorer::new(VecDocSet::new((0..50).collect()), 1.0);
        let (page1, total) = collect_top_k_sorted(&mut s, 5, specs.clone(), None, |_| true, key);
        assert_eq!(total, 50);
        let docs: Vec<DocId> = page1.iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![49, 42, 35, 28, 21]); // key 0, doc desc
        // search_after the last hit of page 1
        let after = page1.last().cloned();
        let mut s = ConstScorer::new(VecDocSet::new((0..50).collect()), 1.0);
        let (page2, _) = collect_top_k_sorted(&mut s, 5, specs.clone(), after, |_| true, key);
        let docs: Vec<DocId> = page2.iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![14, 7, 0, 43, 36]);

        // Missing values sort last by default and first when asked.
        let key_m = |d: DocId| if d % 2 == 0 { vec![SortValue::Missing] } else { vec![SortValue::I64(d as i64)] };
        let mut s = ConstScorer::new(VecDocSet::new((0..10).collect()), 1.0);
        let (hits, _) = collect_top_k_sorted(&mut s, 10, vec![SortSpec { descending: false, missing_first: false }], None, |_| true, key_m);
        let docs: Vec<DocId> = hits.iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![1, 3, 5, 7, 9, 0, 2, 4, 6, 8]);
        let mut s = ConstScorer::new(VecDocSet::new((0..10).collect()), 1.0);
        let (hits, _) = collect_top_k_sorted(&mut s, 3, vec![SortSpec { descending: false, missing_first: true }], None, |_| true, key_m);
        let docs: Vec<DocId> = hits.iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![0, 2, 4]);
    }

    #[test]
    fn top_k_basic() {
        let mut c = TopKCollector::new(3);
        c.collect(10, 1.0);
        c.collect(20, 5.0);
        c.collect(30, 3.0);
        c.collect(40, 4.0);
        c.collect(50, 2.0);
        let v = c.into_sorted_vec();
        assert_eq!(
            v,
            vec![
                ScoredDoc { doc: 20, score: 5.0 },
                ScoredDoc { doc: 40, score: 4.0 },
                ScoredDoc { doc: 30, score: 3.0 },
            ]
        );
    }

    #[test]
    fn top_k_tie_breaks_by_lower_doc() {
        let mut c = TopKCollector::new(2);
        c.collect(10, 1.0);
        c.collect(5, 1.0);
        c.collect(1, 1.0);
        let v = c.into_sorted_vec();
        assert_eq!(
            v,
            vec![
                ScoredDoc { doc: 1, score: 1.0 },
                ScoredDoc { doc: 5, score: 1.0 },
            ]
        );
    }

    #[test]
    fn top_k_zero_and_fewer_than_k() {
        let mut c = TopKCollector::new(0);
        c.collect(1, 9.0);
        assert!(c.into_sorted_vec().is_empty());

        let mut c = TopKCollector::new(5);
        c.collect(1, 2.0);
        c.collect(2, 1.0);
        let v = c.into_sorted_vec();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].doc, 1);
        assert_eq!(v[1].doc, 2);
    }

    #[test]
    fn count_collector() {
        let mut c = CountCollector::new();
        for _ in 0..42 {
            c.collect();
        }
        assert_eq!(c.count(), 42);
    }

    #[test]
    fn collect_top_k_drives_scorer() {
        // Score = 100 - d, so top-k is the k lowest doc ids.
        let docs: Vec<DocId> = (0..100).collect();
        let s = ConstScorer::new(VecDocSet::new(docs), 0.0);
        // Use a custom scorer via a wrapper: score depends on doc.
        struct DescScorer {
            inner: ConstScorer<VecDocSet>,
        }
        impl DocSet for DescScorer {
            fn advance(&mut self) -> DocId {
                self.inner.advance()
            }
            fn seek(&mut self, t: DocId) -> DocId {
                self.inner.seek(t)
            }
            fn doc(&self) -> DocId {
                self.inner.doc()
            }
            fn size_hint(&self) -> u32 {
                self.inner.size_hint()
            }
        }
        impl Scorer for DescScorer {
            fn score(&mut self) -> f32 {
                100.0 - self.inner.doc() as f32
            }
        }
        let mut s = DescScorer { inner: s };
        let (top, total) = collect_top_k(&mut s, 3, |_| true);
        assert_eq!(total, 100);
        assert_eq!(
            top,
            vec![
                ScoredDoc { doc: 0, score: 100.0 },
                ScoredDoc { doc: 1, score: 99.0 },
                ScoredDoc { doc: 2, score: 98.0 },
            ]
        );
        // Rejected (deleted) docs are skipped and not counted.
        let docs: Vec<DocId> = (0..100).collect();
        let mut s = DescScorer {
            inner: ConstScorer::new(VecDocSet::new(docs), 0.0),
        };
        let (top, total) = collect_top_k(&mut s, 3, |d| d % 2 == 0);
        assert_eq!(total, 50);
        assert_eq!(
            top,
            vec![
                ScoredDoc { doc: 0, score: 100.0 },
                ScoredDoc { doc: 2, score: 98.0 },
                ScoredDoc { doc: 4, score: 96.0 },
            ]
        );
    }

    #[test]
    fn count_drives_docset() {
        let docs: Vec<DocId> = (0..100).collect();
        let mut s = VecDocSet::new(docs);
        assert_eq!(count(&mut s, |_| true), 100);
        let docs: Vec<DocId> = (0..100).collect();
        let mut s = VecDocSet::new(docs);
        assert_eq!(count(&mut s, |d| d % 3 == 0), 34);
    }
}
