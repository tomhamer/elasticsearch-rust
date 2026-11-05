//! Index sorting (ES `index.sort.*`, Lucene `IndexWriterConfig#setIndexSort`):
//! every segment written by a flush or a merge holds its documents in the
//! configured order, so `_doc` order (and early termination on a sort that
//! matches it) follows the index sort.
//!
//! Comparison is Lucene's: per sort field the doc's value picked by `mode`
//! (`min` / `max` of a multi-valued field), numeric or byte order, reversed
//! for descending sorts; docs without a value go first or last as
//! configured (independent of the direction). Ties keep the input order
//! (a flush keeps the indexing order; a merge interleaves its inputs in
//! segment order, as Lucene's `MultiSorter`).

use crate::query::reader::LeafReader;
use crate::types::DocId;
use std::cmp::Ordering;

/// Docs from which [`IndexSort::sort_docs_keyed`] sorts in parallel.
const PARALLEL_SORT_DOCS: usize = 4096;

/// The value type a sort field is compared as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexSortKind {
    /// i64 column (long / integer / short / byte / date / boolean).
    Long,
    /// u64 column (`unsigned_long`), compared as unsigned.
    UnsignedLong,
    /// f64 column (double / float / half_float / scaled_float).
    Double,
    /// keyword column, compared as bytes.
    Keyword,
}

/// One sort field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexSortField {
    pub field: u16,
    pub kind: IndexSortKind,
    /// Descending.
    pub reverse: bool,
    /// Multi-valued docs sort by their largest value (else smallest).
    pub max_mode: bool,
    /// Docs without a value sort before every doc with one.
    pub missing_first: bool,
}

/// The index sort: fields compared in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexSort {
    pub fields: Vec<IndexSortField>,
}

/// One doc's value of one sort field.
#[derive(Clone, Debug, PartialEq)]
enum Key {
    Missing,
    I(i64),
    U(u64),
    F(f64),
    S(String),
}

impl IndexSort {
    pub fn new(fields: Vec<IndexSortField>) -> Self {
        IndexSort { fields }
    }

    /// Admit the full sorting peak before roots, key rows or strings are
    /// materialized. Production index-sort inputs are immutable segments.
    pub(crate) fn reserve_workspace(&self, segments: &[std::sync::Arc<crate::segment::Segment>]) -> crate::error::Result<crate::admission::Reservation> {
        self.reserve_workspace_with_budget(segments,&crate::admission::shared())
    }

    pub(crate) fn reserve_workspace_with_budget(&self, segments: &[std::sync::Arc<crate::segment::Segment>], budget:&std::sync::Arc<crate::admission::Budget>) -> crate::error::Result<crate::admission::Reservation> {
        // Segments already in sort order (every flushed and merged segment
        // of a sorted index) merge k ways: the keys of one head per
        // segment, plus the roots and the order (`(usize, DocId)` each).
        if segments.len() > 1 && segments.iter().all(|s| self.leaf_sorted(&LeafReader::Segment(s), 0..s.num_docs)) {
            let docs: u64 = segments.iter().map(|s| s.num_docs as u64).sum();
            let bytes = docs.saturating_mul(2 * std::mem::size_of::<(usize, DocId)>() as u64).saturating_add((segments.len() as u64).saturating_mul(4096)).saturating_add(4096);
            return budget.reserve(bytes, "index sort roots keys and remapping workspace");
        }
        let mut bytes=4096u64;
        for segment in segments {
            bytes=bytes.saturating_add((segment.num_docs as u64).saturating_mul(256)).saturating_add(512);
            let fields=segment.fast_fields();
            for field in &self.fields {
                bytes=bytes.saturating_add((segment.num_docs as u64).saturating_mul(128));
                match field.kind {
                    IndexSortKind::Long => if let Some(c)=fields.get_multi_i64(field.field){bytes=bytes.saturating_add((c.total_values() as u64).saturating_mul(32));},
                    IndexSortKind::UnsignedLong => if let Some(c)=fields.get_multi_u64(field.field){bytes=bytes.saturating_add((c.total_values() as u64).saturating_mul(32));},
                    IndexSortKind::Double => if let Some(c)=fields.get_multi_f64(field.field){bytes=bytes.saturating_add((c.total_values() as u64).saturating_mul(32));},
                    IndexSortKind::Keyword => {
                        if let Some(c)=fields.get_keyword(field.field){for doc in 0..segment.num_docs {if let Some(v)=c.get(doc){bytes=bytes.saturating_add((v.len() as u64).saturating_mul(8).saturating_add(128));}}}
                        if let Some(c)=fields.get_multi_keyword(field.field){for doc in 0..segment.num_docs {for ord in c.ords(doc){if let Some(v)=c.term(ord){bytes=bytes.saturating_add((v.len() as u64).saturating_mul(8).saturating_add(128));}}}}
                    }
                }
            }
        }
        budget.reserve(bytes,"index sort roots keys and remapping workspace")
    }

    fn keys(&self, leaf: &LeafReader<'_>, doc: DocId) -> Vec<Key> {
        self.fields
            .iter()
            .map(|f| {
                let max = f.max_mode;
                match f.kind {
                    IndexSortKind::Long => {
                        let v = leaf.i64_values(f.field, doc);
                        let pick = if max { v.iter().max() } else { v.iter().min() };
                        pick.map_or(Key::Missing, |x| Key::I(*x))
                    }
                    IndexSortKind::UnsignedLong => {
                        let v = leaf.u64_values(f.field, doc);
                        let pick = if max { v.iter().max() } else { v.iter().min() };
                        pick.map_or(Key::Missing, |x| Key::U(*x))
                    }
                    IndexSortKind::Double => {
                        let v = leaf.f64_values(f.field, doc);
                        let pick = if max { v.iter().copied().max_by(f64::total_cmp) } else { v.iter().copied().min_by(f64::total_cmp) };
                        pick.map_or(Key::Missing, Key::F)
                    }
                    IndexSortKind::Keyword => {
                        let v = leaf.keyword_values(f.field, doc);
                        let pick = if max { v.into_iter().max() } else { v.into_iter().min() };
                        pick.map_or(Key::Missing, Key::S)
                    }
                }
            })
            .collect()
    }

    /// Whether `docs` of `leaf` (ascending) are in sort order.
    fn leaf_sorted(&self, leaf: &LeafReader<'_>, docs: impl Iterator<Item = DocId>) -> bool {
        let mut prev: Option<Vec<Key>> = None;
        for d in docs {
            let k = self.keys(leaf, d);
            if prev.as_ref().is_some_and(|p| self.cmp(p, &k) == Ordering::Greater) {
                return false;
            }
            prev = Some(k);
        }
        true
    }

    /// [`IndexSort::sort_docs`] of docs grouped by leaf (leaves ascending,
    /// docs ascending within each) from several leaves each already in
    /// sort order: a k-way merge holding one key per leaf (ties: the
    /// earlier leaf, as the stable sort keeps input order). `None` when the
    /// input is not grouped that way or a leaf is not sorted.
    fn merge_sorted(&self, leaves: &[LeafReader<'_>], docs: &[(usize, DocId)]) -> Option<Vec<(usize, DocId)>> {
        if leaves.len() < 2 || docs.windows(2).any(|w| w[0] >= w[1]) {
            return None;
        }
        // Each leaf's run in `docs`.
        let mut runs: Vec<(usize, usize)> = vec![(0, 0); leaves.len()];
        let mut i = 0;
        while i < docs.len() {
            let l = docs[i].0;
            let start = i;
            while i < docs.len() && docs[i].0 == l {
                i += 1;
            }
            runs[l] = (start, i);
        }
        for (l, &(a, b)) in runs.iter().enumerate() {
            if !self.leaf_sorted(&leaves[l], docs[a..b].iter().map(|x| x.1)) {
                return None;
            }
        }
        struct Head {
            key: Vec<Key>,
            leaf: usize,
            pos: usize,
        }
        let mut heads: Vec<Head> = runs.iter().enumerate().filter(|(_, r)| r.0 < r.1).map(|(l, r)| Head { key: self.keys(&leaves[l], docs[r.0].1), leaf: l, pos: r.0 }).collect();
        let mut out = Vec::with_capacity(docs.len());
        while !heads.is_empty() {
            // Few leaves: a linear scan for the smallest head.
            let mut best = 0;
            for h in 1..heads.len() {
                let o = self.cmp(&heads[h].key, &heads[best].key);
                if o == Ordering::Less || (o == Ordering::Equal && heads[h].leaf < heads[best].leaf) {
                    best = h;
                }
            }
            let h = &mut heads[best];
            out.push(docs[h.pos]);
            h.pos += 1;
            if h.pos < runs[h.leaf].1 {
                h.key = self.keys(&leaves[h.leaf], docs[h.pos].1);
            } else {
                heads.swap_remove(best);
            }
        }
        Some(out)
    }

    fn cmp(&self, a: &[Key], b: &[Key]) -> Ordering {
        for (f, (x, y)) in self.fields.iter().zip(a.iter().zip(b)) {
            let o = match (x, y) {
                (Key::Missing, Key::Missing) => Ordering::Equal,
                (Key::Missing, _) => {
                    if f.missing_first {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (_, Key::Missing) => {
                    if f.missing_first {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (x, y) => {
                    let o = match (x, y) {
                        (Key::I(a), Key::I(b)) => a.cmp(b),
                        (Key::U(a), Key::U(b)) => a.cmp(b),
                        (Key::F(a), Key::F(b)) => a.total_cmp(b),
                        (Key::S(a), Key::S(b)) => a.as_bytes().cmp(b.as_bytes()),
                        _ => Ordering::Equal,
                    };
                    if f.reverse {
                        o.reverse()
                    } else {
                        o
                    }
                }
            };
            if o != Ordering::Equal {
                return o;
            }
        }
        Ordering::Equal
    }

    /// The sorted order of `docs` (`(leaf index, doc)` pairs in input
    /// order) over `leaves`: a stable sort, so ties keep the input order.
    pub fn sort_docs(&self, leaves: &[LeafReader<'_>], docs: Vec<(usize, DocId)>) -> Vec<(usize, DocId)> {
        if let Some(merged) = self.merge_sorted(leaves, &docs) {
            return merged;
        }
        self.sort_docs_keyed(leaves, docs)
    }

    /// [`IndexSort::sort_docs`] by every doc's materialized keys.
    pub(crate) fn sort_docs_keyed(&self, leaves: &[LeafReader<'_>], docs: Vec<(usize, DocId)>) -> Vec<(usize, DocId)> {
        // Many docs (a refresh's sealed memtables): the keys read and the
        // stable sort run in parallel (rayon's `par_sort_by` is stable: the
        // same order).
        // On the encode pool, never the global one: seals and merges sort
        // here, and the global pool runs the ES layer's parallel bulk writes,
        // which wait in `wait_for_seals` for these very seals (a sort queued
        // behind them stalled bulks for minutes on the joins track).
        if docs.len() >= PARALLEL_SORT_DOCS {
            return crate::on_encode_pool(|| {
                use rayon::prelude::*;
                let mut keyed: Vec<(Vec<Key>, (usize, DocId))> = docs.into_par_iter().map(|(l, d)| (self.keys(&leaves[l], d), (l, d))).collect();
                keyed.par_sort_by(|a, b| self.cmp(&a.0, &b.0));
                keyed.into_iter().map(|(_, x)| x).collect()
            });
        }
        let mut keyed: Vec<(Vec<Key>, (usize, DocId))> = docs.into_iter().map(|(l, d)| (self.keys(&leaves[l], d), (l, d))).collect();
        keyed.sort_by(|a, b| self.cmp(&a.0, &b.0));
        keyed.into_iter().map(|(_, x)| x).collect()
    }
}
