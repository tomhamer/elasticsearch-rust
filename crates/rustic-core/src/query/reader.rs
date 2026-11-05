//! `LeafReader`: one uniform read interface over a sealed on-disk
//! [`Segment`] and the in-memory [`SegmentWriter`] (memtable).
//!
//! Lucene's `LeafReader` only ever sees flushed segments; here the memtable
//! is a first-class leaf so a refresh never has to write a file. Query
//! planning code targets this type and does not care where docs live.

use crate::codec::postings::PostingsReader;
use crate::query::bm25::{Bm25Params, Bm25Weight};
use crate::query::bulk::Bm25TermScorer;
use crate::query::scorer::{Scorer, TermScorer};
use crate::query::explain::Explanation;
use crate::query::segment_scorer::{Norms, PhraseScorer, PostingsDocSet};
use crate::query::similarity::{Bm25, Similarity};
use crate::segment::{Segment, SegmentWriter};
use crate::term_dictionary::PostingsLocation;
use crate::types::{DocId, Document, Term};
use std::sync::Arc;

/// Index-wide statistics for a term, summed over every leaf, so a term
/// scores identically regardless of which segment a doc landed in (ES
/// semantics; Lucene scores per segment by default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermStats {
    pub doc_freq: u64,
    pub total_term_freq: u64,
    /// Lucene's field `docCount` across the index: docs with at least one
    /// token in the term's field (every doc for fields without norms),
    /// deleted docs included, as Lucene counts.
    pub num_docs: u64,
}

/// Index-wide per-field statistics (sum of field lengths over all docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldStats {
    pub num_docs: u64,
    pub sum_field_len: u64,
}

impl FieldStats {
    pub fn avg_field_len(&self) -> f32 {
        if self.num_docs == 0 {
            1.0
        } else {
            (self.sum_field_len as f64 / self.num_docs as f64).max(1.0) as f32
        }
    }
}

/// A read-only view of one leaf (segment or memtable).
#[derive(Clone, Copy)]
pub enum LeafReader<'a> {
    Segment(&'a Segment),
    /// A memtable, and the deletions searches see on it: `None` reads the
    /// memtable's own deletions (writer-side leaves); `Some` is the frozen
    /// snapshot a [`crate::reader_view::ReaderView`] took at publication,
    /// which later deletes and publishes never change.
    Memtable(&'a SegmentWriter, Option<&'a Arc<crate::postings::BitSet>>),
}

/// A leaf's shape doc values of one field ([`LeafReader::shape_values`]).
#[derive(Clone, Copy)]
pub enum ShapeValues<'a> {
    None,
    Column(&'a crate::fieldtypes::shape_dv::ShapeDvColumn),
    Memtable(&'a crate::fieldtypes::shape_field::ShapeBuf),
}

impl<'a> ShapeValues<'a> {
    #[inline]
    pub fn get(&self, doc: DocId) -> Option<crate::fieldtypes::shape_dv::ShapeValue<'a>> {
        match *self {
            ShapeValues::None => None,
            ShapeValues::Column(c) => c.get(doc),
            ShapeValues::Memtable(b) => b.value(doc),
        }
    }
}

impl<'a> LeafReader<'a> {
    /// The shape doc value of `geo_shape` field `field` for `doc`
    /// ([`crate::fieldtypes::shape_dv`]), if any.
    pub fn shape_value(&self, field: u16, doc: DocId) -> Option<crate::fieldtypes::shape_dv::ShapeValue<'a>> {
        match *self {
            LeafReader::Segment(s) => s.shape_data(field)?.values.as_ref()?.get(doc),
            LeafReader::Memtable(w, _) => w.shape_buf(field)?.value(doc),
        }
    }

    /// The shape doc values of `geo_shape` field `field`, resolved once for
    /// a pass over many docs.
    pub fn shape_values(&self, field: u16) -> ShapeValues<'a> {
        match *self {
            LeafReader::Segment(s) => s.shape_data(field).and_then(|d| d.values.as_ref()).map_or(ShapeValues::None, ShapeValues::Column),
            LeafReader::Memtable(w, _) => w.shape_buf(field).map_or(ShapeValues::None, ShapeValues::Memtable),
        }
    }

    /// Identity of the underlying segment / memtable (stable for the
    /// lifetime of the borrow; used to key per-leaf query state).
    pub fn id(&self) -> usize {
        match self {
            LeafReader::Segment(s) => *s as *const Segment as usize,
            LeafReader::Memtable(w, ..) => *w as *const SegmentWriter as usize,
        }
    }

    pub fn num_docs(&self) -> u32 {
        match self {
            LeafReader::Segment(s) => s.num_docs,
            LeafReader::Memtable(w, ..) => w.num_docs(),
        }
    }

    pub fn num_live_docs(&self) -> u32 {
        match self {
            LeafReader::Segment(s) => s.num_live_docs(),
            LeafReader::Memtable(w, Some(d)) => w.num_docs() - d.count().min(w.num_docs()),
            LeafReader::Memtable(w, None) => w.num_live_docs(),
        }
    }

    pub fn is_deleted(&self, doc: DocId) -> bool {
        match self {
            LeafReader::Segment(s) => s.is_deleted(doc),
            LeafReader::Memtable(_, Some(d)) => doc < d.num_docs() && d.get(doc),
            LeafReader::Memtable(w, None) => w.is_deleted(doc),
        }
    }

    /// Whether `doc` is deleted, deletes not yet visible to searches
    /// included (realtime reads; see [`crate::Index::set_nrt`]).
    pub fn is_deleted_latest(&self, doc: DocId) -> bool {
        match self {
            LeafReader::Segment(s) => s.is_deleted_latest(doc),
            // A view's memtable is not a realtime leaf: its frozen deletions.
            LeafReader::Memtable(_, Some(d)) => doc < d.num_docs() && d.get(doc),
            LeafReader::Memtable(w, None) => w.is_deleted_latest(doc),
        }
    }

    /// Live docs including unpublished deletes (realtime).
    pub fn num_live_docs_latest(&self) -> u32 {
        match self {
            LeafReader::Segment(s) => s.num_live_docs_latest(),
            LeafReader::Memtable(w, Some(d)) => w.num_docs() - d.count().min(w.num_docs()),
            LeafReader::Memtable(w, None) => w.num_docs() - w.latest_deletions().count(),
        }
    }

    /// [`LeafReader::undeleted_docs`] over every delete applied so far,
    /// unpublished ones included (realtime lookups).
    pub fn latest_undeleted_docs(&self) -> crate::error::Result<impl Fn(DocId) -> bool + Send + Sync + 'static> {
        let bits: Option<Arc<crate::postings::BitSet>> = match self {
            LeafReader::Segment(s) => Some(s.latest_deletions()?).filter(|b| !b.is_empty()),
            LeafReader::Memtable(_, Some(d)) => Some(Arc::clone(d)).filter(|b| !b.is_empty()),
            LeafReader::Memtable(w, None) => Some(Arc::new(w.latest_deletions())).filter(|b| !b.is_empty()),
        };
        Ok(move |d| bits.as_ref().is_none_or(|b| !b.get(d)))
    }

    /// The docs top-level collection must skip (bit set = skip): deleted
    /// docs plus nested child docs (ES hides nested docs from every query
    /// that does not join them through [`Query::Nested`](crate::query::Query::Nested)).
    /// `None` when every doc is a live root doc.
    pub fn deleted_bits(&self) -> Option<Arc<crate::postings::BitSet>> {
        let (dels, children) = match self {
            LeafReader::Segment(s) => (s.deletions_if_any(), s.child_docs().cloned()),
            LeafReader::Memtable(w, frozen) => {
                let dels = match frozen {
                    Some(d) => Some(Arc::clone(d)).filter(|b| !b.is_empty()),
                    None => (w.num_live_docs() != w.num_docs()).then(|| Arc::new(w.deletions())),
                };
                (dels, w.child_docs().map(Arc::new))
            }
        };
        match (dels, children) {
            (None, c) => c,
            (d, None) => d,
            (Some(d), Some(c)) => {
                let mut u = (*d).clone();
                u.or(&c);
                Some(Arc::new(u))
            }
        }
    }

    /// Only the deleted docs (unlike [`LeafReader::deleted_bits`], nested
    /// child docs are not included).
    pub fn deletions_only(&self) -> Option<Arc<crate::postings::BitSet>> {
        match self {
            LeafReader::Segment(s) => s.deletions_if_any(),
            LeafReader::Memtable(_, Some(d)) => Some(Arc::clone(d)).filter(|b| !b.is_empty()),
            LeafReader::Memtable(w, None) => (w.num_live_docs() != w.num_docs()).then(|| Arc::new(w.deletions())),
        }
    }

    /// Nested path ordinals of this leaf (`None` when it has no nested docs).
    pub fn nested_info(&self) -> Option<&'a crate::nested::NestedInfo> {
        match self {
            LeafReader::Segment(s) => s.nested_info(),
            LeafReader::Memtable(w, ..) => w.nested_info(),
        }
    }

    /// Whether `doc` is a top-level (root) document.
    pub fn is_root(&self, doc: DocId) -> bool {
        self.nested_info().is_none_or(|n| n.is_root(doc))
    }

    /// The doc one nesting level up from `doc` (`None` for root docs).
    pub fn nested_parent(&self, doc: DocId) -> Option<DocId> {
        self.nested_info()?.parent(doc)
    }

    /// The enclosing doc of nested `path` (`""` = the root document) that
    /// `doc` belongs to (ES `reverse_nested`); `None` if `doc` is not
    /// inside such a doc.
    pub fn nested_ancestor(&self, doc: DocId, path: &str) -> Option<DocId> {
        let Some(n) = self.nested_info() else { return path.is_empty().then_some(doc) };
        let depth = if path.is_empty() { 0 } else { n.ord_depth(n.path_ord(path)?) };
        let a = n.ancestor_at(doc, depth)?;
        (path.is_empty() || n.path(a) == path).then_some(a)
    }

    /// The root document `doc` belongs to (itself for root docs).
    pub fn nested_root(&self, doc: DocId) -> DocId {
        self.nested_info().map(|n| n.root(doc)).unwrap_or(doc)
    }

    /// Live nested docs of `path` inside `parent`'s block, ascending.
    pub fn nested_children(&self, parent: DocId, path: &str) -> Vec<DocId> {
        let Some(n) = self.nested_info() else { return Vec::new() };
        let Some(ord) = n.path_ord(path) else { return Vec::new() };
        let dels = self.deletions_only();
        n.children(parent, ord).into_iter().filter(|&d| !dels.as_ref().is_some_and(|b| b.get(d))).collect()
    }

    /// Live nested docs of `path` inside `parent`'s block, ascending,
    /// without allocating a `Vec`.
    pub fn nested_children_iter(&self, parent: DocId, path: &str) -> Box<dyn Iterator<Item = DocId> + '_> {
        let Some(n) = self.nested_info() else { return Box::new(std::iter::empty()) };
        let Some(ord) = n.path_ord(path) else { return Box::new(std::iter::empty()) };
        Box::new(n.children_iter(parent, ord).filter(move |&d| !self.is_deleted(d)))
    }

    /// ES `_nested.offset` of a nested doc (0 for root docs).
    pub fn nested_offset(&self, doc: DocId) -> u32 {
        self.nested_info().map(|n| n.offset(doc)).unwrap_or(0)
    }

    /// The nested path of `doc` (`""` for root docs).
    pub fn nested_path(&self, doc: DocId) -> &'a str {
        match self.nested_info() {
            Some(n) => n.path(doc),
            None => "",
        }
    }

    /// Decoded `(lat, lon)` points of a geo-point field of `doc`.
    pub fn geo_values(&self, field: u16, doc: DocId) -> Vec<(f64, f64)> {
        match self {
            LeafReader::Segment(s) => s.geo_index(field).map(|g| g.values().get(doc)).unwrap_or_default(),
            LeafReader::Memtable(w, ..) => w.geo_values(field).map(|g| g.get(doc)).unwrap_or_default(),
        }
    }

    /// Count point values without decoding coordinates or allocating a point list.
    pub fn geo_value_count(&self, field: u16, doc: DocId) -> usize {
        match self {
            LeafReader::Segment(s) => s.geo_index(field).map(|g| g.values().value_count(doc)).unwrap_or(0),
            LeafReader::Memtable(w, ..) => w.geo_values(field).map(|g| g.value_count(doc)).unwrap_or(0),
        }
    }

    /// Whether this leaf has any point in geo-point field `field`.
    pub fn has_geo(&self, field: u16) -> bool {
        match self {
            LeafReader::Segment(s) => s.geo_index(field).is_some(),
            LeafReader::Memtable(w, ..) => w.geo_values(field).is_some(),
        }
    }

    /// Docs with a point of `field` inside `shape`, ascending (KD-tree
    /// leaves outside the shape are skipped on sealed segments).
    pub fn geo_docs(&self, field: u16, shape: &crate::geo::GeoShape) -> Vec<DocId> {
        match self {
            LeafReader::Segment(s) => s.geo_index(field).map(|g| g.query(shape)).unwrap_or_default(),
            LeafReader::Memtable(w, ..) => w.geo_query(field, shape),
        }
    }

    /// [`LeafReader::geo_docs`] as [`crate::geo::DocHits`] (a bitset when
    /// dense: no sort of the KD tree's leaf-ordered hits).
    pub fn geo_hits(&self, field: u16, shape: &crate::geo::GeoShape) -> crate::geo::DocHits {
        match self {
            LeafReader::Segment(s) => match s.geo_index(field) {
                Some(g) => g.query_hits(shape, s.num_docs),
                None => crate::geo::DocHits::Sorted(Vec::new()),
            },
            LeafReader::Memtable(w, ..) => crate::geo::DocHits::Sorted(w.geo_query(field, shape)),
        }
    }

    /// A snapshot of the live-docs predicate, cheap to call per hit.
    pub fn live_docs(&self) -> impl Fn(DocId) -> bool + Send + Sync + 'static {
        let bits = self.deleted_bits();
        move |d| bits.as_ref().is_none_or(|b| !b.get(d))
    }

    /// Whether a doc is not deleted (nested child docs included, unlike
    /// [`LeafReader::live_docs`]).
    pub fn undeleted_docs(&self) -> impl Fn(DocId) -> bool + Send + Sync + 'static {
        let bits = self.deletions_only();
        move |d| bits.as_ref().is_none_or(|b| !b.get(d))
    }

    pub fn doc_freq(&self, term: &Term) -> u32 {
        match self {
            LeafReader::Segment(s) if s.is_synthetic_id(term.field) => s.term_info(term).map_or(0, |l| l.doc_freq),
            LeafReader::Segment(s) => s.term_dict().doc_freq(term),
            LeafReader::Memtable(w, ..) => w.doc_freq(term),
        }
    }

    /// Frequencies for ascending term texts; absent terms do not invoke the callback.
    pub fn doc_freqs_sorted(&self, field: u16, texts: &[&[u8]], mut f: impl FnMut(usize, u32)) {
        match self {
            // A synthetic id field has no dictionary: each id resolves.
            LeafReader::Segment(s) if s.is_synthetic_id(field) => {
                for (i, text) in texts.iter().enumerate() {
                    if let Some(loc) = s.term_info(&Term::new(field, text.to_vec())) { f(i, loc.doc_freq); }
                }
            }
            LeafReader::Segment(s) => s.term_dict().lookup_sorted(field, texts, |i, loc| f(i, loc.doc_freq)),
            LeafReader::Memtable(w, ..) => {
                for (i, text) in texts.iter().enumerate() {
                    let count = w.doc_freq(&Term::new(field, text.to_vec()));
                    if count != 0 { f(i, count); }
                }
            }
        }
    }

    pub fn total_term_freq(&self, term: &Term) -> u64 {
        match self {
            LeafReader::Segment(s) => s.term_dict().total_term_freq(term),
            LeafReader::Memtable(w, ..) => w.total_term_freq(term),
        }
    }

    pub fn postings(&self, term: &Term) -> Option<PostingsReader> {
        match self {
            LeafReader::Segment(s) => s.postings(term),
            LeafReader::Memtable(w, ..) => w.postings(term),
        }
    }

    /// Postings of each of the ascending `texts` of `field` present in this
    /// leaf. On sealed segments this is one galloping pass over the term
    /// dictionary rather than a binary search per term.
    pub fn for_each_postings_sorted(&self, field: u16, texts: &[&[u8]], mut f: impl FnMut(PostingsReader)) {
        match self {
            LeafReader::Segment(s) if s.is_synthetic_id(field) => {
                for t in texts {
                    if let Some(p) = s.postings(&Term::new(field, t.to_vec())) {
                        f(p);
                    }
                }
            }
            LeafReader::Segment(s) => s.term_dict().lookup_sorted(field, texts, |_, loc| {
                if let Some(p) = s.postings_at(&loc) {
                    f(p);
                }
            }),
            LeafReader::Memtable(w, ..) => {
                for t in texts {
                    if let Some(p) = w.postings(&Term::new(field, t.to_vec())) {
                        f(p);
                    }
                }
            }
        }
    }

    /// One dictionary lookup: the term's location and statistics in this
    /// leaf (`offset` is meaningless on the memtable).
    pub fn term_info(&self, term: &Term) -> Option<PostingsLocation> {
        match self {
            LeafReader::Segment(s) => s.term_info(term),
            LeafReader::Memtable(w, ..) => {
                let df = w.doc_freq(term);
                (df > 0).then(|| PostingsLocation { offset: 0, doc_freq: df, total_term_freq: w.total_term_freq(term) })
            }
        }
    }

    /// Postings of `term` given its [`LeafReader::term_info`] (no second
    /// dictionary lookup on sealed segments).
    #[inline]
    pub fn postings_with(&self, term: &Term, info: &PostingsLocation) -> Option<PostingsReader> {
        match self {
            LeafReader::Segment(s) => s.postings_at(info),
            LeafReader::Memtable(w, ..) => w.postings(term),
        }
    }

    /// Terms of `field` in `[lower, upper)` with their stats.
    pub fn term_range(
        &self,
        field: u16,
        lower: Option<&[u8]>,
        upper: Option<&[u8]>,
    ) -> Vec<(Vec<u8>, PostingsLocation)> {
        match self {
            LeafReader::Segment(s) => s.term_dict().range_iter(field, lower, upper),
            LeafReader::Memtable(w, ..) => w.term_range(field, lower, upper),
        }
    }

    /// [`LeafReader::norms`] shared: sealed segments hand out a cached `Arc`.
    pub fn norms_arc(&self, field: u16) -> Arc<Norms> {
        match self {
            LeafReader::Segment(s) => s.norms_arc(field),
            LeafReader::Memtable(w, ..) => Arc::new(w.norms_for(field)),
        }
    }

    /// Norms for a text field (unit norms if none were indexed).
    pub fn norms(&self, field: u16) -> Norms {
        match self {
            LeafReader::Segment(s) => s.norms_shared(field),
            LeafReader::Memtable(w, ..) => w.norms_for(field),
        }
    }

    /// Lucene's collection statistics of `field` on this leaf: `num_docs` is
    /// `docCount` (docs with at least one term in the field) and
    /// `sum_field_len` is `sumTotalTermFreq` (every token, position-
    /// overlapping ones included; one per distinct value for keywords).
    /// BM25 and the other similarities derive `N` and `avgdl` from these.
    /// Deleted docs stay counted until a merge, as in Lucene.
    pub fn field_stats(&self, field: u16) -> FieldStats {
        if let Some(a) = self.field_agg(field) {
            return FieldStats { num_docs: a.doc_count, sum_field_len: a.sum_ttf };
        }
        match self {
            LeafReader::Segment(s) => match s.norms(field) {
                Some(n) => FieldStats { num_docs: n.bytes().iter().filter(|&&b| b != 0).count() as u64, sum_field_len: n.total() },
                None => FieldStats::default(),
            },
            LeafReader::Memtable(w, ..) => match w.fast_fields().norms().get(&field) {
                Some(n) => FieldStats { num_docs: n.docs_with_field(), sum_field_len: n.total() },
                None => FieldStats::default(),
            },
        }
    }

    fn field_agg(&self, field: u16) -> Option<crate::segment::FieldAgg> {
        match self {
            LeafReader::Segment(s) => s.field_agg(field),
            LeafReader::Memtable(w, ..) => w.field_agg(field),
        }
    }

    /// The number of terms of `field` on this leaf and the sum of their doc
    /// freqs (Lucene `Terms.size()` / `getSumDocFreq()`; cached per sealed
    /// segment).
    pub fn field_term_stats(&self, field: u16) -> (u64, u64) {
        match self {
            LeafReader::Segment(s) => s.field_term_stats(field),
            LeafReader::Memtable(..) => {
                let prefix = field.to_le_bytes();
                self.term_range(field, None, None).iter().filter(|(t, _)| t.len() >= 2 && t[..2] == prefix).fold((0, 0), |(n, sum), (_, loc)| (n + 1, sum + loc.doc_freq as u64))
            }
        }
    }

    /// Lucene's `docCount` of `field` on this leaf (docs with at least one
    /// term in it); every doc for fields without collection statistics.
    pub fn field_doc_count(&self, field: u16) -> u64 {
        match self.field_agg(field) {
            Some(a) => a.doc_count,
            None => self.num_docs() as u64,
        }
    }

    pub fn get_doc(&self, doc: DocId) -> crate::error::Result<Option<Document>> {
        match self {
            LeafReader::Segment(s) => s.get_doc(doc),
            LeafReader::Memtable(w, ..) => w.get_doc(doc),
        }
    }

    /// Stored document decoding with a fallible before-allocation callback.
    pub fn get_doc_guarded<E: From<crate::RusticError>>(&self,doc:DocId,before:impl FnMut(u64)->std::result::Result<(),E>)->std::result::Result<Option<Document>,E> {
        match self {
            LeafReader::Segment(s)=>s.get_doc_guarded(doc,before),
            LeafReader::Memtable(w,..)=>w.get_doc_guarded(doc,before),
        }
    }

    /// [`Segment::get_docs_batched`] (a memtable reads each document).
    pub fn get_docs_batched(&self, docs: &[DocId], restore: bool, mut f: impl FnMut(usize, Document)) -> crate::error::Result<()> {
        match self {
            LeafReader::Segment(s) => s.get_docs_batched(docs, restore, f),
            LeafReader::Memtable(w, ..) => {
                for (i, &d) in docs.iter().enumerate() {
                    if let Some(doc) = w.get_doc(d)? { f(i, doc); }
                }
                Ok(())
            }
        }
    }

    /// The stored documents of `docs` (ascending), each store chunk
    /// decompressed once: `f(doc, document)` in order.
    pub fn for_each_doc(&self, docs: &[DocId], f: impl FnMut(DocId, Document)) -> crate::error::Result<()> {
        self.try_for_each_doc(docs, f)
    }

    /// Streaming stored-document reads with storage errors propagated to the caller.
    pub fn try_for_each_doc(&self, docs: &[DocId], mut f: impl FnMut(DocId, Document)) -> crate::error::Result<()> {
        match self {
            LeafReader::Segment(s) => { for &d in docs { if let Some(doc) = s.get_doc(d)? { f(d, doc); } } Ok(()) },
            LeafReader::Memtable(w, ..) => {
                for &d in docs {
                    if let Some(doc) = w.get_doc(d)? { f(d, doc); }
                }
                Ok(())
            }
        }
    }

    /// The stored values of `field` in `doc`, in the order they were added
    /// (the per-document stored-fields block: every value of every field the
    /// writer was handed is persisted in the segment store, so this works
    /// for fields that are neither indexed nor have doc values).
    pub fn stored_values(&self, field: u16, doc: DocId) -> crate::error::Result<crate::types::StoredValues> {
        match self {
            LeafReader::Segment(s) => s.stored_values(field, doc),
            LeafReader::Memtable(w, ..) => w.stored_values(field, doc),
        }
    }

    /// The stored vector of `doc` in vector field `field`, if any.
    pub fn vector(&self, field: u16, doc: DocId) -> Option<Vec<f32>> {
        match self {
            LeafReader::Segment(s) => s.vector(field, doc),
            LeafReader::Memtable(w, ..) => w.vector(field, doc).map(|v| v.to_vec()),
        }
    }

    /// Evaluate a row without allocating a vector copy for every document.
    pub fn with_vector<T>(&self, field: u16, doc: DocId, f: impl FnOnce(&[f32]) -> T) -> Option<T> {
        match self {
            LeafReader::Segment(s) => s.with_vector(field, doc, f),
            LeafReader::Memtable(w, ..) => w.vector(field, doc).map(f),
        }
    }

    /// Scan selected rows without per-document allocation or store locking.
    /// The visitor must not re-enter vector storage or schedule writer jobs.
    pub fn visit_vectors(&self, field: u16, docs: impl IntoIterator<Item = DocId>, mut visit: impl FnMut(DocId, &[f32]) -> bool) -> bool {
        match self {
            LeafReader::Segment(s) => s.visit_vectors(field, docs, visit),
            LeafReader::Memtable(w, ..) => {
                for doc in docs {
                    let Some(vector) = w.vector(field, doc) else { return false };
                    if !visit(doc, vector) { return false; }
                }
                true
            }
        }
    }

    /// Scan selected rows without per-document allocation or store locking.
    /// The visitor must not re-enter vector storage or schedule writer jobs.
    pub fn visit_vectors_with_norm(&self, field: u16, docs: impl IntoIterator<Item = DocId>, mut visit: impl FnMut(DocId, &[f32], f32) -> bool) -> bool {
        match self {
            LeafReader::Segment(s) => s.visit_vectors_with_norm(field, docs, visit),
            LeafReader::Memtable(w, ..) => {
                for doc in docs {
                    let Some(vector) = w.vector(field, doc) else { return false };
                    if !visit(doc, vector, crate::vector::norm(vector)) { return false; }
                }
                true
            }
        }
    }

    /// k-NN over a vector field. Sealed segments route through their
    /// [`VectorStore`](crate::vector::store::VectorStore): the accepted docs
    /// are mapped to point ids and the planner picks exact or filtered HNSW
    /// per vector segment. The memtable is searched exactly.
    pub fn knn(&self, field: u16, query: &[f32], k: usize, ef: usize, accept: impl Fn(DocId) -> bool) -> crate::error::Result<Vec<(DocId, f32)>> {
        self.knn_with(field, query, k, ef, accept, &crate::vector::store::PlannerConfig::default())
    }

    /// [`LeafReader::knn`] with an explicit planner configuration
    /// (quantization / rescoring; the memtable ignores it, it is always
    /// exact `f32`).
    pub fn knn_with(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        accept: impl Fn(DocId) -> bool,
        cfg: &crate::vector::store::PlannerConfig,
    ) -> crate::error::Result<Vec<(DocId, f32)>> {
        match self {
            LeafReader::Segment(s) => s.knn_with(field, query, k, ef, accept, cfg),
            LeafReader::Memtable(w, ..) => Ok(w.knn_with(field, query, k, accept, cfg)),
        }
    }

    /// Fast-path k-NN when there are no deletions and no filter: skips the
    /// O(n) point-map scan. For sealed segments this passes `None` to the
    /// vector store so HNSW traversal has no per-node filter check. The
    /// memtable is always exact and small, so it uses the same path as
    /// `knn_with` with an all-accept predicate.
    pub fn knn_unfiltered(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        cfg: &crate::vector::store::PlannerConfig,
    ) -> crate::error::Result<Vec<(DocId, f32)>> {
        match self {
            LeafReader::Segment(s) => s.knn_unfiltered(field, query, k, ef, cfg),
            LeafReader::Memtable(w, ..) => Ok(w.knn_with(field, query, k, |_| true, cfg)),
        }
    }

    /// Whether every doc of this sealed leaf (deleted ones included) has
    /// exactly one value of numeric field `field`: a single-valued i64,
    /// u64 or f64 column with a value for every doc. False for memtables,
    /// multi-valued or missing columns. Read-only.
    pub fn numeric_one_value_per_doc(&self, field: u16) -> bool {
        let LeafReader::Segment(s) = self else { return false };
        let ff = s.fast_fields();
        let n = s.num_docs as u64;
        if let Some(c) = ff.get_i64(field) {
            return c.num_docs() as u64 == n && c.value_doc_count() == n;
        }
        if let Some(c) = ff.get_u64(field) {
            return c.num_docs() as u64 == n && c.value_doc_count() == n;
        }
        if let Some(c) = ff.get_f64(field) {
            return c.num_docs() as u64 == n && c.value_doc_count() == n;
        }
        false
    }

    /// Numeric i64 fast-field range `[lower, upper]` as a doc bitset (one
    /// bit per doc, `num_docs.div_ceil(64)` words). Single-valued sealed
    /// columns use the points index; nothing is materialised per match.
    /// `None` when the field has no numeric column on this leaf.
    pub fn range_i64_bits(&self, field: u16, lower: i64, upper: i64) -> Option<Vec<u64>> {
        let mut words = vec![0u64; (self.num_docs() as usize).div_ceil(64)];
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    c.range_bits(lower, upper, &mut words);
                } else {
                    ff.get_multi_i64(field)?.range_bits(lower, upper, &mut words);
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    scan_bits(c.values(), w.num_docs(), &mut words, |d| c.has_value(d), |v| *v >= lower && *v <= upper);
                } else {
                    let c = ff.get_multi_i64(field)?;
                    scan_multi_bits(c, w.num_docs(), &mut words, |v| *v >= lower && *v <= upper);
                }
            }
        }
        Some(words)
    }

    /// OR the docs with an i64 value in `[lower, upper]` into `words` (bit
    /// `d` = doc `d`, covering the leaf's docs), keeping the other bits.
    pub fn or_range_i64_bits(&self, field: u16, lower: i64, upper: i64, words: &mut [u64]) {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    c.or_range_bits(lower, upper, words);
                } else if let Some(c) = ff.get_multi_i64(field) {
                    c.for_each_doc_in_range(lower, upper, |d| words[d as usize >> 6] |= 1u64 << (d & 63));
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    scan_bits(c.values(), w.num_docs(), words, |d| c.has_value(d), |v| *v >= lower && *v <= upper);
                } else if let Some(c) = ff.get_multi_i64(field) {
                    scan_multi_bits(c, w.num_docs(), words, |v| *v >= lower && *v <= upper);
                }
            }
        }
    }

    /// [`LeafReader::range_i64_bits`] for f64 columns.
    pub fn range_f64_bits(&self, field: u16, lower: f64, upper: f64) -> Option<Vec<u64>> {
        let mut words = vec![0u64; (self.num_docs() as usize).div_ceil(64)];
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    c.range_bits(lower, upper, &mut words);
                } else {
                    ff.get_multi_f64(field)?.range_bits(lower, upper, &mut words);
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    scan_bits(c.values(), w.num_docs(), &mut words, |d| c.has_value(d), |v| *v >= lower && *v <= upper);
                } else {
                    let c = ff.get_multi_f64(field)?;
                    scan_multi_bits(c, w.num_docs(), &mut words, |v| *v >= lower && *v <= upper);
                }
            }
        }
        Some(words)
    }

    /// [`LeafReader::range_i64_bits`] for u64 columns.
    pub fn range_u64_bits(&self, field: u16, lower: u64, upper: u64) -> Option<Vec<u64>> {
        let mut words = vec![0u64; (self.num_docs() as usize).div_ceil(64)];
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    c.range_bits(lower, upper, &mut words);
                } else {
                    ff.get_multi_u64(field)?.range_bits(lower, upper, &mut words);
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    scan_bits(c.values(), w.num_docs(), &mut words, |d| c.has_value(d), |v| *v >= lower && *v <= upper);
                } else {
                    let c = ff.get_multi_u64(field)?;
                    scan_multi_bits(c, w.num_docs(), &mut words, |v| *v >= lower && *v <= upper);
                }
            }
        }
        Some(words)
    }

    /// Numeric i64 fast-field range `[lower, upper]`, ascending doc ids.
    /// A doc matches if ANY of its values is in range (multi-valued fields).
    pub fn range_i64(&self, field: u16, lower: i64, upper: i64) -> Vec<DocId> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    c.range_query(lower, upper)
                } else {
                    ff.get_multi_i64(field)
                        .map(|c| c.range_query(lower, upper))
                        .unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    scan_range(c.values(), w.num_docs(), |v| *v >= lower && *v <= upper).into_iter().filter(|&d| c.has_value(d)).collect()
                } else {
                    ff.get_multi_i64(field)
                        .map(|c| scan_multi(c, w.num_docs(), |v| *v >= lower && *v <= upper))
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Numeric f64 fast-field range `[lower, upper]`, ascending doc ids.
    /// A doc matches if ANY of its values is in range (multi-valued fields).
    pub fn range_f64(&self, field: u16, lower: f64, upper: f64) -> Vec<DocId> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    c.range_query(lower, upper)
                } else {
                    ff.get_multi_f64(field)
                        .map(|c| c.range_query(lower, upper))
                        .unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    scan_range(c.values(), w.num_docs(), |v| *v >= lower && *v <= upper).into_iter().filter(|&d| c.has_value(d)).collect()
                } else {
                    ff.get_multi_f64(field)
                        .map(|c| scan_multi(c, w.num_docs(), |v| *v >= lower && *v <= upper))
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Numeric u64 fast-field range `[lower, upper]`, ascending doc ids.
    /// A doc matches if ANY of its values is in range (multi-valued fields).
    pub fn range_u64(&self, field: u16, lower: u64, upper: u64) -> Vec<DocId> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    c.range_query(lower, upper)
                } else {
                    ff.get_multi_u64(field)
                        .map(|c| c.range_query(lower, upper))
                        .unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    scan_range(c.values(), w.num_docs(), |v| *v >= lower && *v <= upper).into_iter().filter(|&d| c.has_value(d)).collect()
                } else {
                    ff.get_multi_u64(field)
                        .map(|c| scan_multi(c, w.num_docs(), |v| *v >= lower && *v <= upper))
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Maximum single-valued i64 across physical documents, including deleted
    /// documents. Recovery uses this to avoid reusing an assigned sequence number.
    /// Open the segment column once, rather than revalidating its sparse rank index
    /// for every document lookup.
    pub fn max_i64_value(&self, field: u16) -> Option<i64> {
        match self {
            LeafReader::Segment(s) => s.fast_fields().get_i64(field)?.max(),
            LeafReader::Memtable(w, ..) => {
                let column = w.fast_fields().get_i64(field)?;
                (0..w.num_docs()).filter_map(|doc| column.get(doc)).max()
            }
        }
    }

    /// The smallest and largest i64 fast-field value of `field` over every
    /// physical document, deleted ones included (as Lucene's point values
    /// are): the bounds the can-match phase tests a range against. `None`:
    /// no value at all.
    pub fn min_max_i64(&self, field: u16) -> Option<(i64, i64)> {
        let fold = |acc: Option<(i64, i64)>, v: i64| Some(acc.map_or((v, v), |(lo, hi)| (lo.min(v), hi.max(v))));
        match self {
            LeafReader::Segment(s) => s.min_max_i64(field),
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    return (0..w.num_docs()).filter_map(|d| c.get(d)).fold(None, fold);
                }
                let c = ff.get_multi_i64(field)?;
                (0..w.num_docs()).flat_map(|d| c.doc_values(d).to_vec()).fold(None, fold)
            }
        }
    }

    /// i64 fast-field value of a doc.
    pub fn i64_value(&self, field: u16, doc: DocId) -> Option<i64> {
        match self {
            LeafReader::Segment(s) => s.fast_fields().get_i64(field)?.get(doc),
            LeafReader::Memtable(w, ..) => w.fast_fields().get_i64(field)?.get(doc),
        }
    }

    /// All i64 fast-field values of a doc (0 or 1 for single-valued fields).
    /// A boolean column reads as 0/1 so booleans can be sorted like numbers.
    pub fn i64_values(&self, field: u16, doc: DocId) -> Vec<i64> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    c.get(doc).into_iter().collect()
                } else if let Some(c) = ff.get_multi_i64(field) {
                    c.values_vec(doc)
                } else {
                    // A doc without a value has none (a column that knows).
                    ff.get_bool(field).filter(|c| c.has_value(doc)).map(|c| vec![c.get(doc) as i64]).unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    c.get(doc).into_iter().collect()
                } else if let Some(c) = ff.get_multi_i64(field) {
                    c.values(doc).to_vec()
                } else {
                    // A doc without a value has none (a column that knows).
                    ff.get_bool(field).filter(|c| c.has_value(doc)).map(|c| vec![c.get(doc) as i64]).unwrap_or_default()
                }
            }
        }
    }

    /// f64 fast-field value of a doc.
    pub fn f64_value(&self, field: u16, doc: DocId) -> Option<f64> {
        match self {
            LeafReader::Segment(s) => s.fast_fields().get_f64(field)?.get(doc),
            LeafReader::Memtable(w, ..) => w.fast_fields().get_f64(field)?.get(doc),
        }
    }

    /// All f64 fast-field values of a doc (0 or 1 for single-valued fields).
    pub fn f64_values(&self, field: u16, doc: DocId) -> Vec<f64> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    c.get(doc).into_iter().collect()
                } else {
                    ff.get_multi_f64(field).map(|c| c.values_vec(doc)).unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    c.get(doc).into_iter().collect()
                } else {
                    ff.get_multi_f64(field).map(|c| c.values(doc).to_vec()).unwrap_or_default()
                }
            }
        }
    }

    /// All u64 fast-field values of a doc (0 or 1 for single-valued fields).
    pub fn u64_values(&self, field: u16, doc: DocId) -> Vec<u64> {
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    c.get(doc).into_iter().collect()
                } else {
                    ff.get_multi_u64(field).map(|c| c.values_vec(doc)).unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_u64(field) {
                    c.get(doc).into_iter().collect()
                } else {
                    ff.get_multi_u64(field).map(|c| c.values(doc).to_vec()).unwrap_or_default()
                }
            }
        }
    }

    /// Keyword fast-field value of a doc.
    pub fn keyword_value(&self, field: u16, doc: DocId) -> Option<String> {
        match self {
            // A field without a column (`_id` from format v15): one doc's
            // value from its stored fields, as ES reads `_id` (a derived
            // column is only built for whole-segment uses).
            // `_id`s in an id column ([`crate::codec::uid_column`]).
            LeafReader::Segment(s) if s.uid_column_field() == Some(field) => s.column_uid(doc),
            LeafReader::Segment(s) if s.keyword_from_postings(field) => s.get_doc(doc).ok().flatten()
                .and_then(|d| d.get(field).and_then(|v| v.as_text().map(str::to_string))),
            LeafReader::Segment(s) => s.keyword_reader(field)?.get(doc).map(|v| v.into_owned()),
            LeafReader::Memtable(w, ..) => w.fast_fields().get_keyword(field)?.get(doc).map(str::to_string),
        }
    }

    /// Lazy reader metadata that must be reserved before obtaining columns.
    pub fn field_reader_allocation_bound(&self,field:u16)->u64 {
        match self {LeafReader::Segment(s)=>s.keyword_metadata_allocation_bound(field),LeafReader::Memtable(..)=>0}
    }

    /// Conservative allocation bound for numeric/keyword values of one doc.
    /// Reads lengths and borrowed ordinals before owned value vectors/strings
    /// are constructed. Includes runtime wrapper/JSON conversion expansion.
    pub fn field_values_allocation_bound(&self,field:u16,doc:DocId)->u64 {
        let numeric=match self {
            LeafReader::Segment(s)=>{
                let ff=s.fast_fields();
                let i=ff.get_multi_i64(field).map(|c|c.count(doc) as u64).unwrap_or(0);
                let f=ff.get_multi_f64(field).map(|c|c.count(doc) as u64).unwrap_or(0);
                let u=ff.get_multi_u64(field).map(|c|c.count(doc) as u64).unwrap_or(0);
                i.max(f).max(u).max(1)
            },
            LeafReader::Memtable(w,..)=>{
                let ff=w.fast_fields();
                let i=ff.get_multi_i64(field).map(|c|c.values(doc).len() as u64).unwrap_or(0);
                let f=ff.get_multi_f64(field).map(|c|c.values(doc).len() as u64).unwrap_or(0);
                let u=ff.get_multi_u64(field).map(|c|c.values(doc).len() as u64).unwrap_or(0);
                i.max(f).max(u).max(1)
            },
        };
        let keyword=match self {
            // A column-less `_id` (an id is at most 512 bytes): no column
            // is derived just to bound one doc's value.
            LeafReader::Segment(s) if s.keyword_from_postings(field)=>512+32,
            LeafReader::Segment(s)=>{
                if let Some(c)=s.keyword_reader(field) {c.get(doc).map(|v|(v.len() as u64).saturating_add(32)).unwrap_or(0)}
                else {s.multi_keyword_reader(field).map(|c|c.ords(doc).filter_map(|o|c.term(o)).fold(0u64,|n,v|n.saturating_add(v.len() as u64).saturating_add(32))).unwrap_or(0)}
            },
            LeafReader::Memtable(w,..)=>{
                let ff=w.fast_fields();
                if let Some(c)=ff.get_keyword(field) {c.get(doc).map(|v|(v.len() as u64).saturating_add(32)).unwrap_or(0)}
                else {ff.get_multi_keyword(field).map(|c|c.value_bytes(doc)).unwrap_or(0)}
            },
        };
        numeric.saturating_mul(256).saturating_add(keyword.saturating_mul(256)).saturating_add(256)
    }

    /// [`Self::field_values_allocation_bound`] summed over `docs`, with each
    /// column reader opened once for the batch instead of once per doc (a
    /// column lookup scans the segment's column directory, and a multi-valued
    /// keyword reader takes the segment's table lock).
    pub fn field_values_allocation_bounds(&self,field:u16,docs:&[DocId])->u64 {
        let LeafReader::Segment(s)=self else {
            return docs.iter().fold(0u64,|sum,&doc|sum.saturating_add(self.field_values_allocation_bound(field,doc)));
        };
        if docs.is_empty() {return 0;}
        let ff=s.fast_fields();
        let (mi,mf,mu)=(ff.get_multi_i64(field),ff.get_multi_f64(field),ff.get_multi_u64(field));
        let postings=s.keyword_from_postings(field);
        let single=if postings {None} else {s.keyword_reader(field)};
        let multi=if postings || single.is_some() {None} else {s.multi_keyword_reader(field)};
        // A single-valued term's length by ordinal: from the segment's
        // decoded terms, else decoded once per ordinal here (a front-coded
        // term decodes by replaying its block; per doc it dominated SQL row
        // reads over a keyword column).
        let decoded=match &single {Some(_)=>s.keyword_terms_decoded(field).ok().flatten(),None=>None};
        let mut lens:std::collections::HashMap<u32,u64>=std::collections::HashMap::new();
        let mut single_len=|c:&crate::codec::fastfields::KeywordReader<'_>,doc:DocId|->u64 {
            let ord=c.ord(doc);
            if ord==crate::codec::fastfields::MISSING_ORD {return 0;}
            if let Some(t)=decoded.as_ref().and_then(|t|t.get(ord as usize)) {return (t.len() as u64).saturating_add(32);}
            *lens.entry(ord).or_insert_with(||c.term(ord).map(|v|(v.len() as u64).saturating_add(32)).unwrap_or(0))
        };
        docs.iter().fold(0u64,|sum,&doc|{
            let i=mi.as_ref().map(|c|c.count(doc) as u64).unwrap_or(0);
            let f=mf.as_ref().map(|c|c.count(doc) as u64).unwrap_or(0);
            let u=mu.as_ref().map(|c|c.count(doc) as u64).unwrap_or(0);
            let numeric=i.max(f).max(u).max(1);
            let keyword=if postings {512+32}
                else if let Some(c)=&single {single_len(c,doc)}
                else {multi.as_ref().map(|c|c.ords(doc).filter_map(|o|c.term(o)).fold(0u64,|n,v|n.saturating_add(v.len() as u64).saturating_add(32))).unwrap_or(0)};
            sum.saturating_add(numeric.saturating_mul(256).saturating_add(keyword.saturating_mul(256)).saturating_add(256))
        })
    }

    /// [`Self::keyword_values`] of every doc of `docs` (same values, same
    /// order), the column reader opened once for the batch.
    pub fn keyword_values_many(&self, field: u16, docs: &[DocId]) -> Vec<Vec<String>> {
        match self {
            LeafReader::Segment(s) if !s.keyword_from_postings(field) => {
                if let Some(c) = s.keyword_reader(field) {
                    docs.iter().map(|&doc| c.get(doc).map(|v| v.into_owned()).into_iter().collect()).collect()
                } else if let Some(c) = s.multi_keyword_reader(field) {
                    docs.iter().map(|&doc| c.values(doc).into_iter().map(str::to_string).collect()).collect()
                } else if let Some(c) = s.fast_fields().get_features(field) {
                    docs.iter().map(|&doc| c.values(doc)).collect()
                } else {
                    vec![Vec::new(); docs.len()]
                }
            }
            _ => docs.iter().map(|&doc| self.keyword_values(field, doc)).collect(),
        }
    }

    /// All keyword fast-field values of a doc (0 or 1 for single-valued
    /// fields; deduped, ascending for multi-valued).
    /// Whether `doc` has a doc value in `field`'s column, read without
    /// decoding the value (a numeric column's presence, a keyword's
    /// ordinal); a memtable reads its values.
    /// [`crate::segment::Segment::column_field_ids`] of a sealed segment;
    /// `None` for a memtable.
    pub fn column_field_ids(&self) -> Option<Vec<u16>> {
        match self {
            LeafReader::Segment(s) => Some(s.column_field_ids()),
            _ => None,
        }
    }

    pub fn has_doc_value(&self, field: u16, doc: DocId) -> bool {
        if let LeafReader::Segment(s) = self {
            if !s.keyword_from_postings(field) {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) { return c.has_value(doc); }
                if let Some(c) = ff.get_u64(field) { return c.has_value(doc); }
                if let Some(c) = ff.get_f64(field) { return c.has_value(doc); }
                if let Some(c) = s.keyword_reader(field) { return c.ord(doc) != crate::codec::fastfields::MISSING_ORD; }
                if let Some(c) = s.multi_keyword_reader(field) { return c.count(doc) > 0; }
                if let Some(c) = ff.get_multi_i64(field) { return c.count(doc) > 0; }
                if let Some(c) = ff.get_multi_u64(field) { return c.count(doc) > 0; }
                if let Some(c) = ff.get_multi_f64(field) { return c.count(doc) > 0; }
                if let Some(c) = ff.get_bool(field) { return c.has_value(doc); }
            }
        }
        !self.keyword_values(field, doc).is_empty() || !self.i64_values(field, doc).is_empty()
            || !self.u64_values(field, doc).is_empty() || !self.f64_values(field, doc).is_empty()
    }

    pub fn keyword_values(&self, field: u16, doc: DocId) -> Vec<String> {
        match self {
            LeafReader::Segment(s) if s.keyword_from_postings(field) => self.keyword_value(field, doc).into_iter().collect(),
            LeafReader::Segment(s) => {
                if let Some(c) = s.keyword_reader(field) {
                    c.get(doc).map(|v| v.into_owned()).into_iter().collect()
                } else if let Some(c) = s.multi_keyword_reader(field) {
                    c.values(doc).into_iter().map(str::to_string).collect()
                } else {
                    s.fast_fields().get_features(field).map(|c| c.values(doc)).unwrap_or_default()
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_keyword(field) {
                    c.get(doc).map(str::to_string).into_iter().collect()
                } else {
                    ff.get_multi_keyword(field)
                        .map(|c| c.values(doc).into_iter().map(str::to_string).collect())
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Uninverted terms of text field `field` (ES `fielddata: true`),
    /// cached per leaf: built on first use, then shared.
    pub fn uninverted(&self, field: u16) -> crate::error::Result<Arc<crate::fielddata::Uninverted>> {
        match self {
            LeafReader::Segment(s) => s.uninverted(field),
            LeafReader::Memtable(w, ..) => w.uninverted(field),
        }
    }

    /// Bool fast-field value of a doc.
    pub fn bool_value(&self, field: u16, doc: DocId) -> Option<bool> {
        match self {
            LeafReader::Segment(s) => s.fast_fields().get_bool(field).filter(|c| c.has_value(doc)).map(|c| c.get(doc)),
            LeafReader::Memtable(w, ..) => w.fast_fields().get_bool(field).filter(|c| c.has_value(doc)).map(|c| c.get(doc)),
        }
    }

    /// Whether `doc` has a value of boolean `field`, when the leaf knows
    /// (`None`: a column before [`crate::codec::BOOL_PRESENCE_SINCE`],
    /// whose docs without a value read `false`); `Some(false)` without a
    /// column.
    pub fn bool_presence(&self, field: u16, doc: DocId) -> Option<bool> {
        match self {
            LeafReader::Segment(s) => match s.fast_fields().get_bool(field) {
                Some(c) => c.presence_known().then(|| c.has_value(doc)),
                None => Some(false),
            },
            LeafReader::Memtable(w, ..) => match w.fast_fields().get_bool(field) {
                Some(c) => c.presence_known().then(|| c.has_value(doc)),
                None => Some(false),
            },
        }
    }

    /// Field statistics to score with: the index-wide `given` ones, or this
    /// leaf's own when none were gathered.
    fn effective_field_stats(&self, field: u16, given: FieldStats) -> FieldStats {
        if given.num_docs > 0 {
            given
        } else {
            let own = self.field_stats(field);
            if own.num_docs > 0 {
                own
            } else {
                FieldStats { num_docs: 0, sum_field_len: 0 }
            }
        }
    }

    /// BM25 term scorer using index-wide `stats` (pass this leaf's own
    /// numbers for Lucene's per-segment behaviour). Thin wrapper over
    /// [`LeafReader::term_scorer_sim`] kept for `Bm25Params` callers.
    pub fn term_scorer(&self, term: &Term, params: Bm25Params, stats: TermStats, field_stats: FieldStats) -> Option<Box<dyn Scorer>> {
        self.term_scorer_sim(term, &Bm25::from_params(params), 1.0, stats, field_stats)
    }

    /// Term scorer scoring through `sim`, with `boost` folded into the
    /// similarity weight. BM25 uses a statically dispatched weight.
    pub fn term_scorer_sim(
        &self,
        term: &Term,
        sim: &dyn Similarity,
        boost: f32,
        stats: TermStats,
        field_stats: FieldStats,
    ) -> Option<Box<dyn Scorer>> {
        let postings = self.postings(term)?;
        self.term_scorer_postings(postings, term.field, sim, boost, stats, field_stats)
    }

    /// [`LeafReader::term_scorer_sim`] over already fetched `postings` of a
    /// term of `field`.
    #[inline]
    pub fn term_scorer_postings(
        &self,
        postings: PostingsReader,
        field: u16,
        sim: &dyn Similarity,
        boost: f32,
        stats: TermStats,
        field_stats: FieldStats,
    ) -> Option<Box<dyn Scorer>> {
        let norms = self.norms_arc(field);
        if let Some(bm25) = sim.as_bm25() {
            let avg = if field_stats.num_docs > 0 {
                field_stats.avg_field_len()
            } else {
                norms.avg()
            };
            let weight = Bm25Weight::new(stats.doc_freq, stats.num_docs, bm25.params(), avg).with_boost(boost);
            return Some(Box::new(Bm25TermScorer::new(postings, weight, norms)));
        }
        let fs = self.effective_field_stats(field, field_stats);
        let weight = sim.weight(stats, fs, boost);
        Some(Box::new(TermScorer::with_norms(PostingsDocSet::new(postings), weight, norms)))
    }

    /// Explanation of [`LeafReader::term_scorer_sim`]'s score for `doc`
    /// (`None` if the term does not occur in it).
    pub fn explain_term(
        &self,
        term: &Term,
        sim: &dyn Similarity,
        boost: f32,
        stats: TermStats,
        field_stats: FieldStats,
        doc: DocId,
    ) -> Option<Explanation> {
        let (tf, len) = self.term_freq_at(term, doc)?;
        let fs = if sim.as_bm25().is_some() && field_stats.num_docs == 0 {
            // Mirror the BM25 fast path, which falls back to this leaf's norms.
            let norms = self.norms(term.field);
            FieldStats { num_docs: 1_000_000, sum_field_len: (norms.avg() as f64 * 1_000_000.0) as u64 }
        } else {
            self.effective_field_stats(term.field, field_stats)
        };
        let e = sim.weight(stats, fs, boost).explain(tf, len);
        Some(Explanation::new(e.value, format!("weight({term} in {doc}) [{}]", sim.name()), vec![e]))
    }

    /// `(term frequency, field length)` of `term` in `doc`, if it occurs.
    pub fn term_freq_at(&self, term: &Term, doc: DocId) -> Option<(u32, u32)> {
        let mut p = self.postings(term)?;
        if p.advance(doc)? != doc {
            return None;
        }
        let tf = p.term_freq().unwrap_or(1);
        Some((tf, self.norms(term.field).len(doc)))
    }

    /// Phrase scorer over consecutive `terms` with `slop`, scored with the
    /// summed index-wide IDF of the terms. Thin wrapper over
    /// [`LeafReader::phrase_scorer_sim`] kept for `Bm25Params` callers.
    pub fn phrase_scorer(
        &self,
        terms: &[Term],
        slop: u32,
        params: Bm25Params,
        term_stats: &[TermStats],
        field_stats: FieldStats,
    ) -> Option<Box<dyn Scorer>> {
        self.phrase_scorer_sim(terms, slop, &Bm25::from_params(params), term_stats, field_stats)
    }

    /// Phrase scorer scoring the phrase frequency through `sim`
    /// ([`Similarity::phrase_weight`] over the terms' statistics).
    pub fn phrase_scorer_sim(
        &self,
        terms: &[Term],
        slop: u32,
        sim: &dyn Similarity,
        term_stats: &[TermStats],
        field_stats: FieldStats,
    ) -> Option<Box<dyn Scorer>> {
        let positions: Vec<Vec<Term>> = terms.iter().map(|t| vec![t.clone()]).collect();
        let s = self.multi_phrase_raw(&positions, slop, sim, term_stats, field_stats)?;
        Some(Box::new(s))
    }

    /// Multi-phrase scorer (Lucene `MultiPhraseQuery`): `positions[i]` lists
    /// the alternative terms accepted at query position `i`. `term_stats`
    /// are the statistics of every term of every position, in order; they
    /// are summed into the weight as Lucene does.
    pub fn multi_phrase_scorer(
        &self,
        positions: &[Vec<Term>],
        slop: u32,
        sim: &dyn Similarity,
        term_stats: &[TermStats],
        field_stats: FieldStats,
    ) -> Option<Box<dyn Scorer>> {
        Some(Box::new(self.multi_phrase_raw(positions, slop, sim, term_stats, field_stats)?))
    }

    /// The concrete [`PhraseScorer`] behind [`LeafReader::multi_phrase_scorer`]
    /// (exposes the phrase frequency, for explanations).
    pub fn multi_phrase_raw(
        &self,
        positions: &[Vec<Term>],
        slop: u32,
        sim: &dyn Similarity,
        term_stats: &[TermStats],
        field_stats: FieldStats,
    ) -> Option<PhraseScorer> {
        self.multi_phrase_raw_with(positions, slop, sim, term_stats, field_stats, &|t| self.postings(t))
    }

    /// [`LeafReader::multi_phrase_raw`] fetching each term's postings
    /// through `postings` (e.g. from locations cached by the stats pass).
    pub fn multi_phrase_raw_with(
        &self,
        positions: &[Vec<Term>],
        slop: u32,
        sim: &dyn Similarity,
        term_stats: &[TermStats],
        field_stats: FieldStats,
        postings: &dyn Fn(&Term) -> Option<PostingsReader>,
    ) -> Option<PhraseScorer> {
        self.multi_phrase_raw_boosted(positions, slop, sim, term_stats, field_stats, 1.0, postings)
    }

    /// [`LeafReader::multi_phrase_raw_with`] with a query `boost` folded into
    /// the phrase weight.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_phrase_raw_boosted(
        &self,
        positions: &[Vec<Term>],
        slop: u32,
        sim: &dyn Similarity,
        term_stats: &[TermStats],
        field_stats: FieldStats,
        boost: f32,
        postings: &dyn Fn(&Term) -> Option<PostingsReader>,
    ) -> Option<PhraseScorer> {
        let field = positions.first()?.first()?.field;
        let mut groups = Vec::with_capacity(positions.len());
        for (i, alts) in positions.iter().enumerate() {
            let readers: Vec<PostingsReader> = alts.iter().filter_map(postings).collect();
            if readers.is_empty() {
                return None;
            }
            groups.push((readers, i as u32));
        }
        let norms = self.norms_arc(field);
        let mut fs = self.effective_field_stats(field, field_stats);
        if fs.num_docs == 0 {
            // No norms anywhere: score as if every doc had the leaf's average.
            fs = FieldStats { num_docs: self.num_docs().max(1) as u64, sum_field_len: (norms.avg() as f64 * self.num_docs().max(1) as f64) as u64 };
        }
        let weight = sim.phrase_weight(term_stats, fs, boost);
        PhraseScorer::new_multi(groups, slop, weight, norms)
    }

    /// Doc ids that have a value in `field`, read from fast-field value
    /// presence (keyword / multi-valued columns on both leaf kinds, plus
    /// single numeric columns on the memtable). `None` when this leaf
    /// cannot tell presence from fast fields (callers fall back to postings).
    /// The docs with a value of single-valued numeric segment column
    /// `field`, without listing them: `Some(None)` when every doc has one,
    /// `Some(Some(words))` the column's presence bitset (bit `d` = doc `d`).
    /// `None` for other columns and memtables ([`Self::value_presence`]
    /// answers those); the docs are those `value_presence` lists.
    pub fn numeric_presence(&self, field: u16) -> Option<Option<Vec<u64>>> {
        let LeafReader::Segment(s) = self else { return None };
        let ff = s.fast_fields();
        // `value_presence` reads a keyword column first.
        if ff.get_keyword(field).is_some() {
            return None;
        }
        let n = s.num_docs;
        // A dense column covering fewer docs than the segment lists its own.
        let dense = |docs: u32| (docs != n).then(|| {
            let mut words = vec![u64::MAX; (docs as usize).div_ceil(64)];
            if docs % 64 != 0 {
                if let Some(last) = words.last_mut() {
                    *last = (1u64 << (docs % 64)) - 1;
                }
            }
            words
        });
        if let Some(c) = ff.get_i64(field) {
            return Some(c.presence_words().or_else(|| dense(c.num_docs())));
        }
        if let Some(c) = ff.get_f64(field) {
            return Some(c.presence_words().or_else(|| dense(c.num_docs())));
        }
        if let Some(c) = ff.get_u64(field) {
            return Some(c.presence_words().or_else(|| dense(c.num_docs())));
        }
        // A boolean column that knows its docs (v37: `exists` on a
        // doc-values-only boolean, a time series index's flags).
        if let Some(words) = ff.get_bool(field).and_then(|c| c.presence_words()) {
            return Some(Some(words));
        }
        None
    }

    /// The docs with a value in `field` as presence words (bit `d` = doc
    /// `d`), without listing them: a single numeric column's presence
    /// bitmap, a keyword column's (sparse columns store it; dense ones are
    /// read an ordinal at a time, no term decoded), else from
    /// [`Self::value_presence`]. `None` when fast fields cannot tell.
    pub fn presence_words(&self, field: u16) -> Option<Vec<u64>> {
        let n = self.num_docs() as usize;
        if let Some(present) = self.numeric_presence(field) {
            return Some(present.unwrap_or_else(|| {
                let mut words = vec![u64::MAX; n.div_ceil(64)];
                if n % 64 != 0 { if let Some(last) = words.last_mut() { *last = (1u64 << (n % 64)) - 1; } }
                words
            }));
        }
        if let LeafReader::Segment(s) = self {
            if let Some(c) = s.fast_fields().get_keyword(field) {
                return Some(c.presence_words());
            }
        }
        let docs = self.value_presence(field)?;
        let mut words = vec![0u64; n.div_ceil(64)];
        for d in docs { if let Some(w) = words.get_mut(d as usize / 64) { *w |= 1u64 << (d % 64); } }
        Some(words)
    }

    pub fn value_presence(&self, field: u16) -> Option<Vec<DocId>> {
        let n = self.num_docs();
        match self {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_keyword(field) {
                    let words = c.presence_words();
                    return Some((0..n).filter(|&d| words[d as usize / 64] >> (d % 64) & 1 == 1).collect());
                }
                if let Some(c) = ff.get_i64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = ff.get_f64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = ff.get_u64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = s.multi_keyword_reader(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_i64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_f64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_u64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(g) = s.geo_index(field) {
                    return Some(geo_docs(g.values()));
                }
                if let Some(c) = ff.get_bool(field).filter(|c| c.presence_known()) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                None
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_keyword(field) {
                    return Some((0..n).filter(|&d| c.get(d).is_some()).collect());
                }
                if let Some(c) = ff.get_i64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = ff.get_f64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = ff.get_u64(field) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                if let Some(c) = ff.get_multi_keyword(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_i64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_f64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(c) = ff.get_multi_u64(field) {
                    return Some((0..n).filter(|&d| c.count(d) > 0).collect());
                }
                if let Some(g) = w.geo_values(field) {
                    return Some(geo_docs(g));
                }
                if let Some(c) = ff.get_bool(field).filter(|c| c.presence_known()) {
                    return Some((0..n).filter(|&d| c.has_value(d)).collect());
                }
                None
            }
        }
    }

    /// All numeric fast-field values of a doc as `f64` (f64, then i64, then
    /// u64 columns); empty if the doc has none.
    pub fn numeric_values(&self, field: u16, doc: DocId) -> Vec<f64> {
        let v = self.f64_values(field, doc);
        if !v.is_empty() {
            return v;
        }
        let v = self.i64_values(field, doc);
        if !v.is_empty() {
            return v.into_iter().map(|x| x as f64).collect();
        }
        self.u64_values(field, doc).into_iter().map(|x| x as f64).collect()
    }
}

/// Linear scan over an in-memory column (memtable path, no block index).
fn scan_range<T>(values: &[T], num_docs: u32, pred: impl Fn(&T) -> bool) -> Vec<DocId> {
    values
        .iter()
        .take(num_docs as usize)
        .enumerate()
        .filter(|(_, v)| pred(v))
        .map(|(i, _)| i as DocId)
        .collect()
}

/// Docs `< num_docs` whose value passes `pred`; `has`: whether the doc has a
/// value at all (gaps in a sparse column hold the padding `0`).
fn scan_bits<T>(values: &[T], num_docs: u32, words: &mut [u64], has: impl Fn(DocId) -> bool, pred: impl Fn(&T) -> bool) {
    for (i, v) in values.iter().take(num_docs as usize).enumerate() {
        if pred(v) && has(i as DocId) {
            words[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

fn scan_multi_bits<T>(col: &impl MultiValues<T>, num_docs: u32, words: &mut [u64], pred: impl Fn(&T) -> bool) {
    for d in 0..num_docs {
        if col.doc_values(d).iter().any(&pred) {
            words[d as usize >> 6] |= 1u64 << (d & 63);
        }
    }
}

/// Uniform per-doc value access over the in-memory multi-valued columns.
trait MultiValues<T> {
    fn doc_values(&self, doc: DocId) -> &[T];
}

impl MultiValues<i64> for crate::fast_fields::MultiI64Column {
    fn doc_values(&self, doc: DocId) -> &[i64] {
        self.values(doc)
    }
}

impl MultiValues<f64> for crate::fast_fields::MultiF64Column {
    fn doc_values(&self, doc: DocId) -> &[f64] {
        self.values(doc)
    }
}

impl MultiValues<u64> for crate::fast_fields::MultiU64Column {
    fn doc_values(&self, doc: DocId) -> &[u64] {
        self.values(doc)
    }
}

/// Linear scan over an in-memory multi-valued column: a doc matches if ANY
/// of its values satisfies `pred` (memtable path, no block index).
fn scan_multi<T>(col: &impl MultiValues<T>, num_docs: u32, pred: impl Fn(&T) -> bool) -> Vec<DocId> {
    (0..num_docs)
        .filter(|&d| col.doc_values(d).iter().any(&pred))
        .collect()
}

/// Lucene's field `docCount` over `leaves`: a leaf with no collection
/// statistics for `field` (none of its docs has a term in it, e.g. a
/// segment written before the field was mapped) adds nothing, as a Lucene
/// segment without the field does. Only when no leaf has any (a field kind
/// without statistics) does every doc count ([`LeafReader::field_doc_count`]).
/// Counting such a leaf's docs made `N` depend on whether its segments were
/// merged away yet (deleted older versions of the documents).
fn field_doc_count(leaves: &[LeafReader<'_>], field: u16) -> u64 {
    let (mut stats, mut n) = (false, 0u64);
    for l in leaves {
        if let Some(a) = l.field_agg(field) {
            stats = true;
            n += a.doc_count;
        }
    }
    if stats { n } else { leaves.iter().map(|l| l.num_docs() as u64).sum() }
}

/// Sum term statistics over a set of leaves.
pub fn term_stats(leaves: &[LeafReader<'_>], term: &Term) -> TermStats {
    let mut s = TermStats::default();
    for l in leaves {
        if let Some(info) = l.term_info(term) {
            s.doc_freq += info.doc_freq as u64;
            s.total_term_freq += info.total_term_freq;
        }
    }
    s.num_docs = field_doc_count(leaves, term.field);
    s
}

/// [`term_stats`] plus each leaf's [`LeafReader::term_info`] (in leaf
/// order), so scorers can skip their own dictionary lookups.
pub fn term_stats_with_infos(leaves: &[LeafReader<'_>], term: &Term) -> (TermStats, Vec<Option<PostingsLocation>>) {
    let mut s = TermStats::default();
    let mut infos = Vec::with_capacity(leaves.len());
    for l in leaves {
        let info = l.term_info(term);
        if let Some(info) = &info {
            s.doc_freq += info.doc_freq as u64;
            s.total_term_freq += info.total_term_freq;
        }
        infos.push(info);
    }
    s.num_docs = field_doc_count(leaves, term.field);
    (s, infos)
}

/// Sum field statistics over a set of leaves.
pub fn field_stats(leaves: &[LeafReader<'_>], field: u16) -> FieldStats {
    let mut s = FieldStats::default();
    for l in leaves {
        let f = l.field_stats(field);
        s.num_docs += f.num_docs;
        s.sum_field_len += f.sum_field_len;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::RamDirectory;
    use crate::query::collector::collect_top_k;
    use crate::types::{Field, FieldType};

    fn schema() -> Vec<Field> {
        vec![
            Field::new(0, "body", FieldType::Text),
            Field::new(1, "n", FieldType::I64),
            Field::new(2, "tag", FieldType::Keyword),
        ]
    }

    #[test]
    fn batched_keyword_values_and_allocation_bounds_match_per_doc_reads() {
        let dir = RamDirectory::new();
        let fields = vec![
            Field::new(0, "tag", FieldType::Keyword),
            Field::new(1, "tags", FieldType::Keyword),
            Field::new(2, "x", FieldType::F64),
            Field::new(3, "xs", FieldType::F64),
        ];
        let mut writer = SegmentWriter::new(0);
        for doc in 0..500u32 {
            let mut d = Document::new();
            if doc % 5 != 0 { d.add_keyword(0, &format!("t{}", doc % 37)); }
            for k in 0..doc % 3 { d.add_keyword(1, &format!("m{}", (doc + k) % 11)); }
            if doc % 4 == 0 { d.add_f64(2, doc as f64 / 8.0); }
            for k in 0..doc % 3 { d.add_f64(3, (doc * k) as f64); }
            writer.add_document(&d, &fields);
        }
        let docs: Vec<DocId> = (0..500).filter(|d| d % 3 != 1).collect();
        let check = |leaf: LeafReader<'_>| {
            for field in 0..5u16 {
                let per_doc: Vec<Vec<String>> = docs.iter().map(|&d| leaf.keyword_values(field, d)).collect();
                assert_eq!(leaf.keyword_values_many(field, &docs), per_doc, "field {field}");
                let bound: u64 = docs.iter().map(|&d| leaf.field_values_allocation_bound(field, d)).sum();
                assert_eq!(leaf.field_values_allocation_bounds(field, &docs), bound, "field {field}");
                assert_eq!(leaf.field_values_allocation_bounds(field, &[]), 0);
            }
        };
        check(LeafReader::Memtable(&writer, None));
        let segment = writer.write_to(&dir).unwrap();
        check(LeafReader::Segment(&segment));
    }

    #[test]
    fn physical_i64_max_preserves_sparse_negative_and_deleted_values() {
        let dir = RamDirectory::new();
        let fields = schema();
        for sparse in [false, true] {
            let mut writer = SegmentWriter::new(0);
            for doc in 0..300u32 {
                let mut value = Document::new();
                if !sparse || doc % 7 == 0 {
                    value.add_i64(1, -1 - doc as i64);
                }
                writer.add_document(&value, &fields);
            }
            // Recovery must still see the maximum assigned value after deletion.
            writer.delete_doc(0);
            let check = |leaf: LeafReader<'_>| {
                assert_eq!(leaf.max_i64_value(1), Some(-1));
                assert_eq!(leaf.max_i64_value(99), None);
                assert_eq!(leaf.max_i64_value(1),
                    (0..leaf.num_docs()).filter_map(|doc| leaf.i64_value(1, doc)).max());
            };
            check(LeafReader::Memtable(&writer, None));
            let segment = writer.write_to(&dir).unwrap();
            check(LeafReader::Segment(&segment));
        }
    }

    #[test]
    fn memtable_and_segment_leaves_agree() {
        let dir = RamDirectory::new();
        let schema = schema();
        let mut sealed = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        for i in 0..200u32 {
            let mut d = Document::new();
            d.add_text(0, &format!("rust search {} engine", if i % 3 == 0 { "fast" } else { "slow" }));
            d.add_i64(1, i as i64);
            d.add_keyword(2, if i % 2 == 0 { "even" } else { "odd" });
            if i < 100 {
                sealed.add_document(&d, &schema);
            } else {
                mem.add_document(&d, &schema);
            }
        }
        let seg = sealed.write_to(&dir).unwrap();
        mem.delete_doc(5); // doc 105 overall
        let leaves = [LeafReader::Segment(&seg), LeafReader::Memtable(&mem, None)];

        let fast = Term::new_text(0, "fast");
        let ts = term_stats(&leaves, &fast);
        assert_eq!(ts.num_docs, 200);
        assert_eq!(ts.doc_freq, 67);
        let fs = field_stats(&leaves, 0);
        assert_eq!(fs.num_docs, 200);
        assert!((fs.avg_field_len() - 4.0).abs() < 1e-6);

        // Same index-wide weight in both leaves → identical scores for identical docs.
        let mut scores = Vec::new();
        for l in &leaves {
            let mut s = l.term_scorer(&fast, Bm25Params::default(), ts, fs).unwrap();
            let live = l.live_docs();
            let (hits, total) = collect_top_k(&mut *s, 100, live);
            scores.push((hits[0].score, total));
        }
        assert!((scores[0].0 - scores[1].0).abs() < 1e-6);
        // 67 docs contain "fast"; doc 105 (i % 3 == 0) is deleted in the memtable.
        assert_eq!(scores[0].1 + scores[1].1, 66);

        // Ranges and values across both leaf kinds.
        assert_eq!(leaves[0].range_i64(1, 10, 20).len(), 11);
        assert_eq!(leaves[1].range_i64(1, 150, 160).len(), 11);
        assert_eq!(leaves[1].i64_value(1, 0), Some(100));
        assert_eq!(leaves[1].keyword_value(2, 1), Some("odd".into()));
        assert_eq!(leaves[0].term_range(2, None, None).len(), 2);
        assert_eq!(leaves[1].term_range(2, None, None).len(), 2);
        assert!(leaves[1].is_deleted(5));
        assert!(!leaves[1].live_docs()(5));
        assert!(leaves[0].live_docs()(5));

        // Phrase on both leaves.
        let phrase = [Term::new_text(0, "rust"), Term::new_text(0, "search")];
        let stats: Vec<TermStats> = phrase.iter().map(|t| term_stats(&leaves, t)).collect();
        for l in &leaves {
            let mut p = l.phrase_scorer(&phrase, 0, Bm25Params::default(), &stats, fs).unwrap();
            let (_, total) = collect_top_k(&mut *p, 10, l.live_docs());
            assert!(total == 100 || total == 99);
        }

        // *_values on single-valued columns: 0 or 1 element.
        assert_eq!(leaves[0].i64_values(1, 7), vec![7]);
        assert_eq!(leaves[1].i64_values(1, 0), vec![100]);
        assert_eq!(leaves[0].keyword_values(2, 1), vec!["odd".to_string()]);
        assert_eq!(leaves[1].i64_values(99, 0), Vec::<i64>::new()); // unknown field
    }

    /// Values of a field that is neither indexed nor in the schema (a
    /// stored-only field) are persisted per doc, in order, on both leaf kinds.
    #[test]
    fn stored_values_roundtrip_on_segment_and_memtable() {
        let dir = RamDirectory::new();
        let mut schema = schema();
        let mut blob = Field::new(7, "_stored", FieldType::Keyword);
        blob.indexed = false;
        blob.fast = false;
        schema.push(blob);
        let mut sealed = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        for i in 0..70u32 {
            let mut d = Document::new();
            d.add_text(0, "doc");
            d.add_keyword(7, &format!("{{\"n\":[{i}]}}"));
            d.add_keyword(7, "second");
            d.add_keyword(99, "not in schema");
            if i < 40 {
                sealed.add_document(&d, &schema);
            } else {
                mem.add_document(&d, &schema);
            }
        }
        let seg = sealed.write_to(&dir).unwrap();
        let leaves = [LeafReader::Segment(&seg), LeafReader::Memtable(&mem, None)];
        for (li, l) in leaves.iter().enumerate() {
            let base = li as u32 * 40;
            for d in 0..l.num_docs() {
                let got: Vec<String> = l.stored_values(7, d).unwrap().iter().map(|v| v.as_text().unwrap().to_string()).collect();
                assert_eq!(got, vec![format!("{{\"n\":[{}]}}", base + d), "second".to_string()], "leaf {li} doc {d}");
                assert_eq!(l.stored_values(99, d).unwrap().len(), 1);
                assert!(l.stored_values(5, d).unwrap().is_empty());
            }
            assert!(l.stored_values(7, 1000).unwrap().is_empty());
        }
        // The term dictionary has nothing for the unindexed stored field.
        assert!(leaves[0].term_range(7, None, None).is_empty());
    }

    #[test]
    fn multi_valued_values_on_segment_and_memtable() {
        let dir = RamDirectory::new();
        let schema = schema();
        let mut sealed = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        // Values per doc: i % 3 (0, 1 or 2), so both leaves get promoted.
        let vals = |i: u32| -> Vec<i64> { (0..(i % 3)).map(|k| (i * 10 + k) as i64).collect() };
        let tags = |i: u32| -> Vec<&'static str> {
            let all = ["x", "y", "z"];
            (0..(i % 3)).map(|k| all[((i + k) % 3) as usize]).collect()
        };
        for i in 0..60u32 {
            let mut d = Document::new();
            d.add_text(0, "doc");
            for v in vals(i) {
                d.add_i64(1, v);
            }
            for t in tags(i) {
                d.add_keyword(2, t);
            }
            if i < 30 {
                sealed.add_document(&d, &schema);
            } else {
                mem.add_document(&d, &schema);
            }
        }
        let seg = sealed.write_to(&dir).unwrap();
        let leaves = [LeafReader::Segment(&seg), LeafReader::Memtable(&mem, None)];
        for (li, l) in leaves.iter().enumerate() {
            let base = li as u32 * 30;
            for d in 0..30u32 {
                let i = base + d;
                assert_eq!(l.i64_values(1, d), vals(i), "leaf {li} doc {d}");
                let mut expected_tags: Vec<String> = tags(i).into_iter().map(str::to_string).collect();
                expected_tags.sort();
                expected_tags.dedup();
                assert_eq!(l.keyword_values(2, d), expected_tags, "leaf {li} doc {d}");
            }
            // range_i64 falls through to the multi column (any-value semantics).
            let expected: Vec<DocId> = (0..30u32)
                .filter(|&d| vals(base + d).iter().any(|v| *v >= (base as i64 + 10) * 10 && *v <= (base as i64 + 12) * 10 + 1))
                .collect();
            assert_eq!(l.range_i64(1, (base as i64 + 10) * 10, (base as i64 + 12) * 10 + 1), expected, "leaf {li}");
            assert!(!expected.is_empty());
            // Docs with no values yield empty lists; nothing panics past the end.
            assert_eq!(l.i64_values(1, 0), Vec::<i64>::new());
            assert_eq!(l.i64_values(1, 1000), Vec::<i64>::new());
            assert_eq!(l.f64_values(1, 0), Vec::<f64>::new()); // wrong type → empty
        }
    }
}

/// The docs with a point in `values` (doc-sorted points), ascending, each once.
fn geo_docs(values: &crate::geo::GeoValues) -> Vec<DocId> {
    let mut out: Vec<DocId> = Vec::new();
    for p in values.iter() {
        let doc = p.doc;
        if out.last() != Some(&doc) {
            out.push(doc);
        }
    }
    out
}
