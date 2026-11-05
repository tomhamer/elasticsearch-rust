//! Segment: the basic unit of index storage.
//!
//! A `SegmentWriter` accumulates documents in memory, then `write_to`
//! encodes them into one immutable segment file (see [`crate::codec`]).
//! A `Segment` is a reader over such a file: the term dictionary index is
//! loaded on open; postings, fast fields and stored docs are read lazily
//! and zero-copy from the mapped bytes. Deletions are a separate bitmap
//! persisted to a sibling `.del` file.

use crate::codec::directory::{Directory, FileSlice};
use crate::codec::fastfields::{parse_column_directory, ColumnEntry, FastFieldsReader, KeywordTable, NormsReader, TAG_KEYWORD};
use crate::codec::postings::{write_postings_with_norms, BlockNorms, PostingsReader};
use crate::query::segment_scorer::Norms;
use crate::codec::store::{StoreReader, StoreTable};
use crate::codec::varint::write_vint;
use crate::codec::terms::{TermDictReader, TermDictWriter};
use crate::codec::vectors::{encode_point_map, similarity_by_name, PointMapReader, VectorFieldBuilder, NO_POINT};
use crate::vector::store::{PlannerConfig, VectorStore};
use crate::vector::Similarity;
use crate::codec::{decode_extra, deletes_file_name, encode_extra, open_segment_file, segment_file_name, write_segment_file_with_extra, SegmentFileFrame, EXTRA_COLUMN_STORED, EXTRA_FIELD_STATS, EXTRA_KEYWORD_FROM_POSTINGS, EXTRA_GEO, EXTRA_GEO2, EXTRA_NESTED, EXTRA_SHAPE_DV, EXTRA_SHAPE_LEGACY, EXTRA_SHAPE_TREE};
use crate::geo::{GeoIndex, GeoShape, GeoValues};
use crate::nested::NestedInfo;
use crate::error::Result;
use crate::fast_fields::FastFields;
use crate::postings::{BitSet, DecodedPostings, MemPostings};
use crate::term_dictionary::PostingsLocation;
use crate::types::{DocId, Document, Field, FieldType, Term};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// Graph admission completed before document mutation. Reservations roll back
/// their graph rows if this token is abandoned before writer publication.
#[cfg(feature = "mutable-hnsw-poc")]
pub struct PreparedDocumentVectors {
    store: Arc<VectorStore>,
    batches: Vec<(u16, usize, crate::vector::store::PointReservation)>,
    /// `(field, start, rows)` admitted while the field had no graph (`flat`,
    /// or `hnsw` before its graph is built): legitimately unreserved, so the
    /// seal stores them and inserts them into a graph built since.
    graphless: Vec<(u16, usize, usize)>,
}

/// A segment's live docs' points for one vector field at a deletion count
/// (see [`Segment::cached_live_points`]): dense words over point ids from
/// word `first_word`.
pub struct LivePoints {
    pub field: u16,
    pub deleted: u32,
    pub first_word: usize,
    pub words: Vec<u64>,
    _memory: crate::admission::Reservation,
}

/// Docs `doc..doc + len` of a segment whose vectors are points
/// `point..point + len` (see [`Segment::point_runs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointRun {
    pub doc: u32,
    pub point: u64,
    pub len: u32,
}

/// A sealed, immutable segment backed by its on-disk file.
///
/// The file-backed data and every cache built over it live in a shared
/// [`SegmentData`] (reached through `Deref`); the deletions and the vector
/// store binding are per `Segment`. A [`crate::reader_view::ReaderView`]
/// holds cheap copies of the searchable segments
/// ([`Segment::reader_snapshot`]) whose deletions are frozen as searches saw
/// them at publication, sharing data and caches with the live segment.
pub struct Segment {
    pub id: u32,
    pub num_docs: u32,
    /// Copy-on-write so readers snapshot with an `Arc` clone, not a copy.
    /// Searches see [`DelState::visible`]; realtime lookups the latest.
    deletions: RwLock<DelState>,
    /// The vector store this segment's points live in (set by
    /// [`VectorStore::attach`]).
    vector_store: OnceLock<Arc<VectorStore>>,
    /// Some doc is deleted (set under the deletions write lock, never
    /// cleared), so readers can skip scanning an all-zero bitmap.
    any_deleted: std::sync::atomic::AtomicBool,
    /// Bumped under the deletions' write lock at every change of them:
    /// [`Segment::num_live_docs`] caches its count per generation.
    del_gen: std::sync::atomic::AtomicU64,
    /// `(generation + 1, live docs)` of the last [`Segment::num_live_docs`]
    /// count (0: none).
    live_count: std::sync::atomic::AtomicU64,
    /// File-backed data and caches, shared with every reader snapshot of
    /// this segment. Its strong count is what keeps the segment files alive
    /// after a merge retires them (see `Index::commit`).
    data: Arc<SegmentData>,
}

/// One vector field's point range in a segment, with its lazily built
/// `PointId -> doc` inverse ([`Segment::doc_of_point`]).
struct PointBounds {
    field: u16,
    min: u64,
    max: u64,
    inverse: OnceLock<PointInverse>,
}

/// A segment field's point map inverted (sorted by point). Its bytes are
/// counted in [`point_inverse_bytes`] (the node stat
/// `refresh.rustic_point_inverse_size_in_bytes`) and charged to node-wide
/// accounting ([`crate::accounting::Category::VectorPoints`], the engine's
/// `accounting` breaker) while it exists.
struct PointInverse(Vec<(u64, DocId)>);

static POINT_INVERSE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes held by every segment's point inverse (see [`Segment::doc_of_point`]).
pub fn point_inverse_bytes() -> u64 {
    POINT_INVERSE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

impl PointInverse {
    fn new(pairs: Vec<(u64, DocId)>) -> Self {
        POINT_INVERSE_BYTES.fetch_add(Self::bytes(&pairs), std::sync::atomic::Ordering::Relaxed);
        crate::accounting::add(crate::accounting::Category::VectorPoints, Self::bytes(&pairs));
        PointInverse(pairs)
    }
    fn bytes(pairs: &[(u64, DocId)]) -> u64 {
        (pairs.len() * std::mem::size_of::<(u64, DocId)>()) as u64
    }
}

impl Drop for PointInverse {
    fn drop(&mut self) {
        POINT_INVERSE_BYTES.fetch_sub(Self::bytes(&self.0), std::sync::atomic::Ordering::Relaxed);
        crate::accounting::sub(crate::accounting::Category::VectorPoints, Self::bytes(&self.0));
    }
}

/// A decoded chunk's charge follows every query/cache alias, including
/// eviction while another reader still owns its bytes.
struct StoredChunk {
    data: Vec<u8>,
    /// Where each complete doc of `data` starts (built on the first read):
    /// a doc of a large chunk (zstd `best_compression`, ~600 docs) is then
    /// found without walking every length before it.
    offsets: std::sync::OnceLock<Vec<u32>>,
    charge: crate::accounting::Charge,
    _memory:Option<crate::admission::Reservation>,
}
impl StoredChunk {
    fn new(data:Vec<u8>,memory:Option<crate::admission::Reservation>)->Self {
        let bytes=(data.capacity() as u64).saturating_add(std::mem::size_of::<Self>() as u64);
        Self{data,offsets:std::sync::OnceLock::new(),charge:crate::accounting::Charge::new(crate::accounting::Category::ReadCaches,bytes),_memory:memory}
    }
    /// The framed bytes of doc `within` ([`StoreReader::doc_bytes_in_chunk`]).
    fn doc_bytes(&self,within:usize)->Result<&[u8]> {
        let offsets=self.offsets.get_or_init(|| crate::codec::store::chunk_doc_offsets(&self.data));
        match offsets.get(within) {
            Some(&at) => crate::codec::store::doc_bytes_at(&self.data,at as usize),
            None => StoreReader::doc_bytes_in_chunk(&self.data,within),
        }
    }
    fn doc_as(&self,within:usize,compact:bool)->Result<Document> {
        crate::codec::store::decode_doc(self.doc_bytes(within)?,compact)
    }
}
impl std::ops::Deref for StoredChunk {
    type Target=Vec<u8>;
    fn deref(&self)->&Self::Target{&self.data}
}

/// A prepared row can outlive its segment in live or retained point tables.
pub(crate) struct VectorPayloadRow {
    pub(crate) values: Vec<(u16, Arc<str>)>,
    _memory: Arc<crate::admission::Reservation>,
}

/// On-disk bytes of one field of a segment, by structure (see
/// [`Segment::disk_usage`]). Field `u16::MAX` collects section framing
/// that belongs to no field.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldDiskUsage {
    /// Term dictionary entries (with their block's first-key copy and
    /// index entry for a block's first term).
    pub terms: u64,
    pub term_count: u64,
    /// Postings lists (doc ids, frequencies, skip data, positions).
    pub postings: u64,
    /// Fast-field column bytes other than those below (values, ordinals,
    /// presence bitmaps, block min/max).
    pub doc_values: u64,
    /// A keyword column's term table.
    pub doc_values_terms: u64,
    /// A numeric column's points index.
    pub points: u64,
    pub norms: u64,
    /// The field's share of the compressed stored-field chunks.
    pub stored: u64,
    /// The field's serialized (uncompressed) stored bytes.
    pub stored_raw: u64,
    /// `[extra]` geo-point index.
    pub geo: u64,
    /// `[extra]` geo_shape triangle trees.
    pub shape: u64,
    /// Other `[extra]` entries (nested ordinals, field statistics).
    pub other_extra: u64,
    pub vectors: u64,
}

impl FieldDiskUsage {
    /// Every on-disk byte (`stored_raw` and `term_count` are not bytes on disk).
    pub fn total(&self) -> u64 {
        self.terms + self.postings + self.doc_values + self.doc_values_terms + self.points + self.norms + self.stored + self.geo + self.shape + self.other_extra + self.vectors
    }
    pub fn add(&mut self, o: &FieldDiskUsage) {
        self.terms += o.terms; self.term_count += o.term_count; self.postings += o.postings;
        self.doc_values += o.doc_values; self.doc_values_terms += o.doc_values_terms; self.points += o.points;
        self.norms += o.norms; self.stored += o.stored; self.stored_raw += o.stored_raw; self.geo += o.geo;
        self.shape += o.shape; self.other_extra += o.other_extra; self.vectors += o.vectors;
    }
}

/// [`Segment::disk_usage`]: section sizes and per-field bytes.
#[derive(Clone, Debug, Default)]
pub struct SegmentDiskUsage {
    pub terms: u64,
    pub postings: u64,
    pub fast: u64,
    pub store: u64,
    pub vectors: u64,
    pub extra: u64,
    pub fields: std::collections::BTreeMap<u16, FieldDiskUsage>,
}

impl SegmentDiskUsage {
    fn field(&mut self, id: u16) -> &mut FieldDiskUsage {
        self.fields.entry(id).or_default()
    }
}

/// Append the synthetic id field `field` ([`crate::synthetic_id`]) to a
/// stored `doc` lacking it, from its `(_tsid value, timestamp, routing
/// hash)` columns (`None`: a column lacks the doc; left without).
fn restore_synthetic_id(doc: &mut Document, field: u16, columns: impl FnOnce() -> Option<(Vec<u8>, i64, i64)>) {
    if doc.fields.iter().any(|(f, _)| *f == field) {
        return;
    }
    let Some(codec) = crate::synthetic_id::codec() else { return };
    if let Some(id) = columns().and_then(|(tsid, ts, rh)| (codec.encode)(&tsid, ts, rh as i32)) {
        doc.fields.insert(0, (field, crate::types::FieldValue::Keyword(id)));
    }
}

/// Append `fields` (column-stored, see [`Field::column_stored`]) that
/// `doc` lacks, with the values `value` gives.
fn restore_column_stored(doc: &mut Document, fields: &[u16], value: impl Fn(u16) -> Option<i64>) {
    for &f in fields {
        if !doc.fields.iter().any(|(g, _)| *g == f) {
            if let Some(v) = value(f) {
                doc.fields.push((f, crate::types::FieldValue::I64(v)));
            }
        }
    }
}

/// The dictionary location of a single-doc list kept inline (format
/// [`crate::codec::INLINE_POSTINGS_SINCE`]+), when `inline` and the list
/// qualifies: one doc, term frequency 1, no positions; or (`positions`:
/// format [`crate::codec::INLINE_POSITIONS_SINCE`]+) with its one position
/// (`list_positions`, the list's flattened positions, when it has any).
pub(crate) fn inline_singleton(inline: bool, positions: bool, docs: &[DocId], tfs: &[u32], list_positions: Option<&[u32]>) -> Option<u64> {
    if !inline || docs.len() != 1 || tfs.first() != Some(&1) {
        return None;
    }
    match list_positions {
        None => Some(crate::term_dictionary::inline_offset(docs[0], None)),
        Some(&[p]) if positions && p <= crate::term_dictionary::INLINE_MAX_POSITION => Some(crate::term_dictionary::inline_offset(docs[0], Some(p))),
        Some(_) => None,
    }
}

/// The reader of an inline single-doc list: the one posting encoded as a
/// postings list would be (norms unknown: a safe impact bound).
/// The first doc in `[lo, hi)` for which `before` is false (`before` is
/// true on a prefix of the range).
fn partition(mut lo: DocId, mut hi: DocId, before: impl Fn(DocId) -> bool) -> DocId {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if before(mid) { lo = mid + 1 } else { hi = mid }
    }
    lo
}

/// A synthetic-id segment's series (`_tsid` ordinals) by hash, and each
/// series' docs: a time series write looks every id up in every segment,
/// and a binary search of the front-coded term table plus two of the
/// ordinals per segment and id took most of the ingest's CPU (Rally tsdb).
/// About 16 bytes a series.
struct SeriesIndex {
    /// `(hash of the _tsid term, ordinal)`, sorted.
    by_hash: Vec<(u64, u32)>,
    /// First doc of each ordinal, then the doc count (`num_terms + 1`
    /// entries; docs without a `_tsid`, none in a time series index, sort
    /// first, before `starts[0]`).
    starts: Vec<u32>,
}

/// Hash of a `_tsid` term for [`SeriesIndex`].
fn series_hash(term: &[u8]) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = ahash::RandomState::with_seeds(0x5eed, 0x7517, 0x1d5, 0x9e37).build_hasher();
    h.write(term);
    h.finish()
}

thread_local! {
    /// The last synthetic id decoded on this thread and its parts (tsid,
    /// timestamp, tsid hash): one id is looked up in every segment.
    static LAST_SYNTHETIC_ID: std::cell::RefCell<(Vec<u8>, Option<(Vec<u8>, i64, u64)>)> = const { std::cell::RefCell::new((Vec::new(), None)) };
}

/// Synthetic id field of `segment` (for merges to carry).
pub(crate) fn synthetic_id_field(segment: &Segment) -> Option<(u16, crate::synthetic_id::SyntheticId)> {
    segment.synthetic_id
}

fn inline_postings(loc: &PostingsLocation) -> Option<PostingsReader> {
    let doc = crate::term_dictionary::inline_doc(loc.offset);
    let position = crate::term_dictionary::inline_position(loc.offset);
    let mut bytes = Vec::with_capacity(24);
    crate::codec::postings::write_postings_impl(&[doc], &[1], position.as_ref().map(std::slice::from_ref), crate::codec::postings::BlockNorms::Unknown, false, &mut bytes);
    PostingsReader::open(FileSlice::from_vec(bytes)).ok()
}

/// Column-stored fields of `segment` (for merges to carry).
pub(crate) fn column_stored_fields(segment: &Segment) -> &[u16] {
    &segment.column_stored
}

/// Keyword fields of `segment` without a column (for merges).
pub(crate) fn keyword_from_postings_fields(segment: &Segment) -> Vec<u16> {
    let mut v: Vec<u16> = segment.derived_keywords.keys().copied().collect();
    v.sort_unstable();
    v
}

/// The immutable, shared part of a [`Segment`] (see there).
pub struct SegmentData {
    terms: Arc<TermDictReader>,
    postings_data: FileSlice,
    fast_data: FileSlice,
    store_data: FileSlice,
    /// The `doc -> PointId` map per vector field (vectors themselves live
    /// in the [`VectorStore`]).
    vectors_data: FileSlice,
    /// The `[extra]` section, kept for disk-usage reporting (its entries
    /// are parsed at open).
    extra_data: FileSlice,
    /// Encoded bytes of the terms section.
    terms_len: u64,
    /// Immutable point-map offsets, parsed once instead of per filter/hit.
    point_directory: OnceLock<Vec<(u16, usize, u32)>>,
    projection_id: u64,
    /// Fast-field column directory, parsed once at open (read side).
    fast_dir: Vec<ColumnEntry>,
    /// Parsed keyword dictionary offsets, shared by every reader of this segment.
    keyword_tables: std::collections::HashMap<u16, OnceLock<Option<KeywordTable>>>,
    /// Single-valued numeric columns whose presence ranks were validated.
    numeric_checks: crate::codec::fastfields::NumericChecks,
    /// Shared zero-copy norms of every field that has them, built on first
    /// use (scorers clone the `Arc` instead of re-resolving the column).
    norms_cache: OnceLock<Vec<(u16, Arc<Norms>)>>,
    /// The stored-fields chunk table, parsed on first fetch (re-parsing it
    /// per document would walk every chunk entry on every hit).
    store_table: OnceLock<crate::admission::Allocation<StoreTable>>,
    /// Per normed field: docs with a non-zero norm (Lucene `docCount`).
    field_doc_counts: OnceLock<Vec<(u16, u64)>>,
    /// The synthetic id's timestamp column's (min, max), computed on first
    /// lookup ([`Segment::synthetic_id_doc`]): an id whose timestamp is
    /// outside it is in no doc here.
    synthetic_ts_bounds: OnceLock<Option<(i64, i64)>>,
    /// The synthetic id's series index, built on first lookup
    /// ([`SeriesIndex`]).
    synthetic_series: OnceLock<Option<SeriesIndex>>,
    /// Persisted collection statistics per indexed text / keyword field
    /// (format >= 9; empty for older files, see [`Segment::field_agg`]).
    field_aggs: Vec<(u16, FieldAgg)>,
    /// The synthetic id field and its columns ([`crate::codec::EXTRA_SYNTHETIC_ID`]):
    /// no postings or stored values of that field here.
    synthetic_id: Option<(u16, crate::synthetic_id::SyntheticId)>,
    /// The id field and its column ([`crate::codec::EXTRA_UID_COLUMN`],
    /// checked when opened): its stored documents hold no `_id`.
    uid_column: Option<(u16, FileSlice)>,
    /// Statistics derived on demand for segments older than format 9.
    legacy_aggs: parking_lot::Mutex<Vec<(u16, Option<FieldAgg>)>>,
    /// Per field: its number of terms and their summed doc freqs, counted
    /// on first use ([`Segment::field_term_stats`]).
    term_stats: parking_lot::Mutex<Vec<(u16, (u64, u64))>>,
    /// Nested path ordinals (absent when every doc is a root doc).
    nested: Option<NestedInfo>,
    /// The non-root (nested child) docs, hidden from top-level queries.
    child_bits: Option<Arc<BitSet>>,
    /// Geo index per geo-point field.
    geo: Vec<(u16, GeoIndex)>,
    /// The triangle tree and doc values of each `geo_shape` field.
    shapes: Vec<(u16, crate::fieldtypes::shape_field::ShapeFieldData)>,
    /// Fields whose stored value is their column's ([`EXTRA_COLUMN_STORED`]).
    column_stored: Vec<u16>,
    /// Keyword fields without a column ([`EXTRA_KEYWORD_FROM_POSTINGS`]),
    /// and the column derived from their postings on first use (encoded
    /// as a keyword column body, with its parsed table).
    derived_keywords: std::collections::HashMap<u16, OnceLock<Option<(crate::admission::Allocation<Vec<u8>>, KeywordTable)>>>,
    /// Filter bitsets built over this (immutable) segment, keyed by the
    /// filter's identity (see [`Segment::cached_filter_bits`]). Lucene's
    /// query cache for a numeric range reused across requests (a hybrid
    /// search's pre-filter); deletions are not baked in, so entries never
    /// go stale. Dropped with the segment.
    filter_cache: parking_lot::Mutex<Vec<(FilterCacheKey, BuildSlot<Arc<crate::admission::Allocation<Vec<u64>>>>)>>,
    /// The filters used on this segment most recently (Lucene's
    /// `UsageTrackingQueryCachingPolicy` history), see
    /// [`Segment::note_filter_use`].
    filter_uses: parking_lot::Mutex<std::collections::VecDeque<FilterCacheKey>>,
    /// Set-bit counts of [`Segment::filter_cache`] bitsets, by key (see
    /// [`Segment::filter_bits_count`]).
    filter_counts: parking_lot::Mutex<Vec<(FilterCacheKey, u64)>>,
    /// Per vector field: `Some(base)` when the `doc -> PointId` column is
    /// `base + doc` for every doc (see [`Segment::point_identity_base`]).
    point_identity: OnceLock<Vec<(u16, Option<u64>)>>,
    /// Per vector field: the `doc -> PointId` column as runs of consecutive
    /// docs with consecutive points, when they are long (see
    /// [`Segment::point_runs`]).
    point_runs: OnceLock<Vec<(u16, Option<Arc<[PointRun]>>)>>,
    /// The points of this segment's live docs for one vector field, at a
    /// deletion count (see [`Segment::cached_live_points`]).
    live_points: parking_lot::Mutex<Option<Arc<LivePoints>>>,
    vector_payload: OnceLock<Vec<Arc<VectorPayloadRow>>>,
    /// Per vector field: the point range and, built on first need, the
    /// `PointId -> doc` inverse of the point map (see [`Segment::doc_of_point`]).
    point_bounds: OnceLock<Vec<PointBounds>>,
    /// Uninverted text fielddata per field, built on first sort /
    /// aggregation (see [`Segment::uninverted`]). Dropped with the segment.
    fielddata: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::fielddata::Uninverted>>)>>,
    /// Per single-valued keyword field: block ordinal bounds, built on the
    /// first sort by the field (see [`Segment::keyword_ord_blocks`]).
    ord_blocks: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::admission::Allocation<Vec<crate::codec::fastfields::OrdBlock>>>>)>>,
    /// Per numeric sort (field, descending, by largest value, missing key):
    /// the column's blocks best first ([`Segment::sort_block_ranking`]).
    sort_ranks: parking_lot::Mutex<Vec<(SortRankKey, BuildSlot<SortRanking>)>>,
    /// Parsed keyword term tables per (field, multi-valued), built on first
    /// use so opening a keyword column per request is O(1).
    sort_keyword_tables: parking_lot::Mutex<Vec<((u16, bool), Arc<crate::codec::fastfields::KeywordOffsets>)>>,
    /// Parsed range endpoints, only for queried range fields. Per-field
    /// OnceLock avoids duplicate decoding under concurrent searches.
    range_dictionaries: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::fieldtypes::range_query::RangeDictionary>>)>>,
    /// Per i64 field: its min / max over every document
    /// ([`Segment::min_max_i64`]; a sparse or multi-valued column is a scan).
    field_bounds: parking_lot::Mutex<Vec<(u16, Option<(i64, i64)>)>>,
    /// Per geo-point field: per-block bounding boxes (geo-distance sort).
    geo_blocks: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::admission::Allocation<Vec<crate::geo::GeoBlock>>>>)>>,
    /// The last few decompressed stored-field chunks (most recent last):
    /// fetching consecutive docs (scroll, `_doc` order, sorted pages)
    /// decompresses each chunk once instead of once per doc.
    store_chunks: parking_lot::Mutex<Vec<(usize, Arc<StoredChunk>)>>,
    /// Chunks recently decoded only up to a doc (most recent last): a later
    /// visit decodes all of the chunk even once its prefix left
    /// `store_chunks` (several sequential readers of one segment).
    store_partial: parking_lot::Mutex<Vec<usize>>,
    /// Per single-valued keyword field: docs grouped by ordinal, built on
    /// the first sort that can use it (see [`Segment::keyword_ord_index`]).
    ord_index: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::admission::Allocation<crate::codec::fastfields::OrdIndex>>>)>>,
    /// Per single-valued keyword field with a front-coded term table: its
    /// terms decoded once (see [`Segment::keyword_terms_decoded`]).
    decoded_terms: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<crate::admission::Allocation<Vec<Box<str>>>>>)>>,
    /// Segment format version from the file header.
    format_version: u32,
}

impl std::ops::Deref for Segment {
    type Target = SegmentData;
    #[inline]
    fn deref(&self) -> &SegmentData {
        &self.data
    }
}

/// A segment's deletions. `latest` holds every delete applied so far (what
/// realtime lookups see); `visible`, when set, is the snapshot searches see:
/// the deletes as of the last [`Segment::publish_deletions`] (ES refresh).
/// `None` means searches see `latest` (nothing pending).
struct DelState {
    latest: Arc<BitSet>,
    visible: Option<Arc<BitSet>>,
    // Ready for the next copy-on-write before any acknowledged delete.
    spare: Option<BitSet>,
}

impl DelState {
    fn prepare_spare(&mut self) -> Result<()> {
        if self.spare.is_none() { self.spare = Some(self.latest.try_clone_admitted()?); }
        Ok(())
    }
    #[inline]
    fn visible(&self) -> &Arc<BitSet> {
        self.visible.as_ref().unwrap_or(&self.latest)
    }
}

#[cfg(test)]
thread_local! { static REFUSE_SET_DELETIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Deletions of a memtable ([`SegmentWriter`]).
struct MemDels {
    latest: BitSet,
    visible: Option<BitSet>,
    // Before WAL, reserve the snapshot used by the first pending delete.
    spare_visible: BitSet,
}

impl MemDels {
    #[inline]
    fn visible(&self) -> &BitSet {
        self.visible.as_ref().unwrap_or(&self.latest)
    }
}

/// Lucene's collection statistics of one indexed text / keyword field of a
/// segment (deleted docs included until a merge): `docCount` (docs with at
/// least one term in the field) and `sumTotalTermFreq` (every token,
/// position-overlapping ones included; for fields indexed without
/// frequencies, i.e. keywords, one per distinct term per doc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldAgg {
    pub doc_count: u64,
    pub sum_ttf: u64,
}

impl FieldAgg {
    /// The `[extra]` payload ([`EXTRA_FIELD_STATS`]).
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        b.extend_from_slice(&self.doc_count.to_le_bytes());
        b.extend_from_slice(&self.sum_ttf.to_le_bytes());
        b
    }

    /// Inverse of [`FieldAgg::encode`].
    pub fn decode(b: &[u8]) -> Option<Self> {
        Some(Self {
            doc_count: u64::from_le_bytes(b.get(..8)?.try_into().ok()?),
            sum_ttf: u64::from_le_bytes(b.get(8..16)?.try_into().ok()?),
        })
    }
}

/// `(kind, field, lower bits, upper bits)` of a cached filter bitset.
pub type FilterCacheKey = (u8, u16, u64, u64);

/// Filter bitsets remembered per segment (each is `num_docs / 8` bytes).
const FILTER_CACHE_ENTRIES: usize = 16;

/// Set-bit counts remembered per segment ([`Segment::filter_bits_count`]):
/// a few words each, so more than the bitsets themselves.
const FILTER_COUNT_ENTRIES: usize = 64;

/// Process-wide counts of the segment filter-bitset cache
/// ([`Segment::cached_filter_bits`]): lookups, builds (misses), and builds
/// the memory budget refused. A refused build makes the caller fall back to
/// its uncached path (a lazy range scorer, the per-doc intersection), so a
/// rising `refused` explains a filtered query that is fast on a small index
/// and slow on a large one.
#[derive(Debug, Default)]
pub struct FilterCacheStats {
    pub requests: std::sync::atomic::AtomicU64,
    pub builds: std::sync::atomic::AtomicU64,
    pub refused: std::sync::atomic::AtomicU64,
    pub refused_bytes: std::sync::atomic::AtomicU64,
}

/// The process's [`FilterCacheStats`].
pub static FILTER_CACHE_STATS: FilterCacheStats = FilterCacheStats {
    requests: std::sync::atomic::AtomicU64::new(0),
    builds: std::sync::atomic::AtomicU64::new(0),
    refused: std::sync::atomic::AtomicU64::new(0),
    refused_bytes: std::sync::atomic::AtomicU64::new(0),
};

/// Count a filter bitset the budget refused, and log the 1st, 2nd, 4th,
/// 8th, ... refusal with the running counts (the log stays quiet under a
/// steady stream; the last line holds the latest totals within 2x).
fn note_filter_cache_refusal(bytes: u64, error: &crate::error::RusticError) {
    use std::sync::atomic::Ordering::Relaxed;
    let s = &FILTER_CACHE_STATS;
    let n = s.refused.fetch_add(1, Relaxed) + 1;
    let refused_bytes = s.refused_bytes.fetch_add(bytes, Relaxed) + bytes;
    if n.is_power_of_two() {
        tracing::warn!(refused = n, refused_bytes, requests = s.requests.load(Relaxed), builds = s.builds.load(Relaxed),
            "segment filter cache: a filter bitset was refused by the memory budget; the query runs its uncached path: {error}");
    }
}

/// One cached per-segment structure: empty until its first requester built
/// it. Builders hold only their key's slot, never the map, so a long build
/// (an un-inversion) never blocks requesters of other keys.
type BuildSlot<V> = Arc<parking_lot::Mutex<Option<V>>>;

/// A numeric sort's cache key ([`Segment::sort_block_ranking`]): field,
/// descending, by largest value, the key of a missing value.
pub(crate) type SortRankKey = (u16, bool, bool, u128);
/// Blocks by their best sort key, best first (ties by block).
pub(crate) type SortRanking = Arc<crate::admission::Allocation<Vec<(u128, u32)>>>;

/// The value of `key` in `map`, built with `build` by its first requester
/// outside the map lock. Concurrent requesters of the same key wait for that
/// build instead of repeating it; a refused, failed or absent (`Ok(None)`)
/// build caches nothing, and the next requester builds again. With `lru`,
/// a hit moves the key to the back and an insert evicts the front past that
/// many entries.
/// Most terms [`Segment::keyword_terms_decoded`] decodes for a column.
pub const DECODED_TERMS_MAX: usize = 1 << 16;

fn cached_build<K: PartialEq, V: Clone, E>(map: &parking_lot::Mutex<Vec<(K, BuildSlot<V>)>>, key: K, lru: Option<usize>,
    build: impl FnOnce() -> std::result::Result<Option<V>, E>) -> std::result::Result<Option<V>, E> {
    let slot = {
        let mut map = map.lock();
        match map.iter().position(|(k, _)| *k == key) {
            Some(i) if lru.is_some() => { let entry = map.remove(i); let slot = entry.1.clone(); map.push(entry); slot }
            Some(i) => map[i].1.clone(),
            None => {
                if let Some(cap) = lru { if map.len() >= cap { map.remove(0); } }
                let slot: BuildSlot<V> = Arc::new(parking_lot::Mutex::new(None));
                map.push((key, slot.clone()));
                slot
            }
        }
    };
    let mut value = slot.lock();
    if let Some(value) = value.as_ref() { return Ok(Some(value.clone())); }
    let built = build();
    if let Ok(Some(built)) = &built {
        *value = Some(built.clone());
    } else {
        // Nothing to keep: the empty slot leaves the map (still under its own
        // lock, so no requester can have filled it), never caching a refusal.
        map.lock().retain(|(_, s)| !Arc::ptr_eq(s, &slot));
    }
    built
}

impl Segment {
    /// Capture a retained document/vector view for explicit PIT/scroll use.
    /// Each shared store is captured once. Files remain mapped/shared; deletion
    /// masks and point-owner metadata no longer follow subsequent live changes.
    /// Called on a reader view's segments, whose deletions are already frozen
    /// (see `ViewSlot::retain`).
    pub fn capture_read_view(segments: &[Arc<Segment>]) -> Result<Vec<Arc<Segment>>> {
        let mut stores = std::collections::HashMap::new();
        segments.iter().map(|segment| {
            let store = if let Some(store) = segment.vector_store() {
                let key = Arc::as_ptr(store) as usize;
                if let std::collections::hash_map::Entry::Vacant(entry) = stores.entry(key) {
                    entry.insert(store.capture_read_store()?);
                }
                Some(stores[&key].clone())
            } else { None };
            Ok(Arc::new(segment.capture_leaf(store)))
        }).collect()
    }

    fn capture_leaf(&self, store: Option<Arc<VectorStore>>) -> Self {
        let deletions = self.deletions.read().visible().clone();
        self.with_frozen_deletions(deletions, store, true)
    }

    /// A copy of this segment for a reader view: the same shared data and
    /// caches (nothing is re-read or rebuilt), the same vector store, and the
    /// deletions searches see now, frozen (later deletes and publishes of the
    /// live segment do not reach it). Cheap: one allocation.
    pub(crate) fn reader_snapshot(&self) -> Self {
        let deletions = self.deletions.read().visible().clone();
        // Never deleted: skip scanning the (empty) bitmap.
        let maybe_deleted = self.any_deleted.load(std::sync::atomic::Ordering::Acquire);
        self.with_frozen_deletions(deletions, self.vector_store.get().cloned(), maybe_deleted)
    }

    fn with_frozen_deletions(&self, deletions: Arc<BitSet>, store: Option<Arc<VectorStore>>, maybe_deleted: bool) -> Self {
        let vector_store = OnceLock::new();
        if let Some(store) = store { let _ = vector_store.set(store); }
        Self {
            id: self.id,
            num_docs: self.num_docs,
            any_deleted: std::sync::atomic::AtomicBool::new(maybe_deleted && !deletions.is_empty()),
            del_gen: std::sync::atomic::AtomicU64::new(0),
            live_count: std::sync::atomic::AtomicU64::new(0),
            deletions: RwLock::new(DelState { latest: deletions, visible: None, spare: None }),
            vector_store,
            data: self.data.clone(),
        }
    }

    /// A weak handle on this segment's shared data: dead once no `Segment`
    /// (live, reader snapshot, PIT capture) refers to the data any more,
    /// i.e. once nothing can read the segment's files.
    pub(crate) fn data_handle(&self) -> std::sync::Weak<SegmentData> {
        Arc::downgrade(&self.data)
    }

    /// Open segment `id` from `dir`, loading deletions if present.
    ///
    /// Standalone open: if the segment has vector fields, a private
    /// [`VectorStore`] over the directory's vector segment files is opened
    /// and attached. Use [`Segment::open_with_store`] inside an index.
    pub fn open(dir: &dyn Directory, id: u32) -> Result<Self> {
        let seg = Self::open_inner(dir, id)?;
        if !seg.point_map().is_empty() {
            VectorStore::open_dir(dir)?.attach(&seg);
        }
        Ok(seg)
    }

    /// Open segment `id` and attach it to `store` (its points become owned
    /// by this segment and alive unless deleted).
    pub fn open_with_store(dir: &dyn Directory, id: u32, store: &Arc<VectorStore>) -> Result<Self> {
        let seg = Self::open_inner(dir, id)?;
        store.attach(&seg);
        Ok(seg)
    }

    /// Open encoded text/point maps without changing any shared point ownership.
    /// Background merges attach only once the index accepts their inputs.
    pub(crate) fn open_detached(dir: &dyn Directory, id: u32) -> Result<Self> {
        Self::open_inner(dir, id)
    }

    fn open_inner(dir: &dyn Directory, id: u32) -> Result<Self> {
        Self::open_inner_admitted(dir, id, &crate::admission::shared())
    }

    fn open_inner_admitted(dir: &dyn Directory, id: u32, budget: &Arc<crate::admission::Budget>) -> Result<Self> {
        let data = dir.open_read(&segment_file_name(id))?;
        let file = open_segment_file(data, false)?;
        let terms_len = file.terms.len() as u64;
        let terms = Arc::new(TermDictReader::open_versioned(file.terms, file.version)?);
        let mut deletions = BitSet::new_admitted(file.num_docs, budget)?;
        let del_name = deletes_file_name(id);
        if dir.exists(&del_name) {
            let del = dir.open_read(&del_name)?;
            deletions.load_words(del.as_slice());
        }
        let fast_dir = parse_column_directory(file.fast.as_slice()).unwrap_or_default();
        let keyword_tables = fast_dir.iter().filter(|c| c.tag == TAG_KEYWORD).map(|c| (c.field, OnceLock::new())).collect();
        let numeric_checks = crate::codec::fastfields::numeric_checks(&fast_dir);
        let mut nested = None;
        let mut geo = Vec::new();
        let mut shapes: Vec<(u16, crate::fieldtypes::shape_field::ShapeFieldData)> = Vec::new();
        let mut field_aggs = Vec::new();
        let mut column_stored = Vec::new();
        let mut synthetic_id = None;
        let mut uid_column = None;
        let mut derived_keywords = std::collections::HashMap::new();
        let extra = file.extra.as_slice();
        for (tag, field, bytes) in decode_extra(extra)? {
            match tag {
                EXTRA_NESTED => nested = Some(NestedInfo::decode(bytes)?),
                EXTRA_COLUMN_STORED => column_stored.push(field),
                crate::codec::EXTRA_SYNTHETIC_ID => {
                    let s = crate::synthetic_id::decode_extra(bytes).ok_or_else(|| crate::RusticError::Corrupted("synthetic id entry".into()))?;
                    synthetic_id = Some((field, s));
                }
                EXTRA_KEYWORD_FROM_POSTINGS => { derived_keywords.insert(field, OnceLock::new()); }
                // Read in place from the mapped file, checked once here.
                crate::codec::EXTRA_UID_COLUMN => {
                    let at = bytes.as_ptr() as usize - extra.as_ptr() as usize;
                    let slice = file.extra.slice(at..at + bytes.len());
                    crate::codec::uid_column::UidColumn::open(slice.as_slice())?;
                    uid_column = Some((field, slice));
                }
                // Read in place from the mapped file (see `GeoIndex::open`).
                EXTRA_GEO => {
                    let at = bytes.as_ptr() as usize - extra.as_ptr() as usize;
                    geo.push((field, GeoIndex::open(&file.extra.slice(at..at + bytes.len()))?));
                }
                EXTRA_GEO2 => {
                    let at = bytes.as_ptr() as usize - extra.as_ptr() as usize;
                    geo.push((field, GeoIndex::open_compact(&file.extra.slice(at..at + bytes.len()))?));
                }
                EXTRA_FIELD_STATS => {
                    if let Some(a) = FieldAgg::decode(bytes) {
                        field_aggs.push((field, a));
                    }
                }
                // Read in place from the mapped file, like the geo index.
                EXTRA_SHAPE_TREE | EXTRA_SHAPE_DV => {
                    let at = bytes.as_ptr() as usize - extra.as_ptr() as usize;
                    let slice = file.extra.slice(at..at + bytes.len());
                    let i = match shapes.iter().position(|(f, _)| *f == field) {
                        Some(i) => i,
                        None => {
                            shapes.push((field, Default::default()));
                            shapes.len() - 1
                        }
                    };
                    let data = &mut shapes[i].1;
                    if tag == EXTRA_SHAPE_TREE {
                        data.tree.push(crate::fieldtypes::shape_bkd::ShapeBkd::open(slice)?);
                    } else {
                        data.values.get_or_insert_with(Default::default).push_part(crate::fieldtypes::shape_dv::DvPart::open(slice)?);
                    }
                }
                // Shape data of an earlier layout: refused (no mixed-version
                // reads of shapes; reindex).
                EXTRA_SHAPE_LEGACY | crate::codec::EXTRA_SHAPE_TREE_V23 | crate::codec::EXTRA_SHAPE_DV_V23 | crate::codec::EXTRA_SHAPE_TREE_V24 | crate::codec::EXTRA_SHAPE_DV_V24 => {
                    return Err(crate::error::RusticError::Segment(format!(
                        "segment {id} holds geo_shape data in a layout before segment format v{} (written as v{}); reindex the index",
                        crate::codec::SHAPE_TREE_SINCE,
                        file.version
                    )));
                }
                _ => {}
            }
        }
        let child_bits = nested.as_ref().and_then(|n: &NestedInfo| n.child_bits(file.num_docs)).map(Arc::new);
        if file.version < crate::codec::LEGACY_PRESENCE_BELOW {
            tracing::warn!("segment {id} predates fast-field presence bitmaps; sparse numeric fields may read missing values as 0 - reindex to fix");
        }
        let any_deleted = std::sync::atomic::AtomicBool::new(!deletions.is_empty());
        let spare = deletions.try_clone_admitted()?;
        let segment = Self {
            any_deleted,
            del_gen: std::sync::atomic::AtomicU64::new(0),
            live_count: std::sync::atomic::AtomicU64::new(0),
            id,
            num_docs: file.num_docs,
            deletions: RwLock::new(DelState { latest: Arc::new(deletions), visible: None, spare: Some(spare) }),
            vector_store: OnceLock::new(),
            data: Arc::new(SegmentData {
                terms,
                fast_dir,
                keyword_tables,
                numeric_checks,
                postings_data: file.postings,
                fast_data: file.fast,
                store_data: file.store,
                vectors_data: file.vectors,
                extra_data: file.extra.clone(),
                terms_len,
                point_directory: OnceLock::new(),
                projection_id: {
                    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                    NEXT.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |n| n.checked_add(1)).expect("segment identity exhausted")
                },
                norms_cache: OnceLock::new(),
                store_table: OnceLock::new(),
                field_doc_counts: OnceLock::new(),
                synthetic_ts_bounds: OnceLock::new(),
                synthetic_series: OnceLock::new(),
                field_aggs,
                synthetic_id,
                uid_column,
                legacy_aggs: parking_lot::Mutex::new(Vec::new()),
                term_stats: parking_lot::Mutex::new(Vec::new()),
                nested,
                child_bits,
                geo,
                shapes,
                column_stored,
                derived_keywords,
                filter_cache: parking_lot::Mutex::new(Vec::new()),
                filter_uses: parking_lot::Mutex::new(std::collections::VecDeque::new()),
                filter_counts: parking_lot::Mutex::new(Vec::new()),
                point_identity: OnceLock::new(),
                point_runs: OnceLock::new(),
                live_points: parking_lot::Mutex::new(None),
                vector_payload: OnceLock::new(),
                point_bounds: OnceLock::new(),
                fielddata: parking_lot::Mutex::new(Vec::new()),
                ord_blocks: parking_lot::Mutex::new(Vec::new()),
                sort_ranks: parking_lot::Mutex::new(Vec::new()),
                sort_keyword_tables: parking_lot::Mutex::new(Vec::new()),
                range_dictionaries: parking_lot::Mutex::new(Vec::new()),
                field_bounds: parking_lot::Mutex::new(Vec::new()),
                geo_blocks: parking_lot::Mutex::new(Vec::new()),
                store_chunks: parking_lot::Mutex::new(Vec::new()),
                store_partial: parking_lot::Mutex::new(Vec::new()),
                ord_index: parking_lot::Mutex::new(Vec::new()),
                decoded_terms: parking_lot::Mutex::new(Vec::new()),
                format_version: file.version,
            }),
        };
        segment.prepare_vector_payload(budget)?;
        Ok(segment)
    }

    /// Segment format version this segment was written with.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// On-disk bytes of this segment per field and structure (ES
    /// `_disk_usage`). Reads every term, column header and stored chunk:
    /// an expensive, diagnostic call.
    pub fn disk_usage(&self) -> Result<SegmentDiskUsage> {
        self.disk_usage_sampled(usize::MAX)
    }

    /// [`Segment::disk_usage`] decoding at most `max_chunks` stored chunks
    /// (evenly spaced) to split the stored bytes by field: the shares of
    /// the sampled chunks are scaled to every chunk's bytes. Every other
    /// structure (and the stored total) is exact; only the per-field split
    /// of stored bytes is estimated, and only past `max_chunks` chunks (a
    /// 330 GB index decoded every stored document).
    pub fn disk_usage_sampled(&self, max_chunks: usize) -> Result<SegmentDiskUsage> {
        let mut out = SegmentDiskUsage {
            terms: self.terms_len,
            postings: self.postings_data.len() as u64,
            fast: self.fast_data.len() as u64,
            store: self.store_data.len() as u64,
            vectors: self.vectors_data.len() as u64,
            extra: self.extra_data.len() as u64,
            ..Default::default()
        };
        // Terms and postings: a list runs from its offset to the next
        // list's (lists are written in term order).
        // Streamed (a dictionary can hold one term per doc): lists are
        // written in term order, so a list ends where the next term's starts.
        let mut prev: Option<(u16, u64)> = None;
        let mut ordered = true;
        let total_postings = out.postings;
        let mut per: std::collections::BTreeMap<u16, (u64, u64, u64)> = std::collections::BTreeMap::new();
        let unattributed = self.terms.for_each_term_bytes(|key, bytes, loc| {
            let field = key.get(..2).map_or(u16::MAX, |k| u16::from_le_bytes([k[0], k[1]]));
            let e = per.entry(field).or_default();
            e.0 += bytes;
            e.1 += 1;
            // An inline list's bytes are its dictionary entry's.
            if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 { return; }
            if let Some((pf, poff)) = prev {
                if loc.offset < poff { ordered = false; }
                per.entry(pf).or_default().2 += loc.offset.saturating_sub(poff);
            }
            prev = Some((field, loc.offset));
        })?;
        if let Some((pf, poff)) = prev {
            per.entry(pf).or_default().2 += total_postings.saturating_sub(poff);
        }
        out.field(u16::MAX).terms += unattributed;
        for (field, (bytes, count, postings)) in per {
            let f = out.field(field);
            f.terms += bytes;
            f.term_count += count;
            if ordered { f.postings += postings; }
        }
        if !ordered {
            out.field(u16::MAX).postings += total_postings;
        }
        // Columns.
        let fast = self.fast_data.as_slice();
        let mut columns = 0u64;
        for c in &self.fast_dir {
            let data = &fast[c.start..c.end];
            columns += data.len() as u64;
            let parts = crate::codec::fastfields::column_parts(c.tag, data)?;
            let f = out.field(c.field);
            f.points += parts.points;
            f.norms += parts.norms;
            f.doc_values_terms += parts.term_table;
            f.doc_values += data.len() as u64 - parts.points - parts.norms - parts.term_table;
        }
        out.field(u16::MAX).doc_values += out.fast.saturating_sub(columns);
        // Extra entries.
        let mut extra = 0u64;
        for (tag, field, bytes) in decode_extra(self.extra_data.as_slice())? {
            extra += bytes.len() as u64;
            let f = out.field(field);
            match tag {
                EXTRA_GEO | EXTRA_GEO2 => f.geo += bytes.len() as u64,
                EXTRA_SHAPE_TREE | EXTRA_SHAPE_DV => f.shape += bytes.len() as u64,
                // Stored `_id`s in their column.
                crate::codec::EXTRA_UID_COLUMN => f.stored += bytes.len() as u64,
                _ => f.other_extra += bytes.len() as u64,
            }
        }
        out.field(u16::MAX).other_extra += out.extra.saturating_sub(extra);
        // Point maps of vector fields: attributed by field from the directory.
        out.field(u16::MAX).vectors += out.vectors;
        // Stored fields: each field's serialized bytes, and its share of
        // each chunk's compressed bytes.
        let store = self.store()?;
        let n = store.num_chunks();
        let chunks: u64 = (0..n).map(|c| store.raw_chunk(c).len() as u64).sum();
        let step = n.div_ceil(max_chunks.max(1)).max(1);
        let mut sampled = 0u64;
        let mut split: std::collections::BTreeMap<u16, (u64, u64)> = std::collections::BTreeMap::new();
        for c in (0..n).step_by(step) {
            let compressed = store.raw_chunk(c).len() as u64;
            sampled += compressed;
            let raw = store.decompress(c)?;
            let compact = store.chunk_compact(c);
            let mut per: Vec<(u16, u64)> = Vec::new();
            let mut pos = 0usize;
            while pos < raw.len() {
                let start = pos;
                let len = crate::codec::varint::read_vint(&raw, &mut pos)? as usize;
                let body = crate::codec::varint::read_bytes(&raw, &mut pos, len)?;
                // Each value's own encoded bytes (a binary id's 15, not its
                // keyword's 20).
                let spans: Vec<(u16, u64)> = if compact {
                    crate::codec::store::compact_field_spans(body)?
                } else {
                    let doc: Document = crate::codec::store::decode_doc(body, false)?;
                    doc.fields.iter().map(|fv| (fv.0, bincode::serialized_size(fv).unwrap_or(0))).collect()
                };
                let mut fields = 0u64;
                for (field, n) in spans {
                    fields += n;
                    match per.iter_mut().find(|(f, _)| *f == field) {
                        Some((_, b)) => *b += n,
                        None => per.push((field, n)),
                    }
                }
                let framing = ((pos - start) as u64).saturating_sub(fields);
                match per.iter_mut().find(|(f, _)| *f == u16::MAX) {
                    Some((_, b)) => *b += framing,
                    None => per.push((u16::MAX, framing)),
                }
            }
            let raw_len = raw.len().max(1) as u64;
            for (field, n) in per {
                let e = split.entry(field).or_default();
                e.0 += n;
                e.1 += n * compressed / raw_len;
            }
        }
        // Scaled from the sampled chunks to all of them (exact when every
        // chunk was decoded).
        let scale = |x: u64| if sampled == 0 || sampled == chunks { x } else { (x as u128 * chunks as u128 / sampled as u128) as u64 };
        for (field, (raw, compressed)) in split {
            let f = out.field(field);
            f.stored_raw += scale(raw);
            f.stored += scale(compressed);
        }
        out.field(u16::MAX).stored += out.store.saturating_sub(chunks);
        Ok(out)
    }

    /// The segment predates fast-field presence bitmaps (format < v8):
    /// docs missing from a sparse numeric column may read as `0`.
    pub fn legacy_presence(&self) -> bool {
        self.format_version < crate::codec::LEGACY_PRESENCE_BELOW
    }

    /// `Some(base)` when every doc `d` of this segment has point `base + d`
    /// in vector field `field` (computed once per field, on first use).
    pub(crate) fn projection_id(&self) -> u64 { self.projection_id }

    /// Process-unique identity for caches of immutable segment contents.
    /// Unlike a borrowed reader address, this cannot be reused after close.
    /// Snapshot copies share it because their immutable columns are shared;
    /// deletion-sensitive results must not use this identity alone.
    pub fn immutable_cache_identity(&self) -> u64 { self.projection_id }

    /// `field`'s `doc -> PointId` column as runs (docs `doc..doc + len`
    /// have points `point..point + len`), when they average at least 64
    /// docs (a merged segment keeps each source's consecutive points), so
    /// a doc bitset maps to points a word at a time. Docs without a vector
    /// are in no run. Computed once per segment.
    pub fn point_runs(&self, field: u16) -> Option<Arc<[PointRun]>> {
        let v = self.point_runs.get_or_init(|| {
            let map = self.point_map();
            map.fields().into_iter().map(|f| (f, map.column(f).and_then(|c| {
                let n = c.num_docs();
                let mut runs: Vec<PointRun> = Vec::new();
                for d in 0..n {
                    let Some(p) = c.get(d) else { continue };
                    match runs.last_mut() {
                        Some(r) if r.doc + r.len == d && r.point + u64::from(r.len) == p => r.len += 1,
                        _ => {
                            if runs.len() as u64 * 64 > u64::from(n) {
                                return None;
                            }
                            runs.push(PointRun { doc: d, point: p, len: 1 });
                        }
                    }
                }
                Some(Arc::from(runs))
            }))).collect()
        });
        v.iter().find(|(f, _)| *f == field).and_then(|(_, r)| r.clone())
    }

    /// The points of this segment's live docs for `field` (`deleted`: its
    /// deletion count, which only grows, so it names the deletion set),
    /// built by `build` (`(first word, dense words over point ids)`) and
    /// kept until the deletions or the field change: an unfiltered
    /// `bbq_disk` search over deleted docs reuses it.
    pub fn cached_live_points(&self, field: u16, deleted: u32, build: impl FnOnce() -> (usize, Vec<u64>)) -> crate::error::Result<Arc<LivePoints>> {
        if let Some(lp) = self.live_points.lock().as_ref().filter(|lp| lp.field == field && lp.deleted == deleted) {
            return Ok(Arc::clone(lp));
        }
        let (first_word, words) = build();
        let memory = crate::admission::shared().reserve((words.len() as u64).saturating_mul(8).saturating_add(64), "segment live points")?;
        let lp = Arc::new(LivePoints { field, deleted, first_word, words, _memory: memory });
        *self.live_points.lock() = Some(Arc::clone(&lp));
        Ok(lp)
    }

    pub fn point_identity_base(&self, field: u16) -> Option<u64> {
        let v = self.point_identity.get_or_init(|| {
            let map = self.point_map();
            map.fields().into_iter().map(|f| (f, map.column(f).and_then(|c| c.identity_base()))).collect()
        });
        v.iter().find(|(f, _)| *f == field).and_then(|(_, b)| *b)
    }

    /// Per vector field: the lowest and highest point of this segment, from
    /// one pass over the point map (cached for the segment's lifetime; racing
    /// first callers each compute it, one result is kept, nobody waits).
    fn point_bounds(&self) -> &[PointBounds] {
        if let Some(bounds) = self.point_bounds.get() { return bounds; }
        let map = self.point_map();
        let bounds: Vec<PointBounds> = map.fields().into_iter().filter_map(|field| {
            let (min, max) = map.iter(field).fold((u64::MAX, 0), |(lo, hi), (_, p)| (lo.min(p), hi.max(p)));
            (min <= max).then(|| PointBounds { field, min, max, inverse: OnceLock::new() })
        }).collect();
        let _ = self.point_bounds.set(bounds);
        self.point_bounds.get().expect("point bounds just set")
    }

    /// The doc of this segment whose `field` vector is `point`, if any. A
    /// point outside the segment's point range is rejected at once; an
    /// identity column answers directly; otherwise from the sorted inverse
    /// of the point map, built once for the segment's lifetime (lock-free:
    /// racing builders each build, one is kept) and accounted in
    /// [`point_inverse_bytes`]. A query resolving a point the live owner
    /// table no longer places in its view never builds a per-query map.
    pub fn doc_of_point(&self, field: u16, point: u64) -> Option<DocId> {
        if let Some(base) = self.point_identity_base(field) {
            let doc = point.checked_sub(base)?;
            return (doc < u64::from(self.num_docs)).then_some(doc as DocId);
        }
        let bounds = self.point_bounds().iter().find(|b| b.field == field)?;
        if point < bounds.min || point > bounds.max { return None; }
        let inverse = match bounds.inverse.get() {
            Some(inverse) => inverse,
            None => {
                let mut pairs: Vec<(u64, DocId)> = self.point_map().iter(field).map(|(doc, p)| (p, doc)).collect();
                pairs.sort_unstable();
                crate::reader_view::note_moved_owner_rebuild(pairs.len());
                let _ = bounds.inverse.set(PointInverse::new(pairs));
                bounds.inverse.get().expect("point inverse just set")
            }
        };
        let i = inverse.0.binary_search_by_key(&point, |(p, _)| *p).ok()?;
        Some(inverse.0[i].1)
    }

    /// The filter bitset for `key` (bit `d` = doc `d` matches; deleted docs
    /// not cleared), built with `build` on the first request and remembered
    /// for later ones. A small LRU: the segment is immutable, so entries
    /// stay valid for its lifetime.
    pub fn cached_filter_bits(&self, key: FilterCacheKey, build: impl FnOnce() -> Vec<u64>) -> Result<Arc<crate::admission::Allocation<Vec<u64>>>> {
        self.cached_filter_bits_admitted(key,build,&crate::admission::shared())
    }
    /// How many bits of `bits`, the [`Segment::cached_filter_bits`] bitset
    /// of `key`, are set: counted on the first request and remembered. A
    /// key names its bitset's content exactly (the segment is immutable and
    /// deletions are not baked in), so the count holds for any later build
    /// of the same key. A scorer over a cached bitset needs its size (a
    /// conjunction picks its lead by it); counting a billion-doc
    /// segment's 16 MB bitset again on every search cost more than the
    /// search (big5's scored `@timestamp` range beside `query_string`).
    pub fn filter_bits_count(&self, key: FilterCacheKey, bits: &[u64]) -> u64 {
        if let Some(&(_, n)) = self.filter_counts.lock().iter().find(|(k, _)| *k == key) {
            return n;
        }
        let n: u64 = bits.iter().map(|w| u64::from(w.count_ones())).sum();
        let mut counts = self.filter_counts.lock();
        if !counts.iter().any(|(k, _)| *k == key) {
            if counts.len() >= FILTER_COUNT_ENTRIES {
                counts.remove(0);
            }
            counts.push((key, n));
        }
        n
    }

    /// Record a use of the filter `key` (Lucene's
    /// `UsageTrackingQueryCachingPolicy.onUse`, its history of the last
    /// 256 uses) and return how many of the remembered uses are `key`'s.
    pub fn note_filter_use(&self, key: FilterCacheKey) -> usize {
        const HISTORY: usize = 256;
        let mut uses = self.filter_uses.lock();
        if uses.len() >= HISTORY {
            uses.pop_front();
        }
        uses.push_back(key);
        uses.iter().filter(|k| **k == key).count()
    }

    /// How many of the remembered filter uses ([`Segment::note_filter_use`])
    /// are `key`'s, recording none.
    pub fn filter_uses_of(&self, key: FilterCacheKey) -> usize {
        self.filter_uses.lock().iter().filter(|k| **k == key).count()
    }

    /// The bitset [`Segment::cached_filter_bits`] holds for `key`, if one is
    /// built (never builds, never waits for a build in progress, and does
    /// not touch the LRU order).
    pub fn peek_filter_bits(&self, key: FilterCacheKey) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
        let slot = self.filter_cache.lock().iter().find(|(k, _)| *k == key).map(|(_, s)| s.clone())?;
        let value = slot.try_lock()?;
        (*value).clone()
    }
    fn cached_filter_bits_admitted(&self,key:FilterCacheKey,build:impl FnOnce()->Vec<u64>,budget:&Arc<crate::admission::Budget>)->Result<Arc<crate::admission::Allocation<Vec<u64>>>> {
        use std::sync::atomic::Ordering::Relaxed;
        FILTER_CACHE_STATS.requests.fetch_add(1,Relaxed);
        let bits=cached_build(&self.filter_cache,key,Some(FILTER_CACHE_ENTRIES),||{
            FILTER_CACHE_STATS.builds.fetch_add(1,Relaxed);
            let bytes=(self.num_docs as u64).div_ceil(64).saturating_mul(8).saturating_add(512);
            let memory=budget.reserve(bytes,"numeric filter cache").inspect_err(|error|note_filter_cache_refusal(bytes,error))?;
            Ok::<_,crate::error::RusticError>(Some(Arc::new(crate::admission::Allocation::new(build(),memory))))
        })?;
        Ok(bits.expect("a filter build always yields bits"))
    }

    /// Nested path ordinals, when this segment holds nested docs.
    pub fn nested_info(&self) -> Option<&NestedInfo> {
        self.nested.as_ref()
    }

    /// The nested (non-root) docs, if any.
    pub fn child_docs(&self) -> Option<&Arc<BitSet>> {
        self.child_bits.as_ref()
    }

    /// The triangle tree and doc values of `geo_shape` field `field`.
    pub fn shape_data(&self, field: u16) -> Option<&crate::fieldtypes::shape_field::ShapeFieldData> {
        self.shapes.iter().find(|(f, _)| *f == field).map(|(_, p)| p)
    }

    /// The `geo_shape` fields with a tree or doc values.
    pub fn shape_fields(&self) -> Vec<u16> {
        self.shapes.iter().map(|(f, _)| *f).collect()
    }

    /// The geo index of a geo-point field.
    pub fn geo_index(&self, field: u16) -> Option<&GeoIndex> {
        self.geo.iter().find(|(f, _)| *f == field).map(|(_, g)| g)
    }

    /// Bounding boxes of geo-point field `field` per block of `block` docs
    /// (built once per segment).
    pub fn geo_blocks(&self, field:u16, block:usize)->Option<Arc<crate::admission::Allocation<Vec<crate::geo::GeoBlock>>>> {
        self.geo_blocks_admitted(field,block,&crate::admission::shared())
    }
    fn geo_blocks_admitted(&self,field:u16,block:usize,budget:&Arc<crate::admission::Budget>)->Option<Arc<crate::admission::Allocation<Vec<crate::geo::GeoBlock>>>> {
        if block==0{return None;}
        let values=self.geo_index(field)?.values();
        cached_build(&self.geo_blocks,field,None,||{
            let bytes=(self.num_docs as u64).div_ceil(block as u64).saturating_mul(std::mem::size_of::<crate::geo::GeoBlock>() as u64).saturating_add(256);
            let memory=budget.reserve(bytes,"geo sort bounds").map_err(|_|crate::geo::structures::refused(&crate::geo::structures::GEO_SORT_BOUNDS_REFUSED,"geo sort bounds",bytes))?;
            Ok::<_,()>(Some(Arc::new(crate::admission::Allocation::new(values.blocks(self.num_docs,block),memory))))
        }).ok().flatten()
    }

    /// Borrow the geo field directory without allocating a listing.
    pub fn geo_field_ids(&self) -> impl Iterator<Item=u16> + '_ { self.geo.iter().map(|(f,_)|*f) }

    /// Fields with a geo index.
    pub fn geo_fields(&self) -> Vec<u16> { self.geo_field_ids().collect() }

    /// Live docs as searches see them.
    pub fn num_live_docs(&self) -> u32 {
        // No deletions ever (an append-only index): no lock, no count of
        // the deletes bitmap (a request-cache key reads every segment's
        // count on every search).
        if !self.any_deleted.load(std::sync::atomic::Ordering::Acquire) {
            return self.num_docs;
        }
        // Counted once per deletes generation, not per call (a request
        // cache key reads it for every segment on every search).
        let dels = self.deletions.read();
        let generation = self.del_gen.load(std::sync::atomic::Ordering::Acquire);
        let cached = self.live_count.load(std::sync::atomic::Ordering::Acquire);
        if cached >> 32 == (generation.wrapping_add(1) & 0xffff_ffff) {
            return cached as u32;
        }
        let n = self.num_docs - dels.visible().count();
        self.live_count.store(((generation.wrapping_add(1) & 0xffff_ffff) << 32) | n as u64, std::sync::atomic::Ordering::Release);
        n
    }

    /// Live docs including deletes not yet published (realtime).
    pub fn num_live_docs_latest(&self) -> u32 {
        self.num_docs - self.deletions.read().latest.count()
    }

    /// Whether `doc_id` is deleted as searches see it.
    pub fn is_deleted(&self, doc_id: DocId) -> bool {
        // No deletions ever: skip the lock (an atomic RMW per call).
        if !self.any_deleted.load(std::sync::atomic::Ordering::Acquire) {
            return false;
        }
        self.deletions.read().visible().get(doc_id)
    }

    /// Whether `doc_id` is deleted, unpublished deletes included (realtime).
    pub fn is_deleted_latest(&self, doc_id: DocId) -> bool {
        self.deletions.read().latest.get(doc_id)
    }

    /// Bytes of the segment's data sections (postings, doc values, stored
    /// fields, point map; the term dictionary is not counted).
    /// Bytes of the whole segment file's sections (terms and `[extra]`
    /// included): the size the merge policy weighs (Lucene `sizeInBytes`).
    pub fn segment_bytes(&self) -> u64 {
        self.terms_len + self.extra_data.len() as u64 + self.file_bytes() as u64
    }

    /// Whether the segment holds vector data (its stored chunks of large
    /// docs are small, see [`crate::codec::store::chunk_full_for`]).
    pub fn has_vector_data(&self) -> bool {
        !self.vectors_data.is_empty()
    }

    pub fn file_bytes(&self) -> usize {
        self.postings_data.len() + self.fast_data.len() + self.store_data.len() + self.vectors_data.len()
    }

    /// Delete a doc, visible to searches at once. Deleting a document that
    /// has nested children deletes its whole block (the children precede it).
    pub fn delete_doc(&self, doc_id: DocId) -> Result<()> {
        if doc_id >= self.num_docs { return Ok(()); }
        let start = self.nested.as_ref().map(|n| n.block_start(doc_id)).unwrap_or(doc_id);
        {
            let mut guard = self.deletions.write();
            self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if (start..=doc_id).all(|d| guard.visible().get(d)) { return Ok(()); }
            // Admit every replacement before changing either visibility or
            // point liveness, including the entire nested block.
            let latest = (Arc::strong_count(&guard.latest) != 1).then(|| guard.latest.try_clone_admitted()).transpose()?;
            let visible = guard.visible.as_ref().filter(|v| Arc::strong_count(v) != 1).map(|v| v.try_clone_admitted()).transpose()?;
            if let Some(bits) = latest { guard.latest = Arc::new(bits); }
            if let Some(bits) = visible { guard.visible = Some(Arc::new(bits)); }
            for d in start..=doc_id {
                Arc::get_mut(&mut guard.latest).expect("unique admitted deletion bitmap").set(d);
                if let Some(v) = guard.visible.as_mut() { Arc::get_mut(v).expect("unique admitted visible bitmap").set(d); }
            }
            self.any_deleted.store(true, std::sync::atomic::Ordering::Release);
        }
        for d in start..=doc_id { self.kill_points(d); }
        Ok(())
    }

    /// The doc's vectors die with it (point liveness).
    fn kill_points(&self, doc_id: DocId) {
        if let Some(store) = self.vector_store.get() {
            let map = self.point_map();
            for field in map.fields() {
                if let Some(p) = map.get(field, doc_id) {
                    store.delete(p, self.id, doc_id);
                }
            }
        }
    }

    /// Delete a doc for realtime lookups only: searches keep seeing it
    /// (and its vectors) until [`Segment::publish_deletions`] (Lucene's
    /// buffered deletes, applied to the searcher at refresh). Deletes the
    /// whole nested block like [`Segment::delete_doc`].
    pub fn delete_doc_pending(&self, doc_id: DocId) -> Result<()> {
        if doc_id >= self.num_docs { return Ok(()); }
        let start = self.nested.as_ref().map(|n| n.block_start(doc_id)).unwrap_or(doc_id);
        let mut guard = self.deletions.write();
        self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if (start..=doc_id).all(|d| guard.latest.get(d)) { return Ok(()); }
        let needs_copy = guard.visible.is_none() || Arc::strong_count(&guard.latest) != 1;
        if needs_copy {
            // Live segments maintain this spare at open, refresh and escaped
            // latest-snapshot boundaries; no fresh admission after WAL.
            guard.prepare_spare()?;
            let mut replacement = guard.spare.take().unwrap();
            guard.latest.copy_into_prepared(&mut replacement)?;
            if guard.visible.is_none() { guard.visible = Some(guard.latest.clone()); }
            guard.latest = Arc::new(replacement);
        }
        let latest = Arc::get_mut(&mut guard.latest).expect("unique admitted deletion bitmap");
        for d in start..=doc_id { latest.set(d); }
        self.any_deleted.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Some delete is not yet visible to searches.
    pub fn has_pending_deletions(&self) -> bool {
        self.deletions.read().visible.is_some()
    }

    /// Make every pending delete visible to searches (ES refresh): the
    /// newly deleted docs' vectors die too.
    pub fn publish_deletions(&self) -> Result<()> {
        let (old, latest) = {
            let mut guard = self.deletions.write();
            self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if guard.visible.is_none() { return Ok(()); }
            // Refusal must leave the old searchable bitmap and pending state
            // intact so a later refresh can retry safely.
            guard.prepare_spare()?;
            let old = guard.visible.take().unwrap();
            (old, guard.latest.clone())
        };
        if self.vector_store.get().is_none() || self.point_map().is_empty() {
            return Ok(());
        }
        for (wi, (&l, &o)) in latest.words().iter().zip(old.words()).enumerate() {
            let mut fresh = l & !o;
            while fresh != 0 {
                let b = fresh.trailing_zeros();
                fresh &= fresh - 1;
                self.kill_points(wi as u32 * 64 + b);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    /// Test seam: refuse the next this-many [`Segment::set_deletions_prepaid`]
    /// calls on this thread.
    pub(crate) fn refuse_next_seal_installs(count: usize) { REFUSE_SET_DELETIONS.with(|n| n.set(count)); }

    /// Install a sealed memtable's deletions: `latest`, and the snapshot
    /// searches see when it differs.
    pub(crate) fn set_deletions(&self, mut latest: BitSet, mut visible: Option<BitSet>) -> Result<()> {
        latest.admit_owned()?;
        if let Some(v) = &mut visible { v.admit_owned()?; }
        let any = !latest.is_empty();
        let spare = latest.try_clone_admitted()?;
        { let mut guard = self.deletions.write(); self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel); *guard = DelState { latest: Arc::new(latest), visible: visible.map(Arc::new), spare: Some(spare) }; }
        if any {
            self.any_deleted.store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    /// [`Segment::set_deletions`] with every bitmap, the spare copy too,
    /// charged to `memory`, admitted up front by the caller: a seal install
    /// that already holds its admission cannot be refused here.
    pub(crate) fn set_deletions_prepaid(&self, latest: BitSet, visible: Option<BitSet>, memory: &Arc<crate::admission::Reservation>) -> Result<()> {
        #[cfg(test)]
        if REFUSE_SET_DELETIONS.with(|n| { let left = n.get(); if left > 0 { n.set(left - 1); } left > 0 }) {
            return Err(crate::error::RusticError::Memory("injected seal install refusal".into()));
        }
        let latest = latest.with_memory(memory.clone());
        let visible = visible.map(|v| v.with_memory(memory.clone()));
        let any = !latest.is_empty();
        let spare = latest.clone_with_memory(memory.clone());
        { let mut guard = self.deletions.write(); self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel); *guard = DelState { latest: Arc::new(latest), visible: visible.map(Arc::new), spare: Some(spare) }; }
        if any {
            self.any_deleted.store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    /// Snapshot of the deletion bitmap searches see (an `Arc` clone;
    /// consistent for the duration of a query).
    pub fn deletions(&self) -> Arc<BitSet> {
        self.deletions.read().visible().clone()
    }

    /// Snapshot of every delete applied so far, unpublished ones included.
    pub fn latest_deletions(&self) -> Result<Arc<BitSet>> {
        let mut guard = self.deletions.write();
        self.del_gen.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        guard.prepare_spare()?;
        Ok(guard.latest.clone())
    }

    /// Synchronous vector attachment reads without creating an escaped latest
    /// snapshot or introducing fallible admission after graph registration.
    pub(crate) fn attachment_deletions(&self, latest: bool) -> parking_lot::MappedRwLockReadGuard<'_, BitSet> {
        parking_lot::RwLockReadGuard::map(self.deletions.read(), |state| {
            if latest { state.latest.as_ref() } else { state.visible().as_ref() }
        })
    }

    /// [`Segment::deletions`] when some doc is deleted, without scanning
    /// the bitmap otherwise.
    pub fn deletions_if_any(&self) -> Option<Arc<BitSet>> {
        if !self.any_deleted.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let bits = self.deletions.read().visible().clone();
        (!bits.is_empty()).then_some(bits)
    }

    /// Persist the deletion bitmap (no-op if nothing is deleted and no file exists).
    pub fn write_deletions(&self, dir: &dyn Directory) -> Result<()> {
        let d = self.deletions.read().latest.clone();
        self.write_deletions_of(dir, &d)
    }

    /// Write `d` as this segment's deletion file (a commit's point-in-time
    /// deletions, see `CommitJob::with_deletions`).
    pub fn write_deletions_of(&self, dir: &dyn Directory, d: &BitSet) -> Result<()> {
        let name = deletes_file_name(self.id);
        if d.is_empty() {
            if dir.exists(&name) {
                dir.delete(&name)?;
            }
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(d.words().len() * 8);
        for w in d.words() {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        dir.write(&name, &bytes)
    }

    pub fn term_dict(&self) -> &TermDictReader {
        &self.terms
    }

    /// The uninverted terms of (text) field `field`: built once on first
    /// use by walking the field's terms and postings, then shared by every
    /// later sort / aggregation (the segment is immutable; deleted docs
    /// keep their entries, as in Lucene). Concurrent first uses of the same
    /// field build it once.
    pub fn uninverted(&self, field: u16) -> Result<Arc<crate::fielddata::Uninverted>> {
        let value=cached_build(&self.fielddata,field,None,||self.uninvert(field).map(Some))?;
        Ok(value.expect("an un-inversion always yields a table"))
    }

    fn uninvert(&self, field: u16) -> Result<Arc<crate::fielddata::Uninverted>> {
        let mut cursor_memory=crate::admission::shared().reserve(128,"fielddata term cursor")?;
        let prefix=field.to_le_bytes();let upper=crate::types::Term::field_key_end(field);
        let mut cursor=self.terms.cursor_from(&prefix);
        let mut builder=crate::fielddata::UninvertedBuilder::new(self.num_docs,0)?;
        while cursor.advance_guarded(|needed|{if needed>cursor_memory.bytes(){cursor_memory.grow(needed-cursor_memory.bytes(),"fielddata term cursor")?;}Ok(())})? {
            let key=cursor.key();if upper.as_ref().is_some_and(|upper|key>=upper.as_slice()){break;}
            if !key.starts_with(&prefix){continue;}
            let ord=builder.add_term(&key[2..])?;
            if let Some(mut postings)=self.postings_at(&cursor.loc()) {
                while let Some(doc)=postings.next(){builder.add_doc(ord,doc)?;}
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    /// Whether field `field`'s fielddata was uninverted (test hook).
    pub fn has_uninverted(&self, field: u16) -> bool {
        self.fielddata.lock().iter().any(|(f, slot)| *f == field && slot.lock().is_some())
    }

    pub fn term_dict_arc(&self) -> Arc<TermDictReader> {
        self.terms.clone()
    }

    /// Fast fields reader (free: the column directory is parsed at open).
    pub fn fast_fields(&self) -> FastFieldsReader<'_> {
        FastFieldsReader::with_segment_tables(self.fast_data.as_slice(), &self.fast_dir, &self.keyword_tables, &self.numeric_checks)
    }

    /// Bounded, single-valued keyword metadata for vector graph partitions.
    /// High-cardinality and multi-valued columns retain ordinary traversal.
    pub(crate) fn vector_payload_rows(&self) -> &[Arc<VectorPayloadRow>] {
        self.vector_payload.get().expect("payload admitted during fallible segment open")
    }

    fn prepare_vector_payload(&self, budget: &Arc<crate::admission::Budget>) -> Result<()> {
        if self.vector_payload.get().is_some() { return Ok(()); }
        if self.point_map().is_empty() {
            let _ = self.vector_payload.set(Vec::new());
            return Ok(());
        }
        // Fixed scratch covers at most 65 borrowed terms used to reject a
        // high-cardinality field, plus eight column descriptors. Row/vector
        // headers are reserved before construction, then string/tuple growth
        // is admitted from borrowed column values before any owned copy.
        let base = (self.num_docs as u64).saturating_mul(128).saturating_add(65536);
        let mut memory = budget.reserve(base, "vector partition payload rows")?;
        let ff = self.fast_fields();
        let mut fields = Vec::with_capacity(8);
        for entry in self.fast_dir.iter().filter(|c| c.tag == TAG_KEYWORD).take(8) {
            let Some(column) = ff.get_keyword(entry.field) else { continue };
            let mut values = std::collections::BTreeSet::new();
            for doc in 0..self.num_docs {
                if let Some(value) = column.get(doc) { values.insert(value); }
                if values.len() > 64 { break; }
            }
            if values.len() > 64 { continue; }
            // Each distinct string has one Arc allocation and one transient
            // dictionary node. Per-document slack covers tuple Vec capacity.
            let strings = values.iter().fold(0u64, |n, value|
                n.saturating_add(value.len() as u64).saturating_add(128));
            let bytes = (0..self.num_docs).filter(|&doc| column.get(doc).is_some())
                .fold(strings, |n, _| n.saturating_add(128));
            memory.grow(bytes, "vector partition payload values")?;
            fields.push(entry.field);
        }
        let memory = Arc::new(memory);
        let mut rows: Vec<Vec<(u16, Arc<str>)>> = vec![Vec::new(); self.num_docs as usize];
        // Partition fields have at most 64 distinct values each. Intern their
        // immutable bytes once per segment; retained tables clone row owners,
        // and different rows share the same string allocation. The existing
        // preflight includes distinct value bytes and dictionary/header slack,
        // covering the dictionary and row storage simultaneously.
        let mut interned: std::collections::BTreeMap<std::borrow::Cow<'_, str>, Arc<str>> = std::collections::BTreeMap::new();
        for field in fields {
            let Some(column) = ff.get_keyword(field) else { continue };
            for doc in 0..self.num_docs {
                if let Some(value) = column.get(doc) {
                    let value = match interned.get(&*value) {
                        Some(v) => v.clone(),
                        None => {
                            let shared = Arc::<str>::from(&*value);
                            interned.insert(value, shared.clone());
                            shared
                        }
                    };
                    rows[doc as usize].push((field, value));
                }
            }
        }
        drop(interned);
        let rows = rows.into_iter().map(|values| Arc::new(VectorPayloadRow { values, _memory: memory.clone() })).collect();
        let _ = self.vector_payload.set(rows);
        Ok(())
    }

    fn keyword_table(&self, key: (u16, bool), data: &[u8]) -> Option<Arc<crate::codec::fastfields::KeywordOffsets>> {
        if let Some((_, t)) = self.sort_keyword_tables.lock().iter().find(|(k, _)| *k == key) {
            return Some(t.clone());
        }
        let t = crate::codec::fastfields::keyword_term_table(data).ok()?;
        let mut c = self.sort_keyword_tables.lock();
        if let Some((_, t)) = c.iter().find(|(k, _)| *k == key) {
            return Some(t.clone());
        }
        c.push((key, t.clone()));
        Some(t)
    }

    /// Lazy keyword dictionary metadata required by a reader, measured from
    /// borrowed headers before a request triggers its first parse.
    pub fn keyword_metadata_allocation_bound(&self,field:u16)->u64 {
        let ff=self.fast_fields();
        let single=self.keyword_tables.get(&field).is_some_and(|c|c.get().is_some());
        let multi=self.sort_keyword_tables.lock().iter().any(|(key,_)|*key==(field,true));
        let bound=|data:Option<&[u8]>| data.and_then(crate::codec::fastfields::term_count).map(|n|n.saturating_mul(32).saturating_add(256)).unwrap_or(0);
        (if single {0}else{bound(ff.keyword_column_bytes(field))}).saturating_add(if multi {0}else{bound(ff.multi_keyword_column_bytes(field))})
    }

    /// The single-valued keyword column of `field`, its term table parsed
    /// once per segment.
    pub fn keyword_reader(&self, field: u16) -> Option<crate::codec::fastfields::KeywordReader<'_>> {
        // Presence checks, shape filters, sorting and owned scorers must all
        // share the same single-flight dictionary cache. A separate cache here
        // doubles the offsets for high-cardinality fields (e.g. geo shapes).
        self.fast_fields().get_keyword(field).or_else(|| self.derived_keyword_reader(field))
    }

    /// Whether `field` has no keyword column in this segment, its values
    /// following from its postings (per doc: from the stored fields, see
    /// [`crate::query::reader::LeafReader::keyword_value`]).
    pub fn keyword_from_postings(&self, field: u16) -> bool {
        self.derived_keywords.contains_key(&field)
    }

    /// The fields with a doc-values column here (keywords derived from
    /// their postings too), ascending, each once: of a mapping's thousands
    /// of column fields, the hundred-odd a time series segment holds.
    pub fn column_field_ids(&self) -> Vec<u16> {
        let mut ids: Vec<u16> = self.fast_dir.iter().map(|c| c.field).chain(self.derived_keywords.keys().copied()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// The keyword column of a field without one, derived from its postings
    /// (each doc holds at most one of its terms) on first use and kept with
    /// the segment: for sorts, aggregations and scripts on `_id`. `None`
    /// when its memory is not admitted.
    fn derived_keyword_reader(&self, field: u16) -> Option<crate::codec::fastfields::KeywordReader<'_>> {
        let cell = self.derived_keywords.get(&field)?;
        let (bytes, table) = cell.get_or_init(|| self.derive_keyword_column(field).ok()).as_ref()?;
        Some(crate::codec::fastfields::KeywordReader::with_table(bytes, table))
    }

    fn derive_keyword_column(&self, field: u16) -> Result<(crate::admission::Allocation<Vec<u8>>, KeywordTable)> {
        let n = self.num_docs as usize;
        let prefix = field.to_le_bytes();
        // Every term once for the table, 4 bytes a doc while collecting.
        let mut memory = crate::admission::shared().reserve((n as u64).saturating_mul(8).saturating_add(4096), "derived keyword column")?;
        let mut ords = vec![crate::codec::fastfields::MISSING_ORD; n];
        let mut table = Vec::new();
        let mut count = 0u32;
        let mut cursor = self.terms.cursor_from(&prefix);
        while cursor.advance()? {
            let key = cursor.key();
            if key.len() < 2 || key[..2] < prefix[..] {
                continue;
            }
            if key[..2] != prefix[..] {
                break;
            }
            let text = &key[2..];
            memory.grow(text.len() as u64 + 10, "derived keyword column")?;
            crate::codec::varint::write_vint(text.len() as u64, &mut table);
            table.extend_from_slice(text);
            if let Some(mut postings) = self.postings_at(&cursor.loc()) {
                while let Some(d) = postings.next() {
                    if let Some(o) = ords.get_mut(d as usize) {
                        *o = count;
                    }
                }
            }
            count += 1;
        }
        let mut body = Vec::with_capacity(table.len() + n * 4 + 16);
        crate::codec::varint::write_vint(count as u64, &mut body);
        body.extend_from_slice(&table);
        drop(table);
        crate::codec::fastfields::write_keyword_ords(&mut body, count as usize, self.num_docs, false, &mut |f: &mut dyn FnMut(u32)| {
            for &o in &ords {
                f(o);
            }
        });
        drop(ords);
        let parsed = KeywordTable::parse(&body)?;
        memory.shrink(memory.bytes().saturating_sub(body.capacity() as u64 + (count as u64).saturating_mul(16) + 512));
        Ok((crate::admission::Allocation::new(body, memory), parsed))
    }

    /// Own the single-valued column bytes without copying documents. The
    /// segment's parsed term table is shared with ordinary keyword readers.
    pub fn owned_keyword_column(&self, field: u16) -> Option<crate::codec::fastfields::OwnedKeywordColumn> {
        let entry = self.fast_dir.iter().find(|e| e.field == field && e.tag == crate::codec::fastfields::TAG_KEYWORD)?;
        let bytes = self.fast_data.slice(entry.start..entry.end);
        let table = self.keyword_reader(field)?.shared_terms();
        crate::codec::fastfields::OwnedKeywordColumn::open_layout(bytes, Some(table), entry.layout()).ok()
    }

    pub(crate) fn range_dictionary(&self, field:u16, column:&crate::codec::fastfields::OwnedKeywordColumn) -> Result<Arc<crate::fieldtypes::range_query::RangeDictionary>> {
        // The dictionary's reservation includes its cache entry. Concurrent
        // first readers of the field wait for one construction (only its
        // slot is held, not the map); a failed one is not published.
        let dictionary=cached_build(&self.range_dictionaries,field,None,||crate::fieldtypes::range_query::RangeDictionary::new(column).map(|d|Some(Arc::new(d))))?;
        Ok(dictionary.expect("a range dictionary build always yields one"))
    }

    /// The multi-valued keyword column of `field` (term table cached).
    pub fn multi_keyword_reader(&self, field: u16) -> Option<crate::codec::fastfields::MultiKeywordReader<'_>> {
        let data = self.fast_fields().multi_keyword_column_bytes(field)?;
        let table = self.keyword_table((field, true), data);
        crate::codec::fastfields::MultiKeywordReader::open_with(data, table).ok()
    }

    /// The docs of single-valued keyword column `field` grouped by ordinal
    /// (built once per segment, then shared).
    pub fn keyword_ord_index(&self, field: u16) -> Result<Option<Arc<crate::admission::Allocation<crate::codec::fastfields::OrdIndex>>>> {
        self.keyword_ord_index_with_budget(field,&crate::admission::shared())
    }
    fn keyword_ord_index_with_budget(&self,field:u16,budget:&Arc<crate::admission::Budget>)->Result<Option<Arc<crate::admission::Allocation<crate::codec::fastfields::OrdIndex>>>> {
        let Some(reader)=self.keyword_reader(field) else{return Ok(None)};
        cached_build(&self.ord_index,field,None,||{
            let peak=(reader.num_terms() as u64).saturating_add(1).saturating_mul(8).saturating_add((reader.num_docs() as u64).saturating_mul(4)).saturating_add(512);
            let mut memory=budget.reserve(peak,"keyword ordinal index")?;
            let value=reader.ord_index();
            let retained=(value.docs.capacity() as u64).saturating_add(value.starts.capacity() as u64).saturating_mul(4).saturating_add(512);
            memory.shrink(memory.bytes().saturating_sub(retained));
            Ok(Some(Arc::new(crate::admission::Allocation::new(value,memory))))
        })
    }

    /// The terms of single-valued keyword column `field`, decoded once and
    /// shared, when its term table is front-coded (decoding a term replays
    /// its block) and holds at most [`DECODED_TERMS_MAX`] terms: callers that
    /// read a term per document (SQL rows over a keyword column) index it by
    /// ordinal instead. `None`: a plain table (already direct), a large
    /// dictionary, or no column; read terms from the reader then.
    pub fn keyword_terms_decoded(&self, field: u16) -> Result<Option<Arc<crate::admission::Allocation<Vec<Box<str>>>>>> {
        let Some(reader)=self.keyword_reader(field) else{return Ok(None)};
        if !reader.terms_front_coded() || reader.num_terms() > DECODED_TERMS_MAX {
            return Ok(None);
        }
        cached_build(&self.decoded_terms,field,None,||{
            let bytes=(reader.num_terms() as u64).saturating_mul(16).saturating_add(reader.term_table_len() as u64).saturating_add(512);
            let memory=crate::admission::shared().reserve(bytes,"decoded keyword terms")?;
            let mut terms=Vec::with_capacity(reader.num_terms());
            for ord in 0..reader.num_terms() as u32 {
                match reader.term(ord) {
                    Some(t)=>terms.push(Box::<str>::from(&*t)),
                    None=>return Ok(None),
                }
            }
            Ok(Some(Arc::new(crate::admission::Allocation::new(terms,memory))))
        })
    }

    /// Optional sort acceleration: callers can scan ordinary doc values when
    /// this shared cache cannot be admitted, without changing matching/sort.
    pub fn keyword_ord_blocks(&self, field: u16) -> Result<Option<Arc<crate::admission::Allocation<Vec<crate::codec::fastfields::OrdBlock>>>>> {
        let Some(reader)=self.keyword_reader(field) else{return Ok(None)};
        cached_build(&self.ord_blocks,field,None,||{
            let bytes=(reader.num_docs() as u64).div_ceil(crate::codec::fastfields::BLOCK_SIZE as u64).saturating_mul(std::mem::size_of::<crate::codec::fastfields::OrdBlock>() as u64).saturating_add(512);
            let memory=crate::admission::shared().reserve(bytes,"keyword ordinal blocks")?;
            Ok(Some(Arc::new(crate::admission::Allocation::new(reader.ord_blocks(),memory))))
        })
    }

    /// The best `blocks` blocks of a numeric sort's column, best key first,
    /// built once (`build`, admitted: 32 bytes a block) and kept with the segment (a
    /// few sorts at a time): a sort that needs no count walks them instead
    /// of ranking every block per search. `None` when not admitted.
    pub(crate) fn sort_block_ranking(&self, key: SortRankKey, blocks: usize, build: impl FnOnce() -> Vec<(u128, u32)>) -> Option<SortRanking> {
        cached_build(&self.sort_ranks, key, Some(4), || {
            let bytes = (blocks as u64).saturating_mul(32).saturating_add(256);
            let memory = crate::admission::shared().reserve(bytes, "sort block ranking")?;
            Ok::<_, crate::RusticError>(Some(Arc::new(crate::admission::Allocation::new(build(), memory))))
        }).ok().flatten()
    }

    /// Postings for a term, if present. The reader owns an `Arc` to the
    /// mapped file, so it can outlive this borrow and be boxed into a
    /// scorer tree.
    pub fn postings(&self, term: &Term) -> Option<PostingsReader> {
        let loc = self.term_info(term)?;
        self.postings_at(&loc)
    }

    /// Postings at a dictionary location.
    #[inline]
    pub fn postings_at(&self, loc: &PostingsLocation) -> Option<PostingsReader> {
        if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 {
            return inline_postings(loc);
        }
        let len = self.postings_data.len();
        // The dictionary entry carries the header's figures: nothing to decode.
        PostingsReader::open_known(self.postings_data.slice(loc.offset as usize..len), loc.doc_freq, loc.total_term_freq).ok()
    }

    /// Per-doc field lengths for a text field (BM25 norms), if indexed.
    pub fn norms(&self, field: u16) -> Option<NormsReader<'_>> {
        self.fast_fields().get_norms(field)
    }

    /// Average field length for BM25; 1.0 when the field has no norms.
    pub fn avg_field_len(&self, field: u16) -> f32 {
        self.norms(field).map(|n| n.avg()).unwrap_or(1.0)
    }

    /// Docs of this segment with at least one term in `field` (Lucene's
    /// `Terms.getDocCount`, deleted docs included). `None` for fields without
    /// collection statistics (not a text / keyword field).
    pub fn field_doc_count(&self, field: u16) -> Option<u64> {
        self.field_agg(field).map(|a| a.doc_count)
    }

    /// Collection statistics of `field` in this segment: persisted since
    /// format 9, derived once (term dictionary, norms, fast-field presence)
    /// for older files. `None` when the field has no terms of its own kind.
    pub fn field_agg(&self, field: u16) -> Option<FieldAgg> {
        if self.format_version >= crate::codec::LEGACY_FIELD_STATS_BELOW {
            return self.field_aggs.iter().find(|(f, _)| *f == field).map(|(_, a)| *a);
        }
        let mut cache = self.legacy_aggs.lock();
        if let Some((_, a)) = cache.iter().find(|(f, _)| *f == field) {
            return *a;
        }
        let a = self.derive_agg(field);
        cache.push((field, a));
        a
    }

    /// The number of terms of `field` and the sum of their doc freqs
    /// (Lucene `Terms.size()` / `Terms.getSumDocFreq()`): one walk of the
    /// field's terms on first use, then cached with the segment.
    pub fn field_term_stats(&self, field: u16) -> (u64, u64) {
        if let Some((_, stats)) = self.term_stats.lock().iter().find(|(f, _)| *f == field) {
            return *stats;
        }
        let prefix = field.to_le_bytes();
        let (mut terms, mut sum_doc_freq) = (0u64, 0u64);
        let mut cursor = self.terms.cursor_from(&prefix);
        while let Ok(true) = cursor.advance() {
            let key = cursor.key();
            if key.len() < 2 || key[..2] < prefix[..] {
                continue;
            }
            if key[..2] != prefix[..] {
                break;
            }
            terms += 1;
            sum_doc_freq += cursor.loc().doc_freq as u64;
        }
        let mut cache = self.term_stats.lock();
        if !cache.iter().any(|(f, _)| *f == field) {
            cache.push((field, (terms, sum_doc_freq)));
        }
        (terms, sum_doc_freq)
    }

    /// Statistics of a pre-format-9 segment, from what the file has.
    fn derive_agg(&self, field: u16) -> Option<FieldAgg> {
        let terms = self.terms.range_iter(field, None, None);
        if terms.is_empty() {
            return None;
        }
        let sum_ttf = terms.iter().map(|(_, l)| l.total_term_freq).sum();
        let doc_count = match self.norms_doc_count(field) {
            Some(n) => n,
            None => self.docs_with_field(field, &terms, false),
        };
        Some(FieldAgg { doc_count, sum_ttf })
    }

    /// Docs with a non-zero norm byte (cached per segment); `None` when the
    /// field has no norms.
    fn norms_doc_count(&self, field: u16) -> Option<u64> {
        let cache = self.field_doc_counts.get_or_init(|| {
            let mut fields: Vec<u16> = self.fast_dir.iter().map(|e| e.field).collect();
            fields.sort_unstable();
            fields.dedup();
            fields
                .into_iter()
                .filter_map(|f| self.norms(f).map(|n| (f, n.bytes().iter().filter(|&&b| b != 0).count() as u64)))
                .collect()
        });
        cache.iter().find(|(f, _)| *f == field).map(|(_, c)| *c)
    }

    /// Number of docs having at least one term of `field` (only live docs
    /// when `live_only`): keyword doc-values presence when the field has a
    /// column, else the union of its postings.
    fn docs_with_field(&self, field: u16, terms: &[(Vec<u8>, PostingsLocation)], live_only: bool) -> u64 {
        let keep = |d: DocId| !live_only || !self.is_deleted(d);
        let ff = self.fast_fields();
        if let Some(c) = ff.get_keyword(field) {
            return (0..self.num_docs).filter(|&d| keep(d) && c.get(d).is_some()).count() as u64;
        }
        if let Some(c) = ff.get_multi_keyword(field) {
            return (0..self.num_docs).filter(|&d| keep(d) && c.count(d) > 0).count() as u64;
        }
        if let Some(c) = ff.get_features(field) {
            return (0..self.num_docs).filter(|&d| keep(d) && c.count(d) > 0).count() as u64;
        }
        let mut bits = BitSet::new(self.num_docs);
        for (_, loc) in terms {
            let Some(mut r) = self.postings_at(loc) else { continue };
            let mut next = 0;
            while let Some(d) = r.seek_forward(next) {
                bits.set(d);
                next = d + 1;
            }
        }
        bits.iter_set_bits().filter(|&d| keep(d)).count() as u64
    }

    /// [`Segment::field_agg`] counting only live docs' `doc_count` (what a
    /// merge keeps). `sum_ttf` stays the segment total: merges recompute it
    /// from the surviving postings.
    pub(crate) fn live_field_doc_count(&self, field: u16) -> Option<u64> {
        let agg = self.field_agg(field)?;
        if !self.any_deleted.load(std::sync::atomic::Ordering::Relaxed) {
            return Some(agg.doc_count);
        }
        if let Some(n) = self.norms(field) {
            let bytes = n.bytes();
            return Some((0..self.num_docs).filter(|&d| !self.is_deleted(d) && bytes.get(d as usize).copied().unwrap_or(0) != 0).count() as u64);
        }
        let terms = self.terms.range_iter(field, None, None);
        Some(self.docs_with_field(field, &terms, true))
    }

    /// [`Segment::norms_shared`] behind a per-segment cached `Arc`.
    pub fn norms_arc(&self, field: u16) -> Arc<Norms> {
        let cache = self.norms_cache.get_or_init(|| {
            let mut fields: Vec<u16> = self.fast_dir.iter().map(|e| e.field).collect();
            fields.sort_unstable();
            fields.dedup();
            fields
                .into_iter()
                .filter(|&f| self.norms(f).is_some())
                .map(|f| (f, Arc::new(self.norms_shared(f))))
                .collect()
        });
        match cache.iter().find(|(f, _)| *f == field) {
            Some((_, n)) => n.clone(),
            None => unit_norms(),
        }
    }

    /// Zero-copy norms for a scorer (no per-query copy of the column).
    pub fn norms_shared(&self, field: u16) -> Norms {
        let Some(n) = self.norms(field) else {
            return Norms::unit();
        };
        let bytes = n.lengths_bytes();
        if let Some(run) = n.constant_run() {
            // A run shared by the process, not in the segment's mapping.
            return Norms::constant(run, n.avg());
        }
        let base = self.fast_data.as_slice().as_ptr() as usize;
        let off = bytes.as_ptr() as usize - base;
        Norms::mapped(self.fast_data.slice(off..off + bytes.len()), n.avg())
    }

    /// The smallest and largest i64 fast-field value of `field` over every
    /// document, deleted ones included; computed once per segment (the
    /// can-match phase asks per search).
    pub fn min_max_i64(&self, field: u16) -> Option<(i64, i64)> {
        if let Some((_, b)) = self.field_bounds.lock().iter().find(|(f, _)| *f == field) {
            return *b;
        }
        let ff = self.fast_fields();
        let bounds = match ff.get_i64(field) {
            Some(c) => c.min().zip(c.max()),
            None => ff.get_multi_i64(field).and_then(|c| {
                (0..self.num_docs).filter_map(|d| c.min_of(d).zip(c.max_of(d))).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
            }),
        };
        self.field_bounds.lock().push((field, bounds));
        bounds
    }

    /// Fetch a stored document.
    pub fn get_doc(&self, doc_id: DocId) -> Result<Option<Document>> {
        self.get_doc_admitted(doc_id, &crate::admission::shared())
    }

    /// Put back the column-stored fields ([`EXTRA_COLUMN_STORED`]) a stored
    /// doc of `doc_id` was written without (docs stored with them, from
    /// older segments merged in, keep theirs).
    fn restore_column_stored(&self, doc_id: DocId, doc: &mut Document) {
        if !self.column_stored.is_empty() {
            let ff = self.fast_fields();
            restore_column_stored(doc, &self.column_stored, |f| ff.get_i64(f).and_then(|c| c.get(doc_id)));
        }
        self.restore_synthetic_id(doc_id, doc);
    }

    /// The id field of this segment's id column ([`crate::codec::EXTRA_UID_COLUMN`]).
    pub fn uid_column_field(&self) -> Option<u16> {
        self.uid_column.as_ref().map(|c| c.0)
    }

    /// This segment's id column, if it has one.
    pub fn uid_column(&self) -> Option<crate::codec::uid_column::UidColumn<'_>> {
        let (_, data) = self.uid_column.as_ref()?;
        crate::codec::uid_column::UidColumn::open_unchecked(data.as_slice())
    }

    /// The `_id` of `doc` from the id column (`None`: no column, or no id).
    pub fn column_uid(&self, doc: DocId) -> Option<String> {
        let col = self.uid_column()?;
        // The thread's last block of this segment: a doc in a block not read
        // before decodes only up to itself (a page's scattered hits); a
        // second doc of the same block decodes the whole block once (a
        // page's hits in doc order); the same doc again is remembered (a
        // hit's fetch and its `_id` render).
        thread_local! {
            static LAST: std::cell::RefCell<(u64, usize, bool, crate::codec::uid_column::IdList, DocId, Option<String>)> = std::cell::RefCell::new((0, usize::MAX, false, Default::default(), 0, None));
        }
        let block = doc as usize / crate::codec::uid_column::BLOCK;
        LAST.with(|last| {
            let mut last = last.borrow_mut();
            let here = last.0 == self.projection_id && last.1 == block;
            if here && last.2 {
                return crate::uid::decode(last.3.get(doc as usize % crate::codec::uid_column::BLOCK)?);
            }
            if here && last.4 == doc {
                return last.5.clone();
            }
            if !here {
                let id = crate::uid::decode(&col.get(doc)?);
                *last = (self.projection_id, block, false, Default::default(), doc, id.clone());
                return id;
            }
            let mut cur = crate::codec::uid_column::Cursor::new(col);
            cur.get(doc).ok()?;
            last.2 = true;
            last.3 = cur.into_block();
            crate::uid::decode(last.3.get(doc as usize % crate::codec::uid_column::BLOCK)?)
        })
    }

    /// Put back the synthetic id ([`crate::synthetic_id`]) or the column's
    /// `_id` a stored doc of this segment was written without.
    fn restore_synthetic_id(&self, doc_id: DocId, doc: &mut Document) {
        if let Some(field) = self.uid_column_field() {
            if !doc.fields.iter().any(|(f, _)| *f == field) {
                if let Some(id) = self.column_uid(doc_id) {
                    doc.fields.insert(0, (field, crate::types::FieldValue::Keyword(id)));
                }
            }
        }
        let Some((field, s)) = self.synthetic_id else { return };
        restore_synthetic_id(doc, field, || {
            let ff = self.fast_fields();
            let tsid = self.keyword_reader(s.tsid)?.get(doc_id)?.as_bytes().to_vec();
            Some((tsid, ff.get_i64(s.timestamp)?.get(doc_id)?, ff.get_i64(s.routing_hash)?.get(doc_id)?))
        });
    }

    /// Whether `field` is this segment's synthetic id field (no postings).
    pub fn is_synthetic_id(&self, field: u16) -> bool {
        self.synthetic_id.is_some_and(|(f, _)| f == field)
    }

    /// The doc of synthetic id `text` ([`crate::synthetic_id`]): its `_tsid`
    /// docs found by binary search (the segment is sorted by `_tsid`
    /// ascending, then the timestamp descending), then its timestamp's. Of
    /// several (a deleted copy and its replacement), the live one.
    fn synthetic_id_doc(&self, text: &[u8]) -> Option<DocId> {
        let (_, s) = self.synthetic_id?;
        LAST_SYNTHETIC_ID.with(|last| {
            let mut last = last.borrow_mut();
            if last.0 != text {
                let decoded = (crate::synthetic_id::codec()?.decode)(text).map(|(tsid, ts)| {
                    let h = series_hash(&tsid);
                    (tsid, ts, h)
                });
                last.0.clear();
                last.0.extend_from_slice(text);
                last.1 = decoded;
            }
            let (tsid, ts, hash) = last.1.as_ref()?;
            self.synthetic_series_doc(s, tsid, *ts, *hash)
        })
    }

    /// [`Segment::synthetic_id_doc`] of a decoded id: series `tsid` (its
    /// [`series_hash`] `hash`) at timestamp `ts`.
    fn synthetic_series_doc(&self, s: crate::synthetic_id::SyntheticId, tsid: &[u8], ts: i64, hash: u64) -> Option<DocId> {
        // Outside this segment's time range: no doc of the id here (a
        // time series write checks every segment, and appends carry newer
        // timestamps than all but the newest segments).
        let bounds = *self.data.synthetic_ts_bounds.get_or_init(|| {
            let c = self.fast_fields().get_i64(s.timestamp)?;
            Some((c.min()?, c.max()?))
        });
        if bounds.is_some_and(|(lo, hi)| ts < lo || ts > hi) {
            return None;
        }
        let n = self.num_docs;
        // The series' ordinal by hash (then its term compared), then its
        // docs by ordinal (the terms and the index sort order `_tsid`
        // alike, by bytes): no term table search and no ordinal probes.
        let index = self.data.synthetic_series.get_or_init(|| {
            let kw = self.keyword_reader(s.tsid)?;
            let mut by_hash = Vec::with_capacity(kw.num_terms());
            kw.for_each_term(|ord, term| {
                by_hash.push((series_hash(term), ord));
                true
            })?;
            by_hash.sort_unstable();
            let mut starts = vec![0u32; kw.num_terms() + 1];
            let mut without = 0u32;
            for d in 0..n {
                match kw.ord(d) {
                    crate::codec::fastfields::MISSING_ORD => without += 1,
                    o => *starts.get_mut(o as usize + 1)? += 1,
                }
            }
            starts[0] = without;
            for i in 1..starts.len() {
                starts[i] += starts[i - 1];
            }
            Some(SeriesIndex { by_hash, starts })
        });
        // The series of the hash (one, but for a collision): none here
        // opens no column.
        let candidates: &[(u64, u32)] = match index {
            Some(index) => {
                let at = index.by_hash.partition_point(|&(h, _)| h < hash);
                let len = index.by_hash[at..].iter().take_while(|&&(h, _)| h == hash).count();
                if len == 0 {
                    return None;
                }
                &index.by_hash[at..at + len]
            }
            None => &[],
        };
        let ts_col = self.fast_fields().get_i64(s.timestamp)?;
        // Timestamps descend within a series: the doc of `ts` in `[lower,
        // upper)`, a live one first.
        let dels = self.deletions();
        let in_series = |lower: DocId, upper: DocId| -> Option<DocId> {
            let first = partition(lower, upper, |d| ts_col.get(d).is_some_and(|t| t > ts));
            let mut found = None;
            for d in first..upper {
                if ts_col.get(d) != Some(ts) {
                    break;
                }
                if !dels.get(d) {
                    return Some(d);
                }
                found.get_or_insert(d);
            }
            found
        };
        match index {
            // Its term compared only once a doc has the timestamp (a new
            // id, the common case, decodes no term).
            Some(index) => candidates.iter().find_map(|&(_, ord)| {
                let (lower, upper) = (*index.starts.get(ord as usize)?, *index.starts.get(ord as usize + 1)?);
                let doc = in_series(lower, upper)?;
                self.keyword_reader(s.tsid)?.term_bytes(ord).is_some_and(|t| *t == *tsid).then_some(doc)
            }),
            None => {
                let kw = self.keyword_reader(s.tsid)?;
                let ord = kw.lookup_ord(std::str::from_utf8(tsid).ok()?)?;
                let missing = crate::codec::fastfields::MISSING_ORD;
                let lower = partition(0, n, |d| { let o = kw.ord(d); o == missing || o < ord });
                let upper = partition(lower, n, |d| { let o = kw.ord(d); o != missing && o <= ord });
                in_series(lower, upper)
            }
        }
    }

    /// [`Segment::postings`]' dictionary half: the term's location and
    /// statistics (a synthetic id's doc as an inline singleton).
    pub fn term_info(&self, term: &Term) -> Option<PostingsLocation> {
        if self.is_synthetic_id(term.field) {
            let doc = self.synthetic_id_doc(&term.text)?;
            return Some(PostingsLocation { offset: crate::term_dictionary::inline_offset(doc, None), doc_freq: 1, total_term_freq: 1 });
        }
        self.terms.get(term)
    }

    /// Whether this segment may hold the unique (id) term hashed to `hash`
    /// (the id filter; a synthetic id field has none: true).
    pub fn may_contain_unique(&self, field: u16, hash: u64) -> bool {
        self.is_synthetic_id(field) || self.terms.may_contain_unique_hash(field, hash)
    }

    /// [`Segment::may_contain_unique`] of a term.
    pub fn may_contain_unique_term(&self, term: &Term) -> bool {
        self.is_synthetic_id(term.field) || self.terms.may_contain_unique_term(term)
    }
    /// The stored documents of `docs` (any order, each at most once),
    /// `f(index in docs, document)`, each store chunk decompressed once (to
    /// the last of its docs asked), bypassing the per-segment chunk cache: a
    /// page of hits reads each document once. One reservation, sized to what
    /// the documents retain, is shared by them (what `get_doc` charges each
    /// document, together). Documents the store does not hold are skipped.
    /// `restore`: put back the column-stored fields (`get_doc` does); a
    /// reader that reads none of them (a hit's `_source`) skips them.
    pub fn get_docs_batched(&self, docs: &[DocId], restore: bool, mut f: impl FnMut(usize, Document)) -> Result<()> {
        let store = self.store()?;
        let mut order: Vec<(usize, usize, usize)> = docs.iter().enumerate()
            .filter_map(|(i, &d)| store.locate(d).map(|(c, w)| (c, w, i))).collect();
        order.sort_unstable();
        let mut memory = crate::admission::shared().reserve(0, "stored document decode")?;
        let mut decoded = Vec::with_capacity(order.len());
        let mut retained = 0u64;
        let mut k = 0;
        while k < order.len() {
            let chunk = order[k].0;
            let end = k + order[k..].iter().take_while(|o| o.0 == chunk).count();
            let transient = store.decoded_chunk_bytes(chunk)
                .ok_or_else(|| crate::RusticError::Corrupted("stored chunk is missing LZ4 size".into()))? as u64 + 256;
            memory.grow(transient, "stored document decode")?;
            let raw = store.decompress_through(chunk, order[end - 1].1)?;
            let compact = store.chunk_compact(chunk);
            // The docs come in chunk order: one walk over the chunk's
            // frames, not one from its start per doc.
            let (mut pos, mut at) = (0usize, 0usize);
            for &(_, within, i) in &order[k..end] {
                while at < within {
                    let len = crate::codec::varint::read_vint(&raw, &mut pos)? as usize;
                    pos = pos.checked_add(len).filter(|&p| p <= raw.len()).ok_or_else(|| crate::RusticError::Corrupted("stored doc out of range".into()))?;
                    at += 1;
                }
                let bytes = crate::codec::store::doc_bytes_at(&raw, pos)?;
                let peak = Document::decode_peak_bytes_as(bytes.len() as u64, compact);
                memory.grow(peak, "stored document decode")?;
                let mut document: Document = crate::codec::store::decode_doc(bytes, compact)?;
                if restore { self.restore_column_stored(docs[i], &mut document); } else { self.restore_synthetic_id(docs[i], &mut document); }
                let keep = document.decoded_bytes().saturating_add(256);
                memory.shrink(peak.saturating_sub(keep));
                retained = retained.saturating_add(keep);
                decoded.push((i, document));
            }
            memory.shrink(transient);
            k = end;
        }
        memory.shrink(memory.bytes().saturating_sub(retained));
        let memory = Arc::new(memory);
        for (i, mut document) in decoded {
            document.decode_memory = Some(memory.clone());
            f(i, document);
        }
        Ok(())
    }
    fn get_doc_admitted(&self, doc: DocId, budget: &Arc<crate::admission::Budget>) -> Result<Option<Document>> {
        let mut memory = budget.reserve(0, "stored document decode")?;
        let result = self.get_doc_guarded(doc, |bytes| {
            if bytes > memory.bytes() { memory.grow(bytes - memory.bytes(), "stored document decode")?; }
            Ok::<_, crate::RusticError>(())
        })?;
        // Retain what the decoded document holds (its capacities), not a
        // multiple of its encoded size.
        Ok(result.map(|mut document| { let retained = document.decoded_bytes().saturating_add(256); memory.shrink(memory.bytes().saturating_sub(retained)); document.decode_memory = Some(Arc::new(memory)); document }))
    }
    #[cfg(test)]
    fn get_doc_with_cache_budget(&self,doc_id:DocId,budget:&Arc<crate::admission::Budget>)->Option<Document> {
        let store = self.store().ok()?;
        let (chunk, within) = store.locate(doc_id)?;
        let cached = {
            // Least recently used goes first: a hit moves to the back.
            let mut c = self.store_chunks.lock();
            c.iter().position(|(i, _)| *i == chunk).map(|p| {
                let e = c.remove(p);
                let raw = e.1.clone();
                c.push(e);
                raw
            })
        };
        use crate::query::stats::{add, add_since, start, C};
        add(C::store_docs, 1);
        // A cached chunk may be a prefix that ends before this doc.
        let t = start();
        let revisit = cached.is_some() || self.store_partial.lock().contains(&chunk);
        if let Some(doc) = cached.and_then(|raw| raw.doc_as(within, store.chunk_compact(chunk)).ok()) {
            add(C::store_cache_hits, 1);
            add_since(C::store_decode_ns, t);
            return Some(doc);
        }
        let t = start();
        // A second visit past the cached prefix (sequential reads: copies,
        // slices, scripts reading `_source`) decodes and caches the whole
        // chunk, so a chunk is never re-decoded from its start per doc.
        let memory=store.decoded_chunk_bytes(chunk).and_then(|n|budget.reserve((n as u64).saturating_mul(2).saturating_add(512),"stored chunk cache").ok());
        let raw = if revisit && memory.is_some() { store.decompress(chunk) } else { store.decompress_through(chunk, within) };
        let raw = Arc::new(StoredChunk::new(raw.ok()?,memory));
        {
            let mut p = self.store_partial.lock();
            p.retain(|&c| c != chunk);
            if !revisit && within + 1 < store.chunk_docs(chunk).1 as usize {
                if p.len() >= Self::STORE_PARTIAL_CHUNKS {
                    p.remove(0);
                }
                p.push(chunk);
            }
        }
        add_since(C::store_decompress_ns, t);
        add(C::store_decompressed_bytes, raw.len() as u64);
        let t = start();
        let doc = raw.doc_as(within, store.chunk_compact(chunk)).ok();
        add_since(C::store_decode_ns, t);
        if raw._memory.is_none() {return doc;}
        let mut c = self.store_chunks.lock();
        match c.iter_mut().find(|(i, _)| *i == chunk) {
            Some(e) if e.1.len() < raw.len() => e.1 = raw,
            Some(_) => {}
            None => {
                if c.len() >= Self::STORE_CHUNK_CACHE {
                    c.remove(0);
                }
                c.push((chunk, raw));
            }
        }
        doc
    }

    /// Fallible preflight at each owned decode boundary. The callback receives
    /// a conservative peak byte bound before table/decompression/document
    /// allocation; a caller may retain the reservation with the decoded value.
    pub fn get_doc_guarded<E: From<crate::RusticError>>(&self,doc:DocId,mut before:impl FnMut(u64)->std::result::Result<(),E>)->std::result::Result<Option<Document>,E> {
        use crate::query::stats::{add, add_since, start, C};
        add(C::store_docs, 1);
        let decode_started = start();
        let table_bytes=if self.store_table.get().is_none() {
            let bytes=StoreTable::allocation_bound(self.store_data.as_slice()).map_err(E::from)?;
            before(bytes)?; bytes
        } else {0};
        let store=self.store().map_err(E::from)?;
        let Some((chunk,within))=store.locate(doc) else {return Ok(None)};
        let cached={
            let mut cache=self.store_chunks.lock();
            cache.iter().position(|(c,_)|*c==chunk).map(|pos|{let entry=cache.remove(pos);let raw=entry.1.clone();cache.push(entry);raw})
        };
        let revisit=cached.is_some() || self.store_partial.lock().contains(&chunk);
        if let Some(raw)=cached {
            if let Ok(bytes)=raw.doc_bytes(within) {
                let compact=store.chunk_compact(chunk);
                before(table_bytes.saturating_add(Document::decode_peak_bytes_as(bytes.len() as u64,compact)))?;
                let mut document: Document = crate::codec::store::decode_doc(bytes, compact).map_err(E::from)?;
                self.restore_column_stored(doc, &mut document);
                add(C::store_cache_hits, 1); add_since(C::store_decode_ns, decode_started);
                return Ok(Some(document));
            }
        }
        let decoded=store.decoded_chunk_bytes(chunk).ok_or_else(|| E::from(crate::RusticError::Corrupted("stored chunk is missing LZ4 size".into())))?;
        let peak=table_bytes.saturating_add(decoded as u64).saturating_add(256);
        before(peak)?;
        let memory=crate::admission::shared().reserve((decoded as u64).saturating_mul(2).saturating_add(512),"stored chunk cache").ok();
        let decompress_started = start();
        let raw=if revisit && memory.is_some() {store.decompress(chunk)} else {store.decompress_through(chunk,within)};
        let raw=raw.map_err(E::from)?;
        add_since(C::store_decompress_ns, decompress_started); add(C::store_decompressed_bytes, raw.len() as u64);
        let raw=Arc::new(StoredChunk::new(raw,memory));
        let bytes=raw.doc_bytes(within).map_err(E::from)?;
        let compact=store.chunk_compact(chunk);
        before(peak.saturating_add(Document::decode_peak_bytes_as(bytes.len() as u64,compact)))?;
        let mut document: Document = crate::codec::store::decode_doc(bytes, compact).map_err(E::from)?;
        self.restore_column_stored(doc, &mut document);
        let doc = Some(document);
        {
            let mut partial=self.store_partial.lock();
            partial.retain(|&c|c!=chunk);
            if !revisit && within+1<store.chunk_docs(chunk).1 as usize {
                if partial.len()>=Self::STORE_PARTIAL_CHUNKS {partial.remove(0);}
                partial.push(chunk);
            }
        }
        if raw._memory.is_none(){return Ok(doc);}
        let mut cache=self.store_chunks.lock();
        match cache.iter_mut().find(|(c,_)|*c==chunk) {
            Some(entry) if entry.1.len()<raw.len()=>entry.1=raw,
            Some(_)=>{},
            None=>{if cache.len()>=Self::STORE_CHUNK_CACHE {cache.remove(0);}cache.push((chunk,raw));},
        }
        Ok(doc)
    }

    /// Decompressed stored-field chunks kept per segment.
    const STORE_CHUNK_CACHE: usize = 2;
    /// Prefix-decoded chunks remembered per segment (see `store_partial`).
    const STORE_PARTIAL_CHUNKS: usize = 16;

    /// The stored values of `field` in `doc` (see [`Document::stored_values`]).
    pub fn stored_values(&self, field: u16, doc_id: DocId) -> Result<crate::types::StoredValues> {
        crate::types::StoredValues::from_document(self.get_doc(doc_id)?, field)
    }

    /// Stored-fields reader (chunk table parsed once per segment).
    pub fn store(&self) -> Result<StoreReader<'_>> {
        self.store_with_budget(&crate::admission::shared())
    }

    fn store_with_budget(&self, budget: &Arc<crate::admission::Budget>) -> Result<StoreReader<'_>> {
        let data = self.store_data.as_slice();
        if let Some(t) = self.store_table.get() {
            return Ok(StoreReader::with_table(data, t));
        }
        let memory = budget.reserve(StoreTable::allocation_bound(data)?, "stored table cache")?;
        let table = StoreTable::parse(data)?;
        // Losing initializers release their table and reservation together.
        // The installed owner stays alive for every reader borrowing it.
        let admitted = crate::admission::Allocation::new(table, memory);
        Ok(StoreReader::with_table(data, self.store_table.get_or_init(|| admitted)))
    }

    /// The `doc -> PointId` map of every vector field.
    pub fn point_map(&self) -> PointMapReader<'_> {
        let data = self.vectors_data.as_slice();
        let directory = self.point_directory.get_or_init(|| {
            PointMapReader::open(data).map(PointMapReader::into_directory).unwrap_or_default()
        });
        PointMapReader::from_directory(data, directory)
    }

    /// Alias of [`Segment::point_map`].
    pub fn vectors(&self) -> PointMapReader<'_> {
        self.point_map()
    }

    /// The vector store this segment is attached to.
    pub fn vector_store(&self) -> Option<&Arc<VectorStore>> {
        self.vector_store.get()
    }

    /// Install the vector store (first call wins; called by `attach`).
    pub(crate) fn set_vector_store(&self, store: Arc<VectorStore>) {
        let _ = self.vector_store.set(store);
    }

    /// The point id of `doc`'s vector in `field`.
    pub fn point(&self, field: u16, doc: DocId) -> Option<u64> {
        self.point_map().get(field, doc)
    }

    /// The vector of `doc` in `field`, if any.
    pub fn vector(&self, field: u16, doc: DocId) -> Option<Vec<f32>> {
        self.with_vector(field, doc, <[f32]>::to_vec)
    }

    pub fn with_vector<T>(&self, field: u16, doc: DocId, f: impl FnOnce(&[f32]) -> T) -> Option<T> {
        self.vector_store.get()?.with_vector(self.point(field, doc)?, f)
    }

    pub fn visit_vectors(&self, field: u16, docs: impl IntoIterator<Item = DocId>, visit: impl FnMut(DocId, &[f32]) -> bool) -> bool {
        let Some(store) = self.vector_store.get() else { return false };
        let Some(column) = self.point_map().column(field) else { return false };
        store.visit_vectors(docs.into_iter().map(|doc| (doc, column.get(doc))), visit)
    }

    pub fn visit_vectors_with_norm(&self, field: u16, docs: impl IntoIterator<Item = DocId>, visit: impl FnMut(DocId, &[f32], f32) -> bool) -> bool {
        let Some(store) = self.vector_store.get() else { return false };
        let Some(column) = self.point_map().column(field) else { return false };
        store.visit_vectors_with_norm(docs.into_iter().map(|doc| (doc, column.get(doc))), visit)
    }

    /// k-NN over `field` restricted to this segment's live docs accepted by
    /// `accept`, through the vector store: the accepted docs are mapped to
    /// their point ids (the pre-filter) and the store's planner picks exact
    /// or filtered-HNSW search per vector segment. `ef` is the HNSW beam.
    pub fn knn(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        accept: impl Fn(DocId) -> bool,
    ) -> Result<Vec<(DocId, f32)>> {
        self.knn_with(field, query, k, ef, accept, &PlannerConfig::default())
    }

    /// Exact k-NN by brute force (recall checks).
    pub fn knn_exact(&self, field: u16, query: &[f32], k: usize, accept: impl Fn(DocId) -> bool) -> Result<Vec<(DocId, f32)>> {
        self.knn_with(field, query, k, k, accept, &PlannerConfig::exact())
    }

    /// [`Segment::knn`] with an explicit planner configuration.
    pub fn knn_with(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        accept: impl Fn(DocId) -> bool,
        cfg: &PlannerConfig,
    ) -> Result<Vec<(DocId, f32)>> {
        let Some(store) = self.vector_store.get() else {
            return Ok(Vec::new());
        };
        let dels = self.deletions();
        let points: Vec<u64> = self
            .point_map()
            .iter(field)
            .filter(|(d, _)| !dels.get(*d) && accept(*d))
            .map(|(_, p)| p)
            .collect();
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let (hits, _) = store.search(field, query, k, ef.max(k), Some(&points), cfg)?;
        Ok(hits.into_iter().filter(|h| h.segment == self.id).map(|h| (h.doc, h.score)).collect())
    }

    /// Fast-path k-NN when there are no deletions and no filter: skips the
    /// O(n) point-map scan entirely and passes `None` to the store, so the
    /// HNSW traversal visits all graph nodes without a per-node filter check.
    pub fn knn_unfiltered(
        &self,
        field: u16,
        query: &[f32],
        k: usize,
        ef: usize,
        cfg: &PlannerConfig,
    ) -> Result<Vec<(DocId, f32)>> {
        let Some(store) = self.vector_store.get() else {
            return Ok(Vec::new());
        };
        let (hits, _) = store.search(field, query, k, ef.max(k), None, cfg)?;
        Ok(hits.into_iter().filter(|h| h.segment == self.id).map(|h| (h.doc, h.score)).collect())
    }
}

/// The shared unit norms (fields without norms), allocated once.
fn unit_norms() -> Arc<Norms> {
    static UNIT: OnceLock<Arc<Norms>> = OnceLock::new();
    UNIT.get_or_init(|| Arc::new(Norms::unit())).clone()
}

/// A mutable, *searchable* in-memory segment (the memtable).
///
/// Documents are appended in doc-id order. Terms are interned into a
/// [`MemPostings`] (key arena + hash table of term ids; per-term
/// delta/vint doc+freq and position streams in one slice pool, Lucene
/// `TermsHash` style), fast fields go to columnar builders, and stored
/// fields are bincode-serialized straight into one byte buffer (the store
/// format's payload, LZ4-chunked at flush). Unlike Lucene's
/// `DocumentsWriter`, this run is queryable in place:
/// [`SegmentWriter::postings`] decodes one term's streams and encodes them
/// on the fly into the same [`PostingsReader`] the on-disk segments use,
/// so every scorer works on unflushed documents and a refresh never has to
/// write a file. Writing a file happens only on `write_to` (flush / size
/// threshold); the flush encodes postings, fast fields and the store in
/// parallel.
///
/// Concurrency: the owner serialises writers; readers that hold `&self`
/// see a consistent prefix of the run because every structure is
/// append-only and `num_docs` is bumped last.
pub struct SegmentWriter {
    pub id: u32,
    num_docs: u32,
    /// The inverted index, keyed by `Term::key()` bytes.
    postings: MemPostings,
    fast_fields: FastFields,
    /// Stored documents, serialized straight into the store format's
    /// uncompressed chunk payloads (`vint len, bincode(Document)` per doc,
    /// chunks framed by [`crate::codec::store::StoredCompression::chunk_full`]);
    /// the flush only compresses them.
    store_chunks: Vec<Vec<u8>>,
    /// First doc of every chunk in `store_chunks`.
    store_starts: Vec<u32>,
    /// Per chunk of `store_chunks`: its docs are compact-encoded (decided
    /// when the chunk opens, from the format then written).
    store_compact: Vec<bool>,
    /// Docs of the open store chunk over [`crate::codec::store::SIZED_DOC_BYTES`].
    store_sized: usize,
    /// Some stored doc holds a binary id ([`crate::uid`]; written from
    /// format [`crate::codec::UID_STORED_SINCE`]): a seal below that
    /// format re-encodes the compact chunks.
    store_uid: bool,
    /// How stored chunks are framed (as docs arrive) and compressed (at
    /// seal): the index's `index.codec` ([`SegmentWriter::set_stored_compression`]).
    store_compression: crate::codec::store::StoredCompression,
    /// Per doc: (offset of its bincode bytes in its chunk, length).
    store_locs: Vec<(u32, u32)>,
    /// Identity (pointer, length) of the schema the caches below describe.
    schema_key: (usize, usize),
    /// Schema index by field id (`NO_SLOT` when absent).
    slot_of: Vec<u32>,
    /// Schema fields marked [`Field::column_stored`] (i64 columns).
    column_stored: Vec<u16>,
    /// The schema's synthetic id field and its columns
    /// ([`crate::synthetic_id`]): its terms stay in memory, sealed segments
    /// keep none, and stored documents get it back from the columns.
    synthetic_id: Option<(u16, crate::synthetic_id::SyntheticId)>,
    /// Analyzers by field id, resolved from the schema (shared, never cloned
    /// per document).
    analyzers: Vec<Option<Arc<crate::analysis::Analyzer>>>,
    default_analyzer: Arc<crate::analysis::Analyzer>,
    /// Reusable term key buffer (field id + term bytes).
    key_buf: Vec<u8>,
    /// Per-doc scratch: token count per text field, value count per fast field.
    field_lens: Vec<(u16, u32)>,
    fast_counts: Vec<(u16, u32)>,
    /// Per-doc scratch: first position of the next value of a text field
    /// (multi-valued text keeps increasing positions plus the gap).
    pos_next: Vec<(u16, u32)>,
    /// Deletes of docs still in memory (updates before a flush), and what
    /// searches see while some delete is pending (see
    /// [`SegmentWriter::delete_doc_pending`]). Behind a lock so a memtable
    /// shared with a background flush (`Arc`) still takes deletes.
    dels: RwLock<MemDels>,
    /// Heap use not covered by the structures above (vectors).
    bytes: usize,
    /// Dense vectors by field (exact-searchable until flush, when they move
    /// to the vector store).
    vectors: BTreeMap<u16, VectorFieldBuilder>,
    /// Ingestion reservations, retained across text encoding retries. Offsets
    /// address vector rows, never mutable/renumbered text segment identities.
    vector_reservations: BTreeMap<u16, Vec<(usize, crate::vector::store::PointReservation)>>,
    /// Row ranges admitted while their field had no graph (see
    /// [`PreparedDocumentVectors`]). Every other row of a field with a graph
    /// must hold a reservation: a writer row without either is unstaged.
    #[cfg(feature = "mutable-hnsw-poc")]
    graphless_vector_rows: BTreeMap<u16, Vec<std::ops::Range<usize>>>,
    #[cfg(feature = "mutable-hnsw-poc")]
    mutable_store: Option<Arc<VectorStore>>,
    // Reservations only append. Cache the verified prefix per retained owner.
    #[cfg(feature = "mutable-hnsw-poc")]
    verified_reservations: parking_lot::Mutex<BTreeMap<u16, (Arc<crate::vector::mutable_ingest::MutableIngestField>, usize)>>,
    /// Fast fields (by id) promoted to multi-valued columns (a doc supplied
    /// a second value). Promoted fields live in the multi columns of
    /// `fast_fields` only.
    multi_fields: Vec<bool>,
    /// Nested path ordinals (created by the first nested doc).
    nested: Option<NestedInfo>,
    /// Geo points per geo-point field.
    geo: BTreeMap<u16, GeoValues>,
    /// Triangles and doc values per `geo_shape` field ([`Field::shape`]).
    shapes: BTreeMap<u16, crate::fieldtypes::shape_field::ShapeBuf>,
    /// Collection statistics per field id (text / keyword fields), updated
    /// as docs are added so a query never scans the postings.
    aggs: Vec<FieldAgg>,
    /// Per-doc scratch: tokens (overlaps included) per text field.
    agg_doc: Vec<(u16, u32)>,
    /// Per-doc scratch: every indexed keyword value (field, start, len into
    /// `kw_buf`), to count distinct values per field.
    kw_seen: Vec<(u16, u32, u32)>,
    kw_buf: Vec<u8>,
    /// This doc's feature postings `(field, token start, len, tf)` over
    /// `feat_buf`, posted once per token with its largest tf.
    feat_seen: Vec<(u16, u32, u32, u32)>,
    feat_buf: Vec<u8>,
    /// Uninverted text fielddata per field and the doc count it covers
    /// (rebuilt once more docs were added; see [`SegmentWriter::uninverted`]).
    fielddata: parking_lot::Mutex<Vec<(u16, u32, Arc<crate::fielddata::Uninverted>)>>,
    /// Negative filter over the document ids (`_id`), kept as they are
    /// added: id lookups skip memtables that cannot hold the id.
    unique: UniqueIds,
    /// Encoded postings of large terms read by searches, keyed by term id
    /// with the doc freq they were encoded at (postings only grow, so an
    /// equal doc freq means the same list): a published memtable is read
    /// by many queries, and decoding and encoding a long list each time
    /// cost more than the query. Bounded by [`ENCODED_CACHE_BYTES`].
    encoded: parking_lot::Mutex<EncodedCache>,
    /// [`EncodedCache::bytes`], readable without its lock (counted in
    /// [`SegmentWriter::estimated_bytes`]).
    encoded_bytes: std::sync::atomic::AtomicUsize,
    /// Node-wide accounting of a closed memtable's memory, held until the
    /// last reference (a reader view, a seal job) drops it.
    closed_charge: crate::accounting::Charge,
}

/// [`SegmentWriter::unique`]: the filter of the schema's keyword `_id`
/// field (the ES layer's document id), maintained on every add (never
/// rebuilt from the term table). `bits` (12 per id, a power of two) is
/// regrown from `hashes` when the ids outgrow it.
#[derive(Default)]
struct UniqueIds {
    /// The id field (`None`: the schema has none; nothing is filtered).
    field: Option<u16>,
    /// [`crate::codec::terms::negative_filter_hash`] of every id added.
    hashes: Vec<u64>,
    bits: Vec<u64>,
    /// `true`: ids were added untracked (the field changed with documents
    /// present, or was indexed as analyzed text); never trusted.
    broken: bool,
}

impl UniqueIds {
    fn add(&mut self, hash: u64) {
        self.hashes.push(hash);
        if (self.bits.len() as u64) * 64 < self.hashes.len() as u64 * 12 {
            // Regrow for twice the ids (at least 4096).
            let words = (((self.hashes.len().max(2048) * 2) as u64 * 12).div_ceil(64).next_power_of_two()).min(8 * 1024 * 1024) as usize;
            self.bits = vec![0u64; words];
            for &h in &self.hashes {
                let (word, mask) = crate::codec::terms::negative_filter_slot(h, words);
                self.bits[word] |= mask;
            }
        } else {
            let (word, mask) = crate::codec::terms::negative_filter_slot(hash, self.bits.len());
            self.bits[word] |= mask;
        }
    }
}

const NO_SLOT: u32 = u32::MAX;

/// Count one more value of `field` in a per-doc scratch list; returns the
/// count before this one.
#[inline]
fn bump(list: &mut Vec<(u16, u32)>, field: u16, by: u32) -> u32 {
    for e in list.iter_mut() {
        if e.0 == field {
            let old = e.1;
            e.1 += by;
            return old;
        }
    }
    list.push((field, by));
    0
}

/// Terms with at least this many documents have their encoded postings
/// cached by [`SegmentWriter::postings`].
const ENCODED_CACHE_MIN_DOCS: u32 = 1024;
/// Most encoded bytes one memtable caches.
const ENCODED_CACHE_BYTES: usize = 16 << 20;

#[derive(Default)]
struct EncodedCache {
    by_term: std::collections::HashMap<u32, (u32, crate::codec::directory::FileSlice)>,
    bytes: usize,
    /// `bytes` on node-wide accounting (it grows after the memtable closed,
    /// so the closed charge leaves it out), released with the memtable.
    charge: crate::accounting::Charge,
}

impl SegmentWriter {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            num_docs: 0,
            postings: MemPostings::new(),
            fast_fields: FastFields::new(0),
            store_chunks: Vec::new(),
            store_starts: Vec::new(),
            store_compact: Vec::new(),
            store_sized: 0,
            store_uid: false,
            store_compression: crate::codec::store::StoredCompression::Fast,
            store_locs: Vec::new(),
            schema_key: (0, usize::MAX),
            slot_of: Vec::new(),
            column_stored: Vec::new(),
            synthetic_id: None,
            analyzers: Vec::new(),
            default_analyzer: Arc::new(crate::analysis::Analyzer::standard()),
            key_buf: Vec::with_capacity(64),
            field_lens: Vec::new(),
            fast_counts: Vec::new(),
            pos_next: Vec::new(),
            dels: RwLock::new(MemDels { latest: BitSet::new(0), visible: None, spare_visible: BitSet::new(0) }),
            bytes: 0,
            vectors: BTreeMap::new(),
            vector_reservations: BTreeMap::new(),
            #[cfg(feature = "mutable-hnsw-poc")]
            graphless_vector_rows: BTreeMap::new(),
            #[cfg(feature = "mutable-hnsw-poc")]
            mutable_store: None,
            #[cfg(feature = "mutable-hnsw-poc")]
            verified_reservations: parking_lot::Mutex::new(BTreeMap::new()),
            multi_fields: Vec::new(),
            nested: None,
            geo: BTreeMap::new(),
            shapes: BTreeMap::new(),
            aggs: Vec::new(),
            agg_doc: Vec::new(),
            kw_seen: Vec::new(),
            kw_buf: Vec::new(),
            feat_seen: Vec::new(),
            feat_buf: Vec::new(),
            fielddata: parking_lot::Mutex::new(Vec::new()),
            unique: UniqueIds::default(),
            encoded: parking_lot::Mutex::new(EncodedCache { charge: crate::accounting::Charge::new(crate::accounting::Category::ReadCaches, 0), ..Default::default() }),
            encoded_bytes: std::sync::atomic::AtomicUsize::new(0),
            closed_charge: crate::accounting::Charge::default(),
        }
    }

    /// Whether [`SegmentWriter::sorted_union`] can write `parts` (closed
    /// memtables of one index-sorted core) as one segment: no vectors,
    /// nested docs, geo points or shapes, one schema, one stored-chunk
    /// format and compression.
    pub(crate) fn sorted_union_supported(parts: &[&SegmentWriter]) -> bool {
        let Some(first) = parts.first() else { return false };
        let compact = first.store_compact.first().copied();
        parts.iter().all(|w| {
            w.vectors.is_empty() && w.vector_reservations.is_empty() && w.geo.is_empty() && w.shapes.is_empty()
                && w.nested.as_ref().is_none_or(|n| !n.has_nested())
                && w.schema_key == first.schema_key && w.store_compression == first.store_compression && w.synthetic_id == first.synthetic_id
                && w.store_compact.iter().all(|&c| Some(c) == compact)
                && w.store_starts.len() == w.store_chunks.len() && w.store_locs.len() == w.num_docs as usize
        })
    }

    /// The memtables `parts` as one memtable whose docs are `order`
    /// (`order[new] = (part, old)`, every doc of every part once): an
    /// index-sorted seal encodes it once, where writing each memtable and
    /// sort-merging the segments encoded everything twice and decoded it
    /// once. Postings are rebuilt per term in the new doc order, doc-value
    /// columns are filled in that order (multi-valued keyword values, and
    /// so feature columns, keep their per-doc value order; the encoder
    /// decides their on-disk layout as for any memtable), and stored docs
    /// are copied, re-chunked by the same rule. Deletions are not carried:
    /// the caller remaps each part's deletes at installation. `None` when
    /// unsupported ([`SegmentWriter::sorted_union_supported`]) or when a
    /// term is posted differently across parts.
    pub(crate) fn sorted_union(parts: &[&SegmentWriter], order: &[(usize, DocId)], id: u32) -> Option<SegmentWriter> {
        if !Self::sorted_union_supported(parts) || order.len() != parts.iter().map(|w| w.num_docs as usize).sum::<usize>() {
            return None;
        }
        let num_docs = order.len() as u32;
        let mut remap: Vec<Vec<DocId>> = parts.iter().map(|w| vec![DocId::MAX; w.num_docs as usize]).collect();
        for (new, &(p, old)) in order.iter().enumerate() {
            *remap.get_mut(p)?.get_mut(old as usize)? = new as DocId;
        }
        if remap.iter().any(|m| m.contains(&DocId::MAX)) {
            return None;
        }
        let first = parts[0];
        let ((postings, fast_fields), (store, aggs)) = rayon::join(
            || rayon::join(|| { let t = std::time::Instant::now(); let r = Self::union_postings(parts, &remap); tracing::debug!(target: "seal_profile", ms = t.elapsed().as_millis() as u64, "union postings"); r },
                || { let t = std::time::Instant::now(); let r = Self::union_fast_fields(parts, order, num_docs); tracing::debug!(target: "seal_profile", ms = t.elapsed().as_millis() as u64, "union fast fields"); r }),
            || rayon::join(|| Self::union_store(parts, order), || {
                let mut aggs: Vec<FieldAgg> = Vec::new();
                for w in parts {
                    if aggs.len() < w.aggs.len() { aggs.resize_with(w.aggs.len(), FieldAgg::default); }
                    for (a, b) in aggs.iter_mut().zip(&w.aggs) { a.doc_count += b.doc_count; a.sum_ttf += b.sum_ttf; }
                }
                aggs
            }),
        );
        let postings = postings?;
        let (store_chunks, store_starts, store_compact, store_locs) = store;
        let mut multi_fields: Vec<bool> = Vec::new();
        for w in parts {
            if multi_fields.len() < w.multi_fields.len() { multi_fields.resize(w.multi_fields.len(), false); }
            for (a, &b) in multi_fields.iter_mut().zip(&w.multi_fields) { *a |= b; }
        }
        let mut out = SegmentWriter::new(id);
        out.num_docs = num_docs;
        out.postings = postings;
        out.fast_fields = fast_fields;
        out.store_chunks = store_chunks;
        out.store_starts = store_starts;
        out.store_compact = store_compact;
        out.store_uid = parts.iter().any(|w| w.store_uid);
        out.store_compression = first.store_compression;
        out.store_locs = store_locs;
        out.schema_key = first.schema_key;
        out.slot_of = first.slot_of.clone();
        out.column_stored = first.column_stored.clone();
        out.synthetic_id = first.synthetic_id;
        out.analyzers = first.analyzers.clone();
        out.default_analyzer = first.default_analyzer.clone();
        out.multi_fields = multi_fields;
        out.aggs = aggs;
        Some(out)
    }

    /// [`SegmentWriter::sorted_union`]'s postings: every term of every
    /// part, its docs remapped and ascending.
    fn union_postings(parts: &[&SegmentWriter], remap: &[Vec<DocId>]) -> Option<MemPostings> {
        let t0 = std::time::Instant::now();
        let mut out = MemPostings::new();
        // Per output term: its kind and its (part, term) sources.
        let mut kinds: Vec<(bool, bool)> = Vec::new();
        let mut sources: Vec<Vec<(u32, u32)>> = Vec::new();
        for (p, w) in parts.iter().enumerate() {
            for t in 0..w.postings.len() as u32 {
                let kind = w.postings.term_kind(t);
                let g = out.intern(w.postings.key(t), kind.0) as usize;
                if g == sources.len() {
                    kinds.push(kind);
                    sources.push(Vec::new());
                } else if kinds[g].0 != kind.0 || (kinds[g].1 != kind.1 && w.postings.doc_freq(t) > 0) {
                    return None;
                }
                sources[g].push((p as u32, t));
            }
        }
        // Decoding, remapping and sorting each term's docs is the work:
        // done in parallel, a wave of terms at a time (bounded memory), then
        // appended in term order (the streams only append).
        use rayon::prelude::*;
        // (new doc, tf, start in the term's positions)
        type Rebuilt = (Vec<(DocId, u32, u32)>, Vec<u32>);
        let num_docs: usize = remap.iter().map(Vec::len).sum();
        let rebuild = |(g, srcs): (usize, &Vec<(u32, u32)>)| -> Rebuilt {
            let mut decoded = crate::postings::DecodedPostings::default();
            let (has_positions, freqs) = kinds[g];
            let df: usize = srcs.iter().map(|&(p, t)| parts[p as usize].postings.doc_freq(t) as usize).sum();
            if !has_positions && !freqs && srcs.len() > 1 && df * 16 >= num_docs {
                // A common term (tf 1, no positions): its new docs marked in
                // a bitset and read back ascending, instead of sorted.
                let mut bits = vec![0u64; num_docs.div_ceil(64)];
                for &(p, t) in srcs {
                    let map = &remap[p as usize];
                    parts[p as usize].postings.decode(t, &mut decoded);
                    for &doc in &decoded.doc_ids {
                        let new = map[doc as usize] as usize;
                        bits[new / 64] |= 1 << (new % 64);
                    }
                }
                let mut docs = Vec::with_capacity(df);
                for (w, &word) in bits.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        docs.push(((w * 64) as DocId + word.trailing_zeros(), 1, 0));
                        word &= word - 1;
                    }
                }
                return (docs, Vec::new());
            }
            let mut docs: Vec<(DocId, u32, u32)> = Vec::new();
            let mut positions: Vec<u32> = Vec::new();
            for &(p, t) in srcs {
                let map = &remap[p as usize];
                parts[p as usize].postings.decode(t, &mut decoded);
                let mut at = 0usize;
                for (i, &doc) in decoded.doc_ids.iter().enumerate() {
                    let tf = decoded.term_freqs[i];
                    docs.push((map[doc as usize], tf, positions.len() as u32));
                    if decoded.has_positions {
                        positions.extend_from_slice(&decoded.positions[at..at + tf as usize]);
                        at += tf as usize;
                    }
                }
            }
            if srcs.len() > 1 || docs.windows(2).any(|w| w[0].0 > w[1].0) {
                docs.sort_unstable_by_key(|d| d.0);
            }
            (docs, positions)
        };
        const WAVE: usize = 8192;
        let interned = t0.elapsed();
        let mut rebuilding = std::time::Duration::ZERO;
        for (w, wave) in sources.chunks(WAVE).enumerate() {
            let tr = std::time::Instant::now();
            let rebuilt: Vec<Rebuilt> = wave.par_iter().enumerate().with_min_len(64).map(|(k, srcs)| rebuild((w * WAVE + k, srcs))).collect();
            rebuilding += tr.elapsed();
            for (k, (docs, positions)) in rebuilt.iter().enumerate() {
                let g = w * WAVE + k;
                let (has_positions, freqs) = kinds[g];
                for &(doc, tf, start) in docs {
                    let pos: &[u32] = if has_positions { &positions[start as usize..start as usize + tf as usize] } else { &[] };
                    out.append_rebuilt(g as u32, freqs, doc, tf, pos);
                }
            }
        }
        tracing::debug!(target: "seal_profile", terms = sources.len(), intern_ms = interned.as_millis() as u64, rebuild_ms = rebuilding.as_millis() as u64,
            total_ms = t0.elapsed().as_millis() as u64, "union postings phases");
        Some(out)
    }

    /// [`SegmentWriter::sorted_union`]'s doc-value columns, filled in the
    /// new doc order. A field multi-valued in any part is written to its
    /// multi-valued column only (as a memtable promotes it).
    fn union_fast_fields(parts: &[&SegmentWriter], order: &[(usize, DocId)], num_docs: u32) -> FastFields {
        let mut out = FastFields::new(0);
        let ffs: Vec<&FastFields> = parts.iter().map(|w| &w.fast_fields).collect();
        let mut fields: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        for ff in &ffs {
            fields.extend(ff.i64_columns().keys().chain(ff.u64_columns().keys()).chain(ff.f64_columns().keys())
                .chain(ff.bool_columns().keys()).chain(ff.keyword_columns().keys()).chain(ff.norms().keys())
                .chain(ff.multi_i64_columns().keys()).chain(ff.multi_u64_columns().keys()).chain(ff.multi_f64_columns().keys())
                .chain(ff.multi_keyword_columns().keys()).copied());
        }
        use rayon::prelude::*;
        let fields: Vec<u16> = fields.into_iter().collect();
        let per_field: Vec<FastFields> = fields.par_iter().map(|&field| {
            let mut out = FastFields::new(0);
            if ffs.iter().any(|ff| !ff.has_points(field)) { out.skip_points(field); }
            if ffs.iter().any(|ff| ff.is_from_postings(field)) { out.derive_from_postings(field); }
            macro_rules! numeric {
                ($get:ident, $get_multi:ident, $single:ident, $multi:ident) => {{
                    let cols: Vec<_> = ffs.iter().map(|ff| (ff.$get(field), ff.$get_multi(field))).collect();
                    if cols.iter().any(|c| c.0.is_some() || c.1.is_some()) {
                        if cols.iter().any(|c| c.1.is_some()) {
                            let mut target = None;
                            for (new, &(p, old)) in order.iter().enumerate() {
                                let (single, multi) = cols[p];
                                if let Some(v) = single.and_then(|c| c.get(old)) { target.get_or_insert_with(Vec::new).push((new as DocId, v)); }
                                if let Some(c) = multi { for &v in c.values(old) { target.get_or_insert_with(Vec::new).push((new as DocId, v)); } }
                            }
                            if let Some(values) = target {
                                let col = out.$multi(field);
                                for (doc, v) in values { col.push(doc, v); }
                            }
                        } else {
                            let mut col = None;
                            for (new, &(p, old)) in order.iter().enumerate() {
                                if let Some(v) = cols[p].0.and_then(|c| c.get(old)) {
                                    col.get_or_insert_with(Vec::new).push((new as DocId, v));
                                }
                            }
                            if let Some(values) = col {
                                let c = out.$single(field);
                                for (doc, v) in values { c.set(doc, v); }
                            }
                        }
                    }
                }};
            }
            numeric!(get_i64, get_multi_i64, i64_column_mut, multi_i64_column_mut);
            numeric!(get_u64, get_multi_u64, u64_column_mut, multi_u64_column_mut);
            numeric!(get_f64, get_multi_f64, f64_column_mut, multi_f64_column_mut);
            let bools: Vec<_> = ffs.iter().map(|ff| ff.get_bool(field)).collect();
            if bools.iter().any(Option::is_some) {
                let col = out.bool_column_mut(field);
                if bools.iter().flatten().any(|c| !c.presence_known()) {
                    col.set_presence_unknown();
                }
                for (new, &(p, old)) in order.iter().enumerate() {
                    if let Some(c) = bools[p] { if c.has_value(old) { col.set(new as DocId, c.get(old)); } }
                }
            }
            let kw: Vec<_> = ffs.iter().map(|ff| (ff.get_keyword(field), ff.get_multi_keyword(field))).collect();
            if kw.iter().any(|c| c.1.is_some()) {
                // Each doc's values in their (ascending) order: appended at
                // the end of the doc, never shifted.
                let mut values: Vec<(DocId, &str)> = Vec::new();
                for (new, &(p, old)) in order.iter().enumerate() {
                    let (single, multi) = kw[p];
                    if let Some(v) = single.and_then(|c| c.get(old)) { values.push((new as DocId, v)); }
                    if let Some(c) = multi { values.extend(c.values(old).into_iter().map(|v| (new as DocId, v))); }
                }
                if !values.is_empty() {
                    let col = out.multi_keyword_column_mut(field);
                    for (doc, v) in values { col.push(doc, v); }
                }
            } else if kw.iter().any(|c| c.0.is_some()) {
                let values: Vec<(DocId, &str)> = order.iter().enumerate()
                    .filter_map(|(new, &(p, old))| kw[p].0.and_then(|c| c.get(old)).map(|v| (new as DocId, v))).collect();
                if !values.is_empty() {
                    let col = out.keyword_column_mut(field);
                    for (doc, v) in values { col.set(doc, v); }
                }
            }
            let norms: Vec<_> = ffs.iter().map(|ff| ff.norms().get(&field)).collect();
            if norms.iter().any(Option::is_some) {
                let col = out.norms_mut(field);
                for (new, &(p, old)) in order.iter().enumerate() {
                    if let Some(len) = norms[p].and_then(|c| c.lengths().get(old as usize)).copied().filter(|&l| l > 0) {
                        col.set(new as DocId, len);
                    }
                }
            }
            out
        }).collect();
        for part in per_field { out.absorb(part); }
        out.set_num_docs(num_docs);
        out
    }

    /// [`SegmentWriter::sorted_union`]'s stored docs: each doc's bytes
    /// copied in the new order, chunked by the chunk rule.
    #[allow(clippy::type_complexity)]
    fn union_store(parts: &[&SegmentWriter], order: &[(usize, DocId)]) -> (Vec<Vec<u8>>, Vec<u32>, Vec<bool>, Vec<(u32, u32)>) {
        let first = parts[0];
        let compact = first.store_compact.first().copied().unwrap_or(crate::codec::write_version() >= crate::codec::COMPACT_STORED_SINCE);
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        let mut starts: Vec<u32> = Vec::new();
        let mut locs: Vec<(u32, u32)> = Vec::with_capacity(order.len());
        let mut sized = 0usize;
        // Small chunks for large docs only where fetches want them.
        let small_chunks = parts.iter().any(|w| !w.vectors.is_empty());
        for (new, &(p, old)) in order.iter().enumerate() {
            let w = parts[p];
            let chunk = w.store_starts.partition_point(|&s| s <= old) - 1;
            let (off, len) = w.store_locs[old as usize];
            let bytes = &w.store_chunks[chunk][off as usize..off as usize + len as usize];
            let full = match (chunks.last(), starts.last()) {
                (Some(c), Some(&start)) => first.store_compression.chunk_full_for(new - start as usize, sized, c.len(), small_chunks),
                _ => true,
            };
            if full {
                sized = 0;
                let cap = chunks.last().map(|c| c.len() + c.len() / 8).unwrap_or(16 << 10);
                chunks.push(Vec::with_capacity(cap));
                starts.push(new as DocId);
            }
            let out = chunks.last_mut().expect("a chunk is open");
            write_vint(len as u64, out);
            locs.push((out.len() as u32, len));
            out.extend_from_slice(bytes);
            if len as usize > crate::codec::store::SIZED_DOC_BYTES {
                sized += 1;
            }
        }
        let flags = vec![compact; chunks.len()];
        (chunks, starts, flags, locs)
    }

    /// Account this memtable as closed: it takes no more documents, so its
    /// memory is charged once, until it is dropped.
    pub(crate) fn account_closed(&mut self) {
        // Read caches carry their own charge (they grow after closing too).
        self.closed_charge = crate::accounting::Charge::new(crate::accounting::Category::Memtables, self.indexing_bytes() as u64);
    }

    /// [`SegmentWriter::estimated_bytes`] without the read caches searches
    /// build over the memtable (ES-scoring codes, encoded postings), which
    /// are accounted on their own.
    /// No lock is taken (the read caches are not read).
    pub fn indexing_bytes(&self) -> usize {
        self.postings.heap_bytes()
            + self.store_chunks.iter().map(|c| c.capacity()).sum::<usize>()
            + self.store_locs.len() * 8
            + self.store_starts.capacity() * 4
            + self.num_docs as usize * 48
            + self.bytes
            + self.geo.values().map(|g| g.heap_bytes()).sum::<usize>()
            + self.shapes.values().map(|g| g.heap_bytes()).sum::<usize>()
    }

    /// The memtable's triangles and doc values of `geo_shape` field `field`.
    pub fn shape_buf(&self, field: u16) -> Option<&crate::fieldtypes::shape_field::ShapeBuf> {
        self.shapes.get(&field)
    }

    /// Whether `term` may be in this memtable (`false`: certainly not).
    pub fn may_contain_unique_term(&self, term: &Term) -> bool {
        self.may_contain_unique_hash(term.field, crate::codec::terms::negative_filter_hash(&term.text))
    }

    /// [`SegmentWriter::may_contain_unique_term`] by the term's hash
    /// ([`crate::codec::terms::negative_filter_hash`], computed once for
    /// every leaf): one word of the memtable's id filter; `true` for any
    /// other field.
    #[inline]
    pub fn may_contain_unique_hash(&self, field: u16, hash: u64) -> bool {
        let u = &self.unique;
        if u.field != Some(field) || u.broken {
            return true;
        }
        if u.bits.is_empty() {
            return false;
        }
        let (word, mask) = crate::codec::terms::negative_filter_slot(hash, u.bits.len());
        u.bits[word] & mask == mask
    }

    /// [`Segment::uninverted`] over the docs added so far. Docs are only
    /// ever appended (an update deletes and re-adds), so the cached table
    /// stays valid until the doc count changes.
    pub fn uninverted(&self, field: u16) -> Result<Arc<crate::fielddata::Uninverted>> {
        let n=self.num_docs;let mut cache=self.fielddata.lock();
        if let Some((_,_,value))=cache.iter().find(|(f,at,_)|*f==field&&*at==n){return Ok(value.clone());}
        let mut scratch=crate::admission::shared().reserve((self.postings.len() as u64).saturating_mul(8).saturating_add(512),"fielddata term ids")?;
        let prefix=field.to_le_bytes();
        let mut ids:Vec<u32>=(0..self.postings.len() as u32).filter(|&id|self.postings.key(id).starts_with(&prefix)).collect();
        ids.sort_unstable_by(|&a,&b|self.postings.key(a).cmp(self.postings.key(b)));
        let mut builder=crate::fielddata::UninvertedBuilder::new(n,0)?;
        let mut decoded=DecodedPostings::default();let mut decode_budget=0;
        for id in ids {
            let key=self.postings.key(id);let ord=builder.add_term(&key[2..])?;
            let needed=(self.postings.doc_freq(id) as u64).saturating_mul(96).saturating_add(self.postings.total_term_freq(id).saturating_mul(8));
            if needed>decode_budget {scratch.grow(needed-decode_budget,"fielddata postings decode")?;decode_budget=needed;}
            self.postings.decode(id,&mut decoded);
            for &doc in &decoded.doc_ids {builder.add_doc(ord,doc)?;}
        }
        let value=Arc::new(builder.finish());cache.retain(|(f,_,_)|*f!=field);cache.push((field,n,value.clone()));Ok(value)
    }

    /// Collection statistics of `field` over the docs added so far (deleted
    /// ones included, as Lucene counts them until a merge); `None` when no
    /// doc has a term in it.
    pub fn field_agg(&self, field: u16) -> Option<FieldAgg> {
        self.aggs.get(field as usize).copied().filter(|a| a.doc_count > 0)
    }

    fn add_agg(&mut self, field: u16, tokens: u32) {
        let f = field as usize;
        if self.aggs.len() <= f {
            self.aggs.resize(f + 1, FieldAgg::default());
        }
        let a = &mut self.aggs[f];
        a.doc_count += 1;
        a.sum_ttf += tokens as u64;
    }

    /// Nested path ordinals, once a nested doc was added.
    pub fn nested_info(&self) -> Option<&NestedInfo> {
        self.nested.as_ref()
    }

    /// The nested (non-root) docs, if any (a fresh bitset per call).
    pub fn child_docs(&self) -> Option<BitSet> {
        self.nested.as_ref()?.child_bits(self.num_docs)
    }

    /// The in-memory geo points of a geo-point field.
    pub fn geo_values(&self, field: u16) -> Option<&GeoValues> {
        self.geo.get(&field)
    }

    /// Docs with a point of `field` in `shape` (linear scan).
    pub fn geo_query(&self, field: u16, shape: &GeoShape) -> Vec<DocId> {
        self.geo.get(&field).map(|g| g.query(shape)).unwrap_or_default()
    }

    /// Brute-force k-NN over the in-memory vectors of `field`.
    pub fn knn(&self, field: u16, query: &[f32], k: usize, accept: impl Fn(DocId) -> bool) -> Vec<(DocId, f32)> {
        self.knn_with(field, query, k, accept, &PlannerConfig::default())
    }

    /// [`SegmentWriter::knn`] under a planner config: a quantized field ES
    /// would not rescore scores with ES's quantized estimates (see
    /// [`crate::vector::es_quant`]), else exact `f32`.
    pub fn knn_with(&self, field: u16, query: &[f32], k: usize, accept: impl Fn(DocId) -> bool, cfg: &PlannerConfig) -> Vec<(DocId, f32)> {
        match self.vectors.get(&field) {
            Some(b) => match b.es_quant(cfg) {
                Some(spec) => b.knn_es(&spec, query, k, accept),
                None => b.knn(query, k, accept),
            },
            None => Vec::new(),
        }
    }

    /// Replace the vector policy the rows of `field` are searched with.
    pub fn set_vector_options(&mut self, field: u16, options: Option<crate::vector::store::VectorFieldOptions>) {
        if let Some(b) = self.vectors.get_mut(&field) {
            b.options = options;
        }
    }

    /// The vector of `doc` in `field`, if any.
    pub fn vector(&self, field: u16, doc: DocId) -> Option<&[f32]> {
        self.vectors.get(&field)?.get(doc)
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Whether any dense vector was added.
    pub fn has_vectors(&self) -> bool {
        !self.vectors.is_empty()
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn has_vector_field(&self, field: u16) -> bool {
        self.vectors.get(&field).is_some_and(|rows| rows.len() != 0)
    }

    /// Live docs as searches see them.
    pub fn num_live_docs(&self) -> u32 {
        self.num_docs - self.dels.read().visible().count()
    }

    /// Whether `doc_id` is deleted as searches see it.
    pub fn is_deleted(&self, doc_id: DocId) -> bool {
        let g = self.dels.read();
        let d = g.visible();
        doc_id < d.num_docs() && d.get(doc_id)
    }

    /// Whether `doc_id` is deleted, unpublished deletes included.
    pub fn is_deleted_latest(&self, doc_id: DocId) -> bool {
        let g = self.dels.read();
        doc_id < g.latest.num_docs() && g.latest.get(doc_id)
    }

    /// Mark an in-memory doc deleted; carried into the `.del` file on write.
    /// Deleting a document with nested children deletes its whole block.
    pub fn delete_doc(&self, doc_id: DocId) {
        if doc_id < self.num_docs {
            let start = self.nested.as_ref().map(|n| n.block_start(doc_id)).unwrap_or(doc_id);
            let mut g = self.dels.write();
            #[cfg(feature = "mutable-hnsw-poc")]
            let fresh: Vec<_> = (start..=doc_id).filter(|doc| !g.visible().get(*doc)).collect();
            for d in start..=doc_id {
                g.latest.set(d);
                if let Some(v) = g.visible.as_mut() {
                    v.set(d);
                }
            }
            drop(g);
            #[cfg(feature = "mutable-hnsw-poc")]
            for doc in fresh { self.queue_mutable_deleted_range(doc, doc); }
        }
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn queue_mutable_deleted_range(&self, start: DocId, end: DocId) {
        let Some(store) = &self.mutable_store else { return };
        for &field in self.vector_reservations.keys() {
            for doc in start..=end {
                if let Some(point) = self.reserved_vector_point(field, doc) { store.queue_mutable_delete(point); }
            }
        }
    }

    /// [`SegmentWriter::delete_doc`] for realtime lookups only: searches
    /// keep seeing the doc until [`SegmentWriter::publish_deletions`].
    pub fn delete_doc_pending(&self, doc_id: DocId) -> Result<()> {
        if doc_id < self.num_docs {
            let mut g = self.dels.write();
            if g.visible.is_none() {
                let MemDels { latest, spare_visible, .. } = &mut *g;
                latest.copy_into_prepared(spare_visible)?;
                g.visible = Some(std::mem::replace(&mut g.spare_visible, BitSet::new(0)));
            }
            let start = self.nested.as_ref().map(|n| n.block_start(doc_id)).unwrap_or(doc_id);
            for d in start..=doc_id { g.latest.set(d); }
        }
        Ok(())
    }

    /// Make pending deletes visible to searches.
    pub fn publish_deletions(&self) {
        let mut guard = self.dels.write();
        if let Some(old) = guard.visible.take() {
            #[cfg(feature = "mutable-hnsw-poc")]
            if let Some(store) = &self.mutable_store {
                for (wi, (&new, &previous)) in guard.latest.words().iter().zip(old.words()).enumerate() {
                    let mut changed = new & !previous;
                    while changed != 0 {
                        let doc = (wi * 64 + changed.trailing_zeros() as usize) as DocId;
                        changed &= changed - 1;
                        for &field in self.vector_reservations.keys() {
                            if let Some(point) = self.reserved_vector_point(field, doc) { store.queue_mutable_delete(point); }
                        }
                    }
                }
            }
            guard.spare_visible = old;
        }
    }

    /// Some delete is not yet visible to searches.
    pub fn has_pending_deletions(&self) -> bool {
        self.dels.read().visible.is_some()
    }

    /// Snapshot of the deletion bitmap searches see.
    pub fn deletions(&self) -> BitSet {
        BitSet::from_words(self.dels.read().visible().words().to_vec(), self.num_docs)
    }

    /// Snapshot of every delete, unpublished ones included.
    pub fn latest_deletions(&self) -> BitSet {
        BitSet::from_words(self.dels.read().latest.words().to_vec(), self.num_docs)
    }

    pub(crate) fn latest_deletions_admitted(&self) -> Result<BitSet> {
        self.dels.read().latest.try_clone_admitted()
    }

    pub(crate) fn visible_deletions_admitted(&self) -> Result<Option<BitSet>> {
        self.dels.read().visible.as_ref().map(BitSet::try_clone_admitted).transpose()
    }

    /// Rough memory footprint, used to decide when to flush.
    pub fn estimated_bytes(&self) -> usize {
        self.indexing_bytes()
            + self.vectors.values().map(|v| v.cached_bytes()).sum::<usize>()
            + self.encoded_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of distinct terms.
    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    // ---- searchable view -------------------------------------------------

    /// Document frequency of a term in this run.
    pub fn doc_freq(&self, term: &Term) -> u32 {
        self.postings.get(&term.key()).map(|id| self.postings.doc_freq(id)).unwrap_or(0)
    }

    /// Total term frequency of a term in this run.
    pub fn total_term_freq(&self, term: &Term) -> u64 {
        self.postings.get(&term.key()).map(|id| self.postings.total_term_freq(id)).unwrap_or(0)
    }

    /// Postings for a term, decoded from the memtable streams and encoded on
    /// the fly into the on-disk reader type (O(doc_freq); the run is
    /// size-bounded so this stays small).
    pub fn postings(&self, term: &Term) -> Option<PostingsReader> {
        let id = self.postings.get(&term.key())?;
        let doc_freq = self.postings.doc_freq(id);
        let cacheable = doc_freq >= ENCODED_CACHE_MIN_DOCS;
        if cacheable {
            if let Some((at, bytes)) = self.encoded.lock().by_term.get(&id) {
                if *at == doc_freq {
                    return PostingsReader::open(bytes.clone()).ok();
                }
            }
        }
        thread_local! {
            /// Decode scratch of memtable postings reads (one per searched
            /// term and memtable leaf: not three allocations each time).
            static DEC: std::cell::RefCell<DecodedPostings> = std::cell::RefCell::new(DecodedPostings::default());
        }
        let buf = DEC.with(|dec| {
            let mut dec = dec.borrow_mut();
            self.postings.decode(id, &mut dec);
            let mut buf = Vec::with_capacity(dec.doc_ids.len() * 3 + 16);
            let norms = match self.fast_fields.norms().get(&term.field) {
                Some(n) => BlockNorms::Bytes(n.bytes()),
                None => BlockNorms::Absent,
            };
            let positions = dec.has_positions.then_some(dec.positions.as_slice());
            write_postings_with_norms(&dec.doc_ids, &dec.term_freqs, positions, norms, &mut buf);
            // A huge list's scratch is not kept by the thread.
            if dec.doc_ids.capacity() > 1 << 18 || dec.positions.capacity() > 1 << 20 {
                *dec = DecodedPostings::default();
            }
            buf
        });
        if cacheable {
            let bytes = crate::codec::directory::FileSlice::from_vec(buf);
            let mut cache = self.encoded.lock();
            let len = bytes.len();
            let replaced = cache.by_term.get(&id).map_or(0, |(_, b)| b.len());
            if cache.bytes - replaced + len <= ENCODED_CACHE_BYTES {
                cache.bytes = cache.bytes - replaced + len;
                cache.by_term.insert(id, (doc_freq, bytes.clone()));
                let total = cache.bytes;
                cache.charge.resize(total as u64);
                self.encoded_bytes.store(total, std::sync::atomic::Ordering::Relaxed);
            }
            drop(cache);
            return PostingsReader::open(bytes).ok();
        }
        PostingsReader::from_bytes(buf).ok()
    }

    /// Terms of `field` with text in `[lower, upper)` (`None` = unbounded),
    /// with their statistics; `offset` is meaningless here.
    pub fn term_range(
        &self,
        field: u16,
        lower: Option<&[u8]>,
        upper: Option<&[u8]>,
    ) -> Vec<(Vec<u8>, PostingsLocation)> {
        use std::ops::Bound;
        let prefix = field.to_le_bytes();
        let mut lo = prefix.to_vec();
        if let Some(l) = lower {
            lo.extend_from_slice(l);
        }
        let hi = match upper {
            Some(u) => {
                let mut h = prefix.to_vec();
                h.extend_from_slice(u);
                Bound::Excluded(h)
            }
            None => match crate::types::Term::field_key_end(field) {
                Some(end) => Bound::Excluded(end),
                None => Bound::Unbounded,
            },
        };
        let in_range = |k: &[u8]| {
            k >= lo.as_slice()
                && match &hi {
                    Bound::Excluded(h) => k < h.as_slice(),
                    _ => true,
                }
        };
        let p = &self.postings;
        let mut out: Vec<(Vec<u8>, PostingsLocation)> = (0..p.len() as u32)
            .filter(|&id| in_range(p.key(id)))
            .map(|id| {
                (
                    p.key(id).to_vec(),
                    PostingsLocation {
                        offset: 0,
                        doc_freq: p.doc_freq(id),
                        total_term_freq: p.total_term_freq(id),
                    },
                )
            })
            .collect();
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// BM25 norms for a text field (unit norms if the field has none).
    pub fn norms_for(&self, field: u16) -> Norms {
        match self.fast_fields.norms().get(&field) {
            Some(n) => Norms::from_bytes(n.bytes().to_vec(), self.num_docs, n.total()),
            None => Norms::unit(),
        }
    }

    /// The in-memory fast-field columns (numeric/bool/keyword readers with
    /// `get`, `range_query`, `exact_match`; block indexes are built lazily
    /// by [`FastFields::build_all`], so range scans here are linear).
    pub fn fast_fields(&self) -> &FastFields {
        &self.fast_fields
    }

    /// Fetch a stored document (deserialized from the store buffer).
    pub fn get_doc(&self, doc_id: DocId) -> Result<Option<Document>> {
        let mut memory = crate::admission::shared().reserve(0, "stored document decode")?;
        let result = self.get_doc_guarded(doc_id, |bytes| {
            if bytes > memory.bytes() { memory.grow(bytes - memory.bytes(), "stored document decode")?; }
            Ok::<_, crate::RusticError>(())
        })?;
        // Retain what the decoded document holds (its capacities), not a
        // multiple of its encoded size.
        Ok(result.map(|mut document| { let retained = document.decoded_bytes().saturating_add(256); memory.shrink(memory.bytes().saturating_sub(retained)); document.decode_memory = Some(Arc::new(memory)); document }))
    }

    /// Reserve before the in-memory store decoder creates owned fields.
    pub fn get_doc_guarded<E: From<crate::RusticError>>(&self,doc:DocId,mut before:impl FnMut(u64)->std::result::Result<(),E>)->std::result::Result<Option<Document>,E> {
        let Some(&(off,len))=self.store_locs.get(doc as usize) else {return Ok(None)};
        let c=self.store_starts.partition_point(|&s|s<=doc)-1;
        let compact=self.store_compact.get(c).copied().unwrap_or(false);
        before(Document::decode_peak_bytes_as(len as u64,compact))?;
        let chunk=&self.store_chunks[c];
        let mut document: Document = crate::codec::store::decode_doc(&chunk[off as usize..(off+len) as usize], compact).map_err(E::from)?;
        if !self.column_stored.is_empty() {
            let ff = &self.fast_fields;
            restore_column_stored(&mut document, &self.column_stored, |f| ff.get_i64(f).and_then(|c| c.get(doc)));
        }
        if let Some((field, s)) = &self.synthetic_id {
            let ff = &self.fast_fields;
            restore_synthetic_id(&mut document, *field, || {
                Some((ff.get_keyword(s.tsid)?.get(doc)?.as_bytes().to_vec(), ff.get_i64(s.timestamp)?.get(doc)?, ff.get_i64(s.routing_hash)?.get(doc)?))
            });
        }
        Ok(Some(document))
    }

    /// The stored values of `field` in `doc_id` (see [`Document::stored_values`]).
    pub fn stored_values(&self, field: u16, doc_id: DocId) -> Result<crate::types::StoredValues> {
        crate::types::StoredValues::from_document(self.get_doc(doc_id)?, field)
    }

    #[inline]
    fn is_multi(&self, field: u16) -> bool {
        self.multi_fields.get(field as usize).copied().unwrap_or(false)
    }

    fn mark_multi(&mut self, field: u16) {
        let f = field as usize;
        if self.multi_fields.len() <= f {
            self.multi_fields.resize(f + 1, false);
        }
        self.multi_fields[f] = true;
    }

    /// Promote a single-valued fast field to a multi-valued column: every
    /// doc's one value becomes a one-element list, docs without a value an
    /// empty list, and the single column is dropped.
    ///
    /// Called mid-`add_document`, so the doc being added (`num_docs`) may
    /// already hold its first value and is included in the copy.
    fn promote_i64(&mut self, field: u16) {
        if let Some(col) = self.fast_fields.take_i64_column(field) {
            for d in 0..=self.num_docs {
                if col.has_value(d) {
                    if let Some(v) = col.get(d) {
                        self.fast_fields.multi_i64_column_mut(field).push(d, v);
                    }
                }
            }
        }
        self.mark_multi(field);
    }

    fn promote_u64(&mut self, field: u16) {
        if let Some(col) = self.fast_fields.take_u64_column(field) {
            for d in 0..=self.num_docs {
                if col.has_value(d) {
                    if let Some(v) = col.get(d) {
                        self.fast_fields.multi_u64_column_mut(field).push(d, v);
                    }
                }
            }
        }
        self.mark_multi(field);
    }

    fn promote_f64(&mut self, field: u16) {
        if let Some(col) = self.fast_fields.take_f64_column(field) {
            for d in 0..=self.num_docs {
                if col.has_value(d) {
                    if let Some(v) = col.get(d) {
                        self.fast_fields.multi_f64_column_mut(field).push(d, v);
                    }
                }
            }
        }
        self.mark_multi(field);
    }

    fn promote_keyword(&mut self, field: u16) {
        if let Some(col) = self.fast_fields.take_keyword_column(field) {
            for d in 0..=self.num_docs {
                if let Some(v) = col.get(d) {
                    self.fast_fields.multi_keyword_column_mut(field).push(d, v);
                }
            }
        }
        self.mark_multi(field);
    }

    /// Frame (from the next chunk on) and compress (at seal) stored fields
    /// as `mode` (the index's `index.codec`).
    pub fn set_stored_compression(&mut self, mode: crate::codec::store::StoredCompression) {
        self.store_compression = mode;
    }

    /// See [`SegmentWriter::set_stored_compression`].
    pub fn stored_compression(&self) -> crate::codec::store::StoredCompression {
        self.store_compression
    }

    /// Append `doc`'s stored fields to the open store chunk.
    fn store_document(&mut self, doc_id: DocId, doc: &Document, schema: &[Field]) {
        let full = match (self.store_chunks.last(), self.store_starts.last()) {
            (Some(c), Some(&start)) => self.store_compression.chunk_full_for((doc_id - start) as usize, self.store_sized, c.len(), !self.vectors.is_empty()),
            _ => true,
        };
        if full {
            self.store_sized = 0;
            let cap = self.store_chunks.last().map(|c| c.len() + c.len() / 8).unwrap_or(16 << 10);
            self.store_chunks.push(Vec::with_capacity(cap));
            self.store_starts.push(doc_id);
            self.store_compact.push(crate::codec::write_version() >= crate::codec::COMPACT_STORED_SINCE);
        }
        let compact = *self.store_compact.last().unwrap_or(&false);
        // Values of fields the schema marks unstored are left out. The view
        // serializes exactly like a `Document` holding only the kept fields.
        #[derive(serde::Serialize)]
        struct StoredView<'a> {
            fields: Vec<&'a (u16, crate::types::FieldValue)>,
        }
        let slot_of = &self.slot_of;
        // From v15 column-stored fields come back from their column.
        let from_column: &[u16] = if !self.column_stored.is_empty() && crate::codec::write_version() >= crate::codec::COLUMN_STORED_SINCE { &self.column_stored } else { &[] };
        let stored = |f: u16| !from_column.contains(&f) && match slot_of.get(f as usize) {
            Some(&s) if s != NO_SLOT => schema[s as usize].stored,
            _ => true,
        };
        let chunk = self.store_chunks.last_mut().unwrap();
        let off;
        let len;
        if compact {
            let mut buf = Vec::with_capacity(64);
            let kept: Vec<&(u16, crate::types::FieldValue)> = doc.fields.iter().filter(|(f, _)| stored(*f)).collect();
            // From v31 the `_id` is stored in its binary form.
            let uid = self.unique.field.filter(|_| crate::codec::write_version() >= crate::codec::UID_STORED_SINCE);
            if uid.is_some_and(|u| kept.iter().any(|(f, _)| *f == u)) {
                self.store_uid = true;
            }
            crate::codec::store::encode_doc_compact_uid(kept.into_iter(), uid, &mut buf);
            len = buf.len() as u64;
            write_vint(len, chunk);
            off = chunk.len();
            chunk.extend_from_slice(&buf);
        } else if doc.fields.iter().all(|(f, _)| stored(*f)) {
            len = bincode::serialized_size(doc).expect("Document serializes");
            write_vint(len, chunk);
            off = chunk.len();
            bincode::serialize_into(&mut *chunk, doc).expect("Document serializes");
        } else {
            let view = StoredView { fields: doc.fields.iter().filter(|(f, _)| stored(*f)).collect() };
            len = bincode::serialized_size(&view).expect("Document serializes");
            write_vint(len, chunk);
            off = chunk.len();
            bincode::serialize_into(&mut *chunk, &view).expect("Document serializes");
        }
        if len as usize > crate::codec::store::SIZED_DOC_BYTES {
            self.store_sized += 1;
        }
        self.store_locs.push((off as u32, len as u32));
    }

    /// Rebuild the per-field-id caches when a different schema is passed.
    /// Forget the schema the per-field caches describe (they are keyed by
    /// the schema slice's identity, which a fresh copy may reuse).
    pub fn reset_schema_cache(&mut self) {
        self.schema_key = (0, usize::MAX);
    }

    fn sync_schema(&mut self, schema: &[Field]) {
        let key = (schema.as_ptr() as usize, schema.len());
        if key == self.schema_key {
            return;
        }
        self.schema_key = key;
        let id_field = schema.iter().find(|f| f.name == "_id" && f.field_type == FieldType::Keyword).map(|f| f.id);
        if id_field != self.unique.field {
            // Documents already added were not tracked under this field.
            if self.num_docs > 0 {
                self.unique.broken = true;
            }
            self.unique.field = id_field;
        }
        let max = schema.iter().map(|f| f.id as usize + 1).max().unwrap_or(0);
        self.column_stored = schema.iter().filter(|f| f.column_stored && f.fast && f.field_type == FieldType::I64).map(|f| f.id).collect();
        self.synthetic_id = schema.iter().find_map(|f| f.synthetic_id.filter(|_| f.field_type == FieldType::Keyword).map(|s| (f.id, s)));
        for f in schema.iter().filter(|f| f.column_from_postings && f.fast && f.indexed && f.field_type == FieldType::Keyword) {
            self.fast_fields.derive_from_postings(f.id);
        }
        self.column_stored.sort_unstable();
        self.column_stored.dedup();
        self.slot_of = vec![NO_SLOT; max];
        let mut analyzers: Vec<Option<Arc<crate::analysis::Analyzer>>> = vec![None; max];
        for (i, field) in schema.iter().enumerate() {
            let f = field.id as usize;
            // First definition wins (as `iter().find` did).
            if self.slot_of[f] == NO_SLOT {
                self.slot_of[f] = i as u32;
            }
            if !field.points {
                self.fast_fields.skip_points(field.id);
            }
            if field.field_type == FieldType::Text && field.indexed && analyzers[f].is_none() {
                let a = if field.analyzer == "standard" {
                    self.default_analyzer.clone()
                } else {
                    Arc::new(
                        crate::analysis::Analyzer::by_name(&field.analyzer)
                            .unwrap_or_else(crate::analysis::Analyzer::standard),
                    )
                };
                analyzers[f] = Some(a);
            }
        }
        self.analyzers = analyzers;
    }

    /// Assign stable point IDs to rows appended since the previous reservation.
    /// This does not publish documents or build a graph. Mutable ingestion must
    /// complete graph admission for these IDs before checking in/publishing the
    /// writer. Existing standalone writers can still reserve lazily at flush.
    pub fn reserve_vector_points(&mut self, store: &VectorStore) -> Result<()> {
        if self.vector_reservations.values().flatten().any(|(_, r)| !r.belongs_to(store)) {
            return Err(crate::error::RusticError::Segment("writer vector reservations belong to another store".into()));
        }
        for (&field, builder) in &self.vectors {
            let batches = self.vector_reservations.entry(field).or_default();
            let start = batches.last().map_or(0, |(start, r)| *start + (r.points().end - r.points().start) as usize);
            if start < builder.len() {
                batches.push((start, store.reserve_points(builder.len() - start)?));
            }
        }
        Ok(())
    }

    /// Prepare an independent document before ES assigns its sequence number.
    /// The opaque token owns rollback until appended to a private writer.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn prepare_vector_document(doc: &Document, schema: &[Field], store: &Arc<VectorStore>) -> Result<PreparedDocumentVectors> {
        Self::new(0).prepare_document_vectors(doc, schema, store)
    }

    /// [`prepare_vector_document`](Self::prepare_vector_document) of a bulk's
    /// documents at once: each field's rows of every document are staged by
    /// one [`MutableIngestField::stage_documents`] call (one sequencer hold,
    /// one parallel prepare pass, one commit) instead of one staging per
    /// document. Each document keeps its own token, so its own rollback. An
    /// error fails the whole batch (every token is dropped, so rolled back):
    /// the caller then prepares the documents one by one, which reports each
    /// document's own error.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn prepare_vector_documents(docs: &[&Document], schema: &[Field], store: &Arc<VectorStore>)
        -> Result<Vec<PreparedDocumentVectors>>
    {
        fn collect<'a>(doc: &'a Document, schema: &[Field], rows: &mut BTreeMap<u16, Vec<&'a [f32]>>) {
            for (_, child) in &doc.nested { collect(child, schema, rows); }
            let mut seen = std::collections::HashSet::new();
            for (id, value) in &doc.fields {
                let Some(vector) = value.as_vector() else { continue };
                let Some(field) = schema.iter().find(|f| f.id == *id) else { continue };
                let FieldType::Vector { dim } = field.field_type else { continue };
                if vector.len() == dim as usize && seen.insert(*id) { rows.entry(*id).or_default().push(vector); }
            }
        }
        let mut tokens: Vec<_> = docs.iter().map(|_| PreparedDocumentVectors { store: store.clone(), batches: Vec::new(), graphless: Vec::new() }).collect();
        // Field -> (document, its rows), in document order.
        let mut fields: BTreeMap<u16, Vec<(usize, Vec<&[f32]>)>> = BTreeMap::new();
        for (i, doc) in docs.iter().enumerate() {
            if !doc.has_vectors() { continue; }
            let mut rows = BTreeMap::new();
            collect(doc, schema, &mut rows);
            for (id, rows) in rows { fields.entry(id).or_default().push((i, rows)); }
        }
        for (id, docs_rows) in fields {
            let Some(field) = store.registered_mutable_field(id) else {
                // No graph yet: admitted graphless, as one document at a time.
                for (doc, rows) in docs_rows { tokens[doc].graphless.push((id, 0, rows.len())); }
                continue;
            };
            let mapping = schema.iter().find(|f| f.id == id).expect("collected schema field");
            let options = mapping.vector_options.ok_or_else(|| crate::error::RusticError::Unsupported("mutable field requires explicit vector mapping".into()))?;
            let migrating = store.migration_pending(id);
            if !(options.graph || migrating)
                || !(migrating || field.matches_index_options(options)) || similarity_by_name(&mapping.similarity) != Some(field.similarity())
                || docs_rows.iter().any(|(_, rows)| rows.iter().any(|row| row.len() != field.dimensions() as usize)) {
                return Err(crate::error::RusticError::Segment("mutable field schema changed".into()));
            }
            let documents: Vec<Vec<&[f32]>> = docs_rows.iter().map(|(_, rows)| rows.clone()).collect();
            let staged = field.stage_documents(&documents)?;
            for ((doc, _), staged) in docs_rows.into_iter().zip(staged) {
                tokens[doc].batches.push((id, 0, staged.accept()?.0));
            }
        }
        Ok(tokens)
    }

    /// Batch GPU admission retains a separate cancellation token per document.
    /// Unsupported shapes use the existing per-document CPU path unchanged.
    #[cfg(all(feature = "mutable-hnsw-poc", feature = "gpu"))]
    pub fn prepare_gpu_documents(docs: &[&Document], schema: &[Field], store: &Arc<VectorStore>)
        -> Result<Option<Vec<PreparedDocumentVectors>>>
    {
        if docs.len() < 2 || !store.gpu_allowed() || docs.iter().any(|doc| !doc.nested.is_empty()) { return Ok(None); }
        let mut fields: BTreeMap<u16, Vec<(usize, &[f32])>> = BTreeMap::new();
        for (i, doc) in docs.iter().enumerate() {
            let mut seen = std::collections::HashSet::new();
            for (id, value) in &doc.fields {
                let Some(row) = value.as_vector() else { continue };
                if !seen.insert(*id) { continue; }
                let Some(mapping) = schema.iter().find(|f| f.id == *id) else { return Ok(None) };
                let Some(options) = mapping.vector_options else { return Ok(None) };
                let Some(field) = store.registered_mutable_field(*id) else { return Ok(None) };
                if !options.gpu || !options.graph
                    || !field.matches_index_options(options) || row.len() != field.dimensions() as usize
                    || similarity_by_name(&mapping.similarity) != Some(field.similarity()) { return Ok(None); }
                fields.entry(*id).or_default().push((i, row));
            }
        }
        if fields.is_empty() { return Ok(None); }
        let mut tokens: Vec<_> = docs.iter().map(|_| PreparedDocumentVectors { store: store.clone(), batches: Vec::new(), graphless: Vec::new() }).collect();
        for (id, rows) in fields {
            let field = store.registered_mutable_field(id).expect("validated mutable field");
            let batches: Vec<_> = rows.iter().map(|(_, row)| vec![*row]).collect();
            let Some(staged) = field.stage_gpu_documents(&batches)? else {
                // Dropping already prepared tokens rolls back earlier fields;
                // the caller then admits the complete documents on the CPU.
                return Ok(None);
            };
            for ((doc, _), staged) in rows.into_iter().zip(staged) {
                tokens[doc].batches.push((id, 0, staged.accept()?.0));
            }
        }
        Ok(Some(tokens))
    }

    /// Append the same vector payload/schema used for preparation; metadata may
    /// be added between preparation and this call under the shard write fence.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn append_prepared_document(&mut self, doc: &Document, schema: &[Field], mut prepared: PreparedDocumentVectors) {
        let mut offsets = BTreeMap::new();
        for (field, start, reservation) in &mut prepared.batches {
            let offset = offsets.entry(*field).or_insert_with(|| self.vectors.get(field).map_or(0, |builder| builder.len()));
            *start = *offset;
            *offset += (reservation.points().end - reservation.points().start) as usize;
        }
        for (field, start, rows) in &mut prepared.graphless {
            let offset = offsets.entry(*field).or_insert_with(|| self.vectors.get(field).map_or(0, |builder| builder.len()));
            *start = *offset;
            *offset += *rows;
        }
        self.add_document(doc, schema);
        self.accept_document_vectors(prepared);
    }

    /// Fallible document admission for a private ES bulk writer. The caller
    /// publishes this writer only after its successful operations reach WAL.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn add_document_with_vectors(&mut self, doc: &Document, schema: &[Field], store: &Arc<VectorStore>) -> Result<()> {
        let prepared = self.prepare_document_vectors(doc, schema, store)?;
        self.add_document(doc, schema);
        self.accept_document_vectors(prepared);
        Ok(())
    }

    /// Prepare only this document's vector rows, in the same nested postorder
    /// and first-value-per-field order used by add_document. No text is changed.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn prepare_document_vectors(&self, doc: &Document, schema: &[Field], store: &Arc<VectorStore>)
        -> Result<PreparedDocumentVectors>
    {
        self.prepare_document_vectors_reusing(doc, schema, store, &BTreeMap::new())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn prepare_document_vectors_reusing(&self, doc: &Document, schema: &[Field], store: &Arc<VectorStore>,
        previous: &BTreeMap<(u16, u32), u64>) -> Result<PreparedDocumentVectors> {
        // Every document of every index comes through here: one without
        // vectors (the common case) must not pay a schema scan per field.
        if !doc.has_vectors() {
            return Ok(PreparedDocumentVectors { store: store.clone(), batches: Vec::new(), graphless: Vec::new() });
        }
        fn collect<'a>(doc: &'a Document, schema: &[Field], rows: &mut BTreeMap<u16, Vec<(u32, &'a [f32])>>, position: &mut u32) {
            for (_, child) in &doc.nested { collect(child, schema, rows, position); }
            let mut seen = std::collections::HashSet::new();
            for (id, value) in &doc.fields {
                let Some(vector) = value.as_vector() else { continue };
                let Some(field) = schema.iter().find(|f| f.id == *id) else { continue };
                let FieldType::Vector { dim } = field.field_type else { continue };
                if vector.len() == dim as usize && seen.insert(*id) { rows.entry(*id).or_default().push((*position, vector)); }
            }
            *position += 1;
        }
        let mut rows = BTreeMap::new();
        collect(doc, schema, &mut rows, &mut 0);
        let mut batches = Vec::new();
        let mut graphless = Vec::new();
        for (id, rows) in rows {
            let Some(field) = store.registered_mutable_field(id) else {
                graphless.push((id, self.vectors.get(&id).map_or(0, |builder| builder.len()), rows.len()));
                continue;
            };
            let mapping = schema.iter().find(|f| f.id == id).expect("collected schema field");
            let options = mapping.vector_options.ok_or_else(|| crate::error::RusticError::Unsupported("mutable field requires explicit vector mapping".into()))?;
            // An `index_options` migration in progress: the current graph
            // admits the rows until its successor is swapped in.
            let migrating = store.migration_pending(id);
            if !(options.graph || migrating)
                || !(migrating || field.matches_index_options(options)) || similarity_by_name(&mapping.similarity) != Some(field.similarity())
                || rows.iter().any(|(_, row)| row.len() != field.dimensions() as usize) {
                return Err(crate::error::RusticError::Segment("mutable field schema changed".into()));
            }
            let start = self.vectors.get(&id).map_or(0, |builder| builder.len());
            let reserved = self.vector_reservations.get(&id);
            // Every earlier row holds a reservation or was admitted while the
            // field had no graph; any other row is unstaged.
            let graphless_end = self.graphless_vector_rows.get(&id).and_then(|ranges| ranges.last()).map_or(0, |range| range.end);
            if !self.reservations_owned_by(id, &field)
                || reserved.and_then(|batches| batches.last()).map_or(0, |(offset, reservation)|
                    *offset + (reservation.points().end - reservation.points().start) as usize).max(graphless_end) != start {
                return Err(crate::error::RusticError::Segment("mutable document admission found unstaged writer rows".into()));
            }
            if previous.is_empty() {
                let values: Vec<_> = rows.iter().map(|(_, row)| *row).collect();
                let (reservation, _) = field.stage(&values)?.accept()?;
                batches.push((id, start, reservation));
            } else {
                for (ordinal, (position, row)) in rows.into_iter().enumerate() {
                    let reused = previous.get(&(id, position)).map(|point| field.reuse_unchanged(*point, row)).transpose()?.flatten();
                    let reservation = match reused {
                        Some(reservation) => reservation,
                        None => field.stage(&[row])?.accept()?.0,
                    };
                    batches.push((id, start + ordinal, reservation));
                }
            }
        }
        Ok(PreparedDocumentVectors { store: store.clone(), batches, graphless })
    }

    /// Infallible after preparation while the writer/schema remain under the
    /// index write fence. Only appends reservation ownership, never graph work.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn accept_document_vectors(&mut self, prepared: PreparedDocumentVectors) {
        if !prepared.batches.is_empty() { self.mutable_store = Some(prepared.store); }
        for (field, start, reservation) in prepared.batches {
            debug_assert!(self.vectors.get(&field).map_or(0, |b| b.len()) >=
                start + (reservation.points().end - reservation.points().start) as usize);
            self.vector_reservations.entry(field).or_default().push((start, reservation));
        }
        for (field, start, rows) in prepared.graphless {
            let ranges = self.graphless_vector_rows.entry(field).or_default();
            match ranges.last_mut() {
                Some(last) if last.end == start => last.end += rows,
                _ => ranges.push(start..start + rows),
            }
        }
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn reservations_owned_by(&self, field: u16, owner: &Arc<crate::vector::mutable_ingest::MutableIngestField>) -> bool {
        let Some(batches) = self.vector_reservations.get(&field) else { return true; };
        let mut verified = self.verified_reservations.lock();
        let start = verified.get(&field).filter(|(previous, _)| Arc::ptr_eq(previous, owner))
            .map_or(0, |(_, count)| *count);
        if batches[start..].iter().any(|(_, reservation)| !owner.owns_reservation(reservation)) { return false; }
        verified.insert(field, (owner.clone(), batches.len()));
        true
    }

    /// Complete new rows in the shared field graph before handing this writer
    /// to the shard publication path. The retained reservation owns rollback if
    /// the writer is abandoned before its text/vector files are published.
    /// ES WAL sequencing and refresh visibility remain the caller's obligation.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn stage_mutable_field(&mut self, field: u16, ingest: &Arc<crate::vector::mutable_ingest::MutableIngestField>) -> Result<usize> {
        let Some(builder) = self.vectors.get(&field) else { return Ok(0) };
        if !builder.options.is_some_and(|o| o.graph) {
            return Err(crate::error::RusticError::Unsupported("mutable ingestion requires explicit hnsw mapping".into()));
        }
        let migrating = ingest.store().is_ok_and(|store| store.migration_pending(field));
        if !(migrating || ingest.matches_index_options(builder.options.expect("validated mapping"))) || field != ingest.field_id() || builder.dim() != ingest.dimensions() || builder.similarity != ingest.similarity() {
            return Err(crate::error::RusticError::Segment("mutable field schema mismatch".into()));
        }
        if !self.reservations_owned_by(field, ingest) {
            return Err(crate::error::RusticError::Segment("writer contains reservations from another vector backend".into()));
        }
        let batches = self.vector_reservations.entry(field).or_default();
        let start = batches.last().map_or(0, |(start, r)| *start + (r.points().end - r.points().start) as usize);
        if start == builder.len() { return Ok(0) }
        let rows: Vec<_> = builder.rows()[start..].iter().map(|(_, values)| values.as_slice()).collect();
        let store = ingest.store()?;
        let staged = ingest.stage(&rows)?;
        self.mutable_store = Some(store);
        let (reservation, _) = staged.accept()?;
        batches.push((start, reservation));
        Ok(rows.len())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn mutable_vector_store(&self) -> Option<&Arc<VectorStore>> { self.mutable_store.as_ref() }

    #[cfg(feature = "mutable-hnsw-poc")]
    pub(crate) fn reserved_vector_point(&self, field: u16, doc: DocId) -> Option<u64> {
        let ordinal = self.vectors.get(&field)?.rows().binary_search_by_key(&doc, |row| row.0).ok()?;
        let batches = self.vector_reservations.get(&field)?;
        let batch = batches.partition_point(|(start, _)| *start <= ordinal).checked_sub(1)?;
        let (start, reservation) = &batches[batch];
        let point = reservation.points().start + (ordinal - start) as u64;
        (point < reservation.points().end).then_some(point)
    }

    /// Documents of this writer whose `field` row was admitted while the
    /// field had no graph (`flat`, `int8_flat`, or a graph type before its
    /// graph was built): they hold no point and are in no graph until the
    /// seal stores them (`insert_unreserved_rows`), so a graph kNN must
    /// score them itself meanwhile.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn graphless_vector_docs(&self, field: u16) -> Vec<DocId> {
        let Some(builder) = self.vectors.get(&field) else { return Vec::new() };
        let rows = builder.rows();
        self.graphless_vector_rows.get(&field).into_iter().flatten()
            .flat_map(|range| rows.get(range.clone()).into_iter().flatten().map(|row| row.0))
            .collect()
    }

    /// This writer's reserved point ranges of `field` as `(first point,
    /// end point, first vector row)`, sorted by point: one entry per
    /// admitted batch, so a kNN resolves a hit's point to its doc by binary
    /// search ([`Self::vector_row_doc`]) instead of listing every reserved
    /// point into a map on each query.
    pub fn reserved_point_ranges(&self, field: u16) -> Vec<(u64, u64, usize)> {
        let mut ranges: Vec<(u64, u64, usize)> = self.vector_reservations.get(&field).into_iter().flatten()
            .map(|(start, reservation)| (reservation.points().start, reservation.points().end, *start)).collect();
        ranges.sort_unstable_by_key(|r| r.0);
        ranges
    }

    /// The doc of vector row `row` of `field`.
    #[inline]
    pub fn vector_row_doc(&self, field: u16, row: usize) -> Option<DocId> {
        self.vectors.get(&field)?.rows().get(row).map(|(doc, _)| *doc)
    }

    /// Reserved identities paired with local text documents. Segment renumbering
    /// and a later flush must preserve these IDs. Unreserved rows are omitted.
    pub fn reserved_vector_points(&self, field: u16) -> Vec<(DocId, u64)> {
        let Some(builder) = self.vectors.get(&field) else { return Vec::new() };
        self.vector_reservations.get(&field).into_iter().flatten().flat_map(|(start, reservation)| {
            builder.rows()[*start..].iter().zip(reservation.points()).map(|((doc, _), point)| (*doc, point))
        }).collect()
    }

    /// Count actual bitmap slots, including nested children, before sequencing.
    pub fn document_slots(doc: &Document) -> crate::error::Result<usize> {
        doc.nested.iter().try_fold(1usize, |count, (_, child)| {
            count.checked_add(Self::document_slots(child)?).ok_or_else(|| crate::error::RusticError::Memory("document bitmap count overflow".into()))
        })
    }
    pub fn prepare_bitmap_batch(&mut self, documents: usize) -> crate::error::Result<crate::postings::BitmapCredits> {
        let credits = crate::postings::BitmapCredits::reserve(documents)?;
        self.prepare_bitmap_capacity(documents)?;
        Ok(credits)
    }
    pub fn prepare_bitmap_capacity(&mut self, documents: usize) -> crate::error::Result<()> {
        if documents == 0 { return Ok(()); }
        let end = usize::try_from(self.num_docs).unwrap().checked_add(documents)
            .and_then(|v| u32::try_from(v).ok()).ok_or_else(|| crate::error::RusticError::Memory("writer document capacity overflow".into()))?;
        let dels = self.dels.get_mut();
        dels.latest.prepare_capacity(end)?;
        if let Some(visible) = &mut dels.visible { visible.prepare_capacity(end)?; }
        else { dels.spare_visible.prepare_capacity(end)?; }
        Ok(())
    }
    /// Whether a previously prepared idle writer can accept this batch without
    /// new bitmap admission. The caller retains exclusive ownership while using it.
    pub fn has_prepared_bitmap_capacity(&self, documents: usize) -> bool {
        if documents==0 {return true;}
        let Some(end)=usize::try_from(self.num_docs).ok().and_then(|n|n.checked_add(documents)).and_then(|n|u32::try_from(n).ok()) else {return false};
        let dels=self.dels.read();
        dels.latest.has_prepared_capacity(end)
            && dels.visible.as_ref().unwrap_or(&dels.spare_visible).has_prepared_capacity(end)
    }
    /// Consume only the pre-WAL allowance when a refresh replaced the writer.
    /// This never performs fresh budget admission after the durable boundary.
    pub fn ensure_document_bitmap(&mut self, doc: &Document, credits: &mut crate::postings::BitmapCredits) {
        let slots = Self::document_slots(doc).expect("preflighted document slots");
        let end = self.num_docs.checked_add(u32::try_from(slots).expect("preflighted document count")).expect("preflighted writer capacity");
        let dels = self.dels.get_mut();
        dels.latest.ensure_credited_capacity(end, credits);
        if let Some(visible) = &mut dels.visible { visible.ensure_credited_capacity(end, credits); }
        else { dels.spare_visible.ensure_credited_capacity(end, credits); }
    }

    /// Add a document to the segment. A document with nested children
    /// (`doc.nested`) is added as one block: every child (recursively, its
    /// own children first) and then the document itself.
    pub fn add_document(&mut self, doc: &Document, schema: &[Field]) {
        if doc.nested.is_empty() && self.nested.is_none() {
            self.add_one(doc, schema);
        } else {
            self.add_block(doc, schema, "", 0);
        }
    }

    fn add_block(&mut self, doc: &Document, schema: &[Field], path: &str, depth: u16) {
        for (p, child) in &doc.nested {
            self.add_block(child, schema, p, depth + 1);
        }
        let ord = if depth == 0 { 0 } else { self.nested.get_or_insert_with(NestedInfo::default).intern(path, depth) };
        let id = self.num_docs;
        self.add_one(doc, schema);
        if let Some(n) = &mut self.nested {
            n.set(id, ord);
        }
    }

    fn add_one(&mut self, doc: &Document, schema: &[Field]) {
        self.sync_schema(schema);
        let doc_id = self.num_docs;
        self.store_document(doc_id, doc, schema);
        self.field_lens.clear();
        self.fast_counts.clear();
        self.pos_next.clear();
        self.agg_doc.clear();
        self.kw_seen.clear();
        self.kw_buf.clear();

        for (field_id, value) in &doc.fields {
            let field_id = *field_id;
            let slot = self.slot_of.get(field_id as usize).copied().unwrap_or(NO_SLOT);
            if slot == NO_SLOT {
                continue;
            }
            let field = &schema[slot as usize];
            match &field.field_type {
                FieldType::Text | FieldType::Keyword => {
                    if let Some(text) = value.as_text() {
                        if field.indexed {
                            if field.field_type == FieldType::Text {
                                let base = self.pos_next.iter().find(|e| e.0 == field_id).map(|e| e.1).unwrap_or(0);
                                let (n, end, all) = self.index_text(doc_id, field_id, text, base);
                                if field.norms {
                                    bump(&mut self.field_lens, field_id, if field.discount_overlaps { n } else { all });
                                }
                                bump(&mut self.agg_doc, field_id, all);
                                let next = end.saturating_add(field.position_increment_gap);
                                match self.pos_next.iter_mut().find(|e| e.0 == field_id) {
                                    Some(e) => e.1 = next,
                                    None => self.pos_next.push((field_id, next)),
                                }
                            } else if field.features {
                                // `token\0hexbits`: the term is the token, its
                                // tf the weight (empty / malformed: no term).
                                if let Some((token, w)) = crate::fieldtypes::features::decode(text) {
                                    self.feat_seen.push((field_id, self.feat_buf.len() as u32, token.len() as u32, crate::fieldtypes::features::weight_to_freq(w)));
                                    self.feat_buf.extend_from_slice(token.as_bytes());
                                    self.kw_seen.push((field_id, self.kw_buf.len() as u32, token.len() as u32));
                                    self.kw_buf.extend_from_slice(token.as_bytes());
                                }
                            } else {
                                self.add_key_posting(field_id, text.as_bytes(), doc_id);
                                self.kw_seen.push((field_id, self.kw_buf.len() as u32, text.len() as u32));
                                self.kw_buf.extend_from_slice(text.as_bytes());
                            }
                        }
                        if field.fast {
                            if self.is_multi(field_id) {
                                self.fast_fields.multi_keyword_column_mut(field_id).push(doc_id, text);
                            } else if bump(&mut self.fast_counts, field_id, 1) >= 1 {
                                self.promote_keyword(field_id);
                                self.fast_fields.multi_keyword_column_mut(field_id).push(doc_id, text);
                            } else {
                                self.fast_fields.keyword_column_mut(field_id).set(doc_id, text);
                            }
                        }
                    }
                }
                FieldType::I64 | FieldType::Date => {
                    if let Some(v) = value.as_i64() {
                        // A numeric field with a column is queried through
                        // it (points index or block min/max), never through
                        // postings: write them only for a column-less field.
                        if field.indexed && !field.fast {
                            self.add_key_posting(field_id, &v.to_be_bytes(), doc_id);
                        }
                        if field.fast {
                            if self.is_multi(field_id) {
                                self.fast_fields.multi_i64_column_mut(field_id).push(doc_id, v);
                            } else if bump(&mut self.fast_counts, field_id, 1) >= 1 {
                                self.promote_i64(field_id);
                                self.fast_fields.multi_i64_column_mut(field_id).push(doc_id, v);
                            } else {
                                self.fast_fields.i64_column_mut(field_id).set(doc_id, v);
                            }
                        }
                    }
                }
                FieldType::U64 => {
                    if let Some(v) = value.as_u64() {
                        if field.indexed && !field.fast {
                            self.add_key_posting(field_id, &v.to_be_bytes(), doc_id);
                        }
                        if field.fast {
                            if self.is_multi(field_id) {
                                self.fast_fields.multi_u64_column_mut(field_id).push(doc_id, v);
                            } else if bump(&mut self.fast_counts, field_id, 1) >= 1 {
                                self.promote_u64(field_id);
                                self.fast_fields.multi_u64_column_mut(field_id).push(doc_id, v);
                            } else {
                                self.fast_fields.u64_column_mut(field_id).set(doc_id, v);
                            }
                        }
                    }
                }
                FieldType::F64 => {
                    if let Some(v) = value.as_f64() {
                        if field.indexed && !field.fast {
                            self.add_key_posting(field_id, &v.to_bits().to_be_bytes(), doc_id);
                        }
                        if field.fast {
                            if self.is_multi(field_id) {
                                self.fast_fields.multi_f64_column_mut(field_id).push(doc_id, v);
                            } else if bump(&mut self.fast_counts, field_id, 1) >= 1 {
                                self.promote_f64(field_id);
                                self.fast_fields.multi_f64_column_mut(field_id).push(doc_id, v);
                            } else {
                                self.fast_fields.f64_column_mut(field_id).set(doc_id, v);
                            }
                        }
                    }
                }
                FieldType::Bool => {
                    if let Some(v) = value.as_bool() {
                        if field.indexed {
                            self.add_key_posting(field_id, &[v as u8], doc_id);
                        }
                        if field.fast {
                            self.fast_fields.bool_column_mut(field_id).set(doc_id, v);
                        }
                    }
                }
                FieldType::GeoPoint => {
                    if let Some((lat, lon)) = value.as_geo_point() {
                        if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) {
                            self.geo.entry(field_id).or_default().push(doc_id, lat, lon);
                        }
                    }
                }
                FieldType::Vector { dim } => {
                    if let Some(v) = value.as_vector() {
                        if v.len() == *dim as usize {
                            let sim = similarity_by_name(&field.similarity).unwrap_or(Similarity::Cosine);
                            let b = self
                                .vectors
                                .entry(field_id)
                                .or_insert_with(|| VectorFieldBuilder::new(*dim as u32, sim).with_options(field.vector_options));
                            // Dedupe: one vector per doc per field (last wins is
                            // not supported; first wins).
                            if b.get(doc_id).is_none() {
                                b.set(doc_id, v);
                                self.bytes += v.len() * 4 + 16;
                            }
                        }
                    }
                }
            }
        }
        for i in 0..self.field_lens.len() {
            let (field, len) = self.field_lens[i];
            self.fast_fields.norms_mut(field).set(doc_id, len);
        }
        for i in 0..self.agg_doc.len() {
            let (field, all) = self.agg_doc[i];
            if all > 0 {
                self.add_agg(field, all);
            }
        }
        self.note_keyword_aggs();
        self.post_features(doc_id);
        // `geo_shape` values: the triangles tessellated at parse time feed
        // the tree (indexed fields) and the doc value (with the centroid of
        // every value, fields with doc values).
        for (field_id, tris) in &doc.shape_triangles {
            let slot = self.slot_of.get(*field_id as usize).copied().unwrap_or(NO_SLOT);
            if slot == NO_SLOT || tris.is_empty() {
                continue;
            }
            let field = &schema[slot as usize];
            if !field.shape {
                continue;
            }
            // The value written at parse time; written here for docs built
            // without it.
            let owned;
            let value = if !field.fast {
                None
            } else if let Some(v) = doc.shape_value(*field_id) {
                Some(v)
            } else {
                owned = doc.shape_centroid(*field_id).and_then(|c| crate::fieldtypes::shape_field::doc_value(tris, c).ok());
                owned.as_deref()
            };
            let tree: &[[i32; 7]] = if field.indexed { tris } else { &[] };
            self.shapes.entry(*field_id).or_default().push(doc_id, tree, value);
        }

        self.num_docs += 1;
        let n = self.num_docs;
        let d = self.dels.get_mut();
        d.latest.resize(n);
        if let Some(v) = d.visible.as_mut() {
            v.resize(n);
        }
    }

    /// Encode this segment into `dir` and reopen it as an immutable `Segment`.
    /// In-memory deletions are carried over into the segment's `.del` file.
    ///
    /// Standalone flush: vectors go to a private [`VectorStore`] over `dir`
    /// (see [`Segment::open`]). Inside an index use [`write_to_store`](Self::write_to_store).
    pub fn write_to(self, dir: &dyn Directory) -> Result<Segment> {
        let store = if self.vectors.is_empty() { None } else { Some(VectorStore::open_dir(dir)?) };
        self.write_to_store(dir, store.as_ref())
    }

    /// Flush into `dir`: each vector field's live rows become one appendable
    /// chunk in `store` (new point ids, no graph build), the text segment
    /// stores `doc -> PointId`, and the opened segment is attached.
    pub fn write_to_store(self, dir: &dyn Directory, store: Option<&Arc<VectorStore>>) -> Result<Segment> {
        self.write_to_store_as(dir, store, false)
    }

    /// [`SegmentWriter::write_to_store`]; with `hidden`, the segment is not
    /// yet searchable (sealed before an ES refresh published it): its
    /// points are kept but not searchable until
    /// [`VectorStore::reveal`]. Otherwise deletes still pending on the
    /// memtable stay pending on the segment (its points stay searchable
    /// until [`Segment::publish_deletions`]).
    pub fn write_to_store_as(self, dir: &dyn Directory, store: Option<&Arc<VectorStore>>, hidden: bool) -> Result<Segment> {
        crate::on_encode_pool(|| self.write_to_store_in_pool(dir, store, hidden))
    }

    /// [`SegmentWriter::write_to_store_as`] without consuming the memtable:
    /// it stays readable (and deletable) while it is encoded, so a
    /// background flush can run without any index lock. Deletes that land
    /// meanwhile are not in the result: the caller installs the
    /// memtable's final deletions on the segment.
    pub fn write_to_store_ref(&self, dir: &dyn Directory, store: Option<&Arc<VectorStore>>, hidden: bool) -> Result<Segment> {
        crate::on_encode_pool(|| self.write_to_store_in_pool(dir, store, hidden))
    }

    fn write_to_store_in_pool(&self, dir: &dyn Directory, store: Option<&Arc<VectorStore>>, hidden: bool) -> Result<Segment> {
        let id = self.id;
        let latest = self.latest_deletions_admitted()?;
        // Hidden text is not searchable, but mutable document leases still
        // need the pending-deletion delta when refresh publishes this segment.
        #[cfg(feature = "mutable-hnsw-poc")]
        let retain_pending = self.mutable_store.is_some();
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let retain_pending = false;
        let visible = if hidden && !retain_pending { None } else { self.visible_deletions_admitted()? };
        // Every row gets a point, deleted docs' too (dead from the start):
        // the chunk holds exactly the vectors of this flush, as a Lucene
        // segment does, which is what ES fits a quantized field's scoring
        // quantizer on (see `vector::es_quant`).
        let mut point_cols: Vec<(u16, Vec<u64>)> = Vec::new();
        let mut prepared_vectors = Vec::new();
        // Points stored without a graph reservation (rows written while their
        // field had no graph): inserted into the field's graph, if it has one
        // now, once the segment is attached (`insert_unreserved_rows`).
        #[cfg(feature = "mutable-hnsw-poc")]
        let mut unreserved: Vec<(u16, std::ops::Range<u64>)> = Vec::new();
        if let Some(store) = store {
            for (field, b) in &self.vectors {
                let rows: Vec<(DocId, &[f32])> = b.rows().iter().map(|(d, v)| (*d, v.as_slice())).collect();
                let mut col = vec![NO_POINT; self.num_docs as usize];
                if !rows.is_empty() {
                    // Unreserved stretches get fresh points first; then the
                    // reservations are listed in row order. Before or between
                    // reserved batches only rows admitted while the field had
                    // no graph may be unreserved (any other row there was
                    // never staged); an unreserved tail is reserved here as
                    // before, and only its graphless rows join a graph.
                    #[cfg(feature = "mutable-hnsw-poc")]
                    let graphless: &[std::ops::Range<usize>] = self.graphless_vector_rows.get(field).map(Vec::as_slice).unwrap_or(&[]);
                    #[cfg(not(feature = "mutable-hnsw-poc"))]
                    let graphless: &[std::ops::Range<usize>] = &[];
                    let batches: Vec<&(usize, crate::vector::store::PointReservation)> = self.vector_reservations.get(field).into_iter().flatten().collect();
                    let mut next = 0;
                    let mut gaps: Vec<(usize, crate::vector::store::PointReservation)> = Vec::new();
                    for (start, reservation) in &batches {
                        let end = *start + (reservation.points().end - reservation.points().start) as usize;
                        if *start < next || end > rows.len()
                            || (*start > next && !graphless.iter().any(|range| range.start <= next && *start <= range.end)) {
                            return Err(crate::error::RusticError::Segment("writer vector reservation coverage mismatch".into()));
                        }
                        if *start > next { gaps.push((next, store.reserve_points(*start - next)?)); }
                        next = end;
                    }
                    if next < rows.len() { gaps.push((next, store.reserve_points(rows.len() - next)?)); }
                    #[cfg(feature = "mutable-hnsw-poc")]
                    for (start, gap) in &gaps {
                        let end = *start + (gap.points().end - gap.points().start) as usize;
                        for range in graphless {
                            let (lo, hi) = (range.start.max(*start), range.end.min(end));
                            if lo < hi { unreserved.push((*field, gap.points().start + (lo - start) as u64..gap.points().start + (hi - start) as u64)); }
                        }
                    }
                    let mut ordered: Vec<(usize, &crate::vector::store::PointReservation)> = batches.iter().map(|(start, r)| (*start, r)).chain(gaps.iter().map(|(start, r)| (*start, r))).collect();
                    ordered.sort_by_key(|(start, _)| *start);
                    let reservations: Vec<&crate::vector::store::PointReservation> = ordered.into_iter().map(|(_, r)| r).collect();
                    let prepared = store.prepare_reserved_appends(dir, *field, b.similarity, b.dim(), &rows, &reservations)?;
                    for ((doc, _), &point) in rows.iter().zip(prepared.points()) { col[*doc as usize] = point; }
                    prepared_vectors.push(prepared);
                }
                point_cols.push((*field, col));
            }
        }
        // The file's one format version, read once: the floor may rise
        // while the sections encode.
        let version = if self.shapes.is_empty() { crate::codec::write_version() } else { crate::codec::shape_write_version() };
        if version < crate::codec::SHAPE_TREE_SINCE && !self.shapes.is_empty() {
            return Err(crate::error::RusticError::Unsupported(format!("geo_shape fields need segment format v{} (writing v{version})", crate::codec::SHAPE_TREE_SINCE)));
        }
        let sections = self.encode_sections(&point_cols, version);
        let s = sections.refs();
        // Stream the sections to the file (no assembled copy of the segment).
        let frame = SegmentFileFrame::new_versioned(s, sections.num_docs, sections.version);
        // Durable at the next commit point, which alone references it
        // (Lucene's refresh does not fsync either): `Index::commit` syncs it.
        dir.write_parts_unsynced(&segment_file_name(id), &frame.parts(s))?;
        let segment = Segment::open_inner(dir, id)?;
        let any_deleted = !latest.is_empty();
        segment.set_deletions(latest, visible)?;
        // Every fallible write must finish before registering the vectors
        // with their document owner. On an error the provisional guards drop
        // their chunks without exposing them to optimizer deletion cleanup.
        if any_deleted {
            segment.write_deletions(dir)?;
        }
        if let Some(st) = store {
            st.attach_prepared(&segment, &mut prepared_vectors, hidden);
            #[cfg(feature = "mutable-hnsw-poc")]
            for (field, points) in unreserved {
                st.insert_unreserved_rows(field, points);
            }
        }
        Ok(segment)
    }

    /// Encode the segment file bytes. Vectors are not part of a text
    /// segment: use [`write_to`](Self::write_to) / [`write_to_store`](Self::write_to_store)
    /// to keep them.
    pub fn encode(self) -> Vec<u8> {
        let version = if self.shapes.is_empty() { crate::codec::write_version() } else { crate::codec::shape_write_version() };
        self.encode_versioned(version)
    }

    /// [`encode`](Self::encode) as format `version` (a writer's snapshot of
    /// [`crate::codec::write_version`]).
    pub fn encode_versioned(self, version: u32) -> Vec<u8> {
        let s = self.encode_sections(&[], version);
        let frame = SegmentFileFrame::new_versioned(s.refs(), s.num_docs, s.version);
        frame.parts(s.refs()).concat()
    }

    /// Encode every section as format `version`, captured once by the
    /// caller: one format version for the header and every postings run,
    /// whichever worker encodes it, even if the cluster's format floor
    /// rises meanwhile.
    fn encode_sections(&self, point_cols: &[(u16, Vec<u64>)], version: u32) -> EncodedSections {
        let num_docs = self.num_docs;
        let impacts = version >= crate::codec::IMPACTS_SINCE;
        let mut extra_entries: Vec<(u8, u16, Vec<u8>)> = Vec::new();
        if let Some(n) = &self.nested {
            if n.has_nested() {
                let mut n = n.clone();
                n.resize(num_docs);
                let mut b = Vec::new();
                n.encode(&mut b);
                extra_entries.push((EXTRA_NESTED, 0, b));
            }
        }
        for (field, g) in &self.geo {
            let mut b = Vec::new();
            let index = GeoIndex::build(g.to_vec());
            if version >= crate::codec::GEO_COMPACT_SINCE {
                index.encode_compact(&mut b);
                extra_entries.push((EXTRA_GEO2, *field, b));
            } else {
                index.encode(&mut b);
                extra_entries.push((EXTRA_GEO, *field, b));
            }
        }
        // `geo_shape` fields (only at v25 or later: write_to refuses them below).
        if version >= crate::codec::SHAPE_TREE_SINCE {
            for (field, buf) in &self.shapes {
                for part in buf.encode_tree(num_docs) {
                    extra_entries.push((EXTRA_SHAPE_TREE, *field, part));
                }
                for part in buf.encode_values(num_docs) {
                    extra_entries.push((EXTRA_SHAPE_DV, *field, part));
                }
            }
        }
        for (f, a) in self.aggs.iter().enumerate() {
            if a.doc_count > 0 {
                extra_entries.push((EXTRA_FIELD_STATS, f as u16, a.encode()));
            }
        }
        if version >= crate::codec::COLUMN_STORED_SINCE {
            for &f in &self.column_stored {
                extra_entries.push((EXTRA_COLUMN_STORED, f, Vec::new()));
            }
        }
        if let Some((field, s)) = &self.synthetic_id {
            extra_entries.push((crate::codec::EXTRA_SYNTHETIC_ID, *field, crate::synthetic_id::encode_extra(s)));
        }
        if version >= crate::codec::KEYWORD_FROM_POSTINGS_SINCE {
            let mut from_postings: Vec<u16> = self.fast_fields.keyword_columns().keys().copied().filter(|&f| self.fast_fields.is_from_postings(f)).collect();
            from_postings.sort_unstable();
            for f in from_postings {
                extra_entries.push((EXTRA_KEYWORD_FROM_POSTINGS, f, Vec::new()));
            }
        }
        let extra = encode_extra(&extra_entries);
        // The encoders read the columns as they are (`num_docs` is passed
        // explicitly): the memtable stays intact for readers meanwhile.
        let fast_fields = &self.fast_fields;
        // Chunks opened compact are re-encoded plain if this seal writes
        // below v18 (the format is read per chunk and again at seal; the
        // floor only rises, so this is for held-back test writers).
        let plain: Option<(Vec<Vec<u8>>, Vec<bool>)> = (version < crate::codec::COMPACT_STORED_SINCE && self.store_compact.contains(&true)).then(|| {
            let chunks = self.store_chunks.iter().zip(self.store_compact.iter().copied().chain(std::iter::repeat(false))).map(|(c, compact)| {
                if !compact { return c.clone(); }
                let mut out = Vec::with_capacity(c.len() * 2);
                let mut pos = 0usize;
                while pos < c.len() {
                    let len = crate::codec::varint::read_vint(c, &mut pos).expect("memtable stored chunk") as usize;
                    crate::codec::store::append_compact_as_plain(&c[pos..pos + len], &mut out).expect("memtable stored doc");
                    pos += len;
                }
                out
            }).collect();
            (chunks, Vec::new())
        });
        // Binary ids are re-encoded as keywords if this seal writes below
        // v31 (as above, for held-back writers).
        let plain = plain.or_else(|| (version < crate::codec::UID_STORED_SINCE && self.store_uid).then(|| {
            let chunks = self.store_chunks.iter().zip(self.store_compact.iter().copied().chain(std::iter::repeat(false))).map(|(c, compact)| {
                if compact { crate::codec::store::compact_chunk_without_uid(c).expect("memtable stored chunk") } else { c.clone() }
            }).collect();
            (chunks, self.store_compact.clone())
        }));
        let (store_chunks, store_compact) = match &plain { Some((c, f)) => (c, f), None => (&self.store_chunks, &self.store_compact) };
        let postings = &self.postings;
        let store_counts: Vec<u32> =
            self.store_starts.iter().enumerate().map(|(i, &s)| self.store_starts.get(i + 1).copied().unwrap_or(num_docs) - s).collect();
        let ff = fast_fields;
        // Postings + dictionary, fast fields and the store are independent.
        let ((terms, postings), (fast, store)) = rayon::join(
            || encode_postings(postings, ff, num_docs, impacts, version >= crate::codec::INLINE_POSTINGS_SINCE, version >= crate::codec::COMPACT_TERMS_SINCE, version >= crate::codec::INLINE_POSITIONS_SINCE, version >= crate::codec::PACKED_TERMS_SINCE, version >= crate::codec::INT_TERMS_SINCE, self.synthetic_id.map(|(f, _)| f)),
            || rayon::join(|| crate::codec::fastfields::encode_fast_fields_versioned(ff, num_docs, version), || crate::codec::store::encode_store_chunks_uid(store_chunks, &store_counts, store_compact, self.store_compression.zstd_at(version), version >= crate::codec::UID_STORED_SINCE)),
        );
        let vectors = encode_point_map(point_cols);
        EncodedSections { terms, postings, fast, store, vectors, extra, num_docs, version }
    }

    /// Fold this doc's keyword values into the field statistics: one
    /// `sumTotalTermFreq` per distinct value per field (keywords are indexed
    /// without frequencies), one `docCount` per field present.
    fn note_keyword_aggs(&mut self) {
        let mut i = 0;
        while i < self.kw_seen.len() {
            let field = self.kw_seen[i].0;
            if self.kw_seen[..i].iter().any(|e| e.0 == field) {
                i += 1;
                continue;
            }
            let mut vals: Vec<&[u8]> = Vec::new();
            for &(f, st, len) in &self.kw_seen[i..] {
                if f == field {
                    vals.push(&self.kw_buf[st as usize..(st + len) as usize]);
                }
            }
            let distinct = if vals.len() == 1 {
                1
            } else {
                vals.sort_unstable();
                vals.dedup();
                vals.len() as u32
            };
            self.add_agg(field, distinct);
            i += 1;
        }
    }

    /// Index the tokens of one text value at positions `base..`; returns
    /// the length counting only tokens that start a new position (Lucene's
    /// `discountOverlaps` field length), one past the last position used
    /// (`base` when there were no tokens), and the total token count
    /// (overlapping tokens included; `sumTotalTermFreq`).
    fn index_text(&mut self, doc_id: DocId, field: u16, text: &str, base: u32) -> (u32, u32, u32) {
        if self.unique.field == Some(field) {
            self.unique.broken = true;
        }
        let mut key = std::mem::take(&mut self.key_buf);
        key.clear();
        key.extend_from_slice(&field.to_le_bytes());
        let analyzer: &crate::analysis::Analyzer = match self.analyzers.get(field as usize) {
            Some(Some(a)) => a,
            _ => &self.default_analyzer,
        };
        let postings = &mut self.postings;
        let mut n = 0u32;
        let mut all = 0u32;
        let mut end = base;
        let mut last_pos: Option<u32> = None;
        analyzer.for_each_token(text, &mut key, |k, pos| {
            // Field length discounts overlaps (tokens stacked on one position).
            if last_pos != Some(pos) {
                n += 1;
            }
            all += 1;
            last_pos = Some(pos);
            let p = base + pos;
            end = end.max(p + 1);
            postings.push_occurrence(k, doc_id, p);
        });
        postings.flush();
        self.key_buf = key;
        (n, end, all)
    }

    /// Post this doc's feature tokens, each once with its largest weight
    /// (a token repeated across values, e.g. an array of sparse vectors,
    /// scores by its maximum as Lucene's `FeatureField` does).
    fn post_features(&mut self, doc_id: DocId) {
        if self.feat_seen.is_empty() {
            return;
        }
        let mut seen = std::mem::take(&mut self.feat_seen);
        let buf = std::mem::take(&mut self.feat_buf);
        let bytes = |e: &(u16, u32, u32, u32)| &buf[e.1 as usize..(e.1 + e.2) as usize];
        if seen.len() > 1 {
            seen.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| bytes(a).cmp(bytes(b))).then(b.3.cmp(&a.3)));
        }
        let mut i = 0;
        while i < seen.len() {
            let e = seen[i];
            self.add_feature_posting(e.0, bytes(&e), doc_id, e.3);
            i += 1;
            while i < seen.len() && seen[i].0 == e.0 && bytes(&seen[i]) == bytes(&e) {
                i += 1;
            }
        }
        seen.clear();
        self.feat_seen = seen;
        self.feat_buf = buf;
        self.feat_buf.clear();
    }

    /// Index `doc` under the feature term `field` + `token` with term
    /// frequency `tf` (no positions).
    fn add_feature_posting(&mut self, field: u16, token: &[u8], doc_id: DocId, tf: u32) {
        let key = &mut self.key_buf;
        key.clear();
        key.extend_from_slice(&field.to_le_bytes());
        key.extend_from_slice(token);
        self.postings.add_doc_tf(key, doc_id, tf);
    }

    /// Index `doc` under the term `field` + `bytes` (no positions, tf 1).
    fn add_key_posting(&mut self, field: u16, bytes: &[u8], doc_id: DocId) {
        if self.unique.field == Some(field) {
            self.unique.add(crate::codec::terms::negative_filter_hash(bytes));
        }
        let key = &mut self.key_buf;
        key.clear();
        key.extend_from_slice(&field.to_le_bytes());
        key.extend_from_slice(bytes);
        self.postings.add_doc(key, doc_id);
    }
}

/// The encoded sections of one segment file.
struct EncodedSections {
    terms: Vec<u8>,
    postings: Vec<u8>,
    fast: Vec<u8>,
    store: Vec<u8>,
    vectors: Vec<u8>,
    extra: Vec<u8>,
    num_docs: u32,
    /// The format version the sections were encoded as.
    version: u32,
}

impl EncodedSections {
    fn refs(&self) -> [&[u8]; 6] {
        [&self.terms, &self.postings, &self.fast, &self.store, &self.vectors, &self.extra]
    }
}

/// Encode every term's postings (sorted by key) and the term dictionary.
/// Contiguous runs of terms are encoded in parallel into separate buffers,
/// then concatenated (offsets rebased), so the bytes equal a serial encode.
/// `inline`: format [`crate::codec::INLINE_POSTINGS_SINCE`]+, single-doc
/// lists without positions go into the dictionary entry.
/// `skip`: a field whose terms are left out (a synthetic id field: sealed
/// segments derive it, see [`crate::synthetic_id`]).
#[allow(clippy::too_many_arguments)]
fn encode_postings(p: &MemPostings, ff: &FastFields, num_docs: u32, impacts: bool, inline: bool, compact: bool, inline_positions: bool, packed: bool, int_keys: bool, skip: Option<u16>) -> (Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    let field_of = |id: u32| { let key = p.key(id); u16::from_le_bytes([key[0], key.get(1).copied().unwrap_or(0)]) };
    let mut ids: Vec<u32> = (0..p.len() as u32).filter(|&id| skip.is_none_or(|f| field_of(id) != f)).collect();
    ids.par_sort_unstable_by(|&a, &b| p.cmp_keys(a, b));
    let chunk = (ids.len() / 64).max(2048);
    let encode_run = |run: &[u32]| {
        let mut out = Vec::new();
        let mut locs = Vec::with_capacity(run.len());
        let mut dec = DecodedPostings::default();
        let mut cur_field = u32::MAX;
        let mut norms = BlockNorms::Absent;
        for &id in run {
            p.decode(id, &mut dec);
            let key = p.key(id);
            let field = u16::from_le_bytes([key[0], key.get(1).copied().unwrap_or(0)]);
            if field as u32 != cur_field {
                cur_field = field as u32;
                norms = match ff.norms().get(&field) {
                    Some(n) => BlockNorms::Bytes(n.bytes()),
                    None => BlockNorms::Absent,
                };
            }
            if let Some(at) = inline_singleton(inline, inline_positions, &dec.doc_ids, &dec.term_freqs, dec.has_positions.then_some(dec.positions.as_slice())) {
                locs.push(PostingsLocation { offset: at, doc_freq: 1, total_term_freq: 1 });
                continue;
            }
            let offset = out.len() as u64;
            let positions = dec.has_positions.then_some(dec.positions.as_slice());
            crate::codec::postings::write_postings_impl(&dec.doc_ids, &dec.term_freqs, positions, norms, impacts, &mut out);
            locs.push(PostingsLocation {
                offset,
                doc_freq: p.doc_freq(id),
                total_term_freq: p.total_term_freq(id),
            });
        }
        (out, locs)
    };
    let parts: Vec<(Vec<u8>, Vec<PostingsLocation>)> = ids.par_chunks(chunk).map(encode_run).collect();
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut postings_bytes: Vec<u8> = Vec::new();
    let mut dict = TermDictWriter::new().tagged(inline).compact(compact).inline_positions(inline_positions).packed(packed).int_keys(packed && int_keys);
    let mut i = 0usize;
    for (bytes, locs) in parts {
        let base = postings_bytes.len() as u64;
        if postings_bytes.is_empty() {
            // The first run's buffer becomes the section (not copied).
            postings_bytes = bytes;
            postings_bytes.reserve(total - postings_bytes.len());
        } else {
            postings_bytes.extend_from_slice(&bytes);
        }
        for loc in locs {
            dict.add(
                p.key(ids[i]),
                &PostingsLocation {
                    offset: if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 { loc.offset } else { base + loc.offset },
                    doc_freq: loc.doc_freq,
                    total_term_freq: loc.total_term_freq,
                },
            );
            i += 1;
        }
    }
    (dict.finish(num_docs), postings_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::RamDirectory;

    #[test]
    fn closed_memtables_are_charged_once_until_dropped() {
        let mut w = SegmentWriter::new(0);
        assert_eq!(w.closed_charge.bytes(), 0, "open memtables are the indexing buffer's");
        w.account_closed();
        assert_eq!(w.closed_charge.bytes(), w.estimated_bytes() as u64, "no vector caches here");
        // Shared by views and seal jobs: one charge, held by the last owner.
        let a = Arc::new(w);
        let b = a.clone();
        drop(a);
        assert_eq!(b.closed_charge.bytes(), b.estimated_bytes() as u64);
    }

    /// `index.codec: best_compression`: the memtable frames stored fields in
    /// large chunks and the seal compresses them with zstd from v26 (LZ4,
    /// same framing, below it); docs read back the same from the memtable
    /// and the segment, and the store is far smaller than the default's.
    #[test]
    fn best_compression_frames_large_zstd_chunks() {
        use crate::codec::store::StoredCompression;
        let schema = vec![Field::new(0, "_source", FieldType::Keyword), Field::new(1, "n", FieldType::I64)];
        let shared: String = (0..150).map(|i| format!("\"fe80::{:x}:{:x}ff:fe{:02x}\",", i * 7919 % 65536, i * 31 % 256, i % 256)).collect();
        let fill = |mode: StoredCompression| {
            let mut w = SegmentWriter::new(0);
            w.set_stored_compression(mode);
            for i in 0..2000u32 {
                let mut d = Document::new();
                d.add_keyword(0, &format!("{{\"t\":{},\"ip\":[{shared}\"x\"],\"v\":{}}}", 1_700_000_000u64 + i as u64 * 10, i * 7 % 101));
                d.add_i64(1, i as i64);
                w.add_document(&d, &schema);
            }
            w
        };
        for version in [crate::codec::ZSTD_STORED_SINCE - 1, crate::codec::ZSTD_STORED_SINCE] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let fast = fill(StoredCompression::Fast).write_to(&RamDirectory::new()).unwrap();
            let best_w = fill(StoredCompression::Best);
            assert_eq!(best_w.stored_compression(), StoredCompression::Best);
            let from_memtable: Vec<_> = (0..2000).step_by(97).map(|d| format!("{:?}", best_w.get_doc(d).unwrap().unwrap().fields)).collect();
            let best = best_w.write_to(&RamDirectory::new()).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            let (fs, bs) = (fast.store().unwrap(), best.store().unwrap());
            assert!(bs.num_chunks() < fs.num_chunks(), "{} best chunks vs {} fast", bs.num_chunks(), fs.num_chunks());
            assert_eq!(bs.any_zstd(), version >= crate::codec::ZSTD_STORED_SINCE, "v{version}");
            assert!(!fs.any_zstd());
            if bs.any_zstd() {
                assert!(best.disk_usage().unwrap().store * 3 < fast.disk_usage().unwrap().store, "zstd blocks compress the repeated values");
            }
            for (k, d) in (0..2000).step_by(97).enumerate() {
                let got = format!("{:?}", best.get_doc(d).unwrap().unwrap().fields);
                assert_eq!(got, format!("{:?}", fast.get_doc(d).unwrap().unwrap().fields), "doc {d}");
                assert_eq!(got, from_memtable[k], "doc {d}");
            }
        }
    }

    fn test_schema() -> Vec<Field> {
        vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "body", FieldType::Text),
            Field::new(2, "price", FieldType::F64),
            Field::new(3, "in_stock", FieldType::Bool),
            Field::new(4, "category", FieldType::Keyword),
        ]
    }

    /// Uninverted fielddata matches between the memtable and the sealed
    /// segment, is cached on the segment, and a field's terms never pick
    /// up another field's: with little-endian key prefixes, field 265
    /// (`[9, 1]`) sorts between fields 9 and 10.
    #[test]
    fn uninverted_fielddata_per_field() {
        let schema = vec![Field::new(9, "a", FieldType::Text), Field::new(265, "b", FieldType::Text), Field::new(255, "c", FieldType::Text)];
        let mut w = SegmentWriter::new(0);
        for (a, b) in [(Some("Bar Quick"), "zzz"), (None, "yyy"), (Some("apple zebra apple"), "xxx")] {
            let mut d = Document::new();
            if let Some(a) = a {
                d.add_text(9, a);
            }
            d.add_text(265, b);
            d.add_text(255, "only");
            w.add_document(&d, &schema);
        }
        let check = |u: &crate::fielddata::Uninverted| {
            let terms: Vec<&[u8]> = (0..u.num_terms()).map(|o| u.term(o)).collect();
            assert_eq!(terms, [&b"apple"[..], b"bar", b"quick", b"zebra"]);
            assert_eq!(u.ords(0), &[1, 2]);
            assert_eq!(u.ords(1), &[] as &[u32]);
            assert_eq!(u.ords(2), &[0, 3]);
        };
        check(&w.uninverted(9).unwrap());
        assert!(Arc::ptr_eq(&w.uninverted(9).unwrap(), &w.uninverted(9).unwrap()), "memtable cache reused while no doc is added");
        assert_eq!(w.uninverted(255).unwrap().num_terms(), 1);
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        assert_eq!(seg.term_dict().range_iter(9, None, None).len(), 4);
        assert_eq!(seg.term_dict().range_iter(255, None, None).len(), 1);
        assert_eq!(seg.term_dict().range_iter(265, None, None).len(), 3);
        assert!(!seg.has_uninverted(9));
        check(&seg.uninverted(9).unwrap());
        assert!(seg.has_uninverted(9));
        assert!(Arc::ptr_eq(&seg.uninverted(9).unwrap(), &seg.uninverted(9).unwrap()));
        assert_eq!(seg.uninverted(265).unwrap().term(0), b"xxx");
    }

    /// A long postings list of a memtable is encoded once per state of
    /// the list, not per read: repeated reads share the cached bytes, and
    /// documents added later are seen (the doc freq keys the cache).
    #[test]
    fn long_memtable_postings_are_encoded_once_per_state() {
        let schema = vec![Field::new(9, "a", FieldType::Text)];
        let mut w = SegmentWriter::new(0);
        let add = |w: &mut SegmentWriter, n: u32| for _ in 0..n {
            let mut d = Document::new();
            d.add_text(9, "common word");
            w.add_document(&d, &schema);
        };
        add(&mut w, ENCODED_CACHE_MIN_DOCS + 10);
        let term = Term::new_text(9, "common");
        let first = collect(w.postings(&term).unwrap());
        assert_eq!(first.len() as u32, ENCODED_CACHE_MIN_DOCS + 10);
        let cached = w.encoded.lock().by_term.values().next().map(|(_, b)| b.as_slice().as_ptr()).expect("the long list was cached");
        let again = collect(w.postings(&term).unwrap());
        assert_eq!(again, first);
        assert_eq!(w.encoded.lock().by_term.values().next().map(|(_, b)| b.as_slice().as_ptr()), Some(cached), "re-encoded on a repeated read");
        add(&mut w, 5);
        assert_eq!(collect(w.postings(&term).unwrap()).len() as u32, ENCODED_CACHE_MIN_DOCS + 15, "a stale cached list was read");
        // The cache is part of the memtable's memory (the indexing buffer,
        // stalls and breakers see it).
        let cached = w.encoded.lock().bytes;
        assert!(cached > 0);
        let mut plain = SegmentWriter::new(2);
        add(&mut plain, ENCODED_CACHE_MIN_DOCS + 15);
        assert_eq!(w.estimated_bytes(), plain.estimated_bytes() + cached, "the cached lists are not counted");
        // Short lists are not cached.
        let mut short = SegmentWriter::new(1);
        add(&mut short, 3);
        assert_eq!(collect(short.postings(&term).unwrap()).len(), 3);
        assert!(short.encoded.lock().by_term.is_empty());
    }

    fn collect(mut p: PostingsReader) -> Vec<(DocId, u32)> {
        let mut v = Vec::new();
        while let Some(d) = p.next() {
            v.push((d, p.term_freq().unwrap()));
        }
        v
    }

    /// A flush under a lock that every global rayon worker is blocked on
    /// (an ES layer's parallel bulk writes waiting for the shard lock a
    /// refresh holds) must still complete: encoding has its own pool.
    #[test]
    fn flush_does_not_need_the_global_rayon_pool() {
        let _test_gate = crate::global_rayon_test_gate();
        let schema = test_schema();
        let lock = Arc::new(parking_lot::Mutex::new(()));
        let guard = lock.lock();
        let busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let n = rayon::current_num_threads();
        for _ in 0..n {
            let (lock, busy) = (lock.clone(), busy.clone());
            rayon::spawn(move || {
                busy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(lock.lock());
            });
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while busy.load(std::sync::atomic::Ordering::SeqCst) < n {
            if std::time::Instant::now() >= deadline {
                drop(guard);
                panic!("global Rayon blockers did not start");
            }
            std::thread::yield_now();
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let dir = RamDirectory::new();
            let mut w = SegmentWriter::new(0);
            for i in 0..50 {
                let mut doc = Document::new();
                doc.add_text(0, &format!("doc {i}"));
                doc.add_keyword(4, "k");
                w.add_document(&doc, &schema);
            }
            let _ = tx.send(w.write_to(&dir).map(|s| s.num_docs));
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(20));
        drop(guard);
        assert_eq!(got.expect("flush deadlocked on the global rayon pool").unwrap(), 50);
    }

    /// The same from a worker of the global pool that holds a lock other
    /// queued global jobs need (a bulk write flushing a shard under its
    /// write lock while more writes to that shard wait): the worker must
    /// not steal those jobs while its flush encodes.
    #[test]
    fn flush_from_a_global_worker_does_not_steal_lock_waiters() {
        let _test_gate = crate::global_rayon_test_gate();
        let schema = test_schema();
        let lock = Arc::new(parking_lot::Mutex::new(()));
        let (tx, rx) = std::sync::mpsc::channel();
        let l = lock.clone();
        rayon::spawn(move || {
            let _held = l.lock();
            // Jobs that need the held lock, queued where this worker could steal them.
            for _ in 0..4 * rayon::current_num_threads() {
                let l = l.clone();
                rayon::spawn(move || drop(l.lock()));
            }
            let dir = RamDirectory::new();
            let mut w = SegmentWriter::new(0);
            for i in 0..2_000 {
                let mut doc = Document::new();
                doc.add_text(0, &format!("doc {i} some words to encode"));
                doc.add_keyword(4, "k");
                w.add_document(&doc, &schema);
            }
            let _ = tx.send(w.write_to(&dir).map(|s| s.num_docs));
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(20));
        assert_eq!(got.expect("flush on a global worker deadlocked").unwrap(), 2_000);
        drop(lock);
    }

    #[test]
    fn stored_chunk_charge_follows_reader_after_cache_eviction() {
        let schema=test_schema();
        let dir=RamDirectory::new();
        let mut writer=SegmentWriter::new(0);
        let mut doc=Document::new();doc.add_text(0,"cached payload");
        writer.add_document(&doc,&schema);
        let segment=writer.write_to(&dir).unwrap();
        segment.get_doc(0).unwrap().unwrap();
        let retained=segment.store_chunks.lock()[0].1.clone();
        let weak=Arc::downgrade(&retained);
        assert!(retained.charge.bytes()>=retained.capacity() as u64);
        segment.store_chunks.lock().clear();
        assert_eq!(Arc::strong_count(&retained),1);
        assert!(weak.upgrade().is_some());
        drop(retained);
        assert!(weak.upgrade().is_none(),"last reader releases both bytes and charge");
    }

    #[test]
    fn guarded_stored_decode_rejects_before_copy_and_preserves_document() {
        let schema=test_schema();
        let dir=RamDirectory::new();
        let mut writer=SegmentWriter::new(0);
        let mut doc=Document::new();
        doc.add_text(0,&"payload".repeat(10000));
        writer.add_document(&doc,&schema);
        let mut seen=0;
        let denied=writer.get_doc_guarded(0,|bytes|{seen=bytes;Err::<(),_>(crate::RusticError::Memory("limit".into()))});
        assert!(matches!(denied.unwrap_err(),crate::RusticError::Memory(_)));
        assert!(seen>70000);
        let allowed=writer.get_doc_guarded(0,|_|Ok::<(),crate::RusticError>(())).unwrap().unwrap();
        assert_eq!(allowed.get(0).unwrap().as_text(),doc.get(0).unwrap().as_text());
        let segment=writer.write_to(&dir).unwrap();
        let denied=segment.get_doc_guarded(0,|_|Err::<(),_>(crate::RusticError::Memory("limit".into())));
        assert!(matches!(denied.unwrap_err(),crate::RusticError::Memory(_)));
        assert!(segment.store_chunks.lock().is_empty());
        let allowed=segment.get_doc_guarded(0,|_|Ok::<(),crate::RusticError>(())).unwrap().unwrap();
        assert_eq!(allowed.get(0).unwrap().as_text(),doc.get(0).unwrap().as_text());
        // Exercise a pre-existing shared cache too; decode still preflights.
        segment.get_doc(0).unwrap().unwrap();
        assert!(matches!(segment.get_doc_guarded(0,|_|Err::<(),_>(crate::RusticError::Memory("limit".into()))).unwrap_err(),crate::RusticError::Memory(_)));
    }

    #[test]
    fn stored_decode_retains_its_decoded_size_and_preflight_covers_the_worst_case() {
        let schema=test_schema();
        let dir=RamDirectory::new();
        let mut writer=SegmentWriter::new(0);
        let mut article=Document::new();
        article.add_text(0,&"wikipedia ".repeat(2000)); // a 20 KB article
        writer.add_document(&article,&schema);
        let segment=writer.write_to(&dir).unwrap();
        let budget=crate::admission::Budget::new(1<<30);
        let doc=segment.get_doc_admitted(0,&budget).unwrap().unwrap();
        // Was 256x the encoded size (about 5 MB) for the document's lifetime.
        assert!(budget.used()>=20_000 && budget.used()<24_000,"retained {}",budget.used());
        assert_eq!(budget.used(),doc.decoded_bytes()+256);
        drop(doc);
        assert_eq!(budget.used(),0);
        // The densest encoding (1-byte bools) decodes within the preflight,
        // including serde's doubling past its 1 MiB slot preallocation.
        for n in [1usize,1000,100_000] {
            let mut bools=Document::new();
            for i in 0..n {bools.fields.push((i as u16,crate::types::FieldValue::Bool(true)));}
            let encoded=bincode::serialize(&bools).unwrap();
            let decoded:Document=bincode::deserialize(&encoded).unwrap();
            assert!(Document::decode_peak_bytes(encoded.len() as u64)>=3*decoded.decoded_bytes()/2,"{n} bools");
        }
    }

    #[test]
    fn get_doc_through_chunk_cache_matches_store() {
        let schema = test_schema();
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..700u32 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("title {i} {}", "x".repeat(400 + (i % 50) as usize)));
            writer.add_document(&doc, &schema);
        }
        let seg = writer.write_to(&dir).unwrap();
        let store = seg.store().unwrap();
        assert!(store.num_chunks() > 4, "{} chunks", store.num_chunks());
        // Sequential, backwards and scattered fetches all agree with the store.
        let order: Vec<DocId> = (0..700).chain((0..700).rev()).chain((0..700).map(|i| (i * 337) % 700)).collect();
        for d in order {
            assert_eq!(format!("{:?}", seg.get_doc(d).unwrap()), format!("{:?}", store.get(d).unwrap()), "doc {d}");
        }
        assert!(seg.get_doc(700).unwrap().is_none());
        // Reading a chunk's docs in order decodes a prefix once, then the
        // whole chunk once: the cache then holds all of it.
        seg.store_chunks.lock().clear();
        seg.store_partial.lock().clear();
        let (first, n) = store.chunk_docs(1);
        assert!(n > 2);
        seg.get_doc(first).unwrap().unwrap();
        let cached = |seg: &Segment| seg.store_chunks.lock().iter().find(|(c, _)| *c == 1).map(|(_, raw)| raw.len());
        assert!(cached(&seg).unwrap() < store.decompress(1).unwrap().len());
        seg.get_doc(first + 1).unwrap().unwrap();
        assert_eq!(cached(&seg), Some(store.decompress(1).unwrap().len()));
        for d in first..first + n {
            assert_eq!(format!("{:?}", seg.get_doc(d).unwrap()), format!("{:?}", store.get(d).unwrap()), "doc {d}");
        }
        // Three readers interleaved over three chunks (more than the cache
        // holds): each chunk's second visit still decodes all of it.
        seg.store_chunks.lock().clear();
        seg.store_partial.lock().clear();
        let chunks = [1usize, 2, 3].map(|c| store.chunk_docs(c));
        for (s, _) in chunks {
            seg.get_doc(s).unwrap().unwrap();
        }
        for (c, (s, _)) in [1usize, 2, 3].into_iter().zip(chunks) {
            seg.get_doc(s + 1).unwrap().unwrap();
            let len = seg.store_chunks.lock().iter().find(|(i, _)| *i == c).map(|(_, r)| r.len());
            assert_eq!(len, Some(store.decompress(c).unwrap().len()), "chunk {c}");
        }
    }

    /// From v16 single-doc postings lists (doc freq 1, tf 1, no positions)
    /// live in the term dictionary: lookups, cursors and merges read the
    /// same postings, the postings section holds none of them, and lists
    /// with more docs, a tf above 1 or positions keep their bytes.
    #[test]
    fn single_doc_postings_are_inline_from_v15() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword), Field::new(1, "tag", FieldType::Keyword), Field::new(2, "body", FieldType::Text)];
        let dir = RamDirectory::new();
        let build = |seg: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(seg);
            for i in 0..500u32 {
                let mut doc = Document::new();
                doc.add_keyword(0, &format!("id-{seg}-{i}"));
                doc.add_keyword(1, if i % 2 == 0 { "even" } else { "odd" });
                doc.add_text(2, &format!("word{i} common"));
                w.add_document(&doc, &schema);
            }
            let s = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(s)
        };
        let docs = |s: &Segment, t: &Term| -> Vec<DocId> { let mut p = s.postings(t).unwrap(); let mut v = Vec::new(); while let Some(d) = p.next() { v.push(d); } v };
        let (old, new) = (build(1, 15), build(2, 16));
        assert!(old.disk_usage().unwrap().fields[&0].postings > 0);
        assert_eq!(new.disk_usage().unwrap().fields.get(&0).map_or(0, |u| u.postings), 0, "every _id list inline");
        for (s, seg) in [(&old, 1u32), (&new, 2)] {
            for i in [0u32, 7, 499] {
                assert_eq!(docs(s, &Term::new_text(0, &format!("id-{seg}-{i}"))), vec![i]);
            }
            assert_eq!(docs(s, &Term::new_text(1, "even")).len(), 250);
            assert_eq!(docs(s, &Term::new_text(2, "common")).len(), 500);
            // A single-doc text term has positions: not inline.
            let p = s.postings(&Term::new_text(2, "word3")).unwrap();
            assert!(p.has_positions());
        }
        let mut seen = 0;
        let mut cursor = new.term_dict().cursor_from(&0u16.to_le_bytes());
        while cursor.advance().unwrap() {
            if cursor.key()[..2] != 0u16.to_le_bytes() { break; }
            let p = new.postings_at(&cursor.loc()).unwrap();
            assert_eq!(p.doc_freq(), 1);
            seen += 1;
        }
        assert_eq!(seen, 500);
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(16));
        let merged = crate::merge::merge_segments(&dir, &[old.clone(), new.clone()], 9).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        assert_eq!(merged.disk_usage().unwrap().fields.get(&0).map_or(0, |u| u.postings), 0);
        assert_eq!(docs(&merged, &Term::new_text(0, "id-1-7")), vec![7]);
        assert_eq!(docs(&merged, &Term::new_text(0, "id-2-7")), vec![507]);
        assert_eq!(docs(&merged, &Term::new_text(1, "odd")).len(), 500);
    }

    /// From v19 a single-doc list of a positional (text) field with term
    /// frequency 1 is inline too, with its one position: lookups, phrase
    /// positions, cursors and merges (also with v18 inputs, and into v18
    /// outputs, which write the list again) read the same postings.
    #[test]
    fn single_doc_text_postings_are_inline_with_their_position_from_v19() {
        let schema = vec![Field::new(0, "body", FieldType::Text)];
        let dir = RamDirectory::new();
        let build = |seg: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(seg);
            for i in 0..400u32 {
                let mut doc = Document::new();
                // `common` everywhere, a unique word at position 1, and a
                // word twice in one doc (tf 2: not inline).
                doc.add_text(0, &format!("common u{seg}x{i} pair pair"));
                w.add_document(&doc, &schema);
            }
            let s = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(s)
        };
        let read = |s: &Segment, t: &str| -> Vec<(DocId, Vec<u32>)> {
            let Some(mut p) = s.postings(&Term::new_text(0, t)) else { return Vec::new() };
            let mut v = Vec::new();
            while let Some(d) = p.next() { v.push((d, p.positions().to_vec())); }
            v
        };
        let (old, new) = (build(1, 18), build(2, 19));
        let (pu_old, pu_new) = (old.disk_usage().unwrap().fields[&0].postings, new.disk_usage().unwrap().fields[&0].postings);
        assert!(pu_new * 2 < pu_old, "unique words' lists inline: {pu_new} vs {pu_old}");
        for (s, seg) in [(&old, 1u32), (&new, 2)] {
            for i in [0u32, 9, 399] {
                assert_eq!(read(s, &format!("u{seg}x{i}")), vec![(i, vec![1])]);
            }
            assert_eq!(read(s, "common").len(), 400);
            assert!(read(s, "pair").iter().all(|(_, p)| p == &vec![2, 3]));
        }
        let mut cursor = new.term_dict().cursor_from(&0u16.to_le_bytes());
        let mut inline = 0;
        while cursor.advance().unwrap() {
            if cursor.loc().offset & crate::term_dictionary::INLINE_POSTINGS != 0 {
                assert_eq!(crate::term_dictionary::inline_position(cursor.loc().offset), Some(1));
                inline += 1;
            }
        }
        assert_eq!(inline, 400);
        for (version, id) in [(19u32, 20u32), (18, 21)] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let merged = crate::merge::merge_segments(&dir, &[old.clone(), new.clone()], id).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            assert_eq!(read(&merged, "u1x7"), vec![(7, vec![1])], "v{version}");
            assert_eq!(read(&merged, "u2x7"), vec![(407, vec![1])], "v{version}");
            assert_eq!(read(&merged, "common").len(), 800);
            let has_inline_pos = {
                let mut c = merged.term_dict().cursor_from(&0u16.to_le_bytes());
                let mut any = false;
                while c.advance().unwrap() { any |= crate::term_dictionary::inline_position(c.loc().offset).is_some(); }
                any
            };
            assert_eq!(has_inline_pos, version >= 19, "v{version}");
        }
    }

    /// From v15 `_id`-like keyword fields (`column_from_postings`) get no
    /// column: a doc's value reads from its stored fields, and a whole
    /// column is derived from the postings (sorted terms, every doc's
    /// ordinal), also after merging v14 (with a column) and v15 segments.
    #[test]
    fn keyword_from_postings_reads_stored_values_and_derives_a_column() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword).with_column_from_postings(true), Field::new(1, "n", FieldType::I64)];
        let dir = RamDirectory::new();
        let id = |seg: u32, i: u32| format!("{:04}-{seg}", (i * 7919) % 1000);
        let build = |seg: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(seg);
            for i in 0..300u32 {
                let mut doc = Document::new();
                doc.add_keyword(0, &id(seg, i));
                doc.add_i64(1, i as i64);
                w.add_document(&doc, &schema);
            }
            let s = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(s)
        };
        let check = |s: &Segment, parts: &[(u32, u32)]| {
            let leaf = crate::query::reader::LeafReader::Segment(s);
            let expected: Vec<String> = parts.iter().flat_map(|&(seg, n)| (0..n).map(move |i| id(seg, i))).collect();
            assert_eq!(s.num_docs as usize, expected.len());
            let column = s.keyword_reader(0).expect("a column, stored or derived");
            let mut terms: Vec<&String> = expected.iter().collect();
            terms.sort();
            terms.dedup();
            assert_eq!(column.num_terms(), terms.len());
            for (d, v) in expected.iter().enumerate() {
                assert_eq!(leaf.keyword_value(0, d as DocId).as_deref(), Some(v.as_str()), "doc {d}");
                assert_eq!(column.get(d as DocId).as_deref(), Some(v.as_str()), "doc {d}");
                assert_eq!(column.ord(d as DocId) as usize, terms.binary_search(&v).unwrap());
            }
        };
        let (old, new) = (build(1, 14), build(2, 15));
        assert!(!old.keyword_from_postings(0) && new.keyword_from_postings(0));
        assert!(new.fast_fields().keyword_column(0).is_none(), "no column from v15");
        assert!(old.fast_fields().keyword_column(0).is_some());
        let du = new.disk_usage().unwrap();
        assert_eq!(du.fields[&0].doc_values + du.fields[&0].doc_values_terms, 0);
        check(&old, &[(1, 300)]);
        check(&new, &[(2, 300)]);
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(15));
        let merged = crate::merge::merge_segments(&dir, &[old.clone(), new.clone()], 9).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        assert!(merged.keyword_from_postings(0) && merged.fast_fields().keyword_column(0).is_none());
        check(&merged, &[(1, 300), (2, 300)]);
    }

    /// From v15 a column-stored field is left out of the stored fields and
    /// restored from its column on fetch (memtable and sealed segment, and
    /// through merges of v14 and v15 segments), never duplicated.
    #[test]
    fn column_stored_fields_come_back_from_their_column() {
        let schema = vec![Field::new(0, "id", FieldType::Keyword), Field::new(1, "_version", FieldType::I64).with_column_stored(true)];
        let dir = RamDirectory::new();
        let build = |id: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(id);
            for i in 0..200i64 {
                let mut doc = Document::new();
                doc.add_keyword(0, &format!("d{id}-{i}"));
                doc.add_i64(1, i * 3 + id as i64);
                w.add_document(&doc, &schema);
            }
            if version >= crate::codec::COLUMN_STORED_SINCE {
                let got = w.get_doc(7).unwrap().unwrap();
                assert_eq!(got.get(1).and_then(|v| v.as_i64()), Some(21 + id as i64), "memtable");
            }
            let seg = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(seg)
        };
        let (old, new) = (build(1, 14), build(2, 15));
        assert!(old.disk_usage().unwrap().fields[&1].stored_raw > 0);
        assert_eq!(new.disk_usage().unwrap().fields.get(&1).map_or(0, |u| u.stored_raw), 0, "not stored from v15");
        let check = |seg: &Segment, base: i64, first: u32| {
            for d in 0..200u32 {
                let doc = seg.get_doc(first + d).unwrap().unwrap();
                assert_eq!(doc.fields.iter().filter(|(f, _)| *f == 1).count(), 1, "doc {d}: {:?}", doc.fields);
                assert_eq!(doc.get(1).and_then(|v| v.as_i64()), Some(d as i64 * 3 + base));
            }
        };
        check(&old, 1, 0);
        check(&new, 2, 0);
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(15));
        let merged = crate::merge::merge_segments(&dir, &[old.clone(), new.clone()], 9).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        check(&merged, 1, 0);
        check(&merged, 2, 200);
    }

    /// `cargo test -p rustic-core --release --lib metricbeat_memtable_microbench -- --ignored --nocapture`
    /// (`MB_BENCH_DOCS`, default 200k): metricbeat-shaped docs (300 sparse
    /// fields, about 30 present per doc: keywords of a few values, numbers)
    /// added to a memtable and sealed, at the current and the v14+ format.
    #[test]
    #[ignore]
    fn metricbeat_memtable_microbench() {
        let n: u32 = std::env::var("MB_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
        let mut schema: Vec<Field> = Vec::new();
        for f in 0..300u16 {
            let mut field = Field::new(f, &format!("f{f}"), if f % 2 == 0 { FieldType::Keyword } else { FieldType::F64 });
            field.stored = false;
            schema.push(field);
        }
        for version in [13u32, 16] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let dir = RamDirectory::new();
            let mut w = SegmentWriter::new(1);
            let mut seed = 7u64;
            let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed >> 33 };
            let t = std::time::Instant::now();
            for _ in 0..n {
                let mut doc = Document::new();
                // A metricset's ~30 fields out of 300.
                let set = next() % 10;
                for k in 0..30u64 {
                    let f = ((set * 30 + k) % 300) as u16;
                    if f % 2 == 0 { doc.add_keyword(f, &format!("v{}", next() % 8)); } else { doc.add_f64(f, (next() % 10_000) as f64 / 10.0); }
                }
                w.add_document(&doc, &schema);
            }
            let added = t.elapsed();
            let t = std::time::Instant::now();
            let seg = w.write_to(&dir).unwrap();
            let sealed = t.elapsed();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            eprintln!("v{version}: add {:.0} docs/s, seal {:.1} ms, {} bytes ({n} docs)", n as f64 / added.as_secs_f64(), sealed.as_secs_f64() * 1e3, seg.segment_bytes());
        }
    }

    /// Disk usage attributes every byte of every section to a field or to
    /// the framing entry, and each structure to the field that owns it.
    #[test]
    fn disk_usage_accounts_every_section_byte() {
        let schema = test_schema();
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..1000 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("title {i}"));
            doc.add_text(1, "a body of text");
            doc.add_f64(2, i as f64 * 0.5);
            doc.add_keyword(4, if i % 3 == 0 { "books" } else { "music" });
            writer.add_document(&doc, &schema);
        }
        let segment = writer.write_to(&dir).unwrap();
        let du = segment.disk_usage().unwrap();
        let mut sum = FieldDiskUsage::default();
        for u in du.fields.values() { sum.add(u); }
        assert_eq!(sum.terms, du.terms);
        assert_eq!(sum.postings, du.postings);
        assert_eq!(sum.doc_values + sum.doc_values_terms + sum.points + sum.norms, du.fast);
        // Each field's share of a chunk rounds down: at most a byte per
        // field per chunk is unattributed.
        assert!(sum.stored <= du.store && du.store - sum.stored <= 128, "{} of {}", sum.stored, du.store);
        assert_eq!(sum.geo + sum.shape + sum.other_extra, du.extra);
        let price = &du.fields[&2];
        assert!(price.points > 0 && price.doc_values > 0, "{price:?}");
        assert_eq!(price.terms + price.postings, 0);
        let category = &du.fields[&4];
        assert_eq!(category.term_count, 2);
        assert!(category.doc_values_terms > 0 && category.postings > 0, "{category:?}");
        assert!(du.fields[&0].norms > 0 && du.fields[&0].stored > 0);
        assert_eq!(du.fields[&0].term_count, 1001, "`title` and 1000 numbers");
    }

    /// The current format (binary `_id`, packed terms): each stored field
    /// is attributed its own encoded bytes (an id its 15, not 20), so the
    /// split never exceeds the store and the framing stays small; sampling
    /// a few chunks gives the same split within a few percent.
    #[test]
    fn disk_usage_attributes_binary_ids_and_samples_chunks() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword), Field::new(1, "body", FieldType::Text), Field::new(2, "n", FieldType::I64)];
        let dir = RamDirectory::new();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
        let mut writer = SegmentWriter::new(0);
        for i in 0..20_000u64 {
            let mut doc = Document::new();
            doc.add_keyword(0, &format!("AaG{:016x}x", i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            doc.add_text(1, &format!("{{\"message\":\"GET /english/images/{} HTTP/1.0\",\"n\":{i}}}", i % 97));
            doc.add_i64(2, i as i64);
            writer.add_document(&doc, &schema);
        }
        let segment = writer.write_to(&dir).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        let exact = segment.disk_usage().unwrap();
        let mut sum = FieldDiskUsage::default();
        for u in exact.fields.values() { sum.add(u); }
        assert!(sum.stored <= exact.store + 64, "{} of {}", sum.stored, exact.store);
        let framing = exact.fields.get(&u16::MAX).map_or(0, |f| f.stored_raw);
        assert!(framing < 20_000 * 4, "framing {framing}");
        assert!(exact.fields[&0].stored_raw <= 20_000 * 18, "id {} raw bytes", exact.fields[&0].stored_raw);
        assert!(segment.store().unwrap().num_chunks() > 8);
        let sampled = segment.disk_usage_sampled(4).unwrap();
        for f in [0u16, 1] {
            let (a, b) = (exact.fields[&f].stored as f64, sampled.fields[&f].stored as f64);
            assert!((a - b).abs() <= a * 0.05, "field {f}: exact {a} sampled {b}");
        }
        assert_eq!((sampled.terms, sampled.postings, sampled.fast, sampled.store), (exact.terms, exact.postings, exact.fast, exact.store));
    }

    #[test]
    fn v18_writes_compact_stored_docs() {
        let mut schema = test_schema();
        for f in &mut schema { if f.id == 1 { f.stored = false; } }
        let doc_of = |i: i64| {
            let mut doc = Document::new();
            doc.add_text(0, &format!("title {i}"));
            doc.add_text(1, "a body of text");
            doc.add_f64(2, i as f64 * 0.5);
            doc.add_keyword(4, if i % 3 == 0 { "books" } else { "music" });
            doc
        };
        let fields = |d: &Document| format!("{:?}", d.fields.iter().filter(|(f, _)| *f != 1).collect::<Vec<_>>());
        // (version while docs are added, version at seal)
        let build = |add: u32, seal: u32| {
            let dir = RamDirectory::new();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(add));
            let mut writer = SegmentWriter::new(0);
            for i in 0..1000 { writer.add_document(&doc_of(i), &schema); }
            for i in [0u32, 500, 999] {
                assert_eq!(fields(&writer.get_doc(i).unwrap().unwrap()), fields(&doc_of(i as i64)), "memtable v{add}");
                assert!(writer.get_doc(i).unwrap().unwrap().fields.iter().all(|(f, _)| *f != 1), "unstored field kept");
            }
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(seal));
            let seg = writer.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            (dir, seg)
        };
        let ((_d1, old), (_d2, new), (_d3, held)) = (build(17, 17), build(18, 18), build(18, 17));
        let compact = |s: &Segment| { let r = s.store().unwrap(); (0..r.num_chunks()).map(|c| r.chunk_compact(c)).collect::<Vec<_>>() };
        assert!(compact(&old).iter().all(|&c| !c));
        assert!(compact(&new).iter().all(|&c| c));
        assert!(compact(&held).iter().all(|&c| !c), "a pre-v18 seal re-encodes compact chunks");
        assert!(new.store_data.len() < old.store_data.len(), "{} vs {}", new.store_data.len(), old.store_data.len());
        for seg in [&old, &new, &held] {
            for i in 0..1000u32 {
                assert_eq!(fields(&seg.get_doc(i).unwrap().unwrap()), fields(&doc_of(i as i64)), "doc {i}");
            }
            let du = seg.disk_usage().unwrap();
            let stored: u64 = du.fields.values().map(|u| u.stored).sum();
            assert!(stored <= du.store && du.store - stored <= 128, "{stored} of {}", du.store);
            assert!(du.fields[&0].stored > 0 && du.fields.get(&1).is_none_or(|u| u.stored == 0));
        }
    }

    #[test]
    fn unstored_fields_are_indexed_but_not_stored() {
        let mut schema = test_schema();
        for f in &mut schema {
            f.stored = f.id == 0;
        }
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        let mut doc = Document::new();
        doc.add_text(0, "Rust Programming");
        doc.add_text(1, "A systems programming language");
        doc.add_f64(2, 49.99);
        doc.add_keyword(4, "books");
        writer.add_document(&doc, &schema);
        let got = writer.get_doc(0).unwrap().unwrap();
        assert_eq!(got.fields.len(), 1);
        assert_eq!(got.get(0).unwrap().as_text(), Some("Rust Programming"));

        let segment = writer.write_to(&dir).unwrap();
        let got = segment.get_doc(0).unwrap().unwrap();
        assert_eq!(got.fields.len(), 1);
        assert_eq!(got.get(0).unwrap().as_text(), Some("Rust Programming"));
        assert_eq!(segment.postings(&Term::new_text(1, "systems")).unwrap().doc_freq(), 1);
        assert_eq!(segment.postings(&Term::new_text(4, "books")).unwrap().doc_freq(), 1);
        assert_eq!(segment.fast_fields().get_f64(2).unwrap().get(0), Some(49.99));
    }

    #[test]
    fn write_and_read_segment() {
        let schema = test_schema();
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);

        let mut doc = Document::new();
        doc.add_text(0, "Rust Programming");
        doc.add_text(1, "A systems programming language");
        doc.add_f64(2, 49.99);
        doc.add_bool(3, true);
        doc.add_keyword(4, "books");
        writer.add_document(&doc, &schema);

        let mut doc2 = Document::new();
        doc2.add_text(0, "Learning Rust");
        doc2.add_text(1, "Rust is fun and fast");
        doc2.add_f64(2, 29.99);
        doc2.add_bool(3, false);
        doc2.add_keyword(4, "books");
        writer.add_document(&doc2, &schema);

        let segment = writer.write_to(&dir).unwrap();
        assert_eq!(segment.num_docs, 2);
        assert_eq!(segment.num_live_docs(), 2);
        assert!(dir.exists("seg_00000000.rstc"));

        assert_eq!(collect(segment.postings(&Term::new_text(0, "rust")).unwrap()), vec![(0, 1), (1, 1)]);
        assert_eq!(collect(segment.postings(&Term::new_text(1, "rust")).unwrap()), vec![(1, 1)]);
        assert_eq!(segment.postings(&Term::new_text(4, "books")).unwrap().doc_freq(), 2);
        assert!(segment.postings(&Term::new_text(0, "python")).is_none());
        assert_eq!(segment.term_dict().doc_freq(&Term::new_text(0, "rust")), 2);

        let ff = segment.fast_fields();
        assert_eq!(ff.get_f64(2).unwrap().get(0), Some(49.99));
        assert_eq!(ff.get_bool(3).unwrap().count_true(), 1);
        assert_eq!(ff.get_keyword(4).unwrap().get(1).as_deref(), Some("books"));

        let got = segment.get_doc(1).unwrap().unwrap();
        assert_eq!(got.get(0).unwrap().as_text(), Some("Learning Rust"));
        assert!(segment.get_doc(2).unwrap().is_none());
    }

    /// Lucene's collection statistics: `docCount` counts docs with the
    /// field, `sumTotalTermFreq` counts every token (stacked ones too) and,
    /// for keywords, one per distinct value per doc; identical in the
    /// memtable and the segment it flushes to. A pre-v9 segment derives the
    /// same numbers from its term dictionary.
    #[test]
    fn field_stats_count_overlaps_and_distinct_keywords() {
        let schema = vec![
            Field::new(0, "syn", FieldType::Text).with_analyzer("_pretokenized"),
            Field::new(1, "kw", FieldType::Keyword),
            Field::new(2, "nn", FieldType::Text).with_norms(false),
            Field::new(3, "n", FieldType::I64),
        ];
        let mut w = SegmentWriter::new(0);
        let mut d = Document::new();
        // 3 tokens on 2 positions: the norm length is 2, sumTotalTermFreq 3.
        d.add_text(0, "0\u{1}a\u{2}0\u{1}b\u{2}1\u{1}c");
        d.add_i64(3, 7);
        w.add_document(&d, &schema);
        let mut d = Document::new();
        d.add_keyword(1, "x");
        d.add_keyword(1, "x");
        d.add_keyword(1, "y");
        w.add_document(&d, &schema);
        let mut d = Document::new();
        d.add_keyword(1, "x");
        d.add_text(2, "hello hello world");
        w.add_document(&d, &schema);
        w.delete_doc(2); // deleted docs stay counted until a merge

        let check = |agg: &dyn Fn(u16) -> Option<FieldAgg>| {
            assert_eq!(agg(0), Some(FieldAgg { doc_count: 1, sum_ttf: 3 }));
            assert_eq!(agg(1), Some(FieldAgg { doc_count: 2, sum_ttf: 3 }));
            assert_eq!(agg(2), Some(FieldAgg { doc_count: 1, sum_ttf: 3 }));
            assert_eq!(agg(3), None);
            assert_eq!(agg(9), None);
        };
        check(&|f| w.field_agg(f));
        assert_eq!(w.norms_for(0).len(0), 2);
        assert!(w.norms_for(2).is_unit(), "norms: false keeps no norms");
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        check(&|f| seg.field_agg(f));
        assert_eq!(seg.field_doc_count(1), Some(2));

        // The same segment rewritten as format v8 (no stats section) derives them.
        let file = open_segment_file(dir.open_read(&segment_file_name(0)).unwrap(), false).unwrap();
        let kept: Vec<(u8, u16, Vec<u8>)> =
            decode_extra(file.extra.as_slice()).unwrap().into_iter().filter(|(t, _, _)| *t != EXTRA_FIELD_STATS).map(|(t, f, b)| (t, f, b.to_vec())).collect();
        let mut bytes = write_segment_file_with_extra(
            file.terms.as_slice(),
            file.postings.as_slice(),
            file.fast.as_slice(),
            file.store.as_slice(),
            file.vectors.as_slice(),
            &encode_extra(&kept),
            file.num_docs,
        );
        bytes[4..8].copy_from_slice(&8u32.to_le_bytes());
        dir.write(&segment_file_name(1), &bytes).unwrap();
        let old = Segment::open(&dir, 1).unwrap();
        assert_eq!(old.format_version(), 8);
        // (a numeric field's derived numbers are never read: nothing scores it)
        check(&|f| if f == 3 { None } else { old.field_agg(f) });
    }

    /// A segment written before format v8 opens (never refused) with
    /// `legacy_presence` set; a current one does not.
    #[test]
    fn old_format_segment_opens_with_legacy_presence() {
        let schema = vec![Field::new(0, "title", FieldType::Text)];
        let mut w = SegmentWriter::new(0);
        let mut doc = Document::new();
        doc.add_text(0, "x");
        w.add_document(&doc, &schema);
        let mut bytes = w.encode();
        let dir = RamDirectory::new();
        dir.write(&segment_file_name(0), &bytes).unwrap();
        let cur = Segment::open(&dir, 0).unwrap();
        assert_eq!(cur.format_version(), crate::codec::write_version());
        assert!(!cur.legacy_presence());
        // Rewrite the header version as v7 (the body layout is compatible).
        bytes[4..8].copy_from_slice(&7u32.to_le_bytes());
        dir.write(&segment_file_name(1), &bytes).unwrap();
        let old = Segment::open(&dir, 1).unwrap();
        assert_eq!(old.format_version(), 7);
        assert!(old.legacy_presence());
        assert_eq!(old.num_docs, 1);
    }

    /// A segment's header and its postings carry the version the writer
    /// captured, not whatever the live write version says meanwhile (the
    /// cluster floor rising mid-seal must not split a segment's versions).
    #[test]
    fn captured_write_version_governs_header_and_postings() {
        let schema = vec![Field::new(0, "title", FieldType::Text)];
        let build = || {
            let mut w = SegmentWriter::new(0);
            for i in 0..400u32 {
                let mut doc = Document::new();
                let filler = ["x"; 7][..(i % 7) as usize].join(" ");
                doc.add_text(0, &format!("{} {filler}", ["the"; 5][..1 + (i % 5) as usize].join(" ")));
                w.add_document(&doc, &schema);
            }
            w
        };
        for (live, captured) in [(crate::codec::VERSION, crate::codec::MIN_WRITE_VERSION), (crate::codec::MIN_WRITE_VERSION, crate::codec::VERSION)] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(live));
            let bytes = build().encode_versioned(captured);
            assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), captured);
            let dir = RamDirectory::new();
            dir.write(&segment_file_name(0), &bytes).unwrap();
            let seg = Segment::open(&dir, 0).unwrap();
            assert_eq!(seg.format_version(), captured);
            let postings = seg.postings(&Term::new_text(0, "the")).unwrap();
            assert!(postings.num_full_blocks() > 0);
            assert_eq!(postings.has_impact_staircases(), captured >= crate::codec::IMPACTS_SINCE, "live v{live}, captured v{captured}");
        }
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// Docs without a value in a fast column (and a geo field) stay absent
    /// after the memtable is flushed: `exists` presence, range hits and geo
    /// queries on the sealed segment match the memtable's.
    #[test]
    fn flushed_segment_keeps_sparse_doc_values() {
        use crate::query::reader::LeafReader;
        let mut price = Field::new(2, "price", FieldType::F64);
        price.fast = true;
        let mut lat = Field::new(5, "g#lat", FieldType::F64);
        lat.fast = true;
        let geo = Field::new(6, "g#geo", FieldType::GeoPoint);
        let schema = vec![Field::new(0, "title", FieldType::Text), price, lat, geo];
        let mut w = SegmentWriter::new(0);
        for d in 0..4u32 {
            let mut doc = Document::new();
            doc.add_text(0, "x");
            if d != 2 {
                doc.add_f64(2, if d == 0 { -1.0 } else { 1.0 + d as f64 });
                doc.add_f64(5, d as f64);
                doc.add_geo_point(6, d as f64, 20.0);
            }
            w.add_document(&doc, &schema);
        }
        let probe = |leaf: &LeafReader<'_>| {
            let bits = leaf.range_f64_bits(2, -10.0, 10.0).unwrap();
            let range: Vec<u32> = (0..4).filter(|&d| bits[0] >> d & 1 == 1).collect();
            let shape = GeoShape::BoundingBox { top: 10.0, left: 0.0, bottom: -10.0, right: 30.0 };
            (leaf.value_presence(2).unwrap(), leaf.value_presence(5).unwrap(), range, leaf.geo_docs(6, &shape))
        };
        let mem = probe(&LeafReader::Memtable(&w, None));
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        let sealed = probe(&LeafReader::Segment(&seg));
        assert_eq!(mem.0, vec![0, 1, 3]);
        assert_eq!(sealed, mem, "sealed segment == memtable");
        assert_eq!(sealed.2, vec![0, 1, 3], "missing doc 2 is not the padding 0");
    }

    #[test]
    fn term_frequency() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(3);
        let mut doc = Document::new();
        doc.add_text(0, "rust rust Rust go");
        writer.add_document(&doc, &test_schema());
        let segment = writer.write_to(&dir).unwrap();
        let p = segment.postings(&Term::new_text(0, "rust")).unwrap();
        assert_eq!(p.doc_freq(), 1);
        assert_eq!(p.total_term_freq(), 3);
        assert_eq!(collect(p), vec![(0, 3)]);
    }

    #[test]
    fn positions_and_norms() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        let schema = test_schema();
        let mut d0 = Document::new();
        d0.add_text(0, "the quick brown fox jumps over the lazy dog");
        writer.add_document(&d0, &schema);
        let mut d1 = Document::new();
        d1.add_text(0, "fox");
        d1.add_text(1, "a b c d e f g h i j");
        writer.add_document(&d1, &schema);
        let segment = writer.write_to(&dir).unwrap();

        let mut p = segment.postings(&Term::new_text(0, "the")).unwrap();
        assert!(p.has_positions());
        assert_eq!(p.next(), Some(0));
        assert_eq!(p.positions(), &[0, 6]);
        let mut p = segment.postings(&Term::new_text(0, "fox")).unwrap();
        assert_eq!(p.next(), Some(0));
        assert_eq!(p.positions(), &[3]);
        assert_eq!(p.next(), Some(1));
        assert_eq!(p.positions(), &[0]);

        let norms = segment.norms(0).unwrap();
        assert_eq!(norms.get(0), 9);
        assert_eq!(norms.get(1), 1);
        assert_eq!(norms.total(), 10);
        assert!((segment.avg_field_len(0) - 5.0).abs() < 1e-6);
        assert_eq!(segment.norms(1).unwrap().get(1), 10);
        assert_eq!(segment.norms(1).unwrap().get(0), 0);
        assert!(segment.norms(2).is_none());
    }

    #[test]
    fn memtable_is_searchable_before_flush() {
        let schema = test_schema();
        let mut w = SegmentWriter::new(9);
        for i in 0..500u32 {
            let mut d = Document::new();
            d.add_text(0, &format!("alpha beta word{} {}", i % 7, if i % 2 == 0 { "even" } else { "odd" }));
            d.add_f64(2, i as f64);
            d.add_keyword(4, if i % 3 == 0 { "x" } else { "y" });
            w.add_document(&d, &schema);
        }
        assert_eq!(w.doc_freq(&Term::new_text(0, "alpha")), 500);
        assert_eq!(w.doc_freq(&Term::new_text(0, "even")), 250);
        assert_eq!(w.doc_freq(&Term::new_text(0, "nope")), 0);
        let mut p = w.postings(&Term::new_text(0, "word3")).unwrap();
        assert!(p.has_positions());
        assert_eq!(p.next(), Some(3));
        assert_eq!(p.positions(), &[2]);
        assert_eq!(p.advance(100), Some(101));
        let r = w.term_range(0, Some(b"word"), Some(b"wore"));
        assert_eq!(r.len(), 7);
        assert_eq!(w.term_range(4, None, None).len(), 2);
        assert_eq!(w.norms_for(0).len(3), 4);
        assert_eq!(w.fast_fields().get_f64(2).unwrap().get(7), Some(7.0));
        assert_eq!(w.fast_fields().get_keyword(4).unwrap().get(3), Some("x"));
        assert_eq!(w.get_doc(5).unwrap().unwrap().get(2).unwrap().as_f64(), Some(5.0));
        assert!(w.estimated_bytes() > 10_000);

        // Deletions in memory carry into the flushed segment.
        w.delete_doc(4);
        w.delete_doc(400);
        assert!(w.is_deleted(4) && !w.is_deleted(5));
        assert_eq!(w.num_live_docs(), 498);
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        assert_eq!(seg.num_live_docs(), 498);
        assert!(seg.is_deleted(400));
        assert!(dir.exists("seg_00000009.del"));
        assert_eq!(seg.postings(&Term::new_text(0, "alpha")).unwrap().doc_freq(), 500);
    }

    #[test]
    fn vectors_index_search_and_flush() {
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "emb", FieldType::Vector { dim: 8 }).with_similarity("l2_norm"),
        ];
        let mut w = SegmentWriter::new(0);
        for i in 0..400u32 {
            let mut d = Document::new();
            d.add_text(0, &format!("doc {i}"));
            if i % 10 != 9 {
                let v: Vec<f32> = (0..8).map(|j| ((i * 7 + j * 3) % 50) as f32 / 50.0 + i as f32 * 1e-3).collect();
                d.add_vector(1, v);
            }
            w.add_document(&d, &schema);
        }
        let q: Vec<f32> = (0..8).map(|j| ((123 * 7 + j * 3) % 50) as f32 / 50.0 + 123.0 * 1e-3).collect();
        // Memtable: exact.
        let mem = w.knn(1, &q, 3, |_| true);
        assert_eq!(mem[0].0, 123);
        assert!((mem[0].1 - 1.0).abs() < 1e-6);
        assert!(w.vector(1, 9).is_none());
        assert_eq!(w.vector(1, 123).unwrap(), q.as_slice());

        let dir = RamDirectory::new();
        let builds = crate::vector::HnswBuilder::builds_on_this_thread();
        let seg = w.write_to(&dir).unwrap();
        assert_eq!(crate::vector::HnswBuilder::builds_on_this_thread(), builds, "flush builds no graph");
        assert!(dir.exists("vec_00000000.rvec"));
        assert_eq!(seg.vectors().fields(), vec![1]);
        assert_eq!(seg.vector(1, 123).unwrap(), q);
        assert!(seg.vector(1, 9).is_none());
        let approx = seg.knn(1, &q, 5, 50, |_| true).unwrap();
        assert_eq!(approx[0].0, 123);
        let exact = seg.knn_exact(1, &q, 5, |_| true).unwrap();
        assert_eq!(exact[0].0, 123);
        // Filtered: exclude 123, best must be something else and accepted.
        let f = seg.knn(1, &q, 5, 50, |d| d != 123).unwrap();
        assert!(f.iter().all(|(d, _)| *d != 123));
        assert_eq!(f.len(), 5);
        // Second call hits the graph cache (same answer).
        assert_eq!(seg.knn(1, &q, 5, 50, |_| true).unwrap()[0].0, 123);
        assert!(seg.knn(7, &q, 5, 50, |_| true).unwrap().is_empty());
    }

    #[test]
    fn multi_valued_fast_field_promotion_end_to_end() {
        use crate::query::reader::LeafReader;
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "prices", FieldType::I64),
            Field::new(2, "tags", FieldType::Keyword),
            Field::new(3, "weights", FieldType::F64),
            Field::new(4, "ids", FieldType::U64),
        ];
        let mut w = SegmentWriter::new(0);

        // doc0: one value each → stays single-valued.
        let mut d0 = Document::new();
        d0.add_text(0, "zero");
        d0.add_i64(1, 10);
        d0.add_keyword(2, "red");
        d0.add_f64(3, 1.5);
        d0.add_u64(4, 100);
        w.add_document(&d0, &schema);
        assert!(w.fast_fields().get_i64(1).is_some());
        assert!(w.fast_fields().get_multi_i64(1).is_none());

        // doc1: two values each → promotion (arrays arrive as repeated add_*).
        let mut d1 = Document::new();
        d1.add_text(0, "one");
        d1.add_i64(1, 20);
        d1.add_i64(1, 30);
        d1.add_keyword(2, "green");
        d1.add_keyword(2, "blue");
        d1.add_keyword(2, "green"); // dup within doc
        d1.add_f64(3, 2.5);
        d1.add_f64(3, 3.5);
        d1.add_u64(4, 200);
        d1.add_u64(4, 300);
        w.add_document(&d1, &schema);
        assert!(w.fast_fields().get_i64(1).is_none());
        assert!(w.fast_fields().get_multi_i64(1).is_some());
        assert!(w.fast_fields().get_keyword(2).is_none());
        assert!(w.fast_fields().get_multi_keyword(2).is_some());

        // doc2: no fast values → empty lists.
        let mut d2 = Document::new();
        d2.add_text(0, "two");
        w.add_document(&d2, &schema);

        // doc3: one value → one-element list in the multi column.
        let mut d3 = Document::new();
        d3.add_text(0, "three");
        d3.add_i64(1, 40);
        d3.add_keyword(2, "red");
        d3.add_f64(3, 4.5);
        d3.add_u64(4, 400);
        w.add_document(&d3, &schema);

        // Memtable view.
        let mem = LeafReader::Memtable(&w, None);
        assert_eq!(mem.i64_values(1, 0), vec![10]);
        assert_eq!(mem.i64_values(1, 1), vec![20, 30]);
        assert_eq!(mem.i64_values(1, 2), Vec::<i64>::new());
        assert_eq!(mem.i64_values(1, 3), vec![40]);
        assert_eq!(mem.keyword_values(2, 1), vec!["blue".to_string(), "green".to_string()]);
        assert_eq!(mem.keyword_values(2, 2), Vec::<String>::new());
        assert_eq!(mem.f64_values(3, 1), vec![2.5, 3.5]);
        assert_eq!(mem.u64_values(4, 1), vec![200, 300]);
        assert_eq!(mem.range_i64(1, 20, 20), vec![1]);
        assert_eq!(mem.range_i64(1, 30, 30), vec![1]);
        assert_eq!(mem.range_i64(1, 10, 40), vec![0, 1, 3]);
        assert_eq!(mem.range_f64(3, 3.0, 4.0), vec![1]);
        assert_eq!(mem.range_u64(4, 300, 400), vec![1, 3]);
        // The in-memory multi column itself.
        let mc = w.fast_fields().get_multi_i64(1).unwrap();
        assert_eq!(mc.values(0), &[10]);
        assert_eq!(mc.values(1), &[20, 30]);
        assert_eq!(mc.values(2), &[] as &[i64]);
        assert_eq!(mc.values(3), &[40]);

        // Flush and read back from the segment.
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        let leaf = LeafReader::Segment(&seg);
        assert_eq!(leaf.i64_values(1, 0), vec![10]);
        assert_eq!(leaf.i64_values(1, 1), vec![20, 30]);
        assert_eq!(leaf.i64_values(1, 2), Vec::<i64>::new());
        assert_eq!(leaf.i64_values(1, 3), vec![40]);
        assert_eq!(leaf.keyword_values(2, 1), vec!["blue".to_string(), "green".to_string()]);
        assert_eq!(leaf.keyword_values(2, 3), vec!["red".to_string()]);
        assert_eq!(leaf.f64_values(3, 1), vec![2.5, 3.5]);
        assert_eq!(leaf.f64_values(3, 2), Vec::<f64>::new());
        assert_eq!(leaf.u64_values(4, 3), vec![400]);
        // range_i64 matches on either value.
        assert_eq!(leaf.range_i64(1, 20, 20), vec![1]);
        assert_eq!(leaf.range_i64(1, 30, 30), vec![1]);
        assert_eq!(leaf.range_i64(1, 10, 40), vec![0, 1, 3]);
        assert_eq!(leaf.range_i64(1, 25, 35), vec![1]);
        assert_eq!(leaf.range_f64(3, 3.0, 4.0), vec![1]);
        assert_eq!(leaf.range_u64(4, 300, 400), vec![1, 3]);
        // Single-valued accessors on a promoted field return nothing (the
        // column is now multi-valued); the *_values accessors are the API.
        assert_eq!(leaf.i64_value(1, 0), None);
        let ff = seg.fast_fields();
        assert!(ff.get_i64(1).is_none());
        let mk = ff.get_multi_keyword(2).unwrap();
        assert_eq!(mk.exact_match("red"), vec![0, 3]);
        assert_eq!(mk.exact_match("green"), vec![1]);
        assert_eq!(mk.ordinal_counts(|_| true), vec![1, 1, 2]); // blue, green, red
        // Postings for the keyword field still see every value.
        assert_eq!(seg.postings(&Term::new_text(2, "green")).unwrap().doc_freq(), 1);
        assert_eq!(seg.postings(&Term::new_text(2, "red")).unwrap().doc_freq(), 2);
    }

    #[test]
    fn single_valued_fields_stay_single_valued() {
        // A field whose docs each carry exactly one value must keep the
        // existing single-valued column (and its tag).
        let schema = test_schema();
        let mut w = SegmentWriter::new(0);
        for i in 0..50u32 {
            let mut d = Document::new();
            d.add_f64(2, i as f64);
            d.add_keyword(4, "k");
            w.add_document(&d, &schema);
        }
        assert!(w.fast_fields().get_f64(2).is_some());
        assert!(w.fast_fields().get_multi_f64(2).is_none());
        let dir = RamDirectory::new();
        let seg = w.write_to(&dir).unwrap();
        let ff = seg.fast_fields();
        assert!(ff.get_f64(2).is_some());
        assert!(ff.get_multi_f64(2).is_none());
        assert!(ff.get_keyword(4).is_some());
        assert_eq!(ff.fields(), vec![(2, 3), (4, 5)]);
    }

    /// The live count cached per deletes generation follows every change:
    /// visible deletes, pending ones once published, repeated deletes of a
    /// doc, and equals a fresh count each time.
    #[test]
    fn live_count_follows_every_deletes_generation() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..300 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc {i}"));
            writer.add_document(&doc, &test_schema());
        }
        drop(writer.write_to(&dir).unwrap());
        let segment = Segment::open(&dir, 0).unwrap();
        let fresh = |s: &Segment| s.num_docs - s.deletions.read().visible().count();
        assert_eq!(segment.num_live_docs(), 300);
        segment.delete_doc(7).unwrap();
        assert_eq!(segment.num_live_docs(), 299);
        assert_eq!(segment.num_live_docs(), fresh(&segment));
        segment.delete_doc(7).unwrap();
        assert_eq!(segment.num_live_docs(), 299);
        segment.delete_doc_pending(8).unwrap();
        assert_eq!(segment.num_live_docs(), 299, "pending deletes are not visible yet");
        segment.publish_deletions().unwrap();
        assert_eq!(segment.num_live_docs(), 298);
        for d in 100..150 { segment.delete_doc(d).unwrap(); }
        assert_eq!(segment.num_live_docs(), 248);
        assert_eq!(segment.num_live_docs(), fresh(&segment));
    }

    #[test]
    fn resident_deletion_snapshot_refusal_preserves_visibility_and_owner() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc {i}"));
            writer.add_document(&doc, &test_schema());
        }
        drop(writer.write_to(&dir).unwrap());
        let budget = crate::admission::Budget::new(u64::MAX);
        let segment = Segment::open_inner_admitted(&dir, 0, &budget).unwrap();
        let retained = segment.deletions();
        let before = budget.used();
        budget.set_limit(before);
        segment.delete_doc_pending(3).unwrap();
        assert!(segment.is_deleted_latest(3));
        assert!(!segment.is_deleted(3));
        assert!(segment.has_pending_deletions());
        assert!(!retained.get(3));
        assert_eq!(budget.used(), before, "acknowledged delete consumes preadmitted spare");
        assert!(matches!(segment.publish_deletions(), Err(crate::error::RusticError::Memory(_))));
        assert!(!segment.is_deleted(3), "refused refresh must preserve visible snapshot");
        assert!(segment.has_pending_deletions(), "refused refresh remains retryable");
        budget.set_limit(before + BitSet::storage_bytes(10));
        segment.publish_deletions().unwrap();
        assert!(segment.is_deleted(3));
        assert!(!retained.get(3));
        let escaped_latest = segment.latest_deletions().unwrap();
        budget.set_limit(budget.used());
        segment.delete_doc_pending(4).unwrap();
        assert!(segment.is_deleted_latest(4));
        assert!(!escaped_latest.get(4), "commit/merge snapshot must remain immutable");
        assert!(matches!(segment.latest_deletions(), Err(crate::error::RusticError::Memory(_))), "another escaping snapshot must replenish before sharing");
        drop(escaped_latest);
        drop(segment);
        assert_eq!(budget.used(), BitSet::storage_bytes(10));
        drop(retained);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn deletions_persist() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc {i}"));
            writer.add_document(&doc, &test_schema());
        }
        let segment = writer.write_to(&dir).unwrap();
        segment.delete_doc(0).unwrap();
        segment.delete_doc(5).unwrap();
        assert_eq!(segment.num_live_docs(), 8);
        segment.write_deletions(&dir).unwrap();

        let reopened = Segment::open(&dir, 0).unwrap();
        assert_eq!(reopened.num_live_docs(), 8);
        assert!(reopened.is_deleted(0) && reopened.is_deleted(5) && !reopened.is_deleted(1));
    }

    #[test]
    fn many_docs_many_terms() {
        let dir = RamDirectory::new();
        let mut writer = SegmentWriter::new(0);
        for i in 0..2000u32 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("word{} common even{}", i % 97, i % 2));
            doc.add_f64(2, i as f64);
            writer.add_document(&doc, &test_schema());
        }
        let segment = writer.write_to(&dir).unwrap();
        assert_eq!(segment.postings(&Term::new_text(0, "common")).unwrap().doc_freq(), 2000);
        let even = segment.postings(&Term::new_text(0, "even0")).unwrap();
        assert_eq!(even.doc_freq(), 1000);
        let docs = collect(even);
        assert!(docs.iter().all(|(d, _)| d % 2 == 0));
        let mut p = segment.postings(&Term::new_text(0, "word5")).unwrap();
        assert_eq!(p.advance(600), Some(684)); // first doc ≡ 5 (mod 97) that is >= 600
        assert_eq!(segment.fast_fields().get_f64(2).unwrap().count_in_range(100.0, 199.0), 100);
    }
}


#[cfg(test)]
mod ordinal_admission_tests {
    use super::*;
    #[test]
    fn ordinal_cache_refuses_before_build_and_retains_escaped_reader() {
        let mut writer=SegmentWriter::new(0);
        let field=crate::types::Field::new(0,"keyword",crate::types::FieldType::Keyword);
        for i in 0..20 {let mut doc=crate::types::Document::new();doc.add_keyword(0,&format!("term{i}"));writer.add_document(&doc,std::slice::from_ref(&field));}
        let dir=crate::codec::directory::RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let small=crate::admission::Budget::new(1);
        assert!(matches!(segment.keyword_ord_index_with_budget(0,&small),Err(crate::RusticError::Memory(_))));
        assert!(segment.ord_index.lock().is_empty());assert_eq!(small.used(),0);
        let budget=crate::admission::Budget::new(1<<20);
        let index=segment.keyword_ord_index_with_budget(0,&budget).unwrap().unwrap();
        assert_eq!(index.docs.len(),20);drop(segment);assert!(budget.used()>0);
        drop(index);assert_eq!(budget.used(),0);
    }
}

#[cfg(test)]
mod filter_admission_tests {
    use super::*;
    #[test]
    fn filter_cache_refuses_before_build_and_keeps_escaped_reader_charged() {
        let mut writer=SegmentWriter::new(0);
        let field=crate::types::Field::new(0,"keyword",crate::types::FieldType::Keyword);
        let mut doc=crate::types::Document::new();doc.add_keyword(0,"term");writer.add_document(&doc,&[field]);
        let dir=crate::codec::directory::RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let small=crate::admission::Budget::new(1);
        let (requests,refused)=(FILTER_CACHE_STATS.requests.load(std::sync::atomic::Ordering::Relaxed),FILTER_CACHE_STATS.refused.load(std::sync::atomic::Ordering::Relaxed));
        assert!(segment.cached_filter_bits_admitted((0,0,0,0),||panic!("must refuse before build"),&small).is_err());
        // Counted (other tests may count concurrently: at least this one).
        assert!(FILTER_CACHE_STATS.requests.load(std::sync::atomic::Ordering::Relaxed)>requests);
        assert!(FILTER_CACHE_STATS.refused.load(std::sync::atomic::Ordering::Relaxed)>refused,"a refused filter build is counted");
        assert!(segment.filter_cache.lock().is_empty());assert_eq!(small.used(),0);
        let budget=crate::admission::Budget::new(4096);
        let bits=segment.cached_filter_bits_admitted((0,0,0,0),||vec![1],&budget).unwrap();
        let mut scorer=crate::query::docset::BitSetDocSet::shared(bits.clone());
        drop(segment);drop(bits);assert!(budget.used()>0);
        assert_eq!(crate::query::docset::DocSet::advance(&mut scorer),0);
        drop(scorer);assert_eq!(budget.used(),0);
    }

    /// A slow build holds only its own key: other keys of the same cache are
    /// served meanwhile, concurrent requesters of the key build it once, and
    /// a failed build caches nothing (the next requester builds it).
    #[test]
    fn cache_builds_run_outside_the_map_lock_once_per_key() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let map: parking_lot::Mutex<Vec<(u16, BuildSlot<Arc<u32>>)>> = parking_lot::Mutex::new(Vec::new());
        let builds = AtomicUsize::new(0);
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            let (map_ref, builds_ref) = (&map, &builds);
            let slow = scope.spawn(move || cached_build(map_ref, 1, None, || {
                let builds = builds_ref;
                builds.fetch_add(1, Ordering::SeqCst);
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok::<_, ()>(Some(Arc::new(1)))
            }));
            started_rx.recv().unwrap();
            // Another key while key 1 builds: not blocked by it.
            let other = cached_build(&map, 2, None, || Ok::<_, ()>(Some(Arc::new(2)))).unwrap().unwrap();
            assert_eq!(*other, 2);
            // A second requester of key 1 waits for the first build.
            let same = scope.spawn(|| cached_build(&map, 1, None, || { builds.fetch_add(1, Ordering::SeqCst); Ok::<_, ()>(Some(Arc::new(9))) }));
            std::thread::sleep(std::time::Duration::from_millis(20));
            release_tx.send(()).unwrap();
            assert_eq!(*slow.join().unwrap().unwrap().unwrap(), 1);
            assert_eq!(*same.join().unwrap().unwrap().unwrap(), 1);
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        // A refusal is not cached and does not poison the slot.
        assert!(cached_build(&map, 3, None, || Err::<Option<Arc<u32>>, ()>(())).is_err());
        assert!(!map.lock().iter().any(|(k, _)| *k == 3));
        assert_eq!(*cached_build(&map, 3, None, || Ok::<_, ()>(Some(Arc::new(3)))).unwrap().unwrap(), 3);
        // A panicking build neither poisons the slot nor caches anything.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cached_build(&map, 4, None, || -> std::result::Result<Option<Arc<u32>>, ()> { panic!("build failed") })));
        assert!(panicked.is_err());
        assert_eq!(*cached_build(&map, 4, None, || Ok::<_, ()>(Some(Arc::new(4)))).unwrap().unwrap(), 4);
    }
}

#[cfg(test)]
mod geo_bound_admission_tests {
    use super::*;
    #[test]
    fn optional_geo_sort_bounds_refuse_before_cache_and_follow_reader() {
        let mut writer=SegmentWriter::new(0);
        let field=crate::types::Field::new(0,"geo",crate::types::FieldType::GeoPoint);
        let mut doc=crate::types::Document::new();doc.add_geo_point(0,10.0,20.0);writer.add_document(&doc,&[field]);
        let dir=crate::codec::directory::RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let small=crate::admission::Budget::new(1);
        assert!(segment.geo_blocks_admitted(0,128,&small).is_none());assert!(segment.geo_blocks.lock().is_empty());
        let budget=crate::admission::Budget::new(4096);
        let blocks=segment.geo_blocks_admitted(0,128,&budget).unwrap();assert_eq!(blocks[0].count,1);
        drop(segment);assert!(budget.used()>0);drop(blocks);assert_eq!(budget.used(),0);
    }
}

#[cfg(test)]
mod stored_cache_admission_tests {
    use super::*;
    #[test]
    fn declined_chunk_retention_still_returns_documents_and_preserves_budget() {
        let mut writer=SegmentWriter::new(0);
        let field=crate::types::Field::new(0,"keyword",crate::types::FieldType::Keyword);
        for i in 0..20 {let mut doc=crate::types::Document::new();doc.add_keyword(0,&format!("value-{i}"));writer.add_document(&doc,std::slice::from_ref(&field));}
        let dir=crate::codec::directory::RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let small=crate::admission::Budget::new(1);
        for doc in 0..20 {assert!(segment.get_doc_with_cache_budget(doc,&small).is_some());}
        assert!(segment.store_chunks.lock().is_empty());assert_eq!(small.used(),0);
        let budget=crate::admission::Budget::new(1<<20);
        assert!(segment.get_doc_with_cache_budget(0,&budget).is_some());
        let reader=segment.store_chunks.lock()[0].1.clone();drop(segment);
        assert!(budget.used()>0);drop(reader);assert_eq!(budget.used(),0);
    }
}

#[cfg(test)]
mod vector_payload_admission_tests {
    use super::*;
    #[test]
    fn payload_refusal_precedes_attachment_and_row_owner_outlives_segment() {
        let dir = crate::codec::directory::RamDirectory::new();
        let fields = [crate::Field::new(0, "group", crate::FieldType::Keyword),
            crate::Field::new(1, "vector", crate::FieldType::Vector { dim: 2 })];
        let mut writer = SegmentWriter::new(0);
        for _ in 0..2 {
            let mut doc = crate::Document::new();
            doc.add_keyword(0, "retained");doc.add_vector(1, vec![1.0, 0.0]);
            writer.add_document(&doc, &fields);
        }
        drop(writer.write_to(&dir).unwrap());
        let store = VectorStore::open_dir(&dir).unwrap();
        let small = crate::admission::Budget::new(1);
        assert!(matches!(Segment::open_inner_admitted(&dir, 0, &small), Err(crate::RusticError::Memory(_))));
        assert_eq!(store.num_alive(1), 0);
        assert_eq!(small.used(), 0);
        let budget = crate::admission::Budget::new(1 << 20);
        let segment = Segment::open_inner_admitted(&dir, 0, &budget).unwrap();
        assert_eq!(segment.vector_payload_rows()[0].values, vec![(0, "retained".into())]);
        assert!(Arc::ptr_eq(&segment.vector_payload_rows()[0].values[0].1,
            &segment.vector_payload_rows()[1].values[0].1), "repeated payload bytes must share storage");
        store.attach(&segment);
        assert_eq!(store.num_alive(1), 2);
        drop(segment);
        assert!(budget.used() > 0, "table rows retain their admission after segment drop");
        drop(store);assert_eq!(budget.used(), 0);
    }
}

#[cfg(test)]
mod ordinary_decode_admission_tests {
    use super::*;
    #[test]
    fn ordinary_decode_refuses_before_cache_population_and_retains_document_owner() {
        let dir = crate::codec::directory::RamDirectory::new();
        let schema = [crate::Field::new(0, "body", crate::FieldType::Text)];
        let mut writer = SegmentWriter::new(0);
        let mut document = Document::new(); document.add_text(0, &"x".repeat(4096));
        writer.add_document(&document, &schema);
        let segment = writer.write_to(&dir).unwrap();
        let small = crate::admission::Budget::new(1);
        assert!(matches!(segment.get_doc_admitted(0, &small), Err(crate::RusticError::Memory(_))));
        assert!(segment.store_chunks.lock().is_empty());
        assert_eq!(small.used(), 0);
        let budget = crate::admission::Budget::new(4 << 20);
        let decoded = segment.get_doc_admitted(0, &budget).unwrap().unwrap();
        assert!(budget.used() > 0);
        let bytes = bincode::serialize(&decoded).unwrap();
        assert_eq!(bytes, bincode::serialize(&document).unwrap(), "admission is not stored metadata");
        drop(segment); assert!(budget.used() > 0);
        drop(decoded); assert_eq!(budget.used(), 0);
    }
}

#[cfg(test)]
mod streaming_decode_corruption_tests {
    use super::*;
    #[test]
    fn streaming_reads_propagate_table_chunk_and_document_corruption() {
        let invalid_document = crate::codec::store::encode_store_chunks(&[vec![1, 255]], &[1]);
        let mut invalid_compression = invalid_document.clone();
        invalid_compression[4] = 0xf0; // promises literals absent from this block
        for encoded in [vec![0], invalid_compression, invalid_document] {
            let dir = crate::codec::directory::RamDirectory::new();
            let fields = [crate::Field::new(0, "body", crate::FieldType::Text)];
            let mut writer = SegmentWriter::new(0);
            let mut doc = Document::new(); doc.add_text(0, "valid before corruption");
            writer.add_document(&doc, &fields);
            let mut segment = writer.write_to(&dir).unwrap();
            let data = Arc::get_mut(&mut segment.data).expect("unpublished fixture");
            data.store_data = FileSlice::from_vec(encoded);
            data.store_table = OnceLock::new();
            let leaf = crate::query::LeafReader::Segment(&segment);
            let mut callbacks = 0;
            let result = leaf.try_for_each_doc(&[0], |_, _| callbacks += 1);
            assert!(result.is_err(), "corruption must not become a successful empty page");
            assert_eq!(callbacks, 0);
        }
    }
}

#[cfg(test)]
mod store_table_admission_tests {
    use super::*;
    #[test]
    fn stored_table_refusal_does_not_publish_and_cached_reader_retains_owner() {
        let dir = crate::codec::directory::RamDirectory::new();
        let fields = [crate::Field::new(0, "body", crate::FieldType::Text)];
        let mut writer = SegmentWriter::new(0);
        let mut doc = Document::new(); doc.add_text(0, "stored value");
        writer.add_document(&doc, &fields);
        let segment = writer.write_to(&dir).unwrap();
        let budget = crate::admission::Budget::new(0);
        assert!(matches!(segment.store_with_budget(&budget), Err(crate::RusticError::Memory(_))));
        assert!(segment.store_table.get().is_none());
        assert_eq!(budget.used(), 0);
        budget.set_limit(4096);
        assert_eq!(segment.store_with_budget(&budget).unwrap().num_docs(), 1);
        let retained = budget.used(); assert!(retained > 0);
        budget.set_limit(0);
        assert_eq!(segment.store_with_budget(&budget).unwrap().num_docs(), 1);
        assert_eq!(budget.used(), retained, "cached reader does not reserve twice");
        drop(segment);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn malformed_table_releases_reservation_without_publishing() {
        let dir = crate::codec::directory::RamDirectory::new();
        let fields = [crate::Field::new(0, "body", crate::FieldType::Text)];
        let mut writer = SegmentWriter::new(0);
        let mut doc = Document::new(); doc.add_text(0, "stored value");
        writer.add_document(&doc, &fields);
        let mut segment = writer.write_to(&dir).unwrap();
        // Valid table header says one chunk, but contains no chunk entries.
        let mut encoded = vec![1, 1]; encoded.extend_from_slice(&0u64.to_le_bytes());
        Arc::get_mut(&mut segment.data).unwrap().store_data = FileSlice::from_vec(encoded);
        let budget = crate::admission::Budget::new(4096);
        assert!(matches!(segment.store_with_budget(&budget), Err(crate::RusticError::Corrupted(_))));
        assert!(segment.store_table.get().is_none());
        assert_eq!(budget.used(), 0);
    }
}

#[cfg(test)]
mod batched_doc_tests {
    use super::*;
    /// A batched read returns each asked document exactly as `get_doc`
    /// does, across several store chunks, in any order and with repeats of
    /// a chunk's docs, and skips documents the store does not hold.
    #[test]
    fn batched_reads_equal_single_reads() {
        let mut writer = SegmentWriter::new(0);
        let field = crate::types::Field::new(0, "keyword", crate::types::FieldType::Keyword);
        let n = 3000u32;
        for i in 0..n { let mut doc = crate::types::Document::new(); doc.add_keyword(0, &format!("value-{i}-{}", "x".repeat((i % 17) as usize))); writer.add_document(&doc, std::slice::from_ref(&field)); }
        let dir = crate::codec::directory::RamDirectory::new();
        let segment = writer.write_to(&dir).unwrap();
        let store = segment.store().unwrap();
        assert!(store.locate(n - 1).unwrap().0 > 1, "several chunks");
        drop(store);
        let docs: Vec<DocId> = vec![2999, 5, 6, 1500, 0, 4, 2998, 1501, n + 10, 777];
        let mut got: Vec<Option<Document>> = (0..docs.len()).map(|_| None).collect();
        segment.get_docs_batched(&docs, true, |i, d| { assert!(got[i].is_none()); got[i] = Some(d); }).unwrap();
        for (i, &d) in docs.iter().enumerate() {
            let want = segment.get_doc(d).unwrap();
            assert_eq!(got[i].as_ref().map(|g| format!("{:?}", g.fields)), want.as_ref().map(|w| format!("{:?}", w.fields)), "doc {d}");
        }
        assert!(got[docs.len() - 2].is_none());
    }
}
