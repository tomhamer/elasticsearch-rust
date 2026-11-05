use serde::{Deserialize, Serialize};
use std::fmt;

/// A document identifier within a segment.
pub type DocId = u32;

/// A global document identifier (segment_id, local_doc_id).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct DocAddress {
    pub segment_id: u32,
    pub doc_id: DocId,
}

impl DocAddress {
    pub fn new(segment_id: u32, doc_id: DocId) -> Self {
        Self { segment_id, doc_id }
    }
}

impl fmt::Display for DocAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seg{}:doc{}", self.segment_id, self.doc_id)
    }
}

/// A term in the index.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Term {
    pub field: u16,
    pub text: Vec<u8>,
}

/// Exclusive lexicographic bound for a term's two-byte field prefix. Field
/// IDs are little-endian in keys, so the next numeric ID is not this bound.
pub(crate) fn field_key_end(field: u16) -> Option<Vec<u8>> {
    u16::from_be_bytes(field.to_le_bytes()).checked_add(1)
        .map(|next| next.to_be_bytes().to_vec())
}

impl Term {
    pub fn new(field: u16, text: impl Into<Vec<u8>>) -> Self {
        Self {
            field,
            text: text.into(),
        }
    }

    pub fn new_text(field: u16, text: &str) -> Self {
        Self {
            field,
            text: text.as_bytes().to_vec(),
        }
    }

    pub fn new_u64(field: u16, value: u64) -> Self {
        Self {
            field,
            text: value.to_be_bytes().to_vec(),
        }
    }

    pub fn new_i64(field: u16, value: i64) -> Self {
        Self {
            field,
            text: value.to_be_bytes().to_vec(),
        }
    }

    pub fn new_f64(field: u16, value: f64) -> Self {
        Self {
            field,
            text: value.to_bits().to_be_bytes().to_vec(),
        }
    }

    pub fn new_bool(field: u16, value: bool) -> Self {
        Self {
            field,
            text: vec![value as u8],
        }
    }

    /// Dictionary key: 2-byte little-endian field id followed by the term bytes.
    /// Keys sort by field first, then by term, which is what range queries need.
    #[inline]
    pub fn key(&self) -> Vec<u8> {
        let mut key = Vec::with_capacity(2 + self.text.len());
        key.extend_from_slice(&self.field.to_le_bytes());
        key.extend_from_slice(&self.text);
        key
    }

    /// Exclusive upper bound of every [`Term::key`] of `field`: the
    /// byte-wise successor of its 2-byte little-endian prefix (`None`: no
    /// key sorts after the field's). `field + 1` is not a bound: keys are
    /// compared as bytes, so with a little-endian prefix field 265
    /// (`[9, 1]`) sorts between fields 9 (`[9, 0]`) and 10 (`[10, 0]`).
    pub fn field_key_end(field: u16) -> Option<Vec<u8>> {
        let [lo, hi] = field.to_le_bytes();
        if hi < u8::MAX {
            Some(vec![lo, hi + 1])
        } else if lo < u8::MAX {
            Some(vec![lo + 1])
        } else {
            None
        }
    }

    /// Inverse of [`Term::key`].
    pub fn from_key(key: &[u8]) -> Self {
        if key.len() < 2 {
            return Term::new(0, Vec::new());
        }
        Term {
            field: u16::from_le_bytes([key[0], key[1]]),
            text: key[2..].to_vec(),
        }
    }

    /// Returns the term as a u64 value (for numeric terms).
    pub fn as_u64(&self) -> Option<u64> {
        if self.text.len() == 8 {
            Some(u64::from_be_bytes(self.text.as_slice().try_into().ok()?))
        } else {
            None
        }
    }

    /// Returns the term as an i64 value (for numeric terms).
    pub fn as_i64(&self) -> Option<i64> {
        if self.text.len() == 8 {
            Some(i64::from_be_bytes(self.text.as_slice().try_into().ok()?))
        } else {
            None
        }
    }

    /// Returns the term as an f64 value (for numeric terms).
    pub fn as_f64(&self) -> Option<f64> {
        if self.text.len() == 8 {
            Some(f64::from_bits(u64::from_be_bytes(
                self.text.as_slice().try_into().ok()?,
            )))
        } else {
            None
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(text) = std::str::from_utf8(&self.text) {
            write!(f, "field{}:{}", self.field, text)
        } else {
            write!(f, "field{}:0x{:02x?}", self.field, self.text)
        }
    }
}

/// A field type in the index schema.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum FieldType {
    /// Full-text field with inverted index.
    Text,
    /// Keyword field (exact match, no tokenization).
    Keyword,
    /// 64-bit integer.
    I64,
    /// 64-bit unsigned integer.
    U64,
    /// 64-bit floating point.
    F64,
    /// Boolean.
    Bool,
    /// Date (stored as i64 milliseconds since epoch).
    Date,
    /// Geographic point (lat, lon).
    GeoPoint,
    /// Vector (for HNSW search).
    Vector { dim: u16 },
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldType::Text => write!(f, "text"),
            FieldType::Keyword => write!(f, "keyword"),
            FieldType::I64 => write!(f, "i64"),
            FieldType::U64 => write!(f, "u64"),
            FieldType::F64 => write!(f, "f64"),
            FieldType::Bool => write!(f, "bool"),
            FieldType::Date => write!(f, "date"),
            FieldType::GeoPoint => write!(f, "geo_point"),
            FieldType::Vector { dim } => write!(f, "vector({dim})"),
        }
    }
}

/// A field in the index schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    pub id: u16,
    pub name: String,
    pub field_type: FieldType,
    pub indexed: bool,
    pub stored: bool,
    pub fast: bool,
    /// Analyzer name for text fields (ES built-ins: "standard", "simple",
    /// "whitespace", "keyword", "stop", "english"). Defaults to "standard".
    #[serde(default)]
    pub analyzer: String,
    /// Vector similarity for `Vector` fields (ES names: "cosine",
    /// "dot_product", "l2_norm"). Defaults to "cosine".
    #[serde(default = "default_similarity")]
    pub similarity: String,
    /// Positions inserted between the values of a multi-valued text field
    /// (Lucene `Analyzer.getPositionIncrementGap`; ES defaults to 100).
    #[serde(default)]
    pub position_increment_gap: u32,
    /// Whether a text field keeps per-doc length norms (ES `norms`; default
    /// on). Without norms every doc scores with field length 1.
    #[serde(default = "default_true")]
    pub norms: bool,
    /// Whether tokens stacked on one position (synonyms, shingles, ...) are
    /// left out of the length norm (Lucene `discountOverlaps`, ES similarity
    /// setting `discount_overlaps`; default on).
    #[serde(default = "default_true")]
    pub discount_overlaps: bool,
    /// How a `Vector` field is quantized, indexed and rescored (ES
    /// `index_options` + `rescore_vector`). `None` = the store's defaults.
    /// Mirrored into the index's vector store by
    /// [`Index::sync_vector_field_options`](crate::Index::sync_vector_field_options).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_options: Option<crate::vector::store::VectorFieldOptions>,
    /// Whether a numeric column keeps its points index (doc ids sorted by
    /// value) next to the per-block min/max. Without it, range queries skip
    /// blocks by min/max only (Lucene's doc-values skipper, which ES uses
    /// instead of points for e.g. `@timestamp` in logsdb).
    #[serde(default = "default_true")]
    pub points: bool,
    /// A synthetic id field ([`crate::synthetic_id`]): derived from these
    /// columns in sealed segments (no postings, no stored value there).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic_id: Option<crate::synthetic_id::SyntheticId>,
    /// A feature field (ES `sparse_vector` / `rank_features`, Lucene
    /// `FeatureField`): each keyword value is a `token\0hexbits` pair
    /// ([`crate::fieldtypes::features::encode`]), indexed as the term
    /// `token` with the weight's top 16 bits as its term frequency, so
    /// postings carry the weights and their block maxima.
    #[serde(default)]
    pub features: bool,
    /// A `geo_shape` field: no keyword values, but the doc's triangles and
    /// centroid ([`crate::types::Document::shape_triangles`]) feed a
    /// triangle tree when [`Field::indexed`] and shape doc values when
    /// [`Field::fast`] (ES `LatLonShape` points and
    /// `BinaryShapeDocValuesField`, see
    /// [`crate::fieldtypes::shape_field`]).
    #[serde(default)]
    pub shape: bool,
    /// A single-valued numeric field whose stored value is its column's
    /// (`_version`, `_seq_no`, `_primary_term`): from segment format
    /// [`crate::codec::COLUMN_STORED_SINCE`] it is left out of the stored
    /// fields and put back from the column when a document is fetched.
    #[serde(default)]
    pub column_stored: bool,
    /// A single-valued keyword field whose docs each hold at most one of its
    /// terms (`_id`): from segment format
    /// [`crate::codec::KEYWORD_FROM_POSTINGS_SINCE`] sealed segments write
    /// no doc-values column for it. A doc's value is read from its stored
    /// fields; sorts, aggregations and scripts get a column derived from
    /// the postings on first use.
    #[serde(default)]
    pub column_from_postings: bool,
}

fn default_true() -> bool {
    true
}

fn default_similarity() -> String {
    "cosine".to_string()
}

impl Field {
    pub fn new(id: u16, name: &str, field_type: FieldType) -> Self {
        Self {
            id,
            name: name.to_string(),
            field_type,
            indexed: true,
            stored: true,
            fast: matches!(
                field_type,
                FieldType::I64
                    | FieldType::U64
                    | FieldType::F64
                    | FieldType::Bool
                    | FieldType::Date
                    | FieldType::Keyword
            ),
            analyzer: "standard".to_string(),
            similarity: "cosine".to_string(),
            position_increment_gap: 0,
            norms: true,
            discount_overlaps: true,
            vector_options: None,
            points: true,
            synthetic_id: None,
            features: false,
            shape: false,
            column_stored: false,
            column_from_postings: false,
        }
    }

    /// See [`Field::column_from_postings`].
    pub fn with_column_from_postings(mut self, on: bool) -> Self {
        self.column_from_postings = on;
        self
    }

    /// Serve the stored value from the column (see [`Field::column_stored`]).
    pub fn with_column_stored(mut self, on: bool) -> Self {
        self.column_stored = on;
        self
    }

    /// Derive this (keyword id) field from columns in sealed segments.
    pub fn with_synthetic_id(mut self, s: Option<crate::synthetic_id::SyntheticId>) -> Self {
        self.synthetic_id = s;
        self
    }

    /// Keep (`true`, default) or drop the points index of a numeric column.
    pub fn with_points(mut self, on: bool) -> Self {
        self.points = on;
        self
    }

    /// Index a keyword field's `token\0hexbits` values as weighted feature
    /// terms (see [`Field::features`]).
    pub fn with_features(mut self, on: bool) -> Self {
        self.features = on;
        self
    }

    /// Mark a keyword field as a `geo_shape` column (see [`Field::shape`]).
    pub fn with_shape(mut self, on: bool) -> Self {
        self.shape = on;
        self
    }

    /// Keep (`true`, default) or drop the per-doc length norms of a text field.
    pub fn with_norms(mut self, on: bool) -> Self {
        self.norms = on;
        self
    }

    /// Whether overlapping tokens are excluded from the length norm.
    pub fn with_discount_overlaps(mut self, on: bool) -> Self {
        self.discount_overlaps = on;
        self
    }

    /// Set the analyzer for a text field.
    pub fn with_analyzer(mut self, name: &str) -> Self {
        self.analyzer = name.to_string();
        self
    }

    /// Set the position gap between values of a multi-valued text field.
    pub fn with_position_increment_gap(mut self, gap: u32) -> Self {
        self.position_increment_gap = gap;
        self
    }

    /// Set the similarity for a vector field.
    pub fn with_similarity(mut self, name: &str) -> Self {
        self.similarity = name.to_string();
        self
    }

    /// Set the vector indexing options of a `Vector` field.
    pub fn with_vector_options(mut self, options: Option<crate::vector::store::VectorFieldOptions>) -> Self {
        self.vector_options = options;
        self
    }
}

/// A document to be indexed.
///
/// `nested` holds nested child documents (ES `nested` objects) as
/// `(nested path, child)`; a document with children is indexed as one
/// contiguous block (children first, parent last), see [`crate::nested`].
/// Children are not part of the parent's stored fields.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Document {
    pub fields: Vec<(u16, FieldValue)>,
    #[serde(skip)]
    pub nested: Vec<(String, Document)>,
    #[serde(skip)]
    pub(crate) decode_memory: Option<std::sync::Arc<crate::admission::Reservation>>,
    #[serde(skip)]
    allocation_memory: Option<std::sync::Arc<std::sync::Mutex<crate::admission::Reservation>>>,
    /// Bytes `allocation_memory` already admitted that no producer has
    /// claimed yet (see [`Document::admit_allocation`]). Never copied by
    /// `Clone`: a clone shares the reservation, not the unclaimed credit.
    #[serde(skip)]
    allocation_credit: AllocationCredit,
    /// Per `geo_shape` field (see [`Field::shape`]): the triangles ES
    /// indexes for this doc's values, tessellated when the doc was parsed
    /// ([`crate::fieldtypes::tessellator::geometry_triangles`]), and the
    /// centroid of all its values (ES `CentroidCalculator`, over the values
    /// as parsed) for its shape doc value. Not serialized: the field's
    /// structures are built from them when the doc is indexed.
    #[serde(skip)]
    pub shape_triangles: Vec<(u16, Vec<[i32; 7]>)>,
    #[serde(skip)]
    pub shape_centroids: Vec<(u16, crate::fieldtypes::shape::CentroidCalc)>,
    /// Each shape field's doc value, written when the doc was parsed
    /// ([`Document::encode_shape_values`]: off the indexing lock).
    #[serde(skip)]
    pub shape_values: Vec<(u16, Vec<u8>)>,
}

/// Admitted-but-unclaimed producer bytes of one [`Document`]. `Clone`
/// yields zero so two copies sharing one reservation never both spend it.
#[derive(Default)]
struct AllocationCredit(u64);
impl Clone for AllocationCredit {fn clone(&self)->Self{Self(0)}}

/// Granularity of [`Document::admit_allocation`]'s reservation growth. A
/// producer's admission is taken from the document's credit when it fits;
/// otherwise the reservation grows by this much (by exactly the shortfall
/// when that is larger, or when the chunk does not fit the budget), so a
/// small document costs one budget admission instead of one per field.
/// [`Document::settle_allocation`] returns the unclaimed rest, so the charge
/// left on a parsed document is exactly what its producers admitted.
pub const ALLOCATION_CHUNK: u64 = 16 << 10;

/// Decoded heap per encoded byte of a stored document, at most. The
/// smallest encoded field (u16 id, u32 variant tag, a 1-byte bool: 7 bytes)
/// decodes into one `(u16, FieldValue)` slot; serde preallocates at most
/// 1 MiB of slots and grows past that by doubling, so old and new slots
/// briefly coexist (3 slots per field). Text and vector payloads decode to
/// their own bytes plus an allocator header against an 8-byte encoded
/// length, well under that ratio.
pub(crate) const STORED_DECODE_FACTOR: u64 = (3 * std::mem::size_of::<(u16, FieldValue)>() as u64).div_ceil(7) + 1;

/// Stored-field copies keep the decode admission through their last owner.
#[derive(Debug)]
pub struct StoredValues {
    values: Vec<FieldValue>,
    _decode_memory: Option<std::sync::Arc<crate::admission::Reservation>>,
    _allocation_memory: Option<std::sync::Arc<std::sync::Mutex<crate::admission::Reservation>>>,
    _slots: Option<crate::admission::Reservation>,
}
impl std::ops::Deref for StoredValues {
    type Target = [FieldValue];
    fn deref(&self) -> &Self::Target { &self.values }
}
impl StoredValues {
    pub(crate) fn from_document(document: Option<Document>, field: u16) -> crate::error::Result<Self> {
        Self::from_document_on(document,field,&crate::admission::shared())
    }
    fn from_document_on(document:Option<Document>,field:u16,budget:&std::sync::Arc<crate::admission::Budget>)->crate::error::Result<Self>{
        let Some(document)=document else{return Ok(Self{values:Vec::new(),_decode_memory:None,_allocation_memory:None,_slots:None})};
        let count=document.fields.iter().filter(|(id,_)|*id==field).count();
        let bytes=count.checked_mul(std::mem::size_of::<FieldValue>()).ok_or_else(||crate::error::RusticError::Memory("stored-value slots overflow".into()))?;
        let slots=budget.reserve(bytes as u64,"stored-value result slots")?;
        let mut values=Vec::new();values.try_reserve_exact(count).map_err(|e|crate::error::RusticError::Memory(format!("stored-value slots: {e}")))?;
        // Move decoded payloads, retaining their original owner. Deep-cloning
        // here would temporarily duplicate every selected string/vector.
        let Document{fields,nested,decode_memory,allocation_memory,..}=document;
        for (id,value) in fields {if id==field{values.push(value);}}
        drop(nested);
        Ok(Self{values,_decode_memory:decode_memory,_allocation_memory:allocation_memory,_slots:Some(slots)})
    }
}

impl std::fmt::Debug for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("Document").field("fields", &self.fields).field("nested", &self.nested).finish() }
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add triangles of a `geo_shape` column's value (several values:
    /// their triangles together).
    pub fn add_shape_triangles(&mut self, field: u16, tris: Vec<[i32; 7]>) {
        match self.shape_triangles.iter_mut().find(|(f, _)| *f == field) {
            Some((_, v)) => v.extend(tris),
            None => self.shape_triangles.push((field, tris)),
        }
    }

    /// The triangles of `field`, when the doc was parsed with them.
    pub fn shape_triangles(&self, field: u16) -> Option<&[[i32; 7]]> {
        self.shape_triangles.iter().find(|(f, _)| *f == field).map(|(_, v)| v.as_slice())
    }

    /// Set the centroid of all of `field`'s values (see
    /// [`Document::shape_triangles`]).
    pub fn set_shape_centroid(&mut self, field: u16, c: crate::fieldtypes::shape::CentroidCalc) {
        match self.shape_centroids.iter_mut().find(|(f, _)| *f == field) {
            Some((_, v)) => *v = c,
            None => self.shape_centroids.push((field, c)),
        }
    }

    /// The centroid of `field`'s values.
    pub fn shape_centroid(&self, field: u16) -> Option<&crate::fieldtypes::shape::CentroidCalc> {
        self.shape_centroids.iter().find(|(f, _)| *f == field).map(|(_, v)| v)
    }

    /// Write every shape field's doc value from its triangles and centroid
    /// (once all its values are in; parsing calls this, in parallel across
    /// a bulk's docs). A field without a writable value gets none.
    pub fn encode_shape_values(&mut self) {
        self.shape_values.clear();
        for (f, tris) in &self.shape_triangles {
            if tris.is_empty() {
                continue;
            }
            let Some(c) = self.shape_centroids.iter().find(|(g, _)| g == f).map(|(_, c)| c) else { continue };
            if let Ok(v) = crate::fieldtypes::shape_field::doc_value(tris, c) {
                self.shape_values.push((*f, v));
            }
        }
    }

    /// The doc value [`Document::encode_shape_values`] wrote for `field`.
    pub fn shape_value(&self, field: u16) -> Option<&[u8]> {
        self.shape_values.iter().find(|(f, _)| *f == field).map(|(_, v)| v.as_slice())
    }

    /// Admission for parser-produced containers/payloads. The owner moves
    /// with this Document into pending writes and is independent of decode
    /// admission. Each producer must call this before allocating its storage.
    ///
    /// Admission is chunked ([`ALLOCATION_CHUNK`]): bytes come out of credit
    /// the reservation already holds, so the bytes are still admitted before
    /// the producer allocates; only the budget round trips are batched. A
    /// chunk that does not fit falls back to growing by exactly the
    /// shortfall, so a document whose own producers fit the budget today
    /// still fits; only the unclaimed credit (below one chunk per document
    /// being parsed, returned by [`Self::settle_allocation`]) is extra.
    pub fn admit_allocation(&mut self,bytes:usize)->crate::error::Result<()> {
        if bytes==0{return Ok(())}
        const LABEL:&str="document producer allocation";
        let bytes=u64::try_from(bytes).map_err(|_|crate::error::RusticError::Memory("document allocation overflow".into()))?;
        if let Some(owner)=&self.allocation_memory {
            let credit=self.allocation_credit.0;
            if credit>=bytes {self.allocation_credit.0=credit-bytes;return Ok(())}
            let shortfall=bytes-credit;
            let mut reservation=owner.lock().unwrap_or_else(|e|e.into_inner());
            if shortfall<ALLOCATION_CHUNK && reservation.grow(ALLOCATION_CHUNK,LABEL).is_ok() {
                self.allocation_credit.0=ALLOCATION_CHUNK-shortfall;
            } else {
                reservation.grow(shortfall,LABEL)?;
                self.allocation_credit.0=0;
            }
        } else {
            let metadata=std::mem::size_of::<std::sync::Mutex<crate::admission::Reservation>>()+2*std::mem::size_of::<usize>();
            let exact=bytes.checked_add(metadata as u64).ok_or_else(||crate::error::RusticError::Memory("document allocation overflow".into()))?;
            let budget=crate::admission::shared();
            let chunked=if exact<ALLOCATION_CHUNK {budget.reserve(ALLOCATION_CHUNK,LABEL).ok()} else {None};
            let (owner,credit)=match chunked {
                Some(owner)=>(owner,ALLOCATION_CHUNK-exact),
                None=>(budget.reserve(exact,LABEL)?,0),
            };
            self.allocation_memory=Some(std::sync::Arc::new(std::sync::Mutex::new(owner)));
            self.allocation_credit.0=credit;
        }
        Ok(())
    }
    /// Return the credit no producer claimed (and that of nested children)
    /// to the budget: afterwards the document's reservation holds exactly
    /// the bytes its producers admitted, as unchunked admission did. Call
    /// when a document is complete (end of parse, before it is retained by
    /// pending writes); a later admission starts a new chunk. Idempotent.
    pub fn settle_allocation(&mut self) {
        for (_,child) in &mut self.nested {child.settle_allocation();}
        let credit=std::mem::take(&mut self.allocation_credit.0);
        if credit==0 {return}
        if let Some(owner)=&self.allocation_memory {owner.lock().unwrap_or_else(|e|e.into_inner()).shrink(credit);}
    }
    /// Bytes admitted for producers of this document that none claimed yet.
    pub fn unclaimed_allocation(&self)->u64 {self.allocation_credit.0}
    pub fn reserve_field_slots(&mut self,additional:usize)->crate::error::Result<()> {
        let required=self.fields.len().checked_add(additional).ok_or_else(||crate::error::RusticError::Memory("document fields overflow".into()))?;
        if required<=self.fields.capacity(){return Ok(())}
        let capacity=required.max(self.fields.capacity().saturating_mul(2)).max(4);
        let bytes=capacity.checked_mul(std::mem::size_of::<(u16,FieldValue)>()).ok_or_else(||crate::error::RusticError::Memory("document field bytes overflow".into()))?;
        // Keep the old capacity charged during replacement; geometric growth
        // bounds conservative retained charges without per-field quadratic use.
        self.admit_allocation(bytes)?;
        self.fields.try_reserve_exact(capacity-self.fields.len()).map_err(|e|crate::error::RusticError::Memory(e.to_string()))?;
        Ok(())
    }
    pub fn try_push_field(&mut self,value:(u16,FieldValue))->crate::error::Result<()> {
        self.reserve_field_slots(1)?;self.fields.push(value);Ok(())
    }
    pub fn try_extend_fields_from(&mut self,other:&Document)->crate::error::Result<()> {
        use crate::error::RusticError;
        self.reserve_field_slots(other.fields.len())?;
        let mut bytes=0usize;
        for (_,v) in &other.fields {
            let n=match v {FieldValue::Text(s)|FieldValue::Keyword(s)=>s.len(),FieldValue::Vector(v)=>v.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(||RusticError::Memory("field copy overflow".into()))?,_=>0};
            bytes=bytes.checked_add(n).ok_or_else(||RusticError::Memory("field copy overflow".into()))?;
        }
        self.admit_allocation(bytes)?;
        for (id,v) in &other.fields {
            let value=match v {
                FieldValue::Text(s)|FieldValue::Keyword(s)=>{
                    let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|RusticError::Memory(e.to_string()))?;out.push_str(s);
                    if matches!(v,FieldValue::Text(_)){FieldValue::Text(out)}else{FieldValue::Keyword(out)}
                }
                FieldValue::Vector(v)=>{let mut out=Vec::new();out.try_reserve_exact(v.len()).map_err(|e|RusticError::Memory(e.to_string()))?;out.extend_from_slice(v);FieldValue::Vector(out)},
                value=>value.clone(),
            };
            self.fields.push((*id,value));
        }
        Ok(())
    }
    pub fn try_add_keyword(&mut self,field:u16,value:&str)->crate::error::Result<()> {
        self.reserve_field_slots(1)?;self.admit_allocation(value.len())?;
        let mut text=String::new();text.try_reserve_exact(value.len()).map_err(|e|crate::error::RusticError::Memory(e.to_string()))?;text.push_str(value);
        self.fields.push((field,FieldValue::Keyword(text)));Ok(())
    }
    pub fn try_add_nested(&mut self,path:&str,child:Document)->crate::error::Result<()> {
        if self.nested.len()==self.nested.capacity(){
            let capacity=self.nested.capacity().saturating_mul(2).max(4);
            self.admit_allocation(capacity.checked_mul(std::mem::size_of::<(String,Document)>()).ok_or_else(||crate::error::RusticError::Memory("nested document slots overflow".into()))?)?;
            self.nested.try_reserve_exact(capacity-self.nested.len()).map_err(|e|crate::error::RusticError::Memory(e.to_string()))?;
        }
        self.admit_allocation(path.len())?;
        let mut key=String::new();key.try_reserve_exact(path.len()).map_err(|e|crate::error::RusticError::Memory(e.to_string()))?;key.push_str(path);
        self.nested.push((key,child));Ok(())
    }

    /// Independently admit a deep copy before any owned field/container is
    /// allocated. Sharing the source decode guard cannot cover a second copy.
    pub fn try_clone(&self) -> crate::error::Result<Self> {
        self.try_clone_on(&crate::admission::shared())
    }
    fn try_clone_on(&self,budget:&std::sync::Arc<crate::admission::Budget>) -> crate::error::Result<Self> {
        use crate::error::RusticError;
        fn overflow()->RusticError{RusticError::Memory("document copy size overflow".into())}
        fn allocation(e:std::collections::TryReserveError)->RusticError{RusticError::Memory(format!("document copy allocation: {e}"))}
        fn text(value:&str)->crate::error::Result<String>{
            let mut out=String::new();out.try_reserve_exact(value.len()).map_err(allocation)?;out.push_str(value);Ok(out)
        }
        let mut bytes=self.fields.len().checked_mul(std::mem::size_of::<(u16,FieldValue)>()).ok_or_else(overflow)?;
        bytes=bytes.checked_add(self.nested.len().checked_mul(std::mem::size_of::<(String,Document)>()).ok_or_else(overflow)?).ok_or_else(overflow)?;
        for (_,value) in &self.fields {
            let n=match value {FieldValue::Text(s)|FieldValue::Keyword(s)=>s.len(),FieldValue::Vector(v)=>v.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(overflow)?,_=>0};
            bytes=bytes.checked_add(n).ok_or_else(overflow)?;
        }
        for (path,_) in &self.nested {bytes=bytes.checked_add(path.len()).ok_or_else(overflow)?;}
        // Arc reservation payload and its strong/weak counters are owned too.
        bytes=bytes.checked_add(std::mem::size_of::<crate::admission::Reservation>()+2*std::mem::size_of::<usize>()).ok_or_else(overflow)?;
        let memory=budget.reserve(bytes as u64,"independent document copy")?;
        let mut fields=Vec::new();fields.try_reserve_exact(self.fields.len()).map_err(allocation)?;
        let mut nested=Vec::new();nested.try_reserve_exact(self.nested.len()).map_err(allocation)?;
        for (id,value) in &self.fields {
            let value=match value {
                FieldValue::Text(s)=>FieldValue::Text(text(s)?),
                FieldValue::Keyword(s)=>FieldValue::Keyword(text(s)?),
                FieldValue::Vector(v)=>{let mut out=Vec::new();out.try_reserve_exact(v.len()).map_err(allocation)?;out.extend_from_slice(v);FieldValue::Vector(out)},
                value=>value.clone(),
            };
            fields.push((*id,value));
        }
        for (path,child) in &self.nested {nested.push((text(path)?,child.try_clone_on(budget)?));}
        Ok(Self{fields,nested,decode_memory:Some(std::sync::Arc::new(memory)),allocation_memory:None,allocation_credit:AllocationCredit(0),shape_triangles:Vec::new(),shape_centroids:Vec::new(),shape_values:Vec::new()})
    }

    /// Heap a decoded document retains, from its capacities: the field slots,
    /// each text/vector payload with its allocator header, nested children.
    /// One pass over the fields (no payload bytes are read).
    pub fn decoded_bytes(&self) -> u64 {
        const HEADER: u64 = 16;
        let payload = |v: &FieldValue| match v {
            FieldValue::Text(s) | FieldValue::Keyword(s) if s.capacity() > 0 => s.capacity() as u64 + HEADER,
            FieldValue::Vector(v) if v.capacity() > 0 => (v.capacity() * std::mem::size_of::<f32>()) as u64 + HEADER,
            _ => 0,
        };
        let fields = (self.fields.capacity() * std::mem::size_of::<(u16, FieldValue)>()) as u64;
        let nested = (self.nested.capacity() * std::mem::size_of::<(String, Document)>()) as u64;
        self.fields.iter().map(|(_, v)| payload(v))
            .chain(self.nested.iter().map(|(path, child)| path.capacity() as u64 + HEADER + child.decoded_bytes()))
            .fold(fields.saturating_add(nested), u64::saturating_add)
    }

    /// Peak heap of decoding a stored document of `encoded` bincode bytes
    /// (see [`STORED_DECODE_FACTOR`]), its decode admission held until
    /// [`Self::decoded_bytes`] is known.
    pub(crate) fn decode_peak_bytes(encoded: u64) -> u64 {
        encoded.saturating_mul(STORED_DECODE_FACTOR).saturating_add(256)
    }

    /// [`Self::decode_peak_bytes`] for the compact encoding
    /// ([`crate::codec::store::encode_doc_compact`]): every field takes at
    /// least 3 encoded bytes and decodes into one exactly reserved slot
    /// (no doubling), payloads into their own bytes plus a header.
    pub(crate) fn decode_peak_bytes_as(encoded: u64, compact: bool) -> u64 {
        if !compact {
            return Self::decode_peak_bytes(encoded);
        }
        let factor = (std::mem::size_of::<(u16, FieldValue)>() as u64).div_ceil(3) + 2;
        encoded.saturating_mul(factor).saturating_add(256)
    }

    /// Whether the document (or a nested child) carries a dense vector.
    pub fn has_vectors(&self) -> bool {
        self.fields.iter().any(|(_, v)| matches!(v, FieldValue::Vector(_))) || self.nested.iter().any(|(_, c)| c.has_vectors())
    }

    pub fn add_text(&mut self, field: u16, text: &str) {
        self.fields.push((field, FieldValue::Text(text.to_string())));
    }

    pub fn add_keyword(&mut self, field: u16, value: &str) {
        self.fields.push((field, FieldValue::Keyword(value.to_string())));
    }

    pub fn add_i64(&mut self, field: u16, value: i64) {
        self.fields.push((field, FieldValue::I64(value)));
    }

    pub fn add_u64(&mut self, field: u16, value: u64) {
        self.fields.push((field, FieldValue::U64(value)));
    }

    pub fn add_f64(&mut self, field: u16, value: f64) {
        self.fields.push((field, FieldValue::F64(value)));
    }

    pub fn add_bool(&mut self, field: u16, value: bool) {
        self.fields.push((field, FieldValue::Bool(value)));
    }

    pub fn add_date(&mut self, field: u16, millis: i64) {
        self.fields.push((field, FieldValue::Date(millis)));
    }

    pub fn add_geo_point(&mut self, field: u16, lat: f64, lon: f64) {
        self.fields.push((field, FieldValue::GeoPoint(lat, lon)));
    }

    pub fn add_vector(&mut self, field: u16, values: Vec<f32>) {
        self.fields.push((field, FieldValue::Vector(values)));
    }

    /// Attach a nested child document under `path` (children keep the
    /// order they are added in: it is their `_nested.offset`).
    pub fn add_nested(&mut self, path: &str, child: Document) {
        self.nested.push((path.to_string(), child));
    }

    /// Number of docs this document occupies when indexed (itself plus
    /// all nested descendants).
    pub fn block_len(&self) -> usize {
        1 + self.nested.iter().map(|(_, c)| c.block_len()).sum::<usize>()
    }

    pub fn get(&self, field: u16) -> Option<&FieldValue> {
        self.fields
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, v)| v)
    }

    pub fn get_all(&self, field: u16) -> impl Iterator<Item = &FieldValue> {
        self.fields.iter().filter(move |(f, _)| *f == field).map(|(_, v)| v)
    }

    /// Owned copies of every value of `field`, in insertion order (the
    /// stored-fields view of a document).
    pub fn stored_values(&self, field: u16) -> Vec<FieldValue> {
        self.get_all(field).cloned().collect()
    }
}

/// A field value in a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FieldValue {
    Text(String),
    Keyword(String),
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Date(i64),
    GeoPoint(f64, f64),
    Vector(Vec<f32>),
}

impl FieldValue {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(s) | FieldValue::Keyword(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FieldValue::I64(v) | FieldValue::Date(v) => Some(*v),
            FieldValue::U64(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            FieldValue::U64(v) => Some(*v),
            FieldValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FieldValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_geo_point(&self) -> Option<(f64, f64)> {
        match self {
            FieldValue::GeoPoint(lat, lon) => Some((*lat, *lon)),
            _ => None,
        }
    }

    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            FieldValue::Vector(v) => Some(v),
            _ => None,
        }
    }
}

/// A scored search hit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub doc_address: DocAddress,
    pub score: f32,
}

impl Hit {
    pub fn new(doc_address: DocAddress, score: f32) -> Self {
        Self { doc_address, score }
    }
}

/// Search results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SearchResults {
    pub total_hits: u64,
    pub hits: Vec<Hit>,
    pub max_score: f32,
    pub took_ms: u64,
}

impl SearchResults {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A range query specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeQuery {
    pub field: u16,
    pub lower: Option<RangeBound>,
    pub upper: Option<RangeBound>,
}

impl RangeQuery {
    pub fn new(field: u16, lower: Option<RangeBound>, upper: Option<RangeBound>) -> Self {
        Self { field, lower, upper }
    }

    pub fn range_i64(field: u16, lower: Option<i64>, upper: Option<i64>) -> Self {
        Self {
            field,
            lower: lower.map(RangeBound::I64),
            upper: upper.map(RangeBound::I64),
        }
    }

    pub fn range_f64(field: u16, lower: Option<f64>, upper: Option<f64>) -> Self {
        Self {
            field,
            lower: lower.map(RangeBound::F64),
            upper: upper.map(RangeBound::F64),
        }
    }
}

/// A bound in a range query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RangeBound {
    I64(i64),
    U64(u64),
    F64(f64),
    Date(i64),
    Inclusive,
    Exclusive,
}

/// An aggregation specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Aggregation {
    Terms {
        field: u16,
        size: usize,
        order: AggOrder,
    },
    Range {
        field: u16,
        ranges: Vec<(Option<RangeBound>, Option<RangeBound>)>,
    },
    Sum {
        field: u16,
    },
    Avg {
        field: u16,
    },
    Min {
        field: u16,
    },
    Max {
        field: u16,
    },
    Count,
}

/// Ordering for aggregation results.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AggOrder {
    CountDesc,
    CountAsc,
    TermDesc,
    TermAsc,
}

/// Aggregation results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregationResults {
    pub buckets: Vec<AggBucket>,
    pub metrics: Vec<AggMetric>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggBucket {
    pub key: String,
    pub doc_count: u64,
    pub sub_aggregations: Option<Box<AggregationResults>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggMetric {
    pub name: String,
    pub value: f64,
}

/// Index settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexSettings {
    pub number_of_shards: u32,
    pub number_of_replicas: u32,
    pub refresh_interval_ms: u64,
    pub max_segment_docs: u32,
    pub compression: Compression,
}

impl Default for IndexSettings {
    fn default() -> Self {
        Self {
            number_of_shards: 1,
            number_of_replicas: 1,
            refresh_interval_ms: 1000,
            max_segment_docs: 10_000_000,
            compression: Compression::Lz4,
        }
    }
}

/// Compression algorithm for index data.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Compression {
    None,
    Lz4,
    Zstd,
}

impl Default for Compression {
    fn default() -> Self {
        Compression::Lz4
    }
}

#[cfg(test)]
mod admitted_document_copy_tests {
    use super::*;
    fn producer_metadata()->u64 {(std::mem::size_of::<std::sync::Mutex<crate::admission::Reservation>>()+2*std::mem::size_of::<usize>()) as u64}
    /// Chunked producer admission: one budget admission covers a small
    /// document; settling leaves exactly what unchunked admission charged.
    #[test]
    fn chunked_producer_admission_settles_to_exactly_the_admitted_bytes() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let mut doc=Document::new();
        doc.admit_allocation(100).unwrap();
        assert_eq!(owner.bytes().current,ALLOCATION_CHUNK,"one chunk for a small document");
        assert_eq!(doc.unclaimed_allocation(),ALLOCATION_CHUNK-100-producer_metadata());
        for _ in 0..10 {doc.try_add_keyword(1,"0123456789").unwrap();}
        assert_eq!(owner.bytes().current,ALLOCATION_CHUNK,"admissions inside the chunk visit no budget");
        let copy=doc.clone();
        assert_eq!(copy.unclaimed_allocation(),0,"a clone never spends the original's credit");
        drop(copy);
        doc.settle_allocation();
        assert_eq!(doc.unclaimed_allocation(),0);
        // Unchunked admission charged: metadata, the 100 bytes, the ten
        // keywords, and the field-slot growth 0 -> 4 -> 8 -> 16.
        let slot=std::mem::size_of::<(u16,FieldValue)>() as u64;
        assert_eq!(owner.bytes().current,producer_metadata()+100+10*10+(4+8+16)*slot);
        doc.settle_allocation();
        assert_eq!(owner.bytes().current,producer_metadata()+100+10*10+(4+8+16)*slot,"settling is idempotent");
        // Past a chunk: exactly the shortfall, no spare credit.
        let before=owner.bytes().current;
        doc.admit_allocation(3*ALLOCATION_CHUNK as usize).unwrap();
        assert_eq!(doc.unclaimed_allocation(),0);
        assert_eq!(owner.bytes().current,before+3*ALLOCATION_CHUNK);
        drop(doc);assert_eq!(owner.bytes().current,0);
    }
    /// A plan (or budget) that cannot take a chunk gets exact admissions:
    /// what fits today still fits, and what does not is still refused
    /// before allocation, charging nothing.
    #[test]
    fn chunked_producer_admission_still_refuses_what_exceeds_the_budget() {
        let lease=crate::admission::Lease::reserve(crate::admission::shared(),4096,"document admission test plan").unwrap();
        let mut doc=Document::new();
        {
            let _plan=lease.enter();
            doc.admit_allocation(1000).unwrap();
            assert_eq!(doc.unclaimed_allocation(),0,"the chunk does not fit the plan: exact admission");
            assert_eq!(4096-lease.available(),1000+producer_metadata());
            doc.admit_allocation(1000).unwrap();
            assert_eq!(4096-lease.available(),2000+producer_metadata());
            assert!(doc.admit_allocation(4096).is_err(),"an admission past the plan is refused");
            assert_eq!(4096-lease.available(),2000+producer_metadata(),"a refused admission charges nothing");
            let mut nested=Document::new();
            assert!(nested.try_add_keyword(1,&"x".repeat(8192)).is_err(),"an oversized first producer is refused too");
            assert!(nested.fields.is_empty());
        }
        drop(doc);
        assert_eq!(lease.available(),4096);
    }
    /// `cargo test -p rustic-core --release --lib producer_admission_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn producer_admission_microbench() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let docs=200_000usize;let started=std::time::Instant::now();
        for i in 0..docs {
            let mut doc=Document::new();
            for f in 0..8u16 {doc.try_add_keyword(f,"AE000041196").unwrap();}
            for f in 8..24u16 {doc.try_push_field((f,FieldValue::F64(i as f64))).unwrap();}
            doc.settle_allocation();
            std::hint::black_box(&doc);
        }
        let secs=started.elapsed().as_secs_f64();
        eprintln!("producer admission: {docs} docs x 24 fields, {:.0} ns/doc",secs*1e9/docs as f64);
        assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn producer_containers_and_nested_parent_copies_keep_owner_until_drop() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let mut child=Document::new();child.try_add_keyword(1,"child-value").unwrap();
        let mut parent=Document::new();parent.try_extend_fields_from(&child).unwrap();
        parent.try_add_nested("path",child).unwrap();
        assert!(owner.bytes().current>0);
        assert_eq!(parent.fields[0].1.as_text(),Some("child-value"));
        assert_ne!(parent.fields[0].1.as_text().unwrap().as_ptr(),parent.nested[0].1.fields[0].1.as_text().unwrap().as_ptr());
        drop(parent);assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn stored_projection_moves_payload_and_admits_overlapping_result_slots() {
        let budget=crate::admission::Budget::new(1<<20);
        let mut source=Document::new();source.add_text(1,"move-this");source.add_text(2,"drop-this");
        let document=source.try_clone_on(&budget).unwrap();
        let pointer=document.fields[0].1.as_text().unwrap().as_ptr();
        let projection=StoredValues::from_document_on(Some(document),1,&budget).unwrap();
        assert_eq!(projection.len(),1);assert_eq!(projection[0].as_text().unwrap().as_ptr(),pointer);
        assert!(budget.used()>0);drop(projection);assert_eq!(budget.used(),0);
        let document=source.try_clone_on(&budget).unwrap();
        let tiny=crate::admission::Budget::new(1);
        assert!(StoredValues::from_document_on(Some(document),1,&tiny).is_err());
        assert_eq!(budget.used(),0);assert_eq!(tiny.used(),0);
    }
    #[test]
    fn independent_nested_copy_retains_own_budget_and_refusal_preserves_source() {
        let mut source=Document::new();source.add_text(1,"payload");source.add_vector(2,vec![1.,2.,3.]);
        let mut child=Document::new();child.add_keyword(3,"child");source.add_nested("nested",child);
        let budget=crate::admission::Budget::new(1<<20);
        let mut copy=source.try_clone_on(&budget).unwrap();
        assert!(budget.used()>0);
        let (_,escaped_child)=copy.nested.pop().unwrap();
        drop(copy);assert!(budget.used()>0,"child keeps its own payload admission");
        drop(escaped_child);assert_eq!(budget.used(),0);
        let tiny=crate::admission::Budget::new(1);
        assert!(source.try_clone_on(&tiny).is_err());assert_eq!(tiny.used(),0);
        assert_eq!(source.fields[0].1.as_text(),Some("payload"));
        assert_eq!(source.nested[0].1.fields[0].1.as_text(),Some("child"));
    }
}
