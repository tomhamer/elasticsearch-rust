//! Index mappings: parsing, validation, merging (put-mapping rules),
//! dynamic mapping + dynamic templates, and the resolved per-field view
//! ([`FieldLookup`]) that document parsing and query building use.
//!
//! Every leaf field (including multi-fields like `title.keyword`) is backed
//! by one core field id (two for `geo_point`: `<path>#lat`, `<path>#lon`).
//! Ids are assigned once and persisted in [`Mappings::core_ids`].

mod wire;

use crate::analysis::{AnalysisRegistry, IndexAnalyzer};
use crate::date::{DateFormatter, DEFAULT_FORMAT};
use crate::error::{EsError, Result};
use rustic_core::analysis::Analyzer;
use rustic_core::types::{Field as CoreField, FieldType as CoreType};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub const ID_FIELD: u16 = 0;
pub const SOURCE_FIELD: u16 = 1;
pub const VERSION_FIELD: u16 = 2;
pub const SEQ_NO_FIELD: u16 = 3;
pub const ROUTING_FIELD: u16 = 4;
pub const IGNORED_FIELD: u16 = 5;
pub const PRIMARY_TERM_FIELD: u16 = 6;
/// Per-document stored-fields block: JSON `{field: [values]}` of the fields
/// mapped `store: true` (see [`crate::stored`]).
pub const STORED_FIELD: u16 = 7;
/// `core_ids` key of the per-document stored-fields block's own field (a
/// stored, never indexed keyword; indices whose mapping predates it keep
/// the block in [`STORED_FIELD`], shared with `_field_names`).
pub const STORED_BLOCK_KEY: &str = "#stored";
/// The schema name the stored-fields block's field had in commits written
/// before it was named by [`STORED_BLOCK_KEY`] (adopted as that key).
pub const LEGACY_STORED_BLOCK_NAME: &str = "_stored_fields";
/// Hidden `_field_names`: the names of the fields a doc has a value for but
/// no doc-values / postings signal (binary without doc values, unindexed or
/// empty text, ...). `exists` falls back to a term query on it.
pub const FIELD_NAMES_FIELD: u16 = 7;
/// Per-document `_ignored_source` blob (base64 of the ES binary format, see
/// [`crate::synthetic::ignored_entry`]).
pub const IGNORED_SOURCE_FIELD: u16 = 8;
pub const FIRST_USER_FIELD: u16 = 9;
/// The `_doc_count` metadata field (`DocCountFieldMapper`): the number of
/// documents a (pre-aggregated) document stands for in bucket counts.
pub const DOC_COUNT_FIELD_NAME: &str = "_doc_count";
/// The `_index_mode` metadata field name.
pub const INDEX_MODE_FIELD: &str = "_index_mode";

/// Core fields every index has.
pub fn reserved_core_fields() -> Vec<CoreField> {
    // `_id` doc values: from segment format v15 sealed segments keep none
    // (ES has none); a hit's `_id` comes from the stored fields, sorts and
    // aggregations from a column derived from the postings.
    let mut id = CoreField::new(ID_FIELD, "_id", CoreType::Keyword).with_column_from_postings(true);
    id.fast = true;
    let mut src = CoreField::new(SOURCE_FIELD, "_source", CoreType::Keyword);
    src.indexed = false;
    src.fast = false;
    // `_version` and `_primary_term` are doc values only in ES (never
    // range-queried through points): no points index.
    // `_version`, `_seq_no`, `_primary_term`: stored values come from their
    // columns from segment format v15 (not in the stored fields, as in ES).
    let mut ver = CoreField::new(VERSION_FIELD, "_version", CoreType::I64).with_points(false).with_column_stored(true);
    ver.indexed = false;
    let mut seq = CoreField::new(SEQ_NO_FIELD, "_seq_no", CoreType::I64).with_column_stored(true);
    seq.indexed = false;
    let mut routing = CoreField::new(ROUTING_FIELD, "_routing", CoreType::Keyword);
    routing.fast = false;
    let ignored = CoreField::new(IGNORED_FIELD, "_ignored", CoreType::Keyword);
    let mut pt = CoreField::new(PRIMARY_TERM_FIELD, "_primary_term", CoreType::I64).with_points(false).with_column_stored(true);
    pt.indexed = false;
    let mut field_names = CoreField::new(FIELD_NAMES_FIELD, "_field_names", CoreType::Keyword);
    field_names.fast = false;
    let mut ignored_src = CoreField::new(IGNORED_SOURCE_FIELD, "_ignored_source", CoreType::Keyword);
    ignored_src.indexed = false;
    ignored_src.fast = false;
    vec![id, src, ver, seq, routing, ignored, pt, field_names, ignored_src]
}

/// The reserved `long` metadata fields that can be sorted / aggregated on
/// (`_seq_no`, `_primary_term`): `(name, core id)`, or None.
pub fn reserved_long_field_id(name: &str) -> Option<u16> {
    match name {
        "_seq_no" => Some(SEQ_NO_FIELD),
        "_primary_term" => Some(PRIMARY_TERM_FIELD),
        _ => None,
    }
}

/// A `doc_values` sub-parameter of a field in a strict columnar index: the
/// field's own value, else the index default (`index.mapping.doc_values.*`).
/// `None` outside columnar modes or without doc values.
fn dv_flag(f: &FieldMapping, reg: &AnalysisRegistry, k: &str, default: bool) -> Option<bool> {
    if !reg.strict_columnar() {
        return None;
    }
    match f.params.get("doc_values") {
        Some(Value::Object(o)) => o.get(k).and_then(Value::as_bool).or(Some(default)),
        Some(Value::Bool(false)) => None,
        _ if matches!(f.ty, EsType::Alias | EsType::Object | EsType::Nested) => None,
        _ => Some(default),
    }
}

/// A synthetic resolved field for a reserved `long` metadata field (doc
/// values only, not indexed), for the sort / aggregation paths, which
/// otherwise only know mapped fields.
pub fn reserved_long_field(name: &str, id: u16) -> ResolvedField {
    ResolvedField {
        name: name.to_string(),
        ty: EsType::Long,
        core_id: Some(id),
        lat_id: None,
        lon_id: None,
        geo_id: None,
        index_analyzer: None,
        index_analyzer_name: None,
        search_analyzer: None,
        search_quote_analyzer: None,
        normalizer: None,
        normalizer_skip_store: false,
        date_format: DateFormatter::new(DEFAULT_FORMAT).unwrap(),
        ignore_above: None,
        null_value: None,
        coerce: true,
        ignore_malformed: false,
        scaling_factor: 1.0,
        dims: None,
        similarity: "cosine".to_string(),
        indexed: false,
        doc_values: true,
        copy_to: Vec::new(),
        multi_fields: Vec::new(),
        is_multi_field: false,
        alias_path: None,
        constant_value: None,
        nested_path: None,
        meta: None,
        declared: EsType::Long,
        ext: None,
        aux: Vec::new(),
        position_increment_gap: 100,
        norms: true,
        discount_overlaps: true,
        stored: false,
        fielddata: false,
        time_series_dimension: false,
        time_series_metric: None,
        script: None,
        script_error_continue: false,
        single_value: false,
        required: false,
        index_declared: false,
        skipper: false,
    }
}

/// A synthetic resolved field for the `_index` metadata field of one
/// index's segments, for the aggregation paths (ES `IndexFieldMapper`:
/// every document's value is its concrete index name, read as a constant
/// as `constant_keyword` is).
pub fn index_name_field(index: &str) -> ResolvedField {
    let mut f = reserved_long_field("_index", 0);
    f.ty = EsType::ConstantKeyword;
    f.declared = EsType::ConstantKeyword;
    f.core_id = None;
    f.constant_value = Some(Value::String(index.to_string()));
    f
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EsType {
    Text,
    MatchOnlyText,
    PatternText,
    Keyword,
    ConstantKeyword,
    CountedKeyword,
    Wildcard,
    Flattened,
    Long,
    Integer,
    Short,
    Byte,
    Double,
    Float,
    HalfFloat,
    ScaledFloat,
    UnsignedLong,
    Boolean,
    Date,
    DateNanos,
    Ip,
    GeoPoint,
    DenseVector,
    Binary,
    Alias,
    Object,
    Nested,
    AnnotatedText,
    SearchAsYouType,
    SemanticText,
    /// ES 9.4+ `semantic` (`SemanticFieldMapper`): a multimodal inference
    /// field (text and base64 media values) over an `embedding` model.
    Semantic,
    TokenCount,
    Murmur3,
    Version,
    Completion,
    Join,
    Percolator,
    IntegerRange,
    LongRange,
    FloatRange,
    DoubleRange,
    DateRange,
    IpRange,
    GeoShape,
    Point,
    Shape,
    Histogram,
    ExponentialHistogram,
    TDigest,
    AggregateMetricDouble,
    SparseVector,
    RankFeature,
    RankFeatures,
    RankVectors,
}

impl EsType {
    pub fn parse(s: &str) -> Option<EsType> {
        Some(match s {
            "text" => EsType::Text,
            "match_only_text" => EsType::MatchOnlyText,
            "pattern_text" => EsType::PatternText,
            "keyword" => EsType::Keyword,
            "constant_keyword" => EsType::ConstantKeyword,
            "counted_keyword" => EsType::CountedKeyword,
            "wildcard" => EsType::Wildcard,
            "flattened" => EsType::Flattened,
            "long" => EsType::Long,
            "integer" => EsType::Integer,
            "short" => EsType::Short,
            "byte" => EsType::Byte,
            "double" => EsType::Double,
            "float" => EsType::Float,
            "half_float" => EsType::HalfFloat,
            "scaled_float" => EsType::ScaledFloat,
            "unsigned_long" => EsType::UnsignedLong,
            "boolean" => EsType::Boolean,
            "date" => EsType::Date,
            "date_nanos" => EsType::DateNanos,
            "ip" => EsType::Ip,
            "geo_point" => EsType::GeoPoint,
            "dense_vector" => EsType::DenseVector,
            "binary" => EsType::Binary,
            "alias" => EsType::Alias,
            "object" => EsType::Object,
            "annotated_text" => EsType::AnnotatedText,
            "search_as_you_type" => EsType::SearchAsYouType,
            "semantic_text" => EsType::SemanticText,
            "semantic" => EsType::Semantic,
            "token_count" => EsType::TokenCount,
            "murmur3" => EsType::Murmur3,
            "version" => EsType::Version,
            "completion" => EsType::Completion,
            "join" => EsType::Join,
            "percolator" => EsType::Percolator,
            "integer_range" => EsType::IntegerRange,
            "long_range" => EsType::LongRange,
            "float_range" => EsType::FloatRange,
            "double_range" => EsType::DoubleRange,
            "date_range" => EsType::DateRange,
            "ip_range" => EsType::IpRange,
            "geo_shape" => EsType::GeoShape,
            "point" => EsType::Point,
            "shape" => EsType::Shape,
            "histogram" => EsType::Histogram,
            "exponential_histogram" => EsType::ExponentialHistogram,
            "tdigest" => EsType::TDigest,
            "aggregate_metric_double" => EsType::AggregateMetricDouble,
            "sparse_vector" => EsType::SparseVector,
            "rank_feature" => EsType::RankFeature,
            "rank_features" => EsType::RankFeatures,
            "rank_vectors" => EsType::RankVectors,
            "nested" => EsType::Nested,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            EsType::Text => "text",
            EsType::MatchOnlyText => "match_only_text",
            EsType::PatternText => "pattern_text",
            EsType::Keyword => "keyword",
            EsType::ConstantKeyword => "constant_keyword",
            EsType::CountedKeyword => "counted_keyword",
            EsType::Wildcard => "wildcard",
            EsType::Flattened => "flattened",
            EsType::Long => "long",
            EsType::Integer => "integer",
            EsType::Short => "short",
            EsType::Byte => "byte",
            EsType::Double => "double",
            EsType::Float => "float",
            EsType::HalfFloat => "half_float",
            EsType::ScaledFloat => "scaled_float",
            EsType::UnsignedLong => "unsigned_long",
            EsType::Boolean => "boolean",
            EsType::Date => "date",
            EsType::DateNanos => "date_nanos",
            EsType::Ip => "ip",
            EsType::GeoPoint => "geo_point",
            EsType::DenseVector => "dense_vector",
            EsType::Binary => "binary",
            EsType::Alias => "alias",
            EsType::Object => "object",
            EsType::Nested => "nested",
            EsType::AnnotatedText => "annotated_text",
            EsType::SearchAsYouType => "search_as_you_type",
            EsType::SemanticText => "semantic_text",
            EsType::Semantic => "semantic",
            EsType::TokenCount => "token_count",
            EsType::Murmur3 => "murmur3",
            EsType::Version => "version",
            EsType::Completion => "completion",
            EsType::Join => "join",
            EsType::Percolator => "percolator",
            EsType::IntegerRange => "integer_range",
            EsType::LongRange => "long_range",
            EsType::FloatRange => "float_range",
            EsType::DoubleRange => "double_range",
            EsType::DateRange => "date_range",
            EsType::IpRange => "ip_range",
            EsType::GeoShape => "geo_shape",
            EsType::Point => "point",
            EsType::Shape => "shape",
            EsType::Histogram => "histogram",
            EsType::ExponentialHistogram => "exponential_histogram",
            EsType::TDigest => "tdigest",
            EsType::AggregateMetricDouble => "aggregate_metric_double",
            EsType::SparseVector => "sparse_vector",
            EsType::RankFeature => "rank_feature",
            EsType::RankFeatures => "rank_features",
            EsType::RankVectors => "rank_vectors",
        }
    }

    pub fn is_text(&self) -> bool {
        matches!(self, EsType::Text | EsType::MatchOnlyText | EsType::PatternText | EsType::AnnotatedText | EsType::SearchAsYouType | EsType::SemanticText | EsType::Semantic)
    }

    /// An inference field type (`semantic_text` or `semantic`).
    pub fn is_semantic(&self) -> bool {
        matches!(self, EsType::SemanticText | EsType::Semantic)
    }

    pub fn is_keyword_like(&self) -> bool {
        matches!(self, EsType::Keyword | EsType::ConstantKeyword | EsType::CountedKeyword | EsType::Wildcard | EsType::Flattened)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::UnsignedLong)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub fn is_date(&self) -> bool {
        matches!(self, EsType::Date | EsType::DateNanos)
    }

    /// Allowed mapping parameters (besides `type`).
    fn allowed_params(&self) -> &'static [&'static str] {
        const COMMON: &[&str] = &["meta", "boost"];
        let _ = COMMON;
        match self {
            EsType::Text => &[
                "analyzer", "search_analyzer", "search_quote_analyzer", "index", "index_options", "index_phrases", "index_prefixes", "norms",
                "position_increment_gap", "store", "similarity", "term_vector", "fielddata", "fielddata_frequency_filter",
                "eager_global_ordinals", "fields", "copy_to", "meta", "boost", "doc_values",
            ],
            EsType::MatchOnlyText => &["fields", "meta", "copy_to", "store", "index", "doc_values"],
            EsType::PatternText => &["meta", "index_options", "analyzer", "disable_templating", "copy_to", "fields"],
            EsType::Keyword => &[
                "index", "doc_values", "store", "ignore_above", "null_value", "normalizer", "normalizer_skip_store_original_value", "norms", "index_options", "eager_global_ordinals",
                "similarity", "split_queries_on_whitespace", "fields", "copy_to", "meta", "time_series_dimension", "script", "on_script_error", "boost",
            ],
            EsType::ConstantKeyword => &["value", "meta"],
            EsType::Wildcard => &["ignore_above", "null_value", "meta", "fields", "copy_to"],
            EsType::Flattened => &[
                "depth_limit", "doc_values", "eager_global_ordinals", "ignore_above", "index", "index_options", "null_value", "similarity",
                "split_queries_on_whitespace", "meta", "time_series_dimensions", "properties", "passthrough",
            ],
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::UnsignedLong => &[
                "index", "doc_values", "store", "null_value", "coerce", "ignore_malformed", "meta", "copy_to", "fields", "time_series_dimension",
                "time_series_metric", "script", "on_script_error", "boost",
            ],
            EsType::ScaledFloat => &[
                "scaling_factor", "index", "doc_values", "store", "null_value", "coerce", "ignore_malformed", "meta", "copy_to", "fields",
                "time_series_dimension", "time_series_metric", "script", "on_script_error", "boost",
            ],
            EsType::Boolean => &["index", "doc_values", "store", "null_value", "ignore_malformed", "meta", "copy_to", "fields", "script", "on_script_error", "boost", "time_series_dimension"],
            EsType::Date | EsType::DateNanos => &[
                "format", "locale", "index", "doc_values", "store", "null_value", "ignore_malformed", "meta", "copy_to", "fields", "script",
                "on_script_error", "boost",
            ],
            EsType::Ip => &[
                "index", "doc_values", "store", "null_value", "ignore_malformed", "meta", "copy_to", "fields", "script", "on_script_error",
                "time_series_dimension", "boost",
            ],
            EsType::GeoPoint => &["ignore_malformed", "ignore_z_value", "index", "doc_values", "store", "null_value", "meta", "script", "on_script_error", "copy_to", "fields", "time_series_metric", "time_series_dimension"],
            EsType::DenseVector => &["dims", "element_type", "index", "similarity", "index_options", "meta"],
            EsType::Binary => &["doc_values", "store", "meta", "synthetic_source_keep", "copy_to"],
            EsType::Alias => &["path"],
            EsType::Object | EsType::Nested => &[],
            t => crate::fieldtypes::allowed_params(*t),
        }
    }

    /// Parameters that a mapping update may change in place.
    fn updatable(param: &str) -> bool {
        matches!(
            param,
            "ignore_above" | "search_analyzer" | "search_quote_analyzer" | "meta" | "boost" | "eager_global_ordinals" | "fields" | "copy_to" | "ignore_malformed" | "coerce" | "fielddata" | "fielddata_frequency_filter" | "dims"
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dynamic {
    True,
    False,
    Strict,
    Runtime,
}

impl Dynamic {
    fn parse(v: &Value) -> Result<Dynamic> {
        match v {
            Value::Bool(true) => Ok(Dynamic::True),
            Value::Bool(false) => Ok(Dynamic::False),
            Value::String(s) => match s.as_str() {
                "true" => Ok(Dynamic::True),
                "false" => Ok(Dynamic::False),
                "strict" => Ok(Dynamic::Strict),
                "runtime" => Ok(Dynamic::Runtime),
                other => Err(crate::fieldtypes::params::iae_caused(
                    "Could not convert [dynamic.dynamic] to boolean",
                    EsError::illegal_argument(format!("Failed to parse value [{other}] as only [true] or [false] are allowed.")),
                )),
            },
            Value::Null => {
                let m = "Cannot invoke \"Object.toString()\" because \"fieldNode\" is null";
                Err(EsError::mapper_parsing(m).caused_by(EsError::new(400, "null_pointer_exception", m)))
            }
            other => Err(crate::fieldtypes::params::iae_caused(
                "Could not convert [dynamic.dynamic] to boolean",
                EsError::illegal_argument(format!("Failed to parse value [{other}] as only [true] or [false] are allowed.")),
            )),
        }
    }
    fn to_json(self) -> Value {
        match self {
            Dynamic::True => json!("true"),
            Dynamic::False => json!("false"),
            Dynamic::Strict => json!("strict"),
            Dynamic::Runtime => json!("runtime"),
        }
    }
}

/// A leaf field mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldMapping {
    pub ty: EsType,
    /// Parameters as given (minus `type` and `fields`).
    pub params: Map<String, Value>,
    pub fields: BTreeMap<String, FieldMapping>,
}

/// An `object` / `nested` mapping.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ObjectMapping {
    pub nested: bool,
    pub explicit_type: bool,
    pub dynamic: Option<Dynamic>,
    pub enabled: Option<bool>,
    pub params: Map<String, Value>,
    pub properties: BTreeMap<String, Mapper>,
    /// Root of a strict columnar index (`columnar`, `logsdb_columnar`):
    /// objects are flattened into dotted leaf names
    /// (`ObjectMapper.Defaults.SUBOBJECTS_COLUMNAR`).
    pub columnar: bool,
    /// Columnar root: per-object settings by full dotted object path
    /// (`prefix_properties`).
    pub prefix: BTreeMap<String, PrefixProps>,
}

/// `PrefixProperties`: an object's settings kept by a columnar root.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PrefixProps {
    pub dynamic: Option<Dynamic>,
    pub passthrough: Option<i64>,
    pub enabled: Option<bool>,
}

impl PrefixProps {
    /// `PrefixProperties.merge`: incoming facets win.
    fn merge(&mut self, o: &PrefixProps) {
        if o.dynamic.is_some() {
            self.dynamic = o.dynamic;
        }
        if o.passthrough.is_some() {
            self.passthrough = o.passthrough;
        }
        if o.enabled.is_some() {
            self.enabled = o.enabled;
        }
    }

    fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(d) = self.dynamic {
            m.insert("dynamic".into(), d.to_json());
        }
        if let Some(p) = self.passthrough {
            m.insert("passthrough".into(), json!(p));
        }
        if let Some(e) = self.enabled {
            m.insert("enabled".into(), json!(e));
        }
        Value::Object(m)
    }

    /// `RootObjectMapper` parsing of one `prefix_properties` entry.
    fn parse(key: &str, v: &Value) -> Result<PrefixProps> {
        let o = v.as_object().ok_or_else(|| EsError::mapper_parsing(format!("[prefix_properties.{key}] must be an object")))?;
        let mut pp = PrefixProps::default();
        if let Some(d) = o.get("dynamic").filter(|d| !d.is_null()) {
            let s = d.as_str().ok_or_else(|| EsError::mapper_parsing(format!("[prefix_properties.{key}.dynamic] must be a string value")))?;
            pp.dynamic = Some(match s.to_ascii_lowercase().as_str() {
                "runtime" => return Err(EsError::mapper_parsing(format!("[prefix_properties.{key}.dynamic] does not support [runtime]"))),
                "true" => Dynamic::True,
                "false" => Dynamic::False,
                "strict" => Dynamic::Strict,
                _ => return Err(EsError::mapper_parsing(format!("unknown value [{s}] for [prefix_properties.{key}.dynamic]"))),
            });
        }
        if let Some(p) = o.get("passthrough").filter(|p| !p.is_null()) {
            pp.passthrough = Some(p.as_i64().or_else(|| p.as_f64().map(|f| f as i64)).or_else(|| p.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| EsError::mapper_parsing(format!("[prefix_properties.{key}.passthrough] must be an integer value")))?);
        }
        if let Some(e) = o.get("enabled").filter(|e| !e.is_null()) {
            pp.enabled = Some(crate::fieldtypes::params::object_bool(&format!("prefix_properties.{key}.enabled"), e)?);
        }
        Ok(pp)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mapper {
    Field(FieldMapping),
    Object(ObjectMapping),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicTemplate {
    pub name: String,
    pub raw: Value,
    match_mapping_type: Vec<String>,
    unmatch_mapping_type: Vec<String>,
    matches: Vec<String>,
    unmatches: Vec<String>,
    path_match: Vec<String>,
    path_unmatch: Vec<String>,
    regex: bool,
    pub mapping: Option<Value>,
}

/// The whole mapping of an index.
#[derive(Clone, Debug, PartialEq)]
pub struct Mappings {
    pub root: ObjectMapping,
    pub dynamic_templates: Vec<DynamicTemplate>,
    /// `dynamic_templates` was given explicitly (an empty list then prints as `[]`).
    pub dt_explicit: bool,
    pub date_detection: Option<bool>,
    pub numeric_detection: Option<bool>,
    pub dynamic_date_formats: Option<Vec<String>>,
    pub source: Option<Value>,
    /// The `index.mapping.source.mode` setting value (wins over the mapping's
    /// `_source.mode` when present).
    pub source_mode_setting: Option<String>,
    pub routing: Option<Value>,
    pub meta: Option<Value>,
    pub runtime: Option<Value>,
    /// `_size` (mapper-size): `{"enabled": true}` indexes the source size.
    pub size: Option<Value>,
    /// `_data_stream_timestamp: {enabled: true}` (time series / data stream indices).
    pub data_stream_timestamp: bool,
    /// `_id.mode` when configured (`document` / `columnar`).
    pub id_mode: Option<String>,
    /// Leaf path (and `<geo>#lat`/`#lon`) → core field id.
    pub core_ids: BTreeMap<String, u16>,
    pub next_core_id: u16,
}

impl Default for Mappings {
    fn default() -> Self {
        Self {
            root: ObjectMapping::default(),
            dynamic_templates: Vec::new(),
            dt_explicit: false,
            date_detection: None,
            numeric_detection: None,
            dynamic_date_formats: None,
            source: None,
            source_mode_setting: None,
            routing: None,
            meta: None,
            runtime: None,
            size: None,
            data_stream_timestamp: false,
            id_mode: None,
            core_ids: BTreeMap::new(),
            next_core_id: FIRST_USER_FIELD,
        }
    }
}

fn wrap_parse_err(e: EsError) -> EsError {
    if e.kind == "mapper_parsing_exception" && !e.reason.starts_with("Failed to parse mapping") {
        // `Failed to parse mapping: m` caused by an inner exception: when the
        // cause repeats the message (an `illegal_argument_exception`) it is the
        // direct cause; otherwise the mapper exception sits in between.
        let reason = format!("Failed to parse mapping: {}", e.reason);
        let cause = match e.caused_by.clone() {
            Some(c) if c.reason == e.reason => *c,
            _ => e,
        };
        EsError::mapper_parsing(reason).caused_by(cause)
    } else {
        e
    }
}

/// The Java class name ES prints for a JSON value in "but got a class .." messages.
fn java_class_of(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "java.lang.String",
        Value::Bool(_) => "java.lang.Boolean",
        Value::Number(n) if n.is_f64() => "java.lang.Double",
        Value::Number(_) => "java.lang.Integer",
        Value::Array(_) => "java.util.ArrayList",
        Value::Object(_) => "java.util.LinkedHashMap",
        Value::Null => "null",
    }
}

fn simple_match(pattern: &str, s: &str) -> bool {
    // ES Regex.simpleMatch: '*' wildcards only.
    if !pattern.contains('*') {
        return pattern == s;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0;
    for (i, p) in parts.iter().enumerate() {
        if p.is_empty() {
            continue;
        }
        if i == 0 {
            if !s.starts_with(p) {
                return false;
            }
            pos = p.len();
        } else if i == parts.len() - 1 {
            return s.len() >= pos + p.len() && s[pos..].ends_with(p);
        } else {
            match s[pos..].find(p) {
                Some(idx) => pos += idx + p.len(),
                None => return false,
            }
        }
    }
    true
}

pub fn wildcard_match(pattern: &str, s: &str) -> bool {
    simple_match(pattern, s)
}

fn str_or_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

impl DynamicTemplate {
    fn parse(name: &str, v: &Value, registry: &AnalysisRegistry) -> Result<Self> {
        use crate::fieldtypes::params::{iae, jstr, mpe};
        let obj = v.as_object().ok_or_else(|| EsError::mapper_parsing(format!("dynamic template [{name}] must be an object")))?;
        for k in obj.keys() {
            if !matches!(k.as_str(), "match_mapping_type" | "unmatch_mapping_type" | "match" | "unmatch" | "path_match" | "path_unmatch" | "match_pattern" | "mapping" | "runtime") {
                return Err(iae(format!("Illegal dynamic template parameter: [{k}]")));
            }
        }
        if obj.contains_key("mapping") && obj.contains_key("runtime") {
            return Err(mpe(format!("mapping and runtime cannot be both specified in the same dynamic template [{name}]")));
        }
        if !obj.contains_key("mapping") && !obj.contains_key("runtime") {
            return Err(mpe(format!("template [{name}] must have either mapping or runtime set")));
        }
        let regex = match obj.get("match_pattern").map(|p| p.as_str()) {
            None => false,
            Some(Some("regex")) => true,
            Some(Some("simple")) => false,
            Some(other) => return Err(iae(format!("No matching pattern matched on [{}]", other.map(str::to_string).unwrap_or_else(|| jstr(&obj["match_pattern"]))))),
        };
        const MAPPING_TYPES: &[&str] = &["object", "string", "long", "double", "boolean", "date", "binary"];
        const RUNTIME_TYPES: &[&str] = &["string", "long", "double", "boolean", "date"];
        let is_runtime = obj.contains_key("runtime");
        let known = if is_runtime { RUNTIME_TYPES } else { MAPPING_TYPES };
        let mmt = str_or_list(obj.get("match_mapping_type"));
        let ummt = str_or_list(obj.get("unmatch_mapping_type"));
        for t in mmt.iter().chain(ummt.iter()) {
            if t != "*" && !MAPPING_TYPES.contains(&t.as_str()) {
                return Err(iae(format!("No field type matched on [{t}], possible values are [{}]", MAPPING_TYPES.join(", "))));
            }
        }
        if regex {
            for key in ["match", "unmatch", "path_match", "path_unmatch"] {
                for p in str_or_list(obj.get(key)) {
                    if let Err(e) = regex::Regex::new(&p) {
                        let mut first = e.to_string().lines().last().unwrap_or("").trim_start_matches("error: ").to_string();
                        if let Some(c) = first.get(..1) {
                            first = format!("{}{}", c.to_uppercase(), &first[1..]);
                        }
                        let m = format!("Pattern [{p}] of type [regex] is invalid. Cannot create dynamic template [{name}].");
                        return Err(EsError::mapper_parsing(m).caused_by(EsError::new(400, "pattern_syntax_exception", format!("{first} near index 0\n{p}\n^"))));
                    }
                }
            }
        }
        // Validate the mapping against every dynamic type the template can match.
        let mapping = obj.get("mapping").cloned();
        let candidates: Vec<&str> = known.iter().copied().filter(|t| (mmt.is_empty() || mmt.iter().any(|m| m == "*" || m == t)) && !ummt.iter().any(|u| u == t)).collect();
        // A template without match conditions (only usable by name) is not validated.
        let has_conditions = ["match_mapping_type", "unmatch_mapping_type", "match", "unmatch", "path_match", "path_unmatch"].iter().any(|k| obj.contains_key(*k));
        if let (Some(Value::Object(m)), true) = (&mapping, has_conditions) {
            let mut last: Option<EsError> = None;
            let mut ok = candidates.is_empty();
            for t in &candidates {
                let dt = match *t {
                    "string" => "text",
                    "double" => "float",
                    o => o,
                };
                let mut spec = substitute(&Value::Object(m.clone()), "__dummy__", dt).as_object().cloned().unwrap_or_default();
                if spec.get("type").and_then(Value::as_str).is_some_and(|x| x == "{dynamic_type}" || x == "{dynamicType}") {
                    spec.insert("type".into(), json!(dt));
                }
                spec.entry("type").or_insert_with(|| json!(dt));
                let mut holder = ObjectMapping::default();
                match holder.insert_property("", &format!("__dynamic__{name}"), &Value::Object(spec), registry, 0) {
                    Ok(()) => {
                        ok = true;
                        break;
                    }
                    Err(e) => last = Some(e),
                }
            }
            if !ok {
                let inner = last.map(|e| match e.caused_by.as_deref() {
                    Some(c) if c.reason == e.reason => c.clone(),
                    _ if e.reason.starts_with("The mapper type [") => {
                        let t = e.reason.trim_start_matches("The mapper type [").split(']').next().unwrap_or("");
                        EsError::illegal_argument(format!("No mapper found for type [{t}]"))
                    }
                    _ => EsError { caused_by: None, ..e },
                });
                let msg = format!(
                    "dynamic template [{name}] has invalid content [{}], attempted to validate it with the following match_mapping_type: [{}]",
                    serde_json::to_string(v).unwrap_or_default(),
                    candidates.join(", ")
                );
                let mut e = EsError::illegal_argument(msg.clone());
                if let Some(i) = inner {
                    e = e.caused_by(i);
                }
                return Err(EsError::mapper_parsing(msg).caused_by(e));
            }
        } else if is_runtime {
            let rt = obj.get("runtime").cloned().unwrap_or(Value::Null);
            if let Some(t) = rt.get("type").and_then(Value::as_str) {
                if !RUNTIME_TYPES.contains(&t) && !matches!(t, "keyword" | "ip" | "composite" | "geo_point") {
                    let msg = format!(
                        "dynamic template [{name}] has invalid content [{}], attempted to validate it with the following match_mapping_type: [{}]",
                        serde_json::to_string(v).unwrap_or_default(),
                        candidates.join(", ")
                    );
                    let e = EsError::illegal_argument(msg.clone()).caused_by(EsError::illegal_argument(format!("No runtime field found for type [{t}]")));
                    return Err(EsError::mapper_parsing(msg).caused_by(e));
                }
            }
        }
        let mapping = obj.get("mapping").cloned().or_else(|| obj.get("runtime").map(|_| Value::Null));
        // A dense_vector template's `dims` is validated when the template is
        // parsed (ES tries the template against a dummy field).
        if let Some(m) = mapping.as_ref().and_then(Value::as_object) {
            if m.get("type").and_then(Value::as_str) == Some("dense_vector") {
                if let Some(d) = m.get("dims").and_then(Value::as_u64) {
                    let max = crate::vectors::max_dims(m.get("element_type").and_then(Value::as_str));
                    // A bit vector's lower bound and multiple of 8 are
                    // checked when the field itself is parsed.
                    if d == 0 || d > max {
                        return Err(EsError::mapper_parsing(format!("The number of dimensions should be in the range [1, {max}] but was [{d}]")));
                    }
                }
            }
        }
        Ok(Self {
            name: name.to_string(),
            raw: v.clone(),
            match_mapping_type: mmt,
            unmatch_mapping_type: ummt,
            matches: str_or_list(obj.get("match")),
            unmatches: str_or_list(obj.get("unmatch")),
            path_match: str_or_list(obj.get("path_match")),
            path_unmatch: str_or_list(obj.get("path_unmatch")),
            regex,
            mapping: mapping.filter(|m| !m.is_null()),
        })
    }

    fn pat(&self, p: &str, s: &str) -> bool {
        if self.regex {
            regex::Regex::new(&format!("^(?:{p})$")).map(|r| r.is_match(s)).unwrap_or(false)
        } else {
            simple_match(p, s)
        }
    }

    fn matches(&self, name: &str, path: &str, ty: &str) -> bool {
        // A template without any match condition only applies when a write
        // requests it by name (bulk `dynamic_templates`).
        if self.match_mapping_type.is_empty()
            && self.unmatch_mapping_type.is_empty()
            && self.matches.is_empty()
            && self.unmatches.is_empty()
            && self.path_match.is_empty()
            && self.path_unmatch.is_empty()
        {
            return false;
        }
        if !self.match_mapping_type.is_empty() && !self.match_mapping_type.iter().any(|t| t == "*" || t == ty) {
            return false;
        }
        if self.unmatch_mapping_type.iter().any(|t| t == ty) {
            return false;
        }
        if !self.matches.is_empty() && !self.matches.iter().any(|p| self.pat(p, name)) {
            return false;
        }
        if self.unmatches.iter().any(|p| self.pat(p, name)) {
            return false;
        }
        if !self.path_match.is_empty() && !self.path_match.iter().any(|p| simple_match(p, path)) {
            return false;
        }
        if self.path_unmatch.iter().any(|p| simple_match(p, path)) {
            return false;
        }
        true
    }
}

fn substitute(v: &Value, name: &str, dynamic_type: &str) -> Value {
    substitute_with(v, name, dynamic_type, &std::collections::HashMap::new())
}

/// `DynamicTemplate.processMap`: `{name}` / `{dynamic_type}` (single or
/// double braces) and `{{param}}` placeholders (bulk
/// `dynamic_template_params`; unknown params become empty) in keys and
/// string values.
fn substitute_with(v: &Value, name: &str, dynamic_type: &str, params: &std::collections::HashMap<String, String>) -> Value {
    match v {
        Value::String(s) => Value::String(process_string(s, name, dynamic_type, params)),
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (process_string(k, name, dynamic_type, params), substitute_with(v, name, dynamic_type, params))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(|x| substitute_with(x, name, dynamic_type, params)).collect()),
        other => other.clone(),
    }
}

fn process_string(s: &str, name: &str, dynamic_type: &str, params: &std::collections::HashMap<String, String>) -> String {
    let mut s = s.to_string();
    if s.contains("{{") {
        s = s.replace("{{name}}", name).replace("{{dynamic_type}}", dynamic_type).replace("{{dynamicType}}", dynamic_type);
    }
    if s.contains('{') {
        s = s.replace("{name}", name).replace("{dynamic_type}", dynamic_type).replace("{dynamicType}", dynamic_type);
    }
    if !s.contains("{{") {
        return s;
    }
    // `\{\{(.*?)}}`
    let mut out = String::with_capacity(s.len());
    let mut rest = s.as_str();
    while let Some(i) = rest.find("{{") {
        let after = &rest[i + 2..];
        let Some(j) = after.find("}}") else { break };
        out.push_str(&rest[..i]);
        out.push_str(params.get(&after[..j]).map(String::as_str).unwrap_or(""));
        rest = &after[j + 2..];
    }
    out.push_str(rest);
    out
}

impl FieldMapping {
    pub fn new(ty: EsType) -> Self {
        Self { ty, params: Map::new(), fields: BTreeMap::new() }
    }

    pub fn param(&self, k: &str) -> Option<&Value> {
        self.params.get(k)
    }

    pub fn bool_param(&self, k: &str, default: bool) -> bool {
        match self.params.get(k) {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => s == "true",
            // `doc_values: {multi_value, nullability}`: enabled.
            Some(Value::Object(_)) if k == "doc_values" => true,
            _ => default,
        }
    }

    pub fn str_param(&self, k: &str) -> Option<&str> {
        self.params.get(k).and_then(Value::as_str)
    }

    fn parse(name: &str, spec: &Map<String, Value>, ty: EsType, registry: &AnalysisRegistry, multi: bool) -> Result<Self> {
        use crate::fieldtypes::params;
        let mut fm = FieldMapping::new(ty);
        let leaf = name.rsplit('.').next().unwrap_or(name);
        // `FieldMapper.Builder.storeParam`: no stored fields in strict
        // columnar index modes.
        if registry.strict_columnar() && spec.get("store").is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true")) {
            return Err(params::iae(format!("[store] cannot be enabled on field [{leaf}] in [{}] index mode", registry.index_mode_name())));
        }
        for (k, v) in spec {
            if k == "type" {
                continue;
            }
            if k == "fields" {
                if matches!(ty, EsType::Flattened) {
                    return Err(params::iae(format!("flattened field [{name}] does not support [fields]")));
                }
                let obj = v.as_object().ok_or_else(|| {
                    params::mpe(format!("expected map for property [fields] on field [{}] or [fields] but got a class {}", params::jstr(v), java_class_of(v)))
                })?;
                for (sub, sspec) in obj {
                    let sspec = sspec.as_object().ok_or_else(|| params::mpe(format!("Expected map for property [fields] on field [{sub}] but got a class {}", java_class_of(&obj[sub]))))?;
                    if multi {
                        return Err(params::iae(format!("Encountered a multi-field [{leaf}] which itself contains a multi-field. Defining chained multi-fields is not supported.")));
                    }
                    let st = match sspec.get("type") {
                        None | Some(Value::Null) => return Err(params::mpe(format!("No type specified for field [{sub}]"))),
                        Some(Value::String(t)) => t.clone(),
                        Some(other) => params::jstr(other),
                    };
                    let sty = EsType::parse(&st).ok_or_else(|| params::mpe(format!("The mapper type [{st}] declared on field [{sub}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.")))?;
                    if matches!(sty, EsType::Object | EsType::Nested) {
                        return Err(params::mpe(format!("Type [{st}] cannot be used in multi field")));
                    }
                    fm.fields.insert(sub.clone(), FieldMapping::parse(&format!("{name}.{sub}"), sspec, sty, registry, true)?);
                }
                continue;
            }
            if k == "boost" {
                return Err(params::mpe(format!("Unknown parameter [boost] on mapper [{leaf}]")));
            }
            // `flattened` mapped sub-fields (`properties`, keyed by dotted
            // name) and `passthrough: {priority}` (9.5).
            if ty == EsType::Flattened && k == "properties" {
                let o = v.as_object().ok_or_else(|| params::mpe(format!("[properties] on flattened field [{leaf}] must be an object")))?;
                let mut out = Map::new();
                for (sub, sspec) in o {
                    let sspec = sspec.as_object().ok_or_else(|| params::mpe(format!("Expected map for property [{sub}] on field [{leaf}]")))?;
                    let st = sspec.get("type").and_then(Value::as_str).ok_or_else(|| params::mpe(format!("No type specified for field [{sub}]")))?;
                    let sty = EsType::parse(st).ok_or_else(|| params::mpe(format!("The mapper type [{st}] declared on field [{sub}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.")))?;
                    if matches!(sty, EsType::Object | EsType::Nested | EsType::Flattened) {
                        return Err(params::mpe(format!("Type [{st}] cannot be used as a mapped sub-field of flattened field [{leaf}]")));
                    }
                    let sf = FieldMapping::parse(&format!("{name}.{sub}"), sspec, sty, registry, false)?;
                    out.insert(sub.clone(), sf.to_json());
                }
                fm.params.insert(k.clone(), Value::Object(out));
                continue;
            }
            if ty == EsType::Flattened && k == "passthrough" {
                let p = v.get("priority").and_then(Value::as_i64).filter(|p| *p >= 0).ok_or_else(|| params::mpe(format!("[passthrough] on flattened field [{leaf}] requires a non-negative [priority]")))?;
                fm.params.insert(k.clone(), json!({"priority": p}));
                continue;
            }
            if !ty.allowed_params().contains(&k.as_str()) && k != "synthetic_source_keep" && !(k == "copy_to" && ty != EsType::Alias) && !(k == "meta" && ty != EsType::Alias) {
                if ty == EsType::Alias {
                    return Err(params::mpe(format!("Mapping definition for [{leaf}] has unsupported parameters:  [{k} : {}]", params::jstr(v))));
                }
                return Err(params::mpe(format!("unknown parameter [{k}] on mapper [{leaf}] of type [{}]", ty.name())));
            }
            // `DocValuesParameter`: the object form (`multi_value`,
            // `nullability`) only exists in strict columnar index modes.
            if k == "doc_values" && v.is_object() {
                if !registry.strict_columnar() {
                    return Err(EsError::mapper_parsing(format!("unsupported doc_values configuration for field [{leaf}] in non-columnar index; supported values: [true, false]")));
                }
                let o = v.as_object().expect("object");
                let mut dv = Map::new();
                for (sk, default) in [("multi_value", registry.dv_multi_value), ("nullability", registry.dv_nullability)] {
                    let b = match o.get(sk) {
                        Some(x) => params::object_bool(&format!("{sk}"), x)?,
                        None => default,
                    };
                    dv.insert(sk.into(), json!(b));
                }
                fm.params.insert(k.clone(), Value::Object(dv));
                continue;
            }
            if let Some(nv) = params::normalize(ty, k, v)? {
                fm.params.insert(k.clone(), nv);
            }
        }
        if ty == EsType::DenseVector {
            crate::vectors::normalize_mapping(name, &mut fm.params, crate::vectors::VecMappingCtx::of(registry))?;
        }
        if ty.is_semantic() {
            crate::inference::semantic_map::normalize_mapping(name, ty, &mut fm.params)?;
        }
        fm.validate(name, registry)?;
        let explicit_index = fm.params.get("index").cloned();
        let explicit_store = fm.params.get("store").cloned();
        params::finish(name, ty, &mut fm.params)?;
        // The generic parameter canonicalizer assumes store:false is the
        // default. Sparse vectors default to true when source vectors are
        // excluded, so their explicit false must survive normalization.
        if ty == EsType::SparseVector && registry.exclude_source_vectors {
            if let Some(v @ Value::Bool(false)) = explicit_store {
                fm.params.insert("store".into(), v);
            }
        }
        // `index.mapping.index_disabled_by_default`: an explicit `index: true`
        // is then not the default and stays in the mapping.
        if registry.index_disabled_by_default {
            if let Some(v @ Value::Bool(true)) = explicit_index {
                fm.params.insert("index".into(), v);
            }
        }
        Ok(fm)
    }

    fn validate(&self, name: &str, registry: &AnalysisRegistry) -> Result<()> {
        for key in ["analyzer", "search_analyzer", "search_quote_analyzer"] {
            if let Some(a) = self.str_param(key) {
                if !registry.has_analyzer(a) {
                    return Err(crate::fieldtypes::params::iae(format!("analyzer [{a}] has not been configured in mappings")));
                }
            }
        }
        if let Some(n) = self.str_param("normalizer") {
            if registry.normalizer(n).is_none() {
                return Err(EsError::mapper_parsing(format!("normalizer [{n}] not found for field [{name}]")));
            }
        }
        if let Some(f) = self.str_param("format") {
            DateFormatter::new(f).map_err(|e| crate::fieldtypes::params::iae_caused(format!("Error parsing [format] on field [{name}]: {}", e.reason), e))?;
        }
        if let Some(l) = self.str_param("locale") {
            crate::fieldtypes::params::check_locale(l)?;
        }
        if self.ty == EsType::DenseVector {
            if let Some(d) = self.param("dims") {
                let d = d.as_u64().ok_or_else(|| EsError::mapper_parsing("[dims] must be an integer"))?;
                // `bit` allows up to MAX_DIMS_COUNT_BIT (its own range and
                // multiple-of-8 checks run in `vectors`).
                let max = crate::vectors::max_dims(self.str_param("element_type"));
                if d == 0 || d > max {
                    let min = if max == crate::vectors::MAX_DIMS_COUNT_BIT { 8 } else { 1 };
                    return Err(EsError::mapper_parsing(format!("The number of dimensions should be in the range [{min}, {max}] but was [{d}]")));
                }
            }
            if let Some(s) = self.str_param("similarity") {
                if !matches!(s, "l2_norm" | "dot_product" | "cosine" | "max_inner_product") {
                    return Err(EsError::mapper_parsing(format!("Unknown value [{s}] for field [similarity] - accepted values are [l2_norm, cosine, dot_product, max_inner_product]")));
                }
            }
        }
        if self.ty == EsType::Alias && self.str_param("path").is_none() {
            return Err(EsError::mapper_parsing(format!("The [path] property must be specified for field [{name}].")));
        }
        if let Some(ia) = self.param("ignore_above") {
            if ia.as_i64().map(|v| v < 0).unwrap_or(true) {
                return Err(crate::fieldtypes::params::iae(format!("[ignore_above] must be positive, got [{ia}]")));
            }
        }
        if let Some(io) = self.str_param("index_options") {
            let allowed: Option<&[&str]> = match self.ty {
                EsType::Keyword | EsType::Flattened => Some(&["docs", "freqs"]),
                EsType::Text | EsType::SearchAsYouType | EsType::AnnotatedText => Some(&["positions", "docs", "freqs", "offsets"]),
                _ => None,
            };
            if let Some(a) = allowed {
                if !a.contains(&io) {
                    return Err(EsError::mapper_parsing(format!("Unknown value [{io}] for field [index_options] - accepted values are [{}]", a.join(", "))));
                }
            }
        }
        if let Some(sim) = self.str_param("similarity") {
            if self.ty != EsType::DenseVector && !registry.sims.knows(sim) && !self.ty.is_semantic() {
                return Err(EsError::mapper_parsing(format!("Unknown Similarity type [{sim}] for field [{name}]")));
            }
        }
        crate::fieldtypes::validate(name, self, registry)?;
        Ok(())
    }

    /// A `flattened` field's mapped sub-fields (dotted name, mapping).
    pub fn flattened_mapped(&self) -> Vec<(String, FieldMapping)> {
        let Some(o) = self.params.get("properties").and_then(Value::as_object).filter(|_| self.ty == EsType::Flattened) else { return Vec::new() };
        let reg = AnalysisRegistry::default();
        o.iter()
            .filter_map(|(k, v)| {
                let spec = v.as_object()?;
                let ty = EsType::parse(spec.get("type")?.as_str()?)?;
                FieldMapping::parse(k, spec, ty, &reg, false).ok().map(|f| (k.clone(), f))
            })
            .collect()
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("type".into(), Value::String(self.ty.name().into()));
        for (k, v) in &self.params {
            m.insert(k.clone(), v.clone());
        }
        if !self.fields.is_empty() {
            m.insert("fields".into(), Value::Object(self.fields.iter().map(|(k, f)| (k.clone(), f.to_json())).collect()));
        }
        Value::Object(m)
    }

    fn merge(&mut self, name: &str, new: &FieldMapping) -> Result<()> {
        use crate::fieldtypes::params as pm;
        if self.ty != new.ty {
            if self.ty == EsType::Alias {
                return Err(EsError::illegal_argument(format!("Cannot merge a field alias mapping [{name}] with a mapping that is not for a field alias.")));
            }
            return Err(EsError::illegal_argument(format!("mapper [{name}] cannot be changed from type [{}] to [{}]", self.ty.name(), new.ty.name())));
        }
        if self.ty.is_semantic() {
            self.params = crate::inference::semantic_map::merge(name, &self.params, &new.params)?;
            for (sub, sf) in &new.fields {
                match self.fields.get_mut(sub) {
                    Some(existing) => existing.merge(&format!("{name}.{sub}"), sf)?,
                    None => {
                        self.fields.insert(sub.clone(), sf.clone());
                    }
                }
            }
            return Ok(());
        }
        // dense_vector: both sides are compared with every default materialized,
        // so an identical re-put is a no-op.
        let (old_dv, new_dv);
        let new = if self.ty == EsType::DenseVector {
            let mut o = self.params.clone();
            crate::vectors::normalize_mapping(name, &mut o, crate::vectors::VecMappingCtx::quiet(crate::version::IndexVersion::CURRENT))?;
            let mut n = new.clone();
            crate::vectors::normalize_mapping(name, &mut n.params, crate::vectors::VecMappingCtx::quiet(crate::version::IndexVersion::CURRENT))?;
            old_dv = o;
            new_dv = n;
            &new_dv
        } else {
            old_dv = Map::new();
            new_dv = new.clone();
            new
        };
        let strip_default = |p: &Map<String, Value>| {
            let mut p = p.clone();
            for k in ["analyzer", "search_analyzer"] {
                if p.get(k) == Some(&json!("default")) {
                    p.remove(k);
                }
            }
            p
        };
        let (self_p, new_p) = if self.ty == EsType::DenseVector { (old_dv.clone(), new.params.clone()) } else { (strip_default(&self.params), strip_default(&new.params)) };
        let mut conflicts = Vec::new();
        let mut keys: Vec<&String> = self_p.keys().chain(new_p.keys()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let order = pm::param_order(self.ty);
        keys.sort_by_key(|k| order.iter().position(|o| o == k).unwrap_or(usize::MAX));
        let mut updates = Vec::new();
        for k in keys {
            let old = self_p.get(k);
            let nv = new_p.get(k);
            if old == nv {
                continue;
            }
            if self.ty == EsType::DenseVector {
                // dense_vector dims may only be set once (an update may not drop or change them).
                let dims_ok = k != "dims" || old.is_none_or(Value::is_null);
                if (EsType::updatable(k) && dims_ok) || (k == "index_options" && crate::vectors::index_options_updatable(old, nv)) {
                    updates.push((k.clone(), nv.cloned()));
                    continue;
                }
            } else if pm::updatable(self.ty, k, old, nv) {
                updates.push((k.clone(), nv.cloned()));
                continue;
            }
            if self.ty == EsType::Join && k == "relations" {
                conflicts.extend(pm::join_conflicts(old, nv));
            }
            conflicts.push(format!("Cannot update parameter [{k}] from [{}] to [{}]", pm::display(self.ty, k, old), pm::display(self.ty, k, nv)));
        }
        if !conflicts.is_empty() {
            return Err(EsError::illegal_argument(format!("Mapper for [{name}] conflicts with existing mapper:\n\t{}", conflicts.join("\n\t"))));
        }
        if self.ty == EsType::DenseVector {
            // Re-store in the canonical (parse-time) form.
            for (k, v) in updates.iter() {
                if let Some(v) = v {
                    old_dv_apply(&mut self.params, k, v);
                }
            }
            updates.clear();
            crate::fieldtypes::params::finish(name, self.ty, &mut self.params)?;
        }
        for (k, v) in updates {
            match v {
                Some(v) => {
                    self.params.insert(k, v);
                }
                // An updatable parameter missing from the update resets to its default.
                None if self.ty != EsType::DenseVector => {
                    self.params.remove(&k);
                }
                None => {}
            }
        }
        crate::fieldtypes::params::derive_analyzers(self.ty, &mut self.params);
        for (sub, sf) in &new.fields {
            match self.fields.get_mut(sub) {
                Some(existing) => existing.merge(&format!("{name}.{sub}"), sf)?,
                None => {
                    self.fields.insert(sub.clone(), sf.clone());
                }
            }
        }
        Ok(())
    }
}

impl ObjectMapping {
    fn parse(path: &str, spec: &Map<String, Value>, nested: bool, registry: &AnalysisRegistry, depth: usize) -> Result<Self> {
        use crate::fieldtypes::params;
        if depth > 20 {
            return Err(EsError::illegal_argument("Limit of mapping depth [20] has been exceeded due to object field"));
        }
        let mut om = ObjectMapping { nested, explicit_type: spec.contains_key("type"), ..Default::default() };
        let leaf = path.rsplit('.').next().unwrap_or(path);
        let mut unsupported = Vec::new();
        // Object parsing context cannot depend on JSON member order. A delta
        // often writes properties before its committed subobjects setting.
        for (k, v) in spec.iter().filter(|(k, _)| k.as_str() != "properties")
            .chain(spec.iter().filter(|(k, _)| k.as_str() == "properties")) {
            match k.as_str() {
                "type" => {}
                "subobjects" if registry.strict_columnar() => {
                    if v.as_bool() == Some(true) || v.as_str() == Some("true") {
                        return Err(EsError::mapper_parsing("subobjects [true] is not supported in columnar mode"));
                    }
                }
                "subobjects" => {
                    match v {
                        Value::Bool(_) => {}
                        Value::String(t) if t == "true" || t == "false" => {}
                        other => {
                            let m = format!("unknown subobjects value: {}", params::jstr(other));
                            return Err(EsError::mapper_parsing(m.clone()).caused_by(EsError::new(400, "parse_exception", m)));
                        }
                    }
                    om.params.insert("subobjects".into(), if v == &json!("true") { json!(true) } else if v == &json!("false") { json!(false) } else { v.clone() });
                }
                "properties" => {
                    let props = match v {
                        Value::Object(p) => p,
                        Value::Array(a) if a.is_empty() => continue,
                        _ => {
                            let m = "properties must be a map type";
                            return Err(EsError::mapper_parsing(m).caused_by(EsError::new(400, "parse_exception", m)));
                        }
                    };
                    for (name, fspec) in props {
                        om.insert_property(path, name, fspec, registry, depth)?;
                    }
                }
                "dynamic" => {
                    let d = Dynamic::parse(v)?;
                    if d == Dynamic::Runtime && registry.strict_columnar() {
                        return Err(EsError::mapper_parsing("dynamic [runtime] is not supported in strict columnar mode"));
                    }
                    om.dynamic = Some(d);
                }
                "enabled" => om.enabled = Some(params::object_bool("enabled.enabled", v)?),
                "include_in_parent" | "include_in_root" if nested => {
                    let b = params::object_bool(&format!("{path}.{k}"), v)?;
                    om.params.insert(k.clone(), json!(b));
                }
                "synthetic_source_keep" => {
                    match v.as_str() {
                        Some("none" | "arrays" | "all") => {}
                        _ => {
                            return Err(EsError::mapper_parsing(format!(
                                "Unknown synthetic_source_keep value [{}], accepted values are [none,arrays,all]",
                                v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())
                            )))
                        }
                    }
                    om.params.insert(k.clone(), v.clone());
                }
                _ => unsupported.push(format!("{k} : {}", params::jstr(v))),
            }
        }
        if !unsupported.is_empty() {
            return Err(EsError::mapper_parsing(format!("Mapping definition for [{leaf}] has unsupported parameters:  [{}]", unsupported.join("] ["))));
        }
        // `include_in_root` is moot (dropped) when `include_in_parent` is set too.
        if om.params.get("include_in_parent") == Some(&json!(true)) && om.params.get("include_in_root") == Some(&json!(true)) {
            om.params.remove("include_in_root");
        }
        Ok(om)
    }

    /// `subobjects: false` (dotted names stay leaf names).
    pub fn no_subobjects(&self) -> bool {
        self.columnar || matches!(self.params.get("subobjects"), Some(Value::Bool(false))) || self.params.get("subobjects").and_then(Value::as_str) == Some("false") || self.is_passthrough()
    }

    /// A `passthrough` object.
    pub fn is_passthrough(&self) -> bool {
        self.params.get("type").and_then(Value::as_str) == Some("passthrough")
    }

    fn insert_property(&mut self, parent: &str, name: &str, fspec: &Value, registry: &AnalysisRegistry, depth: usize) -> Result<()> {
        if name.is_empty() {
            return Err(crate::fieldtypes::params::iae("field name cannot be an empty string"));
        }
        if name.trim().is_empty() {
            return Err(crate::fieldtypes::params::iae("field name cannot contain only whitespaces"));
        }
        if name.chars().all(|c| c == '.') {
            return Err(crate::fieldtypes::params::iae("field name cannot contain only dots"));
        }
        if name.starts_with('.') || name.contains("..") {
            return Err(crate::fieldtypes::params::iae("field name cannot be an empty string"));
        }
        // A trailing dot is dropped (`a.` is the field `a`).
        let name = name.strip_suffix('.').unwrap_or(name);
        // Dotted names expand into objects (unless `subobjects: false`).
        if let (Some((head, rest)), false) = (name.split_once('.'), self.no_subobjects()) {
            if !head.is_empty() && !rest.is_empty() {
                let full = if parent.is_empty() { head.to_string() } else { format!("{parent}.{head}") };
                let entry = self.properties.entry(head.to_string()).or_insert_with(|| Mapper::Object(ObjectMapping::default()));
                return match entry {
                    Mapper::Object(o) => o.insert_property(&full, rest, fspec, registry, depth + 1),
                    Mapper::Field(_) => Err(EsError::illegal_argument(format!("can't merge a non object mapping [{full}] with an object mapping"))),
                };
            }
        }
        let full = if parent.is_empty() { name.to_string() } else { format!("{parent}.{name}") };
        let spec = fspec.as_object().ok_or_else(|| EsError::mapper_parsing(format!("Expected map for property [fields] on field [{name}] but got a class {}", java_class_of(fspec))))?;
        if let Some(t) = spec.get("type") {
            if !t.is_string() && !t.is_null() {
                return Err(EsError::mapper_parsing(format!("The mapper type [{}] declared on field [{name}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.", crate::fieldtypes::params::jstr(t))));
            }
        }
        let tname = spec.get("type").and_then(Value::as_str);
        let mapper = match tname {
            None | Some("object") => {
                if (tname.is_none() && spec.get("type").is_some()) || (tname.is_none() && !spec.contains_key("properties") && spec.keys().any(|k| !matches!(k.as_str(), "dynamic" | "enabled" | "subobjects")) && !spec.is_empty()) {
                    return Err(EsError::mapper_parsing(format!("No type specified for field [{name}]")));
                }
                Mapper::Object(ObjectMapping::parse(&full, spec, false, registry, depth + 1)?)
            }
            Some("nested") => {
                let om = ObjectMapping::parse(&full, spec, true, registry, depth + 1)?;
                // Columnar index modes support a single level of nesting.
                if registry.strict_columnar() {
                    fn inner_nested(o: &ObjectMapping, path: &str) -> Option<String> {
                        o.properties.iter().find_map(|(k, m)| match m {
                            Mapper::Object(c) if c.nested => Some(format!("{path}.{k}")),
                            Mapper::Object(c) => inner_nested(c, &format!("{path}.{k}")),
                            _ => None,
                        })
                    }
                    if let Some(n) = inner_nested(&om, &full) {
                        return Err(EsError::mapper_parsing(format!("nested object [{n}] is nested inside another nested object, but columnar index modes support only a single level of nesting")));
                    }
                }
                Mapper::Object(om)
            }
            Some("passthrough") => {
                let mut inner = spec.clone();
                let mut extra = Map::new();
                for k in ["type", "priority", "time_series_dimension"] {
                    if let Some(v) = inner.remove(k) {
                        extra.insert(k.to_string(), v);
                    }
                }
                if let Some(v) = inner.get("subobjects") {
                    let shown = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                    return Err(EsError::mapper_parsing(format!("Mapping definition for [{full}] has unsupported parameters:  [subobjects : {shown}]")));
                }
                match extra.get("priority") {
                    Some(p) => {
                        if p.as_i64().map(|x| x < 0).unwrap_or(true) {
                            return Err(EsError::mapper_parsing(format!("[priority] must be a non-negative integer for passthrough object [{full}]")));
                        }
                    }
                    None => {
                        let m = format!("Pass-through object [{full}] is missing a non-negative value for parameter [priority]");
                        return Err(EsError::mapper_parsing(m.clone()).caused_by(EsError::new(500, "mapper_exception", m)));
                    }
                }
                // A pass-through object never has sub-objects: its object
                // children are flattened into dotted leaf names.
                let mut spec2 = Map::new();
                spec2.insert("subobjects".into(), json!(false));
                spec2.extend(inner);
                let mut om = ObjectMapping::parse(&full, &spec2, false, registry, depth + 1)?;
                om.params.remove("subobjects");
                om.params.extend(extra);
                Mapper::Object(om)
            }
            Some(t) => {
                let ty = EsType::parse(t).ok_or_else(|| EsError::mapper_parsing(format!("The mapper type [{t}] declared on field [{name}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.")))?;
                if ty == EsType::Object {
                    Mapper::Object(ObjectMapping::parse(&full, spec, false, registry, depth + 1)?)
                } else {
                    Mapper::Field(FieldMapping::parse(&full, spec, ty, registry, false)?)
                }
            }
        };
        // A columnar root flattens every object (`asFlattenedFieldBuilders`):
        // leaves get dotted names, nested objects keep their full path, and
        // the objects' `dynamic` / pass-through priority / `enabled: false`
        // go to `prefix_properties` (a disabled object drops its subtree).
        if let (true, Mapper::Object(o)) = (self.columnar, &mapper) {
            if !o.nested {
                fn walk(path: &str, o: &ObjectMapping, out: &mut Vec<(String, Mapper)>, prefix: &mut BTreeMap<String, PrefixProps>) {
                    let mut pp = PrefixProps { dynamic: o.dynamic, ..Default::default() };
                    if o.is_passthrough() {
                        pp.passthrough = o.params.get("priority").and_then(Value::as_i64);
                    }
                    if o.enabled == Some(false) {
                        pp.enabled = Some(false);
                    }
                    if pp != PrefixProps::default() {
                        prefix.entry(path.to_string()).or_default().merge(&pp);
                    }
                    if o.enabled == Some(false) {
                        return;
                    }
                    for (k, m) in &o.properties {
                        let n = format!("{path}.{k}");
                        match m {
                            Mapper::Object(c) if !c.nested => walk(&n, c, out, prefix),
                            other => out.push((n, other.clone())),
                        }
                    }
                }
                let mut leaves = Vec::new();
                let mut prefix = std::mem::take(&mut self.prefix);
                walk(name, o, &mut leaves, &mut prefix);
                self.prefix = prefix;
                for (n, m) in leaves {
                    self.properties.insert(n, m);
                }
                return Ok(());
            }
        }
        // Under `subobjects: false` (and in a pass-through object) an
        // explicit non-nested object is flattened: its leaves get dotted names.
        if let (true, Mapper::Object(o)) = (self.no_subobjects(), &mapper) {
            if !o.nested && !o.is_passthrough() && !o.properties.is_empty() && o.enabled != Some(false) {
                fn flatten(prefix: &str, o: &ObjectMapping, out: &mut Vec<(String, Mapper)>) {
                    for (k, m) in &o.properties {
                        let n = format!("{prefix}.{k}");
                        match m {
                            Mapper::Object(c) if !c.nested && !c.is_passthrough() && !c.properties.is_empty() => flatten(&n, c, out),
                            other => out.push((n, other.clone())),
                        }
                    }
                }
                let mut leaves = Vec::new();
                flatten(name, o, &mut leaves);
                for (n, m) in leaves {
                    self.properties.insert(n, m);
                }
                return Ok(());
            }
        }
        self.properties.insert(name.to_string(), mapper);
        Ok(())
    }

    fn to_json(&self) -> Value {
        let mut m = Map::new();
        if self.nested {
            m.insert("type".into(), json!("nested"));
        } else if self.properties.is_empty() {
            // ES prints `type: object` only for objects without properties.
            m.insert("type".into(), json!("object"));
        }
        if let Some(d) = self.dynamic {
            m.insert("dynamic".into(), d.to_json());
        }
        if let Some(e) = self.enabled.filter(|e| !e) {
            m.insert("enabled".into(), json!(e));
        }
        for (k, v) in &self.params {
            m.insert(k.clone(), v.clone());
        }
        if !self.properties.is_empty() {
            m.insert("properties".into(), props_json(&self.properties));
        }
        Value::Object(m)
    }

    fn merge(&mut self, path: &str, new: &ObjectMapping) -> Result<()> {
        if self.nested != new.nested {
            return Err(EsError::illegal_argument(format!("can't merge a non-nested mapping [{path}] with a nested mapping")));
        }
        let mapper_exc = |what: &str| EsError::new(500, "mapper_exception", format!("the [{what}] parameter can't be updated for the object mapping [{path}]"));
        if let Some(d) = new.dynamic {
            self.dynamic = Some(d);
        }
        if let Some(e) = new.enabled {
            if self.enabled.unwrap_or(true) != e {
                return Err(mapper_exc("enabled"));
            }
        }
        if let Some(so) = new.params.get("subobjects") {
            let old_false = self.no_subobjects();
            let new_false = so == &json!(false);
            if old_false != new_false {
                return Err(mapper_exc("subobjects"));
            }
        }
        if self.nested {
            for k in ["include_in_parent", "include_in_root"] {
                if let Some(v) = new.params.get(k) {
                    if self.params.get(k).unwrap_or(&json!(false)) != v {
                        return Err(EsError::new(500, "mapper_exception", format!("the [{k}] parameter can't be updated on a nested object mapping")));
                    }
                }
            }
        }
        for (k, pp) in &new.prefix {
            self.prefix.entry(k.clone()).or_default().merge(pp);
        }
        // Under `subobjects: false` a dotted name of the update (parsed as nested
        // objects) is a flat leaf name again.
        let flat;
        let incoming: &BTreeMap<String, Mapper> = if self.no_subobjects() {
            fn flatten(props: &BTreeMap<String, Mapper>, prefix: &str, out: &mut BTreeMap<String, Mapper>) {
                for (k, m) in props {
                    let name = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    match m {
                        Mapper::Object(o) if !o.explicit_type && !o.nested && o.dynamic.is_none() && o.enabled.is_none() && o.params.is_empty() => flatten(&o.properties, &name, out),
                        other => {
                            out.insert(name, other.clone());
                        }
                    }
                }
            }
            let mut f = BTreeMap::new();
            flatten(&new.properties, "", &mut f);
            flat = f;
            &flat
        } else {
            &new.properties
        };
        for (name, m) in incoming {
            let full = if path.is_empty() { name.clone() } else { format!("{path}.{name}") };
            match (self.properties.get_mut(name), m) {
                (None, _) => {
                    self.properties.insert(name.clone(), m.clone());
                }
                (Some(Mapper::Field(a)), Mapper::Field(b)) => a.merge(&full, b)?,
                (Some(Mapper::Object(a)), Mapper::Object(b)) => a.merge(&full, b)?,
                (Some(Mapper::Field(_)), Mapper::Object(_)) => {
                    return Err(EsError::illegal_argument(format!("can't merge a non object mapping [{full}] with an object mapping")));
                }
                (Some(Mapper::Object(_)), Mapper::Field(_)) => {
                    return Err(EsError::illegal_argument(format!("can't merge a non object mapping [{full}] with an object mapping")));
                }
            }
        }
        Ok(())
    }
}

fn old_dv_apply(p: &mut Map<String, Value>, k: &str, v: &Value) {
    p.insert(k.to_string(), v.clone());
}

fn props_json(props: &BTreeMap<String, Mapper>) -> Value {
    Value::Object(
        props
            .iter()
            .map(|(k, m)| {
                (
                    k.clone(),
                    match m {
                        Mapper::Field(f) => f.to_json(),
                        Mapper::Object(o) => o.to_json(),
                    },
                )
            })
            .collect(),
    )
}

impl Mappings {
    /// Parse a mapping body (`{"properties": ..., "dynamic": ...}`).
    pub fn parse(v: &Value, registry: &AnalysisRegistry) -> Result<Mappings> {
        let m = Self::parse_update(v, registry)?;
        // Cross-field checks are not wrapped in `Failed to parse mapping`.
        m.validate_passthrough()?;
        m.validate_aliases()?;
        m.validate_copy_to()?;
        Ok(m)
    }

    /// Parse a mapping *update* body: the cross-field checks (aliases,
    /// `copy_to`) run on the merged result instead.
    pub fn parse_update(v: &Value, registry: &AnalysisRegistry) -> Result<Mappings> {
        let top = v.get("_doc").filter(|_| v.as_object().is_some_and(|o| o.len() == 1)).unwrap_or(v);
        if let Some(props) = top.get("properties").and_then(Value::as_object) {
            for name in props.keys() {
                if matches!(name.as_str(), "_doc_count" | "_ignored" | "_id" | "_source" | "_index" | "_routing" | "_seq_no" | "_version" | "_field_names" | "_nested_path" | "_feature" | "_ignored_source" | "_index_mode") {
                    return Err(EsError::mapper_parsing(format!("Field [{name}] is defined more than once")));
                }
            }
        }
        Self::parse_inner(v, registry).map_err(wrap_parse_err)
    }

    fn parse_inner(v: &Value, registry: &AnalysisRegistry) -> Result<Mappings> {
        let mut m = Mappings::default();
        if v.is_null() {
            return Ok(m);
        }
        let obj = v.as_object().ok_or_else(|| EsError::mapper_parsing("Root mapping definition must be an object"))?;
        // Typed mappings (`{"_doc": {...}}`) are a 7.x leftover; unwrap `_doc`.
        if obj.len() == 1 {
            if let Some(inner) = obj.get("_doc") {
                if inner.is_object() {
                    return Self::parse_inner(inner, registry);
                }
            }
        }
        use crate::fieldtypes::params::{iae, iae_caused, jstr, mpe, object_bool};
        let mut unsupported = Vec::new();
        m.root.columnar = registry.strict_columnar();
        // Columnar roots never have sub-objects: `subobjects: false` is a
        // no-op, `true` is rejected.
        if m.root.columnar {
            if let Some(v) = obj.get("subobjects") {
                if v.as_bool() == Some(true) || v.as_str() == Some("true") {
                    return Err(EsError::mapper_parsing("subobjects [true] is not supported in columnar mode"));
                }
            }
        }
        if let Some(v) = obj.get("subobjects").filter(|_| !m.root.columnar) {
            match v {
                Value::Bool(_) => m.root.params.insert("subobjects".into(), v.clone()),
                Value::String(t) if t == "true" || t == "false" => m.root.params.insert("subobjects".into(), json!(t == "true")),
                other => {
                    let msg = format!("unknown subobjects value: {}", jstr(other));
                    return Err(EsError::mapper_parsing(msg.clone()).caused_by(EsError::new(400, "parse_exception", msg)));
                }
            };
        }
        let java_cast = |what: &str| {
            let msg = format!("class java.lang.String cannot be cast to class java.util.{what} (java.lang.String and java.util.{what} are in module java.base of loader 'bootstrap')");
            EsError::mapper_parsing(msg.clone()).caused_by(EsError::new(400, "class_cast_exception", msg))
        };
        for (k, val) in obj {
            match k.as_str() {
                "_size" => {
                    let enabled = val.get("enabled").ok_or_else(|| EsError::mapper_parsing(format!("Root mapping definition has unsupported parameters:  [_size : {}]", jstr(val))))?;
                    if !enabled.is_boolean() {
                        return Err(EsError::mapper_parsing(format!("Failed to parse mapping: Cannot parse value [{enabled}] as only [true] or [false] are allowed.")));
                    }
                    m.size = Some(val.clone());
                }
                "properties" => {
                    let props = match val {
                        Value::Object(p) => p,
                        Value::Array(a) if a.is_empty() => continue,
                        _ => {
                            let msg = "properties must be a map type";
                            return Err(EsError::mapper_parsing(msg).caused_by(EsError::new(400, "parse_exception", msg)));
                        }
                    };
                    for (name, fspec) in props {
                        m.root.insert_property("", name, fspec, registry, 0)?;
                    }
                }
                "dynamic" => {
                    let d = Dynamic::parse(val)?;
                    if d == Dynamic::Runtime && registry.strict_columnar() {
                        return Err(EsError::mapper_parsing("dynamic [runtime] is not supported in strict columnar mode"));
                    }
                    m.root.dynamic = Some(d);
                }
                "enabled" => {
                    let e = object_bool("enabled.enabled", val)?;
                    if !e && registry.strict_columnar() {
                        return Err(EsError::mapper_parsing("[enabled] cannot be set to [false] on the root object in strict columnar index modes; this would prevent all fields from being indexed"));
                    }
                    m.root.enabled = Some(e);
                }
                "prefix_properties" if registry.strict_columnar() => {
                    let o = val.as_object().ok_or_else(|| EsError::mapper_parsing("[prefix_properties] must be an object"))?;
                    for (prefix, pv) in o {
                        let pp = PrefixProps::parse(prefix, pv)?;
                        m.root.prefix.entry(prefix.clone()).or_default().merge(&pp);
                    }
                }
                "dynamic_templates" => {
                    m.dt_explicit = true;
                    let arr = val.as_array().ok_or_else(|| mpe("Dynamic template syntax error. An array of named objects is expected."))?;
                    for t in arr {
                        let to = t.as_object().ok_or_else(|| mpe("Dynamic template syntax error. An array of named objects is expected."))?;
                        if to.len() != 1 {
                            return Err(mpe("A dynamic template must be defined with a name"));
                        }
                        let (name, def) = to.iter().next().unwrap();
                        m.dynamic_templates.push(DynamicTemplate::parse(name, def, registry)?);
                    }
                }
                "date_detection" | "numeric_detection" => {
                    let b = match val {
                        Value::Bool(b) => *b,
                        Value::String(t) if t == "true" || t == "false" => t == "true",
                        other => {
                            let inner = EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", jstr(other)));
                            return Err(iae_caused(format!("Could not convert [{k}] to boolean"), inner));
                        }
                    };
                    if k == "date_detection" {
                        m.date_detection = Some(b);
                    } else {
                        m.numeric_detection = Some(b);
                    }
                }
                "dynamic_date_formats" => {
                    let fmts = str_or_list(Some(val));
                    for f in &fmts {
                        if f.starts_with("epoch_") {
                            return Err(mpe(format!("Epoch [{f}] is not supported as dynamic date format")));
                        }
                        DateFormatter::new(f).map_err(|e| {
                            let inner = e.caused_by.as_deref().cloned().unwrap_or_else(|| EsError::illegal_argument(e.reason.clone()));
                            iae_caused(e.reason.clone(), inner)
                        })?;
                    }
                    m.dynamic_date_formats = Some(fmts);
                }
                "_source" => {
                    let o = val.as_object().ok_or_else(|| mpe("_source must be an object"))?;
                    for (ik, iv) in o {
                        match ik.as_str() {
                            "enabled" => {
                                crate::fieldtypes::params::to_bool(iv)?;
                            }
                            "includes" | "excludes" => {
                                if !iv.is_array() {
                                    return Err(java_cast("List"));
                                }
                            }
                            "mode" => {
                                if !matches!(iv.as_str(), Some("stored" | "synthetic" | "disabled")) {
                                    let u = iv.as_str().map(str::to_ascii_uppercase).unwrap_or_default();
                                    return Err(iae(format!("No enum constant org.elasticsearch.index.mapper.SourceFieldMapper.Mode.{u}")));
                                }
                            }
                            "synthetic" | "mode_legacy" => {}
                            other => return Err(mpe(format!("unknown parameter [{other}] on metadata field [_source]"))),
                        }
                    }
                    let mut norm = val.clone();
                    if let Some(Value::Bool(_) | Value::String(_)) = norm.get("enabled") {
                        norm["enabled"] = json!(crate::fieldtypes::params::to_bool(&norm["enabled"])?);
                    }
                    m.source = Some(norm);
                }
                "_routing" => {
                    let o = val.as_object().ok_or_else(|| mpe("_routing must be an object"))?;
                    let mut norm = Map::new();
                    for (ik, iv) in o {
                        match ik.as_str() {
                            "required" => {
                                norm.insert("required".into(), json!(crate::fieldtypes::params::to_bool(iv)?));
                            }
                            // `RoutingFieldMapper.docValues` (9.x): routing kept in
                            // doc values, not as a stored field.
                            "doc_values" => {
                                norm.insert("doc_values".into(), json!(crate::fieldtypes::params::to_bool(iv)?));
                            }
                            other => return Err(mpe(format!("unknown parameter [{other}] on metadata field [_routing]"))),
                        }
                    }
                    m.routing = Some(Value::Object(norm));
                }
                "_meta" => {
                    if !val.is_object() {
                        return Err(java_cast("Map"));
                    }
                    m.meta = Some(val.clone());
                }
                "runtime" => {
                    let rt = normalize_runtime(val)?;
                    // `IndexMode.validateNoMappingRuntimeFields`.
                    if registry.strict_columnar() && rt.as_object().is_some_and(|o| !o.is_empty()) {
                        return Err(EsError::illegal_argument(format!("mapping-level runtime fields are not allowed in index using [{}] index mode", registry.index_mode_name())));
                    }
                    m.runtime = Some(rt);
                }
                "_data_stream_timestamp" => {
                    let o = val.as_object().ok_or_else(|| EsError::mapper_parsing("[_data_stream_timestamp] config must be an object"))?;
                    for (ik, iv) in o {
                        if ik != "enabled" {
                            return Err(EsError::mapper_parsing(format!("Unknown parameter [{ik}] for mapper [_data_stream_timestamp]")));
                        }
                        if iv.as_bool() == Some(false) || iv.as_str() == Some("false") {
                            return Err(EsError::mapper_parsing("[_data_stream_timestamp] meta field has been disabled"));
                        }
                    }
                    m.data_stream_timestamp = true;
                }
                "_id" => {
                    // `ProvidedIdFieldMapper` `mode` (`document` / `columnar`).
                    let im = registry.index_mode.as_deref().map(str::to_ascii_lowercase);
                    if im.as_deref() == Some("time_series") {
                        return Err(mpe("_id is not configurable if index mode is time_series"));
                    }
                    let o = val.as_object().ok_or_else(|| mpe(format!("Expected map for property [_id] but got a class {}", java_class_of(val))))?;
                    for (ik, iv) in o {
                        if ik != "mode" {
                            return Err(mpe(format!("unknown parameter [{ik}] on metadata field [_id]")));
                        }
                        let mode = iv.as_str().map(str::to_ascii_lowercase).unwrap_or_default();
                        if !matches!(mode.as_str(), "document" | "columnar") {
                            return Err(mpe(format!("Unknown value [{}] for field [mode] - accepted values are [document, columnar]", jstr(iv))));
                        }
                        if mode == "document" && registry.strict_columnar() {
                            return Err(mpe(format!("_id does not support [mode=document] in a strictly columnar index mode [{}]", registry.index_mode_name())));
                        }
                        m.id_mode = Some(mode);
                    }
                }
                "_index" => return Err(mpe(format!("{k} is not configurable"))),
                "_field_names" | "subobjects" => {}
                other => unsupported.push(format!("{other} : {}", jstr(val))),
            }
        }
        if !unsupported.is_empty() {
            return Err(EsError::mapper_parsing(format!("Root mapping definition has unsupported parameters:  [{}]", unsupported.join("] ["))));
        }
        // `IndexMode.validateAllFieldsReconstructableFromDocValues`.
        if m.root.columnar {
            fn first(props: &BTreeMap<String, Mapper>, prefix: &str) -> Option<String> {
                props.iter().find_map(|(k, mp)| {
                    let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    match mp {
                        Mapper::Field(f) if f.params.get("doc_values") == Some(&Value::Bool(false)) => Some(full),
                        Mapper::Object(o) => first(&o.properties, &full),
                        _ => None,
                    }
                })
            }
            if let Some(f) = first(&m.root.properties, "") {
                return Err(EsError::illegal_argument(format!(
                    "field [{f}] cannot reconstruct _source from doc values; every field must be reconstructable from doc values in index using [{}] index mode",
                    registry.index_mode_name()
                )));
            }
        }
        Ok(m)
    }

    /// Pass-through objects with their priority, in mapping order.
    pub fn passthrough_objects(&self) -> Vec<(String, i64, bool)> {
        fn walk(props: &BTreeMap<String, Mapper>, prefix: &str, out: &mut Vec<(String, i64, bool)>) {
            for (k, m) in props {
                if let Mapper::Object(o) = m {
                    let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    if o.is_passthrough() {
                        let dim = o.params.get("time_series_dimension").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))) == Some(true);
                        out.push((full.clone(), o.params.get("priority").and_then(Value::as_i64).unwrap_or(-1), dim));
                    }
                    walk(&o.properties, &full, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.root.properties, "", &mut out);
        for (p, pp) in &self.root.prefix {
            if let Some(prio) = pp.passthrough {
                out.push((p.clone(), prio, false));
            }
        }
        out
    }

    /// Two pass-through objects may not share a priority.
    pub fn validate_passthrough(&self) -> Result<()> {
        let objs = self.passthrough_objects();
        for (i, (name, prio, _)) in objs.iter().enumerate() {
            if let Some((other, _, _)) = objs[..i].iter().find(|(_, p, _)| p == prio) {
                return Err(EsError::new(500, "mapper_exception", format!("Pass-through object [{name}] has a conflicting param [priority={prio}] with object [{other}]")));
            }
        }
        Ok(())
    }

    /// Is `path` a (dynamic) leaf of a pass-through object whose fields are
    /// time series dimensions?
    pub fn under_passthrough_dimension(&self, path: &str) -> bool {
        self.passthrough_objects().iter().any(|(p, _, dim)| *dim && path.len() > p.len() + 1 && path.starts_with(p.as_str()) && path.as_bytes()[p.len()] == b'.')
    }

    /// `copy_to` targets must be leaf fields in a compatible nested scope, and
    /// multi-fields cannot copy.
    pub fn validate_copy_to(&self) -> Result<()> {
        struct Src {
            path: String,
            nested: Option<String>,
            targets: Vec<String>,
            multi: bool,
        }
        fn walk(props: &BTreeMap<String, Mapper>, prefix: &str, nested: Option<&str>, srcs: &mut Vec<Src>, objects: &mut Vec<(String, bool)>) {
            for (k, m) in props {
                let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                match m {
                    Mapper::Field(f) => {
                        if let Some(c) = f.params.get("copy_to").and_then(Value::as_array) {
                            srcs.push(Src { path: full.clone(), nested: nested.map(str::to_string), targets: c.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(), multi: false });
                        }
                        for (sk, sf) in &f.fields {
                            if let Some(c) = sf.params.get("copy_to").and_then(Value::as_array) {
                                if !c.is_empty() {
                                    srcs.push(Src { path: format!("{full}.{sk}"), nested: nested.map(str::to_string), targets: Vec::new(), multi: true });
                                }
                            }
                        }
                    }
                    Mapper::Object(o) => {
                        objects.push((full.clone(), o.nested));
                        let np = if o.nested { Some(full.clone()) } else { nested.map(str::to_string) };
                        walk(&o.properties, &full, np.as_deref(), srcs, objects);
                    }
                }
            }
        }
        let (mut srcs, mut objects) = (Vec::new(), Vec::new());
        walk(&self.root.properties, "", None, &mut srcs, &mut objects);
        for s in &srcs {
            if s.multi {
                return Err(EsError::illegal_argument(format!("[copy_to] may not be used to copy from a multi-field: [{}]", s.path)));
            }
            for t in &s.targets {
                if objects.iter().any(|(o, _)| o == t) {
                    return Err(EsError::illegal_argument(format!("Cannot copy to field [{t}] since it is mapped as an object")));
                }
                // The nested scope of the target: the deepest nested object above it.
                let tn = objects.iter().filter(|(o, nested)| *nested && t.starts_with(&format!("{o}."))).map(|(o, _)| o.clone()).max_by_key(String::len);
                if let Some(tn) = tn {
                    let ok = s.nested.as_deref().is_some_and(|sn| sn == tn || sn.starts_with(&format!("{tn}.")));
                    if !ok {
                        return Err(EsError::illegal_argument(format!(
                            "Illegal combination of [copy_to] and [nested] mappings: [copy_to] may only copy data to the current nested document or any of its parents, however one [copy_to] directive is trying to copy data from nested object [{}] to [{tn}]",
                            s.nested.as_deref().unwrap_or("null")
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Is the object at `path` (`""` = root) mapped with `subobjects: false`?
    pub fn no_subobjects_at(&self, path: &str) -> bool {
        let mut node = &self.root;
        let mut path = path;
        // A columnar root has no objects: every path is flat, except inside
        // a nested object (kept under its full dotted path).
        if self.root.columnar && !path.is_empty() {
            let nested = self.root.properties.iter().find_map(|(k, m)| match m {
                Mapper::Object(o) if o.nested && (path == k || path.strip_prefix(k.as_str()).is_some_and(|r| r.starts_with('.'))) => Some((k.len(), o)),
                _ => None,
            });
            match nested {
                Some((n, o)) => {
                    node = o;
                    path = path[n..].trim_start_matches('.');
                }
                None => return true,
            }
        }
        if !path.is_empty() {
            let mut rest = path;
            'outer: while !rest.is_empty() {
                if node.no_subobjects() {
                    return false;
                }
                // The next object may carry dots in its name only under subobjects:false.
                let mut idx = 0;
                loop {
                    let end = rest[idx..].find('.').map(|p| idx + p).unwrap_or(rest.len());
                    if let Some(Mapper::Object(o)) = node.properties.get(&rest[..end]) {
                        node = o;
                        rest = rest[end..].trim_start_matches('.');
                        continue 'outer;
                    }
                    if end == rest.len() {
                        return false;
                    }
                    idx = end + 1;
                }
            }
        }
        node.no_subobjects()
    }

    /// `_size` is enabled.
    pub fn size_enabled(&self) -> bool {
        self.size.as_ref().and_then(|s| s.get("enabled")).and_then(Value::as_bool).unwrap_or(false)
    }

    /// Every `alias` must point at an existing concrete leaf (not an alias,
    /// not an object), and only one `join` field may exist.
    pub fn validate_aliases(&self) -> Result<()> {
        let mut leaves: BTreeMap<String, EsType> = BTreeMap::new();
        let mut objects: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        fn walk(props: &BTreeMap<String, Mapper>, prefix: &str, leaves: &mut BTreeMap<String, EsType>, objects: &mut std::collections::BTreeSet<String>, nested: Option<&str>, nested_of: &mut BTreeMap<String, Option<String>>) {
            for (k, m) in props {
                let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                match m {
                    Mapper::Field(f) => {
                        leaves.insert(full.clone(), f.ty);
                        nested_of.insert(full.clone(), nested.map(str::to_string));
                        for s in f.fields.keys() {
                            leaves.insert(format!("{full}.{s}"), f.fields[s].ty);
                        }
                    }
                    Mapper::Object(o) => {
                        objects.insert(full.clone());
                        let np = if o.nested { Some(full.clone()) } else { nested.map(str::to_string) };
                        walk(&o.properties, &full, leaves, objects, np.as_deref(), nested_of);
                    }
                }
            }
        }
        let mut nested_of = BTreeMap::new();
        walk(&self.root.properties, "", &mut leaves, &mut objects, None, &mut nested_of);
        let joins: Vec<&String> = leaves.iter().filter(|(_, t)| **t == EsType::Join).map(|(k, _)| k).collect();
        if joins.len() > 1 {
            return Err(EsError::illegal_argument(format!("Only one [parent-join] field can be defined per index, got [{}]", joins.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))));
        }
        fn find<'a>(props: &'a BTreeMap<String, Mapper>, path: &str) -> Option<&'a FieldMapping> {
            if let Some(Mapper::Field(f)) = props.get(path) {
                return Some(f);
            }
            let mut idx = 0;
            while let Some(p) = path[idx..].find('.') {
                let (head, rest) = (&path[..idx + p], &path[idx + p + 1..]);
                if let Some(Mapper::Object(o)) = props.get(head) {
                    if let Some(f) = find(&o.properties, rest) {
                        return Some(f);
                    }
                }
                idx += p + 1;
            }
            None
        }
        for (name, ty) in &leaves {
            if *ty != EsType::Alias {
                continue;
            }
            let Some(f) = find(&self.root.properties, name) else { continue };
            let path = f.str_param("path").unwrap_or("");
            if path == name {
                return Err(EsError::mapper_parsing(format!("Invalid [path] value [{path}] for field alias [{name}]: an alias cannot refer to itself.")));
            }
            match leaves.get(path) {
                None if matches!(path, "_id" | "_index") => {}
                None if objects.contains(path) => {
                    return Err(EsError::mapper_parsing(format!("Invalid [path] value [{path}] for field alias [{name}]: an alias must refer to an existing field in the mappings.")));
                }
                None => {
                    return Err(EsError::mapper_parsing(format!("Invalid [path] value [{path}] for field alias [{name}]: an alias must refer to an existing field in the mappings.")));
                }
                Some(EsType::Alias) => {
                    return Err(EsError::mapper_parsing(format!("Invalid [path] value [{path}] for field alias [{name}]: an alias cannot refer to another alias.")));
                }
                _ => {}
            }
            let an = nested_of.get(name).cloned().flatten();
            let tn = nested_of.get(path).cloned().flatten();
            if let Some(tn) = &tn {
                if an.as_deref().map(|a| !(a == tn || a.starts_with(&format!("{tn}.")))).unwrap_or(true) {
                    return Err(EsError::illegal_argument(format!("Invalid [path] value [{path}] for field alias [{name}]: an alias must have the same nested scope as its target. The alias is not nested, but the target's nested scope is [{tn}].")));
                }
            }
        }
        Ok(())
    }

    /// The ES mapping JSON (`{"properties": ...}`), as GET _mapping returns it.
    pub fn to_json(&self) -> Value {
        self.to_json_with(true)
    }

    /// Exact two-pass borrowed encoding. The caller holds the metadata read
    /// lock across both passes, so no mapping tree or string is copied first.
    pub fn to_json_admitted(&self) -> Result<(String, crate::memory::Reservation)> {
        self.to_json_admitted_with(crate::memory::breakers())
    }
    fn to_json_admitted_with(&self, breakers: &'static crate::memory::Breakers) -> Result<(String, crate::memory::Reservation)> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.checked_add(b.len()).ok_or_else(|| std::io::Error::other("mapping length overflow"))?; Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, &wire::Wire(self))?;
        let memory = breakers.reserve_request(count.0 as u64, "replication mapping JSON")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(count.0).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("mapping allocation refused: {e}")))?;
        struct Fixed<'a>(&'a mut Vec<u8>);
        impl std::io::Write for Fixed<'_> {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if b.len() > self.0.capacity() - self.0.len() { return Err(std::io::Error::other("mapping encoding exceeded admitted size")); }
                self.0.extend_from_slice(b); Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        serde_json::to_writer(Fixed(&mut bytes), &wire::Wire(self))?;
        let text = String::from_utf8(bytes).map_err(|e| EsError::internal(e.to_string()))?;
        Ok((text, memory))
    }


    fn to_json_with(&self, hide_internal: bool) -> Value {
        let mut m = Map::new();
        if let Some(d) = self.root.dynamic {
            m.insert("dynamic".into(), d.to_json());
        }
        if let Some(e) = self.root.enabled {
            m.insert("enabled".into(), json!(e));
        }
        if !self.dynamic_templates.is_empty() || self.dt_explicit {
            m.insert(
                "dynamic_templates".into(),
                Value::Array(self.dynamic_templates.iter().map(|t| json!({ t.name.clone(): t.raw })).collect()),
            );
        }
        if let Some(d) = self.date_detection {
            m.insert("date_detection".into(), json!(d));
        }
        if let Some(d) = self.numeric_detection {
            m.insert("numeric_detection".into(), json!(d));
        }
        if let Some(f) = &self.dynamic_date_formats {
            m.insert("dynamic_date_formats".into(), json!(f));
        }
        if let Some(s) = &self.source {
            let mut src = s.clone();
            // Remove the stored mode; we'll re-insert the effective mode.
            if let Some(o) = src.as_object_mut() {
                o.remove("mode");
            }
            // Re-insert the effective mode only if the mapping declared a
            // mode or the setting is present.
            let declared_mode = self.source.as_ref().and_then(|s| s.get("mode")).is_some();
            if declared_mode || self.source_mode_setting.is_some() {
                if let Some(em) = self.effective_source_mode() {
                    if let Some(o) = src.as_object_mut() {
                        o.insert("mode".into(), Value::String(em.to_string()));
                    }
                }
            }
            // If the resulting object is empty, emit `_source: {}`.
            m.insert("_source".into(), src);
        }
        if let Some(r) = self.routing.as_ref() {
            // Only non-default parameters are rendered.
            let mut o = Map::new();
            if r.get("required").and_then(Value::as_bool).unwrap_or(false) {
                o.insert("required".into(), json!(true));
            }
            if r.get("doc_values").and_then(Value::as_bool).unwrap_or(false) {
                o.insert("doc_values".into(), json!(true));
            }
            if !o.is_empty() {
                m.insert("_routing".into(), Value::Object(o));
            }
        }
        if let Some(meta) = &self.meta {
            m.insert("_meta".into(), meta.clone());
        }
        if let Some(rt) = &self.runtime {
            m.insert("runtime".into(), rt.clone());
        }
        if let Some(sz) = &self.size {
            m.insert("_size".into(), sz.clone());
        }
        if self.data_stream_timestamp {
            m.insert("_data_stream_timestamp".into(), json!({"enabled": true}));
        }
        if let Some(mode) = &self.id_mode {
            m.insert("_id".into(), json!({ "mode": mode }));
        }
        if let Some(so) = self.root.params.get("subobjects") {
            m.insert("subobjects".into(), so.clone());
        }
        if !self.root.prefix.is_empty() {
            m.insert("prefix_properties".into(), Value::Object(self.root.prefix.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()));
        }
        if !self.root.properties.is_empty() {
            let mut props = props_json(&self.root.properties);
            if hide_internal {
                // `_tsid` (and a synthetic id's routing hash) are hidden
                // fields of time series indices.
                if let Some(o) = props.as_object_mut() {
                    o.remove(crate::tsdb::TSID_FIELD);
                    o.remove(crate::tsdb::ROUTING_HASH_FIELD);
                }
            }
            if props.as_object().is_some_and(|o| !o.is_empty()) {
                m.insert("properties".into(), props);
            }
        }
        Value::Object(m)
    }

    /// Persisted form: ES JSON plus core ids.
    pub fn to_persisted(&self) -> Value {
        let mut v = json!({ "mapping": self.to_json_with(false), "core_ids": self.core_ids, "next_core_id": self.next_core_id });
        if let Some(ref s) = self.source_mode_setting {
            v["source_mode_setting"] = Value::String(s.clone());
        }
        v
    }

    /// Give the index a `_doc_count` doc-values column (first document
    /// carrying `_doc_count`). Returns whether it was added.
    pub fn enable_doc_count(&mut self) -> bool {
        if self.core_ids.contains_key(DOC_COUNT_FIELD_NAME) {
            return false;
        }
        self.core_ids.insert(DOC_COUNT_FIELD_NAME.to_string(), self.next_core_id);
        self.next_core_id += 1;
        true
    }

    pub fn from_persisted(v: &Value, registry: &AnalysisRegistry) -> Result<Mappings> {
        let mut m = Mappings::parse(&v["mapping"], registry)?;
        if let Some(ids) = v.get("core_ids").and_then(Value::as_object) {
            for (k, id) in ids {
                m.core_ids.insert(k.clone(), id.as_u64().unwrap_or(0) as u16);
            }
        }
        m.next_core_id = v.get("next_core_id").and_then(Value::as_u64).map(|x| x as u16).unwrap_or(FIRST_USER_FIELD);
        m.source_mode_setting = v.get("source_mode_setting").and_then(Value::as_str).map(str::to_string);
        m.assign_core_ids();
        Ok(m)
    }

    /// Merge `new` into self (put-mapping semantics). Returns an error for
    /// conflicting changes; self is left unchanged on error.
    pub fn merge(&mut self, new: &Mappings) -> Result<()> {
        let mut next = self.clone();
        next.root.merge("", &new.root)?;
        let conflict = |meta: &str, k: &str, old: &Value, nv: &Value| {
            EsError::illegal_argument(format!(
                "Mapper for [{meta}] conflicts with existing mapper:\n\tCannot update parameter [{k}] from [{}] to [{}]",
                crate::fieldtypes::params::jstr(old),
                crate::fieldtypes::params::jstr(nv)
            ))
        };
        if let Some(nm) = &new.id_mode {
            let cur = self.id_mode.clone().unwrap_or_else(|| "document".into());
            if &cur != nm {
                return Err(conflict("_id", "mode", &json!(cur), &json!(nm)));
            }
            next.id_mode = Some(nm.clone());
        }
        if let Some(src) = &new.source {
            if let Some(old) = &self.source {
                for k in ["includes", "excludes"] {
                    let (o, n) = (old.get(k).cloned().unwrap_or(json!([])), src.get(k).cloned().unwrap_or(json!([])));
                    if o != n {
                        return Err(conflict("_source", k, &o, &n));
                    }
                }
                // `mode` is only updatable to a *different* value when the
                // index has no effective mode yet (the setting is unset and
                // the mapping never declared one); otherwise the mapper
                // conflicts, reporting the *current* value (null when unset).
                if let Some(nv) = src.get("mode") {
                    let cur = self
                        .source
                        .as_ref()
                        .and_then(|s| s.get("mode"))
                        .cloned()
                        .or_else(|| self.effective_source_mode().map(|m| Value::String(m.to_string())));
                    if cur.as_ref().and_then(Value::as_str) != nv.as_str() {
                        return Err(conflict("_source", "mode", cur.as_ref().unwrap_or(&Value::Null), nv));
                    }
                }
            }
            next.source = Some(src.clone());
        }
        if let Some(r) = &new.routing {
            if self.routing.is_some() {
                let o = self.routing_required();
                let n = r.get("required").and_then(Value::as_bool).unwrap_or(false);
                if o != n {
                    return Err(conflict("_routing", "required", &json!(o), &json!(n)));
                }
            }
            next.routing = Some(r.clone());
        }
        if new.meta.is_some() {
            next.meta = new.meta.clone();
        }
        if let Some(rt) = &new.runtime {
            let mut merged = self.runtime.as_ref().and_then(Value::as_object).cloned().unwrap_or_default();
            if let Some(o) = rt.as_object() {
                for (k, v) in o {
                    merged.insert(k.clone(), v.clone());
                }
            }
            next.runtime = Some(Value::Object(merged));
        }
        if new.size.is_some() {
            next.size = new.size.clone();
        }
        if new.data_stream_timestamp {
            next.data_stream_timestamp = true;
        }
        next.validate_passthrough()?;
        next.validate_aliases()?;
        next.validate_copy_to()?;
        // A mapping update that names `dynamic_templates` replaces the list.
        if new.dt_explicit {
            next.dynamic_templates = new.dynamic_templates.clone();
            next.dt_explicit = true;
        }
        if new.date_detection.is_some() {
            next.date_detection = new.date_detection;
        }
        if new.numeric_detection.is_some() {
            next.numeric_detection = new.numeric_detection;
        }
        if new.dynamic_date_formats.is_some() {
            next.dynamic_date_formats = new.dynamic_date_formats.clone();
        }
        next.assign_core_ids();
        *self = next;
        Ok(())
    }

    /// Give every leaf a core id (new ones only).
    pub fn assign_core_ids(&mut self) -> Vec<String> {
        let mut paths = Vec::new();
        fn walk<'m>(props: &'m BTreeMap<String, Mapper>, prefix: &str, out: &mut Vec<(String, &'m FieldMapping)>) {
            for (k, m) in props {
                let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                match m {
                    Mapper::Field(f) => {
                        out.push((full.clone(), f));
                        for (s, sf) in &f.fields {
                            out.push((format!("{full}.{s}"), sf));
                        }
                    }
                    Mapper::Object(o) => walk(&o.properties, &full, out),
                }
            }
        }
        let mut leaves = Vec::new();
        walk(&self.root.properties, "", &mut leaves);
        let mut leaves: Vec<(String, FieldMapping)> = leaves.into_iter().map(|(p, f)| (p, f.clone())).collect();
        // `flattened` mapped sub-fields are leaves of their own.
        let subs: Vec<(String, FieldMapping)> = leaves.iter().flat_map(|(p, f)| f.flattened_mapped().into_iter().map(move |(k, sf)| (format!("{p}.{k}"), sf))).collect();
        leaves.extend(subs);
        if self.size_enabled() {
            leaves.push(("_size".to_string(), FieldMapping::new(EsType::Long)));
        }
        let mut added = Vec::new();
        for (p, fm) in leaves {
            let ty = fm.ty;
            let keys: Vec<String> = match ty {
                // The core geo field holds the quantized points and doc
                // values (ES's `LatLonPoint` / `LatLonDocValuesField`).
                // Indices created before kept `#lat` / `#lon` columns too.
                EsType::GeoPoint => vec![format!("{p}#geo")],
                EsType::Alias | EsType::Object | EsType::Nested => vec![],
                // Text doc values default on in strict columnar modes.
                EsType::Text | EsType::MatchOnlyText if self.root.columnar && fm.param("doc_values").is_none() => vec![p.clone(), format!("{p}#dv")],
                _ => crate::fieldtypes::core_keys(&p, &fm).unwrap_or_else(|| vec![p.clone()]),
            };
            for k in keys {
                if !self.core_ids.contains_key(&k) {
                    self.core_ids.insert(k.clone(), self.next_core_id);
                    self.next_core_id += 1;
                    added.push(k);
                }
            }
            paths.push(p);
        }
        // The stored-fields block's own (stored, never indexed) field: in
        // `_field_names`' indexed field its JSON became one keyword term
        // per document.
        if !self.core_ids.contains_key(STORED_BLOCK_KEY) {
            self.core_ids.insert(STORED_BLOCK_KEY.to_string(), self.next_core_id);
            self.next_core_id += 1;
            added.push(STORED_BLOCK_KEY.to_string());
        }
        added
    }

    pub fn source_enabled(&self) -> bool {
        self.source.as_ref().and_then(|s| s.get("enabled")).and_then(Value::as_bool).unwrap_or(true)
    }

    /// The `_source.mode` the index was created with (the mapping's declared
    /// mode, or the index-mode default when the mapping never declared one).
    /// `None` when the index has no effective mode (standard index, no
    /// setting, no mapping declaration) — `put_mapping` may then set one.
    pub fn effective_source_mode(&self) -> Option<crate::synthetic::SourceMode> {
        use crate::synthetic::SourceMode;
        // The `index.mapping.source.mode` setting wins over the mapping's
        // `_source.mode`.
        if let Some(ref s) = self.source_mode_setting {
            return Some(match s.as_str() {
                "synthetic" | "columnar_stored" => SourceMode::Synthetic,
                "disabled" => SourceMode::Disabled,
                _ => SourceMode::Stored,
            });
        }
        let declared = self.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str);
        if declared.is_some() {
            return Some(match declared.map(str::to_ascii_lowercase).as_deref() {
                Some("synthetic") => SourceMode::Synthetic,
                Some("disabled") => SourceMode::Disabled,
                _ => SourceMode::Stored,
            });
        }
        if !self.source_enabled() {
            return Some(SourceMode::Disabled);
        }
        None
    }

    /// `_routing.doc_values`: routing values live in doc values (not
    /// returned as document metadata).
    pub fn routing_doc_values(&self) -> bool {
        // `RoutingFieldMapper`: on by default in strict columnar index modes.
        self.routing.as_ref().and_then(|s| s.get("doc_values")).and_then(Value::as_bool).unwrap_or(self.root.columnar)
    }

    pub fn routing_required(&self) -> bool {
        self.routing.as_ref().and_then(|s| s.get("required")).and_then(Value::as_bool).unwrap_or(false)
    }

    pub fn total_fields(&self) -> usize {
        fn count(props: &BTreeMap<String, Mapper>) -> usize {
            props
                .values()
                .map(|m| match m {
                    Mapper::Field(f) => 1 + f.fields.len(),
                    Mapper::Object(o) => 1 + count(&o.properties),
                })
                .sum()
        }
        count(&self.root.properties)
    }

    /// Insert a mapper at a dotted path, creating parent objects (used for
    /// dynamic updates).
    pub fn insert_at(&mut self, path: &str, mapper: Mapper) {
        self.insert_at_with(path, mapper, None)
    }

    /// [`Mappings::insert_at`] that creates missing intermediate objects
    /// through the dynamic templates (when `registry` is given) and keeps
    /// the rest of the path as one leaf name under `subobjects: false`.
    pub fn insert_at_with(&mut self, path: &str, mapper: Mapper, registry: Option<&AnalysisRegistry>) {
        let parts: Vec<&str> = path.split('.').collect();
        let templates: Vec<Option<Mapper>> = match registry {
            Some(reg) => (0..parts.len().saturating_sub(1))
                .map(|i| {
                    let sub = parts[..=i].join(".");
                    self.dynamic_templates
                        .iter()
                        .any(|t| t.matches(parts[i], &sub, "object"))
                        .then(|| self.dynamic_mapper(parts[i], &sub, "object", reg).ok().flatten())
                        .flatten()
                })
                .collect(),
            None => Vec::new(),
        };
        // A columnar root keeps nested objects under their full dotted
        // path: a field inside one is mapped inside it.
        let nested_hit = if self.root.columnar {
            (1..parts.len()).rev().find(|&j| matches!(self.root.properties.get(&parts[..j].join(".")), Some(Mapper::Object(o)) if o.nested))
        } else {
            None
        };
        let (mut node, mut i) = match nested_hit {
            Some(j) => match self.root.properties.get_mut(&parts[..j].join(".")) {
                Some(Mapper::Object(o)) => (o, j),
                _ => return,
            },
            None => (&mut self.root, 0),
        };
        while i + 1 < parts.len() {
            if node.no_subobjects() {
                break;
            }
            let p = parts[i];
            let tmpl = templates.get(i).cloned().flatten();
            let e = node.properties.entry(p.to_string()).or_insert_with(|| match tmpl {
                Some(m @ Mapper::Object(_)) => m,
                _ => Mapper::Object(ObjectMapping::default()),
            });
            node = match e {
                Mapper::Object(o) => o,
                Mapper::Field(_) => return,
            };
            i += 1;
        }
        let last = parts[i..].join(".");
        match (node.properties.get_mut(&last), mapper) {
            (Some(Mapper::Field(f)), Mapper::Field(nf)) => {
                // e.g. dense_vector dims learnt from the first document
                for (k, v) in nf.params {
                    match f.params.get(&k) {
                        None | Some(Value::Null) => {
                            f.params.insert(k, v);
                        }
                        _ => {}
                    }
                }
                // ... or a multi-field's (semantic_text model settings).
                for (sub, snf) in nf.fields {
                    if let Some(sf) = f.fields.get_mut(&sub) {
                        for (k, v) in snf.params {
                            if sf.params.get(&k).is_none_or(Value::is_null) {
                                sf.params.insert(k, v);
                            }
                        }
                    }
                }
            }
            (None, m) => {
                node.properties.insert(last, m);
            }
            _ => {}
        }
    }

    /// Mapper for a dynamically seen value (templates first, then the
    /// default rules). `kind` is ES's dynamic type: `string`, `long`,
    /// `double`, `boolean`, `date`, `object`.
    pub fn dynamic_mapper(&self, name: &str, path: &str, kind: &str, registry: &AnalysisRegistry) -> Result<Option<Mapper>> {
        self.dynamic_mapper_named(name, path, kind, None, registry)
    }

    /// [`Mappings::dynamic_mapper`] with the template requested by name
    /// (bulk `dynamic_templates`): the named template applies whatever its
    /// match conditions; a missing name is a parsing error.
    pub fn dynamic_mapper_named(&self, name: &str, path: &str, kind: &str, requested: Option<&str>, registry: &AnalysisRegistry) -> Result<Option<Mapper>> {
        if let Some(r) = requested {
            if !self.dynamic_templates.iter().any(|t| t.name == r) {
                return Err(EsError::document_parsing(format!("[1:1] Can't find dynamic template for dynamic template name [{r}] of field [{path}]")));
            }
        }
        for t in &self.dynamic_templates {
            let hit = match requested {
                Some(r) => t.name == r,
                None => t.matches(name, path, kind),
            };
            if hit {
                let Some(mapping) = &t.mapping else {
                    // runtime template: not indexed here
                    return Ok(None);
                };
                let default_type = match kind {
                    "string" => "text",
                    "double" => "float",
                    k => k,
                };
                let spec = substitute_with(mapping, name, default_type, &crate::document::template_params(path));
                let mut spec = spec.as_object().cloned().unwrap_or_default();
                if !spec.contains_key("type") {
                    spec.insert("type".into(), json!(if kind == "object" { "object" } else { default_type }));
                }
                let mut holder = ObjectMapping::default();
                holder.insert_property("", name, &Value::Object(spec), registry, 0).map_err(wrap_parse_err)?;
                return Ok(holder.properties.remove(name));
            }
        }
        Ok(Some(match kind {
            "string" if registry.dynamic_strings_keyword_only => Mapper::Field(FieldMapping::new(EsType::Keyword)),
            "string" => {
                let mut f = FieldMapping::new(EsType::Text);
                let mut kw = FieldMapping::new(EsType::Keyword);
                kw.params.insert("ignore_above".into(), json!(256));
                f.fields.insert("keyword".into(), kw);
                Mapper::Field(f)
            }
            "long" => Mapper::Field(FieldMapping::new(EsType::Long)),
            "double" => Mapper::Field(FieldMapping::new(EsType::Float)),
            "boolean" => Mapper::Field(FieldMapping::new(EsType::Boolean)),
            "date" => Mapper::Field(FieldMapping::new(EsType::Date)),
            "dense_vector" => {
                // Dynamic dense vectors are indexed with cosine similarity
                // (dims are learnt from the document).
                let mut f = FieldMapping::new(EsType::DenseVector);
                crate::vectors::normalize_mapping(name, &mut f.params, crate::vectors::VecMappingCtx::of(registry))?;
                Mapper::Field(f)
            }
            _ => Mapper::Object(ObjectMapping::default()),
        }))
    }

    /// Is `path` a runtime field of the mapping?
    pub fn has_runtime(&self, path: &str) -> bool {
        self.runtime.as_ref().and_then(|r| r.get(path)).is_some()
    }

    /// The runtime field definition for a dynamically seen value: a matching
    /// `runtime` dynamic template, else (under `dynamic: runtime`) the
    /// default type for `kind`. `None` when a template asks for a concrete
    /// field or no runtime field applies.
    pub fn dynamic_runtime(&self, name: &str, path: &str, kind: &str, requested: Option<&str>, dynamic_runtime: bool) -> Option<Value> {
        let default_type = match kind {
            "string" => "keyword",
            "long" => "long",
            "double" => "double",
            "boolean" => "boolean",
            "date" => "date",
            _ => return None,
        };
        for t in &self.dynamic_templates {
            let hit = match requested {
                Some(r) => t.name == r,
                None => t.matches(name, path, kind),
            };
            if !hit {
                continue;
            }
            let rt = t.raw.get("runtime")?;
            let mut def = substitute_with(rt, name, default_type, &crate::document::template_params(path)).as_object().cloned().unwrap_or_default();
            if def.get("type").and_then(Value::as_str).is_none_or(|x| x == "{dynamic_type}") {
                def.insert("type".into(), json!(default_type));
            }
            return Some(Value::Object(def));
        }
        dynamic_runtime.then(|| json!({"type": default_type}))
    }

    pub fn date_detection(&self) -> bool {
        self.date_detection.unwrap_or(true)
    }

    pub fn numeric_detection(&self) -> bool {
        self.numeric_detection.unwrap_or(false)
    }

    pub fn dynamic_date_formatters(&self) -> Vec<DateFormatter> {
        match &self.dynamic_date_formats {
            Some(f) => f.iter().filter_map(|x| DateFormatter::new(x).ok()).collect(),
            None => crate::date::default_dynamic_date_formats(),
        }
    }
}

/// Validate and canonicalise the `runtime` section of a mapping.
fn normalize_runtime(v: &Value) -> Result<Value> {
    use crate::fieldtypes::params::{jstr, mpe};
    let obj = v.as_object().ok_or_else(|| {
        let m = "runtime must be a map type";
        EsError::mapper_parsing(m).caused_by(EsError::new(400, "parse_exception", m))
    })?;
    let mut out = Map::new();
    for (name, def) in obj {
        let d = def.as_object().ok_or_else(|| mpe(format!("Expected map for runtime field [{name}] definition but got a {}", java_class_of(def))))?;
        let ty = d.get("type").and_then(Value::as_str).ok_or_else(|| mpe(format!("No type specified for runtime field [{name}]")))?;
        if !matches!(ty, "boolean" | "composite" | "date" | "double" | "geo_point" | "geo_shape" | "ip" | "keyword" | "long" | "lookup") {
            return Err(mpe(format!("The mapper type [{ty}] declared on runtime field [{name}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.")));
        }
        let mut nd = Map::new();
        for (k, x) in d {
            let allowed = match k.as_str() {
                "type" | "script" | "meta" | "on_script_error" => true,
                "format" => ty == "date",
                "fields" => ty == "composite",
                "target_index" | "input_field" | "target_field" | "fetch_fields" => ty == "lookup",
                _ => false,
            };
            if !allowed {
                return Err(mpe(format!("unknown parameter [{k}] on runtime field [{name}] of type [{ty}]")));
            }
            let nv = match k.as_str() {
                "script" => crate::fieldtypes::params::normalize(EsType::Long, "script", x)?.unwrap_or_else(|| x.clone()),
                "fetch_fields" => match x {
                    Value::Array(a) => Value::Array(a.iter().map(|f| if let Value::String(s) = f { json!({"field": s}) } else { f.clone() }).collect()),
                    o => o.clone(),
                },
                _ => x.clone(),
            };
            nd.insert(k.clone(), nv);
        }
        let _ = jstr;
        out.insert(name.clone(), Value::Object(nd));
    }
    Ok(Value::Object(out))
}

// ---------------------------------------------------------------------------
// Resolved view.

/// Everything document parsing and query building need about one leaf.
#[derive(Clone)]
pub struct ResolvedField {
    pub name: String,
    pub ty: EsType,
    pub core_id: Option<u16>,
    pub lat_id: Option<u16>,
    pub lon_id: Option<u16>,
    /// `geo_point`: the core geo field (KD-tree index + geo doc values).
    pub geo_id: Option<u16>,
    pub index_analyzer: Option<IndexAnalyzer>,
    pub index_analyzer_name: Option<String>,
    pub search_analyzer: Option<Arc<Analyzer>>,
    pub search_quote_analyzer: Option<Arc<Analyzer>>,
    pub normalizer: Option<Arc<Analyzer>>,
    /// Keyword `normalizer_skip_store_original_value` (default: the built-in
    /// `lowercase` normalizer): synthetic source returns the normalized
    /// doc values instead of storing the original value.
    pub normalizer_skip_store: bool,
    pub date_format: DateFormatter,
    pub ignore_above: Option<usize>,
    pub null_value: Option<Value>,
    pub coerce: bool,
    pub ignore_malformed: bool,
    pub scaling_factor: f64,
    pub dims: Option<usize>,
    pub similarity: String,
    pub indexed: bool,
    pub doc_values: bool,
    pub copy_to: Vec<String>,
    /// Full names of this field's multi-fields.
    pub multi_fields: Vec<String>,
    pub is_multi_field: bool,
    pub alias_path: Option<String>,
    pub constant_value: Option<Value>,
    pub nested_path: Option<String>,
    pub meta: Option<Value>,
    /// The mapped type (`ty` is the type whose behaviour it shares, e.g.
    /// `token_count` behaves like `integer`).
    pub declared: EsType,
    /// Type-specific configuration ([`crate::fieldtypes::FieldExt`]).
    pub ext: Option<Arc<crate::fieldtypes::FieldExt>>,
    /// Auxiliary core ids (`<path>#<suffix>`), by suffix.
    pub aux: Vec<(String, u16)>,
    /// Text: positions between array values.
    pub position_increment_gap: u32,
    /// Text: `norms` (per-doc length norms; default on).
    pub norms: bool,
    /// Text: the field's similarity counts position-overlapping tokens out
    /// of the length norm (`discount_overlaps`, default on).
    pub discount_overlaps: bool,
    /// `store: true`.
    pub stored: bool,
    /// Text: `fielddata: true` (uninverted index for sort/agg).
    pub fielddata: bool,
    /// `time_series_dimension: true` (time-series index).
    pub time_series_dimension: bool,
    /// `time_series_metric` (`counter` / `gauge` / `sum` / `avg`).
    pub time_series_metric: Option<String>,
    /// Index-time `script` computing the field's values.
    pub script: Option<Value>,
    /// `on_script_error: continue`.
    pub script_error_continue: bool,
    /// `doc_values.multi_value: false` (columnar): one value per document.
    pub single_value: bool,
    /// `index` as mapped, with the `index.mapping.index_disabled_by_default`
    /// default (strict columnar modes: off): what `*` field expansion uses.
    pub index_declared: bool,
    /// `doc_values.nullability: false` (columnar): every document needs a value.
    pub required: bool,
    /// Range-filtered through a doc-values skipper instead of points (ES
    /// `IndexType.skippers()`): the numeric column carries no points index.
    pub skipper: bool,
}

impl std::fmt::Debug for ResolvedField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedField").field("name", &self.name).field("ty", &self.ty).field("core_id", &self.core_id).finish()
    }
}

impl ResolvedField {
    /// Auxiliary core id by suffix.
    pub fn aux_id(&self, suffix: &str) -> Option<u16> {
        self.aux.iter().find(|(s, _)| s == suffix).map(|(_, id)| *id)
    }

    /// Key of a synthesized `flattened` sub-field.
    pub fn flattened_key(&self) -> Option<&str> {
        self.ext.as_ref().and_then(|e| e.flattened_key.as_deref())
    }

    pub fn is_searchable(&self) -> bool {
        self.indexed || self.doc_values
    }

    pub fn is_aggregatable(&self) -> bool {
        (self.doc_values || (self.ty.is_text() && self.fielddata))
            && !matches!(
                self.declared,
                EsType::DenseVector | EsType::Binary | EsType::Completion | EsType::Percolator | EsType::RankFeature | EsType::RankFeatures | EsType::SparseVector | EsType::RankVectors | EsType::GeoShape | EsType::Shape
            )
    }

    /// The core field definition backing this ES field.
    fn core_fields(&self) -> Vec<CoreField> {
        if let Some(v) = crate::fieldtypes::core_fields(self) {
            return v;
        }
        let mut out = Vec::new();
        let mk = |id: u16, name: &str, ty: CoreType, indexed: bool, fast: bool| {
            let mut f = CoreField::new(id, name, ty);
            f.indexed = indexed;
            f.fast = fast;
            f
        };
        let Some(id) = self.core_id else {
            if let (Some(lat), Some(lon)) = (self.lat_id, self.lon_id) {
                // Geo queries run on the `#geo` index; these columns serve
                // doc values (scripts, docvalue_fields, fallbacks) and never
                // a range query: no points index.
                out.push(mk(lat, &format!("{}#lat", self.name), CoreType::F64, false, true).with_points(false));
                out.push(mk(lon, &format!("{}#lon", self.name), CoreType::F64, false, true).with_points(false));
            }
            if let Some(g) = self.geo_id {
                out.push(mk(g, &format!("{}#geo", self.name), CoreType::GeoPoint, true, false));
            }
            return out;
        };
        let f = match self.ty {
            EsType::Text | EsType::MatchOnlyText => {
                let analyzer = match &self.index_analyzer {
                    Some(IndexAnalyzer::Builtin(n)) => n.clone(),
                    Some(IndexAnalyzer::Custom(_)) => "_pretokenized".into(),
                    None => "standard".into(),
                };
                // Text doc values: the whole values in a keyword column.
                if let Some(dv) = self.aux_id("dv") {
                    out.push(mk(dv, &format!("{}#dv", self.name), CoreType::Keyword, false, true));
                }
                // `match_only_text` scores every match alike (ES keeps no
                // norms for it): a norms column would only take space.
                let norms = self.norms && self.ty != EsType::MatchOnlyText;
                mk(id, &self.name, CoreType::Text, self.indexed, false).with_analyzer(&analyzer).with_position_increment_gap(self.position_increment_gap).with_norms(norms).with_discount_overlaps(self.discount_overlaps)
            }
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened | EsType::Ip => {
                mk(id, &self.name, CoreType::Keyword, self.indexed, self.doc_values)
            }
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => mk(id, &self.name, CoreType::I64, self.indexed, true).with_points(!self.skipper),
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => mk(id, &self.name, CoreType::F64, self.indexed, true).with_points(!self.skipper),
            EsType::UnsignedLong => mk(id, &self.name, CoreType::U64, self.indexed, true).with_points(!self.skipper),
            EsType::Boolean => mk(id, &self.name, CoreType::Bool, self.indexed, true),
            EsType::Date | EsType::DateNanos => mk(id, &self.name, CoreType::Date, self.indexed, true).with_points(!self.skipper),
            EsType::DenseVector => {
                let Some(d) = self.dims else { return out };
                // `index: false` (no ANN structure; script/rescore access
                // only) is stored like an explicit f32 `flat` field: the
                // mutable HNSW backend requires every graph-mapped field to
                // be explicit, and an implicit default would be a graph.
                let options = crate::vectors::field_options(self)
                    .unwrap_or_else(|| rustic_core::vector::store::VectorFieldOptions::f32_hnsw().flat());
                // `byte` / `bit` rows are packed: their core dim is the
                // row's lane count and their similarity the packed variant.
                let et = crate::vectors::element_type(self);
                let core_dim = crate::vectors::core_dims(et, d);
                mk(id, &self.name, CoreType::Vector { dim: core_dim as u16 }, true, false)
                    .with_similarity(&crate::vectors::core_similarity(et, &self.similarity)).with_vector_options(Some(options))
            }
            _ => return out,
        };
        out.push(f);
        out
    }
}

/// Whether ES 9.5.5 gives field `name` a doc-values skipper instead of
/// points (`DateFieldMapper` / `NumberFieldMapper` / `IpFieldMapper`
/// `indexType`): with `index.mapping.use_doc_values_skipper` and doc values,
/// a non-indexed field (indices created on or after
/// `STANDARD_INDEXES_USE_SKIPPERS`), and `@timestamp` when it is an index
/// sort field of a time series or logsdb index.
fn uses_skipper(f: &FieldMapping, name: &str, reg: &crate::analysis::AnalysisRegistry, indexed: bool, doc_values: bool) -> bool {
    use crate::version::IndexVersion;
    if !reg.use_dv_skipper || !doc_values {
        return false;
    }
    let standard = !indexed && reg.created_version.on_or_after(IndexVersion::STANDARD_INDEXES_USE_SKIPPERS);
    match f.ty {
        EsType::Date | EsType::DateNanos => {
            standard
                || (name == "@timestamp"
                    && matches!(reg.index_mode.as_deref().map(str::to_ascii_lowercase).as_deref(), Some("time_series" | "logsdb"))
                    && reg.sort_fields.iter().any(|s| s == name))
        }
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::UnsignedLong | EsType::Ip => standard,
        _ => false,
    }
}

/// Resolved fields of a mapping, keyed by full dotted path.
#[derive(Clone, Default, Debug)]
pub struct FieldLookup {
    /// The fields synthetic source rebuilds from their columns (computed
    /// on first use; see [`crate::synthetic::residual_for`]).
    pub synthetic_columns: std::sync::OnceLock<Arc<crate::synthetic::ColumnFields>>,
    /// TSDB selections belonging to this immutable mapping lookup.
    pub tsdb_fields: crate::tsdb::PreparedFields,
    /// (ahash: looked up several times per field of every indexed document.)
    pub fields: HashMap<String, Arc<ResolvedField>, ahash::RandomState>,
    /// Object paths with their effective `dynamic` / `enabled` / nested-ness.
    pub objects: HashMap<String, ObjectInfo, ahash::RandomState>,
    pub registry: AnalysisRegistry,
    pub root_dynamic: Option<Dynamic>,
    pub root_enabled: bool,
    /// `passthrough` objects with their priority.
    pub passthrough: Vec<(String, i64)>,
    /// Unprefixed names registered for `passthrough` leaves.
    pub passthrough_alias: std::collections::HashSet<String>,
    /// Synthesized `flattened` sub-fields.
    pub synth: crate::fieldtypes::SynthCache,
    /// `copy_to` target -> the fields copying into it (descending by name).
    pub copy_sources: HashMap<String, Vec<String>>,
    /// The `_index_mode` metadata field (a constant keyword: `index.mode`).
    pub index_mode_field: Option<Arc<ResolvedField>>,
    /// Core id of the `_doc_count` metadata field (doc values), once a
    /// document of the index carried one.
    pub doc_count_id: Option<u16>,
    /// The stored-fields block's field (`None`: [`STORED_FIELD`]).
    pub stored_block_id: Option<u16>,
    /// Some field has an index-time `script`.
    pub has_index_scripts: bool,
    /// Fields with `doc_values.nullability: false` (full name, nested scope).
    pub required: Vec<(String, Option<String>)>,
    /// Top-level `store: true` fields (not `_`-prefixed, not nested),
    /// sorted: the stored-fields block of every indexed document.
    pub stored_fields: Vec<String>,
    /// Conservative absence guards, rebuilt with each mapping lookup. A
    /// field-security view may remove fields, so true still requires scanning.
    pub may_have_vector_fields: bool,
    pub may_have_semantic_fields: bool,
    /// [`FieldLookup::core_schema_cached`]'s value for this instance.
    pub core_schema_memo: CoreSchemaMemo,
}

/// Memoized core schema of one [`FieldLookup`] instance. A clone starts
/// empty: clones are edited in place (field-security views remove fields),
/// so a copied value could be stale.
#[derive(Default, Debug)]
pub struct CoreSchemaMemo(std::sync::OnceLock<Arc<Vec<CoreField>>>);

impl Clone for CoreSchemaMemo {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectInfo {
    pub dynamic: Dynamic,
    pub enabled: bool,
    pub nested: bool,
    /// `nested` objects: also index the fields into the parent / root doc.
    pub include_in_parent: bool,
    pub include_in_root: bool,
}

/// [`FieldLookup::build`] calls so far (process-wide; benchmarks and
/// tests see how often the write path rebuilds lookups).
pub static LOOKUP_BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl FieldLookup {
    pub fn build(m: &Mappings, registry: &AnalysisRegistry) -> FieldLookup {
        LOOKUP_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut lk = FieldLookup { tsdb_fields: crate::tsdb::PreparedFields::new(m), registry: registry.clone(), root_dynamic: m.root.dynamic, root_enabled: m.root.enabled.unwrap_or(true), ..Default::default() };
        lk.doc_count_id = m.core_ids.get(DOC_COUNT_FIELD_NAME).copied();
        lk.stored_block_id = m.core_ids.get(STORED_BLOCK_KEY).copied();
        let root_dyn = m.root.dynamic.unwrap_or(Dynamic::True);
        lk.walk(m, &m.root.properties, "", root_dyn, None);
        // Columnar roots keep pass-through objects as prefixes.
        for (p, pp) in &m.root.prefix {
            if let Some(prio) = pp.passthrough {
                lk.passthrough.push((p.clone(), prio));
            }
        }
        crate::fieldtypes::extend_lookup(&mut lk, m);
        lk.may_have_vector_fields = lk.fields.values().any(|f| matches!(f.declared, EsType::DenseVector | EsType::SparseVector | EsType::RankVectors));
        lk.may_have_semantic_fields = lk.fields.values().any(|f| f.declared.is_semantic());
        lk.has_index_scripts = lk.fields.values().any(|f| f.script.is_some());
        lk.required = lk.fields.values().filter(|f| f.required && f.alias_path.is_none()).map(|f| (f.name.clone(), f.nested_path.clone())).collect();
        lk.required.sort();
        for (name, f) in &lk.fields {
            for t in &f.copy_to {
                lk.copy_sources.entry(t.clone()).or_default().push(name.clone());
            }
        }
        lk.copy_sources.values_mut().for_each(|v| v.sort_by(|a, b| b.cmp(a)));
        let mut stored: Vec<String> = lk.fields.iter().filter(|(n, f)| f.stored && f.nested_path.is_none() && !n.starts_with('_')).map(|(n, _)| n.clone()).collect();
        stored.sort();
        lk.stored_fields = stored;
        let mut im = reserved_long_field(INDEX_MODE_FIELD, u16::MAX);
        im.ty = EsType::ConstantKeyword;
        im.declared = EsType::ConstantKeyword;
        im.core_id = None;
        im.doc_values = true;
        im.indexed = true;
        im.constant_value = Some(json!(registry.index_mode.as_deref().unwrap_or("standard")));
        lk.index_mode_field = Some(Arc::new(im));
        lk
    }

    /// Resolve a field mapping outside the mapping tree (`_size`).
    pub fn resolve_public(&self, m: &Mappings, name: &str, f: &FieldMapping) -> ResolvedField {
        self.resolve(m, name, f, false, None)
    }

    fn walk(&mut self, m: &Mappings, props: &BTreeMap<String, Mapper>, prefix: &str, dynamic: Dynamic, nested: Option<&str>) {
        for (k, mapper) in props {
            let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            match mapper {
                Mapper::Object(o) => {
                    if o.params.get("type").and_then(Value::as_str) == Some("passthrough") {
                        let prio = o.params.get("priority").and_then(Value::as_i64).unwrap_or(0);
                        self.passthrough.push((full.clone(), prio));
                    }
                    let d = o.dynamic.unwrap_or(dynamic);
                    let flag = |k: &str| o.nested && o.params.get(k).and_then(Value::as_bool).unwrap_or(false);
                    self.objects.insert(
                        full.clone(),
                        ObjectInfo {
                            dynamic: d,
                            enabled: o.enabled.unwrap_or(true),
                            nested: o.nested,
                            include_in_parent: flag("include_in_parent"),
                            include_in_root: flag("include_in_root"),
                        },
                    );
                    let np = if o.nested { Some(full.clone()) } else { nested.map(str::to_string) };
                    self.walk(m, &o.properties, &full, d, np.as_deref());
                }
                Mapper::Field(f) => {
                    let rf = self.resolve(m, &full, f, false, nested);
                    let mut multi = Vec::new();
                    for (s, sf) in &f.fields {
                        let sub = format!("{full}.{s}");
                        let r = self.resolve(m, &sub, sf, true, nested);
                        self.fields.insert(sub.clone(), Arc::new(r));
                        multi.push(sub);
                    }
                    let mut rf = rf;
                    rf.multi_fields = multi;
                    // `flattened` mapped sub-fields (and their pass-through).
                    if f.ty == EsType::Flattened {
                        for (k, sf) in f.flattened_mapped() {
                            let sub = format!("{full}.{k}");
                            let r = self.resolve(m, &sub, &sf, false, nested);
                            self.fields.insert(sub, Arc::new(r));
                        }
                        if let Some(p) = f.params.get("passthrough").and_then(|p| p.get("priority")).and_then(Value::as_i64) {
                            self.passthrough.push((full.clone(), p));
                        }
                    }
                    // `pattern_text`: its `template_id` keyword sub-field.
                    if f.ty == EsType::PatternText {
                        let tid = format!("{full}.template_id");
                        if m.core_ids.contains_key(&tid) {
                            // Doc values only (`index: false`, low cardinality).
                            let mut kf = FieldMapping::new(EsType::Keyword);
                            kf.params.insert("index".into(), json!(false));
                            let r = self.resolve(m, &tid, &kf, true, nested);
                            self.fields.insert(tid, Arc::new(r));
                        }
                    }
                    self.fields.insert(full, Arc::new(rf));
                }
            }
        }
    }

    fn resolve(&self, m: &Mappings, name: &str, f: &FieldMapping, multi: bool, nested: Option<&str>) -> ResolvedField {
        let reg = &self.registry;
        let (core_id, lat_id, lon_id) = if f.ty == EsType::GeoPoint {
            (None, m.core_ids.get(&format!("{name}#lat")).copied(), m.core_ids.get(&format!("{name}#lon")).copied())
        } else {
            (m.core_ids.get(name).copied(), None, None)
        };
        let declared = f.ty;
        let behavior = crate::fieldtypes::behavior(f.ty);
        let (index_analyzer, index_analyzer_name, search_analyzer, quote) = if f.ty == EsType::PatternText {
            let an = crate::fieldtypes::pattern_text::analyzer_name(f.str_param("analyzer")).to_string();
            (Some(reg.index_analyzer(&an)), Some(an.clone()), reg.analyzer(&an), reg.analyzer(&an))
        } else if f.ty.is_text() {
            let an = f.str_param("analyzer").map(str::to_string).unwrap_or_else(|| reg.default_analyzer_name().to_string());
            let sn = f
                .str_param("search_analyzer")
                .map(str::to_string)
                .unwrap_or_else(|| if f.str_param("analyzer").is_some() { an.clone() } else { reg.default_search_analyzer_name().to_string() });
            let qn = f.str_param("search_quote_analyzer").map(str::to_string).unwrap_or_else(|| sn.clone());
            (Some(reg.index_analyzer(&an)), Some(an.clone()), reg.analyzer(&sn), reg.analyzer(&qn))
        } else {
            (None, None, None, None)
        };
        let normalizer = f.str_param("normalizer").and_then(|n| reg.normalizer(n));
        let date_format = f.str_param("format").and_then(|s| DateFormatter::new(s).ok()).unwrap_or_else(|| DateFormatter::new(DEFAULT_FORMAT).unwrap()).with_locale(f.str_param("locale"));
        let default_dv = (!f.ty.is_text() || (reg.strict_columnar() && matches!(f.ty, EsType::Text | EsType::MatchOnlyText))) && !matches!(f.ty, EsType::Binary | EsType::DenseVector);
        // ES 9.5 `index` defaults in a time series index
        // (`FieldMapper.Parameter.indexParam` / `useTimeSeriesDocValuesSkippers`):
        // with doc-values skippers (the time series default) keyword, ip,
        // boolean, numeric and unsigned_long fields are not indexed (doc
        // values and their skipper only; dimensions from
        // TIME_SERIES_DIMENSIONS_USE_SKIPPERS, every field from
        // TIME_SERIES_ALL_FIELDS_USE_SKIPPERS), and without skippers a
        // `counter` / `gauge` number still is not (`NumberFieldMapper`).
        // Queries on them run on doc values.
        let index_default = {
            use crate::version::IndexVersion;
            let tsdb = reg.index_mode_name() == "time_series";
            let dimension = f.bool_param("time_series_dimension", false);
            let ts_skippers = reg.use_dv_skipper && tsdb
                && reg.created_version.on_or_after(if dimension { IndexVersion::TIME_SERIES_DIMENSIONS_USE_SKIPPERS } else { IndexVersion::TIME_SERIES_ALL_FIELDS_USE_SKIPPERS });
            let numeric = matches!(f.ty, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float
                | EsType::HalfFloat | EsType::ScaledFloat | EsType::UnsignedLong);
            let skipper_typed = numeric || matches!(f.ty, EsType::Keyword | EsType::Ip | EsType::Boolean);
            if skipper_typed && ts_skippers {
                false
            } else if numeric && tsdb {
                !matches!(f.str_param("time_series_metric"), Some("counter" | "gauge"))
            } else {
                true
            }
        };
        ResolvedField {
            name: name.to_string(),
            ty: behavior,
            core_id,
            lat_id,
            lon_id,
            geo_id: if f.ty == EsType::GeoPoint { m.core_ids.get(&format!("{name}#geo")).copied() } else { None },
            index_analyzer,
            index_analyzer_name,
            search_analyzer,
            search_quote_analyzer: quote,
            normalizer_skip_store: normalizer.is_some()
                && f.bool_param("normalizer_skip_store_original_value", f.str_param("normalizer") == Some("lowercase") && !reg.is_custom_normalizer("lowercase")),
            normalizer,
            date_format,
            ignore_above: f.param("ignore_above").and_then(Value::as_u64).map(|v| v as usize).or(if matches!(f.ty, EsType::Keyword | EsType::Flattened | EsType::Wildcard) && !multi { reg.default_ignore_above } else { None }),
            null_value: f.param("null_value").cloned().filter(|v| !v.is_null()),
            coerce: f.bool_param("coerce", true),
            ignore_malformed: f.bool_param(
                "ignore_malformed",
                reg.default_ignore_malformed
                    && matches!(
                        f.ty,
                        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::UnsignedLong
                            | EsType::Date | EsType::DateNanos | EsType::Ip | EsType::Boolean | EsType::GeoPoint | EsType::GeoShape | EsType::Shape | EsType::Point
                            | EsType::AggregateMetricDouble | EsType::Histogram
                    ),
            ),
            scaling_factor: f.param("scaling_factor").and_then(Value::as_f64).unwrap_or(1.0),
            dims: f.param("dims").and_then(Value::as_u64).map(|v| v as usize),
            similarity: f.str_param("similarity").unwrap_or("cosine").to_string(),
            indexed: f.bool_param("index", index_default),
            doc_values: f.bool_param("doc_values", default_dv),
            copy_to: str_or_list(f.param("copy_to")),
            multi_fields: Vec::new(),
            is_multi_field: multi,
            alias_path: f.str_param("path").map(str::to_string),
            constant_value: f.param("value").cloned(),
            nested_path: nested.map(str::to_string),
            meta: f.param("meta").cloned(),
            declared,
            ext: crate::fieldtypes::ext_for(f, reg),
            aux: crate::fieldtypes::aux_ids(m, name),
            position_increment_gap: f.param("position_increment_gap").and_then(Value::as_u64).unwrap_or(100) as u32,
            // ES: text-like fields have norms by default, keyword does not.
            norms: f.bool_param("norms", f.ty != EsType::Keyword),
            discount_overlaps: reg.sims.resolve(f.str_param("similarity")).discount_overlaps,
            // A 9.x index that excludes source vectors stores sparse vectors by default.
            stored: f.bool_param("store", f.ty == EsType::SparseVector && reg.exclude_source_vectors),
            fielddata: f.ty.is_text() && f.bool_param("fielddata", false),
            time_series_dimension: f.bool_param("time_series_dimension", false),
            time_series_metric: f.str_param("time_series_metric").map(str::to_string),
            script: f.param("script").cloned().filter(|v| !v.is_null()),
            script_error_continue: f.str_param("on_script_error") == Some("continue"),
            single_value: dv_flag(f, reg, "multi_value", reg.dv_multi_value) == Some(false),
            index_declared: f.bool_param("index", !reg.index_disabled_by_default),
            required: dv_flag(f, reg, "nullability", reg.dv_nullability) == Some(false),
            skipper: uses_skipper(f, name, reg, f.bool_param("index", index_default), f.bool_param("doc_values", default_dv)),
        }
    }

    /// Look a field up, following `alias` fields.
    pub fn get(&self, name: &str) -> Option<&Arc<ResolvedField>> {
        if name == INDEX_MODE_FIELD {
            return self.index_mode_field.as_ref();
        }
        let Some(f) = self.fields.get(name) else {
            if !name.contains('.') {
                return None;
            }
            return self.synth.get_or_insert(name, || crate::fieldtypes::synth_flattened(self, name));
        };
        if f.ty == EsType::Alias {
            if let Some(p) = &f.alias_path {
                return self.fields.get(p);
            }
        }
        Some(f)
    }

    /// Field names matching a wildcard pattern (`*`, `user.*`, `*_name`).
    pub fn matching(&self, pattern: &str) -> Vec<String> {
        let mut v: Vec<String> = if pattern.contains('*') {
            self.fields.keys().filter(|k| simple_match(pattern, k)).cloned().collect()
        } else if self.fields.contains_key(pattern) {
            vec![pattern.to_string()]
        } else {
            Vec::new()
        };
        v.sort();
        v
    }

    /// Core schema for this mapping (reserved fields + every leaf).
    /// [`FieldLookup::core_schema`], computed once per lookup instance. Only
    /// for lookups that are no longer edited.
    pub fn core_schema_cached(&self) -> Arc<Vec<CoreField>> {
        self.core_schema_memo.0.get_or_init(|| Arc::new(self.core_schema())).clone()
    }

    pub fn core_schema(&self) -> Vec<CoreField> {
        let mut out = reserved_core_fields();
        // `_seq_no` is doc values only (no points; ranges skip blocks by
        // their min/max) in logsdb, time series and columnar indices, as
        // ES 9 indexes it there (`index.seq_no.index_options:
        // doc_values_only`, `IndexSettings.indexModeSupportsSeqNoDocValuesOnly`).
        if matches!(self.registry.index_mode.as_deref().map(str::to_ascii_lowercase).as_deref(), Some("logsdb" | "time_series" | "columnar" | "logsdb_columnar")) {
            if let Some(seq) = out.iter_mut().find(|f| f.id == SEQ_NO_FIELD) {
                seq.points = false;
            }
        }
        // A synthetic `_id`: neither indexed into sealed segments nor
        // stored; rebuilt from `_tsid`, `@timestamp` and the routing hash.
        if self.registry.synthetic_id {
            let id_of = |name: &str| self.fields.get(name).and_then(|f| f.core_id);
            if let (Some(tsid), Some(timestamp), Some(routing_hash)) = (id_of(crate::tsdb::TSID_FIELD), id_of("@timestamp"), id_of(crate::tsdb::ROUTING_HASH_FIELD)) {
                crate::tsdb::install_synthetic_id_codec();
                if let Some(id) = out.iter_mut().find(|f| f.id == ID_FIELD) {
                    id.stored = false;
                    id.synthetic_id = Some(rustic_core::synthetic_id::SyntheticId { tsid, timestamp, routing_hash });
                }
            }
        }
        // `_field_names` is only searched (ES indexes it, stores nothing);
        // with the stored-fields block in its own field, field 7 no longer
        // needs storing.
        if self.stored_block_id.is_some() {
            if let Some(f) = out.iter_mut().find(|f| f.id == FIELD_NAMES_FIELD) {
                f.stored = false;
            }
        }
        let mut leaf: Vec<CoreField> = self.fields.values().flat_map(|f| f.core_fields()).collect();
        if let Some(id) = self.doc_count_id {
            let mut dc = CoreField::new(id, DOC_COUNT_FIELD_NAME, CoreType::I64);
            dc.indexed = false;
            dc.fast = true;
            leaf.push(dc);
        }
        // Fetch reads only the reserved fields back from the segment store
        // (`_source`, `_id`, `_routing`, the `store: true` block, ...); user
        // field values live in postings and doc values, so keep them out of it.
        for f in &mut leaf {
            f.stored = false;
        }
        if let Some(id) = self.stored_block_id {
            // Named by its `core_ids` key, as every field is in a commit's
            // schema: a copy that takes the files' ids (file recovery,
            // snapshot restore: `IndexService::adopt_core_ids`) matches the
            // block to its own key. Files written as `_stored_fields` are
            // read as this key there.
            let mut block = CoreField::new(id, STORED_BLOCK_KEY, CoreType::Keyword);
            block.indexed = false;
            block.fast = false;
            leaf.push(block);
        }
        leaf.sort_by_key(|f| f.id);
        out.extend(leaf);
        out
    }

    pub fn is_object(&self, path: &str) -> bool {
        self.objects.contains_key(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_object_properties_parse_independently_of_json_member_order() {
        let registry = AnalysisRegistry::default();
        let late: Value = serde_json::from_str(r#"{"properties":{"o":{"properties":{"time":{"type":"long"},"time.min":{"type":"long"}},"subobjects":false}}}"#).unwrap();
        let early: Value = serde_json::from_str(r#"{"properties":{"o":{"subobjects":false,"properties":{"time":{"type":"long"},"time.min":{"type":"long"}}}}}"#).unwrap();
        let a = Mappings::parse(&late, &registry).unwrap();
        let b = Mappings::parse(&early, &registry).unwrap();
        assert_eq!(a.to_json(), b.to_json());
        assert_eq!(a.to_json()["properties"]["o"]["properties"]["time.min"]["type"], "long");
    }

    fn reg() -> AnalysisRegistry {
        AnalysisRegistry::default()
    }

    fn parse(v: Value) -> Result<Mappings> {
        let mut m = Mappings::parse(&v, &reg())?;
        m.assign_core_ids();
        Ok(m)
    }

    #[test]
    fn parse_all_p0_types_and_roundtrip() {
        let body = json!({
            "dynamic": "strict",
            "_source": {"excludes": ["secret"]},
            "_routing": {"required": false},
            "properties": {
                "title": {"type": "text", "analyzer": "english", "search_analyzer": "standard", "fields": {"raw": {"type": "keyword", "ignore_above": 100}}},
                "tag": {"type": "keyword", "normalizer": "lowercase"},
                "l": {"type": "long"}, "i": {"type": "integer"}, "s": {"type": "short"}, "b": {"type": "byte"},
                "d": {"type": "double"}, "f": {"type": "float"}, "hf": {"type": "half_float"},
                "sf": {"type": "scaled_float", "scaling_factor": 100},
                "ul": {"type": "unsigned_long"}, "ok": {"type": "boolean"},
                "ts": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                "ip": {"type": "ip"}, "loc": {"type": "geo_point"},
                "vec": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm", "index": true},
                "user": {"properties": {"name": {"type": "keyword"}, "age": {"type": "integer"}}},
                "comments": {"type": "nested", "properties": {"text": {"type": "text"}}},
                "a.b": {"type": "keyword"}
            }
        });
        let m = parse(body.clone()).unwrap();
        let out = m.to_json();
        assert_eq!(out["dynamic"], json!("strict"));
        assert_eq!(out["properties"]["title"]["fields"]["raw"]["ignore_above"], json!(100));
        assert_eq!(out["properties"]["user"]["properties"]["age"]["type"], json!("integer"));
        assert!(out["properties"]["user"].get("type").is_none());
        assert_eq!(out["properties"]["comments"]["type"], json!("nested"));
        assert_eq!(out["properties"]["a"]["properties"]["b"]["type"], json!("keyword"));
        assert_eq!(Mappings::parse(&out, &reg()).unwrap().to_json(), out);
        let lk = FieldLookup::build(&m, &reg());
        assert_eq!(lk.get("title").unwrap().ty, EsType::Text);
        assert_eq!(lk.get("title.raw").unwrap().ignore_above, Some(100));
        assert!(lk.get("title.raw").unwrap().is_multi_field);
        assert!(lk.get("loc").unwrap().geo_id.is_some() && lk.get("loc").unwrap().lat_id.is_none());
        assert_eq!(lk.get("vec").unwrap().dims, Some(3));
        assert_eq!(lk.get("comments.text").unwrap().nested_path.as_deref(), Some("comments"));
        let schema = lk.core_schema();
        assert!(schema.iter().any(|f| f.name == "loc#geo") && !schema.iter().any(|f| f.name == "loc#lat"));
        assert!(schema.iter().any(|f| f.name == "vec" && f.field_type == CoreType::Vector { dim: 3 }));
        let ids: std::collections::HashSet<u16> = schema.iter().map(|f| f.id).collect();
        assert_eq!(ids.len(), schema.len(), "core ids are unique");
    }

    #[test]
    fn parse_errors() {
        let e = parse(json!({"properties": {"x": {"type": "bogus"}}})).unwrap_err();
        assert_eq!(e.kind, "mapper_parsing_exception");
        assert_eq!(e.reason, "Failed to parse mapping: The mapper type [bogus] declared on field [x] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.");
        let e = parse(json!({"properties": {"x": {"type": "keyword", "foo": 1}}})).unwrap_err();
        assert!(e.reason.contains("unknown parameter [foo] on mapper [x] of type [keyword]"), "{}", e.reason);
        let e = parse(json!({"properties": {"x": {"type": "text", "analyzer": "nope"}}})).unwrap_err();
        assert!(e.reason.contains("analyzer [nope] has not been configured in mappings"));
        let e = parse(json!({"properties": {"x": {"type": "scaled_float"}}})).unwrap_err();
        assert!(e.reason.contains("scaling_factor"));
        let e = parse(json!({"foo": "bar"})).unwrap_err();
        assert!(e.reason.contains("Root mapping definition has unsupported parameters"));
        let e = parse(json!({"properties": {"v": {"type": "dense_vector", "dims": 5000}}})).unwrap_err();
        assert!(e.reason.contains("[1, 4096]"));
        // ES MAX_DIMS_COUNT_BIT: bit vectors take up to 4096 bytes.
        parse(json!({"properties": {"v": {"type": "dense_vector", "element_type": "bit", "dims": 4104}}})).unwrap();
        parse(json!({"properties": {"v": {"type": "dense_vector", "element_type": "bit", "dims": 32768, "index": false}}})).unwrap();
        let e = parse(json!({"properties": {"v": {"type": "dense_vector", "element_type": "bit", "dims": 32776}}})).unwrap_err();
        assert!(e.reason.contains("[8, 32768] but was [32776]"), "{}", e.reason);
        let e = parse(json!({"properties": {"v": {"type": "dense_vector", "element_type": "byte", "dims": 4104}}})).unwrap_err();
        assert!(e.reason.contains("[1, 4096]"), "{}", e.reason);
    }

    #[test]
    fn merge_rules() {
        let mut m = parse(json!({"properties": {"a": {"type": "long"}, "t": {"type": "text", "analyzer": "standard"}, "k": {"type": "keyword", "ignore_above": 10}}})).unwrap();
        // additive change ok
        m.merge(&parse(json!({"properties": {"b": {"type": "keyword"}, "t": {"type": "text", "analyzer": "standard", "fields": {"kw": {"type": "keyword"}}}}})).unwrap()).unwrap();
        assert!(m.core_ids.contains_key("b"));
        assert!(m.core_ids.contains_key("t.kw"));
        // updatable parameter ok
        m.merge(&parse(json!({"properties": {"k": {"type": "keyword", "ignore_above": 20}}})).unwrap()).unwrap();
        assert_eq!(m.to_json()["properties"]["k"]["ignore_above"], json!(20));
        // type change rejected
        let e = m.merge(&parse(json!({"properties": {"a": {"type": "keyword"}}})).unwrap()).unwrap_err();
        assert_eq!(e.kind, "illegal_argument_exception");
        assert_eq!(e.reason, "mapper [a] cannot be changed from type [long] to [keyword]");
        // non-updatable parameter rejected
        let e = m.merge(&parse(json!({"properties": {"t": {"type": "text", "analyzer": "english"}}})).unwrap()).unwrap_err();
        assert!(e.reason.contains("Cannot update parameter [analyzer] from [standard] to [english]"), "{}", e.reason);
        // object vs field
        let e = m.merge(&parse(json!({"properties": {"a": {"properties": {"x": {"type": "long"}}}}})).unwrap()).unwrap_err();
        assert!(e.reason.contains("can't merge a non object mapping [a]"));
        // ids stable across merges
        let id_a = m.core_ids["a"];
        m.merge(&parse(json!({"properties": {"z": {"type": "long"}}})).unwrap()).unwrap();
        assert_eq!(m.core_ids["a"], id_a);
    }

    /// Re-putting an identical mapping is a no-op, whatever defaults are
    /// spelled out or materialized.
    #[test]
    fn identical_reput_is_noop() {
        let variants = [
            json!({"type": "dense_vector", "dims": 3}),
            json!({"type": "dense_vector", "dims": 3, "element_type": "float", "similarity": "cosine", "index": true}),
            json!({"type": "dense_vector", "dims": 4, "element_type": "byte", "index": true, "similarity": "l2_norm"}),
            json!({"type": "dense_vector", "dims": 8, "element_type": "bit"}),
            json!({"type": "dense_vector", "dims": 3, "index": false}),
            json!({"type": "dense_vector", "dims": 3, "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100}}),
            json!({"type": "dense_vector", "dims": 3, "index_options": {"type": "int8_hnsw"}}),
            json!({"type": "dense_vector", "dims": 64, "index_options": {"type": "bbq_hnsw"}}),
            json!({"type": "dense_vector", "dims": 3, "index_options": {"type": "flat"}}),
            json!({"type": "keyword", "store": false, "index": true, "doc_values": true, "ignore_above": 2147483647}),
            json!({"type": "text", "norms": true, "index_options": "positions", "analyzer": "standard", "fields": {"k": {"type": "keyword", "ignore_above": 256}}}),
            json!({"type": "date", "format": "strict_date_optional_time||epoch_millis", "locale": "en"}),
            json!({"type": "scaled_float", "scaling_factor": 100}),
            json!({"type": "long", "copy_to": "other", "meta": {"a": "b"}, "null_value": 5}),
            json!({"type": "join", "relations": {"p": ["c1", "c2"]}}),
            json!({"type": "completion"}),
            json!({"type": "search_as_you_type"}),
            json!({"type": "geo_shape", "orientation": "cw"}),
        ];
        for v in variants {
            let body = json!({"properties": {"f": v, "other": {"type": "keyword"}}});
            let mut m = parse(body.clone()).unwrap();
            m.merge(&parse(body.clone()).unwrap()).unwrap_or_else(|e| panic!("{body}: {}", e.reason));
            // A GET-mapping round trip re-puts cleanly too.
            let out = m.to_json();
            m.merge(&Mappings::parse_update(&out, &reg()).unwrap()).unwrap_or_else(|e| panic!("{out}: {}", e.reason));
        }
    }

    #[test]
    fn dynamic_templates_and_defaults() {
        let m = parse(json!({
            "dynamic_templates": [
                {"strings_as_kw": {"match_mapping_type": "string", "match": "*_id", "mapping": {"type": "keyword"}}},
                {"longs": {"match_mapping_type": "long", "mapping": {"type": "integer"}}},
                {"by_path": {"path_match": "meta.*", "mapping": {"type": "keyword", "doc_values": false}}}
            ]
        }))
        .unwrap();
        let r = reg();
        match m.dynamic_mapper("user_id", "user_id", "string", &r).unwrap().unwrap() {
            Mapper::Field(f) => assert_eq!(f.ty, EsType::Keyword),
            _ => panic!(),
        }
        match m.dynamic_mapper("title", "title", "string", &r).unwrap().unwrap() {
            Mapper::Field(f) => {
                assert_eq!(f.ty, EsType::Text);
                assert_eq!(f.fields["keyword"].param("ignore_above"), Some(&json!(256)));
            }
            _ => panic!(),
        }
        match m.dynamic_mapper("n", "n", "long", &r).unwrap().unwrap() {
            Mapper::Field(f) => assert_eq!(f.ty, EsType::Integer),
            _ => panic!(),
        }
        match m.dynamic_mapper("x", "meta.x", "double", &r).unwrap().unwrap() {
            Mapper::Field(f) => assert_eq!(f.ty, EsType::Keyword),
            _ => panic!(),
        }
        match m.dynamic_mapper("x", "x", "double", &r).unwrap().unwrap() {
            Mapper::Field(f) => assert_eq!(f.ty, EsType::Float),
            _ => panic!(),
        }
        assert_eq!(m.to_json()["dynamic_templates"][0]["strings_as_kw"]["match"], json!("*_id"));
    }

    #[test]
    fn persisted_roundtrip_keeps_ids() {
        let m = parse(json!({"properties": {"a": {"type": "long"}, "g": {"type": "geo_point"}}})).unwrap();
        let p = m.to_persisted();
        let m2 = Mappings::from_persisted(&p, &reg()).unwrap();
        assert_eq!(m.core_ids, m2.core_ids);
        assert_eq!(m.next_core_id, m2.next_core_id);
    }

    #[test]
    fn wildcard_simple_match() {
        assert!(simple_match("*_id", "user_id"));
        assert!(simple_match("a*b*c", "aXXbYYc"));
        assert!(!simple_match("a*b", "ac"));
        assert!(simple_match("*", "anything"));
        assert!(simple_match("meta.*", "meta.x"));
    }
}
