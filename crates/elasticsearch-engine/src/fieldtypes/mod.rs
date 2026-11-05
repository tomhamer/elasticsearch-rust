//! The specialised Elasticsearch field types and mapping parameters that
//! sit on top of the basic leaf types in [`crate::mapping`]:
//!
//! | ES type | core backing |
//! |---|---|
//! | `ip` | 16-byte sortable hex keyword (+ CIDR ranges) |
//! | `flattened` | root keyword values + `#keyed` `key\0value` keyword |
//! | `join` | relation keyword + `#parent` keyword; queries/aggs in [`join`] |
//! | `alias` | resolved by [`crate::mapping::FieldLookup::get`] |
//! | `date_nanos` | epoch-nanos longs (aggs see millis) |
//! | `scaled_float` / `unsigned_long` | saturating scaled doubles / u64 |
//! | `*_range` | `(min,max)` sortable-key doc values + [`FilterQuery`] relations |
//! | `match_only_text` | text, constant scores |
//! | `constant_keyword` | mapping value; queries rewritten to match_all/none |
//! | `wildcard` | keyword doc values + `#ngram` 3-gram index, verified |
//! | `completion` | doc values → per-segment FST ([`suggest`]) |
//! | `search_as_you_type` | text + `._2gram`/`._3gram`/`._index_prefix` sub-fields |
//! | `token_count` / `murmur3` / `_size` | longs computed at index time |
//! | `binary` | base64 keyword doc values |
//! | `geo_shape` / `shape` / `point` | encoded geometry doc values + relations |
//! | `histogram` / `tdigest` / `exponential_histogram` | encoded `(value,count)` doc values |
//! | `aggregate_metric_double` | `#min/#max/#sum/#value_count` doubles |
//! | `version` | original keyword + `#sort` semver-sortable keyword |
//! | `sparse_vector` / `rank_features` / `rank_feature` | `token\0weight` doc values / double |
//! | `percolator` | stored query JSON, matched against an in-memory index |
//! | `annotated_text` | text with annotation tokens appended |
//! | `semantic_text` | text plus its inference companion |
//! | `rank_vectors` | JSON vectors in doc values |
//! | `passthrough` | object whose leaves are also reachable unprefixed |
//!
//! [`FilterQuery`]: rustic_core::fieldtypes::filter::FilterQuery

/// Joins the words of a multi-word token (shingles, annotations) so the
/// core whitespace re-split of pre-analyzed text keeps it whole.
pub const TOKEN_JOIN: &str = "\u{1f}";

pub mod aggs;
pub mod exphist;
pub mod pattern_text;
pub mod geo;
pub(crate) mod geo_shape_query;
pub mod index;
pub mod join;
pub mod limits;
pub mod mvt_feature;
pub mod params;
pub mod percolate;
pub mod query;
pub mod suggest;

#[cfg(test)]
mod tests;

use crate::analysis::AnalysisRegistry;
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, FieldMapping, Mapper, Mappings, ResolvedField};
use rustic_core::analysis::{Analyzer, Token, TokenFilter};
use rustic_core::fieldtypes::range::RangeKind;
use rustic_core::types::{Field as CoreField, FieldType as CoreType};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Type-specific configuration of a resolved field.
#[derive(Clone, Default)]
pub struct FieldExt {
    /// The field's mapping parameters.
    pub params: Arc<Map<String, Value>>,
    /// Set on the synthesized `<flattened>.<key>` sub-fields.
    pub flattened_key: Option<String>,
    /// Analyzer used at index time by `token_count` / `annotated_text`
    /// / `completion`.
    pub analyzer: Option<Arc<Analyzer>>,
}

impl FieldExt {
    pub fn str(&self, k: &str) -> Option<&str> {
        self.params.get(k).and_then(Value::as_str)
    }
    pub fn bool(&self, k: &str, default: bool) -> bool {
        match self.params.get(k) {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => s == "true",
            _ => default,
        }
    }
    pub fn u64(&self, k: &str) -> Option<u64> {
        match self.params.get(k) {
            Some(Value::Number(n)) => n.as_u64(),
            Some(Value::String(s)) => s.parse().ok(),
            _ => None,
        }
    }
    pub fn f64(&self, k: &str) -> Option<f64> {
        match self.params.get(k) {
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::String(s)) => s.parse().ok(),
            _ => None,
        }
    }
}

/// The leaf type whose query/agg/sort behaviour a declared type shares.
pub fn behavior(ty: EsType) -> EsType {
    match ty {
        EsType::AnnotatedText | EsType::SearchAsYouType | EsType::SemanticText | EsType::Semantic => EsType::Text,
        EsType::TokenCount => EsType::Integer,
        EsType::Murmur3 => EsType::Long,
        EsType::Version => EsType::Keyword,
        EsType::CountedKeyword => EsType::Keyword,
        EsType::PatternText => EsType::MatchOnlyText,
        t => t,
    }
}

pub fn range_kind(ty: EsType) -> Option<RangeKind> {
    Some(match ty {
        EsType::IntegerRange => RangeKind::Integer,
        EsType::LongRange => RangeKind::Long,
        EsType::FloatRange => RangeKind::Float,
        EsType::DoubleRange => RangeKind::Double,
        EsType::DateRange => RangeKind::Date,
        EsType::IpRange => RangeKind::Ip,
        _ => return None,
    })
}

pub fn is_histogram(ty: EsType) -> bool {
    matches!(ty, EsType::Histogram | EsType::TDigest | EsType::ExponentialHistogram)
}

pub fn is_shape(ty: EsType) -> bool {
    matches!(ty, EsType::GeoShape | EsType::Shape | EsType::Point)
}

/// Mapping parameters of the types implemented here.
pub fn allowed_params(ty: EsType) -> &'static [&'static str] {
    const RANGE: &[&str] = &["coerce", "index", "doc_values", "store", "boost", "meta", "format", "locale", "copy_to"];
    match ty {
        EsType::IntegerRange | EsType::LongRange | EsType::FloatRange | EsType::DoubleRange | EsType::DateRange | EsType::IpRange => RANGE,
        EsType::Histogram => &["ignore_malformed", "coerce", "meta", "time_series_metric"],
        EsType::TDigest => &["digest_type", "compression", "ignore_malformed", "meta", "time_series_metric"],
        EsType::ExponentialHistogram => &["ignore_malformed", "coerce", "meta", "time_series_metric", "copy_to"],
        EsType::AnnotatedText => &[
            "analyzer", "search_analyzer", "search_quote_analyzer", "index", "index_options", "norms", "position_increment_gap", "store",
            "similarity", "term_vector", "fields", "copy_to", "meta",
        ],
        EsType::SearchAsYouType => &[
            "analyzer", "search_analyzer", "search_quote_analyzer", "index", "index_options", "norms", "max_shingle_size", "store", "similarity",
            "term_vector", "copy_to", "meta", "doc_values",
        ],
        EsType::SemanticText | EsType::Semantic => &["inference_id", "search_inference_id", "index_options", "chunking_settings", "meta", "model_settings"],
        EsType::TokenCount => &["analyzer", "enable_position_increments", "doc_values", "index", "null_value", "store", "boost", "meta"],
        EsType::Murmur3 => &["doc_values", "store", "meta", "index"],
        EsType::Version => &["meta", "fields", "copy_to"],
        EsType::CountedKeyword => &["index", "meta"],
        EsType::Completion => &[
            "analyzer", "search_analyzer", "preserve_separators", "preserve_position_increments", "max_input_length", "contexts", "meta",
        ],
        EsType::Join => &["relations", "eager_global_ordinals", "meta"],
        EsType::Percolator => &["meta"],
        EsType::GeoShape => &["orientation", "ignore_malformed", "ignore_z_value", "coerce", "index", "doc_values", "store", "meta", "strategy", "tree", "precision"],
        EsType::Shape => &["orientation", "ignore_malformed", "ignore_z_value", "coerce", "index", "doc_values", "meta"],
        EsType::Point => &["ignore_malformed", "ignore_z_value", "null_value", "index", "doc_values", "store", "meta"],
        EsType::AggregateMetricDouble => &["metrics", "default_metric", "time_series_metric", "meta", "ignore_malformed"],
        EsType::SparseVector => &["store", "index_options", "meta"],
        EsType::RankFeature => &["positive_score_impact", "meta"],
        EsType::RankFeatures => &["positive_score_impact", "meta"],
        EsType::RankVectors => &["dims", "element_type", "meta"],
        _ => &[],
    }
}

fn mp(reason: String) -> EsError {
    EsError::mapper_parsing(reason)
}

fn ia(reason: String) -> EsError {
    params::iae(reason)
}

/// Extra validation of a field mapping (types here + shared parameters).
pub fn validate(name: &str, f: &FieldMapping, registry: &AnalysisRegistry) -> Result<()> {
    // `meta`: at most 5 entries, keys <= 20 chars, string values <= 50 chars.
    if let Some(meta) = f.param("meta") {
        let m = meta.as_object().ok_or_else(|| mp(format!("[meta] must be an object, got {} for field [{name}]", params::jclass_value(meta))))?;
        if m.len() > 5 {
            return Err(mp(format!("[meta] can't have more than 5 entries, but got {} on field [{name}]", m.len())));
        }
        for (k, v) in m {
            if k.chars().count() > 20 {
                return Err(mp(format!("[meta] keys can't be longer than 20 chars, but got [{k}] for field [{name}]")));
            }
            match v {
                Value::String(s) if s.chars().count() > 50 => {
                    return Err(mp(format!("[meta] values can't be longer than 50 chars, but got [{s}] for field [{name}]")))
                }
                Value::String(_) => {}
                Value::Null => return Err(mp(format!("[meta] values can't be null (field [{name}])"))),
                _ => return Err(mp(format!("[meta] values can only be strings, but got {} for field [{name}]", params::jclass_value(v)))),
            }
        }
    }
    if let Some(v) = f.param("position_increment_gap") {
        if v.as_i64().map(|x| x < 0).unwrap_or(true) {
            return Err(mp(format!("[position_increment_gap] must be positive, got [{v}]")));
        }
    }
    if let Some(v) = f.str_param("term_vector") {
        if !matches!(
            v,
            "no" | "yes" | "with_positions" | "with_offsets" | "with_positions_offsets" | "with_positions_payloads" | "with_positions_offsets_payloads"
        ) {
            return Err(mp(format!("Unknown value [{v}] for field [term_vector] - accepted values are [no, yes, with_positions, with_offsets, with_positions_offsets, with_positions_payloads, with_positions_offsets_payloads]")));
        }
    }
    if let Some(v) = f.str_param("on_script_error") {
        if !matches!(v, "fail" | "continue") {
            return Err(mp(format!("Unknown value [{v}] for field [on_script_error] - accepted values are [fail, continue]")));
        }
        if f.param("script").is_none() {
            return Err(ia("Field [on_script_error] requires field [script] to be configured".to_string()));
        }
    }
    if f.param("script").is_some() {
        for p in ["null_value", "ignore_malformed", "copy_to"] {
            if f.param(p).is_some() {
                return Err(ia(format!("Field [{p}] cannot be set in conjunction with field [script]")));
            }
        }
    }
    if f.ty == EsType::PatternText {
        if let Some(io) = f.param("index_options") {
            let v = io.as_str().map(str::to_string).unwrap_or_else(|| io.to_string());
            if !matches!(v.as_str(), "positions" | "docs") {
                return Err(EsError::mapper_parsing(format!("Unknown value [{v}] for field [index_options] - accepted values are [positions, docs]")));
            }
        }
        if let Some(a) = f.param("analyzer") {
            let v = a.as_str().map(str::to_string).unwrap_or_else(|| a.to_string());
            if !matches!(v.as_str(), "standard" | "delimiter") {
                let leaf = name.rsplit('.').next().unwrap_or(name);
                return Err(ia(format!("unsupported analyzer [{v}] for field [{leaf}], supported analyzers are [standard, log]")));
            }
        }
    }
    if f.ty == EsType::TDigest {
        if let Some(c) = f.param("compression") {
            let c = c.as_f64().or_else(|| c.as_str().and_then(|s| s.parse().ok())).unwrap_or(f64::NAN);
            if !(c > 0.0 && c <= 10000.0) {
                return Err(ia(format!("compression must be a positive integer between 1 and 10000.0 was [{}]", crate::values::java_double(c))));
            }
        }
        if let Some(d) = f.str_param("digest_type") {
            if !matches!(d.to_ascii_lowercase().as_str(), "default" | "high_accuracy") {
                return Err(ia(format!("Unknown value [{d}] for field [digest_type] - accepted values are [DEFAULT, HIGH_ACCURACY]")));
            }
        }
    }
    if let Some(m) = f.str_param("time_series_metric") {
        let ok: &[&str] = match f.ty {
            EsType::Histogram | EsType::ExponentialHistogram | EsType::TDigest => &["histogram"],
            EsType::AggregateMetricDouble => &["gauge"],
            EsType::GeoPoint => &["position"],
            _ => &["gauge", "counter"],
        };
        if !ok.contains(&m) {
            return Err(mp(format!("Unknown value [{m}] for field [time_series_metric] - accepted values are [{}]", ok.join(", "))));
        }
    }
    if f.bool_param("time_series_dimension", false) {
        if !f.bool_param("index", true) || !f.bool_param("doc_values", true) {
            return Err(ia("Field [time_series_dimension] requires that [index] and [doc_values] are true".to_string()));
        }
        if f.param("ignore_above").is_some() && f.ty == EsType::Keyword {
            // ES allows it; nothing to do.
        }
    }
    if let Some(k) = f.str_param("synthetic_source_keep") {
        if !matches!(k, "none" | "arrays" | "all") {
            return Err(mp(format!("Unknown synthetic_source_keep value [{k}], accepted values are [none,arrays,all]")));
        }
    }
    match f.ty {
        EsType::Alias => {
            if f.str_param("path").map(str::is_empty).unwrap_or(true) {
                return Err(mp(format!("The [path] property must be specified for field [{name}].")));
            }
        }
        EsType::ConstantKeyword => {
            if let Some(v) = f.param("value") {
                if !(v.is_string() || v.is_number() || v.is_boolean() || v.is_null()) {
                    return Err(mp(format!("Property [value] on field [{name}] must be a number or a string, but got [{}]", params::jstr(v))));
                }
            }
        }
        EsType::Flattened => {
            if let Some(d) = f.param("depth_limit") {
                if d.as_i64().map(|x| x < 0).unwrap_or(true) {
                    return Err(ia(format!("[depth_limit] must be positive, got [{d}]")));
                }
            }
        }
        EsType::ScaledFloat => {
            let sf = match f.param("scaling_factor") {
                Some(Value::String(t)) => Some(t.trim().parse::<f64>().map_err(|_| {
                    let m = format!("For input string: \"{t}\"");
                    EsError::mapper_parsing(m.clone()).caused_by(EsError::new(400, "number_format_exception", m))
                })?),
                other => other.and_then(Value::as_f64),
            };
            match sf {
                None => return Err(ia("Field [scaling_factor] is required".to_string())),
                Some(x) if !(x.is_finite() && x > 0.0) => {
                    return Err(ia(format!("[scaling_factor] must be a positive number, got [{}]", params::jdouble(x))));
                }
                _ => {}
            }
        }
        EsType::SearchAsYouType => {
            if let Some(m) = f.param("max_shingle_size") {
                let n = m.as_i64().unwrap_or(-1);
                if !(2..=4).contains(&n) {
                    return Err(mp(format!("[max_shingle_size] must be at least [2] and at most [4], got [{m}]")));
                }
            }
        }
        EsType::TokenCount => {
            if f.str_param("analyzer").is_none() {
                return Err(mp(format!("Analyzer must be set for field [{name}] but wasn't.")));
            }
        }
        EsType::Join => {
            let empty = Map::new();
            let rel = match f.param("relations") {
                None => &empty,
                Some(v) => v.as_object().ok_or_else(|| mp(format!("[relations] must be an object for field [{name}]")))?,
            };
            let mut parents_of: HashMap<String, String> = HashMap::new();
            for (p, c) in rel {
                let children: Vec<String> = match c {
                    Value::String(s) => vec![s.clone()],
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                    _ => return Err(mp(format!("[relations] values must be strings or arrays of strings for field [{name}]"))),
                };
                for ch in children {
                    if ch == *p {
                        return Err(mp(format!("cannot have the same name [{p}] for parent and child")));
                    }
                    if parents_of.insert(ch.clone(), p.clone()).is_some() {
                        return Err(mp(format!("[{ch}] cannot have multiple parents")));
                    }
                }
            }
        }
        EsType::Completion => {
            if let Some(m) = f.param("max_input_length") {
                if m.as_i64().map(|x| x <= 0).unwrap_or(true) {
                    return Err(ia(format!("[max_input_length] must be > 0 but was [{m}]")));
                }
            }
            if let Some(ctxs) = f.param("contexts") {
                let arr = match ctxs {
                    Value::Array(a) => a.clone(),
                    Value::Object(_) => vec![ctxs.clone()],
                    _ => return Err(mp("[contexts] must be an array or an object".into())),
                };
                for c in arr {
                    let name_ok = c.get("name").and_then(Value::as_str).is_some();
                    let ty = c.get("type").and_then(Value::as_str).unwrap_or("");
                    if !name_ok {
                        return Err(mp("contexts [name] is required".into()));
                    }
                    if !matches!(ty.to_ascii_lowercase().as_str(), "category" | "geo") {
                        return Err(mp(format!("unknown context type [{ty}]")));
                    }
                }
            }
            for k in ["analyzer", "search_analyzer"] {
                if let Some(a) = f.str_param(k) {
                    if !registry.has_analyzer(a) {
                        return Err(mp(format!("analyzer [{a}] has not been configured in mappings")));
                    }
                }
            }
        }
        EsType::AggregateMetricDouble => {
            let metrics: Vec<String> = f
                .param("metrics")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            if metrics.is_empty() {
                return Err(ia(format!("Property [metrics] is required for field [{name}].")));
            }
            for m in &metrics {
                if !matches!(m.as_str(), "min" | "max" | "sum" | "value_count") {
                    let e = EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.aggregatemetric.mapper.AggregateMetricDoubleFieldMapper.Metric.{m}"));
                    return Err(params::iae_caused(format!("Metric [{m}] is not supported."), e));
                }
            }
            // ES 9.4+: `default_metric` is optional, deprecated and no
            // longer has to be one of the metrics.
            if let Some(d) = f.str_param("default_metric") {
                if !matches!(d, "min" | "max" | "sum" | "value_count") {
                    let e = EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.aggregatemetric.mapper.AggregateMetricDoubleFieldMapper.Metric.{d}"));
                    return Err(params::iae_caused(format!("Metric [{d}] is not supported."), e));
                }
            }
        }
        EsType::RankVectors => {
            if let Some(d) = f.param("dims") {
                let n = d.as_u64().unwrap_or(0);
                if n == 0 || n > 4096 {
                    return Err(mp(format!("The number of dimensions should be in the range [1, 4096] but was [{d}]")));
                }
            }
            if let Some(e) = f.str_param("element_type") {
                if !matches!(e, "float" | "byte" | "bit" | "bfloat16") {
                    return Err(mp(format!("invalid element_type [{e}]; available types are [byte, bfloat16, bit, float]")));
                }
            }
        }
        EsType::GeoShape | EsType::Shape => {
            if let Some(o) = f.str_param("orientation") {
                if !matches!(
                    o.to_ascii_lowercase().as_str(),
                    "right" | "ccw" | "counterclockwise" | "left" | "cw" | "clockwise"
                ) {
                    return Err(ia(format!("Unknown orientation [{o}]")));
                }
            }
        }
        EsType::SemanticText | EsType::Semantic => {
            // Any inference id is accepted at mapping time.
        }
        t if range_kind(t).is_some() => {
            if t != EsType::DateRange && f.param("format").is_some() {
                return Err(ia(format!("field [{name}] of type [range] should not define a dateTimeFormatter unless it is a DATE type")));
            }
        }
        _ => {}
    }
    Ok(())
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `null_value` must be a valid value of the field's type.
#[allow(dead_code)]
fn validate_null_value(name: &str, ty: EsType, nv: &Value) -> Result<()> {
    if nv.is_null() {
        return Ok(());
    }
    let bad = |what: &str| mp(format!("Failed to parse mapping: Error parsing [null_value] on field [{name}]: {what}"));
    let s = match nv {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match behavior(ty) {
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
            if s.trim().parse::<i64>().is_err() && s.trim().parse::<f64>().map(|x| x.fract() != 0.0).unwrap_or(true) {
                return Err(bad(&format!("For input string: \"{s}\"")));
            }
        }
        EsType::UnsignedLong => {
            if s.trim().parse::<u64>().is_err() {
                return Err(bad(&format!("Value \"{s}\" is not a valid unsigned long")));
            }
        }
        EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
            if s.trim().parse::<f64>().is_err() {
                return Err(bad(&format!("For input string: \"{s}\"")));
            }
        }
        EsType::Boolean => {
            if !matches!(nv, Value::Bool(_)) && !matches!(s.as_str(), "true" | "false" | "") {
                return Err(bad(&format!("Failed to parse value [{s}] as only [true] or [false] are allowed.")));
            }
        }
        EsType::Ip => {
            if rustic_core::fieldtypes::ip::parse(&s).is_none() {
                return Err(bad(&format!("'{s}' is not an IP string literal.")));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Core field keys a declared field needs (`None` = the default: one key,
/// the field path). Auxiliary keys are `<path>#<suffix>`.
pub fn core_keys(path: &str, f: &FieldMapping) -> Option<Vec<String>> {
    let aux = |s: &[&str]| -> Vec<String> {
        let mut v = vec![path.to_string()];
        v.extend(s.iter().map(|x| format!("{path}#{x}")));
        v
    };
    Some(match f.ty {
        EsType::Flattened => aux(&["keyed"]),
        EsType::Wildcard => aux(&["ngram"]),
        EsType::Version => aux(&["sort"]),
        EsType::Text | EsType::MatchOnlyText if f.bool_param("doc_values", false) => aux(&["dv"]),
        // `<field>.template_id` (unless templating is disabled).
        EsType::PatternText if !f.bool_param("disable_templating", false) => vec![path.to_string(), format!("{path}.template_id")],
        EsType::Join => aux(&["parent"]),
        EsType::Percolator => aux(&["terms"]),
        EsType::SemanticText | EsType::Semantic => aux(&["inference"]),
        EsType::AggregateMetricDouble => ["min", "max", "sum", "value_count", "avg"].iter().map(|x| format!("{path}#{x}")).collect(),
        EsType::SearchAsYouType => {
            let n = f.param("max_shingle_size").and_then(Value::as_u64).unwrap_or(3);
            let mut v = vec![path.to_string()];
            for k in 2..=n {
                v.push(format!("{path}._{k}gram"));
            }
            v.push(format!("{path}._index_prefix"));
            v
        }
        EsType::Alias | EsType::Object | EsType::Nested => vec![],
        _ => return None,
    })
}

/// Auxiliary core ids of a field (`suffix → id`).
pub fn aux_ids(m: &Mappings, name: &str) -> Vec<(String, u16)> {
    let prefix = format!("{name}#");
    m.core_ids.range(prefix.clone()..).take_while(|(k, _)| k.starts_with(&prefix)).map(|(k, id)| (k[prefix.len()..].to_string(), *id)).collect()
}

/// Type-specific config for a resolved field.
pub fn ext_for(f: &FieldMapping, registry: &AnalysisRegistry) -> Option<Arc<FieldExt>> {
    let needs = matches!(
        f.ty,
        EsType::TokenCount | EsType::AnnotatedText | EsType::Completion | EsType::Flattened
    ) || !matches!(f.ty, EsType::Text | EsType::Keyword | EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::Boolean | EsType::Date | EsType::GeoPoint | EsType::Object | EsType::Nested);
    if !needs {
        return None;
    }
    let analyzer = match f.ty {
        EsType::TokenCount | EsType::AnnotatedText => registry.analyzer(f.str_param("analyzer").unwrap_or("standard")),
        EsType::Completion => registry.analyzer(f.str_param("analyzer").unwrap_or("simple")),
        _ => None,
    };
    Some(Arc::new(FieldExt { params: Arc::new(f.params.clone()), flattened_key: None, analyzer }))
}

fn mk(id: u16, name: &str, ty: CoreType, indexed: bool, fast: bool) -> CoreField {
    let mut f = CoreField::new(id, name, ty);
    f.indexed = indexed;
    f.fast = fast;
    f
}

/// Core fields for the types implemented here (`None` = default handling).
pub fn core_fields(rf: &ResolvedField) -> Option<Vec<CoreField>> {
    let aux = |s: &str| rf.aux_id(s);
    let main = rf.core_id;
    let n = &rf.name;
    let mut out = Vec::new();
    let kw_dv = |id: u16| mk(id, n, CoreType::Keyword, false, true);
    match rf.declared {
        EsType::Flattened => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Keyword, rf.indexed, rf.doc_values));
            }
            if let Some(id) = aux("keyed") {
                out.push(mk(id, &format!("{n}#keyed"), CoreType::Keyword, rf.indexed, rf.doc_values));
            }
        }
        EsType::Wildcard => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Keyword, true, true));
            }
            if let Some(id) = aux("ngram") {
                out.push(mk(id, &format!("{n}#ngram"), CoreType::Keyword, true, false));
            }
        }
        EsType::Version => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Keyword, rf.indexed, rf.doc_values));
            }
            if let Some(id) = aux("sort") {
                out.push(mk(id, &format!("{n}#sort"), CoreType::Keyword, true, true));
            }
        }
        EsType::Join => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Keyword, true, true));
            }
            if let Some(id) = aux("parent") {
                out.push(mk(id, &format!("{n}#parent"), CoreType::Keyword, true, true));
            }
        }
        EsType::AggregateMetricDouble => {
            for m in ["min", "max", "sum", "value_count", "avg"] {
                if let Some(id) = aux(m) {
                    out.push(mk(id, &format!("{n}#{m}"), CoreType::F64, false, true));
                }
            }
        }
        EsType::DateNanos => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::I64, rf.indexed, true));
            }
        }
        EsType::TokenCount | EsType::Murmur3 => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::I64, rf.declared == EsType::TokenCount && rf.indexed, true));
            }
        }
        EsType::RankFeature => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::F64, false, true));
            }
        }
        EsType::SparseVector | EsType::RankFeatures => {
            if let Some(id) = main {
                // Weighted feature postings (term = token, tf = weight) next
                // to the `token\0hexbits` doc values.
                out.push(mk(id, n, CoreType::Keyword, true, true).with_features(true));
            }
        }
        EsType::SemanticText | EsType::Semantic => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Text, true, false).with_analyzer("standard").with_position_increment_gap(rf.position_increment_gap));
            }
            if let Some(id) = aux("inference") {
                // Stored too: updates re-read a document's inference results.
                let mut f = mk(id, &format!("{n}#inference"), CoreType::Keyword, false, true);
                f.stored = true;
                out.push(f);
            }
        }
        EsType::AnnotatedText => {
            if let Some(id) = main {
                out.push(mk(id, n, CoreType::Text, rf.indexed, false).with_analyzer("whitespace").with_position_increment_gap(rf.position_increment_gap));
            }
        }
        EsType::Binary => {
            if let (Some(id), true) = (main, rf.doc_values) {
                out.push(kw_dv(id));
            }
        }
        EsType::Completion | EsType::Percolator | EsType::RankVectors => {
            if let Some(id) = main {
                out.push(kw_dv(id));
            }
            if rf.declared == EsType::Percolator {
                if let Some(tid) = aux("terms") {
                    out.push(mk(tid, &format!("{n}#terms"), CoreType::Keyword, true, false));
                }
            }
        }
        EsType::GeoShape => {
            if let Some(id) = main {
                // No keyword column: the doc's triangles feed the segment's
                // triangle tree (`index`) and shape doc values
                // (`doc_values`), see `rustic_core::fieldtypes::shape_field`.
                out.push(mk(id, n, CoreType::Keyword, rf.indexed, rf.doc_values).with_shape(true));
            }
        }
        t if range_kind(t).is_some() || is_histogram(t) || is_shape(t) => {
            if let Some(id) = main {
                out.push(kw_dv(id));
            }
        }
        _ => return None,
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Lookup extensions.

/// A token filter prefixing every token (flattened keyed sub-fields: the
/// "normalizer" turns `value` into `key\0value`).
struct PrefixFilter(String);

impl TokenFilter for PrefixFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = format!("{}{}", self.0, t.text);
                t
            })
            .collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(PrefixFilter(self.0.clone()))
    }
}

/// A shingle filter (ES `shingle` with `output_unigrams: false`), joining
/// tokens with a non-breaking space so pre-analysis keeps them whole.
pub struct ShingleFilter {
    pub size: usize,
}

impl TokenFilter for ShingleFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        if tokens.len() < self.size {
            return Vec::new();
        }
        tokens
            .windows(self.size)
            .enumerate()
            .map(|(i, w)| Token {
                text: w.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(TOKEN_JOIN),
                position: i as u32,
                offset_from: w[0].offset_from,
                offset_to: w[w.len() - 1].offset_to,
            })
            .collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(ShingleFilter { size: self.size })
    }
}

/// Edge n-grams (1..=20 chars) of every shingle of size 1..=n (ES
/// `_index_prefix` sub-field of `search_as_you_type`).
pub struct IndexPrefixFilter {
    pub max_shingle: usize,
}

impl TokenFilter for IndexPrefixFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::new();
        for i in 0..tokens.len() {
            let n = self.max_shingle.min(tokens.len() - i);
            let shingle: String = tokens[i..i + n].iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(TOKEN_JOIN);
            let chars: Vec<char> = shingle.chars().collect();
            for l in 1..=chars.len().min(20) {
                out.push(Token { text: chars[..l].iter().collect(), position: i as u32, offset_from: tokens[i].offset_from, offset_to: tokens[i + n - 1].offset_to });
            }
        }
        out
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(IndexPrefixFilter { max_shingle: self.max_shingle })
    }
}

fn analyzer_with(base: &str, registry: &AnalysisRegistry, filter: Box<dyn TokenFilter>) -> Analyzer {
    // `standard` tokenizer + lowercase, like the default text analysis.
    let _ = (base, registry);
    let tok = Analyzer::tokenizer_by_name("standard").expect("standard tokenizer");
    let mut a = Analyzer::new(tok);
    if let Some(l) = Analyzer::token_filter_by_name("lowercase") {
        a = a.with_filter(l);
    }
    a.with_filter(filter)
}

/// Fields that exist in the lookup without being leaves of the mapping:
/// `search_as_you_type` sub-fields, `passthrough` aliases and `_size`.
pub fn extend_lookup(lk: &mut FieldLookup, m: &Mappings) {
    // Percolator `terms` aux field: register it so a terms query on
    // `<field>#terms` resolves to the extracted-terms core id.
    let percolators: Vec<Arc<ResolvedField>> = lk.fields.values().filter(|f| f.declared == EsType::Percolator).cloned().collect();
    for f in percolators {
        let name = format!("{}#terms", f.name);
        if let Some(id) = m.core_ids.get(&name).copied() {
            let mut sf = (*f).clone();
            sf.name = name.clone();
            sf.core_id = Some(id);
            sf.ty = EsType::Keyword;
            sf.declared = EsType::Keyword;
            sf.index_analyzer = None;
            sf.search_analyzer = None;
            sf.search_quote_analyzer = None;
            sf.is_multi_field = true;
            sf.multi_fields = Vec::new();
            sf.copy_to = Vec::new();
            lk.fields.insert(name, Arc::new(sf));
        }
    }
    // search_as_you_type sub-fields.
    let sayt: Vec<Arc<ResolvedField>> = lk.fields.values().filter(|f| f.declared == EsType::SearchAsYouType && !f.is_multi_field).cloned().collect();
    for f in sayt {
        let n = f.ext.as_ref().and_then(|e| e.u64("max_shingle_size")).unwrap_or(3) as usize;
        let mut subs = Vec::new();
        let mut add = |suffix: String, analyzer: Analyzer, search: Option<Arc<Analyzer>>| {
            let name = format!("{}.{suffix}", f.name);
            let Some(id) = m.core_ids.get(&name).copied() else { return };
            let mut sf = (*f).clone();
            sf.name = name.clone();
            sf.core_id = Some(id);
            let a = Arc::new(analyzer);
            sf.index_analyzer = Some(crate::analysis::IndexAnalyzer::Custom(a.clone()));
            sf.search_analyzer = Some(search.clone().unwrap_or_else(|| a.clone()));
            sf.search_quote_analyzer = Some(search.unwrap_or(a));
            sf.is_multi_field = true;
            sf.multi_fields = Vec::new();
            sf.copy_to = Vec::new();
            subs.push(name.clone());
            lk.fields.insert(name, Arc::new(sf));
        };
        for k in 2..=n {
            add(format!("_{k}gram"), analyzer_with("standard", &lk.registry, Box::new(ShingleFilter { size: k })), None);
        }
        let search_prefix = {
            let mut a = Analyzer::new(Analyzer::tokenizer_by_name("standard").expect("standard tokenizer"));
            if let Some(l) = Analyzer::token_filter_by_name("lowercase") {
                a = a.with_filter(l);
            }
            Arc::new(a)
        };
        add("_index_prefix".to_string(), analyzer_with("standard", &lk.registry, Box::new(IndexPrefixFilter { max_shingle: n })), Some(search_prefix));
        let mut main = (*f).clone();
        main.multi_fields.extend(subs);
        lk.fields.insert(f.name.clone(), Arc::new(main));
    }
    // passthrough: leaves are also reachable without the object prefix;
    // conflicts are resolved by priority (higher wins), mapped fields win.
    let mut pass: Vec<(String, i64)> = lk.passthrough.clone();
    pass.sort_by_key(|(_, p)| *p);
    let mut aliases: Vec<(String, Arc<ResolvedField>)> = Vec::new();
    for (path, _) in &pass {
        let prefix = format!("{path}.");
        for (k, f) in &lk.fields {
            if let Some(short) = k.strip_prefix(&prefix) {
                aliases.push((short.to_string(), f.clone()));
            }
        }
    }
    for (short, f) in aliases {
        let mapped_leaf = lk.fields.get(&short).map(|x| !lk.passthrough_alias.contains(&short) && x.name == short).unwrap_or(false);
        if mapped_leaf {
            continue;
        }
        lk.passthrough_alias.insert(short.clone());
        lk.fields.insert(short, f);
    }
    // _size.
    if m.size_enabled() {
        if let Some(id) = m.core_ids.get("_size").copied() {
            let mut fm = FieldMapping::new(EsType::Long);
            fm.params.insert("doc_values".into(), Value::Bool(true));
            let mut probe = Mappings::default();
            probe.core_ids.insert("_size".into(), id);
            let rf = lk.resolve_public(&probe, "_size", &fm);
            lk.fields.insert("_size".into(), Arc::new(rf));
        }
    }
}

/// Synthesize `<flattened>.<key>` (searched through the keyed field).
pub fn synth_flattened(lk: &FieldLookup, name: &str) -> Option<Arc<ResolvedField>> {
    let mut idx = name.len();
    while let Some(p) = name[..idx].rfind('.') {
        let root = &name[..p];
        if let Some(f) = lk.fields.get(root) {
            if f.declared == EsType::Flattened && f.ext.as_ref().map(|e| e.flattened_key.is_none()).unwrap_or(true) {
                let key = &name[p + 1..];
                let keyed = f.aux_id("keyed")?;
                let mut sf = (**f).clone();
                sf.name = name.to_string();
                sf.core_id = Some(keyed);
                let tok = Analyzer::tokenizer_by_name("keyword")?;
                let prefix = rustic_core::fieldtypes::flattened::key_prefix(key);
                sf.normalizer = Some(Arc::new(Analyzer::new(tok).with_filter(Box::new(PrefixFilter(prefix)))));
                let mut ext = f.ext.as_deref().cloned().unwrap_or_default();
                ext.flattened_key = Some(key.to_string());
                sf.ext = Some(Arc::new(ext));
                sf.multi_fields = Vec::new();
                sf.copy_to = Vec::new();
                sf.is_multi_field = false;
                return Some(Arc::new(sf));
            }
            return None;
        }
        idx = p;
    }
    None
}

/// Append-only cache of synthesized fields handing out references that
/// live as long as the cache (entries are boxed, never moved or removed).
#[derive(Default, Debug)]
pub struct SynthCache(parking_lot::Mutex<HashMap<String, Box<Arc<ResolvedField>>>>);

impl Clone for SynthCache {
    fn clone(&self) -> Self {
        SynthCache::default()
    }
}

impl SynthCache {
    pub fn get_or_insert(&self, name: &str, make: impl FnOnce() -> Option<Arc<ResolvedField>>) -> Option<&Arc<ResolvedField>> {
        let mut m = self.0.lock();
        if !m.contains_key(name) {
            let v = make()?;
            m.insert(name.to_string(), Box::new(v));
        }
        let p: *const Arc<ResolvedField> = &**m.get(name)?;
        drop(m);
        // SAFETY: the boxed value is never dropped, replaced or moved while
        // `self` is alive (the map only grows; rehashing moves the Box
        // pointer, not its heap contents), and the returned reference is
        // tied to `&self`.
        Some(unsafe { &*p })
    }
}

/// Mappers to add for a `copy_to` / dynamic value of a new type: none.
pub fn _unused(_: &Mapper) {}

/// Default metric of an `aggregate_metric_double` field.
///
/// ES 9.4+ (`AggregateMetricDoubleFieldType`): queries, sorting, scripts and
/// value-source aggregations read the single configured metric, or the
/// average (`sum / value_count`, kept in the `#avg` column) when both are
/// configured; `None` otherwise. `default_metric` is deprecated and ignored.
pub fn aggs_default_metric(rf: &ResolvedField) -> Option<String> {
    let metrics = amd_metrics(rf);
    if metrics.len() == 1 {
        return metrics.first().cloned();
    }
    (metrics.iter().any(|m| m == "sum") && metrics.iter().any(|m| m == "value_count")).then(|| "avg".to_string())
}

/// The configured metrics of an `aggregate_metric_double` field, in ES's
/// `Metric` enum order (min, max, sum, value_count).
pub fn amd_metrics(rf: &ResolvedField) -> Vec<String> {
    let listed: Vec<String> = rf.ext.as_ref().and_then(|x| x.params.get("metrics").and_then(Value::as_array)).map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()).unwrap_or_default();
    ["min", "max", "sum", "value_count"].iter().filter(|m| listed.iter().any(|l| l == *m)).map(|m| m.to_string()).collect()
}

/// `docvalue_fields` values for the types here (`None` = generic).
pub fn docvalues(leaf: &rustic_core::query::LeafReader<'_>, f: &ResolvedField, doc: rustic_core::types::DocId, format: Option<&str>) -> Option<Vec<Value>> {
    if let Some(k) = f.flattened_key() {
        let id = f.core_id?;
        let prefix = rustic_core::fieldtypes::flattened::key_prefix(k);
        return Some(leaf.keyword_values(id, doc).into_iter().filter_map(|v| v.strip_prefix(&prefix).map(|s| Value::String(s.to_string()))).collect());
    }
    match f.declared {
        EsType::DateNanos => {
            let id = f.core_id?;
            let explicit = f.ext.as_ref().and_then(|e| e.str("format")).map(str::to_string);
            let spec = match format {
                Some(s) if s != "use_field_mapping" => Some(s.to_string()),
                _ => explicit,
            };
            Some(leaf.i64_values(id, doc).into_iter().map(|n| Value::String(format_nanos(n, spec.as_deref(), f))).collect())
        }
        EsType::ConstantKeyword => Some(f.constant_value.iter().map(|v| Value::String(match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })).collect()),
        EsType::AggregateMetricDouble => {
            let id = f.aux_id(&aggs_default_metric(f)?)?;
            Some(leaf.f64_values(id, doc).into_iter().map(crate::values::java_double).collect())
        }
        EsType::Binary | EsType::Version => {
            let id = f.core_id?;
            Some(leaf.keyword_values(id, doc).into_iter().map(Value::String).collect())
        }
        // One value per doc: its vectors (floats as f32, bytes as ints).
        EsType::RankVectors => {
            let id = f.core_id?;
            let float = !matches!(f.ext.as_ref().and_then(|e| e.str("element_type")), Some("byte" | "bit"));
            Some(
                leaf.keyword_values(id, doc)
                    .first()
                    .and_then(|j| serde_json::from_str::<Vec<Vec<f64>>>(j).ok())
                    .map(|vs| {
                        Value::Array(
                            vs.into_iter()
                                .map(|v| Value::Array(v.into_iter().map(|x| if float { crate::values::java_double((x as f32).to_string().parse::<f64>().unwrap_or(x)) } else { Value::from(x as i64) }).collect()))
                                .collect(),
                        )
                    })
                    .into_iter()
                    .collect(),
            )
        }
        t if range_kind(t).is_some() || is_shape(t) || is_histogram(t) => Some(Vec::new()),
        _ => None,
    }
}

/// Render epoch nanos with a date format.
pub fn format_nanos(n: i64, spec: Option<&str>, f: &ResolvedField) -> String {
    use rustic_core::fieldtypes::numeric;
    let ms = numeric::nanos_to_millis(n);
    let sub = n.rem_euclid(1_000_000);
    match spec {
        None | Some("strict_date_optional_time_nanos") | Some("date_optional_time_nanos") => numeric::format_nanos(n),
        Some("epoch_millis") => {
            if sub == 0 {
                ms.to_string()
            } else {
                let mut frac = format!("{sub:06}");
                while frac.ends_with('0') {
                    frac.pop();
                }
                format!("{ms}.{frac}")
            }
        }
        Some(s) => {
            let fmt = crate::date::DateFormatter::new(s).unwrap_or_else(|_| f.date_format.clone());
            let out = fmt.format(ms, crate::date::Tz::UTC);
            if s.contains("SSSSSSSSS") {
                let ms_frac = format!("{:03}000000", ms.rem_euclid(1000));
                let full = format!("{:09}", n.rem_euclid(1_000_000_000));
                out.replacen(&ms_frac, &full, 1)
            } else {
                out
            }
        }
    }
}
