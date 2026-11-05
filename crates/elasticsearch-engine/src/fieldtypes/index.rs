//! Document parsing (`_source` value → core field values) for the field
//! types of this module. Called by [`crate::document`] before its own
//! handling; returns the scalar values that multi-fields and `copy_to`
//! should also receive.

use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, FieldMapping, Mapper, ResolvedField};
use crate::values;
use rustic_core::fieldtypes::{features, flattened, histogram, murmur3, ngram, numeric, range as rng, shape, version};
use rustic_core::types::{Document, FieldValue};
use serde_json::{json, Map, Value};
use std::collections::HashSet;

/// What emission may touch in the document being parsed.
pub struct Emit<'a> {
    pub id: &'a str,
    pub max_regex_length: usize,
    pub lookup: &'a FieldLookup,
    pub doc: &'a mut Document,
    pub ignored: &'a mut Vec<String>,
    pub pending: &'a mut Vec<(String, Mapper)>,
    pub pending_paths: &'a mut HashSet<String>,
    pub now: i64,
}

impl Emit<'_> {
    fn push(&mut self, id: Option<u16>, v: FieldValue) {
        if let Some(id) = id {
            self.doc.fields.push((id, v));
        }
    }
    fn ignore(&mut self, f: &ResolvedField) {
        if !self.ignored.contains(&f.name) {
            self.ignored.push(f.name.clone());
        }
    }
}

/// Does this module parse values of `f`?
pub fn handles(f: &ResolvedField) -> bool {
    if f.flattened_key().is_some() {
        return false;
    }
    matches!(
        f.declared,
        EsType::Flattened
            | EsType::Wildcard
            | EsType::DateNanos
            | EsType::ConstantKeyword
            | EsType::Binary
            | EsType::Version
            | EsType::TokenCount
            | EsType::Murmur3
            | EsType::AnnotatedText
            | EsType::SemanticText
            | EsType::Semantic
            | EsType::Completion
            | EsType::Join
            | EsType::Percolator
            | EsType::IntegerRange
            | EsType::LongRange
            | EsType::FloatRange
            | EsType::DoubleRange
            | EsType::DateRange
            | EsType::IpRange
            | EsType::GeoShape
            | EsType::Shape
            | EsType::Point
            | EsType::Histogram
            | EsType::TDigest
            | EsType::ExponentialHistogram
            | EsType::AggregateMetricDouble
            | EsType::SparseVector
            | EsType::RankFeature
            | EsType::RankFeatures
            | EsType::RankVectors
    )
}

fn preview(v: &Value) -> String {
    crate::document::preview(v)
}

pub fn parse_err(f: &ResolvedField, id: &str, v: &Value, cause: EsError) -> EsError {
    // ES's DocumentParsingException includes the cause's message in the
    // top-level reason: "failed to parse field [x] of type [y] ... : <cause>".
    // For histogram-like fields, ES uses a shorter "error parsing field" format.
    // `exponential_histogram`: the mapper's own exception wrapped as
    // `failed to parse field [x] of type [exponential_histogram]`.
    if f.declared == EsType::ExponentialHistogram {
        return EsError::document_parsing(format!("[1:1] failed to parse field [{}] of type [exponential_histogram]", f.name)).caused_by(cause);
    }
    let is_histogram = super::is_histogram(f.declared);
    if is_histogram {
        // ES's HistogramFieldMapper throws a DocumentParsingException whose message
        // is "[<location>] error parsing field [x], <cause>" (the location prefix is
        // added by DocumentParsingException when the token location is known), with
        // no `caused_by` in the response. The yaml catch regexes match the space
        // before "error", so the "[1:1] " prefix is required.
        EsError::document_parsing(format!(
            "[1:1] error parsing field [{}], {}",
            f.name,
            cause.reason
        ))
    } else {
        EsError::document_parsing(format!(
            "[1:1] failed to parse field [{}] of type [{}] in document with id '{}'. Preview of field's value: '{}': {}",
            f.name,
            f.declared.name(),
            id,
            preview(v),
            cause.reason
        ))
        .caused_by(cause)
    }
}

fn malformed(f: &ResolvedField, e: &mut Emit, v: &Value, cause: EsError) -> Result<()> {
    if f.ignore_malformed {
        e.ignore(f);
        return Ok(());
    }
    Err(parse_err(f, e.id, v, cause))
}

fn scalar_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Values of a possibly-array field with `null_value` substitution.
fn each<'v>(f: &ResolvedField, v: &'v Value, out: &mut Vec<Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| each(f, x, out)),
        Value::Null => {
            if let Some(nv) = &f.null_value {
                out.push(nv.clone());
            }
        }
        x => out.push(x.clone()),
    }
}

fn ext_bool(f: &ResolvedField, k: &str, d: bool) -> bool {
    f.ext.as_ref().map(|e| e.bool(k, d)).unwrap_or(d)
}

/// Parse `v` into core values of `f`; returns the scalars for
/// multi-fields / `copy_to`.
pub fn emit(f: &ResolvedField, v: &Value, e: &mut Emit) -> Result<Vec<Value>> {
    match f.declared {
        EsType::Flattened => flattened_emit(f, v, e).map(|_| vec![]),
        EsType::Wildcard | EsType::Version => {
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            let mut out = Vec::new();
            for x in vals {
                let Some(s) = scalar_str(&x) else {
                    return Err(parse_err(f, e.id, &x, EsError::illegal_argument(format!("Can't get text on a START_OBJECT at 1:1"))));
                };
                if let Some(ia) = f.ignore_above {
                    if s.chars().count() > ia {
                        e.ignore(f);
                        continue;
                    }
                }
                e.push(f.core_id, FieldValue::Keyword(s.clone()));
                if f.declared == EsType::Wildcard {
                    if let Some(g) = f.aux_id("ngram") {
                        for t in ngram::ngrams(&s) {
                            e.doc.fields.push((g, FieldValue::Keyword(t)));
                        }
                    }
                } else if let Some(sid) = f.aux_id("sort") {
                    e.doc.fields.push((sid, FieldValue::Keyword(version::encode(&s))));
                }
                out.push(x);
            }
            Ok(out)
        }
        EsType::DateNanos => {
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            let mut out = Vec::new();
            for x in vals {
                match parse_nanos(f, &x) {
                    Ok(n) => {
                        e.push(f.core_id, FieldValue::Date(n));
                        out.push(x);
                    }
                    Err(c) => malformed(f, e, &x, c)?,
                }
            }
            Ok(out)
        }
        EsType::ConstantKeyword => {
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            for x in vals {
                let Some(s) = scalar_str(&x) else { continue };
                match &f.constant_value {
                    Some(c) if scalar_str(c).as_deref() != Some(s.as_str()) => {
                        return Err(parse_err(
                            f,
                            e.id,
                            &x,
                            EsError::illegal_argument(format!(
                                "[constant_keyword] field [{}] only accepts values that are equal to the value defined in the mappings [{}], but got [{s}]",
                                f.name,
                                scalar_str(c).unwrap_or_default()
                            )),
                        ));
                    }
                    Some(_) => {}
                    None => {
                        if !e.pending_paths.contains(&f.name) {
                            let mut fm = FieldMapping::new(EsType::ConstantKeyword);
                            fm.params.insert("value".into(), Value::String(s.clone()));
                            e.pending_paths.insert(f.name.clone());
                            e.pending.push((f.name.clone(), Mapper::Field(fm)));
                        }
                    }
                }
                e.push(f.core_id, FieldValue::Keyword(s));
            }
            Ok(vec![])
        }
        EsType::Binary => {
            use base64::Engine;
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            for x in vals {
                let Value::String(s) = &x else {
                    return Err(parse_err(f, e.id, &x, EsError::illegal_argument(format!("Current token ({}) not VALUE_STRING or VALUE_EMBEDDED_OBJECT, can not access as binary", token_name(&x)))));
                };
                let ok = base64::engine::general_purpose::STANDARD.decode(s.trim()).is_ok() || base64::engine::general_purpose::STANDARD_NO_PAD.decode(s.trim()).is_ok();
                if !ok {
                    return Err(parse_err(f, e.id, &x, EsError::new(400, "json_parse_exception", "Failed to decode VALUE_STRING as base64 (MIME-NO-LINEFEEDS): Illegal character")));
                }
                if f.doc_values {
                    e.push(f.core_id, FieldValue::Keyword(s.trim().to_string()));
                }
            }
            Ok(vec![])
        }
        EsType::TokenCount => {
            let mut vals = Vec::new();
            match v {
                Value::Array(a) => vals.extend(a.iter().cloned()),
                x => vals.push(x.clone()),
            }
            for x in vals {
                let n = match &x {
                    Value::Null => match f.null_value.as_ref().and_then(|n| n.as_i64().or_else(|| n.as_str().and_then(|s| s.parse().ok()))) {
                        Some(n) => n,
                        None => continue,
                    },
                    other => {
                        let Some(s) = scalar_str(other) else { continue };
                        let toks = match f.ext.as_ref().and_then(|x| x.analyzer.clone()) {
                            Some(a) => a.analyze(&s),
                            None => rustic_core::analysis::Analyzer::standard().analyze(&s),
                        };
                        if ext_bool(f, "enable_position_increments", true) {
                            toks.last().map(|t| t.position as i64 + 1).unwrap_or(0)
                        } else {
                            toks.len() as i64
                        }
                    }
                };
                e.push(f.core_id, FieldValue::I64(n));
            }
            Ok(vec![])
        }
        EsType::Murmur3 => {
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            for x in vals {
                if let Some(s) = scalar_str(&x) {
                    e.push(f.core_id, FieldValue::I64(murmur3::murmur3_field_hash(&s)));
                }
            }
            Ok(vec![])
        }
        EsType::AnnotatedText => {
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            let mut out = Vec::new();
            for x in vals {
                let Some(s) = scalar_str(&x) else { continue };
                let (plain, annotations) = parse_annotated(&s);
                let toks = match f.ext.as_ref().and_then(|x| x.analyzer.clone()) {
                    Some(a) => a.analyze(&plain),
                    None => rustic_core::analysis::Analyzer::standard().analyze(&plain),
                };
                let mut text: Vec<String> = toks.into_iter().map(|t| t.text.replace(' ', super::TOKEN_JOIN)).collect();
                text.extend(annotations.into_iter().map(|a| a.replace(' ', super::TOKEN_JOIN)));
                e.push(f.core_id, FieldValue::Text(text.join(" ")));
                out.push(Value::String(plain));
            }
            Ok(out)
        }
        EsType::SemanticText if crate::inference::semantic::is_legacy_value(v) => {
            // Legacy-format value carrying its inference results.
            let texts = crate::inference::semantic::legacy_texts(v);
            let id = f.ext.as_ref().and_then(|x| x.str("inference_id")).unwrap_or(crate::inference::semantic::DEFAULT_INFERENCE_ID).to_string();
            let mapped = f.ext.as_ref().and_then(|x| x.params.get("model_settings").cloned()).filter(|x| !x.is_null());
            let blob = crate::inference::semantic::legacy_blob(&f.name, &id, mapped.as_ref(), v, |c| parse_err(f, e.id, v, c))?;
            for t in &texts {
                e.push(f.core_id, FieldValue::Text(t.clone()));
            }
            e.push(f.aux_id("inference"), FieldValue::Keyword(blob));
            Ok(texts.into_iter().map(Value::String).collect())
        }
        EsType::Semantic => {
            // Text values are searchable lexically; media values only
            // through their inference results.
            crate::inference::semantic::check_semantic_input(&f.name, v)?;
            let texts: Vec<String> = crate::inference::semantic::semantic_inputs(v).into_iter().filter(|(ty, _)| ty == "text").map(|(_, t)| t).collect();
            for t in &texts {
                e.push(f.core_id, FieldValue::Text(t.clone()));
            }
            Ok(texts.into_iter().map(Value::String).collect())
        }
        EsType::SemanticText => {
            // The inference results are indexed per document (supplied or
            // computed `_inference_fields`, see `document.rs`); the values
            // themselves are only searchable lexically.
            crate::inference::semantic::check_input(&f.name, v)?;
            let mut vals = Vec::new();
            each(f, v, &mut vals);
            let texts: Vec<String> = vals.iter().filter_map(scalar_str).collect();
            for t in &texts {
                e.push(f.core_id, FieldValue::Text(t.clone()));
            }
            Ok(texts.into_iter().map(Value::String).collect())
        }
        EsType::Completion => completion_emit(f, v, e).map(|_| vec![]),
        EsType::Join => {
            let (name, parent) = match v {
                Value::String(s) => (s.clone(), None),
                Value::Object(o) => (
                    o.get("name").and_then(Value::as_str).map(str::to_string).ok_or_else(|| parse_err(f, e.id, v, EsError::illegal_argument(format!("[name] is missing for join field [{}]", f.name))))?,
                    o.get("parent").map(|p| scalar_str(p).unwrap_or_default()),
                ),
                Value::Null => return Ok(vec![]),
                other => return Err(parse_err(f, e.id, other, EsError::illegal_argument(format!("[{}] expected START_OBJECT or VALUE_STRING but was: {}", f.name, token_name(other))))),
            };
            let rel = super::join::Relations::of(f);
            if !rel.knows(&name) {
                return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("unknown join name [{name}] for field [{}]", f.name))));
            }
            if rel.parent_of(&name).is_some() {
                let Some(p) = parent else {
                    return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("[parent] is missing for join field [{}]", f.name))));
                };
                if let Some(pid) = f.aux_id("parent") {
                    e.doc.fields.push((pid, FieldValue::Keyword(p)));
                }
            } else if parent.is_some() {
                return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("[parent] is not allowed for parent [{name}] in join field [{}]", f.name))));
            }
            e.push(f.core_id, FieldValue::Keyword(name));
            Ok(vec![])
        }
        EsType::Percolator => {
            if !v.is_object() {
                return Err(parse_err(f, e.id, v, EsError::parsing("query malformed, must start with start_object")));
            }
            super::percolate::validate_stored_query(v, e.lookup, e.max_regex_length).map_err(|c| parse_err(f, e.id, v, c))?;
            e.push(f.core_id, FieldValue::Keyword(v.to_string()));
            // Extracted terms for candidate pruning at percolate time.
            let tid = f.aux_id("terms");
            if let Some(tid) = tid {
                let terms = super::percolate::extract_terms(v, e.lookup);
                for t in terms {
                    e.push(Some(tid), FieldValue::Keyword(t));
                }
            }
            Ok(vec![])
        }
        t if super::range_kind(t).is_some() => {
            let kind = super::range_kind(t).unwrap();
            let mut vals = Vec::new();
            match v {
                Value::Array(a) => vals.extend(a.iter().filter(|x| !x.is_null()).cloned()),
                Value::Null => {}
                x => vals.push(x.clone()),
            }
            for x in vals {
                match parse_range_value(f, kind, &x, e.now) {
                    Ok((lo, hi)) => e.push(f.core_id, FieldValue::Keyword(rng::encode(lo, hi))),
                    Err(c) => return Err(parse_err(f, e.id, &x, c)),
                }
            }
            Ok(vec![])
        }
        t if super::is_shape(t) => {
            let mut vals = Vec::new();
            match v {
                Value::Array(a) if t == EsType::Point && a.len() >= 2 && a.iter().all(Value::is_number) => vals.push(v.clone()),
                Value::Array(a) if t != EsType::Point && a.len() >= 2 && a.iter().all(Value::is_number) => vals.push(v.clone()),
                Value::Array(a) => vals.extend(a.iter().cloned()),
                Value::Null => {
                    if let Some(nv) = &f.null_value {
                        vals.push(nv.clone());
                    }
                }
                x => vals.push(x.clone()),
            }
            for x in vals {
                #[cfg(test)]
                let t_parse = std::time::Instant::now();
                let parsed = parse_shape(f, &x);
                #[cfg(test)]
                shape_ingest_stats::add(&shape_ingest_stats::PARSE_NS, t_parse);
                match parsed {
                    // `geo_shape` values: the triangles ES indexes
                    // (`GeoShapeIndexer`, one tessellation per value: a
                    // polygon Lucene cannot tessellate fails the doc as in
                    // ES, or is ignored with `ignore_malformed`) feed the
                    // segment's triangle tree and the doc value; every
                    // indexed value adds to the doc's centroid (ES
                    // `BinaryShapeDocValuesField.add`, raw coordinates).
                    Ok(g) if t == EsType::GeoShape => {
                        let version = rustic_core::codec::shape_write_version();
                        if version < rustic_core::codec::SHAPE_TREE_SINCE {
                            return Err(EsError::illegal_argument(format!(
                                "geo_shape fields need segment format v{} but the cluster writes v{version}",
                                rustic_core::codec::SHAPE_TREE_SINCE
                            )));
                        }
                        let mut tris = Vec::new();
                        #[cfg(test)]
                        let t_tess = std::time::Instant::now();
                        let tessellated = rustic_core::fieldtypes::tessellator::geometry_triangles(&shape::normalized(&g), &mut tris);
                        #[cfg(test)]
                        shape_ingest_stats::add(&shape_ingest_stats::TESSELLATE_NS, t_tess);
                        match tessellated {
                            Ok(()) if tris.is_empty() => {}
                            Ok(()) => {
                                if let Some(id) = f.core_id {
                                    #[cfg(test)]
                                    let t_enc = std::time::Instant::now();
                                    let mut calc = e.doc.shape_centroid(id).copied().unwrap_or_default();
                                    calc.add(&g);
                                    e.doc.set_shape_centroid(id, calc);
                                    e.doc.add_shape_triangles(id, tris);
                                    #[cfg(test)]
                                    shape_ingest_stats::add(&shape_ingest_stats::ENCODE_NS, t_enc);
                                }
                            }
                            Err(msg) => malformed(f, e, &x, EsError::illegal_argument(msg))?,
                        }
                    }
                    Ok(g) => e.push(f.core_id, FieldValue::Keyword(g.encode())),
                    Err(c) => malformed(f, e, &x, c)?,
                }
            }
            Ok(vec![])
        }
        EsType::ExponentialHistogram => {
            let one = match v {
                Value::Null => return Ok(vec![]),
                Value::Array(a) => {
                    let vals: Vec<&Value> = a.iter().filter(|x| !x.is_null()).collect();
                    if vals.len() > 1 {
                        return malformed(f, e, v, EsError::illegal_argument(format!("Field [{}] of type [exponential_histogram] doesn't support indexing multiple values for the same field in the same document", f.name))).map(|_| vec![]);
                    }
                    match vals.first() {
                        Some(x) => *x,
                        None => return Ok(vec![]),
                    }
                }
                x => x,
            };
            match super::exphist::parse(&f.name, one, f.coerce) {
                Ok(h) => e.push(f.core_id, FieldValue::Keyword(super::exphist::encode(&h))),
                Err(c) => malformed(f, e, v, c.into_es(&f.name))?,
            }
            Ok(vec![])
        }
        t if super::is_histogram(t) => {
            if v.is_null() {
                return Ok(vec![]);
            }
            // `histogram` with `coerce` (9.x): an exponential histogram
            // object is converted to centroids at its bucket centers.
            if t == EsType::Histogram && f.coerce && v.as_object().and_then(|o| o.keys().next()).is_some_and(|k| super::exphist::is_root_field(k)) {
                match super::exphist::parse(&f.name, v, false) {
                    Ok(h) => {
                        let (vs, cs): (Vec<f64>, Vec<i64>) = super::exphist::to_centroids(&h).into_iter().map(|(m, c)| (m, c as i64)).unzip();
                        e.push(f.core_id, FieldValue::Keyword(histogram::encode(&vs, &cs)));
                    }
                    Err(c) => malformed(f, e, v, c.into_es(&f.name))?,
                }
                return Ok(vec![]);
            }
            match parse_histogram(t, v) {
                Ok((vs, cs)) => e.push(f.core_id, FieldValue::Keyword(histogram::encode(&vs, &cs))),
                Err(c) => malformed(f, e, v, c)?,
            }
            Ok(vec![])
        }
        EsType::AggregateMetricDouble => {
            let Some(o) = v.as_object() else {
                if v.is_null() {
                    return Ok(vec![]);
                }
                return malformed(f, e, v, EsError::illegal_argument(format!("Failed to parse object: expecting token of type [START_OBJECT] but found [{}]", token_name(v)))).map(|_| vec![]);
            };
            let metrics: Vec<String> = f
                .ext
                .as_ref()
                .and_then(|x| x.params.get("metrics").and_then(Value::as_array).map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()))
                .unwrap_or_default();
            for k in o.keys() {
                if !metrics.contains(k) {
                    return malformed(f, e, v, EsError::illegal_argument(format!("Aggregate metric [{k}] does not exist in the mapping of field [{}]", f.name))).map(|_| vec![]);
                }
            }
            if let Some(missing) = metrics.iter().find(|m| !o.contains_key(*m)) {
                let _ = missing;
                return malformed(f, e, v, EsError::illegal_argument(format!("Aggregate metric field [{}] must contain all metrics {:?}", f.name, metrics).replace('"', ""))).map(|_| vec![]);
            }
            let mut vals = Vec::new();
            for m in &metrics {
                let x = &o[m];
                let n = match x.as_f64() {
                    Some(n) if n.is_finite() => n,
                    _ => return malformed(f, e, v, EsError::illegal_argument(format!("Failed to parse object: expecting token of type [VALUE_NUMBER] but found [{}]", token_name(x)))).map(|_| vec![]),
                };
                if m == "value_count" && (n < 0.0 || n.fract() != 0.0) {
                    return malformed(f, e, v, EsError::illegal_argument(format!("Aggregate metric [value_count] of field [{}] cannot be a negative number", f.name))).map(|_| vec![]);
                }
                vals.push((m.clone(), n));
            }
            if let (Some(mn), Some(mx)) = (vals.iter().find(|x| x.0 == "min"), vals.iter().find(|x| x.0 == "max")) {
                if mn.1 > mx.1 {
                    return malformed(f, e, v, EsError::illegal_argument(format!("Aggregate metric field [{}] max value cannot be smaller than min value", f.name))).map(|_| vec![]);
                }
            }
            // The average (`AggregateMetricAverageScript`) the field reads as
            // when it carries both `sum` and `value_count`.
            if let (Some(s), Some(c)) = (vals.iter().find(|x| x.0 == "sum"), vals.iter().find(|x| x.0 == "value_count")) {
                let avg = s.1 / c.1;
                if avg.is_finite() {
                    if let Some(id) = f.aux_id("avg") {
                        e.doc.fields.push((id, FieldValue::F64(avg)));
                    }
                }
            }
            for (m, n) in vals {
                if let Some(id) = f.aux_id(&m) {
                    e.doc.fields.push((id, FieldValue::F64(n)));
                }
            }
            Ok(vec![])
        }
        EsType::SparseVector | EsType::RankFeatures => {
            let positive = ext_bool(f, "positive_score_impact", true);
            let objs: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                Value::Null => vec![],
                x => vec![x],
            };
            let ty = f.declared.name();
            let mut seen: HashSet<String> = HashSet::new();
            for o in objs {
                let Some(m) = o.as_object() else {
                    return Err(parse_err(f, e.id, o, EsError::illegal_argument(format!("[{ty}] fields must be json objects, expected a START_OBJECT but got: {}", token_name(o)))));
                };
                if m.is_empty() {
                    e.push(f.core_id, FieldValue::Keyword(String::new()));
                }
                for (tok, w) in m {
                    let Some(mut x) = w.as_f64() else {
                        return Err(parse_err(f, e.id, o, EsError::illegal_argument(format!("[{ty}] fields take hashes that map a feature to a strictly positive float, but got unexpected token {}", token_name(w)))));
                    };
                    if !(x > 0.0 && x.is_finite()) {
                        return Err(parse_err(f, e.id, o, EsError::illegal_argument(format!("[{ty}] fields do not support negative or zero values, but got [{x}] for feature [{tok}]"))));
                    }
                    if f.declared == EsType::RankFeatures && !seen.insert(tok.clone()) {
                        return Err(parse_err(f, e.id, o, EsError::illegal_argument(format!("[rank_features] fields do not support indexing multiple values for the same rank feature [{}.{tok}] in the same document", f.name))));
                    }
                    if !positive {
                        x = 1.0 / x;
                    }
                    e.push(f.core_id, FieldValue::Keyword(features::encode(tok, features::quantize(x as f32))));
                }
            }
            Ok(vec![])
        }
        EsType::RankFeature => {
            if v.is_array() {
                return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("[rank_feature] fields do not support indexing multiple values for the same field [{}] in the same document", f.name))));
            }
            if v.is_null() {
                return Ok(vec![]);
            }
            let x = match v {
                Value::Number(n) => n.as_f64(),
                Value::String(s) => s.trim().parse().ok(),
                _ => None,
            };
            let Some(mut x) = x.filter(|x| x.is_finite() && *x > 0.0) else {
                return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("[rank_feature] fields do not support negative or zero values, but got [{}] for field [{}]", preview(v), f.name))));
            };
            if !ext_bool(f, "positive_score_impact", true) {
                x = 1.0 / x;
            }
            e.push(f.core_id, FieldValue::F64(features::quantize(x as f32) as f64));
            Ok(vec![])
        }
        EsType::RankVectors => {
            let Value::Array(a) = v else {
                if v.is_null() {
                    return Ok(vec![]);
                }
                return Err(parse_err(f, e.id, v, EsError::illegal_argument("[rank_vectors] fields must be arrays of vectors".to_string())));
            };
            let et_name = f.ext.as_ref().and_then(|x| x.str("element_type")).unwrap_or("float").to_string();
            let mut vecs: Vec<Vec<f64>> = Vec::new();
            for x in a {
                if let Value::String(s) = x {
                    let dims = f.ext.as_ref().and_then(|x| x.u64("dims")).map(|d| d as usize);
                    let decoded = crate::vectors::decode_vector_string(s, &et_name, dims).map_err(|c| parse_err(f, e.id, v, c))?;
                    vecs.push(decoded);
                    continue;
                }
                let Some(inner) = x.as_array() else {
                    return Err(parse_err(f, e.id, v, EsError::illegal_argument("[rank_vectors] fields must be arrays of vectors".to_string())));
                };
                let mut vv = Vec::new();
                for n in inner {
                    vv.push(n.as_f64().ok_or_else(|| parse_err(f, e.id, v, EsError::illegal_argument("[rank_vectors] values must be numbers".to_string())))?);
                }
                vecs.push(vv);
            }
            let et = f.ext.as_ref().and_then(|x| x.str("element_type")).unwrap_or("float").to_string();
            // `bit` vectors hold dims/8 bytes; `byte` / `bit` values are bytes.
            let per_dim = if et == "bit" { 8 } else { 1 };
            if et == "bfloat16" {
                // Kept at bfloat16 precision, as ES stores them.
                for x in vecs.iter_mut().flatten() {
                    *x = crate::vectors::bf16_round(*x as f32) as f64;
                }
            } else if et != "float" {
                for x in vecs.iter().flatten() {
                    if x.fract() != 0.0 || !(-128.0..=127.0).contains(x) {
                        return Err(parse_err(f, e.id, v, EsError::illegal_argument(format!("element_type [{et}] vectors only support integers between [-128, 127] but found [{x}] at dim [0]; Preview of invalid vector: {}", Value::from(vecs.first().cloned().unwrap_or_default())))));
                    }
                }
            }
            let dims = f.ext.as_ref().and_then(|x| x.u64("dims")).map(|d| d as usize / per_dim);
            let first = vecs.first().map(|x| x.len()).unwrap_or(0);
            match dims {
                None if first > 0 => {
                    if !e.pending_paths.contains(&f.name) {
                        let mut fm = FieldMapping::new(EsType::RankVectors);
                        fm.params.insert("dims".into(), json!(first * per_dim));
                        if et != "float" {
                            fm.params.insert("element_type".into(), json!(et));
                        }
                        e.pending_paths.insert(f.name.clone());
                        e.pending.push((f.name.clone(), Mapper::Field(fm)));
                    }
                }
                _ => {}
            }
            let want = dims.unwrap_or(first);
            if let Some(bad) = vecs.iter().find(|x| x.len() != want) {
                return Err(EsError::document_parsing(format!(
                    "[1:1] failed to parse: The [rank_vectors] field [{}] in doc [document with id '{}'] has a different number of dimensions [{}] than defined in the mapping [{}]",
                    f.name,
                    e.id,
                    bad.len() * per_dim,
                    want * per_dim
                )));
            }
            e.push(f.core_id, FieldValue::Keyword(serde_json::to_string(&vecs)?));
            Ok(vec![])
        }
        _ => Ok(vec![]),
    }
}

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(true) => "VALUE_TRUE",
        Value::Bool(false) => "VALUE_FALSE",
        Value::Number(n) if n.is_f64() => "VALUE_NUMBER_FLOAT",
        Value::Number(_) => "VALUE_NUMBER_INT",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn flattened_emit(f: &ResolvedField, v: &Value, e: &mut Emit) -> Result<()> {
    if v.is_null() {
        return Ok(());
    }
    let objs: Vec<&Value> = match v {
        Value::Array(a) => a.iter().filter(|x| !x.is_null()).collect(),
        x => vec![x],
    };
    for o in objs {
        if !o.is_object() {
            return Err(EsError::document_parsing(format!(
                "[1:1] failed to parse field [{}] of type [flattened] in document with id '{}'. Preview of field's value: '{}'",
                f.name,
                e.id,
                preview(o)
            ))
            .caused_by(EsError::new(400, "parsing_exception", format!("Failed to parse object: expecting token of type [START_OBJECT] but found [{}]", token_name(o)))));
        }
        let depth = f.ext.as_ref().and_then(|x| x.u64("depth_limit")).unwrap_or(20) as usize;
        let nv = f.null_value.as_ref().and_then(scalar_str);
        let leaves = flattened::flatten(o, depth, nv.as_deref()).map_err(|_| {
            EsError::document_parsing(format!("[1:1] failed to parse field [{}] of type [flattened] in document with id '{}'. Preview of field's value: '{}'", f.name, e.id, preview(o)))
                .caused_by(EsError::illegal_argument(format!("The provided [flattened] field [{}] exceeds the maximum depth limit of [{depth}].", f.name)))
        })?;
        let keyed = f.aux_id("keyed");
        for (k, val) in leaves {
            if let Some(ia) = f.ignore_above {
                if val.chars().count() > ia {
                    e.ignore(f);
                    continue;
                }
            }
            e.push(f.core_id, FieldValue::Keyword(val.clone()));
            if let Some(kid) = keyed {
                e.doc.fields.push((kid, FieldValue::Keyword(flattened::keyed(&k, &val))));
            }
        }
    }
    Ok(())
}

/// Epoch nanos of a `date_nanos` value.
pub fn parse_nanos(f: &ResolvedField, v: &Value) -> std::result::Result<i64, EsError> {
    let (nanos, shown) = match v {
        Value::String(s) => {
            let ms = f.date_format.parse(s).map_err(|_| EsError::illegal_argument(format!("failed to parse date field [{s}] with format [{}]", f.date_format.spec)))?;
            let n = match numeric::fraction_nanos(s) {
                Some(frac) => numeric::nanos_from_parts(ms.div_euclid(1000), frac).map_err(|e| (e, s.clone())),
                None => numeric::nanos_from_millis(ms, 0).map_err(|e| (e, s.clone())),
            };
            (n, s.clone())
        }
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            let ms = f.date_format.parse_number(x.trunc()).map_err(|_| EsError::illegal_argument(format!("failed to parse date field [{n}] with format [{}]", f.date_format.spec)))?;
            let sub = ((x - x.trunc()) * 1e6).round() as u32;
            (numeric::nanos_from_millis(ms, sub).map_err(|e| (e, n.to_string())), n.to_string())
        }
        _ => return Err(EsError::illegal_argument(format!("failed to parse date field [{}] with format [{}]", preview(v), f.date_format.spec))),
    };
    let _ = shown;
    nanos.map_err(|(e, s)| match e {
        numeric::NanosError::BeforeEpoch => EsError::illegal_argument(format!("date[{s}] is before the epoch in 1970 and cannot be stored in nanosecond resolution")),
        numeric::NanosError::AfterMax => EsError::illegal_argument(format!("date[{s}] is after {} and cannot be stored in nanosecond resolution", numeric::NANOS_MAX_DISPLAY)),
    })
}

/// `[text](value&value2)` markup → (plain text, annotation values).
pub fn parse_annotated(s: &str) -> (String, Vec<String>) {
    let mut plain = String::with_capacity(s.len());
    let mut ann = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if b[i] == b'[' {
            if let Some(close) = s[i + 1..].find("](") {
                let text_end = i + 1 + close;
                if let Some(paren) = s[text_end + 2..].find(')') {
                    let text = &s[i + 1..text_end];
                    let vals = &s[text_end + 2..text_end + 2 + paren];
                    if !text.contains('[') {
                        plain.push_str(text);
                        for v in vals.split('&') {
                            let d = url_decode(v);
                            if !d.is_empty() {
                                ann.push(d);
                            }
                        }
                        i = text_end + 2 + paren + 1;
                        continue;
                    }
                }
            }
        }
        let ch = s[i..].chars().next().unwrap();
        plain.push(ch);
        i += ch.len_utf8();
    }
    (plain, ann)
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 => {
                match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    Ok(x) => {
                        out.push(x);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn completion_emit(f: &ResolvedField, v: &Value, e: &mut Emit) -> Result<()> {
    let ctx_defs = super::suggest::context_defs(f);
    let mut entries: Vec<(String, u64, Vec<(String, String)>)> = Vec::new();
    let add_obj = |o: &Map<String, Value>, entries: &mut Vec<(String, u64, Vec<(String, String)>)>| -> std::result::Result<(), EsError> {
        let inputs: Vec<String> = match o.get("input") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            _ => return Err(EsError::illegal_argument("input is missing")),
        };
        let weight = match o.get("weight") {
            None | Some(Value::Null) => 1,
            Some(Value::Number(n)) => match n.as_u64() {
                Some(w) if w <= i32::MAX as u64 => w,
                _ => return Err(EsError::illegal_argument(format!("weight must be an integer, but was [{n}]"))),
            },
            Some(Value::String(s)) => s.parse::<u64>().map_err(|_| EsError::illegal_argument(format!("weight must be an integer, but was [{s}]")))?,
            Some(other) => return Err(EsError::illegal_argument(format!("weight must be a number or string, but was [{}]", token_name(other)))),
        };
        let mut ctxs = Vec::new();
        if let Some(c) = o.get("contexts").and_then(Value::as_object) {
            for (name, val) in c {
                let def = ctx_defs.iter().find(|d| &d.name == name).ok_or_else(|| EsError::illegal_argument(format!("context [{name}] is not defined for field [{}]", f.name)))?;
                for cv in super::suggest::context_values(def, val)? {
                    ctxs.push((name.clone(), cv));
                }
            }
        }
        for i in inputs {
            entries.push((i, weight, ctxs.clone()));
        }
        Ok(())
    };
    let items: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        Value::Null => vec![],
        x => vec![x],
    };
    for it in items {
        match it {
            Value::String(s) => entries.push((s.clone(), 1, Vec::new())),
            Value::Object(o) => add_obj(o, &mut entries).map_err(|c| parse_err(f, e.id, it, c))?,
            other => return Err(parse_err(f, e.id, other, EsError::illegal_argument(format!("failed to parse [{}]: expected text or object, but got {}", f.name, token_name(other))))),
        }
    }
    for (input, w, ctxs) in entries {
        e.push(f.core_id, FieldValue::Keyword(super::suggest::encode_entry(&input, w, &ctxs)));
    }
    Ok(())
}

/// Parse a range field value into inclusive sortable keys.
pub fn parse_range_value(f: &ResolvedField, kind: rng::RangeKind, v: &Value, now: i64) -> std::result::Result<(u128, u128), EsError> {
    use rng::RangeKind as K;
    if kind == K::Ip {
        if let Value::String(s) = v {
            let (lo, hi) = rustic_core::fieldtypes::ip::cidr(s).ok_or_else(|| EsError::illegal_argument(format!("Expected [ip/prefix] but was [{s}]")))?;
            return Ok((rng::key_ip(lo), rng::key_ip(hi)));
        }
    }
    let o = v.as_object().ok_or_else(|| EsError::illegal_argument(format!("error parsing field [{}], expected an object but got {}", f.name, token_name(v))))?;
    let mut lo: Option<(u128, bool)> = None;
    let mut hi: Option<(u128, bool)> = None;
    let mut inc_lo = true;
    let mut inc_hi = true;
    for (k, x) in o {
        match k.as_str() {
            "gte" | "gt" | "from" => {
                if !x.is_null() {
                    // `gt` rounds a partial date up (`gt: 2017-09-01` starts after the whole day).
                    lo = Some((range_key(f, kind, x, k == "gt", now)?, k == "gt"));
                } else if k == "gt" {
                    // `gt: null` is an exclusive unbounded bound (8.15 off-by-one fix).
                    lo = Some((kind.min_key(), true));
                }
            }
            "lte" | "lt" | "to" => {
                if !x.is_null() {
                    hi = Some((range_key(f, kind, x, k != "lt", now)?, k == "lt"));
                } else if k == "lt" {
                    hi = Some((kind.max_key(), true));
                }
            }
            "include_lower" => inc_lo = x.as_bool().unwrap_or(true),
            "include_upper" => inc_hi = x.as_bool().unwrap_or(true),
            other => return Err(EsError::illegal_argument(format!("error parsing field [{}], with unknown parameter [{other}]", f.name))),
        }
    }
    let lo = match lo {
        Some((k, excl)) if excl || (!inc_lo && o.contains_key("from")) => kind.next_up(k),
        Some((k, _)) => k,
        None => kind.min_key(),
    };
    let hi = match hi {
        Some((k, excl)) if excl || (!inc_hi && o.contains_key("to")) => kind.next_down(k),
        Some((k, _)) => k,
        None => kind.max_key(),
    };
    if lo > hi {
        return Err(EsError::illegal_argument(format!("min value ({}) is greater than max value ({})", key_display(kind, lo), key_display(kind, hi))));
    }
    Ok((lo, hi))
}

pub fn key_display(kind: rng::RangeKind, k: u128) -> String {
    use rng::RangeKind as K;
    match kind {
        K::Integer | K::Long | K::Date => rng::i64_of_key(k).to_string(),
        K::Float | K::Double => rng::f64_of_key(k).to_string(),
        K::Ip => rustic_core::fieldtypes::ip::display(rng::ip_of_key(k)),
    }
}

/// One bound → key. `round_up` applies to date math rounding.
pub fn range_key(f: &ResolvedField, kind: rng::RangeKind, v: &Value, round_up: bool, now: i64) -> std::result::Result<u128, EsError> {
    use rng::RangeKind as K;
    let num = |v: &Value| -> std::result::Result<f64, EsError> {
        match v {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
            Value::String(s) => s.trim().parse::<f64>().map_err(|_| EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))),
            _ => Err(EsError::illegal_argument(format!("Current token ({}) not numeric", token_name(v)))),
        }
    };
    Ok(match kind {
        K::Integer | K::Long => {
            let x = match v {
                Value::Number(n) if n.is_i64() => n.as_i64().unwrap(),
                Value::String(s) if s.trim().parse::<i64>().is_ok() => s.trim().parse().unwrap(),
                other => {
                    let x = num(other)?;
                    if round_up {
                        x.floor() as i64
                    } else {
                        x.ceil() as i64
                    }
                }
            };
            if kind == K::Integer && (x < i32::MIN as i64 || x > i32::MAX as i64) {
                return Err(EsError::illegal_argument(format!("Value [{x}] is out of range for an integer")));
            }
            rng::key_i64(x)
        }
        K::Float => rng::key_f64(values::to_float(num(v)?)),
        K::Double => rng::key_f64(num(v)?),
        K::Date => {
            let ms = match v {
                Value::Number(n) => n.as_f64().unwrap_or(0.0) as i64,
                Value::String(s) => crate::date::parse_date_math(s, &f.date_format, crate::date::Tz::UTC, now, round_up)?,
                other => return Err(EsError::illegal_argument(format!("failed to parse date field [{}]", preview(other)))),
            };
            rng::key_i64(ms)
        }
        K::Ip => {
            let s = scalar_str(v).unwrap_or_default();
            rng::key_ip(rustic_core::fieldtypes::ip::parse(&s).ok_or_else(|| EsError::illegal_argument(format!("'{s}' is not an IP string literal.")))?)
        }
    })
}

fn parse_shape(f: &ResolvedField, v: &Value) -> std::result::Result<shape::Geometry, EsError> {
    let geo = f.declared == EsType::GeoShape;
    let opts = shape::ParseOpts {
        geo,
        coerce: ext_bool(f, "coerce", false),
        ignore_z: ext_bool(f, "ignore_z_value", true),
    };
    if f.declared == EsType::Point {
        return parse_xy_point(v, &opts).map(shape::Geometry::Point).map_err(EsError::illegal_argument);
    }
    shape::parse(v, &opts).map_err(EsError::illegal_argument)
}

/// Cartesian `point` input forms: `{x,y}`, `[x,y]`, `"x,y"`, WKT, GeoJSON.
pub fn parse_xy_point(v: &Value, o: &shape::ParseOpts) -> std::result::Result<(f64, f64), String> {
    match v {
        Value::Object(m) if m.contains_key("x") => {
            let x = m.get("x").and_then(|x| x.as_f64().or_else(|| x.as_str().and_then(|s| s.parse().ok()))).ok_or("field [x] missing")?;
            let y = m.get("y").and_then(|x| x.as_f64().or_else(|| x.as_str().and_then(|s| s.parse().ok()))).ok_or("field [y] missing")?;
            Ok((x, y))
        }
        Value::Array(a) if a.len() >= 2 => {
            let x = a[0].as_f64().ok_or("numeric value expected")?;
            let y = a[1].as_f64().ok_or("numeric value expected")?;
            if a.len() > 2 && !o.ignore_z {
                return Err(format!("Exception parsing coordinates: found Z value [{}] but [ignore_z_value] parameter is [false]", a[2]));
            }
            Ok((x, y))
        }
        Value::String(s) if s.contains(',') && !s.trim_start().to_ascii_uppercase().starts_with("POINT") => {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() < 2 {
                return Err(format!("expected 2 or 3 coordinates but found: [{}]", parts.len()));
            }
            Ok((parts[0].trim().parse().map_err(|_| format!("[{s}] is not a point"))?, parts[1].trim().parse().map_err(|_| format!("[{s}] is not a point"))?))
        }
        other => match shape::parse(other, o)? {
            shape::Geometry::Point(p) => Ok(p),
            _ => Err("[point] supports only POINT among WKT primitives".to_string()),
        },
    }
}

/// `(values, counts)` of a histogram-like value.
/// A parsed `tdigest` value (`TDigestParser.ParsedTDigest`).
pub struct ParsedTDigest {
    pub centroids: Vec<f64>,
    pub counts: Vec<i64>,
    pub sum: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ParsedTDigest {
    /// The explicit or derived summary (`min`, `max`, `sum`); `None` for an
    /// empty digest.
    pub fn summary(&self) -> Option<(f64, f64, f64)> {
        let (first, last) = (self.centroids.first()?, self.centroids.last()?);
        let sum = self.sum.unwrap_or_else(|| self.centroids.iter().zip(&self.counts).fold(0.0, |acc, (c, n)| acc + c * *n as f64));
        Some((self.min.unwrap_or(*first), self.max.unwrap_or(*last), sum))
    }
}

/// `TDigestParser.parse`: messages are the `error parsing field [f], ...`
/// tails.
pub fn parse_tdigest(v: &Value) -> std::result::Result<ParsedTDigest, EsError> {
    let o = v.as_object().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [START_OBJECT] but found [{}]", token_name(v))))?;
    let num = |x: &Value| x.as_f64().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [VALUE_NUMBER] but found [{}]", token_name(x))));
    let (mut centroids, mut counts, mut sum, mut min, mut max) = (None, None, None, None, None);
    for (k, x) in o {
        match k.as_str() {
            "centroids" => {
                let a = x.as_array().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [START_ARRAY] but found [{}]", token_name(x))))?;
                let mut prev = -f64::MAX;
                let mut out = Vec::with_capacity(a.len());
                for y in a {
                    let val = num(y)?;
                    if val < prev {
                        return Err(EsError::illegal_argument(format!(
                            "[centroids] centroids must be in increasing order, got [{}] but previous value was [{}]",
                            crate::values::java_double_string(val),
                            crate::values::java_double_string(prev)
                        )));
                    }
                    out.push(val);
                    prev = val;
                }
                centroids = Some(out);
            }
            "counts" => {
                let a = x.as_array().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [START_ARRAY] but found [{}]", token_name(x))))?;
                let mut out = Vec::with_capacity(a.len());
                for y in a {
                    let c = y.as_i64().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [VALUE_NUMBER_INT] but found [{}]", token_name(y))))?;
                    if c < 0 {
                        return Err(EsError::illegal_argument(format!("[counts] elements must be >= 0 but got {c}")));
                    }
                    out.push(c);
                }
                counts = Some(out);
            }
            "sum" => sum = Some(num(x)?),
            "min" => min = Some(num(x)?),
            "max" => max = Some(num(x)?),
            other => return Err(EsError::illegal_argument(format!("with unknown parameter [{other}]"))),
        }
    }
    let centroids = centroids.ok_or_else(|| EsError::illegal_argument("expected field called [centroids]"))?;
    let counts = counts.ok_or_else(|| EsError::illegal_argument("expected field called [counts]"))?;
    if centroids.len() != counts.len() {
        return Err(EsError::illegal_argument(format!("expected same length from [centroids] and [counts] but got [{} != {}]", centroids.len(), counts.len())));
    }
    if centroids.is_empty() {
        (sum, min, max) = (None, None, None);
    }
    Ok(ParsedTDigest { centroids, counts, sum, min, max })
}

pub fn parse_histogram(t: EsType, v: &Value) -> std::result::Result<(Vec<f64>, Vec<i64>), EsError> {
    let o = v.as_object().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [START_OBJECT] but found [{}]", token_name(v))))?;
    let nums = |k: &str| -> std::result::Result<Vec<f64>, EsError> {
        o.get(k)
            .and_then(Value::as_array)
            .ok_or_else(|| EsError::illegal_argument(format!("expected field called [{k}]")))?
            .iter()
            .map(|x| x.as_f64().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [VALUE_NUMBER] but found [{}]", token_name(x)))))
            .collect()
    };
    let ints = |k: &str| -> std::result::Result<Vec<i64>, EsError> {
        o.get(k)
            .and_then(Value::as_array)
            .ok_or_else(|| EsError::illegal_argument(format!("expected field called [{k}]")))?
            .iter()
            .map(|x| x.as_i64().ok_or_else(|| EsError::illegal_argument(format!("expecting token of type [VALUE_NUMBER_INT] but found [{}]", token_name(x)))))
            .collect()
    };
    match t {
        EsType::Histogram => {
            for k in o.keys() {
                if k != "values" && k != "counts" {
                    return Err(EsError::illegal_argument(format!("with unknown parameter [{k}]")));
                }
            }
            let (vs, cs) = (nums("values")?, ints("counts")?);
            histogram::validate(&vs, &cs).map_err(EsError::illegal_argument)?;
            Ok((vs, cs))
        }
        EsType::TDigest => {
            // `exponential_histogram` coerce input: histogram style.
            if !o.contains_key("centroids") && o.contains_key("values") {
                let (vs, cs) = (nums("values")?, ints("counts")?);
                histogram::validate(&vs, &cs).map_err(EsError::illegal_argument)?;
                return Ok((vs, cs));
            }
            let t = parse_tdigest(v)?;
            Ok((t.centroids, t.counts))
        }
        _ => {
            // exponential_histogram: bucket i at scale s covers
            // (base^i, base^(i+1)], base = 2^(2^-s); represented by its
            // geometric midpoint.
            let scale = o.get("scale").and_then(Value::as_i64).unwrap_or(0);
            if !(-11..=38).contains(&scale) {
                return Err(EsError::illegal_argument(format!("scale field must be in range [-11, 38] but got {scale}")));
            }
            let base = 2f64.powf(2f64.powi(-(scale as i32)));
            let mut pts: Vec<(f64, i64)> = Vec::new();
            for (side, sign) in [("negative", -1.0), ("positive", 1.0)] {
                if let Some(b) = o.get(side).and_then(Value::as_object) {
                    let idx: Vec<i64> = b.get("indices").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
                    let cnt: Vec<i64> = b.get("counts").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
                    if idx.len() != cnt.len() {
                        return Err(EsError::illegal_argument(format!("expected same length from [indices] and [counts] but got [{} != {}]", idx.len(), cnt.len())));
                    }
                    for (i, c) in idx.iter().zip(cnt) {
                        if c < 0 {
                            return Err(EsError::illegal_argument(format!("[counts] elements must be >= 0 but got {c}")));
                        }
                        pts.push((sign * base.powf(*i as f64 + 0.5), c));
                    }
                }
            }
            if let Some(z) = o.get("zero").and_then(Value::as_object) {
                let c = z.get("count").and_then(Value::as_i64).unwrap_or(0);
                if c > 0 {
                    pts.push((0.0, c));
                }
            }
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut vs: Vec<f64> = Vec::new();
            let mut cs: Vec<i64> = Vec::new();
            for (v, c) in pts {
                if vs.last() == Some(&v) {
                    *cs.last_mut().unwrap() += c;
                } else {
                    vs.push(v);
                    cs.push(c);
                }
            }
            Ok((vs, cs))
        }
    }
}

/// `_size`: the source length in bytes.
pub fn emit_size(source: &Value, lookup: &FieldLookup, doc: &mut Document) {
    if let Some(f) = lookup.fields.get("_size") {
        if let Some(id) = f.core_id {
            let n = serde_json::to_vec(source).map(|b| b.len()).unwrap_or(0);
            doc.fields.push((id, FieldValue::I64(n as i64)));
        }
    }
}

/// Doc value display for `docvalue_fields` of the types here.
pub fn _display_unused(_: &EsType) {}

#[allow(dead_code)]
fn geo_hash(lat: f64, lon: f64, p: usize) -> String {
    values::geohash_encode(lat, lon, p)
}

/// Shape indexing phase timers (test builds; the geo suite microbench
/// reports them): nanoseconds parsing shape values, normalizing and
/// tessellating them, and encoding their doc values (with the header),
/// summed over threads.
#[cfg(test)]
pub(crate) mod shape_ingest_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static PARSE_NS: AtomicU64 = AtomicU64::new(0);
    pub static TESSELLATE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
    pub fn add(c: &AtomicU64, since: std::time::Instant) {
        c.fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
    }
    /// `[parse, tessellate, encode]` ns, then zeroed.
    pub fn take() -> [u64; 3] {
        [&PARSE_NS, &TESSELLATE_NS, &ENCODE_NS].map(|c| c.swap(0, Relaxed))
    }
}
