//! `POST /_scripts/painless/_execute`.
//!
//! Like ES, the `context_setup.document` is indexed into a transient
//! in-memory index built with the target index's mapping, so `doc['f']`
//! sees exactly what a search would (every field type, doc values, vectors,
//! ...); `context_setup.query` is run against it to produce `_score`.

use super::{compile_json_source, error, score_of, RtType, ScriptContext, Val};
use crate::error::{EsError, Result};
use crate::mapping::{FieldLookup, Mappings, ID_FIELD};
use crate::query::dsl::parse_query;
use rustic_core::types::IndexSettings;
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// Contexts `_execute` supports (ES 8.19 `PainlessExecuteAction`).
const CONTEXTS: &[&str] = &[
    "painless_test",
    "filter",
    "score",
    "boolean_field",
    "date_field",
    "double_field",
    "geo_point_field",
    "ip_field",
    "keyword_field",
    "long_field",
    "composite_field",
];

fn xcontent(reason: impl Into<String>) -> EsError {
    EsError::new(400, "x_content_parse_exception", reason)
}

fn validation(errors: &[&str]) -> EsError {
    let mut s = String::from("Validation Failed: ");
    for (i, e) in errors.iter().enumerate() {
        s.push_str(&format!("{}: {};", i + 1, e));
    }
    EsError::new(400, "action_request_validation_exception", s)
}

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "VALUE_STRING",
        Value::Number(n) if n.is_f64() => "VALUE_NUMBER_FLOAT",
        Value::Number(_) => "VALUE_NUMBER_INT",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

/// Execute API. `index_lookup` resolves `context_setup.index` to its mapping.
pub fn painless_execute(body: &Value, index_lookup: &dyn Fn(&str) -> Result<Arc<FieldLookup>>, stored: Option<&super::stored::StoredScripts>) -> Result<Value> {
    let result = painless_execute_source(body, None, index_lookup, stored)?;
    Ok(serde_json::from_str(&result.text())?)
}

/// HTTP entry point retaining raw script parameters/literals and result units.
pub fn painless_execute_source(body: &Value, raw: Option<&crate::source_json::Source>, index_lookup: &dyn Fn(&str) -> Result<Arc<FieldLookup>>, stored: Option<&super::stored::StoredScripts>) -> Result<crate::source_json::Source> {
    let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body or source parameter is required"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "script" | "context" | "context_setup") {
            return Err(xcontent(format!("[1:2] [painless_execute_request] unknown field [{k}]")));
        }
    }
    let script_v = o.get("script").ok_or_else(|| EsError::illegal_argument("Required [script]"))?;
    let Some(script_o) = script_v.as_object() else {
        return Err(xcontent(format!("[1:12] [painless_execute_request] script doesn't support values of type: {}", token_name(script_v))));
    };
    for k in script_o.keys() {
        if !matches!(k.as_str(), "source" | "id" | "params" | "lang" | "options") {
            let inner = xcontent(format!("[1:28] [script] unknown field [{k}]"));
            let mut outer = xcontent("[1:35] [painless_execute_request] failed to parse field [script]");
            outer.caused_by = Some(Box::new(inner.clone()));
            // ES reports the inner exception as the root cause.
            let _ = outer;
            let mut e = inner.clone();
            e.reason = "[1:35] [painless_execute_request] failed to parse field [script]".into();
            e.caused_by = Some(Box::new(inner));
            return Err(e);
        }
    }
    let context = match o.get("context") {
        None | Some(Value::Null) => "painless_test",
        Some(Value::String(s)) => s.as_str(),
        Some(v) => return Err(xcontent(format!("[1:12] [painless_execute_request] context doesn't support values of type: {}", token_name(v)))),
    };
    if !CONTEXTS.contains(&context) {
        let mut e = xcontent("Failed to build [painless_execute_request] after last required field arrived");
        e.caused_by = Some(Box::new(EsError::new(400, "unsupported_operation_exception", format!("unsupported script context name [{context}]"))));
        return Err(e);
    }
    let setup = match o.get("context_setup") {
        None | Some(Value::Null) => None,
        Some(Value::Object(m)) => Some(m),
        Some(v) => return Err(xcontent(format!("[1:12] [painless_execute_request] context_setup doesn't support values of type: {}", token_name(v)))),
    };
    let needs_doc = context != "painless_test";
    let mut errors: Vec<&str> = Vec::new();
    if script_o.contains_key("id") {
        errors.push("only inline scripts are supported");
    }
    if needs_doc {
        let Some(setup) = setup else {
            return Err(EsError::new(500, "null_pointer_exception", "Cannot read field \"index\" because \"this.contextSetup\" is null"));
        };
        if setup.get("index").map_or(true, Value::is_null) {
            errors.push("index is a required parameter for current context");
        }
        if setup.get("document").map_or(true, Value::is_null) {
            errors.push("document is a required parameter for current context");
        }
    }
    if !errors.is_empty() {
        return Err(validation(&errors));
    }
    if let Some(lang) = script_o.get("lang").and_then(Value::as_str) {
        if lang != "painless" {
            let src = script_o.get("source").and_then(Value::as_str).unwrap_or("");
            let cause = match lang {
                "mustache" => Some(format!("mustache engine does not know how to handle context [{context}]")),
                "expression" => Some(format!("expression engine does not know how to handle script context [{context}]")),
                _ => None,
            };
            return Err(match cause {
                Some(c) => {
                    let mut e = EsError::new(500, "general_script_exception", format!("Failed to compile inline script [{src}] using lang [{lang}]"));
                    e.caused_by = Some(Box::new(EsError::illegal_argument(c)));
                    e
                }
                None => EsError::illegal_argument(format!("script_lang not supported [{lang}]")),
            });
        }
    }

    let ctx = match context {
        "painless_test" => ScriptContext::PainlessTest,
        "filter" => ScriptContext::Filter,
        "score" => ScriptContext::Score,
        "long_field" => ScriptContext::RuntimeLong,
        "double_field" => ScriptContext::RuntimeDouble,
        "boolean_field" => ScriptContext::RuntimeBoolean,
        "date_field" => ScriptContext::RuntimeDate,
        "ip_field" => ScriptContext::RuntimeIp,
        "geo_shape_field" | "geometry_field" => ScriptContext::RuntimeGeoShape,
        _ => ScriptContext::RuntimeKeyword,
    };
    let mut spec = script_v.clone();
    if let Some(m) = spec.as_object_mut() {
        m.remove("lang");
    }
    let cs = compile_json_source(&spec, raw.and_then(|r| r.get("script")), ctx, stored)?;

    // Index the document into a transient in-memory index.
    let index = setup.and_then(|s| s.get("index")).and_then(Value::as_str);
    let lookup = match index {
        Some(i) => Some(index_lookup(i)?),
        None => None,
    };
    let document = setup.and_then(|s| s.get("document"));
    let mem = match (&lookup, document) {
        (Some(lk), Some(doc)) => Some(build_memory_index(lk, doc)?),
        _ => None,
    };
    let mut exec = cs.executor(lookup.clone(), None);
    if let Some(doc) = document {
        exec.rt.doc.source_json = Some(doc.clone());
        exec.set_source(Some(Val::from_json(doc)));
    }
    let leaves = mem.as_ref().map(|m| m.leaves()).unwrap_or_default();
    if let Some(leaf) = leaves.first() {
        // SAFETY: `mem` outlives `exec`, which is dropped before this fn returns.
        unsafe { exec.bind_leaf(leaf) };
        exec.set_doc(0);
    }
    let mut score = 0.0f64;
    if context == "score" {
        if let (Some(q), Some(lk)) = (setup.and_then(|s| s.get("query")), &lookup) {
            let qctx = crate::fieldtypes::percolate::qctx(lk, &leaves);
            let query = parse_query(q, &qctx)?;
            score = crate::fieldtypes::join::run(&query, &leaves)?.first().map(|h| h.2 as f64).unwrap_or(0.0);
        }
    }
    exec.set_score(score);

    let v = exec.run().map_err(|e| cs.error(&e))?;
    if context == "painless_test" {
        let result = super::source::to_source(&Val::utf16(v.string_units()))?;
        return Ok(crate::source_json::Source::Object(std::collections::BTreeMap::from([("result".into(), result)])));
    }
    let result = match context {
        "filter" => Value::Bool(v.as_bool().map_err(|e| cs.error(&e))?),
        "score" => {
            let d = score_of(&v).map_err(|e| cs.error(&e.at(exec.program().result_pos)))?;
            if d.is_nan() {
                Value::String("NaN".into())
            } else if d.is_infinite() {
                Value::String(if d > 0.0 { "Infinity" } else { "-Infinity" }.into())
            } else {
                crate::values::java_double(d)
            }
        }
        _ => emitted_result(context, exec.take_emitted()).map_err(|e| cs.error(&e))?,
    };
    Ok(crate::source_json::Source::from_value(&json!({ "result": result })))
}

/// Index `doc` with `lookup`'s mapping into a one-document memory index.
pub(crate) fn build_memory_index(lookup: &FieldLookup, doc: &Value) -> Result<rustic_core::Index> {
    build_memory_index_sealed(lookup, doc, true)
}

/// [`build_memory_index`]; `seal: false` leaves the document in the
/// index's memtable, which [`rustic_core::Index::leaves`] reads like a
/// segment (index-time field scripts: sealing a one-document segment per
/// indexed document cost more than parsing it).
pub(crate) fn build_memory_index_sealed(lookup: &FieldLookup, doc: &Value, seal: bool) -> Result<rustic_core::Index> {
    let mut lk = lookup.clone();
    lk.root_dynamic = Some(crate::mapping::Dynamic::False);
    let schema = lookup.core_schema();
    let mut mem = rustic_core::Index::new("_painless_execute", schema, IndexSettings::default());
    let m = Mappings::default();
    if let crate::document::ParseOutcome::Doc(pd) = crate::document::parse_document(doc, "0", &m, &lk)? {
        let mut d = pd.doc;
        d.add_keyword(ID_FIELD, "0");
        mem.add_document(&d).map_err(|e| EsError::internal(e.to_string()))?;
    }
    if seal {
        mem.refresh().map_err(|e| EsError::internal(e.to_string()))?;
    }
    Ok(mem)
}

/// Normalise and render the values a field script emitted, like the
/// corresponding ES `*FieldScript` + `fields` rendering.
pub(crate) fn emitted_result(context: &str, emitted: Vec<Val>) -> std::result::Result<Value, Box<error::ScriptErr>> {
    let ty = match context {
        "boolean_field" => RtType::Boolean,
        "date_field" => RtType::Date,
        "double_field" => RtType::Double,
        "keyword_field" => RtType::Keyword,
        "ip_field" => RtType::Ip,
        "long_field" => RtType::Long,
        _ => RtType::Long,
    };
    if context == "geo_point_field" {
        let mut out = Vec::new();
        for v in emitted {
            let (lat, lon) = match &v {
                Val::Map(m) => {
                    let m = m.borrow();
                    let Some((k, x)) = m.iter().next() else { continue };
                    (k.to_val().to_f64()?, x.to_f64()?)
                }
                other => return Err(error::class_cast(format!("Cannot cast {} to double", other.type_name()))),
            };
            out.push(json!({ "coordinates": [lon, lat], "type": "Point" }));
        }
        return Ok(Value::Array(out));
    }
    if context == "composite_field" {
        let mut o: Map<String, Value> = Map::new();
        for v in emitted {
            if let Val::Map(m) = &v {
                for (k, x) in m.borrow().iter() {
                    let key = format!("composite_field.{}", k.as_string());
                    let arr = o.entry(key).or_insert_with(|| Value::Array(Vec::new()));
                    if let Value::Array(a) = arr {
                        a.push(x.to_json());
                    }
                }
            }
        }
        return Ok(Value::Object(o));
    }
    let mut fields = super::RuntimeFields::default();
    fields.parse_into(&json!({ "f": { "type": ty.name(), "format": "strict_date_optional_time" } })).map_err(|e| error::illegal_arg(e.reason))?;
    let rf = fields.fields.pop().ok_or_else(|| error::illegal_arg("no runtime field"))?;
    let mut vals = emitted.into_iter().map(|v| rf.convert(v)).collect::<std::result::Result<Vec<_>, _>>()?;
    if ty == RtType::Boolean {
        // BooleanFieldScript keeps counts of falses and trues: all falses, then all trues.
        let f = vals.iter().filter(|v| matches!(v, Val::Bool(false))).count();
        let t = vals.len() - f;
        vals = std::iter::repeat(Val::Bool(false)).take(f).chain(std::iter::repeat(Val::Bool(true)).take(t)).collect();
    }
    Ok(Value::Array(vals.iter().map(|v| rf.render(v)).collect()))
}

#[cfg(test)]
mod utf16_tests {
    use super::*;
    #[test]
    fn execute_preserves_raw_parameters_literals_and_result() {
        for (input, expected) in [
            (r#"{"script":{"source":"params.s","params":{"s":"\ud800"}}}"#, 0xd800),
            (r#"{"script":{"source":"'\udfff'"}}"#, 0xdfff),
            (r#"{"script":{"source":"params.s.substring(0,1)","params":{"s":"😀"}}}"#, 0xd83d),
        ] {
            let (body, raw) = crate::source_json::update_request(input).unwrap();
            let result = painless_execute_source(&body, raw.as_ref(), &|_| panic!("no index lookup"), None).unwrap();
            let units = crate::source_json::decode_string(&result.get("result").unwrap().text());
            assert_eq!(units, vec![expected], "{input}");
            assert!(!result.text().contains('�'));
        }
    }
}
