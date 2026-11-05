//! `stored_fields` for get / mget: values of `store: true` fields as
//! `fields`, and `_source` only when listed.

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::shard::GetResult;
use elasticsearch_engine::stored;
use serde_json::{Map, Value};

/// A `stored_fields` value (`"a,b"` or `["a", "b"]`).
pub(crate) fn list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The mapping definition of `path` (multi-fields included).
fn field_def<'m>(mapping: &'m Value, path: &str) -> Option<&'m Value> {
    let mut cur = mapping;
    let parts: Vec<&str> = path.split('.').collect();
    let mut i = 0;
    while i < parts.len() {
        let props = cur.get("properties").and_then(Value::as_object);
        // Try the longest dotted key first (fields with dots in their name).
        let mut found = None;
        for j in (i + 1..=parts.len()).rev() {
            let key = parts[i..j].join(".");
            if let Some(d) = props.and_then(|p| p.get(&key)) {
                found = Some((d, j));
                break;
            }
            if let Some(d) = cur.get("fields").and_then(|f| f.get(&key)) {
                found = Some((d, j));
                break;
            }
        }
        let (d, j) = found?;
        cur = d;
        i = j;
    }
    Some(cur)
}

fn flatten(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten(x, out)),
        Value::Null => {}
        other => out.push(other.clone()),
    }
}

/// Rewrite a found get JSON (rendered with the full source) for
/// `stored_fields`.
pub(crate) fn apply(svc: &IndexService, json: &str, stored: &[String], g: &GetResult) -> Result<String, EsError> {
    let mut v: Value = serde_json::from_str(json)?;
    let Some(o) = v.as_object_mut() else { return Ok(json.to_string()) };
    let src = o.get("_source").cloned().unwrap_or(Value::Null);
    let mapping = svc.meta().mappings.to_json();
    let mut fields = Map::new();
    for f in stored {
        if f == "_source" {
            continue;
        }
        if f == "_ignored" {
            // Metadata field: a top-level key of the response, not under `fields`.
            let ig = stored::values_of(&g.doc, "_ignored");
            if !ig.is_empty() {
                o.insert("_ignored".into(), Value::Array(ig));
            }
            continue;
        }
        let names: Vec<String> = if f.contains('*') {
            let mut all = Vec::new();
            collect_paths(&mapping, "", &mut all);
            all.into_iter().filter(|p| elasticsearch_engine::mapping::wildcard_match(f, p)).collect()
        } else {
            vec![f.clone()]
        };
        for name in names {
            let is_stored = field_def(&mapping, &name).and_then(|d| d.get("store")).map(|s| s.as_bool() == Some(true) || s.as_str() == Some("true")).unwrap_or(false);
            if !is_stored {
                continue;
            }
            // The stored-fields block first (it exists even without `_source`);
            // documents written before it existed fall back to the source.
            let mut vals = stored::values_of(&g.doc, &name);
            if vals.is_empty() && stored::from_doc(&g.doc).is_empty() {
                for x in elasticsearch_engine::highlight::source_values(&src, &name) {
                    flatten(x, &mut vals);
                }
            }
            if !vals.is_empty() {
                fields.insert(name, Value::Array(vals));
            }
        }
    }
    if !stored.iter().any(|s| s == "_source") {
        o.remove("_source");
    }
    if !fields.is_empty() {
        o.insert("fields".into(), Value::Object(fields));
    }
    Ok(serde_json::to_string(&v)?)
}

fn collect_paths(m: &Value, prefix: &str, out: &mut Vec<String>) {
    if let Some(p) = m.get("properties").and_then(Value::as_object) {
        for (k, d) in p {
            let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            out.push(full.clone());
            collect_paths(d, &full, out);
            if let Some(fs) = d.get("fields").and_then(Value::as_object) {
                for sub in fs.keys() {
                    out.push(format!("{full}.{sub}"));
                }
            }
        }
    }
}
