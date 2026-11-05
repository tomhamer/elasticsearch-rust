//! `GET [/{index}]/_mapping/field/{fields}` (get field mapping API).

use crate::{Req, Resp, R};
use elasticsearch_engine::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};

/// Leaf field mappers of a mapping: `(full name, leaf name, definition)`,
/// multi-fields included (objects are not field mappers).
fn leaves(props: &Map<String, Value>, prefix: &str, out: &mut Vec<(String, String, Value)>) {
    for (name, def) in props {
        let full = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
        let ty = def.get("type").and_then(Value::as_str);
        if let Some(p) = def.get("properties").and_then(Value::as_object) {
            if matches!(ty, None | Some("object") | Some("nested")) {
                leaves(p, &full, out);
                continue;
            }
        }
        if matches!(ty, None | Some("object") | Some("nested")) {
            continue;
        }
        out.push((full.clone(), name.clone(), def.clone()));
        if let Some(fs) = def.get("fields").and_then(Value::as_object) {
            for (sub, sdef) in fs {
                out.push((format!("{full}.{sub}"), sub.clone(), sdef.clone()));
            }
        }
    }
}

fn simple_match(pattern: &str, s: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == s;
    }
    let mut rest = s;
    for (i, p) in parts.iter().enumerate() {
        if i == 0 {
            if !rest.starts_with(p) {
                return false;
            }
            rest = &rest[p.len()..];
        } else if i == parts.len() - 1 {
            return rest.ends_with(p);
        } else {
            match rest.find(p) {
                Some(j) => rest = &rest[j + p.len()..],
                None => return false,
            }
        }
    }
    true
}

/// Metadata fields `get field mapping` reports (with an empty mapping).
const METADATA_FIELDS: &[&str] = &[
    "_data_stream_timestamp", "_doc_count", "_feature", "_field_names", "_id", "_ignored", "_ignored_source", "_index", "_index_mode", "_inference_fields", "_nested_path", "_routing", "_seq_no", "_source", "_tier", "_version",
];

/// Parameter defaults printed by `include_defaults=true`, per field type.
static TYPE_DEFAULTS: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| serde_json::from_str(include_str!("include_defaults.json")).unwrap_or(Value::Null));

fn with_defaults(mut def: Value) -> Value {
    if let Some(o) = def.as_object_mut() {
        let t = o.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        if let Some(d) = TYPE_DEFAULTS.get(&t).and_then(Value::as_object) {
            for (k, v) in d {
                o.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    def
}

/// Handles the request when it is a field mapping request (`.../_mapping/field/...`).
pub(crate) fn get_field_mapping(node: &Node, req: &Req, index: Option<&str>) -> R {
    let off = if index.is_some() { 3 } else { 2 };
    let fields: Vec<String> = req.seg(off).split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let defaults = req.flag("include_defaults")?;
    let targets = node.resolve(index.unwrap_or("_all"), ResolveOpts::from_params(&req.params))?;
    let mut out = Map::new();
    for t in targets {
        let m = t.svc.meta().mappings.to_json();
        let mut all = Vec::new();
        if let Some(p) = m.get("properties").and_then(Value::as_object) {
            leaves(p, "", &mut all);
        }
        let mut found = Map::new();
        for pat in &fields {
            for (full, leaf, def) in &all {
                if (pat.contains('*') && simple_match(pat, full)) || pat == full {
                    let mut def = if defaults { with_defaults(def.clone()) } else { def.clone() };
                    // `semantic_text`: `include_defaults` shows the index options the
                    // field resolves to (`SemanticTextFieldMapper.defaultIndexOptions`).
                    if defaults && def.get("type").and_then(Value::as_str) == Some("semantic_text") && def.get("index_options").is_none_or(Value::is_null) {
                        if let Some(o) = def.as_object_mut() {
                            if let Some(io) = elasticsearch_engine::inference::semantic_map::default_index_options(o, t.svc.created_version()) {
                                o.insert("index_options".into(), io);
                            }
                        }
                    }
                    found.insert(full.clone(), json!({"full_name": full, "mapping": {leaf.as_str(): def}}));
                }
            }
            for meta_field in METADATA_FIELDS {
                if (pat.contains('*') && simple_match(pat, meta_field)) || pat == meta_field {
                    found.entry(meta_field.to_string()).or_insert_with(|| json!({"full_name": meta_field, "mapping": {}}));
                }
            }
        }
        out.insert(t.svc.name.clone(), json!({"mappings": found}));
    }
    Ok(Resp::ok(Value::Object(out)))
}
