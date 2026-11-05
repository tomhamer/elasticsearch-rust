//! Stored fields (`store: true`).
//!
//! ES keeps the values of fields mapped `store: true` in the segment's
//! stored-fields block, separately from `_source`, so `stored_fields`
//! returns them (and highlighting can read them) even when `_source` is
//! disabled, filtered with includes/excludes, or synthetic.
//!
//! Here the block is one field holding a JSON object
//! `{ "<field>": [values...] }` per document: the index's own stored, never
//! indexed keyword ([`FieldLookup::stored_block_id`]), or, in indices whose
//! mapping predates it, [`STORED_FIELD`] (shared with the indexed
//! `_field_names`, which then also indexed the JSON as a term). It is written at index time next to `_source`, persisted by
//! the segment store like every other core field value (so it survives
//! flush, merge and restart), and read back through
//! `LeafReader::stored_values` / [`GetResult::doc`](crate::shard::GetResult).

use crate::mapping::{FieldLookup, FIRST_USER_FIELD, IGNORED_FIELD, STORED_FIELD};
use rustic_core::types::{Document, FieldValue};
use serde_json::{Map, Value};

/// Does any field of the mapping keep stored values?
pub fn has_stored(lk: &FieldLookup) -> bool {
    lk.fields.values().any(|f| f.stored)
}

/// The stored values of `src` for every `store: true` field, as the JSON
/// stored in [`STORED_FIELD`]. `None` when the document has none.
pub fn build(src: &Value, lk: &FieldLookup) -> Option<String> {
    if lk.stored_fields.is_empty() {
        return None;
    }
    let mut out = Map::new();
    for n in &lk.stored_fields {
        let Some(f) = lk.fields.get(n) else { continue };
        let path = if f.is_multi_field { n.rsplit_once('.').map(|x| x.0).unwrap_or(n).to_string() } else { n.clone() };
        let vals = crate::search::fields_api_values(lk, n, &path, src, None);
        if !vals.is_empty() {
            out.insert(n.clone(), Value::Array(vals));
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(Value::Object(out).to_string())
    }
}

/// Add the stored-fields block of `src` to `doc`.
pub fn attach(doc: &mut Document, src: &Value, lk: &FieldLookup) {
    if let Some(json) = build(src, lk) {
        doc.fields.push((block_field(lk), FieldValue::Keyword(json)));
    }
}

/// The field the stored-fields block of `lk`'s index is written to.
fn block_field(lk: &FieldLookup) -> u16 {
    lk.stored_block_id.unwrap_or(STORED_FIELD)
}

/// Remove `names` from the stored-fields block of a document being
/// indexed (the whole block when nothing else is left).
pub fn drop_from_block(doc: &mut Document, lk: &FieldLookup, names: &[String]) {
    if names.is_empty() {
        return;
    }
    let block = block_field(lk);
    let Some(i) = doc.fields.iter().position(|(id, v)| *id == block && v.as_text().is_some_and(|s| s.starts_with('{'))) else { return };
    let Some(Value::Object(mut m)) = doc.fields[i].1.as_text().and_then(|s| serde_json::from_str::<Value>(s).ok()) else { return };
    let before = m.len();
    for n in names {
        m.shift_remove(n);
    }
    if m.len() == before {
        return;
    }
    if m.is_empty() {
        doc.fields.remove(i);
    } else {
        doc.fields[i].1 = FieldValue::Keyword(Value::Object(m).to_string());
    }
}

/// The stored-fields block of a stored document. A stored document holds
/// only reserved fields plus the block: the block is the JSON object in
/// [`STORED_FIELD`] (older indices; `_field_names` values there are plain
/// names) or the one field past the reserved ids.
pub fn from_doc(doc: &Document) -> Map<String, Value> {
    doc.fields
        .iter()
        .filter(|(id, _)| *id == STORED_FIELD || *id >= FIRST_USER_FIELD)
        .filter_map(|(_, v)| v.as_text())
        .find(|s| s.starts_with('{'))
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .and_then(|v| if let Value::Object(m) = v { Some(m) } else { None })
        .unwrap_or_default()
}

/// Stored values of `field` of a stored document: `_ignored`, or a
/// `store: true` field.
pub fn values_of(doc: &Document, field: &str) -> Vec<Value> {
    if field == "_ignored" {
        // Stored sorted and unique (a keyword doc-values field).
        let mut v: Vec<&str> = doc.get_all(IGNORED_FIELD).filter_map(|v| v.as_text()).collect();
        v.sort_unstable();
        v.dedup();
        return v.into_iter().map(|s| Value::String(s.to_string())).collect();
    }
    match from_doc(doc).remove(field) {
        Some(Value::Array(a)) => a,
        _ => Vec::new(),
    }
}

/// Nest `{"a.b": [v]}` entries into `{"a": {"b": [v]}}`, keeping only the
/// names under `prefix` (`""` = all) and stripping it.
fn expand(map: &Map<String, Value>, prefix: &str) -> Option<Value> {
    let mut root = Map::new();
    for (name, vals) in map {
        let rel = if prefix.is_empty() {
            name.as_str()
        } else {
            match name.strip_prefix(prefix).and_then(|r| r.strip_prefix('.')) {
                Some(r) => r,
                None => continue,
            }
        };
        let mut cur = &mut root;
        let mut parts = rel.split('.').peekable();
        while let Some(p) = parts.next() {
            if parts.peek().is_none() {
                cur.insert(p.to_string(), vals.clone());
            } else {
                let e = cur.entry(p.to_string()).or_insert_with(|| Value::Object(Map::new()));
                match e {
                    Value::Object(m) => cur = m,
                    _ => break,
                }
            }
        }
    }
    if root.is_empty() {
        None
    } else {
        Some(Value::Object(root))
    }
}

/// A source-shaped object holding only the stored fields of a root
/// document (what highlighting reads when `_source` is disabled).
pub fn expanded_source(doc: &Document) -> Option<Value> {
    expand(&from_doc(doc), "")
}

/// The stored fields of a nested child document as an object relative to
/// its nested `path`.
pub fn element_source(child: &Document, path: &str) -> Option<Value> {
    expand(&from_doc(child), path)
}

/// `src` with the values of `stored` (stored fields, as an object in the
/// same shape) laid over it: highlighting reads a `store: true` field from
/// its stored values, which `_source` filtering may have dropped.
pub fn overlay(src: Option<Value>, stored: Option<Value>) -> Option<Value> {
    fn merge(dst: &mut Value, add: Value) {
        match (dst, add) {
            (Value::Object(d), Value::Object(a)) => {
                for (k, v) in a {
                    match d.get_mut(&k) {
                        Some(x) if x.is_object() && v.is_object() => merge(x, v),
                        _ => {
                            d.insert(k, v);
                        }
                    }
                }
            }
            (d, a) => *d = a,
        }
    }
    match (src, stored) {
        (Some(mut s), Some(st)) => {
            merge(&mut s, st);
            Some(s)
        }
        (s, st) => s.or(st),
    }
}

/// Add the stored-fields block of one nested element (`elem`, the object
/// at nested `path`) to its child document.
pub fn attach_nested(child: &mut Document, elem: &serde_json::Map<String, Value>, path: &str, lk: &FieldLookup) {
    let mut names: Vec<&String> = lk.fields.iter().filter(|(n, f)| f.stored && f.nested_path.as_deref() == Some(path) && !n.starts_with('_')).map(|(n, _)| n).collect();
    if names.is_empty() {
        return;
    }
    names.sort();
    let src = Value::Object(elem.clone());
    let mut out = Map::new();
    for n in names {
        let f = &lk.fields[n];
        let full = if f.is_multi_field { n.rsplit_once('.').map(|x| x.0).unwrap_or(n).to_string() } else { n.clone() };
        let rel = full.strip_prefix(path).and_then(|r| r.strip_prefix('.')).unwrap_or(&full).to_string();
        let vals = crate::search::fields_api_values(lk, n, &rel, &src, None);
        if !vals.is_empty() {
            out.insert(n.clone(), Value::Array(vals));
        }
    }
    if !out.is_empty() {
        child.fields.push((block_field(lk), FieldValue::Keyword(Value::Object(out).to_string())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use crate::mapping::Mappings;
    use serde_json::json;

    fn lookup(mapping: Value) -> FieldLookup {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&mapping, &reg).unwrap();
        m.assign_core_ids();
        FieldLookup::build(&m, &reg)
    }

    #[test]
    fn block_holds_store_true_fields_in_document_order() {
        let lk = lookup(json!({"properties": {
            "t": {"type": "text", "store": true}, "k": {"type": "keyword", "store": true, "ignore_above": 3},
            "n": {"type": "long", "store": true}, "d": {"type": "date", "store": true}, "plain": {"type": "keyword"},
            "o": {"properties": {"inner": {"type": "keyword", "store": true}}}
        }}));
        let src = json!({"t": ["the fox", "a"], "k": ["ab", "toolong", "cd"], "n": [3, 1], "d": "2020-01-01", "plain": "x", "o": {"inner": "v"}});
        let mut doc = Document::new();
        attach(&mut doc, &src, &lk);
        let m = from_doc(&doc);
        assert_eq!(Value::Object(m), json!({"d": ["2020-01-01T00:00:00.000Z"], "k": ["ab", "cd"], "n": [3, 1], "o.inner": ["v"], "t": ["the fox", "a"]}));
        assert_eq!(values_of(&doc, "n"), vec![json!(3), json!(1)]);
        assert!(values_of(&doc, "plain").is_empty());
        // A document with nothing stored gets no block.
        let mut empty = Document::new();
        attach(&mut empty, &json!({"plain": "x"}), &lk);
        assert!(empty.get(STORED_FIELD).is_none());
        // Source-shaped view for highlighting.
        assert_eq!(expanded_source(&doc).unwrap()["o"]["inner"], json!(["v"]));
    }

    #[test]
    fn ignored_values_are_sorted_and_unique_and_nested_blocks_are_relative() {
        let mut doc = Document::new();
        for f in ["z", "a", "z"] {
            doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f.to_string())));
        }
        assert_eq!(values_of(&doc, "_ignored"), vec![json!("a"), json!("z")]);
        let lk = lookup(json!({"properties": {"n": {"type": "nested", "properties": {"s": {"type": "text", "store": true}}}}}));
        let mut child = Document::new();
        attach_nested(&mut child, json!({"s": "quick fox"}).as_object().unwrap(), "n", &lk);
        assert_eq!(element_source(&child, "n").unwrap(), json!({"s": ["quick fox"]}));
    }
}
