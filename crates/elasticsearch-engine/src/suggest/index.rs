//! Index-time completion context handling that needs the whole document:
//! contexts read from a `path` field, and the "contexts are mandatory"
//! check of context-enabled completion fields.

use crate::error::{EsError, Result};
use crate::fieldtypes::suggest::{context_defs, context_values, decode_entry, encode_entry};
use crate::highlight::source_values;
use crate::mapping::{EsType, FieldLookup};
use rustic_core::types::{Document, FieldValue};
use serde_json::Value;

/// Post-process the completion values of a parsed document.
pub fn apply_contexts(source: &Value, lookup: &FieldLookup, doc: &mut Document) -> Result<()> {
    // Cheap pre-check: completion doc values carry the 0x1f separator.
    if !doc.fields.iter().any(|(_, v)| matches!(v, FieldValue::Keyword(s) if s.contains('\u{1f}'))) {
        return Ok(());
    }
    for f in lookup.fields.values() {
        if f.declared != EsType::Completion {
            continue;
        }
        let Some(id) = f.core_id else { continue };
        let defs = context_defs(f);
        if defs.is_empty() {
            continue;
        }
        // Context values from `path` fields.
        let mut from_path: Vec<(String, String)> = Vec::new();
        for d in defs.iter() {
            let Some(p) = &d.path else { continue };
            if d.geo {
                // A geo_point path: one point (object / string / [lon, lat]) or several.
                let raw: Vec<Value> = match path_value(source, p) {
                    Some(Value::Array(a)) if !(a.len() == 2 && a.iter().all(Value::is_number)) => a.clone(),
                    Some(v) => vec![v.clone()],
                    None => vec![],
                };
                for v in raw {
                    if let Ok(cv) = context_values(d, &v) {
                        from_path.extend(cv.into_iter().map(|x| (d.name.clone(), x)));
                    }
                }
            } else {
                for v in source_values(source, p) {
                    let s = match v {
                        Value::String(s) => s.clone(),
                        Value::Null => continue,
                        other => other.to_string(),
                    };
                    from_path.push((d.name.clone(), s));
                }
            }
        }
        for (fid, v) in doc.fields.iter_mut() {
            if *fid != id {
                continue;
            }
            let FieldValue::Keyword(s) = v else { continue };
            let Some((input, w, mut ctxs)) = decode_entry(s) else { continue };
            for c in &from_path {
                if !ctxs.contains(c) {
                    ctxs.push(c.clone());
                }
            }
            if ctxs.is_empty() {
                return Err(EsError::document_parsing(format!("[1:1] failed to parse: Contexts are mandatory in context enabled completion field [{}]", f.name))
                    .caused_by(EsError::illegal_argument(format!("Contexts are mandatory in context enabled completion field [{}]", f.name))));
            }
            *s = encode_entry(&input, w, &ctxs);
        }
    }
    Ok(())
}

fn path_value<'v>(src: &'v Value, path: &str) -> Option<&'v Value> {
    let mut cur = src;
    let parts: Vec<&str> = path.split('.').collect();
    let mut i = 0;
    while i < parts.len() {
        let m = cur.as_object()?;
        let mut found = None;
        for j in (i + 1..=parts.len()).rev() {
            if let Some(c) = m.get(&parts[i..j].join(".")) {
                found = Some((c, j));
                break;
            }
        }
        let (c, j) = found?;
        cur = c;
        i = j;
    }
    Some(cur)
}

/// The `input` strings of a completion value (string, object or array).
pub fn completion_inputs(v: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| out.extend(completion_inputs(x))),
        Value::Object(o) => match o.get("input") {
            Some(Value::String(s)) => out.push(s.clone()),
            Some(Value::Array(a)) => out.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
            _ => {}
        },
        _ => {}
    }
    out
}
