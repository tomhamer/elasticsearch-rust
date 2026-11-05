//! The `semantic` highlighter (`SemanticTextHighlighter`): a semantic_text
//! field's best chunks for the query, as snippets of the source text.

use super::chunk_query::{self, ChunkQuery};
use crate::highlight::HighlightOpts;
use crate::mapping::FieldLookup;
use rustic_core::query::{LeafReader, Query};
use rustic_core::types::DocId;
use serde_json::{Map, Value};

/// Whether `q` contains a `match_all` (the highlighter then takes every chunk).
fn has_match_all(q: &Query) -> bool {
    match q {
        Query::MatchAll => true,
        Query::Bool { must, should, filter, .. } => must.iter().chain(should).chain(filter).any(has_match_all),
        Query::Boost { query, .. } | Query::ConstScore { query, .. } | Query::FunctionScore { query, .. } => has_match_all(query),
        Query::DisMax { queries, .. } => queries.iter().any(has_match_all),
        _ => false,
    }
}

/// Whether the semantic highlighter handles `name` for these options: the
/// `semantic` type, or no type on a semantic_text field (its default).
pub fn handles(lookup: &FieldLookup, name: &str, o: &HighlightOpts) -> bool {
    match o.ty.as_deref() {
        Some("semantic") => true,
        None => lookup.get(name).is_some_and(|f| f.declared.is_semantic()),
        _ => false,
    }
}

/// The text values of a source field joined as UTF-16 (the offsets of the
/// inference chunks index into it).
fn joined(src: &Value, field: &str) -> Vec<u16> {
    let v = crate::highlight::source_values(src, field);
    let mut vals: Vec<String> = Vec::new();
    for x in v {
        match x {
            Value::Object(_) if super::semantic::is_legacy_value(x) => vals.extend(super::semantic::legacy_texts(x)),
            Value::String(s) => vals.push(s.clone()),
            Value::Null | Value::Object(_) | Value::Array(_) => {}
            o => vals.push(o.to_string()),
        }
    }
    vals.join("\n").encode_utf16().collect()
}

/// Semantic highlights of one hit, by field.
pub fn highlight_hit(spec: &Map<String, Value>, lookup: &FieldLookup, leaf: &LeafReader<'_>, doc: DocId, terms: &crate::highlight::QueryTerms, src: &Value) -> Map<String, Value> {
    let global = HighlightOpts::from_json(spec, None);
    let mut out = Map::new();
    for (pattern, fo) in crate::highlight::field_specs(spec) {
        let o = HighlightOpts::from_json(&fo, Some(&global));
        let names = if pattern.contains('*') { lookup.matching(&pattern) } else { vec![pattern.clone()] };
        for name in names {
            if out.contains_key(&name) || !handles(lookup, &name, &o) {
                continue;
            }
            let Some(f) = lookup.get(&name) else { continue };
            if !f.declared.is_semantic() {
                continue;
            }
            let q = terms.query_for(&pattern);
            if let Some(snippets) = field(f.name.as_str(), f.aux_id("inference"), &o, leaf, doc, q, src) {
                out.insert(name, Value::Array(snippets.into_iter().map(Value::String).collect()));
            }
        }
    }
    out
}

fn field(name: &str, aux: Option<u16>, o: &HighlightOpts, leaf: &LeafReader<'_>, doc: DocId, q: &Query, src: &Value) -> Option<Vec<String>> {
    let aux = aux?;
    let blob = super::semantic::blob_of(leaf, aux, doc)?;
    let mut queries: Vec<&ChunkQuery> = Vec::new();
    chunk_query::find(q, name, &mut queries);
    let all = has_match_all(q);
    if queries.is_empty() && !all {
        return None;
    }
    let chunks = blob["c"].as_array().cloned().unwrap_or_default();
    if chunks.is_empty() {
        return None;
    }
    // Every chunk is a candidate (the chunk filter is required, the field's
    // queries optional): its score is the sum of the matching queries'.
    let mut scores = vec![0f32; chunks.len()];
    for cq in &queries {
        for (i, s) in cq.chunk_scores(leaf, doc).into_iter().enumerate() {
            if let (Some(s), Some(slot)) = (s, scores.get_mut(i)) {
                *slot += s;
            }
        }
    }
    if all {
        for s in scores.iter_mut() {
            *s += 1.0;
        }
    }
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    // Stable: equal scores keep chunk order.
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let n = if o.number_of_fragments == 0 { 1 } else { o.number_of_fragments };
    order.truncate(n.min(chunks.len()));
    if !o.order_score {
        order.sort();
    }
    let mut cache: std::collections::HashMap<String, Vec<u16>> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for i in order {
        let c = &chunks[i];
        let srcf = c[0].as_str().unwrap_or(name).to_string();
        // A `semantic` media chunk highlights as its input value.
        if let Some(idx) = c.get(3).and_then(Value::as_u64) {
            let vals = crate::highlight::source_values(src, &srcf);
            let v = vals.get(idx as usize).map(|x| match x {
                Value::Object(o) => o.get("value").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default(),
                Value::String(s) => s.clone(),
                o => o.to_string(),
            });
            out.push(v.unwrap_or_default());
            continue;
        }
        let units = cache.entry(srcf.clone()).or_insert_with(|| joined(src, &srcf));
        let (s, e) = (c[1].as_u64().unwrap_or(0) as usize, c[2].as_u64().unwrap_or(0) as usize);
        out.push(units.get(s.min(units.len())..e.min(units.len())).map(String::from_utf16_lossy).unwrap_or_default());
    }
    Some(out)
}
