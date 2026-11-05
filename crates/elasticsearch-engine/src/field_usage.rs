//! Per-field usage counters behind `GET /<index>/_field_usage_stats`.
//!
//! ES counts, per shard, which Lucene structures a search touched for each
//! field. Here the search request is walked once per target index: query
//! leaves hit the inverted index, `knn` the vectors, sort / aggregation
//! fields the doc values, and fetched hits the stored `_id`.

use crate::search::{SearchRequest, Target};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

/// Counters of one field.
#[derive(Clone, Debug, Default)]
pub struct FieldUse {
    pub terms: u64,
    pub postings: u64,
    pub stored_fields: u64,
    pub doc_values: u64,
    pub knn_vectors: u64,
    pub proximity: u64,
    pub positions: u64,
    pub term_frequencies: u64,
    pub points: u64,
    pub norms: u64,
    pub term_vectors: u64,
    pub offsets: u64,
}

impl FieldUse {
    /// ES's `any`: how many times the field was touched at all.
    pub fn any(&self) -> u64 {
        self.terms.max(self.postings) + self.stored_fields + self.doc_values + self.knn_vectors + self.points + self.norms + self.term_vectors
    }
}

static USAGE: LazyLock<Mutex<HashMap<String, BTreeMap<String, FieldUse>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Usage of the index with `uuid` since it was created.
pub fn snapshot(uuid: &str) -> BTreeMap<String, FieldUse> {
    USAGE.lock().ok().and_then(|m| m.get(uuid).cloned()).unwrap_or_default()
}

/// Count one use of `field` (`kind`: `term_vectors`).
pub fn bump(uuid: &str, field: &str, kind: &str) {
    if let Ok(mut m) = USAGE.lock() {
        let u = m.entry(uuid.to_string()).or_default().entry(field.to_string()).or_default();
        if kind == "term_vectors" {
            u.term_vectors += 1;
        }
    }
}

/// Forget an index (deleted).
pub fn forget(uuid: &str) {
    if let Ok(mut m) = USAGE.lock() {
        m.remove(uuid);
    }
}

/// Query types whose body is `{field: ...}` (the field is the first key).
const FIELD_KEYED: &[&str] = &[
    "term", "terms", "match", "match_phrase", "match_phrase_prefix", "match_bool_prefix", "range", "prefix", "wildcard", "regexp", "fuzzy", "terms_set", "intervals",
    "span_term",
];

fn walk_query(v: &Value, f: &mut impl FnMut(&str, &str)) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                if FIELD_KEYED.contains(&k.as_str()) {
                    if let Some(field) = x.as_object().and_then(|o| o.keys().next()) {
                        let kind = match k.as_str() {
                            "match_phrase" | "match_phrase_prefix" | "intervals" | "span_term" => "phrase",
                            "range" => "range",
                            _ => "index",
                        };
                        f(field, kind);
                    }
                } else if k == "exists" {
                    if let Some(field) = x.get("field").and_then(Value::as_str) {
                        f(field, "doc_values");
                    }
                } else if k == "knn" {
                    if let Some(field) = x.get("field").and_then(Value::as_str) {
                        f(field, "knn");
                    }
                    if let Some(flt) = x.get("filter") {
                        walk_query(flt, f);
                    }
                    continue;
                }
                walk_query(x, f);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| walk_query(x, f)),
        _ => {}
    }
}

fn walk_aggs(v: &Value, f: &mut impl FnMut(&str, &str)) {
    if let Value::Object(m) = v {
        for (k, x) in m {
            if k == "field" {
                if let Some(s) = x.as_str() {
                    f(s, "doc_values");
                }
            } else {
                walk_aggs(x, f);
            }
        }
    }
}

/// Record the structures `req` touches on each target. The two phases of
/// a query_then_fetch count once between them: the query phase what the
/// query, sort and aggregations read, the fetch phase the highlighter's
/// offsets and the stored fields.
pub fn record(targets: &[Target], req: &SearchRequest) {
    let (query, fetch) = (req.fetch_docs.is_none() && !req.refetch, !req.doc_refs);
    let mut touched: Vec<(String, &'static str)> = Vec::new();
    let mut add = |field: &str, kind: &str| {
        let kind: &'static str = match kind {
            "index" => "index",
            "phrase" => "phrase",
            "range" => "range",
            "knn" => "knn",
            _ => "doc_values",
        };
        touched.push((field.to_string(), kind));
    };
    if let Some(q) = &req.query {
        walk_query(q, &mut add);
    }
    if let Some(q) = &req.post_filter {
        walk_query(q, &mut add);
    }
    for k in &req.knn {
        if let Some(field) = k.get("field").and_then(Value::as_str) {
            add(field, "knn");
        }
        if let Some(flt) = k.get("filter") {
            walk_query(flt, &mut add);
        }
    }
    for s in &req.sort {
        if !s.field.starts_with('_') {
            add(&s.field, "doc_values");
        }
    }
    if let Some(a) = &req.aggs {
        walk_aggs(a, &mut add);
    }
    // Scores read the norms; a highlighter reads the offsets.
    let scoring = req.sort.is_empty() || req.sort.iter().any(|s| s.field == "_score") || req.track_scores;
    // A query_then_fetch query phase carries the highlight unrendered.
    let highlighted = req.highlight.is_some() && fetch;
    let Ok(mut all) = USAGE.lock() else { return };
    for t in targets {
        let idx = all.entry(t.svc.uuid.clone()).or_default();
        for (field, kind) in &touched {
            let u = idx.entry(field.clone()).or_default();
            if !query {
                if highlighted && matches!(*kind, "index" | "phrase" | "range") && t.svc.lookup().get(field).map_or(true, |f| f.ty.is_text()) {
                    u.offsets += 1;
                }
                continue;
            }
            match *kind {
                "index" | "phrase" | "range" => {
                    let ty = t.svc.lookup().get(field).map(|f| f.ty);
                    let inverted = ty.map_or(true, |ty| ty.is_text() || ty.is_keyword_like());
                    if inverted {
                        let text = ty.map_or(true, |ty| ty.is_text());
                        if text && scoring {
                            u.norms += 1;
                        }
                        if text && highlighted {
                            u.offsets += 1;
                        }
                        u.terms += 1;
                        u.postings += 1;
                        if ty.map_or(true, |ty| ty.is_text()) {
                            u.term_frequencies += 1;
                        }
                        if *kind == "phrase" {
                            u.proximity += 1;
                            u.positions += 1;
                        }
                    } else {
                        u.points += 1;
                        if *kind == "range" {
                            u.doc_values += 1;
                        }
                    }
                }
                "knn" => u.knn_vectors += 1,
                _ => u.doc_values += 1,
            }
        }
        if fetch && req.size > 0 && !req.stored_fields_none {
            idx.entry("_id".to_string()).or_default().stored_fields += 1;
            idx.entry("_source".to_string()).or_default().stored_fields += 1;
        }
    }
}
