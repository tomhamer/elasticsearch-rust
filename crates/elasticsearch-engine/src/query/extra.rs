//! Less common Query DSL leaves: `distance_feature` and `more_like_this`.

use super::dsl::{expand_fields, parse_msm, scalar_string, QueryContext};
use crate::date::{parse_date_math, Tz};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, ID_FIELD};
use crate::values;
use rustic_core::analysis::Analyzer;
use rustic_core::query::explain::Explanation;
use rustic_core::query::{CustomQuery, LeafReader, MinShouldMatch, Query, Scorer, SearchContext};
use rustic_core::types::{DocId, Term};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

fn parsing(reason: impl Into<String>) -> EsError {
    EsError::parsing(reason)
}

fn shard_err(ctx: &QueryContext, reason: impl Into<String>) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, reason)
}

fn obj<'v>(name: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object().ok_or_else(|| parsing(format!("[{name}] query malformed, no start_object after query name")))
}

/// Validated by `dsl::parse_query` (`valid_boost`) before any of these runs.
fn boost_of(o: &Map<String, Value>) -> f32 {
    o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32
}

// ---------------------------------------------------------------------------
// distance_feature.

#[derive(Debug, Clone, Copy)]
enum Origin {
    /// Dates (millis or nanos, as the field stores them).
    Long { field: u16, origin: i64 },
    /// Geo points: `(lat, lon)`; `geo` = core geo field, else lat/lon columns.
    Geo { geo: Option<u16>, lat: Option<u16>, lon: Option<u16>, origin: (f64, f64) },
}

#[derive(Debug)]
struct DistanceFeature {
    origin: Origin,
    pivot: f64,
    boost: f32,
}

impl DistanceFeature {
    fn distance(&self, leaf: &LeafReader<'_>, doc: DocId) -> Option<f64> {
        match self.origin {
            Origin::Long { field, origin } => leaf.i64_values(field, doc).into_iter().map(|v| (v as f64 - origin as f64).abs()).fold(None, |a: Option<f64>, d| Some(a.map_or(d, |a| a.min(d)))),
            Origin::Geo { geo, lat, lon, origin } => {
                let pts: Vec<(f64, f64)> = match geo {
                    Some(g) if leaf.has_geo(g) => leaf.geo_values(g, doc),
                    _ => match (lat, lon) {
                        (Some(lat), Some(lon)) => leaf.f64_values(lat, doc).into_iter().zip(leaf.f64_values(lon, doc)).collect(),
                        _ => Vec::new(),
                    },
                };
                pts.into_iter().map(|(la, lo)| values::haversine(origin.0, origin.1, la, lo)).fold(None, |a: Option<f64>, d| Some(a.map_or(d, |a| a.min(d))))
            }
        }
    }
}

impl CustomQuery for DistanceFeature {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut docs = Vec::new();
        for d in 0..leaf.num_docs() {
            if leaf.is_deleted(d) {
                continue;
            }
            if let Some(dist) = self.distance(leaf, d) {
                docs.push((d, (self.boost as f64 * self.pivot / (self.pivot + dist)) as f32));
            }
        }
        if docs.is_empty() {
            return None;
        }
        Some(Box::new(rustic_core::query::query::ScoredDocs::new(docs)))
    }
    fn name(&self) -> &'static str {
        "distance_feature"
    }
    fn explain(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        let d = self.distance(leaf, doc).unwrap_or(0.0);
        Explanation::leaf(score, format!("Distance score, computed as weight * pivotDistance / (pivotDistance + abs(value - origin)) from: weight={}, pivotDistance={}, distance={d}", self.boost, self.pivot))
    }
}

/// Time value in nanoseconds (`1h`, `100000000nanos`, `5ms`).
fn time_nanos(s: &str) -> Option<f64> {
    let s = s.trim();
    let split = s.find(|c: char| c.is_ascii_alphabetic())?;
    let (n, unit) = s.split_at(split);
    let n: f64 = n.trim().parse().ok()?;
    let mult = match unit {
        "nanos" => 1.0,
        "micros" => 1e3,
        "ms" => 1e6,
        "s" => 1e9,
        "m" => 60e9,
        "h" => 3600e9,
        "d" => 86400e9,
        _ => return None,
    };
    Some(n * mult)
}

pub fn distance_feature(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("distance_feature", body)?;
    for k in o.keys() {
        if !matches!(k.as_str(), "field" | "origin" | "pivot" | "boost" | "_name") {
            return Err(parsing(format!("[distance_feature] unknown field [{k}]")));
        }
    }
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [field]"))?;
    let origin = o.get("origin").ok_or_else(|| EsError::illegal_argument("Required [origin]"))?;
    let pivot = o.get("pivot").ok_or_else(|| EsError::illegal_argument("Required [pivot]"))?;
    let boost = boost_of(o);
    let Some(f) = ctx.lookup.get(field).cloned() else {
        return Ok(Query::MatchNone);
    };
    let pivot_s = scalar_string(pivot);
    let q = match f.declared {
        EsType::Date | EsType::DateNanos => {
            let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
            let nanos = f.declared == EsType::DateNanos;
            let o_ms: i64 = match origin {
                Value::Number(n) => n.as_i64().unwrap_or(0),
                _ => parse_date_math(&scalar_string(origin), &f.date_format, Tz::UTC, ctx.now, true)
                    .map_err(|e| shard_err(ctx, format!("failed to create query: {}", e.reason)))?,
            };
            let o = if nanos { o_ms * 1_000_000 } else { o_ms };
            let p = time_nanos(&pivot_s).ok_or_else(|| shard_err(ctx, format!("failed to create query: failed to parse setting [distance_feature.pivot] with value [{pivot_s}] as a time value: unit is missing or unrecognized")))?;
            let pivot = if nanos { p } else { p / 1e6 };
            DistanceFeature { origin: Origin::Long { field: id, origin: o }, pivot, boost }
        }
        EsType::GeoPoint => {
            if f.geo_id.is_none() && f.lat_id.is_none() {
                return Ok(Query::MatchNone);
            }
            let (lat, lon) = (f.lat_id, f.lon_id);
            let og = values::parse_geo_point(origin).ok_or_else(|| parsing(format!("failed to parse [origin] [{origin}]")))?;
            let p = values::parse_distance(pivot).ok_or_else(|| parsing(format!("failed to parse [pivot] [{pivot_s}]")))?;
            DistanceFeature { origin: Origin::Geo { geo: f.geo_id, lat, lon, origin: og }, pivot: p, boost }
        }
        _ => {
            return Err(shard_err(
                ctx,
                format!("Asked to search on field [{field}] of type [{}], but only date, date_nanos and geo_point fields are supported", f.declared.name()),
            ))
        }
    };
    Ok(Query::Custom(Arc::new(q)))
}

// ---------------------------------------------------------------------------
// more_like_this.

/// A like/unlike item: free text or a document.
enum Item {
    Text(String),
    /// field values, the `_id` (same-index items, excluded unless `include`),
    /// and the `per_field_analyzer` of an artificial document.
    Doc(BTreeMap<String, Vec<String>>, Option<String>, BTreeMap<String, String>),
}

fn flat_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flat_strings(x, out)),
        Value::Null => {}
        Value::Object(_) => {}
        other => out.push(scalar_string(other)),
    }
}

fn items(v: Option<&Value>, fields: &[String], ctx: &QueryContext) -> Result<Vec<Item>> {
    let list: Vec<&Value> = match v {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(Value::Array(a)) => a.iter().collect(),
        Some(other) => vec![other],
    };
    let mut out = Vec::new();
    for it in list {
        match it {
            Value::String(s) => out.push(Item::Text(s.clone())),
            Value::Object(o) => {
                let index = o.get("_index").and_then(Value::as_str).unwrap_or(ctx.index_name);
                let item_fields: Vec<String> = match o.get("fields").and_then(Value::as_array) {
                    Some(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                    None => fields.to_vec(),
                };
                let mut vals: BTreeMap<String, Vec<String>> = BTreeMap::new();
                if let Some(doc) = o.get("doc") {
                    // ES leaves an artificial doc's field selection unset (every field of the
                    // parsed doc, multi-fields included) unless the item names `fields`.
                    let all_fields: Vec<String>;
                    let doc_fields: &Vec<String> = if o.get("fields").is_some() {
                        &item_fields
                    } else {
                        all_fields = ctx
                            .lookup
                            .matching("*")
                            .into_iter()
                            .filter(|n| ctx.lookup.get(n).is_some_and(|f| (f.ty.is_text() || f.ty.is_keyword_like()) && f.core_id.is_some()))
                            .collect();
                        &all_fields
                    };
                    for f in doc_fields {
                        let mut v = Vec::new();
                        let src_path = match ctx.lookup.get(f) {
                            Some(rf) if rf.is_multi_field => f.rsplit_once('.').map(|x| x.0.to_string()).unwrap_or_else(|| f.clone()),
                            _ => f.clone(),
                        };
                        for x in crate::highlight::source_values(doc, &src_path) {
                            flat_strings(x, &mut v);
                        }
                        if !v.is_empty() {
                            vals.insert(f.clone(), v);
                        }
                    }
                    let pfa: BTreeMap<String, String> = o
                        .get("per_field_analyzer")
                        .and_then(Value::as_object)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), scalar_string(v))).collect())
                        .unwrap_or_default();
                    out.push(Item::Doc(vals, None, pfa));
                } else if let Some(id) = o.get("_id") {
                    let id = scalar_string(id);
                    let routing = o.get("routing").and_then(Value::as_str);
                    if let Some(look) = ctx.terms_lookup {
                        for f in &item_fields {
                            let mut v = Vec::new();
                            for x in look(index, &id, f, routing).unwrap_or_default() {
                                flat_strings(&x, &mut v);
                            }
                            if !v.is_empty() {
                                vals.insert(f.clone(), v);
                            }
                        }
                    }
                    let same_index = index == ctx.index_name || ctx.aliases.iter().any(|a| a == index);
                    out.push(Item::Doc(vals, if same_index { Some(id) } else { None }, BTreeMap::new()));
                } else {
                    return Err(parsing("[more_like_this] query malformed: a like item must have [_id] or [doc]"));
                }
            }
            _ => return Err(parsing("[more_like_this] like items must be strings or objects")),
        }
    }
    Ok(out)
}

pub fn more_like_this(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("more_like_this", body)?;
    for k in o.keys() {
        if !matches!(
            k.as_str(),
            "fields" | "like" | "unlike" | "max_query_terms" | "min_term_freq" | "min_doc_freq" | "max_doc_freq" | "min_word_length" | "max_word_length"
                | "stop_words" | "analyzer" | "minimum_should_match" | "boost_terms" | "include" | "fail_on_unsupported_field" | "boost" | "_name"
                | "routing" | "version" | "version_type"
        ) {
            return Err(parsing(format!("[mlt] query does not support [{k}]")));
        }
    }
    let like_v = o.get("like").ok_or_else(|| parsing("more_like_this requires 'like' to be specified"))?;
    let specs: Vec<String> = match o.get("fields") {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => super::dsl::default_specs(ctx)?,
    };
    let fail_unsupported = o.get("fail_on_unsupported_field").and_then(Value::as_bool).unwrap_or(true);
    let mut fields: Vec<String> = Vec::new();
    for (name, _) in expand_fields(&specs, ctx) {
        match ctx.lookup.get(&name) {
            Some(f) if (f.ty.is_text() || f.ty.is_keyword_like()) && f.core_id.is_some() => fields.push(name),
            Some(f) if fail_unsupported && !name.contains('*') && specs.contains(&name) => {
                return Err(shard_err(ctx, format!("failed to create query: more_like_this only supports text/keyword fields: [{}]", f.name)));
            }
            _ => {}
        }
    }
    let like = items(Some(like_v), &fields, ctx)?;
    if like.is_empty() {
        return Err(EsError::illegal_argument("more_like_this requires 'like' to be specified"));
    }
    let unlike = items(o.get("unlike"), &fields, ctx)?;
    let num = |k: &str, d: i64| o.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))).unwrap_or(d);
    let max_query_terms = num("max_query_terms", 25).max(0) as usize;
    let min_term_freq = num("min_term_freq", 2);
    let min_doc_freq = num("min_doc_freq", 5);
    let max_doc_freq = num("max_doc_freq", i32::MAX as i64);
    let min_word_len = num("min_word_length", 0) as usize;
    let max_word_len = num("max_word_length", 0) as usize;
    let boost_terms = o.get("boost_terms").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let include = o.get("include").and_then(Value::as_bool).unwrap_or(false);
    let stop: Vec<String> = o.get("stop_words").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let analyzer_override: Option<Arc<Analyzer>> = match o.get("analyzer").and_then(Value::as_str) {
        Some(a) => Some(ctx.lookup.registry.analyzer(a).ok_or_else(|| EsError::illegal_argument(format!("failed to find analyzer [{a}]")))?),
        None => None,
    };
    let msm = match o.get("minimum_should_match") {
        Some(v) => parse_msm(v).ok_or_else(|| parsing("[mlt] minimum_should_match must be an integer or a string"))?,
        None => MinShouldMatch::Combination("30%".into()),
    };
    // Per field term frequencies of the like / unlike items. Lucene's
    // `MoreLikeThisQuery` analyzes free text with the *first* field only;
    // documents contribute every requested field (their term vectors, i.e.
    // the field's index analyzer or the item's `per_field_analyzer`).
    let tf_of = |its: &[&Item]| -> BTreeMap<String, BTreeMap<String, i64>> {
        let mut m: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
        let doc_analyze = |f: &str, rf: &crate::mapping::ResolvedField, pfa: &BTreeMap<String, String>, s: &str| -> Vec<String> {
            if let Some(a) = pfa.get(f).and_then(|n| ctx.lookup.registry.analyzer(n).or_else(|| Analyzer::by_name(n).map(Arc::new))) {
                return a.analyze(s).into_iter().map(|t| t.text).collect();
            }
            if rf.ty.is_keyword_like() {
                return vec![match &rf.normalizer {
                    Some(n) => n.analyze(s).into_iter().map(|t| t.text).collect::<String>(),
                    None => s.to_string(),
                }];
            }
            crate::highlight::analyzer_for(rf).analyze(s).into_iter().map(|t| t.text).collect()
        };
        for it in its.iter().copied() {
            match it {
                Item::Text(t) => {
                    let Some(first) = fields.first() else { continue };
                    let Some(rf) = ctx.lookup.get(first) else { continue };
                    let words: Vec<String> = if let Some(a) = &analyzer_override {
                        a.analyze(t).into_iter().map(|t| t.text).collect()
                    } else if rf.ty.is_keyword_like() {
                        vec![match &rf.normalizer {
                            Some(n) => n.analyze(t).into_iter().map(|t| t.text).collect::<String>(),
                            None => t.clone(),
                        }]
                    } else {
                        let a = rf.search_analyzer.clone().unwrap_or_else(|| crate::highlight::analyzer_for(rf));
                        a.analyze(t).into_iter().map(|t| t.text).collect()
                    };
                    let e = m.entry(first.clone()).or_default();
                    for w in words {
                        *e.entry(w).or_insert(0) += 1;
                    }
                }
                Item::Doc(vals, _, pfa) => {
                    for (f, texts) in vals {
                        let Some(rf) = ctx.lookup.get(f) else { continue };
                        let e = m.entry(f.clone()).or_default();
                        for t in texts {
                            for w in doc_analyze(f, rf, pfa, t) {
                                *e.entry(w).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }
        m
    };
    let all_like: Vec<&Item> = like.iter().collect();
    let all_unlike: Vec<&Item> = unlike.iter().collect();
    let unlike_tf = tf_of(&all_unlike);
    let doc_like: Vec<&Item> = like.iter().filter(|i| matches!(i, Item::Doc(..))).collect();
    let text_like: Vec<&Item> = like.iter().filter(|i| matches!(i, Item::Text(_))).collect();
    let _ = &all_like;
    let doc_tf = tf_of(&doc_like);
    let text_tf = tf_of(&text_like);
    let num_docs: u64 = ctx.leaves.iter().map(|l| l.num_live_docs() as u64).sum();
    let mut scored: Vec<(f32, u16, String)> = Vec::new();
    // Lucene walks `HashMap<field, HashMap<word, Int>>`: ties at the cutoff go to the
    // term met first, so iterate in Java hash order.
    let java_cap = |n: usize| {
        let mut c = 16usize;
        while n as f64 > c as f64 * 0.75 {
            c *= 2;
        }
        c
    };
    let doc_freq_of = |id: u16, w: &str| -> u64 {
        let t = Term::new_text(id, w);
        ctx.leaves.iter().map(|l| l.doc_freq(&t) as u64).sum()
    };
    // `top`: free text is analyzed once (first field); each word is scored on the
    // requested field where it is most frequent (first field wins ties).
    let mut gather = |tfm: &BTreeMap<String, BTreeMap<String, i64>>, top: bool| {
        let field_order = crate::fieldtypes::suggest::java_order(tfm.keys().map(|k| (k.clone(), ())).collect(), java_cap(tfm.len()));
        for (fname0, _) in &field_order {
            let terms = &tfm[fname0];
            let word_order = crate::fieldtypes::suggest::java_order(terms.keys().map(|k| (k.clone(), ())).collect(), java_cap(terms.len()));
            for (w, _) in &word_order {
                let tf = &terms[w];
                let (fname, id, df) = if top {
                    let mut best: Option<(String, u16, u64)> = None;
                    for f in &fields {
                        let Some(fid) = ctx.lookup.get(f).and_then(|x| x.core_id) else { continue };
                        let d = doc_freq_of(fid, w);
                        if best.as_ref().map(|b| d > b.2).unwrap_or(true) {
                            best = Some((f.clone(), fid, d));
                        }
                    }
                    let Some((f, fid, d)) = best else { continue };
                    (f, fid, d)
                } else {
                    let Some(fid) = ctx.lookup.get(fname0).and_then(|f| f.core_id) else { continue };
                    (fname0.clone(), fid, doc_freq_of(fid, w))
                };
                let len = w.encode_utf16().count();
                if (min_word_len > 0 && len < min_word_len) || (max_word_len > 0 && len > max_word_len) || stop.contains(w) {
                    continue;
                }
                if unlike_tf.get(&fname).map(|u| u.contains_key(w)).unwrap_or(false) || (top && unlike_tf.get(fname0).map(|u| u.contains_key(w)).unwrap_or(false)) {
                    continue;
                }
                if min_term_freq > 0 && *tf < min_term_freq {
                    continue;
                }
                if min_doc_freq > 0 && (df as i64) < min_doc_freq {
                    continue;
                }
                if df as i64 > max_doc_freq || df == 0 {
                    continue;
                }
                // ClassicSimilarity idf: ln((docCount + 1) / (docFreq + 1)) + 1.
                let idf = (((num_docs as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0) as f32;
                scored.push((*tf as f32 * idf, id, w.clone()));
            }
        }
    };
    gather(&doc_tf, false);
    gather(&text_tf, true);
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_query_terms);
    let mut should = Vec::new();
    let best = scored.first().map(|s| s.0).unwrap_or(1.0);
    for (score, id, w) in &scored {
        let q = Query::Term(Term::new_text(*id, w));
        should.push(if boost_terms > 0.0 { Query::Boost { query: Box::new(q), boost: boost_terms * score / best } } else { q });
    }
    let mut must_not = Vec::new();
    if !include {
        let ids: Vec<Vec<u8>> = like.iter().filter_map(|i| match i {
            Item::Doc(_, Some(id), _) => Some(id.clone().into_bytes()),
            _ => None,
        }).collect();
        if !ids.is_empty() {
            must_not.push(Query::TermSet { field: ID_FIELD, terms: ids, boost: 1.0 });
        }
    }
    if should.is_empty() {
        return Ok(Query::MatchNone);
    }
    let mlt = Query::Bool { must: vec![], should, must_not: vec![], filter: vec![], minimum_should_match: msm };
    let q = if must_not.is_empty() { mlt } else { Query::Bool { must: vec![mlt], should: vec![], must_not, filter: vec![], minimum_should_match: MinShouldMatch::Count(0) } };
    let b = boost_of(o);
    Ok(if (b - 1.0).abs() > f32::EPSILON { Query::Boost { query: Box::new(q), boost: b } } else { q })
}
