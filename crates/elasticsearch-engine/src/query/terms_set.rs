//! `terms_set`: documents matching at least N of the given terms, N taken
//! from a numeric field of the document (`minimum_should_match_field`).
//! Score = sum of the matching terms' scores (Lucene `CoveringQuery`).

use super::dsl::{term_query, QueryContext};
use crate::error::{EsError, Result};
use rustic_core::query::{CustomQuery, DocSet, LeafReader, Query, Scorer, SearchContext, TERMINATED};
use serde_json::{Map, Value};
use std::sync::Arc;

#[derive(Debug)]
struct TermsSet {
    terms: Vec<Query>,
    /// `thresholds[k - 1]`: documents whose `minimum_should_match_field`
    /// (its smallest value) is at most `k`, i.e. that `k` matching terms
    /// satisfy (`None`: a non-numeric field, which no document satisfies).
    thresholds: Vec<Option<Query>>,
    boost: f32,
}

impl CustomQuery for TermsSet {
    /// Lucene `CoveringQuery`'s scorer: the children's iterators merged by
    /// document, each document's count and score computed when it is
    /// reached and its threshold checked then, so the workspace is the
    /// query's width, never its matching documents.
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut kids: Vec<(Box<dyn Scorer>, u32)> = Vec::new();
        for q in &self.terms {
            if let Some(mut s) = q.scorer(leaf, ctx) {
                let d = s.advance();
                if d != TERMINATED {
                    kids.push((s, d));
                }
            }
        }
        if kids.is_empty() {
            return None;
        }
        let thresholds: Vec<Option<Box<dyn Scorer>>> = self.thresholds.iter().map(|q| q.as_ref().and_then(|q| q.scorer(leaf, ctx))).collect();
        Some(Box::new(Covering { kids, thresholds, boost: self.boost, doc: u32::MAX, score: 0.0, started: false }))
    }
    fn children(&self) -> Vec<&Query> {
        self.terms.iter().collect()
    }
    fn name(&self) -> &'static str {
        "terms_set"
    }
}

/// The streaming [`TermsSet`] scorer.
struct Covering {
    /// Each child and its current document (`TERMINATED` once exhausted).
    kids: Vec<(Box<dyn Scorer>, u32)>,
    thresholds: Vec<Option<Box<dyn Scorer>>>,
    boost: f32,
    doc: u32,
    score: f32,
    started: bool,
}

impl DocSet for Covering {
    fn advance(&mut self) -> u32 {
        self.started = true;
        loop {
            let d = self.kids.iter().map(|(_, d)| *d).min().unwrap_or(TERMINATED);
            if d == TERMINATED {
                self.doc = TERMINATED;
                return TERMINATED;
            }
            let (mut n, mut score) = (0usize, 0f32);
            for (s, cur) in self.kids.iter_mut() {
                if *cur == d {
                    n += 1;
                    score += s.score();
                    *cur = s.advance();
                }
            }
            let ok = match self.thresholds.get_mut(n - 1) {
                Some(Some(t)) => t.seek_exact(d),
                _ => false,
            };
            if ok {
                self.doc = d;
                self.score = score * self.boost;
                return d;
            }
        }
    }
    fn doc(&self) -> u32 {
        if self.started { self.doc } else { TERMINATED }
    }
    fn size_hint(&self) -> u32 {
        self.kids.iter().fold(0u32, |a, (s, _)| a.saturating_add(s.size_hint()))
    }
}

impl Scorer for Covering {
    fn score(&mut self) -> f32 {
        self.score
    }
}

pub fn terms_set(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no start_object after query name"))?;
    let (field, spec) = o.iter().next().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no field specified"))?;
    if o.len() > 1 {
        let k2 = o.keys().nth(1).unwrap();
        return Err(EsError::parsing(format!("[terms_set] query doesn't support multiple fields, found [{field}] and [{k2}]")));
    }
    let so: &Map<String, Value> = spec.as_object().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no start_object after field name"))?;
    for k in so.keys() {
        if !matches!(k.as_str(), "terms" | "minimum_should_match_field" | "minimum_should_match_script" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[terms_set] query does not support [{k}]")));
        }
    }
    // ES `doToQuery`: no terms is a match-none query, checked before the
    // minimum-should-match source.
    let empty = Vec::new();
    let terms = match so.get("terms") {
        None => &empty,
        Some(v) => v.as_array().ok_or_else(|| EsError::parsing("[terms_set] query requires a terms array"))?,
    };
    if terms.is_empty() {
        return Ok(Query::MatchNone);
    }
    let boost = so.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let mf = match (so.get("minimum_should_match_field").and_then(Value::as_str), so.get("minimum_should_match_script")) {
        (Some(f), None) => f,
        (None, Some(_)) => return Err(EsError::illegal_argument("[terms_set] minimum_should_match_script is not supported")),
        (Some(_), Some(_)) => return Err(EsError::illegal_argument("[terms_set] query requires exactly one of [minimum_should_match_field] or [minimum_should_match_script]")),
        (None, None) => return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, "failed to create query: No minimum should match has been specified")),
    };
    let Some(f) = ctx.lookup.get(field) else { return Ok(Query::MatchNone) };
    let msm = ctx
        .lookup
        .get(mf)
        .and_then(|m| m.core_id)
        .ok_or_else(|| EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("failed to find minimum_should_match field [{mf}]")))?;
    // Each child is the mapped field's term query (ES `TermsSetQueryBuilder`
    // uses `fieldType.termQuery`), doc values included for a field that is
    // not indexed.
    let qs = terms
        .iter()
        .map(|t| term_query(f, t, false, ctx).map(|q| super::docvalues::rewrite("term", &serde_json::json!({ field: t }), q, ctx)))
        .collect::<Result<Vec<_>>>()?;
    let numeric = ctx.lookup.get(mf).is_some_and(|m| {
        use crate::mapping::EsType::*;
        matches!(m.ty, Long | Integer | Short | Byte | Double | Float | HalfFloat | ScaledFloat | UnsignedLong)
    });
    let _ = msm;
    let thresholds = (1..=qs.len())
        .map(|k| if numeric { super::dsl::parse_query(&serde_json::json!({"range": {mf: {"lte": k}}}), ctx).map(Some) } else { Ok(None) })
        .collect::<Result<Vec<_>>>()?;
    Ok(Query::Custom(Arc::new(TermsSet { terms: qs, thresholds, boost })))
}

#[cfg(test)]
mod covering_tests {
    use serde_json::{json, Value};

    /// Review finding 347: `terms_set` streams its children (Lucene
    /// `CoveringQuery`): each document's count is checked against its own
    /// `minimum_should_match_field` (smallest value), with exact matches
    /// and scores, for one and several terms, a non-numeric field
    /// matching nothing.
    #[test]
    fn covering_scorer_checks_each_documents_threshold() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index("ts", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"tags": {"type": "keyword"}, "need": {"type": "long"}, "kw": {"type": "keyword"}}}})).unwrap();
        let svc = node.get_index("ts").unwrap();
        let docs = [
            ("a1", json!({"tags": ["x"], "need": 1})),
            ("a2", json!({"tags": ["x"], "need": 2})),
            ("b2", json!({"tags": ["x", "y"], "need": 2})),
            ("b3", json!({"tags": ["x", "y"], "need": 3})),
            ("multi", json!({"tags": ["y"], "need": [5, 1]})),
            ("none", json!({"tags": ["x", "y"]})),
            ("zero", json!({"tags": ["z"], "need": 0})),
        ];
        let reqs = docs.iter().map(|(id, d)| crate::index::WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let search = |q: Value, size: usize| -> (u64, Vec<String>) {
            let req = crate::search::parse_request(&json!({"query": q, "size": size, "track_total_hits": true}), &std::collections::HashMap::new()).unwrap();
            let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
            let out = crate::search::execute(&[crate::search::Target::new(node.get_index("ts").unwrap())], &req, &opts).unwrap();
            let v: Value = serde_json::from_str(&out.json).unwrap();
            let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
            ids.sort();
            (v["hits"]["total"]["value"].as_u64().unwrap(), ids)
        };
        let two = json!({"terms_set": {"tags": {"terms": ["x", "y"], "minimum_should_match_field": "need"}}});
        assert_eq!(search(two.clone(), 100).1, vec!["a1", "b2", "multi"]);
        let (total, top) = search(two, 1);
        assert_eq!((total, top.len()), (3, 1), "a small window still counts every match");
        let one = json!({"terms_set": {"tags": {"terms": ["x"], "minimum_should_match_field": "need"}}});
        assert_eq!(search(one, 100).1, vec!["a1"]);
        let z = json!({"terms_set": {"tags": {"terms": ["z", "q"], "minimum_should_match_field": "need"}}});
        assert_eq!(search(z, 100).1, vec!["zero"]);
        let kw = json!({"terms_set": {"tags": {"terms": ["x"], "minimum_should_match_field": "kw"}}});
        assert_eq!(search(kw, 100).0, 0);
    }
}
