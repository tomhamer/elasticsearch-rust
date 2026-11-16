//! Field collapsing extras on top of core's per-index collapse:
//! request validation (ES `CollapseBuilder` / `SearchRequest.validate`),
//! cross-index de-duplication of groups, and the expand phase
//! (`collapse.inner_hits`, ES `ExpandSearchPhase`): one sub-search per
//! returned group (`query` AND `term(group)`, or `must_not exists` for the
//! null group), rendered as `inner_hits.<name>`.

use crate::error::{EsError, Result};
use crate::search::SearchRequest;
use rustic_core::query::TrackTotalHits;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Validate a `collapse` object (`field`, `inner_hits`, `max_concurrent_group_searches`).
pub fn validate(v: &Value) -> Result<()> {
    let o = v.as_object().ok_or_else(|| EsError::parsing("[collapse] must be an object"))?;
    if o.get("field").and_then(Value::as_str).is_none() {
        return Err(crate::search_strict::shard_failure("no collapse field specified"));
    }
    for (k, x) in o {
        match k.as_str() {
            "field" | "inner_hits" => {}
            "max_concurrent_group_searches" => {
                if x.as_i64().map(|n| n < 1).unwrap_or(true) && !x.is_string() {
                    return Err(EsError::illegal_argument("maxConcurrentGroupRequests must be positive, got: ".to_string() + &x.to_string()));
                }
            }
            other => return Err(EsError::parsing(format!("[collapse] unknown field [{other}]"))),
        }
    }
    let ihs = inner_hits_list(v);
    // Nested collapse rules first (ES fails them while parsing the inner hit builder).
    for ih in &ihs {
        if let Some(c) = ih.get("collapse") {
            if c.get("inner_hits").is_some() {
                return Err(EsError::new(400, "parse_exception", "[inner_collapse] failed to parse field [inner_hits]").caused_by(EsError::illegal_argument(
                    "inner_hits field is not supported for the inner collapse",
                )));
            }
            if c.get("collapse").is_some() {
                return Err(EsError::new(400, "parse_exception", "[inner_collapse] unknown field [collapse]"));
            }
        }
    }
    for ih in &ihs {
        if ih.get("name").and_then(Value::as_str).map(|s| s.is_empty()).unwrap_or(true) {
            return Err(EsError::illegal_argument("[inner_hits] must have a [name] when used with [collapse]"));
        }
    }
    Ok(())
}

fn inner_hits_list(v: &Value) -> Vec<Map<String, Value>> {
    match v.get("inner_hits") {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_object().cloned()).collect(),
        Some(Value::Object(o)) => vec![o.clone()],
        _ => Vec::new(),
    }
}

/// Request-level checks that need the whole request.
pub fn check_request(req: &SearchRequest, scroll: bool) -> Result<()> {
    let Some(field) = &req.collapse else { return Ok(()) };
    if scroll {
        return Err(EsError::action_validation(&["cannot use `collapse` in a scroll context"]));
    }
    if req.search_after.is_some() && (req.sort.len() != 1 || &req.sort[0].field != field) {
        return Err(EsError::action_validation(&[
            "Cannot use [collapse] in conjunction with [search_after] unless the search is sorted on the same field. Multiple sort fields are not allowed.",
        ]));
    }
    Ok(())
}

/// The expand phase: add `inner_hits` to every hit of `json` (a complete
/// search response). `run` executes a sub-request over the same targets.
pub fn expand(json: &str, req: &SearchRequest, run: &dyn Fn(&SearchRequest) -> Result<String>) -> Result<Option<String>> {
    let (Some(field), Some(spec)) = (&req.collapse, &req.collapse_spec) else { return Ok(None) };
    let ihs = inner_hits_list(spec);
    if ihs.is_empty() {
        return Ok(None);
    }
    // Lossless: the collapsed and inner hits' sources pass through raw
    // (one side table for the whole response).
    let mut sources = crate::source_json::ResponseSources::default();
    let mut out: Value = sources.parse_lazy(json.as_bytes())?;
    let Some(hits) = out.pointer_mut("/hits/hits").and_then(Value::as_array_mut) else { return Ok(None) };
    if hits.is_empty() {
        return Ok(None);
    }
    let mut cache: HashMap<(String, usize), Value> = HashMap::new();
    // Each sub-search's `timed_out` / `_shards`, merged into the response
    // once its hits are expanded (see [`merge_completeness`]).
    let mut completeness: Vec<Value> = Vec::new();
    for h in hits.iter_mut() {
        let group = h.pointer(&format!("/fields/{}", field.replace('~', "~0").replace('/', "~1"))).and_then(|v| v.get(0)).cloned().unwrap_or(Value::Null);
        let key = group.to_string();
        let mut inner = Map::new();
        for (i, ih) in ihs.iter().enumerate() {
            let name = ih.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            let res = match cache.get(&(key.clone(), i)) {
                Some(v) => v.clone(),
                None => {
                    let sub = sub_request(req, field, &group, ih)?;
                    let s = run(&sub)?;
                    let v: Value = sources.parse_lazy(s.as_bytes())?;
                    completeness.push(json!({ "timed_out": v.get("timed_out").cloned().unwrap_or(Value::Null), "_shards": v.get("_shards").cloned().unwrap_or(Value::Null) }));
                    let hv = v.get("hits").cloned().unwrap_or(Value::Null);
                    cache.insert((key.clone(), i), hv.clone());
                    hv
                }
            };
            inner.insert(name, json!({ "hits": res }));
        }
        if let Some(m) = h.as_object_mut() {
            m.insert("inner_hits".into(), Value::Object(inner));
        }
    }
    for c in &completeness {
        merge_completeness(&mut out, c);
    }
    // Aggregations as the shard rendered them (raw scripted envelopes kept):
    // the tree render only when a part needed it.
    let bytes = sources.render_lazy(&out).unwrap_or_else(|| sources.render(&out));
    Ok(Some(String::from_utf8(bytes).expect("JSON text is UTF-8")))
}

/// Carry an expand sub-search's incompleteness into the search response
/// `resp` (the expand phase runs when partial results are allowed; a
/// sub-search that fails outright fails the request): a timed out
/// sub-search times the response out, and the shard failures it reports
/// join the response's `_shards` (once per shard, counted as failed), so
/// inner hits missing a failed shard's documents are never presented as
/// complete.
pub fn merge_completeness(resp: &mut Value, sub: &Value) {
    if sub.get("timed_out").and_then(Value::as_bool) == Some(true) {
        if let Some(o) = resp.as_object_mut() {
            o.insert("timed_out".into(), Value::Bool(true));
        }
    }
    let Some(failures) = sub.pointer("/_shards/failures").and_then(Value::as_array).filter(|f| !f.is_empty()) else { return };
    let Some(shards) = resp.get_mut("_shards").and_then(Value::as_object_mut) else { return };
    let shard_of = |f: &Value| (f.get("index").cloned().unwrap_or(Value::Null), f.get("shard").cloned().unwrap_or(Value::Null));
    let mut list: Vec<Value> = shards.get("failures").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut added = 0u64;
    for f in failures {
        if !list.iter().any(|x| shard_of(x) == shard_of(f)) {
            list.push(f.clone());
            added += 1;
        }
    }
    if added == 0 {
        return;
    }
    let total = shards.get("total").and_then(Value::as_u64).unwrap_or(0);
    let failed = shards.get("failed").and_then(Value::as_u64).unwrap_or(0).saturating_add(added).min(total.max(1));
    let successful = shards.get("successful").and_then(Value::as_u64).unwrap_or(0).saturating_sub(added);
    shards.insert("successful".into(), json!(successful));
    shards.insert("failed".into(), json!(failed));
    shards.insert("failures".into(), Value::Array(list));
}

fn sub_request(req: &SearchRequest, field: &str, group: &Value, ih: &Map<String, Value>) -> Result<SearchRequest> {
    let group_q = if group.is_null() {
        json!({"bool": {"must_not": [{"exists": {"field": field}}]}})
    } else {
        json!({"term": {field: group}})
    };
    let mut must = Vec::new();
    if let Some(q) = &req.query {
        must.push(q.clone());
    } else if let Some(p) = &req.rank_plan {
        // Compound retriever: groups are expanded over every matching doc.
        must.push(p.union().clone());
    }
    let mut filter = vec![group_q];
    if let Some(pf) = &req.post_filter {
        filter.push(pf.clone());
    }
    let mut body = Map::new();
    body.insert("query".into(), json!({"bool": {"must": must, "filter": filter}}));
    body.insert("from".into(), ih.get("from").cloned().unwrap_or(json!(0)));
    body.insert("size".into(), ih.get("size").cloned().unwrap_or(json!(3)));
    for k in ["sort", "_source", "version", "seq_no_primary_term", "fields", "docvalue_fields", "highlight", "explain", "script_fields", "track_scores", "stored_fields"] {
        if let Some(v) = ih.get(k) {
            body.insert(k.into(), v.clone());
        }
    }
    if let Some(c) = ih.get("collapse") {
        body.insert("collapse".into(), c.clone());
    }
    let mut sub = crate::search::parse_request(&Value::Object(body), &HashMap::new())?;
    sub.rest_total_hits_as_int = req.rest_total_hits_as_int;
    sub.track_total_hits = Some(TrackTotalHits::True);
    sub.runtime_mappings = req.runtime_mappings.clone();
    Ok(sub)
}

/// Key of a group value (for cross-index de-duplication).
pub fn group_key(vals: &[Value]) -> String {
    match vals.first() {
        None => "null".to_string(),
        Some(v) => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(index: &str, shard: u64) -> Value {
        json!({"shard": shard, "index": index, "node": "n", "reason": {"type": "illegal_state_exception", "reason": "boom"}})
    }

    /// Finding 265: an expand sub-search that lost a shard (or timed out)
    /// marks the response incomplete; a complete one changes nothing.
    #[test]
    fn expand_carries_sub_search_failures_into_the_response() {
        let first = json!({
            "took": 1, "timed_out": false,
            "_shards": {"total": 2, "successful": 2, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 2, "relation": "eq"}, "max_score": null, "hits": [
                {"_index": "i", "_id": "1", "_score": null, "fields": {"g": ["a"]}},
                {"_index": "i", "_id": "2", "_score": null, "fields": {"g": ["b"]}}
            ]}
        });
        let req = crate::search::parse_request(&json!({"collapse": {"field": "g", "inner_hits": {"name": "top"}}}), &HashMap::new()).unwrap();
        let sub = |partial: bool| {
            move |_: &SearchRequest| -> Result<String> {
                let (shards, timed_out) = if partial {
                    (json!({"total": 2, "successful": 1, "skipped": 0, "failed": 1, "failures": [failure("i", 1)]}), true)
                } else {
                    (json!({"total": 2, "successful": 2, "skipped": 0, "failed": 0}), false)
                };
                Ok(json!({"took": 1, "timed_out": timed_out, "_shards": shards, "hits": {"total": {"value": 1, "relation": "eq"}, "max_score": 1.0, "hits": []}}).to_string())
            }
        };
        // All healthy: unchanged completeness.
        let out: Value = serde_json::from_str(&expand(&first.to_string(), &req, &sub(false)).unwrap().unwrap()).unwrap();
        assert_eq!(out["_shards"], first["_shards"]);
        assert_eq!(out["timed_out"], false);
        assert!(out["hits"]["hits"][0]["inner_hits"]["top"]["hits"].is_object());
        // A shard of the expansion failed (both groups' sub-searches): one
        // failure for that shard, counted once.
        let out: Value = serde_json::from_str(&expand(&first.to_string(), &req, &sub(true)).unwrap().unwrap()).unwrap();
        assert_eq!(out["timed_out"], true);
        assert_eq!(out["_shards"]["failed"], 1, "{}", out["_shards"]);
        assert_eq!(out["_shards"]["successful"], 1);
        assert_eq!(out["_shards"]["failures"].as_array().map(Vec::len), Some(1));
        assert_eq!(out["_shards"]["failures"][0]["shard"], 1);
    }

    #[test]
    fn merge_completeness_keeps_existing_failures_once() {
        let mut resp = json!({"timed_out": false, "_shards": {"total": 3, "successful": 2, "skipped": 0, "failed": 1, "failures": [failure("i", 0)]}});
        merge_completeness(&mut resp, &json!({"timed_out": false, "_shards": {"total": 3, "successful": 1, "failed": 2, "failures": [failure("i", 0), failure("i", 2)]}}));
        assert_eq!(resp["_shards"]["failed"], 2);
        assert_eq!(resp["_shards"]["successful"], 1);
        assert_eq!(resp["_shards"]["failures"].as_array().map(Vec::len), Some(2));
        assert_eq!(resp["timed_out"], false);
    }
}
