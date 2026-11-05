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
    // Aggregations as the shard rendered them (raw scripted envelopes kept):
    // the tree render only when a part needed it.
    let bytes = sources.render_lazy(&out).unwrap_or_else(|| sources.render(&out));
    Ok(Some(String::from_utf8(bytes).expect("JSON text is UTF-8")))
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
