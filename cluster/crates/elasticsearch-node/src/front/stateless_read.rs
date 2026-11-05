//! Reads of stateless shards without a search copy.
//!
//! On a stateless cluster every read (search, count, msearch, get, mget,
//! explain, termvectors, field caps, validate, terms enum, disk usage, ...)
//! goes to search copies only (ES `ShardIterator.allSearchableShards` /
//! `OperationRouting.searchShards` keep only `ShardRouting.isSearchable`
//! copies; the indexing copy is `INDEX_ONLY`). A shard whose search copies
//! are all unassigned (or still recovering) is then treated exactly like an
//! unassigned shard: the responses below are ES 9.5.5's for that case
//! (`AbstractSearchAsyncAction.failOnUnavailable`,
//! `TransportSingleShardAction` "No shard available for [...]", ...).

use super::{RestReq, RestResp};
use elasticsearch_cluster::state::ClusterState;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// A `(index, shard)` with no search copy.
pub type Missing = (String, u32);

/// One `_shards.failures` / `failed_shards` entry of an unavailable shard.
pub fn search_failure(index: &str, shard: u32) -> Value {
    json!({"shard": shard, "index": index, "node": null, "reason": {"type": "no_shard_available_action_exception", "reason": null}})
}

/// `SearchPhaseExecutionException("all shards failed")` when no target
/// shard has a search copy (`failed_shards` grouped per index, as
/// `ExceptionsHelper.groupBy` does for one cause).
pub fn all_shards_failed(missing: &[Missing]) -> RestResp {
    let mut seen = BTreeSet::new();
    let failed: Vec<Value> = missing.iter().filter(|(i, _)| seen.insert(i.clone())).map(|(i, k)| search_failure(i, *k)).collect();
    RestResp::json(503, &json!({"error": {
        "root_cause": [{"type": "no_shard_available_action_exception", "reason": null}],
        "type": "search_phase_execution_exception", "reason": "all shards failed", "phase": "query", "grouped": true,
        "failed_shards": failed,
    }, "status": 503}))
}

/// `allow_partial_search_results=false` with missing shards
/// (`AbstractSearchAsyncAction` "Search rejected due to missing shards").
pub fn missing_shards_rejected(missing: &[Missing]) -> RestResp {
    let list = missing.iter().map(|(i, k)| format!("[{i}][{k}]")).collect::<Vec<_>>().join(", ");
    let reason = format!("Search rejected due to missing shards [{list}]. Consider using `allow_partial_search_results` setting to bypass this error.");
    RestResp::json(503, &json!({"error": {
        "root_cause": [], "type": "search_phase_execution_exception", "reason": "", "phase": "query", "grouped": true, "failed_shards": [],
        "caused_by": {"type": "search_phase_execution_exception", "reason": reason, "phase": "query", "grouped": true, "failed_shards": []},
    }, "status": 503}))
}

/// Add the missing shards to a successful search / count response's
/// `_shards` (`total` and `failed` up, one failure each).
pub fn add_missing(mut v: Value, missing: &[Missing]) -> Value {
    let target = if v.get("_shards").is_some() { Some(&mut v) } else { v.get_mut("response").filter(|r| r.get("_shards").is_some()) };
    let Some(t) = target else { return v };
    let s = &mut t["_shards"];
    let n = missing.len() as u64;
    s["total"] = json!(s["total"].as_u64().unwrap_or(0) + n);
    s["failed"] = json!(s["failed"].as_u64().unwrap_or(0) + n);
    let mut failures = s.get("failures").and_then(Value::as_array).cloned().unwrap_or_default();
    failures.extend(missing.iter().map(|(i, k)| search_failure(i, *k)));
    if let Some(o) = s.as_object_mut() {
        // `failures` follows the counters, as ES renders `_shards`.
        o.remove("failures");
        o.insert("failures".into(), Value::Array(failures));
    }
    v
}

/// `NoShardAvailableActionException` of a single-shard read
/// (`TransportSingleShardAction`): "No shard available for [<request>]".
pub fn single_shard_unavailable(request: &str) -> Value {
    let reason = format!("No shard available for [{request}]");
    json!({"root_cause": [{"type": "no_shard_available_action_exception", "reason": reason}], "type": "no_shard_available_action_exception", "reason": reason})
}

fn error_resp(error: Value) -> RestResp {
    RestResp::json(503, &json!({"error": error, "status": 503}))
}

/// The request names of the single-shard reads (`toString()` of the shard
/// request): `GetRequest` renders itself, the others fall back to
/// `Object.toString` of an unset task (`<class>/unset`).
pub fn get_request(index: &str, id: &str, routing: Option<&str>) -> String {
    format!("get [{index}][{id}]: routing [{}]", routing.unwrap_or("null"))
}
pub const EXPLAIN_REQUEST: &str = "org.elasticsearch.action.explain.ExplainRequest/unset";
pub const TERMVECTORS_REQUEST: &str = "org.elasticsearch.action.termvectors.TermVectorsRequest/unset";
pub const MGET_SHARD_REQUEST: &str = "org.elasticsearch.action.get.MultiGetShardRequest/unset";
pub const MTERMVECTORS_SHARD_REQUEST: &str = "org.elasticsearch.action.termvectors.MultiTermVectorsShardRequest/unset";

/// A `_mget` / `_mtermvectors` item whose shard has no search copy.
pub fn multi_doc_item(op: &str, index: &str, id: &Value) -> Value {
    let request = if op == "_mtermvectors" { MTERMVECTORS_SHARD_REQUEST } else { MGET_SHARD_REQUEST };
    json!({"_index": index, "_id": id, "error": single_shard_unavailable(request)})
}

/// `NoShardAvailableActionException(shardId, "primary shard is not
/// active")`: a stateless realtime get / mget / termvectors needs the
/// indexing shard (ES `TransportGetAction.getCurrentNodeOfPrimary`: the
/// translog lookup runs there).
pub fn primary_inactive(index: &str, uuid: &str, shard: u32) -> Value {
    let e = json!({"type": "no_shard_available_action_exception", "reason": "primary shard is not active", "index_uuid": uuid, "shard": shard.to_string(), "index": index});
    let mut o = serde_json::Map::new();
    o.insert("root_cause".into(), json!([e.clone()]));
    for (k, v) in e.as_object().into_iter().flatten() {
        o.insert(k.clone(), v.clone());
    }
    Value::Object(o)
}

/// Realtime is the default of get, mget and termvectors.
pub fn realtime(req: &RestReq) -> bool {
    req.params().get("realtime").map(String::as_str) != Some("false")
}

/// A stateless realtime document read whose shard's primary is not active
/// fails (ES); `shards` are the `(index, shard)` the request reads, `docs`
/// the `_mget` / `_mtermvectors` items (`(index, id)` per item, in order).
pub fn realtime_without_primary(req: &RestReq, state: &ClusterState, shards: &[Missing]) -> Option<RestResp> {
    if !realtime(req) {
        return None;
    }
    let inactive = |i: &str, k: u32| state.routing_table.shard_table(i, k).is_some_and(|t| t.is_stateless(&state.nodes) && t.active_primary().is_none());
    let uuid = |i: &str| state.metadata.indices.get(i).map(|m| m.uuid.clone()).unwrap_or_default();
    match classify(req) {
        ReadKind::Get { .. } | ReadKind::TermVectors => {
            let (i, k) = shards.iter().find(|(i, k)| inactive(i, *k))?;
            Some(error_resp(primary_inactive(i, &uuid(i), *k)))
        }
        // Single-shard targets (one shard 0 each): every item fails when
        // every target's primary is inactive.
        ReadKind::MultiDoc(_) if !shards.is_empty() && shards.iter().all(|(i, k)| inactive(i, *k)) => {
            let (i0, _) = &shards[0];
            let body = req.json().unwrap_or(json!({}));
            let items: Vec<(String, Value)> = match body.get("docs").and_then(Value::as_array) {
                Some(d) => d.iter().map(|d| (d.get("_index").and_then(Value::as_str).unwrap_or(i0).to_string(), d.get("_id").cloned().unwrap_or(Value::Null))).collect(),
                None => body.get("ids").and_then(Value::as_array).map(|ids| ids.iter().map(|id| (i0.clone(), id.clone())).collect()).unwrap_or_default(),
            };
            let docs: Vec<Value> = items.iter().map(|(i, id)| json!({"_index": i, "_id": id, "error": primary_inactive(i, &uuid(i), 0)})).collect();
            Some(RestResp::json(200, &json!({"docs": docs})))
        }
        _ => None,
    }
}

/// What kind of read a request is, for its unavailable-shard response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadKind {
    /// `_search`, `_count`, `_async_search` submit, `_knn_search`.
    Search,
    Get { id: String },
    Explain,
    TermVectors,
    MultiDoc(String),
    FieldCaps,
    Validate,
    TermsEnum,
    DiskUsage,
    Other,
}

pub fn classify(req: &RestReq) -> ReadKind {
    let segs = req.segments();
    let last = segs.last().map(String::as_str).unwrap_or("");
    if segs.iter().any(|s| s == "_mget" || s == "_mtermvectors") {
        return ReadKind::MultiDoc(segs.iter().find(|s| *s == "_mget" || *s == "_mtermvectors").cloned().unwrap_or_default());
    }
    if segs.iter().any(|s| s == "_explain") {
        return ReadKind::Explain;
    }
    if segs.iter().any(|s| s == "_termvectors") {
        return ReadKind::TermVectors;
    }
    if segs.len() == 3 && (segs[1] == "_doc" || segs[1] == "_source") {
        return ReadKind::Get { id: segs[2].clone() };
    }
    match last {
        "_search" | "_count" | "_knn_search" => ReadKind::Search,
        "_async_search" if req.method() == axum::http::Method::POST => ReadKind::Search,
        "_field_caps" => ReadKind::FieldCaps,
        "query" if segs.iter().any(|s| s == "_validate") => ReadKind::Validate,
        "_terms_enum" => ReadKind::TermsEnum,
        "_disk_usage" => ReadKind::DiskUsage,
        _ => ReadKind::Other,
    }
}

/// The ES response for a read whose every target shard in `missing` lacks
/// a search copy and no target shard has one (`None`: the request also
/// reads shards that do have one, or is a kind this module does not
/// answer; the caller runs the rest).
pub fn all_missing_response(req: &RestReq, state: &ClusterState, missing: &[Missing]) -> Option<RestResp> {
    let params = req.params();
    let (index, shard) = missing.first()?.clone();
    let uuid = state.metadata.indices.get(&index).map(|m| m.uuid.clone()).unwrap_or_default();
    Some(match classify(req) {
        ReadKind::Search => {
            if params.get("allow_partial_search_results").map(String::as_str) == Some("false") {
                missing_shards_rejected(missing)
            } else {
                all_shards_failed(missing)
            }
        }
        ReadKind::Get { id } => error_resp(single_shard_unavailable(&get_request(&index, &id, params.get("routing").map(String::as_str)))),
        ReadKind::Explain => error_resp(single_shard_unavailable(EXPLAIN_REQUEST)),
        ReadKind::TermVectors => error_resp(single_shard_unavailable(TERMVECTORS_REQUEST)),
        ReadKind::MultiDoc(op) => {
            let body = req.json().unwrap_or(json!({}));
            let docs: Vec<Value> = match body.get("docs").and_then(Value::as_array) {
                Some(d) => d.iter().map(|d| multi_doc_item(&op, d.get("_index").and_then(Value::as_str).unwrap_or(&index), d.get("_id").unwrap_or(&Value::Null))).collect(),
                None => body.get("ids").and_then(Value::as_array).map(|ids| ids.iter().map(|id| multi_doc_item(&op, &index, id)).collect()).unwrap_or_default(),
            };
            RestResp::json(200, &json!({"docs": docs}))
        }
        ReadKind::FieldCaps => {
            let reason = format!("index [{index}] has no active shard copy");
            error_resp(json!({"root_cause": [{"type": "no_shard_available_action_exception", "reason": reason}], "type": "no_shard_available_action_exception", "reason": reason}))
        }
        ReadKind::Validate => {
            let failures: Vec<Value> = missing.iter().map(|(i, k)| {
                let u = state.metadata.indices.get(i).map(|m| m.uuid.clone()).unwrap_or_default();
                json!({"shard": k, "index": i, "status": "SERVICE_UNAVAILABLE", "reason": {"type": "no_shard_available_action_exception", "reason": null, "index_uuid": u, "shard": k.to_string(), "index": i}})
            }).collect();
            let n = missing.len();
            RestResp::json(200, &json!({"_shards": {"total": n, "successful": 0, "failed": n, "failures": failures}, "valid": true}))
        }
        ReadKind::TermsEnum => RestResp::json(200, &json!({"_shards": {"total": 0, "successful": 0, "failed": 0}, "terms": [], "complete": true})),
        ReadKind::DiskUsage => {
            let e = json!({"type": "no_shard_available_action_exception", "reason": null, "index_uuid": uuid, "shard": shard.to_string(), "index": index});
            let mut err = e.clone();
            err["root_cause"] = json!([e]);
            // `root_cause` first, as ES renders it.
            let mut o = serde_json::Map::new();
            o.insert("root_cause".into(), err["root_cause"].clone());
            for (k, v) in e.as_object().into_iter().flatten() {
                o.insert(k.clone(), v.clone());
            }
            error_resp(Value::Object(o))
        }
        ReadKind::Other => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, uri: &str, body: &str) -> RestReq {
        RestReq::try_new(method,uri,&[],body.as_bytes(),0,Default::default(),None).unwrap()
    }

    fn body(r: &RestResp) -> Value {
        r.body_json().unwrap()
    }

    #[test]
    fn classifies_reads() {
        assert_eq!(classify(&req("GET", "/i/_search", "")), ReadKind::Search);
        assert_eq!(classify(&req("POST", "/i/_count", "")), ReadKind::Search);
        assert_eq!(classify(&req("POST", "/i/_async_search", "")), ReadKind::Search);
        assert_eq!(classify(&req("GET", "/_async_search/abc", "")), ReadKind::Other);
        assert_eq!(classify(&req("GET", "/i/_doc/7", "")), ReadKind::Get { id: "7".into() });
        assert_eq!(classify(&req("GET", "/i/_source/7", "")), ReadKind::Get { id: "7".into() });
        assert_eq!(classify(&req("GET", "/i/_explain/7", "")), ReadKind::Explain);
        assert_eq!(classify(&req("GET", "/i/_termvectors/7", "")), ReadKind::TermVectors);
        assert_eq!(classify(&req("GET", "/i/_mget", "")), ReadKind::MultiDoc("_mget".into()));
        assert_eq!(classify(&req("GET", "/i/_validate/query", "")), ReadKind::Validate);
        assert_eq!(classify(&req("GET", "/i/_field_caps", "")), ReadKind::FieldCaps);
    }

    /// The bodies below are ES 9.5.5's for an index whose only shard has no
    /// readable copy (recorded from the reference distribution).
    #[test]
    fn unavailable_responses_match_es() {
        let state = ClusterState::empty("c");
        let m = vec![("una".to_string(), 0)];
        let r = all_missing_response(&req("GET", "/una/_search", ""), &state, &m).unwrap();
        assert_eq!(r.status, 503);
        assert_eq!(body(&r), serde_json::from_str::<Value>(r#"{"error":{"root_cause":[{"type":"no_shard_available_action_exception","reason":null}],"type":"search_phase_execution_exception","reason":"all shards failed","phase":"query","grouped":true,"failed_shards":[{"shard":0,"index":"una","node":null,"reason":{"type":"no_shard_available_action_exception","reason":null}}]},"status":503}"#).unwrap());
        let r = all_missing_response(&req("GET", "/una/_doc/1", ""), &state, &m).unwrap();
        assert_eq!(body(&r).to_string(), r#"{"error":{"root_cause":[{"type":"no_shard_available_action_exception","reason":"No shard available for [get [una][1]: routing [null]]"}],"type":"no_shard_available_action_exception","reason":"No shard available for [get [una][1]: routing [null]]"},"status":503}"#);
        let r = all_missing_response(&req("GET", "/una/_explain/1", "{}"), &state, &m).unwrap();
        assert_eq!(body(&r)["error"]["reason"], "No shard available for [org.elasticsearch.action.explain.ExplainRequest/unset]");
        let r = all_missing_response(&req("GET", "/una/_termvectors/1", ""), &state, &m).unwrap();
        assert_eq!(body(&r)["error"]["reason"], "No shard available for [org.elasticsearch.action.termvectors.TermVectorsRequest/unset]");
        let r = all_missing_response(&req("POST", "/una/_mget", r#"{"docs":[{"_index":"una","_id":"1"}]}"#), &state, &m).unwrap();
        assert_eq!(body(&r).to_string(), r#"{"docs":[{"_index":"una","_id":"1","error":{"root_cause":[{"type":"no_shard_available_action_exception","reason":"No shard available for [org.elasticsearch.action.get.MultiGetShardRequest/unset]"}],"type":"no_shard_available_action_exception","reason":"No shard available for [org.elasticsearch.action.get.MultiGetShardRequest/unset]"}}]}"#);
        let r = all_missing_response(&req("GET", "/una/_search?allow_partial_search_results=false", ""), &state, &m).unwrap();
        assert_eq!(body(&r).to_string(), r#"{"error":{"root_cause":[],"type":"search_phase_execution_exception","reason":"","phase":"query","grouped":true,"failed_shards":[],"caused_by":{"type":"search_phase_execution_exception","reason":"Search rejected due to missing shards [[una][0]]. Consider using `allow_partial_search_results` setting to bypass this error.","phase":"query","grouped":true,"failed_shards":[]}},"status":503}"#);
        let r = all_missing_response(&req("GET", "/una/_field_caps?fields=*", ""), &state, &m).unwrap();
        assert_eq!(body(&r)["error"]["reason"], "index [una] has no active shard copy");
        let r = all_missing_response(&req("GET", "/una/_terms_enum", ""), &state, &m).unwrap();
        assert_eq!(body(&r).to_string(), r#"{"_shards":{"total":0,"successful":0,"failed":0},"terms":[],"complete":true}"#);
        let r = all_missing_response(&req("POST", "/una/_disk_usage?run_expensive_tasks=true", ""), &state, &m).unwrap();
        assert_eq!(body(&r).to_string(), r#"{"error":{"root_cause":[{"type":"no_shard_available_action_exception","reason":null,"index_uuid":"","shard":"0","index":"una"}],"type":"no_shard_available_action_exception","reason":null,"index_uuid":"","shard":"0","index":"una"},"status":503}"#);
        assert!(all_missing_response(&req("GET", "/una/_stats", ""), &state, &m).is_none());
    }

    #[test]
    fn primary_inactive_matches_es() {
        assert_eq!(primary_inactive("s", "u1", 0).to_string(), r#"{"root_cause":[{"type":"no_shard_available_action_exception","reason":"primary shard is not active","index_uuid":"u1","shard":"0","index":"s"}],"type":"no_shard_available_action_exception","reason":"primary shard is not active","index_uuid":"u1","shard":"0","index":"s"}"#);
        assert!(realtime(&req("GET", "/s/_doc/1", "")));
        assert!(!realtime(&req("GET", "/s/_doc/1?realtime=false", "")));
    }

    #[test]
    fn partial_responses_add_the_missing_shards() {
        let ok = json!({"took": 4, "timed_out": false, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}, "hits": {"total": {"value": 1, "relation": "eq"}, "max_score": null, "hits": []}});
        let v = add_missing(ok, &[("part".into(), 1)]);
        assert_eq!(v["_shards"].to_string(), r#"{"total":2,"successful":1,"skipped":0,"failed":1,"failures":[{"shard":1,"index":"part","node":null,"reason":{"type":"no_shard_available_action_exception","reason":null}}]}"#);
        let count = add_missing(json!({"count": 1, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}}), &[("part".into(), 1)]);
        assert_eq!(count["_shards"]["failed"], 1);
        let grouped = all_shards_failed(&[("a".into(), 0), ("a".into(), 1), ("b".into(), 0)]);
        assert_eq!(body(&grouped)["error"]["failed_shards"].as_array().unwrap().len(), 2, "grouped per index");
    }
}
