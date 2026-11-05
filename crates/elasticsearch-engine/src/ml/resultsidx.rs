//! The anomaly detection results and state indices.
//!
//! ES keeps every result document of a job (`bucket`, `record`,
//! `influencer`, `model_plot`, `model_forecast`, `model_size_stats`,
//! `data_counts`, `timing_stats`, model snapshot descriptors, categories) in
//! `.ml-anomalies-<results_index_name>-000001` (default `.ml-anomalies-shared-000001`),
//! read through the filtered alias `.ml-anomalies-<job>` and written through
//! `.ml-anomalies-.write-<job>`. Model state and quantiles live in
//! `.ml-state-000001` (alias `.ml-state-write`).

use crate::coord::{self, path_segment, Coordinator};
use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::shard::WriteCond;
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};

pub const STATE_INDEX: &str = ".ml-state-000001";
pub const STATE_WRITE_ALIAS: &str = ".ml-state-write";
pub const STATE_PATTERN: &str = ".ml-state*";

/// `.ml-anomalies-<job>` (the filtered read alias).
pub fn read_alias(job_id: &str) -> String {
    format!(".ml-anomalies-{job_id}")
}

/// `.ml-anomalies-.write-<job>`.
pub fn write_alias(job_id: &str) -> String {
    format!(".ml-anomalies-.write-{job_id}")
}

/// The default shared results index (`.ml-anomalies-shared` + the
/// six-digit rollover suffix every 9.x results index carries).
pub const SHARED_RESULTS_INDEX: &str = ".ml-anomalies-shared-000001";

/// Whether `name` ends with a `-NNNNNN` rollover suffix (`MlIndexAndAlias.has6DigitSuffix`).
pub fn has_six_digit_suffix(name: &str) -> bool {
    let last = name.rsplit('-').next().unwrap_or("");
    last.len() == 6 && last.bytes().all(|b| b.is_ascii_digit())
}

/// The concrete results index of a stored job config: ES 9
/// (`MlIndexAndAlias.ensureValidResultsIndexName`) appends `-000001` to
/// `.ml-anomalies-<results_index_name>` unless it already has a six-digit suffix.
pub fn results_index(job: &Value) -> String {
    let base = format!(".ml-anomalies-{}", job["results_index_name"].as_str().unwrap_or("shared"));
    if has_six_digit_suffix(&base) {
        base
    } else {
        format!("{base}-000001")
    }
}

/// ES's `.ml-anomalies-*` mappings (`anomalydetection/results_index_mappings.json`
/// as an 8.19 node creates them).
pub fn anomalies_mapping() -> Value {
    serde_json::from_str(include_str!("anomalies_mapping.json")).unwrap_or_else(|_| json!({}))
}

/// Field names a job's results add as keyword term fields
/// (`ElasticsearchMappings.addTermFields`): by / over / partition field
/// names and influencers.
pub fn term_fields(job: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |f: &str| {
        if !f.is_empty() && !out.iter().any(|x| x == f) {
            out.push(f.to_string());
        }
    };
    for d in job["analysis_config"]["detectors"].as_array().into_iter().flatten() {
        for k in ["by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(f) = d[k].as_str() {
                add(f);
            }
        }
    }
    for i in job["analysis_config"]["influencers"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        add(i);
    }
    out
}

/// Create the job's results index (if missing) and its read / write
/// aliases (ES `JobResultsProvider.createJobResultIndex`).
pub fn ensure_results_index(node: &Node, id: &str, job: &Value) {
    let index = results_index(job);
    if let Some(c) = coord::current() {
        return ensure_results_index_on(node, &c, id, job, &index);
    }
    let svc = match node.get_index(&index) {
        Some(s) => Some(s),
        None => {
            let body = json!({"settings": {"index": {"hidden": true, "number_of_shards": 1, "auto_expand_replicas": "0-1", "query": {"default_field": "all_field_values"}}},
                "mappings": anomalies_mapping()});
            node.create_index(&index, &body).ok().or_else(|| node.get_index(&index))
        }
    };
    if let Some(svc) = svc {
        let mut props = Map::new();
        let existing = svc.meta().mappings.to_json();
        for f in term_fields(job) {
            if existing["properties"].get(&f).is_none() && !f.contains('.') {
                props.insert(f, json!({"type": "keyword"}));
            }
        }
        if !props.is_empty() {
            let _ = svc.put_mapping(&json!({"properties": props}));
        }
    }
    let _ = node.update_aliases(&[
        json!({"add": {"index": index, "alias": write_alias(id), "is_hidden": true}}),
        json!({"add": {"index": index, "alias": read_alias(id), "filter": {"term": {"job_id": {"value": id}}}, "is_hidden": true}}),
    ]);
}

/// Remove the job's aliases.
pub fn remove_aliases(node: &Node, id: &str, job: &Value) {
    let index = results_index(job);
    if let Some(c) = coord::current() {
        if node.get_index(&index).is_some() {
            let _ = c("POST", "/_aliases", Some(json!({"actions": [
                {"remove": {"index": index, "alias": write_alias(id), "must_exist": false}},
                {"remove": {"index": index, "alias": read_alias(id), "must_exist": false}},
            ]})));
        }
        return;
    }
    if node.get_index(&index).is_some() {
        let _ = node.update_aliases(&[
            json!({"remove": {"index": index, "alias": write_alias(id), "must_exist": false}}),
            json!({"remove": {"index": index, "alias": read_alias(id), "must_exist": false}}),
        ]);
    }
}

/// Create `.ml-state-000001` + `.ml-state-write` when missing.
pub fn ensure_state_index(node: &Node) {
    if node.get_index(STATE_INDEX).is_none() && node.write_index(STATE_WRITE_ALIAS, false).is_err() {
        let body = json!({"settings": {"index": {"hidden": true, "number_of_shards": 1, "auto_expand_replicas": "0-1"}},
            "mappings": {"_meta": {"version": "10000007"}, "enabled": false},
            "aliases": {STATE_WRITE_ALIAS: {"is_hidden": true}}});
        match coord::current() {
            Some(c) => {
                let _ = c("PUT", &format!("/{STATE_INDEX}"), Some(body));
            }
            None => {
                let _ = node.create_index(STATE_INDEX, &body);
            }
        }
    }
}

/// [`ensure_results_index`] through the cluster coordinator.
fn ensure_results_index_on(node: &Node, c: &Coordinator, id: &str, job: &Value, index: &str) {
    if node.get_index(index).is_none() {
        let body = json!({"settings": {"index": {"hidden": true, "number_of_shards": 1, "auto_expand_replicas": "0-1", "query": {"default_field": "all_field_values"}}},
            "mappings": anomalies_mapping()});
        let _ = c("PUT", &format!("/{}", path_segment(index)), Some(body));
    }
    let existing = node.get_index(index).map(|svc| svc.meta().mappings.to_json()).unwrap_or(Value::Null);
    let mut props = Map::new();
    for f in term_fields(job) {
        if existing["properties"].get(&f).is_none() && !f.contains('.') {
            props.insert(f, json!({"type": "keyword"}));
        }
    }
    if !props.is_empty() {
        let _ = c("PUT", &format!("/{}/_mapping", path_segment(index)), Some(json!({"properties": props})));
    }
    let _ = c("POST", "/_aliases", Some(json!({"actions": [
        {"add": {"index": index, "alias": write_alias(id), "is_hidden": true}},
        {"add": {"index": index, "alias": read_alias(id), "filter": {"term": {"job_id": {"value": id}}}, "is_hidden": true}},
    ]})));
}

/// One `_bulk` (with `refresh=true`) through the coordinator; the first
/// failed item's error.
fn bulk_on(c: &Coordinator, lines: String) -> Result<Value> {
    let resp = c("POST", "/_bulk?refresh=true", Some(Value::String(lines)))?;
    if resp["errors"].as_bool() == Some(true) {
        for item in resp["items"].as_array().into_iter().flatten() {
            let op = item.as_object().and_then(|o| o.values().next()).cloned().unwrap_or(Value::Null);
            if let Some(err) = op.get("error").filter(|e| e.is_object()) {
                // (A delete of a missing document has no `error`.)
                let status = op["status"].as_u64().unwrap_or(500) as u16;
                return Err(EsError::new(status, err["type"].as_str().unwrap_or("exception"), err["reason"].as_str().unwrap_or("bulk item failed")));
            }
        }
    }
    Ok(resp)
}

fn index_lines(target: &str, docs: Vec<(String, Value)>) -> String {
    let mut lines = String::new();
    for (id, v) in docs {
        lines.push_str(&json!({"index": {"_index": target, "_id": id}}).to_string());
        lines.push('\n');
        lines.push_str(&v.to_string());
        lines.push('\n');
    }
    lines
}

fn delete_lines(target: &str, ids: &[String]) -> String {
    let mut lines = String::new();
    for id in ids {
        lines.push_str(&json!({"delete": {"_index": target, "_id": id}}).to_string());
        lines.push('\n');
    }
    lines
}

/// Whether `target` (an index or alias) exists in the cluster metadata.
fn exists(node: &Node, target: &str) -> bool {
    node.get_index(target).is_some() || node.write_index(target, false).is_ok()
}

fn write_to(node: &Node, target: &str, docs: Vec<(String, Value)>) -> Result<()> {
    if docs.is_empty() {
        return Ok(());
    }
    if let Some(c) = coord::current() {
        // Fail like the engine would when the target is gone, rather than
        // letting `_bulk` auto-create an index named after the alias.
        node.write_index(target, false)?;
        return bulk_on(&c, index_lines(target, docs)).map(|_| ());
    }
    let svc = node.write_index(target, false)?;
    let reqs: Vec<WriteReq> = docs.into_iter().map(|(id, v)| WriteReq::Index { id: Some(id), source: v.to_string(), routing: None, cond: WriteCond::default() }).collect();
    for r in svc.write(reqs, false) {
        r?;
    }
    svc.refresh()?;
    Ok(())
}

/// Index result documents through the job's write alias (recreating the
/// index / aliases when they were deleted) and refresh.
pub fn write_results(node: &Node, job_id: &str, job: &Value, docs: Vec<(String, Value)>) -> Result<()> {
    if docs.is_empty() {
        return Ok(());
    }
    if node.write_index(&write_alias(job_id), false).is_err() {
        ensure_results_index(node, job_id, job);
    }
    write_to(node, &write_alias(job_id), docs)
}

/// Index model state / quantiles documents.
pub fn write_state(node: &Node, docs: Vec<(String, Value)>) -> Result<()> {
    ensure_state_index(node);
    write_to(node, STATE_WRITE_ALIAS, docs)
}

/// Delete documents by id from the job's results index.
pub fn delete_results(node: &Node, job_id: &str, job: &Value, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    if let Some(c) = coord::current() {
        let target = if exists(node, &write_alias(job_id)) { write_alias(job_id) } else { results_index(job) };
        if !exists(node, &target) {
            return Ok(());
        }
        let _ = bulk_on(&c, delete_lines(&target, ids));
        return Ok(());
    }
    let target = match node.write_index(&write_alias(job_id), false) {
        Ok(s) => s,
        Err(_) => match node.get_index(&results_index(job)) {
            Some(s) => s,
            None => return Ok(()),
        },
    };
    let reqs: Vec<WriteReq> = ids.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
    for r in target.write(reqs, false) {
        let _ = r;
    }
    target.refresh()?;
    Ok(())
}

/// Search an index expression; a missing index answers with no hits.
pub fn search(node: &Node, target: &str, body: &Value) -> Result<Value> {
    if let Some(c) = coord::current() {
        let uri = format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true&expand_wildcards=open,hidden", path_segment(target));
        return match c("POST", &uri, Some(body.clone())) {
            Err(e) if e.kind == "index_not_found_exception" => Ok(json!({"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}})),
            other => other,
        };
    }
    let opts = ResolveOpts { ignore_unavailable: true, allow_no_indices: true, expand_open: true, expand_closed: false, expand_hidden: true };
    let targets = match node.resolve(target, opts) {
        Ok(t) => t,
        Err(e) if e.kind == "index_not_found_exception" => Vec::new(),
        Err(e) => return Err(e),
    };
    if targets.is_empty() {
        return Ok(json!({"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}));
    }
    for t in &targets {
        if !t.svc.is_open() {
            return Err(EsError::new(400, "search_phase_execution_exception", "all shards failed"));
        }
    }
    let req = crate::search::parse_request(body, &Default::default())?;
    let out = crate::search::execute(
        &targets,
        &req,
        &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets().max(65_536), scroll_id: None, pit_id: None },
    )?;
    serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))
}

/// Hits (`_id`, `_source`) of a search.
pub fn hits(resp: &Value) -> Vec<(String, Value)> {
    resp["hits"]["hits"].as_array().into_iter().flatten().map(|h| (h["_id"].as_str().unwrap_or("").to_string(), h["_source"].clone())).collect()
}

/// `hits.total.value`.
pub fn total(resp: &Value) -> u64 {
    resp["hits"]["total"]["value"].as_u64().or_else(|| resp["hits"]["total"].as_u64()).unwrap_or(0)
}

/// Every (id, source) matching `query` in `target` (paged by `from`).
pub fn search_all(node: &Node, target: &str, query: Value, sort: Value) -> Result<Vec<(String, Value)>> {
    let mut out = Vec::new();
    let page = 5000usize;
    loop {
        let body = json!({"from": out.len(), "size": page, "query": query, "sort": sort, "track_total_hits": true});
        let resp = search(node, target, &body)?;
        let h = hits(&resp);
        let n = h.len();
        out.extend(h);
        if n < page || out.len() >= 10_000 {
            break;
        }
    }
    Ok(out)
}

/// Delete every document of `target` matching `query`; the number deleted.
pub fn delete_by_query(node: &Node, target: &str, query: Value) -> u64 {
    if let Some(c) = coord::current() {
        let uri = format!("/{}/_delete_by_query?refresh=true&conflicts=proceed&ignore_unavailable=true&allow_no_indices=true&expand_wildcards=open,hidden", path_segment(target));
        return c("POST", &uri, Some(json!({"query": query}))).ok().and_then(|r| r["deleted"].as_u64()).unwrap_or(0);
    }
    let docs = match search_all(node, target, query, json!(["_doc"])) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let opts = ResolveOpts { ignore_unavailable: true, allow_no_indices: true, expand_open: true, expand_closed: false, expand_hidden: true };
    let targets = node.resolve(target, opts).unwrap_or_default();
    let mut deleted = 0;
    let ids: std::collections::HashSet<String> = docs.into_iter().map(|(id, _)| id).collect();
    for t in targets {
        let reqs: Vec<WriteReq> = ids.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
        for r in t.svc.write(reqs, false) {
            if let Ok(w) = r {
                if w.result == "deleted" {
                    deleted += 1;
                }
            }
        }
        let _ = t.svc.refresh();
    }
    deleted
}

/// Delete an index (a job's dedicated results index).
pub fn delete_index(node: &Node, index: &str) -> Result<()> {
    match coord::current() {
        Some(c) => c("DELETE", &format!("/{}", path_segment(index)), None).map(|_| ()),
        None => node.delete_index(index),
    }
}

/// A document by id from a concrete index / alias.
pub fn get(node: &Node, target: &str, id: &str) -> Option<Value> {
    let resp = search(node, target, &json!({"size": 1, "query": {"ids": {"values": [id]}}})).ok()?;
    hits(&resp).into_iter().next().map(|(_, s)| s)
}

/// Epoch millis of a result timestamp (number, numeric string or date).
pub fn ts_millis(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.parse::<i64>().ok().or_else(|| crate::date::DateFormatter::new("strict_date_optional_time").ok().and_then(|f| f.parse(s).ok())),
        _ => None,
    }
}

/// Java `BigInteger(byte[])` of the 128-bit MurmurHash3 of the
/// concatenated values plus `_<length>` (`MachineLearningField.valuesToId`).
pub fn values_to_id(values: &[Option<&str>]) -> String {
    let combined: String = values.iter().flatten().copied().collect();
    let (h1, h2) = crate::tsdb::murmur3_128(combined.as_bytes(), 0);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&h1.to_be_bytes());
    bytes[8..].copy_from_slice(&h2.to_be_bytes());
    format!("{}_{}", i128::from_be_bytes(bytes), combined.encode_utf16().count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    type Calls = Arc<Mutex<Vec<(String, String, Option<Value>)>>>;

    /// A coordinator that records its calls and answers like a cluster
    /// whose shards hold what the local (sparse) engine does not.
    fn recording(calls: Calls) -> Coordinator {
        Arc::new(move |m: &str, u: &str, b: Option<Value>| {
            calls.lock().unwrap().push((m.to_string(), u.to_string(), b));
            if u.contains("_search") {
                return Ok(json!({"hits": {"total": {"value": 3, "relation": "eq"}, "hits": [{"_id": "a", "_source": {"job_id": "j"}}]}}));
            }
            if u.starts_with("/_bulk") {
                return Ok(json!({"errors": false, "items": []}));
            }
            Ok(json!({"acknowledged": true}))
        })
    }

    #[test]
    fn cluster_results_go_through_the_coordinator() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let calls: Calls = Default::default();
        let job = json!({"job_id": "j", "analysis_config": {"detectors": [{"by_field_name": "airline"}]}});
        crate::coord::scoped(Some(recording(calls.clone())), || {
            // Reads see every shard: the local engine has no results index.
            let r = search(&node, &read_alias("j"), &json!({"size": 1})).unwrap();
            assert_eq!(total(&r), 3);
            ensure_results_index(&node, "j", &job);
        });
        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].0, "POST");
        assert_eq!(calls[0].1, "/.ml-anomalies-j/_search?ignore_unavailable=true&allow_no_indices=true&expand_wildcards=open,hidden");
        assert!(calls.iter().any(|(m, u, _)| m == "PUT" && u == "/.ml-anomalies-shared-000001"));
        assert!(calls.iter().any(|(m, u, b)| m == "PUT" && u == "/.ml-anomalies-shared-000001/_mapping" && b.as_ref().unwrap()["properties"]["airline"]["type"] == "keyword"));
        assert!(calls.iter().any(|(m, u, b)| m == "POST" && u == "/_aliases" && b.as_ref().unwrap()["actions"][1]["add"]["alias"] == ".ml-anomalies-j"));
        // Nothing was created in the local engine behind the cluster's back.
        assert!(node.get_index(SHARED_RESULTS_INDEX).is_none());
    }

    #[test]
    fn cluster_result_writes_are_one_refreshing_bulk() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("res", &json!({"aliases": {"w": {}}})).unwrap();
        let calls: Calls = Default::default();
        crate::coord::scoped(Some(recording(calls.clone())), || {
            write_to(&node, "w", vec![("1".into(), json!({"a": 1})), ("2".into(), json!({"a": 2}))]).unwrap();
            // A missing target fails instead of auto-creating an index.
            assert!(write_to(&node, "missing", vec![("1".into(), json!({}))]).is_err());
        });
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "/_bulk?refresh=true");
        let Some(Value::String(lines)) = &calls[0].2 else { panic!("NDJSON body expected") };
        assert_eq!(lines, "{\"index\":{\"_index\":\"w\",\"_id\":\"1\"}}\n{\"a\":1}\n{\"index\":{\"_index\":\"w\",\"_id\":\"2\"}}\n{\"a\":2}\n");
        // The engine's own copy was not written.
        assert_eq!(node.get_index("res").unwrap().doc_count_realtime().0, 0);
    }

    #[test]
    fn bulk_item_failure_is_an_error() {
        let c: Coordinator = Arc::new(|_, _, _| Ok(json!({"errors": true, "items": [
            {"delete": {"status": 404, "result": "not_found"}},
            {"index": {"status": 400, "error": {"type": "mapper_parsing_exception", "reason": "bad"}}}
        ]})));
        let e = bulk_on(&c, String::new()).unwrap_err();
        assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "mapper_parsing_exception", "bad"));
    }

    #[test]
    fn values_to_id_matches_es() {
        // ES 8.19: by_field_value "JZA" / "AAL"; no values -> "0_0".
        assert_eq!(values_to_id(&[Some("JZA")]), "53516671625173823515115038745671684847_3");
        assert_eq!(values_to_id(&[Some("AAL")]), "136042187336583395197382916679594696967_3");
        assert_eq!(values_to_id(&[None, None]), "0_0");
    }
}
