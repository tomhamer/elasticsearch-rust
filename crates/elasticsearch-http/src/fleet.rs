//! Fleet APIs (`x-pack/plugin/fleet`), as Fleet Server calls them:
//! `GET /{index}/_fleet/global_checkpoints`, `/{index}/_fleet/_fleet_search`,
//! `[/{index}]/_fleet/_fleet_msearch` and the secrets store
//! (`POST /_fleet/secret`, `GET|DELETE /_fleet/secret/{id}`).

use crate::search_api::run_search;
use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::fleet;
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::node::{Node, ResolveOpts, NODE_ID};
use elasticsearch_engine::search::{self, ExecOpts, Target};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let m = req.method.clone();
    let off = usize::from(index.is_some());
    let rest: Vec<String> = req.segs.iter().skip(off + 1).cloned().collect();
    let rest: Vec<&str> = rest.iter().map(String::as_str).collect();
    match (index, rest.as_slice()) {
        (Some(index), ["global_checkpoints"]) => match m {
            Method::GET => blocking(move || global_checkpoints(&node, &req, &index)).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        (Some(index), ["_fleet_search"]) => match m {
            Method::GET | Method::POST => blocking(move || fleet_search(&node, &req, &index)).await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        (index, ["_fleet_msearch"]) => match m {
            Method::GET | Method::POST => blocking(move || fleet_msearch(&node, &req, index.as_deref())).await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        // ES answers a POST here with "no handler" and other methods with a
        // 405 naming POST.
        (None, ["_fleet_search"]) if m == Method::POST => crate::not_found_handler(&req),
        (None, ["_fleet_search"]) => method_not_allowed(&req, &["POST"]),
        (None, ["secret"]) => match m {
            Method::POST => blocking(move || post_secret(&node, &req)).await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        (None, ["secret", id]) => {
            let id = id.to_string();
            match m {
                Method::GET => blocking(move || Ok(Resp::ok(get_secret(&node, &req, &id)?))).await,
                Method::DELETE => blocking(move || {
                    delete_secret(&node, &req, &id)?;
                    Ok(Resp::ok(json!({"deleted": true})))
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "DELETE"]),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

// ------------------------------------------------------------ params

/// Comma separated longs (`checkpoints`, `wait_for_checkpoints`).
fn long_list(v: Option<&str>) -> Result<Vec<i64>, EsError> {
    let Some(v) = v else { return Ok(Vec::new()) };
    v.split(',').filter(|s| !s.is_empty()).map(parse_long).collect()
}

fn parse_long(s: &str) -> Result<i64, EsError> {
    s.trim().parse::<i64>().map_err(|_| EsError::new(400, "number_format_exception", format!("For input string: \"{s}\"")))
}

fn time_param(req: &Req, k: &str) -> Result<Option<Duration>, EsError> {
    match req.param(k) {
        None => Ok(None),
        Some(v) => elasticsearch_engine::date::parse_time_value(v)
            .filter(|_| v == "-1" || v == "0" || v.chars().last().is_some_and(|c| c.is_ascii_alphabetic()))
            .map(|ms| Some(Duration::from_millis(ms.max(0) as u64)))
            .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{v}] as a time value: unit is missing or unrecognized"))),
    }
}

// ------------------------------------------------------------ global checkpoints

fn global_checkpoints(node: &Node, req: &Req, index: &str) -> R {
    let wait_for_advance = req.flag("wait_for_advance")?;
    let wait_for_index = req.flag("wait_for_index")?;
    let checkpoints = long_list(req.param("checkpoints"))?;
    let timeout = time_param(req, "timeout")?.unwrap_or(DEFAULT_TIMEOUT);
    let mut errs: Vec<String> = Vec::new();
    if checkpoints.iter().any(|c| *c < -1) {
        errs.push(format!("All checkpoints must be >= -1. Found: [{}]", checkpoints.iter().map(i64::to_string).collect::<Vec<_>>().join(", ")));
    }
    if wait_for_index && !wait_for_advance {
        errs.push("If wait_for_index is set to true, wait_for_advance must also be set to true.".into());
    }
    if !errs.is_empty() {
        return Err(EsError::action_validation(&errs.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    let (timed_out, cps) = fleet::get_global_checkpoints(node, index, wait_for_advance, wait_for_index, &checkpoints, timeout)?;
    Ok(Resp::ok(json!({"timed_out": timed_out, "global_checkpoints": cps})))
}

// ------------------------------------------------------------ fleet search

/// Shard failures of a `wait_for_checkpoints` pre-check: (shard, reason).
type Failures = Vec<(u32, String)>;

fn single_index_error(exprs: &[&str]) -> EsError {
    EsError::illegal_argument(format!("Fleet search API only supports searching a single index. Found: [[{}]].", exprs.join(", ")))
}

/// Resolve the target of a `wait_for_checkpoints` search, check the
/// checkpoints against its shards, and wait (up to `timeout` overall) for
/// each shard's copy to have refreshed up to its checkpoint.
fn check_checkpoints(node: &Node, expr: &str, checkpoints: &[i64], timeout: Duration) -> Result<(Arc<IndexService>, Failures), EsError> {
    let svc = fleet::single_index(node, expr).map_err(|e| {
        EsError::illegal_argument(format!("Failed to resolve wait_for_checkpoints target [{expr}]. Configured target must resolve to a single open index.")).caused_by(e)
    })?;
    let n = svc.number_of_shards() as usize;
    if n != checkpoints.len() {
        return Err(EsError::illegal_argument(format!(
            "Target configured with wait_for_checkpoints must search the same number of shards as checkpoints provided. [{}] checkpoints provided. Target [{expr}] which resolved to index [{}] has [{n}] shards.",
            checkpoints.len(),
            svc.name
        )));
    }
    let mut failures = Vec::new();
    let deadline = Instant::now() + timeout.min(fleet::MAX_WAIT);
    for (i, cp) in checkpoints.iter().enumerate() {
        let shard = svc.shard(i as u32);
        if let Some(r) = fleet::checkpoint_failure(&svc, shard.as_deref(), *cp) {
            failures.push((i as u32, r));
        } else if let Some(r) = shard.as_deref().and_then(|s| fleet::wait_refreshed(&svc, s, *cp, deadline.saturating_duration_since(Instant::now()))) {
            failures.push((i as u32, r));
        }
    }
    Ok((svc, failures))
}

fn shard_failure_error(index: &str, failures: &Failures, all_failed: bool) -> (u16, Value) {
    fleet::shard_failure_error(index, failures, all_failed)
}

/// Search the shards that passed the checkpoint check and report the
/// others as failed (`allow_partial_search_results: true`).
fn partial_search(node: &Node, body: &Value, params: &HashMap<String, String>, svc: &Arc<IndexService>, failures: &Failures) -> Result<String, EsError> {
    let failed: Vec<u32> = failures.iter().map(|f| f.0).collect();
    let targets: Vec<Target> = (0..svc.number_of_shards())
        .filter(|n| !failed.contains(n))
        .filter_map(|n| svc.shard(n))
        .map(|s| {
            let mut t = Target::new(svc.clone());
            t.shard = Some(s);
            t
        })
        .collect();
    let sreq = search::parse_request(body, params)?;
    let out = search::execute(&targets, &sreq, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
    let total = svc.number_of_shards() as usize;
    let shards = json!({
        "total": total, "successful": total - failures.len(), "skipped": 0, "failed": failures.len(),
        "failures": failures.iter().map(|(s, r)| json!({"shard": s, "index": svc.name, "node": NODE_ID, "reason": {"type": "illegal_argument_exception", "reason": r}})).collect::<Vec<_>>(),
    });
    elasticsearch_engine::source_json::with_member(&out.json, "_shards", &shards.to_string(), false)
        .ok_or_else(|| EsError::internal("unreadable partial fleet search response"))
}

/// One fleet search: `Ok((status, raw JSON))` or a locally rendered error.
/// `rest`: on a cluster, the search runs through the coordinator (every
/// shard from a started copy); the local engine holds only this node's.
#[allow(clippy::too_many_arguments)]
fn one_search(node: &Node, body: &Value, params: &HashMap<String, String>, index: &str, checkpoints: &[i64], timeout: Duration, allow_partial: bool, resolve: ResolveOpts, rest: Option<&crate::ClusterRestRaw>) -> Result<(u16, String), (u16, Value)> {
    let err = |e: EsError| (e.status, e.error_object(false));
    if !checkpoints.is_empty() {
        let (svc, failures) = check_checkpoints(node, index, checkpoints, timeout).map_err(err)?;
        if !failures.is_empty() {
            let all = failures.len() == svc.number_of_shards() as usize;
            if all || !allow_partial {
                return Err(shard_failure_error(&svc.name, &failures, all));
            }
            return partial_search(node, body, params, &svc, &failures).map(|body| (200, body)).map_err(err);
        }
    }
    match rest {
        Some(rest) => (rest.0)("POST", &crate::search_api::coordinator_search_uri(index, params), Some(body.clone())).map_err(err),
        None => run_search(node, body, params, Some(index), resolve, None).map(|body| (200, body)).map_err(err),
    }
}

/// Search parameters without the fleet-only ones.
fn search_params(req: &Req) -> HashMap<String, String> {
    let mut p = req.params.clone();
    for k in ["wait_for_checkpoints", "wait_for_checkpoints_timeout"] {
        p.remove(k);
    }
    p
}

fn fleet_search(node: &Node, req: &Req, index: &str) -> R {
    let checkpoints = long_list(req.param("wait_for_checkpoints"))?;
    let timeout = time_param(req, "wait_for_checkpoints_timeout")?.unwrap_or(DEFAULT_TIMEOUT);
    let allow_partial = req.bool_param("allow_partial_search_results")?.unwrap_or(true);
    if !checkpoints.is_empty() && index.contains(',') {
        let exprs: Vec<&str> = index.split(',').collect();
        return Err(single_index_error(&exprs));
    }
    let body = req.json()?;
    let params = search_params(req);
    match one_search(node, &body, &params, index, &checkpoints, timeout, allow_partial, ResolveOpts::from_params(&req.params), req.extensions.get::<crate::ClusterRestRaw>()) {
        Ok((status, s)) => Ok(Resp::raw(status, s)),
        Err((status, e)) => Ok(Resp::json(status, json!({"error": e, "status": status}))),
    }
}

fn header_checkpoints(v: Option<&Value>) -> Result<Vec<i64>, EsError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_i64().map(Ok).unwrap_or_else(|| parse_long(&elasticsearch_engine::query::dsl::scalar_string(x)))).collect(),
        Some(Value::String(s)) => long_list(Some(s)),
        Some(x) => x.as_i64().map(|n| Ok(vec![n])).unwrap_or_else(|| parse_long(&x.to_string()).map(|n| vec![n])),
    }
}

fn fleet_msearch(node: &Node, req: &Req, index: Option<&str>) -> R {
    let start = Instant::now();
    let text = req.body_str()?;
    if text.trim().is_empty() {
        return Err(EsError::new(400, "parse_exception", "request body or source parameter is required"));
    }
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    // Parse every item first: a multi-index `wait_for_checkpoints` target
    // fails the whole request.
    let mut items: Vec<(Value, Value, Option<String>, Vec<i64>)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let header: Value = serde_json::from_str(lines[i]).map_err(|e| EsError::json_parse(&e))?;
        let body: Value = match lines.get(i + 1) {
            Some(l) => serde_json::from_str(l).map_err(|e| EsError::json_parse(&e))?,
            None => return Err(EsError::illegal_argument("msearch request must be terminated by a newline [\n]")),
        };
        i += 2;
        let idx: Option<String> = match header.get("index") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => Some(a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(",")),
            _ => index.map(str::to_string),
        };
        let cps = header_checkpoints(header.get("wait_for_checkpoints"))?;
        if !cps.is_empty() {
            if let Some(ix) = &idx {
                if ix.contains(',') {
                    return Err(single_index_error(&ix.split(',').collect::<Vec<_>>()));
                }
            }
        }
        items.push((header, body, idx, cps));
    }
    let mut params: HashMap<String, String> = HashMap::new();
    for k in ["rest_total_hits_as_int", "typed_keys"] {
        if let Some(v) = req.param(k) {
            params.insert(k.into(), v.into());
        }
    }
    let mut responses: Vec<String> = Vec::new();
    for (header, body, idx, cps) in items {
        let mut hp = params.clone();
        let mut allow_partial = true;
        if let Some(o) = header.as_object() {
            for (k, v) in o {
                match k.as_str() {
                    "ignore_unavailable" | "allow_no_indices" | "expand_wildcards" => {
                        hp.insert(k.clone(), elasticsearch_engine::query::dsl::scalar_string(v));
                    }
                    "allow_partial_search_results" => {
                        allow_partial = v.as_bool().unwrap_or(v.as_str() != Some("false"));
                        hp.insert(k.clone(), allow_partial.to_string());
                    }
                    _ => {}
                }
            }
        }
        let resolve = ResolveOpts::from_params(&hp);
        let target = idx.unwrap_or_else(|| "_all".to_string());
        match one_search(node, &body, &hp, &target, &cps, DEFAULT_TIMEOUT, allow_partial, resolve, req.extensions.get::<crate::ClusterRestRaw>()) {
            Ok((status, s)) => {
                let s = elasticsearch_engine::source_json::with_member(&s, "status", &status.to_string(), false)
                    .ok_or_else(|| EsError::internal("unreadable fleet search response"))?;
                responses.push(s);
            }
            Err((status, e)) => responses.push(json!({"error": e, "status": status}).to_string()),
        }
    }
    Ok(Resp::raw(200, format!("{{\"took\":{},\"responses\":[{}]}}", start.elapsed().as_millis(), responses.join(","))))
}

// ------------------------------------------------------------ secrets

fn post_secret(node: &Node, req: &Req) -> R {
    let text = req.body_str()?;
    if text.trim().is_empty() {
        return Err(EsError::new(400, "parse_exception", "request body is required"));
    }
    let body = req.json()?;
    let o = body.as_object().ok_or_else(|| EsError::x_content_parse("[1:1] [post_secret_request] failed to parse object"))?;
    let mut value: Option<String> = None;
    for (k, v) in o {
        let (line, col) = elasticsearch_engine::reindex::key_position(text, k, false);
        let (vline, vcol) = elasticsearch_engine::reindex::key_position(text, k, true);
        match k.as_str() {
            "value" => match v {
                Value::String(s) => value = Some(s.clone()),
                Value::Null => {}
                other => {
                    let tok = match other {
                        Value::Array(_) => "START_ARRAY",
                        Value::Object(_) => "START_OBJECT",
                        Value::Number(_) => "VALUE_NUMBER",
                        Value::Bool(_) => "VALUE_BOOLEAN",
                        _ => "VALUE_NULL",
                    };
                    return Err(EsError::x_content_parse(format!("[{vline}:{vcol}] [post_secret_request] value doesn't support values of type: {tok}")));
                }
            },
            other => return Err(EsError::x_content_parse(format!("[{line}:{col}] [post_secret_request] unknown field [{other}]"))),
        }
    }
    let Some(value) = value else { return Err(EsError::action_validation(&["value is missing"])) };
    let id = if let Some(rest) = req.internal_cluster_rest() {
        // Create metadata through the elected master; never create an
        // engine-only index on the node receiving this Fleet request.
        match (rest.0)("PUT", &format!("/{}", fleet::SECRETS_INDEX), Some(fleet::secrets_index_definition())) {
            Ok(_) => {},
            Err(e) if e.kind == "resource_already_exists_exception" => {},
            Err(e) => return Err(e),
        }
        let id = elasticsearch_engine::index::auto_id();
        (rest.0)("PUT", &secret_uri(&id), Some(json!({"value": value})))?;
        id
    } else {
        fleet::post_secret(node, &value)?
    };
    Ok(Resp::ok(json!({"id": id})))
}


fn secret_uri(id: &str) -> String {
    format!("/{}/_doc/{}", fleet::SECRETS_INDEX, crate::search_api::uri_component(id))
}

fn missing_secret(id: &str) -> EsError {
    EsError::resource_not_found(format!("No secret with id [{id}]"))
}

fn get_secret(node: &Node, req: &Req, id: &str) -> Result<Value, EsError> {
    let Some(rest) = req.internal_cluster_rest() else { return fleet::get_secret(node, id) };
    let result = (rest.0)("GET", &secret_uri(id), None).map_err(|e| if e.status == 404 { missing_secret(id) } else { e })?;
    if result["found"] != true { return Err(missing_secret(id)); }
    Ok(json!({"id": id, "value": result["_source"]["value"]}))
}

fn delete_secret(node: &Node, req: &Req, id: &str) -> Result<(), EsError> {
    let Some(rest) = req.internal_cluster_rest() else { return fleet::delete_secret(node, id) };
    // Do not auto-create a missing system index for a delete request.
    (rest.0)("GET", &format!("/{}", fleet::SECRETS_INDEX), None)
        .map_err(|e| if e.status == 404 { missing_secret(id) } else { e })?;
    let result = (rest.0)("DELETE", &secret_uri(id), None).map_err(|e| if e.status == 404 { missing_secret(id) } else { e })?;
    if result["result"] != "deleted" { return Err(missing_secret(id)); }
    Ok(())
}


#[cfg(test)]
mod secret_cluster_tests {
    use super::*;
    use std::sync::Mutex;

    fn request(rest: crate::ClusterRest, body: Value) -> Req {
        let mut extensions = axum::http::Extensions::new();
        extensions.insert(crate::InternalClusterRest(rest.0));
        Req { method: Method::POST, path: "/_fleet/secret".into(),
            segs: vec!["_fleet".into(), "secret".into()], params: Default::default(),
            body: serde_json::to_vec(&body).unwrap().into(), headers: Default::default(),
            access: None, extensions, api_version: crate::RestApiVersion::V9 }
    }

    #[test]
    fn secrets_use_cluster_documents_without_local_engine_writes() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let document = Arc::new(Mutex::new(None::<(String, Value)>));
        let stored = document.clone();
        let req = request(crate::ClusterRest(Arc::new(move |method, uri, body| {
            if uri == format!("/{}", fleet::SECRETS_INDEX) {
                if method == "PUT" { assert_eq!(body.unwrap(), fleet::secrets_index_definition()); }
                return Ok(json!({"acknowledged": true}));
            }
            let mut doc = stored.lock().unwrap();
            match method {
                "PUT" => { *doc = Some((uri.to_owned(), body.unwrap())); Ok(json!({"result": "created"})) },
                "GET" => Ok(match doc.as_ref().filter(|(path, _)| path == uri) {
                    Some((_, source)) => json!({"found":true,"_source":source}),
                    None => json!({"found":false}),
                }),
                "DELETE" => Ok(json!({"result": if doc.take().is_some() { "deleted" } else { "not_found" }})),
                _ => panic!("unexpected internal request"),
            }
        })), json!({"value":"example-secret"}));
        let response = post_secret(&node, &req).unwrap();
        let crate::RespBody::Json(body) = response.body else { panic!("expected JSON") };
        let id = body["id"].as_str().unwrap();
        assert_eq!(get_secret(&node, &req, id).unwrap(), json!({"id":id,"value":"example-secret"}));
        assert!(node.get_index(fleet::SECRETS_INDEX).is_none());
        delete_secret(&node, &req, id).unwrap();
        assert_eq!(get_secret(&node, &req, id).unwrap_err().status, 404);
        assert_eq!(delete_secret(&node, &req, id).unwrap_err().status, 404);
    }

    #[test]
    fn invalid_secret_does_not_create_cluster_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let req = request(crate::ClusterRest(Arc::new(|_, _, _| panic!("invalid request reached cluster"))), json!({"value":42}));
        assert_eq!(post_secret(&node, &req).err().unwrap().status, 400);
    }
}
