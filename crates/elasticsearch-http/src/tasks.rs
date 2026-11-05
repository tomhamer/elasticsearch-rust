//! `_tasks` (list / get / cancel), `_reindex`, `_update_by_query`,
//! `_delete_by_query` (sync, or `wait_for_completion=false` → task id) and
//! their `_rethrottle` endpoints.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, NODE_ID};
use elasticsearch_engine::reindex::{self, ByQueryRequest, Kind};
use elasticsearch_engine::tasks::{parse_task_id, Task};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::Duration;

/// Global endpoints: `/_tasks...`, `/_reindex...`, `/_update_by_query/{id}/_rethrottle`, `/_delete_by_query/{id}/_rethrottle`.
pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let a = req.seg(0).to_string();
    let b = req.seg(1).to_string();
    let c = req.seg(2).to_string();
    let m = req.method.clone();
    match (a.as_str(), b.as_str(), c.as_str(), &m) {
        ("_tasks", "", "", &Method::GET) => blocking(move || list(&node, &req)).await,
        ("_tasks", "_cancel", "", &Method::POST) => blocking(move || cancel(&node, &req, None)).await,
        ("_tasks", id, "", &Method::GET) => {
            let id = id.to_string();
            blocking(move || get(&node, &req, &id)).await
        }
        ("_tasks", id, "_cancel", &Method::POST) => {
            let id = id.to_string();
            blocking(move || cancel(&node, &req, Some(&id))).await
        }
        ("_reindex", "", "", &Method::POST) => run(node, req, None, Kind::Reindex).await,
        // Reindex management (ES 9 `reindex-management` module).
        ("_reindex", "", "", &Method::GET) => blocking(move || reindex_list(&node, &req)).await,
        ("_reindex", "", "", _) => method_not_allowed(&req, &["GET", "POST"]),
        ("_reindex", id, "", &Method::GET) => {
            let id = id.to_string();
            blocking(move || reindex_get(&node, &req, &id)).await
        }
        ("_reindex", id, "_cancel", &Method::POST) => {
            let id = id.to_string();
            blocking(move || reindex_cancel(&node, &req, &id)).await
        }
        ("_reindex", id, "_rethrottle", &Method::POST) | ("_update_by_query", id, "_rethrottle", &Method::POST) | ("_delete_by_query", id, "_rethrottle", &Method::POST) => {
            let id = id.to_string();
            blocking(move || rethrottle(&node, &req, &id)).await
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `/{index}/_update_by_query`, `/{index}/_delete_by_query`.
pub async fn by_query(node: Arc<Node>, req: Req, index: String) -> R {
    let api = req.seg(1).to_string();
    let kind = if api == "_update_by_query" { Kind::UpdateByQuery } else { Kind::DeleteByQuery };
    if req.method != Method::POST {
        return method_not_allowed(&req, &["POST"]);
    }
    if !req.seg(2).is_empty() {
        return crate::not_found_handler(&req);
    }
    run(node, req, Some(index), kind).await
}

fn opaque_headers(req: &Req) -> Map<String, Value> {
    let mut h = Map::new();
    if let Some(v) = req.headers.get("x-opaque-id").and_then(|v| v.to_str().ok()) {
        h.insert("X-Opaque-Id".into(), json!(v));
    }
    h
}

/// ES `reindex.remote.whitelist` (a node setting: a list or a comma
/// separated string of `host:port` patterns). `ES_REINDEX_REMOTE_WHITELIST`
/// adds patterns too (older launchers and fixtures set it).
fn remote_whitelist(node: &Node) -> Vec<String> {
    let split = |s: &str| s.split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect::<Vec<_>>();
    let mut out = match node.node_settings.read().get("reindex.remote.whitelist") {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).flat_map(split).collect(),
        Some(Value::String(s)) => split(s.trim_matches(|c| c == '[' || c == ']')),
        _ => Vec::new(),
    };
    out.extend(split(&std::env::var("ES_REINDEX_REMOTE_WHITELIST").unwrap_or_default()));
    out
}

async fn run(node: Arc<Node>, req: Req, index: Option<String>, kind: Kind) -> R {
    blocking(move || {
        if kind == Kind::Reindex && req.body.is_empty() && req.param("source").is_none() {
            return Err(EsError::new(400, "parse_exception", "request body is required"));
        }
        let text = req.body_str()?.to_string();
        let (body,raw) = match req.json() {
            Ok(body)=>(body,None),
            Err(_) => elasticsearch_engine::source_json::update_request(&text)?,
        };
        // ES re-serialises these bodies through a `HashMap` before parsing them
        // as a search source: positions are relative to that text.
        let canon = elasticsearch_engine::jsonpos::java_hashmap_json(&body);
        let mut r: ByQueryRequest = match &index {
            None => reindex::parse_reindex(&body, &text, &req.params)?,
            Some(i) => reindex::parse_by_query(kind, i, &body, &req.params).map_err(|e| elasticsearch_engine::jsonpos::resolve(e, canon.as_bytes()))?,
        };
        r.raw_script=raw.as_ref().and_then(|r|r.get("script")).cloned();
        if kind == Kind::DeleteByQuery && r.query.is_none() {
            return Err(EsError::action_validation(&["query is missing"]));
        }
        if index.is_some() {
            if let Some(f) = crate::docs::source_filter_from_params(&req.params) {
                r.source_filter = f;
            }
        }
        if let Some(remote) = &r.remote {
            remote.check_whitelist(&remote_whitelist(&node))?;
        }
        let wait = req.bool_param("wait_for_completion")?.unwrap_or(true);
        // Fail fast on missing source indices (ES validates before starting).
        if r.remote.is_some() {
        } else if let Some(i) = &index {
            node.resolve(i, r.resolve)?;
        } else if r.source != "_all" {
            node.resolve(&r.source, r.resolve)?;
        }
        let task = node.tasks.register(kind.action(), Some(r.description()), true, !wait, opaque_headers(&req));
        task.set_rps(r.rps);
        if wait {
            let res = reindex::execute(&node, &r, &task);
            node.tasks.finish(&task, Ok(Value::Null));
            let (status, v) = res?;
            return Ok(Resp::json(status, v));
        }
        let id = task.task_id();
        let n = node.clone();
        std::thread::Builder::new()
            .name("es-bulk-by-scroll".into())
            .spawn(move || {
                let res = reindex::execute(&n, &r, &task);
                let outcome = match res {
                    Ok((_, mut v)) => {
                        reindex::with_human(&mut v);
                        Ok(v)
                    }
                    Err(e) => Err(error_body(&e)),
                };
                n.tasks.finish(&task, outcome.clone());
                store_task_doc(&n, &task, outcome);
            })
            .map_err(|e| EsError::internal(format!("failed to start task: {e}")))?;
        Ok(Resp::ok(json!({ "task": id })))
    })
    .await
}

/// Persist a finished `wait_for_completion=false` task in the `.tasks` index, as ES does.
fn store_task_doc(node: &Node, task: &Task, outcome: Result<Value, Value>) {
    let Some(stored) = node.tasks.get_result(task.id) else { return };
    let mut info = stored.task;
    if let Some(o) = info.as_object_mut() {
        o.remove("cancelled");
    }
    let mut doc = Map::new();
    doc.insert("completed".into(), json!(true));
    doc.insert("task".into(), info);
    match outcome {
        Ok(v) => doc.insert("response".into(), v),
        Err(e) => doc.insert("error".into(), e),
    };
    let Ok(svc) = node.write_index(".tasks", true) else { return };
    let src = Value::Object(doc).to_string();
    let _ = svc.write(vec![elasticsearch_engine::index::WriteReq::Index { id: Some(task.task_id()), source: src, routing: None, cond: Default::default() }], true);
}

fn error_body(e: &EsError) -> Value {
    let mut v = e.error_object(false);
    if let Some(o) = v.as_object_mut() {
        o.remove("root_cause");
    }
    v
}

fn node_header(node: &Node) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("name".into(), json!(node.name));
    m.insert("transport_address".into(), json!("127.0.0.1:9300"));
    m.insert("host".into(), json!("127.0.0.1"));
    m.insert("ip".into(), json!("127.0.0.1:9300"));
    m.insert("roles".into(), json!(["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "ingest", "master", "ml", "remote_cluster_client", "transform"]));
    m.insert("attributes".into(), crate::observability::node_attributes());
    m
}

/// Render tasks in the `ListTasksResponse` shape for `group_by`.
fn list_json(node: &Node, tasks: &[Arc<Task>], detailed: bool, group_by: &str) -> Result<Value, EsError> {
    match group_by {
        "nodes" => {
            if tasks.is_empty() {
                return Ok(json!({"nodes": {}}));
            }
            let mut n = node_header(node);
            n.insert("tasks".into(), Value::Object(tasks.iter().map(|t| (t.task_id(), t.info(detailed))).collect()));
            let mut nodes = Map::new();
            nodes.insert(NODE_ID.into(), Value::Object(n));
            Ok(json!({ "nodes": nodes }))
        }
        "parents" => {
            // Children nest under their parent when it is listed too.
            let ids: std::collections::HashSet<String> = tasks.iter().map(|t| t.task_id()).collect();
            let mut out = Map::new();
            for t in tasks.iter().filter(|t| t.parent.as_ref().map(|p| !ids.contains(p)).unwrap_or(true)) {
                let mut v = t.info(detailed);
                let kids: Vec<Value> = tasks.iter().filter(|c| c.parent.as_deref() == Some(t.task_id().as_str())).map(|c| c.info(detailed)).collect();
                if !kids.is_empty() {
                    v["children"] = Value::Array(kids);
                }
                out.insert(t.task_id(), v);
            }
            Ok(json!({"tasks": out}))
        }
        "none" => Ok(json!({"tasks": tasks.iter().map(|t| t.info(detailed)).collect::<Vec<_>>()})),
        o => Err(EsError::illegal_argument(format!("[group_by] must be one of [nodes], [parents] or [none] but was [{o}]"))),
    }
}

fn filtered(node: &Node, req: &Req) -> Vec<Arc<Task>> {
    let actions: Option<Vec<String>> = req.param("actions").map(|a| a.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect());
    let parent = req.param("parent_task_id").map(str::to_string);
    let nodes_ok = req.param("nodes").map(|n| n.split(',').any(|x| x == NODE_ID || x == "_local" || x == "_all" || x == "*" || x == node.name)).unwrap_or(true);
    if !nodes_ok {
        return Vec::new();
    }
    node.tasks
        .running()
        .into_iter()
        .filter(|t| actions.as_ref().map(|a| a.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, &t.action))).unwrap_or(true))
        .filter(|t| parent.as_ref().map(|p| t.parent.as_deref() == Some(p.as_str())).unwrap_or(true))
        .collect()
}

fn list(node: &Node, req: &Req) -> R {
    let detailed = req.flag("detailed")?;
    let group_by = req.param("group_by").unwrap_or("nodes").to_string();
    // The list request is itself a running task, as in ES.
    let me = node.tasks.register("cluster:monitor/tasks/lists", Some(String::new()), true, false, opaque_headers(req));
    // ...and its per-node child, as ES fans the list out to every node.
    let child = node.tasks.register_with_parent("cluster:monitor/tasks/lists[n]", Some(String::new()), true, false, opaque_headers(req), Some(me.task_id()));
    let result = (|| {
        let mut tasks = filtered(node, req);
        if req.flag("wait_for_completion")? {
            let timeout = timeout_param(req)?;
            tasks.retain(|t| t.id != me.id && t.id != child.id);
            for t in &tasks {
                node.tasks.wait(t.id, timeout);
            }
            tasks = Vec::new();
        }
        list_json(node, &tasks, detailed, &group_by)
    })();
    node.tasks.finish(&child, Ok(Value::Null));
    node.tasks.finish(&me, Ok(Value::Null));
    Ok(Resp::ok(result?))
}

fn timeout_param(req: &Req) -> Result<Duration, EsError> {
    match req.param("timeout") {
        None => Ok(Duration::from_secs(30)),
        Some(t) => elasticsearch_engine::date::parse_time_value(t)
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [timeout] with value [{t}] as a time value: unit is missing or unrecognized"))),
    }
}

fn not_stored(id: &str) -> EsError {
    EsError::resource_not_found(format!("task [{id}] isn't running and hasn't stored its results"))
}

fn get(node: &Node, req: &Req, id: &str) -> R {
    let (n, num) = parse_task_id(id)?;
    if n != NODE_ID {
        return Err(EsError::resource_not_found(format!("task [{id}] belongs to the node [{n}] which isn't part of the cluster and there is no record of the task")));
    }
    if node.tasks.get_running(num).is_some() && req.flag("wait_for_completion")? {
        let timeout = timeout_param(req)?;
        if !node.tasks.wait(num, timeout) {
            return Err(EsError::new(429, "timeout_exception", format!("Timed out waiting for completion of task [{id}]")));
        }
    }
    if let Some(t) = node.tasks.get_running(num) {
        return Ok(Resp::ok(json!({"completed": false, "task": t.info(true)})));
    }
    match node.tasks.get_result(num) {
        Some(r) => {
            let mut m = Map::new();
            m.insert("completed".into(), json!(true));
            m.insert("task".into(), r.task);
            match r.outcome {
                Ok(v) => m.insert("response".into(), v),
                Err(e) => m.insert("error".into(), e),
            };
            Ok(Resp::ok(Value::Object(m)))
        }
        None => Err(not_stored(id)),
    }
}

const REINDEX_ACTION: &str = "indices:data/write/reindex";

/// `GetReindexResponse.taskInfoToXContent`.
fn reindex_task_info(task: &Value, human: bool) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("id".into(), json!(format!("{}:{}", task["node"].as_str().unwrap_or(NODE_ID), task["id"])));
    if let Some(d) = task["description"].as_str().filter(|d| d.starts_with("reindex from [")) {
        m.insert("description".into(), json!(d));
    }
    let start = task["start_time_in_millis"].as_i64().unwrap_or(0);
    if human {
        m.insert("start_time".into(), json!(elasticsearch_engine::date::human_date(start)));
    }
    m.insert("start_time_in_millis".into(), json!(start));
    let nanos = task["running_time_in_nanos"].as_u64().unwrap_or(0);
    if human {
        m.insert("running_time".into(), json!(format!("{}ms", nanos / 1_000_000)));
    }
    m.insert("running_time_in_nanos".into(), json!(nanos));
    m.insert("cancelled".into(), json!(task["cancelled"].as_bool().unwrap_or(false)));
    if let Some(st) = task.get("status").filter(|s| !s.is_null()) {
        m.insert("status".into(), st.clone());
    }
    m
}

fn reindex_not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("Reindex operation [{id}] not found"))
}

/// `GET /_reindex[?detailed]`: running reindex operations.
fn reindex_list(node: &Node, req: &Req) -> R {
    let detailed = req.flag("detailed")?;
    let human = req.flag("human")?;
    let mut tasks: Vec<Arc<Task>> = node.tasks.running().into_iter().filter(|t| t.action == REINDEX_ACTION && t.parent.is_none()).collect();
    tasks.sort_by_key(|t| t.id);
    let out: Vec<Value> = tasks.iter().map(|t| Value::Object(reindex_task_info(&t.info(detailed), human))).collect();
    Ok(Resp::ok(json!({"reindex": out})))
}

/// `GET /_reindex/{task_id}[?wait_for_completion][&timeout]`.
fn reindex_get(node: &Node, req: &Req, id: &str) -> R {
    let (n, num) = parse_task_id(id).map_err(|_| reindex_not_found(id))?;
    if n != NODE_ID {
        return Err(reindex_not_found(id));
    }
    let probe = match (node.tasks.get_running(num), node.tasks.get_result(num)) {
        (Some(t), _) => t.info(true),
        (None, Some(r)) => r.task,
        (None, None) => return Err(reindex_not_found(id)),
    };
    if probe["action"].as_str() != Some(REINDEX_ACTION) || probe.get("parent_task_id").is_some_and(|p| !p.is_null()) {
        return Err(reindex_not_found(id));
    }
    if node.tasks.get_running(num).is_some() && req.flag("wait_for_completion")? {
        let timeout = timeout_param(req)?;
        if !node.tasks.wait(num, timeout) {
            return Err(EsError::new(429, "timeout_exception", format!("Timed out waiting for completion of task [{id}]")));
        }
    }
    reindex_response(node, req, id, num)
}

fn reindex_response(node: &Node, req: &Req, id: &str, num: u64) -> R {
    let human = req.flag("human")?;
    if let Some(t) = node.tasks.get_running(num) {
        let mut m = Map::new();
        m.insert("completed".into(), json!(false));
        m.extend(reindex_task_info(&t.info(true), human));
        return Ok(Resp::ok(Value::Object(m)));
    }
    let r = node.tasks.get_result(num).ok_or_else(|| reindex_not_found(id))?;
    let mut m = Map::new();
    m.insert("completed".into(), json!(true));
    m.extend(reindex_task_info(&r.task, human));
    match r.outcome {
        Ok(v) => m.insert("response".into(), v),
        Err(e) => m.insert("error".into(), e),
    };
    Ok(Resp::ok(Value::Object(m)))
}

/// `POST /_reindex/{task_id}/_cancel[?wait_for_completion=true]`.
fn reindex_cancel(node: &Node, req: &Req, id: &str) -> R {
    let not_found = || EsError::resource_not_found(format!("reindex task [{id}] either not found or completed"));
    let wait = req.bool_param("wait_for_completion")?.unwrap_or(true);
    let (n, num) = parse_task_id(id).map_err(|_| not_found())?;
    let running = node.tasks.get_running(num).filter(|t| n == NODE_ID && t.action == REINDEX_ACTION && t.parent.is_none()).ok_or_else(not_found)?;
    drop(running);
    node.tasks.cancel(num)?.ok_or_else(not_found)?;
    if !wait {
        return Ok(Resp::ok(json!({"acknowledged": true})));
    }
    node.tasks.wait(num, std::time::Duration::from_secs(30));
    reindex_response(node, req, id, num)
}

fn missing_task_failure(id: &str) -> Value {
    json!({
        "node_failures": [{
            "type": "failed_node_exception",
            "reason": format!("Failed node [{NODE_ID}]"),
            "node_id": NODE_ID,
            "caused_by": {"type": "resource_not_found_exception", "reason": format!("task [{id}] is not found")}
        }],
        "nodes": {}
    })
}

fn cancel(node: &Node, req: &Req, id: Option<&str>) -> R {
    let group_by = req.param("group_by").unwrap_or("nodes").to_string();
    let mut cancelled = Vec::new();
    match id {
        Some(id) => {
            let (n, num) = parse_task_id(id)?;
            if n != NODE_ID {
                return Ok(Resp::ok(missing_task_failure(id)));
            }
            match node.tasks.cancel(num)? {
                Some(t) => cancelled.push(t),
                None => return Ok(Resp::ok(missing_task_failure(id))),
            }
        }
        None if req.param("actions").is_none() && req.param("nodes").is_none() && req.param("parent_task_id").is_none() => {
            // ES cancels every cancellable task, its own node-level cancel
            // task included, and reports that as a node failure.
            for t in filtered(node, req) {
                if t.cancellable {
                    node.tasks.cancel(t.id)?;
                }
            }
            return Ok(Resp::ok(json!({
                "node_failures": [{
                    "type": "failed_node_exception",
                    "reason": format!("Failed node [{NODE_ID}]"),
                    "node_id": NODE_ID,
                    "caused_by": {"type": "task_cancelled_exception", "reason": "task cancelled [by user request]"}
                }],
                "nodes": {}
            })));
        }
        None => {
            for t in filtered(node, req) {
                if t.cancellable {
                    if let Some(t) = node.tasks.cancel(t.id)? {
                        cancelled.push(t);
                    }
                }
            }
        }
    }
    if req.flag("wait_for_completion")? {
        let timeout = timeout_param(req)?;
        for t in &cancelled {
            node.tasks.wait(t.id, timeout);
        }
    }
    Ok(Resp::ok(list_json(node, &cancelled, false, &group_by)?))
}

fn rethrottle(node: &Node, req: &Req, id: &str) -> R {
    let rps = req.param("requests_per_second").ok_or_else(|| EsError::illegal_argument("requests_per_second is a required parameter"))?;
    let rps = reindex::parse_rps(rps)?;
    let (n, num) = parse_task_id(id)?;
    if n != NODE_ID {
        return Ok(Resp::ok(json!({
            "node_failures": [{
                "type": "failed_node_exception",
                "reason": format!("Failed node [{n}]"),
                "node_id": n,
                "caused_by": {"type": "no_such_node_exception", "reason": format!("No such node [{n}]"), "node_id": n}
            }],
            "nodes": {}
        })));
    }
    match node.tasks.get_running(num) {
        Some(t) => {
            t.set_rps(rps);
            Ok(Resp::ok(list_json(node, &[t], true, req.param("group_by").unwrap_or("nodes"))?))
        }
        None => Ok(Resp::ok(missing_task_failure(id))),
    }
}
