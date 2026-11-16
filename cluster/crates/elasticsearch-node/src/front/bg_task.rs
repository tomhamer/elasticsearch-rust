//! `wait_for_completion=false` bulk-by-scroll tasks that run on the cluster
//! layer (reindex / update-by-query / delete-by-query over multi-shard
//! indices). `_tasks` itself is served cluster-wide by `rest::tasks`.
//!
//! A cluster-run task is registered in the coordinating node's engine task
//! manager (so `_tasks/{id}` there reports it running, waits for it, and
//! serves its stored result exactly like an engine task) and runs as a tokio
//! task. Engine task ids carry the engine's fixed node id, the same on every
//! node; the id handed to the client carries the owning cluster node's id
//! instead, as ES's `node:number` does. The running work can see its task
//! (`current`) to stop between batches once it is cancelled.

use super::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_engine::tasks::Task;
use serde_json::{json, Map, Value};
use std::future::Future;
use std::sync::Arc;

tokio::task_local! {
    static CURRENT: Arc<Task>;
}

/// The cluster task the calling future runs under (inside `spawn`'s work).
pub(crate) fn current() -> Option<Arc<Task>> {
    CURRENT.try_with(|t| t.clone()).ok()
}

/// `fut` run under cluster task `task` (when there is one), as the future
/// that captured it: work handed to a task of its own keeps its parent.
pub(crate) async fn under<F: Future>(task: Option<Arc<Task>>, fut: F) -> F::Output {
    match task {
        Some(t) => CURRENT.scope(t, fut).await,
        None => fut.await,
    }
}

pub(crate) fn opaque_headers(req: &RestReq) -> Map<String, Value> {
    let mut h = Map::new();
    if let Some((_, v)) = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("x-opaque-id")) {
        h.insert("X-Opaque-Id".into(), json!(v));
    }
    h
}

/// Start `work` as a background task and answer `{"task": "node:n"}`.
pub(crate) fn spawn<F>(node: &Arc<ClusterNode>, req: &RestReq, action: &str, description: String, rps: f64, work: F) -> RestResp
where
    F: Future<Output = RestResp> + Send + 'static,
{
    let task = node.engine.tasks.register(action, Some(description), true, true, opaque_headers(req));
    task.set_rps(rps);
    let id = format!("{}:{}", node.local.id, task.id);
    let node2 = node.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        let outcome = outcome_of(CURRENT.scope(task.clone(), work).await);
        node2.engine.tasks.finish(&task, outcome.clone());
        store_task_doc(&node2, &task, &doc_id, outcome).await;
    });
    RestResp::json(200, &json!({ "task": id }))
}

/// The stored outcome: the BulkByScroll response (with the human-readable
/// throttle fields), or the error object of a failed task.
fn outcome_of(r: RestResp) -> Result<Value, Value> {
    let v = r.body_json().unwrap_or(Value::Null);
    match v.get("error") {
        Some(e) if !r.is_success() => {
            let mut e = e.clone();
            if let Some(o) = e.as_object_mut() {
                o.remove("root_cause");
            }
            Err(e)
        }
        _ if !r.is_success() && !v.is_object() => Err(json!({"type": "exception", "reason": String::from_utf8_lossy(&r.body)})),
        _ => {
            let mut v = v;
            elasticsearch_engine::reindex::with_human(&mut v);
            Ok(v)
        }
    }
}

/// The `.tasks` index ES's `TaskResultsService` creates (`task-index-mapping.json`):
/// the response and error are stored, not indexed.
fn tasks_index_body() -> Value {
    json!({
        "settings": {"index.hidden": true, "index.number_of_shards": 1, "index.auto_expand_replicas": "0-1"},
        "mappings": {
            "dynamic": false,
            "properties": {
                "completed": {"type": "boolean"},
                "task": {"properties": {
                    "action": {"type": "keyword"}, "cancellable": {"type": "boolean"}, "cancelled": {"type": "boolean"},
                    "id": {"type": "long"}, "parent_task_id": {"type": "keyword"}, "node": {"type": "keyword"},
                    "running_time_in_nanos": {"type": "long"}, "start_time_in_millis": {"type": "long"}, "type": {"type": "keyword"},
                    "status": {"type": "object", "enabled": false}, "description": {"type": "text"},
                    "headers": {"type": "object", "enabled": false}
                }},
                "response": {"type": "object", "enabled": false},
                "error": {"type": "object", "enabled": false}
            }
        }
    })
}

fn system_request(method: &str, uri: std::fmt::Arguments<'_>, body: Option<&Value>) -> elasticsearch_engine::error::Result<RestReq> {
    let headers = super::SharedHeaders::try_pairs(&[(super::SYSTEM_HEADER, "true"), ("content-type", "application/json")])?;
    let mut req = RestReq::try_new_uri(method, uri, &headers, &[], 0, Default::default(), None)?;
    if let Some(body) = body {
        req.set_json(body)?;
    }
    Ok(req)
}

async fn system_call(node: &Arc<ClusterNode>, req: elasticsearch_engine::error::Result<RestReq>) -> RestResp {
    match req {
        Ok(req) => super::dispatch(node, req).await,
        Err(e) => RestResp::json(e.status, &e.to_json()),
    }
}

/// Persist the finished task in the cluster's `.tasks` index (ES
/// `TaskResultsService.storeResult`: a replicated write, retried with
/// backoff), so `GET _tasks/{id}` finds it after its node restarts or
/// leaves the cluster.
async fn store_task_doc(node: &Arc<ClusterNode>, task: &Task, id: &str, outcome: Result<Value, Value>) {
    let Some(stored) = node.engine.tasks.get_result(task.id) else { return };
    let mut info = stored.task;
    if let Some(o) = info.as_object_mut() {
        o.remove("cancelled");
        if let Some(n) = id.split_once(':').map(|(n, _)| n) {
            o.insert("node".into(), json!(n));
        }
    }
    let mut doc = Map::new();
    doc.insert("completed".into(), json!(true));
    doc.insert("task".into(), info);
    match outcome {
        Ok(v) => doc.insert("response".into(), v),
        Err(e) => doc.insert("error".into(), e),
    };
    store_doc(node, id, Value::Object(doc)).await;
}

/// Shard engine `k`'s task result sink: an engine-run task's document,
/// stored under the task's cluster id (`<node>:<encoded k, n>`).
pub(crate) fn task_result_sinks(node: std::sync::Weak<ClusterNode>) -> crate::shard_engines::TaskSinks {
    let rt = tokio::runtime::Handle::current();
    Arc::new(move |k: u32| -> elasticsearch_engine::node::TaskResultSink {
        let (node, rt) = (node.clone(), rt.clone());
        Arc::new(move |n: u64, mut doc: Value| {
            let Some(node) = node.upgrade() else { return };
            let num = crate::rest::tasks::encode_id(k, n);
            let id = format!("{}:{num}", node.local.id);
            if let Some(t) = doc.get_mut("task").and_then(Value::as_object_mut) {
                t.insert("node".into(), json!(node.local.id));
                t.insert("id".into(), json!(num));
            }
            rt.spawn(async move { store_doc(&node, &id, doc).await });
        })
    })
}

async fn store_doc(node: &Arc<ClusterNode>, id: &str, doc: Value) {
    let mut delay = std::time::Duration::from_millis(250);
    for attempt in 0..8 {
        if !node.state().metadata.indices.contains_key(TASKS_INDEX) {
            // fmt::Arguments is not Send: build each request before suspending.
            let req = system_request("PUT", format_args!("/{TASKS_INDEX}"), Some(&tasks_index_body()));
            let r = system_call(node, req).await;
            if !r.is_success() && r.body_json().map(|v| v["error"]["type"] != "resource_already_exists_exception").unwrap_or(true) {
                tracing::debug!(task = id, "creating {TASKS_INDEX} failed: {}", String::from_utf8_lossy(&r.body));
            }
        }
        let req = system_request("PUT", format_args!("/{TASKS_INDEX}/_doc/{}", super::UriComponent(id)), Some(&doc));
        let r = system_call(node, req).await;
        if r.is_success() {
            return;
        }
        tracing::warn!(task = id, attempt, "failed to store the task result: {}", String::from_utf8_lossy(&r.body));
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(std::time::Duration::from_secs(10));
    }
    tracing::error!(task = id, "gave up storing the task result in {TASKS_INDEX}");
}

pub(crate) const TASKS_INDEX: &str = ".tasks";

/// The stored result of finished task `id` from `.tasks` (ES
/// `TransportGetTaskAction.getFinishedTaskFromIndex`), read through the
/// cluster: `None` when it is not there.
pub(crate) async fn stored_result(node: &Arc<ClusterNode>, id: &str) -> Option<Value> {
    if !node.state().metadata.indices.contains_key(TASKS_INDEX) {
        return None;
    }
    let req = system_request("GET", format_args!("/{TASKS_INDEX}/_doc/{}", super::UriComponent(id)), None);
    let r = system_call(node, req).await;
    let v = r.body_json()?;
    if !r.is_success() || v["found"] != true {
        return None;
    }
    v.get("_source").filter(|s| s.is_object()).cloned()
}
