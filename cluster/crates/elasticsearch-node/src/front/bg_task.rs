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
    let engine = node.engine.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        let outcome = outcome_of(CURRENT.scope(task.clone(), work).await);
        let _ = tokio::task::spawn_blocking(move || {
            engine.tasks.finish(&task, outcome.clone());
            store_task_doc(&engine, &task, &doc_id, outcome);
        })
        .await;
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

/// Persist the finished task in `.tasks`, as the engine does for its own.
fn store_task_doc(engine: &elasticsearch_engine::node::Node, task: &elasticsearch_engine::tasks::Task, id: &str, outcome: Result<Value, Value>) {
    let Some(stored) = engine.tasks.get_result(task.id) else { return };
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
    let Ok(svc) = engine.write_index(".tasks", true) else { return };
    let src = Value::Object(doc).to_string();
    let _ = svc.write(vec![elasticsearch_engine::index::WriteReq::Index { id: Some(id.to_string()), source: src, routing: None, cond: Default::default() }], true);
}
