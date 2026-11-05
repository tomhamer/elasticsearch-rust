//! Enrich REST handlers: `_enrich/policy` CRUD, `_execute`, `_stats`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::enrich;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ES_VERSION};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let segs: Vec<String> = req.segs.clone();
    let s = |i: usize| segs.get(i).map(String::as_str).unwrap_or("");
    let m = req.method.clone();
    match (s(1), segs.len()) {
        ("_stats", 2) => match m {
            Method::GET => Ok(Resp::ok(enrich::stats(&node))),
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("policy", 2) => match m {
            Method::GET => blocking(move || Ok(Resp::ok(enrich::get_policies(&node, &[])))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("policy", 3) => {
            let name = s(2).to_string();
            match m {
                Method::GET => blocking(move || {
                    let names: Vec<String> = name.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
                    Ok(Resp::ok(enrich::get_policies(&node, &names)))
                })
                .await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    let warn = enrich::put_policy(&node, &name, &body)?;
                    let mut r = Resp::ack();
                    if warn {
                        r.headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"{}\"", crate::cluster::BUILD_HASH, enrich::VERSION_DEPRECATION)));
                    }
                    Ok(r)
                })
                .await,
                Method::DELETE => blocking(move || {
                    enrich::delete_policy(&node, &name)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("policy", 4) if s(3) == "_execute" => {
            let name = s(2).to_string();
            match m {
                Method::PUT | Method::POST => blocking(move || execute(&node, &req, &name)).await,
                _ => method_not_allowed(&req, &["PUT", "POST"]),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

/// Policy index operations through the cluster coordinator.
struct RestPolicyOps(crate::ClusterRest);

impl RestPolicyOps {
    fn call(&self, method: &str, uri: &str, body: Option<Value>) -> Result<Value, EsError> {
        (self.0 .0)(method, uri, body)
    }
}

impl enrich::PolicyIndexOps for RestPolicyOps {
    fn create_index(&self, name: &str, body: &Value) -> Result<(), EsError> {
        self.call("PUT", &format!("/{}", name), Some(body.clone())).map(|_| ())
    }
    fn reindex(&self, body: &Value) -> Result<Value, EsError> {
        self.call("POST", "/_reindex?refresh=true", Some(body.clone()))
    }
    fn update_aliases(&self, actions: &[Value]) -> Result<(), EsError> {
        self.call("POST", "/_aliases", Some(json!({"actions": actions}))).map(|_| ())
    }
    fn block_writes(&self, index: &str) -> Result<(), EsError> {
        self.call("PUT", &format!("/{}/_settings", index), Some(json!({"index": {"blocks": {"write": true}}}))).map(|_| ())
    }
    fn delete_index(&self, index: &str) -> Result<(), EsError> {
        self.call("DELETE", &format!("/{}", index), None).map(|_| ())
    }
}

fn run(node: &Node, name: &str, task: &elasticsearch_engine::tasks::Task, rest: Option<&crate::ClusterRest>) -> Result<String, EsError> {
    match rest {
        Some(rest) => enrich::execute_policy_with(node, name, task, &RestPolicyOps(rest.clone())),
        None => enrich::execute_policy(node, name, task),
    }
}

fn execute(node: &Arc<Node>, req: &Req, name: &str) -> R {
    let wait = req.bool_param("wait_for_completion")?.unwrap_or(true);
    if enrich::store(node).get(name).is_none() {
        return Err(EsError::resource_not_found(format!("policy [{name}] does not exist")));
    }
    let mut headers = Map::new();
    if let Some(v) = req.headers.get("x-opaque-id").and_then(|v| v.to_str().ok()) {
        headers.insert("X-Opaque-Id".into(), json!(v));
    }
    // ES runs policy execution under ENRICH_ORIGIN.
    let rest = req.internal_cluster_rest();
    let task = node.tasks.register("cluster:admin/xpack/enrich/execute", Some(format!("policy execution [{name}]")), false, !wait, headers);
    if wait {
        let r = run(node, name, &task, rest.as_ref());
        node.tasks.finish(&task, Ok(Value::Null));
        r?;
        return Ok(Resp::ok(json!({"status": {"phase": "COMPLETE"}})));
    }
    let id = task.task_id();
    let n = node.clone();
    let name = name.to_string();
    std::thread::Builder::new()
        .name("es-enrich-execute".into())
        .spawn(move || {
            let outcome = match run(&n, &name, &task, rest.as_ref()) {
                Ok(_) => Ok(json!({"status": {"phase": "COMPLETE"}})),
                Err(e) => {
                    let mut v = e.error_object(false);
                    if let Some(o) = v.as_object_mut() {
                        o.remove("root_cause");
                    }
                    Err(v)
                }
            };
            n.tasks.finish(&task, outcome);
        })
        .map_err(|e| EsError::internal(format!("failed to start task: {e}")))?;
    Ok(Resp::ok(json!({ "task": id })))
}
