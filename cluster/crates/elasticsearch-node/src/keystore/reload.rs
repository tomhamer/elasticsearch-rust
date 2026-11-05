//! `POST _nodes[/{node_id}]/reload_secure_settings` (ES
//! `TransportNodesReloadSecureSettingsAction`): every selected node re-reads
//! and decrypts its keystore with the request's `secure_settings_password`
//! and hands the new secure settings to its reloadable consumers (the S3
//! repository clients); each node answers its own entry, with a
//! `reload_exception` on failure (missing keystore, wrong password, ...).

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::Duration;

/// The per-node action.
pub const ACTION: &str = "cluster:admin/nodes/reload_secure_settings[n]";

#[derive(Serialize, Deserialize)]
struct NodeReq {
    password: Option<String>,
}

/// Register the per-node transport action.
pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION, move |r: NodeReq| {
        let n = n.clone();
        async move { Ok::<String, RemoteError>(reload_local(&n, r.password.as_deref().unwrap_or("")).await.to_string()) }
    });
}

/// This node's entry of the response.
pub async fn reload_local(node: &ClusterNode, password: &str) -> Value {
    let conf = super::conf_dir(&node.config.settings);
    let mut entry = Map::new();
    entry.insert("name".into(), json!(node.local.name));
    let pw = password.to_string();
    let result = tokio::task::spawn_blocking(move || super::reload(&conf, &pw)).await;
    match result {
        Ok(Ok(r)) => {
            crate::snapshot_api::common::reload_repository_clients(node).await;
            entry.insert("secure_setting_names".into(), json!(r.names));
            entry.insert("keystore_path".into(), json!(r.path));
            entry.insert("keystore_digest".into(), json!(r.digest));
            if let Some(ms) = r.last_modified.and_then(chrono::DateTime::<chrono::Utc>::from_timestamp_millis) {
                entry.insert("keystore_last_modified_time".into(), json!(ms.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()));
            }
        }
        Ok(Err(e)) => {
            entry.insert("reload_exception".into(), e.to_json());
        }
        Err(e) => {
            entry.insert("reload_exception".into(), json!({"type": "exception", "reason": e.to_string()}));
        }
    }
    Value::Object(entry)
}

fn transport_tls_enabled(node: &ClusterNode) -> bool {
    match node.config.settings.get("xpack.security.transport.ssl.enabled") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true",
        _ => false,
    }
}

/// The REST handler.
pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let st = node.state();
    let segs = req.segments();
    let spec = match segs.get(1) {
        Some(s) if s != "reload_secure_settings" => s.clone(),
        _ => String::new(),
    };
    let password = match req.json() {
        None => None,
        Some(Value::Object(body)) => match body.get("secure_settings_password") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(other) => Some(other.to_string()),
        },
        Some(_) => return RestResp::error(400, "parse_exception", "request body is required to be an object"),
    };
    let has_password = password.as_deref().is_some_and(|p| !p.is_empty());
    let ids = crate::rest::nodes::resolve_nodes(&st, &node.local.id, &spec);
    let node_local = ids.len() == 1 && ids.contains(&node.local.id);
    if has_password && !node_local && !transport_tls_enabled(node) {
        return RestResp::error(
            500,
            "exception",
            "Secure settings cannot be updated cluster wide when TLS for the transport layer is not enabled. Enable TLS or use the API with a `_local` filter on each node.",
        );
    }
    let pw = password.unwrap_or_default();
    let mut nodes = Map::new();
    let mut failures = Vec::new();
    for id in &ids {
        if *id == node.local.id {
            nodes.insert(id.clone(), reload_local(node, &pw).await);
            continue;
        }
        let Some(addr) = node.address_of(id) else { continue };
        let req = NodeReq { password: Some(pw.clone()) };
        match node.transport.send_request::<_, String>(addr, ACTION, &req, Duration::from_secs(30)).await {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(v) => {
                    nodes.insert(id.clone(), v);
                }
                Err(e) => failures.push(failure(id, &e.to_string())),
            },
            Err(e) => failures.push(failure(id, &e.to_string())),
        }
    }
    let mut header = json!({"total": nodes.len() + failures.len(), "successful": nodes.len(), "failed": failures.len()});
    if !failures.is_empty() {
        header["failures"] = Value::Array(failures);
    }
    RestResp::json(200, &json!({"_nodes": header, "cluster_name": st.cluster_name, "nodes": nodes}))
}

/// A `FailedNodeException` in the `_nodes` header.
fn failure(id: &str, reason: &str) -> Value {
    json!({
        "type": "failed_node_exception",
        "reason": format!("Failed node [{id}]"),
        "node_id": id,
        "caused_by": {"type": "exception", "reason": reason}
    })
}
