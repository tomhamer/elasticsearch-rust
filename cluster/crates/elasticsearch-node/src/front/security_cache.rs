//! The `_clear_cache` security APIs (ES `TransportClearRolesCacheAction`,
//! `TransportClearApiKeyCacheAction`, `TransportClearRealmCacheAction`,
//! `TransportClearPrivilegesCacheAction`,
//! `TransportClearServiceAccountTokenCacheAction`): nodes actions that
//! invalidate the security caches on every node of the cluster. The engine
//! API handler authorizes the request, then calls [`broadcast`] through the
//! `SecurityCacheClear` request extension.
use super::*;
use elasticsearch_engine::error::EsError;
use serde_json::Map;

const ACTION: &str = "cluster:admin/xpack/security/cache/clear[n]";

#[derive(Serialize, Deserialize)]
struct NodeReq {
    kind: String,
    names: Vec<String>,
    usernames: Vec<String>,
}

pub(super) fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION, move |r: NodeReq| {
        let n = n.clone();
        async move {
            n.engine.security.clear_cache(&r.kind, &r.names, &r.usernames).map_err(|e| RemoteError::new(&e.kind, e.reason, e.status))?;
            Ok::<String, RemoteError>(n.local.name.clone())
        }
    });
}

/// Clear `kind` on this node and on every other node of the applied state;
/// the ES nodes response (`_nodes` header with failures, `nodes` by id).
pub(super) async fn broadcast(node: &ClusterNode, kind: String, names: Vec<String>, usernames: Vec<String>) -> Result<Value, EsError> {
    node.engine.security.clear_cache(&kind, &names, &usernames)?;
    let state = node.state();
    let mut nodes = Map::new();
    nodes.insert(node.local.id.clone(), json!({"name": node.local.name}));
    let mut failures = Vec::new();
    for id in state.nodes.keys() {
        if *id == node.local.id {
            continue;
        }
        let Some(address) = node.address_of(id) else {
            failures.push(failure(id, "no transport address"));
            continue;
        };
        let request = NodeReq { kind: kind.clone(), names: names.clone(), usernames: usernames.clone() };
        match node.transport.send_request::<_, String>(address, ACTION, &request, Duration::from_secs(30)).await {
            Ok(name) => {
                nodes.insert(id.clone(), json!({"name": name}));
            }
            Err(error) => failures.push(failure(id, &error.to_string())),
        }
    }
    let mut header = json!({"total": nodes.len() + failures.len(), "successful": nodes.len(), "failed": failures.len()});
    if !failures.is_empty() {
        header["failures"] = Value::Array(failures);
    }
    Ok(json!({"_nodes": header, "cluster_name": state.cluster_name, "nodes": nodes}))
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
