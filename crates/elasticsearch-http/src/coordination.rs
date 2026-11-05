//! Voting configuration exclusions (`_cluster/voting_config_exclusions`).
//!
//! A single node cannot give up its vote, so ids/names that do not belong to this
//! node are recorded as absent exclusions (like ES does for nodes that already left);
//! excluding the node itself is refused because it would leave no voting node.

use crate::{Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, NODE_ID};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn store() -> &'static Mutex<HashMap<String, Vec<(String, String)>>> {
    static S: OnceLock<Mutex<HashMap<String, Vec<(String, String)>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

const ABSENT: &str = "_absent_";

/// `metadata.cluster_coordination.voting_config_exclusions` of the cluster state.
pub fn exclusions(node: &Node) -> Value {
    let st = store().lock().unwrap();
    Value::Array(st.get(&node.cluster_uuid).map(|v| v.iter().map(|(id, name)| json!({"node_id": id, "node_name": name})).collect()).unwrap_or_default())
}

pub fn handle(node: &Node, req: &Req) -> R {
    match &req.method {
        &Method::POST => {
            let list = |k: &str| req.param(k).map(|v| v.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect::<Vec<_>>());
            let (ids, names) = (list("node_ids"), list("node_names"));
            let one = list("node_name");
            let set = [ids.is_some(), names.is_some(), one.is_some()].iter().filter(|b| **b).count();
            if set != 1 {
                return Err(EsError::illegal_argument("You must set [node_names] or [node_ids] but not both"));
            }
            let mut add: Vec<(String, String)> = Vec::new();
            for id in ids.unwrap_or_default() {
                if id == NODE_ID {
                    return Err(EsError::illegal_argument("add voting config exclusions request for [node_ids] would leave the cluster without a voting node"));
                }
                add.push((id, ABSENT.to_string()));
            }
            for n in names.into_iter().chain(one).flatten() {
                if n == node.name {
                    return Err(EsError::illegal_argument("add voting config exclusions request for [node_names] would leave the cluster without a voting node"));
                }
                add.push((ABSENT.to_string(), n));
            }
            let mut st = store().lock().unwrap();
            let cur = st.entry(node.cluster_uuid.clone()).or_default();
            for a in add {
                if !cur.contains(&a) {
                    cur.push(a);
                }
            }
            Ok(Resp::text(String::new()))
        }
        &Method::DELETE => {
            store().lock().unwrap().remove(&node.cluster_uuid);
            Ok(Resp::text(String::new()))
        }
        _ => crate::method_not_allowed(req, &["POST", "DELETE"]),
    }
}
