//! Polls every node's *local* applied cluster state (`GET _cluster/state`
//! is served from the node's own state, never forwarded) and records
//! `(node, term, version, state_uuid, master, voting configuration,
//! content hash)`.

use crate::api::{call, client};
use crate::cluster::{Cluster, ProcState};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize)]
pub struct Obs {
    pub t: f64,
    pub node: usize,
    pub term: i64,
    pub version: i64,
    pub state_uuid: String,
    /// Master node *name* (`node-3`), None when the node has no master.
    pub master: Option<String>,
    /// `metadata.cluster_coordination.last_committed_config` and
    /// `last_accepted_config`, as node *names* where the id is known
    /// (a bootstrap placeholder or an unknown id is kept as is).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub committed_config: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accepted_config: Vec<String>,
    pub hash: u64,
}

pub struct Monitor {
    pub obs: Mutex<Vec<Obs>>,
    /// First full state text per (version, hash), kept to show forks.
    pub samples: Mutex<BTreeMap<(i64, u64), String>>,
}

/// Canonical (key-sorted) rendering of a JSON value.
pub fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let sorted: BTreeMap<&String, String> = m.iter().map(|(k, v)| (k, canonical(v))).collect();
            let parts: Vec<String> = sorted.into_iter().map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap_or_default(), v)).collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(canonical).collect::<Vec<_>>().join(",")),
        other => other.to_string(),
    }
}

impl Monitor {
    pub fn new() -> Monitor {
        Monitor { obs: Mutex::new(Vec::new()), samples: Mutex::new(BTreeMap::new()) }
    }

    pub async fn run(self: Arc<Self>, cluster: Arc<Cluster>, t0: Instant, stop: Arc<AtomicBool>) {
        let http = client(Duration::from_secs(1));
        // Node id -> name, accumulated over every poll: a voting
        // configuration can name a node the current state no longer has.
        let mut names: BTreeMap<String, String> = BTreeMap::new();
        while !stop.load(Ordering::SeqCst) {
            let futs = (0..cluster.n).filter(|i| cluster.state(*i) == ProcState::Running).map(|i| {
                let url = format!("{}/_cluster/state/master_node,version,nodes,metadata,routing_table", cluster.url(i));
                let http = http.clone();
                async move { (i, call(&http, "GET", &url, None, None).await) }
            });
            for (i, r) in futures::future::join_all(futs).await {
                let Ok((200, v)) = r else { continue };
                if let Some(nodes) = v["nodes"].as_object() {
                    for (id, n) in nodes {
                        if let Some(name) = n["name"].as_str() {
                            names.insert(id.clone(), name.to_string());
                        }
                    }
                }
                let config = |key: &str| -> Vec<String> {
                    v["metadata"]["cluster_coordination"][key]
                        .as_array()
                        .map(|a| a.iter().filter_map(|id| id.as_str()).map(|id| names.get(id).cloned().unwrap_or_else(|| id.to_string())).collect())
                        .unwrap_or_default()
                };
                let (committed_config, accepted_config) = (config("last_committed_config"), config("last_accepted_config"));
                let master = v["master_node"].as_str().map(|id| v["nodes"][id]["name"].as_str().unwrap_or(id).to_string());
                let term = v["metadata"]["cluster_coordination"]["term"].as_i64().unwrap_or(-1);
                let version = v["version"].as_i64().unwrap_or(-1);
                let state_uuid = v["state_uuid"].as_str().unwrap_or("").to_string();
                // Content that must be identical for one committed version.
                let content = format!("{}|{}|{}", canonical(&v["metadata"]), canonical(&v["routing_table"]), v["master_node"]);
                let mut h = DefaultHasher::new();
                content.hash(&mut h);
                let hash = h.finish();
                if master.is_some() {
                    self.samples.lock().entry((version, hash)).or_insert(content);
                }
                self.obs.lock().push(Obs { t: t0.elapsed().as_secs_f64(), node: i, term, version, state_uuid, master, committed_config, accepted_config, hash });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}
