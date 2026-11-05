//! A two-node cluster for integration tests that must not depend on which
//! node wins the master election: tests keep shard copies off the elected
//! master ([`Cluster::exclude_master`]) and talk to both nodes.
#![allow(dead_code)]

pub mod procs;
pub mod evidence;
pub mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

/// One two-node cluster per test: its nodes' bound (http, transport)
/// ports (OS-assigned; see [`ports`]).
#[derive(Clone, Copy, Debug)]
pub struct Ports {
    pub http: [u16; 2],
    pub transport: [u16; 2],
}

/// Node `i`'s settings: port 0 for both listeners, the nodes started before
/// it (known transport ports in `p`) as seed hosts.
fn settings(cluster: &str, p: Ports, i: usize, dir: &std::path::Path, extra: &[(&str, &str)]) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join(format!("n{i}")).to_string_lossy()));
    put("path.logs", json!(dir.join(format!("logs{i}")).to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(&p.transport)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    for (k, v) in extra {
        put(k, json!(v));
    }
    s
}

/// A started cluster: client, ports, and the master / other node positions.
pub struct Cluster {
    pub c: reqwest::Client,
    pub p: Ports,
    pub master: usize,
    pub other: usize,
    /// Basic credentials (`elastic`, bootstrap password) of a secured cluster.
    pub auth: Option<(String, String)>,
    /// The nodes' data directory (see [`data_dir`]).
    pub dir: std::path::PathBuf,
}

impl Cluster {
    pub async fn call(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        call_as(&self.c, self.p, self.auth.as_ref(), node, method, path, body).await
    }

    pub fn master_name(&self) -> String {
        format!("n{}", self.master)
    }

    /// Keep every shard copy allocated from now on off the elected master.
    pub async fn exclude_master(&self) {
        let (st, v) = self.call(self.master, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.routing.allocation.exclude._name": self.master_name()}}))).await;
        assert_eq!(st, 200, "{v}");
    }

    /// Whether the started primary of `index` is on the non-master node.
    pub async fn primary_off_master(&self, index: &str) -> bool {
        let (_, shards) = self.call(self.master, Method::GET, &format!("/_cat/shards/{index}?format=json&h=prirep,state,node"), None).await;
        shards.as_array().is_some_and(|a| a.iter().any(|s| s["prirep"] == "p" && s["state"] == "STARTED" && s["node"] == format!("n{}", self.other).as_str()))
    }
}

pub async fn call(c: &reqwest::Client, p: Ports, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    call_as(c, p, None, node, method, path, body).await
}

pub async fn call_as(c: &reqwest::Client, p: Ports, auth: Option<&(String, String)>, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", p.http[node]));
    if let Some((user, password)) = auth {
        r = r.basic_auth(user, Some(password));
    }
    if let Some(b) = body {
        r = r.json(&b);
    }
    // Status 0: no answer (a node still starting).
    let Ok(resp) = r.send().await else { return (0, Value::Null) };
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

/// A fresh retained data directory for the cluster named `cluster`.
/// Set `RUSTIC_TEST_ARTIFACT_DIR` to group artifacts beneath a run result root.
///
/// The nodes run on the test's tokio runtime and their coordinator persists
/// cluster state into it; the runtime (and the actor threads finishing a
/// persist) outlive the test body. A directory removed when the test body
/// returned (a `TempDir` dropped with the `Cluster`) made a late persist
/// fail, and a node that cannot persist its coordination state exits the
/// process (`coord.rs`): the whole test binary died with status 1, printing
/// nothing. Retain directories for collection; a dead process is not proof
/// of success. Later tests must never delete earlier failure evidence.
pub fn data_dir(cluster: &str) -> std::path::PathBuf {
    scratch_dir(cluster)
}

/// [`data_dir`] by name: for tests that start in-process nodes themselves.
#[cfg(unix)]
pub use libc::{SIGCONT, SIGSTOP, SIGTERM};
// Placeholders: `signal` does nothing without Unix signals.
#[cfg(not(unix))]
pub const SIGCONT: i32 = 18;
#[cfg(not(unix))]
pub const SIGSTOP: i32 = 19;
#[cfg(not(unix))]
pub const SIGTERM: i32 = 15;

/// Send `sig` to process `pid` (0: only check that it exists). Windows has
/// no signals: nothing is sent and the process counts as present.
#[cfg(unix)]
pub fn signal(pid: u32, sig: i32) -> bool {
    // SAFETY: signalling a process by id; no memory is shared.
    unsafe { libc::kill(pid as i32, sig) == 0 }
}

#[cfg(not(unix))]
pub fn signal(_pid: u32, sig: i32) -> bool {
    sig == 0
}

pub fn scratch_dir(tag: &str) -> std::path::PathBuf {
    evidence::scratch_dir(tag)
}

/// Start an in-process node as `elasticsearch_node::run` does, serving HTTP
/// on the test's runtime; its bound ports are in `node.local`.
pub async fn spawn_in_process(cfg: NodeConfig) -> Arc<ClusterNode> {
    let name = cfg.node_name.clone();
    let (node, http) = ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node {name}: {e}"));
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let n = node.clone();
    tokio::spawn(async move {
        if let Err(e) = elasticsearch_node::front::serve(n, http).await {
            panic!("node {name}: {e}");
        }
    });
    node
}

/// Start both nodes (with `extra` node settings) on OS-assigned ports and
/// wait for the cluster to form.
pub async fn start(extra: &[(&str, &str)]) -> Cluster {
    let cluster = ports::unique("two-nodes");
    let dir = data_dir(&cluster);
    let mut p = Ports { http: [0; 2], transport: [0; 2] };
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(settings(&cluster, p, i, &dir, extra)).expect("config");
        let node = spawn_in_process(cfg).await;
        p.http[i] = ports::http_port(&node);
        p.transport[i] = ports::transport_port(&node);
    }
    let c = reqwest::Client::new();
    let auth = extra.iter().find(|(k, _)| *k == "bootstrap.password").map(|(_, v)| ("elastic".to_string(), v.to_string()));
    // Every node formed and recovered, not just the first to answer.
    ports::wait_formed(&c, &p.http, auth.as_ref(), 2, Duration::from_secs(60)).await
        .unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    let (_, m) = call_as(&c, p, auth.as_ref(), 0, Method::GET, "/_cat/master?format=json", None).await;
    let master = match m[0]["node"].as_str() {
        Some("n0") => 0,
        Some("n1") => 1,
        other => panic!("no master: {other:?}"),
    };
    Cluster { c, p, master, other: 1 - master, auth, dir }
}

