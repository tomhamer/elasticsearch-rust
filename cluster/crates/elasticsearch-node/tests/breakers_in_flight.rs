//! The production (cluster) HTTP edge holds client bodies on the
//! `inflight_requests` breaker, once per node: two `elasticsearch-node`
//! processes (each with its own breakers) on OS-assigned ports, read back
//! from their `node.portsfile` files.
//! - An oversized search body trips at either node's edge with ES's 429,
//!   and each node counts its own trip.
//! - A slow search over several shards local to the coordinating node
//!   holds its body once while it runs, not once more per local shard
//!   dispatch (`inflight_requests.estimated_size` sampled mid-request).
//! - `_bulk` (streamed in ES 9) and cluster settings never trip.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// One node process; killed when dropped.
struct Node {
    child: Option<Child>,
    http: u16,
    transport: u16,
    name: String,
    log: PathBuf,
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Start node `name` on OS-assigned ports with `seeds` (transport ports of
/// the nodes started before it) and wait until it reports the ports it
/// bound (`node.portsfile`); started again if a bind failed meanwhile.
async fn spawn(dir: &Path, name: &str, seeds: &[u16]) -> Node {
    let bin = env!("CARGO_BIN_EXE_elasticsearch-node");
    let log = dir.join(format!("{name}.log"));
    let logs = dir.join(format!("{name}-logs"));
    for _ in 0..8 {
        ports::clear_ports_files(&logs);
        let offset = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0) as usize;
        let out = std::fs::OpenOptions::new().create(true).append(true).open(&log).unwrap();
        let mut args: Vec<String> = Vec::new();
        let mut e = |k: &str, v: &str| {
            args.push("-E".into());
            args.push(format!("{k}={v}"));
        };
        e("cluster.name", "breakers-in-flight");
        e("node.name", name);
        e("path.data", &dir.join(name).to_string_lossy());
        e("path.logs", &logs.to_string_lossy());
        e("network.host", "127.0.0.1");
        e("http.port", "0");
        e("transport.port", "0");
        e("node.portsfile", "true");
        let seeds = ports::seeds(seeds);
        if !seeds.is_empty() {
            e("discovery.seed_hosts", &seeds.join(","));
        }
        e("cluster.initial_master_nodes", "n0,n1");
        e("xpack.security.enabled", "false");
        for w in ["low", "high", "flood_stage"] {
            e(&format!("cluster.routing.allocation.disk.watermark.{w}"), "1b");
        }
        let child = Command::new(bin).args(&args).stdin(Stdio::null()).stdout(out.try_clone().unwrap()).stderr(out).spawn().expect("spawn elasticsearch-node");
        // Killed on drop, also if a check below fails.
        let mut node = Node { child: Some(child), http: 0, transport: 0, name: name.into(), log: log.clone() };
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let exited = node.child.as_mut().expect("spawned").try_wait().expect("poll child");
            if let Some(status) = exited {
                let text = std::fs::read(&log).unwrap_or_default();
                let since = String::from_utf8_lossy(&text[offset.min(text.len())..]).into_owned();
                assert!(ports::is_addr_in_use(&since), "{name} exited ({status}) before binding:\n{since}");
                node.child = None;
                break;
            }
            // `http.ports` is written after `transport.ports`.
            if let Some(http) = ports::read_ports_file(&logs, "http") {
                if let Some(transport) = ports::read_ports_file(&logs, "transport") {
                    node.http = http;
                    node.transport = transport;
                    return node;
                }
            }
            assert!(Instant::now() < deadline, "{name} did not report its ports: {}", std::fs::read_to_string(&log).unwrap_or_default());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    panic!("{name} found no free ports: {}", std::fs::read_to_string(&log).unwrap_or_default());
}

async fn call(c: &reqwest::Client, node: &Node, method: reqwest::Method, path: &str, body: Option<String>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", node.http));
    if let Some(b) = body {
        let ct = if path.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
        r = r.header("content-type", ct).body(b);
    }
    let resp = r.send().await.expect("request");
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

/// This node's own `inflight_requests` trip count.
async fn tripped(c: &reqwest::Client, node: &Node) -> u64 {
    let (_, stats) = call(c, node, reqwest::Method::GET, "/_nodes/stats/breaker", None).await;
    stats["nodes"].as_object().into_iter().flat_map(|m| m.values()).find(|n| n["name"] == node.name.as_str())
        .and_then(|n| n["breakers"]["inflight_requests"]["tripped"].as_u64()).unwrap_or(0)
}

fn pad_values(kb: usize) -> Vec<String> {
    (0..kb * 1024 / 224).map(|i| format!("{i:0>220}")).collect()
}

/// A search body of about `kb` kilobytes (a terms query over many values).
fn search_body(kb: usize) -> String {
    json!({"query": {"terms": {"pad": pad_values(kb)}}}).to_string()
}

/// [`search_body`] that also takes a while: every doc is scored by a script
/// with a loop (the padding terms clause only adds size).
fn slow_search_body(kb: usize) -> String {
    json!({"size": 0, "query": {"bool": {
        "must": [{"script_score": {"query": {"match_all": {}}, "script": {"source": "long t = 0; for (int i = 0; i < 20000; i++) { t += i; } return 1;"}}}],
        "should": [{"terms": {"pad": pad_values(kb)}}]
    }}}).to_string()
}

/// n0's `inflight_requests` estimate (bytes held now, before the overhead).
async fn in_flight_estimate(c: &reqwest::Client, node: &Node) -> u64 {
    let (_, stats) = call(c, node, reqwest::Method::GET, "/_nodes/stats/breaker", None).await;
    stats["nodes"].as_object().into_iter().flat_map(|m| m.values()).find(|n| n["name"] == node.name.as_str())
        .and_then(|n| n["breakers"]["inflight_requests"]["estimated_size_in_bytes"].as_u64()).unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_bodies_are_held_once_at_the_cluster_edge() {
    let dir = evidence::retained_dir("breakers_in_flight");
    let n0 = spawn(dir.path(), "n0", &[]).await;
    let n1 = spawn(dir.path(), "n1", &[n0.transport]).await;
    let nodes = [n0, n1];
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &[nodes[0].http, nodes[1].http], None, 2, Duration::from_secs(120)).await
        .unwrap_or_else(|e| panic!("cluster did not form: {e}: {}", std::fs::read_to_string(&nodes[0].log).unwrap_or_default()));
    // Three shards, all on n0: a search coordinated by n0 dispatches to
    // each of them locally.
    let (st, v) = call(&c, &nodes[0], reqwest::Method::PUT, "/inf", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"}}).to_string())).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, &nodes[0], reqwest::Method::GET, "/_cluster/health/inf?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(st, 200, "{v}");
    // Docs for the slow search to score (a _bulk: streamed, not held).
    let doc = |i: usize| format!("{{\"index\":{{}}}}\n{{\"i\":{i}}}\n");
    let (st, v) = call(&c, &nodes[0], reqwest::Method::POST, "/inf/_bulk?refresh=true", Some((0..2000).map(doc).collect())).await;
    assert_eq!(st, 200, "{v}");
    // 1mb with overhead 2: a body over 512kb trips.
    let (st, v) = call(&c, &nodes[0], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"transient": {"network.breaker.inflight_requests.limit": "1mb"}}).to_string())).await;
    assert_eq!(st, 200, "{v}");
    for _ in 0..50 {
        // The limit reaches n0 with the cluster state.
        let (_, s) = call(&c, &nodes[0], reqwest::Method::GET, "/_nodes/stats/breaker", None).await;
        let limit = s["nodes"].as_object().into_iter().flat_map(|m| m.values()).find(|n| n["name"] == "n0").map(|n| n["breakers"]["inflight_requests"]["limit_size_in_bytes"].clone());
        if limit == Some(json!(1 << 20)) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // A slow ~300kb search over n0's three local shards: while it runs, n0
    // holds its body once (the edge's copy); held again per local shard
    // dispatch it would hold ~4x, and with the overhead trip the limit.
    let body = slow_search_body(300);
    assert!(body.len() > 256 << 10 && body.len() < 400 << 10);
    let before = tripped(&c, &nodes[0]).await;
    let search = {
        let (c, url, body) = (c.clone(), format!("http://127.0.0.1:{}/inf/_search", nodes[0].http), body.clone());
        tokio::spawn(async move {
            let r = c.post(url).header("content-type", "application/json").body(body).send().await.expect("search");
            let st = r.status().as_u16();
            (st, r.json::<Value>().await.unwrap_or(Value::Null))
        })
    };
    let mut peak = 0;
    while !search.is_finished() {
        peak = peak.max(in_flight_estimate(&c, &nodes[0]).await);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (st, v) = search.await.unwrap();
    assert_eq!(st, 200, "a body held once must pass: {v}");
    let len = body.len() as u64;
    assert!(peak >= len * 9 / 10, "the search ran long enough to see its body held: peak {peak}, body {len}");
    assert!(peak < 2 * len, "the body is held once on the coordinating node: peak {peak}, body {len}");
    assert_eq!(tripped(&c, &nodes[0]).await, before, "no trip for the held-once body");
    // An oversized body trips at either node's edge; each counts its own.
    let big = search_body(700);
    for node in &nodes {
        let before = tripped(&c, node).await;
        let mut last = (0, Value::Null);
        for _ in 0..50 {
            last = call(&c, node, reqwest::Method::POST, "/inf/_search", Some(big.clone())).await;
            if last.0 == 429 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let (st, v) = last;
        assert_eq!(st, 429, "{}: {v}", node.name);
        assert_eq!(v["error"]["type"], "circuit_breaking_exception", "{v}");
        let reason = v["error"]["reason"].as_str().unwrap_or("");
        assert!(reason.starts_with("[inflight_requests] Data too large, data for [<http_request>]"), "{reason}");
        assert_eq!(v["error"]["bytes_limit"], 1 << 20, "{v}");
        assert_eq!(v["error"]["durability"], "TRANSIENT", "{v}");
        assert!(tripped(&c, node).await > before, "{} counted its trip", node.name);
    }
    // A _bulk as large goes through (ES 9 streams it), as do small searches.
    let doc = |i: usize| format!("{{\"index\":{{}}}}\n{{\"pad\":\"{}\",\"i\":{i}}}\n", "x".repeat(200));
    let bulk: String = (0..3000).map(doc).collect();
    let (st, v) = call(&c, &nodes[1], reqwest::Method::POST, "/inf/_bulk?refresh=true", Some(bulk)).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false, "{v}");
    let (st, v) = call(&c, &nodes[1], reqwest::Method::POST, "/inf/_search", Some(json!({"size": 0}).to_string())).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 5000, "{v}");
    // Cluster settings never trip: the limit can always be raised again.
    let (st, v) = call(&c, &nodes[1], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"transient": {"network.breaker.inflight_requests.limit": null}}).to_string())).await;
    assert_eq!(st, 200, "{v}");
}
