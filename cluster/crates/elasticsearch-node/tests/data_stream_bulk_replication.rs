//! A bulk into a data stream whose backing-index primaries are spread over
//! two nodes: the part a node forwards to the other node's primary must go
//! through that primary's write path (replicate before the ack, refresh
//! every copy for `refresh=true`, publish dynamic mappings), exactly like
//! the part it executes itself. Before, the forwarded part ran straight on
//! the remote engine: its replica only caught up through the background
//! pump, unrefreshed, and its dynamic mapping update never reached the
//! master (`search` right after `bulk?refresh=true` missed documents on
//! that replica; `GET _mapping` lacked the new field).

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::time::Duration;

/// The nodes' bound HTTP ports (OS-assigned), set once the test has
/// started them (one cluster per test binary).
static HTTP_PORTS: std::sync::OnceLock<[u16; 2]> = std::sync::OnceLock::new();

fn http() -> [u16; 2] {
    *HTTP_PORTS.get().expect("nodes started")
}

/// Node `i` on port 0; `seeds`: the transport ports of the nodes started
/// before it.
fn settings(i: usize, dir: &std::path::Path, seeds: &[u16]) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!("ds-bulk-replication"));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join(format!("n{i}")).to_string_lossy()));
    put("path.logs", json!(dir.join(format!("logs{i}")).to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn call(c: &reqwest::Client, node: usize, method: reqwest::Method, path: &str, body: Option<String>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", http()[node]));
    if let Some(b) = body {
        let ct = if path.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
        r = r.header("content-type", ct).body(b);
    }
    let resp = r.send().await.expect("request");
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forwarded_backing_index_bulk_is_replicated_refreshed_and_mapped() {
    let dir = evidence::retained_dir("data_stream_bulk_replication");
    let mut http_ports = [0u16; 2];
    let mut transport_ports: Vec<u16> = Vec::new();
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(settings(i, dir.path(), &transport_ports)).expect("config");
        let (node, listener) = ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node {i}: {e}"));
        elasticsearch_node::rest::register(&node);
        elasticsearch_node::snapshot_api::register(&node);
        http_ports[i] = ports::http_port(&node);
        transport_ports.push(ports::transport_port(&node));
        tokio::spawn(async move {
            if let Err(e) = elasticsearch_node::front::serve(node, listener).await {
                panic!("node {i}: {e}");
            }
        });
    }
    HTTP_PORTS.set(http_ports).expect("one cluster per test binary");
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &http(), None, 2, Duration::from_secs(60)).await
        .unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    let template = json!({"index_patterns": ["logs-rr-*"], "priority": 1000, "data_stream": {},
        "template": {"settings": {"index": {"number_of_shards": 2, "number_of_replicas": 1}},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "message": {"type": "text"}}}}});
    let (st, v) = call(&c, 0, reqwest::Method::PUT, "/_index_template/rr", Some(template.to_string())).await;
    assert_eq!(st, 200, "{v}");
    for round in 0..4 {
        // Alternate the coordinating node: each round, one shard's primary
        // is on the other node.
        let via = round % 2;
        let (st, v) = call(&c, via, reqwest::Method::PUT, "/_data_stream/logs-rr-test", None).await;
        assert_eq!(st, 200, "{v}");
        let (st, v) = call(&c, via, reqwest::Method::GET, "/_cluster/health/logs-rr-test?wait_for_status=green&timeout=60s", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["status"], "green", "{v}");
        assert_eq!(v["timed_out"], false, "{v}");
        let mut body = String::new();
        for d in 0..12 {
            body.push_str(&format!("{{\"create\":{{}}}}\n{{\"@timestamp\":\"2024-02-12T10:{d:02}:00Z\",\"message\":\"m {d}\",\"dyn{round}\":\"v\"}}\n"));
        }
        let (st, v) = call(&c, via, reqwest::Method::POST, "/logs-rr-test/_bulk?refresh=true", Some(body)).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["errors"], false, "{v}");
        // Every copy of every shard is searchable right after the ack.
        let (_, shards) = call(&c, via, reqwest::Method::GET, "/_cat/shards/.ds-logs-rr-test*?format=json&h=shard,prirep,docs,node", None).await;
        // Capture actual search visibility separately from cat counters so a
        // mismatch identifies a stale statistic versus missing replica data.
        let mut visibility = Vec::new();
        for n in 0..2 {
            let (_, count) = call(&c, n, reqwest::Method::GET, "/logs-rr-test/_count?preference=_local", None).await;
            visibility.push(count);
        }
        let rows = shards.as_array().cloned().unwrap_or_default();
        assert_eq!(rows.len(), 4, "{shards}");
        for s in ["0", "1"] {
            let docs: Vec<&str> = rows.iter().filter(|r| r["shard"] == s).map(|r| r["docs"].as_str().unwrap_or("")).collect();
            assert!(docs.len() == 2 && docs[0] == docs[1], "round {round}: shard {s} copies differ: {shards}; local search visibility: {visibility:?}; bulk: {v}");
        }
        for n in 0..2 {
            let (_, count) = call(&c, n, reqwest::Method::GET, "/logs-rr-test/_count", None).await;
            assert_eq!(count["count"], 12, "round {round} via node {n}: {count}");
            // The dynamic field is in the published mapping on every node.
            let (_, mapping) = call(&c, n, reqwest::Method::GET, "/logs-rr-test/_mapping", None).await;
            let all = mapping.as_object().map(|m| m.values().all(|v| v["mappings"]["properties"].get(format!("dyn{round}")).is_some())).unwrap_or(false);
            assert!(all, "round {round} via node {n}: dyn{round} missing: {mapping}");
        }
        let (st, v) = call(&c, via, reqwest::Method::DELETE, "/_data_stream/logs-rr-test", None).await;
        assert_eq!(st, 200, "{v}");
    }
}
