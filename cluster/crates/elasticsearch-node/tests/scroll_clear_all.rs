//! `DELETE /_search/scroll/_all` (and body `{"scroll_id":"_all"}`) on a
//! two-node cluster frees the scroll contexts of every node and of every
//! per-shard engine, and answers like ES `ClearScrollResponse`:
//! `{"succeeded":true,"num_freed":N}`, 404 when nothing was freed.

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
    put("cluster.name", json!("scroll-clear-all"));
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

async fn call(c: &reqwest::Client, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", http()[0]));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request");
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

async fn open_scroll(c: &reqwest::Client) -> String {
    let (st, v) = call(c, reqwest::Method::POST, "/s/_search?scroll=5m", Some(json!({"size": 1, "sort": ["n"]}))).await;
    assert_eq!(st, 200, "{v}");
    v["_scroll_id"].as_str().expect("scroll id").to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clear_all_frees_every_node_and_shard_engine() {
    let dir = evidence::retained_dir("scroll_clear_all");
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

    let (st, v) = call(&c, reqwest::Method::PUT, "/s", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "integer"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, reqwest::Method::GET, "/_cluster/health/s?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(st, 200, "{v}");
    for n in 0..20 {
        let (st, v) = call(&c, reqwest::Method::PUT, &format!("/s/_doc/{n}"), Some(json!({"n": n}))).await;
        assert!(st == 201 || st == 200, "{v}");
    }
    call(&c, reqwest::Method::POST, "/s/_refresh", None).await;

    // Two scrolls over a 2-shard index: one engine context per shard each,
    // shard 1 living in its own `~k1` engine.
    let s1 = open_scroll(&c).await;
    let _s2 = open_scroll(&c).await;
    let (st, v) = call(&c, reqwest::Method::DELETE, "/_search/scroll/_all", None).await;
    assert_eq!((st, &v), (200, &json!({"succeeded": true, "num_freed": 4})));
    let (st, v) = call(&c, reqwest::Method::POST, "/_search/scroll", Some(json!({"scroll": "1m", "scroll_id": s1}))).await;
    assert_eq!(st, 404, "cleared scroll must be gone: {v}");

    // Body form.
    let _s3 = open_scroll(&c).await;
    let (st, v) = call(&c, reqwest::Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": "_all"}))).await;
    assert_eq!((st, &v), (200, &json!({"succeeded": true, "num_freed": 2})));

    // Nothing left: ES answers NOT_FOUND with `succeeded` still true.
    let (st, v) = call(&c, reqwest::Method::DELETE, "/_search/scroll/_all", None).await;
    assert_eq!((st, &v), (404, &json!({"succeeded": true, "num_freed": 0})));
}
