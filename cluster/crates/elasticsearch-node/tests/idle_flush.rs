//! A cluster node flushes shard copies that hold uncommitted operations but
//! saw no write for `indices.memory.shard_inactive_time` (ES `flushOnIdle`),
//! so a restart does not replay (and re-index) an idle translog tail.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use serde_json::{json, Value};
use std::time::Duration;

fn settings(dir: &std::path::Path) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!("idle-flush"));
    put("node.name", json!("n0"));
    put("path.data", json!(dir.join("n0").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.type", json!("single-node"));
    put("indices.memory.shard_inactive_time", json!("1s"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

/// The node's OS-assigned HTTP port, set once it has started.
static HTTP: std::sync::OnceLock<u16> = std::sync::OnceLock::new();

async fn uncommitted(c: &reqwest::Client) -> i64 {
    let http = HTTP.get().expect("node started");
    let v: Value = c.get(format!("http://127.0.0.1:{http}/docs/_stats/translog")).send().await.unwrap().json().await.unwrap();
    v["_all"]["primaries"]["translog"]["uncommitted_operations"].as_i64().unwrap_or(-1)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn idle_shard_is_flushed() {
    let dir = evidence::retained_dir("idle_flush");
    let cfg = NodeConfig::from_settings(settings(dir.path())).expect("config");
    // Started as `elasticsearch_node::run` does, keeping the node to read
    // its OS-assigned HTTP port.
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node: {e}"));
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let http = ports::http_port(&node);
    HTTP.set(http).expect("one node per test binary");
    tokio::spawn(async move {
        if let Err(e) = elasticsearch_node::front::serve(node, listener).await {
            panic!("node: {e}");
        }
    });
    let c = reqwest::Client::new();
    let url = |p: &str| format!("http://127.0.0.1:{http}{p}");
    let mut up = false;
    for _ in 0..120 {
        if c.get(url("/_cluster/health?wait_for_status=green&timeout=1s")).send().await.is_ok_and(|r| r.status().is_success()) {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(up, "node did not start");
    let r = c.put(url("/docs")).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    for i in 0..20 {
        let r = c.put(url(&format!("/docs/_doc/{i}"))).json(&json!({"n": i})).send().await.unwrap();
        assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    }
    assert!(uncommitted(&c).await > 0, "ops are uncommitted right after indexing");
    // Idle for 1s, checked every 5s ticks: committed well within 30s.
    let mut flushed = false;
    for _ in 0..60 {
        if uncommitted(&c).await == 0 {
            flushed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(flushed, "idle shard was never flushed");
}
