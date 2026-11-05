//! M13: a cluster node's single-document writes are limited by indexing
//! pressure (they used to run as coordinator-local primary work, which is
//! never rejected, although no coordinating stage held their bytes).

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
    put("cluster.name", json!("single-doc-pressure"));
    put("node.name", json!("n0"));
    put("path.data", json!(dir.join("n0").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.type", json!("single-node"));
    put("indexing_pressure.memory.limit", json!("1kb"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_document_writes_are_limited_by_indexing_pressure() {
    let dir = evidence::retained_dir("single_doc_pressure");
    let cfg = NodeConfig::from_settings(settings(dir.path())).expect("config");
    // Started as `elasticsearch_node::run` does, keeping the node to read
    // its OS-assigned HTTP port.
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node: {e}"));
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let http = ports::http_port(&node);
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
    let big = "x".repeat(4096);
    // Alone, a write larger than the limit is admitted (ES: already read).
    let r = c.put(url("/docs/_doc/alone")).json(&json!({"text": big})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    // Other write bytes in flight on this node: the same write is rejected.
    let held = elasticsearch_engine::indexing_pressure::global().coordinating(100).expect("room for the held bytes");
    let r = c.put(url("/docs/_doc/limited")).json(&json!({"text": big})).send().await.unwrap();
    let status = r.status().as_u16();
    let body = r.text().await.unwrap();
    drop(held);
    assert_eq!(status, 429, "a single-document write passed the indexing-pressure limit: {body}");
    assert!(body.contains("es_rejected_execution_exception"), "{body}");
}
