//! Unquantized vectors use one mutable HNSW per shard. Check publication,
//! searchable results, and durable graph checkpointing instead of waiting
//! for the retired per-segment graph optimizer.

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
    put("cluster.name", json!("vector-optimizer"));
    put("node.name", json!("n0"));
    put("path.data", json!(dir.join("n0").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.type", json!("single-node"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

fn has_mutable_checkpoint(dir: &std::path::Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    rd.flatten().any(|e| {
        let p = e.path();
        if p.is_dir() { return has_mutable_checkpoint(&p); }
        p.extension().is_some_and(|e| e == "rmvc") && e.metadata().is_ok_and(|m| m.len() > 0)
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn appended_vectors_publish_and_checkpoint_mutable_graph() {
    let dir = evidence::retained_dir("vector_optimizer");
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

    let mapping = json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4, "similarity": "cosine", "index_options": {"type": "hnsw"}}}}
    });
    let r = c.put(url("/vecs")).json(&mapping).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    let mut bulk = String::new();
    for i in 0..200 {
        let f = i as f32;
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n"));
        bulk.push_str(&json!({"v": [f.sin(), f.cos(), (f * 0.5).sin(), 1.0]}).to_string());
        bulk.push('\n');
    }
    let r = c.post(url("/vecs/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());

    let body: Value = r.json().await.unwrap();
    assert_eq!(body["errors"], false, "{body}");
    let response = c.post(url("/vecs/_flush")).send().await.unwrap();
    assert!(response.status().is_success(), "{}", response.text().await.unwrap());
    assert!(has_mutable_checkpoint(dir.path()), "flush did not persist the mutable HNSW graph");

    let q = json!({"knn": {"field": "v", "query_vector": [0.0, 1.0, 0.0, 1.0], "k": 3, "num_candidates": 20}, "_source": false});
    let v: Value = c.post(url("/vecs/_search")).json(&q).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["hits"]["hits"].as_array().map(Vec::len), Some(3), "{v}");
    assert_eq!(v["hits"]["hits"][0]["_id"], "0", "{v}");
}
