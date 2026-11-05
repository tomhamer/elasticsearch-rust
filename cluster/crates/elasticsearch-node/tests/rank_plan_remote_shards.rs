//! A coordinated `retriever` / legacy `rank` search on a node that holds
//! none of the target's shards. The plan phase runs on the coordinating
//! node's engine and must not be pinned to a local shard 0. Before the fix
//! it answered `503 [rprobe][0] no active shard copy` (rank-rrf YAML failed
//! whenever the fixture's election put shard 0 on the other node).

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
    put("cluster.name", json!("rank-plan-remote"));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join(format!("n{i}")).to_string_lossy()));
    put("path.logs", json!(dir.join(format!("logs{i}")).to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    put("xpack.license.self_generated.type", json!("trial"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

/// `(status, body, Warning headers)`.
async fn call(c: &reqwest::Client, node: usize, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value, Vec<String>) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", http()[node]));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request");
    let status = resp.status().as_u16();
    let warnings = resp.headers().get_all("warning").iter().filter_map(|v| v.to_str().ok().map(str::to_string)).collect();
    (status, resp.json().await.unwrap_or(Value::Null), warnings)
}

fn ids(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| h["_id"].as_str().map(str::to_string)).collect();
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rank_search_coordinated_by_a_node_without_the_shards() {
    let dir = evidence::retained_dir("rank_plan_remote_shards");
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

    use reqwest::Method as M;
    // One index per holder, so whichever node won the election, some search
    // is coordinated by a node (master or not) that holds none of the shards.
    for holder in ["n1", "n0"] {
        let index = format!("rprobe-{holder}");
        let (st, v, _) = call(&c, 0, M::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.include._name": holder},
            "mappings": {"properties": {"text": {"type": "text"}, "vector": {"type": "dense_vector", "dims": 1, "index": true, "similarity": "l2_norm"}}}}))).await;
        assert_eq!(st, 200, "{v}");
        let (st, v, _) = call(&c, 0, M::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(st, 200, "{v}");
        for (id, text, x) in [("1", "term term", 0.0), ("2", "other", 1.0), ("3", "term", 2.0)] {
            let (st, v, _) = call(&c, 0, M::PUT, &format!("/{index}/_doc/{id}"), Some(json!({"text": text, "vector": [x]}))).await;
            assert!(st == 200 || st == 201, "{v}");
        }
        call(&c, 0, M::POST, &format!("/{index}/_refresh"), None).await;
        let (_, shards, _) = call(&c, 0, M::GET, &format!("/_cat/shards/{index}?format=json"), None).await;
        assert!(shards.as_array().unwrap().iter().all(|s| s["node"] == holder), "{shards}");

        for node in 0..2 {
            let ctx = format!("{index} via n{node}");
            let (st, v, _) = call(&c, node, M::POST, &format!("/{index}/_search"), Some(json!({"retriever": {"rrf": {"retrievers": [
                {"standard": {"query": {"match": {"text": "term"}}}},
                {"knn": {"field": "vector", "query_vector": [0.0], "k": 3, "num_candidates": 3}}]}}}))).await;
            assert_eq!(st, 200, "{ctx}: {v}");
            assert_eq!(ids(&v), ["1", "2", "3"], "{ctx}: {v}");

            // Legacy `rank`: the plan phase's deprecation warning reaches the client.
            let (st, v, warnings) = call(&c, node, M::POST, &format!("/{index}/_search"), Some(json!({
                "sub_searches": [{"query": {"match": {"text": "term"}}}, {"query": {"match": {"text": "other"}}}],
                "rank": {"rrf": {"rank_window_size": 10, "rank_constant": 1}}}))).await;
            assert_eq!(st, 200, "{ctx}: {v}");
            assert_eq!(ids(&v), ["1", "2", "3"], "{ctx}: {v}");
            assert!(warnings.iter().any(|w| w.contains("Deprecated field [rank] used, replaced by [retriever]")), "{ctx}: {warnings:?}");

            // The plan phase's validation errors.
            let (st, v, _) = call(&c, node, M::POST, &format!("/{index}/_search"), Some(json!({
                "sub_searches": [{"query": {"match": {"text": "term"}}}, {"query": {"match": {"text": "other"}}}],
                "rank": {"rrf": {"rank_window_size": 10, "rank_constant": 0}}}))).await;
            assert_eq!(st, 400, "{ctx}: {v}");
            assert!(v.to_string().contains("[rank_constant] must be greater or equal to [1] for [rrf]"), "{ctx}: {v}");
        }
    }
}
