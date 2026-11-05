//! Two real nodes: index stats and node stats of a multi-shard index sum
//! every shard copy, whether the copy's shard engine is on the answering
//! node (measured in process) or on the other node (asked over the
//! transport), and render the same with `?human` and `level=shards`.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

const SHARDS: u64 = 4;
const DOCS: u64 = 240;

fn u(v: &Value) -> u64 {
    v.as_u64().unwrap_or_else(|| panic!("not a count: {v}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_shard_index_and_node_stats_sum_every_copy() {
    let cl = start(&[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/sums", Some(json!({
        "settings": {"number_of_shards": SHARDS, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/sums?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(status, 200, "{body}");
    let mut bulk = String::new();
    for i in 0..DOCS {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{{\"k\":\"key-{i}\",\"n\":{i}}}\n"));
    }
    let body: Value = cl.c.post(format!("http://127.0.0.1:{}/sums/_bulk?refresh=true", cl.p.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(body["errors"], false, "{body}");
    // Shards spread over both nodes: each node's answer mixes local and
    // remote copies.
    let (status, shards) = cl.call(0, Method::GET, "/_cat/shards/sums?format=json&h=shard,node", None).await;
    assert_eq!(status, 200, "{shards}");
    assert_eq!(shards.as_array().unwrap().len() as u64, SHARDS, "{shards}");
    for node in 0..2 {
        // Rally's `index-stats` call.
        let (status, s) = cl.call(node, Method::GET, "/_all/_stats/_all", None).await;
        assert_eq!(status, 200, "{s}");
        assert!(u(&s["_shards"]["total"]) >= SHARDS, "{s}");
        for section in ["primaries", "total"] {
            let idx = &s["indices"]["sums"][section];
            assert_eq!(u(&idx["docs"]["count"]), DOCS, "{node} {section}: {s}");
            assert_eq!(u(&idx["indexing"]["index_total"]), DOCS, "{node} {section}: {s}");
            assert_eq!(u(&idx["shard_stats"]["total_count"]), SHARDS, "{node} {section}: {s}");
            assert!(u(&idx["segments"]["count"]) >= SHARDS, "{node} {section}: {s}");
            assert!(idx["translog"].is_object() && idx["store"].is_object(), "{s}");
        }
        // Per-copy entries add up to the index figures.
        let (status, sh) = cl.call(node, Method::GET, "/sums/_stats?level=shards", None).await;
        assert_eq!(status, 200, "{sh}");
        let per = sh["indices"]["sums"]["shards"].as_object().unwrap_or_else(|| panic!("{sh}"));
        assert_eq!(per.len() as u64, SHARDS, "{sh}");
        let copies: Vec<&Value> = per.values().flat_map(|v| v.as_array().unwrap().iter()).collect();
        let sum = |path: &[&str]| -> u64 { copies.iter().map(|c| u(path.iter().fold(*c, |v, k| &v[*k]))).sum() };
        let primaries = &sh["indices"]["sums"]["primaries"];
        assert_eq!(sum(&["docs", "count"]), DOCS, "{sh}");
        assert_eq!(sum(&["indexing", "index_total"]), u(&primaries["indexing"]["index_total"]), "{sh}");
        assert_eq!(sum(&["segments", "count"]), u(&primaries["segments"]["count"]), "{sh}");
        // `?human` renders the summed sizes and times.
        let (status, h) = cl.call(node, Method::GET, "/sums/_stats/indexing,store,translog?human", None).await;
        assert_eq!(status, 200, "{h}");
        let hp = &h["indices"]["sums"]["primaries"];
        assert_eq!(u(&hp["indexing"]["index_total"]), DOCS, "{h}");
        assert!(hp["indexing"]["index_time"].is_string() && hp["store"]["size"].is_string() && hp["translog"]["size"].is_string(), "{h}");
        // Rally's `node-stats` call: every node reports its own copies,
        // shard engines included.
        let (status, ns) = cl.call(node, Method::GET, "/_nodes/stats/_all?level=indices", None).await;
        assert_eq!(status, 200, "{ns}");
        let nodes = ns["nodes"].as_object().unwrap();
        assert_eq!(nodes.len(), 2, "{ns}");
        let per_node = |path: &[&str]| -> u64 { nodes.values().map(|n| path.iter().fold(&n["indices"]["indices"]["sums"], |v, k| &v[*k]).as_u64().unwrap_or(0)).sum() };
        assert_eq!(per_node(&["docs", "count"]), DOCS, "{ns}");
        assert_eq!(per_node(&["indexing", "index_total"]), DOCS, "{ns}");
        for n in nodes.values() {
            for section in ["os", "process", "jvm", "thread_pool", "fs", "transport", "http", "breakers"] {
                assert!(n[section].is_object(), "{section}: {ns}");
            }
            assert!(n["fs"]["total"]["total_in_bytes"].is_u64(), "{ns}");
        }
    }
}
