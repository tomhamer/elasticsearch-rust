//! Mutable owners survive an empty-index interval while cluster maintenance runs.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mutable_vectors_remain_searchable_after_empty_maintenance_interval() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (status, body) = cl.call(0, Method::PUT, "/mutable-maintenance", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 1},
        "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4,
            "index_options": {"type": "hnsw"}}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, health) = cl.call(0, Method::GET,
        "/_cluster/health/mutable-maintenance?wait_for_status=green&wait_for_no_initializing_shards=true&timeout=60s", None).await;
    assert_eq!(status, 200, "{health}");
    assert_eq!(health["status"], "green", "{health}");
    assert_eq!(health["timed_out"], false, "{health}");
    let mut bulk = String::new();
    for id in 0..100 { bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{id}\"}}}}\n{{\"v\":[1,0,0,{}]}}\n", id+1)); }
    let result: Value = cl.c.post(format!("http://127.0.0.1:{}/mutable-maintenance/_bulk?refresh=true", ports.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(result["errors"], false, "{result}");
    let mut deletes = String::new();
    for id in 0..100 { deletes.push_str(&format!("{{\"delete\":{{\"_id\":\"{id}\"}}}}\n")); }
    let result: Value = cl.c.post(format!("http://127.0.0.1:{}/mutable-maintenance/_bulk?refresh=true", ports.http[1]))
        .header("content-type", "application/x-ndjson").body(deletes).send().await.unwrap().json().await.unwrap();
    assert_eq!(result["errors"], false, "{result}");
    // Exercise repeated scheduler ticks with empty mutable graphs, checking
    // that the two-node fixture stays responsive rather than only sleeping.
    for _ in 0..8 {
        let (status, body) = cl.call(0, Method::GET, "/_cluster/health", None).await;
        assert_eq!(status, 200, "{body}"); assert_eq!(body["status"], "green", "{body}");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let (status, body) = cl.call(0, Method::PUT, "/mutable-maintenance/_doc/new?refresh=true", Some(json!({"v":[1,0,0,1]}))).await;
    assert_eq!(status, 201, "{body}");
    for node in 0..2 {
        let (status, body) = cl.call(node, Method::POST, "/mutable-maintenance/_search", Some(json!({
            "knn":{"field":"v","query_vector":[1,0,0,1],"k":10,"num_candidates":100}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let hits = body["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "{body}"); assert_eq!(hits[0]["_id"], "new");
    }
}
