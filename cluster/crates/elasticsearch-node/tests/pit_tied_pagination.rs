//! Composite PIT cursor boundaries must include document zero on each shard.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tied_pit_pages_cover_every_shard_in_both_directions() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (status, v) = cl.call(0, Method::PUT, "/ties", Some(json!({
        "settings": {"number_of_shards": 4, "number_of_replicas": 1},
        "mappings": {"properties": {"rank": {"type": "long"}}}
    }))).await;
    assert_eq!(status, 200, "{v}");
    let mut bulk = String::new();
    for id in 0..32 {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{id}\"}}}}\n{{\"rank\":1}}\n"));
    }
    let v: Value = cl.c.post(format!("http://127.0.0.1:{}/ties/_bulk?refresh=true", ports.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
    let (_, health) = cl.call(0, Method::GET, "/_cluster/health/ties?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(health["status"], "green", "{health}");
    for direction in ["asc", "desc"] {
        for size in [1, 3, 10] {
            let (status, v) = cl.call(0, Method::POST, "/ties/_pit?keep_alive=1m", None).await;
            assert_eq!(status, 200, "{v}");
            let mut pit = v["id"].clone();
            let mut cursor: Option<Value> = None;
            let mut ids = Vec::new();
            for page in 0..40 {
                let mut body = json!({"size": size, "pit": {"id": pit, "keep_alive": "1m"},
                    "sort": [{"rank": "asc"}, {"_shard_doc": direction}]});
                if let Some(value) = &cursor { body["search_after"] = value.clone(); }
                let (status, v) = cl.call(page % 2, Method::POST, "/_search", Some(body)).await;
                assert_eq!(status, 200, "{v}");
                if let Some(id) = v.get("pit_id") { pit = id.clone(); }
                let hits = v["hits"]["hits"].as_array().unwrap();
                if hits.is_empty() { break; }
                ids.extend(hits.iter().map(|h| h["_id"].as_str().unwrap().to_string()));
                cursor = Some(hits.last().unwrap()["sort"].clone());
            }
            let (status, v) = cl.call(0, Method::DELETE, "/_pit", Some(json!({"id": pit}))).await;
            assert_eq!(status, 200, "{v}");
            assert_eq!(ids.len(), 32, "{direction}, page size {size}: {ids:?}");
            assert_eq!(ids.into_iter().collect::<BTreeSet<_>>(), (0..32).map(|i| i.to_string()).collect());
        }
    }
}
