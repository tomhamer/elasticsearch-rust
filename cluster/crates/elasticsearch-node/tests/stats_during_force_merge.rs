//! Two real nodes: cat statistics remain correct while force merge runs.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cat_statistics_during_force_merge() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (status, body) = cl.call(0, Method::PUT, "/merge-stats", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 1, "refresh_interval": "-1"},
        "mappings": {"properties": {"value": {"type": "keyword"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    for batch in 0..3 {
        let mut bulk = String::new();
        for row in 0..5000 {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n{{\"value\":\"unique-value-{}\"}}\n", batch*5000+row, batch*5000+row));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/merge-stats/_bulk?refresh=true", ports.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
        let (status, body) = cl.call(0, Method::POST, "/merge-stats/_flush", None).await;
        assert_eq!(status, 200, "{body}");
    }
    let client = cl.c.clone();
    let merge = tokio::spawn(async move {
        client.post(format!("http://127.0.0.1:{}/merge-stats/_forcemerge?max_num_segments=1", ports.http[0]))
            .timeout(Duration::from_secs(60)).send().await.unwrap()
    });
    for i in 0..12 {
        let response = cl.c.get(format!("http://127.0.0.1:{}/_cat/indices/merge-stats?format=json&h=index,docs.count,health", ports.http[i%2]))
            .timeout(Duration::from_secs(5)).send().await.expect("cat statistics blocked during merge");
        assert!(response.status().is_success());
        let rows: Value = response.json().await.unwrap();
        assert_eq!(rows[0]["docs.count"], "15000", "{rows}");
        assert_eq!(rows[0]["health"], "green", "{rows}");
        let response = cl.c.get(format!("http://127.0.0.1:{}/merge-stats/_stats/docs,merge,flush?level=shards", ports.http[i%2]))
            .timeout(Duration::from_secs(5)).send().await.expect("counter statistics blocked during merge");
        assert!(response.status().is_success());
        let stats: Value = response.json().await.unwrap();
        assert!(stats["indices"]["merge-stats"]["total"]["merges"].is_object(), "{stats}");
        assert_eq!(stats["indices"]["merge-stats"]["primaries"]["docs"]["count"], 15000, "{stats}");
        assert!(stats["indices"]["merge-stats"]["total"].get("translog").is_none(), "{stats}");
        let response = cl.c.get(format!("http://127.0.0.1:{}/_nodes/stats/indices/docs,merge,flush?level=shards", ports.http[i%2]))
            .timeout(Duration::from_secs(5)).send().await.expect("node counter statistics blocked during merge");
        assert!(response.status().is_success());
        let stats: Value = response.json().await.unwrap();
        let nodes = stats["nodes"].as_object().unwrap();
        assert_eq!(nodes.len(), 2, "{stats}");
        for node in nodes.values() {
            assert!(node["indices"]["merges"].is_object(), "{stats}");
            assert!(node["indices"].get("translog").is_none(), "{stats}");
        }

    }
    let response = merge.await.unwrap();
    assert!(response.status().is_success());
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["_shards"]["failed"], 0, "{body}");
    let (status, body) = cl.call(1, Method::GET, "/merge-stats/_count", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["count"], 15000, "{body}");
}
