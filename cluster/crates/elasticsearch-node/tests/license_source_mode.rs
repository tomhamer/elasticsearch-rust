//! Licensing is hardwired to enterprise/trial (AGENTS.md "Licensing: always
//! enterprise"): a fresh cluster reports an active trial before anything
//! is requested, a synthetic-source data stream created on it (by the
//! master's scratch engine) keeps synthetic source with no `start_trial`
//! call, and requests for a basic license (`start_basic`, `PUT` / `DELETE
//! /_license`) are acknowledged without changing the effective license.

mod common;
use common::start;
use reqwest::Method;
use serde_json::json;

/// `index.mapping.source.mode` of data stream `ds`'s write index.
async fn source_mode(c: &common::Cluster, node: usize, ds: &str) -> String {
    let (status, body) = c.call(node, Method::GET, &format!("/_data_stream/{ds}"), None).await;
    assert_eq!(status, 200, "{body}");
    let backing = body["data_streams"][0]["indices"][0]["index_name"].as_str().unwrap().to_string();
    let (status, body) = c.call(node, Method::GET, &format!("/{backing}/_settings?flat_settings=true&include_defaults=true"), None).await;
    assert_eq!(status, 200, "{body}");
    let s = &body[&backing];
    s["settings"]["index.mapping.source.mode"].as_str().or_else(|| s["defaults"]["index.mapping.source.mode"].as_str()).unwrap_or("").to_ascii_lowercase()
}

async fn assert_trial(c: &common::Cluster, node: usize, when: &str) {
    let (status, body) = c.call(node, Method::GET, "/_license", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["license"]["type"], "trial", "{when}: {body}");
    assert_eq!(body["license"]["status"], "active", "{when}: {body}");
    let (_, info) = c.call(node, Method::GET, "/_xpack", None).await;
    for f in ["ml", "graph", "ccr", "searchable_snapshots", "watcher"] {
        assert_eq!(info["features"][f]["available"], json!(true), "{when}: {f} {info}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fresh_cluster_is_trial_and_data_streams_keep_synthetic_source() {
    let c = start(&[]).await;
    for node in [c.master, c.other] {
        assert_trial(&c, node, "fresh").await;
    }
    let (status, body) = c.call(c.master, Method::PUT, "/_index_template/synth", Some(json!({
        "index_patterns": ["synth-*"], "data_stream": {}, "priority": 200,
        "template": {"settings": {"index": {"codec": "best_compression", "mapping": {"source.mode": "synthetic"}, "number_of_replicas": 0}},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"properties": {"ip": {"type": "ip"}}}}}}}))).await;
    assert_eq!(status, 200, "{body}");
    // No start_trial: created from either node, the stream is synthetic.
    for (i, node) in [c.other, c.master].into_iter().enumerate() {
        let ds = format!("synth-{i}");
        let (status, body) = c.call(node, Method::PUT, &format!("/_data_stream/{ds}"), None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(source_mode(&c, node, &ds).await, "synthetic", "{ds}");
    }
    let (status, body) = c.call(c.other, Method::POST, "/synth-0/_doc?refresh=true",
        Some(json!({"@timestamp": "2026-01-01T00:00:00Z", "host": {"ip": ["10.0.0.2", "10.0.0.1", "10.0.0.2"]}}))).await;
    assert_eq!(status, 201, "{body}");
    let (status, body) = c.call(c.master, Method::POST, "/synth-0/_search", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"]["hits"][0]["_source"]["host"]["ip"], json!(["10.0.0.1", "10.0.0.2"]), "{body}");
    // Requests for basic succeed and change nothing effective.
    let (status, body) = c.call(c.master, Method::POST, "/_license/start_basic?acknowledge=true", None).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = c.call(c.other, Method::DELETE, "/_license", None).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = c.call(c.master, Method::POST, "/_license/start_trial?acknowledge=true", None).await;
    assert_eq!(status, 200, "{body}");
    for node in [c.master, c.other] {
        assert_trial(&c, node, "after start_basic / DELETE / start_trial").await;
    }
    let (_, body) = c.call(c.other, Method::GET, "/_license/trial_status", None).await;
    assert_eq!(body, json!({"eligible_to_start_trial": false}));
    let (_, body) = c.call(c.other, Method::GET, "/_license/basic_status", None).await;
    assert_eq!(body, json!({"eligible_to_start_basic": true}));
    let (status, body) = c.call(c.other, Method::PUT, "/_data_stream/synth-after-basic", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(source_mode(&c, c.other, "synth-after-basic").await, "synthetic");
}
