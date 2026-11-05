//! Stats aggregation must retain per-index vector details while omitting them
//! from totals, regardless of response level or metric selection.
use reqwest::{Client, Method};
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread")]
async fn stats_levels_preserve_vector_details_and_selected_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let node = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(node, listener).await; });
    let client = Client::new();
    async fn call(client: &Client, base: &str, method: Method, path: &str, body: Option<Value>) -> Value {
        let mut req = client.request(method, format!("{base}{path}"));
        if let Some(body) = body { req = req.json(&body); }
        let response = req.send().await.unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert!(status.is_success(), "{path}: {status}: {body}");
        body
    }
    for (name, count) in [("stats-a", 1), ("stats-b", 2)] {
        call(&client, &base, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4, "index_options": {"type": "flat"}}}}
        }))).await;
        for id in 0..count {
            call(&client, &base, Method::PUT, &format!("/{name}/_doc/{id}?refresh=true"), Some(json!({"v": [1.0, 0.0, 0.0, 0.0]}))).await;
        }
    }
    for level in ["indices", "shards", "cluster"] {
        let body = call(&client, &base, Method::GET, &format!("/stats-*/_stats?level={level}"), None).await;
        for total in ["primaries", "total"] {
            assert_eq!(body["_all"][total]["docs"]["count"], 3, "{body}");
            assert_eq!(body["_all"][total]["dense_vector"]["value_count"], 3, "{body}");
            assert!(body["_all"][total]["dense_vector"]["off_heap"].get("fielddata").is_none(), "{body}");
        }
        if level == "cluster" {
            assert!(body.get("indices").is_none(), "{body}");
        } else {
            for (name, count) in [("stats-a", 1), ("stats-b", 2)] {
                for total in ["primaries", "total"] {
                    assert_eq!(body["indices"][name][total]["docs"]["count"], count, "{body}");
                    assert!(body["indices"][name][total]["dense_vector"]["off_heap"]["fielddata"]["v"].is_object(), "{body}");
                }
                if level == "shards" {
                    assert!(body["indices"][name]["shards"]["0"][0]["dense_vector"]["off_heap"]["fielddata"]["v"].is_object(), "{body}");
                }
            }
        }
    }
    let docs = call(&client, &base, Method::GET, "/stats-*/_stats/docs", None).await;
    assert_eq!(docs["_all"]["primaries"].as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>(), ["docs"]);
    assert_eq!(docs["_all"]["primaries"]["docs"]["count"], 3);
    assert!(docs["indices"]["stats-a"]["primaries"].get("dense_vector").is_none());
}
