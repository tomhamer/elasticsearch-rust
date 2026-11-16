//! #545 / #546: semantic_text byte and bit models, and supplied embeddings.
//! - A bit model's query vector is one byte per 8 dimensions: a direct kNN on
//!   the field accepts it and scores by Hamming similarity.
//! - Supplied (`_inference_fields`) dense embeddings are validated like any
//!   indexed vector: a wrong width or non-numeric element fails the item
//!   (only it, in a bulk); valid ones are searchable.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
}

impl Es {
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn bulk(&self, p: &str, lines: &[Value]) -> Value {
        let body: String = lines.iter().map(|l| l.to_string() + "\n").collect();
        let resp = self.c.post(format!("{}{}", self.base, p)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        serde_json::from_str(&resp.text().await.unwrap()).unwrap()
    }
}

#[tokio::test]
async fn semantic_byte_bit_models_and_supplied_embeddings() {
    elasticsearch_engine::inference::mock::enable();
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let es = Es { base: format!("http://{addr}"), c: Client::new() };
    // A bit model: indexing embeds, the semantic query scores 1.0 for the
    // identical text, a direct kNN takes the packed byte query vector.
    let (st, v) = es.call(Method::PUT, "/_inference/text_embedding/bits", Some(json!({"service": "text_embedding_test_service",
        "service_settings": {"model": "m", "dimensions": 16, "api_key": "k", "similarity": "L2_NORM", "element_type": "bit"}, "task_settings": {}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.call(Method::PUT, "/b", Some(json!({"mappings": {"properties": {"f": {"type": "semantic_text", "inference_id": "bits"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.call(Method::PUT, "/b/_doc/1?refresh=true", Some(json!({"f": "inference test"}))).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.call(Method::POST, "/b/_search", Some(json!({"query": {"semantic": {"field": "f", "query": "inference test"}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert!((v["hits"]["hits"][0]["_score"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-4, "{v}");
    let (st, v) = es.call(Method::POST, "/b/_search", Some(json!({"knn": {"field": "f", "query_vector": [0, 0], "k": 1, "num_candidates": 5}}))).await;
    assert_eq!(st, 200, "a 2-byte query of a 16-bit model: {v}");
    // Supplied embeddings of a 4-dim float model.
    let (st, v) = es.call(Method::PUT, "/_inference/text_embedding/dense4", Some(json!({"service": "text_embedding_test_service",
        "service_settings": {"model": "m", "dimensions": 4, "api_key": "k", "similarity": "cosine"}, "task_settings": {}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.call(Method::PUT, "/s", Some(json!({"mappings": {"properties": {"f": {"type": "semantic_text", "inference_id": "dense4"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let ms = json!({"task_type": "text_embedding", "dimensions": 4, "similarity": "cosine", "element_type": "float"});
    let supplied = |emb: Value| json!({"f": "abcd", "_inference_fields": {"f": {"inference": {"inference_id": "dense4", "model_settings": ms,
        "chunks": {"f": [{"start_offset": 0, "end_offset": 4, "embeddings": emb}]}}}}});
    let r = es.bulk("/s/_bulk?refresh=true", &[
        json!({"index": {"_id": "ok"}}), supplied(json!([0.1, 0.2, 0.3, 0.4])),
        json!({"index": {"_id": "short"}}), supplied(json!([0.1, 0.2, 0.3])),
        json!({"index": {"_id": "text"}}), supplied(json!([0.1, "x", 0.3, 0.4])),
    ]).await;
    let status: Vec<i64> = r["items"].as_array().unwrap().iter().map(|i| i["index"]["status"].as_i64().unwrap()).collect();
    assert_eq!(status[0], 201, "{r}");
    assert_eq!(status[1], 400, "a 3-dim chunk of a 4-dim model fails: {r}");
    assert_eq!(status[2], 400, "a non-numeric element fails: {r}");
    let (st, v) = es.call(Method::POST, "/s/_search", Some(json!({"query": {"knn": {"field": "f", "query_vector": [0.1, 0.2, 0.3, 0.4], "k": 5, "num_candidates": 5}}}))).await;
    assert_eq!(st, 200, "{v}");
    let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().filter_map(|h| h["_id"].as_str()).collect();
    assert_eq!(ids, vec!["ok"], "{v}");
}
