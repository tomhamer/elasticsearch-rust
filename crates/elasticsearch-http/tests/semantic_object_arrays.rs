//! Finding 200: a semantic_text field inside an ordinary (non-nested)
//! object array is embedded: every element's value becomes a chunk (ES
//! reads source fields with `XContentMapValues.extractValue`, which walks
//! lists), so semantic search and highlighting see all of them.

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
}

#[tokio::test]
async fn semantic_values_in_object_arrays_are_embedded() {
    elasticsearch_engine::inference::mock::enable();
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let es = Es { base: format!("http://{addr}"), c: Client::new() };
    let (st, v) = es.call(Method::PUT, "/_inference/text_embedding/dense", Some(json!({
        "service": "text_embedding_test_service",
        "service_settings": {"model": "m", "dimensions": 4, "similarity": "cosine", "api_key": "x"},
        "task_settings": {}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.call(Method::PUT, "/docs", Some(json!({"mappings": {"properties": {
        "sections": {"properties": {"text": {"type": "semantic_text", "inference_id": "dense"}}}
    }}}))).await;
    assert_eq!(st, 200, "{v}");
    for (id, doc) in [
        ("arr", json!({"sections": [{"text": "first passage"}, {"text": "second passage"}]})),
        ("single", json!({"sections": {"text": "only passage"}})),
        ("empty", json!({"sections": []})),
        ("null", json!({"sections": [{"text": null}]})),
    ] {
        let (st, v) = es.call(Method::PUT, &format!("/docs/_doc/{id}?refresh=true"), Some(doc)).await;
        assert_eq!(st, 201, "{id}: {v}");
    }
    // Both array elements are chunks of the document.
    let (st, v) = es.call(Method::POST, "/docs/_search", Some(json!({
        "query": {"semantic": {"field": "sections.text", "query": "first passage"}},
        "highlight": {"fields": {"sections.text": {"number_of_fragments": 5, "order": "score"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let hits = v["hits"]["hits"].as_array().unwrap();
    let arr = hits.iter().find(|h| h["_id"] == "arr").unwrap_or_else(|| panic!("the object-array document is not found semantically: {v}"));
    let mut frags: Vec<String> = arr["highlight"]["sections.text"].as_array().unwrap_or_else(|| panic!("{arr}")).iter().map(|f| f.as_str().unwrap().to_string()).collect();
    frags.sort();
    assert_eq!(frags, vec!["first passage".to_string(), "second passage".to_string()], "{arr}");
    assert!(hits.iter().any(|h| h["_id"] == "single"), "{v}");
    assert!(!hits.iter().any(|h| h["_id"] == "empty" || h["_id"] == "null"), "{v}");
}
