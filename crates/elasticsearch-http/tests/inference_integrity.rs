//! Finding 127: a hosted embedding provider that answers 200 with a
//! malformed body (too few or too many rows, repeated or out-of-range
//! indices, rows of another shape) fails the document's ingest atomically;
//! nothing is indexed. Complete rows in shuffled index order still index,
//! each vector with its own chunk. The provider is a deterministic fixture
//! installed as the inference transport (this test binary's own process).

use elasticsearch_engine::inference::providers::{self, HttpResult, Transport};
use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

/// Embeddings by first word (`alpha`, `beta`, anything else); a mode word in
/// any input makes the answer malformed in one way.
struct Fixture;

fn vector(text: &str) -> Value {
    if text.starts_with("alpha") {
        json!([1.0, 0.0, 0.0, 0.1])
    } else if text.starts_with("beta") {
        json!([0.0, 1.0, 0.0, 0.1])
    } else {
        json!([0.0, 0.0, 1.0, 0.1])
    }
}

impl Transport for Fixture {
    fn post(&self, _url: &str, _headers: &[(String, String)], body: &Value, _timeout: Duration) -> HttpResult {
        let input: Vec<String> = match &body["input"] {
            Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        };
        let all = input.join(" ");
        let row = |i: usize, t: &str| json!({"index": i, "embedding": vector(t)});
        let rows: Vec<Value> = if all.contains("ZERO") {
            Vec::new()
        } else if all.contains("ONE") {
            vec![row(0, &input[0])]
        } else if all.contains("THREE") {
            vec![row(0, &input[0]), row(1, &input[1]), row(2, &input[0])]
        } else if all.contains("DUP") {
            vec![row(0, &input[0]), row(0, &input[1])]
        } else if all.contains("OOR") {
            vec![row(0, &input[0]), row(7, &input[1])]
        } else if all.contains("SHAPE") {
            vec![row(0, &input[0]), json!({"index": 1, "embedding": [1.0, 2.0]})]
        } else if all.contains("SHUF") {
            input.iter().enumerate().rev().map(|(i, t)| row(i, t)).collect()
        } else {
            input.iter().enumerate().map(|(i, t)| row(i, t)).collect()
        };
        Ok((200, json!({"object": "list", "data": rows, "model": "m"})))
    }
}

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
async fn malformed_provider_answers_fail_the_document_atomically() {
    providers::set_transport(Arc::new(Fixture));
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let es = Es { base: format!("http://{addr}"), c: Client::new() };
    let (st, v) = es.call(Method::PUT, "/_inference/text_embedding/fx", Some(json!({
        "service": "openai",
        "service_settings": {"api_key": "k", "model_id": "m", "url": "http://fixture.invalid/v1/embeddings", "dimensions": 4, "similarity": "cosine"},
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.call(Method::PUT, "/sem", Some(json!({"mappings": {"properties": {"body": {
        "type": "semantic_text", "inference_id": "fx", "chunking_settings": {"strategy": "none"}
    }}}}))).await;
    assert_eq!(st, 200, "{v}");
    for mode in ["ZERO", "ONE", "THREE", "DUP", "OOR", "SHAPE"] {
        let (st, v) = es.call(Method::PUT, &format!("/sem/_doc/{mode}?refresh=true"), Some(json!({"body": [format!("alpha {mode}"), format!("beta {mode}")]}))).await;
        assert!(st >= 400, "{mode}: indexed despite a malformed provider answer: {v}");
        let (st, _) = es.call(Method::GET, &format!("/sem/_doc/{mode}"), None).await;
        assert_eq!(st, 404, "{mode}");
    }
    // Shuffled but complete: indexed, each chunk with its own vector.
    let (st, v) = es.call(Method::PUT, "/sem/_doc/shuffled?refresh=true", Some(json!({"body": ["alpha SHUF", "beta SHUF"]}))).await;
    assert_eq!(st, 201, "{v}");
    for (query, best) in [("alpha", "alpha SHUF"), ("beta", "beta SHUF")] {
        let (st, v) = es.call(Method::POST, "/sem/_search", Some(json!({
            "query": {"semantic": {"field": "body", "query": query}},
            "highlight": {"fields": {"body": {"number_of_fragments": 1}}}
        }))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
        assert_eq!(v["hits"]["hits"][0]["highlight"]["body"][0], best, "{query}: {v}");
    }
}
