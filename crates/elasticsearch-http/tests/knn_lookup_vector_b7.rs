//! SIDE 016: a standalone `query_vector_builder.lookup` reads the reference
//! document's vector from the reference index (as ES's
//! `LookupQueryVectorBuilder` and the cluster coordinator do): it works with
//! the reference `_source` disabled, and an unrelated same-named field of
//! another type in the searched index does not reject a valid vector.

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
async fn lookup_vector_reads_the_reference_index_vector() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let es = Es { base: format!("http://{addr}"), c: Client::new() };
    let vector = |dims: u32| json!({"type": "dense_vector", "dims": dims, "index": true, "similarity": "l2_norm", "index_options": {"type": "flat"}});
    for (name, source) in [("ref-on", true), ("ref-off", false)] {
        let (st, v) = es.call(Method::PUT, &format!("/{name}"), Some(json!({"mappings": {"_source": {"enabled": source}, "properties": {"reference": {"properties": {"v": vector(2)}}}}}))).await;
        assert_eq!(st, 200, "{v}");
        let (st, v) = es.call(Method::PUT, &format!("/{name}/_doc/q?refresh=true"), Some(json!({"reference": {"v": [1.0, 0.0]}}))).await;
        assert_eq!(st, 201, "{v}");
    }
    // The searched index maps `reference.v` as a number: irrelevant here.
    let (st, v) = es.call(Method::PUT, "/target", Some(json!({"mappings": {"properties": {"embedding": vector(2), "reference": {"properties": {"v": {"type": "long"}}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    for (id, e) in [("near", [1.0, 0.1]), ("far", [5.0, 5.0])] {
        let (st, v) = es.call(Method::PUT, &format!("/target/_doc/{id}?refresh=true"), Some(json!({"embedding": e}))).await;
        assert_eq!(st, 201, "{v}");
    }
    let explicit = es.call(Method::POST, "/target/_search", Some(json!({"knn": {"field": "embedding", "query_vector": [1.0, 0.0], "k": 2, "num_candidates": 10}, "_source": false}))).await.1;
    let ranked = |v: &Value| -> Vec<(String, f64)> { v["hits"]["hits"].as_array().unwrap_or(&vec![]).iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect() };
    for reference in ["ref-on", "ref-off"] {
        let (st, v) = es.call(Method::POST, "/target/_search", Some(json!({"knn": {"field": "embedding", "k": 2, "num_candidates": 10,
            "query_vector_builder": {"lookup": {"index": reference, "id": "q", "path": "reference.v"}}}, "_source": false}))).await;
        assert_eq!(st, 200, "{reference}: {v}");
        assert_eq!(ranked(&v), ranked(&explicit), "{reference}: same as the explicit query vector");
    }
}
