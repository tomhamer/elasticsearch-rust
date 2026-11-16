//! Finding 123: the shard-wide top-k scan of a `knn` query on a dense
//! `semantic_text` field polls the search's deadline and cancellation per
//! document (and within a document's chunks): a cancelled search stops
//! scoring at once and fails with `task_cancelled_exception`, a search past
//! its timeout stops scanning, and the bounded top-k heap ranks exactly as
//! the full sort did. A scan probe counts the documents scored.

use elasticsearch_engine::inference::chunk_query::set_scan_probe;
use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

const DOCS: usize = 300;

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

fn ids(v: &Value) -> Vec<String> {
    v["hits"]["hits"].as_array().into_iter().flatten().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn semantic_knn_scan_honours_cancellation_deadline_and_ranking() {
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
    let (st, v) = es.call(Method::PUT, "/sem", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"body": {"type": "semantic_text", "inference_id": "dense"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let mut bulk = String::new();
    for i in 0..DOCS {
        bulk.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("d{i}")}}), json!({"body": format!("document number {i} about topic {}", i % 17)})));
    }
    let r = es.c.post(format!("{}/sem/_bulk?refresh=true", es.base)).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");

    let scored = Arc::new(AtomicUsize::new(0));
    // 0: count; 1: cancel this search's task at the third document; 2: sleep.
    let mode = Arc::new(AtomicU8::new(0));
    {
        let (scored, mode) = (scored.clone(), mode.clone());
        set_scan_probe(Some(Arc::new(move || {
            let n = scored.fetch_add(1, Ordering::SeqCst) + 1;
            match mode.load(Ordering::SeqCst) {
                1 if n == 3 => {
                    if let Some(token) = rustic_core::cancel::current() {
                        token.cancel("probe");
                    }
                }
                2 => std::thread::sleep(std::time::Duration::from_millis(20)),
                _ => {}
            }
        })));
    }
    let knn = |k: usize| json!({"size": k, "query": {"knn": {"field": "body", "query_vector": [0.3, 0.1, 0.5, 0.2], "k": k}}});

    // Ranking: the top 5 are the first 5 of the full ranking.
    let (st, all) = es.call(Method::POST, "/sem/_search", Some(knn(DOCS))).await;
    assert_eq!(st, 200, "{all}");
    assert_eq!(ids(&all).len(), DOCS);
    let n = scored.swap(0, Ordering::SeqCst);
    assert!(n >= DOCS && n % DOCS == 0, "an uninterrupted scan scores every document: {n}");
    let (st, top) = es.call(Method::POST, "/sem/_search", Some(knn(5))).await;
    assert_eq!(st, 200, "{top}");
    assert_eq!(ids(&top), ids(&all)[..5].to_vec());
    let scores = |v: &Value| v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_score"].as_f64().unwrap()).collect::<Vec<_>>();
    assert_eq!(scores(&top), scores(&all)[..5].to_vec());
    scored.store(0, Ordering::SeqCst);

    // Cancellation after three scored documents: no fourth is scored.
    mode.store(1, Ordering::SeqCst);
    let (st, v) = es.call(Method::POST, "/sem/_search", Some(knn(5))).await;
    assert_eq!(scored.load(Ordering::SeqCst), 3, "documents scored after the cancellation: {v}");
    assert_ne!(st, 200, "{v}");
    assert!(v.to_string().contains("task_cancelled_exception"), "{st}: {v}");

    // A 1ms timeout with 20ms per document: the scan stops after the first.
    scored.store(0, Ordering::SeqCst);
    mode.store(2, Ordering::SeqCst);
    let (st, v) = es.call(Method::POST, "/sem/_search?timeout=1ms", Some(knn(5))).await;
    // (0 when the deadline passed before the scan began.)
    assert!(scored.load(Ordering::SeqCst) <= 1, "documents scored after the deadline ({}): {v}", scored.load(Ordering::SeqCst));
    assert!(st == 200 || st >= 400, "{v}");
    set_scan_probe(None);
}
