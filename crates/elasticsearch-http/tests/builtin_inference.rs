//! Built-in inference models over REST: without the opt-in they fail like an
//! undeployed model in ES; with `xpack.inference.builtin_fallback: lexical`
//! the stand-ins serve them, and responses and GETs say so; an endpoint
//! never switches implementation under existing data.

use elasticsearch_engine::inference::backend::{self, Implementation};
use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};
use std::path::PathBuf;

async fn serve(dir: &std::path::Path, fallback: bool) -> (String, PathBuf) {
    let node = Node::open(dir).unwrap();
    if fallback {
        node.node_settings.write().insert(backend::FALLBACK_SETTING.into(), json!("lexical"));
    }
    let state = node.data.join("_state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    (format!("http://127.0.0.1:{}", addr.port()), state)
}

async fn call(c: &Client, method: &str, url: String, body: Option<Value>) -> (u16, Value, Vec<String>) {
    let mut r = c.request(method.parse().unwrap(), url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let warnings: Vec<String> = resp.headers().get_all("warning").iter().map(|v| v.to_str().unwrap_or("").to_string()).collect();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)), warnings)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn default_config_fails_clearly() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _) = serve(dir.path(), false).await;
    let c = Client::new();
    let (st, v, w) = call(&c, "POST", format!("{base}/_inference/.elser-2-elasticsearch"), Some(json!({"input": "hello"}))).await;
    assert_eq!(st, 409, "{v}");
    assert_eq!(v["error"]["type"], "status_exception");
    assert_eq!(v["error"]["reason"], "Trained model deployment [.elser_model_2_linux-x86_64] is not allocated to any nodes");
    assert!(v["error"]["caused_by"]["reason"].as_str().unwrap().contains("xpack.inference.builtin_fallback"), "{v}");
    assert!(w.is_empty(), "{w:?}");
    let (st, v, _) = call(&c, "POST", format!("{base}/_inference/text_embedding/.multilingual-e5-small-elasticsearch"), Some(json!({"input": "hello"}))).await;
    assert_eq!(st, 409, "{v}");
    let (st, v, _) = call(&c, "POST", format!("{base}/_inference/rerank/.rerank-v1-elasticsearch"), Some(json!({"query": "q", "input": ["a", "b"]}))).await;
    assert_eq!(st, 409, "{v}");
    // GET renders as ES does (no implementation).
    let (_, v, _) = call(&c, "GET", format!("{base}/_inference/.elser-2-elasticsearch"), None).await;
    assert!(v["endpoints"][0].get("implementation").is_none(), "{v}");
    // A semantic_text document (default ELSER endpoint) is refused, not embedded by a stand-in.
    let (st, _, _) = call(&c, "PUT", format!("{base}/sem"), Some(json!({"mappings": {"properties": {"t": {"type": "semantic_text"}}}}))).await;
    assert_eq!(st, 200);
    let (st, v, _) = call(&c, "PUT", format!("{base}/sem/_doc/1"), Some(json!({"t": "hello world"}))).await;
    assert!(st >= 400, "{v}");
    assert!(v.to_string().contains("is not allocated to any nodes"), "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opt_in_reports_implementation_and_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let (base, state) = serve(dir.path(), true).await;
    let c = Client::new();
    let (st, v, w) = call(&c, "POST", format!("{base}/_inference/.elser-2-elasticsearch"), Some(json!({"input": "running dogs"}))).await;
    assert_eq!(st, 200, "{v}");
    assert!(v["sparse_embedding"][0]["embedding"].is_object(), "{v}");
    assert!(w.iter().any(|w| w.contains("[lexical-fallback] stand-in")), "{w:?}");
    let (_, v, _) = call(&c, "GET", format!("{base}/_inference/.elser-2-elasticsearch"), None).await;
    assert_eq!(v["endpoints"][0]["implementation"], "lexical-fallback", "{v}");

    // A user E5 endpoint embeds documents of a semantic_text index.
    let e5 = json!({"service": "elasticsearch", "service_settings": {"model_id": ".multilingual-e5-small", "num_allocations": 1, "num_threads": 1}});
    let (st, v, _) = call(&c, "PUT", format!("{base}/_inference/text_embedding/my-e5"), Some(e5.clone())).await;
    assert_eq!(st, 200, "{v}");
    let (st, v, _) = call(&c, "PUT", format!("{base}/sem"), Some(json!({"mappings": {"properties": {"t": {"type": "semantic_text", "inference_id": "my-e5"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v, _) = call(&c, "PUT", format!("{base}/sem/_doc/1?refresh=true"), Some(json!({"t": "hello world"}))).await;
    assert_eq!(st, 201, "{v}");
    let imp = backend::pin_of(&state, "my-e5").expect("implementation recorded");
    let (st, v, _) = call(&c, "POST", format!("{base}/sem/_search"), Some(json!({"query": {"semantic": {"field": "t", "query": "hello"}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");

    // Re-creating the endpoint while the index holds its embeddings keeps
    // the recorded implementation...
    let (st, _, _) = call(&c, "DELETE", format!("{base}/_inference/my-e5?force=true"), None).await;
    assert_eq!(st, 200);
    let (st, v, _) = call(&c, "PUT", format!("{base}/_inference/text_embedding/my-e5"), Some(e5.clone())).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(backend::pin_of(&state, "my-e5").as_deref(), Some(imp.as_str()));
    // ...and a call that would embed with another implementation is refused.
    let other = if imp == "candle" { Implementation::HashingFallback } else { Implementation::Candle };
    backend::clear_pin(&state, "my-e5");
    backend::check_pin(&state, "my-e5", other).unwrap();
    let (st, v, _) = call(&c, "POST", format!("{base}/_inference/my-e5"), Some(json!({"input": "x"}))).await;
    assert_eq!(st, 409, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("embeddings from different implementations are not comparable"), "{v}");
    let (st, v, _) = call(&c, "PUT", format!("{base}/sem/_doc/2"), Some(json!({"t": "more"}))).await;
    assert!(st >= 400, "{v}");

    // Once no index uses the id, a new endpoint starts afresh.
    let (st, _, _) = call(&c, "DELETE", format!("{base}/sem"), None).await;
    assert_eq!(st, 200);
    let (st, _, _) = call(&c, "DELETE", format!("{base}/_inference/my-e5"), None).await;
    assert_eq!(st, 200);
    let (st, v, _) = call(&c, "PUT", format!("{base}/_inference/text_embedding/my-e5"), Some(e5)).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(backend::pin_of(&state, "my-e5"), None);
    let (st, v, _) = call(&c, "POST", format!("{base}/_inference/my-e5"), Some(json!({"input": "x"}))).await;
    assert_eq!(st, 200, "{v}");
}
