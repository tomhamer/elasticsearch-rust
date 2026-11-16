//! Trained models live in the `.ml-inference-*` system indices (ES
//! `TrainedModelProvider`), not in node-local files, and models left in the
//! legacy `<path.data>/ml-models` files are migrated into them.

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};

async fn serve(dir: &std::path::Path) -> (String, std::path::PathBuf) {
    let node = Node::open(dir).unwrap();
    let data = node.data.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    (format!("http://127.0.0.1:{}", addr.port()), data)
}

async fn call(c: &Client, method: &str, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method.parse().unwrap(), url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn models_are_documents_of_the_model_indices() {
    let dir = tempfile::tempdir().unwrap();
    let (base, data) = serve(dir.path()).await;
    let c = Client::new();
    let (st, v) = call(&c, "PUT", format!("{base}/_ml/trained_models/tiny"), Some(json!({
        "model_type": "pytorch", "input": {"field_names": ["text_field"]},
        "inference_config": {"text_embedding": {}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, "PUT", format!("{base}/_ml/trained_models/tiny/vocabulary"), Some(json!({"vocabulary": ["a", "b"]}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, "PUT", format!("{base}/_ml/trained_models/tiny/definition/0"), Some(json!({"definition": "AAAA", "total_definition_length": 3, "total_parts": 1}))).await;
    assert_eq!(st, 200, "{v}");
    // Nothing in the node-local files; everything in the indices.
    assert!(!data.join("ml-models").join("models.json").exists());
    assert!(!data.join("ml-models").join("tiny").exists());
    let (_, v) = call(&c, "POST", format!("{base}/.ml-inference-000005/_search?expand_wildcards=all"), Some(json!({"query": {"match_all": {}}}))).await;
    let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().filter_map(|h| h["_id"].as_str()).collect();
    assert!(ids.contains(&"trained_model_config-tiny") && ids.contains(&"trained_model_vocabulary-tiny"), "{v}");
    let (_, v) = call(&c, "GET", format!("{base}/.ml-inference-native-000002/_doc/trained_model_definition_doc-tiny-0"), None).await;
    assert_eq!(v["_source"]["body"]["definition"], "AAAA", "{v}");
    let (_, v) = call(&c, "GET", format!("{base}/_ml/trained_models/tiny"), None).await;
    assert_eq!(v["count"], 1, "{v}");
    // Deleting the model deletes its documents.
    let (st, v) = call(&c, "DELETE", format!("{base}/_ml/trained_models/tiny"), None).await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = call(&c, "POST", format!("{base}/.ml-inference-*/_search?expand_wildcards=all"), Some(json!({"query": {"term": {"model_id": "tiny"}}}))).await;
    assert_eq!(v["hits"]["total"]["value"], 0, "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn legacy_model_files_are_migrated() {
    let dir = tempfile::tempdir().unwrap();
    // A model stored by an older version (node-local files).
    let root = dir.path().join("ml-models");
    std::fs::create_dir_all(root.join("old")).unwrap();
    std::fs::write(root.join("models.json"), json!({"old": {"model_id": "old", "model_type": "pytorch", "create_time": 1,
        "input": {"field_names": ["text_field"]}, "inference_config": {"text_embedding": {}}, "tags": []}}).to_string()).unwrap();
    std::fs::write(root.join("old").join("vocabulary.json"), json!({"vocabulary": ["x"]}).to_string()).unwrap();
    std::fs::write(root.join("old").join("definition-0.json"), json!({"definition": "AAAA", "total_definition_length": 3, "total_parts": 1}).to_string()).unwrap();
    let (base, _) = serve(dir.path()).await;
    let c = Client::new();
    let (st, v) = call(&c, "GET", format!("{base}/_ml/trained_models/old"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["trained_model_configs"][0]["model_id"], "old", "{v}");
    let (_, v) = call(&c, "GET", format!("{base}/.ml-inference-native-000002/_doc/trained_model_definition_doc-old-0"), None).await;
    assert_eq!(v["found"], true, "{v}");
    let (_, v) = call(&c, "GET", format!("{base}/.ml-inference-000005/_doc/trained_model_vocabulary-old"), None).await;
    assert_eq!(v["_source"]["body"]["vocabulary"][0], "x", "{v}");
    assert!(root.join(".migrated-to-ml-inference-index").exists());
}
