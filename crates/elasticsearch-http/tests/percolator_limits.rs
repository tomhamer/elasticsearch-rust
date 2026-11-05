//! Public percolator limit/error contracts, including diagnostic query contexts.
use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

async fn call(client: &Client, base: &str, method: Method, path: &str, body: Value) -> (u16, Value) {
    let response = client.request(method, format!("{base}{path}")).json(&body).send().await.unwrap();
    (response.status().as_u16(), response.json().await.unwrap())
}

#[tokio::test]
async fn percolator_limits_keep_search_validate_explain_and_profile_fallible() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(elasticsearch_http::serve_listener(node, listener));
    let client = Client::new();
    let (status, value) = call(&client, &base, Method::PUT, "/p", json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.max_regex_length": 3},
        "mappings": {"properties": {"query": {"type": "percolator"}, "tag": {"type": "keyword"}}}
    })).await;
    assert_eq!(status, 200, "{value}");
    let (status, value) = call(&client, &base, Method::PUT, "/p/_doc/1?refresh=true", json!({"query": {"regexp": {"tag": "😀a"}}})).await;
    assert_eq!(status, 201, "{value}");
    let body = json!({"query": {"percolate": {"field": "query", "document": {"tag": "😀a"}}}});
    for path in ["/p/_search", "/p/_validate/query?explain=true", "/p/_explain/1"] {
        let (status, value) = call(&client, &base, Method::POST, path, body.clone()).await;
        assert_eq!(status, 200, "{path}: {value}");
        if path.contains("validate") { assert_eq!(value["valid"], true, "{value}"); }
        else if path.contains("explain") { assert_eq!(value["matched"], true, "{value}"); }
        else { assert_eq!(value["hits"]["total"]["value"], 1, "{value}"); }
    }
    let mut profile = body.clone();
    profile["profile"] = json!(true);
    let (status, value) = call(&client, &base, Method::POST, "/p/_search", profile.clone()).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(value["hits"]["total"]["value"], 1);
    assert!(value["profile"]["shards"].as_array().is_some_and(|shards| !shards.is_empty()), "{value}");
    let (status, value) = call(&client, &base, Method::PUT, "/p/_settings", json!({"index.max_regex_length": 2})).await;
    assert_eq!(status, 200, "{value}");
    for (path, request) in [("/p/_search", body.clone()), ("/p/_search", profile), ("/p/_explain/1", body)] {
        let (status, value) = call(&client, &base, Method::POST, path, request).await;
        assert!(status >= 400, "must not turn a lazy regex failure into a miss: {path}: {value}");
        assert!(value["error"].to_string().contains("index.max_regex_length"), "{value}");
    }
    server.abort();
}
