//! Public coordinator routes and every shard must enforce the same security policy.
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::time::Duration;

/// The node's base URL, set once it has bound its OS-assigned HTTP port.
static BASE_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
fn base() -> &'static str {
    BASE_URL.get().expect("node started")
}
async fn call(client: &Client, user: Option<&str>, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut request = client.request(method, format!("{}{path}", base()));
    if let Some(user) = user { request = request.basic_auth(user, Some("test-password")); }
    if let Some(body) = body { request = request.json(&body); }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn public_coordinator_enforces_security_across_shards() {
    let dir = evidence::retained_dir("security_frontend");
    let mut settings = Settings::new();
    for (key, value) in [
        ("cluster.name", json!("security-frontend")), ("node.name", json!("n0")),
        ("path.data", json!(dir.path().join("data").to_string_lossy())),
        ("path.logs", json!(dir.path().join("logs").to_string_lossy())),
        ("network.host", json!("127.0.0.1")), ("http.port", json!("0")),
        ("transport.port", json!("0")), ("discovery.type", json!("single-node")),
        ("xpack.security.enabled", json!("true")), ("bootstrap.password", json!("test-password")),
        ("xpack.license.self_generated.type", json!("trial")),
        ("xpack.security.authc.password_hashing.algorithm", json!("pbkdf2_1000")),
        ("cluster.routing.allocation.disk.watermark.low", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.high", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.flood_stage", json!("1b")),
    ] { settings.insert(key.into(), value); }
    let config = NodeConfig::from_settings(settings).unwrap();
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(config).await.unwrap();
    BASE_URL.set(format!("http://127.0.0.1:{}", ports::http_port(&node))).expect("one node per test binary");
    assert!(node.engine.security.enabled());
    assert_eq!(node.engine.security.cfg.read().bootstrap_password.as_deref(), Some("test-password"));
    assert!(node.engine.security.authenticate_password("elastic", "test-password").is_ok());
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(node, listener).await.unwrap(); });
    let client = Client::builder().timeout(Duration::from_secs(10)).build().unwrap();
    let mut ready = false;
    let mut last_health = String::new();
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        if let Ok(response) = client.get(format!("{}/_cluster/health", base())).basic_auth("elastic", Some("test-password")).send().await {
            if response.status().is_success() { ready = true; break; }
            last_health = format!("{} {}", response.status(), response.text().await.unwrap_or_default());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "secured node did not become ready: {last_health}");
    let (status, body) = call(&client, Some("elastic"), Method::PUT, "/_connector//_name", Some(json!({"name": "invalid"}))).await;
    assert_eq!(status, 400, "empty connector identifier must survive authorization: {body}");
    // Data-stream requirements must be checked before coordinator auto-creation.
    let (status, missing) = call(&client, Some("elastic"), Method::POST, "/require-stream-missing/_doc?require_data_stream=true", Some(json!({"f":1}))).await;
    assert_eq!(status, 404, "{missing}");
    assert!(missing["error"]["reason"].as_str().unwrap().contains("no matching index template with data stream template"));
    assert_eq!(call(&client, Some("elastic"), Method::HEAD, "/require-stream-missing", None).await.0, 404);
    let response = client.post(format!("{}/_bulk?require_data_stream=true", base())).basic_auth("elastic", Some("test-password"))
        .header("content-type", "application/x-ndjson")
        .body("{\"create\":{\"_index\":\"require-stream-bulk\"}}\n{\"f\":1}\n").send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let bulk: Value = response.json().await.unwrap();
    assert_eq!(bulk["items"][0]["create"]["status"], 404, "{bulk}");
    assert_eq!(call(&client, Some("elastic"), Method::HEAD, "/require-stream-bulk", None).await.0, 404);
    for path in ["/", "/_cluster/state", "/_cluster/health", "/_nodes", "/_cat/indices"] {
        assert_eq!(call(&client, None, Method::GET, path, None).await.0, 401, "{path}");
    }
    let (status, info) = call(&client, Some("elastic"), Method::GET, "/_nodes/os,settings?flat_settings=true", None).await;
    assert_eq!(status, 200, "{info}");
    let entry = info["nodes"].as_object().unwrap().values().next().unwrap();
    assert!(entry["os"]["available_processors"].as_u64().unwrap() > 0);
    assert!(entry["settings"].get("bootstrap.password").is_none(), "node settings leaked a password");
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/public", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}, "mappings": {"properties": {"visible": {"type": "keyword"}, "secret": {"type": "keyword"}}}}))).await.0, 200);
    for i in 0..6 {
        let (status, body) = call(&client, Some("elastic"), Method::PUT, &format!("/public/_doc/{i}?refresh=true"), Some(json!({"visible": if i % 2 == 0 { "yes" } else { "no" }, "secret": "hidden"}))).await;
        assert_eq!(status, 201, "{body}");
    }
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/_security/role/reader", Some(json!({"indices": [{"names": ["public"], "privileges": ["read", "view_index_metadata"], "query": {"term": {"visible": "yes"}}, "field_security": {"grant": ["visible"]}}]}))).await.0, 200);
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/_security/user/reader", Some(json!({"password": "test-password", "roles": ["reader"]}))).await.0, 200);
    let (status, cluster_metadata) = call(&client, Some("elastic"), Method::GET, "/_cluster/state/metadata", None).await;
    assert_eq!(status, 200, "{cluster_metadata}");
    assert!(cluster_metadata["metadata"].get("_security_state").is_none(), "native credential state leaked through public metadata");
    let security_doc = client.get(format!("{}/.security/_doc/user-reader", base())).basic_auth("elastic", Some("test-password")).send().await.unwrap();
    assert_eq!(security_doc.status().as_u16(), 200);
    assert!(security_doc.headers().get("warning").unwrap().to_str().unwrap().contains("system indices: [.security-7]"));
    assert_eq!(security_doc.json::<Value>().await.unwrap()["found"], true);
    assert_eq!(call(&client, Some("reader"), Method::GET, "/_cluster/state", None).await.0, 403);
    let (status, hits) = call(&client, Some("reader"), Method::POST, "/public/_search", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(status, 200, "{hits}");
    assert_eq!(hits["hits"]["total"]["value"], 3, "DLS across all three shards: {hits}");
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/private-other", Some(json!({}))).await.0, 200);
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/private-other/_doc/1?refresh=true", Some(json!({"visible": "yes"}))).await.0, 201);
    let (status, wildcard) = call(&client, Some("reader"), Method::GET, "/*/_count", None).await;
    assert_eq!(status, 200, "wildcard must retain the authorized target rewrite: {wildcard}");
    assert_eq!(wildcard["count"], 3, "private index must be excluded and DLS retained: {wildcard}");
    for (user, expected) in [("elastic", 6), ("reader", 3)] {
        let (status, result) = call(&client, Some(user), Method::POST, "/_query", Some(json!({"query":"FROM public | STATS docs = COUNT(*)"}))).await;
        assert_eq!(status, 200, "{result}");
        assert_eq!(result["values"][0][0], expected, "ES|QL must read all three shards and preserve DLS: {result}");
    }
    let (status, result) = call(&client, Some("reader"), Method::POST, "/_query", Some(json!({"query":"FROM public"}))).await;
    assert_eq!(status, 200, "{result}");
    assert_eq!(result["values"].as_array().unwrap().len(), 3);
    assert!(!result["columns"].as_array().unwrap().iter().any(|c| c["name"] == "secret"), "ES|QL source expansion leaked an FLS column: {result}");

    for hit in hits["hits"]["hits"].as_array().unwrap() {
        assert_eq!(hit["_source"]["visible"], "yes");
        assert!(hit["_source"].get("secret").is_none(), "FLS leaked a field: {hit}");
    }
    let (status, mapping) = call(&client, Some("reader"), Method::GET, "/public/_mapping", None).await;
    assert_eq!(status, 200, "{mapping}");
    assert!(mapping["public"]["mappings"]["properties"].get("secret").is_none(), "FLS leaked a mapping: {mapping}");
    assert_eq!(call(&client, Some("reader"), Method::HEAD, "/public/_mapping", None).await.0, 405);
    assert_eq!(call(&client, Some("reader"), Method::HEAD, "/public", None).await.0, 200);
    for (index, id, path, expected_status, expected_hits) in [
        ("public", "0", "visible", 200, 3),
        ("public", "0", "secret", 200, 0),
        ("public", "1", "visible", 200, 0),
        ("private-other", "1", "visible", 403, 0),
    ] {
        let (status, response) = call(&client, Some("reader"), Method::POST, "/public/_search", Some(json!({
            "query":{"terms":{"visible":{"index":index,"id":id,"path":path}}}
        }))).await;
        assert_eq!(status, expected_status, "coordinator lookup authorization: {response}");
        if status == 200 { assert_eq!(response["hits"]["total"]["value"], expected_hits, "lookup DLS/FLS: {response}"); }
    }
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/single-app-index", Some(json!({"settings":{"number_of_shards":1,"number_of_replicas":0}}))).await.0, 200);
    assert_eq!(call(&client, Some("elastic"), Method::PUT, "/single-app-index/_doc/one?refresh=true", Some(json!({"value":"one"}))).await.0, 201);
    let (status, application) = call(&client, Some("elastic"), Method::PUT, "/_application/search_application/single-app", Some(json!({
        "indices":["single-app-index"], "template":{"script":{"source":{"retriever":{"standard":{"query":{"match_all":{}}}}}}}
    }))).await;
    assert!(matches!(status, 200 | 201), "{application}");
    let (status, application_hits) = call(&client, Some("elastic"), Method::POST, "/_application/search_application/single-app/_search", Some(json!({}))).await;
    assert_eq!(status, 200, "single-shard application retrievers must keep their normal execution path: {application_hits}");
    assert_eq!(application_hits["hits"]["total"]["value"], 1);
    let (status, license) = call(&client, Some("elastic"), Method::POST, "/_license/start_basic?acknowledge=true", None).await;
    assert_eq!(status, 200, "{license}");
    // Licensing is hardwired to trial (AGENTS.md "Licensing: always
    // enterprise"): start_basic changes nothing, so rrf is licensed and the
    // request fails its own validation (one result set), not a licence check.
    let (status, ranked) = call(&client, Some("elastic"), Method::POST, "/public/_search", Some(json!({"query":{"match_all":{}}, "rank":{"rrf":{}}}))).await;
    assert_eq!(status, 400, "rrf stays licensed after start_basic: {ranked}");
    assert_eq!(ranked["error"]["type"], "action_request_validation_exception", "{ranked}");
    let (status, malformed) = call(&client, Some("elastic"), Method::POST, "/public/_search", Some(json!({"size":"invalid", "rank":{"rrf":{}}}))).await;
    assert_eq!(status, 400, "parse validation still applies: {malformed}");
    assert_eq!(call(&client, Some("reader"), Method::PUT, "/public/_doc/0", Some(json!({"visible": "yes"}))).await.0, 403);
    let (_, count) = call(&client, Some("elastic"), Method::GET, "/public/_count", None).await;
    assert_eq!(count["count"], 6);
    let (status, reset) = call(&client, Some("elastic"), Method::POST, "/_features/_reset", None).await;
    assert_eq!(status, 200, "{reset}");
    assert_eq!(call(&client, Some("reader"), Method::GET, "/_security/_authenticate", None).await.0, 401);
    assert_eq!(call(&client, Some("elastic"), Method::GET, "/_cluster/health", None).await.0, 200);
    server.abort();
}
