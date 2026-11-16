//! Security contexts through the cluster front: async searches kept by the
//! coordinating node belong to their submitter, and watch executions run
//! as the watch's own subject (with its DLS / FLS) whichever principal made
//! the latest watcher request.
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::time::Duration;

static BASE_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
fn base() -> &'static str {
    BASE_URL.get().expect("node started")
}
async fn call(client: &Client, user: &str, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut request = client.request(method, format!("{}{path}", base())).basic_auth(user, Some("test-password"));
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap_or(Value::Null))
}

/// Names of a watch record's search input hits, and whether any shows `salary`.
fn input_hits(rec: &Value) -> (Vec<String>, bool) {
    let hits = rec["result"]["input"]["payload"]["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let mut names: Vec<String> = hits.iter().map(|h| h["_source"]["name"].as_str().unwrap_or("?").to_string()).collect();
    names.sort();
    (names, hits.iter().any(|h| h["_source"].get("salary").is_some()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_results_and_watches_keep_their_security_context() {
    let dir = evidence::retained_dir("security_contexts");
    let mut settings = Settings::new();
    for (key, value) in [
        ("cluster.name", json!("security-contexts")), ("node.name", json!("n0")),
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
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(node, listener).await.unwrap(); });
    let client = Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
    let mut ready = false;
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        if let Ok(response) = client.get(format!("{}/_cluster/health", base())).basic_auth("elastic", Some("test-password")).send().await {
            if response.status().is_success() { ready = true; break; }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "secured node did not become ready");

    let (st, v) = call(&client, "elastic", Method::PUT, "/docs", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"name": {"type": "keyword"}, "dept": {"type": "keyword"}, "salary": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    for (id, name, dept, salary) in [("1", "A", "eng", 100), ("2", "B", "eng", 200), ("3", "C", "sales", 300)] {
        let (st, v) = call(&client, "elastic", Method::PUT, &format!("/docs/_doc/{id}?refresh=true"), Some(json!({"name": name, "dept": dept, "salary": salary}))).await;
        assert_eq!(st, 201, "{v}");
    }
    let read = json!([{"names": ["docs"], "privileges": ["read", "view_index_metadata"]}]);
    for (name, body) in [
        ("reader", json!({"indices": read})),
        ("monitor_reader", json!({"cluster": ["monitor"], "indices": read})),
        ("canceller", json!({"cluster": ["cancel_task"], "indices": read})),
        ("watch_restricted", json!({"cluster": ["manage_watcher"], "indices": [{"names": ["docs"], "privileges": ["read"],
            "query": {"term": {"dept": "eng"}}, "field_security": {"grant": ["*"], "except": ["salary"]}}]})),
    ] {
        let (st, v) = call(&client, "elastic", Method::PUT, &format!("/_security/role/{name}"), Some(body)).await;
        assert_eq!(st, 200, "role {name}: {v}");
    }
    for (user, role) in [("alice", "reader"), ("bob", "reader"), ("carol", "monitor_reader"), ("dave", "canceller"), ("wat", "watch_restricted")] {
        let (st, v) = call(&client, "elastic", Method::PUT, &format!("/_security/user/{user}"), Some(json!({"password": "test-password", "roles": [role]}))).await;
        assert_eq!(st, 200, "user {user}: {v}");
    }

    // ---- async search kept by the coordinating node
    let (st, v) = call(&client, "alice", Method::POST, "/docs/_async_search?keep_on_completion=true&wait_for_completion_timeout=10s", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(st, 200, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    let (st, v) = call(&client, "alice", Method::GET, &format!("/_async_search/{id}"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["response"]["hits"]["total"]["value"], 3, "{v}");
    for (user, method, path) in [
        ("bob", Method::GET, format!("/_async_search/{id}")),
        ("bob", Method::GET, format!("/_async_search/{id}?keep_alive=10d")),
        ("bob", Method::GET, format!("/_async_search/status/{id}")),
        ("bob", Method::DELETE, format!("/_async_search/{id}")),
        ("carol", Method::GET, format!("/_async_search/{id}")),
        ("elastic", Method::GET, format!("/_async_search/{id}")),
    ] {
        let (st, v) = call(&client, user, method.clone(), &path, None).await;
        assert_eq!(st, 404, "{user} {method} {path}: {v}");
    }
    let (st, v) = call(&client, "alice", Method::GET, &format!("/_async_search/status/{id}"), None).await;
    assert_eq!(st, 200, "the submitter reads its status: {v}");
    let (st, v) = call(&client, "carol", Method::GET, &format!("/_async_search/status/{id}"), None).await;
    assert_eq!(st, 200, "monitor reads any status: {v}");
    let (st, v) = call(&client, "dave", Method::DELETE, &format!("/_async_search/{id}"), None).await;
    assert_eq!(st, 200, "cancel_task deletes any search: {v}");
    let (st, v) = call(&client, "alice", Method::GET, &format!("/_async_search/{id}"), None).await;
    assert_eq!(st, 404, "{v}");

    // ---- watches: `_execute` runs as the caller, scheduled runs as the owner
    let watch = json!({
        "trigger": {"schedule": {"interval": "1h"}},
        "input": {"search": {"request": {"indices": ["docs"], "body": {"query": {"match_all": {}}, "size": 10}}}},
        "condition": {"always": {}},
        "actions": {"log": {"logging": {"text": "hits {{ctx.payload.hits.total}}"}}}
    });
    let (st, v) = call(&client, "wat", Method::PUT, "/_watcher/watch/w_restricted", Some(watch.clone())).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = call(&client, "elastic", Method::PUT, "/_watcher/watch/w_admin", Some(watch.clone())).await;
    assert_eq!(st, 201, "{v}");
    // The two principals alternate; `_execute` runs as its caller (ES
    // `TransportExecuteWatchAction` swaps in the request's security headers),
    // with the caller's DLS / FLS, whoever stored the watch.
    for round in 0..3 {
        for (caller, watch_id, owner, names, salary) in [
            ("elastic", "w_restricted", "elastic", vec!["A", "B", "C"], true),
            ("wat", "w_admin", "wat", vec!["A", "B"], false),
            ("wat", "w_restricted", "wat", vec!["A", "B"], false),
            ("elastic", "w_admin", "elastic", vec!["A", "B", "C"], true),
        ] {
            let (st, v) = call(&client, caller, Method::POST, &format!("/_watcher/watch/{watch_id}/_execute"), Some(json!({}))).await;
            assert_eq!(st, 200, "round {round}: {caller} executes {watch_id}: {v}");
            let rec = &v["watch_record"];
            assert_eq!(rec["user"], owner, "round {round}: {caller} executes {watch_id}: {v}");
            assert_eq!(rec["result"]["input"]["status"], "success", "{v}");
            let (got, shows_salary) = input_hits(rec);
            assert_eq!(got, names, "round {round}: {caller} executes {watch_id}: {v}");
            assert_eq!(shows_salary, salary, "round {round}: {caller} executes {watch_id}: {v}");
        }
    }
    // An inline watch runs as its caller, right after the other principal's request.
    let (st, v) = call(&client, "elastic", Method::GET, "/_watcher/watch/w_admin", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&client, "wat", Method::POST, "/_watcher/watch/_execute", Some(json!({"watch": watch}))).await;
    assert_eq!(st, 200, "{v}");
    let (got, shows_salary) = input_hits(&v["watch_record"]);
    assert_eq!(got, ["A", "B"], "{v}");
    assert!(!shows_salary, "{v}");
    server.abort();
}
