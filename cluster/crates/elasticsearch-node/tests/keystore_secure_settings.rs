//! Keystore secure settings on a running node: injected at startup the way
//! `cli::main` injects them, never rendered by an API, re-read by
//! `_nodes/reload_secure_settings` (and handed to the S3 repository
//! clients), and a missing keystore / wrong password reported per node.
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::keystore::KeyStore;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;

const ACCESS_KEY: &str = "AKIA-KEYSTORE-IT";
const SECRET_KEY: &str = "keystore-it-secret-value";

async fn call(client: &Client, base: &str, method: Method, path: &str, body: Option<Value>) -> (u16, String) {
    let mut request = client.request(method, format!("{base}{path}"));
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap_or_default())
}

fn json_of(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or(Value::Null)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keystore_settings_are_injected_hidden_and_reloadable() {
    let dir = tempfile::tempdir().unwrap();
    let conf = dir.path().join("config");
    std::fs::create_dir_all(&conf).unwrap();
    let mut ks = KeyStore::create().unwrap();
    ks.set_string("s3.client.default.access_key", ACCESS_KEY).unwrap();
    ks.set_string("s3.client.default.secret_key", SECRET_KEY).unwrap();
    ks.save(&conf, "").unwrap();
    let seed = ks.get_string("keystore.seed").unwrap();

    let mut settings = Settings::new();
    for (key, value) in [
        ("cluster.name", json!("keystore-it")), ("node.name", json!("n0")),
        ("path.conf", json!(conf.to_string_lossy())),
        ("path.data", json!(dir.path().join("data").to_string_lossy())),
        ("path.logs", json!(dir.path().join("logs").to_string_lossy())),
        ("network.host", json!("127.0.0.1")), ("http.port", json!("0")),
        ("transport.port", json!("0")), ("discovery.type", json!("single-node")),
        ("xpack.security.enabled", json!("false")),
        ("s3.client.default.endpoint", json!("127.0.0.1:1")),
        ("cluster.routing.allocation.disk.watermark.low", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.high", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.flood_stage", json!("1b")),
    ] { settings.insert(key.into(), value); }
    // What `cli::main` does after `config::load`.
    elasticsearch_node::keystore::load_into_settings(&mut settings, &BTreeMap::new()).unwrap();
    assert_eq!(settings["s3.client.default.access_key"], ACCESS_KEY);

    let config = NodeConfig::from_settings(settings).unwrap();
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(config).await.unwrap();
    let base = format!("http://127.0.0.1:{}", ports::http_port(&node));
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let serving = node.clone();
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(serving, listener).await.unwrap(); });
    let client = Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
    let mut ready = false;
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        if call(&client, &base, Method::GET, "/_cluster/health", None).await.0 == 200 {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "node did not become ready");

    // The node uses the keystore's credentials ...
    let ctx = elasticsearch_node::snapshot_api::common::repo_context(&node);
    assert_eq!(ctx.s3_clients["default"].access_key.as_deref(), Some(ACCESS_KEY));
    // ... and no API renders them.
    for path in [
        "/_nodes", "/_nodes/settings", "/_nodes/settings?flat_settings=true", "/_nodes/_local/settings",
        "/_nodes/stats", "/_cluster/settings?include_defaults=true", "/_cluster/settings?include_defaults=true&flat_settings=true",
        "/_cluster/state", "/_cluster/stats",
    ] {
        let (status, text) = call(&client, &base, Method::GET, path, None).await;
        assert_eq!(status, 200, "{path}: {text}");
        for secret in [ACCESS_KEY, SECRET_KEY, seed.as_str()] {
            assert!(!text.contains(secret), "{path} leaked a keystore value: {text}");
        }
    }
    let (_, text) = call(&client, &base, Method::GET, "/_nodes/settings?flat_settings=true", None).await;
    let info = json_of(&text);
    let entry = info["nodes"].as_object().unwrap().values().next().unwrap();
    assert_eq!(entry["settings"]["s3.client.default.endpoint"], "127.0.0.1:1", "plain settings are still rendered: {entry}");
    assert!(entry["settings"].get("s3.client.default.access_key").is_none());

    // Reload: no password, every node.
    let (status, text) = call(&client, &base, Method::POST, "/_nodes/reload_secure_settings", None).await;
    assert_eq!(status, 200, "{text}");
    let body = json_of(&text);
    assert_eq!(body["_nodes"], json!({"total": 1, "successful": 1, "failed": 0}));
    assert_eq!(body["cluster_name"], "keystore-it");
    let entry = &body["nodes"][&node.local.id];
    assert_eq!(entry["name"], "n0");
    assert!(entry.get("reload_exception").is_none(), "{entry}");
    assert_eq!(entry["secure_setting_names"], json!(["keystore.seed", "s3.client.default.access_key", "s3.client.default.secret_key"]));
    assert_eq!(entry["keystore_path"], conf.join("elasticsearch.keystore").display().to_string());
    assert_eq!(entry["keystore_digest"].as_str().unwrap().len(), 64);
    assert!(entry["keystore_last_modified_time"].as_str().unwrap().ends_with('Z'));

    // Rotate the credentials in the keystore, reload: the S3 clients follow.
    let mut ks = KeyStore::load(&conf).unwrap().unwrap();
    ks.decrypt("").unwrap();
    ks.set_string("s3.client.default.access_key", "AKIA-ROTATED").unwrap();
    ks.save(&conf, "").unwrap();
    let (status, text) = call(&client, &base, Method::POST, "/_nodes/_local/reload_secure_settings", Some(json!({}))).await;
    assert_eq!(status, 200, "{text}");
    assert!(json_of(&text)["nodes"][&node.local.id].get("reload_exception").is_none(), "{text}");
    let ctx = elasticsearch_node::snapshot_api::common::repo_context(&node);
    assert_eq!(ctx.s3_clients["default"].access_key.as_deref(), Some("AKIA-ROTATED"));
    assert_eq!(ctx.s3_clients["default"].secret_key.as_deref(), Some(SECRET_KEY));

    // Wrong password (ES's `nodes.reload_secure_settings/10_basic.yml`).
    let (status, text) = call(&client, &base, Method::POST, "/_nodes/_local/reload_secure_settings", Some(json!({"secure_settings_password": "awrongpasswordhere"}))).await;
    assert_eq!(status, 200, "{text}");
    let e = &json_of(&text)["nodes"][&node.local.id]["reload_exception"];
    assert_eq!(e["type"], "security_exception", "{text}");
    assert_eq!(e["reason"], "Provided keystore password was incorrect");
    // The credentials stay as they were.
    assert_eq!(elasticsearch_node::snapshot_api::common::repo_context(&node).s3_clients["default"].access_key.as_deref(), Some("AKIA-ROTATED"));

    // Repository-level credentials are refused (ES 9 without
    // `es.allow_insecure_settings`).
    let (status, text) = call(&client, &base, Method::PUT, "/_snapshot/insecure?verify=false",
        Some(json!({"type": "s3", "settings": {"bucket": "b", "access_key": "a", "secret_key": "s"}}))).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("Setting [access_key] is insecure, use the elasticsearch keystore instead"), "{text}");

    // No keystore at all.
    std::fs::remove_file(conf.join("elasticsearch.keystore")).unwrap();
    let (status, text) = call(&client, &base, Method::POST, "/_nodes/reload_secure_settings", None).await;
    assert_eq!(status, 200, "{text}");
    let e = &json_of(&text)["nodes"][&node.local.id]["reload_exception"];
    assert_eq!(*e, json!({"type": "illegal_state_exception", "reason": "Keystore is missing"}));
}
