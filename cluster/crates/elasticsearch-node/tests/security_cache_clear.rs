//! The security `_clear_cache` APIs are nodes actions (ES): a request on
//! either node invalidates the caches of every node of the cluster.
mod common;

use common::{ports, spawn_in_process, Ports};
use elasticsearch_node::config::NodeConfig;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::Duration;

const PASSWORD: &str = "cache-clear-password";

async fn call(c: &reqwest::Client, p: Ports, auth: (&str, &str), node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let creds = (auth.0.to_string(), auth.1.to_string());
    common::call_as(c, p, Some(&creds), node, method, path, body).await
}

async fn call_key(c: &reqwest::Client, p: Ports, key: &str, node: usize, path: &str) -> u16 {
    let r = c.get(format!("http://127.0.0.1:{}{path}", p.http[node])).header("authorization", format!("ApiKey {key}")).send().await;
    r.map(|r| r.status().as_u16()).unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn security_clear_cache_reaches_every_node() {
    let cluster = ports::unique("security-cache-clear");
    let dir = common::data_dir(&cluster);
    let extra = [
        ("xpack.security.enabled", "true"),
        ("bootstrap.password", PASSWORD),
        ("xpack.license.self_generated.type", "trial"),
        ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000"),
    ];
    let mut p = Ports { http: [0; 2], transport: [0; 2] };
    let mut nodes = Vec::new();
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(common::settings(&cluster, p, i, &dir, &extra)).expect("config");
        let node = spawn_in_process(cfg).await;
        p.http[i] = ports::http_port(&node);
        p.transport[i] = ports::transport_port(&node);
        nodes.push(node);
    }
    let c = reqwest::Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
    let admin = ("elastic", PASSWORD);
    let admin_creds = (admin.0.to_string(), admin.1.to_string());
    ports::wait_formed(&c, &p.http, Some(&admin_creds), 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));

    let (st, r) = call(&c, p, admin, 0, Method::PUT, "/_security/role/r", Some(json!({"cluster": ["monitor"]}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = call(&c, p, admin, 0, Method::PUT, "/_security/user/u", Some(json!({"password": PASSWORD, "roles": ["r"]}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = call(&c, p, admin, 0, Method::POST, "/_security/api_key", Some(json!({"name": "k"}))).await;
    assert_eq!(st, 200, "{r}");
    let (id, key) = (r["id"].as_str().unwrap().to_string(), r["encoded"].as_str().unwrap().to_string());

    let check = |r: &Value| {
        assert_eq!(r["_nodes"]["total"], 2, "{r}");
        assert_eq!(r["_nodes"]["successful"], 2, "{r}");
        assert_eq!(r["nodes"].as_object().map(|n| n.len()), Some(2), "{r}");
    };
    let key_tag = format!("apikey\u{0}{id}");
    let key_limit = format!("apikey-limit:{id}");
    let cached = |node: usize, roles: bool, entry: &str| {
        let (r, v) = nodes[node].engine.security.cache_keys();
        (if roles { r } else { v }).iter().any(|k| k == entry)
    };
    // Fill both nodes' caches (waiting for the security state to reach each
    // node and for the entries to be cached there).
    let user = ("u", PASSWORD);
    let fill = || async {
        for node in 0..2 {
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            loop {
                let user_ok = call(&c, p, user, node, Method::GET, "/_cluster/health", None).await.0 == 200;
                let key_ok = call_key(&c, p, &key, node, "/_cluster/health").await == 200;
                if user_ok && key_ok && cached(node, true, "roles:r") && cached(node, false, "basic\u{0}u") && cached(node, false, &key_tag) && cached(node, true, &key_limit) {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "node {node} never cached u and the key: {:?}", nodes[node].engine.security.cache_keys());
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    };

    fill().await;
    // Each API from each node: every node's cache is cleared.
    for from in 0..2 {
        let (st, r) = call(&c, p, admin, from, Method::POST, "/_security/role/r/_clear_cache", None).await;
        assert_eq!(st, 200, "{r}");
        check(&r);
        let (st, r) = call(&c, p, admin, from, Method::POST, "/_security/realm/default_native/_clear_cache?usernames=u", None).await;
        assert_eq!(st, 200, "{r}");
        check(&r);
        let (st, r) = call(&c, p, admin, from, Method::POST, &format!("/_security/api_key/{id}/_clear_cache"), None).await;
        assert_eq!(st, 200, "{r}");
        check(&r);
        for node in 0..2 {
            let keys = nodes[node].engine.security.cache_keys();
            assert!(!cached(node, true, "roles:r"), "from {from}, node {node}: role still cached {keys:?}");
            assert!(!cached(node, false, "basic\u{0}u"), "from {from}, node {node}: user still cached {keys:?}");
            assert!(!cached(node, false, &key_tag) && !cached(node, true, &key_limit), "from {from}, node {node}: key still cached {keys:?}");
        }
        fill().await;
        let (st, r) = call(&c, p, admin, from, Method::POST, "/_security/privilege/app/_clear_cache", None).await;
        assert_eq!(st, 200, "{r}");
        check(&r);
        for node in 0..2 {
            assert!(!cached(node, true, "roles:r"), "from {from}, node {node}: privilege clear-cache left roles cached");
        }
        fill().await;
    }
}
