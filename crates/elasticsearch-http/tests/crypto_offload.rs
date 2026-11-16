//! Production review #17, remaining paths: requests that reach the engine's
//! security layer without the cluster's client edge (forwarded from another
//! node, in-process dispatch, the engine's own listener) and body password
//! grants (OAuth2 token, API key grant, profile activation) verify bcrypt
//! on the crypto pool behind the failed-authentication limiter, never on a
//! runtime worker.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use elasticsearch_http::security::edge;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const ADMIN_PW: &str = "admin-password";

fn basic(user: &str, pass: &str) -> String {
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{pass}")))
}

async fn call(client: &Client, base: &str, auth: Option<String>, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = client.request(method, format!("{base}{path}"));
    if let Some(a) = auth { r = r.header("authorization", a); }
    if let Some(b) = body { r = r.json(&b); }
    let r = r.send().await.unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forwarded_requests_and_password_grants_verify_on_the_crypto_pool() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    cfg.token_enabled = true;
    cfg.api_key_enabled = true;
    cfg.trial = true;
    // `admin` is cheap (no pool), `hashed` is bcrypt.
    cfg.file_users.insert("admin".into(), Hasher::Ssha256.hash(ADMIN_PW));
    cfg.file_user_roles.insert("admin".into(), vec!["superuser".into()]);
    cfg.file_users.insert("hashed".into(), Hasher::Bcrypt(10).hash("right"));
    cfg.file_user_roles.insert("hashed".into(), vec!["superuser".into()]);
    node.security.configure(cfg);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let n = node.clone();
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(n, listener).await; });
    let client = Client::builder().timeout(Duration::from_secs(60)).pool_max_idle_per_host(0).build().unwrap();
    edge::set_failure_limits(u32::MAX, u32::MAX, Duration::from_secs(60));
    edge::set_crypto_pool(2, 1000);
    let pool = edge::crypto_pool();
    let admin = || Some(basic("admin", ADMIN_PW));
    for _ in 0..50 {
        if call(&client, &base, admin(), Method::GET, "/_cluster/health", None).await.0 == 200 { break; }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Header credentials through the security layer (no client edge): 100
    // concurrent failed bcrypt logins run on the pool; the two runtime
    // workers keep serving.
    let before = pool.completed();
    let attack: Vec<_> = (0..100).map(|i| {
        let (client, base) = (client.clone(), base.clone());
        tokio::spawn(async move { call(&client, &base, Some(basic("hashed", &format!("wrong-{i}"))), Method::GET, "/_cluster/health", None).await.0 })
    }).collect();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut latencies = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        assert_eq!(call(&client, &base, admin(), Method::GET, "/_cluster/health", None).await.0, 200);
        latencies.push(t.elapsed());
    }
    let mut statuses = Vec::new();
    for t in attack { statuses.push(t.await.unwrap()); }
    latencies.sort();
    let p99 = latencies[latencies.len() * 99 / 100 - 1];
    eprintln!("[crypto_offload] 100 failed bcrypt logins via the security layer: {} x 401; pool verified {}; authenticated p50 {:?} p99 {p99:?}",
        statuses.iter().filter(|s| **s == 401).count(), pool.completed() - before, latencies[latencies.len() / 2]);
    assert!(statuses.iter().all(|s| *s == 401), "{statuses:?}");
    assert!(pool.completed() - before >= 100, "failed verifications did not run on the pool");
    assert!(p99 < Duration::from_millis(250), "authenticated requests starved: {p99:?}");
    // The right password: verified on the pool once, then a cache hit.
    let before = pool.completed();
    assert_eq!(call(&client, &base, Some(basic("hashed", "right")), Method::GET, "/_security/_authenticate", None).await.0, 200);
    assert_eq!(call(&client, &base, Some(basic("hashed", "right")), Method::GET, "/_security/_authenticate", None).await.0, 200);
    assert_eq!(pool.completed() - before, 1);

    // Body password grants: failures verified on the pool (once each),
    // successes from the pool then the cache.
    let grants: [(&str, Value); 3] = [
        ("/_security/oauth2/token", json!({"grant_type": "password", "username": "hashed", "password": "nope"})),
        ("/_security/api_key/grant", json!({"grant_type": "password", "username": "hashed", "password": "nope", "api_key": {"name": "k"}})),
        ("/_security/profile/_activate", json!({"grant_type": "password", "username": "hashed", "password": "nope"})),
    ];
    for (path, body) in &grants {
        let before = pool.completed();
        let (status, reply) = call(&client, &base, admin(), Method::POST, path, Some(body.clone())).await;
        assert_eq!(status, 401, "{path}: {reply}");
        assert_eq!(pool.completed() - before, 1, "{path}: failed grant must be verified once, on the pool");
    }
    let (status, reply) = call(&client, &base, admin(), Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "password", "username": "hashed", "password": "right"}))).await;
    assert_eq!(status, 200, "{reply}");
    assert!(reply["access_token"].is_string());

    // The limiter guards both paths. (Issuing the token changed security
    // state, which invalidates cached successes: re-establish one first.)
    assert_eq!(call(&client, &base, Some(basic("hashed", "right")), Method::GET, "/_security/_authenticate", None).await.0, 200);
    edge::set_failure_limits(3, u32::MAX, Duration::from_secs(60));
    let mut seen = Vec::new();
    for i in 0..5 {
        seen.push(call(&client, &base, admin(), Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "password", "username": "hashed", "password": format!("bad-{i}")}))).await.0);
    }
    seen.push(call(&client, &base, Some(basic("hashed", "bad-header")), Method::GET, "/_cluster/health", None).await.0);
    eprintln!("[crypto_offload] limiter 3/min, grants then header: {seen:?}");
    assert_eq!(seen, vec![401, 401, 401, 429, 429, 429]);
    // A cached success is not limited.
    assert_eq!(call(&client, &base, Some(basic("hashed", "right")), Method::GET, "/_security/_authenticate", None).await.0, 200);
    edge::set_failure_limits(u32::MAX, u32::MAX, Duration::from_secs(60));
}
