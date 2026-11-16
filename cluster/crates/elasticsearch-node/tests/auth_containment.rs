//! Production review #17 / #18 at the real HTTP edge of a secured node:
//! failed logins to a bcrypt account are verified on the bounded crypto
//! pool (never on the networking runtime), with admission and a failed-
//! authentication limiter, while authenticated requests keep their latency;
//! request bodies held open mid-stream are refused before being read
//! (invalid credentials) or reclaimed by the body deadlines, and
//! control-plane requests are served throughout.
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use reqwest::Client;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn percentile(v: &mut [Duration], p: f64) -> Duration {
    v.sort();
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

async fn timed_authenticated(client: &Client, base: &str, n: usize) -> Vec<Duration> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let r = client.get(format!("{base}/_cluster/health")).basic_auth("elastic", Some("test-password")).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
        let _ = r.bytes().await;
        out.push(t.elapsed());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    out
}

async fn failed_logins(_client: &Client, base: &str, n: usize, user: &str) -> (usize, usize, Duration) {
    // Its own client without idle pooling: the attack's connections close
    // when it ends instead of holding file descriptors for later phases.
    let client = &Client::builder().timeout(Duration::from_secs(60)).pool_max_idle_per_host(0).build().unwrap();
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(n);
    for i in 0..n {
        let client = client.clone();
        let url = format!("{base}/_cluster/health");
        let user = user.to_string();
        tasks.push(tokio::spawn(async move {
            client.get(url).basic_auth(user, Some(format!("wrong-{i}"))).send().await.map(|r| r.status().as_u16()).unwrap_or(0)
        }));
    }
    let (mut unauthorized, mut rejected) = (0, 0);
    for t in tasks {
        match t.await.unwrap() {
            401 => unauthorized += 1,
            429 => rejected += 1,
            other => panic!("failed login answered {other}"),
        }
    }
    (unauthorized, rejected, started.elapsed())
}

/// Send headers and the first `sent` bytes of a `length`-byte body, then hold.
async fn partial_request(addr: &str, auth: Option<&str>, path: &str, length: usize, sent: usize) -> tokio::net::TcpStream {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let auth = auth.map(|a| format!("Authorization: {a}\r\n")).unwrap_or_default();
    let head = format!("POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {length}\r\n{auth}\r\n");
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(&vec![b' '; sent]).await.unwrap();
    s
}

/// The status of the response on `s`, or `None` when none arrives within `wait`.
async fn response_status(s: &mut tokio::net::TcpStream, wait: Duration) -> Option<u16> {
    let mut buf = vec![0u8; 4096];
    let n = tokio::time::timeout(wait, s.read(&mut buf)).await.ok()?.ok()?;
    let text = String::from_utf8_lossy(&buf[..n]);
    text.split_whitespace().nth(1)?.parse().ok()
}

fn basic(user: &str, pass: &str) -> String {
    use base64::Engine;
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{pass}")))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_logins_and_stalled_bodies_do_not_starve_the_edge() {
    let dir = evidence::retained_dir("auth_containment");
    let mut settings = Settings::new();
    for (key, value) in [
        ("cluster.name", json!("auth-containment")), ("node.name", json!("n0")),
        ("path.data", json!(dir.path().join("data").to_string_lossy())),
        ("path.logs", json!(dir.path().join("logs").to_string_lossy())),
        ("network.host", json!("127.0.0.1")), ("http.port", json!("0")),
        ("transport.port", json!("0")), ("discovery.type", json!("single-node")),
        ("xpack.security.enabled", json!("true")), ("bootstrap.password", json!("test-password")),
        ("xpack.license.self_generated.type", json!("trial")),
        ("xpack.security.authc.password_hashing.algorithm", json!("bcrypt")),
        ("cluster.routing.allocation.disk.watermark.low", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.high", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.flood_stage", json!("1b")),
    ] { settings.insert(key.into(), value); }
    let config = NodeConfig::from_settings(settings).unwrap();
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(config).await.unwrap();
    let addr = format!("127.0.0.1:{}", ports::http_port(&node));
    let base = format!("http://{addr}");
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let engine = node.engine.clone();
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(node, listener).await.unwrap(); });
    let client = Client::builder().timeout(Duration::from_secs(60)).pool_max_idle_per_host(8).build().unwrap();
    let mut ready = false;
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        if let Ok(r) = client.get(format!("{base}/_cluster/health")).basic_auth("elastic", Some("test-password")).send().await {
            if r.status().is_success() { ready = true; break; }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "secured node did not become ready");
    let r = client.put(format!("{base}/_security/user/hashed")).basic_auth("elastic", Some("test-password"))
        .json(&json!({"password": "correct-horse", "roles": ["superuser"]})).send().await.unwrap();
    assert!(r.status().is_success(), "{:?}", r.text().await);
    let mut stored = None;
    for _ in 0..100 {
        stored = engine.security.state.read().users.get("hashed").map(|u| u.password_hash.clone());
        if stored.is_some() { break; }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let stored = stored.expect("native user visible to the edge");
    assert!(elasticsearch_engine::security::hash::is_expensive(&stored), "test account must use an expensive hash: {stored}");

    // ---- #17 load: failed logins to a known bcrypt account vs authenticated latency.
    let mut idle = timed_authenticated(&client, &base, 100).await;
    let (idle_p50, idle_p99) = (percentile(&mut idle, 0.5), percentile(&mut idle, 0.99));
    elasticsearch_http::security::edge::set_failure_limits(u32::MAX, u32::MAX, Duration::from_secs(60));
    let pool = elasticsearch_http::security::edge::crypto_pool();
    let completed_before = pool.completed();
    let attack = { let (client, base) = (client.clone(), base.clone()); tokio::spawn(async move { failed_logins(&client, &base, 400, "hashed").await }) };
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut loaded = timed_authenticated(&client, &base, 100).await;
    let (unauthorized, rejected, attack_took) = attack.await.unwrap();
    let (loaded_p50, loaded_p99) = (percentile(&mut loaded, 0.5), percentile(&mut loaded, 0.99));
    eprintln!("[auth_containment] crypto pool {} threads / queue {}; 400 failed bcrypt logins: {unauthorized} x 401, {rejected} x 429 in {attack_took:?}; verified on pool: {}",
        pool.threads, pool.queue, pool.completed() - completed_before);
    eprintln!("[auth_containment] authenticated /_cluster/health: idle p50 {idle_p50:?} p99 {idle_p99:?}; under failed-login load p50 {loaded_p50:?} p99 {loaded_p99:?}");
    assert_eq!(unauthorized + rejected, 400);
    assert!(unauthorized > 0);
    assert!(loaded_p99 < Duration::from_millis(250), "authenticated requests starved by failed logins: p99 {loaded_p99:?}");

    // A small pool: admission refuses (429) instead of queueing without bound.
    elasticsearch_http::security::edge::set_crypto_pool(1, 4);
    let (unauthorized, rejected, took) = failed_logins(&client, &base, 200, "hashed").await;
    eprintln!("[auth_containment] pool 1/4: 200 failed logins: {unauthorized} x 401, {rejected} x 429 in {took:?}");
    assert!(rejected > 100, "{rejected} rejected");
    // The right password still authenticates (on the pool), then from the success cache.
    for _ in 0..2 {
        let r = client.get(format!("{base}/_security/_authenticate")).basic_auth("hashed", Some("correct-horse")).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
    }

    // The failed-authentication limiter: per user, from any address.
    elasticsearch_http::security::edge::set_crypto_pool(2, 1000);
    elasticsearch_http::security::edge::set_failure_limits(5, u32::MAX, Duration::from_secs(60));
    let mut statuses = Vec::new();
    for i in 0..10 {
        let t = Instant::now();
        let r = client.get(format!("{base}/_cluster/health")).basic_auth("hashed", Some(format!("nope-{i}"))).send().await.unwrap();
        statuses.push((r.status().as_u16(), t.elapsed()));
    }
    eprintln!("[auth_containment] limiter 5/min: {:?}", statuses.iter().map(|s| s.0).collect::<Vec<_>>());
    assert!(statuses[..5].iter().all(|s| s.0 == 401));
    assert!(statuses[5..].iter().all(|s| s.0 == 429), "{statuses:?}");
    // A cached success is not limited.
    let r = client.get(format!("{base}/_security/_authenticate")).basic_auth("hashed", Some("correct-horse")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    elasticsearch_http::security::edge::set_failure_limits(u32::MAX, u32::MAX, Duration::from_secs(60));

    // ---- #18: many incomplete bodies held open.
    elasticsearch_http::set_body_deadlines(Duration::from_millis(500), Duration::from_secs(5), Duration::from_secs(2));
    let inflight_before = engine.breakers.get(elasticsearch_engine::memory::Kind::InflightRequests).reserved();
    let started = Instant::now();
    let mut no_creds = Vec::new();
    let mut bad_creds = Vec::new();
    let mut good_creds = Vec::new();
    for i in 0..150 {
        no_creds.push(partial_request(&addr, None, "/stall/_doc", 10 << 20, 1024).await);
        if i % 3 == 0 { bad_creds.push(partial_request(&addr, Some(&basic("hashed", "wrong")), "/stall/_doc", 10 << 20, 1024).await); }
        if i % 3 == 1 { good_creds.push(partial_request(&addr, Some(&basic("elastic", "test-password")), "/_cluster/settings", 10 << 20, 64 << 10).await); }
    }
    // Unauthenticated: answered 401 without the body being read.
    for s in no_creds.iter_mut().chain(bad_creds.iter_mut()) {
        assert_eq!(response_status(s, Duration::from_secs(5)).await, Some(401));
    }
    let refused_in = started.elapsed();
    // While the authenticated control bodies stall: control-plane requests are served.
    let t = Instant::now();
    let r = client.put(format!("{base}/_cluster/settings")).basic_auth("elastic", Some("test-password"))
        .json(&json!({"persistent": {"cluster.routing.allocation.enable": "all"}})).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200, "{:?}", r.text().await);
    let r = client.get(format!("{base}/_cluster/health")).basic_auth("elastic", Some("test-password")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let control_took = t.elapsed();
    assert_eq!(elasticsearch_http::unauthenticated_body_bytes(), 0, "invalid credentials must not buffer bodies");
    // Authenticated stalled bodies: reclaimed by the idle deadline (408).
    for s in good_creds.iter_mut() {
        assert_eq!(response_status(s, Duration::from_secs(10)).await, Some(408));
    }
    let reclaimed_in = started.elapsed();
    let inflight_after = engine.breakers.get(elasticsearch_engine::memory::Kind::InflightRequests).reserved();
    eprintln!("[auth_containment] stalled bodies: 150 no-credential + {} bad-credential refused 401 by {refused_in:?}; {} authenticated stalled bodies reclaimed (408) by {reclaimed_in:?}; control requests during stall {control_took:?}; inflight before {inflight_before} after {inflight_after}",
        bad_creds.len(), good_creds.len());
    assert!(control_took < Duration::from_secs(2));
    assert!(reclaimed_in < Duration::from_secs(8));
    assert_eq!(inflight_after, inflight_before);
    drop((no_creds, bad_creds, good_creds));
    let r: Value = client.get(format!("{base}/_cluster/health")).basic_auth("elastic", Some("test-password")).send().await.unwrap().json().await.unwrap();
    assert!(r["status"].is_string());
    elasticsearch_http::set_body_deadlines(Duration::from_secs(30), Duration::from_secs(300), Duration::from_secs(10));

    // ---- Internal routing / coordination headers sent from outside have no effect.
    let internal: &[(&str, &str)] = &[
        ("x-es-cluster-shard", "7"), ("x-es-cluster-prefilter-shards", "[]"), ("x-es-cluster-exec-local", "1"),
        ("x-es-cluster-ack-through", "0:999999:1"), ("x-es-cluster-parent-task", "spoofed:1"),
        ("x-rustic-internal-system", "1"), ("x-rustic-internal-identity", "garbage"), ("x-rustic-internal-nested", "1"), ("x-rustic-internal-trips", "1"),
        ("x-rustic-internal-prepared-ingest", "garbage"), ("x-rustic-internal-failure-token", "garbage"),
        ("x-rustic-internal-failure-tokens", "garbage"), ("x-rustic-internal-failure-store-ack", "1"),
        ("x-rustic-internal-failure-store-positions", "garbage"), ("x-rustic-internal-datastream-backing-write", "1"),
        ("x-rustic-internal-esql-fetch", "garbage"), ("x-rustic-internal-esql-coordinator", "garbage"),
        ("x-rustic-internal-esql-forward", "garbage"), ("x-rustic-internal-esql-shards", "garbage"),
        ("x-rustic-internal-synonyms-inner", "1"), ("x-rustic-internal-rank-plan", "garbage"), ("x-rustic-internal-qtf", "1"),
        ("x-rustic-internal-qtf-ctx", "garbage"), ("x-rustic-internal-qtf-pinned", "garbage"), ("x-rustic-internal-qtf-light", "1"),
        ("x-rustic-internal-hit-spans", "1"), ("x-rustic-internal-scroll-touch", "garbage"), ("x-rustic-internal-read-copies", "garbage"),
        ("x-rustic-internal-fleet-wait", "garbage"), ("x-rustic-internal-ars", "garbage"), ("x-rustic-internal-ars-stats", "garbage"),
        ("x-rustic-internal-published-version", "999999"), ("x-es-original-content-type", "application/yaml"), ("x-es-alias-catalog", "spoof"),
    ];
    let r = client.put(format!("{base}/spoof/_doc/1?refresh=true")).basic_auth("elastic", Some("test-password")).json(&json!({"f": 1})).send().await.unwrap();
    assert!(r.status().is_success());
    let r = client.put(format!("{base}/_security/user/nobody")).basic_auth("elastic", Some("test-password"))
        .json(&json!({"password": "nobody-pass", "roles": []})).send().await.unwrap();
    assert!(r.status().is_success());
    let send = |method: reqwest::Method, path: &str, user: Option<(&str, &str)>, headers: &[(&str, &str)], body: Option<Value>| {
        let mut rb = client.request(method, format!("{base}{path}"));
        if let Some((u, p)) = user { rb = rb.basic_auth(u, Some(p)); }
        for (k, v) in headers { rb = rb.header(*k, *v); }
        if let Some(b) = body { rb = rb.json(&b); }
        async move { let r = rb.send().await.unwrap(); let status = r.status().as_u16(); (status, r.json::<Value>().await.unwrap_or(Value::Null)) }
    };
    let elastic = Some(("elastic", "test-password"));
    assert_eq!(send(reqwest::Method::GET, "/spoof/_search", elastic, &[], None).await.0, 200);
    for one in internal.iter().map(std::slice::from_ref).chain(std::iter::once(internal)) {
        let names: Vec<&str> = one.iter().map(|h| h.0).collect();
        // No identity from a header: unauthenticated stays 401, a user without roles stays 403.
        assert_eq!(send(reqwest::Method::GET, "/spoof/_search", None, one, None).await.0, 401, "{names:?}");
        assert_eq!(send(reqwest::Method::GET, "/spoof/_search", Some(("nobody", "nobody-pass")), one, None).await.0, 403, "{names:?}");
        // No routing / coordination effect: the same answer as without them.
        let (_, base_body) = send(reqwest::Method::GET, "/spoof/_search", elastic, &[], None).await;
        let (status, body) = send(reqwest::Method::GET, "/spoof/_search", elastic, one, None).await;
        assert_eq!(status, 200, "{names:?}: {body}");
        assert_eq!(body["hits"]["total"], base_body["hits"]["total"], "{names:?}");
        assert_eq!(body["_shards"], base_body["_shards"], "{names:?}");
        let (status, body) = send(reqwest::Method::POST, "/spoof/_doc?refresh=true", elastic, one, Some(json!({"g": 2}))).await;
        assert_eq!(status, 201, "{names:?}: {body}");
        let id = body["_id"].as_str().unwrap().to_string();
        let (status, body) = send(reqwest::Method::GET, &format!("/spoof/_doc/{id}"), elastic, one, None).await;
        assert_eq!((status, body["_source"].clone()), (200, json!({"g": 2})), "{names:?}");
        assert_eq!(send(reqwest::Method::GET, "/_cluster/health", elastic, one, None).await.0, 200, "{names:?}");
    }
    eprintln!("[auth_containment] {} internal headers sent from outside, one at a time and all together: no identity, routing or result change", internal.len());
}
