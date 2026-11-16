//! Index authorization of ES|QL, SQL and EQL on a secured two-node cluster
//! whose shards all live on the node that is not the coordinator: a user
//! or API key with `read` on `public` only, and an authenticated account
//! with no privileges, never read `private`, whichever node coordinates.

mod common;

use reqwest::Method;
use serde_json::{json, Value};

const PW: &str = "test-password";

/// How a request authenticates.
#[derive(Clone)]
enum As {
    User(&'static str),
    Key(String),
}

async fn call(c: &common::Cluster, node: usize, who: &As, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.c.request(method, format!("http://127.0.0.1:{}{path}", c.p.http[node]));
    r = match who {
        As::User(u) => r.basic_auth(u, Some(PW)),
        As::Key(k) => r.header("authorization", format!("ApiKey {k}")),
    };
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request");
    let st = resp.status().as_u16();
    (st, resp.json().await.unwrap_or(Value::Null))
}

fn no_leak(what: &str, r: &Value) {
    let s = r.to_string();
    assert!(!s.contains("top-secret") && !s.contains("\"v\":2"), "{what} leaked private data: {s}");
}

fn unknown_index(what: &str, st: u16, r: &Value, name: &str) {
    assert_eq!(st, 400, "{what}: {r}");
    assert!(r["error"]["reason"].as_str().unwrap_or("").contains(&format!("Unknown index [{name}]")), "{what}: {r}");
    no_leak(what, r);
}

fn forbidden(what: &str, st: u16, r: &Value) {
    assert_eq!(st, 403, "{what}: {r}");
    assert_eq!(r["error"]["type"], "security_exception", "{what}: {r}");
    no_leak(what, r);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn query_languages_authorize_indices_on_every_node() {
    let c = common::start(&[
        ("xpack.security.enabled", "true"),
        ("bootstrap.password", PW),
        ("xpack.license.self_generated.type", "trial"),
        ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000"),
    ])
    .await;
    let admin = As::User("elastic");
    // Every shard on the non-master node: the master coordinates, the other node reads.
    c.exclude_master().await;
    for index in ["public", "private"] {
        let (st, r) = call(&c, c.master, &admin, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": 2, "number_of_replicas": 0},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}}, "v": {"type": "long"}, "secret": {"type": "keyword"}}}
        }))).await;
        assert_eq!(st, 200, "{index}: {r}");
    }
    for (index, v, secret) in [("public", 1, "public-data"), ("private", 2, "top-secret")] {
        for id in 0..4 {
            let (st, r) = call(&c, c.master, &admin, Method::PUT, &format!("/{index}/_doc/{id}?refresh=true"), Some(json!({"@timestamp": "2026-11-01T00:00:00Z", "event": {"category": "process"}, "v": v, "secret": secret}))).await;
            assert_eq!(st, 201, "{r}");
        }
    }
    for _ in 0..100 {
        if c.primary_off_master("public").await && c.primary_off_master("private").await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(c.primary_off_master("public").await && c.primary_off_master("private").await);
    for (role, body) in [
        ("pub_read", json!({"indices": [{"names": ["public"], "privileges": ["read"]}]})),
        ("nothing", json!({"cluster": ["monitor"]})),
    ] {
        let (st, r) = call(&c, c.master, &admin, Method::PUT, &format!("/_security/role/{role}"), Some(body)).await;
        assert_eq!(st, 200, "{r}");
    }
    for (user, role) in [("reader", "pub_read"), ("nobody", "nothing")] {
        let (st, r) = call(&c, c.master, &admin, Method::PUT, &format!("/_security/user/{user}"), Some(json!({"password": PW, "roles": [role]}))).await;
        assert_eq!(st, 200, "{r}");
    }
    let (st, r) = call(&c, c.master, &admin, Method::POST, "/_security/api_key", Some(json!({"name": "pub", "role_descriptors": {"r": {"indices": [{"names": ["public"], "privileges": ["read"]}]}}}))).await;
    assert_eq!(st, 200, "{r}");
    let key = As::Key(r["encoded"].as_str().unwrap().to_string());
    let reader = As::User("reader");
    let nobody = As::User("nobody");

    // The superuser reads everything through the master.
    let (st, r) = call(&c, c.master, &admin, Method::POST, "/_query", Some(json!({"query": "FROM private | STATS n = COUNT(*)"}))).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["values"], json!([[4]]));

    for node in [c.master, c.other] {
        for who in [&reader, &key] {
            let esql = |q: &str| json!({"query": q});
            let (st, r) = call(&c, node, who, Method::POST, "/_query", Some(esql("FROM public | STATS n = COUNT(*)"))).await;
            assert_eq!(st, 200, "{r}");
            assert_eq!(r["values"], json!([[4]]));
            for (q, name) in [("FROM private | LIMIT 10", "private"), ("FROM public,private | LIMIT 10", "private"), ("FROM priv* | LIMIT 10", "priv*")] {
                let (st, r) = call(&c, node, who, Method::POST, "/_query", Some(esql(q))).await;
                unknown_index(q, st, &r, name);
                let (st, r) = call(&c, node, who, Method::POST, "/_query/async", Some(json!({"query": q, "wait_for_completion_timeout": "30s"}))).await;
                unknown_index(q, st, &r, name);
            }
            let (st, r) = call(&c, node, who, Method::POST, "/_query", Some(esql("FROM * | STATS m = MAX(v), n = COUNT(*)"))).await;
            assert_eq!(st, 200, "{r}");
            assert_eq!(r["values"], json!([[1, 4]]), "{r}");
            let (st, r) = call(&c, node, who, Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT secret FROM private"}))).await;
            unknown_index("sql", st, &r, "private");
            let (st, r) = call(&c, node, who, Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT MAX(v) AS m, COUNT(*) AS n FROM \"*\""}))).await;
            assert_eq!(st, 200, "{r}");
            assert_eq!(r["rows"], json!([[1, 4]]), "{r}");
            let (st, r) = call(&c, node, who, Method::POST, "/private/_eql/search", Some(json!({"query": "process where true"}))).await;
            forbidden("eql", st, &r);
        }
        // An authenticated account without index privileges runs nothing.
        for path in ["/_query", "/_query/async"] {
            let (st, r) = call(&c, node, &nobody, Method::POST, path, Some(json!({"query": "FROM public | LIMIT 1"}))).await;
            forbidden(path, st, &r);
        }
        let (st, r) = call(&c, node, &nobody, Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT * FROM public"}))).await;
        forbidden("sql", st, &r);
    }

    // Stored results and cursors stay with their owner on every node.
    let (st, r) = call(&c, c.master, &admin, Method::POST, "/_query/async", Some(json!({"query": "FROM private | KEEP secret | LIMIT 1", "keep_on_completion": true, "wait_for_completion_timeout": "30s"}))).await;
    assert_eq!(st, 200, "{r}");
    let id = r["id"].as_str().unwrap().to_string();
    let (st, first) = call(&c, c.master, &admin, Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT secret FROM private", "fetch_size": 1}))).await;
    assert_eq!(st, 200, "{first}");
    let cursor = first["cursor"].as_str().unwrap().to_string();
    for node in [c.master, c.other] {
        for who in [&reader, &key] {
            let (st, r) = call(&c, node, who, Method::GET, &format!("/_query/async/{id}"), None).await;
            assert_eq!(st, 404, "{r}");
            no_leak("async get", &r);
            let (st, r) = call(&c, node, who, Method::POST, "/_sql?format=json", Some(json!({"cursor": cursor}))).await;
            assert_eq!(st, 404, "{r}");
            no_leak("cursor", &r);
        }
        let (st, r) = call(&c, node, &nobody, Method::GET, &format!("/_query/async/{id}"), None).await;
        forbidden("async get", st, &r);
    }
    let (st, r) = call(&c, c.master, &admin, Method::GET, &format!("/_query/async/{id}"), None).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["values"], json!([["top-secret"]]));
    let (st, r) = call(&c, c.master, &admin, Method::POST, "/_sql?format=json", Some(json!({"cursor": cursor}))).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["rows"], json!([["top-secret"]]));
}
