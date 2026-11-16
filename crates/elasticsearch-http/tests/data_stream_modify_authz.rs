//! Review finding 374: `POST /_data_stream/_modify` authorizes every data
//! stream and index named in its body: a `manage` grant scoped to one
//! dataset never detaches (or deletes) another dataset's backing indices.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};

const PW: &str = "ds-modify-authz-password";

async fn call(client: &Client, base: &str, user: &str, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let auth = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{PW}")));
    let mut r = client.request(m, format!("{base}{path}")).header("authorization", auth);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    (st, serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_stream_modify_authorizes_every_named_stream_and_index() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    cfg.trial = true;
    for u in ["admin", "alice"] {
        cfg.file_users.insert(u.into(), Hasher::Pbkdf2(1000).hash(PW));
    }
    cfg.file_user_roles.insert("admin".into(), vec!["superuser".into()]);
    cfg.file_user_roles.insert("alice".into(), vec!["logs_a_manager".into()]);
    node.security.configure(cfg);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let n = node.clone();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    let c = Client::new();
    let ok = |(st, v): (u16, Value)| assert!(st < 300, "{st} {v}");
    ok(call(&c, &base, "admin", Method::PUT, "/_security/role/logs_a_manager", Some(json!({"indices": [{"names": ["logs-a*"], "privileges": ["manage"]}]}))).await);
    ok(call(&c, &base, "admin", Method::PUT, "/_index_template/logs", Some(json!({"index_patterns": ["logs-*"], "data_stream": {}, "template": {"settings": {"number_of_replicas": 0}}}))).await);
    for ds in ["logs-a", "logs-b"] {
        ok(call(&c, &base, "admin", Method::PUT, &format!("/_data_stream/{ds}"), None).await);
        ok(call(&c, &base, "admin", Method::POST, &format!("/{ds}/_rollover"), None).await);
    }
    let first = |ds: &str| node.data_stream(ds).unwrap().indices[0].clone();
    let (a0, b0) = (first("logs-a"), first("logs-b"));
    let remove = |ds: &str, ix: &str| json!({"remove_backing_index": {"data_stream": ds, "index": ix}});
    // Another dataset's member: refused, nothing changes.
    let (st, v) = call(&c, &base, "alice", Method::POST, "/_data_stream/_modify", Some(json!({"actions": [remove("logs-b", &b0)]}))).await;
    assert_eq!(st, 403, "{v}");
    assert_eq!(node.data_stream("logs-b").unwrap().indices.len(), 2);
    // Mixed: one permitted, one not: refused as a whole, nothing changes.
    let (st, v) = call(&c, &base, "alice", Method::POST, "/_data_stream/_modify", Some(json!({"actions": [remove("logs-a", &a0), remove("logs-b", &b0)]}))).await;
    assert_eq!(st, 403, "{v}");
    assert_eq!(node.data_stream("logs-a").unwrap().indices.len(), 2);
    assert_eq!(node.data_stream("logs-b").unwrap().indices.len(), 2);
    // Its own dataset: allowed.
    ok(call(&c, &base, "alice", Method::POST, "/_data_stream/_modify", Some(json!({"actions": [remove("logs-a", &a0)]}))).await);
    assert_eq!(node.data_stream("logs-a").unwrap().indices.len(), 1);
}
