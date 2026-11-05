//! Licensing is hardwired to enterprise/trial (AGENTS.md "Licensing: always
//! enterprise"). This test fails if anything makes the effective license
//! non-enterprise: a node asked for a basic self-generated license, then
//! `start_basic`, a `POST /_license` of a basic license and `DELETE
//! /_license`, still reports an active trial, serves every licensed feature
//! and keeps synthetic source.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

async fn call(c: &Client, base: &str, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(m, format!("{base}{p}"));
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

async fn assert_enterprise(c: &Client, base: &str, when: &str) {
    let (status, body) = call(c, base, Method::GET, "/_license", None).await;
    assert_eq!(status, 200, "{when}: {body}");
    assert_eq!(body["license"]["type"], "trial", "{when}: {body}");
    assert_eq!(body["license"]["status"], "active", "{when}: {body}");
    assert!(body["license"]["expiry_date_in_millis"].as_i64().unwrap() > 4_000_000_000_000, "{when}: {body}");
    let (status, info) = call(c, base, Method::GET, "/_xpack", None).await;
    assert_eq!(status, 200, "{when}: {info}");
    assert_eq!(info["license"]["mode"], "trial", "{when}: {info}");
    for (name, f) in info["features"].as_object().unwrap() {
        assert_eq!(f["available"], json!(true), "{when}: feature {name} unavailable: {info}");
    }
    let (_, body) = call(c, base, Method::GET, "/_license/trial_status", None).await;
    assert_eq!(body, json!({"eligible_to_start_trial": false}), "{when}");
    let (_, body) = call(c, base, Method::GET, "/_license/basic_status", None).await;
    assert_eq!(body, json!({"eligible_to_start_basic": true}), "{when}");
}

#[tokio::test(flavor = "multi_thread")]
async fn licence_stays_enterprise_whatever_is_requested() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("basic"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let c = Client::new();
    assert_enterprise(&c, &base, "fresh node").await;
    let (status, body) = call(&c, &base, Method::PUT, "/synthetic", Some(json!({"settings": {"index.mapping.source.mode": "synthetic"}}))).await;
    assert_eq!(status, 200, "{body}");
    for (m, p, b) in [
        (Method::POST, "/_license/start_basic?acknowledge=true", None),
        (Method::POST, "/_license?acknowledge=true", Some(json!({"license": {"uid": "893361dc-9749-4997-93cb-802e3d7fa4xx", "type": "basic", "issue_date_in_millis": 1411948800000_i64,
            "expiry_date_in_millis": 1914278399999_i64, "max_nodes": 1, "issued_to": "issuedTo", "issuer": "issuer"}}))),
        (Method::DELETE, "/_license", None),
        (Method::POST, "/_license/start_trial?acknowledge=true", None),
    ] {
        let (status, body) = call(&c, &base, m.clone(), p, b).await;
        assert_eq!(status, 200, "{m} {p}: {body}");
        assert_enterprise(&c, &base, &format!("after {m} {p}")).await;
    }
    let (status, body) = call(&c, &base, Method::PUT, "/synthetic-after", Some(json!({"settings": {"index.mapping.source.mode": "synthetic"}}))).await;
    assert_eq!(status, 200, "{body}");
    for index in ["synthetic", "synthetic-after"] {
        let (_, body) = call(&c, &base, Method::GET, &format!("/{index}/_settings?flat_settings=true"), None).await;
        assert_eq!(body[index]["settings"]["index.mapping.source.mode"], "synthetic", "{index}: {body}");
    }
}
