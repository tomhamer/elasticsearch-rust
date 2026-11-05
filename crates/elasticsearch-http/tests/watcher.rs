//! Watcher end-to-end tests: a scheduled watch with a search input, a compare
//! condition and an index action fires on its own; history documents, throttling
//! and acking; the REST surface used by Kibana's Watcher UI.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::time::Duration;

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    // Like ES's own test clusters: a self-generated trial license, so the
    // license-gated APIs exercised here are served (ES 403s them on basic).
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.call(Method::GET, p, None).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    /// POST without a body (ES rejects a body on `_refresh`, `_ack`, `_stop`, `_start`).
    async fn post0(&self, p: &str) -> (u16, Value) {
        self.call(Method::POST, p, None).await
    }
    async fn wait_for(&self, what: &str, f: impl Fn(&Value) -> bool, p: &str, body: Option<Value>) -> Value {
        for _ in 0..80 {
            let (_, v) = self.call(if body.is_some() { Method::POST } else { Method::GET }, p, body.clone()).await;
            if f(&v) {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        panic!("timed out waiting for {what}");
    }
}

#[tokio::test]
async fn scheduled_watch_fires_indexes_and_records_history() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    for (i, n) in [3, 8, 11].iter().enumerate() {
        let (st, _) = es.put(&format!("/logs/_doc/{i}?refresh=true"), json!({"level": "error", "n": n})).await;
        assert!(st == 200 || st == 201);
    }
    let watch = json!({
        "trigger": {"schedule": {"interval": "1s"}},
        "input": {"search": {"request": {"indices": ["logs"], "body": {"query": {"term": {"level": "error"}}}}}},
        "condition": {"compare": {"ctx.payload.hits.total": {"gte": 3}}},
        "actions": {
            "save": {"index": {"index": "alerts", "doc_id": "a1", "execution_time_field": "at"}},
            "note": {"logging": {"text": "errors: {{ctx.payload.hits.total}}"}, "throttle_period": "1h"}
        }
    });
    let (st, r) = es.put("/_watcher/watch/errors", watch).await;
    assert_eq!(st, 201, "{r}");
    assert_eq!(r["created"], true);

    // the scheduler fires it on its own
    let hist = es
        .wait_for(
            "history record",
            |v| v["hits"]["hits"].as_array().map(|a| a.len() >= 2).unwrap_or(false),
            "/.watcher-history-17*/_search?size=10&sort=@timestamp:asc",
            Some(json!({"query": {"term": {"watch_id": "errors"}}})),
        )
        .await;
    let rec = &hist["hits"]["hits"][0]["_source"];
    assert_eq!(rec["state"], "executed");
    assert_eq!(rec["trigger_event"]["type"], "schedule");
    assert_eq!(rec["result"]["input"]["type"], "search");
    assert_eq!(rec["result"]["input"]["payload"]["hits"]["total"], 3);
    assert_eq!(rec["result"]["condition"]["met"], true);
    let acts = rec["result"]["actions"].as_array().unwrap();
    assert!(acts.iter().any(|a| a["id"] == "save" && a["status"] == "success"));
    // the second firing: the logging action is throttled (1h), the index action (5s default) may be too
    let rec2 = &hist["hits"]["hits"][1]["_source"];
    let note = rec2["result"]["actions"].as_array().unwrap().iter().find(|a| a["id"] == "note").unwrap().clone();
    assert_eq!(note["status"], "throttled", "{note}");
    assert!(note["reason"].as_str().unwrap().starts_with("throttling interval is set to [1h]"));

    // index action wrote the document
    es.post0("/alerts/_refresh").await;
    let (st, d) = es.get("/alerts/_doc/a1").await;
    assert_eq!(st, 200);
    assert!(d["_source"]["at"].is_string());

    // ack state machine
    let (_, g) = es.get("/_watcher/watch/errors").await;
    assert_eq!(g["status"]["actions"]["note"]["ack"]["state"], "ackable");
    let (st, a) = es.post0("/_watcher/watch/errors/_ack/note").await;
    assert_eq!(st, 200);
    assert_eq!(a["status"]["actions"]["note"]["ack"]["state"], "acked");
    assert_eq!(a["status"]["actions"]["save"]["ack"]["state"], "ackable");

    // deactivate stops firing
    let (_, d) = es.call(Method::PUT, "/_watcher/watch/errors/_deactivate", None).await;
    assert_eq!(d["status"]["state"]["active"], false);
    let (_, before) = es.get("/_watcher/stats").await;
    assert_eq!(before["stats"][0]["watch_count"], 0);

    // stats / settings / query used by Kibana
    let (_, q) = es.post("/_watcher/_query/watches", json!({"query": {"match_all": {}}})).await;
    assert_eq!(q["count"], 1);
    assert_eq!(q["watches"][0]["_id"], "errors");
    let (st, _) = es.post0("/_watcher/_stop").await;
    assert_eq!(st, 200);
    let (_, s) = es.get("/_watcher/stats").await;
    assert_eq!(s["stats"][0]["watcher_state"], "stopped");
    es.post0("/_watcher/_start").await;
}

#[tokio::test]
async fn execute_modes_conditions_and_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, r) = es
        .post(
            "/_watcher/watch/_execute",
            json!({"watch": {
                "trigger": {"schedule": {"interval": "1h"}},
                "input": {"simple": {"v": [1, 2, 3]}},
                "condition": {"script": "return ctx.payload.v.size() > 2"},
                "actions": {"a": {"logging": {"text": "n={{ctx.payload.v.0}}"}}}
            }}),
        )
        .await;
    assert_eq!(st, 200, "{r}");
    let rec = &r["watch_record"];
    assert_eq!(rec["watch_id"], "_inlined_");
    assert_eq!(rec["result"]["condition"]["type"], "script");
    assert_eq!(rec["result"]["actions"][0]["logging"]["logged_text"], "n=1");

    let (st, e) = es.put("/_watcher/watch/bad", json!({"input": {"simple": {}}})).await;
    assert_eq!(st, 400);
    assert_eq!(e["error"]["reason"], "could not parse watch [bad]. missing required field [trigger]");
    let (st, e) = es.get("/_watcher/watch/missing").await;
    assert_eq!(st, 404);
    assert_eq!(e["found"], false);
}
