//! Production review #94 and #95 on the engine's REST layer:
//!
//! * #94: changed partial updates without `routing` (single and bulk) keep
//!   the stored routing, so a reindex into more shards still routes each
//!   document by it; explicit, required, no-op and scripted controls;
//! * #95: an invalid `refresh` fails index / create / delete / update /
//!   bulk before anything is created or changed; corrected and retried, a
//!   script runs once in total.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), client: Client::new(), _task: task }
}

impl Es {
    async fn call(&self, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json().await.unwrap_or(Value::Null))
    }

    async fn ok(&self, m: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(m.clone(), path, body).await;
        assert!((200..300).contains(&st), "{m} {path}: {st} {v}");
        v
    }

    async fn ndjson(&self, path: &str, lines: &[Value]) -> (u16, Value) {
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let resp = self.client.post(format!("{}{}", self.base, path)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json().await.unwrap_or(Value::Null))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partial_updates_keep_stored_routing_through_reindex() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.ok(Method::PUT, "/src", Some(json!({"settings": {"number_of_shards": 1}}))).await;
    for id in ["single", "bulk", "noop", "explicit", "script"] {
        es.ok(Method::PUT, &format!("/src/_doc/{id}?routing=r{id}"), Some(json!({"n": 1, "id": id}))).await;
    }
    let v = es.ok(Method::POST, "/src/_update/single", Some(json!({"doc": {"n": 2}}))).await;
    assert_eq!(v["result"], "updated");
    let (st, v) = es.ndjson("/_bulk", &[json!({"update": {"_index": "src", "_id": "bulk"}}), json!({"doc": {"n": 2}})]).await;
    assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
    let v = es.ok(Method::POST, "/src/_update/noop", Some(json!({"doc": {"n": 1}}))).await;
    assert_eq!(v["result"], "noop");
    es.ok(Method::POST, "/src/_update/explicit?routing=other", Some(json!({"doc": {"n": 2}}))).await;
    es.ok(Method::POST, "/src/_update/script", Some(json!({"script": {"source": "ctx._source.n += 1"}}))).await;
    es.ok(Method::POST, "/src/_refresh", None).await;
    let want = |id: &str| if id == "explicit" { "other".to_string() } else { format!("r{id}") };
    for id in ["single", "bulk", "noop", "explicit", "script"] {
        let v = es.ok(Method::GET, &format!("/src/_doc/{id}?routing={}", want(id)), None).await;
        assert_eq!(v["_routing"], want(id), "{v}");
    }
    // Reindex (routing kept by default) into more shards: each document
    // is found by its routing there.
    es.ok(Method::PUT, "/dst", Some(json!({"settings": {"number_of_shards": 5}}))).await;
    let v = es.ok(Method::POST, "/_reindex?refresh=true", Some(json!({"source": {"index": "src"}, "dest": {"index": "dst"}}))).await;
    assert_eq!(v["created"], 5, "{v}");
    for id in ["single", "bulk", "noop", "explicit", "script"] {
        let v = es.ok(Method::GET, &format!("/dst/_doc/{id}?routing={}", want(id)), None).await;
        assert_eq!((&v["found"], &v["_routing"]), (&json!(true), &json!(want(id))), "{v}");
    }
    // Required routing: an update without it is still refused.
    es.ok(Method::PUT, "/req", Some(json!({"mappings": {"_routing": {"required": true}}}))).await;
    es.ok(Method::PUT, "/req/_doc/1?routing=a", Some(json!({"n": 1}))).await;
    let (st, v) = es.call(Method::POST, "/req/_update/1", Some(json!({"doc": {"n": 2}}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("routing_missing_exception")), "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_refresh_is_rejected_before_any_change() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let unknown = |st: u16, v: &Value| st == 400 && v["error"]["type"] == "illegal_argument_exception" && v["error"]["reason"] == "Unknown value for refresh: [typo].";
    // Nothing auto-created.
    let (st, v) = es.call(Method::PUT, "/fresh/_doc/1?refresh=typo", Some(json!({"n": 1}))).await;
    assert!(unknown(st, &v), "{st} {v}");
    let (st, v) = es.ndjson("/_bulk?refresh=typo", &[json!({"index": {"_index": "fresh2", "_id": "1"}}), json!({"n": 1})]).await;
    assert!(unknown(st, &v), "{st} {v}");
    for index in ["fresh", "fresh2"] {
        assert_eq!(es.call(Method::GET, &format!("/{index}"), None).await.0, 404, "[{index}] must not exist");
    }
    es.ok(Method::PUT, "/s/_doc/1?refresh=true", Some(json!({"n": 0}))).await;
    let before = es.ok(Method::GET, "/s/_doc/1", None).await;
    let script = json!({"script": {"source": "ctx._source.n += 1"}});
    for (m, path, body) in [
        (Method::PUT, "/s/_doc/1?refresh=typo", Some(json!({"n": 100}))),
        (Method::POST, "/s/_doc?refresh=typo", Some(json!({"n": 100}))),
        (Method::PUT, "/s/_create/2?refresh=typo", Some(json!({"n": 100}))),
        (Method::DELETE, "/s/_doc/1?refresh=typo", None),
        (Method::POST, "/s/_update/1?refresh=typo", Some(script.clone())),
    ] {
        let (st, v) = es.call(m.clone(), path, body).await;
        assert!(unknown(st, &v), "{m} {path}: {st} {v}");
    }
    let (st, v) = es.ndjson("/_bulk?refresh=typo", &[json!({"update": {"_index": "s", "_id": "1"}}), script.clone(), json!({"index": {"_index": "s", "_id": "3"}}), json!({"n": 3})]).await;
    assert!(unknown(st, &v), "{st} {v}");
    es.ok(Method::POST, "/s/_refresh", None).await;
    let after = es.ok(Method::GET, "/s/_doc/1", None).await;
    for k in ["_version", "_seq_no", "_primary_term", "_source"] {
        assert_eq!(after[k], before[k], "{k}: {before} -> {after}");
    }
    assert_eq!(es.ok(Method::POST, "/s/_count", None).await["count"], 1);
    // Corrected and retried: executed once in total.
    es.ok(Method::POST, "/s/_update/1?refresh=wait_for", Some(script)).await;
    let v = es.ok(Method::GET, "/s/_doc/1", None).await;
    assert_eq!((&v["_source"]["n"], &v["_seq_no"]), (&json!(1), &json!(1)), "{v}");
    // Valid values still work.
    for r in ["", "true", "false", "wait_for"] {
        es.ok(Method::PUT, &format!("/s/_doc/v?refresh={r}"), Some(json!({"n": 1}))).await;
    }
}
