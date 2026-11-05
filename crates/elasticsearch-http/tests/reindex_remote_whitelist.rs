//! `reindex.remote.whitelist` comes from the node settings (as in ES), not
//! only from the `ES_REINDEX_REMOTE_WHITELIST` environment variable.

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};

async fn serve(dir: &std::path::Path, whitelist: Option<Value>) -> String {
    let node = Node::open(dir).unwrap();
    if let Some(w) = whitelist {
        node.node_settings.write().insert("reindex.remote.whitelist".into(), w);
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    format!("127.0.0.1:{}", addr.port())
}

async fn call(c: &Client, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.post(url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn node_setting_whitelist_allows_and_refuses() {
    let src_dir = tempfile::tempdir().unwrap();
    let src = serve(src_dir.path(), None).await;
    let c = Client::new();
    let bulk: String = (0..5).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "src", "_id": i.to_string()}}), json!({"n": i}))).collect();
    let r = c.post(format!("http://{src}/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert!(r.status().is_success());

    // Allowed: the list form, as `reindex.remote.whitelist: ["host:port"]` in elasticsearch.yml.
    let ok_dir = tempfile::tempdir().unwrap();
    let ok = serve(ok_dir.path(), Some(json!([src.clone()]))).await;
    let body = json!({"source": {"remote": {"host": format!("http://{src}")}, "index": "src"}, "dest": {"index": "dst"}});
    let (st, r) = call(&c, format!("http://{ok}/_reindex?refresh=true"), Some(body.clone())).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["created"], 5, "{r}");

    // Allowed: a comma separated string with a wildcard pattern.
    let wild_dir = tempfile::tempdir().unwrap();
    let wild = serve(wild_dir.path(), Some(json!("otherhost:9200, 127.0.0.1:*"))).await;
    let (st, r) = call(&c, format!("http://{wild}/_reindex?refresh=true"), Some(body.clone())).await;
    assert_eq!(st, 200, "{r}");

    // Refused: not in the node's whitelist.
    let no_dir = tempfile::tempdir().unwrap();
    let no = serve(no_dir.path(), Some(json!(["otherhost:9200"]))).await;
    let (st, r) = call(&c, format!("http://{no}/_reindex"), Some(body)).await;
    assert_eq!(st, 400, "{r}");
    assert!(r["error"]["reason"].as_str().unwrap_or("").contains("not whitelisted in reindex.remote.whitelist"), "{r}");
}
