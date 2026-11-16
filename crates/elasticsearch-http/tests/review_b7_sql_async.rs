//! Review batch 7, finding 612: async SQL keeps `fetch_size` paging (a
//! first page and a cursor for the rest, every row exactly once), on the
//! submit and on the saved result's GET.

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};

async fn start(dir: &std::path::Path) -> String {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    format!("http://{addr}")
}

async fn call(c: &Client, method: reqwest::Method, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    (st, resp.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn async_sql_keeps_fetch_size_paging() {
    let tmp = tempfile::tempdir().unwrap();
    let base = start(tmp.path()).await;
    let c = Client::new();
    let (st, v) = call(&c, reqwest::Method::PUT, format!("{base}/t"), Some(json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"n": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let mut bulk = String::new();
    for i in 0..250 {
        bulk.push_str(&format!("{{\"index\":{{}}}}\n{{\"n\":{i}}}\n"));
    }
    let r = c.post(format!("{base}/t/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let q = json!({"query": "SELECT n FROM t ORDER BY n", "fetch_size": 100, "wait_for_completion_timeout": "10s", "keep_on_completion": true});
    let (st, first) = call(&c, reqwest::Method::POST, format!("{base}/_sql?format=json"), Some(q)).await;
    assert_eq!(st, 200, "{first}");
    assert_eq!(first["is_running"], false, "{first}");
    assert_eq!(first["rows"].as_array().unwrap().len(), 100, "the async result ignored fetch_size");
    let id = first["id"].as_str().unwrap().to_string();
    let mut seen: Vec<i64> = first["rows"].as_array().unwrap().iter().map(|r| r[0].as_i64().unwrap()).collect();
    let mut cursor = first["cursor"].as_str().expect("no cursor").to_string();
    loop {
        let (st, page) = call(&c, reqwest::Method::POST, format!("{base}/_sql?format=json"), Some(json!({"cursor": cursor}))).await;
        assert_eq!(st, 200, "{page}");
        seen.extend(page["rows"].as_array().unwrap().iter().map(|r| r[0].as_i64().unwrap()));
        match page["cursor"].as_str() {
            Some(c2) => cursor = c2.to_string(),
            None => break,
        }
    }
    assert_eq!(seen, (0..250).collect::<Vec<i64>>(), "every row exactly once");
    // The saved result: the same bounded first page.
    let (st, saved) = call(&c, reqwest::Method::GET, format!("{base}/_sql/async/{id}?format=json"), None).await;
    assert_eq!(st, 200, "{saved}");
    assert_eq!(saved["rows"].as_array().unwrap().len(), 100, "{saved}");
    assert!(saved["cursor"].is_string(), "{saved}");
    let (st, status) = call(&c, reqwest::Method::GET, format!("{base}/_sql/async/status/{id}"), None).await;
    assert_eq!((st, &status["is_running"], &status["completion_status"]), (200, &json!(false), &json!(200)), "{status}");
}
