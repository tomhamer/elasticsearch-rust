//! Review #106: background-job admission with a deliberately small gate
//! (one running job, one queued): overflow is rejected (429) and leaves no
//! task behind; cancellation frees the capacity. Its own binary: the gate
//! reads its size from the environment once per process.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
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
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    async fn post0(&self, p: &str) -> (u16, Value) {
        self.call(Method::POST, p, None).await
    }
    async fn bulk(&self, lines: &[Value]) {
        let mut s = String::new();
        for l in lines {
            s.push_str(&l.to_string());
            s.push('\n');
        }
        let r = self.c.post(format!("{}/_bulk?refresh=true", self.base)).header("content-type", "application/x-ndjson").body(s).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }
}

fn running_byquery(l: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for n in l["nodes"].as_object().into_iter().flat_map(|m| m.values()) {
        for id in n["tasks"].as_object().into_iter().flat_map(|m| m.keys()) {
            out.push(id.clone());
        }
    }
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overflow_is_rejected_without_orphans_and_cancel_frees_capacity() {
    // SAFETY: set before any thread of this test binary reads it (the
    // binary's only test; the gate reads it on first use).
    unsafe {
        std::env::set_var("ES_BACKGROUND_TASKS_MAX_RUNNING", "1");
        std::env::set_var("ES_BACKGROUND_TASKS_QUEUE_SIZE", "1");
    }
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/adm", json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for i in 0..4 {
        lines.push(json!({"index": {"_index": "adm", "_id": i.to_string()}}));
        lines.push(json!({"n": i}));
    }
    es.bulk(&lines).await;
    let gate = elasticsearch_http::pools::background_jobs();
    assert_eq!(gate.limit(), 1);

    // A: one document, then a throttle pause of ~1000 s; B queues behind it.
    let slow = |n: u32| ("/adm/_update_by_query?wait_for_completion=false&requests_per_second=0.001&scroll_size=1".to_string(), json!({"query": {"term": {"n": n}}}));
    let (p, b) = slow(0);
    let (st, a) = es.post(&p, b).await;
    assert_eq!(st, 200, "{a}");
    let a = a["task"].as_str().unwrap().to_string();
    let (p, b) = slow(1);
    let (st, bq) = es.post(&p, b).await;
    assert_eq!(st, 200, "{bq}");
    let bq = bq["task"].as_str().unwrap().to_string();
    assert_eq!(gate.stats(), elasticsearch_http::pools::JobGateStats { running: 1, queued: 1 });

    // C: the queue is full: ES's rejection, and no task left behind.
    let (p, b) = slow(2);
    let (st, c) = es.post(&p, b).await;
    assert_eq!(st, 429, "{c}");
    assert_eq!(c["error"]["type"], "es_rejected_execution_exception", "{c}");
    let (_, l) = es.get("/_tasks?actions=*byquery&detailed").await;
    let mut want = vec![a.clone(), bq.clone()];
    want.sort();
    assert_eq!(running_byquery(&l), want, "{l}");

    // Cancelling A frees its permit promptly; B runs; cancelling B frees all.
    let started = std::time::Instant::now();
    let (st, _) = es.post0(&format!("/_tasks/{a}/_cancel")).await;
    assert_eq!(st, 200);
    let (st, t) = es.get(&format!("/_tasks/{a}?wait_for_completion=true&timeout=20s")).await;
    assert_eq!((st, t["completed"].clone()), (200, json!(true)), "{t}");
    let (st, _) = es.post0(&format!("/_tasks/{bq}/_cancel")).await;
    assert_eq!(st, 200);
    let (st, t) = es.get(&format!("/_tasks/{bq}?wait_for_completion=true&timeout=20s")).await;
    assert_eq!((st, t["completed"].clone()), (200, json!(true)), "{t}");
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "cancellation is prompt: {:?}", started.elapsed());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while gate.stats() != elasticsearch_http::pools::JobGateStats::default() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(gate.stats(), elasticsearch_http::pools::JobGateStats::default());
    let (_, l) = es.get("/_tasks?actions=*byquery").await;
    assert_eq!(l, json!({"nodes": {}}), "no task left running");

    // The freed capacity admits new work, which completes.
    let (st, d) = es.post("/adm/_update_by_query?wait_for_completion=false", json!({"query": {"term": {"n": 3}}})).await;
    assert_eq!(st, 200, "{d}");
    let d = d["task"].as_str().unwrap().to_string();
    let (st, t) = es.get(&format!("/_tasks/{d}?wait_for_completion=true&timeout=20s")).await;
    assert_eq!(st, 200, "{t}");
    assert_eq!(t["response"]["updated"], 1, "{t}");
}
