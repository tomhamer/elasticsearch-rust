//! Review #106: `wait_for_completion=false` by-query / reindex jobs run on
//! the bounded background-job gate, never a native thread per job. Its own
//! binary: the thread count and the gate are process-wide.

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

/// The process's live OS threads (Linux).
fn os_threads() -> usize {
    std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
}

/// Review #106: 500 concurrent throttled `wait_for_completion=false`
/// update-by-query jobs run on the bounded background-job gate (ES's
/// generic pool tasks), never a native thread each: the process's thread
/// count stays bounded while they run, at most the gate's limit run at
/// once, and every job still completes with its document updated.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_async_by_query_jobs_run_on_bounded_threads() {
    const JOBS: usize = 500;
    // The 500 concurrent `_tasks` long-polls hold a client and a server
    // socket each in this one process: raise the soft open-file limit to the
    // hard one, as the node binary does at startup (`natives.rs`), or the
    // default 1024 runs out (EMFILE on connect / accept).
    unsafe {
        let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut r) == 0 && r.rlim_cur < r.rlim_max {
            r.rlim_cur = r.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &r);
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let es = std::sync::Arc::new(start(tmp.path()).await);
    let (st, v) = es.put("/many", json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}, "done": {"type": "boolean"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for i in 0..JOBS {
        lines.push(json!({"index": {"_index": "many", "_id": i.to_string()}}));
        lines.push(json!({"n": i}));
    }
    es.bulk(&lines).await;
    // Warm every executor (started on first use) before the baseline.
    let (st, _) = es.post("/many/_update_by_query?wait_for_completion=false", json!({"query": {"term": {"n": -1}}})).await;
    assert_eq!(st, 200);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let baseline = os_threads();
    assert!(baseline > 0, "needs /proc/self/task");

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let stop = stop.clone();
        tokio::spawn(async move {
            let (mut peak_threads, mut peak_running) = (0usize, 0usize);
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                peak_threads = peak_threads.max(os_threads());
                peak_running = peak_running.max(elasticsearch_http::pools::background_jobs().stats().running);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            (peak_threads, peak_running)
        })
    };
    let mut submits = tokio::task::JoinSet::new();
    for i in 0..JOBS {
        let es = es.clone();
        submits.spawn(async move {
            // One document per job, then a 200 ms throttle pause (5 docs/s).
            let (st, v) = es.post("/many/_update_by_query?wait_for_completion=false&requests_per_second=5", json!({"query": {"term": {"n": i}}, "script": {"source": "ctx._source.done = true"}})).await;
            assert_eq!(st, 200, "{v}");
            v["task"].as_str().unwrap().to_string()
        });
    }
    let mut ids = Vec::new();
    while let Some(id) = submits.join_next().await {
        ids.push(id.unwrap());
    }
    let mut waits = tokio::task::JoinSet::new();
    for id in ids {
        let es = es.clone();
        waits.spawn(async move {
            let (st, t) = es.get(&format!("/_tasks/{id}?wait_for_completion=true&timeout=120s")).await;
            assert_eq!(st, 200, "{t}");
            assert_eq!(t["completed"], true, "{t}");
            assert_eq!(t["response"]["updated"], 1, "{t}");
        });
    }
    while let Some(r) = waits.join_next().await {
        r.unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let (peak_threads, peak_running) = sampler.await.unwrap();
    let limit = elasticsearch_http::pools::background_jobs().limit();
    assert!(peak_running >= 1 && peak_running <= limit.min(JOBS), "peak running jobs {peak_running}");
    // A thread per job would add ~500; the gate adds none of its own.
    assert!(peak_threads < baseline + 100, "threads grew from {baseline} to {peak_threads}");
    // Each job's permit returns right after its task finished.
    let gate = elasticsearch_http::pools::background_jobs();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while gate.stats() != elasticsearch_http::pools::JobGateStats::default() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(gate.stats(), elasticsearch_http::pools::JobGateStats::default());
    let (_, _) = es.post0("/many/_refresh").await;
    let (_, c) = es.post("/many/_count", json!({"query": {"term": {"done": true}}})).await;
    assert_eq!(c["count"], JOBS, "{c}");
    let (_, l) = es.get("/_tasks?actions=*byquery").await;
    assert_eq!(l, json!({"nodes": {}}), "no task left running");
}
