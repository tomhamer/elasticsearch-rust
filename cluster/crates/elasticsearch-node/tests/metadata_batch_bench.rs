//! Benchmark harness (ignored by default): latency of master metadata
//! batches as the cluster's metadata grows, and the scratch copy's rebuild
//! time (what a new master pays once). Run on the build host:
//!
//! ```text
//! BENCH_INDICES=10,1000,5000 BENCH_FIELDS=1000 cargo test -p elasticsearch-node --release \
//!     --test metadata_batch_bench -- --ignored --nocapture
//! ```
//!
//! For each index count N it creates N indices with BENCH_FIELDS keyword
//! fields each, then times (10 samples each, p50 / max): creating an index,
//! adding a field with PUT _mapping, rolling an alias over, and rebuilding
//! a fresh scratch copy from the committed metadata.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A single-node cluster on OS-assigned ports (read back with
/// [`ports::http_port`]); `cluster` tells the rounds' clusters apart.
fn settings(cluster: &str) -> Settings {
    let dir = evidence::scratch_dir(cluster);
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!("n0"));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.type", json!("single-node"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn spawn_node(s: Settings) -> Arc<ClusterNode> {
    let cfg = NodeConfig::from_settings(s).expect("config");
    let (node, http) = ClusterNode::start(cfg).await.expect("node starts");
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let n = node.clone();
    tokio::spawn(async move {
        let _ = elasticsearch_node::front::serve(n, http).await;
    });
    node
}

async fn call(c: &reqwest::Client, http: u16, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{http}{path}"));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let Ok(resp) = r.send().await else { return (0, Value::Null) };
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

fn report(what: &str, n: usize, mut samples: Vec<Duration>) {
    samples.sort();
    let p50 = samples[samples.len() / 2];
    let max = samples[samples.len() - 1];
    println!("BENCH indices={n:>6} {what:<14} p50={:>8.2}ms max={:>8.2}ms", p50.as_secs_f64() * 1e3, max.as_secs_f64() * 1e3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn master_metadata_batch_latency() {
    let counts: Vec<usize> = std::env::var("BENCH_INDICES").unwrap_or_else(|_| "10,1000".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let fields: usize = std::env::var("BENCH_FIELDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let mut properties = serde_json::Map::new();
    for f in 0..fields {
        properties.insert(format!("f{f}"), json!({"type": "keyword"}));
    }
    let body = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mapping.total_fields.limit": fields + 100}, "mappings": {"properties": properties}});
    for (round, n) in counts.into_iter().enumerate() {
        let node = spawn_node(settings(&ports::unique(&format!("metadata-bench-{round}")))).await;
        let http = ports::http_port(&node);
        let c = reqwest::Client::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        while call(&c, http, reqwest::Method::GET, "/_cluster/health?timeout=1s", None).await.0 != 200 {
            assert!(Instant::now() < deadline, "node did not start");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let fill = Instant::now();
        for i in 0..n {
            let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/bulk-{i}"), Some(body.clone())).await;
            assert_eq!(st, 200, "{v}");
        }
        println!("BENCH indices={n:>6} fill            {:>8.2}s", fill.elapsed().as_secs_f64());
        let (st, v) = call(&c, http, reqwest::Method::PUT, "/roll-000001", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "aliases": {"roll": {"is_write_index": true}}}))).await;
        assert_eq!(st, 200, "{v}");
        let mut create = Vec::new();
        let mut mapping = Vec::new();
        let mut rollover = Vec::new();
        for s in 0..10 {
            let t = Instant::now();
            let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/sample-{s}"), Some(body.clone())).await;
            assert_eq!(st, 200, "{v}");
            create.push(t.elapsed());
            let t = Instant::now();
            let (st, v) = call(&c, http, reqwest::Method::PUT, "/bulk-0/_mapping", Some(json!({"properties": {format!("added{s}"): {"type": "long"}}}))).await;
            assert_eq!(st, 200, "{v}");
            mapping.push(t.elapsed());
            let t = Instant::now();
            let (st, v) = call(&c, http, reqwest::Method::POST, "/roll/_rollover", Some(json!({}))).await;
            assert_eq!(st, 200, "{v}");
            rollover.push(t.elapsed());
        }
        report("create index", n, create);
        report("put mapping", n, mapping);
        report("rollover", n, rollover);
        // What a new master pays once: a fresh copy built from the state.
        let dir = evidence::scratch_dir(&format!("metadata-bench-{http}-rebuild"));
        let t = Instant::now();
        let scratch = elasticsearch_node::scratch::ScratchEngine::open(&dir, &node.engine).expect("scratch");
        scratch.sync(&node.state().metadata);
        println!("BENCH indices={n:>6} scratch rebuild {:>8.2}ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
