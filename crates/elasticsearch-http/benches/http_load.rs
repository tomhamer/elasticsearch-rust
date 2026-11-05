//! HTTP load benchmark: bulk indexing throughput and search QPS / latency
//! for `match`, `bool`, `terms` agg and `date_histogram`, driven over real
//! HTTP so the same harness can target a real Elasticsearch for comparison.
//!
//! ```text
//! cargo bench -p elasticsearch-http --bench http_load
//! # against another server (e.g. real ES 8.x on :9201):
//! ES_BENCH_URL=http://127.0.0.1:9201 cargo bench -p elasticsearch-http --bench http_load
//! # knobs: ES_BENCH_DOCS (default 200000), ES_BENCH_BATCH (1000),
//! #        ES_BENCH_QUERIES (2000 per query type), ES_BENCH_CONCURRENCY (16)
//! ```
//! Prints one JSON line per measurement (docs/s, qps, p50/p99 ms).

use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const WORDS: &[&str] = &[
    "rust", "search", "engine", "lucene", "elastic", "fast", "index", "query", "vector", "token", "shard", "segment", "merge", "score", "field",
    "document", "analyzer", "phrase", "match", "filter", "range", "bucket", "metric", "cluster", "node", "replica", "translog", "commit", "flush",
    "refresh",
];
const TAGS: &[&str] = &["red", "green", "blue", "yellow", "black", "white", "purple", "orange"];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn pick<'a>(&mut self, v: &[&'a str]) -> &'a str {
        v[(self.next() % v.len() as u64) as usize]
    }
}

fn doc(r: &mut Rng, i: usize) -> String {
    let title: Vec<&str> = (0..4).map(|_| r.pick(WORDS)).collect();
    let body: Vec<&str> = (0..30).map(|_| r.pick(WORDS)).collect();
    let ts = 1_600_000_000_000u64 + (r.next() % (365 * 86_400_000));
    format!(
        r#"{{"title":"{}","body":"{}","tag":"{}","price":{:.2},"count":{},"ts":{}}}"#,
        title.join(" "),
        body.join(" "),
        r.pick(TAGS),
        (r.next() % 100_000) as f64 / 100.0,
        i % 1000,
        ts
    )
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

async fn run_queries(client: &reqwest::Client, url: &str, name: &str, bodies: Vec<String>, conc: usize) {
    let bodies = Arc::new(bodies);
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..conc {
        let (c, u, b, n) = (client.clone(), format!("{url}/bench/_search"), bodies.clone(), next.clone());
        tasks.push(tokio::spawn(async move {
            let mut lat = Vec::new();
            loop {
                let i = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= b.len() {
                    break;
                }
                let t = Instant::now();
                let r = c.post(&u).header("content-type", "application/json").body(b[i].clone()).send().await.expect("search");
                assert!(r.status().is_success(), "{} {}", r.status(), r.text().await.unwrap_or_default());
                let _ = r.bytes().await;
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            lat
        }));
    }
    let mut all = Vec::new();
    for t in tasks {
        all.extend(t.await.unwrap());
    }
    let secs = start.elapsed().as_secs_f64();
    all.sort_by(|a, b| a.total_cmp(b));
    println!(
        "{{\"bench\":\"search_{name}\",\"queries\":{},\"qps\":{:.0},\"p50_ms\":{:.3},\"p99_ms\":{:.3}}}",
        all.len(),
        all.len() as f64 / secs,
        pct(&all, 0.5),
        pct(&all, 0.99)
    );
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let n_docs = env_usize("ES_BENCH_DOCS", 200_000);
        let batch = env_usize("ES_BENCH_BATCH", 1000);
        let n_q = env_usize("ES_BENCH_QUERIES", 2000);
        let conc = env_usize("ES_BENCH_CONCURRENCY", 16);
        let tmp = tempfile::tempdir().unwrap();
        let url = match std::env::var("ES_BENCH_URL") {
            Ok(u) => u,
            Err(_) => {
                let node = elasticsearch_engine::node::Node::open(tmp.path()).unwrap();
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let app = elasticsearch_http::router(node);
                tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                format!("http://{addr}")
            }
        };
        let client = reqwest::Client::builder().pool_max_idle_per_host(64).timeout(Duration::from_secs(300)).build().unwrap();
        let _ = client.delete(format!("{url}/bench")).send().await;
        let r = client
            .put(format!("{url}/bench"))
            .header("content-type", "application/json")
            .body(r#"{"settings":{"number_of_shards":1,"number_of_replicas":0,"refresh_interval":"1s"},"mappings":{"properties":{"title":{"type":"text"},"body":{"type":"text"},"tag":{"type":"keyword"},"price":{"type":"double"},"count":{"type":"long"},"ts":{"type":"date"}}}}"#)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "{}", r.text().await.unwrap());

        // Bulk indexing (4 concurrent bulk requests).
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut payloads = Vec::new();
        let mut i = 0;
        while i < n_docs {
            let mut s = String::new();
            for j in i..(i + batch).min(n_docs) {
                s.push_str("{\"index\":{}}\n");
                s.push_str(&doc(&mut rng, j));
                s.push('\n');
            }
            payloads.push(s);
            i += batch;
        }
        let payloads = Arc::new(payloads);
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let start = Instant::now();
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let (c, u, p, n) = (client.clone(), format!("{url}/bench/_bulk"), payloads.clone(), next.clone());
            tasks.push(tokio::spawn(async move {
                loop {
                    let k = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= p.len() {
                        break;
                    }
                    let r = c.post(&u).header("content-type", "application/x-ndjson").body(p[k].clone()).send().await.expect("bulk");
                    let v: serde_json::Value = r.json().await.unwrap();
                    assert_eq!(v["errors"], false, "bulk errors");
                }
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let secs = start.elapsed().as_secs_f64();
        println!("{{\"bench\":\"bulk_index\",\"docs\":{n_docs},\"secs\":{secs:.3},\"docs_per_sec\":{:.0}}}", n_docs as f64 / secs);
        client.post(format!("{url}/bench/_refresh")).send().await.unwrap();

        let mut rng = Rng(42);
        let mk = |f: &mut dyn FnMut() -> String| (0..n_q).map(|_| f()).collect::<Vec<_>>();
        let matchq = mk(&mut || format!(r#"{{"query":{{"match":{{"body":"{} {}"}}}}}}"#, rng.pick(WORDS), rng.pick(WORDS)));
        let mut rng = Rng(43);
        let boolq = mk(&mut || {
            format!(
                r#"{{"query":{{"bool":{{"must":[{{"match":{{"title":"{}"}}}}],"filter":[{{"term":{{"tag":"{}"}}}},{{"range":{{"price":{{"gte":100,"lte":600}}}}}}]}}}}}}"#,
                rng.pick(WORDS),
                rng.pick(TAGS)
            )
        });
        let terms = (0..n_q / 4).map(|_| r#"{"size":0,"aggs":{"t":{"terms":{"field":"tag","size":10}}}}"#.to_string()).collect();
        let dh = (0..n_q / 4)
            .map(|_| r#"{"size":0,"aggs":{"h":{"date_histogram":{"field":"ts","calendar_interval":"month"},"aggs":{"p":{"avg":{"field":"price"}}}}}}"#.to_string())
            .collect();
        run_queries(&client, &url, "match", matchq, conc).await;
        run_queries(&client, &url, "bool", boolq, conc).await;
        run_queries(&client, &url, "terms_agg", terms, conc).await;
        run_queries(&client, &url, "date_histogram", dh, conc).await;
    });
}
