//! Microbench (ignored): write contention at scale on limited cores. Run
//! the test binary under `taskset -c <cores>` (the node's pools size
//! themselves to the cores it may use).
//!
//! The index is dense_vector-shaped (96-d dot product, int8_hnsw m 32 /
//! ef_construction 100, one shard), built once with `RUSTIC_CB_DOCS`
//! (default 5,000,000) docs into `RUSTIC_CB_DIR` (default
//! `/tmp/rustic-contention`) and copied to a scratch directory for each run,
//! so every run starts from the same index. A run sends `RUSTIC_CB_CLIENTS`
//! (default 16) bulk clients appending 1,000-doc bulks of new docs while
//! `RUSTIC_CB_SEARCHERS` (default 4) clients search back to back (kNN
//! 10/100), for `RUSTIC_CB_SECS` (default 60) seconds, after a 10 s search-
//! only baseline. Reports append docs/s and bulk p50/p99, and search
//! p50/p99/max and throughput, without and with the bulks. Set
//! `RUSTIC_PREPARE_SEARCH_SHARE` to compare preparation shares.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const DIMS: usize = 96;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn vector(i: usize) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<f32> = (0..DIMS).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }).collect();
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    for a in &mut v { *a /= n; }
    v
}

fn bulk(range: std::ops::Range<usize>) -> Vec<WriteReq> {
    range.map(|i| WriteReq::Index { id: None, source: json!({"v": vector(i)}).to_string(), routing: None, cond: Default::default() }).collect()
}

fn knn(node: &Node, q: usize) -> Duration {
    let body = json!({"knn": {"field": "v", "query_vector": vector(1_000_000_000 + q), "k": 10, "num_candidates": 100}, "size": 10, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("cb", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let t = Instant::now();
    let out = search::execute(&targets, &req, &opts).unwrap();
    assert!(!out.json.is_empty());
    t.elapsed()
}

fn stats(mut times: Vec<Duration>) -> String {
    if times.is_empty() { return "none".into(); }
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    format!("p50 {:.2} ms p99 {:.2} ms max {:.2} ms (n {})", ms(times[times.len() / 2]), ms(times[(times.len() * 99 / 100).min(times.len() - 1)]), ms(*times.last().unwrap()), times.len())
}

/// Search back to back on `searchers` threads until `stop`.
fn searching(node: &Node, searchers: usize, stop: &AtomicBool) -> Vec<Duration> {
    let all = std::sync::Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..searchers {
            scope.spawn(|| {
                let mut mine = Vec::new();
                while !stop.load(Ordering::Acquire) { mine.push(knn(node, next.fetch_add(1, Ordering::Relaxed))); }
                all.lock().unwrap().extend(mine);
            });
        }
    });
    all.into_inner().unwrap()
}

#[test]
#[ignore]
fn contention_bulk_clients_with_knn_on_limited_cores() {
    let docs = env("RUSTIC_CB_DOCS", 5_000_000);
    let clients = env("RUSTIC_CB_CLIENTS", 16);
    let searchers = env("RUSTIC_CB_SEARCHERS", 4);
    let secs = env("RUSTIC_CB_SECS", 60);
    let pristine = std::path::PathBuf::from(std::env::var("RUSTIC_CB_DIR").unwrap_or_else(|_| "/tmp/rustic-contention".into()));
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    if !pristine.join("built").exists() {
        let _ = std::fs::remove_dir_all(&pristine);
        std::fs::create_dir_all(&pristine).unwrap();
        let node = Node::open(&pristine).unwrap();
        node.create_index("cb", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "dot_product",
                "index_options": {"type": "int8_hnsw", "m": 32, "ef_construction": 100}}}}})).unwrap();
        let svc = node.get_index("cb").unwrap();
        let next = AtomicUsize::new(0);
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| loop {
                    let start = next.fetch_add(5_000, Ordering::AcqRel);
                    if start >= docs { break; }
                    for r in svc.write(bulk(start..(start + 5_000).min(docs)), false) { r.unwrap(); }
                });
            }
        });
        svc.refresh().unwrap();
        svc.flush().unwrap();
        eprintln!("built {docs} docs in {:?}", started.elapsed());
        drop(svc);
        drop(node);
        std::fs::write(pristine.join("built"), docs.to_string()).unwrap();
    }
    // A fresh copy of the built index for this run.
    let work = tempfile::Builder::new().prefix("rustic-contention-run").tempdir_in(pristine.parent().unwrap_or(std::path::Path::new("/tmp"))).unwrap();
    let status = std::process::Command::new("cp").arg("-a").arg(format!("{}/.", pristine.display())).arg(work.path()).status().unwrap();
    assert!(status.success());
    let node = Node::open(work.path()).unwrap();
    let svc = node.get_index("cb").unwrap();
    svc.refresh().unwrap();
    for q in 0..200 { knn(&node, q); }
    // Baseline: searches alone.
    let stop = AtomicBool::new(false);
    let alone = std::thread::scope(|scope| {
        let s = scope.spawn(|| searching(&node, searchers, &stop));
        std::thread::sleep(Duration::from_secs(10));
        stop.store(true, Ordering::Release);
        s.join().unwrap()
    });
    // Contention: bulks and searches together.
    let stop = AtomicBool::new(false);
    let written = AtomicUsize::new(0);
    let next = AtomicUsize::new(docs);
    let bulk_times = std::sync::Mutex::new(Vec::new());
    let started = Instant::now();
    let during = std::thread::scope(|scope| {
        for _ in 0..clients {
            scope.spawn(|| {
                let svc = node.get_index("cb").unwrap();
                while !stop.load(Ordering::Acquire) {
                    let start = next.fetch_add(1_000, Ordering::AcqRel);
                    let t = Instant::now();
                    for r in svc.write(bulk(start..start + 1_000), false) { r.unwrap(); }
                    bulk_times.lock().unwrap().push(t.elapsed());
                    written.fetch_add(1_000, Ordering::Relaxed);
                }
            });
        }
        let s = scope.spawn(|| searching(&node, searchers, &stop));
        std::thread::sleep(Duration::from_secs(secs as u64));
        stop.store(true, Ordering::Release);
        s.join().unwrap()
    });
    let elapsed = started.elapsed();
    let alone_qps = alone.len() as f64 / 10.0;
    let during_qps = during.len() as f64 / secs as f64;
    eprintln!("cores {cores}, {docs} docs, {clients} bulk clients, {searchers} searchers, share {}",
        std::env::var("RUSTIC_PREPARE_SEARCH_SHARE").unwrap_or_else(|_| "default".into()));
    eprintln!("search alone:  {} at {alone_qps:.0} qps", stats(alone));
    eprintln!("search during: {} at {during_qps:.0} qps", stats(during));
    eprintln!("append: {:.0} docs/s, bulk {}", written.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64(), stats(bulk_times.into_inner().unwrap()));
}
