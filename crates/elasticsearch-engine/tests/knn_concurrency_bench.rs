//! Microbench (ignored) shaped like Rally's dense_vector (9.5) concurrent
//! phase: `cargo test --release -p elasticsearch-engine --test
//! knn_concurrency_bench -- --ignored --nocapture`.
//!
//! The track: 96-d dot product, int8_hnsw m=32 ef_construction=100, ES's
//! default refresh (1 s) and background merges, 2 shards and 1 replica on
//! 2 nodes. `index-append` loads the corpus with 5,000-doc bulks (2,500 per
//! shard); after the merges finish, in parallel, `index-update-concurrent-
//! with-searches` sends one client's 5,000-doc bulks of the corpus' first 5%
//! again (Rally's `ingest-percentage: 5`; the corpus has no ids, so these are
//! APPENDS of new auto-id documents whose vectors duplicate existing ones,
//! not updates: the race ends at 10.5M docs) while
//! `knn-search-100-1000-concurrent-with-indexing` runs k=100 /
//! num_candidates=1000 on one client.
//!
//! One process is one node: it holds `RUSTIC_BENCH_SHARDS` (default 1)
//! primaries of `bench` and, with `RUSTIC_BENCH_REPLICA=1` (the default),
//! the same shards again in `bench_r`, written after each primary bulk as
//! ES writes the replica after the primary (so a bulk's latency and the
//! node's CPU include both copies, as on each of the track's nodes).
//! Searches read `bench` only (a node searches one copy of each shard).
//!
//! Knobs: `RUSTIC_BENCH_DOCS` (docs per shard, default 1,000,000; the track
//! has 5,000,000), `RUSTIC_BENCH_BULK` (docs per shard per bulk, default
//! 2,500), `RUSTIC_BENCH_LOAD_CLIENTS` (default 4: the load is not measured),
//! `RUSTIC_BENCH_UPDATE_DOCS` (default 5% of the corpus),
//! `RUSTIC_BENCH_UPDATE_MODE` (`append`, the track's, default; `reindex`
//! rewrites existing ids with new vectors). Reports bulk latency p50/p99 and
//! docs/s, kNN 100/1000 p50/p99 during the bulks and alone afterwards (with
//! the segments the bulks left, as `_multiple_segments`), and kNN 10/100.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
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

/// `append`: new auto-id docs with the vectors of corpus docs `ids`;
/// otherwise docs `ids` rewritten with new vectors (`salt`).
fn bulk(ids: impl Iterator<Item = usize>, salt: usize, append: bool) -> Vec<WriteReq> {
    ids.map(|i| WriteReq::Index { id: (!append).then(|| format!("d{i}")), source: json!({"v": vector(i + salt)}).to_string(),
        routing: None, cond: Default::default() }).collect()
}

fn knn(node: &Node, q: usize) -> (usize, Duration) {
    knn_k(node, q, 100, 1000)
}

fn knn_k(node: &Node, q: usize, k: usize, num_candidates: usize) -> (usize, Duration) {
    let body = json!({"knn": {"field": "v", "query_vector": vector(1_000_000_000 + q), "k": k, "num_candidates": num_candidates}, "size": k, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("bench", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let t = Instant::now();
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    (out["hits"]["hits"].as_array().unwrap().len(), t.elapsed())
}

fn percentiles(mut times: Vec<Duration>) -> (Duration, Duration) {
    times.sort();
    (times[times.len() / 2], times[(times.len() * 99 / 100).min(times.len() - 1)])
}

/// `clients` threads each sending `bulk_docs`-doc bulks of the ids `ids`
/// maps out until `count` docs are written, each to `bench` and then (a
/// replica) to `bench_r`; returns docs/s and every bulk's latency.
#[allow(clippy::too_many_arguments)]
fn run_bulks(node: &Node, clients: usize, count: usize, bulk_docs: usize, salt: usize, append: bool, replica: bool,
    ids: impl Fn(usize) -> usize + Sync) -> (f64, Vec<Duration>) {
    let next = AtomicUsize::new(0);
    let latencies = std::sync::Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..clients {
            scope.spawn(|| {
                let svc = node.get_index("bench").unwrap();
                let copy = replica.then(|| node.get_index("bench_r").unwrap());
                loop {
                    let start = next.fetch_add(bulk_docs, Ordering::AcqRel);
                    if start >= count { break; }
                    let end = (start + bulk_docs).min(count);
                    let t = Instant::now();
                    for r in svc.write(bulk((start..end).map(&ids), salt, append), false) { r.unwrap(); }
                    if let Some(copy) = &copy {
                        for r in copy.write(bulk((start..end).map(&ids), salt, append), false) { r.unwrap(); }
                    }
                    latencies.lock().unwrap().push(t.elapsed());
                }
            });
        }
    });
    (count as f64 / started.elapsed().as_secs_f64(), latencies.into_inner().unwrap())
}

/// Wait until no shard of `index` merges (the track's
/// `wait-until-merges-finish` steps), at most `limit`.
fn wait_for_merges(node: &Node, index: &str, limit: Duration) {
    let svc = node.get_index(index).unwrap();
    let started = Instant::now();
    let mut quiet = 0;
    while started.elapsed() < limit && quiet < 3 {
        let merging = svc.local_shards().iter().any(|sh| sh.shard.counters.merge_current.load(Ordering::Relaxed) > 0);
        quiet = if merging { 0 } else { quiet + 1 };
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
#[ignore]
fn rally_dense_vector_concurrent_updates_and_knn() {
    let docs = env("RUSTIC_BENCH_DOCS", 1_000_000);
    let shards = env("RUSTIC_BENCH_SHARDS", 1);
    let corpus = docs * shards;
    let bulk_docs = env("RUSTIC_BENCH_BULK", 2_500) * shards;
    let load_clients = env("RUSTIC_BENCH_LOAD_CLIENTS", 4);
    let updates = env("RUSTIC_BENCH_UPDATE_DOCS", corpus / 20);
    let replica = env("RUSTIC_BENCH_REPLICA", 1) == 1;
    let append = std::env::var("RUSTIC_BENCH_UPDATE_MODE").map_or(true, |m| m != "reindex");
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    // ES defaults otherwise: refresh every second, background merges.
    // `RUSTIC_BENCH_INDEX_TYPE` (default the track's `int8_hnsw`), e.g.
    // `bbq_hnsw`.
    let index_type = std::env::var("RUSTIC_BENCH_INDEX_TYPE").unwrap_or_else(|_| "int8_hnsw".into());
    for index in ["bench", "bench_r"].iter().take(if replica { 2 } else { 1 }) {
        node.create_index(index, &json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0},
            "mappings": {"dynamic": "strict", "properties": {"v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "dot_product",
                "index_options": {"type": index_type, "m": 32, "ef_construction": 100}}}}})).unwrap();
    }
    // Auto ids, as the track's corpus (explicit ids for `reindex` to rewrite).
    let (rate, _) = run_bulks(&node, load_clients, corpus, bulk_docs, 0, append, replica, |i| i);
    eprintln!("index-append: {corpus} docs ({shards} shard(s){}), {load_clients} clients: {rate:.0} docs/s",
        if replica { " + replica" } else { "" });
    let svc = node.get_index("bench").unwrap();
    let t = Instant::now();
    svc.refresh().unwrap();
    if replica { node.get_index("bench_r").unwrap().refresh().unwrap(); }
    eprintln!("refresh-after-index: {:?}", t.elapsed());
    wait_for_merges(&node, "bench", Duration::from_secs(600));
    if replica { wait_for_merges(&node, "bench_r", Duration::from_secs(600)); }
    let (p50, p99) = percentiles((0..200).map(|q| { let (n, t) = knn(&node, q); assert_eq!(n, 100); t }).collect());
    eprintln!("knn 100/1000 before the bulks: p50 {p50:?} p99 {p99:?}");

    let done = AtomicBool::new(false);
    let parallel = Instant::now();
    let ((rate, bulks), during) = std::thread::scope(|scope| {
        let searcher = scope.spawn(|| {
            let mut times = Vec::new();
            let mut q = 0;
            while !done.load(Ordering::Acquire) { let (n, t) = knn(&node, q); assert_eq!(n, 100); times.push(t); q += 1; }
            times
        });
        // `append`: the corpus' first 5% again, as new documents; otherwise
        // existing ids spread over the corpus (updates of every segment).
        let out = if append { run_bulks(&node, 1, updates, bulk_docs, 0, true, replica, |i| i) }
            else { run_bulks(&node, 1, updates, bulk_docs, 7, false, replica, |i| (i * 7_919) % corpus) };
        done.store(true, Ordering::Release);
        (out, searcher.join().unwrap())
    });
    let parallel = parallel.elapsed();
    let searches = during.len();
    let (bulk_p50, bulk_p99) = percentiles(bulks);
    let (p50, p99) = percentiles(during);
    eprintln!("index-update-concurrent-with-searches ({}): {updates} docs, 1 client: {rate:.0} docs/s, bulk p50 {bulk_p50:?} p99 {bulk_p99:?}",
        if append { "append" } else { "reindex" });
    eprintln!("knn 100/1000 concurrent with indexing: p50 {p50:?} p99 {p99:?} ({searches} searches, {:.0} ops/s)",
        searches as f64 / parallel.as_secs_f64());
    svc.refresh().unwrap();
    if replica { node.get_index("bench_r").unwrap().refresh().unwrap(); }
    wait_for_merges(&node, "bench", Duration::from_secs(600));
    for (k, nc) in [(10, 100), (100, 1000)] {
        let (p50, p99) = percentiles((0..500).map(|q| { let (n, t) = knn_k(&node, q, k, nc); assert_eq!(n, k); t }).collect());
        eprintln!("knn {k}/{nc} multiple segments (afterwards): p50 {p50:?} p99 {p99:?}");
    }
}
