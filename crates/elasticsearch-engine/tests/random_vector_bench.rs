//! Microbench (ignored) shaped like Rally's random_vector track (9.5,
//! defaults): `cargo test --release -p elasticsearch-engine --test
//! random_vector_bench -- --ignored --nocapture`.
//!
//! The track: one shard, index sorted by `partition_id`, a 128-d `emb`
//! field with cosine similarity and `bbq_flat` (1-bit codes, brute force),
//! 1,000 bulks of 1,000 docs of uniform random vectors, each doc in a
//! partition picked by tier weight (100 small, 20 medium, 5 large; about
//! 1.25k, 12.5k and 125k docs each). Searches: `k: 10, num_candidates: 10`
//! with a `term` filter on one partition of a tier, no `rescore_vector`.
//!
//! The index is built once into `RUSTIC_RV_DIR` (default
//! `/tmp/rustic-random-vector`) and reused by later runs (delete the
//! directory to rebuild; `RUSTIC_RV_DOCS` sets the corpus, default 1M).
//! Each run reports, per tier, search p50/p99 over `RUSTIC_RV_QUERIES`
//! (default 300) queries and recall@10 against exact f32 cosine over the
//! partition, plus the load's docs/s when it builds.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const DIMS: usize = 128;
const TIERS: [(&str, usize, usize, usize); 3] = [("small", 100, 1_000, 10_000), ("medium", 20, 10_000, 100_000), ("large", 5, 100_000, 1_000_000)];

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// Doc `i`'s vector (uniform in [0, 1), as the track's `np.random.rand`).
fn vector(i: usize) -> Vec<f32> {
    let mut rng = Rng((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    (0..DIMS).map(|_| rng.unit() as f32).collect()
}

/// Doc `i`'s partition, by tier weight `count * midpoint(range)`.
fn partition(i: usize) -> String {
    let mut rng = Rng((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
    let total: f64 = TIERS.iter().map(|&(_, n, lo, hi)| n as f64 * (lo + hi) as f64 / 2.0).sum();
    let mut r = rng.unit() * total;
    for &(tier, n, lo, hi) in &TIERS {
        let w = n as f64 * (lo + hi) as f64 / 2.0;
        if r < w { return format!("{tier}-{}", rng.next() % n as u64); }
        r -= w;
    }
    format!("large-{}", rng.next() % 5)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) { d += x * y; na += x * x; nb += y * y; }
    d / (na.sqrt() * nb.sqrt())
}

fn percentiles(mut times: Vec<Duration>) -> (Duration, Duration) {
    times.sort();
    (times[times.len() / 2], times[(times.len() * 99 / 100).min(times.len() - 1)])
}

#[test]
#[ignore]
fn random_vector_filtered_bbq_flat() {
    let docs = env("RUSTIC_RV_DOCS", 1_000_000);
    let queries = env("RUSTIC_RV_QUERIES", 300);
    let dir = std::path::PathBuf::from(std::env::var("RUSTIC_RV_DIR").unwrap_or_else(|_| "/tmp/rustic-random-vector".into()));
    let fresh = !dir.join("built").exists();
    if fresh { let _ = std::fs::remove_dir_all(&dir); std::fs::create_dir_all(&dir).unwrap(); }
    let node = Node::open(&dir).unwrap();
    if fresh {
        node.create_index("rv", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index": {"sort": {"field": "partition_id"}}},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "partition_id": {"type": "keyword"}, "gen": {"type": "long"},
                "emb": {"type": "dense_vector", "element_type": "float", "dims": DIMS, "similarity": "cosine", "index_options": {"type": "bbq_flat"}}}}})).unwrap();
        let svc = node.get_index("rv").unwrap();
        let started = Instant::now();
        let mut bulk_times = Vec::new();
        for start in (0..docs).step_by(1_000) {
            let reqs = (start..(start + 1_000).min(docs)).map(|i| WriteReq::Index { id: None,
                source: json!({"@timestamp": 1_700_000_000_000u64 + i as u64, "partition_id": partition(i), "gen": i, "emb": vector(i)}).to_string(),
                routing: None, cond: Default::default() }).collect();
            let t = Instant::now();
            for r in svc.write(reqs, false) { r.unwrap(); }
            bulk_times.push(t.elapsed());
        }
        let load = started.elapsed();
        let t = Instant::now();
        svc.refresh().unwrap();
        let refresh = t.elapsed();
        svc.flush().unwrap();
        let (p50, p99) = percentiles(bulk_times);
        eprintln!("load: {docs} docs in {load:?} ({:.0} docs/s), bulk p50 {p50:?} p99 {p99:?}, refresh {refresh:?}", docs as f64 / load.as_secs_f64());
        std::fs::write(dir.join("built"), docs.to_string()).unwrap();
    }
    let svc = node.get_index("rv").unwrap();
    svc.refresh().unwrap();
    // The server's background maintenance (seal and merge the vector
    // payload chunks); `RUSTIC_RV_OPTIMIZE=0` searches the load's layout.
    if env("RUSTIC_RV_OPTIMIZE", 1) == 1 {
        let t = Instant::now();
        for _ in 0..20 { node.optimize_vectors_all(); }
        eprintln!("vector optimize passes: {:?}", t.elapsed());
    }
    {
        let shard = &svc.local_shards()[0].shard;
        let core = shard.core.read();
        let field = core.field_id("emb").unwrap();
        let layout: Vec<(u64, u64, usize)> = core.vector_store().segments(field).iter()
            .map(|s| { let p = s.points(); (p.first().copied().unwrap_or(0), p.last().copied().unwrap_or(0), p.len()) }).collect();
        let shown: Vec<_> = layout.iter().take(8).collect();
        eprintln!("text segments {}, vector segments {} (first point, last point, points; first 8) {shown:?}", shard.reader_view().segments().len(), layout.len());
    }
    // The partitions' docs (by generator index), for the exact ground truth.
    let mut members: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..docs { members.entry(partition(i)).or_default().push(i); }
    let targets = node.resolve("rv", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    // A hit's generator index is its `gen` doc value (the track has no such
    // field; it only serves the recall check).
    // `RUSTIC_RV_TIER` runs one tier; `RUSTIC_RV_PROFILE=1` skips the recall
    // check (its exact scan would dominate a profile).
    let only = std::env::var("RUSTIC_RV_TIER").ok();
    let profile = std::env::var("RUSTIC_RV_PROFILE").is_ok();
    for &(tier, count, _, _) in TIERS.iter().filter(|t| only.as_deref().is_none_or(|o| o == t.0)) {
        let mut rng = Rng(0xA076_1D64_78BD_642F ^ count as u64);
        let mut times = Vec::with_capacity(queries);
        let mut recall = 0.0;
        let mut sizes = 0usize;
        for q in 0..queries + 20 {
            let p = format!("{tier}-{}", rng.next() % count as u64);
            let query = vector(2_000_000_000 + q * 7 + count);
            let body = json!({"knn": {"field": "emb", "query_vector": query, "k": 10, "num_candidates": 10,
                "filter": {"term": {"partition_id": p}}}, "size": 10, "_source": false, "docvalue_fields": ["gen"]});
            let req = search::parse_request(&body, &HashMap::new()).unwrap();
            let t = Instant::now();
            let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
            let elapsed = t.elapsed();
            if q < 20 { continue; } // warm-up
            times.push(elapsed);
            if profile { continue; }
            let found: std::collections::HashSet<usize> = out["hits"]["hits"].as_array().unwrap().iter()
                .filter_map(|h| h["fields"]["gen"][0].as_u64().map(|g| g as usize))
                .collect();
            let empty = Vec::new();
            let ids = members.get(&p).unwrap_or(&empty);
            sizes += ids.len();
            let mut exact: Vec<(f32, usize)> = ids.iter().map(|&i| (cosine(&query, &vector(i)), i)).collect();
            exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let want: Vec<usize> = exact.iter().take(10).map(|e| e.1).collect();
            recall += if want.is_empty() { 1.0 } else { want.iter().filter(|i| found.contains(i)).count() as f64 / want.len() as f64 };
        }
        let (p50, p99) = percentiles(times);
        eprintln!("{tier:<6} partition (mean {} docs): p50 {:.3} ms p99 {:.3} ms recall@10 {:.3}", sizes / queries.max(1),
            p50.as_secs_f64() * 1e3, p99.as_secs_f64() * 1e3, recall / queries as f64);
    }
}

