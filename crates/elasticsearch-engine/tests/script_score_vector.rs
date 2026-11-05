//! Exact `script_score` vector search (so_vector's `dotProduct(params.query,
//! 'titleVector') + 1.0`, alone and under a filter): the bound bulk scan,
//! split into parallel slices on large segments, returns exactly the brute
//! force top k (score, then doc id), and a microbench (ignored) times it.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;

fn vector(i: usize, dims: usize) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<f32> = (0..dims).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }).collect();
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    for a in &mut v { *a /= n; }
    v
}

fn build(node: &Node, docs: usize, dims: usize) {
    node.create_index("ss", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "v": {"type": "dense_vector", "dims": dims, "index": true, "similarity": "dot_product", "index_options": {"type": "flat"}},
            "tag": {"type": "keyword"}}}})).unwrap();
    let svc = node.get_index("ss").unwrap();
    for start in (0..docs).step_by(2_000) {
        let reqs: Vec<WriteReq> = (start..(start + 2_000).min(docs)).map(|i| WriteReq::Index { id: Some(format!("d{i}")),
            source: if i % 3 == 0 { json!({"v": vector(i, dims), "tag": "a"}) } else { json!({"v": vector(i, dims)}) }.to_string(),
            routing: None, cond: Default::default() }).collect();
        assert!(svc.write(reqs, false).iter().all(Result::is_ok));
    }
    svc.refresh().unwrap();
    svc.flush().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
}

fn script_score(node: &Node, q: &[f32], filtered: bool) -> Vec<(String, f64)> {
    let inner = if filtered { json!({"term": {"tag": "a"}}) } else { json!({"match_all": {}}) };
    let body = json!({"size": 10, "_source": false, "query": {"script_score": {"query": inner,
        "script": {"source": "dotProduct(params.query, 'v') + 1.0", "params": {"query": q}}}}});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("ss", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    out["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
}

fn brute(docs: usize, dims: usize, q: &[f32], filtered: bool) -> Vec<(String, f64)> {
    let mut all: Vec<(f32, usize)> = (0..docs).filter(|i| !filtered || i % 3 == 0)
        .map(|i| (rustic_core::vector::dot(q, &vector(i, dims)) + 1.0, i)).collect();
    // Ties: doc order, which is insertion order here (one merged segment).
    all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(10).map(|(s, i)| (format!("d{i}"), f64::from(s))).collect()
}

#[test]
fn bulk_dot_product_script_score_equals_brute_force_with_and_without_a_filter() {
    // Above two parallel slices (2 x 16,384 rows) in one segment.
    let (docs, dims) = (40_000, 8);
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    build(&node, docs, dims);
    for q in [7usize, 1_000_003, 42] {
        let query = vector(q + 500_000, dims);
        for filtered in [false, true] {
            let got = script_score(&node, &query, filtered);
            let want = brute(docs, dims, &query, filtered);
            assert_eq!(got.iter().map(|h| &h.0).collect::<Vec<_>>(), want.iter().map(|h| &h.0).collect::<Vec<_>>(), "filtered {filtered}");
            for ((_, a), (_, b)) in got.iter().zip(&want) { assert!((a - b).abs() < 1e-5, "{a} vs {b}"); }
        }
    }
}

/// `cargo test --release -p elasticsearch-engine --test script_score_vector
/// -- --ignored --nocapture` (`RUSTIC_BENCH_DOCS`, default 200,000, 768-d):
/// p50 of the exact dotProduct script_score, match_all and a 1/3 filter.
#[test]
#[ignore]
fn bench_exact_script_score_768d() {
    let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let dims = 768;
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let t = std::time::Instant::now();
    build(&node, docs, dims);
    eprintln!("built {docs} x {dims}d in {:?}", t.elapsed());
    for filtered in [false, true] {
        let mut times = Vec::new();
        for q in 0..20 {
            let query = vector(10_000_000 + q, dims);
            let t = std::time::Instant::now();
            let hits = script_score(&node, &query, filtered);
            times.push(t.elapsed());
            if q == 0 { assert_eq!(hits.iter().map(|h| h.0.clone()).collect::<Vec<_>>(), brute(docs, dims, &query, filtered).into_iter().map(|h| h.0).collect::<Vec<_>>()); }
        }
        times.sort();
        eprintln!("script_score dotProduct {}: p50 {:?}", if filtered { "filtered 1/3" } else { "match_all   " }, times[times.len() / 2]);
    }
}
