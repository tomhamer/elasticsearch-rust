//! `bbq_disk` (ES 9 DiskBBQ) through the search API once a force merge has
//! indexed every row in the fields' shared IVF indexes: rescored scores are the `f32` similarities, recall
//! against brute force at the default visit ratio, `visit_percentage`
//! validation and effect, filtered, nested and multi-shard kNN, and
//! `oversample: 0` estimates.

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

const DIMS: usize = 64;
const DOCS: usize = 4000;
const K: usize = 10;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = ((self.next() as f64 + 1.0) / (1u64 << 31) as f64).min(1.0);
        let u2 = self.next() as f64 / (1u64 << 31) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Embedding-like vectors: 32 centres, each vector its centre plus a
/// variation in 8 latent directions of that centre and a little noise.
fn corpus(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..32).map(|_| (0..DIMS).map(|_| r.unit()).collect()).collect();
    let basis: Vec<Vec<f32>> = (0..32 * 8).map(|_| (0..DIMS).map(|_| r.gaussian() / (DIMS as f32).sqrt()).collect()).collect();
    (0..n).map(|_| {
        let c = r.next() as usize % centres.len();
        let mut v = centres[c].clone();
        for b in &basis[c * 8..c * 8 + 8] {
            let w = 0.6 * r.gaussian();
            for (x, y) in v.iter_mut().zip(b) {
                *x += w * y;
            }
        }
        for x in v.iter_mut() {
            *x += 0.02 * r.gaussian();
        }
        v
    }).collect()
}

fn l2_score(q: &[f32], v: &[f32]) -> f64 {
    1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>())
}

fn chain(e: &EsError) -> String {
    let mut out = format!("{}: {}", e.kind, e.reason);
    let mut cause = e.caused_by.as_deref();
    while let Some(c) = cause {
        out.push_str(&format!("; caused by {}: {}", c.kind, c.reason));
        cause = c.caused_by.as_deref();
    }
    out
}

fn search(node: &Arc<Node>, index: &str, body: Value) -> Result<Value, String> {
    let req = search::parse_request(&body, &HashMap::new()).map_err(|e| chain(&e))?;
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).map_err(|e| chain(&e))?;
    Ok(serde_json::from_str(&out.json).unwrap())
}

fn hits(r: &Value) -> Vec<(usize, f64)> {
    r["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap()[1..].parse().unwrap(), h["_score"].as_f64().unwrap())).collect()
}

struct Fixture {
    node: Arc<Node>,
    _dir: tempfile::TempDir,
    vectors: Vec<Vec<f32>>,
    parts: Vec<[Vec<f32>; 2]>,
    queries: Vec<Vec<f32>>,
}

fn fixture(shards: u32) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let io = json!({"type": "bbq_disk", "cluster_size": 128, "flat_index_threshold": 0});
    node.create_index("vec", &json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0}, "mappings": {"properties": {
        "v": {"type": "dense_vector", "dims": DIMS, "similarity": "l2_norm", "index_options": io},
        "tag": {"type": "keyword"},
        "parts": {"type": "nested", "properties": {"v": {"type": "dense_vector", "dims": DIMS, "similarity": "l2_norm", "index_options": io}}}}}})).unwrap();
    let mut all = corpus(5, DOCS * 3 + 12);
    let queries = all.split_off(DOCS * 3);
    let vectors: Vec<Vec<f32>> = all.drain(..DOCS).collect();
    let parts: Vec<[Vec<f32>; 2]> = (0..DOCS).map(|d| [all[2 * d].clone(), all[2 * d + 1].clone()]).collect();
    let svc = node.get_index("vec").unwrap();
    for chunk in (0..DOCS).collect::<Vec<_>>().chunks(1000) {
        let reqs = chunk.iter().map(|&d| {
            let src = json!({"v": vectors[d], "tag": if d % 2 == 0 { "even" } else { "odd" }, "parts": [{"v": parts[d][0]}, {"v": parts[d][1]}]});
            WriteReq::Index { id: Some(format!("d{d}")), source: src.to_string(), routing: None, cond: Default::default() }
        }).collect();
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }
    // A force merge indexes every row: each field has its shared index,
    // covering every vector segment.
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    for sh in svc.local_shards() {
        let store = sh.shard.core.read().vector_store().clone();
        let segments: Vec<_> = store.segment_ids().into_iter().filter_map(|id| store.segment(id)).collect();
        assert!(!segments.is_empty());
        for s in &segments {
            let state = store.ivf_state(s.field).expect("a shared DiskBBQ index");
            assert!(s.points().iter().all(|&p| state.indexed().contains(p)), "segment {} fully indexed", s.id);
        }
    }
    Fixture { node, _dir: dir, vectors, parts, queries }
}

fn exact(fx: &Fixture, q: &[f32], docs: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut all: Vec<(usize, f64)> = docs.map(|d| (d, l2_score(q, &fx.vectors[d]))).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(K).map(|x| x.0).collect()
}

fn recall(got: &[(usize, f64)], want: &[usize]) -> f64 {
    let want: BTreeSet<usize> = want.iter().copied().collect();
    got.iter().filter(|g| want.contains(&g.0)).count() as f64 / want.len().max(1) as f64
}

fn knn(q: &[f32], extra: Value) -> Value {
    let mut k = json!({"field": "v", "query_vector": q, "k": K, "num_candidates": 15});
    for (key, v) in extra.as_object().unwrap() {
        k[key] = v.clone();
    }
    k
}

fn assert_f32_scores(fx: &Fixture, q: &[f32], got: &[(usize, f64)]) {
    for (d, s) in got {
        let want = l2_score(q, &fx.vectors[*d]);
        assert!((s - want).abs() <= 1e-5 * want.max(1.0), "doc {d}: {s} != f32 {want}");
    }
}

#[test]
fn rescored_recall_and_visit_percentage() {
    let fx = fixture(1);
    let (mut default, mut wide) = (0.0, 0.0);
    for q in &fx.queries {
        let truth = exact(&fx, q, 0..DOCS);
        let top = hits(&search(&fx.node, "vec", json!({"knn": knn(q, json!({})), "size": K, "_source": false})).unwrap());
        assert_eq!(top.len(), K);
        assert_f32_scores(&fx, q, &top);
        default += recall(&top, &truth);
        // The knn query takes visit_percentage too.
        let all = hits(&search(&fx.node, "vec", json!({"query": {"knn": knn(q, json!({"visit_percentage": 100.0}))}, "size": K, "_source": false})).unwrap());
        assert_f32_scores(&fx, q, &all);
        wide += recall(&all, &truth);
        // oversample 0: the IVF estimates are the scores.
        let raw = hits(&search(&fx.node, "vec", json!({"knn": knn(q, json!({"rescore_vector": {"oversample": 0}})), "size": K, "_source": false})).unwrap());
        assert_eq!(raw.len(), K);
        assert!(raw.iter().any(|(d, s)| (s - l2_score(q, &fx.vectors[*d])).abs() > 1e-6), "estimates");
    }
    let n = fx.queries.len() as f64;
    let (default, wide) = (default / n, wide / n);
    eprintln!("BBQ_DISK_RECALL default={default:.3} visit_100={wide:.3}");
    assert!(default >= 0.9, "recall@{K} at the default visit ratio {default}");
    // Visiting every list also scores far lists' spilled copies, whose
    // 1-bit estimates are coarse (as in ES): only a sanity floor here; the
    // monotonicity of recall in the visit ratio is tested in rustic-core.
    assert!(wide >= 0.5, "recall@{K} visiting everything {wide}");
}

#[test]
fn visit_percentage_is_validated() {
    let fx = fixture(1);
    let q = &fx.queries[0];
    for bad in [-1.0, 100.5] {
        let e = search(&fx.node, "vec", json!({"knn": knn(q, json!({"visit_percentage": bad}))})).unwrap_err();
        assert!(e.contains("[visit_percentage] must be between 0 and 100"), "{bad}: {e}");
        let e = search(&fx.node, "vec", json!({"query": {"knn": knn(q, json!({"visit_percentage": bad}))}})).unwrap_err();
        assert!(e.contains("[visit_percentage] must be between 0.0 and 100.0"), "{bad}: {e}");
    }
    for ok in [0.0, 1.0, 100.0] {
        search(&fx.node, "vec", json!({"knn": knn(q, json!({"visit_percentage": ok}))})).unwrap();
    }
}

#[test]
fn filtered_nested_and_multi_shard() {
    for shards in [1, 3] {
        let fx = fixture(shards);
        let mut total = 0.0;
        for q in &fx.queries {
            // Filtered: only even docs, scores rescored.
            let got = hits(&search(&fx.node, "vec", json!({"knn": knn(q, json!({"filter": {"term": {"tag": "even"}}})), "size": K, "_source": false})).unwrap());
            assert!(got.iter().all(|(d, _)| d % 2 == 0));
            assert_f32_scores(&fx, q, &got);
            total += recall(&got, &exact(&fx, q, (0..DOCS).step_by(2)));
            // Nested: each parent scored by its best child.
            let body = json!({"knn": {"field": "parts.v", "query_vector": q, "k": K, "num_candidates": 100}, "size": K, "_source": false});
            let nested = hits(&search(&fx.node, "vec", body).unwrap());
            assert_eq!(nested.len(), K);
            let best = |d: usize| fx.parts[d].iter().map(|v| l2_score(q, v)).fold(f64::MIN, f64::max);
            for (d, s) in &nested {
                assert!((s - best(*d)).abs() <= 1e-5, "nested doc {d}: {s} != best child {}", best(*d));
            }
            let mut truth: Vec<(usize, f64)> = (0..DOCS).map(|d| (d, best(d))).collect();
            truth.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let truth: Vec<usize> = truth.into_iter().take(K).map(|x| x.0).collect();
            assert!(recall(&nested, &truth) >= 0.7, "nested recall {}", recall(&nested, &truth));
            // Multi-shard: one global order of the rescored scores.
            let top = hits(&search(&fx.node, "vec", json!({"knn": knn(q, json!({})), "size": K, "_source": false})).unwrap());
            assert!(top.windows(2).all(|w| w[0].1 >= w[1].1), "{shards} shards: descending");
        }
        let r = total / fx.queries.len() as f64;
        assert!(r >= 0.85, "{shards} shards: filtered recall {r}");
    }
}
