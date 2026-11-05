//! kNN search forms end to end through the search API on an unquantized
//! `hnsw` field (scores are ES's exact similarities, so every answer can be
//! checked against brute force), on two shards with several segments and
//! memtables each:
//! - pre-filtered kNN: a broad filter keeps the recall of the exact
//!   filtered top k, a selective one (no more matches than
//!   `num_candidates`) is the exact filtered top k;
//! - nested kNN with `inner_hits`: parents ranked by their best child,
//!   inner hits are that parent's children best first, the first scoring
//!   the parent;
//! - `collapse` on a kNN search: one hit per group, each its group's best
//!   doc among the kNN top k, best first;
//! - an `rrf` retriever over two kNN retrievers: the fused scores and order
//!   are ES's RRF of the two ranked lists.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

const DIMS: usize = 24;
const DOCS: usize = 1500;
const K: usize = 10;
const GROUPS: usize = 30;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
    }
}

fn corpus(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..30).map(|_| (0..DIMS).map(|_| r.unit()).collect()).collect();
    (0..n).map(|i| {
        let c = &centres[(r.next() as usize + i) % centres.len()];
        c.iter().map(|x| x + 0.3 * r.unit()).collect()
    }).collect()
}

fn l2(q: &[f32], v: &[f32]) -> f64 {
    1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>())
}

struct Fixture {
    node: Arc<Node>,
    _dir: tempfile::TempDir,
    v: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    parts: Vec<[Vec<f32>; 3]>,
}

fn tag(d: usize) -> String {
    format!("t{}", d % 5)
}

fn group(d: usize) -> String {
    format!("g{}", d % GROUPS)
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let vec = |sim: &str| json!({"type": "dense_vector", "dims": DIMS, "index": true, "similarity": sim, "index_options": {"type": "hnsw"}});
    node.create_index("forms", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"v": vec("l2_norm"), "w": vec("l2_norm"), "tag": {"type": "keyword"}, "g": {"type": "keyword"},
            "n": {"type": "long"}, "parts": {"type": "nested", "properties": {"pv": vec("l2_norm"), "i": {"type": "long"}}}}}})).unwrap();
    let v = corpus(1, DOCS);
    let w = corpus(2, DOCS);
    let flat = corpus(3, DOCS * 3);
    let parts: Vec<[Vec<f32>; 3]> = (0..DOCS).map(|d| [flat[3 * d].clone(), flat[3 * d + 1].clone(), flat[3 * d + 2].clone()]).collect();
    let svc = node.get_index("forms").unwrap();
    for (round, chunk) in (0..DOCS).collect::<Vec<_>>().chunks(250).enumerate() {
        let reqs = chunk.iter().map(|&d| WriteReq::Index { id: Some(format!("d{d}")), source: json!({
            "v": v[d], "w": w[d], "tag": tag(d), "g": group(d), "n": d,
            "parts": [{"pv": parts[d][0], "i": 0}, {"pv": parts[d][1], "i": 1}, {"pv": parts[d][2], "i": 2}]}).to_string(),
            routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        if round % 2 == 1 {
            svc.flush().unwrap();
        }
    }
    Fixture { node, _dir: dir, v, w, parts }
}

fn search(fx: &Fixture, body: Value) -> Value {
    let req = search::parse_request(&body, &HashMap::new()).unwrap_or_else(|e| panic!("{body}: {}", e.reason));
    let targets = fx.node.resolve("forms", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {}", e.reason));
    serde_json::from_str(&out.json).unwrap()
}

fn doc(h: &Value) -> usize {
    h["_id"].as_str().unwrap()[1..].parse().unwrap()
}

fn hits(r: &Value) -> Vec<(usize, f64)> {
    r["hits"]["hits"].as_array().unwrap().iter().map(|h| (doc(h), h["_score"].as_f64().unwrap())).collect()
}

fn exact(q: &[f32], rows: &[Vec<f32>], docs: impl Iterator<Item = usize>, k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = docs.map(|d| (d, l2(q, &rows[d]))).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn recall(got: &[(usize, f64)], want: &[(usize, f64)]) -> f64 {
    let want: BTreeSet<usize> = want.iter().map(|w| w.0).collect();
    got.iter().filter(|g| want.contains(&g.0)).count() as f64 / want.len().max(1) as f64
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * b.abs().max(1.0)
}

#[test]
fn prefiltered_knn_matches_the_exact_filtered_top_k() {
    let fx = fixture();
    let queries = corpus(11, 8);
    let mut broad = 0.0;
    for q in &queries {
        // Broad: one doc in five.
        let body = json!({"knn": {"field": "v", "query_vector": q, "k": K, "num_candidates": 100, "filter": {"term": {"tag": "t2"}}}, "size": K, "_source": false});
        let got = hits(&search(&fx, body));
        assert_eq!(got.len(), K);
        assert!(got.iter().all(|(d, _)| tag(*d) == "t2"), "the filter was not applied: {got:?}");
        assert!(got.iter().all(|(d, s)| close(*s, l2(q, &fx.v[*d]))), "scores are the exact similarities");
        broad += recall(&got, &exact(q, &fx.v, (0..DOCS).filter(|d| tag(*d) == "t2"), K));
        // Selective: 50 matches, fewer than num_candidates: the exact answer.
        let body = json!({"knn": {"field": "v", "query_vector": q, "k": K, "num_candidates": 100,
            "filter": {"bool": {"filter": [{"term": {"g": "g7"}}, {"range": {"n": {"lt": 1500}}}]}}}, "size": K, "_source": false});
        let got = hits(&search(&fx, body));
        let want = exact(q, &fx.v, (0..DOCS).filter(|d| group(*d) == "g7"), K);
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), want.iter().map(|w| w.0).collect::<Vec<_>>(), "selective filter: not the exact top k");
        // The knn query form with the same filter agrees.
        let body = json!({"query": {"knn": {"field": "v", "query_vector": q, "k": K, "num_candidates": 100, "filter": {"term": {"g": "g7"}}}}, "size": K, "_source": false});
        let got = hits(&search(&fx, body));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), want.iter().map(|w| w.0).collect::<Vec<_>>(), "knn query with a selective filter");
    }
    let broad = broad / queries.len() as f64;
    assert!(broad >= 0.9, "broad pre-filter recall {broad}");
}

#[test]
fn nested_knn_inner_hits_are_the_parents_best_children() {
    let fx = fixture();
    let queries = corpus(12, 6);
    let mut total = 0.0;
    for q in &queries {
        let body = json!({"knn": {"field": "parts.pv", "query_vector": q, "k": K, "num_candidates": 100,
            "inner_hits": {"size": 2, "_source": false}}, "size": K, "_source": false});
        let r = search(&fx, body);
        let got = hits(&r);
        assert_eq!(got.len(), K);
        let child = |d: usize, i: usize| l2(q, &fx.parts[d][i]);
        let best = |d: usize| (0..3).map(|i| child(d, i)).fold(f64::MIN, f64::max);
        for h in r["hits"]["hits"].as_array().unwrap() {
            let (d, s) = (doc(h), h["_score"].as_f64().unwrap());
            assert!(close(s, best(d)), "doc {d}: score {s} != best child {}", best(d));
            let inner = h["inner_hits"]["parts.pv"]["hits"]["hits"].as_array()
                .or_else(|| h["inner_hits"].as_object().and_then(|m| m.values().next()).and_then(|v| v["hits"]["hits"].as_array()))
                .unwrap_or_else(|| panic!("doc {d}: no inner hits: {h}"));
            assert_eq!(inner.len(), 2, "doc {d}: inner hits size");
            let mut want: Vec<(usize, f64)> = (0..3).map(|i| (i, child(d, i))).collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (ih, (i, score)) in inner.iter().zip(&want) {
                assert_eq!(ih["_id"].as_str(), Some(format!("d{d}").as_str()), "inner hit of another parent");
                assert_eq!(ih["_nested"]["field"], "parts");
                assert_eq!(ih["_nested"]["offset"].as_u64(), Some(*i as u64), "doc {d}: inner hits not best child first: {inner:?}");
                assert!(close(ih["_score"].as_f64().unwrap(), *score), "doc {d}: inner hit score");
            }
            assert!(close(inner[0]["_score"].as_f64().unwrap(), s), "doc {d}: the first inner hit scores the parent");
        }
        let mut truth: Vec<(usize, f64)> = (0..DOCS).map(|d| (d, best(d))).collect();
        truth.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        truth.truncate(K);
        total += recall(&got, &truth);
    }
    let total = total / queries.len() as f64;
    assert!(total >= 0.9, "nested kNN recall {total}");
}

/// ES rewrites a nested kNN for `inner_hits` into an exact kNN over the
/// children of every parent hit, so a parent matched only by another clause
/// (or ranked beyond `k`) still lists its best children, exactly scored.
#[test]
fn nested_knn_inner_hits_cover_parents_from_other_clauses() {
    let fx = fixture();
    let q = &corpus(14, 1)[0];
    let child = |d: usize, i: usize| l2(q, &fx.parts[d][i]);
    let extra = [3usize, 777, 1201];
    let ih = |h: &Value| -> Vec<Value> {
        h["inner_hits"].as_object().and_then(|m| m.values().next()).and_then(|v| v["hits"]["hits"].as_array()).cloned().unwrap_or_default()
    };
    let check = |r: &Value, what: &str| {
        let all = r["hits"]["hits"].as_array().unwrap();
        for &d in &extra {
            let h = all.iter().find(|h| doc(h) == d).unwrap_or_else(|| panic!("{what}: doc {d} not returned"));
            let inner = ih(h);
            assert_eq!(inner.len(), 2, "{what}: doc {d} matched by the term clause has no kNN inner hits: {h}");
            let mut want: Vec<(usize, f64)> = (0..3).map(|i| (i, child(d, i))).collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (hit, (i, score)) in inner.iter().zip(&want) {
                assert_eq!(hit["_nested"]["offset"].as_u64(), Some(*i as u64), "{what}: doc {d}: not best child first");
                assert!(close(hit["_score"].as_f64().unwrap(), *score), "{what}: doc {d}: inner hit score not exact");
            }
        }
        // Every returned parent has inner hits, kNN winner or not.
        for h in all {
            assert_eq!(ih(h).len(), 2, "{what}: doc {} lacks inner hits", doc(h));
        }
    };
    // Top-level knn section with inner_hits next to a query.
    let body = json!({"knn": {"field": "parts.pv", "query_vector": q, "k": 3, "num_candidates": 50, "inner_hits": {"size": 2, "_source": false}},
        "query": {"terms": {"n": extra}}, "size": 20, "_source": false});
    check(&search(&fx, body), "knn section");
    // A nested knn query with inner_hits, OR-ed with another clause.
    let body = json!({"query": {"bool": {"should": [
        {"nested": {"path": "parts", "query": {"knn": {"field": "parts.pv", "query_vector": q, "k": 3, "num_candidates": 50}}, "inner_hits": {"size": 2, "_source": false}}},
        {"terms": {"n": extra}}]}}, "size": 20, "_source": false});
    check(&search(&fx, body), "nested knn query");
}

#[test]
fn collapse_on_knn_keeps_each_groups_best_hit() {
    let fx = fixture();
    let queries = corpus(13, 6);
    for q in &queries {
        let knn = json!({"field": "v", "query_vector": q, "k": 40, "num_candidates": 200});
        let plain = hits(&search(&fx, json!({"knn": knn, "size": 40, "_source": false})));
        assert_eq!(plain.len(), 40);
        let collapsed = search(&fx, json!({"knn": knn, "size": 40, "_source": false, "collapse": {"field": "g"}}));
        let got = hits(&collapsed);
        // ES: the kNN top k first, then one hit per group, best first.
        let mut best: BTreeMap<String, (usize, f64)> = BTreeMap::new();
        for &(d, s) in &plain {
            best.entry(group(d)).or_insert((d, s));
        }
        let mut want: Vec<(usize, f64)> = best.values().copied().collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), want.iter().map(|w| w.0).collect::<Vec<_>>(), "collapsed kNN hits");
        for h in collapsed["hits"]["hits"].as_array().unwrap() {
            assert_eq!(h["fields"]["g"][0].as_str(), Some(group(doc(h)).as_str()), "collapse key field");
        }
        let groups: BTreeSet<String> = got.iter().map(|g| group(g.0)).collect();
        assert_eq!(groups.len(), got.len(), "a group twice");
    }
}

#[test]
fn rrf_retriever_fuses_two_knn_rankings_as_es_does() {
    let fx = fixture();
    let queries = corpus(14, 6);
    let (window, constant) = (30usize, 60.0f64);
    for q in &queries {
        let knn = |field: &str| json!({"field": field, "query_vector": q, "k": window, "num_candidates": 150});
        let v = hits(&search(&fx, json!({"knn": knn("v"), "size": window, "_source": false})));
        let w = hits(&search(&fx, json!({"knn": knn("w"), "size": window, "_source": false})));
        assert!(v.iter().all(|(d, s)| close(*s, l2(q, &fx.v[*d]))));
        assert!(w.iter().all(|(d, s)| close(*s, l2(q, &fx.w[*d]))));
        // ES RRFRetrieverBuilder: sum over retrievers of 1 / (rank_constant + rank), rank from 1.
        let mut fused: BTreeMap<usize, f64> = BTreeMap::new();
        for list in [&v, &w] {
            for (rank, (d, _)) in list.iter().enumerate() {
                *fused.entry(*d).or_default() += 1.0 / (constant + rank as f64 + 1.0);
            }
        }
        let body = json!({"retriever": {"rrf": {"retrievers": [{"knn": knn("v")}, {"knn": knn("w")}],
            "rank_window_size": window, "rank_constant": constant as u64}}, "size": K, "_source": false});
        let got = hits(&search(&fx, body));
        assert_eq!(got.len(), K);
        for (d, s) in &got {
            let want = fused.get(d).unwrap_or_else(|| panic!("doc {d} is in neither ranking"));
            assert!((s - want).abs() <= 1e-6, "doc {d}: rrf score {s} != {want}");
        }
        assert!(got.windows(2).all(|p| p[0].1 >= p[1].1), "rrf scores descending");
        // The top K by fused score (ties may order either way).
        let mut want: Vec<f64> = fused.values().copied().collect();
        want.sort_by(|a, b| b.total_cmp(a));
        for (i, (_, s)) in got.iter().enumerate() {
            assert!((s - want[i]).abs() <= 1e-6, "rank {i}: {s} != {}", want[i]);
        }
    }
}
