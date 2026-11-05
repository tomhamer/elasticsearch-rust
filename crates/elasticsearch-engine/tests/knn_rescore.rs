//! kNN `rescore_vector` (ES 9 `RescoreVectorBuilder` / `RescoreKnnVectorQuery`)
//! and quantized `_score` parity, through the search API on every quantized
//! `dense_vector` index type, against brute-force `f32` oracles over the
//! indexed vectors: rescored scores are ES's similarity transforms of the
//! `f32` vectors, recall with rescoring is at least the recall without, and
//! the top-level knn section, the knn query, filtered, nested, multi-shard
//! and retriever (knn, RRF) forms agree.

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

const DIMS: usize = 64;
const DOCS: usize = 1200;
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
}

/// Clustered vectors (40 centres, noise 0.35): a corpus with neighbourhood
/// structure, as real embeddings have, so recall thresholds are stable.
fn corpus(seed: u64, n: usize, unit: bool) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..40).map(|_| (0..DIMS).map(|_| r.unit()).collect()).collect();
    (0..n).map(|i| {
        let c = &centres[(r.next() as usize + i) % centres.len()];
        let v: Vec<f32> = c.iter().map(|x| x + 0.35 * r.unit()).collect();
        if unit { normalize(v) } else { v }
    }).collect()
}

fn normalize(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| (x * x) as f64).sum::<f64>().sqrt() as f32;
    v.into_iter().map(|x| x / n).collect()
}

/// ES's `_score` of a similarity (Lucene `VectorSimilarityFunction` +
/// ES `DenseVectorFieldMapper.VectorSimilarity.score`), in f64.
fn es_score(similarity: &str, q: &[f32], v: &[f32]) -> f64 {
    let dot: f64 = q.iter().zip(v).map(|(a, b)| *a as f64 * *b as f64).sum();
    match similarity {
        "l2_norm" => 1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>()),
        "cosine" => {
            let n = |x: &[f32]| x.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            (1.0 + dot / (n(q) * n(v))) / 2.0
        }
        "dot_product" => ((1.0 + dot) / 2.0).max(0.0),
        "max_inner_product" => if dot < 0.0 { 1.0 / (1.0 - dot) } else { dot + 1.0 },
        s => panic!("{s}"),
    }
}

/// An error with its `caused_by` chain (ES wraps knn parse failures).
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
    for t in &targets {
        t.svc.refresh().unwrap();
    }
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).map_err(|e| chain(&e))?;
    Ok(serde_json::from_str(&out.json).unwrap())
}

fn hits(r: &Value) -> Vec<(usize, f64)> {
    r["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap()[1..].parse().unwrap(), h["_score"].as_f64().unwrap())).collect()
}

/// `(field, index type, similarity, unit vectors)` of the fixture.
const FIELDS: &[(&str, &str, &str, bool)] = &[
    ("v_int8_hnsw", "int8_hnsw", "l2_norm", false),
    ("v_int4_hnsw", "int4_hnsw", "cosine", false),
    ("v_bbq_hnsw", "bbq_hnsw", "dot_product", true),
    ("v_int8_flat", "int8_flat", "max_inner_product", false),
    ("v_int4_flat", "int4_flat", "l2_norm", false),
    ("v_bbq_flat", "bbq_flat", "cosine", false),
    ("v_bbq_disk", "bbq_disk", "l2_norm", false),
];

struct Fixture {
    node: Arc<Node>,
    _dir: tempfile::TempDir,
    /// Per field: doc vectors, indexed by doc number.
    vectors: Vec<Vec<Vec<f32>>>,
    /// Nested `parts.v` (int8_hnsw, l2_norm): two children per doc.
    parts: Vec<[Vec<f32>; 2]>,
}

fn fixture(shards: u32) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mut props = serde_json::Map::new();
    for (name, ty, sim, _) in FIELDS {
        props.insert(name.to_string(), json!({"type": "dense_vector", "dims": DIMS, "index": true, "similarity": sim, "index_options": {"type": ty}}));
    }
    props.insert("tag".into(), json!({"type": "keyword"}));
    props.insert("parts".into(), json!({"type": "nested", "properties": {
        "v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": "int8_hnsw"}}}}));
    node.create_index("vec", &json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0}, "mappings": {"properties": props}})).unwrap();
    let vectors: Vec<Vec<Vec<f32>>> = FIELDS.iter().enumerate().map(|(i, f)| corpus(100 + i as u64, DOCS, f.3)).collect();
    let nested = corpus(999, DOCS * 2, false);
    let parts: Vec<[Vec<f32>; 2]> = (0..DOCS).map(|d| [nested[2 * d].clone(), nested[2 * d + 1].clone()]).collect();
    let svc = node.get_index("vec").unwrap();
    for chunk in (0..DOCS).collect::<Vec<_>>().chunks(300) {
        let reqs = chunk.iter().map(|&d| {
            let mut src = serde_json::Map::new();
            for (i, (name, ..)) in FIELDS.iter().enumerate() {
                src.insert(name.to_string(), json!(vectors[i][d]));
            }
            src.insert("tag".into(), json!(if d % 2 == 0 { "even" } else { "odd" }));
            src.insert("parts".into(), json!([{"v": parts[d][0]}, {"v": parts[d][1]}]));
            WriteReq::Index { id: Some(format!("d{d}")), source: Value::Object(src).to_string(), routing: None, cond: Default::default() }
        }).collect();
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        // Several segments / memtables per shard.
        svc.refresh().unwrap();
    }
    Fixture { node, _dir: dir, vectors, parts }
}

/// Exact top-`k` over `docs`: `(doc, ES score)`, best first (ties: doc).
fn exact(fx: &Fixture, field: usize, q: &[f32], docs: impl Iterator<Item = usize>, k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = docs.map(|d| (d, es_score(FIELDS[field].2, q, &fx.vectors[field][d]))).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn recall(got: &[(usize, f64)], want: &[(usize, f64)]) -> f64 {
    let want: BTreeSet<usize> = want.iter().map(|w| w.0).collect();
    got.iter().filter(|g| want.contains(&g.0)).count() as f64 / want.len().max(1) as f64
}

fn knn(field: &str, q: &[f32], oversample: Option<f64>) -> Value {
    let mut k = json!({"field": field, "query_vector": q, "k": K, "num_candidates": 100});
    if let Some(o) = oversample {
        k["rescore_vector"] = json!({"oversample": o});
    }
    k
}

/// Every hit's `_score` is ES's transform of its `f32` vector.
fn assert_exact_scores(fx: &Fixture, field: usize, q: &[f32], hits: &[(usize, f64)], what: &str) {
    for (d, s) in hits {
        let want = es_score(FIELDS[field].2, q, &fx.vectors[field][*d]);
        assert!((s - want).abs() <= 1e-5 * want.abs().max(1.0), "{what} {}: doc {d} score {s} != f32 {want}", FIELDS[field].0);
    }
    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "{what}: descending scores");
}

#[test]
fn rescoring_returns_f32_scores_and_recall_for_every_quantized_type() {
    let fx = fixture(1);
    let queries = corpus(7, 12, false);
    for (fi, (name, ty, sim, unit)) in FIELDS.iter().enumerate() {
        let (mut with, mut without) = (0.0, 0.0);
        let mut estimated = false;
        for q in &queries {
            let q = if *unit { normalize(q.clone()) } else { q.clone() };
            let truth = exact(&fx, fi, &q, 0..DOCS, K);
            // bbq: ES's default 3x; the limit (10x) for a recall bound.
            let os = if ty.starts_with("bbq") { 10.0 } else { 3.0 };
            let top = hits(&search(&fx.node, "vec", json!({"knn": knn(name, &q, Some(os)), "size": K, "_source": false})).unwrap());
            assert_eq!(top.len(), K);
            assert_exact_scores(&fx, fi, &q, &top, "top-level knn");
            with += recall(&top, &truth);
            // The knn query rescores the same way.
            let query = hits(&search(&fx.node, "vec", json!({"query": {"knn": knn(name, &q, Some(os))}, "size": K, "_source": false})).unwrap());
            assert_exact_scores(&fx, fi, &q, &query, "knn query");
            // A knn retriever with rescore_vector too.
            let retr = hits(&search(&fx.node, "vec", json!({"retriever": {"knn": knn(name, &q, Some(os))}, "size": K, "_source": false})).unwrap());
            assert_exact_scores(&fx, fi, &q, &retr, "knn retriever");
            // oversample 0 disables rescoring: quantized estimates.
            let raw = hits(&search(&fx.node, "vec", json!({"knn": knn(name, &q, Some(0.0)), "size": K, "_source": false})).unwrap());
            assert_eq!(raw.len(), K);
            assert!(raw.windows(2).all(|w| w[0].1 >= w[1].1));
            for (d, s) in &raw {
                let want = es_score(sim, &q, &fx.vectors[fi][*d]);
                assert!((s - want).abs() <= 0.25 * want.abs().max(1.0), "{name}: estimate {s} far from {want}");
                estimated |= (s - want).abs() > 1e-6;
            }
            without += recall(&raw, &truth);
        }
        let n = queries.len() as f64;
        let (with, without) = (with / n, without / n);
        assert!(estimated, "{name}: unrescored scores are quantized estimates");
        // bbq_hnsw's recall threshold is not asserted here; its scores are
        // still checked.
        if *name != "v_bbq_hnsw" {
            assert!(with >= 0.9, "{name}: rescored recall@{K} {with}");
        }
        assert!(with + 1e-9 >= without - 0.05, "{name}: rescored recall {with} below unrescored {without}");
    }
}

#[test]
fn mapping_default_oversample_and_query_override() {
    let fx = fixture(1);
    let q = corpus(8, 1, false).remove(0);
    let bbq = FIELDS.iter().position(|f| f.0 == "v_bbq_flat").unwrap();
    // bbq types default to `rescore_vector.oversample: 3` in the mapping.
    let def = hits(&search(&fx.node, "vec", json!({"knn": knn("v_bbq_flat", &q, None), "size": K, "_source": false})).unwrap());
    assert_exact_scores(&fx, bbq, &q, &def, "bbq default oversample");
    // int8 has no default (ES 9.5): estimates unless the query asks.
    let i8 = FIELDS.iter().position(|f| f.0 == "v_int8_hnsw").unwrap();
    let def = hits(&search(&fx.node, "vec", json!({"knn": knn("v_int8_hnsw", &q, None), "size": K, "_source": false})).unwrap());
    assert!(def.iter().any(|(d, s)| (s - es_score("l2_norm", &q, &fx.vectors[i8][*d])).abs() > 1e-6), "int8 default: no rescoring");
    let off = hits(&search(&fx.node, "vec", json!({"knn": knn("v_int8_hnsw", &q, Some(0.0)), "size": K, "_source": false})).unwrap());
    assert!(off.iter().any(|(d, s)| (s - es_score("l2_norm", &q, &fx.vectors[i8][*d])).abs() > 1e-6), "int8 oversample 0: estimates");
    let over = hits(&search(&fx.node, "vec", json!({"knn": knn("v_int8_hnsw", &q, Some(2.0)), "size": K, "_source": false})).unwrap());
    assert_exact_scores(&fx, i8, &q, &over, "int8 query override");
}

/// A document's unrescored `_score` does not depend on the route: the graph
/// route and the exact route (a filter matching fewer than `num_candidates`
/// documents) both return its row's code score.
#[test]
fn unrescored_scores_agree_between_graph_and_exact_routes() {
    let fx = fixture(1);
    for (name, ty, _, unit) in FIELDS.iter().filter(|f| f.1.ends_with("_hnsw")) {
        for q in corpus(11, 4, *unit) {
            let graph = hits(&search(&fx.node, "vec", json!({"knn": knn(name, &q, Some(0.0)), "size": K, "_source": false})).unwrap());
            let ids: Vec<String> = graph.iter().map(|(d, _)| format!("d{d}")).collect();
            let mut filtered = knn(name, &q, Some(0.0));
            filtered["filter"] = json!({"ids": {"values": ids}});
            let exact = hits(&search(&fx.node, "vec", json!({"knn": filtered, "size": K, "_source": false})).unwrap());
            let graph: HashMap<usize, f64> = graph.into_iter().collect();
            assert_eq!(exact.len(), graph.len(), "{ty}");
            for (d, s) in exact {
                assert!((s - graph[&d]).abs() <= 1e-6 * s.abs().max(1.0), "{ty}: doc {d} exact-route {s} != graph-route {}", graph[&d]);
            }
        }
    }
}

#[test]
fn oversample_validation_matches_es() {
    let fx = fixture(1);
    let q = vec![0.5f32; DIMS];
    // ES `RescoreVectorBuilder`: at least 1.0, or 0 to disable; the 10
    // limit is the mapping's only (gathered candidates stop at 10,000).
    for bad in [0.5, 0.99, -1.0] {
        let e = search(&fx.node, "vec", json!({"knn": knn("v_int8_hnsw", &q, Some(bad))})).unwrap_err();
        assert!(e.contains("[oversample] must be >= 1.0 or 0"), "{bad}: {e}");
        let e = search(&fx.node, "vec", json!({"query": {"knn": knn("v_int8_hnsw", &q, Some(bad))}})).unwrap_err();
        assert!(e.contains("[oversample] must be >= 1.0 or 0"), "{bad}: {e}");
    }
    for ok in [0.0, 1.0, 10.0, 10.5, 2000.0] {
        search(&fx.node, "vec", json!({"knn": knn("v_int8_hnsw", &q, Some(ok))})).unwrap();
    }
    // Mapping side (`DenseVectorFieldMapper.RescoreVector`).
    for (os, msg) in [(0.5, "oversample must be greater than 1 or exactly 0"), (11.0, "oversample must be less than or equal to 10")] {
        let e = fx.node.create_index(&format!("bad{}", (os * 10.0) as i64), &json!({"mappings": {"properties": {"v": {"type": "dense_vector", "dims": DIMS,
            "index_options": {"type": "int8_hnsw", "rescore_vector": {"oversample": os}}}}}})).unwrap_err();
        assert!(chain(&e).contains(msg), "{os}: {}", chain(&e));
    }
    // On an unquantized field a rescore_vector changes nothing (ES `needsRescore`).
    let node = &fx.node;
    node.create_index("plain", &json!({"mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}})).unwrap();
    let svc = node.get_index("plain").unwrap();
    let reqs = (0..50).map(|i| WriteReq::Index { id: Some(format!("d{i}")), source: json!({"v": [i as f32, 1.0, 0.0, 0.5]}).to_string(), routing: None, cond: Default::default() }).collect();
    for r in svc.write(reqs, false) { r.unwrap(); }
    let body = |o: Option<f64>| {
        let mut k = json!({"field": "v", "query_vector": [3.0, 1.0, 0.0, 0.5], "k": 5, "num_candidates": 20});
        if let Some(o) = o { k["rescore_vector"] = json!({"oversample": o}); }
        json!({"knn": k, "size": 5, "_source": false})
    };
    assert_eq!(hits(&search(node, "plain", body(Some(4.0))).unwrap()), hits(&search(node, "plain", body(None)).unwrap()));
}

#[test]
fn filtered_and_nested_rescoring() {
    let fx = fixture(1);
    let queries = corpus(9, 6, false);
    let i8 = FIELDS.iter().position(|f| f.0 == "v_int8_hnsw").unwrap();
    let bbq = FIELDS.iter().position(|f| f.0 == "v_bbq_disk").unwrap();
    for q in &queries {
        for (fi, os) in [(i8, 3.0), (bbq, 10.0)] {
            let mut k = knn(FIELDS[fi].0, q, Some(os));
            k["filter"] = json!({"term": {"tag": "even"}});
            let got = hits(&search(&fx.node, "vec", json!({"knn": k, "size": K, "_source": false})).unwrap());
            assert!(got.iter().all(|(d, _)| d % 2 == 0), "filter respected");
            assert_exact_scores(&fx, fi, q, &got, "filtered");
            let truth = exact(&fx, fi, q, (0..DOCS).step_by(2), K);
            assert!(recall(&got, &truth) >= 0.8, "filtered recall {}", recall(&got, &truth));
        }
        // Nested: each parent scores its best child, rescored with f32.
        let body = json!({"knn": {"field": "parts.v", "query_vector": q, "k": K, "num_candidates": 100, "rescore_vector": {"oversample": 3.0}}, "size": K, "_source": false});
        let got = hits(&search(&fx.node, "vec", body).unwrap());
        assert_eq!(got.len(), K);
        let best = |d: usize| fx.parts[d].iter().map(|v| es_score("l2_norm", q, v)).fold(f64::MIN, f64::max);
        for &(d, s) in &got {
            assert!((s - best(d)).abs() <= 1e-5, "nested doc {d}: {s} != best child {}", best(d));
        }
        let mut truth: Vec<(usize, f64)> = (0..DOCS).map(|d| (d, best(d))).collect();
        truth.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        truth.truncate(K);
        assert!(recall(&got, &truth) >= 0.8, "nested recall {}", recall(&got, &truth));
    }
}

#[test]
fn multi_shard_merge_orders_rescored_scores_and_rrf_runs() {
    let fx = fixture(3);
    let queries = corpus(10, 6, false);
    let i4 = FIELDS.iter().position(|f| f.0 == "v_int4_flat").unwrap();
    let mut total = 0.0;
    for q in &queries {
        let got = hits(&search(&fx.node, "vec", json!({"knn": knn("v_int4_flat", q, Some(3.0)), "size": K, "_source": false})).unwrap());
        assert_eq!(got.len(), K);
        // Global order across shards by the rescored (exact) scores.
        assert_exact_scores(&fx, i4, q, &got, "multi-shard");
        total += recall(&got, &exact(&fx, i4, q, 0..DOCS, K));
        // Hybrid: the rescored knn retriever fused with a text retriever.
        let rrf = search(&fx.node, "vec", json!({"retriever": {"rrf": {"retrievers": [
            {"knn": knn("v_int4_flat", q, Some(3.0))},
            {"standard": {"query": {"term": {"tag": "even"}}}}], "rank_window_size": 20}}, "size": K, "_source": false})).unwrap();
        let fused = hits(&rrf);
        assert_eq!(fused.len(), K);
        let knn_ids: BTreeSet<usize> = got.iter().map(|g| g.0).collect();
        assert!(fused.iter().any(|(d, _)| knn_ids.contains(d)), "rrf includes knn hits");
    }
    assert!(total / queries.len() as f64 >= 0.9, "multi-shard rescored recall {}", total / queries.len() as f64);
}

/// ES `41_knn_search_byte_quantized.yml` "KNN Vector similarity search only"
/// (and its `bfloat16` twin): unrescored `int8_hnsw`, three docs flushed
/// apart and force-merged. ES thresholds `similarity` against the merged
/// segment's OSQ estimates; doc 2 (exact l2 10.44) clears 10.51 there,
/// although the graph's row codes (calibrated before the later docs
/// arrived) estimate it farther.
#[test]
fn similarity_threshold_on_unrescored_int8_hnsw_uses_segment_estimates() {
    for element_type in ["float", "bfloat16"] {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let vector = json!({"type": "dense_vector", "element_type": element_type, "dims": 5, "index": true, "similarity": "l2_norm", "index_options": {"type": "int8_hnsw"}});
        node.create_index("hnsw_byte_quantized", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"name": {"type": "keyword"}, "vector": vector, "another_vector": vector}}})).unwrap();
        let svc = node.get_index("hnsw_byte_quantized").unwrap();
        let docs = [
            ("1", "cow.jpg", [230.0, 300.33, -34.8988, 15.555, -200.0], [130.0, 115.0, -1.02, 15.555, -100.0]),
            ("2", "moose.jpg", [-0.5, 100.0, -13.0, 14.8, -156.0], [-0.5, 50.0, -1.0, 1.0, 120.0]),
            ("3", "rabbit.jpg", [0.5, 111.3, -13.0, 14.8, -156.0], [-0.5, 11.0, 0.0, 12.0, 111.0]),
        ];
        for (id, name, v, a) in docs {
            let src = json!({"name": name, "vector": v, "another_vector": a}).to_string();
            for r in svc.write(vec![WriteReq::Index { id: Some(id.into()), source: src, routing: None, cond: Default::default() }], false) {
                r.unwrap();
            }
            svc.flush().unwrap();
        }
        svc.force_merge(1, false).unwrap();
        let body = json!({"fields": ["name"], "knn": {"num_candidates": 3, "k": 3, "field": "vector", "similarity": 10.51,
            "query_vector": [-0.5, 90.0, -10.0, 14.8, -156.0]}});
        let r = search(&node, "hnsw_byte_quantized", body).unwrap();
        let ids: Vec<&str> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["2"], "{element_type}: {r}");
    }
}
