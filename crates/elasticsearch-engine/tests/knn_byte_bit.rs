//! `byte` and `bit` `dense_vector` fields on the shared mutable graph
//! (`hnsw`) and flat (`flat`) paths, through the search API: recall
//! against brute force over ES's scores of the indexed values (`byte`
//! `l2_norm`, `cosine`, `dot_product`; `bit` Hamming, a 4104-bit field
//! included: bit fields take up to 32768 dims), exact ES scores, and the
//! same answers after the node is reopened, both from a flushed commit
//! (graph checkpoint) and from translog replay.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

const DOCS: usize = 900;
const K: usize = 10;
const BYTE_DIMS: usize = 32;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Clustered byte vectors in [-128, 127].
fn bytes(seed: u64, n: usize, dims: usize) -> Vec<Vec<i8>> {
    let mut r = Rng(seed);
    let centres: Vec<Vec<i32>> = (0..20).map(|_| (0..dims).map(|_| (r.next() % 200) as i32 - 100).collect()).collect();
    (0..n).map(|i| {
        let c = &centres[(r.next() as usize + i) % centres.len()];
        c.iter().map(|x| (x + (r.next() % 41) as i32 - 20).clamp(-128, 127) as i8).collect()
    }).collect()
}

/// Packed bit vectors (`dims / 8` bytes), clustered: each a centre with a
/// few flipped bits.
fn bits(seed: u64, n: usize, dims: usize) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    let len = dims / 8;
    let centres: Vec<Vec<u8>> = (0..16).map(|_| (0..len).map(|_| r.next() as u8).collect()).collect();
    (0..n).map(|i| {
        let mut v = centres[(r.next() as usize + i) % centres.len()].clone();
        for _ in 0..(dims / 16) {
            let b = r.next() as usize % dims;
            v[b / 8] ^= 1 << (b % 8);
        }
        v
    }).collect()
}

fn byte_score(similarity: &str, q: &[i8], v: &[i8]) -> f64 {
    let dot: f64 = q.iter().zip(v).map(|(a, b)| *a as f64 * *b as f64).sum();
    match similarity {
        "l2_norm" => 1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>()),
        "cosine" => {
            let n = |x: &[i8]| x.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            (1.0 + dot / (n(q) * n(v))) / 2.0
        }
        "dot_product" => 0.5 + dot / (q.len() as f64 * 32768.0),
        s => panic!("{s}"),
    }
}

/// ES's `bit` score: `(dims - hamming) / dims`.
fn bit_score(q: &[u8], v: &[u8]) -> f64 {
    let dims = (q.len() * 8) as f64;
    let hamming: u32 = q.iter().zip(v).map(|(a, b)| (a ^ b).count_ones()).sum();
    (dims - hamming as f64) / dims
}

/// `(field, element_type, dims, similarity, index type)`.
const FIELDS: &[(&str, &str, usize, &str, &str)] = &[
    ("b_l2_hnsw", "byte", BYTE_DIMS, "l2_norm", "hnsw"),
    ("b_cos_hnsw", "byte", BYTE_DIMS, "cosine", "hnsw"),
    ("b_dot_hnsw", "byte", BYTE_DIMS, "dot_product", "hnsw"),
    ("b_l2_flat", "byte", BYTE_DIMS, "l2_norm", "flat"),
    ("x_hnsw", "bit", 256, "l2_norm", "hnsw"),
    ("x_flat", "bit", 256, "l2_norm", "flat"),
    ("x_wide", "bit", 4104, "l2_norm", "hnsw"),
];

enum Data {
    Byte(Vec<Vec<i8>>),
    Bit(Vec<Vec<u8>>),
}

impl Data {
    fn json(&self, d: usize) -> Value {
        match self {
            Data::Byte(v) => json!(v[d]),
            // Bit vectors as byte arrays (ES accepts both arrays and hex).
            Data::Bit(v) => json!(v[d].iter().map(|b| *b as i8).collect::<Vec<_>>()),
        }
    }
    fn score(&self, similarity: &str, q: usize, d: usize) -> f64 {
        match self {
            Data::Byte(v) => byte_score(similarity, &v[q], &v[d]),
            Data::Bit(v) => bit_score(&v[q], &v[d]),
        }
    }
}

fn data() -> Vec<Data> {
    FIELDS.iter().enumerate().map(|(i, (_, et, dims, ..))| match *et {
        "byte" => Data::Byte(bytes(40 + i as u64, DOCS, *dims)),
        _ => Data::Bit(bits(40 + i as u64, DOCS, *dims)),
    }).collect()
}

fn create(node: &Arc<Node>) {
    let mut props = serde_json::Map::new();
    for (name, et, dims, sim, ty) in FIELDS {
        props.insert(name.to_string(), json!({"type": "dense_vector", "element_type": et, "dims": dims, "index": true,
            "similarity": sim, "index_options": {"type": ty}}));
    }
    node.create_index("bb", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": props}})).unwrap();
}

fn write(node: &Arc<Node>, data: &[Data], docs: std::ops::Range<usize>) {
    let svc = node.get_index("bb").unwrap();
    let all: Vec<usize> = docs.collect();
    for chunk in all.chunks(300) {
        let reqs = chunk.iter().map(|&d| {
            let mut src = serde_json::Map::new();
            for (i, (name, ..)) in FIELDS.iter().enumerate() {
                src.insert(name.to_string(), data[i].json(d));
            }
            WriteReq::Index { id: Some(format!("d{d}")), source: Value::Object(src).to_string(), routing: None, cond: Default::default() }
        }).collect();
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }
}

fn knn(node: &Arc<Node>, field: &str, q: &Value) -> Vec<(usize, f64)> {
    let body = json!({"knn": {"field": field, "query_vector": q, "k": K, "num_candidates": 100}, "size": K, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("bb", Default::default()).unwrap();
    for t in &targets {
        t.svc.refresh().unwrap();
    }
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let v: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    v["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap()[1..].parse().unwrap(), h["_score"].as_f64().unwrap())).collect()
}

/// Per field and query: the hits, after checking scores and recall.
fn check(node: &Arc<Node>, data: &[Data], queries: &[usize]) -> Vec<Vec<(usize, f64)>> {
    let mut out = Vec::new();
    for (fi, (name, _, _, sim, ty)) in FIELDS.iter().enumerate() {
        let mut found = 0;
        for &q in queries {
            let hits = knn(node, name, &data[fi].json(q));
            assert_eq!(hits.len(), K, "{name} query {q}");
            for (d, s) in &hits {
                let want = data[fi].score(sim, q, *d);
                assert!((s - want).abs() <= 1e-5 * want.abs().max(1.0), "{name} query {q}: doc {d} score {s} != ES {want}");
            }
            assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "{name}: descending scores");
            let mut exact: Vec<(usize, f64)> = (0..DOCS).map(|d| (d, data[fi].score(sim, q, d))).collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            // Ties at the k-th score (common for bit vectors) count as found.
            let kth = exact[K - 1].1;
            let truth: BTreeSet<usize> = exact.iter().take_while(|e| e.1 >= kth).map(|e| e.0).collect();
            found += hits.iter().filter(|h| truth.contains(&h.0)).count();
            out.push(hits);
        }
        let recall = found as f64 / (K * queries.len()) as f64;
        // Flat fields search exactly.
        let floor = if *ty == "flat" { 1.0 } else { 0.9 };
        assert!(recall >= floor, "{name}: recall {recall:.3} < {floor}");
    }
    out
}

#[test]
fn byte_and_bit_knn_recall_and_survival_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let data = data();
    let queries: Vec<usize> = (0..10).map(|i| i * 89 + 7).collect();
    let committed = {
        let node = Node::open(dir.path()).unwrap();
        create(&node);
        write(&node, &data, 0..DOCS * 2 / 3);
        node.get_index("bb").unwrap().flush().unwrap();
        // The rest only in the translog (no flush before the reopen).
        write(&node, &data, DOCS * 2 / 3..DOCS);
        node.get_index("bb").unwrap().sync_translog().unwrap();
        check(&node, &data, &queries)
    };
    // Reopen: graph checkpoint plus translog replay above it. Every field
    // keeps its recall and exact scores (checked again); flat fields, which
    // search exactly, return the very same hits. (A reopened graph draws
    // node levels from a fresh generator, so HNSW answers may differ in
    // ties and near misses.)
    let same_flat = |got: &[Vec<(usize, f64)>], what: &str| {
        for (fi, (name, _, _, _, ty)) in FIELDS.iter().enumerate() {
            if *ty != "flat" { continue; }
            for qi in 0..queries.len() {
                let i = fi * queries.len() + qi;
                assert_eq!(got[i], committed[i], "{name} query {}: flat answers changed {what}", queries[qi]);
            }
        }
    };
    let node = Node::open(dir.path()).unwrap();
    assert_eq!(node.get_index("bb").unwrap().doc_count_realtime().0 as usize, DOCS, "docs lost across the reopen");
    same_flat(&check(&node, &data, &queries), "across a reopen");
    // And once more from a flushed commit only.
    node.get_index("bb").unwrap().flush().unwrap();
    drop(node);
    let node = Node::open(dir.path()).unwrap();
    same_flat(&check(&node, &data, &queries), "across a reopen from a flushed commit");
}

fn run(node: &Arc<Node>, index: &str, body: &Value) -> Value {
    let req = search::parse_request(body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap()
}

fn hits_of(v: &Value) -> Vec<(usize, f64)> {
    v["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap()[1..].parse().unwrap(), h["_score"].as_f64().unwrap())).collect()
}

/// `got` is the top of `all` (every doc's expected score): the same scores
/// rank by rank, each hit one of the docs holding its score (ties may come
/// in any order).
fn assert_top(got: &[(usize, f64)], all: &[(usize, f64)], k: usize, what: &str) {
    let mut want = all.to_vec();
    want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    want.truncate(k);
    assert_eq!(got.len(), want.len(), "{what}: hit count");
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0);
    for (rank, ((d, s), (_, w))) in got.iter().zip(&want).enumerate() {
        assert!(close(*s, *w), "{what}: rank {rank} doc {d} score {s} != {w}");
        let own = all.iter().find(|e| e.0 == *d).unwrap_or_else(|| panic!("{what}: doc {d} unknown")).1;
        assert!(close(own, *s), "{what}: rank {rank} doc {d} scored {s}, expected {own}");
    }
}

/// Packed storage: a 1024-bit row is 128 bytes and a 768-dim byte row 768
/// bytes in the vector store (no f32 per value), and `docvalue_fields`
/// returns the indexed values.
#[test]
fn packed_rows_store_dims_over_8_bytes_for_bit_and_dims_bytes_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let n = 300;
    let bit_rows = bits(7, n, 1024);
    let byte_rows = bytes(8, n, 768);
    node.create_index("pk", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "bits": {"type": "dense_vector", "element_type": "bit", "dims": 1024, "index": true, "similarity": "l2_norm", "index_options": {"type": "hnsw"}},
            "bytes": {"type": "dense_vector", "element_type": "byte", "dims": 768, "index": true, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}})).unwrap();
    let svc = node.get_index("pk").unwrap();
    let reqs = (0..n).map(|d| WriteReq::Index { id: Some(format!("d{d}")),
        source: json!({"bits": bit_rows[d].iter().map(|b| *b as i8).collect::<Vec<_>>(), "bytes": byte_rows[d]}).to_string(),
        routing: None, cond: Default::default() }).collect();
    for r in svc.write(reqs, false) { r.unwrap(); }
    svc.refresh().unwrap();
    svc.flush().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    let lookup = svc.lookup();
    let bits_id = lookup.get("bits").unwrap().core_id.unwrap();
    let bytes_id = lookup.get("bytes").unwrap().core_id.unwrap();
    let mut stored: HashMap<u16, Vec<Vec<u8>>> = HashMap::new();
    for sh in svc.local_shards() {
        let store = sh.shard.core.read().vector_store().clone();
        for id in store.segment_ids() {
            let Some(s) = store.segment(id) else { continue };
            for ord in 0..s.len() {
                let row = s.vector(ord);
                assert_eq!(row.len() as u32, s.dim);
                stored.entry(s.field).or_default().push(rustic_core::vector::packed::as_bytes(row).to_vec());
            }
        }
    }
    let sorted = |mut v: Vec<Vec<u8>>| { v.sort(); v };
    let bits_stored = sorted(stored.remove(&bits_id).unwrap_or_default());
    let bytes_stored = sorted(stored.remove(&bytes_id).unwrap_or_default());
    assert!(bits_stored.iter().all(|r| r.len() == 1024 / 8), "bit rows hold dims / 8 bytes");
    assert!(bytes_stored.iter().all(|r| r.len() == 768), "byte rows hold dims bytes");
    assert_eq!(bits_stored, sorted(bit_rows.clone()), "bit rows are the indexed bytes");
    assert_eq!(bytes_stored, sorted(byte_rows.iter().map(|r| r.iter().map(|b| *b as u8).collect()).collect()), "byte rows are the indexed values");
    // Reads unpack: doc values give back the indexed values.
    let v = run(&node, "pk", &json!({"size": 3, "query": {"ids": {"values": ["d3", "d4", "d5"]}}, "_source": false,
        "docvalue_fields": ["bits", "bytes"]}));
    for h in v["hits"]["hits"].as_array().unwrap() {
        let d: usize = h["_id"].as_str().unwrap()[1..].parse().unwrap();
        assert_eq!(h["fields"]["bits"][0], json!(bit_rows[d].iter().map(|b| *b as i8).collect::<Vec<_>>()), "doc {d} bits");
        assert_eq!(h["fields"]["bytes"][0], json!(byte_rows[d]), "doc {d} bytes");
    }
}

/// Exact routes on packed rows (`flat` fields, and an `hnsw` field under a
/// 1% filter, which Vespa's policy searches exactly) and `script_score`
/// vector functions return the brute-force top `k` of the unpacked values,
/// with ES's scores.
#[test]
fn packed_knn_and_script_score_equal_brute_force_over_unpacked_values() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let n = 2000;
    const BYTES: usize = 768;
    const BITS: usize = 1024;
    let bit_rows = bits(17, n, BITS);
    let byte_rows = bytes(18, n, BYTES);
    let sims = ["l2_norm", "cosine", "dot_product", "max_inner_product"];
    let mut props = serde_json::Map::new();
    props.insert("bits".into(), json!({"type": "dense_vector", "element_type": "bit", "dims": BITS, "index": true, "similarity": "l2_norm", "index_options": {"type": "flat"}}));
    props.insert("bits_hnsw".into(), json!({"type": "dense_vector", "element_type": "bit", "dims": BITS, "index": true, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}));
    for sim in sims {
        props.insert(format!("b_{sim}"), json!({"type": "dense_vector", "element_type": "byte", "dims": BYTES, "index": true, "similarity": sim, "index_options": {"type": "flat"}}));
    }
    props.insert("b_hnsw".into(), json!({"type": "dense_vector", "element_type": "byte", "dims": BYTES, "index": true, "similarity": "cosine", "index_options": {"type": "hnsw"}}));
    props.insert("tag".into(), json!({"type": "keyword"}));
    node.create_index("dx", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": props}})).unwrap();
    let svc = node.get_index("dx").unwrap();
    let tagged = |d: usize| d % 100 == 3;
    for start in (0..n).step_by(500) {
        let reqs = (start..(start + 500).min(n)).map(|d| {
            let mut src = serde_json::Map::new();
            let bits_json = json!(bit_rows[d].iter().map(|b| *b as i8).collect::<Vec<_>>());
            src.insert("bits".into(), bits_json.clone());
            src.insert("bits_hnsw".into(), bits_json);
            for sim in sims { src.insert(format!("b_{sim}"), json!(byte_rows[d])); }
            src.insert("b_hnsw".into(), json!(byte_rows[d]));
            if tagged(d) { src.insert("tag".into(), json!("x")); }
            WriteReq::Index { id: Some(format!("d{d}")), source: Value::Object(src).to_string(), routing: None, cond: Default::default() }
        }).collect();
        for r in svc.write(reqs, false) { r.unwrap(); }
        svc.refresh().unwrap();
    }
    svc.flush().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    let es_byte = |sim: &str, q: &[i8], v: &[i8]| -> f64 {
        let dot: f64 = q.iter().zip(v).map(|(a, b)| *a as f64 * *b as f64).sum();
        match sim {
            "max_inner_product" => if dot < 0.0 { 1.0 / (1.0 - dot) } else { dot + 1.0 },
            s => byte_score(s, q, v),
        }
    };
    for q in [0usize, 777, 1999] {
        let bq = json!(bit_rows[q].iter().map(|b| *b as i8).collect::<Vec<_>>());
        // kNN: flat (exact) and filtered hnsw (exact route).
        let all: Vec<(usize, f64)> = (0..n).map(|d| (d, bit_score(&bit_rows[q], &bit_rows[d]))).collect();
        let got = hits_of(&run(&node, "dx", &json!({"knn": {"field": "bits", "query_vector": bq, "k": K, "num_candidates": 100}, "size": K, "_source": false})));
        assert_top(&got, &all, K, &format!("bit flat q{q}"));
        let filtered: Vec<(usize, f64)> = all.iter().copied().filter(|e| tagged(e.0)).collect();
        let got = hits_of(&run(&node, "dx", &json!({"knn": {"field": "bits_hnsw", "query_vector": bq, "k": K, "num_candidates": 100,
            "filter": {"term": {"tag": "x"}}}, "size": K, "_source": false})));
        assert_top(&got, &filtered, K, &format!("bit hnsw 1% filter q{q}"));
        for sim in sims {
            let all: Vec<(usize, f64)> = (0..n).map(|d| (d, es_byte(sim, &byte_rows[q], &byte_rows[d]))).collect();
            let got = hits_of(&run(&node, "dx", &json!({"knn": {"field": format!("b_{sim}"), "query_vector": byte_rows[q], "k": K, "num_candidates": 100}, "size": K, "_source": false})));
            assert_top(&got, &all, K, &format!("byte {sim} flat q{q}"));
        }
        let all: Vec<(usize, f64)> = (0..n).filter(|&d| tagged(d)).map(|d| (d, es_byte("cosine", &byte_rows[q], &byte_rows[d]))).collect();
        let got = hits_of(&run(&node, "dx", &json!({"knn": {"field": "b_hnsw", "query_vector": byte_rows[q], "k": K, "num_candidates": 100,
            "filter": {"term": {"tag": "x"}}}, "size": K, "_source": false})));
        assert_top(&got, &all, K, &format!("byte cosine hnsw 1% filter q{q}"));
        // script_score over the unpacked values (bit fields read as one 0/1
        // value per bit, byte fields as their values).
        let q01: Vec<f32> = (0..BITS).map(|i| ((bit_rows[q][i / 8] >> (7 - i % 8)) & 1) as f32).collect();
        let unpacked_bits = |d: usize| -> Vec<f32> { (0..BITS).map(|i| ((bit_rows[d][i / 8] >> (7 - i % 8)) & 1) as f32).collect() };
        let qb: Vec<f32> = byte_rows[q].iter().map(|x| *x as f32).collect();
        let row = |d: usize| -> Vec<f32> { byte_rows[d].iter().map(|x| *x as f32).collect() };
        let script = |field: &str, source: &str, query: &[f32]| hits_of(&run(&node, "dx", &json!({"size": K, "_source": false,
            "query": {"script_score": {"query": {"match_all": {}}, "script": {"source": source.replace("F", field), "params": {"q": query}}}}})));
        let f32_score = |s: f64| s as f32 as f64;
        let dot = |a: &[f32], b: &[f32]| rustic_core::vector::dot(a, b);
        let cases: Vec<(&str, &str, &str, Vec<f32>, Box<dyn Fn(usize) -> f64 + '_>)> = vec![
            ("bits", "hamming(params.q, 'F') + 1.0", "bit hamming", q01.clone(), Box::new(|d: usize| f32_score(q01.iter().zip(&unpacked_bits(d)).filter(|(a, b)| a != b).count() as f64 + 1.0))),
            ("bits", "dotProduct(params.q, 'F') + 1.0", "bit dotProduct (bound)", q01.clone(), Box::new(|d: usize| f32_score(dot(&q01, &unpacked_bits(d)) as f64 + 1.0))),
            ("b_l2_norm", "dotProduct(params.q, 'F') + 20000000.0", "byte dotProduct (bound)", qb.clone(), Box::new(|d: usize| f32_score(dot(&qb, &row(d)) as f64 + 20000000.0))),
            ("b_cosine", "cosineSimilarity(params.q, 'F') + 1.0", "byte cosineSimilarity (bound)", qb.clone(), Box::new(|d: usize| {
                let r = row(d); f32_score((dot(&qb, &r) / (rustic_core::vector::norm(&qb) * rustic_core::vector::norm(&r))) as f64 + 1.0) })),
            ("b_dot_product", "l1norm(params.q, 'F')", "byte l1norm", qb.clone(), Box::new(|d: usize| f32_score(qb.iter().zip(&row(d)).map(|(a, b)| (a - b).abs() as f64).sum()))),
            ("b_max_inner_product", "l2norm(params.q, 'F')", "byte l2norm", qb.clone(), Box::new(|d: usize| f32_score((rustic_core::vector::l2_sq(&qb, &row(d)) as f64).sqrt()))),
            ("b_hnsw", "hamming(params.q, 'F')", "byte hamming", qb.clone(), Box::new(|d: usize| f32_score(byte_rows[q].iter().zip(&byte_rows[d]).map(|(a, b)| ((*a as u8) ^ (*b as u8)).count_ones() as f64).sum()))),
        ];
        for (field, source, what, query, expected) in &cases {
            let all: Vec<(usize, f64)> = (0..n).map(|d| (d, expected(d))).collect();
            assert_top(&script(field, source, query), &all, K, &format!("{what} q{q}"));
        }
    }
}

/// `cargo test --release -p elasticsearch-engine --test knn_byte_bit --
/// --ignored --nocapture packed_rows_microbench` (`RUSTIC_BENCH_DOCS`,
/// default 50,000): build time, kNN p50 and stored bytes per vector of a
/// 1024-bit and a 768-dim byte field, packed (now) and as the f32 rows
/// they used to be stored as (a float field of the same 0/1 or byte
/// values under the same similarity: the old layout and kernels).
#[test]
#[ignore]
fn packed_rows_microbench() {
    let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let bit_rows = bits(91, docs, 1024);
    let byte_rows = bytes(92, docs, 768);
    let unpack_bits = |r: &[u8]| -> Vec<f32> { (0..r.len() * 8).map(|i| ((r[i / 8] >> (7 - i % 8)) & 1) as f32).collect() };
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    // (index, element_type, dims, value of doc d)
    let cases: Vec<(&str, &str, usize, Box<dyn Fn(usize) -> Value + '_>)> = vec![
        ("bit_packed", "bit", 1024, Box::new(|d: usize| json!(bit_rows[d].iter().map(|b| *b as i8).collect::<Vec<_>>()))),
        ("bit_as_f32", "float", 1024, Box::new(|d: usize| json!(unpack_bits(&bit_rows[d])))),
        ("byte_packed", "byte", 768, Box::new(|d: usize| json!(byte_rows[d]))),
        ("byte_as_f32", "float", 768, Box::new(|d: usize| json!(byte_rows[d].iter().map(|x| *x as f32).collect::<Vec<_>>()))),
    ];
    for (index, et, dims, value) in &cases {
        node.create_index(index, &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
            "mappings": {"properties": {"v": {"type": "dense_vector", "element_type": et, "dims": dims, "index": true,
                "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}})).unwrap();
        let svc = node.get_index(index).unwrap();
        let t = std::time::Instant::now();
        for start in (0..docs).step_by(2_000) {
            let reqs = (start..(start + 2_000).min(docs)).map(|d| WriteReq::Index { id: Some(format!("d{d}")),
                source: json!({"v": value(d)}).to_string(), routing: None, cond: Default::default() }).collect();
            for r in svc.write(reqs, false) { r.unwrap(); }
        }
        svc.refresh().unwrap();
        svc.flush().unwrap();
        let build = t.elapsed();
        let mut row_bytes = 0;
        for sh in svc.local_shards() {
            let store = sh.shard.core.read().vector_store().clone();
            if let Some(s) = store.segment_ids().into_iter().find_map(|id| store.segment(id)) { row_bytes = s.dim as usize * 4; }
        }
        let mut times = Vec::new();
        for q in 0..50 {
            let body = json!({"knn": {"field": "v", "query_vector": value((q * 997) % docs), "k": 10, "num_candidates": 100}, "size": 10, "_source": false});
            let t = std::time::Instant::now();
            let v = run(&node, index, &body);
            times.push(t.elapsed());
            assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 10);
        }
        times.sort();
        eprintln!("{index:12} {docs} x {dims}{}: build {build:?}, knn p50 {:?} p99 {:?}, {row_bytes} bytes/vector",
            if *et == "bit" { " bits" } else { "" }, times[times.len() / 2], times[times.len() * 99 / 100]);
    }
}
