//! Quantized graph fields (`int8_hnsw`, `bbq_hnsw`) reopened from their
//! commit: the codes and their calibration epochs are in the graph
//! checkpoint, so unrescored kNN (the code estimates, `oversample: 0`)
//! returns bit-identical hits and scores after a reopen from a commit whose
//! checkpoint is current. When the checkpoint lags the commit (the graph
//! save schedule skipped the last flush), the rows above it are inserted
//! again at open against the restored calibration state: the checkpointed
//! rows keep their exact estimates, rescored scores stay exact, and the
//! test reports whether the re-inserted rows' estimates also matched.
//! Saving the graphs again makes the next reopen bit-identical.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const DIMS: usize = 64; // bbq_hnsw needs at least 64 dims
const K: usize = 10;
const FIRST: usize = 1200;
const TAIL: usize = 300;

fn vector(i: usize, unit: bool) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let c = (i % 17) as f32;
    let v: Vec<f32> = (0..DIMS).map(|d| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        (d as f32 * 0.31 + c).sin() + 0.3 * ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
    }).collect();
    if unit {
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.into_iter().map(|a| a / n).collect()
    } else {
        v
    }
}

/// `(field, index type, similarity, unit vectors)`.
const FIELDS: &[(&str, &str, &str, bool)] = &[
    ("q8", "int8_hnsw", "l2_norm", false),
    ("bq", "bbq_hnsw", "dot_product", true),
];

fn create(node: &Arc<Node>) {
    let mut props = serde_json::Map::new();
    for (name, ty, sim, _) in FIELDS {
        props.insert(name.to_string(), json!({"type": "dense_vector", "dims": DIMS, "index": true, "similarity": sim, "index_options": {"type": ty}}));
    }
    node.create_index("ri", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": props}})).unwrap();
}

fn write(node: &Arc<Node>, docs: std::ops::Range<usize>) {
    let svc = node.get_index("ri").unwrap();
    let all: Vec<usize> = docs.collect();
    for chunk in all.chunks(300) {
        let reqs = chunk.iter().map(|&d| {
            let mut src = serde_json::Map::new();
            for (name, _, _, unit) in FIELDS {
                src.insert(name.to_string(), json!(vector(d, *unit)));
            }
            WriteReq::Index { id: Some(format!("d{d}")), source: Value::Object(src).to_string(), routing: None, cond: Default::default() }
        }).collect();
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }
}

fn knn(node: &Arc<Node>, field: &str, q: &[f32], oversample: f64) -> Vec<(usize, f64)> {
    let body = json!({"knn": {"field": field, "query_vector": q, "k": K, "num_candidates": 100, "rescore_vector": {"oversample": oversample}}, "size": K, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("ri", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let v: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    v["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap()[1..].parse().unwrap(), h["_score"].as_f64().unwrap())).collect()
}

/// Per field and query: unrescored and rescored hits.
type Answers = Vec<(Vec<(usize, f64)>, Vec<(usize, f64)>)>;

fn answers(node: &Arc<Node>) -> Answers {
    let mut out = Vec::new();
    for (name, _, _, unit) in FIELDS {
        for qi in 0..8 {
            let q = vector(qi * 131 + 50_000, *unit);
            out.push((knn(node, name, &q, 0.0), knn(node, name, &q, 5.0)));
        }
    }
    out
}

fn lag(node: &Arc<Node>) -> Option<i64> {
    node.get_index("ri").unwrap().local_shards()[0].shard.graph_save_lag()
}

#[test]
fn quantized_graph_estimates_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    // A current checkpoint: the first flush always saves the graphs.
    let current = {
        let node = Node::open(dir.path()).unwrap();
        create(&node);
        write(&node, 0..FIRST);
        node.get_index("ri").unwrap().flush().unwrap();
        assert_eq!(lag(&node), Some(0), "the first flush saves the graphs");
        answers(&node)
    };
    let node = Node::open(dir.path()).unwrap();
    assert_eq!(answers(&node), current, "unrescored estimates changed across a reopen from a current checkpoint");

    // A lagging checkpoint: the next flush commits the tail without saving
    // the graphs (far below the save schedule's op threshold).
    write(&node, FIRST..FIRST + TAIL);
    node.get_index("ri").unwrap().flush().unwrap();
    let behind = lag(&node);
    assert!(behind.is_some_and(|l| l >= TAIL as i64), "the checkpoint should lag the commit by the tail: {behind:?}");
    let lagging = answers(&node);
    drop(node);
    let node = Node::open(dir.path()).unwrap();
    let reopened = answers(&node);
    let mut tail_same = true;
    for (i, ((raw0, exact0), (raw1, exact1))) in lagging.iter().zip(&reopened).enumerate() {
        // Rescored scores are exact f32: a doc in both answers has the same
        // score (the re-inserted rows' links may route the beam to another
        // near miss).
        assert_eq!(exact1.len(), K);
        let exact_before: BTreeMap<usize, f64> = exact0.iter().copied().collect();
        for (d, s) in exact1 {
            if let Some(b) = exact_before.get(d) {
                assert_eq!(s, b, "answer {i}: rescored score of d{d} changed");
            }
        }
        let shared = exact1.iter().filter(|h| exact_before.contains_key(&h.0)).count();
        assert!(shared * 10 >= K * 8, "answer {i}: rescored hits changed too much across the reopen: {exact0:?} vs {exact1:?}");
        // Checkpointed rows keep their codes and calibration: identical
        // estimates wherever both answers hold them.
        let before: BTreeMap<usize, f64> = raw0.iter().copied().collect();
        for (d, s) in raw1 {
            match before.get(d) {
                Some(b) if *d < FIRST => assert_eq!(s, b, "answer {i}: checkpointed doc d{d} estimate changed ({b} -> {s})"),
                Some(b) => tail_same &= s == b,
                None => tail_same &= *d < FIRST,
            }
        }
        tail_same &= raw0 == raw1;
    }
    eprintln!("lagging checkpoint ({behind:?} ops behind): re-inserted rows' estimates {}", if tail_same { "identical" } else { "differ (refitted calibration)" });
    // Saved again, a reopen is bit-identical once more.
    node.get_index("ri").unwrap().local_shards()[0].shard.request_graph_save();
    node.get_index("ri").unwrap().flush().unwrap();
    assert_eq!(lag(&node), Some(0));
    let saved = answers(&node);
    drop(node);
    let node = Node::open(dir.path()).unwrap();
    assert_eq!(answers(&node), saved, "unrescored estimates changed across a reopen from a re-saved checkpoint");
}
