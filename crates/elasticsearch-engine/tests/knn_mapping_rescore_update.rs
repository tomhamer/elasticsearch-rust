//! A `rescore_vector` changed by a mapping update applies to the next kNN
//! search (ES reads the field's `index_options` per query): with the
//! mapping's oversample raised from 0 to 2, a `bbq_disk` search without a
//! request `rescore_vector` returns the exact f32 scores, and set back to 0
//! the quantized estimates (`46_knn_search_bbq_ivf` score consistency).

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;

fn vector(i: usize) -> Vec<f32> {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..64).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5 }).collect()
}

fn mapping(oversample: u32) -> Value {
    json!({"properties": {"vector": {"type": "dense_vector", "dims": 64, "index": true, "similarity": "max_inner_product",
        "index_options": {"type": "bbq_disk", "rescore_vector": {"oversample": oversample}}}}})
}

fn scores(node: &Node, q: &[f32], body_extra: Option<Value>) -> Vec<f64> {
    let mut knn = json!({"field": "vector", "query_vector": q, "k": 3, "num_candidates": 3});
    if let Some(rv) = body_extra { knn["rescore_vector"] = rv; }
    let req = search::parse_request(&json!({"knn": knn, "_source": false}), &HashMap::new()).unwrap();
    let targets = node.resolve("rv", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    out["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_score"].as_f64().unwrap()).collect()
}

#[test]
fn mapping_rescore_vector_update_applies_to_the_next_search() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("rv", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping(0)})).unwrap();
    let svc = node.get_index("rv").unwrap();
    let reqs: Vec<WriteReq> = (0..3).map(|i| WriteReq::Index { id: Some(format!("{i}")), source: json!({"vector": vector(i)}).to_string(), routing: None, cond: Default::default() }).collect();
    assert!(svc.write(reqs, true).iter().all(Result::is_ok));
    svc.refresh().unwrap();
    let q = vector(100);
    let estimated = scores(&node, &q, None);
    let rescored = scores(&node, &q, Some(json!({"oversample": 2})));
    assert_ne!(estimated, rescored, "1-bit estimates differ from the f32 scores");
    svc.put_mapping(&mapping(2)).unwrap();
    assert_eq!(scores(&node, &q, None), rescored, "the mapping's oversample 2 rescores");
    svc.put_mapping(&mapping(0)).unwrap();
    assert_eq!(scores(&node, &q, None), estimated, "oversample 0 again: estimates");
}
