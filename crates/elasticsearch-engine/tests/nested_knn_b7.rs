//! SIDE 001 / SIDE 002: nested kNN parent identity and copied fields.
//! - Two nested levels (`groups.items.v`): a top-level kNN diversifies by
//!   root document, so two groups of one root are one candidate and k=2
//!   returns two distinct roots; a root filter applies to roots.
//! - `include_in_parent` / `include_in_root`: a filter on a field the nested
//!   object copies to its parent is a parent filter, not a child filter.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;

fn ids(node: &Node, index: &str, body: Value) -> Vec<String> {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    out["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

fn index(node: &Node, name: &str, mappings: Value, docs: Vec<(&str, Value)>) {
    node.create_index(name, &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mappings})).unwrap();
    let svc = node.get_index(name).unwrap();
    let reqs = docs.into_iter().map(|(id, d)| WriteReq::Index { id: Some(id.into()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
    assert!(svc.write(reqs, true).iter().all(Result::is_ok));
    svc.refresh().unwrap();
}

#[test]
fn deep_nested_knn_diversifies_by_root() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mappings = json!({"properties": {"tenant": {"type": "keyword"}, "groups": {"type": "nested", "properties": {"items": {"type": "nested", "properties": {
        "v": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm", "index_options": {"type": "flat"}}}}}}}});
    index(&node, "deep", mappings, vec![
        ("A", json!({"tenant": "t", "groups": [{"items": [{"v": [1.0, 0.0]}]}, {"items": [{"v": [1.01, 0.0]}]}]})),
        ("B", json!({"tenant": "t", "groups": [{"items": [{"v": [1.1, 0.0]}]}]})),
        ("C", json!({"tenant": "u", "groups": [{"items": [{"v": [5.0, 0.0]}]}]})),
    ]);
    let knn = |filter: Option<Value>| {
        let mut k = json!({"field": "groups.items.v", "query_vector": [1.0, 0.0], "k": 2, "num_candidates": 10});
        if let Some(f) = filter { k["filter"] = f; }
        json!({"knn": k, "size": 10, "_source": false})
    };
    assert_eq!(ids(&node, "deep", knn(None)), vec!["A", "B"], "two distinct roots");
    assert_eq!(ids(&node, "deep", knn(Some(json!({"term": {"tenant": "t"}})))), vec!["A", "B"], "the root filter applies to roots");
}

#[test]
fn copied_nested_fields_are_parent_filters() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    for (name, flag) in [("plain", None), ("parent", Some("include_in_parent")), ("root", Some("include_in_root"))] {
        let mut parts = json!({"type": "nested", "properties": {"color": {"type": "keyword"},
            "v": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm", "index_options": {"type": "flat"}}}});
        if let Some(f) = flag { parts[f] = json!(true); }
        index(&node, name, json!({"properties": {"parts": parts}}), vec![
            ("A", json!({"parts": [{"color": "red"}, {"color": "blue", "v": [1.0, 0.0]}]})),
            ("B", json!({"parts": [{"color": "red", "v": [3.0, 0.0]}]})),
        ]);
        let got = ids(&node, name, json!({"knn": {"field": "parts.v", "query_vector": [1.0, 0.0], "k": 2, "num_candidates": 10, "filter": {"term": {"parts.color": "red"}}}, "size": 10, "_source": false}));
        if flag.is_some() {
            assert_eq!(got, vec!["A", "B"], "{name}: A has a red part copied to it, its nearest vector ranks first");
        } else {
            assert_eq!(got, vec!["B"], "{name}: without copying, the filter restricts the vector-bearing parts");
        }
    }
}
