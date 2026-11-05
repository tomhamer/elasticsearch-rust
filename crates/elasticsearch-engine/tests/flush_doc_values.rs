//! Doc values survive `_flush` (memtable -> sealed segment) and merges:
//! `exists` on geo_point (latitude column presence), geo queries on a
//! doc-values-only geo_point, and float / half_float ranges must not match
//! a doc that has no value as if it were 0. Queries run with docs spread
//! over flushed segments and the memtable, then again after a force merge.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

fn search(node: &Arc<Node>, index: &str, body: Value) -> BTreeSet<String> {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {e:?}"));
    let r: Value = serde_json::from_str(&out.json).unwrap();
    r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

fn put(node: &Arc<Node>, id: &str, src: Value) {
    let svc = node.get_index("t").unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: Default::default() }], false);
    r.into_iter().next().unwrap().unwrap();
}

fn set(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|s| s.to_string()).collect()
}

#[test]
fn doc_values_survive_flush_and_merge() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "t",
        &json!({"mappings": {"dynamic": false, "properties": {
            "geo": {"type": "geo_point"},
            "geo_dv": {"type": "geo_point", "index": false},
            "f": {"type": "float"},
            "h": {"type": "half_float"},
            "n": {"type": "long"},
            "k": {"type": "keyword"}
        }}}),
    )
    .unwrap();
    let flush = || node.get_index("t").unwrap().shard(0).unwrap().shard.flush().unwrap();
    let full = |v: f64| json!({"geo": "41.12,-71.34", "geo_dv": "41.12,-71.34", "f": v, "h": v, "n": v as i64, "k": "x"});
    // Segment 1: one full doc and one without the fields.
    put(&node, "1", full(1.0));
    put(&node, "2", json!({"k": "x"}));
    flush();
    // Segment 2: a multi-valued doc (the columns get promoted) and a full one.
    put(&node, "3", json!({"geo": ["41.12,-71.34", "1,1"], "geo_dv": "41.12,-71.34", "f": [-1.0, -2.0], "h": [-1.0, -2.0], "n": [-1, -2], "k": "x"}));
    put(&node, "4", full(2.0));
    flush();
    // Memtable: another doc without the fields.
    put(&node, "5", json!({"k": "x"}));

    let check = |stage: &str| {
        // Searches see the shard as of its last refresh (ES NRT).
        node.get_index("t").unwrap().refresh().unwrap();
        let with = set(&["1", "3", "4"]);
        for f in ["geo", "geo_dv", "f", "h", "n"] {
            assert_eq!(search(&node, "t", json!({"query": {"exists": {"field": f}}})), with, "{stage}: exists {f}");
        }
        assert_eq!(search(&node, "t", json!({"query": {"exists": {"field": "k"}}})).len(), 5, "{stage}");
        for f in ["f", "h", "n"] {
            // 0 is inside both ranges: docs 2 and 5 have no value and must not match.
            assert_eq!(search(&node, "t", json!({"query": {"range": {f: {"gte": -10, "lte": 10}}}})), with, "{stage}: {f} gte/lte");
            assert_eq!(search(&node, "t", json!({"query": {"range": {f: {"gt": -1.5, "lt": 1.5}}}})), set(&["1", "3"]), "{stage}: {f} gt/lt");
        }
        let bbox = |f: &str| json!({"query": {"geo_bounding_box": {f: {"top_left": {"lat": 42, "lon": -72}, "bottom_right": {"lat": 40, "lon": -70}}}}});
        assert_eq!(search(&node, "t", bbox("geo")), with, "{stage}: geo bbox");
        assert_eq!(search(&node, "t", bbox("geo_dv")), with, "{stage}: doc-values-only geo bbox");
    };
    check("flushed");
    flush();
    check("all flushed");
    node.get_index("t").unwrap().shard(0).unwrap().shard.force_merge(1, false).unwrap();
    check("merged");
}
