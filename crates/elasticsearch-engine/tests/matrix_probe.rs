//! Matrix mapping coverage and multi-index document statistics regressions.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;

fn put(node: &Node, index: &str, id: &str, src: &Value) {
    let svc = node.get_index(index).unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: Default::default() }], true);
    r.into_iter().next().unwrap().unwrap();
    // `write(..., true)` syncs the WAL; searches require an explicit refresh.
    svc.refresh().unwrap();
}

fn search(node: &Node, index: &str, body: Value) -> Value {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {e:?}"));
    serde_json::from_str(&out.json).unwrap()
}

#[test]
fn probe_matrix_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test1", &json!({"mappings": {"properties": {"value1": {"type": "double"}, "value2": {"type": "double"}, "value3": {"type": "double"}}}})).unwrap();
    node.create_index("test2", &json!({"mappings": {"properties": {"value1": {"type": "double"}, "value2": {"type": "double"}, "value4": {"type": "double"}}}})).unwrap();
    for (i, v) in [("1", 1.0), ("2", 2.0), ("3", 3.0)] {
        put(&node, "test1", i, &json!({"value1": v, "value2": v * 10.0, "value3": v * 100.0}));
    }
    for (i, v) in [("1", 1.0), ("2", 2.0), ("3", 3.0)] {
        put(&node, "test2", i, &json!({"value1": v, "value2": v * 10.0, "value4": v * 100.0}));
    }
    let body = json!({"size": 0, "aggs": {"stats": {"matrix_stats": {"fields": ["value1", "value2"]}}}});
    let resp = search(&node, "test1,test2", body);
    eprintln!("HITS: {}", serde_json::to_string(&resp["hits"]).unwrap());
    eprintln!("AGGS: {}", serde_json::to_string(&resp["aggregations"]).unwrap());
    assert_eq!(resp["aggregations"]["stats"]["doc_count"], 6);
    // Now the bad-request case:
    let body2 = json!({"size": 0, "aggs": {"stats": {"matrix_stats": {"fields": ["value3", "value4"]}}}});
    let req2 = search::parse_request(&body2, &HashMap::new()).unwrap();
    let targets2 = node.resolve("test1,test2", Default::default()).unwrap();
    let opts2 = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let error = search::execute(&targets2, &req2, &opts2).err().expect("different index mappings must reject incomplete matrix fields");
    assert_eq!(error.status, 400);
    assert!(format!("{error:?}").contains("all fields must exist in all indices"));
}

#[test]
fn empty_mapped_index_preserves_matrix_mapping_coverage() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    for name in ["fulla","fullb","empty"] {
        node.create_index(name, &json!({"mappings":{"properties":{"x":{"type":"double"},"y":{"type":"double"}}}})).unwrap();
    }
    put(&node,"fulla","1",&json!({"x":1.0,"y":2.0}));
    put(&node,"fullb","1",&json!({"x":3.0,"y":6.0}));
    let query=json!({"size":0,"aggs":{"m":{"matrix_stats":{"fields":["x","y"]}}}});
    let response=search(&node,"fulla,fullb,empty",query.clone());
    assert_eq!(response["aggregations"]["m"]["doc_count"],2);
    node.create_index("missing",&json!({"mappings":{"properties":{"x":{"type":"double"}}}})).unwrap();
    let req=search::parse_request(&query,&HashMap::new()).unwrap();
    let targets=node.resolve("fulla,fullb,missing",Default::default()).unwrap();
    let opts=ExecOpts{terms_lookup:None,cursors:None,max_buckets:65_536,scroll_id:None,pit_id:None};
    let error=search::execute(&targets,&req,&opts).err().expect("empty unmapped index must not disappear from coverage");
    assert_eq!(error.status,400);
    assert!(format!("{error:?}").contains("all fields must exist in all indices"));
}
