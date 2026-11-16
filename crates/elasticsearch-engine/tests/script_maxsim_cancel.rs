//! SIDE 013: `maxSimDotProduct` / `maxSimInvHamming` charge their query x
//! document vector product to the script fuel, so a search `timeout` stops
//! a large product within its poll interval instead of after the whole
//! per-document computation; small inputs keep their exact results.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn run(node: &Node, body: Value) -> (Result<Value, String>, Duration) {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("mv", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let t = Instant::now();
    let out = search::execute(&targets, &req, &opts).map(|o| serde_json::from_str::<Value>(&o.json).unwrap()).map_err(|e| e.reason);
    (out, t.elapsed())
}

#[test]
fn max_sim_stops_at_the_search_deadline_and_keeps_small_results() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("mv", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"v": {"type": "rank_vectors", "dims": 64}, "s": {"type": "rank_vectors", "dims": 2}}}})).unwrap();
    let svc = node.get_index("mv").unwrap();
    let row = |i: usize| -> Vec<f32> { (0..64).map(|j| ((i * 31 + j * 7) % 17) as f32 / 17.0).collect() };
    let big: Vec<Vec<f32>> = (0..6000).map(row).collect();
    let docs = vec![
        WriteReq::Index { id: Some("big".into()), source: json!({"v": big}).to_string(), routing: None, cond: Default::default() },
        WriteReq::Index { id: Some("small".into()), source: json!({"s": [[1.0, 2.0], [3.0, -1.0]]}).to_string(), routing: None, cond: Default::default() },
    ];
    assert!(svc.write(docs, true).iter().all(Result::is_ok));
    svc.refresh().unwrap();
    // Exact on a small fixture: max(1*1+2*0.5, 3*1-1*0.5) + max(1*0+2*1, 3*0-1) = 2.5 + 2 = 4.5.
    let (small, _) = run(&node, json!({"query": {"script_score": {"query": {"ids": {"values": ["small"]}},
        "script": {"source": "maxSimDotProduct(params.q, 's')", "params": {"q": [[1.0, 0.5], [0.0, 1.0]]}}}}}));
    let small = small.unwrap();
    assert!((small["hits"]["hits"][0]["_score"].as_f64().unwrap() - 4.5).abs() < 1e-6, "{small}");
    // 6000 x 6000 x 64 components for one document: stopped at the timeout.
    let q: Vec<Vec<f32>> = (7000..13000).map(row).collect();
    let (out, took) = run(&node, json!({"timeout": "50ms", "query": {"script_score": {"query": {"ids": {"values": ["big"]}},
        "script": {"source": "maxSimDotProduct(params.q, 'v')", "params": {"q": q}}}}}));
    assert!(took < Duration::from_millis(1500), "the product ran on for {took:?}");
    match out {
        Ok(v) => assert_eq!(v["timed_out"], json!(true), "{}", v.to_string().chars().take(300).collect::<String>()),
        Err(reason) => assert!(reason.contains("Time exceeded") || reason.contains("time"), "{reason}"),
    }
}
