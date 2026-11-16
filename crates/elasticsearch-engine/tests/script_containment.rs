//! A search `timeout` stops scripts whose work is in enhanced `for` loops
//! or branching recursion (no ordinary loop to poll from), and the search
//! worker is promptly available again (production review #15).

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn search_json(node: &Node, body: Value) -> Result<Value, String> {
    let req = search::parse_request(&body, &HashMap::new()).map_err(|e| e.to_string())?;
    let targets = node.resolve("sc", Default::default()).map_err(|e| e.to_string())?;
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).map_err(|e| e.to_string())?;
    Ok(serde_json::from_str(&out.json).unwrap())
}

#[test]
fn search_timeout_stops_enhanced_loops_and_recursion_promptly() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("sc", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}}}})).unwrap();
    let svc = node.get_index("sc").unwrap();
    let reqs: Vec<WriteReq> = (0..4).map(|i| WriteReq::Index { id: Some(format!("d{i}")), source: json!({"n": i}).to_string(), routing: None, cond: Default::default() }).collect();
    assert!(svc.write(reqs, false).iter().all(Result::is_ok));
    svc.refresh().unwrap();
    let loops = "List l = new ArrayList(); for (int i = 0; i < 1500; i++) { l.add(i); } long s = 0; for (def a : l) { for (def b : l) { for (def c : l) { s += a + b + c; } } }";
    let recursion = "long f(int n) { if (n < 2) { return n; } return f(n - 1) + f(n - 2); } long s = f(70);";
    for body in [loops, recursion] {
        for query in [
            json!({"script_score": {"query": {"match_all": {}}, "script": {"source": format!("{body} return s;")}}}),
            json!({"bool": {"filter": {"script": {"script": {"source": format!("{body} return s > 0;")}}}}}),
        ] {
            let started = Instant::now();
            let out = search_json(&node, json!({"timeout": "100ms", "query": query}));
            let took = started.elapsed();
            assert!(took < Duration::from_secs(3), "search held its worker for {took:?}: {query}");
            match out {
                Ok(v) => assert_eq!(v["timed_out"], json!(true), "{v}"),
                Err(e) => assert!(e.to_lowercase().contains("time"), "{e}"),
            }
        }
    }
    // The workers are free: a cheap search right after completes at once.
    let started = Instant::now();
    let out = search_json(&node, json!({"query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "doc['n'].value + 1"}}}})).unwrap();
    assert_eq!(out["hits"]["total"]["value"], json!(4));
    assert!(started.elapsed() < Duration::from_secs(2));
}
