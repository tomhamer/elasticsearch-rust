//! Real two-node searches over two single-shard indices (on different
//! nodes, and both on one node): the hit total and every aggregation cover
//! the documents of both indices, and a `terms` on `_index` names each
//! index with its own documents. The fault gate's search checker flags a
//! `jepsen-*` search whose `hits.total` differs from its `stats.count`, and
//! reads `by_index` (terms on `_index`) to name the short index.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

const DOCS_X: u64 = 300;
const DOCS_Y: u64 = 200;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_index_search_aggregates_every_index() {
    let cl = start(&[]).await;
    // `ma-*`: one index on each node; `mb-*`: both on n0.
    for (name, on) in [("ma-x", "n0"), ("ma-y", "n1"), ("mb-x", "n0"), ("mb-y", "n0")] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": on},
            "mappings": {"properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(status, 200, "{body}");
        let docs = if name.ends_with('x') { DOCS_X } else { DOCS_Y };
        let mut bulk = String::new();
        for i in 0..docs {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n"));
            bulk.push_str(&json!({"k": format!("k{}", i % 5), "v": i}).to_string());
            bulk.push('\n');
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", cl.p.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    let total = DOCS_X + DOCS_Y;
    let sum = (0..DOCS_X).sum::<u64>() + (0..DOCS_Y).sum::<u64>();
    let body = json!({"size": 0, "track_total_hits": true, "aggs": {
        "values": {"stats": {"field": "v"}},
        "mean": {"avg": {"field": "v"}},
        "n": {"value_count": {"field": "v"}},
        "keys": {"cardinality": {"field": "k"}},
        "by_k": {"terms": {"field": "k", "size": 10}},
        // The fault gate's diagnostic: `_index` doc counts against each
        // index's own `v` count.
        "by_index": {"terms": {"field": "_index", "size": 10}, "aggs": {"vc": {"value_count": {"field": "v"}}}}
    }});
    for pattern in ["ma-*", "mb-*", "ma-x,ma-y", "mb-x,mb-y"] {
        for node in 0..2 {
            for query in ["", "?allow_partial_search_results=false", "?request_cache=false"] {
                let (status, got) = cl.call(node, Method::POST, &format!("/{pattern}/_search{query}"), Some(body.clone())).await;
                assert_eq!(status, 200, "{pattern} on n{node}{query}: {got}");
                let what = format!("{pattern} on n{node}{query}: {got}");
                assert_eq!(got["_shards"]["total"], 2, "{what}");
                assert_eq!(got["_shards"]["failed"], 0, "{what}");
                assert_eq!(got["hits"]["total"]["value"], total, "{what}");
                let a = &got["aggregations"];
                assert_eq!(a["values"]["count"], total, "stats.count == hits.total: {what}");
                assert_eq!(a["values"]["sum"].as_f64(), Some(sum as f64), "{what}");
                assert_eq!(a["values"]["min"].as_f64(), Some(0.0), "{what}");
                assert_eq!(a["values"]["max"].as_f64(), Some((DOCS_X - 1) as f64), "{what}");
                assert_eq!(a["n"]["value"], total, "{what}");
                assert!((a["mean"]["value"].as_f64().unwrap() - sum as f64 / total as f64).abs() < 1e-9, "{what}");
                assert_eq!(a["keys"]["value"], 5, "{what}");
                let buckets = a["by_k"]["buckets"].as_array().unwrap();
                assert_eq!(buckets.len(), 5, "{what}");
                assert_eq!(buckets.iter().map(|b| b["doc_count"].as_u64().unwrap()).sum::<u64>(), total, "{what}");
                // One `_index` bucket per concrete index, its documents and
                // their `v` values (ES `IndexFieldMapper` terms).
                let mut by_index: Vec<(String, u64, u64)> = a["by_index"]["buckets"].as_array().unwrap().iter()
                    .map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap(), b["vc"]["value"].as_u64().unwrap())).collect();
                by_index.sort();
                let names: Vec<&str> = pattern.trim_end_matches('*').split(',').collect();
                let (x, y) = if pattern.ends_with('*') { (format!("{}x", names[0]), format!("{}y", names[0])) } else { (names[0].to_string(), names[1].to_string()) };
                assert_eq!(by_index, vec![(x, DOCS_X, DOCS_X), (y, DOCS_Y, DOCS_Y)], "terms on _index: {what}");
                assert_eq!(a["by_index"]["sum_other_doc_count"], 0, "{what}");
            }
        }
    }
}
