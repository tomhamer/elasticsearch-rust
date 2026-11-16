//! Review finding 397: field caps' data-dependent pruning (`index_filter`,
//! `include_empty_fields: false`) over a multi-shard index whose shards sit
//! on different nodes answers as ES does from either node: an index is kept
//! when any shard can match the filter, a field when any shard holds a value.

mod common;

use reqwest::Method;
use serde_json::{json, Value};

fn sorted(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v.as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|x| x.as_str().unwrap().to_string()).collect();
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_shard_pruning_judges_every_shard() {
    let c = common::start(&[]).await;
    let mapping = json!({"properties": {"n": {"type": "long"}, "d": {"type": "date"}, "k": {"type": "keyword"}, "rare": {"type": "keyword"}, "never": {"type": "keyword"}}});
    let (st, v) = c.call(0, Method::PUT, "/fc", Some(json!({"settings": {"number_of_shards": 4, "number_of_replicas": 0}, "mappings": mapping}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = c.call(0, Method::PUT, "/one", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = c.call(0, Method::GET, "/_cluster/health/fc,one?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    // The four shards of `fc` are spread over both nodes.
    let (_, shards) = c.call(0, Method::GET, "/_cat/shards/fc?format=json", None).await;
    let nodes: std::collections::BTreeSet<String> = shards.as_array().unwrap().iter().filter_map(|s| s["node"].as_str().map(str::to_string)).collect();
    assert_eq!(nodes.len(), 2, "{shards}");
    for i in 0..40 {
        let (st, v) = c.call(0, Method::PUT, &format!("/fc/_doc/{i}?routing=r{i}"), Some(json!({"n": i, "k": "x", "d": "2020-01-01T00:00:00Z"}))).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    // One document holds `rare` (and the only large `n`), on one shard.
    let (st, v) = c.call(0, Method::PUT, "/fc/_doc/rare?routing=only", Some(json!({"n": 1000, "rare": "y", "d": "2030-01-01T00:00:00Z"}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = c.call(0, Method::PUT, "/one/_doc/1", Some(json!({"n": 5, "k": "x", "d": "2020-06-01T00:00:00Z"}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = c.call(0, Method::POST, "/fc,one/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    for node in 0..2 {
        let (st, v) = c.call(node, Method::GET, "/fc/_field_caps?fields=n,k,rare,never&include_empty_fields=false", None).await;
        assert_eq!(st, 200, "{v}");
        let fields = v["fields"].as_object().unwrap();
        assert!(fields.contains_key("rare") && fields.contains_key("n") && fields.contains_key("k"), "node {node}: {v}");
        assert!(!fields.contains_key("never"), "node {node}: {v}");
        let (st, v) = c.call(node, Method::GET, "/fc,one/_field_caps?fields=rare,never&include_empty_fields=false", None).await;
        assert_eq!(st, 200, "{v}");
        assert!(v["fields"].get("rare").is_some() && v["fields"].get("never").is_none(), "node {node}: {v}");
        // A date range rewrites against each shard's min / max (ES's
        // can-match): it reaches one shard of `fc` only.
        let (st, v) = c.call(node, Method::POST, "/fc,one/_field_caps?fields=n", Some(json!({"index_filter": {"range": {"d": {"gte": "2029-01-01"}}}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(sorted(&v["indices"]), vec!["fc"], "node {node}: {v}");
        let (st, v) = c.call(node, Method::POST, "/fc,one/_field_caps?fields=n", Some(json!({"index_filter": {"range": {"d": {"gte": "2040-01-01"}}}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(sorted(&v["indices"]), Vec::<String>::new(), "node {node}: {v}");
        let (st, v) = c.call(node, Method::POST, "/fc,one/_field_caps?fields=n", Some(json!({"index_filter": {"range": {"d": {"lte": "2021-01-01"}}}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(sorted(&v["indices"]), vec!["fc", "one"], "node {node}: {v}");
        // A range on a long never rewrites (ES rewrites date ranges only):
        // both indices are kept.
        let (st, v) = c.call(node, Method::POST, "/fc,one/_field_caps?fields=n", Some(json!({"index_filter": {"range": {"n": {"gte": 5000}}}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(sorted(&v["indices"]), vec!["fc", "one"], "node {node}: {v}");
    }
}
