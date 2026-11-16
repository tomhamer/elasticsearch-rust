//! Compound retrievers (`rrf`) on a two-node cluster:
//!
//! * a rank doc is its index, `_id` and routing: two documents may share an
//!   `_id` under different routings on different shards, and the ranking
//!   returns exactly the one its children selected (production review #56);
//! * a child search that misses a shard is not hidden: a strict request
//!   (`allow_partial_search_results=false`) fails, a partial one reports
//!   the omission in its `_shards` (review #57). The failing shard fails
//!   only while the children collect candidates (the final phase does not
//!   run the failing query).
mod common;
use common::{start, Cluster};
use reqwest::Method;
use serde_json::{json, Value};

/// Two routing values whose documents land on different shards of the
/// two-shard index `index`: probe docs are indexed with routings `r0..`,
/// then each shard is searched alone (`preference=_shards:n`).
async fn routings_on_two_shards(cl: &Cluster, index: &str) -> (String, String) {
    let mut nd = String::new();
    for i in 0..20 {
        nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("probe-{i}"), "routing": format!("r{i}")}}), json!({"probe": true})));
    }
    let r = cl.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", cl.p.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
    let mut on = [None, None];
    for (shard, slot) in on.iter_mut().enumerate() {
        let (st, v) = cl.call(0, Method::POST, &format!("/{index}/_search?preference=_shards:{shard}&size=20"), Some(json!({"query": {"exists": {"field": "probe"}}}))).await;
        assert_eq!(st, 200, "{v}");
        *slot = v["hits"]["hits"].as_array().unwrap().first().and_then(|h| h["_routing"].as_str()).map(str::to_string);
    }
    let [Some(a), Some(b)] = on else { panic!("probe routings did not reach both shards: {on:?}") };
    assert_ne!(a, b);
    // The probes match nothing the tests search.
    let (st, v) = cl.call(0, Method::POST, &format!("/{index}/_delete_by_query?refresh=true"), Some(json!({"query": {"exists": {"field": "probe"}}}))).await;
    assert_eq!(st, 200, "{v}");
    (a, b)
}

async fn put(cl: &Cluster, index: &str, id: &str, routing: &str, doc: Value) {
    let (st, v) = cl.call(0, Method::PUT, &format!("/{index}/_doc/{id}?routing={routing}&refresh=true"), Some(doc)).await;
    assert!(st == 200 || st == 201, "{v}");
}

fn rrf(children: Vec<Value>) -> Value {
    json!({"rrf": {"retrievers": children.into_iter().map(|q| json!({"standard": {"query": q}})).collect::<Vec<_>>()}})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ranking_keeps_documents_with_one_id_on_different_shards_apart() {
    let cl = start(&[]).await;
    let (st, v) = cl.call(0, Method::PUT, "/ri", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    cl.call(0, Method::GET, "/_cluster/health/ri?wait_for_status=green&timeout=60s", None).await;
    let (ra, rb) = routings_on_two_shards(&cl, "ri").await;
    put(&cl, "ri", "dup", &ra, json!({"text": "apple"})).await;
    put(&cl, "ri", "dup", &rb, json!({"text": "banana"})).await;
    for node in 0..2 {
        // Both children select only the `apple` doc (routing `ra`).
        let (st, v) = cl.call(node, Method::POST, "/ri/_search", Some(json!({
            "retriever": rrf(vec![json!({"match": {"text": "apple"}}), json!({"match": {"text": "apple"}})])}))).await;
        assert_eq!(st, 200, "{v}");
        let hits = v["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "exactly the selected document: {v}");
        assert_eq!(hits[0]["_id"], "dup", "{v}");
        assert_eq!(hits[0]["_routing"], ra.as_str(), "{v}");
        assert_eq!(hits[0]["_source"]["text"], "apple", "{v}");

        // One child each: two documents, each with its own source and score.
        let (st, v) = cl.call(node, Method::POST, "/ri/_search", Some(json!({
            "retriever": rrf(vec![json!({"match": {"text": "apple"}}), json!({"match": {"text": "banana"}})])}))).await;
        assert_eq!(st, 200, "{v}");
        let hits = v["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 2, "{v}");
        let mut seen: Vec<(String, String)> = hits.iter().map(|h| (h["_routing"].as_str().unwrap().to_string(), h["_source"]["text"].as_str().unwrap().to_string())).collect();
        seen.sort();
        let mut want = vec![(ra.clone(), "apple".to_string()), (rb.clone(), "banana".to_string())];
        want.sort();
        assert_eq!(seen, want, "{v}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_child_search_missing_a_shard_fails_strict_and_is_reported_partial() {
    let cl = start(&[]).await;
    let (st, v) = cl.call(0, Method::PUT, "/rp", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"n": {"type": "integer"}, "k": {"type": "keyword"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    cl.call(0, Method::GET, "/_cluster/health/rp?wait_for_status=green&timeout=60s", None).await;
    let (ra, rb) = routings_on_two_shards(&cl, "rp").await;
    for i in 1..=3 {
        put(&cl, "rp", &format!("a{i}"), &ra, json!({"k": "x", "n": i})).await;
        // No `n`: the scoring script fails on this shard only.
        put(&cl, "rp", &format!("b{i}"), &rb, json!({"k": "x"})).await;
    }
    // The failing child runs only while candidates are collected: the
    // final phase looks up rank docs and (no total tracked) runs no child
    // query, so that shard answers it well.
    let body = json!({"track_total_hits": false, "retriever": rrf(vec![
        json!({"script_score": {"query": {"term": {"k": "x"}}, "script": {"source": "doc['n'].value"}}}),
        json!({"term": {"k": "x"}})])});
    for node in 0..2 {
        let (st, v) = cl.call(node, Method::POST, "/rp/_search?allow_partial_search_results=false", Some(body.clone())).await;
        assert_ne!(st, 200, "a strict request must not hide the child's missing shard: {v}");

        let (st, v) = cl.call(node, Method::POST, "/rp/_search?allow_partial_search_results=true", Some(body.clone())).await;
        assert_eq!(st, 200, "{v}");
        assert!(!v["hits"]["hits"].as_array().unwrap().is_empty(), "{v}");
        assert!(v["_shards"]["failed"].as_u64().unwrap_or(0) >= 1, "the omission is reported: {v}");
        let failures = v["_shards"]["failures"].as_array().cloned().unwrap_or_default();
        assert!(failures.iter().any(|f| f["index"] == "rp"), "{v}");
        assert_eq!(v["_shards"]["successful"].as_u64().unwrap() + v["_shards"]["failed"].as_u64().unwrap() <= v["_shards"]["total"].as_u64().unwrap(), true, "{v}");
    }
}
