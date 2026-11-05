//! A document of an index with `_source` disabled is replicated whole: the
//! replica re-parses the operation's source into the same Lucene documents
//! (the parent and its nested children), as ES's translog keeps the
//! original source whatever the mapping stores. The replica used to get an
//! empty source and index the parent alone, so a nested query (YAML
//! `search.inner_hits/10_basic.yml`, "Inner hits with disabled _source")
//! found nothing whenever it was served by the replica.

mod common;

use common::start;
use reqwest::Method;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nested_documents_of_a_disabled_source_index_reach_the_replica() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (st, v) = cl.call(0, Method::PUT, "/disabled_source", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1},
        "mappings": {"_source": {"enabled": false}, "properties": {"nested_field": {"type": "nested", "properties": {"sub_nested_field": {"type": "nested"}}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = cl.call(0, Method::GET, "/_cluster/health/disabled_source?wait_for_status=green&timeout=60s", None).await;
    assert_eq!((st, v["status"].clone()), (200, json!("green")), "{v}");
    let (st, v) = cl.call(0, Method::PUT, "/disabled_source/_doc/1", Some(json!({"nested_field": {"field": "value", "sub_nested_field": {"field": "value"}}}))).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = cl.call(0, Method::POST, "/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    // Both copies hold the parent and its two nested documents.
    let (_, shards) = cl.call(0, Method::GET, "/_cat/shards/disabled_source?format=json&h=prirep,docs", None).await;
    let docs: Vec<(String, String)> = shards.as_array().unwrap().iter().map(|s| (s["prirep"].as_str().unwrap().to_string(), s["docs"].as_str().unwrap_or("").to_string())).collect();
    assert_eq!(docs.len(), 2, "{shards}");
    assert!(docs.iter().all(|(_, d)| d == "3"), "copies differ: {docs:?}");
    // The nested query finds the document on each copy (`_local` from each node).
    let q = json!({"query": {"nested": {"path": "nested_field.sub_nested_field", "query": {"match_all": {}}, "inner_hits": {}}}});
    for node in 0..2 {
        let (st, v) = cl.call(node, Method::POST, "/disabled_source/_search?rest_total_hits_as_int=true&preference=_local", Some(q.clone())).await;
        assert_eq!((st, v["hits"]["total"].clone()), (200, json!(1)), "via n{node}: {v}");
        assert_eq!(v["hits"]["hits"][0]["inner_hits"]["nested_field.sub_nested_field"]["hits"]["hits"][0]["_id"], "1", "{v}");
        // Still no `_source` to return, on either copy.
        assert!(v["hits"]["hits"][0].get("_source").is_none(), "{v}");
    }
    // Exercise the perf branch's private append-only writer as well as the
    // explicit-ID/nested path above. WAL source must not become stored source.
    let (st, v) = cl.call(0, Method::PUT, "/disabled_auto", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 1},
        "mappings": {"_source": {"enabled": false}, "properties": {"value": {"type": "keyword"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (_, health) = cl.call(0, Method::GET, "/_cluster/health/disabled_auto?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(health["status"], "green", "{health}");
    let response = cl.c.post(format!("http://127.0.0.1:{}/disabled_auto/_bulk?refresh=true", ports.http[0]))
        .header("Content-Type", "application/x-ndjson")
        .body("{\"index\":{}}\n{\"value\":\"kept\"}\n".repeat(64))
        .send().await.unwrap();
    assert!(response.status().is_success());
    let bulk: serde_json::Value = response.json().await.unwrap();
    assert_eq!(bulk["errors"], false, "{bulk}");
    for node in 0..2 {
        let (st, hits) = cl.call(node, Method::POST, "/disabled_auto/_search?preference=_local", Some(json!({
            "size": 100, "query": {"term": {"value": "kept"}}
        }))).await;
        assert_eq!(st, 200, "{hits}");
        assert_eq!(hits["hits"]["total"]["value"], 64, "{hits}");
        assert!(hits["hits"]["hits"].as_array().unwrap().iter().all(|h| h.get("_source").is_none()), "{hits}");
    }

}
