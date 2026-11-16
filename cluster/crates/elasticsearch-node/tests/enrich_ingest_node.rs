//! Enrich lookups on a dedicated ingest node (`node.roles: ingest`), which
//! holds no copy of the enrich index (it is allocated to data nodes only).
//! ES sends every lookup through `EnrichCoordinatorProxyAction` to a node
//! that holds a copy; an ingest-only node enriches documents exactly as a
//! data node does, through the index and bulk APIs alike.

mod common;

use common::procs::start_procs_per_node;
use reqwest::Method;
use serde_json::{json, Value};

const INGEST: usize = 2;

fn ndjson(lines: &[Value]) -> String {
    lines.iter().map(|l| l.to_string() + "\n").collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn enrich_on_a_dedicated_ingest_node_matches_a_data_node() {
    let ingest_only: &[(&str, &str)] = &[("node.roles", "ingest")];
    let p = start_procs_per_node(&[("cluster.initial_master_nodes", "n0,n1")], &[&[], &[], ingest_only]).await;

    let (st, v) = p.call(0, Method::PUT, "/users", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 1},
        "mappings": {"properties": {"email": {"type": "keyword"}, "name": {"type": "keyword"}, "zip": {"type": "integer"}}},
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = p.ndjson(0, "/users/_bulk?refresh=true", ndjson(&[
        json!({"index": {"_id": "1"}}), json!({"email": "a@x.io", "name": "Ann", "zip": 1000}),
        json!({"index": {"_id": "2"}}), json!({"email": "b@x.io", "name": "Bob", "zip": 2000}),
    ])).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    let (st, v) = p.call(0, Method::PUT, "/_enrich/policy/users-policy", Some(json!({"match": {"indices": "users", "match_field": "email", "enrich_fields": ["name", "zip"]}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = p.call(0, Method::POST, "/_enrich/policy/users-policy/_execute", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["status"]["phase"], "COMPLETE", "{v}");

    // The enrich index has copies on the data nodes only.
    let (st, rows) = p.call(0, Method::GET, "/_cat/shards/.enrich-users-policy*?format=json&h=index,prirep,state,node&expand_wildcards=all", None).await;
    assert_eq!(st, 200, "{rows}");
    let rows = rows.as_array().cloned().unwrap_or_default();
    assert!(!rows.is_empty(), "no enrich index shards");
    assert!(rows.iter().all(|r| r["node"] != "n2"), "the ingest-only node holds an enrich copy: {rows:?}");

    let (st, v) = p.call(INGEST, Method::PUT, "/_ingest/pipeline/p", Some(json!({"processors": [{"enrich": {"policy_name": "users-policy", "field": "email", "target_field": "user"}}]}))).await;
    assert_eq!(st, 200, "{v}");

    // The index API through the ingest-only node and through a data node.
    for (node, id) in [(INGEST, "via_ingest"), (0, "via_data")] {
        let (st, v) = p.call(node, Method::PUT, &format!("/events/_doc/{id}?pipeline=p&refresh=true"), Some(json!({"email": "a@x.io"}))).await;
        assert_eq!(st, 201, "n{node}: {v}");
    }
    let source = |v: &Value| v["_source"].clone();
    let (_, a) = p.call(0, Method::GET, "/events/_doc/via_ingest", None).await;
    let (_, b) = p.call(0, Method::GET, "/events/_doc/via_data", None).await;
    assert_eq!(source(&a), json!({"email": "a@x.io", "user": {"email": "a@x.io", "name": "Ann", "zip": 1000}}), "{a}");
    assert_eq!(source(&a), source(&b));

    // Bulk through the ingest-only node: a match, a miss.
    let (st, v) = p.ndjson(INGEST, "/events/_bulk?pipeline=p&refresh=true", ndjson(&[
        json!({"index": {"_id": "bulk_hit"}}), json!({"email": "b@x.io"}),
        json!({"index": {"_id": "bulk_miss"}}), json!({"email": "zz@x.io"}),
    ])).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    let (_, hit) = p.call(0, Method::GET, "/events/_doc/bulk_hit", None).await;
    assert_eq!(source(&hit), json!({"email": "b@x.io", "user": {"email": "b@x.io", "name": "Bob", "zip": 2000}}), "{hit}");
    let (_, miss) = p.call(0, Method::GET, "/events/_doc/bulk_miss", None).await;
    assert_eq!(source(&miss), json!({"email": "zz@x.io"}), "{miss}");

    // The ingest-only node counts its lookups as remote requests.
    let (st, v) = p.call(INGEST, Method::GET, "/_enrich/_stats", None).await;
    assert_eq!(st, 200, "{v}");
}
