//! Production review findings on three node processes ([`common::procs`]):
//! - #136: quoted `if_seq_no` / `if_primary_term` / `version` in a cluster
//!   `_bulk` (split by `front/bulk.rs`, executed on the primaries) are
//!   enforced, from every coordinator;
//! - #231: `PUT /{index}/_block/write` acknowledges shards once every node
//!   drained its admitted writes, and later writes are refused;
//! - #232: `_reload_search_analyzers` rebuilds the analyzers on every node.
mod common;
use common::procs::start_procs;
use reqwest::Method;
use serde_json::{json, Value};

fn nd(lines: &[Value]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quoted_bulk_conditions_and_block_drain_on_a_cluster() {
    let pl = start_procs(3, &[]).await;
    let master = pl.master(0).await;
    // #136
    let (st, v) = pl.call(master, Method::PUT, "/cas", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "cas", "green", 90).await;
    let (st, v) = pl.call(0, Method::PUT, "/cas/_doc/1?refresh=true", Some(json!({"n": 1}))).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = pl.call(1, Method::PUT, "/cas/_doc/1?refresh=true", Some(json!({"n": 2}))).await;
    assert_eq!(st, 200, "{v}");
    let (seq, term) = (v["_seq_no"].as_i64().unwrap(), v["_primary_term"].as_i64().unwrap());
    for via in 0..3 {
        for (op, body) in [("index", Some(json!({"n": 9}))), ("update", Some(json!({"doc": {"n": 9}}))), ("delete", None)] {
            let mut lines = vec![json!({op: {"_index": "cas", "_id": "1", "if_seq_no": (seq - 1).to_string(), "if_primary_term": term.to_string()}})];
            lines.extend(body);
            let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&lines)).await;
            assert_eq!(st, 200, "{v}");
            assert_eq!(v["items"][0][op]["status"].as_u64(), Some(409), "n{via} {op}: {v}");
        }
    }
    let (st, v) = pl.ndjson(2, "/_bulk?refresh=true", nd(&[json!({"index": {"_index": "cas", "_id": "e", "version": "5", "version_type": "external"}}), json!({"n": 1})])).await;
    assert_eq!((st, v["items"][0]["index"]["_version"].as_u64()), (200, Some(5)), "{v}");
    let (_, v) = pl.ndjson(0, "/_bulk", nd(&[json!({"index": {"_index": "cas", "_id": "e", "version": "4", "version_type": "external"}}), json!({"n": 2})])).await;
    assert_eq!(v["items"][0]["index"]["status"].as_u64(), Some(409), "{v}");
    let (_, v) = pl.ndjson(1, "/_bulk?refresh=true", nd(&[json!({"index": {"_index": "cas", "_id": "1", "if_seq_no": seq.to_string(), "if_primary_term": term.to_string()}}), json!({"n": 3})])).await;
    assert_eq!(v["items"][0]["index"]["status"].as_u64(), Some(200), "current quoted conditions apply: {v}");
    // #241: a scripted reindex delete carries ES's internal version
    // equality through the cluster bulk: newer and older destination copies
    // conflict, an equal one is deleted.
    for (dest, dest_version, deleted) in [("vnew", 4, 0), ("vold", 2, 0), ("veq", 3, 1)] {
        let (st, v) = pl.call(0, Method::PUT, &format!("/{dest}/_doc/1?version={dest_version}&version_type=external&refresh=true"), Some(json!({"n": dest_version}))).await;
        assert_eq!(st, 201, "{v}");
        let (st, v) = pl.call(1, Method::PUT, "/vsrc/_doc/1?version=3&version_type=external&refresh=true", Some(json!({"n": 3}))).await;
        assert!(st == 201 || st == 409, "{v}");
        let (st, v) = pl.call(2, Method::POST, "/_reindex?refresh=true", Some(json!({"conflicts": "proceed", "source": {"index": "vsrc"}, "dest": {"index": dest, "version_type": "external"}, "script": {"source": "ctx.op = 'delete'"}}))).await;
        assert_eq!((st, v["deleted"].as_u64()), (200, Some(deleted)), "{dest}: {v}");
        assert_eq!(v["version_conflicts"].as_u64(), Some(1 - deleted), "{dest}: {v}");
    }
    // #231
    let (st, v) = pl.call(1, Method::PUT, "/cas/_block/write", None).await;
    assert_eq!((st, v["acknowledged"].as_bool(), v["shards_acknowledged"].as_bool()), (200, Some(true), Some(true)), "{v}");
    for via in 0..3 {
        let (st, v) = pl.call(via, Method::PUT, &format!("/cas/_doc/after{via}"), Some(json!({"n": 1}))).await;
        assert_eq!(st, 403, "n{via}: writes are blocked once acknowledged: {v}");
    }
}

/// #232 on a cluster: every node and shard engine rebuilds its analyzers
/// with the published rules; every copy searches with them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn analyzer_reload_on_a_cluster() {
    let pl = start_procs(3, &[]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/_synonyms/vs", Some(json!({"synonyms_set": [{"id": "r", "synonyms": "vehicle => car"}]}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = pl.call(master, Method::PUT, "/syn", Some(json!({
        "settings": {"number_of_shards": 3, "number_of_replicas": 1, "analysis": {"filter": {"f": {"type": "synonym_graph", "synonyms_set": "vs", "updateable": true}},
            "analyzer": {"s": {"tokenizer": "standard", "filter": ["lowercase", "f"]}}}},
        "mappings": {"properties": {"t": {"type": "text", "analyzer": "standard", "search_analyzer": "s"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "syn", "green", 90).await;
    let (st, v) = pl.ndjson(0, "/_bulk?refresh=true", nd(&[json!({"index": {"_index": "syn", "_id": "car"}}), json!({"t": "car"}), json!({"index": {"_index": "syn", "_id": "truck"}}), json!({"t": "truck"})])).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::PUT, "/_synonyms/vs?refresh=false", Some(json!({"synonyms_set": [{"id": "r", "synonyms": "vehicle => truck"}]}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, reload) = pl.call(2, Method::POST, "/syn/_reload_search_analyzers", None).await;
    assert_eq!((st, reload["_shards"]["failed"].as_u64()), (200, Some(0)), "{reload}");
    for via in 0..3 {
        let (st, v) = pl.call(via, Method::POST, "/syn/_search?request_cache=false&preference=_local", Some(json!({"query": {"match": {"t": "vehicle"}}}))).await;
        assert_eq!(st, 200, "{v}");
        let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().filter_map(|h| h["_id"].as_str()).collect();
        let (_, an) = pl.call(via, Method::POST, "/syn/_analyze", Some(json!({"analyzer": "s", "text": "vehicle"}))).await;
        let (_, syn) = pl.call(via, Method::GET, "/_synonyms/vs", None).await;
        assert_eq!(ids, vec!["truck"], "n{via}: {v} (reload: {reload}) (analyze: {an}) (set: {syn})");
    }
}

/// Production review #385: node-removal prevalidation counts a red shard's
/// store on the node's disk even without a routing entry (an unassigned
/// copy on a rejoined node): that node is not safe to remove.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prevalidation_counts_unassigned_stores_on_the_node() {
    let mut pl = start_procs(3, &[]).await;
    let master = pl.master(0).await;
    let holder = (0..3).find(|i| *i != master).unwrap();
    let ids = pl.node_ids(master).await;
    let (_, nodes) = pl.call(master, Method::GET, "/_nodes", None).await;
    let holder_name = nodes["nodes"][&ids[&holder]]["name"].as_str().unwrap().to_string();
    let (st, v) = pl.call(master, Method::PUT, "/solo", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": holder_name}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "solo", "green", 90).await;
    let (st, v) = pl.call(master, Method::PUT, "/solo/_doc/1?refresh=true", Some(json!({"n": 1}))).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = pl.call(master, Method::POST, "/solo/_flush", None).await;
    assert_eq!(st, 200, "{v}");
    pl.settings(master, "persistent", json!({"cluster.routing.allocation.enable": "none"})).await;
    pl.stop(holder, 60).await;
    pl.wait_nodes(master, 2, 60).await;
    pl.spawn(holder, &[]).await;
    pl.wait_nodes(master, 3, 90).await;
    let (_, h) = pl.call(master, Method::GET, "/_cluster/health/solo", None).await;
    assert_eq!(h["status"], json!("red"), "{h}");
    let (st, v) = pl.call(master, Method::POST, &format!("/_internal/prevalidate_node_removal?ids={}", ids[&holder]), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["is_safe"], json!(false), "the node holds the red shard's only store: {v}");
    pl.settings(master, "persistent", json!({"cluster.routing.allocation.enable": null})).await;
}
