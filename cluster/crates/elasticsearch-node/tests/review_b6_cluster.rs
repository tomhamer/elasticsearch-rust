//! Production review batch 6 on node processes ([`common::procs`]):
//! - #436: an operator's `allocate_stale_primary` starts the stale copy
//!   (it was refused as "not an in-sync copy");
//! - #437: `allocate_empty_primary` on a node holding a stale copy starts
//!   an empty primary (it reopened the old documents);
//! - #450: a relocation whose target fails after the source handed off is
//!   cancelled, and the source's writes recover (they waited forever).
mod common;
use common::procs::{start_procs, start_procs_env};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn nd(lines: &[Value]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

async fn count(pl: &common::procs::Procs, via: usize, index: &str) -> u64 {
    let _ = pl.call(via, Method::POST, &format!("/{index}/_refresh"), None).await;
    let (st, v) = pl.call(via, Method::GET, &format!("/{index}/_count"), None).await;
    assert_eq!(st, 200, "{v}");
    v["count"].as_u64().unwrap_or(u64::MAX)
}

async fn reroute(pl: &common::procs::Procs, via: usize, cmd: Value) {
    // No retry: the master validates the command against the state its
    // task runs on, never a lagging applied view (it answered "primary is
    // already assigned" for a primary that had left with its node, and
    // "no matching nodes" for a node that had just joined).
    let (st, v) = pl.call(via, Method::POST, "/_cluster/reroute", Some(json!({"commands": [cmd]}))).await;
    if st != 200 {
        let (_, shards) = pl.call(via, Method::GET, "/_cat/shards?format=json&h=index,shard,prirep,state,node,unassigned.reason", None).await;
        panic!("{cmd}: {st} {v}; shards {shards}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forced_stale_and_empty_primaries_start() {
    let mut pl = start_procs(5, &[]).await;
    let via = 0;
    // The copies live on n1 (primary) and n2 (replica) only.
    let (st, v) = pl.call(via, Method::PUT, "/st", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1,
        "index.routing.allocation.include._name": "n1,n2", "index.unassigned.node_left.delayed_timeout": "0"}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "st", "green", 120).await;
    let rows = pl.shards(via, "st").await;
    let primary_node = rows.iter().find(|r| r["prirep"] == "p").and_then(|r| r["node"].as_str()).unwrap().trim_start_matches('n').parse::<usize>().unwrap();
    let replica_node = if primary_node == 1 { 2 } else { 1 };
    let docs: Vec<Value> = (0..5).flat_map(|i| [json!({"index": {"_index": "st", "_id": format!("d{i}")}}), json!({"n": i})]).collect();
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&docs)).await;
    assert_eq!((st, v["errors"].as_bool()), (200, Some(false)), "{v}");
    // The replica goes away; a later write leaves its copy stale.
    pl.stop(replica_node, 60).await;
    pl.wait_health(via, "st", "yellow", 120).await;
    let (st, v) = pl.call(via, Method::PUT, "/st/_doc/late?refresh=true", Some(json!({"n": 99}))).await;
    assert_eq!(st, 201, "{v}");
    // The primary goes too; the stale copy comes back alone: red.
    pl.stop(primary_node, 60).await;
    // The primary's node is gone before the stale copy's node is back (no
    // replica recovery toward it starts).
    pl.wait_nodes(via, 3, 120).await;
    pl.spawn(replica_node, &[]).await;
    pl.wait_nodes(via, 4, 120).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, h) = pl.call(via, Method::GET, "/_cluster/health/st", None).await;
        if h["status"] == "red" {
            break;
        }
        assert!(Instant::now() < deadline, "st did not go red: {h}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // #436: the operator accepts the loss of `late`.
    reroute(&pl, via, json!({"allocate_stale_primary": {"index": "st", "shard": 0, "node": format!("n{replica_node}"), "accept_data_loss": true}})).await;
    pl.wait_health(via, "st", "yellow", 120).await;
    assert_eq!(count(&pl, via, "st").await, 5, "the stale copy's documents, without the lost write");
    let (st, v) = pl.call(via, Method::PUT, "/st/_doc/after?refresh=true", Some(json!({"n": 7}))).await;
    assert_eq!(st, 201, "the forced stale primary takes writes: {v}");
    // #437: now that copy goes away and the old primary's (stale) copy
    // comes back alone; the operator asks for an empty primary there.
    pl.stop(replica_node, 60).await;
    pl.wait_nodes(via, 3, 120).await;
    pl.spawn(primary_node, &[]).await;
    pl.wait_nodes(via, 4, 120).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, h) = pl.call(via, Method::GET, "/_cluster/health/st", None).await;
        if h["status"] == "red" {
            break;
        }
        assert!(Instant::now() < deadline, "st did not go red again: {h}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    reroute(&pl, via, json!({"allocate_empty_primary": {"index": "st", "shard": 0, "node": format!("n{primary_node}"), "accept_data_loss": true}})).await;
    pl.wait_health(via, "st", "yellow", 120).await;
    assert_eq!(count(&pl, via, "st").await, 0, "an empty primary, not the old store's documents");
    let (st, v) = pl.call(via, Method::GET, "/st/_doc/late", None).await;
    assert_eq!(st, 404, "the abandoned document is not republished: {v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_relocation_after_handoff_recovers_writes() {
    let pl = start_procs_env(3, &[], &[("RUSTIC_TEST_FAIL_RELOCATION_FINALIZE", "rel")]).await;
    let via = 0;
    let (st, v) = pl.call(via, Method::PUT, "/rel", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "rel", "green", 120).await;
    let docs: Vec<Value> = (0..20).flat_map(|i| [json!({"index": {"_index": "rel", "_id": format!("d{i}")}}), json!({"n": i})]).collect();
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&docs)).await;
    assert_eq!((st, v["errors"].as_bool()), (200, Some(false)), "{v}");
    let rows = pl.shards(via, "rel").await;
    let from = rows[0]["node"].as_str().unwrap().to_string();
    let to = if from == "n1" { "n2" } else { "n1" };
    // The target fails after the source handed off (seam): the master
    // cancels the relocation back to the source.
    reroute(&pl, via, json!({"move": {"index": "rel", "shard": 0, "from_node": from, "to_node": to}})).await;
    // Writes recover without a restart (they used to time out with 503
    // forever: the source stayed write-fenced).
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut last = Value::Null;
    loop {
        let (st, v) = pl.call(via, Method::PUT, "/rel/_doc/after?refresh=true&timeout=5s", Some(json!({"n": 1}))).await;
        if st == 201 || st == 200 {
            break;
        }
        last = v;
        assert!(Instant::now() < deadline, "writes never recovered: {last}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    pl.wait_health(via, "rel", "green", 120).await;
    assert_eq!(count(&pl, via, "rel").await, 21, "every accepted document is on the elected copy");
}

/// #485: a cluster clone copies documents through `_source`: their
/// embeddings come along (the index default leaves them out of `_source`)
/// and the target answers kNN for every document; a field the mapping
/// leaves out of `_source` is copied from the indexing input.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_resize_keeps_vectors_and_filtered_source_fields() {
    let pl = start_procs(3, &[]).await;
    let via = 0;
    let (st, v) = pl.call(via, Method::PUT, "/vx", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0},
        "mappings": {"properties": {"emb": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm"}, "code": {"type": "keyword"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "vx", "green", 120).await;
    let docs: Vec<Value> = (0..30).flat_map(|i| [json!({"index": {"_index": "vx", "_id": format!("d{i}")}}), json!({"emb": [i as f32, 1.0, 0.5], "code": format!("c{i}")})]).collect();
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&docs)).await;
    assert_eq!((st, v["errors"].as_bool()), (200, Some(false)), "{v}");
    let (st, v) = pl.call(via, Method::PUT, "/vx/_block/write", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(via, Method::POST, "/vx/_clone/vy", None).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "vy", "green", 120).await;
    assert_eq!(count(&pl, via, "vy").await, 30);
    let (st, v) = pl.call(via, Method::POST, "/vy/_search", Some(json!({"size": 30, "_source": {"exclude_vectors": false}}))).await;
    assert_eq!(st, 200, "{v}");
    let with_vectors = v["hits"]["hits"].as_array().map(|a| a.iter().filter(|h| h["_source"]["emb"].as_array().is_some_and(|e| e.len() == 3)).count()).unwrap_or(0);
    assert_eq!(with_vectors, 30, "every copied document kept its vector: {v}");
    let (st, v) = pl.call(via, Method::POST, "/vy/_search", Some(json!({"knn": {"field": "emb", "query_vector": [7.0, 1.0, 0.5], "k": 1, "num_candidates": 30}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_id"], "d7", "{v}");
    // Filtered source: the excluded field is copied from the indexing
    // input and still matches on the clone (and a split).
    let (st, v) = pl.call(via, Method::PUT, "/fx", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"_source": {"excludes": ["code"]}, "properties": {"code": {"type": "keyword"}, "n": {"type": "integer"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "fx", "green", 120).await;
    let docs: Vec<Value> = (0..20).flat_map(|i| [json!({"index": {"_index": "fx", "_id": format!("f{i}")}}), json!({"code": format!("k{i}"), "n": i})]).collect();
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&docs)).await;
    assert_eq!((st, v["errors"].as_bool()), (200, Some(false)), "{v}");
    let (st, v) = pl.call(via, Method::PUT, "/fx/_block/write", None).await;
    assert_eq!(st, 200, "{v}");
    for (api, target, shards) in [("_clone", "fy", None), ("_split", "fz", Some(4))] {
        let body = shards.map(|n: u32| json!({"settings": {"index.number_of_shards": n}}));
        let (st, v) = pl.call(via, Method::POST, &format!("/fx/{api}/{target}"), body).await;
        assert_eq!(st, 200, "{api}: {v}");
        pl.wait_health(via, target, "green", 120).await;
        assert_eq!(count(&pl, via, target).await, 20, "{api}");
        let (st, v) = pl.call(via, Method::POST, &format!("/{target}/_search"), Some(json!({"query": {"term": {"code": "k7"}}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["hits"]["total"]["value"], 1, "{api}: the field left out of _source still matches: {v}");
        assert!(v["hits"]["hits"][0]["_source"].get("code").is_none(), "{api}: the target's _source keeps the exclusion: {v}");
    }
}

fn find_named(dir: &std::path::Path, name: &str) -> bool {
    std::fs::read_dir(dir).into_iter().flatten().flatten().any(|e| {
        let p = e.path();
        e.file_name().to_string_lossy() == name || (p.is_dir() && find_named(&p, name))
    })
}

/// #436 follow-up: the stale copy's node returns while the primary is
/// still up; its replica file recovery sets the old store aside
/// (`shard-0.reuse`) and fails (seam). The primary then goes: the forced
/// stale primary still takes that set-aside copy.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forced_stale_primary_takes_a_set_aside_copy() {
    let trigger = std::env::temp_dir().join(format!("rustic-b6b-fail-file-recovery-{}", std::process::id()));
    let _ = std::fs::remove_file(&trigger);
    let mut pl = start_procs_env(5, &[], &[("RUSTIC_TEST_FAIL_FILE_RECOVERY", trigger.to_str().unwrap())]).await;
    let via = 0;
    let (st, v) = pl.call(via, Method::PUT, "/sa", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.include._name": "n1,n2", "index.unassigned.node_left.delayed_timeout": "0"}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "sa", "green", 120).await;
    let rows = pl.shards(via, "sa").await;
    let primary_node = rows.iter().find(|r| r["prirep"] == "p").and_then(|r| r["node"].as_str()).unwrap().trim_start_matches('n').parse::<usize>().unwrap();
    let replica_node = if primary_node == 1 { 2 } else { 1 };
    let docs: Vec<Value> = (0..5).flat_map(|i| [json!({"index": {"_index": "sa", "_id": format!("d{i}")}}), json!({"n": i})]).collect();
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", nd(&docs)).await;
    assert_eq!((st, v["errors"].as_bool()), (200, Some(false)), "{v}");
    let (st, v) = pl.call(via, Method::PUT, "/sa/_settings", Some(json!({"index.number_of_replicas": 1}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "sa", "green", 120).await;
    // From now on the replica's file recoveries fail after the set-aside.
    std::fs::write(&trigger, "sa").unwrap();
    pl.stop(replica_node, 60).await;
    pl.wait_health(via, "sa", "yellow", 120).await;
    let (st, v) = pl.call(via, Method::PUT, "/sa/_doc/late?refresh=true", Some(json!({"n": 99}))).await;
    assert_eq!(st, 201, "{v}");
    // The replica's copy is stale: out of the in-sync set.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, m) = pl.call(via, Method::GET, "/_cluster/state/metadata/sa", None).await;
        if m["metadata"]["indices"]["sa"]["in_sync_allocations"]["0"].as_array().is_some_and(|a| a.len() == 1) {
            break;
        }
        assert!(Instant::now() < deadline, "the replica stayed in sync: {m}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // Back while the primary is up: its file recovery sets the store aside
    // and fails.
    pl.spawn(replica_node, &[]).await;
    let data = pl.log_path(replica_node).with_file_name(format!("n{replica_node}"));
    let deadline = Instant::now() + Duration::from_secs(60);
    while !find_named(&data, "shard-0.reuse") {
        assert!(Instant::now() < deadline, "the replica recovery never set the old store aside");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    pl.stop(primary_node, 60).await;
    pl.wait_nodes(via, 4, 120).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, h) = pl.call(via, Method::GET, "/_cluster/health/sa", None).await;
        if h["status"] == "red" && h["initializing_shards"] == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "sa did not go red: {h}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    reroute(&pl, via, json!({"allocate_stale_primary": {"index": "sa", "shard": 0, "node": format!("n{replica_node}"), "accept_data_loss": true}})).await;
    pl.wait_health(via, "sa", "yellow", 120).await;
    assert_eq!(count(&pl, via, "sa").await, 5, "the set-aside stale copy's documents");
    let (st, _) = pl.call(via, Method::GET, "/sa/_doc/late", None).await;
    assert_eq!(st, 404);
    let _ = std::fs::remove_file(&trigger);
}
