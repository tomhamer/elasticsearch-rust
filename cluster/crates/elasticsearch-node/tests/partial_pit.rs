//! A PIT opened with `allow_partial_search_results=true` without some of its
//! shards (here: `preference=_only_local` on a node holding half of them)
//! keeps those shards: every later search reports them as failed shards, a
//! search that does not allow partial results is rejected, and they never
//! join the PIT later. A newly opened PIT covers every shard again.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeSet;

const DOCS: usize = 60;

/// The shards named by `_shards.failures` (ES groups failures with the same
/// index and reason, so one entry may stand for several shards).
fn failed_shards(v: &Value) -> BTreeSet<u64> {
    v["_shards"]["failures"].as_array().map(|a| a.iter().filter_map(|f| f["shard"].as_u64()).collect()).unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partial_pit_reports_missing_shards_on_every_page() {
    let cl = start(&[]).await;
    let (status, v) = cl.call(0, Method::PUT, "/pp", Some(json!({
        "settings": {"number_of_shards": 4, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 2},
        "mappings": {"properties": {"n": {"type": "long"}}}
    }))).await;
    assert_eq!(status, 200, "{v}");
    let (_, health) = cl.call(0, Method::GET, "/_cluster/health/pp?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(health["status"], "green", "{health}");
    let mut bulk = String::new();
    for i in 0..DOCS {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{{\"n\":{i}}}\n"));
    }
    let v: Value = cl.c.post(format!("http://127.0.0.1:{}/pp/_bulk?refresh=true", cl.p.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");

    // Which shards each node holds (two each).
    let (_, shards) = cl.call(0, Method::GET, "/_cat/shards/pp?format=json&h=shard,node,state", None).await;
    let on = |node: &str| -> BTreeSet<u64> {
        shards.as_array().unwrap().iter().filter(|s| s["node"] == node && s["state"] == "STARTED").map(|s| s["shard"].as_str().unwrap().parse().unwrap()).collect()
    };
    let (local, remote) = (on("n0"), on("n1"));
    assert_eq!((local.len(), remote.len()), (2, 2), "{shards}");
    // The docs the local shards hold.
    let (status, v) = cl.call(0, Method::POST, "/pp/_search?preference=_only_local", Some(json!({"size": DOCS, "_source": false}))).await;
    assert_eq!(status, 200, "{v}");
    let local_ids: BTreeSet<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    assert!(!local_ids.is_empty() && local_ids.len() < DOCS, "{v}");

    // Without partial results the open is refused.
    let (status, v) = cl.call(0, Method::POST, "/pp/_pit?keep_alive=1m&preference=_only_local", None).await;
    assert_eq!(status, 503, "{v}");

    // Partial open: the remote shards are failures.
    let (status, v) = cl.call(0, Method::POST, "/pp/_pit?keep_alive=1m&preference=_only_local&allow_partial_search_results=true", None).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["_shards"]["total"], 4, "{v}");
    assert_eq!(v["_shards"]["successful"], 2, "{v}");
    assert_eq!(v["_shards"]["failed"], 2, "{v}");
    assert!(!failed_shards(&v).is_empty() && failed_shards(&v).is_subset(&remote), "{v}");
    let partial_pit = v["id"].clone();

    // Every page, from either node and with or without the parameter,
    // reports the shards the PIT misses.
    let mut pit = partial_pit.clone();
    let mut cursor: Option<Value> = None;
    let mut ids = BTreeSet::new();
    for page in 0..40 {
        let mut body = json!({"size": 7, "pit": {"id": pit, "keep_alive": "1m"}, "sort": [{"n": "asc"}], "_source": false});
        if let Some(c) = &cursor { body["search_after"] = c.clone(); }
        let path = if page % 3 == 2 { "/_search?allow_partial_search_results=true" } else { "/_search" };
        let (status, v) = cl.call(page % 2, Method::POST, path, Some(body)).await;
        assert_eq!(status, 200, "page {page}: {v}");
        assert_eq!(v["_shards"]["total"], 4, "page {page}: {v}");
        assert_eq!(v["_shards"]["failed"], 2, "page {page}: {v}");
        assert!(!failed_shards(&v).is_empty() && failed_shards(&v).is_subset(&remote), "page {page}: {v}");
        for f in v["_shards"]["failures"].as_array().unwrap() {
            assert_eq!(f["index"], "pp", "{f}");
            assert_eq!(f["reason"]["type"], "no_shard_available_action_exception", "{f}");
        }
        if let Some(id) = v.get("pit_id") { pit = id.clone(); }
        let hits = v["hits"]["hits"].as_array().unwrap();
        if hits.is_empty() { break; }
        ids.extend(hits.iter().map(|h| h["_id"].as_str().unwrap().to_string()));
        cursor = Some(hits.last().unwrap()["sort"].clone());
    }
    assert_eq!(ids, local_ids, "the PIT holds exactly the local shards' docs");

    // Aggregations and counts over the partial PIT report the same.
    let (status, v) = cl.call(1, Method::POST, "/_search", Some(json!({"size": 0, "pit": {"id": partial_pit}, "aggs": {"s": {"sum": {"field": "n"}}}}))).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["_shards"]["failed"], 2, "{v}");
    assert_eq!(v["hits"]["total"]["value"], local_ids.len(), "{v}");

    // A search that does not allow partial results is rejected.
    for node in 0..2 {
        let (status, v) = cl.call(node, Method::POST, "/_search?allow_partial_search_results=false", Some(json!({"pit": {"id": partial_pit}}))).await;
        assert_eq!(status, 503, "{v}");
        assert!(v.to_string().contains("missing shards"), "{v}");
    }
    // So is one under `search.default_allow_partial_results: false`.
    let (status, _) = cl.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"search.default_allow_partial_results": false}}))).await;
    assert_eq!(status, 200);
    let (status, v) = cl.call(0, Method::POST, "/_search", Some(json!({"pit": {"id": partial_pit}}))).await;
    assert_eq!(status, 503, "{v}");
    let (status, _) = cl.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"search.default_allow_partial_results": null}}))).await;
    assert_eq!(status, 200);

    // The missing shards are reachable again (no preference, the node that
    // holds them): the PIT still never includes them.
    let (status, v) = cl.call(1, Method::POST, "/_search", Some(json!({"size": DOCS, "pit": {"id": partial_pit}, "_source": false}))).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["_shards"]["failed"], 2, "{v}");
    assert_eq!(v["hits"]["total"]["value"], local_ids.len(), "{v}");

    // A newly opened PIT covers every shard.
    let (status, v) = cl.call(0, Method::POST, "/pp/_pit?keep_alive=1m", None).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["_shards"]["failed"], 0, "{v}");
    let full_pit = v["id"].clone();
    for node in 0..2 {
        let (status, v) = cl.call(node, Method::POST, "/_search?allow_partial_search_results=false", Some(json!({"size": DOCS, "pit": {"id": full_pit}, "_source": false}))).await;
        assert_eq!(status, 200, "{v}");
        assert_eq!(v["_shards"]["total"], 4, "{v}");
        assert_eq!(v["_shards"]["failed"], 0, "{v}");
        assert!(v["_shards"].get("failures").is_none(), "{v}");
        assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), DOCS, "{v}");
    }
    // A partial PIT from the other node misses the other half.
    let (status, v) = cl.call(1, Method::POST, "/pp/_pit?keep_alive=1m&preference=_only_local&allow_partial_search_results=true", None).await;
    assert_eq!(status, 200, "{v}");
    let other_pit = v["id"].clone();
    let (status, v) = cl.call(0, Method::POST, "/_search", Some(json!({"size": DOCS, "pit": {"id": other_pit}, "_source": false}))).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["_shards"]["failed"], 2, "{v}");
    assert!(!failed_shards(&v).is_empty() && failed_shards(&v).is_subset(&local), "{v}");
    assert_eq!(v["hits"]["total"]["value"], DOCS - local_ids.len(), "{v}");

    for id in [partial_pit, full_pit, other_pit] {
        let (status, v) = cl.call(0, Method::DELETE, "/_pit", Some(json!({"id": id}))).await;
        assert_eq!(status, 200, "{v}");
    }
}
