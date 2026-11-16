//! ES 9.5.5 diffs found by the Rally reviewer, on node processes:
//! - the watcher registry's `watch-history-ilm-policy-16` exists from the
//!   start, with ES's body, used by `.watch-history-18`;
//! - a search failing on every shard names each shard's node in
//!   `failed_shards[].node` (it was null).
mod common;
use common::procs::start_procs;
use reqwest::Method;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watcher_policy_and_failed_shard_nodes() {
    let pl = start_procs(3, &[]).await;
    let via = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let policy = loop {
        // The policy is built in; the registry installs `.watch-history-18`
        // with the master's first metadata job.
        let (st, v) = pl.call(via, Method::GET, "/_ilm/policy/watch-history-ilm-policy-16", None).await;
        if st == 200 && v["watch-history-ilm-policy-16"]["in_use_by"]["composable_templates"].as_array().is_some_and(|a| !a.is_empty()) {
            break v;
        }
        assert!(std::time::Instant::now() < deadline, "{st} {v}");
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    };
    let p = &policy["watch-history-ilm-policy-16"];
    assert_eq!(p["version"], 1, "{policy}");
    assert_eq!(p["policy"]["phases"], json!({
        "hot": {"min_age": "0ms", "actions": {"rollover": {"max_age": "3d", "max_primary_shard_size": "50gb"}}},
        "delete": {"min_age": "4d", "actions": {"delete": {"delete_searchable_snapshot": true}}}
    }), "{policy}");
    assert_eq!(p["policy"]["_meta"], json!({"managed": true, "description": "default policy for the watcher history indices"}), "{policy}");
    assert_eq!(p["in_use_by"]["composable_templates"], json!([".watch-history-18"]), "{policy}");
    let ids = pl.node_ids(via).await;
    for (index, shards) in [("one", 1), ("two", 2)] {
        let (st, v) = pl.call(via, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0}}))).await;
        assert_eq!(st, 200, "{v}");
        pl.wait_health(via, index, "green", 60).await;
        for path in [format!("/{index}/_search?size=10001"), format!("/{index}/_mvt/geo/0/0/0?size=10001")] {
            let (st, v) = pl.call(via, Method::GET, &path, None).await;
            if path.contains("_mvt") && v["error"]["type"] != "search_phase_execution_exception" {
                continue; // no geo field: not this error shape
            }
            assert_eq!(st, 400, "{path}: {v}");
            let failed = v["error"]["failed_shards"].as_array().cloned().unwrap_or_default();
            assert!(!failed.is_empty(), "{path}: {v}");
            for f in failed {
                let node = f["node"].as_str().unwrap_or_default();
                assert!(ids.values().any(|id| id == node), "{path}: failed shard node {node:?} is not a node id ({ids:?}): {v}");
            }
        }
    }
}
