//! A restore resumed by a new master after a failover (`snapshot::restore`):
//! an index whose restore completed (published `done`) and then took writes
//! keeps its incarnation and every acknowledged write; only the incomplete
//! target is deleted (by its recorded incarnation) and restored again.
//!
//! Three node processes; the `RUSTIC_TEST_RESTORE_STALL` seam stops the
//! first restore of `b-second` for good right before publishing it, and the
//! master running it is then killed.

mod common;

use common::procs::{start_procs_env, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

async fn ok(pl: &Procs, i: usize, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = pl.call(i, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path} on n{i}: {st} {v}");
    v
}

async fn bulk(pl: &Procs, i: usize, nd: String) {
    let (st, v) = pl.ndjson(i, "/_bulk?refresh=true", nd).await;
    assert!(st == 200 && v["errors"] == false, "{st} {v}");
}

fn docs(index: &str, ids: std::ops::Range<usize>, v: &str) -> String {
    ids.map(|i| format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("d{i}")}}), json!({"n": i, "v": v}))).collect()
}

async fn uuid_of(pl: &Procs, i: usize, index: &str) -> String {
    let v = ok(pl, i, Method::GET, &format!("/{index}/_settings?flat_settings=true"), None).await;
    v[index]["settings"]["index.uuid"].as_str().unwrap_or_else(|| panic!("{v}")).to_string()
}

/// The restore entry's per-index shard states (`index` → state), as `i` sees them.
async fn restore_states(pl: &Procs, i: usize) -> Option<Vec<(String, String)>> {
    let (_, v) = pl.call(i, Method::GET, "/_cluster/state", None).await;
    let e = v["restore"]["snapshots"].as_array()?.first()?.clone();
    Some(e["shards"].as_array()?.iter().map(|s| (s["index"].as_str().unwrap_or_default().to_string(), s["state"].as_str().unwrap_or_default().to_string())).collect())
}

async fn count(pl: &Procs, i: usize, index: &str) -> u64 {
    let v = ok(pl, i, Method::GET, &format!("/{index}/_count"), None).await;
    v["count"].as_u64().unwrap_or_else(|| panic!("{v}"))
}

/// File restores (the snapshot's files adopted).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completed_index_and_its_writes_survive_a_restore_resumed_after_failover() {
    failover(false).await;
}

/// Re-indexing restores (a source-only repository): the target is
/// write-blocked during its document replay too, so no client write can
/// land in an index a new master deletes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reindexing_restore_keeps_clients_out_until_published_and_survives_failover() {
    failover(true).await;
}

async fn failover(source_only: bool) {
    let repo = common::scratch_dir(if source_only { "restore-failover-src-repo" } else { "restore-failover-repo" });
    let repo_s = repo.to_string_lossy().to_string();
    let mut pl = start_procs_env(3, &[("path.repo", repo_s.as_str())], &[("RUSTIC_TEST_RESTORE_STALL", "b-second")]).await;
    let m = pl.master(0).await;
    let repo_body = if source_only {
        json!({"type": "source", "settings": {"delegate_type": "fs", "location": repo_s}})
    } else {
        json!({"type": "fs", "settings": {"location": repo_s}})
    };
    ok(&pl, m, Method::PUT, "/_snapshot/repo", Some(repo_body)).await;
    for index in ["a-first", "b-second"] {
        ok(&pl, m, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
        bulk(&pl, m, docs(index, 0..50, "snapshot")).await;
    }
    let v = ok(&pl, m, Method::PUT, "/_snapshot/repo/snap?wait_for_completion=true", Some(json!({"indices": "a-first,b-second"}))).await;
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    ok(&pl, m, Method::DELETE, "/a-first,b-second", None).await;

    let v = ok(&pl, m, Method::POST, "/_snapshot/repo/snap/_restore", Some(json!({"indices": "a-first,b-second", "index_settings": {"index.number_of_replicas": 1}}))).await;
    assert_eq!(v["accepted"], true, "{v}");
    // a-first is published done; b-second is created and stalled before
    // its publication.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let states = restore_states(&pl, m).await;
        let b_exists = pl.call(m, Method::HEAD, "/b-second", None).await.0 == 200;
        if let Some(s) = &states {
            if s.contains(&("a-first".into(), "SUCCESS".into())) && s.contains(&("b-second".into(), "INIT".into())) && b_exists {
                break;
            }
        }
        assert!(Instant::now() < deadline, "restore did not reach the stall point: {states:?}\n{}", pl.log_tail(m));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    pl.wait_health(m, "a-first", "green", 120).await;
    let other = (0..3).find(|i| *i != m).unwrap();
    // Acknowledged writes to the completed index: new documents and an
    // overwrite of a restored one.
    bulk(&pl, other, docs("a-first", 50..60, "after")).await;
    bulk(&pl, other, docs("a-first", 0..1, "overwritten")).await;
    assert_eq!(count(&pl, other, "a-first").await, 60);
    let a_uuid = uuid_of(&pl, other, "a-first").await;
    let b_uuid = uuid_of(&pl, other, "b-second").await;
    // The incomplete target takes no client write.
    let (st, v) = pl.ndjson(other, "/_bulk?refresh=true", docs("b-second", 200..201, "too-early")).await;
    assert!(st == 200 && v["errors"] == true && v["items"][0]["index"]["status"] == 403, "a client wrote to an unpublished restore target: {st} {v}");

    pl.kill(m);
    let survivor = (0..3).find(|i| *i != m).unwrap();
    pl.wait_nodes(survivor, 2, 120).await;
    let new_master = pl.master(survivor).await;
    assert_ne!(new_master, m);
    // The new master resumes the restore and finishes it.
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if restore_states(&pl, survivor).await.is_none() {
            break;
        }
        assert!(Instant::now() < deadline, "the resumed restore did not finish\n{}", pl.log_tail(new_master));
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    pl.wait_health(survivor, "a-first", "green", 120).await;
    pl.wait_health(survivor, "b-second", "green", 120).await;
    // The completed index is the same incarnation with every write.
    assert_eq!(uuid_of(&pl, survivor, "a-first").await, a_uuid, "the completed index was deleted and restored again");
    assert_eq!(count(&pl, survivor, "a-first").await, 60, "acknowledged writes to the completed index were lost");
    let d0 = ok(&pl, survivor, Method::GET, "/a-first/_doc/d0", None).await;
    assert_eq!(d0["_source"]["v"], "overwritten", "{d0}");
    let d55 = ok(&pl, survivor, Method::GET, "/a-first/_doc/d55", None).await;
    assert_eq!(d55["_source"]["v"], "after", "{d55}");
    // The incomplete index was deleted (its own incarnation) and restored
    // again from the snapshot, and is writable.
    assert_ne!(uuid_of(&pl, survivor, "b-second").await, b_uuid, "the incomplete target was not restored again");
    assert_eq!(count(&pl, survivor, "b-second").await, 50);
    let v = ok(&pl, survivor, Method::GET, "/b-second/_settings?flat_settings=true", None).await;
    assert!(v["b-second"]["settings"].get("index.blocks.write").is_none_or(|b| b == "false"), "the restore's write block is left: {v}");
    bulk(&pl, survivor, docs("b-second", 100..101, "after")).await;
    assert_eq!(count(&pl, survivor, "b-second").await, 51);
}
