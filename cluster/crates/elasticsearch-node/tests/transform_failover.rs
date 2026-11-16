//! Transforms on a cluster of real node processes ([`common::procs`]):
//! - a continuous transform survives a change of master: its config, state
//!   and checkpoints live in `.transform-internal-007`, its persistent task
//!   moves to the new master (new allocation id), which resumes after the
//!   last checkpoint, while the old master, still alive, stops executing it;
//! - a checkpoint never completes on partial source results: with a source
//!   shard lost the checkpoint fails and is retried, and completes with every
//!   group once the shard is back.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const NODES: usize = 3;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}

async fn index_docs(cl: &Procs, via: usize, index: &str, docs: &[Value]) {
    let mut body = String::new();
    for d in docs {
        body.push_str(&json!({"index": {"_index": index}}).to_string());
        body.push('\n');
        body.push_str(&d.to_string());
        body.push('\n');
    }
    let (st, v) = cl.ndjson(via, "/_bulk?refresh=true", body).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false, "{v}");
}

async fn stats(cl: &Procs, via: usize, id: &str) -> Value {
    let (_, v) = cl.call(via, Method::GET, &format!("/_transform/{id}/_stats"), None).await;
    v["transforms"][0].clone()
}

/// Wait until `f` holds for the transform's stats (as `via` serves them).
async fn wait_stats(cl: &Procs, via: usize, id: &str, secs: u64, what: &str, f: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut last = Value::Null;
    while Instant::now() < deadline {
        last = stats(cl, via, id).await;
        if f(&last) {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("{what}: {last}");
}

/// The destination documents, by group key.
async fn dest_groups(cl: &Procs, via: usize, index: &str) -> Vec<(String, i64)> {
    let (_, v) = cl.call(via, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 100, "sort": [{"k": "asc"}]}))).await;
    v["hits"]["hits"].as_array().cloned().unwrap_or_default().iter().map(|h| (h["_source"]["k"].as_str().unwrap_or("").to_string(), h["_source"]["n"].as_i64().unwrap_or(-1))).collect()
}

fn checkpoint(s: &Value) -> u64 {
    s["checkpointing"]["last"]["checkpoint"].as_u64().unwrap_or(0)
}

fn log_since(cl: &Procs, i: usize, from: usize) -> String {
    let text = std::fs::read_to_string(cl.log_path(i)).unwrap_or_default();
    text.get(from.min(text.len())..).unwrap_or("").to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn transform_moves_to_the_new_master_and_the_old_one_stops() {
    let cl = start_procs(NODES, &[]).await;
    let old = cl.master(0).await;
    let (st, v) = cl.call(old, Method::PUT, "/src", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 1},
        "mappings": {"properties": {"time": {"type": "date"}, "k": {"type": "keyword"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    cl.wait_health(old, "src", "green", 60).await;
    let t0 = now_ms() - 60_000;
    index_docs(&cl, old, "src", &[json!({"time": t0, "k": "k1"}), json!({"time": t0, "k": "k2"})]).await;
    let (st, v) = cl.call(old, Method::PUT, "/_transform/t1", Some(json!({
        "source": {"index": "src"},
        "dest": {"index": "dst"},
        "frequency": "1s",
        "sync": {"time": {"field": "time", "delay": "1s"}},
        "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = cl.call(old, Method::POST, "/_transform/t1/_start", None).await;
    assert_eq!(st, 200, "{v}");
    let s = wait_stats(&cl, old, "t1", 60, "first checkpoint", |s| checkpoint(s) >= 1).await;
    let cp1 = checkpoint(&s);
    assert_eq!(dest_groups(&cl, old, "dst").await, vec![("k1".into(), 1), ("k2".into(), 1)]);
    let ids = cl.node_ids(old).await;
    let (_, state) = cl.call(old, Method::GET, "/_cluster/state/metadata", None).await;
    let task = state["metadata"]["persistent_tasks"]["tasks"].as_array().and_then(|a| a.iter().find(|t| t["id"] == "t1").cloned()).unwrap_or_else(|| panic!("no task: {state}"));
    assert_eq!(task["assignment"]["executor_node"], ids[&old].as_str(), "{task}");
    let alloc1 = task["allocation_id"].as_u64().unwrap();

    // A different master, the old one alive (voting exclusion: it abdicates).
    let (st, v) = cl.call(old, Method::POST, &format!("/_cluster/voting_config_exclusions?node_names=n{old}"), None).await;
    assert_eq!(st, 200, "{v}");
    let other = (0..NODES).find(|i| *i != old).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    let new = loop {
        let m = cl.master(other).await;
        if m != old {
            break m;
        }
        assert!(Instant::now() < deadline, "the master did not change");
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    // The new master found the transform in the cluster's storage and took
    // its task over: same config, checkpoint continues, new allocation id.
    let (st, v) = cl.call(new, Method::GET, "/_transform/t1", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["count"], 1, "{v}");
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, state) = cl.call(new, Method::GET, "/_cluster/state/metadata", None).await;
        let task = state["metadata"]["persistent_tasks"]["tasks"].as_array().and_then(|a| a.iter().find(|t| t["id"] == "t1").cloned()).unwrap_or(Value::Null);
        if task["assignment"]["executor_node"] == ids[&new].as_str() {
            assert!(task["allocation_id"].as_u64().unwrap() > alloc1, "{task}");
            break;
        }
        assert!(Instant::now() < deadline, "task not re-assigned: {state}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let s = stats(&cl, new, "t1").await;
    assert!(checkpoint(&s) >= cp1, "the checkpoint continues from the stored state: {s}");
    assert_ne!(s["state"], "stopped", "{s}");
    let old_log_mark = std::fs::metadata(cl.log_path(old)).map(|m| m.len() as usize).unwrap_or(0);
    // New data is picked up by the new owner, twice.
    for (round, key) in ["k3", "k4"].iter().enumerate() {
        let before = checkpoint(&stats(&cl, new, "t1").await);
        index_docs(&cl, new, "src", &[json!({"time": now_ms() - 100, "k": key})]).await;
        wait_stats(&cl, new, "t1", 60, &format!("checkpoint after round {round}"), |s| checkpoint(s) > before).await;
        let deadline = Instant::now() + Duration::from_secs(30);
        while !dest_groups(&cl, new, "dst").await.iter().any(|(k, _)| k == key) {
            assert!(Instant::now() < deadline, "{key} not transformed");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    let groups = dest_groups(&cl, new, "dst").await;
    assert_eq!(groups, vec![("k1".into(), 1), ("k2".into(), 1), ("k3".into(), 1), ("k4".into(), 1)], "no group lost or counted twice");
    // Only the current owner executes: the old master (alive) logged that
    // it stopped (when it gave the transforms up, or when a checkpoint in
    // flight hit the ownership fence first), and completed no checkpoint
    // after the hand-over.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let old_full = log_since(&cl, old, 0);
    assert!(old_full.contains("[t1] transform stopped executing on this node"), "{}", cl.log_tail(old));
    let old_after = log_since(&cl, old, old_log_mark);
    assert!(!old_after.contains("[t1] finished indexing for transform checkpoint"), "the old master still executes t1:\n{old_after}");
    let new_full = log_since(&cl, new, 0);
    assert!(new_full.contains("[t1] transform task assigned to this node"), "{}", cl.log_tail(new));
    assert!(new_full.contains("[t1] finished indexing for transform checkpoint"), "{}", cl.log_tail(new));
    let (st, v) = cl.call(new, Method::POST, "/_transform/t1/_stop?wait_for_completion=true", None).await;
    assert_eq!(st, 200, "{v}");
    let (_, state) = cl.call(new, Method::GET, "/_cluster/state/metadata", None).await;
    assert!(!state["metadata"]["persistent_tasks"]["tasks"].as_array().is_some_and(|a| a.iter().any(|t| t["id"] == "t1")), "stop removes the task: {state}");
}

#[tokio::test(flavor = "multi_thread")]
async fn transform_checkpoint_waits_for_a_lost_source_shard() {
    let mut cl = start_procs(NODES, &[]).await;
    let master = cl.master(0).await;
    let victim = (0..NODES).find(|i| *i != master).unwrap();
    let keeper = (0..NODES).find(|i| *i != master && *i != victim).unwrap();
    for (index, on) in [("src-a", keeper), ("src-b", victim)] {
        let (st, v) = cl.call(master, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": format!("n{on}")},
            "mappings": {"properties": {"k": {"type": "keyword"}}}
        }))).await;
        assert_eq!(st, 200, "{v}");
        cl.wait_health(master, index, "green", 60).await;
    }
    index_docs(&cl, master, "src-a", &[json!({"k": "a1"}), json!({"k": "a2"})]).await;
    index_docs(&cl, master, "src-b", &[json!({"k": "b1"})]).await;
    // The only copy of src-b goes away.
    cl.kill(victim);
    cl.wait_nodes(master, NODES - 1, 60).await;
    let (st, v) = cl.call(master, Method::PUT, "/_transform/lossy?defer_validation=true", Some(json!({
        "source": {"index": ["src-a", "src-b"]},
        "dest": {"index": "lossy-out"},
        "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = cl.call(master, Method::POST, "/_transform/lossy/_start", None).await;
    assert_eq!(st, 200, "{v}");
    let s = wait_stats(&cl, master, "lossy", 60, "a failed attempt", |s| s["stats"]["search_failures"].as_u64().unwrap_or(0) >= 1).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let s2 = stats(&cl, master, "lossy").await;
    for s in [&s, &s2] {
        assert_eq!(checkpoint(s), 0, "a checkpoint completed without src-b: {s}");
        assert_ne!(s["state"], "stopped", "{s}");
        assert_ne!(s["state"], "failed", "a lost shard is retried: {s}");
    }
    assert_eq!(s2["health"]["status"], "yellow", "{s2}");
    // The shard comes back: the retried checkpoint sees every group.
    cl.spawn(victim, &[]).await;
    cl.wait_nodes(master, NODES, 120).await;
    cl.wait_health(master, "src-b", "green", 120).await;
    let (st, v) = cl.call(master, Method::POST, "/_transform/lossy/_schedule_now", None).await;
    assert_eq!(st, 200, "{v}");
    let s = wait_stats(&cl, master, "lossy", 120, "the checkpoint after recovery", |s| checkpoint(s) == 1 && s["state"] == "stopped").await;
    assert_eq!(s["health"]["status"], "green", "{s}");
    let mut groups = dest_groups(&cl, master, "lossy-out").await;
    groups.sort();
    assert_eq!(groups, vec![("a1".into(), 1), ("a2".into(), 1), ("b1".into(), 1)], "every group, src-b's included");
}
