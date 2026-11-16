//! Completed task results outlive their node (production review #96), on
//! three node processes ([`common::procs`]):
//!
//! * a finished `wait_for_completion=false` task — run by the cluster
//!   (multi-shard by-query, successful and failing) or by a shard engine
//!   (single-shard by-query) — is stored in the cluster's `.tasks` index,
//!   and `GET _tasks/{id}` returns that stored result while its node is out
//!   of the cluster and after the node restarted (ES `TaskResultsService`,
//!   `TransportGetTaskAction.getFinishedTaskFromIndex`);
//! * a restarted node never hands out a task number it used before, so an
//!   old task id never names new, unrelated work.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

async fn ok(pl: &Procs, via: usize, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = pl.call(via, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path} via n{via}: {st} {v}");
    v
}

/// What a completed task's `GET _tasks/{id}` must keep reporting.
fn digest(v: &Value) -> Value {
    json!({
        "completed": v["completed"], "action": v["task"]["action"], "description": v["task"]["description"],
        "response": v["response"], "error": v["error"],
    })
}

async fn completed(pl: &Procs, via: usize, task: &str, secs: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (st, v) = pl.call(via, Method::GET, &format!("/_tasks/{task}"), None).await;
        if st == 200 && v["completed"] == true {
            return v;
        }
        assert!(Instant::now() < deadline, "task {task} via n{via} did not complete: {st} {v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Wait until task `task`'s result document is in `.tasks`.
async fn stored(pl: &Procs, via: usize, task: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (st, v) = pl.call(via, Method::GET, &format!("/.tasks/_doc/{task}"), None).await;
        if st == 200 && v["found"] == true {
            return;
        }
        assert!(Instant::now() < deadline, "task {task} was never stored: {st} {v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn start_task(pl: &Procs, via: usize, path: &str, body: Value) -> String {
    let v = ok(pl, via, Method::POST, path, Some(body)).await;
    v["task"].as_str().unwrap_or_else(|| panic!("{path}: {v}")).to_string()
}

fn number(task: &str) -> u64 {
    task.rsplit_once(':').and_then(|(_, n)| n.parse().ok()).unwrap_or_else(|| panic!("task id {task}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completed_task_results_survive_their_node_and_task_ids_never_repeat() {
    let mut pl = start_procs(3, &[]).await;
    for (index, shards) in [("multi", 2), ("single", 1)] {
        ok(&pl, 0, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 1}, "mappings": {"properties": {"v": {"type": "long"}}}}))).await;
        let lines: String = (0..20).map(|i| format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("d{i}")}}), json!({"v": i}))).collect();
        let (st, v) = pl.ndjson(0, "/_bulk?refresh=true", lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
        pl.wait_health(0, index, "green", 60).await;
    }
    let ids: BTreeMap<usize, String> = pl.node_ids(0).await;
    let position = |task: &str| -> usize {
        let node = task.rsplit_once(':').unwrap().0;
        *ids.iter().find(|(_, id)| id.as_str() == node).unwrap_or_else(|| panic!("task {task} on an unknown node: {ids:?}")).0
    };

    // Cluster-run (successful and failing) and engine-run tasks.
    let mut tasks: Vec<String> = Vec::new();
    tasks.push(start_task(&pl, 2, "/multi/_update_by_query?wait_for_completion=false&refresh=true", json!({"script": {"source": "ctx._source.v += 1"}})).await);
    tasks.push(start_task(&pl, 2, "/multi/_update_by_query?wait_for_completion=false", json!({"script": {"source": "ctx._source.v = ctx._source.nope.length()"}})).await);
    tasks.push(start_task(&pl, 1, "/single/_update_by_query?wait_for_completion=false&refresh=true", json!({"script": {"source": "ctx._source.v += 1"}})).await);
    let mut expected: BTreeMap<String, Value> = BTreeMap::new();
    for t in &tasks {
        let v = completed(&pl, 0, t, 60).await;
        expected.insert(t.clone(), digest(&v));
        stored(&pl, 0, t).await;
    }
    assert!(expected[&tasks[1]]["error"].is_object() || expected[&tasks[1]]["response"]["failures"].as_array().is_some_and(|f| !f.is_empty()), "the second task failed: {}", expected[&tasks[1]]);

    let mut owners: Vec<usize> = tasks.iter().map(|t| position(t)).collect();
    owners.sort_unstable();
    owners.dedup();
    let mut highest: BTreeMap<usize, u64> = BTreeMap::new();
    for t in &tasks {
        let e = highest.entry(position(t)).or_default();
        *e = (*e).max(number(t));
    }
    for &o in &owners {
        let other = (o + 1) % 3;
        pl.wait_health(other, ".tasks", "green", 60).await;
        // The owner leaves: its results are read from `.tasks`.
        pl.kill(o);
        pl.wait_nodes(other, 2, 60).await;
        pl.wait_health(other, ".tasks", "yellow", 60).await;
        for t in tasks.iter().filter(|t| position(t) == o) {
            let (st, v) = pl.call(other, Method::GET, &format!("/_tasks/{t}"), None).await;
            assert_eq!(st, 200, "{t} with its node n{o} gone: {v}");
            assert_eq!(digest(&v), expected[t], "{t} with its node n{o} gone");
        }
        // The owner restarts with an empty memory: still from `.tasks`.
        pl.spawn(o, &[]).await;
        pl.wait_nodes(other, 3, 120).await;
        pl.wait_nodes(o, 3, 120).await;
        for t in tasks.iter().filter(|t| position(t) == o) {
            for via in [o, other] {
                let (st, v) = pl.call(via, Method::GET, &format!("/_tasks/{t}"), None).await;
                assert_eq!(st, 200, "{t} via n{via} after n{o} restarted: {v}");
                assert_eq!(digest(&v), expected[t], "{t} via n{via} after n{o} restarted");
            }
        }
    }

    // New work on the restarted nodes never takes an old task's identity.
    pl.wait_health(0, "multi", "green", 120).await;
    pl.wait_health(0, "single", "green", 120).await;
    for &o in &owners {
        for _ in 0..3 {
            let t = start_task(&pl, o, "/multi/_update_by_query?wait_for_completion=false&refresh=true", json!({"script": {"source": "ctx._source.v += 1"}})).await;
            assert!(!tasks.contains(&t), "new task {t} reuses an old id");
            assert!(number(&t) > highest[&o], "n{o} handed out task number {} after restarting; it had reached {}", number(&t), highest[&o]);
            completed(&pl, o, &t, 60).await;
        }
    }
    for t in &tasks {
        let (st, v) = pl.call(0, Method::GET, &format!("/_tasks/{t}"), None).await;
        assert_eq!(st, 200, "{t}: {v}");
        assert_eq!(digest(&v), expected[t], "{t} still names its own, old result");
    }
}
