//! A stored script change that a follower cannot persist: the follower
//! still runs the committed script (never the older one), does not count as
//! having applied the change (the update is not acknowledged), and catches
//! up by itself once its disk recovers.

mod common;
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Every `_state` directory under `dir` (one per engine of a node).
fn state_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                if e.file_name() == "_state" {
                    out.push(p.clone());
                }
                stack.push(p);
            }
        }
    }
    out
}

/// Make every engine's catalog file on the node unreplaceable (a non-empty
/// directory in its place), or undo it.
fn block_catalogs(node_dir: &Path, block: bool) {
    for d in state_dirs(node_dir) {
        let file = d.join("scripts.json");
        if block {
            let _ = std::fs::remove_file(&file);
            std::fs::create_dir_all(file.join("blocker")).unwrap();
        } else if file.is_dir() {
            std::fs::remove_dir_all(&file).unwrap();
        }
    }
}

async fn put_script(c: &common::Cluster, via: usize, n: i64) -> (u16, Value) {
    c.call(via, Method::PUT, "/_scripts/inc?timeout=3s", Some(json!({"script": {"lang": "painless", "source": format!("ctx._source.n += {n}")}}))).await
}

async fn bump(c: &common::Cluster, via: usize) -> i64 {
    let (st, v) = c.call(via, Method::POST, "/scr/_update/d?refresh=true", Some(json!({"script": {"id": "inc"}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = c.call(via, Method::GET, "/scr/_doc/d", None).await;
    assert_eq!(st, 200, "{v}");
    v["_source"]["n"].as_i64().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn acknowledged_script_is_the_one_writes_use() {
    let c = common::start(&[]).await;
    let (master, follower) = (c.master, c.other);
    let follower_dir = c.dir.join(format!("n{follower}"));
    // The only copy lives on the follower, so the follower runs the script.
    let (st, v) = c.call(master, Method::PUT, "/scr", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": format!("n{follower}")}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = c.call(master, Method::PUT, "/scr/_doc/d?refresh=true", Some(json!({"n": 0}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = put_script(&c, master, 1).await;
    assert_eq!((st, v["acknowledged"].clone()), (200, json!(true)), "{v}");
    assert_eq!(bump(&c, master).await, 1);

    // Follower-only catalog write failure: +1 becomes +10.
    block_catalogs(&follower_dir, true);
    let (st, v) = put_script(&c, master, 10).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["acknowledged"], json!(false), "a node that could not apply the change must not be counted as having applied it: {v}");
    // Whatever was acknowledged or not, the follower never runs the old +1.
    assert_eq!(bump(&c, master).await, 11, "the committed script must be the one writes use");
    assert_eq!(bump(&c, follower).await, 21);

    // The disk recovers: the follower persists the catalog on its own retry
    // and acknowledges again.
    block_catalogs(&follower_dir, false);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (st, v) = put_script(&c, master, 100).await;
        assert_eq!(st, 200, "{v}");
        if v["acknowledged"] == json!(true) {
            break;
        }
        assert!(Instant::now() < deadline, "the follower never caught up: {v}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert_eq!(bump(&c, master).await, 121, "the acknowledged script is the one used");
    let persisted: Vec<String> = state_dirs(&follower_dir).iter().filter_map(|d| std::fs::read_to_string(d.join("scripts.json")).ok()).collect();
    assert!(!persisted.is_empty() && persisted.iter().all(|s| s.contains("+= 100")), "{persisted:?}");
}
