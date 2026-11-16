//! Corruption of a shard's durable translog data, on real node processes
//! ([`common::procs`]), as ES handles a `TranslogCorruptedException`:
//! - the copy is never opened with its acknowledged operations silently cut
//!   back: its store is marked corrupted (`corrupted_<id>`), the shard is
//!   failed, and `_cluster/allocation/explain` shows the copy's
//!   `store_exception`;
//! - a single-copy shard goes red ("all found copies of the shard are either
//!   stale or corrupt") and the corrupted generation is left as it was;
//! - a replicated shard recovers from its healthy copy, with every
//!   acknowledged operation on every copy.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// No idle flush during the test: the operations stay in the translog.
const EXTRA: [(&str, &str); 1] = [("indices.memory.shard_inactive_time", "1h")];
const DOCS: [&str; 3] = ["a", "b", "c"];

/// The shard directory (`.../shard-<n>`) of `index`'s copy on node `node`:
/// the one whose index directory holds the copy record naming `index`.
fn copy_dir(pl: &Procs, node: usize, index: &str) -> PathBuf {
    let mut stack = vec![pl.dir.join(format!("n{node}"))];
    let mut found = Vec::new();
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = e.path();
            if !path.is_dir() {
                continue;
            }
            let is_shard = path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("shard-") && !n.to_string_lossy().contains('.'));
            let record = path.parent().map(|p| p.join("cluster-copy.json"));
            if is_shard && record.as_ref().is_some_and(|r| r.exists()) {
                let rec: Value = serde_json::from_slice(&std::fs::read(record.unwrap()).unwrap()).unwrap();
                if rec["index"] == index {
                    found.push(path);
                }
            } else {
                stack.push(path);
            }
        }
    }
    assert_eq!(found.len(), 1, "copies of {index} on n{node}: {found:?}");
    found.pop().unwrap()
}

/// The newest non-empty translog generation of a shard directory.
fn newest_translog(shard: &Path) -> PathBuf {
    let mut gens: Vec<PathBuf> = std::fs::read_dir(shard.join("translog")).unwrap().flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "tlog") && std::fs::metadata(p).unwrap().len() > 0).collect();
    gens.sort();
    gens.pop().unwrap_or_else(|| panic!("no translog data under {}", shard.display()))
}

/// Flip a byte in the body of frame `n` (0-based) of a translog file whose
/// frames are `[len u32][crc32 u32][body]`; at least `n + 2` frames exist.
fn corrupt_frame(path: &Path, n: usize) -> Vec<u8> {
    let mut data = std::fs::read(path).unwrap();
    let mut starts = Vec::new();
    let mut pos = 0;
    while pos + 8 <= data.len() {
        starts.push(pos);
        pos += 8 + u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    }
    assert!(starts.len() >= n + 2, "{} has {} frames", path.display(), starts.len());
    data[starts[n] + 8 + 2] ^= 0xFF;
    std::fs::write(path, &data).unwrap();
    data
}

fn markers(shard: &Path) -> Vec<String> {
    std::fs::read_dir(shard).into_iter().flatten().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("corrupted_") && !n.ends_with(".tmp")).collect()
}

async fn create(pl: &Procs, index: &str, replicas: u32) {
    let (st, v) = pl.call(0, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": replicas}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, index, "green", 90).await;
    // One request (one translog frame) per document, each fsynced before
    // it is acknowledged (request durability).
    for (i, id) in DOCS.iter().enumerate() {
        let (st, v) = pl.call(0, Method::PUT, &format!("/{index}/_doc/{id}"), Some(json!({"n": i}))).await;
        assert_eq!(st, 201, "{v}");
    }
}

/// A single-copy shard whose translog has a corrupted record (B of A/B/C,
/// all acknowledged): the copy is failed and marked corrupted, the index
/// goes red with the copy's store exception in the allocation explanation,
/// and the translog is left as it was. Never "A only".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_corrupted_translog_record_fails_a_single_copy_shard() {
    let mut pl = start_procs(1, &EXTRA).await;
    create(&pl, "single", 0).await;
    pl.kill(0);
    let shard = copy_dir(&pl, 0, "single");
    let tlog = newest_translog(&shard);
    let corrupted = corrupt_frame(&tlog, 1);
    pl.spawn(0, &[]).await;
    pl.wait_nodes(0, 1, 120).await;

    let deadline = Instant::now() + Duration::from_secs(90);
    let explain = loop {
        let (_, ex) = pl.call(0, Method::GET, "/_cluster/allocation/explain", Some(json!({"index": "single", "shard": 0, "primary": true}))).await;
        let node = ex["node_allocation_decisions"][0].clone();
        if ex["can_allocate"] == "no_valid_shard_copy" && node["store"]["store_exception"].is_object() {
            break ex;
        }
        assert!(Instant::now() < deadline, "the corrupted copy was not reported: {ex}\n{}", pl.log_tail(0));
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    assert_eq!(explain["allocate_explanation"], "cannot allocate because all found copies of the shard are either stale or corrupt", "{explain}");
    let store = &explain["node_allocation_decisions"][0]["store"];
    assert_eq!(store["in_sync"], true, "{explain}");
    assert_eq!(store["store_exception"]["type"], "corrupt_index_exception", "{explain}");
    assert!(store["store_exception"]["reason"].as_str().unwrap().contains("corrupted"), "{explain}");
    let (_, health) = pl.call(0, Method::GET, "/_cluster/health/single", None).await;
    assert_eq!(health["status"], "red", "{health}");
    let rows = pl.shards(0, "single").await;
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["state"], "UNASSIGNED", "{rows:?}");
    // `_shard_stores` lists the copy with its store exception (red by default).
    let (st, stores) = pl.call(0, Method::GET, "/single/_shard_stores", None).await;
    assert_eq!(st, 200, "{stores}");
    let listed = stores["indices"]["single"]["shards"]["0"]["stores"].as_array().cloned().unwrap_or_default();
    let marked: Vec<&Value> = listed.iter().filter(|s| s["store_exception"].is_object()).collect();
    assert_eq!(marked.len(), 1, "{stores}");
    assert_eq!(marked[0]["store_exception"]["type"], "corrupt_index_exception", "{stores}");
    assert_eq!(marked[0]["allocation"], "unused", "{stores}");
    // The store is marked, and its translog left exactly as it was.
    assert_eq!(markers(&shard).len(), 1, "no corruption marker in {}", shard.display());
    assert_eq!(std::fs::read(&tlog).unwrap(), corrupted, "the corrupted translog was changed");
    // And it stays that way across another restart.
    pl.kill(0);
    pl.spawn(0, &[]).await;
    pl.wait_nodes(0, 1, 120).await;
    tokio::time::sleep(Duration::from_secs(5)).await;
    let (_, health) = pl.call(0, Method::GET, "/_cluster/health/single", None).await;
    assert_eq!(health["status"], "red", "{health}");
    assert_eq!(std::fs::read(&tlog).unwrap(), corrupted, "the corrupted translog was changed");
}

/// Both nodes of a replicated shard go down; the former primary's
/// translog has a corrupted record. Its copy is failed and marked, the
/// healthy copy becomes the primary, the corrupted one is recovered from it,
/// and every acknowledged document is on both copies.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_corrupted_translog_record_is_recovered_from_the_healthy_copy() {
    let mut pl = start_procs(2, &EXTRA).await;
    create(&pl, "repl", 1).await;
    let ids = pl.node_ids(0).await;
    let rows = pl.shards(0, "repl").await;
    let holder = |prirep: &str| -> usize {
        rows.iter().find(|r| r["prirep"] == prirep).and_then(|r| r["node"].as_str()).unwrap_or_else(|| panic!("{rows:?}"))
            .trim_start_matches('n').parse().unwrap()
    };
    let (primary, replica) = (holder("p"), holder("r"));
    // Down together: neither copy is failed while the other is up.
    pl.kill(replica);
    pl.kill(primary);
    let shard = copy_dir(&pl, primary, "repl");
    corrupt_frame(&newest_translog(&shard), 1);
    pl.spawn(0, &[]).await;
    pl.spawn(1, &[]).await;
    for i in 0..2 {
        pl.wait_nodes(i, 2, 120).await;
    }
    pl.wait_health(0, "repl", "green", 180).await;
    let (st, v) = pl.call(0, Method::POST, "/repl/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    for n in 0..2 {
        let path = format!("/repl/_search?preference=_only_nodes:{}", ids[&n]);
        let (st, v) = pl.call(0, Method::POST, &path, Some(json!({"size": 10, "query": {"match_all": {}}}))).await;
        assert!(st == 200 && v["_shards"]["successful"] == 1, "n{n}: {st} {v}");
        let mut got: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        got.sort();
        assert_eq!(got, DOCS, "the copy on n{n} lost acknowledged writes: {v}");
    }
    // The corrupted copy was replaced (its marker went with it), not reused.
    let rows = pl.shards(0, "repl").await;
    assert!(rows.iter().all(|r| r["state"] == "STARTED"), "{rows:?}");
    let shard = copy_dir(&pl, primary, "repl");
    assert!(markers(&shard).is_empty(), "the recovered copy still carries a corruption marker");
}
