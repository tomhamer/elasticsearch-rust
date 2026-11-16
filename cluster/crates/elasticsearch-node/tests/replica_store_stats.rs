//! A replica copy's write-ahead log (`<shard path>/cluster-replica-wal`)
//! is that copy's translog: `_stats` reports its bytes under `translog`,
//! never under `store` (ES `StoreStats` covers the index files only).
mod common;
use common::procs::start_procs;
use reqwest::Method;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn size(p: &Path, skip: &[&str]) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            if skip.iter().any(|s| e.file_name() == *s) {
                continue;
            }
            let Ok(md) = e.metadata() else { continue };
            total += if md.is_dir() { size(&e.path(), &[]) } else { md.len() };
        }
    }
    total
}

/// Directories named `name` under `root`.
fn find(root: &Path, name: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    for e in rd.flatten() {
        if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if e.file_name() == name {
                out.push(e.path());
            } else {
                find(&e.path(), name, out);
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replica_wal_is_translog_not_store() {
    let pl = start_procs(2, &[]).await;
    let (st, v) = pl.call(0, Method::PUT, "/rw", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "rw", "green", 120).await;
    // Incompressible padding: the replica WAL is stored compressed, and a
    // repeated character shrank 3 MB of batches to ~55 KB of WAL.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut pad = || -> String {
        (0..63).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); format!("{:016x}", x) }).collect()
    };
    for batch in 0..6 {
        let lines: String = (0..500)
            .map(|i| format!("{}\n{}\n", json!({"index": {"_index": "rw", "_id": format!("{batch}-{i}")}}), json!({"n": i, "pad": pad()})))
            .collect();
        let (st, v) = pl.ndjson(0, "/_bulk", lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
    }
    let (st, _) = pl.call(0, Method::POST, "/rw/_refresh", None).await;
    assert_eq!(st, 200);
    // Past the store stats cache (10 s), nothing written meanwhile.
    tokio::time::sleep(Duration::from_secs(11)).await;
    let mut wals = Vec::new();
    find(&pl.dir, "cluster-replica-wal", &mut wals);
    assert_eq!(wals.len(), 1, "one replica copy: {wals:?}");
    let shard_dir = wals[0].parent().unwrap().to_path_buf();
    let wal = size(&wals[0], &[]);
    // 3,000 docs of ~1 KB random hex: at least half a megabyte compressed,
    // enough that counting it as store would show in the comparison below.
    assert!(wal > 512 << 10, "the replica WAL holds the batches: {wal} bytes");
    let files = size(&shard_dir, &["translog", "cluster-replica-wal"]);
    let (st, v) = pl.call(0, Method::GET, "/rw/_stats?level=shards", None).await;
    assert_eq!(st, 200, "{v}");
    let replica = v["indices"]["rw"]["shards"]["0"].as_array().unwrap().iter().find(|c| c["routing"]["primary"] == false).cloned().expect("replica copy");
    let store = replica["store"]["size_in_bytes"].as_u64().unwrap();
    let translog = replica["translog"]["size_in_bytes"].as_u64().unwrap();
    assert!(store.abs_diff(files) < wal / 2, "replica store {store} vs index files {files} (WAL {wal}): the WAL was counted as store");
    assert!(translog >= wal, "replica translog {translog} < its WAL {wal}");
}
