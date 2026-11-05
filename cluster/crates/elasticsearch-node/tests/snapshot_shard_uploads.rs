//! Shard snapshots uploaded by the node holding each primary
//! (`snapshot_api::upload`, ES `SnapshotShardsService`):
//!
//! * a master that holds no primary never receives shard data in its
//!   staging directory, `_status` reports the data nodes' progress, the data
//!   nodes drop their frozen images afterwards, and the snapshot restores;
//! * a data node killed mid-upload fails its shards: `PARTIAL`, the failures
//!   naming that node;
//! * a master killed mid-snapshot: the new master completes it (the data
//!   nodes' uploads are joined, not restarted) and it restores;
//! * `DELETE` of a running snapshot stops the uploads on the data nodes;
//! * a node that cannot upload (no upload action, as an older build): the
//!   master pulls that shard's files instead, and refuses a pull that would
//!   cross the high disk watermark (the shard fails, `PARTIAL`).
//!
//! The first four run real node processes (they kill nodes); the last runs
//! in process (it removes a node's action handler).

mod common;

use common::procs::{start_procs, Procs};
use common::{ports, spawn_in_process};
use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::snapshot_api::upload;
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const DOCS: usize = 4000;

/// Regular files under `dir` (count, bytes); a missing directory is empty.
fn files_under(dir: &Path) -> (usize, u64) {
    let mut out = (0, 0);
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for e in entries.flatten() {
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            let (n, b) = files_under(&e.path());
            out.0 += n;
            out.1 += b;
        } else if ft.is_file() {
            out.0 += 1;
            out.1 += e.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    out
}

fn staging(pl: &Procs, i: usize) -> PathBuf {
    pl.dir.join(format!("n{i}")).join("snapshot-staging")
}

async fn ok(pl: &Procs, i: usize, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = pl.call(i, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path} on n{i}: {st} {v}");
    v
}

fn bulk_body(index: &str, n: usize) -> String {
    let mut nd = String::new();
    for i in 0..n {
        // ~1kb of distinct text per document, so the shards take a while to
        // upload under the repository throttle.
        let text: String = (0..64).map(|k| format!("w{:x} ", (i * 131 + k * 7919) % 100_003)).collect();
        nd.push_str(&format!("{{\"index\":{{\"_index\":\"{index}\",\"_id\":\"{i}\"}}}}\n"));
        nd.push_str(&json!({"n": i, "text": text}).to_string());
        nd.push('\n');
    }
    nd
}

/// Three node processes sharing `repo` (`path.repo`), an fs repository
/// throttled to `rate` per node, and index `idx` (4 primaries, no replica)
/// kept off the elected master. Returns the master's position.
async fn setup(pl: &Procs, repo: &Path, rate: &str) -> usize {
    let m = pl.master(0).await;
    ok(pl, m, Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": repo.to_string_lossy(), "max_snapshot_bytes_per_sec": rate, "compress": false}}))).await;
    ok(pl, m, Method::PUT, "/idx", Some(json!({"settings": {"number_of_shards": 4, "number_of_replicas": 0, "index.routing.allocation.exclude._name": format!("n{m}")}}))).await;
    pl.wait_health(m, "idx", "green", 120).await;
    for chunk in 0..4 {
        let (st, v) = pl.ndjson(m, "/_bulk?refresh=true", bulk_body("idx", DOCS / 4).replace("\"_id\":\"", &format!("\"_id\":\"{chunk}-"))).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
    }
    let rows = pl.shards(m, "idx").await;
    assert!(rows.iter().all(|r| r["node"] != format!("n{m}") && r["state"] == "STARTED"), "primaries must be off the master: {rows:?}");
    m
}

async fn start_snapshot(pl: &Procs, via: usize, name: &str) {
    let v = ok(pl, via, Method::PUT, &format!("/_snapshot/repo/{name}?wait_for_completion=false"), Some(json!({"indices": "idx"}))).await;
    assert_eq!(v["accepted"], true, "{v}");
}

/// The snapshot once it is no longer in progress (as `via` reports it).
async fn wait_done(pl: &Procs, via: usize, name: &str, secs: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (st, v) = pl.call(via, Method::GET, &format!("/_snapshot/repo/{name}"), None).await;
        if st == 200 {
            let s = &v["snapshots"][0];
            if matches!(s["state"].as_str(), Some("SUCCESS" | "PARTIAL" | "FAILED")) {
                return s.clone();
            }
        }
        assert!(Instant::now() < deadline, "snapshot {name} did not finish: {st} {v}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Wait until some shard of the running snapshot is uploading (`STARTED`)
/// and return the `_status` shards.
async fn wait_uploading(pl: &Procs, via: usize, name: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (st, v) = pl.call(via, Method::GET, &format!("/_snapshot/repo/{name}/_status"), None).await;
        let shards = v["snapshots"][0]["indices"]["idx"]["shards"].clone();
        if st == 200 && shards.as_object().is_some_and(|m| m.values().any(|s| s["stage"] == "STARTED" && s["stats"]["processed"]["size_in_bytes"].as_u64().unwrap_or(0) > 0)) {
            return shards;
        }
        assert!(Instant::now() < deadline, "no shard upload started: {st} {v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn count(pl: &Procs, via: usize, index: &str) -> u64 {
    ok(pl, via, Method::POST, &format!("/{index}/_refresh"), None).await;
    ok(pl, via, Method::GET, &format!("/{index}/_count"), None).await["count"].as_u64().unwrap()
}

/// Every shard the finalized snapshot `snap` counts as successful has its
/// shard-level blobs in the repository where the root `index-N` points:
/// `indices/<id>/<k>/snap-<uuid>.dat` and the recorded `index-<gen>`
/// (the repository is uncompressed, so `index-N` is plain JSON). Also the
/// completed `_status`, which reads every shard's `snap-<uuid>.dat`.
async fn assert_shard_snapshots_in_repo(pl: &Procs, via: usize, repo: &Path, snap: &str) {
    let info = ok(pl, via, Method::GET, &format!("/_snapshot/repo/{snap}"), None).await["snapshots"][0].clone();
    let uuid = info["uuid"].as_str().unwrap_or_else(|| panic!("{info}")).to_string();
    let failed: Vec<u64> = info["failures"].as_array().into_iter().flatten().filter_map(|f| f["shard_id"].as_u64()).collect();
    let latest = std::fs::read_dir(repo).unwrap().flatten()
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("index-")).and_then(|g| g.parse::<i64>().ok()))
        .max().expect("a root index-N");
    let data: Value = serde_json::from_slice(&std::fs::read(repo.join(format!("index-{latest}"))).unwrap()).expect("plain index-N");
    let entry = &data["indices"]["idx"];
    let id = entry["id"].as_str().unwrap_or_else(|| panic!("idx not in index-{latest}: {data}"));
    assert!(entry["snapshots"].as_array().is_some_and(|a| a.iter().any(|u| u.as_str() == Some(uuid.as_str()))), "{data}");
    for k in 0..4u64 {
        if failed.contains(&k) {
            continue;
        }
        let dir = repo.join("indices").join(id).join(k.to_string());
        let blob = dir.join(format!("snap-{uuid}.dat"));
        assert!(blob.is_file(), "shard [{k}] counts as successful but {} is missing", blob.display());
        let generation = entry["shard_generations"][k as usize].as_str().unwrap_or_else(|| panic!("no generation for shard {k}: {data}"));
        assert!(dir.join(format!("index-{generation}")).is_file(), "shard [{k}] generation index-{generation} is missing");
    }
    let st = ok(pl, via, Method::GET, &format!("/_snapshot/repo/{snap}/_status"), None).await;
    for k in 0..4u64 {
        if !failed.contains(&k) {
            assert_eq!(st["snapshots"][0]["indices"]["idx"]["shards"][k.to_string()]["stage"], "DONE", "{st}");
        }
    }
}

async fn restore_as(pl: &Procs, via: usize, snap: &str, target: &str) {
    let v = ok(pl, via, Method::POST, &format!("/_snapshot/repo/{snap}/_restore?wait_for_completion=true"),
        Some(json!({"indices": "idx", "rename_pattern": "idx", "rename_replacement": target}))).await;
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    pl.wait_health(via, target, "green", 120).await;
    assert_eq!(count(pl, via, target).await, DOCS as u64);
}

async fn wait_no_staged_files(pl: &Procs, nodes: &[usize]) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let left: Vec<(usize, usize)> = nodes.iter().map(|i| (*i, files_under(&staging(pl, *i)).0)).filter(|(_, n)| *n > 0).collect();
        if left.is_empty() {
            return;
        }
        assert!(Instant::now() < deadline, "frozen images left behind (node, files): {left:?}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn master_without_primaries_never_stages_shard_data() {
    let repo = common::scratch_dir("snap-up-plain-repo");
    let pl = start_procs(3, &[("path.repo", repo.to_str().unwrap())]).await;
    let m = setup(&pl, &repo, "512kb").await;
    let ids = pl.node_ids(m).await;
    let data: Vec<usize> = (0..3).filter(|i| *i != m).collect();
    start_snapshot(&pl, m, "s").await;
    let mut seen_on_data = false;
    let mut status_checked = false;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let (master_files, master_bytes) = files_under(&staging(&pl, m));
        assert_eq!((master_files, master_bytes), (0, 0), "the master staged shard data");
        seen_on_data |= data.iter().any(|i| files_under(&staging(&pl, *i)).0 > 0);
        let (_, v) = pl.call(m, Method::GET, "/_snapshot/repo/s/_status", None).await;
        let s = &v["snapshots"][0];
        if s["state"] == "SUCCESS" || s["state"] == "FAILED" {
            break;
        }
        if let Some(shards) = s["indices"]["idx"]["shards"].as_object() {
            for sh in shards.values().filter(|sh| sh["stage"] == "STARTED") {
                // Progress comes from the data node doing the upload.
                assert_ne!(sh["node"], json!(ids[&m]), "{s}");
                assert!(data.iter().any(|i| sh["node"] == json!(ids[i])), "{s}");
                status_checked = true;
            }
        }
        assert!(Instant::now() < deadline, "snapshot did not finish: {v}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(seen_on_data, "the data nodes froze their primaries");
    assert!(status_checked, "_status reported a running upload");
    let info = wait_done(&pl, m, "s", 60).await;
    assert_eq!(info["state"], "SUCCESS", "{info}");
    assert_eq!((info["shards"]["total"].as_u64(), info["shards"]["successful"].as_u64()), (Some(4), Some(4)), "{info}");
    assert_eq!(files_under(&staging(&pl, m)), (0, 0));
    wait_no_staged_files(&pl, &data).await;
    // The completed status reads the shard snapshots the data nodes wrote.
    let st = ok(&pl, m, Method::GET, "/_snapshot/repo/s/_status", None).await;
    for k in 0..4 {
        let sh = &st["snapshots"][0]["indices"]["idx"]["shards"][k.to_string()];
        assert_eq!(sh["stage"], "DONE", "{st}");
        assert!(sh["stats"]["total"]["file_count"].as_u64().unwrap_or(0) > 0, "{st}");
    }
    // Incremental: an unchanged index uploads nothing the second time.
    ok(&pl, m, Method::PUT, "/_snapshot/repo/s2?wait_for_completion=true", Some(json!({"indices": "idx"}))).await;
    let st = ok(&pl, m, Method::GET, "/_snapshot/repo/s2/_status", None).await;
    assert_eq!(st["snapshots"][0]["stats"]["incremental"]["file_count"], 0, "{st}");
    assert_shard_snapshots_in_repo(&pl, m, &repo, "s").await;
    restore_as(&pl, m, "s", "restored").await;
}

/// Registrations sent right after formation, while the master re-runs its
/// election to collect the grown voting configuration's votes, wait for the
/// re-elected master (ES `TransportMasterNodeAction`) instead of failing
/// with a 500 "is not the elected master".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repository_puts_during_the_post_formation_reelection_wait_for_the_master() {
    let repo = common::scratch_dir("snap-up-reelect-repo");
    let pl = start_procs(3, &[("path.repo", repo.to_str().unwrap())]).await;
    let m = pl.master(0).await;
    let until = Instant::now() + Duration::from_secs(5);
    let mut k = 0;
    while Instant::now() < until {
        let location = repo.join(format!("r{k}"));
        ok(&pl, m, Method::PUT, &format!("/_snapshot/r{k}?verify=false"), Some(json!({"type": "fs", "settings": {"location": location.to_string_lossy()}}))).await;
        k += 1;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn data_node_killed_mid_upload_makes_the_snapshot_partial() {
    let repo = common::scratch_dir("snap-up-kill-repo");
    let mut pl = start_procs(3, &[("path.repo", repo.to_str().unwrap())]).await;
    let m = setup(&pl, &repo, "128kb").await;
    let ids = pl.node_ids(m).await;
    let victim = (0..3).find(|i| *i != m).unwrap();
    start_snapshot(&pl, m, "s").await;
    wait_uploading(&pl, m, "s").await;
    pl.kill(victim);
    let info = wait_done(&pl, m, "s", 180).await;
    assert_eq!(info["state"], "PARTIAL", "{info}");
    let failures = info["failures"].as_array().cloned().unwrap_or_default();
    assert!(!failures.is_empty(), "{info}");
    for f in &failures {
        assert_eq!(f["node_id"], json!(ids[&victim]), "a failure names the node that died: {info}");
    }
    let other = (0..3).find(|i| *i != m && *i != victim).unwrap();
    assert_eq!(info["shards"]["successful"].as_u64(), Some(4 - failures.len() as u64), "{info}");
    assert!(info["shards"]["successful"].as_u64().unwrap() >= 1, "the surviving node's shards are in the snapshot: {info}");
    // The repository stays usable: the next snapshot of what is left works.
    ok(&pl, m, Method::PUT, "/_snapshot/repo/after?wait_for_completion=true", Some(json!({"indices": "idx", "partial": true}))).await;
    assert_eq!(files_under(&staging(&pl, m)), (0, 0));
    wait_no_staged_files(&pl, &[other]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn master_failover_mid_snapshot_completes_on_the_new_master() {
    let repo = common::scratch_dir("snap-up-failover-repo");
    let mut pl = start_procs(3, &[("path.repo", repo.to_str().unwrap())]).await;
    let m = setup(&pl, &repo, "256kb").await;
    start_snapshot(&pl, m, "s").await;
    wait_uploading(&pl, m, "s").await;
    pl.kill(m);
    let via = (0..3).find(|i| *i != m).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while pl.master(via).await == m {
        assert!(Instant::now() < deadline, "no new master");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let info = wait_done(&pl, via, "s", 240).await;
    assert_eq!(info["state"], "SUCCESS", "{info}");
    assert_eq!(info["shards"]["successful"].as_u64(), Some(4), "{info}");
    let survivors: Vec<usize> = (0..3).filter(|i| *i != m).collect();
    wait_no_staged_files(&pl, &survivors).await;
    assert_shard_snapshots_in_repo(&pl, via, &repo, "s").await;
    restore_as(&pl, via, "s", "restored").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deleting_a_running_snapshot_stops_the_data_node_uploads() {
    let repo = common::scratch_dir("snap-up-abort-repo");
    let pl = start_procs(3, &[("path.repo", repo.to_str().unwrap())]).await;
    let m = setup(&pl, &repo, "64kb").await;
    let data: Vec<usize> = (0..3).filter(|i| *i != m).collect();
    start_snapshot(&pl, m, "s").await;
    wait_uploading(&pl, m, "s").await;
    ok(&pl, m, Method::DELETE, "/_snapshot/repo/s", None).await;
    let (st, _) = pl.call(m, Method::GET, "/_snapshot/repo/s", None).await;
    assert_eq!(st, 404);
    wait_no_staged_files(&pl, &data).await;
    // Nothing is written to the repository any more.
    let before = files_under(&repo);
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert_eq!(files_under(&repo), before, "uploads continued after the abort");
    // And the repository takes the next snapshot.
    let v = ok(&pl, m, Method::PUT, "/_snapshot/repo/next?wait_for_completion=true", Some(json!({"indices": "idx"}))).await;
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
}

// ----- mixed capabilities (in process) --------------------------------------

fn node_settings(cluster: &str, i: usize, seeds: &[u16], dir: &Path, repo: &Path) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join(format!("n{i}")).to_string_lossy()));
    put("path.logs", json!(dir.join(format!("logs{i}")).to_string_lossy()));
    put("path.repo", json!(repo.to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_that_cannot_upload_is_pulled_under_the_disk_guard() {
    let cluster = ports::unique("snap-up-mixed");
    let dir = common::scratch_dir(&cluster);
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    let mut nodes = Vec::new();
    let mut http = Vec::new();
    let mut transport: Vec<u16> = Vec::new();
    for i in 0..2 {
        let node = spawn_in_process(NodeConfig::from_settings(node_settings(&cluster, i, &transport, &dir, &repo)).expect("config")).await;
        http.push(ports::http_port(&node));
        transport.push(ports::transport_port(&node));
        nodes.push(node);
    }
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &http, None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    let deadline = Instant::now() + Duration::from_secs(60);
    let m = loop {
        if let Some(m) = nodes.iter().position(|n| n.is_master()) {
            break m;
        }
        assert!(Instant::now() < deadline, "no master");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    let d = 1 - m;
    let call = |method: Method, path: String, body: Option<Value>| {
        let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", http[m])).timeout(Duration::from_secs(120));
        if let Some(b) = body {
            r = r.json(&b);
        }
        async move {
            let resp = r.send().await.expect("request");
            let st = resp.status().as_u16();
            let v: Value = resp.json().await.unwrap_or(Value::Null);
            assert!((200..300).contains(&st), "{path}: {st} {v}");
            v
        }
    };
    call(Method::PUT, "/_snapshot/repo".into(), Some(json!({"type": "fs", "settings": {"location": repo.to_string_lossy()}}))).await;
    call(Method::PUT, "/old".into(), Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.exclude._name": format!("n{m}")}}))).await;
    call(Method::GET, "/_cluster/health/old?wait_for_status=green&timeout=60s".into(), None).await;
    let nd = bulk_body("old", 300);
    let r = c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", http[m])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");

    // `d` behaves as a build without the upload action.
    assert!(nodes[d].transport.unregister_handler(upload::ACTION_UPLOAD));
    let pulled = upload::PULLED_BYTES.load(Ordering::SeqCst);
    let v = call(Method::PUT, "/_snapshot/repo/pulled?wait_for_completion=true".into(), Some(json!({"indices": "old"}))).await;
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    assert!(upload::PULLED_BYTES.load(Ordering::SeqCst) > pulled, "the master pulled the shards");
    let master_staging = dir.join(format!("n{m}")).join("snapshot-staging");
    assert_eq!(files_under(&master_staging).0, 0, "pulled copies are dropped after the upload");
    let v = call(Method::POST, "/_snapshot/repo/pulled/_restore?wait_for_completion=true".into(),
        Some(json!({"indices": "old", "rename_pattern": "old", "rename_replacement": "old-restored"}))).await;
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    call(Method::GET, "/_cluster/health/old-restored?wait_for_status=green&timeout=60s".into(), None).await;
    call(Method::POST, "/old-restored/_refresh".into(), None).await;
    assert_eq!(call(Method::GET, "/old-restored/_count".into(), None).await["count"], 300);

    // A pull that would cross the high watermark fails its shard instead.
    upload::PULL_DISK_RESERVE.store(u64::MAX / 4, Ordering::SeqCst);
    call(Method::POST, "/old/_doc?refresh=true".into(), Some(json!({"n": -1}))).await;
    let pulled = upload::PULLED_BYTES.load(Ordering::SeqCst);
    let v = call(Method::PUT, "/_snapshot/repo/guarded?wait_for_completion=true".into(), Some(json!({"indices": "old"}))).await;
    upload::PULL_DISK_RESERVE.store(u64::MAX, Ordering::SeqCst);
    assert_eq!(v["snapshot"]["state"], "PARTIAL", "{v}");
    let failures = v["snapshot"]["failures"].as_array().cloned().unwrap_or_default();
    assert_eq!(failures.len(), 2, "{v}");
    for f in &failures {
        let reason = f["reason"].as_str().unwrap_or_default();
        assert!(reason.contains("insufficient disk space on the master"), "{reason}");
        assert!(reason.contains("cluster.routing.allocation.disk.watermark.high"), "{reason}");
    }
    assert_eq!(upload::PULLED_BYTES.load(Ordering::SeqCst), pulled, "nothing was copied");
    assert_eq!(files_under(&master_staging).0, 0);
}
