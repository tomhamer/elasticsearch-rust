//! Writes while a replica's node restarts, as ES's `ReplicationOperation`
//! handles them, on three node processes ([`common::procs`]):
//! - the in-sync copy the restarted node had is no longer in the routing
//!   table: the primary marks it stale (`markUnavailableShardsAsStale`)
//!   before acknowledging, without a `_shards.failures` entry;
//! - the copy recovering there gets the writes through its recovery's
//!   translog phase until it is tracked, then from the primary; a copy that
//!   fails a replica operation is failed through the master and recovered
//!   again, never left diverged;
//! - `_shards.total` counts every copy the routing table lists, and any
//!   `_shards.failures` entry has ES's `ShardInfo.Failure` shape: `_index`,
//!   `_shard`, the failed copy's real `_node` id, `reason`, `status`,
//!   `primary: false`;
//! - once green, every copy holds every acknowledged write.
//!
//! They assert ES's behaviour, not this engine's at any one commit.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const NODES: usize = 3;
/// Replica restarts per run (alternating SIGTERM and SIGKILL).
const ROUNDS: usize = 4;
const WRITERS: usize = 3;
const PRELOAD: usize = 20_000;
const FAST_DETECTION: [(&str, &str); 3] = [("cluster.fault_detection.follower_check.interval", "500ms"),
    ("cluster.fault_detection.follower_check.timeout", "2s"), ("cluster.fault_detection.follower_check.retry_count", "2")];

/// What the writers saw.
#[derive(Default)]
struct Seen {
    acked: Vec<String>,
    /// Every `_shards` of an acknowledged write that reported a failure.
    failures: Vec<Value>,
    /// `_shards` objects that broke an invariant (with why).
    bad: Vec<String>,
    /// Acknowledged writes made while a copy was initializing.
    during_recovery: usize,
}

/// `_shards` of an acknowledged write: ES's accounting and failure shape.
fn check_shards(s: &Value, ids: &BTreeMap<usize, String>, index: &str) -> Result<(), String> {
    let (total, ok, failed) = (s["total"].as_u64(), s["successful"].as_u64(), s["failed"].as_u64());
    let (Some(total), Some(ok), Some(failed)) = (total, ok, failed) else { return Err(format!("not numbers: {s}")) };
    // One primary, one replica: the routing table always lists two copies.
    if total != 2 || ok < 1 || ok + failed > total {
        return Err(format!("accounting: {s}"));
    }
    let failures = s["failures"].as_array().map(Vec::as_slice).unwrap_or_default();
    if failures.len() as u64 != failed {
        return Err(format!("failed != failures: {s}"));
    }
    for f in failures {
        let node = f["_node"].as_str().unwrap_or_default();
        let shape = f["_index"] == index && f["_shard"] == 0 && ids.values().any(|id| id == node)
            && f["reason"]["type"].as_str().is_some_and(|t| !t.is_empty()) && f["reason"]["reason"].is_string()
            && f["status"].as_str().is_some_and(|st| st.chars().all(|c| c.is_ascii_uppercase() || c == '_'))
            && f["primary"] == false;
        if !shape {
            return Err(format!("failure shape: {f}"));
        }
    }
    Ok(())
}

/// Single-document writes through `via` until `stop` (none while
/// `paused`), each id once.
#[allow(clippy::too_many_arguments)]
fn writer(pl: &Procs, via: usize, w: usize, index: &str, ids: BTreeMap<usize, String>, recovering: Arc<AtomicBool>,
    paused: Arc<AtomicBool>, stop: Arc<AtomicBool>, seen: Arc<Mutex<Seen>>) -> tokio::task::JoinHandle<()> {
    let (c, port, index) = (pl.c.clone(), pl.p.http[via], index.to_string());
    tokio::spawn(async move {
        let mut i = 0u64;
        while !stop.load(Ordering::Relaxed) {
            if paused.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
                continue;
            }
            let id = format!("w{w}-{i}");
            i += 1;
            let during = recovering.load(Ordering::SeqCst);
            let sent = c.put(format!("http://127.0.0.1:{port}/{index}/_doc/{id}")).timeout(Duration::from_secs(90))
                .json(&json!({"k": id, "x": i})).send().await;
            let Ok(r) = sent else { continue };
            let st = r.status().as_u16();
            let v: Value = r.json().await.unwrap_or(Value::Null);
            if st != 201 && st != 200 {
                continue;
            }
            let mut seen = seen.lock().unwrap();
            if let Err(e) = check_shards(&v["_shards"], &ids, &index) {
                seen.bad.push(format!("{id}: {e}"));
            }
            if v["_shards"]["failed"].as_u64().unwrap_or(0) > 0 {
                seen.failures.push(v["_shards"].clone());
            }
            if during && recovering.load(Ordering::SeqCst) {
                seen.during_recovery += 1;
            }
            seen.acked.push(id);
        }
    })
}

/// Sets `recovering` while any copy of `index` is initializing.
fn monitor(pl: &Procs, via: usize, index: &str, recovering: Arc<AtomicBool>, stop: Arc<AtomicBool>, polls: Arc<AtomicUsize>) -> tokio::task::JoinHandle<()> {
    let (c, url) = (pl.c.clone(), format!("http://127.0.0.1:{}/_cat/shards/{index}?format=json&h=state", pl.p.http[via]));
    tokio::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            if let Ok(r) = c.get(&url).timeout(Duration::from_secs(10)).send().await {
                let rows: Value = r.json().await.unwrap_or(Value::Null);
                let init = rows.as_array().is_some_and(|a| a.iter().any(|r| r["state"] == "INITIALIZING"));
                recovering.store(init, Ordering::SeqCst);
                if init {
                    polls.fetch_add(1, Ordering::Relaxed);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
}

/// Count of `ids` and of all docs in the copy of shard 0 on node `id`.
async fn copy_on(pl: &Procs, via: usize, index: &str, node_id: &str, ids: &[String]) -> (u64, u64) {
    let path = format!("/{index}/_search?preference=_shards:0%7C_only_nodes:{node_id}&request_cache=false");
    let mut held = 0;
    for chunk in ids.chunks(5000) {
        let (st, v) = pl.call(via, Method::POST, &path, Some(json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": chunk}}}))).await;
        assert!(st == 200 && v["_shards"]["failed"] == 0 && v["_shards"]["successful"] == 1, "{node_id}: {st} {v}");
        held += v["hits"]["total"]["value"].as_u64().unwrap();
    }
    let (st, v) = pl.call(via, Method::POST, &path, Some(json!({"size": 0, "track_total_hits": true}))).await;
    assert!(st == 200 && v["_shards"]["successful"] == 1, "{node_id}: {st} {v}");
    (held, v["hits"]["total"]["value"].as_u64().unwrap())
}

/// A replica's node restarts under a steady write load, while its copy
/// recovers: every acknowledged write ends up on every copy, the stale copy
/// is never reported as a failure with no node, and every `_shards` is
/// ES's. Odd rounds kill the node with writes in flight (the primary fails
/// the copy its replica operations cannot reach); even rounds restart it
/// gracefully between writes, which resume once it has rejoined: the old
/// copy is then an in-sync allocation the routing table no longer has, the
/// case that used to be a `_shards.failures` entry with an empty `_node`.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn writes_while_a_replica_restarts_reach_every_copy() {
    let mut pl = start_procs(NODES, &FAST_DETECTION).await;
    let master = pl.master(0).await;
    // Recoveries throttled: a ~10mb copy takes seconds to come back.
    pl.settings(master, "persistent", json!({"indices.recovery.max_bytes_per_sec": "4mb"})).await;
    let (st, v) = pl.call(master, Method::PUT, "/rw", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1},
        "mappings": {"properties": {"k": {"type": "keyword"}, "x": {"type": "long"}, "pad": {"type": "text"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "rw", "green", 90).await;
    let ids = pl.node_ids(master).await;
    let mut acked: Vec<String> = Vec::new();
    let all: Vec<usize> = (0..PRELOAD).collect();
    for chunk in all.chunks(5000) {
        let lines: String = chunk.iter().map(|i| {
            let pad: Vec<String> = (0..60).map(|j| format!("w{}", (i * 31 + j * 7) % 5000)).collect();
            format!("{}\n{}\n", json!({"index": {"_id": format!("p-{i}")}}), json!({"k": format!("p-{i}"), "x": i, "pad": pad.join(" ")}))
        }).collect();
        let (st, v) = pl.ndjson(master, "/rw/_bulk", lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
        acked.extend(chunk.iter().map(|i| format!("p-{i}")));
    }

    let mut failures: Vec<Value> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    let mut during_recovery = 0;
    let mut recovering_polls = 0;
    for round in 0..ROUNDS {
        let rows = pl.shards(master, "rw").await;
        let node_of = |prirep: &str| -> usize {
            rows.iter().find(|r| r["prirep"] == prirep && r["state"] == "STARTED").and_then(|r| r["node"].as_str())
                .unwrap_or_else(|| panic!("{rows:?}")).trim_start_matches('n').parse().unwrap()
        };
        let (primary, replica) = (node_of("p"), node_of("r"));
        let coordinator = (0..NODES).find(|n| *n != primary && *n != replica).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let recovering = Arc::new(AtomicBool::new(false));
        let polls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(Mutex::new(Seen::default()));
        let mon = monitor(&pl, coordinator, "rw", recovering.clone(), stop.clone(), polls.clone());
        let writers: Vec<_> = (0..WRITERS).map(|w| writer(&pl, coordinator, round * WRITERS + w, "rw", ids.clone(), recovering.clone(), paused.clone(), stop.clone(), seen.clone())).collect();
        tokio::time::sleep(Duration::from_secs(1)).await;
        if round % 2 == 0 {
            paused.store(true, Ordering::SeqCst);
            // Let in-flight writes finish.
            tokio::time::sleep(Duration::from_millis(500)).await;
            pl.stop(replica, 30).await;
            pl.spawn(replica, &[]).await;
            pl.wait_nodes(coordinator, NODES, 90).await;
            paused.store(false, Ordering::SeqCst);
        } else {
            pl.kill(replica);
            // Writes go on with the copy gone, then while it recovers.
            tokio::time::sleep(Duration::from_millis(1500)).await;
            pl.spawn(replica, &[]).await;
            pl.wait_nodes(coordinator, NODES, 90).await;
        }
        pl.wait_health(coordinator, "rw", "green", 180).await;
        tokio::time::sleep(Duration::from_secs(1)).await;
        stop.store(true, Ordering::Relaxed);
        for w in writers {
            w.await.unwrap();
        }
        mon.await.unwrap();
        let seen = std::mem::take(&mut *seen.lock().unwrap());
        let empty_node = seen.failures.iter().filter(|s| s["failures"].as_array().is_some_and(|f| f.iter().any(|f| f["_node"] == ""))).count();
        eprintln!("round {round}: {} acked, {} with failures ({empty_node} naming no node), {} bad `_shards`, first: {:?}",
            seen.acked.len(), seen.failures.len(), seen.bad.len(), &seen.bad[..seen.bad.len().min(3)]);
        bad.extend(seen.bad.into_iter().map(|b| format!("round {round}: {b}")));
        assert!(!seen.acked.is_empty(), "round {round}: nothing acknowledged");
        acked.extend(seen.acked);
        failures.extend(seen.failures);
        during_recovery += seen.during_recovery;
        recovering_polls += polls.load(Ordering::Relaxed);

        // Green: every copy holds every acknowledged write, and the same docs.
        let (st, v) = pl.call(coordinator, Method::POST, "/rw/_refresh", None).await;
        assert_eq!(st, 200, "{v}");
        let rows = pl.shards(coordinator, "rw").await;
        let mut copies = Vec::new();
        for r in &rows {
            assert_eq!(r["state"], "STARTED", "{rows:?}");
            let n: usize = r["node"].as_str().unwrap().trim_start_matches('n').parse().unwrap();
            let copy = copy_on(&pl, coordinator, "rw", &ids[&n], &acked).await;
            assert_eq!(copy.0, acked.len() as u64, "round {round}: the copy on n{n} lost acknowledged writes: {copy:?}");
            copies.push((n, copy));
        }
        assert_eq!(copies.len(), 2, "{rows:?}");
        assert_eq!(copies[0].1, copies[1].1, "round {round}: copies diverged: {copies:?}");
    }
    eprintln!("acked {} writes; {during_recovery} while a copy was initializing ({recovering_polls} initializing polls); {} with `_shards.failures`: {:?}",
        acked.len(), failures.len(), &failures[..failures.len().min(5)]);
    assert!(bad.is_empty(), "{} acknowledged writes with a `_shards` ES would not give, first: {:?}", bad.len(), &bad[..bad.len().min(5)]);
    assert!(recovering_polls > 0 && during_recovery > 0, "no write overlapped a recovering copy");
}
