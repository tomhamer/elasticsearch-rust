//! Shard failures with replicas, as ES's `AbstractSearchAsyncAction`
//! handles them, on three node processes ([`common::procs`]):
//! - a shard copy that fails (circuit breaking, a script error, a dead or
//!   hung node) moves the shard on to its next copy (`onShardFailure` ->
//!   `performPhaseOnShard`); a later success clears the earlier copy's
//!   failure;
//! - a shard whose every copy failed is one `ShardSearchFailure` (shard,
//!   index, node, reason) in `_shards.failures` of a 200;
//! - every shard failing is `SearchPhaseExecutionException` "all shards
//!   failed" with the failures' status (429 for `CircuitBreakingException`);
//! - a shard with no active copy is a failed shard of a partial 200, or
//!   (`allow_partial_search_results=false`) 503 "Search rejected due to
//!   missing shards [...]".
//! - Nothing comes up short silently: while copies relocate or recover in
//!   place, a search either counts every doc or reports a failed shard.
//!
//! They assert ES's behaviour, not this engine's at any one commit.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BREAKERS: usize = 3;
const MOVING: usize = 3;
const IN_PLACE: usize = 3;
const SCRIPT: usize = 3;
const NO_COPY: usize = 3;
const HUNG: usize = 3;
const FLEET_PAUSED: usize = 3;
const TIGHT_FAILOVER: usize = 3;
/// Fault detection in a few seconds (ES's defaults: 1s / 10s / 3).
const FAST_DETECTION: [(&str, &str); 3] = [("cluster.fault_detection.follower_check.interval", "500ms"),
    ("cluster.fault_detection.follower_check.timeout", "2s"), ("cluster.fault_detection.follower_check.retry_count", "2")];

fn with_pref(path: &str, pref: &Option<String>) -> String {
    match pref {
        None => path.to_string(),
        Some(p) => format!("{path}{}preference={p}", if path.contains('?') { '&' } else { '?' }),
    }
}

/// A 200 where every shard answered.
async fn search(pl: &Procs, node: usize, path: &str, body: Value) -> Result<Value, String> {
    let (st, v) = pl.call(node, Method::POST, path, Some(body.clone())).await;
    let s = &v["_shards"];
    if st != 200 || s["failed"] != 0 || s["successful"] != s["total"] {
        return Err(format!("{path} {body}: {st} {v}"));
    }
    Ok(v)
}

/// `n` docs `{"k": "<prefix>-i", "x": i}` (routed, if given) into `index`.
async fn fill(pl: &Procs, via: usize, index: &str, n: usize, prefix: &str, routing: Option<&str>) {
    let all: Vec<usize> = (0..n).collect();
    for chunk in all.chunks(5000) {
        let action = match routing {
            Some(r) => json!({"index": {"routing": r}}),
            None => json!({"index": {}}),
        };
        let lines: String = chunk.iter().map(|i| format!("{action}\n{}\n", json!({"k": format!("{prefix}-{i:06}"), "x": i}))).collect();
        let (st, v) = pl.ndjson(via, &format!("/{index}/_bulk"), lines).await;
        assert!(st == 200 && v["errors"] == false, "{v}");
    }
}

async fn create(pl: &Procs, via: usize, index: &str, shards: u32, replicas: u32, extra: Value) {
    let mut settings = json!({"number_of_shards": shards, "number_of_replicas": replicas});
    for (k, v) in extra.as_object().into_iter().flatten() {
        settings[k] = v.clone();
    }
    let (st, v) = pl.call(via, Method::PUT, &format!("/{index}"), Some(json!({"settings": settings,
        "mappings": {"properties": {"k": {"type": "keyword"}, "x": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, index, "green", 90).await;
}

/// This node's own request breaker.
async fn request_breaker(pl: &Procs, node: usize) -> Value {
    let (st, v) = pl.call(node, Method::GET, "/_nodes/_local/stats/breaker", None).await;
    assert_eq!(st, 200, "n{node}: {v}\n{}", pl.log_tail(node));
    v["nodes"].as_object().and_then(|m| m.values().next()).map(|n| n["breakers"]["request"].clone()).unwrap_or(Value::Null)
}

/// Wait until every live node's request breaker has `limit` bytes.
async fn wait_limit(pl: &Procs, limit: u64) {
    for n in pl.live() {
        let deadline = Instant::now() + Duration::from_secs(10);
        while request_breaker(pl, n).await["limit_size_in_bytes"] != limit {
            assert!(Instant::now() < deadline, "n{n} limit: {}", request_breaker(pl, n).await);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Every live node's request breaker drains back to zero.
async fn wait_released(pl: &Procs) {
    for n in pl.live() {
        let deadline = Instant::now() + Duration::from_secs(10);
        while request_breaker(pl, n).await["estimated_size_in_bytes"] != 0 {
            assert!(Instant::now() < deadline, "n{n} request breaker held: {}", request_breaker(pl, n).await);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

fn assert_breaking(e: &Value, limit: u64) {
    assert_eq!(e["type"], "circuit_breaking_exception", "{e}");
    let reason = e["reason"].as_str().unwrap_or_default();
    assert!(reason.starts_with("[request] Data too large, data for ["), "{reason}");
    assert!(reason.contains(&format!("which is larger than the limit of [{limit}/")), "{reason}");
    assert_eq!(e["bytes_limit"], limit, "{e}");
    assert_eq!(e["durability"], "TRANSIENT", "{e}");
}

/// ES's `ShardSearchFailure` of the heavy shard: shard, index, node, reason.
fn assert_heavy_failure(f: &Value, heavy: u64, nodes: &BTreeMap<usize, String>, limit: u64) {
    assert_eq!(f["shard"], heavy, "{f}");
    assert_eq!(f["index"], "brk", "{f}");
    assert!(nodes.values().any(|id| f["node"] == id.as_str()), "{f}");
    assert_breaking(&f["reason"], limit);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tripped_copies_move_on_and_tripped_shards_are_listed() {
    let mut pl = start_procs(BREAKERS, &[]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/brk", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": 2},
        "mappings": {"properties": {"k": {"type": "keyword"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "brk", "green", 90).await;
    // 20k distinct keys routed to one shard (a terms aggregation over them
    // trips a 1mb request breaker), 500 docs of ten keys spread out.
    let all: Vec<usize> = (0..20_000).collect();
    for chunk in all.chunks(5000) {
        let lines: String = chunk.iter().map(|i| format!("{{\"index\":{{\"routing\":\"heavy\"}}}}\n{{\"k\":\"h-{i:06}\"}}\n")).collect();
        let (st, v) = pl.ndjson(master, "/brk/_bulk", lines).await;
        assert!(st == 200 && v["errors"] == false, "{v}");
    }
    let lines: String = (0..500).map(|i| format!("{{\"index\":{{}}}}\n{{\"k\":\"l{}\"}}\n", i % 10)).collect();
    let (st, v) = pl.ndjson(master, "/brk/_bulk?refresh=true", lines).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    pl.wait_health(master, "brk", "green", 90).await;
    let mut heavy = 0;
    let mut most = 0;
    for k in 0..5u64 {
        let v = search(&pl, master, &format!("/brk/_count?preference=_shards:{k}"), json!({})).await.unwrap();
        if v["count"].as_u64().unwrap() > most {
            (heavy, most) = (k, v["count"].as_u64().unwrap());
        }
    }
    assert!(most >= 20_000, "heavy shard {heavy}: {most}");
    let others = (0..5u64).filter(|k| *k != heavy).map(|k| k.to_string()).collect::<Vec<_>>().join(",");
    let agg = json!({"size": 0, "aggs": {"t": {"terms": {"field": "k", "size": 10}}}});
    // Not from (or into) the shard request cache: a `size: 0` result of
    // the heavy shard cached here, before the limit, would answer a later
    // search from cache without running (or tripping) the aggregation, as
    // ES's request cache would. Which copy holds it depends on where this
    // search ran, so the trip below must not depend on it.
    let full = search(&pl, master, "/brk/_search?request_cache=false", agg.clone()).await.unwrap();
    let light = search(&pl, master, &format!("/brk/_search?preference=_shards:{others}"), agg.clone()).await.unwrap();
    let nodes = pl.node_ids(master).await;
    const MB: u64 = 1 << 20;

    // Every copy of the heavy shard trips: that shard is one failure, the
    // other four answer.
    pl.settings(master, "transient", json!({"indices.breaker.request.limit": "1mb"})).await;
    wait_limit(&pl, MB).await;
    for n in 0..pl.len() {
        for pref in [None, Some("_local".to_string())] {
            let path = with_pref("/brk/_search", &pref);
            let (st, v) = pl.call(n, Method::POST, &path, Some(agg.clone())).await;
            assert_eq!(st, 200, "n{n} {path}: partial results: {v}");
            let s = &v["_shards"];
            assert_eq!((&s["total"], &s["successful"], &s["skipped"], &s["failed"]), (&json!(5), &json!(4), &json!(0), &json!(1)), "n{n} {path}: {v}");
            let failures = s["failures"].as_array().unwrap_or_else(|| panic!("no failures: {v}"));
            assert_eq!(failures.len(), 1, "one entry per shard, not per copy: {v}");
            assert_heavy_failure(&failures[0], heavy, &nodes, MB);
            assert_eq!(v["aggregations"]["t"]["buckets"], light["aggregations"]["t"]["buckets"], "n{n} {path}");
        }
    }
    // Only the heavy shard: all shards failed, with the breaker's 429.
    let (st, v) = pl.call(master, Method::POST, &format!("/brk/_search?preference=_shards:{heavy}"), Some(agg.clone())).await;
    assert_eq!(st, 429, "{v}");
    assert_eq!(v["status"], 429, "{v}");
    let e = &v["error"];
    assert_eq!(e["type"], "search_phase_execution_exception", "{v}");
    assert_eq!(e["reason"], "all shards failed", "{v}");
    assert_eq!(e["phase"], "query", "{v}");
    assert_eq!(e["grouped"], true, "{v}");
    assert_eq!(e["failed_shards"].as_array().map_or(0, Vec::len), 1, "{v}");
    assert_heavy_failure(&e["failed_shards"][0], heavy, &nodes, MB);
    assert_breaking(&e["root_cause"][0], MB);

    // The default limit again: everything answers, the breakers drain.
    pl.settings(master, "transient", json!({"indices.breaker.request.limit": null})).await;
    let v = search(&pl, master, "/brk/_search", agg.clone()).await.unwrap();
    assert_eq!(v["aggregations"], full["aggregations"]);
    wait_released(&pl).await;

    // Only n2's copies trip (its own node setting): a search coordinated by
    // n2 preferring its local copies moves the heavy shard on to another
    // copy, and the retried shard is no failure.
    let n2 = pl.len() - 1;
    pl.kill(n2);
    pl.spawn(n2, &[("indices.breaker.request.limit", "1mb")]).await;
    // The restarted node itself must have joined: the master may not have
    // noticed the killed process yet (it still counts three nodes, the
    // index still green), and the new process answers HTTP only once up.
    pl.wait_nodes(n2, pl.len(), 90).await;
    pl.wait_nodes(master, pl.len(), 90).await;
    pl.wait_health(master, "brk", "green", 120).await;
    let nodes = pl.node_ids(master).await;
    assert_eq!(request_breaker(&pl, n2).await["limit_size_in_bytes"], MB);
    let before = request_breaker(&pl, n2).await["tripped"].as_u64().unwrap_or(0);
    let v = search(&pl, n2, "/brk/_search?preference=_local", agg.clone()).await.unwrap_or_else(|e| panic!("retried on another copy: {e}"));
    assert!(v["_shards"].get("failures").is_none_or(|f| f.as_array().is_some_and(Vec::is_empty)), "{v}");
    assert_eq!(v["aggregations"], full["aggregations"]);
    assert!(request_breaker(&pl, n2).await["tripped"].as_u64().unwrap_or(0) > before, "n2's own copy was tried first");
    // `_only_local`: no other copy to move on to.
    let (st, v) = pl.call(n2, Method::POST, "/brk/_search?preference=_only_local", Some(agg.clone())).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_shards"]["failed"], 1, "{v}");
    assert_heavy_failure(&v["_shards"]["failures"][0], heavy, &nodes, MB);
    assert_eq!(v["_shards"]["failures"][0]["node"], nodes[&n2].as_str(), "{v}");
    wait_released(&pl).await;
}

/// One `size: 0` counting search through `port`: (status, `_shards.failed`,
/// total hits).
async fn count_once(c: &reqwest::Client, port: u16, path: &str) -> (u16, u64, u64) {
    let (st, failed, total, _) = count_body(c, port, path).await;
    (st, failed, total)
}

/// [`count_once`] with the response body.
async fn count_body(c: &reqwest::Client, port: u16, path: &str) -> (u16, u64, u64, Value) {
    let r = c.post(format!("http://127.0.0.1:{port}{path}")).timeout(Duration::from_secs(60)).json(&json!({"size": 0, "track_total_hits": true})).send().await;
    let Ok(r) = r else { return (0, 0, 0, Value::Null) };
    let st = r.status().as_u16();
    let v: Value = r.json().await.unwrap_or(Value::Null);
    (st, v["_shards"]["failed"].as_u64().unwrap_or(0), v["hits"]["total"]["value"].as_u64().unwrap_or(0), v)
}

/// ES's answer when the one shard a search targets has no copy it may use
/// (`_shards:k|_only_local` on the node whose copy is unassigned or
/// recovering): `onlyNodeActiveInitializingShardsIt` leaves no copy, or
/// only the initializing one, whose shard request fails as unavailable
/// (`IllegalIndexShardStateException`); `AbstractSearchAsyncAction` wraps
/// it as `NoShardAvailableActionException` and, with no shard left,
/// answers 503 `SearchPhaseExecutionException` "all shards failed".
fn only_shard_unavailable(st: u16, v: &Value) -> bool {
    let e = &v["error"];
    let failed = e["failed_shards"].as_array();
    st == 503 && v["status"] == 503 && e["type"] == "search_phase_execution_exception" && e["reason"] == "all shards failed"
        && failed.is_some_and(|f| f.len() == 1 && f[0]["reason"]["type"] == "no_shard_available_action_exception")
}

/// `n` docs like [`fill`]'s with a ~400-byte text `pad`: shards large
/// enough that moving one takes a while.
async fn fill_padded(pl: &Procs, via: usize, index: &str, n: usize) {
    let all: Vec<usize> = (0..n).collect();
    for chunk in all.chunks(5000) {
        let lines: String = chunk.iter().map(|i| {
            let pad: Vec<String> = (0..60).map(|j| format!("w{}", (i * 31 + j * 7) % 5000)).collect();
            format!("{{\"index\":{{}}}}\n{}\n", json!({"k": format!("{index}-{i:07}"), "x": i, "pad": pad.join(" ")}))
        }).collect();
        let (st, v) = pl.ndjson(via, &format!("/{index}/_bulk"), lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
    }
}

/// Recovery throttled (`indices.recovery.max_bytes_per_sec`, a dynamic ES
/// setting), so each move of a ~5mb shard takes a second or more.
async fn throttle_recoveries(pl: &Procs, via: usize) {
    pl.settings(via, "persistent", json!({"indices.recovery.max_bytes_per_sec": "4mb"})).await;
}

/// Per move (an allocation change, then the wait for green): whether the
/// monitor saw a copy relocating or initializing, and how many searches ran
/// while it did.
#[derive(Default, Clone, Copy, Debug)]
struct Move {
    saw_moving: bool,
    overlapped: usize,
}

/// Shared by the monitor, the searchers and the test body.
#[derive(Clone)]
struct Moves {
    /// The move under way (an index into `stats`).
    current: Arc<std::sync::atomic::AtomicUsize>,
    moving: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    stats: Arc<std::sync::Mutex<Vec<Move>>>,
}

impl Moves {
    fn new(n: usize) -> Moves {
        Moves { current: Arc::default(), moving: Arc::default(), stop: Arc::default(), stats: Arc::new(std::sync::Mutex::new(vec![Move::default(); n])) }
    }

    fn start(&self, i: usize) {
        self.current.store(i, Ordering::SeqCst);
    }
}

/// Polls `_cat/shards` of `indices` until stopped, marking the current move
/// as seen moving while any copy is relocating or initializing.
fn shard_monitor(pl: &Procs, via: usize, indices: &str, m: Moves) -> tokio::task::JoinHandle<()> {
    let (c, url) = (pl.c.clone(), format!("http://127.0.0.1:{}/_cat/shards/{indices}?format=json&h=state", pl.p.http[via]));
    tokio::spawn(async move {
        while !m.stop.load(Ordering::Relaxed) {
            let step = m.current.load(Ordering::SeqCst);
            if let Ok(r) = c.get(&url).timeout(Duration::from_secs(10)).send().await {
                let rows: Value = r.json().await.unwrap_or(Value::Null);
                let busy = rows.as_array().is_some_and(|a| a.iter().any(|r| r["state"] == "RELOCATING" || r["state"] == "INITIALIZING"));
                m.moving.store(busy, Ordering::SeqCst);
                if busy {
                    m.stats.lock().unwrap()[step].saw_moving = true;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
}

/// What the counting searches saw: how many ran, and every answer that was
/// not ES's (a 200 with the exact total, a 200 naming a failed shard, or,
/// for a one-shard target, ES's [`only_shard_unavailable`]).
struct Counted {
    runs: usize,
    bad: Vec<String>,
}

/// Count `targets` (path, expected total) from every node until stopped;
/// a search that starts or ends while a copy moves counts for that move.
fn counters(pl: &Procs, targets: Vec<(String, u64)>, m: Moves) -> Vec<tokio::task::JoinHandle<Counted>> {
    pl.live().into_iter().map(|n| {
        let (c, port, targets, m) = (pl.c.clone(), pl.p.http[n], targets.clone(), m.clone());
        tokio::spawn(async move {
            let mut out = Counted { runs: 0, bad: Vec::new() };
            while !m.stop.load(Ordering::Relaxed) {
                for (path, want) in &targets {
                    let (step, before) = (m.current.load(Ordering::SeqCst), m.moving.load(Ordering::SeqCst));
                    let (st, failed, total, v) = count_body(&c, port, path).await;
                    out.runs += 1;
                    if before || m.moving.load(Ordering::SeqCst) {
                        m.stats.lock().unwrap()[step].overlapped += 1;
                    }
                    let one_shard = path.contains("preference=_shards:") && !path.contains("%2C") && !path.contains(',');
                    match (st, failed) {
                        (200, 0) if total == *want => {}
                        (200, 0) => out.bad.push(format!("n{n} {path}: {total} of {want} with no failed shard")),
                        (200, _) => {}
                        _ if one_shard && only_shard_unavailable(st, &v) => {}
                        _ => out.bad.push(format!("n{n} {path}: status {st} {v}")),
                    }
                }
            }
            out
        })
    }).collect()
}

/// Every answer was ES's, every move in `must_move` was seen moving, and
/// every move seen moving had at least `per_move` searches running during it.
async fn finish(m: Moves, tasks: Vec<tokio::task::JoinHandle<Counted>>, monitor: tokio::task::JoinHandle<()>, must_move: &[usize], per_move: usize) {
    m.stop.store(true, Ordering::Relaxed);
    let (mut runs, mut bad) = (0, Vec::new());
    for t in tasks {
        let c = t.await.unwrap();
        runs += c.runs;
        bad.extend(c.bad);
    }
    monitor.await.unwrap();
    bad.truncate(40);
    assert!(bad.is_empty(), "{} of {runs} searches were not ES's answer:\n{}", bad.len(), bad.join("\n"));
    let stats = m.stats.lock().unwrap().clone();
    for (i, s) in stats.iter().enumerate().skip(1) {
        assert!(s.saw_moving || !must_move.contains(&i), "move {i}: no copy was seen relocating or initializing: {stats:?}");
        assert!(!s.saw_moving || s.overlapped >= per_move, "move {i}: only {} searches ran while copies moved (want {per_move}): {stats:?}", s.overlapped);
    }
}

const MOVING_DOCS: usize = 60_000;

/// P0-1: while copies relocate (allocation exclude toggled through each
/// node), every search counts every doc or reports a failed shard, and
/// during each move several searches run while a copy is relocating or
/// initializing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn searches_never_come_up_short_while_copies_relocate() {
    let pl = start_procs(MOVING, &[]).await;
    let master = pl.master(0).await;
    throttle_recoveries(&pl, master).await;
    for (index, replicas) in [("one", 1), ("two", 2)] {
        create(&pl, master, index, 5, replicas, json!({})).await;
        fill_padded(&pl, master, index, MOVING_DOCS).await;
    }
    let (st, v) = pl.call(master, Method::POST, "/one,two/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let excludes = [json!("n0"), json!("n1"), json!("n2"), Value::Null, json!("n1"), Value::Null];
    // Move 0 is the idle time before the first change.
    let m = Moves::new(excludes.len() + 1);
    let monitor = shard_monitor(&pl, master, "one,two", m.clone());
    let n = MOVING_DOCS as u64;
    let targets = vec![("/one/_search".to_string(), n), ("/two/_search".to_string(), n), ("/one,two/_search".to_string(), 2 * n)];
    let tasks = counters(&pl, targets, m.clone());
    for (i, exclude) in excludes.iter().enumerate() {
        m.start(i + 1);
        for index in ["one", "two"] {
            let (st, v) = pl.call(master, Method::PUT, &format!("/{index}/_settings"), Some(json!({"index.routing.allocation.exclude._name": exclude.clone()}))).await;
            assert_eq!(st, 200, "{v}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        // 5x1 moves off the excluded node; 5x2 cannot (every node holds a
        // copy) and stays green.
        pl.wait_health(master, "one", "green", 120).await;
        pl.wait_health(master, "two", "green", 120).await;
    }
    // Excluding a node always moves the 5x1 copies it holds; clearing the
    // exclusion may or may not rebalance (checked only if it moved).
    finish(m, tasks, monitor, &[1, 2, 3, 5], 3).await;
}

/// P0-1: a replica failed and recovered in place (`_cluster/reroute`
/// `cancel`: with every node holding a copy it can only come back where it
/// was) is never searched while it is empty or partial; each recovery
/// overlaps several searches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_copy_recovering_in_place_is_never_served_short() {
    let pl = start_procs(IN_PLACE, &[]).await;
    let master = pl.master(0).await;
    throttle_recoveries(&pl, master).await;
    create(&pl, master, "inplace", 5, 2, json!({})).await;
    fill_padded(&pl, master, "inplace", MOVING_DOCS).await;
    let (st, v) = pl.call(master, Method::POST, "/inplace/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let mut targets = vec![("/inplace/_search".to_string(), MOVING_DOCS as u64)];
    for k in 0..5 {
        let v = search(&pl, master, &format!("/inplace/_count?preference=_shards:{k}"), json!({})).await.unwrap();
        targets.push((format!("/inplace/_search?preference=_shards:{k}%7C_only_local"), v["count"].as_u64().unwrap()));
    }
    let m = Moves::new(6);
    let monitor = shard_monitor(&pl, master, "inplace", m.clone());
    let tasks = counters(&pl, targets, m.clone());
    for k in 0..5u64 {
        m.start(k as usize + 1);
        let rows = pl.shards(master, "inplace").await;
        let replica = rows.iter().find(|r| r["shard"] == k.to_string() && r["prirep"] == "r" && r["state"] == "STARTED").unwrap_or_else(|| panic!("no replica of {k}: {}", json!(rows)));
        let node = replica["node"].as_str().unwrap().to_string();
        let (st, v) = pl.call(master, Method::POST, "/_cluster/reroute", Some(json!({"commands": [{"cancel": {"index": "inplace", "shard": k, "node": node, "allow_primary": false}}]}))).await;
        assert_eq!(st, 200, "{v}");
        tokio::time::sleep(Duration::from_millis(200)).await;
        pl.wait_health(master, "inplace", "green", 120).await;
    }
    // A copy reset in place may recover by reusing its committed prefix
    // (operations only, not throttled, ES's operations-based recovery):
    // faster than the shard monitor's poll. Only the moves it saw are held
    // to the overlap rule, and at least one full recovery must be seen.
    let seen = m.clone();
    finish(m, tasks, monitor, &[], 3).await;
    let observed = seen.stats.lock().unwrap().iter().skip(1).filter(|s| s.saw_moving).count();
    assert!(observed >= 1, "no recovery in place was seen initializing");
}

/// P1-1: a script failing on every copy of one shard only (its docs lack
/// the field) fails that shard alone: 200, one failure naming the shard.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_script_failing_on_one_shard_is_a_partial_failure() {
    let pl = start_procs(SCRIPT, &[]).await;
    let master = pl.master(0).await;
    create(&pl, master, "scr", 5, 2, json!({})).await;
    fill(&pl, master, "scr", 2000, "x", None).await;
    let lines: String = (0..50).map(|i| format!("{{\"index\":{{\"routing\":\"nox\"}}}}\n{{\"k\":\"nox-{i}\"}}\n")).collect();
    let (st, v) = pl.ndjson(master, "/scr/_bulk", lines).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    // The whole index, as ES would need too: a bulk's refresh=true refreshes
    // only the shards it wrote (here the nox shard), and a shard searched
    // within its 1s refresh_interval of the fill shows none of it.
    let (st, v) = pl.call(master, Method::POST, "/scr/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let all = search(&pl, master, "/scr/_count", json!({})).await.unwrap();
    assert_eq!(all["count"], 2050, "every doc visible before the partial search: {all}");
    let mut bad = None;
    for k in 0..5u64 {
        let v = search(&pl, master, &format!("/scr/_count?preference=_shards:{k}"), json!({"query": {"bool": {"must_not": {"exists": {"field": "x"}}}}})).await.unwrap();
        if v["count"].as_u64().unwrap() > 0 {
            bad = Some(k);
        }
    }
    let bad = bad.expect("the nox shard");
    let v = search(&pl, master, &format!("/scr/_count?preference=_shards:{bad}"), json!({"query": {"exists": {"field": "x"}}})).await.unwrap();
    let lost = v["count"].as_u64().unwrap();
    let q = json!({"size": 10, "track_total_hits": true, "query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "doc['x'].value"}}}});
    for n in pl.live() {
        let (st, v) = pl.call(n, Method::POST, "/scr/_search", Some(q.clone())).await;
        assert_eq!(st, 200, "n{n}: partial results: {v}");
        let s = &v["_shards"];
        assert_eq!((&s["total"], &s["successful"], &s["failed"]), (&json!(5), &json!(4), &json!(1)), "n{n}: {v}");
        let f = &s["failures"][0];
        assert_eq!(s["failures"].as_array().map_or(0, Vec::len), 1, "n{n}: one entry per shard: {v}");
        assert_eq!((&f["shard"], &f["index"]), (&json!(bad), &json!("scr")), "n{n}: {v}");
        assert_eq!(f["reason"]["type"], "script_exception", "n{n}: {v}");
        assert_eq!(v["hits"]["total"]["value"], 2000 - lost, "n{n}: the other shards' docs: {v}");
    }
}

/// P1-2: a shard with no active copy (5x0, its node killed, no delayed
/// allocation) fails alone, at once; a 5x1 index promotes and answers in
/// full.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shard_without_an_active_copy_fails_alone() {
    let mut pl = start_procs(NO_COPY, &FAST_DETECTION).await;
    let master = pl.master(0).await;
    let now = json!({"index.unassigned.node_left.delayed_timeout": "0"});
    create(&pl, master, "bare", 6, 0, now.clone()).await;
    create(&pl, master, "held", 6, 1, now).await;
    fill(&pl, master, "bare", 1200, "b", None).await;
    fill(&pl, master, "held", 1200, "h", None).await;
    let (st, v) = pl.call(master, Method::POST, "/bare,held/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let victim = (master + 1) % pl.len();
    let rows = pl.shards(master, "bare").await;
    let gone: Vec<u64> = rows.iter().filter(|r| r["node"] == format!("n{victim}")).filter_map(|r| r["shard"].as_str()?.parse().ok()).collect();
    assert!(!gone.is_empty(), "{}", json!(rows));
    let mut lost_docs = 0;
    for k in &gone {
        lost_docs += search(&pl, master, &format!("/bare/_count?preference=_shards:{k}"), json!({})).await.unwrap()["count"].as_u64().unwrap();
    }
    pl.kill(victim);
    pl.wait_nodes(master, pl.len() - 1, 60).await;
    pl.wait_health(master, "held", "yellow", 60).await;
    for n in pl.live() {
        let v = search(&pl, n, "/held/_search", json!({"size": 0, "track_total_hits": true})).await.unwrap();
        assert_eq!(v["hits"]["total"]["value"], 1200, "n{n}: {v}");
        let t = Instant::now();
        let (st, v) = pl.call(n, Method::POST, "/bare/_search", Some(json!({"size": 0, "track_total_hits": true}))).await;
        assert!(t.elapsed() < Duration::from_secs(5), "n{n}: answered after {:?}", t.elapsed());
        assert_eq!(st, 200, "n{n}: partial results: {v}");
        let s = &v["_shards"];
        let lost = gone.len() as u64;
        assert_eq!((&s["total"], &s["successful"], &s["failed"]), (&json!(6), &json!(6 - lost), &json!(lost)), "n{n}: {v}");
        assert_eq!(v["hits"]["total"]["value"], 1200 - lost_docs, "n{n}: {v}");
        let (st, v) = pl.call(n, Method::POST, "/bare/_search?allow_partial_search_results=false", Some(json!({"size": 0}))).await;
        assert_eq!(st, 503, "n{n}: {v}");
        assert_eq!(v["error"]["type"], "search_phase_execution_exception", "n{n}: {v}");
        // ES 9.5.5's shape: `run()` throws the rejection, which
        // `executePhase` wraps (`onPhaseFailure(phase, "", e)`): an empty
        // top-level reason, the message in `caused_by` (see
        // `stateless_read::missing_shards_rejected`).
        assert_eq!(v["error"]["reason"], "", "n{n}: {v}");
        assert_eq!(v["error"]["caused_by"]["type"], "search_phase_execution_exception", "n{n}: {v}");
        assert!(v["error"]["caused_by"]["reason"].as_str().unwrap_or_default().starts_with("Search rejected due to missing shards ["), "n{n}: {v}");
    }
}

/// P1-3 / P1-7: a hung node (SIGSTOP: its connections stay open) holding a
/// copy of every shard delays searches only until fault detection drops it;
/// single-shard, multi-index and multi-shard searches all answer in full.
/// Once it is back, every search context is released.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hung_node_delays_searches_only_until_it_is_dropped() {
    let pl = start_procs(HUNG, &FAST_DETECTION).await;
    let master = pl.master(0).await;
    for (index, shards) in [("a", 1), ("b", 1), ("s", 5)] {
        create(&pl, master, index, shards, 2, json!({})).await;
        fill(&pl, master, index, 500, index, None).await;
    }
    let (st, v) = pl.call(master, Method::POST, "/a,b,s/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let victim = (master + 1) % pl.len();
    let other = (master + 2) % pl.len();
    pl.pause(victim);
    // Detection takes about interval + retries x timeout (~5s here).
    let bound = Duration::from_secs(12);
    let mut slow = Vec::new();
    for _ in 0..10 {
        for n in [master, other] {
            for (path, want) in [("/a/_search", 500), ("/a,b/_search", 1000), ("/s/_search", 500)] {
                let t = Instant::now();
                let r = tokio::time::timeout(bound, count_once(&pl.c, pl.p.http[n], path)).await;
                match r {
                    Ok((200, 0, total)) if total == want => {}
                    Ok(got) => slow.push(format!("n{n} {path}: {got:?} after {:?}", t.elapsed())),
                    Err(_) => slow.push(format!("n{n} {path}: no answer within {bound:?}")),
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    pl.resume(victim);
    pl.wait_nodes(master, pl.len(), 60).await;
    for index in ["a", "b", "s"] {
        pl.wait_health(master, index, "green", 90).await;
    }
    assert!(slow.is_empty(), "{} searches failed or hung:\n{}", slow.len(), slow.join("\n"));
    for n in pl.live() {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let (_, v) = pl.call(n, Method::GET, "/_nodes/_local/stats/indices/search", None).await;
            let open = v["nodes"].as_object().and_then(|m| m.values().next()).and_then(|x| x["indices"]["search"]["open_contexts"].as_u64());
            if open == Some(0) {
                break;
            }
            assert!(Instant::now() < deadline, "n{n} search contexts still open: {open:?}");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}

/// A fleet search waiting for its checkpoints waits only on the copy it
/// searches (ES `SearchService`'s wait in the shard query): a replica on a
/// hung node (SIGSTOP, still routed) does not fail or hold up the search
/// that the primary serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fleet_search_is_not_held_up_by_a_hung_replica() {
    let pl = start_procs(FLEET_PAUSED, &[]).await;
    let master = pl.master(0).await;
    create(&pl, master, "fl", 1, 1, json!({"refresh_interval": "1s"})).await;
    let rows = pl.shards(master, "fl").await;
    let at = |prirep: &str| -> usize {
        let name = rows.iter().find(|r| r["prirep"] == prirep && r["state"] == "STARTED").and_then(|r| r["node"].as_str()).unwrap_or_else(|| panic!("{rows:?}")).to_string();
        name.trim_start_matches('n').parse().unwrap()
    };
    let (primary, replica) = (at("p"), at("r"));
    let (st, v) = pl.call(primary, Method::PUT, "/fl/_doc/1", Some(json!({"k": "one", "x": 1}))).await;
    assert_eq!(st, 201, "{v}");
    let seq_no = v["_seq_no"].as_i64().unwrap();
    pl.pause(replica);
    let started = Instant::now();
    let r = tokio::time::timeout(Duration::from_secs(10), pl.call(primary, Method::POST, &format!("/fl/_fleet/_fleet_search?wait_for_checkpoints={seq_no}"), Some(json!({"track_total_hits": true})))).await;
    pl.resume(replica);
    let (st, v) = r.expect("the primary's copy answers without the hung replica");
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["hits"]["total"]["value"].as_u64(), v["_shards"]["failed"].as_u64()), (Some(1), Some(0)), "{v}");
    assert!(started.elapsed() < Duration::from_secs(5), "took {:?}", started.elapsed());
}

/// The documents of `ids` that `node`'s own copy of the one-shard `index`
/// holds, and that copy's document count.
async fn local_copy(pl: &Procs, node: usize, index: &str, ids: &[String]) -> (u64, u64) {
    let path = format!("/{index}/_search?preference=_shards:0%7C_only_local&request_cache=false");
    let held = search(pl, node, &path, json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": ids}}})).await
        .unwrap_or_else(|e| panic!("n{node}: {e}"));
    let all = search(pl, node, &path, json!({"size": 0, "track_total_hits": true})).await.unwrap_or_else(|e| panic!("n{node}: {e}"));
    (held["hits"]["total"]["value"].as_u64().unwrap(), all["hits"]["total"]["value"].as_u64().unwrap())
}

/// Control-plane work is never dropped for memory (ES does not apply the
/// request breaker to shard-state reports, recovery control or the
/// primary-replica resync). Every node runs a 1mb request breaker while
/// aggregations keep tripping it and writes keep arriving; the primary's
/// node is killed. The promoted primary resyncs the other copy and
/// replicates to it, and once the old node is back its copy recovers and is
/// reported started: the shard goes green and every copy holds the same
/// documents, every acknowledged write included. Before, a refused resync
/// was skipped (diverged copies kept serving), a refused replication target
/// left a started copy out of every write, and a refused shard report left
/// a copy initializing for good.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failover_under_a_tight_breaker_resyncs_and_reports_every_copy() {
    let mut extra = vec![("indices.breaker.request.limit", "1mb")];
    extra.extend(FAST_DETECTION);
    let mut pl = start_procs(TIGHT_FAILOVER, &extra).await;
    let master = pl.master(0).await;
    let now = json!({"index.unassigned.node_left.delayed_timeout": "0"});
    create(&pl, master, "rsy", 1, 2, now.clone()).await;
    create(&pl, master, "hvy", 1, 2, now).await;
    // A terms aggregation over 20k distinct keys trips the 1mb breaker on
    // every copy: run continuously, it keeps the breaker near its limit.
    // Loaded in small bulks, each well under the breaker.
    let keys: Vec<usize> = (0..20_000).collect();
    for chunk in keys.chunks(250) {
        let lines: String = chunk.iter().map(|i| format!("{{\"index\":{{}}}}\n{{\"k\":\"h-{i:06}\"}}\n")).collect();
        let (st, v) = pl.ndjson(master, "/hvy/_bulk", lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
    }
    let mut acked: Vec<String> = Vec::new();
    for chunk in (0..2000).collect::<Vec<usize>>().chunks(250) {
        let lines: String = chunk.iter().map(|i| format!("{{\"index\":{{\"_id\":\"d-{i}\"}}}}\n{}\n", json!({"k": format!("d-{i}"), "x": i}))).collect();
        let (st, v) = pl.ndjson(master, "/rsy/_bulk", lines).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
        acked.extend(chunk.iter().map(|i| format!("d-{i}")));
    }
    let (st, v) = pl.call(master, Method::POST, "/hvy,rsy/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "rsy", "green", 90).await;

    let rows = pl.shards(master, "rsy").await;
    let victim: usize = rows.iter().find(|r| r["prirep"] == "p" && r["state"] == "STARTED").and_then(|r| r["node"].as_str())
        .unwrap_or_else(|| panic!("{rows:?}")).trim_start_matches('n').parse().unwrap();
    let survivor = (victim + 1) % pl.len();

    let stop = Arc::new(AtomicBool::new(false));
    let pressure: Vec<_> = pl.live().into_iter().map(|n| {
        let (c, port, stop) = (pl.c.clone(), pl.p.http[n], stop.clone());
        tokio::spawn(async move {
            let agg = json!({"size": 0, "aggs": {"t": {"terms": {"field": "k", "size": 10}}}});
            while !stop.load(Ordering::Relaxed) {
                let sent = c.post(format!("http://127.0.0.1:{port}/hvy/_search?request_cache=false")).timeout(Duration::from_secs(30)).json(&agg).send().await;
                if sent.is_err() {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        })
    }).collect();
    // Writes through the failover, coordinated by a node that stays up;
    // only acknowledged ones are counted.
    let writer = {
        let (c, port, stop) = (pl.c.clone(), pl.p.http[survivor], stop.clone());
        tokio::spawn(async move {
            let mut acked = Vec::new();
            let mut i = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let id = format!("w-{i}");
                i += 1;
                let r = c.put(format!("http://127.0.0.1:{port}/rsy/_doc/{id}")).timeout(Duration::from_secs(60)).json(&json!({"k": id, "x": i})).send().await;
                if r.is_ok_and(|r| r.status().as_u16() == 201) {
                    acked.push(id);
                }
            }
            acked
        })
    };
    tokio::time::sleep(Duration::from_secs(2)).await;
    pl.kill(victim);
    pl.wait_nodes(survivor, pl.len() - 1, 90).await;
    pl.wait_health(survivor, "rsy", "yellow", 120).await;
    // Writes go on against the promoted primary for a while.
    tokio::time::sleep(Duration::from_secs(5)).await;
    stop.store(true, Ordering::Relaxed);
    acked.extend(writer.await.unwrap());
    for task in pressure {
        let _ = task.await;
    }
    assert!(acked.len() > 2000, "no write was acknowledged around the failover");
    let (st, v) = pl.call(survivor, Method::POST, "/rsy/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    // The two copies left agree, and hold every acknowledged write.
    let live = pl.live();
    let first = local_copy(&pl, live[0], "rsy", &acked).await;
    for &n in &live {
        let copy = local_copy(&pl, n, "rsy", &acked).await;
        assert_eq!(copy.0, acked.len() as u64, "n{n} lost acknowledged writes: {copy:?}");
        assert_eq!(copy, first, "n{n} and n{} diverged", live[0]);
    }

    // The old primary's node comes back (still under the tight breaker):
    // its copy recovers, is reported started, and agrees with the others.
    pl.spawn(victim, &[]).await;
    pl.wait_nodes(survivor, pl.len(), 90).await;
    pl.wait_health(survivor, "rsy", "green", 180).await;
    let (st, v) = pl.call(survivor, Method::POST, "/rsy/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    for n in pl.live() {
        let copy = local_copy(&pl, n, "rsy", &acked).await;
        assert_eq!(copy, first, "n{n} after the old primary's node returned");
    }
    wait_released(&pl).await;
}
