//! `merges.current` while background merges are mid-flight: shard, index
//! (`_stats`) and node (`_nodes/stats`) level all report the running merges
//! (Rally's wait-until-merges-finish polls these and must not see 0).
//! Merges are held inside their I/O with the engine's merge-run seam.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

fn settings(dir: &std::path::Path) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!("merge-stats"));
    put("node.name", json!("n0"));
    put("path.data", json!(dir.join("n0").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.type", json!("single-node"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

/// Held merges and the gate releasing them.
struct Gate {
    held: AtomicU64,
    open: Mutex<bool>,
    cv: Condvar,
}

impl Gate {
    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

fn sum_shards(v: &Value) -> u64 {
    v["indices"]["m"]["shards"].as_object().map(|m| m.values().flat_map(|copies| copies.as_array().cloned().unwrap_or_default()).map(|c| c["merges"]["current"].as_u64().unwrap_or(0)).sum()).unwrap_or(0)
}

fn node_current(v: &Value) -> u64 {
    v["nodes"].as_object().map(|m| m.values().map(|n| n["indices"]["merges"]["current"].as_u64().unwrap_or(0)).sum()).unwrap_or(0)
}

/// The index metric is `merge` (ES `CommonStatsFlags`); `merges` and other
/// unknown index metrics are ES's 400, not a response without the section.
async fn unknown_index_metrics_are_rejected(c: &reqwest::Client, http: u16) {
    for (path, want) in [
        ("/_nodes/stats/indices/merges", "request [/_nodes/stats/indices/merges] contains unrecognized index metric: [merges] -> did you mean [merge]?"),
        ("/_nodes/stats/indices/bogus", "request [/_nodes/stats/indices/bogus] contains unrecognized index metric: [bogus]"),
        ("/_nodes/_local/stats/indices/merge,merges", "request [/_nodes/_local/stats/indices/merge,merges] contains unrecognized index metric: [merges] -> did you mean [merge]?"),
    ] {
        let r = c.get(format!("http://127.0.0.1:{http}{path}")).send().await.unwrap();
        let st = r.status().as_u16();
        let v: Value = r.json().await.unwrap();
        assert_eq!((st, v["error"]["type"].as_str()), (400, Some("illegal_argument_exception")), "{path}: {v}");
        assert_eq!(v["error"]["reason"].as_str(), Some(want), "{path}: {v}");
    }
    let r = c.get(format!("http://127.0.0.1:{http}/_nodes/stats/indices/merge")).send().await.unwrap();
    assert!(r.status().is_success());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn running_merges_are_reported_at_shard_index_and_node_level() {
    let dir = evidence::retained_dir("merge_stats_current");
    let cfg = NodeConfig::from_settings(settings(dir.path())).expect("config");
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node: {e}"));
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let http = ports::http_port(&node);
    tokio::spawn(async move {
        let _ = elasticsearch_node::front::serve(node, listener).await;
    });
    let c = reqwest::Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
    let url = |p: &str| format!("http://127.0.0.1:{http}{p}");
    let get = |p: String| {
        let c = c.clone();
        async move {
            let r = c.get(p).send().await.unwrap();
            let st = r.status();
            let v: Value = r.json().await.unwrap();
            assert!(st.is_success(), "{v}");
            v
        }
    };
    let mut up = false;
    for _ in 0..120 {
        if c.get(url("/_cluster/health?wait_for_status=green&timeout=1s")).send().await.is_ok_and(|r| r.status().is_success()) {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(up, "node did not start");
    unknown_index_metrics_are_rejected(&c, http).await;

    let gate = Arc::new(Gate { held: AtomicU64::new(0), open: Mutex::new(false), cv: Condvar::new() });
    let g = gate.clone();
    *elasticsearch_engine::shard::MERGE_RUN_HOOK.lock() = Some(Arc::new(move || {
        g.held.fetch_add(1, Ordering::SeqCst);
        let mut open = g.open.lock().unwrap();
        while !*open {
            open = g.cv.wait(open).unwrap();
        }
    }));

    // Two shards (shard 1 lives on its own shard engine), merging every
    // two segments.
    let r = c.put(url("/m")).json(&json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "-1",
        "merge.policy.segments_per_tier": 2, "merge.policy.max_merge_at_once": 2, "merge.policy.floor_segment": "1kb"}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    let outcome = async {
        for round in 0..6 {
            let mut nd = String::new();
            for i in 0..40 {
                nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "m", "_id": format!("{round}-{i}")}}), json!({"n": i, "t": format!("text {round} {i}")})));
            }
            let r = c.post(url("/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
            assert!(r.status().is_success(), "{}", r.text().await.unwrap());
            // One committed segment per round and shard.
            let r = c.post(url("/m/_flush")).send().await.unwrap();
            assert!(r.status().is_success(), "{}", r.text().await.unwrap());
        }
        // Background merges, or else a force merge, reach the gate.
        let deadline = Instant::now() + Duration::from_secs(10);
        while gate.held.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if gate.held.load(Ordering::SeqCst) == 0 {
            let segs = get(url("/m/_stats/segments")).await;
            eprintln!("no background merge; segments {} — force merging", segs["indices"]["m"]["total"]["segments"]["count"]);
            let r = c.post(url("/m/_forcemerge?max_num_segments=1&wait_for_completion=false")).send().await.unwrap();
            assert!(r.status().is_success(), "{}", r.text().await.unwrap());
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        while gate.held.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let held = gate.held.load(Ordering::SeqCst);
        // Shard 0 merges in the node's main engine, shard 1 in its own
        // `engine-shard-1`: one merge held in each.
        assert!(held >= 2, "a background merge did not start on both shards (held {held})");
        let per_shard = get(url("/m/_stats/merge?level=shards")).await;
        for k in ["0", "1"] {
            let n = per_shard["indices"]["m"]["shards"][k][0]["merges"]["current"].as_u64().unwrap_or(0);
            assert!(n >= 1, "shard {k} merges.current {n}: {per_shard}");
        }
        let shards = sum_shards(&per_shard);
        let all_shards = sum_shards(&get(url("/_stats/merge?level=shards")).await);
        let all_index = get(url("/_stats/merge")).await["_all"]["total"]["merges"]["current"].as_u64().unwrap_or(0);
        let index = get(url("/m/_stats")).await;
        let node_counters = node_current(&get(url("/_nodes/stats/indices/merge")).await);
        let node_full = node_current(&get(url("/_nodes/stats")).await);
        (held, shards.min(all_shards), index["indices"]["m"]["total"]["merges"]["current"].as_u64().unwrap_or(0), index["_all"]["total"]["merges"]["current"].as_u64().unwrap_or(0).min(all_index), node_counters, node_full)
    };
    // The gate opens whatever the checks found (a failed assert unwinds
    // through here only after the stats were read).
    let result = tokio::time::timeout(Duration::from_secs(180), outcome).await;
    gate.release();
    *elasticsearch_engine::shard::MERGE_RUN_HOOK.lock() = None;
    let (held, shards, index, all, node_counters, node_full) = result.expect("stats timed out");
    assert!(shards >= held, "shard-level merges.current {shards} < {held} running");
    assert!(index >= held, "index _stats merges.current {index} < {held} running");
    assert!(all >= held, "_all merges.current {all} < {held} running");
    assert!(node_counters >= held, "_nodes/stats/indices/merge merges.current {node_counters} < {held} running");
    assert!(node_full >= held, "_nodes/stats merges.current {node_full} < {held} running");
    // Released: the merges finish and every level drops back to 0.
    let settle = || async {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let n = node_current(&get(url("/_nodes/stats/indices/merge")).await);
            if n == 0 {
                break;
            }
            assert!(Instant::now() < deadline, "merges.current stuck at {n} after the merges were released");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };
    settle().await;

    // A force merge (Rally's force-merge operation) of a single-shard
    // index whose policy wants no background merge.
    let gate = Arc::new(Gate { held: AtomicU64::new(0), open: Mutex::new(false), cv: Condvar::new() });
    let g = gate.clone();
    *elasticsearch_engine::shard::MERGE_RUN_HOOK.lock() = Some(Arc::new(move || {
        g.held.fetch_add(1, Ordering::SeqCst);
        let mut open = g.open.lock().unwrap();
        while !*open {
            open = g.cv.wait(open).unwrap();
        }
    }));
    let r = c.put(url("/f")).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "merge.policy.segments_per_tier": 50}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    let outcome = async {
        for round in 0..4 {
            let nd: String = (0..20).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "f", "_id": format!("{round}-{i}")}}), json!({"n": i}))).collect();
            let r = c.post(url("/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
            assert!(r.status().is_success(), "{}", r.text().await.unwrap());
            let r = c.post(url("/f/_flush")).send().await.unwrap();
            assert!(r.status().is_success(), "{}", r.text().await.unwrap());
        }
        let fm = c.post(url("/f/_forcemerge?max_num_segments=1")).send();
        let fm = tokio::spawn(fm);
        let deadline = Instant::now() + Duration::from_secs(60);
        while gate.held.load(Ordering::SeqCst) < 1 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let held = gate.held.load(Ordering::SeqCst);
        assert!(held >= 1, "the force merge did not start");
        let shards = get(url("/_stats/merge?level=shards")).await["indices"]["f"]["shards"]["0"][0]["merges"]["current"].as_u64().unwrap_or(0);
        let index = get(url("/f/_stats")).await["indices"]["f"]["total"]["merges"]["current"].as_u64().unwrap_or(0);
        let all = get(url("/_stats/merge")).await["_all"]["total"]["merges"]["current"].as_u64().unwrap_or(0);
        let node_counters = node_current(&get(url("/_nodes/stats/indices/merge")).await);
        let node_full = node_current(&get(url("/_nodes/stats")).await);
        (held, shards, index, all, node_counters, node_full, fm)
    };
    let result = tokio::time::timeout(Duration::from_secs(180), outcome).await;
    gate.release();
    *elasticsearch_engine::shard::MERGE_RUN_HOOK.lock() = None;
    let (_held, shards, index, all, node_counters, node_full, fm) = result.expect("stats timed out");
    for (what, n) in [("shard", shards), ("index", index), ("_all", all), ("node (merge metric)", node_counters), ("node", node_full)] {
        assert!(n >= 1, "{what} merges.current {n} while a force merge runs");
    }
    let r = fm.await.unwrap().unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    settle().await;
}
