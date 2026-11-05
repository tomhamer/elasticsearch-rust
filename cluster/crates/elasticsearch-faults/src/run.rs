//! One scenario run: fresh cluster, setup, workload + nemesis + monitor,
//! heal, convergence, final per-copy reads, checks, artifacts.

use crate::api::{call, Api};
use crate::checker::{self, Copy, Doc, Input, Verdict};
use crate::cluster::Cluster;
use crate::history::{blank, History, Kind, Op, Outcome};
use crate::monitor::Monitor;
use crate::nemesis::{Nemesis, Params};
use crate::net::Net;
use crate::workload::{register_key, Workload};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Config {
    pub nodes: usize,
    pub port_base: u16,
    pub bin: PathBuf,
    pub out: PathBuf,
    pub short: bool,
    pub dur: Duration,
    pub workers: usize,
    pub op_timeout: Duration,
    pub netem: bool,
    pub registers: usize,
    pub keep_data: bool,
    /// The node honours `cluster.fault_detection.*` (see FAULT_RESULTS.md
    /// bug #1: it currently does not, so this defaults to false).
    pub fast_fd: bool,
}

impl Config {
    /// `-E` settings for every node. Short mode shortens fault detection so
    /// a fault is detected within ~10s instead of ~30-40s (ES defaults);
    /// every value is a documented, user-settable ES setting.
    pub fn node_settings(&self) -> Vec<String> {
        if self.short && self.fast_fd {
            vec![
                "cluster.fault_detection.follower_check.interval=1s".into(),
                "cluster.fault_detection.follower_check.timeout=2s".into(),
                "cluster.fault_detection.follower_check.retry_count=3".into(),
                "cluster.fault_detection.leader_check.interval=1s".into(),
                "cluster.fault_detection.leader_check.timeout=2s".into(),
                "cluster.fault_detection.leader_check.retry_count=3".into(),
                "cluster.publish.timeout=10s".into(),
                "cluster.follower_lag.timeout=20s".into(),
            ]
        } else {
            vec![]
        }
    }
    /// Seconds after a partition starts by which the minority must have
    /// noticed (leader checks: retry_count × (interval + timeout) + slack).
    pub fn grace(&self) -> f64 {
        self.detect() + 5.0
    }
    /// Expected failure-detection time: retry_count x (interval + timeout).
    pub fn detect(&self) -> f64 {
        if self.short && self.fast_fd { 9.0 } else { 33.0 }
    }
    pub fn delayed_timeout(&self) -> &'static str {
        if self.short { "5s" } else { "1m" }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RunResult {
    pub scenario: String,
    pub seed: u64,
    pub nodes: usize,
    pub dir: String,
    pub wall_secs: f64,
    pub converge_secs: Option<f64>,
    pub verdict: Verdict,
    pub setup_error: Option<String>,
}

impl RunResult {
    pub fn ok(&self) -> bool {
        self.setup_error.is_none() && self.verdict.violations.is_empty()
    }
}

fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) {
    if let Ok(f) = std::fs::File::create(path) {
        let mut w = std::io::BufWriter::new(f);
        for it in items {
            let _ = writeln!(w, "{}", serde_json::to_string(it).unwrap_or_default());
        }
    }
}

async fn setup(api: &Api, cfg: &Config, hist: &History, indices: &[String], preload: usize) -> Result<(), String> {
    let n = cfg.nodes;
    api.wait_converged(n, Duration::from_secs(90)).await.map_err(|e| format!("cluster did not form: {e}"))?;
    api.cluster.mark_formed();
    let replicas = (n - 1).min(2);
    for ix in indices {
        let body = json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": replicas, "index.unassigned.node_left.delayed_timeout": cfg.delayed_timeout()},
            "mappings": {"properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}}
        });
        let (s, v) = api.send(0, "PUT", &format!("/{ix}"), Some(body.to_string()), Duration::from_secs(60)).await.map_err(|e| format!("create {ix}: {}", e.msg()))?;
        if s != 200 {
            return Err(format!("create {ix}: {s} {v}"));
        }
    }
    api.wait_converged(n, Duration::from_secs(90)).await.map_err(|e| format!("indices not green: {e}"))?;
    // Registers start at value 0, acknowledged.
    for ix in indices {
        for k in 0..cfg.registers {
            let key = register_key(k);
            let mut op = blank(usize::MAX, 0, Kind::Init, ix, &key);
            op.value = Some(0);
            let id = hist.invoke(op);
            let (s, v) = api.send(0, "PUT", &format!("/{ix}/_doc/{key}"), Some(json!({"k": key, "v": 0}).to_string()), Duration::from_secs(30)).await.map_err(|e| e.msg())?;
            if s / 100 != 2 {
                return Err(format!("init {ix}/{key}: {s} {v}"));
            }
            hist.complete(id, |o| {
                o.outcome = Outcome::Ok;
                o.status = Some(s);
                o.seq_no = v["_seq_no"].as_i64();
                o.primary_term = v["_primary_term"].as_i64();
            });
        }
    }
    // Bulk preload (so recoveries take long enough to be interrupted).
    let mut done = 0;
    while done < preload {
        let batch = 2000.min(preload - done);
        let mut body = String::new();
        let mut ids = Vec::new();
        for j in done..done + batch {
            let ix = &indices[j % indices.len()];
            let key = format!("p{j}");
            let mut op = blank(usize::MAX, 0, Kind::Init, ix, &key);
            op.value = Some(j as i64);
            ids.push(hist.invoke(op));
            body.push_str(&json!({"create": {"_index": ix, "_id": key}}).to_string());
            body.push('\n');
            body.push_str(&json!({"k": key, "v": j, "pad": "x".repeat(200)}).to_string());
            body.push('\n');
        }
        let (s, v) = api.send(0, "POST", "/_bulk", Some(body), Duration::from_secs(120)).await.map_err(|e| e.msg())?;
        if s != 200 || v["errors"] == true {
            return Err(format!("preload bulk: {s} {}", v.to_string().chars().take(500).collect::<String>()));
        }
        for (pos, id) in ids.iter().enumerate() {
            let item = &v["items"][pos]["create"];
            hist.complete(*id, |o| {
                o.outcome = Outcome::Ok;
                o.status = Some(201);
                o.seq_no = item["_seq_no"].as_i64();
                o.primary_term = item["_primary_term"].as_i64();
            });
        }
        done += batch;
    }
    Ok(())
}

/// Result of the final per-copy read: the accepted pass, how many passes it
/// took, and whether two consecutive passes agreed.
pub struct FinalRead {
    pub copies: Vec<Copy>,
    pub passes: usize,
    pub stable: bool,
}

/// Repeat full read passes until two consecutive passes are identical.
///
/// One pass reads the copies one after another, so it is not a snapshot.
/// A write that lands mid-pass shows up in some copies and not in others,
/// or in a copy's `_count` but not in its `_mget`, and looks like G5. ES 8.19
/// allows exactly that after the clients stop: an unacked bulk shard request
/// keeps being retried by its coordinator for up to the bulk `timeout`
/// (default 1m), and client disconnect does not cancel it. Two identical
/// back-to-back passes mean no write landed while either was running, so the
/// accepted pass is a consistent cut. Copies that still diverge in a stable
/// pass are real G5. Gives up after `max_passes` and returns the last pass.
pub async fn read_stable<F, Fut>(mut pass: F, max_passes: usize, pause: Duration) -> FinalRead
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Vec<Copy>>,
{
    let mut prev: Option<Vec<Copy>> = None;
    let mut n = 0;
    loop {
        n += 1;
        let cur = pass().await;
        if prev.as_ref() == Some(&cur) {
            return FinalRead { copies: cur, passes: n, stable: true };
        }
        if n >= max_passes.max(1) {
            return FinalRead { copies: cur, passes: n, stable: false };
        }
        prev = Some(cur);
        tokio::time::sleep(pause).await;
    }
}

/// Read every copy of every index directly from the node holding it
/// (reads are served by the receiving node when it holds an active copy).
/// One pass; `read_stable` repeats it until it is quiescent.
async fn read_copies(api: &Api, ops: &[Op], indices: &[String]) -> Vec<Copy> {
    let mut copies = Vec::new();
    let Some(rt) = api.routing().await else { return copies };
    let mut keys: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for o in ops {
        keys.entry(o.index.as_str()).or_default().insert(o.key.as_str());
    }
    let http = crate::api::client(Duration::from_secs(60));
    for ix in indices {
        let Some((p, reps)) = rt.started.get(ix).cloned() else { continue };
        if let Some(any) = p.or(reps.first().copied()) {
            let _ = api.send(any, "POST", &format!("/{ix}/_refresh"), None, Duration::from_secs(60)).await;
        }
        let holders: Vec<(usize, bool)> = p.into_iter().map(|x| (x, true)).chain(reps.into_iter().map(|x| (x, false))).collect();
        let ids: Vec<&str> = keys.get(ix.as_str()).map(|s| s.iter().copied().collect()).unwrap_or_default();
        for (node, primary) in holders {
            let mut cp = Copy { node, index: ix.clone(), primary, docs: BTreeMap::new(), count: None, error: None };
            for chunk in ids.chunks(1000) {
                let url = format!("{}/{ix}/_mget", api.cluster.url(node));
                match call(&http, "POST", &url, Some(json!({"ids": chunk}).to_string()), None).await {
                    Ok((200, v)) => {
                        for d in v["docs"].as_array().cloned().unwrap_or_default() {
                            if d["found"] == true {
                                let id = d["_id"].as_str().unwrap_or("").to_string();
                                cp.docs.insert(id, Doc { seq_no: d["_seq_no"].as_i64().unwrap_or(-1), version: d["_version"].as_i64().unwrap_or(-1), value: d["_source"]["v"].as_i64() });
                            } else if d.get("error").is_some() {
                                cp.error = Some(format!("mget item error: {}", d["error"]));
                            }
                        }
                    }
                    Ok((s, v)) => cp.error = Some(format!("mget {s}: {}", v.to_string().chars().take(300).collect::<String>())),
                    Err(e) => cp.error = Some(e.msg()),
                }
            }
            match call(&http, "GET", &format!("{}/{ix}/_count", api.cluster.url(node)), None, None).await {
                Ok((200, v)) => cp.count = v["count"].as_u64(),
                Ok((s, v)) => cp.error = Some(format!("count {s}: {v}")),
                Err(e) => cp.error = Some(e.msg()),
            }
            copies.push(cp);
        }
    }
    copies
}

pub async fn run_one(cfg: &Config, scenario: &str, seed: u64) -> RunResult {
    let wall = Instant::now();
    let dir = crate::artifacts::unique_dir(&cfg.out, &format!("{scenario}-n{}-s{seed}", cfg.nodes)).expect("unique scenario directory");
    let mut res = RunResult { scenario: scenario.into(), seed, nodes: cfg.nodes, dir: dir.display().to_string(), wall_secs: 0.0, converge_secs: None, verdict: Verdict::default(), setup_error: None };
    println!("=== {scenario} seed={seed} nodes={} -> {}", cfg.nodes, dir.display());

    let cluster = Arc::new(Cluster::new(cfg.nodes, cfg.port_base, cfg.bin.clone(), dir.clone(), cfg.node_settings()));
    if let Err(e) = cluster.setup_cgroups() {
        res.setup_error = Some(e);
        return res;
    }
    let net = match Net::setup(cfg.port_base, cfg.nodes) {
        Ok(n) => Arc::new(n),
        Err(e) => {
            res.setup_error = Some(e);
            return res;
        }
    };
    for i in 0..cfg.nodes {
        if let Err(e) = cluster.start(i) {
            res.setup_error = Some(e);
            return res;
        }
    }
    let t0 = Instant::now();
    let hist = Arc::new(History::new(t0));
    let api = Api::new(cluster.clone());
    let indices = vec!["jepsen-a".to_string(), "jepsen-b".to_string()];
    let preload = if scenario == "kill-during-recovery" { if cfg.short { 20_000 } else { 100_000 } } else { 0 };
    if let Err(e) = setup(&api, cfg, &hist, &indices, preload).await {
        println!("  SETUP FAILED: {e}");
        res.setup_error = Some(e);
        cluster.kill_all();
        return res;
    }

    let stop_workers = Arc::new(AtomicBool::new(false));
    let stop_monitor = Arc::new(AtomicBool::new(false));
    let monitor = Arc::new(Monitor::new());
    let mon_task = tokio::spawn(monitor.clone().run(cluster.clone(), t0, stop_monitor.clone()));
    let crashes: Arc<Mutex<Vec<(usize, f64, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let reaper = {
        let (cl, cr, stop) = (cluster.clone(), crashes.clone(), stop_monitor.clone());
        tokio::spawn(async move {
            while !stop.load(Ordering::SeqCst) {
                for (i, st) in cl.reap_crashed() {
                    println!("    [{:7.2}s] !!! n{} exited on its own: {st}", t0.elapsed().as_secs_f64(), i + 1);
                    cr.lock().push((i, t0.elapsed().as_secs_f64(), st));
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
    };
    let wl = Arc::new(Workload { indices: indices.clone(), registers: cfg.registers, hist: hist.clone(), cluster: cluster.clone(), op_timeout: cfg.op_timeout, stop: stop_workers.clone(), bulk_seq: AtomicU64::new(0) });
    let workers: Vec<_> = (0..cfg.workers).map(|w| tokio::spawn(wl.clone().run_worker(w, seed))).collect();

    let nem = Nemesis {
        cluster: cluster.clone(),
        net: net.clone(),
        api: Api::new(cluster.clone()),
        t0,
        indices: indices.clone(),
        events: Mutex::new(Vec::new()),
        windows: Mutex::new(Vec::new()),
        params: Params { dur: cfg.dur, short: cfg.short, netem: cfg.netem, detect: cfg.detect() },
    };
    // A few seconds of fault-free load first.
    tokio::time::sleep(Duration::from_secs(3)).await;
    nem.log("nemesis-start", vec![], vec![], scenario.into());
    let mut rng = StdRng::seed_from_u64(seed);
    nem.run(scenario, &mut rng).await;
    nem.heal_all();
    nem.log("nemesis-end", vec![], vec![], "all faults healed".into());
    // Keep the load running while the cluster recovers.
    tokio::time::sleep(Duration::from_secs(5)).await;
    stop_workers.store(true, Ordering::SeqCst);
    for w in workers {
        let _ = tokio::time::timeout(cfg.op_timeout * 3 + Duration::from_secs(5), w).await;
    }
    nem.log("clients-stopped", vec![], vec![], format!("{} ops", hist.len()));
    let converged = api.wait_converged(cfg.nodes, Duration::from_secs(if cfg.short { 120 } else { 300 })).await;
    match &converged {
        Ok(s) => nem.log("converged", vec![], vec![], format!("green + single master after {s:.1}s")),
        Err(e) => {
            nem.log("no-convergence", vec![], vec![], e.clone());
            let _ = std::fs::write(dir.join("DIAG-no-convergence.txt"), diagnostics(&api).await);
        }
    }
    res.converge_secs = converged.as_ref().ok().copied();
    let ops = hist.snapshot();
    let fin = read_stable(|| read_copies(&api, &ops, &indices), 30, Duration::from_secs(2)).await;
    nem.log("final-read", vec![], vec![], format!("{} pass(es), stable={}", fin.passes, fin.stable));
    if !fin.stable {
        println!("  final copy read did not stabilise after {} passes", fin.passes);
    }
    let copies = fin.copies;
    stop_monitor.store(true, Ordering::SeqCst);
    let _ = mon_task.await;
    let _ = reaper.await;

    let obs = monitor.obs.lock().clone();
    let samples = monitor.samples.lock().clone();
    let events = nem.events.lock().clone();
    let windows = nem.windows.lock().clone();
    let crashes = crashes.lock().clone();
    let verdict = checker::check(&Input { ops: &ops, copies: &copies, obs: &obs, samples: &samples, events: &events, windows: &windows, grace: cfg.grace(), converged: converged.clone(), crashes: &crashes });

    write_jsonl(&dir.join("history.jsonl"), &ops);
    write_jsonl(&dir.join("nemesis.jsonl"), &events);
    write_jsonl(&dir.join("windows.jsonl"), &windows);
    write_jsonl(&dir.join("monitor.jsonl"), &obs);
    let _ = std::fs::write(dir.join("final-copies.json"), serde_json::to_string(&copies).unwrap_or_default());
    let _ = std::fs::write(dir.join("verdict.json"), serde_json::to_string_pretty(&verdict).unwrap_or_default());
    if !verdict.violations.is_empty() {
        let _ = std::fs::write(dir.join("REPRO.md"), repro(cfg, scenario, seed, &verdict, &events));
    }

    cluster.kill_all();
    drop(nem);
    drop(net);
    if !cfg.keep_data && verdict.violations.is_empty() {
        for i in 0..cfg.nodes {
            let _ = std::fs::remove_dir_all(cluster.data_dir(i));
        }
    }
    res.verdict = verdict;
    res.wall_secs = wall.elapsed().as_secs_f64();
    print_result(&res);
    res
}

/// Allocation state from the master's point of view, for liveness failures.
async fn diagnostics(api: &Api) -> String {
    let m = api.master().await.or(api.cluster.running().first().copied()).unwrap_or(0);
    let mut out = format!("asked node-{}\n", m + 1);
    for path in ["/_cat/shards?v&h=index,shard,prirep,state,node,unassigned.reason,unassigned.details", "/_cluster/allocation/explain?pretty", "/_cluster/health?level=shards&pretty", "/_cat/recovery?v&active_only=false", "/_cluster/state/routing_table,metadata?pretty&filter_path=routing_table,metadata.indices.*.in_sync_allocations,metadata.indices.*.primary_terms"] {
        out.push_str(&format!("\n===== GET {path}\n"));
        match api.get(m, path, Duration::from_secs(10)).await {
            Ok((s, v)) => out.push_str(&format!("{s}\n{}\n", match v { Value::String(t) => t, other => serde_json::to_string_pretty(&other).unwrap_or_default() })),
            Err(e) => out.push_str(&e.msg()),
        }
    }
    out
}

pub fn repro(cfg: &Config, scenario: &str, seed: u64, v: &Verdict, events: &[crate::nemesis::Event]) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Reproduction: {scenario} seed {seed}\n\n"));
    s.push_str("```\n");
    s.push_str(&format!(
        "cluster/scripts/faults/run.sh --scenario {scenario} --seed {seed} --nodes {} --mode {} --duration {} --workers {} --op-timeout {}{}{}\n",
        cfg.nodes,
        if cfg.short { "short" } else { "long" },
        cfg.dur.as_secs(),
        cfg.workers,
        cfg.op_timeout.as_secs(),
        if cfg.netem { " --netem" } else { "" },
        if cfg.fast_fd { " --fast-fd" } else { "" }
    ));
    s.push_str("```\n\nThe seed fixes the nemesis schedule's shape; interleavings vary run to run. Node settings:\n\n```\n");
    for e in cfg.node_settings() {
        s.push_str(&format!("-E {e}\n"));
    }
    s.push_str("```\n\n## Nemesis schedule (as executed)\n\n```\n");
    for e in events {
        s.push_str(&format!("{:8.2}s {:<16} [{}] {} {}\n", e.t, e.kind, e.nodes.iter().map(|n| format!("n{}", n + 1)).collect::<Vec<_>>().join(","), e.roles.join(","), e.detail));
    }
    s.push_str("```\n\n## Violations\n\n");
    for x in &v.violations {
        s.push_str(&format!("- **{} {}**{}: {}\n", x.guarantee, x.kind, x.t.map(|t| format!(" @{t:.2}s")).unwrap_or_default(), x.detail.replace('\n', "\n  ")));
    }
    s
}

pub fn print_result(r: &RunResult) {
    let v = &r.verdict;
    let c = &v.counts;
    let status = if r.setup_error.is_some() { "ERROR" } else if v.violations.is_empty() { "PASS" } else { "FAIL" };
    println!(
        "  {status} {} seed={} n={} | ops {} ok {} fail {} info {} | acked writes {} lost {} | minority writes {} (blocked {}) | elections {} failover {} | converge {} | {:.0}s",
        r.scenario,
        r.seed,
        r.nodes,
        c.attempted,
        c.ok,
        c.fail,
        c.info,
        c.writes_acked,
        c.lost,
        c.minority_writes,
        c.minority_writes_rejected_block,
        v.elections.len(),
        fmt_lat(&v.failover_latencies),
        r.converge_secs.map(|s| format!("{s:.1}s")).unwrap_or("NO".into()),
        r.wall_secs
    );
    if let Some(e) = &r.setup_error {
        println!("    setup error: {e}");
    }
    for x in v.violations.iter().take(15) {
        println!("    VIOLATION {} {}: {}", x.guarantee, x.kind, x.detail.lines().next().unwrap_or(""));
    }
    if v.violations.len() > 15 {
        println!("    ... {} more (see {}/REPRO.md)", v.violations.len() - 15, r.dir);
    }
    for x in v.warnings.iter().take(5) {
        println!("    warning {} {}: {}", x.guarantee, x.kind, x.detail.chars().take(300).collect::<String>());
    }
}

pub fn fmt_lat(l: &[f64]) -> String {
    if l.is_empty() {
        return "-".into();
    }
    let mut s = l.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    format!("{:.1}/{:.1}/{:.1}s", s[0], s[s.len() / 2], s[s.len() - 1])
}

pub fn summary_md(results: &[RunResult], cfg: &Config) -> String {
    let mut s = String::new();
    s.push_str(&format!("mode={} nodes={} duration={}s workers={} op_timeout={}s\n\n", if cfg.short { "short" } else { "long" }, cfg.nodes, cfg.dur.as_secs(), cfg.workers, cfg.op_timeout.as_secs()));
    s.push_str("| scenario | seed | verdict | ops | ok | fail | info | acked writes | lost | info present/absent | minority writes (blocked) | block latency | stale reads in fault | elections | failover min/med/max | converge | G1 | G2 | G3 | G5 | G7 | G8 |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in results {
        let v = &r.verdict;
        let c = &v.counts;
        let g = |k: &str| v.guarantees.get(k).cloned().unwrap_or("-".into());
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} ({}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.scenario,
            r.seed,
            if r.setup_error.is_some() { "ERROR" } else if v.violations.is_empty() { "PASS" } else { "**FAIL**" },
            c.attempted,
            c.ok,
            c.fail,
            c.info,
            c.writes_acked,
            c.lost,
            c.info_writes_present,
            c.info_writes_absent,
            c.minority_writes,
            c.minority_writes_rejected_block,
            fmt_lat(&c.minority_block_latency),
            c.stale_reads_in_fault,
            v.elections.len(),
            fmt_lat(&v.failover_latencies),
            r.converge_secs.map(|s| format!("{s:.1}s")).unwrap_or("NO".into()),
            g("G1"),
            g("G2"),
            g("G3"),
            g("G5"),
            g("G7"),
            g("G8"),
        ));
    }
    s.push('\n');
    for r in results.iter().filter(|r| !r.ok()) {
        s.push_str(&format!("### {} seed {} ({})\n\n", r.scenario, r.seed, r.dir));
        if let Some(e) = &r.setup_error {
            s.push_str(&format!("setup error: {e}\n\n"));
        }
        for x in r.verdict.violations.iter().take(10) {
            s.push_str(&format!("- {} {}: {}\n", x.guarantee, x.kind, x.detail.lines().next().unwrap_or("")));
        }
        s.push('\n');
    }
    let _ = Value::Null;
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    // Replays asymmetric-n5-s1 (§10.8): jepsen-a, primary n1 + replicas n3/n4.
    // Acked writes a0..a2; the unacked (info) bulk 766 items b0..b4 are
    // retried by their coordinator and executed on n1 after the clients
    // stopped, while the final read is running: n1's _mget ran before the
    // write and its _count after it; n3/n4 were read after it.
    fn ops() -> Vec<Op> {
        let mut v = Vec::new();
        for (i, k) in ["a0", "a1", "a2"].iter().enumerate() {
            let mut o = blank(0, 0, Kind::Create, "jepsen-a", k);
            o.id = i as u64;
            o.value = Some(i as i64);
            o.t_invoke = 1.0;
            o.t_complete = 2.0;
            o.outcome = Outcome::Ok;
            o.seq_no = Some(i as i64);
            o.primary_term = Some(1);
            v.push(o);
        }
        for i in 0..5 {
            let mut o = blank(13, 1, Kind::BulkCreate, "jepsen-a", &format!("b{i}"));
            o.id = 10 + i as u64;
            o.value = Some(100 + i as i64);
            o.t_invoke = 17.2;
            o.t_complete = 27.2;
            o.outcome = Outcome::Info;
            v.push(o);
        }
        v
    }
    fn copy(node: usize, primary: bool, late: bool, count: u64) -> Copy {
        let mut docs = BTreeMap::new();
        for i in 0..3 {
            docs.insert(format!("a{i}"), Doc { seq_no: i, version: 1, value: Some(i) });
        }
        if late {
            for i in 0..5 {
                docs.insert(format!("b{i}"), Doc { seq_no: 3 + i, version: 1, value: Some(100 + i) });
            }
        }
        Copy { node, index: "jepsen-a".into(), primary, docs, count: Some(count), error: None }
    }
    fn torn() -> Vec<Copy> {
        vec![copy(0, true, false, 8), copy(3, false, true, 8), copy(2, false, true, 8)]
    }
    fn settled() -> Vec<Copy> {
        vec![copy(0, true, true, 8), copy(3, false, true, 8), copy(2, false, true, 8)]
    }
    fn g5(copies: &[Copy]) -> Vec<String> {
        let ops = ops();
        let samples = BTreeMap::new();
        let v = checker::check(&Input { ops: &ops, copies, obs: &[], samples: &samples, events: &[], windows: &[], grace: 5.0, converged: Ok(0.0), crashes: &[] });
        v.violations.iter().filter(|x| x.guarantee == "G5").map(|x| x.kind.clone()).collect()
    }
    async fn replay(passes: Vec<Vec<Copy>>, max: usize) -> FinalRead {
        let q = Mutex::new(passes.into_iter().collect::<VecDeque<_>>());
        read_stable(
            || {
                let mut q = q.lock();
                let next = if q.len() > 1 { q.pop_front().unwrap() } else { q.front().cloned().unwrap() };
                async move { next }
            },
            max,
            Duration::ZERO,
        )
        .await
    }

    #[tokio::test]
    async fn straggler_write_during_final_read_is_not_g5() {
        // Old behaviour (one pass, no stabilisation): the torn pass is G5.
        let old = replay(vec![torn(), settled()], 1).await;
        assert!(!old.stable);
        let k = g5(&old.copies);
        assert!(k.contains(&"copies-diverge".to_string()) && k.contains(&"unknown-docs".to_string()), "{k:?}");
        // Fixed: the torn pass differs from the next one, so it is re-read
        // until two passes agree; the accepted cut is consistent.
        let new = replay(vec![torn(), settled()], 30).await;
        assert!(new.stable);
        assert_eq!(new.passes, 3);
        assert!(g5(&new.copies).is_empty(), "{:?}", g5(&new.copies));
    }

    #[tokio::test]
    async fn persistent_divergence_is_still_g5() {
        let r = replay(vec![torn()], 30).await;
        assert!(r.stable);
        assert_eq!(r.passes, 2);
        assert!(g5(&r.copies).contains(&"copies-diverge".to_string()));
    }

    #[tokio::test]
    async fn never_quiescent_returns_last_pass_unstable() {
        let r = replay(vec![torn(), settled(), torn(), settled(), torn()], 4).await;
        assert!(!r.stable);
        assert_eq!(r.passes, 4);
    }
}
