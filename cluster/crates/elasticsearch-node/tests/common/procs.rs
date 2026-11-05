//! An N-node cluster of real `elasticsearch-node` processes, for tests that
//! kill and restart nodes (an in-process node cannot be stopped: its tasks
//! live on the test's runtime). Node settings as [`super::start`]'s; each
//! node's data directory ([`super::scratch_dir`]) survives a restart.
//!
//! Ports: a node binds port 0 and reports what it took (`node.portsfile`);
//! the nodes started before it are its seed hosts. A restarted node asks
//! for the ports it had (as a fixed-port cluster would), and for new ones
//! if those were taken meanwhile. A binary that cannot report its ports
//! (an old release in a rolling-upgrade test) gets probed free ports, and is
//! started again on fresh ones if it fails with `Address already in use`.

use super::ports;
use reqwest::Method;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The nodes' bound (http, transport) ports, one pair per node (0: not
/// known yet).
#[derive(Clone, Debug)]
pub struct ProcPorts {
    pub http: Vec<u16>,
    pub transport: Vec<u16>,
}

/// A started process cluster; every node is killed (SIGKILL) when dropped.
pub struct Procs {
    pub c: reqwest::Client,
    pub p: ProcPorts,
    pub dir: PathBuf,
    /// `cluster.name` of every node (unique per cluster).
    pub cluster: String,
    extra: Vec<(String, String)>,
    children: Vec<Option<Child>>,
    binaries: Vec<PathBuf>,
    /// This build's `elasticsearch-node`: the one binary known to write
    /// `node.portsfile`.
    current: PathBuf,
}

/// How a spawned node's startup ended.
enum Launch {
    Bound { http: u16, transport: u16 },
    /// It exited because a port it asked for was taken: try other ports.
    AddrInUse,
    /// It exited on its own for another reason (left for [`Procs::wait_exit`]).
    Exited,
    /// Neither bound nor exited in time.
    TimedOut,
}

/// Attempts at binding before a spawn gives up.
const BIND_ATTEMPTS: usize = 8;

impl Drop for Procs {
    fn drop(&mut self) {
        for i in 0..self.children.len() {
            self.kill(i);
        }
    }
}

impl Procs {
    pub fn len(&self) -> usize {
        self.p.http.len()
    }

    /// The positions of the nodes running now.
    pub fn live(&self) -> Vec<usize> {
        (0..self.len()).filter(|i| self.children[*i].is_some()).collect()
    }

    /// Start node `i` (again), with `node_extra` on top of the cluster's
    /// extra settings; returns once it has bound its ports (or exited).
    pub async fn spawn(&mut self, i: usize, node_extra: &[(&str, &str)]) {
        self.spawn_with_env(i, node_extra, &[]).await;
    }

    /// Per-child environment overrides, never process-global test mutations.
    pub async fn spawn_with_env(&mut self, i: usize, node_extra: &[(&str, &str)], env: &[(&str, &str)]) {
        assert!(self.children[i].is_none(), "n{i} is running");
        let reports = self.binaries[i] == self.current;
        // A restart first asks for the ports the node had.
        let mut sticky = self.p.http[i] != 0 && self.p.transport[i] != 0;
        for _ in 0..BIND_ATTEMPTS {
            let (http, transport) = if sticky {
                (self.p.http[i], self.p.transport[i])
            } else if reports {
                (0, 0)
            } else {
                let p = ports::probe_free_ports(2);
                (p[0], p[1])
            };
            match self.launch(i, http, transport, reports, node_extra, env).await {
                Launch::Bound { http, transport } => {
                    self.p.http[i] = http;
                    self.p.transport[i] = transport;
                    return;
                }
                Launch::AddrInUse => sticky = false,
                Launch::Exited => return,
                Launch::TimedOut if http != 0 => {
                    self.p.http[i] = http;
                    self.p.transport[i] = transport;
                    return;
                }
                Launch::TimedOut => panic!("n{i} did not report its ports:\n{}", self.log_tail(i)),
            }
        }
        panic!("n{i} found no free ports in {BIND_ATTEMPTS} attempts:\n{}", self.log_tail(i));
    }

    /// Start node `i`'s process on (`http`, `transport`) (0: OS-assigned)
    /// and wait until it has bound them or exited.
    async fn launch(&mut self, i: usize, http: u16, transport: u16, reports: bool, node_extra: &[(&str, &str)], env: &[(&str, &str)]) -> Launch {
        let logs = self.dir.join(format!("logs{i}"));
        ports::clear_ports_files(&logs);
        let offset = std::fs::metadata(self.log_path(i)).map(|m| m.len()).unwrap_or(0);
        let out = std::fs::OpenOptions::new().create(true).append(true).open(self.log_path(i)).unwrap();
        let mut args: Vec<String> = Vec::new();
        let mut e = |k: &str, v: &str| {
            args.push("-E".into());
            args.push(format!("{k}={v}"));
        };
        e("cluster.name", &self.cluster);
        e("node.name", &format!("n{i}"));
        e("path.data", &self.dir.join(format!("n{i}")).to_string_lossy());
        e("path.logs", &logs.to_string_lossy());
        e("network.host", "127.0.0.1");
        e("http.port", &http.to_string());
        e("transport.port", &transport.to_string());
        if reports {
            e("node.portsfile", "true");
        }
        // Every other node whose transport port is known (those started
        // before it); a node learns later ones when they reach it.
        let seeds = ports::seeds(&(0..self.len()).filter(|j| *j != i).map(|j| self.p.transport[j]).collect::<Vec<_>>());
        if !seeds.is_empty() {
            e("discovery.seed_hosts", &seeds.join(","));
        }
        e("cluster.initial_master_nodes", &(0..self.len()).map(|i| format!("n{i}")).collect::<Vec<_>>().join(","));
        e("xpack.security.enabled", "false");
        for w in ["low", "high", "flood_stage"] {
            e(&format!("cluster.routing.allocation.disk.watermark.{w}"), "1b");
        }
        for (k, v) in self.extra.iter().map(|(k, v)| (k.as_str(), v.as_str())).chain(node_extra.iter().copied()) {
            e(k, v);
        }
        let child = Command::new(&self.binaries[i]).args(&args).envs(env.iter().copied()).stdin(Stdio::null()).stdout(out.try_clone().unwrap()).stderr(out).spawn().expect("spawn elasticsearch-node");
        self.children[i] = Some(child);

        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let exited = self.children[i].as_mut().expect("just spawned").try_wait().expect("poll child").is_some();
            if exited {
                let text = std::fs::read(self.log_path(i)).unwrap_or_default();
                let since = String::from_utf8_lossy(&text[(offset as usize).min(text.len())..]).into_owned();
                if ports::is_addr_in_use(&since) {
                    self.children[i] = None;
                    return Launch::AddrInUse;
                }
                return Launch::Exited;
            }
            if reports {
                // `http.ports` is written after `transport.ports`.
                if let Some(h) = ports::read_ports_file(&logs, "http") {
                    if let Some(t) = ports::read_ports_file(&logs, "transport") {
                        return Launch::Bound { http: h, transport: t };
                    }
                }
            } else if self.answers_as(i, http).await {
                return Launch::Bound { http, transport };
            }
            if Instant::now() >= deadline {
                return Launch::TimedOut;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Whether `http` is served by node `i` of this cluster (not by whatever
    /// else may hold the port).
    async fn answers_as(&self, i: usize, http: u16) -> bool {
        let r = self.c.get(format!("http://127.0.0.1:{http}/")).timeout(Duration::from_secs(2)).send().await;
        let Ok(resp) = r else { return false };
        let v: Value = resp.json().await.unwrap_or(Value::Null);
        v["name"] == format!("n{i}").as_str() && v["cluster_name"] == self.cluster.as_str()
    }

    /// Select an explicit executable for the next restart of a stopped node.
    /// Keeping this per-node permits genuine old/new rolling-upgrade tests.
    pub fn set_binary(&mut self, i: usize, binary: impl AsRef<std::path::Path>) {
        assert!(self.children[i].is_none(), "stop n{i} before replacing its binary");
        let binary = std::fs::canonicalize(binary).expect("resolve node binary");
        assert!(binary.is_file(), "node binary must be a regular file");
        self.binaries[i] = binary;
    }

    /// SIGKILL node `i` (no shutdown: in-flight requests are cut).
    pub fn kill(&mut self, i: usize) {
        if let Some(mut c) = self.children[i].take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    /// Require a process to exit on its own; used to distinguish an explicit
    /// downgrade refusal from a hung or merely undiscoverable node.
    pub async fn wait_exit(&mut self, i: usize, secs: u64) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(secs);
        loop {
            let child = self.children[i].as_mut().expect("node process exists");
            if let Some(status) = child.try_wait().expect("poll child") {
                self.children[i].take();
                return status;
            }
            assert!(Instant::now() < deadline, "n{i} did not exit:\n{}", self.log_tail(i));
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// SIGTERM node `i` and wait (up to `secs`) for it to exit, SIGKILL
    /// after that.
    pub async fn stop(&mut self, i: usize, secs: u64) {
        if let Some(mut c) = self.children[i].take() {
            self.resume_pid(&c);
            super::signal(c.id(), super::SIGTERM);
            let deadline = Instant::now() + Duration::from_secs(secs);
            while Instant::now() < deadline && matches!(c.try_wait(), Ok(None)) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    /// SIGSTOP node `i`: its sockets stay open, it answers nothing (a hung
    /// node, not a dead one).
    pub fn pause(&self, i: usize) {
        let c = self.children[i].as_ref().unwrap_or_else(|| panic!("n{i} is not running"));
        super::signal(c.id(), super::SIGSTOP);
    }

    /// SIGCONT node `i` after [`Procs::pause`].
    pub fn resume(&self, i: usize) {
        if let Some(c) = self.children[i].as_ref() {
            self.resume_pid(c);
        }
    }

    fn resume_pid(&self, c: &Child) {
        super::signal(c.id(), super::SIGCONT);
    }

    pub fn log_path(&self, i: usize) -> PathBuf {
        self.dir.join(format!("n{i}.log"))
    }

    /// The end of node `i`'s output, for assertion messages.
    pub fn log_tail(&self, i: usize) -> String {
        let text = std::fs::read_to_string(self.log_path(i)).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(40)..].join("\n")
    }

    pub async fn call(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.p.http[node])).timeout(Duration::from_secs(120));
        if let Some(b) = body {
            r = r.json(&b);
        }
        // Status 0: no answer (a node starting or killed).
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    /// An `application/x-ndjson` POST (`_bulk`, `_msearch`).
    pub async fn ndjson(&self, node: usize, path: &str, body: String) -> (u16, Value) {
        let r = self.c.post(format!("http://127.0.0.1:{}{path}", self.p.http[node])).timeout(Duration::from_secs(120))
            .header("content-type", "application/x-ndjson").body(body);
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    /// Wait (up to `secs`) until `node` sees `count` nodes.
    /// Also waits for the cluster state to be recovered (ES reports red, and
    /// refuses writes, while the "state not recovered" block is up).
    pub async fn wait_nodes(&self, node: usize, count: usize, secs: u64) {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            let (_, v) = self.call(node, Method::GET, &format!("/_cluster/health?wait_for_nodes={count}&timeout=1s"), None).await;
            let recovered = v["status"] == "green" || v["status"] == "yellow" || self.recovered(node).await;
            if v["number_of_nodes"] == count && v["timed_out"] == false && recovered {
                return;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        panic!("n{node} did not see {count} nodes:\n{}", self.log_tail(node));
    }

    /// No "state not recovered" global block on `node` (a red index does not
    /// make the cluster unrecovered).
    async fn recovered(&self, node: usize) -> bool {
        let (st, v) = self.call(node, Method::GET, "/_cluster/state/blocks?local=true", None).await;
        st == 200 && !v["blocks"]["global"].as_object().is_some_and(|g| g.values().any(|b| b["description"] == "state not recovered / initialized"))
    }

    /// Wait (up to `secs`) until `index` is at least `status` with no
    /// initializing or relocating shards, as `node` sees it.
    pub async fn wait_health(&self, node: usize, index: &str, status: &str, secs: u64) {
        let deadline = Instant::now() + Duration::from_secs(secs);
        let mut last = Value::Null;
        while Instant::now() < deadline {
            let path = format!("/_cluster/health/{index}?wait_for_status={status}&wait_for_no_initializing_shards=true&wait_for_no_relocating_shards=true&timeout=1s");
            let (_, v) = self.call(node, Method::GET, &path, None).await;
            if v["timed_out"] == false && (v["status"] == status || v["status"] == "green") {
                return;
            }
            last = v;
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        panic!("{index} not {status}: {last}");
    }

    /// The elected master's position.
    /// Waits up to 60 s for one to be elected (a just-started or
    /// just-faulted cluster may still be electing).
    pub async fn master(&self, via: usize) -> usize {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let (_, m) = self.call(via, Method::GET, "/_cat/master?format=json", None).await;
            if let Some(name) = m[0]["node"].as_str() {
                return name.trim_start_matches('n').parse().unwrap_or_else(|_| panic!("master {name}"));
            }
            assert!(Instant::now() < deadline, "no master: {m}");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Node position -> node id.
    pub async fn node_ids(&self, via: usize) -> BTreeMap<usize, String> {
        let (st, v) = self.call(via, Method::GET, "/_nodes", None).await;
        assert_eq!(st, 200, "{v}");
        v["nodes"].as_object().into_iter().flatten()
            .filter_map(|(id, n)| Some((n["name"].as_str()?.trim_start_matches('n').parse().ok()?, id.clone())))
            .collect()
    }

    /// `_cat/shards` rows of `index` (`shard`, `prirep`, `state`, `node`, `docs`).
    pub async fn shards(&self, via: usize, index: &str) -> Vec<Value> {
        let (_, v) = self.call(via, Method::GET, &format!("/_cat/shards/{index}?format=json&h=shard,prirep,state,node,docs"), None).await;
        v.as_array().cloned().unwrap_or_default()
    }

    /// Put persistent (or, with `transient`, transient) cluster settings.
    pub async fn settings(&self, via: usize, kind: &str, s: Value) {
        let mut body = serde_json::Map::new();
        body.insert(kind.to_string(), s);
        let (st, v) = self.call(via, Method::PUT, "/_cluster/settings", Some(Value::Object(body))).await;
        assert_eq!(st, 200, "{v}");
    }
}

/// Start `nodes` nodes (with `extra` node settings) and wait for the
/// cluster to form.
pub async fn start_procs(nodes: usize, extra: &[(&str, &str)]) -> Procs {
    let binaries = vec![PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node")); nodes];
    start_procs_with_binaries(extra, &binaries).await
}

/// Start real nodes with explicitly selected executables, one per node.
/// The caller must record the binaries' revisions/hashes with test evidence.
pub async fn start_procs_with_binaries(extra: &[(&str, &str)], binaries: &[PathBuf]) -> Procs {
    let binaries: Vec<PathBuf> = binaries.iter().map(|path| {
        let path = std::fs::canonicalize(path).expect("resolve node binary");
        assert!(path.is_file(), "node binary must be a regular file");
        path
    }).collect();
    let n = binaries.len();
    let cluster = ports::unique("procs");
    let mut procs = Procs {
        c: reqwest::Client::new(),
        p: ProcPorts { http: vec![0; n], transport: vec![0; n] },
        dir: super::scratch_dir(&cluster),
        cluster,
        extra: extra.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        children: (0..n).map(|_| None).collect(),
        binaries,
        current: std::fs::canonicalize(env!("CARGO_BIN_EXE_elasticsearch-node")).expect("resolve this build's node binary"),
    };
    // One at a time: each node seeds from the ports of those before it.
    for i in 0..n {
        procs.spawn(i, &[]).await;
    }
    for i in 0..n {
        procs.wait_nodes(i, n, 120).await;
    }
    procs
}
