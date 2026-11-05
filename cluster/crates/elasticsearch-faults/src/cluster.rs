//! Real node processes on one host.
//!
//! Node `i` (0-based) binds HTTP and transport to its own loopback address
//! `127.0.<port_base/100>.<i+1>` (so partitions can be expressed as address
//! pairs, see `net.rs`), HTTP port `port_base+i`, transport port
//! `port_base+100+i`, data in `<dir>/node-<i+1>`. Each process is placed in
//! its own cgroup (`/sys/fs/cgroup/es-faults-<port_base>/n<i+1>`) *before* it
//! execs the binary, so the SNAT rule for that cgroup applies to every
//! connection it ever opens.

use parking_lot::Mutex;
use std::fs::OpenOptions;
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[cfg(unix)]
use libc::{SIGCONT, SIGSTOP, SIGTERM};
// Placeholders: `signal` is a no-op without Unix signals.
#[cfg(not(unix))]
const SIGCONT: i32 = 18;
#[cfg(not(unix))]
const SIGSTOP: i32 = 19;
#[cfg(not(unix))]
const SIGTERM: i32 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProcState {
    Down,
    Running,
    Paused,
}

struct Slot {
    child: Option<Child>,
    state: ProcState,
    /// SIGTERM sent by the harness: an exit is expected, not a crash.
    stopping: bool,
}

pub struct Cluster {
    pub n: usize,
    pub port_base: u16,
    pub bin: PathBuf,
    pub dir: PathBuf,
    /// Extra `-E` settings for every node.
    pub extra: Vec<String>,
    slots: Mutex<Vec<Slot>>,
    /// Set once the cluster has formed: after that no node is ever given
    /// `cluster.initial_master_nodes` again (a wiped node must join, never
    /// bootstrap a second cluster).
    formed: AtomicBool,
}

pub fn cgroup_root(port_base: u16) -> String {
    format!("es-faults-{port_base}")
}

pub fn node_ip(port_base: u16, i: usize) -> Ipv4Addr {
    Ipv4Addr::new(127, 0, (port_base / 100) as u8, (i + 1) as u8)
}

pub fn cluster_name(port_base: u16) -> String {
    format!("faults-{port_base}")
}

fn sudo(args: &[&str]) -> Result<String, String> {
    let out = Command::new("sudo").arg("-n").args(args).output().map_err(|e| format!("sudo {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!("sudo {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

impl Cluster {
    pub fn new(n: usize, port_base: u16, bin: PathBuf, dir: PathBuf, extra: Vec<String>) -> Cluster {
        let slots = (0..n).map(|_| Slot { child: None, state: ProcState::Down, stopping: false }).collect();
        Cluster { n, port_base, bin, dir, extra, slots: Mutex::new(slots), formed: AtomicBool::new(false) }
    }

    pub fn ip(&self, i: usize) -> Ipv4Addr {
        node_ip(self.port_base, i)
    }
    pub fn http_port(&self, i: usize) -> u16 {
        self.port_base + i as u16
    }
    pub fn transport_port(&self, i: usize) -> u16 {
        self.port_base + 100 + i as u16
    }
    pub fn url(&self, i: usize) -> String {
        format!("http://{}:{}", self.ip(i), self.http_port(i))
    }
    pub fn name(i: usize) -> String {
        format!("node-{}", i + 1)
    }
    pub fn index_of_name(name: &str) -> Option<usize> {
        name.strip_prefix("node-").and_then(|s| s.parse::<usize>().ok()).and_then(|x| x.checked_sub(1))
    }
    pub fn data_dir(&self, i: usize) -> PathBuf {
        self.dir.join(format!("node-{}", i + 1))
    }
    pub fn log_path(&self, i: usize) -> PathBuf {
        self.dir.join(format!("node-{}.log", i + 1))
    }
    pub fn cgroup(&self, i: usize) -> String {
        format!("{}/n{}", cgroup_root(self.port_base), i + 1)
    }
    pub fn mark_formed(&self) {
        self.formed.store(true, Ordering::SeqCst);
    }

    pub fn setup_cgroups(&self) -> Result<(), String> {
        for i in 0..self.n {
            sudo(&["mkdir", "-p", &format!("/sys/fs/cgroup/{}", self.cgroup(i))])?;
        }
        Ok(())
    }

    pub fn state(&self, i: usize) -> ProcState {
        self.slots.lock()[i].state
    }
    pub fn states(&self) -> Vec<ProcState> {
        self.slots.lock().iter().map(|s| s.state).collect()
    }
    pub fn running(&self) -> Vec<usize> {
        self.states().iter().enumerate().filter(|(_, s)| **s == ProcState::Running).map(|(i, _)| i).collect()
    }

    fn args(&self, i: usize) -> Vec<String> {
        let seeds: Vec<String> = (0..self.n).map(|j| format!("{}:{}", self.ip(j), self.transport_port(j))).collect();
        let mut a = vec![
            format!("cluster.name={}", cluster_name(self.port_base)),
            format!("node.name={}", Self::name(i)),
            format!("path.data={}", self.data_dir(i).display()),
            format!("network.host={}", self.ip(i)),
            format!("http.port={}", self.http_port(i)),
            format!("transport.port={}", self.transport_port(i)),
            format!("discovery.seed_hosts={}", seeds.join(",")),
            format!("node.attr.zone=zone-{}", i % 3),
            // Security defaults to on (ES 8+); the fault matrix is not about it.
            "xpack.security.enabled=false".to_string(),
        ];
        // A node that persisted a cluster state (`state-<generation>.json`,
        // or the older `coordination.json`) never bootstraps again.
        let persisted = std::fs::read_dir(self.data_dir(i).join("_cluster")).map(|d| {
            d.filter_map(|e| e.ok()).any(|e| e.file_name().to_str().is_some_and(|n| n == "coordination.json" || (n.starts_with("state-") && n.ends_with(".json"))))
        }).unwrap_or(false);
        if !self.formed.load(Ordering::SeqCst) && !persisted {
            let masters: Vec<String> = (0..self.n).map(Self::name).collect();
            a.push(format!("cluster.initial_master_nodes={}", masters.join(",")));
        }
        a.extend(self.extra.iter().cloned());
        a.into_iter().flat_map(|s| ["-E".to_string(), s]).collect()
    }

    pub fn start(&self, i: usize) -> Result<(), String> {
        let mut slots = self.slots.lock();
        if slots[i].state != ProcState::Down {
            return Ok(());
        }
        std::fs::create_dir_all(self.data_dir(i)).map_err(|e| e.to_string())?;
        let conf = self.dir.join("conf-none");
        std::fs::create_dir_all(&conf).map_err(|e| e.to_string())?;
        let log = OpenOptions::new().create(true).append(true).open(self.log_path(i)).map_err(|e| e.to_string())?;
        let log2 = log.try_clone().map_err(|e| e.to_string())?;
        // The shell joins the node's cgroup, then execs the binary (same pid).
        let script = format!("echo $$ | sudo -n tee /sys/fs/cgroup/{}/cgroup.procs >/dev/null || exit 97; exec \"$0\" \"$@\"", self.cgroup(i));
        let child = Command::new("sh")
            .arg("-c")
            .arg(script)
            .arg(&self.bin)
            .args(self.args(i))
            .env("ES_PATH_CONF", &conf)
            .env("RUST_LOG", std::env::var("FAULTS_NODE_LOG").unwrap_or_else(|_| "info".into()))
            .stdin(Stdio::null())
            .stdout(log)
            .stderr(log2)
            .spawn()
            .map_err(|e| format!("spawn node-{}: {e}", i + 1))?;
        let _ = std::fs::write(self.dir.join(format!("node-{}.pid", i + 1)), child.id().to_string());
        slots[i] = Slot { child: Some(child), state: ProcState::Running, stopping: false };
        Ok(())
    }

    #[cfg(unix)]
    fn signal(&self, i: usize, sig: i32) -> bool {
        let slots = self.slots.lock();
        match slots[i].child.as_ref() {
            Some(c) => unsafe { libc::kill(c.id() as i32, sig) == 0 },
            None => false,
        }
    }

    /// No signals on Windows: pause/resume report failure and a graceful
    /// stop falls back to killing the process.
    #[cfg(not(unix))]
    fn signal(&self, _i: usize, _sig: i32) -> bool {
        false
    }

    /// `kill -9`; reaps the child.
    pub fn kill9(&self, i: usize) {
        let mut slots = self.slots.lock();
        if let Some(mut c) = slots[i].child.take() {
            // SIGKILL on Unix, TerminateProcess on Windows.
            let _ = c.kill();
            let _ = c.wait();
        }
        slots[i].state = ProcState::Down;
    }

    pub fn pause(&self, i: usize) -> bool {
        if self.state(i) != ProcState::Running {
            return false;
        }
        let ok = self.signal(i, SIGSTOP);
        if ok {
            self.slots.lock()[i].state = ProcState::Paused;
        }
        ok
    }

    pub fn resume(&self, i: usize) -> bool {
        if self.state(i) != ProcState::Paused {
            return false;
        }
        let ok = self.signal(i, SIGCONT);
        self.slots.lock()[i].state = ProcState::Running;
        ok
    }

    /// Graceful stop (SIGTERM), escalating to SIGKILL after `grace`.
    pub async fn stop(&self, i: usize, grace: Duration) {
        if self.state(i) == ProcState::Paused {
            self.resume(i);
        }
        self.slots.lock()[i].stopping = true;
        if !self.signal(i, SIGTERM) {
            self.kill9(i);
            return;
        }
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            let exited = {
                let mut slots = self.slots.lock();
                match slots[i].child.as_mut() {
                    Some(c) => matches!(c.try_wait(), Ok(Some(_))),
                    None => true,
                }
            };
            if exited {
                let mut slots = self.slots.lock();
                slots[i].child = None;
                slots[i].state = ProcState::Down;
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                self.kill9(i);
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Delete the node's data directory (node must be down).
    pub fn wipe(&self, i: usize) -> Result<(), String> {
        if self.state(i) != ProcState::Down {
            return Err(format!("node-{} is not down", i + 1));
        }
        let d = self.data_dir(i);
        if d.exists() {
            std::fs::remove_dir_all(&d).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Detect nodes that died on their own (crash, panic=abort, exit).
    pub fn reap_crashed(&self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        let mut slots = self.slots.lock();
        for (i, s) in slots.iter_mut().enumerate() {
            if s.stopping {
                continue;
            }
            if let Some(c) = s.child.as_mut() {
                if let Ok(Some(st)) = c.try_wait() {
                    out.push((i, format!("{st}")));
                    s.child = None;
                    s.state = ProcState::Down;
                }
            }
        }
        out
    }

    pub fn kill_all(&self) {
        for i in 0..self.n {
            if self.state(i) == ProcState::Paused {
                self.signal(i, SIGCONT);
            }
            self.kill9(i);
        }
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        self.kill_all();
    }
}

/// Best-effort removal of everything a (possibly crashed) harness run on
/// `port_base` left behind: node processes and cgroups. Network rules are
/// removed by `net::teardown`.
pub fn cleanup_processes(port_base: u16) {
    let _ = Command::new("pkill").args(["-9", "-f", &format!("cluster.name={} ", cluster_name(port_base))]).status();
    let _ = Command::new("pkill").args(["-9", "-f", &format!("cluster.name={}$", cluster_name(port_base))]).status();
    std::thread::sleep(Duration::from_millis(300));
    let root = cgroup_root(port_base);
    for i in 0..16 {
        let _ = sudo(&["rmdir", &format!("/sys/fs/cgroup/{root}/n{}", i + 1)]);
    }
    let _ = sudo(&["rmdir", &format!("/sys/fs/cgroup/{root}")]);
}
