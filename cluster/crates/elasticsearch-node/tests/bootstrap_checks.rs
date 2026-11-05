//! ES `BootstrapChecks` on a real `elasticsearch-node` process.
//!
//! Each node runs with limits lowered for the child only (`setrlimit` in
//! `pre_exec`): `RLIMIT_FSIZE` 1 TiB fails `MaxFileSizeCheck`, and
//! `RLIMIT_MEMLOCK` 64 KiB makes `bootstrap.memory_lock: true` fail
//! (`MlockallCheck`, unless the test runs with `CAP_IPC_LOCK`). No
//! discovery setting fails `DiscoveryConfiguredCheck`.
//! - Transport bound to a non-loopback address: exit 78 with ES's
//!   "bootstrap check failure [i] of [n]" lines.
//! - The same with `discovery.type: single-node`: not enforced, the node
//!   starts and `_nodes` reports `process.mlockall: false`.
//! - Loopback only: not enforced either (only warnings), unless
//!   `-Des.enforce.bootstrap.checks=true` is in `ES_JAVA_OPTS`.
#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5";
const FSIZE: u64 = 1 << 40;

struct Node {
    child: Child,
    dir: PathBuf,
}

impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Node {
    fn stderr(&self) -> String {
        std::fs::read_to_string(self.dir.join("stderr.log")).unwrap_or_default()
    }

    fn output(&self) -> String {
        format!("--- stdout\n{}\n--- stderr\n{}", std::fs::read_to_string(self.dir.join("stdout.log")).unwrap_or_default(), self.stderr())
    }

    /// Exit status, or `None` once the node has written its ports file
    /// (it started).
    fn wait_exit_or_start(&mut self) -> Option<ExitStatus> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(status) = self.child.try_wait().expect("poll child") {
                return Some(status);
            }
            if self.http_addr().is_some() {
                return None;
            }
            assert!(Instant::now() < deadline, "node neither exited nor started:\n{}", self.output());
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn http_addr(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.dir.join("logs").join("http.ports")).ok()?;
        text.lines().next().map(str::to_string)
    }
}

/// Start the node binary with `settings` and the lowered limits.
fn spawn(settings: &[(&str, &str)], env: &[(&str, &str)]) -> Node {
    let dir = tempfile::Builder::new().prefix("bootstrap-checks-").tempdir().expect("temp dir").keep();
    let mut args: Vec<String> = Vec::new();
    let base = [
        ("cluster.name", "bootstrap-checks".to_string()),
        ("node.name", "bc0".to_string()),
        ("path.data", dir.join("data").to_string_lossy().into_owned()),
        ("path.logs", dir.join("logs").to_string_lossy().into_owned()),
        ("http.host", "127.0.0.1".to_string()),
        ("http.port", "0".to_string()),
        ("transport.port", "0".to_string()),
        ("node.portsfile", "true".to_string()),
        ("xpack.security.enabled", "false".to_string()),
        ("bootstrap.memory_lock", "true".to_string()),
    ];
    for (k, v) in base.iter().map(|(k, v)| (*k, v.as_str())).chain(settings.iter().copied()) {
        args.push("-E".into());
        args.push(format!("{k}={v}"));
    }
    let stdout = std::fs::File::create(dir.join("stdout.log")).unwrap();
    let stderr = std::fs::File::create(dir.join("stderr.log")).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_elasticsearch-node"));
    cmd.args(&args).env_remove("ES_JAVA_OPTS").env_remove("ES_ENFORCE_BOOTSTRAP_CHECKS").envs(env.iter().copied()).stdin(Stdio::null()).stdout(stdout).stderr(stderr);
    unsafe {
        cmd.pre_exec(|| {
            let lower = |resource, value: u64| {
                let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
                if libc::getrlimit(resource, &mut r) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                r.rlim_cur = (value as libc::rlim_t).min(r.rlim_max);
                if libc::setrlimit(resource, &r) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            };
            lower(libc::RLIMIT_FSIZE, FSIZE)?;
            lower(libc::RLIMIT_MEMLOCK, 64 * 1024)?;
            Ok(())
        });
    }
    let child = cmd.spawn().expect("spawn elasticsearch-node");
    Node { child, dir }
}

fn has_ipc_lock() -> bool {
    unsafe { libc::geteuid() == 0 }
}

/// `GET path` over plain HTTP/1.1; the body with whitespace removed.
fn get(addr: &str, path: &str) -> String {
    let mut s = std::net::TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out.chars().filter(|c| !c.is_whitespace()).collect()
}

fn fsize_failure(user: &str) -> String {
    format!("max file size [{FSIZE}] for user [{user}] is too low, increase to [unlimited]; for more information see [{DOCS}#bootstrap-checks-max-file-size]")
}

fn discovery_failure() -> String {
    format!("the default discovery settings are unsuitable for production use; at least one of [discovery.seed_hosts, discovery.seed_providers, cluster.initial_master_nodes] must be configured; for more information see [{DOCS}#bootstrap-checks-discovery-configuration]")
}

fn memlock_failure() -> String {
    format!("memory locking requested for elasticsearch process but memory is not locked; for more information see [{DOCS}#bootstrap-checks-memory-lock]")
}

/// The failure lines of ES's `NodeValidationException` in `stderr`
/// (after `ERROR: `), checked for ES's header and numbering.
fn failure_lines(stderr: &str) -> Vec<String> {
    let start = stderr.find("ERROR: [").unwrap_or_else(|| panic!("no node validation error:\n{stderr}"));
    let mut lines = stderr[start + "ERROR: ".len()..].lines();
    let header = lines.next().unwrap();
    let n: usize = header[1..header.find(']').unwrap()].parse().unwrap();
    assert_eq!(header, format!("[{n}] bootstrap checks failed. You must address the points described in the following [{n}] lines before starting Elasticsearch. For more information see [{DOCS}]"));
    (1..=n)
        .map(|i| {
            let line = lines.next().unwrap_or_else(|| panic!("missing failure {i} of {n}:\n{stderr}"));
            let prefix = format!("bootstrap check failure [{i}] of [{n}]: ");
            line.strip_prefix(prefix.as_str()).unwrap_or_else(|| panic!("[{line}] does not start with [{prefix}]")).to_string()
        })
        .collect()
}

fn user() -> String {
    elasticsearch_node::natives::user_name()
}

#[test]
fn non_loopback_transport_enforces_the_checks_with_exit_78() {
    let mut node = spawn(&[("transport.host", "0.0.0.0")], &[]);
    let status = node.wait_exit_or_start().unwrap_or_else(|| panic!("started despite failing bootstrap checks:\n{}", node.output()));
    assert_eq!(status.code(), Some(78), "{}", node.output());
    let failures = failure_lines(&node.stderr());
    // ES's order: ... MlockallCheck, ..., MaxFileSizeCheck, ..., DiscoveryConfiguredCheck.
    let fsize = failures.iter().position(|f| *f == fsize_failure(&user())).unwrap_or_else(|| panic!("no max file size failure in {failures:#?}"));
    let discovery = failures.iter().position(|f| *f == discovery_failure()).unwrap_or_else(|| panic!("no discovery failure in {failures:#?}"));
    assert!(fsize < discovery, "{failures:#?}");
    if !has_ipc_lock() && cfg!(target_os = "linux") {
        let memlock = failures.iter().position(|f| *f == memlock_failure()).unwrap_or_else(|| panic!("no memory lock failure in {failures:#?}"));
        assert!(memlock < fsize, "{failures:#?}");
        assert!(node.output().contains("Unable to lock JVM Memory: error=12"), "{}", node.output());
    }
}

#[test]
fn single_node_discovery_skips_enforcement() {
    let mut node = spawn(&[("transport.host", "0.0.0.0"), ("discovery.type", "single-node")], &[]);
    assert!(node.wait_exit_or_start().is_none(), "single-node discovery must not enforce bootstrap checks:\n{}", node.output());
    let addr = node.http_addr().unwrap();
    let body = get(&addr, "/_nodes/_local/process?filter_path=nodes.*.process.mlockall");
    let expect = format!("\"mlockall\":{}", has_ipc_lock());
    assert!(body.contains(&expect), "{expect} not in {body}");
    // The failures are logged as warnings instead.
    let out = node.output();
    assert!(out.contains(&fsize_failure(&user())), "{out}");
    assert!(!out.contains("bootstrap checks failed"), "{out}");
}

#[test]
fn loopback_only_warns_unless_explicitly_enforced() {
    let mut node = spawn(&[("network.host", "127.0.0.1")], &[]);
    assert!(node.wait_exit_or_start().is_none(), "a loopback node must not enforce bootstrap checks:\n{}", node.output());
    let out = node.output();
    assert!(out.contains(&discovery_failure()), "the failure is a warning: {out}");
    drop(node);

    let mut node = spawn(&[("network.host", "127.0.0.1")], &[("ES_JAVA_OPTS", "-Des.enforce.bootstrap.checks=true")]);
    let status = node.wait_exit_or_start().unwrap_or_else(|| panic!("started despite explicit enforcement:\n{}", node.output()));
    assert_eq!(status.code(), Some(78), "{}", node.output());
    let failures = failure_lines(&node.stderr());
    assert!(failures.contains(&discovery_failure()), "{failures:#?}");
}

