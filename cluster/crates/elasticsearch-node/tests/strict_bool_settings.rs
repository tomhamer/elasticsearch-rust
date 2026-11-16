//! Boolean settings and parameters parse as ES `Booleans.parseBoolean`:
//! `true` / `false` only. A mistyped `xpack.security.enabled: tru` used to
//! read as false (and, being set, skipped security auto-configuration), so
//! a typo silently disabled authentication.
//!
//! - A node whose boolean setting is invalid (elasticsearch.yml, `-E`, a
//!   dotted environment variable) does not start: exit 1 with ES's
//!   message, before it binds a port or writes its data path.
//! - `PUT _cluster/settings`, index creation, index settings updates,
//!   templates and boolean request parameters answer 400 with ES's error.
#![cfg(unix)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const MESSAGE: &str = "as only [true] or [false] are allowed.";

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
    fn output(&self) -> String {
        let read = |f: &str| std::fs::read_to_string(self.dir.join(f)).unwrap_or_default();
        format!("--- stdout\n{}\n--- stderr\n{}", read("stdout.log"), read("stderr.log"))
    }

    fn http_addr(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.dir.join("logs").join("http.ports")).ok()?;
        text.lines().next().map(str::to_string)
    }

    fn started(&mut self) -> String {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("node exited with {status}:\n{}", self.output());
            }
            if let Some(addr) = self.http_addr() {
                if req(&addr, "GET", "/_cluster/health?wait_for_status=green&timeout=30s", "").0 == 200 {
                    return addr;
                }
            }
            assert!(Instant::now() < deadline, "node did not start:\n{}", self.output());
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// The exit status of a node that must fail its start.
    fn exited(&mut self) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "node did not exit:\n{}", self.output());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// A single-node loopback node on `dir`: `-E` settings (the base ones
/// unless overridden), an optional elasticsearch.yml and extra environment.
fn spawn(dir: &Path, settings: &[(&str, &str)], yml: Option<&str>, env: &[(&str, &str)]) -> Node {
    let conf = dir.join("config");
    std::fs::create_dir_all(&conf).unwrap();
    std::fs::write(conf.join("elasticsearch.yml"), yml.unwrap_or("")).unwrap();
    let base = [
        ("cluster.name", "strict-bool".to_string()),
        ("node.name", "sb0".to_string()),
        ("path.data", dir.join("data").to_string_lossy().into_owned()),
        ("path.logs", dir.join("logs").to_string_lossy().into_owned()),
        ("network.host", "127.0.0.1".to_string()),
        ("http.port", "0".to_string()),
        ("transport.port", "0".to_string()),
        ("discovery.type", "single-node".to_string()),
        ("node.portsfile", "true".to_string()),
    ];
    let mut args: Vec<String> = Vec::new();
    for (k, v) in base.iter().map(|(k, v)| (*k, v.as_str())).chain(settings.iter().copied()) {
        args.push("-E".into());
        args.push(format!("{k}={v}"));
    }
    let stdout = std::fs::File::create(dir.join("stdout.log")).unwrap();
    let stderr = std::fs::File::create(dir.join("stderr.log")).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_elasticsearch-node"));
    cmd.args(&args).env_remove("ES_JAVA_OPTS").env("ES_PATH_CONF", &conf).stdin(Stdio::null()).stdout(stdout).stderr(stderr);
    for (k, v) in env {
        cmd.env(k, v);
    }
    Node { child: cmd.spawn().expect("spawn elasticsearch-node"), dir: dir.to_path_buf() }
}

fn req(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let Ok(mut s) = std::net::TcpStream::connect(addr) else { return (0, String::new()) };
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    if s.read_to_string(&mut out).is_err() {
        return (0, String::new());
    }
    let status = out.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let body = out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, body)
}

/// A 400 `illegal_argument_exception` with ES's boolean message for `value`.
fn bad_bool(addr: &str, method: &str, path: &str, body: &str, value: &str) {
    let (status, text) = req(addr, method, path, body);
    assert_eq!(status, 400, "{method} {path}: {text}");
    assert!(text.contains("illegal_argument_exception"), "{method} {path}: {text}");
    assert!(text.contains(&format!("Failed to parse value [{value}] {MESSAGE}")), "{method} {path}: {text}");
}

#[test]
fn an_invalid_security_boolean_fails_the_start_before_binding() {
    let cases: Vec<(&str, Vec<(&str, &str)>, Option<&str>, Vec<(&str, &str)>, &str)> = vec![
        ("-E", vec![("xpack.security.enabled", "tru")], None, vec![], "tru"),
        ("yml", vec![], Some("xpack.security.enabled: tru\n"), vec![], "tru"),
        ("env", vec![], None, vec![("xpack.security.enabled", "TRUE")], "TRUE"),
        ("http ssl", vec![("xpack.security.enabled", "true"), ("xpack.security.http.ssl.enabled", "yes")], None, vec![], "yes"),
        ("transport ssl", vec![("xpack.security.transport.ssl.enabled", "1")], None, vec![], "1"),
        ("realm", vec![("xpack.security.authc.realms.file.file1.order", "0"), ("xpack.security.authc.realms.file.file1.enabled", "fals")], None, vec![], "fals"),
    ];
    for (what, settings, yml, env, value) in cases {
        let dir = tempfile::Builder::new().prefix("strict-bool-").tempdir().unwrap();
        let mut node = spawn(dir.path(), &settings, yml, &env);
        let status = node.exited();
        assert_eq!(status.code(), Some(1), "{what}: {}", node.output());
        let stderr = std::fs::read_to_string(dir.path().join("stderr.log")).unwrap();
        assert!(stderr.contains(&format!("java.lang.IllegalArgumentException: Failed to parse value [{value}] {MESSAGE}")), "{what}: {stderr}");
        // Nothing bound or written: no ports file, no node state.
        assert!(node.http_addr().is_none(), "{what}: the node bound a port");
        let data = dir.path().join("data");
        assert!(!data.exists() || std::fs::read_dir(&data).unwrap().next().is_none(), "{what}: the node wrote its data path");
    }
}

#[test]
fn invalid_booleans_in_requests_are_400s() {
    let dir = tempfile::Builder::new().prefix("strict-bool-api-").tempdir().unwrap();
    let mut node = spawn(dir.path(), &[("xpack.security.enabled", "false")], None, &[]);
    let addr = node.started();
    // Cluster settings, persistent and transient, strings of any case.
    bad_bool(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"action.destructive_requires_name":"tru"}}"#, "tru");
    bad_bool(&addr, "PUT", "/_cluster/settings", r#"{"transient":{"cluster.routing.allocation.disk.threshold_enabled":"TRUE"}}"#, "TRUE");
    bad_bool(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"search.allow_expensive_queries":"yes"}}"#, "yes");
    let (status, text) = req(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"search.allow_expensive_queries":"false"}}"#);
    assert_eq!(status, 200, "{text}");
    let (status, text) = req(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"search.allow_expensive_queries":true}}"#);
    assert_eq!(status, 200, "{text}");
    // `action.auto_create_index` takes patterns too.
    let (status, text) = req(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"action.auto_create_index":"+logs*,-*"}}"#);
    assert_eq!(status, 200, "{text}");
    let (status, text) = req(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"action.auto_create_index":null}}"#);
    assert_eq!(status, 200, "{text}");
    // Index settings: creation, update, templates.
    bad_bool(&addr, "PUT", "/i1", r#"{"settings":{"index.hidden":"tru"}}"#, "tru");
    bad_bool(&addr, "PUT", "/i1", r#"{"settings":{"index":{"blocks":{"write":"True"}}}}"#, "True");
    let (status, text) = req(&addr, "GET", "/i1", "");
    assert_eq!(status, 404, "a refused index was created: {text}");
    let (status, text) = req(&addr, "PUT", "/i2", r#"{"settings":{"number_of_replicas":0}}"#);
    assert_eq!(status, 200, "{text}");
    bad_bool(&addr, "PUT", "/i2/_settings", r#"{"index.blocks.read_only":"on"}"#, "on");
    bad_bool(&addr, "PUT", "/i2/_settings", r#"{"index":{"queries":{"cache":{"enabled":"0"}}}}"#, "0");
    let (status, text) = req(&addr, "PUT", "/i2/_settings", r#"{"index.blocks.read_only":"false"}"#);
    assert_eq!(status, 200, "{text}");
    bad_bool(&addr, "PUT", "/_index_template/t1", r#"{"index_patterns":["t1-*"],"template":{"settings":{"index.hidden":"tru"}}}"#, "tru");
    bad_bool(&addr, "PUT", "/_component_template/c1", r#"{"template":{"settings":{"index.hidden":"tru"}}}"#, "tru");
    bad_bool(&addr, "PUT", "/_template/l1", r#"{"index_patterns":["l1-*"],"settings":{"index.hidden":"tru"}}"#, "tru");
    // Request parameters ES reads with `paramAsBoolean`.
    bad_bool(&addr, "GET", "/i2/_search?rest_total_hits_as_int=tru", "", "tru");
    bad_bool(&addr, "GET", "/i2/_search?ignore_unavailable=yes", "", "yes");
    bad_bool(&addr, "GET", "/_cluster/settings?flat_settings=1", "", "1");
    bad_bool(&addr, "PUT", "/i2/_doc/1?require_alias=TRUE", r#"{"a":1}"#, "TRUE");
    bad_bool(&addr, "GET", "/_cat/indices?v=tru", "", "tru");
    for path in ["/i2/_search?rest_total_hits_as_int", "/i2/_search?rest_total_hits_as_int=true", "/_cat/indices?v", "/_cluster/settings?flat_settings=false"] {
        let (status, text) = req(&addr, "GET", path, "");
        assert_eq!(status, 200, "{path}: {text}");
    }
}
