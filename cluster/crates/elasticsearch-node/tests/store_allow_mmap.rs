//! `node.store.allow_mmap: false` on a real `elasticsearch-node` process.
//!
//! - Indexing, refresh, deletes, a force merge, search and a restart
//!   (recovery from the commit point) give the same answers as with
//!   mapping, and on Linux `/proc/<pid>/maps` shows no file under
//!   `path.data` mapped. The same node restarted with the default maps
//!   them (the check sees mappings).
//! - `index.store.type: mmapfs` / `hybridfs` are refused with ES's
//!   `IndexModule` error.
//! - An unparsable value fails the start like any ES boolean setting.
#![cfg(unix)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

    /// The HTTP address once the node answers (panics if it exits).
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

    fn stop(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(self.dir.join("logs").join("http.ports"));
    }

    /// Files under `path.data` the node process maps now.
    #[cfg(target_os = "linux")]
    fn mapped_data_files(&self) -> Vec<String> {
        let data = self.dir.join("data").canonicalize().unwrap();
        let data = data.to_string_lossy().into_owned();
        let maps = std::fs::read_to_string(format!("/proc/{}/maps", self.child.id())).unwrap();
        let mut paths: Vec<String> = maps.lines().filter_map(|l| l.split_whitespace().nth(5).filter(|p| p.starts_with(data.as_str())).map(str::to_string)).collect();
        paths.sort();
        paths.dedup();
        paths
    }
}

/// Start a single-node, loopback node on `dir` with `settings`.
fn spawn(dir: &Path, settings: &[(&str, &str)]) -> Node {
    let mut args: Vec<String> = Vec::new();
    let base = [
        ("cluster.name", "allow-mmap".to_string()),
        ("node.name", "am0".to_string()),
        ("path.data", dir.join("data").to_string_lossy().into_owned()),
        ("path.logs", dir.join("logs").to_string_lossy().into_owned()),
        ("network.host", "127.0.0.1".to_string()),
        ("http.port", "0".to_string()),
        ("transport.port", "0".to_string()),
        ("discovery.type", "single-node".to_string()),
        ("node.portsfile", "true".to_string()),
        ("xpack.security.enabled", "false".to_string()),
    ];
    for (k, v) in base.iter().map(|(k, v)| (*k, v.as_str())).chain(settings.iter().copied()) {
        args.push("-E".into());
        args.push(format!("{k}={v}"));
    }
    let stdout = std::fs::File::create(dir.join("stdout.log")).unwrap();
    let stderr = std::fs::File::create(dir.join("stderr.log")).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_elasticsearch-node"))
        .args(&args)
        .env_remove("ES_JAVA_OPTS")
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .expect("spawn elasticsearch-node");
    Node { child, dir: dir.to_path_buf() }
}

/// One HTTP/1.1 request: (status, body).
fn req(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let Ok(mut s) = std::net::TcpStream::connect(addr) else { return (0, String::new()) };
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let ctype = if path.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    if s.read_to_string(&mut out).is_err() {
        return (0, String::new());
    }
    let status = out.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let body = out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, body)
}

fn ok(addr: &str, method: &str, path: &str, body: &str) -> serde_json::Value {
    let (status, text) = req(addr, method, path, body);
    assert!((200..300).contains(&status), "{method} {path}: {status} {text}");
    // Chunked bodies: the JSON object is between the first `{` and the last `}`.
    let json = &text[text.find('{').unwrap()..=text.rfind('}').unwrap()];
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{method} {path}: {e}: {text}"))
}

fn count(addr: &str, q: &str) -> u64 {
    ok(addr, "GET", &format!("/i/_count?q={q}"), "")["count"].as_u64().unwrap()
}

#[test]
fn unmapped_store_indexes_merges_searches_and_recovers_without_mapping_files() {
    let dir = tempfile::Builder::new().prefix("allow-mmap-").tempdir().unwrap();
    let no_mmap = [("node.store.allow_mmap", "false")];
    let mut node = spawn(dir.path(), &no_mmap);
    let addr = node.started();

    // ES `IndexModule`: the mapping store types are refused.
    for store_type in ["mmapfs", "hybridfs"] {
        let (status, body) = req(&addr, "PUT", &format!("/{store_type}-index"), &format!(r#"{{"settings":{{"index.store.type":"{store_type}"}}}}"#));
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("illegal_argument_exception"), "{body}");
        assert!(body.contains(&format!("store type [{store_type}] is not allowed because mmap is disabled")), "{body}");
    }
    // Other store types are accepted, as in ES.
    ok(&addr, "PUT", "/niofs-index", r#"{"settings":{"index.store.type":"niofs","number_of_replicas":0}}"#);

    ok(&addr, "PUT", "/i", r#"{"settings":{"number_of_shards":1,"number_of_replicas":0,"refresh_interval":"-1"}}"#);
    for batch in 0..3 {
        let mut bulk = String::new();
        for i in 0..300 {
            let id = batch * 300 + i;
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{id}\"}}}}\n{{\"t\":\"item {id} widgets b{batch}\",\"n\":{id}}}\n"));
        }
        let r = ok(&addr, "POST", "/i/_bulk", &bulk);
        assert_eq!(r["errors"], false, "{r}");
        ok(&addr, "POST", "/i/_refresh", "");
    }
    ok(&addr, "DELETE", "/i/_doc/7", "");
    ok(&addr, "POST", "/i/_refresh", "");
    ok(&addr, "POST", "/i/_forcemerge?max_num_segments=1", "");
    ok(&addr, "POST", "/i/_flush", "");
    let check = |addr: &str| {
        assert_eq!(count(addr, "t:widgets"), 899);
        assert_eq!(count(addr, "t:b1"), 300);
        assert_eq!(count(addr, "n:[100%20TO%20199]"), 100);
        assert_eq!(count(addr, "n:[0%20TO%209]"), 9);
        let doc = ok(addr, "GET", "/i/_doc/42", "");
        assert_eq!(doc["_source"]["t"], "item 42 widgets b0");
        let top = ok(addr, "GET", "/i/_search", r#"{"query":{"match":{"t":"widgets"}},"sort":[{"n":"desc"}],"size":1}"#);
        assert_eq!(top["hits"]["hits"][0]["_id"], "899");
    };
    check(&addr);
    let segments = ok(&addr, "GET", "/i/_segments", "");
    assert_eq!(segments["indices"]["i"]["shards"]["0"][0]["num_search_segments"], 1, "{segments}");
    #[cfg(target_os = "linux")]
    assert_eq!(node.mapped_data_files(), Vec::<String>::new());
    node.stop();

    // Recovery from the commit point, still unmapped.
    let mut node = spawn(dir.path(), &no_mmap);
    let addr = node.started();
    check(&addr);
    #[cfg(target_os = "linux")]
    assert_eq!(node.mapped_data_files(), Vec::<String>::new());
    node.stop();

    // The default maps the same files (and the check above would see it).
    let mut node = spawn(dir.path(), &[]);
    let addr = node.started();
    check(&addr);
    #[cfg(target_os = "linux")]
    assert!(!node.mapped_data_files().is_empty(), "the default store maps segment files");
    // With mapping allowed, mmapfs is accepted.
    ok(&addr, "PUT", "/mmapfs-index", r#"{"settings":{"index.store.type":"mmapfs","number_of_replicas":0}}"#);
    node.stop();
}

#[test]
fn unparsable_allow_mmap_fails_the_start() {
    let dir = tempfile::Builder::new().prefix("allow-mmap-bad-").tempdir().unwrap();
    let mut node = spawn(dir.path(), &[("node.store.allow_mmap", "yes")]);
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = node.child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "node did not exit:\n{}", node.output());
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(1), "{}", node.output());
    let stderr = std::fs::read_to_string(dir.path().join("stderr.log")).unwrap();
    assert!(stderr.contains("java.lang.IllegalArgumentException: Failed to parse value [yes] as only [true] or [false] are allowed."), "{stderr}");
}

/// Wait until `f` returns `Some`, or panic with `what` and the node output.
fn until<T>(node: &Node, what: &str, secs: u64, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}:\n{}", node.output());
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// `allow_mmap: false` with the node's memory limit below what an index
/// needs: recovering its shard copy from disk fails cleanly (shard-failed,
/// ES's allocation failure with the store file refusal), the node stays up
/// and serves the other index, and once the limit allows it,
/// `_cluster/reroute?retry_failed` brings the index back green.
#[test]
fn an_unmapped_index_over_the_memory_limit_fails_its_copy_and_recovers_once_allowed() {
    let dir = tempfile::Builder::new().prefix("allow-mmap-limit-").tempdir().unwrap();
    let no_mmap = ("node.store.allow_mmap", "false");
    let mut node = spawn(dir.path(), &[no_mmap]);
    let addr = node.started();
    let parent_used = |addr: &str| -> u64 {
        let stats = ok(addr, "GET", "/_nodes/_local/stats/breaker", "");
        stats["nodes"].as_object().unwrap().values().next().unwrap()["breakers"]["parent"]["estimated_size_in_bytes"].as_u64().unwrap()
    };

    ok(&addr, "PUT", "/small", r#"{"settings":{"number_of_shards":1,"number_of_replicas":0}}"#);
    ok(&addr, "POST", "/small/_bulk?refresh=true", "{\"index\":{\"_id\":\"1\"}}\n{\"t\":\"small one\"}\n{\"index\":{\"_id\":\"2\"}}\n{\"t\":\"small two\"}\n");
    // What a node uses before it holds the big index.
    std::thread::sleep(Duration::from_secs(2));
    let baseline = parent_used(&addr);
    // Incompressible source, one segment: one large store file.
    ok(&addr, "PUT", "/big", r#"{"settings":{"number_of_shards":1,"number_of_replicas":0,"refresh_interval":"-1"},
        "mappings":{"properties":{"blob":{"type":"keyword","index":false,"doc_values":false}}}}"#);
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let docs = 120_000;
    for batch in 0..docs / 4000 {
        let mut bulk = String::new();
        for i in 0..4000 {
            let blob: String = (0..1024).map(|_| B64[(next() % 64) as usize] as char).collect();
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n{{\"blob\":\"{blob}\"}}\n", batch * 4000 + i));
        }
        assert_eq!(ok(&addr, "POST", "/big/_bulk", &bulk)["errors"], false);
    }
    ok(&addr, "POST", "/big/_refresh", "");
    ok(&addr, "POST", "/big/_forcemerge?max_num_segments=1", "");
    ok(&addr, "POST", "/big/_flush", "");
    assert_eq!(ok(&addr, "GET", "/big/_count", "")["count"], docs);
    let store = ok(&addr, "GET", "/big/_stats/store", "")["_all"]["primaries"]["store"]["size_in_bytes"].as_u64().unwrap();
    assert!(store > 60 << 20, "the big index is {store} bytes");
    node.stop();

    // Restart with room for half of the big index above that baseline:
    // recovering its copy reads the store file into memory, refused.
    let limit = baseline + store / 2;
    let limit_setting = limit.to_string() + "b";
    let mut node = spawn(dir.path(), &[no_mmap, ("indices.breaker.total.limit", limit_setting.as_str())]);
    let deadline = Instant::now() + Duration::from_secs(120);
    let addr = loop {
        if let Some(addr) = node.http_addr() {
            if req(&addr, "GET", "/_cluster/health/small?wait_for_status=green&timeout=5s", "").0 == 200 {
                break addr;
            }
        }
        assert!(node.child.try_wait().unwrap().is_none(), "the node exited:\n{}", node.output());
        assert!(Instant::now() < deadline, "node did not start:\n{}", node.output());
        std::thread::sleep(Duration::from_millis(200));
    };
    // Every allocation attempt is refused (the master retries up to
    // `index.allocation.max_retries`), each failing the copy at once.
    let explain = until(&node, "the big copy to fail its allocation for memory", 180, || {
        let (status, body) = req(&addr, "GET", "/_cluster/allocation/explain", r#"{"index":"big","shard":0,"primary":true}"#);
        if status != 200 {
            return None;
        }
        let v: serde_json::Value = serde_json::from_str(&body[body.find('{')?..=body.rfind('}')?]).ok()?;
        let failed = v["unassigned_info"]["reason"] == "ALLOCATION_FAILED" && v["unassigned_info"]["details"].as_str().is_some_and(|d| d.contains("cannot read store file"));
        // The default `index.allocation.max_retries` (5) used up: the
        // master stops retrying until `retry_failed`.
        (failed && v["unassigned_info"]["failed_allocation_attempts"] == 5).then_some(v)
    });
    let details = explain["unassigned_info"]["details"].as_str().unwrap_or_default().to_string();
    assert!(details.contains("failed to open shard copy: [circuit_breaking_exception] cannot read store file"), "{explain}");
    assert!(details.contains("[seg_") && details.contains(&format!("limit {limit}")), "names the file and the limit [{limit}]: {explain}");
    assert!(details.contains("[node.store.allow_mmap] is false") && details.contains("[vm.max_map_count]"), "{explain}");
    assert_eq!(explain["unassigned_info"]["last_allocation_status"], "no", "{explain}");
    // This node's explain reports a primary out of retries as
    // `no_valid_shard_copy` (ES: `no`, from the max-retry decider).
    assert!(matches!(explain["can_allocate"].as_str(), Some("no" | "no_valid_shard_copy")), "{explain}");
    std::thread::sleep(Duration::from_secs(2));
    let again = ok(&addr, "GET", "/_cluster/allocation/explain", r#"{"index":"big","shard":0,"primary":true}"#);
    assert_eq!(again["unassigned_info"]["failed_allocation_attempts"], 5, "no automatic retry past max_retries: {again}");

    // The node is up and serves the other index.
    assert!(node.child.try_wait().unwrap().is_none(), "the node exited:\n{}", node.output());
    assert_eq!(count_of(&addr, "small", "t:small"), 2);
    assert_eq!(ok(&addr, "GET", "/_cluster/health/small", "")["status"], "green");
    assert_eq!(ok(&addr, "GET", "/_cluster/health/big", "")["status"], "red");

    // With room for it (the cluster setting overrides the node's), a retry
    // brings it back.
    ok(&addr, "PUT", "/_cluster/settings", r#"{"persistent":{"indices.breaker.total.limit":"95%"}}"#);
    ok(&addr, "POST", "/_cluster/reroute?retry_failed=true", "");
    until(&node, "the big index to go green", 120, || (ok(&addr, "GET", "/_cluster/health/big", "")["status"] == "green").then_some(()));
    assert_eq!(ok(&addr, "GET", "/big/_count", "")["count"], docs);
    let hit = ok(&addr, "GET", "/big/_doc/119999", "");
    assert_eq!(hit["found"], true, "{hit}");
    node.stop();
}

fn count_of(addr: &str, index: &str, q: &str) -> u64 {
    ok(addr, "GET", &format!("/{index}/_count?q={q}"), "")["count"].as_u64().unwrap()
}
