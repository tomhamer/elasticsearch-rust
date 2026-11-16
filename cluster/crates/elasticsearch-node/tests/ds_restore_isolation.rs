//! Data stream restores never touch live streams (production review #103,
//! #104), on a two-node in-process cluster:
//!
//! * #103: restoring an expired old backing index by its own name, while
//!   its stream lives on in a newer generation, restores a plain index: the
//!   live stream's membership, reads and write target are unchanged. A
//!   whole-stream restore onto the live stream's name is rejected before
//!   anything changes; restored under a new name it is a stream of its own.
//! * #104: a stream restored under a new name owns its failure store (its
//!   failure indices renamed like it, ES `RestoreService.updateDataStream`);
//!   failures written to the copy land in the copy's store, and deleting
//!   the copy leaves every normal and failure index and document of the
//!   original — with one and with two failure generations.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

fn settings(cluster: &str, name: &str, seeds: &[u16], repo: &Path) -> Settings {
    let dir = evidence::scratch_dir(&format!("{cluster}-{name}"));
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(name));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("path.repo", json!(repo.to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    if !seeds.is_empty() {
        put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    }
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn spawn_node(s: Settings) -> Arc<ClusterNode> {
    let cfg = NodeConfig::from_settings(s).expect("config");
    let (node, http) = ClusterNode::start(cfg).await.expect("node starts");
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let n = node.clone();
    tokio::spawn(async move {
        let _ = elasticsearch_node::front::serve(n, http).await;
    });
    node
}

struct Cluster {
    _nodes: [Arc<ClusterNode>; 2],
    http: [u16; 2],
    c: reqwest::Client,
    _repo: PathBuf,
}

impl Cluster {
    async fn start(tag: &str) -> Cluster {
        let cluster = ports::unique(tag);
        let repo = evidence::scratch_dir(&format!("{cluster}-repo"));
        let a = spawn_node(settings(&cluster, "n0", &[], &repo)).await;
        let b = spawn_node(settings(&cluster, "n1", &[ports::transport_port(&a)], &repo)).await;
        let http = [ports::http_port(&a), ports::http_port(&b)];
        let c = reqwest::Client::new();
        ports::wait_formed(&c, &http, None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
        let cl = Cluster { _nodes: [a, b], http, c, _repo: repo.clone() };
        cl.ok(0, Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": repo.to_string_lossy()}}))).await;
        cl.ok(0, Method::PUT, "/_index_template/rs-restore-test", Some(json!({
            "index_patterns": ["rs-*"], "priority": 500, "data_stream": {},
            "template": {
                "settings": {"number_of_shards": 1, "number_of_replicas": 1},
                "data_stream_options": {"failure_store": {"enabled": true}},
                "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}}}
            }
        }))).await;
        cl
    }

    async fn call(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.http[node]));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn ok(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(node, method.clone(), path, body).await;
        assert!((200..300).contains(&st), "{method} {path} via n{node}: {st} {v}");
        v
    }

    /// (backing indices, failure indices) of `ds` as node `node` sees it.
    async fn stream(&self, node: usize, ds: &str) -> (Vec<String>, Vec<String>) {
        let v = self.ok(node, Method::GET, &format!("/_data_stream/{ds}"), None).await;
        let d = &v["data_streams"][0];
        let names = |a: &Value| -> Vec<String> { a.as_array().map(|a| a.iter().filter_map(|i| i["index_name"].as_str().map(str::to_string)).collect()).unwrap_or_default() };
        (names(&d["indices"]), names(&d["failure_store"]["indices"]))
    }

    async fn count(&self, node: usize, target: &str) -> u64 {
        let v = self.ok(node, Method::GET, &format!("/{target}/_count"), None).await;
        v["count"].as_u64().unwrap_or_else(|| panic!("{target}: {v}"))
    }

    /// Write a document; `bad` makes it fail mapping (`n` is a long) so it
    /// lands in the failure store. Returns the index it landed in.
    async fn write(&self, node: usize, ds: &str, n: u64, bad: bool) -> String {
        let doc = if bad { json!({"@timestamp": "2026-11-01T00:00:00Z", "n": format!("not-a-number-{n}")}) } else { json!({"@timestamp": "2026-11-01T00:00:00Z", "n": n}) };
        let (st, v) = self.call(node, Method::GET, &format!("/_data_stream/{ds}"), None).await;
        if st == 404 {
            let (st, v) = self.call(node, Method::PUT, &format!("/_data_stream/{ds}"), None).await;
            assert!(st == 200 || v["error"]["type"] == "resource_already_exists_exception", "create {ds}: {st} {v}");
        }
        let v = self.ok(node, Method::POST, &format!("/{ds}/_doc?op_type=create&refresh=true"), Some(doc)).await;
        if bad {
            assert_eq!(v["failure_store"], "used", "{v}");
        }
        v["_index"].as_str().unwrap_or_else(|| panic!("{v}")).to_string()
    }

    async fn snapshot(&self, snap: &str, indices: &str) {
        let v = self.ok(0, Method::PUT, &format!("/_snapshot/repo/{snap}?wait_for_completion=true"), Some(json!({"indices": indices, "include_global_state": false}))).await;
        assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    }

    async fn restore(&self, snap: &str, body: Value) -> (u16, Value) {
        self.call(0, Method::POST, &format!("/_snapshot/repo/{snap}/_restore?wait_for_completion=true"), Some(body)).await
    }

    async fn green(&self, index: &str) {
        let v = self.ok(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(v["status"], "green", "{v}");
    }
}

/// #103.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restoring_an_old_backing_index_never_replaces_the_live_stream() {
    let cl = Cluster::start("ds-restore-backing").await;
    let gen1 = cl.write(0, "rs-live", 1, false).await;
    cl.snapshot("s1", "rs-live").await;
    // The stream rolls over and its first generation expires.
    cl.ok(0, Method::POST, "/rs-live/_rollover", None).await;
    let gen2 = cl.write(1, "rs-live", 2, false).await;
    assert_ne!(gen1, gen2);
    cl.ok(0, Method::DELETE, &format!("/{gen1}"), None).await;
    let live = cl.stream(0, "rs-live").await;
    assert_eq!(live.0, vec![gen2.clone()]);

    // The old generation restored by its own name: a plain index.
    let (st, v) = cl.restore("s1", json!({"indices": gen1, "include_global_state": false})).await;
    assert_eq!(st, 200, "{v}");
    cl.green(&gen1).await;
    for node in 0..2 {
        assert_eq!(cl.stream(node, "rs-live").await, live, "n{node}: the live stream's membership is unchanged");
        assert_eq!(cl.count(node, "rs-live").await, 1, "n{node}: stream reads see only the live generation");
        assert_eq!(cl.count(node, &gen1).await, 1, "n{node}: the restored index holds its document");
    }
    for node in 0..2 {
        assert_eq!(cl.write(node, "rs-live", 10 + node as u64, false).await, gen2, "n{node}: writes still go to the live write index");
    }

    // A whole-stream restore onto the live stream's name is rejected and
    // changes nothing (also with global state).
    cl.ok(0, Method::DELETE, &format!("/{gen1}"), None).await;
    for global in [false, true] {
        let (st, v) = cl.restore("s1", json!({"indices": "rs-live", "include_global_state": global})).await;
        assert!(st >= 400, "global {global}: {st} {v}");
        assert_eq!(v["error"]["type"], "snapshot_restore_exception", "{v}");
        let (_, idx) = cl.call(0, Method::GET, &format!("/{gen1}"), None).await;
        assert_eq!(idx["status"], 404, "global {global}: nothing was restored: {idx}");
        assert_eq!(cl.stream(0, "rs-live").await.0, vec![gen2.clone()]);
    }
    assert_eq!(cl.count(0, "rs-live").await, 3);

    // The intentional whole-stream restore, under a free name: its own
    // stream, the live one untouched.
    let (st, v) = cl.restore("s1", json!({"indices": "rs-live", "include_global_state": false, "rename_pattern": "rs-live", "rename_replacement": "rs-old"})).await;
    assert_eq!(st, 200, "{v}");
    let old = cl.stream(0, "rs-old").await;
    assert_eq!(old.0.len(), 1, "{old:?}");
    assert!(old.0[0].starts_with(".ds-rs-old-"), "{old:?}");
    cl.green(&old.0[0]).await;
    assert_eq!(cl.count(0, "rs-old").await, 1);
    assert_eq!(cl.stream(0, "rs-live").await.0, vec![gen2.clone()]);
    assert_eq!(cl.count(1, "rs-live").await, 3);
}

/// #104, with `generations` failure indices in the original.
async fn renamed_copy_owns_its_failure_store(cl: &Cluster, generations: usize) {
    let (orig, copy) = (format!("rs-fs{generations}"), format!("rs-copy{generations}"));
    cl.write(0, &orig, 1, false).await;
    cl.write(0, &orig, 2, true).await;
    for g in 1..generations {
        cl.ok(0, Method::POST, &format!("/{orig}::failures/_rollover"), None).await;
        cl.write(1, &orig, 2 + g as u64, true).await;
    }
    let (orig_backing, orig_failures) = cl.stream(0, &orig).await;
    assert_eq!(orig_failures.len(), generations, "{orig_failures:?}");
    let snap = format!("fs{generations}");
    cl.snapshot(&snap, &orig).await;

    let (st, v) = cl.restore(&snap, json!({"indices": orig, "include_global_state": false, "rename_pattern": orig, "rename_replacement": copy})).await;
    assert_eq!(st, 200, "{v}");
    let (copy_backing, copy_failures) = cl.stream(0, &copy).await;
    assert_eq!(copy_backing.len(), orig_backing.len(), "{copy_backing:?}");
    assert!(copy_backing.iter().all(|i| i.starts_with(&format!(".ds-{copy}-"))), "{copy_backing:?}");
    assert_eq!(copy_failures.len(), generations, "the copy has its own failure store: {copy_failures:?}");
    for f in &copy_failures {
        assert!(f.starts_with(&format!(".fs-{copy}-")), "renamed like the stream: {copy_failures:?}");
        assert!(!orig_failures.contains(f), "never the original's: {copy_failures:?} vs {orig_failures:?}");
        cl.green(f).await;
    }
    for b in &copy_backing {
        cl.green(b).await;
    }
    assert_eq!(cl.count(0, &format!("{copy}::failures")).await, generations as u64);

    // A failure written to the copy lands in the copy's store.
    let landed = cl.write(1, &copy, 99, true).await;
    assert!(copy_failures.contains(&landed) || landed.starts_with(&format!(".fs-{copy}-")), "{landed}");
    assert!(!orig_failures.contains(&landed), "{landed}");
    assert_eq!(cl.count(0, &format!("{orig}::failures")).await, generations as u64, "the original's store took nothing");

    // Deleting the copy leaves every index and document of the original.
    cl.ok(0, Method::DELETE, &format!("/_data_stream/{copy}"), None).await;
    for node in 0..2 {
        assert_eq!(cl.stream(node, &orig).await, (orig_backing.clone(), orig_failures.clone()), "n{node}");
        for f in &orig_failures {
            assert_eq!(cl.count(node, f).await, 1, "n{node}: failure index {f} and its document survive");
        }
        assert_eq!(cl.count(node, &orig).await, 1, "n{node}");
        assert_eq!(cl.count(node, &format!("{orig}::failures")).await, generations as u64, "n{node}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn renamed_restored_stream_owns_its_failure_store_one_generation() {
    let cl = Cluster::start("ds-restore-fs1").await;
    renamed_copy_owns_its_failure_store(&cl, 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn renamed_restored_stream_owns_its_failure_store_two_generations() {
    let cl = Cluster::start("ds-restore-fs2").await;
    renamed_copy_owns_its_failure_store(&cl, 2).await;
}
