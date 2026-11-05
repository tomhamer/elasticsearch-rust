//! File-based peer recovery of vector shards on a three-node in-process
//! cluster (B4):
//!
//! * a replica recovered from files serves identical kNN hits and scores,
//!   inserted nothing into its graph and re-indexed no document;
//! * the transfer is throttled by `indices.recovery.max_bytes_per_sec`,
//!   re-requests damaged chunks and resumes an interrupted attempt without
//!   receiving any byte twice;
//! * a recovery under concurrent indexing ends with identical doc counts and
//!   seq_nos on both copies;
//! * a relocated vector shard keeps its graph;
//! * the cluster applier keeps applying states while the received copy
//!   opens (off the apply lock);
//! * a failed open of the received copy is retried with the received
//!   files, then the copy is failed and the master reallocates it;
//! * a replica on a node that joined after the mapping grew (its shell
//!   numbers the fields in sorted order) reads the files under the
//!   primary's field ids: identical answers on both copies;
//! * the fault gate's `jepsen-*` search (stats + terms on `_index`) on such
//!   file-recovered copies counts every value and names every index.
//!
//! Small fixtures (a few thousand 16-dim vectors), multi-chunk files through
//! a 16kb `indices.recovery.chunk_size`. Restart reuse (operations-based
//! catch-up of local data) is covered by the replication unit tests: an
//! in-process node cannot be restarted.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use elasticsearch_node::replication::file_recovery as fr;
use elasticsearch_node::shards::Role;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The tests share process-wide recovery seams and counters.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const DIMS: usize = 16;

struct Three {
    nodes: Vec<Arc<ClusterNode>>,
    http: [u16; 3],
    c: reqwest::Client,
}

/// Node `i` on port 0; `seeds`: the transport ports of the nodes started
/// before it.
fn settings(cluster: &str, i: usize, seeds: &[u16]) -> Settings {
    let dir = evidence::scratch_dir(&format!("{cluster}-n{i}"));
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1", "n2"]));
    // Every recovery source saves its graphs first: the shipped checkpoint
    // is exact, so the target's graph is the source's.
    put(fr::GRAPH_LAG_OPS, json!("0"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

/// Start node `i` (seeded by the `transport` ports of nodes before it).
async fn spawn(cluster: &str, i: usize, transport: &[u16]) -> Arc<ClusterNode> {
    let cfg = NodeConfig::from_settings(settings(cluster, i, transport)).expect("config");
    tokio::spawn(async move {
        let (node, listener) = ClusterNode::start(cfg).await.expect("node starts");
        elasticsearch_node::rest::register(&node);
        elasticsearch_node::snapshot_api::register(&node);
        let n = node.clone();
        tokio::spawn(async move {
            let _ = elasticsearch_node::front::serve(n, listener).await;
        });
        node
    }).await.unwrap()
}

/// n0 and n1 only (a quorum of the three initial masters); n2 joins later
/// ([`Three::join_n2`]).
async fn start_two(cluster: &str) -> Three {
    let mut http = [0u16; 3];
    let mut transport: Vec<u16> = Vec::new();
    let mut nodes = Vec::new();
    for i in 0..2 {
        let node = spawn(cluster, i, &transport).await;
        http[i] = ports::http_port(&node);
        transport.push(ports::transport_port(&node));
        nodes.push(node);
    }
    let three = Three { nodes, http, c: reqwest::Client::new() };
    ports::wait_formed(&three.c, &three.http[..2], None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("n0, n1 did not form: {e}"));
    three
}

/// Start three nodes on OS-assigned ports, one after another: each seeds
/// from the transport ports of those before it.
async fn start(cluster: &str) -> Three {
    let mut http = [0u16; 3];
    let mut transport: Vec<u16> = Vec::new();
    let mut nodes = Vec::new();
    for i in 0..3 {
        let node = spawn(cluster, i, &transport).await;
        http[i] = ports::http_port(&node);
        transport.push(ports::transport_port(&node));
        nodes.push(node);
    }
    let three = Three { nodes, http, c: reqwest::Client::new() };
    // Formed and recovered on every node (as replica_search.rs): health is
    // red while the "state not recovered" global block is up, and settings
    // updates and index creation are refused with that block.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut ready = 0;
    loop {
        let (_, h) = three.call(ready, Method::GET, "/_cluster/health?wait_for_nodes=3&wait_for_status=yellow&timeout=1s", None).await;
        if h["number_of_nodes"] == 3 && h["timed_out"] == false && (h["status"] == "green" || h["status"] == "yellow") {
            ready += 1;
            if ready == 3 {
                return three;
            }
            continue;
        }
        assert!(Instant::now() < deadline, "cluster did not form and recover on n{ready}: {h}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

impl Three {
    /// Start n2 (after [`start_two`]) and wait until all three formed.
    async fn join_n2(&mut self) {
        let seeds: Vec<u16> = self.nodes.iter().map(|n| ports::transport_port(n)).collect();
        let node = spawn(&self.nodes[0].config.cluster_name, 2, &seeds).await;
        self.http[2] = ports::http_port(&node);
        self.nodes.push(node);
        ports::wait_formed(&self.c, &self.http, None, 3, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("n2 did not join: {e}"));
    }

    async fn call(&self, i: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.http[i]));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn ok(&self, i: usize, method: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(i, method.clone(), path, body).await;
        assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
        v
    }

    async fn cluster_settings(&self, persistent: Value) {
        self.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": persistent}))).await;
    }

    /// Green with `copies` active copies of the (one-shard) index, as
    /// every node sees it. Health, `_recovery` and `_stats` answer from the
    /// answering node's applied state: asked on one node right after a
    /// settings update, they can describe the state before it (no replica
    /// yet, so "green") while the other nodes have moved on.
    async fn green(&self, index: &str, copies: u32) {
        for i in 0..3 {
            let v = self.ok(i, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&wait_for_active_shards={copies}&wait_for_no_relocating_shards=true&wait_for_no_initializing_shards=true&timeout=120s"), None).await;
            assert!(v["status"] == "green" && v["active_shards"] == copies && v["timed_out"] == false, "n{i}: {v}");
        }
    }

    /// A one-shard vector index whose copies may live only on `nodes`.
    async fn vector_index(&self, index: &str, nodes: &str) {
        self.ok(0, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.include._name": nodes},
            "mappings": {"properties": {"emb": {"type": "dense_vector", "dims": DIMS, "similarity": "l2_norm", "index_options": {"type": "hnsw", "m": 8, "ef_construction": 50}},
                "tag": {"type": "keyword"}, "n": {"type": "integer"}}}
        }))).await;
        self.green(index, 1).await;
    }

    async fn bulk(&self, index: &str, ids: std::ops::Range<usize>) {
        let mut nd = String::new();
        for i in ids {
            nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("d{i}")}}), json!({"emb": vector(i), "tag": format!("t{}", i % 7), "n": i})));
        }
        let r = self.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", self.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }

    /// kNN hits (id, score) answered by the copy on node `n{node}`.
    async fn knn(&self, index: &str, node: usize, q: usize) -> Vec<(String, f64)> {
        let v = self.ok(0, Method::POST, &format!("/{index}/_search?preference=_only_nodes:n{node}"), Some(json!({
            "knn": {"field": "emb", "query_vector": vector(1_000_000 + q), "k": 10, "num_candidates": 50}, "_source": false, "size": 10
        }))).await;
        assert_eq!(v["_shards"]["failed"], 0, "{v}");
        v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
    }

    /// The `_recovery` entry of the copy on node `n{node}`.
    async fn recovery_on(&self, index: &str, node: usize) -> Value {
        let v = self.ok(0, Method::GET, &format!("/{index}/_recovery?detailed=true"), None).await;
        v[index]["shards"].as_array().unwrap().iter().find(|s| s["target"]["name"] == format!("n{node}")).cloned().unwrap_or_else(|| panic!("no recovery on n{node}: {v}"))
    }

    /// The node position holding the copy of shard 0 of `index`, by role.
    fn copy_node(&self, index: &str, primary: bool) -> usize {
        self.nodes.iter().position(|n| n.shards.get(index, 0).is_some_and(|s| matches!(s.lock().role, Role::Primary(_)) == primary)).unwrap_or_else(|| panic!("no {} copy of {index}", if primary { "primary" } else { "replica" }))
    }

    /// Graph insertions of the copy on node `i` since it opened.
    fn graph_inserts(&self, i: usize, index: &str) -> u64 {
        let slot = self.nodes[i].shards.get(index, 0).expect("copy");
        let g = slot.lock();
        let copy = match &g.role {
            Role::Primary(p) => p.copy(),
            Role::Replica(r) => r.copy(),
        };
        let field = g.svc.lookup().get("emb").and_then(|f| f.core_id).expect("vector field");
        let inserts = copy.shard.core.read().vector_store().mutable_graph_inserts(field);
        inserts.expect("a mutable graph")
    }

    /// `_stats?level=shards` of the copy on node `i` (by node id).
    async fn shard_stats(&self, index: &str, i: usize) -> Value {
        let v = self.ok(0, Method::GET, &format!("/{index}/_stats?level=shards"), None).await;
        let id = self.nodes[i].local.id.clone();
        v["indices"][index]["shards"]["0"].as_array().unwrap().iter().find(|c| c["routing"]["node"] == id.as_str()).cloned().unwrap_or_else(|| panic!("no copy on n{i}: {v}"))
    }
}

fn vector(i: usize) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 2000) as f32 / 1000.0 - 1.0
    }).collect()
}

/// The replica recovered from the primary's files: same graph (no insert),
/// no document re-indexed, identical kNN hits and scores; files arrive in
/// many chunks and `_recovery` reports them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn vector_replica_recovers_from_files_with_identical_knn_and_no_reindex() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-vector").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    three.vector_index("vecs", "n0").await;
    three.bulk("vecs", 0..3000).await;
    three.ok(0, Method::POST, "/vecs/_flush", None).await;
    let baseline: Vec<_> = futures::future::join_all((0..5).map(|q| three.knn("vecs", 0, q))).await;
    let received = elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed);
    three.ok(0, Method::PUT, "/vecs/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
    three.green("vecs", 2).await;
    let r = three.copy_node("vecs", false);
    assert_eq!(r, 1);
    assert_eq!(three.graph_inserts(r, "vecs"), 0, "the shipped graph checkpoint is complete: nothing re-inserted");
    let st = three.shard_stats("vecs", r).await;
    assert_eq!(st["indexing"]["index_total"], 0, "no document re-indexed: {st}");
    assert_eq!(st["docs"]["count"], 3000, "{st}");
    let rec = three.recovery_on("vecs", r).await;
    assert_eq!((rec["type"].as_str(), rec["stage"].as_str()), (Some("PEER"), Some("DONE")), "{rec}");
    let files = &rec["index"]["files"];
    assert!(files["total"].as_u64().unwrap() >= 3, "{rec}");
    assert_eq!(files["recovered"], files["total"], "{rec}");
    assert_eq!(files["reused"], 0, "a new copy has nothing to reuse: {rec}");
    assert_eq!(files["percent"], "100.0%", "{rec}");
    let bytes = rec["index"]["size"]["recovered_in_bytes"].as_u64().unwrap();
    assert_eq!(bytes, rec["index"]["size"]["total_in_bytes"].as_u64().unwrap());
    assert!(bytes > 10 * 16 * 1024, "multi-chunk files: {bytes} bytes");
    assert_eq!(elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed) - received, bytes, "every byte received once");
    assert!(files["details"].as_array().is_some_and(|d| d.iter().any(|f| f["name"].as_str().is_some_and(|n| n.ends_with(".rmvc")))), "the graph checkpoint is shipped: {rec}");
    assert_eq!(rec["translog"]["recovered"], 0, "nothing written since the commit: {rec}");
    for (q, expected) in baseline.iter().enumerate() {
        assert_eq!(&three.knn("vecs", r, q).await, expected, "query {q}: replica hits and scores");
    }
    // Writes after the recovery replicate as usual, and a search over the
    // tag filter agrees on both copies.
    three.bulk("vecs", 3000..3100).await;
    for node in [0, r] {
        let v = three.ok(0, Method::POST, &format!("/vecs/_count?preference=_only_nodes:n{node}"), Some(json!({"query": {"term": {"tag": "t3"}}}))).await;
        assert_eq!(v["count"], (0..3100).filter(|i| i % 7 == 3).count(), "n{node}: {v}");
    }
}

/// Throttle, damaged chunks and an interrupted transfer: the copy still
/// recovers, takes at least the throttled time, re-requests each damaged
/// chunk, and the retry resumes where the first attempt stopped (no byte is
/// received twice).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn throttled_recovery_survives_damaged_chunks_and_resumes_after_an_interruption() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-throttle").await;
    three.vector_index("slow", "n0").await;
    three.bulk("slow", 0..3000).await;
    three.ok(0, Method::POST, "/slow/_flush", None).await;
    let baseline = three.knn("slow", 0, 1).await;
    // Validated as ES does.
    let (st, e) = three.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.recovery.max_bytes_per_sec": "fast"}}))).await;
    assert_eq!(st, 400, "{e}");
    three.cluster_settings(json!({"indices.recovery.max_bytes_per_sec": "1mb", "indices.recovery.chunk_size": "16kb", "indices.recovery.max_concurrent_file_chunks": 2})).await;
    let received = elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed);
    fr::CORRUPT_CHUNKS.store(3, Ordering::SeqCst);
    fr::INTERRUPT_AFTER_CHUNKS.store(20, Ordering::SeqCst);
    let started = Instant::now();
    three.ok(0, Method::PUT, "/slow/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n2"}}))).await;
    three.green("slow", 2).await;
    let took = started.elapsed();
    assert_eq!(fr::CORRUPT_CHUNKS.load(Ordering::SeqCst), 0, "the damaged chunks were received");
    assert_eq!(fr::INTERRUPT_AFTER_CHUNKS.load(Ordering::SeqCst), -1, "the interruption happened");
    let r = three.copy_node("slow", false);
    let rec = three.recovery_on("slow", r).await;
    let size = &rec["index"]["size"];
    let to_fetch = size["total_in_bytes"].as_u64().unwrap() - size["reused_in_bytes"].as_u64().unwrap();
    assert_eq!(elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed) - received, to_fetch, "resumed, not restarted: {rec}");
    // 1mb/s on each node: the files alone take that long.
    let floor = Duration::from_secs_f64(to_fetch as f64 / (1 << 20) as f64 * 0.8);
    assert!(took >= floor, "{took:?} < {floor:?} for {to_fetch} bytes");
    let throttled = rec["index"]["source_throttle_time_in_millis"].as_u64().unwrap() + rec["index"]["target_throttle_time_in_millis"].as_u64().unwrap();
    assert!(throttled > 0, "{rec}");
    assert_eq!(three.knn("slow", r, 1).await, baseline);
    assert_eq!(three.graph_inserts(r, "slow"), 0);
    three.cluster_settings(json!({"indices.recovery.max_bytes_per_sec": null, "indices.recovery.chunk_size": null, "indices.recovery.max_concurrent_file_chunks": null})).await;
}

/// Writes (index, overwrite, delete) continue while the replica recovers
/// from files and replays the translog above the shipped commit: both
/// copies end with the same documents, max seq_no and local checkpoint.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_under_concurrent_indexing_ends_with_identical_copies() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-concurrent").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb", "indices.recovery.max_bytes_per_sec": "4mb"})).await;
    three.vector_index("busy", "n1").await;
    three.bulk("busy", 0..2000).await;
    three.ok(0, Method::POST, "/busy/_flush", None).await;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let (c, http, stop) = (three.c.clone(), three.http[0], stop.clone());
        tokio::spawn(async move {
            let mut round = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let mut nd = String::new();
                for k in 0..50 {
                    let i = 2000 + round * 50 + k;
                    nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "busy", "_id": format!("d{i}")}}), json!({"emb": vector(i), "tag": "w", "n": i})));
                    // Overwrite an old document and delete another.
                    let old = (round * 37 + k * 11) % 2000;
                    nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "busy", "_id": format!("d{old}")}}), json!({"emb": vector(old + 7), "tag": "u", "n": old})));
                    nd.push_str(&format!("{}\n", json!({"delete": {"_index": "busy", "_id": format!("d{}", (old + 1000) % 2000)}})));
                }
                let r = c.post(format!("http://127.0.0.1:{http}/_bulk")).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
                let v: Value = r.json().await.unwrap();
                assert_eq!(v["errors"], false, "{v}");
                round += 1;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            round
        })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    three.ok(0, Method::PUT, "/busy/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n1,n2"}}))).await;
    three.green("busy", 2).await;
    // Keep writing a little past the hand-off, then stop.
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.store(true, Ordering::Relaxed);
    let rounds = writer.await.unwrap();
    assert!(rounds > 2, "writes ran during the recovery");
    three.ok(0, Method::POST, "/busy/_refresh", None).await;
    let (p, r) = (three.copy_node("busy", true), three.copy_node("busy", false));
    let (ps, rs) = (three.shard_stats("busy", p).await, three.shard_stats("busy", r).await);
    for key in ["max_seq_no", "local_checkpoint"] {
        assert_eq!(ps["seq_no"][key], rs["seq_no"][key], "{key}: primary {ps} replica {rs}");
    }
    assert_eq!(ps["docs"]["count"], rs["docs"]["count"], "primary {ps} replica {rs}");
    let rec = three.recovery_on("busy", r).await;
    assert!(rec["index"]["files"]["total"].as_u64().unwrap() > 0, "{rec}");
    for q in [json!({"match_all": {}}), json!({"term": {"tag": "u"}}), json!({"range": {"n": {"gte": 1500}}})] {
        let mut counts = Vec::new();
        for node in [p, r] {
            let v = three.ok(0, Method::POST, &format!("/busy/_count?preference=_only_nodes:n{node}"), Some(json!({"query": q}))).await;
            counts.push(v["count"].as_u64().unwrap());
        }
        assert_eq!(counts[0], counts[1], "{q}");
    }
    three.cluster_settings(json!({"indices.recovery.chunk_size": null, "indices.recovery.max_bytes_per_sec": null})).await;
}

/// A vector primary relocates by the same file path: the new copy keeps the
/// graph (nothing re-inserted) and answers identically.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vector_shard_relocates_by_copying_its_files() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-relocate").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    three.vector_index("moving", "n0").await;
    three.bulk("moving", 0..2500).await;
    three.ok(0, Method::POST, "/moving/_flush", None).await;
    let baseline: Vec<_> = futures::future::join_all((0..3).map(|q| three.knn("moving", 0, q))).await;
    three.ok(0, Method::PUT, "/moving/_settings", Some(json!({"index.routing.allocation.include._name": "n2"}))).await;
    three.green("moving", 1).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    while three.nodes[2].shards.get("moving", 0).is_none() || three.nodes[0].shards.get("moving", 0).is_some() {
        assert!(Instant::now() < deadline, "the primary did not move to n2");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Every node applied the relocation's end (stats answer from its state).
    three.green("moving", 1).await;
    assert_eq!(three.copy_node("moving", true), 2);
    assert_eq!(three.graph_inserts(2, "moving"), 0, "the relocated graph is the source's");
    let st = three.shard_stats("moving", 2).await;
    assert_eq!(st["docs"]["count"], 2500, "{st}");
    let rec = three.recovery_on("moving", 2).await;
    assert!(rec["index"]["files"]["recovered"].as_u64().unwrap() > 0, "{rec}");
    for (q, expected) in baseline.iter().enumerate() {
        assert_eq!(&three.knn("moving", 2, q).await, expected, "query {q}");
    }
    // The relocated primary takes writes.
    three.bulk("moving", 2500..2510).await;
    let v = three.ok(0, Method::GET, "/moving/_count", None).await;
    assert_eq!(v["count"], 2510);
}

/// The node's allocation id of copy 0 of `index` on node `i` once it
/// holds one (polled: the recovery inserts it).
async fn local_allocation(three: &Three, i: usize, index: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(slot) = three.nodes[i].shards.get(index, 0) {
            return slot.lock().allocation_id.clone();
        }
        assert!(Instant::now() < deadline, "no copy of {index} on n{i}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The received copy opens (graph checkpoint reconciliation, possibly a
/// long catch-up) without the node's apply lock: while the open is paused,
/// the target node applies a new cluster state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_applier_runs_while_the_recovered_copy_opens() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-open").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    three.vector_index("opening", "n0").await;
    three.bulk("opening", 0..1000).await;
    three.ok(0, Method::POST, "/opening/_flush", None).await;
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    *fr::INSTALL_OPEN_HOOK.lock() = Some(Box::new(move || {
        let _ = entered_tx.send(());
        let _ = release_rx.recv_timeout(Duration::from_secs(120));
    }));
    three.ok(0, Method::PUT, "/opening/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
    let entered = tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(60))).await.unwrap();
    assert!(entered.is_ok(), "the received copy never started opening");
    // A cluster state change while the open is paused: n1 applies it.
    let changed = {
        let c = three.c.clone();
        let http = three.http[0];
        tokio::spawn(async move {
            c.put(format!("http://127.0.0.1:{http}/other")).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.include._name": "n0"}})).send().await.map(|r| r.status().as_u16())
        })
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    while !three.nodes[1].state().metadata.indices.contains_key("other") {
        assert!(Instant::now() < deadline, "n1 applied no cluster state while the received copy opened");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _ = release_tx.send(());
    assert_eq!(changed.await.unwrap().ok(), Some(200));
    three.green("opening", 2).await;
    *fr::INSTALL_OPEN_HOOK.lock() = None;
    let r = three.copy_node("opening", false);
    assert_eq!(r, 1);
    assert_eq!(three.knn("opening", 1, 0).await, three.knn("opening", 0, 0).await);
}

/// A failed open of the received copy is retried in place with the files
/// already received (nothing fetched again, same allocation); when every
/// attempt fails the copy is failed, the master allocates a new one, and
/// that one recovers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_install_is_retried_then_reported() {
    let _serial = SERIAL.lock().await;
    let three = start("fr-install").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    three.vector_index("inst", "n0").await;
    three.bulk("inst", 0..1500).await;
    three.ok(0, Method::POST, "/inst/_flush", None).await;
    let baseline = three.knn("inst", 0, 2).await;
    // One failed open: retried in place.
    let received = elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed);
    fr::FAIL_INSTALLS.store(1, Ordering::SeqCst);
    three.ok(0, Method::PUT, "/inst/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
    let first = local_allocation(&three, 1, "inst").await;
    three.green("inst", 2).await;
    assert_eq!(fr::FAIL_INSTALLS.load(Ordering::SeqCst), 0, "the open failed once");
    assert_eq!(local_allocation(&three, 1, "inst").await, first, "retried in place: not reallocated");
    let rec = three.recovery_on("inst", 1).await;
    let size = &rec["index"]["size"];
    let to_fetch = size["total_in_bytes"].as_u64().unwrap() - size["reused_in_bytes"].as_u64().unwrap();
    assert_eq!(elasticsearch_node::node::FILE_CHUNK_BYTES_RECEIVED.load(Ordering::Relaxed) - received, to_fetch, "the retry opened the files already received: {rec}");
    assert_eq!(three.knn("inst", 1, 2).await, baseline);
    // Every attempt fails (a fresh index: a new copy, nothing local to
    // reuse): the copy is reported failed and reallocated.
    three.vector_index("inst2", "n0").await;
    three.bulk("inst2", 0..1500).await;
    three.ok(0, Method::POST, "/inst2/_flush", None).await;
    fr::FAIL_INSTALLS.store(fr::INSTALL_ATTEMPTS, Ordering::SeqCst);
    three.ok(0, Method::PUT, "/inst2/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
    let failing = local_allocation(&three, 1, "inst2").await;
    three.green("inst2", 2).await;
    assert_eq!(fr::FAIL_INSTALLS.load(Ordering::SeqCst), 0, "every attempt of the first allocation failed");
    let recovered = local_allocation(&three, 1, "inst2").await;
    assert_ne!(recovered, failing, "the failed copy was reported and the master allocated a new one");
    assert_eq!(three.knn("inst2", 1, 2).await, three.knn("inst2", 0, 2).await);
}

/// The core field ids the copy of shard 0 on node `i` reads its files with.
fn core_ids(three: &Three, i: usize, index: &str) -> std::collections::BTreeMap<String, u16> {
    let slot = three.nodes[i].shards.get(index, 0).unwrap_or_else(|| panic!("no copy of {index} on n{i}"));
    let ids = slot.lock().svc.meta().mappings.core_ids;
    ids.into_iter().filter(|(_, id)| *id >= elasticsearch_engine::mapping::FIRST_USER_FIELD).collect()
}

/// Field ids are assigned as fields first appear. The primary's node saw
/// the mapping grow one change at a time (`z_num`, then `y_tag`, `b_emb`,
/// `a_text`); a node that joins afterwards builds its index shell from the
/// final mapping, in sorted order. Its replica receives the primary's files
/// and must read them under the primary's ids (the shell takes them before
/// opening the files): every query answers the same on both copies, and
/// writes after the recovery replicate as usual.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replica_on_a_later_node_reads_the_files_under_the_primarys_field_ids() {
    let _serial = SERIAL.lock().await;
    let mut three = start_two("fr-field-ids").await;
    three.ok(0, Method::PUT, "/grown", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.include._name": "n0"},
        "mappings": {"properties": {"z_num": {"type": "integer"}}}
    }))).await;
    three.ok(0, Method::GET, "/_cluster/health/grown?wait_for_status=green&timeout=60s", None).await;
    let (client, port0) = (three.c.clone(), three.http[0]);
    let bulk = |docs: Vec<(usize, Value)>| {
        let mut nd = String::new();
        for (i, d) in docs {
            nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "grown", "_id": format!("d{i}")}}), d));
        }
        nd
    };
    let send = |nd: String| {
        let (c, port) = (client.clone(), port0);
        async move {
            let v: Value = c.post(format!("http://127.0.0.1:{port}/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap().json().await.unwrap();
            assert_eq!(v["errors"], false, "{v}");
        }
    };
    // One mapping change per step, in reverse name order.
    send(bulk((0..400).map(|i| (i, json!({"z_num": i}))).collect())).await;
    send(bulk((0..400).map(|i| (i, json!({"z_num": i, "y_tag": format!("t{}", i % 5)}))).collect())).await;
    three.ok(0, Method::PUT, "/grown/_mapping", Some(json!({"properties": {"b_emb": {"type": "dense_vector", "dims": DIMS, "similarity": "l2_norm", "index_options": {"type": "hnsw", "m": 8, "ef_construction": 50}}}}))).await;
    send(bulk((0..400).map(|i| (i, json!({"z_num": i, "y_tag": format!("t{}", i % 5), "b_emb": vector(i), "a_text": format!("alpha w{}", i % 3)}))).collect())).await;
    three.ok(0, Method::POST, "/grown/_flush", None).await;
    let primary_ids = core_ids(&three, 0, "grown");
    assert!(primary_ids["z_num"] < primary_ids["y_tag"] && primary_ids["y_tag"] < primary_ids["b_emb"] && primary_ids["b_emb"] < primary_ids["a_text"], "numbered as the mapping grew: {primary_ids:?}");
    let queries = [
        json!({"query": {"term": {"z_num": 77}}}),
        json!({"query": {"range": {"z_num": {"gte": 390}}}, "sort": [{"z_num": "desc"}], "size": 20}),
        json!({"query": {"match": {"y_tag": "t3"}}, "size": 0, "track_total_hits": true}),
        json!({"query": {"match": {"a_text": "w1"}}, "size": 5}),
        json!({"size": 0, "aggs": {"tags": {"terms": {"field": "y_tag.keyword"}}, "words": {"terms": {"field": "a_text.keyword"}}, "max": {"max": {"field": "z_num"}}}}),
        json!({"knn": {"field": "b_emb", "query_vector": vector(1_000_123), "k": 10, "num_candidates": 50}, "_source": false, "size": 10}),
    ];
    let on = |node: usize, body: Value| {
        let (c, port) = (client.clone(), port0);
        async move {
            let v: Value = c.post(format!("http://127.0.0.1:{port}/grown/_search?preference=_only_nodes:n{node}")).json(&body).send().await.unwrap().json().await.unwrap();
            assert_eq!(v["_shards"]["failed"], 0, "{v}");
            json!({"hits": v["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_score"], h["sort"]])).collect::<Vec<_>>(),
                "total": v["hits"]["total"], "aggs": v["aggregations"]})
        }
    };
    let mut baseline = Vec::new();
    for q in &queries {
        baseline.push(on(0, q.clone()).await);
    }
    // n2 joins now: its shell numbers the fields in sorted order.
    three.join_n2().await;
    three.ok(0, Method::PUT, "/grown/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n2"}}))).await;
    three.green("grown", 2).await;
    assert_eq!(three.copy_node("grown", false), 2);
    let rec = three.recovery_on("grown", 2).await;
    assert_eq!(rec["type"], "PEER", "{rec}");
    assert!(rec["index"]["files"]["total"].as_u64().unwrap() > 0, "recovered from files: {rec}");
    let st = three.shard_stats("grown", 2).await;
    assert_eq!(st["indexing"]["index_total"], 0, "no document re-indexed: {st}");
    let replica_ids = core_ids(&three, 2, "grown");
    for (field, id) in &primary_ids {
        assert_eq!(replica_ids.get(field), Some(id), "n2 reads [{field}] under the primary's id: {replica_ids:?}");
    }
    for (q, want) in queries.iter().zip(&baseline) {
        assert_eq!(&on(2, q.clone()).await, want, "replica answers {q}");
    }
    // Writes after the recovery (with a field new to both) replicate.
    send(bulk((400..450).map(|i| (i, json!({"z_num": i, "y_tag": "t9", "c_new": i}))).collect())).await;
    for node in [0, 2] {
        let v = on(node, json!({"query": {"bool": {"filter": [{"term": {"y_tag.keyword": "t9"}}, {"range": {"c_new": {"gte": 425}}}]}}, "size": 0, "track_total_hits": true})).await;
        assert_eq!(v["total"]["value"], 25, "n{node}: {v}");
    }
}

/// The fault gate's `kill-during-recovery` failure (candidates 28x and
/// earlier): `jepsen-a` / `jepsen-b` are created with `k`, `v` mapped, then
/// grow `pad` dynamically, so their nodes number `v` before `pad`; a node
/// whose data was wiped rebuilds both shells from the final mapping (sorted:
/// `pad` before `v`) and recovers its copies from the primaries' files. Read
/// under its own ids, those files had no `v` values: a complete
/// `jepsen-*` search answered `hits.total` N with `stats(v).count` about N/2
/// (one index's copy) or 0 (both), and the gate's `by_index` diagnostic
/// (terms on `_index`) answered no buckets at all. On the recovered copies,
/// as on the primaries, the gate's exact search must count every value and
/// name each index with its own documents.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jepsen_wildcard_search_on_file_recovered_copies_counts_every_value() {
    let _serial = SERIAL.lock().await;
    let mut three = start_two("fr-jepsen-search").await;
    const PER_INDEX: usize = 600;
    let (client, port0) = (three.c.clone(), three.http[0]);
    for ix in ["jepsen-a", "jepsen-b"] {
        three.ok(0, Method::PUT, &format!("/{ix}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.include._name": "n0"},
            "mappings": {"properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}}
        }))).await;
    }
    // n2 has not joined: health on n0 only (`green` asks every node).
    three.ok(0, Method::GET, "/_cluster/health/jepsen-a,jepsen-b?wait_for_status=green&timeout=60s", None).await;
    // Registers (v = 0, as the harness's init), then the preload with the
    // dynamic `pad` (v = j), alternating indices as the harness does.
    let mut nd = String::new();
    for ix in ["jepsen-a", "jepsen-b"] {
        for r in 0..5 {
            nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": ix, "_id": format!("r{r}")}}), json!({"k": format!("r{r}"), "v": 0})));
        }
    }
    for j in 0..2 * (PER_INDEX - 5) {
        let ix = if j % 2 == 0 { "jepsen-a" } else { "jepsen-b" };
        nd.push_str(&format!("{}\n{}\n", json!({"create": {"_index": ix, "_id": format!("p{j}")}}), json!({"k": format!("p{j}"), "v": j, "pad": "x".repeat(200)})));
    }
    let v: Value = client.post(format!("http://127.0.0.1:{port0}/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
    three.ok(0, Method::POST, "/jepsen-a,jepsen-b/_flush", None).await;
    let primary_ids = core_ids(&three, 0, "jepsen-a");
    assert!(primary_ids["v"] < primary_ids["pad"], "v numbered before the dynamic pad: {primary_ids:?}");
    // n2 joins with neither index: its shells number `pad` before `v`. Its
    // copies come from the primaries' files.
    three.join_n2().await;
    for ix in ["jepsen-a", "jepsen-b"] {
        three.ok(0, Method::PUT, &format!("/{ix}/_settings"), Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n2"}}))).await;
    }
    for ix in ["jepsen-a", "jepsen-b"] {
        three.green(ix, 2).await;
        assert_eq!(three.copy_node(ix, false), 2, "{ix}'s replica on n2");
        let rec = three.recovery_on(ix, 2).await;
        assert!(rec["index"]["files"]["total"].as_u64().unwrap() > 0, "{ix} recovered from files: {rec}");
    }
    // The gate's search body, exactly.
    let body = json!({"size": 0, "track_total_hits": true, "aggs": {
        "values": {"stats": {"field": "v"}},
        "by_index": {"terms": {"field": "_index", "size": 10}, "aggs": {"vc": {"value_count": {"field": "v"}}}}
    }});
    let total = (2 * PER_INDEX) as u64;
    for coordinator in 0..3 {
        for query in ["allow_partial_search_results=false", "allow_partial_search_results=false&request_cache=false",
            "allow_partial_search_results=false&preference=_only_nodes:n2", "allow_partial_search_results=false&preference=_only_nodes:n0"] {
            let (status, got) = three.call(coordinator, Method::POST, &format!("/jepsen-*/_search?{query}"), Some(body.clone())).await;
            let what = format!("n{coordinator} ?{query}: {got}");
            assert_eq!(status, 200, "{what}");
            assert_eq!(got["_shards"]["total"], 2, "{what}");
            assert_eq!(got["_shards"]["failed"], 0, "{what}");
            assert_eq!(got["hits"]["total"]["value"], total, "{what}");
            let a = &got["aggregations"];
            assert_eq!(a["values"]["count"], total, "stats(v).count == hits.total: {what}");
            assert_eq!(a["values"]["min"].as_f64(), Some(0.0), "the registers' v = 0 counted: {what}");
            assert_eq!(a["values"]["max"].as_f64(), Some((2 * (PER_INDEX - 5) - 1) as f64), "{what}");
            let mut by_index: Vec<(String, u64, u64)> = a["by_index"]["buckets"].as_array().unwrap().iter()
                .map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap(), b["vc"]["value"].as_u64().unwrap())).collect();
            by_index.sort();
            let n = PER_INDEX as u64;
            assert_eq!(by_index, vec![("jepsen-a".to_string(), n, n), ("jepsen-b".to_string(), n, n)], "by_index: {what}");
        }
    }
    let replica_ids = core_ids(&three, 2, "jepsen-a");
    assert_eq!(replica_ids.get("v"), primary_ids.get("v"), "n2 reads jepsen-a's files under the primary's ids: {replica_ids:?}");
}

/// Panics anywhere in this process from here on. A release node aborts on
/// any panic (`panic = "abort"`); in a test a panic on a blocking stats
/// worker only turns into an empty answer, so it is recorded here.
fn panics_seen() -> &'static std::sync::Mutex<Vec<String>> {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> = std::sync::OnceLock::new();
    SEEN.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(seen) = SEEN.get() {
                seen.lock().unwrap_or_else(|e| e.into_inner()).push(info.to_string());
            }
            previous(info)
        }));
        std::sync::Mutex::new(Vec::new())
    })
}

/// The stats readers a YAML suite polls while shards recover, on node `i`:
/// each must answer (200) whatever state the node's copies are in.
async fn poll_local_stats(c: &reqwest::Client, port: u16, index: &str) -> Vec<(String, u16)> {
    let mut out = Vec::new();
    for path in ["/_cat/shards?format=json".to_string(), "/_cat/recovery?format=json".to_string(), "/_cat/segments?format=json".to_string(),
        "/_stats?level=shards".to_string(), format!("/{index}/_recovery?detailed=true")] {
        let status = c.get(format!("http://127.0.0.1:{port}{path}")).send().await.map(|r| r.status().as_u16()).unwrap_or(0);
        out.push((path, status));
    }
    out
}

/// candidate 32o (2-node ES YAML run): a node aborted with "shard
/// [index2][0] is not open in this engine" right after "recovery files
/// received". Installing the received files retires the placeholder copy
/// (under the apply lock), opens the received copy off the lock, then
/// publishes it: in between, the allocation is registered on the node but
/// its engine holds no copy, and shard stats (`_cat/shards`,
/// `_cat/recovery`, `_stats`, `_cat/segments`) took that copy with a
/// panicking accessor. Stats now report such a copy without a store, every
/// other taker fails its operation, and an index deleted in that window
/// fails the recovery (stale) while the node goes on: the same name
/// re-created recovers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_and_a_delete_while_the_received_copy_opens_fail_nothing() {
    let _serial = SERIAL.lock().await;
    let seen = panics_seen();
    seen.lock().unwrap().clear();
    let three = start("fr-race").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    for round in 0..2 {
        three.vector_index("index2", "n0").await;
        three.bulk("index2", 0..800).await;
        three.ok(0, Method::POST, "/index2/_flush", None).await;
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        *fr::INSTALL_OPEN_HOOK.lock() = Some(Box::new(move || {
            let _ = entered_tx.send(());
            let _ = release_rx.recv_timeout(Duration::from_secs(120));
        }));
        three.ok(0, Method::PUT, "/index2/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
        let entered = tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(60))).await.unwrap();
        assert!(entered.is_ok(), "round {round}: the received copy never started opening");
        assert!(three.nodes[1].shards.get("index2", 0).is_some(), "the recovering copy is registered on n1");
        for i in 0..3 {
            for (path, status) in poll_local_stats(&three.c, three.http[i], "index2").await {
                assert_eq!(status, 200, "round {round}: n{i} {path} while the received copy opens");
            }
        }
        let panics = seen.lock().unwrap().clone();
        assert!(panics.is_empty(), "round {round}: panics while the received copy opens: {:?}", panics);
        if round == 0 {
            // Released: the copy installs and starts.
            let _ = release_tx.send(());
            three.green("index2", 2).await;
            assert_eq!(three.knn("index2", 1, 0).await, three.knn("index2", 0, 0).await);
        } else {
            // Deleted while it opens: the recovery is stale and fails, and
            // the name is free for a new index that recovers.
            let deleted = {
                let (c, http) = (three.c.clone(), three.http[0]);
                tokio::spawn(async move { c.delete(format!("http://127.0.0.1:{http}/index2")).send().await.map(|r| r.status().as_u16()) })
            };
            let deadline = Instant::now() + Duration::from_secs(30);
            while three.nodes[1].state().metadata.indices.contains_key("index2") {
                assert!(Instant::now() < deadline, "n1 never applied the delete while the received copy opened");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            for (path, status) in poll_local_stats(&three.c, three.http[1], "_all").await {
                assert_eq!(status, 200, "n1 {path} after the delete, the copy still opening");
            }
            let _ = release_tx.send(());
            assert_eq!(deleted.await.unwrap().ok(), Some(200));
            let deadline = Instant::now() + Duration::from_secs(30);
            while three.nodes[1].shards.get("index2", 0).is_some() {
                assert!(Instant::now() < deadline, "the deleted index's copy stayed registered on n1");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            three.vector_index("index2", "n0").await;
            three.bulk("index2", 0..300).await;
            three.ok(0, Method::POST, "/index2/_flush", None).await;
            three.ok(0, Method::PUT, "/index2/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
            three.green("index2", 2).await;
            assert_eq!(three.knn("index2", 1, 0).await, three.knn("index2", 0, 0).await);
        }
        *fr::INSTALL_OPEN_HOOK.lock() = None;
        if round == 0 {
            three.ok(0, Method::DELETE, "/index2", None).await;
        }
    }
    let panics = seen.lock().unwrap().clone();
    assert!(panics.is_empty(), "panics: {:?}", panics);
}

/// Stress for the same race without a seam: an index created, its replica
/// recovered from files onto a second node, then deleted (immediately, after
/// a short random delay, or once green), and re-created under the same name,
/// while every node's shard stats are polled without pause.
/// `RECOVERY_RACE_ITERS` iterations (default 20; the repro ran several
/// hundred).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_recover_delete_loop_under_stats_polling() {
    use rand::Rng;
    let _serial = SERIAL.lock().await;
    let seen = panics_seen();
    seen.lock().unwrap().clear();
    let three = start("fr-race-loop").await;
    three.cluster_settings(json!({"indices.recovery.chunk_size": "16kb"})).await;
    let iterations: usize = std::env::var("RECOVERY_RACE_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pollers: Vec<_> = (0..3).map(|i| {
        let (c, port, stop) = (three.c.clone(), three.http[i], stop.clone());
        tokio::spawn(async move {
            let (mut polls, mut failed) = (0usize, Vec::new());
            while !stop.load(Ordering::Relaxed) {
                for (path, status) in poll_local_stats(&c, port, "_all").await {
                    polls += 1;
                    // 0: the node is gone (it aborted); 5xx: a failed reader.
                    if status == 0 || status >= 500 {
                        failed.push(format!("n{i} {path}: {status}"));
                    }
                }
            }
            (polls, failed)
        })
    }).collect();
    let started = Instant::now();
    for it in 0..iterations {
        three.vector_index("index2", "n0").await;
        three.bulk("index2", 0..(100 + 50 * (it % 5))).await;
        three.ok(0, Method::POST, "/index2/_flush", None).await;
        three.ok(0, Method::PUT, "/index2/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
        match it % 3 {
            0 => {}
            1 => tokio::time::sleep(Duration::from_millis(rand::thread_rng().gen_range(0..300))).await,
            _ => three.green("index2", 2).await,
        }
        three.ok(0, Method::DELETE, "/index2", None).await;
        let panics = seen.lock().unwrap().clone();
        assert!(panics.is_empty(), "iteration {it}: panics: {:?}", panics);
    }
    stop.store(true, Ordering::Relaxed);
    let mut polls = 0;
    for p in pollers {
        let (n, failed) = p.await.unwrap();
        polls += n;
        assert!(failed.is_empty(), "stats readers failed: {failed:?}");
    }
    // Every node still serves, and the name recovers once more.
    three.vector_index("index2", "n0").await;
    three.bulk("index2", 0..300).await;
    three.ok(0, Method::PUT, "/index2/_settings", Some(json!({"index": {"number_of_replicas": 1, "routing.allocation.include._name": "n0,n1"}}))).await;
    three.green("index2", 2).await;
    assert_eq!(three.knn("index2", 1, 0).await, three.knn("index2", 0, 0).await);
    let panics = seen.lock().unwrap().clone();
    assert!(panics.is_empty(), "panics: {:?}", panics);
    eprintln!("{iterations} create/recover/delete iterations, {polls} stats requests, {:?}", started.elapsed());
}
