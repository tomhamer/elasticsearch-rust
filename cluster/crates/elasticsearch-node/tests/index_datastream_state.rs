//! Index metadata and data stream state on a two-node cluster (production
//! review #84–#86):
//!
//! * #84: a closed index rejects a shard count change (ES: final setting;
//!   resizing goes through shrink / split / clone) and reopens with every
//!   document; before, the smaller count was published and the shard
//!   engines deleted the shards beyond it.
//! * #85: after a rollover, a write through a data stream alias lands in
//!   the stream's new write index on every node, not in the oldest backing
//!   index still carrying the alias's write flag.
//! * #86: an accepted lazy rollover (`?lazy=true`) rolls the stream over
//!   before the next write through the cluster, exactly once however many
//!   writers race, by the stream's name or through its alias.
//!
//! Nodes run in-process (on the test runtime), one cluster per test.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn settings(name: &str, cluster: &str, seeds: &[u16]) -> Settings {
    let dir = evidence::scratch_dir(&format!("{cluster}-{name}"));
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(name));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
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
    nodes: [Arc<ClusterNode>; 2],
    http: [u16; 2],
    c: reqwest::Client,
}

impl Cluster {
    async fn start(tag: &str) -> Cluster {
        let cluster = ports::unique(tag);
        let a = spawn_node(settings("n0", &cluster, &[])).await;
        let b = spawn_node(settings("n1", &cluster, &[ports::transport_port(&a)])).await;
        let http = [ports::http_port(&a), ports::http_port(&b)];
        let c = reqwest::Client::new();
        ports::wait_formed(&c, &http, None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
        Cluster { nodes: [a, b], http, c }
    }

    async fn call(&self, node: usize, method: Method, path: &str, body: Option<String>) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.http[node]));
        if let Some(b) = body {
            let ct = if path.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
            r = r.header("content-type", ct).body(b);
        }
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn ok(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(node, method.clone(), path, body.map(|b| b.to_string())).await;
        assert!((200..300).contains(&st), "{method} {path} via n{node}: {st} {v}");
        v
    }

    async fn green(&self, index: &str) {
        let v = self.ok(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(v["status"], "green", "{v}");
    }

    /// The backing indices of data stream `ds`, oldest first, as node
    /// `node` sees them, and its `rollover_on_write`.
    async fn backing(&self, node: usize, ds: &str) -> (Vec<String>, bool) {
        let v = self.ok(node, Method::GET, &format!("/_data_stream/{ds}"), None).await;
        let d = &v["data_streams"][0];
        let names = d["indices"].as_array().map(|a| a.iter().filter_map(|i| i["index_name"].as_str().map(str::to_string)).collect()).unwrap_or_default();
        (names, d["rollover_on_write"].as_bool().unwrap_or(false))
    }

    async fn count(&self, node: usize, index: &str) -> u64 {
        let v = self.ok(node, Method::GET, &format!("/{index}/_count"), None).await;
        v["count"].as_u64().unwrap_or_else(|| panic!("{v}"))
    }
}

/// #84: `index.number_of_shards` on a closed two-shard index is rejected
/// with ES's error; the index reopens with both shards and every document.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn closed_index_shard_count_change_is_rejected_and_keeps_every_document() {
    let cl = Cluster::start("closed-shards").await;
    cl.ok(0, Method::PUT, "/closed-two", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1}}))).await;
    cl.green("closed-two").await;
    let mut body = String::new();
    for i in 0..40 {
        body.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{{\"n\":{i}}}\n"));
    }
    let (st, v) = cl.call(1, Method::POST, "/closed-two/_bulk?refresh=true", Some(body)).await;
    assert!(st == 200 && v["errors"] == false, "{st} {v}");
    let shards = cl.ok(0, Method::GET, "/_cat/shards/closed-two?format=json&h=shard,prirep,docs", None).await;
    let primaries: Vec<u64> = shards.as_array().unwrap().iter().filter(|r| r["prirep"] == "p").map(|r| r["docs"].as_str().unwrap_or("0").parse().unwrap_or(0)).collect();
    assert!(primaries.len() == 2 && primaries.iter().all(|d| *d > 0), "both shards hold documents: {shards}");
    cl.ok(0, Method::POST, "/closed-two/_close", None).await;
    for (via, body) in [(0, json!({"index.number_of_shards": 1})), (1, json!({"index": {"number_of_shards": 1}})), (0, json!({"index.number_of_shards": 4}))] {
        for reopen in ["", "?reopen=true"] {
            let (st, v) = cl.call(via, Method::PUT, &format!("/closed-two/_settings{reopen}"), Some(body.to_string())).await;
            assert_eq!(st, 400, "{v}");
            assert_eq!(v["error"]["type"], "illegal_argument_exception", "{v}");
            assert_eq!(v["error"]["reason"], "final index setting [index.number_of_shards], not updateable", "{v}");
        }
    }
    for n in 0..2 {
        let im = &cl.nodes[n].state().metadata.indices["closed-two"];
        assert_eq!(im.number_of_shards, 2, "n{n}");
    }
    cl.ok(1, Method::POST, "/closed-two/_open", None).await;
    cl.green("closed-two").await;
    for n in 0..2 {
        let v = cl.ok(n, Method::GET, "/closed-two/_settings/index.number_of_shards?flat_settings=true", None).await;
        assert_eq!(v["closed-two"]["settings"]["index.number_of_shards"], "2", "{v}");
        assert_eq!(cl.count(n, "closed-two").await, 40, "via n{n}");
        for i in 0..40 {
            let v = cl.ok(n, Method::GET, &format!("/closed-two/_doc/{i}"), None).await;
            assert_eq!(v["_source"]["n"], i, "{v}");
        }
    }
    let shards = cl.ok(0, Method::GET, "/_cat/shards/closed-two?format=json&h=shard,prirep,docs,state", None).await;
    let rows = shards.as_array().unwrap();
    assert_eq!(rows.len(), 4, "{shards}");
    for s in ["0", "1"] {
        let docs: Vec<&str> = rows.iter().filter(|r| r["shard"] == s).map(|r| r["docs"].as_str().unwrap_or("")).collect();
        assert!(docs.len() == 2 && docs[0] == docs[1] && docs[0] != "0", "shard {s}: {shards}");
    }
}

const TEMPLATE: &str = r#"{"index_patterns": ["logs-st-*"], "priority": 1000, "data_stream": {},
    "template": {"settings": {"index": {"number_of_shards": 2, "number_of_replicas": 1}},
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}}}}}"#;

fn doc(n: usize) -> Value {
    json!({"@timestamp": "2026-11-08T10:00:00Z", "n": n})
}

/// #85: writes through a data stream alias follow every rollover (manual,
/// on either node) into the newest generation, through the single-document
/// API and bulk, coordinated by either node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn data_stream_alias_writes_land_in_the_new_generation_after_rollover() {
    let cl = Cluster::start("ds-alias-rollover").await;
    cl.ok(0, Method::PUT, "/_index_template/st", Some(serde_json::from_str(TEMPLATE).unwrap())).await;
    cl.ok(0, Method::PUT, "/_data_stream/logs-st-alias", None).await;
    cl.ok(1, Method::POST, "/_aliases", Some(json!({"actions": [{"add": {"index": "logs-st-alias", "alias": "st-write", "is_write_index": true}}]}))).await;
    cl.green("logs-st-alias").await;
    let mut n = 0;
    for generation in 1..=3usize {
        if generation > 1 {
            // Alternate the node the rollover is sent to.
            let v = cl.ok(generation % 2, Method::POST, "/logs-st-alias/_rollover", None).await;
            assert_eq!(v["rolled_over"], true, "{v}");
            cl.green("logs-st-alias").await;
        }
        let (indices, _) = cl.backing(0, "logs-st-alias").await;
        assert_eq!(indices.len(), generation, "{indices:?}");
        let newest = indices.last().unwrap().clone();
        assert!(newest.ends_with(&format!("-{generation:06}")), "{newest}");
        for via in 0..2 {
            n += 1;
            let v = cl.ok(via, Method::POST, "/st-write/_doc?refresh=true", Some(doc(n))).await;
            assert_eq!(v["_index"], newest.as_str(), "generation {generation}, single doc via n{via}: {v}");
            let mut body = String::new();
            for _ in 0..4 {
                n += 1;
                body.push_str(&format!("{{\"create\":{{}}}}\n{}\n", doc(n)));
            }
            let (st, v) = cl.call(via, Method::POST, "/st-write/_bulk?refresh=true", Some(body)).await;
            assert!(st == 200 && v["errors"] == false, "{st} {v}");
            for item in v["items"].as_array().unwrap() {
                assert_eq!(item["create"]["_index"], newest.as_str(), "generation {generation}, bulk via n{via}: {v}");
            }
        }
        // Each generation holds exactly what was written while it was the
        // write index (10 per generation), on every node.
        for node in 0..2 {
            for (g, idx) in indices.iter().enumerate() {
                assert_eq!(cl.count(node, idx).await, 10, "generation {} ({idx}) via n{node}", g + 1);
            }
        }
    }
}

/// #86: a lazy rollover accepted on the cluster rolls the stream over on
/// the next write, exactly once with many concurrent writers on both
/// nodes; every write after it lands in the new generation. A second lazy
/// rollover is performed by a write through the stream's alias.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn lazy_rollover_happens_once_before_the_next_cluster_write() {
    let cl = Cluster::start("ds-lazy-rollover").await;
    cl.ok(0, Method::PUT, "/_index_template/st", Some(serde_json::from_str(TEMPLATE).unwrap())).await;
    cl.ok(1, Method::POST, "/logs-st-lazy/_doc?refresh=true", Some(doc(0))).await;
    cl.green("logs-st-lazy").await;
    let (first, pending) = cl.backing(0, "logs-st-lazy").await;
    assert_eq!((first.len(), pending), (1, false), "{first:?}");
    for round in 0..2usize {
        let v = cl.ok(round % 2, Method::POST, "/logs-st-lazy/_rollover?lazy=true", None).await;
        assert_eq!((v["acknowledged"].as_bool(), v["rolled_over"].as_bool(), v["lazy"].as_bool()), (Some(true), Some(false), Some(true)), "{v}");
        for node in 0..2 {
            let (indices, pending) = cl.backing(node, "logs-st-lazy").await;
            assert_eq!(indices.len(), round + 1, "lazy rollover must not roll over by itself: {indices:?}");
            assert!(pending, "rollover_on_write published to n{node}");
        }
        // Round 1 writes through an alias of the stream.
        let target = if round == 0 {
            "logs-st-lazy".to_string()
        } else {
            cl.ok(0, Method::POST, "/_aliases", Some(json!({"actions": [{"add": {"index": "logs-st-lazy", "alias": "lazy-write", "is_write_index": true}}]}))).await;
            "lazy-write".to_string()
        };
        let writers: Vec<_> = (0..12usize).map(|w| {
            let (c, port, target) = (cl.c.clone(), cl.http[w % 2], target.clone());
            tokio::spawn(async move {
                let mut body = String::new();
                for k in 0..5 {
                    body.push_str(&format!("{{\"create\":{{}}}}\n{}\n", doc(1000 * (round + 1) + 10 * w + k)));
                }
                let resp = c.post(format!("http://127.0.0.1:{port}/{target}/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(body).send().await.expect("bulk");
                let status = resp.status().as_u16();
                let v: Value = resp.json().await.unwrap_or(Value::Null);
                assert!(status == 200 && v["errors"] == false, "writer {w}: {status} {v}");
                v["items"].as_array().unwrap().iter().map(|i| i["create"]["_index"].as_str().unwrap_or_default().to_string()).collect::<Vec<_>>()
            })
        }).collect();
        let mut landed = Vec::new();
        for w in writers {
            landed.extend(w.await.expect("writer"));
        }
        cl.green("logs-st-lazy").await;
        let (indices, pending) = cl.backing(0, "logs-st-lazy").await;
        assert_eq!(indices.len(), round + 2, "exactly one generation advance per accepted lazy rollover: {indices:?}");
        assert!(!pending, "the lazy rollover was consumed");
        let newest = indices.last().unwrap();
        assert!(landed.iter().all(|i| i == newest), "every write after the lazy rollover lands in {newest}: {landed:?}");
        assert_eq!(cl.count(1, newest).await, 60);
        for node in 0..2 {
            let (seen, pending) = cl.backing(node, "logs-st-lazy").await;
            assert_eq!((seen.len(), pending), (round + 2, false), "n{node}: {seen:?}");
        }
        // The rolled-over generation kept only what was written before.
        let previous = &indices[indices.len() - 2];
        assert_eq!(cl.count(0, previous).await, if round == 0 { 1 } else { 60 });
    }
}
