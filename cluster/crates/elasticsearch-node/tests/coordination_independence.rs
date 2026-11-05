//! Cluster coordination never waits on engine work (Step 1 of the
//! coordination redesign):
//!
//! * slow metadata work and state application run off the coordination
//!   actor: fault detection keeps answering, no election happens;
//! * a publication that cannot commit leaves no trace in the master's
//!   engine (the metadata service rolls it back);
//! * the background duties are independent: a shard whose flush is stuck
//!   holds up neither context expiry nor other shards nor allocation.
//!
//! Nodes run in-process (on the test runtime) so the test can reach into
//! them (`ClusterNode`).

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A node on OS-assigned ports (read back with [`ports::http_port`]); with no
/// `masters` a single-node cluster, else a member that seeds from the
/// already-started nodes' transport ports `seeds` (none for the first).
fn settings(name: &str, cluster: &str, seeds: &[u16], masters: &[&str], extra: &[(&str, &str)]) -> Settings {
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
    if masters.is_empty() {
        put("discovery.type", json!("single-node"));
    } else {
        if !seeds.is_empty() {
            put("discovery.seed_hosts", json!(ports::seeds(seeds)));
        }
        put("cluster.initial_master_nodes", json!(masters));
    }
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    for (k, v) in extra {
        put(k, json!(v));
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

async fn call(c: &reqwest::Client, http: u16, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{http}{path}"));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let Ok(resp) = r.send().await else { return (0, Value::Null) };
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

/// The nodes' bound HTTP ports.
fn http_ports(nodes: &[Arc<ClusterNode>]) -> [u16; 2] {
    [ports::http_port(&nodes[0]), ports::http_port(&nodes[1])]
}

/// The nodes' bound transport ports.
fn transport_ports(nodes: &[Arc<ClusterNode>]) -> [u16; 2] {
    [ports::transport_port(&nodes[0]), ports::transport_port(&nodes[1])]
}

/// Start two master-eligible nodes and wait until one is the master and
/// both are members.
async fn two_nodes(cluster: &str, extra: &[(&str, &str)]) -> [Arc<ClusterNode>; 2] {
    let a = spawn_node(settings("n0", cluster, &[], &["n0", "n1"], extra)).await;
    let b = spawn_node(settings("n1", cluster, &[ports::transport_port(&a)], &["n0", "n1"], extra)).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let formed = [&a, &b].iter().all(|n| n.state().nodes.len() == 2 && n.state().master_node_id.is_some());
        if formed && (a.is_master() || b.is_master()) {
            return [a, b];
        }
        assert!(Instant::now() < deadline, "cluster did not form");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Index creations with very large mappings, back to back, on the master:
/// with aggressive fault detection a stalled actor would fail leader /
/// follower checks and trigger an election. The term and the master stay.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_metadata_work_does_not_cause_an_election() {
    let fd = [
        ("cluster.fault_detection.leader_check.interval", "200ms"),
        ("cluster.fault_detection.leader_check.timeout", "500ms"),
        ("cluster.fault_detection.leader_check.retry_count", "1"),
        ("cluster.fault_detection.follower_check.interval", "200ms"),
        ("cluster.fault_detection.follower_check.timeout", "500ms"),
        ("cluster.fault_detection.follower_check.retry_count", "1"),
    ];
    let nodes = two_nodes("slow-apply", &fd).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let term = nodes[master].coord().info().await.expect("info").term;
    let master_id = nodes[master].local.id.clone();
    let c = reqwest::Client::new();
    let mut properties = serde_json::Map::new();
    for i in 0..4000 {
        properties.insert(format!("field_{i}"), json!({"type": if i % 2 == 0 { "keyword" } else { "long" }}));
    }
    let mappings = json!({"properties": properties});
    for i in 0..8 {
        let body = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1, "index.mapping.total_fields.limit": 10000}, "mappings": mappings});
        let (st, v) = call(&c, http[master], reqwest::Method::PUT, &format!("/big-{i}"), Some(body)).await;
        assert_eq!(st, 200, "{v}");
    }
    for n in &nodes {
        let info = n.coord().info().await.expect("info");
        assert_eq!(info.term, term, "no election happened");
        assert_eq!(n.state().master_node_id.as_deref(), Some(master_id.as_str()), "the master did not change");
        assert_eq!(n.state().nodes.len(), 2, "no node was removed by fault detection");
    }
}

/// The follower is cut off: the master cannot commit the publication that
/// would create the index. The request fails and the master's engine does
/// not keep the index.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_publication_leaves_no_trace_in_the_masters_engine() {
    // Two master-eligible nodes keep both in the voting configuration
    // (ES would shrink it to one, and the master alone could commit).
    let extra = [("cluster.publish.timeout", "2s"), ("cluster.fault_detection.follower_check.interval", "200ms"), ("cluster.fault_detection.follower_check.timeout", "500ms"), ("cluster.auto_shrink_voting_configuration", "false")];
    let nodes = two_nodes("failed-publication", &extra).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let follower = 1 - master;
    // The voting configuration is both nodes: without the follower no
    // publication can commit.
    nodes[follower].transport.close().await;
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/ghost", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert!(!(200..300).contains(&st), "creation must not be acknowledged without a commit: {st} {v}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while nodes[master].engine.get_index("ghost").is_some() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(nodes[master].engine.get_index("ghost").is_none(), "the uncommitted index was rolled back from the master's engine");
    assert!(!nodes[master].state().metadata.indices.contains_key("ghost"), "never applied");
}

/// A primary whose write gate is held (a stuck flush, a hung replica)
/// blocks only its own background work: context expiry, other shards and
/// allocation of a new index carry on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stuck_shard_does_not_stall_other_background_work() {
    let node = spawn_node(settings("n0", "stuck-shard", &[], &[], &[("indices.memory.shard_inactive_time", "1s")])).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    for index in ["held", "scrolled"] {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
        assert_eq!(st, 200, "{v}");
        for i in 0..5 {
            let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{index}/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
            assert!((200..300).contains(&st), "{v}");
        }
    }
    // Hold the write gate of `held` exclusively (as a stuck recovery
    // handoff or a hung write would): its background replication top-up
    // waits for it forever.
    let primary = node.shards.primary("held", 0).expect("primary of held");
    let gate = primary.gate.clone().write_owned().await;
    let (st, v) = call(&c, http, reqwest::Method::POST, "/scrolled/_search?scroll=1s", Some(json!({"size": 1}))).await;
    assert_eq!(st, 200, "{v}");
    let scroll_id = v["_scroll_id"].as_str().expect("scroll id").to_string();
    // Two flush rounds (5s apart) and several maintenance ticks.
    tokio::time::sleep(Duration::from_secs(12)).await;
    let (st, v) = call(&c, http, reqwest::Method::POST, "/_search/scroll", Some(json!({"scroll_id": scroll_id}))).await;
    assert_eq!(st, 404, "the expired scroll was reaped while a shard's flush was stuck: {v}");
    let started = Instant::now();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/fresh?wait_for_active_shards=1", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_cluster/health/fresh?wait_for_status=green&timeout=10s", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["status"], "green", "{v}");
    assert!(started.elapsed() < Duration::from_secs(10), "allocation was not delayed by the stuck shard");
    drop(gate);
}

/// The mapping of `index` in every local engine of `node`.
fn engine_mappings(node: &ClusterNode, index: &str) -> Vec<Value> {
    let mut out: Vec<Value> = node.engine.get_index(index).map(|s| s.mapping_json()).into_iter().collect();
    for (_, se) in node.shard_engines.all() {
        if let Some(s) = se.engine.get_index(index) {
            out.push(s.mapping_json());
        }
    }
    out
}

/// A mapping update whose publication cannot commit (the follower is cut
/// off) never reaches the master's engine: `explicit` sends `PUT _mapping`,
/// otherwise a document introduces the field dynamically on a primary the
/// master holds.
#[derive(Clone, Copy, PartialEq)]
enum MappingPath {
    Dynamic,
    Explicit,
    /// Explicit, with a YAML body (decoded at the cluster edge).
    ExplicitYaml,
}

async fn failed_mapping_commit_leaves_no_field(cluster: &str, path: MappingPath) {
    let explicit = path != MappingPath::Dynamic;
    // Two master-eligible nodes keep both in the voting configuration
    // (ES would shrink it to one, and the master alone could commit).
    let extra = [("cluster.publish.timeout", "2s"), ("cluster.fault_detection.follower_check.interval", "200ms"), ("cluster.fault_detection.follower_check.timeout", "500ms"), ("cluster.auto_shrink_voting_configuration", "false")];
    let nodes = two_nodes(cluster, &extra).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let follower = 1 - master;
    let c = reqwest::Client::new();
    let master_name = nodes[master].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/kept", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": master_name},
        "mappings": {"properties": {"a": {"type": "long"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/kept?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    nodes[follower].transport.close().await;
    let (st, v) = if path == MappingPath::ExplicitYaml {
        let r = c.put(format!("http://127.0.0.1:{}/kept/_mapping", http[master])).header("content-type", "application/yaml")
            .body("properties:\n  new_field:\n    type: keyword\n").send().await;
        match r {
            Ok(r) => {
                let st = r.status().as_u16();
                (st, Value::String(r.text().await.unwrap_or_default()))
            }
            Err(_) => (0, Value::Null),
        }
    } else if explicit {
        call(&c, http[master], reqwest::Method::PUT, "/kept/_mapping", Some(json!({"properties": {"new_field": {"type": "keyword"}}}))).await
    } else {
        call(&c, http[master], reqwest::Method::PUT, "/kept/_doc/1", Some(json!({"a": 1, "new_field": "x"}))).await
    };
    assert!(!(200..300).contains(&st), "the update must not be acknowledged without a commit: {st} {v}");
    // Give a (wrong) late local application every chance to show up.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let mappings = engine_mappings(&nodes[master], "kept");
    assert!(!mappings.is_empty(), "the master holds the index");
    for m in mappings {
        assert!(m["properties"].get("new_field").is_none(), "an uncommitted field reached the master's engine: {m}");
        assert_eq!(m["properties"]["a"]["type"], "long");
    }
    assert!(nodes[master].state().metadata.indices["kept"].mappings["properties"].get("new_field").is_none());
    if !explicit {
        let (_, doc) = call(&c, http[master], reqwest::Method::GET, "/kept/_doc/1", None).await;
        assert_ne!(doc["found"], json!(true), "the document needing the field was not indexed: {doc}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_publication_leaves_no_explicit_mapping_field_in_the_masters_engine() {
    failed_mapping_commit_leaves_no_field("failed-explicit-mapping", MappingPath::Explicit).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_publication_leaves_no_dynamic_mapping_field_in_the_masters_engine() {
    failed_mapping_commit_leaves_no_field("failed-dynamic-mapping", MappingPath::Dynamic).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_publication_leaves_no_yaml_mapping_field_in_the_masters_engine() {
    failed_mapping_commit_leaves_no_field("failed-yaml-mapping", MappingPath::ExplicitYaml).await;
}

/// A fresh two-node cluster whose publications fail fast once the follower
/// is cut off; returns the nodes and the master's position.
async fn fast_failing_cluster(cluster: &str) -> ([Arc<ClusterNode>; 2], usize) {
    // Two master-eligible nodes keep both in the voting configuration
    // (ES would shrink it to one, and the master alone could commit).
    let extra = [("cluster.publish.timeout", "2s"), ("cluster.fault_detection.follower_check.interval", "200ms"), ("cluster.fault_detection.follower_check.timeout", "500ms"), ("cluster.auto_shrink_voting_configuration", "false")];
    let nodes = two_nodes(cluster, &extra).await;
    let master = if nodes[0].is_master() { 0 } else { 1 };
    (nodes, master)
}

/// A delete whose publication fails destroys nothing: the master's copy of
/// the index is intact and its documents are still there (the delete ran
/// on the scratch copy; the applier deletes data only after a commit).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_delete_leaves_the_masters_index_intact() {
    let (nodes, master) = fast_failing_cluster("failed-delete").await;
    let http = http_ports(&nodes);
    let follower = 1 - master;
    let c = reqwest::Client::new();
    let master_name = nodes[master].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/victim", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": master_name}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/victim?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..5 {
        let (st, v) = call(&c, http[master], reqwest::Method::PUT, &format!("/victim/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        assert!((200..300).contains(&st), "{v}");
    }
    nodes[follower].transport.close().await;
    let (st, v) = call(&c, http[master], reqwest::Method::DELETE, "/victim", None).await;
    assert!(!(200..300).contains(&st), "a delete must not be acknowledged without a commit: {st} {v}");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let svc = nodes[master].engine.get_index("victim").expect("the master's index survives a failed delete");
    assert_eq!(svc.doc_count().0, 5, "its documents are intact");
    assert!(nodes[master].state().metadata.indices.contains_key("victim"));
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/victim/_doc/3", None).await;
    assert!(st == 200 && v["found"] == json!(true), "still readable on the master: {st} {v}");
}

/// While a metadata batch is in flight (its publication cannot commit), the
/// master's engine shows nothing of it: no new index, no alias, no setting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_uncommitted_batch_is_invisible_on_the_master() {
    let (nodes, master) = fast_failing_cluster("invisible-batch").await;
    let http = http_ports(&nodes);
    let follower = 1 - master;
    let c = reqwest::Client::new();
    nodes[follower].transport.close().await;
    let port = http[master];
    let client = c.clone();
    let create = tokio::spawn(async move {
        call(&client, port, reqwest::Method::PUT, "/pending", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.refresh_interval": "7s"}, "aliases": {"pending-alias": {}}}))).await
    });
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut polls = 0;
    while Instant::now() < deadline {
        let engine = &nodes[master].engine;
        assert!(engine.get_index("pending").is_none(), "an uncommitted index reached the master's engine");
        assert!(!engine.all_indices().iter().any(|s| s.aliases().contains_key("pending-alias")), "an uncommitted alias reached the master's engine");
        let (st, _) = call(&c, port, reqwest::Method::GET, "/pending/_settings", None).await;
        assert_ne!(st, 200, "a local read on the master saw the uncommitted index");
        polls += 1;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(polls > 10);
    let (st, v) = create.await.unwrap();
    assert!(!(200..300).contains(&st), "{st} {v}");
    assert!(nodes[master].engine.get_index("pending").is_none());
}

/// ILM rolls over on the master while index-level metadata batches run:
/// the rollover is published (ES runs ILM steps as cluster-state tasks) and
/// survives every later state application, and no batch fails or erases it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_rollover_on_the_master_is_published_beside_dry_runs() {
    let node = spawn_node(settings("n0", "ilm-beside-dry-runs", &[], &[], &[])).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/roll", Some(json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"max_docs": 1}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_index_template/ilm-logs", Some(json!({"index_patterns": ["ilm-logs-*"], "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "roll", "index.lifecycle.rollover_alias": "ilm-logs"}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/ilm-logs-000001", Some(json!({"aliases": {"ilm-logs": {"is_write_index": true}}}))).await;
    ok(st, &v);
    for i in 0..3 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/ilm-logs/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    // Dry-run batches (index creations) back to back meanwhile.
    let client = c.clone();
    let creations = tokio::spawn(async move {
        let mut out = Vec::new();
        for i in 0..30 {
            out.push(call(&client, http, reqwest::Method::PUT, &format!("/side-{i}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        out
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    while !node.state().metadata.indices.contains_key("ilm-logs-000002") {
        if Instant::now() >= deadline {
            let (_, explain) = call(&c, http, reqwest::Method::GET, "/ilm-logs-000001/_ilm/explain", None).await;
            let engine_has = node.engine.get_index("ilm-logs-000002").is_some();
            panic!("the lifecycle rollover was never published (rolled over in the master's engine: {engine_has}); explain: {explain}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    for (st, v) in creations.await.unwrap() {
        ok(st, &v);
    }
    // Later state applications keep it in the engine and the state.
    for i in 0..5 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/after-{i}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
        ok(st, &v);
    }
    assert!(node.engine.get_index("ilm-logs-000002").is_some(), "the rollover index stays in the master's engine");
    assert!(node.state().metadata.indices.contains_key("ilm-logs-000002"));
    let write_index = node.state().metadata.indices["ilm-logs-000002"].aliases.get("ilm-logs").and_then(|a| a.get("is_write_index")).cloned();
    assert_eq!(write_index, Some(json!(true)), "the alias moved with the rollover");
}

/// Requests that change nothing publish nothing: an empty delta (the
/// scratch copy round-trips the committed metadata exactly).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_op_metadata_requests_publish_an_empty_delta() {
    let node = spawn_node(settings("n0", "no-op-delta", &[], &[], &[])).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/same", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "5s"},
        "mappings": {"_meta": {"k": "v"}, "properties": {"a": {"type": "keyword"}}},
        "aliases": {"same-alias": {"is_hidden": true}, "same-routed": {"routing": "r"}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_cluster/health/same?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    tokio::time::sleep(Duration::from_secs(1)).await;
    let publications = node.metadata_service().stats.publications.load(std::sync::atomic::Ordering::Relaxed);
    let version = node.state().metadata.indices["same"].version;
    for (method, path, body) in [
        (reqwest::Method::PUT, "/same/_settings", json!({"index": {"refresh_interval": "5s"}})),
        (reqwest::Method::PUT, "/same/_mapping", json!({"properties": {"a": {"type": "keyword"}}})),
        (reqwest::Method::PUT, "/same/_alias/same-alias", json!({"is_hidden": true})),
        (reqwest::Method::PUT, "/same/_alias/same-routed", json!({"routing": "r"})),
    ] {
        let (st, v) = call(&c, http, method, path, Some(body)).await;
        assert_eq!(st, 200, "{path}: {v}");
    }
    assert_eq!(node.metadata_service().stats.publications.load(std::sync::atomic::Ordering::Relaxed), publications, "no-op requests published something");
    assert_eq!(node.state().metadata.indices["same"].version, version);
}

/// The write index's primary lives on the other node: the master's own copy
/// is a metadata-only shell with no documents. ILM still rolls over on the
/// master, judging the conditions by the cluster-wide primary stats (ES:
/// `IndicesStats` across the data nodes), and the rollover is published.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_rollover_judges_cluster_wide_stats() {
    let nodes = two_nodes("ilm-cluster-stats", &[]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let other = 1 - master;
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let other_name = nodes[other].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_ilm/policy/roll-remote", Some(json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"max_docs": 2}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_index_template/remote-logs", Some(json!({"index_patterns": ["remote-logs-*"], "template": {"settings": {
        "number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "roll-remote", "index.lifecycle.rollover_alias": "remote-logs",
        "index.routing.allocation.require._name": other_name}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/remote-logs-000001", Some(json!({"aliases": {"remote-logs": {"is_write_index": true}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/remote-logs-000001?wait_for_status=green&timeout=30s", None).await;
    ok(st, &v);
    for i in 0..3 {
        let (st, v) = call(&c, http[other], reqwest::Method::PUT, &format!("/remote-logs/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while !nodes[master].state().metadata.indices.contains_key("remote-logs-000002") {
        if Instant::now() >= deadline {
            let (_, explain) = call(&c, http[master], reqwest::Method::GET, "/remote-logs-000001/_ilm/explain", None).await;
            panic!("no rollover on cluster-wide stats; explain: {explain}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    for n in &nodes {
        let deadline = Instant::now() + Duration::from_secs(10);
        while n.engine.get_index("remote-logs-000002").is_none() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(n.engine.get_index("remote-logs-000002").is_some(), "every node applied the rollover");
    }
}

/// ILM rolls over while no publication can commit (both nodes drop their
/// publish requests): the rollover is never visible anywhere, the index's
/// lifecycle state reverts with it (it is part of the index metadata, as
/// in ES), and once publications commit again ILM retries and rolls over
/// exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_lifecycle_publication_reverts_and_ilm_retries() {
    let extra = [("cluster.publish.timeout", "2s"), ("cluster.auto_shrink_voting_configuration", "false")];
    let nodes = two_nodes("ilm-failed-publication", &extra).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let master_name = nodes[master].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_ilm/policy/roll-flaky", Some(json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"max_docs": 1}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_index_template/flaky-logs", Some(json!({"index_patterns": ["flaky-logs-*"], "template": {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "roll-flaky", "index.lifecycle.rollover_alias": "flaky-logs",
            "index.routing.allocation.require._name": master_name},
        "mappings": {"properties": {"n": {"type": "long"}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/flaky-logs-000001", Some(json!({"aliases": {"flaky-logs": {"is_write_index": true}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/flaky-logs-000001?wait_for_status=green&timeout=30s", None).await;
    ok(st, &v);
    for n in &nodes {
        n.faults.drop_publications.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    for i in 0..2 {
        let (st, v) = call(&c, http[master], reqwest::Method::PUT, &format!("/flaky-logs-000001/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    // ILM polls every second; every attempt's publication fails.
    let until = Instant::now() + Duration::from_secs(8);
    while Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(500)).await;
        for n in &nodes {
            assert!(!n.state().metadata.indices.contains_key("flaky-logs-000002"), "an uncommitted rollover was applied");
        }
    }
    // The last attempt's rollback has run (its publication timed out).
    tokio::time::sleep(Duration::from_secs(3)).await;
    for n in &nodes {
        assert!(n.engine.get_index("flaky-logs-000002").is_none(), "an uncommitted rollover index survived its failed publication on [{}]", n.local.name);
    }
    for n in &nodes {
        n.faults.drop_publications.store(false, std::sync::atomic::Ordering::SeqCst);
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if nodes.iter().all(|n| n.state().metadata.indices.contains_key("flaky-logs-000002")) {
            break;
        }
        if Instant::now() >= deadline {
            let port = if nodes[0].is_master() { http[0] } else { http[1] };
            let (_, explain) = call(&c, port, reqwest::Method::GET, "/flaky-logs-000001/_ilm/explain", None).await;
            panic!("ILM did not retry the rollover after publications recovered; explain: {explain}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    for n in &nodes {
        assert!(!n.state().metadata.indices.contains_key("flaky-logs-000003"), "rolled over exactly once");
    }
}

async fn single_node(cluster: &str) -> Arc<ClusterNode> {
    let node = spawn_node(settings("n0", cluster, &[], &[], &[])).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    node
}

/// Indices and data streams committed while lifecycle passes run
/// (every second) survive the kept jobs (`PUT _ingest/pipeline`, the
/// registry reconcile) published right after them: a kept change is the
/// diff of its own job, never "engine 0 against the newest state".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indices_committed_during_lifecycle_passes_survive_kept_jobs() {
    let node = single_node("lifecycle-kept").await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/b1-roll", Some(json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"max_docs": 1}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_index_template/b1-ds", Some(json!({"index_patterns": ["b1-ds-*"], "data_stream": {}, "priority": 200,
        "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "b1-roll"}}}))).await;
    ok(st, &v);
    // Kept jobs back to back meanwhile.
    let client = c.clone();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_kept = stop.clone();
    let kept = tokio::spawn(async move {
        let mut i = 0;
        while !stop_kept.load(std::sync::atomic::Ordering::SeqCst) {
            let (st, v) = call(&client, http, reqwest::Method::PUT, &format!("/_ingest/pipeline/b1-{}", i % 3), Some(json!({"processors": [{"set": {"field": "p", "value": i}}]}))).await;
            assert!((200..300).contains(&st), "{st} {v}");
            i += 1;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    for i in 0..12 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/b1-plain-{i}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
        ok(st, &v);
        let mut doc = json!({"n": i});
        doc[format!("f{i}")] = json!(i);
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/b1-plain-{i}/_doc/1?refresh=true"), Some(doc)).await;
        ok(st, &v);
        // Auto-created data stream (ILM rolls it over once it has a doc).
        let (st, v) = call(&c, http, reqwest::Method::POST, &format!("/b1-ds-{i}/_doc?refresh=true"), Some(json!({"@timestamp": "2026-10-01T00:00:00Z", "n": i}))).await;
        ok(st, &v);
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    // A few more lifecycle passes and kept jobs after the last commit.
    tokio::time::sleep(Duration::from_secs(4)).await;
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    kept.await.unwrap();
    let state = node.state();
    for i in 0..12 {
        let name = format!("b1-plain-{i}");
        let im = state.metadata.indices.get(&name).unwrap_or_else(|| panic!("[{name}] was deleted by a later publication"));
        assert!(im.mappings["properties"].get(format!("f{i}")).is_some(), "[{name}]'s dynamic field was reverted: {}", *im.mappings);
        let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/{name}/_count"), None).await;
        assert_eq!((st, v["count"].as_u64()), (200, Some(1)), "{v}");
        let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/b1-ds-{i}/_count"), None).await;
        assert_eq!((st, v["count"].as_u64()), (200, Some(1)), "data stream [b1-ds-{i}] lost its document: {v}");
    }
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/b1-ds-*", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["data_streams"].as_array().map(Vec::len), Some(12), "{v}");
}

/// A multi-index delete that fails part-way on the scratch copy
/// (a data stream's write index cannot be deleted) leaves nothing behind:
/// the index it did delete there is restored, so a later create of that
/// name is refused instead of replacing the live index (and its data).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partly_failed_delete_does_not_let_a_create_replace_the_index() {
    let node = single_node("partial-delete").await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_index_template/h1-ds", Some(json!({"index_patterns": ["h1-ds*"], "data_stream": {}, "priority": 200,
        "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::POST, "/h1-ds/_doc?refresh=true", Some(json!({"@timestamp": "2026-10-01T00:00:00Z"}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h1-a", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h1-a/_doc/1?refresh=true", Some(json!({"k": "v"}))).await;
    ok(st, &v);
    let uuid = node.state().metadata.indices["h1-a"].uuid.clone();
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/h1-ds", None).await;
    ok(st, &v);
    let write = v["data_streams"][0]["indices"].as_array().and_then(|a| a.last()).and_then(|i| i["index_name"].as_str()).expect("write index").to_string();
    let (st, v) = call(&c, http, reqwest::Method::DELETE, &format!("/h1-a,{write}"), None).await;
    assert_eq!(st, 400, "deleting a data stream's write index is refused: {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h1-a", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("resource_already_exists_exception")), "{v}");
    assert_eq!(node.state().metadata.indices["h1-a"].uuid, uuid, "the index was replaced");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/h1-a/_count", None).await;
    assert_eq!((st, v["count"].as_u64()), (200, Some(1)), "{v}");
}

/// A dynamic proposal built on a stale mapping (or an older
/// primary's whole local mapping) adds its new field but cannot revert a
/// committed `dynamic: strict`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stale_mapping_proposal_cannot_revert_dynamic_strict() {
    let node = single_node("stale-proposal").await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h3", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"a": {"type": "keyword"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h3/_mapping", Some(json!({"dynamic": "strict"}))).await;
    assert_eq!(st, 200, "{v}");
    let uuid = node.state().metadata.indices["h3"].uuid.clone();
    // The primary's mapping before the strict change, plus a new field.
    let stale = json!({"dynamic": "true", "properties": {"a": {"type": "keyword"}, "b": {"type": "keyword"}}});
    // Exercise the public admitted wire boundary, not the private queue API:
    // the decoded envelope retains its allocation owner through publication.
    let proposal: elasticsearch_node::node::MappingUpdateReq = serde_json::from_value(json!({
        "index": "h3", "mapping_json": stale.to_string(), "index_uuid": uuid,
        "dynamic_delta": null
    })).expect("the stale proposal decodes with ownership");
    elasticsearch_node::mapping_update::handle(&node, proposal).await.expect("the addition commits");
    let m = node.state().metadata.indices["h3"].mappings.clone();
    assert_eq!(m["dynamic"], json!("strict"), "a stale proposal reverted dynamic: {}", *m);
    assert!(m["properties"].get("b").is_some(), "the new field is added: {}", *m);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/h3/_doc/1", Some(json!({"c": 1}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("strict_dynamic_mapping_exception")), "{v}");
}

/// An index the master's engine 0 does
/// not hold (its application failed there, or has not happened yet) is not
/// published as a delete by the next kept job. The index's primary lives on
/// the other node, so its documents are reachable however engine 0 fares.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_index_missing_from_engine_0_survives_the_next_kept_job() {
    let nodes = two_nodes("kept-drift", &[("cluster.auto_shrink_voting_configuration", "false")]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let other = 1 - master;
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let other_name = nodes[other].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/drift", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": other_name}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/drift?wait_for_status=green&timeout=30s", None).await;
    ok(st, &v);
    for i in 0..5 {
        let (st, v) = call(&c, http[other], reqwest::Method::PUT, &format!("/drift/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    let uuid = nodes[master].state().metadata.indices["drift"].uuid.clone();
    // Engine 0 on the master loses the index (a failed application).
    nodes[master].engine.delete_index("drift").expect("engine-0 copy removed");
    // Kept jobs: engine APIs that publish engine 0's changes.
    for i in 0..3 {
        let (st, v) = call(&c, http[master], reqwest::Method::PUT, &format!("/_ingest/pipeline/drift-{i}"), Some(json!({"processors": [{"set": {"field": "p", "value": i}}]}))).await;
        ok(st, &v);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    for n in &nodes {
        let im = n.state().metadata.indices.get("drift").cloned();
        assert_eq!(im.map(|im| im.uuid), Some(uuid.clone()), "a kept job published engine 0's missing index as a delete on [{}]", n.local.name);
    }
    let (st, v) = call(&c, http[other], reqwest::Method::GET, "/drift/_count", None).await;
    assert_eq!((st, v["count"].as_u64()), (200, Some(5)), "{v}");
    // The engine API's change itself is published (and not
    // reverted on the master by a re-sync).
    for n in &nodes {
        assert!(n.state().metadata.customs.get("ingest").is_some_and(|c| c.to_string().contains("drift-2")), "the pipeline was not published to [{}]", n.local.name);
    }
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_ingest/pipeline/drift-2", None).await;
    assert_eq!(st, 200, "the master's own pipeline was reverted: {v}");
    let (st, v) = call(&c, http[other], reqwest::Method::GET, "/_ingest/pipeline/drift-2", None).await;
    assert_eq!(st, 200, "the pipeline did not reach the other node: {v}");
}

/// An ILM shrink goes through the cluster's resize (documents
/// copied from the source copies), waits for the copies, and deletes the
/// source only once the target holds its documents. Lifecycle passes run
/// on the master's metadata copy, which holds no documents.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ilm_shrink_preserves_the_documents() {
    let node = single_node("ilm-shrink").await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/shrinker", Some(json!({"policy": {"phases": {"warm": {"min_age": "0ms", "actions": {"shrink": {"number_of_shards": 1}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/to-shrink", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.lifecycle.name": "shrinker"}}))).await;
    ok(st, &v);
    let mut bulk = String::new();
    for i in 0..40 {
        bulk.push_str(&format!("{{\"index\":{{\"_index\":\"to-shrink\",\"_id\":\"{i}\"}}}}\n{{\"n\":{i}}}\n"));
    }
    let r = c.post(format!("http://127.0.0.1:{http}/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let deadline = Instant::now() + Duration::from_secs(120);
    let target = loop {
        let st = node.state();
        let shrunk = st.metadata.indices.keys().find(|n| n.starts_with("shrink-") && n.ends_with("-to-shrink")).cloned();
        if let Some(t) = shrunk.filter(|_| !st.metadata.indices.contains_key("to-shrink")) {
            break t;
        }
        if Instant::now() >= deadline {
            let (_, explain) = call(&c, http, reqwest::Method::GET, "/to-shrink,shrink-*/_ilm/explain", None).await;
            panic!("the shrink did not complete; explain: {explain}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/{target}/_count"), None).await;
    assert_eq!((st, v["count"].as_u64()), (200, Some(40)), "the shrunken index lost documents: {v}");
    // The source name now resolves to the target.
    let (st, v) = call(&c, http, reqwest::Method::GET, "/to-shrink/_count", None).await;
    assert_eq!((st, v["count"].as_u64()), (200, Some(40)), "{v}");
    assert_eq!(node.state().metadata.indices[&target].number_of_shards, 1);
}

/// Engine APIs changing customs concurrently (pipelines
/// and lifecycle policies, each published by its own kept job): a commit
/// applied while another API is in flight never wipes that API's change,
/// and every acknowledged change reaches every node.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_engine_api_custom_changes_are_all_published() {
    let nodes = two_nodes("api-customs", &[("cluster.auto_shrink_voting_configuration", "false")]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let c = reqwest::Client::new();
    let mut calls = Vec::new();
    for i in 0..20 {
        let c1 = c.clone();
        calls.push(tokio::spawn(async move {
            call(&c1, http[master], reqwest::Method::PUT, &format!("/_ingest/pipeline/conc-{i}"), Some(json!({"processors": [{"set": {"field": "f", "value": i}}]}))).await
        }));
        let c2 = c.clone();
        calls.push(tokio::spawn(async move {
            call(&c2, http[master], reqwest::Method::PUT, &format!("/_ilm/policy/conc-{i}"), Some(json!({"policy": {"phases": {"hot": {"actions": {"set_priority": {"priority": i}}}}}}))).await
        }));
    }
    for r in calls {
        let (st, v) = r.await.unwrap();
        assert!((200..300).contains(&st), "{st} {v}");
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    for n in [master, 1 - master] {
        for i in 0..20 {
            loop {
                let (p, _) = call(&c, http[n], reqwest::Method::GET, &format!("/_ingest/pipeline/conc-{i}"), None).await;
                let (q, _) = call(&c, http[n], reqwest::Method::GET, &format!("/_ilm/policy/conc-{i}"), None).await;
                if p == 200 && q == 200 {
                    break;
                }
                assert!(Instant::now() < deadline, "pipeline / policy conc-{i} missing on node {n} ({p}, {q})");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Public API contract: self aliases cannot substitute a local leader before
/// or after a mapping change (cross-cluster replication is not supported).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unsupported_ccr_self_alias_does_not_follow_mapping_changes() {
    let node = spawn_node(settings("n0", "ccr-mapping", &[], &[], &[("xpack.license.self_generated.type", "trial")])).await;
    let (http, transport) = (ports::http_port(&node), ports::transport_port(&node));
    let _ = &node;
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let ok = |st: u16, v: &Value| assert!((200..300).contains(&st), "{st} {v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.remote.self.seeds": [format!("127.0.0.1:{transport}")]}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/leader", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"a": {"type": "keyword"}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/leader/_doc/1?refresh=true", Some(json!({"a": "x"}))).await;
    ok(st, &v);
    for changed in [false, true] {
        if changed {
            let (st, v) = call(&c, http, reqwest::Method::PUT, "/leader/_doc/2?refresh=true", Some(json!({"a": "y", "b": 5}))).await;
            ok(st, &v);
        }
        let (st, v) = call(&c, http, reqwest::Method::PUT, "/follower/_ccr/follow?wait_for_active_shards=1", Some(json!({"remote_cluster": "self", "leader_index": "leader"}))).await;
        assert_eq!(st, 502, "{v}");
        assert_eq!(v["error"]["type"], "connect_transport_exception");
        assert_eq!(call(&c, http, reqwest::Method::GET, "/follower", None).await.0, 404);
        assert!(!node.state().metadata.indices.contains_key("follower"));
    }
    let (st, value) = call(&c, http, reqwest::Method::GET, "/leader/_count", None).await;
    assert_eq!(st, 200, "{value}");
    assert_eq!(value["count"], 2);
    assert!(node.state().metadata.indices["leader"].mappings["properties"].get("b").is_some());
}

/// A change made through a node that is not the master is visible on that
/// node as soon as the request answers: the master acknowledges once every
/// node applied the new state, and the coordinating node waits until it
/// applied it itself (ES's acknowledged requests).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_change_through_a_non_master_is_visible_there_at_once() {
    let nodes = two_nodes("ack-visible", &[("cluster.auto_shrink_voting_configuration", "false")]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let other = 1 - master;
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http[other], reqwest::Method::PUT, "/ackv", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..20 {
        let replicas = i % 2;
        let (st, v) = call(&c, http[other], reqwest::Method::PUT, "/ackv/_settings", Some(json!({"index": {"number_of_replicas": replicas}}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["acknowledged"], json!(true), "{v}");
        let (st, v) = call(&c, http[other], reqwest::Method::GET, "/ackv/_settings?flat_settings=true", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["ackv"]["settings"]["index.number_of_replicas"], json!(replicas.to_string()), "round {i}: the change was not visible on the coordinating node: {v}");
        let (st, v) = call(&c, http[other], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": (2 + i).to_string()}}))).await;
        assert_eq!(st, 200, "{v}");
        let (st, v) = call(&c, http[other], reqwest::Method::GET, "/_cluster/settings?flat_settings=true", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["persistent"]["cluster.routing.allocation.node_concurrent_recoveries"], json!((2 + i).to_string()), "round {i}: {v}");
        // The routing table reflects it too (a replica row appears or goes).
        let (st, v) = call(&c, http[other], reqwest::Method::GET, "/_cat/shards/ackv?format=json", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v.as_array().map(Vec::len), Some(1 + replicas), "round {i}: {v}");
    }
    let _ = master;
}

/// Four real network nodes in two independent clusters. This is a rejection
/// contract, not a CCS/CCR parity test: a configured seed must never make local
/// documents masquerade as remote documents while remote transport is absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn unsupported_remote_transport_never_reads_or_follows_local_indices() {
    let license = [("xpack.license.self_generated.type", "trial")];
    let _a = two_nodes("remote-reject-a", &license).await;
    let _b = two_nodes("remote-reject-b", &license).await;
    let (http_a, transport_a) = (http_ports(&_a), transport_ports(&_a));
    let (http_b, transport_b) = (http_ports(&_b), transport_ports(&_b));
    let client = reqwest::Client::new();
    let (_, a) = call(&client, http_a[0], reqwest::Method::GET, "/", None).await;
    let (_, b) = call(&client, http_b[0], reqwest::Method::GET, "/", None).await;
    assert!(a["cluster_uuid"].as_str().is_some_and(|s| !s.is_empty() && s != "_na_"), "{a}");
    assert!(b["cluster_uuid"].as_str().is_some_and(|s| !s.is_empty() && s != "_na_"), "{b}");
    assert_ne!(a["cluster_uuid"], b["cluster_uuid"]);
    for (http, marker) in [(http_a[0], "local-only"), (http_b[0], "remote-only")] {
        let (status, body) = call(&client, http, reqwest::Method::PUT, "/remote-contract", Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 1}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = call(&client, http, reqwest::Method::PUT, "/remote-contract/_doc/1?refresh=true&wait_for_active_shards=all", Some(json!({"marker": marker}))).await;
        assert_eq!(status, 201, "{body}");
    }
    let (status, body) = call(&client, http_a[0], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {
        "cluster.remote.actual.seeds": [format!("127.0.0.1:{}", transport_b[0])],
        "cluster.remote.actual.skip_unavailable": false,
        "cluster.remote.self.seeds": [format!("127.0.0.1:{}", transport_a[0])],
        "cluster.remote.self.skip_unavailable": false,
        "cluster.remote.optional.seeds": [format!("127.0.0.1:{}", transport_b[1])],
        "cluster.remote.optional.skip_unavailable": true
    }}))).await;
    assert_eq!(status, 200, "{body}");
    // Probe both local coordinators, including the node that did not accept the
    // settings update. Neither may infer connectivity from cluster membership.
    for http in http_a {
        let (_, info) = call(&client, http, reqwest::Method::GET, "/_remote/info", None).await;
        for alias in ["actual", "self", "optional"] {
            assert_eq!(info[alias]["connected"], false, "{info}");
            assert_eq!(info[alias]["num_nodes_connected"], 0, "{info}");
        }
        for alias in ["actual", "self"] {
            let (status, body) = call(&client, http, reqwest::Method::POST, &format!("/remote-contract,{alias}:remote-contract/_search"), Some(json!({"query": {"match_all": {}}}))).await;
            assert_eq!(status, 502, "{body}");
            assert_eq!(body["error"]["type"], "connect_transport_exception", "{body}");
            assert!(body.get("hits").is_none(), "must not return local partial success: {body}");
            for follower in ["new-follower", "remote-contract"] {
                let (status, body) = call(&client, http, reqwest::Method::PUT, &format!("/{follower}/_ccr/follow"), Some(json!({"remote_cluster": alias, "leader_index": "remote-contract"}))).await;
                assert_eq!(status, 502, "{body}");
                assert_eq!(body["error"]["type"], "connect_transport_exception", "{body}");
            }
        }
        let (status, body) = call(&client, http, reqwest::Method::POST, "/remote-contract,optional:remote-contract/_search", Some(json!({"query": {"match_all": {}}}))).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["hits"]["total"]["value"], 1, "{body}");
        assert_eq!(body["hits"]["hits"][0]["_source"]["marker"], "local-only", "{body}");
        assert_eq!(body["_clusters"]["successful"], 1, "{body}");
        assert_eq!(body["_clusters"]["skipped"], 1, "{body}");
        assert_eq!(call(&client, http, reqwest::Method::GET, "/new-follower", None).await.0, 404);
        let (status, body) = call(&client, http, reqwest::Method::GET, "/remote-contract/_doc/1", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["_source"]["marker"], "local-only", "{body}");
        let (_, mapping) = call(&client, http, reqwest::Method::GET, "/remote-contract/_settings", None).await;
        assert_ne!(mapping["remote-contract"]["settings"]["index"]["xpack"]["ccr"]["following_index"], "true", "{mapping}");
    }
    let (_, remote) = call(&client, http_b[0], reqwest::Method::GET, "/remote-contract/_doc/1", None).await;
    assert_eq!(remote["_source"]["marker"], "remote-only", "{remote}");
}

/// Managed resources must be committed at the metadata acknowledgement
/// boundary, rather than relying on the ten-second periodic reconcile.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn registry_resources_are_published_before_health_barrier_and_auto_creation() {
    let masters=["registry-0","registry-1"];
    let mut nodes: Vec<Arc<ClusterNode>>=Vec::new();
    for i in 0..2 {
        let seeds: Vec<u16>=nodes.iter().map(|n| ports::transport_port(n)).collect();
        nodes.push(spawn_node(settings(masters[i],"registry-ack",&seeds,&masters,
            &[("stack.templates.enabled","true"),("xpack.apm_data.enabled","true")])).await);
    }
    let http=http_ports(&nodes);
    let client=reqwest::Client::new();
    let deadline=Instant::now()+Duration::from_secs(60);
    while nodes.iter().any(|n|n.state().master_node_id.is_none()) {
        assert!(Instant::now()<deadline,"cluster did not elect a master");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let peer=if nodes[0].is_master(){http[1]}else{http[0]};
    let (status,value)=call(&client,peer,reqwest::Method::GET,"/_cluster/health?wait_for_events=languid&timeout=30s",None).await;
    assert_eq!(status,200,"{value}");
    for port in http {
        let (status,value)=call(&client,port,reqwest::Method::GET,"/_component_template/apm@mappings",None).await;
        assert_eq!(status,200,"{value}");
        assert_eq!(value["component_templates"][0]["component_template"]["template"]["mappings"]["properties"]["timestamp"]["properties"]["us"]["type"],"long");
    }
    for health_barrier in [false,true] {
        let (status,value)=call(&client,peer,reqwest::Method::DELETE,"/_index_template/traces-apm@template",None).await;
        assert_eq!(status,200,"{value}");
        if health_barrier {
            let (status,value)=call(&client,peer,reqwest::Method::GET,"/_cluster/health?wait_for_events=languid",None).await;
            assert_eq!(status,200,"{value}");
            let (status,value)=call(&client,peer,reqwest::Method::GET,"/_index_template/traces-apm@template",None).await;
            assert_eq!(status,200,"{value}");
        }
        let name=if health_barrier{"traces-apm-after-barrier"}else{"traces-apm-before-timer"};
        let (status,value)=call(&client,peer,reqwest::Method::POST,&format!("/{name}/_doc?refresh=true"),Some(json!({"@timestamp":"2026-10-24T00:00:00Z","timestamp":{"us":1772469840000000.0}}))).await;
        assert_eq!(status,201,"{value}");
        let (status,value)=call(&client,peer,reqwest::Method::GET,&format!("/{name}/_mapping"),None).await;
        assert_eq!(status,200,"{value}");
        let mapping=value.as_object().unwrap().values().next().unwrap();
        assert_eq!(mapping["mappings"]["properties"]["timestamp"]["properties"]["us"]["type"],"long","{value}");
    }
}

/// Collection acknowledgement must include the data-stream custom on every
/// node, not just the backing index created on the master.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn analytics_collection_acknowledgement_publishes_stream_on_both_nodes() {
    let _nodes = two_nodes("analytics-ack",
        &[("xpack.license.self_generated.type", "trial")]).await;
    let http = http_ports(&_nodes);
    let client = reqwest::Client::new();
    let path = "/_application/analytics/ack-test";
    let (status, body) = call(&client, http[1], reqwest::Method::PUT, path, None).await;
    assert_eq!(status, 201, "{body}");
    assert_eq!(body["acknowledged"], true, "{body}");
    for port in http {
        let (status, body) = call(&client, port, reqwest::Method::GET, path, None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["ack-test"]["event_data_stream"]["name"], "behavioral_analytics-events-ack-test");
        let (status, body) = call(&client, port, reqwest::Method::GET, "/_xpack/usage", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["enterprise_search"]["analytics_collections"]["count"], 1, "{body}");
    }
    let (status, body) = call(&client, http[0], reqwest::Method::DELETE, path, None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["acknowledged"], true, "{body}");
    for port in http {
        let (status, body) = call(&client, port, reqwest::Method::GET, path, None).await;
        assert_eq!(status, 404, "{body}");
    }
}
