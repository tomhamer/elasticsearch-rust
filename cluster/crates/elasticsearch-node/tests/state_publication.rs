//! Cluster state publication over the binary wire with diffs (Step 3 of
//! the coordination redesign): a node that joins gets the full state, later
//! publications reach it as diffs, and every node ends up with exactly the
//! master's state.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn settings(name: &str, seeds: &[u16]) -> Settings {
    settings_for("state-publication", name, seeds, "n0")
}

/// A node on OS-assigned ports (read back with [`ports::http_port`]) that
/// seeds from the already-started nodes' transport ports `seeds`.
fn settings_for(cluster: &str, name: &str, seeds: &[u16], initial_master: &str) -> Settings {
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
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!([initial_master]));
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

async fn converged(a: &ClusterNode, b: &ClusterNode) -> bool {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        let (sa, sb) = (a.state(), b.state());
        if sa.version == sb.version && sa.master_node_id.is_some() {
            return *sa == *sb;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn joining_node_gets_the_full_state_and_diffs_keep_it_identical() {
    let master = spawn_node(settings("n0", &[])).await;
    let mut http = [ports::http_port(&master), 0];
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !master.is_master() {
        assert!(Instant::now() < deadline, "n0 did not become master");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // State the joining node has never seen.
    for i in 0..10 {
        let r = c.put(format!("http://127.0.0.1:{}/pre-{i}", http[0])).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"f": {"type": "keyword"}}}})).send().await.unwrap();
        assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    }
    let joiner = spawn_node(settings("n1", &[ports::transport_port(&master)])).await;
    http[1] = ports::http_port(&joiner);
    let deadline = Instant::now() + Duration::from_secs(60);
    while joiner.state().nodes.len() < 2 || master.state().nodes.len() < 2 {
        assert!(Instant::now() < deadline, "n1 did not join");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(converged(&master, &joiner).await, "the joining node holds the master's full state");
    // Incremental changes (published as diffs to the follower).
    let r = c.put(format!("http://127.0.0.1:{}/pre-3/_mapping", http[0])).json(&json!({"properties": {"g": {"type": "long"}}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    let r = c.delete(format!("http://127.0.0.1:{}/pre-4", http[0])).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    let r = c.put(format!("http://127.0.0.1:{}/_cluster/settings", http[1])).json(&json!({"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": 3}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    assert!(converged(&master, &joiner).await, "diff publications rebuild exactly the master's state");
    let stats = master.coord().wire_stats();
    assert!(stats.diffs.load(Ordering::Relaxed) > 0, "publications to the follower went as diffs");
    assert!(stats.full_states.load(Ordering::Relaxed) > 0, "the joining node got a full state");
    assert_eq!(stats.downgrades.load(Ordering::Relaxed), 0, "no legacy fallback between upgraded nodes");
    let st = joiner.state();
    assert_eq!(st.metadata.indices["pre-3"].mappings["properties"]["g"]["type"], "long");
    assert!(!st.metadata.indices.contains_key("pre-4"));
    // The follower's engine follows too.
    let deadline = Instant::now() + Duration::from_secs(10);
    while joiner.engine.get_index("pre-4").is_some() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(joiner.engine.get_index("pre-4").is_none());
    assert_eq!(joiner.engine.get_index("pre-3").expect("pre-3 mirrored").mapping_json()["properties"]["g"]["type"], "long");
}

/// Rolling upgrade: a peer without the binary coordination action (an older
/// node only registers the JSON one). The upgraded master answers its
/// `action_not_found` by re-sending in the legacy format and keeps using it
/// for that peer (full states, no diffs): publications, fault detection and
/// metadata changes keep working and both nodes hold the same state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_without_the_binary_wire_format_gets_legacy_full_states() {
    let master = spawn_node(settings_for("mixed-wire", "n0", &[], "n0")).await;
    let http = [ports::http_port(&master)];
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !master.is_master() {
        assert!(Instant::now() < deadline, "n0 did not become master");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let old = spawn_node(settings_for("mixed-wire", "n1", &[ports::transport_port(&master)], "n0")).await;
    // The "old" node: only the legacy JSON coordination action.
    assert!(old.transport.unregister_handler(elasticsearch_node::coord::ACTION_V2));
    let deadline = Instant::now() + Duration::from_secs(60);
    while old.state().nodes.len() < 2 || master.state().nodes.len() < 2 {
        assert!(Instant::now() < deadline, "n1 did not join over the legacy format");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    for i in 0..5 {
        let r = c.put(format!("http://127.0.0.1:{}/mixed-{i}", http[0])).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"f": {"type": "keyword"}}}})).send().await.unwrap();
        assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    }
    let r = c.put(format!("http://127.0.0.1:{}/mixed-2/_mapping", http[0])).json(&json!({"properties": {"g": {"type": "long"}}})).send().await.unwrap();
    assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    assert!(converged(&master, &old).await, "the legacy-format peer holds the master's state");
    let stats = master.coord().wire_stats();
    assert!(stats.downgrades.load(Ordering::Relaxed) > 0, "the master fell back to the legacy format");
    assert!(stats.legacy_messages.load(Ordering::Relaxed) > 0);
    assert_eq!(old.state().metadata.indices["mixed-2"].mappings["properties"]["g"]["type"], "long");
    // Fault detection over the legacy format keeps both nodes in the cluster.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(master.state().nodes.len(), 2, "the legacy-format peer was not removed");
    assert!(master.is_master());
}

/// Rolling upgrade, the other direction: the master is an older node (only
/// the legacy JSON action, sends only JSON) and an upgraded node joins. The
/// upgraded node's binary messages are refused (`action_not_found`) and
/// re-sent in the legacy format at once; it applies the master's legacy full
/// states and stays a member.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_upgraded_node_follows_a_legacy_master() {
    let legacy = spawn_node(settings_for("legacy-master", "n0", &[], "n0")).await;
    legacy.faults.legacy_wire.store(true, Ordering::SeqCst);
    assert!(legacy.transport.unregister_handler(elasticsearch_node::coord::ACTION_V2));
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !legacy.is_master() {
        assert!(Instant::now() < deadline, "n0 did not become master");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let upgraded = spawn_node(settings_for("legacy-master", "n1", &[ports::transport_port(&legacy)], "n0")).await;
    let http = [ports::http_port(&legacy), ports::http_port(&upgraded)];
    let deadline = Instant::now() + Duration::from_secs(60);
    while upgraded.state().nodes.len() < 2 || legacy.state().nodes.len() < 2 {
        assert!(Instant::now() < deadline, "the upgraded node did not join the legacy master");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    for i in 0..3 {
        let r = c.put(format!("http://127.0.0.1:{}/legacy-{i}", http[1])).json(&json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).send().await.unwrap();
        assert!(r.status().is_success(), "{}", r.text().await.unwrap());
    }
    assert!(converged(&legacy, &upgraded).await, "the upgraded node holds the legacy master's state");
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(legacy.state().nodes.len(), 2, "fault detection over the legacy format keeps the upgraded node");
    let up = upgraded.coord().wire_stats();
    assert!(up.downgrades.load(Ordering::Relaxed) > 0, "the upgraded node fell back to the legacy format for the master");
    let old = legacy.coord().wire_stats();
    assert_eq!(old.diffs.load(Ordering::Relaxed), 0, "a legacy master sends no diffs");
    assert!(old.full_states.load(Ordering::Relaxed) > 0);
}
