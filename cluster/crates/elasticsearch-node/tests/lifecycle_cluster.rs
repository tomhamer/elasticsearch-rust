//! Lifecycle management on a cluster, where ILM and data stream lifecycle
//! passes run on the master's metadata copy:
//!
//! * ILM `wait_for_snapshot` resolves the SLM policy and its last success
//!   from the cluster's SLM metadata (`_slm` is a cluster API);
//! * ILM `check-allocation` judges the routing table, not the replica count;
//! * data stream lifecycle downsampling runs through the cluster's
//!   `_downsample` and swaps the backing index.
//!
//! Nodes run in-process (on the test runtime).

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    put("path.repo", json!(dir.join("repo").to_string_lossy()));
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

async fn single_node(cluster: &str, extra: &[(&str, &str)]) -> Arc<ClusterNode> {
    let node = spawn_node(settings("n0", cluster, &[], &[], extra)).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while call(&c, http, reqwest::Method::GET, "/_cluster/health?wait_for_status=green&timeout=1s", None).await.0 != 200 {
        assert!(Instant::now() < deadline, "node did not start");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    node
}

/// Two master-eligible data nodes with `node.attr.box` `a` and `b`.
async fn two_nodes(cluster: &str) -> [Arc<ClusterNode>; 2] {
    let a = spawn_node(settings("n0", cluster, &[], &["n0", "n1"], &[("node.attr.box", "a")])).await;
    let b = spawn_node(settings("n1", cluster, &[ports::transport_port(&a)], &["n0", "n1"], &[("node.attr.box", "b")])).await;
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

fn ok(st: u16, v: &Value) {
    assert!((200..300).contains(&st), "{st} {v}");
}

async fn explain(c: &reqwest::Client, http: u16, index: &str) -> Value {
    let (_, v) = call(c, http, reqwest::Method::GET, &format!("/{index}/_ilm/explain"), None).await;
    v["indices"][index].clone()
}

/// An SLM policy created through the cluster API can be referenced by an
/// ILM `wait_for_snapshot`; the index waits until a snapshot of that
/// policy that started after the wait began succeeds, and only then moves
/// on to its deletion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ilm_wait_for_snapshot_follows_the_cluster_slm_policy() {
    let repo_root = evidence::scratch_dir("ilm-slm-repo");
    let repo_path = repo_root.to_string_lossy().to_string();
    let node = single_node("ilm-slm", &[("path.repo", repo_path.as_str())]).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let repo = repo_root.join("backups");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_snapshot/backups", Some(json!({"type": "fs", "settings": {"location": repo.to_string_lossy()}}))).await;
    ok(st, &v);
    // ILM refuses a policy no SLM policy backs.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/needs-backup", Some(json!({"policy": {"phases": {"delete": {"min_age": "0ms", "actions": {"wait_for_snapshot": {"policy": "nightly"}, "delete": {}}}}}}))).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap_or_default().contains("no such snapshot lifecycle policy [nightly]"), "{v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_slm/policy/nightly", Some(json!({"schedule": "0 30 1 1 1 ? 2099", "name": "<nightly-{now/d}>", "repository": "backups", "config": {"indices": ["*"]}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/needs-backup", Some(json!({"policy": {"phases": {"delete": {"min_age": "0ms", "actions": {"wait_for_snapshot": {"policy": "nightly"}, "delete": {}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/backed-up", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "needs-backup"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/backed-up/_doc/1?refresh=true", Some(json!({"n": 1}))).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let e = explain(&c, http, "backed-up").await;
        if e["step"] == "wait-for-snapshot" && e["step_info"]["message"].as_str().is_some_and(|m| m.starts_with("waiting for policy 'nightly' to be executed")) {
            break;
        }
        assert!(Instant::now() < deadline, "never reached wait-for-snapshot: {e}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // Several polls without a snapshot: still waiting, nothing deleted.
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(node.state().metadata.indices.contains_key("backed-up"), "deleted without a snapshot");
    assert_eq!(explain(&c, http, "backed-up").await["step"], "wait-for-snapshot");
    let (st, v) = call(&c, http, reqwest::Method::POST, "/_slm/policy/nightly/_execute", None).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, p) = call(&c, http, reqwest::Method::GET, "/_slm/policy/nightly", None).await;
        let success = p["nightly"]["last_success"].is_object();
        if !node.state().metadata.indices.contains_key("backed-up") {
            assert!(success, "deleted before the snapshot succeeded: {p}");
            break;
        }
        if Instant::now() >= deadline {
            panic!("the successful snapshot did not release the wait; slm: {p}; explain: {}", explain(&c, http, "backed-up").await);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Review finding 118: `wait_for_snapshot` checks the snapshot itself. A
/// successful SLM snapshot that does not include the waiting index fails
/// the step (retryable ERROR with ES's reason), never deleted; once the
/// policy's next success includes it, it moves on to its deletion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ilm_wait_for_snapshot_requires_the_index_in_the_snapshot() {
    let repo_root = evidence::scratch_dir("ilm-slm-membership");
    let repo_path = repo_root.to_string_lossy().to_string();
    let node = single_node("ilm-slm-membership", &[("path.repo", repo_path.as_str())]).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let repo = repo_root.join("backups");
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_snapshot/backups", Some(json!({"type": "fs", "settings": {"location": repo.to_string_lossy()}}))).await;
    ok(st, &v);
    let slm = |indices: Value| json!({"schedule": "0 30 1 1 1 ? 2099", "name": "<nightly-{now/d}>", "repository": "backups", "config": {"indices": indices}});
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_slm/policy/nightly", Some(slm(json!(["other"])))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/needs-backup", Some(json!({"policy": {"phases": {"delete": {"min_age": "0ms", "actions": {"wait_for_snapshot": {"policy": "nightly"}, "delete": {}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/other", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/kept", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "needs-backup"}}))).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(60);
    while explain(&c, http, "kept").await["step"] != "wait-for-snapshot" {
        assert!(Instant::now() < deadline, "never reached wait-for-snapshot");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // A successful snapshot of [other] only.
    let (st, v) = call(&c, http, reqwest::Method::POST, "/_slm/policy/nightly/_execute", None).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let e = explain(&c, http, "kept").await;
        assert!(node.state().metadata.indices.contains_key("kept"), "deleted after a snapshot without it");
        if e["step"] == "ERROR" && e["failed_step"] == "wait-for-snapshot" && e["step_info"]["reason"].as_str().is_some_and(|m| m.contains("does not include index 'kept'")) {
            break;
        }
        assert!(Instant::now() < deadline, "no membership diagnostic: {e}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(node.state().metadata.indices.contains_key("kept"), "deleted after a snapshot without it");
    let e = explain(&c, http, "kept").await;
    assert!(e["step"] == "wait-for-snapshot" || (e["step"] == "ERROR" && e["failed_step"] == "wait-for-snapshot"), "{e}");
    // The policy's next success includes it: released, then deleted.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_slm/policy/nightly", Some(slm(json!(["other", "kept"])))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::POST, "/_slm/policy/nightly/_execute", None).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(60);
    while node.state().metadata.indices.contains_key("kept") {
        assert!(Instant::now() < deadline, "a snapshot including [kept] did not release it: {}", explain(&c, http, "kept").await);
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// `check-allocation` on a replicated index reads the routing table: a
/// green index with one replica advances (and its later delete phase
/// runs); one whose replica cannot satisfy the requested filter waits
/// until the filter can be met.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ilm_check_allocation_judges_the_routing_table() {
    let nodes = two_nodes("ilm-alloc").await;
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let http = ports::http_port(&nodes[master]);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    // Allocated as requested (one copy per node): advances to deletion.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/spread", Some(json!({"policy": {"phases": {
        "warm": {"min_age": "0ms", "actions": {"allocate": {"number_of_replicas": 1, "include": {"box": "a,b"}}}},
        "delete": {"min_age": "0ms", "actions": {"delete": {}}}}}}))).await;
    ok(st, &v);
    // A filter only one node meets: the replica cannot be placed.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/pinned", Some(json!({"policy": {"phases": {
        "warm": {"min_age": "0ms", "actions": {"allocate": {"require": {"box": "a"}}}},
        "delete": {"min_age": "0ms", "actions": {"delete": {}}}}}}))).await;
    ok(st, &v);
    for (index, policy) in [("replicated", "spread"), ("pinned-idx", "pinned")] {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1, "index.lifecycle.name": policy}}))).await;
        ok(st, &v);
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    while nodes[master].state().metadata.indices.contains_key("replicated") {
        if Instant::now() >= deadline {
            panic!("a green replicated index never left check-allocation: {}", explain(&c, http, "replicated").await);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // The pinned index waits in check-allocation (its replica cannot be
    // on the only `box: a` node, which holds the primary).
    let deadline = Instant::now() + Duration::from_secs(60);
    let waiting = loop {
        let e = explain(&c, http, "pinned-idx").await;
        let message = e["step_info"]["message"].as_str().unwrap_or_default();
        let placed = message == "Waiting for all shard copies to be active" || message.starts_with("Waiting for [1] shards to be allocated to nodes matching the given filters");
        if e["step"] == "check-allocation" && placed {
            break e;
        }
        assert!(Instant::now() < deadline, "never reached check-allocation: {e}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    assert_eq!(waiting["action"], "allocate", "{waiting}");
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(nodes[master].state().metadata.indices.contains_key("pinned-idx"), "advanced with a misplaced or unassigned replica");
    assert_eq!(explain(&c, http, "pinned-idx").await["step"], "check-allocation");
    // Once the filter can be met (both nodes are `box: a`-compatible), the
    // index advances and its retention runs.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/pinned-idx/_settings", Some(json!({"index.routing.allocation.require.box": "a,b"}))).await;
    ok(st, &v);
    let deadline = Instant::now() + Duration::from_secs(90);
    while nodes[master].state().metadata.indices.contains_key("pinned-idx") {
        if Instant::now() >= deadline {
            panic!("the index never advanced after its copies were placed: {}", explain(&c, http, "pinned-idx").await);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Data stream lifecycle downsampling on a cluster: the backing index is
/// downsampled through the cluster (documents read from the shard copies,
/// not the master's empty metadata copy), the downsample takes its place
/// and the source is deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dlm_downsampling_runs_through_the_cluster() {
    let node = single_node("dlm-downsample", &[]).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_index_template/tsds-test", Some(json!({"index_patterns": ["tsmetrics-*"], "data_stream": {}, "priority": 500, "template": {
        "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true}, "cpu": {"type": "double", "time_series_metric": "gauge"}}},
        "lifecycle": {"downsampling": [{"after": "1d", "fixed_interval": "1h"}]}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_stream.past_tsdb_index_creation_enabled": true}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_data_stream/tsmetrics-a", None).await;
    ok(st, &v);
    let now = chrono::Utc::now().timestamp_millis();
    // Five hours ago: a past backing index (before the write index's
    // window, so past its time bounds) receives the documents.
    let hour = now - now.rem_euclid(3_600_000) - 5 * 3_600_000;
    let mut bulk = String::new();
    for i in 0..40i64 {
        let ts = chrono::DateTime::from_timestamp_millis(hour + i * 60_000).unwrap().format("%Y-%m-%dT%H:%M:%S%.3fZ");
        bulk.push_str(&format!("{{\"create\":{{}}}}\n{{\"@timestamp\":\"{ts}\",\"host\":\"h{}\",\"cpu\":{}}}\n", i % 4, i));
    }
    let r = c.post(format!("http://127.0.0.1:{http}/tsmetrics-a/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["errors"], false, "{body}");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/tsmetrics-a", None).await;
    ok(st, &v);
    let indices = v["data_streams"][0]["indices"].as_array().unwrap().clone();
    assert_eq!(indices.len(), 2, "a past backing index beside the write index: {v}");
    let first = indices[0]["index_name"].as_str().unwrap().to_string();
    let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/{first}/_count"), None).await;
    assert_eq!((st, v["count"].as_u64()), (200, Some(40)), "{v}");
    // Originated two days ago: the round is due.
    let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{first}/_settings"), Some(json!({
        "index.lifecycle.origination_date": (now - 2 * 86_400_000).to_string()}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_streams.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let target = format!("downsample-1h-{first}");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let st = node.state();
        let swapped = !st.metadata.indices.contains_key(&first) && st.metadata.indices.contains_key(&target);
        if swapped {
            break;
        }
        if Instant::now() >= deadline {
            let (_, e) = call(&c, http, reqwest::Method::GET, &format!("/{first},{target}/_lifecycle/explain"), None).await;
            panic!("the backing index was never downsampled and swapped; explain: {e}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/tsmetrics-a", None).await;
    ok(st, &v);
    assert_eq!(v["data_streams"][0]["indices"][0]["index_name"], json!(target), "{v}");
    // One document per host and hour, with the source's metrics.
    let (st, v) = call(&c, http, reqwest::Method::POST, &format!("/{target}/_search"), Some(json!({"size": 0, "track_total_hits": true,
        "aggs": {"sum": {"sum": {"field": "cpu"}}, "max": {"max": {"field": "cpu"}}, "min": {"min": {"field": "cpu"}}}}))).await;
    ok(st, &v);
    assert_eq!(v["hits"]["total"]["value"], 4, "{v}");
    assert_eq!(v["aggregations"]["sum"]["value"], json!((0..40).sum::<i64>() as f64), "{v}");
    assert_eq!(v["aggregations"]["max"]["value"], json!(39.0), "{v}");
    assert_eq!(v["aggregations"]["min"]["value"], json!(0.0), "{v}");
    // The target's private status reached `success` (through the master,
    // as `_system`); users cannot set it.
    let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/{target}/_settings?flat_settings=true"), None).await;
    ok(st, &v);
    assert_eq!(v[target.as_str()]["settings"]["index.downsample.status"], "success", "{v}");
    let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{target}/_settings"), Some(json!({"index.downsample.status": "started"}))).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["reason"], "can not update private setting [index.downsample.status]; this setting is managed by Elasticsearch", "{v}");
}

/// Data stream lifecycle `frozen_after` through the cluster (ES 9.5.5
/// `dlm-frozen-transition`): a backing index past it is marked with the
/// default repository, write blocked, cloned without replicas
/// (`dlm-clone-*`), force-merged, snapshotted (`dlm-frozen-dlm-clone-*`),
/// mounted as a partial searchable snapshot (`dlm-frozen-*`) and swapped
/// into the data stream; the clone and the original are deleted. The
/// frozen index answers searches with every document. Without a default
/// repository (first), nothing is converted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dlm_frozen_after_converts_through_the_cluster() {
    let repo_root = evidence::scratch_dir("dlm-frozen-repo");
    let repo_path = repo_root.to_string_lossy().to_string();
    let node = single_node("dlm-frozen", &[("path.repo", repo_path.as_str()), ("dlm.frozen.transition.poll_interval", "1s")]).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_index_template/frozen", Some(json!({"index_patterns": ["frz-*"], "data_stream": {}, "priority": 500, "template": {
        "settings": {"number_of_shards": 2, "number_of_replicas": 1},
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}}},
        "lifecycle": {"frozen_after": "1d", "data_retention": "30d"}}}))).await;
    ok(st, &v);
    let mut bulk = String::new();
    for i in 0..30 {
        bulk.push_str(&format!("{{\"create\":{{}}}}\n{{\"@timestamp\":\"2026-01-01T00:00:{:02}Z\",\"n\":{i}}}\n", i % 60));
    }
    let r = c.post(format!("http://127.0.0.1:{http}/frz-a/_bulk?refresh=true")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["errors"], false, "{body}");
    let (st, v) = call(&c, http, reqwest::Method::POST, "/frz-a/_rollover", None).await;
    ok(st, &v);
    let (_, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/frz-a", None).await;
    let first = v["data_streams"][0]["indices"][0]["index_name"].as_str().unwrap().to_string();
    let now = chrono::Utc::now().timestamp_millis();
    let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/{first}/_settings"), Some(json!({"index.lifecycle.origination_date": (now - 2 * 86_400_000).to_string()}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_streams.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    // No default repository: the index stays as it is.
    tokio::time::sleep(Duration::from_secs(4)).await;
    let st = node.state();
    assert!(st.metadata.indices.contains_key(&first) && !st.metadata.indices.keys().any(|k| k.starts_with("dlm-")), "{:?}", st.metadata.indices.keys().collect::<Vec<_>>());
    let (_, s) = call(&c, http, reqwest::Method::GET, &format!("/{first}/_settings/index.blocks.write?flat_settings=true"), None).await;
    assert!(s[first.as_str()]["settings"].get("index.blocks.write").is_none(), "{s}");
    drop(st);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_snapshot/frozen-repo", Some(json!({"type": "fs", "settings": {"location": repo_root.join("frozen").to_string_lossy()}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"repositories.default_repository": "frozen-repo"}}))).await;
    ok(st, &v);
    let mounted = format!("dlm-frozen-{first}");
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let st = node.state();
        let swapped = !st.metadata.indices.contains_key(&first) && !st.metadata.indices.contains_key(&format!("dlm-clone-{first}")) && st.metadata.indices.contains_key(&mounted);
        if swapped {
            break;
        }
        if Instant::now() >= deadline {
            let (_, e) = call(&c, http, reqwest::Method::GET, &format!("/{first}/_lifecycle/explain"), None).await;
            panic!("never converted to frozen; indices {:?}; explain: {e}", st.metadata.indices.keys().collect::<Vec<_>>());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_data_stream/frz-a", None).await;
    ok(st, &v);
    assert_eq!(v["data_streams"][0]["indices"][0]["index_name"], json!(mounted), "{v}");
    let (st, v) = call(&c, http, reqwest::Method::GET, &format!("/{mounted}/_settings?flat_settings=true"), None).await;
    ok(st, &v);
    let s = &v[mounted.as_str()]["settings"];
    assert_eq!(s["index.store.type"], "snapshot", "{v}");
    assert_eq!(s["index.store.snapshot.partial"], "true", "{v}");
    assert_eq!(s["index.dlm.frozen.created"], "true", "{v}");
    assert_eq!(s["index.store.snapshot.snapshot_name"], json!(format!("dlm-frozen-dlm-clone-{first}")), "{v}");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_snapshot/frozen-repo/_all", None).await;
    ok(st, &v);
    assert_eq!(v["snapshots"].as_array().map(Vec::len), Some(1), "{v}");
    let (_, count) = call(&c, http, reqwest::Method::GET, "/frz-a/_count", None).await;
    assert_eq!(count["count"], 30, "{count}");
}

/// Review #87: a managed index restored from a snapshot continues its
/// lifecycle where it was (ES restores the `ilm` execution state and the
/// lifecycle date with the index): an index in the warm phase is restored
/// in the warm phase, aged from its original lifecycle date, and its
/// phases do not restart. `file_based`: the restore adopts the snapshot's
/// files, else it re-indexes the documents.
async fn restored_index_keeps_its_lifecycle_phase_and_age(cluster: &str, file_based: bool) {
    let repo_root = evidence::scratch_dir(&format!("{cluster}-repo"));
    let repo_path = repo_root.to_string_lossy().to_string();
    let mut extra = vec![("path.repo", repo_path.as_str())];
    if !file_based {
        extra.push(("snapshot.restore.file_based", "false"));
    }
    let node = single_node(cluster, &extra).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_snapshot/backups", Some(json!({"type": "fs", "settings": {"location": repo_root.join("backups").to_string_lossy()}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_ilm/policy/aging", Some(json!({"policy": {"phases": {
        "hot": {"min_age": "0ms", "actions": {"set_priority": {"priority": 100}}},
        "warm": {"min_age": "2s", "actions": {"set_priority": {"priority": 50}}},
        "delete": {"min_age": "30d", "actions": {"delete": {}}}}}}))).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/aged", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "aging"}}))).await;
    ok(st, &v);
    for i in 0..5 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/aged/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let before = loop {
        let e = explain(&c, http, "aged").await;
        if e["phase"] == "warm" && e["action"] == "complete" && e["step"] == "complete" {
            break e;
        }
        assert!(Instant::now() < deadline, "never completed the warm phase: {e}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/_snapshot/backups/s1?wait_for_completion=true", Some(json!({"indices": "aged"}))).await;
    ok(st, &v);
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    // Restored a while later, after the original is gone.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let (st, v) = call(&c, http, reqwest::Method::DELETE, "/aged", None).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::POST, "/_snapshot/backups/s1/_restore?wait_for_completion=true", Some(json!({"indices": "aged"}))).await;
    ok(st, &v);
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    let (st, v) = call(&c, http, reqwest::Method::GET, "/_cluster/health/aged?wait_for_status=green&timeout=60s", None).await;
    ok(st, &v);
    let (_, count) = call(&c, http, reqwest::Method::GET, "/aged/_count", None).await;
    assert_eq!(count["count"], 5, "{count}");
    // Several lifecycle polls: still where it was, never restarted.
    for _ in 0..4 {
        let after = explain(&c, http, "aged").await;
        assert_eq!((&after["policy"], &after["phase"], &after["action"], &after["step"]), (&json!("aging"), &json!("warm"), &json!("complete"), &json!("complete")), "restored index restarted its lifecycle: {after}; before: {before}");
        assert_eq!(after["lifecycle_date_millis"], before["lifecycle_date_millis"], "lifecycle age reset: {after}; before: {before}");
        assert_eq!(after["phase_time_millis"], before["phase_time_millis"], "{after}; before: {before}");
        assert_eq!(after["phase_execution"]["policy"], "aging", "{after}");
        tokio::time::sleep(Duration::from_millis(1100)).await;
    }
    let (_, s) = call(&c, http, reqwest::Method::GET, "/aged/_settings/index.priority,index.lifecycle.name?flat_settings=true", None).await;
    assert_eq!(s["aged"]["settings"]["index.priority"], "50", "{s}");
    assert_eq!(s["aged"]["settings"]["index.lifecycle.name"], "aging", "{s}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn file_restore_keeps_the_lifecycle_phase_and_age() {
    restored_index_keeps_its_lifecycle_phase_and_age("ilm-restore-files", true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn document_restore_keeps_the_lifecycle_phase_and_age() {
    restored_index_keeps_its_lifecycle_phase_and_age("ilm-restore-docs", false).await;
}

async fn step_outcome(f: impl Fn() -> elasticsearch_engine::ilm::StepProgress) -> elasticsearch_engine::ilm::StepProgress {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match f() {
            elasticsearch_engine::ilm::StepProgress::Pending(_) => {}
            other => return other,
        }
        assert!(Instant::now() < deadline, "the step never finished");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Review finding 128: the ILM shrink helper owns only the target
/// incarnation its source's resize created. An existing index under the
/// target name that this source did not resize into (a restored lifecycle
/// state naming an old target, an unrelated index) is neither adopted —
/// whatever its document count — nor deleted, and the source is kept. A
/// cleanup delete is conditioned on the target incarnation it inspected:
/// one held past a re-creation of the name is refused. A real shrink is
/// the positive control.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ilm_shrink_owns_only_its_own_target_incarnation() {
    use elasticsearch_engine::ilm::{ClusterLifecycleSteps, StepProgress};
    let node = single_node("ilm-shrink-owner", &[]).await;
    let http = ports::http_port(&node);
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/src", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    for i in 0..3 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/src/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        ok(st, &v);
    }
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/src/_settings", Some(json!({"index.blocks.write": true}))).await;
    ok(st, &v);
    // An unrelated index under the target name, holding more documents.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/shrink-x-src", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    for i in 0..5 {
        let (st, v) = call(&c, http, reqwest::Method::PUT, &format!("/shrink-x-src/_doc/other-{i}?refresh=true"), Some(json!({"other": i}))).await;
        ok(st, &v);
    }
    let steps = elasticsearch_node::lifecycle_steps::ClusterSteps::new(Arc::downgrade(&node));
    let body = json!({"settings": {"index.number_of_shards": "1", "index.number_of_replicas": "0", "index.routing.allocation.require._id": null}});
    match step_outcome(|| steps.shrink("src", "shrink-x-src", &body)).await {
        StepProgress::Failed(why) => assert!(why.contains("not created by a resize of this source"), "{why}"),
        other => panic!("a foreign target was adopted: {other:?}"),
    }
    match step_outcome(|| steps.holds_all_docs("src", "shrink-x-src")).await {
        StepProgress::Failed(_) => {}
        other => panic!("a foreign target was accepted as the shrink: {other:?}"),
    }
    let (_, v) = call(&c, http, reqwest::Method::GET, "/shrink-x-src/_count", None).await;
    assert_eq!(v["count"], 5, "the foreign target is untouched: {v}");
    assert!(node.state().metadata.indices.contains_key("src"));
    // A cleanup conditioned on an incarnation the name no longer has.
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/reused", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    let old = node.state().metadata.indices["reused"].uuid.clone();
    let (st, v) = call(&c, http, reqwest::Method::DELETE, "/reused", None).await;
    ok(st, &v);
    let (st, v) = call(&c, http, reqwest::Method::PUT, "/reused", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    ok(st, &v);
    let r = c.delete(format!("http://127.0.0.1:{http}/reused")).header(elasticsearch_http::IF_INDEX_UUID_HEADER, old.as_str()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 409, "{:?}", r.text().await);
    assert!(node.state().metadata.indices.contains_key("reused"), "the re-created index survives a stale conditioned delete");
    let cur = node.state().metadata.indices["reused"].uuid.clone();
    let r = c.delete(format!("http://127.0.0.1:{http}/reused")).header(elasticsearch_http::IF_INDEX_UUID_HEADER, cur.as_str()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    // Positive control: the source's own completed resize.
    match step_outcome(|| steps.shrink("src", "shrink-y-src", &body)).await {
        StepProgress::Done => {}
        other => panic!("the shrink did not complete: {other:?}"),
    }
    match step_outcome(|| steps.holds_all_docs("src", "shrink-y-src")).await {
        StepProgress::Done => {}
        other => panic!("{other:?}"),
    }
    // Re-running the step adopts that completed target.
    match step_outcome(|| steps.shrink("src", "shrink-y-src", &body)).await {
        StepProgress::Done => {}
        other => panic!("{other:?}"),
    }
}
