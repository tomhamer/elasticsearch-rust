//! Search tasks on a three-node in-process cluster (ES `SearchTask` /
//! `SearchShardTask`, `TaskCancellationService`, `RestCancellableNodeClient`):
//!
//! * a running `_search` lists in `_tasks?actions=*search*&detailed` as its
//!   coordinating `indices:data/read/search` task with ES's description,
//!   and its shard task on the node holding the shard as a
//!   `[phase/query]` child naming it as `parent_task_id`;
//! * `_tasks/<id>/_cancel` answers the search with `task_cancelled_exception`
//!   and cancels the shard task on the other node (its thread is freed);
//! * a client that disconnects cancels its search, local or remote shards;
//! * `search.default_search_timeout` applies to a search without `timeout`;
//! * `timeout` bounds aggregation collection (`timed_out: true`).
//!
//! The long-running search scores with a script that loops (practically)
//! forever under a long `timeout`: only a cancellation stops it.

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

struct Three {
    nodes: Vec<Arc<ClusterNode>>,
    http: [u16; 3],
    c: reqwest::Client,
}

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
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1", "n2"]));
    put("xpack.license.self_generated.type", json!("trial"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn start(cluster: &str) -> Three {
    let cluster = ports::unique(cluster);
    let mut http = [0u16; 3];
    let mut transport: Vec<u16> = Vec::new();
    let mut nodes = Vec::new();
    for i in 0..3 {
        let cfg = NodeConfig::from_settings(settings(&cluster, i, &transport)).expect("config");
        let node = tokio::spawn(async move {
            let (node, listener) = ClusterNode::start(cfg).await.expect("node starts");
            elasticsearch_node::rest::register(&node);
            elasticsearch_node::snapshot_api::register(&node);
            let n = node.clone();
            tokio::spawn(async move {
                let _ = elasticsearch_node::front::serve(n, listener).await;
            });
            node
        }).await.unwrap();
        http[i] = ports::http_port(&node);
        transport.push(ports::transport_port(&node));
        nodes.push(node);
    }
    let three = Three { nodes, http, c: reqwest::Client::new() };
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

    /// A one-shard index whose shard lives on node `on`, with `n` docs.
    async fn index_on(&self, index: &str, on: usize, n: usize) {
        self.ok(0, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": format!("n{on}")},
            "mappings": {"properties": {"n": {"type": "long"}, "k": {"type": "keyword"}}}
        }))).await;
        let v = self.ok(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(v["status"], "green", "{v}");
        let mut at = 0;
        while at < n {
            let end = (at + 10_000).min(n);
            let mut nd = String::new();
            for i in at..end {
                nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": index, "_id": i.to_string()}}), json!({"n": i, "k": format!("k{}", i % 100)})));
            }
            let r = self.c.post(format!("http://127.0.0.1:{}/_bulk", self.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
            let v: Value = r.json().await.unwrap();
            assert_eq!(v["errors"], false, "{v}");
            at = end;
        }
        self.ok(0, Method::POST, &format!("/{index}/_refresh"), None).await;
    }

    /// Running tasks whose action matches `*search*`: (task id, info).
    async fn search_tasks(&self, i: usize) -> Vec<(String, Value)> {
        let v = self.ok(i, Method::GET, "/_tasks?actions=*search*&detailed", None).await;
        let mut out = Vec::new();
        for node in v["nodes"].as_object().into_iter().flat_map(|n| n.values()) {
            for (id, t) in node["tasks"].as_object().into_iter().flatten() {
                out.push((id.clone(), t.clone()));
            }
        }
        out
    }

    /// Wait until no search task runs anywhere (each node's own listing).
    async fn no_search_tasks(&self, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let mut left = Vec::new();
            for i in 0..3 {
                left.extend(self.search_tasks(i).await);
            }
            if left.is_empty() {
                return;
            }
            assert!(Instant::now() < deadline, "{what}: search tasks still running: {left:?}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// A search scoring every doc with a loop that only a cancellation (or
/// its `timeout`) ends.
fn endless_search(timeout: Option<&str>) -> Value {
    let mut body = json!({"query": {"script_score": {"query": {"match_all": {}},
        "script": {"source": "double t = 0; for (int i = 0; i < 2000000000; i++) { t += i; } return 1;"}}}});
    if let Some(t) = timeout {
        body["timeout"] = json!(t);
    }
    body
}

/// Wait for the coordinating search task and its shard child.
async fn listed(three: &Three, index: &str) -> (String, Vec<(String, Value)>) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let tasks = three.search_tasks(0).await;
        let parent = tasks.iter().find(|(_, t)| t["action"] == "indices:data/read/search").cloned();
        if let Some((id, info)) = parent {
            let children: Vec<(String, Value)> = tasks.iter().filter(|(_, t)| t["parent_task_id"] == id.as_str()).cloned().collect();
            if !children.is_empty() {
                let description = info["description"].as_str().unwrap_or_default();
                assert!(description.starts_with(&format!("indices[{index}], search_type[QUERY_THEN_FETCH], source[")), "{info}");
                assert_eq!(info["cancellable"], true, "{info}");
                assert_eq!(info["cancelled"], false, "{info}");
                return (id, children);
            }
        }
        assert!(Instant::now() < deadline, "search task and its shard child never listed: {tasks:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_search_task_answers_task_cancelled_and_stops_its_remote_shard_task() {
    let three = start("sc-cancel").await;
    three.index_on("slow", 2, 20).await;
    let client = three.c.clone();
    let url = format!("http://127.0.0.1:{}/slow/_search", three.http[0]);
    let pending = tokio::spawn(async move { client.post(url).json(&endless_search(Some("1h"))).send().await.unwrap() });
    let (id, children) = listed(&three, "slow").await;
    // The shard task runs on n2 (the shard's only copy), as a child.
    let n2 = three.nodes[2].local.id.clone();
    for (child, info) in &children {
        assert!(child.starts_with(&format!("{n2}:")), "the shard task runs where the shard is: {child} {info}");
        assert_eq!(info["action"], "indices:data/read/search[phase/query]", "{info}");
        assert_eq!(info["description"], "shardId[[slow][0]]", "{info}");
        assert_eq!(info["cancellable"], true, "{info}");
    }
    let cancelled = three.ok(0, Method::POST, &format!("/_tasks/{id}/_cancel"), None).await;
    assert!(cancelled.to_string().contains("indices:data/read/search"), "{cancelled}");
    let response = tokio::time::timeout(Duration::from_secs(10), pending).await.expect("a cancelled search kept running").unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let value: Value = response.json().await.unwrap();
    assert_eq!(value["error"]["type"], "task_cancelled_exception", "{value}");
    assert_eq!(value["error"]["reason"], "task cancelled [by user request]", "{value}");
    assert_eq!(value["status"], 400, "{value}");
    // The ban reached n2: its shard task stopped (its executor thread is
    // free: the task lists until its work ends).
    three.no_search_tasks("after _tasks/_cancel").await;
    // The node still searches.
    let ok = three.ok(0, Method::POST, "/slow/_search", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(ok["hits"]["total"]["value"], 20, "{ok}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_disconnected_client_cancels_its_search_on_local_and_remote_shards() {
    let three = start("sc-disconnect").await;
    three.index_on("slow-remote", 2, 20).await;
    three.index_on("slow-local", 0, 20).await;
    for index in ["slow-remote", "slow-local"] {
        let gone = three.c.post(format!("http://127.0.0.1:{}/{index}/_search", three.http[0])).timeout(Duration::from_secs(2))
            .json(&endless_search(Some("1h"))).send().await;
        assert!(gone.is_err(), "{index}: the endless search answered");
        // A shard task lists until its executor thread is done with the
        // work: gone, the thread was freed (the endless loop only stops
        // when cancelled).
        three.no_search_tasks(index).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn default_search_timeout_applies_and_timeout_bounds_aggregations() {
    let three = start("sc-timeouts").await;
    three.index_on("slow", 2, 20).await;
    // `search.default_search_timeout`: a search naming no `timeout` stops.
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"search.default_search_timeout": "300ms"}}))).await;
    let started = Instant::now();
    let (status, v) = tokio::time::timeout(Duration::from_secs(30), three.call(0, Method::POST, "/slow/_search", Some(endless_search(None))))
        .await.expect("the default search timeout did not apply");
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["timed_out"], true, "{v}");
    assert!(started.elapsed() < Duration::from_secs(20));
    // A request's own `timeout` wins over the default.
    let (status, v) = three.call(0, Method::POST, "/slow/_search", Some(json!({"timeout": "30s", "query": {"match_all": {}}}))).await;
    assert_eq!((status, &v["timed_out"]), (200, &json!(false)), "{v}");
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"search.default_search_timeout": null}}))).await;
    three.no_search_tasks("after the default timeout").await;

    // `timeout` bounds aggregation collection: the query phase collects
    // nothing (size 0, no total), the aggregation's per-doc collection
    // stops at the deadline and the search reports it timed out.
    three.index_on("aggs", 0, 60_000).await;
    let agg = |timeout: Option<&str>| {
        let mut body = json!({"size": 0, "track_total_hits": false,
            "query": {"script": {"script": {"source": "doc['n'].value % 7 != 100"}}},
            "aggs": {"t": {"terms": {"field": "k", "size": 100}}}});
        if let Some(t) = timeout {
            body["timeout"] = json!(t);
        }
        body
    };
    let counted = |v: &Value| -> u64 {
        v["aggregations"]["t"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum::<u64>()
            + v["aggregations"]["t"]["sum_other_doc_count"].as_u64().unwrap_or(0)
    };
    let full = three.ok(0, Method::POST, "/aggs/_search", Some(agg(None))).await;
    assert_eq!(full["timed_out"], false, "{full}");
    assert_eq!(counted(&full), 60_000, "{full}");
    let bounded = three.ok(0, Method::POST, "/aggs/_search", Some(agg(Some("1ms")))).await;
    assert_eq!(bounded["timed_out"], true, "aggregation collection ignored the timeout: {bounded}");
    assert!(bounded["aggregations"]["t"]["buckets"].is_array(), "partial aggregations are returned: {bounded}");
    assert!(counted(&bounded) < 60_000, "{bounded}");
}

/// `scripted_metric` and script-valued aggregations whose scripts loop per
/// doc: a `timeout` (or `search.default_search_timeout`) answers partial
/// aggregations with `timed_out: true` promptly, and `_tasks/_cancel` stops
/// them (no timeout) with `task_cancelled_exception`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn aggregation_scripts_keep_to_the_timeout_and_stop_when_cancelled() {
    let three = start("sc-agg-scripts").await;
    three.index_on("aggs", 0, 60_000).await;
    const LOOP: &str = "double t = 0; for (int i = 0; i < 2000; i++) { t += i; }";
    let scripted = |timeout: Option<&str>| {
        let mut body = json!({"size": 0, "aggs": {"m": {"scripted_metric": {
            "init_script": "state.n = 0",
            "map_script": format!("{LOOP} state.n += 1;"),
            "combine_script": "return state.n",
            "reduce_script": "double s = 0; for (int i = 0; i < states.size(); i++) { s += states[i]; } return s;"}}}});
        if let Some(t) = timeout {
            body["timeout"] = json!(t);
        }
        body
    };
    let script_terms = |timeout: Option<&str>| {
        let mut body = json!({"size": 0, "aggs": {"t": {"terms": {"size": 100,
            "script": {"source": format!("{LOOP} return doc['k'].value;")}}}}});
        if let Some(t) = timeout {
            body["timeout"] = json!(t);
        }
        body
    };
    let scripted_under_terms = json!({"size": 0, "timeout": "1ms", "aggs": {"t": {"terms": {"field": "k", "size": 20000},
        "aggs": {"m": scripted(None)["aggs"]["m"].clone()}}}});
    for (what, body) in [("scripted_metric", scripted(Some("1ms"))), ("script terms", script_terms(Some("1ms"))), ("terms + scripted_metric", scripted_under_terms)] {
        let started = Instant::now();
        let v = three.ok(0, Method::POST, "/aggs/_search", Some(body)).await;
        assert_eq!(v["timed_out"], true, "{what}: {v}");
        assert!(v["aggregations"].is_object(), "{what}: partial aggregations are returned: {v}");
        assert!(started.elapsed() < Duration::from_secs(3), "{what} ran {:?} past a 1ms timeout", started.elapsed());
    }
    // A partial scripted_metric counts fewer docs than the index holds.
    let v = three.ok(0, Method::POST, "/aggs/_search", Some(scripted(Some("1ms")))).await;
    assert!(v["aggregations"]["m"]["value"].as_f64().unwrap_or(0.0) < 60_000.0, "{v}");

    // The default search timeout reaches aggregation scripts too.
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"search.default_search_timeout": "1ms"}}))).await;
    let started = Instant::now();
    let v = three.ok(0, Method::POST, "/aggs/_search", Some(scripted(None))).await;
    assert_eq!(v["timed_out"], true, "{v}");
    assert!(started.elapsed() < Duration::from_secs(3), "{:?}", started.elapsed());
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"search.default_search_timeout": null}}))).await;

    // No timeout: `_tasks/_cancel` stops the map phase.
    let client = three.c.clone();
    let url = format!("http://127.0.0.1:{}/aggs/_search", three.http[0]);
    // Heavier per doc (still within the loop limit): it runs far longer
    // than the cancellation takes.
    let mut body = scripted(None);
    body["aggs"]["m"]["scripted_metric"]["map_script"] = json!("double t = 0; for (int i = 0; i < 50000; i++) { t += i; } state.n += 1;");
    let pending = tokio::spawn(async move { client.post(url).json(&body).send().await.unwrap() });
    let (id, _) = listed(&three, "aggs").await;
    three.ok(0, Method::POST, &format!("/_tasks/{id}/_cancel"), None).await;
    let response = tokio::time::timeout(Duration::from_secs(5), pending).await.expect("a cancelled scripted aggregation kept running").unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let value: Value = response.json().await.unwrap();
    assert_eq!(value["error"]["type"], "task_cancelled_exception", "{value}");
    three.no_search_tasks("after cancelling the scripted aggregation").await;
}

/// A client that hangs up on a search without a `timeout` whose script is
/// slow per doc (but within the loop limit): its shard tasks stop within
/// about a second (the script's loops poll the task's cancellation).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_disconnect_stops_a_slow_script_without_timeout_within_a_second() {
    let three = start("sc-slow-script").await;
    three.index_on("slow-docs", 2, 3000).await;
    let body = json!({"query": {"script_score": {"query": {"match_all": {}},
        "script": {"source": "double t = 0; for (int i = 0; i < 200000; i++) { t += Math.sqrt(i); } return 1;"}}}});
    let gone = three.c.post(format!("http://127.0.0.1:{}/slow-docs/_search", three.http[0])).timeout(Duration::from_secs(2))
        .json(&body).send().await;
    assert!(gone.is_err(), "the slow search answered within 2s");
    let hung_up = Instant::now();
    let deadline = hung_up + Duration::from_secs(3);
    loop {
        let mut left = Vec::new();
        for i in 0..3 {
            left.extend(three.search_tasks(i).await);
        }
        if left.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "shard tasks still running {:?} after the client hung up: {left:?}", hung_up.elapsed());
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
