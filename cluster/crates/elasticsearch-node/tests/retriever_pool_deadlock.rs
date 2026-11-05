//! Retriever (`rrf` over `standard` / `knn`) searches on a two-node
//! cluster whose search pool is tiny (`node.processors: 1`: 2 search
//! threads), at a concurrency well past it.
//!
//! Incident 2026-11-08 (vec1b, 256 concurrent hybrid RRF requests): every
//! `es-search` thread of both data nodes ran a retriever's plan phase and
//! waited (`Handle::block_on`) on child cluster searches whose shard query
//! phases were queued behind them on the same pool: a permanent deadlock.
//! ES never parks a search thread on child searches: the coordination runs
//! off the search pool and the shard phases go to it.
//!
//! The same incident left client-timed-out searches running: a child
//! search of a retriever was not linked to the client's search task, so a
//! disconnect (ES `RestCancellableNodeClient`) did not cancel it.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

struct Two {
    http: [u16; 2],
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
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    put("xpack.license.self_generated.type", json!("trial"));
    // The executors are sized from the allocated processors (one process,
    // shared by both nodes): `search` = 1 * 3 / 2 + 1 = 2 threads.
    put("node.processors", json!(1));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn start(cluster: &str) -> Two {
    let cluster = ports::unique(cluster);
    let mut http = [0u16; 2];
    let mut transport: Vec<u16> = Vec::new();
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(settings(&cluster, i, &transport)).expect("config");
        let node = tokio::spawn(async move {
            let (node, listener) = elasticsearch_node::node::ClusterNode::start(cfg).await.expect("node starts");
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
    }
    let two = Two { http, c: reqwest::Client::new() };
    ports::wait_formed(&two.c, &two.http, None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    two
}

impl Two {
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

    /// Running `*search*` tasks on every node.
    async fn search_tasks(&self) -> Vec<(String, Value)> {
        let v = self.ok(0, Method::GET, "/_tasks?actions=*search*&detailed", None).await;
        let mut out = Vec::new();
        for node in v["nodes"].as_object().into_iter().flat_map(|n| n.values()) {
            for (id, t) in node["tasks"].as_object().into_iter().flatten() {
                out.push((id.clone(), t.clone()));
            }
        }
        out
    }

    async fn no_search_tasks(&self, what: &str, within: Duration) {
        let deadline = Instant::now() + within;
        loop {
            let left = self.search_tasks().await;
            if left.is_empty() {
                return;
            }
            assert!(Instant::now() < deadline, "{what}: search tasks still running: {left:?}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Active `search` executor threads on every node.
    async fn search_active(&self) -> u64 {
        let v = self.ok(0, Method::GET, "/_nodes/stats/thread_pool", None).await;
        v["nodes"].as_object().into_iter().flat_map(|n| n.values()).map(|n| n["thread_pool"]["search"]["active"].as_u64().unwrap_or(0)).sum()
    }
}

fn hybrid(i: usize) -> Value {
    let x = (i % 10) as f32;
    match i % 3 {
        0 => json!({"retriever": {"rrf": {"retrievers": [
            {"standard": {"query": {"match": {"text": "alpha"}}}},
            {"knn": {"field": "vector", "query_vector": [x, 1.0], "k": 5, "num_candidates": 10}}],
            "rank_window_size": 10}}, "size": 5}),
        1 => json!({"retriever": {"rrf": {"retrievers": [
            {"standard": {"query": {"match": {"text": "alpha"}}}},
            {"standard": {"query": {"match": {"text": "beta"}}}}]}}, "size": 5}),
        // Nested: an rrf whose child is an rrf.
        _ => json!({"retriever": {"rrf": {"retrievers": [
            {"rrf": {"retrievers": [
                {"standard": {"query": {"match": {"text": "beta"}}}},
                {"knn": {"field": "vector", "query_vector": [1.0, x], "k": 5, "num_candidates": 10}}]}},
            {"standard": {"query": {"match_all": {}}}}]}}, "size": 5}),
    }
}

/// A child query that runs for far longer than the test (each doc's
/// score loops just under painless's loop limit, over thousands of docs):
/// only a cancellation stops it in time.
fn endless_retriever() -> Value {
    json!({"retriever": {"rrf": {"retrievers": [
        {"standard": {"query": {"match": {"text": "alpha"}}}},
        {"standard": {"query": {"script_score": {"query": {"match_all": {}},
            "script": {"source": "double t = 0; for (int i = 0; i < 999000; i++) { t += Math.sqrt(i); } return 1;"}}}}}]}}})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retriever_searches_past_the_search_pool_complete_and_disconnects_cancel_them() {
    let two = start("retriever-pool").await;
    let v = two.ok(0, Method::GET, "/_nodes/stats/thread_pool", None).await;
    for n in v["nodes"].as_object().unwrap().values() {
        let threads = n["thread_pool"]["search"]["threads"].as_u64().unwrap();
        assert!(threads <= 4, "the test needs a small search pool: {threads}");
    }
    two.ok(0, Method::PUT, "/hy", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}, "vector": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm"}}}
    }))).await;
    let v = two.ok(0, Method::GET, "/_cluster/health/hy?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(v["status"], "green", "{v}");
    let mut nd = String::new();
    for i in 0..200 {
        let text = match i % 3 { 0 => "alpha", 1 => "beta", _ => "alpha beta" };
        nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "hy", "_id": i.to_string()}}), json!({"text": text, "vector": [(i % 10) as f32, (i % 7) as f32]})));
    }
    let r = two.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", two.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
    // The slow searches' index: many docs, each slow to score.
    two.ok(0, Method::PUT, "/slowhy", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}}}
    }))).await;
    two.ok(0, Method::GET, "/_cluster/health/slowhy?wait_for_status=green&timeout=60s", None).await;
    let mut nd = String::new();
    for i in 0..6000 {
        nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": "slowhy", "_id": i.to_string()}}), json!({"text": if i % 2 == 0 { "alpha" } else { "beta" }})));
    }
    let r = two.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", two.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
    let shards = two.ok(0, Method::GET, "/_cat/shards/hy?format=json", None).await;
    let holders: std::collections::BTreeSet<&str> = shards.as_array().unwrap().iter().filter_map(|s| s["node"].as_str()).collect();
    eprintln!("shards of hy on {holders:?}");

    // 1. Many more concurrent retriever searches than search threads, on
    // both coordinating nodes: all answer (the deadlock never answers).
    const CONCURRENCY: usize = 48;
    let started = Instant::now();
    let runs = (0..CONCURRENCY).map(|i| {
        let c = two.c.clone();
        let url = format!("http://127.0.0.1:{}/hy/_search", two.http[i % 2]);
        tokio::spawn(async move {
            let r = c.post(url).json(&hybrid(i)).send().await.expect("request");
            let st = r.status().as_u16();
            (i, st, r.json::<Value>().await.unwrap_or(Value::Null))
        })
    }).collect::<Vec<_>>();
    let answers = tokio::time::timeout(Duration::from_secs(120), futures::future::join_all(runs)).await
        .unwrap_or_else(|_| panic!("retriever searches past the search pool hung (deadlock) after {:?}", started.elapsed()));
    for a in answers {
        let (i, st, v) = a.unwrap();
        assert_eq!(st, 200, "request {i}: {v}");
        assert!(!v["hits"]["hits"].as_array().map_or(true, Vec::is_empty), "request {i}: {v}");
    }
    eprintln!("{CONCURRENCY} concurrent retriever searches answered in {:?}", started.elapsed());
    two.no_search_tasks("after the concurrent retrievers", Duration::from_secs(20)).await;

    // 2. A client that goes away cancels its retriever search: the plan
    // phase and its child searches (whose shard tasks are children of the
    // client's task) stop; their search threads are free again.
    for node in 0..2 {
        let gone = two.c.post(format!("http://127.0.0.1:{}/slowhy/_search", two.http[node])).timeout(Duration::from_secs(2))
            .json(&endless_retriever()).send().await;
        if let Ok(answer) = gone {
            let st = answer.status();
            panic!("via n{node}: the endless retriever search answered {st}: {}", answer.text().await.unwrap_or_default());
        }
        two.no_search_tasks(&format!("after the client of a retriever search via n{node} hung up"), Duration::from_secs(20)).await;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while two.search_active().await > 0 {
        assert!(Instant::now() < deadline, "search threads still busy after the cancelled searches");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // 3. `_tasks/_cancel` of the client's task answers task_cancelled and
    // stops the children.
    let client = two.c.clone();
    let url = format!("http://127.0.0.1:{}/slowhy/_search", two.http[0]);
    let pending = tokio::spawn(async move { client.post(url).json(&endless_retriever()).send().await.unwrap() });
    let deadline = Instant::now() + Duration::from_secs(20);
    let id = loop {
        let tasks = two.search_tasks().await;
        if let Some((id, _)) = tasks.iter().find(|(_, t)| t["action"] == "indices:data/read/search" && t["parent_task_id"].is_null()) {
            // Wait for its children (the plan phase) before cancelling.
            if tasks.iter().any(|(_, t)| t["parent_task_id"] == id.as_str()) {
                break id.clone();
            }
        }
        assert!(Instant::now() < deadline, "the retriever search task never listed: {tasks:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    two.ok(0, Method::POST, &format!("/_tasks/{id}/_cancel"), None).await;
    let response = tokio::time::timeout(Duration::from_secs(15), pending).await.expect("a cancelled retriever search kept running").unwrap();
    let st = response.status().as_u16();
    let v: Value = response.json().await.unwrap_or(Value::Null);
    assert_eq!(st, 400, "{v}");
    assert!(v.to_string().contains("task_cancelled_exception"), "{v}");
    two.no_search_tasks("after _tasks/_cancel of a retriever search", Duration::from_secs(20)).await;

    // The nodes still answer retriever searches.
    for node in 0..2 {
        let v = two.ok(node, Method::POST, "/hy/_search", Some(hybrid(node))).await;
        assert!(!v["hits"]["hits"].as_array().unwrap().is_empty(), "{v}");
    }
}
