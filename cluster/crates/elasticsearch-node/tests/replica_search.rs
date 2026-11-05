//! Replicated-search regressions on a three-node in-process cluster:
//!
//! * a shard-level read reaching a node that does not hold a readable copy
//!   fails (`shard_not_found_exception`) instead of answering with that
//!   shard's documents missing, so counts stay exact while shards move;
//! * a shard failure is retried on the next copy, then reported as a
//!   partial result (`_shards.failures`), as ES does;
//! * a shard without an active copy is a partial failure, not a 503;
//! * `preference` values select copies as ES does, a custom string the same
//!   copy from every coordinator.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    // Retrievers and rank (RRF) need a trial license.
    put("xpack.license.self_generated.type", json!("trial"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

/// Start three nodes on OS-assigned ports, one after another: each seeds
/// from the transport ports of those before it.
async fn start(cluster: &str) -> Three {
    let mut http = [0u16; 3];
    let mut transport: Vec<u16> = Vec::new();
    let mut nodes = Vec::new();
    for i in 0..3 {
        let cfg = NodeConfig::from_settings(settings(cluster, i, &transport)).expect("config");
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
    // Formed and recovered on every node: ES (`ClusterStateHealth`) reports
    // red while the "state not recovered" global block is up, and refuses
    // index creation with that block.
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

    async fn bulk(&self, index: &str, docs: impl Iterator<Item = (String, Option<String>, Value)>) {
        let mut nd = String::new();
        for (id, routing, src) in docs {
            let meta = match routing {
                Some(r) => json!({"index": {"_index": index, "_id": id, "routing": r}}),
                None => json!({"index": {"_index": index, "_id": id}}),
            };
            nd.push_str(&format!("{meta}\n{src}\n"));
        }
        let r = self.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", self.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }

    async fn green(&self, index: &str) {
        let v = self.ok(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&wait_for_no_relocating_shards=true&wait_for_no_initializing_shards=true&timeout=60s"), None).await;
        assert_eq!(v["status"], "green", "{v}");
    }

    /// `[index][shard]` → nodes of its started copies.
    async fn copies(&self, index: &str) -> BTreeMap<u32, BTreeSet<String>> {
        let st = self.nodes[0].state();
        let mut out = BTreeMap::new();
        for t in &st.routing_table.indices[index].shards {
            out.insert(t.shard, t.copies.iter().filter(|c| c.active()).filter_map(|c| c.node_id.clone()).collect());
        }
        out
    }
}

/// A routing value that `index` (on `n` shards) sends to shard `k`.
async fn routing_to(three: &Three, index: &str, k: u32) -> String {
    for i in 0..500 {
        let r = format!("r{i}");
        let v = three.ok(0, Method::GET, &format!("/{index}/_search_shards?routing={r}"), None).await;
        if v["shards"][0][0]["shard"] == k {
            return r;
        }
    }
    panic!("no routing value for shard {k}");
}

fn shards_failed(v: &Value) -> u64 {
    v["_shards"]["failed"].as_u64().unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_shard_level_read_of_a_copy_this_node_does_not_hold_is_shard_not_found() {
    let three = start("rs-guard").await;
    three.ok(0, Method::PUT, "/g", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    three.green("g").await;
    three.bulk("g", (0..20).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    let copies = three.copies("g").await;
    for (k, holders) in copies {
        let holder = holders.iter().next().unwrap().clone();
        for node in &three.nodes {
            // What a coordinator sends a copy holder: execute locally, on
            // shard `k`, these copies.
            let req = elasticsearch_node::front::RestReq::try_new("POST","/g/_search?size=0&track_total_hits=true", &[
                ("content-type".into(), "application/json".into()),
                ("x-es-cluster-exec-local".into(), "1".into()),
                (elasticsearch_node::shard_engines::SHARD_HEADER.into(), k.to_string()),
                ("x-rustic-internal-read-copies".into(), "g".into()),
            ],b"{}",1,Default::default(),None).unwrap();
            let r = elasticsearch_node::front::dispatch(node, req).await;
            let v = r.body_json().unwrap_or(Value::Null);
            if node.local.id == holder {
                assert_eq!(r.status, 200, "the holder of [g][{k}] answers: {v}");
            } else {
                // Not 200 with zero hits: the coordinator must move on.
                assert_eq!((r.status, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "[g][{k}] on {}: {v}", node.local.name);
            }
        }
    }
}

/// While every shard relocates back and forth, a count from any
/// coordinator is exact or says shards failed: never short with
/// `failed: 0`.
async fn counts_stay_exact_while_shards_move(cluster: &str, replicas: u32) {
    const DOCS: usize = 2_000;
    let three = start(cluster).await;
    three.ok(0, Method::PUT, "/moving", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": replicas}}))).await;
    three.green("moving").await;
    for chunk in 0..(DOCS / 500) {
        three.bulk("moving", (chunk * 500..(chunk + 1) * 500).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    }
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mover = {
        let (c, http, stop) = (three.c.clone(), three.http, stop.clone());
        tokio::spawn(async move {
            let mut round = 0;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                // 5x1 on 3 nodes moves on every exclusion; with replicas
                // only two nodes remain, so copies relocate and re-recover.
                let name = if round % 4 == 3 { String::new() } else { format!("n{}", round % 3) };
                let _ = c.put(format!("http://127.0.0.1:{}/_cluster/settings", http[0]))
                    .json(&json!({"transient": {"cluster.routing.allocation.exclude._name": name}})).send().await;
                round += 1;
                tokio::time::sleep(Duration::from_millis(700)).await;
            }
        })
    };
    let (mut exact, mut reported) = (0, 0);
    let deadline = Instant::now() + Duration::from_secs(25);
    let mut i = 0;
    while Instant::now() < deadline {
        let coord = i % 3;
        i += 1;
        let (st, v) = three.call(coord, Method::POST, "/moving/_search", Some(json!({"size": 0, "track_total_hits": true}))).await;
        if st != 200 {
            assert_eq!(v["error"]["type"], "search_phase_execution_exception", "via n{coord}: {st} {v}");
            reported += 1;
            continue;
        }
        let total = v["hits"]["total"]["value"].as_u64().unwrap();
        if total == DOCS as u64 {
            exact += 1;
            assert_eq!(v["_shards"]["successful"], 5, "{v}");
        } else {
            assert!(shards_failed(&v) > 0, "via n{coord}: {total} of {DOCS} with failed 0: {v}");
            reported += 1;
        }
        let (st, v) = three.call(coord, Method::GET, "/moving/_count", None).await;
        if st == 200 && v["count"] != DOCS {
            assert!(shards_failed(&v) > 0, "via n{coord}: count {v}");
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    mover.await.unwrap();
    assert!(exact > reported, "the next copy should answer almost always: {exact} exact, {reported} reported");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn counts_stay_exact_while_shards_move_without_replicas() {
    counts_stay_exact_while_shards_move("rs-move0", 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn counts_stay_exact_while_shards_move_with_replicas() {
    counts_stay_exact_while_shards_move("rs-move1", 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_failing_shard_is_reported_as_a_partial_result() {
    let three = start("rs-partial").await;
    three.ok(0, Method::PUT, "/p", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 1}, "mappings": {"properties": {"x": {"type": "long"}}}}))).await;
    three.green("p").await;
    let mut docs = Vec::new();
    for k in 0..3 {
        let r = routing_to(&three, "p", k).await;
        docs.push((format!("with-{k}"), Some(r.clone()), json!({"x": 1})));
        if k == 2 {
            // Only shard 2 has a document without `x`: the script fails there.
            docs.push(("without".into(), Some(r), json!({"y": 1})));
        }
    }
    three.bulk("p", docs.into_iter()).await;
    let q = json!({"query": {"script": {"script": {"source": "doc['x'].value > 0"}}}});
    for coord in 0..3 {
        // Every copy of shard 2 fails the same way: retried, then reported.
        let (st, v) = three.call(coord, Method::POST, "/p/_search", Some(q.clone())).await;
        assert_eq!(st, 200, "via n{coord}: {v}");
        assert_eq!(v["_shards"]["total"], 3, "{v}");
        assert_eq!(v["_shards"]["successful"], 2, "{v}");
        assert_eq!(v["_shards"]["failed"], 1, "{v}");
        assert_eq!(v["_shards"]["failures"][0]["shard"], 2, "{v}");
        assert_eq!(v["_shards"]["failures"][0]["index"], "p", "{v}");
        assert_eq!(v["hits"]["total"]["value"], 2, "{v}");
        // Not allowed partial: ES's phase failure.
        let (st, v) = three.call(coord, Method::POST, "/p/_search?allow_partial_search_results=false", Some(q.clone())).await;
        assert_eq!(st, 400, "{v}");
        assert_eq!(v["error"]["type"], "search_phase_execution_exception", "{v}");
        assert_eq!(v["error"]["reason"], "Partial shards failure", "{v}");
        assert_eq!(v["error"]["failed_shards"][0]["shard"], 2, "{v}");
    }
    // Every shard fails: ES's all-shards-failed shape.
    let (st, v) = three.call(0, Method::POST, "/p/_search", Some(json!({"query": {"script": {"script": {"source": "doc['missing'].value > 0"}}}}))).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!((v["error"]["type"].as_str(), v["error"]["reason"].as_str()), (Some("search_phase_execution_exception"), Some("all shards failed")), "{v}");
    // A request error stays the engine's own error.
    let (st, v) = three.call(1, Method::POST, "/p/_search", Some(json!({"query": {"no_such_query": {}}}))).await;
    assert_eq!(st, 400, "{v}");
    assert_ne!(v["error"]["type"], "search_phase_execution_exception", "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_shard_without_an_active_copy_is_a_partial_failure() {
    let three = start("rs-unassigned").await;
    three.ok(0, Method::PUT, "/live", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    three.green("live").await;
    three.bulk("live", (0..10).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    // Single and multi-shard indices that can never be allocated.
    for (name, shards) in [("gone1", 1), ("gone2", 2)] {
        three.ok(0, Method::PUT, &format!("/{name}?wait_for_active_shards=0"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0,
            "index.routing.allocation.require._name": "no-such-node"}}))).await;
    }
    for coord in 0..3 {
        let started = Instant::now();
        let (st, v) = three.call(coord, Method::POST, "/live,gone1,gone2/_search", Some(json!({"track_total_hits": true}))).await;
        assert_eq!(st, 200, "via n{coord}: {v}");
        assert!(started.elapsed() < Duration::from_secs(5), "no wait for copies that are not coming");
        assert_eq!(v["hits"]["total"]["value"], 10, "{v}");
        assert_eq!((v["_shards"]["total"].as_u64(), v["_shards"]["failed"].as_u64()), (Some(5), Some(3)), "{v}");
        assert_eq!(v["_shards"]["failures"][0]["reason"]["type"], "no_shard_available_action_exception", "{v}");
        let (st, v) = three.call(coord, Method::POST, "/live,gone1/_count", None).await;
        assert_eq!((st, v["count"].as_u64(), shards_failed(&v)), (200, Some(10), 1), "{v}");
        // Every shard missing, or partial results not allowed: 503.
        let (st, v) = three.call(coord, Method::POST, "/gone1,gone2/_search", None).await;
        assert_eq!((st, v["error"]["reason"].as_str()), (503, Some("all shards failed")), "{v}");
        let (st, v) = three.call(coord, Method::POST, "/live,gone1/_search?allow_partial_search_results=false", None).await;
        assert_eq!(st, 503, "{v}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn preference_selects_copies_like_es() {
    let three = start("rs-pref").await;
    three.ok(0, Method::PUT, "/pref", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": 2}}))).await;
    three.green("pref").await;
    three.bulk("pref", (0..200).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    // The (shard → node) a search used, from its explained hits.
    async fn served(three: &Three, coord: usize, preference: &str) -> (u16, BTreeMap<String, String>, Value) {
        let (st, v) = three.call(coord, Method::POST, &format!("/pref/_search?preference={preference}&explain=true&size=200"), Some(json!({}))).await;
        let mut by_shard = BTreeMap::new();
        for h in v["hits"]["hits"].as_array().into_iter().flatten() {
            if let (Some(s), Some(n)) = (h["_shard"].as_str(), h["_node"].as_str()) {
                let prev = by_shard.insert(s.to_string(), n.to_string());
                assert!(prev.is_none_or(|p| p == n), "one copy per shard: {h}");
            }
        }
        (st, by_shard, v)
    }
    // A custom string: the same copy of every shard from every coordinator.
    for session in ["sess1", "another-session", "u42"] {
        let (_, first, v) = served(&three, 0, session).await;
        assert_eq!(first.len(), 5, "{v}");
        for coord in 1..3 {
            assert_eq!(served(&three, coord, session).await.1, first, "{session} via n{coord}");
        }
    }
    // Different sessions spread over the copies.
    let mut used = BTreeSet::new();
    for i in 0..20 {
        used.extend(served(&three, 0, &format!("s{i}")).await.1.into_values());
    }
    assert_eq!(used.len(), 3, "sessions land on every node: {used:?}");
    // `_local` / `_only_local`: every node holds a copy of every shard (5x2 on 3 nodes).
    for coord in 0..3 {
        let me = three.nodes[coord].local.id.clone();
        for p in ["_local", "_only_local", "_shards:0,1|_local"] {
            let (st, by_shard, v) = served(&three, coord, p).await;
            assert_eq!(st, 200, "{v}");
            assert!(by_shard.values().all(|n| *n == me), "{p} via n{coord}: {by_shard:?}");
            assert_eq!(v["_shards"]["total"], if p.starts_with("_shards") { 2 } else { 5 }, "{v}");
        }
        let other = three.nodes[(coord + 1) % 3].local.name.clone();
        let (_, by_shard, v) = served(&three, coord, &format!("_only_nodes:{other}")).await;
        let other_id = three.nodes[(coord + 1) % 3].local.id.clone();
        assert!(!by_shard.is_empty() && by_shard.values().all(|n| *n == other_id), "{v}");
        let (_, by_shard, _) = served(&three, coord, &format!("_prefer_nodes:{other_id}")).await;
        assert!(by_shard.values().all(|n| *n == other_id), "{by_shard:?}");
    }
    // Unknown preference types and nodes that match nothing: 400.
    let (st, v) = three.call(0, Method::POST, "/pref/_search?preference=_primary", None).await;
    assert_eq!((st, v["error"]["reason"].as_str()), (400, Some("no Preference for [_primary]")), "{v}");
    let (st, v) = three.call(1, Method::POST, "/pref/_search?preference=_only_nodes:no-such-node", None).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("illegal_argument_exception")), "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("no data nodes with criteria [no-such-node] found for shard: [pref]["), "{v}");
    // search_after pages under one session never repeat or skip, whichever node coordinates.
    let mut seen = BTreeSet::new();
    let mut after: Option<Value> = None;
    for page in 0.. {
        let mut body = json!({"size": 7, "sort": [{"n": "asc"}]});
        if let Some(a) = &after {
            body["search_after"] = a.clone();
        }
        let v = three.ok(page % 3, Method::POST, "/pref/_search?preference=paging-session", Some(body)).await;
        let hits = v["hits"]["hits"].as_array().unwrap().clone();
        if hits.is_empty() {
            break;
        }
        for h in &hits {
            assert!(seen.insert(h["_id"].as_str().unwrap().to_string()), "repeated {h}");
        }
        after = Some(hits.last().unwrap()["sort"].clone());
    }
    assert_eq!(seen.len(), 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn single_shard_indices_on_different_nodes_need_no_common_holder() {
    let three = start("rs-groups").await;
    // i1 only on n0 and n1, i2 only on n0 and n2: no node but n0 holds both.
    for (name, nodes) in [("i1", "n0,n1"), ("i2", "n0,n2")] {
        three.ok(0, Method::PUT, &format!("/{name}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1, "index.routing.allocation.include._name": nodes}}))).await;
        three.green(name).await;
        three.bulk(name, (0..5).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    }
    for coord in 0..3 {
        for p in ["", "?preference=_only_nodes:n1,n2", "?preference=xyz"] {
            let (st, v) = three.call(coord, Method::POST, &format!("/i1,i2/_search{p}"), Some(json!({"track_total_hits": true}))).await;
            assert_eq!((st, v["hits"]["total"]["value"].as_u64()), (200, Some(10)), "via n{coord}{p}: {v}");
            assert_eq!(v["_shards"]["failed"], 0, "{v}");
        }
    }
}

/// Fail the primary of `[index][0]` over to a replica (reroute `cancel`
/// with `allow_primary`), and wait until the shard is whole again.
async fn fail_over(three: &Three, index: &str) -> u64 {
    let st = three.nodes[0].state();
    let term = st.metadata.indices[index].primary_term(0);
    let primary = st.routing_table.shard_table(index, 0).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone()).unwrap();
    three.ok(0, Method::POST, "/_cluster/reroute", Some(json!({"commands": [{"cancel": {"index": index, "shard": 0, "node": primary, "allow_primary": true}}]}))).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let st = three.nodes[0].state();
        if st.metadata.indices[index].primary_term(0) > term {
            break;
        }
        assert!(Instant::now() < deadline, "no failover");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    three.green(index).await;
    three.nodes[0].state().metadata.indices[index].primary_term(0)
}

/// `_seq_no`, `_primary_term`, `_version` of `id` as each copy holds it
/// (realtime GET pinned to each copy's node).
async fn per_copy(three: &Three, index: &str, id: &str) -> Vec<(u16, Value)> {
    let mut out = Vec::new();
    for n in &three.nodes {
        let (st, v) = three.call(0, Method::GET, &format!("/{index}/_doc/{id}?preference=_only_nodes:{}", n.local.id), None).await;
        out.push((st, json!([v["_seq_no"], v["_primary_term"], v["_version"], v["found"]])));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn every_copy_reports_the_ops_own_primary_term() {
    let three = start("rs-terms").await;
    three.ok(0, Method::PUT, "/t", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 2, "refresh_interval": "-1"}}))).await;
    three.green("t").await;
    three.bulk("t", (0..30).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    let mut written: BTreeMap<String, u64> = (0..30).map(|i| (i.to_string(), 1)).collect();
    for round in 0..2 {
        let term = fail_over(&three, "t").await;
        assert_eq!(term, round + 2);
        // New docs and updates of old ones in the new term, from every coordinator.
        for i in 0..10 {
            let id = if i % 2 == 0 { format!("new-{round}-{i}") } else { (i + round as usize * 10).to_string() };
            let v = three.ok(i % 3, Method::PUT, &format!("/t/_doc/{id}"), Some(json!({"n": i, "round": round}))).await;
            assert_eq!(v["_primary_term"], term, "{v}");
            written.insert(id, term);
        }
        for (id, term) in &written {
            let copies = per_copy(&three, "t", id).await;
            assert!(copies.iter().all(|(st, _)| *st == 200), "{id}: {copies:?}");
            assert!(copies.windows(2).all(|w| w[0].1 == w[1].1), "{id}: copies disagree: {copies:?}");
            assert_eq!(copies[0].1[1], *term, "{id}: {copies:?}");
        }
        // The same through a search on each copy, once every copy refreshed:
        // with `refresh_interval: -1` a search shows each copy's last
        // refresh, and a copy recovered after the writes (the failed-over
        // one, re-recovered) refreshed at the end of its recovery while the
        // others did not (as in ES). The realtime GETs above already showed
        // that every copy holds every write.
        three.ok(0, Method::POST, "/t/_refresh", None).await;
        let mut seen: Option<Vec<Value>> = None;
        for n in &three.nodes {
            let v = three.ok(1, Method::POST, &format!("/t/_search?preference=_only_nodes:{}&size=100&seq_no_primary_term=true&version=true", n.local.id), Some(json!({"sort": ["_doc"]}))).await;
            let mut hits: Vec<Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_seq_no"], h["_primary_term"], h["_version"]])).collect();
            hits.sort_by_key(|h| h[0].as_str().unwrap().to_string());
            if let Some(s) = &seen {
                assert_eq!(&hits, s, "copy on {} differs", n.local.name);
            }
            seen = Some(hits);
        }
        // A compare-and-set built from any copy's read succeeds on the primary.
        for (i, n) in three.nodes.iter().enumerate() {
            let id = format!("cas-{round}-{i}");
            three.ok(0, Method::PUT, &format!("/t/_doc/{id}"), Some(json!({"v": 0}))).await;
            let v = three.ok(0, Method::GET, &format!("/t/_doc/{id}?preference=_only_nodes:{}", n.local.id), None).await;
            let path = format!("/t/_doc/{id}?if_seq_no={}&if_primary_term={}", v["_seq_no"], v["_primary_term"]);
            let (st, r) = three.call(i, Method::PUT, &path, Some(json!({"v": 1}))).await;
            assert_eq!(st, 200, "CAS from the copy on {}: {r}", n.local.name);
            written.insert(id, term);
        }
    }
    // Realtime GET on every copy right after the ack, without a refresh: the
    // new version, and not found after a delete.
    let v = three.ok(2, Method::PUT, "/t/_doc/rt", Some(json!({"v": 1}))).await;
    let copies = per_copy(&three, "t", "rt").await;
    assert!(copies.iter().all(|(st, c)| *st == 200 && c[0] == v["_seq_no"]), "{copies:?}");
    three.ok(1, Method::DELETE, "/t/_doc/rt", None).await;
    let copies = per_copy(&three, "t", "rt").await;
    assert!(copies.iter().all(|(st, c)| *st == 404 && c[3] == false), "{copies:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn external_versions_survive_a_failover() {
    let three = start("rs-extver").await;
    three.ok(0, Method::PUT, "/ev", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 2}}))).await;
    three.green("ev").await;
    three.ok(0, Method::PUT, "/ev/_doc/1?version=5&version_type=external", Some(json!({"a": 1}))).await;
    three.ok(1, Method::DELETE, "/ev/_doc/1?version=6&version_type=external", None).await;
    let (st, v) = three.call(2, Method::PUT, "/ev/_doc/1?version=4&version_type=external", Some(json!({"a": 2}))).await;
    assert_eq!(st, 409, "{v}");
    fail_over(&three, "ev").await;
    // The new primary was a replica: it holds the delete's version too.
    for coord in 0..3 {
        let (st, v) = three.call(coord, Method::PUT, "/ev/_doc/1?version=4&version_type=external", Some(json!({"a": 2}))).await;
        assert_eq!(st, 409, "via n{coord}: {v}");
    }
    let v = three.ok(0, Method::PUT, "/ev/_doc/1?version=7&version_type=external", Some(json!({"a": 3}))).await;
    assert_eq!(v["_version"], 7, "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn knn_dfs_scores_the_top_doc_not_its_namesake_under_another_routing() {
    let three = start("rs-knn").await;
    three.ok(0, Method::PUT, "/kv", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1},
        "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4, "index": true, "similarity": "l2_norm"}, "t": {"type": "keyword"}}}}))).await;
    three.green("kv").await;
    let (r0, r1) = (routing_to(&three, "kv", 0).await, routing_to(&three, "kv", 1).await);
    // The same _id on both shards: near the query on shard 0, far on shard 1.
    three.bulk("kv", [("dup".to_string(), Some(r0), json!({"v": [1.0, 0.0, 0.0, 0.0], "t": "a"})), ("dup".to_string(), Some(r1), json!({"v": [-9.0, 9.0, 9.0, 9.0], "t": "b"}))].into_iter()).await;
    for coord in 0..3 {
        // kNN with a query: ES's DFS folds the global top k into the query.
        let v = three.ok(coord, Method::POST, "/kv/_search", Some(json!({"knn": {"field": "v", "query_vector": [1.0, 0.0, 0.0, 0.0], "k": 1, "num_candidates": 10},
            "query": {"match_none": {}}, "size": 10}))).await;
        let hits = v["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "exactly k kNN hits: {v}");
        assert_eq!(hits[0]["_routing"], json!(routing_to(&three, "kv", 0).await), "{v}");
    }
}

/// Writes across rollovers land in the write index each resolved. In-process
/// nodes cannot be paused or killed, so no attempt here loses its answer:
/// the lost-answer retry across a rollover is forced in
/// `replicated_writes.rs` (`an_indeterminate_data_stream_write_is_retried_where_it_was_resolved`).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn data_stream_writes_land_in_the_resolved_write_index() {
    let three = start("rs-ds").await;
    three.ok(0, Method::PUT, "/_index_template/ds", Some(json!({"index_patterns": ["logs-rs-*"], "data_stream": {},
        "template": {"settings": {"number_of_shards": 2, "number_of_replicas": 1}}}))).await;
    let mut ids = BTreeSet::new();
    for round in 0..4 {
        for i in 0..12 {
            let v = three.ok(i % 3, Method::POST, "/logs-rs-a/_doc?refresh=true", Some(json!({"@timestamp": "2026-10-23T00:00:00Z", "n": i}))).await;
            assert!(ids.insert(v["_id"].as_str().unwrap().to_string()), "{v}");
            let ds = three.ok(0, Method::GET, "/_data_stream/logs-rs-a", None).await;
            let write_index = ds["data_streams"][0]["indices"].as_array().unwrap().last().unwrap()["index_name"].clone();
            assert_eq!(v["_index"], write_index, "round {round}: {v}");
        }
        three.ok(round % 3, Method::POST, "/logs-rs-a/_rollover", None).await;
    }
    let v = three.ok(1, Method::POST, "/logs-rs-a/_search?size=100", Some(json!({"track_total_hits": true}))).await;
    assert_eq!(v["hits"]["total"]["value"], 48, "{v}");
    let found: BTreeSet<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    assert_eq!(found, ids, "no duplicate or lost _id across backing indices");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn conflicting_dynamic_types_converge_with_replicas() {
    let three = start("rs-dyn").await;
    three.ok(0, Method::PUT, "/conflict", Some(json!({"settings": {"number_of_shards": 4, "number_of_replicas": 2}}))).await;
    three.green("conflict").await;
    let writes = (0..60).map(|i| {
        let three = &three;
        let value = match i % 3 { 0 => json!(7), 1 => json!("seven"), _ => json!({"nested": true}) };
        async move { (i, three.call(i % 3, Method::PUT, &format!("/conflict/_doc/{i}?refresh=true"), Some(json!({"x": value, "y": i}))).await) }
    });
    let results = futures::future::join_all(writes).await;
    let ok = results.iter().filter(|(_, (st, _))| (200..300).contains(st)).count();
    assert!(ok > 0, "{results:?}");
    assert!(results.iter().all(|(_, (st, _))| (200..300).contains(st) || *st == 400), "{results:?}");
    three.green("conflict").await;
    // No copy failed or re-recovered over a mapping.
    let recoveries = three.ok(0, Method::GET, "/_cat/recovery/conflict?format=json&h=shard,type,stage", None).await;
    assert!(recoveries.as_array().unwrap().iter().all(|r| r["type"] != "peer" || r["stage"] == "done"), "{recoveries}");
    let st = three.nodes[0].state();
    let committed = st.metadata.indices["conflict"].mappings.clone();
    for n in &three.nodes {
        for (_, engine) in n.all_shard_engines() {
            if let Some(svc) = engine.get_index("conflict") {
                assert_eq!(svc.mapping_json()["properties"]["x"]["type"], committed["properties"]["x"]["type"], "{} diverged", n.local.name);
            }
        }
    }
    // Every copy counts the same documents.
    let mut counts = BTreeSet::new();
    for n in &three.nodes {
        let v = three.ok(0, Method::GET, &format!("/conflict/_count?preference=_only_nodes:{}", n.local.id), None).await;
        counts.insert(v["count"].as_u64().unwrap());
    }
    assert_eq!(counts, BTreeSet::from([ok as u64]), "per-copy counts");
}

/// Shard directories (`indices/<uuid>/shard-<n>`) under `dir`.
fn shard_dirs(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = e.path();
            if !path.is_dir() {
                continue;
            }
            // The master's scratch engine: a metadata-only copy of every
            // index, wherever its shards are allocated (no shard data).
            if path.file_name().is_some_and(|n| n == "scratch-engine") && path.parent().and_then(|p| p.file_name()).is_some_and(|n| n == "_cluster") {
                continue;
            }
            let in_indices = path.parent().and_then(|p| p.parent()).and_then(|p| p.file_name()).is_some_and(|n| n == "indices");
            if in_indices && path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("shard-")) {
                out.push(path);
            } else {
                stack.push(path);
            }
        }
    }
    out
}

/// A scroll whose shard relocates away mid-scroll (0 replicas). ES 9.5.5
/// keeps the scroll's reader context: `SearchService.afterIndexRemoved`
/// frees contexts only for DELETED / CLOSED / REOPENED, not for a copy that
/// is no longer assigned, and a scroll page runs on the context's own
/// `IndexShard` and searcher (no `indexServiceSafe` lookup). So the scroll
/// goes on from the moved copy's pinned searcher: every page a 200 with no
/// failed shard, all 300 docs, none twice. Clearing the scroll releases the
/// context; then nothing stays open and, at the next routing change
/// (`IndicesStore` deletes unallocated shard data on routing changes once
/// the shard lock is free), the source node keeps no shard directory.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_scroll_over_a_relocated_shard_keeps_its_context_until_cleared() {
    let three = start("rs-scroll").await;
    three.ok(0, Method::PUT, "/sc", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}}))).await;
    three.green("sc").await;
    three.bulk("sc", (0..300).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    // The node holding shard 0 is the one whose copies move.
    let copies = three.copies("sc").await;
    let holder = copies[&0].iter().next().unwrap().clone();
    let mover = three.nodes.iter().position(|n| n.local.id == holder).unwrap();
    let first = three.ok(0, Method::POST, "/sc/_search?scroll=1m", Some(json!({"size": 10, "sort": ["n"]}))).await;
    let mut id = first["_scroll_id"].as_str().unwrap().to_string();
    let mut seen: BTreeSet<String> = first["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    let name = three.nodes[mover].local.name.clone();
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"cluster.routing.allocation.exclude._name": name}}))).await;
    three.green("sc").await;
    // Green on the master is not enough: wait until the source node itself
    // has applied the handoff and holds no copy of `sc`.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let st = three.nodes[mover].state();
        let held = st.routing_table.indices.get("sc").is_some_and(|t| t.shards.iter().any(|s| s.copies.iter().any(|c| c.node_id.as_deref() == Some(holder.as_str()))));
        if !held {
            break;
        }
        assert!(Instant::now() < deadline, "the source node still holds a copy of sc");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    loop {
        let (st, v) = three.call(0, Method::POST, "/_search/scroll", Some(json!({"scroll_id": id, "scroll": "1m"}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(shards_failed(&v), 0, "the moved shard's context keeps serving: {v}");
        let hits = v["hits"]["hits"].as_array().unwrap().clone();
        if hits.is_empty() {
            break;
        }
        for h in hits {
            assert!(seen.insert(h["_id"].as_str().unwrap().to_string()), "repeated {h}");
        }
        id = v["_scroll_id"].as_str().unwrap().to_string();
    }
    assert_eq!(seen.len(), 300, "every doc, from the pinned searchers");
    three.ok(0, Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": [id]}))).await;
    // A routing change on another node: the moment ES deletes unallocated
    // shard data.
    let other = three.nodes[(mover + 1) % 3].local.name.clone();
    three.ok(0, Method::PUT, "/sc-nudge", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": other}}))).await;
    three.green("sc-nudge").await;
    let data = std::env::temp_dir().join("rustic-replica-search").join(format!("{}-rs-scroll-n{mover}", std::process::id())).join("data");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (_, stats) = three.call(0, Method::GET, "/_nodes/stats/indices/search", None).await;
        let open: u64 = stats["nodes"].as_object().into_iter().flatten().map(|(_, n)| n["indices"]["search"]["open_contexts"].as_u64().unwrap_or(0)).sum();
        let left = shard_dirs(&data);
        if open == 0 && left.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "{open} contexts open; shard dirs left on the source node: {left:?}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let _ = three.call(0, Method::PUT, "/_cluster/settings", Some(json!({"transient": {"cluster.routing.allocation.exclude._name": ""}}))).await;
}

/// B1: after a failover, a full recovery of a new copy ships the documents
/// the old term wrote with that term, and completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_copy_added_after_a_failover_recovers() {
    let three = start("rs-recover-after-failover").await;
    three.ok(0, Method::PUT, "/r", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1}}))).await;
    three.green("r").await;
    three.bulk("r", (0..50).map(|i| (i.to_string(), None, json!({"n": i})))).await;
    assert_eq!(fail_over(&three, "r").await, 2);
    // A third copy: a full snapshot of term-1 documents under term 2.
    three.ok(0, Method::PUT, "/r/_settings", Some(json!({"index": {"number_of_replicas": 2}}))).await;
    three.green("r").await;
    for n in &three.nodes {
        let v = three.ok(0, Method::GET, &format!("/r/_count?preference=_only_nodes:{}", n.local.id), None).await;
        assert_eq!(v["count"], 50, "{v}");
    }
    let copies = per_copy(&three, "r", "7").await;
    assert!(copies.iter().all(|(st, c)| *st == 200 && c[1] == 1), "{copies:?}");
}

/// H-A: whole-index reads the engine serves for the cluster without a shard
/// selection (a rank plan phase reading metadata, _mvt, _fleet_search) are
/// not refused for the shards this node does not hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn rank_mvt_and_fleet_search_work_over_a_multi_shard_index() {
    let three = start("rs-rank-mvt").await;
    three.ok(0, Method::PUT, "/rk", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0},
        "mappings": {"properties": {"t": {"type": "text"}, "k": {"type": "keyword"}, "loc": {"type": "geo_point"}}}}))).await;
    three.green("rk").await;
    three.bulk("rk", (0..30).map(|i| (i.to_string(), None, json!({"t": format!("word{} common", i % 5), "k": format!("k{}", i % 3), "loc": {"lat": 1.0 + i as f64 / 100.0, "lon": 2.0}})))).await;
    for coord in 0..3 {
        let (st, v) = three.call(coord, Method::POST, "/rk/_search", Some(json!({"retriever": {"rrf": {"retrievers": [
            {"standard": {"query": {"match": {"t": "common"}}}},
            {"standard": {"query": {"term": {"k": "k1"}}}}
        ], "rank_window_size": 30, "rank_constant": 1}}, "size": 30}))).await;
        assert_eq!(st, 200, "rrf via n{coord}: {v}");
        assert_eq!(v["hits"]["hits"].as_array().map(Vec::len), Some(30), "{v}");
        // RRF fuses every shard's docs: all 30 come back, each once.
        let ids: BTreeSet<&str> = v["hits"]["hits"].as_array().unwrap().iter().filter_map(|h| h["_id"].as_str()).collect();
        assert_eq!(ids.len(), 30, "{v}");
        let r = three.c.post(format!("http://127.0.0.1:{}/rk/_mvt/loc/0/0/0", three.http[coord])).json(&json!({"grid_precision": 0})).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200, "_mvt via n{coord}");
        let tile = mvt_layers(&r.bytes().await.unwrap());
        // Every shard's docs are features of the hits layer, and the meta
        // layer counts them all.
        assert_eq!(tile.get("hits").map(|l| l.0), Some(30), "_mvt hits layer via n{coord}: {tile:?}");
        assert_eq!(tile.get("meta").and_then(|l| l.1.get("hits.total.value").cloned()), Some(30), "_mvt meta via n{coord}: {tile:?}");
        let (st, v) = three.call(coord, Method::POST, "/rk/_fleet/_fleet_search", Some(json!({"query": {"match_all": {}}, "track_total_hits": true}))).await;
        assert_eq!(st, 200, "_fleet_search via n{coord}: {v}");
        assert_eq!((v["hits"]["total"]["value"].as_u64(), v["_shards"]["total"].as_u64(), shards_failed(&v)), (Some(30), Some(3), 0), "_fleet_search via n{coord}: {v}");
    }
}

/// Layers of a Mapbox vector tile: name → (feature count, integer
/// properties of its first feature). Just enough protobuf to read counts.
fn mvt_layers(b: &[u8]) -> BTreeMap<String, (usize, BTreeMap<String, i64>)> {
    fn varint(b: &[u8], p: &mut usize) -> u64 {
        let (mut v, mut shift) = (0u64, 0);
        while *p < b.len() {
            let byte = b[*p];
            *p += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        v
    }
    // (field, wire type, bytes of a length-delimited field / varint value)
    fn fields(b: &[u8]) -> Vec<(u64, Vec<u8>, u64)> {
        let (mut out, mut p) = (Vec::new(), 0);
        while p < b.len() {
            let key = varint(b, &mut p);
            match key & 7 {
                0 => { let v = varint(b, &mut p); out.push((key >> 3, Vec::new(), v)); }
                1 => { p += 8; }
                2 => { let n = varint(b, &mut p) as usize; out.push((key >> 3, b[p..p + n].to_vec(), 0)); p += n; }
                5 => { p += 4; }
                _ => break,
            }
        }
        out
    }
    let mut layers = BTreeMap::new();
    for (f, layer, _) in fields(b) {
        if f != 3 {
            continue;
        }
        let parts = fields(&layer);
        let name = parts.iter().find(|p| p.0 == 1).map(|p| String::from_utf8_lossy(&p.1).to_string()).unwrap_or_default();
        let features: Vec<&Vec<u8>> = parts.iter().filter(|p| p.0 == 2).map(|p| &p.1).collect();
        let keys: Vec<String> = parts.iter().filter(|p| p.0 == 3).map(|p| String::from_utf8_lossy(&p.1).to_string()).collect();
        // Value: int_value (4), uint_value (5), sint_value (6), double (3).
        let values: Vec<Option<i64>> = parts.iter().filter(|p| p.0 == 4).map(|p| {
            let vf = fields(&p.1);
            vf.iter().find(|v| matches!(v.0, 4 | 5)).map(|v| v.2 as i64)
                .or_else(|| vf.iter().find(|v| v.0 == 6).map(|v| ((v.2 >> 1) as i64) ^ -((v.2 & 1) as i64)))
                .or_else(|| {
                    // A double is wire type 1: re-read it raw.
                    let mut q = 0;
                    let key = varint(&p.1, &mut q);
                    (key == (3 << 3 | 1) && p.1.len() >= q + 8).then(|| f64::from_le_bytes(p.1[q..q + 8].try_into().unwrap()) as i64)
                })
        }).collect();
        let mut props = BTreeMap::new();
        if let Some(first) = features.first() {
            // Packed tags (field 2): key index, value index pairs.
            if let Some(tags) = fields(first).into_iter().find(|t| t.0 == 2) {
                let (mut q, mut ix) = (0, Vec::new());
                while q < tags.1.len() {
                    ix.push(varint(&tags.1, &mut q) as usize);
                }
                for pair in ix.chunks(2) {
                    if let (Some(k), Some(Some(v))) = (keys.get(pair[0]), pair.get(1).and_then(|i| values.get(*i))) {
                        props.insert(k.clone(), *v);
                    }
                }
            }
        }
        layers.insert(name, (features.len(), props));
    }
    layers
}

/// Fleet's read-your-writes: a write acknowledged on one node, then a fleet
/// search from another node waiting for its shard checkpoints sees it; a
/// checkpoint no shard issued fails that shard (partial results unless
/// `allow_partial_search_results=false`).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fleet_search_waits_for_its_checkpoints_on_every_shard() {
    let three = start("rs-fleet-wait").await;
    three.ok(0, Method::PUT, "/fl", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1, "refresh_interval": "1s"}}))).await;
    three.green("fl").await;
    let (r0, r1) = (routing_to(&three, "fl", 0).await, routing_to(&three, "fl", 1).await);
    for round in 0..3 {
        // Written on node A, not refreshed.
        let mut checkpoints = vec![-1i64; 2];
        for (k, r) in [(0, &r0), (1, &r1)] {
            let v = three.ok(0, Method::PUT, &format!("/fl/_doc/d{round}-{k}?routing={r}"), Some(json!({"round": round}))).await;
            checkpoints[k] = v["_seq_no"].as_i64().unwrap();
        }
        let cps = checkpoints.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
        // Searched from node B (and C), waiting for those checkpoints.
        for coord in 1..3 {
            let v = three.ok(coord, Method::POST, &format!("/fl/_fleet/_fleet_search?wait_for_checkpoints={cps}"), Some(json!({"query": {"term": {"round": round}}, "track_total_hits": true}))).await;
            assert_eq!(v["hits"]["total"]["value"], 2, "round {round} via n{coord}: {v}");
            assert_eq!((v["_shards"]["total"].as_u64(), shards_failed(&v)), (Some(2), 0), "{v}");
        }
        // The same through _fleet_msearch.
        let body = format!("{{\"index\":\"fl\",\"wait_for_checkpoints\":[{cps}]}}\n{{\"query\":{{\"term\":{{\"round\":{round}}}}},\"track_total_hits\":true}}\n");
        let r = three.c.post(format!("http://127.0.0.1:{}/_fleet/_fleet_msearch", three.http[2])).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["responses"][0]["hits"]["total"]["value"], 2, "{v}");
        assert_eq!(v["responses"][0]["status"], 200, "{v}");
    }
    // A checkpoint shard 1 never issued: that shard fails.
    let (st, v) = three.call(1, Method::POST, "/fl/_fleet/_fleet_search?wait_for_checkpoints=0,100000&wait_for_checkpoints_timeout=2s", Some(json!({"track_total_hits": true}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["_shards"]["total"].as_u64(), shards_failed(&v)), (Some(2), 1), "{v}");
    assert_eq!(v["_shards"]["failures"][0]["shard"], 1, "{v}");
    let (st, v) = three.call(2, Method::POST, "/fl/_fleet/_fleet_search?wait_for_checkpoints=0,100000&allow_partial_search_results=false", Some(json!({}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("search_phase_execution_exception")), "{v}");
    // Every shard: all shards failed.
    let (st, v) = three.call(0, Method::POST, "/fl/_fleet/_fleet_search?wait_for_checkpoints=100000,100000", Some(json!({}))).await;
    assert_eq!((st, v["error"]["reason"].as_str()), (400, Some("all shards failed")), "{v}");
}

/// A checkpointed fleet search served by a replica answers promptly: the
/// replica knows the global checkpoint its replication requests and
/// checkpoint syncs carried (ES `getLastKnownGlobalCheckpoint`), so its
/// wait does not run into the timeout and over to the primary.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_fleet_search_served_by_a_replica_answers_promptly() {
    let three = start("rs-fleet-replica").await;
    three.ok(0, Method::PUT, "/flr", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 2, "refresh_interval": "1s"}}))).await;
    three.green("flr").await;
    let st = three.nodes[0].state();
    let table = st.routing_table.shard_table("flr", 0).unwrap();
    let primary = table.active_primary().and_then(|c| c.node_id.clone()).unwrap();
    let replicas: Vec<String> = table.copies.iter().filter(|c| !c.primary && c.active()).filter_map(|c| c.node_id.clone()).collect();
    assert_eq!(replicas.len(), 2, "{table:?}");
    // The coordinator has applied the green state too (health answers from
    // the master's): it sees both replicas started.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !three.nodes[1].state().routing_table.shard_table("flr", 0).is_some_and(|t| t.copies.iter().all(|c| c.active())) {
        assert!(Instant::now() < deadline, "n1 never saw every copy of flr started");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    for round in 0..3 {
        let v = three.ok(0, Method::PUT, &format!("/flr/_doc/r{round}"), Some(json!({"round": round}))).await;
        let seq_no = v["_seq_no"].as_i64().unwrap();
        for replica in &replicas {
            let started = Instant::now();
            // Only the replica's copy may serve it (no fallback to the primary).
            let v = three.ok(1, Method::POST, &format!("/flr/_fleet/_fleet_search?wait_for_checkpoints={seq_no}&wait_for_checkpoints_timeout=20s&preference=_only_nodes:{replica}"),
                Some(json!({"query": {"term": {"round": round}}, "track_total_hits": true}))).await;
            assert_eq!((v["hits"]["total"]["value"].as_u64(), shards_failed(&v)), (Some(1), 0), "round {round} on {replica} (primary {primary}): {v}");
            assert!(started.elapsed() < Duration::from_secs(5), "the replica waited {:?}", started.elapsed());
        }
    }
}

/// A healthy remote batch that takes longer than the former 60s coordinator
/// bound must finish once, not launch its items again while it is still live.
/// The transport handler delays the answer deterministically; no expensive
/// query or timing-dependent script is needed to reproduce the scheduler bug.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_remote_msearch_batch_is_not_executed_again_after_sixty_seconds() {
    use elasticsearch_node::front::{RestReq, RestResp, ACTION_FORWARD};
    use elasticsearch_transport::RemoteError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let three = start("rs-slow-msearch").await;
    three.ok(0, Method::PUT, "/slow-batch", Some(json!({"settings": {
        "number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": "n2"
    }}))).await;
    three.green("slow-batch").await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    three.nodes[2].transport.register_handler(ACTION_FORWARD, move |req: RestReq| {
        let observed = observed.clone();
        async move {
            observed.fetch_add(1, Ordering::SeqCst);
            if !req.uri.contains("_msearch") {
                return Ok::<RestResp, RemoteError>(RestResp::error(500, "duplicate_execution", "a live batch was retried as an item"));
            }
            tokio::time::sleep(Duration::from_secs(62)).await;
            let item = json!({"hits":{"total":{"value":0,"relation":"eq"},"hits":[]},"status":200});
            Ok(RestResp::json(200, &json!({"responses":[item.clone(), item]})))
        }
    });
    let response = three.c.post(format!("http://127.0.0.1:{}/slow-batch/_msearch", three.http[0]))
        .timeout(Duration::from_secs(90)).header("content-type", "application/x-ndjson")
        .body("{}\n{}\n{}\n{}\n").send().await.unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1, "still-running batch was duplicated: {body}");
    assert_eq!(body["responses"].as_array().unwrap().len(), 2);
    for item in body["responses"].as_array().unwrap() {
        assert_eq!(item["status"], 200, "{body}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn msearch_request_errors_survive_remote_batch_without_item_retries() {
    use elasticsearch_node::front::{RestReq, RestResp, ACTION_FORWARD};
    use elasticsearch_transport::RemoteError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let three = start("rs-msearch-request-errors").await;
    three.ok(0, Method::PUT, "/request-errors", Some(json!({"settings": {
        "number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": "n2"
    }}))).await;
    three.green("request-errors").await;
    // A request error is the whole request's answer. A breaker trip on the
    // node running the batch is a data-node failure: ES retries the shard
    // on its next copy, so the items run one by one.
    for (status, kind, reason, request_wide) in [
        (429, "circuit_breaking_exception", "[request] Data too large, data for [<msearch_response>]", false),
        (400, "illegal_argument_exception", "[rest_total_hits_as_int] cannot be used if the tracking of total hits is not accurate, got 10", true),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        three.nodes[2].transport.register_handler(ACTION_FORWARD, move |req: RestReq| {
            let observed = observed.clone();
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                if !req.uri.contains("_msearch") {
                    return Ok::<RestResp, RemoteError>(RestResp::error(500, "item_retry", "item retried on its own"));
                }
                Ok(RestResp::json(status, &json!({"error":{"type":kind,"reason":reason,"custom_detail":"preserved"},"status":status})))
            }
        });
        let response = three.c.post(format!("http://127.0.0.1:{}/request-errors/_msearch", three.http[0]))
            .header("content-type", "application/x-ndjson").body("{}\n{}\n{}\n{}\n").send().await.unwrap();
        let code = response.status().as_u16();
        let value: Value = response.json().await.unwrap();
        if request_wide {
            assert_eq!(code, status, "{value}");
            assert_eq!(value["error"]["type"], kind);
            assert_eq!(value["error"]["custom_detail"], "preserved");
            assert_eq!(calls.load(Ordering::SeqCst), 1, "request-wide error must not execute items again: {value}");
        } else {
            assert_eq!(code, 200, "a remote breaker trip must not fail the whole msearch: {value}");
            let items = value["responses"].as_array().unwrap();
            assert_eq!(items.len(), 2, "{value}");
            // Each item is answered by its own retry (here: a copy failure).
            for item in items { assert!(item["error"].is_object() && item["status"].as_u64().is_some_and(|s| s >= 500), "{value}"); }
            assert!(calls.load(Ordering::SeqCst) >= 3, "items were not retried individually: {value}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_an_msearch_stops_waiting_for_its_remote_batch_and_names_the_parent() {
    use elasticsearch_node::front::{RestReq, RestResp, ACTION_FORWARD};
    use elasticsearch_transport::RemoteError;
    use std::sync::Mutex;
    let three = start("rs-msearch-cancel").await;
    three.ok(0, Method::PUT, "/cancel-batch", Some(json!({"settings": {
        "number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": "n2"
    }}))).await;
    three.green("cancel-batch").await;
    let parents = Arc::new(Mutex::new(Vec::<String>::new()));
    let observed = parents.clone();
    three.nodes[2].transport.register_handler(ACTION_FORWARD, move |req: RestReq| {
        let observed = observed.clone();
        async move {
            if req.uri.contains("_msearch") {
                let parent = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("x-es-cluster-parent-task")).map(|(_, v)| v.clone());
                observed.lock().unwrap().push(parent.unwrap_or_default());
                // A stuck-but-pinging node: the batch never answers.
                std::future::pending::<()>().await;
            }
            Ok::<RestResp, RemoteError>(RestResp::error(500, "unexpected", "items must not run after cancellation"))
        }
    });
    let tasks = |three: &Three| {
        let c = three.c.clone();
        let port = three.http[0];
        async move {
            let v: Value = c.get(format!("http://127.0.0.1:{port}/_tasks?actions=indices:data/read/msearch&detailed")).send().await.unwrap().json().await.unwrap();
            v["nodes"].as_object().map(|nodes| nodes.values().flat_map(|n| n["tasks"].as_object().map(|t| t.keys().cloned().collect::<Vec<_>>()).unwrap_or_default()).collect::<Vec<_>>()).unwrap_or_default()
        }
    };
    // `_tasks/{id}/_cancel` answers the msearch promptly.
    let client = three.c.clone();
    let url = format!("http://127.0.0.1:{}/cancel-batch/_msearch", three.http[0]);
    let pending = tokio::spawn(async move {
        client.post(url).header("content-type", "application/x-ndjson").body("{}\n{}\n{}\n{}\n").send().await.unwrap()
    });
    let deadline = Instant::now() + Duration::from_secs(20);
    let id = loop {
        if let Some(id) = tasks(&three).await.into_iter().next() { break id; }
        assert!(Instant::now() < deadline, "msearch task never listed");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    while parents.lock().unwrap().is_empty() {
        assert!(Instant::now() < deadline, "batch never forwarded");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(parents.lock().unwrap()[0], id, "the batch must carry its coordinator task as parent");
    three.ok(0, Method::POST, &format!("/_tasks/{id}/_cancel"), None).await;
    let response = tokio::time::timeout(Duration::from_secs(10), pending).await.expect("cancelled msearch kept waiting").unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let value: Value = response.json().await.unwrap();
    assert_eq!(value["error"]["type"], "task_cancelled_exception", "{value}");
    assert!(tasks(&three).await.is_empty());
    // A client that goes away ends the task too (its future is dropped).
    let gone = three.c.post(format!("http://127.0.0.1:{}/cancel-batch/_msearch", three.http[0])).timeout(Duration::from_secs(2))
        .header("content-type", "application/x-ndjson").body("{}\n{}\n{}\n{}\n").send().await;
    assert!(gone.is_err());
    let deadline = Instant::now() + Duration::from_secs(20);
    while !tasks(&three).await.is_empty() {
        assert!(Instant::now() < deadline, "msearch task outlived its disconnected client");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
