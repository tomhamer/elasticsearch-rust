//! Circuit breakers end to end: a high-cardinality terms aggregation and a
//! large ES|QL STATS on a small request breaker fail with ES's 429
//! `circuit_breaking_exception`, the node keeps serving, the breakers
//! return to zero and `_nodes/stats` counts the trips; a shard that trips
//! alone is a partial failure; PIT contexts show in `accounting` until
//! closed; an oversized body trips `inflight_requests` (a `_bulk`, streamed
//! in ES 9, does not).
//!
//! The breakers are the process's, so the tests here run one at a time.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let es = Es { base: format!("http://{addr}"), c: Client::new() };
    // Whatever an earlier test left: back to the defaults.
    es.limits(json!({"indices.breaker.request.limit": null, "network.breaker.inflight_requests.limit": null})).await;
    es
}

impl Es {
    async fn send(&self, m: Method, p: &str, ct: &str, body: Option<String>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", ct).body(b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.send(Method::GET, p, "application/json", None).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.send(Method::POST, p, "application/json", Some(b.to_string())).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.send(Method::PUT, p, "application/json", Some(b.to_string())).await
    }
    async fn delete(&self, p: &str, b: Value) -> (u16, Value) {
        self.send(Method::DELETE, p, "application/json", Some(b.to_string())).await
    }
    async fn bulk(&self, index: &str, body: String) -> (u16, Value) {
        self.send(Method::POST, &format!("/{index}/_bulk?refresh=true"), "application/x-ndjson", Some(body)).await
    }
    async fn limits(&self, transient: Value) {
        let (st, v) = self.put("/_cluster/settings", json!({ "transient": transient })).await;
        assert_eq!(st, 200, "{v}");
    }
    async fn breakers(&self) -> Value {
        let (st, v) = self.get("/_nodes/stats/breaker").await;
        assert_eq!(st, 200, "{v}");
        v["nodes"].as_object().unwrap().values().next().unwrap()["breakers"].clone()
    }
    /// Index `n` docs with distinct `k` values (`prefix-i`).
    async fn fill(&self, index: &str, shards: u32, n: usize, prefix: &str) {
        let (st, v) = self.put(&format!("/{index}"), json!({"settings": {"number_of_shards": shards}, "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}})).await;
        assert_eq!(st, 200, "{v}");
        for chunk in (0..n).collect::<Vec<_>>().chunks(5000) {
            let mut body = String::new();
            for i in chunk {
                body.push_str("{\"index\":{}}\n");
                body.push_str(&json!({"k": format!("{prefix}-{i:06}"), "n": i}).to_string());
                body.push('\n');
            }
            let (st, v) = self.bulk(index, body).await;
            assert_eq!(st, 200, "{v}");
            assert_eq!(v["errors"], false, "{v}");
        }
    }
    /// Every transient breaker reservation is released.
    async fn assert_released(&self) {
        let b = self.breakers().await;
        assert_eq!(b["request"]["estimated_size_in_bytes"], 0, "{b}");
        assert_eq!(b["inflight_requests"]["estimated_size_in_bytes"], 0, "{b}");
        assert_eq!(b["eql_sequence"]["estimated_size_in_bytes"], 0, "{b}");
    }
}

fn assert_breaking(e: &Value, breaker: &str, limit: u64) {
    assert_eq!(e["type"], "circuit_breaking_exception", "{e}");
    let reason = e["reason"].as_str().unwrap();
    assert!(reason.starts_with(&format!("[{breaker}] Data too large, data for [")), "{reason}");
    assert!(reason.contains(&format!("which is larger than the limit of [{limit}/")), "{reason}");
    assert!(reason.contains("; for more information, see https://www.elastic.co/docs/troubleshoot/elasticsearch/circuit-breaker-errors?version="), "{reason}");
    assert_eq!(e["bytes_limit"], limit, "{e}");
    assert!(e["bytes_wanted"].as_u64().unwrap() > limit, "{e}");
    assert_eq!(e["durability"], "TRANSIENT", "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn high_cardinality_terms_trips_the_request_breaker() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.fill("hc", 1, 20_000, "key").await;
    let before = es.breakers().await["request"]["tripped"].as_u64().unwrap();
    es.limits(json!({"indices.breaker.request.limit": "1mb"})).await;
    let (st, v) = es.post("/hc/_search", json!({"size": 0, "aggs": {"t": {"terms": {"field": "k"}}}})).await;
    assert_eq!(st, 429, "{v}");
    assert_eq!(v["status"], 429);
    // ES: the shard failed, so did the search (all shards failed).
    assert_eq!(v["error"]["type"], "search_phase_execution_exception", "{v}");
    assert_eq!(v["error"]["reason"], "all shards failed");
    assert_breaking(&v["error"]["root_cause"][0], "request", 1 << 20);
    assert!(v["error"]["root_cause"][0]["reason"].as_str().unwrap().contains("data for [<agg [t]>]"), "{v}");
    assert_breaking(&v["error"]["failed_shards"][0]["reason"], "request", 1 << 20);
    // The node keeps serving, smaller aggregations included.
    let (st, v) = es.get("/_cluster/health").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/hc/_search", json!({"size": 0, "query": {"term": {"k": "key-000007"}}, "aggs": {"t": {"terms": {"field": "k"}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["aggregations"]["t"]["buckets"][0]["key"], "key-000007");
    es.assert_released().await;
    let b = es.breakers().await;
    assert_eq!(b["request"]["limit_size_in_bytes"], 1 << 20);
    assert_eq!(b["request"]["limit_size"], "1mb");
    assert!(b["request"]["tripped"].as_u64().unwrap() > before, "{b}");
    // Back to the default limit: the same aggregation succeeds.
    es.limits(json!({"indices.breaker.request.limit": null})).await;
    let (st, v) = es.post("/hc/_search", json!({"size": 0, "aggs": {"t": {"terms": {"field": "k", "size": 3}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["aggregations"]["t"]["buckets"].as_array().unwrap().len(), 3);
    es.assert_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shard_that_trips_alone_is_a_partial_failure() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.fill("big", 2, 20_000, "key").await;
    es.fill("small", 1, 5, "s").await;
    es.limits(json!({"indices.breaker.request.limit": "512kb"})).await;
    let (st, v) = es.post("/big,small/_search", json!({"size": 0, "aggs": {"t": {"terms": {"field": "k", "size": 10}}}})).await;
    assert_eq!(st, 200, "{v}");
    let sh = &v["_shards"];
    assert_eq!(sh["total"], 3, "{v}");
    assert_eq!(sh["failed"], 2, "{v}");
    assert_eq!(sh["successful"], 1, "{v}");
    for f in sh["failures"].as_array().unwrap() {
        assert_eq!(f["index"], "big", "{v}");
        assert_breaking(&f["reason"], "request", 512 << 10);
    }
    let keys: Vec<&str> = v["aggregations"]["t"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_str().unwrap()).collect();
    assert_eq!(keys, ["s-000000", "s-000001", "s-000002", "s-000003", "s-000004"]);
    es.assert_released().await;
    es.limits(json!({"indices.breaker.request.limit": null})).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_esql_stats_trips_the_request_breaker() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.fill("eq", 1, 20_000, "key").await;
    es.limits(json!({"indices.breaker.request.limit": "1mb"})).await;
    let (st, v) = es.post("/_query", json!({"query": "FROM eq | STATS c = COUNT(*), s = SUM(n) BY k | SORT c DESC | LIMIT 5"})).await;
    assert_eq!(st, 429, "{v}");
    assert_eq!(v["status"], 429);
    assert_breaking(&v["error"], "request", 1 << 20);
    let (st, v) = es.get("/_cluster/health").await;
    assert_eq!(st, 200, "{v}");
    es.assert_released().await;
    es.limits(json!({"indices.breaker.request.limit": null})).await;
    let (st, v) = es.post("/_query", json!({"query": "FROM eq | STATS c = COUNT(*) BY k | SORT k | LIMIT 2"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([[1, "key-000000"], [1, "key-000001"]]));
    es.assert_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pit_contexts_are_accounted_until_closed() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.fill("pit", 1, 1_000, "key").await;
    let (st, v) = es.send(Method::POST, "/pit/_flush", "application/json", None).await;
    assert_eq!(st, 200, "{v}");
    // Not shown by ES 9's APIs: read in process.
    let accounting = |_: ()| elasticsearch_engine::memory::process_breakers().accounting.used();
    let before = accounting(());
    let mut ids = Vec::new();
    for _ in 0..8 {
        let (st, v) = es.send(Method::POST, "/pit/_pit?keep_alive=1m", "application/json", None).await;
        assert_eq!(st, 200, "{v}");
        ids.push(v["id"].as_str().unwrap().to_string());
    }
    let open = accounting(());
    // Each context holds at least its own bookkeeping (1kb + per target).
    assert!(open >= before + 8 * 1024, "before {before}, open {open}");
    for id in &ids {
        let (st, v) = es.delete("/_pit", json!({"id": id})).await;
        assert_eq!(st, 200, "{v}");
    }
    let closed = accounting(());
    assert!(closed + 8 * 1024 <= open, "open {open}, closed {closed}");
    es.assert_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oversized_bodies_trip_the_in_flight_breaker() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/inf", json!({})).await;
    assert_eq!(st, 200, "{v}");
    // 1mb with overhead 2: bodies over 512kb trip.
    es.limits(json!({"network.breaker.inflight_requests.limit": "1mb"})).await;
    // A search body of ~700kb (a terms query over many values).
    let values: Vec<String> = (0..3000).map(|i| format!("{i:0>220}")).collect();
    let big = json!({"query": {"terms": {"pad": values}}}).to_string();
    assert!(big.len() > 600 << 10);
    // Two oversized searches at once: both trip, before parsing.
    let search = |body: String| es.send(Method::POST, "/inf/_search", "application/json", Some(body));
    let (a, b) = tokio::join!(search(big.clone()), search(big.clone()));
    for (st, v) in [a, b] {
        assert_eq!(st, 429, "{v}");
        assert_breaking(&v["error"], "inflight_requests", 1 << 20);
        assert!(v["error"]["reason"].as_str().unwrap().contains("data for [<http_request>]"), "{v}");
    }
    // ES 9 streams _bulk (its bytes are indexing pressure's): a bulk as
    // large goes through, and so does a small search.
    let doc = |i: usize| format!("{{\"index\":{{}}}}\n{{\"pad\":\"{}\",\"i\":{i}}}\n", "x".repeat(200));
    let bulk: String = (0..3000).map(doc).collect();
    assert!(bulk.len() > 600 << 10);
    let (st, v) = es.bulk("inf", bulk).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/inf/_search", json!({"size": 0})).await;
    assert_eq!(st, 200, "{v}");
    let b = es.breakers().await;
    assert!(b["inflight_requests"]["tripped"].as_u64().unwrap() >= 2, "{b}");
    assert_eq!(b["inflight_requests"]["overhead"], 2.0);
    es.limits(json!({"network.breaker.inflight_requests.limit": null})).await;
    es.assert_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn breaker_stats_and_settings_match_es() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let b = es.breakers().await;
    let names: Vec<&String> = b.as_object().unwrap().keys().collect();
    // Exactly ES 9's breakers: the internal `accounting` one is never shown.
    assert_eq!(names, ["model_inference", "inflight_requests", "request", "fielddata", "eql_sequence", "parent"]);
    let (st, v) = es.get("/_nodes/stats/breaker?filter_path=nodes.*.breakers.accounting").await;
    assert_eq!(st, 200, "{v}");
    assert!(v.as_object().is_none_or(|o| o.is_empty()), "{v}");
    let (st, v) = es.get("/_cat/circuit_breaker?format=json&h=breaker").await;
    assert_eq!(st, 200, "{v}");
    assert!(!v.as_array().unwrap().iter().any(|r| r["breaker"] == "accounting"), "{v}");
    // ES 9 has no accounting settings: rejected, and absent from the defaults.
    let (st, v) = es.put("/_cluster/settings", json!({"transient": {"indices.breaker.accounting.limit": "50%"}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["reason"], "transient setting [indices.breaker.accounting.limit], not recognized", "{v}");
    let (st, v) = es.get("/_cluster/settings?include_defaults=true&flat_settings=true").await;
    assert_eq!(st, 200, "{v}");
    assert!(v["defaults"].get("indices.breaker.accounting.limit").is_none() && v["defaults"].get("indices.breaker.request.limit").is_some(), "{v}");
    for (name, s) in b.as_object().unwrap() {
        for k in ["limit_size_in_bytes", "estimated_size_in_bytes", "tripped"] {
            assert!(s[k].is_i64() || s[k].is_u64(), "{name}.{k}: {s}");
        }
        assert!(s["limit_size"].is_string() && s["estimated_size"].is_string() && s["overhead"].is_f64(), "{name}: {s}");
    }
    assert_eq!(b["fielddata"]["overhead"], 1.03);
    // The parent tracks the process's real memory.
    assert!(b["parent"]["estimated_size_in_bytes"].as_u64().unwrap() > 0, "{b}");
    let request = b["request"]["limit_size_in_bytes"].as_u64().unwrap();
    let parent = b["parent"]["limit_size_in_bytes"].as_u64().unwrap();
    assert!(parent > request, "{b}");
    // Malformed values are rejected like ES's MemorySizeValue.
    let (st, v) = es.put("/_cluster/settings", json!({"transient": {"indices.breaker.request.limit": "abc"}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "illegal_argument_exception");
    assert_eq!(v["error"]["reason"], "failed to parse setting [indices.breaker.request.limit] with value [abc] as a size in bytes: unit is missing or unrecognized");
    // Overheads and the parent are dynamic too.
    es.limits(json!({"indices.breaker.request.overhead": 2, "indices.breaker.total.limit": "90%"})).await;
    let b = es.breakers().await;
    assert_eq!(b["request"]["overhead"], 2.0);
    assert!(b["parent"]["limit_size_in_bytes"].as_u64().unwrap() < parent);
    es.limits(json!({"indices.breaker.request.overhead": null, "indices.breaker.total.limit": null})).await;
    assert_eq!(es.breakers().await["parent"]["limit_size_in_bytes"], parent);
    let (st, v) = es.get("/_cat/circuit_breaker?format=json&h=breaker,limit_bytes,tripped").await;
    assert_eq!(st, 200, "{v}");
    let text = |x: &Value| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string());
    assert!(v.as_array().unwrap().iter().any(|r| r["breaker"] == "request" && text(&r["limit_bytes"]) == request.to_string()), "{v}");
}

/// ES 9 checks even a streamed `_bulk` (charged 0) against the parent, so
/// real-memory pressure trips it; monitoring and settings requests still
/// get through, a bodyless `_cat` request trips like any other.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streamed_bulk_is_still_parent_checked() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/pb", json!({})).await;
    assert_eq!(st, 200, "{v}");
    /// Real memory reads as over any limit while alive.
    struct Pressure;
    impl Drop for Pressure {
        fn drop(&mut self) {
            elasticsearch_engine::memory::process_breakers().set_real_memory_for_tests(None);
        }
    }
    let pressure = Pressure;
    elasticsearch_engine::memory::process_breakers().set_real_memory_for_tests(Some(u64::MAX / 4));
    let (st, v) = es.bulk("pb", "{\"index\":{}}\n{\"a\":1}\n".into()).await;
    assert_eq!(st, 429, "{v}");
    let reason = v["error"]["reason"].as_str().unwrap_or("");
    assert!(reason.starts_with("[parent] Data too large, data for [<http_request>]"), "{reason}");
    let (st, v) = es.get("/_cat/indices?format=json").await;
    assert_eq!(st, 429, "_cat trips in ES 9: {v}");
    let (st, v) = es.get("/_cluster/health").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.get("/_nodes/stats/breaker").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"transient": {"indices.breaker.request.limit": null}})).await;
    assert_eq!(st, 200, "{v}");
    drop(pressure);
    let (st, v) = es.bulk("pb", "{\"index\":{}}\n{\"a\":1}\n".into()).await;
    assert_eq!(st, 200, "{v}");
}

/// One index fails in the query phase, the other trips in its aggregation:
/// every target failed, so the search fails as ES's "all shards failed"
/// with both failures (the query-phase one's status first).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_failure_and_a_trip_are_all_shards_failed() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/qa", json!({"mappings": {"properties": {"f": {"type": "long"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.bulk("qa", "{\"index\":{}}\n{\"f\":1}\n".into()).await;
    assert_eq!(st, 200, "{v}");
    es.fill("qb", 1, 20_000, "key").await;
    es.limits(json!({"indices.breaker.request.limit": "1mb"})).await;
    // `f` is a long in qa (the term cannot parse there) and unmapped in qb
    // (the term matches nothing there; match_all matches every doc).
    let body = json!({"size": 0, "query": {"bool": {"should": [{"match_all": {}}, {"term": {"f": "notanumber"}}]}}, "aggs": {"t": {"terms": {"field": "k"}}}});
    let (st, v) = es.post("/qa,qb/_search", body).await;
    assert_eq!(v["error"]["type"], "search_phase_execution_exception", "{v}");
    assert_eq!(v["error"]["reason"], "all shards failed", "{v}");
    assert_eq!(st, 400, "the first failure's status (SearchPhaseExecutionException): {v}");
    let failed = v["error"]["failed_shards"].as_array().cloned().unwrap_or_default();
    let kinds: Vec<&str> = failed.iter().filter_map(|f| f["reason"]["type"].as_str()).collect();
    assert!(kinds.contains(&"query_shard_exception") && kinds.contains(&"circuit_breaking_exception"), "{v}");
    es.limits(json!({"indices.breaker.request.limit": null})).await;
    es.assert_released().await;
}

/// The reverse of the above (combined review M28): the index that trips
/// sorts first, so ES's first failure in shard order is the trip and the
/// search answers 429, although the other index failed earlier (in the
/// query phase). `failed_shards` lists them in shard order too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_shards_failed_takes_the_status_in_shard_order() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.fill("ra", 1, 20_000, "key").await;
    let (st, v) = es.put("/rb", json!({"mappings": {"properties": {"f": {"type": "long"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.bulk("rb", "{\"index\":{}}\n{\"f\":1}\n".into()).await;
    assert_eq!(st, 200, "{v}");
    es.limits(json!({"indices.breaker.request.limit": "1mb"})).await;
    let body = json!({"size": 0, "query": {"bool": {"should": [{"match_all": {}}, {"term": {"f": "notanumber"}}]}}, "aggs": {"t": {"terms": {"field": "k"}}}});
    let (st, v) = es.post("/rb,ra/_search", body).await;
    assert_eq!(v["error"]["reason"], "all shards failed", "{v}");
    assert_eq!(st, 429, "ra's trip comes first in shard order: {v}");
    let failed = v["error"]["failed_shards"].as_array().cloned().unwrap_or_default();
    let order: Vec<&str> = failed.iter().filter_map(|f| f["index"].as_str()).collect();
    assert_eq!(order, ["ra", "rb"], "{v}");
    assert_eq!(failed[0]["reason"]["type"], "circuit_breaking_exception", "{v}");
    es.limits(json!({"indices.breaker.request.limit": null})).await;
    es.assert_released().await;
}

/// Mixed-version upgrade, as ES's `archiveUnknownOrInvalidSettings` at
/// metadata load: a persisted setting this version does not know, or whose
/// value no longer validates, is archived (the node starts,
/// `_cluster/settings` shows it under `archived.*`, nothing applies it, and
/// `archived.*: null` removes it). `use_real_memory`, which an older
/// version of this server accepted dynamically, is a known (node) setting:
/// ES keeps it in the cluster settings, and the breakers ignore it there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upgraded_breaker_settings_are_archived() {
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("_state")).unwrap();
    std::fs::write(
        tmp.path().join("_state/cluster.json"),
        json!({"persistent": {
            "indices.breaker.total.use_real_memory": "false",
            "indices.breaker.request.limit": "abc",
            "indices.breaker.fielddata.limit": "30%",
            "cluster.removed_in_this_version": "x",
        }}).to_string(),
    )
    .unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.get("/_cluster/settings?flat_settings=true").await;
    assert_eq!(st, 200, "{v}");
    let p = &v["persistent"];
    assert_eq!(p["archived.indices.breaker.request.limit"], "abc", "{v}");
    assert_eq!(p["archived.cluster.removed_in_this_version"], "x", "{v}");
    assert!(p.get("indices.breaker.request.limit").is_none() && p.get("cluster.removed_in_this_version").is_none(), "{v}");
    assert_eq!(p["indices.breaker.total.use_real_memory"], "false", "a known node setting is kept, as ES: {v}");
    // A still-valid setting is kept and applied.
    assert_eq!(p["indices.breaker.fielddata.limit"], "30%", "{v}");
    let b = es.breakers().await;
    let budget = elasticsearch_engine::memory::budget() as f64;
    let parent = b["parent"]["limit_size_in_bytes"].as_u64().unwrap() as f64;
    assert!((parent / budget - 0.95).abs() < 0.01, "real memory stays on (95%): only the node setting counts: {b}");
    let request = b["request"]["limit_size_in_bytes"].as_u64().unwrap() as f64;
    assert!((request / budget - 0.60).abs() < 0.01, "the archived limit is not applied: {b}");
    // Archived settings can only be removed; the node setting cannot be
    // set dynamically.
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {"archived.indices.breaker.request.limit": "1gb"}})).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {"indices.breaker.total.use_real_memory": "true"}})).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {"archived.*": null, "indices.breaker.fielddata.limit": null}})).await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = es.get("/_cluster/settings?flat_settings=true").await;
    assert_eq!(v["persistent"], json!({"indices.breaker.total.use_real_memory": "false"}), "{v}");
}
