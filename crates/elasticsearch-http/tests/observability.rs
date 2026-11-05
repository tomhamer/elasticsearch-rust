//! HTTP tests for the monitoring / observability APIs: nodes stats, cluster
//! stats, cat thread_pool / recovery / tasks, migration deprecations,
//! health report, validate query, terms enum, explain, profile, blocks,
//! resolve index, search shards, allocation explain, reroute.

use elasticsearch_engine::node::{Node, NODE_ID};
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    start_licensed(dir, None).await
}

/// [`start`] with `xpack.license.self_generated.type` (synthetic source
/// needs an enterprise / trial license: under basic, ES 9.5.5 creates such
/// indices with stored source).
async fn start_licensed(dir: &std::path::Path, license: Option<&str>) -> Es {
    let node = Node::open(dir).unwrap();
    if let Some(l) = license {
        node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!(l));
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn raw(&self, m: Method, p: &str, body: Option<Value>) -> (u16, String) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.text().await.unwrap())
    }
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let (st, t) = self.raw(m, p, body).await;
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.call(Method::GET, p, None).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    /// POST without a body (ES rejects a body on e.g. `_disk_usage`).
    async fn post0(&self, p: &str) -> (u16, Value) {
        self.call(Method::POST, p, None).await
    }
    /// PUT without a body (ES rejects a body on `_block`).
    async fn put0(&self, p: &str) -> (u16, Value) {
        self.call(Method::PUT, p, None).await
    }
    async fn text(&self, p: &str) -> String {
        self.raw(Method::GET, p, None).await.1
    }
}

async fn seed(es: &Es) {
    let (st, v) = es
        .put("/books", json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"title": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"}}}}))
        .await;
    assert_eq!(st, 200, "{v}");
    for (i, (t, tag)) in [("the quick brown fox", "apple"), ("lazy dog", "apricot"), ("quick quick cat", "banana")].iter().enumerate() {
        let (st, v) = es.put(&format!("/books/_doc/{}?refresh=true", i + 1), json!({"title": t, "tag": tag, "n": i})).await;
        assert!(st == 201 || st == 200, "{v}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn nodes_stats_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.get("/_nodes/stats").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_nodes"], json!({"total": 1, "successful": 1, "failed": 0}));
    let n = &v["nodes"][NODE_ID];
    for s in ["indices", "os", "process", "jvm", "thread_pool", "fs", "transport", "http", "breakers", "script", "discovery", "ingest", "adaptive_selection", "script_cache", "indexing_pressure", "repositories", "allocations"] {
        assert!(n.get(s).is_some(), "missing section {s}");
    }
    assert_eq!(n["indices"]["docs"]["count"], 3);
    assert_eq!(n["indices"]["shard_stats"]["total_count"], 1);
    assert!(n["indices"]["mappings"]["total_count"].as_u64().unwrap() >= 3);
    for k in ["warmer", "query_cache", "fielddata", "completion", "segments", "translog", "request_cache", "recovery", "bulk", "dense_vector", "sparse_vector"] {
        assert!(n["indices"][k].is_object(), "indices.{k}");
    }
    assert!(n["jvm"]["mem"]["heap_max_in_bytes"].as_u64().unwrap() > 0);
    assert!(n["jvm"]["mem"]["heap_used_in_bytes"].is_u64());
    // Native compatibility telemetry must identify its measurement and must
    // not invent JVM generations or collectors. The direct pool reports the
    // in-flight request buffers the breaker accounts (ES's network buffers).
    assert_eq!(n["jvm"]["runtime"], "native");
    assert_eq!(n["jvm"]["jvm_measurements_available"], false);
    assert_eq!(n["jvm"]["gc"]["collectors"], json!({}));
    let pools = n["jvm"]["buffer_pools"].as_object().unwrap();
    assert_eq!(pools.keys().collect::<Vec<_>>(), vec!["direct"]);
    assert_eq!(pools["direct"]["measurement"], "in_flight_request_buffers");
    for k in ["count", "used_in_bytes", "total_capacity_in_bytes"] {
        assert!(pools["direct"][k].is_u64(), "buffer_pools.direct.{k}");
    }
    assert_eq!(n["jvm"]["mem"]["pools"], json!({}));
    assert!(n["jvm"]["mem"].get("non_heap_used_in_bytes").is_none());
    assert_eq!(n["jvm"]["mem"]["heap_capacity_measurement"], "configured_compatibility_budget");
    assert_eq!(n["process"]["mem"]["measurement"], "native_process_resident_memory");
    assert!(n["process"]["mem"]["resident_in_bytes"].is_u64());
    assert!(n["process"]["mem"]["native_budget_in_bytes"].as_u64().unwrap() > 0);
    if let Some(anonymous) = n["process"]["mem"]["anonymous_resident_in_bytes"].as_u64() {
        assert_eq!(n["jvm"]["mem"]["heap_measurement"], "native_anonymous_resident");
        assert_eq!(n["jvm"]["mem"]["heap_used_in_bytes"], anonymous);
    } else {
        assert_eq!(n["jvm"]["mem"]["heap_measurement"], "native_accounted_allocations");
    }
    assert!(n["os"]["mem"]["total_in_bytes"].as_u64().unwrap() > 0);
    assert!(n["os"]["cpu"]["load_average"]["1m"].is_number());
    assert!(n["process"]["max_file_descriptors"].is_u64());
    assert!(n["fs"]["total"]["total_in_bytes"].as_u64().unwrap() > 0);
    assert!(n["fs"]["data"][0]["path"].is_string());
    for p in ["write", "search", "management", "flush", "refresh", "force_merge", "get"] {
        let tp = &n["thread_pool"][p];
        for k in ["threads", "queue", "active", "rejected", "largest", "completed"] {
            assert!(tp[k].is_u64(), "thread_pool.{p}.{k}");
        }
    }
    assert!(n["thread_pool"]["write"]["completed"].as_u64().unwrap() >= 3);
    for b in ["parent", "fielddata", "inflight_requests", "request"] {
        assert!(n["breakers"][b]["limit_size_in_bytes"].as_u64().unwrap() > 0, "breaker {b}");
    }
    assert!(n["http"]["total_opened"].as_u64().unwrap() >= 1);
    assert!(n["http"]["clients"][0]["last_uri"].is_string());
    assert!(n["indexing_pressure"]["memory"]["current"]["all_in_bytes"].is_u64());
    assert!(n["allocations"]["current_disk_usage_in_bytes"].is_u64());

    // metric / index_metric filtering
    let (_, v) = es.get("/_nodes/stats/indices,jvm").await;
    let n = &v["nodes"][NODE_ID];
    assert!(n["indices"].is_object() && n["jvm"].is_object() && n.get("os").is_none());
    let (_, v) = es.get("/_nodes/_local/stats/indices/docs,merge").await;
    let n = &v["nodes"][NODE_ID]["indices"];
    assert!(n["docs"].is_object() && n["merges"].is_object() && n.get("store").is_none());
    let (_, v) = es.get("/_nodes/stats/indices?level=indices").await;
    assert_eq!(v["nodes"][NODE_ID]["indices"]["indices"]["books"]["docs"]["count"], 3);
    let (_, v) = es.get("/_nodes/stats/indices?level=shards").await;
    assert!(v["nodes"][NODE_ID]["indices"]["shards"]["books"][0]["0"]["docs"].is_object());
    assert!(v["nodes"][NODE_ID]["indices"]["shards"]["books"][0]["0"].get("mappings").is_none());
    let (_, v) = es.get("/_nodes/stats/indices/segments?include_segment_file_sizes=true").await;
    assert!(v["nodes"][NODE_ID]["indices"]["segments"]["file_sizes"].is_object());
    let (_, v) = es.get("/_nodes/stats?human=true").await;
    assert!(v["nodes"][NODE_ID]["jvm"]["mem"]["heap_max"].is_string(), "{}", v["nodes"][NODE_ID]["jvm"]["mem"]);
    // unknown metric
    let (st, v) = es.get("/_nodes/stats/transprot").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "illegal_argument_exception");
    assert_eq!(v["error"]["reason"], "request [/_nodes/stats/transprot] contains unrecognized metric: [transprot] -> did you mean [transport]?");
    // node selection
    let (_, v) = es.get("/_nodes/nope/stats").await;
    assert_eq!(v["_nodes"]["total"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cluster_stats_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (_, v) = es.get("/_cluster/stats").await;
    assert_eq!(v["indices"]["mappings"]["field_types"], json!([]));
    seed(&es).await;
    let (st, v) = es.get("/_cluster/stats").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["status"], "green");
    assert!(v["cluster_uuid"].is_string() && v["timestamp"].is_number());
    assert_eq!(v["indices"]["count"], 1);
    assert_eq!(v["indices"]["docs"]["count"], 3);
    assert_eq!(v["indices"]["shards"]["total"], 1);
    for k in ["store", "fielddata", "query_cache", "completion", "segments", "analysis", "versions", "dense_vector", "sparse_vector", "search"] {
        assert!(!v["indices"][k].is_null(), "indices.{k}");
    }
    let names: Vec<&str> = v["indices"]["mappings"]["field_types"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["keyword", "long", "text"]);
    assert_eq!(v["indices"]["mappings"]["field_types"][0]["index_count"], 1);
    assert_eq!(v["nodes"]["count"]["total"], 1);
    assert_eq!(v["nodes"]["versions"][0], "9.5.5");
    for k in ["os", "process", "jvm", "fs", "plugins", "network_types", "discovery_types", "packaging_types", "ingest", "indexing_pressure"] {
        assert!(!v["nodes"][k].is_null(), "nodes.{k}");
    }
    assert!(v["nodes"]["os"]["mem"]["total_in_bytes"].as_u64().unwrap() > 0);
    assert!(v["nodes"]["jvm"]["mem"]["heap_max_in_bytes"].as_u64().unwrap() > 0);
    assert!(v["ccs"].is_object());
    let (st, v) = es.get(&format!("/_cluster/stats/nodes/{NODE_ID}")).await;
    assert_eq!((st, v["_nodes"]["total"].clone()), (200, json!(1)));
}

#[tokio::test(flavor = "multi_thread")]
async fn cat_thread_pool_and_extras() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let t = es.text("/_cat/thread_pool?v").await;
    let lines: Vec<&str> = t.lines().collect();
    assert_eq!(lines[0].split_whitespace().collect::<Vec<_>>(), vec!["node_name", "name", "active", "queue", "rejected"]);
    assert!(lines.iter().any(|l| l.split_whitespace().nth(1) == Some("write")));
    let t = es.text("/_cat/thread_pool/write,search?h=name,active").await;
    let names: Vec<&str> = t.lines().map(|l| l.split_whitespace().next().unwrap()).collect();
    assert_eq!(names, vec!["search", "write"]);
    let (_, v) = es.get("/_cat/thread_pool/write?format=json&h=name,type,queue_size").await;
    assert_eq!(v[0]["name"], "write");
    assert_eq!(v[0]["type"], "fixed");
    // recovery
    let t = es.text("/_cat/recovery?h=i,s,ty,st,b,top").await;
    let row: Vec<&str> = t.lines().next().unwrap().split_whitespace().collect();
    assert_eq!(row, vec!["books", "0", "empty_store", "done", "0b", "100.0%"]);
    // tasks lists itself
    let t = es.text("/_cat/tasks").await;
    assert!(t.contains("cluster:monitor/tasks/lists"), "{t}");
    // empty tables
    assert_eq!(es.text("/_cat/repositories").await, "");
    assert_eq!(es.text("/_cat/pending_tasks").await, "");
    assert_eq!(es.text("/_cat/transforms").await, "");
    assert_eq!(es.text("/_cat/ml/anomaly_detectors").await, "");
    assert_eq!(es.text("/_cat/ml/data_frame/analytics").await, "");
    let (st, _) = es.get("/_cat/ml/datafeeds/nope").await;
    assert_eq!(st, 404);
    let t = es.text("/_cat/fielddata?help").await;
    assert!(t.starts_with("id "), "{t}");
    // ES9 exposes circuit breakers through cat.
    let (st, v) = es.get("/_cat/circuit_breaker?format=json&h=breaker").await;
    assert_eq!(st, 200, "{v}");
    assert!(v.as_array().unwrap().iter().any(|row| row["breaker"] == "parent"));
    es.put("/_component_template/ct1", json!({"template": {"settings": {"number_of_shards": 1}}})).await;
    let (_, v) = es.get("/_cat/component_templates/ct1?format=json").await;
    assert_eq!(v.as_array().unwrap().len(), 1, "{v}");
    assert_eq!(v[0]["name"], "ct1");
    assert_eq!(v[0]["settings_count"], "1");
}

#[tokio::test(flavor = "multi_thread")]
async fn deprecations_health_and_admin() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.get("/_migration/deprecations").await;
    assert_eq!(st, 200);
    assert_eq!(v, json!({"cluster_settings": [], "node_settings": [], "ml_settings": [], "index_settings": {}, "data_streams": {}, "templates": {}, "ilm_policies": {}}));
    let (st, _) = es.get("/books/_migration/deprecations").await;
    assert_eq!(st, 200);

    let (st, v) = es.get("/_health_report").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["status"], "green");
    assert_eq!(v["indicators"]["master_is_stable"]["symptom"], "The cluster has a stable master node");
    assert_eq!(v["indicators"]["master_is_stable"]["details"]["current_master"]["node_id"], NODE_ID);
    assert_eq!(v["indicators"]["repository_integrity"]["symptom"], "No snapshot repositories configured.");
    assert_eq!(v["indicators"]["shards_availability"]["symptom"], "This cluster has all shards available.");
    for i in ["shards_capacity", "disk", "ilm", "slm", "data_stream_lifecycle"] {
        assert_eq!(v["indicators"][i]["status"], "green", "{i}");
    }
    let (_, v) = es.get("/_health_report/disk?verbose=false").await;
    assert!(v.get("status").is_none());
    assert!(v["indicators"]["disk"].get("details").is_none());
    assert_eq!(v["indicators"].as_object().unwrap().len(), 1);
    let (st, _) = es.get("/_health_report/nope").await;
    assert_eq!(st, 404);

    // allocation explain
    let (st, v) = es.get("/_cluster/allocation/explain").await;
    assert_eq!((st, v["error"]["type"].clone()), (400, json!("illegal_argument_exception")));
    let (st, v) = es.post("/_cluster/allocation/explain", json!({"index": "books", "shard": 0, "primary": true})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["current_state"], "started");
    assert_eq!(v["current_node"]["id"], NODE_ID);
    es.put("/rep", json!({"settings": {"number_of_replicas": 1}})).await;
    let (_, v) = es.get("/_cluster/allocation/explain?include_disk_info=true").await;
    assert_eq!(v["current_state"], "unassigned");
    assert_eq!(v["unassigned_info"]["reason"], "INDEX_CREATED");
    assert_eq!(v["can_allocate"], "no");
    assert!(v["cluster_info"].is_object());

    // reroute
    let (st, v) = es.post("/_cluster/reroute?metric=none", json!({})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (_, v) = es.post("/_cluster/reroute?explain=true&dry_run=true&metric=none", json!({"commands": [{"cancel": {"index": "books", "shard": 9, "node": NODE_ID}}]})).await;
    assert_eq!(v["explanations"][0]["command"], "cancel");
    assert_eq!(v["explanations"][0]["parameters"]["allow_primary"], false);
    assert_eq!(v["explanations"][0]["decisions"][0]["decider"], "cancel_allocation_command");
    assert_eq!(v["explanations"][0]["decisions"][0]["decision"], "NO");
    let (_, v) = es.post("/_cluster/reroute?metric=metadata", json!({})).await;
    assert_eq!(v, json!({"acknowledged": true}));
    let (_, v) = es.get("/_cluster/pending_tasks").await;
    assert_eq!(v, json!({"tasks": []}));

    // recovery / shard stores / search shards / resolve
    let (_, v) = es.get("/books/_recovery?human=true&detailed=true").await;
    assert_eq!(v["books"]["shards"][0]["type"], "EMPTY_STORE");
    assert_eq!(v["books"]["shards"][0]["stage"], "DONE");
    assert_eq!(v["books"]["shards"][0]["index"]["files"]["details"], json!([]));
    assert!(v["books"]["shards"][0]["start_time"].as_str().unwrap().starts_with('2'));
    let (_, v) = es.get("/_recovery?active_only=true").await;
    assert_eq!(v, json!({}));
    let (_, v) = es.get("/_shard_stores?status=green").await;
    assert_eq!(v["indices"]["books"]["shards"]["0"]["stores"][0]["allocation"], "primary");
    let (_, v) = es.get("/_shard_stores").await;
    assert!(v["indices"].get("books").is_none() && v["indices"]["rep"].is_object());
    es.post("/_aliases", json!({"actions": [{"add": {"index": "books", "alias": "fb", "filter": {"term": {"tag": "apple"}}}}]})).await;
    let (_, v) = es.get("/fb/_search_shards").await;
    assert_eq!(v["shards"][0][0]["index"], "books");
    assert_eq!(v["indices"]["books"]["aliases"], json!(["fb"]));
    assert_eq!(v["indices"]["books"]["filter"]["term"]["tag"]["value"], "apple");
    assert!(v["nodes"][NODE_ID]["name"].is_string());
    let (_, v) = es.get("/_resolve/index/*").await;
    assert_eq!(v["indices"][0]["name"], "books");
    assert_eq!(v["indices"][0]["aliases"], json!(["fb"]));
    assert_eq!(v["indices"][0]["attributes"], json!(["open"]));
    assert_eq!(v["aliases"][0], json!({"name": "fb", "indices": ["books"]}));

    // usage counters + hot threads
    let (_, v) = es.get("/_nodes/usage").await;
    // ES's `RestHealthAction` (`GET /_health_report`) is named `health_action`.
    assert!(v["nodes"][NODE_ID]["rest_actions"]["health_action"].as_u64().unwrap() >= 1, "{v}");
    let t = es.text("/_nodes/hot_threads").await;
    assert!(t.contains("Hot threads at"), "{t}");
    let (st, v) = es.get("/_nodes/hot_threads?type=gpu").await;
    assert_eq!((st, v["error"]["reason"].clone()), (400, json!("type not supported [gpu]")));

    // disk usage / field usage / reload analyzers
    let (st, _) = es.post0("/books/_disk_usage").await;
    assert_eq!(st, 400);
    let (st, v) = es.post0("/books/_disk_usage?run_expensive_tasks=true").await;
    assert_eq!(st, 200, "{v}");
    assert!(v["books"]["store_size_in_bytes"].as_u64().unwrap() > 0);
    assert!(v["books"]["fields"]["tag"]["inverted_index"]["total_in_bytes"].as_u64().unwrap() > 0);
    let (_, v) = es.get("/books/_field_usage_stats").await;
    assert_eq!(v["books"]["shards"][0]["routing"]["primary"], true);
    let (_, v) = es.post0("/books/_reload_search_analyzers").await;
    assert_eq!(v["reload_details"][0]["index"], "books");
}

#[tokio::test(flavor = "multi_thread")]
async fn index_blocks() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.put0("/books/_block/write").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v, json!({"acknowledged": true, "shards_acknowledged": true, "indices": [{"name": "books", "blocked": true}]}));
    let (st, v) = es.put("/books/_doc/9", json!({"title": "x"})).await;
    assert_eq!(st, 403);
    assert_eq!(v["error"]["reason"], "index [books] blocked by: [FORBIDDEN/8/index write (api)];");
    let (st, _) = es.get("/books/_search").await;
    assert_eq!(st, 200);
    // ES9 removes the block through DELETE.
    let (st, v) = es.call(Method::DELETE, "/books/_block/write", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v, json!({"acknowledged": true, "indices": [{"name": "books", "unblocked": true}]}));
    let (st, _) = es.put("/books/_settings", json!({"index.blocks.write": false})).await;
    assert_eq!(st, 200);
    let (st, _) = es.put("/books/_doc/9", json!({"title": "x"})).await;
    assert_eq!(st, 201);
    es.put0("/books/_block/read").await;
    let (st, v) = es.get("/books/_search").await;
    assert_eq!(st, 403);
    assert_eq!(v["error"]["reason"], "index [books] blocked by: [FORBIDDEN/7/index read (api)];");
    let (st, _) = es.get("/books/_doc/1").await;
    assert_eq!(st, 403);
    let (st, v) = es.put0("/books/_block/bogus").await;
    assert_eq!((st, v["error"]["type"].clone()), (400, json!("illegal_argument_exception")));
    let (st, _) = es.put("/books/_settings", json!({"index.blocks.read": false})).await;
    assert_eq!(st, 200);
    es.put0("/books/_block/read_only").await;
    let (st, v) = es.put("/books/_mapping", json!({"properties": {"z": {"type": "keyword"}}})).await;
    assert_eq!(st, 403, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_query_api() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (_, v) = es.get("/books/_validate/query?q=title:quick").await;
    assert_eq!(v["valid"], true);
    assert_eq!(v["_shards"]["failed"], 0);
    let (_, v) = es.get("/_validate/query?explain=true").await;
    assert_eq!(v["valid"], true);
    assert_eq!(v["explanations"][0]["index"], "books");
    assert_eq!(v["explanations"][0]["explanation"], "*:*");
    let (_, v) = es.post("/books/_validate/query?explain=true", json!({"query": {"bool": {"must": [{"term": {"tag": "apple"}}], "filter": [{"range": {"n": {"gte": 1}}}]}}})).await;
    assert_eq!(v["explanations"][0]["explanation"], "+tag:apple #n:[1 TO 9223372036854775807]");
    let (_, v) = es.post("/books/_validate/query", json!({"query": {"invalid_query": {}}})).await;
    assert_eq!(v, json!({"valid": false}));
    let (_, v) = es.post("/books/_validate/query?explain=true", json!({"query": {"boool": {}}})).await;
    assert_eq!(v["valid"], false);
    let e = v["error"].as_str().unwrap();
    assert!(e.contains("unknown query [boool] did you mean [bool]?") && !e.ends_with('?'), "{e}");
    // Exactly as ES 8.19 reports it.
    assert_eq!(e, "org.elasticsearch.common.ParsingException: unknown query [boool] did you mean [bool]?; org.elasticsearch.xcontent.NamedObjectNotFoundException: [1:19] unknown field [boool]");
    let (_, v) = es.post("/books/_validate/query?explain=true", json!({"match_all": {}})).await;
    assert_eq!(v["error"], "org.elasticsearch.common.ParsingException: request does not support [match_all]");
    let (_, v) = es.get("/books/_validate/query?q=n:foo&lenient=true").await;
    assert_eq!(v["valid"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn terms_enum_api() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/books/_terms_enum", json!({"field": "tag", "string": "ap"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v, json!({"_shards": {"total": 1, "successful": 1, "failed": 0}, "terms": ["apple", "apricot"], "complete": true}));
    let (_, v) = es.post("/books/_terms_enum", json!({"field": "tag", "string": "AP", "case_insensitive": true, "size": 1})).await;
    assert_eq!(v["terms"], json!(["apple"]));
    let (_, v) = es.post("/books/_terms_enum", json!({"field": "tag", "search_after": "apple"})).await;
    assert_eq!(v["terms"], json!(["apricot", "banana"]));
    let (_, v) = es.post("/books/_terms_enum", json!({"field": "tag", "index_filter": {"term": {"tag": "nothing"}}})).await;
    assert_eq!(v["terms"], json!([]));
    let (_, v) = es.post("/books/_terms_enum", json!({"field": "title"})).await;
    assert_eq!(v["terms"], json!([]));
    let (st, _) = es.post("/books/_terms_enum", json!({"string": "a"})).await;
    assert_eq!(st, 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_api() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/books/_explain/1", json!({"query": {"match_all": {}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_index"], "books");
    assert_eq!(v["_id"], "1");
    assert_eq!(v["matched"], true);
    assert_eq!(v["explanation"]["value"], 1.0);
    let (_, v) = es.post("/books/_explain/1", json!({"query": {"match": {"title": "quick"}}})).await;
    assert_eq!(v["matched"], true);
    assert!(v["explanation"]["value"].as_f64().unwrap() > 0.0);
    assert!(v["explanation"]["description"].as_str().unwrap().contains("title:quick"), "{v}");
    assert!(v["explanation"]["details"].is_array());
    let (_, v) = es.post("/books/_explain/2", json!({"query": {"match": {"title": "quick"}}})).await;
    assert_eq!(v["matched"], false);
    let (_, v) = es.get("/books/_explain/3?q=quick&df=title").await;
    assert_eq!(v["matched"], true);
    let (st, v) = es.post("/books/_explain/404", json!({"query": {"match_all": {}}})).await;
    assert_eq!((st, v), (404, json!({"_index": "books", "_id": "404", "matched": false})));
    let (st, _) = es.post("/books/_explain/1", json!({"match_all": {}})).await;
    assert_eq!(st, 400);
    let (_, v) = es.post("/books/_explain/1?_source_includes=tag", json!({"query": {"match_all": {}}})).await;
    assert_eq!(v["get"]["_source"], json!({"tag": "apple"}));
}

#[tokio::test(flavor = "multi_thread")]
async fn search_profile() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es
        .post("/books/_search", json!({"profile": true, "query": {"bool": {"must": [{"match": {"title": "quick"}}], "filter": [{"term": {"tag": "apple"}}]}}, "aggs": {"t": {"terms": {"field": "tag"}}}}))
        .await;
    assert_eq!(st, 200, "{v}");
    let shard = &v["profile"]["shards"][0];
    assert_eq!(shard["id"], format!("[{NODE_ID}][books][0]"));
    assert_eq!(shard["node_id"], NODE_ID);
    assert_eq!(shard["index"], "books");
    assert_eq!(shard["shard_id"], 0);
    let q = &shard["searches"][0]["query"][0];
    assert_eq!(q["type"], "BooleanQuery");
    assert_eq!(q["description"], "+title:quick #tag:apple");
    assert!(q["time_in_nanos"].as_u64().unwrap() > 0);
    assert!(q["breakdown"]["create_weight"].as_u64().unwrap() > 0);
    assert_eq!(q["children"][0]["type"], "TermQuery");
    assert_eq!(q["children"][0]["description"], "title:quick");
    assert!(shard["searches"][0]["rewrite_time"].as_u64().unwrap() > 0);
    assert_eq!(shard["searches"][0]["collector"][0]["name"], "QueryPhaseCollector");
    assert_eq!(shard["aggregations"][0]["description"], "t");
    assert_eq!(shard["aggregations"][0]["type"], "GlobalOrdinalsStringTermsAggregator");
    let f = &shard["fetch"];
    assert!(f["time_in_nanos"].as_u64().unwrap() > 0);
    assert_eq!(f["debug"]["stored_fields"], json!(["_id", "_routing", "_source"]));
    let kids: Vec<&str> = f["children"].as_array().unwrap().iter().map(|c| c["type"].as_str().unwrap()).collect();
    assert_eq!(kids, vec!["FetchFieldsPhase", "FetchSourcePhase", "StoredFieldsPhase"]);
    let (_, v) = es.post("/books/_search", json!({"profile": true, "stored_fields": "_none_"})).await;
    assert_eq!(v["profile"]["shards"][0]["fetch"]["debug"]["stored_fields"], json!([]));
    assert!(v["profile"]["shards"][0]["fetch"].get("children").is_none());
    let (_, v) = es.post("/books/_search?search_type=dfs_query_then_fetch", json!({"profile": true, "query": {"term": {"tag": "apple"}}})).await;
    assert_eq!(v["profile"]["shards"][0]["dfs"]["statistics"]["type"], "statistics");
}

#[tokio::test(flavor = "multi_thread")]
async fn dense_vector_live_stats_follow_update_delete_flush_and_merge() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (status, body) = es.put("/vector-stats", json!({"settings":{"number_of_replicas":0},
        "mappings":{"properties":{"v":{"type":"dense_vector","dims":3,"similarity":"l2_norm"}}}})).await;
    assert_eq!(status,200,"{body}");
    for id in 0..2 {
        let (status, body) = es.put(&format!("/vector-stats/_doc/{id}?refresh=true"), json!({"v":[id as f64,1.0,2.0]})).await;
        assert_eq!(status,201,"{body}");
    }
    for stage in 0..5 {
        match stage {
            1 => { let (status,body)=es.post0("/vector-stats/_flush").await; assert_eq!(status,200,"{body}"); },
            2 => { let (status,body)=es.put("/vector-stats/_doc/0?refresh=true",json!({"v":[2.0,3.0,4.0]})).await; assert_eq!(status,200,"{body}"); },
            3 => { let (status,body)=es.call(Method::DELETE,"/vector-stats/_doc/1?refresh=true",None).await; assert_eq!(status,200,"{body}"); },
            4 => { let (status,body)=es.post0("/vector-stats/_forcemerge?max_num_segments=1").await; assert_eq!(status,200,"{body}"); },
            _ => {}
        }
        let expected=if stage<3 {2} else {1};
        let (status,body)=es.get("/_nodes/stats/indices/dense_vector").await;
        assert_eq!(status,200,"{body}");
        assert_eq!(body["nodes"][NODE_ID]["indices"]["dense_vector"]["value_count"],expected,"stage {stage}: {body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn disk_usage_counts_persisted_ignored_source() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start_licensed(tmp.path(), Some("trial")).await;
    let (status, body) = es.put("/synthetic", json!({
        "settings": {"index.mapping.source.mode": "synthetic"},
        "mappings": {"properties": {"ignore_me": {"type": "object", "enabled": false}}}
    })).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = es.put("/synthetic/_doc/1?refresh=true", json!({"foo":"bar", "ignore_me":{"foo":"bar"}})).await;
    assert_eq!(status, 201, "{body}");
    for flush in [false, true] {
        let (status, body) = es.post0(&format!("/synthetic/_disk_usage?run_expensive_tasks=true&flush={flush}")).await;
        assert_eq!(status, 200, "{body}");
        assert!(body["synthetic"]["fields"]["_ignored_source"]["stored_fields_in_bytes"].as_u64().unwrap_or(0) > 0, "{body}");
        assert!(body["synthetic"]["fields"].get("_source").is_none(), "{body}");
        assert!(body["synthetic"]["fields"].get("_recovery_source").is_none(), "{body}");
    }
    let (status, body) = es.put("/ordinary", json!({})).await;
    assert_eq!(status, 200, "{body}");
    es.put("/ordinary/_doc/1?refresh=true", json!({"foo":"bar"})).await;
    let (_, body) = es.post0("/ordinary/_disk_usage?run_expensive_tasks=true").await;
    assert!(body["ordinary"]["fields"].get("_ignored_source").is_none(), "{body}");
}

/// Counter monitoring must remain available while maintenance owns the core.
/// Hold the lock explicitly so this cannot pass just because a merge finished
/// before the first HTTP request reached the server.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn node_counter_stats_do_not_wait_for_maintenance() {
    use std::time::Duration;
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server_node = node.clone();
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(server_node, listener).await; });
    let es = Es { base, c: Client::new() };
    seed(&es).await;
    let svc = node.all_indices().into_iter().find(|s| s.name == "books").unwrap();
    let sh = svc.local_shards()[0].clone();
    sh.shard.counters.merge_total.store(7, std::sync::atomic::Ordering::Relaxed);
    let (_, full) = es.get("/_nodes/stats/indices").await;
    let expected = full["nodes"][NODE_ID]["indices"]["merges"].clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _guard = sh.shard.core.write();
        ready_tx.send(()).unwrap();
        // Bounded even if the test panics.
        let _ = release_rx.recv_timeout(Duration::from_secs(20));
    });
    ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let mut results = Vec::new();
    for path in [
        "/_nodes/stats/indices/merge,flush",
        "/_nodes/stats/indices/merge,flush?level=indices&include_segment_file_sizes=true",
        "/_nodes/stats?metric=indices&index_metric=merge,flush",
    ] {
        results.push((path, es.c.get(format!("{}{path}", es.base)).timeout(Duration::from_secs(2)).send().await));
    }
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    for (path, result) in results {
        let response = result.unwrap_or_else(|e| panic!("counter stats blocked for {path}: {e}"));
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        let stats = &body["nodes"][NODE_ID]["indices"];
        assert_eq!(stats["merges"], expected, "{path}: {body}");
        assert!(stats.get("translog").is_none(), "{body}");
        if path.contains("level=indices") {
            assert_eq!(stats["indices"]["books"]["merges"], expected);
        }
    }
    // Mixed/default and shard-level requests must still report actual data.
    for path in ["/_nodes/stats/indices", "/_nodes/stats/indices/docs,merge,translog"] {
        let (status, body) = es.get(path).await;
        assert_eq!(status, 200);
        assert_eq!(body["nodes"][NODE_ID]["indices"]["docs"]["count"], 3);
        assert!(body["nodes"][NODE_ID]["indices"]["translog"]["size_in_bytes"].as_u64().unwrap() > 0);
    }
    let (status, body) = es.get("/_nodes/stats/indices/merge?level=shards").await;
    assert_eq!(status, 200);
    assert_eq!(body["nodes"][NODE_ID]["indices"]["shards"]["books"][0]["0"]["commit"]["num_docs"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn selected_stats_skip_translog_during_force_merge() {
    use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
    use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
    use std::time::Duration;
    #[derive(Debug)]
    struct PausedDirectory {
        inner: RamDirectory, pause: AtomicBool,
        entered: std::sync::mpsc::Sender<()>,
        resume: Mutex<std::sync::mpsc::Receiver<()>>,
    }
    impl Directory for PausedDirectory {
        fn open_read(&self, n: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(n) }
        fn write(&self, n: &str, data: &[u8]) -> rustic_core::error::Result<()> {
            if self.pause.swap(false, Ordering::AcqRel) {
                self.entered.send(()).unwrap();
                self.resume.lock().unwrap().recv_timeout(Duration::from_secs(30)).unwrap();
            }
            self.inner.write(n, data)
        }
        fn exists(&self, n: &str) -> bool { self.inner.exists(n) }
        fn delete(&self, n: &str) -> rustic_core::error::Result<()> { self.inner.delete(n) }
        fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
        fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
    }
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("merge-stats", &json!({"settings":{"number_of_shards":2,"number_of_replicas":0}})).unwrap();
    let sh = svc.local_shards()[0].clone();
    let (entered, wait) = std::sync::mpsc::channel();
    let (resume, release) = std::sync::mpsc::channel();
    let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(), pause: AtomicBool::new(false), entered, resume: Mutex::new(release) });
    let schema = sh.shard.core.read().schema.clone();
    let mut core = rustic_core::Index::create_in(dir.clone(), "merge-stats", schema, Default::default()).unwrap();
    core.set_nrt(true);
    for _ in 0..2 { core.add_document(&rustic_core::types::Document::new()).unwrap(); core.seal().unwrap(); core.refresh().unwrap(); }
    assert_eq!(core.num_segments(), 2);
    // Searches and doc stats read the shard's view slot.
    core.adopt_view_slot(sh.shard.view_slot().clone()).unwrap();
    *sh.shard.core.write() = core;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let es = Es { base: format!("http://{}", listener.local_addr().unwrap()), c: Client::new() };
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(node, listener).await; });
    // Finish HTTP startup before measuring requests under the held merge.
    assert_eq!(es.get("/merge-stats/_stats/docs").await.0, 200);
    let (_, before) = es.get("/merge-stats/_stats/translog").await;
    dir.pause.store(true, Ordering::Release);
    let merging = sh.clone();
    let merge = std::thread::spawn(move || merging.shard.force_merge(1, false));
    wait.recv_timeout(Duration::from_secs(5)).unwrap();
    let core_readable = sh.shard.core.try_read().is_some();
    let mut results = Vec::new();
    for path in [
        "/merge-stats/_stats/docs,merge",
        "/merge-stats/_stats/docs,merge?level=shards",
        "/_nodes/stats/indices/docs,merge",
        "/_nodes/stats/indices/docs,merge?level=shards",
    ] {
        results.push((path, es.c.get(format!("{}{path}", es.base)).timeout(Duration::from_secs(2)).send().await));
    }
    // Requested translog figures are real, not the placeholder zeros used for
    // discarded sections: the merge published them when it took the write
    // state, and no write can change them until its own commit.
    let during = es.c.get(format!("{}/merge-stats/_stats/translog", es.base))
        .timeout(Duration::from_secs(2)).send().await;
    resume.send(()).unwrap();
    merge.join().unwrap().unwrap();
    assert!(core_readable, "fixture must pause merge output outside the core write lock");
    let during: Value = during.expect("requested translog stats blocked behind the force merge").json().await.unwrap();
    for key in ["size_in_bytes", "uncommitted_operations"] {
        assert_eq!(during["_all"]["primaries"]["translog"][key], before["_all"]["primaries"]["translog"][key], "{key}: {during} vs {before}");
    }
    for (path, response) in results {
        let response = response.unwrap_or_else(|e| panic!("unrequested translog blocked {path}: {e}"));
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        let stats = if path.starts_with("/_nodes") { &body["nodes"][NODE_ID]["indices"] } else { &body["_all"]["primaries"] };
        assert_eq!(stats["docs"]["count"], 2, "{path}: {body}");
        assert_eq!(stats["merges"]["current"], 1, "{path}: {body}");
        assert!(stats.get("translog").is_none(), "{body}");
    }
    let (status, body) = es.get("/merge-stats/_stats/translog").await;
    assert_eq!(status, 200, "{body}");
    assert!(body["_all"]["primaries"]["translog"]["size_in_bytes"].as_u64().is_some());
}
