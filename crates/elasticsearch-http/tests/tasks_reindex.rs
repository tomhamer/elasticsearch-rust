//! HTTP tests for `_reindex`, `_update_by_query`, `_delete_by_query`,
//! `_tasks`, `_field_caps`, `_async_search`, data streams and rollover
//! (exact ES response shapes).

use elasticsearch_engine::node::{Node, NODE_ID};
use reqwest::{Client, Method};
use serde_json::{json, Value};

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
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
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
    /// POST without a body (ES rejects a body on `_tasks/_cancel`).
    async fn post0(&self, p: &str) -> (u16, Value) {
        self.call(Method::POST, p, None).await
    }
    /// PUT without a body (ES rejects a body on `PUT _data_stream/{name}`).
    async fn put0(&self, p: &str) -> (u16, Value) {
        self.call(Method::PUT, p, None).await
    }
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.call(Method::DELETE, p, None).await
    }
    async fn bulk(&self, lines: &[Value]) {
        let mut s = String::new();
        for l in lines {
            s.push_str(&l.to_string());
            s.push('\n');
        }
        let r = self.c.post(format!("{}/_bulk?refresh=true", self.base)).header("content-type", "application/x-ndjson").body(s).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }
}

fn strip_took(mut v: Value) -> Value {
    v.as_object_mut().unwrap().remove("took");
    v
}

async fn seed(es: &Es, n: usize) {
    let (st, v) = es.put("/src", json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}, "tag": {"type": "keyword"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for i in 0..n {
        lines.push(json!({"index": {"_index": "src", "_id": i.to_string()}}));
        lines.push(json!({"n": i, "tag": if i % 2 == 0 { "even" } else { "odd" }, "extra": "x"}));
    }
    es.bulk(&lines).await;
}

#[tokio::test]
async fn reindex_shapes_and_options() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es, 25).await;

    // Plain reindex, small batches.
    let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": "src", "size": 10}, "dest": {"index": "dst"}})).await;
    assert_eq!(st, 200, "{v}");
    assert!(v["took"].is_u64());
    assert_eq!(
        strip_took(v),
        json!({"timed_out": false, "total": 25, "updated": 0, "created": 25, "deleted": 0, "batches": 3, "version_conflicts": 0, "noops": 0,
               "retries": {"bulk": 0, "search": 0}, "throttled_millis": 0, "requests_per_second": -1.0, "throttled_until_millis": 0, "failures": []})
    );
    let (_, c) = es.get("/dst/_count").await;
    assert_eq!(c["count"], 25);
    let (_, d) = es.get("/dst/_doc/3").await;
    assert_eq!(d["_source"], json!({"n": 3, "tag": "odd", "extra": "x"}));

    // Second run overwrites (internal versioning) → updated.
    let (_, v) = es.post("/_reindex", json!({"source": {"index": "src"}, "dest": {"index": "dst"}})).await;
    assert_eq!((v["updated"].clone(), v["created"].clone()), (json!(25), json!(0)));

    // Query + max_docs + _source filtering.
    let (_, v) = es
        .post("/_reindex?refresh=true", json!({"max_docs": 3, "source": {"index": "src", "query": {"term": {"tag": "even"}}, "_source": ["n"]}, "dest": {"index": "evens"}}))
        .await;
    assert_eq!((v["total"].clone(), v["created"].clone()), (json!(3), json!(3)), "{v}");
    let (_, s) = es.post("/evens/_search", json!({"sort": ["n"]})).await;
    assert_eq!(s["hits"]["hits"][0]["_source"], json!({"n": 0}));

    // op_type=create: conflicts abort with 409 + failures; proceed counts them.
    let (st, v) = es.post("/_reindex", json!({"source": {"index": "src"}, "dest": {"index": "dst", "op_type": "create"}})).await;
    assert_eq!(st, 409, "{v}");
    assert_eq!(v["version_conflicts"], 25);
    let f = &v["failures"][0];
    assert_eq!(f["index"], "dst");
    assert_eq!(f["status"], 409);
    assert_eq!(f["cause"]["type"], "version_conflict_engine_exception");
    assert!(f["cause"]["reason"].as_str().unwrap().contains("document already exists"));
    // `conflicts` is a body field for _reindex (ES rejects it as a URL parameter).
    let (st, v) = es.post("/_reindex?conflicts=proceed", json!({"source": {"index": "src"}, "dest": {"index": "dst", "op_type": "create"}})).await;
    assert_eq!((st, v["error"]["reason"].clone()), (400, json!("request [/_reindex] contains unrecognized parameter: [conflicts]")));
    let (st, v) = es.post("/_reindex", json!({"conflicts": "proceed", "source": {"index": "src"}, "dest": {"index": "dst", "op_type": "create"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["version_conflicts"].clone(), v["created"].clone(), v["failures"].clone()), (json!(25), json!(0), json!([])));

    // version_type=external keeps source versions.
    let (_, v) = es.post("/_reindex", json!({"source": {"index": "src"}, "dest": {"index": "ext", "version_type": "external"}, "conflicts": "proceed"})).await;
    assert_eq!(v["created"], 25);
    let (_, d) = es.get("/ext/_doc/1").await;
    assert_eq!(d["_version"], 1);

    // slices accepted.
    let (st, _) = es.post("/_reindex?slices=auto", json!({"source": {"index": "src"}, "dest": {"index": "dst"}})).await;
    assert_eq!(st, 200);

    // Validation errors.
    let (st, v) = es.post("/_reindex", json!({"source": {"index": "src"}, "dest": {"index": "src"}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "action_request_validation_exception");
    assert_eq!(v["error"]["reason"], "Validation Failed: 1: reindex cannot write into an index its reading from [src];");
    let (_, v) = es.post("/_reindex", json!({"source": {"index": "src"}})).await;
    assert_eq!(v["error"]["reason"], "Validation Failed: 1: index must be specified;");
    let (st, v) = es.post("/_reindex", json!({"source": {"index": "nope"}, "dest": {"index": "x"}})).await;
    assert_eq!((st, v["error"]["type"].clone()), (404, json!("index_not_found_exception")));
    // Scripts run: `ctx._source` edits, `ctx.op = 'noop'`, `ctx._id` renames.
    let (st, v) = es
        .post(
            "/_reindex?refresh=true",
            json!({"source": {"index": "src"}, "dest": {"index": "scripted"}, "script": {
                "source": "if (ctx._source.n % 5 == 0) { ctx.op = 'noop' } else { ctx._source.n++; ctx._source.remove('extra'); if (ctx._source.n == 2) { ctx._id = 'renamed' } }"
            }}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["total"].clone(), v["created"].clone(), v["noops"].clone()), (json!(25), json!(20), json!(5)), "{v}");
    let (_, d) = es.get("/scripted/_doc/3").await;
    assert_eq!(d["_source"], json!({"n": 4, "tag": "odd"}));
    let (st, _) = es.get("/scripted/_doc/5").await;
    assert_eq!(st, 404);
    let (st, d) = es.get("/scripted/_doc/renamed").await;
    assert_eq!((st, d["_source"]["n"].clone()), (200, json!(2)));
    let (st, v) = es.post("/_reindex", json!({"source": {"index": "src"}, "dest": {"index": "x"}, "script": {"source": "ctx._source.n++", "lang": "groovy"}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "script_lang not supported [groovy]");
    let (_, v) = es.post("/_reindex", json!({"conflicts": "maybe", "source": {"index": "src"}, "dest": {"index": "x"}})).await;
    assert_eq!(v["error"]["caused_by"]["reason"], "conflicts may only be \"proceed\" or \"abort\" but was [maybe]", "{v}");
}

#[tokio::test]
async fn by_query_and_tasks() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es, 10).await;

    let (st, v) = es.post("/src/_update_by_query?refresh=true", json!({"query": {"term": {"tag": "odd"}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(
        strip_took(v),
        json!({"timed_out": false, "total": 5, "updated": 5, "deleted": 0, "batches": 1, "version_conflicts": 0, "noops": 0,
               "retries": {"bulk": 0, "search": 0}, "throttled_millis": 0, "requests_per_second": -1.0, "throttled_until_millis": 0, "failures": []})
    );
    let (_, d) = es.get("/src/_doc/1").await;
    assert_eq!(d["_version"], 2);
    let (_, d) = es.get("/src/_doc/0").await;
    assert_eq!(d["_version"], 1);

    // Scripted update_by_query: edits, `ctx.op = 'noop'` and `ctx.op = 'delete'` are honoured.
    for (id, n) in [("a", 1), ("b", 2), ("c", 3)] {
        let (st, v) = es.put(&format!("/ubq/_doc/{id}?refresh=true"), json!({"n": n})).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    let (st, v) = es
        .post(
            "/ubq/_update_by_query?refresh=true",
            json!({"script": {"source": "if (ctx._source.n == 1) { ctx.op = 'noop' } else if (ctx._source.n == 2) { ctx.op = 'delete' } else { ctx._source.n += params.d }", "lang": "painless", "params": {"d": 10}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["total"].clone(), v["updated"].clone(), v["deleted"].clone(), v["noops"].clone()), (json!(3), json!(1), json!(1), json!(1)), "{v}");
    let (_, d) = es.get("/ubq/_doc/c").await;
    assert_eq!((d["_source"]["n"].clone(), d["_version"].clone()), (json!(13), json!(2)));
    let (_, d) = es.get("/ubq/_doc/a").await;
    assert_eq!(d["_version"], 1);
    let (st, _) = es.get("/ubq/_doc/b").await;
    assert_eq!(st, 404);
    let (st, v) = es.post("/ubq/_update_by_query", json!({"script": "ctx.op = 'bogus'"})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["reason"], "Operation type [bogus] not allowed, only [noop, index, delete] are allowed");

    let (st, v) = es.post("/src/_delete_by_query?refresh=true&scroll_size=2", json!({"query": {"range": {"n": {"lt": 4}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(
        strip_took(v),
        json!({"timed_out": false, "total": 4, "deleted": 4, "batches": 2, "version_conflicts": 0, "noops": 0,
               "retries": {"bulk": 0, "search": 0}, "throttled_millis": 0, "requests_per_second": -1.0, "throttled_until_millis": 0, "failures": []})
    );
    let (_, c) = es.get("/src/_count").await;
    assert_eq!(c["count"], 6);
    let (st, v) = es.post("/src/_delete_by_query", json!({})).await;
    assert_eq!((st, v["error"]["reason"].clone()), (400, json!("Validation Failed: 1: query is missing;")));
    let (_, v) = es.post("/src/_delete_by_query?q=tag:even&refresh=true", json!({})).await;
    assert_eq!(v["deleted"], 3);

    // Async reindex → task id → GET _tasks/{id}.
    let (st, v) = es.post("/_reindex?wait_for_completion=false", json!({"source": {"index": "src"}, "dest": {"index": "copy"}})).await;
    assert_eq!(st, 200);
    let id = v["task"].as_str().unwrap().to_string();
    assert!(id.starts_with(&format!("{NODE_ID}:")), "{id}");
    let (st, t) = es.get(&format!("/_tasks/{id}?wait_for_completion=true&timeout=10s")).await;
    assert_eq!(st, 200, "{t}");
    assert_eq!(t["completed"], true);
    assert_eq!(t["task"]["node"], NODE_ID);
    assert_eq!(t["task"]["action"], "indices:data/write/reindex");
    assert_eq!(t["task"]["type"], "transport");
    assert_eq!(t["task"]["description"], "reindex from [src] to [copy]");
    assert_eq!(t["task"]["cancellable"], true);
    assert_eq!(t["task"]["cancelled"], false);
    assert_eq!(t["task"]["status"]["created"], 3);
    assert_eq!(t["response"]["created"], 3);
    assert_eq!(t["response"]["failures"], json!([]));
    let keys: Vec<&String> = t["task"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["node", "id", "type", "action", "status", "description", "start_time_in_millis", "running_time_in_nanos", "cancellable", "cancelled", "headers"]);

    // Task list / errors / cancel.
    let (st, l) = es.get("/_tasks?actions=*reindex&detailed").await;
    assert_eq!((st, l), (200, json!({"nodes": {}})));
    // ES lists the running list-tasks request itself.
    let (st, l) = es.get("/_tasks?group_by=none").await;
    assert_eq!(st, 200);
    assert!(l["tasks"].as_array().unwrap().iter().all(|t| t["action"].as_str().unwrap().starts_with("cluster:monitor/tasks/lists")), "{l}");
    let (st, e) = es.get("/_tasks/bogus").await;
    assert_eq!((st, e["error"]["reason"].clone()), (400, json!("malformed task id bogus")));
    let (st, e) = es.get(&format!("/_tasks/{NODE_ID}:999")).await;
    assert_eq!(st, 404);
    assert_eq!(e["error"]["type"], "resource_not_found_exception");
    assert_eq!(e["error"]["reason"], format!("task [{NODE_ID}:999] isn't running and hasn't stored its results"));
    let (st, c) = es.post0(&format!("/_tasks/{NODE_ID}:999/_cancel")).await;
    assert_eq!(st, 200);
    assert_eq!(c["node_failures"][0]["caused_by"]["type"], "resource_not_found_exception");
    let (st, c) = es.post0("/_tasks/_cancel?actions=*reindex").await;
    assert_eq!((st, c), (200, json!({"nodes": {}})));
}

#[tokio::test]
async fn field_caps_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.put("/a", json!({"mappings": {"properties": {"title": {"type": "text", "fields": {"raw": {"type": "keyword"}}}, "n": {"type": "long", "meta": {"unit": "ms"}}, "user": {"properties": {"name": {"type": "keyword"}}}}}})).await;
    es.put("/b", json!({"mappings": {"properties": {"title": {"type": "keyword"}, "n": {"type": "long", "index": false}}}})).await;

    let (st, v) = es.get("/a,b/_field_caps?fields=title,n").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(
        v,
        json!({
            "indices": ["a", "b"],
            "fields": {
                "n": {"long": {"type": "long", "metadata_field": false, "searchable": true, "aggregatable": true, "inference": false, "meta": {"unit": ["ms"]}}},
                "title": {
                    "keyword": {"type": "keyword", "metadata_field": false, "searchable": true, "aggregatable": true, "inference": false, "indices": ["b"]},
                    "text": {"type": "text", "metadata_field": false, "searchable": true, "aggregatable": false, "inference": false, "indices": ["a"]}
                }
            }
        })
    );
    let (_, v) = es.post("/a,b/_field_caps?include_unmapped=true", json!({"fields": ["user.*", "title.raw"]})).await;
    assert_eq!(v["fields"]["user.name"]["unmapped"], json!({"type": "unmapped", "metadata_field": false, "searchable": false, "aggregatable": false, "indices": ["b"]}));
    assert_eq!(v["fields"]["user.name"]["keyword"]["indices"], json!(["a"]));
    assert_eq!(v["fields"]["title.raw"]["keyword"]["aggregatable"], true);
    let (_, v) = es.get("/a/_field_caps?fields=*").await;
    assert_eq!(v["fields"]["user"], json!({"object": {"type": "object", "metadata_field": false, "searchable": false, "aggregatable": false, "inference": false}}));
    assert_eq!(v["fields"]["_id"]["_id"], json!({"type": "_id", "metadata_field": true, "searchable": true, "aggregatable": false, "inference": false}));
    assert_eq!(v["fields"]["_index"]["_index"]["aggregatable"], true);
    let (_, v) = es.get("/a/_field_caps?fields=*&filters=-metadata,-multifield,-parent").await;
    assert!(v["fields"].get("_id").is_none() && v["fields"].get("title.raw").is_none() && v["fields"].get("user").is_none());
    let (_, v) = es.get("/a/_field_caps?fields=*&types=keyword").await;
    let names: Vec<&String> = v["fields"].as_object().unwrap().keys().collect();
    // ES also reports the `_tier` metadata field (keyword).
    assert_eq!(names, ["_tier", "title.raw", "user.name"]);
    let (st, v) = es.get("/a/_field_caps").await;
    assert_eq!((st, v["error"]["reason"].clone()), (400, json!("Validation Failed: 1: no fields specified;")));
    let (st, v) = es.get("/nope/_field_caps?fields=*").await;
    assert_eq!((st, v["error"]["type"].clone()), (404, json!("index_not_found_exception")));
}

#[tokio::test]
async fn async_search_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es, 5).await;

    // Completes within the timeout, not stored → no id.
    let (st, v) = es.post("/src/_async_search?size=2", json!({"query": {"match_all": {}}, "aggs": {"t": {"terms": {"field": "tag"}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert!(v.get("id").is_none());
    assert_eq!(v["is_partial"], false);
    assert_eq!(v["is_running"], false);
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["is_partial", "is_running", "start_time_in_millis", "expiration_time_in_millis", "completion_time_in_millis", "response"]);
    assert_eq!(v["response"]["hits"]["total"]["value"], 5);
    assert_eq!(v["response"]["hits"]["hits"].as_array().unwrap().len(), 2);
    assert_eq!(v["response"]["aggregations"]["t"]["buckets"].as_array().unwrap().len(), 2);

    // keep_on_completion → stored; get / status / delete.
    let (_, v) = es.post("/src/_async_search?keep_on_completion=true&wait_for_completion_timeout=5s&keep_alive=1m", json!({})).await;
    let id = v["id"].as_str().unwrap().to_string();
    let (st, g) = es.get(&format!("/_async_search/{id}")).await;
    assert_eq!(st, 200);
    assert_eq!((g["id"].as_str(), g["is_running"].clone()), (Some(id.as_str()), json!(false)));
    assert_eq!(g["response"]["hits"]["total"]["value"], 5);
    let (st, s) = es.get(&format!("/_async_search/status/{id}")).await;
    assert_eq!(st, 200);
    let keys: Vec<&String> = s.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["id", "is_running", "is_partial", "start_time_in_millis", "expiration_time_in_millis", "completion_time_in_millis", "_shards", "completion_status"]);
    assert_eq!(s["_shards"], json!({"total": 1, "successful": 1, "skipped": 0, "failed": 0}));
    assert_eq!(s["completion_status"], 200);
    let (st, d) = es.delete(&format!("/_async_search/{id}")).await;
    assert_eq!((st, d), (200, json!({"acknowledged": true})));
    let (st, e) = es.get(&format!("/_async_search/{id}")).await;
    assert_eq!(st, 404);
    assert_eq!(e["error"]["type"], "resource_not_found_exception");
    assert_eq!(e["error"]["reason"], id);
    let (st, _) = es.delete(&format!("/_async_search/{id}")).await;
    assert_eq!(st, 404);

    // Request errors come back as plain ES errors.
    let (st, e) = es.post("/src/_async_search", json!({"bogus": 1})).await;
    assert_eq!((st, e["error"]["type"].clone()), (400, json!("parsing_exception")));
    // The search itself fails asynchronously: ES reports it as the task's failure.
    let (st, e) = es.post("/missing/_async_search", json!({})).await;
    assert_eq!((st, e["error"]["type"].clone(), e["error"]["caused_by"]["type"].clone()), (404, json!("status_exception"), json!("index_not_found_exception")));
}

#[tokio::test]
async fn data_streams_and_rollover() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;

    let (st, e) = es.put0("/_data_stream/logs-app").await;
    assert_eq!((st, e["error"]["reason"].clone()), (400, json!("no matching index template found for data stream [logs-app]")));
    let (st, _) = es.put("/_index_template/logs", json!({"index_patterns": ["logs-*"], "data_stream": {}, "priority": 10, "template": {"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"msg": {"type": "keyword"}}}}})).await;
    assert_eq!(st, 200);

    // Auto-created on first write (POST _doc = op_type create).
    let (st, v) = es.post("/logs-app/_doc?refresh=true", json!({"@timestamp": "2024-01-01T00:00:00Z", "msg": "hello"})).await;
    assert_eq!(st, 201, "{v}");
    let backing = v["_index"].as_str().unwrap().to_string();
    assert!(backing.starts_with(".ds-logs-app-") && backing.ends_with("-000001"), "{backing}");
    // op_type=index refused, @timestamp required.
    let (st, e) = es.put("/logs-app/_doc/1", json!({"@timestamp": "2024-01-01T00:00:00Z"})).await;
    assert_eq!((st, e["error"]["reason"].clone()), (400, json!("only write ops with an op_type of create are allowed in data streams")));
    let (st, e) = es.put("/logs-app/_create/1", json!({"msg": "no ts"})).await;
    assert_eq!(st, 400);
    assert_eq!(e["error"]["type"], "document_parsing_exception");
    assert_eq!(e["error"]["caused_by"]["reason"], "data stream timestamp field [@timestamp] is missing");
    let (st, _) = es.put("/logs-app/_create/1?refresh=true", json!({"@timestamp": "2024-01-02T00:00:00Z", "msg": "two"})).await;
    assert_eq!(st, 201);

    let (st, g) = es.get("/_data_stream/logs-app").await;
    assert_eq!(st, 200);
    let ds = &g["data_streams"][0];
    assert_eq!(ds["name"], "logs-app");
    assert_eq!(ds["timestamp_field"], json!({"name": "@timestamp"}));
    assert_eq!(ds["generation"], 1);
    assert_eq!(ds["template"], "logs");
    assert_eq!(ds["status"], "GREEN");
    assert_eq!(ds["indices"][0]["index_name"], backing);
    let (st, e) = es.put0("/_data_stream/logs-app").await;
    assert_eq!((st, e["error"]["type"].clone()), (400, json!("resource_already_exists_exception")));

    // Rollover: dry run, unmet conditions, then a real one.
    let (st, r) = es.post("/logs-app/_rollover?dry_run", json!({"conditions": {"max_docs": 1}})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["dry_run"], true);
    assert_eq!(r["rolled_over"], false);
    assert_eq!(r["conditions"], json!({"[max_docs: 1]": true}));
    let (_, r) = es.post("/logs-app/_rollover", json!({"conditions": {"max_docs": 100, "max_age": "7d"}})).await;
    assert_eq!((r["rolled_over"].clone(), r["acknowledged"].clone()), (json!(false), json!(false)));
    assert_eq!(r["conditions"], json!({"[max_docs: 100]": false, "[max_age: 7d]": false}));
    let (st, r) = es.post("/logs-app/_rollover", json!({})).await;
    assert_eq!(st, 200);
    let keys: Vec<&String> = r.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["acknowledged", "shards_acknowledged", "old_index", "new_index", "rolled_over", "dry_run", "lazy", "conditions"]);
    assert_eq!(r["old_index"], backing);
    assert!(r["new_index"].as_str().unwrap().ends_with("-000002"));
    let (_, v) = es.post("/logs-app/_doc?refresh=true", json!({"@timestamp": "2024-01-03T00:00:00Z"})).await;
    assert_eq!(v["_index"], r["new_index"]);
    let (_, c) = es.get("/logs-app/_count").await;
    assert_eq!(c["count"], 3);
    let (_, c) = es.get("/logs-*/_count").await;
    assert_eq!(c["count"], 3);
    let (_, g) = es.get("/_data_stream").await;
    assert_eq!(g["data_streams"][0]["generation"], 2);

    let (st, _) = es.delete("/_data_stream/logs-app").await;
    assert_eq!(st, 200);
    let (st, e) = es.get("/_data_stream/logs-app").await;
    assert_eq!((st, e["error"]["type"].clone()), (404, json!("index_not_found_exception")));
    let (st, _) = es.get(&format!("/{backing}")).await;
    assert_eq!(st, 404);

    // Alias rollover with an explicit write index.
    es.put("/web-000001", json!({"aliases": {"web": {"is_write_index": true}}})).await;
    let (st, r) = es.post("/web/_rollover", json!({"conditions": {"max_docs": 0}})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!((r["old_index"].clone(), r["new_index"].clone(), r["rolled_over"].clone()), (json!("web-000001"), json!("web-000002"), json!(true)));
    let (_, a) = es.get("/_alias/web").await;
    assert_eq!(a, json!({"web-000001": {"aliases": {"web": {"is_write_index": false}}}, "web-000002": {"aliases": {"web": {"is_write_index": true}}}}));
    // Alias without explicit write index moves.
    es.put("/app-1", json!({"aliases": {"app": {}}})).await;
    let (_, r) = es.post("/app/_rollover/app-new", json!({})).await;
    assert_eq!(r["new_index"], "app-new");
    let (_, a) = es.get("/_alias/app").await;
    assert_eq!(a, json!({"app-new": {"aliases": {"app": {}}}}));
    let (st, e) = es.post("/app-1/_rollover", json!({})).await;
    assert_eq!((st, e["error"]["reason"].clone()), (400, json!("rollover target is a [concrete index] but one of [alias,data_stream] was expected")));
    let (_, e) = es.post("/ghost/_rollover", json!({})).await;
    assert_eq!(e["error"]["reason"], "rollover target [ghost] does not exist");
    es.put("/plain", json!({"aliases": {"pl": {}}})).await;
    let (_, e) = es.post("/pl/_rollover", json!({})).await;
    assert_eq!(e["error"]["reason"], "index name [plain] does not match pattern '^.*-\\d+$'");
}

#[tokio::test]
async fn asynchronous_force_merge_retains_real_task_result() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    assert_eq!(es.put("/merge", json!({"settings":{"number_of_shards":1,"number_of_replicas":0}})).await.0, 200);
    assert_eq!(es.put("/merge/_doc/1", json!({"text":"merge task"})).await.0, 201);
    assert_eq!(es.post0("/merge/_refresh").await.0, 200);
    let (status, response) = es.post0("/merge/_forcemerge?wait_for_completion=false&max_num_segments=1").await;
    assert_eq!(status, 200);
    let task = response["task"].as_str().unwrap();
    let (status, result) = es.get(&format!("/_tasks/{task}?wait_for_completion=true&timeout=30s")).await;
    assert_eq!(status, 200, "{result}");
    assert_eq!(result["completed"], true);
    assert_eq!(result["task"]["action"], "indices:admin/forcemerge");
    assert_eq!(result["response"]["_shards"]["successful"], 1);
    assert_eq!(es.get("/merge/_count").await.1["count"], 1);
    // Native ES permits the explicit default -1 with expunge-only.
    assert_eq!(es.post0("/merge/_forcemerge?only_expunge_deletes=true&max_num_segments=-1").await.0, 200);
    assert_eq!(es.post0("/merge/_forcemerge?only_expunge_deletes=true&max_num_segments=1").await.0, 400);
}
