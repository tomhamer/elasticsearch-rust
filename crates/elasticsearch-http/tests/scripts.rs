//! Painless scripting over HTTP: every wired script context.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

struct Es {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn start(dir: &Path) -> (Es, Arc<Node>) {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let n = node.clone();
    let task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    (Es { base: format!("http://{addr}"), client: Client::new(), _task: task }, node)
}

impl Es {
    async fn req(&self, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.req(Method::GET, p, None).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.req(Method::PUT, p, Some(b)).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.req(Method::POST, p, Some(b)).await
    }
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.req(Method::DELETE, p, None).await
    }
    async fn search(&self, idx: &str, b: Value) -> Value {
        let (st, v) = self.post(&format!("/{idx}/_search"), b.clone()).await;
        assert_eq!(st, 200, "{b} -> {v}");
        v
    }
}

fn ids(v: &Value) -> Vec<String> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

async fn seed(es: &Es) {
    let (st, v) = es
        .put(
            "/items",
            json!({"mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "title": {"type": "text"},
                    "price": {"type": "double"},
                    "qty": {"type": "long"},
                    "tags": {"type": "keyword"},
                    "sold": {"type": "date"},
                    "active": {"type": "boolean"}
                },
                "runtime": {"price_x2": {"type": "double", "script": {"source": "emit(doc['price'].value * 2)"}}}
            }}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    let docs = [
        ("1", json!({"name": "apple", "title": "red apple", "price": 1.5, "qty": 10, "tags": ["fruit", "red"], "sold": "2024-01-01T10:00:00Z", "active": true})),
        ("2", json!({"name": "banana", "title": "yellow banana", "price": 0.5, "qty": 30, "tags": ["fruit"], "sold": "2024-01-06T10:00:00Z", "active": false})),
        ("3", json!({"name": "carrot", "title": "orange carrot", "price": 2.0, "qty": 5, "tags": ["veg"], "sold": "2024-02-03T12:00:00Z", "active": true})),
        ("4", json!({"name": "durian", "title": "smelly durian", "price": 9.0, "tags": ["fruit"], "active": true})),
    ];
    for (id, d) in docs {
        let (st, v) = es.put(&format!("/items/_doc/{id}?refresh=true"), d).await;
        assert!(st == 201 || st == 200, "{v}");
    }
}

#[tokio::test]
async fn script_score_and_function_score() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let v = es
        .search("items", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "doc['price'].value * params.factor + _score", "params": {"factor": 2}}}}}))
        .await;
    assert_eq!(ids(&v), ["4", "3", "1", "2"]);
    assert_eq!(v["hits"]["hits"][0]["_score"], json!(19.0));
    assert_eq!(v["hits"]["max_score"], json!(19.0));
    // min_score + boost
    let v = es
        .search("items", json!({"query": {"script_score": {"query": {"term": {"tags": "fruit"}}, "script": "doc['qty'].size() == 0 ? 0 : doc['qty'].value", "min_score": 9, "boost": 1}}}))
        .await;
    assert_eq!(ids(&v), ["2", "1"]);
    // function_score with script_score + weight + filter
    let v = es
        .search(
            "items",
            json!({"query": {"function_score": {
                "query": {"match_all": {}},
                "functions": [
                    {"script_score": {"script": {"source": "Math.log(2 + doc['price'].value)"}}},
                    {"filter": {"term": {"name": "banana"}}, "weight": 100}
                ],
                "score_mode": "sum", "boost_mode": "replace"
            }}}),
        )
        .await;
    assert_eq!(ids(&v)[0], "2");
    let s = v["hits"]["hits"][1]["_score"].as_f64().unwrap();
    assert!((s - (11.0f64).ln()).abs() < 1e-5, "{v}");
    // Runtime error → search_phase_execution_exception wrapping script_exception.
    let (st, v) = es.post("/items/_search", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": "doc['qty'].value"}}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "search_phase_execution_exception");
    assert_eq!(v["error"]["root_cause"][0]["type"], "script_exception");
    let reason = &v["error"]["failed_shards"][0]["reason"];
    assert_eq!(reason["type"], "script_exception");
    assert_eq!(reason["reason"], "runtime error");
    assert_eq!(reason["lang"], "painless");
    assert_eq!(reason["script"], "doc['qty'].value");
    assert_eq!(reason["script_stack"], json!(["doc['qty'].value", "          ^---- HERE"]));
    assert_eq!(reason["position"], json!({"offset": 10, "start": 0, "end": 16}));
    assert_eq!(reason["caused_by"]["type"], "illegal_state_exception");
    // Compile error.
    let (st, v) = es.post("/items/_search", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": "doc['qty'].value +"}}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["failed_shards"][0]["reason"]["reason"], "compile error");
    assert!(v["error"]["failed_shards"][0]["reason"]["position"]["offset"].is_number());
}

#[tokio::test]
async fn script_query_fields_and_sort() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    // script filter
    let v = es
        .search("items", json!({"query": {"bool": {"filter": {"script": {"script": {"source": "doc['price'].value > params.min && doc['active'].value", "params": {"min": 1.0}}}}}}, "sort": ["name"]}))
        .await;
    assert_eq!(ids(&v), ["1", "3", "4"]);
    // script in must with a selective term (seek path)
    let v = es.search("items", json!({"query": {"bool": {"must": [{"term": {"tags": "fruit"}}, {"script": {"script": "doc['name'].value.startsWith('b')"}}]}}})).await;
    assert_eq!(ids(&v), ["2"]);
    // script_fields: no _source unless asked
    let v = es
        .search(
            "items",
            json!({"query": {"ids": {"values": ["1"]}}, "script_fields": {
                "total": {"script": {"source": "doc['price'].value * doc['qty'].value"}},
                "upper": {"script": "doc['name'].value.toUpperCase()"},
                "tags": {"script": "doc['tags']"},
                "month": {"script": "doc['sold'].value.getMonthValue()"},
                "src": {"script": "params._source.title"}
            }}),
        )
        .await;
    let h = &v["hits"]["hits"][0];
    assert!(h.get("_source").is_none(), "{h}");
    assert_eq!(h["fields"]["total"], json!([15.0]));
    assert_eq!(h["fields"]["upper"], json!(["APPLE"]));
    assert_eq!(h["fields"]["tags"], json!(["fruit", "red"]));
    assert_eq!(h["fields"]["month"], json!([1]));
    assert_eq!(h["fields"]["src"], json!(["red apple"]));
    // max_script_fields
    let (st, v) = es.put("/limited", json!({"settings": {"index.max_script_fields": 1}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/limited/_search", json!({"script_fields": {"a": {"script": "1"}, "b": {"script": "2"}}})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["root_cause"][0]["reason"].as_str().unwrap().contains("Trying to retrieve too many script_fields"), "{v}");
    // _script sort (number and string)
    let v = es
        .search("items", json!({"sort": {"_script": {"type": "number", "script": {"source": "doc['price'].value * -1"}, "order": "asc"}}}))
        .await;
    assert_eq!(ids(&v), ["4", "3", "1", "2"]);
    assert_eq!(v["hits"]["hits"][0]["sort"], json!([-9.0]));
    let v = es.search("items", json!({"sort": [{"_script": {"type": "string", "script": "doc['name'].value.substring(1)", "order": "desc"}}], "size": 2})).await;
    assert_eq!(ids(&v), ["4", "1"]);
    assert_eq!(v["hits"]["hits"][0]["sort"], json!(["urian"]));
}

#[tokio::test]
async fn scripted_update_and_upsert() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/items/_update/1", json!({"script": {"source": "ctx._source.qty += params.n; ctx._source.tags.add('new')", "params": {"n": 5}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "updated");
    let (_, g) = es.get("/items/_doc/1").await;
    assert_eq!(g["_source"]["qty"], json!(15));
    assert_eq!(g["_source"]["tags"], json!(["fruit", "red", "new"]));
    // noop
    let (st, v) = es.post("/items/_update/1", json!({"script": "ctx.op = 'noop'"})).await;
    assert_eq!(st, 200);
    assert_eq!(v["result"], "noop");
    // delete
    let (st, v) = es.post("/items/_update/2", json!({"script": "if (ctx._source.price < 1) { ctx.op = 'delete' }"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "deleted");
    let (st, _) = es.get("/items/_doc/2").await;
    assert_eq!(st, 404);
    // upsert (script not run) and scripted_upsert
    let (st, v) = es.post("/items/_update/9", json!({"script": "ctx._source.qty += 1", "upsert": {"qty": 1}})).await;
    assert_eq!(st, 201, "{v}");
    let (_, g) = es.get("/items/_doc/9").await;
    assert_eq!(g["_source"]["qty"], json!(1));
    let (st, v) = es.post("/items/_update/10", json!({"script": {"source": "ctx._source.qty = params.q * 2", "params": {"q": 21}}, "upsert": {}, "scripted_upsert": true})).await;
    assert_eq!(st, 201, "{v}");
    let (_, g) = es.get("/items/_doc/10").await;
    assert_eq!(g["_source"]["qty"], json!(42));
    // missing doc without upsert
    let (st, v) = es.post("/items/_update/404", json!({"script": "ctx._source.x = 1"})).await;
    assert_eq!(st, 404);
    assert_eq!(v["error"]["type"], "document_missing_exception");
    // script error
    let (st, v) = es.post("/items/_update/1", json!({"script": "ctx._source.nope.length()"})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "script_exception");
    assert_eq!(v["error"]["caused_by"]["type"], "null_pointer_exception");
    // bad op
    let (st, v) = es.post("/items/_update/1", json!({"script": "ctx.op = 'explode'"})).await;
    assert_eq!(st, 400, "{v}");
}

#[tokio::test]
async fn update_by_query_with_script() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es
        .post("/items/_update_by_query?refresh=true", json!({"query": {"term": {"tags": "fruit"}}, "script": {"source": "if (ctx._source.price > 5) { ctx.op = 'noop' } else { ctx._source.price *= 10 }"}}))
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["updated"], 2, "{v}");
    assert_eq!(v["noops"], 1, "{v}");
    let (_, g) = es.get("/items/_doc/1").await;
    assert_eq!(g["_source"]["price"], json!(15.0));
}

#[tokio::test]
async fn aggregation_scripts_and_pipelines() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let v = es
        .search(
            "items",
            json!({"size": 0, "aggs": {
                "first_letter": {"terms": {"script": "doc['name'].value.substring(0, 1)", "order": {"_key": "asc"}}},
                "total": {"sum": {"script": {"source": "doc['qty'].size() == 0 ? 0 : doc['qty'].value * params.m", "params": {"m": 2}}}},
                "avg_doubled": {"avg": {"field": "price", "script": "_value * 2"}},
                "by_active": {"terms": {"field": "active"}, "aggs": {
                    "p": {"sum": {"field": "price"}},
                    "c": {"value_count": {"field": "price"}},
                    "ratio": {"bucket_script": {"buckets_path": {"p": "p", "c": "c"}, "script": "params.p / params.c"}},
                    "keep": {"bucket_selector": {"buckets_path": {"c": "c"}, "script": "params.c > 1"}}
                }}
            }}),
        )
        .await;
    let a = &v["aggregations"];
    let keys: Vec<&str> = a["first_letter"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_str().unwrap()).collect();
    assert_eq!(keys, ["a", "b", "c", "d"]);
    assert_eq!(a["total"]["value"], json!(90.0));
    assert_eq!(a["avg_doubled"]["value"], json!(6.5));
    let bs = a["by_active"]["buckets"].as_array().unwrap();
    assert_eq!(bs.len(), 1, "{v}");
    assert_eq!(bs[0]["key_as_string"], "true");
    assert!((bs[0]["ratio"]["value"].as_f64().unwrap() - 12.5 / 3.0).abs() < 1e-9);
}

#[tokio::test]
async fn runtime_fields() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let v = es
        .search(
            "items",
            json!({
                "runtime_mappings": {
                    "day": {"type": "keyword", "script": "if (!doc['sold'].empty) { emit(doc['sold'].value.getDayOfWeek().toString()) }"},
                    "cheap": {"type": "boolean", "script": "emit(doc['price'].value < 1.6)"},
                    "sold_plus": {"type": "date", "script": "if (doc['sold'].size() > 0) { emit(doc['sold'].value.toInstant().toEpochMilli() + 86400000L) }"},
                    "name": {"type": "keyword"}
                },
                "query": {"bool": {"filter": [{"term": {"cheap": true}}, {"exists": {"field": "day"}}]}},
                "fields": ["day", "price_x2", "sold_plus", "cheap"],
                "sort": [{"price_x2": "desc"}],
                "aggs": {"days": {"terms": {"field": "day"}}}
            }),
        )
        .await;
    assert_eq!(ids(&v), ["1", "2"], "{v}");
    let h = &v["hits"]["hits"][0];
    assert_eq!(h["fields"]["day"], json!(["MONDAY"]));
    assert_eq!(h["fields"]["price_x2"], json!([3.0]));
    assert_eq!(h["fields"]["sold_plus"], json!(["2024-01-02T10:00:00.000Z"]));
    assert_eq!(h["sort"], json!([3.0]));
    let days: Vec<&str> = v["aggregations"]["days"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_str().unwrap()).collect();
    assert_eq!(days, ["MONDAY", "SATURDAY"]);
    // range on a runtime double from the mapping, and doc access to it
    let v = es
        .search("items", json!({"query": {"range": {"price_x2": {"gte": 4}}}, "script_fields": {"rt": {"script": "doc['price_x2'].value + 1"}}, "sort": ["name"]}))
        .await;
    assert_eq!(ids(&v), ["3", "4"]);
    assert_eq!(v["hits"]["hits"][0]["fields"]["rt"], json!([5.0]));
}

#[tokio::test]
async fn stored_scripts_and_execute_api() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _n) = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es.put("/_scripts/calc", json!({"script": {"lang": "painless", "source": "doc['price'].value * params.f"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v, json!({"acknowledged": true}));
    let (st, v) = es.get("/_scripts/calc").await;
    assert_eq!(st, 200);
    assert_eq!(v, json!({"_id": "calc", "found": true, "script": {"lang": "painless", "source": "doc['price'].value * params.f"}}));
    let v = es.search("items", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": {"id": "calc", "params": {"f": 3}}}}, "size": 1})).await;
    assert_eq!(v["hits"]["hits"][0]["_score"], json!(27.0));
    let (st, v) = es.put("/_scripts/bad", json!({"script": {"lang": "painless", "source": "1 +"}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "script_exception");
    let (st, _) = es.delete("/_scripts/calc").await;
    assert_eq!(st, 200);
    let (st, v) = es.get("/_scripts/calc").await;
    assert_eq!(st, 404);
    assert_eq!(v, json!({"_id": "calc", "found": false}));
    let (st, v) = es.delete("/_scripts/calc").await;
    assert_eq!(st, 404);
    assert_eq!(v["error"]["type"], "resource_not_found_exception");
    let (st, v) = es.post("/items/_search", json!({"query": {"script": {"script": {"id": "calc"}}}})).await;
    assert_eq!(st, 404, "{v}");
    // _execute
    let (st, v) = es.post("/_scripts/painless/_execute", json!({"script": {"source": "params.count / params.total", "params": {"count": 100.0, "total": 1000.0}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v, json!({"result": "0.1"}));
    let (_, v) = es
        .post("/_scripts/painless/_execute", json!({"script": {"source": "doc['price'].value > 1"}, "context": "filter", "context_setup": {"index": "items", "document": {"price": 2.5}}}))
        .await;
    assert_eq!(v, json!({"result": true}));
    let (_, v) = es
        .post("/_scripts/painless/_execute", json!({"script": {"source": "emit(doc['name'].value + '!')"}, "context": "keyword_field", "context_setup": {"index": "items", "document": {"name": "x"}}}))
        .await;
    assert_eq!(v, json!({"result": ["x!"]}));
    let (st, v) = es.post("/_scripts/painless/_execute", json!({"script": {"source": "for (;;) {}"}})).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = es.get("/_script_context").await;
    assert_eq!(st, 200);
    assert_eq!(v["contexts"][0]["methods"][0]["name"], "execute");
    let (_, v) = es.get("/_script_language").await;
    assert_eq!(v["types_allowed"], json!(["inline", "stored"]));
}
