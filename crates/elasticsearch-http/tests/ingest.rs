//! HTTP tests for ingest pipelines: CRUD, simulate (pipeline + ingest
//! variants), processors, and `?pipeline=` / default / final pipelines on
//! index, bulk and reindex.

use elasticsearch_engine::node::Node;
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
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.call(Method::DELETE, p, None).await
    }
    async fn source(&self, index: &str, id: &str) -> Value {
        let (st, v) = self.get(&format!("/{index}/_doc/{id}")).await;
        assert_eq!(st, 200, "{v}");
        v["_source"].clone()
    }
    async fn simulate(&self, processors: Value, docs: Value) -> Value {
        let (st, v) = self.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": processors}, "docs": docs})).await;
        assert_eq!(st, 200, "{v}");
        v
    }
}

#[tokio::test]
async fn pipeline_crud() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/my_pipeline", json!({"description": "_description", "version": 10, "_meta": {"a": 1}, "processors": [{"set": {"field": "x", "value": 1}}]})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (st, v) = es.get("/_ingest/pipeline/my_pipeline").await;
    assert_eq!(st, 200);
    assert_eq!(v["my_pipeline"]["description"], "_description");
    assert_eq!(v["my_pipeline"]["version"], 10);
    assert_eq!(v["my_pipeline"]["_meta"], json!({"a": 1}));
    es.put("/_ingest/pipeline/second", json!({"description": "second", "processors": []})).await;
    // Get-all also lists the built-in pipelines the template registries install.
    let (_, v) = es.get("/_ingest/pipeline").await;
    assert_eq!(v["second"]["description"], "second");
    assert_eq!(v["my_pipeline"]["version"], 10);
    let (_, v) = es.get("/_ingest/pipeline?summary=true").await;
    assert_eq!((&v["my_pipeline"], &v["second"]), (&json!({}), &json!({})));
    assert!(v.as_object().unwrap().values().all(|p| p == &json!({})), "{v}");
    let (_, v) = es.get("/_ingest/pipeline/my_*").await;
    assert_eq!(v.as_object().unwrap().keys().collect::<Vec<_>>(), vec!["my_pipeline"]);
    // ES has no HEAD route for pipelines: 405 with `Allow: GET,PUT,DELETE`.
    let r = es.c.head(format!("{}/_ingest/pipeline/second", es.base)).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 405);
    // if_version
    let (st, v) = es.put("/_ingest/pipeline/my_pipeline?if_version=9", json!({"processors": []})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "version conflict, required version [9] for pipeline [my_pipeline] but current version is [10]");
    let (st, _) = es.put("/_ingest/pipeline/my_pipeline?if_version=10", json!({"processors": []})).await;
    assert_eq!(st, 200);
    let (_, v) = es.get("/_ingest/pipeline/my_pipeline").await;
    assert_eq!(v["my_pipeline"]["version"], 11);
    // invalid configs
    let (st, v) = es.put("/_ingest/pipeline/bad", json!({"processors": [], "invalid_field": {}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "parse_exception");
    let (st, v) = es.put("/_ingest/pipeline/bad", json!({"processors": [{"set": {"value": 1}}]})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "[field] required property is missing");
    assert_eq!(v["error"]["processor_type"], "set");
    assert_eq!(v["error"]["property_name"], "field");
    let (st, v) = es.put("/_ingest/pipeline/bad", json!({"processors": [{"unknown_proc": {}}]})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "No processor type exists with name [unknown_proc]");
    // delete
    let (st, v) = es.delete("/_ingest/pipeline/my_pipeline").await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (st, v) = es.get("/_ingest/pipeline/my_pipeline").await;
    assert_eq!((st, v), (404, json!({})));
    let (st, v) = es.delete("/_ingest/pipeline/my_pipeline").await;
    assert_eq!(st, 404);
    assert_eq!(v["error"]["type"], "resource_not_found_exception");
    assert_eq!(v["error"]["reason"], "pipeline [my_pipeline] is missing");
    // persisted across restart
    drop(es);
    let es = start(tmp.path()).await;
    let (st, v) = es.get("/_ingest/pipeline/second").await;
    assert_eq!(st, 200);
    assert_eq!(v["second"]["description"], "second");
    // grok patterns
    let (st, v) = es.get("/_ingest/processor/grok").await;
    assert_eq!(st, 200);
    assert!(v["patterns"]["COMBINEDAPACHELOG"].is_string());
    assert!(v["patterns"]["IP"].is_string());
}

#[tokio::test]
async fn simulate_pipeline_api() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let v = es
        .simulate(
            json!([{"set": {"field": "field2", "value": "_value"}}, {"remove": {"field": "drop_me"}}, {"rename": {"field": "old", "target_field": "new"}}]),
            json!([{"_index": "index", "_id": "id", "_source": {"foo": "bar", "drop_me": 1, "old": 2}}]),
        )
        .await;
    let d = &v["docs"][0]["doc"];
    assert_eq!(d["_index"], "index");
    assert_eq!(d["_id"], "id");
    assert_eq!(d["_version"], "-3");
    assert_eq!(d["_source"], json!({"foo": "bar", "field2": "_value", "new": 2}));
    assert!(d["_ingest"]["timestamp"].is_string());
    // stored pipeline + verbose
    es.put("/_ingest/pipeline/p1", json!({"processors": [{"set": {"field": "a", "value": 1, "tag": "t"}}, {"set": {"field": "b", "value": 2, "if": "ctx.a == 2"}}, {"fail": {"message": "x", "ignore_failure": true}}]})).await;
    let (st, v) = es.post("/_ingest/pipeline/p1/_simulate?verbose=true", json!({"docs": [{"_source": {}}]})).await;
    assert_eq!(st, 200, "{v}");
    let pr = &v["docs"][0]["processor_results"];
    assert_eq!(pr[0]["processor_type"], "set");
    assert_eq!(pr[0]["status"], "success");
    assert_eq!(pr[0]["tag"], "t");
    assert_eq!(pr[0]["doc"]["_source"], json!({"a": 1}));
    assert_eq!(pr[0]["doc"]["_ingest"]["pipeline"], "p1");
    assert_eq!(pr[1]["status"], "skipped");
    assert_eq!(pr[1]["if"], json!({"condition": "ctx.a == 2", "result": false}));
    assert_eq!(pr[2]["status"], "error_ignored");
    assert_eq!(pr[2]["ignored_error"]["error"]["type"], "fail_processor_exception");
    // failure + drop in non-verbose mode
    let v = es.simulate(json!([{"fail": {"message": "boom {{f}}"}}]), json!([{"_source": {"f": "z"}}])).await;
    assert_eq!(v["docs"][0]["error"]["type"], "fail_processor_exception");
    assert_eq!(v["docs"][0]["error"]["reason"], "boom z");
    assert_eq!(v["docs"][0]["error"]["root_cause"][0]["type"], "fail_processor_exception");
    let v = es.simulate(json!([{"drop": {}}]), json!([{"_source": {}}])).await;
    assert_eq!(v["docs"][0], Value::Null);
    // validation errors
    let (st, v) = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": []}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "[docs] required property is missing");
    assert_eq!(v["error"]["type"], "parse_exception");
    for (docs, kind, reason) in [
        (Value::Null, "parse_exception", "[docs] required property is missing"),
        (json!([]), "illegal_argument_exception", "must specify at least one document in [docs]"),
    ] {
        let (st, v) = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": []}, "docs": docs})).await;
        assert_eq!(st, 400, "{v}");
        assert_eq!(v["error"]["type"], kind);
        assert_eq!(v["error"]["reason"], reason);
    }
    let (st, v) = es.post("/_ingest/pipeline/nope/_simulate", json!({"docs": [{"_source": {}}]})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["reason"], "pipeline [nope] does not exist");
}

#[tokio::test]
async fn processors_via_simulate() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let line = r#"127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://www.example.com/start.html" "Mozilla/4.08 [en] (Win98; I ;Nav)""#;
    let v = es
        .simulate(
            json!([
                {"grok": {"field": "message", "patterns": ["%{COMBINEDAPACHELOG}"]}},
                {"date": {"field": "timestamp", "formats": ["dd/MMM/yyyy:HH:mm:ss Z"]}},
                {"convert": {"field": "response", "type": "integer"}},
                {"convert": {"field": "bytes", "type": "long"}},
                {"json": {"field": "payload", "target_field": "parsed"}},
                {"foreach": {"field": "tags", "processor": {"uppercase": {"field": "_ingest._value"}}}},
                {"script": {"source": "ctx.total = ctx.bytes * params.m", "params": {"m": 2}}},
                {"remove": {"field": ["message", "payload"]}},
                {"rename": {"field": "clientip", "target_field": "client.ip"}}
            ]),
            json!([{"_source": {"message": line, "payload": "{\"k\": [1, 2]}", "tags": ["a", "b"]}}]),
        )
        .await;
    let s = &v["docs"][0]["doc"]["_source"];
    assert_eq!(s["client"]["ip"], "127.0.0.1", "{v}");
    assert_eq!(s["verb"], "GET");
    assert_eq!(s["request"], "/apache_pb.gif");
    assert_eq!(s["response"], 200);
    assert_eq!(s["bytes"], 2326);
    assert_eq!(s["@timestamp"], "2000-10-10T20:55:36.000Z");
    assert_eq!(s["parsed"], json!({"k": [1, 2]}));
    assert_eq!(s["tags"], json!(["A", "B"]));
    assert_eq!(s["total"], 4652);
    assert!(s.get("message").is_none());
    // on_failure metadata
    let v = es
        .simulate(
            json!([{"convert": {"field": "n", "type": "integer", "tag": "conv", "on_failure": [{"set": {"field": "error.message", "value": "{{ _ingest.on_failure_message }}"}}, {"set": {"field": "error.type", "value": "{{_ingest.on_failure_processor_type}}/{{_ingest.on_failure_processor_tag}}"}}]}}]),
            json!([{"_source": {"n": "abc"}}]),
        )
        .await;
    let s = &v["docs"][0]["doc"]["_source"];
    assert_eq!(s["error"], json!({"message": "unable to convert [abc] to integer", "type": "convert/conv"}));
    // pipeline-level on_failure
    let v = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": [{"rename": {"field": "missing", "target_field": "x"}}], "on_failure": [{"set": {"field": "_index", "value": "failed-{{ _index }}"}}]}, "docs": [{"_index": "logs", "_source": {}}]})).await.1;
    assert_eq!(v["docs"][0]["doc"]["_index"], "failed-logs");
    // date errors surface
    let v = es.simulate(json!([{"date": {"field": "t", "formats": ["ISO8601"]}}]), json!([{"_source": {"t": "garbage"}}])).await;
    assert_eq!(v["docs"][0]["error"]["reason"], "unable to parse date [garbage]");
}

#[tokio::test]
async fn index_bulk_with_pipelines() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.put("/_ingest/pipeline/up", json!({"processors": [{"uppercase": {"field": "name"}}, {"set": {"field": "via", "value": "{{_ingest.pipeline}}"}}]})).await;
    es.put("/_ingest/pipeline/def", json!({"processors": [{"set": {"field": "default_ran", "value": true}}, {"drop": {"if": "ctx.skip == true"}}]})).await;
    es.put("/_ingest/pipeline/fin", json!({"processors": [{"set": {"field": "final_ran", "value": true}}]})).await;
    // ?pipeline= on the index API
    let (st, v) = es.put("/plain/_doc/1?pipeline=up&refresh=true", json!({"name": "bob"})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.source("plain", "1").await, json!({"name": "BOB", "via": "up"}));
    let (st, v) = es.put("/plain/_doc/2?pipeline=nope", json!({"name": "x"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "pipeline with id [nope] does not exist");
    // default + final pipelines from index settings
    let (st, _) = es.put("/withdef", json!({"settings": {"index.default_pipeline": "def", "index.final_pipeline": "fin"}})).await;
    assert_eq!(st, 200);
    es.put("/withdef/_doc/1", json!({"a": 1})).await;
    assert_eq!(es.source("withdef", "1").await, json!({"a": 1, "default_ran": true, "final_ran": true}));
    // explicit pipeline replaces the default, final still runs; _none disables default
    es.put("/withdef/_doc/2?pipeline=up", json!({"name": "x"})).await;
    assert_eq!(es.source("withdef", "2").await, json!({"name": "X", "via": "up", "final_ran": true}));
    es.put("/withdef/_doc/3?pipeline=_none", json!({"a": 3})).await;
    assert_eq!(es.source("withdef", "3").await, json!({"a": 3, "final_ran": true}));
    // drop → noop
    let (st, v) = es.put("/withdef/_doc/4", json!({"skip": true})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "noop");
    assert_eq!(v["_version"], -3);
    let (st, _) = es.get("/withdef/_doc/4").await;
    assert_eq!(st, 404);
    // bulk: request-level and item-level pipeline, failures per item, drops
    es.put("/_ingest/pipeline/failing", json!({"processors": [{"fail": {"message": "nope {{id}}"}}]})).await;
    let body = [
        json!({"index": {"_index": "bulk1", "_id": "1"}}),
        json!({"name": "a"}),
        json!({"index": {"_index": "bulk1", "_id": "2", "pipeline": "failing"}}),
        json!({"id": 2}),
        json!({"create": {"_index": "withdef", "_id": "9", "pipeline": "def"}}),
        json!({"skip": true}),
    ]
    .iter()
    .map(|l| l.to_string() + "\n")
    .collect::<String>();
    let r = es.c.post(format!("{}/_bulk?pipeline=up&refresh=true", es.base)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], true, "{v}");
    assert_eq!(v["items"][0]["index"]["status"], 201);
    assert_eq!(v["items"][1]["index"]["status"], 500);
    assert_eq!(v["items"][1]["index"]["error"]["type"], "fail_processor_exception");
    assert_eq!(v["items"][1]["index"]["error"]["reason"], "nope 2");
    assert_eq!(v["items"][2]["create"]["result"], "noop", "{v}");
    assert_eq!(es.source("bulk1", "1").await, json!({"name": "A", "via": "up"}));
    // a request pipeline that reroutes via _index runs only the new index's
    // final pipeline (ES 8.19: the default pipeline is not re-resolved)
    es.put("/_ingest/pipeline/route", json!({"processors": [{"set": {"field": "_index", "value": "withdef"}}]})).await;
    let (st, v) = es.put("/elsewhere/_doc/r1?pipeline=route", json!({"x": 1})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(v["_index"], "withdef");
    assert_eq!(es.source("withdef", "r1").await, json!({"x": 1, "final_ran": true}));
    // pipelines used by an index cannot be deleted
    let (st, v) = es.delete("/_ingest/pipeline/def").await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("pipeline [def] cannot be deleted because it is the default pipeline"), "{v}");
}

/// A bulk source line with text after its first object (Rally's
/// github_archive corpus has a few): with a pipeline, ES reads the source
/// with `XContentParser.map()`, which stops at the root object's closing
/// brace, and writes the pipeline's output, so the rest is ignored and the
/// document indexed. rustic failed these items with serde's "trailing
/// characters at line 1 column N". Without a pipeline ES's
/// `DocumentParser.validateEnd` rejects the extra data.
#[tokio::test]
async fn bulk_pipeline_ignores_text_after_the_source_object() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/stamp", json!({"processors": [{"set": {"field": "p", "value": true}}]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/gh", json!({"settings": {"index.default_pipeline": "stamp"}})).await;
    assert_eq!(st, 200, "{v}");
    let body = concat!(
        "{\"index\":{\"_index\":\"gh\",\"_id\":\"1\"}}\n",
        "{\"a\":1,\"s\":\"x}{\"}{\"a\":2}\n",
        "{\"index\":{\"_index\":\"gh\",\"_id\":\"2\"}}\n",
        "{\"a\":3} trailing ] garbage\r\n",
        "{\"index\":{\"_index\":\"gh\",\"_id\":\"3\"}}\n",
        "{\"a\":4}}\n",
        "{\"index\":{\"_index\":\"plain\",\"_id\":\"1\"}}\n",
        "{\"a\":5}{\"a\":6}\n",
    );
    let r = es.c.post(format!("{}/_bulk?refresh=true", es.base)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    for i in 0..3 {
        assert_eq!(v["items"][i]["index"]["status"], 201, "item {i}: {v}");
    }
    assert_eq!(v["items"][3]["index"]["status"], 400, "{v}");
    assert_eq!(es.source("gh", "1").await, json!({"a": 1, "s": "x}{", "p": true}));
    assert_eq!(es.source("gh", "2").await, json!({"a": 3, "p": true}));
    assert_eq!(es.source("gh", "3").await, json!({"a": 4, "p": true}));
    let (st, v) = es.get("/gh/_count").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["count"], 3, "{v}");
}

#[tokio::test]
async fn simulate_ingest_and_reindex() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.put("/_ingest/pipeline/my-pipeline", json!({"processors": []})).await;
    es.put("/_ingest/pipeline/my-final-pipeline", json!({"processors": [{"set": {"field": "fin", "value": 1}}]})).await;
    es.put("/index", json!({"settings": {"default_pipeline": "my-pipeline", "final_pipeline": "my-final-pipeline"}})).await;
    let (st, v) = es
        .post(
            "/_ingest/_simulate",
            json!({"docs": [{"_index": "index", "_id": "id", "_source": {"foo": "bar"}}, {"_index": "other", "_id": "id", "_source": {"foo": "rab"}}],
                   "pipeline_substitutions": {"my-pipeline": {"processors": [{"set": {"field": "sub", "value": true}}]}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["docs"][0]["doc"]["_index"], "index");
    assert_eq!(v["docs"][0]["doc"]["_source"], json!({"foo": "bar", "sub": true, "fin": 1}));
    assert_eq!(v["docs"][0]["doc"]["executed_pipelines"], json!(["my-pipeline", "my-final-pipeline"]));
    assert_eq!(v["docs"][1]["doc"]["executed_pipelines"], json!([]));
    assert_eq!(v["docs"][1]["doc"]["_source"], json!({"foo": "rab"}));
    let (_, v) = es.post("/_ingest/test-index/_simulate?pipeline=my-final-pipeline", json!({"docs": [{"_id": "1", "_source": {"a": 1}}]})).await;
    assert_eq!(v["docs"][0]["doc"]["_index"], "test-index");
    assert_eq!(v["docs"][0]["doc"]["executed_pipelines"], json!(["my-final-pipeline"]));
    // mapping validation of the simulated result
    es.put("/strict", json!({"mappings": {"dynamic": "strict", "properties": {"foo": {"type": "text"}}}})).await;
    let (_, v) = es.post("/_ingest/_simulate", json!({"docs": [{"_index": "strict", "_id": "1", "_source": {"foob": "x"}}]})).await;
    assert_eq!(v["docs"][0]["doc"]["error"]["type"], "strict_dynamic_mapping_exception");
    // reindex with dest.pipeline
    es.put("/_ingest/pipeline/tag", json!({"processors": [{"set": {"field": "tagged", "value": true}}]})).await;
    es.put("/src/_doc/1?refresh=true", json!({"n": 1})).await;
    let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": "src"}, "dest": {"index": "dst", "pipeline": "tag"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["created"], 1);
    assert_eq!(es.source("dst", "1").await, json!({"n": 1, "tagged": true}));
}

#[tokio::test]
async fn verbose_simulate_keeps_status_conditions_and_lossless_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let response = es.c.post(format!("{}/_ingest/pipeline/_simulate?verbose=true", es.base))
        .header("content-type", "application/json")
        .body(r#"{"pipeline":{"processors":[{"set":{"field":"a","value":1,"description":"first","if":"true"}},{"set":{"field":"b","value":2,"if":"false"}},{"fail":{"message":"ignored","ignore_failure":true}},{"drop":{}}]},"docs":[{"_source":{"raw":"\ud800"}}]}"#)
        .send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let text = response.text().await.unwrap();
    // Check the raw JSON boundary before using a Unicode-only parser for metadata.
    assert!(text.to_ascii_lowercase().contains("\\ud800"), "{text}");
    let value: Value = serde_json::from_str(&text.replace("\\ud800", "raw-unit")).unwrap();
    let entries = value["docs"][0]["processor_results"].as_array().unwrap();
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0]["status"], "success");
    assert_eq!(entries[0]["description"], "first");
    assert_eq!(entries[0]["if"], json!({"condition":"true","result":true}));
    assert_eq!(entries[0]["doc"]["_source"]["a"], 1);
    assert_eq!(entries[1]["status"], "skipped");
    assert_eq!(entries[1]["if"], json!({"condition":"false","result":false}));
    assert!(entries[1].get("doc").is_none());
    assert_eq!(entries[2]["status"], "error_ignored");
    assert_eq!(entries[2]["ignored_error"]["error"]["type"], "fail_processor_exception");
    assert_eq!(entries[2]["doc"]["_source"]["raw"], "raw-unit");
    assert_eq!(entries[3]["status"], "dropped");
}
