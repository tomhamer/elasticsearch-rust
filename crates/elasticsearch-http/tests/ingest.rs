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

async fn bulk(es: &Es, lines: &[Value]) -> Value {
    let body = lines.iter().map(|l| l.to_string() + "\n").collect::<String>();
    let r = es.c.post(format!("{}/_bulk?refresh=true", es.base)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
    r.json().await.unwrap()
}

/// Version metadata a pipeline sets applies to the write (ES
/// `IngestService.updateIndexRequestMetadata`): an external version taken
/// from the event makes an older event conflict (409) and keeps the newer
/// document; the pipeline sees the request's own version and version type.
#[tokio::test]
async fn pipeline_version_metadata_applies_to_the_write() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/ext", json!({"processors": [
        {"set": {"field": "_version", "copy_from": "seq"}},
        {"set": {"field": "_version_type", "value": "external"}},
    ]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/ev/_doc/1?pipeline=ext", json!({"seq": 10, "state": "newer"})).await;
    assert_eq!((st, &v["_version"]), (201, &json!(10)), "{v}");
    // An older event conflicts and leaves version 10 in place.
    let (st, v) = es.put("/ev/_doc/1?pipeline=ext", json!({"seq": 9, "state": "older"})).await;
    assert_eq!(st, 409, "{v}");
    assert_eq!(v["error"]["type"], "version_conflict_engine_exception", "{v}");
    let (_, v) = es.get("/ev/_doc/1").await;
    assert_eq!((&v["_version"], &v["_source"]["state"]), (&json!(10), &json!("newer")), "{v}");
    let (st, v) = es.put("/ev/_doc/1?pipeline=ext", json!({"seq": 11, "state": "newest"})).await;
    assert_eq!((st, &v["_version"]), (200, &json!(11)), "{v}");
    // The same through bulk: per item.
    let v = bulk(&es, &[
        json!({"index": {"_index": "ev", "_id": "1", "pipeline": "ext"}}), json!({"seq": 5, "state": "stale"}),
        json!({"index": {"_index": "ev", "_id": "2", "pipeline": "ext"}}), json!({"seq": 3, "state": "first"}),
    ]).await;
    assert_eq!(v["items"][0]["index"]["status"], 409, "{v}");
    assert_eq!(v["items"][1]["index"]["status"], 201, "{v}");
    assert_eq!(v["items"][1]["index"]["_version"], 3, "{v}");
    let (_, v) = es.get("/ev/_doc/1").await;
    assert_eq!((&v["_version"], &v["_source"]["state"]), (&json!(11), &json!("newest")), "{v}");
    // Pipelines read the request's version and version type.
    es.put("/_ingest/pipeline/seen", json!({"processors": [
        {"set": {"field": "v", "copy_from": "_version"}},
        {"set": {"field": "vt", "copy_from": "_version_type"}},
    ]})).await;
    let (st, v) = es.put("/seen/_doc/1?pipeline=seen&version=7&version_type=external", json!({})).await;
    assert_eq!((st, &v["_version"]), (201, &json!(7)), "{v}");
    assert_eq!(es.source("seen", "1").await, json!({"v": 7, "vt": "external"}));
    es.put("/seen/_doc/2?pipeline=seen", json!({})).await;
    assert_eq!(es.source("seen", "2").await, json!({"v": -3, "vt": "internal"}));
}

/// Sequence-number conditions a pipeline sets apply to the write: a stale
/// `_if_seq_no` conflicts (409) on the index API and in bulk.
#[tokio::test]
async fn pipeline_seq_no_conditions_apply_to_the_write() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/cas/_doc/1", json!({"n": 1})).await;
    assert_eq!(st, 201, "{v}");
    let (seq, term) = (v["_seq_no"].as_i64().unwrap(), v["_primary_term"].as_i64().unwrap());
    let (st, v) = es.put("/_ingest/pipeline/cas", json!({"processors": [
        {"set": {"field": "_if_seq_no", "copy_from": "expect_seq"}},
        {"set": {"field": "_if_primary_term", "copy_from": "expect_term"}},
        {"remove": {"field": ["expect_seq", "expect_term"]}},
    ]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/cas/_doc/1?pipeline=cas", json!({"n": 2, "expect_seq": seq + 5, "expect_term": term})).await;
    assert_eq!(st, 409, "{v}");
    assert_eq!(v["error"]["type"], "version_conflict_engine_exception", "{v}");
    let v = bulk(&es, &[json!({"index": {"_index": "cas", "_id": "1", "pipeline": "cas"}}), json!({"n": 3, "expect_seq": seq + 5, "expect_term": term})]).await;
    assert_eq!(v["items"][0]["index"]["status"], 409, "{v}");
    assert_eq!(es.source("cas", "1").await, json!({"n": 1}));
    // The current sequence number and term pass.
    let (st, v) = es.put("/cas/_doc/1?pipeline=cas", json!({"n": 4, "expect_seq": seq, "expect_term": term})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(es.source("cas", "1").await, json!({"n": 4}));
}

/// `field_access_pattern: flexible` (ES 9 `IngestDocument`, FLEXIBLE):
/// paths match literal dotted keys at any level, writes into a missing path
/// add one dotted key to the deepest existing object, and the pattern is
/// the running pipeline's own (a classic pipeline it calls stays classic).
#[tokio::test]
async fn flexible_field_access_pattern() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let shapes = [
        ("normalized", json!({"a": {"b": {"c": {"d": 0}}}})),
        ("dotted_only", json!({"a.b.c.d": 0})),
        ("split_dots", json!({"a.b": {"c.d": 0}})),
        ("middle_dot", json!({"a": {"b.c": {"d": 0}}})),
    ];
    let flexible = |processors: Value| json!({"field_access_pattern": "flexible", "processors": processors});
    // Writes.
    es.put("/_ingest/pipeline/w", flexible(json!([{"set": {"field": "a.b.c.d", "value": "1"}}]))).await;
    es.put("/fx/_doc/none?pipeline=w", json!({"foo": "bar"})).await;
    assert_eq!(es.source("fx", "none").await, json!({"foo": "bar", "a.b.c.d": "1"}));
    for ((id, doc), want) in shapes.iter().zip([json!({"a": {"b": {"c": {"d": "1"}}}}), json!({"a.b.c.d": "1"}), json!({"a.b": {"c.d": "1"}}), json!({"a": {"b.c": {"d": "1"}}})]) {
        let (st, v) = es.put(&format!("/fx/_doc/w_{id}?pipeline=w"), doc.clone()).await;
        assert!(st == 201, "{id}: {v}");
        assert_eq!(es.source("fx", &format!("w_{id}")).await, want, "{id}");
    }
    // Reads (copy_from), existence (rename without override), removal.
    es.put("/_ingest/pipeline/r", flexible(json!([{"set": {"field": "result", "copy_from": "a.b.c.d"}}]))).await;
    es.put("/_ingest/pipeline/x", flexible(json!([{"rename": {"field": "foo", "target_field": "a.b.c.d", "override": false}}]))).await;
    es.put("/_ingest/pipeline/d", flexible(json!([{"remove": {"field": "a.b.c.d"}}]))).await;
    for (id, doc) in &shapes {
        es.put(&format!("/fx/_doc/r_{id}?pipeline=r"), doc.clone()).await;
        assert_eq!(es.source("fx", &format!("r_{id}")).await["result"], 0, "{id}");
        let mut with_foo = doc.clone();
        with_foo["foo"] = json!("bar");
        let (st, v) = es.put(&format!("/fx/_doc/x_{id}?pipeline=x"), with_foo.clone()).await;
        assert_eq!(st, 400, "{id}: {v}");
        assert_eq!(v["error"]["root_cause"][0]["reason"], "field [a.b.c.d] already exists", "{id}");
        let (st, v) = es.put(&format!("/fx/_doc/d_{id}?pipeline=d"), with_foo).await;
        assert_eq!(st, 201, "{id}: {v}");
        let src = es.source("fx", &format!("d_{id}")).await;
        assert_eq!(src["foo"], "bar", "{id}");
        assert!(!src.to_string().contains("\"d\":0") && !src.to_string().contains(":0"), "{id}: {src}");
    }
    // Missing fields: ignore_missing still skips, without it the dotted
    // key it tried is named.
    es.put("/_ingest/pipeline/m", flexible(json!([{"remove": {"field": "a.x.y", "ignore_missing": true}}, {"set": {"field": "copied", "copy_from": "a.x.y", "ignore_empty_value": true}}]))).await;
    let (st, v) = es.put("/fx_missing/_doc/m?pipeline=m", json!({"a": {"b": 1}})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.source("fx_missing", "m").await, json!({"a": {"b": 1}}));
    // The classic pattern (default) is unchanged: nested objects only.
    es.put("/_ingest/pipeline/c", json!({"processors": [{"set": {"field": "a.b.c.d", "value": "1"}}]})).await;
    es.put("/fx/_doc/classic?pipeline=c", json!({"a.b.c.d": 0})).await;
    assert_eq!(es.source("fx", "classic").await, json!({"a.b.c.d": 0, "a": {"b": {"c": {"d": "1"}}}}));
    // A classic pipeline called from a flexible one keeps its own pattern.
    es.put("/_ingest/pipeline/outer", flexible(json!([{"pipeline": {"name": "c"}}, {"set": {"field": "x.y", "value": 2}}]))).await;
    es.put("/fx/_doc/nested?pipeline=outer", json!({"k": 1})).await;
    assert_eq!(es.source("fx", "nested").await, json!({"k": 1, "a": {"b": {"c": {"d": "1"}}}, "x.y": 2}));
    // Lists cannot be indexed with the flexible pattern.
    es.put("/_ingest/pipeline/l", flexible(json!([{"set": {"field": "arr.0", "value": 1}}]))).await;
    let (st, v) = es.put("/fx/_doc/list?pipeline=l", json!({"arr": [0]})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["root_cause"][0]["reason"], "path [arr.0] is not valid", "{v}");
    // ES's validation of the option.
    let (st, v) = es.put("/_ingest/pipeline/bad", json!({"field_access_pattern": "loose", "processors": []})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "parse_exception", "{v}");
    assert_eq!(v["error"]["reason"], "pipeline [bad] doesn't support value of [loose] for parameter [field_access_pattern]", "{v}");
    // Simulate runs the pattern too.
    let (st, v) = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": flexible(json!([{"set": {"field": "out", "copy_from": "a.b.c.d"}}])), "docs": [{"_source": {"a.b": {"c.d": 5}}}]})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["docs"][0]["doc"]["_source"]["out"], 5, "{v}");
}

/// A PDF whose text is in a TrueType font with private codes that only its
/// `ToUnicode` map decodes (LibreOffice writes these), one paragraph per
/// text object; the stream lengths are indirect objects.
fn tounicode_pdf(paragraphs: &[&str]) -> Vec<u8> {
    let mut chars: Vec<char> = Vec::new();
    for p in paragraphs {
        for c in p.chars() {
            if !chars.contains(&c) {
                chars.push(c);
            }
        }
    }
    let code = |c: char| chars.iter().position(|x| *x == c).unwrap() + 1;
    let mut content = String::new();
    for (i, p) in paragraphs.iter().enumerate() {
        let hex: String = p.chars().map(|c| format!("{:02X}", code(c))).collect();
        content.push_str(&format!("q 0 0 0 rg\nBT\n56.8 {} Td /F1 18 Tf<{hex}>Tj\nET\nQ\n", 769.0 - 63.6 * i as f64));
    }
    let mut cmap = format!("/CIDInit/ProcSet findresource begin\n12 dict begin\nbegincmap\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n{} beginbfchar\n", chars.len());
    for c in &chars {
        cmap.push_str(&format!("<{:02X}> <{:04X}>\n", code(*c), *c as u32));
    }
    cmap.push_str("endbfchar\nendcmap\nend\nend\n");
    let deflate = |s: &str| {
        use std::io::Write;
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(s.as_bytes()).unwrap();
        z.finish().unwrap()
    };
    let (content, cmap) = (deflate(&content), deflate(&cmap));
    let mut out = b"%PDF-1.4\n%\xe4\xfc\xf6\n".to_vec();
    out.extend_from_slice(b"2 0 obj\n<</Length 3 0 R/Filter/FlateDecode>>\nstream\n");
    out.extend_from_slice(&content);
    out.extend_from_slice(format!("\nendstream\nendobj\n3 0 obj\n{}\nendobj\n", content.len()).as_bytes());
    out.extend_from_slice(b"8 0 obj\n<</Length 6 0 R/Filter/FlateDecode>>\nstream\n");
    out.extend_from_slice(&cmap);
    out.extend_from_slice(format!("\nendstream\nendobj\n6 0 obj\n{}\nendobj\n", cmap.len()).as_bytes());
    out.extend_from_slice(b"9 0 obj\n<</Type/Font/Subtype/TrueType/BaseFont/BAAAAA+LiberationSerif/FirstChar 0/LastChar 40/ToUnicode 8 0 R>>\nendobj\n");
    out.extend_from_slice(b"10 0 obj\n<</F1 9 0 R>>\nendobj\n11 0 obj\n<</Font 10 0 R/ProcSet[/PDF/Text]>>\nendobj\n");
    out.extend_from_slice(b"1 0 obj\n<</Type/Page/Parent 4 0 R/Resources 11 0 R/MediaBox[0 0 595 842]/Contents 2 0 R>>\nendobj\n");
    out.extend_from_slice(b"4 0 obj\n<</Type/Pages/Resources 11 0 R/Kids[ 1 0 R ]/Count 1>>\nendobj\n");
    out.extend_from_slice(b"12 0 obj\n<</Type/Catalog/Pages 4 0 R>>\nendobj\n");
    out.extend_from_slice(b"13 0 obj\n<</Creator<FEFF005700720069007400650072>/CreationDate(D:20160930151958+02'00')>>\nendobj\n");
    out.extend_from_slice(b"trailer\n<</Size 14/Root 12 0 R/Info 13 0 R>>\nstartxref\n0\n%%EOF\n");
    out
}

/// The attachment processor extracts the text of PDF, Word (docx and doc)
/// documents, which is then searchable; a document it cannot parse fails
/// with ES's `parse_exception` and keeps its binary (`ignore_failure`),
/// never indexing an attachment without its text.
#[tokio::test]
async fn attachment_extracts_binary_documents_or_fails() {
    use base64::Engine;
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    const DOCX: &[u8] = include_bytes!("../../elasticsearch-engine/src/ingest/attachment_fixtures/sample.docx");
    const DOC: &[u8] = include_bytes!("../../elasticsearch-engine/src/ingest/attachment_fixtures/sample.doc");
    let text = "Rustic attachment fixture\n\nThe quick brown fox jumps over the lazy dog.";
    let pdf = tounicode_pdf(&["Rustic attachment fixture", "The quick brown fox jumps over the lazy dog."]);
    let (st, v) = es.put("/_ingest/pipeline/att", json!({"processors": [{"attachment": {"field": "data", "remove_binary": true}}]})).await;
    assert_eq!(st, 200, "{v}");
    for (id, bytes, content_type) in [
        ("docx", DOCX, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("doc", DOC, "application/msword"),
        ("pdf", &pdf[..], "application/pdf"),
    ] {
        let (st, v) = es.put(&format!("/docs/_doc/{id}?pipeline=att&refresh=true"), json!({"data": b64(bytes)})).await;
        assert_eq!(st, 201, "{id}: {v}");
        let src = es.source("docs", id).await;
        assert_eq!(src["attachment"]["content"], text, "{id}: {src}");
        assert_eq!(src["attachment"]["content_type"], content_type, "{id}: {src}");
        assert!(src.get("data").is_none(), "{id}: the binary is removed only after extraction");
    }
    let src = es.source("docs", "pdf").await;
    assert_eq!((&src["attachment"]["date"], &src["attachment"]["creator_tool"]), (&json!("2016-09-30T13:19:58Z"), &json!("Writer")), "{src}");
    let (st, v) = es.post("/docs/_search", json!({"query": {"match": {"attachment.content": "fox"}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 3, "{v}");
    // Damaged binaries fail the document; nothing is indexed.
    for (id, bytes) in [("bad_pdf", &b"%PDF-1.7\nnot a pdf body"[..]), ("bad_docx", &DOCX[..DOCX.len() / 2]), ("bad_doc", &DOC[..2048])] {
        let (st, v) = es.put(&format!("/docs/_doc/{id}?pipeline=att"), json!({"data": b64(bytes)})).await;
        assert_eq!(st, 400, "{id}: {v}");
        assert_eq!(v["error"]["root_cause"][0]["type"], "parse_exception", "{id}: {v}");
        assert_eq!(v["error"]["root_cause"][0]["reason"], "Error parsing document in field [data]", "{id}: {v}");
        let (st, _) = es.get(&format!("/docs/_doc/{id}")).await;
        assert_eq!(st, 404, "{id}");
    }
    // Bulk: the item fails, its neighbours index.
    let v = bulk(&es, &[
        json!({"index": {"_index": "docs", "_id": "bulk_bad", "pipeline": "att"}}), json!({"data": b64(b"%PDF-1.4 broken")}),
        json!({"index": {"_index": "docs", "_id": "bulk_ok", "pipeline": "att"}}), json!({"data": b64(DOCX)}),
    ]).await;
    assert_eq!(v["items"][0]["index"]["status"], 400, "{v}");
    assert_eq!(v["items"][0]["index"]["error"]["type"], "parse_exception", "{v}");
    assert_eq!(v["items"][1]["index"]["status"], 201, "{v}");
    // `ignore_failure`: the document is indexed with its binary intact and
    // no attachment; `ignore_missing`: a document without the field passes.
    es.put("/_ingest/pipeline/lenient", json!({"processors": [{"attachment": {"field": "data", "remove_binary": true, "ignore_failure": true, "ignore_missing": true}}]})).await;
    let broken = b64(b"%PDF-1.4 broken");
    let (st, v) = es.put("/docs/_doc/kept?pipeline=lenient", json!({"data": broken})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.source("docs", "kept").await, json!({"data": broken}));
    let (st, v) = es.put("/docs/_doc/absent?pipeline=lenient", json!({"other": 1})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.source("docs", "absent").await, json!({"other": 1}));
}

/// A ZIP of stored (uncompressed) entries.
fn stored_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for (name, body) in files {
        let local = out.len() as u32;
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(body);
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&(body.len() as u32).to_le_bytes());
        central.extend_from_slice(&(body.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]);
        central.extend_from_slice(&local.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let at = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&at.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

/// Finding 113: `indexed_chars` bounds the extraction of a workbook whose
/// cells all refer to one long shared string (default, 0, small positive);
/// the text stops there and the document indexes.
#[tokio::test]
async fn attachment_indexed_chars_bounds_shared_string_fan_out() {
    use base64::Engine;
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let shared = format!("<sst><si><t>{}</t></si></sst>", "y".repeat(32_767));
    let row = format!("<row>{}</row>", "<c t=\"s\"><v>0</v></c>".repeat(100));
    let sheet = format!("<worksheet><sheetData>{}</sheetData></worksheet>", row.repeat(200));
    let book = stored_zip(&[
        ("xl/workbook.xml", br#"<workbook><sheets><sheet name="S"/></sheets></workbook>"#),
        ("xl/sharedStrings.xml", shared.as_bytes()),
        ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
    ]);
    let data = base64::engine::general_purpose::STANDARD.encode(&book);
    for (id, limit, want) in [("default", None, 100_000usize), ("zero", Some(0), 0), ("small", Some(25), 25)] {
        let mut processor = json!({"field": "data", "remove_binary": true});
        if let Some(l) = limit {
            processor["indexed_chars"] = json!(l);
        }
        es.put(&format!("/_ingest/pipeline/x_{id}"), json!({"processors": [{"attachment": processor}]})).await;
        let (st, v) = es.put(&format!("/books/_doc/{id}?pipeline=x_{id}"), json!({"data": data})).await;
        assert_eq!(st, 201, "{id}: {v}");
        let src = es.source("books", id).await;
        assert_eq!(src["attachment"]["content_length"], want, "{id}");
        assert_eq!(src["attachment"]["content"].as_str().map_or(0, |c| c.chars().count()), want, "{id}");
        assert_eq!(src["attachment"]["content_type"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "{id}");
    }
}

/// Old finding 73: an extraction that would be incomplete (an archive entry
/// that cannot be read, a PDF page that cannot be decoded) fails the
/// document like an unreadable one, so `remove_binary` never discards a
/// binary whose text was only partly extracted.
#[tokio::test]
async fn attachment_partial_extraction_fails_and_keeps_the_binary() {
    use base64::Engine;
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.put("/_ingest/pipeline/att", json!({"processors": [{"attachment": {"field": "data", "remove_binary": true}}]})).await;
    es.put("/_ingest/pipeline/lenient", json!({"processors": [{"attachment": {"field": "data", "remove_binary": true, "ignore_failure": true}}]})).await;
    // An archive whose second entry has an unsupported compression method.
    let mut zip = stored_zip(&[("a.txt", b"alpha text"), ("b.txt", b"beta text")]);
    // The last central directory record is b.txt's; its method is at +10.
    let b_central = memchr::memmem::rfind(&zip, b"PK\x01\x02").unwrap();
    zip[b_central + 10] = 99;
    // A PDF whose second page's content stream cannot be decoded.
    let pdf = {
        let good = tounicode_pdf(&["readable page"]);
        let mut p = good.clone();
        let at = memchr::memmem::find(&p, b"/Kids[ 1 0 R ]").unwrap();
        p.splice(at..at + 14, b"/Kids[ 1 0 R 20 0 R ]".iter().copied());
        let at = memchr::memmem::find(&p, b"trailer").unwrap();
        p.splice(at..at, b"20 0 obj\n<</Type/Page/Parent 4 0 R/Resources 11 0 R/Contents 21 0 R>>\nendobj\n21 0 obj\n<</Length 8/Filter/LZWDecode>>\nstream\n\x80\x0b\x60\x50\x22\x0c\x0c\x85\nendstream\nendobj\n".iter().copied());
        p
    };
    for (id, bytes) in [("partial_zip", &zip[..]), ("partial_pdf", &pdf[..])] {
        let (st, v) = es.put(&format!("/parts/_doc/{id}?pipeline=att"), json!({"data": b64(bytes)})).await;
        assert_eq!(st, 400, "{id}: {v}");
        assert_eq!(v["error"]["root_cause"][0]["type"], "parse_exception", "{id}: {v}");
        assert_eq!(v["error"]["root_cause"][0]["reason"], "Error parsing document in field [data]", "{id}: {v}");
        let (st, _) = es.get(&format!("/parts/_doc/{id}")).await;
        assert_eq!(st, 404, "{id}");
        // With ignore_failure the document indexes with its binary intact.
        let (st, v) = es.put(&format!("/parts/_doc/{id}_kept?pipeline=lenient"), json!({"data": b64(bytes)})).await;
        assert_eq!(st, 201, "{id}: {v}");
        assert_eq!(es.source("parts", &format!("{id}_kept")).await, json!({"data": b64(bytes)}), "{id}");
    }
    // The same document without the unreadable part extracts in full.
    let whole = stored_zip(&[("a.txt", b"alpha text"), ("b.txt", b"beta text")]);
    let (st, v) = es.put("/parts/_doc/whole?pipeline=att", json!({"data": b64(&whole)})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.source("parts", "whole").await["attachment"]["content"], "a.txt\nalpha text\nb.txt\nbeta text");
}

/// Old review finding 72: a reindex's destination pipeline sees the
/// destination request's version conditions and the ones it sets apply to
/// the write (ES `updateIndexRequestMetadata`), as for the index API.
#[tokio::test]
async fn reindex_applies_pipeline_write_conditions() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/ver", json!({"processors": [
        {"set": {"field": "_version", "value": 7}},
        {"set": {"field": "_version_type", "value": "external"}}
    ]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = es.put("/src/_doc/1?refresh=true", json!({"n": 1})).await;
    assert_eq!(st, 201);
    let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": "src"}, "dest": {"index": "dst", "pipeline": "ver"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["created"], 1, "{v}");
    let (st, v) = es.get("/dst/_doc/1").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_version"], 7, "the pipeline's external version applies: {v}");
    // A second run with the same external version is a conflict (as the
    // index API with that pipeline would be), not a silent overwrite.
    let (st, v) = es.post("/_reindex?refresh=true", json!({"conflicts": "proceed", "source": {"index": "src"}, "dest": {"index": "dst", "pipeline": "ver"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["version_conflicts"], 1, "{v}");
}
