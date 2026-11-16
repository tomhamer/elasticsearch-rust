//! End-to-end tests: real server on 127.0.0.1:0, reqwest client, exact ES
//! response shapes.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method, StatusCode};
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
    async fn raw(&self, m: Method, path: &str, body: Option<String>, ct: &str) -> (StatusCode, reqwest::header::HeaderMap, String) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path));
        if let Some(b) = body {
            r = r.header("content-type", ct).body(b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status();
        let h = resp.headers().clone();
        (st, h, resp.text().await.unwrap())
    }
    async fn req(&self, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let (st, _, text) = self.raw(m, path, body.map(|b| b.to_string()), "application/json").await;
        let v = if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) };
        (st.as_u16(), v)
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
    /// POST without a body (ES rejects a body on e.g. `_close`, `_refresh`).
    async fn post0(&self, p: &str) -> (u16, Value) {
        self.req(Method::POST, p, None).await
    }
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.req(Method::DELETE, p, None).await
    }
    async fn ndjson(&self, p: &str, lines: &[Value]) -> (u16, Value) {
        let mut s = String::new();
        for l in lines {
            s.push_str(&l.to_string());
            s.push('\n');
        }
        let (st, _, t) = self.raw(Method::POST, p, Some(s), "application/x-ndjson").await;
        (st.as_u16(), serde_json::from_str(&t).unwrap())
    }
    async fn search(&self, idx: &str, b: Value) -> Value {
        let (st, v) = self.post(&format!("/{idx}/_search"), b).await;
        assert_eq!(st, 200, "{v}");
        v
    }
}

fn ids(v: &Value) -> Vec<String> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

async fn seed(es: &Es) {
    let (st, _) = es
        .put(
            "/books",
            json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {
                "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "author": {"type": "keyword"},
                "year": {"type": "integer"},
                "price": {"type": "double"},
                "published": {"type": "date"},
                "tags": {"type": "keyword"},
                "vec": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm"}
            }}}),
        )
        .await;
    assert_eq!(st, 200);
    let docs = [
        ("1", json!({"title": "The quick brown fox", "author": "alice", "year": 2001, "price": 10.5, "published": "2001-05-01", "tags": ["animals", "classic"], "vec": [1.0, 0.0, 0.0]})),
        ("2", json!({"title": "Quick recipes for busy people", "author": "bob", "year": 2010, "price": 25.0, "published": "2010-01-15", "tags": ["cooking"], "vec": [0.0, 1.0, 0.0]})),
        ("3", json!({"title": "The lazy dog sleeps", "author": "alice", "year": 2015, "price": 7.25, "published": "2015-07-20", "tags": ["animals"], "vec": [0.9, 0.1, 0.0]})),
        ("4", json!({"title": "Brown bread baking", "author": "carol", "year": 2020, "price": 30.0, "published": "2020-03-03", "tags": ["cooking", "bread"], "vec": [0.0, 0.0, 1.0]})),
    ];
    let mut lines = Vec::new();
    for (id, d) in docs {
        lines.push(json!({"index": {"_index": "books", "_id": id}}));
        lines.push(d);
    }
    let (st, r) = es.ndjson("/_bulk?refresh=true", &lines).await;
    assert_eq!(st, 200);
    assert_eq!(r["errors"], false);
}

#[tokio::test(flavor = "multi_thread")]
async fn root_info_and_headers() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (st, h, body) = es.raw(Method::GET, "/", None, "").await;
    assert_eq!(st, 200);
    assert_eq!(h["x-elastic-product"], "Elasticsearch");
    let v: Value = serde_json::from_str(&body).unwrap();
    // ES 9.5.5 reports its own version whatever `compatible-with` asks for.
    assert_eq!(v["version"]["number"], "9.5.5");
    assert_eq!(v["version"]["minimum_wire_compatibility_version"], "8.19.0");
    assert_eq!(v["version"]["minimum_index_compatibility_version"], "8.0.0");
    assert_eq!(v["version"]["build_flavor"], "default");
    assert_eq!(v["tagline"], "You Know, for Search");
    // `compatible-with=8` still reports 9.5.5.
    let resp = es.client.get(format!("{}/", es.base)).header("accept", "application/vnd.elasticsearch+json;compatible-with=8").send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let v8: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v8["version"]["number"], "9.5.5");
    for k in ["name", "cluster_name", "cluster_uuid"] {
        assert!(v[k].is_string());
    }
    // HEAD: no body, header present.
    let (st, h, body) = es.raw(Method::HEAD, "/", None, "").await;
    assert_eq!(st, 200);
    assert!(body.is_empty());
    assert_eq!(h["x-elastic-product"], "Elasticsearch");
    // 404 also carries the header.
    let (st, h, _) = es.raw(Method::GET, "/nope/_doc/1", None, "").await;
    assert_eq!(st, 404);
    assert_eq!(h["x-elastic-product"], "Elasticsearch");
    // Unknown endpoint.
    let (st, v) = es.get("/_nonexistent_api").await;
    assert_eq!(st, 400);
    assert_eq!(v, json!({"error": "no handler found for uri [/_nonexistent_api] and method [GET]"}));
    // Vendor media type echoed; bad compatible-with rejected.
    let r = es.client.get(format!("{}/", es.base)).header("accept", "application/vnd.elasticsearch+json;compatible-with=8").send().await.unwrap();
    assert_eq!(r.headers()["content-type"], "application/vnd.elasticsearch+json;compatible-with=8");
    let r = es.client.get(format!("{}/", es.base)).header("accept", "application/vnd.elasticsearch+json;compatible-with=99").send().await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["type"], "media_type_header_exception");
}

#[tokio::test(flavor = "multi_thread")]
async fn index_lifecycle_and_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (st, v) = es.put("/idx", json!({"mappings": {"properties": {"n": {"type": "long"}}}})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true, "shards_acknowledged": true, "index": "idx"})));
    let (st, v) = es.put("/idx", json!({})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "resource_already_exists_exception");
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("index [idx/"));
    assert_eq!(v["error"]["root_cause"][0]["type"], "resource_already_exists_exception");
    assert_eq!(v["status"], 400);
    let (st, v) = es.get("/missing").await;
    assert_eq!(st, 404);
    assert_eq!(
        v,
        json!({"error": {"root_cause": [{"type": "index_not_found_exception", "reason": "no such index [missing]", "resource.type": "index_or_alias", "resource.id": "missing", "index_uuid": "_na_", "index": "missing"}],
            "type": "index_not_found_exception", "reason": "no such index [missing]", "resource.type": "index_or_alias", "resource.id": "missing", "index_uuid": "_na_", "index": "missing"}, "status": 404})
    );
    let (st, v) = es.put("/BadName", json!({})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "invalid_index_name_exception");
    let (st, v) = es.get("/idx").await;
    assert_eq!(st, 200);
    assert_eq!(v["idx"]["mappings"]["properties"]["n"]["type"], "long");
    assert_eq!(v["idx"]["settings"]["index"]["number_of_shards"], "1");
    assert!(v["idx"]["aliases"].is_object());
    let (st, _, b) = es.raw(Method::HEAD, "/idx", None, "").await;
    assert_eq!((st.as_u16(), b.as_str()), (200, ""));
    let (st, _, _) = es.raw(Method::HEAD, "/nope", None, "").await;
    assert_eq!(st.as_u16(), 404);
    // mapping put/get + conflict
    let (st, v) = es.put("/idx/_mapping", json!({"properties": {"t": {"type": "text"}}})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (_, v) = es.get("/idx/_mapping").await;
    assert_eq!(v, json!({"idx": {"mappings": {"properties": {"n": {"type": "long"}, "t": {"type": "text"}}}}}));
    let (st, v) = es.put("/idx/_mapping", json!({"properties": {"n": {"type": "keyword"}}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "illegal_argument_exception");
    assert_eq!(v["error"]["reason"], "mapper [n] cannot be changed from type [long] to [keyword]");
    let (st, v) = es.put("/bad", json!({"mappings": {"properties": {"x": {"type": "nope"}}}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "mapper_parsing_exception");
    // settings rules
    let (st, _) = es.put("/idx/_settings", json!({"index": {"refresh_interval": "5s"}})).await;
    assert_eq!(st, 200);
    let (_, v) = es.get("/idx/_settings?flat_settings=true").await;
    assert_eq!(v["idx"]["settings"]["index.refresh_interval"], "5s");
    let (st, v) = es.put("/idx/_settings", json!({"index": {"number_of_shards": 3}})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("Can't update non dynamic settings [[index.number_of_shards]] for open indices [[idx/"));
    // delete: wildcard blocked (action.destructive_requires_name defaults to
    // true, as in ES 8), name ok
    let (st, v) = es.delete("/id*").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "Wildcard expressions or all indices are not allowed");
    let (st, v) = es.delete("/idx").await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (st, _) = es.delete("/idx").await;
    assert_eq!(st, 404);
}

#[tokio::test(flavor = "multi_thread")]
async fn document_crud_versions_conflicts() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (st, v) = es.put("/docs/_doc/1?refresh=true", json!({"a": 1})).await;
    assert_eq!(st, 201);
    assert_eq!(
        v,
        json!({"_index": "docs", "_id": "1", "_version": 1, "result": "created", "forced_refresh": true, "_shards": {"total": 2, "successful": 1, "failed": 0}, "_seq_no": 0, "_primary_term": 1})
    );
    let (st, v) = es.put("/docs/_doc/1", json!({"a": 2, "b": "x"})).await;
    assert_eq!(st, 200);
    assert_eq!((v["result"].clone(), v["_version"].clone(), v["_seq_no"].clone()), (json!("updated"), json!(2), json!(1)));
    let (st, v) = es.get("/docs/_doc/1").await;
    assert_eq!(st, 200);
    assert_eq!(v, json!({"_index": "docs", "_id": "1", "_version": 2, "_seq_no": 1, "_primary_term": 1, "found": true, "_source": {"a": 2, "b": "x"}}));
    let (_, v) = es.get("/docs/_doc/1?_source_includes=b").await;
    assert_eq!(v["_source"], json!({"b": "x"}));
    let (_, v) = es.get("/docs/_source/1").await;
    assert_eq!(v, json!({"a": 2, "b": "x"}));
    let (st, v) = es.get("/docs/_doc/nope").await;
    assert_eq!((st, v), (404, json!({"_index": "docs", "_id": "nope", "found": false})));
    // op_type create conflict
    let (st, v) = es.put("/docs/_create/1", json!({"a": 3})).await;
    assert_eq!(st, 409);
    assert_eq!(v["error"]["type"], "version_conflict_engine_exception");
    assert_eq!(v["error"]["reason"], "[1]: version conflict, document already exists (current version [2])");
    assert_eq!(v["error"]["shard"], "0");
    assert_eq!(v["error"]["index"], "docs");
    // if_seq_no
    let (st, v) = es.put("/docs/_doc/1?if_seq_no=0&if_primary_term=1", json!({"a": 4})).await;
    assert_eq!(st, 409);
    assert_eq!(v["error"]["reason"], "[1]: version conflict, required seqNo [0], primary term [1]. current document has seqNo [1] and primary term [1]");
    let (st, _) = es.put("/docs/_doc/1?if_seq_no=1&if_primary_term=1", json!({"a": 4})).await;
    assert_eq!(st, 200);
    // external versions
    let (st, v) = es.put("/docs/_doc/ext?version=5&version_type=external", json!({"a": 1})).await;
    assert_eq!((st, v["_version"].clone()), (201, json!(5)));
    let (st, v) = es.put("/docs/_doc/ext?version=5&version_type=external", json!({"a": 1})).await;
    assert_eq!(st, 409);
    assert_eq!(v["error"]["reason"], "[ext]: version conflict, current version [5] is higher or equal to the one provided [5]");
    // auto id
    let (st, v) = es.post("/docs/_doc", json!({"auto": true})).await;
    assert_eq!(st, 201);
    assert_eq!(v["_id"].as_str().unwrap().len(), 20);
    // update
    let (st, v) = es.post("/docs/_update/1", json!({"doc": {"c": true}})).await;
    assert_eq!((st, v["result"].clone()), (200, json!("updated")));
    let (_, v) = es.post("/docs/_update/1", json!({"doc": {"c": true}})).await;
    assert_eq!(v["result"], "noop");
    let (st, v) = es.post("/docs/_update/u1", json!({"doc": {"x": 1}, "doc_as_upsert": true})).await;
    assert_eq!((st, v["result"].clone()), (201, json!("created")));
    let (st, v) = es.post("/docs/_update/u2", json!({"doc": {"x": 1}, "upsert": {"x": 0}})).await;
    assert_eq!(st, 201);
    let _ = v;
    let (_, v) = es.get("/docs/_source/u2").await;
    assert_eq!(v, json!({"x": 0}));
    let (st, v) = es.post("/docs/_update/missing", json!({"doc": {"x": 1}})).await;
    assert_eq!(st, 404);
    assert_eq!(v["error"]["type"], "document_missing_exception");
    assert_eq!(v["error"]["reason"], "[missing]: document missing");
    let (_, v) = es.post("/docs/_update/1?_source=true", json!({"doc": {"d": 1}})).await;
    assert_eq!(v["get"]["found"], true);
    assert_eq!(v["get"]["_source"]["d"], 1);
    // delete
    let (st, v) = es.delete("/docs/_doc/1").await;
    assert_eq!((st, v["result"].clone()), (200, json!("deleted")));
    let (st, v) = es.delete("/docs/_doc/1").await;
    assert_eq!((st, v["result"].clone()), (404, json!("not_found")));
    // mget
    let (st, v) = es.post("/_mget", json!({"docs": [{"_index": "docs", "_id": "ext"}, {"_index": "docs", "_id": "zzz"}, {"_index": "nope", "_id": "1"}]})).await;
    assert_eq!(st, 200);
    assert_eq!(v["docs"][0]["found"], true);
    assert_eq!(v["docs"][1], json!({"_index": "docs", "_id": "zzz", "found": false}));
    assert_eq!(v["docs"][2]["error"]["type"], "index_not_found_exception");
    let (_, v) = es.post("/docs/_mget", json!({"ids": ["ext", "u1"]})).await;
    assert_eq!(v["docs"].as_array().unwrap().len(), 2);
    // document parsing error shape
    let (st, v) = es.put("/typed", json!({"mappings": {"properties": {"n": {"type": "integer"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/typed/_doc/1", json!({"n": "abc"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "document_parsing_exception");
    assert!(v["error"]["reason"].as_str().unwrap().contains("failed to parse field [n] of type [integer] in document with id '1'"));
}

#[tokio::test(flavor = "multi_thread")]
async fn bulk_mixed_items() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (st, v) = es
        .ndjson(
            "/b/_bulk",
            &[
                json!({"index": {"_id": "1"}}),
                json!({"n": 1}),
                json!({"create": {"_id": "1"}}),
                json!({"n": 2}),
                json!({"update": {"_id": "1"}}),
                json!({"doc": {"m": 3}}),
                json!({"delete": {"_id": "2"}}),
                json!({"index": {"_index": "b2", "_id": "x"}}),
                json!({"n": 1}),
                json!({"update": {"_id": "nope"}}),
                json!({"doc": {"m": 3}}),
                json!({"delete": {"_id": "1"}}),
            ],
        )
        .await;
    assert_eq!(st, 200);
    assert_eq!(v["errors"], true);
    assert!(v["took"].is_number());
    let items = v["items"].as_array().unwrap();
    assert_eq!(items.len(), 7);
    assert_eq!(items[0]["index"]["status"], 201);
    assert_eq!(items[0]["index"]["result"], "created");
    assert_eq!(items[0]["index"]["_shards"], json!({"total": 2, "successful": 1, "failed": 0}));
    assert_eq!(items[1]["create"]["status"], 409);
    assert_eq!(items[1]["create"]["error"]["type"], "version_conflict_engine_exception");
    assert_eq!(items[1]["create"]["_id"], "1");
    assert_eq!(items[2]["update"]["status"], 200);
    assert_eq!(items[2]["update"]["result"], "updated");
    assert_eq!(items[3]["delete"]["status"], 404);
    assert_eq!(items[3]["delete"]["result"], "not_found");
    assert_eq!(items[4]["index"]["_index"], "b2");
    assert_eq!(items[5]["update"]["status"], 404);
    assert_eq!(items[5]["update"]["error"]["type"], "document_missing_exception");
    assert_eq!(items[6]["delete"]["result"], "deleted");
    // malformed action line → whole request 400
    let (st, _, t) = es.raw(Method::POST, "/_bulk", Some("\"bad\"\n".into()), "application/x-ndjson").await;
    assert_eq!(st.as_u16(), 400);
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["error"]["type"], "illegal_argument_exception");
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("Malformed action/metadata line [1]"));
}

#[tokio::test(flavor = "multi_thread")]
async fn search_query_dsl() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    let v = es.search("books", json!({"query": {"match": {"title": "quick"}}})).await;
    assert_eq!(v["hits"]["total"], json!({"value": 2, "relation": "eq"}));
    assert_eq!(v["timed_out"], false);
    assert_eq!(v["_shards"], json!({"total": 1, "successful": 1, "skipped": 0, "failed": 0}));
    assert!(v["hits"]["max_score"].as_f64().unwrap() > 0.0);
    let h = &v["hits"]["hits"][0];
    for k in ["_index", "_id", "_score", "_source"] {
        assert!(!h[k].is_null(), "{k}");
    }
    let mut got = ids(&v);
    got.sort();
    assert_eq!(got, vec!["1", "2"]);
    let q = |b: Value| es.search("books", b);
    let set = |v: Value| {
        let mut x = ids(&v);
        x.sort();
        x
    };
    assert_eq!(set(q(json!({"query": {"match": {"title": {"query": "quick brown", "operator": "and"}}}})).await), vec!["1"]);
    assert_eq!(set(q(json!({"query": {"match_phrase": {"title": "lazy dog"}}})).await), vec!["3"]);
    assert_eq!(set(q(json!({"query": {"match_phrase": {"title": {"query": "quick fox", "slop": 1}}}})).await), vec!["1"]);
    assert_eq!(set(q(json!({"query": {"match_phrase_prefix": {"title": "brown br"}}})).await), vec!["4"]);
    assert_eq!(set(q(json!({"query": {"term": {"author": "alice"}}})).await), vec!["1", "3"]);
    assert_eq!(set(q(json!({"query": {"terms": {"tags": ["bread", "classic"]}}})).await), vec!["1", "4"]);
    assert_eq!(set(q(json!({"query": {"range": {"year": {"gte": 2010, "lt": 2020}}}})).await), vec!["2", "3"]);
    assert_eq!(set(q(json!({"query": {"range": {"published": {"gte": "2015-01-01"}}}})).await), vec!["3", "4"]);
    assert_eq!(set(q(json!({"query": {"bool": {"must": [{"match": {"title": "brown"}}], "filter": [{"term": {"author": "carol"}}]}}})).await), vec!["4"]);
    assert_eq!(set(q(json!({"query": {"bool": {"must_not": [{"term": {"author": "alice"}}]}}})).await), vec!["2", "4"]);
    assert_eq!(set(q(json!({"query": {"bool": {"should": [{"term": {"author": "bob"}}, {"term": {"author": "carol"}}], "minimum_should_match": 1}}})).await), vec!["2", "4"]);
    assert_eq!(set(q(json!({"query": {"multi_match": {"query": "alice", "fields": ["title", "author"]}}})).await), vec!["1", "3"]);
    assert_eq!(set(q(json!({"query": {"query_string": {"query": "title:quick AND author:bob"}}})).await), vec!["2"]);
    assert_eq!(set(q(json!({"query": {"query_string": {"query": "year:[2011 TO *]"}}})).await), vec!["3", "4"]);
    assert_eq!(set(q(json!({"query": {"simple_query_string": {"query": "brown -bread", "fields": ["title"], "default_operator": "and"}}})).await), vec!["1"]);
    assert_eq!(set(q(json!({"query": {"prefix": {"author": "ca"}}})).await), vec!["4"]);
    assert_eq!(set(q(json!({"query": {"wildcard": {"author": {"value": "a*e"}}}})).await), vec!["1", "3"]);
    assert_eq!(set(q(json!({"query": {"exists": {"field": "tags"}}})).await).len(), 4);
    assert_eq!(set(q(json!({"query": {"ids": {"values": ["2", "4", "9"]}}})).await), vec!["2", "4"]);
    assert_eq!(set(q(json!({"query": {"constant_score": {"filter": {"term": {"author": "bob"}}, "boost": 2.5}}})).await), vec!["2"]);
    let v = q(json!({"query": {"constant_score": {"filter": {"term": {"author": "bob"}}, "boost": 2.5}}})).await;
    assert_eq!(v["hits"]["hits"][0]["_score"], 2.5);
    let v = q(json!({"knn": {"field": "vec", "query_vector": [1.0, 0.0, 0.0], "k": 2, "num_candidates": 10}})).await;
    assert_eq!(ids(&v), vec!["1", "3"]);
    let v = q(json!({"query": {"knn": {"field": "vec", "query_vector": [0.0, 0.0, 1.0], "k": 1}}})).await;
    assert_eq!(ids(&v)[0], "4");
    // URI search
    let (_, v) = es.get("/books/_search?q=author:bob").await;
    assert_eq!(ids(&v), vec!["2"]);
    // errors
    let (st, v) = es.post("/books/_search", json!({"query": {"nonsense": {}}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "parsing_exception");
    assert_eq!(v["error"]["reason"], "unknown query [nonsense]");
    let (st, v) = es.post("/books/_search", json!({"query": {"term": {"year": "notanumber"}}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "search_phase_execution_exception");
    assert_eq!(v["error"]["root_cause"][0]["type"], "query_shard_exception");
    assert_eq!(v["error"]["failed_shards"][0]["index"], "books");
    let (st, v) = es.post("/books/_search", json!({"from": 9990, "size": 20})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["root_cause"][0]["reason"].as_str().unwrap().starts_with("Result window is too large"));
    let (st, v) = es.post("/books/_search", json!({"bogus": 1})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "parsing_exception");
}

#[tokio::test(flavor = "multi_thread")]
async fn search_features() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    // sort + search_after paging
    let v = es.search("books", json!({"sort": [{"year": "desc"}], "size": 2})).await;
    assert_eq!(ids(&v), vec!["4", "3"]);
    assert_eq!(v["hits"]["hits"][0]["sort"], json!([2020]));
    assert!(v["hits"]["hits"][0]["_score"].is_null());
    assert!(v["hits"]["max_score"].is_null());
    let after = v["hits"]["hits"][1]["sort"].clone();
    let v = es.search("books", json!({"sort": [{"year": "desc"}], "size": 2, "search_after": after})).await;
    assert_eq!(ids(&v), vec!["2", "1"]);
    let v = es.search("books", json!({"sort": [{"author": "asc"}, {"price": {"order": "desc"}}]})).await;
    assert_eq!(ids(&v), vec!["1", "3", "2", "4"]);
    let v = es.search("books", json!({"sort": ["published"], "size": 1})).await;
    assert_eq!(v["hits"]["hits"][0]["sort"], json!([988675200000i64]));
    // from/size
    let v = es.search("books", json!({"sort": [{"year": "asc"}], "from": 1, "size": 2})).await;
    assert_eq!(ids(&v), vec!["2", "3"]);
    // _source filtering, fields, docvalue_fields
    let v = es.search("books", json!({"query": {"ids": {"values": ["1"]}}, "_source": ["author", "year"]})).await;
    assert_eq!(v["hits"]["hits"][0]["_source"], json!({"author": "alice", "year": 2001}));
    let v = es.search("books", json!({"query": {"ids": {"values": ["1"]}}, "_source": false, "fields": ["title.raw", "year", {"field": "published", "format": "yyyy"}]})).await;
    assert!(v["hits"]["hits"][0].get("_source").is_none());
    assert_eq!(v["hits"]["hits"][0]["fields"]["title.raw"], json!(["The quick brown fox"]));
    assert_eq!(v["hits"]["hits"][0]["fields"]["year"], json!([2001]));
    assert_eq!(v["hits"]["hits"][0]["fields"]["published"], json!(["2001"]));
    let v = es.search("books", json!({"query": {"ids": {"values": ["1"]}}, "docvalue_fields": ["tags", "price"]})).await;
    assert_eq!(v["hits"]["hits"][0]["fields"]["tags"], json!(["animals", "classic"]));
    assert_eq!(v["hits"]["hits"][0]["fields"]["price"], json!([10.5]));
    // highlight
    let v = es.search("books", json!({"query": {"match": {"title": "fox"}}, "highlight": {"fields": {"title": {}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["title"], json!(["The quick brown <em>fox</em>"]));
    // track_total_hits
    let v = es.search("books", json!({"track_total_hits": 2})).await;
    assert_eq!(v["hits"]["total"], json!({"value": 2, "relation": "gte"}));
    let v = es.search("books", json!({"track_total_hits": false})).await;
    assert!(v["hits"].get("total").is_none());
    let (_, v) = es.post("/books/_search?rest_total_hits_as_int=true", json!({})).await;
    assert_eq!(v["hits"]["total"], 4);
    // post_filter
    let v = es.search("books", json!({"query": {"match_all": {}}, "post_filter": {"term": {"author": "bob"}}})).await;
    assert_eq!(ids(&v), vec!["2"]);
    // min_score
    let v = es.search("books", json!({"query": {"match": {"title": "brown"}}, "min_score": 100.0})).await;
    assert_eq!(v["hits"]["total"]["value"], 0);
    // explain, version, seq_no
    let v = es.search("books", json!({"query": {"term": {"author": "bob"}}, "explain": true, "version": true, "seq_no_primary_term": true})).await;
    let h = &v["hits"]["hits"][0];
    assert!(h["_explanation"]["value"].is_number());
    assert!(h["_explanation"]["description"].is_string());
    assert_eq!(h["_version"], 1);
    assert_eq!(h["_primary_term"], 1);
    assert!(h["_seq_no"].is_number());
    // count
    let (st, v) = es.post("/books/_count", json!({"query": {"term": {"author": "alice"}}})).await;
    assert_eq!((st, v), (200, json!({"count": 2, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}})));
    let (_, v) = es.get("/_count").await;
    assert_eq!(v["count"], 4);
    // msearch
    let (st, v) = es.ndjson("/_msearch", &[json!({"index": "books"}), json!({"query": {"term": {"author": "bob"}}}), json!({"index": "nope"}), json!({})]).await;
    assert_eq!(st, 200);
    assert_eq!(v["responses"][0]["status"], 200);
    assert_eq!(v["responses"][0]["hits"]["total"]["value"], 1);
    assert_eq!(v["responses"][1]["status"], 404);
    assert_eq!(v["responses"][1]["error"]["type"], "index_not_found_exception");
    // filter_path & pretty
    let (_, _, t) = es.raw(Method::GET, "/books/_search?filter_path=hits.hits._id&size=1&sort=year", None, "").await;
    assert_eq!(serde_json::from_str::<Value>(&t).unwrap(), json!({"hits": {"hits": [{"_id": "1"}]}}));
    let (_, _, t) = es.raw(Method::GET, "/books/_count?pretty", None, "").await;
    assert!(t.ends_with("}\n") && t.contains("\n  \"count\""));
}

#[tokio::test(flavor = "multi_thread")]
async fn aggregations_over_http() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    let v = es
        .search(
            "books",
            json!({"size": 0, "aggs": {"by_author": {"terms": {"field": "author"}, "aggs": {"avg_price": {"avg": {"field": "price"}}}}, "years": {"date_histogram": {"field": "published", "calendar_interval": "year", "min_doc_count": 1}}}}),
        )
        .await;
    let b = &v["aggregations"]["by_author"]["buckets"];
    assert_eq!(b[0]["key"], "alice");
    assert_eq!(b[0]["doc_count"], 2);
    assert_eq!(b[0]["avg_price"]["value"], 8.875);
    assert_eq!(v["aggregations"]["by_author"]["sum_other_doc_count"], 0);
    assert_eq!(v["aggregations"]["years"]["buckets"].as_array().unwrap().len(), 4);
    assert_eq!(v["hits"]["hits"], json!([]));
    let (_, v) = es.post("/books/_search?typed_keys=true", json!({"size": 0, "aggs": {"a": {"avg": {"field": "price"}}}})).await;
    assert!(v["aggregations"]["avg#a"]["value"].is_number());
}

#[tokio::test(flavor = "multi_thread")]
async fn scroll_and_pit() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/books/_search?scroll=1m", json!({"size": 3, "sort": ["_doc"]})).await;
    assert_eq!(st, 200, "{v}");
    let sid = v["_scroll_id"].as_str().unwrap().to_string();
    let mut seen = ids(&v);
    assert_eq!(seen.len(), 3);
    let (_, v) = es.post("/_search/scroll", json!({"scroll": "1m", "scroll_id": sid})).await;
    seen.extend(ids(&v));
    let (_, v) = es.post("/_search/scroll", json!({"scroll": "1m", "scroll_id": sid})).await;
    assert!(ids(&v).is_empty());
    seen.sort();
    assert_eq!(seen, vec!["1", "2", "3", "4"]);
    let (st, v) = es.req(Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": sid}))).await;
    assert_eq!((st, v), (200, json!({"succeeded": true, "num_freed": 1})));
    let (st, v) = es.post("/_search/scroll", json!({"scroll_id": sid})).await;
    assert_eq!(st, 404, "{v}");
    // PIT
    let (st, v) = es.post("/books/_pit?keep_alive=1m", json!({})).await;
    assert_eq!(st, 200);
    let pit = v["id"].as_str().unwrap().to_string();
    // A doc added after the PIT is not visible through it.
    es.put("/books/_doc/5?refresh=true", json!({"title": "new", "year": 2024})).await;
    let (_, v) = es.post("/_search", json!({"pit": {"id": pit, "keep_alive": "1m"}, "size": 2, "sort": [{"year": "asc"}]})).await;
    assert_eq!(v["pit_id"], pit);
    assert_eq!(ids(&v), vec!["1", "2"]);
    let after = v["hits"]["hits"][1]["sort"].clone();
    assert_eq!(after.as_array().unwrap().len(), 2, "implicit _shard_doc tiebreaker");
    let (_, v) = es.post("/_search", json!({"pit": {"id": pit}, "size": 5, "sort": [{"year": "asc"}], "search_after": after})).await;
    assert_eq!(ids(&v), vec!["3", "4"]);
    let (st, v) = es.req(Method::DELETE, "/_pit", Some(json!({"id": pit}))).await;
    assert_eq!((st, v), (200, json!({"succeeded": true, "num_freed": 1})));
}

/// A scroll cursor advances for sorted and unsorted requests; the page after the last hit is empty.
#[tokio::test(flavor = "multi_thread")]
async fn scroll_totals_are_exact_above_default_limit_on_every_page() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (status, response) = es.put("/scroll-exact", json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"n": {"type": "long"}}}})).await;
    assert_eq!(status, 200, "{response}");
    const TOTAL: usize = 10_007;
    let lines: Vec<Value> = (0..TOTAL).flat_map(|i| [json!({"index": {"_id": i.to_string()}}), json!({"n": i})]).collect();
    let (status, response) = es.ndjson("/scroll-exact/_bulk?refresh=true", &lines).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["errors"], false, "{response}");
    let normal = es.search("scroll-exact", json!({"size": 0})).await;
    assert_eq!(normal["hits"]["total"], json!({"value": 10_000, "relation": "gte"}));
    let (status, mut response) = es.post("/scroll-exact/_search?scroll=1m&sort=_doc&size=4096", json!({})).await;
    assert_eq!(status, 200, "{response}");
    let mut id = response["_scroll_id"].as_str().unwrap().to_owned();
    let expected = json!({"value": TOTAL, "relation": "eq"});
    let mut seen = std::collections::HashSet::new();
    // The original snapshot's exact total must survive refreshes and paging.
    let (status, added) = es.put("/scroll-exact/_doc/later?refresh=true", json!({"n": TOTAL})).await;
    assert_eq!(status, 201, "{added}");
    let mut finished = false;
    for _ in 0..5 {
        assert_eq!(response["hits"]["total"], expected, "{response}");
        let page = ids(&response);
        if page.is_empty() { finished = true; break; }
        for hit in page { assert!(seen.insert(hit), "scroll returned a duplicate document"); }
        let (status, next) = es.post("/_search/scroll", json!({"scroll": "1m", "scroll_id": id})).await;
        assert_eq!(status, 200, "{next}");
        id = next["_scroll_id"].as_str().unwrap().to_owned();
        response = next;
    }
    assert!(finished, "scroll must reach an empty page");
    assert_eq!(seen.len(), TOTAL);
    let (status, response) = es.post("/_search/scroll?rest_total_hits_as_int=true", json!({"scroll": "1m", "scroll_id": id})).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["hits"]["total"], TOTAL);
    assert!(ids(&response).is_empty());
    es.req(Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": id}))).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn scroll_total_validation_respects_url_override_over_http() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    for (body, suffix, accepted) in [
        (json!({}), "&track_total_hits=false", false),
        (json!({}), "&track_total_hits=10000", false),
        (json!({}), "&track_total_hits=-1", false),
        (json!({"track_total_hits": true}), "&track_total_hits=1", false),
        (json!({"track_total_hits": false}), "&track_total_hits=true", true),
        (json!({"track_total_hits": 1}), "&track_total_hits=2147483647", true),
    ] {
        let (status, response) = es.post(&format!("/books/_search?scroll=1m{suffix}"), body).await;
        if accepted {
            assert_eq!(status, 200, "{suffix}: {response}");
            assert_eq!(response["hits"]["total"], json!({"value": 4, "relation": "eq"}));
            es.req(Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": response["_scroll_id"]}))).await;
        } else {
            assert_eq!(status, 400, "{suffix}: {response}");
            assert_eq!(response["error"]["type"], "action_request_validation_exception");
            assert!(response["error"]["reason"].as_str().unwrap().contains("disabling [track_total_hits]"));
        }
    }
    let (status, response) = es.post("/books/_search?track_total_hits=-1", json!({})).await;
    assert_eq!(status, 200, "{response}");
    assert!(response["hits"].get("total").is_none(), "{response}");
}

/// A scroll cursor advances for sorted and unsorted requests; the page after the last hit is empty.
#[tokio::test(flavor = "multi_thread")]
async fn scroll_advances_sorted_and_unsorted() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    for (label, body) in [
        ("unsorted match_all", json!({"size": 3, "query": {"match_all": {}}})),
        ("unsorted size 1", json!({"size": 1})),
        ("unsorted scored", json!({"size": 2, "query": {"range": {"year": {"gte": 0}}}})),
        ("sorted", json!({"size": 3, "sort": [{"year": "asc"}]})),
    ] {
        let (st, v) = es.post("/books/_search?scroll=1m", body).await;
        assert_eq!(st, 200, "{label}: {v}");
        let sid = v["_scroll_id"].as_str().unwrap().to_string();
        let mut seen = ids(&v);
        let mut pages = 0;
        loop {
            let (_, v) = es.post("/_search/scroll", json!({"scroll": "1m", "scroll_id": sid})).await;
            let page = ids(&v);
            if page.is_empty() {
                break;
            }
            seen.extend(page);
            pages += 1;
            assert!(pages < 10, "{label}: scroll never ends: {seen:?}");
        }
        seen.sort();
        assert_eq!(seen, vec!["1", "2", "3", "4"], "{label}: every hit exactly once");
        // And it stays empty.
        let (_, v) = es.post("/_search/scroll", json!({"scroll": "1m", "scroll_id": sid})).await;
        assert!(ids(&v).is_empty(), "{label}");
    }
}

/// `_shard_doc` sort directions and composite PIT before/after-all cursors.
#[tokio::test(flavor = "multi_thread")]
async fn shard_doc_desc_and_before_all() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    es.put("/books/_doc/5?refresh=true", json!({"title": "later segment", "year": 2024})).await;
    let (st, v) = es.post("/books/_pit?keep_alive=1m", json!({})).await;
    assert_eq!(st, 200);
    let pit = v["id"].as_str().unwrap().to_string();
    let run = |sort: &'static str, after: Option<i64>, size: u64| {
        let es = &es;
        let pit = pit.clone();
        async move {
            let mut body = json!({"pit": {"id": pit, "keep_alive": "1m"}, "size": size, "sort": [{"_shard_doc": sort}]});
            if let Some(a) = after {
                body["search_after"] = json!([a]);
            }
            let (st, v) = es.post("/_search", body).await;
            assert_eq!(st, 200, "{v}");
            v
        }
    };
    let asc = ids(&run("asc", None, 10).await);
    let mut desc = ids(&run("desc", None, 10).await);
    assert_eq!(asc.len(), 5);
    desc.reverse();
    assert_eq!(asc, desc, "desc is the exact reverse of asc");
    // Page through descending with the last hit's sort value.
    let mut seen = Vec::new();
    let mut after = None;
    for _ in 0..6 {
        let v = run("desc", after, 2).await;
        let page = ids(&v);
        if page.is_empty() {
            break;
        }
        after = v["hits"]["hits"].as_array().unwrap().last().unwrap()["sort"][0].as_i64();
        seen.extend(page);
    }
    seen.reverse();
    assert_eq!(seen, asc, "descending pagination visits every doc once");
    // Negative search_after: before the first doc in either direction.
    assert_eq!(ids(&run("asc", Some(-1), 10).await), asc);
    let mut d = ids(&run("desc", Some(-1), 10).await);
    d.reverse();
    assert_eq!(d, asc);
    // Composite PIT's exhausted-sub sentinel must suppress all ties in either direction.
    for direction in ["asc", "desc"] {
        assert!(ids(&run(direction, Some(i64::MIN), 10).await).is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn aliases_templates_analyze() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/_aliases", json!({"actions": [{"add": {"index": "books", "alias": "alice_books", "filter": {"term": {"author": "alice"}}}}, {"add": {"index": "books", "alias": "all_books"}}]})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true, "errors": false})));
    let v = es.search("alice_books", json!({})).await;
    assert_eq!(v["hits"]["total"]["value"], 2);
    let v = es.search("all_books", json!({})).await;
    assert_eq!(v["hits"]["total"]["value"], 4);
    let (_, v) = es.get("/_alias/alice_books").await;
    assert_eq!(v["books"]["aliases"]["alice_books"]["filter"], json!({"term": {"author": "alice"}}));
    let (st, v) = es.get("/_alias/nothing").await;
    assert_eq!((st, v), (404, json!({"error": "alias [nothing] missing", "status": 404})));
    let (st, _) = es.delete("/books/_alias/all_books").await;
    assert_eq!(st, 200);
    let (st, _) = es.get("/all_books/_search").await;
    assert_eq!(st, 404);
    // composable + component templates applied on auto-create
    let (st, _) = es.put("/_component_template/ct", json!({"template": {"mappings": {"properties": {"level": {"type": "keyword"}}}}})).await;
    assert_eq!(st, 200);
    let (st, v) = es.put("/_index_template/logs", json!({"index_patterns": ["logs-*"], "composed_of": ["ct"], "priority": 10, "template": {"settings": {"number_of_replicas": 0}, "aliases": {"logs": {}}}})).await;
    assert_eq!((st, v), (200, json!({"acknowledged": true})));
    let (_, v) = es.get("/_index_template/logs").await;
    assert_eq!(v["index_templates"][0]["name"], "logs");
    assert_eq!(v["index_templates"][0]["index_template"]["priority"], 10);
    let (st, _) = es.put("/logs-1/_doc/1?refresh=true", json!({"level": "warn", "msg": "x"})).await;
    assert_eq!(st, 201);
    let (_, v) = es.get("/logs-1/_mapping").await;
    assert_eq!(v["logs-1"]["mappings"]["properties"]["level"]["type"], "keyword");
    let (_, v) = es.get("/logs-1/_settings").await;
    assert_eq!(v["logs-1"]["settings"]["index"]["number_of_replicas"], "0");
    let v = es.search("logs", json!({})).await;
    assert_eq!(v["hits"]["total"]["value"], 1);
    let (st, _) = es.get("/_index_template/nope").await;
    assert_eq!(st, 404);
    // simulate: resolves components without storing a template named `_simulate`
    let (st, v) = es
        .post("/_index_template/_simulate/sim", json!({"index_patterns": ["sim-*"], "composed_of": ["ct"], "template": {"settings": {"hidden": true, "auto_expand_replicas": "0-1"}, "mappings": {"dynamic": false}}}))
        .await;
    assert_eq!(st, 200);
    assert_eq!(v["template"]["mappings"]["properties"]["level"]["type"], "keyword");
    assert_eq!(v["template"]["mappings"]["dynamic"], false);
    assert_eq!(v["template"]["settings"]["index"]["auto_expand_replicas"], "0-1");
    assert_eq!(v["overlapping"], json!([]));
    // ES registers only `POST /_index_template/_simulate`: 405, not a template lookup.
    let (st, _) = es.get("/_index_template/_simulate").await;
    assert_eq!(st, 405);
    let (_, v) = es.post0("/_index_template/_simulate/logs").await;
    assert_eq!(v["template"]["aliases"], json!({"logs": {}}));
    let (_, v) = es.post0("/_index_template/_simulate_index/logs-2").await;
    assert_eq!(v["template"]["settings"]["index"]["number_of_replicas"], "0");
    let (st, v) = es.post("/_index_template/_simulate", json!({"index_patterns": ["x-*"], "composed_of": ["missing"]})).await;
    assert_eq!((st, v["error"]["type"].clone()), (400, json!("invalid_index_template_exception")));
    // analyze
    let (st, v) = es.post("/_analyze", json!({"analyzer": "standard", "text": "Quick Brown"})).await;
    assert_eq!(st, 200);
    assert_eq!(
        v,
        json!({"tokens": [{"token": "quick", "start_offset": 0, "end_offset": 5, "type": "<ALPHANUM>", "position": 0}, {"token": "brown", "start_offset": 6, "end_offset": 11, "type": "<ALPHANUM>", "position": 1}]})
    );
    let (_, v) = es.post("/_analyze", json!({"tokenizer": "whitespace", "filter": ["lowercase"], "text": "A B"})).await;
    assert_eq!(v["tokens"][1]["token"], "b");
    assert_eq!(v["tokens"][1]["type"], "word");
    let (_, v) = es.post("/books/_analyze", json!({"field": "title", "text": "Foxes"})).await;
    assert_eq!(v["tokens"][0]["token"], "foxes");
}

#[tokio::test(flavor = "multi_thread")]
async fn cluster_cat_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.get("/_cluster/health").await;
    assert_eq!(st, 200);
    assert_eq!(v["status"], "green");
    assert_eq!(v["number_of_nodes"], 1);
    assert_eq!(v["active_primary_shards"], 1);
    assert_eq!(v["active_shards_percent_as_number"], 100.0);
    es.put("/withrep", json!({})).await;
    let (_, v) = es.get("/_cluster/health").await;
    assert_eq!(v["status"], "yellow");
    assert_eq!(v["unassigned_shards"], 1);
    let (st, v) = es.get("/_cluster/health?wait_for_status=green&timeout=1s").await;
    assert_eq!((st, v["timed_out"].clone()), (408, json!(true)));
    let (_, v) = es.get("/_cluster/state").await;
    assert!(v["metadata"]["indices"]["books"].is_object());
    assert!(v["master_node"].is_string());
    let (_, v) = es.get("/_nodes").await;
    assert_eq!(v["_nodes"], json!({"total": 1, "successful": 1, "failed": 0}));
    let (_, v) = es.get("/_nodes/stats").await;
    assert!(v["nodes"].as_object().unwrap().values().next().unwrap()["jvm"].is_object());
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {"search.max_buckets": 1000}})).await;
    assert_eq!(st, 200);
    assert_eq!(v["persistent"]["search"]["max_buckets"], "1000");
    let (_, v) = es.get("/_cluster/settings?flat_settings=true").await;
    assert_eq!(v["persistent"]["search.max_buckets"], "1000");
    let (_, v) = es.get("/books/_stats").await;
    assert_eq!(v["_all"]["primaries"]["docs"]["count"], 4);
    assert_eq!(v["indices"]["books"]["primaries"]["docs"]["count"], 4);
    assert!(v["indices"]["books"]["uuid"].is_string());
    let (_, _, t) = es.raw(Method::GET, "/_cat/indices?v&h=index,docs.count&s=index", None, "").await;
    let lines: Vec<&str> = t.lines().collect();
    assert_eq!(lines[0].split_whitespace().collect::<Vec<_>>(), vec!["index", "docs.count"]);
    assert_eq!(lines[1].split_whitespace().collect::<Vec<_>>(), vec!["books", "4"]);
    let (_, v) = es.get("/_cat/indices/books?format=json").await;
    assert_eq!(v[0]["index"], "books");
    assert_eq!(v[0]["health"], "green");
    assert_eq!(v[0]["docs.count"], "4");
    let (_, _, t) = es.raw(Method::GET, "/_cat/count/books", None, "").await;
    assert!(t.trim_end().ends_with(" 4"));
    let (_, _, t) = es.raw(Method::GET, "/_cat/health?v", None, "").await;
    assert!(t.starts_with("epoch"));
    let (_, _, t) = es.raw(Method::GET, "/_cat/shards/withrep", None, "").await;
    assert!(t.contains("UNASSIGNED"));
    let (_, v) = es.get("/_xpack").await;
    // Licensing is hardwired to trial (AGENTS.md "Licensing: always enterprise").
    assert_eq!(v["license"]["type"], "trial");
    let (_, v) = es.get("/_license").await;
    assert_eq!(v["license"]["status"], "active");
    // open/close
    let (st, v) = es.post0("/withrep/_close").await;
    assert_eq!(st, 200);
    assert_eq!(v["indices"]["withrep"]["closed"], true);
    let (st, v) = es.get("/withrep/_search").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "index_closed_exception");
    let (st, _) = es.post0("/withrep/_open").await;
    assert_eq!(st, 200);
    // forcemerge, flush, refresh
    for p in ["/books/_refresh", "/books/_flush", "/books/_forcemerge?max_num_segments=1"] {
        let (st, v) = es.post0(p).await;
        assert_eq!(st, 200, "{p}");
        assert_eq!(v["_shards"]["failed"], 0);
    }
    let v = es.search("books", json!({})).await;
    assert_eq!(v["hits"]["total"]["value"], 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn gzip_and_params() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    seed(&es).await;
    // gzip request body
    use flate2::write::GzEncoder;
    use std::io::Write;
    let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(br#"{"query":{"term":{"author":"bob"}}}"#).unwrap();
    let body = enc.finish().unwrap();
    let r = es
        .client
        .post(format!("{}/books/_search", es.base))
        .header("content-type", "application/json")
        .header("content-encoding", "gzip")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(ids(&v), vec!["2"]);
    // gzip response (reqwest decodes transparently when gzip feature is on)
    let r = es.client.get(format!("{}/books/_search", es.base)).header("accept-encoding", "gzip").send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["hits"]["total"]["value"], 4);
    // bad boolean param
    let (st, v) = es.get("/books/_search?explain=yes").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "illegal_argument_exception");
    // error_trace
    let (_, v) = es.get("/nope/_search?error_trace=true").await;
    assert!(v["error"]["stack_trace"].is_string());
    // GET with body
    let (_, v) = es.req(Method::GET, "/books/_search", Some(json!({"query": {"term": {"author": "carol"}}}))).await;
    assert_eq!(ids(&v), vec!["4"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let (es, node) = start(tmp.path()).await;
        es.put("/r", json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).await;
        es.put("/r/_doc/1", json!({"k": "a"})).await;
        es.post0("/r/_flush").await;
        es.put("/r/_doc/2", json!({"k": "b"})).await; // only in the translog
        es.delete("/r/_doc/1").await;
        es._task.abort();
        drop(es);
        drop(node);
    }
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (es, _) = start(tmp.path()).await;
    let (st, v) = es.get("/r/_doc/2").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_source"], json!({"k": "b"}));
    assert_eq!(v["_seq_no"], 1);
    let (st, _) = es.get("/r/_doc/1").await;
    assert_eq!(st, 404);
    let v = es.search("r", json!({"query": {"term": {"k": "b"}}})).await;
    assert_eq!(ids(&v), vec!["2"]);
    let (_, v) = es.put("/r/_doc/3", json!({"k": "c"})).await;
    assert_eq!(v["_seq_no"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn knn_nested_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _) = start(tmp.path()).await;
    let (st, v) = es
        .put(
            "/nt",
            json!({"settings": {"index": {"number_of_shards": 2}}, "mappings": {"properties": {
                "name": {"type": "keyword"},
                "nested": {"type": "nested", "properties": {
                    "paragraph_id": {"type": "keyword"},
                    "vector": {"type": "dense_vector", "dims": 5, "index": true, "similarity": "l2_norm",
                               "index_options": {"type": "hnsw", "m": 16, "ef_construction": 200}}}}}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    es.put("/nt/_doc/1", json!({"name": "cow.jpg", "nested": [{"paragraph_id": 0, "vector": [230.0, 300.33, -34.8988, 15.555, -200.0]}, {"paragraph_id": 1, "vector": [240.0, 300, -3, 1, -20]}]})).await;
    es.put("/nt/_doc/2", json!({"name": "moose.jpg", "nested": [{"paragraph_id": 0, "vector": [-0.5, 100.0, -13, 14.8, -156.0]}, {"paragraph_id": 2, "vector": [0, 100.0, 0, 14.8, -156.0]}, {"paragraph_id": 3, "vector": [0, 1.0, 0, 1.8, -15.0]}]})).await;
    es.put("/nt/_doc/3", json!({"name": "rabbit.jpg", "nested": [{"paragraph_id": 0, "vector": [0.5, 111.3, -13.0, 14.8, -156.0]}]})).await;
    let knn = json!({"field": "nested.vector", "query_vector": [-0.5, 90.0, -10, 14.8, -156.0], "k": 2, "num_candidates": 3});
    // published (still in-memory) memtables
    es.post0("/nt/_refresh").await;
    let v = es.search("nt", json!({"fields": ["name"], "knn": knn})).await;
    assert_eq!(ids(&v), vec!["2", "3"], "memtable: {v}");
    es.post0("/nt/_refresh").await;
    let v = es.search("nt", json!({"fields": ["name"], "knn": knn})).await;
    assert_eq!(ids(&v), vec!["2", "3"], "segment: {v}");
    assert_eq!(v["hits"]["hits"][0]["fields"]["name"][0], "moose.jpg");
    // inner_hits
    let mut k2 = knn.clone();
    k2["inner_hits"] = json!({"size": 1, "fields": ["nested.paragraph_id"], "_source": false});
    let v = es.search("nt", json!({"fields": ["name"], "knn": k2})).await;
    assert_eq!(ids(&v), vec!["2", "3"], "{v}");
    assert_eq!(v["hits"]["hits"][0]["inner_hits"]["nested"]["hits"]["hits"][0]["fields"]["nested"][0]["paragraph_id"][0], "0", "{v}");
    // filter on parent
    let mut k3 = knn.clone();
    k3["filter"] = json!({"term": {"name": "rabbit.jpg"}});
    let v = es.search("nt", json!({"knn": k3})).await;
    assert_eq!(ids(&v), vec!["3"], "{v}");
    // knn query form
    let v = es.search("nt", json!({"query": {"nested": {"path": "nested", "query": {"knn": {"field": "nested.vector", "query_vector": [-0.5, 90.0, -10, 14.8, -156.0], "num_candidates": 3}}}}})).await;
    assert_eq!(ids(&v), vec!["2", "3", "1"], "query: {v}");
}

#[tokio::test]
async fn fleet_partial_search_preserves_raw_source_with_failed_checkpoint() {
    use elasticsearch_engine::source_json::{raw_elements, raw_members};
    fn field(text: &str, key: &str) -> String {
        raw_members(text).unwrap().into_iter().find(|(k, _)| k == key).unwrap().1.get().to_string()
    }
    let dir = tempfile::tempdir().unwrap();
    let (es, node) = start(dir.path()).await;
    let (status, response) = es.put("/fleet-partial", json!({
        "settings":{"number_of_shards":2,"number_of_replicas":0},
        "mappings":{"properties":{"z":{"type":"keyword"},"n":{"type":"double"}}}
    })).await;
    assert_eq!(status, 200, "{response}");
    let source = r#"{"z":"\ud800","n":1.50}"#;
    let (status, _, text) = es.raw(Method::PUT, "/fleet-partial/_doc/one?refresh=true", Some(source.into()), "application/json").await;
    assert_eq!(status, 201, "{text}");
    let good_shard = node.get_index("fleet-partial").unwrap().route("one", None).unwrap();
    let checkpoints = if good_shard == 0 { "-1,999999" } else { "999999,-1" };
    let (status, _, text) = es.raw(Method::POST,
        &format!("/fleet-partial/_fleet/_fleet_search?wait_for_checkpoints={checkpoints}&allow_partial_search_results=true"),
        Some("{}".into()), "application/json").await;
    assert_eq!(status, 200, "{text}");
    let hits = raw_elements(&field(&field(&text, "hits"), "hits")).unwrap();
    assert_eq!(hits.len(), 1, "{text}");
    assert_eq!(field(hits[0].get(), "_source"), source);
    let shards: Value = serde_json::from_str(&field(&text, "_shards")).unwrap();
    assert_eq!(shards["successful"], 1);
    assert_eq!(shards["failed"], 1);
    assert_eq!(shards["failures"][0]["shard"], 1 - good_shard);
}

/// Production review #36: a bulk update with stale `if_seq_no` /
/// `if_primary_term` conflicts (409) even when it would be a no-op: an
/// identical partial document and a script setting `ctx.op = 'noop'`.
/// Current conditions still answer the no-op.
#[tokio::test(flavor = "multi_thread")]
async fn bulk_conditional_noop_updates_check_their_conditions() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/cnd/_doc/1?refresh=true", json!({"n": 1})).await;
    assert_eq!(st, 201, "{v}");
    let (seq, term) = (v["_seq_no"].as_i64().unwrap(), v["_primary_term"].as_i64().unwrap());
    let (st, v) = es.put("/cnd/_doc/1", json!({"n": 1, "m": 2})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/cnd/_doc/1", json!({"n": 1})).await;
    assert_eq!(st, 200, "{v}");
    let (cur_seq, cur_term) = (v["_seq_no"].as_i64().unwrap(), v["_primary_term"].as_i64().unwrap());
    assert!(cur_seq > seq);
    let noop_doc = json!({"doc": {"n": 1}});
    let noop_script = json!({"script": {"source": "ctx.op = 'noop'"}});
    for body in [&noop_doc, &noop_script] {
        // Stale: 409 in the bulk item and on the single endpoint.
        let (st, v) = es.ndjson("/_bulk", &[json!({"update": {"_index": "cnd", "_id": "1", "if_seq_no": seq, "if_primary_term": term}}), body.clone()]).await;
        assert_eq!(st, 200, "{v}");
        let item = &v["items"][0]["update"];
        assert_eq!((item["status"].as_u64(), item["error"]["type"].as_str()), (Some(409), Some("version_conflict_engine_exception")), "{body}: {v}");
        let (st, v) = es.post(&format!("/cnd/_update/1?if_seq_no={seq}&if_primary_term={term}"), body.clone()).await;
        assert_eq!((st, v["error"]["type"].as_str()), (409, Some("version_conflict_engine_exception")), "{body}: {v}");
        // Current: the no-op.
        let (st, v) = es.ndjson("/_bulk", &[json!({"update": {"_index": "cnd", "_id": "1", "if_seq_no": cur_seq, "if_primary_term": cur_term}}), body.clone()]).await;
        assert_eq!(st, 200, "{v}");
        let item = &v["items"][0]["update"];
        assert_eq!((item["status"].as_u64(), item["result"].as_str()), (Some(200), Some("noop")), "{body}: {v}");
    }
}

/// Production review #37: a bulk request's URL `routing` applies to every
/// item without its own, exactly as item routing: required-routing
/// mappings accept it, the document is stored with that `_routing`
/// (readable only with it), and a reindex keeps it.
#[tokio::test(flavor = "multi_thread")]
async fn bulk_url_routing_is_the_items_routing() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    for (index, required) in [("rreq", true), ("ropt", false)] {
        let (st, v) = es.put(&format!("/{index}"), json!({"settings": {"number_of_shards": 3}, "mappings": {"_routing": {"required": required}, "properties": {"n": {"type": "long"}}}})).await;
        assert_eq!(st, 200, "{v}");
        // URL routing, item routing: the same outcome.
        let (st, v) = es.ndjson(&format!("/{index}/_bulk?routing=r1&refresh=true"), &[json!({"index": {"_id": "u"}}), json!({"n": 1}), json!({"create": {"_id": "u2"}}), json!({"n": 2})]).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{index}: {v}");
        let (st, v) = es.ndjson(&format!("/{index}/_bulk?refresh=true"), &[json!({"index": {"_id": "i", "routing": "r1"}}), json!({"n": 1})]).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{index}: {v}");
        // The item's own routing wins over the URL's.
        let (st, v) = es.ndjson(&format!("/{index}/_bulk?routing=r1&refresh=true"), &[json!({"index": {"_id": "o", "routing": "r2"}}), json!({"n": 3})]).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{index}: {v}");
        for (id, r) in [("u", "r1"), ("u2", "r1"), ("i", "r1"), ("o", "r2")] {
            let (st, v) = es.get(&format!("/{index}/_doc/{id}?routing={r}")).await;
            assert_eq!((st, v["_routing"].as_str()), (200, Some(r)), "{index}/{id}: {v}");
        }
        // Update and delete items take the URL routing too.
        let (st, v) = es.ndjson(&format!("/{index}/_bulk?routing=r1&refresh=true"), &[json!({"update": {"_id": "u"}}), json!({"doc": {"n": 10}}), json!({"delete": {"_id": "u2"}})]).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{index}: {v}");
        let (_, v) = es.get(&format!("/{index}/_doc/u?routing=r1")).await;
        assert_eq!(v["_source"]["n"], 10, "{v}");
        let (st, _) = es.get(&format!("/{index}/_doc/u2?routing=r1")).await;
        assert_eq!(st, 404);
        // A reindex keeps each document's routing.
        let dest = format!("{index}-copy");
        let (st, v) = es.put(&format!("/{dest}"), json!({"settings": {"number_of_shards": 3}, "mappings": {"_routing": {"required": required}}})).await;
        assert_eq!(st, 200, "{v}");
        let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": index}, "dest": {"index": dest}})).await;
        assert_eq!((st, v["created"].as_u64(), v["failures"].as_array().map(Vec::len)), (200, Some(3), Some(0)), "{v}");
        for (id, r) in [("u", "r1"), ("i", "r1"), ("o", "r2")] {
            let (st, v) = es.get(&format!("/{dest}/_doc/{id}?routing={r}")).await;
            assert_eq!((st, v["_routing"].as_str()), (200, Some(r)), "{dest}/{id}: {v}");
        }
    }
}

/// Production review #135: a delete without the routing a
/// `_routing.required` mapping demands is refused before it is sequenced,
/// singly and in bulk, on one shard and on several: the document keeps its
/// source, version and seq_no. With its routing it deletes.
#[tokio::test(flavor = "multi_thread")]
async fn deletes_without_required_routing_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    for (index, shards) in [("rq1", 1), ("rq3", 3)] {
        let (st, v) = es.put(&format!("/{index}"), json!({"settings": {"number_of_shards": shards}, "mappings": {"_routing": {"required": true}}})).await;
        assert_eq!(st, 200, "{v}");
        for id in ["1", "2"] {
            let (st, v) = es.put(&format!("/{index}/_doc/{id}?routing=tenant-a&refresh=true"), json!({"n": 1})).await;
            assert_eq!(st, 201, "{v}");
        }
        let (_, before) = es.get(&format!("/{index}/_doc/1?routing=tenant-a")).await;
        let (st, v) = es.delete(&format!("/{index}/_doc/1")).await;
        assert_eq!((st, v["error"]["type"].as_str()), (400, Some("routing_missing_exception")), "{index}: {v}");
        let (st, v) = es.ndjson(&format!("/{index}/_bulk?refresh=true"), &[json!({"delete": {"_id": "1"}}), json!({"delete": {"_id": "2", "routing": "tenant-a"}})]).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!((v["items"][0]["delete"]["status"].as_u64(), v["items"][0]["delete"]["error"]["type"].as_str()), (Some(400), Some("routing_missing_exception")), "{index}: {v}");
        assert_eq!(v["items"][1]["delete"]["result"].as_str(), Some("deleted"), "{index}: {v}");
        let (st, after) = es.get(&format!("/{index}/_doc/1?routing=tenant-a")).await;
        assert_eq!(st, 200, "{index}: {after}");
        for k in ["_source", "_version", "_seq_no", "_primary_term"] {
            assert_eq!(after[k], before[k], "{index}: {k}");
        }
        let (st, v) = es.delete(&format!("/{index}/_doc/1?routing=tenant-a")).await;
        assert_eq!((st, v["result"].as_str()), (200, Some("deleted")), "{index}: {v}");
    }
}

/// Production review #136: bulk `if_seq_no` / `if_primary_term` /
/// `version` given as JSON strings are numbers (ES `BulkRequestParser`
/// reads them with a coercing `longValue()`): a stale quoted condition
/// conflicts for index, delete and update, a current one applies, and an
/// unparsable one fails the item before it writes. Integers still work.
#[tokio::test(flavor = "multi_thread")]
async fn bulk_quoted_numeric_conditions_are_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/q/_doc/1?refresh=true", json!({"n": 1})).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.put("/q/_doc/1?refresh=true", json!({"n": 2})).await;
    assert_eq!(st, 200, "{v}");
    let (seq, term) = (v["_seq_no"].as_i64().unwrap(), v["_primary_term"].as_i64().unwrap());
    let stale = (seq - 1).to_string();
    let item = |v: &Value, op: &str| (v["items"][0][op]["status"].as_u64(), v["items"][0][op]["error"]["type"].as_str().map(str::to_string));
    let conflict = (Some(409), Some("version_conflict_engine_exception".to_string()));
    for (op, body) in [("index", Some(json!({"n": 9}))), ("update", Some(json!({"doc": {"n": 9}}))), ("delete", None)] {
        let mut lines = vec![json!({op: {"_index": "q", "_id": "1", "if_seq_no": stale, "if_primary_term": term.to_string()}})];
        lines.extend(body.clone());
        let (st, v) = es.ndjson("/_bulk?refresh=true", &lines).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(item(&v, op), conflict, "{op}: {v}");
        // Unparsable: refused, nothing written.
        let mut lines = vec![json!({op: {"_index": "q", "_id": "1", "if_seq_no": "zero", "if_primary_term": term.to_string()}})];
        lines.extend(body.clone());
        let (st, v) = es.ndjson("/_bulk?refresh=true", &lines).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(item(&v, op).0, Some(400), "{op}: {v}");
        let (_, doc) = es.get("/q/_doc/1").await;
        assert_eq!((doc["_source"]["n"].as_i64(), doc["_seq_no"].as_i64()), (Some(2), Some(seq)), "{op}: unchanged");
    }
    // External version as a string: stale conflicts.
    let (st, v) = es.ndjson("/_bulk?refresh=true", &[json!({"index": {"_index": "q", "_id": "e", "version": "5", "version_type": "external"}}), json!({"n": 1})]).await;
    assert_eq!((st, v["items"][0]["index"]["_version"].as_u64()), (200, Some(5)), "{v}");
    let (_, v) = es.ndjson("/_bulk", &[json!({"index": {"_index": "q", "_id": "e", "version": "4", "version_type": "external"}}), json!({"n": 2})]).await;
    assert_eq!(item(&v, "index"), conflict, "{v}");
    // Current quoted conditions apply; integer conditions still do.
    let (_, v) = es.ndjson("/_bulk?refresh=true", &[json!({"update": {"_index": "q", "_id": "1", "if_seq_no": seq.to_string(), "if_primary_term": term.to_string()}}), json!({"doc": {"n": 3}})]).await;
    assert_eq!(v["items"][0]["update"]["result"].as_str(), Some("updated"), "{v}");
    let seq = v["items"][0]["update"]["_seq_no"].as_i64().unwrap();
    let (_, v) = es.ndjson("/_bulk?refresh=true", &[json!({"index": {"_index": "q", "_id": "1", "if_seq_no": seq - 1, "if_primary_term": term}}), json!({"n": 4})]).await;
    assert_eq!(item(&v, "index"), conflict, "{v}");
    let (_, v) = es.ndjson("/_bulk?refresh=true", &[json!({"delete": {"_index": "q", "_id": "1", "if_seq_no": seq, "if_primary_term": term}})]).await;
    assert_eq!(v["items"][0]["delete"]["result"].as_str(), Some("deleted"), "{v}");
}

/// Production review #191, #199, #241 (standalone reindex): the script's
/// metadata changes apply over the destination request. A cleared `_id`
/// generates ids (no overwrite of the destination's `1`, both sources
/// copied); a scripted `_routing` wins over `dest.routing: discard`; a
/// scripted delete keeps the external version condition.
#[tokio::test(flavor = "multi_thread")]
async fn reindex_script_metadata_applies_over_the_destination_request() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    for (index, body) in [("s1", json!({"t": "a"})), ("s2", json!({"t": "b"})), ("dst", json!({"keep": true}))] {
        let (st, v) = es.put(&format!("/{index}/_doc/1?refresh=true"), body).await;
        assert_eq!(st, 201, "{v}");
    }
    let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": ["s1", "s2"]}, "dest": {"index": "dst"}, "script": {"source": "ctx._id = null"}})).await;
    assert_eq!((st, v["created"].as_u64()), (200, Some(2)), "{v}");
    let (_, v) = es.get("/dst/_doc/1").await;
    assert_eq!((v["_source"].clone(), v["_version"].as_u64()), (json!({"keep": true}), Some(1)), "{v}");
    let (_, v) = es.get("/dst/_count").await;
    assert_eq!(v["count"], 3, "{v}");
    // Scripted routing over `dest.routing: discard`.
    let (st, v) = es.put("/rsrc/_doc/1?routing=old&refresh=true", json!({"tenant": "t1"})).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.put("/rdst", json!({"settings": {"number_of_shards": 3}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/_reindex?refresh=true", json!({"source": {"index": "rsrc"}, "dest": {"index": "rdst", "routing": "discard"}, "script": {"source": "ctx._routing = ctx._source.tenant"}})).await;
    assert_eq!((st, v["created"].as_u64()), (200, Some(1)), "{v}");
    let (st, v) = es.get("/rdst/_doc/1?routing=t1").await;
    assert_eq!((st, v["_routing"].as_str()), (200, Some("t1")), "{v}");
    // Scripted delete with external versioning: source 3, destination 4.
    let (st, v) = es.put("/vsrc/_doc/1?version=3&version_type=external&refresh=true", json!({"n": 1})).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.put("/vdst/_doc/1?version=4&version_type=external&refresh=true", json!({"n": "newer"})).await;
    assert_eq!(st, 201, "{v}");
    let body = json!({"conflicts": "proceed", "source": {"index": "vsrc"}, "dest": {"index": "vdst", "version_type": "external"}, "script": {"source": "ctx.op = 'delete'"}});
    let (st, v) = es.post("/_reindex?refresh=true", body.clone()).await;
    assert_eq!((st, v["deleted"].as_u64(), v["version_conflicts"].as_u64()), (200, Some(0), Some(1)), "{v}");
    let (st, v) = es.get("/vdst/_doc/1").await;
    assert_eq!((st, v["_version"].as_u64()), (200, Some(4)), "the newer destination stays: {v}");
    // Destination at the source's version: deleted.
    let (st, v) = es.put("/vdst2/_doc/1?version=3&version_type=external&refresh=true", json!({"n": 1})).await;
    assert_eq!(st, 201, "{v}");
    let mut body = body;
    body["dest"]["index"] = json!("vdst2");
    let (st, v) = es.post("/_reindex?refresh=true", body.clone()).await;
    assert_eq!((st, v["deleted"].as_u64()), (200, Some(1)), "{v}");
    let (st, _) = es.get("/vdst2/_doc/1").await;
    assert_eq!(st, 404);
    // An older destination copy conflicts too (ES internal version equality).
    let (st, v) = es.put("/vdst3/_doc/1?version=2&version_type=external&refresh=true", json!({"n": "older"})).await;
    assert_eq!(st, 201, "{v}");
    body["dest"]["index"] = json!("vdst3");
    let (st, v) = es.post("/_reindex?refresh=true", body).await;
    assert_eq!((st, v["deleted"].as_u64(), v["version_conflicts"].as_u64()), (200, Some(0), Some(1)), "{v}");
    let (st, v) = es.get("/vdst3/_doc/1").await;
    assert_eq!((st, v["_version"].as_u64()), (200, Some(2)), "{v}");
}

/// Production review #186: quoted `_seq_no` range bounds are numbers (ES
/// `SeqNoFieldMapper` parses strings with `Long.parseLong`): they select
/// exactly what the numeric bounds do, and a malformed bound fails the
/// query (and a by-query mutation) instead of disappearing.
#[tokio::test(flavor = "multi_thread")]
async fn quoted_seq_no_range_bounds_apply() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let lines: Vec<Value> = (0..6).flat_map(|i| [json!({"index": {"_index": "sq", "_id": i.to_string()}}), json!({"n": i})]).collect();
    let (st, v) = es.ndjson("/_bulk?refresh=true", &lines).await;
    assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
    let count = |range: Value| {
        let es = &es;
        async move {
            let (st, v) = es.post("/sq/_search", json!({"query": {"range": {"_seq_no": range}}, "size": 0})).await;
            (st, v["hits"]["total"]["value"].as_u64())
        }
    };
    assert_eq!(count(json!({"gte": 2})).await, (200, Some(4)));
    assert_eq!(count(json!({"gte": "2"})).await, (200, Some(4)));
    assert_eq!(count(json!({"gte": "2", "lt": "4"})).await, (200, Some(2)));
    assert_eq!(count(json!({"gt": "2", "lte": 4})).await, (200, Some(2)));
    let (st, v) = es.post("/sq/_search", json!({"query": {"range": {"_seq_no": {"gte": "two"}}}})).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = es.post("/sq/_delete_by_query?refresh=true", json!({"query": {"range": {"_seq_no": {"gte": "x", "lt": "4"}}}})).await;
    assert!(st >= 400, "{st} {v}");
    let (_, v) = es.get("/sq/_count").await;
    assert_eq!(v["count"], 6, "a malformed bound deletes nothing: {v}");
    let (st, v) = es.post("/sq/_delete_by_query?refresh=true", json!({"query": {"range": {"_seq_no": {"gte": "4"}}}})).await;
    assert_eq!((st, v["deleted"].as_u64()), (200, Some(2)), "{v}");
    let (_, v) = es.get("/sq/_count").await;
    assert_eq!(v["count"], 4, "{v}");
}

/// Production review #192: metadata keys in an ingested source never take
/// over the write's metadata (ES `CtxMap` rejects them): an id-less bulk
/// `index` whose source carries `_id` fails instead of overwriting that
/// document, with or without a pipeline; `_routing` likewise.
#[tokio::test(flavor = "multi_thread")]
async fn ingest_source_metadata_cannot_take_over_a_write() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/tag", json!({"processors": [{"set": {"field": "tagged", "value": true}}]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/ing/_doc/victim?refresh=true", json!({"owner": "me"})).await;
    assert_eq!(st, 201, "{v}");
    for (pipeline, source) in [("?pipeline=tag", json!({"_id": "victim", "owner": "attacker"})), ("?pipeline=tag", json!({"_routing": "r9", "owner": "x"})), ("", json!({"_id": "victim", "owner": "attacker"}))] {
        let (st, v) = es.ndjson(&format!("/ing/_bulk{pipeline}"), &[json!({"index": {}}), source.clone()]).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["items"][0]["index"]["status"].as_u64(), Some(400), "{pipeline} {source}: {v}");
    }
    let (_, v) = es.get("/ing/_doc/victim").await;
    assert_eq!((v["_source"].clone(), v["_version"].as_u64()), (json!({"owner": "me"}), Some(1)), "{v}");
    // Without metadata keys an id-less pipeline write gets a generated id.
    let (_, v) = es.ndjson("/ing/_bulk?pipeline=tag&refresh=true", &[json!({"index": {}}), json!({"owner": "new"})]).await;
    assert_eq!(v["items"][0]["index"]["status"].as_u64(), Some(201), "{v}");
    assert_ne!(v["items"][0]["index"]["_id"], json!("victim"));
}

/// Production review #193: `index.routing_partition_size` and
/// `index.mapping.exclude_source_vectors` are final (ES): a closed index
/// or a `reopen` update rejects them and keeps settings and documents.
#[tokio::test(flavor = "multi_thread")]
async fn final_routing_and_vector_source_settings_cannot_change() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/fin", json!({"settings": {"number_of_shards": 4, "routing_partition_size": 2}, "mappings": {"_routing": {"required": true}, "properties": {"emb": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/fin/_doc/1?routing=r&refresh=true", json!({"emb": [1.0, 2.0], "n": 1})).await;
    assert_eq!(st, 201, "{v}");
    for setting in [json!({"index.routing_partition_size": 3}), json!({"index.mapping.exclude_source_vectors": false})] {
        let (st, v) = es.put("/fin/_settings?reopen=true", setting.clone()).await;
        assert_eq!(st, 400, "reopen {setting}: {v}");
        es.post0("/fin/_close").await;
        let (st, v) = es.put("/fin/_settings", setting.clone()).await;
        assert_eq!((st, v["error"]["type"].as_str()), (400, Some("illegal_argument_exception")), "closed {setting}: {v}");
        assert!(v["error"]["reason"].as_str().unwrap_or("").contains("final index setting"), "{v}");
        es.post0("/fin/_open").await;
    }
    let (_, v) = es.get("/fin/_settings?flat_settings=true").await;
    assert_eq!(v["fin"]["settings"]["index.routing_partition_size"], json!("2"), "{v}");
    assert!(v["fin"]["settings"].get("index.mapping.exclude_source_vectors").is_none(), "{v}");
    let (st, v) = es.get("/fin/_doc/1?routing=r").await;
    assert_eq!((st, v["_source"]["n"].as_i64()), (200, Some(1)), "{v}");
}

/// Production review #194: when a composable template matches, legacy
/// templates do not apply at all (ES): no legacy pipeline, source pruning
/// or alias leaks into the new index. Without a composable match the
/// legacy template still applies.
#[tokio::test(flavor = "multi_thread")]
async fn composable_templates_exclude_legacy_templates() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/drop-all", json!({"processors": [{"drop": {}}]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_template/old", json!({"index_patterns": ["*"], "order": 0, "settings": {"index.default_pipeline": "drop-all"}, "mappings": {"_source": {"excludes": ["secret"]}}, "aliases": {"legacy-alias": {}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_index_template/events", json!({"index_patterns": ["events-*"], "priority": 100, "template": {"mappings": {"properties": {"msg": {"type": "keyword"}}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/events-1/_doc/1?refresh=true", json!({"msg": "hello", "secret": "kept"})).await;
    assert_eq!((st, v["result"].as_str()), (201, Some("created")), "{v}");
    let (_, v) = es.get("/events-1/_doc/1").await;
    assert_eq!(v["_source"], json!({"msg": "hello", "secret": "kept"}), "{v}");
    let (_, v) = es.get("/events-1").await;
    assert!(v["events-1"]["settings"]["index"].get("default_pipeline").is_none(), "{v}");
    assert_eq!(v["events-1"]["aliases"], json!({}), "{v}");
    assert!(v["events-1"]["mappings"].get("_source").is_none(), "{v}");
    // Legacy-only control.
    let (st, v) = es.put("/other", json!({})).await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = es.get("/other").await;
    assert_eq!(v["other"]["settings"]["index"]["default_pipeline"], json!("drop-all"), "{v}");
    assert!(v["other"]["aliases"].get("legacy-alias").is_some(), "{v}");
}

/// Production review #195: an alias never gets a second write index (ES
/// `IndexAbstraction.Alias`): adding one through `_aliases` or index
/// creation fails and changes nothing; an atomic swap moves writes; a
/// rollover still moves the write index.
#[tokio::test(flavor = "multi_thread")]
async fn an_alias_has_at_most_one_write_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/logs-000001", json!({"aliases": {"live": {"is_write_index": true}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/logs-000002", json!({})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/_aliases", json!({"actions": [{"add": {"index": "logs-000002", "alias": "live", "is_write_index": true}}]})).await;
    assert_eq!((st, v["error"]["type"].as_str()), (500, Some("illegal_state_exception")), "{v}");
    assert!(v["error"]["reason"].as_str().unwrap_or("").contains("more than one write index"), "{v}");
    let (_, v) = es.get("/logs-000002/_alias").await;
    assert_eq!(v["logs-000002"]["aliases"], json!({}), "unchanged: {v}");
    let (st, v) = es.put("/logs-000003", json!({"aliases": {"live": {"is_write_index": true}}})).await;
    assert!(st >= 400, "{v}");
    let (st, _) = es.get("/logs-000003").await;
    assert_eq!(st, 404);
    // Atomic swap: writes now land in the new index only.
    let (st, v) = es.post("/_aliases", json!({"actions": [
        {"add": {"index": "logs-000001", "alias": "live", "is_write_index": false}},
        {"add": {"index": "logs-000002", "alias": "live", "is_write_index": true}}]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/live/_doc?refresh=true", json!({"n": 1})).await;
    assert_eq!((st, v["_index"].as_str()), (201, Some("logs-000002")), "{v}");
    // Rollover moves the write index.
    let (st, v) = es.post0("/live/_rollover").await;
    assert_eq!((st, v["rolled_over"].as_bool()), (200, Some(true)), "{v}");
    let (st, v) = es.post("/live/_doc?refresh=true", json!({"n": 2})).await;
    assert_eq!((st, v["_index"].as_str()), (201, Some("logs-000003")), "{v}");
}

/// Production review #232: `_reload_search_analyzers` rebuilds the search
/// analyzers it reports (a synonyms set updated with `refresh=false`
/// takes effect only then).
#[tokio::test(flavor = "multi_thread")]
async fn reload_search_analyzers_rebuilds_them() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/_synonyms/vs", json!({"synonyms_set": [{"id": "r", "synonyms": "vehicle => car"}]})).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = es.put("/syn", json!({
        "settings": {"analysis": {"filter": {"f": {"type": "synonym_graph", "synonyms_set": "vs", "updateable": true}},
            "analyzer": {"s": {"tokenizer": "standard", "filter": ["lowercase", "f"]}}}},
        "mappings": {"properties": {"t": {"type": "text", "analyzer": "standard", "search_analyzer": "s"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let lines = vec![json!({"index": {"_index": "syn", "_id": "car"}}), json!({"t": "car"}), json!({"index": {"_index": "syn", "_id": "truck"}}), json!({"t": "truck"})];
    let (st, _) = es.ndjson("/_bulk?refresh=true", &lines).await;
    assert_eq!(st, 200);
    let hit = || {
        let es = &es;
        async move {
            let (_, v) = es.post("/syn/_search?request_cache=false", json!({"query": {"match": {"t": "vehicle"}}})).await;
            v["hits"]["hits"].as_array().map(|a| a.iter().map(|h| h["_id"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>()).unwrap_or_default()
        }
    };
    assert_eq!(hit().await, vec!["car".to_string()]);
    let (st, v) = es.put("/_synonyms/vs?refresh=false", json!({"synonyms_set": [{"id": "r", "synonyms": "vehicle => truck"}]})).await;
    assert!(st == 200 || st == 201, "{v}");
    assert_eq!(hit().await, vec!["car".to_string()], "refresh=false defers the change");
    let (st, v) = es.post0("/syn/_reload_search_analyzers").await;
    assert_eq!((st, v["_shards"]["failed"].as_u64()), (200, Some(0)), "{v}");
    assert_eq!(v["reload_details"][0]["reloaded_analyzers"], json!(["s"]), "{v}");
    assert_eq!(hit().await, vec!["truck".to_string()], "the reported reload took effect");
}

/// Production review #230: a stored percolator `query_string` without
/// fields searches the percolator index's `index.query.default_field`, as
/// the same query on that index would.
#[tokio::test(flavor = "multi_thread")]
async fn percolator_query_string_uses_the_index_default_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/rules", json!({"settings": {"index.query.default_field": ["message"]}, "mappings": {"properties": {"q": {"type": "percolator"}, "message": {"type": "text"}, "tag": {"type": "text"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/rules/_doc/default?refresh=true", json!({"q": {"query_string": {"query": "danger"}}})).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.put("/rules/_doc/tagged?refresh=true", json!({"q": {"query_string": {"query": "danger", "fields": ["tag"]}}})).await;
    assert_eq!(st, 201, "{v}");
    let matched = |doc: Value| {
        let es = &es;
        async move {
            let (st, v) = es.post("/rules/_search", json!({"query": {"percolate": {"field": "q", "document": doc}}})).await;
            assert_eq!(st, 200, "{v}");
            let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
            ids.sort();
            ids
        }
    };
    assert_eq!(matched(json!({"message": "safe", "tag": "danger"})).await, vec!["tagged".to_string()]);
    assert_eq!(matched(json!({"message": "danger", "tag": "safe"})).await, vec!["default".to_string()]);
}

/// Production review #367: a component template still composed into an
/// index template, and an index template that is the only one a live data
/// stream matches, cannot be deleted (ES); optional components can; and a
/// create never silently drops a missing required component.
#[tokio::test(flavor = "multi_thread")]
async fn required_templates_cannot_be_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, v) = es.put("/_component_template/req", json!({"template": {"mappings": {"properties": {"amount": {"type": "long"}}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_component_template/opt", json!({"template": {"settings": {"index.number_of_replicas": 0}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_index_template/ev", json!({"index_patterns": ["ev-*"], "data_stream": {}, "priority": 10, "composed_of": ["req", "opt"], "ignore_missing_component_templates": ["opt"]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.req(Method::PUT, "/_data_stream/ev-1", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.delete("/_component_template/req").await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("illegal_argument_exception")), "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("component templates [req] cannot be removed as they are still in use by index templates [ev]"), "{v}");
    let (st, _) = es.get("/_component_template/req").await;
    assert_eq!(st, 200, "kept");
    let (st, v) = es.delete("/_component_template/opt").await;
    assert_eq!(st, 200, "an optional component can go: {v}");
    let (st, v) = es.delete("/_index_template/ev").await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("unable to remove composable templates [ev] as they are in use by a data streams [ev-1]"), "{v}");
    // Rollover keeps the required mapping.
    let (st, v) = es.post0("/ev-1/_rollover").await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = es.get("/ev-1/_mapping").await;
    assert!(v.as_object().unwrap().values().all(|m| m["mappings"]["properties"]["amount"]["type"] == json!("long")), "{v}");
    // Once the stream is gone, both can be deleted.
    let (st, v) = es.delete("/_data_stream/ev-1").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.delete("/_index_template/ev").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.delete("/_component_template/req").await;
    assert_eq!(st, 200, "{v}");
}

/// Production review #429: `if_seq_no` / `if_primary_term` in an update
/// body are conditions (ES `UpdateRequest` body fields): a stale one
/// conflicts and leaves the newer revision; a current one applies; URL
/// conditions still work.
#[tokio::test(flavor = "multi_thread")]
async fn update_body_concurrency_conditions_are_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let (st, a) = es.put("/u/_doc/1?refresh=true", json!({"f": "a", "g": 1})).await;
    assert_eq!(st, 201, "{a}");
    let (st, b) = es.post("/u/_update/1", json!({"doc": {"g": 2}})).await;
    assert_eq!(st, 200, "{b}");
    let stale = json!({"doc": {"f": "stale"}, "if_seq_no": a["_seq_no"], "if_primary_term": a["_primary_term"]});
    let (st, v) = es.post("/u/_update/1", stale).await;
    assert_eq!((st, v["error"]["type"].as_str()), (409, Some("version_conflict_engine_exception")), "{v}");
    let (_, doc) = es.get("/u/_doc/1").await;
    assert_eq!(doc["_source"], json!({"f": "a", "g": 2}), "revision B unchanged");
    let current = json!({"doc": {"f": "c"}, "if_seq_no": b["_seq_no"], "if_primary_term": b["_primary_term"]});
    let (st, v) = es.post("/u/_update/1", current).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post(&format!("/u/_update/1?if_seq_no={}&if_primary_term={}", b["_seq_no"], b["_primary_term"]), json!({"doc": {"f": "d"}})).await;
    assert_eq!(st, 409, "URL conditions still checked: {v}");
}
