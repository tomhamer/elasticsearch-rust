//! HTTP compat: 406 on unsupported Content-Type, `?human`, real `_stats`
//! timing counters, typed_keys for rare/significant terms.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

async fn start(dir: &std::path::Path) -> (String, Client) {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    (format!("http://{addr}"), Client::new())
}

async fn call(c: &Client, m: Method, url: String, body: Option<String>, ct: Option<&str>) -> (u16, String) {
    let mut r = c.request(m, url);
    if let Some(ct) = ct {
        r = r.header("content-type", ct);
    }
    if let Some(b) = body {
        r = r.body(b);
    }
    let resp = r.send().await.unwrap();
    (resp.status().as_u16(), resp.text().await.unwrap())
}

async fn json_call(c: &Client, m: Method, url: String, body: Option<Value>) -> (u16, Value) {
    let (st, t) = call(c, m, url, body.map(|b| b.to_string()), Some("application/json")).await;
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

#[tokio::test]
async fn unsupported_content_type_is_406() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (st, t) = call(&c, Method::PUT, format!("{base}/i/_doc/1"), Some("{\"a\":1}".into()), Some("application/x-www-form-urlencoded")).await;
    assert_eq!(st, 406);
    assert_eq!(serde_json::from_str::<Value>(&t).unwrap(), json!({"error": "Content-Type header [application/x-www-form-urlencoded] is not supported", "status": 406}));
    let (st, t) = call(&c, Method::POST, format!("{base}/_search"), Some("{}".into()), Some("text/plain")).await;
    assert_eq!(st, 406);
    assert_eq!(serde_json::from_str::<Value>(&t).unwrap(), json!({"error": "Content-Type header [text/plain] is not supported", "status": 406}));
    // Index was not created by the rejected request.
    let (st, _) = call(&c, Method::HEAD, format!("{base}/i"), None, None).await;
    assert_eq!(st, 404);
    // Supported types (with parameters, vendor types) and body-less requests pass.
    let (st, t) = call(&c, Method::PUT, format!("{base}/i/_doc/1"), Some("{\"a\":1}".into()), Some("application/json; charset=UTF-8")).await;
    assert_eq!(st, 201, "{t}");
    let (st, t) = call(&c, Method::PUT, format!("{base}/i/_doc/2"), Some("{\"a\":2}".into()), Some("application/vnd.elasticsearch+json; compatible-with=8")).await;
    assert_eq!(st, 400, "{t}");
    assert_eq!(serde_json::from_str::<Value>(&t).unwrap()["error"]["type"], "media_type_header_exception");
    // ES9 requires matching compatibility versions on both headers.
    let response = c.put(format!("{base}/i/_doc/2"))
        .header("content-type", "application/vnd.elasticsearch+json; compatible-with=8")
        .header("accept", "application/vnd.elasticsearch+json; compatible-with=8")
        .body("{\"a\":2}").send().await.unwrap();
    assert_eq!(response.status().as_u16(), 201, "{}", response.text().await.unwrap());
    let (st, _) = call(&c, Method::GET, format!("{base}/i/_doc/1"), None, Some("text/plain")).await;
    assert_eq!(st, 200);
}

#[tokio::test]
async fn stats_counters_and_human() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (st, v) = json_call(&c, Method::PUT, format!("{base}/s"), Some(json!({"settings": {"number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..3 {
        let (st, v) = json_call(&c, Method::PUT, format!("{base}/s/_doc/{i}"), Some(json!({"n": i}))).await;
        assert_eq!(st, 201, "{v}");
    }
    let (st, _) = json_call(&c, Method::DELETE, format!("{base}/s/_doc/2"), None).await;
    assert_eq!(st, 200);
    // noop update
    let (st, v) = json_call(&c, Method::POST, format!("{base}/s/_update/0"), Some(json!({"doc": {"n": 0}}))).await;
    assert_eq!((st, v["result"].clone()), (200, json!("noop")));
    let _ = json_call(&c, Method::GET, format!("{base}/s/_doc/1"), None).await; // exists
    let _ = json_call(&c, Method::GET, format!("{base}/s/_doc/nope"), None).await; // missing
    // Searches see the shard as of its last refresh.
    let _ = json_call(&c, Method::POST, format!("{base}/s/_refresh"), None).await;
    let _ = json_call(&c, Method::POST, format!("{base}/s/_search"), Some(json!({"query": {"match_all": {}}}))).await; // query + fetch
    let _ = json_call(&c, Method::POST, format!("{base}/s/_search"), Some(json!({"size": 0}))).await; // query only
    let _ = json_call(&c, Method::POST, format!("{base}/s/_flush"), None).await;

    let (st, v) = json_call(&c, Method::GET, format!("{base}/s/_stats"), None).await;
    assert_eq!(st, 200, "{v}");
    let p = &v["indices"]["s"]["primaries"];
    let ix = &p["indexing"];
    assert_eq!(ix["index_total"], 3);
    assert_eq!(ix["delete_total"], 1);
    assert_eq!(ix["noop_update_total"], 1);
    assert_eq!(ix["index_current"], 0);
    assert_eq!(ix["index_failed"], 0);
    assert!(ix["index_time_in_millis"].is_u64() && ix["delete_time_in_millis"].is_u64());
    let g = &p["get"];
    // GET exists + GET missing + the update's internal get.
    assert_eq!(g["exists_total"], 2, "{g}");
    assert_eq!(g["missing_total"], 1, "{g}");
    assert_eq!(g["total"], 3);
    assert_eq!(g["current"], 0);
    let s = &p["search"];
    assert_eq!(s["query_total"], 2, "{s}");
    assert_eq!(s["fetch_total"], 1, "{s}");
    assert_eq!(s["query_current"], 0);
    assert_eq!(s["open_contexts"], 0);
    assert!(s["query_time_in_millis"].is_u64() && s["fetch_time_in_millis"].is_u64());
    assert!(p["refresh"]["total"].as_u64().unwrap() >= 1);
    assert!(p["flush"]["total"].as_u64().unwrap() >= 1);
    assert!(p["flush"]["total_time_in_millis"].is_u64());
    // Without ?human there are no human fields.
    assert!(p["store"].get("size").is_none());
    assert!(ix.get("index_time").is_none());

    // ?human adds the readable field right before the raw one.
    let (st, t) = call(&c, Method::GET, format!("{base}/s/_stats/store,indexing?human"), None, None).await;
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&t).unwrap();
    let p = &v["indices"]["s"]["primaries"];
    let keys: Vec<&String> = p["store"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["size", "size_in_bytes", "total_data_set_size", "total_data_set_size_in_bytes", "reserved", "reserved_in_bytes"]);
    let size = p["store"]["size"].as_str().unwrap();
    assert!(size.ends_with("b") && size.chars().next().unwrap().is_ascii_digit(), "{size}");
    assert_eq!(p["store"]["reserved"], "0b");
    let it = p["indexing"]["index_time"].as_str().unwrap();
    assert!(it == "0s" || it.ends_with("ms") || it.ends_with('s'), "{it}");
    assert_eq!(p["indexing"]["throttle_time"], "0s");
    assert!(p.get("search").is_none());
    // Node stats carry the same summed sections and honour ?human too.
    let (st, v) = json_call(&c, Method::GET, format!("{base}/_nodes/stats?human"), None).await;
    assert_eq!(st, 200);
    let node = v["nodes"].as_object().unwrap().values().next().unwrap().clone();
    assert_eq!(node["indices"]["indexing"]["index_total"], 3);
    assert!(node["indices"]["store"]["size"].is_string());
    assert!(node["jvm"]["uptime"].is_string());
    // Documents are never humanised, even under ?human.
    let (_, v) = json_call(&c, Method::PUT, format!("{base}/d/_doc/1?refresh=true"), Some(json!({"size_in_bytes": 5}))).await;
    assert_eq!(v["result"], "created");
    let (_, v) = json_call(&c, Method::GET, format!("{base}/d/_doc/1?human"), None).await;
    assert_eq!(v["_source"], json!({"size_in_bytes": 5}));
}

#[tokio::test]
async fn typed_keys_for_rare_and_significant_terms() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (st, v) = json_call(&c, Method::PUT, format!("{base}/t"), Some(json!({"mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    for (i, (k, n)) in [("a", 1), ("a", 1), ("b", 2), ("a", 1)].iter().enumerate() {
        let _ = json_call(&c, Method::PUT, format!("{base}/t/_doc/{i}?refresh=true"), Some(json!({"k": k, "n": n}))).await;
    }
    let body = json!({"size": 0, "query": {"term": {"k": "b"}}, "aggs": {
        "rs": {"rare_terms": {"field": "k"}},
        "rl": {"rare_terms": {"field": "n"}},
        "ss": {"significant_terms": {"field": "k", "min_doc_count": 1}},
        "sl": {"significant_terms": {"field": "n", "min_doc_count": 1}}
    }});
    let (st, v) = json_call(&c, Method::POST, format!("{base}/t/_search?typed_keys"), Some(body)).await;
    assert_eq!(st, 200, "{v}");
    let mut keys: Vec<&String> = v["aggregations"].as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(keys, ["lrareterms#rl", "siglterms#sl", "sigsterms#ss", "srareterms#rs"]);
    assert_eq!(v["aggregations"]["srareterms#rs"]["buckets"], json!([{"key": "b", "doc_count": 1}]));
    assert_eq!(v["aggregations"]["sigsterms#ss"]["buckets"][0]["key"], "b");
}
