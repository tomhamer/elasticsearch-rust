//! Production review batch 7 (aggregations and search): end-to-end closure
//! tests over the HTTP API of a standalone node.

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};

async fn serve(dir: &std::path::Path) -> String {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    format!("http://127.0.0.1:{}", addr.port())
}

async fn call(c: &Client, method: &str, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method.parse().unwrap(), url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

async fn raw(c: &Client, method: &str, url: String, body: String) -> (u16, Value) {
    let resp = c.request(method.parse().unwrap(), url).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

async fn ok(c: &Client, method: &str, url: String, body: Option<Value>) -> Value {
    let (st, v) = call(c, method, url, body).await;
    assert!(st < 300, "{st}: {v}");
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch7_search_and_aggregation_closures() {
    let dir = tempfile::tempdir().unwrap();
    let base = serve(dir.path()).await;
    let c = Client::new();

    // #614: _count reports the shard that failed its query.
    ok(&c, "PUT", format!("{base}/cnt1"), Some(json!({"mappings": {"properties": {"f": {"type": "keyword"}}}}))).await;
    ok(&c, "PUT", format!("{base}/cnt2"), Some(json!({"mappings": {"properties": {"f": {"type": "long"}}}}))).await;
    ok(&c, "PUT", format!("{base}/cnt1/_doc/1?refresh=true"), Some(json!({"f": "abc"}))).await;
    ok(&c, "PUT", format!("{base}/cnt2/_doc/1?refresh=true"), Some(json!({"f": 5}))).await;
    let v = ok(&c, "POST", format!("{base}/cnt1,cnt2/_count"), Some(json!({"query": {"regexp": {"f": "a.*"}}}))).await;
    assert_eq!(v["count"], 1, "{v}");
    assert_eq!(v["_shards"]["failed"], 1, "{v}");
    assert_eq!(v["_shards"]["successful"], 1, "{v}");
    assert!(v["_shards"]["failures"].as_array().is_some_and(|f| f.len() == 1), "{v}");
    let v = ok(&c, "POST", format!("{base}/cnt1/_count"), Some(json!({"query": {"regexp": {"f": "a.*"}}}))).await;
    assert_eq!(v["_shards"], json!({"total": 1, "successful": 1, "skipped": 0, "failed": 0}), "{v}");

    // SIDE 010: a bucket filter's `_index` term matches the index's alias;
    // a terms-lookup filter works inside a filter aggregation.
    ok(&c, "PUT", format!("{base}/orders-000001"), Some(json!({"aliases": {"orders": {}}, "mappings": {"properties": {"k": {"type": "keyword"}}}}))).await;
    ok(&c, "PUT", format!("{base}/orders-000001/_doc/1?refresh=true"), Some(json!({"k": "a"}))).await;
    ok(&c, "PUT", format!("{base}/lk/_doc/1?refresh=true"), Some(json!({"ks": ["a"]}))).await;
    let v = ok(&c, "POST", format!("{base}/orders-000001/_search"), Some(json!({"size": 0, "aggs": {
        "by_alias": {"filter": {"term": {"_index": "orders"}}},
        "by_lookup": {"filter": {"terms": {"k": {"index": "lk", "id": "1", "path": "ks"}}}},
        "named": {"filters": {"filters": {"o": {"term": {"_index": "orders"}}}}}
    }}))).await;
    assert_eq!(v["aggregations"]["by_alias"]["doc_count"], 1, "{v}");
    assert_eq!(v["aggregations"]["by_lookup"]["doc_count"], 1, "{v}");
    assert_eq!(v["aggregations"]["named"]["buckets"]["o"]["doc_count"], 1, "{v}");

    // SIDE 009: a field runtime in one index and indexed in another sums both.
    ok(&c, "PUT", format!("{base}/rt-a"), Some(json!({"mappings": {"runtime": {"price": {"type": "double", "script": {"source": "emit(10.0)"}}}, "properties": {"k": {"type": "keyword"}}}}))).await;
    ok(&c, "PUT", format!("{base}/rt-b"), Some(json!({"mappings": {"properties": {"price": {"type": "double"}}}}))).await;
    ok(&c, "PUT", format!("{base}/rt-a/_doc/1?refresh=true"), Some(json!({"k": "x"}))).await;
    ok(&c, "PUT", format!("{base}/rt-b/_doc/1?refresh=true"), Some(json!({"price": 100.0}))).await;
    for expr in ["rt-a,rt-b", "rt-b,rt-a"] {
        let v = ok(&c, "POST", format!("{base}/{expr}/_search"), Some(json!({"size": 0, "aggs": {"s": {"sum": {"field": "price"}}, "n": {"value_count": {"field": "price"}}}}))).await;
        assert_eq!(v["aggregations"]["s"]["value"].as_f64(), Some(110.0), "{expr}: {v}");
        assert_eq!(v["aggregations"]["n"]["value"], 2, "{expr}: {v}");
    }

    // SIDE 014: nested sort `max_children` bounds the children per root.
    ok(&c, "PUT", format!("{base}/nsort"), Some(json!({"mappings": {"properties": {"offers": {"type": "nested", "properties": {"price": {"type": "long"}}}}}}))).await;
    ok(&c, "PUT", format!("{base}/nsort/_doc/A"), Some(json!({"offers": [{"price": 100}, {"price": 1}]}))).await;
    ok(&c, "PUT", format!("{base}/nsort/_doc/B?refresh=true"), Some(json!({"offers": [{"price": 50}]}))).await;
    let ids = |v: &Value| v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    let v = ok(&c, "POST", format!("{base}/nsort/_search"), Some(json!({"sort": [{"offers.price": {"order": "asc", "mode": "min", "nested": {"path": "offers", "max_children": 1}}}]}))).await;
    assert_eq!(ids(&v), ["B", "A"], "{v}");
    assert_eq!(v["hits"]["hits"][1]["sort"], json!([100]), "{v}");
    let v = ok(&c, "POST", format!("{base}/nsort/_search"), Some(json!({"sort": [{"offers.price": {"order": "asc", "mode": "min", "nested": {"path": "offers"}}}]}))).await;
    assert_eq!(ids(&v), ["A", "B"], "{v}");

    // #334: an msearch header's routing selects the shards of its search.
    ok(&c, "PUT", format!("{base}/ms"), Some(json!({"settings": {"number_of_shards": 4}}))).await;
    ok(&c, "PUT", format!("{base}/ms/_doc/1?routing=r1&refresh=true"), Some(json!({"v": 1}))).await;
    let (st, v) = raw(&c, "POST", format!("{base}/_msearch"), "{\"index\":\"ms\",\"routing\":\"r1\"}\n{\"query\":{\"match_all\":{}}}\n{\"index\":\"ms\"}\n{\"query\":{\"match_all\":{}}}\n".into()).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["responses"][0]["_shards"]["total"], 1, "{v}");
    assert_eq!(v["responses"][1]["_shards"]["total"], 4, "{v}");
    let (st, v) = raw(&c, "POST", format!("{base}/_msearch/template"), "{\"index\":\"ms\",\"routing\":\"r1\"}\n{\"source\":{\"query\":{\"match_all\":{}}}}\n".into()).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["responses"][0]["_shards"]["total"], 1, "{v}");

    // #604: a sampler's shard_size applies per physical shard, also when a
    // sibling forces the union collection.
    ok(&c, "PUT", format!("{base}/samp"), Some(json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"x": {"type": "long"}}}}))).await;
    let mut bulk = String::new();
    for i in 0..40 {
        bulk.push_str(&format!("{{\"index\":{{\"_index\":\"samp\",\"_id\":\"{i}\"}}}}\n{{\"x\":{i}}}\n"));
    }
    let (st, v) = raw(&c, "POST", format!("{base}/_bulk?refresh=true"), bulk).await;
    assert_eq!(st, 200, "{v}");
    let plain = ok(&c, "POST", format!("{base}/samp/_search"), Some(json!({"size": 0, "aggs": {"s": {"sampler": {"shard_size": 1}, "aggs": {"n": {"value_count": {"field": "x"}}}}}}))).await;
    assert_eq!(plain["aggregations"]["s"]["doc_count"], 2, "{plain}");
    let mixed = ok(&c, "POST", format!("{base}/samp/_search"), Some(json!({"size": 0, "aggs": {
        "s": {"sampler": {"shard_size": 1}, "aggs": {"n": {"value_count": {"field": "x"}}}},
        "r": {"random_sampler": {"probability": 1}, "aggs": {"t": {"sum": {"field": "x"}}}}
    }}))).await;
    assert_eq!(mixed["aggregations"]["s"]["doc_count"], 2, "{mixed}");

    // #603: a children aggregation joins within each index (same parent `_id`
    // in two indices; a mixed metric type selects the union collection).
    for (i, ty) in [("ja", "long"), ("jb", "double")] {
        ok(&c, "PUT", format!("{base}/{i}"), Some(json!({"mappings": {"properties": {"j": {"type": "join", "relations": {"q": "ans"}}, "m": {"type": ty}}}}))).await;
        ok(&c, "PUT", format!("{base}/{i}/_doc/p"), Some(json!({"j": "q"}))).await;
        ok(&c, "PUT", format!("{base}/{i}/_doc/c?routing=p&refresh=true"), Some(json!({"j": {"name": "ans", "parent": "p"}, "m": if i == "ja" { json!(1) } else { json!(2.5) }}))).await;
    }
    let v = ok(&c, "POST", format!("{base}/ja,jb/_search"), Some(json!({"size": 0,
        "query": {"bool": {"filter": [{"term": {"_index": "ja"}}, {"term": {"j": "q"}}]}},
        "aggs": {"kids": {"children": {"type": "ans"}, "aggs": {"m": {"sum": {"field": "m"}}}}}}))).await;
    assert_eq!(v["aggregations"]["kids"]["doc_count"], 1, "{v}");
    assert_eq!(v["aggregations"]["kids"]["m"]["value"].as_f64(), Some(1.0), "{v}");
}
