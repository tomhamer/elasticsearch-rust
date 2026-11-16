//! Production review batch 6, standalone node end to end: #444 version
//! build identity, #451/#452 half floats, #464 mapped-format date bounds,
//! #465 week dates, #466 date locales, #468 date_nanos epoch seconds, #481
//! close safeguards.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::sync::Arc;

struct Es {
    base: String,
    client: Client,
    _node: Arc<Node>,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let n = node.clone();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    Es { base: format!("http://{addr}"), client: Client::new(), _node: node }
}

impl Es {
    async fn req(&self, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn ok(&self, m: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.req(m.clone(), path, body).await;
        assert!((200..300).contains(&st), "{m} {path}: {st} {v}");
        v
    }
    async fn ids(&self, index: &str, body: Value) -> Vec<String> {
        let v = self.ok(Method::POST, &format!("/{index}/_search"), Some(body)).await;
        let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    }
    async fn index(&self, index: &str, docs: &[(&str, Value)]) {
        for (id, d) in docs {
            self.ok(Method::PUT, &format!("/{index}/_doc/{id}"), Some(d.clone())).await;
        }
        self.ok(Method::POST, &format!("/{index}/_refresh"), None).await;
    }
}

#[tokio::test]
async fn version_ranges_and_sorts_keep_build_metadata() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    es.ok(Method::PUT, "/v", Some(json!({"mappings": {"properties": {"v": {"type": "version"}}}}))).await;
    es.index("v", &[("b1", json!({"v": "1.2.3+build.1"})), ("b2", json!({"v": "1.2.3+build.2"})), ("z", json!({"v": "01.2"}))]).await;
    assert_eq!(es.ids("v", json!({"query": {"range": {"v": {"gt": "1.2.3+build.1"}}}})).await, vec!["b2".to_string(), "z".to_string()]);
    let v = es.ok(Method::POST, "/v/_search", Some(json!({"sort": [{"v": "asc"}]}))).await;
    let sorted: Vec<&Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| &h["sort"][0]).collect();
    assert_eq!(sorted, vec![&json!("1.2.3+build.1"), &json!("1.2.3+build.2"), &json!("01.2")], "{v}");
    let v = es.ok(Method::POST, "/v/_search", Some(json!({"sort": [{"v": "asc"}], "search_after": ["1.2.3+build.1"]}))).await;
    assert_eq!(v["hits"]["hits"][0]["_id"], "b2", "{v}");
}

#[tokio::test]
async fn half_float_bounds_and_subnormals() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    es.ok(Method::PUT, "/h", Some(json!({"mappings": {"properties": {"h": {"type": "half_float"}}}}))).await;
    es.index("h", &[("one", json!({"h": 1.0})), ("next", json!({"h": 1.0009765625})), ("tiny", json!({"h": 0.00000004}))]).await;
    assert_eq!(es.ids("h", json!({"query": {"range": {"h": {"gte": 1.0001}}}})).await, vec!["next"]);
    assert_eq!(es.ids("h", json!({"query": {"range": {"h": {"gt": 1.0008}}}})).await, vec!["next"]);
    assert_eq!(es.ids("h", json!({"query": {"range": {"h": {"lte": 1.0008, "gt": 0.5}}}})).await, vec!["one"]);
    assert_eq!(es.ids("h", json!({"query": {"range": {"h": {"lt": 1.0009765625, "gt": 0.5}}}})).await, vec!["one"]);
    // #452: the smallest subnormal, not zero.
    assert_eq!(es.ids("h", json!({"query": {"range": {"h": {"gt": 0, "lt": 0.5}}}})).await, vec!["tiny"]);
    let v = es.ok(Method::POST, "/h/_search", Some(json!({"size": 0, "query": {"ids": {"values": ["tiny"]}}, "aggs": {"m": {"min": {"field": "h"}}}}))).await;
    assert_eq!(v["aggregations"]["m"]["value"].as_f64(), Some(5.960464477539063e-8), "{v}");
}

#[tokio::test]
async fn date_bounds_round_up_with_mapped_formats_and_partial_times() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    es.ok(Method::PUT, "/d", Some(json!({"mappings": {"properties": {"t": {"type": "date", "format": "yyyy/MM/dd||epoch_millis"}, "iso": {"type": "date"}}}}))).await;
    // Noon on 2025-01-02 (epoch millis), and 10:30 for the ISO field.
    es.index("d", &[("noon", json!({"t": 1735819200000i64, "iso": "2025-01-02T10:30:00Z"}))]).await;
    assert_eq!(es.ids("d", json!({"query": {"range": {"t": {"lte": "2025/01/02"}}}})).await, vec!["noon"], "lte ends the day");
    assert!(es.ids("d", json!({"query": {"range": {"t": {"gt": "2025/01/02"}}}})).await.is_empty(), "gt excludes the day");
    assert_eq!(es.ids("d", json!({"query": {"range": {"t": {"gte": "2025/01/02", "lt": "2025/01/03"}}}})).await, vec!["noon"]);
    // Partial ISO times round up to the end of the finest given field.
    assert_eq!(es.ids("d", json!({"query": {"range": {"iso": {"lte": "2025-01-02T10"}}}})).await, vec!["noon"]);
    assert!(es.ids("d", json!({"query": {"range": {"iso": {"gt": "2025-01-02T10"}}}})).await.is_empty());
    assert_eq!(es.ids("d", json!({"query": {"range": {"iso": {"gt": "2025-01-02T10:29"}}}})).await, vec!["noon"]);
    assert!(es.ids("d", json!({"query": {"range": {"iso": {"lt": "2025-01-02T10:30"}}}})).await.is_empty());
}

#[tokio::test]
async fn week_dates_and_locales_parse_what_the_mapping_accepts() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    // #465: two adjacent ISO weeks are two dates.
    es.ok(Method::PUT, "/w", Some(json!({"mappings": {"properties": {"d": {"type": "date", "format": "YYYY-'W'ww-e"}}}}))).await;
    es.index("w", &[("w2", json!({"d": "2025-W02-1"})), ("w3", json!({"d": "2025-W03-1"}))]).await;
    let v = es.ok(Method::POST, "/w/_search", Some(json!({"sort": [{"d": "asc"}], "docvalue_fields": [{"field": "d", "format": "strict_date"}]}))).await;
    let days: Vec<&Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| &h["fields"]["d"][0]).collect();
    assert_eq!(days, vec![&json!("2025-01-06"), &json!("2025-01-13")], "{v}");
    // #466: German month names.
    es.ok(Method::PUT, "/l", Some(json!({"mappings": {"properties": {"d": {"type": "date", "format": "dd-MMMM-yyyy", "locale": "de"}}}}))).await;
    es.index("l", &[("mai", json!({"d": "02-Mai-2025"})), ("maerz", json!({"d": "15-März-2025"}))]).await;
    assert_eq!(es.ids("l", json!({"query": {"range": {"d": {"gte": "01-Mai-2025"}}}})).await, vec!["mai"]);
    let v = es.ok(Method::POST, "/l/_search", Some(json!({"query": {"ids": {"values": ["mai"]}}, "docvalue_fields": [{"field": "d", "format": "strict_date"}]}))).await;
    assert_eq!(v["hits"]["hits"][0]["fields"]["d"][0], "2025-05-02", "{v}");
    // Every JDK locale's names (the shared CLDR table): Polish months in
    // the format (genitive) style.
    es.ok(Method::PUT, "/l2", Some(json!({"mappings": {"properties": {"d": {"type": "date", "format": "dd MMMM yyyy", "locale": "pl"}}}}))).await;
    es.index("l2", &[("pl", json!({"d": "02 maja 2025"}))]).await;
    let v = es.ok(Method::POST, "/l2/_search", Some(json!({"docvalue_fields": [{"field": "d", "format": "strict_date"}]}))).await;
    assert_eq!(v["hits"]["hits"][0]["fields"]["d"][0], "2025-05-02", "{v}");
    // Day periods (`a`) are localized too (CLDR, as the JDK prints them).
    es.ok(Method::PUT, "/l3", Some(json!({"mappings": {"properties": {"d": {"type": "date", "format": "yyyy-MM-dd a hh:mm", "locale": "ko"}}}}))).await;
    es.index("l3", &[("ko", json!({"d": "2025-05-02 오후 03:30"}))]).await;
    let v = es.ok(Method::POST, "/l3/_search", Some(json!({"docvalue_fields": [{"field": "d", "format": "strict_date_hour_minute"}]}))).await;
    assert_eq!(v["hits"]["hits"][0]["fields"]["d"][0], "2025-05-02T15:30", "{v}");
    let (st, v) = es.req(Method::PUT, "/l3/_doc/en", Some(json!({"d": "2025-05-02 PM 03:30"}))).await;
    assert_eq!(st, 400, "English AM/PM is not Korean: {v}");
}

#[tokio::test]
async fn date_nanos_epoch_seconds_keep_their_fraction() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    es.ok(Method::PUT, "/n", Some(json!({"mappings": {"properties": {"t": {"type": "date_nanos", "format": "epoch_second"}}}}))).await;
    es.index("n", &[("half", json!({"t": 1735689600.5})), ("whole", json!({"t": 1735689600}))]).await;
    // Indexed at the half second (it was 0.5 ms).
    assert_eq!(es.ids("n", json!({"query": {"range": {"t": {"gte": "2025-01-01T00:00:00.4Z", "lte": "2025-01-01T00:00:00.6Z", "format": "strict_date_optional_time_nanos"}}}})).await, vec!["half"]);
    // The query with its own format finds the half second.
    assert_eq!(es.ids("n", json!({"query": {"range": {"t": {"gte": 1735689600.5, "lte": 1735689600.5, "format": "epoch_second"}}}})).await, vec!["half"]);
    // A numeric bound without a format is epoch millis (ES).
    assert_eq!(es.ids("n", json!({"query": {"range": {"t": {"gte": 1735689600000i64, "lte": 1735689600000i64}}}})).await, vec!["whole"]);
}

#[tokio::test]
async fn closing_honours_the_destructive_safeguards() {
    let t = tempfile::tempdir().unwrap();
    let es = start(t.path()).await;
    es.ok(Method::PUT, "/c1", None).await;
    es.ok(Method::PUT, "/c2", None).await;
    let (st, v) = es.req(Method::POST, "/c*/_close", None).await;
    assert_eq!((st, v["error"]["reason"].as_str()), (400, Some("Wildcard expressions or all indices are not allowed")), "{v}");
    let (st, _) = es.req(Method::POST, "/_all/_close", None).await;
    assert_eq!(st, 400);
    for i in ["c1", "c2"] {
        es.ok(Method::PUT, &format!("/{i}/_doc/1"), Some(json!({"n": 1}))).await;
    }
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.indices.close.enable": false}}))).await;
    let (st, v) = es.req(Method::POST, "/c1/_close", None).await;
    assert_eq!((st, v["error"]["type"].as_str()), (500, Some("illegal_state_exception")), "{v}");
    assert!(v["error"]["reason"].as_str().unwrap_or_default().starts_with("closing indices is disabled - set [cluster.indices.close.enable: true] to enable it"), "{v}");
    es.ok(Method::PUT, "/c1/_doc/2", Some(json!({"n": 2}))).await;
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.indices.close.enable": null, "action.destructive_requires_name": false}}))).await;
    es.ok(Method::POST, "/c*/_close", None).await;
}
