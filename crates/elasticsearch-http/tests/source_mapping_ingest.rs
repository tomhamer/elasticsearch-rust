//! Production review batch 6 closures over HTTP: source pruning (431),
//! null values (432, 433), date_nanos offsets (445), integral ranges
//! (448), geo-point normalization (457), narrow floats (458), flattened
//! sub-fields with configured analysis (469), constant_keyword terms
//! lookup (477), and ingest: foreach (454, 476), localized dates (461,
//! 462), dissect appends (463).

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
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

    async fn ok(&self, m: Method, p: &str, body: Value) -> Value {
        let (st, v) = self.call(m, p, Some(body)).await;
        assert!(st == 200 || st == 201, "{p}: {st} {v}");
        v
    }

    async fn count(&self, index: &str, q: Value) -> u64 {
        let (st, v) = self.call(Method::POST, &format!("/{index}/_search"), Some(json!({"query": q, "track_total_hits": true}))).await;
        assert_eq!(st, 200, "{v}");
        v["hits"]["total"]["value"].as_u64().unwrap()
    }

    async fn source(&self, index: &str, id: &str) -> Value {
        let (st, v) = self.call(Method::GET, &format!("/{index}/_doc/{id}"), None).await;
        assert_eq!(st, 200, "{v}");
        v["_source"].clone()
    }
}

async fn start() -> (tempfile::TempDir, Es) {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    (tmp, Es { base: format!("http://{addr}"), c: Client::new() })
}

#[tokio::test]
async fn source_and_mapping_fidelity() {
    let (_tmp, es) = start().await;

    // 431: includes reach nested arrays and wildcard-matched objects.
    es.ok(Method::PUT, "/inc", json!({"mappings": {"_source": {"includes": ["a.b"]}, "properties": {"a": {"properties": {"b": {"type": "long"}, "c": {"type": "long"}}}}}})).await;
    es.ok(Method::PUT, "/inc/_doc/1?refresh=true", json!({"a": [[{"b": 1, "c": 2}]]})).await;
    assert_eq!(es.source("inc", "1").await, json!({"a": [[{"b": 1}]]}));
    es.ok(Method::PUT, "/wild", json!({"mappings": {"_source": {"includes": ["*.b"]}}})).await;
    es.ok(Method::PUT, "/wild/_doc/1?refresh=true", json!({"a": {"b": {"c": 1}}, "x": 2})).await;
    assert_eq!(es.source("wild", "1").await, json!({"a": {"b": {"c": 1}}}));

    // 432: synthetic geo_point shows the null_value it indexed.
    es.ok(Method::PUT, "/geo", json!({"settings": {"index.mapping.source.mode": "synthetic"}, "mappings": {"properties": {"location": {"type": "geo_point", "null_value": {"lat": 10, "lon": 20}}}}})).await;
    es.ok(Method::PUT, "/geo/_doc/1?refresh=true", json!({"location": null})).await;
    let s = es.source("geo", "1").await;
    let (lat, lon) = (s["location"]["lat"].as_f64().unwrap_or(f64::NAN), s["location"]["lon"].as_f64().unwrap_or(f64::NAN));
    assert!((lat - 10.0).abs() < 1e-6 && (lon - 20.0).abs() < 1e-6, "{s}");

    // 433: a multi-field's own null_value applies to a null parent value.
    es.ok(Method::PUT, "/mf", json!({"mappings": {"properties": {"text": {"type": "text", "fields": {"raw": {"type": "keyword", "null_value": "NA"}}}}}})).await;
    es.ok(Method::PUT, "/mf/_doc/1?refresh=true", json!({"text": null})).await;
    es.ok(Method::PUT, "/mf/_doc/2?refresh=true", json!({"text": ["x", null]})).await;
    es.ok(Method::PUT, "/mf/_doc/3?refresh=true", json!({"text": "y"})).await;
    assert_eq!(es.count("mf", json!({"term": {"text.raw": "NA"}})).await, 2);

    // 445: date_nanos keeps the fraction of an offset timestamp.
    es.ok(Method::PUT, "/dn", json!({"mappings": {"properties": {"d": {"type": "date_nanos"}}}})).await;
    es.ok(Method::PUT, "/dn/_doc/1?refresh=true", json!({"d": "2025-01-01T01:00:00.123456789+01:00"})).await;
    assert_eq!(es.count("dn", json!({"term": {"d": "2025-01-01T00:00:00.123456789Z"}})).await, 1);
    assert_eq!(es.count("dn", json!({"range": {"d": {"gt": "2025-01-01T00:00:00.123456788Z"}}})).await, 1);
    assert_eq!(es.count("dn", json!({"range": {"d": {"gt": "2025-01-01T00:00:00.123456789Z"}}})).await, 0);

    // 448: integral range bounds are truncated, as ES parses them.
    es.ok(Method::PUT, "/ir", json!({"mappings": {"properties": {"r": {"type": "integer_range"}, "s": {"type": "long_range", "coerce": false}}}})).await;
    es.ok(Method::PUT, "/ir/_doc/1?refresh=true", json!({"r": {"gte": 1.2, "lte": 4.8}})).await;
    es.ok(Method::PUT, "/ir/_doc/2?refresh=true", json!({"r": {"gte": 1.2, "lte": 1.8}})).await;
    assert_eq!(es.count("ir", json!({"term": {"r": 1}})).await, 2);
    assert_eq!(es.count("ir", json!({"term": {"r": 4}})).await, 1);
    assert_eq!(es.count("ir", json!({"term": {"r": 5}})).await, 0);
    let (st, v) = es.call(Method::PUT, "/ir/_doc/3", Some(json!({"s": {"gte": 1.5, "lte": 3}}))).await;
    assert_eq!(st, 400, "{v}");

    // 457: ignore_malformed geo points are normalized, not dropped.
    es.ok(Method::PUT, "/gp", json!({"mappings": {"properties": {"g": {"type": "geo_point", "ignore_malformed": true}, "strict": {"type": "geo_point"}}}})).await;
    es.ok(Method::PUT, "/gp/_doc/1?refresh=true", json!({"g": {"lat": 10, "lon": 190}})).await;
    assert_eq!(es.count("gp", json!({"geo_bounding_box": {"g": {"top_left": {"lat": 11, "lon": -171}, "bottom_right": {"lat": 9, "lon": -169}}}})).await, 1);
    let (st, v) = es.call(Method::PUT, "/gp/_doc/2", Some(json!({"strict": {"lat": 10, "lon": 190}}))).await;
    assert_eq!(st, 400, "{v}");

    // 458: values beyond the narrow format are refused (or ignored).
    es.ok(Method::PUT, "/nf", json!({"mappings": {"properties": {"f": {"type": "float"}, "h": {"type": "half_float"}, "fi": {"type": "float", "ignore_malformed": true}}}})).await;
    for doc in [json!({"f": 1e40}), json!({"h": 70000})] {
        let (st, v) = es.call(Method::PUT, "/nf/_doc/x", Some(doc.clone())).await;
        assert_eq!(st, 400, "{doc}: {v}");
        assert!(v.to_string().contains("supports only finite values"), "{v}");
    }
    es.ok(Method::PUT, "/nf/_doc/1?refresh=true", json!({"fi": 1e40, "f": 3.0e38})).await;
    assert_eq!(es.count("nf", json!({"term": {"_ignored": "fi"}})).await, 1);
    assert_eq!(es.count("nf", json!({"exists": {"field": "fi"}})).await, 0);

    // 469: a flattened property with a configured normalizer is indexed.
    es.ok(Method::PUT, "/fl", json!({
        "settings": {"analysis": {"normalizer": {"lc": {"type": "custom", "filter": ["lowercase"]}}}},
        "mappings": {"properties": {"attributes": {"type": "flattened", "properties": {"service": {"type": "keyword", "normalizer": "lc"}}}}}
    })).await;
    es.ok(Method::PUT, "/fl/_doc/1?refresh=true", json!({"attributes": {"service": "CHECKOUT"}})).await;
    assert_eq!(es.count("fl", json!({"term": {"attributes.service": "checkout"}})).await, 1);

    // 477: a terms lookup on constant_keyword matches like the inline form.
    es.ok(Method::PUT, "/ck", json!({"mappings": {"properties": {"dataset": {"type": "constant_keyword", "value": "logs"}, "n": {"type": "long"}}}})).await;
    for i in 0..3 {
        es.ok(Method::PUT, &format!("/ck/_doc/{i}?refresh=true"), json!({"n": i})).await;
    }
    es.ok(Method::PUT, "/lists/_doc/sel?refresh=true", json!({"vals": ["logs"]})).await;
    es.ok(Method::PUT, "/lists/_doc/other?refresh=true", json!({"vals": ["metrics"]})).await;
    assert_eq!(es.count("ck", json!({"terms": {"dataset": ["logs"]}})).await, 3);
    assert_eq!(es.count("ck", json!({"terms": {"dataset": {"index": "lists", "id": "sel", "path": "vals"}}})).await, 3);
    assert_eq!(es.count("ck", json!({"terms": {"dataset": {"index": "lists", "id": "other", "path": "vals"}}})).await, 0);
}

#[tokio::test]
async fn ingest_fidelity() {
    let (_tmp, es) = start().await;
    let pipeline = |id: &'static str, processors: Value| (id, processors);
    for (id, processors) in [
        pipeline("fe_null", json!([{"foreach": {"field": "items", "ignore_missing": true, "processor": {"uppercase": {"field": "_ingest._value"}}}}])),
        pipeline("fe_key", json!([{"foreach": {"field": "products", "processor": {"set": {"field": "_ingest._key", "value": "{{_ingest._value.display_name}}"}}}}])),
        pipeline("din", json!([{"date_index_name": {"field": "d", "index_name_prefix": "evt-", "date_rounding": "M", "date_formats": ["dd-MMMM-yyyy"], "locale": "fr"}}])),
        pipeline("date_es", json!([{"date": {"field": "d", "formats": ["dd-MMMM-yyyy"], "locale": "es"}}])),
        pipeline("dissect", json!([{"dissect": {"field": "line", "pattern": "%{host} %{+timestamp} %{timestamp} %{message}", "append_separator": " "}}])),
    ] {
        es.ok(Method::PUT, &format!("/_ingest/pipeline/{id}"), json!({"processors": processors})).await;
    }
    // 454
    es.ok(Method::PUT, "/ing/_doc/1?pipeline=fe_null&refresh=true", json!({"items": null, "x": 1})).await;
    assert_eq!(es.source("ing", "1").await, json!({"items": null, "x": 1}));
    // 476
    es.ok(Method::PUT, "/ing/_doc/2?pipeline=fe_key&refresh=true", json!({"products": {"p1": {"display_name": ""}, "p2": {"display_name": "Two"}}})).await;
    assert_eq!(es.source("ing", "2").await, json!({"products": {"Two": {"display_name": "Two"}}}));
    // 461
    let v = es.ok(Method::PUT, "/ignored/_doc/3?pipeline=din&refresh=true", json!({"d": "02-janvier-2025"})).await;
    assert_eq!(v["_index"], "evt-2025-01-01", "{v}");
    // 462: every Spanish month.
    let months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
    for (i, m) in months.iter().enumerate() {
        let id = format!("m{i}");
        es.ok(Method::PUT, &format!("/ing/_doc/{id}?pipeline=date_es&refresh=true"), json!({"d": format!("02-{m}-2025")})).await;
        assert_eq!(es.source("ing", &id).await["@timestamp"], format!("2025-{:02}-02T00:00:00.000Z", i + 1), "{m}");
    }
    // 463
    es.ok(Method::PUT, "/ing/_doc/4?pipeline=dissect&refresh=true", json!({"line": "node1 2025-01-02 12:30:00 ready"})).await;
    let s = es.source("ing", "4").await;
    assert_eq!((s["host"].clone(), s["timestamp"].clone(), s["message"].clone()), (json!("node1"), json!("2025-01-02 12:30:00"), json!("ready")), "{s}");
}
