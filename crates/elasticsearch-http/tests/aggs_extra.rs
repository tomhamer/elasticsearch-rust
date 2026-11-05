//! HTTP-level tests for the less common aggregations (exact JSON shapes).

use elasticsearch_engine::node::Node;
use reqwest::Client;
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

async fn call(c: &Client, m: reqwest::Method, url: String, body: Option<String>, ct: &str) -> (u16, Value) {
    let mut r = c.request(m, url);
    if let Some(b) = body {
        r = r.header("content-type", ct).body(b);
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

#[tokio::test]
async fn significant_text_two_shard_sampler_over_http() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (status, response) = call(&c, reqwest::Method::PUT, format!("{base}/texts"), Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}}}
    }).to_string()), "application/json").await;
    assert_eq!(status, 200, "{response}");
    let mut bulk = String::new();
    for i in 0..200 {
        bulk.push_str(&json!({"index": {"_id": i.to_string()}}).to_string());
        bulk.push('\n');
        bulk.push_str("{\"text\":\"common other\"}\n");
    }
    let (status, response) = call(&c, reqwest::Method::POST, format!("{base}/texts/_bulk?refresh=true"), Some(bulk), "application/x-ndjson").await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["errors"], false, "{response}");
    let body = json!({"size": 0, "aggs": {"sample": {"sampler": {"shard_size": 1}, "aggs": {
        "sig": {"significant_text": {"field": "text", "include": ["common"], "min_doc_count": 1, "percentage": {}}}
    }}}});
    let (status, response) = call(&c, reqwest::Method::POST, format!("{base}/texts/_search?typed_keys=true"), Some(body.to_string()), "application/json").await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["_shards"]["successful"], 2, "{response}");
    let sample = &response["aggregations"]["sampler#sample"];
    assert_eq!(sample["doc_count"], 2, "{response}");
    assert_eq!(sample["sigsterms#sig"]["doc_count"], 2);
    assert_eq!(sample["sigsterms#sig"]["bg_count"], 200);
    assert_eq!(sample["sigsterms#sig"]["buckets"], json!([{"key": "common", "doc_count": 2, "bg_count": 200, "score": 0.01}]));
}

#[tokio::test]
async fn extra_aggregations_over_http() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (st, v) = call(
        &c,
        reqwest::Method::PUT,
        format!("{base}/sales"),
        Some(json!({"mappings": {"properties": {"tag": {"type": "keyword"}, "color": {"type": "keyword"}, "price": {"type": "double"}, "qty": {"type": "long"}, "ts": {"type": "date"}}}}).to_string()),
        "application/json",
    )
    .await;
    assert_eq!(st, 200, "{v}");
    let docs = [
        json!({"tag": "a", "color": "red", "price": 10.0, "qty": 1, "ts": "2020-01-01T10:00:00Z"}),
        json!({"tag": "a", "color": "red", "price": 20.0, "qty": 2, "ts": "2020-01-01T12:00:00Z"}),
        json!({"tag": "b", "color": "blue", "price": 30.0, "qty": 3, "ts": "2020-01-03T00:00:00Z"}),
        json!({"tag": "b", "color": "red", "price": 40.0, "qty": 4, "ts": "2020-01-03T05:00:00Z"}),
        json!({"tag": "c", "qty": 5, "ts": "2020-02-10T00:00:00Z"}),
    ];
    let mut nd = String::new();
    for (i, d) in docs.iter().enumerate() {
        nd.push_str(&json!({"index": {"_id": i.to_string()}}).to_string());
        nd.push('\n');
        nd.push_str(&d.to_string());
        nd.push('\n');
    }
    let (st, v) = call(&c, reqwest::Method::POST, format!("{base}/sales/_bulk?refresh=true"), Some(nd), "application/x-ndjson").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false);
    let body = json!({"size": 0, "aggs": {
        "tm": {"top_metrics": {"metrics": {"field": "price"}, "sort": {"qty": "desc"}}},
        "wa": {"weighted_avg": {"value": {"field": "price"}, "weight": {"field": "qty"}}},
        "mt": {"multi_terms": {"terms": [{"field": "tag"}, {"field": "color"}], "size": 1}},
        "rt": {"rare_terms": {"field": "tag"}},
        "adh": {"auto_date_histogram": {"field": "ts", "buckets": 2}},
        "bp": {"boxplot": {"field": "qty"}},
        "mad": {"median_absolute_deviation": {"field": "qty"}},
        "ss": {"string_stats": {"field": "color"}},
        "dh": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"r": {"rate": {"field": "qty", "unit": "year"}}}}
    }});
    let (st, v) = call(&c, reqwest::Method::POST, format!("{base}/sales/_search"), Some(body.to_string()), "application/json").await;
    assert_eq!(st, 200, "{v}");
    let a = &v["aggregations"];
    assert_eq!(a["tm"], json!({"top": [{"sort": [5], "metrics": {"price": null}}]}));
    assert_eq!(a["wa"], json!({"value": 30.0}));
    assert_eq!(a["mt"], json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 2, "buckets": [{"key": ["a", "red"], "key_as_string": "a|red", "doc_count": 2}]}));
    assert_eq!(a["rt"], json!({"buckets": [{"key": "c", "doc_count": 1}]}));
    assert_eq!(
        a["adh"],
        json!({"buckets": [
            {"key_as_string": "2020-01-01T00:00:00.000Z", "key": 1577836800000i64, "doc_count": 4},
            {"key_as_string": "2020-02-01T00:00:00.000Z", "key": 1580515200000i64, "doc_count": 1}
        ], "interval": "1M"})
    );
    assert_eq!(a["bp"]["min"], 1.0);
    assert_eq!(a["bp"]["max"], 5.0);
    assert_eq!(a["bp"]["q2"], 3.0);
    assert_eq!(a["mad"], json!({"value": 1.0}));
    assert_eq!(a["ss"]["count"], 4);
    assert_eq!(a["ss"]["min_length"], 3);
    assert_eq!(a["ss"]["max_length"], 4);
    assert_eq!(a["ss"]["avg_length"], 3.25);
    assert_eq!(a["dh"]["buckets"][0]["r"], json!({"value": 120.0}));
    assert_eq!(a["dh"]["buckets"][1]["r"], json!({"value": 60.0}));

    // significant_terms against the whole index as background.
    let body = json!({"size": 0, "query": {"term": {"color": "red"}}, "aggs": {"s": {"significant_terms": {"field": "tag", "min_doc_count": 1}}}});
    let (st, v) = call(&c, reqwest::Method::POST, format!("{base}/sales/_search"), Some(body.to_string()), "application/json").await;
    assert_eq!(st, 200, "{v}");
    let s = &v["aggregations"]["s"];
    assert_eq!(s["doc_count"], 3);
    assert_eq!(s["bg_count"], 5);
    assert_eq!(s["buckets"][0]["key"], "a");
    assert_eq!(s["buckets"][0]["doc_count"], 2);
    assert_eq!(s["buckets"][0]["bg_count"], 2);
    // (2/3 - 2/5) * ((2/3) / (2/5))
    let jlh = (2.0f64 / 3.0 - 0.4) * ((2.0 / 3.0) / 0.4);
    assert!((s["buckets"][0]["score"].as_f64().unwrap() - jlh).abs() < 1e-12);

    // rate outside a date_histogram is an error.
    let body = json!({"size": 0, "aggs": {"r": {"rate": {"field": "qty"}}}});
    let (st, v) = call(&c, reqwest::Method::POST, format!("{base}/sales/_search"), Some(body.to_string()), "application/json").await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["root_cause"][0]["type"], "illegal_argument_exception");

    // t_test
    let body = json!({"size": 0, "aggs": {"t": {"t_test": {"a": {"field": "price", "filter": {"term": {"tag": "a"}}}, "b": {"field": "price", "filter": {"term": {"tag": "b"}}}, "type": "heteroscedastic"}}}});
    let (st, v) = call(&c, reqwest::Method::POST, format!("{base}/sales/_search"), Some(body.to_string()), "application/json").await;
    assert_eq!(st, 200, "{v}");
    let p = v["aggregations"]["t"]["value"].as_f64().unwrap();
    // Welch: means 15 vs 35, both var 50 → t = -2√2, df = 2 → p = 1 - t/√(t²+2).
    let t = 8f64.sqrt();
    assert!((p - (1.0 - t / (t * t + 2.0).sqrt())).abs() < 1e-9, "{p}");
}
