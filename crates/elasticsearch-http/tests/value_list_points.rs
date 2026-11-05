//! Value-list points (segment format v34): a numeric field of few distinct
//! values (an http status) keeps per-value doc lists as its points index.
//! Term, terms and range queries, counts, sorts and aggregations on it
//! answer exactly what a scan of the documents gives, refreshed and merged.
//!
//! The test raises this process's segment format floor to the current
//! format (each integration test file is its own process).

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};

async fn call(c: &Client, base: &str, m: reqwest::Method, p: &str, body: Option<String>, ct: &str) -> Value {
    let mut r = c.request(m, format!("{base}{p}"));
    if let Some(b) = body {
        r = r.header("content-type", ct).body(b);
    }
    let t = r.send().await.unwrap().text().await.unwrap();
    serde_json::from_str(&t).unwrap_or(Value::String(t))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn low_cardinality_points_answer_like_a_scan() {
    use elasticsearch_engine::rustic_core::codec;
    codec::apply_cluster_segment_format(codec::VERSION);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(node, listener).await; });
    let c = Client::new();
    let r = call(&c, &base, reqwest::Method::PUT, "/vl", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.max_result_window": 20000},
        "mappings": {"properties": {"status": {"type": "integer"}, "rate": {"type": "double"}, "n": {"type": "long"}}}
    }).to_string()), "application/json").await;
    assert_eq!(r["acknowledged"], json!(true), "{r}");
    let codes = [200i64, 304, 404, 500, 206, 302, 403, 400];
    let n = 12_000usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
    let docs: Vec<(i64, f64)> = (0..n).map(|i| {
        let s = if i == 4242 { 501 } else if rnd(100) < 85 { 200 } else { codes[1 + rnd(7) as usize] };
        (s, [0.5, 1.25, 3.0][rnd(3) as usize])
    }).collect();
    for chunk in 0..3 {
        let mut bulk = String::new();
        for i in (chunk * n / 3)..((chunk + 1) * n / 3) {
            bulk.push_str(&json!({"index": {"_id": format!("d{i}")}}).to_string());
            bulk.push('\n');
            bulk.push_str(&json!({"status": docs[i].0, "rate": docs[i].1, "n": i}).to_string());
            bulk.push('\n');
        }
        let r = call(&c, &base, reqwest::Method::POST, "/vl/_bulk", Some(bulk), "application/x-ndjson").await;
        assert_eq!(r["errors"], json!(false));
        call(&c, &base, reqwest::Method::POST, "/vl/_refresh", None, "").await;
    }
    let search = |body: Value| {
        let (c, base) = (c.clone(), base.clone());
        async move { call(&c, &base, reqwest::Method::POST, "/vl/_search", Some(body.to_string()), "application/json").await }
    };
    let check = |stage: &'static str, live: Vec<bool>| {
        let docs = docs.clone();
        let search = &search;
        async move {
            let alive = |i: usize| live[i];
            let ids = |v: &Value| -> Vec<usize> { v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap()[1..].parse().unwrap()).collect() };
            let queries: Vec<(Value, Box<dyn Fn(i64, f64) -> bool>)> = vec![
                (json!({"term": {"status": 404}}), Box::new(|s, _| s == 404)),
                (json!({"term": {"status": 501}}), Box::new(|s, _| s == 501)),
                (json!({"terms": {"status": [304, 500, 999]}}), Box::new(|s, _| s == 304 || s == 500)),
                (json!({"range": {"status": {"gte": 300, "lte": 404}}}), Box::new(|s, _| (300..=404).contains(&s))),
                (json!({"range": {"status": {"gt": 200}}}), Box::new(|s, _| s > 200)),
                (json!({"range": {"status": {"lt": 300}}}), Box::new(|s, _| s < 300)),
                (json!({"range": {"status": {"gte": 201, "lt": 206}}}), Box::new(|_, _| false)),
                (json!({"range": {"rate": {"gte": 1.0}}}), Box::new(|_, r| r >= 1.0)),
                (json!({"bool": {"filter": [{"term": {"status": 200}}, {"term": {"rate": 3.0}}]}}), Box::new(|s, r| s == 200 && r == 3.0)),
            ];
            for (q, pred) in &queries {
                let v = search(json!({"size": n, "track_total_hits": true, "query": q, "sort": [{"n": "asc"}]})).await;
                let want: Vec<usize> = (0..n).filter(|&i| alive(i) && pred(docs[i].0, docs[i].1)).collect();
                assert_eq!(v["hits"]["total"]["value"], json!(want.len()), "{stage}: {q}");
                assert_eq!(ids(&v), want, "{stage}: {q}");
            }
            // Sorts: by status (ties by n), and status alone (ties in index order).
            for order in ["asc", "desc"] {
                let v = search(json!({"size": 500, "query": {"match_all": {}}, "sort": [{"status": order}, {"n": "asc"}]})).await;
                let mut want: Vec<usize> = (0..n).filter(|&i| alive(i)).collect();
                want.sort_by(|&a, &b| if order == "asc" { docs[a].0.cmp(&docs[b].0) } else { docs[b].0.cmp(&docs[a].0) }.then(a.cmp(&b)));
                assert_eq!(ids(&v), want[..500].to_vec(), "{stage}: sort {order}");
                let v = search(json!({"size": 1000, "query": {"match_all": {}}, "sort": [{"status": order}]})).await;
                let got: Vec<i64> = ids(&v).iter().map(|&i| docs[i].0).collect();
                let want_vals: Vec<i64> = want[..1000].iter().map(|&i| docs[i].0).collect();
                assert_eq!(got, want_vals, "{stage}: sort {order} values");
                let v = search(json!({"size": 50, "query": {"match_all": {}}, "sort": [{"status": order}], "search_after": [304]})).await;
                let got: Vec<i64> = ids(&v).iter().map(|&i| docs[i].0).collect();
                assert!(got.iter().all(|&s| if order == "asc" { s > 304 } else { s < 304 }), "{stage}: search_after {order} {got:?}");
            }
            // Aggregations: terms counts, min / max, range buckets.
            let v = search(json!({"size": 0, "aggs": {
                "codes": {"terms": {"field": "status", "size": 20}},
                "lo": {"min": {"field": "status"}}, "hi": {"max": {"field": "status"}},
                "r": {"range": {"field": "status", "ranges": [{"to": 300}, {"from": 300, "to": 450}, {"from": 450}]}}
            }})).await;
            let mut counts: std::collections::BTreeMap<i64, u64> = Default::default();
            for i in (0..n).filter(|&i| alive(i)) { *counts.entry(docs[i].0).or_default() += 1; }
            for b in v["aggregations"]["codes"]["buckets"].as_array().unwrap() {
                assert_eq!(b["doc_count"].as_u64(), counts.get(&b["key"].as_i64().unwrap()).copied(), "{stage}: {b}");
            }
            assert_eq!(v["aggregations"]["lo"]["value"].as_f64(), counts.keys().next().map(|&k| k as f64), "{stage}");
            assert_eq!(v["aggregations"]["hi"]["value"].as_f64(), counts.keys().last().map(|&k| k as f64), "{stage}");
            let r = &v["aggregations"]["r"]["buckets"];
            let cnt = |p: &dyn Fn(i64) -> bool| (0..n).filter(|&i| alive(i) && p(docs[i].0)).count() as u64;
            assert_eq!(r[0]["doc_count"].as_u64(), Some(cnt(&|s| s < 300)), "{stage}");
            assert_eq!(r[1]["doc_count"].as_u64(), Some(cnt(&|s| (300..450).contains(&s))), "{stage}");
            assert_eq!(r[2]["doc_count"].as_u64(), Some(cnt(&|s| s >= 450)), "{stage}");
        }
    };
    let mut live = vec![true; n];
    check("refreshed", live.clone()).await;
    call(&c, &base, reqwest::Method::POST, "/vl/_forcemerge?max_num_segments=1", None, "").await;
    call(&c, &base, reqwest::Method::POST, "/vl/_refresh", None, "").await;
    check("merged", live.clone()).await;
    for i in [3usize, 4242, 5000, 11_999] {
        call(&c, &base, reqwest::Method::DELETE, &format!("/vl/_doc/d{i}"), None, "").await;
        live[i] = false;
    }
    call(&c, &base, reqwest::Method::POST, "/vl/_refresh", None, "").await;
    check("deleted", live).await;
}
