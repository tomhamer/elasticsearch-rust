//! A time series index (Rally tsdb's shape: series of several metricsets,
//! keyword dimensions and labels present on some series only, metrics
//! present on one metricset's series) answers the same before and after a
//! merge into the current store (v36: keyword ordinals packed per block
//! with missing docs, compact metric columns as presence runs with values
//! packed per block; v37: booleans that know their docs with a value, a
//! keyword of one term every doc has without ordinals, two-value 1-bit
//! blocks): synthetic `_source` (a boolean only where the document has
//! one), `_id` lookups (an overwrite of an existing id updates it), exists
//! / term / range queries (`exists` on a doc-values boolean), stats,
//! `date_histogram` over a `match_all` (counted from the timestamp
//! column's blocks) and `time_series` sums.
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

/// Doc `i` of series `s` at minute `t`.
fn doc(s: usize, t: usize) -> Value {
    let ts = format!("2021-04-28T{:02}:{:02}:{:02}.000Z", 10 + t / 60, t % 60, (s * 7) % 60);
    match s % 3 {
        // A container series: dimensions, a label, gauges and a counter.
        0 => json!({"@timestamp": ts, "metricset": {"name": "container"}, "ecs": {"version": "1.4.0"},
            "kubernetes": {"node": {"name": format!("node-{}", s % 5)}, "container": {"name": format!("c-{s}"), "ready": s % 2 == 0,
                "memory": {"usage": {"bytes": 1_000_000_000i64 + (s as i64) * 1_000 + (t as i64 % 7) * 4096}},
                "cpu": {"usage": {"core": {"ns": (t as i64) * 10_000 + s as i64}}}},
                "labels": {"app": format!("app-{}", s % 4)}}}),
        // A pod series: other metrics, no container fields.
        1 => json!({"@timestamp": ts, "metricset": {"name": "pod"}, "ecs": {"version": "1.4.0"},
            "kubernetes": {"node": {"name": format!("node-{}", s % 5)}, "pod": {"name": format!("p-{s}"),
                "network": {"rx": {"bytes": (t as i64) * 333 + s as i64}}}}}),
        // A node series: almost nothing.
        _ => json!({"@timestamp": ts, "metricset": {"name": "node"}, "ecs": {"version": "1.4.0"}, "kubernetes": {"node": {"name": format!("node-{s}")}},
            "event": {"duration": 1_000_000i64 + (t as i64) * 17}}),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tsdb_answers_alike_before_and_after_a_v36_merge() {
    use elasticsearch_engine::rustic_core::codec;
    codec::apply_cluster_segment_format(codec::VERSION);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(node, listener).await; });
    let c = Client::new();
    let kw_dim = json!({"type": "keyword", "time_series_dimension": true});
    let r = call(&c, &base, reqwest::Method::PUT, "/ts", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "mode": "time_series",
            "routing_path": ["metricset.name", "kubernetes.node.name", "kubernetes.container.name", "kubernetes.pod.name"],
            "time_series": {"start_time": "2000-01-01T00:00:00.000Z", "end_time": "9999-01-01T00:00:00.000Z"},
            "mapping": {"source": {"mode": "synthetic"}}, "max_result_window": 50000},
        "mappings": {"properties": {
            "@timestamp": {"type": "date"},
            "metricset": {"properties": {"name": kw_dim}},
            "ecs": {"properties": {"version": {"type": "keyword"}}},
            "event": {"properties": {"duration": {"type": "long"}}},
            "kubernetes": {"properties": {
                "node": {"properties": {"name": kw_dim}},
                "pod": {"properties": {"name": kw_dim, "network": {"properties": {"rx": {"properties": {"bytes": {"type": "long", "time_series_metric": "counter"}}}}}}},
                "labels": {"properties": {"app": {"type": "keyword"}}},
                "container": {"properties": {"name": kw_dim, "ready": {"type": "boolean"},
                    "memory": {"properties": {"usage": {"properties": {"bytes": {"type": "long", "time_series_metric": "gauge"}}}}},
                    "cpu": {"properties": {"usage": {"properties": {"core": {"properties": {"ns": {"type": "long", "time_series_metric": "counter"}}}}}}}}}}}}}
    }).to_string()), "application/json").await;
    assert_eq!(r["acknowledged"], json!(true), "{r}");
    let (series, minutes) = (60usize, 300usize);
    // Three bulks of interleaved time ranges (Rally's clients), each its
    // own segment.
    for part in 0..3 {
        let mut bulk = String::new();
        for t in (part * minutes / 3)..((part + 1) * minutes / 3) {
            for s in 0..series {
                bulk.push_str("{\"index\":{}}\n");
                bulk.push_str(&doc(s, t).to_string());
                bulk.push('\n');
            }
        }
        let r = call(&c, &base, reqwest::Method::POST, "/ts/_bulk", Some(bulk), "application/x-ndjson").await;
        assert_eq!(r["errors"], json!(false), "{}", &r.to_string()[..r.to_string().len().min(2000)]);
        call(&c, &base, reqwest::Method::POST, "/ts/_refresh", None, "").await;
    }
    let n = series * minutes;
    // Overwrites of existing ids (every id is looked up in every segment):
    // updated, not added.
    let mut bulk = String::new();
    for (s, t) in [(0usize, 5usize), (1, 150), (2, 299), (31, 77)] {
        bulk.push_str("{\"index\":{}}\n");
        bulk.push_str(&doc(s, t).to_string());
        bulk.push('\n');
    }
    let r = call(&c, &base, reqwest::Method::POST, "/ts/_bulk", Some(bulk), "application/x-ndjson").await;
    assert_eq!(r["errors"], json!(false), "{r}");
    for item in r["items"].as_array().unwrap() {
        assert_eq!(item["index"]["result"], json!("updated"), "{item}");
    }
    call(&c, &base, reqwest::Method::POST, "/ts/_refresh", None, "").await;
    let search = |body: Value| {
        let (c, base) = (c.clone(), base.clone());
        async move { call(&c, &base, reqwest::Method::POST, "/ts/_search", Some(body.to_string()), "application/json").await }
    };
    let snapshot = || async {
        let mut out = Vec::new();
        out.push(search(json!({"size": 0, "track_total_hits": true, "aggs": {"h": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1h"}}}})).await["aggregations"].clone());
        out.push(search(json!({"size": 0, "aggs": {"d": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1d"}}}})).await["aggregations"].clone());
        out.push(search(json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1h"},
            "aggs": {"ts": {"time_series": {"keyed": false}, "aggs": {"m": {"sum": {"field": "kubernetes.container.memory.usage.bytes"}}}},
                     "min": {"min_bucket": {"buckets_path": "ts>m"}}}}}})).await["aggregations"].clone());
        for f in ["kubernetes.container.name", "kubernetes.labels.app", "kubernetes.pod.network.rx.bytes", "event.duration", "kubernetes.container.cpu.usage.core.ns", "kubernetes.container.ready", "ecs.version"] {
            out.push(search(json!({"size": 0, "track_total_hits": true, "query": {"exists": {"field": f}}})).await["hits"]["total"].clone());
        }
        out.push(search(json!({"size": 0, "aggs": {"t": {"terms": {"field": "kubernetes.labels.app", "size": 10}}, "s": {"stats": {"field": "kubernetes.container.memory.usage.bytes"}},
            "r": {"stats": {"field": "kubernetes.pod.network.rx.bytes"}}, "e": {"stats": {"field": "event.duration"}}}})).await["aggregations"].clone());
        out.push(search(json!({"size": 0, "track_total_hits": true, "query": {"range": {"kubernetes.container.memory.usage.bytes": {"gte": 1_000_020_000i64, "lt": 1_000_040_000i64}}}})).await["hits"]["total"].clone());
        out.push(search(json!({"size": 0, "track_total_hits": true, "query": {"term": {"kubernetes.container.name": "c-33"}}})).await["hits"]["total"].clone());
        out.push(search(json!({"size": 0, "track_total_hits": true, "query": {"term": {"kubernetes.container.ready": true}}})).await["hits"]["total"].clone());
        out.push(search(json!({"size": 0, "aggs": {"r": {"terms": {"field": "kubernetes.container.ready"}}, "v": {"terms": {"field": "ecs.version"}}}})).await["aggregations"].clone());
        // ES|QL's rows (every column of the synthetic sources, read through
        // a batch's opened columns): per column its non-null count and the
        // sum of its numbers / true values.
        let r = call(&c, &base, reqwest::Method::POST, "/_query", Some(json!({"query": "FROM ts | WHERE metricset.name == \"container\" | LIMIT 10000"}).to_string()), "application/json").await;
        let cols = r["columns"].as_array().unwrap_or_else(|| panic!("{r}"));
        let rows = r["values"].as_array().unwrap();
        let mut stats: Vec<(String, u64, f64)> = cols.iter().map(|c| (c["name"].as_str().unwrap().to_string(), 0, 0.0)).collect();
        for row in rows {
            for (i, v) in row.as_array().unwrap().iter().enumerate() {
                if !v.is_null() {
                    stats[i].1 += 1;
                    stats[i].2 += v.as_f64().unwrap_or(if v == &json!(true) { 1.0 } else { 0.0 });
                }
            }
        }
        stats.sort_by(|a, b| a.0.cmp(&b.0));
        out.push(json!({"rows": rows.len(), "stats": stats}));
        // Every document's synthetic source, by id.
        let all = search(json!({"size": n, "query": {"match_all": {}}, "sort": [{"@timestamp": "asc"}, {"_tsid": "asc"}]})).await;
        out.push(json!(all["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_source"]])).collect::<Vec<_>>()));
        out
    };
    let before = snapshot().await;
    // Sanity: the counts, and every source is its document.
    assert_eq!(before[0]["h"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum::<u64>(), n as u64);
    assert_eq!(before[1]["d"]["buckets"][0]["doc_count"], json!(n));
    assert_eq!(before[3], json!({"value": n / 3, "relation": "eq"}));
    // `exists` on the boolean: the container series only; the constant
    // keyword: every doc.
    assert_eq!(before[8], json!({"value": n / 3, "relation": "eq"}));
    assert_eq!(before[9], json!({"value": n, "relation": "eq"}));
    assert_eq!(before[13], json!({"value": n / 6, "relation": "eq"}));
    assert_eq!(before[14]["r"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum::<u64>(), n as u64 / 3, "{}", before[14]);
    // ES|QL: every container row, each column as the documents have it.
    let esql = &before[15];
    assert_eq!(esql["rows"], json!(n / 3));
    let stat = |name: &str| esql["stats"].as_array().unwrap().iter().find(|s| s[0] == json!(name)).cloned().unwrap_or_else(|| panic!("{name}"));
    assert_eq!(stat("kubernetes.container.ready")[1], json!(n / 3));
    assert_eq!(stat("kubernetes.container.ready")[2], json!((n / 6) as f64));
    assert_eq!(stat("ecs.version")[1], json!(n / 3));
    assert_eq!(stat("kubernetes.node.name")[1], json!(n / 3));
    let mem: f64 = (0..minutes).flat_map(|t| (0..series).filter(|s| s % 3 == 0).map(move |s| doc(s, t))).map(|d| d["kubernetes"]["container"]["memory"]["usage"]["bytes"].as_f64().unwrap()).sum();
    assert_eq!(stat("kubernetes.container.memory.usage.bytes")[2], json!(mem));
    let hits = before.last().unwrap().as_array().unwrap();
    assert_eq!(hits.len(), n);
    let mut want: Vec<Value> = (0..minutes).flat_map(|t| (0..series).map(move |s| doc(s, t))).collect();
    let mut got: Vec<Value> = hits.iter().map(|h| h[1].clone()).collect();
    let key = |v: &Value| v.to_string();
    want.sort_by_key(key);
    got.sort_by_key(key);
    assert!(want == got, "synthetic sources differ from the documents");
    let r = call(&c, &base, reqwest::Method::POST, "/ts/_forcemerge?max_num_segments=1", None, "").await;
    assert!(r.get("error").is_none(), "{r}");
    call(&c, &base, reqwest::Method::POST, "/ts/_refresh", None, "").await;
    let segs = call(&c, &base, reqwest::Method::GET, "/ts/_segments", None, "").await;
    assert_eq!(segs["indices"]["ts"]["shards"]["0"][0]["num_search_segments"], json!(1), "{segs}");
    let after = snapshot().await;
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        assert!(a == b, "answer {i} changed by the merge:\n{}\nvs\n{}", &a.to_string()[..a.to_string().len().min(1500)], &b.to_string()[..b.to_string().len().min(1500)]);
    }
    // Ids still resolve after the merge: overwrites update.
    let mut bulk = String::new();
    for (s, t) in [(3usize, 3usize), (59, 299)] {
        bulk.push_str("{\"index\":{}}\n");
        bulk.push_str(&doc(s, t).to_string());
        bulk.push('\n');
    }
    let r = call(&c, &base, reqwest::Method::POST, "/ts/_bulk", Some(bulk), "application/x-ndjson").await;
    for item in r["items"].as_array().unwrap() {
        assert_eq!(item["index"]["result"], json!("updated"), "{item}");
    }
}
