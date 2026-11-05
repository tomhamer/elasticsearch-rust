//! Integer-keyed `_id` blocks and delta-packed timestamps (segment format
//! v34): a logsdb index of auto (k-ordered) ids and index-sorted
//! `@timestamp` answers gets, multi-gets, `ids` and `terms` queries,
//! overwrites and deletes by id, and timestamp ranges, sorts and histograms
//! exactly, refreshed and after a merge.
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
async fn k_ordered_ids_and_timestamps_read_back() {
    use elasticsearch_engine::rustic_core::codec;
    codec::apply_cluster_segment_format(codec::VERSION);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { let _ = elasticsearch_http::serve_listener(node, listener).await; });
    let c = Client::new();
    let m = reqwest::Method::POST;
    let r = call(&c, &base, reqwest::Method::PUT, "/lg", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mode": "logsdb", "refresh_interval": "-1", "index.max_result_window": 50000},
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}, "msg": {"type": "keyword"}}}
    }).to_string()), "application/json").await;
    assert_eq!(r["acknowledged"], json!(true), "{r}");
    // Seconds in milliseconds, a few docs a second, a gap now and then.
    let n = 9000usize;
    let mut ts = 1_700_000_000_000i64;
    let mut stamps = Vec::with_capacity(n);
    for i in 0..n {
        if i % 3 == 0 {
            ts += 1000 * if i % 997 == 0 { 60 } else { 1 };
        }
        stamps.push(ts);
    }
    let mut ids: Vec<String> = Vec::with_capacity(n);
    for chunk in 0..3 {
        let mut bulk = String::new();
        for i in (chunk * n / 3)..((chunk + 1) * n / 3) {
            bulk.push_str("{\"index\":{}}\n");
            bulk.push_str(&json!({"@timestamp": stamps[i], "n": i, "msg": format!("m{}", i % 17)}).to_string());
            bulk.push('\n');
        }
        let r = call(&c, &base, m.clone(), "/lg/_bulk", Some(bulk), "application/x-ndjson").await;
        assert_eq!(r["errors"], json!(false), "{r}");
        for item in r["items"].as_array().unwrap() {
            ids.push(item["index"]["_id"].as_str().unwrap().to_string());
        }
        call(&c, &base, m.clone(), "/lg/_refresh", None, "").await;
    }
    assert!(ids.iter().all(|id| id.len() == 20), "auto ids");
    let search = |body: Value| {
        let (c, base) = (c.clone(), base.clone());
        async move { call(&c, &base, reqwest::Method::POST, "/lg/_search", Some(body.to_string()), "application/json").await }
    };
    let check = |stage: &'static str, live: Vec<bool>, overwritten: Option<usize>| {
        let (ids, stamps, c, base) = (ids.clone(), stamps.clone(), c.clone(), base.clone());
        let search = &search;
        async move {
            let alive = |i: usize| live[i];
            let n_of = |v: &Value| -> Vec<i64> { v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{stage}: {v}")).iter().map(|h| h["_source"]["n"].as_i64().unwrap()).collect() };
            // Gets.
            for i in (0..n).step_by(37) {
                let v = call(&c, &base, reqwest::Method::GET, &format!("/lg/_doc/{}", ids[i]), None, "").await;
                assert_eq!(v["found"], json!(alive(i)), "{stage}: get {i} {v}");
                if alive(i) {
                    let want = if overwritten == Some(i) { -1 } else { i as i64 };
                    assert_eq!(v["_source"]["n"].as_i64(), Some(want), "{stage}: get {i}");
                    // Synthetic source renders the date in its format.
                    let iso = chrono::DateTime::from_timestamp_millis(stamps[i]).unwrap().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
                    assert_eq!(v["_source"]["@timestamp"].as_str(), Some(iso.as_str()), "{stage}: get {i} {v}");
                }
            }
            // Gets of ids that are not there (one character off; k-ordered
            // ids differ in their last characters, so skip real ones).
            let known: std::collections::HashSet<&String> = ids.iter().collect();
            for at in [19usize, 18, 15, 3] {
                for ch in [b'A', b'-', b'_', b'z', b'0'] {
                    let mut miss = ids[100].clone().into_bytes();
                    miss[at] = ch;
                    let miss = String::from_utf8(miss).unwrap();
                    if known.contains(&miss) { continue; }
                    let v = call(&c, &base, reqwest::Method::GET, &format!("/lg/_doc/{miss}"), None, "").await;
                    assert_eq!(v["found"], json!(false), "{stage}: miss {miss}");
                }
            }
            // Multi-get.
            let want: Vec<usize> = (0..n).step_by(91).collect();
            let v = call(&c, &base, reqwest::Method::POST, "/lg/_mget", Some(json!({"ids": want.iter().map(|&i| &ids[i]).collect::<Vec<_>>()}).to_string()), "application/json").await;
            let docs = v["docs"].as_array().unwrap();
            for (d, &i) in docs.iter().zip(&want) {
                assert_eq!(d["found"], json!(alive(i)), "{stage}: mget {i}");
            }
            // ids and terms queries over many ids.
            let some: Vec<&String> = (0..n).step_by(7).map(|i| &ids[i]).collect();
            let expect = (0..n).step_by(7).filter(|&i| alive(i)).count();
            let v = search(json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": some}}})).await;
            assert_eq!(v["hits"]["total"]["value"].as_u64(), Some(expect as u64), "{stage}: ids");
            let v = search(json!({"size": 0, "track_total_hits": true, "query": {"terms": {"_id": some}}})).await;
            assert_eq!(v["hits"]["total"]["value"].as_u64(), Some(expect as u64), "{stage}: terms _id");
            // Timestamp ranges.
            for (lo, hi) in [(0usize, n - 1), (100, 200), (4000, 4003), (8990, 8999)] {
                let v = search(json!({"size": 0, "track_total_hits": true, "query": {"range": {"@timestamp": {"gte": stamps[lo], "lte": stamps[hi]}}}})).await;
                let want = (0..n).filter(|&i| alive(i) && stamps[i] >= stamps[lo] && stamps[i] <= stamps[hi]).count();
                assert_eq!(v["hits"]["total"]["value"].as_u64(), Some(want as u64), "{stage}: range {lo}..{hi}");
            }
            // Sorts by timestamp (ties by n).
            for order in ["asc", "desc"] {
                let v = search(json!({"size": 300, "query": {"match_all": {}}, "sort": [{"@timestamp": order}, {"n": "asc"}]})).await;
                let mut want: Vec<usize> = (0..n).filter(|&i| alive(i)).collect();
                want.sort_by(|&a, &b| if order == "asc" { stamps[a].cmp(&stamps[b]) } else { stamps[b].cmp(&stamps[a]) }.then_with(|| {
                    let k = |i: usize| if overwritten == Some(i) { -1 } else { i as i64 };
                    k(a).cmp(&k(b))
                }));
                let want: Vec<i64> = want[..300].iter().map(|&i| if overwritten == Some(i) { -1 } else { i as i64 }).collect();
                assert_eq!(n_of(&v), want, "{stage}: sort {order}");
            }
            // A minute histogram.
            let v = search(json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1m"}}, "lo": {"min": {"field": "@timestamp"}}, "hi": {"max": {"field": "@timestamp"}}}})).await;
            let mut counts: std::collections::BTreeMap<i64, u64> = Default::default();
            for i in (0..n).filter(|&i| alive(i)) {
                *counts.entry(stamps[i].div_euclid(60_000) * 60_000).or_default() += 1;
            }
            for b in v["aggregations"]["h"]["buckets"].as_array().unwrap() {
                assert_eq!(b["doc_count"].as_u64().unwrap(), counts.get(&b["key"].as_i64().unwrap()).copied().unwrap_or(0), "{stage}: {b}");
            }
            let first = (0..n).find(|&i| alive(i)).unwrap();
            let last = (0..n).rev().find(|&i| alive(i)).unwrap();
            assert_eq!(v["aggregations"]["lo"]["value"].as_f64(), Some(stamps[first] as f64), "{stage}");
            assert_eq!(v["aggregations"]["hi"]["value"].as_f64(), Some(stamps[last] as f64), "{stage}");
        }
    };
    let mut live = vec![true; n];
    check("refreshed", live.clone(), None).await;
    call(&c, &base, m.clone(), "/lg/_forcemerge?max_num_segments=1", None, "").await;
    call(&c, &base, m.clone(), "/lg/_refresh", None, "").await;
    check("merged", live.clone(), None).await;
    // Deletes and an overwrite by id, then another merge.
    for i in [0usize, 37, 4242, 8999, 5550] {
        let v = call(&c, &base, reqwest::Method::DELETE, &format!("/lg/_doc/{}", ids[i]), None, "").await;
        assert_eq!(v["result"], json!("deleted"), "{v}");
        live[i] = false;
    }
    let v = call(&c, &base, reqwest::Method::PUT, &format!("/lg/_doc/{}", ids[74]), Some(json!({"@timestamp": stamps[74], "n": -1, "msg": "x"}).to_string()), "application/json").await;
    assert_eq!(v["result"], json!("updated"), "{v}");
    call(&c, &base, m.clone(), "/lg/_refresh", None, "").await;
    check("deleted", live.clone(), Some(74)).await;
    call(&c, &base, m.clone(), "/lg/_forcemerge?max_num_segments=1", None, "").await;
    call(&c, &base, m.clone(), "/lg/_refresh", None, "").await;
    check("remerged", live, Some(74)).await;
}
