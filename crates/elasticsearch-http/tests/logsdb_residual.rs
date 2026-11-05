//! Empty synthetic residuals (segment format v34): a logsdb document whose
//! columns rebuild every field it holds stores `_source` as a two-byte
//! marker, its fields told back by which columns hold a value. Every
//! source reads back byte-identical to the canonical synthetic source
//! (field order, missing fields, nulls, arrays, dotted keys, dynamic
//! fields) through gets and searches, before and after a refresh, a flush,
//! a merge and reopening the node.
//!
//! The test raises this process's segment format floor to the current
//! format (a cluster's floor; each integration test file is its own
//! process).

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;

struct Es {
    base: String,
    c: Client,
    task: tokio::task::JoinHandle<()>,
}

async fn serve(node: Arc<Node>) -> Es {
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new(), task }
}

impl Es {
    async fn call(&self, m: reqwest::Method, p: &str, body: Option<String>, ct: &str) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", ct).body(b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
    }
}

fn docs() -> Vec<Value> {
    let mut out = Vec::new();
    for i in 0..300i64 {
        let d = match i % 6 {
            // Every mapped field, in source order.
            0 => json!({"@timestamp": 894225602 + i, "clientip": format!("10.{}.0.{}", i % 7, i % 250), "request": format!("GET /images/{}.gif HTTP/1.0", i % 13), "status": 200, "size": 1000 + i}),
            // Keys in another order; a missing field.
            1 => json!({"size": i, "status": 304, "@timestamp": 894225602 + i, "request": "GET /a HTTP/1.0"}),
            // Only a timestamp and a dynamic keyword.
            2 => json!({"@timestamp": 894225602 + i, "extra": format!("x{}", i % 4)}),
            // Nested object and dotted key.
            3 => json!({"@timestamp": 894225602 + i, "host": {"name": format!("h{}", i % 3)}, "ratio": 0.5 * (i % 4) as f64, "flag": i % 2 == 0}),
            4 => json!({"@timestamp": 894225602 + i, "host.name": format!("h{}", i % 3), "status": 404}),
            // What a column cannot render: nulls, arrays, a long request.
            _ => match i % 18 {
                5 => json!({"@timestamp": 894225602 + i, "status": null, "size": [3, 1, 2]}),
                11 => json!({"@timestamp": 894225602 + i, "request": "GET /a/very/long/path/that/is/over/the/limit/of/the/raw/keyword HTTP/1.0"}),
                _ => json!({"@timestamp": 894225602 + i, "clientip": ["10.0.0.2", "10.0.0.1"], "flag": false}),
            },
        };
        out.push(d);
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_residuals_rebuild_sources_byte_for_byte() {
    use elasticsearch_engine::rustic_core::codec;
    codec::apply_cluster_segment_format(codec::VERSION);
    assert!(codec::write_version() >= codec::EMPTY_RESIDUAL_SINCE);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let es = serve(node.clone()).await;
    let (st, r) = es.call(reqwest::Method::PUT, "/lr", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mode": "logsdb", "refresh_interval": "-1"},
        "mappings": {"properties": {
            "@timestamp": {"type": "date", "format": "epoch_second"},
            "clientip": {"type": "ip"},
            "request": {"type": "match_only_text", "fields": {"raw": {"type": "keyword", "ignore_above": 32}}},
            "status": {"type": "integer"}, "size": {"type": "integer"},
            "ratio": {"type": "double"}, "flag": {"type": "boolean"},
            "host": {"properties": {"name": {"type": "keyword"}}}
        }}
    }).to_string()), "application/json").await;
    assert_eq!(st, 200, "{r}");
    let docs = docs();
    let mut bulk = String::new();
    for (i, d) in docs.iter().enumerate() {
        bulk.push_str(&json!({"index": {"_id": format!("d{i}")}}).to_string());
        bulk.push('\n');
        bulk.push_str(&d.to_string());
        bulk.push('\n');
    }
    let (st, r) = es.call(reqwest::Method::POST, "/lr/_bulk", Some(bulk), "application/x-ndjson").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["errors"], json!(false), "{r}");
    let expected: Vec<String> = {
        let svc = node.get_index("lr").unwrap();
        let meta = svc.meta();
        let lk = svc.lookup();
        docs.iter().map(|d| elasticsearch_engine::synthetic::synthesize(d, &meta.mappings, &lk, elasticsearch_engine::synthetic::Keep::of_index(&meta.settings)).unwrap().0.to_string()).collect()
    };
    let check = |es: &Es, when: &'static str| {
        let expected = expected.clone();
        let base = es.base.clone();
        let c = es.c.clone();
        async move {
            for (i, want) in expected.iter().enumerate() {
                let t = c.get(format!("{base}/lr/_doc/d{i}")).send().await.unwrap().text().await.unwrap();
                let v: Value = serde_json::from_str(&t).unwrap();
                assert_eq!(v["_source"].to_string(), *want, "{when}: get d{i}");
            }
            let t = c.post(format!("{base}/lr/_search")).header("content-type", "application/json")
                .body(json!({"size": 1000, "query": {"match_all": {}}}).to_string()).send().await.unwrap().text().await.unwrap();
            let v: Value = serde_json::from_str(&t).unwrap();
            let hits = v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{when}: {v}"));
            if when != "memtable" {
                assert_eq!(hits.len(), expected.len(), "{when}");
            }
            for h in hits {
                let i: usize = h["_id"].as_str().unwrap()[1..].parse().unwrap();
                assert_eq!(h["_source"].to_string(), expected[i], "{when}: hit d{i}");
            }
        }
    };
    // JSON numbers in date queries are epoch milliseconds whatever the
    // field's format (ES 9); strings and an explicit format use the format.
    {
        es.call(reqwest::Method::POST, "/lr/_refresh", None, "").await;
        let count = |q: Value| { let es = &es; async move { es.call(reqwest::Method::POST, "/lr/_count", Some(json!({"query": q}).to_string()), "application/json").await.1["count"].clone() } };
        assert_eq!(count(json!({"term": {"@timestamp": 894225602000i64}})).await, json!(1));
        assert_eq!(count(json!({"term": {"@timestamp": 894225602}})).await, json!(0));
        assert_eq!(count(json!({"term": {"@timestamp": "894225602"}})).await, json!(1));
        assert_eq!(count(json!({"range": {"@timestamp": {"gte": 894225602000i64, "lt": 894225612000i64}}})).await, json!(10));
        assert_eq!(count(json!({"range": {"@timestamp": {"gte": 894225602, "lt": 894225612}}})).await, json!(0));
        assert_eq!(count(json!({"range": {"@timestamp": {"gte": 894225602, "lt": 894225612, "format": "epoch_second"}}})).await, json!(10));
        assert_eq!(count(json!({"range": {"@timestamp": {"gte": "894225602", "lt": "894225612"}}})).await, json!(10));
        let (st, _) = es.call(reqwest::Method::POST, "/lr/_count", Some(json!({"query": {"range": {"@timestamp": {"gte": 894225602000.5}}}}).to_string()), "application/json").await;
        assert_eq!(st, 400);
    }
    check(&es, "memtable").await;
    es.call(reqwest::Method::POST, "/lr/_refresh", None, "").await;
    check(&es, "refreshed").await;
    es.call(reqwest::Method::POST, "/lr/_flush", None, "").await;
    check(&es, "flushed").await;
    // Most documents store the marker.
    let empty = {
        let svc = node.get_index("lr").unwrap();
        let sh = svc.local_shards().remove(0);
        let core = sh.shard.core.read();
        let mut n = 0;
        for seg in core.segments() {
            for d in 0..seg.num_docs {
                if let Ok(Some(doc)) = seg.get_doc(d) {
                    if doc.get(elasticsearch_engine::mapping::SOURCE_FIELD).and_then(|v| v.as_text()) == Some(elasticsearch_engine::synthetic::EMPTY_RESIDUAL) {
                        n += 1;
                    }
                }
            }
        }
        n
    };
    assert!(empty >= 150, "{empty} of {} documents store the empty marker", docs.len());
    es.call(reqwest::Method::POST, "/lr/_forcemerge?max_num_segments=1", None, "").await;
    es.call(reqwest::Method::POST, "/lr/_refresh", None, "").await;
    check(&es, "merged").await;
    // Reopen the node from its data directory.
    let Es { task, .. } = es;
    task.abort();
    let _ = task.await;
    drop(node);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let node = Node::open(tmp.path()).unwrap();
    let es = serve(node.clone()).await;
    check(&es, "reopened").await;
}
