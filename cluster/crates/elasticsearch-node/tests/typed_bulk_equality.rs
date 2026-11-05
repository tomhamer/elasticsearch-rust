//! Typed in-process bulk results (`elasticsearch_http::typed_bulk`) render
//! exactly the JSON path's response: the same scenarios, run once with
//! typed shard groups and once with `UNTYPED_BULK`, against identical fresh
//! indices, answer byte-identical bodies (`took` and the ids the
//! coordinator generates normalized). The two-node cluster sends part of
//! every bulk to the other node's primaries (JSON), so responses mix typed
//! and JSON groups; each bulk goes through both nodes.

mod common;

use reqwest::Method;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

async fn bulk_raw(c: &common::Cluster, node: usize, path: &str, lines: &str) -> (u16, String) {
    let r = c.c.post(format!("http://127.0.0.1:{}{path}", c.p.http[node]))
        .header("content-type", "application/x-ndjson").body(lines.to_string()).send().await.expect("bulk");
    (r.status().as_u16(), r.text().await.expect("bulk body"))
}

/// `took` zeroed; every `_id` value the request lines do not contain (a
/// coordinator-generated id, a failure document's id) and every
/// `index_uuid` (the indices are recreated between the runs) replaced.
fn normalize(body: &str, lines: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(i) = rest.find("\"took\":") {
        out.push_str(&rest[..i + 7]);
        rest = &rest[i + 7..];
        let digits = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        out.push('0');
        rest = &rest[digits..];
    }
    out.push_str(rest);
    let mut v: Value = serde_json::from_str(&out).unwrap_or(Value::String(out.clone()));
    fn ids(v: &mut Value, lines: &str) {
        match v {
            Value::Object(o) => {
                // A generated id routes anywhere: its item's `_seq_no` is
                // that shard's.
                let generated = o.get("_id").and_then(Value::as_str).is_some_and(|s| !lines.contains(&format!("\"{s}\"")));
                if generated && o.contains_key("_seq_no") { o.insert("_seq_no".into(), Value::String("<seq>".into())); }
                for (k, v) in o.iter_mut() {
                    if k == "index_uuid" {
                        *v = Value::String("<uuid>".into());
                    } else if k == "_id" {
                        if let Value::String(s) = v {
                            if !lines.contains(&format!("\"{s}\"")) { *v = Value::String("<generated>".into()); }
                        }
                    } else {
                        ids(v, lines);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|v| ids(v, lines)),
            _ => {}
        }
    }
    // Order-preserving: ids replaced in place, the rest byte for byte.
    if v.is_object() {
        fn generated(v: &Value, lines: &str, out: &mut Vec<String>) {
            match v {
                Value::Object(o) => for (k, v) in o {
                    match (k.as_str(), v) {
                        ("_id", Value::String(s)) if !lines.contains(&format!("\"{s}\"")) => out.push(s.clone()),
                        _ => generated(v, lines, out),
                    }
                },
                Value::Array(a) => a.iter().for_each(|v| generated(v, lines, out)),
                _ => {}
            }
        }
        let mut made = Vec::new();
        generated(&v, lines, &mut made);
        ids(&mut v, lines);
        let mut text = serde_json::to_string(&v).unwrap();
        // A generated id quoted in an error reason too.
        for id in made { text = text.replace(&id, "<generated>"); }
        text
    } else {
        out
    }
}

async fn setup(c: &common::Cluster) {
    let m = c.master;
    for path in ["/eq-a", "/eq-b", "/_data_stream/eq-ds", "/_index_template/eq-ds-template", "/_ingest/pipeline/eq-pipe"] {
        let _ = c.call(m, Method::DELETE, path, None).await;
    }
    let (st, r) = c.call(m, Method::PUT, "/eq-a", Some(json!({
        "settings": {"number_of_shards": 3, "number_of_replicas": 0},
        "mappings": {"dynamic": "strict", "properties": {"k": {"type": "keyword"}, "n": {"type": "long"}, "t": {"type": "text"}}}}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = c.call(m, Method::PUT, "/eq-b", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = c.call(m, Method::PUT, "/_ingest/pipeline/eq-pipe", Some(json!({"processors": [{"set": {"field": "piped", "value": true}}]}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = c.call(m, Method::PUT, "/_index_template/eq-ds-template", Some(json!({
        "index_patterns": ["eq-ds*"], "data_stream": {}, "priority": 500,
        // One shard: whether a rejected document reaches the failure store
        // can depend on which backing shard its generated id routes to (in
        // either mode); one shard keeps the two runs comparable.
        "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}}},
            "data_stream_options": {"failure_store": {"enabled": true}}}}))).await;
    assert_eq!(st, 200, "{r}");
    // The failure store in place before the scenarios: the first rejected
    // document of a new data stream may race the failure index's creation
    // (in either mode), which is not what these runs compare.
    let bad = line("create", json!({"_index": "eq-ds"}), Some(json!({"@timestamp": "2026-11-01T00:00:00Z", "n": "warm up"})));
    let mut used = false;
    for _ in 0..50 {
        let (_, body) = bulk_raw(c, c.master, "/_bulk", &bad).await;
        if body.contains("\"failure_store\":\"used\"") { used = true; break; }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(used, "the failure store never took a rejected document");
}

fn line(action: &str, meta: Value, source: Option<Value>) -> String {
    let mut s = json!({action: meta}).to_string();
    s.push('\n');
    if let Some(src) = source { s.push_str(&src.to_string()); s.push('\n'); }
    s
}

/// The scenarios: (node, path, lines), run in order (later ones see the
/// documents earlier ones wrote).
fn scenarios() -> Vec<(usize, String, String)> {
    let mut v = Vec::new();
    // Plain writes over two multi-shard indices, some routed.
    let mut s = String::new();
    for i in 0..40 {
        let meta = if i % 5 == 0 { json!({"_index": "eq-a", "_id": format!("a{i}"), "routing": format!("r{}", i % 3)}) } else { json!({"_index": "eq-a", "_id": format!("a{i}")}) };
        s.push_str(&line("index", meta, Some(json!({"k": format!("k{i}"), "n": i, "t": "some text"}))));
        s.push_str(&line("index", json!({"_index": "eq-b", "_id": format!("b{i}")}), Some(json!({"x": i}))));
    }
    v.push((0, "/_bulk".to_string(), s.clone()));
    // Overwrites (updated, version 2), through the other node.
    v.push((1, "/_bulk".to_string(), s));
    // Mixed success and failure: create conflicts, deletes (present and
    // missing), updates (present, missing, noop), CAS failures, strict
    // mapping rejections, external versions, op_type create.
    let mut s = String::new();
    for i in 0..6 { s.push_str(&line("create", json!({"_index": "eq-a", "_id": format!("a{i}")}), Some(json!({"n": 1})))); }
    for i in 6..9 { s.push_str(&line("delete", json!({"_index": "eq-a", "_id": format!("a{i}")}), None)); }
    s.push_str(&line("delete", json!({"_index": "eq-a", "_id": "nope"}), None));
    s.push_str(&line("update", json!({"_index": "eq-a", "_id": "a10"}), Some(json!({"doc": {"n": 99}}))));
    s.push_str(&line("update", json!({"_index": "eq-a", "_id": "a11"}), Some(json!({"doc": {"n": 11}}))));
    s.push_str(&line("update", json!({"_index": "eq-a", "_id": "missing"}), Some(json!({"doc": {"n": 1}}))));
    s.push_str(&line("index", json!({"_index": "eq-a", "_id": "a12", "if_seq_no": 999, "if_primary_term": 1}), Some(json!({"n": 1}))));
    s.push_str(&line("index", json!({"_index": "eq-a", "_id": "strict"}), Some(json!({"unknown_field": 1}))));
    s.push_str(&line("index", json!({"_index": "eq-a", "_id": "badnum"}), Some(json!({"n": "not a number"}))));
    s.push_str(&line("index", json!({"_index": "eq-b", "_id": "ext", "version": 7, "version_type": "external"}), Some(json!({"x": 1}))));
    s.push_str(&line("index", json!({"_index": "eq-b", "_id": "ext", "version": 3, "version_type": "external"}), Some(json!({"x": 2}))));
    s.push_str(&line("index", json!({"_index": "eq-b", "_id": "oc", "op_type": "create"}), Some(json!({"x": 1}))));
    s.push_str(&line("index", json!({"_index": "eq-b", "_id": "oc", "op_type": "create"}), Some(json!({"x": 1}))));
    for i in 20..30 { s.push_str(&line("index", json!({"_index": "eq-b", "_id": format!("b{i}")}), Some(json!({"x": -i})))); }
    v.push((0, "/_bulk".to_string(), s.clone()));
    v.push((1, "/_bulk".to_string(), s));
    // Pipelines: the request's and an item's.
    let mut s = String::new();
    for i in 0..10 { s.push_str(&line("index", json!({"_index": "eq-b", "_id": format!("p{i}")}), Some(json!({"x": i})))); }
    v.push((1, "/_bulk?pipeline=eq-pipe".to_string(), s));
    let mut s = String::new();
    for i in 10..20 {
        let meta = if i % 2 == 0 { json!({"_index": "eq-b", "_id": format!("p{i}"), "pipeline": "eq-pipe"}) } else { json!({"_index": "eq-b", "_id": format!("p{i}")}) };
        s.push_str(&line("index", meta, Some(json!({"x": i}))));
    }
    v.push((0, "/_bulk".to_string(), s));
    // The failure store: rejected documents of a data stream.
    let mut s = String::new();
    for i in 0..12 {
        let n = if i % 3 == 0 { json!("not a number") } else { json!(i) };
        s.push_str(&line("create", json!({"_index": "eq-ds"}), Some(json!({"@timestamp": "2026-11-01T00:00:00Z", "n": n}))));
    }
    v.push((0, "/_bulk".to_string(), s.clone()));
    v.push((1, "/_bulk".to_string(), s));
    // Refreshes: forced (`true`) and waited for.
    let mut s = String::new();
    for i in 0..8 { s.push_str(&line("index", json!({"_index": "eq-a", "_id": format!("f{i}")}), Some(json!({"n": i})))); }
    v.push((0, "/_bulk?refresh=true".to_string(), s.clone()));
    v.push((1, "/_bulk?refresh=wait_for".to_string(), s));
    // A large mixed bulk over both indices.
    let mut s = String::new();
    for i in 0..400 {
        match i % 7 {
            0 => s.push_str(&line("delete", json!({"_index": "eq-b", "_id": format!("b{}", i % 40)}), None)),
            1 => s.push_str(&line("create", json!({"_index": "eq-a", "_id": format!("a{}", i % 40)}), Some(json!({"n": i})))),
            _ => s.push_str(&line("index", json!({"_index": if i % 2 == 0 { "eq-a" } else { "eq-b" }, "_id": format!("m{i}")}), Some(json!({"n": i})))),
        }
    }
    v.push((1, "/_bulk".to_string(), s));
    // Auto ids, last: they land on any shard, (the coordinator generates them: `renamed` create items).
    let mut s = String::new();
    for i in 0..30 { s.push_str(&line("index", json!({"_index": if i % 2 == 0 { "eq-a" } else { "eq-b" }}), Some(json!({"n": i})))); }
    v.push((1, "/_bulk".to_string(), s.clone()));
    v.push((0, "/eq-b/_bulk".to_string(), s.replace(",\"_index\":\"eq-a\"", "").replace("{\"_index\":\"eq-a\"}", "{}")));
    v
}

async fn run_all(c: &common::Cluster, untyped: bool) -> Vec<(u16, String)> {
    elasticsearch_node::front::UNTYPED_BULK.store(untyped, Ordering::Relaxed);
    setup(c).await;
    let mut out = Vec::new();
    for (node, path, lines) in scenarios() {
        let (st, body) = bulk_raw(c, node, &path, &lines).await;
        out.push((st, normalize(&body, &lines)));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn typed_bulk_responses_equal_the_json_path() {
    // A fresh cluster for each mode: nothing (a failure store, an id
    // sequence) carries over from the other run.
    let typed = run_all(&common::start(&[]).await, false).await;
    let json = run_all(&common::start(&[]).await, true).await;
    elasticsearch_node::front::UNTYPED_BULK.store(false, Ordering::Relaxed);
    assert_eq!(typed.len(), json.len());
    for (i, (t, j)) in typed.iter().zip(&json).enumerate() {
        assert_eq!(t.0, j.0, "scenario {i}: status");
        if t.1 != j.1 {
            let at = t.1.bytes().zip(j.1.bytes()).position(|(a, b)| a != b).unwrap_or(t.1.len().min(j.1.len()));
            let from = at.saturating_sub(300);
            panic!("scenario {i}: bodies differ at byte {at}\ntyped: ...{}\njson:  ...{}", &t.1[from..(at + 300).min(t.1.len())], &j.1[from..(at + 300).min(j.1.len())]);
        }
    }
    // The scenarios reach what they are meant to cover.
    let all: String = typed.iter().map(|(_, b)| b.as_str()).collect();
    for needle in ["\"result\":\"created\"", "\"result\":\"updated\"", "\"result\":\"deleted\"", "\"result\":\"noop\"",
        "version_conflict_engine_exception", "strict_dynamic_mapping_exception", "document_parsing_exception",
        "\"failure_store\":\"used\"", "\"forced_refresh\":true", "\"status\":404"] {
        assert!(all.contains(needle), "no scenario produced {needle}");
    }
}
