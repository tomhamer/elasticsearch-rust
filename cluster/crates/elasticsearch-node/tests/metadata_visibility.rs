//! An acknowledged metadata change is visible to the requests that follow
//! it (ES applies each committed cluster state whole before acking): a
//! template is used by the index created right after it, a cluster setting
//! reads back at once, and a custom (an ingest pipeline) is usable at once,
//! also while other engine APIs of its family are in flight.

mod common;

use reqwest::Method;
use serde_json::{json, Value};
use std::sync::Arc;

async fn bulk(c: &common::Cluster, node: usize, path: &str, lines: String) -> (u16, Value) {
    let r = c.c.post(format!("http://127.0.0.1:{}{path}", c.p.http[node]))
        .header("content-type", "application/x-ndjson").body(lines).send().await.expect("bulk");
    let st = r.status().as_u16();
    (st, r.json().await.unwrap_or(Value::Null))
}

/// (a) Put a template, then immediately create an index (explicitly, or by
/// a bulk auto-create, or as a data stream) and bulk into it, from many
/// clients at once through both nodes: every index gets the template's
/// settings and mappings and every document is accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn template_then_create_and_bulk_concurrently() {
    let c = Arc::new(common::start(&[]).await);
    let mut tasks = Vec::new();
    for client in 0..12usize {
        let c = c.clone();
        tasks.push(tokio::spawn(async move {
            for round in 0..6usize {
                let node = (client + round) % 2;
                let mode = (client + round) % 3;
                let base = format!("vis-{client}-{round}");
                let mut template = json!({
                    "index_patterns": [format!("{base}*")],
                    "priority": 500,
                    "template": {
                        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
                        "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}, "@timestamp": {"type": "date"}}}
                    }
                });
                if mode == 2 {
                    template["data_stream"] = json!({});
                }
                let (st, v) = c.call(node, Method::PUT, &format!("/_index_template/t-{base}"), Some(template)).await;
                assert_eq!(st, 200, "template {base}: {v}");
                assert_eq!(v["acknowledged"], true, "{v}");
                let target = match mode {
                    0 => {
                        let (st, v) = c.call(1 - node, Method::PUT, &format!("/{base}-x"), None).await;
                        assert_eq!(st, 200, "create {base}: {v}");
                        format!("{base}-x")
                    }
                    1 => format!("{base}-auto"),
                    _ => format!("{base}-ds"),
                };
                let mut lines = String::new();
                for i in 0..25 {
                    lines.push_str(&format!("{}\n{}\n", json!({"create": {}}), json!({"k": format!("v{i}"), "n": i, "@timestamp": "2026-10-29T00:00:00Z"})));
                }
                let (st, v) = bulk(&c, 1 - node, &format!("/{target}/_bulk?refresh=true"), lines).await;
                assert_eq!(st, 200, "bulk {target}: {v}");
                assert_eq!(v["errors"], false, "bulk {target}: {v}");
                // The template's mapping and settings, not dynamic defaults.
                let (st, m) = c.call(node, Method::GET, &format!("/{target}/_mapping"), None).await;
                assert_eq!(st, 200, "{m}");
                let (_, index) = m.as_object().unwrap().iter().next().expect("one index");
                assert_eq!(index["mappings"]["properties"]["k"]["type"], "keyword", "{target}: {m}");
                assert_eq!(index["mappings"]["properties"]["n"]["type"], "long", "{target}: {m}");
                let (_, s) = c.call(node, Method::GET, &format!("/{target}/_settings?flat_settings=true"), None).await;
                let (_, settings) = s.as_object().unwrap().iter().next().expect("one index");
                assert_eq!(settings["settings"]["index.number_of_shards"], "2", "{target}: {s}");
                if mode == 2 {
                    let (st, d) = c.call(node, Method::GET, &format!("/_data_stream/{target}"), None).await;
                    assert_eq!(st, 200, "{target} is a data stream: {d}");
                }
                let (_, n) = c.call(1 - node, Method::GET, &format!("/{target}/_count"), None).await;
                assert_eq!(n["count"], 25, "{target}: {n}");
            }
        }));
    }
    for t in tasks {
        t.await.expect("client");
    }
}

/// (b) A cluster settings update reads back at once, through either node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_settings_read_back_immediately() {
    let c = common::start(&[]).await;
    for round in 0..40u64 {
        let node = (round % 2) as usize;
        let value = format!("{}", round % 7 + 1);
        let (st, v) = c.call(node, Method::PUT, "/_cluster/settings", Some(json!({
            "persistent": {"cluster.routing.allocation.node_concurrent_recoveries": value},
            "transient": {"search.default_search_timeout": format!("{}s", round + 1)}
        }))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["acknowledged"], true, "{v}");
        for reader in [0usize, 1] {
            let (st, s) = c.call(reader, Method::GET, "/_cluster/settings?flat_settings=true", None).await;
            assert_eq!(st, 200, "{s}");
            assert_eq!(s["persistent"]["cluster.routing.allocation.node_concurrent_recoveries"], value.as_str(), "round {round} via n{reader}: {s}");
            assert_eq!(s["transient"]["search.default_search_timeout"], format!("{}s", round + 1).as_str(), "round {round} via n{reader}: {s}");
        }
    }
}

/// (c) Ingest pipelines (a custom changed by engine APIs of one family)
/// put by many clients at once: each acked pipeline is readable through
/// both nodes and used by the very next index request, although other
/// pipeline APIs are in flight when it is published.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_pipeline_puts_are_usable_once_acked() {
    let c = Arc::new(common::start(&[]).await);
    let (st, v) = c.call(0, Method::PUT, "/piped", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    let mut tasks = Vec::new();
    for client in 0..10usize {
        let c = c.clone();
        tasks.push(tokio::spawn(async move {
            for round in 0..8usize {
                let id = format!("p-{client}-{round}");
                let node = (client + round) % 2;
                let (st, v) = c.call(node, Method::PUT, &format!("/_ingest/pipeline/{id}"), Some(json!({
                    "processors": [{"set": {"field": "via", "value": id}}]
                }))).await;
                assert_eq!(st, 200, "{id}: {v}");
                assert_eq!(v["acknowledged"], true, "{id}: {v}");
                for reader in [0usize, 1] {
                    let (st, p) = c.call(reader, Method::GET, &format!("/_ingest/pipeline/{id}"), None).await;
                    assert_eq!(st, 200, "{id} via n{reader}: {p}");
                }
                let (st, v) = c.call(1 - node, Method::PUT, &format!("/piped/_doc/{id}?pipeline={id}&refresh=true"), Some(json!({"k": 1}))).await;
                assert!(st == 200 || st == 201, "{id}: {v}");
                let (st, d) = c.call(node, Method::GET, &format!("/piped/_doc/{id}"), None).await;
                assert_eq!(st, 200, "{d}");
                assert_eq!(d["_source"]["via"], id.as_str(), "{id}: {d}");
            }
        }));
    }
    for t in tasks {
        t.await.expect("client");
    }
}
