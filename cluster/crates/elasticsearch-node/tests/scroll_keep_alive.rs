//! A composite (cross-node) scroll keeps every shard sub-context alive while
//! the scroll is: sub-contexts a page needs nothing from are renewed too.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn idle_sub_contexts_live_as_long_as_the_scroll() {
    let cl = start(&[]).await;
    let ports = cl.p;
    // One index per node: the scroll's pages come from `low` first, so
    // `high`'s sub-context sits on its buffered first page for a while.
    for (name, node, base, n) in [("low", "n0", 0, 25), ("high", "n1", 1000, 30)] {
        let (st, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": node},
            "mappings": {"properties": {"n": {"type": "long"}}}
        }))).await;
        assert_eq!(st, 200, "{body}");
        let (st, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(st, 200, "{body}");
        let mut bulk = String::new();
        for i in 0..n {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n{}\n", base + i, json!({"n": base + i})));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", ports.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    // A 2 s scroll whose first continuation (at 0.5 s) raises the keep-alive
    // to 10 s, then 4 s of quiet: `high` gives nothing to the first pages,
    // so unless that continuation renewed it too, its context expired at
    // t = 2 s (an expired context is gone on lookup) and the page that
    // first needs more of `high` fails.
    let (st, first) = cl.call(1, Method::POST, "/low,high/_search?scroll=2s", Some(json!({"size": 10, "sort": [{"n": "asc"}]}))).await;
    assert_eq!(st, 200, "{first}");
    let mut id = first["_scroll_id"].as_str().unwrap().to_string();
    let mut seen: Vec<i64> = first["hits"]["hits"].as_array().unwrap().iter().map(|h| h["sort"][0].as_i64().unwrap()).collect();
    for page in 2..=7 {
        tokio::time::sleep(Duration::from_millis(match page { 2 => 500, 3 => 4000, _ => 0 })).await;
        let (st, next) = cl.call(1, Method::POST, "/_search/scroll", Some(json!({"scroll_id": id, "scroll": "10s"}))).await;
        assert_eq!(st, 200, "page {page}: {next}");
        id = next["_scroll_id"].as_str().unwrap().to_string();
        let hits = next["hits"]["hits"].as_array().unwrap();
        seen.extend(hits.iter().map(|h| h["sort"][0].as_i64().unwrap()));
        if hits.is_empty() {
            break;
        }
    }
    let want: Vec<i64> = (0..25).chain(1000..1030).collect();
    assert_eq!(seen, want);
}

/// Scroll contexts held by shard engines (shard 1 of a two-shard index) are
/// reaped by the node's maintenance once they expire, like engine 0's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shard_engine_scroll_contexts_are_reaped() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (st, body) = cl.call(0, Method::PUT, "/two", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{body}");
    let (st, body) = cl.call(0, Method::GET, "/_cluster/health/two?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(st, 200, "{body}");
    let mut bulk = String::new();
    for i in 0..40 {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{}\n", json!({"n": i})));
    }
    let body: Value = cl.c.post(format!("http://127.0.0.1:{}/two/_bulk?refresh=true", ports.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(body["errors"], false, "{body}");
    let open = || async {
        let (st, s) = cl.call(0, Method::GET, "/two/_stats/search", None).await;
        assert_eq!(st, 200, "{s}");
        s["_all"]["total"]["search"]["scroll_current"].as_u64().unwrap_or(0)
    };
    let (st, first) = cl.call(1, Method::POST, "/two/_search?scroll=1s", Some(json!({"size": 1}))).await;
    assert_eq!(st, 200, "{first}");
    // One context per shard, shard 1's in a shard engine.
    assert_eq!(open().await, 2);
    tokio::time::sleep(Duration::from_millis(3500)).await;
    assert_eq!(open().await, 0, "expired scroll contexts left open");
}

/// Rolling upgrade: a node of an earlier build answers a scroll touch with a
/// page (it ignores the touch header and advances its cursor). The
/// composite scroll buffers that page instead of dropping it: every doc
/// comes back once, in order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_touch_answered_with_a_page_loses_no_hits() {
    let cl = start(&[]).await;
    let ports = cl.p;
    for (name, node, base, n) in [("low", "n0", 0, 23), ("high", "n1", 1000, 31)] {
        let (st, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": node},
            "mappings": {"properties": {"n": {"type": "long"}}}
        }))).await;
        assert_eq!(st, 200, "{body}");
        let (st, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(st, 200, "{body}");
        let mut bulk = String::new();
        for i in 0..n {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n{}\n", base + i, json!({"n": base + i})));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", ports.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    elasticsearch_http::SCROLL_TOUCH_AS_PAGE.store(true, std::sync::atomic::Ordering::Relaxed);
    // `high` sits on its buffered first page while `low` is read, and is
    // touched on every page.
    let (st, first) = cl.call(1, Method::POST, "/low,high/_search?scroll=1m", Some(json!({"size": 4, "sort": [{"n": "asc"}]}))).await;
    assert_eq!(st, 200, "{first}");
    let mut id = first["_scroll_id"].as_str().unwrap().to_string();
    let mut seen: Vec<i64> = first["hits"]["hits"].as_array().unwrap().iter().map(|h| h["sort"][0].as_i64().unwrap()).collect();
    for page in 2..=40 {
        let (st, next) = cl.call(1, Method::POST, "/_search/scroll", Some(json!({"scroll_id": id, "scroll": "1m"}))).await;
        assert_eq!(st, 200, "page {page}: {next}");
        id = next["_scroll_id"].as_str().unwrap().to_string();
        let hits = next["hits"]["hits"].as_array().unwrap();
        seen.extend(hits.iter().map(|h| h["sort"][0].as_i64().unwrap()));
        if hits.is_empty() {
            break;
        }
    }
    elasticsearch_http::SCROLL_TOUCH_AS_PAGE.store(false, std::sync::atomic::Ordering::Relaxed);
    let want: Vec<i64> = (0..23).chain(1000..1031).collect();
    assert_eq!(seen, want);
}
