//! The coordinator merges shard hits on their typed sort values: adjacent
//! unsigned_long values past i64::MAX, dates rendered in a day-first
//! `format`, ips and versions order across four shards on two nodes exactly
//! as one shard orders them, for top-K pages, `search_after` pages, scrolls
//! and point-in-time pages. Clients only ever see the rendered values.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

const DOCS: u64 = 120;

fn doc(i: u64) -> Value {
    // Pairs of docs share an unsigned value (the tie-breaker decides);
    // neighbours differ by one near u64::MAX, which f64 cannot tell apart.
    let u = u64::MAX - (i / 2) * if i % 3 == 0 { 1 } else { 2 };
    let day = 1 + (i * 7) % 28;
    let month = 1 + (i * 5) % 12;
    let minute = (i * 13) % 60;
    json!({"id": format!("d{i:03}"), "u": u, "d": format!("2021-{month:02}-{day:02}T10:{minute:02}:00Z"),
        "ip": if i % 4 == 0 { format!("2001:db8::{:x}", i) } else { format!("10.0.{}.{}", i % 3, i % 17) },
        "ver": format!("{}.{}.{}", i % 4, (i * 3) % 11, i % 5)})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn typed_sort_values_merge_like_one_shard() {
    let cl = start(&[]).await;
    for (name, shards) in [("one", 1), ("four", 4)] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 2},
            "mappings": {"properties": {"id": {"type": "keyword"}, "u": {"type": "unsigned_long"}, "d": {"type": "date"}, "ip": {"type": "ip"}, "ver": {"type": "version"}}}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (_, health) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(health["status"], "green", "{health}");
        let mut bulk = String::new();
        for i in 0..DOCS {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{}\n", doc(i)));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", cl.p.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    let (_, shards) = cl.call(0, Method::GET, "/_cat/shards/four?format=json&h=node", None).await;
    let nodes: std::collections::BTreeSet<&str> = shards.as_array().unwrap().iter().filter_map(|s| s["node"].as_str()).collect();
    assert_eq!(nodes.len(), 2, "{shards}");

    let sorts = [
        json!([{"u": "desc"}, {"id": "asc"}]),
        json!([{"u": "asc"}, {"id": "desc"}]),
        json!([{"d": {"order": "asc", "format": "dd/MM/yyyy HH:mm"}}, {"id": "asc"}]),
        json!([{"d": {"order": "desc", "format": "dd/MM/yyyy HH:mm:ss"}}, {"id": "asc"}]),
        json!([{"ip": "asc"}, {"id": "asc"}]),
        json!([{"ver": "desc"}, {"id": "asc"}]),
    ];
    let page = |index: &'static str, node: usize, body: Value| {
        let cl = &cl;
        async move {
            let (status, v) = cl.call(node, Method::POST, &format!("/{index}/_search"), Some(body)).await;
            assert_eq!(status, 200, "{index}: {v}");
            assert!(!v.to_string().contains("_sort_raw"), "{v}");
            v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["sort"].clone())).collect::<Vec<_>>()
        }
    };
    for sort in &sorts {
        // The whole order on one shard.
        let want = page("one", 0, json!({"size": DOCS, "sort": sort, "_source": false})).await;
        assert_eq!(want.len(), DOCS as usize);
        // Top-K pages.
        for (node, size) in [(0, 10), (1, 25), (0, DOCS as usize)] {
            let got = page("four", node, json!({"size": size, "sort": sort, "_source": false})).await;
            assert_eq!(got, want[..size], "{sort} top {size} via n{node}");
        }
        // `search_after` pages with the rendered values the client got.
        for node in 0..2 {
            let mut all = Vec::new();
            let mut after: Option<Value> = None;
            loop {
                let mut body = json!({"size": 7, "sort": sort, "_source": false});
                if let Some(a) = &after { body["search_after"] = a.clone(); }
                let got = page("four", node, body).await;
                if got.is_empty() { break; }
                after = Some(got.last().unwrap().1.clone());
                all.extend(got);
            }
            assert_eq!(all, want, "{sort} search_after via n{node}");
        }
        // Scroll pages.
        let (status, v) = cl.call(1, Method::POST, "/four/_search?scroll=1m", Some(json!({"size": 9, "sort": sort, "_source": false}))).await;
        assert_eq!(status, 200, "{v}");
        let mut scroll_id = v["_scroll_id"].clone();
        let mut hits: Vec<Value> = v["hits"]["hits"].as_array().unwrap().clone();
        loop {
            let (status, v) = cl.call(0, Method::POST, "/_search/scroll", Some(json!({"scroll": "1m", "scroll_id": scroll_id}))).await;
            assert_eq!(status, 200, "{v}");
            assert!(!v.to_string().contains("_sort_raw"), "{v}");
            scroll_id = v["_scroll_id"].clone();
            let page = v["hits"]["hits"].as_array().unwrap().clone();
            if page.is_empty() { break; }
            hits.extend(page);
        }
        let _ = cl.call(0, Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": scroll_id}))).await;
        let got: Vec<(String, Value)> = hits.iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["sort"].clone())).collect();
        assert_eq!(got, want, "{sort} scroll");
        // Point-in-time pages (the PIT adds its `_shard_doc` tiebreak).
        let (status, v) = cl.call(0, Method::POST, "/four/_pit?keep_alive=1m", None).await;
        assert_eq!(status, 200, "{v}");
        let pit = v["id"].clone();
        let mut ids = Vec::new();
        let mut after: Option<Value> = None;
        for p in 0..40 {
            let mut body = json!({"size": 11, "sort": sort, "_source": false, "pit": {"id": pit, "keep_alive": "1m"}});
            if let Some(a) = &after { body["search_after"] = a.clone(); }
            let (status, v) = cl.call(p % 2, Method::POST, "/_search", Some(body)).await;
            assert_eq!(status, 200, "{v}");
            assert!(!v.to_string().contains("_sort_raw"), "{v}");
            let hits = v["hits"]["hits"].as_array().unwrap();
            if hits.is_empty() { break; }
            after = Some(hits.last().unwrap()["sort"].clone());
            ids.extend(hits.iter().map(|h| h["_id"].as_str().unwrap().to_string()));
        }
        let _ = cl.call(0, Method::DELETE, "/_pit", Some(json!({"id": pit}))).await;
        assert_eq!(ids, want.iter().map(|w| w.0.clone()).collect::<Vec<_>>(), "{sort} pit");
    }
}
