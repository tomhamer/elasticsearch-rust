//! A SQL PIVOT read in three cursor pages (Rally sql `pivot_3pages`) holds
//! the rows one big page holds, in the same order, from the node holding the
//! shards and from the other one.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pivot_in_three_pages_matches_one_page() {
    let cl = start(&[]).await;
    let (st, b) = cl.call(0, Method::PUT, "/weather", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"station": {"properties": {"name": {"type": "keyword"}, "country_code": {"type": "keyword"}, "elevation": {"type": "float"}}}, "TMIN": {"type": "double"}}}
    }))).await;
    assert_eq!(st, 200, "{b}");
    let (_, h) = cl.call(0, Method::GET, "/_cluster/health/weather?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(h["status"], "green", "{h}");
    let countries = ["US", "US", "AS", "CA", "US", "MX"];
    let mut bulk = String::new();
    for i in 0..6000usize {
        bulk.push_str(&json!({"index": {"_id": i.to_string()}}).to_string());
        bulk.push('\n');
        // 300 stations, each always in one country, at its own elevation.
        let station = i % 300;
        bulk.push_str(&json!({"station": {"name": format!("station {station:03}"), "country_code": countries[station % countries.len()], "elevation": (station * 13 % 3000) as f64},
            "TMIN": (i % 40) as f64 - 10.0}).to_string());
        bulk.push('\n');
    }
    let b: Value = cl.c.post(format!("http://127.0.0.1:{}/weather/_bulk?refresh=true", cl.p.http[0])).header("content-type", "application/x-ndjson")
        .body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(b["errors"], false, "{b}");
    // Force-merged, as Rally's index (the PIVOT ordinal path reads segments).
    let (st, b) = cl.call(0, Method::POST, "/weather/_forcemerge?max_num_segments=1", None).await;
    assert_eq!(st, 200, "{b}");
    let read = |node: usize, query: String, fetch_size: usize| {
        let cl = &cl;
        async move {
            let (st, mut b) = cl.call(node, Method::POST, "/_sql", Some(json!({"query": query, "fetch_size": fetch_size}))).await;
            assert_eq!(st, 200, "{b}");
            let (columns, mut rows, mut pages) = (b["columns"].clone(), Vec::new(), 0);
            loop {
                pages += 1;
                rows.extend(b["rows"].as_array().unwrap().iter().cloned());
                let Some(cursor) = b.get("cursor").cloned() else { break };
                let (st, next) = cl.call(node, Method::POST, "/_sql", Some(json!({"cursor": cursor}))).await;
                assert_eq!(st, 200, "{next}");
                assert!(next.get("columns").is_none(), "{next}");
                b = next;
            }
            (columns, rows, pages)
        }
    };
    for query in [
        // Rally `pivot_3pages`.
        "SELECT * FROM (SELECT station.name, station.country_code FROM weather WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))",
        "SELECT * FROM (SELECT station.name, station.country_code, TMIN FROM weather WHERE station.elevation > 400) PIVOT (AVG(TMIN) FOR station.country_code IN ('US', 'CA', 'ZZ'))",
        "SELECT * FROM (SELECT station.name, station.country_code FROM weather) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))",
    ] {
        for node in [0, 1] {
            let (columns, whole, pages) = read(node, query.to_string(), 10_000).await;
            assert_eq!(pages, 1, "{query}");
            assert!(whole.len() >= 9, "{query}: {} rows", whole.len());
            let size = whole.len().div_ceil(3);
            let (paged_columns, paged, pages) = read(node, query.to_string(), size).await;
            assert_eq!(pages, 3, "{query}: pages of {size} rows");
            assert_eq!(paged_columns, columns, "{query}");
            assert_eq!(paged, whole, "{query} from n{node}");
        }
    }
}
