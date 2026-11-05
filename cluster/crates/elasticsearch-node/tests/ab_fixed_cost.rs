//! A/B bench of the per-request fixed cost (no phase hooks, so it builds on
//! any candidate): tiny searches on a 5-shard index on the queried node.
mod common;
use common::start;
use reqwest::Method;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore]
async fn ab_fixed_cost() {
    let runs: usize = std::env::var("FIXED_COST_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000);
    let cl = start(&[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/geo", Some(json!({
        "settings": {"number_of_shards": 5, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"name": {"type": "text"}, "country_code": {"type": "keyword"}, "population": {"type": "long"}, "ts": {"type": "date"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/geo?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{body}");
    let mut bulk = String::new();
    for i in 0..20_000u64 {
        bulk.push_str(&format!("{{\"index\":{{}}}}\n{{\"name\":\"place {i}\",\"country_code\":\"C{}\",\"population\":{},\"ts\":{}}}\n", i % 200, (i * 7919) % 100_000, 1_550_000_000_000u64 + i * 60_000));
    }
    let r = cl.c.post(format!("http://127.0.0.1:{}/geo/_bulk?refresh=true", cl.p.http[0])).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert!(r.status().is_success());
    let url = format!("http://127.0.0.1:{}/geo/_search", cl.p.http[0]);
    for (name, q) in [
        ("default", json!({"query": {"match_all": {}}})),
        ("term", json!({"query": {"term": {"country_code": "C7"}}})),
        ("match", json!({"query": {"match": {"name": "place"}}})),
        ("sort", json!({"query": {"match_all": {}}, "sort": [{"population": "desc"}]})),
        ("date_histo", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "ts", "calendar_interval": "hour"}}}})),
    ] {
        let body = q.to_string();
        for _ in 0..300 { cl.c.post(&url).header("content-type", "application/json").body(body.clone()).send().await.unwrap().bytes().await.unwrap(); }
        let mut times: Vec<Duration> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            let r = cl.c.post(&url).header("content-type", "application/json").body(body.clone()).send().await.unwrap();
            assert!(r.status().is_success());
            r.bytes().await.unwrap();
            times.push(t.elapsed());
        }
        times.sort();
        eprintln!("AB {name:<11} p50 {:>8.1} us  p99 {:>8.1} us", times[runs / 2].as_secs_f64() * 1e6, times[runs * 99 / 100].as_secs_f64() * 1e6);
    }
}
