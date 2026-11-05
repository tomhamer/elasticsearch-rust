//! Per-request fixed cost of fast searches (Rally geonames `default` /
//! `term`, nested's small queries): a 5-shard index of small documents on
//! the queried node, tiny `match_all` and `term` searches (size 10) from one
//! closed-loop client, p50 / p99 over `FIXED_COST_RUNS` (default 10,000)
//! calls each, with the coordinator's and the shard searches' phase
//! medians (`front::stats_phases`, `search::search_phases`). Ignored; `cargo test --release -p elasticsearch-node --test
//! fixed_cost_latency -- --ignored --nocapture`.
mod common;
use common::start;
use reqwest::Method;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore]
async fn fixed_cost_latency() {
    let runs: usize = std::env::var("FIXED_COST_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000);
    let cl = start(&[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/geo", Some(json!({
        "settings": {"number_of_shards": 5, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"name": {"type": "text"}, "country_code": {"type": "keyword"}, "population": {"type": "long"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/geo?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{body}");
    let mut bulk = String::new();
    for i in 0..20_000u32 {
        bulk.push_str(&format!("{{\"index\":{{}}}}\n{{\"name\":\"place {i}\",\"country_code\":\"C{}\",\"population\":{}}}\n", i % 200, (i * 7919) % 100_000));
    }
    let r = cl.c.post(format!("http://127.0.0.1:{}/geo/_bulk?refresh=true", cl.p.http[0])).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert!(r.status().is_success());
    let url = format!("http://127.0.0.1:{}/geo/_search", cl.p.http[0]);
    for (name, q) in [
        ("default", json!({"query": {"match_all": {}}})),
        ("term", json!({"query": {"term": {"country_code": "C7"}}})),
        ("sort", json!({"query": {"match_all": {}}, "sort": [{"population": "desc"}]})),
    ] {
        let body = q.to_string();
        for _ in 0..200 { cl.c.post(&url).header("content-type", "application/json").body(body.clone()).send().await.unwrap().bytes().await.unwrap(); }
        elasticsearch_node::front::stats_phases::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        elasticsearch_engine::search::search_phases::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        elasticsearch_node::front::stats_phases::take();
        elasticsearch_engine::search::search_phases::take();
        elasticsearch_engine::search::search_phases::take_named();
        let mut times: Vec<Duration> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            let r = cl.c.post(&url).header("content-type", "application/json").body(body.clone()).send().await.unwrap();
            assert!(r.status().is_success());
            r.bytes().await.unwrap();
            times.push(t.elapsed());
        }
        times.sort();
        eprintln!("{name:<8} p50 {:?}  p99 {:?}  ({runs} calls)", times[runs / 2], times[runs * 99 / 100]);
        // Where the time goes: the coordinator's phases (one record per
        // request) and the shard engines' (one per shard search), median /
        // p90 us each. The client time beyond the coordinator's total is
        // HTTP, routing, security and target resolution.
        let mut rows: Vec<(String, Vec<(&str, Vec<u128>)>)> = Vec::new();
        let mut add = |kind: String, phases: Vec<(&'static str, u128)>| {
            let at = rows.iter().position(|(k, _)| *k == kind).unwrap_or_else(|| { rows.push((kind, Vec::new())); rows.len() - 1 });
            for (i, (n, us)) in phases.into_iter().enumerate() {
                if rows[at].1.len() <= i { rows[at].1.push((n, Vec::new())); }
                rows[at].1[i].1.push(us);
            }
        };
        for (kind, phases) in elasticsearch_node::front::stats_phases::take() { add(kind.to_string(), phases); }
        for phases in elasticsearch_engine::search::search_phases::take() { add("shard_search".to_string(), phases); }
        for (kind, phases) in elasticsearch_engine::search::search_phases::take_named() { add(kind.to_string(), phases); }
        for (kind, phases) in rows {
            let line: Vec<String> = phases.into_iter().map(|(n, mut v)| { v.sort(); format!("{n} {}/{}", v[v.len() / 2], v[v.len() * 9 / 10]) }).collect();
            eprintln!("  {kind:<20} median/p90 us: {}", line.join("  "));
        }
    }
    elasticsearch_node::front::stats_phases::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    elasticsearch_engine::search::search_phases::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
}
