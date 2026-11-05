//! Latency of the stats APIs the Rally geonames track times (`_stats` and
//! `_nodes/stats`, every metric) on an in-process cluster holding a
//! 5-shard index: median and p99 over `STATS_BENCH_RUNS` (default 300)
//! calls each, and the median / p90 of each recorded phase
//! (`front::stats_phases`: index stats primary check, read, adjust and its
//! parse / resolve / doc stats / sections / render; node stats probe,
//! parse, entries, merge, render). Ignored; run with `--ignored
//! --nocapture` (release).
mod common;
use common::start;
use reqwest::Method;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn stats_api_latency() {
    let runs: usize = std::env::var("STATS_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let cl = start(&[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/geonames", Some(json!({
        "settings": {"number_of_shards": 5, "number_of_replicas": 0},
        "mappings": {"properties": {"name": {"type": "text"}, "country_code": {"type": "keyword"}, "population": {"type": "long"}, "location": {"type": "geo_point"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, health) = cl.call(0, Method::GET, "/_cluster/health/geonames?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{health}");
    let mut bulk = String::new();
    for i in 0..20_000u32 {
        bulk.push_str(&format!("{{\"index\":{{}}}}\n{{\"name\":\"place {i}\",\"country_code\":\"C{}\",\"population\":{i},\"location\":{{\"lat\":{},\"lon\":{}}}}}\n", i % 200, (i % 170) as f64 - 85.0, (i % 350) as f64 - 175.0));
    }
    let r = cl.c.post(format!("http://127.0.0.1:{}/geonames/_bulk?refresh=true", cl.p.http[0])).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
    assert!(r.status().is_success());
    elasticsearch_node::front::stats_phases::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    for path in ["/_stats", "/_nodes/stats"] {
        for _ in 0..20 { cl.call(0, Method::GET, path, None).await; }
        elasticsearch_node::front::stats_phases::take();
        let mut times: Vec<Duration> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            let (status, _) = cl.call(0, Method::GET, path, None).await;
            times.push(t.elapsed());
            assert_eq!(status, 200, "{path}");
        }
        times.sort();
        eprintln!("{path:<14} median {:?}  p99 {:?}  ({runs} calls)", times[runs / 2], times[runs * 99 / 100]);
        // Where the time goes: each recorded phase's median and p90 (us).
        let mut by_kind: std::collections::BTreeMap<&str, std::collections::BTreeMap<usize, (&str, Vec<u128>)>> = Default::default();
        for (kind, phases) in elasticsearch_node::front::stats_phases::take() {
            let k = by_kind.entry(kind).or_default();
            for (i, (name, us)) in phases.into_iter().enumerate() {
                k.entry(i).or_insert_with(|| (name, Vec::new())).1.push(us);
            }
        }
        for (kind, phases) in by_kind {
            let line: Vec<String> = phases.into_values().map(|(name, mut v)| {
                v.sort();
                format!("{name} {}/{}", v[v.len() / 2], v[v.len() * 9 / 10])
            }).collect();
            eprintln!("  {kind:<20} median/p90 us: {}", line.join("  "));
        }
    }
    elasticsearch_node::front::stats_phases::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
}
