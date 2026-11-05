//! Rally `esql` track's `_search` legs through the cluster front
//! (`search_limit_1` / `_10` / `_1000` / `_10000`): `{"size": n}` over
//! `idx_*`, `SEARCH_LIMIT_INDICES` (default 5) indices of 5 shards with 500
//! mapped keyword fields each and `SEARCH_LIMIT_DOCS` (default 100,000)
//! small documents, all on the queried node, uncompressed (as Rally asks). Median and p90 ms per size,
//! with the coordinator's phase medians (`SEARCH_LIMIT_SIZES` =
//! `size:runs,...` picks the legs). Ignored; `cargo test --release -p
//! elasticsearch-node --test search_limit_bench -- --ignored --nocapture`.
mod common;
use common::start;
use reqwest::Method;
use serde_json::json;
use std::time::Instant;

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore]
async fn search_limit_bench() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (indices, docs) = (env("SEARCH_LIMIT_INDICES", 5), env("SEARCH_LIMIT_DOCS", 100_000));
    let cl = start(&[]).await;
    let base = format!("http://127.0.0.1:{}", cl.p.http[0]);
    let load = Instant::now();
    for i in 0..indices {
        let p = i % 20;
        let mut props = serde_json::Map::new();
        props.insert("id".into(), json!({"type": "long"}));
        props.insert("@timestamp".into(), json!({"type": "date"}));
        for j in 0..500 { props.insert(format!("field_{p}_{j}"), json!({"type": "keyword"})); }
        for k in ["key_1000", "key_100000", "key_200000", "key_500000", "key_1000000", "key_5000000", "key_100000000"] { props.insert(k.into(), json!({"type": "keyword"})); }
        let name = format!("idx_{i}");
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": 5, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
            "mappings": {"properties": props}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(status, 200, "{body}");
        let mut seed = i as u64 + 31;
        let mut sent = 0;
        while sent < docs {
            let mut bulk = String::new();
            for _ in 0..(docs - sent).min(10_000) {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = (seed >> 33) % 100_000;
                bulk.push_str("{\"index\":{}}\n{");
                bulk.push_str(&format!("\"key_100000\":\"{v}\""));
                for k in 0..10 { bulk.push_str(&format!(",\"lookup_keyword_{k}\":\"val {v} rep 0\"")); }
                bulk.push_str("}\n");
                sent += 1;
            }
            let r = cl.c.post(format!("{base}/{name}/_bulk")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
            assert!(r.status().is_success());
        }
    }
    let (status, body) = cl.call(0, Method::POST, "/idx_*/_refresh", None).await;
    assert_eq!(status, 200, "{body}");
    eprintln!("loaded {} docs in {:.1} s", indices * docs, load.elapsed().as_secs_f64());
    let url = format!("{base}/idx_*/_search");
    // `SEARCH_LIMIT_SIZES` (`size:runs,...`) replaces the default legs.
    let legs: Vec<(usize, usize)> = std::env::var("SEARCH_LIMIT_SIZES").ok()
        .map(|v| v.split(',').filter_map(|l| { let (s, r) = l.split_once(':')?; Some((s.parse().ok()?, r.parse().ok()?)) }).collect())
        .unwrap_or_else(|| vec![(1, 200), (10, 200), (1000, 21), (10_000, 9)]);
    for (size, runs) in legs {
        let body = json!({"size": size}).to_string();
        for _ in 0..3 { cl.c.post(&url).header("content-type", "application/json").header("accept-encoding", "identity").body(body.clone()).send().await.unwrap().bytes().await.unwrap(); }
        elasticsearch_node::front::stats_phases::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        elasticsearch_engine::search::search_phases::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        elasticsearch_node::front::stats_phases::take();
        elasticsearch_engine::search::search_phases::take();
        elasticsearch_engine::search::search_phases::take_named();
        let mut times = Vec::with_capacity(runs);
        let mut bytes = 0;
        for _ in 0..runs {
            let t = Instant::now();
            let r = cl.c.post(&url).header("content-type", "application/json").header("accept-encoding", "identity").body(body.clone()).send().await.unwrap();
            assert!(r.status().is_success());
            bytes = r.bytes().await.unwrap().len();
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        eprintln!("size {size:<6} median {:>8.2} ms  p90 {:>8.2} ms  ({bytes} bytes)", times[runs / 2], times[runs * 9 / 10]);
        let mut rows: Vec<(String, Vec<(&str, Vec<u128>)>)> = Vec::new();
        let named = elasticsearch_node::front::stats_phases::take().into_iter().map(|(k, p)| (k.to_string(), p))
            .chain(elasticsearch_engine::search::search_phases::take().into_iter().map(|p| ("shard_search".to_string(), p)))
            .chain(elasticsearch_engine::search::search_phases::take_named().into_iter().map(|(k, p)| (k.to_string(), p)));
        for (kind, phases) in named {
            let at = rows.iter().position(|(k, _)| *k == kind).unwrap_or_else(|| { rows.push((kind, Vec::new())); rows.len() - 1 });
            for (i, (n, us)) in phases.into_iter().enumerate() {
                if rows[at].1.len() <= i { rows[at].1.push((n, Vec::new())); }
                rows[at].1[i].1.push(us);
            }
        }
        for (kind, phases) in rows {
            let line: Vec<String> = phases.into_iter().map(|(n, mut v)| { v.sort(); format!("{n} {}/{}", v[v.len() / 2], v[v.len() - 1]) }).collect();
            eprintln!("  {kind:<20} median/max us: {}", line.join("  "));
        }
        elasticsearch_node::front::stats_phases::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        elasticsearch_engine::search::search_phases::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
