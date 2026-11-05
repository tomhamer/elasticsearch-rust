//! Tail latency of Rally pmc's short searches (`term` body:physician,
//! `phrase` body:"newspaper coverage", `default` match_all; size 10, full
//! `_source`) on an in-process cluster: a 2-shard index of long text
//! documents held on the queried node, refreshed in small batches with a
//! wide merge tier so each shard keeps tens of segments (pmc ran with 79).
//! Closed-loop clients; per query p50 / p90 / p99 / p99.9 / max over
//! `PMC_TAIL_RUNS` (default 2,000) each, once answered in a single round
//! (each shard renders its top 10, the coordinator merges them) and once in
//! two phases (query, then fetch of the merged page only), which separates
//! the coordinator's cost of taking every shard's rendered hits from the
//! shards' own search.
//!
//! `cargo test --release -p elasticsearch-node --test pmc_tail_latency -- --ignored --nocapture`
//! (`PMC_TAIL_DOCS` default 20,000; `PMC_TAIL_WORDS` body words, default
//! 1,500 (~10 KB); `PMC_TAIL_BATCH` docs per refresh, default 250;
//! `PMC_TAIL_CLIENTS` default 2).
mod common;
use common::start;
use reqwest::Method;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const WORDS: &[&str] = &["the", "of", "and", "patients", "study", "clinical", "cells", "protein", "analysis", "results", "treatment", "health", "data", "expression",
    "disease", "risk", "group", "model", "effect", "response", "care", "levels", "time", "gene", "control", "women", "children", "cancer", "hospital", "public"];

fn body(i: usize, words: usize) -> String {
    let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut out = String::with_capacity(words * 8);
    for w in 0..words {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        // Zipf-ish: low ranks far more often.
        let r = ((x % 1000) as f64 / 1000.0).powi(3);
        out.push_str(WORDS[(r * WORDS.len() as f64) as usize % WORDS.len()]);
        out.push(' ');
        if w % 97 == 0 && x % 13 == 0 { out.push_str("physician "); }
        if w % 211 == 0 && x % 29 == 0 { out.push_str("newspaper coverage "); }
        if w % 89 == 0 && x % 7 == 0 { out.push_str("newspaper "); }
    }
    out
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn pmc_tail_latency() {
    let (docs, words, batch, clients, runs) = (env("PMC_TAIL_DOCS", 20_000), env("PMC_TAIL_WORDS", 1_500), env("PMC_TAIL_BATCH", 250), env("PMC_TAIL_CLIENTS", 2), env("PMC_TAIL_RUNS", 2_000));
    let cl = start(&[]).await;
    let (status, b) = cl.call(0, Method::PUT, "/pmc", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "-1",
            "index.routing.allocation.require._name": "n0", "index.merge.policy.segments_per_tier": 50, "index.merge.policy.max_merge_at_once": 50},
        "mappings": {"dynamic": "strict", "properties": {"name": {"type": "keyword"}, "journal": {"type": "text"}, "timestamp": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
            "pmid": {"type": "integer"}, "body": {"type": "text"}}}
    }))).await;
    assert_eq!(status, 200, "{b}");
    let (status, health) = cl.call(0, Method::GET, "/_cluster/health/pmc?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{health}");
    let url = format!("http://127.0.0.1:{}/pmc/_bulk?refresh=true", cl.p.http[0]);
    for start_doc in (0..docs).step_by(batch) {
        let mut bulk = String::new();
        for i in start_doc..(start_doc + batch).min(docs) {
            bulk.push_str("{\"index\":{}}\n");
            bulk.push_str(&json!({"name": format!("PMC{i}"), "journal": "Journal of Studies", "timestamp": format!("20{:02}-{:02}-01 00:00:00", 10 + i % 10, 1 + i % 12),
                "pmid": i, "body": body(i, words)}).to_string());
            bulk.push('\n');
        }
        let r = cl.c.post(&url).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
        assert!(r.status().is_success());
    }
    let (_, segs) = cl.call(0, Method::GET, "/_cat/segments/pmc?format=json", None).await;
    let (_, shards) = cl.call(0, Method::GET, "/_cat/shards/pmc?format=json", None).await;
    eprintln!("pmc: {docs} docs of {words} words, {} segments; shards on {:?}", segs.as_array().map_or(0, |a| a.len()),
        shards.as_array().map(|a| a.iter().map(|s| s["node"].clone()).collect::<Vec<_>>()));
    let queries = [
        ("term", json!({"query": {"term": {"body": "physician"}}})),
        ("phrase", json!({"query": {"match_phrase": {"body": "newspaper coverage"}}})),
        ("default", json!({"query": {"match_all": {}}})),
    ];
    for two_phase in [false, true] {
        elasticsearch_node::front::QTF_SINGLE_ROUND_DISABLED.store(two_phase, Ordering::Relaxed);
        // `single`: a single round even for large hits (the default now
        // fetches only the winners of large documents, as `two-phase`).
        elasticsearch_node::front::LARGE_HITS_SINGLE_ROUND.store(!two_phase, Ordering::Relaxed);
        for (name, q) in &queries {
            let path = format!("http://127.0.0.1:{}/pmc/_search", cl.p.http[0]);
            for _ in 0..50 { cl.c.post(&path).json(q).send().await.unwrap().bytes().await.unwrap(); }
            let per = runs / clients.max(1);
            let tasks: Vec<_> = (0..clients.max(1)).map(|_| {
                let (c, path, q) = (cl.c.clone(), path.clone(), q.clone());
                tokio::spawn(async move {
                    let mut t = Vec::with_capacity(per);
                    for _ in 0..per {
                        let s = Instant::now();
                        let r = c.post(&path).json(&q).send().await.unwrap();
                        assert!(r.status().is_success());
                        r.bytes().await.unwrap();
                        t.push(s.elapsed());
                    }
                    t
                })
            }).collect();
            let mut all = Vec::new();
            for t in tasks { all.extend(t.await.unwrap()); }
            all.sort();
            eprintln!("{:<10} {name:<8} p50 {:>7.2?}  p90 {:>7.2?}  p99 {:>7.2?}  p99.9 {:>7.2?}  max {:>7.2?}  ({} runs, {clients} clients)",
                if two_phase { "two-phase" } else { "single" }, pct(&all, 0.5), pct(&all, 0.9), pct(&all, 0.99), pct(&all, 0.999), all[all.len() - 1], all.len());
        }
    }
    elasticsearch_node::front::QTF_SINGLE_ROUND_DISABLED.store(false, Ordering::Relaxed);
    elasticsearch_node::front::LARGE_HITS_SINGLE_ROUND.store(false, Ordering::Relaxed);
}
