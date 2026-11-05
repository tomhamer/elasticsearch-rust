//! The coordinator's fan-in of rendered hits with ~50 KB `_source`s (Rally
//! pmc): two shard answers parsed (`_source` kept as raw text), merged and
//! rendered, as a single round takes them (each shard's top 10: 20 hits in,
//! 10 out) and as a query_then_fetch fetch phase does (only the 10 winners
//! rendered). Median and p99 over `MERGE_BENCH_RUNS` (default 500).
//!
//! `cargo test --release -p elasticsearch-node --test coordinator_merge_microbench -- --ignored --nocapture`
use elasticsearch_engine::source_json::ResponseSources;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn shard_answer(shard: usize, hits: usize, source_bytes: usize) -> Vec<u8> {
    let text: String = "physician newspaper coverage clinical \"study\" naïve\n ".repeat(source_bytes / 52);
    let hits: Vec<Value> = (0..hits).map(|i| {
        let score = 10.0 - (i * 2 + shard) as f64 * 0.1;
        json!({"_index": "pmc", "_id": format!("{shard}-{i}"), "_score": score,
            "_source": {"name": format!("PMC{i}"), "journal": "J", "timestamp": "2013-01-01 00:00:00", "pmid": i, "body": text}})
    }).collect();
    let max = hits.first().map_or(Value::Null, |h| h["_score"].clone());
    serde_json::to_vec(&json!({"took": 3, "timed_out": false, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {"total": {"value": 10000, "relation": "gte"}, "max_score": max, "hits": hits}})).unwrap()
}

#[test]
#[ignore]
fn coordinator_merge_microbench() {
    let runs: usize = std::env::var("MERGE_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    let body = json!({"query": {"term": {"body": "physician"}}});
    let raw = body.to_string().into_bytes();
    let params = std::collections::HashMap::new();
    for (label, answers) in [
        ("single round: 2 shards x 10 hits in", vec![shard_answer(0, 10, 50_000), shard_answer(1, 10, 50_000)]),
        ("fetch phase: the 10 winners in", vec![shard_answer(0, 5, 50_000), shard_answer(1, 5, 50_000)]),
    ] {
        let input: usize = answers.iter().map(Vec::len).sum();
        let mut times: Vec<Duration> = Vec::with_capacity(runs);
        let mut out = 0;
        for _ in 0..runs + 20 {
            let t = Instant::now();
            let mut sources = ResponseSources::default();
            let parts: Vec<Value> = answers.iter().map(|a| sources.parse_lazy(a).unwrap()).collect();
            let merged = elasticsearch_node::front::fanout::merge_search_with_source(&parts, &body, &params, 65_536, &raw);
            let rendered = sources.render(&merged);
            out = rendered.len();
            times.push(t.elapsed());
        }
        let mut times = times.split_off(20);
        times.sort();
        eprintln!("{label:<38} {:>5} KB in, {:>4} KB out: median {:?}  p99 {:?}", input / 1024, out / 1024, times[runs / 2], times[runs * 99 / 100]);
    }
}
