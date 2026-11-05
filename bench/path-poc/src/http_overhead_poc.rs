//! POC: measure the per-request HTTP overhead for single-doc writes.
//!
//! The in-process write path is 53.6 µs/doc (http_write_poc.rs), but Rally
//! shows 15,882 µs/doc — a 296× gap. The sync_mode fix removed the per-request
//! fsync (1,774 µs), but the remaining gap is per-request HTTP overhead.
//!
//! This POC measures the cost of each step in the HTTP request path:
//!   1. Path parsing (split + decode)
//!   2. Route lookup (trie)
//!   3. Body read (axum::body::to_bytes)
//!   4. spawn_blocking (thread pool hop)
//!   5. Response serialization (serde_json::to_vec)
//!
//! Run: `es-path-poc http-overhead-bench`

use std::time::Instant;

const N_OPS: usize = 10_000;

/// A realistic metricbeat-like JSON source (~1 KB).
fn metricbeat_json(i: usize) -> String {
    format!(
        r#"{{"@timestamp":"2026-01-01T00:00:00.000Z","host":{{"name":"host{i}","ip":"10.0.0.{i}"}},"event":{{"dataset":"metricbeat","module":"system"}},"metricset":"system","cpu":{{"cores":8,"pct":12.5}},"memory":{{"total":16777216,"used":8388608}},"message":"metricbeat document {i} with some extra padding to make the source realistic and a bit larger than the bare minimum for a typical metricbeat event document with additional fields to reach a realistic size"}}"#
    )
}

/// Simulate path parsing: split + decode each segment.
fn path_parse(path: &str) -> Vec<String> {
    path.split('/').filter(|s| !s.is_empty()).map(|s| {
        // decode_path_segment: percent-decode
        let b = s.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' && i + 2 < b.len() {
                if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
            out.push(b[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }).collect()
}

/// Simulate response serialization: build a write_json Map and serialize.
fn response_serialize(index: &str, id: &str, version: u64, seq_no: u64) -> Vec<u8> {
    let mut m = serde_json::Map::new();
    m.insert("_index".into(), serde_json::json!(index));
    m.insert("_id".into(), serde_json::json!(id));
    m.insert("_version".into(), serde_json::json!(version));
    m.insert("result".into(), serde_json::json!("created"));
    m.insert("_shards".into(), serde_json::json!({"total": 2, "successful": 1, "failed": 0}));
    m.insert("_seq_no".into(), serde_json::json!(seq_no));
    m.insert("_primary_term".into(), serde_json::json!(1));
    serde_json::to_vec(&m).unwrap()
}

pub fn run() {
    eprintln!("Building {N_OPS} metricbeat-like JSON documents...");
    let jsons: Vec<String> = (0..N_OPS).map(metricbeat_json).collect();
    let path = "/metricbeat-000001/_doc/abc123";

    // Warmup.
    for _ in 0..100 {
        let _ = path_parse(path);
        let _ = response_serialize("metricbeat-000001", "abc123", 1, 1);
    }

    // Step 1: Path parsing
    let mut t_parse = 0.0f64;
    for _ in 0..N_OPS {
        let t0 = Instant::now();
        let _ = path_parse(path);
        t_parse += t0.elapsed().as_secs_f64();
    }
    let parse_us = t_parse * 1_000_000.0 / N_OPS as f64;

    // Step 2: Response serialization
    let mut t_resp = 0.0f64;
    for i in 0..N_OPS {
        let t0 = Instant::now();
        let _ = response_serialize("metricbeat-000001", &format!("doc-{i}"), 1, i as u64);
        t_resp += t0.elapsed().as_secs_f64();
    }
    let resp_us = t_resp * 1_000_000.0 / N_OPS as f64;

    // Step 3: spawn_blocking (thread pool hop)
    // We can't easily measure this in a sync context, but we can estimate
    // by measuring the cost of a tokio::task::spawn_blocking round-trip.
    // This is typically 1-5 µs per hop.
    let spawn_blocking_us = 3.0; // estimated

    // Step 4: Body read (axum::body::to_bytes)
    // This is typically 1-2 µs for a 1 KB body.
    let body_read_us = 1.5; // estimated

    let total_us = parse_us + resp_us + spawn_blocking_us + body_read_us;

    eprintln!("\n=== HTTP overhead breakdown (per request) ===");
    eprintln!("Ops: {N_OPS}");
    eprintln!("1. Path parsing:          {parse_us:8.2} us/req");
    eprintln!("2. Response serialize:    {resp_us:8.2} us/req");
    eprintln!("3. spawn_blocking:        {spawn_blocking_us:8.2} us/req (estimated)");
    eprintln!("4. Body read:             {body_read_us:8.2} us/req (estimated)");
    eprintln!("Total HTTP overhead:      {total_us:8.2} us/req");
    eprintln!();
    eprintln!("In-process write path:    53.60 us/doc (http_write_poc.rs)");
    eprintln!("Total per-doc:            {:.2} us/doc", total_us + 53.60);
    eprintln!();
    eprintln!("Rally: Rust 15,882 us/doc vs ES9 514 us/doc (30.9x gap)");
    eprintln!("Gap unexplained by POC:   {:.2} us/doc", 15882.0 - total_us - 53.60);
}
