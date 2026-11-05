//! Search-only micro-benchmark of Rally `wikipedia` `query-string-search`
//! (query_string over `default_field: "*"`, size 20), without HTTP.
//!
//! ```text
//! WB_PAGES=~/wiki/pages.json WB_QUERIES=~/wiki/queries.csv WB_DOCS=1000000 WB_DIR=/data/wb \
//!   cargo run --profile fast -p elasticsearch-engine --features query-stats --example wiki_search_bench
//! ```
//! The index (2 shards, the track's mapping) is built once per `WB_DIR` from
//! the first `WB_DOCS` docs of the Rally bulk file, refreshed every
//! `WB_DOCS / WB_SEGMENTS` docs so each shard has about `WB_SEGMENTS`
//! segments (the mixed benchmark runs with ~20-26), and reused by later
//! runs. Queries are cleaned and sampled by weight as `wiki_mixed.py` does.
//!
//! Per variant (`full`: as Rally; `nosrc`: `_source: false`, so without the
//! fetch of stored fields; `exact`: `track_total_hits: true`, which counts
//! every match and so scores the OR exhaustively) it prints p50 / p90 / p99
//! / max wall ms over `WB_N` queries run from `WB_THREADS` closed-loop
//! threads, and CPU ms per query. `WB_CHECK=1` also checks that `full`
//! returns the same ids and scores as `exact` (pruning must not change
//! hits) and a `gte` total only past 10,000. `WB_SLOW=N` then reruns each
//! variant's `N` slowest distinct queries one at a time and prints where
//! their time goes: parse, term statistics, leaves (bulk vs per-doc path),
//! block OR work (postings vs docs scored, windows, partitions, essential
//! clauses, blocks walked, probes), fetch and the rest (serialization).
//! The counters ([`rustic_core::query::stats`]) need the `query-stats`
//! feature, which production builds leave off.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Value};
use std::io::BufRead;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn cpu_ms() -> f64 {
    // utime + stime of the whole process, in clock ticks (Linux).
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit_once(')').map(|x| x.1).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let t = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (t(11) + t(12)) * 10.0
}

/// `queries.csv` rows (query, weight), header skipped; quoted fields allowed.
fn read_queries(path: &str) -> Vec<(String, f64)> {
    let text = std::fs::read_to_string(path).expect("WB_QUERIES");
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let mut fields = vec![String::new()];
        let (mut quoted, mut chars) = (false, line.chars().peekable());
        while let Some(c) = chars.next() {
            match c {
                '"' if quoted && chars.peek() == Some(&'"') => {
                    chars.next();
                    fields.last_mut().unwrap().push('"');
                }
                '"' => quoted = !quoted,
                ',' if !quoted => fields.push(String::new()),
                c => fields.last_mut().unwrap().push(c),
            }
        }
        if fields.len() >= 2 {
            let w = fields[fields.len() - 1].trim().parse().unwrap_or(0.0);
            rows.push((fields[..fields.len() - 1].join(","), w));
        }
    }
    rows
}

/// `re.sub('[^0-9a-zA-Z]+', ' ', q).lower()`.
fn clean(q: &str) -> String {
    let mut out = String::new();
    let mut gap = false;
    for c in q.chars() {
        if c.is_ascii_alphanumeric() {
            if gap {
                out.push(' ');
                gap = false;
            }
            out.push(c.to_ascii_lowercase());
        } else {
            gap = true;
        }
    }
    if gap {
        out.push(' ');
    }
    out
}

fn build(node: &Node, pages: &str, n: u64, segments: u64) {
    let t0 = Instant::now();
    let svc = node
        .create_index(
            "wikipedia",
            &json!({
                "settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "-1"},
                "mappings": {"properties": {
                    "title": {"type": "text"}, "content": {"type": "text"},
                    "namespace": {"ignore_above": 1024, "type": "keyword"}, "redirect": {"ignore_above": 1024, "type": "keyword"}
                }}
            }),
        )
        .unwrap();
    let every = (n / segments.max(1)).max(1);
    let mut lines = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(pages).expect("WB_PAGES")).lines();
    let (mut done, mut since) = (0u64, 0u64);
    while done < n {
        let mut reqs = Vec::with_capacity(5000);
        while reqs.len() < 5000 && done + (reqs.len() as u64) < n {
            // Rally bulk lines: action, then source.
            let (Some(Ok(_action)), Some(Ok(src))) = (lines.next(), lines.next()) else { break };
            reqs.push(WriteReq::Index { id: Some((done + reqs.len() as u64).to_string()), source: src, routing: None, cond: WriteCond::default() });
        }
        if reqs.is_empty() {
            break;
        }
        done += reqs.len() as u64;
        since += reqs.len() as u64;
        for r in svc.write(reqs, false) {
            r.unwrap();
        }
        if since >= every {
            svc.refresh().unwrap();
            since = 0;
        }
    }
    svc.refresh().unwrap();
    svc.flush().unwrap();
    eprintln!("indexed {done} docs in {:.1}s", t0.elapsed().as_secs_f64());
}

/// Parse, execute (query, fetch) and serialize one search: the response JSON.
fn run(node: &Node, body: &Value) -> String {
    let targets = node.resolve("wikipedia", Default::default()).unwrap();
    let req = search::parse_request(body, &Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    search::execute(&targets, &req, &opts).unwrap().json
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let pages = std::env::var("WB_PAGES").unwrap_or_else(|_| format!("{home}/wiki/pages.json"));
    let qpath = std::env::var("WB_QUERIES").unwrap_or_else(|_| format!("{home}/wiki/queries.csv"));
    let n: u64 = env("WB_DOCS", 1_000_000);
    let segments: u64 = env("WB_SEGMENTS", 24);
    let nq: usize = env("WB_N", 2000);
    let threads: usize = env("WB_THREADS", 1);
    let check = env("WB_CHECK", 0u8) == 1;
    let slow: usize = env("WB_SLOW", 0);
    let only = std::env::var("WB_ONLY").ok();
    let dir = std::env::var("WB_DIR").unwrap_or_else(|_| format!("/tmp/wiki-search-bench-{n}"));
    std::fs::create_dir_all(&dir).unwrap();
    let node = Node::open(std::path::Path::new(&dir)).unwrap();
    if node.get_index("wikipedia").is_none() {
        build(&node, &pages, n, segments);
    }
    node.get_index("wikipedia").unwrap().refresh().unwrap();

    // Weighted sample (xorshift; not Python's stream, same distribution).
    let rows = read_queries(&qpath);
    let total_w: f64 = rows.iter().map(|r| r.1).sum();
    let mut cum = Vec::with_capacity(rows.len());
    let mut acc = 0.0;
    for r in &rows {
        acc += r.1 / total_w;
        cum.push(acc);
    }
    let mut s = 0x9E3779B97F4A7C15u64;
    let qs: Vec<String> = (0..nq)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            clean(&rows[cum.partition_point(|&c| c < u).min(rows.len() - 1)].0)
        })
        .collect();

    let qs_body = |q: &str| json!({"query_string": {"query": q, "default_field": "*"}});
    let variants: Vec<(&str, Box<dyn Fn(&str) -> Value + Sync>)> = vec![
        ("full", Box::new(move |q: &str| json!({"query": qs_body(q), "size": 20}))),
        ("nosrc", Box::new(move |q: &str| json!({"query": qs_body(q), "size": 20, "_source": false}))),
        ("exact", Box::new(move |q: &str| json!({"query": qs_body(q), "size": 20, "track_total_hits": true}))),
    ];

    if check {
        let mut bad = 0;
        for q in qs.iter().take(nq.min(500)) {
            let parse = |v: String| -> Value { serde_json::from_str(&v).unwrap() };
            let got = parse(run(&node, &variants[0].1(q)));
            let want = parse(run(&node, &variants[2].1(q)));
            let hits = |v: &Value| -> Vec<(String, f64)> {
                v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
            };
            let (g, w) = (hits(&got), hits(&want));
            let same = g.len() == w.len() && g.iter().zip(&w).all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-6 * b.1.abs().max(1.0));
            let exact = want["hits"]["total"]["value"].as_u64().unwrap();
            let (tv, rel) = (got["hits"]["total"]["value"].as_u64().unwrap(), got["hits"]["total"]["relation"].as_str().unwrap().to_string());
            let total_ok = if exact > 10_000 { tv == 10_000 && rel == "gte" } else { tv == exact && rel == "eq" };
            if !same || !total_ok {
                bad += 1;
                if bad <= 10 {
                    eprintln!("MISMATCH {q:?}: total {tv} {rel} vs exact {exact}\n  got  {g:?}\n  want {w:?}");
                }
            }
        }
        println!("check: {bad} mismatches over {} queries", nq.min(500));
    }

    for (name, mk) in &variants {
        if only.as_deref().is_some_and(|o| !o.split(',').any(|x| x == *name)) {
            continue;
        }
        // Warm up (page cache, term dictionaries).
        for q in qs.iter().take(200) {
            std::hint::black_box(run(&node, &mk(q)));
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let c0 = cpu_ms();
        let t0 = Instant::now();
        let mut lat: Vec<(f64, usize)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= qs.len() {
                                break;
                            }
                            let body = mk(&qs[i]);
                            let t = Instant::now();
                            std::hint::black_box(run(&node, &body));
                            mine.push((t.elapsed().as_secs_f64() * 1000.0, i));
                        }
                        mine
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_ms() - c0;
        lat.sort_by(|a, b| a.0.total_cmp(&b.0));
        let p = |x: f64| lat[((x * lat.len() as f64) as usize).min(lat.len() - 1)].0;
        println!(
            "{name:6} n={} threads={threads} p50={:.2} p90={:.2} p99={:.2} p99.9={:.2} max={:.2} ms  qps={:.0}  cpu_ms/query={:.2}",
            lat.len(),
            p(0.5),
            p(0.9),
            p(0.99),
            p(0.999),
            lat[lat.len() - 1].0,
            lat.len() as f64 / wall,
            cpu / lat.len() as f64
        );
        if slow > 0 {
            slow_report(&node, &qs, &lat, slow, mk.as_ref());
        }
    }
}

/// `WB_SLOW=N`: rerun the `N` slowest distinct queries of a variant one at a
/// time and print where their time goes ([`rustic_core::query::stats`]).
fn slow_report(node: &Node, qs: &[String], lat: &[(f64, usize)], n: usize, mk: &(dyn Fn(&str) -> Value + Sync)) {
    use elasticsearch_engine::rustic_core::query::stats;
    let mut seen = std::collections::HashSet::new();
    let picked: Vec<(f64, &str)> = lat.iter().rev().filter(|(_, i)| seen.insert(qs[*i].as_str())).take(n).map(|&(ms, i)| (ms, qs[i].as_str())).collect();
    println!("  slowest {} distinct queries (rerun one at a time; ms = mean of 3 reruns):", picked.len());
    for (first_ms, q) in picked {
        let body = mk(q);
        let mut ms = 0.0;
        let mut parse_ms = 0.0;
        let mut out = String::new();
        stats::reset();
        for _ in 0..3 {
            let t = Instant::now();
            let req = search::parse_request(&body, &Default::default()).unwrap();
            parse_ms += t.elapsed().as_secs_f64() * 1000.0 / 3.0;
            let targets = node.resolve("wikipedia", Default::default()).unwrap();
            let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
            out = search::execute(&targets, &req, &opts).unwrap().json;
            ms += t.elapsed().as_secs_f64() * 1000.0 / 3.0;
        }
        let v: Value = serde_json::from_str(&out).unwrap_or(Value::Null);
        let c: std::collections::HashMap<&str, u64> = stats::snapshot().into_iter().map(|(k, v, _)| (k, v / 3)).collect();
        let msf = |k: &str| c[k] as f64 / 1e6;
        let other = ms - parse_ms - msf("context_ns") - msf("search_ns") - msf("fetch_ns");
        println!(
            "  {first_ms:7.2} ms (rerun {ms:.2}) {q:?}: total {} {}  bytes {}",
            v["hits"]["total"]["value"], v["hits"]["total"]["relation"], out.len()
        );
        println!(
            "      parse {parse_ms:.2}  term stats {:.2} ({} lookups)  leaves {:.2} wall ({} leaves, sum {:.2}, max {:.2}; bulk {} per-doc {})  fetch {:.2} ({} hits)  other/serialize {other:.2}",
            msf("context_ns"), c["term_lookups"], msf("search_ns"), c["leaves"], msf("leaf_ns"), msf("leaf_max_ns"), c["leaf_bulk"], c["leaf_per_doc"], msf("fetch_ns"), c["fetch_hits"]
        );
        if c["store_docs"] > 0 {
            println!(
                "      stored fields: {} docs ({} from the chunk cache), decompress {:.2} ms ({} KB), decode {:.2} ms",
                c["store_docs"], c["store_cache_hits"], msf("store_decompress_ns"), c["store_decompressed_bytes"] / 1024, msf("store_decode_ns")
            );
        }
        if c["or_calls"] > 0 {
            let calls = c["or_calls"].max(1);
            println!(
                "      block OR x{}: {:.2} ms, {} terms / {} groups per leaf, postings {} -> scored {} ({:.1}%), counted {} (exact while counting, then candidates only), count windows {}, partitions {} (avg essential groups {:.1} of {}, required {}; clause-level {}, avg walked clauses {:.1} of {}), skipped outer windows {}, inner windows {}, blocks walked {}, blocks skipped by bound {}, probes {}",
                c["or_calls"], msf("or_ns"), c["or_terms"] / calls, c["or_groups"] / calls, c["or_postings"], c["or_scored"],
                100.0 * c["or_scored"] as f64 / c["or_postings"].max(1) as f64, c["or_counted"], c["or_count_windows"], c["or_partitions"],
                c["or_essential"] as f64 / c["or_partitions"].max(1) as f64, c["or_groups"] / calls, c["or_required"], c["or_clause_partitions"],
                c["or_essential_clauses"] as f64 / c["or_partitions"].max(1) as f64, c["or_terms"] / calls, c["or_skipped_windows"],
                c["or_inner_windows"], c["or_blocks"], c["or_blocks_skipped"], c["or_probes"]
            );
        }
    }
}
