//! `compare`: head-to-head benchmark harness for rustic-core vs Tantivy.
//!
//! Subcommands (every one writes `results/<system>-<subcommand>.json`):
//!
//! * `index   --system rustic|tantivy --corpus corpus.jsonl --out DIR [--refresh-every 100000] [--threads 1] [--heap-mb 1024] [--tantivy-commit-every 0] [--force-merge]`
//! * `startup --system X --index DIR --queries queries.jsonl [--runs 5] [--no-purge]`
//! * `query   --system X --index DIR --queries queries.jsonl [--warmup 2] [--iters 5] [--global-warmup 1]`
//! * `vectors --system rustic|tantivy --vectors vectors.f32 [--dim 128] [--queries 20]`
//! * `hybrid  --system rustic|tantivy|hnswlib --corpus corpus.jsonl --queries queries.jsonl --vectors vectors.f32 [--vec-queries 100] [--n 100]` (see hybrid.rs)
//! * `report  --inputs results/*.json --out RESULTS.md`
//!
//! Global: `--results DIR` (default `./results`); `--label NAME` overrides the
//! system name used in the result file (e.g. `rustic-1seg`) so several index
//! layouts of one system can appear side by side in the report.  See
//! `bench/README.md` for methodology.

mod common;
mod engine;
mod hybrid;
mod report;
mod rustic_engine;
mod tantivy_engine;
mod vectors;
mod vectors_filtered;

use common::{for_each_doc, load_queries, Args, LatencyStats, Query, ResultFile};
use engine::{QueryResult, System};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn usage() -> ! {
    eprintln!(
        "usage: compare <index|startup|query|vectors|report> [--flags]\n\
         \n  index   --system rustic|tantivy --corpus corpus.jsonl --out DIR [--refresh-every 100000] [--threads 1] [--heap-mb 1024] [--force-merge]\
         \n  startup --system X --index DIR --queries queries.jsonl [--runs 5] [--no-purge]\
         \n  query   --system X --index DIR --queries queries.jsonl [--warmup 2] [--iters 5]\
         \n  vectors --system rustic --vectors vectors.f32 [--dim 128] [--queries 20] [--ef 50,100,200]\
         \n  report  --inputs results/*.json --out RESULTS.md\
         \n  (all)   --results DIR   where JSON results are written (default ./results)\
         \n          --label NAME    system name to record (default: the --system value)"
    );
    std::process::exit(2);
}

fn results_dir(args: &Args) -> PathBuf {
    PathBuf::from(args.get_or("results", "results"))
}

/// Name recorded in the result file: `--label` if given, else the system.
fn label(args: &Args, system: System) -> String {
    args.get_or("label", system.name())
}

fn main() {
    let args = Args::parse();
    let results = results_dir(&args);
    match args.subcommand.as_str() {
        "index" => cmd_index(&args, &results),
        "startup" => cmd_startup(&args, &results),
        "startup-child" => cmd_startup_child(&args),
        "query" => cmd_query(&args, &results),
        "vectors" => vectors::run(&args, &results),
        "vectors-filtered" => vectors_filtered::run(&args, &results),
        "hybrid" => hybrid::run(&args, &results),
        "hybrid-merge" => hybrid::merge(&args, &results),
        "report" => report::run(&args),
        _ => usage(),
    }
}

// ---------------------------------------------------------------------------
// index
// ---------------------------------------------------------------------------

fn cmd_index(args: &Args, results: &Path) {
    let system = System::parse(&args.require("system"));
    let corpus = PathBuf::from(args.require("corpus"));
    let out = PathBuf::from(args.require("out"));
    let refresh_every = args.get_usize("refresh-every", 100_000);
    let threads = args.get_usize("threads", 1);
    let heap_mb = args.get_usize("heap-mb", 1024);
    let commit_every = args.get_usize("tantivy-commit-every", 0);
    let force_merge = args.get("force-merge").is_some();
    let name = label(args, system);

    if out.exists() {
        std::fs::remove_dir_all(&out).expect("remove old index dir");
    }
    std::fs::create_dir_all(&out).expect("create index dir");

    eprintln!(
        "[index] {name} -> {} (refresh every {refresh_every}, {threads} thread(s), {heap_mb} MB heap{})",
        out.display(),
        if force_merge { ", force-merge" } else { "" }
    );
    let t0 = Instant::now();
    let mut indexer = system.indexer(&out, threads, heap_mb << 20, commit_every, force_merge);
    let mut n = 0usize;
    let mut refreshes = 0usize;
    let docs = for_each_doc(&corpus, |doc| {
        indexer.add(&doc);
        n += 1;
        if refresh_every > 0 && n % refresh_every == 0 {
            indexer.refresh();
            refreshes += 1;
            eprintln!("[index]   {n} docs, {:.1}s", t0.elapsed().as_secs_f64());
        }
    })
    .expect("read corpus");
    let fin = indexer.finish();
    let segments = fin.segments;
    let wall = t0.elapsed();
    let bytes = common::dir_size(&out);
    let dps = docs as f64 / wall.as_secs_f64();
    println!(
        "{name}: {docs} docs in {:.2}s = {dps:.0} docs/s, {:.1} MB on disk, {segments} segment(s)",
        wall.as_secs_f64(),
        bytes as f64 / 1e6
    );
    let r = ResultFile::new(
        &name,
        "index",
        json!({
            "system": system.name(), "corpus": corpus, "out": out, "refresh_every": refresh_every,
            "threads": threads, "heap_mb": heap_mb, "tantivy_commit_every": commit_every,
            "force_merge": force_merge,
        }),
        json!({
            "docs": docs, "wall_s": wall.as_secs_f64(), "docs_per_sec": dps,
            "index_bytes": bytes, "segments": segments, "refreshes": refreshes,
            "force_merge_s": fin.force_merge_s,
        }),
    );
    let p = r.write(results).expect("write result");
    println!("wrote {}", p.display());
}

// ---------------------------------------------------------------------------
// startup (parent) / startup-child
// ---------------------------------------------------------------------------

/// Try to drop the OS page cache. Returns a label describing what happened.
fn purge_page_cache() -> (bool, String) {
    // `purge` needs root on modern macOS; try non-interactive sudo first so we
    // never block on a password prompt.
    for cmd in [vec!["sudo", "-n", "purge"], vec!["purge"]] {
        let ok = Command::new(cmd[0])
            .args(&cmd[1..])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return (true, format!("purged via `{}`", cmd.join(" ")));
        }
    }
    (false, "NOT purged (`purge` needs sudo; numbers are warm-page-cache)".to_string())
}

fn first_term_query(queries: &[Query]) -> &Query {
    queries
        .iter()
        .find(|q| q.class == "term")
        .or_else(|| queries.iter().find(|q| q.kind == "term"))
        .expect("queries.jsonl has no term query")
}

fn cmd_startup(args: &Args, results: &Path) {
    let system = System::parse(&args.require("system"));
    let index = PathBuf::from(args.require("index"));
    let queries_path = PathBuf::from(args.require("queries"));
    let runs = args.get_usize("runs", 5);
    let no_purge = args.get("no-purge").is_some();
    let name = label(args, system);
    let queries = load_queries(&queries_path).expect("load queries");
    let q = first_term_query(&queries);
    let exe = std::env::current_exe().expect("current_exe");

    let mut open_ms = Vec::new();
    let mut open_first_ms = Vec::new();
    let mut first_query_ms = Vec::new();
    let mut process_ms = Vec::new();
    let mut purge_labels = Vec::new();
    let mut hits = 0u64;
    for i in 0..runs {
        let (purged, label) = if no_purge {
            (false, "not attempted (--no-purge)".to_string())
        } else {
            purge_page_cache()
        };
        purge_labels.push(json!({"run": i, "purged": purged, "label": label}));
        let t0 = Instant::now();
        let out = Command::new(&exe)
            .args([
                "startup-child",
                "--system",
                system.name(),
                "--index",
                index.to_str().unwrap(),
                "--field",
                &q.field,
                "--term",
                q.term.as_deref().unwrap_or(""),
            ])
            .output()
            .expect("spawn startup-child");
        let wall = t0.elapsed();
        if !out.status.success() {
            eprintln!("startup-child failed: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("child JSON");
        let o = v["open_ms"].as_f64().unwrap();
        let fq = v["first_query_ms"].as_f64().unwrap();
        hits = v["hits"].as_u64().unwrap_or(0);
        open_ms.push(o);
        first_query_ms.push(fq);
        open_first_ms.push(o + fq);
        process_ms.push(wall.as_secs_f64() * 1e3);
        eprintln!(
            "[startup] run {i}: open {o:.2} ms, first query {fq:.2} ms, process wall {:.1} ms ({label})",
            wall.as_secs_f64() * 1e3
        );
    }
    let med = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        common::percentile_sorted(&s, 50.0)
    };
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let any_purged = purge_labels.iter().any(|p| p["purged"].as_bool() == Some(true));
    let cache_label = if any_purged { "purged" } else { "warm" };
    println!(
        "{name}: open median {:.2} ms, open+first-query median {:.2} ms, process wall median {:.1} ms [page cache: {cache_label}]",
        med(&open_ms),
        med(&open_first_ms),
        med(&process_ms)
    );
    let r = ResultFile::new(
        &name,
        "startup",
        json!({"system": system.name(), "index": index, "queries": queries_path, "runs": runs, "first_query": q.describe()}),
        json!({
            "page_cache": cache_label,
            "purge": purge_labels,
            "open_ms": open_ms, "first_query_ms": first_query_ms,
            "open_plus_first_query_ms": open_first_ms, "process_wall_ms": process_ms,
            "open_ms_median": med(&open_ms), "open_ms_min": min(&open_ms),
            "first_query_ms_median": med(&first_query_ms),
            "open_plus_first_query_ms_median": med(&open_first_ms),
            "process_wall_ms_median": med(&process_ms),
            "first_query_hits": hits,
        }),
    );
    let p = r.write(results).expect("write result");
    println!("wrote {}", p.display());
}

/// Child process: open the index, run one term query, print timings as JSON.
fn cmd_startup_child(args: &Args) {
    let system = System::parse(&args.require("system"));
    let index = PathBuf::from(args.require("index"));
    let q = Query {
        id: 0,
        class: "term".into(),
        field: args.require("field"),
        kind: "term".into(),
        term: Some(args.require("term")),
        terms: vec![],
        lower: None,
        upper: None,
    };
    let t0 = Instant::now();
    let searcher = system.open(&index);
    let open = t0.elapsed();
    let t1 = Instant::now();
    let r = searcher.run(&q);
    let first = t1.elapsed();
    println!(
        "{}",
        json!({
            "open_ms": open.as_secs_f64() * 1e3,
            "first_query_ms": first.as_secs_f64() * 1e3,
            "hits": r.total_hits,
            "segments": searcher.num_segments(),
        })
    );
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

fn cmd_query(args: &Args, results: &Path) {
    let system = System::parse(&args.require("system"));
    let index = PathBuf::from(args.require("index"));
    let queries_path = PathBuf::from(args.require("queries"));
    let warmup = args.get_usize("warmup", 2);
    let iters = args.get_usize("iters", 5);
    let global_warmup = args.get_usize("global-warmup", 1);
    let schedule = args.get_or("schedule", "grouped");
    assert!(["grouped", "mixed"].contains(&schedule.as_str()));
    let name = label(args, system);
    let queries = load_queries(&queries_path).expect("load queries");

    let t0 = Instant::now();
    let searcher = system.open(&index);
    eprintln!(
        "[query] {name} opened {} segment(s) in {:.1} ms; {} queries, warmup {warmup}, iters {iters}",
        searcher.num_segments(),
        t0.elapsed().as_secs_f64() * 1e3,
        queries.len()
    );

    // Untimed pass(es) over the whole set first (page cache, allocator; the
    // Lucene harness needs it for JIT and does the same), then per-query
    // warm-up + timed iterations.
    for _ in 0..global_warmup {
        for q in &queries {
            let _ = searcher.run(q);
        }
    }

    // Per class: pooled latencies of every timed execution.
    let mut class_lat: BTreeMap<String, Vec<Duration>> = BTreeMap::new();
    let mut class_hits: BTreeMap<String, u64> = BTreeMap::new();
    let mut class_n: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_query = Vec::with_capacity(queries.len());
    let bench_start = Instant::now();
    let mut mixed_lat = vec![Vec::new(); queries.len()];
    let mut mixed_last = vec![QueryResult::default(); queries.len()];
    if schedule == "mixed" {
        for r in 0..warmup+iters {
            for (qi,q) in queries.iter().enumerate() {
                let start=Instant::now(); let result=searcher.run(q); let elapsed=start.elapsed();
                if r>=warmup { mixed_lat[qi].push(elapsed); mixed_last[qi]=result; }
            }
        }
    }
    for (qi,q) in queries.iter().enumerate() {
        for _ in 0..if schedule == "grouped" { warmup } else { 0 } {
            let _ = searcher.run(q);
        }
        let mut lat = Vec::with_capacity(iters);
        let mut last = QueryResult::default();
        for _ in 0..if schedule == "grouped" { iters } else { 0 } {
            let s = Instant::now();
            last = searcher.run(q);
            lat.push(s.elapsed());
        }
        if schedule == "mixed" { lat=std::mem::take(&mut mixed_lat[qi]); last=mixed_last[qi].clone(); }
        let stats = LatencyStats::from_durations(&lat);
        class_lat.entry(q.class.clone()).or_default().extend(lat);
        *class_hits.entry(q.class.clone()).or_default() += last.total_hits;
        *class_n.entry(q.class.clone()).or_default() += 1;
        per_query.push(json!({
            "id": q.id, "class": q.class, "query": q.describe(),
            "hits": last.total_hits, "top_score": last.top_scores.first().copied().unwrap_or(0.0),
            "p50_us": stats.p50_us, "min_us": stats.min_us,
        }));
    }
    let bench_wall = bench_start.elapsed();

    let mut classes = serde_json::Map::new();
    println!("{:<8} {:>5} {:>12} {:>12} {:>12} {:>14}", "class", "n", "p50_us", "p99_us", "mean_us", "total_hits");
    for (class, lat) in &class_lat {
        let stats = LatencyStats::from_durations(lat);
        let hits = class_hits[class];
        let n = class_n[class];
        println!(
            "{class:<8} {n:>5} {:>12.1} {:>12.1} {:>12.1} {hits:>14}",
            stats.p50_us, stats.p99_us, stats.mean_us
        );
        classes.insert(
            class.clone(),
            json!({"n_queries": n, "total_hits": hits, "latency": stats}),
        );
    }

    let r = ResultFile::new(
        &name,
        "query",
        json!({
            "system": system.name(), "index": index, "queries": queries_path,
            "schedule":schedule,"warmup": warmup, "iters": iters, "global_warmup": global_warmup, "top_k": engine::TOP_K,
        }),
        json!({
            "segments": searcher.num_segments(),
            "bench_wall_s": bench_wall.as_secs_f64(),
            "classes": classes,
            "per_query": per_query,
        }),
    );
    let p = r.write(results).expect("write result");
    println!("wrote {}", p.display());

    // Cross-check hit counts against every other `*-query.json` already in
    // the results directory.
    let mut others: Vec<PathBuf> = std::fs::read_dir(results)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.ends_with("-query.json"))
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    others.retain(|o| o.file_name() != p.file_name());
    others.sort();
    for other_path in others {
        let other = other_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .trim_end_matches("-query.json")
            .to_string();
        let mismatches = report::compare_hits(&p, &other_path);
        if mismatches.is_empty() {
            println!("hit counts: {name} == {other} for all {} queries", queries.len());
        } else {
            println!("hit counts: {} MISMATCHES between {name} and {other}:", mismatches.len());
            for m in mismatches.iter().take(50) {
                println!("  {m}");
            }
            if mismatches.len() > 50 {
                println!("  ... {} more", mismatches.len() - 50);
            }
        }
    }
}
