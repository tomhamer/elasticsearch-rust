//! `report`: merge `results/*.json` (from this harness and from
//! `bench/lucene`) into one markdown comparison, and the hit-count
//! cross-check used by `query`.

use crate::common::{Args, ResultFile};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

const SYSTEM_ORDER: [&str; 5] = ["rustic", "tantivy", "lucene", "qdrant", "hnswlib"];

fn load(path: &Path) -> Option<ResultFile> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn f(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn fmt_num(x: Option<f64>, decimals: usize) -> String {
    match x {
        Some(v) => format!("{v:.decimals$}"),
        None => "n/a".to_string(),
    }
}

fn fmt_int(x: Option<f64>) -> String {
    match x {
        Some(v) => {
            let s = format!("{:.0}", v);
            // thousands separators
            let mut out = String::new();
            for (i, c) in s.chars().rev().enumerate() {
                if i > 0 && i % 3 == 0 {
                    out.push(',');
                }
                out.push(c);
            }
            out.chars().rev().collect()
        }
        None => "n/a".to_string(),
    }
}

/// Compare per-query hit counts of two `query` result files. Returns one
/// line per mismatch.
pub fn compare_hits(a: &Path, b: &Path) -> Vec<String> {
    let (Some(ra), Some(rb)) = (load(a), load(b)) else {
        return vec![format!("could not load {} or {}", a.display(), b.display())];
    };
    let map = |r: &ResultFile| -> BTreeMap<u64, (String, u64)> {
        r.metrics["per_query"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|q| {
                        (
                            q["id"].as_u64().unwrap_or(0),
                            (
                                q["query"].as_str().unwrap_or("").to_string(),
                                q["hits"].as_u64().unwrap_or(0),
                            ),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let ma = map(&ra);
    let mb = map(&rb);
    let mut out = Vec::new();
    for (id, (query, ha)) in &ma {
        match mb.get(id) {
            Some((_, hb)) if hb == ha => {}
            Some((_, hb)) => out.push(format!(
                "q{id} {query}: {}={ha} {}={hb}",
                ra.system, rb.system
            )),
            None => out.push(format!("q{id} {query}: missing in {}", rb.system)),
        }
    }
    out
}

pub fn run(args: &Args) {
    let inputs = args.get_all("inputs");
    if inputs.is_empty() {
        eprintln!("report: --inputs results/*.json required");
        std::process::exit(2);
    }
    let out = args.get_or("out", "RESULTS.md");

    // (system, subcommand) -> result
    let mut res: BTreeMap<(String, String), ResultFile> = BTreeMap::new();
    for p in &inputs {
        if let Some(r) = load(Path::new(p)) {
            res.insert((r.system.clone(), r.subcommand.clone()), r);
        } else {
            eprintln!("report: skipping unreadable {p}");
        }
    }
    // Order: rustic, rustic-* variants, tantivy, tantivy-*, lucene, lucene-*.
    let base_rank = |name: &str| {
        SYSTEM_ORDER
            .iter()
            .position(|b| name == *b || name.starts_with(&format!("{b}-")))
            .unwrap_or(SYSTEM_ORDER.len())
    };
    let mut systems: Vec<String> = res.keys().map(|(sys, _)| sys.clone()).collect();
    systems.sort();
    systems.dedup();
    systems.sort_by_key(|s| (base_rank(s), s.len(), s.clone()));
    let get = |sys: &str, sub: &str| res.get(&(sys.to_string(), sub.to_string()));

    let mut md = String::new();
    md.push_str("# rustic vs Tantivy vs Lucene: benchmark results\n\n");
    // Machine of the main tables: the first non-hybrid result (hybrid runs
    // may come from another host; their section names its own machine).
    if let Some(any) = res.values().find(|r| r.subcommand != "hybrid").or_else(|| res.values().next()) {
        let m = &any.machine;
        md.push_str(&format!(
            "Machine: {} ({} logical / {} physical cores), {:.0} GB RAM, {} {}.\n\n",
            m.cpu,
            m.logical_cores,
            m.physical_cores,
            m.ram_bytes as f64 / (1u64 << 30) as f64,
            m.os,
            m.arch
        ));
    }
    md.push_str("Systems present: ");
    md.push_str(&systems.join(", "));
    md.push_str(". Missing cells mean that result file was not produced (e.g. Lucene not run).\n\n");

    // ---- Indexing ----
    md.push_str("## Indexing\n\n");
    md.push_str("| system | docs | wall (s) | docs/s | on-disk (MB) | segments | threads | refresh/commit every |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for sys in &systems {
        if let Some(r) = get(sys, "index") {
            let m = &r.metrics;
            let p = &r.params;
            let every = if sys.starts_with("rustic") {
                let mut e = format!("refresh {}", fmt_int(f(p, "refresh_every")));
                if p.get("force_merge").and_then(Value::as_bool) == Some(true) {
                    e.push_str(&format!(" + force-merge ({} s)", fmt_num(f(m, "force_merge_s"), 1)));
                }
                e
            } else {
                match f(p, "tantivy_commit_every").or_else(|| f(p, "commit_every")) {
                    Some(v) if v > 0.0 => format!("commit {}", fmt_int(Some(v))),
                    _ => "commit at end".into(),
                }
            };
            md.push_str(&format!(
                "| {sys} | {} | {} | {} | {} | {} | {} | {} |\n",
                fmt_int(f(m, "docs")),
                fmt_num(f(m, "wall_s"), 1),
                fmt_int(f(m, "docs_per_sec")),
                fmt_num(f(m, "index_bytes").map(|b| b / 1e6), 1),
                fmt_int(f(m, "segments")),
                fmt_int(f(p, "threads").or(Some(1.0))),
                every
            ));
        }
    }
    md.push('\n');

    // ---- Startup ----
    md.push_str("## Cold startup (fresh process per run, median of N runs)\n\n");
    md.push_str("| system | runs | page cache | open (ms) | first query (ms) | open + first query (ms) | process wall (ms) | runtime startup (ms) |\n");
    md.push_str("|---|---:|---|---:|---:|---:|---:|---:|\n");
    for sys in &systems {
        if let Some(r) = get(sys, "startup") {
            let m = &r.metrics;
            md.push_str(&format!(
                "| {sys} | {} | {} | {} | {} | {} | {} | {} |\n",
                fmt_int(f(&r.params, "runs")),
                m["page_cache"].as_str().unwrap_or("?"),
                fmt_num(f(m, "open_ms_median"), 2),
                fmt_num(f(m, "first_query_ms_median"), 2),
                fmt_num(f(m, "open_plus_first_query_ms_median"), 2),
                fmt_num(f(m, "process_wall_ms_median"), 1),
                fmt_num(f(m, "jvm_startup_ms_median"), 1),
            ));
        }
    }
    md.push_str("\n`open` = `Index::open` / `Index::open_in_dir` / `DirectoryReader.open`; `first query` = the first term query after open; `process wall` includes process spawn. `runtime startup` is JVM start-to-main (Lucene only).\n\n");

    // ---- Query latency ----
    md.push_str("## Warm query latency (µs; pooled over all queries x iterations of a class)\n\n");
    let mut classes: Vec<String> = Vec::new();
    for sys in &systems {
        if let Some(r) = get(sys, "query") {
            if let Some(obj) = r.metrics["classes"].as_object() {
                for k in obj.keys() {
                    if !classes.contains(k) {
                        classes.push(k.clone());
                    }
                }
            }
        }
    }
    let class_order = ["term", "and2", "or2", "and3", "phrase", "range", "tag"];
    classes.sort_by_key(|c| class_order.iter().position(|o| o == c).unwrap_or(usize::MAX));
    md.push_str("| class | n |");
    for sys in &systems {
        md.push_str(&format!(" {sys} p50 | {sys} p99 |"));
    }
    md.push_str(" hits (sum) |\n|---|---:|");
    for _ in &systems {
        md.push_str("---:|---:|");
    }
    md.push_str("---:|\n");
    for class in &classes {
        let mut row = String::new();
        let mut n = None;
        let mut hits: Vec<String> = Vec::new();
        for sys in &systems {
            let c = get(sys, "query").and_then(|r| r.metrics["classes"].get(class).cloned());
            match c {
                Some(c) => {
                    n = n.or(f(&c, "n_queries"));
                    let h = fmt_int(f(&c, "total_hits"));
                    if !hits.contains(&h) {
                        hits.push(h);
                    }
                    row.push_str(&format!(
                        " {} | {} |",
                        fmt_num(f(&c["latency"], "p50_us"), 1),
                        fmt_num(f(&c["latency"], "p99_us"), 1)
                    ));
                }
                None => row.push_str(" n/a | n/a |"),
            }
        }
        md.push_str(&format!(
            "| {class} | {} |{row} {} |\n",
            fmt_int(n),
            if hits.len() == 1 { hits[0].clone() } else { format!("DIFFER: {}", hits.join(" / ")) }
        ));
    }
    md.push_str("\nWarm-up and iteration counts: ");
    for sys in &systems {
        if let Some(r) = get(sys, "query") {
            md.push_str(&format!(
                "{sys} warmup={} iters={} segments={}; ",
                fmt_int(f(&r.params, "warmup")),
                fmt_int(f(&r.params, "iters")),
                fmt_int(f(&r.metrics, "segments"))
            ));
        }
    }
    md.push_str("\n\n");

    // ---- Hit-count mismatches ----
    md.push_str("## Hit-count cross-check\n\n");
    let query_files: Vec<(&str, &str)> = systems
        .iter()
        .filter_map(|s| {
            inputs
                .iter()
                .find(|p| p.ends_with(&format!("{s}-query.json")))
                .map(|p| (s.as_str(), p.as_str()))
        })
        .collect();
    if query_files.len() < 2 {
        md.push_str("Fewer than two systems have query results; nothing to compare.\n\n");
    } else {
        for i in 0..query_files.len() {
            for j in (i + 1)..query_files.len() {
                let (sa, pa) = query_files[i];
                let (sb, pb) = query_files[j];
                let mm = compare_hits(Path::new(pa), Path::new(pb));
                if mm.is_empty() {
                    md.push_str(&format!("* {sa} vs {sb}: all per-query hit counts identical.\n"));
                } else {
                    md.push_str(&format!("* {sa} vs {sb}: **{} mismatching queries**\n", mm.len()));
                    for m in mm.iter().take(40) {
                        md.push_str(&format!("  * `{m}`\n"));
                    }
                    if mm.len() > 40 {
                        md.push_str(&format!("  * ... and {} more\n", mm.len() - 40));
                    }
                }
            }
        }
        md.push('\n');
    }

    // ---- Vectors ----
    // `<sys>-mt` vector results are multi-threaded builds of `<sys>`: only
    // their build time is shown (in the base system's row).
    md.push_str("## Vectors, unfiltered (HNSW m=16, ef_construction=100, L2, 200k x 128, recall@10 vs brute force)\n\n");
    md.push_str("| system | n | dim | build 1 thread (s) | build all threads (s) | brute QPS | ef | QPS | p50 (µs) | server p50 (µs) | recall@10 |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for sys in &systems {
        if sys.ends_with("-mt") {
            continue;
        }
        if let Some(r) = get(sys, "vectors") {
            let m = &r.metrics;
            if m.get("not_applicable").and_then(Value::as_bool) == Some(true) {
                md.push_str(&format!(
                    "| {sys} | n/a | | | | | | | | | ({}) |\n",
                    m["reason"].as_str().unwrap_or("not applicable")
                ));
                continue;
            }
            // Build times: single-threaded builds report build_s; Qdrant's
            // build is always multi-threaded (max_indexing_threads = auto).
            let multi_only = sys.starts_with("qdrant");
            let b1 = if multi_only { None } else { f(m, "build_s") };
            let ball = if multi_only {
                f(m, "build_s")
            } else {
                f(m, "build_s_all_threads").or_else(|| get(&format!("{sys}-mt"), "vectors").and_then(|x| f(&x.metrics, "build_s")))
            };
            if let Some(runs) = m["runs"].as_array() {
                for (i, run) in runs.iter().enumerate() {
                    let first = i == 0;
                    md.push_str(&format!(
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                        if first { sys.as_str() } else { "" },
                        if first { fmt_int(f(&r.params, "n")) } else { String::new() },
                        if first { fmt_int(f(&r.params, "dim")) } else { String::new() },
                        if first { fmt_num(b1, 1) } else { String::new() },
                        if first { fmt_num(ball, 1) } else { String::new() },
                        if first { fmt_num(f(m, "brute_force_qps"), 1) } else { String::new() },
                        fmt_int(f(run, "ef")),
                        fmt_int(f(run, "qps")),
                        fmt_num(f(&run["latency"], "p50_us"), 0),
                        f(run, "server_p50_us").map(|v| format!("{v:.0}")).unwrap_or_else(|| "-".into()),
                        fmt_num(f(run, "recall_at_k"), 4),
                    ));
                }
            }
        }
    }
    md.push_str("\nQPS and p50 are single-threaded query loops in the client process. Lucene has no separate `ef` parameter: `KnnFloatVectorQuery(field, q, k=ef)` is used and truncated to 10, which is the closest equivalent. ");
    md.push_str("**Qdrant** QPS/p50 include a localhost HTTP/1.1 keep-alive round trip plus JSON encode/decode in a Python client (`server p50` is the `time` Qdrant reports in each response, i.e. its own handling time, which is the fairer column to compare with the in-process engines); its build is upload + optimizer until green, always multi-threaded. ");
    md.push_str("**hnswlib** is driven from Python, so its latencies include a few µs of Python→C++ call overhead per query. ");
    md.push_str("rustic `build all threads` = `HnswParams.threads = 0` (rayon pool).\n\n");

    // ---- Filtered vectors ----
    md.push_str("## Vectors, filtered (payload tag = i % 100, bucket = i % 10; ground truth = brute-force top-10 among matching vectors)\n\n");
    md.push_str("| filter | selectivity | system | ef | QPS | p50 (µs) | server p50 (µs) | recall@10 | notes |\n");
    md.push_str("|---|---:|---|---:|---:|---:|---:|---:|---|\n");
    let filter_order = ["tag_eq", "bucket_eq", "bucket_lt5"];
    let mut frows: Vec<(usize, usize, String, Value)> = Vec::new();
    for (si, sys) in systems.iter().enumerate() {
        if let Some(r) = get(sys, "vectors-filtered") {
            if let Some(runs) = r.metrics["runs"].as_array() {
                for run in runs {
                    let name = run["filter"].as_str().unwrap_or("?");
                    let fi = filter_order.iter().position(|x| *x == name).unwrap_or(99);
                    frows.push((fi, si, sys.clone(), run.clone()));
                }
            }
        }
    }
    frows.sort_by_key(|(fi, si, _, run)| (*fi, *si, f(run, "ef").unwrap_or(0.0) as u64));
    let mut last_filter = usize::MAX;
    for (fi, _, sys, run) in &frows {
        let new_group = *fi != last_filter;
        last_filter = *fi;
        let mut notes = Vec::new();
        if let Some(p) = run["plan"].as_str() {
            notes.push(p.split(" (over").next().unwrap_or(p).to_string());
        }
        if let Some(q) = f(run, "store_only_qps") {
            notes.push(format!("VectorStore::search alone: {} QPS, p50 {} µs", fmt_int(Some(q)), fmt_num(f(run, "store_only_p50_us"), 0)));
        }
        if sys.starts_with("hnswlib") {
            notes.push("Python filter callback per visited node".into());
        }
        if sys.starts_with("qdrant") {
            notes.push("incl. HTTP round trip".into());
        }
        md.push_str(&format!(
            "| {} | {} | {sys} | {} | {} | {} | {} | {} | {} |\n",
            if new_group { format!("`{}`", run["expr"].as_str().unwrap_or("?")) } else { String::new() },
            if new_group { format!("{:.0} %", f(run, "selectivity").unwrap_or(0.0) * 100.0) } else { String::new() },
            fmt_int(f(run, "ef")),
            fmt_int(f(run, "qps")),
            fmt_num(f(&run["latency"], "p50_us"), 0),
            f(run, "server_p50_us").map(|v| format!("{v:.0}")).unwrap_or_else(|| "-".into()),
            fmt_num(f(run, "recall_at_k"), 4),
            notes.join("; "),
        ));
    }
    md.push_str("\nFilter mechanics: rustic `Index::knn_search_with(KnnRequest { filter: RangeI64 })` with the default `PlannerConfig` (exact scan when <= 10 000 or <= 5 % of a segment's points match, else filtered HNSW; the filter query is evaluated and mapped to point ids per query, inside the timed region). ");
    md.push_str("Lucene `KnnFloatVectorQuery(field, q, k=ef, IntField.newExactQuery/newRangeQuery)` (Lucene picks exact vs filtered HNSW itself). ");
    md.push_str("Qdrant `match`/`range` filter on integer payload indexes (its planner uses `full_scan_threshold` = 10 MB). ");
    md.push_str("hnswlib `knn_query(filter=callable)`: the callable is Python, called for every candidate, so those numbers measure Python callback overhead rather than hnswlib's C++ search, and hnswlib has no exact-scan fallback.\n");

    let main_machine = res.values().find(|r| r.subcommand != "hybrid").map(|r| r.machine.clone());
    hybrid_section(&mut md, &systems, &|s: &str| get(s, "hybrid"), main_machine.as_ref());

    std::fs::write(&out, &md).expect("write report");
    println!("wrote {out}");
}

/// Mean overlap@k of two per-query top-id lists: |A ∩ B| / max(|A|, |B|)
/// (1 when both are empty).
fn overlap(a: &Value, b: &Value) -> Option<f64> {
    let (a, b) = (a.as_array()?, b.as_array()?);
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b) {
        let xs: Vec<i64> = x.as_array().map(|v| v.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
        let ys: Vec<i64> = y.as_array().map(|v| v.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
        let denom = xs.len().max(ys.len());
        sum += if denom == 0 { 1.0 } else { xs.iter().filter(|i| ys.contains(i)).count() as f64 / denom as f64 };
    }
    Some(sum / a.len() as f64)
}

fn hybrid_section<'a>(
    md: &mut String,
    systems: &[String],
    get: &dyn Fn(&str) -> Option<&'a ResultFile>,
    main_machine: Option<&crate::common::MachineInfo>,
) {
    let present: Vec<(&String, &ResultFile)> = systems.iter().filter_map(|s| get(s).map(|r| (s, r))).collect();
    if present.is_empty() {
        return;
    }
    md.push_str("\n## Hybrid search (BM25 `body` + k-NN, 200k docs, k = 10, num_candidates = 100, 100 queries)\n\n");
    // The hybrid results may come from a different host than the tables above.
    let mut machines: Vec<String> = Vec::new();
    for (sys, r) in &present {
        if r.metrics.get("not_applicable").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let m = &r.machine;
        let d = format!(
            "{}{} ({} logical / {} physical cores, {:.0} GB RAM, {} {})",
            m.cpu,
            if m.label.is_empty() { String::new() } else { format!(", {}", m.label) },
            m.logical_cores,
            m.physical_cores,
            m.ram_bytes as f64 / (1u64 << 30) as f64,
            m.os,
            if m.arch == "amd64" { "x86_64" } else { m.arch.as_str() } // JVM os.arch spelling
        );
        match machines.iter_mut().find(|x| x.starts_with(&d)) {
            Some(x) => x.push_str(&format!(", {sys}")),
            None => machines.push(format!("{d}: {sys}")),
        }
    }
    let differs = main_machine.is_some_and(|mm| {
        present.iter().any(|(_, r)| r.metrics.get("not_applicable").is_none() && (r.machine.cpu != mm.cpu || r.machine.label != mm.label))
    });
    md.push_str(&format!("Machine for this section: {}.", machines.join("; ")));
    if differs {
        md.push_str(&format!(
            " **These hybrid numbers come from a different machine than every other table in this file ({}): compare systems within this section only, not with the tables above.**",
            main_machine.map(|m| m.cpu.clone()).unwrap_or_default()
        ));
    }
    md.push_str("\n\n");
    let mut na = Vec::new();
    let mut rows: Vec<(usize, usize, usize, &String, &Value)> = Vec::new();
    let modes = ["sum", "rrf"];
    let filters = ["none", "price_p10", "price_p50"];
    for (si, (sys, r)) in present.iter().enumerate() {
        if r.metrics.get("not_applicable").and_then(Value::as_bool) == Some(true) {
            na.push(format!("{sys}: N/A ({})", r.metrics["reason"].as_str().unwrap_or("not applicable")));
            continue;
        }
        if let Some(runs) = r.metrics["runs"].as_array() {
            for run in runs {
                if run.get("untimed").and_then(Value::as_bool) == Some(true) {
                    continue;
                }
                let mi = modes.iter().position(|m| run["mode"] == *m).unwrap_or(9);
                let fi = filters.iter().position(|f| run["filter"] == *f).unwrap_or(9);
                rows.push((mi, fi, si, sys, run));
            }
        }
    }
    rows.sort_by_key(|(mi, fi, si, _, _)| (*mi, *fi, *si));
    md.push_str("| mode | filter | selectivity | system | QPS | p50 (µs) | p99 (µs) | server p50 (µs) | k-NN leg recall@10 |\n");
    md.push_str("|---|---|---:|---|---:|---:|---:|---:|---:|\n");
    let mut last = (99, 99);
    for (mi, fi, _, sys, run) in &rows {
        let new_group = (*mi, *fi) != last;
        last = (*mi, *fi);
        let fname = run["filter"].as_str().unwrap_or("?");
        md.push_str(&format!(
            "| {} | {} | {} | {sys} | {} | {} | {} | {} | {} |\n",
            if new_group { run["mode"].as_str().unwrap_or("?").to_string() } else { String::new() },
            if new_group { if fname == "none" { "none".to_string() } else { format!("`{fname}`") } } else { String::new() },
            if new_group { format!("{:.0} %", f(run, "selectivity").unwrap_or(1.0) * 100.0) } else { String::new() },
            fmt_int(f(run, "qps")),
            fmt_num(f(&run["latency"], "p50_us"), 0),
            fmt_num(f(&run["latency"], "p99_us"), 0),
            f(run, "server_p50_us").map(|v| format!("{v:.0}")).unwrap_or_else(|| "-".into()),
            fmt_num(f(run, "knn_recall_at_k"), 4),
        ));
    }
    if !na.is_empty() {
        md.push_str(&format!("\n{}.\n", na.join("; ")));
    }

    // Overlap@10 of the fused top-10 between systems.
    let find = |sys: &str, mode: &str, filter: &str| -> Option<&Value> {
        let r = get(sys)?;
        r.metrics["runs"].as_array()?.iter().find(|x| x["mode"] == mode && x["filter"] == filter).map(|x| &x["top_ids"])
    };
    let names: Vec<&String> = present
        .iter()
        .filter(|(_, r)| r.metrics.get("not_applicable").and_then(Value::as_bool) != Some(true))
        .map(|(s, _)| *s)
        .collect();
    let mut pairs: Vec<(String, String, String, String)> = Vec::new(); // (label, sysA, modeA, sysB/modeB)
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            pairs.push((format!("{} vs {}", names[i], names[j]), names[i].clone(), names[j].clone(), String::new()));
        }
    }
    md.push_str("\nOverlap@10 of the fused top-10 (mean over queries of |A ∩ B| / 10):\n\n| mode | filter |");
    for p in &pairs {
        md.push_str(&format!(" {} |", p.0));
    }
    // Only older rustic result files (BM25 with the legacy (k1 + 1) factor)
    // carry the `sum_lucene_scale` diagnostic.
    let has_diag = names.iter().any(|s| s.starts_with("rustic"))
        && names.iter().any(|s| s.as_str() == "lucene")
        && filters.iter().any(|f| find("rustic", "sum_lucene_scale", f).is_some());
    if has_diag {
        md.push_str(" rustic (k-NN boost 2.2) vs lucene |");
    }
    md.push_str("\n|---|---|");
    for _ in 0..pairs.len() + has_diag as usize {
        md.push_str("---:|");
    }
    md.push('\n');
    for mode in modes {
        for filter in filters {
            md.push_str(&format!("| {mode} | {filter} |"));
            for (_, a, b, _) in &pairs {
                let o = match (find(a, mode, filter), find(b, mode, filter)) {
                    (Some(x), Some(y)) => overlap(x, y),
                    _ => None,
                };
                md.push_str(&format!(" {} |", o.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into())));
            }
            if has_diag {
                let o = if mode == "sum" {
                    match (find("rustic", "sum_lucene_scale", filter), find("lucene", "sum", filter)) {
                        (Some(x), Some(y)) => overlap(x, y),
                        _ => None,
                    }
                } else {
                    None
                };
                md.push_str(&format!(" {} |", o.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into())));
            }
            md.push('\n');
        }
    }
    md.push_str("\nWorkload: query vector i (of 100, appended to `vectors-q100.f32`) paired with an `or2` (even i) or `term` (odd i) text query; doc i carries base vector i. ");
    md.push_str("`sum` = ES `query` + `knn` (bool.should: BM25 + k-NN score for docs in the global k-NN top-10, num_candidates 100): rustic `Index::hybrid_search`, Lucene `BooleanQuery(SHOULD text, SHOULD KnnFloatVectorQuery(k=100) truncated to the top 10)`, Qdrant a `formula` query `$score[0] + 1/(1+$score[1]^2)` over a BM25 prefetch (limit 100) and a dense prefetch (limit 10), which only approximates ES (k-NN-only docs outside the BM25 top-100 get text score 0). ");
    md.push_str("`rrf` = text top-100 + k-NN top-100 fused with RRF (rank_constant 60, window 100), top 10: rustic `fusion::rrf`, Lucene RRF in Java, Qdrant `prefetch` + `rrf`. ");
    md.push_str("Filters `price <= p10/p50 quantile` apply to the text query (bool filter) and inside the k-NN graph search. ");
    md.push_str("Qdrant's BM25 is a client-built sparse vector (`tf/(tf + k1(1-b+b·dl/avgdl))`) with the server-side `idf` modifier; its QPS/p50 include an HTTP round trip from a Python client (`server p50` is Qdrant's own time). ");
    if has_diag {
        md.push_str("Score scales: L2 scores are 1/(1+d²) in all three; rustic's BM25 keeps the legacy `(k1+1)` factor, i.e. 2.2x Lucene/ES's, so its `sum` weighs text 2.2x more than Lucene does; the last overlap column reruns rustic `sum` with the k-NN leg boosted by 2.2 (Lucene's BM25 scale, untimed) to isolate that difference. ");
    } else {
        md.push_str("Score scales: L2 scores are 1/(1+d²) in all three, and rustic's BM25 is Lucene's `BM25Similarity` (no legacy `(k1+1)` factor), so `sum` mixes the two legs on the same scale as Lucene/ES. ");
    }
    md.push_str("Lucene runs with the query cache disabled and exact total-hit counting (rustic always counts exactly).\n");
}
