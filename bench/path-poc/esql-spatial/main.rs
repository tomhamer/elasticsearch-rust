//! POC: ES|QL spatial queries on the Rally `geopoint` corpus.
//!
//! Builds (once per `--dir`) the `osmgeopoints` index the track creates
//! (5 shards, `location: geo_point`) through the engine, then:
//!
//! * `current`: replays, step by step, what the ES|QL `FROM` path of
//!   master does for a spatial query (`load` in `esql/exec.rs`): `count`,
//!   one search for every matching hit with `_source`, the JSON round
//!   trip, `_source` -> row extraction, then the per-row predicate (which
//!   master cannot push to the index). Times and RSS per step.
//! * `pushed`: the same predicates as index queries (KD tree), the counts
//!   and doc-value aggregations the fast paths compute, step by step.
//! * `esql`: every geopoint ES|QL query of the track through
//!   `esql::run` (the integrated path of this build).
//!
//! ```text
//! es-esql-spatial-poc --corpus ~/rally-1pct/.../documents.json --dir /tmp/gp \
//!     --mode current|pushed|esql [--docs N] [--runs 5] [--only name]
//! ```

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::{Node, ResolveOpts};
use elasticsearch_engine::search::{self, ExecOpts};
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::BufRead;
use std::time::Instant;

fn status_kb(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok()).unwrap_or(0)
}

fn mem() -> String {
    format!("rss {} MB, peak {} MB", status_kb("VmRSS:") / 1024, status_kb("VmHWM:") / 1024)
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

const POLY: &str = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";

fn opts() -> ExecOpts<'static> {
    ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }
}

fn build(node: &Node, corpus: &str, n: usize) {
    if node.get_index("osmgeopoints").is_some() {
        return;
    }
    let t0 = Instant::now();
    let svc = node
        .create_index(
            "osmgeopoints",
            &json!({"settings": {"number_of_shards": 5, "number_of_replicas": 0, "refresh_interval": "-1"},
                    "mappings": {"dynamic": "strict", "properties": {"location": {"type": "geo_point"}}}}),
        )
        .unwrap();
    let f = std::io::BufReader::new(std::fs::File::open(corpus).expect("corpus"));
    let mut batch = Vec::with_capacity(10_000);
    for line in f.lines().take(n) {
        batch.push(WriteReq::Index { id: None, source: line.unwrap(), routing: None, cond: WriteCond::default() });
        if batch.len() == 10_000 {
            for r in svc.write(std::mem::take(&mut batch), false) {
                r.unwrap();
            }
        }
    }
    for r in svc.write(batch, false) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.flush().unwrap();
    svc.refresh().unwrap();
    eprintln!("indexed + force-merged in {:.1}s, {}", t0.elapsed().as_secs_f64(), mem());
}

/// Median and min wall ms of `runs` calls.
fn time<R>(runs: usize, mut f: impl FnMut() -> R) -> (f64, f64, R) {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        last = Some(f());
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0], last.unwrap())
}

/// Master's `load` for a residual (non-pushable) WHERE, step by step.
fn current(node: &Node, runs: usize) {
    let targets = node.resolve("osmgeopoints", ResolveOpts::from_params(&HashMap::new())).unwrap();
    let all = json!({"match_all": {}});
    let (ms, _, n) = time(runs, || search::count(&targets, Some(&all), None).unwrap());
    println!("current  count(match_all)                {ms:9.1} ms   -> {n} docs");
    // One search returning every hit, `_source` included (`size` = count).
    let body = json!({"query": all, "_source": true, "track_total_hits": false, "size": n});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let t0 = Instant::now();
    // ES|QL's searches lift the result window (`exec_opts`: an empty scroll id).
    let out = search::execute(&targets, &req, &ExecOpts { scroll_id: Some(String::new()), ..opts() }).unwrap();
    let search_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("current  search(size={n}, _source)       {search_ms:9.1} ms   json {} MB  ({})", out.json.len() >> 20, mem());
    let t0 = Instant::now();
    let mut v: Value = serde_json::from_str(&out.json).unwrap();
    drop(out);
    let hits = match v["hits"]["hits"].take() {
        Value::Array(h) => h,
        _ => Vec::new(),
    };
    println!("current  parse response -> {} hit Values   {:9.1} ms   ({})", hits.len(), t0.elapsed().as_secs_f64() * 1000.0, mem());
    // `_source` -> row: the location as WKT (what `extract` builds).
    let t0 = Instant::now();
    let rows: Vec<String> = hits.iter().filter_map(|h| elasticsearch_engine::esql::spatial::from_source(&h["_source"]["location"], "geo_point")).collect();
    println!("current  extract rows (WKT)               {:9.1} ms   ({})", t0.elapsed().as_secs_f64() * 1000.0, mem());
    drop(hits);
    // The residual predicate per row: both arguments parsed from WKT, related.
    let t0 = Instant::now();
    let mut matched = 0usize;
    for r in &rows {
        let a = elasticsearch_engine::esql::spatial::prims(r).unwrap();
        let b = elasticsearch_engine::esql::spatial::prims(&elasticsearch_engine::esql::spatial::parse_wkt(POLY, "geo_shape").unwrap()).unwrap();
        if rustic_core::fieldtypes::shape::relate(&a, &b, rustic_core::fieldtypes::shape::SpatialRelation::Intersects) {
            matched += 1;
        }
    }
    println!("current  per-row ST_INTERSECTS            {:9.1} ms   -> {matched} rows", t0.elapsed().as_secs_f64() * 1000.0);
}

/// The same work with predicates in the index and doc values.
fn pushed(node: &Node, runs: usize) {
    let targets = node.resolve("osmgeopoints", ResolveOpts::from_params(&HashMap::new())).unwrap();
    let poly = json!({"geo_shape": {"location": {"shape": POLY, "relation": "intersects"}}});
    let (ms, min, n) = time(runs, || search::count(&targets, Some(&poly), None).unwrap());
    println!("pushed   count(geo_shape intersects)      {ms:9.1} ms (min {min:.1})  -> {n}");
    let (ms, min, n) = time(runs, || search::par_count(&targets, &poly).unwrap());
    println!("pushed   KD leaf count (geo_shape)        {ms:9.1} ms (min {min:.1})  -> {n}");
    let dist = json!({"geo_distance": {"distance": "200000m", "location": {"lat": 55.0, "lon": 7.0}}});
    let (ms, min, n) = time(runs, || search::count(&targets, Some(&dist), None).unwrap());
    println!("pushed   count(geo_distance 200km, KD)    {ms:9.1} ms (min {min:.1})  -> {n}");
    // Centroid over geo doc values, shards in parallel.
    let all = json!({"match_all": {}});
    for (name, q) in [("match_all", &all), ("polygon", &poly)] {
        let (ms, min, (c, sx, sy)) = time(runs, || {
            let states = search::par_for_each_match(&targets, q, 1 << 16, &|_| (0u64, 0f64, 0f64), &|st: &mut (u64, f64, f64), p| {
                let id = p.lookup.get("location").and_then(|f| f.geo_id).unwrap();
                let dv = rustic_core::query::GeoDv::open(p.leaf, id);
                let mut add = |e: &rustic_core::geo::GeoPointEntry| {
                    st.0 += 1;
                    st.1 += rustic_core::geo::decode_lon(e.lon);
                    st.2 += rustic_core::geo::decode_lat(e.lat);
                };
                if p.whole_leaf {
                    dv.points().iter().for_each(&mut add);
                } else {
                    for d in p.doc_ids() {
                        dv.entries(d).iter().for_each(&mut add);
                    }
                }
                Ok(true)
            })
            .unwrap();
            states.into_iter().fold((0, 0.0, 0.0), |a, s| (a.0 + s.0, a.1 + s.1, a.2 + s.2))
        });
        println!("pushed   centroid doc values ({name:9})   {ms:9.1} ms (min {min:.1})  -> {c} pts ({:.6}, {:.6})", sx / c as f64, sy / c as f64);
    }
}

fn esql_queries() -> Vec<(&'static str, String)> {
    let raw = std::env::var("POC_QUERIES").ok().and_then(|p| std::fs::read_to_string(p).ok());
    if let Some(raw) = raw {
        // `name<TAB>query` lines (extracted from the Rally track).
        return raw
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .map(|(n, q)| (&*Box::leak(n.to_string().into_boxed_str()), q.to_string()))
            .collect();
    }
    vec![
        ("centroid-esql", "FROM osmgeopoints | STATS centroid=ST_CENTROID_AGG(location)".into()),
        ("polygon-intersects-count-esql", format!("FROM osmgeopoints | WHERE ST_Intersects(location, TO_GEOSHAPE(\"{POLY}\")) | STATS count=COUNT(*)")),
    ]
}

fn esql(node: &Node, runs: usize, only: Option<&str>) {
    for (name, q) in esql_queries() {
        if only.is_some_and(|o| !name.contains(o)) {
            continue;
        }
        let (ms, min, r) = time(runs, || elasticsearch_engine::esql::run(node, &q, Default::default()));
        let summary = match r {
            Ok(r) if std::env::var("POC_ALL_ROWS").is_ok() => format!("{} rows\n{}", r.table.rows.len(), r.table.rows.iter().map(|x| format!("  {x:?}")).collect::<Vec<_>>().join("\n")),
            Ok(r) => format!("{} rows, first {:?}", r.table.rows.len(), r.table.rows.first()).chars().take(140).collect::<String>(),
            Err(e) => format!("ERROR {}", e.reason),
        };
        println!("esql  {name:38} {ms:9.2} ms (min {min:8.2})  {}  {summary}", mem());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg(&args, "--corpus").unwrap_or_else(|| "documents.json".into());
    let dir = arg(&args, "--dir").unwrap_or_else(|| "/tmp/esql-spatial-poc".into());
    let n: usize = arg(&args, "--docs").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let runs: usize = arg(&args, "--runs").and_then(|v| v.parse().ok()).unwrap_or(5);
    let mode = arg(&args, "--mode").unwrap_or_else(|| "esql".into());
    let only = arg(&args, "--only");
    std::fs::create_dir_all(&dir).unwrap();
    let node = Node::open(std::path::Path::new(&dir)).unwrap();
    build(&node, &corpus, n);
    eprintln!("open: {}", mem());
    match mode.as_str() {
        "current" => current(&node, runs),
        "pushed" => pushed(&node, runs),
        _ => esql(&node, runs, only.as_deref()),
    }
}
