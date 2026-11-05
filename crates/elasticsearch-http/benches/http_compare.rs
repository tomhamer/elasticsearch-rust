//! HTTP comparison harness: this server vs real Elasticsearch on identical
//! data (the bench corpus), driven over HTTP by one client so both servers
//! pay the same client cost. Reproduce with `bench/http/run.sh`.
//!
//! ```text
//! HB_TARGETS=rustic=http://127.0.0.1:9212,es=http://127.0.0.1:9211 \
//! HB_CORPUS=bench/corpus cargo bench -p elasticsearch-http --bench http_compare
//! ```
//!
//! Phases (`HB_PHASES`, default `ingest,load,check,search`):
//! * `ingest`: bulk throughput (docs/s) of `HB_INGEST_DOCS` text docs (no
//!   vectors) for every batch size in `HB_BATCHES` x client count in
//!   `HB_INGEST_CLIENTS`, into a scratch index that is deleted afterwards.
//! * `load`: loads the whole corpus (text + 128-d vectors) into `bench`,
//!   then refresh + force-merge to one segment on every target.
//! * `check`: sends the first `HB_CHECK` requests of every type to every
//!   target and compares hit counts, top ids (ties excluded), aggregations,
//!   `_count`, `_mget` and get results. Prints `MISMATCH` lines.
//! * `search`: QPS / p50 / p99 for each request type at each client count in
//!   `HB_CLIENTS`, `HB_SECS` seconds per run, `HB_ROUNDS` rounds with the
//!   targets interleaved (order alternates per round); best round reported.
//!
//! Every measurement is a JSON line on stdout; a Markdown summary is printed
//! at the end and written to `HB_OUT` when set.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| d.to_string())
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_list(k: &str, d: &str) -> Vec<usize> {
    env_str(k, d).split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

#[derive(Clone)]
struct Target {
    name: String,
    url: String,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

struct Corpus {
    /// Raw JSON lines (`{"id":..,"title":..,"body":..,"price":..,"ts":..,"tag":..}`).
    docs: Vec<String>,
    ids: Vec<u64>,
    dim: usize,
    vecs: Vec<f32>,
    qvecs: Vec<Vec<f32>>,
    queries: Vec<Value>,
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load_corpus(dir: &str, n_docs: usize) -> Corpus {
    let text = std::fs::read_to_string(format!("{dir}/corpus.jsonl")).expect("corpus.jsonl");
    let docs: Vec<String> = text.lines().take(n_docs).map(str::to_string).collect();
    let ids = docs.iter().map(|l| serde_json::from_str::<Value>(l).unwrap()["id"].as_u64().unwrap()).collect();
    let dim = 128;
    let all = read_f32(&format!("{dir}/vectors.f32"));
    let n_total = all.len() / dim;
    let n_base = n_total - 20;
    assert!(docs.len() <= n_base, "fewer vectors than docs");
    let vecs = all[..docs.len() * dim].to_vec();
    // Query vectors: vectors-q100.f32 (same base, 100 queries) when present.
    let mut qvecs: Vec<Vec<f32>> = (n_base..n_total).map(|i| all[i * dim..(i + 1) * dim].to_vec()).collect();
    if let Ok(meta) = std::fs::metadata(format!("{dir}/vectors-q100.f32")) {
        let q = read_f32(&format!("{dir}/vectors-q100.f32"));
        let nq = meta.len() as usize / 4 / dim;
        if nq >= n_base + 100 && q[..dim * 4] == all[..dim * 4] {
            qvecs = (n_base..n_base + 100).map(|i| q[i * dim..(i + 1) * dim].to_vec()).collect();
        }
    }
    let queries = std::fs::read_to_string(format!("{dir}/queries.jsonl")).expect("queries.jsonl").lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    Corpus { docs, ids, dim, vecs, qvecs, queries }
}

fn vec_json(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 11);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

/// Bulk bodies of `batch` docs each (`with_vec`: append the `vec` field).
fn bulk_payloads(c: &Corpus, n: usize, batch: usize, with_vec: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let mut s = String::new();
        for j in i..(i + batch).min(n) {
            let _ = write!(s, "{{\"index\":{{\"_id\":\"{}\"}}}}\n", c.ids[j]);
            let d = &c.docs[j];
            if with_vec {
                s.push_str(&d[..d.len() - 1]);
                s.push_str(",\"vec\":");
                s.push_str(&vec_json(&c.vecs[j * c.dim..(j + 1) * c.dim]));
                s.push('}');
            } else {
                s.push_str(d);
            }
            s.push('\n');
        }
        out.push(s);
        i += batch;
    }
    out
}

const TEXT_PROPS: &str = r#""id":{"type":"long"},"title":{"type":"text"},"body":{"type":"text"},"price":{"type":"double"},"ts":{"type":"date"},"tag":{"type":"keyword"}"#;

fn bench_mapping(with_vec: bool) -> String {
    let vec = if with_vec {
        r#","vec":{"type":"dense_vector","dims":128,"index":true,"similarity":"l2_norm","index_options":{"type":"hnsw","m":16,"ef_construction":100}}"#
    } else {
        ""
    };
    format!(r#"{{"settings":{{"number_of_shards":1,"number_of_replicas":0}},"mappings":{{"properties":{{{TEXT_PROPS}{vec}}}}}}}"#)
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Rq {
    method: reqwest::Method,
    path: String,
    body: Option<String>,
}

impl Rq {
    fn post(path: &str, body: String) -> Rq {
        Rq { method: reqwest::Method::POST, path: path.to_string(), body: Some(body) }
    }
    fn get(path: String) -> Rq {
        Rq { method: reqwest::Method::GET, path, body: None }
    }
}

async fn send(c: &reqwest::Client, base: &str, r: &Rq) -> (u16, bytes::Bytes) {
    let mut b = c.request(r.method.clone(), format!("{base}{}", r.path));
    if let Some(body) = &r.body {
        let ct = if r.path.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
        b = b.header("content-type", ct).body(body.clone());
    }
    match b.send().await {
        Ok(resp) => {
            let s = resp.status().as_u16();
            (s, resp.bytes().await.unwrap_or_default())
        }
        Err(e) => (599, bytes::Bytes::from(e.to_string())),
    }
}

async fn call(c: &reqwest::Client, base: &str, method: reqwest::Method, path: &str, body: Option<&str>) -> Value {
    let r = Rq { method, path: path.to_string(), body: body.map(str::to_string) };
    let (s, b) = send(c, base, &r).await;
    let v: Value = serde_json::from_slice(&b).unwrap_or(Value::Null);
    if s >= 300 && s != 404 {
        eprintln!("{base}{path} -> {s}: {}", String::from_utf8_lossy(&b[..b.len().min(500)]));
    }
    v
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

async fn bulk_all(c: &reqwest::Client, t: &Target, payloads: Arc<Vec<String>>, clients: usize) -> f64 {
    let next = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..clients {
        let (c, u, p, n) = (c.clone(), t.url.clone(), payloads.clone(), next.clone());
        tasks.push(tokio::spawn(async move {
            loop {
                let k = n.fetch_add(1, Ordering::Relaxed);
                if k >= p.len() {
                    break;
                }
                let (s, b) = send(&c, &u, &Rq::post("/_bulk", p[k].clone())).await;
                let v: Value = serde_json::from_slice(&b).unwrap_or(Value::Null);
                assert!(s == 200 && v["errors"] == false, "bulk failed on {u}: {s} {}", String::from_utf8_lossy(&b[..b.len().min(800)]));
            }
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    start.elapsed().as_secs_f64()
}

async fn recreate(c: &reqwest::Client, t: &Target, index: &str, body: &str) {
    call(c, &t.url, reqwest::Method::DELETE, &format!("/{index}"), None).await;
    let v = call(c, &t.url, reqwest::Method::PUT, &format!("/{index}"), Some(body)).await;
    assert_eq!(v["acknowledged"], true, "create {index} on {}: {v}", t.name);
}

/// Rewrite the `_bulk` payloads to target `index`.
fn retarget(p: &[String], index: &str) -> Vec<String> {
    p.iter().map(|s| s.replace("{\"index\":{", &format!("{{\"index\":{{\"_index\":\"{index}\","))).collect()
}

// ---------------------------------------------------------------------------
// Request mix
// ---------------------------------------------------------------------------

const SRC: &str = r#""_source":{"excludes":["vec"]}"#;

fn requests(c: &Corpus, n: usize) -> Vec<(&'static str, Vec<Rq>)> {
    let by = |cls: &str| -> Vec<&Value> { c.queries.iter().filter(|q| q["class"] == cls).collect() };
    let (or2, and2, phrase, range, tag, term) = (by("or2"), by("and2"), by("phrase"), by("range"), by("tag"), by("term"));
    let terms = |q: &Value| q["terms"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect::<Vec<_>>().join(" ");
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut tags: Vec<String> = tag.iter().map(|q| q["term"].as_str().unwrap().to_string()).collect();
    tags.sort();
    tags.dedup();
    let s = "/bench/_search";
    let mut out: Vec<(&'static str, Vec<Rq>)> = Vec::new();
    let mut add = |name: &'static str, f: &mut dyn FnMut(usize, &mut Rng) -> Rq| {
        let v = (0..n).map(|i| f(i, &mut rng)).collect();
        out.push((name, v));
    };
    add("match", &mut |i, _| Rq::post(s, format!(r#"{{"query":{{"match":{{"body":"{}"}}}},{SRC}}}"#, terms(or2[i % or2.len()]))));
    add("bool_filter", &mut |i, r| {
        let q = and2[i % and2.len()];
        let rg = range[r.below(range.len())];
        Rq::post(
            s,
            format!(
                r#"{{"query":{{"bool":{{"must":[{{"match":{{"body":"{}"}}}}],"filter":[{{"term":{{"tag":"{}"}}}},{{"range":{{"price":{{"gte":{},"lte":{}}}}}}}]}}}},{SRC}}}"#,
                q["terms"][0].as_str().unwrap(),
                tags[r.below(tags.len())],
                rg["lower"].as_f64().unwrap().min(50.0),
                (rg["upper"].as_f64().unwrap() + 40.0).min(100.0)
            ),
        )
    });
    add("phrase", &mut |i, _| Rq::post(s, format!(r#"{{"query":{{"match_phrase":{{"body":"{}"}}}},{SRC}}}"#, terms(phrase[i % phrase.len()]))));
    add("range", &mut |i, _| {
        let q = range[i % range.len()];
        Rq::post(s, format!(r#"{{"query":{{"range":{{"price":{{"gte":{},"lte":{}}}}}}},{SRC}}}"#, q["lower"], q["upper"]))
    });
    add("terms_agg", &mut |i, _| {
        // Half over everything, half over a text query's matches.
        let q = if i % 2 == 0 { "{\"match_all\":{}}".to_string() } else { format!(r#"{{"match":{{"body":"{}"}}}}"#, terms(or2[i % or2.len()])) };
        Rq::post(s, format!(r#"{{"size":0,"query":{q},"aggs":{{"t":{{"terms":{{"field":"tag","size":10}}}}}}}}"#))
    });
    add("date_histogram", &mut |i, _| {
        let q = if i % 2 == 0 { "{\"match_all\":{}}".to_string() } else { format!(r#"{{"match":{{"body":"{}"}}}}"#, terms(or2[i % or2.len()])) };
        Rq::post(s, format!(r#"{{"size":0,"query":{q},"aggs":{{"h":{{"date_histogram":{{"field":"ts","calendar_interval":"month"}},"aggs":{{"p":{{"avg":{{"field":"price"}}}}}}}}}}}}"#))
    });
    add("composite_agg", &mut |i, _| {
        let q = if i % 2 == 0 { "{\"match_all\":{}}".to_string() } else { format!(r#"{{"match":{{"body":"{}"}}}}"#, terms(or2[i % or2.len()])) };
        Rq::post(
            s,
            format!(r#"{{"size":0,"query":{q},"aggs":{{"c":{{"composite":{{"size":100,"sources":[{{"tag":{{"terms":{{"field":"tag"}}}}}},{{"m":{{"date_histogram":{{"field":"ts","calendar_interval":"month"}}}}}}]}}}}}}}}"#),
        )
    });
    let qv = |i: usize| vec_json(&c.qvecs[i % c.qvecs.len()]);
    add("knn", &mut |i, _| Rq::post(s, format!(r#"{{"knn":{{"field":"vec","query_vector":{},"k":10,"num_candidates":100}},{SRC}}}"#, qv(i))));
    add("knn_filtered", &mut |i, r| {
        Rq::post(
            s,
            format!(r#"{{"knn":{{"field":"vec","query_vector":{},"k":10,"num_candidates":100,"filter":{{"term":{{"tag":"{}"}}}}}},{SRC}}}"#, qv(i), tags[r.below(tags.len())]),
        )
    });
    add("knn_hybrid", &mut |i, _| {
        Rq::post(
            s,
            format!(r#"{{"query":{{"match":{{"body":"{}"}}}},"knn":{{"field":"vec","query_vector":{},"k":10,"num_candidates":100}},{SRC}}}"#, terms(or2[i % or2.len()]), qv(i)),
        )
    });
    add("sort_search_after", &mut |_, r| {
        let p = r.below(9_900) as f64 / 100.0;
        Rq::post(s, format!(r#"{{"query":{{"match_all":{{}}}},"sort":[{{"price":"asc"}},{{"id":"asc"}}],"search_after":[{p},0],"size":10,{SRC}}}"#))
    });
    add("count", &mut |i, _| Rq::post("/bench/_count", format!(r#"{{"query":{{"match":{{"body":"{}"}}}}}}"#, term[i % term.len()]["term"].as_str().unwrap())));
    let nd = c.ids.len();
    add("mget", &mut |_, r| {
        let ids: Vec<String> = (0..10).map(|_| format!("\"{}\"", c.ids[r.below(nd)])).collect();
        Rq::post("/bench/_mget?_source_excludes=vec", format!(r#"{{"ids":[{}]}}"#, ids.join(",")))
    });
    add("get", &mut |_, r| Rq::get(format!("/bench/_doc/{}?_source_excludes=vec", c.ids[r.below(nd)])));
    // The same aggregations bypassing the shard request cache (ES caches
    // `size: 0` responses by default): engine work on every request.
    for (name, nc) in [("terms_agg", "terms_agg_nocache"), ("date_histogram", "date_histogram_nocache"), ("composite_agg", "composite_agg_nocache")] {
        let v: Vec<Rq> = out.iter().find(|(n, _)| *n == name).unwrap().1.iter().map(|r| Rq { path: format!("{}?request_cache=false", r.path), ..r.clone() }).collect();
        out.push((nc, v));
    }
    out
}

// ---------------------------------------------------------------------------
// Agreement check
// ---------------------------------------------------------------------------

fn num_eq(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// First difference between two JSON values (numbers approximately).
fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (!num_eq(x.as_f64().unwrap(), y.as_f64().unwrap())).then(|| format!("{path}: {a} vs {b}")),
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in x {
                match y.get(k) {
                    Some(w) => {
                        if let Some(d) = diff(v, w, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    None => return Some(format!("{path}.{k}: missing on second")),
                }
            }
            y.keys().find(|k| !x.contains_key(*k)).map(|k| format!("{path}.{k}: missing on first"))
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: len {} vs {}", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().find_map(|(i, (v, w))| diff(v, w, &format!("{path}[{i}]")))
        }
        _ => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}

/// Top ids in order. Unsorted requests: hits with equal scores (relative
/// 1e-5) form a group compared as a set, since tie order depends on
/// segment-internal doc ids (insertion order under concurrent bulks); the
/// last group is dropped when the page is full (who makes the cut-off is a
/// tie too).
fn top_ids(v: &Value, sorted: bool) -> Vec<String> {
    let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let id = |h: &Value| h["_id"].as_str().unwrap_or("").to_string();
    if sorted {
        return hits.iter().map(id).collect();
    }
    let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
    for h in &hits {
        let s = h["_score"].as_f64().unwrap_or(0.0);
        match groups.last_mut() {
            Some((g, ids)) if (s - *g).abs() <= 1e-5 * g.abs().max(1e-9) => ids.push(id(h)),
            _ => groups.push((s, vec![id(h)])),
        }
    }
    if hits.len() == 10 {
        groups.pop();
    }
    groups
        .into_iter()
        .flat_map(|(_, mut ids)| {
            ids.sort();
            ids
        })
        .collect()
}

#[derive(Default)]
struct CheckStats {
    compared: usize,
    mismatches: usize,
    knn_overlap: Vec<f64>,
    notes: Vec<String>,
}

async fn check(c: &reqwest::Client, targets: &[Target], reqs: &[(&'static str, Vec<Rq>)], n: usize) -> BTreeMap<&'static str, CheckStats> {
    let mut out = BTreeMap::new();
    for (name, list) in reqs {
        let mut st = CheckStats::default();
        for (i, r) in list.iter().take(n).enumerate() {
            let mut r = r.clone();
            if r.path.contains("/_search") {
                r.body = r.body.map(|b| format!("{{\"track_total_hits\":true,{}", &b[1..]));
            }
            let mut res = Vec::new();
            for t in targets {
                let (s, b) = send(c, &t.url, &r).await;
                res.push((s, serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null)));
            }
            let (s0, v0) = &res[0];
            for (t, (s1, v1)) in targets.iter().zip(&res).skip(1) {
                st.compared += 1;
                let mut bad: Option<String> = None;
                if s0 != s1 {
                    bad = Some(format!("status {s0} vs {s1}: {} / {}", v0.to_string().chars().take(300).collect::<String>(), v1.to_string().chars().take(300).collect::<String>()));
                } else if *name == "count" {
                    if v0["count"] != v1["count"] {
                        bad = Some(format!("count {} vs {}", v0["count"], v1["count"]));
                    }
                } else if *name == "mget" || *name == "get" {
                    let strip = |v: &Value| -> Value {
                        let mut v = v.clone();
                        let docs: Vec<&mut Value> = match v.get_mut("docs").and_then(Value::as_array_mut) {
                            Some(a) => a.iter_mut().collect(),
                            None => vec![],
                        };
                        for d in docs {
                            if let Some(o) = d.as_object_mut() {
                                for k in ["_seq_no", "_primary_term", "_version"] {
                                    o.remove(k);
                                }
                            }
                        }
                        if let Some(o) = v.as_object_mut() {
                            for k in ["_seq_no", "_primary_term", "_version"] {
                                o.remove(k);
                            }
                        }
                        v
                    };
                    bad = diff(&strip(v0), &strip(v1), "");
                } else {
                    let (t0, t1) = (&v0["hits"]["total"]["value"], &v1["hits"]["total"]["value"]);
                    if name.starts_with("knn") {
                        let (a, b) = (top_ids(v0, true), top_ids(v1, true));
                        let ov = a.iter().filter(|x| b.contains(x)).count() as f64 / a.len().max(1) as f64;
                        st.knn_overlap.push(ov);
                        if *name == "knn_filtered" {
                            // Every hit must satisfy the filter on both.
                        }
                    } else if t0 != t1 {
                        bad = Some(format!("total {t0} vs {t1}"));
                    } else if v0.get("aggregations").is_some() || v1.get("aggregations").is_some() {
                        bad = diff(&v0["aggregations"], &v1["aggregations"], "aggregations");
                    } else {
                        let sorted = *name == "sort_search_after";
                        let (a, b) = (top_ids(v0, sorted), top_ids(v1, sorted));
                        if a != b {
                            bad = Some(format!("top ids {a:?} vs {b:?}"));
                        }
                    }
                }
                if let Some(d) = bad {
                    st.mismatches += 1;
                    let line = format!("MISMATCH type={name} req#{i} {} vs {}: {d} | request {} {}", targets[0].name, t.name, r.path, r.body.as_deref().unwrap_or("").chars().take(400).collect::<String>());
                    println!("{line}");
                    if st.notes.len() < 3 {
                        st.notes.push(d.chars().take(300).collect());
                    }
                }
            }
        }
        out.insert(*name, st);
    }
    out
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Run {
    qps: f64,
    p50: f64,
    p99: f64,
    errors: usize,
}

async fn run_load(c: &reqwest::Client, t: &Target, list: Arc<Vec<Rq>>, clients: usize, secs: f64, seed: usize) -> Run {
    let stop = Arc::new(AtomicBool::new(false));
    let next = Arc::new(AtomicUsize::new(seed));
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..clients {
        let (c, u, l, n, stop) = (c.clone(), t.url.clone(), list.clone(), next.clone(), stop.clone());
        tasks.push(tokio::spawn(async move {
            let mut lat = Vec::with_capacity(4096);
            let mut errors = 0;
            while !stop.load(Ordering::Relaxed) {
                let i = n.fetch_add(1, Ordering::Relaxed) % l.len();
                let t0 = Instant::now();
                let (s, _) = send(&c, &u, &l[i]).await;
                if s >= 300 {
                    errors += 1;
                }
                lat.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            (lat, errors)
        }));
    }
    tokio::time::sleep(Duration::from_secs_f64(secs)).await;
    stop.store(true, Ordering::Relaxed);
    let mut all = Vec::new();
    let mut errors = 0;
    for t in tasks {
        let (l, e) = t.await.unwrap();
        all.extend(l);
        errors += e;
    }
    let el = start.elapsed().as_secs_f64();
    all.sort_by(|a, b| a.total_cmp(b));
    Run { qps: all.len() as f64 / el, p50: pct(&all, 0.5), p99: pct(&all, 0.99), errors }
}

// ---------------------------------------------------------------------------

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(env_usize("HB_CLIENT_THREADS", 4)).enable_all().build().unwrap();
    rt.block_on(async {
        let targets: Vec<Target> = env_str("HB_TARGETS", "rustic=http://127.0.0.1:9212,es=http://127.0.0.1:9211")
            .split(',')
            .map(|p| {
                let (n, u) = p.split_once('=').expect("HB_TARGETS: name=url,...");
                Target { name: n.into(), url: u.trim_end_matches('/').into() }
            })
            .collect();
        let phases = env_str("HB_PHASES", "ingest,load,check,search");
        let corpus_dir = env_str("HB_CORPUS", "bench/corpus");
        let n_docs = env_usize("HB_DOCS", 200_000);
        let t0 = Instant::now();
        let corpus = load_corpus(&corpus_dir, n_docs);
        eprintln!("corpus: {} docs, {} query vectors ({:.1}s)", corpus.docs.len(), corpus.qvecs.len(), t0.elapsed().as_secs_f64());
        let client = reqwest::Client::builder().no_gzip().pool_max_idle_per_host(256).timeout(Duration::from_secs(3600)).build().unwrap();
        let mut md = String::new();
        let names: Vec<&str> = targets.iter().map(|t| t.name.as_str()).collect();
        let _ = writeln!(md, "Targets: {}\n", targets.iter().map(|t| format!("`{}` = {}", t.name, t.url)).collect::<Vec<_>>().join(", "));

        if phases.contains("ingest") {
            let n = env_usize("HB_INGEST_DOCS", 50_000).min(corpus.docs.len());
            let rounds = env_usize("HB_INGEST_ROUNDS", 1);
            let _ = writeln!(md, "### Bulk ingest ({n} text docs, no vectors; docs/s, best of {rounds})\n");
            let _ = writeln!(md, "| batch | clients | {} |", names.join(" | "));
            let _ = writeln!(md, "|---|---|{}", "---|".repeat(names.len()));
            for batch in env_list("HB_BATCHES", "100,1000,5000") {
                let payloads = Arc::new(retarget(&bulk_payloads(&corpus, n, batch, false), "ingest"));
                for clients in env_list("HB_INGEST_CLIENTS", "1,4,16") {
                    let mut best = vec![0f64; targets.len()];
                    for round in 0..rounds {
                        for k in 0..targets.len() {
                            let ti = if round % 2 == 0 { k } else { targets.len() - 1 - k };
                            let t = &targets[ti];
                            recreate(&client, t, "ingest", &bench_mapping(false)).await;
                            let secs = bulk_all(&client, t, payloads.clone(), clients).await;
                            let dps = n as f64 / secs;
                            println!("{{\"bench\":\"ingest\",\"target\":\"{}\",\"batch\":{batch},\"clients\":{clients},\"docs\":{n},\"secs\":{secs:.3},\"docs_per_sec\":{dps:.0}}}", t.name);
                            best[ti] = best[ti].max(dps);
                            call(&client, &t.url, reqwest::Method::DELETE, "/ingest", None).await;
                        }
                    }
                    let _ = writeln!(md, "| {batch} | {clients} | {} |", best.iter().map(|d| format!("{d:.0}")).collect::<Vec<_>>().join(" | "));
                }
            }
            md.push('\n');
        }

        if phases.contains("load") {
            let batch = env_usize("HB_LOAD_BATCH", 1000);
            let clients = env_usize("HB_LOAD_CLIENTS", 8);
            let payloads = Arc::new(retarget(&bulk_payloads(&corpus, corpus.docs.len(), batch, true), "bench"));
            let _ = writeln!(md, "### Full load ({} docs incl. 128-d vectors, batch {batch}, {clients} clients)\n", corpus.docs.len());
            let _ = writeln!(md, "| target | bulk docs/s | refresh+force_merge s |\n|---|---|---|");
            for t in &targets {
                recreate(&client, t, "bench", &bench_mapping(true)).await;
                let secs = bulk_all(&client, t, payloads.clone(), clients).await;
                let t1 = Instant::now();
                call(&client, &t.url, reqwest::Method::POST, "/bench/_refresh", None).await;
                call(&client, &t.url, reqwest::Method::POST, "/bench/_forcemerge?max_num_segments=1", None).await;
                call(&client, &t.url, reqwest::Method::POST, "/bench/_refresh", None).await;
                let fm = t1.elapsed().as_secs_f64();
                let dps = corpus.docs.len() as f64 / secs;
                println!("{{\"bench\":\"load\",\"target\":\"{}\",\"docs_per_sec\":{dps:.0},\"merge_secs\":{fm:.1}}}", t.name);
                let _ = writeln!(md, "| {} | {dps:.0} | {fm:.1} |", t.name);
            }
            md.push('\n');
        }

        let reqs = requests(&corpus, env_usize("HB_REQUESTS", 500));
        let only = env_str("HB_ONLY", "");
        let reqs: Vec<(&'static str, Vec<Rq>)> = reqs.into_iter().filter(|(n, _)| only.is_empty() || only.split(',').any(|o| o == *n)).collect();

        if phases.contains("check") && targets.len() > 1 {
            let n = env_usize("HB_CHECK", 50);
            let st = check(&client, &targets, &reqs, n).await;
            let _ = writeln!(md, "### Agreement ({} vs {}, first {n} requests per type, `track_total_hits: true`)\n", targets[0].name, targets[1..].iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(","));
            let _ = writeln!(md, "| type | compared | mismatches | note |\n|---|---|---|---|");
            for (name, s) in &st {
                let note = if !s.knn_overlap.is_empty() {
                    format!("top-10 overlap {:.3} (approximate kNN)", s.knn_overlap.iter().sum::<f64>() / s.knn_overlap.len() as f64)
                } else {
                    s.notes.first().cloned().unwrap_or_default().replace('|', "\\|")
                };
                let _ = writeln!(md, "| {name} | {} | {} | {note} |", s.compared, s.mismatches);
            }
            md.push('\n');
        }

        if phases.contains("search") {
            let secs = env_usize("HB_SECS", 5) as f64;
            let rounds = env_usize("HB_ROUNDS", 2);
            let clients = env_list("HB_CLIENTS", "1,8,32");
            let _ = writeln!(md, "### Search / read ({secs}s per run, best of {rounds} interleaved rounds; QPS, p50/p99 ms)\n");
            let hdr: Vec<String> = names.iter().map(|n| format!("{n} QPS | {n} p50 | {n} p99")).collect();
            let ratio = if targets.len() == 2 { format!(" {}/{} QPS |", names[0], names[1]) } else { String::new() };
            let _ = writeln!(md, "| request | clients | {} |{ratio}", hdr.join(" | "));
            let _ = writeln!(md, "|---|---|{}{}", "---|".repeat(3 * names.len()), if targets.len() == 2 { "---|" } else { "" });
            for (name, list) in &reqs {
                let list = Arc::new(list.clone());
                for t in &targets {
                    run_load(&client, t, list.clone(), 4, 1.0, 0).await; // warm-up
                }
                for &cl in &clients {
                    let mut best = vec![Run::default(); targets.len()];
                    for round in 0..rounds {
                        for k in 0..targets.len() {
                            let ti = if round % 2 == 0 { k } else { targets.len() - 1 - k };
                            let r = run_load(&client, &targets[ti], list.clone(), cl, secs, round * 7919).await;
                            println!(
                                "{{\"bench\":\"search\",\"type\":\"{name}\",\"target\":\"{}\",\"clients\":{cl},\"round\":{round},\"qps\":{:.0},\"p50_ms\":{:.3},\"p99_ms\":{:.3},\"errors\":{}}}",
                                targets[ti].name, r.qps, r.p50, r.p99, r.errors
                            );
                            if r.qps > best[ti].qps {
                                best[ti] = r;
                            }
                        }
                    }
                    let cells: Vec<String> = best.iter().map(|r| format!("{:.0}{} | {:.2} | {:.2}", r.qps, if r.errors > 0 { " (errors!)" } else { "" }, r.p50, r.p99)).collect();
                    let ratio = if targets.len() == 2 { format!(" {:.2}x |", best[0].qps / best[1].qps.max(1e-9)) } else { String::new() };
                    let _ = writeln!(md, "| {name} | {cl} | {} |{ratio}", cells.join(" | "));
                }
            }
            md.push('\n');
        }

        println!("\n{md}");
        if let Ok(p) = std::env::var("HB_OUT") {
            std::fs::write(&p, &md).unwrap();
        }
    });
}
