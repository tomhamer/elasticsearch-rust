//! `hybrid`: BM25 text + k-NN vector in one request (ES `query` + `knn`).
//!
//! Workload (identical in `bench/lucene` `Hybrid.java` and
//! `bench/qdrant/hybrid.py`):
//!
//! * Index: the 200k corpus docs, `body` as BM25 text (k1 = 1.2, b = 0.75),
//!   `price` as a numeric fast field, and corpus doc `i`'s vector = base
//!   vector `i` of `vectors.f32` (128-d, L2, HNSW m = 16 / ef_construction
//!   = 100). One text segment, one vector segment.
//! * Queries: query vector `i` (i < `--n`, default 100) is paired with text
//!   query [`pair_text`]`(i)`: even `i` -> the `i/2`-th `or2` query, odd `i`
//!   -> the `i/2`-th `term` query of `queries.jsonl`.
//! * Filters: none, `price <= q10` (~10 %), `price <= q50` (~50 %) where
//!   `qX` is the nearest-rank X-th percentile of the corpus prices. With a
//!   filter the text query becomes `bool { must: text, filter: range }` and
//!   the same range is the k-NN pre-filter (applied during graph traversal).
//! * Modes:
//!   * `sum` (ES `query` + `knn`, bool.should): score = BM25 + knn score for
//!     docs in the global k-NN top-k (k = 10, num_candidates = 100); rustic
//!     `Index::hybrid_search(Some(&text), &[knn], 10)`.
//!   * `rrf`: text top-100 + k-NN top-100 (num_candidates = 100) fused with
//!     `rustic_core::vector::fusion::rrf(rank_constant = 60, window = 100)`,
//!     top 10.
//! * Per (mode, filter): `--warmup-rounds` untimed passes over all queries,
//!   then `--rounds` timed passes; p50 / p99 per query execution, QPS =
//!   executions / wall. Fused top-10 corpus ids of the first timed round are
//!   kept for the cross-system overlap@10. The k-NN leg alone (k = 10,
//!   num_candidates = 100, same filter) is run untimed for recall@10 vs
//!   brute force.
//! * BM25 is Lucene's/ES's (no legacy `(k1 + 1)` factor since the
//!   rustic-core fix), so `sum` scores are on the same scale in both.

use crate::common::{for_each_doc, load_queries, Args, LatencyStats, Query, ResultFile};
use rustic_core::index::{Index, KnnRequest};
use rustic_core::query::query::{Bound, Query as RQuery};
use rustic_core::query::searcher::{SearchOptions, Searcher};
use rustic_core::vector::fusion::rrf;
use rustic_core::vector::OptimizerPolicy;
use rustic_core::{Document, Field, FieldType, IndexSettings};
use serde_json::json;
use std::path::Path;
use std::time::{Duration, Instant};

const F_ID: u16 = 0;
const F_BODY: u16 = 1;
const F_PRICE: u16 = 2;
const F_EMB: u16 = 3;

pub const RANK_CONSTANT: f32 = 60.0;

/// Text query paired with query vector `i`.
pub fn pair_text(queries: &[Query], n: usize) -> Vec<Query> {
    let or2: Vec<&Query> = queries.iter().filter(|q| q.class == "or2").collect();
    let term: Vec<&Query> = queries.iter().filter(|q| q.class == "term").collect();
    (0..n)
        .map(|i| if i % 2 == 0 { or2[(i / 2) % or2.len()].clone() } else { term[(i / 2) % term.len()].clone() })
        .collect()
}

/// Nearest-rank percentile of the prices.
pub fn price_quantile(sorted: &[f64], p: f64) -> f64 {
    crate::common::percentile_sorted(sorted, p)
}

fn text_query(q: &Query, filter: Option<f64>) -> RQuery {
    let t = match q.kind.as_str() {
        "term" => RQuery::term(F_BODY, q.term.as_deref().unwrap_or("")),
        "or" => {
            let mut b = RQuery::bool_query();
            for t in &q.terms {
                b = b.should(RQuery::term(F_BODY, t));
            }
            b.build()
        }
        other => panic!("hybrid: unsupported text query kind {other}"),
    };
    match filter {
        None => t,
        Some(hi) => RQuery::bool_query().must(t).filter(price_filter(hi)).build(),
    }
}

fn price_filter(hi: f64) -> RQuery {
    RQuery::RangeF64 { field: F_PRICE, lower: Bound::Unbounded, upper: Bound::Included(hi) }
}

fn knn_req(v: &[f32], k: usize, nc: usize, filter: Option<f64>) -> KnnRequest {
    let mut r = KnnRequest::new(F_EMB, v.to_vec(), k);
    r.num_candidates = nc;
    r.filter = filter.map(price_filter);
    r
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn brute(base: &[f32], dim: usize, q: &[f32], k: usize, prices: &[f64], hi: Option<f64>) -> Vec<u32> {
    let mut d: Vec<(f32, u32)> = base
        .chunks_exact(dim)
        .enumerate()
        .filter(|(i, _)| hi.is_none_or(|h| prices[*i] <= h))
        .map(|(i, v)| (l2(v, q), i as u32))
        .collect();
    let kk = k.min(d.len());
    if kk == 0 {
        return Vec::new();
    }
    d.select_nth_unstable_by(kk - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(kk);
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().map(|x| x.1).collect()
}

fn not_applicable(system: &str, reason: &str, results_dir: &Path) {
    let r = ResultFile::new(system, "hybrid", json!({}), json!({"not_applicable": true, "reason": reason}));
    let p = r.write(results_dir).expect("write result");
    println!("{system}: hybrid N/A. Wrote {}", p.display());
}

pub fn run(args: &Args, results_dir: &Path) {
    let system = args.get_or("system", "rustic");
    if system == "tantivy" {
        return not_applicable("tantivy", "tantivy 0.22 has no vector index (text only)", results_dir);
    }
    if system == "hnswlib" {
        return not_applicable("hnswlib", "hnswlib has no text index (vectors only)", results_dir);
    }
    let corpus = args.require("corpus");
    let queries_path = args.require("queries");
    let vectors_path = args.require("vectors");
    let dim = args.get_usize("dim", 128);
    let vec_queries = args.get_usize("vec-queries", 100);
    let n_q = args.get_usize("n", 100);
    let k = args.get_usize("k", 10);
    let nc = args.get_usize("num-candidates", 100);
    let window = args.get_usize("window", 100);
    let rounds = args.get_usize("rounds", 10);
    let warmup_rounds = args.get_usize("warmup-rounds", 3);
    let m = args.get_usize("m", 16);
    let ef_construction = args.get_usize("ef-construction", 100);
    let threads = args.get_usize("threads", 0);
    let label = args.get_or("label", "rustic");

    eprintln!("[hybrid] loading vectors {vectors_path} ...");
    let (base, qvecs) = crate::vectors::load(Path::new(&vectors_path), dim, vec_queries);
    assert!(qvecs.len() >= n_q, "need {n_q} query vectors, file has {}", qvecs.len());
    let n = base.len() / dim;
    let texts = pair_text(&load_queries(Path::new(&queries_path)).expect("load queries"), n_q);

    let mut schema = vec![
        Field::new(F_ID, "id", FieldType::I64),
        Field::new(F_BODY, "body", FieldType::Text),
        Field::new(F_PRICE, "price", FieldType::F64),
        Field::new(F_EMB, "emb", FieldType::Vector { dim: dim as u16 }).with_similarity("l2_norm"),
    ];
    schema[0].indexed = false;
    schema[2].indexed = false;
    let mut idx = Index::new("hybrid", schema, IndexSettings::default());
    let mut policy = OptimizerPolicy::default();
    policy.hnsw.m = m;
    policy.hnsw.ef_construction = ef_construction;
    policy.hnsw.threads = threads;
    policy.indexing_threshold = 1000;
    // `--quantize`: also store an int8 copy, so graph searches traverse on
    // in-memory int8 codes and touch f32 only for the rescore (ES int8_hnsw).
    policy.quantize = args.get("quantize").is_some();
    policy.max_segments = 1;
    idx.optimizer_policy = policy;

    eprintln!("[hybrid] indexing {corpus} + vectors (m={m} efc={ef_construction} threads={threads}) ...");
    let t0 = Instant::now();
    let mut prices: Vec<f64> = Vec::with_capacity(n);
    let mut i = 0usize;
    for_each_doc(Path::new(&corpus), |doc| {
        assert!(i < n, "corpus has more docs than base vectors");
        let mut d = Document::new();
        d.add_i64(F_ID, doc.id);
        d.add_text(F_BODY, &doc.body);
        d.add_f64(F_PRICE, doc.price);
        d.add_vector(F_EMB, base[i * dim..(i + 1) * dim].to_vec());
        idx.add_document(&d).expect("add");
        prices.push(doc.price);
        i += 1;
    })
    .expect("read corpus");
    assert_eq!(i, n, "corpus docs != base vectors");
    idx.refresh().expect("refresh");
    let ingest_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    idx.optimize_vectors().expect("optimize_vectors");
    let optimize_s = t1.elapsed().as_secs_f64();
    let build_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "[hybrid] ingest {ingest_s:.2}s + optimize_vectors {optimize_s:.2}s; {} text segment(s), {} vector segment(s)",
        idx.num_segments(),
        idx.vector_store().segments(F_EMB).len()
    );

    let mut sorted = prices.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q10 = price_quantile(&sorted, 10.0);
    let q50 = price_quantile(&sorted, 50.0);
    let sel = |hi: f64| prices.iter().filter(|&&p| p <= hi).count() as f64 / n as f64;
    let only = args.get_or("filters", "");
    let filters: Vec<(&str, Option<f64>)> = vec![("none", None), ("price_p10", Some(q10)), ("price_p50", Some(q50))]
        .into_iter()
        .filter(|(n, _)| only.is_empty() || only.split(',').any(|o| o == *n))
        .collect();

    let leaves = idx.leaves();
    let id_of = |leaf: usize, doc: u32| -> i64 { leaves[leaf].i64_value(F_ID, doc).expect("id") };

    let mut runs = Vec::new();
    let mut filter_meta = Vec::new();
    for (fname, hi) in &filters {
        let selectivity = hi.map(sel).unwrap_or(1.0);
        filter_meta.push(json!({"name": fname, "price_lte": hi, "selectivity": selectivity}));
        // k-NN leg recall vs brute force (untimed).
        let tb = Instant::now();
        let truth: Vec<Vec<u32>> = qvecs[..n_q].iter().map(|q| brute(&base, dim, q, k, &prices, *hi)).collect();
        eprintln!("[hybrid] {fname}: brute force truth in {:.1}s", tb.elapsed().as_secs_f64());
        let mut rec = 0.0;
        for (qi, q) in qvecs[..n_q].iter().enumerate() {
            let hits = idx.knn_search(&knn_req(q, k, nc, *hi)).unwrap();
            let t = &truth[qi];
            let found = hits.iter().filter(|h| t.contains(&(id_of(h.leaf, h.doc) as u32))).count();
            rec += if t.is_empty() { 1.0 } else { found as f64 / t.len() as f64 };
        }
        let knn_recall = rec / n_q as f64;

        let tqs: Vec<RQuery> = texts.iter().map(|t| text_query(t, *hi)).collect();
        let sum_reqs: Vec<KnnRequest> = qvecs[..n_q].iter().map(|q| knn_req(q, k, nc, *hi)).collect();
        let rrf_reqs: Vec<KnnRequest> = qvecs[..n_q].iter().map(|q| knn_req(q, window, nc.max(window), *hi)).collect();

        let sum_fn = |qi: usize, boost: f32| -> Vec<(i64, f32)> {
            let mut r = sum_reqs[qi].clone();
            r.boost = boost;
            idx.hybrid_search(Some(&tqs[qi]), &[r], k).unwrap().into_iter().map(|h| (id_of(h.leaf, h.doc), h.score)).collect()
        };
        let rrf_fn = |qi: usize| -> Vec<(i64, f32)> {
            let td = Searcher::new(idx.leaves()).search(&tqs[qi], &SearchOptions { top_k: window, ..Default::default() });
            let text: Vec<((usize, u32), f32)> = td.hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
            let knn: Vec<((usize, u32), f32)> = idx.knn_search(&rrf_reqs[qi]).unwrap().into_iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
            let mut fused = rrf(&[text, knn], RANK_CONSTANT, window);
            fused.truncate(k);
            fused.into_iter().map(|((l, d), s)| (id_of(l, d), s)).collect()
        };

        for mode in ["sum", "rrf"] {
            let exec = |qi: usize| if mode == "sum" { sum_fn(qi, 1.0) } else { rrf_fn(qi) };
            for _ in 0..warmup_rounds {
                for qi in 0..n_q {
                    let _ = exec(qi);
                }
            }
            let mut lat: Vec<Duration> = Vec::with_capacity(rounds * n_q);
            let mut first: Vec<Vec<(i64, f32)>> = Vec::new();
            let ts = Instant::now();
            for r in 0..rounds {
                for qi in 0..n_q {
                    let s = Instant::now();
                    let out = exec(qi);
                    lat.push(s.elapsed());
                    if r == 0 {
                        first.push(out);
                    }
                }
            }
            let total = ts.elapsed().as_secs_f64();
            let st = LatencyStats::from_durations(&lat);
            let qps = lat.len() as f64 / total;
            eprintln!(
                "[hybrid] {label} {mode} filter={fname} ({:.0} %): {qps:.0} QPS p50 {:.0}us p99 {:.0}us; knn recall@{k} {knn_recall:.4}",
                selectivity * 100.0,
                st.p50_us,
                st.p99_us
            );
            runs.push(json!({
                "mode": mode, "filter": fname, "selectivity": selectivity, "qps": qps, "latency": st,
                "knn_recall_at_k": knn_recall,
                "mean_top1_score": first.iter().filter_map(|v| v.first().map(|x| x.1 as f64)).sum::<f64>() / n_q as f64,
                "top_ids": first.iter().map(|v| v.iter().map(|x| x.0).collect::<Vec<_>>()).collect::<Vec<_>>(),
            }));
        }
        // Component breakdown of the sum path (same requests, timed alone):
        // text top-k, k-NN leg, and the per-hit `Query::score_at` that
        // hybrid_search uses for k-NN-only docs.
        let time_mean = |f: &dyn Fn(usize)| -> f64 {
            for qi in 0..n_q {
                f(qi);
            }
            let t = Instant::now();
            for _ in 0..3 {
                for qi in 0..n_q {
                    f(qi);
                }
            }
            t.elapsed().as_secs_f64() * 1e6 / (3 * n_q) as f64
        };
        let text_us = time_mean(&|qi| {
            let _ = Searcher::new(idx.leaves()).search(&tqs[qi], &SearchOptions { top_k: k, ..Default::default() });
        });
        let knn_us = time_mean(&|qi| {
            let _ = idx.knn_search(&sum_reqs[qi]).unwrap();
        });
        let score_at_us = time_mean(&|qi| {
            let ls = idx.leaves();
            let ctx = rustic_core::query::query::SearchContext::for_query(&tqs[qi], &ls, rustic_core::query::Bm25Params::default());
            let _ = tqs[qi].score_at(&ls[0], &ctx, (qi * 1999) as u32 % n as u32);
        });
        eprintln!("[hybrid] {fname} breakdown: text top-{k} {text_us:.0}us, knn leg {knn_us:.0}us, one score_at {score_at_us:.0}us");
        if let Some(r) = runs.iter_mut().rev().find(|r| r["mode"] == "sum") {
            r["breakdown_us"] = json!({"text_top_k": text_us, "knn_leg": knn_us, "one_score_at": score_at_us});
        }
    }

    let r = ResultFile::new(
        &label,
        "hybrid",
        json!({
            "corpus": corpus, "queries": queries_path, "vectors": vectors_path, "dim": dim, "n_docs": n,
            "n_queries": n_q, "k": k, "num_candidates": nc, "rrf_window": window, "rank_constant": RANK_CONSTANT,
            "m": m, "ef_construction": ef_construction, "similarity": "l2 (score 1/(1+d^2))", "build_threads": threads,
            "rounds": rounds, "warmup_rounds": warmup_rounds,
            "text_pairing": "query vector i <-> even i: or2[i/2], odd i: term[i/2]",
            "sum_path": "Index::hybrid_search(Some(&text), &[KnnRequest{k:10,num_candidates:100,filter}], 10)",
            "rrf_path": "Searcher::search(text, top_k=100) + Index::knn_search(k=100,num_candidates=100) + fusion::rrf(60, 100)",
            "filter_path": "text: bool{must: text, filter: RangeF64(price)}; knn: KnnRequest.filter = RangeF64(price)",
            "bm25": "k1=1.2 b=0.75, Lucene BM25Similarity (no legacy (k1+1) factor), same scale as Lucene/ES",
        }),
        json!({
            "build_s": build_s, "ingest_s": ingest_s, "optimize_s": optimize_s,
            "text_segments": idx.num_segments(), "filters": filter_meta, "runs": runs,
        }),
    );
    let p = r.write(results_dir).expect("write");
    println!("wrote {}", p.display());
}

/// `hybrid-merge --inputs r1/x-hybrid.json r2/x-hybrid.json ... --results DIR`:
/// per system, keep for every (mode, filter) the round with the highest QPS
/// (better-of-N; single runs on this shared machine swing 10-20 %) and record
/// every round's QPS / p50 next to it. Writes `DIR/<system>-hybrid.json`.
pub fn merge(args: &Args, results_dir: &Path) {
    use std::collections::BTreeMap;
    let mut by_sys: BTreeMap<String, Vec<ResultFile>> = BTreeMap::new();
    for p in args.get_all("inputs") {
        let Ok(bytes) = std::fs::read(&p) else { continue };
        let Ok(r) = serde_json::from_slice::<ResultFile>(&bytes) else {
            eprintln!("hybrid-merge: skipping unreadable {p}");
            continue;
        };
        if r.subcommand == "hybrid" {
            by_sys.entry(r.system.clone()).or_default().push(r);
        }
    }
    for (sys, files) in by_sys {
        let mut best = files[0].clone();
        let runs: Vec<serde_json::Value> = best.metrics["runs"].as_array().cloned().unwrap_or_default();
        let mut merged = Vec::new();
        for run in runs {
            if run.get("untimed").and_then(|v| v.as_bool()) == Some(true) {
                merged.push(run);
                continue;
            }
            let same = |x: &&serde_json::Value| x["mode"] == run["mode"] && x["filter"] == run["filter"];
            let all: Vec<serde_json::Value> =
                files.iter().filter_map(|f| f.metrics["runs"].as_array().and_then(|rs| rs.iter().find(same)).cloned()).collect();
            let qps = |v: &serde_json::Value| v["qps"].as_f64().unwrap_or(0.0);
            let mut pick = all.iter().max_by(|a, b| qps(a).total_cmp(&qps(b))).cloned().unwrap_or(run.clone());
            pick["rounds_qps"] = json!(all.iter().map(qps).collect::<Vec<_>>());
            pick["rounds_p50_us"] = json!(all.iter().map(|v| v["latency"]["p50_us"].as_f64().unwrap_or(0.0)).collect::<Vec<_>>());
            merged.push(pick);
        }
        best.metrics["runs"] = json!(merged);
        best.metrics["rounds"] = json!(files.len());
        best.metrics["round_selection"] = json!("better of rounds (highest QPS) per mode/filter");
        if best.metrics["build_s"].is_null() {
            best.metrics["build_s"] = files.iter().map(|f| f.metrics["build_s"].clone()).find(|v| !v.is_null()).unwrap_or(json!(null));
        }
        let p = best.write(results_dir).expect("write");
        println!("hybrid-merge: {sys}: {} round(s) -> {}", files.len(), p.display());
    }
}
