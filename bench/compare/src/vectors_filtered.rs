//! `vectors-filtered`: filtered k-NN through the engine path.
//!
//! Every base vector `i` gets two payload fields derived from its index:
//! `tag = i % 100` and `bucket = i % 10` (i64 fast fields). The query
//! vectors are random cluster draws (independent of `i`), so the filters are
//! uncorrelated with vector position. Three filters per query `qi`:
//!
//! | name        | predicate             | selectivity |
//! |-------------|-----------------------|------------:|
//! | `tag_eq`    | `tag == qi % 100`     | 1 %  |
//! | `bucket_eq` | `bucket == qi % 10`   | 10 % |
//! | `bucket_lt5`| `bucket < 5`          | 50 % |
//!
//! rustic side: an in-memory `Index` (fields `tag`, `bucket` i64 + `emb`
//! vector l2_norm), one refresh, then `optimize_vectors()` with HNSW
//! m = 16 / ef_construction = 100 (the only place the engine builds graphs).
//! Each search is `Index::knn_search_with(KnnRequest { filter:
//! Some(RangeI64 ..), num_candidates: ef })` with the default
//! `PlannerConfig`, i.e. exactly what the engine does: the filter query is
//! evaluated per leaf, mapped to point ids, and `VectorStore::search`'s
//! planner picks exact scan vs filtered HNSW per segment. Filter
//! evaluation is inside the timed region (as it is for Lucene and Qdrant).
//!
//! Ground truth: brute force over the matching vectors only.

use crate::common::{Args, LatencyStats, ResultFile};
use rustic_core::index::{Index, KnnRequest};
use rustic_core::query::{Bound, LeafReader, Query};
use rustic_core::vector::{OptimizerPolicy, PlannerConfig, SearchStats};
use rustic_core::{Document, Field, FieldType, IndexSettings};
use serde_json::json;
use std::path::Path;
use std::time::{Duration, Instant};

const F_ID: u16 = 0;
const F_TAG: u16 = 1;
const F_BUCKET: u16 = 2;
const F_EMB: u16 = 3;

/// One filter: name, approx selectivity, and per-query (field, lo, hi) inclusive.
pub struct FilterSpec {
    pub name: &'static str,
    pub selectivity: f64,
    pub expr: &'static str,
    pub pred: fn(qi: usize) -> (u16, i64, i64),
}

pub fn filters() -> Vec<FilterSpec> {
    vec![
        FilterSpec { name: "tag_eq", selectivity: 0.01, expr: "tag == qi % 100", pred: |qi| (F_TAG, (qi % 100) as i64, (qi % 100) as i64) },
        FilterSpec { name: "bucket_eq", selectivity: 0.10, expr: "bucket == qi % 10", pred: |qi| (F_BUCKET, (qi % 10) as i64, (qi % 10) as i64) },
        FilterSpec { name: "bucket_lt5", selectivity: 0.50, expr: "bucket < 5", pred: |_| (F_BUCKET, 0, 4) },
    ]
}

fn value_of(field: u16, i: usize) -> i64 {
    if field == F_TAG {
        (i % 100) as i64
    } else {
        (i % 10) as i64
    }
}

fn load(path: &Path, dim: usize, n_queries: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let bytes = std::fs::read(path).expect("read vectors file");
    let total = bytes.len() / (4 * dim);
    let n = total - n_queries;
    let all: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let queries = all[n * dim..].chunks_exact(dim).map(|v| v.to_vec()).collect();
    let mut base = all;
    base.truncate(n * dim);
    (base, queries)
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact top-k ids among `i` with `lo <= value(field, i) <= hi`.
fn brute(base: &[f32], dim: usize, q: &[f32], k: usize, field: u16, lo: i64, hi: i64) -> Vec<u32> {
    let mut d: Vec<(f32, u32)> = base
        .chunks_exact(dim)
        .enumerate()
        .filter(|(i, _)| {
            let v = value_of(field, *i);
            v >= lo && v <= hi
        })
        .map(|(i, v)| (l2(v, q), i as u32))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(k);
    d.into_iter().map(|x| x.1).collect()
}

pub fn run(args: &Args, results_dir: &Path) {
    let path = args.require("vectors");
    let dim = args.get_usize("dim", 128);
    let n_queries = args.get_usize("queries", 20);
    let k = args.get_usize("k", 10);
    let rounds = args.get_usize("rounds", 20);
    let warmup_rounds = args.get_usize("warmup-rounds", 5);
    let m = args.get_usize("m", 16);
    let ef_construction = args.get_usize("ef-construction", 100);
    let threads = args.get_usize("threads", 0);
    let efs: Vec<usize> = args.get_or("ef", "100").split(',').map(|s| s.trim().parse().expect("ef list")).collect();
    let label = args.get_or("label", "rustic");

    eprintln!("[vectors-filtered] loading {path} ...");
    let (base, queries) = load(Path::new(&path), dim, n_queries);
    let n = base.len() / dim;

    let schema = vec![
        Field::new(F_ID, "id", FieldType::I64),
        Field::new(F_TAG, "tag", FieldType::I64),
        Field::new(F_BUCKET, "bucket", FieldType::I64),
        Field::new(F_EMB, "emb", FieldType::Vector { dim: dim as u16 }).with_similarity("l2_norm"),
    ];
    let mut idx = Index::new("vf", schema, IndexSettings::default());
    let mut policy = OptimizerPolicy::default();
    policy.hnsw.m = m;
    policy.hnsw.ef_construction = ef_construction;
    policy.hnsw.threads = threads;
    policy.indexing_threshold = 1000;
    policy.max_segments = 1;
    idx.optimizer_policy = policy;

    eprintln!("[vectors-filtered] indexing {n} docs, then optimize_vectors (m={m} efc={ef_construction} threads={threads}) ...");
    let t0 = Instant::now();
    for (i, v) in base.chunks_exact(dim).enumerate() {
        let mut d = Document::new();
        d.add_i64(F_ID, i as i64);
        d.add_i64(F_TAG, value_of(F_TAG, i));
        d.add_i64(F_BUCKET, value_of(F_BUCKET, i));
        d.add_vector(F_EMB, v.to_vec());
        idx.add_document(&d).expect("add");
    }
    idx.refresh().expect("refresh");
    let ingest_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let rep = idx.optimize_vectors().expect("optimize");
    let optimize_s = t1.elapsed().as_secs_f64();
    let build_s = t0.elapsed().as_secs_f64();
    let sealed: Vec<(u32, u32, bool)> =
        idx.vector_store().segments(F_EMB).iter().map(|s| (s.id, s.len(), s.is_sealed())).collect();
    eprintln!("[vectors-filtered] ingest {ingest_s:.2}s + optimize {optimize_s:.2}s; report {rep:?}; vector segments {sealed:?}");

    let leaves = idx.leaves();
    let id_of = |leaf: usize, doc: u32| leaves[leaf].i64_value(F_ID, doc).expect("id") as u32;
    let cfg = PlannerConfig::default();

    let mut out = Vec::new();
    for f in filters() {
        let truth: Vec<Vec<u32>> = queries
            .iter()
            .enumerate()
            .map(|(qi, q)| {
                let (fld, lo, hi) = (f.pred)(qi);
                brute(&base, dim, q, k, fld, lo, hi)
            })
            .collect();
        let reqs: Vec<KnnRequest> = queries
            .iter()
            .enumerate()
            .map(|(qi, q)| {
                let (fld, lo, hi) = (f.pred)(qi);
                let mut r = KnnRequest::new(F_EMB, q.clone(), k);
                r.filter = Some(Query::RangeI64 { field: fld, lower: Bound::Included(lo), upper: Bound::Included(hi) });
                r
            })
            .collect();
        for &ef in &efs {
            let reqs: Vec<KnnRequest> = reqs.iter().cloned().map(|mut r| { r.num_candidates = ef; r }).collect();
            for _ in 0..warmup_rounds {
                for r in &reqs {
                    let _ = idx.knn_search_with(r, &cfg).unwrap();
                }
            }
            let mut lat: Vec<Duration> = Vec::with_capacity(rounds * reqs.len());
            let mut rec = 0.0;
            let mut stats = SearchStats::default();
            let ts = Instant::now();
            for round in 0..rounds {
                for (qi, r) in reqs.iter().enumerate() {
                    let s = Instant::now();
                    let (hits, st) = idx.knn_search_with(r, &cfg).unwrap();
                    lat.push(s.elapsed());
                    if round == 0 {
                        stats += st;
                        let t = &truth[qi];
                        let hit = hits.iter().filter(|h| t.contains(&id_of(h.leaf, h.doc))).count();
                        rec += if t.is_empty() { 1.0 } else { hit as f64 / t.len() as f64 };
                    }
                }
            }
            let total = ts.elapsed().as_secs_f64();
            // Same searches with the point-id filter precomputed: isolates
            // VectorStore::search (planner + exact/HNSW) from the per-query
            // filter evaluation + doc->point mapping done by knn_search_with.
            let store = idx.vector_store();
            let pfilters: Vec<Vec<u64>> = (0..reqs.len())
                .map(|qi| {
                    let (fld, lo, hi) = (f.pred)(qi);
                    (0..n).filter(|&i| { let v = value_of(fld, i); v >= lo && v <= hi })
                        .map(|i| match &leaves[0] { LeafReader::Segment(s) => s.point_map().get(F_EMB, i as u32).expect("point"), _ => unreachable!() })
                        .collect()
                })
                .collect();
            for (qi, r) in reqs.iter().enumerate() {
                let _ = store.search(F_EMB, &r.vector, k, ef, Some(&pfilters[qi]), &cfg);
            }
            let mut slat: Vec<Duration> = Vec::with_capacity(rounds * reqs.len());
            let ts2 = Instant::now();
            for _ in 0..rounds {
                for (qi, r) in reqs.iter().enumerate() {
                    let s = Instant::now();
                    let _ = store.search(F_EMB, &r.vector, k, ef, Some(&pfilters[qi]), &cfg);
                    slat.push(s.elapsed());
                }
            }
            let store_qps = slat.len() as f64 / ts2.elapsed().as_secs_f64();
            let sst = LatencyStats::from_durations(&slat);
            let st = LatencyStats::from_durations(&lat);
            let qps = lat.len() as f64 / total;
            let recall = rec / reqs.len() as f64;
            let plan = format!(
                "exact_segments={} graph_segments={} fallbacks={} (over {} queries)",
                stats.exact_segments, stats.graph_segments, stats.fallbacks, reqs.len()
            );
            eprintln!("[vectors-filtered] {} ef={ef}: {qps:.0} QPS p50 {:.0}us recall {recall:.4}  [{plan}]  store-only: {store_qps:.0} QPS p50 {:.0}us", f.name, st.p50_us, sst.p50_us);
            out.push(json!({
                "filter": f.name, "expr": f.expr, "selectivity": f.selectivity, "ef": ef,
                "qps": qps, "recall_at_k": recall, "latency": st, "plan": plan,
                "store_only_qps": store_qps, "store_only_p50_us": sst.p50_us,
            }));
        }
    }

    let r = ResultFile::new(
        &label,
        "vectors-filtered",
        json!({
            "vectors": path, "dim": dim, "n": n, "queries": queries.len(), "k": k, "m": m,
            "ef_construction": ef_construction, "similarity": "l2", "rounds": rounds,
            "warmup_rounds": warmup_rounds, "threads": threads,
            "path": "Index::knn_search_with(KnnRequest{filter: RangeI64}) + PlannerConfig::default()",
            "payload": "tag = i % 100, bucket = i % 10 (i64 fast fields)",
        }),
        json!({
            "build_s": build_s, "ingest_s": ingest_s, "optimize_s": optimize_s,
            "runs": out,
        }),
    );
    let p = r.write(results_dir).expect("write");
    println!("wrote {}", p.display());
}
