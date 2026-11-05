//! Vector benchmark (rustic only; tantivy has no HNSW).
//!
//! Loads `vectors.f32`, encodes the base vectors into rustic's flat column
//! wire format (so the HNSW graph and brute force both read through
//! `FlatVectors`, exactly as a segment would), builds an HNSW graph
//! (m = 16, ef_construction = 100, L2) and reports build time, brute-force
//! QPS, and for ef in {50, 100, 200}: QPS, latency and recall@10 against
//! brute force.
//!
//! Note: `FlatVectorsBuilder::set` is O(n) per call (duplicate check), i.e.
//! O(n^2) for 200k vectors, so the column bytes are written directly in the
//! documented format instead (see `bench/results/ISSUES.md`).
//!
//! Flags beyond the Lucene-compatible ones:
//!
//! * `--threads N` — HNSW insertion threads (`1` = sequential, the default;
//!   `0` = all cores).
//! * `--quant none,int8,binary` — search modes to run on the one graph:
//!   full-precision `f32`, int8 scalar quantization and 1-bit (BBQ-style)
//!   quantization, the latter two rescoring the best `k * oversample`
//!   candidates with `f32`. Each mode writes its own result file
//!   (`<label>`, `<label>-int8`, `<label>-bbq`).
//! * `--oversample X` (default 3) — int8 rescore multiplier;
//!   `--bbq-oversample X` (default 10) — the 1-bit one (1 bit/dim at 128
//!   dims is coarse, so it needs a wider rescore window than int8).
//! * `--label NAME` — system name in the result files (default `rustic`).

use crate::common::{Args, LatencyStats, ResultFile};
use rustic_core::vector::{
    FlatVectors, HnswBuilder, HnswGraph, HnswParams, Quantization, QuantizedVectors, Similarity, VectorSource,
};
use serde_json::json;
use std::path::Path;
use std::time::{Duration, Instant};

/// Read a raw little-endian f32 file into (base vectors, query vectors).
pub(crate) fn load(path: &Path, dim: usize, n_queries: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let bytes = std::fs::read(path).expect("read vectors file");
    assert!(bytes.len() % (4 * dim) == 0, "vectors file size is not a multiple of dim*4");
    let total = bytes.len() / (4 * dim);
    assert!(total > n_queries, "vectors file holds fewer vectors than queries");
    let n = total - n_queries;
    let mut all: Vec<f32> = Vec::with_capacity(total * dim);
    for c in bytes.chunks_exact(4) {
        all.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    let queries = all[n * dim..].chunks_exact(dim).map(|v| v.to_vec()).collect();
    all.truncate(n * dim);
    (all, queries)
}

/// Encode a dense flat column in rustic's wire format:
/// `u32 dim, u32 num_docs, u8 0 (dense), pad to 4, f32 LE x dim x num_docs`.
fn encode_flat(dim: u32, vecs: &[f32]) -> Vec<u8> {
    let n = (vecs.len() / dim as usize) as u32;
    let mut out = Vec::with_capacity(12 + vecs.len() * 4);
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.push(0);
    while out.len() % 4 != 0 {
        out.push(0);
    }
    for v in vecs {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn recall(found: &[(u32, f32)], truth: &[(u32, f32)]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let hit = found.iter().filter(|(d, _)| truth.iter().any(|(t, _)| t == d)).count();
    hit as f64 / truth.len() as f64
}

pub fn run(args: &Args, results_dir: &Path) {
    let system = args.get_or("system", "rustic");
    if system != "rustic" {
        // Tantivy has no vector index; record N/A so `report` can say so.
        let r = ResultFile::new(
            &system,
            "vectors",
            json!({}),
            json!({"not_applicable": true, "reason": "tantivy 0.22 has no HNSW / vector index"}),
        );
        let p = r.write(results_dir).expect("write result");
        println!("{system}: vectors N/A (no HNSW). Wrote {}", p.display());
        return;
    }
    let path = args.require("vectors");
    let dim = args.get_usize("dim", 128);
    let n_queries = args.get_usize("queries", 20);
    let similarity_name = args.get_or("similarity", "l2");
    let similarity = match similarity_name.as_str() { "l2"=>Similarity::L2,"cosine"=>Similarity::Cosine,_=>panic!("unsupported similarity") };
    let k = args.get_usize("k", 10);
    let rounds = args.get_usize("rounds", 20);
    let warmup_rounds = args.get_usize("warmup-rounds", 5);
    let m = args.get_usize("m", 16);
    let ef_construction = args.get_usize("ef-construction", 100);
    let efs: Vec<usize> = args
        .get_or("ef", "50,100,200")
        .split(',')
        .map(|s| s.trim().parse().expect("ef list"))
        .collect();
    let threads = args.get_usize("threads", 1);
    let oversample_int8: f32 = args.get_or("oversample", "3").parse().expect("oversample");
    let oversample_bbq: f32 = args.get_or("bbq-oversample", "10").parse().expect("bbq-oversample");
    let label = args.get_or("label", "rustic");
    let quants: Vec<Quantization> = args
        .get_or("quant", "none")
        .split(',')
        .map(|s| Quantization::parse(s.trim()).expect("--quant: none|int8|binary"))
        .collect();

    eprintln!("[vectors] loading {path} ...");
    let (base, queries) = load(Path::new(&path), dim, n_queries);
    let n = base.len() / dim;
    eprintln!("[vectors] {n} base vectors, {} queries, dim {dim}", queries.len());

    let flat_bytes = encode_flat(dim as u32, &base);
    drop(base);
    let flat = FlatVectors::open(&flat_bytes).expect("FlatVectors::open");
    assert_eq!(flat.num_docs() as usize, n);

    // ---- Build ----
    eprintln!("[vectors] building HNSW m={m} ef_construction={ef_construction} threads={threads} ...");
    let t0 = Instant::now();
    let graph: HnswGraph = HnswBuilder::build(
        dim as u32,
        &flat,
        n as u32,
        similarity,
        HnswParams {
            m,
            ef_construction,
            threads,
            ..HnswParams::default()
        },
    );
    let build = t0.elapsed();
    let graph_bytes = graph.encode().len();
    eprintln!(
        "[vectors] built in {:.2}s, {} layers, {:.1} MB graph, mean degree {:.1}",
        build.as_secs_f64(),
        graph.num_layers(),
        graph_bytes as f64 / 1e6,
        graph.mean_degree()
    );

    // ---- Ground truth (brute force) ----
    let t0 = Instant::now();
    let truth: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| flat.brute_force_knn(q, k, similarity, |_| true))
        .collect();
    let brute = t0.elapsed();
    let brute_qps = queries.len() as f64 / brute.as_secs_f64();
    eprintln!("[vectors] brute force: {brute_qps:.1} QPS");

    for quant in quants {
        let (suffix, qv, quant_s) = match quant {
            Quantization::Int8 | Quantization::Binary => {
                let t0 = Instant::now();
                let qv = if quant == Quantization::Int8 {
                    QuantizedVectors::int8(&flat, similarity)
                } else {
                    QuantizedVectors::binary(&flat, similarity)
                };
                let s = t0.elapsed().as_secs_f64();
                eprintln!("[vectors] {quant:?} index: {:.1} MB codes, built in {s:.2}s", qv.code_bytes() as f64 / 1e6);
                (if quant == Quantization::Int8 { "-int8" } else { "-bbq" }, Some(qv), s)
            }
            _ => ("", None, 0.0),
        };
        let oversample = if quant == Quantization::Binary { oversample_bbq } else { oversample_int8 };
        let search = |q: &[f32], ef: usize| match &qv {
            Some(qv) => graph.search_quantized(q, k, ef, oversample, &flat as &dyn VectorSource, similarity, qv, |_| true),
            None => graph.search(q, k, ef, &flat as &dyn VectorSource, similarity, |_| true),
        };

        // ---- HNSW search per ef ----
        let mut runs = Vec::new();
        for &ef in &efs {
            let mut lat: Vec<Duration> = Vec::with_capacity(queries.len() * rounds);
            let mut rec_sum = 0.0;
            let mut rec_n = 0usize;
            // Warm-up rounds (not timed; same count as the Lucene harness).
            for _ in 0..warmup_rounds {
                for q in &queries {
                    let _ = search(q, ef);
                }
            }
            let t0 = Instant::now();
            for r in 0..rounds {
                for (qi, q) in queries.iter().enumerate() {
                    let s = Instant::now();
                    let found = search(q, ef);
                    lat.push(s.elapsed());
                    if r == 0 {
                        rec_sum += recall(&found, &truth[qi]);
                        rec_n += 1;
                    }
                }
            }
            let total = t0.elapsed();
            let qps = lat.len() as f64 / total.as_secs_f64();
            let stats = LatencyStats::from_durations(&lat);
            let recall_at_k = rec_sum / rec_n.max(1) as f64;
            eprintln!(
                "[vectors] {label}{suffix} ef={ef}: {qps:.0} QPS, p50 {:.0}us, recall@{k} {recall_at_k:.4}",
                stats.p50_us
            );
            runs.push(json!({
                "ef": ef,
                "qps": qps,
                "recall_at_k": recall_at_k,
                "latency": stats,
            }));
        }

        let system = format!("{label}{suffix}");
        let r = ResultFile::new(
            &system,
            "vectors",
            json!({
                "vectors": path, "dim": dim, "n": n, "queries": queries.len(), "k": k,
                "m": m, "ef_construction": ef_construction, "similarity": similarity_name, "rounds": rounds,
                "warmup_rounds": warmup_rounds, "threads": threads,
                "quantization": format!("{quant:?}").to_lowercase(),
                "oversample": if qv.is_some() { Some(oversample) } else { None },
            }),
            json!({
                "build_s": build.as_secs_f64(),
                "build_vectors_per_sec": n as f64 / build.as_secs_f64(),
                "quantize_s": quant_s,
                "quantized_code_bytes": qv.as_ref().map(|q| q.code_bytes()),
                "graph_bytes": graph_bytes,
                "graph_layers": graph.num_layers(),
                "graph_mean_degree": graph.mean_degree(),
                "brute_force_qps": brute_qps,
                "runs": runs,
            }),
        );
        let p = r.write(results_dir).expect("write result");
        println!("wrote {}", p.display());
    }
}
