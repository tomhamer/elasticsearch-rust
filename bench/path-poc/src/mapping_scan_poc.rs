//! Isolate profile-confirmed per-document mapping scans. Run on EC2 only.
//! Input is targeted_poc.py's Metricbeat load report, with its real mapping.
//! The candidate caches absence only; mappings with affected fields still
//! execute the existing implementation. This is not end-to-end throughput.
use elasticsearch_engine::{analysis::AnalysisRegistry, mapping::{FieldLookup, Mappings}};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let rounds: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    let registry = AnalysisRegistry::default();
    let mappings = Mappings::parse(&report["mapping"]["mappings"], &registry).unwrap();
    let lookup = FieldLookup::build(&mappings, &registry);
    let has_vectors = elasticsearch_engine::vector_source::has_synthetic_vectors(&lookup);
    let has_semantic = lookup.fields.values().any(|f| f.declared.is_semantic());
    // Constructing a new lookup after each mapping change would recompute
    // these properties; do not cache endpoint-dependent validation results.
    assert!(!has_vectors && !has_semantic, "Expected non-vector Metricbeat mapping");
    let mut results = Vec::new();
    for pass in 0..6 {
        for cached in if pass % 2 == 0 { [false, true] } else { [true, false] } {
            let started = Instant::now();
            for _ in 0..rounds {
                let lk = black_box(&lookup);
                let vector = if cached && !black_box(has_vectors) { false }
                    else { elasticsearch_engine::vector_source::has_synthetic_vectors(lk) };
                black_box(vector);
                if !cached || black_box(has_semantic) {
                    elasticsearch_engine::inference::semantic_map::validate_fields(black_box(&mappings), lk, None).unwrap();
                }
            }
            results.push(serde_json::json!({"pass":pass,"cached":cached,"ns_per_document":started.elapsed().as_nanos() as f64 / rounds as f64}));
        }
    }
    println!("{}", serde_json::json!({"scope":"component mapping scans only; not HTTP or Rally throughput", "fields":lookup.fields.len(),"documents":rounds,"results":results}));
}
