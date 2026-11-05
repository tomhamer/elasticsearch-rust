//! POC: measure the per-document cost breakdown in the write path.
//!
//! The smoke test shows 1,993 docs/s (502 µs/doc) but ES9 does 46,553 docs/s
//! (21 µs/doc) — a 24× gap. The in-process write path is 53.6 µs/doc
//! (http_write_poc.rs), so the remaining ~450 µs/doc is in the engine's
//! `shard.apply` path.
//!
//! This POC measures each step of `apply_index`:
//!   1. `current()` — ID lookup (postings scan across segments)
//!   2. `core.add_document()` — add to memtable
//!
//! Run: `es-path-poc write-path-bench`

use std::time::Instant;

const N_DOCS: usize = 10_000;

pub fn run() {
    use elasticsearch_engine::analysis::AnalysisRegistry;
    use elasticsearch_engine::document::parse_with_limits;
    use elasticsearch_engine::fieldtypes::limits::Limits;
    use elasticsearch_engine::mapping::{FieldLookup, Mappings};
    use rustic_core::{Document, Field, FieldValue, FieldType, Index, IndexSettings};

    let dir = std::env::temp_dir().join(format!("write-path-poc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let schema = vec![
        Field::new(0, "_id", FieldType::Keyword),
        Field::new(1, "@timestamp", FieldType::Date),
        Field::new(2, "host.name", FieldType::Keyword),
        Field::new(3, "cpu.pct", FieldType::F64),
        Field::new(4, "memory.total", FieldType::I64),
    ];
    let mut idx = Index::create(&dir, "write-path-poc", schema, IndexSettings::default()).unwrap();

    let mappings = Mappings::default();
    let registry = AnalysisRegistry::default();
    let lookup = FieldLookup::build(&mappings, &registry);
    let limits = Limits::default();

    let json = r#"{"@timestamp":"2026-01-01T00:00:00Z","host.name":"host1","cpu.pct":12.5,"memory.total":16777216}"#;

    // Warmup: index 100 docs to populate the memtable.
    for i in 0..100 {
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        let (pd, _) = parse_with_limits(&v, &format!("warmup-{i}"), &mappings, &lookup, &limits).unwrap();
        idx.add_document(&pd.doc).unwrap();
    }

    // Measure each step of apply_index for N_DOCS new docs.
    let mut t_lookup = 0.0f64;
    let mut t_add_doc = 0.0f64;

    for i in 0..N_DOCS {
        let id = format!("doc-{i}");
        let v: serde_json::Value = serde_json::from_str(json).unwrap();

        // Parse (already measured at ~52 µs/doc in http_write_poc)
        let (pd, _) = parse_with_limits(&v, &id, &mappings, &lookup, &limits).unwrap();

        // Step 1: ID lookup (current()) — scan all segments' postings for the ID term.
        let t0 = Instant::now();
        let term = rustic_core::Term::new_text(0, &id); // _id field
        let _ = idx.search_term(&term, 1);
        t_lookup += t0.elapsed().as_secs_f64();

        // Step 2: add_document
        let t1 = Instant::now();
        idx.add_document(&pd.doc).unwrap();
        t_add_doc += t1.elapsed().as_secs_f64();
    }

    let lookup_us = t_lookup * 1_000_000.0 / N_DOCS as f64;
    let add_doc_us = t_add_doc * 1_000_000.0 / N_DOCS as f64;

    eprintln!("\n=== Write path cost breakdown (per doc) ===");
    eprintln!("Docs: {N_DOCS}");
    eprintln!("1. ID lookup (search_term):  {lookup_us:8.2} us/doc");
    eprintln!("2. add_document:             {add_doc_us:8.2} us/doc");
    eprintln!();
    eprintln!("Parse (from http_write_poc): ~52 us/doc");
    eprintln!("Total write path:            {:.2} us/doc", lookup_us + add_doc_us + 52.0);
    eprintln!();
    eprintln!("Smoke test:                  502 us/doc (1,993 docs/s)");
    eprintln!("ES9:                          21 us/doc (46,553 docs/s)");
    eprintln!("Gap:                          {:.2} us/doc", 502.0 - lookup_us - add_doc_us - 52.0);

    let _ = std::fs::remove_dir_all(&dir);
}
