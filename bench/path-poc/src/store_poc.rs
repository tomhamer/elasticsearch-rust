//! POC: verify that `bincode::serialize(doc)` in `store_document` is the
//! dominant per-doc cost in the HTTP indexing path, and that skipping the
//! `_source` field (which is already in the translog) during serialization
//! brings the store cost down to near-zero.
//!
//! Run: `es-path-poc store-bench`

use rustic_core::{Document, FieldValue};
use std::time::Instant;

const N_DOCS: usize = 100_000;
const N_ITERS: usize = 5;

/// Build a realistic metricbeat-like document: ~20 fields, one of which is
/// the full `_source` JSON string (~1 KB).
fn make_doc(i: usize) -> Document {
    let source = format!(
        r#"{{"@timestamp":"2026-01-01T00:00:00.000Z","host.name":"host{i}","event.dataset":"metricbeat","metricset":"system","cpu":{{"cores":8,"pct":12.5}},"memory":{{"total":16777216,"used":8388608}},"process":{{"name":"nginx","pid":1234}},"system":{{"load":{{"1":0.5,"5":0.3,"15":0.2}}}},"tags":["prod","us-east-1"],"message":"metricbeat document {i} with some extra padding to make the source realistic and a bit larger than the bare minimum for a typical metricbeat event document"}}"#
    );
    let mut d = Document::new();
    // _id
    d.fields.push((0, FieldValue::Keyword(format!("doc-{i}"))));
    // _source (the big one — ~1 KB)
    d.fields.push((1, FieldValue::Keyword(source)));
    // @timestamp
    d.fields.push((2, FieldValue::I64(1735689600000i64)));
    // host.name
    d.fields.push((3, FieldValue::Keyword(format!("host{i}"))));
    // event.dataset
    d.fields.push((4, FieldValue::Keyword("metricbeat".into())));
    // metricset
    d.fields.push((5, FieldValue::Keyword("system".into())));
    // cpu.pct
    d.fields.push((6, FieldValue::F64(12.5)));
    // memory.total
    d.fields.push((7, FieldValue::I64(16777216)));
    // memory.used
    d.fields.push((8, FieldValue::I64(8388608)));
    // process.name
    d.fields.push((9, FieldValue::Keyword("nginx".into())));
    // process.pid
    d.fields.push((10, FieldValue::I64(1234)));
    // system.load.1
    d.fields.push((11, FieldValue::F64(0.5)));
    // system.load.5
    d.fields.push((12, FieldValue::F64(0.3)));
    // system.load.15
    d.fields.push((13, FieldValue::F64(0.2)));
    // tags[0]
    d.fields.push((14, FieldValue::Keyword("prod".into())));
    // tags[1]
    d.fields.push((15, FieldValue::Keyword("us-east-1".into())));
    // message
    d.fields.push((16, FieldValue::Text(format!("metricbeat document {i}"))));
    // _version
    d.fields.push((17, FieldValue::I64(1)));
    // _seq_no
    d.fields.push((18, FieldValue::I64(i as i64)));
    // _primary_term
    d.fields.push((19, FieldValue::I64(1)));
    d
}

/// Current path: serialize the full document (including _source).
fn bench_full(docs: &[Document]) -> (f64, usize) {
    let mut total_bytes = 0usize;
    let start = Instant::now();
    for _ in 0..N_ITERS {
        for d in docs {
            let body = bincode::serialize(d).expect("Document serializes");
            total_bytes += body.len();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total_bytes)
}

/// Fixed path: serialize the document with the _source field (field id 1)
/// removed. The source is already in the translog; the store only needs it
/// for `get_doc` read-back, which can be served from the translog or a
/// separate source store.
fn bench_no_source(docs: &[Document]) -> (f64, usize) {
    let mut total_bytes = 0usize;
    let start = Instant::now();
    for _ in 0..N_ITERS {
        for d in docs {
            // Build a copy without the _source field (id 1).
            let filtered: Vec<(u16, FieldValue)> = d
                .fields
                .iter()
                .filter(|(f, _)| *f != 1)
                .cloned()
                .collect();
            let mut d2 = Document::new();
            d2.fields = filtered;
            let body = bincode::serialize(&d2).expect("Document serializes");
            total_bytes += body.len();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total_bytes)
}

pub fn run() {
    eprintln!("Building {N_DOCS} metricbeat-like documents...");
    let docs: Vec<Document> = (0..N_DOCS).map(make_doc).collect();

    // Measure the size of one full doc serialization vs no-source.
    let full_one = bincode::serialize(&docs[0]).unwrap();
    let no_src_one: Vec<(u16, FieldValue)> = docs[0].fields.iter().filter(|(f, _)| *f != 1).cloned().collect();
    let mut d2 = Document::new();
    d2.fields = no_src_one;
    let no_src_bytes = bincode::serialize(&d2).unwrap();

    let full_len = full_one.len();
    let no_src_len = no_src_bytes.len();
    eprintln!("Per-doc serialized size: full = {full_len} bytes, no-source = {no_src_len} bytes (savings: {}%)",
        (100.0 * (full_len - no_src_len) as f64 / full_len as f64));

    let (full_time, full_bytes) = bench_full(&docs);
    let (no_src_time, no_src_total_bytes) = bench_no_source(&docs);

    let full_ms = full_time * 1000.0 / N_ITERS as f64;
    let no_src_ms = no_src_time * 1000.0 / N_ITERS as f64;
    let full_us_per_doc = full_ms * 1000.0 / N_DOCS as f64;
    let no_src_us_per_doc = no_src_ms * 1000.0 / N_DOCS as f64;

    eprintln!("\n=== store_document serialization benchmark ===");
    eprintln!("Docs: {N_DOCS}, iters: {N_ITERS}");
    eprintln!("Full doc (with _source):  {full_ms:.1} ms/iter, {full_us_per_doc:.2} us/doc, {} bytes total", full_bytes / N_ITERS);
    eprintln!("No-source doc:            {no_src_ms:.1} ms/iter, {no_src_us_per_doc:.2} us/doc, {} bytes total", no_src_total_bytes / N_ITERS);
    eprintln!("Speedup: {:.1}x", full_time / no_src_time);
    eprintln!("Bytes saved per doc: {} bytes ({}%)",
        full_len - no_src_len,
        (100.0 * (full_len - no_src_len) as f64 / full_len as f64));

    // Estimate the impact on indexing throughput.
    // Current: 4,035 docs/s (Rally). If store serialization is X% of per-doc cost,
    // removing it gives a proportional speedup.
    let store_fraction_full = full_us_per_doc / 247.8; // 247.8 us/doc = 1/4035
    let store_fraction_nosrc = no_src_us_per_doc / 247.8;
    eprintln!("\nIf store serialization is the only per-doc cost (upper bound):");
    eprintln!("  Current store cost: {full_us_per_doc:.2} us/doc = {:.1}% of 247.8 us/doc budget", store_fraction_full * 100.0);
    eprintln!("  Fixed store cost:   {no_src_us_per_doc:.2} us/doc = {:.1}% of 247.8 us/doc budget", store_fraction_nosrc * 100.0);
    eprintln!("  Potential speedup (if store is 100% of cost): {:.1}x", full_time / no_src_time);
}
