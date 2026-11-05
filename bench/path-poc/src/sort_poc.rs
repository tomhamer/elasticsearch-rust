//! POC: Geonames ascending population sort — isolate the per-doc fast-field
//! column-open cost that dominates the numeric sort path.
//!
//! The sort collector's `primary_numeric` closure (searcher.rs) calls
//! `leaf.fast_fields().get_i64(field)` for **every doc** in the loop. For a
//! segment leaf that is `FastFieldsReader::get_i64` → `ColumnReader::open`,
//! which re-parses the column header (num_docs, block counts, mins/maxs/values
//! byte slices) on each call. This POC times:
//!
//!   1. `get_i64(field).get(doc)` per doc  (the current per-doc open path)
//!   2. open the column once, then `col.get(doc)` per doc  (the fix)
//!
//! over a synthetic Geonames-like index (1M docs, single i64 `population`).

use rustic_core::query::LeafReader;
use rustic_core::segment::Segment;
use rustic_core::{Document, Field, FieldType, Index, IndexSettings};
use std::time::Instant;

const POP: u16 = 1;

/// The segment behind a leaf (the POC index is a single refreshed segment).
fn seg<'a>(leaf: &'a LeafReader<'a>) -> &'a Segment {
    match leaf {
        LeafReader::Segment(s) => s,
        LeafReader::Memtable(..) => panic!("POC expects a segment leaf"),
    }
}

/// Build a synthetic Geonames-like index: `n` docs each with an i64 population.
fn build_index(n: usize) -> Index {
    let mut schema = vec![
        Field::new(0, "_id", FieldType::Keyword),
        Field::new(POP, "population", FieldType::I64),
    ];
    for f in schema.iter_mut() {
        f.indexed = true;
        f.stored = f.id == 0;
    }
    let mut idx = Index::new("sort-poc", schema, IndexSettings::default());
    for i in 0..n {
        let mut d = Document::new();
        d.add_keyword(0, &i.to_string());
        // Geonames-like: mostly small populations, a few large cities.
        d.add_i64(POP, (i % 100_000) as i64);
        idx.add_document(&d);
    }
    idx.refresh().unwrap();
    idx
}

/// Current path: open the column per doc (what `primary_numeric` does today).
fn time_per_doc_open(idx: &Index, iters: usize) -> f64 {
    let leaves = idx.leaves();
    let leaf = &leaves[0];
    let n = idx.num_docs() as u32;
    let start = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for doc in 0..n {
            // Exactly what primary_numeric's closure does per doc.
            if let Some(c) = seg(leaf).fast_fields().get_i64(POP) {
                acc += c.get(doc).unwrap_or(0);
            }
        }
    }
    std::hint::black_box(acc);
    start.elapsed().as_secs_f64()
}

/// Fixed path: open the column once, then read per doc.
fn time_open_once(idx: &Index, iters: usize) -> f64 {
    let leaves = idx.leaves();
    let leaf = &leaves[0];
    let n = idx.num_docs() as u32;
    // Open once (the fix).
    let col = seg(leaf).fast_fields().get_i64(POP).expect("population column");
    let start = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for doc in 0..n {
            acc += col.get(doc).unwrap_or(0);
        }
    }
    std::hint::black_box(acc);
    start.elapsed().as_secs_f64()
}

pub fn run() {
    let n = 1_000_000;
    let iters = 5;
    println!("Building Geonames-like index: {n} docs (i64 population)...");
    let t0 = Instant::now();
    let idx = build_index(n);
    println!("  built in {:.2}s", t0.elapsed().as_secs_f64());

    // Warm up.
    time_per_doc_open(&idx, 1);
    time_open_once(&idx, 1);

    let t1 = time_per_doc_open(&idx, iters);
    let t2 = time_open_once(&idx, iters);

    println!("\nGeonames ascending population sort POC ({iters} iterations, {n} docs):");
    println!("  per-doc column open:  {:.1} ms/iter  ({:.2} ns/doc)", t1 / iters as f64 * 1000.0, t1 / iters as f64 / n as f64 * 1e9);
    println!("  open-once column:     {:.1} ms/iter  ({:.2} ns/doc)", t2 / iters as f64 * 1000.0, t2 / iters as f64 / n as f64 * 1e9);
    println!("  ratio (per-doc/open-once): {:.1}×", t1 / t2);
}
