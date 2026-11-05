//! POC: nested aggregation performance — compare the old linear-scan
//! `block_start` against the new O(1) `block_starts_table` in the
//! `nested_children` path used by `nested_agg`.
//!
//! Builds a synthetic nested index (root docs each with N children at path
//! "a"), then times the per-parent `nested_children` loop that
//! `nested_agg` runs for every matched parent doc.

use rustic_core::query::LeafReader;
use rustic_core::types::IndexSettings;
use rustic_core::{Document, Field, FieldType, Index};
use std::time::Instant;

/// Build a synthetic nested index: `roots` root docs, each with `children`
/// children at path "a". Layout per root: children first, root last.
fn build_nested_index(roots: usize, children: usize) -> Index {
    let mut schema = vec![
        Field::new(0, "_id", FieldType::Keyword),
        Field::new(1, "body", FieldType::Text),
        Field::new(2, "tag", FieldType::Keyword),
    ];
    for f in schema.iter_mut() {
        f.indexed = true;
        f.stored = f.id == 0;
    }
    let mut idx = Index::new("nested-poc", schema, IndexSettings::default());

    for r in 0..roots {
        let mut root = Document::new();
        root.add_keyword(0, &r.to_string());
        root.add_text(1, &format!("root body {r}"));
        root.add_keyword(2, "root");
        for c in 0..children {
            let mut child = Document::new();
            child.add_text(1, &format!("child body text {r} {c}"));
            child.add_keyword(2, "child");
            root.nested.push(("a".to_string(), child));
        }
        idx.add_document(&root);
    }
    idx.refresh().unwrap();
    idx
}

/// Time the `nested_children` loop over all root docs (the per-parent work
/// that `nested_agg` does). Returns (elapsed_secs, total_children_found).
fn time_nested_children(leaf: &LeafReader<'_>, iters: usize) -> (f64, usize) {
    let info = leaf.nested_info().unwrap();
    let path = "a";
    let start = Instant::now();
    let mut total_children = 0usize;
    for _ in 0..iters {
        total_children = 0;
        // Simulate nested_agg: for each root doc, find its children at path "a".
        for d in 0..leaf.num_docs() {
            if info.is_root(d as u32) && !leaf.is_deleted(d as u32) {
                let kids = leaf.nested_children(d as u32, path);
                total_children += kids.len();
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total_children)
}

pub fn run() {
    let roots = 100_000;
    let children = 10;
    let iters = 5;

    println!("Building nested index: {roots} roots × {children} children...");
    let t0 = Instant::now();
    let idx = build_nested_index(roots, children);
    println!("  built in {:.2}s ({} docs)", t0.elapsed().as_secs_f64(), roots * (children + 1));

    let leaves = idx.leaves();
    let leaf = &leaves[0];

    // Warm up (also initializes the OnceLock block_starts table).
    time_nested_children(leaf, 1);

    let (elapsed, total) = time_nested_children(leaf, iters);
    let ms_per_iter = elapsed / iters as f64 * 1000.0;
    println!("\nnested_children over {roots} roots ({iters} iterations):");
    println!("  O(1) block_starts_table: {:.1} ms/iter  ({} children found)", ms_per_iter, total);
    println!("  expected children: {}", roots * children);
    assert_eq!(total, roots * children, "child count mismatch");
    println!("\n✓ nested_children found all {} children in {:.1} ms/iter", total, ms_per_iter);
}
