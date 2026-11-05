//! POC: nested date histogram — isolate the `nested_children` / `block_start`
//! cost that dominates the nested aggregation path.
//!
//! The `nested_agg` code path calls `leaf.nested_children(parent, path)` for
//! each parent doc. `nested_children` → `NestedInfo::children` →
//! `block_start(parent)` which does a linear backward scan:
//!
//! ```rust
//! pub fn block_start(&self, doc: DocId) -> DocId {
//!     let k = self.depth(doc);
//!     let mut s = doc;
//!     while s > 0 && self.depth(s - 1) > k { s -= 1; }
//!     s
//! }
//! ```
//!
//! For a nested index with `roots` parents each having `children` children,
//! the total work is O(roots × children) = O(n) per parent × roots parents
//! = O(n²) overall. This POC times the `nested_children` loop to confirm.

use rustic_core::query::LeafReader;
use rustic_core::{Document, Field, FieldType, Index, IndexSettings};
use std::time::Instant;

/// Build a synthetic nested index: `roots` root docs, each with `children`
/// children at path "a". Layout per root: children first, root last.
fn build_nested_index(roots: usize, children: usize) -> Index {
    let mut schema = vec![
        Field::new(0, "_id", FieldType::Keyword),
        Field::new(1, "ts", FieldType::I64),
    ];
    for f in schema.iter_mut() {
        f.indexed = true;
        f.stored = f.id == 0;
    }
    let mut idx = Index::new("nested-datehist-poc", schema, IndexSettings::default());

    for r in 0..roots {
        let mut root = Document::new();
        root.add_keyword(0, &r.to_string());
        root.add_i64(1, 1_700_000_000_000 + (r as i64) * 86_400_000);
        for c in 0..children {
            let mut child = Document::new();
            child.add_i64(1, 1_700_000_000_000 + ((r * children + c) as i64) * 86_400_000);
            root.nested.push(("a".to_string(), child));
        }
        idx.add_document(&root);
    }
    idx.refresh().unwrap();
    idx
}

/// Time the `nested_children` loop — the exact path `nested_agg` takes.
fn time_nested_children(idx: &Index, iters: usize) -> (f64, usize) {
    let leaves = idx.leaves();
    let leaf = &leaves[0];
    let n = idx.num_docs() as u32;

    let start = Instant::now();
    let mut total_children = 0usize;
    for _ in 0..iters {
        for parent in 0..n {
            let children = leaf.nested_children_iter(parent, "a");
            total_children += children.count();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total_children)
}

/// Time the `i64_values` loop over all docs (the date_histogram per-doc path).
fn time_i64_values(idx: &Index, iters: usize) -> (f64, usize) {
    let leaves = idx.leaves();
    let leaf = &leaves[0];
    let field = 1u16;
    let n = idx.num_docs() as u32;

    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..iters {
        for doc in 0..n {
            total += leaf.i64_values(field, doc).len();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total)
}

pub fn run() {
    let roots = 100_000;
    let children = 10;
    let iters = 5;
    let total_docs = roots * (children + 1);

    println!("Building nested index: {roots} roots × {children} children = {total_docs} docs...");
    let t0 = Instant::now();
    let idx = build_nested_index(roots, children);
    println!("  built in {:.2}s", t0.elapsed().as_secs_f64());

    // Warm up
    time_nested_children(&idx, 1);
    time_i64_values(&idx, 1);

    let (t1, total_children) = time_nested_children(&idx, iters);
    let (t2, total_values) = time_i64_values(&idx, iters);

    println!("\nNested date histogram POC ({iters} iterations):");
    println!("  nested_children loop:  {:.1} ms/iter  ({} children total)", t1 / iters as f64 * 1000.0, total_children / iters);
    println!("  i64_values loop:       {:.1} ms/iter  ({} values total)", t2 / iters as f64 * 1000.0, total_values / iters);
    println!("  ratio (nested/i64):    {:.1}×", t1 / t2);
    println!("\n✓ nested_children finds {} children per iteration (expected {})", total_children / iters, roots * children);
}
