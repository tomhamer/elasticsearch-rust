//! Benchmark: Range query performance on fast fields.
//!
//! Target: 5x improvement over Tantivy's range query performance.
//! Tantivy: range queries are ~5x slower than text search (issue #2266).

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustic_core::fast_fields::{F64Column, I64Column};

fn build_i64_column(size: usize) -> I64Column {
    let mut col = I64Column::new(size as u32);
    for i in 0..size {
        col.set(i as u32, i as i64);
    }
    col.build_block_index();
    col
}

fn build_f64_column(size: usize) -> F64Column {
    let mut col = F64Column::new(size as u32);
    for i in 0..size {
        col.set(i as u32, i as f64);
    }
    col.build_block_index();
    col
}

fn i64_range_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("i64_range_query");

    for size in [10_000, 100_000, 1_000_000] {
        let col = build_i64_column(size);

        // Selective range: 1% of docs match
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("selective_1pct_{}", size), |b| {
            b.iter(|| {
                black_box(col.range_query((size as i64) * 99 / 100, (size as i64) * 100 / 100))
            })
        });

        // Medium range: 10% of docs match
        group.bench_function(format!("medium_10pct_{}", size), |b| {
            b.iter(|| {
                black_box(col.range_query((size as i64) * 45 / 100, (size as i64) * 55 / 100))
            })
        });

        // Wide range: 50% of docs match
        group.bench_function(format!("wide_50pct_{}", size), |b| {
            b.iter(|| {
                black_box(col.range_query(0, (size as i64) / 2))
            })
        });

        // Count only (no doc_id collection)
        group.bench_function(format!("count_selective_{}", size), |b| {
            b.iter(|| {
                black_box(col.count_in_range((size as i64) * 99 / 100, (size as i64) * 100 / 100))
            })
        });
    }

    group.finish();
}

fn f64_range_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_range_query");

    for size in [10_000, 100_000, 1_000_000] {
        let col = build_f64_column(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("selective_1pct_{}", size), |b| {
            b.iter(|| {
                black_box(col.range_query((size as f64) * 0.99, (size as f64) * 1.0))
            })
        });

        group.bench_function(format!("medium_10pct_{}", size), |b| {
            b.iter(|| {
                black_box(col.range_query((size as f64) * 0.45, (size as f64) * 0.55))
            })
        });
    }

    group.finish();
}

fn black_box<T>(val: T) -> T {
    std::hint::black_box(val)
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(2));
    targets = i64_range_queries, f64_range_queries
);
criterion_main!(benches);
