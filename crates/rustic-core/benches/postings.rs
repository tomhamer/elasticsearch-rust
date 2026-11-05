//! Benchmark: Postings list operations.
//!
//! Targets:
//! - 2-level skip data: 33% speedup over Lucene's 10-level (Lucene 10.2)
//! - Count optimization: 60% speedup via popcnt (Lucene 10.2)
//! - SIMD intersection: faster AND of two postings lists

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustic_core::postings::{intersect, union, BitSet, PostingsList};

fn build_postings(size: usize, step: usize) -> PostingsList {
    let mut pl = PostingsList::new();
    for i in (0..size).step_by(step) {
        pl.add_doc(i as u32, 1, &[]);
    }
    pl.build_skip_tables();
    pl
}

fn postings_advance(c: &mut Criterion) {
    let mut group = c.benchmark_group("postings_advance");

    for size in [10_000, 100_000, 1_000_000] {
        let pl = build_postings(size, 1);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("advance_mid_{}", size), |bencher| {
            bencher.iter(|| black_box(pl.advance(size as u32 / 2)))
        });

        group.bench_function(format!("advance_end_{}", size), |bencher| {
            bencher.iter(|| black_box(pl.advance(size as u32 - 1)))
        });
    }

    group.finish();
}

fn postings_intersect(c: &mut Criterion) {
    let mut group = c.benchmark_group("postings_intersect");

    for size in [10_000, 100_000, 1_000_000] {
        // Even doc IDs vs. multiples of 3: ~1/6 of the space overlaps.
        let list_a = build_postings(size, 2);
        let list_b = build_postings(size, 3);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("intersect_{}", size), |bencher| {
            bencher.iter(|| black_box(intersect(&list_a, &list_b)))
        });
    }

    group.finish();
}

fn postings_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("postings_union");

    for size in [10_000, 100_000, 1_000_000] {
        // Even vs. odd doc IDs: no overlap, perfectly interleaved.
        let mut list_a = PostingsList::new();
        let mut list_b = PostingsList::new();
        for i in (0..size).step_by(2) {
            list_a.add_doc(i as u32, 1, &[]);
        }
        for i in (1..size).step_by(2) {
            list_b.add_doc(i as u32, 1, &[]);
        }
        list_a.build_skip_tables();
        list_b.build_skip_tables();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("union_no_overlap_{}", size), |bencher| {
            bencher.iter(|| black_box(union(&list_a, &list_b)))
        });
    }

    group.finish();
}

fn bitset_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitset_count");

    for size in [10_000, 100_000, 1_000_000] {
        let mut bs = BitSet::new(size as u32);
        for i in (0..size).step_by(2) {
            bs.set(i as u32);
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("count_50pct_{}", size), |bencher| {
            bencher.iter(|| black_box(bs.count()))
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
    targets = postings_advance, postings_intersect, postings_union, bitset_count
);
criterion_main!(benches);
