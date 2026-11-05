//! Benchmark: Term dictionary lookup performance.
//!
//! Target: <100ns per lookup (Tantivy: ~5ms cold, ~100ns hot)

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustic_core::term_dictionary::{PostingsLocation, TermDictionary};
use rustic_core::types::Term;
use std::collections::BTreeMap;

fn build_dictionary(c: &mut Criterion) {
    let mut group = c.benchmark_group("term_dictionary");

    // Build dictionaries of various sizes
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let mut dict = TermDictionary::new(size as u32);
        for i in 0..size {
            let term = Term::new_text(0, &format!("term_{}", i));
            dict.insert(
                &term,
                PostingsLocation {
                    offset: i as u64,
                    doc_freq: 1,
                    total_term_freq: 1,
                },
            );
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("lookup_{}", size), |b| {
            b.iter(|| {
                // Look up a term that exists
                let term = Term::new_text(0, "term_500");
                black_box(dict.get(&term))
            })
        });

        group.bench_function(format!("contains_{}", size), |b| {
            b.iter(|| {
                let term = Term::new_text(0, "term_500");
                black_box(dict.contains(&term))
            })
        });
    }

    group.finish();
}

fn empty_dictionary_creation(c: &mut Criterion) {
    // This is the key benchmark: Tantivy's TermDictionary::empty()
    // takes ~5ms in debug mode. Ours should be O(1).
    c.bench_function("empty_dictionary_creation", |b| {
        b.iter(|| {
            black_box(TermDictionary::new(0))
        })
    });
}

fn range_iteration(c: &mut Criterion) {
    let mut dict = TermDictionary::new(100_000);
    for i in 0..100_000 {
        let term = Term::new_text(0, &format!("term_{:06}", i));
        dict.insert(
            &term,
            PostingsLocation {
                offset: i as u64,
                doc_freq: 1,
                total_term_freq: 1,
            },
        );
    }

    c.bench_function("range_iter_100_terms", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in dict.range_iter(0, Some(b"term_000500"), Some(b"term_000600")) {
                count += 1;
            }
            black_box(count)
        })
    });
}

fn black_box<T>(val: T) -> T {
    std::hint::black_box(val)
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(2));
    targets = build_dictionary, empty_dictionary_creation, range_iteration
);
criterion_main!(benches);
