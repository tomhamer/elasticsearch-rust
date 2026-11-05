//! Micro-benchmarks for the on-disk codec hot paths: postings decode and
//! advance, term dictionary lookup, bit unpacking.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustic_core::codec::bitpack::{pack, unpack};
use rustic_core::codec::directory::FileSlice;
use rustic_core::codec::postings::{write_postings_full, PostingsReader};
use rustic_core::codec::terms::{TermDictReader, TermDictWriter};
use rustic_core::term_dictionary::PostingsLocation;
use rustic_core::types::Term;

fn black_box<T>(v: T) -> T {
    std::hint::black_box(v)
}

fn postings_bytes(n: u32, step: u32, with_positions: bool) -> Vec<u8> {
    let docs: Vec<u32> = (0..n).map(|i| i * step).collect();
    let tfs: Vec<u32> = (0..n).map(|i| (i % 3) + 1).collect();
    let positions: Vec<u32> = tfs
        .iter()
        .flat_map(|&tf| (0..tf).map(|k| k * 5 + 1))
        .collect();
    let mut buf = Vec::new();
    write_postings_full(&docs, &tfs, with_positions.then_some(positions.as_slice()), &mut buf);
    buf
}

fn postings_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("postings_decode");
    for &n in &[1_000u32, 100_000, 1_000_000] {
        let bytes = postings_bytes(n, 3, true);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("iterate_{n}"), |b| {
            b.iter(|| {
                let mut r = PostingsReader::open(FileSlice::from_vec(bytes.clone())).unwrap();
                let mut sum = 0u64;
                while let Some(d) = r.next() {
                    sum += d as u64 + r.term_freq().unwrap_or(0) as u64;
                }
                black_box(sum)
            })
        });
        g.bench_function(format!("iterate_with_positions_{n}"), |b| {
            b.iter(|| {
                let mut r = PostingsReader::open(FileSlice::from_vec(bytes.clone())).unwrap();
                let mut sum = 0u64;
                while let Some(_) = r.next() {
                    sum += r.positions().len() as u64;
                }
                black_box(sum)
            })
        });
        g.bench_function(format!("advance_every_1000th_{n}"), |b| {
            b.iter(|| {
                let mut r = PostingsReader::open(FileSlice::from_vec(bytes.clone())).unwrap();
                let mut hits = 0;
                let mut t = 0;
                while let Some(_) = r.advance(t) {
                    hits += 1;
                    t += 3000;
                }
                black_box(hits)
            })
        });
    }
    g.finish();
}

fn term_dict_lookup(c: &mut Criterion) {
    let mut g = c.benchmark_group("term_dict_disk");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut keys: Vec<Vec<u8>> = (0..n).map(|i| Term::new_text(0, &format!("term_{i:07}")).key()).collect();
        keys.sort();
        let mut w = TermDictWriter::new();
        for (i, k) in keys.iter().enumerate() {
            w.add(k, &PostingsLocation { offset: i as u64, doc_freq: 1, total_term_freq: 1 });
        }
        let bytes = w.finish(n as u32);
        let reader = TermDictReader::open(FileSlice::from_vec(bytes.clone())).unwrap();
        let probe = Term::new_text(0, &format!("term_{:07}", n / 2));
        let miss = Term::new_text(0, "zzzz");
        g.bench_function(format!("get_hit_{n}"), |b| b.iter(|| black_box(reader.get(&probe))));
        g.bench_function(format!("get_miss_{n}"), |b| b.iter(|| black_box(reader.get(&miss))));
        g.bench_function(format!("open_{n}"), |b| {
            b.iter(|| black_box(TermDictReader::open(FileSlice::from_vec(bytes.clone())).unwrap().num_terms()))
        });
    }
    g.finish();
}

fn bitpack_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("bitpack");
    for &bits in &[5u32, 13, 17, 24] {
        let max = (1u32 << bits) - 1;
        let vals: Vec<u32> = (0..128u32).map(|i| i.wrapping_mul(2654435761) & max).collect();
        let mut buf = Vec::new();
        pack(&vals, &mut buf);
        g.throughput(Throughput::Elements(128));
        g.bench_function(format!("unpack_128_x{bits}bit"), |b| {
            let mut out = Vec::with_capacity(128);
            b.iter(|| {
                out.clear();
                black_box(unpack(&buf, 128, &mut out).unwrap())
            })
        });
    }
    g.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(1));
    targets = postings_decode, term_dict_lookup, bitpack_bench
);
criterion_main!(benches);
