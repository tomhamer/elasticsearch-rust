//! Indexing-path micro-benchmarks: where does a document's time go?
//! `add_document` (analysis + postings + fast fields + stored copy) vs
//! `encode` (dictionary, postings packing, columns, LZ4 store).

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rustic_core::analysis::Analyzer;
use rustic_core::segment::SegmentWriter;
use rustic_core::types::{Document, Field, FieldType};

fn black_box<T>(v: T) -> T {
    std::hint::black_box(v)
}

fn schema() -> Vec<Field> {
    vec![
        Field::new(0, "title", FieldType::Text),
        Field::new(1, "body", FieldType::Text),
        Field::new(2, "price", FieldType::F64),
        Field::new(3, "ts", FieldType::I64),
        Field::new(4, "tag", FieldType::Keyword),
    ]
}

/// Deterministic Zipf-ish text: 120 words from a 50k vocabulary.
fn docs(n: usize) -> Vec<Document> {
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    };
    (0..n)
        .map(|i| {
            let mut body = String::with_capacity(900);
            for _ in 0..120 {
                let r = next();
                // skewed term ids: most draws land in a small band
                let id = if r % 4 == 0 { (r >> 8) % 50_000 } else { (r >> 8) % 2_000 };
                body.push_str("w");
                body.push_str(&id.to_string());
                body.push(' ');
            }
            let mut d = Document::new();
            d.add_text(0, &format!("title {} item {}", i % 97, i));
            d.add_text(1, &body);
            d.add_f64(2, (i % 1000) as f64 * 1.5);
            d.add_i64(3, i as i64 * 1000);
            d.add_keyword(4, &format!("t{}", i % 16));
            d
        })
        .collect()
}

fn add_documents(c: &mut Criterion) {
    let mut g = c.benchmark_group("indexing");
    let n = 5_000;
    let docs = docs(n);
    let schema = schema();
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("add_document_5k", |b| {
        b.iter_batched(
            || SegmentWriter::new(0),
            |mut w| {
                for d in &docs {
                    w.add_document(d, &schema);
                }
                black_box(w.num_docs())
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("add_then_encode_5k", |b| {
        b.iter_batched(
            || SegmentWriter::new(0),
            |mut w| {
                for d in &docs {
                    w.add_document(d, &schema);
                }
                black_box(w.encode().len())
            },
            BatchSize::LargeInput,
        )
    });
    let analyzer = Analyzer::standard();
    let total_tokens: usize = docs.iter().map(|d| analyzer.analyze(d.get(1).unwrap().as_text().unwrap()).len()).sum();
    g.throughput(Throughput::Elements(total_tokens as u64));
    g.bench_function("analyze_only_body_5k", |b| {
        b.iter(|| {
            let mut n = 0;
            for d in &docs {
                n += analyzer.analyze(d.get(1).unwrap().as_text().unwrap()).len();
            }
            black_box(n)
        })
    });
    g.bench_function("stream_tokens_body_5k", |b| {
        let mut buf = Vec::with_capacity(64);
        b.iter(|| {
            let mut n = 0usize;
            for d in &docs {
                buf.clear();
                buf.extend_from_slice(&1u16.to_le_bytes());
                analyzer.for_each_token(d.get(1).unwrap().as_text().unwrap(), &mut buf, |k, _| n += k.len());
            }
            black_box(n)
        })
    });
    g.bench_function("stream_tokens_into_postings_body_5k", |b| {
        let mut buf = Vec::with_capacity(64);
        b.iter_batched(
            rustic_core::postings::MemPostings::new,
            |mut m| {
                for (i, d) in docs.iter().enumerate() {
                    buf.clear();
                    buf.extend_from_slice(&1u16.to_le_bytes());
                    analyzer.for_each_token(d.get(1).unwrap().as_text().unwrap(), &mut buf, |k, p| {
                        m.push_occurrence(k, i as u32, p)
                    });
                    m.flush();
                }
                black_box(m.len())
            },
            BatchSize::LargeInput,
        )
    });
    g.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(1)).sample_size(10);
    targets = add_documents
);
criterion_main!(benches);
