//! Single-thread indexing throughput on the HTTP bench corpus (ignored;
//! run with `--ignored --nocapture`, `RUSTIC_CORPUS` = corpus.jsonl path).

use rustic_core::segment::SegmentWriter;
use rustic_core::types::{Document, Field, FieldType};
use std::io::BufRead;

fn schema() -> Vec<Field> {
    let mut f = vec![
        Field::new(0, "id", FieldType::I64),
        Field::new(1, "title", FieldType::Text),
        Field::new(2, "body", FieldType::Text),
        Field::new(3, "price", FieldType::F64),
        Field::new(4, "ts", FieldType::Date),
        Field::new(5, "tag", FieldType::Keyword),
    ];
    for x in f.iter_mut() {
        x.fast = !matches!(x.field_type, FieldType::Text);
        x.indexed = true;
    }
    f
}

#[test]
#[ignore]
fn ingest_micro() {
    let path = std::env::var("RUSTIC_CORPUS").unwrap_or_else(|_| "bench/corpus/corpus.jsonl".into());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let f = std::fs::File::open(&path).expect("corpus");
    let mut docs = Vec::with_capacity(n);
    let mut tokens = 0usize;
    for line in std::io::BufReader::new(f).lines().take(n) {
        let v: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let t = v["title"].as_str().unwrap();
        let b = v["body"].as_str().unwrap();
        tokens += t.split(' ').count() + b.split(' ').count();
        let mut d = Document::new();
        d.fields.push((0, rustic_core::types::FieldValue::I64(v["id"].as_i64().unwrap())));
        d.fields.push((1, rustic_core::types::FieldValue::Text(t.to_string())));
        d.fields.push((2, rustic_core::types::FieldValue::Text(b.to_string())));
        d.fields.push((3, rustic_core::types::FieldValue::F64(v["price"].as_f64().unwrap())));
        d.fields.push((4, rustic_core::types::FieldValue::Date(v["ts"].as_i64().unwrap())));
        d.fields.push((5, rustic_core::types::FieldValue::Keyword(v["tag"].as_str().unwrap().to_string())));
        docs.push(d);
    }
    let schema = schema();
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    for _ in 0..rounds {
        let mut w = SegmentWriter::new(0);
        let t0 = std::time::Instant::now();
        for d in &docs {
            w.add_document(d, &schema);
        }
        let add = t0.elapsed();
        let t1 = std::time::Instant::now();
        let enc = w.encode();
        let encode = t1.elapsed();
        println!(
            "add {} docs: {:.3}s = {:.0} docs/s, {:.1} ns/token ({} tokens); encode {:.3}s ({} bytes)",
            docs.len(),
            add.as_secs_f64(),
            docs.len() as f64 / add.as_secs_f64(),
            add.as_nanos() as f64 / tokens as f64,
            tokens,
            encode.as_secs_f64(),
            enc.len()
        );
    }
}
