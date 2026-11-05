//! POC: Metricbeat-style indexing — isolate the in-process `add_documents`
//! hot path. Builds a synthetic metricbeat-like document (timestamp, host,
//! process, several keyword + numeric fields) and times sequential and
//! parallel indexing to find where the 10.7× gap vs ES9 comes from.

use super::*;
use rustic_core::{Document, Field, FieldType, Index, IndexSettings};
use std::time::Instant;

// Field IDs (must match schema order)
const HOST_NAME: u16 = 0;
const HOST_IP: u16 = 1;
const PROC_NAME: u16 = 2;
const PROC_PID: u16 = 3;
const SVC_TYPE: u16 = 4;
const EVT_DATASET: u16 = 5;
const TIMESTAMP: u16 = 6;
const UPTIME: u16 = 7;
const CPU_PCT: u16 = 8;
const LOAD1: u16 = 9;
const MEM_USED: u16 = 10;
const MESSAGE: u16 = 11;

/// Build a metricbeat-like document: ~12 fields, mix of keyword, i64, f64,
/// and one short text field.
fn metricbeat_doc(i: u64) -> Document {
    let mut d = Document::new();
    d.add_keyword(HOST_NAME, &format!("host-{}", i % 100));
    d.add_keyword(HOST_IP, &format!("10.0.{}.{}", (i / 256) % 256, i % 256));
    d.add_keyword(PROC_NAME, &format!("proc-{}", i % 50));
    d.add_keyword(PROC_PID, &format!("{}", 1000 + i % 9000));
    d.add_keyword(SVC_TYPE, "system");
    d.add_keyword(EVT_DATASET, "process");
    d.add_date(TIMESTAMP, 1_700_000_000_000i64 + (i as i64) * 1000);
    d.add_i64(UPTIME, (i % 86400) as i64 * 1000);
    d.add_f64(CPU_PCT, (i % 100) as f64 * 0.01);
    d.add_f64(LOAD1, (i % 1000) as f64 * 0.001);
    d.add_f64(MEM_USED, (i % 10000) as f64 * 1024.0 * 1024.0);
    d.add_text(MESSAGE, &format!("process {} started on host {}", i % 50, i % 100));
    d
}

/// Schema matching the fields above (order defines field IDs).
fn metricbeat_schema() -> Vec<Field> {
    vec![
        Field::new(HOST_NAME, "host.name", FieldType::Keyword),
        Field::new(HOST_IP, "host.ip", FieldType::Keyword),
        Field::new(PROC_NAME, "process.name", FieldType::Keyword),
        Field::new(PROC_PID, "process.pid", FieldType::Keyword),
        Field::new(SVC_TYPE, "service.type", FieldType::Keyword),
        Field::new(EVT_DATASET, "event.dataset", FieldType::Keyword),
        Field::new(TIMESTAMP, "@timestamp", FieldType::Date),
        Field::new(UPTIME, "process.uptime", FieldType::I64),
        Field::new(CPU_PCT, "process.cpu.percent", FieldType::F64),
        Field::new(LOAD1, "system.load.1", FieldType::F64),
        Field::new(MEM_USED, "system.memory.used.bytes", FieldType::F64),
        Field::new(MESSAGE, "message", FieldType::Text),
    ]
}

pub(super) fn run(args: &[String]) {
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let schema = metricbeat_schema();
    let docs: Vec<Document> = (0..n as u64).map(metricbeat_doc).collect();

    // Warmup
    let mut idx = Index::create(
        std::path::Path::new("/tmp/index-poc-warmup"),
        "index-poc",
        schema.clone(),
        IndexSettings::default(),
    ).unwrap();
    idx.add_documents(&docs).unwrap();
    drop(idx);

    let mut best = f64::MAX;
    for _ in 0..iters {
        let mut idx = Index::create(
            std::path::Path::new("/tmp/index-poc"),
            "index-poc",
            schema.clone(),
            IndexSettings::default(),
        ).unwrap();
        let start = Instant::now();
        if threads == 1 {
            idx.add_documents(&docs).unwrap();
        } else {
            idx.add_documents_parallel(&docs, threads).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
        drop(idx);
    }

    let docs_per_s = n as f64 / best;
    println!("Indexing POC: {} docs, {} threads, {} iters", n, threads, iters);
    println!("  best: {:.3}s  ({:.0} docs/s)", best, docs_per_s);
    println!("  ES9 reference: 43,303 docs/s");
    println!("  ratio vs ES9: {:.2}x", 43303.0 / docs_per_s);
}
