//! POC: replicate the full HTTP write path for Metricbeat-style docs to
//! isolate which step dominates the 30.9× gap vs ES9.
//!
//! Steps measured:
//!   1. JSON parse (serde_json::from_str)
//!   2. parse_with_limits (JSON → Document via mapping walk)
//!   3. add_document (Document → segment)
//!   4. translog serialize (bincode::serialize of TranslogOp)
//!
//! Run: `es-path-poc http-write-bench`

use elasticsearch_engine::analysis::AnalysisRegistry;
use elasticsearch_engine::document::parse_with_limits;
use elasticsearch_engine::fieldtypes::limits::Limits;
use elasticsearch_engine::mapping::{FieldLookup, Mappings};
use rustic_core::{Document, Field, FieldType, Index, IndexSettings};
use serde_json::Value;
use std::time::Instant;

const N_DOCS: usize = 10_000;
const N_ITERS: usize = 5;

/// A realistic metricbeat-like JSON source (~1 KB).
fn metricbeat_json(i: usize) -> String {
    format!(
        r#"{{"@timestamp":"2026-01-01T00:00:00.000Z","host":{{"name":"host{i}","ip":"10.0.0.{i}"}},"event":{{"dataset":"metricbeat","module":"system"}},"metricset":"system","cpu":{{"cores":8,"pct":12.5,"user":{{"pct":8.2}},"system":{{"pct":4.3}}}},"memory":{{"total":16777216,"used":8388608,"free":8388608,"pct":50.0}},"process":{{"name":"nginx","pid":1234,"cpu":{{"pct":2.1}},"memory":{{"size":1048576}}}},"system":{{"load":{{"1":0.5,"5":0.3,"15":0.2,"norm":{{"1":0.06,"5":0.04,"15":0.025}}}},"uptime":{{"ms":86400000}}}},"tags":["prod","us-east-1"],"message":"metricbeat document {i} with some extra padding to make the source realistic and a bit larger than the bare minimum for a typical metricbeat event document with additional fields to reach a realistic size"}}"#
    )
}

/// Schema matching the metricbeat fields.
fn metricbeat_schema() -> Vec<Field> {
    vec![
        Field::new(0, "@timestamp", FieldType::Date),
        Field::new(1, "host.name", FieldType::Keyword),
        Field::new(2, "host.ip", FieldType::Keyword),
        Field::new(3, "event.dataset", FieldType::Keyword),
        Field::new(4, "event.module", FieldType::Keyword),
        Field::new(5, "metricset", FieldType::Keyword),
        Field::new(6, "cpu.cores", FieldType::I64),
        Field::new(7, "cpu.pct", FieldType::F64),
        Field::new(8, "cpu.user.pct", FieldType::F64),
        Field::new(9, "cpu.system.pct", FieldType::F64),
        Field::new(10, "memory.total", FieldType::I64),
        Field::new(11, "memory.used", FieldType::I64),
        Field::new(12, "memory.free", FieldType::I64),
        Field::new(13, "memory.pct", FieldType::F64),
        Field::new(14, "process.name", FieldType::Keyword),
        Field::new(15, "process.pid", FieldType::I64),
        Field::new(16, "process.cpu.pct", FieldType::F64),
        Field::new(17, "process.memory.size", FieldType::I64),
        Field::new(18, "system.load.1", FieldType::F64),
        Field::new(19, "system.load.5", FieldType::F64),
        Field::new(20, "system.load.15", FieldType::F64),
        Field::new(21, "system.load.norm.1", FieldType::F64),
        Field::new(22, "system.load.norm.5", FieldType::F64),
        Field::new(23, "system.load.norm.15", FieldType::F64),
        Field::new(24, "system.uptime.ms", FieldType::I64),
        Field::new(25, "tags", FieldType::Keyword),
        Field::new(26, "message", FieldType::Text),
    ]
}

pub fn run() {
    eprintln!("Building {N_DOCS} metricbeat-like JSON documents...");
    let jsons: Vec<String> = (0..N_DOCS).map(metricbeat_json).collect();
    let schema = metricbeat_schema();

    // Build a Mappings + FieldLookup for parse_with_limits.
    // We need to construct these from the schema. The engine's Mappings
    // type is built from a JSON mapping definition, so we build one here.
    let mapping_json = serde_json::json!({
        "properties": {
            "@timestamp": {"type": "date"},
            "host": {"properties": {"name": {"type": "keyword"}, "ip": {"type": "keyword"}}},
            "event": {"properties": {"dataset": {"type": "keyword"}, "module": {"type": "keyword"}}},
            "metricset": {"type": "keyword"},
            "cpu": {"properties": {"cores": {"type": "long"}, "pct": {"type": "float"},
                "user": {"properties": {"pct": {"type": "float"}}},
                "system": {"properties": {"pct": {"type": "float"}}}}},
            "memory": {"properties": {"total": {"type": "long"}, "used": {"type": "long"},
                "free": {"type": "long"}, "pct": {"type": "float"}}},
            "process": {"properties": {"name": {"type": "keyword"}, "pid": {"type": "long"},
                "cpu": {"properties": {"pct": {"type": "float"}}},
                "memory": {"properties": {"size": {"type": "long"}}}}},
            "system": {"properties": {"load": {"properties": {"1": {"type": "float"}, "5": {"type": "float"},
                "15": {"type": "float"}, "norm": {"properties": {"1": {"type": "float"}, "5": {"type": "float"}, "15": {"type": "float"}}}}},
                "uptime": {"properties": {"ms": {"type": "long"}}}}},
            "tags": {"type": "keyword"},
            "message": {"type": "text"}
        }
    });
    let reg = AnalysisRegistry::default();
    let mappings = Mappings::parse_update(&mapping_json, &reg).expect("mapping parses");
    let lookup = FieldLookup::build(&mappings, &reg);
    let limits = Limits::default();

    // Warmup: parse one doc through the full path.
    let v: Value = serde_json::from_str(&jsons[0]).unwrap();
    let (pd, _) = parse_with_limits(&v, "warmup", &mappings, &lookup, &limits).expect("parse_with_limits works");
    let _ = pd.doc;

    // Step 1: JSON parse
    let mut t_parse = 0.0f64;
    let mut values: Vec<Value> = Vec::with_capacity(N_DOCS);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        for j in &jsons {
            values.push(serde_json::from_str(j).unwrap());
        }
        t_parse += start.elapsed().as_secs_f64();
        values.clear();
    }
    let parse_ms = t_parse * 1000.0 / N_ITERS as f64;
    let parse_us = parse_ms * 1000.0 / N_DOCS as f64;

    // Step 2: parse_with_limits (JSON → Document)
    let mut t_pw = 0.0f64;
    let mut docs: Vec<Document> = Vec::with_capacity(N_DOCS);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        for j in &jsons {
            let v: Value = serde_json::from_str(j).unwrap();
            let (pd, _) = parse_with_limits(&v, "id", &mappings, &lookup, &limits).unwrap();
            docs.push(pd.doc);
        }
        t_pw += start.elapsed().as_secs_f64();
        docs.clear();
    }
    let pw_ms = t_pw * 1000.0 / N_ITERS as f64;
    let pw_us = pw_ms * 1000.0 / N_DOCS as f64;

    // Step 3: add_document (Document → segment)
    let mut t_add = 0.0f64;
    for _ in 0..N_ITERS {
        let mut idx = Index::create(
            std::path::Path::new("/tmp/http-write-poc"),
            "http-write-poc",
            schema.clone(),
            IndexSettings::default(),
        ).unwrap();
        let start = Instant::now();
        for j in &jsons {
            let v: Value = serde_json::from_str(j).unwrap();
            let (pd, _) = parse_with_limits(&v, "id", &mappings, &lookup, &limits).unwrap();
            idx.add_document(&pd.doc).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();
        drop(idx);
    }
    let add_ms = t_add * 1000.0 / N_ITERS as f64;
    let add_us = add_ms * 1000.0 / N_DOCS as f64;

    // Step 4: translog serialize (bincode of TranslogOp)
    // We can't easily construct a TranslogOp here without the engine's
    // internal types, so we approximate with bincode::serialize of a
    // struct that has the same shape (id: String, source_json: String,
    // seq_no: u64, version: u64, routing: Option<String>).
    #[derive(serde::Serialize)]
    struct FakeTranslogOp {
        id: String,
        source_json: String,
        seq_no: u64,
        version: u64,
        routing: Option<String>,
    }
    // serde is available transitively; if the derive fails, fall back to
    // measuring bincode::serialize of a tuple struct with the same shape.
    let mut t_translog = 0.0f64;
    let mut translog_bytes = 0usize;
    for _ in 0..N_ITERS {
        let start = Instant::now();
        for (i, j) in jsons.iter().enumerate() {
            let op = FakeTranslogOp {
                id: format!("doc-{i}"),
                source_json: j.clone(),
                seq_no: i as u64,
                version: 1,
                routing: None,
            };
            let body = bincode::serialize(&op).unwrap();
            translog_bytes += body.len();
        }
        t_translog += start.elapsed().as_secs_f64();
    }
    let translog_ms = t_translog * 1000.0 / N_ITERS as f64;
    let translog_us = translog_ms * 1000.0 / N_DOCS as f64;

    let total_ms = parse_ms + pw_ms + add_ms + translog_ms;
    let total_us = total_ms * 1000.0 / N_DOCS as f64;

    eprintln!("\n=== HTTP write path breakdown (per doc) ===");
    eprintln!("Docs: {N_DOCS}, iters: {N_ITERS}");
    eprintln!("1. JSON parse:          {parse_ms:8.1} ms/iter, {parse_us:6.2} us/doc");
    eprintln!("2. parse_with_limits:   {pw_ms:8.1} ms/iter, {pw_us:6.2} us/doc");
    eprintln!("3. add_document:        {add_ms:8.1} ms/iter, {add_us:6.2} us/doc");
    eprintln!("4. translog serialize:  {translog_ms:8.1} ms/iter, {translog_us:6.2} us/doc");
    eprintln!("Total:                  {total_ms:8.1} ms/iter, {total_us:6.2} us/doc");
    eprintln!();
    eprintln!("Rally: Rust 15,882 ms/batch vs ES9 514 ms/batch (30.9x gap)");
    eprintln!("If batch = 1000 docs: Rust per-doc = {:.2} us, ES9 per-doc = {:.2} us",
        15882.0 * 1000.0 / 1000.0, 514.0 * 1000.0 / 1000.0);
    eprintln!("Our POC per-doc: {:.2} us", total_us);
    eprintln!("Gap unexplained by POC: {:.2} us/doc", 15882.0 - total_us);
}
