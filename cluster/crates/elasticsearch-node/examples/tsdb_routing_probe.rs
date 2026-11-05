//! Isolated routing CPU diagnostic, not end-to-end indexing throughput.
use elasticsearch_cluster::state::{IndexMetadata, Metadata};
use elasticsearch_node::shard_engines::{tsdb_source_shard, TsdbRoutingCache};
use serde_json::{json, Map, Value};
use std::{hint::black_box, time::Instant};
fn main() {
    let mut properties = Map::new();
    let mut source = Map::new();
    properties.insert("@timestamp".into(), json!({"type":"date"}));
    source.insert("@timestamp".into(),json!("2026-01-01T00:00:00Z"));
    for n in 0..256 {
        let name = format!("field{n}");
        properties.insert(name.clone(), if n<24 {json!({"type":"keyword","time_series_dimension":true})} else {json!({"type":"long"})});
        source.insert(name, if n<24 {json!(format!("value{n}"))} else {json!(n)});
    }
    let im: IndexMetadata = serde_json::from_value(json!({
        "name":"ts","uuid":"probe","version":1,"mapping_version":1,"settings_version":1,"aliases_version":1,
        "state":"open","settings":{"index.mode":"time_series","index.routing_path":["field0"]},
        "mappings":{"properties":properties},"aliases":{},"number_of_shards":4,"number_of_replicas":1,
        "routing_num_shards":4,"primary_terms":[1,1,1,1],"in_sync_allocation_ids":{},"creation_date":0
    })).unwrap();
    let mut md = Metadata::default(); md.indices.insert("ts".into(),im);
    let doc = Value::Object(source);
    let mut cache = TsdbRoutingCache::default();
    let iterations = 1000;
    let start = Instant::now();
    let mut baseline = 0u64;
    for _ in 0..iterations { baseline += black_box(tsdb_source_shard(&md,"ts",black_box(&doc))) as u64; }
    let baseline_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let mut optimized = 0u64;
    for _ in 0..iterations { optimized += black_box(cache.source_shard(&md,"ts",black_box(&doc))) as u64; }
    let cached_ns = start.elapsed().as_nanos();
    assert_eq!(baseline,optimized);
    println!("{}",json!({"iterations":iterations,"mapping_fields":257,"dimensions":24,"baseline_ns":baseline_ns,"cached_ns":cached_ns,"routing_cpu_ratio":baseline_ns as f64/cached_ns as f64,"equal_results":true}));
}
