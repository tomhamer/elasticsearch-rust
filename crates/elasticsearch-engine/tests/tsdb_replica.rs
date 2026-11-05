//! Hidden TSID doc values must survive source-based replication and WAL replay.
use elasticsearch_engine::hooks::changes::Durability;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

fn buckets(node: &Arc<Node>) -> Value {
    let request = search::parse_request(&json!({"size":0,"aggs":{"series":{"terms":{"field":"_tsid"}}}}), &HashMap::new()).unwrap();
    let targets = node.resolve("metrics", Default::default()).unwrap();
    let options = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let result = search::execute(&targets, &request, &options).unwrap();
    serde_json::from_str::<Value>(&result.json).unwrap()["aggregations"]["series"]["buckets"].clone()
}

#[test]
fn tsid_doc_values_survive_replication_and_restart() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = Node::open(primary_dir.path()).unwrap();
    let replica = Node::open(replica_dir.path()).unwrap();
    let definition = json!({"settings":{"index.mode":"time_series","index.routing_path":["host"],
        "index.time_series.start_time":"2024-01-01T00:00:00Z","index.time_series.end_time":"2025-01-01T00:00:00Z",
        "number_of_shards":1,"number_of_replicas":0},
        "mappings":{"properties":{"host":{"type":"keyword","time_series_dimension":true},"@timestamp":{"type":"date"}}}});
    let source = primary.create_index("metrics", &definition).unwrap();
    let target = replica.create_index("metrics", &definition).unwrap();
    source.write(vec![WriteReq::Index { id: None, routing: None,
        source: json!({"host":"a","extra":"dynamic","@timestamp":"2024-06-01T00:00:00Z"}).to_string(), cond: Default::default() }], true)
        .into_iter().next().unwrap().unwrap();
    source.refresh().unwrap();
    let shard = source.shard(0).unwrap();
    let operations = shard.shard.changes_since(0).unwrap().map(|op| (*op).clone()).collect();
    target.shard(0).unwrap().apply_replica_ops(shard.shard.primary_term(), operations, Durability::Request).unwrap();
    target.refresh().unwrap();
    let expected = buckets(&primary);
    assert_eq!(expected.as_array().unwrap().len(), 1);
    assert_eq!(buckets(&replica), expected);
    let recovered_dir = tempfile::tempdir().unwrap();
    let recovered = Node::open(recovered_dir.path()).unwrap();
    let files = source.snapshot_files(true).unwrap();
    let staging = recovered.new_staging_dir().unwrap();
    files.write_to_dir(&staging, false).unwrap();
    recovered.adopt_index(&staging).unwrap();
    assert_eq!(buckets(&recovered), expected, "file recovery preserves hidden TSID fields with dynamic mappings");
    drop(target);
    drop(replica);
    let restarted = Node::open(replica_dir.path()).unwrap();
    assert_eq!(buckets(&restarted), expected);
}

#[test]
fn sparse_cluster_metadata_restores_hidden_tsid_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let metadata = json!({"settings":{"index.mode":"time_series","index.routing_path":["host"],
        "index.time_series.start_time":"2024-01-01T00:00:00Z","index.time_series.end_time":"2025-01-01T00:00:00Z",
        "number_of_shards":1},"mappings":{"properties":{"host":{"type":"keyword","time_series_dimension":true},
        "@timestamp":{"type":"date"}}}});
    let service = node.create_index_sparse("metrics", "AAAAAAAAAAAAAAAAAAAAAA", &metadata).unwrap();
    assert!(service.lookup().get("_tsid").and_then(|field| field.core_id).is_some());
    assert!(service.meta().mappings.to_json()["properties"].get("_tsid").is_none(), "hidden mapping is not exposed by REST");
    service.apply_metadata(&metadata["mappings"], &service.meta().settings, 1).unwrap();
    assert!(service.lookup().get("_tsid").and_then(|field| field.core_id).is_some());
}
