//! Finding 559: a new logsdb index defaults `index.mapping.ignore_malformed`
//! to true (ES `FieldMapper.IGNORE_MALFORMED_SETTING` for logsdb indices
//! created at or after `ENABLE_IGNORE_MALFORMED_LOGSDB`): a log event with a
//! malformed numeric value is kept with that field ignored. An explicit
//! `false`, an older creation version and a standard index reject it.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use serde_json::json;

fn write(node: &Node, index: &str) -> bool {
    let svc = node.get_index(index).unwrap();
    let doc = json!({"@timestamp": "2025-01-01T00:00:00Z", "message": "ok", "bytes": "not-a-number"}).to_string();
    let ok = svc.write(vec![WriteReq::Index { id: None, source: doc, routing: None, cond: Default::default() }], true).into_iter().all(|r| r.is_ok());
    svc.refresh().unwrap();
    ok
}

#[test]
fn logsdb_keeps_events_with_malformed_values_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mapping = json!({"properties": {"@timestamp": {"type": "date"}, "message": {"type": "text"}, "bytes": {"type": "long"}}});
    node.create_index("logs", &json!({"settings": {"index.mode": "logsdb", "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    assert!(write(&node, "logs"), "logsdb keeps the event");
    let svc = node.get_index("logs").unwrap();
    let ignored = svc.lookup().fields.get("bytes").map(|f| f.ignore_malformed);
    assert_eq!(ignored, Some(true), "the field inherits the logsdb default");
    node.create_index("logs-strict", &json!({"settings": {"index.mode": "logsdb", "index.mapping.ignore_malformed": false, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    assert!(!write(&node, "logs-strict"), "explicit false rejects");
    node.create_index("std", &json!({"settings": {"number_of_replicas": 0}, "mappings": mapping})).unwrap();
    assert!(!write(&node, "std"), "standard indices keep false");
    let old = elasticsearch_engine::version::IndexVersion(8_513_0_00).to_string();
    node.create_index("logs-old", &json!({"settings": {"index.mode": "logsdb", "index.version.created": old, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    assert!(!write(&node, "logs-old"), "older logsdb indices keep false");
}
