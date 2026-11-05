//! Stored fields (`store: true`) and synthetic `_source` end to end: they
//! must survive refresh, flush, merge and a restart, and work when the
//! regular `_source` is disabled or canonicalised.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

fn put(node: &Arc<Node>, index: &str, id: &str, src: &Value) {
    let svc = node.get_index(index).unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: Default::default() }], false);
    r.into_iter().next().unwrap().unwrap();
}

fn search(node: &Arc<Node>, index: &str, body: Value) -> Value {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    // Searches see each shard as of its last refresh (ES NRT): refresh
    // first, like a REST test's `?refresh`.
    for t in &targets {
        t.svc.refresh().unwrap();
    }
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {e:?}"));
    serde_json::from_str(&out.json).unwrap()
}

fn source_of(node: &Arc<Node>, index: &str, id: &str) -> Value {
    let g = node.get_index(index).unwrap().get(id).unwrap().expect("doc exists");
    serde_json::from_str(g.source.as_deref().expect("source stored")).unwrap()
}

#[test]
fn synthetic_source_is_canonical_and_survives_flush_merge_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let want = json!({"k": ["a", "b"], "n": [1, 3, 3], "o": {"x": "q", "y": [1, 2]}, "when": "2020-01-01T00:00:00.000Z"});
    {
        let node = Node::open(tmp.path()).unwrap();
        node.create_index(
            "syn",
            &json!({
                "settings": {"index": {"mapping": {"source": {"mode": "synthetic"}}}},
                "mappings": {"properties": {
                    "k": {"type": "keyword"}, "n": {"type": "long"}, "when": {"type": "date"},
                    "o": {"properties": {"x": {"type": "keyword"}, "y": {"type": "integer"}}}
                }}
            }),
        )
        .unwrap();
        put(&node, "syn", "1", &json!({"n": [3, 1, 3], "o": [{"y": 2, "x": "q"}, {"y": 1}], "when": "2020-01-01", "k": ["b", "a", "b"]}));
        assert_eq!(source_of(&node, "syn", "1"), want);
        let svc = node.get_index("syn").unwrap();
        svc.shard(0).unwrap().shard.flush().unwrap();
        assert_eq!(source_of(&node, "syn", "1"), want);
        // A second segment, then a merge of both.
        put(&node, "syn", "2", &json!({"k": "z"}));
        svc.shard(0).unwrap().shard.flush().unwrap();
        svc.shard(0).unwrap().shard.force_merge(1, false).unwrap();
        assert_eq!(source_of(&node, "syn", "1"), want);
        assert_eq!(source_of(&node, "syn", "2"), json!({"k": "z"}));
        // Updating a synthetic document re-canonicalises the merged result.
        svc.update("2", &json!({"doc": {"k": ["y", "x"]}}), None, 0, Default::default()).unwrap();
        assert_eq!(source_of(&node, "syn", "2"), json!({"k": ["x", "y"]}));
        // `_source` hits come back canonical from search too.
        let r = search(&node, "syn", json!({"query": {"ids": {"values": ["1"]}}}));
        assert_eq!(r["hits"]["hits"][0]["_source"], want);
    }
    // Restart: the translog / segments are replayed and the source is unchanged.
    let node = Node::open(tmp.path()).unwrap();
    assert_eq!(source_of(&node, "syn", "1"), want);
}

#[test]
fn stored_fields_work_without_source() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "nosrc",
        &json!({"mappings": {"_source": {"enabled": false}, "properties": {
            "title": {"type": "text", "store": true},
            "tag": {"type": "keyword", "store": true},
            "count": {"type": "long", "store": true},
            "when": {"type": "date", "store": true},
            "plain": {"type": "keyword"},
            "kw": {"type": "keyword", "ignore_above": 4, "store": true}
        }}}),
    )
    .unwrap();
    put(&node, "nosrc", "1", &json!({"title": "the quick brown fox", "tag": ["b", "a", "b"], "count": 7, "when": "2020-01-01", "plain": "x", "kw": "toolong"}));
    let svc = node.get_index("nosrc").unwrap();
    svc.shard(0).unwrap().shard.flush().unwrap();
    put(&node, "nosrc", "2", &json!({"title": "another", "tag": "t"}));
    let r = search(&node, "nosrc", json!({"query": {"ids": {"values": ["1"]}}, "_source": false, "stored_fields": ["title", "tag", "count", "when", "plain", "kw"]}));
    let h = &r["hits"]["hits"][0];
    assert!(h.get("_source").is_none());
    // Stored in document order, not sorted or de-duplicated; unmapped-as-stored and ignored values absent.
    assert_eq!(h["fields"], json!({"title": ["the quick brown fox"], "tag": ["b", "a", "b"], "count": [7], "when": ["2020-01-01T00:00:00.000Z"]}));
    // Memtable doc as well.
    let r = search(&node, "nosrc", json!({"query": {"ids": {"values": ["2"]}}, "stored_fields": ["title", "tag"]}));
    assert_eq!(r["hits"]["hits"][0]["fields"], json!({"title": ["another"], "tag": ["t"]}));
    // Highlighting reads the stored field when there is no `_source`.
    let r = search(&node, "nosrc", json!({"query": {"match": {"title": "quick"}}, "highlight": {"fields": {"title": {}}}}));
    assert_eq!(r["hits"]["hits"][0]["highlight"]["title"][0], json!("the <em>quick</em> brown fox"));
    // GET-side accessors see the block too, and `_ignored` is available.
    let g = svc.get("1").unwrap().unwrap();
    assert_eq!(elasticsearch_engine::stored::values_of(&g.doc, "count"), vec![json!(7)]);
    assert_eq!(elasticsearch_engine::stored::values_of(&g.doc, "_ignored"), vec![json!("kw")]);
    // After a restart and a merge.
    drop(r);
    svc.shard(0).unwrap().shard.force_merge(1, false).unwrap();
    // The shard lock is held until every handle is gone (one open copy per shard).
    drop(svc);
    drop(node);
    let node = Node::open(tmp.path()).unwrap();
    let r = search(&node, "nosrc", json!({"query": {"ids": {"values": ["1"]}}, "_source": false, "stored_fields": ["tag"]}));
    assert_eq!(r["hits"]["hits"][0]["fields"], json!({"tag": ["b", "a", "b"]}));
}
