//! Sorting / aggregating on `text` fields with `fielddata: true`: values
//! come from the uninverted index (a doc's analyzed terms), as in ES.

use crate::index::{IndexService, WriteReq};
use crate::shard::WriteCond;
use crate::settings;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

fn idx(id: &str, body: Value) -> WriteReq {
    WriteReq::Index { id: Some(id.into()), source: body.to_string(), routing: None, cond: WriteCond::default() }
}

/// Docs and their analyzed terms (standard analyzer):
/// a {bar, quick}, b {apple, zebra}, c {mango}, d (no `foo`), e {delta, yak}.
/// Written so that a+b and c+d are sealed segments and e is in the memtable.
fn index(dir: &std::path::Path, shards: u32) -> Arc<IndexService> {
    let s = Arc::new(
        IndexService::create(
            dir,
            "t",
            settings::normalize_index_settings(&json!({"number_of_shards": shards, "number_of_replicas": 0})),
            &json!({"properties": {"foo": {"type": "text", "fielddata": true}, "n": {"type": "long"}}}),
            BTreeMap::new(),
        )
        .unwrap(),
    );
    let w = |reqs: Vec<WriteReq>| assert!(s.write(reqs, true).iter().all(|r| r.is_ok()));
    w(vec![idx("a", json!({"foo": "Bar Quick", "n": 1})), idx("b", json!({"foo": "zebra apple", "n": 2}))]);
    s.flush().unwrap();
    w(vec![idx("c", json!({"foo": "mango", "n": 3})), idx("d", json!({"n": 4}))]);
    s.flush().unwrap();
    w(vec![idx("e", json!({"foo": ["delta", "Yak"], "n": 5}))]);
    s.refresh().unwrap();
    s
}

fn search(s: &Arc<IndexService>, body: Value) -> Value {
    let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(s.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap()
}

/// `(id, sort)` of every hit.
fn hits(v: &Value) -> Vec<(String, Value)> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["sort"].clone())).collect()
}

fn expect(v: &Value, want: &[(&str, Value)]) {
    let want: Vec<(String, Value)> = want.iter().map(|(i, s)| (i.to_string(), s.clone())).collect();
    assert_eq!(hits(v), want);
}

#[test]
fn text_fielddata_sort_asc_uses_smallest_term() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    assert!(s.num_segments() >= 2, "docs span several segments");
    let v = search(&s, json!({"sort": "foo"}));
    expect(&v, &[("b", json!(["apple"])), ("a", json!(["bar"])), ("e", json!(["delta"])), ("c", json!(["mango"])), ("d", json!([null]))]);
}

#[test]
fn text_fielddata_sort_desc_uses_largest_term() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let v = search(&s, json!({"sort": [{"foo": "desc"}]}));
    expect(&v, &[("b", json!(["zebra"])), ("e", json!(["yak"])), ("a", json!(["quick"])), ("c", json!(["mango"])), ("d", json!([null]))]);
}

#[test]
fn text_fielddata_sort_missing_first() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let v = search(&s, json!({"sort": [{"foo": {"order": "asc", "missing": "_first"}}]}));
    expect(&v, &[("d", json!([null])), ("b", json!(["apple"])), ("a", json!(["bar"])), ("e", json!(["delta"])), ("c", json!(["mango"]))]);
    // A custom missing value sorts the doc as if it held it.
    let v = search(&s, json!({"sort": [{"foo": {"order": "asc", "missing": "c"}}]}));
    expect(&v, &[("b", json!(["apple"])), ("a", json!(["bar"])), ("d", json!(["c"])), ("e", json!(["delta"])), ("c", json!(["mango"]))]);
}

#[test]
fn text_fielddata_sort_modes() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let v = search(&s, json!({"sort": [{"foo": {"order": "asc", "mode": "max"}}]}));
    expect(&v, &[("c", json!(["mango"])), ("a", json!(["quick"])), ("e", json!(["yak"])), ("b", json!(["zebra"])), ("d", json!([null]))]);
    let v = search(&s, json!({"sort": [{"foo": {"order": "desc", "mode": "min"}}]}));
    expect(&v, &[("c", json!(["mango"])), ("e", json!(["delta"])), ("a", json!(["bar"])), ("b", json!(["apple"])), ("d", json!([null]))]);
}

#[test]
fn text_fielddata_sort_secondary_and_paging() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let v = search(&s, json!({"sort": [{"foo": "asc"}, {"n": "desc"}], "size": 2}));
    expect(&v, &[("b", json!(["apple", 2])), ("a", json!(["bar", 1]))]);
    let v = search(&s, json!({"sort": [{"foo": "asc"}, {"n": "desc"}], "size": 2, "search_after": ["bar", 1]}));
    expect(&v, &[("e", json!(["delta", 5])), ("c", json!(["mango", 3]))]);
}

#[test]
fn text_fielddata_sort_across_shards() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 2);
    let v = search(&s, json!({"sort": "foo"}));
    expect(&v, &[("b", json!(["apple"])), ("a", json!(["bar"])), ("e", json!(["delta"])), ("c", json!(["mango"])), ("d", json!([null]))]);
    let v = search(&s, json!({"sort": [{"foo": "desc"}]}));
    expect(&v, &[("b", json!(["zebra"])), ("e", json!(["yak"])), ("a", json!(["quick"])), ("c", json!(["mango"])), ("d", json!([null]))]);
}

#[test]
fn text_fielddata_is_uninverted_once_per_segment() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let id = s.lookup().get("foo").and_then(|f| f.core_id).unwrap();
    let shard = s.local_shards().into_iter().next().unwrap();
    let segs: Vec<_> = shard.shard.core.read().segments().to_vec();
    assert!(segs.len() >= 2);
    assert!(segs.iter().all(|g| !g.has_uninverted(id)), "nothing built before a sort");
    search(&s, json!({"sort": "foo"}));
    assert!(segs.iter().all(|g| g.has_uninverted(id)), "a sort builds every segment's fielddata");
    let first: Vec<_> = segs.iter().map(|g| g.uninverted(id).unwrap()).collect();
    search(&s, json!({"sort": [{"foo": "desc"}]}));
    for (g, f) in segs.iter().zip(&first) {
        assert!(Arc::ptr_eq(f, &g.uninverted(id).unwrap()), "later sorts reuse the cached table");
    }
}

#[test]
fn text_fielddata_terms_agg_and_docvalue_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let s = index(tmp.path(), 1);
    let v = search(&s, json!({"size": 0, "aggs": {"t": {"terms": {"field": "foo", "order": {"_key": "asc"}}}}}));
    let keys: Vec<(&str, u64)> = v["aggregations"]["t"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"].as_str().unwrap(), b["doc_count"].as_u64().unwrap())).collect();
    assert_eq!(keys, [("apple", 1), ("bar", 1), ("delta", 1), ("mango", 1), ("quick", 1), ("yak", 1), ("zebra", 1)]);
    let v = search(&s, json!({"query": {"ids": {"values": ["a"]}}, "docvalue_fields": ["foo"]}));
    assert_eq!(v["hits"]["hits"][0]["fields"]["foo"], json!(["bar", "quick"]));
}
