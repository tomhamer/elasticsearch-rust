//! Index + query + aggregation tests for every field type and mapping
//! parameter of this module, on an in-memory core index.

use crate::aggs::{run_aggregations, AggEnv, AggSeg};
use crate::analysis::AnalysisRegistry;
use crate::document::{parse_with_dynamic, parse_with_limits};
use crate::error::Result;
use crate::mapping::{FieldLookup, Mappings, ID_FIELD};
use crate::query::dsl::{parse_query, QueryContext};
use rustic_core::query::LeafReader;
use rustic_core::types::{DocId, IndexSettings};
use rustic_core::Index;
use serde_json::{json, Value};

struct Fx {
    index: Index,
    lookup: FieldLookup,
    mappings: Mappings,
    ids: Vec<String>,
}

fn fx(mapping: Value, docs: &[Value]) -> Fx {
    try_fx(mapping, docs).unwrap()
}

fn try_fx(mapping: Value, docs: &[Value]) -> Result<Fx> {
    let reg = AnalysisRegistry::default();
    let mut m = Mappings::parse(&mapping, &reg)?;
    m.assign_core_ids();
    let mut lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    let mut ids: Vec<String> = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        let id = d.get("_test_id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| (i + 1).to_string());
        let mut src = d.clone();
        if let Some(o) = src.as_object_mut() {
            o.remove("_test_id");
        }
        let (mut pd, upd) = parse_with_dynamic(&src, &id, &m, &lookup, 1000)?;
        if let Some((nm, nlk)) = upd {
            m = nm;
            lookup = nlk;
            // Schema may have grown: rebuild the index with all docs so far.
            let mut fresh = Index::new("idx", lookup.core_schema(), IndexSettings::default());
            for (j, prev) in docs[..i].iter().enumerate() {
                let mut p = prev.clone();
                if let Some(o) = p.as_object_mut() {
                    o.remove("_test_id");
                }
                let (mut pp, _) = parse_with_dynamic(&p, &ids[j], &m, &lookup, 1000)?;
                pp.doc.add_keyword(ID_FIELD, &ids[j]);
                fresh.add_document(&pp.doc).unwrap();
            }
            index = fresh;
            let (pd2, _) = parse_with_dynamic(&src, &id, &m, &lookup, 1000)?;
            pd = pd2;
        }
        pd.doc.add_keyword(ID_FIELD, &id);
        index.add_document(&pd.doc).unwrap();
        ids.push(id);
        if i == docs.len() / 2 {
            index.refresh().unwrap();
        }
    }
    Ok(Fx { index, lookup, mappings: m, ids })
}

fn qctx<'a>(f: &'a Fx, leaves: &'a [LeafReader<'a>]) -> QueryContext<'a> {
    QueryContext {
        deadline: None,
        lookup: &f.lookup,
        index_name: "idx",
        index_uuid: "uuid",
        aliases: vec![],
        now: 1_600_000_000_000,
        leaves,
        default_fields: vec!["*".into()],
        terms_lookup: None,
        max_terms_count: 65536,
        max_regex_length: 1000,
        max_result_window: 10_000,
        seq_no_disabled: false,
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: true,
        mappings: None,
    }
}

/// Ids of the docs matching `q`, sorted by score desc then id.
fn search(f: &Fx, q: Value) -> Vec<String> {
    try_search(f, q).unwrap().into_iter().map(|(id, _)| id).collect()
}

fn try_search(f: &Fx, q: Value) -> Result<Vec<(String, f32)>> {
    crate::script::with_env(crate::script::ScriptEnv::new(None, None), || {
        let result = try_search_with_context(f, q);
        match crate::script::env().and_then(|env| env.take_error()) {
            Some(error) => Err(error),
            None => result,
        }
    })
}

fn try_search_with_context(f: &Fx, q: Value) -> Result<Vec<(String, f32)>> {
    let leaves = f.index.leaves();
    let ctx = qctx(f, &leaves);
    let query = parse_query(&q, &ctx)?;
    let mut out: Vec<(String, f32)> = super::join::run(&query, &leaves).unwrap()
        .into_iter()
        .map(|(l, d, s)| (leaves[l].keyword_value(ID_FIELD, d).unwrap_or_default(), s))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    Ok(out)
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn aggs(f: &Fx, query: Option<Value>, a: Value) -> Result<Value> {
    let leaves = f.index.leaves();
    let matching: Vec<Vec<DocId>> = match query {
        None => leaves.iter().map(|l| (0..l.num_docs()).collect()).collect(),
        Some(q) => {
            let ctx = qctx(f, &leaves);
            let query = parse_query(&q, &ctx)?;
            let mut per: Vec<Vec<DocId>> = vec![Vec::new(); leaves.len()];
            for (l, d, _) in super::join::run(&query, &leaves).unwrap() {
                per[l].push(d);
            }
            per.iter_mut().for_each(|v| v.sort());
            per
        }
    };
    let segs: Vec<AggSeg> = leaves
        .iter()
        .zip(matching)
        .map(|(l, docs)| AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, scores: vec![1.0; docs.len()], docs })
        .collect();
    let fetch = |_s: usize, _d: DocId, _sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({})) };
    let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    run_aggregations(&a, &segs, &env)
}

fn keys(v: &Value) -> Vec<(Value, u64)> {
    v["buckets"].as_array().unwrap().iter().map(|b| (b["key"].clone(), b["doc_count"].as_u64().unwrap())).collect()
}

// ---------------------------------------------------------------------------

#[test]
fn ip_cidr_range_and_aggs() {
    let f = fx(
        json!({"properties": {"ip": {"type": "ip", "null_value": "127.0.0.1"}, "bad": {"type": "ip", "ignore_malformed": true}}}),
        &[json!({"ip": "192.168.1.10"}), json!({"ip": "192.168.200.1"}), json!({"ip": "10.0.0.1", "bad": "nope"}), json!({"ip": null}), json!({"ip": "2001:db8::1"})],
    );
    assert_eq!(sorted(search(&f, json!({"term": {"ip": "192.168.0.0/16"}}))), ids(&["1", "2"]));
    assert_eq!(search(&f, json!({"term": {"ip": "127.0.0.1"}})), ids(&["4"]));
    assert_eq!(sorted(search(&f, json!({"range": {"ip": {"gte": "10.0.0.0", "lt": "192.168.100.0"}}}))), ids(&["1", "3", "4"]));
    assert_eq!(search(&f, json!({"term": {"ip": "2001:db8::/32"}})), ids(&["5"]));
    let r = aggs(&f, None, json!({"t": {"terms": {"field": "ip", "size": 10}}, "r": {"ip_range": {"field": "ip", "ranges": [{"to": "100.0.0.0"}, {"mask": "192.168.0.0/16"}]}}})).unwrap();
    assert_eq!(r["t"]["buckets"].as_array().unwrap().len(), 5);
    assert_eq!(r["r"]["buckets"][0]["doc_count"], 1);
    assert_eq!(r["r"]["buckets"][1], json!({"key": "192.168.0.0/16", "from": "192.168.0.0", "to": "192.169.0.0", "doc_count": 2}));
    let e = aggs(&f, None, json!({"m": {"min": {"field": "ip"}}})).unwrap_err();
    assert!(e.reason.contains("is not supported for aggregation [min]"), "{}", e.reason);
    // ignore_malformed lists the field in _ignored
    let reg = AnalysisRegistry::default();
    let (pd, _) = parse_with_dynamic(&json!({"bad": "zzz"}), "x", &f.mappings, &FieldLookup::build(&f.mappings, &reg), 1000).unwrap();
    assert_eq!(pd.ignored, vec!["bad".to_string()]);
}

#[test]
fn flattened_root_keyed_and_aggs() {
    let f = fx(
        json!({"properties": {"labels": {"type": "flattened", "ignore_above": 10}}}),
        &[
            json!({"labels": {"env": "prod", "team": {"name": "search"}, "tags": ["a", "b"]}}),
            json!({"labels": {"env": "dev", "team": {"name": "ml"}}}),
            json!({"labels": {"env": "prod", "long": "this value is too long"}}),
        ],
    );
    assert_eq!(sorted(search(&f, json!({"term": {"labels.env": "prod"}}))), ids(&["1", "3"]));
    assert_eq!(sorted(search(&f, json!({"term": {"labels": "prod"}}))), ids(&["1", "3"]));
    assert_eq!(search(&f, json!({"term": {"labels.team.name": "ml"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"term": {"labels.env": "search"}})), Vec::<String>::new());
    assert_eq!(sorted(search(&f, json!({"exists": {"field": "labels.team.name"}}))), ids(&["1", "2"]));
    assert_eq!(search(&f, json!({"exists": {"field": "labels.nope"}})), Vec::<String>::new());
    assert_eq!(sorted(search(&f, json!({"prefix": {"labels.env": "pr"}}))), ids(&["1", "3"]));
    assert_eq!(search(&f, json!({"range": {"labels.env": {"lt": "e"}}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"range": {"labels.env": {"gte": "a"}}}))), ids(&["1", "2", "3"]));
    assert_eq!(search(&f, json!({"wildcard": {"labels.team.name": "m*"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"query_string": {"query": "labels.env:dev"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"exists": {"field": "labels.long"}})), Vec::<String>::new());
    let r = aggs(&f, None, json!({"e": {"terms": {"field": "labels.env"}}, "all": {"terms": {"field": "labels", "size": 20}}})).unwrap();
    assert_eq!(keys(&r["e"]), vec![(json!("prod"), 2), (json!("dev"), 1)]);
    assert!(keys(&r["all"]).contains(&(json!("search"), 1)));
    // depth_limit
    let e = try_fx(json!({"properties": {"l": {"type": "flattened", "depth_limit": 1}}}), &[json!({"l": {"a": {"b": "c"}}})]).err().unwrap();
    assert!(e.caused_by.unwrap().reason.contains("exceeds the maximum depth limit of [1]"));
}

#[test]
fn join_queries_and_aggs() {
    let f = fx(
        json!({"properties": {"rel": {"type": "join", "relations": {"question": "answer"}}, "text": {"type": "text"}, "votes": {"type": "long"}}}),
        &[
            json!({"_test_id": "q1", "rel": "question", "text": "how to rust"}),
            json!({"_test_id": "q2", "rel": {"name": "question"}, "text": "how to java"}),
            json!({"_test_id": "a1", "rel": {"name": "answer", "parent": "q1"}, "text": "use cargo", "votes": 3}),
            json!({"_test_id": "a2", "rel": {"name": "answer", "parent": "q1"}, "text": "read the book", "votes": 5}),
            json!({"_test_id": "a3", "rel": {"name": "answer", "parent": "q2"}, "text": "use maven", "votes": 1}),
        ],
    );
    assert_eq!(search(&f, json!({"has_child": {"type": "answer", "query": {"match": {"text": "book"}}}})), ids(&["q1"]));
    assert_eq!(sorted(search(&f, json!({"has_child": {"type": "answer", "query": {"match": {"text": "use"}}}}))), ids(&["q1", "q2"]));
    assert_eq!(search(&f, json!({"has_child": {"type": "answer", "query": {"match_all": {}}, "min_children": 2}})), ids(&["q1"]));
    assert_eq!(sorted(search(&f, json!({"has_parent": {"parent_type": "question", "query": {"match": {"text": "rust"}}}}))), ids(&["a1", "a2"]));
    assert_eq!(sorted(search(&f, json!({"parent_id": {"type": "answer", "id": "q1"}}))), ids(&["a1", "a2"]));
    let sc = try_search(&f, json!({"has_child": {"type": "answer", "query": {"function_score": {"query": {"match_all": {}}, "functions": [{"field_value_factor": {"field": "votes"}}], "boost_mode": "replace"}}, "score_mode": "sum"}})).unwrap();
    assert_eq!(sc[0].0, "q1");
    assert!((sc[0].1 - 8.0).abs() < 1e-4, "{sc:?}");
    let e = try_search(&f, json!({"has_child": {"type": "nope", "query": {"match_all": {}}}})).unwrap_err();
    assert!(format!("{} {:?}", e.reason, e.caused_by).contains("join field [rel] doesn't hold [nope] as a child"));
    // children / parent aggregations.
    let r = aggs(&f, Some(json!({"term": {"rel": "question"}})), json!({"q": {"filter": {"ids": {"values": ["q1"]}}, "aggs": {"c": {"children": {"type": "answer"}, "aggs": {"v": {"sum": {"field": "votes"}}}}}}})).unwrap();
    let q1 = &r["q"];
    assert_eq!(q1["c"]["doc_count"], 2);
    assert_eq!(q1["c"]["v"]["value"], 8.0);
    let r = aggs(&f, Some(json!({"term": {"rel": "answer"}})), json!({"p": {"parent": {"type": "answer"}}})).unwrap();
    assert_eq!(r["p"]["doc_count"], 2);
    // child without parent
    let e = try_fx(json!({"properties": {"rel": {"type": "join", "relations": {"question": "answer"}}}}), &[json!({"rel": "answer"})]).err().unwrap();
    assert!(e.caused_by.unwrap().reason.contains("[parent] is missing"));
    // only one join field per index
    let e = Mappings::parse(&json!({"properties": {"a": {"type": "join", "relations": {"p": "c"}}, "b": {"type": "join", "relations": {"x": "y"}}}}), &AnalysisRegistry::default()).unwrap_err();
    assert!(e.reason.contains("Only one [parent-join] field"), "{}", e.reason);
}

#[test]
fn alias_resolution_and_validation() {
    let f = fx(
        json!({"properties": {"distance": {"type": "long"}, "route_length_miles": {"type": "alias", "path": "distance"}}}),
        &[json!({"distance": 10}), json!({"distance": 20})],
    );
    assert_eq!(search(&f, json!({"range": {"route_length_miles": {"gte": 15}}})), ids(&["2"]));
    let r = aggs(&f, None, json!({"m": {"max": {"field": "route_length_miles"}}})).unwrap();
    assert_eq!(r["m"]["value"], 20.0);
    let reg = AnalysisRegistry::default();
    let e = Mappings::parse(&json!({"properties": {"a": {"type": "alias", "path": "missing"}}}), &reg).unwrap_err();
    assert!(e.reason.contains("an alias must refer to an existing field in the mappings"), "{}", e.reason);
    let e = Mappings::parse(&json!({"properties": {"x": {"type": "long"}, "a": {"type": "alias", "path": "x"}, "b": {"type": "alias", "path": "a"}}}), &reg).unwrap_err();
    assert!(e.reason.contains("an alias cannot refer to another alias"), "{}", e.reason);
}

#[test]
fn date_nanos_precision_and_errors() {
    let f = fx(
        json!({"properties": {"d": {"type": "date_nanos"}}}),
        &[json!({"d": "2018-10-29T12:12:12.123456789Z"}), json!({"d": "2018-10-29T12:12:12.987654321Z"}), json!({"d": "2018-10-30T00:00:00Z"})],
    );
    let leaves = f.index.leaves();
    let id = f.lookup.get("d").unwrap().core_id.unwrap();
    let mut all: Vec<i64> = leaves.iter().flat_map(|l| (0..l.num_docs()).flat_map(move |d| l.i64_values(id, d))).collect();
    all.sort();
    assert_eq!(all[0], 1540815132123456789);
    assert_eq!(search(&f, json!({"term": {"d": "2018-10-29T12:12:12.123456789Z"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"range": {"d": {"gt": "2018-10-29T12:12:12.123456789Z", "lt": "2018-10-30"}}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"range": {"d": {"gte": "2018-10-29||/d", "lte": "2018-10-29||/d"}}}))), ids(&["1", "2"]));
    let r = aggs(&f, None, json!({"h": {"date_histogram": {"field": "d", "calendar_interval": "1d"}}, "m": {"min": {"field": "d"}}})).unwrap();
    assert_eq!(r["h"]["buckets"][0]["key"], 1540771200000i64);
    assert_eq!(r["h"]["buckets"][0]["doc_count"], 2);
    assert_eq!(r["m"]["value"], 1540815132123.0);
    let rf = f.lookup.get("d").unwrap();
    assert_eq!(super::format_nanos(1540815132123456789, None, rf), "2018-10-29T12:12:12.123456789Z");
    assert_eq!(super::format_nanos(1540815132123456789, Some("epoch_millis"), rf), "1540815132123.456789");
    assert_eq!(super::format_nanos(1540815132123456789, Some("strict_date_optional_time"), rf), "2018-10-29T12:12:12.123Z");
    let e = try_fx(json!({"properties": {"d": {"type": "date_nanos"}}}), &[json!({"d": "1969-10-28T12:12:12.123456789Z"})]).err().unwrap();
    assert_eq!(e.kind, "document_parsing_exception");
    assert_eq!(e.caused_by.unwrap().reason, "date[1969-10-28T12:12:12.123456789Z] is before the epoch in 1970 and cannot be stored in nanosecond resolution");
    let e = try_fx(json!({"properties": {"d": {"type": "date_nanos"}}}), &[json!({"d": "2263-10-29T12:12:12.123456789Z"})]).err().unwrap();
    assert_eq!(e.caused_by.unwrap().reason, "date[2263-10-29T12:12:12.123456789Z] is after 2262-04-11T23:47:16.854775807 and cannot be stored in nanosecond resolution");
}

#[test]
fn scaled_float_and_unsigned_long() {
    let f = fx(
        json!({"properties": {"p": {"type": "scaled_float", "scaling_factor": 100}, "u": {"type": "unsigned_long"}}}),
        &[json!({"p": 1.005, "u": 18446744073709551615u64}), json!({"p": 2.5, "u": 9223372036854775808u64}), json!({"p": 1e300, "u": 1})],
    );
    assert_eq!(search(&f, json!({"term": {"u": "18446744073709551615"}})), ids(&["1"]));
    assert_eq!(sorted(search(&f, json!({"range": {"u": {"gte": 9223372036854775808u64}}}))), ids(&["1", "2"]));
    assert_eq!(search(&f, json!({"range": {"u": {"lt": "9223372036854775808"}}})), ids(&["3"]));
    assert_eq!(search(&f, json!({"range": {"p": {"gte": 2.0, "lte": 3.0}}})), ids(&["2"]));
    let r = aggs(&f, Some(json!({"range": {"p": {"lte": 100}}})), json!({"s": {"sum": {"field": "p"}}, "mx": {"max": {"field": "u"}}})).unwrap();
    assert_eq!(r["s"]["value"], 3.5);
    assert_eq!(r["mx"]["value"], 18446744073709551615.0);
    // Saturation: 1e300 * 100 → Long.MAX / 100.
    let leaves = f.index.leaves();
    let id = f.lookup.get("p").unwrap().core_id.unwrap();
    let mx = leaves.iter().flat_map(|l| (0..l.num_docs()).flat_map(move |d| l.f64_values(id, d))).fold(0.0f64, f64::max);
    assert_eq!(mx, i64::MAX as f64 / 100.0);
}

#[test]
fn range_fields_relations_and_aggs() {
    let f = fx(
        json!({"properties": {
            "ir": {"type": "integer_range"}, "dr": {"type": "date_range", "format": "yyyy-MM-dd"}, "fr": {"type": "float_range"},
            "ipr": {"type": "ip_range"}, "lr": {"type": "long_range"}, "dbl": {"type": "double_range"}
        }}),
        &[
            json!({"ir": {"gte": 1, "lte": 5}, "dr": {"gte": "2017-09-01", "lte": "2017-09-05"}, "fr": {"gte": 1.5, "lt": 2.5}, "ipr": "192.168.0.0/16", "lr": {"gt": 10}}),
            json!({"ir": {"gte": 1, "lte": 3}, "dr": {"gte": "2017-09-01", "lte": "2017-09-03"}, "ipr": {"gte": "10.0.0.1", "lte": "10.0.0.5"}, "dbl": [{"gte": 0.5, "lte": 1.0}, {"gte": 10.0}]}),
            json!({"ir": {"gte": 4, "lte": 5}, "dr": {"gt": "2017-09-03", "lt": "2017-09-06"}}),
            json!({"ir": null}),
        ],
    );
    let q = |rel: &str| json!({"range": {"ir": {"gte": 3, "lte": 4, "relation": rel}}});
    assert_eq!(sorted(search(&f, q("intersects"))), ids(&["1", "2", "3"]));
    assert_eq!(search(&f, q("contains")), ids(&["1"]));
    assert_eq!(search(&f, q("within")), Vec::<String>::new());
    // ES `RangeQueryBuilder.relation`: `disjoint` is a shape relation range queries refuse.
    assert_eq!(try_search(&f, q("disjoint")).unwrap_err().reason, "[range] query does not support relation [disjoint]");
    assert_eq!(sorted(search(&f, json!({"range": {"ir": {"gte": 0, "lte": 3, "relation": "within"}}}))), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"term": {"ir": 2}}))), ids(&["1", "2"]));
    assert_eq!(sorted(search(&f, json!({"query_string": {"query": "ir:[3 TO 4]"}}))), ids(&["1", "2", "3"]));
    assert_eq!(sorted(search(&f, json!({"range": {"dr": {"gte": "2017-09-03", "lte": "2017-09-04"}}}))), ids(&["1", "2", "3"]));
    assert_eq!(search(&f, json!({"range": {"dr": {"gte": "2017-09-03", "lte": "2017-09-04", "relation": "contains"}}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"ipr": "192.168.3.4"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"range": {"ipr": {"gte": "10.0.0.0", "lte": "10.0.0.255", "relation": "contains"}}})), Vec::<String>::new());
    assert_eq!(search(&f, json!({"range": {"ipr": {"gte": "10.0.0.0", "lte": "10.0.0.255", "relation": "within"}}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"term": {"fr": 2.5}})), Vec::<String>::new());
    assert_eq!(search(&f, json!({"term": {"fr": 2.4}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"lr": 10}})), Vec::<String>::new());
    assert_eq!(search(&f, json!({"term": {"lr": 11}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"dbl": 1e9}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"exists": {"field": "ir"}}))), ids(&["1", "2", "3"]));
    let r = aggs(&f, None, json!({"h": {"histogram": {"field": "ir", "interval": 2}}})).unwrap();
    assert_eq!(keys(&r["h"]), vec![(json!(0.0), 2), (json!(2.0), 2), (json!(4.0), 2)]);
    let e = aggs(&f, None, json!({"t": {"terms": {"field": "ir"}}})).unwrap_err();
    assert!(e.reason.contains("not supported for aggregation [terms]"));
    let e = try_fx(json!({"properties": {"ir": {"type": "integer_range"}}}), &[json!({"ir": {"gte": 5, "lte": 1}})]).err().unwrap();
    assert!(e.caused_by.unwrap().reason.contains("min value (5) is greater than max value (1)"));
}

#[test]
fn match_only_text_constant_scores() {
    let f = fx(json!({"properties": {"m": {"type": "match_only_text"}}}), &[json!({"m": "quick brown fox"}), json!({"m": "quick quick dog"}), json!({"m": "lazy dog"})]);
    let r = try_search(&f, json!({"match": {"m": "quick"}})).unwrap();
    assert_eq!(r.len(), 2);
    assert!(r.iter().all(|(_, s)| (*s - 1.0).abs() < 1e-6), "{r:?}");
    assert_eq!(search(&f, json!({"match_phrase": {"m": "brown fox"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"match_phrase": {"m": "fox brown"}})), Vec::<String>::new());
    let r = try_search(&f, json!({"match": {"m": {"query": "dog", "boost": 2.0}}})).unwrap();
    assert!(r.iter().all(|(_, s)| (*s - 2.0).abs() < 1e-6));
}

#[test]
fn constant_keyword_rewrites() {
    let f = fx(json!({"properties": {"ds": {"type": "constant_keyword", "value": "logs"}, "n": {"type": "long"}}}), &[json!({"n": 1}), json!({"ds": "logs", "n": 2})]);
    assert_eq!(sorted(search(&f, json!({"term": {"ds": "logs"}}))), ids(&["1", "2"]));
    assert!(search(&f, json!({"term": {"ds": "metrics"}})).is_empty());
    assert_eq!(search(&f, json!({"prefix": {"ds": "lo"}})).len(), 2);
    assert_eq!(search(&f, json!({"wildcard": {"ds": "l*s"}})).len(), 2);
    assert!(search(&f, json!({"regexp": {"ds": "x.*"}})).is_empty());
    assert_eq!(search(&f, json!({"exists": {"field": "ds"}})).len(), 2);
    assert_eq!(search(&f, json!({"range": {"ds": {"gte": "a", "lte": "m"}}})).len(), 2);
    let r = aggs(&f, None, json!({"t": {"terms": {"field": "ds"}}})).unwrap();
    assert_eq!(keys(&r["t"]), vec![(json!("logs"), 2)]);
    let e = try_fx(json!({"properties": {"ds": {"type": "constant_keyword", "value": "logs"}}}), &[json!({"ds": "other"})]).err().unwrap();
    assert!(e.caused_by.unwrap().reason.contains("only accepts values that are equal to the value defined in the mappings [logs], but got [other]"));
    // value learnt from the first document
    let f = fx(json!({"properties": {"ds": {"type": "constant_keyword"}}}), &[json!({"ds": "metrics"})]);
    assert_eq!(f.mappings.to_json()["properties"]["ds"]["value"], json!("metrics"));
}

#[test]
fn wildcard_field_ngram_verification() {
    let f = fx(
        json!({"properties": {"w": {"type": "wildcard", "ignore_above": 30}}}),
        &[json!({"w": "C:\\Windows\\System32\\cmd.exe"}), json!({"w": "/usr/bin/bash"}), json!({"w": "powershell.EXE"}), json!({"w": "x".repeat(40)})],
    );
    assert_eq!(search(&f, json!({"wildcard": {"w": "*cmd*"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"wildcard": {"w": {"value": "*.exe", "case_insensitive": true}}})).len(), 2);
    assert_eq!(search(&f, json!({"wildcard": {"w": "*.exe"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"prefix": {"w": "/usr"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"regexp": {"w": "/usr/.*/bash"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"term": {"w": "/usr/bin/bash"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"wildcard": {"w": "x*"}})), Vec::<String>::new());
    let r = aggs(&f, None, json!({"t": {"terms": {"field": "w"}}})).unwrap();
    assert_eq!(r["t"]["buckets"].as_array().unwrap().len(), 3);
}

#[test]
fn completion_suggester() {
    let f = fx(
        json!({"properties": {"s": {"type": "completion", "contexts": [{"name": "genre", "type": "category"}]}, "title": {"type": "keyword"}}}),
        &[
            json!({"s": {"input": ["Nevermind", "Nirvana"], "weight": 34, "contexts": {"genre": "rock"}}, "title": "a"}),
            json!({"s": {"input": "Nirvana Live", "weight": 50, "contexts": {"genre": ["live", "rock"]}}}),
            json!({"s": [{"input": "bar", "weight": 3, "contexts": {"genre": "x"}}, {"input": "baz", "weight": 9, "contexts": {"genre": "x"}}]}),
        ],
    );
    let leaves = f.index.leaves();
    let t = [super::suggest::SuggestTarget { index: "idx", leaves: &leaves, lookup: &f.lookup, svc: None }];
    let r = super::suggest::run(&json!({"r": {"prefix": "nir", "completion": {"field": "s", "contexts": {"genre": ["rock", "live", "x"]}}}}), &t).unwrap();
    let opts = r["r"][0]["options"].as_array().unwrap();
    assert_eq!(opts.iter().map(|o| o["text"].as_str().unwrap()).collect::<Vec<_>>(), vec!["Nirvana Live", "Nirvana"]);
    assert_eq!(opts[0]["_score"], 50.0);
    assert_eq!(r["r"][0]["text"], "nir");
    let r = super::suggest::run(&json!({"r": {"text": "b", "completion": {"field": "s", "size": 1, "contexts": {"genre": ["rock", "live", "x"]}}}}), &t).unwrap();
    assert_eq!(r["r"][0]["options"][0]["text"], "baz");
    let r = super::suggest::run(&json!({"r": {"prefix": "nir", "completion": {"field": "s", "contexts": {"genre": ["live"]}}}}), &t).unwrap();
    assert_eq!(r["r"][0]["options"].as_array().unwrap().len(), 1);
    let r = super::suggest::run(&json!({"r": {"prefix": "nirv", "completion": {"field": "s", "fuzzy": {"fuzziness": 1}, "contexts": {"genre": ["rock", "live", "x"]}}}}), &t).unwrap();
    assert_eq!(r["r"][0]["options"].as_array().unwrap().len(), 2);
    let r = super::suggest::run(&json!({"r": {"prefix": "nevr", "completion": {"field": "s", "fuzzy": {"fuzziness": 1}, "contexts": {"genre": ["rock", "live", "x"]}}}}), &t).unwrap();
    assert_eq!(r["r"][0]["options"][0]["text"], "Nevermind");
    // ES: a context-enabled field needs query contexts.
    assert!(super::suggest::run(&json!({"r": {"prefix": "nir", "completion": {"field": "s"}}}), &t).is_err());
}

#[test]
fn search_as_you_type_subfields() {
    let f = fx(json!({"properties": {"t": {"type": "search_as_you_type"}}}), &[json!({"t": "quick brown fox jumps"}), json!({"t": "quick red fox"}), json!({"t": "brown dog"})]);
    for sub in ["t._2gram", "t._3gram", "t._index_prefix"] {
        assert!(f.lookup.get(sub).is_some(), "{sub}");
    }
    let q = |text: &str| json!({"multi_match": {"query": text, "type": "bool_prefix", "fields": ["t", "t._2gram", "t._3gram"]}});
    let r = search(&f, q("quick br"));
    assert_eq!(r[0], "1");
    assert_eq!(search(&f, json!({"match_phrase": {"t._2gram": "quick brown"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"match": {"t._3gram": "quick red fox"}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"match_bool_prefix": {"t._index_prefix": "bro"}}))), ids(&["1", "3"]));
}

#[test]
fn token_count_murmur3_binary_size() {
    let f = fx(
        json!({"_size": {"enabled": true}, "properties": {
            "name": {"type": "text", "fields": {"length": {"type": "token_count", "analyzer": "standard"}}},
            "h": {"type": "murmur3"}, "b": {"type": "binary", "doc_values": true}
        }}),
        &[json!({"name": "John Smith", "h": "a", "b": "U29tZSBiaW5hcnkgYmxvYg=="}), json!({"name": "Rachel Alice Williams", "h": "b"}), json!({"name": "x", "h": "a"})],
    );
    assert_eq!(search(&f, json!({"term": {"name.length": 3}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"range": {"name.length": {"lte": 2}}}))), ids(&["1", "3"]));
    let r = aggs(&f, None, json!({"c": {"cardinality": {"field": "h"}}, "m": {"max": {"field": "name.length"}}, "sz": {"max": {"field": "_size"}}})).unwrap();
    assert_eq!(r["c"]["value"], 2);
    assert_eq!(r["m"]["value"], 3.0);
    assert!(r["sz"]["value"].as_f64().unwrap() > 20.0);
    assert!(search(&f, json!({"range": {"_size": {"gt": 0}}})).len() == 3);
    let e = try_search(&f, json!({"term": {"h": "a"}})).unwrap_err();
    assert!(e.reason.contains("Murmur3 fields are not searchable"));
    let e = try_fx(json!({"properties": {"b": {"type": "binary"}}}), &[json!({"b": "not base64!!"})]).err().unwrap();
    assert_eq!(e.kind, "document_parsing_exception");
    assert_eq!(search(&f, json!({"exists": {"field": "b"}})), ids(&["1"]));
    let leaves = f.index.leaves();
    let rf = f.lookup.get("b").unwrap().clone();
    let dv: Vec<Value> = leaves.iter().flat_map(|l| (0..l.num_docs()).flat_map(|d| super::docvalues(l, &rf, d, None).unwrap_or_default()).collect::<Vec<_>>()).collect();
    assert_eq!(dv, vec![json!("U29tZSBiaW5hcnkgYmxvYg==")]);
}

#[test]
fn geo_shape_shape_point() {
    let f = fx(
        json!({"properties": {"g": {"type": "geo_shape"}, "s": {"type": "shape"}, "p": {"type": "point"}}}),
        &[
            json!({"g": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}, "s": "POLYGON ((1000 1000, 2000 1000, 2000 2000, 1000 2000, 1000 1000))", "p": {"x": 1500, "y": 1500}}),
            json!({"g": "LINESTRING (20 20, 30 30)", "s": {"type": "point", "coordinates": [5000, 5000]}, "p": [10, 10]}),
            json!({"g": {"type": "point", "coordinates": [5, 5]}, "p": "POINT (3000 3000)"}),
            json!({"g": {"type": "envelope", "coordinates": [[-50, 50], [-40, 40]]}}),
        ],
    );
    let gq = |shape: Value, rel: &str| json!({"geo_shape": {"g": {"shape": shape, "relation": rel}}});
    let env = json!({"type": "envelope", "coordinates": [[-1, 11], [11, -1]]});
    assert_eq!(sorted(search(&f, gq(env.clone(), "intersects"))), ids(&["1", "3"]));
    assert_eq!(sorted(search(&f, gq(env.clone(), "within"))), ids(&["1", "3"]));
    assert_eq!(sorted(search(&f, gq(env.clone(), "disjoint"))), ids(&["2", "4"]));
    assert_eq!(search(&f, gq(json!({"type": "point", "coordinates": [2, 2]}), "contains")), ids(&["1"]));
    assert_eq!(search(&f, gq(json!("BBOX (-55, -35, 55, 35)"), "intersects")), ids(&["4"]));
    assert_eq!(search(&f, json!({"shape": {"s": {"shape": {"type": "envelope", "coordinates": [[0, 3000], [3000, 0]]}}}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"shape": {"p": {"shape": "BBOX (1000, 2000, 2000, 1000)", "relation": "intersects"}}})), ids(&["1"]));
    let e = try_search(&f, json!({"term": {"g": "x"}})).unwrap_err();
    assert!(e.reason.contains("Geometry fields do not support exact searching"));
    let e = try_fx(json!({"properties": {"g": {"type": "geo_shape"}}}), &[json!({"g": {"type": "point", "coordinates": [0, 95]}})]).err().unwrap();
    assert_eq!(e.kind, "document_parsing_exception");
    let f2 = fx(json!({"properties": {"g": {"type": "geo_shape", "ignore_malformed": true}}}), &[json!({"g": {"type": "point", "coordinates": [0, 95]}})]);
    assert!(search(&f2, json!({"exists": {"field": "g"}})).is_empty());
}

#[test]
fn histogram_and_aggregate_metric_double() {
    let f = fx(
        json!({"properties": {
            "h": {"type": "histogram"},
            "m": {"type": "aggregate_metric_double", "metrics": ["min", "max", "sum", "value_count"], "default_metric": "max"}
        }}),
        &[
            json!({"h": {"values": [0.1, 0.2, 0.3], "counts": [3, 7, 23]}, "m": {"min": 1.0, "max": 10.0, "sum": 30.0, "value_count": 5}}),
            json!({"h": {"values": [0.1, 0.25, 0.4], "counts": [1, 2, 3]}, "m": {"min": -2.0, "max": 3.0, "sum": 4.0, "value_count": 3}}),
        ],
    );
    let r = aggs(
        &f,
        None,
        json!({
            "vc": {"value_count": {"field": "h"}}, "s": {"sum": {"field": "h"}}, "mn": {"min": {"field": "h"}}, "mx": {"max": {"field": "h"}},
            "p": {"percentiles": {"field": "h", "percents": [50]}}, "hist": {"histogram": {"field": "h", "interval": 0.1}},
            "b": {"boxplot": {"field": "h"}}, "pr": {"percentile_ranks": {"field": "h", "values": [0.3]}},
            "mmin": {"min": {"field": "m"}}, "mmax": {"max": {"field": "m"}}, "msum": {"sum": {"field": "m"}}, "mvc": {"value_count": {"field": "m"}}, "mavg": {"avg": {"field": "m"}}
        }),
    )
    .unwrap();
    assert_eq!(r["vc"]["value"], 39);
    assert!((r["s"]["value"].as_f64().unwrap() - (0.3 + 1.4 + 6.9 + 0.1 + 0.5 + 1.2)).abs() < 1e-9);
    assert_eq!(r["mn"]["value"], 0.1);
    assert_eq!(r["mx"]["value"], 0.4);
    let p50 = r["p"]["values"]["50.0"].as_f64().unwrap();
    assert!(p50 > 0.2 && p50 <= 0.3, "{p50}");
    assert_eq!(r["hist"]["buckets"][0]["doc_count"], 4);
    assert!(r["b"]["q2"].as_f64().is_some());
    assert!(r["pr"]["values"]["0.3"].as_f64().unwrap() > 50.0);
    assert_eq!(r["mmin"]["value"], -2.0);
    assert_eq!(r["mmax"]["value"], 10.0);
    assert_eq!(r["msum"]["value"], 34.0);
    assert_eq!(r["mvc"]["value"], 8);
    assert_eq!(r["mavg"]["value"], 4.25);
    assert_eq!(search(&f, json!({"range": {"m": {"gte": 5}}})), ids(&["1"]));
    // ES 9 ignores the deprecated `default_metric` for queries: with sum and
    // value_count mapped, term/range match the average (doc 1: 30/5 = 6).
    assert_eq!(search(&f, json!({"term": {"m": 6.0}})), ids(&["1"]));
    assert!(search(&f, json!({"term": {"m": 3.0}})).is_empty());
    let e = try_fx(json!({"properties": {"h": {"type": "histogram"}}}), &[json!({"h": {"values": [2, 1], "counts": [1, 1]}})]).err().unwrap();
    // ES puts histogram causes in the top-level reason, with no `caused_by`.
    assert!(e.reason.contains("error parsing field [h], [values] values must be in increasing order"), "{}", e.reason);
    let e = try_fx(json!({"properties": {"m": {"type": "aggregate_metric_double", "metrics": ["min", "max"], "default_metric": "max"}}}), &[json!({"m": {"min": 1}})]).err().unwrap();
    assert!(e.caused_by.unwrap().reason.contains("must contain all metrics"));
    // `default_metric` is optional (and deprecated) in ES 9.
    assert!(Mappings::parse(&json!({"properties": {"m": {"type": "aggregate_metric_double", "metrics": ["min", "max"]}}}), &AnalysisRegistry::default()).is_ok());
    // tdigest / exponential_histogram share the histogram path.
    let f = fx(
        json!({"properties": {"t": {"type": "tdigest"}, "e": {"type": "exponential_histogram"}}}),
        &[json!({"t": {"centroids": [1.0, 2.0], "counts": [1, 1]}, "e": {"scale": 0, "positive": {"indices": [0, 1], "counts": [2, 3]}, "zero": {"count": 1}}})],
    );
    let r = aggs(&f, None, json!({"t": {"value_count": {"field": "t"}}, "e": {"value_count": {"field": "e"}}, "emin": {"min": {"field": "e"}}})).unwrap();
    assert_eq!(r["t"]["value"], 2);
    assert_eq!(r["e"]["value"], 6);
    assert_eq!(r["emin"]["value"], 0.0);
}

#[test]
fn passthrough_and_subobjects() {
    let f = fx(
        json!({"properties": {"attributes": {"type": "passthrough", "priority": 10, "properties": {"host": {"type": "keyword"}}}}}),
        &[json!({"attributes": {"host": "a"}}), json!({"attributes": {"host": "b"}})],
    );
    assert_eq!(search(&f, json!({"term": {"host": "a"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"attributes.host": "b"}})), ids(&["2"]));
    assert_eq!(f.mappings.to_json()["properties"]["attributes"]["type"], json!("passthrough"));
    // subobjects: false keeps dotted leaves flat
    let f = fx(
        json!({"properties": {"metrics": {"type": "object", "subobjects": false, "properties": {"time": {"type": "long"}, "time.max": {"type": "long"}}}}}),
        &[json!({"metrics": {"time": 10, "time.max": 100}}), json!({"metrics.time": 1, "metrics.time.max": 5})],
    );
    assert_eq!(search(&f, json!({"range": {"metrics.time.max": {"gte": 50}}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"metrics.time": 1}})), ids(&["2"]));
    let j = f.mappings.to_json();
    assert_eq!(j["properties"]["metrics"]["properties"]["time.max"]["type"], json!("long"));
    // root subobjects: false + dynamic dotted leaves
    let f = fx(json!({"subobjects": false, "properties": {"host.name": {"type": "keyword"}}}), &[json!({"host.name": "x", "host.id": 1, "time": 10, "time.max": 100})]);
    assert!(f.lookup.get("host.id").is_some());
    assert!(f.lookup.get("time.max").is_some());
    assert_eq!(search(&f, json!({"term": {"time.max": 100}})), ids(&["1"]));
}

#[test]
fn version_field() {
    let f = fx(
        json!({"properties": {"v": {"type": "version"}}}),
        &[json!({"v": "1.0.0"}), json!({"v": "1.10.0"}), json!({"v": "1.9.0-beta"}), json!({"v": "2.0.0-alpha.1"}), json!({"v": "not-a-version"})],
    );
    assert_eq!(sorted(search(&f, json!({"range": {"v": {"gte": "1.2.0", "lt": "2.0.0"}}}))), ids(&["2", "3", "4"]));
    assert_eq!(sorted(search(&f, json!({"range": {"v": {"gt": "1.9.0"}}}))), ids(&["2", "4", "5"]));
    assert_eq!(search(&f, json!({"term": {"v": "1.10.0"}})), ids(&["2"]));
    assert_eq!(sorted(search(&f, json!({"prefix": {"v": "1."}}))), ids(&["1", "2", "3"]));
    assert_eq!(search(&f, json!({"wildcard": {"v": "*beta"}})), ids(&["3"]));
    let r = aggs(&f, None, json!({"t": {"terms": {"field": "v"}}})).unwrap();
    assert_eq!(r["t"]["buckets"].as_array().unwrap().len(), 5);
}

#[test]
fn sparse_vector_and_rank_features() {
    let f = fx(
        json!({"properties": {"ml": {"type": "sparse_vector"}, "topics": {"type": "rank_features"}, "pagerank": {"type": "rank_feature"}, "url_len": {"type": "rank_feature", "positive_score_impact": false}}}),
        &[
            json!({"ml": [{"running": 2.4, "exercise": 0.1}, {"exercise": 1.1652642}], "topics": {"politics": 20, "economics": 50.8}, "pagerank": 10, "url_len": 20}),
            json!({"ml": {"walking": 2.47, "exercise": 2.07, "walk": 1.64}, "topics": {"politics": 5.2}, "pagerank": 100, "url_len": 100}),
            json!({"ml": {}}),
            json!({"ml": []}),
        ],
    );
    let r = try_search(&f, json!({"bool": {"should": [
        {"term": {"ml": {"value": "walk", "boost": 1.9790847}}}, {"term": {"ml": {"value": "walking", "boost": 1.7092685}}}, {"term": {"ml": {"value": "exercise", "boost": 0.84076905}}}
    ]}})).unwrap();
    assert_eq!(r[0].0, "2");
    assert_eq!(r[1].0, "1");
    assert!((r[1].1 - 0.9797).abs() < 0.01, "{r:?}");
    assert_eq!(sorted(search(&f, json!({"exists": {"field": "ml"}}))), ids(&["1", "2", "3"]));
    let r = try_search(&f, json!({"sparse_vector": {"field": "ml", "query_vector": {"running": 1.0, "walk": 2.0}}})).unwrap();
    assert_eq!(r[0].0, "2");
    assert!((r[0].1 - 3.28).abs() < 0.05);
    let r = try_search(&f, json!({"rank_feature": {"field": "pagerank", "saturation": {"pivot": 10}}})).unwrap();
    assert_eq!(r[0].0, "2");
    assert!((r[1].1 - 0.5).abs() < 0.01);
    let r = try_search(&f, json!({"rank_feature": {"field": "pagerank", "log": {"scaling_factor": 1}}})).unwrap();
    assert!((r[0].1 - (101f32).ln()).abs() < 0.05);
    let r = try_search(&f, json!({"rank_feature": {"field": "pagerank", "sigmoid": {"pivot": 10, "exponent": 1}}})).unwrap();
    assert!((r[1].1 - 0.5).abs() < 0.01);
    let r = try_search(&f, json!({"rank_feature": {"field": "url_len", "linear": {}}})).unwrap();
    assert_eq!(r[0].0, "1"); // shorter urls rank higher (inverted)
    let r = try_search(&f, json!({"rank_feature": {"field": "topics.politics"}})).unwrap();
    assert_eq!(r[0].0, "1");
    assert_eq!(r.len(), 2);
    assert!(try_search(&f, json!({"rank_feature": {"field": "topics.economics"}})).unwrap().len() == 1);
    let e = aggs(&f, None, json!({"t": {"terms": {"field": "pagerank"}}})).unwrap_err();
    assert!(e.reason.contains("do not support sorting, scripting or aggregating"));
    let e = try_fx(json!({"properties": {"r": {"type": "rank_feature"}}}), &[json!({"r": -1})]).err().unwrap();
    assert_eq!(e.kind, "document_parsing_exception");
}

#[test]
fn percolator_matches_stored_queries() {
    let f = fx(
        json!({"properties": {"query": {"type": "percolator"}, "message": {"type": "text"}, "n": {"type": "long"}}}),
        &[
            json!({"query": {"match": {"message": "bonsai tree"}}}),
            json!({"query": {"range": {"n": {"gte": 10}}}}),
            json!({"query": {"bool": {"must": [{"match": {"message": "tree"}}, {"term": {"n": 3}}]}}}),
        ],
    );
    assert_eq!(sorted(search(&f, json!({"percolate": {"field": "query", "document": {"message": "A new bonsai tree in the office", "n": 3}}}))), ids(&["1", "3"]));
    assert_eq!(sorted(search(&f, json!({"percolate": {"field": "query", "documents": [{"message": "nothing"}, {"n": 50}]}}))), ids(&["2"]));
    let e = try_fx(json!({"properties": {"query": {"type": "percolator"}}}), &[json!({"query": {"bogus_query": {}}})]).err().unwrap();
    assert_eq!(e.kind, "document_parsing_exception");
}

#[test]
fn percolator_terms_aux_field_populated() {
    // Verify the `terms` aux field is indexed at write time so candidate
    // pruning works: a percolate doc whose tokens overlap a stored query's
    // extracted terms must match, and one with no overlap must not.
    let f = fx(
        json!({"properties": {"query": {"type": "percolator"}, "body": {"type": "text"}}}),
        &[
            json!({"query": {"match": {"body": "bonsai tree"}}}),
            json!({"query": {"match": {"body": "hurricane katrina"}}}),
            json!({"query": {"match": {"body": "saddam hussein"}}}),
        ],
    );
    // "bonsai tree" doc → matches stored query 1 (terms overlap: bonsai, tree)
    assert_eq!(sorted(search(&f, json!({"percolate": {"field": "query", "document": {"body": "a bonsai tree in the office"}}}))), ids(&["1"]));
    // "hurricane katrina" doc → matches stored query 2
    assert_eq!(sorted(search(&f, json!({"percolate": {"field": "query", "document": {"body": "hurricane katrina hits the gulf"}}}))), ids(&["2"]));
    // "saddam hussein" doc → matches stored query 3
    assert_eq!(sorted(search(&f, json!({"percolate": {"field": "query", "document": {"body": "saddam hussein captured"}}}))), ids(&["3"]));
    // No overlap → no matches
    assert!(search(&f, json!({"percolate": {"field": "query", "document": {"body": "completely unrelated text"}}})).is_empty());
}

#[test]
fn annotated_and_semantic_text() {
    let f = fx(
        json!({"properties": {"a": {"type": "annotated_text"}, "sem": {"type": "semantic_text", "inference_id": "my-elser"}}}),
        &[json!({"a": "[Beck](Beck) announced a new tour", "sem": "the quick brown fox"}), json!({"a": "Investors in [Apple](Apple+Inc.&Company) rejoiced", "sem": "lazy dogs"})],
    );
    assert_eq!(search(&f, json!({"term": {"a": "Beck"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"a": "Apple Inc."}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"match_phrase": {"a": "new tour"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"match": {"a": "investors"}})), ids(&["2"]));
    assert_eq!(search(&f, json!({"semantic": {"field": "sem", "query": "fox"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"match": {"sem": "dogs"}})), ids(&["2"]));
    let (plain, ann) = super::index::parse_annotated("[Apple](Apple+Inc.&Company) rose");
    assert_eq!(plain, "Apple rose");
    assert_eq!(ann, vec!["Apple Inc.".to_string(), "Company".to_string()]);
}

#[test]
fn rank_vectors_mapping_and_indexing() {
    let f = fx(json!({"properties": {"rv": {"type": "rank_vectors", "dims": 2}}}), &[json!({"rv": [[0.5, 1.0], [1.0, 0.0]]})]);
    assert_eq!(search(&f, json!({"exists": {"field": "rv"}})), ids(&["1"]));
    let e = try_fx(json!({"properties": {"rv": {"type": "rank_vectors", "dims": 2}}}), &[json!({"rv": [[0.5, 1.0, 2.0]]})]).err().unwrap();
    assert!(e.reason.contains("different number of dimensions [3] than defined in the mapping [2]"));
    let f = fx(json!({"properties": {"rv": {"type": "rank_vectors"}}}), &[json!({"rv": [[1.0, 2.0, 3.0]]})]);
    assert_eq!(f.mappings.to_json()["properties"]["rv"]["dims"], json!(3));
}

#[test]
fn mapping_parameters() {
    let reg = AnalysisRegistry::default();
    // copy_to is not recursive and targets are not in _source.
    let f = fx(
        json!({"properties": {"a": {"type": "keyword", "copy_to": "b"}, "b": {"type": "keyword", "copy_to": "c"}, "c": {"type": "keyword"}}}),
        &[json!({"a": "x"}), json!({"b": "y"})],
    );
    assert_eq!(search(&f, json!({"term": {"b": "x"}})), ids(&["1"]));
    assert!(search(&f, json!({"term": {"c": "x"}})).is_empty());
    assert_eq!(search(&f, json!({"term": {"c": "y"}})), ids(&["2"]));
    // meta limits
    let e = Mappings::parse(&json!({"properties": {"x": {"type": "keyword", "meta": {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5", "f": "6"}}}}), &reg).unwrap_err();
    assert!(e.reason.contains("[meta] can't have more than 5 entries"), "{}", e.reason);
    assert!(Mappings::parse(&json!({"properties": {"x": {"type": "keyword", "meta": {"unit": "ms"}}}}), &reg).is_ok());
    // null_value must match the type
    let e = Mappings::parse(&json!({"properties": {"x": {"type": "long", "null_value": "abc"}}}), &reg).unwrap_err();
    assert!(e.reason.contains("For input string: \"abc\""), "{}", e.reason);
    // term_vector / time_series / synthetic_source_keep / script validation
    assert!(Mappings::parse(&json!({"properties": {"t": {"type": "text", "term_vector": "with_positions_offsets"}}}), &reg).is_ok());
    assert!(Mappings::parse(&json!({"properties": {"t": {"type": "text", "term_vector": "bogus"}}}), &reg).is_err());
    assert!(Mappings::parse(&json!({"properties": {"n": {"type": "long", "time_series_metric": "gauge"}, "k": {"type": "keyword", "time_series_dimension": true}}}), &reg).is_ok());
    assert!(Mappings::parse(&json!({"properties": {"n": {"type": "long", "time_series_metric": "summary"}}}), &reg).is_err());
    assert!(Mappings::parse(&json!({"properties": {"n": {"type": "long", "synthetic_source_keep": "arrays"}}}), &reg).is_ok());
    assert!(Mappings::parse(&json!({"properties": {"n": {"type": "long", "synthetic_source_keep": "x"}}}), &reg).is_err());
    assert!(Mappings::parse(&json!({"properties": {"n": {"type": "long", "on_script_error": "continue"}}}), &reg).is_err());
    // ignore_above counts chars, keeps the doc, lists _ignored
    let f = fx(json!({"properties": {"k": {"type": "keyword", "ignore_above": 3}}}), &[json!({"k": "äöü"}), json!({"k": "abcd"})]);
    assert_eq!(search(&f, json!({"term": {"k": "äöü"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"term": {"_ignored": "k"}})), Vec::<String>::new()); // _ignored is set by the index layer
    // position_increment_gap: phrases do not match across array values.
    let f = fx(
        json!({"properties": {"t": {"type": "text"}, "g0": {"type": "text", "position_increment_gap": 0}}}),
        &[json!({"t": ["John Abraham", "Lincoln Smith"], "g0": ["John Abraham", "Lincoln Smith"]})],
    );
    assert!(search(&f, json!({"match_phrase": {"t": "Abraham Lincoln"}})).is_empty());
    assert_eq!(search(&f, json!({"match_phrase": {"g0": "Abraham Lincoln"}})), ids(&["1"]));
    assert_eq!(search(&f, json!({"match_phrase": {"t": {"query": "Abraham Lincoln", "slop": 101}}})), ids(&["1"]));
}

#[test]
fn mapping_limits() {
    use super::limits::{check, Limits};
    let reg = AnalysisRegistry::default();
    let deep = json!({"properties": {"a": {"properties": {"b": {"properties": {"c": {"type": "long"}}}}}}});
    let mut m = Mappings::parse(&deep, &reg).unwrap();
    m.assign_core_ids();
    assert!(check(&m, &Limits { depth: 2, ..Default::default() }).unwrap_err().reason.contains("Limit of mapping depth [2]"));
    assert!(check(&m, &Limits { total_fields: 2, ..Default::default() }).unwrap_err().reason.contains("Limit of total fields [2]"));
    assert!(check(&m, &Limits { field_name_length: 0, ..Default::default() }).unwrap_err().reason.contains("is longer than the limit"));
    let n = Mappings::parse(&json!({"properties": {"n1": {"type": "nested"}, "n2": {"type": "nested"}}}), &reg).unwrap();
    assert!(check(&n, &Limits { nested_fields: 1, ..Default::default() }).unwrap_err().reason.contains("Limit of nested fields [1]"));
    let dims = Mappings::parse(&json!({"properties": {"a": {"type": "keyword", "time_series_dimension": true}, "b": {"type": "keyword", "time_series_dimension": true}}}), &reg).unwrap();
    assert!(check(&dims, &Limits { dimension_fields: 1, ..Default::default() }).is_err());
    let s = Limits::from_settings(|k| match k {
        "index.mapping.total_fields.limit" => Some("5".into()),
        "index.mapping.total_fields.ignore_dynamic_beyond_limit" => Some("true".into()),
        _ => None,
    });
    assert_eq!(s.total_fields, 5);
    assert!(s.ignore_dynamic_beyond_limit);
    // ignore_dynamic_beyond_limit: extra dynamic fields land in _ignored.
    let mut base = Mappings::parse(&json!({"properties": {"a": {"type": "long"}}}), &reg).unwrap();
    base.assign_core_ids();
    let lk = FieldLookup::build(&base, &reg);
    let lim = Limits { total_fields: 2, ignore_dynamic_beyond_limit: true, ..Default::default() };
    let (pd, upd) = parse_with_limits(&json!({"a": 1, "b": 2, "c": 3}), "1", &base, &lk, &lim).unwrap();
    let (nm, _) = upd.unwrap();
    assert_eq!(nm.total_fields(), 2);
    assert_eq!(pd.ignored, vec!["c".to_string()]);
    let lim = Limits { total_fields: 2, ..Default::default() };
    assert!(parse_with_limits(&json!({"a": 1, "b": 2, "c": 3}), "1", &base, &lk, &lim).is_err());
    let lim = Limits { nested_objects: 1, ..Default::default() };
    let mut nb = Mappings::parse(&json!({"properties": {"n": {"type": "nested", "properties": {"x": {"type": "long"}}}}}), &reg).unwrap();
    nb.assign_core_ids();
    let nlk = FieldLookup::build(&nb, &reg);
    let e = parse_with_limits(&json!({"n": [{"x": 1}, {"x": 2}]}), "1", &nb, &nlk, &lim).err().unwrap();
    assert!(e.reason.contains("nested documents has exceeded the allowed limit of [1]"));
}

#[test]
fn field_caps_report_declared_types() {
    let f = fx(json!({"properties": {"tc": {"type": "token_count", "analyzer": "standard"}, "v": {"type": "version"}, "w": {"type": "wildcard"}}}), &[]);
    assert_eq!(f.lookup.get("tc").unwrap().declared.name(), "token_count");
    assert_eq!(f.lookup.get("tc").unwrap().ty.name(), "integer");
    assert_eq!(f.lookup.get("v").unwrap().declared.name(), "version");
    let _ = &f.ids;
    // Every new type round-trips through GET mapping.
    let reg = AnalysisRegistry::default();
    let body = json!({"properties": {
        "a": {"type": "annotated_text"}, "s": {"type": "search_as_you_type", "max_shingle_size": 4}, "tc": {"type": "token_count", "analyzer": "standard"},
        "m": {"type": "murmur3"}, "v": {"type": "version"}, "c": {"type": "completion"}, "j": {"type": "join", "relations": {"p": ["c1", "c2"]}},
        "pq": {"type": "percolator"}, "ir": {"type": "integer_range"}, "gs": {"type": "geo_shape"}, "pt": {"type": "point"}, "sh": {"type": "shape"},
        "h": {"type": "histogram"}, "amd": {"type": "aggregate_metric_double", "metrics": ["max"], "default_metric": "max"},
        "sv": {"type": "sparse_vector"}, "rf": {"type": "rank_feature"}, "rfs": {"type": "rank_features"}, "rv": {"type": "rank_vectors", "dims": 3},
        "st": {"type": "semantic_text", "inference_id": "x"}, "td": {"type": "tdigest"}, "eh": {"type": "exponential_histogram"},
        "ck": {"type": "constant_keyword", "value": "x"}, "fl": {"type": "flattened"}, "w": {"type": "wildcard"}, "dn": {"type": "date_nanos"},
        "b": {"type": "binary"}, "mot": {"type": "match_only_text"}
    }});
    let m = Mappings::parse(&body, &reg).unwrap();
    // Every type survives the round trip (ES prints some extra defaults, e.g.
    // `doc_values: false` for search_as_you_type).
    let out = m.to_json();
    for (k, v) in body["properties"].as_object().unwrap() {
        assert_eq!(out["properties"][k]["type"], v["type"], "{k}");
    }
    assert_eq!(Mappings::parse(&out, &reg).unwrap().to_json(), out);
}


#[test]
fn geo_circle_uses_distance_boundary_and_document_relations() {
    let f = fx(json!({"properties":{"p":{"type":"geo_point"}}}), &[
        json!({"p":[7.958982,55.3655542]}),
        json!({"p":[7.0,55.0]}),
        json!({"p":[[7.0,55.0],[20.0,55.0]]}),
        json!({"p":[20.0,55.0]}),
        json!({}),
    ]);
    let circle = |radius: &str, rel: &str| json!({"geo_shape":{"p":{"shape":{
        "type":"circle","coordinates":[7,55],"radius":radius},"relation":rel}}});
    let ring = json!({"bool":{"must":[circle("73204.78m","intersects"),circle("73204.77m","disjoint")]}});
    assert_eq!(sorted(search(&f,ring)),ids(&["1"]));
    assert_eq!(sorted(search(&f,circle("1km","intersects"))),ids(&["2","3"]));
    assert_eq!(sorted(search(&f,circle("1km","within"))),ids(&["2"]));
    assert_eq!(sorted(search(&f,circle("1km","disjoint"))),ids(&["1","4"]));
    let result = aggs(&f,None,json!({"d":{"geo_distance":{"field":"p","origin":[7,55],
        "unit":"km","ranges":[{"from":0,"to":1},{"from":1,"to":100}]}}})).unwrap();
    assert_eq!(result["d"]["buckets"][0]["key"],"*-1.0");
    assert_eq!(result["d"]["buckets"][0]["from"],0.0);
}

#[test]
fn geo_distance_count_only_matches_subaggregation_collection() {
    let f = fx(json!({"properties":{"p":{"type":"geo_point"}}}), &[
        json!({"p":[[0.0,0.0],[1.0,0.0],[1.0,0.0]]}),
        json!({"p":[1.0,0.0]}),
        json!({"p":[3.0,0.0]}),
        json!({}),
    ]);
    for distance_type in ["arc", "plane"] {
        for keyed in [false, true] {
            let options = json!({"field":"p","origin":[0,0],"unit":"km",
                "distance_type":distance_type,"keyed":keyed,"ranges":[
                    {"key":"near","to":200},{"key":"overlap","from":50,"to":200},
                    {"key":"far","from":200},{"key":"empty","from":1000}]});
            let plain = aggs(&f,None,json!({"d":{"geo_distance":options}})).unwrap();
            let collected = aggs(&f,None,json!({"d":{"geo_distance":options,
                "aggs":{"points":{"value_count":{"field":"p"}}}}})).unwrap();
            for (i, (key, expected)) in [("near",2),("overlap",2),("far",1),("empty",0)].iter().enumerate() {
                let bucket = |response: &Value| if keyed { response["d"]["buckets"][*key].clone() }
                    else { response["d"]["buckets"][i].clone() };
                assert_eq!(bucket(&plain)["doc_count"], *expected);
                assert_eq!(bucket(&plain)["doc_count"], bucket(&collected)["doc_count"]);
            }
        }
    }
}

#[test]
fn regex_length_is_enforced_in_engine_queries_and_query_string() {
    let f = fx(json!({"properties":{"tag":{"type":"keyword"}}}), &[]);
    let leaves = f.index.leaves();
    let mut ctx = qctx(&f, &leaves);
    ctx.max_regex_length = 2;
    assert!(parse_query(&json!({"regexp":{"tag":"😀"}}), &ctx).is_ok());
    for q in [
        json!({"regexp":{"tag":"😀a"}}),
        json!({"prefix":{"tag":{"value":"abc"}}}),
        json!({"bool":{"filter":{"regexp":{"tag":"abc"}}}}),
        json!({"query_string":{"query":"tag:/abc/"}}),
        json!({"query_string":{"query":"tag:/abc/","lenient":true}}),
    ] {
        let error = parse_query(&q, &ctx).unwrap_err();
        let mut cause = &error;
        while let Some(inner) = cause.caused_by.as_deref() { cause = inner; }
        assert_eq!(cause.kind, "illegal_argument_exception", "{q}: {error}");
        assert!(cause.reason.contains("index.max_regex_length"), "{q}: {error}");
    }
}

#[test]
fn percolate_requires_a_fallible_request_context_for_lazy_verification() {
    let f = fx(json!({"properties": {"query": {"type": "percolator"}, "tag": {"type": "keyword"}}}), &[]);
    let leaves = f.index.leaves();
    let context = qctx(&f, &leaves);
    let error = parse_query(&json!({"percolate": {"field": "query", "document": {"tag": "x"}}}), &context).unwrap_err();
    assert!(error.reason.contains("search request error context"), "{error}");
}

#[test]
fn numeric_terms_count_as_one_clause_like_a_point_set() {
    let f = fx(json!({"properties": {"n": {"type": "long"}, "k": {"type": "keyword"}}}), &[json!({"n": 3, "k": "a"})]);
    let n = (crate::query::limits::clause_limit() + 1).min(65_536);
    let values: Vec<i64> = (0..n as i64).collect();
    assert_eq!(search(&f, json!({"terms": {"n": values}})), vec!["1"]);
    // A bool with that many term clauses still counts each one, as Lucene does.
    let shoulds: Vec<Value> = (0..crate::query::limits::clause_limit() + 1).map(|i| json!({"term": {"k": format!("v{i}")}})).collect();
    let error = try_search(&f, json!({"bool": {"should": shoulds}})).unwrap_err();
    assert_eq!(error.kind, "too_many_nested_clauses");
}

#[test]
fn only_builder_made_point_sets_count_as_one_clause() {
    let f = fx(json!({"properties": {"n": {"type": "long"}, "d": {"type": "date"}, "t": {"type": "text"}}}), &[]);
    let leaves = f.index.leaves();
    let ctx = qctx(&f, &leaves);
    let clauses = |q: Value| parse_query(&q, &ctx).unwrap().clause_count(&|_| 1);
    let values: Vec<i64> = (0..50).collect();
    // `terms` on a numeric field is one point-set query in ES.
    assert_eq!(clauses(json!({"terms": {"n": values}})), 1);
    // `date` uses the default per-value disjunction.
    assert_eq!(clauses(json!({"terms": {"d": values}})), 50);
    // A user-written disjunction under `constant_score` is not a point set,
    // whatever its shape.
    let shoulds: Vec<Value> = values.iter().map(|v| json!({"term": {"n": v}})).collect();
    assert_eq!(clauses(json!({"constant_score": {"filter": {"bool": {"should": shoulds}}}})), 50);
    let words: Vec<String> = (0..50).map(|i| format!("w{i}")).collect();
    assert_eq!(clauses(json!({"constant_score": {"filter": {"match": {"t": words.join(" ")}}}})), 50);
}

// ---------------------------------------------------------------------------
// geo_shape fields of points: the segment point tree vs exact relations.

/// Deterministic pseudo-random numbers in `[0, 1)`.
struct ShapeLcg(u64);

impl ShapeLcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    /// A coordinate with OSM's 7 decimals.
    fn coord(&mut self, lo: f64, hi: f64) -> f64 {
        (self.range(lo, hi) * 1e7).round() / 1e7
    }
}

fn shape_point_docs(n: usize, seed: u64) -> Vec<Vec<String>> {
    let special = [(-0.1, 49.0), (-0.1, 55.0), (5.0, 48.0), (14.5, 54.5), (7.0, 61.0), (15.0, 49.0), (2.0, 55.0), (3.5, 50.25), (2.3, 48.85), (180.0, 0.0), (-180.0, 0.0), (179.95, 5.0), (-179.95, -5.0), (170.0, 10.0)];
    let mut r = ShapeLcg(seed);
    let mut docs = vec![
        vec!["MULTIPOINT (3.5 50.25, -0.1 49.0)".to_string()],
        vec!["POINT (3.5 50.25)".to_string(), "POINT (-0.1 49.0)".to_string()],
        vec!["POINT (-0.1 49.0)".to_string(), "POLYGON ((20 20, 21 20, 21 21, 20 21, 20 20))".to_string()],
    ];
    while docs.len() < n {
        let pt = |r: &mut ShapeLcg| -> String {
            let (x, y) = match (r.next() * 10.0) as u32 {
                0 => special[(r.next() * special.len() as f64) as usize],
                1 => (r.coord(2.2, 2.4), r.coord(48.75, 48.95)),
                2 => (r.coord(-180.0, 180.0), r.coord(-15.0, 15.0)),
                _ => (r.coord(-5.0, 20.0), r.coord(45.0, 65.0)),
            };
            format!("POINT ({x} {y})")
        };
        let doc = match (r.next() * 20.0) as u32 {
            0..=12 => vec![pt(&mut r)],
            13 | 14 => (0..2 + (r.next() * 2.0) as usize).map(|_| pt(&mut r)).collect(),
            15 => {
                let ps: Vec<String> = (0..2 + (r.next() * 3.0) as usize).map(|_| pt(&mut r)[7..].trim_end_matches(')').to_string()).collect();
                vec![format!("MULTIPOINT ({})", ps.join(", "))]
            }
            16 => {
                let (x, y) = (r.coord(-5.0, 19.0), r.coord(45.0, 64.0));
                vec![pt(&mut r), format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.5, x + 0.5, y + 0.5, y + 0.5)]
            }
            17 => vec![format!("LINESTRING ({} {}, {} {})", r.coord(-5.0, 20.0), r.coord(45.0, 65.0), r.coord(-5.0, 20.0), r.coord(45.0, 65.0))],
            18 => vec![],
            _ => {
                let (x, y) = (r.coord(-5.0, 19.0), r.coord(45.0, 64.0));
                vec![format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.2, x + 0.2, y + 0.2, y + 0.2)]
            }
        };
        docs.push(doc);
    }
    docs
}

fn shape_point_fixture(docs: &[Vec<String>]) -> Fx {
    let json_docs: Vec<Value> = docs
        .iter()
        .map(|v| match v.len() {
            0 => json!({}),
            1 => json!({"g": v[0]}),
            _ => json!({"g": v}),
        })
        .collect();
    fx(json!({"properties": {"g": {"type": "geo_shape"}}}), &json_docs)
}

/// The relation as ES defines it, one doc at a time: the doc's values
/// tessellated (as indexed), the query built as `LatLonShape` builds it, and
/// the doc related triangle by triangle with the tree's leaf tests (`dv`:
/// through its shape doc value, `LatLonShapeDocValuesQuery`). `Err`: the
/// query is refused.
fn shape_reference(vals: &[String], query: &Value, rel: rustic_core::fieldtypes::shape::SpatialRelation, dv: bool) -> std::result::Result<bool, ()> {
    use rustic_core::fieldtypes::geo2d::{query_geometries, ComponentLeaf, RectLeaf, ShapeQuery};
    use rustic_core::fieldtypes::{shape, shape_bkd::Matcher, shape_dv::ShapeValue, tessellator};
    let geoms = query_geometries(&shape::normalize_geo(&shape::parse_query_shape(query).map_err(|_| ())?), rel == shape::SpatialRelation::Within).map_err(|_| ())?;
    if vals.is_empty() {
        return Ok(false);
    }
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    let mut tris = Vec::new();
    let mut calc = shape::CentroidCalc::default();
    for v in vals {
        let g = shape::parse(&json!(v), &opts).unwrap();
        tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
        calc.add(&g);
    }
    if tris.is_empty() {
        return Ok(false);
    }
    let bytes = rustic_core::fieldtypes::shape_field::doc_value(&tris, &calc).unwrap();
    fn eval(q: &ShapeQuery, tris: &[[i32; 7]], bytes: &[u8], dv: bool) -> bool {
        match q {
            ShapeQuery::None => false,
            ShapeQuery::And(qs) => qs.iter().all(|q| eval(q, tris, bytes, dv)),
            ShapeQuery::Component(c, r) | ShapeQuery::Rect(_, c, r) if dv => ShapeValue::new(bytes).relate_query(c, *r).unwrap(),
            ShapeQuery::Component(c, r) => {
                let leaf = ComponentLeaf { component: c, rel: *r };
                let mut m = Matcher::new(&leaf, 1);
                m.doc(0, tris);
                m.finish().unwrap()[0] & 1 == 1
            }
            ShapeQuery::Rect(e, _, r) => {
                let leaf = RectLeaf { rect: e, rel: *r };
                let mut m = Matcher::new(&leaf, 1);
                m.doc(0, tris);
                m.finish().unwrap()[0] & 1 == 1
            }
        }
    }
    Ok(eval(&ShapeQuery::new(&geoms, crate::fieldtypes::geo_shape_query::query_relation(rel)), &tris, &bytes, dv))
}

fn shape_test_queries() -> Vec<Value> {
    vec![
        json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]}),
        json!({"type": "envelope", "coordinates": [[-0.1, 61.0], [15.0, 48.0]]}),
        json!({"type": "envelope", "coordinates": [[170.0, 10.0], [-170.0, -10.0]]}),
        json!({"type": "polygon", "coordinates": [[[170.0, -10.0], [-170.0, -10.0], [-170.0, 10.0], [170.0, 10.0], [170.0, -10.0]]]}),
        json!({"type": "polygon", "coordinates": [[[0, 50], [10, 50], [10, 60], [0, 60], [0, 50]], [[2, 52], [8, 52], [8, 58], [2, 58], [2, 52]]]}),
        json!("BBOX (2.2, 2.4, 48.95, 48.75)"),
        json!("POINT (-0.1 49.0)"),
        json!("MULTIPOINT ((3.5 50.25), (-0.1 49.0))"),
        json!("LINESTRING (-0.1 49.0, 15.0 49.0)"),
    ]
}

fn shape_complex_docs(n: usize, seed: u64) -> Vec<Vec<String>> {
    let mut r = ShapeLcg(seed);
    (0..n)
        .map(|i| {
            let (x, y) = match (r.next() * 4.0) as u32 {
                0 => (r.coord(2.0, 12.0), r.coord(50.0, 58.0)),
                1 => (r.coord(-5.0, 20.0), r.coord(45.0, 65.0)),
                2 => (r.coord(-170.0, 170.0), r.coord(-60.0, 60.0)),
                _ => (r.coord(-2.0, 2.0), r.coord(-5.0, 5.0)),
            };
            let d = r.coord(0.001, 2.0);
            let (hx, hy, hd) = (x + d / 4.0, y + d / 4.0, d / 4.0);
            let one = |k: u32| -> String {
                match k {
                    0 => format!("LINESTRING ({x} {y}, {} {})", x + d, y + d / 2.0),
                    1 => format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + d, x + d, y + d, y + d),
                    2 => format!(
                        "POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}), ({hx} {hy}, {hx} {}, {} {}, {} {hy}, {hx} {hy}))",
                        x + d,
                        x + d,
                        y + d,
                        y + d,
                        hy + hd,
                        hx + hd,
                        hy + hd,
                        hx + hd
                    ),
                    3 => format!("MULTILINESTRING (({x} {y}, {} {}), ({} {y}, {} {}))", x + d, y + d, x - d, x, y - d),
                    4 => format!("POLYGON ((175 {y}, -175 {y}, -175 {}, 175 {}, 175 {y}))", y + d, y + d),
                    _ => format!("GEOMETRYCOLLECTION (POINT ({x} {y}), POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y})))", x + d, x + d, y + d, y + d),
                }
            };
            let k = (r.next() * 6.0) as u32;
            if i % 23 == 5 {
                vec![]
            } else if i % 17 == 0 {
                vec![one(k), one((k + 1) % 6)]
            } else {
                vec![one(k)]
            }
        })
        .collect()
}

const SHAPE_RELS: [(&str, rustic_core::fieldtypes::shape::SpatialRelation); 4] = [
    ("intersects", rustic_core::fieldtypes::shape::SpatialRelation::Intersects),
    ("within", rustic_core::fieldtypes::shape::SpatialRelation::Within),
    ("disjoint", rustic_core::fieldtypes::shape::SpatialRelation::Disjoint),
    ("contains", rustic_core::fieldtypes::shape::SpatialRelation::Contains),
];

fn rally_queries() -> Vec<Value> {
    let mut queries = shape_test_queries();
    queries.extend([
        json!({"type": "Polygon", "coordinates": [[[12.5706, 55.45074], [12.04631, 55.70469], [12.06618, 55.7991], [12.68098, 55.93474], [13.48894, 55.88091], [13.60262, 55.68478], [13.44479, 55.32661], [12.64235, 55.32723], [12.5706, 55.45074]]]}),
        json!({"type": "multipolygon", "coordinates": [[[[2.0, 50.0], [6.0, 50.0], [6.0, 54.0], [2.0, 54.0], [2.0, 50.0]]], [[[170.0, -10.0], [180.0, -10.0], [180.0, 10.0], [170.0, 10.0], [170.0, -10.0]]]]}),
        json!({"type": "polygon", "coordinates": [[[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0], [-3.0, -6.0]]]}),
        // Small enough for some docs to contain it.
        json!({"type": "polygon", "coordinates": [[[5.0, 52.0], [5.01, 52.0], [5.01, 52.01], [5.0, 52.01], [5.0, 52.0]]]}),
        json!("POINT (5.005 52.005)"),
        json!("POINT (2.0 51.0)"),
    ]);
    queries
}

/// Every relation of the Rally geoshape queries (and boundary, dateline,
/// hole, point and line queries) over lines, polygons with holes,
/// multi-lines, collections, dateline polygons, points and multi-valued
/// docs: the memtable (triangle by triangle) and the sealed segments' trees
/// (their walk) give ES's relation of each doc; the same field without an
/// index (`index: false`) answers from the doc values.
#[test]
fn geo_shape_relations_match_per_doc_relations() {
    let mut docs = shape_complex_docs(3000, 33);
    docs.extend(shape_point_docs(1500, 42));
    docs.push(vec!["MULTIPOLYGON (((1 50, 3 50, 3 52, 1 52, 1 50)), ((10 55, 12 55, 12 57, 10 57, 10 55), (10.5 55.5, 11 55.5, 11 56, 10.5 56, 10.5 55.5)))".to_string()]);
    docs.push(vec!["POLYGON ((-0.1 49, 5 48, 15 49, 14 60, -0.1 61, -0.1 49))".to_string()]);
    let mut f = shape_point_fixture(&docs);
    let json_docs: Vec<Value> = docs.iter().map(|v| match v.len() {
        0 => json!({}),
        1 => json!({"g": v[0]}),
        _ => json!({"g": v}),
    }).collect();
    let mut dvf = fx(json!({"properties": {"g": {"type": "geo_shape", "index": false}}}), &json_docs);
    dvf.index.refresh().unwrap();
    let queries = rally_queries();
    for sealed in [false, true] {
        if sealed {
            f.index.refresh().unwrap();
            let id = f.lookup.get("g").unwrap().core_id.unwrap();
            assert!(f.index.leaves().iter().any(|l| matches!(l, LeafReader::Segment(s) if s.shape_data(id).is_some_and(|d| !d.tree.is_empty()))));
        }
        for q in &queries {
            for (name, rel) in SHAPE_RELS {
                let body = json!({"geo_shape": {"g": {"shape": q, "relation": name}}});
                let got = try_search(&f, body.clone()).map(|v| sorted(v.into_iter().map(|(id, _)| id).collect())).map_err(|_| ());
                let want: std::result::Result<Vec<String>, ()> = (0..docs.len()).map(|i| shape_reference(&docs[i], q, rel, false).map(|b| b.then(|| (i + 1).to_string()))).collect::<std::result::Result<Vec<_>, ()>>().map(|v| sorted(v.into_iter().flatten().collect()));
                assert_eq!(got, want, "{body} sealed={sealed}");
                if sealed {
                    let got = try_search(&dvf, body.clone()).map(|v| sorted(v.into_iter().map(|(id, _)| id).collect())).map_err(|_| ());
                    let want: std::result::Result<Vec<String>, ()> = (0..docs.len()).map(|i| shape_reference(&docs[i], q, rel, true).map(|b| b.then(|| (i + 1).to_string()))).collect::<std::result::Result<Vec<_>, ()>>().map(|v| sorted(v.into_iter().flatten().collect()));
                    assert_eq!(got, want, "doc values {body}");
                }
            }
        }
    }
    // Point and grid queries on a geo_shape field are geo_shape queries.
    for body in [
        json!({"geo_bounding_box": {"g": {"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"}}}),
        json!({"geo_grid": {"g": {"geohash": "u0"}}}),
        json!({"geo_grid": {"g": {"geotile": "6/32/22"}}}),
        json!({"geo_grid": {"g": {"geohex": "811fbffffffffff"}}}),
    ] {
        assert!(try_search(&f, body.clone()).is_ok(), "{body}");
    }
}

/// Counts from the trees (`exact_count`) equal the matching docs, and
/// size-10 searches with bounded totals give the first matches by doc order.
#[test]
fn geo_shape_counts_and_bounded_searches() {
    let mut docs = shape_complex_docs(2500, 77);
    let pts = geo_point_docs(1500, false, 61);
    docs.extend(pts.iter().map(|p| p.iter().map(|&(x, y)| format!("POINT ({x} {y})")).collect::<Vec<_>>()));
    let mut f = shape_point_fixture(&docs);
    f.index.refresh().unwrap();
    let leaves = f.index.leaves();
    let ctx = qctx(&f, &leaves);
    for q in rally_queries().into_iter().take(8) {
        for (name, _) in SHAPE_RELS {
            let body = json!({"geo_shape": {"g": {"shape": q, "relation": name}}});
            let Ok(query) = parse_query(&body, &ctx) else { continue };
            let hits = super::join::run(&query, &leaves).unwrap();
            for (li, leaf) in leaves.iter().enumerate() {
                let n = hits.iter().filter(|(l, _, _)| *l == li).count() as u64;
                assert_eq!(query.exact_count(leaf), Some(n), "{body}");
                // The bounded count (a size-limited search's total), at
                // limits below, at and above the count.
                // Each way: as chosen, the doc-order scan, the tree.
                for mode in [None, Some(true), Some(false)] {
                    super::geo_shape_query::TEST_COUNT_MODE.with(|c| c.set(mode));
                    for limit in [1u64, 37, n, n + 1, 10_000] {
                        if let Some(c) = query.count_at_least(leaf, limit) {
                            assert_eq!(c, n.min(limit), "{body} limit {limit} mode {mode:?}");
                        }
                    }
                }
                super::geo_shape_query::TEST_COUNT_MODE.with(|c| c.set(None));
            }
            let mut all: Vec<usize> = search(&f, body.clone()).iter().map(|s| s.parse().unwrap()).collect();
            all.sort_unstable();
            // The doc-order scan alone, switching to the tree early, and
            // at once.
            for budget in [None, Some(64), Some(1)] {
            super::geo_shape_query::TEST_SCAN_BUDGET.with(|c| c.set(budget));
            for (size, limit) in [(10usize, 10_000u64), (10, 50), (1, 1), (0, 100)] {
                let (ids, total, gte) = bounded_search(&f, body.clone(), size, limit);
                assert_eq!(ids.iter().map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>(), all.iter().copied().take(size).collect::<Vec<_>>(), "{body} size={size} limit={limit}");
                let n = all.len() as u64;
                assert_eq!((total, gte), if n > limit { (limit, true) } else { (n, false) }, "{body} size={size} limit={limit} budget={budget:?}");
            }
            }
            super::geo_shape_query::TEST_SCAN_BUDGET.with(|c| c.set(None));
        }
    }
}

/// Segments of single points (Rally `geopointshape`): counts come from
/// the tree's node counts (no doc ids read) and equal the matching docs;
/// size-10 searches give the first matches in doc order.
#[test]
fn geo_shape_point_counts_from_node_counts() {
    let docs: Vec<Vec<String>> = geo_point_docs(3000, false, 17).into_iter().filter(|p| p.len() == 1).map(|p| vec![format!("POINT ({} {})", p[0].0, p[0].1)]).collect();
    let mut f = shape_point_fixture(&docs);
    f.index.refresh().unwrap();
    let leaves = f.index.leaves();
    let ctx = qctx(&f, &leaves);
    let id = f.lookup.get("g").unwrap().core_id.unwrap();
    assert!(leaves.iter().any(|l| matches!(l, LeafReader::Segment(s) if s.shape_data(id).is_some_and(|d| !d.tree.is_empty() && d.tree.iter().all(|p| p.one_triangle_per_doc())))));
    let mut checked = 0;
    for q in rally_queries() {
        for (name, _) in SHAPE_RELS {
            let body = json!({"geo_shape": {"g": {"shape": q, "relation": name}}});
            let Ok(query) = parse_query(&body, &ctx) else { continue };
            let hits = super::join::run(&query, &leaves).unwrap();
            for (li, leaf) in leaves.iter().enumerate() {
                let n = hits.iter().filter(|(l, _, _)| *l == li).count() as u64;
                assert_eq!(query.exact_count(leaf), Some(n), "{body}");
                checked += usize::from(n > 0);
            }
            let mut all: Vec<usize> = search(&f, body.clone()).iter().map(|s| s.parse().unwrap()).collect();
            all.sort_unstable();
            let (ids, _, _) = bounded_search(&f, body.clone(), 10, 100);
            assert_eq!(ids.iter().map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>(), all.iter().copied().take(10).collect::<Vec<_>>(), "{body}");
        }
    }
    assert!(checked > 5);
}

/// `geo_bounds` / `geo_centroid` over `geo_shape` read each doc's shape doc
/// value: equal to ES's aggregation (`GeoShapeBoundsAggregator`,
/// `GeoShapeCentroidAggregator`) of the values built independently from
/// each doc's tessellation and centroid, on the memtable and sealed
/// segments.
#[test]
fn geo_shape_bounds_centroid_read_doc_values() {
    use rustic_core::fieldtypes::{shape, shape_dv::ShapeValue, tessellator};
    use rustic_core::geo::{decode_lat, decode_lon};
    let docs = shape_complex_docs(900, 4);
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    let values: Vec<Option<Vec<u8>>> = docs
        .iter()
        .map(|vals| {
            if vals.is_empty() {
                return None;
            }
            let mut tris = Vec::new();
            let mut calc = shape::CentroidCalc::default();
            for v in vals {
                let g = shape::parse(&json!(v), &opts).unwrap();
                tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
                calc.add(&g);
            }
            Some(rustic_core::fieldtypes::shape_field::doc_value(&tris, &calc).unwrap())
        })
        .collect();
    let (mut top, mut bottom, mut pl, mut pr, mut nl, mut nr) = (f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for b in values.iter().flatten() {
        let e = ShapeValue::new(b).extent().unwrap();
        top = top.max(decode_lat(e.top));
        bottom = bottom.min(decode_lat(e.bottom));
        if e.has_neg() {
            nl = nl.min(decode_lon(e.neg_left));
            nr = nr.max(decode_lon(e.neg_right));
        }
        if e.has_pos() {
            pl = pl.min(decode_lon(e.pos_left));
            pr = pr.max(decode_lon(e.pos_right));
        }
    }
    let (wl, wr) = if (pr - nl) <= (180.0 - pl) - (-180.0 - nr) { (nl, pr) } else { (pl, nr) };
    let with: Vec<ShapeValue<'_>> = values.iter().flatten().map(|b| ShapeValue::new(b)).collect();
    let dim = with.iter().map(|v| v.dimension()).max().unwrap();
    let (mut slat, mut slon, mut sw) = (0.0, 0.0, 0.0);
    for v in with.iter().filter(|v| v.dimension() == dim) {
        let (x, y) = v.centroid();
        slat += v.weight() * y;
        slon += v.weight() * x;
        sw += v.weight();
    }
    let mut f = shape_point_fixture(&docs);
    let spec = json!({"b": {"geo_bounds": {"field": "g"}}, "bn": {"geo_bounds": {"field": "g", "wrap_longitude": false}}, "c": {"geo_centroid": {"field": "g"}}});
    for sealed in [false, true] {
        if sealed {
            f.index.refresh().unwrap();
        }
        let a = aggs(&f, None, spec.clone()).unwrap();
        let num = |v: &Value| v.as_f64().unwrap();
        for (k, (left, right)) in [("b", (wl, wr)), ("bn", (nl, pr))] {
            let b = &a[k]["bounds"];
            assert_eq!(num(&b["top_left"]["lat"]), top, "{k} {a}");
            assert_eq!(num(&b["bottom_right"]["lat"]), bottom, "{k} {a}");
            assert_eq!(num(&b["top_left"]["lon"]), left, "{k} {a}");
            assert_eq!(num(&b["bottom_right"]["lon"]), right, "{k} {a}");
        }
        assert_eq!(a["c"]["count"], json!(with.len()), "{a}");
        let (lat, lon) = (num(&a["c"]["location"]["lat"]), num(&a["c"]["location"]["lon"]));
        assert!((lat - slat / sw).abs() < 1e-9 && (lon - slon / sw).abs() < 1e-9, "{a} want {} {}", slat / sw, slon / sw);
    }
}

/// Grid aggregations over `geo_shape` (ES's tilers on the doc values): the
/// geotile and geohash cells of every doc equal a brute-force scan of the
/// cells over its bounding box relating each (the tilers' recursive
/// descent prunes, never changes, the cells), with and without bounds; the
/// same buckets on the memtable and sealed segments for every grid.
#[test]
fn geo_shape_grid_aggs_match_brute_force_cells() {
    let mut docs = shape_complex_docs(600, 19);
    docs.extend(shape_point_docs(600, 7));
    let mut f = shape_point_fixture(&docs);
    let bounds = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
    let specs = [
        json!({"a": {"geohash_grid": {"field": "g", "precision": 5, "bounds": bounds}}}),
        json!({"a": {"geotile_grid": {"field": "g", "precision": 13, "bounds": bounds}}}),
        json!({"a": {"geohex_grid": {"field": "g", "precision": 6, "bounds": bounds}}}),
        json!({"a": {"geohash_grid": {"field": "g", "precision": 3}}}),
        json!({"a": {"geotile_grid": {"field": "g", "precision": 6}}}),
        json!({"a": {"geohex_grid": {"field": "g", "precision": 2}}}),
    ];
    let memtable: Vec<Value> = specs.iter().map(|s| aggs(&f, None, s.clone()).unwrap()).collect();
    f.index.refresh().unwrap();
    for (spec, m) in specs.iter().zip(&memtable) {
        let a = aggs(&f, None, spec.clone()).unwrap();
        assert_eq!(&a, m, "{spec}");
        assert!(!a["a"]["buckets"].as_array().unwrap().is_empty(), "{spec}");
    }
    // Unbounded geotiles and geohashes against the brute-force cells.
    use rustic_core::fieldtypes::{shape, shape_dv::ShapeValue, tessellator};
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    for (grid, precision) in [("geotile_grid", 6u32), ("geohash_grid", 3)] {
        let mut want: std::collections::BTreeMap<String, u64> = Default::default();
        for vals in &docs {
            if vals.is_empty() {
                continue;
            }
            let mut tris = Vec::new();
            let mut calc = shape::CentroidCalc::default();
            for v in vals {
                let g = shape::parse(&json!(v), &opts).unwrap();
                tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
                calc.add(&g);
            }
            let bytes = rustic_core::fieldtypes::shape_field::doc_value(&tris, &calc).unwrap();
            let v = ShapeValue::new(&bytes);
            let mut cells = std::collections::BTreeSet::new();
            crate::aggs::shape_tiler::brute_force_cells(grid, precision, None, &v, &mut cells);
            for c in cells {
                *want.entry(c).or_default() += 1;
            }
        }
        let a = aggs(&f, None, json!({"a": {grid: {"field": "g", "precision": precision, "size": 100000}}})).unwrap();
        let got: std::collections::BTreeMap<String, u64> = a["a"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap())).collect();
        assert_eq!(got, want, "{grid}");
    }
}

/// `geo_point` docs: most in western Europe (the Rally polygon's area),
/// points on the test polygon's vertices and edges, on the dateline, at the
/// poles, docs with two or three points (when `multi`) and docs without the
/// field. `(lon, lat)` per point.
fn geo_point_docs(n: usize, multi: bool, seed: u64) -> Vec<Vec<(f64, f64)>> {
    let special = [(-0.1, 49.0), (-0.1, 55.0), (5.0, 48.0), (15.0, 49.0), (14.0, 60.0), (-0.1, 61.0), (14.5, 54.5), (180.0, 0.0), (-180.0, 0.0), (179.95, 5.0), (-179.95, -5.0), (0.0, 90.0), (45.0, -90.0), (7.0, 55.0), (7.958982, 55.3655542)];
    let mut r = ShapeLcg(seed);
    (0..n)
        .map(|i| {
            if i % 53 == 5 {
                return Vec::new();
            }
            let k = if multi && i % 7 == 0 { 2 + i % 2 } else { 1 };
            (0..k)
                .map(|_| match (r.next() * 12.0) as u32 {
                    0 => special[(r.next() * special.len() as f64) as usize],
                    1 => (r.coord(-180.0, 180.0), r.coord(-90.0, 90.0)),
                    2 => (if r.next() < 0.5 { r.coord(170.0, 180.0) } else { r.coord(-180.0, -170.0) }, r.coord(-15.0, 15.0)),
                    3 => (r.coord(-180.0, 180.0), if r.next() < 0.5 { r.coord(85.0, 90.0) } else { r.coord(-90.0, -85.0) }),
                    4 => (r.coord(2.2, 2.4), r.coord(48.75, 48.95)),
                    _ => (r.coord(-5.0, 20.0), r.coord(45.0, 65.0)),
                })
                .collect()
        })
        .collect()
}

fn geo_point_fixture(docs: &[Vec<(f64, f64)>]) -> Fx {
    let json_docs: Vec<Value> = docs
        .iter()
        .map(|v| match v.len() {
            0 => json!({}),
            1 => json!({"g": [v[0].0, v[0].1]}),
            _ => json!({"g": v.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()}),
        })
        .collect();
    fx(json!({"properties": {"g": {"type": "geo_point"}}}), &json_docs)
}

/// The relation of a doc's (quantized) points to the normalized, snapped
/// query shape: ES's per-document definition.
fn geo_point_reference(pts: &[(f64, f64)], query: &Value, rel: rustic_core::fieldtypes::shape::SpatialRelation) -> bool {
    use rustic_core::fieldtypes::shape;
    if pts.is_empty() {
        return false;
    }
    let dp = shape::Prims { points: pts.to_vec(), ..Default::default() }.quantized();
    let qp = shape::Prims::of(&shape::normalize_geo(&shape::parse_query_shape(query).unwrap())).quantized();
    shape::relate(&dp, &qp, rel)
}

/// Ids of the hits of a core search with `size` and a bounded total, and
/// the total: what the REST search collects (doc-order scan when it may
/// stop early).
fn bounded_search(f: &Fx, q: Value, size: usize, limit: u64) -> (Vec<String>, u64, bool) {
    use rustic_core::query::{SearchOptions, Searcher, TotalHitsRelation, TrackTotalHits};
    let leaves = f.index.leaves();
    let ctx = qctx(f, &leaves);
    let query = parse_query(&q, &ctx).unwrap();
    let td = Searcher::new(f.index.leaves()).search(&query, &SearchOptions { top_k: size, track_total_hits: TrackTotalHits::UpTo(limit), ..Default::default() });
    let ids = td.hits.iter().map(|h| leaves[h.leaf].keyword_value(ID_FIELD, h.doc).unwrap_or_default()).collect();
    (ids, td.total_hits, td.total_hits_relation == TotalHitsRelation::Gte)
}

/// `geo_shape` intersects / within / disjoint / contains over a
/// `geo_point` field through the predicate walks: the hits of ES's
/// per-document relation, on single- and multi-valued fields, memtable
/// and sealed segments, with dateline and pole points, plus the first
/// hits and bounded totals of a size-10 search.
#[test]
fn geo_point_shape_relations_match_brute_force() {
    use rustic_core::fieldtypes::shape::SpatialRelation;
    let rels = [("intersects", SpatialRelation::Intersects), ("within", SpatialRelation::Within), ("disjoint", SpatialRelation::Disjoint), ("contains", SpatialRelation::Contains)];
    for multi in [false, true] {
        let docs = geo_point_docs(3000, multi, 17);
        let mut f = geo_point_fixture(&docs);
        for sealed in [false, true] {
            if sealed {
                f.index.refresh().unwrap();
            }
            let mut queries = shape_test_queries();
            queries.push(json!({"type": "polygon", "coordinates": [[[-180.0, 80.0], [180.0, 80.0], [180.0, 90.0], [-180.0, 90.0], [-180.0, 80.0]]]}));
            queries.push(json!({"type": "envelope", "coordinates": [[-180.0, -85.0], [180.0, -90.0]]}));
            for q in queries {
                for (name, rel) in rels {
                    let body = json!({"geo_shape": {"g": {"shape": q, "relation": name}}});
                    let want: Vec<String> = sorted((0..docs.len()).filter(|&i| geo_point_reference(&docs[i], &q, rel)).map(|i| (i + 1).to_string()).collect());
                    assert_eq!(sorted(search(&f, body.clone())), want, "{body} multi={multi} sealed={sealed}");
                    // Size-10 searches return the first matches by doc order
                    // (uniform scores), the tie order of ids being insertion.
                    let order = |ids: &[String]| ids.iter().map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>();
                    let mut first = order(&want);
                    first.sort_unstable();
                    for (size, limit) in [(10usize, 10_000u64), (10, 100), (0, 50)] {
                        let (ids, total, gte) = bounded_search(&f, body.clone(), size, limit);
                        assert_eq!(order(&ids), first.iter().copied().take(size).collect::<Vec<_>>(), "{body} size={size} multi={multi} sealed={sealed}");
                        let n = want.len() as u64;
                        assert_eq!((total, gte), if n > limit { (limit, true) } else { (n, false) }, "{body} limit={limit}");
                    }
                }
            }
        }
    }
}

/// `geo_grid` geohex on `geo_point`: the docs with a point whose H3 cell is
/// the query's (the Rally res-1 cell, a dateline cell, a polar cell, a
/// pentagon's child), memtable and sealed, single and multi-valued, and
/// the first hits / bounded totals of size-10 searches.
#[test]
fn geo_point_geohex_query_matches_cell_membership() {
    let cells = ["811fbffffffffff", "8001fffffffffff", "81f2bffffffffff", "830800fffffffff", "8f1fb46622d8591"];
    let mut extra: Vec<String> = Vec::new();
    for (lat, lon, res) in [(0.0, 180.0, 2u8), (89.9, 0.0, 1), (-89.9, 0.0, 0), (55.0, 7.0, 3), (48.85, 2.3, 6)] {
        extra.push(crate::aggs::geo_hex::point_cell(lat, lon, res as u32).unwrap());
    }
    for multi in [false, true] {
        let docs = geo_point_docs(3000, multi, 23);
        let mut f = geo_point_fixture(&docs);
        for sealed in [false, true] {
            if sealed {
                f.index.refresh().unwrap();
            }
            for cell in cells.iter().map(|s| s.to_string()).chain(extra.iter().cloned()) {
                let Some(c) = crate::aggs::geo_hex::parse_cell(&cell) else { continue };
                let res = crate::h3::resolution(c) as u32;
                let want: Vec<String> = sorted((0..docs.len()).filter(|&i| docs[i].iter().any(|&(lon, lat)| {
                    let (lat, lon) = rustic_core::geo::quantize(lat, lon);
                    crate::aggs::geo_hex::point_cell_id(lat, lon, res) == Some(c)
                })).map(|i| (i + 1).to_string()).collect());
                let body = json!({"geo_grid": {"g": {"geohex": cell}}});
                let got = try_search(&f, body.clone()).unwrap();
                assert!(got.iter().all(|(_, s)| *s == 1.0), "{body}");
                assert_eq!(sorted(got.into_iter().map(|(id, _)| id).collect()), want, "{body} multi={multi} sealed={sealed}");
                let mut first: Vec<usize> = want.iter().map(|s| s.parse().unwrap()).collect();
                first.sort_unstable();
                for (size, limit) in [(10usize, 10_000u64), (10, 20)] {
                    let (ids, total, gte) = bounded_search(&f, body.clone(), size, limit);
                    assert_eq!(ids.iter().map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>(), first.iter().copied().take(size).collect::<Vec<_>>(), "{body}");
                    let n = want.len() as u64;
                    assert_eq!((total, gte), if n > limit { (limit, true) } else { (n, false) }, "{body}");
                }
            }
        }
    }
}

/// `value_count` on a `geo_point` (Rally `distanceFilterCount*`): the
/// number of point values of the matching docs, `missing` counting docs
/// without one, over match_all and filtered doc sets.
#[test]
fn geo_point_value_count_matches_brute_force() {
    for multi in [false, true] {
        let docs = geo_point_docs(2500, multi, 31);
        let mut f = geo_point_fixture(&docs);
        for sealed in [false, true] {
            if sealed {
                f.index.refresh().unwrap();
            }
            let circle = json!({"geo_distance": {"distance": "300km", "g": [7.0, 55.0]}});
            for query in [None, Some(circle.clone())] {
                let selected: Vec<usize> = match &query {
                    None => (0..docs.len()).collect(),
                    Some(q) => search(&f, q.clone()).iter().map(|s| s.parse::<usize>().unwrap() - 1).collect(),
                };
                let points: u64 = selected.iter().map(|&i| docs[i].len() as u64).sum();
                let empty = selected.iter().filter(|&&i| docs[i].is_empty()).count() as u64;
                let r = aggs(&f, query.clone(), json!({"c": {"value_count": {"field": "g"}}, "m": {"value_count": {"field": "g", "missing": "0,0"}}})).unwrap();
                assert_eq!(r["c"]["value"], json!(points), "{query:?} multi={multi} sealed={sealed}");
                assert_eq!(r["m"]["value"], json!(points + empty), "{query:?} multi={multi} sealed={sealed}");
            }
        }
    }
}

/// `geo_centroid` / `geo_bounds` on `geo_point` (Rally `centroid`,
/// `polygon-centroid`): the mean of the matching docs' quantized points
/// (compensated sums, so within rounding of any summation order) and their
/// bounds, over match_all and a polygon, with dateline and pole points.
#[test]
fn geo_point_centroid_and_bounds_match_brute_force() {
    let poly = json!({"geo_shape": {"g": {"shape": {"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]}}}});
    for multi in [false, true] {
        let docs = geo_point_docs(2500, multi, 37);
        let mut f = geo_point_fixture(&docs);
        for sealed in [false, true] {
            if sealed {
                f.index.refresh().unwrap();
            }
            for query in [None, Some(poly.clone())] {
                let selected: Vec<usize> = match &query {
                    None => (0..docs.len()).collect(),
                    Some(q) => search(&f, q.clone()).iter().map(|s| s.parse::<usize>().unwrap() - 1).collect(),
                };
                let pts: Vec<(f64, f64)> = selected.iter().flat_map(|&i| docs[i].iter().map(|&(lon, lat)| rustic_core::geo::quantize(lat, lon))).collect();
                let r = aggs(&f, query.clone(), json!({"c": {"geo_centroid": {"field": "g"}}, "b": {"geo_bounds": {"field": "g", "wrap_longitude": false}}})).unwrap();
                assert_eq!(r["c"]["count"], json!(pts.len()), "{query:?}");
                let n = pts.len() as f64;
                let lat = pts.iter().map(|p| p.0).sum::<f64>() / n;
                let lon = pts.iter().map(|p| p.1).sum::<f64>() / n;
                let got = (r["c"]["location"]["lat"].as_f64().unwrap(), r["c"]["location"]["lon"].as_f64().unwrap());
                assert!((got.0 - lat).abs() < 1e-9 && (got.1 - lon).abs() < 1e-9, "{query:?} {got:?} vs {lat} {lon}");
                let top = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                let bottom = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let left = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                let right = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(r["b"]["bounds"]["top_left"]["lat"].as_f64(), Some(top), "{query:?}");
                assert_eq!(r["b"]["bounds"]["bottom_right"]["lat"].as_f64(), Some(bottom), "{query:?}");
                assert_eq!(r["b"]["bounds"]["top_left"]["lon"].as_f64(), Some(left), "{query:?}");
                assert_eq!(r["b"]["bounds"]["bottom_right"]["lon"].as_f64(), Some(right), "{query:?}");
            }
        }
    }
}

/// A polygon Lucene cannot tessellate fails its doc as in ES (a document
/// parsing error caused by the tessellator's message), unless the field
/// ignores malformed values; valid shapes index with their triangles.
#[test]
fn untessellatable_geo_shapes_are_rejected_like_es() {
    let bow = json!({"g": "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))"});
    let e = try_fx(json!({"properties": {"g": {"type": "geo_shape"}}}), &[bow.clone()]).err().expect("rejected");
    assert_eq!(e.kind, "document_parsing_exception", "{e:?}");
    let cause = format!("{e:?}");
    assert!(cause.contains("self-intersection"), "{cause}");
    let f = fx(json!({"properties": {"g": {"type": "geo_shape", "ignore_malformed": true}}}), &[bow, json!({"g": "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"})]);
    assert_eq!(search(&f, json!({"exists": {"field": "g"}})), ids(&["2"]));
    let f = fx(
        json!({"properties": {"g": {"type": "geo_shape"}}}),
        &[json!({"g": "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0.5 0.5, 0.5 1, 1 1, 1 0.5, 0.5 0.5))"}), json!({"g": "LINESTRING (0 0, 1 1)"}), json!({"g": "POINT (1 1)"})],
    );
    assert_eq!(sorted(search(&f, json!({"exists": {"field": "g"}}))), ids(&["1", "2", "3"]));
}


/// Circle queries on a `geo_shape` field (`geo_shape` with a circle and
/// `geo_distance`) relate Lucene's `Circle2D` (not a polygon): every
/// relation equals the per-doc relation to the circle on the memtable, the
/// sealed trees and an `index: false` field's doc values.
#[test]
fn geo_shape_circle_queries_use_circle2d() {
    use rustic_core::fieldtypes::geo2d::{ComponentLeaf, LatLonGeometry, RectLeaf, ShapeQuery};
    use rustic_core::fieldtypes::{shape, shape_bkd::Matcher, shape_dv::ShapeValue, tessellator};
    let mut docs = shape_complex_docs(2000, 5);
    docs.extend(shape_point_docs(1500, 8));
    let json_docs: Vec<Value> = docs.iter().map(|v| match v.len() {
        0 => json!({}),
        1 => json!({"g": v[0]}),
        _ => json!({"g": v}),
    }).collect();
    let mut f = shape_point_fixture(&docs);
    let mut dvf = fx(json!({"properties": {"g": {"type": "geo_shape", "index": false}}}), &json_docs);
    dvf.index.refresh().unwrap();
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    let per_doc: Vec<Option<(Vec<[i32; 7]>, Vec<u8>)>> = docs
        .iter()
        .map(|vals| {
            let mut tris = Vec::new();
            let mut calc = shape::CentroidCalc::default();
            for v in vals {
                let g = shape::parse(&json!(v), &opts).unwrap();
                tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
                calc.add(&g);
            }
            (!tris.is_empty()).then(|| {
                let b = rustic_core::fieldtypes::shape_field::doc_value(&tris, &calc).unwrap();
                (tris, b)
            })
        })
        .collect();
    let circles = [(51.5, -0.12, 30_000.0), (48.85, 2.3, 120_000.0), (0.0, 179.9, 500_000.0), (55.6, 12.6, 2_000.0), (52.0, 5.0, 400_000.0)];
    for sealed in [false, true] {
        if sealed {
            f.index.refresh().unwrap();
        }
        for &(lat, lon, m) in &circles {
            let (qlat, qlon) = rustic_core::geo::quantize(lat, lon);
            for (name, rel) in SHAPE_RELS {
                let sq = ShapeQuery::new(&[LatLonGeometry::Circle { lat: qlat, lon: qlon, meters: m }], crate::fieldtypes::geo_shape_query::query_relation(rel));
                let leaf: Box<dyn rustic_core::fieldtypes::geo2d::LeafQuery + '_> = match &sq {
                    ShapeQuery::Component(c, r) => Box::new(ComponentLeaf { component: c, rel: *r }),
                    ShapeQuery::Rect(e, _, r) => Box::new(RectLeaf { rect: e, rel: *r }),
                    q => panic!("{q:?}"),
                };
                let want: Vec<String> = sorted(per_doc.iter().enumerate().filter_map(|(i, d)| {
                    let (tris, _) = d.as_ref()?;
                    let mut mt = Matcher::new(leaf.as_ref(), 1);
                    mt.doc(0, tris);
                    (mt.finish().unwrap()[0] & 1 == 1).then(|| (i + 1).to_string())
                }).collect());
                let want_dv: Vec<String> = sorted(per_doc.iter().enumerate().filter_map(|(i, d)| {
                    let (_, b) = d.as_ref()?;
                    crate::fieldtypes::geo_shape_query::value_matches(&ShapeValue::new(b), &sq).unwrap().then(|| (i + 1).to_string())
                }).collect());
                assert_eq!(want, want_dv, "tree and doc-value relations of circle {lat},{lon} {m} {name}");
                let body = json!({"geo_shape": {"g": {"shape": {"type": "circle", "coordinates": [lon, lat], "radius": format!("{m}m")}, "relation": name}}});
                let got = sorted(search(&f, body.clone()));
                assert_eq!(got, want, "{body} sealed={sealed}");
                if sealed {
                    assert_eq!(sorted(search(&dvf, body.clone())), want, "doc values {body}");
                }
                if name == "intersects" {
                    let body = json!({"geo_distance": {"distance": format!("{m}m"), "g": {"lat": lat, "lon": lon}}});
                    assert_eq!(sorted(search(&f, body.clone())), want, "{body} sealed={sealed}");
                }
            }
        }
    }
}

/// A `geo_shape` runtime field (its script emitting the source's shape)
/// answers `geo_shape` queries of every relation, `geo_bounding_box` and
/// `geo_distance` as the indexed field does: the emitted values related as
/// one shape doc value.
#[test]
fn geo_shape_runtime_field_relates_like_the_indexed_field() {
    let tmp = tempfile::tempdir().unwrap();
    let node = crate::node::Node::open(tmp.path()).unwrap();
    node.create_index("rt", &json!({"settings": {"number_of_shards": 1}, "mappings": {
        "runtime": {"r": {"type": "geo_shape", "script": {"source": "emit(params._source.g)"}}},
        "properties": {"g": {"type": "geo_shape"}}}})).unwrap();
    let svc = node.get_index("rt").unwrap();
    let docs = shape_complex_docs(400, 12);
    let reqs: Vec<crate::index::WriteReq> = docs
        .iter()
        .enumerate()
        .filter(|(_, v)| v.len() == 1)
        .map(|(i, v)| crate::index::WriteReq::Index { id: Some(format!("{i}")), source: json!({"g": v[0]}).to_string(), routing: None, cond: Default::default() })
        .collect();
    for r in svc.write(reqs, false) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    let try_ids = |q: Value| -> std::result::Result<Vec<String>, String> {
        let body = json!({"size": 10000, "query": q, "_source": false});
        let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
        let targets = vec![crate::search::Target::new(svc.clone())];
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
        let r = crate::search::execute(&targets, &req, &opts).map_err(|e| format!("{e:?}").replace("[r]", "[g]"))?;
        let v: Value = serde_json::from_str(&r.json).unwrap();
        Ok(sorted(v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()))
    };
    let ids = |q: Value| try_ids(q).unwrap();
    let mut nonempty = 0;
    for q in rally_queries() {
        for (name, _) in SHAPE_RELS {
            let indexed = try_ids(json!({"geo_shape": {"g": {"shape": q, "relation": name}}}));
            let runtime = try_ids(json!({"geo_shape": {"r": {"shape": q, "relation": name}}}));
            assert_eq!(runtime.is_ok(), indexed.is_ok(), "{q} {name}: {runtime:?} / {indexed:?}");
            if let (Ok(r), Ok(i)) = (&runtime, &indexed) {
                assert_eq!(r, i, "{q} {name}");
                nonempty += usize::from(!i.is_empty());
            }
        }
    }
    assert!(nonempty > 10);
    for (a, b) in [
        (json!({"geo_bounding_box": {"g": {"top_left": "POINT (-1 53)", "bottom_right": "POINT (3 49)"}}}), json!({"geo_bounding_box": {"r": {"top_left": "POINT (-1 53)", "bottom_right": "POINT (3 49)"}}})),
        (json!({"geo_distance": {"distance": "200km", "g": {"lat": 51.5, "lon": 0.0}}}), json!({"geo_distance": {"distance": "200km", "r": {"lat": 51.5, "lon": 0.0}}})),
    ] {
        assert_eq!(ids(b.clone()), ids(a.clone()), "{b}");
    }
}

/// geohex parity with ES 9.5.5: `geohex_grid` buckets (precisions 0-15 on
/// `geo_point`, 0-8 on `geo_shape`) and `geo_grid` geohex filter hits on
/// both, over points on cell vertices and edges and a ulp off them, around
/// pentagons, at the poles and the dateline, random points, and small and
/// large shapes across cell edges. The fixture is ES's own answers (see
/// `h3/fixtures/es_fixture.py`), checked in the memtable and sealed.
#[test]
fn geohex_aggs_and_filters_match_es() {
    use std::collections::BTreeMap;
    use std::io::Read;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&include_bytes!("../h3/fixtures/es_geohex.json.gz")[..]).read_to_string(&mut text).unwrap();
    let fixture: Value = serde_json::from_str(&text).unwrap();
    let docs: Vec<Value> = fixture["docs"].as_array().unwrap().iter().enumerate().map(|(i, d)| {
        let mut d = d.clone();
        d["_test_id"] = json!(i.to_string());
        d
    }).collect();
    let mut f = fx(json!({"properties": {"p": {"type": "geo_point"}, "s": {"type": "geo_shape"}, "k": {"type": "keyword"}}}), &docs);
    for sealed in [false, true] {
        if sealed {
            f.index.refresh().unwrap();
        }
        let mut bad: Vec<String> = Vec::new();
        for (key, want) in fixture["aggs"].as_object().unwrap() {
            let (field, prec) = key.split_once('/').unwrap();
            let r = aggs(&f, None, json!({"g": {"geohex_grid": {"field": field, "precision": prec.parse::<u32>().unwrap(), "size": 100000}}})).unwrap();
            let got: BTreeMap<String, u64> = keys(&r["g"]).into_iter().map(|(k, c)| (k.as_str().unwrap().to_string(), c)).collect();
            let want: BTreeMap<String, u64> = want.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
            if got != want {
                let only_got: Vec<_> = got.iter().filter(|(k, v)| want.get(*k) != Some(v)).take(5).collect();
                let only_want: Vec<_> = want.iter().filter(|(k, v)| got.get(*k) != Some(v)).take(5).collect();
                bad.push(format!("agg {key}: ours {only_got:?} vs ES {only_want:?}"));
            }
        }
        for (key, want) in fixture["filters"].as_object().unwrap() {
            let (field, cell) = key.split_once('/').unwrap();
            let mut got: Vec<u64> = search(&f, json!({"geo_grid": {field: {"geohex": cell}}})).iter().map(|s| s.parse().unwrap()).collect();
            got.sort_unstable();
            let want: Vec<u64> = want.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            if got != want {
                let extra: Vec<_> = got.iter().filter(|d| !want.contains(d)).take(5).collect();
                let missing: Vec<_> = want.iter().filter(|d| !got.contains(d)).take(5).collect();
                bad.push(format!("filter {key}: extra {extra:?} missing {missing:?}"));
            }
        }
        assert!(bad.is_empty(), "sealed={sealed}: {} differ from ES:\n{}", bad.len(), bad.join("\n"));
    }
}

/// `range` on a `scaled_float` matches what ES's long range matches: the
/// bounds scaled and rounded to longs (`ceil` of the lower, nudged up when
/// exclusive; `floor` of the upper, nudged down), on grid and off grid,
/// with every inclusive / exclusive combination.
#[test]
fn scaled_float_range_bounds_match_es_long_range() {
    let factor = 100.0f64;
    let longs: Vec<i64> = (-30..=330).step_by(7).chain([0, 1, 29, 30, 31, 299, 300, 301, 88730]).collect();
    let docs: Vec<Value> = longs.iter().map(|&l| json!({"p": l as f64 / factor})).collect();
    let f = fx(json!({"properties": {"p": {"type": "scaled_float", "scaling_factor": 100}}}), &docs);
    let lo_long = |x: f64, inclusive: bool| -> i64 { let d = x * factor; (if inclusive { d } else { d.next_up() }).ceil() as i64 };
    let hi_long = |x: f64, inclusive: bool| -> i64 { let d = x * factor; (if inclusive { d } else { d.next_down() }).floor() as i64 };
    for (a, b) in [(0.3, 3.0), (0.29, 3.01), (0.295, 2.995), (-0.07, 0.0), (1.005, 887.3), (0.1 + 0.2, 0.7)] {
        for (lk, hk) in [("gte", "lte"), ("gt", "lt"), ("gte", "lt"), ("gt", "lte")] {
            let (lo, hi) = (lo_long(a, lk == "gte"), hi_long(b, hk == "lte"));
            let want: Vec<String> = longs.iter().enumerate().filter(|(_, l)| lo <= **l && **l <= hi).map(|(i, _)| (i + 1).to_string()).collect();
            let got = sorted(search(&f, json!({"range": {"p": {lk: a, hk: b}}})));
            let mut want = want;
            want.sort();
            assert_eq!(got, want, "{lk} {a} {hk} {b}");
        }
    }
}
