//! Closure tests for production review findings 491–526 (query semantics):
//! each builds a small index and runs the review's witness request.

use crate::index::{IndexService, WriteReq};
use crate::settings;
use crate::shard::WriteCond;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

struct Ix {
    _dir: tempfile::TempDir,
    svc: Arc<IndexService>,
}

fn index(settings_json: Value, mappings: Value, docs: Vec<(&str, Value)>) -> Ix {
    index_named("t", settings_json, mappings, docs)
}

fn index_named(name: &str, settings_json: Value, mappings: Value, docs: Vec<(&str, Value)>) -> Ix {
    let dir = tempfile::tempdir().unwrap();
    let mut s = json!({"number_of_shards": 1, "number_of_replicas": 0});
    if let (Some(o), Some(extra)) = (s.as_object_mut(), settings_json.as_object()) {
        for (k, v) in extra {
            o.insert(k.clone(), v.clone());
        }
    }
    let svc = Arc::new(IndexService::create(dir.path(), name, settings::normalize_index_settings(&s), &mappings, BTreeMap::new()).unwrap());
    let reqs: Vec<WriteReq> = docs.into_iter().map(|(id, body)| WriteReq::Index { id: Some(id.into()), source: body.to_string(), routing: None, cond: WriteCond::default() }).collect();
    for r in svc.write(reqs, true) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    Ix { _dir: dir, svc }
}

fn search_with(ix: &Ix, body: Value, lookup: Option<crate::query::dsl::TermsLookupFn>) -> Value {
    let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(ix.svc.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: lookup, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap()
}

fn search(ix: &Ix, body: Value) -> Value {
    search_with(ix, body, None)
}

/// Hit ids, sorted.
fn ids(ix: &Ix, query: Value) -> Vec<String> {
    let v = search(ix, json!({"query": query, "size": 200}));
    let mut out: Vec<String> = v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    out.sort();
    out
}

/// (id, score) of the hits in response order.
fn scored(ix: &Ix, body: Value) -> Vec<(String, f64)> {
    let v = search(ix, body);
    v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap_or(0.0))).collect()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// #491 / #492: wildcard regexp uses the Lucene grammar (`.` spans line
/// breaks, `@` is any string) and its n-gram prefilter does not take
/// repetition counts for text; constant_keyword alike.
#[test]
fn specialized_regexp_follows_lucene() {
    let ix = index(json!({}), json!({"properties": {"w": {"type": "wildcard"}, "c": {"type": "constant_keyword", "value": "first\nlast"}}}),
        vec![("1", json!({"w": "serial123"})), ("2", json!({"w": "first\nlast"})), ("3", json!({"w": "other"}))]);
    assert_eq!(ids(&ix, json!({"regexp": {"w": "serial[0-9]{3,5}"}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"regexp": {"w": "first.*last"}})), s(&["2"]));
    assert_eq!(ids(&ix, json!({"regexp": {"w": "@"}})), s(&["1", "2", "3"]));
    assert_eq!(ids(&ix, json!({"regexp": {"w": {"value": "@", "flags": "NONE"}}})), Vec::<String>::new());
    assert_eq!(ids(&ix, json!({"regexp": {"c": "first.*last"}})).len(), 3);
}

/// #497 / #498 / #499: query_string multi-term leaves use the field's own
/// query (flattened key, normalizer), escaped wildcards stay literal, and
/// `analyze_wildcard` / `quote_field_suffix` take effect.
#[test]
fn query_string_multi_term_semantics() {
    let ix = index(
        json!({"analysis": {"analyzer": {"en": {"type": "english"}}}}),
        json!({"properties": {
            "labels": {"type": "flattened"},
            "k": {"type": "keyword", "normalizer": "lowercase"},
            "code": {"type": "keyword"},
            "body": {"type": "text", "analyzer": "en", "fields": {"exact": {"type": "text", "analyzer": "standard"}}}
        }}),
        vec![
            ("1", json!({"labels": {"env": "dev", "owner": "production"}, "k": "prod", "code": "foo*", "body": "running"})),
            ("2", json!({"labels": {"env": "production"}, "code": "foobar", "body": "run"})),
        ],
    );
    let qs = |q: &str| json!({"query_string": {"query": q}});
    assert_eq!(ids(&ix, qs("labels.env:*prod*")), s(&["2"]), "another key's value leaked");
    assert_eq!(ids(&ix, qs("labels.env:prod*")), s(&["2"]));
    assert_eq!(ids(&ix, json!({"simple_query_string": {"query": "prod*", "fields": ["labels.env"]}})), s(&["2"]));
    assert_eq!(ids(&ix, qs("k:PROD*")), s(&["1"]), "the normalizer was not applied");
    // An escaped `*` is literal.
    assert_eq!(ids(&ix, qs("code:foo\\*")), s(&["1"]));
    assert_eq!(ids(&ix, qs("code:foo*")), s(&["1", "2"]));
    // analyze_wildcard: `running*` analyzes to the stem `run`.
    assert_eq!(ids(&ix, json!({"query_string": {"query": "body:running*", "analyze_wildcard": true}})), s(&["1", "2"]));
    // quote_field_suffix: the quoted text searches the unstemmed field.
    assert_eq!(ids(&ix, json!({"query_string": {"query": "\"running\"", "fields": ["body"], "quote_field_suffix": ".exact"}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"query_string": {"query": "\"running\"", "fields": ["body"]}})), s(&["1", "2"]));
}

/// #500: a stored-document like item keeps its `per_field_analyzer`.
#[test]
fn more_like_this_stored_item_uses_its_analyzer() {
    let ix = index(json!({}), json!({"properties": {"body": {"type": "text", "fields": {"en": {"type": "text", "analyzer": "english"}}}}}),
        vec![("1", json!({"body": "running"})), ("2", json!({"body": "run"})), ("3", json!({"body": "other words"}))]);
    let svc = ix.svc.clone();
    let lookup = move |_index: &str, id: &str, path: &str, _routing: Option<&str>| -> crate::error::Result<Vec<Value>> {
        let got = svc.get(id)?;
        let src: Value = got.and_then(|g| g.source).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
        Ok(crate::highlight::source_values(&src, path).into_iter().cloned().collect())
    };
    let mlt = |pfa: Value| json!({"query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "t", "_id": "1", "per_field_analyzer": pfa}],
        "min_term_freq": 1, "min_doc_freq": 1}}});
    let hits = |v: Value| -> Vec<String> { v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect() };
    assert_eq!(hits(search_with(&ix, mlt(json!({"body": "english"})), Some(&lookup))), s(&["2"]), "the analyzer override was dropped");
    // A multi-field of a stored example reads its parent's source.
    let sub = json!({"query": {"more_like_this": {"fields": ["body.en"], "like": [{"_index": "t", "_id": "1"}], "min_term_freq": 1, "min_doc_freq": 1}}});
    assert_eq!(hits(search_with(&ix, sub, Some(&lookup))), s(&["2"]));
}

/// #501: terms_set on a doc-values-only keyword.
#[test]
fn terms_set_on_doc_values_only_keyword() {
    let ix = index(json!({}), json!({"properties": {"tags": {"type": "keyword", "index": false}, "required": {"type": "long"}}}),
        vec![("1", json!({"tags": ["red", "blue"], "required": 1}))]);
    assert_eq!(ids(&ix, json!({"terms_set": {"tags": {"terms": ["red"], "minimum_should_match_field": "required"}}})), s(&["1"]));
}

/// #502: max_expansions never limits match_bool_prefix's final prefix.
#[test]
fn match_bool_prefix_final_prefix_is_complete() {
    let docs: Vec<(String, Value)> = (0..60).map(|i| (format!("{i}"), json!({"t": format!("ab{i:03}")}))).collect();
    let ix = index(json!({}), json!({"properties": {"t": {"type": "text"}}}), docs.iter().map(|(i, d)| (i.as_str(), d.clone())).collect());
    assert_eq!(ids(&ix, json!({"match_bool_prefix": {"t": "ab"}})).len(), 60);
    assert_eq!(ids(&ix, json!({"match_bool_prefix": {"t": {"query": "ab", "max_expansions": 1}}})).len(), 60);
    assert_eq!(ids(&ix, json!({"multi_match": {"query": "ab", "type": "bool_prefix", "fields": ["t"], "max_expansions": 1}})).len(), 60);
}

/// #503: case_insensitive term / prefix / wildcard on case-preserving text.
#[test]
fn case_insensitive_text_queries() {
    let ix = index(json!({}), json!({"properties": {"t": {"type": "text", "analyzer": "whitespace"}}}), vec![("1", json!({"t": "ProductCode"}))]);
    assert_eq!(ids(&ix, json!({"term": {"t": {"value": "productcode", "case_insensitive": true}}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"prefix": {"t": {"value": "product", "case_insensitive": true}}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"wildcard": {"t": {"value": "product*", "case_insensitive": true}}})), s(&["1"]));
    assert!(ids(&ix, json!({"term": {"t": {"value": "productcode"}}})).is_empty(), "control: case-sensitive");
}

/// #504: fuzzy top terms keep the best normalized similarity, not the most
/// frequent term.
#[test]
fn fuzzy_top_terms_prefer_similarity() {
    let ix = index(json!({}), json!({"properties": {"k": {"type": "keyword"}}}), vec![("1", json!({"k": "boo"})), ("2", json!({"k": "boo"})), ("3", json!({"k": "books"}))]);
    assert_eq!(ids(&ix, json!({"fuzzy": {"k": {"value": "book", "fuzziness": 1, "max_expansions": 1}}})), s(&["3"]));
}

/// #505: an ascending `_score` sort is honoured by the hit merge.
#[test]
fn ascending_score_sort() {
    let ix = index(json!({}), json!({"properties": {"t": {"type": "text"}}}), vec![("1", json!({"t": "a"})), ("2", json!({"t": "a b c d e f"}))]);
    let hits = scored(&ix, json!({"query": {"match": {"t": "a"}}, "sort": [{"_score": "asc"}], "track_scores": true}));
    assert_eq!(hits.len(), 2);
    assert!(hits[0].1 <= hits[1].1, "{hits:?}");
    assert_eq!(hits[0].0, "2");
}

/// #510: bool_prefix multi_match sums the fields' scores.
#[test]
fn bool_prefix_multi_match_sums_fields() {
    let ix = index(json!({}), json!({"properties": {"a": {"type": "text"}, "b": {"type": "text"}}}),
        vec![("1", json!({"a": "apple", "b": "apple"})), ("2", json!({"a": "apple"}))]);
    let hits = scored(&ix, json!({"query": {"multi_match": {"query": "app", "type": "bool_prefix", "fields": ["a", "b"]}}}));
    assert_eq!(hits[0].0, "1");
    assert!(hits[0].1 > hits[1].1 * 1.5, "{hits:?}");
}

/// #511: match on a keyword splits on whitespace (mapping flag) or with an
/// explicit analyzer.
#[test]
fn keyword_full_text_analysis() {
    let ix = index(json!({}), json!({"properties": {"tags": {"type": "keyword", "split_queries_on_whitespace": true}, "plain": {"type": "keyword"}}}),
        vec![("1", json!({"tags": ["red", "blue"], "plain": ["red", "blue"]})), ("2", json!({"tags": ["red"], "plain": ["red"]}))]);
    assert_eq!(ids(&ix, json!({"match": {"tags": "red blue"}})), s(&["1", "2"]));
    assert_eq!(ids(&ix, json!({"match": {"tags": {"query": "red blue", "operator": "and"}}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"match": {"plain": {"query": "red blue", "analyzer": "whitespace", "operator": "and"}}})), s(&["1"]));
    assert!(ids(&ix, json!({"match": {"plain": "red blue"}})).is_empty(), "control: one literal term");
}

/// #512: match's fuzzy_rewrite is the rewrite that runs.
#[test]
fn match_fuzzy_rewrite_is_honoured() {
    let ix = index(json!({}), json!({"properties": {"t": {"type": "text"}}}), vec![("1", json!({"t": "book"})), ("2", json!({"t": "books"}))]);
    assert_eq!(ids(&ix, json!({"match": {"t": {"query": "book", "fuzziness": 1, "fuzzy_rewrite": "top_terms_1"}}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"match": {"t": {"query": "book", "fuzziness": 1}}})), s(&["1", "2"]));
}

/// #519 / #520: span_first / span_not keep filtered (containing) children,
/// and span scores carry proximity.
#[test]
fn span_wrappers_and_scoring() {
    let ix = index(json!({}), json!({"properties": {"t": {"type": "text"}}}),
        vec![("1", json!({"t": "red blue filler"})), ("2", json!({"t": "red filler blue"}))]);
    let containing = json!({"span_containing": {
        "big": {"span_near": {"clauses": [{"span_term": {"t": "red"}}, {"span_term": {"t": "blue"}}], "slop": 0, "in_order": true}},
        "little": {"span_term": {"t": "red"}}}});
    assert_eq!(ids(&ix, json!({"span_first": {"match": containing, "end": 3}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"span_not": {"include": {"span_term": {"t": "red"}}, "exclude": containing}})), s(&["2"]));
    let near = json!({"query": {"span_near": {"clauses": [{"span_term": {"t": "red"}}, {"span_term": {"t": "blue"}}], "slop": 1, "in_order": true}}});
    let hits = scored(&ix, near);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].0, "1");
    assert!(hits[0].1 > hits[1].1, "the closer match did not score higher: {hits:?}");
    // A filtered span scores through the similarity, not a constant 1.
    let c = scored(&ix, json!({"query": containing}));
    assert_eq!(c.len(), 1);
    assert!((c[0].1 - 1.0).abs() > 1e-6, "{c:?}");
}

/// #521 / #525: geo_distance's separate unit; a coerced full-world box.
#[test]
fn geo_distance_unit_and_full_world_box() {
    let ix = index(json!({}), json!({"properties": {"p": {"type": "geo_point"}}}), vec![("1", json!({"p": {"lat": 0.0, "lon": 0.01}})), ("2", json!({"p": {"lat": 10.0, "lon": 90.0}}))]);
    assert_eq!(ids(&ix, json!({"geo_distance": {"distance": 2, "unit": "km", "p": [0.0, 0.0]}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"geo_distance": {"distance": "2", "unit": "km", "p": [0.0, 0.0]}})), s(&["1"]));
    assert!(ids(&ix, json!({"geo_distance": {"distance": "2m", "unit": "km", "p": [0.0, 0.0]}})).is_empty(), "a suffix wins over unit");
    assert_eq!(ids(&ix, json!({"geo_bounding_box": {"p": {"top": 80, "bottom": -80, "left": 0, "right": 360}, "validation_method": "COERCE"}})), s(&["1", "2"]));
}

/// #522: exists on a field with neither index nor doc values.
#[test]
fn exists_on_source_only_numeric() {
    let ix = index(json!({}), json!({"properties": {"n": {"type": "long", "index": false, "doc_values": false}}}), vec![("1", json!({"n": 7}))]);
    assert!(ids(&ix, json!({"exists": {"field": "n"}})).is_empty());
    assert_eq!(ids(&ix, json!({"bool": {"must_not": {"exists": {"field": "n"}}}})), s(&["1"]));
}

/// #523: scaled_float term uses the indexed rounding (Java half-up).
#[test]
fn scaled_float_term_rounding() {
    let ix = index(json!({}), json!({"properties": {"x": {"type": "scaled_float", "scaling_factor": 2}}}), vec![("1", json!({"x": -0.25}))]);
    assert_eq!(ids(&ix, json!({"term": {"x": -0.25}})), s(&["1"]));
}

/// #524 / #526: date_nanos range with an explicit format, and
/// distance_feature with a nanosecond origin.
#[test]
fn date_nanos_ranges_and_distance_feature() {
    let ix = index(json!({}), json!({"properties": {"d": {"type": "date_nanos"}}}),
        vec![("1", json!({"d": "2026-01-01T00:00:00.123456789Z"})), ("2", json!({"d": "2026-01-01T00:00:00.123999999Z"})), ("3", json!({"d": "2026-01-01T00:00:00.123000000Z"}))]);
    let at = "2026-01-01T00:00:00.123456789Z";
    assert_eq!(ids(&ix, json!({"range": {"d": {"gte": at, "lte": at, "format": "strict_date_optional_time_nanos"}}})), s(&["1"]));
    assert_eq!(ids(&ix, json!({"range": {"d": {"gt": at, "format": "strict_date_optional_time_nanos"}}})), s(&["2"]));
    assert_eq!(ids(&ix, json!({"range": {"d": {"gte": format!("{at}||+0d"), "lte": format!("{at}||+0d")}}})), s(&["1"]));
    let hits = scored(&ix, json!({"query": {"distance_feature": {"field": "d", "origin": at, "pivot": "100nanos"}}}));
    assert_eq!(hits[0].0, "1", "{hits:?}");
    assert!((hits[0].1 - 1.0).abs() < 1e-6, "{hits:?}");
}

/// #499 follow-up: a wildcard anywhere in a query_string term is matched
/// against its literal runs normalized by the analyzer (ES
/// `getWildcardQuerySingle` / `Analyzer.normalize`: lower-casing, folding,
/// no stemming); a forced request analyzer applies to keywords with
/// `analyze_wildcard`.
#[test]
fn query_string_wildcards_are_normalized_everywhere() {
    let ix = index(
        json!({"analysis": {"analyzer": {"folded": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]}}}}),
        json!({"properties": {"f": {"type": "text", "analyzer": "folded"}, "code": {"type": "keyword"}}}),
        vec![("1", json!({"f": "cafe latte", "code": "fooxbar"})), ("2", json!({"f": "tea", "code": "other"}))],
    );
    let qs = |q: &str| json!({"query_string": {"query": q}});
    assert_eq!(ids(&ix, qs("f:CAF?")), s(&["1"]));
    assert_eq!(ids(&ix, qs("f:CAFÉ*")), s(&["1"]));
    assert_eq!(ids(&ix, qs("f:*ATT?")), s(&["1"]));
    assert_eq!(ids(&ix, qs("f:L?TTE")), s(&["1"]));
    // A keyword keeps its own (case-sensitive) query unless a request
    // analyzer is forced with analyze_wildcard.
    assert!(ids(&ix, qs("code:FOO*BAR")).is_empty());
    assert_eq!(ids(&ix, json!({"query_string": {"query": "code:FOO*BAR", "analyzer": "standard", "analyze_wildcard": true}})), s(&["1"]));
}

/// #505 follow-up: when the first target's query fails (partial results),
/// the merge still sorts by the requested field.
#[test]
fn failed_first_target_keeps_the_field_sort() {
    let old = index_named("old", json!({}), json!({"properties": {"f": {"type": "long"}, "t": {"type": "text"}, "n": {"type": "long"}}}), vec![("o1", json!({"f": 1, "t": "x", "n": 0}))]);
    let new = index_named("new", json!({}), json!({"properties": {"f": {"type": "keyword"}, "t": {"type": "text"}, "n": {"type": "long"}}}),
        vec![("a", json!({"f": "abc", "t": "x x x", "n": 2})), ("b", json!({"f": "abc", "t": "y", "n": 1}))]);
    let body = json!({"query": {"bool": {"must": [{"term": {"f": "abc"}}], "should": [{"match": {"t": "x"}}]}}, "sort": [{"n": "asc"}]});
    let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(old.svc.clone()), crate::search::Target::new(new.svc.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    let v: Value = serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    let order: Vec<&str> = v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap()).collect();
    assert_eq!(order, vec!["b", "a"], "{v}");
    assert_eq!(v["_shards"]["failed"], 1, "{v}");
}

/// A node reopening an index waits (bounded, as ES's 5 s shard lock wait)
/// for a lingering reference to the previous copy - a background refresh
/// or seal pass holding it for a moment - instead of failing the shard.
#[test]
fn reopen_waits_for_a_lingering_shard_reference() {
    let tmp = tempfile::tempdir().unwrap();
    let node = crate::node::Node::open(tmp.path()).unwrap();
    let svc = node.create_index("lingering", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).unwrap();
    let held = svc.shard(0).unwrap();
    let t = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(held);
    });
    drop(svc);
    drop(node);
    let node = crate::node::Node::open(tmp.path()).unwrap();
    assert!(node.get_index("lingering").is_some(), "the reopened index failed to start");
    t.join().unwrap();
}
