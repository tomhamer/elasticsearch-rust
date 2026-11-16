//! Relevance closure tests (review findings 254–259, 262) through the
//! local search path.

use crate::index::WriteReq;
use crate::node::Node;
use crate::shard::WriteCond;
use serde_json::{json, Value};
use std::sync::Arc;

fn index(node: &Arc<Node>, name: &str, body: Value, docs: &[(&str, Value)]) -> Arc<crate::index::IndexService> {
    let svc = node.create_index(name, &body).unwrap();
    let reqs: Vec<WriteReq> = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
    assert!(svc.write(reqs, false).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    svc
}

fn search(node: &Node, index: &str, body: Value) -> Value {
    crate::ml::search(node, index, &body).unwrap()
}

fn ids(resp: &Value) -> Vec<String> {
    let mut v: Vec<String> = resp["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    v.sort();
    v
}

fn scores(resp: &Value) -> std::collections::BTreeMap<String, f64> {
    resp["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
}

fn node() -> (tempfile::TempDir, Arc<Node>) {
    let d = tempfile::tempdir().unwrap();
    let n = Node::open(d.path()).unwrap();
    (d, n)
}

const ONE_SHARD: &str = r#"{"number_of_shards": 1, "number_of_replicas": 0}"#;

fn settings(analysis: Value) -> Value {
    let mut s: Value = serde_json::from_str(ONE_SHARD).unwrap();
    if !analysis.is_null() {
        s["analysis"] = analysis;
    }
    s
}

/// #254: a phrase keeps the holes the analyzer left (stop words): `quick
/// the fox` matches only the document with a word between `quick` and
/// `fox`, as match_phrase, quoted query_string and phrase prefix.
#[test]
fn phrases_keep_stop_word_holes() {
    let (_d, node) = node();
    index(&node, "p", json!({"settings": settings(Value::Null), "mappings": {"properties": {"t": {"type": "text", "analyzer": "stop"}}}}),
        &[("1", json!({"t": "quick the fox"})), ("2", json!({"t": "quick fox"})), ("3", json!({"t": "the quick brown fox"}))]);
    // `quick _ fox`: doc 1, and doc 3's `quick brown fox`.
    assert_eq!(ids(&search(&node, "p", json!({"query": {"match_phrase": {"t": "quick the fox"}}}))), vec!["1", "3"]);
    assert_eq!(ids(&search(&node, "p", json!({"query": {"match_phrase": {"t": "quick fox"}}}))), vec!["2"], "control: no hole");
    assert_eq!(ids(&search(&node, "p", json!({"query": {"query_string": {"default_field": "t", "query": "\"quick the fox\""}}}))), vec!["1", "3"]);
    assert_eq!(ids(&search(&node, "p", json!({"query": {"match_phrase_prefix": {"t": "quick the fo"}}}))), vec!["1", "3"]);
    assert_eq!(ids(&search(&node, "p", json!({"query": {"match_phrase": {"t": "the quick a fox"}}}))), vec!["1", "3"], "a leading stop word, an interior hole");
}

/// #255: multi-word synonyms match token sequences and emit token
/// sequences, in both directions (search-time rules here).
#[test]
fn multiword_synonyms_are_token_sequences() {
    let (_d, node) = node();
    let analysis = json!({"filter": {"ny": {"type": "synonym_graph", "synonyms": ["ny => new york"]}, "nyc": {"type": "synonym", "synonyms": ["new york => nyc"]}},
        "analyzer": {"to_words": {"tokenizer": "standard", "filter": ["lowercase", "ny"]}, "to_one": {"tokenizer": "standard", "filter": ["lowercase", "nyc"]}}});
    index(&node, "s", json!({"settings": settings(analysis), "mappings": {"properties": {
        "a": {"type": "text", "search_analyzer": "to_words"}, "b": {"type": "text", "search_analyzer": "to_one"}}}}),
        &[("1", json!({"a": "I love new york", "b": "nyc marathon"})), ("2", json!({"a": "york minster", "b": "new marathon"}))]);
    assert_eq!(ids(&search(&node, "s", json!({"query": {"match_phrase": {"a": "ny"}}}))), vec!["1"], "ny is the phrase new york");
    assert!(ids(&search(&node, "s", json!({"query": {"match": {"a": "ny"}}}))).contains(&"1".to_string()));
    assert_eq!(ids(&search(&node, "s", json!({"query": {"match": {"b": "new york"}}}))), vec!["1"], "the two words are nyc");
}

/// #256: cross_fields groups fields by search analyzer: two fields with
/// different index analyzers and one search analyzer form one group.
#[test]
fn cross_fields_groups_by_search_analyzer() {
    let (_d, node) = node();
    let analysis = json!({"analyzer": {
        "i1": {"tokenizer": "standard", "filter": ["lowercase"]}, "i2": {"tokenizer": "standard", "filter": ["lowercase"]}, "q": {"tokenizer": "standard", "filter": ["lowercase"]}}});
    index(&node, "c", json!({"settings": settings(analysis), "mappings": {"properties": {
        "first": {"type": "text", "analyzer": "i1", "search_analyzer": "q"}, "last": {"type": "text", "analyzer": "i2", "search_analyzer": "q"}}}}),
        &[("1", json!({"first": "Will", "last": "Smith"}))]);
    let r = search(&node, "c", json!({"query": {"multi_match": {"query": "Will Smith", "type": "cross_fields", "fields": ["first", "last"], "operator": "and"}}}));
    assert_eq!(ids(&r), vec!["1"]);
}

/// #257: a seeded random_score with a field scores by the field's value:
/// unchanged by a merge, equal for equal values.
#[test]
fn seeded_random_score_follows_its_field() {
    let (_d, node) = node();
    let svc = node.create_index("r", &json!({"settings": settings(Value::Null), "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}})).unwrap();
    for batch in [vec!["1", "2", "3"], vec!["4", "5", "6"]] {
        let reqs: Vec<WriteReq> = batch.iter().map(|id| WriteReq::Index { id: Some(id.to_string()), source: json!({"k": format!("v{}", id.parse::<u32>().unwrap() % 3), "n": id.parse::<u32>().unwrap()}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        assert!(svc.write(reqs, false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }
    let body = |field: &str| json!({"size": 10, "query": {"function_score": {"random_score": {"seed": 42, "field": field}, "boost_mode": "replace"}}});
    let (k, n) = (scores(&search(&node, "r", body("k"))), scores(&search(&node, "r", body("n"))));
    assert_eq!(k["1"], k["4"], "equal values, equal scores");
    assert_ne!(n["1"], n["4"]);
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    assert_eq!(scores(&search(&node, "r", body("k"))), k, "stable across the merge");
    assert_eq!(scores(&search(&node, "r", body("n"))), n);
}

/// #258 / #259: a decay reduces a multi-valued field's distances by
/// multi_value_mode; date_nanos decays in the same units as date.
#[test]
fn decay_modes_and_date_nanos_units() {
    let (_d, node) = node();
    index(&node, "d", json!({"settings": settings(Value::Null), "mappings": {"properties": {"n": {"type": "long"}, "dn": {"type": "date_nanos"}, "dm": {"type": "date"}}}}),
        &[("1", json!({"n": [0, 100], "dn": "2024-01-01T00:00:00.000000001Z", "dm": "2024-01-01T00:00:00Z"})), ("2", json!({"n": [5], "dn": "2024-01-02T00:00:00Z", "dm": "2024-01-02T00:00:00Z"}))]);
    let decay = |mode: &str, min_score: f64| json!({"query": {"function_score": {"linear": {"n": {"origin": 0, "scale": 10, "decay": 0.5}, "multi_value_mode": mode}, "boost_mode": "replace", "min_score": min_score}}});
    let s = |mode: &str| scores(&search(&node, "d", decay(mode, 0.0)));
    assert!((s("min")["1"] - 1.0).abs() < 1e-6);
    assert!(s("max")["1"] < 1e-6, "max distance 100 is far past the scale");
    assert!((s("avg")["1"] - s("sum")["1"]).abs() < 1e-6 && s("avg")["1"] < 1e-6);
    assert_eq!(ids(&search(&node, "d", decay("max", 0.5))), vec!["2"], "min_score drops the far doc under max");
    let gauss = |field: &str| scores(&search(&node, "d", json!({"query": {"function_score": {"gauss": {field: {"origin": "2024-01-01T00:00:00Z", "scale": "1d"}}, "boost_mode": "replace"}}})));
    let (nanos, millis) = (gauss("dn"), gauss("dm"));
    assert!(nanos["1"] > 0.99, "{nanos:?}");
    assert!((nanos["2"] - 0.5).abs() < 1e-3 && (millis["2"] - 0.5).abs() < 1e-3, "{nanos:?} {millis:?}");
}

/// #262: a compound retriever's child searches and final fetch read one
/// view: a document replaced and refreshed between them is returned as the
/// version its rank came from.
#[test]
fn compound_retrieval_reads_one_view() {
    let (_d, node) = node();
    let svc = index(&node, "rr", json!({"settings": settings(Value::Null), "mappings": {"properties": {"t": {"type": "text"}}}}),
        &[("1", json!({"t": "alpha beta"})), ("2", json!({"t": "alpha"})), ("3", json!({"t": "beta"}))]);
    let changer = svc.clone();
    let me = std::thread::current().id();
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = fired.clone();
    *crate::retriever::BETWEEN_PHASES.lock() = Some(Box::new(move || {
        if std::thread::current().id() != me {
            return;
        }
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        let r = WriteReq::Index { id: Some("1".into()), source: json!({"t": "gamma"}).to_string(), routing: None, cond: WriteCond::default() };
        let _ = changer.write(vec![r], false);
        let _ = changer.refresh();
    }));
    let r = search(&node, "rr", json!({"retriever": {"rrf": {"retrievers": [{"standard": {"query": {"match": {"t": "alpha"}}}}, {"standard": {"query": {"match": {"t": "beta"}}}}]}}}));
    *crate::retriever::BETWEEN_PHASES.lock() = None;
    assert!(fired.load(std::sync::atomic::Ordering::SeqCst), "the document changed between the phases");
    let hit = r["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == "1").expect("the ranked doc");
    assert_eq!(hit["_source"]["t"], json!("alpha beta"), "fetched from the view it was ranked in");
}

/// #255 (query shape): with an OR operator a multi-word synonym
/// alternative is a phrase next to the one-word alternative (ES
/// `analyzeGraphBoolean`), in match, multi_match and query_string; with
/// `auto_generate_synonyms_phrase_query: false` all its words.
#[test]
fn multiword_synonym_alternatives_are_phrases() {
    let (_d, node) = node();
    let analysis = json!({"filter": {"ny": {"type": "synonym_graph", "synonyms": ["ny, new york"]}},
        "analyzer": {"syn": {"tokenizer": "standard", "filter": ["lowercase", "ny"]}}});
    index(&node, "g", json!({"settings": settings(analysis), "mappings": {"properties": {"a": {"type": "text", "search_analyzer": "syn"}}}}),
        &[("1", json!({"a": "new york city"})), ("2", json!({"a": "york new"})), ("3", json!({"a": "ny"})), ("4", json!({"a": "york"}))]);
    let phrase = vec!["1", "3"];
    assert_eq!(ids(&search(&node, "g", json!({"query": {"match": {"a": "ny"}}}))), phrase);
    assert_eq!(ids(&search(&node, "g", json!({"query": {"match": {"a": {"query": "new york"}}}}))), phrase);
    assert_eq!(ids(&search(&node, "g", json!({"query": {"multi_match": {"query": "ny", "fields": ["a"]}}}))), phrase);
    assert_eq!(ids(&search(&node, "g", json!({"query": {"query_string": {"query": "ny", "default_field": "a"}}}))), phrase);
    assert_eq!(ids(&search(&node, "g", json!({"query": {"match": {"a": {"query": "ny", "auto_generate_synonyms_phrase_query": false}}}}))), vec!["1", "2", "3"]);
}

/// #255 (phrases): a phrase over a synonym graph is a span near of its
/// positions, each multi-word alternative an exact span: `ny city` (rule
/// `ny, new york`) matches `ny city` and `new york city`.
#[test]
fn phrases_over_synonym_graphs_are_spans() {
    let (_d, node) = node();
    let analysis = json!({"filter": {"ny": {"type": "synonym_graph", "synonyms": ["ny, new york"]}},
        "analyzer": {"syn": {"tokenizer": "standard", "filter": ["lowercase", "ny"]}}});
    index(&node, "gp", json!({"settings": settings(analysis), "mappings": {"properties": {"a": {"type": "text", "search_analyzer": "syn"}}}}),
        &[("1", json!({"a": "new york city"})), ("2", json!({"a": "ny city"})), ("3", json!({"a": "york city new"})), ("4", json!({"a": "new jersey city"}))]);
    assert_eq!(ids(&search(&node, "gp", json!({"query": {"match_phrase": {"a": "ny city"}}}))), vec!["1", "2"]);
    assert_eq!(ids(&search(&node, "gp", json!({"query": {"match_phrase": {"a": "new york city"}}}))), vec!["1", "2"]);
    assert_eq!(ids(&search(&node, "gp", json!({"query": {"query_string": {"query": "\"ny city\"", "default_field": "a"}}}))), vec!["1", "2"]);
}
