use super::*;
use crate::aggs::{run_aggregations, AggEnv, AggSeg};
use crate::analysis::AnalysisRegistry;
use crate::document::parse_with_dynamic;
use crate::mapping::{FieldLookup, Mappings, SOURCE_FIELD};
use rustic_core::types::{DocId, IndexSettings};
use rustic_core::Index;

struct Fixture { index: Index, lookup: FieldLookup }

fn fixture(mapping: Value, docs: &[Value]) -> Fixture {
    let registry = AnalysisRegistry::default();
    let mut mapping = Mappings::parse(&mapping, &registry).unwrap();
    mapping.assign_core_ids();
    let lookup = FieldLookup::build(&mapping, &registry);
    let mut index = Index::new("texts", lookup.core_schema(), IndexSettings::default());
    for (i, doc) in docs.iter().enumerate() {
        let (mut parsed, update) = parse_with_dynamic(doc, &i.to_string(), &mapping, &lookup, 1000).unwrap();
        assert!(update.is_none());
        // Parsing emits mapped fields. The production shard's add_meta_fields
        // attaches _source afterwards; this direct core-index fixture must do
        // the same so the source-based collector reaches its real assertions.
        if parsed.doc.get(SOURCE_FIELD).is_none() {
            let source = parsed.source.take().unwrap_or_else(|| doc.to_string());
            parsed.doc.add_keyword(SOURCE_FIELD, &source);
        }
        index.add_document(&parsed.doc).unwrap();
    }
    Fixture { index, lookup }
}

fn mapping() -> Value {
    json!({"properties": {"text": {"type": "text"}, "group": {"type": "keyword"}, "other": {"type": "text"}}})
}

fn execute(f: &Fixture, spec: &Value, docs: &[DocId], visibility: Option<&[DocId]>, partial_mode: bool) -> Result<Value> {
    let leaves = f.index.leaves();
    assert_eq!(leaves.len(), 1);
    let segs = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "texts", max_regex_length: 1000, docs: docs.to_vec(), scores: docs.iter().map(|d| *d as f32).collect() }];
    let fetch = |si: usize, doc: DocId, _score: f32, _spec: &Value| -> Result<Value> {
        let stored = segs[si].leaf.get_doc(doc).unwrap().expect("fixture document must be stored");
        let raw = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).expect("fixture must attach production stored _source after parsing");
        let source: Value = serde_json::from_str(raw).expect("fixture _source must be JSON");
        Ok(json!({"_source": source}))
    };
    let source = |si: usize, docs: &[(DocId, f32)], paths: &[String], emit: &mut dyn FnMut(DocId, Option<Value>) -> Result<()>| -> Result<()> {
        let ids: Vec<_> = docs.iter().map(|&(doc, _)| doc).collect();
        let mut error = None;
        segs[si].leaf.try_for_each_doc(&ids, |doc, stored| {
            if error.is_some() { return; }
            let raw = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).unwrap();
            if let Err(e) = select_source(raw, paths).and_then(|source| emit(doc, Some(source))) { error = Some(e); }
        })?;
        if let Some(e) = error { return Err(e); }
        Ok(())
    };
    let all = |si: usize| -> Vec<DocId> { visibility.map(|ds| ds.to_vec()).unwrap_or_else(|| (0..segs[si].leaf.num_docs()).filter(|&d| !segs[si].leaf.is_deleted(d)).collect()) };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65_536, fetch_hit: &fetch, fetch_source: Some(&source), all_docs: &all, background_filtered: visibility.is_some(), top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    if partial_mode {
        let req = partial::AggRequest { aggs: spec.clone(), typed_keys: false, max_buckets: 65_536, now: 0 };
        Ok(partial::shard_partial(spec, &segs, &env, &req, true)?.tree)
    } else { run_aggregations(spec, &segs, &env) }
}

#[test]
fn significant_text_stored_source_survives_memtable_flush() {
    let mut f = fixture(mapping(), &[
        json!({"text": ["alpha alpha", "beta"]}), json!({"text": "alpha beta"}),
        json!({"text": "other"}), json!({"text": "other"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "text", "min_doc_count": 1}}});
    assert!(matches!(f.index.leaves().as_slice(), [rustic_core::query::LeafReader::Memtable(_, ..)]));
    let before = execute(&f, &spec, &[0, 1], None, false).unwrap();
    assert_eq!(before["sig"]["buckets"], json!([
        {"key": "alpha", "doc_count": 2, "bg_count": 2, "score": 1.0},
        {"key": "beta", "doc_count": 2, "bg_count": 2, "score": 1.0}
    ]));
    f.index.refresh().unwrap();
    assert!(matches!(f.index.leaves().as_slice(), [rustic_core::query::LeafReader::Segment(_)]));
    assert_eq!(execute(&f, &spec, &[0, 1], None, false).unwrap(), before);
}

#[test]
fn significant_text_counts_documents_without_fielddata_and_uses_index_analyzer() {
    let f = fixture(json!({"properties": {"text": {"type": "text", "analyzer": "english", "search_analyzer": "keyword"}}}), &[
        json!({"text": ["running runs", "running"]}), json!({"text": "run"}), json!({"text": "running"}),
        json!({"text": "unrelated"}), json!({"text": "other"}), json!({}),
    ]);
    assert!(!f.lookup.get("text").unwrap().fielddata);
    let spec = json!({"sig": {"significant_text": {"field": "text", "include": ["run"]}}});
    let out = execute(&f, &spec, &[0, 1, 2], None, false).unwrap();
    let sig = &out["sig"];
    assert_eq!(sig["doc_count"], 3);
    assert_eq!(sig["bg_count"], 6);
    assert_eq!(sig["buckets"], json!([{"key": "run", "doc_count": 3, "bg_count": 3, "score": 1.0}]));
}

#[test]
fn significant_text_default_index_analyzer_ignores_search_analyzer() {
    let f = fixture(json!({"properties": {"text": {"type": "text", "search_analyzer": "keyword"}}}), &[
        json!({"text": "Alpha Beta"}), json!({"text": "ALPHA beta"}), json!({"text": "other"}), json!({"text": "other"}),
    ]);
    let out = execute(&f, &json!({"sig": {"significant_text": {"field": "text", "include": ["alpha"], "min_doc_count": 1}}}), &[0, 1], None, false).unwrap();
    assert_eq!(out["sig"]["buckets"], json!([{"key": "alpha", "doc_count": 2, "bg_count": 2, "score": 1.0}]));
}

#[test]
fn significant_text_source_fields_share_document_deduplication() {
    let f = fixture(mapping(), &[
        json!({"text": "alpha alpha", "other": ["alpha", "beta"]}),
        json!({"text": "alpha", "other": "alpha beta"}),
        json!({"text": "unrelated"}), json!({"text": "unrelated"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "text", "source_fields": ["text", "other"], "include": ["alpha"], "min_doc_count": 1}}});
    let out = execute(&f, &spec, &[0, 1], None, false).unwrap();
    assert_eq!(out["sig"]["buckets"][0]["doc_count"], 2);
    assert_eq!(out["sig"]["buckets"][0]["bg_count"], 2);
}

#[test]
fn significant_text_background_filter_and_visibility_are_independent_of_foreground() {
    let f = fixture(mapping(), &[
        json!({"text": "rare", "group": "visible"}), json!({"text": "rare", "group": "visible"}),
        json!({"text": "other", "group": "visible"}), json!({"text": "other", "group": "visible"}),
        json!({"text": "rare", "group": "hidden"}), json!({"text": "rare", "group": "hidden"}),
    ]);
    let base = json!({"sig": {"significant_text": {"field": "text", "min_doc_count": 1}}});
    let visible = execute(&f, &base, &[0, 1], Some(&[0, 1, 2, 3]), false).unwrap();
    let mut filtered = base.clone();
    filtered["sig"]["significant_text"]["background_filter"] = json!({"term": {"group": "visible"}});
    assert_eq!(visible, execute(&f, &filtered, &[0, 1], None, false).unwrap());
    assert_eq!(visible["sig"]["bg_count"], 4);
    assert_eq!(visible["sig"]["buckets"][0]["bg_count"], 2);
}

#[test]
fn significant_text_unfiltered_background_includes_deletes_until_merge() {
    let mut f = fixture(mapping(), &[
        json!({"text": "rare"}), json!({"text": "rare"}), json!({"text": "rare"}), json!({"text": "other"}),
    ]);
    f.index.delete_doc(&rustic_core::types::DocAddress::new(0, 2)).unwrap();
    let spec = json!({"sig": {"significant_text": {"field": "text", "min_doc_count": 1}}});
    let out = execute(&f, &spec, &[0, 1], None, false).unwrap();
    assert_eq!(out["sig"]["bg_count"], 4);
    assert_eq!(out["sig"]["buckets"][0]["bg_count"], 3);
    let out = execute(&f, &spec, &[0, 1], Some(&[0, 1, 3]), false).unwrap();
    assert_eq!(out["sig"]["bg_count"], 3);
    assert_eq!(out["sig"]["buckets"][0]["bg_count"], 2);
}

#[test]
fn significant_text_sampler_is_applied_per_shard_before_final_reduction() {
    let a = fixture(mapping(), &[json!({"text": "a"}), json!({"text": "rare"}), json!({"text": "rare"}), json!({"text": "other"})]);
    let b = fixture(mapping(), &[json!({"text": "b"}), json!({"text": "rare"}), json!({"text": "rare"}), json!({"text": "other"})]);
    let spec = json!({"sample": {"sampler": {"shard_size": 2}, "aggs": {"sig": {"significant_text": {"field": "text", "min_doc_count": 3}}}}});
    let pa = execute(&a, &spec, &[0, 1, 2], None, true).unwrap();
    let pb = execute(&b, &spec, &[0, 1, 2], None, true).unwrap();
    assert_eq!(pa["sample"]["doc_count"], 2);
    assert_eq!(pb["sample"]["doc_count"], 2);
    // Two per-shard occurrences must survive until the global threshold of 3.
    assert_eq!(pa["sample"]["sig"]["buckets"][0]["doc_count"], 2);
    let req = partial::AggRequest { aggs: spec, typed_keys: false, max_buckets: 65_536, now: 0 };
    let node_parts = {
        let _scope = crate::search::ReadShardCountScope::enter(Some(2));
        [pa, pb].into_iter().map(|tree| partial::AggPartial::reduce(vec![partial::AggPartial { version: 1, tree }], &req).unwrap()).collect::<Vec<_>>()
    };
    assert_eq!(node_parts[0]["sample"]["sig"]["buckets"][0]["doc_count"], 2);
    let result = partial::AggPartial::reduce(node_parts.into_iter().map(|tree| partial::AggPartial { version: 1, tree }).collect(), &req).unwrap();
    assert_eq!(result["sample"]["doc_count"], 4);
    assert_eq!(result["sample"]["sig"]["doc_count"], 4);
    assert_eq!(result["sample"]["sig"]["bg_count"], 8);
    assert_eq!(result["sample"]["sig"]["buckets"], json!([{"key": "rare", "doc_count": 4, "bg_count": 4, "score": 1.0}]));
}

#[test]
fn significant_text_intermediate_reduce_keeps_zero_scores_and_candidate_union() {
    let conf = json!({"field": "text", "size": 1, "min_doc_count": 3});
    let a = json!({"doc_count": 2, "bg_count": 2, "buckets": [{"key": "x", "doc_count": 2, "bg_count": 2, "score": 0.0}]});
    let b = json!({"doc_count": 1, "bg_count": 100, "buckets": [{"key": "x", "doc_count": 1, "bg_count": 1, "score": 99.0}, {"key": "y", "doc_count": 1, "bg_count": 1, "score": 99.0}]});
    let part = reduce(conf.as_object().unwrap(), &[&a, &b], 65_536, false).unwrap();
    assert_eq!(part["buckets"].as_array().unwrap().len(), 2);
    let final_result = reduce(conf.as_object().unwrap(), &[&part], 65_536, true).unwrap();
    assert_eq!(final_result["buckets"][0]["key"], "x");
    assert_eq!(final_result["buckets"][0]["doc_count"], 3);
    assert_eq!(final_result["buckets"][0]["bg_count"], 3);
}

#[test]
fn significant_text_output_bucket_limit_does_not_truncate_intermediate_candidates() {
    let conf = json!({"field": "text", "size": 1, "min_doc_count": 1, "percentage": {}});
    let a = json!({"doc_count": 1, "bg_count": 4, "buckets": [{"key": "x", "doc_count": 1, "bg_count": 1, "score": 1.0}]});
    let b = json!({"doc_count": 1, "bg_count": 4, "buckets": [{"key": "y", "doc_count": 1, "bg_count": 1, "score": 1.0}]});
    let part = reduce(conf.as_object().unwrap(), &[&a, &b], 1, false).unwrap();
    assert_eq!(part["buckets"].as_array().unwrap().len(), 2);
    let final_result = reduce(conf.as_object().unwrap(), &[&part], 1, true).unwrap();
    assert_eq!(final_result["buckets"].as_array().unwrap().len(), 1);
    let spec = json!({"parent": {"filter": {"match_all": {}}, "aggs": {
        "a": {"significant_text": conf}, "b": {"significant_text": conf}
    }}});
    let output = json!({"filter#parent": {"doc_count": 2, "sigsterms#a": final_result, "sigsterms#b": final_result}});
    assert_eq!(check_final_bucket_limit(&spec, &output, 1).unwrap_err().kind, "too_many_buckets_exception");
    check_final_bucket_limit(&spec, &output, 2).unwrap();
    let req = partial::AggRequest { aggs: spec, typed_keys: false, max_buckets: 1, now: 0 };
    let tree = json!({"parent": {"doc_count": 2, "a": final_result, "b": final_result}});
    assert_eq!(partial::AggPartial::reduce(vec![partial::AggPartial { version: 1, tree }], &req).unwrap_err().kind, "too_many_buckets_exception");
}

#[test]
fn significant_text_duplicate_filter_tracks_six_token_sequences_per_bucket() {
    let f = fixture(mapping(), &[
        json!({"text": "one two three four five six uniquea"}),
        json!({"text": "one two three four five six uniqueb"}),
        json!({"text": "one two three four five six uniquec"}),
        json!({"text": "other"}), json!({"text": "other"}), json!({"text": "other"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "text", "min_doc_count": 1, "percentage": {}, "filter_duplicate_text": true, "size": 20}}});
    let out = execute(&f, &spec, &[0, 1, 2], None, false).unwrap();
    let buckets = out["sig"]["buckets"].as_array().unwrap();
    assert_eq!(buckets.iter().find(|b| b["key"] == "one").unwrap()["doc_count"], 1);
    assert_eq!(buckets.iter().find(|b| b["key"] == "uniqueb").unwrap()["doc_count"], 1);
    let sibling = json!({"a": spec["sig"], "b": spec["sig"]});
    let two = execute(&f, &sibling, &[0, 1, 2], None, false).unwrap();
    assert_eq!(two["a"], two["b"]);
    let mut budget = Budget { used: 0, limit: 100_000, mem: Default::default() };
    let tokens = rustic_core::analysis::Analyzer::standard().analyze("one two three four five");
    let mut spotter = DuplicateSequences::default();
    assert!(spotter.filter(&tokens, &mut budget).unwrap().iter().all(|b| *b));
    spotter.next_document();
    assert!(spotter.filter(&tokens, &mut budget).unwrap().iter().all(|b| *b));
}

#[test]
fn significant_text_duplicate_window_spans_values_but_buffer_does_not() {
    let f = fixture(mapping(), &[
        json!({"text": "one two three four five six", "other": ["one two three", "four five six"]}),
        json!({"text": "one two three four five six", "other": ["one two three", "four five six"]}),
        json!({"text": "unrelated"}), json!({"text": "unrelated"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "text", "source_fields": ["other"], "min_doc_count": 1,
        "percentage": {}, "filter_duplicate_text": true, "size": 20}}});
    let out = execute(&f, &spec, &[0, 1], None, false).unwrap();
    let buckets = out["sig"]["buckets"].as_array().unwrap();
    // ES keeps the spotter across values, but each value has a new token
    // filter. A duplicate discovered in the second value cannot retract
    // the first value's tokens, which have already been collected.
    for key in ["one", "two", "three"] {
        assert_eq!(buckets.iter().find(|b| b["key"] == key).unwrap()["doc_count"], 2);
    }
    for key in ["four", "five", "six"] {
        assert_eq!(buckets.iter().find(|b| b["key"] == key).unwrap()["doc_count"], 1);
    }
    let mut budget = Budget { used: 0, limit: 100_000, mem: Default::default() };
    let tokens = rustic_core::analysis::Analyzer::standard().analyze("one two three four five six seven");
    let mut spotter = DuplicateSequences::default();
    assert!(spotter.filter(&tokens, &mut budget).unwrap().iter().all(|b| *b));
    spotter.next_document();
    assert!(spotter.filter(&tokens, &mut budget).unwrap().iter().all(|b| !*b));
}

#[test]
fn significant_text_validation_and_memory_failures_are_explicit() {
    let f = fixture(mapping(), &[json!({"text": "alpha"})]);
    for body in [json!({"field": "text", "size": 0}), json!({"field": "text", "source_fields": "text"}),
        json!({"field": "text", "filter_duplicate_text": "yes"}), json!({"field": "text", "script_heuristic": {"script": "1"}}),
        json!({"field": "text", "include": {"partition": 0, "num_partitions": 2}}), json!({"field": "text", "jlh": {"unknown": true}})] {
        assert!(execute(&f, &json!({"sig": {"significant_text": body}}), &[0], None, false).is_err());
    }
    let spec = json!({"sig": {"significant_text": {"field": "text"}, "aggs": {"sub": {"min": {"field": "text"}}}}});
    assert!(crate::aggs::validate_request(&spec).unwrap_err().reason.contains("cannot accept sub-aggregations"));
    let out = execute(&f, &json!({"sig": {"significant_text": {"field": "unmapped"}}}), &[0], None, false).unwrap();
    assert_eq!(out["sig"], json!({"doc_count": 0, "bg_count": 0, "buckets": []}));
    let mut budget = Budget { used: 0, limit: 8, mem: Default::default() };
    assert_eq!(budget.charge(9).unwrap_err().kind, "circuit_breaking_exception");
    assert_eq!(shard_size(json!({"size": 10}).as_object().unwrap()).unwrap(), 50);
    let random = json!({"random": {"random_sampler": {"probability": 0.1}, "aggs": {"sig": {"significant_text": {"field": "text"}}}}});
    assert!(crate::aggs::validate_request(&random).unwrap_err().reason.contains("random_sampler"));
}

#[test]
fn significant_text_disjoint_background_keeps_jlh_terms_absent_from_background() {
    let f = fixture(mapping(), &[
        json!({"text": "rare", "group": "foreground"}), json!({"text": "rare", "group": "foreground"}),
        json!({"text": "other", "group": "background"}), json!({"text": "other", "group": "background"}),
        json!({"text": "other", "group": "background"}), json!({"text": "other", "group": "background"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "text", "min_doc_count": 1, "background_filter": {"term": {"group": "background"}}}}});
    let out = execute(&f, &spec, &[0, 1], None, false).unwrap();
    assert_eq!(out["sig"]["buckets"], json!([{"key": "rare", "doc_count": 2, "bg_count": 0, "score": 3.0}]));
    let heuristic = Heuristic::parse(json!({"gnd": {"background_is_superset": false}}).as_object().unwrap()).unwrap();
    // The disjoint populations are added before computing normalized distance.
    let expected = (-((3f64.ln() - 2f64.ln()) / (12f64.ln() - 2f64.ln()))).exp();
    assert!((heuristic.score(2, 1, 2, 10).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn significant_text_alias_and_multifield_read_the_original_source_path() {
    let f = fixture(json!({"properties": {
        "text": {"type": "text", "fields": {"stem": {"type": "text", "analyzer": "english"}}},
        "alias": {"type": "alias", "path": "text.stem"}
    }}), &[json!({"text": "running"}), json!({"text": "runs"}), json!({"text": "other"}), json!({"text": "other"})]);
    let spec = |field| json!({"sig": {"significant_text": {"field": field, "min_doc_count": 1}}});
    let normal = execute(&f, &spec("text.stem"), &[0, 1], None, false).unwrap();
    let alias = execute(&f, &spec("alias"), &[0, 1], None, false).unwrap();
    assert_eq!(normal, alias);
    assert_eq!(normal["sig"]["buckets"][0]["key"], "run");
    assert_eq!(normal["sig"]["buckets"][0]["doc_count"], 2);
    let explicit_multifield = json!({"sig": {"significant_text": {"field": "alias", "source_fields": ["text.stem"], "min_doc_count": 1}}});
    assert_eq!(execute(&f, &explicit_multifield, &[0, 1], None, false).unwrap(), normal);
    let explicit = json!({"sig": {"significant_text": {"field": "alias", "source_fields": ["alias"], "min_doc_count": 1}}});
    assert_eq!(execute(&f, &explicit, &[0, 1], None, false).unwrap()["sig"]["buckets"], json!([]));
}

#[test]
fn significant_text_copy_to_reads_all_supplying_source_fields() {
    let f = fixture(json!({"properties": {
        "combined": {"type": "text"}, "first": {"type": "text", "copy_to": "combined"},
        "second": {"type": "text", "copy_to": "combined"}
    }}), &[
        json!({"first": "alpha", "second": "alpha beta"}), json!({"first": "alpha beta"}),
        json!({"combined": "other"}), json!({"combined": "other"}),
    ]);
    let spec = json!({"sig": {"significant_text": {"field": "combined", "min_doc_count": 1, "include": ["alpha", "beta"]}}});
    let out = execute(&f, &spec, &[0, 1], None, false).unwrap();
    assert_eq!(out["sig"]["buckets"], json!([
        {"key": "alpha", "doc_count": 2, "bg_count": 2, "score": 1.0},
        {"key": "beta", "doc_count": 2, "bg_count": 2, "score": 1.0}
    ]));
}

#[test]
fn selected_source_retains_dotted_arrays_and_skips_unrelated_trees() {
    let raw = r#"{"text":["alpha","beta"],"nested":[{"text":"gamma","other":42}],"nested.text":"delta","unrelated":{"large":[1,2,3]},"escaped\u002ekey":"epsilon"}"#;
    let paths = vec!["text".into(), "nested.text".into(), "escaped.key".into()];
    let selected = select_source(raw, &paths).unwrap();
    let full: Value = serde_json::from_str(raw).unwrap();
    for path in &paths {
        assert_eq!(crate::highlight::source_values(&selected, path), crate::highlight::source_values(&full, path));
    }
    assert!(selected.get("unrelated").is_none());
    assert!(select_source("{bad}", &paths).is_err());
    assert!(select_source(r#"{"text":"alpha"} trailing"#, &paths).is_err());
}

#[test]
fn selected_source_filters_unrelated_utf16_before_decoding() {
    let raw = r#"{"obj":{"text":"alpha","other":"\ud800"},"extra":"\udc00"}"#;
    let out = select_source(raw, &["obj.text".into()]).unwrap();
    assert_eq!(crate::highlight::source_values(&out, "obj.text"), vec![&json!("alpha")]);
    assert!(select_source(raw, &["obj.other".into()]).is_err());
    assert!(select_source(raw, &["obj.*".into()]).is_err());
}

#[test]
fn jlh_preserves_reference_operation_order() {
    let h = Heuristic::parse(&Map::new()).unwrap();
    // ES 9.5.5 GeoNames response: multiplication after relative probability,
    // not the algebraically equivalent left-associated multiply/divide.
    assert_eq!(h.score(10, 29, 242, 1_139_651).unwrap(), 67.0618645325244);
    assert_eq!(h.score(11_141, 15_099, 369_805, 1_139_651).unwrap(), 0.0383789758494411);
}
