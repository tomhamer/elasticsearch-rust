use elasticsearch_engine::{index::WriteReq, node::Node, search::{self, ExecOpts}};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};

fn put(node: &Arc<Node>, id: &str, source: Value) {
    node.get_index("test").unwrap().write(vec![WriteReq::Index {
        id: Some(id.into()), source: source.to_string(), routing: None, cond: Default::default(),
    }], false).remove(0).unwrap();
}

fn search(node: &Arc<Node>, query: Value) -> Value {
    search_request(node, json!({"size": 100, "_source": false, "query": query}))
}

fn search_request(node: &Arc<Node>, body: Value) -> Value {
    node.get_index("test").unwrap().refresh().unwrap();
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("test", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap()
}

#[test]
fn bounding_box_includes_quantized_corners_and_excludes_adjacent_cells() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"point":{"type":"geo_point"}}}})).unwrap();
    for (id, lat, lon) in [("bottom-left", -40.0, -40.0), ("bottom-right", -40.0, 40.0),
                          ("top-left", 40.0, -40.0), ("top-right", 40.0, 40.0),
                          ("below", -40.000001, 0.0), ("left", 0.0, -40.000001),
                          ("above", 40.000001, 0.0), ("right", 0.0, 40.000001)] {
        put(&node, id, json!({"point":{"lat":lat,"lon":lon}}));
    }
    let result = search(&node, json!({"geo_bounding_box":{"point":{"top":40,"bottom":-40,"left":-40,"right":40}}}));
    assert_eq!(result["hits"]["total"]["value"], 4);
    assert!(result["hits"]["hits"].as_array().unwrap().iter().all(|h| h["_id"].as_str().unwrap().contains('-')));
    let point = search(&node, json!({"geo_bounding_box":{"point":{"top":-40,"bottom":-40,"left":-40,"right":-40}}}));
    assert_eq!(point["hits"]["total"]["value"], 1);
    assert_eq!(point["hits"]["hits"][0]["_id"], "bottom-left");
}

#[test]
fn percolator_slots_follow_original_document_positions_and_single_query_name() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"body":{"type":"text"},"query":{"type":"percolator"}}}})).unwrap();
    put(&node, "alpha", json!({"query":{"term":{"body":"alpha"}}}));
    put(&node, "beta", json!({"query":{"term":{"body":"beta"}}}));
    let result = search(&node, json!({"percolate":{"field":"query","name":"named",
        "documents":[{"body":"gamma"},{"body":"alpha"},{"body":"alpha beta"}]}}));
    assert_eq!(result["hits"]["total"]["value"], 2);
    for hit in result["hits"]["hits"].as_array().unwrap() {
        let expected = if hit["_id"] == "alpha" { json!([1,2]) } else { json!([2]) };
        assert_eq!(hit["fields"]["_percolator_document_slot"], expected);
        assert!(hit["fields"].get("_percolator_document_slot_named").is_none());
    }
    let result = search(&node, json!({"bool":{"should":[
        {"percolate":{"field":"query","name":"a","document":{"body":"alpha"}}},
        {"percolate":{"field":"query","name":"b","document":{"body":"beta"}}}
    ]}}));
    for hit in result["hits"]["hits"].as_array().unwrap() {
        let key = if hit["_id"] == "alpha" { "_percolator_document_slot_a" } else { "_percolator_document_slot_b" };
        assert_eq!(hit["fields"][key], json!([0]));
        assert_eq!(hit["fields"].as_object().unwrap().len(), 1);
    }
}

#[test]
fn percolator_pruning_preserves_unextractable_queries_and_punctuation() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "body":{"type":"text"},"tag":{"type":"keyword"},"query":{"type":"percolator"}
    }}})).unwrap();
    for (id, query) in [
        ("all", json!({"match_all":{}})),
        ("negative", json!({"bool":{"must_not":{"term":{"body":"absent"}}}})),
        ("other-field", json!({"term":{"tag":"present"}})),
        ("alpha", json!({"term":{"body":"alpha"}})),
        ("absent", json!({"term":{"body":"absent"}})),
    ] {
        put(&node, id, json!({"query":query}));
    }
    for body in ["alpha beta", "alpha-beta"] {
        let result = search(&node, json!({"percolate":{"field":"query",
            "document":{"body":body,"tag":"present"}}}));
        let mut ids: Vec<_> = result["hits"]["hits"].as_array().unwrap().iter()
            .map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        assert_eq!(ids, ["all", "alpha", "negative", "other-field"]);
    }
}

#[test]
fn percolator_candidates_use_analyzed_and_copied_terms() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "body":{"type":"text","analyzer":"english"},
        "extra":{"type":"text","copy_to":"body"},"query":{"type":"percolator"}
    }}})).unwrap();
    put(&node, "run", json!({"query":{"term":{"body":"run"}}}));
    for document in [json!({"body":"running"}), json!({"body":"alpha","extra":"running"})] {
        let result = search(&node, json!({"percolate":{"field":"query","document":document}}));
        assert_eq!(result["hits"]["total"]["value"], 1);
        assert_eq!(result["hits"]["hits"][0]["_id"], "run");
    }
}

#[test]
fn field_sort_preserves_explicit_score_tracking() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "body":{"type":"text"},"order":{"type":"long"}
    }}})).unwrap();
    put(&node, "a", json!({"body":"alpha alpha","order":1}));
    put(&node, "b", json!({"body":"alpha beta","order":2}));
    let query = json!({"term":{"body":"alpha"}});
    let unsorted = search(&node, query.clone());
    let expected: HashMap<_, _> = unsorted["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect();
    for track in [false, true] {
        let result = search_request(&node, json!({"query":query,"sort":["order"],"track_scores":track}));
        let hits = result["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits[0]["_id"], "a");
        for hit in hits {
            if track {
                let score = hit["_score"].as_f64().unwrap();
                assert!(score > 0.0);
                assert!((score - expected[hit["_id"].as_str().unwrap()]).abs() < 1e-6);
            } else {
                assert!(hit["_score"].is_null());
            }
        }
        assert_eq!(result["hits"]["max_score"].is_null(), !track);
    }
}

#[test]
fn percolator_analyzed_match_candidates_preserve_slots_and_fallbacks() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "body":{"type":"text","analyzer":"english"},"query":{"type":"percolator"}
    }}})).unwrap();
    for (id, query) in [
        ("or", json!({"match":{"body":"running jumping"}})),
        ("and", json!({"match":{"body":{"query":"running jumping","operator":"and"}}})),
        ("duplicate-a", json!({"match":{"body":"running"}})),
        ("duplicate-b", json!({"match":{"body":"running"}})),
        ("all", json!({"match":{"body":{"query":"the","zero_terms_query":"all"}}})),
        ("fuzzy", json!({"match":{"body":{"query":"runn","fuzziness":1}}})),
        ("absent", json!({"match":{"body":"sleeping"}})),
    ] { put(&node, id, json!({"query":query})); }
    node.get_index("test").unwrap().force_merge(1, false).unwrap();
    let lookup = node.get_index("test").unwrap().lookup();
    let extracted = elasticsearch_engine::fieldtypes::percolate::extract_terms(
        &json!({"match":{"body":"running jumping"}}), &lookup);
    assert!(extracted.iter().any(|t| t.ends_with("body:run")), "{extracted:?}");
    assert!(extracted.iter().any(|t| t.ends_with("body:jump")), "{extracted:?}");
    assert!(!extracted.iter().any(|t| t.contains("fallback")));
    let result = search(&node, json!({"percolate":{"field":"query","documents":[
        {"body":"running"}, {"body":"jumping"}
    ]}}));
    let hits: HashMap<_,_> = result["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap(), h["fields"]["_percolator_document_slot"].clone())).collect();
    assert_eq!(hits.len(), 5, "{result}");
    assert_eq!(hits["or"], json!([0,1]));
    assert_eq!(hits["all"], json!([0,1]));
    for id in ["duplicate-a", "duplicate-b", "fuzzy"] { assert_eq!(hits[id], json!([0])); }
    // A separate request must not reuse scores/slots from the previous documents.
    let next = search(&node, json!({"percolate":{"field":"query","document":{"body":"sleeping"}}}));
    let mut ids: Vec<_> = next["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
    ids.sort();
    assert_eq!(ids, ["absent", "all"]);
}

#[test]
fn mapping_absence_guards_follow_rebuilds_and_filtered_views() {
    use elasticsearch_engine::{analysis::AnalysisRegistry, mapping::{FieldLookup, Mappings}, vector_source};
    let mut registry = AnalysisRegistry::default();
    registry.exclude_source_vectors = true;
    let plain = Mappings::parse(&json!({"properties":{"value":{"type":"long"}}}), &registry).unwrap();
    let plain_lookup = FieldLookup::build(&plain, &registry);
    assert!(!plain_lookup.may_have_vector_fields);
    assert!(!plain_lookup.may_have_semantic_fields);
    assert!(elasticsearch_engine::inference::semantic::infer_document(&json!({"value":1}), &plain_lookup, false).unwrap().is_empty());
    assert!(!vector_source::has_synthetic_vectors(&plain_lookup));
    for (kind, params, synthetic) in [
        ("dense_vector", json!({"dims":3}), true),
        ("dense_vector", json!({"dims":3,"index":false}), false),
        ("sparse_vector", json!({}), true),
        ("sparse_vector", json!({"store":false}), false),
        ("rank_vectors", json!({"dims":3}), true),
    ] {
        let mut field = params;
        field["type"] = json!(kind);
        let mapping = Mappings::parse(&json!({"properties":{"embedding":field}}), &registry).unwrap();
        let mut lookup = FieldLookup::build(&mapping, &registry);
        assert!(lookup.may_have_vector_fields);
        assert!(vector_source::has_vector_fields(&lookup));
        assert_eq!(vector_source::has_synthetic_vectors(&lookup), synthetic, "{kind}");
        // Search field-security views remove fields after lookup construction.
        // A stale true flag must not expose a filtered field.
        lookup.fields.clear();
        assert!(!vector_source::has_vector_fields(&lookup));
        assert!(!vector_source::has_synthetic_vectors(&lookup));
    }
    let semantic = Mappings::parse(&json!({"properties":{"body":{"type":"semantic_text","inference_id":"test"}}}), &registry).unwrap();
    assert!(FieldLookup::build(&semantic, &registry).may_have_semantic_fields);
    assert!(!FieldLookup::build(&plain, &registry).may_have_semantic_fields);
}

#[test]
fn prepared_date_counts_match_general_collector() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "ts":{"type":"date"},"value":{"type":"long"}
    }}})).unwrap();
    // More than the preparation threshold, unsorted values, pre-epoch dates,
    // leap years and both US DST transitions; plus missing values.
    for i in 0..5000 {
        let ts = if i % 7 == 0 { -86_400_000i64 } else { 1_704_067_200_000 + ((i * 7919) % 527040) as i64 * 60_000 };
        put(&node, &i.to_string(), json!({"ts":ts,"value":1}));
    }
    put(&node, "absent", json!({"value":1}));
    node.get_index("test").unwrap().force_merge(1, false).unwrap();
    for zone in ["UTC", "America/New_York"] {
        for interval in [json!({"calendar_interval":"month"}), json!({"calendar_interval":"year"}), json!({"fixed_interval":"30d"})] {
            let mut config = interval;
            config["field"] = json!("ts");
            config["time_zone"] = json!(zone);
            config["offset"] = json!("6h");
            let actual = search_request(&node, json!({"size":0,"aggs":{"a":{"date_histogram":config}}}));
            let expected = search_request(&node, json!({"size":0,"aggs":{"a":{
                "date_histogram":config,"aggs":{"force_general":{"sum":{"field":"value"}}}
            }}}));
            let mut expected = expected["aggregations"]["a"].clone();
            for bucket in expected["buckets"].as_array_mut().unwrap() { bucket.as_object_mut().unwrap().remove("force_general"); }
            assert_eq!(actual["aggregations"]["a"], expected, "{config}");
            assert_eq!(expected["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum::<u64>(), 5000, "{config}: {expected}");
        }
    }
}

#[test]
fn auto_date_scalar_counts_match_general_collector_and_fallbacks() {
    for scenario in ["scalar", "multi", "weighted", "missing"] {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("test", &json!({"mappings":{"properties":{
            "ts":{"type":"date"},"value":{"type":"long"}
        }}})).unwrap();
        // Include both sides of US DST changes and a gap between months.
        for (i, date) in ["2024-03-10T06:59:00Z", "2024-03-10T07:01:00Z",
                          "2024-11-03T05:59:00Z", "2024-11-03T06:01:00Z"].iter().enumerate() {
            let mut source = json!({"ts":date,"value":1});
            if scenario == "multi" { source["ts"] = json!([date,date]); }
            if scenario == "weighted" { source["_doc_count"] = json!(3); }
            put(&node, &i.to_string(), source);
        }
        put(&node, "absent", json!({"value":1}));
        node.get_index("test").unwrap().force_merge(1, false).unwrap();
        for zone in ["UTC", "America/New_York"] {
            for buckets in [2, 10, 100] {
                let mut config = json!({"field":"ts","buckets":buckets,"time_zone":zone});
                if scenario == "missing" { config["missing"] = json!("2024-03-10T07:01:00Z"); }
                let actual = search_request(&node, json!({"size":0,"aggs":{"a":{"auto_date_histogram":config}}}));
                let expected = search_request(&node, json!({"size":0,"aggs":{"a":{
                    "auto_date_histogram":config,"aggs":{"force_general":{"sum":{"field":"value"}}}
                }}}));
                let mut expected = expected["aggregations"]["a"].clone();
                for bucket in expected["buckets"].as_array_mut().unwrap() { bucket.as_object_mut().unwrap().remove("force_general"); }
                assert_eq!(actual["aggregations"]["a"], expected, "{scenario}/{zone}/{buckets}");
                let count: u64 = expected["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
                assert_eq!(count, match scenario { "weighted" => 12, "missing" => 5, _ => 4 });
            }
        }
    }
}

/// Rally metricbeat's `auto-date-histogram(-with-tz)`: a bare one-day
/// range on `@timestamp` and a count-only `auto_date_histogram` of it, whose
/// matches the search leaves uncollected and counts from the points index.
/// Buckets, interval and hits equal the collected path's (a sub-aggregation
/// forces it) over several segments, one with a deletion (collected), a
/// doc without the field, and a `bool` filter wrapper.
#[test]
fn auto_date_of_bare_range_matches_collected_path() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{
        "@timestamp":{"type":"date"},"value":{"type":"long"}
    }}})).unwrap();
    // 2019-02-22T12:00Z .. 2019-02-24T12:00Z, a minute apart, three segments.
    let start = 1_550_836_800_000i64;
    for batch in 0..3i64 {
        for i in 0..1000i64 {
            let n = batch * 1000 + i;
            put(&node, &n.to_string(), json!({"@timestamp": start + (n * 7_919 % 2_880) * 60_000, "value": 1}));
        }
        node.get_index("test").unwrap().refresh().unwrap();
    }
    put(&node, "absent", json!({"value":1}));
    node.get_index("test").unwrap().refresh().unwrap();
    node.get_index("test").unwrap().write(vec![WriteReq::Delete { id: "1500".into(), routing: None, cond: Default::default() }], false).remove(0).unwrap();
    let day = 1_550_880_000_000i64..1_550_966_400_000;
    let in_day = (0..3000i64).filter(|&n| n != 1500 && day.contains(&(start + (n * 7_919 % 2_880) * 60_000))).count() as u64;
    let range = json!({"range":{"@timestamp":{"gte":"23/02/2019","lte":"23/02/2019","format":"dd/MM/yyyy"}}});
    for query in [range.clone(), json!({"bool":{"filter":[range.clone()]}}), json!({"constant_score":{"filter":range.clone()}})] {
        for zone in [None, Some("America/New_York"), Some("Asia/Kolkata")] {
            for buckets in [24, 3, 200] {
                let mut config = json!({"field":"@timestamp","buckets":buckets});
                if let Some(z) = zone { config["time_zone"] = json!(z); }
                let actual = search_request(&node, json!({"size":0,"query":query,"aggs":{"a":{"auto_date_histogram":config}}}));
                let expected = search_request(&node, json!({"size":0,"query":query,"aggs":{"a":{
                    "auto_date_histogram":config,"aggs":{"force_general":{"sum":{"field":"value"}}}
                }}}));
                assert_eq!(actual["hits"], expected["hits"], "{query} {config}");
                let mut want = expected["aggregations"]["a"].clone();
                for bucket in want["buckets"].as_array_mut().unwrap() { bucket.as_object_mut().unwrap().remove("force_general"); }
                assert_eq!(actual["aggregations"]["a"], want, "{query} {config}");
                let count: u64 = want["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
                assert_eq!(count, in_day, "{query} {config}");
            }
        }
    }
}

/// `cargo test -p elasticsearch-engine --release --test perf_correctness auto_date_bare_range_microbench -- --ignored --nocapture`
/// (`AUTO_DATE_BENCH_DOCS`, default 1M over three days, force-merged to
/// one segment): Rally metricbeat's `auto-date-histogram(-with-tz)` search
/// (a bare one-day range, matches left uncollected) against the same
/// range beside an `exists` filter (matches collected, then counted from
/// the points index as before), best of 30.
#[test]
#[ignore]
fn auto_date_bare_range_microbench() {
    let n: i64 = std::env::var("AUTO_DATE_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"@timestamp":{"type":"date"},"value":{"type":"long"}}}})).unwrap();
    let start = 1_550_836_800_000i64;
    let span = 3 * 86_400_000i64;
    let idx = node.get_index("test").unwrap();
    for chunk in (0..n).collect::<Vec<_>>().chunks(10_000) {
        let reqs = chunk.iter().map(|&i| WriteReq::Index {
            id: Some(i.to_string()), source: json!({"@timestamp": start + (i * 7_919_993) % span, "value": 1}).to_string(), routing: None, cond: Default::default(),
        }).collect();
        for r in idx.write(reqs, false) { r.unwrap(); }
    }
    idx.refresh().unwrap();
    idx.force_merge(1, false).unwrap();
    let range = json!({"range":{"@timestamp":{"gte":"23/02/2019","lte":"23/02/2019","format":"dd/MM/yyyy"}}});
    for zone in [None, Some("America/New_York")] {
        let mut config = json!({"field":"@timestamp","buckets":24});
        if let Some(z) = zone { config["time_zone"] = json!(z); }
        for (label, query) in [("bare range (uncollected)", range.clone()), ("range + exists (collected)", json!({"bool":{"filter":[range.clone(), {"exists":{"field":"value"}}]}}))] {
            let body = json!({"size":0,"query":query,"aggs":{"a":{"auto_date_histogram":config}}});
            let first = search_request(&node, body.clone());
            let best = (0..30).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(search_request(&node, body.clone())); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min);
            eprintln!("tz={:<18} {label:<32} {:>8.3} ms  hits={}", zone.unwrap_or("-"), best * 1e3, first["hits"]["total"]);
        }
    }
}

#[test]
fn new_replica_writer_batch_preserves_metadata_and_rejects_overwrites() {
    use elasticsearch_engine::shard::ReplayIndex;
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("test", &json!({"mappings":{"_source":{"excludes":["secret"]},"properties":{
        "n":{"type":"long"},"secret":{"type":"keyword"},"tag":{"type":"keyword","store":true,"ignore_above":4}
    }}})).unwrap();
    let docs: Vec<_> = (0..256).map(|i| (i.to_string(),json!({"n":i,"secret":"hidden","tag":"too-long"}).to_string())).collect();
    let refs: Vec<_> = docs.iter().map(|(id,src)|(id.as_str(),src.as_str())).collect();
    let metadata: Vec<_> = docs.iter().enumerate().map(|(i,(id,source))|ReplayIndex{
        id,source,routing:Some("route"),seq_no:i as i64,version:7,
    }).collect();
    let shard = svc.shard(0).unwrap();
    let mut prepared = svc.prepare_replica_sources(&refs).unwrap();
    let mut duplicates: Vec<_> = metadata.iter().map(|m| ReplayIndex{id:metadata[0].id,source:m.source,routing:m.routing,seq_no:m.seq_no,version:m.version}).collect();
    assert!(!shard.try_replay_new_batch(&duplicates,&mut prepared).unwrap());
    assert_eq!(prepared.len(),256);
    assert_eq!(shard.doc_count_realtime().0,0);
    duplicates.clear();
    assert!(shard.try_replay_new_batch(&metadata,&mut prepared).unwrap());
    assert!(prepared.is_empty());
    assert_eq!(shard.shard.local_checkpoint(),255);
    for i in 0..256 {
        let got = shard.shard.get(&i.to_string()).unwrap().unwrap();
        assert_eq!((got.seq_no,got.version,got.routing.as_deref()),(i,7,Some("route")));
        let source: Value = serde_json::from_str(got.source.as_ref().unwrap()).unwrap();
        assert!(source.get("secret").is_none());
        assert_eq!(source["n"],json!(i));
        assert_eq!(elasticsearch_engine::stored::values_of(&got.doc,"_ignored"),vec![json!("tag")]);
    }
    let mut prepared = svc.prepare_replica_sources(&refs).unwrap();
    assert!(!shard.try_replay_new_batch(&metadata,&mut prepared).unwrap());
    assert_eq!(prepared.len(),256);
    shard.shard.flush().unwrap();
    assert_eq!(shard.doc_count_realtime().0,256);
    // Existing IDs must still reject the append path after sealing, when
    // the optional negative filter is active and version-map entries clear.
    let mut prepared = svc.prepare_replica_sources(&refs).unwrap();
    assert!(!shard.try_replay_new_batch(&metadata,&mut prepared).unwrap());
    assert_eq!(prepared.len(),256);
    let new_docs: Vec<_> = (256..512).map(|i| (i.to_string(),json!({"n":i}).to_string())).collect();
    let new_refs: Vec<_> = new_docs.iter().map(|(id,src)|(id.as_str(),src.as_str())).collect();
    let new_meta: Vec<_> = new_docs.iter().enumerate().map(|(i,(id,source))|ReplayIndex{
        id,source,routing:None,seq_no:(i+256) as i64,version:1,
    }).collect();
    let mut prepared = svc.prepare_replica_sources(&new_refs).unwrap();
    assert!(shard.try_replay_new_batch(&new_meta,&mut prepared).unwrap());
    shard.shard.flush().unwrap();
    assert_eq!(shard.doc_count_realtime().0,512);
}

#[test]
fn replica_preparation_is_read_only_and_matches_serial_parsing() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("test", &json!({"mappings":{"properties":{
        "n":{"type":"long"},"tag":{"type":"keyword","ignore_above":4,"store":true}
    }}})).unwrap();
    let mut docs: Vec<_> = (0..64).map(|i| (i.to_string(), json!({"n":i,"tag":"too-long"}).to_string())).collect();
    let refs: Vec<_> = docs.iter().map(|(id, src)| (id.as_str(),src.as_str())).collect();
    let prepared = svc.prepare_replica_sources(&refs).unwrap();
    for ((id, source), actual) in docs.iter().zip(prepared) {
        let expected = svc.parse_for_replica(source, id).unwrap();
        assert_eq!(serde_json::to_value(actual.doc).unwrap(), serde_json::to_value(expected.doc).unwrap());
        assert_eq!(actual.ignored, expected.ignored);
        assert_eq!(actual.source, expected.source);
    }
    docs[10].1 = json!({"new_field":1}).to_string();
    let refs: Vec<_> = docs.iter().map(|(id, src)| (id.as_str(),src.as_str())).collect();
    assert!(svc.prepare_replica_sources(&refs).is_none());
    assert!(svc.lookup().get("new_field").is_none());
    assert_eq!(svc.doc_count_realtime().0, 0);
    docs[10].1 = json!({"n":"invalid"}).to_string();
    let refs: Vec<_> = docs.iter().map(|(id, src)| (id.as_str(),src.as_str())).collect();
    assert!(svc.prepare_replica_sources(&refs).is_none());
}

#[test]
fn pure_score_script_bulk_paths_match_interpreted_hits_before_and_after_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("test", &json!({"mappings":{"properties":{
        "population":{"type":"long"},"location":{"type":"geo_point"}
    }}})).unwrap();
    let docs = (0..5003).map(|doc| WriteReq::Index { id: Some(doc.to_string()), routing: None, cond: Default::default(),
        source: json!({"population":(doc as u64 * 1234567) % 100000,
                      "location":{"lat":(doc % 179) as f64 - 89.0,"lon":((doc * 1237) % 359) as f64 - 179.0}}).to_string() }).collect();
    assert!(svc.write(docs, false).iter().all(Result::is_ok));
    for sealed in [false, true] {
        if sealed { svc.shard(0).unwrap().shard.flush().unwrap(); }
        for script in ["Math.log10(doc['population'].value * 1.2 + 2)",
                       "decayGeoGauss('52.37, 4.8951', '500km', '0km', 0.1, doc['location'].value)"] {
            let request = |source: &str| json!({"size":100,"track_total_hits":true,"_source":false,
                "query":{"script_score":{"query":{"match_all":{}},"script":{"source":source}}}});
            let fast = search_request(&node, request(script));
            // Adding zero preserves scores but deliberately defeats the pure-call plan.
            let slow = search_request(&node, request(&format!("({script}) + 0.0")));
            assert_eq!(fast["hits"], slow["hits"], "sealed={sealed} script={script}");
        }
        svc.write(vec![WriteReq::Delete { id:"7".into(),routing:None,cond:Default::default() }], false).remove(0).unwrap();
    }
}

#[test]
fn numeric_script_pruning_does_not_hide_missing_or_invalid_values() {
    for bad in [json!({}), json!({"population":-100})] {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("test", &json!({"mappings":{"properties":{"population":{"type":"long"}}}})).unwrap();
        for i in 0..20 { put(&node, &i.to_string(), json!({"population":100000})); }
        put(&node, "bad", bad);
        svc.refresh().unwrap();
        let body = json!({"size":1,"query":{"script_score":{"query":{"match_all":{}},"script":{"source":"Math.log10(doc['population'].value * 1.2 + 2)"}}}});
        let req = search::parse_request(&body, &HashMap::new()).unwrap();
        let targets = node.resolve("test", Default::default()).unwrap();
        let opts = ExecOpts { terms_lookup:None,cursors:None,max_buckets:65536,scroll_id:None,pit_id:None };
        assert!(search::execute(&targets, &req, &opts).is_err());
    }
}

#[test]
fn compound_expression_bounds_preserve_scores_ties_deletes_and_fallbacks() {
    for tied in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("test", &json!({"mappings":{"properties":{
            "population":{"type":"long"},"location":{"type":"geo_point"}
        }}})).unwrap();
        let docs = (0..10003).map(|doc| {
            let population = match doc % 11 { 0 => json!(i64::MIN), 1 => json!(i64::MAX), 2 => json!([-23,500]), _ => json!(1234) };
            let mut source = if tied { json!({"population":1234,"location":[90,45]}) } else {
                json!({"population":population,"location":[((doc * 1237) % 359) as f64 - 179.0,(doc % 179) as f64 - 89.0]})
            };
            if !tied && doc % 13 == 0 { source.as_object_mut().unwrap().remove("population"); }
            WriteReq::Index { id:Some(doc.to_string()),routing:None,cond:Default::default(),source:source.to_string() }
        }).collect();
        assert!(svc.write(docs, false).iter().all(Result::is_ok));
        let script = "abs(ln(abs(doc['population']) + 1) + doc['location'].lon + doc['location'].lat) * _score";
        for phase in 0..3 {
            if phase == 1 {
                svc.shard(0).unwrap().shard.flush().unwrap();
                assert!(svc.write((0..10003).step_by(7).map(|doc| WriteReq::Delete {
                    id:doc.to_string(),routing:None,cond:Default::default()
                }).collect(),false).iter().all(Result::is_ok));
            }
            if phase == 2 {
                put(&node,"multiple",json!({"population":123,"location":[[179,89],[-179,-89]]}));
                svc.force_merge(1,false).unwrap();
            }
            for k in [1,10,100] {
                let request = |source:&str| json!({"size":k,"track_total_hits":true,"_source":false,
                    "query":{"function_score":{"query":{"match_all":{}},"functions":[{"script_score":{"script":{"source":source,"lang":"expression"}}}]}}});
                let fast = search_request(&node,request(script));
                let slow = search_request(&node,request(&format!("({script}) + 0.0")));
                assert_eq!(fast["hits"],slow["hits"],"phase={phase} tied={tied} k={k}");
            }
        }
    }
}

#[test]
fn painless_compound_bounds_match_original_with_narrowing_and_deletes() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("test", &json!({"mappings":{"properties":{
        "population":{"type":"long"},"location":{"type":"geo_point"}
    }}})).unwrap();
    assert!(svc.write((0..10003).map(|doc| WriteReq::Index {
        id:Some(doc.to_string()), routing:None, cond:Default::default(),
        source:json!({"population":if doc%97==0 {6814400000i64} else {doc as i64-5000},
            "location":[((doc*1237)%359) as f64-179.0,(doc%179) as f64-89.0]}).to_string()
    }).collect(),false).iter().all(Result::is_ok));
    let scripts = [
        "Math.abs(Math.log(Math.abs(doc['population'].value) + 1) + doc['location'].lon * doc['location'].lat)/_score",
        "Math.abs(Math.log(Math.abs((int)((List)doc.population).get(0)) + 1) + (double)(doc.location.lon) * (double)(doc.location.lat))/_score",
    ];
    for phase in 0..3 {
        if phase == 1 { svc.shard(0).unwrap().shard.flush().unwrap(); }
        if phase == 2 {
            assert!(svc.write((0..10003).step_by(7).map(|doc| WriteReq::Delete {id:doc.to_string(),routing:None,cond:Default::default()}).collect(),false).iter().all(Result::is_ok));
        }
        for script in scripts {
            for k in [1,10,100] {
                let request = |source:&str| json!({"size":k,"track_total_hits":true,"_source":false,
                    "query":{"function_score":{"query":{"match_all":{}},"script_score":{"script":{"source":source}}}}});
                let fast=search_request(&node,request(script));
                let slow=search_request(&node,request(&format!("({script})+0.0")));
                assert_eq!(fast["hits"],slow["hits"],"phase={phase} k={k} {script}");
            }
        }
    }
}

#[test]
fn painless_compound_pruning_preserves_missing_and_overflow_errors() {
    for population in [None,Some(i64::MAX),Some(i64::MIN),Some(i32::MAX as i64)] {
        let tmp=tempfile::tempdir().unwrap();
        let node=Node::open(tmp.path()).unwrap();
        let svc=node.create_index("test",&json!({"mappings":{"properties":{
            "population":{"type":"long"},"location":{"type":"geo_point"}
        }}})).unwrap();
        for doc in 0..300 {put(&node,&doc.to_string(),json!({"population":100,"location":[179,89]}));}
        let mut bad=json!({"location":[0,0]});
        if let Some(value)=population {bad["population"]=json!(value);}
        put(&node,"bad",bad);
        svc.refresh().unwrap();svc.shard(0).unwrap().shard.flush().unwrap();
        for script in [
            "Math.abs(Math.log(Math.abs(doc['population'].value) + 1) + doc['location'].lon * doc['location'].lat)/_score",
            "Math.abs(Math.log(Math.abs((int)((List)doc.population).get(0)) + 1) + (double)(doc.location.lon) * (double)(doc.location.lat))/_score",
        ] {
            let execute=|source:&str| {
                let body=json!({"size":1,"query":{"function_score":{"query":{"match_all":{}},"script_score":{"script":{"source":source}}}}});
                let req=search::parse_request(&body,&HashMap::new()).unwrap();
                let targets=node.resolve("test",Default::default()).unwrap();
                let opts=ExecOpts {terms_lookup:None,cursors:None,max_buckets:65536,scroll_id:None,pit_id:None};
                search::execute(&targets,&req,&opts).map(|r| serde_json::from_str::<Value>(&r.json).unwrap()["hits"].clone()).map_err(|e| format!("{}:{}",e.status,e.kind))
            };
            assert_eq!(execute(script),execute(&format!("({script})+0.0")),"population={population:?} {script}");
        }
    }
}

#[test]
fn bound_cosine_script_matches_general_executor_before_and_after_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"v":{"type":"dense_vector","dims":3,"index":false}}}})).unwrap();
    for (id, v) in [("a", [3.0,4.0,0.0]), ("b", [-3.0,4.0,1.0]), ("c", [1.0,0.0,0.0]), ("tie", [3.0,4.0,0.0])] {
        put(&node, id, json!({"v":v}));
    }
    for phase in 0..3 {
        let svc = node.get_index("test").unwrap();
        if phase == 1 { svc.flush().unwrap(); }
        if phase == 2 {
            svc.write(vec![WriteReq::Delete { id:"tie".into(), routing:None, cond:Default::default() }], false).remove(0).unwrap();
            put(&node, "c", json!({"v":[0.0,1.0,0.0]}));
        }
        for q in [[3.0,4.0,1.0], [-1.0,0.25,2.0]] {
            for k in [1,3,100] {
                let run = |source: &str| search_request(&node, json!({"size":k,"track_total_hits":true,"_source":false,
                    "query":{"script_score":{"query":{"match_all":{}},"boost":1.2,"script":{"source":source,"params":{"q":q}}}}}));
                let fast = run("cosineSimilarity(params.q, 'v') + 1.0");
                let slow = run("def s = cosineSimilarity(params.q, 'v'); return s + 1.0;");
                assert_eq!(fast["hits"], slow["hits"], "phase={phase} q={q:?} k={k}");
            }
        }
    }
}

#[test]
fn bound_cosine_preserves_missing_dimension_and_negative_score_errors() {
    for case in ["missing-value", "missing-field", "dimensions", "zero-query", "negative-score"] {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("test", &json!({"mappings":{"properties":{"v":{"type":"dense_vector","dims":3,"index":false}}}})).unwrap();
        put(&node, "valid", json!({"v":[1.0,0.0,0.0]}));
        if case == "missing-value" { put(&node, "missing", json!({})); }
        let query = match case {
            "dimensions" => json!([1.0,0.0]), "zero-query" => json!([0.0,0.0,0.0]),
            "negative-score" => json!([-1.0,0.0,0.0]), _ => json!([1.0,0.0,0.0]),
        };
        let field = if case == "missing-field" { "absent" } else { "v" };
        let offset = if case == "negative-score" { 0.0 } else { 1.0 };
        for flushed in [false,true] {
            if flushed { svc.flush().unwrap(); }
            svc.refresh().unwrap();
            for source in [format!("cosineSimilarity(params.q, '{field}') + {offset:.1}"),
                           format!("def s = cosineSimilarity(params.q, '{field}'); return s + {offset:.1};")] {
                let body = json!({"size":1,"query":{"script_score":{"query":{"match_all":{}},"script":{"source":source,"params":{"q":query}}}}});
                let req = search::parse_request(&body, &HashMap::new()).unwrap();
                let targets = node.resolve("test", Default::default()).unwrap();
                let opts = ExecOpts { terms_lookup:None,cursors:None,max_buckets:65536,scroll_id:None,pit_id:None };
                assert!(search::execute(&targets, &req, &opts).is_err(), "case={case} flushed={flushed} source={source}");
            }
        }
    }
}

#[test]
fn percolator_uses_memory_index_bm25_without_scaling_constant_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"body":{"type":"text"},"query":{"type":"percolator"}}}})).unwrap();
    put(&node, "term", json!({"query":{"match":{"body":"google"}}}));
    put(&node, "constant", json!({"query":{"constant_score":{"filter":{"match":{"body":"google"}},"boost":3}}}));
    let result = search(&node, json!({"percolate":{"field":"query","document":{"body":"google"}}}));
    let hits = result["hits"]["hits"].as_array().unwrap();
    let term = hits.iter().find(|h| h["_id"] == "term").unwrap();
    // ES9 MemoryIndex: ln(1 + 0.5 / 1.5) / (1 + 1.2).
    assert!((term["_score"].as_f64().unwrap() - 0.13076457).abs() < 1e-6, "{result}");
    assert_eq!(hits.iter().find(|h| h["_id"] == "constant").unwrap()["_score"], 3.0);
}

#[test]
fn percolator_highlights_submitted_documents_and_names_slots_without_source() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("test", &json!({"mappings":{"properties":{"body":{"type":"text"},"query":{"type":"percolator"}}}})).unwrap();
    put(&node, "q", json!({"query":{"match":{"body":"alpha"}}}));
    let result = search_request(&node, json!({"_source":false,"query":{"percolate":{"field":"query","document":{"body":"alpha beta"}}},"highlight":{"fields":{"body":{}}}}));
    assert_eq!(result["hits"]["hits"][0]["highlight"]["body"], json!(["<em>alpha</em> beta"]));
    assert!(result["hits"]["hits"][0].get("_source").is_none());
    let result = search_request(&node, json!({"_source":false,"query":{"bool":{"should":[
        {"percolate":{"field":"query","name":"a","documents":[{"body":"beta"},{"body":"alpha beta"}]}},
        {"percolate":{"field":"query","name":"b","document":{"body":"alpha"}}}
    ]}},"highlight":{"fields":{"body":{}}}}));
    let highlight = &result["hits"]["hits"][0]["highlight"];
    assert_eq!(highlight["a_1_body"], json!(["<em>alpha</em> beta"]));
    assert_eq!(highlight["b_body"], json!(["<em>alpha</em>"]));
    assert!(highlight.get("a_0_body").is_none());
}
