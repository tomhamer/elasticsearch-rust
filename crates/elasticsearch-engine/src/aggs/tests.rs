use super::*;
use crate::analysis::AnalysisRegistry;
use crate::document::parse_with_dynamic;
use crate::mapping::Mappings;
use rustic_core::types::IndexSettings;
use rustic_core::Index;

struct Fixture {
    index: Index,
    lookup: FieldLookup,
}

fn fixture(mapping: Value, docs: &[Value]) -> Fixture {
    let reg = AnalysisRegistry::default();
    let mut m = Mappings::parse(&mapping, &reg).unwrap();
    m.assign_core_ids();
    let lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    for (i, d) in docs.iter().enumerate() {
        let (pd, upd) = parse_with_dynamic(d, &i.to_string(), &m, &lookup, 1000).unwrap();
        assert!(upd.is_none(), "fixture mapping must be complete");
        index.add_document(&pd.doc).unwrap();
        if i == docs.len() / 2 {
            index.refresh().unwrap(); // two leaves: a segment and the memtable
        }
    }
    Fixture { index, lookup }
}

fn run(f: &Fixture, aggs: Value) -> Result<Value> {
    let leaves = f.index.leaves();
    let segs: Vec<AggSeg> = leaves
        .iter()
        .map(|l| AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: (0..l.num_docs()).collect(), scores: (0..l.num_docs()).map(|d| d as f32).collect() })
        .collect();
    let fetch = |s: usize, d: DocId, sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({"_index": "idx", "_id": format!("{s}-{d}"), "_score": sc})) };
    let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    run_aggregations(&aggs, &segs, &env)
}

fn sales() -> Fixture {
    fixture(
        json!({"properties": {
            "tag": {"type": "keyword"}, "price": {"type": "double"}, "qty": {"type": "long"},
            "ts": {"type": "date"}, "ok": {"type": "boolean"}, "ip": {"type": "ip"}, "title": {"type": "text"}
        }}),
        &[
            json!({"tag": "a", "price": 10.0, "qty": 1, "ts": "2020-01-01T10:00:00Z", "ok": true, "ip": "10.0.0.1", "title": "x"}),
            json!({"tag": "a", "price": 20.0, "qty": 2, "ts": "2020-01-01T12:00:00Z", "ok": false, "ip": "10.0.0.2"}),
            json!({"tag": "b", "price": 30.0, "qty": 3, "ts": "2020-01-03T00:00:00Z", "ok": true, "ip": "10.0.0.1"}),
            json!({"tag": ["b", "c"], "price": 40.0, "qty": 4, "ts": "2020-01-03T05:00:00Z"}),
            json!({"tag": "a", "qty": 5}),
        ],
    )
}

#[test]
fn terms_shapes_order_and_subaggs() {
    let f = sales();
    let r = run(&f, json!({"t": {"terms": {"field": "tag"}, "aggs": {"p": {"avg": {"field": "price"}}}}})).unwrap();
    assert_eq!(
        r,
        json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": [
            {"key": "a", "doc_count": 3, "p": {"value": 15.0}},
            {"key": "b", "doc_count": 2, "p": {"value": 35.0}},
            {"key": "c", "doc_count": 1, "p": {"value": 40.0}}
        ]}})
    );
    // size, sum_other, order by sub-agg, include/exclude, missing
    let r = run(&f, json!({"t": {"terms": {"field": "tag", "size": 1, "order": {"p": "desc"}}, "aggs": {"p": {"max": {"field": "price"}}}}})).unwrap();
    assert_eq!(r["t"]["buckets"][0]["key"], "b");
    assert_eq!(r["t"]["sum_other_doc_count"], 4);
    let r = run(&f, json!({"t": {"terms": {"field": "tag", "include": "a|b", "exclude": ["b"], "order": {"_key": "desc"}}}})).unwrap();
    assert_eq!(r["t"]["buckets"], json!([{"key": "a", "doc_count": 3}]));
    let r = run(&f, json!({"t": {"terms": {"field": "price", "missing": 0, "order": [{"_count": "desc"}, {"_key": "asc"}], "size": 2}}})).unwrap();
    assert_eq!(r["t"]["buckets"][0], json!({"key": 0.0, "doc_count": 1}));
    // boolean, date, ip, long keys
    let r = run(&f, json!({"b": {"terms": {"field": "ok"}}, "d": {"terms": {"field": "ts", "size": 1, "order": {"_key": "asc"}}}, "i": {"terms": {"field": "ip"}}, "q": {"terms": {"field": "qty", "size": 1}}})).unwrap();
    assert_eq!(r["b"]["buckets"][0], json!({"key": 1, "key_as_string": "true", "doc_count": 2}));
    assert_eq!(r["d"]["buckets"][0], json!({"key": 1577872800000i64, "key_as_string": "2020-01-01T10:00:00.000Z", "doc_count": 1}));
    assert_eq!(r["i"]["buckets"][0], json!({"key": "10.0.0.1", "doc_count": 2}));
    assert_eq!(r["q"]["buckets"][0], json!({"key": 1, "doc_count": 1}));
    // show_term_doc_count_error renders the per-bucket bound
    let r = run(&f, json!({"t": {"terms": {"field": "tag", "show_term_doc_count_error": true}}})).unwrap();
    assert_eq!(r["t"]["buckets"][0], json!({"key": "a", "doc_count": 3, "doc_count_error_upper_bound": 0}));
    // default (false) omits it
    let r = run(&f, json!({"t": {"terms": {"field": "tag"}}})).unwrap();
    assert!(r["t"]["buckets"][0].get("doc_count_error_upper_bound").is_none());
    // unmapped
    let r = run(&f, json!({"t": {"terms": {"field": "nope"}}})).unwrap();
    assert_eq!(r["t"]["buckets"], json!([]));
    // text field → fielddata error
    let e = run(&f, json!({"t": {"terms": {"field": "title"}}})).unwrap_err();
    assert!(e.reason.starts_with("Fielddata is disabled on [title]"));
    assert!(run(&f, json!({"t": {"terms": {"field": "tag", "size": 0}}})).is_err());
}

#[test]
fn metrics() {
    let f = sales();
    let r = run(
        &f,
        json!({
            "avg": {"avg": {"field": "price"}}, "sum": {"sum": {"field": "price"}}, "min": {"min": {"field": "qty"}}, "max": {"max": {"field": "ts"}},
            "vc": {"value_count": {"field": "tag"}}, "card": {"cardinality": {"field": "tag"}},
            "st": {"stats": {"field": "qty"}}, "none": {"avg": {"field": "nope"}}, "nsum": {"sum": {"field": "nope"}},
            "miss": {"avg": {"field": "price", "missing": 100}}
        }),
    )
    .unwrap();
    assert_eq!(r["avg"], json!({"value": 25.0}));
    assert_eq!(r["sum"], json!({"value": 100.0}));
    assert_eq!(r["min"], json!({"value": 1.0}));
    assert_eq!(r["max"], json!({"value": 1578027600000.0, "value_as_string": "2020-01-03T05:00:00.000Z"}));
    assert_eq!(r["vc"], json!({"value": 6}));
    assert_eq!(r["card"], json!({"value": 3}));
    assert_eq!(r["st"], json!({"count": 5, "min": 1.0, "max": 5.0, "avg": 3.0, "sum": 15.0}));
    assert_eq!(r["none"], json!({"value": null}));
    assert_eq!(r["nsum"], json!({"value": 0.0}));
    assert_eq!(r["miss"], json!({"value": 40.0}));
    let r = run(&f, json!({"es": {"extended_stats": {"field": "qty"}}})).unwrap();
    assert_eq!(r["es"]["variance"], json!(2.0));
    assert_eq!(r["es"]["sum_of_squares"], json!(55.0));
    assert_eq!(r["es"]["std_deviation_bounds"]["upper"], json!(3.0 + 2.0 * 2f64.sqrt()));
    let r = run(&f, json!({"p": {"percentiles": {"field": "qty", "percents": [50]}}, "pk": {"percentiles": {"field": "qty", "percents": [0, 100], "keyed": false}}, "pr": {"percentile_ranks": {"field": "qty", "values": [3]}}})).unwrap();
    assert_eq!(r["p"], json!({"values": {"50.0": 3.0}}));
    assert_eq!(r["pk"], json!({"values": [{"key": 0.0, "value": 1.0}, {"key": 100.0, "value": 5.0}]}));
    assert_eq!(r["pr"], json!({"values": {"3.0": 50.0}}));
    let r = run(&f, json!({"p": {"percentiles": {"field": "nope"}}})).unwrap();
    assert_eq!(r["p"]["values"]["99.0"], Value::Null);
}

#[test]
fn ranges_and_histograms() {
    let f = sales();
    let r = run(&f, json!({"r": {"range": {"field": "price", "ranges": [{"to": 20}, {"from": 20, "to": 35}, {"from": 35, "key": "hi"}]}}})).unwrap();
    assert_eq!(
        r["r"]["buckets"],
        json!([
            {"key": "*-20.0", "to": 20.0, "doc_count": 1},
            {"key": "20.0-35.0", "from": 20.0, "to": 35.0, "doc_count": 2},
            {"key": "hi", "from": 35.0, "doc_count": 1}
        ])
    );
    let r = run(&f, json!({"r": {"range": {"field": "qty", "keyed": true, "ranges": [{"from": 1, "to": 3}]}}})).unwrap();
    assert_eq!(r["r"]["buckets"], json!({"1.0-3.0": {"from": 1.0, "to": 3.0, "doc_count": 2}}));
    let r = run(&f, json!({"r": {"date_range": {"field": "ts", "ranges": [{"from": "2020-01-02", "to": "2020-01-04"}]}}})).unwrap();
    assert_eq!(
        r["r"]["buckets"][0],
        json!({"key": "2020-01-02T00:00:00.000Z-2020-01-04T00:00:00.000Z", "from": 1577923200000.0, "from_as_string": "2020-01-02T00:00:00.000Z", "to": 1578096000000.0, "to_as_string": "2020-01-04T00:00:00.000Z", "doc_count": 2})
    );
    // Plain `range` over a `date` field: bounds are date math (Kibana
    // WorkloadAggregator sends `from: "now"`), and the value source's
    // formatter renders `from_as_string` / `to_as_string`.
    let r = run(&f, json!({"r": {"range": {"field": "ts", "ranges": [{"from": "now", "to": "now+1h"}]}}})).unwrap();
    assert_eq!(
        r["r"]["buckets"][0],
        json!({"key": "1970-01-01T00:00:00.000Z-1970-01-01T01:00:00.000Z", "from": 0.0, "from_as_string": "1970-01-01T00:00:00.000Z", "to": 3600000.0, "to_as_string": "1970-01-01T01:00:00.000Z", "doc_count": 0})
    );
    let r = run(&f, json!({"r": {"range": {"field": "ts", "ranges": [{"from": "2020-01-02||+0d", "to": "2020-01-03||+1d"}]}}})).unwrap();
    assert_eq!(r["r"]["buckets"][0]["doc_count"], 2);
    // `value_type: date` on a numeric field also switches to date bounds.
    let r = run(&f, json!({"r": {"range": {"field": "qty", "value_type": "date", "ranges": [{"from": "now", "to": "now+1h"}]}}})).unwrap();
    assert_eq!(r["r"]["buckets"][0]["from_as_string"], "1970-01-01T00:00:00.000Z");
    let r = run(&f, json!({"h": {"histogram": {"field": "price", "interval": 15}}})).unwrap();
    assert_eq!(r["h"]["buckets"], json!([{"key": 0.0, "doc_count": 1}, {"key": 15.0, "doc_count": 1}, {"key": 30.0, "doc_count": 2}]));
    let r = run(&f, json!({"h": {"histogram": {"field": "price", "interval": 20, "min_doc_count": 1, "extended_bounds": {"min": 0, "max": 100}}}})).unwrap();
    assert_eq!(r["h"]["buckets"].as_array().unwrap().len(), 3);
    let r = run(&f, json!({"h": {"histogram": {"field": "price", "interval": 20, "extended_bounds": {"min": 0, "max": 80}}}})).unwrap();
    assert_eq!(r["h"]["buckets"].as_array().unwrap().len(), 5);
    let r = run(&f, json!({"d": {"date_histogram": {"field": "ts", "calendar_interval": "day"}}})).unwrap();
    assert_eq!(
        r["d"]["buckets"],
        json!([
            {"key_as_string": "2020-01-01T00:00:00.000Z", "key": 1577836800000i64, "doc_count": 2},
            {"key_as_string": "2020-01-02T00:00:00.000Z", "key": 1577923200000i64, "doc_count": 0},
            {"key_as_string": "2020-01-03T00:00:00.000Z", "key": 1578009600000i64, "doc_count": 2}
        ])
    );
    let r = run(&f, json!({"d": {"date_histogram": {"field": "ts", "fixed_interval": "12h", "min_doc_count": 1, "format": "yyyy-MM-dd HH", "time_zone": "+01:00"}}})).unwrap();
    assert_eq!(r["d"]["buckets"][0]["key_as_string"], "2020-01-01 00");
    assert_eq!(r["d"]["buckets"][1]["key_as_string"], "2020-01-01 12");
    let r = run(&f, json!({"d": {"date_histogram": {"field": "ts", "calendar_interval": "month", "time_zone": "Europe/Paris"}}})).unwrap();
    assert_eq!(r["d"]["buckets"], json!([{"key_as_string": "2020-01-01T00:00:00.000+01:00", "key": 1577833200000i64, "doc_count": 4}]));
    assert!(run(&f, json!({"d": {"date_histogram": {"field": "ts", "calendar_interval": "2d"}}})).is_err());
}

#[test]
fn single_bucket_and_filters() {
    let f = sales();
    let r = run(
        &f,
        json!({
            "f": {"filter": {"term": {"tag": "a"}}, "aggs": {"s": {"sum": {"field": "qty"}}}},
            "fs": {"filters": {"filters": {"aa": {"term": {"tag": "a"}}, "bb": {"range": {"price": {"gte": 30}}}}, "other_bucket_key": "rest"}},
            "fa": {"filters": {"filters": [{"term": {"tag": "c"}}]}},
            "m": {"missing": {"field": "price"}},
            "g": {"global": {}, "meta": {"x": 1}}
        }),
    )
    .unwrap();
    assert_eq!(r["f"], json!({"doc_count": 3, "s": {"value": 8.0}}));
    assert_eq!(r["fs"], json!({"buckets": {"aa": {"doc_count": 3}, "bb": {"doc_count": 2}, "rest": {"doc_count": 0}}}));
    assert_eq!(r["fa"], json!({"buckets": [{"doc_count": 1}]}));
    assert_eq!(r["m"], json!({"doc_count": 1}));
    assert_eq!(r["g"], json!({"meta": {"x": 1}, "doc_count": 5}));
}

#[test]
fn composite_paging() {
    let f = sales();
    let req = |after: Option<Value>| {
        let mut c = json!({"size": 2, "sources": [{"t": {"terms": {"field": "tag"}}}, {"q": {"histogram": {"field": "qty", "interval": 2}}}]});
        if let Some(a) = after {
            c["after"] = a;
        }
        json!({"c": {"composite": c}})
    };
    let r = run(&f, req(None)).unwrap();
    assert_eq!(r["c"]["buckets"], json!([{"key": {"t": "a", "q": 0.0}, "doc_count": 1}, {"key": {"t": "a", "q": 2.0}, "doc_count": 1}]));
    assert_eq!(r["c"]["after_key"], json!({"t": "a", "q": 2.0}));
    let r = run(&f, req(Some(r["c"]["after_key"].clone()))).unwrap();
    assert_eq!(r["c"]["buckets"][0], json!({"key": {"t": "a", "q": 4.0}, "doc_count": 1}));
    let r = run(&f, json!({"c": {"composite": {"sources": [{"p": {"terms": {"field": "price", "missing_bucket": true, "order": "desc"}}}, {"d": {"date_histogram": {"field": "ts", "calendar_interval": "1d", "format": "yyyy-MM-dd", "missing_bucket": true}}}]}}})).unwrap();
    // ES: missing values sort last under `desc`, first under `asc`.
    let bs = r["c"]["buckets"].as_array().unwrap();
    assert!(!bs[0]["key"]["p"].is_null(), "{r}");
    assert!(bs.iter().any(|b| b["key"]["p"].is_null()), "{r}");
    assert!(bs.iter().skip_while(|b| !b["key"]["p"].is_null()).all(|b| b["key"]["p"].is_null()), "{r}");
}

#[test]
fn top_hits_and_pipelines() {
    let f = sales();
    let r = run(&f, json!({"t": {"terms": {"field": "tag", "size": 1}, "aggs": {"top": {"top_hits": {"size": 1, "sort": [{"price": {"order": "desc"}}]}}}}})).unwrap();
    let top = &r["t"]["buckets"][0]["top"]["hits"];
    assert_eq!(top["total"], json!({"value": 3, "relation": "eq"}));
    assert_eq!(top["max_score"], Value::Null);
    assert_eq!(top["hits"][0]["sort"], json!([20.0]));
    assert_eq!(top["hits"][0]["_score"], Value::Null);
    let r = run(&f, json!({"top": {"top_hits": {}}})).unwrap();
    assert_eq!(r["top"]["hits"]["hits"].as_array().unwrap().len(), 3);
    assert!(r["top"]["hits"]["max_score"].is_number());

    let r = run(
        &f,
        json!({
            "per_day": {
                "date_histogram": {"field": "ts", "calendar_interval": "day"},
                "aggs": {
                    "s": {"sum": {"field": "price"}},
                    "cum": {"cumulative_sum": {"buckets_path": "s"}},
                    "der": {"derivative": {"buckets_path": "s"}},
                    "ratio": {"bucket_script": {"buckets_path": {"x": "s", "n": "_count"}, "script": "params.x / params.n"}}
                }
            },
            "avg_day": {"avg_bucket": {"buckets_path": "per_day>s"}},
            "max_day": {"max_bucket": {"buckets_path": "per_day>s"}},
            "min_day": {"min_bucket": {"buckets_path": "per_day>_count"}},
            "sum_day": {"sum_bucket": {"buckets_path": "per_day>s"}},
            "st_day": {"stats_bucket": {"buckets_path": "per_day>s"}}
        }),
    )
    .unwrap();
    let b = &r["per_day"]["buckets"];
    assert_eq!(b[0]["cum"], json!({"value": 30.0}));
    assert_eq!(b[2]["cum"], json!({"value": 100.0}));
    assert!(b[0].get("der").is_none());
    assert_eq!(b[1]["der"], json!({"value": -30.0}));
    assert_eq!(b[0]["ratio"], json!({"value": 15.0}));
    assert_eq!(r["avg_day"], json!({"value": 100.0 / 3.0}));
    assert_eq!(r["max_day"], json!({"value": 70.0, "keys": ["2020-01-03T00:00:00.000Z"]}));
    assert_eq!(r["min_day"], json!({"value": 0.0, "keys": ["2020-01-02T00:00:00.000Z"]}));
    assert_eq!(r["sum_day"], json!({"value": 100.0}));
    assert_eq!(r["st_day"]["count"], json!(3));

    let r = run(
        &f,
        json!({"t": {"terms": {"field": "tag"}, "aggs": {
            "s": {"sum": {"field": "qty"}},
            "sel": {"bucket_selector": {"buckets_path": {"c": "_count"}, "script": {"source": "params.c >= 2"}}},
            "srt": {"bucket_sort": {"sort": [{"s": {"order": "desc"}}], "size": 1}}
        }}}),
    )
    .unwrap();
    assert_eq!(r["t"]["buckets"].as_array().unwrap().len(), 1);
    assert_eq!(r["t"]["buckets"][0]["key"], "a");
    assert_eq!(r["t"]["buckets"][0]["s"], json!({"value": 8.0}));
}

#[test]
fn errors_and_limits() {
    let f = sales();
    let e = run(&f, json!({"x": {"bogus": {}}})).unwrap_err();
    assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "parsing_exception", "Unknown aggregation type [bogus]"));
    assert_eq!(run(&f, json!({"x": {"moving_avg": {"buckets_path": "a"}}})).unwrap_err().status, 400);
    assert!(run(&f, json!({"x": {"terms": {"field": "tag"}, "avg": {"field": "qty"}}})).unwrap_err().reason.contains("Found two aggregation type definitions"));
    let leaves = f.index.leaves();
    let segs: Vec<AggSeg> = leaves.iter().map(|l| AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: (0..l.num_docs()).collect(), scores: vec![] }).collect();
    let fetch = |_: usize, _: DocId, _: f32, _: &Value| -> Result<Value> { Ok(Value::Null) };
    let all = |_: usize| -> Vec<DocId> { vec![] };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 2, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    let e = run_aggregations(&json!({"t": {"terms": {"field": "tag"}}}), &segs, &env).unwrap_err();
    assert_eq!(e.kind, "too_many_buckets_exception");
}

#[test]
fn java_double_format() {
    assert_eq!(jfmt(20.0), "20.0");
    assert_eq!(jfmt(0.5), "0.5");
    assert_eq!(jfmt(1e7), "1.0E7");
    assert_eq!(jfmt(1.25e-4), "1.25E-4");
}

#[test]
fn sampler_and_diversified() {
    let f = sales();
    // ES 8.19.22: sampler{shard_size:3} on 5 docs -> doc_count 3; the sample is
    // per shard (index), not per segment. Scores in the fixture are the doc id,
    // so the best-scoring docs are the highest ids of each leaf.
    let sub = json!({"c": {"value_count": {"field": "qty"}}});
    let r = run(&f, json!({"s": {"sampler": {"shard_size": 3}, "aggs": sub.clone()}})).unwrap();
    assert_eq!(r["s"]["doc_count"], 3);
    let r = run(&f, json!({"s": {"sampler": {}, "aggs": sub.clone()}})).unwrap();
    assert_eq!(r["s"]["doc_count"], 5);
    // diversified: max_docs_per_value defaults to 1 -> one doc per tag value
    // (tags a, b, c; doc 3 has [b, c] and takes the first value).
    let r = run(&f, json!({"s": {"diversified_sampler": {"field": "tag", "shard_size": 3}, "aggs": sub.clone()}})).unwrap();
    assert_eq!(r["s"]["doc_count"], 2);
    let r = run(&f, json!({"s": {"diversified_sampler": {"field": "tag", "max_docs_per_value": 2}, "aggs": sub.clone()}})).unwrap();
    assert_eq!(r["s"]["doc_count"], 4);
    // ES: "Sampler aggregation must be used with child aggregations."
    let e = run(&f, json!({"s": {"sampler": {}}})).unwrap_err();
    assert_eq!(e.reason, "Sampler aggregation must be used with child aggregations.");
}

/// ES skips docs with no value (4 of the 5 fixture docs have a `price`);
/// `missing` substitutes a value instead.
#[test]
fn metrics_skip_docs_without_a_value() {
    let f = sales();
    let r = run(
        &f,
        json!({
            "a": {"avg": {"field": "price"}}, "s": {"sum": {"field": "price"}}, "mn": {"min": {"field": "price"}},
            "vc": {"value_count": {"field": "price"}}, "st": {"stats": {"field": "price"}},
            "p": {"percentiles": {"field": "price", "percents": [50]}}, "c": {"cardinality": {"field": "price"}},
            "am": {"avg": {"field": "price", "missing": 0}}
        }),
    )
    .unwrap();
    assert_eq!(r["a"]["value"], 25.0);
    assert_eq!(r["s"]["value"], 100.0);
    assert_eq!(r["mn"]["value"], 10.0);
    assert_eq!(r["vc"]["value"], 4);
    assert_eq!(r["st"]["count"], 4);
    assert_eq!(r["p"]["values"]["50.0"], 25.0);
    assert_eq!(r["c"]["value"], 4);
    assert_eq!(r["am"]["value"], 20.0);
}

#[test]
fn geo_value_count_counts_values_in_segment_and_memtable() {
    let f = fixture(json!({"properties": {"point": {"type": "geo_point"}}}), &[
        json!({"point": [10, 20]}),
        json!({"point": [[10, 20], [10, 20], [30, 40]]}),
        json!({}), json!({"point": null}), json!({"point": []}),
        json!({"point": [0, 0]}),
    ]);
    let result = run(&f, json!({
        "plain": {"value_count": {"field": "point"}},
        "missing": {"value_count": {"field": "point", "missing": "0,0"}},
        "absent": {"value_count": {"field": "absent"}}
    })).unwrap();
    assert_eq!(result["plain"]["value"], 5);
    assert_eq!(result["missing"]["value"], 8);
    assert_eq!(result["absent"]["value"], 0);
    assert!(run(&f, json!({"bad": {"avg": {"field": "point"}}})).is_err());
    assert!(run(&f, json!({"bad": {"value_count": {"field": "point", "missing": "invalid"}}})).is_err());
}

#[test]
fn bounded_sampler_matches_full_sort_with_ties_and_edge_scores() {
    let docs: Docs = (0..4).map(|s| (0..257).map(|d| (d, match d % 31 {
        0 => f32::NAN, 1 => -0.0, 2 => 0.0, 3 => f32::INFINITY,
        _ => ((d * 17 + s * 7) % 19) as f32,
    })).collect()).collect();
    for limit in [0, 1, 100, 1028, 2000] {
        let mut expected: Vec<_> = docs.iter().enumerate().flat_map(|(s, ds)| ds.iter().map(move |&(d, sc)| (s, d, sc))).collect();
        let order = |a: &(usize, DocId, f32), b: &(usize, DocId, f32)| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1));
        expected.sort_by(order); expected.truncate(limit);
        let mut actual = sample_top(&docs, &[0, 1, 2, 3], limit); actual.sort_by(order);
        assert_eq!(actual.iter().map(|(s,d,sc)| (*s,*d,sc.to_bits())).collect::<Vec<_>>(), expected.iter().map(|(s,d,sc)| (*s,*d,sc.to_bits())).collect::<Vec<_>>());
    }
}

/// Review r1 B1: a date_histogram of 1000 buckets over a terms aggregation
/// on a keyword with 50k terms per segment. Each terms run builds (and
/// frees) one per-term table per segment (~1.2mb each): charged and given
/// back as it goes, the request breaker holds a few MB, where charging
/// every table for the whole request came to ~1.2 GB per segment (24 GB
/// over the review's 20 segments). Two segments (100k docs) keep the test
/// cheap and still overrun the 256mb limit below by ~10x the old way.
#[test]
fn nested_terms_charges_the_live_peak_not_the_sum() {
    let reg = AnalysisRegistry::default();
    let mapping = json!({"properties": {"k": {"type": "keyword"}, "ts": {"type": "date"}}});
    let mut m = Mappings::parse(&mapping, &reg).unwrap();
    m.assign_core_ids();
    let lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    const SEGMENTS: usize = 2;
    const TERMS: usize = 50_000;
    for s in 0..SEGMENTS {
        for i in 0..TERMS {
            // Every segment holds all 50k terms; hour `i % 1000` spreads
            // them over the 1000 date buckets.
            let d = json!({"k": format!("t{i:05}"), "ts": (i % 1000) as i64 * 3_600_000});
            let (pd, _) = parse_with_dynamic(&d, &format!("{s}-{i}"), &m, &lookup, 1000).unwrap();
            index.add_document(&pd.doc).unwrap();
        }
        index.refresh().unwrap();
    }
    let f = Fixture { index, lookup };
    assert_eq!(f.index.leaves().iter().filter(|l| l.num_docs() > 0).count(), SEGMENTS, "the per-segment tables are what the old charging summed");
    // A 256mb request breaker (parent off): charging every per-term table
    // for the whole request (~2.4mb a date bucket) tripped it after ~100
    // of the 1000 buckets.
    let b = crate::memory::Breakers::leak_for_tests(1 << 31);
    b.apply_settings(&|k| match k {
        "indices.breaker.request.limit" => Some("256mb".into()),
        "indices.breaker.total.limit" => Some("-1".into()),
        _ => None,
    });
    let scope = crate::memory::RequestScope::new("nested terms", b);
    let aggs = json!({"h": {"date_histogram": {"field": "ts", "fixed_interval": "1h"}, "aggs": {"t": {"terms": {"field": "k", "size": 3}}}}});
    let out = crate::memory::scoped(Some(scope.clone()), || run(&f, aggs)).unwrap();
    let buckets = out["h"]["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), 1000);
    assert_eq!(buckets[7]["doc_count"], (SEGMENTS * TERMS / 1000) as u64);
    assert_eq!(b.request.reserved(), 0, "everything released");
    assert_eq!(b.request.tripped(), 0);
}

/// `sampler`'s bounded heap and `diversified_sampler`'s sorted copy (and
/// per-value counts) are charged to the request breaker and given back.
/// Over 20k docs (160k of doc lists charged up front): a 20k-doc sample
/// trips (sampler: 320k heap + 160k sampled over 600kb; diversified:
/// 320k copy + ~1.4mb of per-value counts over 1mb), a 100-doc one fits,
/// and everything is released after each run.
#[test]
fn sampler_working_state_is_charged_and_released() {
    let docs: Vec<Value> = (0..20_000).map(|i| json!({"k": format!("v{i:05}")})).collect();
    let f = fixture(json!({"properties": {"k": {"type": "keyword"}}}), &docs);
    let b = crate::memory::Breakers::leak_for_tests(1 << 31);
    let sub = json!({"n": {"value_count": {"field": "k"}}});
    for (t, limit) in [("sampler", "600kb"), ("diversified_sampler", "1mb")] {
        b.apply_settings(&|k| match k {
            "indices.breaker.request.limit" => Some(limit.into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        let attempt = |aggs: Value| {
            let scope = crate::memory::RequestScope::new("sampler", b);
            let out = crate::memory::scoped(Some(scope.clone()), || run(&f, aggs));
            assert_eq!(b.request.reserved(), 0, "everything released");
            out
        };
        let mut spec = json!({"shard_size": 20_000});
        if t == "diversified_sampler" {
            spec["field"] = json!("k");
        }
        let agg = |spec: &Value| {
            let mut s = Map::new();
            s.insert(t.to_string(), spec.clone());
            s.insert("aggs".into(), sub.clone());
            json!({"s": Value::Object(s)})
        };
        let e = attempt(agg(&spec)).expect_err(t);
        assert!(crate::memory::is_breaking(&e), "{t}: {e:?}");
        spec["shard_size"] = json!(100);
        let out = attempt(agg(&spec)).unwrap();
        assert_eq!(out["s"]["doc_count"], 100, "{t}: {out}");
    }
}

#[test]
fn numeric_sketches_and_filter_unions_break_before_growth_and_release() {
    let docs: Vec<Value> = (0..20_000).map(|i| json!({"n":i as f64 + 0.125})).collect();
    let f = fixture(json!({"properties":{"n":{"type":"double"}}}), &docs);
    let b = crate::memory::Breakers::leak_for_tests(1 << 31);
    let attempt = |spec: Value, limit: &str| {
        b.apply_settings(&|key| match key {
            "indices.breaker.request.limit" => Some(limit.into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        let scope = crate::memory::RequestScope::new("agg-memory", b);
        let result = crate::memory::scoped(Some(scope), || run(&f, spec));
        assert_eq!(b.request.reserved(), 0, "success and error release reservations");
        result
    };
    for spec in [
        json!({"p":{"percentiles":{"field":"n","tdigest":{"compression":1000}}}}),
        json!({"p":{"percentiles":{"field":"n","hdr":{"number_of_significant_value_digits":5}}}}),
        json!({"p":{"median_absolute_deviation":{"field":"n"}}}),
    ] {
        let e = attempt(spec.clone(), "600kb").expect_err("sketch must be accounted");
        assert!(crate::memory::is_breaking(&e), "{e:?}");
        attempt(spec, "32mb").unwrap();
    }
    let filters = |other| json!({"f":{"filters":{"filters":[{"match_all":{}}],"other_bucket":other}}});
    let result = attempt(filters(false), "600kb").unwrap();
    assert_eq!(result["f"]["buckets"][0]["doc_count"], 20_000);
    let error = attempt(filters(true), "600kb").expect_err("union table must be accounted");
    assert!(crate::memory::is_breaking(&error), "{error:?}");
    attempt(filters(true), "32mb").unwrap();
}

#[test]
fn filter_aggregations_use_each_index_regex_limit() {
    let f = fixture(json!({"properties": {"tag": {"type": "keyword"}}}), &[json!({"tag": "abc"})]);
    let leaves = f.index.leaves();
    let lookup = &f.lookup;
    let execute = |indices: &[(&str, usize)], pattern: &str| {
        let segs: Vec<AggSeg> = indices.iter().flat_map(|(index, limit)| leaves.iter().map(move |leaf| AggSeg {
            leaf: *leaf, lookup, index: *index, max_regex_length: *limit,
            docs: (0..leaf.num_docs()).collect(), scores: vec![],
        })).collect();
        let fetch = |_: usize, _: DocId, _: f32, _: &Value| -> Result<Value> { Ok(Value::Null) };
        let all = |i: usize| -> Vec<DocId> { (0..segs[i].leaf.num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all,
            background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        run_aggregations(&json!({"outer": {"filter": {"match_all": {}}, "aggs": {
            "inner": {"filters": {"filters": {"regex": {"regexp": {"tag": pattern}}}}}
        }}}), &segs, &env)
    };
    assert_eq!(execute(&[("permissive", 3)], "abc").unwrap()["outer"]["inner"]["buckets"]["regex"]["doc_count"], 1);
    let error = execute(&[("permissive", 3), ("strict", 2)], "abc").unwrap_err();
    assert!(error.reason.contains("index.max_regex_length"), "{error}");
    assert!(execute(&[("utf16", 2)], "😀a").is_err());
    assert!(execute(&[("utf16", 3)], "😀a").is_ok());
    let long = "a".repeat(1001);
    assert!(execute(&[("raised", 1001)], &long).is_ok());
    assert!(execute(&[("default", 1000)], &long).is_err());
}

/// Rally nyc_taxis `distance_amount_agg`: `histogram` (interval 1) over a
/// `scaled_float` with a `stats` sub-aggregation. The single-valued column
/// path (sealed segment) and the generic path (memtable, multi-valued
/// field) give the brute-force buckets, including `missing`,
/// `hard_bounds`, `extended_bounds`, `min_doc_count` and `keyed`.
#[test]
fn numeric_histogram_with_stats_matches_brute_force() {
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    // Distances in cents never on a whole unit (no rounding ambiguity at a
    // bucket edge); amounts in eighths (exact sums in any order).
    let mut docs = Vec::new();
    let mut rows: Vec<(Option<i64>, Option<f64>)> = Vec::new();
    for _ in 0..600 {
        let cents = (next() % 1200) as i64 * 100 + 1 + (next() % 98) as i64;
        let dist = (next() % 10 != 0).then_some(cents);
        let amt = (next() % 7 != 0).then(|| (next() % 800) as f64 / 8.0);
        let mut d = Map::new();
        if let Some(c) = dist {
            d.insert("dist".into(), json!(c as f64 / 100.0));
            d.insert("mdist".into(), json!([c as f64 / 100.0, c as f64 / 100.0 + 0.5]));
        }
        if let Some(a) = amt {
            d.insert("amt".into(), json!(a));
        }
        docs.push(Value::Object(d));
        rows.push((dist, amt));
    }
    let f = fixture(json!({"properties": {"dist": {"type": "scaled_float", "scaling_factor": 100}, "mdist": {"type": "double"}, "amt": {"type": "double"}}}), &docs);
    let brute = |lo: Option<f64>, hi: Option<f64>, missing: Option<f64>| -> std::collections::BTreeMap<i64, (u64, Vec<f64>)> {
        let mut m: std::collections::BTreeMap<i64, (u64, Vec<f64>)> = Default::default();
        for (dist, amt) in &rows {
            let Some(x) = dist.map(|c| c as f64 / 100.0).or(missing) else { continue };
            if lo.is_some_and(|l| x < l) || hi.is_some_and(|h| x > h) {
                continue;
            }
            let e = m.entry(x.floor() as i64).or_default();
            e.0 += 1;
            e.1.extend(amt.iter());
        }
        m
    };
    let check = |agg: Value, want: std::collections::BTreeMap<i64, (u64, Vec<f64>)>, fill: bool| {
        let r = run(&f, json!({"h": agg.clone()})).unwrap();
        let buckets = r["h"]["buckets"].as_array().unwrap_or_else(|| panic!("{agg}: {r}"));
        let (first, last) = (*want.keys().next().unwrap(), *want.keys().last().unwrap());
        let expect: Vec<(i64, u64, Vec<f64>)> = if fill {
            (first..=last).map(|k| want.get(&k).map(|(c, v)| (k, *c, v.clone())).unwrap_or((k, 0, Vec::new()))).collect()
        } else {
            want.into_iter().map(|(k, (c, v))| (k, c, v)).collect()
        };
        assert_eq!(buckets.len(), expect.len(), "{agg}: {r}");
        for (b, (k, c, v)) in buckets.iter().zip(expect) {
            assert_eq!(b["key"], json!(k as f64), "{agg}");
            assert_eq!(b["doc_count"], json!(c), "{agg} key {k}");
            let s = &b["s"];
            assert_eq!(s["count"], json!(v.len()), "{agg} key {k}");
            if v.is_empty() {
                assert!(s["min"].is_null() && s["max"].is_null() && s["avg"].is_null(), "{agg} key {k}: {s}");
            } else {
                let sum: f64 = v.iter().sum();
                assert_eq!(s["min"].as_f64(), v.iter().copied().reduce(f64::min), "{agg} key {k}");
                assert_eq!(s["max"].as_f64(), v.iter().copied().reduce(f64::max), "{agg} key {k}");
                assert_eq!(s["sum"].as_f64(), Some(sum), "{agg} key {k}");
                assert_eq!(s["avg"].as_f64(), Some(sum / v.len() as f64), "{agg} key {k}");
            }
        }
    };
    let stats = json!({"s": {"stats": {"field": "amt"}}});
    check(json!({"histogram": {"field": "dist", "interval": 1}, "aggs": stats}), brute(None, None, None), true);
    check(json!({"histogram": {"field": "dist", "interval": 1, "min_doc_count": 1}, "aggs": stats}), brute(None, None, None), false);
    check(json!({"histogram": {"field": "dist", "interval": 1, "missing": 3.5}, "aggs": stats}), brute(None, None, Some(3.5)), true);
    check(json!({"histogram": {"field": "dist", "interval": 1, "hard_bounds": {"min": 100.0, "max": 400.0}}, "aggs": stats}), brute(Some(100.0), Some(400.0), None), true);
    // The multi-valued field takes the generic path; its first values are `dist`.
    let r = run(&f, json!({"m": {"histogram": {"field": "mdist", "interval": 1, "min_doc_count": 1}}})).unwrap();
    let total: u64 = r["m"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
    assert!(total >= rows.iter().filter(|r| r.0.is_some()).count() as u64);
    let keyed = run(&f, json!({"h": {"histogram": {"field": "dist", "interval": 1, "keyed": true, "min_doc_count": 1}}})).unwrap();
    let k0 = *brute(None, None, None).keys().next().unwrap();
    assert_eq!(keyed["h"]["buckets"][format!("{}.0", k0)]["key"], json!(k0 as f64), "{keyed}");
}

/// Timing for `distance_amount_agg`'s shape: `NYC_HISTO_DOCS` docs (default
/// 2,000,000), one sealed segment. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn numeric_histogram_with_stats_timing() {
    let n: usize = std::env::var("NYC_HISTO_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
    let reg = AnalysisRegistry::default();
    let mapping = json!({"properties": {"trip_distance": {"type": "scaled_float", "scaling_factor": 100}, "total_amount": {"type": "double"}}});
    let mut m = Mappings::parse(&mapping, &reg).unwrap();
    m.assign_core_ids();
    let lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    let mut seed = 3u64;
    for i in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let d = json!({"trip_distance": ((seed >> 33) % 5000) as f64 / 100.0, "total_amount": ((seed >> 20) % 10000) as f64 / 100.0});
        let (pd, _) = parse_with_dynamic(&d, &i.to_string(), &m, &lookup, 1000).unwrap();
        index.add_document(&pd.doc).unwrap();
    }
    index.refresh().unwrap();
    let f = Fixture { index, lookup };
    let agg = json!({"distance_histo": {"histogram": {"field": "trip_distance", "interval": 1}, "aggs": {"total_amount_stats": {"stats": {"field": "total_amount"}}}}});
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let r = run(&f, agg.clone()).unwrap();
        times.push(t.elapsed());
        assert_eq!(r["distance_histo"]["buckets"].as_array().unwrap().len(), 50);
    }
    times.sort();
    eprintln!("histogram+stats over {n} docs: median {:?} (min {:?})", times[2], times[0]);
}

/// A fixture whose docs are all in one sealed segment (no memtable leaf).
fn sealed_fixture(mapping: Value, docs: &[Value]) -> Fixture {
    let reg = AnalysisRegistry::default();
    let mut m = Mappings::parse(&mapping, &reg).unwrap();
    m.assign_core_ids();
    let lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    for (i, d) in docs.iter().enumerate() {
        let (pd, _) = parse_with_dynamic(d, &i.to_string(), &m, &lookup, 1000).unwrap();
        index.add_document(&pd.doc).unwrap();
    }
    index.refresh().unwrap();
    Fixture { index, lookup }
}

/// [`sealed_fixture`] as two segments (the first half, then the rest).
fn sealed_two_segments(mapping: Value, docs: &[Value]) -> Fixture {
    let reg = AnalysisRegistry::default();
    let mut m = Mappings::parse(&mapping, &reg).unwrap();
    m.assign_core_ids();
    let lookup = FieldLookup::build(&m, &reg);
    let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
    for (i, d) in docs.iter().enumerate() {
        let (pd, _) = parse_with_dynamic(d, &i.to_string(), &m, &lookup, 1000).unwrap();
        index.add_document(&pd.doc).unwrap();
        if i == docs.len() / 2 { index.refresh().unwrap(); }
    }
    index.refresh().unwrap();
    Fixture { index, lookup }
}

/// `pmc`-shaped timestamps: `yyyy-MM-dd HH:mm:ss` over 2008-2016 (some
/// months left empty), every `gap`-th doc without one (0: all have one).
fn pmc_dates(n: usize, gap: usize) -> Vec<Value> {
    let mut seed = 11u64;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if gap > 0 && i % gap == 0 {
                return json!({"n": i});
            }
            let mut ms = 1_199_145_600_000i64 + ((seed >> 20) % (9 * 365 * 86_400)) as i64 * 1000;
            // Leave March 2011 empty, so filled empty buckets are checked too.
            if (1_298_937_600_000..1_301_616_000_000).contains(&ms) {
                ms += 40 * 86_400_000;
            }
            json!({"ts": crate::date::DateFormatter::new("yyyy-MM-dd HH:mm:ss").unwrap().format(ms, crate::date::Tz::UTC), "n": i})
        })
        .collect()
}

/// The count-only `date_histogram` of a whole sealed segment counts each
/// bucket from the points index; its buckets (keys, `key_as_string`,
/// counts, filled empty buckets) equal the per-doc collector's (forced with
/// a sub-aggregation) and a brute force over the values.
#[test]
fn date_histogram_whole_segment_points_counts_match_per_doc() {
    let mapping = json!({"properties": {"ts": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, "n": {"type": "long"}}});
    for gap in [0usize, 13] {
        let docs = pmc_dates(9000, gap);
        let f = sealed_fixture(mapping.clone(), &docs);
        let values: Vec<i64> = docs.iter().filter_map(|d| d["ts"].as_str()).map(|s| crate::date::parse_date_math(s, &DateFormatter::new("yyyy-MM-dd HH:mm:ss").unwrap(), Tz::UTC, 0, false).unwrap()).collect();
        for conf in [
            json!({"field": "ts", "calendar_interval": "month"}),
            json!({"field": "ts", "calendar_interval": "quarter", "time_zone": "America/New_York"}),
            json!({"field": "ts", "calendar_interval": "day", "offset": "+6h"}),
            json!({"field": "ts", "calendar_interval": "year", "min_doc_count": 1}),
            json!({"field": "ts", "fixed_interval": "30d"}),
            json!({"field": "ts", "calendar_interval": "month", "keyed": true, "format": "yyyy-MM"}),
        ] {
            let before = POINTS_COUNTED_SEGMENTS.with(|c| c.get());
            let fast = run(&f, json!({"h": {"date_histogram": conf.clone()}})).unwrap();
            assert_eq!(POINTS_COUNTED_SEGMENTS.with(|c| c.get()), before + 1, "{conf}: points path not taken");
            // A sub-aggregation forces the per-doc collector.
            let slow = run(&f, json!({"h": {"date_histogram": conf.clone(), "aggs": {"c": {"value_count": {"field": "ts"}}}}})).unwrap();
            let strip = |v: &Value| -> Value {
                let mut v = v.clone();
                let strip_bucket = |b: &mut Value| { b.as_object_mut().unwrap().shift_remove("c"); };
                match &mut v["h"]["buckets"] {
                    Value::Array(a) => a.iter_mut().for_each(strip_bucket),
                    Value::Object(m) => m.values_mut().for_each(strip_bucket),
                    _ => panic!("{v}"),
                }
                v
            };
            assert_eq!(fast, strip(&slow), "{conf}");
            // Brute force: every value rounded on its own.
            let rounding = Rounding::parse(conf.as_object().unwrap(), "h").unwrap();
            let mut want: std::collections::BTreeMap<i64, u64> = Default::default();
            for &x in &values {
                *want.entry(rounding.round(x)).or_default() += 1;
            }
            let got: std::collections::BTreeMap<i64, u64> = match &fast["h"]["buckets"] {
                Value::Array(a) => a.iter().filter(|b| b["doc_count"] != 0).map(|b| (b["key"].as_i64().unwrap(), b["doc_count"].as_u64().unwrap())).collect(),
                Value::Object(m) => m.values().filter(|b| b["doc_count"] != 0).map(|b| (b["key"].as_i64().unwrap(), b["doc_count"].as_u64().unwrap())).collect(),
                _ => unreachable!(),
            };
            assert_eq!(got, want, "{conf}");
        }
        // Not the whole segment (a filtered doc set): the per-doc paths, same answer.
        let leaves = f.index.leaves();
        let segs: Vec<AggSeg> = leaves.iter().map(|l| AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: (1..l.num_docs()).collect(), scores: vec![] }).collect();
        let fetch = |_: usize, _: DocId, _: f32, _: &Value| -> Result<Value> { Ok(json!({})) };
        let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        let before = POINTS_COUNTED_SEGMENTS.with(|c| c.get());
        let partial = run_aggregations(&json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}}), &segs, &env).unwrap();
        // Only a doc set holding every doc with a value (here: the skipped
        // doc 0 has none) may count from the points index.
        assert_eq!(POINTS_COUNTED_SEGMENTS.with(|c| c.get()), before + usize::from(docs[0]["ts"].as_str().is_none()), "a partial doc set uses the points counts only when it holds every doc with a value");
        let rounding = Rounding::parse(json!({"calendar_interval": "month"}).as_object().unwrap(), "h").unwrap();
        let skipped = docs[0]["ts"].as_str().map(|s| rounding.round(crate::date::parse_date_math(s, &DateFormatter::new("yyyy-MM-dd HH:mm:ss").unwrap(), Tz::UTC, 0, false).unwrap()));
        let total: u64 = partial["h"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
        assert_eq!(total, values.len() as u64 - u64::from(skipped.is_some()));
    }
}

/// The two-leaf fixture (segment + memtable): the memtable has no column, so
/// only the segment can count from points; the merged buckets still equal
/// the per-doc collector's.
#[test]
fn date_histogram_points_counts_with_memtable_leaf() {
    let docs = pmc_dates(12_000, 17);
    let f = fixture(json!({"properties": {"ts": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, "n": {"type": "long"}}}), &docs);
    let fast = run(&f, json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}})).unwrap();
    let mut slow = run(&f, json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"c": {"value_count": {"field": "ts"}}}}})).unwrap();
    for b in slow["h"]["buckets"].as_array_mut().unwrap() {
        b.as_object_mut().unwrap().shift_remove("c");
    }
    assert_eq!(fast, slow);
}

/// Timing for `articles_monthly_agg_*`'s shape: one sealed segment of
/// `PMC_AGG_DOCS` docs (default 115,000, a pmc shard), monthly buckets.
/// Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn date_histogram_monthly_whole_segment_timing() {
    let n: usize = std::env::var("PMC_AGG_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(115_000);
    let f = sealed_fixture(json!({"properties": {"ts": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, "n": {"type": "long"}}}), &pmc_dates(n, 0));
    let agg = json!({"articles_over_time": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}});
    let forced = json!({"articles_over_time": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"c": {"value_count": {"field": "ts"}}}}});
    for (label, spec) in [("points counts", &agg), ("per-doc collector (sub-agg forced)", &forced)] {
        let mut times = Vec::new();
        for _ in 0..21 {
            let t = std::time::Instant::now();
            let r = run(&f, spec.clone()).unwrap();
            times.push(t.elapsed());
            assert!(r["articles_over_time"]["buckets"].as_array().unwrap().len() > 100);
        }
        times.sort();
        eprintln!("{label} over {n} docs: median {:?} (min {:?})", times[10], times[0]);
    }
}

/// The count-only `auto_date_histogram` of whole sealed segments reads the
/// range from the points index and counts each bucket as a range count;
/// its buckets and interval equal the per-doc collector's (forced with a
/// sub-aggregation), with and without a time zone, also with docs missing.
#[test]
fn auto_date_histogram_whole_segment_points_counts_match_per_doc() {
    let mapping = json!({"properties": {"ts": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, "n": {"type": "long"}}});
    for gap in [0usize, 13] {
        let docs = pmc_dates(9000, gap);
        let f = sealed_fixture(mapping.clone(), &docs);
        for conf in [
            json!({"field": "ts", "buckets": 10}),
            json!({"field": "ts", "buckets": 20, "time_zone": "America/New_York"}),
            json!({"field": "ts", "buckets": 7, "time_zone": "Australia/Lord_Howe"}),
            json!({"field": "ts", "buckets": 50, "minimum_interval": "day", "time_zone": "+05:30"}),
        ] {
            let fast = run(&f, json!({"h": {"auto_date_histogram": conf.clone()}})).unwrap();
            let slow = run(&f, json!({"h": {"auto_date_histogram": conf.clone(), "aggs": {"c": {"value_count": {"field": "ts"}}}}})).unwrap();
            let mut slow = slow;
            for b in slow["h"]["buckets"].as_array_mut().unwrap() { b.as_object_mut().unwrap().shift_remove("c"); }
            assert_eq!(fast, slow, "{conf}");
        }
    }
}

/// `cargo test -p elasticsearch-engine --release --lib auto_date_histogram_microbench -- --ignored --nocapture`
/// (`AUTO_DATE_BENCH_DOCS`, default 1M): metricbeat-shaped
/// `auto_date_histogram` (with and without a time zone) and the matching
/// `date_histogram` on one sealed segment, best of 20.
#[test]
#[ignore]
fn auto_date_histogram_microbench() {
    let n: usize = std::env::var("AUTO_DATE_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let mapping = json!({"properties": {"ts": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, "n": {"type": "long"}}});
    let f = sealed_fixture(mapping, &pmc_dates(n, 0));
    for (label, q) in [
        ("auto_date_histogram", json!({"h": {"auto_date_histogram": {"field": "ts", "buckets": 20}}})),
        ("auto_date_histogram tz", json!({"h": {"auto_date_histogram": {"field": "ts", "buckets": 20, "time_zone": "America/New_York"}}})),
        ("date_histogram", json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "week"}}})),
        ("date_histogram tz", json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "week", "time_zone": "America/New_York"}}})),
    ] {
        let best = (0..20).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(run(&f, q.clone()).unwrap()); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min);
        eprintln!("{label:<24} {:>8.3} ms ({n} docs)", best * 1e3);
    }
}

/// The count-only `auto_date_histogram` of a query's matches under a
/// required range on the date field (Rally metricbeat's one-day filter)
/// counts from the points index within the range; buckets and interval
/// equal the per-doc collector's, with and without a time zone, also with
/// docs missing the field, a range covering nothing and a sub-aggregation's
/// doc subset (which must not take the range shortcut).
#[test]
fn auto_date_histogram_in_query_range_counts_match_per_doc() {
    let mapping = json!({"properties": {"ts": {"type": "date"}, "n": {"type": "long"}}});
    let ms_of = |i: usize| (i % 13 != 0).then(|| 1_550_000_000_000i64 + ((i as i64).wrapping_mul(7_919) % 864_000) * 1000);
    let docs: Vec<Value> = (0..12_000).map(|i| match ms_of(i) { Some(ms) => json!({"ts": ms, "n": i}), None => json!({"n": i}) }).collect();
    let f = sealed_fixture(mapping, &docs);
    let field = f.lookup.get("ts").and_then(|r| r.core_id).unwrap();
    let leaves = f.index.leaves();
    assert_eq!(leaves.len(), 1);
    let fetch = |s: usize, d: DocId, sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({"_index": "idx", "_id": format!("{s}-{d}"), "_score": sc})) };
    let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    for (lo, hi) in [(1_550_100_000_000i64, 1_550_400_000_000i64), (1_550_000_000_000, 1_550_086_399_999), (1_400_000_000_000, 1_400_000_000_001)] {
        let matching: Vec<DocId> = (0..docs.len()).filter(|&i| ms_of(i).is_some_and(|ms| lo <= ms && ms <= hi)).map(|i| i as DocId).collect();
        let segs = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: matching.clone(), scores: vec![1.0; matching.len()] }];
        for conf in [
            json!({"field": "ts", "buckets": 24}),
            json!({"field": "ts", "buckets": 24, "time_zone": "America/New_York"}),
            json!({"field": "ts", "buckets": 5, "minimum_interval": "hour"}),
        ] {
            let spec = json!({"h": {"auto_date_histogram": conf.clone()}});
            let ranged = with_query_ranges(vec![Some((field, lo, hi))], || run_aggregations(&spec, &segs, &env)).unwrap();
            let plain = run_aggregations(&spec, &segs, &env).unwrap();
            let mut slow = run_aggregations(&json!({"h": {"auto_date_histogram": conf.clone(), "aggs": {"c": {"value_count": {"field": "ts"}}}}}), &segs, &env).unwrap();
            if let Some(b) = slow["h"]["buckets"].as_array_mut() { for b in b { b.as_object_mut().unwrap().shift_remove("c"); } }
            assert_eq!(ranged, plain, "{conf} [{lo}, {hi}]");
            assert_eq!(ranged, slow, "{conf} [{lo}, {hi}]");
            // Under a parent bucket the doc set is a subset: same as per doc.
            let nested = json!({"t": {"terms": {"field": "n", "size": 3}, "aggs": {"h": {"auto_date_histogram": conf.clone()}}}});
            let a = with_query_ranges(vec![Some((field, lo, hi))], || run_aggregations(&nested, &segs, &env)).unwrap();
            assert_eq!(a, run_aggregations(&nested, &segs, &env).unwrap(), "{conf} nested");
        }
    }
}

/// With the query's matches left uncollected (`with_elided_root`: a bare
/// range query on the histogram's field), the count-only
/// `auto_date_histogram` answers as from the collected matches: from the
/// points index for a single-valued column (dense, or sparse with docs
/// missing the field), and by reading the matches back from the column for
/// a multi-valued one; with and without a time zone, also for a range
/// covering nothing.
#[test]
fn auto_date_histogram_of_uncollected_range_matches_collected() {
    let ms_of = |i: usize| 1_550_000_000_000i64 + ((i as i64).wrapping_mul(7_919) % 864_000) * 1000;
    let dense: Vec<Value> = (0..12_000).map(|i| json!({"ts": ms_of(i), "n": i})).collect();
    let sparse: Vec<Value> = (0..12_000).map(|i| if i % 13 == 0 { json!({"n": i}) } else { json!({"ts": ms_of(i), "n": i}) }).collect();
    let multi: Vec<Value> = (0..6_000).map(|i| json!({"ts": [ms_of(i), ms_of(i + 50_000)], "n": i})).collect();
    let mapping = json!({"properties": {"ts": {"type": "date"}, "n": {"type": "long"}}});
    for (label, docs) in [("dense", &dense), ("sparse", &sparse), ("multi", &multi)] {
        let f = sealed_fixture(mapping.clone(), docs);
        let field = f.lookup.get("ts").and_then(|r| r.core_id).unwrap();
        let leaves = f.index.leaves();
        assert_eq!(leaves.len(), 1);
        let fetch = |s: usize, d: DocId, sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({"_index": "idx", "_id": format!("{s}-{d}"), "_score": sc})) };
        let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        let values = |i: usize| -> Vec<i64> { match &docs[i]["ts"] { Value::Array(a) => a.iter().filter_map(Value::as_i64).collect(), v => v.as_i64().into_iter().collect() } };
        for (lo, hi) in [(1_550_100_000_000i64, 1_550_400_000_000i64), (1_550_000_000_000, 1_550_086_399_999), (1_400_000_000_000, 1_400_000_000_001)] {
            let matching: Vec<DocId> = (0..docs.len()).filter(|&i| values(i).iter().any(|&ms| lo <= ms && ms <= hi)).map(|i| i as DocId).collect();
            let collected = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: matching.clone(), scores: vec![1.0; matching.len()] }];
            let uncollected = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: Vec::new(), scores: Vec::new() }];
            for conf in [
                json!({"field": "ts", "buckets": 24}),
                json!({"field": "ts", "buckets": 24, "time_zone": "America/New_York"}),
                json!({"field": "ts", "buckets": 5, "minimum_interval": "hour"}),
            ] {
                let spec = json!({"h": {"auto_date_histogram": conf.clone()}, "h2": {"auto_date_histogram": conf.clone()}});
                assert_eq!(points_counted_field(&spec), Some("ts"));
                let want = with_query_ranges(vec![Some((field, lo, hi))], || run_aggregations(&spec, &collected, &env)).unwrap();
                let got = with_query_ranges(vec![Some((field, lo, hi))], || with_elided_root(vec![true], || run_aggregations(&spec, &uncollected, &env))).unwrap();
                assert_eq!(got, want, "{label} {conf} [{lo}, {hi}]");
            }
            // Count-only `date_histogram` (Rally nyc_taxis `date_histogram_agg`):
            // calendar and fixed intervals, a time zone, an offset, a
            // format, min_doc_count; buckets clipped to the query's range.
            for conf in [
                json!({"field": "ts", "calendar_interval": "day"}),
                json!({"field": "ts", "calendar_interval": "hour", "time_zone": "America/New_York"}),
                json!({"field": "ts", "fixed_interval": "90m", "offset": "+7m", "min_doc_count": 1}),
                json!({"field": "ts", "calendar_interval": "day", "format": "dd/MM/yyyy", "keyed": true}),
            ] {
                let spec = json!({"h": {"date_histogram": conf.clone()}});
                assert_eq!(points_counted_field(&spec), Some("ts"));
                let want = with_query_ranges(vec![Some((field, lo, hi))], || run_aggregations(&spec, &collected, &env)).unwrap();
                let got = with_query_ranges(vec![Some((field, lo, hi))], || with_elided_root(vec![true], || run_aggregations(&spec, &uncollected, &env))).unwrap();
                assert_eq!(got, want, "{label} {conf} [{lo}, {hi}]");
            }
        }
    }
    // A date_histogram and an auto_date_histogram of one field both count.
    assert_eq!(points_counted_field(&json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "day"}}, "a": {"auto_date_histogram": {"field": "ts"}}})), Some("ts"));
    // Anything but count-only auto_date_histograms of one field collects.
    for spec in [
        json!({"h": {"auto_date_histogram": {"field": "ts"}, "aggs": {"c": {"value_count": {"field": "ts"}}}}}),
        json!({"h": {"auto_date_histogram": {"field": "ts", "missing": "2019-01-01"}}}),
        json!({"h": {"auto_date_histogram": {"field": "ts"}, "meta": {"a": 1}}}),
        json!({"h": {"auto_date_histogram": {"field": "ts"}}, "t": {"auto_date_histogram": {"field": "other"}}}),
        json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "day", "extended_bounds": {"min": 0, "max": 1}}}}),
        json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "day"}, "aggs": {"c": {"value_count": {"field": "ts"}}}}}),
    ] {
        assert_eq!(points_counted_field(&spec), None, "{spec}");
    }
}

/// Like [`run`] over the root docs only (as a search collects them for a
/// `match_all` over an index with nested docs).
fn run_roots(f: &Fixture, aggs: Value) -> Result<Value> {
    run_roots_as(f, aggs, false)
}

/// [`run_roots`] with the roots left uncollected (`with_elided_root`) when
/// `elided`, as a `match_all` search hands a whole-path nested histogram.
fn run_roots_as(f: &Fixture, aggs: Value, elided: bool) -> Result<Value> {
    let leaves = f.index.leaves();
    let segs: Vec<AggSeg> = leaves
        .iter()
        .map(|l| {
            let docs: Vec<DocId> = if elided { Vec::new() } else { (0..l.num_docs()).filter(|&d| l.is_root(d)).collect() };
            let scores = vec![1.0; docs.len()];
            AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs, scores }
        })
        .collect();
    let fetch = |_: usize, _: DocId, _: f32, _: &Value| -> Result<Value> { Ok(json!({})) };
    let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    if elided {
        return with_elided_root(vec![true; segs.len()], || run_aggregations(&aggs, &segs, &env));
    }
    run_aggregations(&aggs, &segs, &env)
}

/// `nested` + count-only `date_histogram` under every root (Rally
/// `nested-date-histo`): segments whose children are the whole path skip
/// the child list and count from the points index (or list the path's docs
/// when the field is also copied to the root), matching the per-doc
/// collector (forced by a sub-aggregation) and a brute force.
#[test]
fn nested_date_histogram_whole_path_matches_per_doc() {
    for include_in_root in [false, true] {
        let mapping = json!({"properties": {
            "tag": {"type": "keyword"},
            "answers": {"type": "nested", "include_in_root": include_in_root, "properties": {"date": {"type": "date"}, "user": {"type": "keyword"}}},
            "comments": {"type": "nested", "properties": {"n": {"type": "long"}}}
        }});
        let mut seed = 5u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed >> 33
        };
        let rounding = Rounding::parse(json!({"calendar_interval": "month"}).as_object().unwrap(), "h").unwrap();
        let mut want: std::collections::BTreeMap<i64, u64> = Default::default();
        let mut answers = 0u64;
        let mut docs = Vec::new();
        for i in 0..3000u64 {
            let k = next() % 6;
            let list: Vec<Value> = (0..k)
                .map(|j| {
                    answers += 1;
                    if (i + j) % 7 == 0 {
                        return json!({"user": "u"});
                    }
                    let ms = 1_199_145_600_000i64 + (next() % (5 * 365 * 86_400)) as i64 * 1000;
                    *want.entry(rounding.round(ms)).or_default() += 1;
                    json!({"user": "u", "date": ms})
                })
                .collect();
            let comments: Vec<Value> = (0..next() % 3).map(|n| json!({"n": n})).collect();
            docs.push(json!({"tag": "t", "answers": list, "comments": comments}));
        }
        let f = sealed_fixture(mapping, &docs);
        let h = json!({"field": "answers.date", "calendar_interval": "month"});
        let before = POINTS_COUNTED_SEGMENTS.with(|c| c.get());
        let fast = run_roots(&f, json!({"a": {"nested": {"path": "answers"}, "aggs": {"h": {"date_histogram": h.clone()}}}})).unwrap();
        let points = POINTS_COUNTED_SEGMENTS.with(|c| c.get()) - before;
        let slow = run_roots(&f, json!({"a": {"nested": {"path": "answers"}, "aggs": {"h": {"date_histogram": h.clone(), "aggs": {"c": {"value_count": {"field": "answers.date"}}}}}}})).unwrap();
        assert_eq!(fast["a"]["doc_count"].as_u64(), Some(answers), "include_in_root {include_in_root}");
        assert_eq!(slow["a"]["doc_count"].as_u64(), Some(answers));
        let buckets = |v: &Value| -> Vec<(i64, u64)> { v["a"]["h"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"].as_i64().unwrap(), b["doc_count"].as_u64().unwrap())).collect() };
        assert_eq!(buckets(&fast), buckets(&slow), "include_in_root {include_in_root}");
        // The roots left uncollected (a match_all search): the same answer.
        let spec = json!({"a": {"nested": {"path": "answers"}, "aggs": {"h": {"date_histogram": h.clone()}}}});
        assert_eq!(nested_whole_path_spec(&spec), Some("answers"));
        assert_eq!(run_roots_as(&f, spec, true).unwrap(), fast, "include_in_root {include_in_root}: uncollected roots");
        let got: std::collections::BTreeMap<i64, u64> = buckets(&fast).into_iter().filter(|b| b.1 > 0).collect();
        assert_eq!(got, want, "include_in_root {include_in_root}");
        // Copied to the root, the field's points also hold root docs.
        assert_eq!(points > 0, !include_in_root, "points path");
    }
}

/// `histogram` + `stats` / `avg` / `sum` / `min` / `max` sub-aggregations of
/// single-valued numeric columns (Rally nyc_taxis `distance_amount_agg`)
/// fold in one pass; results equal the general collector's (forced by an
/// extra `value_count`, then removed) and a brute force: offset, hard and
/// extended bounds, min_doc_count, `_count` order, docs missing either
/// field, a doubles and a scaled_float column, several segments.
#[test]
fn histogram_of_metrics_matches_general_collector() {
    let mapping = json!({"properties": {"d": {"type": "scaled_float", "scaling_factor": 100}, "amt": {"type": "double"}, "n": {"type": "long"}}});
    let mut docs = Vec::new();
    for i in 0..9000u64 {
        let d = ((i * 7919) % 6000) as f64 / 100.0;
        let amt = ((i * 104729) % 100_000) as f64 / 37.0 - 300.0;
        docs.push(match i % 11 { 0 => json!({"amt": amt, "n": i}), 1 => json!({"d": d, "n": i}), _ => json!({"d": d, "amt": amt, "n": i}) });
    }
    let f = sealed_fixture(mapping, &docs);
    for conf in [
        json!({"field": "d", "interval": 1}),
        json!({"field": "d", "interval": 2.5, "offset": 0.3}),
        json!({"field": "d", "interval": 1, "min_doc_count": 1, "order": {"_count": "desc"}}),
        json!({"field": "d", "interval": 5, "hard_bounds": {"min": 3, "max": 41}, "extended_bounds": {"min": 0, "max": 70}}),
        json!({"field": "amt", "interval": 100}),
    ] {
        let metric_field = if conf["field"] == "d" { "amt" } else { "d" };
        let subs = json!({"st": {"stats": {"field": metric_field}}, "a": {"avg": {"field": "n"}}, "s": {"sum": {"field": metric_field}}, "lo": {"min": {"field": metric_field}}, "hi": {"max": {"field": "n"}}});
        let fast = run(&f, json!({"h": {"histogram": conf.clone(), "aggs": subs.clone()}})).unwrap();
        let mut general_subs = subs.clone();
        general_subs["vc"] = json!({"value_count": {"field": metric_field}});
        let mut general = run(&f, json!({"h": {"histogram": conf.clone(), "aggs": general_subs}})).unwrap();
        for b in general["h"]["buckets"].as_array_mut().unwrap() {
            b.as_object_mut().unwrap().shift_remove("vc");
        }
        assert_eq!(fast, general, "{conf}");
        // No sub-aggregations (counts only): the same buckets as the
        // general collector's (forced by a `value_count`, then removed).
        let bare = run(&f, json!({"h": {"histogram": conf.clone()}})).unwrap();
        let mut general = run(&f, json!({"h": {"histogram": conf.clone(), "aggs": {"vc": {"value_count": {"field": metric_field}}}}})).unwrap();
        for b in general["h"]["buckets"].as_array_mut().unwrap() {
            b.as_object_mut().unwrap().shift_remove("vc");
        }
        assert_eq!(bare, general, "{conf} without sub-aggregations");
    }
    // Brute force for the Rally shape.
    let out = run(&f, json!({"h": {"histogram": {"field": "d", "interval": 1}, "aggs": {"st": {"stats": {"field": "amt"}}}}})).unwrap();
    let mut want: std::collections::BTreeMap<i64, (u64, u64, f64)> = Default::default();
    for doc in &docs {
        let Some(d) = doc["d"].as_f64() else { continue };
        let e = want.entry(d.floor() as i64).or_default();
        e.0 += 1;
        if let Some(a) = doc["amt"].as_f64() { e.1 += 1; e.2 += a; }
    }
    let buckets = out["h"]["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), want.len());
    for (b, (k, (n, c, sum))) in buckets.iter().zip(&want) {
        assert_eq!(b["key"].as_f64().unwrap() as i64, *k);
        assert_eq!(b["doc_count"], json!(n));
        assert_eq!(b["st"]["count"], json!(c));
        assert!((b["st"]["sum"].as_f64().unwrap() - sum).abs() < 1e-6 * sum.abs().max(1.0), "{b}");
    }
}

/// The histogram (+ simple metrics) of the field a bare double range
/// filters on, with the range's matches left uncollected
/// (`with_query_f64_ranges` + `with_elided_root`), answers as from the
/// collected matches: the fused pass folds them from the column; a request
/// it does not cover (a `missing`) reads them back. Ranges with excluded
/// and infinite bounds, one covering nothing, NaN-free double and
/// scaled_float columns, docs without the field.
#[test]
fn histogram_of_uncollected_double_range_matches_collected() {
    for ty in [json!({"type": "double"}), json!({"type": "scaled_float", "scaling_factor": 100})] {
        let mapping = json!({"properties": {"d": ty, "amt": {"type": "double"}}});
        let docs: Vec<Value> = (0..8000u64).map(|i| {
            let d = ((i * 7919) % 7000) as f64 / 100.0 - 5.0;
            if i % 17 == 0 { json!({"amt": i as f64}) } else { json!({"d": d, "amt": (i % 977) as f64 / 3.0}) }
        }).collect();
        let f = sealed_fixture(mapping, &docs);
        let field = f.lookup.get("d").and_then(|r| r.core_id).unwrap();
        let leaves = f.index.leaves();
        assert_eq!(leaves.len(), 1);
        let fetch = |s: usize, d: DocId, sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({"_index": "idx", "_id": format!("{s}-{d}"), "_score": sc})) };
        let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        let col_value = |d: DocId| leaves[0].f64_values(field, d).first().copied();
        for (lo, hi) in [(0.0f64, 50.0f64.next_down()), (f64::NEG_INFINITY, f64::INFINITY), (10.5, 10.5), (1e9, 2e9)] {
            let matching: Vec<DocId> = (0..leaves[0].num_docs()).filter(|&d| col_value(d).is_some_and(|x| lo <= x && x <= hi)).collect();
            let collected = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: matching.clone(), scores: vec![1.0; matching.len()] }];
            let uncollected = vec![AggSeg { leaf: leaves[0], lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: Vec::new(), scores: Vec::new() }];
            for spec in [
                json!({"h": {"histogram": {"field": "d", "interval": 1}}}),
                json!({"h": {"histogram": {"field": "d", "interval": 1}, "aggs": {"st": {"stats": {"field": "amt"}}}}}),
                json!({"h": {"histogram": {"field": "d", "interval": 2.5, "offset": 0.5, "min_doc_count": 1}, "aggs": {"a": {"avg": {"field": "amt"}}, "m": {"max": {"field": "d"}}}}}),
            ] {
                assert_eq!(value_range_histogram_field(&spec), Some("d"));
                let want = run_aggregations(&spec, &collected, &env).unwrap();
                let got = with_query_f64_ranges(vec![Some((field, lo, hi))], || with_elided_root(vec![true], || run_aggregations(&spec, &uncollected, &env))).unwrap();
                assert_eq!(got, want, "{ty} {spec} [{lo}, {hi}]");
            }
            // Not the fused shape: read back, then the general collector.
            let spec = json!({"h": {"histogram": {"field": "d", "interval": 1, "missing": 3}, "aggs": {"st": {"stats": {"field": "amt"}}}}});
            assert_eq!(value_range_histogram_field(&spec), None);
            let want = run_aggregations(&spec, &collected, &env).unwrap();
            let got = with_query_f64_ranges(vec![Some((field, lo, hi))], || with_elided_root(vec![true], || run_aggregations(&spec, &uncollected, &env))).unwrap();
            assert_eq!(got, want, "{ty} missing [{lo}, {hi}]");
        }
    }
    for spec in [
        json!({"h": {"histogram": {"field": "d", "interval": 1}, "aggs": {"t": {"terms": {"field": "x"}}}}}),
        json!({"h": {"histogram": {"field": "d", "interval": 1}, "aggs": {"s": {"stats": {"field": "x"}}}}, "o": {"max": {"field": "d"}}}),
    ] {
        assert_eq!(value_range_histogram_field(&spec), None, "{spec}");
    }
}

/// k8s-shaped docs: `pods` pods sampled at irregular seconds over two
/// hours, with a double gauge (sometimes missing), a long and an
/// unsigned_long.
fn pod_docs(n: usize, pods: usize) -> Vec<Value> {
    let mut seed = 7u64;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let ms = 1_684_263_600_000i64 + ((seed >> 24) % 7_200_000) as i64;
            let ts = crate::date::DateFormatter::new("strict_date_optional_time").unwrap().format(ms, crate::date::Tz::UTC);
            let mut d = json!({"pod": format!("p{}", (seed >> 40) as usize % pods), "ts": ts, "mem": (seed >> 33) % 1000, "u": (seed >> 30) % 77});
            if i % 7 != 3 {
                d["cpu"] = json!(((seed >> 13) % 10_000) as f64 / 997.0);
            }
            d
        })
        .collect()
}

fn strip_meta(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.remove("meta");
            m.values_mut().for_each(strip_meta);
        }
        Value::Array(a) => a.iter_mut().for_each(strip_meta),
        _ => {}
    }
}

#[test]
fn fused_date_histogram_metrics_match_per_bucket_metrics() {
    let f = sealed_fixture(
        json!({"properties": {"pod": {"type": "keyword"}, "ts": {"type": "date"}, "cpu": {"type": "double"}, "mem": {"type": "long"}, "u": {"type": "unsigned_long"}}}),
        &pod_docs(3000, 23),
    );
    let metrics = |meta: bool| {
        let m = |t: &str, field: &str| if meta { json!({t: {"field": field}, "meta": {"m": 1}}) } else { json!({t: {"field": field}}) };
        json!({"a": m("avg", "cpu"), "s": m("sum", "mem"), "lo": m("min", "cpu"), "hi": m("max", "u")})
    };
    for hist in [
        json!({"field": "ts", "fixed_interval": "1m", "time_zone": "Europe/Athens", "min_doc_count": 1}),
        json!({"field": "ts", "fixed_interval": "30s", "min_doc_count": 0, "extended_bounds": {"min": 1_684_263_000_000i64, "max": 1_684_271_400_000i64}}),
        json!({"field": "ts", "calendar_interval": "hour", "time_zone": "Europe/Athens", "order": {"a": "desc"}}),
    ] {
        let before = FUSED_HISTOGRAMS.with(|n| n.get());
        let fused = run(&f, json!({"h": {"date_histogram": hist, "aggs": metrics(false)}})).unwrap();
        assert_eq!(FUSED_HISTOGRAMS.with(|n| n.get()), before + 1, "the fused collector ran: {hist}");
        let mut reference = run(&f, json!({"h": {"date_histogram": hist, "aggs": metrics(true)}})).unwrap();
        assert_eq!(FUSED_HISTOGRAMS.with(|n| n.get()), before + 1, "meta keeps the per-bucket metrics");
        strip_meta(&mut reference);
        let buckets = fused["h"]["buckets"].as_array().unwrap();
        assert!(buckets.len() >= 2 && buckets.iter().any(|b| b["a"]["value"].is_f64() && b["hi"]["value"].is_f64()), "{fused}");
        assert_eq!(fused, reference, "{hist}");
    }
}

/// Above 4,096 docs a histogram groups its buckets' docs through the
/// dense bucket table: the same buckets and per-bucket metrics as the
/// fused collector (and the same percentiles from one segment or two).
#[test]
fn dense_histogram_grouping_matches_the_fused_metrics() {
    let mapping = json!({"properties": {"pod": {"type": "keyword"}, "ts": {"type": "date"}, "cpu": {"type": "double"}, "mem": {"type": "long"}, "u": {"type": "unsigned_long"}}});
    let docs = pod_docs(9000, 23);
    let f = sealed_fixture(mapping.clone(), &docs);
    for hist in [
        json!({"field": "ts", "fixed_interval": "5m", "time_zone": "Europe/Athens", "min_doc_count": 1}),
        json!({"field": "ts", "calendar_interval": "hour"}),
        json!({"field": "ts", "fixed_interval": "30s", "min_doc_count": 0}),
    ] {
        let fused = run(&f, json!({"h": {"date_histogram": hist, "aggs": {"a": {"avg": {"field": "cpu"}}, "s": {"sum": {"field": "mem"}}}}})).unwrap();
        let mut generic = run(&f, json!({"h": {"date_histogram": hist, "aggs": {"a": {"avg": {"field": "cpu"}, "meta": {"m": 1}}, "s": {"sum": {"field": "mem"}, "meta": {"m": 1}}}}})).unwrap();
        strip_meta(&mut generic);
        assert_eq!(fused, generic, "{hist}");
        let p = run(&f, json!({"h": {"date_histogram": hist, "aggs": {"p": {"percentiles": {"field": "cpu", "percents": [95]}}}}})).unwrap();
        let buckets = p["h"]["buckets"].as_array().unwrap();
        assert_eq!(buckets.len(), fused["h"]["buckets"].as_array().unwrap().len());
        assert!(buckets.iter().filter(|b| b["doc_count"].as_u64() > Some(0)).all(|b| b["p"]["values"]["95.0"].is_f64()), "{p}");
    }
}

#[test]
fn terms_ordered_by_a_metric_ranks_with_it_and_completes_the_kept_buckets() {
    let f = sealed_fixture(
        json!({"properties": {"pod": {"type": "keyword"}, "ts": {"type": "date"}, "cpu": {"type": "double"}, "mem": {"type": "long"}, "u": {"type": "unsigned_long"}}}),
        &pod_docs(3000, 23),
    );
    let aggs = json!({
        "1": {"date_histogram": {"field": "ts", "fixed_interval": "5m", "time_zone": "Europe/Athens", "min_doc_count": 1}, "aggs": {"2": {"avg": {"field": "cpu"}}}},
        "2": {"avg": {"field": "cpu"}},
    });
    for order in [json!({"2": "desc"}), json!({"2.value": "asc"}), json!([{"2": "desc"}, {"_key": "asc"}])] {
        let top = run(&f, json!({"0": {"terms": {"field": "pod", "size": 5, "order": order}, "aggs": aggs}})).unwrap();
        let all = run(&f, json!({"0": {"terms": {"field": "pod", "size": 1000, "order": order}, "aggs": aggs}})).unwrap();
        let top = top["0"]["buckets"].as_array().unwrap();
        assert_eq!(top.len(), 5);
        assert_eq!(top[..], all["0"]["buckets"].as_array().unwrap()[..5], "{order}");
        assert!(top.iter().all(|b| b["1"]["buckets"].as_array().is_some_and(|h| !h.is_empty())));
    }
    // An order through a bucket sub-aggregation still ranks correctly.
    let nested = json!({"f": {"filter": {"exists": {"field": "cpu"}}, "aggs": {"m": {"max": {"field": "mem"}}}}, "x": {"sum": {"field": "u"}}});
    let top = run(&f, json!({"0": {"terms": {"field": "pod", "size": 4, "order": {"f>m": "desc"}}, "aggs": nested}})).unwrap();
    let all = run(&f, json!({"0": {"terms": {"field": "pod", "size": 1000, "order": {"f>m": "desc"}}, "aggs": nested}})).unwrap();
    assert_eq!(top["0"]["buckets"].as_array().unwrap()[..], all["0"]["buckets"].as_array().unwrap()[..4]);
    assert!(top["0"]["buckets"][0].get("x").is_some());
}

/// The one-pass terms collector (a TSDB dashboard's terms over pods with
/// a date_histogram of metrics and ranking metrics) renders exactly what
/// the general collector does: same buckets, order, counts, sums.
#[test]
fn one_pass_terms_match_the_general_collector() {
    let mapping = json!({"properties": {"pod": {"type": "keyword"}, "ts": {"type": "date"}, "cpu": {"type": "double"}, "mem": {"type": "long"}, "u": {"type": "unsigned_long"}}});
    let docs = pod_docs(4000, 37);
    let sealed = sealed_fixture(mapping.clone(), &docs);
    let split = sealed_two_segments(mapping.clone(), &docs);
    assert_eq!(split.index.leaves().len(), 2);
    let open = fixture(mapping, &docs);
    let shapes = |meta: bool| {
        let m = |t: &str, field: &str| if meta { json!({t: {"field": field}, "meta": {"m": 1}}) } else { json!({t: {"field": field}}) };
        let filt = |meta: bool, q: Value, aggs: Option<Value>| {
            let mut f = json!({"filter": q});
            if let Some(a) = aggs { f["aggs"] = a; }
            if meta { f["meta"] = json!({"m": 1}); }
            f
        };
        vec![
            (json!({"field": "pod", "size": 10, "order": {"2": "desc"}}), json!({
                "1": {"date_histogram": {"field": "ts", "fixed_interval": "30m", "time_zone": "Europe/Athens", "min_doc_count": 1}, "aggs": {"2": m("avg", "cpu")}},
                "2": m("avg", "cpu")})),
            (json!({"field": "pod", "size": 100, "order": {"2": "desc"}}), json!({
                "1": {"date_histogram": {"field": "ts", "fixed_interval": "1m", "time_zone": "Europe/Athens", "min_doc_count": 1}, "aggs": {"a": m("avg", "cpu"), "s": m("sum", "mem")}},
                "2": m("max", "u")})),
            (json!({"field": "pod", "size": 5}), json!({
                "s": m("sum", "mem"),
                "h": {"date_histogram": {"field": "ts", "calendar_interval": "hour", "time_zone": "Europe/Athens"}, "aggs": {"lo": m("min", "cpu")}}})),
            (json!({"field": "pod", "size": 7, "order": [{"lo": "asc"}, {"_key": "desc"}]}), json!({"lo": m("min", "cpu"), "hi": m("max", "mem")})),
            (json!({"field": "pod", "size": 3, "order": {"_key": "asc"}, "min_doc_count": 120}), json!({
                "h": {"date_histogram": {"field": "ts", "fixed_interval": "10m", "min_doc_count": 0}, "aggs": {"a": m("avg", "cpu")}}})),
            // The tx_network dashboard's shape: ranked by a metric under an
            // exists filter.
            (json!({"field": "pod", "size": 6, "shard_size": 25, "order": {"f>m": "desc"}}), json!({
                "f": filt(meta, json!({"bool": {"must": [], "filter": [{"bool": {"should": [{"exists": {"field": "cpu"}}], "minimum_should_match": 1}}], "should": [], "must_not": []}}),
                    Some(json!({"m": m("avg", "u")})))})),
            (json!({"field": "pod", "size": 4, "order": [{"f": "asc"}, {"_key": "asc"}]}), json!({
                "f": filt(meta, json!({"exists": {"field": "cpu"}}), None), "s": m("sum", "mem"),
                "g": filt(meta, json!({"exists": {"field": "mem"}}), Some(json!({"x": m("max", "cpu")})))})),
        ]
    };
    for (i, ((terms, aggs), (_, reference_aggs))) in shapes(false).into_iter().zip(shapes(true)).enumerate() {
        for (f, one_pass) in [(&sealed, true), (&split, true), (&open, false)] {
            let before = ONE_PASS_TERMS.with(|n| n.get());
            let got = run(f, json!({"0": {"terms": terms, "aggs": aggs}})).unwrap();
            if one_pass {
                assert_eq!(ONE_PASS_TERMS.with(|n| n.get()), before + 1, "shape {i}: the one-pass collector ran");
            }
            let mut want = run(f, json!({"0": {"terms": terms, "aggs": reference_aggs}})).unwrap();
            strip_meta(&mut want);
            assert!(got["0"]["buckets"].as_array().is_some_and(|b| !b.is_empty()), "shape {i}: {got}");
            assert_eq!(got, want, "shape {i}");
        }
    }
}

#[test]
fn exists_filters_by_column_match_the_exists_query() {
    let f = sealed_fixture(
        json!({"properties": {"pod": {"type": "keyword"}, "ts": {"type": "date"}, "cpu": {"type": "double"}, "mem": {"type": "long"}, "u": {"type": "unsigned_long"}, "tags": {"type": "keyword"}, "nodv": {"type": "long", "doc_values": false}}}),
        &pod_docs(800, 9).into_iter().enumerate().map(|(i, mut d)| {
            if i % 3 == 0 { d["tags"] = json!(["x", "y"]); }
            if i % 4 == 1 { d["nodv"] = json!(i); }
            d
        }).collect::<Vec<_>>(),
    );
    let wrapped = |field: &str, fast: bool| {
        let exists = json!({"exists": {"field": field}});
        // `boost` is not recognized: the same filter through the query engine.
        let inner = if fast { json!({"bool": {"should": [exists], "minimum_should_match": 1}}) } else { json!({"bool": {"should": [exists], "minimum_should_match": 1, "boost": 1.0}}) };
        json!({"bool": {"must": [], "filter": [inner], "should": [], "must_not": []}})
    };
    for field in ["cpu", "mem", "u", "pod", "tags", "absent", "nodv"] {
        let spec = |fast: bool| json!({"h": {"date_histogram": {"field": "ts", "fixed_interval": "10m"}, "aggs": {"t": {"terms": {"field": "pod", "size": 3, "order": {"b>m": "desc"}},
            "aggs": {"b": {"filter": wrapped(field, fast), "aggs": {"m": {"avg": {"field": "mem"}}}}}}}}});
        assert_eq!(run(&f, spec(true)).unwrap(), run(&f, spec(false)).unwrap(), "{field}");
    }
    assert_eq!(exists_field(&wrapped("cpu", true)), Some("cpu"));
    assert_eq!(exists_field(&wrapped("cpu", false)), None);
    assert_eq!(exists_field(&json!({"bool": {"filter": [{"exists": {"field": "a"}}], "must_not": [{"exists": {"field": "b"}}]}})), None);
    assert_eq!(exists_field(&json!({"bool": {"should": [{"exists": {"field": "a"}}, {"exists": {"field": "b"}}]}})), None);
}

#[test]
fn keyword_cardinality_by_ordinal_set_counts_exactly_below_the_threshold() {
    let docs: Vec<Value> = (0..4000).map(|i| if i % 13 == 0 { json!({"n": i}) } else { json!({"dep": format!("deployment-{}", i % 97), "n": i}) }).collect();
    for f in [sealed_fixture(json!({"properties": {"dep": {"type": "keyword"}, "n": {"type": "long"}}}), &docs), fixture(json!({"properties": {"dep": {"type": "keyword"}, "n": {"type": "long"}}}), &docs)] {
        let r = run(&f, json!({"c": {"cardinality": {"field": "dep"}}, "m": {"cardinality": {"field": "dep", "missing": "none"}}, "few": {"filter": {"range": {"n": {"lt": 40}}}, "aggs": {"c": {"cardinality": {"field": "dep"}}}}})).unwrap();
        assert_eq!(r["c"]["value"], 97);
        assert_eq!(r["m"]["value"], 98);
        // 40 docs, 4 without a value (0, 13, 26, 39): 36 distinct.
        assert_eq!(r["few"]["c"]["value"], 36);
    }
}


/// A `scaled_float`'s aggregated values are ES's doc values: the scaled
/// long times the inverse factor (`ScaledFloatLeafFieldData`), e.g. 887.3
/// with factor 100 aggregates as 887.3000000000001, not the stored
/// quotient. Single values per bucket make sum / min / max / avg exact:
/// fused histogram metrics, general per-bucket metrics, terms and
/// histogram keys, top-level metrics, sealed and memtable leaves.
#[test]
fn scaled_float_aggregations_read_es_doc_values() {
    let es = |l: i64| l as f64 * (1.0 / 100.0);
    let longs: [i64; 6] = [88730, 700, 1001, 33, 123456789, 29];
    assert!(longs.iter().any(|&l| es(l) != l as f64 / 100.0), "the test needs values whose product and quotient differ");
    let docs: Vec<Value> = longs.iter().enumerate().map(|(i, &l)| json!({"k": format!("k{i}"), "sf": l as f64 / 100.0, "n": i})).collect();
    let mapping = json!({"properties": {"k": {"type": "keyword"}, "sf": {"type": "scaled_float", "scaling_factor": 100}, "n": {"type": "long"}}});
    for f in [sealed_fixture(mapping.clone(), &docs), fixture(mapping.clone(), &docs)] {
        // Per-bucket metrics: one doc per term, general path.
        let r = run(&f, json!({"t": {"terms": {"field": "k", "size": 10}, "aggs": {"s": {"stats": {"field": "sf"}}, "m": {"max": {"field": "sf"}}}}})).unwrap();
        for b in r["t"]["buckets"].as_array().unwrap() {
            let i: usize = b["key"].as_str().unwrap()[1..].parse().unwrap();
            let want = es(longs[i]);
            assert_eq!(b["s"]["sum"].as_f64(), Some(want), "{b}");
            assert_eq!(b["s"]["max"].as_f64(), Some(want), "{b}");
            assert_eq!(b["s"]["avg"].as_f64(), Some(want), "{b}");
            assert_eq!(b["m"]["value"].as_f64(), Some(want), "{b}");
        }
        // Fused histogram metrics (one doc per bucket of `n`).
        let r = run(&f, json!({"h": {"histogram": {"field": "n", "interval": 1}, "aggs": {"s": {"stats": {"field": "sf"}}}}})).unwrap();
        for b in r["h"]["buckets"].as_array().unwrap() {
            let i = b["key"].as_f64().unwrap() as usize;
            assert_eq!(b["s"]["sum"].as_f64(), Some(es(longs[i])), "{b}");
            assert_eq!(b["s"]["min"].as_f64(), Some(es(longs[i])), "{b}");
        }
        // Keys: terms and histogram on the scaled_float itself.
        let r = run(&f, json!({"t": {"terms": {"field": "sf", "size": 10}}, "x": {"max": {"field": "sf"}}, "y": {"min": {"field": "sf"}}})).unwrap();
        let mut keys: Vec<f64> = r["t"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_f64().unwrap()).collect();
        keys.sort_by(f64::total_cmp);
        let mut want: Vec<f64> = longs.iter().map(|&l| es(l)).collect();
        want.sort_by(f64::total_cmp);
        assert_eq!(keys, want);
        assert_eq!(r["x"]["value"].as_f64(), want.last().copied());
        assert_eq!(r["y"]["value"].as_f64(), want.first().copied());
    }
}

/// `stats` / `sum` / `avg` / `min` / `max` fold block-decoded columns
/// without listing the values; they equal `extended_stats` (which still
/// lists them, in the same order): long, double and scaled_float columns,
/// docs without the field, `missing`, a filtered subset, two segments.
#[test]
fn simple_metrics_fold_equals_listed_values() {
    let docs: Vec<Value> = (0..5000u64).map(|i| {
        let mut d = json!({"k": if i % 3 == 0 { "a" } else { "b" }});
        if i % 7 != 0 { d["l"] = json!((i * 7919) as i64 % 10007 - 5000); }
        if i % 5 != 0 { d["d"] = json!(((i * 104729) % 100_000) as f64 / 37.0); }
        if i % 11 != 0 { d["sf"] = json!(((i * 31) % 90_000) as f64 / 100.0); }
        d
    }).collect();
    let mapping = json!({"properties": {"k": {"type": "keyword"}, "l": {"type": "long"}, "d": {"type": "double"}, "sf": {"type": "scaled_float", "scaling_factor": 100}}});
    let f = sealed_two_segments(mapping, &docs);
    for field in ["l", "d", "sf"] {
        for missing in [None, Some(json!(1.25))] {
            let mut body = json!({"field": field});
            if let Some(m) = &missing { body["missing"] = m.clone(); }
            let r = run(&f, json!({"t": {"terms": {"field": "k"}, "aggs": {
                "s": {"stats": body.clone()}, "x": {"extended_stats": body.clone()},
                "sum": {"sum": body.clone()}, "avg": {"avg": body.clone()}, "min": {"min": body.clone()}, "max": {"max": body.clone()}}}})).unwrap();
            for b in r["t"]["buckets"].as_array().unwrap() {
                for k in ["count", "min", "max", "avg", "sum"] {
                    assert_eq!(b["s"][k], b["x"][k], "{field} {missing:?} {k}: {b}");
                }
                assert_eq!(b["sum"]["value"], b["x"]["sum"], "{field}");
                assert_eq!(b["avg"]["value"], b["x"]["avg"], "{field}");
                assert_eq!(b["min"]["value"], b["x"]["min"], "{field}");
                assert_eq!(b["max"]["value"], b["x"]["max"], "{field}");
            }
        }
    }
}
