use super::super::{run_aggregations, AggEnv, AggSeg};
use super::t_lower_tail;
use crate::analysis::AnalysisRegistry;
use crate::document::parse_with_dynamic;
use crate::error::Result;
use crate::mapping::{FieldLookup, Mappings};
use rustic_core::types::{DocId, IndexSettings};
use rustic_core::Index;
use serde_json::{json, Value};

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
            index.refresh().unwrap();
        }
    }
    Fixture { index, lookup }
}

/// Runs over all docs, or only those whose global ordinal is in `only`.
fn run_on(f: &Fixture, aggs: Value, only: Option<&[usize]>) -> Result<Value> {
    let leaves = f.index.leaves();
    let mut base = 0usize;
    let mut segs: Vec<AggSeg> = Vec::new();
    for l in leaves.iter() {
        let docs: Vec<DocId> = (0..l.num_docs()).filter(|d| only.map(|o| o.contains(&(base + *d as usize))).unwrap_or(true)).collect();
        base += l.num_docs() as usize;
        let scores = docs.iter().map(|d| *d as f32).collect();
        segs.push(AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs, scores });
    }
    let fetch = |_s: usize, _d: DocId, _sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({})) };
    let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
    let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    run_aggregations(&aggs, &segs, &env)
}

fn run(f: &Fixture, aggs: Value) -> Result<Value> {
    run_on(f, aggs, None)
}

fn sales() -> Fixture {
    fixture(
        json!({"properties": {"tag": {"type": "keyword"}, "color": {"type": "keyword"}, "price": {"type": "double"}, "qty": {"type": "long"}, "ts": {"type": "date"}}}),
        &[
            json!({"tag": "a", "color": "red", "price": 10.0, "qty": 1, "ts": "2020-01-01T10:00:00Z"}),
            json!({"tag": "a", "color": "red", "price": 20.0, "qty": 2, "ts": "2020-01-01T12:00:00Z"}),
            json!({"tag": "b", "color": "blue", "price": 30.0, "qty": 3, "ts": "2020-01-03T00:00:00Z"}),
            json!({"tag": "b", "color": "red", "price": 40.0, "qty": 4, "ts": "2020-01-03T05:00:00Z"}),
            json!({"tag": "c", "qty": 5, "ts": "2020-02-10T00:00:00Z"}),
        ],
    )
}

#[test]
fn top_metrics_and_weighted_avg() {
    let f = sales();
    let r = run(&f, json!({"t": {"top_metrics": {"metrics": [{"field": "price"}, {"field": "tag"}], "sort": {"qty": "desc"}, "size": 2}}})).unwrap();
    assert_eq!(r["t"], json!({"top": [{"sort": [5], "metrics": {"price": null, "tag": "c"}}, {"sort": [4], "metrics": {"price": 40.0, "tag": "b"}}]}));
    let r = run(&f, json!({"t": {"top_metrics": {"metrics": {"field": "ts"}, "sort": [{"price": {"order": "asc"}}]}}})).unwrap();
    assert_eq!(r["t"], json!({"top": [{"sort": [10.0], "metrics": {"ts": "2020-01-01T10:00:00.000Z"}}]}));
    let r = run(&f, json!({"w": {"weighted_avg": {"value": {"field": "price"}, "weight": {"field": "qty"}}}})).unwrap();
    assert_eq!(r["w"], json!({"value": 30.0}));
    let r = run_on(&f, json!({"w": {"weighted_avg": {"value": {"field": "price"}, "weight": {"field": "qty"}}}}), Some(&[])).unwrap();
    assert_eq!(r["w"], json!({"value": null}));
}

#[test]
fn multi_and_rare_terms() {
    let f = sales();
    let r = run(&f, json!({"m": {"multi_terms": {"terms": [{"field": "tag"}, {"field": "color"}]}, "aggs": {"p": {"sum": {"field": "price"}}}}})).unwrap();
    assert_eq!(
        r["m"],
        json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": [
            {"key": ["a", "red"], "key_as_string": "a|red", "doc_count": 2, "p": {"value": 30.0}},
            {"key": ["b", "blue"], "key_as_string": "b|blue", "doc_count": 1, "p": {"value": 30.0}},
            {"key": ["b", "red"], "key_as_string": "b|red", "doc_count": 1, "p": {"value": 40.0}}
        ]})
    );
    let r = run(&f, json!({"m": {"multi_terms": {"terms": [{"field": "tag"}, {"field": "qty"}], "size": 1}}})).unwrap();
    assert_eq!(r["m"]["sum_other_doc_count"], 4);
    assert_eq!(r["m"]["buckets"][0]["key"], json!(["a", 1]));
    let r = run(&f, json!({"r": {"rare_terms": {"field": "tag"}}})).unwrap();
    assert_eq!(r["r"], json!({"buckets": [{"key": "c", "doc_count": 1}]}));
    let r = run(&f, json!({"r": {"rare_terms": {"field": "tag", "max_doc_count": 2, "exclude": ["c"]}}})).unwrap();
    assert_eq!(r["r"]["buckets"], json!([{"key": "a", "doc_count": 2}, {"key": "b", "doc_count": 2}]));
}

#[test]
fn significant_terms_jlh() {
    let mut docs = Vec::new();
    for i in 0..20 {
        docs.push(json!({"tag": if i < 5 { "rare" } else { "common" }, "grp": if i < 8 { "x" } else { "y" }}));
    }
    let f = fixture(json!({"properties": {"tag": {"type": "keyword"}, "grp": {"type": "keyword"}}}), &docs);
    // Foreground = docs 0..8 (grp x): 5 rare / 3 common.
    let fg: Vec<usize> = (0..8).collect();
    let r = run_on(&f, json!({"s": {"significant_terms": {"field": "tag"}}}), Some(&fg)).unwrap();
    let sp = 5.0 / 8.0;
    let bp = 5.0 / 20.0;
    let jlh = (sp - bp) * (sp / bp);
    assert_eq!(r["s"]["doc_count"], 8);
    assert_eq!(r["s"]["bg_count"], 20);
    assert_eq!(r["s"]["buckets"].as_array().unwrap().len(), 1);
    assert_eq!(r["s"]["buckets"][0]["key"], "rare");
    assert_eq!(r["s"]["buckets"][0]["doc_count"], 5);
    assert_eq!(r["s"]["buckets"][0]["bg_count"], 5);
    assert!((r["s"]["buckets"][0]["score"].as_f64().unwrap() - jlh).abs() < 1e-12);
    let r = run_on(&f, json!({"s": {"significant_terms": {"field": "tag", "percentage": {}}}}), Some(&fg)).unwrap();
    assert_eq!(r["s"]["buckets"][0]["score"], 1.0);
    let r = run_on(&f, json!({"s": {"significant_terms": {"field": "tag", "chi_square": {}}}}), Some(&fg)).unwrap();
    assert_eq!(r["s"]["buckets"][0]["key"], "rare");
}

#[test]
fn auto_and_variable_histograms() {
    let f = sales();
    let r = run(&f, json!({"a": {"auto_date_histogram": {"field": "ts", "buckets": 6}}})).unwrap();
    assert_eq!(r["a"]["interval"], "7d");
    assert_eq!(r["a"]["buckets"].as_array().unwrap().len(), 6, "{r}");
    assert_eq!(r["a"]["buckets"][0]["doc_count"], 4);
    let r = run(&f, json!({"a": {"auto_date_histogram": {"field": "ts", "buckets": 2}}})).unwrap();
    assert_eq!(
        r["a"],
        json!({"buckets": [
            {"key_as_string": "2020-01-01T00:00:00.000Z", "key": 1577836800000i64, "doc_count": 4},
            {"key_as_string": "2020-02-01T00:00:00.000Z", "key": 1580515200000i64, "doc_count": 1}
        ], "interval": "1M"})
    );
    let r = run(&f, json!({"v": {"variable_width_histogram": {"field": "qty", "buckets": 2}}})).unwrap();
    let b = r["v"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b[0]["min"], 1.0);
    assert_eq!(b[1]["max"], 5.0);
    assert_eq!(b[0]["doc_count"].as_u64().unwrap() + b[1]["doc_count"].as_u64().unwrap(), 5);
}

#[test]
fn boxplot_mad_string_stats() {
    let f = sales();
    let r = run(&f, json!({"b": {"boxplot": {"field": "qty"}}, "m": {"median_absolute_deviation": {"field": "qty"}}, "s": {"string_stats": {"field": "tag", "show_distribution": true}}})).unwrap();
    assert_eq!(r["b"]["min"], 1.0);
    assert_eq!(r["b"]["max"], 5.0);
    assert_eq!(r["b"]["q2"], 3.0);
    assert_eq!(r["b"]["lower"], 1.0);
    assert_eq!(r["b"]["upper"], 5.0);
    assert_eq!(r["b"].as_object().unwrap().keys().cloned().collect::<Vec<_>>(), vec!["min", "max", "q1", "q2", "q3", "lower", "upper"]);
    assert_eq!(r["m"], json!({"value": 1.0}));
    let ent = -(0.4f64 * 0.4f64.log2() + 0.4 * 0.4f64.log2() + 0.2 * 0.2f64.log2());
    assert_eq!(r["s"]["count"], 5);
    assert_eq!(r["s"]["min_length"], 1);
    assert_eq!(r["s"]["avg_length"], 1.0);
    assert!((r["s"]["entropy"].as_f64().unwrap() - ent).abs() < 1e-12);
    assert_eq!(r["s"]["distribution"], json!({"a": 0.4, "b": 0.4, "c": 0.2}));
    let r = run_on(&f, json!({"s": {"string_stats": {"field": "tag"}}}), Some(&[])).unwrap();
    assert_eq!(r["s"], json!({"count": 0, "min_length": null, "max_length": null, "avg_length": null, "entropy": 0.0}));
}

#[test]
fn rate_inside_date_histogram() {
    let f = sales();
    let r = run(&f, json!({"h": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"r": {"rate": {"field": "qty", "unit": "year"}}, "d": {"rate": {"unit": "day"}}}}})).unwrap();
    let b = &r["h"]["buckets"][0];
    assert_eq!(b["r"], json!({"value": 120.0})); // 10 per month * 12
    assert!((b["d"]["value"].as_f64().unwrap() - 4.0 / 31.0).abs() < 1e-12);
    let e = run(&f, json!({"r": {"rate": {"field": "qty"}}})).unwrap_err();
    assert!(e.reason.starts_with("The rate aggregation can only be used inside a date histogram"), "{e}");
    let r = run(&f, json!({"h": {"date_histogram": {"field": "ts", "fixed_interval": "1d"}, "aggs": {"r": {"rate": {"field": "qty", "unit": "hour"}}}}})).unwrap();
    assert_eq!(r["h"]["buckets"][0]["r"]["value"], 3.0 / 24.0);
}

#[test]
fn t_test_kinds() {
    // Known values: t ≈ -1.0 with df = 4 → two-tailed p ≈ 0.3739.
    assert!((2.0 * t_lower_tail(1.0, 4.0) - 0.373_900_966).abs() < 1e-6);
    assert!((2.0 * t_lower_tail(2.0, 10.0) - 0.073_388_034).abs() < 1e-6);
    let f = fixture(
        json!({"properties": {"a": {"type": "double"}, "b": {"type": "double"}, "g": {"type": "keyword"}}}),
        &[
            json!({"a": 1.0, "b": 2.0, "g": "x"}),
            json!({"a": 2.0, "b": 2.5, "g": "x"}),
            json!({"a": 3.0, "b": 4.5, "g": "y"}),
            json!({"a": 4.0, "b": 4.0, "g": "y"}),
            json!({"a": 5.0, "b": 7.0, "g": "y"}),
        ],
    );
    let r = run(&f, json!({"t": {"t_test": {"a": {"field": "a"}, "b": {"field": "b"}, "type": "paired"}}})).unwrap();
    let p = r["t"]["value"].as_f64().unwrap();
    assert!(p > 0.0 && p < 1.0);
    let r = run(&f, json!({"t": {"t_test": {"a": {"field": "a", "filter": {"term": {"g": "x"}}}, "b": {"field": "a", "filter": {"term": {"g": "y"}}}, "type": "homoscedastic"}}})).unwrap();
    assert!(r["t"]["value"].as_f64().unwrap() < 0.2);
    let e = run(&f, json!({"t": {"t_test": {"a": {"field": "a", "filter": {"match_all": {}}}, "b": {"field": "b"}, "type": "paired"}}})).unwrap_err();
    assert_eq!(e.reason, "Paired t-test doesn't support filters");
}

/// `range` buckets with an `auto_date_histogram` (and metrics) folded from
/// the columns equal the per-range collection: over doc lists, over
/// uncollected `match_all` roots, with missing values, time zones, keyed
/// output and empty ranges.
#[test]
fn fused_range_auto_date_histogram_matches_per_range_collection() {
    let mut docs = Vec::new();
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    for i in 0..40_000u64 {
        let mut d = serde_json::Map::new();
        if i % 17 != 0 { d.insert("size".into(), json!((rnd() % 5000) as i64 - 30)); }
        if i % 23 != 0 { d.insert("ts".into(), json!(1_683_500_000_000i64 + (i as i64) * 4_000 + (rnd() % 1000) as i64)); }
        if i % 5 != 0 { d.insert("tmin".into(), json!((rnd() % 900) as f64 / 7.0)); }
        d.insert("n".into(), json!((rnd() % 100) as i64));
        docs.push(Value::Object(d));
    }
    let mut f = fixture(json!({"properties": {"size": {"type": "long"}, "ts": {"type": "date"}, "tmin": {"type": "double"}, "n": {"type": "long"}}}), &docs);
    // Sealed segments: the fused path reads their columns.
    f.index.refresh().unwrap();
    let fused_before = super::super::FUSED_RANGES.with(std::cell::Cell::get);
    let ranges = json!([{"to": -10}, {"from": -10, "to": 10}, {"from": 10, "to": 100}, {"from": 100, "to": 1000}, {"from": 1000, "to": 2000}, {"from": 2000}, {"from": 9000}]);
    let specs = [
        json!({"r": {"range": {"field": "size", "ranges": ranges}, "aggs": {"d": {"auto_date_histogram": {"field": "ts", "buckets": 20}}}}}),
        json!({"r": {"range": {"field": "size", "ranges": ranges, "keyed": true}, "aggs": {"d": {"auto_date_histogram": {"field": "ts", "buckets": 7, "time_zone": "Europe/Athens"}}}}}),
        json!({"r": {"range": {"field": "size", "ranges": ranges}, "aggs": {"d": {"auto_date_histogram": {"field": "ts", "buckets": 10},
            "aggs": {"tmin": {"min": {"field": "tmin"}}, "tavg": {"avg": {"field": "size"}}, "tmax": {"max": {"field": "size"}}, "s": {"sum": {"field": "n"}}}}}}}),
        json!({"r": {"range": {"field": "size", "ranges": ranges}, "aggs": {"d": {"auto_date_histogram": {"field": "ts", "buckets": 3, "minimum_interval": "hour"}}}}}),
    ];
    let generic = |spec: &Value, only: Option<&[usize]>| {
        super::super::NO_FUSED_RANGE.with(|c| c.set(true));
        let r = run_on(&f, spec.clone(), only);
        super::super::NO_FUSED_RANGE.with(|c| c.set(false));
        r.unwrap()
    };
    let only: Vec<usize> = (0..40_000).filter(|i| i % 3 != 1).collect();
    for spec in &specs {
        let want = generic(spec, None);
        assert_eq!(run(&f, spec.clone()).unwrap(), want, "{spec}");
        assert_eq!(run_on(&f, spec.clone(), Some(&only)).unwrap(), generic(spec, Some(&only)), "subset {spec}");
        // Uncollected roots (a `match_all` over segments without deletions).
        let leaves = f.index.leaves();
        let segs: Vec<AggSeg> = leaves.iter().map(|l| AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: Vec::new(), scores: Vec::new() }).collect();
        let fetch = |_s: usize, _d: DocId, _sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({})) };
        let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        let elided = super::super::with_elided_root(vec![true; segs.len()], || run_aggregations(spec, &segs, &env)).unwrap();
        assert_eq!(elided, want, "elided {spec}");
        super::super::NO_FUSED_RANGE.with(|c| c.set(true));
        let elided_generic = super::super::with_elided_root(vec![true; segs.len()], || run_aggregations(spec, &segs, &env)).unwrap();
        super::super::NO_FUSED_RANGE.with(|c| c.set(false));
        assert_eq!(elided_generic, want, "elided, per range {spec}");
    }
    assert!(super::super::FUSED_RANGES.with(std::cell::Cell::get) >= fused_before + 3 * specs.len(), "the fused path ran");
    assert!(super::super::range_auto_date_fields(&specs[2]).is_some_and(|f| f == ["size", "ts", "tmin", "size", "size", "n"]));
    assert!(super::super::range_auto_date_fields(&json!({"r": {"range": {"field": "size", "ranges": ranges}, "aggs": {"d": {"date_histogram": {"field": "ts", "fixed_interval": "1h"}}}}})).is_none());
}
