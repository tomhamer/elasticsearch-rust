//! Review batch 6 (scoring and aggregations): each test is the closure of
//! one finding, against ES 9.5.5 behaviour.
use elasticsearch_engine::{index::WriteReq, node::Node, search::{self, ExecOpts}};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

fn index(node: &Node, name: &str, def: Value, docs: &[Value]) {
    let svc = node.create_index(name, &def).unwrap();
    let w = docs.iter().enumerate().map(|(i, d)| WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
    for r in svc.write(w, true) {
        r.unwrap();
    }
    svc.refresh().unwrap();
}

fn try_search(node: &Node, targets: &str, body: Value) -> Result<Value, elasticsearch_engine::error::EsError> {
    let request = search::parse_request(&body, &HashMap::new())?;
    let targets = node.resolve(targets, Default::default())?;
    let options = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &request, &options)?;
    Ok(serde_json::from_str(&out.json).unwrap())
}

fn search(node: &Node, targets: &str, body: Value) -> Value {
    try_search(node, targets, body.clone()).unwrap_or_else(|e| panic!("{body}: {e:?}"))
}

fn open() -> (tempfile::TempDir, Arc<Node>) {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    (dir, node)
}

fn one_shard(mappings: Value) -> Value {
    json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mappings})
}

fn ids(r: &Value) -> Vec<String> {
    let mut v: Vec<String> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    v.sort();
    v
}

/// #438: an unmapped field's `missing` takes the factor and modifier; a
/// mapped document without a value and no `missing` fails the search.
#[test]
fn field_value_factor_missing_contract() {
    let (_d, node) = open();
    index(&node, "fa", one_shard(json!({"properties": {"quality": {"type": "long"}}})), &[json!({"quality": 100}), json!({})]);
    index(&node, "fb", one_shard(json!({"properties": {"other": {"type": "long"}}})), &[json!({"other": 1})]);
    let q = |missing: Option<f64>| {
        let mut fvf = json!({"field": "quality", "modifier": "sqrt"});
        if let Some(m) = missing {
            fvf["missing"] = json!(m);
        }
        json!({"query": {"function_score": {"field_value_factor": fvf, "boost_mode": "replace", "min_score": 5}}})
    };
    // sqrt(9) = 3 < 5 for both the mapped-missing and the unmapped doc.
    let r = search(&node, "fa,fb", q(Some(9.0)));
    assert_eq!(r["hits"]["total"]["value"], 1, "{r}");
    assert_eq!(r["hits"]["hits"][0]["_index"], "fa");
    let e = try_search(&node, "fa", q(None)).unwrap_err();
    assert!(format!("{e:?}").contains("Missing value for field [quality]"), "{e:?}");
}

/// #470: an empty function list keeps the query's own score (min_score
/// still applies). #471: the function score's boost reaches the inner query
/// before `min_score` decides.
#[test]
fn function_score_empty_list_and_boost_order() {
    let (_d, node) = open();
    index(&node, "fs", one_shard(json!({"properties": {"t": {"type": "keyword"}}})), &[json!({"t": "a"}), json!({"t": "b"})]);
    let r = search(&node, "fs", json!({"query": {"function_score": {"query": {"match_all": {}}, "functions": [], "boost_mode": "sum", "min_score": 1.5}}}));
    assert_eq!(r["hits"]["total"]["value"], 0, "{r}");
    let r = search(&node, "fs", json!({"query": {"function_score": {"query": {"match_all": {}}, "functions": [], "boost_mode": "replace"}}}));
    assert_eq!(r["hits"]["hits"][0]["_score"], 1.0, "{r}");
    // inner 1 * boost 3 = 3, times weight 2 = 6 >= 5.
    let r = search(&node, "fs", json!({"query": {"function_score": {"query": {"match_all": {}}, "weight": 2, "boost": 3, "boost_mode": "multiply", "min_score": 5}}}));
    assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
    assert_eq!(r["hits"]["hits"][0]["_score"], 6.0, "{r}");
    // sum: 2 * 1 + 10 = 12 (not (1 + 10) * 2 = 22).
    let r = search(&node, "fs", json!({"query": {"function_score": {"query": {"match_all": {}}, "weight": 10, "boost": 2, "boost_mode": "sum"}}}));
    assert_eq!(r["hits"]["hits"][0]["_score"], 12.0, "{r}");
}

/// #453: sigmoid at the pivot is 0.5 even when the powers overflow f32.
/// #449: ES applies explicit parameters to the stored (reciprocal) value of
/// a negative-impact feature, and refuses `log` on one.
#[test]
fn rank_feature_sigmoid_and_negative_impact() {
    let (_d, node) = open();
    index(&node, "rf", one_shard(json!({"properties": {"pop": {"type": "rank_feature"}, "len": {"type": "rank_feature", "positive_score_impact": false}}})),
        &[json!({"pop": 16384, "len": 8})]);
    let r = search(&node, "rf", json!({"query": {"rank_feature": {"field": "pop", "sigmoid": {"pivot": 16384, "exponent": 10}}}}));
    assert_eq!(r["hits"]["hits"][0]["_score"], 0.5, "{r}");
    let r = search(&node, "rf", json!({"query": {"rank_feature": {"field": "len", "saturation": {"pivot": 8}}}}));
    let s = r["hits"]["hits"][0]["_score"].as_f64().unwrap();
    assert!((s - 0.125 / 8.125).abs() < 1e-3, "{r}");
    let e = try_search(&node, "rf", json!({"query": {"rank_feature": {"field": "len", "log": {"scaling_factor": 1}}}})).unwrap_err();
    assert!(format!("{e:?}").contains("Cannot use the [log] function with a field that has a negative score impact"), "{e:?}");
}

/// #472: HDR percentile ranks over a histogram field count at-or-below.
/// #473: histogram and raw indices aggregate together, in either order.
#[test]
fn histogram_hdr_ranks_and_mixed_indices() {
    let (_d, node) = open();
    index(&node, "h1", one_shard(json!({"properties": {"latency": {"type": "histogram"}}})), &[json!({"latency": {"values": [10.0, 1000.0], "counts": [99, 1]}})]);
    let r = search(&node, "h1", json!({"size": 0, "aggs": {"r": {"percentile_ranks": {"field": "latency", "values": [10], "hdr": {"number_of_significant_value_digits": 3}}}}}));
    assert_eq!(r["aggregations"]["r"]["values"]["10.0"], 99.0, "{r}");
    index(&node, "h2", one_shard(json!({"properties": {"latency": {"type": "histogram"}}})), &[json!({"latency": {"values": [10.0], "counts": [2]}})]);
    index(&node, "raw", one_shard(json!({"properties": {"latency": {"type": "double"}}})), &[json!({"latency": 100.0})]);
    for targets in ["h2,raw", "raw,h2"] {
        let r = search(&node, targets, json!({"size": 0, "aggs": {"s": {"sum": {"field": "latency"}}, "n": {"value_count": {"field": "latency"}}, "mx": {"max": {"field": "latency"}}}}));
        assert_eq!(r["aggregations"]["s"]["value"], 120.0, "{targets}: {r}");
        assert_eq!(r["aggregations"]["n"]["value"], 3, "{targets}: {r}");
        assert_eq!(r["aggregations"]["mx"]["value"], 100.0, "{targets}: {r}");
    }
}

fn tsdb(node: &Node, name: &str, mappings: Value, docs: &[Value]) {
    let def = json!({"settings": {"index.mode": "time_series", "index.routing_path": ["host"],
        "index.time_series.start_time": "2024-01-01T00:00:00Z", "index.time_series.end_time": "2025-01-01T00:00:00Z",
        "number_of_shards": 1, "number_of_replicas": 0}, "mappings": mappings});
    let svc = node.create_index(name, &def).unwrap();
    let w = docs.iter().map(|d| WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
    for r in svc.write(w, true) {
        r.unwrap();
    }
    svc.refresh().unwrap();
}

/// #487: flattened `time_series_dimensions` keys are dimensions of the
/// `time_series` buckets. #488: a `bucket_sort` keeps each bucket's key.
#[test]
fn time_series_dimensions_and_sorted_keys() {
    let (_d, node) = open();
    tsdb(&node, "m", json!({"properties": {"host": {"type": "keyword", "time_series_dimension": true},
        "labels": {"type": "flattened", "time_series_dimensions": ["region"]}, "@timestamp": {"type": "date"}, "v": {"type": "long", "time_series_metric": "gauge"}}}),
        &[json!({"host": "web1", "labels": {"region": "us"}, "@timestamp": "2024-06-01T00:00:00Z", "v": 1}),
          json!({"host": "web1", "labels": {"region": "eu"}, "@timestamp": "2024-06-01T00:00:00Z", "v": 10}),
          json!({"host": "web1", "labels": {"region": "eu"}, "@timestamp": "2024-06-01T00:01:00Z", "v": 10})]);
    let r = search(&node, "m", json!({"size": 0, "aggs": {"ts": {"time_series": {"keyed": false}, "aggs": {"s": {"sum": {"field": "v"}}}}}}));
    let b = r["aggregations"]["ts"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 2, "{r}");
    for x in b {
        let want = if x["key"]["labels.region"] == "eu" { 20.0 } else { 1.0 };
        assert_eq!(x["s"]["value"], want, "{r}");
    }
    let r = search(&node, "m", json!({"size": 0, "aggs": {"ts": {"time_series": {}, "aggs": {"top": {"bucket_sort": {"sort": [{"_count": "desc"}], "size": 1}}}}}}));
    let obj = r["aggregations"]["ts"]["buckets"].as_object().unwrap();
    assert_eq!(obj.len(), 1, "{r}");
    let (k, v) = obj.iter().next().unwrap();
    assert_eq!(v["doc_count"], 2, "{r}");
    assert!(k.contains("eu"), "{k}: {r}");
}

/// #489: inside `time_series`, geo_line simplifies the whole route (its
/// end stays) instead of keeping its first `size` points.
#[test]
fn time_series_geo_line_keeps_the_route_end() {
    let (_d, node) = open();
    let docs: Vec<Value> = (0..10).map(|i| json!({"host": "car", "@timestamp": format!("2024-06-01T00:00:{i:02}Z"), "p": {"lat": 0.0, "lon": i as f64 + if i == 5 { 0.0 } else { 0.0 }}})).collect();
    tsdb(&node, "g", json!({"properties": {"host": {"type": "keyword", "time_series_dimension": true}, "@timestamp": {"type": "date"}, "p": {"type": "geo_point"}}}), &docs);
    let r = search(&node, "g", json!({"size": 0, "aggs": {"ts": {"time_series": {"keyed": false}, "aggs": {"line": {"geo_line": {"point": {"field": "p"}, "size": 3}}}}}}));
    let line = &r["aggregations"]["ts"]["buckets"][0]["line"];
    let coords = line["geometry"]["coordinates"].as_array().unwrap();
    assert_eq!(coords.len(), 3, "{r}");
    assert_eq!(coords[0][0], 0.0, "{r}");
    assert_eq!(coords[2][0], 9.0, "the route's end: {r}");
    assert_eq!(line["properties"]["complete"], false);
}

/// #490: a calendar rate divides by the bucket's own width (multi-valued dates).
#[test]
fn calendar_rate_uses_the_bucket_width() {
    let (_d, node) = open();
    index(&node, "rt", one_shard(json!({"properties": {"d": {"type": "date"}, "amount": {"type": "long"}}})), &[json!({"d": ["2025-02-15", "2025-03-15"], "amount": 3100})]);
    let r = search(&node, "rt", json!({"size": 0, "aggs": {"m": {"date_histogram": {"field": "d", "calendar_interval": "month"}, "aggs": {"r": {"rate": {"field": "amount", "unit": "day"}}}}}}));
    let b = r["aggregations"]["m"]["buckets"].as_array().unwrap();
    let march = b.iter().find(|x| x["key_as_string"].as_str().unwrap().starts_with("2025-03")).unwrap();
    assert!((march["r"]["value"].as_f64().unwrap() - 100.0).abs() < 1e-9, "{r}");
}

/// #494: hard bounds hold the rounded bucket keys.
#[test]
fn hard_bounds_bound_buckets() {
    let (_d, node) = open();
    index(&node, "hb", one_shard(json!({"properties": {"x": {"type": "long"}}})), &[json!({"x": 50}), json!({"x": 120}), json!({"x": 180})]);
    for sub in [json!(null), json!({"s": {"sum": {"field": "x"}}})] {
        let mut h = json!({"histogram": {"field": "x", "interval": 100, "hard_bounds": {"min": 10, "max": 150}, "min_doc_count": 1}});
        if !sub.is_null() {
            h["aggs"] = sub;
        }
        let r = search(&node, "hb", json!({"size": 0, "aggs": {"h": h}}));
        let b = r["aggregations"]["h"]["buckets"].as_array().unwrap();
        assert_eq!(b.len(), 1, "{r}");
        assert_eq!((b[0]["key"].as_f64(), b[0]["doc_count"].as_u64()), (Some(100.0), Some(2)), "{r}");
    }
}

/// #447: a date_range histogram uses calendar and time-zone rounding.
#[test]
fn date_range_histogram_calendar_and_zone() {
    let (_d, node) = open();
    index(&node, "dr", one_shard(json!({"properties": {"r": {"type": "date_range"}}})), &[json!({"r": {"gte": "2025-01-01T01:00:00Z", "lte": "2025-01-01T01:30:00Z"}})]);
    let r = search(&node, "dr", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "r", "calendar_interval": "day", "time_zone": "-08:00"}}}}));
    let b = r["aggregations"]["h"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 1, "{r}");
    assert_eq!(b[0]["key"], 1735632000000i64, "local Dec 31 starts 2024-12-31T08:00Z: {r}");
    let r = search(&node, "dr", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "r", "calendar_interval": "month"}}}}));
    assert_eq!(r["aggregations"]["h"]["buckets"][0]["key_as_string"], "2025-01-01T00:00:00.000Z", "{r}");
}

/// #514: derivative `unit` adds `normalized_value`; an empty bucket under
/// `skip` is a gap (null derivatives), not a measured zero.
#[test]
fn derivative_units_and_gaps() {
    let (_d, node) = open();
    index(&node, "dv", one_shard(json!({"properties": {"t": {"type": "date"}, "v": {"type": "long"}}})),
        &[json!({"t": "2025-01-01T00:00:00Z", "v": 100}), json!({"t": "2025-01-01T00:05:00Z", "v": 400})]);
    let r = search(&node, "dv", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "t", "fixed_interval": "5m"}, "aggs": {"s": {"sum": {"field": "v"}}, "d": {"derivative": {"buckets_path": "s", "unit": "minute"}}}}}}));
    let d = &r["aggregations"]["h"]["buckets"][1]["d"];
    assert_eq!((d["value"].as_f64(), d["normalized_value"].as_f64()), (Some(300.0), Some(60.0)), "{r}");
    index(&node, "dg", one_shard(json!({"properties": {"t": {"type": "date"}, "v": {"type": "long"}}})),
        &[json!({"t": "2025-01-01", "v": 10}), json!({"t": "2025-01-03", "v": 20})]);
    let r = search(&node, "dg", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "t", "calendar_interval": "day"}, "aggs": {"s": {"sum": {"field": "v"}}, "d": {"derivative": {"buckets_path": "s"}}}}}}));
    let b = r["aggregations"]["h"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 3, "{r}");
    assert!(b[1]["d"]["value"].is_null() && b[2]["d"]["value"].is_null(), "{r}");
}

/// #515: bucket_sort by `_key` orders string keys. #518: percentiles_bucket
/// rank `round(p/100 * (n-1))`, and `keyed: false`.
#[test]
fn bucket_sort_string_keys_and_percentiles_bucket() {
    let (_d, node) = open();
    index(&node, "bs", one_shard(json!({"properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}})),
        &[json!({"k": "a", "v": 10}), json!({"k": "b", "v": 500}), json!({"k": "b", "v": 500})]);
    let r = search(&node, "bs", json!({"size": 0, "aggs": {"t": {"terms": {"field": "k"}, "aggs": {"o": {"bucket_sort": {"sort": [{"_key": "asc"}], "size": 1}}}}}}));
    assert_eq!(r["aggregations"]["t"]["buckets"][0]["key"], "a", "{r}");
    assert_eq!(r["aggregations"]["t"]["buckets"].as_array().unwrap().len(), 1);
    let r = search(&node, "bs", json!({"size": 0, "aggs": {"t": {"terms": {"field": "k"}, "aggs": {"s": {"sum": {"field": "v"}}}},
        "p": {"percentiles_bucket": {"buckets_path": "t>s", "percents": [50]}}, "pa": {"percentiles_bucket": {"buckets_path": "t>s", "percents": [50], "keyed": false}}}}));
    assert_eq!(r["aggregations"]["p"]["values"]["50.0"], 1000.0, "{r}");
    assert_eq!(r["aggregations"]["pa"]["values"], json!([{"key": 50.0, "value": 1000.0}]), "{r}");
}

/// #516: normalize skips empty buckets.
#[test]
fn normalize_skips_empty_buckets() {
    let (_d, node) = open();
    index(&node, "nz", one_shard(json!({"properties": {"x": {"type": "long"}, "v": {"type": "long"}}})), &[json!({"x": 0, "v": 10}), json!({"x": 20, "v": 20})]);
    let r = search(&node, "nz", json!({"size": 0, "aggs": {"h": {"histogram": {"field": "x", "interval": 10}, "aggs": {"s": {"sum": {"field": "v"}}, "n": {"normalize": {"buckets_path": "s", "method": "rescale_0_1"}}}}}}));
    let b = r["aggregations"]["h"]["buckets"].as_array().unwrap();
    let n: Vec<Value> = b.iter().map(|x| x["n"]["value"].clone()).collect();
    assert_eq!(n, vec![json!(0.0), Value::Null, json!(1.0)], "{r}");
}

/// #517: `buckets_path` selects a named bucket of a multi-bucket agg.
#[test]
fn buckets_path_selects_named_buckets() {
    let (_d, node) = open();
    index(&node, "nb", one_shard(json!({"properties": {"t": {"type": "date"}, "type": {"type": "keyword"}, "sales": {"type": "long"}}})),
        &[json!({"t": "2025-01-01", "type": "hat", "sales": 5}), json!({"t": "2025-01-01", "type": "shoe", "sales": 100}), json!({"t": "2025-01-02", "type": "hat", "sales": 7})]);
    let r = search(&node, "nb", json!({"size": 0, "aggs": {
        "days": {"date_histogram": {"field": "t", "calendar_interval": "day"}, "aggs": {"types": {"terms": {"field": "type"}, "aggs": {"sales": {"sum": {"field": "sales"}}}},
            "hat_trend": {"derivative": {"buckets_path": "types['hat']>sales"}}}},
        "hats": {"sum_bucket": {"buckets_path": "days>types['hat']>sales"}}}}));
    assert_eq!(r["aggregations"]["hats"]["value"], 12.0, "{r}");
    assert_eq!(r["aggregations"]["days"]["buckets"][1]["hat_trend"]["value"], 2.0, "{r}");
}

/// #527: an ip composite pages with its own after_key, both orders.
#[test]
fn composite_ip_after_key_pages() {
    let (_d, node) = open();
    index(&node, "ip", one_shard(json!({"properties": {"a": {"type": "ip"}}})), &[json!({"a": "10.0.0.1"}), json!({"a": "10.0.0.2"})]);
    for (order, want) in [("asc", ["10.0.0.1", "10.0.0.2"]), ("desc", ["10.0.0.2", "10.0.0.1"])] {
        let mut after: Option<Value> = None;
        let mut seen = Vec::new();
        for _ in 0..3 {
            let mut c = json!({"size": 1, "sources": [{"a": {"terms": {"field": "a", "order": order}}}]});
            if let Some(a) = &after {
                c["after"] = a.clone();
            }
            let r = search(&node, "ip", json!({"size": 0, "aggs": {"c": {"composite": c}}}));
            let b = r["aggregations"]["c"]["buckets"].as_array().unwrap().clone();
            if b.is_empty() {
                break;
            }
            seen.push(b[0]["key"]["a"].as_str().unwrap().to_string());
            after = Some(r["aggregations"]["c"]["after_key"].clone());
        }
        assert_eq!(seen, want, "{order}");
    }
}

/// #528: version terms order by version precedence.
#[test]
fn version_terms_order_by_version() {
    let (_d, node) = open();
    index(&node, "vr", one_shard(json!({"properties": {"r": {"type": "version"}}})), &[json!({"r": "2.0.0"}), json!({"r": "10.0.0"})]);
    for (o, want) in [("asc", "2.0.0"), ("desc", "10.0.0")] {
        let r = search(&node, "vr", json!({"size": 0, "aggs": {"t": {"terms": {"field": "r", "size": 1, "order": {"_key": o}}}}}));
        assert_eq!(r["aggregations"]["t"]["buckets"][0]["key"], want, "{o}: {r}");
    }
}

/// #529: geo aggregations honour `missing`.
#[test]
fn geo_aggregations_honour_missing() {
    let (_d, node) = open();
    index(&node, "gm", one_shard(json!({"properties": {"p": {"type": "geo_point"}}})), &[json!({"p": {"lat": 10, "lon": 10}}), json!({})]);
    let r = search(&node, "gm", json!({"size": 0, "aggs": {
        "d": {"geo_distance": {"field": "p", "origin": [0, 0], "missing": "0,0", "ranges": [{"from": 0, "to": 100}]}},
        "c": {"geo_centroid": {"field": "p", "missing": "0,0"}},
        "g": {"geotile_grid": {"field": "p", "precision": 1, "missing": "0,0"}}}}));
    assert_eq!(r["aggregations"]["d"]["buckets"][0]["doc_count"], 1, "{r}");
    assert_eq!(r["aggregations"]["c"]["count"], 2, "{r}");
    assert!((r["aggregations"]["c"]["location"]["lat"].as_f64().unwrap() - 5.0).abs() < 1e-3, "{r}");
    let g: u64 = r["aggregations"]["g"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
    assert_eq!(g, 2, "{r}");
}

/// #530: GND with a separate background keeps a foreground-only term.
#[test]
fn gnd_separate_background_keeps_new_terms() {
    let (_d, node) = open();
    let mut docs: Vec<Value> = (0..10).map(|_| json!({"period": "now", "sig": "NEW_FAILURE"})).collect();
    docs.extend((0..100).map(|i| json!({"period": "before", "sig": format!("old{}", i % 5)})));
    index(&node, "gn", one_shard(json!({"properties": {"period": {"type": "keyword"}, "sig": {"type": "keyword"}}})), &docs);
    let r = search(&node, "gn", json!({"size": 0, "query": {"term": {"period": "now"}}, "aggs": {"s": {"significant_terms": {"field": "sig",
        "background_filter": {"term": {"period": "before"}}, "gnd": {"background_is_superset": false}}}}}));
    let b = r["aggregations"]["s"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 1, "{r}");
    assert_eq!(b[0]["key"], "NEW_FAILURE");
    assert_eq!(b[0]["score"], 1.0, "{r}");
}

/// #528 (composite): a version terms source orders and pages by version.
#[test]
fn composite_version_source_pages_by_version() {
    let (_d, node) = open();
    index(&node, "cv", one_shard(json!({"properties": {"r": {"type": "version"}}})), &[json!({"r": "2.0.0"}), json!({"r": "10.0.0"}), json!({"r": "1.5.0"})]);
    for (order, want) in [("asc", vec!["1.5.0", "2.0.0", "10.0.0"]), ("desc", vec!["10.0.0", "2.0.0", "1.5.0"])] {
        let mut after: Option<Value> = None;
        let mut seen = Vec::new();
        for _ in 0..4 {
            let mut c = json!({"size": 1, "sources": [{"r": {"terms": {"field": "r", "order": order}}}]});
            if let Some(a) = &after {
                c["after"] = a.clone();
            }
            let r = search(&node, "cv", json!({"size": 0, "aggs": {"c": {"composite": c}}}));
            let b = r["aggregations"]["c"]["buckets"].as_array().unwrap().clone();
            if b.is_empty() {
                break;
            }
            seen.push(b[0]["key"]["r"].as_str().unwrap().to_string());
            after = Some(r["aggregations"]["c"]["after_key"].clone());
        }
        assert_eq!(seen, want, "{order}");
    }
}

/// #447: hard bounds clip each stored range (end inclusive, as ES's range
/// collector); extended bounds add empty buckets.
#[test]
fn date_range_histogram_bounds() {
    let (_d, node) = open();
    index(&node, "drb", one_shard(json!({"properties": {"r": {"type": "date_range"}}})), &[json!({"r": {"gte": "2025-01-01", "lte": "2025-01-05"}})]);
    let keys = |r: &Value| -> Vec<String> { r["aggregations"]["h"]["buckets"].as_array().unwrap().iter().map(|b| b["key_as_string"].as_str().unwrap()[..10].to_string()).collect() };
    let r = search(&node, "drb", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "r", "calendar_interval": "day", "hard_bounds": {"min": "2025-01-02", "max": "2025-01-03"}}}}}));
    assert_eq!(keys(&r), vec!["2025-01-02", "2025-01-03"], "{r}");
    let r = search(&node, "drb", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "r", "calendar_interval": "day", "min_doc_count": 0,
        "hard_bounds": {"min": "2025-01-02", "max": "2025-01-02"}, "extended_bounds": {"min": "2025-01-02", "max": "2025-01-02"}}}}}));
    assert_eq!(keys(&r), vec!["2025-01-02"], "{r}");
    let r = search(&node, "drb", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "r", "calendar_interval": "day", "min_doc_count": 0,
        "extended_bounds": {"min": "2024-12-30", "max": "2025-01-07"}}}}}));
    assert_eq!(keys(&r).len(), 9, "{r}");
}

/// #494 (dates): extended bounds round without the offset, then take it
/// (ES `LongBounds.round`); extended bounds outside hard ones are refused.
#[test]
fn date_histogram_extended_bounds_with_offset_and_hard_bounds() {
    let (_d, node) = open();
    index(&node, "dho", one_shard(json!({"properties": {"t": {"type": "date"}}})), &[json!({"t": "2025-01-02T12:00:00Z"})]);
    let r = search(&node, "dho", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "t", "calendar_interval": "day", "offset": "+6h", "min_doc_count": 0,
        "extended_bounds": {"min": "2025-01-01"}}}}}));
    let keys: Vec<&str> = r["aggregations"]["h"]["buckets"].as_array().unwrap().iter().map(|b| b["key_as_string"].as_str().unwrap()).collect();
    assert_eq!(keys, vec!["2025-01-01T06:00:00.000Z", "2025-01-02T06:00:00.000Z"], "{r}");
    let e = try_search(&node, "dho", json!({"size": 0, "aggs": {"h": {"date_histogram": {"field": "t", "calendar_interval": "day",
        "hard_bounds": {"min": "2025-01-02", "max": "2025-01-03"}, "extended_bounds": {"min": "2025-01-01", "max": "2025-01-03"}}}}})).unwrap_err();
    assert!(format!("{e:?}").contains("Extended bounds have to be inside hard bounds, hard bounds: [2025-01-02--2025-01-03], extended bounds: [2025-01-01--2025-01-03]"), "{e:?}");
}

/// #517: ES path tokens are trimmed; a bucket key needs single quotes.
#[test]
fn buckets_path_key_syntax_follows_es() {
    let (_d, node) = open();
    index(&node, "nk", one_shard(json!({"properties": {"type": {"type": "keyword"}, "sales": {"type": "long"}}})),
        &[json!({"type": "hat", "sales": 5}), json!({"type": "shoe", "sales": 100})]);
    let r = search(&node, "nk", json!({"size": 0, "aggs": {"types": {"terms": {"field": "type"}, "aggs": {"sales": {"sum": {"field": "sales"}}}},
        "spaced": {"sum_bucket": {"buckets_path": "types > sales"}}}}));
    assert_eq!(r["aggregations"]["spaced"]["value"], 105.0, "{r}");
    // A bucket key needs ES's single quotes: unquoted names no bucket.
    let r = search(&node, "nk", json!({"size": 0, "aggs": {"d": {"filters": {"filters": {"all": {"match_all": {}}}}, "aggs": {"types": {"terms": {"field": "type"}, "aggs": {"sales": {"sum": {"field": "sales"}}}}}},
        "quoted": {"sum_bucket": {"buckets_path": "d>types['hat']>sales"}}, "bare": {"sum_bucket": {"buckets_path": "d>types[hat]>sales"}}}}));
    assert_eq!(r["aggregations"]["quoted"]["value"], 5.0, "{r}");
    assert_eq!(r["aggregations"]["bare"]["value"], 0.0, "{r}");
}

/// #471: the explanation carries the boost inside the function score.
#[test]
fn function_score_explain_shows_the_boost() {
    let (_d, node) = open();
    index(&node, "fe", one_shard(json!({"properties": {"t": {"type": "keyword"}}})), &[json!({"t": "a"})]);
    let r = search(&node, "fe", json!({"explain": true, "query": {"function_score": {"query": {"match_all": {}}, "weight": 2, "boost": 3, "boost_mode": "multiply"}}}));
    let h = &r["hits"]["hits"][0];
    assert_eq!(h["_score"], 6.0, "{r}");
    let e = &h["_explanation"];
    assert_eq!(e["value"], 6.0, "{r}");
    assert_eq!(e["details"][0]["value"], 3.0, "the boosted sub-query: {r}");
}
