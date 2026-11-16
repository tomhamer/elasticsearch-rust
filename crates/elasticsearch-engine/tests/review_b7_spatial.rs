//! Production review batch 7, spatial closures (findings 548, 550, 626,
//! 627, 628): searches against a standalone node.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

fn index(node: &Arc<Node>, name: &str, mapping: Value, docs: &[(&str, Value)]) {
    node.create_index(name, &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    let svc = node.get_index(name).unwrap();
    let reqs = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
    for r in svc.write(reqs, true) {
        r.unwrap();
    }
}

fn search_with(node: &Arc<Node>, index: &str, body: Value, lookup: Option<elasticsearch_engine::query::dsl::TermsLookupFn<'_>>) -> elasticsearch_engine::error::Result<Value> {
    let req = search::parse_request(&body, &HashMap::new())?;
    let targets = node.resolve(index, Default::default())?;
    for t in &targets {
        t.svc.refresh()?;
    }
    let opts = ExecOpts { terms_lookup: lookup, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts)?;
    Ok(serde_json::from_str(&out.json).unwrap())
}

fn search(node: &Arc<Node>, index: &str, body: Value) -> Value {
    search_with(node, index, body.clone(), None).unwrap_or_else(|e| panic!("{body}: {e:?}"))
}

fn ids(r: &Value) -> Vec<String> {
    let mut v: Vec<String> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    v.sort();
    v
}

/// Finding 626: spatial queries score their boost, 1 by default (ES
/// `ConstantScoreQuery`), so `min_score: 0.5` keeps the match; an explicit
/// boost is kept.
#[test]
fn spatial_queries_score_one_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    index(&node, "g", json!({"properties": {"loc": {"type": "geo_point"}, "area": {"type": "geo_shape"}, "p": {"type": "point"}}}),
        &[("1", json!({"loc": [0, 0], "area": "POINT (0 0)", "p": [0, 0]}))]);
    let env = json!({"type": "envelope", "coordinates": [[-1, 1], [1, -1]]});
    for (q, field) in [("geo_shape", "loc"), ("geo_shape", "area"), ("shape", "p")] {
        let r = search(&node, "g", json!({"query": {q: {field: {"shape": env}}}, "min_score": 0.5}));
        assert_eq!(ids(&r), vec!["1"], "{q} {field}: {r}");
        assert_eq!(r["hits"]["hits"][0]["_score"], 1.0, "{q} {field}: {r}");
        let r = search(&node, "g", json!({"query": {q: {field: {"shape": env}, "boost": 3}}}));
        assert_eq!(r["hits"]["hits"][0]["_score"], 3.0, "{q} {field}: {r}");
    }
}

/// Finding 627: Cartesian points and shapes match in their
/// single-precision encoding: 1.00000001 is the float 1, so it intersects
/// the query point (1, 0); a coordinate distinct as a float does not.
#[test]
fn cartesian_matching_uses_float_coordinates() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    index(&node, "c", json!({"properties": {"p": {"type": "point"}, "s": {"type": "shape"}}}),
        &[("near", json!({"p": [1.00000001, 0], "s": "POINT (1.00000001 0)"})), ("far", json!({"p": [1.001, 0], "s": "POINT (1.001 0)"}))]);
    for field in ["p", "s"] {
        let r = search(&node, "c", json!({"query": {"shape": {field: {"shape": {"type": "point", "coordinates": [1, 0]}, "relation": "intersects"}}}}));
        assert_eq!(ids(&r), vec!["near"], "{field}: {r}");
    }
}

/// Finding 628: every bounds form ES's bounding box parser accepts limits a
/// geo grid aggregation (sides, either corner pair, WKT): the distant
/// point's cell is not reported; a malformed bounds is an error.
#[test]
fn geo_grid_bounds_accept_every_form() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    index(&node, "pts", json!({"properties": {"loc": {"type": "geo_point"}}}), &[("1", json!({"loc": [15, 0]})), ("2", json!({"loc": [-100, 0]}))]);
    let forms = [
        json!({"top": 10, "left": 10, "bottom": -10, "right": 20}),
        json!({"top_left": [10, 10], "bottom_right": [20, -10]}),
        json!({"top_right": [20, 10], "bottom_left": [10, -10]}),
        json!({"wkt": "BBOX (10, 20, 10, -10)"}),
    ];
    for agg in ["geotile_grid", "geohash_grid", "geohex_grid"] {
        let precision = if agg == "geohash_grid" { 3 } else { 4 };
        for b in &forms {
            let r = search(&node, "pts", json!({"size": 0, "aggs": {"g": {agg: {"field": "loc", "precision": precision, "bounds": b}}}}));
            let buckets = r["aggregations"]["g"]["buckets"].as_array().unwrap();
            assert_eq!(buckets.len(), 1, "{agg} {b}: {r}");
            assert_eq!(buckets[0]["doc_count"], 1, "{agg} {b}: {r}");
        }
        let bad = search_with(&node, "pts", json!({"size": 0, "aggs": {"g": {agg: {"field": "loc", "precision": precision, "bounds": {"top": "x"}}}}}), None);
        assert!(bad.is_err(), "{agg}: a malformed bounds was ignored");
    }
}

/// Finding 550: an `indexed_shape` without `index` reads ES's default
/// `shapes` index, never the searched one (whose same-ID document is a
/// decoy).
#[test]
fn indexed_shape_defaults_to_the_shapes_index() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    index(&node, "assets", json!({"properties": {"loc": {"type": "geo_point"}, "shape": {"type": "geo_shape"}}}),
        &[("a", json!({"loc": [1, 1]})), ("region-a", json!({"loc": [50, 50], "shape": {"type": "envelope", "coordinates": [[49, 51], [51, 49]]}}))]);
    let asked = std::sync::Mutex::new(Vec::new());
    let lookup = |index: &str, id: &str, path: &str, _: Option<&str>| -> elasticsearch_engine::error::Result<Vec<Value>> {
        asked.lock().unwrap().push((index.to_string(), id.to_string(), path.to_string()));
        Ok(match (index, id) {
            ("shapes", "region-a") => vec![json!({"type": "envelope", "coordinates": [[0, 2], [2, 0]]})],
            _ => vec![json!({"type": "envelope", "coordinates": [[49, 51], [51, 49]]})],
        })
    };
    let r = search_with(&node, "assets", json!({"query": {"geo_shape": {"loc": {"indexed_shape": {"id": "region-a"}}}}}), Some(&lookup)).unwrap();
    assert_eq!(ids(&r), vec!["a"], "{r}");
    assert_eq!(asked.lock().unwrap()[0], ("shapes".into(), "region-a".into(), "shape".into()));
}

/// Finding 548: a `geo_shape` field mapped `orientation: left` reads its
/// GeoJSON polygons left-handed (the wide region between -170 and 170); a
/// polygon's own `orientation` overrides the mapping; WKT is unaffected.
#[test]
fn geo_shape_orientation_reaches_indexing() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let ring = json!([[-170, -10], [-170, 10], [170, 10], [170, -10], [-170, -10]]);
    index(&node, "o", json!({"properties": {"s": {"type": "geo_shape", "orientation": "left"}}}), &[
        ("left", json!({"s": {"type": "Polygon", "coordinates": [ring.clone()]}})),
        ("own_right", json!({"s": {"type": "Polygon", "orientation": "right", "coordinates": [ring.clone()]}})),
        ("wkt", json!({"s": "POLYGON ((-170 -10, -170 10, 170 10, 170 -10, -170 -10))"})),
    ]);
    let at = |lon: f64| search(&node, "o", json!({"query": {"geo_shape": {"s": {"shape": {"type": "point", "coordinates": [lon, 0]}}}}}));
    assert_eq!(ids(&at(0.0)), vec!["left"]);
    assert_eq!(ids(&at(179.0)), vec!["own_right", "wkt"]);
}
