//! Rally `geoshape` track mapping: a `double` field computed at index time
//! by a script that declares `ScriptDocValues.Geometry` (a dotted inner
//! class type name). The script must compile, every document must index,
//! and the value must be ES's `h * h + w * w` of the shape's spherical
//! Mercator bounding box.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

const SCRIPT: &str = "ScriptDocValues.Geometry geometry = doc[\"shape\"];double w = geometry.getMercatorWidth();double h = geometry.getMercatorHeight();emit(h * h + w * w);";

fn search(node: &Arc<Node>, index: &str, body: Value) -> Value {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    for t in &targets {
        t.svc.refresh().unwrap();
    }
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {e:?}"));
    serde_json::from_str(&out.json).unwrap()
}

/// ES `SphericalMercatorUtils`.
const MERC_MAX: f64 = 20037508.342789244;
fn lon_merc(lon: f64) -> f64 {
    lon * MERC_MAX / 180.0
}
fn lat_merc(lat: f64) -> f64 {
    let y = ((90.0 + lat) * std::f64::consts::PI / 360.0).tan().ln() / (std::f64::consts::PI / 180.0);
    y * MERC_MAX / 180.0
}

#[test]
fn geoshape_track_size_field_compiles_and_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "osmshapes",
        &json!({"mappings": {"dynamic": "strict", "properties": {
            "shape": {"type": "geo_shape"},
            "size": {"type": "double", "script": {"source": SCRIPT}}
        }}}),
    )
    .unwrap_or_else(|e| panic!("the geoshape mapping must be accepted: {e:?}"));
    let docs = [
        ("poly", json!({"shape": "POLYGON ((10.0 50.0, 12.0 50.0, 12.0 52.5, 10.0 52.5, 10.0 50.0))"}), (10.0, 12.0, 50.0, 52.5)),
        ("line", json!({"shape": {"type": "linestring", "coordinates": [[-3.0, 40.0], [-1.0, 41.0]]}}), (-3.0, -1.0, 40.0, 41.0)),
        ("point", json!({"shape": "POINT (7.0 55.0)"}), (7.0, 7.0, 55.0, 55.0)),
    ];
    let svc = node.get_index("osmshapes").unwrap();
    for (id, src, _) in &docs {
        let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: Default::default() }], false);
        r.into_iter().next().unwrap().unwrap_or_else(|e| panic!("{id} must index: {e:?}"));
    }
    let r = search(&node, "osmshapes", json!({"size": 10, "fields": ["size"], "_source": false}));
    assert_eq!(r["hits"]["total"]["value"], 3, "{r}");
    for (id, _, (left, right, bottom, top)) in &docs {
        let hit = r["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == *id).unwrap_or_else(|| panic!("{id}: {r}"));
        let got = hit["fields"]["size"][0].as_f64().unwrap_or_else(|| panic!("{id}: {hit}"));
        let (w, h) = (lon_merc(*right) - lon_merc(*left), lat_merc(*top) - lat_merc(*bottom));
        let want = h * h + w * w;
        // Doc-value coordinates are quantized (~1e-7 degrees).
        assert!((got - want).abs() <= 1e-4 * want.max(1.0), "{id}: {got} vs {want}");
    }
    // The point has an empty box: size 0, and it is searchable as a double.
    let r = search(&node, "osmshapes", json!({"query": {"range": {"size": {"lte": 1.0}}}}));
    assert_eq!(r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect::<Vec<_>>(), vec!["point"], "{r}");
}

/// Rally `geoshape` `index-append-multilinestrings`: ~5% of the OSM
/// multilinestrings encode to more than 64 KiB of WKT (route relations with
/// thousands of members). ES stores the shape as binary doc values with no
/// size limit and indexes them all; the memtable keyword column used to keep
/// a `u16` length, so the value read back truncated, `doc['shape']` was
/// empty and the `size` script failed the bulk item ("Error executing script
/// on field [size]"). The shape must index, compute ES's `size`, round-trip
/// its `_source`, and match a `geo_shape` query on its far end.
#[test]
fn geoshape_track_large_multilinestring_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "osmmultilinestrings",
        &json!({"mappings": {"dynamic": "strict", "properties": {
            "shape": {"type": "geo_shape"},
            "size": {"type": "double", "script": {"source": SCRIPT}}
        }}}),
    )
    .unwrap();
    let lines: Vec<Value> = (0..3000)
        .map(|i| {
            let (x, y) = (8.0 + i as f64 * 0.0003, 49.0 + i as f64 * 0.0003);
            json!([[x, y], [x + 0.0001234, y + 0.0001234]])
        })
        .collect();
    let src = json!({"shape": {"type": "MultiLineString", "coordinates": lines}});
    assert!(src.to_string().len() > 100_000);
    let svc = node.get_index("osmmultilinestrings").unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some("big".into()), source: src.to_string(), routing: None, cond: Default::default() }], false);
    r.into_iter().next().unwrap().unwrap_or_else(|e| panic!("a >64 KiB multilinestring must index: {e:?}"));
    let r = search(&node, "osmmultilinestrings", json!({"fields": ["size"]}));
    let hit = &r["hits"]["hits"][0];
    assert_eq!(hit["_source"], src, "{r}");
    let (left, bottom) = (8.0, 49.0);
    let (right, top) = (8.0 + 2999.0 * 0.0003 + 0.0001234, 49.0 + 2999.0 * 0.0003 + 0.0001234);
    let (w, h) = (lon_merc(right) - lon_merc(left), lat_merc(top) - lat_merc(bottom));
    let want = h * h + w * w;
    let got = hit["fields"]["size"][0].as_f64().unwrap_or_else(|| panic!("{hit}"));
    assert!((got - want).abs() <= 1e-4 * want, "{got} vs {want}");
    // The last member (past the first 64 KiB of WKT) is matched.
    let q = json!({"query": {"geo_shape": {"shape": {"shape": {"type": "envelope", "coordinates": [[right - 0.00006, top - 0.00002], [right - 0.00002, top - 0.00006]]}, "relation": "intersects"}}}});
    assert_eq!(search(&node, "osmmultilinestrings", q)["hits"]["total"]["value"], 1);
}

/// The `index-append-polygons` failures were the same truncation: OSM
/// polygons with thousands of vertices and dozens of holes (82-207 KB of
/// WKT). A WKT polygon past 64 KiB with a hole indexes, and `size` is ES's
/// formula on its extent.
#[test]
fn geoshape_track_large_polygon_with_hole_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "osmpolygons",
        &json!({"mappings": {"dynamic": "strict", "properties": {
            "shape": {"type": "geo_shape"},
            "size": {"type": "double", "script": {"source": SCRIPT}}
        }}}),
    )
    .unwrap();
    // Counter-clockwise outer ring of 4000 vertices, closed.
    let n = 4000;
    let mut outer: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (8.5 + 0.5 * a.cos(), 47.5 + 0.25 * a.sin())
        })
        .collect();
    outer.push(outer[0]);
    let ring = |r: &[(f64, f64)]| r.iter().map(|p| format!("{} {}", p.0, p.1)).collect::<Vec<_>>().join(", ");
    let hole = [(8.4, 47.45), (8.4, 47.55), (8.6, 47.55), (8.6, 47.45), (8.4, 47.45)];
    let wkt = format!("POLYGON(({}), ({}))", ring(&outer), ring(&hole));
    assert!(wkt.len() > 70_000);
    let src = json!({"shape": wkt});
    let svc = node.get_index("osmpolygons").unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some("big".into()), source: src.to_string(), routing: None, cond: Default::default() }], false);
    r.into_iter().next().unwrap().unwrap_or_else(|e| panic!("a >64 KiB polygon must index: {e:?}"));
    let r = search(&node, "osmpolygons", json!({"fields": ["size"]}));
    let hit = &r["hits"]["hits"][0];
    assert_eq!(hit["_source"], src, "{r}");
    let fold = |f: fn(f64, f64) -> f64, init: f64, k: usize| outer.iter().map(|p| if k == 0 { p.0 } else { p.1 }).fold(init, f);
    let (left, right) = (fold(f64::min, f64::INFINITY, 0), fold(f64::max, f64::NEG_INFINITY, 0));
    let (bottom, top) = (fold(f64::min, f64::INFINITY, 1), fold(f64::max, f64::NEG_INFINITY, 1));
    let (w, h) = (lon_merc(right) - lon_merc(left), lat_merc(top) - lat_merc(bottom));
    let want = h * h + w * w;
    let got = hit["fields"]["size"][0].as_f64().unwrap_or_else(|| panic!("{hit}"));
    assert!(got.is_finite() && (got - want).abs() <= 1e-4 * want, "{got} vs {want}");
    // The hole is honoured: a point in it does not intersect, one in the
    // ring's body does.
    let q = |lon: f64, lat: f64| json!({"query": {"geo_shape": {"shape": {"shape": {"type": "point", "coordinates": [lon, lat]}, "relation": "intersects"}}}});
    assert_eq!(search(&node, "osmpolygons", q(8.5, 47.5))["hits"]["total"]["value"], 0);
    assert_eq!(search(&node, "osmpolygons", q(8.9, 47.5))["hits"]["total"]["value"], 1);
}
