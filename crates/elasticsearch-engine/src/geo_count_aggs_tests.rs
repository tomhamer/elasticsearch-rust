//! Aggregations answered from the KD tree, each giving the same response
//! as collecting the docs: `value_count` of the filtering `geo_point`
//! field over a pure geo filter (Rally `distanceFilterCount*`; compared
//! with the `missing` variant, which takes the generic path and counts the
//! same over matches that all hold a point) and count-only `geo_distance`
//! rings (Rally `distanceRange`, `distanceFilterRange`).

use crate::index::{IndexService, WriteReq};
use crate::shard::WriteCond;
use crate::settings;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

fn idx(id: usize, body: Value) -> WriteReq {
    WriteReq::Index { id: Some(id.to_string()), source: body.to_string(), routing: None, cond: WriteCond::default() }
}

fn search(s: &Arc<IndexService>, body: Value, typed_keys: bool) -> Value {
    let mut params = std::collections::HashMap::new();
    if typed_keys {
        params.insert("typed_keys".to_string(), "true".to_string());
    }
    let req = crate::search::parse_request(&body, &params).unwrap();
    let targets = vec![crate::search::Target::new(s.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap()
}

fn service(dir: &std::path::Path, multi: bool) -> Arc<IndexService> {
    let s = Arc::new(
        IndexService::create(dir, "g", settings::normalize_index_settings(&json!({"number_of_shards": 1, "number_of_replicas": 0})), &json!({"properties": {"location": {"type": "geo_point"}}}), BTreeMap::new()).unwrap(),
    );
    let mut reqs = Vec::new();
    for i in 0..4000usize {
        let (lat, lon) = (52.0 + (i % 97) as f64 * 0.07, 3.0 + (i % 89) as f64 * 0.09);
        let loc = if multi && i % 9 == 0 { json!([[lon, lat], [lon + 0.5, lat]]) } else { json!([lon, lat]) };
        reqs.push(idx(i, if i % 31 == 0 { json!({}) } else { json!({"location": loc}) }));
        if i == 2000 {
            assert!(s.write(std::mem::take(&mut reqs), true).iter().all(|r| r.is_ok()));
            s.flush().unwrap();
        }
    }
    assert!(s.write(reqs, true).iter().all(|r| r.is_ok()));
    s.refresh().unwrap();
    s
}

fn bodies() -> Vec<Value> {
    let circle = |r: &str, rel: &str| json!({"geo_shape": {"location": {"shape": {"type": "circle", "radius": r, "coordinates": [7, 55]}, "relation": rel}}});
    vec![
        json!({"geo_distance": {"distance": "200km", "location": [7.0, 55.0]}}),
        json!({"bool": {"must": [circle("200km", "intersects"), circle("100km", "disjoint")]}}),
        json!({"bool": {"must": [circle("300km", "intersects"), {"bool": {"must_not": [circle("200km", "intersects")]}}]}}),
        json!({"geo_bounding_box": {"location": {"top_left": [3.0, 58.0], "bottom_right": [8.0, 52.0]}}}),
    ]
}

/// Single- and multi-valued fields, with and without deletes and memtable
/// docs, with typed_keys: the counted response equals the collected one.
#[test]
fn geo_value_count_from_the_tree_equals_collection() {
    for multi in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let s = service(tmp.path(), multi);
        for stage in 0..3 {
            if stage == 1 {
                assert!(s.write(vec![WriteReq::Delete { id: "5".into(), routing: None, cond: WriteCond::default() }], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            if stage == 2 {
                assert!(s.write(vec![idx(9001, json!({"location": [7.0, 55.0]}))], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            for q in bodies() {
                for typed in [false, true] {
                    // `typed_keys` is a URL parameter.
                    let fast = json!({"query": q, "aggs": {"count": {"value_count": {"field": "location"}}, "again": {"value_count": {"field": "location"}}}});
                    let slow = json!({"query": q, "aggs": {"count": {"value_count": {"field": "location", "missing": "0,0"}}, "again": {"value_count": {"field": "location", "missing": "0,0"}}}});
                    let (a, b) = (search(&s, fast, typed), search(&s, slow, typed));
                    assert_eq!(a["aggregations"], b["aggregations"], "{q} multi={multi} stage={stage} typed={typed}");
                    assert_eq!(a["hits"]["total"], b["hits"]["total"], "{q}");
                    if !multi && stage == 0 && !typed {
                        assert_eq!(a["aggregations"]["count"]["value"], a["hits"]["total"]["value"], "{q}");
                    }
                }
            }
        }
    }
}

/// The bucket counts of a count-only `geo_distance` aggregation, under
/// `match_all` and the pure geo filters, single- and multi-valued, with
/// deletes and memtable docs: equal to the collected aggregation's (the
/// same aggregation with a sub-aggregation, which collects the docs).
#[test]
fn geo_distance_counts_from_the_tree_equal_collection() {
    let mut queries = bodies();
    queries.push(json!({"match_all": {}}));
    let agg = |sub: bool, keyed: bool, unit: &str| {
        let mut d = json!({"geo_distance": {"field": "location", "origin": "55.0, 7.0", "unit": unit, "keyed": keyed,
            "ranges": [{"to": 50}, {"from": 50, "to": 100}, {"from": 100, "to": 200, "key": "mid"}, {"from": 200, "to": 300}, {"from": 250}]}});
        if sub {
            d["aggs"] = json!({"n": {"value_count": {"field": "location"}}});
        }
        d
    };
    let strip = |mut v: Value| {
        match &mut v["buckets"] {
            Value::Array(a) => a.iter_mut().for_each(|b| { b.as_object_mut().unwrap().remove("n"); }),
            Value::Object(o) => o.values_mut().for_each(|b| { b.as_object_mut().unwrap().remove("n"); }),
            _ => panic!("{v}"),
        }
        v
    };
    for multi in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let s = service(tmp.path(), multi);
        for stage in 0..3 {
            if stage == 1 {
                assert!(s.write(vec![WriteReq::Delete { id: "5".into(), routing: None, cond: WriteCond::default() }], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            if stage == 2 {
                assert!(s.write(vec![idx(9001, json!({"location": [7.0, 55.0]}))], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            for q in &queries {
                for (keyed, unit) in [(false, "km"), (true, "km"), (false, "mi")] {
                    // Under `match_all` a value_count is not counted from
                    // the tree: the rings alone.
                    let all = q.get("match_all").is_some();
                    let mut fast = json!({"size": 0, "query": q, "aggs": {"rings": agg(false, keyed, unit)}});
                    let mut slow = json!({"size": 0, "query": q, "aggs": {"rings": agg(true, keyed, unit)}});
                    if !all {
                        fast["aggs"]["count"] = json!({"value_count": {"field": "location"}});
                        slow["aggs"]["count"] = json!({"value_count": {"field": "location", "missing": "0,0"}});
                    }
                    let (a, b) = (search(&s, fast, false), search(&s, slow, false));
                    assert_eq!(a["aggregations"]["rings"], strip(b["aggregations"]["rings"].clone()), "{q} multi={multi} stage={stage} keyed={keyed} {unit}");
                    if !all {
                        assert_eq!(a["aggregations"]["count"], b["aggregations"]["count"], "{q}");
                    }
                }
            }
        }
    }
}

/// A ring's box classification never contradicts its point test: every
/// sampled point of an `Inside` box passes, of an `Outside` box fails.
#[test]
fn ring_classification_is_conservative() {
    use rustic_core::geo::{PointPredicate, Relation};
    let def = json!({"geo_distance": {"field": "location", "origin": [7.0, 55.0], "unit": "km",
        "ranges": [{"to": 100}, {"from": 100, "to": 200}, {"from": 200}, {"from": 0, "to": 0.5}, {"to": 0}]}});
    let dc = crate::aggs::DistanceCounts::parse(def.as_object().unwrap()).unwrap();
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut inside, mut outside) = (0, 0);
    for i in 0..dc.buckets() {
        let ring = dc.ring(i);
        for _ in 0..4000 {
            let (h, w) = (rnd() * rnd() * 6.0, rnd() * rnd() * 6.0);
            let (a, c) = (49.0 + rnd() * 12.0, rnd() * 14.0);
            let (b, d) = ((a + h).min(90.0), (c + w).min(180.0));
            let rel = ring.classify(a, b, c, d);
            if rel == Relation::Crosses {
                continue;
            }
            if rel == Relation::Inside { inside += 1 } else { outside += 1 }
            for k in 0..25 {
                let (lat, lon) = match k {
                    0 => (a, c),
                    1 => (a, d),
                    2 => (b, c),
                    3 => (b, d),
                    _ => (a + (b - a) * rnd(), c + (d - c) * rnd()),
                };
                assert_eq!(ring.test(lat, lon), rel == Relation::Inside, "range {i} box {a},{b},{c},{d} point {lat},{lon}");
            }
        }
    }
    assert!(inside > 100 && outside > 100, "{inside} {outside}");
}


/// `geo_centroid` from the KD tree (threshold lowered for the test) under
/// `match_all` and the pure geo filters, single- and multi-valued, with
/// deletes and memtable docs: the collected aggregation's count and, to
/// the last bits of the compensated sums, its location.
#[test]
fn geo_centroid_from_the_tree_equals_collection() {
    crate::search::TEST_CENTROID_MIN_POINTS.with(|c| c.set(1));
    let mut queries = bodies();
    queries.push(json!({"match_all": {}}));
    let close = |a: &Value, b: &Value| (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-9;
    for multi in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let s = service(tmp.path(), multi);
        for stage in 0..3 {
            if stage == 1 {
                assert!(s.write(vec![WriteReq::Delete { id: "5".into(), routing: None, cond: WriteCond::default() }], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            if stage == 2 {
                assert!(s.write(vec![idx(9001, json!({"location": [7.0, 55.0]}))], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            for q in &queries {
                let fast = json!({"size": 0, "query": q, "aggs": {"c": {"geo_centroid": {"field": "location"}}}});
                // Another aggregation makes the whole request collect.
                let slow = json!({"size": 0, "query": q, "aggs": {"c": {"geo_centroid": {"field": "location"}}, "b": {"geo_bounds": {"field": "location"}}}});
                let (a, b) = (search(&s, fast, false), search(&s, slow, false));
                let (a, b) = (&a["aggregations"]["c"], &b["aggregations"]["c"]);
                assert_eq!(a["count"], b["count"], "{q} multi={multi} stage={stage}");
                assert!(close(&a["location"]["lat"], &b["location"]["lat"]) && close(&a["location"]["lon"], &b["location"]["lon"]), "{q} multi={multi} stage={stage}: {a} vs {b}");
            }
        }
    }
    crate::search::TEST_CENTROID_MIN_POINTS.with(|c| c.set(1 << 20));
}


/// `value_count` of a numeric field (Rally geoshape `polygon-*-count`:
/// `size`) under geo_shape relations, a geo_point filter and match_all:
/// counted as the exact matches where every doc has one value, equal to
/// the collected count (the `missing` variant); with docs lacking the
/// field, multi-valued docs, deletes and memtable docs it falls back and
/// still agrees.
#[test]
fn numeric_value_count_from_exact_matches_equals_collection() {
    for variant in 0..3 {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(
            IndexService::create(tmp.path(), "g", settings::normalize_index_settings(&json!({"number_of_shards": 1, "number_of_replicas": 0})), &json!({"properties": {"shape": {"type": "geo_shape"}, "location": {"type": "geo_point"}, "size": {"type": "long"}}}), BTreeMap::new()).unwrap(),
        );
        let mut reqs = Vec::new();
        for i in 0..3000usize {
            let (x, y) = (-1.0 + (i % 89) as f64 * 0.05, 50.0 + (i % 97) as f64 * 0.04);
            let shape = if i % 3 == 0 { format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.03, x + 0.03, y + 0.03, y + 0.03) } else { format!("LINESTRING ({x} {y}, {} {})", x + 0.05, y + 0.02) };
            let size = match variant {
                1 if i % 17 == 0 => json!(null),
                2 if i % 19 == 0 => json!([1, 2]),
                _ => json!(i),
            };
            let mut doc = json!({"shape": shape, "location": [x, y]});
            if !size.is_null() {
                doc["size"] = size;
            }
            reqs.push(idx(i, doc));
            if i == 1500 {
                assert!(s.write(std::mem::take(&mut reqs), true).iter().all(|r| r.is_ok()));
                s.flush().unwrap();
            }
        }
        assert!(s.write(reqs, true).iter().all(|r| r.is_ok()));
        s.flush().unwrap();
        s.refresh().unwrap();
        let poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
        for stage in 0..3 {
            if stage == 1 {
                assert!(s.write(vec![WriteReq::Delete { id: "5".into(), routing: None, cond: WriteCond::default() }], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            if stage == 2 {
                assert!(s.write(vec![idx(9001, json!({"shape": "POINT (1 52)", "location": [1.0, 52.0], "size": 3}))], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            for q in [
                json!({"geo_shape": {"shape": {"shape": poly, "relation": "intersects"}}}),
                json!({"geo_shape": {"shape": {"shape": poly, "relation": "disjoint"}}}),
                json!({"geo_shape": {"shape": {"shape": poly, "relation": "within"}}}),
                json!({"geo_distance": {"distance": "200km", "location": [1.0, 51.0]}}),
                json!({"match_all": {}}),
            ] {
                let fast = json!({"size": 0, "query": q, "aggs": {"count": {"value_count": {"field": "size"}}}});
                let slow = json!({"size": 0, "query": q, "aggs": {"count": {"value_count": {"field": "size", "missing": -1}}}});
                let (a, b) = (search(&s, fast, false), search(&s, slow, false));
                if variant == 0 {
                    assert_eq!(a["aggregations"], b["aggregations"], "{q} variant={variant} stage={stage}");
                } else {
                    // Collected either way; the missing variant also counts
                    // docs without a value, so compare with a plain count.
                    let plain = search(&s, json!({"size": 0, "query": q, "aggs": {"count": {"value_count": {"field": "size"}}, "x": {"max": {"field": "size"}}}}), false);
                    assert_eq!(a["aggregations"]["count"], plain["aggregations"]["count"], "{q} variant={variant} stage={stage}");
                }
            }
        }
    }
}

/// geohash / geotile / geohex grids of the geo_point field, under a box
/// filter, a distance filter and match_all, with and without bounds,
/// with deletes and memtable docs, single- and multi-valued: counted from
/// the tree (whole leaves in one cell at once) equal to the collected
/// aggregation (the same request with an aggregation that is not
/// shortcut).
#[test]
fn geo_point_grids_from_the_tree_equal_collection() {
    let bounds = json!({"top_left": [3.5, 57.0], "bottom_right": [8.0, 53.0]});
    let aggs = [
        json!({"geohash_grid": {"field": "location", "precision": 4}}),
        json!({"geohash_grid": {"field": "location", "precision": 6, "bounds": bounds}}),
        json!({"geotile_grid": {"field": "location", "precision": 9}}),
        json!({"geotile_grid": {"field": "location", "precision": 13, "bounds": bounds, "size": 50}}),
        json!({"geohex_grid": {"field": "location", "precision": 3}}),
        json!({"geohex_grid": {"field": "location", "precision": 6, "bounds": bounds}}),
    ];
    for multi in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let s = service(tmp.path(), multi);
        for stage in 0..3 {
            if stage == 1 {
                assert!(s.write(vec![WriteReq::Delete { id: "5".into(), routing: None, cond: WriteCond::default() }], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            if stage == 2 {
                assert!(s.write(vec![idx(9001, json!({"location": [7.0, 55.0]}))], true).iter().all(|r| r.is_ok()));
                s.refresh().unwrap();
            }
            let mut queries = bodies();
            queries.push(json!({"match_all": {}}));
            for q in &queries {
                for a in &aggs {
                    let fast = json!({"size": 0, "query": q, "aggs": {"g": a}});
                    let slow = json!({"size": 0, "query": q, "aggs": {"g": a, "b": {"geo_bounds": {"field": "location"}}}});
                    let (x, y) = (search(&s, fast, false), search(&s, slow, false));
                    assert_eq!(x["aggregations"]["g"], y["aggregations"]["g"], "{q} {a} multi={multi} stage={stage}");
                }
            }
        }
    }
}

/// geo_bounds / geo_centroid of a geo_shape field from the segment's
/// cached per-doc extents and centroids equal the per-doc header path (the
/// same docs in a memtable), with docs without the field, multi-valued
/// docs and points, lines and polygons; and the ES|QL forms agree.
#[test]
fn shape_bounds_and_centroid_from_the_meta_column_equal_the_headers() {
    let docs: Vec<Value> = (0..3000usize)
        .map(|i| {
            let (x, y) = (-3.0 + (i % 89) as f64 * 0.07, 48.0 + (i % 97) as f64 * 0.05);
            match i % 5 {
                0 => json!({}),
                1 => json!({"shape": format!("POINT ({x} {y})")}),
                2 => json!({"shape": format!("LINESTRING ({x} {y}, {} {})", x + 0.2, y + 0.1)}),
                3 => json!({"shape": [format!("POINT ({x} {y})"), format!("POINT ({} {})", x + 1.0, y)]}),
                _ => json!({"shape": format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.3, x + 0.3, y + 0.2, y + 0.2)}),
            }
        })
        .collect();
    let make = |flush: bool| {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(IndexService::create(tmp.path(), "g", settings::normalize_index_settings(&json!({"number_of_shards": 1, "number_of_replicas": 0})), &json!({"properties": {"shape": {"type": "geo_shape"}}}), BTreeMap::new()).unwrap());
        assert!(s.write(docs.iter().enumerate().map(|(i, d)| idx(i, d.clone())).collect(), true).iter().all(|r| r.is_ok()));
        if flush {
            s.flush().unwrap();
        }
        s.refresh().unwrap();
        (tmp, s)
    };
    let ((_a, sealed), (_b, mem)) = (make(true), make(false));
    let poly = json!({"type": "polygon", "coordinates": [[[-1.0, 49.0], [2.0, 49.0], [2.0, 51.0], [-1.0, 51.0], [-1.0, 49.0]]]});
    for q in [json!({"match_all": {}}), json!({"geo_shape": {"shape": {"shape": poly, "relation": "intersects"}}})] {
        let body = json!({"size": 0, "query": q, "aggs": {"b": {"geo_bounds": {"field": "shape"}}, "c": {"geo_centroid": {"field": "shape"}}}});
        let (x, y) = (search(&sealed, body.clone(), false), search(&mem, body, false));
        assert_eq!(x["aggregations"], y["aggregations"], "{q}");
        assert!(x["aggregations"]["c"]["count"].as_u64().unwrap() > 0);
    }
}

/// `index_time_in_millis` covers a document's whole engine index
/// operation, its parse into fields (here a large polygon's
/// tessellation) included: on a sequential write, nearly all of the
/// write's wall time.
#[test]
fn index_time_covers_document_parsing() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Arc::new(IndexService::create(tmp.path(), "g", settings::normalize_index_settings(&json!({"number_of_shards": 1, "number_of_replicas": 0})), &json!({"properties": {"shape": {"type": "geo_shape"}}}), BTreeMap::new()).unwrap());
    // A ring of 4000 vertices: its tessellation dominates the write.
    let ring: Vec<String> = (0..4000).map(|i| {
        let a = std::f64::consts::TAU * i as f64 / 4000.0;
        let r = 1.0 + 0.3 * ((i * 7919) % 13) as f64 / 13.0;
        format!("{} {}", 5.0 + r * a.cos(), 50.0 + r * a.sin())
    }).collect();
    let wkt = format!("POLYGON (({}, {}))", ring.join(", "), ring[0]);
    let reqs: Vec<WriteReq> = (0..20).map(|i| idx(i, json!({"shape": wkt}))).collect();
    let started = std::time::Instant::now();
    assert!(s.write(reqs, false).iter().all(|r| r.is_ok()));
    let wall = started.elapsed().as_nanos() as u64;
    let counted = s.local_shards()[0].shard.counters.index_time_nanos.load(std::sync::atomic::Ordering::Relaxed);
    assert!(counted * 2 >= wall, "index time {counted} ns of a {wall} ns write");
}
