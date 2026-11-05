//! `nested` documents (query, inner_hits, sort, nested / reverse_nested
//! aggs) and `geo_point` (queries, `_geo_distance` sort, geo aggs) through
//! the search API, against brute-force oracles over the indexed sources.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::rustic_core::geo::{haversine_meters, quantize};
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

fn search(node: &Arc<Node>, index: &str, body: Value) -> Value {
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    // Searches see each shard as of its last refresh (ES NRT): refresh
    // first, like a REST test's `?refresh`.
    for t in &targets {
        t.svc.refresh().unwrap();
    }
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = search::execute(&targets, &req, &opts).unwrap_or_else(|e| panic!("{body}: {e:?}"));
    serde_json::from_str(&out.json).unwrap()
}

fn ids(r: &Value) -> BTreeSet<String> {
    r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

const AUTHORS: [&str; 3] = ["alice", "bob", "carol"];

fn put(node: &Arc<Node>, index: &str, id: &str, src: &Value) {
    let svc = node.get_index(index).unwrap();
    let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: Default::default() }], false);
    r.into_iter().next().unwrap().unwrap();
}

#[test]
fn nested_documents_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index(
        "blog",
        &json!({"mappings": {"properties": {
            "title": {"type": "keyword"},
            "comments": {"type": "nested", "properties": {
                "author": {"type": "keyword"}, "stars": {"type": "integer"}, "text": {"type": "text"}
            }}
        }}}),
    )
    .unwrap();
    let mut r = Rng(11);
    // id -> comments (author, stars)
    let mut live: BTreeMap<String, Vec<(String, i64)>> = BTreeMap::new();
    for round in 0..3 {
        for i in 0..80 {
            let id = format!("d{}", r.below(150));
            let n = r.below(4);
            let comments: Vec<(String, i64)> = (0..n).map(|_| (AUTHORS[r.below(3) as usize].to_string(), 1 + r.below(5) as i64)).collect();
            let src = json!({
                "title": format!("t{i}"),
                "comments": comments.iter().map(|(a, s)| json!({"author": a, "stars": s, "text": format!("great post by {a}")})).collect::<Vec<_>>()
            });
            put(&node, "blog", &id, &src);
            live.insert(id, comments);
        }
        // Deletes remove whole blocks.
        for _ in 0..10 {
            let id = format!("d{}", r.below(150));
            if live.remove(&id).is_some() {
                let svc = node.get_index("blog").unwrap();
                svc.write(vec![WriteReq::Delete { id, routing: None, cond: Default::default() }], false).into_iter().next().unwrap().unwrap();
            }
        }
        if round == 1 {
            node.get_index("blog").unwrap().shard(0).unwrap().shard.core.write().refresh().unwrap();
        }
    }

    // Top-level queries never see nested docs.
    let all = search(&node, "blog", json!({"size": 1000, "query": {"match_all": {}}}));
    assert_eq!(all["hits"]["total"]["value"], live.len());
    let leak = search(&node, "blog", json!({"query": {"term": {"comments.author": "alice"}}}));
    assert_eq!(leak["hits"]["total"]["value"], 0);

    for a in AUTHORS {
        // nested query: both conditions on the same comment.
        let q = json!({"size": 1000, "query": {"nested": {"path": "comments", "score_mode": "max",
            "query": {"bool": {"must": [{"term": {"comments.author": a}}, {"range": {"comments.stars": {"gte": 3}}}]}},
            "inner_hits": {"size": 10}}}});
        let res = search(&node, "blog", q);
        let want: BTreeSet<String> = live.iter().filter(|(_, c)| c.iter().any(|(x, s)| x == a && *s >= 3)).map(|(k, _)| k.clone()).collect();
        assert_eq!(ids(&res), want, "author {a}");
        for h in res["hits"]["hits"].as_array().unwrap() {
            let id = h["_id"].as_str().unwrap();
            let ih = &h["inner_hits"]["comments"]["hits"];
            let want_offsets: Vec<u64> = live[id].iter().enumerate().filter(|(_, (x, s))| x == a && *s >= 3).map(|(i, _)| i as u64).collect();
            assert_eq!(ih["total"]["value"], want_offsets.len());
            let mut got: Vec<u64> = ih["hits"].as_array().unwrap().iter().map(|x| x["_nested"]["offset"].as_u64().unwrap()).collect();
            got.sort();
            assert_eq!(got, want_offsets, "{id}");
            for x in ih["hits"].as_array().unwrap() {
                assert_eq!(x["_nested"]["field"], "comments");
                assert_eq!(x["_id"], id);
                assert_eq!(x["_source"]["author"], a);
            }
        }
    }

    // Nested sort: best stars descending (mode max); docs without comments last.
    let res = search(&node, "blog", json!({"size": 1000, "sort": [{"comments.stars": {"order": "desc", "mode": "max", "nested": {"path": "comments"}}}, {"title": "asc"}]}));
    let hits = res["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), live.len());
    let mut prev = i64::MAX;
    for h in hits {
        let id = h["_id"].as_str().unwrap();
        let want = live[id].iter().map(|c| c.1).max();
        let got = h["sort"][0].as_i64().unwrap();
        match want {
            Some(w) => {
                assert_eq!(got, w, "{id}");
                assert!(got <= prev);
                prev = got;
            }
            None => prev = i64::MIN,
        }
    }
    // Filtered nested sort.
    let res = search(&node, "blog", json!({"size": 1000, "query": {"nested": {"path": "comments", "query": {"term": {"comments.author": "bob"}}}},
        "sort": [{"comments.stars": {"order": "asc", "mode": "min", "nested": {"path": "comments", "filter": {"term": {"comments.author": "bob"}}}}}]}));
    for h in res["hits"]["hits"].as_array().unwrap() {
        let id = h["_id"].as_str().unwrap();
        let want = live[id].iter().filter(|c| c.0 == "bob").map(|c| c.1).min().unwrap();
        assert_eq!(h["sort"][0].as_i64().unwrap(), want);
    }

    // nested + reverse_nested aggregations.
    let res = search(&node, "blog", json!({"size": 0, "aggs": {"c": {"nested": {"path": "comments"}, "aggs": {
        "by_author": {"terms": {"field": "comments.author"}, "aggs": {"posts": {"reverse_nested": {}}, "avg_stars": {"avg": {"field": "comments.stars"}}}}
    }}}}));
    let total_comments: usize = live.values().map(|c| c.len()).sum();
    assert_eq!(res["aggregations"]["c"]["doc_count"], total_comments);
    for b in res["aggregations"]["c"]["by_author"]["buckets"].as_array().unwrap() {
        let a = b["key"].as_str().unwrap();
        let n = live.values().flatten().filter(|c| c.0 == a).count();
        let posts = live.values().filter(|c| c.iter().any(|x| x.0 == a)).count();
        let sum: i64 = live.values().flatten().filter(|c| c.0 == a).map(|c| c.1).sum();
        assert_eq!(b["doc_count"], n, "{a}");
        assert_eq!(b["posts"]["doc_count"], posts, "{a}");
        assert!((b["avg_stars"]["value"].as_f64().unwrap() - sum as f64 / n as f64).abs() < 1e-9);
    }

    // Errors / ignore_unmapped.
    let req = search::parse_request(&json!({"query": {"nested": {"path": "title", "query": {"match_all": {}}}}}), &HashMap::new()).unwrap();
    let targets = node.resolve("blog", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    assert!(search::execute(&targets, &req, &opts).is_err());
    let res = search(&node, "blog", json!({"query": {"nested": {"path": "nope", "ignore_unmapped": true, "query": {"match_all": {}}}}}));
    assert_eq!(res["hits"]["total"]["value"], 0);
}

#[test]
fn geo_points_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("places", &json!({"mappings": {"properties": {"loc": {"type": "geo_point"}, "n": {"type": "integer"}}}})).unwrap();
    let mut r = Rng(3);
    let mut pts: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for i in 0..400 {
        let (lat, lon) = (35.0 + r.unit() * 30.0, -10.0 + r.unit() * 40.0);
        let id = format!("p{i}");
        // Mixed input forms.
        let loc = match i % 3 {
            0 => json!({"lat": lat, "lon": lon}),
            1 => json!([lon, lat]),
            _ => json!(format!("{lat},{lon}")),
        };
        put(&node, "places", &id, &json!({"loc": loc, "n": i}));
        pts.insert(id, quantize(lat, lon));
        if i == 250 {
            node.get_index("places").unwrap().shard(0).unwrap().shard.core.write().refresh().unwrap();
        }
    }
    put(&node, "places", "nowhere", &json!({"n": -1}));
    let (clat, clon) = (48.85, 2.35);

    // geo_distance query.
    for km in [50.0, 300.0, 1200.0] {
        let res = search(&node, "places", json!({"size": 1000, "query": {"geo_distance": {"distance": format!("{km}km"), "loc": {"lat": clat, "lon": clon}}}}));
        let want: BTreeSet<String> = pts.iter().filter(|(_, (a, b))| haversine_meters(clat, clon, *a, *b) <= km * 1000.0).map(|(k, _)| k.clone()).collect();
        assert_eq!(ids(&res), want, "{km}km");
    }
    // geo_bounding_box.
    let res = search(&node, "places", json!({"size": 1000, "query": {"geo_bounding_box": {"loc": {"top_left": {"lat": 50.0, "lon": 0.0}, "bottom_right": {"lat": 40.0, "lon": 10.0}}}}}));
    let want: BTreeSet<String> = pts.iter().filter(|(_, (a, b))| *a >= 40.0 && *a <= 50.0 && *b >= 0.0 && *b <= 10.0).map(|(k, _)| k.clone()).collect();
    assert_eq!(ids(&res), want);
    // geo_polygon (triangle).
    let res = search(&node, "places", json!({"size": 1000, "query": {"geo_polygon": {"loc": {"points": [{"lat": 40.0, "lon": 0.0}, {"lat": 60.0, "lon": 0.0}, {"lat": 40.0, "lon": 20.0}]}}}}));
    let want: BTreeSet<String> = pts.iter().filter(|(_, (a, b))| *a >= 40.0 && *b >= 0.0 && (*a - 40.0) + *b <= 20.0).map(|(k, _)| k.clone()).collect();
    let got = ids(&res);
    assert!(got.symmetric_difference(&want).count() <= 2, "polygon edge cases only");

    // _geo_distance sort in km, missing last.
    let res = search(&node, "places", json!({"size": 1000, "sort": [{"_geo_distance": {"loc": [clon, clat], "order": "asc", "unit": "km"}}]}));
    let hits = res["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), pts.len() + 1);
    let mut prev = 0.0;
    for h in &hits[..pts.len()] {
        let (a, b) = pts[h["_id"].as_str().unwrap()];
        let d = h["sort"][0].as_f64().unwrap();
        // JSON float parsing may move a coordinate by an ulp, i.e. one
        // quantization step (~5mm): compare to 1cm.
        assert!((d - haversine_meters(clat, clon, a, b) / 1000.0).abs() < 1e-5, "{} vs {}", h["sort"], haversine_meters(clat, clon, a, b) / 1000.0);
        assert!(d >= prev);
        prev = d;
    }
    assert_eq!(hits[pts.len()]["_id"], "nowhere");

    // Geo aggregations.
    let res = search(&node, "places", json!({"size": 0, "aggs": {
        "rings": {"geo_distance": {"field": "loc", "origin": {"lat": clat, "lon": clon}, "unit": "km", "ranges": [{"to": 300}, {"from": 300, "to": 1000}, {"from": 1000}]}},
        "box": {"geo_bounds": {"field": "loc"}},
        "mid": {"geo_centroid": {"field": "loc"}},
        "cells": {"geohash_grid": {"field": "loc", "precision": 2}},
        "tiles": {"geotile_grid": {"field": "loc", "precision": 4}}
    }}));
    let aggs = &res["aggregations"];
    let dist = |p: &(f64, f64)| haversine_meters(clat, clon, p.0, p.1) / 1000.0;
    let rings = aggs["rings"]["buckets"].as_array().unwrap();
    assert_eq!(rings[0]["key"], "*-300.0");
    assert_eq!(rings[0]["doc_count"], pts.values().filter(|p| dist(p) < 300.0).count());
    assert_eq!(rings[1]["doc_count"], pts.values().filter(|p| dist(p) >= 300.0 && dist(p) < 1000.0).count());
    assert_eq!(rings[2]["doc_count"], pts.values().filter(|p| dist(p) >= 1000.0).count());
    let top = pts.values().map(|p| p.0).fold(f64::MIN, f64::max);
    let left = pts.values().map(|p| p.1).fold(f64::MAX, f64::min);
    assert!((aggs["box"]["bounds"]["top_left"]["lat"].as_f64().unwrap() - top).abs() < 1e-6);
    assert!((aggs["box"]["bounds"]["top_left"]["lon"].as_f64().unwrap() - left).abs() < 1e-6);
    assert_eq!(aggs["mid"]["count"], pts.len());
    let mean_lat = pts.values().map(|p| p.0).sum::<f64>() / pts.len() as f64;
    assert!((aggs["mid"]["location"]["lat"].as_f64().unwrap() - mean_lat).abs() < 1e-6);
    for g in ["cells", "tiles"] {
        let b = aggs[g]["buckets"].as_array().unwrap();
        let sum: u64 = b.iter().map(|x| x["doc_count"].as_u64().unwrap()).sum();
        assert_eq!(sum as usize, pts.len(), "{g}");
        assert!(b.windows(2).all(|w| w[0]["doc_count"].as_u64() >= w[1]["doc_count"].as_u64()));
    }
    let want_cell = elasticsearch_engine::rustic_core::geo::geohash(pts["p0"].0, pts["p0"].1, 2);
    assert!(aggs["cells"]["buckets"].as_array().unwrap().iter().any(|b| b["key"] == want_cell.as_str()));
}

#[test]
fn multi_level_inner_hits_and_yaml_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("t", &json!({"mappings": {"properties": {"nested_field": {"type": "nested", "properties": {"sub_nested_field": {"type": "nested"}}}}}})).unwrap();
    put(&node, "t", "1", &json!({"nested_field": [{"field": "a"}, {"field": "value", "sub_nested_field": [{"field": "x"}, {"field": "value"}]}]}));
    // Top-level join straight to the root from a two-level path.
    let r = search(&node, "t", json!({"query": {"nested": {"path": "nested_field.sub_nested_field", "query": {"term": {"nested_field.sub_nested_field.field": "value"}}, "inner_hits": {}}}}));
    assert_eq!(r["hits"]["total"]["value"], 1);
    let h = &r["hits"]["hits"][0]["inner_hits"]["nested_field.sub_nested_field"]["hits"]["hits"][0];
    assert_eq!(h["_id"], "1");
    assert_eq!(h["_nested"], json!({"field": "nested_field", "offset": 1, "_nested": {"field": "sub_nested_field", "offset": 1}}));
    assert_eq!(h["_source"], json!({"field": "value"}));
    // Inner hits of nested-in-nested queries nest.
    let body = json!({"query": {"nested": {"path": "nested_field", "inner_hits": {}, "query": {"nested": {"path": "nested_field.sub_nested_field", "query": {"match_all": {}}, "inner_hits": {}}}}}});
    let req = search::parse_request(&body, &HashMap::from([("rest_total_hits_as_int".to_string(), "true".to_string())])).unwrap();
    let targets = node.resolve("t", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let r: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    let outer = &r["hits"]["hits"][0]["inner_hits"]["nested_field"]["hits"];
    assert_eq!(outer["total"], 1);
    assert_eq!(outer["hits"][0]["_nested"], json!({"field": "nested_field", "offset": 1}));
    let inner = &outer["hits"][0]["inner_hits"]["nested_field.sub_nested_field"]["hits"];
    assert_eq!(inner["total"], 2);
    assert_eq!(inner["hits"][0]["_id"], "1");

    // geo_point null_value feeds geohash_grid (YAML index/90_geo_point).
    node.create_index("g", &json!({"mappings": {"properties": {"point": {"type": "geo_point", "null_value": {"lat": 28.41951122705923, "lon": -81.58118930548268}}}}})).unwrap();
    put(&node, "g", "a", &json!({"point": [null, {"lat": -54.48979082798907, "lon": -37.07576688239972}]}));
    let r = search(&node, "g", json!({"size": 0, "aggs": {"grid": {"geohash_grid": {"field": "point", "precision": 8}}}}));
    let b = r["aggregations"]["grid"]["buckets"].as_array().unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b[0]["key"], "djjcxtd7");
    assert_eq!(b[1]["key"], "5nmdh26m");
}

#[test]
fn inner_hits_select_sources_without_copying_parent_siblings() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("source-walk", &json!({"mappings": {"properties": {
        "payload": {"type": "keyword", "index": false, "doc_values": false},
        "children": {"type": "nested", "properties": {
            "number": {"type": "integer"}, "label": {"type": "keyword"}
        }}
    }}})).unwrap();
    let children: Vec<Value> = (0..32).map(|n| json!({"number": n, "label": format!("child-{n}")})).collect();
    put(&node, "source-walk", "parent", &json!({"payload": "x".repeat(65536), "children": children}));
    let result = search(&node, "source-walk", json!({"_source": false, "query": {
        "nested": {"path": "children", "query": {"range": {"children.number": {"gte": 3, "lt": 29}}},
                   "inner_hits": {"size": 100, "sort": ["_doc"]}}
    }}));
    let parent = &result["hits"]["hits"][0];
    assert!(parent.get("_source").is_none());
    let inner = &parent["inner_hits"]["children"]["hits"];
    assert_eq!(inner["total"]["value"], 26);
    for (i, hit) in inner["hits"].as_array().unwrap().iter().enumerate() {
        let offset = i + 3;
        assert_eq!(hit["_nested"], json!({"field": "children", "offset": offset}));
        assert_eq!(hit["_source"], children[offset]);
        assert!(hit["_source"].get("payload").is_none());
    }
}

/// Parent/child `inner_hits` (ES `ParentChildInnerHitContextBuilder`): a
/// `has_child` hit carries its matching children, a `has_parent` hit its
/// parent, named after the relation, with `_routing`.
#[test]
fn join_inner_hits_children_and_parent() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("qa", &json!({"mappings": {"properties": {"rel": {"type": "join", "relations": {"question": "answer"}}, "text": {"type": "text"}}}})).unwrap();
    let svc = node.get_index("qa").unwrap();
    let w = |id: &str, routing: &str, src: Value| {
        let r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: Some(routing.to_string()), cond: Default::default() }], false);
        r.into_iter().next().unwrap().unwrap();
    };
    w("q1", "q1", json!({"rel": "question", "text": "how to rust"}));
    w("q2", "q2", json!({"rel": "question", "text": "how to java"}));
    w("a1", "q1", json!({"rel": {"name": "answer", "parent": "q1"}, "text": "use cargo"}));
    w("a2", "q1", json!({"rel": {"name": "answer", "parent": "q1"}, "text": "read the book"}));
    w("a3", "q2", json!({"rel": {"name": "answer", "parent": "q2"}, "text": "use maven"}));
    let r = search(&node, "qa", json!({"sort": [{"_id": "asc"}], "query": {"has_child": {"type": "answer", "query": {"match": {"text": "use"}}, "inner_hits": {}}}}));
    let hits = r["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "{r}");
    let q1 = &hits[0]["inner_hits"]["answer"]["hits"];
    assert_eq!(hits[0]["_id"], "q1");
    assert_eq!(q1["total"], json!({"value": 1, "relation": "eq"}), "{r}");
    assert_eq!(q1["hits"][0]["_id"], "a1");
    assert_eq!(q1["hits"][0]["_routing"], "q1");
    assert_eq!(q1["hits"][0]["_source"]["text"], "use cargo");
    assert_eq!(hits[1]["inner_hits"]["answer"]["hits"]["hits"][0]["_id"], "a3", "{r}");
    // Every child, without `_source`, under a custom name.
    let r = search(&node, "qa", json!({"query": {"has_child": {"type": "answer", "query": {"match_all": {}}, "min_children": 2, "inner_hits": {"name": "all", "_source": false, "sort": [{"_id": "desc"}]}}}}));
    let ih = &r["hits"]["hits"][0]["inner_hits"]["all"]["hits"];
    assert_eq!(ih["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect::<Vec<_>>(), vec!["a2", "a1"], "{r}");
    assert!(ih["hits"][0].get("_source").is_none(), "{r}");
    // has_parent: each child hit carries its parent.
    let r = search(&node, "qa", json!({"sort": [{"_id": "asc"}], "query": {"has_parent": {"parent_type": "question", "query": {"match": {"text": "rust"}}, "inner_hits": {}}}}));
    let hits = r["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.iter().map(|h| h["_id"].as_str().unwrap()).collect::<Vec<_>>(), vec!["a1", "a2"], "{r}");
    for h in hits {
        assert_eq!(h["inner_hits"]["question"]["hits"]["hits"][0]["_id"], "q1", "{r}");
    }
}
