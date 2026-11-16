//! Grid aggregations' `bounds` over `geo_shape` (ES's bounded tilers,
//! `GeoTileBoundedPredicate` / `GeoHashBoundedPredicate`) and the `_mvt`
//! aggs layer, whose grid ES bounds to the tile (`RestVectorTileAction`
//! `setGeoBoundingBox`).

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::BTreeSet;

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn send(&self, m: reqwest::Method, p: &str, b: Value) -> (u16, Value) {
        let r = self.c.request(m, format!("{}{}", self.base, p)).header("content-type", "application/json").body(b.to_string()).send().await.unwrap();
        let st = r.status().as_u16();
        let t = r.text().await.unwrap();
        (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
    }
    async fn shapes(&self, index: &str, shapes: &[&str]) {
        let (st, v) = self.send(reqwest::Method::PUT, &format!("/{index}"), json!({"mappings": {"properties": {"s": {"type": "geo_shape"}}}})).await;
        assert_eq!(st, 200, "{v}");
        for (i, s) in shapes.iter().enumerate() {
            let (st, v) = self.send(reqwest::Method::PUT, &format!("/{index}/_doc/{i}?refresh=true"), json!({"s": s})).await;
            assert!(st == 200 || st == 201, "{v}");
        }
    }
    async fn buckets(&self, index: &str, agg: Value) -> Vec<(String, u64)> {
        let (st, v) = self.send(reqwest::Method::POST, &format!("/{index}/_search"), json!({"size": 0, "aggs": {"g": agg}})).await;
        assert_eq!(st, 200, "{v}");
        v["aggregations"]["g"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap())).collect()
    }
}

/// ES's bounded grids on a small fixture: a 60°×60° square around (0, 0).
/// Unbounded it touches the four zoom-3 tiles x 3..4, y 3..4. Bounded to
/// lon [0, 45] (45 is the left edge of tile x 5, which ES's predicate
/// excludes) and lat [-10, 10] only the x 4 column remains; geohash
/// likewise keeps only the cells strictly overlapping (`s`, `k`). Buckets
/// in ES's order: count, then the cell's long, descending.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_grids_match_es() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.shapes("sq", &["POLYGON ((-30 -30, 30 -30, 30 30, -30 30, -30 -30))"]).await;
    let b = json!({"top_left": {"lat": 10.0, "lon": 0.0}, "bottom_right": {"lat": -10.0, "lon": 45.0}});
    let tile = |bounds: Option<&Value>| {
        let mut g = json!({"geotile_grid": {"field": "s", "precision": 3}});
        if let Some(b) = bounds {
            g["geotile_grid"]["bounds"] = b.clone();
        }
        g
    };
    let one = |keys: &[&str]| keys.iter().map(|k| (k.to_string(), 1u64)).collect::<Vec<_>>();
    assert_eq!(es.buckets("sq", tile(None)).await, one(&["3/4/4", "3/4/3", "3/3/4", "3/3/3"]));
    assert_eq!(es.buckets("sq", tile(Some(&b))).await, one(&["3/4/4", "3/4/3"]));
    assert_eq!(es.buckets("sq", json!({"geohash_grid": {"field": "s", "precision": 1}})).await, one(&["s", "k", "e", "7"]));
    assert_eq!(es.buckets("sq", json!({"geohash_grid": {"field": "s", "precision": 1, "bounds": b}})).await, one(&["s", "k"]));
    // geohex: the bounded cells are the unbounded ones the bounds reach.
    let all: BTreeSet<String> = es.buckets("sq", json!({"geohex_grid": {"field": "s", "precision": 1}})).await.into_iter().map(|b| b.0).collect();
    let bounded: BTreeSet<String> = es.buckets("sq", json!({"geohex_grid": {"field": "s", "precision": 1, "bounds": b}})).await.into_iter().map(|b| b.0).collect();
    assert!(!bounded.is_empty() && bounded.is_subset(&all) && bounded.len() < all.len(), "{bounded:?} of {all:?}");
}

/// The aggs layer of a vector tile: (feature count, every string value,
/// i.e. the cells' `_key`s).
fn aggs_layer(b: &[u8]) -> (usize, BTreeSet<String>) {
    fn varint(b: &[u8], p: &mut usize) -> u64 {
        let (mut v, mut shift) = (0u64, 0);
        while *p < b.len() {
            let byte = b[*p];
            *p += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        v
    }
    fn fields(b: &[u8]) -> Vec<(u64, Vec<u8>)> {
        let (mut out, mut p) = (Vec::new(), 0);
        while p < b.len() {
            let key = varint(b, &mut p);
            match key & 7 {
                0 => {
                    varint(b, &mut p);
                }
                1 => p += 8,
                2 => {
                    let n = varint(b, &mut p) as usize;
                    out.push((key >> 3, b[p..p + n].to_vec()));
                    p += n;
                }
                5 => p += 4,
                _ => break,
            }
        }
        out
    }
    for (f, layer) in fields(b) {
        let parts = fields(&layer);
        if f != 3 || !parts.iter().any(|p| p.0 == 1 && p.1 == b"aggs") {
            continue;
        }
        let features = parts.iter().filter(|p| p.0 == 2).count();
        let strings = parts.iter().filter(|p| p.0 == 4).flat_map(|p| fields(&p.1)).filter(|v| v.0 == 1).map(|v| String::from_utf8_lossy(&v.1).to_string()).collect();
        return (features, strings);
    }
    (0, BTreeSet::new())
}

/// A high-zoom `_mvt` tile over a world-spanning shape: the grid is
/// bounded to the tile, so only its 128 × 128 cells (zoom 10 +
/// grid_precision 7) are tiled: every one of them comes back, quickly, and
/// the request breaker never sees the world's 2^34 cells.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn high_zoom_mvt_over_a_world_shape_tiles_only_the_tile() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.shapes("world", &["POLYGON ((-179 -84, 179 -84, 179 84, -179 84, -179 -84))"]).await;
    let (z, x, y) = (10u32, 512u64, 340u64);
    let started = std::time::Instant::now();
    let r = es.c.post(format!("{}/world/_mvt/s/{z}/{x}/{y}", es.base)).json(&json!({"grid_precision": 7, "size": 0})).send().await.unwrap();
    let st = r.status().as_u16();
    let body = r.bytes().await.unwrap();
    assert_eq!(st, 200, "{}", String::from_utf8_lossy(&body));
    assert!(started.elapsed() < std::time::Duration::from_secs(30), "{:?}", started.elapsed());
    let (features, strings) = aggs_layer(&body);
    let want: BTreeSet<String> = (0..128u64).flat_map(|i| (0..128u64).map(move |j| format!("17/{}/{}", x * 128 + i, y * 128 + j))).collect();
    assert_eq!(features, want.len());
    let keys: BTreeSet<String> = strings.into_iter().filter(|s| s.starts_with("17/")).collect();
    assert_eq!(keys, want);
    let v: Value = es.c.get(format!("{}/_nodes/stats/breaker", es.base)).send().await.unwrap().json().await.unwrap();
    let b = v["nodes"].as_object().unwrap().values().next().unwrap()["breakers"]["request"].clone();
    assert_eq!(b["tripped"], 0, "{b}");
}
