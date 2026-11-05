//! One geo microbench for every Rally-shaped geo task (`geopoint`,
//! `geopointshape`-like points and `geoshape`), with result checks:
//!
//! `cargo test -p elasticsearch-engine --release geo_suite_microbench -- --ignored --nocapture`
//!
//! Sizes: `GEO_SUITE_POINTS` (default 5,000,000 points, the core-tree
//! geopoint rows), `GEO_SUITE_POINT_DOCS` (default 2,000,000 geo_point
//! docs, the geopoint search rows), `GEO_SUITE_POINT_SHAPES` (default
//! 2,000,000 points indexed as geo_shape: point-only segments, the
//! geopointshape rows), `GEO_SUITE_SHAPES` (default 2,000,000 OSM-like
//! shapes: polygons, lines, multilines and a tenth points, on
//! triangle-indexed segments), `GEO_SUITE_RUNS` (timed runs per task after
//! one warm-up, best shown; default 3). Each indexed section ends with its
//! ingest rate, a force merge to one segment and the merged store per doc
//! by section. `GEO_SUITE_MERGE_FIRST` force-merges the geopointshape
//! and geoshape indexes before their queries (Rally's layout: one large
//! segment; a number, that many segments, e.g. 3 as Rally's geoshape).
//!
//! It prints one table: task, wall ms, result, the check, and the ES 9.5.5
//! full-corpus p50 for reference (60.8M points / 23.7M shapes there, so
//! only the order of magnitude compares). Every result is checked against a
//! brute-force definition (or, where noted, another path); the test fails
//! after printing the table if any check failed, so "passes" means both
//! timings and correct results.

use crate::index::WriteReq;
use crate::node::Node;
use rustic_core::fieldtypes::shape;
use rustic_core::geo::{self as geo, decode_lat, decode_lon, encode_lat, encode_lon, GeoIndex, GeoPointEntry, GeoPred, GeoShape, GridPred};
use rustic_core::query::{geo_pred_scorer, geo_scorer::with_wanted, DocSet, TERMINATED};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

struct Row {
    task: String,
    ms: f64,
    result: String,
    check: String,
    ok: bool,
    es: Option<f64>,
}

#[derive(Default)]
struct Table {
    rows: Vec<Row>,
    /// `GEO_SUITE_ONLY` tokens (row-name prefixes or section names; empty:
    /// everything).
    only: Vec<String>,
    /// Every row of the section running now is wanted (no filter, or the
    /// section was named).
    whole_section: bool,
}

/// The suite's sections and the row-name stems each produces (to run a
/// section only when some `GEO_SUITE_ONLY` token can match one of its rows).
/// `GEO_SUITE_MERGE_FIRST`: the segments to force-merge to (set without a
/// number: one).
fn merge_first() -> Option<usize> {
    let v = std::env::var("GEO_SUITE_MERGE_FIRST").ok()?;
    Some(v.trim().parse().ok().filter(|k| *k > 0).unwrap_or(1))
}

const SECTIONS: &[(&str, &[&str])] = &[
    ("points", &["geopoint geoGrid_geohex", "geopoint geohex", "geopoint geoGrid_geotile", "geopoint geoGrid_geohash ", "geopoint distanceFilter", "geopoint ring", "geopoint cap", "geopoint polygon", "geopoint distanceRange", "geopoint centroid", "geopoint distance top", "geopoint doc-order"]),
    ("point_grid", &["grid "]),
    ("ingest_rally", &["rally ingest"]),
    ("point_search", &["geopoint geoGrid_aggs", "geopoint distanceFilter10kGTE", "ingest geopoint", "force-merge geopoint", "store geopoint"]),
    ("point_shape", &["geopointshape", "ingest geopointshape", "force-merge geopointshape", "store geopointshape"]),
    ("point_rally", &["geopoint rally"]),
    ("ingest", &["ingest polygons", "ingest linestrings", "ingest geoshape polygons", "ingest geoshape linestrings", "force-merge geoshape polygons", "force-merge geoshape linestrings", "store geoshape polygons", "store geoshape linestrings"]),
    ("shapes", &["polygon-", "bbox-", "geoGrid_", "centroid", "bounds", "st-", "mvt-", "ingest geoshape", "force-merge geoshape", "store geoshape"]),
];

impl Table {
    /// Whether a row (or group of rows) named with prefix `name` is wanted.
    fn want(&self, name: &str) -> bool {
        let is_section = |t: &str| SECTIONS.iter().any(|(s, _)| *s == t);
        self.whole_section || self.only.iter().filter(|t| !is_section(t)).any(|t| name.starts_with(t.as_str()) || t.starts_with(name))
    }

    /// Whether section `section` has to run, and mark it running.
    fn enter(&mut self, section: &str) -> bool {
        self.whole_section = self.only.is_empty() || self.only.iter().any(|t| t == section);
        if self.whole_section {
            return true;
        }
        let stems = SECTIONS.iter().find(|(s, _)| *s == section).map_or(&[][..], |(_, r)| *r);
        // A token naming a section selects that section only.
        let is_section = |t: &str| SECTIONS.iter().any(|(s, _)| *s == t);
        self.only.iter().filter(|t| !is_section(t)).any(|t| stems.iter().any(|s| s.starts_with(t.as_str()) || t.starts_with(s)))
    }

    fn add(&mut self, task: &str, ms: f64, result: impl Into<String>, ok: bool, check: impl Into<String>, es: Option<f64>) {
        if !self.want(task) {
            return;
        }
        let row = Row { task: task.to_string(), ms, result: result.into(), check: check.into(), ok, es };
        eprintln!("  {:<44} {:>10.2} ms  {:<28} {}{}", row.task, row.ms, row.result, if row.ok { "ok " } else { "FAIL " }, row.check);
        self.rows.push(row);
    }

    fn print_and_assert(&self) {
        eprintln!("\n{:<44} {:>11} {:>12}  {:<30} check", "task", "wall ms", "ES p50 ms", "result");
        for r in &self.rows {
            let es = r.es.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into());
            eprintln!("{:<44} {:>11.2} {:>12}  {:<30} {}{}", r.task, r.ms, es, r.result, if r.ok { "ok " } else { "FAIL " }, r.check);
        }
        let failed: Vec<&str> = self.rows.iter().filter(|r| !r.ok).map(|r| r.task.as_str()).collect();
        assert!(failed.is_empty(), "result checks failed: {failed:?}");
    }
}

/// `f` on its own thread, abandoned after `cap` (`None`): a row that runs
/// away fails instead of hanging the suite (the thread runs on, unseen).
fn capped<T: Send + 'static>(cap: std::time::Duration, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(cap).ok()
}

/// The per-call cap (`GEO_SUITE_ROW_CAP_SECS`, default 300).
fn row_cap() -> std::time::Duration {
    std::time::Duration::from_secs(env("GEO_SUITE_ROW_CAP_SECS", 300) as u64)
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Best of `runs` timed calls after one warm-up: the last result and ms.
/// A warm-up over 10 s is the only run (a slow row is not run four times).
fn best<T>(runs: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let t = Instant::now();
    let mut out = f();
    let warm = t.elapsed().as_secs_f64() * 1e3;
    if warm > 10_000.0 {
        return (out, warm);
    }
    let mut ms = f64::INFINITY;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        out = f();
        ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (out, ms)
}

#[test]
#[ignore]
fn geo_suite_microbench() {
    let runs = env("GEO_SUITE_RUNS", 3);
    // Segments as a current cluster writes them (this build's format: the
    // compact geo index and shape doc values among others).
    rustic_core::codec::apply_cluster_segment_format(rustic_core::codec::VERSION);
    let mut t = Table::default();
    // `GEO_SUITE_ONLY`: comma-separated row-name prefixes or section names
    // (points, point_search, point_shape, ingest, shapes): only those rows
    // run, and only the sections holding them are set up.
    t.only = std::env::var("GEO_SUITE_ONLY").map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    // `GEO_SUITE_SECTIONS`: a comma list of points, point_search,
    // point_shape, shapes (default: all), e.g. to run geopointshape alone
    // at 20M points (`GEO_SUITE_POINT_SHAPES=20000000`).
    let sections = std::env::var("GEO_SUITE_SECTIONS").unwrap_or_else(|_| "points,point_search,point_rally,point_shape,ingest,shapes".into());
    let on = |s: &str| sections.split(',').any(|x| x.trim() == s);
    if on("points") && t.enter("points") {
        points_section(env("GEO_SUITE_POINTS", 5_000_000) as u32, runs, &mut t);
    }
    if on("point_search") && t.enter("point_search") {
        point_search_section(env("GEO_SUITE_POINT_DOCS", 2_000_000), runs, &mut t);
    }
    if on("ingest_rally") && t.enter("ingest_rally") {
        ingest_rally_section(env("GEO_INGEST_DOCS", 1_000_000), &mut t);
    }
    if on("point_grid") && t.enter("point_grid") {
        point_grid_section(env("GEO_GRID_POINTS", 20_000_000), runs, &mut t);
    }
    if on("point_rally") && t.enter("point_rally") {
        point_rally_section(env("GEO_SUITE_RALLY_POINTS", 20_000_000), runs, &mut t);
    }
    if on("point_shape") && t.enter("point_shape") {
        point_shape_section(env("GEO_SUITE_POINT_SHAPES", 2_000_000), runs, &mut t);
    }
    if on("ingest") && t.enter("ingest") {
        ingest_section(env("GEO_SUITE_INGEST", 200_000), &mut t);
    }
    if on("shapes") && t.enter("shapes") {
        shapes_section(env("GEO_SUITE_SHAPES", 2_000_000), runs, &mut t);
    }
    t.print_and_assert();
}

// ---------------------------------------------------------------------------
// geo_point (Rally `geopoint`, through the core tree on the compact layout).

fn drain_first(s: &mut dyn DocSet, k: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(k);
    while v.len() < k {
        let d = s.advance();
        if d == TERMINATED {
            break;
        }
        v.push(d);
    }
    v
}

fn points_section(n: u32, runs: usize, t: &mut Table) {
    eprintln!("geo_point: {n} points (a third in western Europe, the rest worldwide)");
    let mut r = Lcg(5);
    let pts: Vec<GeoPointEntry> = (0..n)
        .map(|doc| {
            let (lat, lon) = if doc % 3 == 0 { (r.range(40.0, 60.0), r.range(-5.0, 25.0)) } else { (r.range(-80.0, 80.0), r.range(-180.0, 180.0)) };
            GeoPointEntry { doc, lat: encode_lat(lat), lon: encode_lon(lon) }
        })
        .collect();
    // The segment layout (compact).
    let g = {
        let built = GeoIndex::build(pts);
        let mut b = Vec::new();
        built.encode_compact(&mut b);
        GeoIndex::open_compact(&rustic_core::codec::directory::FileSlice::from_vec(b)).unwrap()
    };
    let decoded: Vec<(u32, f64, f64)> = g.values().iter().map(|p| ({ p.doc }, decode_lat({ p.lat }), decode_lon({ p.lon }))).collect();
    // Brute force: the docs (ascending) passing each test.
    let brute = |f: &dyn Fn(f64, f64) -> bool| -> Vec<u32> { decoded.iter().filter(|(_, a, b)| f(*a, *b)).map(|(d, _, _)| *d).collect() };

    // geoGrid_geohex: cell 811fbffffffffff, the query's grid predicate.
    use crate::aggs::geo_hex::{parse_cell, point_cell_id, HexCellPred, H3_TESTS};
    use std::sync::atomic::Ordering::Relaxed;
    let cell = parse_cell("811fbffffffffff").unwrap();
    let hex = || {
        let p = HexCellPred::new(cell);
        let b = p.plain_bbox().unwrap();
        GeoPred::Leaf(Arc::new(GridPred::new(p, b)))
    };
    let want = brute(&|a, b| point_cell_id(a, b, 1) == Some(u64::from(cell)));
    let ((first, h3_lazy), ms) = best(runs, || {
        H3_TESTS.store(0, Relaxed);
        let mut s = with_wanted(10_001, || geo_pred_scorer(&g, hex(), true, n, 1.0).unwrap());
        (drain_first(&mut *s, 10_001), H3_TESTS.load(Relaxed))
    });
    t.add("geopoint geoGrid_geohex first 10001", ms, format!("{} docs, {h3_lazy} H3", first.len()), first[..] == want[..want.len().min(10_001)], "first docs = brute", Some(24.5));
    let ((count, h3), ms) = best(runs, || {
        H3_TESTS.store(0, Relaxed);
        (g.pred_count(&hex()).unwrap(), H3_TESTS.load(Relaxed))
    });
    t.add("geopoint geoGrid_geohex count", ms, format!("{count}, {h3} H3"), count == want.len() as u64, "= brute H3", None);
    let (walk, ms) = best(runs, || g.any_point_hits(&hex(), n).into_sorted());
    t.add("geopoint geoGrid_geohex walk", ms, format!("{}", walk.len()), walk == want, "= brute", None);
    let (est, ms) = best(runs, || g.pred_estimate(&hex()));
    t.add("geopoint geohex KD estimate", ms, format!("{est}"), est >= want.len() as u64, ">= matches", None);

    // geoGrid_geotile 6/32/22 and geohash u0: plain boxes.
    for (task, (top, left, bottom, right), es) in [
        ("geopoint geoGrid_geotile", crate::fieldtypes::geo::tile_bbox(6, 32, 22), Some(4.15)),
        ("geopoint geoGrid_geohash", geo::geohash_bbox("u0").unwrap(), None),
    ] {
        let shape = GeoShape::BoundingBox { top, left, bottom, right };
        let want = brute(&|a, b| shape.contains(a, b));
        let pred = || GeoPred::Leaf(Arc::new(shape.clone()));
        let (first, ms) = best(runs, || {
            let mut s = with_wanted(10_001, || geo_pred_scorer(&g, pred(), true, n, 1.0).unwrap());
            drain_first(&mut *s, 10_001)
        });
        t.add(&format!("{task} first 10001"), ms, format!("{} docs", first.len()), first[..] == want[..want.len().min(10_001)], "first docs = brute", es);
        let (c, ms) = best(runs, || g.pred_count(&pred()).unwrap());
        t.add(&format!("{task} count"), ms, format!("{c}"), c == want.len() as u64, "= brute", None);
    }

    // distanceFilter (200 km around 55N 7E), the Rally polygon.
    let dist = |m: f64| GeoPred::Leaf(Arc::new(GridPred::new(geo::DistancePred::new(55.0, 7.0, m), GeoShape::Distance { lat: 55.0, lon: 7.0, meters: m }.plain_bbox().unwrap())));
    let dshape = GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 200_000.0 };
    let want = brute(&|a, b| dshape.contains(a, b));
    let (c, ms) = best(runs, || g.pred_count(&dist(200_000.0)).unwrap());
    t.add("geopoint distanceFilterCount200", ms, format!("{c}"), c == want.len() as u64, "= brute", None);
    let (first, ms) = best(runs, || {
        let mut s = with_wanted(10_001, || geo_pred_scorer(&g, dist(200_000.0), true, n, 1.0).unwrap());
        drain_first(&mut *s, 10_001)
    });
    t.add("geopoint distanceFilter first 10001", ms, format!("{} docs", first.len()), first[..] == want[..want.len().min(10_001)], "first docs = brute", None);
    // Shapes as GeoShape predicates (geo_shape circles on geo_point, the
    // lazy-scan cases): a ring and a cap around the south pole.
    let leaf = |sh: GeoShape| GeoPred::Leaf(Arc::new(sh));
    let d200 = GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 200_000.0 };
    let d100 = GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 100_000.0 };
    let pole = GeoShape::Distance { lat: -90.0, lon: 0.0, meters: 300_000.0 };
    for (task, pred, test) in [
        ("geopoint ring 200 km and not 100 km", GeoPred::And(vec![leaf(d200.clone()), GeoPred::Not(Box::new(leaf(d100.clone())))]), Box::new(|a: f64, b: f64| d200.contains(a, b) && !d100.contains(a, b)) as Box<dyn Fn(f64, f64) -> bool>),
        ("geopoint cap 300 km at the south pole", leaf(pole.clone()), Box::new(|a: f64, b: f64| pole.contains(a, b)) as Box<dyn Fn(f64, f64) -> bool>),
    ] {
        let want = brute(&*test);
        let (first, ms) = best(runs, || {
            let mut s = with_wanted(10_001, || geo_pred_scorer(&g, pred.clone(), true, n, 1.0).unwrap());
            drain_first(&mut *s, 10_001)
        });
        t.add(&format!("{task} first 10001"), ms, format!("{} docs", first.len()), first[..] == want[..want.len().min(10_001)], "first docs = brute", None);
        let (c, ms) = best(runs, || g.pred_count(&pred).unwrap());
        t.add(&format!("{task} count"), ms, format!("{c}"), c == want.len() as u64, "= brute", None);
    }
    let poly = GeoShape::Polygon { points: vec![(49.0, -0.1), (48.0, 5.0), (49.0, 15.0), (60.0, 14.0), (61.0, -0.1), (49.0, -0.1)] };
    let ppred = || GeoPred::Leaf(Arc::new(GridPred::new(poly.clone(), poly.plain_bbox().unwrap())));
    let pwant = brute(&|a, b| poly.contains(a, b));
    let (first, ms) = best(runs, || {
        let mut s = with_wanted(10_001, || geo_pred_scorer(&g, ppred(), true, n, 1.0).unwrap());
        drain_first(&mut *s, 10_001)
    });
    t.add("geopoint polygon first 10001", ms, format!("{} docs", first.len()), first[..] == pwant[..pwant.len().min(10_001)], "first docs = brute", None);

    // distanceRange: four rings from the tree.
    let def = json!({"geo_distance": {"field": "location", "origin": "55.0, 7.0", "unit": "km",
        "ranges": [{"from": 0, "to": 100}, {"from": 100, "to": 200}, {"from": 200, "to": 300}, {"from": 300, "to": 400}]}});
    let dc = crate::aggs::DistanceCounts::parse(def.as_object().unwrap()).unwrap();
    let mut scan = [0u64; 4];
    for (_, a, b) in &decoded {
        let d = geo::haversine_meters(55.0, 7.0, *a, *b) / 1000.0;
        for (i, (lo, hi)) in [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 400.0)].iter().enumerate() {
            scan[i] += u64::from(d >= *lo && d < *hi);
        }
    }
    let (rings, ms) = best(runs, || (0..dc.buckets()).map(|i| g.pred_count(&GeoPred::Leaf(Arc::new(dc.ring(i)))).unwrap()).collect::<Vec<u64>>());
    t.add("geopoint distanceRange (rings)", ms, format!("{rings:?}"), rings[..] == scan[..], "= per-point scan", Some(1067.0));

    // polygon-centroid / centroid: tree sums against one sequential sum.
    let seq = |docs: Option<&[u32]>| {
        let (mut k, mut la, mut lo) = (0u64, crate::esql::partial::Kahan::default(), crate::esql::partial::Kahan::default());
        let mut it = docs.map(|d| d.iter().peekable());
        for (d, a, b) in &decoded {
            if let Some(it) = it.as_mut() {
                while it.peek().is_some_and(|x| **x < *d) {
                    it.next();
                }
                if it.peek() != Some(&d) {
                    continue;
                }
            }
            k += 1;
            la.add(*a, 0.0);
            lo.add(*b, 0.0);
        }
        (k, la.value / k.max(1) as f64, lo.value / k.max(1) as f64)
    };
    for (task, filtered, es) in [("geopoint polygon-centroid", true, Some(65.0)), ("geopoint centroid", false, Some(273.0))] {
        let ((k, la, lo), ms) = best(runs, || {
            let hits = filtered.then(|| g.pred_hits(&ppred(), n).unwrap());
            crate::search::tree_centroid_sums(&g, hits)
        });
        let (sk, sla, slo) = seq(filtered.then_some(&pwant[..]));
        let ok = k == sk && (la.value / k as f64 - sla).abs() < 1e-9 && (lo.value / k as f64 - slo).abs() < 1e-9;
        t.add(task, ms, format!("{k} pts ({:.5}, {:.5})", la.value / k as f64, lo.value / k as f64), ok, "= sequential sum", es);
    }

    // Core reads (the layout microbench's): distance-ordered top 10, the
    // doc-order scan, per-doc value counts.
    let (seen, ms) = best(runs, || {
        let mut total = 0usize;
        for (lat, lon) in [(48.0, 2.0), (0.0, 0.0), (-33.0, 151.0), (40.7, -74.0)] {
            let mut s = 0usize;
            g.visit_by_distance(lat, lon, |_, p| {
                s += p.len();
                s < 10
            });
            total += s;
        }
        total
    });
    t.add("geopoint distance top-10 visits (x4)", ms, format!("{seen} pts"), seen >= 40, ">= 40", None);
    let (sum, ms) = best(runs, || g.values().iter().map(|p| (p.lat as i64 + p.lon as i64) as u64).fold(0u64, u64::wrapping_add));
    let want_sum = decoded.iter().zip(g.values().iter()).map(|(_, p)| (p.lat as i64 + p.lon as i64) as u64).fold(0u64, u64::wrapping_add);
    t.add("geopoint doc-order scan", ms, format!("{} pts", decoded.len()), sum == want_sum, "checksum", None);
}

// ---------------------------------------------------------------------------
// geo_shape (Rally `geoshape`, through the search API and ES|QL).

fn fmt_pts(p: &[(f64, f64)]) -> String {
    p.iter().map(|(x, y)| format!("{x:.7} {y:.7}")).collect::<Vec<_>>().join(", ")
}

fn ring(rng: &mut Lcg, cx: f64, cy: f64, r: f64, n: usize) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            let rr = r * (0.75 + 0.25 * rng.next());
            ((cx + rr * a.cos() * 1.6).clamp(-179.9, 179.9), (cy + rr * a.sin()).clamp(-84.0, 84.0))
        })
        .collect();
    v.push(v[0]);
    v
}

fn line(rng: &mut Lcg, mut x: f64, mut y: f64, step: f64, n: usize) -> Vec<(f64, f64)> {
    let mut a = rng.range(0.0, std::f64::consts::TAU);
    (0..n)
        .map(|_| {
            let p = (x, y);
            a += rng.range(-0.6, 0.6);
            x = (x + step * a.cos() * 1.6).clamp(-179.9, 179.9);
            y = (y + step * a.sin()).clamp(-84.0, 84.0);
            p
        })
        .collect()
}

/// One OSM-like shape: half around London (the mvt box, padded), a
/// quarter around Paris (the geoGrid_aggs box) and Øresund, the rest over
/// Europe and the world. Buildings (5-30 vertices), roads (2-60),
/// multilinestrings, a tenth points, districts (hundreds of vertices),
/// long lines and, one in a thousand, large areas of thousands of
/// vertices.
fn shape_wkt(rng: &mut Lcg, i: usize) -> String {
    let (cx, cy) = match i % 8 {
        0..=3 => (rng.range(-0.9, 0.2), rng.range(51.05, 51.75)),
        4 => (rng.range(2.1, 2.5), rng.range(48.7, 49.0)),
        5 => (rng.range(12.0, 13.6), rng.range(55.3, 56.0)),
        6 => (rng.range(-10.0, 25.0), rng.range(36.0, 62.0)),
        _ => (rng.range(-170.0, 170.0), rng.range(-60.0, 70.0)),
    };
    match i % 100 {
        0..=44 => {
            let n = 4 + (rng.next() * rng.next() * 26.0) as usize;
            { let r = rng.range(0.00005, 0.0004); format!("POLYGON (({}))", fmt_pts(&ring(rng, cx, cy, r, n))) }
        }
        45..=69 => {
            let n = 2 + (rng.next() * rng.next() * 58.0) as usize;
            { let r = rng.range(0.0002, 0.002); format!("LINESTRING ({})", fmt_pts(&line(rng, cx, cy, r, n))) }
        }
        70..=79 => format!("POINT ({cx:.7} {cy:.7})"),
        80..=89 => {
            let k = 2 + (rng.next() * 4.0) as usize;
            let parts: Vec<String> = (0..k)
                .map(|_| {
                    let n = 2 + (rng.next() * 40.0) as usize;
                    let (x, y) = (cx + rng.range(-0.01, 0.01), cy + rng.range(-0.01, 0.01));
                    { let r = rng.range(0.0002, 0.002); format!("({})", fmt_pts(&line(rng, x, y, r, n))) }
                })
                .collect();
            format!("MULTILINESTRING ({})", parts.join(", "))
        }
        90..=97 => {
            let n = 100 + (rng.next() * 900.0) as usize;
            { let r = rng.range(0.005, 0.05); format!("POLYGON (({}))", fmt_pts(&ring(rng, cx, cy, r, n))) }
        }
        _ => {
            if i % 1000 == 99 {
                let n = 2000 + (rng.next() * 6000.0) as usize;
                { let r = rng.range(0.3, 1.5); format!("POLYGON (({}))", fmt_pts(&ring(rng, cx, cy, r, n))) }
            } else {
                let n = 200 + (rng.next() * 1500.0) as usize;
                { let r = rng.range(0.001, 0.01); format!("LINESTRING ({})", fmt_pts(&line(rng, cx, cy, r, n))) }
            }
        }
    }
}

/// A doc's triangles and shape doc value, as indexed.
fn doc_shape(wkt: &str) -> (Vec<[i32; 7]>, Vec<u8>) {
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    let g = shape::parse(&json!(wkt), &opts).unwrap();
    let mut tris = Vec::new();
    rustic_core::fieldtypes::tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
    let mut c = shape::CentroidCalc::default();
    c.add(&g);
    let dv = rustic_core::fieldtypes::shape_field::doc_value(&tris, &c).unwrap();
    (tris, dv)
}

/// The `LatLonShape` query ES builds for `q` (normalized, quantized).
fn ref_query(q: &Value, rel: shape::SpatialRelation) -> rustic_core::fieldtypes::geo2d::ShapeQuery {
    use rustic_core::fieldtypes::geo2d::{query_geometries, ShapeQuery};
    let geoms = query_geometries(&shape::normalize_geo(&shape::parse_query_shape(q).unwrap()), rel == shape::SpatialRelation::Within).unwrap();
    ShapeQuery::new(&geoms, crate::fieldtypes::geo_shape_query::query_relation(rel))
}

/// One doc's relation to `q`, triangle by triangle (the trees' leaf tests).
fn ref_matches(q: &rustic_core::fieldtypes::geo2d::ShapeQuery, tris: &[[i32; 7]]) -> bool {
    use rustic_core::fieldtypes::geo2d::{ComponentLeaf, LeafQuery, RectLeaf, ShapeQuery};
    let leaf: Box<dyn LeafQuery + '_> = match q {
        ShapeQuery::None => return false,
        ShapeQuery::And(qs) => return qs.iter().all(|q| ref_matches(q, tris)),
        ShapeQuery::Component(c, r) => Box::new(ComponentLeaf { component: c, rel: *r }),
        ShapeQuery::Rect(e, _, r) => Box::new(RectLeaf { rect: e, rel: *r }),
    };
    let mut m = rustic_core::fieldtypes::shape_bkd::Matcher::new(leaf.as_ref(), 1);
    m.doc(0, tris);
    m.finish().unwrap()[0] & 1 == 1
}

/// Brute-force membership of one reference query.
struct Want {
    name: &'static str,
    query: rustic_core::fieldtypes::geo2d::ShapeQuery,
    bits: Vec<u64>,
    count: u64,
}

impl Want {
    fn has(&self, d: usize) -> bool {
        self.bits.get(d / 64).is_some_and(|w| w >> (d % 64) & 1 == 1)
    }
}

fn shapes_section(n: usize, runs: usize, t: &mut Table) {
    use shape::SpatialRelation as R;
    let rally_poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
    let oresund = json!({"type": "Polygon", "coordinates": [[[12.5706, 55.45074], [12.04631, 55.70469], [12.06618, 55.7991], [12.68098, 55.93474], [13.48894, 55.88091], [13.60262, 55.68478], [13.44479, 55.32661], [12.64235, 55.32723], [12.5706, 55.45074]]]});
    let bbox = json!({"type": "envelope", "coordinates": [[-0.1, 61.0], [15.0, 48.0]]});
    let london = json!({"type": "envelope", "coordinates": [[-0.703129, 51.618014], [0, 51.179343]]});
    let paris = json!({"type": "envelope", "coordinates": [[2.20, 48.95], [2.40, 48.75]]});
    let env = |(top, left, bottom, right): (f64, f64, f64, f64)| json!({"type": "envelope", "coordinates": [[left, top], [right, bottom]]});
    let hex_polys: Vec<Value> = crate::aggs::geo_hex::cell_polys(crate::aggs::geo_hex::parse_cell("811fbffffffffff").unwrap()).into_iter().map(|r| json!([r.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()])).collect();
    let words = n.div_ceil(64);
    let mut wants: Vec<Want> = [
        ("poly intersects", &rally_poly, R::Intersects),
        ("poly contains", &rally_poly, R::Contains),
        ("poly within", &rally_poly, R::Within),
        ("oresund intersects", &oresund, R::Intersects),
        ("bbox intersects", &bbox, R::Intersects),
        ("bbox contains", &bbox, R::Contains),
        ("bbox within", &bbox, R::Within),
        ("london intersects", &london, R::Intersects),
        ("paris intersects", &paris, R::Intersects),
    ]
    .into_iter()
    .map(|(name, q, rel)| Want { name, query: ref_query(q, rel), bits: vec![0; words], count: 0 })
    .collect();
    for (name, q) in [("geohash u0", env(geo::geohash_bbox("u0").unwrap())), ("geotile 6/32/22", env(crate::fieldtypes::geo::tile_bbox(6, 32, 22))), ("geohex 811fb", json!({"type": "multipolygon", "coordinates": hex_polys}))] {
        wants.push(Want { name, query: ref_query(&q, R::Intersects), bits: vec![0; words], count: 0 });
    }
    // The shape doc values of the docs in the Paris box (the grid aggs'
    // brute-force cells) and, with `GEO_SUITE_EXACT_MVT`, the London box.
    let exact_mvt = std::env::var("GEO_SUITE_EXACT_MVT").is_ok();
    let mut paris_dv: Vec<Vec<u8>> = Vec::new();
    let mut london_dv: Vec<Vec<u8>> = Vec::new();

    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("osm", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"shape": {"type": "geo_shape"}, "size": {"type": "long"}}}})).unwrap();
    let svc = node.get_index("osm").unwrap();
    // Rally starts a trial licence (ST_CENTROID_AGG / ST_EXTENT_AGG need it).
    let _ = crate::license::service(&node).start_trial(&node, "trial", true);
    let start = Instant::now();
    let mut rng = Lcg(2024);
    let mut batch = Vec::with_capacity(20_000);
    let flush_at = [n / 3, 2 * n / 3];
    let mut write_ms = 0.0;
    for i in 0..n {
        let w = shape_wkt(&mut rng, i);
        let (tris, dv) = doc_shape(&w);
        for want in &mut wants {
            if ref_matches(&want.query, &tris) {
                want.bits[i / 64] |= 1 << (i % 64);
                want.count += 1;
                if want.name == "paris intersects" {
                    paris_dv.push(dv.clone());
                }
                if exact_mvt && want.name == "london intersects" {
                    london_dv.push(dv.clone());
                }
            }
        }
        let size = w.len() as u64;
        batch.push(WriteReq::Index { id: Some(i.to_string()), source: json!({"shape": w, "size": size}).to_string(), routing: None, cond: Default::default() });
        if batch.len() == 20_000 || i + 1 == n || flush_at.contains(&i) {
            let tw = Instant::now();
            for r in svc.write(std::mem::take(&mut batch), false) {
                r.unwrap();
            }
            if flush_at.contains(&i) {
                svc.flush().unwrap();
            }
            write_ms += tw.elapsed().as_secs_f64() * 1e3;
        }
    }
    let tw = Instant::now();
    svc.flush().unwrap();
    svc.refresh().unwrap();
    write_ms += tw.elapsed().as_secs_f64() * 1e3;
    if let Some(k) = merge_first() {
        let t0 = Instant::now();
        svc.force_merge(k, false).and_then(|_| svc.refresh()).unwrap();
        eprintln!("geo_shape: merged to {} segment(s) in {:.1} s", svc.local_shards()[0].shard.reader_view().segments().len(), t0.elapsed().as_secs_f64());
    }
    eprintln!("geo_shape: indexed {n} shapes (with the brute-force relations) in {:.1} s", start.elapsed().as_secs_f64());
    let w = |name: &str| wants.iter().find(|x| x.name == name).unwrap();
    let all = n as u64;

    // Every call capped (`GEO_SUITE_ROW_CAP_SECS`): a runaway row fails.
    let search = |body: &Value| -> Value {
        let (svc, b) = (svc.clone(), body.clone());
        capped(row_cap(), move || suite_search(&svc, &b)).unwrap_or_else(|| {
            eprintln!("search over the row cap: {body}");
            Value::Null
        })
    };
    let esql_try = |q: &str| {
        let (node, q) = (node.clone(), q.to_string());
        capped(row_cap(), move || crate::esql::run(&node, &q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }))
            .unwrap_or_else(|| Err(crate::error::EsError::internal("over the row cap")))
    };
    // ES|QL failures become an empty result whose warning names the
    // error: its row's check fails, the suite goes on.
    let esql = |q: &str| esql_try(q).unwrap_or_else(|e| {
        eprintln!("ES|QL error: {q}: {e:?}");
        crate::esql::EsqlResult { table: crate::esql::Table { cols: Vec::new(), rows: Vec::new() }, warnings: vec![format!("ERROR {e:?}")], features: Default::default(), functions: Default::default(), took_ms: 0, documents_found: 0, values_loaded: 0, rows_emitted: 0, settings: Default::default(), col_meta: Default::default(), all_columns: None }
    });
    let col = |r: &crate::esql::EsqlResult, name: &str| r.table.cols.iter().position(|c| c.name == name);
    let long = |v: &crate::esql::Val| match v {
        crate::esql::Val::Long(x) | crate::esql::Val::Int(x) => *x as u64,
        _ => u64::MAX,
    };
    let cell = |r: &crate::esql::EsqlResult, name: &str| -> Option<crate::esql::Val> { r.table.rows.first()?.get(col(r, name)?).cloned() };
    let ids = |r: &Value| -> Vec<usize> { r["hits"]["hits"].as_array().map(|h| h.iter().map(|x| x["_id"].as_str().unwrap().parse().unwrap()).collect()).unwrap_or_default() };
    let total = |r: &Value| r["hits"]["total"]["value"].as_u64().unwrap_or(0);

    // Relations: size-10 hits, the -count variants (DSL value_count of
    // `size`, ES|QL COUNT(size)), the ES|QL LIMIT 10 forms.
    let geo_q = |q: &Value, rel: &str| json!({"geo_shape": {"shape": {"shape": q, "relation": rel}}});
    let wkt_poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
    let wkt_ore = "POLYGON((12.5706 55.45074, 12.04631 55.70469, 12.06618 55.7991, 12.68098 55.93474, 13.48894 55.88091, 13.60262 55.68478, 13.44479 55.32661, 12.64235 55.32723, 12.5706 55.45074))";
    struct Rel<'a> {
        task: &'a str,
        q: Value,
        rel: &'a str,
        want: &'a str,
        negate: bool,
        esql_fn: &'a str,
        wkt: &'a str,
        es: [Option<f64>; 4],
    }
    let rels = [
        Rel { task: "polygon-intersects", q: rally_poly.clone(), rel: "intersects", want: "poly intersects", negate: false, esql_fn: "ST_Intersects", wkt: wkt_poly, es: [Some(3.19), Some(13.6), Some(10.1), Some(7.8)] },
        Rel { task: "polygon-contains", q: rally_poly.clone(), rel: "contains", want: "poly contains", negate: false, esql_fn: "ST_Contains", wkt: wkt_poly, es: [Some(1.0), Some(1.1), None, None] },
        Rel { task: "polygon-disjoint", q: rally_poly.clone(), rel: "disjoint", want: "poly intersects", negate: true, esql_fn: "ST_Disjoint", wkt: wkt_poly, es: [Some(1.9), Some(48.0), Some(5.2), Some(4.8)] },
        Rel { task: "polygon-within", q: rally_poly.clone(), rel: "within", want: "poly within", negate: false, esql_fn: "ST_Within", wkt: wkt_poly, es: [Some(2.2), Some(12.9), Some(5.0), Some(4.8)] },
        Rel { task: "polygon-intersects-oresund", q: oresund.clone(), rel: "intersects", want: "oresund intersects", negate: false, esql_fn: "ST_Intersects", wkt: wkt_ore, es: [Some(2.5), Some(1.9), Some(5.9), Some(4.8)] },
    ];
    for r in &rels {
        if !t.want(r.task) {
            continue;
        }
        let wt = w(r.want);
        let member = |d: usize| wt.has(d) != r.negate;
        let count = if r.negate { all - wt.count } else { wt.count };
        let body = json!({"size": 10, "query": geo_q(&r.q, r.rel)});
        let (res, ms) = best(runs, || search(&body));
        let got = ids(&res);
        t.add(r.task, ms, format!("{} hits", got.len()), got.len() as u64 == count.min(10) && got.iter().all(|&d| member(d)), "hits are matches", r.es[0]);
        let body = json!({"size": 0, "query": geo_q(&r.q, r.rel), "aggs": {"count": {"value_count": {"field": "size"}}}});
        let (res, ms) = best(runs, || search(&body));
        let c = res["aggregations"]["count"]["value"].as_u64().unwrap_or(u64::MAX);
        t.add(&format!("{}-count", r.task), ms, format!("{c}"), c == count, "= brute", r.es[1]);
        let q = format!("FROM osm | WHERE {}(shape, TO_GEOSHAPE(\"{}\")) | LIMIT 10", r.esql_fn, r.wkt);
        let (res, ms) = best(runs, || esql(&q));
        t.add(&format!("{}-esql", r.task), ms, format!("{} rows", res.table.rows.len()), res.table.rows.len() as u64 == count.min(10), "rows = min(10, matches)", r.es[2]);
        let q = format!("FROM osm | WHERE {}(shape, TO_GEOSHAPE(\"{}\")) | STATS count=COUNT(size)", r.esql_fn, r.wkt);
        let (res, ms) = best(runs, || esql(&q));
        let c = cell(&res, "count").map_or(u64::MAX, |v| long(&v));
        t.add(&format!("{}-count-esql", r.task), ms, format!("{c}"), c == count, "= brute", r.es[3]);
    }
    let body = json!({"query": geo_q(&rally_poly, "intersects")});
    let (res, ms) = if t.want("polygon-intersects-orig") { best(runs, || search(&body)) } else { (Value::Null, 0.0) };
    let got = ids(&res);
    t.add("polygon-intersects-orig", ms, format!("{} hits, total {}", got.len(), total(&res)), got.len() as u64 == w("poly intersects").count.min(10) && got.iter().all(|&d| w("poly intersects").has(d)), "hits are matches", Some(4.02));

    // bbox-* (default size 10) and their ES|QL forms.
    for (rel, want, negate, f, es) in [("intersects", "bbox intersects", false, "ST_Intersects", Some(2.7)), ("contains", "bbox contains", false, "ST_Contains", Some(1.0)), ("disjoint", "bbox intersects", true, "ST_Disjoint", Some(2.0)), ("within", "bbox within", false, "ST_Within", Some(2.0))] {
        if !t.want(&format!("bbox-{rel}")) {
            continue;
        }
        let wt = w(want);
        let count = if negate { all - wt.count } else { wt.count };
        let body = json!({"query": geo_q(&bbox, rel)});
        let (res, ms) = best(runs, || search(&body));
        let got = ids(&res);
        t.add(&format!("bbox-{rel}"), ms, format!("{} hits", got.len()), got.len() as u64 == count.min(10) && got.iter().all(|&d| wt.has(d) != negate), "hits are matches", es);
        let q = format!("FROM osm | WHERE {f}(shape, TO_GEOSHAPE(\"BBOX((-0.1, 15.0, 61.0, 48.0))\")) | LIMIT 10");
        // The track's literal `BBOX((...))` (doubled parenthesis) is no
        // WKT ES parses: TO_GEOSHAPE gives null with a warning, so ES
        // returns no rows; so must we.
        let (res, ms) = best(runs, || esql(&q));
        t.add(&format!("bbox-{rel}-esql"), ms, format!("{} rows, {} warnings", res.table.rows.len(), res.warnings.len()), res.table.rows.is_empty() && !res.warnings.is_empty(), "0 rows + warning (as ES)", None);
    }

    // geoGrid filter queries (size 10) and their exact totals.
    for (task, q, want, es, esql_q) in [
        ("geoGrid_geohash", json!({"geo_grid": {"shape": {"geohash": "u0"}}}), "geohash u0", Some(2.1), "FROM osm | WHERE ST_GEOHASH(shape, 2) == TO_GEOHASH(\"u0\") | LIMIT 10"),
        ("geoGrid_geotile", json!({"geo_grid": {"shape": {"geotile": "6/32/22"}}}), "geotile 6/32/22", Some(2.2), "FROM osm | WHERE ST_GEOTILE(shape, 6) == TO_GEOTILE(\"6/32/22\") | LIMIT 10"),
        ("geoGrid_geohex", json!({"geo_grid": {"shape": {"geohex": "811fbffffffffff"}}}), "geohex 811fb", Some(2.2), "FROM osm | WHERE ST_GEOHEX(shape, 1) == TO_GEOHEX(\"811fbffffffffff\") | LIMIT 10"),
    ] {
        if !t.want(task) {
            continue;
        }
        let wt = w(want);
        let body = json!({"query": q});
        let (res, ms) = best(runs, || search(&body));
        let got = ids(&res);
        t.add(task, ms, format!("{} hits", got.len()), got.len() as u64 == wt.count.min(10) && got.iter().all(|&d| wt.has(d)), "hits are matches", es);
        let body = json!({"size": 0, "track_total_hits": true, "query": q});
        let (res, ms) = best(runs, || search(&body));
        t.add(&format!("{task} exact total"), ms, format!("{}", total(&res)), total(&res) == wt.count, "= brute", None);
        // ST_GEOHASH / ST_GEOTILE / ST_GEOHEX of geo_shape are "planned"
        // for the Elastic Stack (ES docs); ES 9.5.5 rejects them in
        // verification, so a verification error is the expected answer.
        let (res, ms) = best(runs, || esql_try(esql_q));
        match res {
            Err(e) => t.add(&format!("{task}-esql"), ms, "verification error", format!("{e:?}").contains("verification_exception"), "rejected (geo_shape grid fns not in 9.5.5)", None),
            Ok(r) => t.add(&format!("{task}-esql"), ms, format!("{} rows", r.table.rows.len()), r.table.rows.len() <= 10 && (wt.count == 0 || !r.table.rows.is_empty()), "1..=10 rows", None),
        }
    }

    // geoGrid aggs over the Paris box: geohash / geotile cells checked
    // against a brute-force scan of every matching doc's cells.
    let paris_bounds = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
    let paris_q = json!({"geo_bounding_box": {"shape": paris_bounds}});
    for (task, agg, es, esql_q) in [
        ("geoGrid_aggs_geohash", json!({"geohash_grid": {"field": "shape", "precision": 5, "bounds": paris_bounds}}), Some(16.4), "FROM osm | WHERE ST_Intersects(shape, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | EVAL geohash = ST_GEOHASH(shape, 5, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | STATS count = COUNT(*) BY geohash"),
        ("geoGrid_aggs_geotile", json!({"geotile_grid": {"field": "shape", "precision": 13, "bounds": paris_bounds}}), Some(10.7), "FROM osm | WHERE ST_Intersects(shape, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | EVAL geotile = ST_GEOTILE(shape, 13, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | STATS count = COUNT(*) BY geotile"),
        ("geoGrid_aggs_geohex", json!({"geohex_grid": {"field": "shape", "precision": 6, "bounds": paris_bounds}}), Some(883.0), "FROM osm | WHERE ST_Intersects(shape, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | EVAL geohex = ST_GEOHEX(shape, 6, TO_GEOSHAPE(\"BBOX(2.20, 2.40, 48.95, 48.75)\")) | STATS count = COUNT(*) BY geohex"),
    ] {
        if !t.want(task) {
            continue;
        }
        let body = json!({"query": paris_q, "size": 0, "aggs": {"zoomed-in": agg}});
        let (res, ms) = best(runs, || search(&body));
        let b = &res["aggregations"]["zoomed-in"]["buckets"];
        let pairs: u64 = b.as_array().map(|v| v.iter().map(|x| x["doc_count"].as_u64().unwrap_or(0)).sum()).unwrap_or(0);
        let (ok, check) = grid_check(&agg, b, &paris_dv, w("paris intersects").count);
        t.add(task, ms, format!("{} cells, {pairs} doc-cells", b.as_array().map_or(0, Vec::len)), ok, check, es);
        let (r2, ms) = best(runs, || esql_try(esql_q));
        match r2 {
            Err(e) => t.add(&format!("{task}-esql"), ms, "verification error", format!("{e:?}").contains("verification_exception"), "rejected (geo_shape grid fns not in 9.5.5)", None),
            Ok(r2) => {
                let esql_pairs: u64 = match col(&r2, "count") {
                    Some(cnt) => r2.table.rows.iter().map(|row| long(&row[cnt])).fold(0u64, u64::saturating_add),
                    None => u64::MAX,
                };
                t.add(&format!("{task}-esql"), ms, format!("{} cells, {esql_pairs} doc-cells", r2.table.rows.len()), esql_pairs == pairs && r2.table.rows.len() == b.as_array().map_or(0, Vec::len), "= the DSL agg's cells", None);
            }
        }
    }

    // centroid / bounds, global and under the Rally polygon, DSL against
    // ES|QL (two separate paths).
    let poly_q = geo_q(&rally_poly, "intersects");
    let nums = |s: &str| -> Vec<f64> { s.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'E' || c == 'e')).filter_map(|x| x.parse().ok()).collect() };
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * (1.0 + a.abs());
    for (task, filtered, centroid, bounds, es_dsl, es_esql) in [
        ("polygon-centroid", true, true, false, Some(484.0), Some(61.0)),
        ("polygon-bounds", true, false, true, Some(353.0), Some(59.0)),
        ("polygon-bounds-centroid", true, true, true, Some(769.0), Some(81.0)),
        ("centroid", false, true, false, Some(3595.0), Some(199.0)),
        ("bounds", false, false, true, Some(1901.0), Some(204.0)),
        ("bounds-centroid", false, true, true, Some(4603.0), Some(311.0)),
    ] {
        if !t.want(task) {
            continue;
        }
        let mut aggs = serde_json::Map::new();
        if bounds {
            aggs.insert("envelope".into(), json!({"geo_bounds": {"field": "shape"}}));
        }
        if centroid {
            aggs.insert("centroid".into(), json!({"geo_centroid": {"field": "shape"}}));
        }
        let mut body = json!({"size": 0, "aggs": aggs});
        if filtered {
            body["query"] = poly_q.clone();
        }
        let (res, ms) = best(runs, || search(&body));
        let mut stats = Vec::new();
        if centroid {
            stats.push("centroid = ST_CENTROID_AGG(shape)");
        }
        if bounds {
            stats.push("envelope = ST_EXTENT_AGG(shape)");
        }
        let q = format!("FROM osm {}| STATS {}", if filtered { format!("| WHERE ST_Intersects(shape, TO_GEOSHAPE(\"{wkt_poly}\")) ") } else { String::new() }, stats.join(", "));
        let (er, ems) = best(runs, || esql(&q));
        let mut ok = true;
        let mut shown = String::new();
        if centroid {
            let c = &res["aggregations"]["centroid"];
            let (lat, lon) = (c["location"]["lat"].as_f64().unwrap_or(f64::NAN), c["location"]["lon"].as_f64().unwrap_or(f64::NAN));
            let e = match cell(&er, "centroid") {
                Some(crate::esql::Val::Str(s)) => nums(&s),
                _ => Vec::new(),
            };
            ok &= e.len() == 2 && close(e[0], lon) && close(e[1], lat);
            shown += &format!("c=({lat:.4},{lon:.4}) n={} ", c["count"]);
        }
        if bounds {
            let b = &res["aggregations"]["envelope"]["bounds"];
            let (top, left, bottom, right) = (b["top_left"]["lat"].as_f64().unwrap_or(f64::NAN), b["top_left"]["lon"].as_f64().unwrap_or(f64::NAN), b["bottom_right"]["lat"].as_f64().unwrap_or(f64::NAN), b["bottom_right"]["lon"].as_f64().unwrap_or(f64::NAN));
            let e = match cell(&er, "envelope") {
                Some(crate::esql::Val::Str(s)) => nums(&s),
                _ => Vec::new(),
            };
            // BBOX(minX, maxX, maxY, minY)
            ok &= e.len() == 4 && close(e[0], left) && close(e[1], right) && close(e[2], top) && close(e[3], bottom);
            shown += &format!("b=[{top:.3},{left:.3},{bottom:.3},{right:.3}]");
        }
        t.add(task, ms, shown.clone(), ok, "DSL = ES|QL", es_dsl);
        t.add(&format!("{task}-esql"), ems, shown, ok, "ES|QL = DSL", es_esql);
    }

    // st-envelope / st-simplify.
    for (task, q, es) in [("st-envelope-esql", "FROM osm | EVAL env = ST_ENVELOPE(shape) | KEEP env | LIMIT 10", Some(4.7)), ("st-simplify-esql", "FROM osm | EVAL simplified = ST_SIMPLIFY(shape, 0.01) | KEEP simplified | LIMIT 10", Some(5.4))] {
        if !t.want(task) {
            continue;
        }
        let (res, ms) = best(runs, || esql(q));
        t.add(task, ms, format!("{} rows", res.table.rows.len()), res.table.rows.len() == 10.min(n), "10 rows", es);
    }
    for (task, q, exact) in [
        ("st-envelope-count-esql", "FROM osm | EVAL env = ST_ENVELOPE(shape), points = ST_NPOINTS(env) | WHERE points > 0 | STATS count = COUNT(size) BY points", true),
        ("st-simplify-count-esql", "FROM osm | EVAL simplified = ST_SIMPLIFY(shape, 0.01), points = ST_NPOINTS(simplified) | WHERE points > 0 | STATS count = COUNT(size) BY points", false),
    ] {
        if !t.want(task) {
            continue;
        }
        let (res, ms) = best(runs, || esql(q));
        let sum: u64 = match col(&res, "count") {
            Some(c) => res.table.rows.iter().map(|row| long(&row[c])).fold(0u64, u64::saturating_add),
            None => u64::MAX,
        };
        t.add(task, ms, format!("{sum} docs, {} groups", res.table.rows.len()), if exact { sum == all } else { sum > 0 && sum <= all }, if exact { "= all docs" } else { "1..=all docs" }, None);
    }

    if t.want("mvt-") {
        // mvt-hits and mvt-grid, exactly as rally-tracks geoshape.
        let lbox = json!({"geo_bounding_box": {"shape": {"top_left": [-0.703129, 51.618014], "bottom_right": [0, 51.179343]}}});
        let lw = w("london intersects");
        let body = json!({"_source": false, "size": 10000, "fields": [{"field": "shape", "format": "mvt(9/255/341)"}], "query": lbox, "sort": [{"size": {"order": "desc"}}]});
        let (res, ms) = best(runs, || search(&body));
        let got = ids(&res);
        let features = res["hits"]["hits"].as_array().map_or(0, |h| h.iter().filter(|x| x["fields"]["shape"].as_array().is_some_and(|a| !a.is_empty())).count());
        t.add("mvt-hits", ms, format!("{} hits, {features} features", got.len()), got.len() as u64 == lw.count.min(10_000) && got.iter().all(|&d| lw.has(d)), "hits are matches", Some(2035.0));
        let body = json!({"size": 0, "query": lbox, "aggregations": {"grid": {"geotile_grid": {"field": "shape", "precision": 17, "size": 65536,
            "bounds": {"top_left": [-0.703129, 51.618014], "bottom_right": [0, 51.179343]}}}}});
        let (res, ms) = best(runs, || search(&body));
        let b = res["aggregations"]["grid"]["buckets"].as_array().cloned().unwrap_or_default();
        let pairs: u64 = b.iter().map(|x| x["doc_count"].as_u64().unwrap_or(0)).sum();
        let (mut ok, mut check) = (!b.is_empty() && b.iter().all(|x| x["doc_count"].as_u64().unwrap_or(u64::MAX) <= lw.count), "cells hold matches".to_string());
        if exact_mvt {
            (ok, check) = grid_check(&body["aggregations"]["grid"], &res["aggregations"]["grid"]["buckets"], &london_dv, lw.count);
        }
        t.add("mvt-grid", ms, format!("{} cells, {pairs} doc-cells", b.len()), ok, check, Some(1408.0));
    }
    ingest_rows(t, "geoshape", &svc, n as u64, write_ms, "shape");
}

/// A grid aggregation's buckets checked against every matching doc's
/// cells by a brute-force scan over its bounding box (geohash / geotile)
/// or a flood over the H3 cells it touches (geohex), the bounds applied.
fn grid_check(agg: &Value, buckets: &Value, docs: &[Vec<u8>], matches: u64) -> (bool, String) {
    let Some(b) = buckets.as_array() else { return (false, "no buckets".into()) };
    let (kind, spec) = agg.as_object().and_then(|o| o.iter().next()).map(|(k, v)| (k.as_str(), v)).unwrap_or(("", &Value::Null));
    let precision = spec["precision"].as_u64().unwrap_or(0) as u32;
    let bounds = spec.get("bounds").map(|bv| {
        let p = |k: &str| crate::values::parse_geo_point(&bv[k]).unwrap();
        let ((top, left), (bottom, right)) = (p("top_left"), p("bottom_right"));
        crate::aggs::shape_tiler::Bounds { top, left, bottom, right }
    });
    let _ = matches;
    let mut want: std::collections::BTreeMap<String, u64> = Default::default();
    for dv in docs {
        let v = rustic_core::fieldtypes::shape_dv::ShapeValue::new(dv);
        let mut cells = std::collections::BTreeSet::new();
        if kind == "geohex_grid" {
            cells = crate::aggs::shape_tiler::brute_force_hex_cells(precision, bounds.as_ref(), &v);
        } else {
            crate::aggs::shape_tiler::brute_force_cells(kind, precision, bounds.as_ref(), &v, &mut cells);
        }
        for c in cells {
            *want.entry(c).or_default() += 1;
        }
    }
    let got: std::collections::BTreeMap<String, u64> = b.iter().map(|x| (x["key"].as_str().unwrap_or("").to_string(), x["doc_count"].as_u64().unwrap_or(0))).collect();
    // `size` may cut the bucket list: the kept buckets must be the top ones.
    let ok = if got.len() == want.len() { got == want } else { got.iter().all(|(k, n)| want.get(k) == Some(n)) };
    if !ok && kind == "geohex_grid" {
        let t = crate::aggs::shape_tiler::Tiler::Hex(crate::aggs::shape_tiler::HexTiler::new(precision, bounds.as_ref()));
        let mut out = Vec::new();
        let mut shown = 0;
        for dv in docs {
            let v = rustic_core::fieldtypes::shape_dv::ShapeValue::new(dv);
            t.cells(&v, &mut out);
            let got: std::collections::BTreeSet<String> = out.iter().map(|&h| t.key(h)).collect();
            let want = crate::aggs::shape_tiler::brute_force_hex_cells(precision, bounds.as_ref(), &v);
            if got != want && shown < 3 {
                shown += 1;
                eprintln!("  doc {:?}: tiler {got:?} brute {want:?}", v.centroid());
            }
        }
        eprintln!("  per-doc tiler/brute differences shown: {shown}");
    }
    if !ok {
        let diff: Vec<String> = got.iter().filter(|(k, n)| want.get(*k) != Some(n)).map(|(k, n)| format!("{k}: got {n} want {:?}", want.get(k))).take(5).collect();
        eprintln!("grid_check {kind}: {} got / {} want buckets; {diff:?}", got.len(), want.len());
    }
    (ok, "= brute-force cells".into())
}

/// Ingest and store rows for an index of `n` docs: indexing rate (the
/// engine write path, flushes and refresh; excluding the bench's own
/// generation and brute force), a force merge to one segment, and the
/// merged store per doc by section (from the segment's disk usage).
fn ingest_rows(t: &mut Table, label: &str, svc: &Arc<crate::index::IndexService>, n: u64, write_ms: f64, field: &str) {
    t.add(&format!("ingest {label}"), write_ms, format!("{:.0} docs/s", n as f64 / (write_ms / 1e3)), true, "timing", None);
    let before = svc.local_shards()[0].shard.reader_view().segments().len();
    let tm = Instant::now();
    if let Err(e) = svc.force_merge(1, false).and_then(|_| svc.refresh()) {
        t.add(&format!("force-merge {label} ({before} -> 1 segment)"), tm.elapsed().as_secs_f64() * 1e3, "error", false, format!("{e:?}"), None);
        return;
    }
    let merge_ms = tm.elapsed().as_secs_f64() * 1e3;
    let view = svc.local_shards()[0].shard.reader_view();
    let segs = view.segments().len();
    t.add(&format!("force-merge {label} ({before} -> 1 segment)"), merge_ms, format!("{segs} segments, {} docs", view.num_docs()), segs == 1 && view.num_docs() == n, "1 segment, every doc", None);
    let mut sum = rustic_core::segment::FieldDiskUsage::default();
    let (mut total, mut tiers) = (0u64, (0u64, 0u64, 0u64));
    let lookup = svc.lookup();
    let ids: Vec<u16> = lookup.get(field).into_iter().flat_map(|f| [f.core_id, f.geo_id]).flatten().collect();
    let mut field_bytes = 0u64;
    for seg in view.segments() {
        let u = seg.disk_usage().unwrap();
        total += u.terms + u.postings + u.fast + u.store + u.vectors + u.extra;
        tiers.0 += u.store;
        tiers.1 += u.fast;
        tiers.2 += u.extra;
        for (id, f) in &u.fields {
            sum.shape += f.shape;
            sum.geo += f.geo;
            sum.stored += f.stored;
            sum.doc_values += f.doc_values + f.doc_values_terms;
            sum.terms += f.terms;
            sum.postings += f.postings;
            sum.points += f.points;
            if ids.contains(id) {
                field_bytes += f.shape + f.geo + f.stored + f.doc_values + f.doc_values_terms + f.terms + f.postings + f.points;
            }
        }
    }
    let per = |b: u64| b as f64 / n.max(1) as f64;
    t.add(
        &format!("store {label}"),
        0.0,
        format!("{:.1} B/doc", per(total)),
        total > 0,
        format!(
            "triangles {:.1}, geo index {:.1}, stored/source {:.1}, doc values {:.1}, terms+postings {:.1}, points {:.1}; {field} {:.1} B/doc (store {:.1}, fast {:.1}, extra {:.1})",
            per(sum.shape), per(sum.geo), per(sum.stored), per(sum.doc_values), per(sum.terms + sum.postings), per(sum.points), per(field_bytes), per(tiers.0), per(tiers.1), per(tiers.2)
        ),
        None,
    );
}

// ---------------------------------------------------------------------------
// geo_point through the search API (Rally `geopoint` grid aggs and
// distanceFilter10kGTE) and geo_shape points (Rally `geopointshape`).

/// `n` points: a fifth around Paris (the geoGrid_aggs box), a quarter in
/// western Europe, the rest worldwide. `(lon, lat)`, quantized as indexed.
fn suite_points(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|i| {
            let (lon, lat) = match i % 20 {
                0..=3 => (r.range(2.1, 2.5), r.range(48.7, 49.0)),
                4..=8 => (r.range(-5.0, 20.0), r.range(42.0, 62.0)),
                _ => (r.range(-180.0, 180.0), r.range(-85.0, 85.0)),
            };
            let (lat, lon) = geo::quantize(lat, lon);
            (lon, lat)
        })
        .collect()
}

/// Index `docs` (JSON sources, ids `0..`) into a new index of `node`
/// in three segments: the engine write path's time in ms.
fn index_docs(svc: &Arc<crate::index::IndexService>, n: usize, mut doc: impl FnMut(usize) -> Value) -> f64 {
    let mut ms = 0.0;
    let mut batch = Vec::with_capacity(20_000);
    let flush_at = [n / 3, 2 * n / 3];
    for i in 0..n {
        batch.push(WriteReq::Index { id: Some(i.to_string()), source: doc(i).to_string(), routing: None, cond: Default::default() });
        if batch.len() == 20_000 || i + 1 == n || flush_at.contains(&i) {
            let t = Instant::now();
            for r in svc.write(std::mem::take(&mut batch), false) {
                r.unwrap();
            }
            if flush_at.contains(&i) {
                svc.flush().unwrap();
            }
            ms += t.elapsed().as_secs_f64() * 1e3;
        }
    }
    let t = Instant::now();
    svc.flush().unwrap();
    svc.refresh().unwrap();
    ms + t.elapsed().as_secs_f64() * 1e3
}

fn suite_search(svc: &Arc<crate::index::IndexService>, body: &Value) -> Value {
    let req = crate::search::parse_request(body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(svc.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap_or_else(|e| panic!("{body}: {e}"))
}

/// Whether an indexed point is in the Paris box as ES's
/// `geo_bounding_box` takes it: against the box corners on the index's
/// grid (ES matches a point a quantum west of 2.20).
fn in_paris(&(lon, lat): &(f64, f64)) -> bool {
    let (b, l) = geo::quantize(48.75, 2.20);
    let (t, r) = geo::quantize(48.95, 2.40);
    lat >= b && lat <= t && lon >= l && lon <= r
}

/// The point grid cells (`geohash` 5, `geotile` 13, `geohex` 6) of the
/// points in the Paris box, counted by brute force.
fn paris_cells(points: &[(f64, f64)], grid: &str) -> std::collections::BTreeMap<String, u64> {
    let mut m = std::collections::BTreeMap::new();
    for p in points {
        if !in_paris(p) {
            continue;
        }
        let &(lon, lat) = p;
        let k = match grid {
            "geohash" => geo::geohash(lat, lon, 5),
            "geotile" => {
                let (x, y) = geo::geotile(lat, lon, 13);
                format!("13/{x}/{y}")
            }
            _ => crate::aggs::geo_hex::cell_address(crate::aggs::geo_hex::point_cell_id(lat, lon, 6).unwrap()).unwrap(),
        };
        *m.entry(k).or_insert(0) += 1;
    }
    m
}

fn bucket_map(b: &Value) -> std::collections::BTreeMap<String, u64> {
    b.as_array().into_iter().flatten().map(|x| (x["key"].as_str().unwrap_or_default().to_string(), x["doc_count"].as_u64().unwrap_or(0))).collect()
}

fn point_search_section(n: usize, runs: usize, t: &mut Table) {
    eprintln!("geo_point search: {n} docs");
    let points = suite_points(n, 41);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("osmgeopoints", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
    let svc = node.get_index("osmgeopoints").unwrap();
    let write_ms = index_docs(&svc, n, |i| json!({"location": [points[i].0, points[i].1]}));
    let paris = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
    for (grid, agg, es) in [
        ("geohash", json!({"geohash_grid": {"field": "location", "precision": 5, "bounds": paris}}), Some(3.0)),
        ("geotile", json!({"geotile_grid": {"field": "location", "precision": 13, "bounds": paris}}), Some(3.0)),
        ("geohex", json!({"geohex_grid": {"field": "location", "precision": 6, "bounds": paris}}), Some(6.0)),
    ] {
        let body = json!({"query": {"geo_bounding_box": {"location": paris}}, "size": 0, "aggs": {"zoomed-in": agg}});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let got = bucket_map(&res["aggregations"]["zoomed-in"]["buckets"]);
        let want = paris_cells(&points, grid);
        t.add(&format!("geopoint geoGrid_aggs_{grid}"), ms, format!("{} cells", got.len()), got == want, "= brute cells", es);
    }
    // distanceFilter10kGTE: size 10,000 with _source, a disjoint circle.
    let body = json!({"size": 10000, "query": {"geo_shape": {"location": {"shape": {"type": "circle", "radius": "400km", "coordinates": [7.0, 55.0]}, "relation": "disjoint"}}}});
    let circle = ref_query(&json!({"type": "circle", "radius": "400km", "coordinates": [7.0, 55.0]}), shape::SpatialRelation::Intersects);
    let (res, ms) = best(runs, || suite_search(&svc, &body));
    let hits = res["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let ok = hits.len() == 10_000.min(n)
        && hits.iter().all(|h| {
            let i: usize = h["_id"].as_str().unwrap().parse().unwrap();
            h["_source"]["location"].is_array() && !ref_matches(&circle, &doc_shape(&format!("POINT ({} {})", points[i].0, points[i].1)).0)
        });
    t.add("geopoint distanceFilter10kGTE (fetch)", ms, format!("{} hits", hits.len()), ok, "hits outside the circle, with _source", Some(146.0));
    ingest_rows(t, "geopoint", &svc, n as u64, write_ms, "location");
}

/// The run times of `f` (after one warm-up): `(last result, p50, p90)`.
fn timed<T>(runs: usize, mut f: impl FnMut() -> T) -> (T, f64, f64) {
    let mut out = f();
    let mut ms: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let t = Instant::now();
            out = f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    (out, at(0.5), at(0.9))
}

/// Rally `geopoint`'s grid tasks alone, for iterating on them: the
/// bounded `geohash_grid` / `geotile_grid` / `geohex_grid` aggs over the
/// Paris box and the `geo_grid` filters, on a many-segment index kept on
/// disk (`GEO_GRID_DIR`: built once, reopened by later runs). Rows are
/// p50 (p90 printed), each checked against brute force.
fn point_grid_section(n: usize, runs: usize, t: &mut Table) {
    let points = suite_points(n, 53);
    let keep = std::env::var_os("GEO_GRID_DIR");
    let tmp = tempfile::tempdir().unwrap();
    let dir = keep.as_ref().map(std::path::PathBuf::from).unwrap_or_else(|| tmp.path().to_path_buf());
    let fresh = !dir.join(".grid_built").exists();
    if fresh {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
    }
    let node = Node::open(&dir).unwrap();
    if fresh {
        let segs = env("GEO_GRID_SEGMENTS", 35);
        let settings = json!({"number_of_shards": 1, "number_of_replicas": 0, "index.merge.policy.max_merged_segment": format!("{}b", (n * 90 / segs.max(1)).max(1 << 20))});
        node.create_index("osmgeopoints", &json!({"settings": settings, "mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
        let svc = node.get_index("osmgeopoints").unwrap();
        index_docs(&svc, n, |i| json!({"location": [points[i].0, points[i].1]}));
        std::fs::write(dir.join(".grid_built"), n.to_string()).unwrap();
    }
    let svc = node.get_index("osmgeopoints").unwrap();
    let _ = crate::license::service(&node).start_trial(&node, "trial", true);
    eprintln!("grid: {n} points, {} segments ({})", svc.local_shards()[0].shard.reader_view().segments().len(), if fresh { "built" } else { "reopened" });
    let paris = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
    if t.want("grid bbox count") {
        let in_box = in_paris;
        let want = points.iter().filter(|p| in_box(p)).count() as u64;
        let body = json!({"size": 0, "track_total_hits": true, "query": {"geo_bounding_box": {"location": paris}}});
        let (res, p50, _) = timed(runs, || suite_search(&svc, &body));
        let got = res["hits"]["total"]["value"].as_u64().unwrap_or(0);
        if got != want {
            let body = json!({"size": 10000, "query": {"bool": {"filter": {"geo_bounding_box": {"location": paris}}, "must_not": {"geo_bounding_box": {"location": {"top_left": "POINT (2.2000001 48.9499999)", "bottom_right": "POINT (2.3999999 48.7500001)"}}}}}});
            let res = suite_search(&svc, &body);
            for h in res["hits"]["hits"].as_array().into_iter().flatten() {
                let i: usize = h["_id"].as_str().unwrap().parse().unwrap();
                eprintln!("grid bbox edge doc {i}: {:?} in brute box {}", points[i], in_box(&points[i]));
            }
        }
        t.add("grid bbox count", p50, format!("{got}"), got == want, "= brute", None);
    }
    for (grid, agg) in [
        ("geohash", json!({"geohash_grid": {"field": "location", "precision": 5, "bounds": paris}})),
        ("geotile", json!({"geotile_grid": {"field": "location", "precision": 13, "bounds": paris}})),
        ("geohex", json!({"geohex_grid": {"field": "location", "precision": 6, "bounds": paris}})),
    ] {
        let task = format!("grid aggs_{grid}");
        if !t.want(&task) {
            continue;
        }
        let body = json!({"query": {"geo_bounding_box": {"location": paris}}, "size": 0, "aggs": {"zoomed-in": agg}});
        let (res, p50, p90) = timed(runs, || suite_search(&svc, &body));
        let got = bucket_map(&res["aggregations"]["zoomed-in"]["buckets"]);
        let want = paris_cells(&points, grid);
        eprintln!("{task}: p50 {p50:.2} ms p90 {p90:.2} ms");
        if got != want {
            let d: Vec<_> = want.iter().filter(|(k, v)| got.get(*k) != Some(v)).take(3).map(|(k, v)| (k.clone(), *v, got.get(k).copied())).collect();
            eprintln!("{task}: {} vs {} docs, e.g. (cell, brute, ours) {d:?}", got.values().sum::<u64>(), want.values().sum::<u64>());
        }
        t.add(&task, p50, format!("{} cells, p90 {p90:.2}", got.len()), got == want, "= brute cells", None);
    }
    // Rally geopoint's polygon / circle filters: size 10 with the default
    // bounded total (10,000), checked against brute force.
    let rally_poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
    let circle = json!({"type": "circle", "radius": "400km", "coordinates": [7.0, 55.0]});
    for (task, shape, rel) in [("grid polygon-intersects", &rally_poly, "intersects"), ("grid polygon-disjoint", &rally_poly, "disjoint"), ("grid polygon-within", &rally_poly, "within"), ("grid distanceFilterGTE", &circle, "disjoint")] {
        if !t.want(task) {
            continue;
        }
        let r = match rel {
            "within" => shape::SpatialRelation::Within,
            "disjoint" => shape::SpatialRelation::Disjoint,
            _ => shape::SpatialRelation::Intersects,
        };
        let refq = ref_query(shape, r);
        let member = |&(lon, lat): &(f64, f64)| ref_matches(&refq, &doc_shape(&format!("POINT ({lon} {lat})")).0);
        let total = points.iter().filter(|p| member(p)).count() as u64;
        let body = json!({"size": 10, "query": {"geo_shape": {"location": {"shape": shape, "relation": rel}}}});
        let (res, p50, p90) = timed(runs, || suite_search(&svc, &body));
        let ids: Vec<usize> = res["hits"]["hits"].as_array().into_iter().flatten().map(|h| h["_id"].as_str().unwrap().parse().unwrap()).collect();
        let tot = res["hits"]["total"]["value"].as_u64().unwrap_or(u64::MAX);
        let ok = ids.len() as u64 == total.min(10) && ids.iter().all(|&i| member(&points[i])) && tot == total.min(10_000);
        eprintln!("{task}: p50 {p50:.2} ms p90 {p90:.2} ms");
        t.add(task, p50, format!("{} hits, total {tot} of {total}, p90 {p90:.2}", ids.len()), ok, "hits match, total = min(brute, 10000)", None);
    }
    // Rally geopoint's polygon-centroid: a geo_centroid under the Rally
    // polygon filter (count exact, centroid to 1e-9 of the brute mean).
    if t.want("grid polygon-centroid") {
        let refq = ref_query(&rally_poly, shape::SpatialRelation::Intersects);
        let (mut n, mut sl, mut so) = (0u64, 0.0f64, 0.0f64);
        for &(lon, lat) in &points {
            if ref_matches(&refq, &doc_shape(&format!("POINT ({lon} {lat})")).0) {
                n += 1;
                sl += lat;
                so += lon;
            }
        }
        let body = json!({"size": 0, "query": {"geo_shape": {"location": {"shape": rally_poly}}}, "aggs": {"centroid": {"geo_centroid": {"field": "location"}}}});
        let (res, p50, p90) = timed(runs, || suite_search(&svc, &body));
        let c = &res["aggregations"]["centroid"];
        let got_n = c["count"].as_u64().unwrap_or(0);
        let (glat, glon) = (c["location"]["lat"].as_f64().unwrap_or(f64::NAN), c["location"]["lon"].as_f64().unwrap_or(f64::NAN));
        let ok = got_n == n && (glat - sl / n as f64).abs() < 1e-9 && (glon - so / n as f64).abs() < 1e-9;
        eprintln!("grid polygon-centroid: p50 {p50:.2} ms p90 {p90:.2} ms");
        t.add("grid polygon-centroid", p50, format!("n={got_n} ({glat:.6}, {glon:.6}), p90 {p90:.2}"), ok, "count = brute, centroid = brute mean", None);
    }
    // Rally geopoint's ES|QL grid sorts (`ST_GEO*(location, p)::long`,
    // SORT ASC, LIMIT 10), with and without the location column; the
    // geohex keys checked against brute force.
    let esql = |q: &str| crate::esql::run(&node, q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() });
    for (task, f, p, keep) in [
        ("grid sort_geohex-esql", "ST_GEOHEX", 5, "grid, location"),
        ("grid sort_geohex-esql-nolocation", "ST_GEOHEX", 5, "grid"),
        ("grid sort_geohash-esql", "ST_GEOHASH", 5, "grid, location"),
        ("grid sort_geotile-esql", "ST_GEOTILE", 5, "grid, location"),
    ] {
        if !t.want(task) {
            continue;
        }
        let q = format!("FROM osmgeopoints | EVAL grid = {f}(location, {p})::long | SORT grid ASC | KEEP {keep} | LIMIT 10");
        let (res, p50, p90) = timed(runs.min(3), || esql(&q));
        let Ok(res) = res else {
            t.add(task, p50, "error".to_string(), false, "10 rows", None);
            continue;
        };
        let got: Vec<i64> = res.table.rows.iter().map(|r| match &r[0] {
            crate::esql::Val::Long(x) | crate::esql::Val::Int(x) => *x,
            _ => i64::MIN,
        }).collect();
        let ok = if f == "ST_GEOHEX" {
            let mut all: Vec<i64> = points.iter().map(|&(lon, lat)| crate::aggs::geo_hex::point_cell_id(lat, lon, p as u32).unwrap() as i64).collect();
            all.sort_unstable();
            all.truncate(10);
            got == all
        } else {
            got.len() == 10 && got.windows(2).all(|w| w[0] <= w[1])
        };
        eprintln!("{task}: p50 {p50:.2} ms p90 {p90:.2} ms");
        t.add(task, p50, format!("{} rows, first {:?}", got.len(), got.first()), ok, "keys = brute (geohex), sorted", None);
    }
    let hex_cell = crate::aggs::geo_hex::parse_cell("811fbffffffffff").unwrap();
    for (grid, q) in [("geohash", json!({"geohash": "u0"})), ("geotile", json!({"geotile": "6/32/22"})), ("geohex", json!({"geohex": "811fbffffffffff"}))] {
        let task = format!("grid filter_{grid}");
        if !t.want(&task) {
            continue;
        }
        let member = |&(lon, lat): &(f64, f64)| match grid {
            "geohash" => geo::geohash(lat, lon, 2) == "u0",
            "geotile" => geo::geotile(lat, lon, 6) == (32, 22),
            _ => crate::aggs::geo_hex::point_cell_id(lat, lon, 1) == Some(hex_cell),
        };
        let total = points.iter().filter(|p| member(p)).count() as u64;
        let body = json!({"size": 10, "query": {"geo_grid": {"location": q}}});
        let (res, p50, p90) = timed(runs, || suite_search(&svc, &body));
        let ids: Vec<usize> = res["hits"]["hits"].as_array().into_iter().flatten().map(|h| h["_id"].as_str().unwrap().parse().unwrap()).collect();
        let tot = res["hits"]["total"]["value"].as_u64().unwrap_or(u64::MAX);
        let ok = ids.len() as u64 == total.min(10) && ids.iter().all(|&i| member(&points[i])) && tot == total.min(10_000);
        eprintln!("{task}: p50 {p50:.2} ms p90 {p90:.2} ms");
        t.add(&task, p50, format!("{} hits of {total}, total {tot}, p90 {p90:.2}", ids.len()), ok, "hits in the cell, total = min(brute, 10000)", None);
    }
}

/// Rally `geopoint` rows on an unmerged many-segment index (as Rally's
/// geo_point index is queried: dozens of segments), checked against a
/// brute-force scan of the points: ES|QL distance sorts, the distance
/// count, the geotile grid filter and the disjoint count.
fn point_rally_section(n: usize, runs: usize, t: &mut Table) {
    use shape::SpatialRelation as R;
    eprintln!("geopoint rally: {n} docs");
    let points = suite_points(n, 47);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    // `GEO_SUITE_RALLY_SEGMENTS`: about that many segments (merges stop at
    // the size that leaves them; Rally's geo_point index ends with dozens);
    // default: as merging leaves them.
    let segs = env("GEO_SUITE_RALLY_SEGMENTS", 0);
    let settings = if segs > 0 {
        json!({"number_of_shards": 1, "number_of_replicas": 0, "index.merge.policy.max_merged_segment": format!("{}b", (n * 90 / segs).max(1 << 20))})
    } else {
        json!({"number_of_shards": 1, "number_of_replicas": 0})
    };
    node.create_index("osmgeopoints", &json!({"settings": settings, "mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
    let svc = node.get_index("osmgeopoints").unwrap();
    let _ = crate::license::service(&node).start_trial(&node, "trial", true);
    index_docs(&svc, n, |i| json!({"location": [points[i].0, points[i].1]}));
    eprintln!("geopoint rally: {} segments", svc.local_shards()[0].shard.reader_view().segments().len());
    let esql = |q: &str| crate::esql::run(&node, q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }).unwrap_or_else(|e| panic!("{q}: {e:?}"));
    let num = |v: &crate::esql::Val| match v {
        crate::esql::Val::Double(x) => *x,
        crate::esql::Val::Long(x) | crate::esql::Val::Int(x) => *x as f64,
        _ => f64::NAN,
    };
    // Brute force: ES|QL `ST_DISTANCE` of every point (sloppy haversine
    // over the quantized coordinates).
    let (qlat, qlon) = geo::quantize(55.0, 7.0);
    let mut dists: Vec<f64> = points.iter().map(|&(lon, lat)| geo::sloppy_haversin_meters(lat, lon, qlat, qlon)).collect();
    let within_100 = rustic_core::geo::GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 100_000.0 };
    let count_100 = points.iter().filter(|&&(lon, lat)| within_100.contains(lat, lon)).count() as u64;
    let tile = crate::fieldtypes::geo::tile_bbox(6, 32, 22);
    let env = |(top, left, bottom, right): (f64, f64, f64, f64)| json!({"type": "envelope", "coordinates": [[left, top], [right, bottom]]});
    let rally_poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
    let refs = [ref_query(&env(tile), R::Intersects), ref_query(&rally_poly, R::Intersects)];
    let mut in_tile = 0u64;
    let mut in_poly = 0u64;
    for &(lon, lat) in &points {
        let (tris, _) = doc_shape(&format!("POINT ({lon} {lat})"));
        in_tile += u64::from(ref_matches(&refs[0], &tris));
        in_poly += u64::from(ref_matches(&refs[1], &tris));
    }
    dists.sort_by(f64::total_cmp);
    let top10: Vec<f64> = dists.iter().copied().take(10).collect();
    let close = |a: &[f64]| a.len() == top10.len() && a.iter().zip(&top10).all(|(x, y)| (x - y).abs() <= 1e-6 * y.max(1.0));
    for (task, q, es) in [
        ("geopoint rally distanceSort-esql", "FROM osmgeopoints | EVAL distance = ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) | SORT distance ASC | LIMIT 10", Some(42.0)),
        ("geopoint rally distanceSort-esql-partial", "FROM osmgeopoints | EVAL distance = ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")), loc = location::string | SORT distance ASC, loc ASC | LIMIT 10", None),
        ("geopoint rally distanceFilterSort-esql", "FROM osmgeopoints  | EVAL distance = ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) | WHERE distance <= 400000 | SORT distance ASC | LIMIT 10", None),
        ("geopoint rally distanceFilterSort-esql-partial", "FROM osmgeopoints  | EVAL distance = ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")), loc = location::string | WHERE distance <= 400000 | SORT distance ASC, loc ASC | LIMIT 10", None),
    ] {
        if !t.want(task) {
            continue;
        }
        let (res, ms) = best(runs, || esql(q));
        let di = res.table.cols.iter().position(|c| c.name == "distance").unwrap_or(0);
        let got: Vec<f64> = res.table.rows.iter().map(|r| num(&r[di])).collect();
        t.add(task, ms, format!("{} rows", got.len()), close(&got), "distances = brute top 10", es);
    }
    if t.want("geopoint rally distanceFilterCount100") {
        let body = json!({"query": {"geo_distance": {"distance": "100km", "location": [7.0, 55.0]}}, "aggs": {"count": {"value_count": {"field": "location"}}}});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let c = res["aggregations"]["count"]["value"].as_u64().unwrap_or(u64::MAX);
        t.add("geopoint rally distanceFilterCount100", ms, format!("{c}"), c == count_100, "= brute", None);
        let (res, ms) = best(runs, || esql("FROM osmgeopoints | WHERE ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) <= 100000 | STATS count=COUNT(*)"));
        let c = res.table.rows.first().map_or(f64::NAN, |r| num(&r[0])) as u64;
        t.add("geopoint rally distanceFilterCount100-esql", ms, format!("{c}"), c == count_100, "= brute", None);
    }
    if t.want("geopoint rally distanceFilterEQ-esql") {
        let d = top10.get(3).copied().unwrap_or(0.0);
        let q = format!("FROM osmgeopoints | WHERE ST_Distance(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) == {d:?} | LIMIT 10");
        let (res, ms) = best(runs, || esql(&q));
        let want = dists.iter().filter(|&&x| x == d).count().min(10);
        let got = res.table.rows.len();
        t.add("geopoint rally distanceFilterEQ-esql", ms, format!("{got} rows (brute {want})"), got <= 10 && (got == want || want == 0), "rows = brute (at most 10)", None);
    }
    if t.want("geopoint rally geoGrid_geotile") {
        let body = json!({"size": 10, "query": {"geo_grid": {"location": {"geotile": "6/32/22"}}}});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let hits = res["hits"]["hits"].as_array().map_or(0, Vec::len) as u64;
        t.add("geopoint rally geoGrid_geotile", ms, format!("{hits} hits"), hits == in_tile.min(10), "min(10, brute)", None);
    }
    if t.want("geopoint rally disjoint-count-esql") {
        let (res, ms) = best(runs, || esql("FROM osmgeopoints | WHERE ST_Disjoint(location, TO_GEOSHAPE(\"POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))\")) | STATS count=COUNT(*)"));
        let c = res.table.rows.first().map_or(f64::NAN, |r| num(&r[0])) as u64;
        t.add("geopoint rally disjoint-count-esql", ms, format!("{c}"), c == n as u64 - in_poly, "= brute", None);
    }
}

fn point_shape_section(n: usize, runs: usize, t: &mut Table) {
    use shape::SpatialRelation as R;
    eprintln!("geopointshape: {n} points as geo_shape");
    let points = suite_points(n, 43);
    let rally_poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
    let bbox = json!({"type": "envelope", "coordinates": [[-0.1, 61.0], [15.0, 48.0]]});
    let env = |(top, left, bottom, right): (f64, f64, f64, f64)| json!({"type": "envelope", "coordinates": [[left, top], [right, bottom]]});
    let hex_polys: Vec<Value> = crate::aggs::geo_hex::cell_polys(crate::aggs::geo_hex::parse_cell("811fbffffffffff").unwrap()).into_iter().map(|r| json!([r.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()])).collect();
    let refs: Vec<(&str, rustic_core::fieldtypes::geo2d::ShapeQuery)> = vec![
        ("poly intersects", ref_query(&rally_poly, R::Intersects)),
        ("poly within", ref_query(&rally_poly, R::Within)),
        ("poly contains", ref_query(&rally_poly, R::Contains)),
        ("bbox intersects", ref_query(&bbox, R::Intersects)),
        ("geohash u0", ref_query(&env(geo::geohash_bbox("u0").unwrap()), R::Intersects)),
        ("geotile 6/32/22", ref_query(&env(crate::fieldtypes::geo::tile_bbox(6, 32, 22)), R::Intersects)),
        ("geohex 811fb", ref_query(&json!({"type": "multipolygon", "coordinates": hex_polys}), R::Intersects)),
    ];
    let mut bits: Vec<Vec<bool>> = vec![vec![false; n]; refs.len()];
    // The docs the grid aggs' `geo_bounding_box` query selects.
    let paris_q = ref_query(&json!({"type": "envelope", "coordinates": [[2.20, 48.95], [2.40, 48.75]]}), R::Intersects);
    let mut paris_dv: Vec<Vec<u8>> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let (tris, dv) = doc_shape(&format!("POINT ({} {})", p.0, p.1));
        for (k, (_, q)) in refs.iter().enumerate() {
            bits[k][i] = ref_matches(q, &tris);
        }
        if ref_matches(&paris_q, &tris) {
            paris_dv.push(dv);
        }
    }
    let count = |k: usize| bits[k].iter().filter(|b| **b).count() as u64;
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("osmgeoshapes", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"location": {"type": "geo_shape"}}}})).unwrap();
    let svc = node.get_index("osmgeoshapes").unwrap();
    let _ = crate::license::service(&node).start_trial(&node, "trial", true);
    let write_ms = index_docs(&svc, n, |i| json!({"location": format!("POINT ({} {})", points[i].0, points[i].1)}));
    // `GEO_SUITE_MERGE_FIRST`: query one force-merged segment, as Rally
    // does (with `GEO_SUITE_POINT_SHAPES` at tens of millions: its size).
    if let Some(k) = merge_first() {
        let t0 = Instant::now();
        svc.force_merge(k, false).and_then(|_| svc.refresh()).unwrap();
        eprintln!("geopointshape: merged to {} segment(s) in {:.1} s", svc.local_shards()[0].shard.reader_view().segments().len(), t0.elapsed().as_secs_f64());
    }
    let esql = |q: &str| crate::esql::run(&node, q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }).unwrap_or_else(|e| {
        eprintln!("ES|QL error: {q}: {e:?}");
        crate::esql::EsqlResult { table: crate::esql::Table { cols: Vec::new(), rows: Vec::new() }, warnings: vec![format!("ERROR {e:?}")], features: Default::default(), functions: Default::default(), took_ms: 0, documents_found: 0, values_loaded: 0, rows_emitted: 0, settings: Default::default(), col_meta: Default::default(), all_columns: None }
    });
    let long = |v: &crate::esql::Val| match v {
        crate::esql::Val::Long(x) | crate::esql::Val::Int(x) => *x as u64,
        _ => u64::MAX,
    };
    let ids = |r: &Value| -> Vec<usize> { r["hits"]["hits"].as_array().map(|h| h.iter().map(|x| x["_id"].as_str().unwrap().parse().unwrap()).collect()).unwrap_or_default() };
    let wkt_poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
    let all = n as u64;
    for (rel, k, negate, f, es) in [("intersects", 0usize, false, "ST_Intersects", [Some(2.46), Some(6.1), None]), ("contains", 2, false, "ST_Contains", [Some(1.41), None, None]), ("disjoint", 0, true, "ST_Disjoint", [Some(1.81), Some(4.7), None]), ("within", 1, false, "ST_Within", [Some(1.57), Some(4.3), None])] {
        let member = |d: usize| bits[k][d] != negate;
        let c = if negate { all - count(k) } else { count(k) };
        let q = json!({"geo_shape": {"location": {"shape": rally_poly, "relation": rel}}});
        if !t.want(&format!("geopointshape polygon-{rel}")) {
            continue;
        }
        let body = json!({"size": 10, "query": q});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let got = ids(&res);
        t.add(&format!("geopointshape polygon-{rel}"), ms, format!("{} hits", got.len()), got.len() as u64 == c.min(10) && got.iter().all(|&d| member(d)), "hits are matches", es[0]);
        let body = json!({"size": 0, "track_total_hits": true, "query": q});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let tot = res["hits"]["total"]["value"].as_u64().unwrap_or(u64::MAX);
        t.add(&format!("geopointshape polygon-{rel}-count"), ms, format!("{tot}"), tot == c, "= brute", None);
        let qs = format!("FROM osmgeoshapes | WHERE {f}(location, TO_GEOSHAPE(\"{wkt_poly}\")) | LIMIT 10");
        let (res, ms) = best(runs, || esql(&qs));
        t.add(&format!("geopointshape polygon-{rel}-esql"), ms, format!("{} rows", res.table.rows.len()), res.table.rows.len() as u64 == c.min(10), "rows = min(10, matches)", None);
        let qs = format!("FROM osmgeoshapes | WHERE {f}(location, TO_GEOSHAPE(\"{wkt_poly}\")) | STATS count=COUNT(*)");
        let (res, ms) = best(runs, || esql(&qs));
        let got = res.table.rows.first().and_then(|r| r.first()).map_or(u64::MAX, long);
        t.add(&format!("geopointshape polygon-{rel}-count-esql"), ms, format!("{got}"), got == c, "= brute", es[1]);
    }
    let body = json!({"query": {"geo_shape": {"location": {"shape": bbox}}}});
    let (res, ms) = best(runs, || suite_search(&svc, &body));
    let got = ids(&res);
    t.add("geopointshape bbox-intersects", ms, format!("{} hits", got.len()), got.len() as u64 == count(3).min(10) && got.iter().all(|&d| bits[3][d]), "hits are matches", Some(1.52));
    // The track's `BBOX((...))` does not parse in ES: no rows, a warning.
    let (res, ms) = best(runs, || esql("FROM osmgeoshapes | WHERE ST_Intersects(location, TO_GEOSHAPE(\"BBOX((-0.1, 15.0, 61.0, 48.0))\")) | LIMIT 10"));
    t.add("geopointshape bbox-intersects-esql", ms, format!("{} rows, {} warnings", res.table.rows.len(), res.warnings.len()), res.table.rows.is_empty() && !res.warnings.is_empty(), "0 rows + warning (as ES)", None);
    for (task, q, k, es) in [
        ("geoGrid_geohash", json!({"geo_grid": {"location": {"geohash": "u0"}}}), 4usize, Some(1.64)),
        ("geoGrid_geotile", json!({"geo_grid": {"location": {"geotile": "6/32/22"}}}), 5, Some(1.66)),
        ("geoGrid_geohex", json!({"geo_grid": {"location": {"geohex": "811fbffffffffff"}}}), 6, Some(1.57)),
    ] {
        let body = json!({"query": q});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let got = ids(&res);
        t.add(&format!("geopointshape {task}"), ms, format!("{} hits", got.len()), got.len() as u64 == count(k).min(10) && got.iter().all(|&d| bits[k][d]), "hits are matches", es);
    }
    let paris = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
    for (grid, agg, es) in [
        ("geohash", json!({"geohash_grid": {"field": "location", "precision": 5, "bounds": paris}}), None),
        ("geotile", json!({"geotile_grid": {"field": "location", "precision": 13, "bounds": paris}}), Some(28.2)),
        ("geohex", json!({"geohex_grid": {"field": "location", "precision": 6, "bounds": paris}}), None),
    ] {
        let body = json!({"query": {"geo_bounding_box": {"location": paris}}, "size": 0, "aggs": {"zoomed-in": agg}});
        let (res, ms) = best(runs, || suite_search(&svc, &body));
        let got = &res["aggregations"]["zoomed-in"]["buckets"];
        let in_box = paris_dv.len() as u64;
        let (ok, check) = grid_check(&agg, got, &paris_dv, in_box);
        t.add(&format!("geopointshape geoGrid_aggs_{grid}"), ms, format!("{} cells", got.as_array().map_or(0, Vec::len)), ok, check, es);
    }
    ingest_rows(t, "geopointshape", &svc, n as u64, write_ms, "location");
}

// ---------------------------------------------------------------------------
// Ingest (Rally geoshape index-append-polygons / -linestrings).

/// `n` OSM-like polygons, then `n` linestrings (single and multi), each
/// indexed in 100-doc bulks through the engine write path, as Rally's
/// bulks are (one client here): docs/s, bulk p50 / p90, and the shape
/// phases summed over threads (parse, normalize + tessellate, doc-value
/// encoding with its header) against the shards' indexing time (the
/// whole per-doc operation: JSON parse, mapping, the shape phases,
/// memtable insert, translog share). Then a force merge and the store.
/// Rally geoshape's `index-append-polygons`: the first `n` docs of a Rally
/// corpus file (`GEO_INGEST_FILE`, one JSON doc per line; default the
/// default-polygons corpus), Rally's mapping (with its index-time `size`
/// script), `GEO_INGEST_CLIENTS` writers (8) of `GEO_INGEST_BULK`-doc bulks
/// (100). Reports docs/s, bulk latency and the shape ingest breakdown.
fn ingest_rally_section(n: usize, t: &mut Table) {
    use std::io::BufRead;
    let path = std::env::var("GEO_INGEST_FILE").unwrap_or_else(|_| format!("{}/.rally/benchmarks/data/default-polygons/default-polygons.json", std::env::var("HOME").unwrap_or_default()));
    let Ok(f) = std::fs::File::open(&path) else {
        t.add("rally ingest", 0.0, format!("no corpus at {path}"), false, "corpus", None);
        return;
    };
    let docs: Vec<String> = std::io::BufReader::new(f).lines().take(n).map(|l| l.unwrap()).collect();
    let n = docs.len();
    let clients = env("GEO_INGEST_CLIENTS", 8);
    let bulk = env("GEO_INGEST_BULK", 100);
    // `GEO_INGEST_SYNC=1`: each bulk fsyncs its translog before it returns
    // (`index.translog.durability: request`, as the REST bulk does), in
    // `GEO_INGEST_DIR` (default the temp dir) so the fsyncs hit that disk.
    let sync = env("GEO_INGEST_SYNC", 0) != 0;
    let tmp = match std::env::var("GEO_INGEST_DIR") { Ok(d) => tempfile::tempdir_in(d).unwrap(), Err(_) => tempfile::tempdir().unwrap() };
    let node = Node::open(tmp.path()).unwrap();
    let mapping = json!({"dynamic": "strict", "properties": {"shape": {"type": "geo_shape"}, "size": {"type": "double", "script": {"source": "ScriptDocValues.Geometry geometry = doc[\"shape\"];double w = geometry.getMercatorWidth();double h = geometry.getMercatorHeight();emit(h * h + w * w);"}}}});
    node.create_index("osmpolygons", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    let svc = node.get_index("osmpolygons").unwrap();
    crate::fieldtypes::index::shape_ingest_stats::take();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let lat = std::sync::Mutex::new(Vec::new());
    let failed = std::sync::atomic::AtomicUsize::new(0);
    let started = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..clients {
            sc.spawn(|| loop {
                let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let from = b * bulk;
                if from >= n {
                    break;
                }
                let reqs: Vec<WriteReq> = docs[from..(from + bulk).min(n)].iter().map(|d| WriteReq::Index { id: None, source: d.clone(), routing: None, cond: Default::default() }).collect();
                let tb = Instant::now();
                let bad = svc.write(reqs, sync).iter().filter(|r| r.is_err()).count();
                failed.fetch_add(bad, std::sync::atomic::Ordering::Relaxed);
                lat.lock().unwrap().push(tb.elapsed().as_secs_f64() * 1e3);
            });
        }
    });
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let [parse, tess, enc] = crate::fieldtypes::index::shape_ingest_stats::take();
    let mut lat = lat.into_inner().unwrap();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat.get(((lat.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0);
    let ms = |ns: u64| ns as f64 / 1e6;
    let failed = failed.into_inner();
    eprintln!("rally ingest: {n} docs, {clients} clients x {bulk}: {:.0} docs/s, bulk p50 {:.2} p90 {:.2} p99 {:.2} ms; parse {:.0} ms, tessellate {:.0} ms, encode {:.0} ms (summed over threads); {failed} failed", n as f64 / (wall / 1e3), pct(0.5), pct(0.9), pct(0.99), ms(parse), ms(tess), ms(enc));
    t.add("rally ingest", wall, format!("{:.0} docs/s, bulk p50 {:.2} ms", n as f64 / (wall / 1e3), pct(0.5)), failed == 0, "no failures", None);
}

fn ingest_section(n: usize, t: &mut Table) {
    for (label, is_kind) in [("polygons", (|w: &str| w.starts_with("POLYGON")) as fn(&str) -> bool), ("linestrings", (|w: &str| w.starts_with("LINESTRING") || w.starts_with("MULTILINESTRING")) as fn(&str) -> bool)] {
        let task = format!("ingest {label} (100-doc bulks)");
        if !t.want(&task) {
            continue;
        }
        // The shapes first, so generation is not timed.
        let mut rng = Lcg(77);
        let mut shapes = Vec::with_capacity(n);
        let mut i = 0usize;
        while shapes.len() < n {
            let w = shape_wkt(&mut rng, i);
            i += 1;
            if is_kind(&w) {
                shapes.push(w);
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let name = format!("ingest_{label}");
        node.create_index(&name, &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"shape": {"type": "geo_shape"}, "size": {"type": "long"}}}})).unwrap();
        let svc = node.get_index(&name).unwrap();
        crate::fieldtypes::index::shape_ingest_stats::take();
        let shard = svc.local_shards()[0].clone();
        let index_before = shard.shard.counters.index_time_nanos.load(std::sync::atomic::Ordering::Relaxed);
        let mut lat = Vec::with_capacity(n / 100 + 1);
        let started = Instant::now();
        let mut failed = 0usize;
        for (b, chunk) in shapes.chunks(100).enumerate() {
            let reqs: Vec<WriteReq> = chunk.iter().enumerate().map(|(k, w)| WriteReq::Index { id: Some((b * 100 + k).to_string()), source: json!({"shape": w, "size": w.len()}).to_string(), routing: None, cond: Default::default() }).collect();
            let tb = Instant::now();
            failed += svc.write(reqs, false).iter().filter(|r| r.is_err()).count();
            lat.push(tb.elapsed().as_secs_f64() * 1e3);
        }
        let wall = started.elapsed().as_secs_f64() * 1e3;
        let index_ns = shard.shard.counters.index_time_nanos.load(std::sync::atomic::Ordering::Relaxed) - index_before;
        let [parse, tess, enc] = crate::fieldtypes::index::shape_ingest_stats::take();
        lat.sort_by(f64::total_cmp);
        let pct = |p: f64| lat.get(((lat.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0);
        let ms = |ns: u64| ns as f64 / 1e6;
        t.add(
            &task,
            wall,
            format!("{:.0} docs/s, bulk p50 {:.2} p90 {:.2} ms", n as f64 / (wall / 1e3), pct(0.5), pct(0.9)),
            failed == 0,
            format!(
                "indexing time {:.0} ms: shape parse {:.0}, tessellate {:.0}, encode {:.0}, rest {:.0} (JSON, mapping, memtable, translog)",
                ms(index_ns), ms(parse), ms(tess), ms(enc), ms(index_ns.saturating_sub(parse + tess + enc))
            ),
            Some(if label == "polygons" { 5.7 } else { 6.2 }),
        );
        ingest_rows(t, &format!("geoshape {label}"), &svc, n as u64, wall, "shape");
    }
}

// ---------------------------------------------------------------------------
// A development loop over the shape structures alone (no engine): the
// suite's shapes tessellated, written as doc values and one triangle tree,
// then the queries, header reads and grid tilers timed on them, each
// query checked against the per-doc reference.
//
// `GEO_DEV_SHAPES` (default 400,000), `GEO_DEV_RUNS` (default 3),
// `GEO_DEV_ONLY` (comma list of row prefixes).

#[test]
#[ignore]
fn geo_dev_bench() {
    use rustic_core::fieldtypes::shape_bkd::{build_parts, Entry, LeafBuf, Matcher, ShapeBkd};
    use rustic_core::fieldtypes::shape_dv::ShapeValue;
    use shape::SpatialRelation as R;
    let n = env("GEO_DEV_SHAPES", 400_000);
    let runs = env("GEO_DEV_RUNS", 3);
    let only: Vec<String> = std::env::var("GEO_DEV_ONLY").map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    let want = |name: &str| only.is_empty() || only.iter().any(|o| name.starts_with(o.as_str()));
    let opts = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
    let mut rng = Lcg(2024);
    let wkts: Vec<String> = (0..n).map(|i| shape_wkt(&mut rng, i)).collect();
    let geoms: Vec<shape::Geometry> = wkts.iter().map(|w| shape::parse(&json!(w), &opts).unwrap()).collect();
    let t0 = Instant::now();
    let tris: Vec<Vec<[i32; 7]>> = geoms
        .iter()
        .map(|g| {
            let mut t = Vec::new();
            rustic_core::fieldtypes::tessellator::geometry_triangles(&shape::normalized(g), &mut t).unwrap();
            t
        })
        .collect();
    let tess_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let dvs: Vec<Vec<u8>> = geoms
        .iter()
        .zip(&tris)
        .map(|(g, t)| {
            let mut c = shape::CentroidCalc::default();
            c.add(g);
            rustic_core::fieldtypes::shape_field::doc_value(t, &c).unwrap()
        })
        .collect();
    let dv_ms = t0.elapsed().as_secs_f64() * 1e3;
    let total_tris: usize = tris.iter().map(Vec::len).sum();
    let mut entries: Vec<Entry> = Vec::with_capacity(total_tris);
    for (d, ts) in tris.iter().enumerate() {
        entries.extend(ts.iter().map(|t| Entry { tri: *t, doc: d as u32 }));
    }
    let t0 = Instant::now();
    let parts = build_parts(&mut entries);
    let tree_ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(entries);
    let tree_bytes: usize = parts.iter().map(Vec::len).sum();
    let dv_bytes: usize = dvs.iter().map(Vec::len).sum();
    let bkds: Vec<ShapeBkd> = parts.into_iter().map(|p| ShapeBkd::open(rustic_core::codec::directory::FileSlice::from_vec(p)).unwrap()).collect();
    eprintln!("dev: {n} shapes, {total_tris} triangles ({:.1}/doc); tessellate {tess_ms:.0} ms, doc values {dv_ms:.0} ms, tree {tree_ms:.0} ms; tree {:.1} B/tri ({:.0} B/doc), doc values {:.1} B/tri ({:.0} B/doc)", total_tris as f64 / n as f64, tree_bytes as f64 / total_tris as f64, tree_bytes as f64 / n as f64, dv_bytes as f64 / total_tris as f64, dv_bytes as f64 / n as f64);
    let best_of = |f: &mut dyn FnMut() -> u64| -> (u64, f64) {
        let mut out = f();
        let mut ms = f64::INFINITY;
        for _ in 0..runs.max(1) {
            let t = Instant::now();
            out = f();
            ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (out, ms)
    };
    let rally_poly = json!({"type": "polygon", "coordinates": [[[-0.1, 49.0], [5.0, 48.0], [15.0, 49.0], [14.0, 60.0], [-0.1, 61.0], [-0.1, 49.0]]]});
    let oresund = json!({"type": "Polygon", "coordinates": [[[12.5706, 55.45074], [12.04631, 55.70469], [12.06618, 55.7991], [12.68098, 55.93474], [13.48894, 55.88091], [13.60262, 55.68478], [13.44479, 55.32661], [12.64235, 55.32723], [12.5706, 55.45074]]]});
    let bbox = json!({"type": "envelope", "coordinates": [[-0.1, 61.0], [15.0, 48.0]]});
    let paris = json!({"type": "envelope", "coordinates": [[2.20, 48.95], [2.40, 48.75]]});
    for (name, q, rel) in [
        ("poly intersects", &rally_poly, R::Intersects),
        ("poly contains", &rally_poly, R::Contains),
        ("poly within", &rally_poly, R::Within),
        ("poly disjoint", &rally_poly, R::Disjoint),
        ("oresund intersects", &oresund, R::Intersects),
        ("bbox intersects", &bbox, R::Intersects),
        ("bbox within", &bbox, R::Within),
        ("bbox disjoint", &bbox, R::Disjoint),
        ("paris intersects", &paris, R::Intersects),
    ] {
        if !want(name) {
            continue;
        }
        let sq = ref_query(q, rel);
        let count = |sq: &rustic_core::fieldtypes::geo2d::ShapeQuery| -> Vec<u64> {
            use rustic_core::fieldtypes::geo2d::{ComponentLeaf, LeafQuery, RectLeaf, ShapeQuery};
            fn run(sq: &ShapeQuery, bkds: &[ShapeBkd], n: u32) -> Vec<u64> {
                let leaf: Box<dyn LeafQuery + '_> = match sq {
                    ShapeQuery::None => return vec![0; (n as usize).div_ceil(64)],
                    ShapeQuery::And(qs) => {
                        let mut acc: Option<Vec<u64>> = None;
                        for q in qs {
                            let b = run(q, bkds, n);
                            acc = Some(match acc {
                                None => b,
                                Some(mut a) => {
                                    a.iter_mut().zip(&b).for_each(|(x, y)| *x &= y);
                                    a
                                }
                            });
                        }
                        return acc.unwrap_or_default();
                    }
                    ShapeQuery::Component(c, r) => Box::new(ComponentLeaf { component: c, rel: *r }),
                    ShapeQuery::Rect(e, _, r) => Box::new(RectLeaf { rect: e, rel: *r }),
                };
                let mut m = Matcher::new(leaf.as_ref(), n);
                let mut buf = LeafBuf::default();
                for b in bkds {
                    m.part(b, &mut buf);
                }
                m.finish().unwrap()
            }
            run(sq, &bkds, n as u32)
        };
        let (c, ms) = best_of(&mut || count(&sq).iter().map(|w| w.count_ones() as u64).sum());
        let bits = count(&sq);
        let bad = (0..n).filter(|&d| (bits[d / 64] >> (d % 64) & 1 == 1) != ref_matches(&sq, &tris[d])).count();
        eprintln!("  {name:28} {ms:9.2} ms  {c} docs  {}", if bad == 0 { "ok".to_string() } else { format!("FAIL {bad} docs differ") });
    }
    if want("header") {
        let (c, ms) = best_of(&mut || {
            let mut s = 0.0;
            for d in &dvs {
                let v = ShapeValue::new(d);
                s += v.centroid().0 + v.weight();
            }
            s as u64
        });
        eprintln!("  {:28} {ms:9.2} ms  {c}", "header centroid");
        let (c, ms) = best_of(&mut || {
            let mut s = 0.0;
            for d in &dvs {
                if let Some(b) = ShapeValue::new(d).bounding_box() {
                    s += b.top;
                }
            }
            s as u64
        });
        eprintln!("  {:28} {ms:9.2} ms  {c}", "header bounds");
    }
    // Grid cells of the docs whose boxes meet the Paris box, as the
    // suite's bounded aggs.
    let pb = crate::aggs::shape_tiler::Bounds { top: 48.95, left: 2.20, bottom: 48.75, right: 2.40 };
    let in_paris: Vec<&Vec<u8>> = dvs.iter().filter(|d| ShapeValue::new(d).bounding_box().is_some_and(|b| b.max_x() >= pb.left && b.min_x() <= pb.right && b.max_y() >= pb.bottom && b.min_y() <= pb.top)).collect();
    for (name, tiler) in [
        ("grid geohash 5", crate::aggs::shape_tiler::Tiler::Hash(crate::aggs::shape_tiler::HashTiler::new(5, Some(&pb)))),
        ("grid geotile 13", crate::aggs::shape_tiler::Tiler::Tile(crate::aggs::shape_tiler::TileTiler::new(13, Some(&pb)))),
        ("grid geohex 6", crate::aggs::shape_tiler::Tiler::Hex(crate::aggs::shape_tiler::HexTiler::new(6, Some(&pb)))),
    ] {
        if !want(name) {
            continue;
        }
        let mut out = Vec::new();
        let (c, ms) = best_of(&mut || {
            let mut k = 0u64;
            for d in &in_paris {
                tiler.cells(&ShapeValue::new(d), &mut out);
                k += out.len() as u64;
            }
            k
        });
        let mut bad = 0;
        for d in &in_paris {
            let v = ShapeValue::new(d);
            tiler.cells(&v, &mut out);
            let got: std::collections::BTreeSet<String> = out.iter().map(|&h| tiler.key(h)).collect();
            let want = match name {
                "grid geohex 6" => crate::aggs::shape_tiler::brute_force_hex_cells(6, Some(&pb), &v),
                _ => {
                    let mut w = std::collections::BTreeSet::new();
                    let (g, p) = if name == "grid geohash 5" { ("geohash_grid", 5) } else { ("geotile_grid", 13) };
                    crate::aggs::shape_tiler::brute_force_cells(g, p, Some(&pb), &v, &mut w);
                    w
                }
            };
            if got != want {
                if bad < 3 {
                    eprintln!("    differs: got {got:?} want {want:?}");
                }
                bad += 1;
            }
        }
        eprintln!("  {name:28} {ms:9.2} ms  {c} doc-cells over {} docs  {}", in_paris.len(), if bad == 0 { "ok".to_string() } else { format!("FAIL {bad} docs differ") });
        if std::env::var("GEO_DEV_SLOW").is_ok() {
            let mut per: Vec<(f64, usize, usize, usize)> = in_paris
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let t = Instant::now();
                    tiler.cells(&ShapeValue::new(d), &mut out);
                    (t.elapsed().as_secs_f64() * 1e3, i, out.len(), d.len())
                })
                .collect();
            per.sort_by(|a, b| b.0.total_cmp(&a.0));
            let tot: f64 = per.iter().map(|p| p.0).sum();
            for p in per.iter().take(8) {
                eprintln!("    slow doc {}: {:.2} ms, {} cells, {} value bytes", p.1, p.0, p.2, p.3);
            }
            let top: f64 = per.iter().take(per.len() / 100).map(|p| p.0).sum();
            let st = &crate::aggs::shape_tiler::HEX_STATS;
            eprintln!("    total {tot:.1} ms, slowest 1% {top:.1} ms; hex relates {}, by recursion {}", st[0].swap(0, std::sync::atomic::Ordering::Relaxed), st[1].swap(0, std::sync::atomic::Ordering::Relaxed));
        }
    }
}
