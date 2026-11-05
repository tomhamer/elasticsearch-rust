//! Component experiment only: spatial bounds for compound score formulas.
//! Missing/multivalued data and script errors require product fallbacks.
//! Integer casts and wrapping abs/addition model the two Painless formulas.
use rustic_core::geo::{GeoIndex, GeoPointEntry, Relation, encode_lat, encode_lon, decode_lat, decode_lon};
use std::{cell::Cell, cmp::Ordering, collections::BinaryHeap, io::BufRead, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Hit { doc: u32, score: f32 }
impl Eq for Hit {}
impl PartialOrd for Hit { fn partial_cmp(&self, b: &Self) -> Option<Ordering> { Some(self.cmp(b)) } }
impl Ord for Hit { fn cmp(&self, b: &Self) -> Ordering { b.score.total_cmp(&self.score).then(self.doc.cmp(&b.doc)) } }
fn offer(heap: &mut BinaryHeap<Hit>, hit: Hit, k: usize) {
    if heap.len() < k { heap.push(hit); }
    else if heap.peek().is_some_and(|w| hit.score > w.score || (hit.score == w.score && hit.doc < w.doc)) {
        *heap.peek_mut().unwrap() = hit;
    }
}
fn sorted(heap: BinaryHeap<Hit>) -> Vec<Hit> {
    let mut hits = heap.into_vec();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
    hits
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn ceiling(a: f64, b: f64, c: f64, d: f64, product: bool) -> f32 {
    let (low, high) = if product {
        let corners = [a*c, a*d, b*c, b*d];
        (corners.into_iter().fold(f64::INFINITY, f64::min), corners.into_iter().fold(f64::NEG_INFINITY, f64::max))
    } else { (a+c, b+d) };
    // All POC populations are nonnegative i32 values. Use a deliberately
    // loose long-domain bound; no per-query population scan is required.
    let upper = low.abs().max((high + 64.0_f64 * std::f64::consts::LN_2).abs());
    // Conservative outward padding for these finite bounded operations.
    (upper + 1e-8) as f32
}
fn main() {
    let n = 1_000_003u32;
    for distribution in ["random", "clustered", "ties", "geonames"] {
        let mut populations = Vec::new();
        let points: Vec<_> = if distribution == "geonames" {
            let Some(path) = std::env::args().nth(1) else { continue };
            let input = std::io::BufReader::new(std::fs::File::open(path).unwrap());
            input.lines().enumerate().map(|(doc, line)| {
                let value: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
                let population = value["population"].as_i64().unwrap();
                // Keep the stored f64 exactly equivalent to the source long.
                // The static script narrows to int *before* abs/addition.
                assert!(population.unsigned_abs() <= (1u64 << 53));
                assert!((population as i32).wrapping_abs().wrapping_add(1) > 0,
                    "static formula has invalid logarithm for population={population}");
                populations.push(population as f64);
                GeoPointEntry { doc: doc.try_into().unwrap(), lat: encode_lat(value["location"][1].as_f64().unwrap()), lon: encode_lon(value["location"][0].as_f64().unwrap()) }
            }).collect()
        } else { (0..n).map(|doc| {
            let r = mix(doc as u64);
            let s = mix(r);
            let (lat, lon) = match distribution {
                "clustered" => (40.0 + (r >> 11) as f64 / (1u64 << 53) as f64, -74.0 + (s >> 11) as f64 / (1u64 << 53) as f64),
                "ties" => (45.0, 90.0),
                _ => ((r >> 11) as f64 / (1u64 << 53) as f64 * 180.0 - 90.0, (s >> 11) as f64 / (1u64 << 53) as f64 * 360.0 - 180.0),
            };
            GeoPointEntry { doc, lat: encode_lat(lat), lon: encode_lon(lon) }
        }).collect() };
        let n = points.len();
        if distribution == "geonames" { assert_eq!(n, 11_396_503); }
        let start = Instant::now();
        let index = GeoIndex::build(points.clone());
        let build_seconds = start.elapsed().as_secs_f64();
        for formula in ["expression", "painless_dynamic", "painless_static"] {
            let product = formula != "expression";
            let score = |p: &GeoPointEntry| {
                let population = if distribution == "geonames" { populations[p.doc as usize] }
                    else if distribution == "ties" { 1000.0 } else { (mix(p.doc as u64) % 2_000_000_000) as f64 };
                let log = match formula {
                    "painless_static" => ((population as i64 as i32).wrapping_abs().wrapping_add(1) as f64).ln(),
                    "painless_dynamic" => ((population as i64).wrapping_abs().wrapping_add(1) as f64).ln(),
                    _ => (population.abs() + 1.0).ln(),
                };
                let lat = decode_lat(p.lat); let lon = decode_lon(p.lon);
                (if product { log + lon * lat } else { log + lon + lat }).abs() as f32
            };
            for k in [1, 10, 100] {
              for adaptive in [false, true] {
                let start = Instant::now();
                let mut heap = BinaryHeap::new();
                for p in &points { if p.doc % 7 != 0 { offer(&mut heap, Hit { doc: p.doc, score: score(p) }, k); } }
                let expected = sorted(heap);
                let exhaustive_seconds = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let threshold = Cell::new(f32::NEG_INFINITY);
                let skipped_boxes = Cell::new(0usize);
                let fallback = Cell::new(false);
                // A small deterministic spread across the input supplies a
                // conservative lower bound on the global kth score. Seeds
                // are not inserted into the result heap (no duplicate hits).
                // Strict bound rejection still visits every tied candidate.
                if adaptive {
                    let mut seeds = BinaryHeap::new();
                    for i in 0..256usize.min(points.len()) {
                        let p = &points[i * points.len() / 256usize.min(points.len())];
                        if p.doc % 7 != 0 { offer(&mut seeds, Hit { doc: p.doc, score: score(p) }, k); }
                    }
                    if seeds.len() == k { threshold.set(seeds.peek().unwrap().score); }
                }
                let mut heap = BinaryHeap::new();
                let mut scored = 0;
                index.visit(|a,b,c,d| {
                    if fallback.get() { return Relation::Outside; }
                    if ceiling(a,b,c,d,product) < threshold.get() {
                        skipped_boxes.set(skipped_boxes.get() + 1);
                        Relation::Outside
                    } else { Relation::Crosses }
                }, |p, _| {
                    if p.doc % 7 == 0 { return; }
                    scored += 1;
                    offer(&mut heap, Hit { doc: p.doc, score: score(p) }, k);
                    if heap.len() == k { threshold.set(threshold.get().max(heap.peek().unwrap().score)); }
                    if adaptive && scored >= 8192 && skipped_boxes.get() == 0 { fallback.set(true); }
                });
                if fallback.get() {
                    heap.clear();
                    for p in &points { if p.doc % 7 != 0 { offer(&mut heap, Hit { doc: p.doc, score: score(p) }, k); } }
                }
                let found = sorted(heap);
                let bounded_seconds = start.elapsed().as_secs_f64();
                assert_eq!(expected, found, "{distribution} product={product} k={k}");
                println!("{}", serde_json::json!({"distribution":distribution,"formula":formula,"k":k,"docs":n,"spatial_scored":scored,"adaptive":adaptive,"fallback":fallback.get(),"build_seconds":build_seconds,"exhaustive_seconds":exhaustive_seconds,"bounded_seconds":bounded_seconds,"speedup":exhaustive_seconds/bounded_seconds,"exact_hits_and_scores":true,"scope":"component experiment with synthetic deletions, not server or Rally"}));
              }
            }
        }
    }
}
