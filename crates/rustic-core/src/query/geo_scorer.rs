//! Scorers over a point predicate on one geo field of a sealed segment
//! (Lucene's `LatLonPoint` queries wrapped in an `IndexOrDocValuesQuery`).
//!
//! The usual path walks the segment's KD tree ([`GeoIndex::pred_hits`]):
//! leaves outside the predicate are skipped, leaves inside it are taken
//! whole, only boundary leaves test their points; large trees are walked on
//! a few threads.
//!
//! When the search will stop after a few matches (a uniformly scoring
//! query with a bounded `track_total_hits`, e.g. ES's default 10,000, and
//! a small `size`), [`Searcher`](crate::query::Searcher) says so through
//! [`with_wanted`] while it builds the leaf's scorer. A dense predicate
//! then tests the doc-sorted points in doc order instead (Lucene's
//! doc-values query for a small lead cost), stopping as soon as the
//! collector stops. If the matches turn out sparser than the KD leaf boxes
//! promised, it finishes with the KD walk from where it stopped. Either way
//! it yields exactly the docs the KD walk yields, ascending, with the same
//! score: only the cost differs.

use crate::geo::{decode_lat, decode_lon, DocHits, GeoIndex, GeoPred, GeoValues};
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::query::hits_scorer;
use crate::query::scorer::Scorer;
use crate::types::DocId;
use std::cell::Cell;

thread_local! {
    /// Matches the search building a scorer on this thread stops after (0:
    /// it needs them all).
    static WANTED: Cell<u64> = const { Cell::new(0) };
}

/// Run `f` (one leaf's top-level scorer construction) telling the geo
/// predicate scorer it builds that the caller stops after about `wanted`
/// matches. The first geo scorer built takes the hint.
pub fn with_wanted<R>(wanted: u64, f: impl FnOnce() -> R) -> R {
    let prev = WANTED.with(|c| c.replace(wanted));
    let r = f();
    WANTED.with(|c| c.set(prev));
    r
}

/// Take the hint of [`with_wanted`] (0 when there is none).
fn take_wanted() -> u64 {
    WANTED.with(|c| c.replace(0))
}

/// [`take_wanted`] for geo scorers built outside this module (a
/// `geo_shape` field's triangle index): the matches the search stops
/// after, 0 when it needs them all. Takes the hint.
pub fn take_hint() -> u64 {
    take_wanted()
}

/// Fewest points the doc-order scan tests before it may give up on a
/// predicate sparser than its estimate.
const MIN_BUDGET: usize = 1 << 16;

/// The docs of `g` passing `pred`, scoring `score`. `any_point`: the doc
/// matches when any of its points passes (`pred` must mean that, e.g. a
/// plain shape or `intersects`); otherwise `pred` is a per-doc predicate,
/// valid only on single-valued fields, and `None` is returned when `g` has
/// a doc with several points (callers then use their per-doc path).
pub fn geo_pred_scorer(g: &GeoIndex, pred: GeoPred, any_point: bool, num_docs: u32, score: f32) -> Option<Box<dyn Scorer>> {
    if !any_point && !g.single_valued() {
        return None;
    }
    let wanted = take_wanted();
    if wanted > 0 {
        if let Some(s) = LazyGeoScorer::try_new(g, &pred, any_point, num_docs, score, wanted) {
            return Some(Box::new(s));
        }
    }
    Some(hits_scorer(walk(g, &pred, any_point, num_docs), score))
}

fn walk(g: &GeoIndex, pred: &GeoPred, any_point: bool, num_docs: u32) -> DocHits {
    if any_point {
        g.any_point_hits(pred, num_docs)
    } else {
        // Checked single-valued by the caller.
        g.pred_hits(pred, num_docs).unwrap_or(DocHits::Sorted(Vec::new()))
    }
}

/// Tests the doc-sorted points in order; switches to the KD walk after
/// `budget` tests (see the module docs).
struct LazyGeoScorer {
    /// The segment's index (the scorer never outlives the search holding
    /// the segment: the contract every leaf-bound scorer relies on).
    g: &'static GeoIndex,
    points: &'static GeoValues,
    pred: GeoPred,
    any_point: bool,
    num_docs: u32,
    score: f32,
    /// Next point to test.
    pos: usize,
    /// Tests left before switching to the KD walk.
    budget: usize,
    /// Last doc returned (`-1`: none yet).
    last: i64,
    /// Current doc (`TERMINATED` before the first advance and at the end).
    cur: DocId,
    /// The smallest doc the next match may be (a seek skips untested docs).
    min_next: DocId,
    /// The KD walk's matches after the switch.
    tail: Option<Box<dyn Scorer>>,
    estimate: u64,
    /// Matches the search stops after (the hint).
    wanted: u64,
    /// Matches returned and points tested by the scan so far.
    found: u64,
    tested: u64,
}

impl LazyGeoScorer {
    /// `None` when the predicate looks too sparse for a doc-order scan to
    /// find `wanted` matches before the KD walk would have finished.
    fn try_new(g: &GeoIndex, pred: &GeoPred, any_point: bool, num_docs: u32, score: f32, wanted: u64) -> Option<Self> {
        let n = g.values().len();
        if n == 0 {
            return None;
        }
        // Points in the KD leaves the predicate does not rule out: an upper
        // bound of the matches, and about what the KD walk would touch.
        let estimate = g.pred_estimate(pred);
        if estimate == 0 {
            return None;
        }
        // Points a doc-order scan tests to find `wanted` matches if they
        // are spread evenly over the docs.
        let expect = wanted as f64 * n as f64 / estimate as f64;
        if expect * 2.0 >= estimate as f64 {
            return None;
        }
        let budget = ((expect * 2.0) as usize).max(MIN_BUDGET).min(n);
        // SAFETY: see the `g` field.
        let g: &'static GeoIndex = unsafe { std::mem::transmute::<&GeoIndex, &'static GeoIndex>(g) };
        Some(LazyGeoScorer { g, points: g.values(), pred: pred.clone(), any_point, num_docs, score, pos: 0, budget, last: -1, cur: TERMINATED, min_next: 0, tail: None, estimate, wanted, found: 0, tested: 0 })
    }

    /// More tests for a scan whose budget ran out, when the hit rate it
    /// observed projects the matches still wanted (twice over) within
    /// fewer tests than the KD walk would touch (the estimate): the budget
    /// was sized from the estimate, which overstates the matches when
    /// leaf boxes overlap the predicate loosely. `None`: switch to the
    /// walk (nothing found yet, or the search wants more than the hint).
    fn extension(&self) -> Option<usize> {
        if self.found == 0 || self.found >= self.wanted {
            return None;
        }
        let need = (self.wanted - self.found) as f64;
        let more = (need * self.tested as f64 / self.found as f64 * 2.0).ceil();
        let left = (self.points.len() - self.pos) as f64;
        (more < self.estimate as f64).then_some(more.min(left).max(1024.0) as usize)
    }

    /// The next match from `self.pos` on.
    fn scan(&mut self) -> DocId {
        while self.pos < self.points.len() {
            if self.budget == 0 {
                match self.extension() {
                    Some(more) => self.budget = more,
                    None => return self.switch(),
                }
            }
            let p = self.points.entry(self.pos);
            self.pos += 1;
            self.budget -= 1;
            self.tested += 1;
            let d = { p.doc };
            // Further points of a doc already returned.
            if (d as i64) <= self.last {
                continue;
            }
            if self.pred.test(decode_lat({ p.lat }), decode_lon({ p.lon })) {
                self.last = d as i64;
                self.cur = d;
                self.found += 1;
                return d;
            }
        }
        self.cur = TERMINATED;
        TERMINATED
    }

    /// Continue with the KD walk's matches after the last doc returned:
    /// every doc between it and `pos` was tested and failed, so the walk
    /// (which matches exactly the same docs) has none there either.
    fn switch(&mut self) -> DocId {
        let mut tail = hits_scorer(walk(self.g, &self.pred, self.any_point, self.num_docs), self.score);
        let target = ((self.last + 1) as DocId).max(self.min_next);
        let mut d = tail.advance();
        if d != TERMINATED && d < target {
            d = tail.seek(target);
        }
        self.tail = Some(tail);
        self.cur = d;
        if d != TERMINATED {
            self.last = d as i64;
        }
        d
    }
}

impl DocSet for LazyGeoScorer {
    fn advance(&mut self) -> DocId {
        if let Some(t) = &mut self.tail {
            self.cur = t.advance();
            return self.cur;
        }
        self.scan()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.cur != TERMINATED && self.cur >= target {
            return self.cur;
        }
        if let Some(t) = &mut self.tail {
            self.cur = t.seek(target);
            return self.cur;
        }
        if target == TERMINATED {
            self.cur = TERMINATED;
            self.pos = self.points.len();
            return TERMINATED;
        }
        // Skip the points of earlier docs (they are doc-sorted).
        self.min_next = self.min_next.max(target);
        self.pos = self.points.lower_bound(target, self.pos);
        self.scan()
    }

    fn doc(&self) -> DocId {
        self.cur
    }

    fn size_hint(&self) -> u32 {
        self.estimate.min(u32::MAX as u64) as u32
    }
}

impl Scorer for LazyGeoScorer {
    fn score(&mut self) -> f32 {
        self.score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{encode_lat, encode_lon, GeoPointEntry, GeoShape, PointPredicate, Relation};
    use std::sync::Arc;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn drain(s: &mut dyn Scorer) -> Vec<DocId> {
        let mut out = Vec::new();
        loop {
            let d = s.advance();
            if d == TERMINATED {
                return out;
            }
            out.push(d);
        }
    }

    /// Points clustered by doc range (as a spatially sorted corpus is), a
    /// few docs without points, every tenth doc with two points when
    /// `multi`; poles and the dateline included.
    fn index(n: u32, multi: bool, seed: u64) -> GeoIndex {
        let mut r = Lcg(seed);
        let mut pts = Vec::new();
        for d in 0..n {
            if d % 97 == 13 {
                continue;
            }
            let k = if multi && d % 10 == 0 { 2 } else { 1 };
            for _ in 0..k {
                let (lat, lon) = match (d / (n / 8).max(1)) % 4 {
                    0 => (48.0 + 13.0 * r.next(), -0.5 + 16.0 * r.next()),
                    1 => (-90.0 + 180.0 * r.next(), -180.0 + 360.0 * r.next()),
                    2 => (85.0 + 5.0 * r.next(), 170.0 + 20.0 * r.next() - if r.next() < 0.5 { 360.0 } else { 0.0 }),
                    _ => (-90.0 + 3.0 * r.next(), -180.0 + 360.0 * r.next()),
                };
                let lon = if lon > 180.0 { lon - 360.0 } else { lon };
                pts.push(GeoPointEntry { doc: d, lat: encode_lat(lat), lon: encode_lon(lon) });
            }
        }
        GeoIndex::build(pts)
    }

    fn preds() -> Vec<GeoPred> {
        let leaf = |s: GeoShape| GeoPred::Leaf(Arc::new(s));
        vec![
            leaf(GeoShape::Polygon { points: vec![(49.0, -0.1), (48.0, 5.0), (49.0, 15.0), (60.0, 14.0), (61.0, -0.1), (49.0, -0.1)] }),
            leaf(GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 400_000.0 }),
            leaf(GeoShape::BoundingBox { top: 90.0, left: 170.0, bottom: 80.0, right: -170.0 }),
            leaf(GeoShape::Distance { lat: -90.0, lon: 0.0, meters: 300_000.0 }),
            GeoPred::And(vec![GeoPred::All, GeoPred::Not(Box::new(leaf(GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 400_000.0 })))]),
            GeoPred::And(vec![leaf(GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 200_000.0 }), GeoPred::Not(Box::new(leaf(GeoShape::Distance { lat: 55.0, lon: 7.0, meters: 100_000.0 })))]),
        ]
    }

    fn brute(g: &GeoIndex, pred: &GeoPred) -> Vec<DocId> {
        let mut v: Vec<DocId> = g.values().iter().filter(|p| pred.test(decode_lat({ p.lat }), decode_lon({ p.lon }))).map(|p| { p.doc }).collect();
        v.dedup();
        v
    }

    /// With and without the hint, through advances and seeks, with tiny
    /// and generous budgets (the switch to the KD walk at every point of
    /// the scan): the same docs as testing every point.
    #[test]
    fn lazy_scan_matches_kd_walk_and_brute_force() {
        for multi in [false, true] {
            let g = index(40_000, multi, 9);
            for pred in preds() {
                let any = matches!(pred, GeoPred::Leaf(_));
                if !any && multi {
                    assert!(geo_pred_scorer(&g, pred.clone(), false, 40_000, 1.0).is_none());
                    continue;
                }
                let want = brute(&g, &pred);
                let eager = drain(&mut *geo_pred_scorer(&g, pred.clone(), any, 40_000, 1.0).unwrap());
                assert_eq!(eager, want, "{pred:?} multi={multi}");
                for wanted in [1u64, 11, 10_001] {
                    let mut s = with_wanted(wanted, || geo_pred_scorer(&g, pred.clone(), any, 40_000, 0.5).unwrap());
                    assert_eq!(drain(&mut *s), want, "{pred:?} wanted={wanted}");
                }
                for budget in [0usize, 1, 7, 1000, 39_000] {
                    let mut s = LazyGeoScorer { budget, ..LazyGeoScorer::try_new(&g, &pred, any, 40_000, 2.0, 1).unwrap_or_else(|| forced(&g, &pred, any)) };
                    assert_eq!(drain(&mut s), want, "{pred:?} budget={budget}");
                    // An extended budget (a large hint, the walk's
                    // estimate generous): the same docs.
                    let mut s = LazyGeoScorer { budget, wanted: 10_001, estimate: u64::MAX, ..forced(&g, &pred, any) };
                    assert_eq!(drain(&mut s), want, "{pred:?} budget={budget} extended");
                    let mut s = LazyGeoScorer { budget, wanted: 10_001, estimate: 2_000, ..forced(&g, &pred, any) };
                    assert_eq!(drain(&mut s), want, "{pred:?} budget={budget} extended then walked");
                    let mut s = LazyGeoScorer { budget, ..forced(&g, &pred, any) };
                    let mut got = Vec::new();
                    let mut target = 0;
                    loop {
                        let d = s.seek(target);
                        if d == TERMINATED {
                            break;
                        }
                        assert_eq!(s.score(), 2.0);
                        got.push(d);
                        target = d + 1 + (d % 5);
                    }
                    let mut expect = Vec::new();
                    let mut target = 0;
                    for &d in &want {
                        if d >= target {
                            expect.push(d);
                            target = d + 1 + (d % 5);
                        }
                    }
                    assert_eq!(got, expect, "{pred:?} budget={budget} seeks");
                }
            }
        }
    }

    /// A lazy scorer whatever the estimate says.
    fn forced(g: &GeoIndex, pred: &GeoPred, any: bool) -> LazyGeoScorer {
        let g: &'static GeoIndex = unsafe { std::mem::transmute::<&GeoIndex, &'static GeoIndex>(g) };
        LazyGeoScorer { g, points: g.values(), pred: pred.clone(), any_point: any, num_docs: 40_000, score: 2.0, pos: 0, budget: 0, last: -1, cur: TERMINATED, min_next: 0, tail: None, estimate: 0, wanted: 0, found: 0, tested: 0 }
    }

    /// The hint is taken by the first scorer only, and never outlives its
    /// call; a sparse predicate keeps the KD walk.
    #[test]
    fn hint_is_scoped_and_sparse_predicates_walk() {
        let g = index(20_000, false, 3);
        let dense = preds()[1].clone();
        with_wanted(10, || {
            assert!(LazyGeoScorer::try_new(&g, &dense, true, 20_000, 1.0, 10).is_some());
            let _ = geo_pred_scorer(&g, dense.clone(), true, 20_000, 1.0);
            assert_eq!(take_wanted(), 0);
        });
        assert_eq!(take_wanted(), 0);
        #[derive(Debug)]
        struct Nothing;
        impl PointPredicate for Nothing {
            fn classify(&self, _: f64, _: f64, _: f64, _: f64) -> Relation {
                Relation::Outside
            }
            fn test(&self, _: f64, _: f64) -> bool {
                false
            }
        }
        assert!(LazyGeoScorer::try_new(&g, &GeoPred::Leaf(Arc::new(Nothing)), true, 20_000, 1.0, 10).is_none());
    }

}

#[cfg(test)]
mod search_tests {
    use crate::codec::directory::RamDirectory;
    use crate::geo::GeoShape;
    use crate::query::boolean::MinShouldMatch;
    use crate::query::{LeafReader, Query, SearchOptions, Searcher, TotalHitsRelation, TrackTotalHits};
    use crate::segment::{Segment, SegmentWriter};
    use crate::types::{DocId, Document, Field, FieldType};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Per doc its points: most around (55, 7) out to ~500 km, some on the
    /// rings' boundaries, some far away (poles, dateline), a few without
    /// a point, every 9th with two points when `multi`.
    fn docs(n: usize, multi: bool, seed: u64) -> Vec<Vec<(f64, f64)>> {
        let mut r = Lcg(seed);
        (0..n)
            .map(|i| {
                if i % 101 == 7 {
                    return Vec::new();
                }
                let k = if multi && i % 9 == 0 { 2 } else { 1 };
                (0..k)
                    .map(|_| match (r.next() * 10.0) as u32 {
                        0 => (89.0 + r.next(), -180.0 + 360.0 * r.next()),
                        1 => (-10.0 + 20.0 * r.next(), if r.next() < 0.5 { 179.5 + 0.5 * r.next() } else { -180.0 + 0.5 * r.next() }),
                        _ => (55.0 + 9.0 * (r.next() - 0.5), 7.0 + 15.0 * (r.next() - 0.5)),
                    })
                    .collect()
            })
            .collect()
    }

    fn segment(docs: &[Vec<(f64, f64)>]) -> Segment {
        let schema = vec![Field::new(0, "g", FieldType::GeoPoint)];
        let mut w = SegmentWriter::new(0);
        for pts in docs {
            let mut d = Document::new();
            for &(lat, lon) in pts {
                d.add_geo_point(0, lat, lon);
            }
            w.add_document(&d, &schema);
        }
        w.write_to(&RamDirectory::new()).unwrap()
    }

    fn zero(q: Query) -> Query {
        Query::ConstScore { query: Box::new(q), score: 0.0, point_set: false }
    }

    fn dist(m: f64) -> Query {
        Query::GeoDistance { field: 0, lat: 55.0, lon: 7.0, meters: m }
    }

    /// ES's `geo_shape` circle `disjoint` on a `geo_point` (see the engine's
    /// `shape_query`): a point, none within the radius.
    fn disjoint(m: f64) -> Query {
        Query::Bool { must: vec![], should: vec![], minimum_should_match: MinShouldMatch::Count(0), filter: vec![Query::GeoBoundingBox { field: 0, top: 90.0, bottom: -90.0, left: -180.0, right: 180.0 }], must_not: vec![dist(m)] }
    }

    fn ring(inner: f64, outer: f64) -> Query {
        Query::Bool { must: vec![zero(dist(outer)), zero(disjoint(inner))], should: vec![], minimum_should_match: MinShouldMatch::Count(0), filter: vec![], must_not: vec![] }
    }

    fn within(pts: &[(f64, f64)], m: f64) -> bool {
        let c = GeoShape::Distance { lat: 55.0, lon: 7.0, meters: m };
        pts.iter().any(|&(lat, lon)| {
            let (lat, lon) = crate::geo::quantize(lat, lon);
            c.contains(lat, lon)
        })
    }

    /// Distance rings as Rally writes them (`must` of two zero-scoring
    /// `geo_shape` clauses) and plain distance filters: same hits, scores
    /// and totals as the definition, for exact and bounded totals and
    /// sizes 0 to 10,000, on single- and multi-valued fields.
    #[test]
    fn rings_and_distances_match_brute_force() {
        for multi in [false, true] {
            let ds = docs(30_000, multi, 5);
            let seg = segment(&ds);
            let leaves = vec![LeafReader::Segment(&seg)];
            let s = Searcher::new(leaves);
            let cases: Vec<(Query, Box<dyn Fn(&[(f64, f64)]) -> bool>, f32)> = vec![
                (ring(100_000.0, 200_000.0), Box::new(|p: &[(f64, f64)]| within(p, 200_000.0) && !p.is_empty() && !within(p, 100_000.0)), 0.0),
                (ring(73_204.77, 73_204.78), Box::new(|p: &[(f64, f64)]| within(p, 73_204.78) && !p.is_empty() && !within(p, 73_204.77)), 0.0),
                (zero(disjoint(400_000.0)), Box::new(|p: &[(f64, f64)]| !p.is_empty() && !within(p, 400_000.0)), 0.0),
                (dist(300_000.0), Box::new(|p: &[(f64, f64)]| within(p, 300_000.0)), 1.0),
            ];
            for (q, keep, score) in &cases {
                let want: Vec<DocId> = (0..ds.len()).filter(|&i| keep(&ds[i])).map(|i| i as DocId).collect();
                for (k, tth) in [(10, TrackTotalHits::UpTo(10_000)), (0, TrackTotalHits::UpTo(10_000)), (10, TrackTotalHits::True), (10_000, TrackTotalHits::UpTo(10_000)), (3, TrackTotalHits::UpTo(5)), (10, TrackTotalHits::False)] {
                    let td = s.search(q, &SearchOptions { top_k: k, track_total_hits: tth, ..Default::default() });
                    let got: Vec<DocId> = td.hits.iter().map(|h| h.doc).collect();
                    assert_eq!(got, want.iter().copied().take(k).collect::<Vec<_>>(), "{q:?} k={k} {tth:?} multi={multi}");
                    assert!(td.hits.iter().all(|h| h.score == *score), "{q:?}");
                    let n = want.len() as u64;
                    match tth {
                        TrackTotalHits::True => assert_eq!((td.total_hits, td.total_hits_relation), (n, TotalHitsRelation::Eq)),
                        TrackTotalHits::UpTo(m) if n > m => assert_eq!((td.total_hits, td.total_hits_relation), (m, TotalHitsRelation::Gte)),
                        TrackTotalHits::UpTo(_) => assert_eq!((td.total_hits, td.total_hits_relation), (n, TotalHitsRelation::Eq)),
                        TrackTotalHits::False => {}
                    }
                }
            }
        }
    }

    /// Several segments, dense predicates collected with a bounded total
    /// (the doc-order scan, stopping early): the first hits of the first
    /// segments and the bounded total, as with an exact total.
    #[test]
    fn early_stop_over_segments_matches_exact_collection() {
        let segs: Vec<Segment> = (0..3).map(|i| segment(&docs(12_000, i == 1, 11 + i as u64))).collect();
        let s = Searcher::new(segs.iter().map(LeafReader::Segment).collect());
        let poly = Query::GeoPolygon { field: 0, points: vec![(51.0, 1.0), (51.0, 13.0), (59.0, 13.0), (59.0, 1.0), (51.0, 1.0)] };
        for q in [zero(disjoint(100_000.0)), poly, dist(250_000.0)] {
            let exact = s.search(&q, &SearchOptions { top_k: 40_000, track_total_hits: TrackTotalHits::True, ..Default::default() });
            let all: Vec<(usize, DocId)> = exact.hits.iter().map(|h| (h.leaf, h.doc)).collect();
            for (k, limit) in [(10usize, 10_000u64), (1, 100), (0, 10_000), (500, 20_000), (10, 40_000)] {
                for parallel in [false, true] {
                    let td = s.search(&q, &SearchOptions { top_k: k, track_total_hits: TrackTotalHits::UpTo(limit), parallel, ..Default::default() });
                    let got: Vec<(usize, DocId)> = td.hits.iter().map(|h| (h.leaf, h.doc)).collect();
                    assert_eq!(got, all.iter().copied().take(k).collect::<Vec<_>>(), "{q:?} k={k} limit={limit}");
                    let n = exact.total_hits;
                    let want = if n > limit { (limit, TotalHitsRelation::Gte) } else { (n, TotalHitsRelation::Eq) };
                    assert_eq!((td.total_hits, td.total_hits_relation), want, "{q:?} k={k} limit={limit}");
                }
            }
        }
    }

    /// Two zero-scoring `must` clauses score uniformly; a `should` or a
    /// third `must` clause does not count as uniform.
    #[test]
    fn bool_uniformity() {
        assert!(ring(1.0, 2.0).scores_uniformly());
        let three = Query::Bool { must: vec![zero(dist(1.0)), zero(dist(2.0)), zero(dist(3.0))], should: vec![], minimum_should_match: MinShouldMatch::Count(0), filter: vec![], must_not: vec![] };
        assert!(!three.scores_uniformly());
        let with_should = Query::Bool { must: vec![zero(dist(1.0)), zero(dist(2.0))], should: vec![zero(dist(3.0))], minimum_should_match: MinShouldMatch::Count(0), filter: vec![], must_not: vec![] };
        assert!(!with_should.scores_uniformly());
    }
}
