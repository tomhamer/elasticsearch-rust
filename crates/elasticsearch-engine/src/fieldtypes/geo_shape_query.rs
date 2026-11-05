//! `geo_shape` queries on `geo_shape` fields, as ES 9.5.5 runs them:
//! `LatLonShape.newGeometryQuery` over the segment's triangle tree
//! (`SpatialQuery`'s walk, [`rustic_core::fieldtypes::shape_bkd`]), or,
//! for a field without one (`index: false`), the shape doc values
//! (`LatLonShapeDocValuesQuery`); memtable docs relate triangle by
//! triangle with the tree's leaf tests.

use rustic_core::fieldtypes::geo2d::{ComponentLeaf, LeafQuery, QueryRelation, RectLeaf, ShapeQuery};
use rustic_core::fieldtypes::shape_bkd::Matcher;
use rustic_core::query::{DocSet, LeafReader, Scorer, SearchContext};
use rustic_core::types::DocId;
use std::sync::Arc;

/// A `geo_shape` query on one `geo_shape` field (its core id).
#[derive(Debug)]
pub(crate) struct ShapeTreeQuery {
    pub field: u16,
    pub query: Arc<ShapeQuery>,
    /// Segments whose every match a bounded count found (INTERSECTS walks
    /// that ended short of the limit): `(segment, bits)`, for the scorer
    /// of that leaf, built right after the count within the same search.
    pub full_hits: std::sync::Mutex<Vec<(usize, Vec<u64>)>>,
    /// The segment filter-cache key of this query (see
    /// [`ShapeTreeQuery::cached_hits`]), computed on first use.
    pub cache_key: std::sync::OnceLock<rustic_core::segment::FilterCacheKey>,
    /// Segments this search has already counted as a use of the filter
    /// (Lucene's policy records one use per search, however many times the
    /// search reads the filter on a segment: a count, then hits).
    pub noted: std::sync::Mutex<Vec<usize>>,
}

/// Segment filter-cache kind of `geo_shape` filters.
const SHAPE_FILTER_KIND: u8 = 10;

/// Uses of a filter on a segment from which its matches are cached (Lucene's
/// `UsageTrackingQueryCachingPolicy.minFrequencyToCache` for a query that is
/// not one of its costly kinds, as `LatLonShapeQuery` is not).
const SHAPE_FILTER_CACHE_MIN_USES: usize = 5;

impl ShapeTreeQuery {
    fn key(&self) -> rustic_core::segment::FilterCacheKey {
        *self.cache_key.get_or_init(|| {
            use std::hash::{Hash, Hasher};
            // Two independent 64-bit hashes of the query (its relation and
            // every encoded coordinate).
            let text = format!("{:?}", self.query);
            let mut a = std::collections::hash_map::DefaultHasher::new();
            (0u8, &text).hash(&mut a);
            let mut b = std::collections::hash_map::DefaultHasher::new();
            (1u8, text.len(), &text).hash(&mut b);
            (SHAPE_FILTER_KIND, self.field, a.finish(), b.finish())
        })
    }

    /// The matches of this filter on `seg` from the segment's filter cache,
    /// as ES's node query cache serves the filter (`LRUQueryCache`:
    /// `CachingWrapperWeight.count` returning the cached count, its scorer
    /// iterating the cached docs; a `geo_shape` query is cached in size-
    /// limited searches too, measured on ES 9.5.5): built and kept once the
    /// filter has been used
    /// [`SHAPE_FILTER_CACHE_MIN_USES`] times on the segment, kept for the
    /// segment's lifetime (an immutable segment's matches never change;
    /// deleted docs are left to the caller, as the cache's are in Lucene).
    /// `None`: not (yet) cached, the caller computes.
    fn cached_hits(&self, seg: &rustic_core::segment::Segment) -> Option<std::sync::Arc<rustic_core::admission::Allocation<Vec<u64>>>> {
        let key = self.key();
        let fresh = {
            let mut noted = self.noted.lock().unwrap();
            let id = seg as *const _ as usize;
            !noted.contains(&id) && {
                noted.push(id);
                true
            }
        };
        let uses = if fresh { seg.note_filter_use(key) } else { seg.filter_uses_of(key) };
        if let Some(bits) = seg.peek_filter_bits(key) {
            return Some(bits);
        }
        if uses < SHAPE_FILTER_CACHE_MIN_USES {
            return None;
        }
        let bits = shape_hits(&LeafReader::Segment(seg), self.field, &self.query).ok()?;
        seg.cached_filter_bits(key, move || bits).ok()
    }
}

fn leaf_of(q: &ShapeQuery) -> Option<Box<dyn LeafQuery + Sync + '_>> {
    match q {
        ShapeQuery::Component(c, r) => Some(Box::new(ComponentLeaf { component: c, rel: *r })),
        ShapeQuery::Rect(e, _, r) => Some(Box::new(RectLeaf { rect: e, rel: *r })),
        _ => None,
    }
}

fn num_docs(leaf: &LeafReader<'_>) -> u32 {
    match leaf {
        LeafReader::Segment(s) => s.num_docs,
        LeafReader::Memtable(w, _) => w.num_docs(),
    }
}

/// The docs of `leaf` matching `q` on `field`, as bits.
pub(crate) fn shape_hits(leaf: &LeafReader<'_>, field: u16, q: &ShapeQuery) -> Result<Vec<u64>, String> {
    let n = num_docs(leaf);
    let words = (n as usize).div_ceil(64);
    match q {
        ShapeQuery::None => Ok(vec![0; words]),
        ShapeQuery::And(qs) => {
            let mut out: Option<Vec<u64>> = None;
            for q in qs {
                let h = shape_hits(leaf, field, q)?;
                out = Some(match out {
                    None => h,
                    Some(mut a) => {
                        for (x, y) in a.iter_mut().zip(&h) {
                            *x &= y;
                        }
                        a
                    }
                });
            }
            Ok(out.unwrap_or_else(|| vec![0; words]))
        }
        _ => {
            let lq = leaf_of(q).expect("component or box query");
            match leaf {
                LeafReader::Segment(seg) => {
                    let Some(data) = seg.shape_data(field) else { return Ok(vec![0; words]) };
                    if !data.tree.is_empty() {
                        // CONTAINS of an area over docs of one point each:
                        // a point never contains an area (Lucene's point
                        // `withinTriangle` against a polygon or a box with
                        // area is never a CANDIDATE), so nothing matches.
                        if lq.relation() == QueryRelation::Contains && data.tree.iter().all(|p| p.one_triangle_per_doc()) && query_has_area(q) {
                            return Ok(vec![0; words]);
                        }
                        // CONTAINS with doc values: a doc containing the
                        // query contains a point of it (`sample_point`), so a
                        // triangle of the doc has that point in its box; only
                        // nodes and triangles whose boxes hold the point are
                        // read (without one, every leaf crossing the query).
                        // Each candidate is tested once: its extent must cover
                        // the query's box, then the tree's own per-doc tests
                        // on its doc value.
                        if let (QueryRelation::Contains, Some(col)) = (lq.relation(), &data.values) {
                            let (c, _) = match q {
                                ShapeQuery::Component(c, r) | ShapeQuery::Rect(_, c, r) => (c, r),
                                _ => unreachable!(),
                            };
                            let mut w = ContainsWalk { lq: lq.as_ref(), col, qbox: (c.min_x(), c.max_x(), c.min_y(), c.max_y()), probe: c.sample_point(), seen: vec![0; words], hits: vec![0; words], error: None };
                            let mut buf = rustic_core::fieldtypes::shape_bkd::LeafBuf::default();
                            for p in &data.tree {
                                p.intersect(&mut w, &mut buf);
                            }
                            if let Some(e) = w.error {
                                return Err(e);
                            }
                            return Ok(w.hits);
                        }
                        return rustic_core::fieldtypes::shape_bkd::match_parts(lq.as_ref(), &data.tree, n);
                    }
                    // `index: false`: the doc values (`LatLonShapeDocValuesQuery`,
                    // the component's relations, a box as its `Rectangle2D`).
                    let Some(col) = &data.values else { return Ok(vec![0; words]) };
                    let mut out = vec![0u64; words];
                    for d in 0..n {
                        if let Some(v) = col.get(d) {
                            if value_matches(&v, q)? {
                                out[d as usize / 64] |= 1 << (d % 64);
                            }
                        }
                    }
                    Ok(out)
                }
                LeafReader::Memtable(w, _) => {
                    let mut m = Matcher::new(lq.as_ref(), n);
                    let Some(buf) = w.shape_buf(field) else { return m.finish() };
                    let mut indexed = false;
                    buf.for_each_doc(n, &mut |d, ts| {
                        indexed = true;
                        m.doc(d, ts)
                    });
                    if indexed {
                        return m.finish();
                    }
                    // `index: false`: the doc values, as on a segment.
                    let mut out = vec![0u64; words];
                    for d in 0..n {
                        if let Some(v) = buf.value(d) {
                            if value_matches(&v, q)? {
                                out[d as usize / 64] |= 1 << (d % 64);
                            }
                        }
                    }
                    Ok(out)
                }
            }
        }
    }
}

/// Whether one shape doc value matches `q` (ES's doc-value relation,
/// `GeometryDocValueReader` visited with the query's `Component2D`: a box
/// query as its `Rectangle2D`, a CONTAINS over several shapes as all of
/// them).
pub(crate) fn value_matches(v: &rustic_core::fieldtypes::shape_dv::ShapeValue<'_>, q: &ShapeQuery) -> Result<bool, String> {
    match q {
        ShapeQuery::None => Ok(false),
        ShapeQuery::And(qs) => {
            for q in qs {
                if !value_matches(v, q)? {
                    return Ok(false);
                }
            }
            Ok(!qs.is_empty())
        }
        ShapeQuery::Component(c, r) | ShapeQuery::Rect(_, c, r) => v.relate_query(c, *r),
    }
}

/// The ES|QL / runtime-field relation of geo geometries: `left` written as
/// a shape doc value (tessellated as indexed), `right` as the query ES
/// builds from a literal (normalized, quantized). `None`: no left value.
pub(crate) fn geometries_relate(left: &rustic_core::fieldtypes::shape::Geometry, right: &rustic_core::fieldtypes::shape::Geometry, rel: rustic_core::fieldtypes::shape::SpatialRelation) -> Option<bool> {
    geometry_matches(left, &literal_query(right, rel)?)
}

/// The query ES builds from a literal geometry for `rel`.
pub(crate) fn literal_query(right: &rustic_core::fieldtypes::shape::Geometry, rel: rustic_core::fieldtypes::shape::SpatialRelation) -> Option<ShapeQuery> {
    let geoms = rustic_core::fieldtypes::geo2d::query_geometries(&rustic_core::fieldtypes::shape::normalize_geo(right), false).ok()?;
    Some(ShapeQuery::new(&geoms, query_relation(rel)))
}

/// A query shape that is a circle, as Lucene's `Circle` (ES
/// `toLatLonCircle`: the centre quantized, the radius in meters).
pub(crate) fn circle_of(shape_v: &serde_json::Value) -> Option<rustic_core::fieldtypes::geo2d::LatLonGeometry> {
    if !shape_v.get("type").and_then(serde_json::Value::as_str).is_some_and(|t| t.eq_ignore_ascii_case("circle")) {
        return None;
    }
    let (lat, lon) = shape_v.get("coordinates").and_then(crate::values::parse_geo_point)?;
    let meters = shape_v.get("radius").and_then(crate::values::parse_distance)?;
    let (lat, lon) = rustic_core::geo::quantize(lat, lon);
    Some(rustic_core::fieldtypes::geo2d::LatLonGeometry::Circle { lat, lon, meters })
}

/// Whether a geometry, written as a shape doc value, matches `q`.
pub(crate) fn geometry_matches(left: &rustic_core::fieldtypes::shape::Geometry, q: &ShapeQuery) -> Option<bool> {
    let bytes = crate::aggs::geo_shape::runtime_shape_value(left)?;
    value_matches(&rustic_core::fieldtypes::shape_dv::ShapeValue::new(&bytes), q).ok()
}

/// Whether a component or box query covers an area (a polygon, or a box
/// of positive width and height), so no single point contains it.
fn query_has_area(q: &ShapeQuery) -> bool {
    match q {
        ShapeQuery::Component(c, _) | ShapeQuery::Rect(_, c, _) => c.has_area(),
        _ => false,
    }
}

/// One doc's relation to `lq` from its doc value, as the tree decides it:
/// the doc's box as a node holding only its triangles (`relate_node`,
/// taken whole when inside or outside, as `Matcher` takes a node), else
/// its triangles through the leaf tests. `Err`: a CONTAINS relation Lucene
/// refuses.
fn doc_matches(lq: &dyn LeafQuery, v: &rustic_core::fieldtypes::shape_dv::ShapeValue<'_>) -> Result<bool, String> {
    use rustic_core::fieldtypes::geo2d::{Relation, WithinRelation};
    let Some(e) = v.extent() else { return Ok(false) };
    let (left, right) = (e.neg_left.min(e.pos_left), e.neg_right.max(e.pos_right));
    let (min, max) = ([e.bottom, left, e.bottom, left], [e.top, right, e.top, right]);
    let rel = lq.relation();
    match lq.relate_node(&min, &max) {
        // CONTAINS: a node inside excludes its docs.
        Relation::Inside => return Ok(rel != QueryRelation::Contains),
        Relation::Outside => return Ok(false),
        Relation::Crosses => {}
    }
    match rel {
        // Each stops at the first triangle that decides the doc.
        QueryRelation::Intersects => {
            let mut hit = false;
            v.try_each_triangle(&mut |t| {
                hit = lq.leaf_matches(t);
                !hit
            });
            Ok(hit)
        }
        QueryRelation::Within | QueryRelation::Disjoint => {
            let (mut all, mut any) = (true, false);
            v.try_each_triangle(&mut |t| {
                any = true;
                all = lq.leaf_matches(t);
                all
            });
            Ok(any && all)
        }
        QueryRelation::Contains => {
            let (mut cand, mut not_within, mut err) = (false, false, None);
            v.try_each_triangle(&mut |t| {
                match lq.leaf_within(t) {
                    Ok(WithinRelation::Candidate) => cand = true,
                    Ok(WithinRelation::NotWithin) => not_within = true,
                    Ok(WithinRelation::Disjoint) => {}
                    Err(e) => err = Some(e),
                }
                !not_within && err.is_none()
            });
            match err {
                Some(e) => Err(e),
                None => Ok(cand && !not_within),
            }
        }
    }
}

/// Docs tested in doc order before the scan may give up.
const SCAN_BUDGET: u64 = 1 << 14;

#[cfg(test)]
thread_local! {
    /// Tests: a smaller scan budget, so scans switch to the tree.
    pub(crate) static TEST_SCAN_BUDGET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

fn scan_budget(wanted: u64) -> u64 {
    #[cfg(test)]
    if let Some(b) = TEST_SCAN_BUDGET.with(|c| c.get()) {
        return b;
    }
    SCAN_BUDGET.max(wanted.saturating_mul(2))
}

/// A size-limited search's scorer (`geo_scorer`'s hint): docs tested in
/// doc order from their doc values ([`doc_matches`], the tree's own
/// decisions per doc) while that finds the wanted matches soon, else the
/// tree's matches from the last doc returned. The same docs, ascending,
/// as the tree.
struct LazyShapeScorer {
    query: Arc<ShapeQuery>,
    field: u16,
    seg: &'static rustic_core::segment::Segment,
    col: &'static rustic_core::fieldtypes::shape_dv::ShapeDvColumn,
    num_docs: u32,
    /// The next doc to test.
    next: DocId,
    cur: DocId,
    budget: u64,
    wanted: u64,
    tested: u64,
    found: u64,
    tail: Option<Box<dyn Scorer>>,
    /// Matches of the last tested chunk not returned yet, and where the
    /// tree takes over after them (a doc the scan could not decide).
    pending: Vec<DocId>,
    ppos: usize,
    stop_at: Option<DocId>,
    /// Docs tested per chunk (in parallel), growing.
    chunk: u32,
}

impl LazyShapeScorer {
    /// More tests when the budget ran out: the observed hit rate projects
    /// the wanted matches left (twice over) well within the segment.
    fn extension(&self) -> Option<u64> {
        if self.found == 0 || self.found >= self.wanted {
            return None;
        }
        let more = ((self.wanted - self.found) as f64 * self.tested as f64 / self.found as f64 * 2.0).ceil() as u64;
        #[cfg(test)]
        if TEST_SCAN_BUDGET.with(|c| c.get()).is_some() {
            return (more.saturating_mul(4) < self.num_docs as u64).then_some(more.max(1));
        }
        (more.saturating_mul(4) < self.num_docs as u64).then_some(more.max(1024))
    }

    fn scan(&mut self) -> DocId {
        use rayon::prelude::*;
        use rustic_core::query::TERMINATED;
        let q = self.query.clone();
        let Some(lq) = leaf_of(&q) else { return self.switch() };
        loop {
            if let Some(&d) = self.pending.get(self.ppos) {
                self.ppos += 1;
                self.cur = d;
                return d;
            }
            if let Some(at) = self.stop_at.take() {
                self.next = at;
                return self.switch();
            }
            if self.next >= self.num_docs {
                self.cur = TERMINATED;
                return TERMINATED;
            }
            if self.budget == 0 {
                match self.extension() {
                    Some(more) => self.budget = more,
                    None => return self.switch(),
                }
            }
            // The next chunk's docs, tested in parallel when large.
            let len = (self.chunk as u64).min(self.budget).min((self.num_docs - self.next) as u64) as u32;
            let (from, col, lq) = (self.next, self.col, lq.as_ref());
            let test = |ds: &[DocId]| {
                let mut hits = Vec::new();
                for &d in ds {
                    let Some(v) = col.get(d) else { continue };
                    match doc_matches(lq, &v) {
                        Ok(true) => hits.push(d),
                        Ok(false) => {}
                        // The tree reports the error as it does.
                        Err(_) => return (hits, Some(d)),
                    }
                }
                (hits, None)
            };
            let docs: Vec<DocId> = (from..from + len).collect();
            // A small chunk (the first ones) on this thread: dispatching it
            // costs more than testing it.
            let parts: Vec<(Vec<DocId>, Option<DocId>)> = if len <= 256 { vec![test(&docs)] } else { docs.par_chunks(256).map(test).collect() };
            self.pending.clear();
            self.ppos = 0;
            for (hits, err) in parts {
                self.pending.extend_from_slice(&hits);
                if let Some(e) = err {
                    self.stop_at = Some(e);
                    break;
                }
            }
            self.found += self.pending.len() as u64;
            self.tested += len as u64;
            self.budget = self.budget.saturating_sub(len as u64);
            self.next = from + len;
            self.chunk = (self.chunk * 2).min(1 << 16);
        }
    }

    /// Continue with the tree's matches from the next untested doc.
    fn switch(&mut self) -> DocId {
        use rustic_core::query::TERMINATED;
        let leaf = LeafReader::Segment(self.seg);
        let bits = shape_hits(&leaf, self.field, &self.query).unwrap_or_default();
        let mut tail = rustic_core::query::hits_scorer(rustic_core::geo::DocHits::Bits(bits), 1.0);
        let target = self.next;
        let mut d = tail.advance();
        if d != TERMINATED && d < target {
            d = tail.seek(target);
        }
        self.tail = Some(tail);
        self.cur = d;
        d
    }
}

impl DocSet for LazyShapeScorer {
    fn advance(&mut self) -> DocId {
        if let Some(t) = &mut self.tail {
            self.cur = t.advance();
            return self.cur;
        }
        self.scan()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        use rustic_core::query::TERMINATED;
        if self.cur != TERMINATED && self.cur >= target {
            return self.cur;
        }
        if let Some(t) = &mut self.tail {
            self.cur = t.seek(target);
            return self.cur;
        }
        while self.pending.get(self.ppos).is_some_and(|&d| d < target) {
            self.ppos += 1;
        }
        if self.ppos >= self.pending.len() && self.stop_at.is_none() {
            self.next = self.next.max(target);
        }
        let mut d = self.scan();
        while d != TERMINATED && d < target {
            d = self.advance();
        }
        d
    }

    fn doc(&self) -> DocId {
        self.cur
    }

    fn size_hint(&self) -> u32 {
        self.num_docs
    }
}

impl Scorer for LazyShapeScorer {
    fn score(&mut self) -> f32 {
        1.0
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: bounded shape totals from the doc-order scan only
    /// (`Some(true)`) or from the tree only, whatever the segment's size
    /// (`Some(false)`).
    pub(crate) static TEST_COUNT_MODE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The matches over tree parts of single points, capped at `limit`. A
/// point never contains an area (see `shape_hits`): CONTAINS of one
/// matches nothing.
fn single_count(q: &ShapeQuery, lq: &(dyn LeafQuery + Sync), parts: &[rustic_core::fieldtypes::shape_bkd::ShapeBkd], limit: u64) -> Option<u64> {
    if lq.relation() == QueryRelation::Contains && query_has_area(q) {
        return Some(0);
    }
    let mut n = 0u64;
    for p in parts {
        n += p.count_single_at_least(lq, limit - n)?;
        if n >= limit {
            break;
        }
    }
    Some(n.min(limit))
}

/// Mean triangles per doc up to which WITHIN totals try the nodes taken
/// whole first.
const NODE_PASS_MAX_TRIANGLES_PER_DOC: u64 = 32;

/// Docs a segment needs before bounded totals try [`extent_count`].
const EXTENT_COUNT_MIN_DOCS: u32 = 1 << 16;

/// Docs [`extent_count`] tests per round (in parallel), between checks of
/// the limit.
const EXTENT_COUNT_ROUND: u32 = 1 << 18;

/// At least `limit` docs of a segment that match `lq` by their extent
/// alone (its box, as the tree decides a node: inside the query for
/// INTERSECTS / WITHIN, outside it for DISJOINT): `Some(limit)` once that
/// many are found, reading only doc-value headers (no triangle); `None`
/// when the segment has fewer such docs (the other docs' triangles decide
/// them: the count is then left to the tree or the scan).
fn extent_count(lq: &(dyn LeafQuery + Sync), col: &rustic_core::fieldtypes::shape_dv::ShapeDvColumn, num_docs: u32, limit: u64) -> Option<u64> {
    use rayon::prelude::*;
    use rustic_core::fieldtypes::geo2d::Relation;
    let decided = |d: DocId| -> bool {
        let Some(e) = col.get(d).and_then(|v| v.extent()) else { return false };
        let (left, right) = (e.neg_left.min(e.pos_left), e.neg_right.max(e.pos_right));
        lq.relate_node(&[e.bottom, left, e.bottom, left], &[e.top, right, e.top, right]) == Relation::Inside
    };
    let mut count = 0u64;
    let mut next = 0u32;
    let mut round = EXTENT_COUNT_ROUND / 16;
    while next < num_docs {
        let to = next.saturating_add(round).min(num_docs);
        count += (next..to).into_par_iter().with_min_len(4096).filter(|&d| decided(d)).count() as u64;
        if count >= limit {
            return Some(limit);
        }
        next = to;
        // The rate so far must project the limit within an eighth of the
        // segment, else the tree counts sooner; the next round is what the
        // rate projects for the rest, a quarter over.
        let rate = count as f64 / next as f64;
        if rate * (num_docs as f64 / 8.0) < limit as f64 {
            return None;
        }
        round = ((((limit - count) as f64 / rate) * 1.25) as u32).clamp(4096, EXTENT_COUNT_ROUND);
    }
    None
}

/// Docs tested in doc order before [`scan_count`] projects how many more
/// it needs.
const SCAN_COUNT_SAMPLE: u32 = 4096;

/// The matches of `lq` among a segment's first docs, in doc order, up to
/// `limit` (`min(count, limit)`, exact: each doc decided by
/// [`doc_matches`], the tree's own per-doc decision), when the hit rate of
/// the first [`SCAN_COUNT_SAMPLE`] docs projects `limit` within an eighth
/// of the segment; `None` otherwise (or on a doc the tree reports an
/// error for), and the tree counts.
fn scan_count(lq: &(dyn LeafQuery + Sync), col: &rustic_core::fieldtypes::shape_dv::ShapeDvColumn, num_docs: u32, limit: u64) -> Option<u64> {
    use rayon::prelude::*;
    let test = |from: u32, to: u32| -> Option<u64> {
        (from..to)
            .into_par_iter()
            .with_min_len(512)
            .map(|d| match col.get(d) {
                None => Some(0),
                Some(v) => doc_matches(lq, &v).ok().map(u64::from),
            })
            .try_reduce(|| 0, |a, b| Some(a + b))
    };
    let sample = SCAN_COUNT_SAMPLE.min(num_docs);
    let mut count = test(0, sample)?;
    if count >= limit || sample == num_docs {
        return Some(count.min(limit));
    }
    // Docs projected for the rest (twice over); the scan gives up at a
    // quarter of the segment.
    let rate = count as f64 / sample as f64;
    if rate == 0.0 || (limit - count) as f64 / rate * 2.0 > (num_docs / 8) as f64 {
        return None;
    }
    let mut next = sample;
    while count < limit && next < num_docs {
        if next >= num_docs / 4 {
            return None;
        }
        // The docs the hit rate so far projects for the rest, a quarter
        // over (so the last chunk rarely falls short).
        let rate = count.max(1) as f64 / next as f64;
        let chunk = (((limit - count) as f64 / rate * 1.25) as u32).clamp(4096, 1 << 18);
        let to = next.saturating_add(chunk).min(num_docs);
        count += test(next, to)?;
        next = to;
    }
    Some(count.min(limit))
}

/// Docs a segment needs before WITHIN / DISJOINT bounded totals count
/// from its tree rather than from the doc-order scan.
const TREE_COUNT_MIN_DOCS: u32 = 1 << 20;

/// CONTAINS over the docs of crossing leaves (see `shape_hits`).
struct ContainsWalk<'a> {
    lq: &'a dyn LeafQuery,
    col: &'a rustic_core::fieldtypes::shape_dv::ShapeDvColumn,
    /// The query's box `(minX, maxX, minY, maxY)`, degrees.
    qbox: (f64, f64, f64, f64),
    /// A point of the query (`(x, y)`, degrees): a doc containing the query
    /// contains it, so one of its triangles' boxes holds it; only nodes and
    /// triangles whose boxes hold it are read. `None`: every crossing leaf.
    probe: Option<(f64, f64)>,
    seen: Vec<u64>,
    hits: Vec<u64>,
    error: Option<String>,
}

impl ContainsWalk<'_> {
    fn candidate(&mut self, d: DocId) {
        use rustic_core::geo::{decode_lat, decode_lon};
        let (w, b) = (d as usize / 64, d % 64);
        let Some(word) = self.seen.get_mut(w) else { return };
        if *word >> b & 1 == 1 {
            return;
        }
        *word |= 1 << b;
        let Some(v) = self.col.get(d) else { return };
        let Some(e) = v.extent() else { return };
        let (x0, x1, y0, y1) = self.qbox;
        if decode_lon(e.neg_left.min(e.pos_left)) > x0 || decode_lon(e.neg_right.max(e.pos_right)) < x1 || decode_lat(e.bottom) > y0 || decode_lat(e.top) < y1 {
            return;
        }
        match doc_matches(self.lq, &v) {
            Ok(true) => self.hits[w] |= 1 << b,
            Ok(false) => {}
            Err(e) => self.error = Some(e),
        }
    }
}

impl rustic_core::fieldtypes::shape_bkd::TreeWalk for ContainsWalk<'_> {
    fn compare(&mut self, min: &[i32; 4], max: &[i32; 4]) -> rustic_core::fieldtypes::geo2d::Relation {
        use rustic_core::fieldtypes::geo2d::Relation;
        use rustic_core::geo::{decode_lat, decode_lon};
        if let Some((x, y)) = self.probe {
            // No triangle box of the node can hold the probe point: no
            // containing doc has a triangle here that matters.
            if decode_lon(min[1]) > x || decode_lon(max[3]) < x || decode_lat(min[0]) > y || decode_lat(max[2]) < y {
                return Relation::Outside;
            }
            return Relation::Crosses;
        }
        self.lq.relate_node(min, max)
    }

    fn inside_doc(&mut self, _doc: DocId) {}

    fn wants_inside(&self) -> bool {
        false
    }

    fn done(&self) -> bool {
        self.error.is_some()
    }

    fn wants_leaf_triangles(&self) -> bool {
        self.probe.is_some()
    }

    fn leaf(&mut self, docs: &[DocId], tris: &[[i32; 7]]) {
        use rustic_core::geo::{decode_lat, decode_lon};
        let Some((x, y)) = self.probe else { return };
        for (&d, t) in docs.iter().zip(tris) {
            // `(minY, minX, maxY, maxX)`: the triangle's box holds the probe.
            if decode_lon(t[1]) <= x && decode_lon(t[3]) >= x && decode_lat(t[0]) <= y && decode_lat(t[2]) >= y {
                self.candidate(d);
            }
            if self.error.is_some() {
                return;
            }
        }
    }

    fn leaf_docs(&mut self, docs: &[DocId]) {
        for &d in docs {
            self.candidate(d);
            if self.error.is_some() {
                return;
            }
        }
    }
}

/// [`ShapeTreeQuery`]'s bounded count: the docs of nodes the query takes
/// whole and of crossing leaves, each decided once by [`doc_matches`] from
/// its doc value (the tree's own per-doc decision), counted until `limit`.
/// Every match is in such a node or leaf, so a full walk counts exactly.
struct CountWalk<'a> {
    lq: &'a dyn LeafQuery,
    col: &'a rustic_core::fieldtypes::shape_dv::ShapeDvColumn,
    seen: Vec<u64>,
    count: u64,
    limit: u64,
    error: bool,
    /// The box (`minX, maxX, minY, maxY` over its triangles' boxes) of the
    /// node taken whole being walked: a doc whose extent lies in it lies in
    /// the query (WITHIN) or clear of it (DISJOINT) as the node does.
    node_box: [i32; 4],
    /// WITHIN / DISJOINT: only docs a node taken whole decides by their
    /// extent count; the others are left (no doc value tree read).
    nodes_only: bool,
}

impl CountWalk<'_> {
    #[inline]
    fn is_seen(&self, d: DocId) -> bool {
        self.seen.get(d as usize / 64).is_some_and(|w| w >> (d % 64) & 1 == 1)
    }

    #[inline]
    fn mark(&mut self, d: DocId) {
        if let Some(w) = self.seen.get_mut(d as usize / 64) {
            *w |= 1 << (d % 64);
        }
    }

    fn doc(&mut self, d: DocId) {
        if self.count >= self.limit || self.error {
            return;
        }
        let (w, b) = (d as usize / 64, d % 64);
        let Some(word) = self.seen.get_mut(w) else { return };
        if *word >> b & 1 == 1 {
            return;
        }
        *word |= 1 << b;
        if let Some(v) = self.col.get(d) {
            match doc_matches(self.lq, &v) {
                Ok(true) => self.count += 1,
                Ok(false) => {}
                Err(_) => self.error = true,
            }
        }
    }
}

impl rustic_core::fieldtypes::shape_bkd::TreeWalk for CountWalk<'_> {
    fn compare(&mut self, min: &[i32; 4], max: &[i32; 4]) -> rustic_core::fieldtypes::geo2d::Relation {
        let r = self.lq.relate_node(min, max);
        if r == rustic_core::fieldtypes::geo2d::Relation::Inside {
            self.node_box = [min[1], max[3], min[0], max[2]];
        }
        r
    }

    fn inside_doc(&mut self, doc: DocId) {
        if self.lq.relation() == QueryRelation::Intersects {
            // A triangle in a node inside: the doc intersects.
            if self.count < self.limit && !self.is_seen(doc) {
                self.mark(doc);
                self.count += 1;
            }
            return;
        }
        // WITHIN / DISJOINT: a doc whose whole extent lies in the node's box
        // (which the query takes whole) is decided with it.
        if self.count < self.limit && !self.is_seen(doc) {
            let inside = self.col.get(doc).and_then(|v| v.extent()).is_some_and(|e| {
                let [x0, x1, y0, y1] = self.node_box;
                e.neg_left.min(e.pos_left) >= x0 && e.neg_right.max(e.pos_right) <= x1 && e.bottom >= y0 && e.top <= y1
            });
            if inside {
                self.mark(doc);
                self.count += 1;
                return;
            }
        }
        if self.nodes_only {
            return;
        }
        self.doc(doc);
    }

    fn done(&self) -> bool {
        self.count >= self.limit || self.error
    }

    // Every doc of the leaf decided already (counted or excluded): its
    // triangles change nothing.
    fn skip_leaf(&self, distinct: &[DocId]) -> bool {
        distinct.iter().all(|&d| self.is_seen(d))
    }

    fn leaf(&mut self, docs: &[DocId], tris: &[[i32; 7]]) {
        use rustic_core::fieldtypes::geo2d::Tri;
        let intersects = self.lq.relation() == QueryRelation::Intersects;
        for (&d, t) in docs.iter().zip(tris) {
            if self.count >= self.limit || self.error {
                return;
            }
            if self.is_seen(d) {
                continue;
            }
            let pass = self.lq.leaf_matches(&Tri::decode(t));
            if intersects {
                // One intersecting triangle decides the doc; others leave
                // it to its other triangles.
                if pass {
                    self.mark(d);
                    self.count += 1;
                }
            } else if !pass {
                // WITHIN / DISJOINT: a failing triangle excludes the doc.
                self.mark(d);
            } else {
                self.doc(d);
            }
        }
    }
}

impl rustic_core::query::CustomQuery for ShapeTreeQuery {
    fn count_at_least(&self, leaf: &LeafReader<'_>, limit: u64) -> Option<u64> {
        let LeafReader::Segment(seg) = leaf else { return None };
        let lq = leaf_of(&self.query)?;
        let data = seg.shape_data(self.field)?;
        if data.tree.is_empty() {
            return None;
        }
        // Single points: the node counts, exactly, stopping once the nodes
        // the query takes whole reach the limit.
        if data.tree.iter().all(|p| p.one_triangle_per_doc()) {
            return single_count(&self.query, lq.as_ref(), &data.tree, limit);
        }
        // A count of a filter used again and again: from the segment's
        // filter cache (ES's query cache).
        if let Some(bits) = self.cached_hits(seg) {
            return Some(bits.iter().map(|w| w.count_ones() as u64).sum::<u64>().min(limit));
        }
        // CONTAINS has its own candidate path, see `shape_hits`.
        if lq.relation() == QueryRelation::Contains {
            return None;
        }
        let col = data.values.as_ref()?;
        #[cfg(test)]
        let mode = TEST_COUNT_MODE.with(|c| c.get());
        #[cfg(not(test))]
        let mode: Option<bool> = None;
        // (Tests take this path on segments of any size.)
        let extent_min = if cfg!(test) { 0 } else { EXTENT_COUNT_MIN_DOCS };
        // WITHIN / DISJOINT: first the docs of the nodes the query takes
        // whole whose extent lies in the node's box (integer compares on the
        // doc-value header, no triangle, no polygon test), stopping at the
        // limit; then docs decided by their extents alone (doc-value
        // headers against the query), when they come soon in doc order: a
        // large segment's bounded total often comes from either.
        // (INTERSECTS counts whole tree nodes, see below.)
        if mode.is_none() && seg.num_docs >= extent_min && limit != u64::MAX && matches!(lq.relation(), QueryRelation::Within | QueryRelation::Disjoint) {
            // (WITHIN over docs of few triangles each: a doc of thousands of
            // segments rarely fits a node's box, and DISJOINT's nodes are
            // the query's outside, where the extent pass decides faster.)
            let triangles: u64 = data.tree.iter().map(|p| p.triangles()).sum();
            let small_docs = triangles <= seg.num_docs as u64 * NODE_PASS_MAX_TRIANGLES_PER_DOC;
            if lq.relation() == QueryRelation::Within && small_docs {
            let mut w = CountWalk { lq: lq.as_ref(), col, seen: vec![0; (seg.num_docs as usize).div_ceil(64)], count: 0, limit, error: false, node_box: [0; 4], nodes_only: true };
            let mut buf = rustic_core::fieldtypes::shape_bkd::LeafBuf::default();
            for p in &data.tree {
                p.intersect_whole_nodes(&mut w, &mut buf);
                if w.count >= limit {
                    return Some(limit);
                }
            }
            }
            if let Some(n) = extent_count(lq.as_ref(), col, seg.num_docs, limit) {
                return Some(n);
            }
        }
        // WITHIN matches that come soon in doc order: counted from the doc
        // values in doc order (in parallel), as a size-limited search's
        // hits are. (INTERSECTS counts from the tree's triangles faster,
        // reading no doc value; DISJOINT from its nodes outside the query,
        // whose docs mostly lie outside by their extents alone.)
        if mode != Some(false) && (lq.relation() == QueryRelation::Within || mode == Some(true)) {
            if let Some(n) = scan_count(lq.as_ref(), col, seg.num_docs, limit) {
                return Some(n);
            }
        }
        if mode == Some(true) {
            return None;
        }
        let min_docs = if mode == Some(false) { 0 } else { TREE_COUNT_MIN_DOCS };
        // INTERSECTS: a doc is decided by one triangle, no doc value read.
        // WITHIN / DISJOINT decide most docs from their doc values: on small
        // segments (an unmerged index's) the scorer's own scan reads those
        // in order and in parallel faster, so only large segments count here.
        match lq.relation() {
            QueryRelation::Intersects => {}
            QueryRelation::Within | QueryRelation::Disjoint if seg.num_docs >= min_docs => {}
            _ => return None,
        }
        let mut w = CountWalk { lq: lq.as_ref(), col, seen: vec![0; (seg.num_docs as usize).div_ceil(64)], count: 0, limit, error: false, node_box: [0; 4], nodes_only: false };
        let mut buf = rustic_core::fieldtypes::shape_bkd::LeafBuf::default();
        // INTERSECTS counts the docs of whole nodes first (no triangle
        // tested); when those fall short, every match is needed after all:
        // the tree's parallel match finds them (kept for the scorer).
        let whole_first = lq.relation() == QueryRelation::Intersects;
        if whole_first {
            for p in &data.tree {
                p.intersect_whole_nodes(&mut w, &mut buf);
                if w.count >= limit {
                    return Some(limit);
                }
            }
            let bits = rustic_core::fieldtypes::shape_bkd::match_parts(lq.as_ref(), &data.tree, seg.num_docs).ok()?;
            let n: u64 = bits.iter().map(|x| x.count_ones() as u64).sum();
            if n < limit {
                self.full_hits.lock().unwrap().push((*seg as *const _ as usize, bits));
            }
            return Some(n.min(limit));
        }
        for p in &data.tree {
            if whole_first {
                p.intersect_whole_first(&mut w, &mut buf);
            } else {
                p.intersect(&mut w, &mut buf);
            }
            if w.count >= limit || w.error {
                break;
            }
        }
        // An INTERSECTS walk that ended short of the limit saw every
        // match: they are the docs it marked, and this leaf's scorer (built
        // next on this thread) takes them instead of looking again.
        if !w.error && w.count < limit && lq.relation() == QueryRelation::Intersects {
            self.full_hits.lock().unwrap().push((*seg as *const _ as usize, std::mem::take(&mut w.seen)));
        }
        (!w.error).then_some(w.count.min(limit))
    }

    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        // Every match of this leaf, found by its bounded count.
        if let LeafReader::Segment(seg) = leaf {
            let key = *seg as *const _ as usize;
            let found = {
                let mut v = self.full_hits.lock().unwrap();
                v.iter().position(|(s, _)| *s == key).map(|i| v.swap_remove(i).1)
            };
            if let Some(bits) = found {
                let _ = rustic_core::query::geo_scorer::take_hint();
                if bits.iter().all(|w| *w == 0) {
                    return None;
                }
                return Some(rustic_core::query::hits_scorer(rustic_core::geo::DocHits::Bits(bits), 1.0));
            }
        }
        // A search stopping after a few matches: docs in doc order from
        // their doc values while that is quick (one component query on a
        // segment with doc values).
        let wanted = rustic_core::query::geo_scorer::take_hint();
        // CONTAINS of an area over docs of one point each matches nothing
        // (see `shape_hits`): no scorer, rather than an all-zero bitset of
        // every doc built and scanned.
        if let (LeafReader::Segment(seg), Some(lq)) = (leaf, leaf_of(&self.query)) {
            if lq.relation() == QueryRelation::Contains && query_has_area(&self.query) {
                if let Some(data) = seg.shape_data(self.field) {
                    if !data.tree.is_empty() && data.tree.iter().all(|p| p.one_triangle_per_doc()) {
                        return None;
                    }
                }
            }
        }
        // A filter used again and again: its cached matches (ES's query
        // cache serves size-limited searches as well), read in place. Not
        // for a few hits over single points: the doc-order scan below finds
        // them sooner than a whole cached bitset is set up.
        if let LeafReader::Segment(seg) = leaf {
            if seg.shape_data(self.field).is_some_and(|d| !d.tree.is_empty() && !(wanted > 0 && d.tree.iter().all(|p| p.one_triangle_per_doc()))) {
                if let Some(bits) = self.cached_hits(seg) {
                    return Some(rustic_core::query::shared_bits_scorer(bits, 1.0));
                }
            }
        }
        // (CONTAINS matches few docs, if any: the scan would test its whole
        // budget before switching.)
        let contains = matches!(*self.query, ShapeQuery::Component(_, QueryRelation::Contains) | ShapeQuery::Rect(_, _, QueryRelation::Contains));
        if wanted > 0 && !contains && matches!(*self.query, ShapeQuery::Component(..) | ShapeQuery::Rect(..)) {
            if let LeafReader::Segment(seg) = leaf {
                // (An index-less field answers from the doc values anyway.)
                if let Some(col) = seg.shape_data(self.field).filter(|d| !d.tree.is_empty()).and_then(|d| d.values.as_ref()) {
                    // SAFETY: scorers never outlive the search holding the
                    // segment (the contract every leaf-bound scorer relies on).
                    let (seg, col) = unsafe {
                        (
                            std::mem::transmute::<&rustic_core::segment::Segment, &'static rustic_core::segment::Segment>(seg),
                            std::mem::transmute::<&rustic_core::fieldtypes::shape_dv::ShapeDvColumn, &'static rustic_core::fieldtypes::shape_dv::ShapeDvColumn>(col),
                        )
                    };
                    return Some(Box::new(LazyShapeScorer { query: self.query.clone(), field: self.field, seg, col, num_docs: seg.num_docs, next: 0, cur: rustic_core::query::TERMINATED, budget: scan_budget(wanted), wanted, tested: 0, found: 0, tail: None, pending: Vec::new(), ppos: 0, stop_at: None, chunk: (wanted.saturating_mul(4)).clamp(64, 1024) as u32 }));
                }
            }
        }
        // (A relation Lucene refuses for a component throws there; such
        // queries are split before they get here, see `ShapeQuery::new`.)
        let bits = shape_hits(leaf, self.field, &self.query).ok()?;
        if bits.iter().all(|w| *w == 0) {
            return None;
        }
        Some(rustic_core::query::hits_scorer(rustic_core::geo::DocHits::Bits(bits), 1.0))
    }

    fn exact_count(&self, leaf: &LeafReader<'_>) -> Option<u64> {
        if let LeafReader::Segment(seg) = leaf {
            if seg.shape_data(self.field).is_some_and(|d| !d.tree.is_empty() && !d.tree.iter().all(|p| p.one_triangle_per_doc())) {
                if let Some(bits) = self.cached_hits(seg) {
                    return Some(bits.iter().map(|w| w.count_ones() as u64).sum());
                }
            }
        }
        // Docs of one triangle each (single points): the tree's nodes count
        // them, no doc ids read.
        if let (LeafReader::Segment(seg), Some(lq)) = (leaf, leaf_of(&self.query)) {
            if let Some(data) = seg.shape_data(self.field) {
                if !data.tree.is_empty() && data.tree.iter().all(|p| p.one_triangle_per_doc()) {
                    return single_count(&self.query, lq.as_ref(), &data.tree, u64::MAX);
                }
            }
        }
        let bits = shape_hits(leaf, self.field, &self.query).ok()?;
        Some(bits.iter().map(|w| w.count_ones() as u64).sum())
    }

    fn is_constant_score(&self) -> bool {
        true
    }

    fn scores_uniformly(&self) -> bool {
        true
    }

    fn takes_geo_hint(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "geo_shape"
    }
}

/// `exists` on a `geo_shape` field: docs with triangles or a doc value.
#[derive(Debug)]
pub(crate) struct ShapeExistsQuery {
    pub field: u16,
}

pub(crate) fn shape_docs(leaf: &LeafReader<'_>, field: u16) -> Vec<u64> {
    let n = num_docs(leaf);
    let mut out = vec![0u64; (n as usize).div_ceil(64)];
    let mut set = |d: DocId| {
        if let Some(w) = out.get_mut(d as usize / 64) {
            *w |= 1 << (d % 64);
        }
    };
    match leaf {
        LeafReader::Segment(seg) => {
            if let Some(data) = seg.shape_data(field) {
                match &data.values {
                    Some(col) => {
                        for d in 0..n {
                            if col.has_value(d) {
                                set(d);
                            }
                        }
                    }
                    None => data.for_each_triangle(&mut |d, _| set(d)),
                }
            }
        }
        LeafReader::Memtable(w, _) => {
            if let Some(buf) = w.shape_buf(field) {
                buf.for_each_doc(n, &mut |d, _| set(d));
                for d in 0..n {
                    if buf.value(d).is_some() {
                        set(d);
                    }
                }
            }
        }
    }
    out
}

impl rustic_core::query::CustomQuery for ShapeExistsQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let bits = shape_docs(leaf, self.field);
        if bits.iter().all(|w| *w == 0) {
            return None;
        }
        Some(rustic_core::query::hits_scorer(rustic_core::geo::DocHits::Bits(bits), 1.0))
    }

    fn exact_count(&self, leaf: &LeafReader<'_>) -> Option<u64> {
        Some(shape_docs(leaf, self.field).iter().map(|w| w.count_ones() as u64).sum())
    }

    fn is_constant_score(&self) -> bool {
        true
    }

    fn scores_uniformly(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "exists"
    }
}

/// The relation of a `geo_shape` query.
pub(crate) fn query_relation(r: rustic_core::fieldtypes::shape::SpatialRelation) -> QueryRelation {
    use rustic_core::fieldtypes::shape::SpatialRelation as S;
    match r {
        S::Intersects => QueryRelation::Intersects,
        S::Within => QueryRelation::Within,
        S::Disjoint => QueryRelation::Disjoint,
        S::Contains => QueryRelation::Contains,
    }
}
