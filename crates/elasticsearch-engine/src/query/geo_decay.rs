//! `function_score` with `gauss` / `exp` / `linear` decay on a `geo_point`
//! field (ES `GeoFieldDataScoreFunction`). The core function_score only
//! knows numeric decays, so a function_score using a geo decay runs here as
//! an engine [`CustomQuery`]: the other functions are core
//! [`ScoreFunction`]s evaluated alongside.
//!
//! Distance = arc (sloppy haversine, ES `GeoUtils.arcDistance`) meters from `origin`, minus `offset`
//! (clamped at 0); multi-valued points reduce their distances with
//! `multi_value_mode` (`min` default, `max`, `avg`, `sum`); docs without a
//! point score 1.

use super::dsl::{parse_function, parse_query, QueryContext};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::values;
use rustic_core::query::{BoostMode, CustomQuery, DecayKind, DocSet, GeoDv, NumericDv, PreparedScoreFunction, LeafReader, Query, ScoreFunction, ScoreMode, Scorer, SearchContext, TERMINATED};
use rustic_core::types::DocId;
use serde_json::{Map, Value};
use std::sync::Arc;

const DECAYS: [&str; 3] = ["gauss", "exp", "linear"];

/// Does this `function_score` body use a decay function on a geo_point field?
pub fn has_geo_decay(o: &Map<String, Value>, ctx: &QueryContext) -> bool {
    let is_geo = |fo: &Map<String, Value>| {
        DECAYS.iter().any(|k| {
            fo.get(*k).and_then(Value::as_object).is_some_and(|d| {
                d.keys().filter(|n| n.as_str() != "multi_value_mode").any(|n| ctx.lookup.get(n).is_some_and(|f| f.ty == EsType::GeoPoint || f.declared == EsType::GeoPoint))
            })
        })
    };
    is_geo(o) || o.get("functions").and_then(Value::as_array).is_some_and(|fs| fs.iter().filter_map(Value::as_object).any(is_geo))
}

#[derive(Clone, Copy, Debug)]
enum MultiMode {
    Min,
    Max,
    Avg,
    Sum,
}

#[derive(Clone, Copy, Debug)]
struct GeoDecay {
    geo: Option<u16>,
    latlon: Option<(u16, u16)>,
    lat: f64,
    lon: f64,
    scale: f64,
    offset: f64,
    decay: f64,
    kind: DecayKind,
    mode: MultiMode,
}

/// Where a leaf's points of the decayed field come from.
enum GeoSrc<'a> {
    /// The geo index's doc-sorted points.
    Geo(GeoDv<'a>),
    /// Per-coordinate lat / lon columns.
    LatLon(NumericDv<'a>, NumericDv<'a>),
    None,
}

impl GeoDecay {
    /// Per-leaf point source (same precedence as before: geo index, then
    /// the lat / lon columns).
    fn source(&self, leaf: &LeafReader<'_>) -> GeoSrc<'static> {
        let src = match (self.geo, self.latlon) {
            (Some(g), _) if leaf.has_geo(g) => GeoSrc::Geo(GeoDv::open(leaf, g)),
            (_, Some((la, lo))) => GeoSrc::LatLon(NumericDv::open(leaf, la), NumericDv::open(leaf, lo)),
            (Some(g), None) => GeoSrc::Geo(GeoDv::open(leaf, g)),
            _ => GeoSrc::None,
        };
        // SAFETY: the source lives in a scorer built for `leaf`, which
        // never outlives the search holding the leaf (CustomQuery contract).
        unsafe { std::mem::transmute::<GeoSrc<'_>, GeoSrc<'static>>(src) }
    }

    #[inline]
    fn dist(&self, lat: f64, lon: f64) -> f64 {
        // `GeoDistance.ARC`: `GeoUtils.arcDistance` = `SloppyMath.haversinMeters`.
        (crate::sloppy_math::haversin_meters(self.lat, self.lon, lat, lon) - self.offset).max(0.0)
    }

    fn eval(&self, src: &GeoSrc<'_>, doc: DocId) -> f64 {
        let (mut n, mut min, mut max, mut sum) = (0usize, f64::INFINITY, f64::NEG_INFINITY, 0.0);
        let mut add = |d: f64| {
            n += 1;
            min = min.min(d);
            max = max.max(d);
            sum += d;
        };
        match src {
            GeoSrc::Geo(g) => {
                g.for_each(doc, |a, b| add(self.dist(a, b)));
            }
            GeoSrc::LatLon(la, lo) if la.is_single_valued() && lo.is_single_valued() => {
                if let (Some(a), Some(b)) = (la.first(doc), lo.first(doc)) {
                    add(self.dist(a, b));
                }
            }
            GeoSrc::LatLon(la, lo) => {
                let mut lats = Vec::new();
                la.for_each(doc, |v| lats.push(v));
                let mut i = 0;
                lo.for_each(doc, |b| {
                    if let Some(a) = lats.get(i) {
                        add(self.dist(*a, b));
                    }
                    i += 1;
                });
            }
            GeoSrc::None => {}
        }
        if n == 0 {
            return 1.0;
        }
        let d = match self.mode {
            MultiMode::Min => min,
            MultiMode::Max => max,
            MultiMode::Sum => sum,
            MultiMode::Avg => sum / n as f64,
        };
        self.kind.eval(d, self.scale, self.decay)
    }
}

#[derive(Debug)]
enum Func {
    Core(ScoreFunction),
    Geo { g: GeoDecay, weight: Option<f64>, filter: Option<Query> },
}

#[derive(Debug)]
struct GeoFunctionScore {
    inner: Query,
    funcs: Arc<Vec<Func>>,
    score_mode: ScoreMode,
    boost_mode: BoostMode,
    max_boost: f32,
    min_score: Option<f32>,
    boost: f32,
    filters: Vec<Query>,
}

fn parse_geo_decay(kind_name: &str, d: &Map<String, Value>, ctx: &QueryContext) -> Result<Option<GeoDecay>> {
    let mode = match d.get("multi_value_mode").and_then(Value::as_str).unwrap_or("min") {
        "min" | "MIN" => MultiMode::Min,
        "max" | "MAX" => MultiMode::Max,
        "avg" | "AVG" => MultiMode::Avg,
        "sum" | "SUM" => MultiMode::Sum,
        other => return Err(EsError::illegal_argument(format!("Illegal multi_value_mode [{other}]"))),
    };
    let Some((field, spec)) = d.iter().find(|(n, _)| n.as_str() != "multi_value_mode") else { return Ok(None) };
    let Some(f) = ctx.lookup.get(field) else { return Ok(None) };
    if f.ty != EsType::GeoPoint && f.declared != EsType::GeoPoint {
        return Ok(None);
    }
    let so = spec.as_object().ok_or_else(|| EsError::parsing(format!("[{kind_name}] field must be an object")))?;
    let origin = so.get("origin").ok_or_else(|| EsError::parsing("[origin] must be set for geo fields."))?;
    let (lat, lon) = values::parse_geo_point(origin).ok_or_else(|| EsError::parsing(format!("failed to parse [origin] [{origin}]")))?;
    let scale = so.get("scale").and_then(values::parse_distance).ok_or_else(|| EsError::parsing("[scale] must be set for geo fields."))?;
    let offset = match so.get("offset") {
        Some(v) => values::parse_distance(v).ok_or_else(|| EsError::parsing("failed to parse [offset]"))?,
        None => 0.0,
    };
    let decay = so.get("decay").and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(0.5);
    if !(decay > 0.0 && decay < 1.0) {
        return Err(EsError::illegal_argument(format!("decay must be in the range [0..1] but was {decay}")));
    }
    if scale <= 0.0 {
        return Err(EsError::illegal_argument(format!("scale must be > 0 but was {scale}")));
    }
    let kind = match kind_name {
        "gauss" => DecayKind::Gauss,
        "exp" => DecayKind::Exp,
        _ => DecayKind::Linear,
    };
    let latlon = match (f.lat_id, f.lon_id) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };
    Ok(Some(GeoDecay { geo: f.geo_id, latlon, lat, lon, scale, offset, decay, kind, mode }))
}

/// Build the function_score query (call when [`has_geo_decay`] is true).
pub fn function_score_query(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let inner = match o.get("query") {
        Some(q) => parse_query(q, ctx)?,
        None => Query::MatchAll,
    };
    let mut funcs = Vec::new();
    let mut filters = Vec::new();
    let mut one = |fo: &Map<String, Value>| -> Result<()> {
        for k in DECAYS {
            if let Some(d) = fo.get(k).and_then(Value::as_object) {
                if let Some(g) = parse_geo_decay(k, d, ctx)? {
                    let filter = match fo.get("filter") {
                        Some(f) => {
                            let q = parse_query(f, ctx)?;
                            filters.push(q.clone());
                            Some(q)
                        }
                        None => None,
                    };
                    funcs.push(Func::Geo { g, weight: fo.get("weight").and_then(Value::as_f64), filter });
                    return Ok(());
                }
            }
        }
        if let Some(f) = parse_function(fo, ctx)? {
            funcs.push(Func::Core(f));
        }
        Ok(())
    };
    if let Some(fs) = o.get("functions").and_then(Value::as_array) {
        for f in fs {
            one(f.as_object().ok_or_else(|| EsError::parsing("[functions] entries must be objects"))?)?;
        }
    }
    let mut top = Map::new();
    for k in ["field_value_factor", "random_score", "gauss", "exp", "linear", "weight"] {
        if let Some(v) = o.get(k) {
            top.insert(k.to_string(), v.clone());
        }
    }
    if !top.is_empty() {
        one(&top)?;
    }
    let score_mode = match o.get("score_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => ScoreMode::Multiply,
        "sum" => ScoreMode::Sum,
        "avg" => ScoreMode::Avg,
        "first" => ScoreMode::First,
        "max" => ScoreMode::Max,
        "min" => ScoreMode::Min,
        other => return Err(EsError::parsing(format!("failed to parse [score_mode]. Illegal score_mode: {other}"))),
    };
    let boost_mode = match o.get("boost_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => BoostMode::Multiply,
        "replace" => BoostMode::Replace,
        "sum" => BoostMode::Sum,
        "avg" => BoostMode::Avg,
        "max" => BoostMode::Max,
        "min" => BoostMode::Min,
        other => return Err(EsError::parsing(format!("failed to parse [boost_mode]. Illegal boost_mode: {other}"))),
    };
    let f32_of = |k: &str| o.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).map(|x| x as f32);
    Ok(Query::Custom(Arc::new(GeoFunctionScore {
        inner,
        funcs: Arc::new(funcs),
        score_mode,
        boost_mode,
        max_boost: f32_of("max_boost").unwrap_or(f32::MAX),
        min_score: f32_of("min_score"),
        boost: f32_of("boost").unwrap_or(1.0),
        filters,
    })))
}

/// Per-leaf prepared function: value and weight (for weighted `avg`).
enum Prep {
    Core(PreparedScoreFunction),
    Filtered { bits: Vec<u64>, inner: Box<Prep> },
    Geo { funcs: Arc<Vec<Func>>, i: usize, src: GeoSrc<'static> },
}

fn bits_of(q: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Vec<u64> {
    let mut bits = vec![0u64; (leaf.num_docs() as usize).div_ceil(64).max(1)];
    if let Some(mut s) = q.scorer(leaf, ctx) {
        loop {
            let d = s.advance();
            if d == TERMINATED {
                break;
            }
            if let Some(w) = bits.get_mut(d as usize / 64) {
                *w |= 1 << (d % 64);
            }
        }
    }
    bits
}

fn has_bit(bits: &[u64], d: DocId) -> bool {
    bits.get(d as usize / 64).is_some_and(|w| (w >> (d % 64)) & 1 == 1)
}

impl Prep {
    /// `(value, weight)` of an applicable function.
    #[inline]
    fn eval(&self, doc: DocId) -> Option<(f64, f64)> {
        match self {
            Prep::Core(p) => p.value(doc).map(|v| (v, p.weight_of())),
            Prep::Filtered { bits, inner } => {
                if has_bit(bits, doc) {
                    inner.eval(doc)
                } else {
                    None
                }
            }
            Prep::Geo { funcs, i, src } => match &funcs[*i] {
                Func::Geo { g, weight, .. } => {
                    let v = g.eval(src, doc);
                    Some(match weight {
                        Some(w) => (v * *w, *w),
                        None => (v, 1.0),
                    })
                }
                Func::Core(_) => None,
            },
        }
    }
}

// Restricted to match_all, one unweighted geo decay and exactly one point
// per physical document. Other queries retain the ordinary scorer.
struct GeoTopK<'a> {
    index: &'a rustic_core::geo::GeoIndex,
    decay: GeoDecay,
    num_docs: u32,
}

impl GeoTopK<'_> {
    fn collect(&self, k: usize, deleted: Option<&[u64]>, score: impl Fn(f64) -> f32) -> (Vec<rustic_core::query::collector::ScoredDoc>, u64) {
        use rustic_core::geo::{decode_lat, decode_lon};
        use rustic_core::query::collector::TopKCollector;
        let n = self.num_docs as usize;
        let removed: u64 = deleted.map(|words| words.iter().take(n.div_ceil(64)).enumerate().map(|(i, &word)| {
            let valid = (n - i * 64).min(64);
            let mask = if valid == 64 { u64::MAX } else { (1u64 << valid) - 1 };
            (word & mask).count_ones() as u64
        }).sum()).unwrap_or(0);
        let total = self.num_docs as u64 - removed;
        if k == 0 { return (Vec::new(), total); }
        let mut top = TopKCollector::new(k);
        let g = &self.decay;
        let decay = |distance: f64| g.kind.eval((distance - g.offset).max(0.0), g.scale, g.decay);
        self.index.visit_by_distance(g.lat, g.lon, |bound, points| {
            // Strictly lower: float-score plateaus must retain doc-id ties.
            if top.min_score().is_some_and(|minimum| score(decay(bound)) < minimum) {
                return false;
            }
            for point in points {
                if deleted.is_some_and(|words| words.get(point.doc as usize / 64).is_some_and(|w| (w >> (point.doc % 64)) & 1 != 0)) {
                    continue;
                }
                let value = g.kind.eval(g.dist(decode_lat(point.lat), decode_lon(point.lon)), g.scale, g.decay);
                top.collect(point.doc, score(value));
            }
            true
        });
        (top.into_sorted_vec(), total)
    }
}

struct GeoFsScorer {
    top_k: Option<GeoTopK<'static>>,
    inner: Box<dyn Scorer>,
    preps: Vec<Prep>,
    score_mode: ScoreMode,
    boost_mode: BoostMode,
    max_boost: f32,
    min_score: Option<f32>,
    boost: f32,
    cur: f32,
}

impl GeoFsScorer {
    fn compute(&mut self, d: DocId) -> f32 {
        let qs = self.inner.score();
        let (mut acc, mut wsum): (Option<f64>, f64) = (None, 0.0);
        for p in &self.preps {
            let Some((v, w)) = p.eval(d) else { continue };
            wsum += w;
            acc = Some(match (acc, self.score_mode) {
                (None, _) => v,
                (Some(a), ScoreMode::Multiply) => a * v,
                (Some(a), ScoreMode::Sum | ScoreMode::Avg) => a + v,
                (Some(a), ScoreMode::First) => a,
                (Some(a), ScoreMode::Max) => a.max(v),
                (Some(a), ScoreMode::Min) => a.min(v),
            });
            if self.score_mode == ScoreMode::First {
                break;
            }
        }
        let f = match acc {
            None => 1.0,
            Some(a) if self.score_mode == ScoreMode::Avg => a / wsum,
            Some(a) => a,
        };
        self.finish(qs, f)
    }

    fn finish(&self, qs: f32, f: f64) -> f32 {
        let f = f.min(self.max_boost as f64);
        let qd = qs as f64;
        let s = match self.boost_mode {
            BoostMode::Multiply => qd * f,
            BoostMode::Replace => f,
            BoostMode::Sum => qd + f,
            BoostMode::Avg => (qd + f) / 2.0,
            BoostMode::Max => qd.max(f),
            BoostMode::Min => qd.min(f),
        };
        s as f32 * self.boost
    }

    fn settle(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                return d;
            }
            let s = self.compute(d);
            match self.min_score {
                Some(m) if s < m => d = self.inner.advance(),
                _ => {
                    self.cur = s;
                    return d;
                }
            }
        }
    }
}

impl DocSet for GeoFsScorer {
    fn advance(&mut self) -> DocId {
        let d = self.inner.advance();
        self.settle(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        let d = self.inner.seek(target);
        self.settle(d)
    }
    fn doc(&self) -> DocId {
        self.inner.doc()
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for GeoFsScorer {
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<rustic_core::query::collector::ScoredDoc>, u64)> {
        self.top_k.as_ref().map(|top| top.collect(k, deleted, |f| self.finish(1.0, f)))
    }

    fn score(&mut self) -> f32 {
        self.cur
    }
}

impl CustomQuery for GeoFunctionScore {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let inner = self.inner.scorer(leaf, ctx)?;
        let preps = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| match f {
                Func::Core(c) => Prep::Core(PreparedScoreFunction::build(c, leaf, ctx)),
                Func::Geo { g, filter, .. } => {
                    let p = Prep::Geo { funcs: self.funcs.clone(), i, src: g.source(leaf) };
                    match filter {
                        Some(q) => Prep::Filtered { bits: bits_of(q, leaf, ctx), inner: Box::new(p) },
                        None => p,
                    }
                }
            })
            .collect();
        let top_k = match (&self.inner, self.funcs.as_slice(), leaf) {
            (Query::MatchAll, [Func::Geo { g, weight: None, filter: None }], LeafReader::Segment(segment))
                if self.min_score.is_none() && self.boost.is_finite() && self.boost >= 0.0
                    && self.max_boost.is_finite() && self.max_boost >= 0.0
                    && g.scale.is_finite() && g.scale > 0.0 && g.offset.is_finite() && g.offset >= 0.0
                    && g.decay > 0.0 && g.decay < 1.0 => {
                g.geo.and_then(|field| segment.geo_index(field)).filter(|index| index.docs_with_points() == Some(leaf.num_docs() as u64)).map(|index| {
                    // SAFETY: as for GeoSrc above, the scorer cannot outlive
                    // the search borrowing its segment (CustomQuery contract).
                    let index = unsafe { std::mem::transmute::<&rustic_core::geo::GeoIndex, &'static rustic_core::geo::GeoIndex>(index) };
                    GeoTopK { index, decay: *g, num_docs: leaf.num_docs() }
                })
            }
            _ => None,
        };
        Some(Box::new(GeoFsScorer {
            top_k,
            inner,
            preps,
            score_mode: self.score_mode,
            boost_mode: self.boost_mode,
            max_boost: self.max_boost,
            min_score: self.min_score,
            boost: self.boost,
            cur: 0.0,
        }))
    }
    fn children(&self) -> Vec<&Query> {
        let mut v = vec![&self.inner];
        v.extend(self.filters.iter());
        v
    }
    fn name(&self) -> &'static str {
        "function_score"
    }
}

#[cfg(test)]
mod top_k_tests {
    use super::*;
    use rustic_core::geo::{GeoIndex, GeoPointEntry, encode_lat, encode_lon, decode_lat, decode_lon};
    use rustic_core::query::collector::TopKCollector;

    #[test]
    fn spatial_pruning_matches_exhaustive_scores_ties_and_deletions() {
        let n = 20_003u32;
        let points: Vec<_> = (0..n).map(|doc| GeoPointEntry {
            doc, lat: encode_lat((doc % 179) as f64 - 89.0),
            lon: encode_lon((doc.wrapping_mul(1237) % 359) as f64 - 179.0),
        }).collect();
        let index = GeoIndex::build(points.clone());
        let mut deleted = vec![0u64; (n as usize).div_ceil(64)];
        for d in (0..n).step_by(7) { deleted[d as usize / 64] |= 1 << (d % 64); }
        for (lat, lon) in [(52.37, 4.8951), (89.9, 179.9), (-89.9, -179.9)] {
            for (scale, offset) in [(500_000.0, 0.0), (10_000.0, 0.0), (20_000_000.0, 1_000_000.0)] {
                for kind in [DecayKind::Gauss, DecayKind::Exp, DecayKind::Linear] {
                    let decay = GeoDecay { geo: Some(0), latlon: None, lat, lon, scale, offset,
                        decay: 0.1, kind, mode: MultiMode::Min };
                    let fast = GeoTopK { index: &index, decay, num_docs: n };
                    for k in [0, 1, 10, 100] {
                        let mut expected = TopKCollector::new(k);
                        let mut total = 0;
                        for point in &points {
                            if deleted[point.doc as usize / 64] & (1 << (point.doc % 64)) != 0 { continue; }
                            total += 1;
                            let value = kind.eval(decay.dist(decode_lat(point.lat), decode_lon(point.lon)), scale, 0.1);
                            expected.collect(point.doc, value as f32);
                        }
                        assert_eq!(fast.collect(k, Some(&deleted), |value| value as f32), (expected.into_sorted_vec(), total));
                    }
                }
            }
        }
    }
}
