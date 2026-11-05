//! `function_score` (ES): rescale a query's scores with functions of
//! fast-field values.
//!
//! Each leaf snapshots the values its functions need once per query
//! ([`PreparedFunctions::prepare`]), so the resulting scorer is `'static`
//! and does not borrow the leaf.

use crate::query::docset::{DocSet, TERMINATED};
use crate::query::explain::Explanation;
use crate::query::numeric_dv::NumericDv;
use crate::query::query::{Query, SearchContext};
use crate::query::reader::LeafReader;
use crate::query::scorer::Scorer;
use crate::types::DocId;
use crate::query::collector::{ScoredDoc, TopKCollector};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct RawRank(f64);
impl PartialEq for RawRank { fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal } }
impl Eq for RawRank {}
impl PartialOrd for RawRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for RawRank {
    fn cmp(&self, other: &Self) -> Ordering { self.0.total_cmp(&other.0) }
}

/// Positive monotone transforms can reject raw values before expensive math.
/// Docs arrive in ascending order, so a later doc cannot win a rounded-score
/// tie. Invalid/negative raw values are always evaluated to preserve errors.
/// Callers must prove that `score` is nondecreasing for finite nonnegative
/// raw values, with no per-document effects or errors in that domain, and
/// that every live document matches. `k` must be positive. Values outside
/// that domain always call `score`, so it can report validation errors.
pub fn monotone_top_k(num_docs: u32, k: usize, deleted: Option<&[u64]>, mut value: impl FnMut(DocId) -> f64, mut score: impl FnMut(DocId, f64) -> f32) -> (Vec<ScoredDoc>, u64) {
    assert!(k > 0);
    let mut raw = BinaryHeap::<Reverse<RawRank>>::with_capacity(k);
    let mut top = TopKCollector::new(k);
    let mut total = 0;
    let mut evaluated = 0;
    let mut pruning = true;
    for doc in 0..num_docs {
        if deleted.is_some_and(|words| words.get(doc as usize / 64).is_some_and(|w| w & (1 << (doc % 64)) != 0)) { continue; }
        total += 1;
        let v = value(doc);
        if v.is_finite() && v >= 0.0 && pruning {
            if raw.len() == k && v <= raw.peek().unwrap().0.0 { continue; }
            if raw.len() == k { raw.pop(); }
            raw.push(Reverse(RawRank(v)));
        }
        top.collect(doc, score(doc, v));
        evaluated += 1;
        // Sorted ascending input makes nearly every doc competitive. Avoid
        // maintaining a second heap when the bound is ineffective.
        if total == 4096 && evaluated > 3072 { pruning = false; }
    }
    (top.into_sorted_vec(), total)
}

/// [`monotone_top_k`] read from a single-valued long column's points index
/// (`values`), best value first: docs are visited from the largest value
/// down and the walk stops once a value scores strictly below the k-th
/// best held (`score` is nondecreasing in the value, so that value's other
/// docs and every smaller one can only lose; equal scores still compete by
/// doc id). Instead of reading every doc's value, a top 10 reads about ten.
/// Only when every doc has a value (a missing value would sit anywhere in
/// the order) and the smallest value is finite and nonnegative, so every
/// value is in `score`'s monotone domain, where it raises no error: the
/// docs the walk skips could not have reported one. `None` otherwise (use
/// [`monotone_top_k`]). Same preconditions as [`monotone_top_k`].
pub fn monotone_top_k_from_points(values: &NumericDv<'_>, num_docs: u32, k: usize, deleted: Option<&[u64]>, mut score: impl FnMut(DocId, f64) -> f32) -> Option<(Vec<ScoredDoc>, u64)> {
    let NumericDv::SegI64(c) = values else { return None };
    if k == 0 || c.value_doc_count() != num_docs as u64 || c.num_docs() != num_docs {
        return None;
    }
    let (min, _) = c.points_min_max()?;
    if min < 0 {
        return None;
    }
    let is_deleted = |doc: DocId| deleted.is_some_and(|words| words.get(doc as usize / 64).is_some_and(|w| w & (1 << (doc % 64)) != 0));
    let removed: u64 = deleted.map(|words| words.iter().take((num_docs as usize).div_ceil(64)).enumerate().map(|(i, &word)| {
        let valid = (num_docs as usize - i * 64).min(64);
        (word & if valid == 64 { u64::MAX } else { (1u64 << valid) - 1 }).count_ones() as u64
    }).sum()).unwrap_or(0);
    let mut top = TopKCollector::new(k);
    let walked = c.visit_desc(|doc, v| {
        if is_deleted(doc) {
            return true;
        }
        let s = score(doc, v as f64);
        if top.min_score().is_some_and(|m| s < m) {
            return false;
        }
        top.collect(doc, s);
        true
    });
    walked.then(|| (top.into_sorted_vec(), num_docs as u64 - removed))
}

/// How `field_value_factor` transforms `factor * value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FieldValueModifier {
    /// Unchanged.
    #[default]
    None,
    /// `log10(x)`.
    Log,
    /// `log10(x + 1)`.
    Log1p,
    /// `log10(x + 2)`.
    Log2p,
    /// `ln(x)`.
    Ln,
    /// `ln(x + 1)`.
    Ln1p,
    /// `ln(x + 2)`.
    Ln2p,
    /// `x * x`.
    Square,
    /// `sqrt(x)`.
    Sqrt,
    /// `1 / x`.
    Reciprocal,
}

impl FieldValueModifier {
    /// Apply the modifier.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            FieldValueModifier::None => x,
            FieldValueModifier::Log => x.log10(),
            FieldValueModifier::Log1p => (x + 1.0).log10(),
            FieldValueModifier::Log2p => (x + 2.0).log10(),
            FieldValueModifier::Ln => x.ln(),
            FieldValueModifier::Ln1p => (x + 1.0).ln(),
            FieldValueModifier::Ln2p => (x + 2.0).ln(),
            FieldValueModifier::Square => x * x,
            FieldValueModifier::Sqrt => x.sqrt(),
            FieldValueModifier::Reciprocal => 1.0 / x,
        }
    }
}

/// Shape of a decay function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecayKind {
    /// `exp(-d^2 / (2 sigma^2))` with `sigma^2 = -scale^2 / (2 ln(decay))`.
    Gauss,
    /// `exp(lambda * d)` with `lambda = ln(decay) / scale`.
    Exp,
    /// `max(0, (s - d) / s)` with `s = scale / (1 - decay)`.
    Linear,
}

impl DecayKind {
    /// Decay value at distance `d` (already reduced by `offset`).
    pub fn eval(&self, d: f64, scale: f64, decay: f64) -> f64 {
        match self {
            DecayKind::Gauss => {
                let sigma2 = -scale * scale / (2.0 * decay.ln());
                (-(d * d) / (2.0 * sigma2)).exp()
            }
            DecayKind::Exp => ((decay.ln() / scale) * d).exp(),
            DecayKind::Linear => {
                let s = scale / (1.0 - decay);
                ((s - d) / s).max(0.0)
            }
        }
    }
}

/// One scoring function.
#[derive(Clone, Debug)]
pub enum ScoreFunction {
    /// A constant.
    Weight(f32),
    /// `modifier(factor * value)` of a numeric fast field; `missing` is used
    /// for docs without a value (1.0 when unset). Multi-valued fields use
    /// the first value.
    FieldValueFactor { field: u16, factor: f32, modifier: FieldValueModifier, missing: Option<f64> },
    /// A deterministic pseudo-random value in `[0, 1)` per `(seed, doc)`.
    RandomScore { seed: u64 },
    /// Decay with the distance of a numeric field from `origin` (dates are
    /// i64 millis). Docs without a value score 1; multi-valued fields use
    /// the closest value.
    Decay { field: u16, origin: f64, scale: f64, offset: f64, decay: f64, kind: DecayKind },
    /// `function` applies only to docs matching `filter`.
    Filtered { filter: Box<Query>, function: Box<ScoreFunction> },
    /// `function` multiplied by `weight` (ES per-function `weight`).
    Weighted { function: Box<ScoreFunction>, weight: f32 },
}

/// How function values combine (ES `score_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScoreMode {
    /// Product.
    #[default]
    Multiply,
    /// Sum.
    Sum,
    /// Mean of the applicable functions.
    Avg,
    /// First applicable function.
    First,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

/// How the combined function value meets the query score (ES `boost_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoostMode {
    /// `query * functions`.
    #[default]
    Multiply,
    /// `functions`.
    Replace,
    /// `query + functions`.
    Sum,
    /// `(query + functions) / 2`.
    Avg,
    /// `max(query, functions)`.
    Max,
    /// `min(query, functions)`.
    Min,
}

impl BoostMode {
    fn combine(&self, q: f64, f: f64) -> f64 {
        // Java Math.min/max propagate NaN; Rust's min/max select the other
        // operand. Preserve invalid function scores for the scorer's error gate.
        if q.is_nan() || f.is_nan() {
            return f64::NAN;
        }
        match self {
            BoostMode::Multiply => q * f,
            BoostMode::Replace => f,
            BoostMode::Sum => q + f,
            BoostMode::Avg => (q + f) / 2.0,
            BoostMode::Max => q.max(f),
            BoostMode::Min => q.min(f),
        }
    }
}

/// One [`ScoreFunction`] bound to a leaf (doc-value columns opened once).
pub enum Prepared {
    Weight(f64),
    Fvf { values: NumericDv<'static>, factor: f64, modifier: FieldValueModifier, missing: Option<f64> },
    Random { seed: u64 },
    Decay { values: NumericDv<'static>, origin: f64, scale: f64, offset: f64, decay: f64, kind: DecayKind },
    Filtered { bits: Vec<u64>, inner: Box<Prepared> },
    Weighted { inner: Box<Prepared>, weight: f64 },
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The value of `random_score` for `(seed, doc)`, in `[0, 1)`.
pub fn random_score(seed: u64, doc: DocId) -> f64 {
    (splitmix64(seed ^ splitmix64(doc as u64)) >> 11) as f64 / (1u64 << 53) as f64
}

impl Prepared {
    /// Bind `f` to `leaf`.
    pub fn build(f: &ScoreFunction, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Self {
        match f {
            ScoreFunction::Weight(w) => Prepared::Weight(*w as f64),
            ScoreFunction::FieldValueFactor { field, factor, modifier, missing } => Prepared::Fvf {
                values: open_dv(leaf, *field),
                factor: *factor as f64,
                modifier: *modifier,
                missing: *missing,
            },
            ScoreFunction::RandomScore { seed } => Prepared::Random { seed: *seed },
            ScoreFunction::Decay { field, origin, scale, offset, decay, kind } => Prepared::Decay {
                values: open_dv(leaf, *field),
                origin: *origin,
                scale: *scale,
                offset: *offset,
                decay: *decay,
                kind: *kind,
            },
            ScoreFunction::Filtered { filter, function } => {
                let mut bits = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                if let Some(mut s) = filter.scorer(leaf, ctx) {
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
                Prepared::Filtered { bits, inner: Box::new(Prepared::build(function, leaf, ctx)) }
            }
            ScoreFunction::Weighted { function, weight } => {
                Prepared::Weighted { inner: Box::new(Prepared::build(function, leaf, ctx)), weight: *weight as f64 }
            }
        }
    }

    /// The function's `weight` (1 unless set).
    pub fn weight_of(&self) -> f64 {
        match self {
            Prepared::Weight(w) => *w,
            Prepared::Weighted { weight, .. } => *weight,
            Prepared::Filtered { inner, .. } => inner.weight_of(),
            _ => 1.0,
        }
    }

    /// Value for `doc`, or `None` if a filter excludes it.
    #[inline]
    pub fn value(&self, doc: DocId) -> Option<f64> {
        match self {
            Prepared::Weight(w) => Some(*w),
            Prepared::Fvf { values, factor, modifier, missing } => {
                let v = values.first(doc).or(*missing).unwrap_or(1.0);
                Some(modifier.apply(factor * v))
            }
            Prepared::Random { seed } => Some(random_score(*seed, doc)),
            Prepared::Decay { values, origin, scale, offset, decay, kind } => {
                let mut d = f64::INFINITY;
                if values.for_each(doc, |v| d = d.min(((v - origin).abs() - offset).max(0.0))) == 0 {
                    return Some(1.0);
                }
                Some(kind.eval(d, *scale, *decay))
            }
            Prepared::Filtered { bits, inner } => {
                if bits.get(doc as usize / 64).is_some_and(|w| (w >> (doc % 64)) & 1 == 1) {
                    inner.value(doc)
                } else {
                    None
                }
            }
            Prepared::Weighted { inner, weight } => inner.value(doc).map(|v| v * weight),
        }
    }

    /// [`Prepared::value`] with its explanation text (explain only).
    fn eval(&self, doc: DocId) -> Option<(f64, String)> {
        let v = self.value(doc)?;
        let d = match self {
            Prepared::Weight(_) => "weight".into(),
            Prepared::Fvf { values, factor, modifier, missing } => {
                let x = values.first(doc).or(*missing).unwrap_or(1.0);
                format!("field value function: {modifier:?}(doc value {x} * factor {factor})")
            }
            Prepared::Random { seed } => format!("random score function (seed: {seed})"),
            Prepared::Decay { values, origin, offset, kind, .. } => {
                let mut d = f64::INFINITY;
                if values.for_each(doc, |x| d = d.min(((x - origin).abs() - offset).max(0.0))) == 0 {
                    "decay: missing value".into()
                } else {
                    format!("{kind:?} decay at distance {d}")
                }
            }
            Prepared::Filtered { inner, .. } => format!("filtered: {}", inner.eval(doc)?.1),
            Prepared::Weighted { inner, weight } => format!("{} * weight {weight}", inner.eval(doc)?.1),
        };
        Some((v, d))
    }
}

/// [`NumericDv`] of `field` with the leaf lifetime erased.
fn open_dv(leaf: &LeafReader<'_>, field: u16) -> NumericDv<'static> {
    let dv = NumericDv::open(leaf, field);
    // SAFETY: prepared functions live inside the scorer / explain call
    // built for `leaf`, and scorers never outlive the search holding the
    // leaf's segment (the same contract `CustomQuery` scorers rely on).
    unsafe { std::mem::transmute::<NumericDv<'_>, NumericDv<'static>>(dv) }
}

/// Function-score parameters a scorer applies.
#[derive(Clone, Copy, Debug)]
pub struct FunctionScoreParams {
    /// Combination of function values.
    pub score_mode: ScoreMode,
    /// Combination of query score and function value.
    pub boost_mode: BoostMode,
    /// Upper bound of the combined function value.
    pub max_boost: f32,
    /// Docs scoring below this are dropped.
    pub min_score: Option<f32>,
}

/// The functions of one `function_score`, prepared for one leaf.
pub struct PreparedFunctions {
    funcs: Vec<Prepared>,
    params: FunctionScoreParams,
    invalid: std::sync::Arc<parking_lot::Mutex<Option<(DocId, f32)>>>,
}

impl PreparedFunctions {
    /// Snapshot whatever `functions` need on `leaf`.
    pub fn prepare(functions: &[ScoreFunction], params: FunctionScoreParams, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Self {
        Self { funcs: functions.iter().map(|f| Prepared::build(f, leaf, ctx)).collect(), params, invalid: ctx.invalid_score.clone() }
    }

    /// Combined function value for `doc` (1 when no function applies) and
    /// the per-function parts.
    pub fn factor(&self, doc: DocId) -> (f64, Vec<(f64, String)>) {
        let mut weights: Vec<f64> = Vec::new();
        let parts: Vec<(f64, String)> = self
            .funcs
            .iter()
            .filter_map(|f| {
                let p = f.eval(doc)?;
                weights.push(f.weight_of());
                Some(p)
            })
            .take(if self.params.score_mode == ScoreMode::First { 1 } else { usize::MAX })
            .collect();
        let f = if parts.is_empty() {
            1.0
        } else if self.params.score_mode != ScoreMode::First && parts.iter().any(|p| p.0.is_nan()) {
            f64::NAN
        } else {
            let vals = parts.iter().map(|p| p.0);
            match self.params.score_mode {
                ScoreMode::Multiply => vals.product(),
                ScoreMode::Sum => vals.sum(),
                // Weighted average: the sum of the (weighted) values over the sum of weights.
                ScoreMode::Avg => vals.sum::<f64>() / weights.iter().sum::<f64>(),
                ScoreMode::First => parts[0].0,
                ScoreMode::Max => vals.fold(f64::NEG_INFINITY, f64::max),
                ScoreMode::Min => vals.fold(f64::INFINITY, f64::min),
            }
        };
        // Java `Math.min`: NaN propagates.
        (if f.is_nan() { f } else { f.min(self.params.max_boost as f64) }, parts)
    }

    /// Combined function value for `doc` (1 when no function applies),
    /// without explanation parts.
    #[inline]
    fn factor_value(&self, doc: DocId) -> f64 {
        if let ([f], false) = (self.funcs.as_slice(), self.params.score_mode == ScoreMode::Avg) {
            let value = f.value(doc).unwrap_or(1.0);
            return if value.is_nan() { value } else { value.min(self.params.max_boost as f64) };
        }
        let mut acc: Option<f64> = None;
        let mut wsum = 0.0;
        for p in &self.funcs {
            let Some(v) = p.value(doc) else { continue };
            if v.is_nan() {
                return v;
            }
            if self.params.score_mode == ScoreMode::Avg {
                wsum += p.weight_of();
            }
            acc = Some(match (acc, self.params.score_mode) {
                (None, _) => v,
                (Some(a), ScoreMode::Multiply) => a * v,
                (Some(a), ScoreMode::Sum | ScoreMode::Avg) => a + v,
                (Some(a), ScoreMode::First) => a,
                (Some(a), ScoreMode::Max) => a.max(v),
                (Some(a), ScoreMode::Min) => a.min(v),
            });
            if self.params.score_mode == ScoreMode::First {
                break;
            }
        }
        let f = match acc {
            None => 1.0,
            Some(a) if self.params.score_mode == ScoreMode::Avg => a / wsum,
            Some(a) => a,
        };
        if f.is_nan() { f } else { f.min(self.params.max_boost as f64) }
    }

    /// Final score of `doc` given its query score.
    #[inline]
    pub fn score(&self, doc: DocId, query_score: f32) -> f32 {
        self.params.boost_mode.combine(query_score as f64, self.factor_value(doc)) as f32
    }

    /// Explanation of [`PreparedFunctions::score`].
    pub fn explain(&self, doc: DocId, query: Explanation) -> Explanation {
        let (f, parts) = self.factor(doc);
        let value = self.params.boost_mode.combine(query.value as f64, f) as f32;
        let fdetails = parts.into_iter().map(|(v, d)| Explanation::leaf(v as f32, d)).collect();
        Explanation::new(
            value,
            format!("function score, boost_mode {:?}", self.params.boost_mode),
            vec![
                query,
                Explanation::new(f as f32, format!("functions combined with score_mode {:?}, capped at max_boost {}", self.params.score_mode, self.params.max_boost), fdetails),
            ],
        )
    }
}

/// Wraps a query scorer, combining its score with prepared functions.
pub struct FunctionScoreScorer {
    inner: Box<dyn Scorer>,
    funcs: PreparedFunctions,
    current_score: f32,
    match_all_docs: Option<u32>,
}

impl FunctionScoreScorer {
    /// Wrap `inner`.
    pub fn new(inner: Box<dyn Scorer>, funcs: PreparedFunctions) -> Self {
        Self { inner, funcs, current_score: 0.0, match_all_docs: None }
    }

    pub(crate) fn with_match_all(mut self, num_docs: u32) -> Self {
        self.match_all_docs = Some(num_docs);
        self
    }

    fn settle(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                return d;
            }
            let s = self.funcs.score(d, self.inner.score());
            if !(s >= 0.0) {
                let mut slot = self.funcs.invalid.lock();
                if slot.is_none() {
                    *slot = Some((d, s));
                }
            }
            match self.funcs.params.min_score {
                Some(ms) if s < ms => d = self.inner.advance(),
                _ => {
                    self.current_score = s;
                    return d;
                }
            }
        }
    }
}

impl DocSet for FunctionScoreScorer {
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

impl Scorer for FunctionScoreScorer {
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<ScoredDoc>, u64)> {
        let num_docs = self.match_all_docs?;
        let params = self.funcs.params;
        if k == 0 || params.min_score.is_some() || !params.max_boost.is_finite() || params.max_boost < 0.0 { return None; }
        let [Prepared::Fvf { values, factor, modifier, missing }] = self.funcs.funcs.as_slice() else { return None };
        if !factor.is_finite() || *factor <= 0.0 || !matches!(modifier, FieldValueModifier::Log1p | FieldValueModifier::Log2p | FieldValueModifier::Ln1p | FieldValueModifier::Ln2p) { return None; }
        let invalid = &self.funcs.invalid;
        let score = |doc: DocId, raw: f64| {
            let f = modifier.apply(factor * raw);
            let f = if f.is_nan() { f } else { f.min(params.max_boost as f64) };
            let s = params.boost_mode.combine(1.0, f) as f32;
            if !(s >= 0.0) {
                let mut slot = invalid.lock();
                if slot.is_none() { *slot = Some((doc, s)); }
            }
            s
        };
        if let Some(out) = monotone_top_k_from_points(values, num_docs, k, deleted, score) {
            return Some(out);
        }
        Some(monotone_top_k(num_docs, k, deleted, |doc| values.first(doc).or(*missing).unwrap_or(1.0), score))
    }

    fn score(&mut self) -> f32 {
        self.current_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The points walk returns exactly the full scan's hits and total: dense
    /// long columns of distinct, tied and rounding-tied values (scores of
    /// different values equal as f32), with deletions; columns with a
    /// missing value or a negative one take the scan.
    #[test]
    fn points_walk_matches_the_full_scan() {
        use crate::codec::directory::RamDirectory;
        use crate::segment::SegmentWriter;
        use crate::types::{Document, Field, FieldType};
        let schema = vec![Field::new(0, "population", FieldType::I64)];
        let dir = RamDirectory::new();
        for (pattern, id) in [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (4, 5)] {
            let n = 30_000u32;
            let mut w = SegmentWriter::new(id);
            for doc in 0..n {
                let mut d = Document::new();
                let v: Option<i64> = match pattern {
                    0 => Some(((doc as u64 * 7919) % 1_000_000) as i64),
                    1 => Some((doc % 7) as i64),
                    2 => Some(1_000_000_000 + (doc % 50) as i64),
                    3 => (doc % 101 != 0).then_some(doc as i64),
                    _ => Some(doc as i64 - 5),
                };
                if let Some(v) = v { d.add_i64(0, v); }
                w.add_document(&d, &schema);
            }
            let seg = w.write_to(&dir).unwrap();
            for d in (0..n).step_by(13) { seg.delete_doc(d).unwrap(); }
            let leaf = LeafReader::Segment(&seg);
            let values = NumericDv::open(&leaf, 0);
            let dels = leaf.deleted_bits();
            let deleted = dels.as_ref().map(|b| b.words());
            let score = |_: DocId, raw: f64| (1.2 * raw + 2.0).log10() as f32;
            for k in [1usize, 10, 100, 5000] {
                let walked = monotone_top_k_from_points(&values, n, k, deleted, score);
                let scanned = monotone_top_k(n, k, deleted, |doc| values.first(doc).unwrap_or(1.0), score);
                match pattern {
                    3 | 4 => assert!(walked.is_none(), "pattern {pattern}: a missing or negative value must take the scan"),
                    _ => assert_eq!(walked.as_ref(), Some(&scanned), "pattern {pattern} k={k}"),
                }
            }
        }
    }

    #[test]
    fn monotone_pruning_preserves_scores_ties_and_deletions() {
        let n = 20_003;
        let mut deleted = vec![0u64; (n as usize).div_ceil(64)];
        for doc in (0..n).step_by(7) { deleted[doc as usize / 64] |= 1 << (doc % 64); }
        for pattern in 0..4 {
            let value = |doc: u32| match pattern {
                0 => doc as f64,
                1 => (n - doc) as f64,
                2 => ((doc as u64 * 1234567) % 100000) as f64,
                _ => ((1u64 << 53) + doc as u64 % 100) as f64,
            };
            for modifier in [FieldValueModifier::Log1p, FieldValueModifier::Log2p, FieldValueModifier::Ln1p, FieldValueModifier::Ln2p] {
                for mode in [BoostMode::Multiply, BoostMode::Replace, BoostMode::Sum, BoostMode::Avg, BoostMode::Max, BoostMode::Min] {
                    for k in [1, 10, 100] {
                        let score = |_: u32, raw: f64| mode.combine(1.0, modifier.apply(1.2f32 as f64 * raw).min(100.0)) as f32;
                        let mut expected = TopKCollector::new(k);
                        let mut total = 0;
                        for doc in 0..n {
                            if deleted[doc as usize / 64] & (1 << (doc % 64)) != 0 { continue; }
                            total += 1;
                            expected.collect(doc, score(doc, value(doc)));
                        }
                        assert_eq!(monotone_top_k(n, k, Some(&deleted), value, score), (expected.into_sorted_vec(), total), "pattern={pattern} modifier={modifier:?} mode={mode:?} k={k}");
                    }
                }
            }
        }
    }

    #[test]
    fn monotone_bulk_retains_invalid_score_errors_and_fallbacks() {
        use crate::query::{docset::AllDocSet, scorer::ConstScorer};
        let make = |missing, min_score, factor, match_all| {
            let funcs = PreparedFunctions {
                funcs: vec![Prepared::Fvf { values: NumericDv::Empty, factor, modifier: FieldValueModifier::Log2p, missing: Some(missing) }],
                params: FunctionScoreParams { score_mode: ScoreMode::Avg, boost_mode: BoostMode::Multiply, max_boost: 100.0, min_score },
                invalid: Default::default(),
            };
            let scorer = FunctionScoreScorer::new(Box::new(ConstScorer::new(AllDocSet::new(20), 1.0)), funcs);
            if match_all { scorer.with_match_all(20) } else { scorer }
        };
        for missing in [-10.0, f64::NAN] {
            let mut scorer = make(missing, None, 1.2, true);
            assert!(scorer.collect_top_k_bulk(10, None).is_some());
            assert!(scorer.funcs.invalid.lock().is_some());
        }
        for (min_score, factor, match_all, k) in [(Some(0.0), 1.2, true, 10), (None, -1.0, true, 10), (None, 1.2, false, 10), (None, 1.2, true, 0)] {
            assert!(make(42.0, min_score, factor, match_all).collect_top_k_bulk(k, None).is_none());
        }
    }

    #[test]
    fn invalid_functions_survive_caps_and_score_modes() {
        for mode in [ScoreMode::Multiply, ScoreMode::Sum, ScoreMode::Avg, ScoreMode::First, ScoreMode::Max, ScoreMode::Min] {
            for boost in [BoostMode::Multiply, BoostMode::Replace, BoostMode::Sum, BoostMode::Avg, BoostMode::Max, BoostMode::Min] {
                for multi in [false, true] {
                    let mut funcs = vec![Prepared::Weight(f64::NAN)];
                    if multi { funcs.push(Prepared::Weight(1.0)); }
                    let prepared = PreparedFunctions {
                        funcs,
                        params: FunctionScoreParams { score_mode: mode, boost_mode: boost, max_boost: 2.0, min_score: None },
                        invalid: Default::default(),
                    };
                    assert!(prepared.score(0, 1.0).is_nan(), "{mode:?}/{boost:?}/multi={multi}");
                    assert!(prepared.factor(0).0.is_nan(), "explanation {mode:?}/{boost:?}/multi={multi}");
                }
            }
        }
    }

    #[test]
    fn modifiers_and_decays() {
        assert!((FieldValueModifier::Log1p.apply(9.0) - 1.0).abs() < 1e-12);
        assert!((FieldValueModifier::Ln2p.apply(0.0) - 2f64.ln()).abs() < 1e-12);
        assert_eq!(FieldValueModifier::Square.apply(3.0), 9.0);
        assert_eq!(FieldValueModifier::Reciprocal.apply(4.0), 0.25);
        // At distance == scale every decay kind yields `decay`.
        for k in [DecayKind::Gauss, DecayKind::Exp, DecayKind::Linear] {
            assert!((k.eval(10.0, 10.0, 0.5) - 0.5).abs() < 1e-9, "{k:?}");
            assert!((k.eval(0.0, 10.0, 0.5) - 1.0).abs() < 1e-9);
        }
        assert_eq!(DecayKind::Linear.eval(100.0, 10.0, 0.5), 0.0);
        let r = random_score(42, 7);
        assert!((0.0..1.0).contains(&r));
        assert_eq!(r, random_score(42, 7));
        assert_ne!(r, random_score(43, 7));
    }
}
