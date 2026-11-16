//! Scripts inside searches: `script` / `script_score` queries,
//! `function_score` script functions (as [`CustomQuery`] nodes with their
//! own scorers), `script_fields`, `_script` sort keys, runtime-field
//! queries and aggregation value sources.

use super::doc::Kind;
use super::value::Val;
use super::{compile_json, env, index_env, report_error, CompiledScript, Executor, RuntimeFields, ScriptContext};
use crate::error::{EsError, Result};
use crate::mapping::FieldLookup;
use crate::query::dsl::QueryContext;
use parking_lot::Mutex;
use rustic_core::query::{BoostMode, CustomQuery, FactorAcc, PreparedScoreFunction, DocSet, Explanation, LeafReader, Query, ScoreFunction, ScoreMode, Scorer, SearchContext, TERMINATED};
use rustic_core::types::DocId;
use serde_json::{Map, Value};
use std::sync::Arc;

type ErrSlot = Arc<Mutex<Option<EsError>>>;

/// Everything a script node needs to build per-leaf executors.
#[derive(Clone)]
pub struct Binding {
    pub script: CompiledScript,
    pub lookup: Arc<FieldLookup>,
    pub runtime: Option<Arc<RuntimeFields>>,
    pub errors: ErrSlot,
    /// The search deadline the script polls (`None` outside searches with a timeout).
    pub cancel: Option<Arc<super::runtime::Cancel>>,
    /// The search task its loops poll (executors may be made on search pool
    /// workers, which do not carry the search's thread-local token).
    pub task: Option<Arc<rustic_core::cancel::Token>>,
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Script({:?})", self.script.source())
    }
}

impl Binding {
    pub fn new(script: CompiledScript, ctx: &QueryContext) -> Binding {
        let (lookup, runtime) = match index_env(Some(ctx.index_name)) {
            Some(ie) => (ie.lookup, ie.runtime),
            None => (Arc::new(ctx.lookup.clone()), None),
        };
        let errors = env().map(|e| e.error.clone()).unwrap_or_default();
        let cancel = env().and_then(|e| e.cancel.clone());
        let task = rustic_core::cancel::current();
        Binding { script, lookup, runtime, errors, cancel, task }
    }

    /// Executor bound to `leaf`.
    pub fn executor(&self, leaf: &LeafReader<'_>) -> Executor {
        let mut e = self.script.executor(Some(self.lookup.clone()), self.runtime.clone());
        // SAFETY: executors built here live inside scorers / per-leaf loops
        // that never outlive the search borrowing `leaf` (CustomQuery contract).
        unsafe { e.bind_leaf(leaf) };
        e.rt.cancel = self.cancel.clone();
        e.rt.task = self.task.clone();
        e
    }

    fn fail(&self, e: &super::ScriptErr) {
        if super::runtime::Cancel::absorb(&self.cancel, e) {
            return;
        }
        report_error(&self.errors, self.script.error(e));
    }
}

fn script_desc(s: &CompiledScript) -> String {
    format!("script score function, computed with script:\"Script{{type=inline, lang='painless', idOrCode='{}', options={{}}, params={}}}\"", s.source(), s.params)
}

// ---------------------------------------------------------------------------
// `script` query (filter).

#[derive(Debug)]
struct ScriptFilter {
    b: Binding,
    boost: f32,
}

/// Docs of `leaf` outside the nested scope the scorer is being built in
/// (empty: the leaf has no nested docs). A top-level script never runs on
/// hidden nested child docs (ES ANDs a non-nested filter in front of the
/// query); inside a `nested` query it runs on exactly that path's docs
/// (the nested type filter of `ToParentBlockJoinQuery`'s child query).
fn out_of_scope(leaf: &LeafReader<'_>) -> Vec<bool> {
    let Some(info) = leaf.nested_info() else { return Vec::new() };
    let scope = rustic_core::query::nested_scope_ord();
    (0..leaf.num_docs()).map(|d| info.ord(d) != scope).collect()
}

/// Skips the docs outside the scorer's nested scope ([`out_of_scope`]).
struct RootOnly {
    inner: Box<dyn Scorer>,
    nested: Vec<bool>,
}

impl RootOnly {
    fn wrap(inner: Box<dyn Scorer>, leaf: &LeafReader<'_>) -> Box<dyn Scorer> {
        if leaf.nested_info().is_none() {
            return inner;
        }
        let nested = out_of_scope(leaf);
        Box::new(RootOnly { inner, nested })
    }
    fn skip(&mut self, mut d: DocId) -> DocId {
        while d != TERMINATED && self.nested.get(d as usize).copied().unwrap_or(false) {
            d = self.inner.advance();
        }
        d
    }
}

impl DocSet for RootOnly {
    fn advance(&mut self) -> DocId {
        let d = self.inner.advance();
        self.skip(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        let d = self.inner.seek(target);
        self.skip(d)
    }
    fn doc(&self) -> DocId {
        self.inner.doc()
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for RootOnly {
    fn score(&mut self) -> f32 {
        self.inner.score()
    }
}

struct ScriptFilterScorer {
    exec: Executor,
    b: Binding,
    num_docs: DocId,
    cur: DocId,
    boost: f32,
    /// Hidden nested child docs, which a top-level query never sees (ES
    /// ANDs a non-nested filter in front of the script).
    nested: Vec<bool>,
}

impl ScriptFilterScorer {
    fn find_from(&mut self, mut d: DocId) -> DocId {
        while d < self.num_docs {
            if self.nested.get(d as usize).copied().unwrap_or(false) {
                d += 1;
                continue;
            }
            match super::execute_filter(&mut self.exec, d) {
                Ok(true) => {
                    self.cur = d;
                    return d;
                }
                Ok(false) => d += 1,
                Err(e) => {
                    self.b.fail(&e);
                    break;
                }
            }
        }
        self.cur = TERMINATED;
        TERMINATED
    }
}

impl DocSet for ScriptFilterScorer {
    fn advance(&mut self) -> DocId {
        let next = if self.cur == TERMINATED { 0 } else { self.cur + 1 };
        if self.cur == TERMINATED && self.num_docs == 0 {
            return TERMINATED;
        }
        self.find_from(next)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.cur != TERMINATED && self.cur >= target {
            return self.cur;
        }
        self.find_from(target)
    }
    fn doc(&self) -> DocId {
        self.cur
    }
    fn size_hint(&self) -> u32 {
        self.num_docs
    }
}

impl Scorer for ScriptFilterScorer {
    fn score(&mut self) -> f32 {
        self.boost
    }
}

impl CustomQuery for ScriptFilter {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let exec = self.b.executor(leaf);
        // `cur` starts "before the first doc": advance() moves to 0.
        let nested = out_of_scope(leaf);
        let mut s = ScriptFilterScorer { exec, b: self.b.clone(), num_docs: leaf.num_docs(), cur: TERMINATED, boost: self.boost, nested };
        s.cur = TERMINATED;
        Some(Box::new(FirstAdvance { inner: s, started: false }))
    }
    fn is_constant_score(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "script"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, format!("ScriptQuery(script=Script{{type=inline, lang='painless', idOrCode='{}'}})", self.b.script.source()))
    }
}

/// Distinguishes "not started" from "exhausted" for the filter scorer.
struct FirstAdvance {
    inner: ScriptFilterScorer,
    started: bool,
}

impl DocSet for FirstAdvance {
    fn advance(&mut self) -> DocId {
        if !self.started {
            self.started = true;
            return self.inner.find_from(0);
        }
        if self.inner.cur == TERMINATED {
            return TERMINATED;
        }
        let n = self.inner.cur + 1;
        self.inner.find_from(n)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.started && (self.inner.cur == TERMINATED || self.inner.cur >= target) {
            return self.inner.cur;
        }
        self.started = true;
        self.inner.find_from(target)
    }
    fn doc(&self) -> DocId {
        if self.started { self.inner.cur } else { TERMINATED }
    }
    fn size_hint(&self) -> u32 {
        self.inner.num_docs
    }
}

impl Scorer for FirstAdvance {
    fn score(&mut self) -> f32 {
        self.inner.boost
    }
}

fn boost_of(o: &Map<String, Value>) -> Result<f32> {
    match o.get("boost") {
        None => Ok(1.0),
        Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or(1.0) as f32),
        Some(Value::String(s)) => s.parse().map_err(|_| EsError::parsing("[boost] must be a number")),
        Some(_) => Err(EsError::parsing("[boost] must be a number")),
    }
}

/// `{"script": {"script": ..., "boost": ..}}`.
pub fn script_query(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[script] query malformed, no start_object after query name"))?;
    let s = o.get("script").ok_or_else(|| EsError::parsing("script must be provided with a [script] query").with("line", 1).with("col", 1))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "script" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[script] query does not support [{k}]")).with("line", 1).with("col", 1));
        }
    }
    let script = super::compile_search_script(s, ScriptContext::Filter)?;
    Ok(Query::Custom(Arc::new(ScriptFilter { b: Binding::new(script, ctx), boost: boost_of(o)? })))
}

// ---------------------------------------------------------------------------
// `script_score` query.

#[derive(Debug)]
struct ScriptScore {
    inner: Query,
    b: Binding,
    min_score: Option<f32>,
    boost: f32,
    geo_top_k: Option<super::search_geo_topk::Plan>,
    numeric_top_k: Option<super::search_numeric_topk::Plan>,
    vector_score: Option<Arc<super::search_vector_score::Plan>>,
    /// The inner query is not `match_all`: a bulk vector scan visits its
    /// matches only.
    vector_filtered: bool,
}

struct ScriptScoreScorer {
    inner: Box<dyn Scorer>,
    exec: Executor,
    b: Binding,
    min_score: Option<f32>,
    boost: f32,
    cur: f32,
    geo_top_k: Option<super::search_geo_topk::Bound<'static>>,
    numeric_top_k: Option<super::search_numeric_topk::Bound>,
    vector_score: Option<Arc<super::search_vector_score::Plan>>,
    vector_bulk: Option<super::search_vector_score::Bound>,
    /// A second instance of the inner query for a filtered bulk scan (the
    /// scorer's own `inner` stays untouched for the ordinary fallback).
    vector_docs: Option<Box<dyn Scorer>>,
}

impl ScriptScoreScorer {
    fn evaluate(&mut self, doc: DocId, query_score: f32) -> Option<f32> {
        self.exec.set_doc(doc);
        let value = self.vector_score.as_ref().and_then(|plan| plan.score(&mut self.exec));
        match value.map(Ok).unwrap_or_else(|| super::execute_score(&mut self.exec, doc, query_score)) {
            Ok(v) if v < 0.0 || v.is_nan() => {
                let e = super::error::illegal_arg(format!("script score function must not produce negative scores, but got: [{}]", super::value::java_double_string(v)));
                report_error(&self.b.errors, EsError::new(400, "illegal_argument_exception", e.msg.clone()));
                None
            }
            Ok(v) => Some(v as f32 * self.boost),
            Err(e) => { self.b.fail(&e); None }
        }
    }

    fn settle(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                return d;
            }
            let qs = self.inner.score();
            match self.evaluate(d, qs) {
                Some(s) => {
                    match self.min_score {
                        Some(m) if s < m => d = self.inner.advance(),
                        _ => {
                            self.cur = s;
                            return d;
                        }
                    }
                }
                None => return TERMINATED,
            }
        }
    }
}

impl DocSet for ScriptScoreScorer {
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

impl Scorer for ScriptScoreScorer {
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<rustic_core::query::collector::ScoredDoc>, u64)> {
        if k == 0 { return None; }
        if let Some(plan) = &self.vector_bulk {
            let docs = self.vector_docs.take().map(|mut matches| {
                let mut docs = Vec::new();
                let mut d = matches.advance();
                while d != TERMINATED { docs.push(d); d = matches.advance(); }
                docs
            });
            if let Some(result) = plan.collect(docs.as_deref(), k, deleted, self.boost) { return Some(result); }
        }
        if let Some(plan) = self.geo_top_k.take() {
            let result = plan.collect(k, deleted, self.boost, |doc| self.evaluate(doc, 1.0));
            self.geo_top_k = Some(plan);
            return Some(result);
        }
        let plan = self.numeric_top_k.take()?;
        // Best value first from the column's points index when every value
        // is in the script's monotone domain; else every doc's value.
        let result = match rustic_core::query::function_score::monotone_top_k_from_points(&plan.values, plan.num_docs, k, deleted,
            |doc, _| self.evaluate(doc, 1.0).unwrap_or(f32::NAN)) {
            Some(result) => result,
            None => rustic_core::query::function_score::monotone_top_k(plan.num_docs, k, deleted,
                |doc| plan.values.first(doc).unwrap_or(f64::NEG_INFINITY),
                |doc, _| self.evaluate(doc, 1.0).unwrap_or(f32::NAN)),
        };
        self.numeric_top_k = Some(plan);
        Some(result)
    }

    fn score(&mut self) -> f32 {
        self.cur
    }
}

impl CustomQuery for ScriptScore {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let inner = RootOnly::wrap(self.inner.scorer(leaf, ctx)?, leaf);
        let exec = self.b.executor(leaf);
        let geo_top_k = self.geo_top_k.as_ref().and_then(|plan| plan.bind(leaf));
        let numeric_top_k = self.numeric_top_k.as_ref().and_then(|plan| plan.bind(leaf));
        let vector_bulk = self.vector_score.as_ref().and_then(|plan| plan.bind(&self.b.lookup, leaf));
        let vector_docs = if vector_bulk.is_some() && self.vector_filtered {
            match self.inner.scorer(leaf, ctx) { Some(s) => Some(RootOnly::wrap(s, leaf)), None => return None }
        } else { None };
        // A filtered bulk scan needs the inner matches; without them, score per doc.
        let vector_bulk = vector_bulk.filter(|_| !self.vector_filtered || vector_docs.is_some());
        // No top-k plan for a large sealed segment: its doc ranges score as
        // parts (each its own executor) on the search pool, see
        // `FunctionScoreScript::scorer`.
        if geo_top_k.is_none() && numeric_top_k.is_none() && vector_bulk.is_none() && matches!(leaf, LeafReader::Segment(_))
            && leaf.num_docs() >= rustic_core::query::scorer::part_split_min_docs() {
            let ranges = rustic_core::query::scorer::doc_range_parts(leaf.num_docs());
            if ranges.len() > 1 {
                drop(inner);
                let mut parts: Vec<Box<dyn Scorer>> = Vec::with_capacity(ranges.len());
                for (lo, hi) in ranges {
                    let inner: Box<dyn Scorer> = Box::new(rustic_core::query::scorer::DocRangeScorer::new(RootOnly::wrap(self.inner.scorer(leaf, ctx)?, leaf), lo, hi));
                    parts.push(Box::new(ScriptScoreScorer { inner, exec: self.b.executor(leaf), b: self.b.clone(), min_score: self.min_score, boost: self.boost, cur: 0.0, geo_top_k: None, numeric_top_k: None, vector_score: self.vector_score.clone(), vector_bulk: None, vector_docs: None }));
                }
                return Some(Box::new(rustic_core::query::scorer::PartedScorer::new(parts)));
            }
        }
        Some(Box::new(ScriptScoreScorer { inner, exec, b: self.b.clone(), min_score: self.min_score, boost: self.boost, cur: 0.0, geo_top_k, numeric_top_k, vector_score: self.vector_score.clone(), vector_bulk, vector_docs }))
    }
    fn children(&self) -> Vec<&Query> {
        vec![&self.inner]
    }
    fn name(&self) -> &'static str {
        "script_score"
    }
    fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        let sub = self.inner.explain(leaf, ctx, doc);
        Explanation::new(score, script_desc(&self.b.script), vec![Explanation::new(sub.value, "_score: ", vec![sub])])
    }
}

/// `{"script_score": {"query": .., "script": .., "min_score": .., "boost": ..}}`.
pub fn script_score_query(body: &Value, ctx: &QueryContext, parse: &dyn Fn(&Value, &QueryContext) -> Result<Query>) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[script_score] query malformed, no start_object after query name"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "script" | "query" | "min_score" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[script_score] query does not support [{k}]")).with("line", 1).with("col", 1));
        }
    }
    let q = o.get("query").ok_or_else(|| EsError::parsing("[script_score] requires 'query' field").with("line", 1).with("col", 1))?;
    let s = o.get("script").ok_or_else(|| EsError::parsing("[script_score] requires 'script' field").with("line", 1).with("col", 1))?;
    let inner = parse(q, ctx)?;
    let script = super::compile_search_script(s, ScriptContext::Score)?;
    let min_score = o.get("min_score").and_then(num_f32);
    let b = Binding::new(script, ctx);
    let boost = boost_of(o)?;
    let bulk_ok = min_score.is_none() && boost.is_finite() && boost > 0.0 && b.runtime.is_none() && b.cancel.is_none();
    let eligible = matches!(inner, Query::MatchAll) && bulk_ok;
    let geo_top_k = if eligible {
        super::search_geo_topk::Plan::parse(&b.script, &b.lookup)
    } else { None };
    let numeric_top_k = if eligible && geo_top_k.is_none() {
        super::search_numeric_topk::Plan::parse(&b.script, &b.lookup)
    } else { None };
    // A pure vector-similarity script ignores `_score`: any inner query's
    // matches can be scanned in bulk (so_vector's filtered exact searches).
    let vector_score = if bulk_ok && geo_top_k.is_none() && numeric_top_k.is_none() { super::search_vector_score::Plan::parse(&b.script).map(Arc::new) } else { None };
    let vector_filtered = !matches!(inner, Query::MatchAll);
    Ok(Query::Custom(Arc::new(ScriptScore { inner, b, min_score, boost, geo_top_k, numeric_top_k, vector_score, vector_filtered })))
}

/// The previous rescore pass's window with its scores (ES
/// `KnnScoreDocQuery` over the top N, as `ScriptRescorer` builds it).
#[derive(Debug)]
struct PrevPass;

impl CustomQuery for PrevPass {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let docs = ctx.prev_pass.get(&leaf.id())?;
        Some(Box::new(rustic_core::query::ScoredDocs::new(docs.clone())))
    }
    fn name(&self) -> &'static str {
        "prev_pass"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, "previous pass score")
    }
}

/// `{"script": {"script": ..}}` rescorer (`ScriptRescorerBuilder`): a
/// `script_score` over the previous pass's top window, its `_score` being
/// the hit's current score.
pub fn script_rescore_query(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let s = o.get("script").ok_or_else(|| EsError::parsing("[script] rescorer requires a [script]"))?;
    let script = super::compile_search_script(s, ScriptContext::Score)?;
    Ok(Query::Custom(Arc::new(ScriptScore { inner: Query::Custom(Arc::new(PrevPass)), b: Binding::new(script, ctx), min_score: None, boost: 1.0, geo_top_k: None, numeric_top_k: None, vector_score: None, vector_filtered: true })))
}

// ---------------------------------------------------------------------------
// `function_score` with script functions (evaluated here; the core
// function_score has no script hook).

#[derive(Debug)]
enum Func {
    Core(ScoreFunction),
    Script { b: Binding, weight: Option<f32>, filter: Option<Query> },
}

#[derive(Debug)]
struct FunctionScoreScript {
    compound_top_k: Option<super::search_compound_topk::Plan>,
    inner: Query,
    funcs: Vec<Func>,
    score_mode: ScoreMode,
    boost_mode: BoostMode,
    max_boost: f32,
    min_score: Option<f32>,
    boost: f32,
    filters: Vec<Query>,
}

enum Prep {
    Core(PreparedScoreFunction),
    Filtered { bits: Vec<u64>, inner: Box<Prep> },
    Script { exec: Box<Executor>, weight: f64 },
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
    #[inline]
    fn eval(&mut self, doc: DocId, score: f32) -> std::result::Result<Option<f64>, Box<super::ScriptErr>> {
        Ok(match self {
            Prep::Core(p) => p.value(doc),
            Prep::Filtered { bits, inner } => {
                if has_bit(bits, doc) {
                    inner.eval(doc, score)?
                } else {
                    None
                }
            }
            Prep::Script { exec, weight } => Some(super::execute_score(exec, doc, score)? * *weight),
        })
    }

    /// The function's `weight` (1 unless set): what `avg` divides by.
    fn weight(&self) -> f64 {
        match self {
            Prep::Core(p) => p.weight_of(),
            Prep::Filtered { inner, .. } => inner.weight(),
            Prep::Script { weight, .. } => *weight,
        }
    }
}

struct FsScorer {
    compound_top_k: Option<super::search_compound_topk::Bound>,
    /// The leaf's docs this scorer covers (a part of a large leaf, see
    /// `PartedScorer`; `(0, u32::MAX)`: all).
    range: (DocId, DocId),
    inner: Box<dyn Scorer>,
    preps: Vec<Prep>,
    script_errs: Vec<Option<Binding>>,
    score_mode: ScoreMode,
    boost_mode: BoostMode,
    max_boost: f32,
    min_score: Option<f32>,
    cur: f32,
    /// Where an invalid (negative or NaN) score is reported, as the generic
    /// function_score scorer reports it.
    invalid: std::sync::Arc<parking_lot::Mutex<Option<(DocId, f32)>>>,
}

// SAFETY: see `Executor` (single-owner reference-counted state).
unsafe impl Send for FsScorer {}

impl FsScorer {
    /// The score of `d`: the inner query's score (its `boost` and any
    /// enclosing boost folded in, as ES's `FunctionScoreQuery` passes the
    /// boost to its sub-query's weight) met by `boost_mode` with the
    /// applicable functions combined
    /// as the generic scorer combines them ([`FactorAcc`]: `avg` over the
    /// sum of the weights, `first` evaluating no later function), capped
    /// at `max_boost`, then met with the query score by `boost_mode`.
    fn compute(&mut self, d: DocId) -> std::result::Result<f32, (usize, Box<super::ScriptErr>)> {
        let q = self.inner.score();
        let mut acc = FactorAcc::new(self.score_mode);
        for (i, p) in self.preps.iter_mut().enumerate() {
            match p.eval(d, q) {
                Ok(Some(v)) => {
                    if !acc.push(v, p.weight()) {
                        break;
                    }
                }
                Ok(None) => {}
                Err(e) => return Err((i, e)),
            }
        }
        let s = self.boost_mode.combine(q as f64, acc.finish(self.max_boost)) as f32;
        if !(s >= 0.0) {
            let mut slot = self.invalid.lock();
            if slot.is_none() {
                *slot = Some((d, s));
            }
        }
        Ok(s)
    }

    fn settle(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                return d;
            }
            match self.compute(d) {
                Ok(s) => match self.min_score {
                    Some(m) if s < m => d = self.inner.advance(),
                    _ => {
                        self.cur = s;
                        return d;
                    }
                },
                Err((i, e)) => {
                    if let Some(Some(b)) = self.script_errs.get(i) {
                        b.fail(&e);
                    }
                    return TERMINATED;
                }
            }
        }
    }
}

impl DocSet for FsScorer {
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

impl Scorer for FsScorer {
    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<rustic_core::query::collector::ScoredDoc>, u64)> {
        self.collect_top_k_bulk_min(k, deleted, f32::NEG_INFINITY)
    }
    /// The other leaves' k-th best score (`floor`) rules out blocks before
    /// any of their docs is evaluated (72 small segments otherwise each
    /// rebuild their own threshold from scratch).
    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<rustic_core::query::collector::ScoredDoc>, u64)> {
        if k == 0 { return None; }
        let plan = self.compound_top_k.take()?;
        let range = self.range;
        let result = plan.collect(k, deleted, floor, range, |doc| match self.compute(doc) {
            Ok(score) => Some(score),
            Err((i, error)) => {
                if let Some(Some(binding)) = self.script_errs.get(i) { binding.fail(&error); }
                None
            }
        });
        self.compound_top_k = Some(plan);
        Some(result)
    }
    fn score(&mut self) -> f32 {
        self.cur
    }
}

impl FunctionScoreScript {
    /// The scorer over `leaf`'s docs, or over `range` of them only.
    /// `boost`: this query's own boost times any enclosing one, folded into
    /// the inner query's scores (never applied to the combined score).
    fn build(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, range: Option<(DocId, DocId)>, boost: f32) -> Option<Box<dyn Scorer>> {
        let inner = RootOnly::wrap(self.inner.scorer_b(leaf, ctx, boost)?, leaf);
        let inner: Box<dyn Scorer> = match range {
            Some((lo, hi)) => Box::new(rustic_core::query::scorer::DocRangeScorer::new(inner, lo, hi)),
            None => inner,
        };
        let mut preps = Vec::new();
        let mut errs = Vec::new();
        for f in &self.funcs {
            match f {
                Func::Core(c) => {
                    preps.push(Prep::Core(PreparedScoreFunction::build(c, leaf, ctx)));
                    errs.push(None);
                }
                Func::Script { b, weight, filter } => {
                    let p = Prep::Script { exec: Box::new(b.executor(leaf)), weight: weight.unwrap_or(1.0) as f64 };
                    preps.push(match filter {
                        Some(q) => Prep::Filtered { bits: bits_of(q, leaf, ctx), inner: Box::new(p) },
                        None => p,
                    });
                    errs.push(Some(b.clone()));
                }
            }
        }
        Some(Box::new(FsScorer {
            compound_top_k: self.compound_top_k.as_ref().filter(|_| boost == 1.0).and_then(|plan| plan.bind(leaf)),
            range: range.unwrap_or((0, DocId::MAX)),
            inner,
            preps,
            script_errs: errs,
            score_mode: self.score_mode,
            boost_mode: self.boost_mode,
            max_boost: self.max_boost,
            min_score: self.min_score,
            cur: 0.0,
            invalid: ctx.invalid_score.clone(),
        }))
    }
}

impl CustomQuery for FunctionScoreScript {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        self.scorer_b(leaf, ctx, 1.0)
    }
    /// ES folds the function_score's boost (and an enclosing one) into its
    /// sub-query's weight: the functions and `boost_mode` combine with the
    /// boosted query score, and `min_score` reads that combination.
    fn absorbs_boost(&self) -> bool {
        true
    }
    fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, outer: f32) -> Option<Box<dyn Scorer>> {
        let boost = self.boost * outer;
        // A large sealed segment scores its doc ranges as parts (each its
        // own script executors, and its compound top-k plan over its range
        // only), collected on the search pool: a merged segment of millions
        // of docs otherwise runs its scripts on one thread (geonames'
        // `expression`, whose bounds rule nothing out, scans every doc).
        if matches!(leaf, LeafReader::Segment(_)) && leaf.num_docs() >= rustic_core::query::scorer::part_split_min_docs() {
            let ranges = rustic_core::query::scorer::doc_range_parts(leaf.num_docs());
            if ranges.len() > 1 {
                let mut parts = Vec::with_capacity(ranges.len());
                for r in ranges {
                    parts.push(self.build(leaf, ctx, Some(r), boost)?);
                }
                return Some(Box::new(rustic_core::query::scorer::PartedScorer::new(parts)));
            }
        }
        self.build(leaf, ctx, None, boost)
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

/// A number or a numeric string, as the generic `function_score` parser
/// reads `weight`, `max_boost` and `min_score`.
fn num_f32(v: &Value) -> Option<f32> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).map(|x| x as f32)
}

/// Does this `function_score` body use a `script_score` function?
pub fn function_score_has_script(o: &Map<String, Value>) -> bool {
    o.contains_key("script_score") || o.get("functions").and_then(Value::as_array).is_some_and(|fs| fs.iter().any(|f| f.get("script_score").is_some()))
}

/// `function_score` with at least one `script_score` function. `parse` is
/// the DSL's query parser, `parse_fn` its parser for non-script functions.
pub fn function_score_query(
    o: &Map<String, Value>,
    ctx: &QueryContext,
    parse: &dyn Fn(&Value, &QueryContext) -> Result<Query>,
    parse_fn: &dyn Fn(&Map<String, Value>, &QueryContext) -> Result<Option<ScoreFunction>>,
) -> Result<Query> {
    let inner = match o.get("query") {
        Some(q) => parse(q, ctx)?,
        None => Query::MatchAll,
    };
    let mut funcs = Vec::new();
    let mut filters = Vec::new();
    let one = |fo: &Map<String, Value>, funcs: &mut Vec<Func>, filters: &mut Vec<Query>| -> Result<()> {
        if let Some(ss) = fo.get("script_score") {
            let s = ss.get("script").ok_or_else(|| EsError::parsing("script_score function requires a [script]").with("line", 1).with("col", 1))?;
            let script = super::compile_search_script(s, ScriptContext::Score)?;
            let weight = fo.get("weight").and_then(num_f32);
            let filter = match fo.get("filter") {
                Some(f) => {
                    let q = parse(f, ctx)?;
                    filters.push(q.clone());
                    Some(q)
                }
                None => None,
            };
            funcs.push(Func::Script { b: Binding::new(script, ctx), weight, filter });
        } else if let Some(f) = parse_fn(fo, ctx)? {
            funcs.push(Func::Core(f));
        }
        Ok(())
    };
    if let Some(fs) = o.get("functions").and_then(Value::as_array) {
        for f in fs {
            let fo = f.as_object().ok_or_else(|| EsError::parsing("[functions] entries must be objects"))?;
            one(fo, &mut funcs, &mut filters)?;
        }
    }
    let mut top = Map::new();
    for k in ["field_value_factor", "random_score", "gauss", "exp", "linear", "script_score", "weight"] {
        if let Some(v) = o.get(k) {
            top.insert(k.to_string(), v.clone());
        }
    }
    if !top.is_empty() {
        one(&top, &mut funcs, &mut filters)?;
    }
    let score_mode = match o.get("score_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => ScoreMode::Multiply,
        "sum" => ScoreMode::Sum,
        "avg" => ScoreMode::Avg,
        "first" => ScoreMode::First,
        "max" => ScoreMode::Max,
        "min" => ScoreMode::Min,
        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.FunctionScoreQuery.ScoreMode.{}", other.to_ascii_uppercase()))),
    };
    let boost_mode = match o.get("boost_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => BoostMode::Multiply,
        "replace" => BoostMode::Replace,
        "sum" => BoostMode::Sum,
        "avg" => BoostMode::Avg,
        "max" => BoostMode::Max,
        "min" => BoostMode::Min,
        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.CombineFunction.{}", other.to_ascii_uppercase()))),
    };
    let max_boost = o.get("max_boost").and_then(num_f32).unwrap_or(f32::MAX);
    let min_score = o.get("min_score").and_then(num_f32);
    let boost = boost_of(o)?;
    let compound_top_k = if matches!(inner, Query::MatchAll) && min_score.is_none() && boost == 1.0
        && max_boost == f32::MAX && matches!(boost_mode, BoostMode::Multiply | BoostMode::Replace) {
        match funcs.as_slice() {
            [Func::Script { b, weight: None, filter: None }] if b.runtime.is_none() && b.cancel.is_none() =>
                super::search_compound_topk::Plan::parse(&b.script, &b.lookup),
            _ => None,
        }
    } else { None };
    Ok(Query::Custom(Arc::new(FunctionScoreScript { inner, funcs, score_mode, boost_mode, max_boost, min_score, boost, filters, compound_top_k })))
}

// ---------------------------------------------------------------------------
// Runtime-field queries.

#[derive(Debug)]
enum Pred {
    Exists,
    Terms(Vec<Value>),
    Range { lo: Option<(f64, bool)>, hi: Option<(f64, bool)>, slo: Option<(String, bool)>, shi: Option<(String, bool)> },
    Prefix(String, bool),
    Regex(regex::Regex),
    /// Encoded `geo_shape` values intersecting a (normalized, quantized) geometry.
    Geo(Arc<rustic_core::fieldtypes::geo2d::ShapeQuery>),
}

struct RuntimeFilter {
    rt: Arc<RuntimeFields>,
    idx: usize,
    lookup: Arc<FieldLookup>,
    pred: Pred,
    boost: f32,
    errors: ErrSlot,
    cancel: Option<Arc<super::runtime::Cancel>>,
}

fn val_matches(p: &Pred, v: &Val, ty: super::RtType) -> bool {
    match p {
        Pred::Exists => true,
        Pred::Terms(ts) => ts.iter().any(|t| match (ty, v) {
            (super::RtType::Keyword | super::RtType::Ip, Val::Str(s)) => crate::query::dsl::scalar_string(t) == **s,
            (super::RtType::Boolean, Val::Bool(b)) => t.as_bool() == Some(*b) || t.as_str() == Some(if *b { "true" } else { "false" }),
            (_, v) => match (v.as_f64(), t.as_f64().or_else(|| t.as_str().and_then(|s| s.parse().ok()))) {
                (Some(a), Some(b)) => a == b,
                _ => matches!(v, Val::Date(m) if t.as_i64() == Some(*m)),
            },
        }),
        Pred::Range { lo, hi, slo, shi } => {
            if let Val::Str(s) = v {
                let s: &str = s;
                return slo.as_ref().map(|(b, inc)| if *inc { s >= b.as_str() } else { s > b.as_str() }).unwrap_or(true)
                    && shi.as_ref().map(|(b, inc)| if *inc { s <= b.as_str() } else { s < b.as_str() }).unwrap_or(true);
            }
            let x = match v {
                Val::Date(m) => *m as f64,
                other => match other.as_f64() {
                    Some(x) => x,
                    None => return false,
                },
            };
            lo.map(|(b, inc)| if inc { x >= b } else { x > b }).unwrap_or(true) && hi.map(|(b, inc)| if inc { x <= b } else { x < b }).unwrap_or(true)
        }
        Pred::Prefix(p, ci) => match v {
            Val::Str(s) => {
                if *ci {
                    s.to_lowercase().starts_with(&p.to_lowercase())
                } else {
                    s.starts_with(p.as_str())
                }
            }
            _ => false,
        },
        Pred::Regex(r) => match v {
            Val::Str(s) => r.is_match(s),
            _ => false,
        },
        // A runtime `geo_shape` value related as ES relates it (the value
        // as a shape doc value, the query as its `Component2D`).
        Pred::Geo(q) => match v {
            Val::Str(s) => rustic_core::fieldtypes::shape::decode(s).is_some_and(|g| crate::fieldtypes::geo_shape_query::geometry_matches(&g, q) == Some(true)),
            _ => false,
        },
    }
}

/// A geo query (`geo_bounding_box`, `geo_distance`, ...) on a `geo_shape`
/// runtime field: docs whose shape intersects `geometry` (GeoJSON / WKT).
/// `Ok(None)` when `field` is not a `geo_shape` runtime field.
pub fn runtime_geo_query(field: &str, geometry: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    use rustic_core::fieldtypes::shape;
    let Some(env) = env() else { return Ok(None) };
    let Some(ie) = env.per_index.borrow().get(ctx.index_name).cloned() else { return Ok(None) };
    let Some(rt) = ie.runtime else { return Ok(None) };
    let Some(idx) = rt.index_of(field) else { return Ok(None) };
    if rt.fields[idx].ty != super::RtType::GeoShape {
        return Ok(None);
    }
    let g = shape::parse_query_shape(geometry).map_err(EsError::illegal_argument)?;
    runtime_shape_query(field, &g, crate::fieldtypes::geo_shape_query::circle_of(geometry), shape::SpatialRelation::Intersects, ctx)
}

/// A `geo_shape` query (`rel`) of `g` (or `circle`, a circle query) on a
/// `geo_shape` runtime field:
/// each doc's emitted values related as ES relates a runtime shape (the
/// values as a shape doc value). `Ok(None)` when `field` is not one.
pub fn runtime_shape_query(field: &str, g: &rustic_core::fieldtypes::shape::Geometry, circle: Option<rustic_core::fieldtypes::geo2d::LatLonGeometry>, rel: rustic_core::fieldtypes::shape::SpatialRelation, ctx: &QueryContext) -> Result<Option<Query>> {
    let Some(env) = env() else { return Ok(None) };
    let Some(ie) = env.per_index.borrow().get(ctx.index_name).cloned() else { return Ok(None) };
    let Some(rt) = ie.runtime else { return Ok(None) };
    let Some(idx) = rt.index_of(field) else { return Ok(None) };
    if rt.fields[idx].ty != super::RtType::GeoShape {
        return Ok(None);
    }
    let q = match circle {
        Some(c) => rustic_core::fieldtypes::geo2d::ShapeQuery::new(&[c], crate::fieldtypes::geo_shape_query::query_relation(rel)),
        None => {
            // `GeoShapeUtils.toLuceneGeometry`: as for an indexed field
            // (a WITHIN of lines is refused).
            let geoms = rustic_core::fieldtypes::geo2d::query_geometries(&rustic_core::fieldtypes::shape::normalize_geo(g), rel == rustic_core::fieldtypes::shape::SpatialRelation::Within)
                .map_err(|e| EsError::new(400, "query_shard_exception", format!("Exception creating query on Field [{field}] {e}")))?;
            rustic_core::fieldtypes::geo2d::ShapeQuery::new(&geoms, crate::fieldtypes::geo_shape_query::query_relation(rel))
        }
    };
    let q = Arc::new(q);
    let rf = RuntimeFilter { rt, idx, lookup: ie.lookup.clone(), pred: Pred::Geo(q), boost: 1.0, errors: env.error.clone(), cancel: env.cancel.clone() };
    Ok(Some(Query::Custom(Arc::new(RuntimeFilterQuery::from(rf)))))
}

struct RuntimeFilterScorer {
    exec: Executor,
    q: Arc<RuntimeFilterShared>,
    num_docs: DocId,
    cur: DocId,
    started: bool,
}

#[derive(Debug)]
struct RuntimeFilterShared {
    rt: Arc<RuntimeFields>,
    idx: usize,
    pred: Pred,
    boost: f32,
    errors: ErrSlot,
    cancel: Option<Arc<super::runtime::Cancel>>,
}

impl RuntimeFilterScorer {
    fn find_from(&mut self, mut d: DocId) -> DocId {
        let f = &self.q.rt.fields[self.q.idx];
        while d < self.num_docs {
            match f.values(&mut self.exec, d) {
                Ok(vals) => {
                    let hit = match &self.q.pred {
                        // A runtime shape is all its values (ES builds one
                        // doc value from them): related as a whole.
                        Pred::Geo(q) => {
                            let gs: Vec<rustic_core::fieldtypes::shape::Geometry> = vals.iter().filter_map(|v| match v {
                                Val::Str(s) => rustic_core::fieldtypes::shape::decode(s),
                                _ => None,
                            }).collect();
                            !gs.is_empty() && crate::fieldtypes::geo_shape_query::geometry_matches(&rustic_core::fieldtypes::shape::Geometry::Collection(gs), q) == Some(true)
                        }
                        pred => vals.iter().any(|v| val_matches(pred, v, f.ty)),
                    };
                    if hit {
                        self.cur = d;
                        return d;
                    }
                    d += 1;
                }
                Err(e) => {
                    if super::runtime::Cancel::absorb(&self.q.cancel, &e) {
                        break;
                    }
                    let msg = e.msg.clone();
                    let es = match &f.script {
                        Some(s) => s.error(&e),
                        None => EsError::illegal_argument(msg),
                    };
                    report_error(&self.q.errors, es);
                    break;
                }
            }
        }
        self.cur = TERMINATED;
        TERMINATED
    }
}

impl DocSet for RuntimeFilterScorer {
    fn advance(&mut self) -> DocId {
        if !self.started {
            self.started = true;
            return self.find_from(0);
        }
        if self.cur == TERMINATED {
            return TERMINATED;
        }
        let n = self.cur + 1;
        self.find_from(n)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.started && (self.cur == TERMINATED || self.cur >= target) {
            return self.cur;
        }
        self.started = true;
        self.find_from(target)
    }
    fn doc(&self) -> DocId {
        if self.started { self.cur } else { TERMINATED }
    }
    fn size_hint(&self) -> u32 {
        self.num_docs
    }
}

impl Scorer for RuntimeFilterScorer {
    fn score(&mut self) -> f32 {
        self.q.boost
    }
}

struct RuntimeFilterQuery {
    shared: Arc<RuntimeFilterShared>,
    lookup: Arc<FieldLookup>,
}

impl std::fmt::Debug for RuntimeFilterQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuntimeFilter({:?})", self.shared.pred)
    }
}

impl CustomQuery for RuntimeFilterQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let s = &self.shared;
        let mut exec = s.rt.fields[s.idx].executor(Some(self.lookup.clone()), Some(s.rt.clone()));
        // SAFETY: CustomQuery contract (scorer never outlives the leaf borrow).
        unsafe { exec.bind_leaf(leaf) };
        exec.rt.cancel = s.cancel.clone();
        Some(Box::new(RuntimeFilterScorer { exec, q: s.clone(), num_docs: leaf.num_docs(), cur: TERMINATED, started: false }))
    }
    fn is_constant_score(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "runtime_field"
    }
}

impl From<RuntimeFilter> for RuntimeFilterQuery {
    fn from(r: RuntimeFilter) -> Self {
        RuntimeFilterQuery { shared: Arc::new(RuntimeFilterShared { rt: r.rt, idx: r.idx, pred: r.pred, boost: r.boost, errors: r.errors, cancel: r.cancel }), lookup: r.lookup }
    }
}

fn wildcard_regex(p: &str, ci: bool) -> Result<regex::Regex> {
    let mut re = String::from(if ci { "(?is)^" } else { "(?s)^" });
    for c in p.chars() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
    }
    re.push('$');
    regex::Regex::new(&re).map_err(|e| EsError::illegal_argument(e.to_string()))
}

/// Queries on runtime fields (term/terms/range/exists/match/prefix/
/// wildcard/regexp): `Ok(None)` when `field` is not a runtime field.
pub fn runtime_field_query(name: &str, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    if !matches!(name, "term" | "terms" | "range" | "exists" | "match" | "match_phrase" | "prefix" | "wildcard" | "regexp") {
        return Ok(None);
    }
    let Some(env) = env() else { return Ok(None) };
    let Some(ie) = env.per_index.borrow().get(ctx.index_name).cloned() else { return Ok(None) };
    let Some(rt) = ie.runtime else { return Ok(None) };
    let Some(o) = body.as_object() else { return Ok(None) };
    let (field, spec): (String, Value) = if name == "exists" {
        match o.get("field").and_then(Value::as_str) {
            Some(f) => (f.to_string(), Value::Null),
            None => return Ok(None),
        }
    } else {
        match o.iter().find(|(k, _)| !matches!(k.as_str(), "boost" | "_name")) {
            Some((k, v)) => (k.clone(), v.clone()),
            None => return Ok(None),
        }
    };
    let Some(idx) = rt.index_of(&field) else { return Ok(None) };
    let ty = rt.fields[idx].ty;
    if ty == super::RtType::Lookup {
        return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("Field [{field}] of type [lookup] does not support {name} queries")));
    }
    let mut boost = 1.0f32;
    let inner = |v: &Value, key: &str| -> Option<Value> {
        match v {
            Value::Object(m) => m.get(key).cloned(),
            other => Some(other.clone()),
        }
    };
    if let Value::Object(m) = &spec {
        if let Some(b) = m.get("boost").and_then(Value::as_f64) {
            boost = b as f32;
        }
    }
    let date_bound = |v: &Value, up: bool| -> Result<f64> {
        match v {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
            Value::String(s) => Ok(crate::date::parse_date_math(s, &rt.fields[idx].format, crate::date::Tz::UTC, ctx.now, up)? as f64),
            _ => Err(EsError::parsing("invalid range bound")),
        }
    };
    let pred = match name {
        "exists" => Pred::Exists,
        "term" | "match" | "match_phrase" => {
            let key = if name == "term" { "value" } else { "query" };
            let v = inner(&spec, key).ok_or_else(|| EsError::parsing(format!("[{name}] query on [{field}] requires a value")))?;
            let v = if ty == super::RtType::Date {
                Value::from(date_bound(&v, false)? as i64)
            } else {
                v
            };
            Pred::Terms(vec![v])
        }
        "terms" => {
            let vs = spec.as_array().cloned().ok_or_else(|| EsError::parsing("[terms] query requires an array of values"))?;
            Pred::Terms(vs)
        }
        "range" => {
            let m = spec.as_object().ok_or_else(|| EsError::parsing("[range] query malformed"))?;
            let numeric = !matches!(ty, super::RtType::Keyword | super::RtType::Ip);
            let mut lo = None;
            let mut hi = None;
            let mut slo = None;
            let mut shi = None;
            for (k, v) in m {
                let (is_lo, inc) = match k.as_str() {
                    "gte" | "from" => (true, true),
                    "gt" => (true, false),
                    "lte" | "to" => (false, true),
                    "lt" => (false, false),
                    _ => continue,
                };
                if v.is_null() {
                    continue;
                }
                if numeric {
                    let b = if ty == super::RtType::Date {
                        date_bound(v, !is_lo && inc || is_lo && !inc)?
                    } else {
                        v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| EsError::parsing(format!("invalid range bound [{v}]")))?
                    };
                    if is_lo { lo = Some((b, inc)) } else { hi = Some((b, inc)) }
                } else {
                    let b = crate::query::dsl::scalar_string(v);
                    if is_lo { slo = Some((b, inc)) } else { shi = Some((b, inc)) }
                }
            }
            Pred::Range { lo, hi, slo, shi }
        }
        "prefix" => {
            let v = inner(&spec, "value").ok_or_else(|| EsError::parsing("[prefix] requires a value"))?;
            let ci = spec.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
            Pred::Prefix(crate::query::dsl::scalar_string(&v), ci)
        }
        "wildcard" => {
            let v = inner(&spec, "value").or_else(|| inner(&spec, "wildcard")).ok_or_else(|| EsError::parsing("[wildcard] requires a value"))?;
            let ci = spec.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
            Pred::Regex(wildcard_regex(&crate::query::dsl::scalar_string(&v), ci)?)
        }
        _ => {
            let v = inner(&spec, "value").ok_or_else(|| EsError::parsing("[regexp] requires a value"))?;
            let re = regex::Regex::new(&format!("^(?:{})$", crate::query::dsl::scalar_string(&v))).map_err(|e| EsError::illegal_argument(e.to_string()))?;
            Pred::Regex(re)
        }
    };
    if let Some(b) = o.get("boost").and_then(Value::as_f64) {
        boost = b as f32;
    }
    let rf = RuntimeFilter { rt, idx, lookup: ie.lookup.clone(), pred, boost, errors: env.error.clone(), cancel: env.cancel.clone() };
    Ok(Some(Query::Custom(Arc::new(RuntimeFilterQuery::from(rf)))))
}

// ---------------------------------------------------------------------------
// Value sources for aggregations / sort / fields.

/// A per-leaf value source: a script (optionally applied to a field's
/// values as `_value`) or a runtime field.
pub struct ValueSource {
    script: Option<CompiledScript>,
    runtime: Option<(Arc<RuntimeFields>, usize)>,
    lookup: Arc<FieldLookup>,
    runtime_all: Option<Arc<RuntimeFields>>,
    exec: Option<(usize, Executor)>,
}

impl ValueSource {
    /// Script source for the index named `index` (mapping from the env).
    pub fn script(script: CompiledScript, index: &str, lookup: &FieldLookup) -> ValueSource {
        let (l, rt) = match index_env(Some(index)) {
            Some(ie) => (ie.lookup, ie.runtime),
            None => (Arc::new(lookup.clone()), None),
        };
        ValueSource { script: Some(script), runtime: None, lookup: l, runtime_all: rt, exec: None }
    }

    /// Runtime field `name` of `index`, if it is one.
    pub fn runtime(name: &str, index: &str) -> Option<ValueSource> {
        let ie = index_env(Some(index))?;
        let rt = ie.runtime.clone()?;
        let i = rt.index_of(name)?;
        Some(ValueSource { script: None, runtime: Some((rt.clone(), i)), lookup: ie.lookup, runtime_all: Some(rt), exec: None })
    }

    /// The same source without a bound executor (for another thread).
    pub fn fork(&self) -> ValueSource {
        ValueSource { script: self.script.clone(), runtime: self.runtime.clone(), lookup: self.lookup.clone(), runtime_all: self.runtime_all.clone(), exec: None }
    }

    /// Whether values depend on the doc's `_score`.
    pub fn uses_score(&self) -> bool {
        self.script.as_ref().is_some_and(|s| s.program.uses_score)
    }

    pub fn runtime_type(&self) -> Option<super::RtType> {
        self.runtime.as_ref().map(|(rt, i)| rt.fields[*i].ty)
    }

    /// [`Self::values`] handed to `f` (runtime fields without a per-doc
    /// allocation).
    pub fn with_values<R>(&mut self, leaf_key: usize, leaf: &LeafReader<'_>, doc: DocId, score: f32, f: impl FnOnce(&[Val]) -> R) -> Result<R> {
        if !self.bind(leaf_key, leaf) {
            return Ok(f(&[]));
        }
        let exec = &mut self.exec.as_mut().unwrap().1;
        // Past the search's deadline (or cancelled): no value, no script run.
        if exec.rt.cancel.as_ref().is_some_and(|c| c.stopped()) {
            return Ok(f(&[]));
        }
        match (&self.script, &self.runtime) {
            (Some(s), _) => {
                exec.set_doc(doc);
                exec.set_score(score as f64);
                let v = match exec.run() {
                    Ok(v) => v,
                    // Stopped by the deadline: the value is missing and the
                    // search reports `timed_out` (partial results).
                    Err(e) if super::runtime::Cancel::absorb(&exec.rt.cancel, &e) => return Ok(f(&[])),
                    Err(e) => return Err(s.error(&e)),
                };
                Ok(match &v {
                    Val::List(l) | Val::Set(l) => f(&l.borrow()),
                    Val::Array { items, .. } => f(&items.borrow()),
                    Val::DocValues(d) => f(&d.values),
                    Val::Null => f(&[]),
                    other => f(std::slice::from_ref(other)),
                })
            }
            (None, Some((rt, i))) => {
                let rf = &rt.fields[*i];
                let mut f = Some(f);
                match rf.with_values(exec, doc, |vals: &[Val]| (f.take().expect("called once"))(vals)) {
                    Ok(r) => Ok(r),
                    Err(e) => match f.take() {
                        Some(f) if super::runtime::Cancel::absorb(&exec.rt.cancel, &e) => Ok(f(&[])),
                        _ => Err(match &rf.script {
                            Some(s) => s.error(&e),
                            None => EsError::illegal_argument(e.msg.clone()),
                        }),
                    },
                }
            }
            _ => Ok(f(&[])),
        }
    }

    /// Bind the executor to leaf `leaf_key` (once per leaf); `false` when
    /// there is nothing to evaluate.
    fn bind(&mut self, leaf_key: usize, leaf: &LeafReader<'_>) -> bool {
        if matches!(&self.exec, Some((k, _)) if *k == leaf_key) {
            return true;
        }
        let mut e = match (&self.script, &self.runtime) {
            (Some(s), _) => s.executor(Some(self.lookup.clone()), self.runtime_all.clone()),
            (None, Some((rt, i))) => rt.fields[*i].executor(Some(self.lookup.clone()), Some(rt.clone())),
            _ => return false,
        };
        // SAFETY: the value source is only used while the caller holds the leaf.
        unsafe { e.bind_leaf(leaf) };
        // Script values computed during a search keep to its deadline (ES
        // times out aggregation and sort scripts through its collectors).
        if e.rt.cancel.is_none() {
            e.rt.cancel = env().and_then(|env| env.cancel.clone());
        }
        self.exec = Some((leaf_key, e));
        true
    }

    /// Values for `doc` of leaf `leaf_key` (`_value` = `field_value` for
    /// value scripts).
    pub fn values(&mut self, leaf_key: usize, leaf: &LeafReader<'_>, doc: DocId, score: f32, field_value: Option<Val>) -> Result<Vec<Val>> {
        if !self.bind(leaf_key, leaf) {
            return Ok(Vec::new());
        }
        let exec = &mut self.exec.as_mut().unwrap().1;
        // Past the search's deadline (or cancelled): no value, no script run.
        if exec.rt.cancel.as_ref().is_some_and(|c| c.stopped()) {
            return Ok(Vec::new());
        }
        match (&self.script, &self.runtime) {
            (Some(s), _) => {
                exec.set_doc(doc);
                exec.set_score(score as f64);
                if let Some(v) = field_value {
                    exec.set_value(v);
                }
                let v = match exec.run() {
                    Ok(v) => v,
                    Err(e) if super::runtime::Cancel::absorb(&exec.rt.cancel, &e) => return Ok(Vec::new()),
                    Err(e) => return Err(s.error(&e)),
                };
                Ok(match v {
                    Val::List(l) | Val::Set(l) => l.borrow().clone(),
                    Val::Array { items, .. } => items.borrow().clone(),
                    Val::DocValues(d) => d.values.clone(),
                    Val::Null => Vec::new(),
                    other => vec![other],
                })
            }
            (None, Some((rt, i))) => {
                let f = &rt.fields[*i];
                match f.values(exec, doc) {
                    Ok(v) => Ok(v),
                    Err(e) if super::runtime::Cancel::absorb(&exec.rt.cancel, &e) => Ok(Vec::new()),
                    Err(e) => Err(match &f.script {
                        Some(s) => s.error(&e),
                        None => EsError::illegal_argument(e.msg.clone()),
                    }),
                }
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Field kind for `doc`-style typed access of a mapped field.
pub fn kind_of(lookup: &FieldLookup, name: &str) -> Kind {
    match lookup.get(name) {
        Some(f) => Kind::of(f.ty),
        None => Kind::Unmapped,
    }
}

/// `script_fields` entries: `(name, script, ignore_failure)`.
pub fn parse_script_fields(v: &Value) -> Result<Vec<(String, CompiledScript, bool)>> {
    let o = v.as_object().ok_or_else(|| EsError::parsing("[script_fields] must be an object"))?;
    let mut out = Vec::new();
    for (name, def) in o {
        let d = def.as_object().ok_or_else(|| EsError::parsing(format!("[script_fields] [{name}] must be an object")))?;
        let s = d.get("script").ok_or_else(|| EsError::parsing(format!("must specify a script in script fields")))?;
        let ignore = d.get("ignore_failure").and_then(Value::as_bool).unwrap_or(false);
        let environment=super::env();
        let raw=environment.as_ref().and_then(|e|e.raw_script_fields.as_ref()).and_then(|f|f.get(name)).and_then(|f|f.get("script"));
        out.push((name.clone(), super::compile_json_source(s, raw, ScriptContext::Field, None)?, ignore));
    }
    Ok(out)
}

/// Names of runtime fields of `index` matching `pattern` (wildcards).
pub fn runtime_field_names(index: &str, pattern: &str) -> Vec<String> {
    let Some(ie) = index_env(Some(index)) else { return Vec::new() };
    let Some(rt) = ie.runtime else { return Vec::new() };
    rt.fields.iter().filter(|f| crate::mapping::wildcard_match(pattern, &f.name)).map(|f| f.name.clone()).collect()
}

// ---------------------------------------------------------------------------
// Sorting by `_script` / runtime fields (collected here: core sort keys are
// fast-field based).

use rustic_core::query::{Bm25Params, SortField, SortKind, SortMissing, SortMode, SortValue, TopDocs, TopHit, TotalHitsRelation};

pub enum SortKeyDef {
    Score { desc: bool },
    Doc,
    /// A mapped field (the core's [`SortField::Field`]); `cast`:
    /// `numeric_type: long` over a floating-point field.
    Field { field: u16, kind: SortKind, missing: SortMissing, mode: Option<SortMode>, desc: bool, cast: Option<LongCast> },
    /// A numeric field under a nested path (the core's [`SortField::Nested`]):
    /// its children's (under `filter`) values picked by `mode`; `bits`: the
    /// filter's matches on the leaf (by leaf ordinal) once computed.
    NestedField {
        path: String,
        filter: Option<Box<Query>>,
        bits: Option<(usize, Vec<u64>)>,
        field: u16,
        kind: SortKind,
        missing: SortMissing,
        mode: Option<SortMode>,
        desc: bool,
        cast: Option<LongCast>,
        /// At most this many eligible children per root (SIDE 014).
        max_children: Option<usize>,
    },
    /// Script / runtime field values: `mode` picks among several (min
    /// ascending, max descending by default), `missing` stands in for none.
    /// `version`: string values sort as versions (encoded).
    Values { src: ValueSource, string: bool, desc: bool, version: bool, mode: Option<SortMode>, missing: SortMissing },
    /// `_ts_routing_hash`: the routing hash bytes leading a time series `_id`.
    TsRoutingHash { desc: bool },
}

/// `numeric_type: long` over a floating-point field (ES
/// `LongValuesComparatorSource` over `castToLong` doc values): every value
/// is cast to a long (Java `(long) value`: toward zero, saturating, NaN as
/// 0) before the sort mode picks among them, so ordering, ties and
/// `search_after` cursors all use the cast keys. A `scaled_float` casts its
/// doc value (the scaled long times the inverse factor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongCast {
    pub scaling: Option<f64>,
}

impl LongCast {
    pub fn apply(self, x: f64) -> i64 {
        let x = match self.scaling {
            Some(f) => crate::aggs::es_scaled(x, Some((f, 1.0 / f))),
            None => x,
        };
        x as i64
    }
}

/// The cast of criterion `spec` (resolved to `sf`) when it is a
/// `numeric_type: long` sort of a floating-point field.
pub(crate) fn long_cast(spec: &crate::search::SortSpec, sf: &SortField, lookup: &FieldLookup) -> Option<LongCast> {
    // A nested field's criterion casts its children's values the same way.
    let inner = match sf {
        SortField::Nested { sort, .. } => &**sort,
        other => other,
    };
    if spec.numeric_type.as_deref() != Some("long") || !matches!(inner, SortField::Field { kind: SortKind::F64, .. }) {
        return None;
    }
    let scaling = lookup.get(&spec.field).filter(|f| f.ty == crate::mapping::EsType::ScaledFloat && f.scaling_factor.is_finite() && f.scaling_factor > 0.0).map(|f| f.scaling_factor);
    Some(LongCast { scaling })
}

/// Render of a runtime field used as a sort key (`None` if `name` is not a
/// runtime field of `index`).
pub fn runtime_render(index: &str, name: &str) -> Option<&'static str> {
    let (rt, i) = super::runtime_field(Some(index), name)?;
    Some(match rt.fields[i].ty {
        super::RtType::Long | super::RtType::Boolean => "long",
        super::RtType::Double => "double",
        super::RtType::Date => "date",
        super::RtType::Keyword | super::RtType::Ip | super::RtType::Lookup | super::RtType::GeoShape => "keyword",
    })
}

/// Sorting or aggregating on a `lookup` runtime field is an error.
pub fn check_fielddata(index: &str, name: &str) -> Result<()> {
    match super::runtime_field(Some(index), name) {
        Some((rt, i)) if rt.fields[i].ty == super::RtType::Lookup => {
            Err(EsError::illegal_argument(format!("Fielddata is not supported on field [{name}] of type [lookup]")))
        }
        _ => Ok(()),
    }
}

/// `fields` API value of a `lookup` runtime field: one object per matching
/// target doc, each fetch field mapped to its values.
fn lookup_field_json(spec: &super::LookupSpec, lookup: &FieldLookup, leaf: &LeafReader<'_>, doc: DocId) -> Result<Vec<Value>> {
    let mut inputs = crate::search::docvalue_values(leaf, lookup, &spec.input_field, doc, None)?;
    if inputs.is_empty() {
        // Dynamically mapped strings are text with a `.keyword` sub-field.
        inputs = crate::search::docvalue_values(leaf, lookup, &format!("{}.keyword", spec.input_field), doc, None)?;
    }
    let mut out = Vec::new();
    // Other target fields: a term search of the target index per input value
    // (ES `LookupField`: `term` on the target field, size 1), each match
    // contributing its fetch fields.
    if spec.target_field != "_id" {
        for input in inputs {
            let value = crate::query::dsl::scalar_string(&input);
            out.extend(super::lookup_term(&spec.target_index, &spec.target_field, &value, &spec.fetch_fields)?);
        }
        return Ok(out);
    }
    // `_id` target lookups go through realtime get.
    for input in inputs {
        let id = crate::query::dsl::scalar_string(&input);
        let mut entry = Map::new();
        for f in &spec.fetch_fields {
            if let Ok(vals) = super::lookup_values(&spec.target_index, &id, f) {
                if !vals.is_empty() {
                    entry.insert(f.clone(), Value::Array(vals));
                }
            }
        }
        if !entry.is_empty() {
            out.push(Value::Object(entry));
        }
    }
    Ok(out)
}

/// `{"_script": {"type": "number"|"string", "script": .., "order": ..}}` render.
pub fn script_sort_render(spec: Option<&Value>) -> Result<&'static str> {
    let spec = spec.and_then(Value::as_object).ok_or_else(|| EsError::parsing("[_script] sort requires an object"))?;
    match spec.get("type").and_then(Value::as_str) {
        Some("number") => Ok("double"),
        Some("string") => Ok("keyword"),
        Some("version") => Ok("version"),
        Some(other) => Err(EsError::illegal_argument(format!("Unknown ScriptSortType [{other}]"))),
        None => Err(EsError::parsing("[_script] sort requires [type]").with("line", 1).with("col", 1)),
    }
}

fn sort_mode(m: Option<&str>) -> Result<Option<SortMode>> {
    Ok(match m {
        None => None,
        Some("min") => Some(SortMode::Min),
        Some("max") => Some(SortMode::Max),
        Some("sum") => Some(SortMode::Sum),
        Some("avg") => Some(SortMode::Avg),
        Some("median") => Some(SortMode::Median),
        Some(o) => return Err(EsError::illegal_argument(format!("Unknown SortMode [{o}]"))),
    })
}

/// Sort keys to collect here when any criterion is a `_script`, a runtime
/// field or a `numeric_type: long` cast of a floating-point field; `None` =
/// the core searcher can sort. `fields` are the criteria as resolved
/// (`crate::search::resolve_sort`), one per spec.
pub fn sort_keys(specs: &[crate::search::SortSpec], fields: &[SortField], index: &str, lookup: &FieldLookup) -> Result<Option<Vec<SortKeyDef>>> {
    let tsdb = lookup.get(crate::tsdb::TSID_FIELD).is_some();
    let needs = specs.iter().zip(fields).any(|(s, sf)| s.field == "_script" || runtime_render(index, &s.field).is_some() || (tsdb && s.field == "_ts_routing_hash") || long_cast(s, sf, lookup).is_some());
    if !needs {
        return Ok(None);
    }
    let mut out = Vec::new();
    // `_script` criteria seen so far: each has its own definition.
    let mut scripts = 0usize;
    for (spec, sf) in specs.iter().zip(fields) {
        let desc = spec.descending;
        out.push(match spec.field.as_str() {
            "_score" => SortKeyDef::Score { desc },
            "_doc" | "_shard_doc" => SortKeyDef::Doc,
            "_ts_routing_hash" if tsdb => SortKeyDef::TsRoutingHash { desc },
            "_script" => {
                let ordinal = scripts;
                scripts += 1;
                let def = spec.extra.as_ref();
                let render = script_sort_render(def)?;
                let string = render != "double";
                let s = def.and_then(|s| s.get("script")).ok_or_else(|| EsError::parsing("[_script] sort requires a [script]"))?;
                let environment = env();
                // The raw subtree of the same (`ordinal`-th) `_script` criterion.
                let raw = environment
                    .as_ref()
                    .and_then(|e| e.raw_sort.as_ref())
                    .and_then(|sort| match sort {
                        crate::source_json::Source::Array(entries) => entries.iter().filter_map(|entry| entry.get("_script")).nth(ordinal),
                        entry => entry.get("_script").filter(|_| ordinal == 0),
                    })
                    .and_then(|spec| spec.get("script"));
                let cs = super::compile_json_source(s, raw, if string { ScriptContext::StringSort } else { ScriptContext::NumberSort }, None)?;
                let mode = sort_mode(spec.mode.as_deref())?;
                SortKeyDef::Values { src: ValueSource::script(cs, index, lookup), string, desc, version: render == "version", mode, missing: SortMissing::Last }
            }
            n => match ValueSource::runtime(n, index) {
                Some(src) => {
                    let string = matches!(src.runtime_type(), Some(super::RtType::Keyword | super::RtType::Ip));
                    // The criterion's `mode` and `missing` as resolved.
                    let (mode, missing) = match sf {
                        SortField::Field { mode, missing, .. } => (*mode, missing.clone()),
                        _ => (sort_mode(spec.mode.as_deref())?, SortMissing::Last),
                    };
                    SortKeyDef::Values { src, string, desc, version: false, mode, missing }
                }
                None => match sf {
                    SortField::Field { field, kind, missing, mode, descending } => {
                        SortKeyDef::Field { field: *field, kind: *kind, missing: missing.clone(), mode: *mode, desc: *descending, cast: long_cast(spec, sf, lookup) }
                    }
                    SortField::Nested { path, filter, sort, max_children } if matches!(&**sort, SortField::Field { kind: SortKind::F64 | SortKind::I64, .. }) => {
                        let SortField::Field { field, kind, missing, mode, descending } = &**sort else { unreachable!() };
                        SortKeyDef::NestedField {
                            path: path.clone(),
                            filter: filter.clone(),
                            bits: None,
                            field: *field,
                            kind: *kind,
                            missing: missing.clone(),
                            mode: *mode,
                            desc: *descending,
                            cast: long_cast(spec, sf, lookup),
                            max_children: *max_children,
                        }
                    }
                    _ => return Err(EsError::illegal_argument(format!("sorting on [{n}] together with a script or runtime field sort is not supported"))),
                },
            },
        });
    }
    Ok(Some(out))
}

/// Java's `Math.round` of a double, as a long.
fn round_long(value: f64) -> i64 {
    let floor = value.floor();
    (if value - floor >= 0.5 { floor + 1.0 } else { floor }) as i64
}

/// ES `MultiValueMode` over long values, as the core searcher picks them
/// (`unsigned`: shifted unsigned longs, averaged as doubles).
fn pick_longs(mut v: Vec<i64>, mode: SortMode, unsigned: bool) -> SortValue {
    match (v.len(), mode) {
        (0, _) => SortValue::Missing,
        (_, SortMode::Min) => SortValue::I64(*v.iter().min().unwrap()),
        (_, SortMode::Max) => SortValue::I64(*v.iter().max().unwrap()),
        (_, SortMode::Sum) => SortValue::I64(v.iter().fold(0i64, |sum, &x| sum.wrapping_add(x))),
        (1, _) if !unsigned => SortValue::I64(v[0]),
        (_, SortMode::Avg) if !unsigned => SortValue::I64(round_long(v.iter().fold(0i64, |sum, &x| sum.wrapping_add(x)) as f64 / v.len() as f64)),
        (_, SortMode::Median) if !unsigned => {
            v.sort_unstable();
            let m = v.len() / 2;
            SortValue::I64(if v.len() % 2 == 0 { round_long((v[m - 1] as f64 + v[m] as f64) / 2.0) } else { v[m] })
        }
        (_, m) => pick_doubles(v.into_iter().map(|x| x as f64).collect(), m),
    }
}

/// ES `MultiValueMode` over double values.
fn pick_doubles(mut v: Vec<f64>, mode: SortMode) -> SortValue {
    if v.is_empty() {
        return SortValue::Missing;
    }
    SortValue::F64(match mode {
        SortMode::Min => v.iter().copied().fold(f64::INFINITY, f64::min),
        SortMode::Max => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        SortMode::Sum => v.iter().sum(),
        SortMode::Avg => v.iter().sum::<f64>() / v.len() as f64,
        SortMode::Median => {
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        }
    })
}

/// The sort value of mapped field `field` of `doc`, picked by `mode` (min
/// ascending, max descending by default) as the core searcher picks it,
/// `missing` standing in for a doc without one. Shared by this collector
/// and inner hits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn field_sort_value(leaf: &LeafReader<'_>, doc: DocId, field: u16, kind: SortKind, desc: bool, mode: Option<SortMode>, missing: &SortMissing, cast: Option<LongCast>) -> Result<SortValue> {
    let mode = mode.unwrap_or(if desc { SortMode::Max } else { SortMode::Min });
    let v = match (kind, cast) {
        (SortKind::F64, Some(c)) => pick_longs(leaf.f64_values(field, doc).into_iter().map(|x| c.apply(x)).collect(), mode, false),
        (SortKind::F64, None) => pick_doubles(leaf.f64_values(field, doc), mode),
        (SortKind::I64, _) => {
            let v = leaf.i64_values(field, doc);
            if v.is_empty() {
                // u64 columns (unsigned_long) sort as longs shifted by Long.MIN_VALUE.
                pick_longs(leaf.u64_values(field, doc).into_iter().map(|x| (x ^ (1u64 << 63)) as i64).collect(), mode, true)
            } else {
                pick_longs(v, mode, false)
            }
        }
        (SortKind::Keyword, _) => {
            let mut v = leaf.keyword_values(field, doc);
            v.sort();
            let pick = if mode == SortMode::Max { v.pop() } else { v.into_iter().next() };
            pick.map(SortValue::Str).unwrap_or(SortValue::Missing)
        }
        // Ordinals follow term order: the smallest / largest term.
        (SortKind::Fielddata, _) => {
            let fd = leaf.uninverted(field)?;
            let ords = fd.ords(doc);
            let o = if mode == SortMode::Max { ords.last() } else { ords.first() };
            o.map(|&o| SortValue::Str(fd.term_string(o))).unwrap_or(SortValue::Missing)
        }
    };
    Ok(match (v, missing) {
        // A cast sort's missing value is a long too.
        (SortValue::Missing, SortMissing::Value(SortValue::F64(x))) if cast.is_some() => SortValue::I64(*x as i64),
        (SortValue::Missing, SortMissing::Value(m)) => m.clone(),
        (v, _) => v,
    })
}

/// Pick among several script / runtime values: long rules when every
/// value is a long, double rules for numbers, else the smallest / largest.
fn pick_values(vals: Vec<SortValue>, desc: bool, mode: Option<SortMode>) -> SortValue {
    if vals.is_empty() {
        return SortValue::Missing;
    }
    let mode = mode.unwrap_or(if desc { SortMode::Max } else { SortMode::Min });
    if vals.iter().all(|v| matches!(v, SortValue::I64(_))) {
        return pick_longs(vals.into_iter().map(|v| if let SortValue::I64(x) = v { x } else { 0 }).collect(), mode, false);
    }
    if vals.iter().all(|v| matches!(v, SortValue::I64(_) | SortValue::F64(_))) {
        let num = |v: &SortValue| match v {
            SortValue::I64(i) => *i as f64,
            SortValue::F64(f) => *f,
            _ => 0.0,
        };
        return pick_doubles(vals.iter().map(num).collect(), mode);
    }
    let max = mode == SortMode::Max;
    let mut it = vals.into_iter();
    let mut best = it.next().unwrap_or(SortValue::Missing);
    for v in it {
        let o = cmp_vals(&v, &best);
        if (max && o.is_gt()) || (!max && o.is_lt()) {
            best = v;
        }
    }
    best
}

fn cmp_vals(a: &SortValue, b: &SortValue) -> std::cmp::Ordering {
    use SortValue::*;
    match (a, b) {
        (I64(x), I64(y)) => x.cmp(y),
        (F64(x), F64(y)) => x.total_cmp(y),
        (I64(x), F64(y)) => (*x as f64).total_cmp(y),
        (F64(x), I64(y)) => x.total_cmp(&(*y as f64)),
        (Str(x), Str(y)) => x.cmp(y),
        (Missing, Missing) => std::cmp::Ordering::Equal,
        (Missing, _) => std::cmp::Ordering::Greater,
        (_, Missing) => std::cmp::Ordering::Less,
        (Str(_), _) => std::cmp::Ordering::Greater,
        (_, Str(_)) => std::cmp::Ordering::Less,
    }
}

fn val_to_sort(v: &Val, string: bool) -> SortValue {
    if string {
        return SortValue::Str(v.to_java_string());
    }
    match v {
        Val::Long(l) => SortValue::I64(*l),
        Val::Int(i) => SortValue::I64(*i as i64),
        Val::Date(m) | Val::Instant(m) => SortValue::I64(*m),
        Val::Bool(b) => SortValue::I64(*b as i64),
        other => other.as_f64().map(SortValue::F64).unwrap_or(SortValue::Missing),
    }
}

impl SortKeyDef {
    /// The same key with its own (unbound) per-leaf state.
    fn fork(&self) -> SortKeyDef {
        match self {
            SortKeyDef::Score { desc } => SortKeyDef::Score { desc: *desc },
            SortKeyDef::Doc => SortKeyDef::Doc,
            SortKeyDef::Field { field, kind, missing, mode, desc, cast } => SortKeyDef::Field { field: *field, kind: *kind, missing: missing.clone(), mode: *mode, desc: *desc, cast: *cast },
            SortKeyDef::NestedField { path, filter, field, kind, missing, mode, desc, cast, max_children, .. } => SortKeyDef::NestedField {
                path: path.clone(),
                filter: filter.clone(),
                bits: None,
                field: *field,
                kind: *kind,
                missing: missing.clone(),
                mode: *mode,
                desc: *desc,
                cast: *cast,
                max_children: *max_children,
            },
            SortKeyDef::Values { src, string, desc, version, mode, missing } => {
                SortKeyDef::Values { src: src.fork(), string: *string, desc: *desc, version: *version, mode: *mode, missing: missing.clone() }
            }
            SortKeyDef::TsRoutingHash { desc } => SortKeyDef::TsRoutingHash { desc: *desc },
        }
    }

    fn descending(&self) -> bool {
        match self {
            SortKeyDef::Score { desc } => !*desc,
            SortKeyDef::Doc => false,
            SortKeyDef::Field { desc, .. } | SortKeyDef::NestedField { desc, .. } | SortKeyDef::Values { desc, .. } | SortKeyDef::TsRoutingHash { desc } => *desc,
        }
    }

    /// `(descending, missing first)` of the key.
    fn order(&self) -> (bool, bool) {
        let missing_first = match self {
            SortKeyDef::Field { missing, .. } | SortKeyDef::NestedField { missing, .. } | SortKeyDef::Values { missing, .. } => matches!(missing, SortMissing::First),
            _ => false,
        };
        (self.descending(), missing_first)
    }
}

/// Whether `v` is the value a missing one sorts as (`low`: at the low end
/// of the ascending order), which ties with it, as ES substitutes it.
fn is_sentinel(v: &SortValue, low: bool) -> bool {
    match v {
        SortValue::I64(x) => *x == if low { i64::MIN } else { i64::MAX },
        SortValue::F64(x) => *x == if low { f64::NEG_INFINITY } else { f64::INFINITY },
        _ => false,
    }
}

/// Order of two hits' sort values: per key `(descending, missing first)`;
/// a missing value goes last (first with `missing: _first`) either way.
fn cmp_sort(orders: &[(bool, bool)], a: &[SortValue], b: &[SortValue]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (i, &(desc, missing_first)) in orders.iter().enumerate() {
        let (Some(x), Some(y)) = (a.get(i), b.get(i)) else { break };
        let o = match (x, y) {
            (SortValue::Missing, SortValue::Missing) => Ordering::Equal,
            (SortValue::Missing, v) | (v, SortValue::Missing) if is_sentinel(v, missing_first != desc) => Ordering::Equal,
            (SortValue::Missing, _) => if missing_first { Ordering::Less } else { Ordering::Greater },
            (_, SortValue::Missing) => if missing_first { Ordering::Greater } else { Ordering::Less },
            _ => {
                let o = cmp_vals(x, y);
                if desc { o.reverse() } else { o }
            }
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// One leaf's share of [`collect_sorted`].
struct LeafTop {
    hits: Vec<TopHit>,
    total: u64,
    max_score: f32,
}

#[allow(clippy::too_many_arguments)]
fn collect_leaf(li: usize, leaf: &LeafReader<'_>, query: &Query, ctx: &SearchContext, keys: &mut [SortKeyDef], orders: &[(bool, bool)], k: usize, min_score: Option<f32>, after: Option<&TopHit>) -> Result<LeafTop> {
    let mut out = LeafTop { hits: Vec::new(), total: 0, max_score: f32::NEG_INFINITY };
    let Some(mut s) = query.scorer(leaf, ctx) else { return Ok(out) };
    let live = leaf.live_docs();
    let leaf_key = leaf.id();
    // Hits are kept to at most `2k`: past that the leaf's best `k` are
    // retained and anything not sorting before the k-th is skipped.
    let cap = k.saturating_mul(2).max(64);
    let mut worst: Option<Vec<SortValue>> = None;
    let mut vals: Vec<SortValue> = Vec::with_capacity(keys.len());
    loop {
        let d = s.advance();
        if d == TERMINATED {
            break;
        }
        if !live(d) {
            continue;
        }
        let sc = s.score();
        if min_score.is_some_and(|m| sc < m) {
            continue;
        }
        out.total += 1;
        out.max_score = out.max_score.max(sc);
        vals.clear();
        for key in keys.iter_mut() {
            vals.push(match key {
                SortKeyDef::Score { .. } => SortValue::F64(sc as f64),
                SortKeyDef::Doc => SortValue::I64(d as i64),
                SortKeyDef::Field { field, kind, missing, mode, desc, cast } => field_sort_value(leaf, d, *field, *kind, *desc, *mode, missing, *cast)?,
                SortKeyDef::NestedField { path, filter, bits, field, kind, missing, mode, desc, cast, max_children } => {
                    if let Some(f) = filter {
                        if bits.as_ref().is_none_or(|(at, _)| *at != li) {
                            let mut b = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                            if let Some(mut sc) = f.scorer(leaf, ctx) {
                                loop {
                                    let c = sc.advance();
                                    if c == TERMINATED {
                                        break;
                                    }
                                    if let Some(w) = b.get_mut(c as usize / 64) {
                                        *w |= 1 << (c % 64);
                                    }
                                }
                            }
                            *bits = Some((li, b));
                        }
                    }
                    let kids: Vec<DocId> = leaf
                        .nested_children(d, path)
                        .into_iter()
                        .filter(|&c| bits.as_ref().is_none_or(|(_, b)| b.get(c as usize / 64).is_some_and(|w| (w >> (c % 64)) & 1 == 1)))
                        .take(max_children.unwrap_or(usize::MAX))
                        .collect();
                    let m = mode.unwrap_or(if *desc { SortMode::Max } else { SortMode::Min });
                    let v = match (*kind, *cast) {
                        (SortKind::F64, Some(c)) => pick_longs(kids.iter().flat_map(|&k| leaf.f64_values(*field, k)).map(|x| c.apply(x)).collect(), m, false),
                        (SortKind::F64, None) => pick_doubles(kids.iter().flat_map(|&k| leaf.f64_values(*field, k)).collect(), m),
                        _ => pick_longs(kids.iter().flat_map(|&k| leaf.i64_values(*field, k)).collect(), m, false),
                    };
                    match (v, &*missing) {
                        (SortValue::Missing, SortMissing::Value(x)) => x.clone(),
                        (v, _) => v,
                    }
                }
                SortKeyDef::TsRoutingHash { .. } => {
                    use base64::Engine as _;
                    leaf.keyword_value(crate::mapping::ID_FIELD, d)
                        .and_then(|id| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id.as_bytes()).ok())
                        .filter(|b| b.len() >= 4)
                        .map(|b| SortValue::Str(b[..4].iter().map(|x| *x as char).collect()))
                        .unwrap_or(SortValue::Missing)
                }
                SortKeyDef::Values { src, string, desc, version, mode, missing } => {
                    let (string, desc, version, mode) = (*string, *desc, *version, *mode);
                    let v = src.with_values(leaf_key, leaf, d, sc, |vs| match vs.len() {
                        0 => SortValue::Missing,
                        1 if !version => val_to_sort(&vs[0], string),
                        _ => pick_values(
                            vs.iter()
                                .map(|v| if version { SortValue::Str(rustic_core::fieldtypes::version::encode(&v.to_java_string())) } else { val_to_sort(v, string) })
                                .collect(),
                            desc,
                            mode,
                        ),
                    })?;
                    match (v, &*missing) {
                        (SortValue::Missing, SortMissing::Value(m)) => m.clone(),
                        (v, _) => v,
                    }
                }
            });
        }
        if k == 0 {
            continue;
        }
        // `search_after`: only hits sorting strictly after the given values.
        if after.is_some_and(|a| cmp_sort(orders, &vals, &a.sort) != std::cmp::Ordering::Greater) {
            continue;
        }
        // Same leaf, later doc: a tie with the k-th best sorts after it.
        if worst.as_ref().is_some_and(|w| cmp_sort(orders, &vals, w) != std::cmp::Ordering::Less) {
            continue;
        }
        out.hits.push(TopHit::new(li, d, sc, vals.clone()));
        if out.hits.len() >= cap {
            out.hits.sort_by(|a, b| cmp_sort(orders, &a.sort, &b.sort).then(a.doc.cmp(&b.doc)));
            out.hits.truncate(k);
            worst = out.hits.last().map(|h| h.sort.clone());
        }
    }
    Ok(out)
}

/// Collect the top `k` hits of `query` sorted by `keys` (leaves in
/// parallel, each with its own executors).
pub fn collect_sorted(leaves: &[LeafReader<'_>], query: &Query, keys: &mut [SortKeyDef], k: usize, min_score: Option<f32>, after: Option<&TopHit>) -> Result<TopDocs> {
    use rayon::prelude::*;
    let ctx = SearchContext::for_query(query, leaves, Bm25Params::default());
    ctx.execution_result()?;
    let orders: Vec<(bool, bool)> = keys.iter().map(SortKeyDef::order).collect();
    let parts: Vec<Result<LeafTop>> = if leaves.len() > 1 && rayon::current_thread_index().is_none() {
        let forks: Vec<Vec<SortKeyDef>> = leaves.iter().map(|_| keys.iter().map(SortKeyDef::fork).collect()).collect();
        let (ctx, orders) = (&ctx, &orders[..]);
        rustic_core::query::searcher::in_search_pool(|| forks.into_par_iter().enumerate().map(|(li, mut ks)| collect_leaf(li, &leaves[li], query, ctx, &mut ks, orders, k, min_score, after)).collect())
    } else {
        leaves.iter().enumerate().map(|(li, leaf)| collect_leaf(li, leaf, query, &ctx, keys, &orders, k, min_score, after)).collect()
    };
    let mut all: Vec<TopHit> = Vec::new();
    let mut total = 0u64;
    let mut max_score = f32::NEG_INFINITY;
    for p in parts {
        let p = p?;
        total += p.total;
        max_score = max_score.max(p.max_score);
        all.extend(p.hits);
    }
    if let Some(e) = env().and_then(|e| e.take_error()) {
        return Err(e);
    }
    all.sort_by(|a, b| cmp_sort(&orders, &a.sort, &b.sort).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)));
    all.truncate(k);
    ctx.execution_result()?;
    Ok(TopDocs { total_hits: total, total_hits_relation: TotalHitsRelation::Eq, max_score, hits: all, terminated_early: false, timed_out: ctx.vector_timed_out.load(std::sync::atomic::Ordering::Relaxed), invalid_score: None, execution_error: ctx.execution_error.lock().clone() })
}


/// Validate request-level `script_fields` against `index.max_script_fields`.
pub fn check_script_fields(v: &Value, max: usize) -> Result<()> {
    let n = v.as_object().map(|o| o.len()).unwrap_or(0);
    if n > max {
        return Err(EsError::illegal_argument(format!(
            "Trying to retrieve too many script_fields. Must be less than or equal to: [{max}] but was [{n}]. This limit can be set by changing the [index.max_script_fields] index level setting."
        )));
    }
    parse_script_fields(v).map(|_| ())
}

/// Add `script_fields` and runtime-field `fields` entries of one hit.
#[allow(clippy::too_many_arguments)]
/// Fetch-phase script state of one search over one index: `script_fields`
/// parsed once and every script / runtime field executor bound once per
/// leaf (not per hit). Owned next to the leaves it binds (dropped with them).
#[derive(Default)]
pub struct FetchScripts(Mutex<FetchInner>);

#[derive(Default)]
struct FetchInner {
    index: String,
    /// The `script_fields` request object, with its sources.
    scripts: Option<(Value, Vec<(String, ValueSource, bool)>)>,
    runtime: std::collections::HashMap<String, Option<ValueSource>>,
}

#[allow(clippy::too_many_arguments)]
pub fn add_hit_fields(
    out: &mut Map<String, Value>,
    raw_fields: &mut std::collections::BTreeMap<String, crate::source_json::Source>,
    index: &str,
    lookup: &FieldLookup,
    leaf: &LeafReader<'_>,
    doc: DocId,
    score: f32,
    script_fields: Option<&Value>,
    fields: &[(String, Option<String>)],
    cache: &FetchScripts,
) -> Result<()> {
    // Concurrent users (e.g. `top_hits` across threads) get a private,
    // uncached state rather than waiting.
    let mut guard = cache.0.try_lock();
    let mut private = FetchInner::default();
    let c: &mut FetchInner = match guard.as_mut() {
        Some(g) => g,
        None => &mut private,
    };
    if c.index != index {
        *c = FetchInner { index: index.to_string(), ..FetchInner::default() };
    }
    let leaf_key = leaf.id();
    for (pat, format) in fields {
        for name in runtime_field_names(index, pat) {
            if let Some((rt, i)) = super::runtime_field(Some(index), &name) {
                if let Some(spec) = &rt.fields[i].lookup {
                    let vals = lookup_field_json(spec, lookup, leaf, doc)?;
                    if !vals.is_empty() {
                        out.insert(name, Value::Array(vals));
                    }
                    continue;
                }
            }
            let Some(vs) = c.runtime.entry(name.clone()).or_insert_with(|| ValueSource::runtime(&name, index)) else { continue };
            let vals = vs.values(leaf_key, leaf, doc, 0.0, None)?;
            let (rt, i) = vs.runtime.as_ref().unwrap();
            let vals: Vec<crate::source_json::Source> = vals.iter().map(|v| {
                if matches!(v, Val::Utf16(_) | Val::Char(_) | Val::Builder(_)) {
                    super::source::to_source(v)
                } else {
                    Ok(crate::source_json::Source::from_value(&rt.fields[*i].render_as(v, format.as_deref())))
                }
            }).collect::<Result<_>>()?;
            if !vals.is_empty() {
                raw_fields.insert(name, crate::source_json::Source::Array(vals));
            }
        }
    }
    if let Some(sf) = script_fields {
        // (Inner hits may ask for other script fields than their parent.)
        if c.scripts.as_ref().is_none_or(|(v, _)| v != sf) {
            c.scripts = Some((sf.clone(), parse_script_fields(sf)?.into_iter().map(|(n, cs, ig)| (n, ValueSource::script(cs, index, lookup), ig)).collect()));
        }
        for (name, vs, ignore) in c.scripts.as_mut().unwrap().1.iter_mut() {
            match vs.values(leaf_key, leaf, doc, score, None) {
                // A script returning a list yields its elements; a single value one element.
                Ok(vals) => {
                    let raw = crate::source_json::Source::Array(vals.iter().map(super::source::to_source).collect::<Result<Vec<_>>>()?);
                    raw_fields.insert(name.clone(), raw);
                }
                Err(_) if *ignore => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}
