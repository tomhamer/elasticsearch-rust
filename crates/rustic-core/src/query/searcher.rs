//! `Searcher`: run a [`Query`] over a set of leaves (Lucene's
//! `IndexSearcher`), with top-k by score or by fields, `from`/`size`
//! paging, `search_after`, total-hit tracking, `terminate_after`, timeouts,
//! field collapsing, rescoring, explanations and concurrent per-leaf
//! execution.

use crate::query::bm25::Bm25Params;
use crate::query::bulk::{Floor, SharedFloor};
use crate::query::collector::{compare_sort_keys, count, FieldDoc, ScoredDoc, SortSpec, SortValue, TopFieldCollector, TopKCollector};
use crate::query::docset::TERMINATED;
use crate::query::explain::Explanation;
use crate::query::query::{Query, SearchContext};
use crate::fielddata::Uninverted;
use crate::query::reader::LeafReader;
use crate::query::sort_fast;
use crate::query::similarity::{default_similarity, Bm25, Similarity};
use crate::types::DocId;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

thread_local! {
    static SEARCH_POOL_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True only for workers belonging to the isolated query pool. Nested work
/// here cannot steal indexing jobs blocked on the caller's read lock.
pub(crate) fn on_search_pool_worker() -> bool {
    SEARCH_POOL_WORKER.with(std::cell::Cell::get)
}

/// Query work must not wait behind indexing jobs blocked on a reader's shard
/// lock. Callers already running on Rayon execute sequentially instead: a
/// cross-pool install can steal jobs from the caller's pool while waiting.
pub fn in_search_pool<R: Send>(run: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    // The installed closure keeps the caller's search task token, whichever
    // worker runs it (one may be inside another search's work).
    let token = crate::cancel::current();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rustic-search-{i}"))
        .start_handler(|_| SEARCH_POOL_WORKER.with(|worker| worker.set(true)))
        .build().expect("create search worker pool"))
        .install(move || crate::cancel::with_search(token, run))
}

/// `f(0..n)` on the search pool, concurrently, results in index order.
/// Sequential where nested work is unsafe (a caller already on another
/// Rayon pool, see [`in_search_pool`]) or pointless (`n <= 1`).
pub fn par_map_search<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    use rayon::prelude::*;
    if n <= 1 || (rayon::current_thread_index().is_some() && !on_search_pool_worker()) {
        return (0..n).map(f).collect();
    }
    // Items run on other workers under the caller's search task token
    // (their cancellation checks read it): one scope per item, set even
    // when the caller has none, so an item a worker steals while inside
    // another search's item never sees that search's token.
    let token = crate::cancel::current();
    let g = |i: usize| crate::cancel::with_search(token.clone(), || f(i));
    if on_search_pool_worker() {
        return (0..n).into_par_iter().map(&g).collect();
    }
    in_search_pool(|| (0..n).into_par_iter().map(&g).collect())
}

/// Threads of the search pool ([`in_search_pool`]).
pub fn search_pool_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| in_search_pool(rayon::current_num_threads))
}

/// What to sort a field by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKind {
    /// i64 fast field (also dates).
    I64,
    /// f64 fast field.
    F64,
    /// Keyword fast field.
    Keyword,
    /// A text field's uninverted terms (ES `fielddata: true`): a doc sorts
    /// by its smallest term (largest for `max` / descending), rendered as
    /// a string like a keyword. Built once per leaf and cached there.
    Fielddata,
}

/// How a multi-valued field picks its sort value (ES `mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortMode {
    /// Smallest value (ES default for ascending sorts).
    Min,
    /// Largest value (ES default for descending sorts).
    Max,
    /// Sum of the values (numeric only; keywords use `Min`).
    Sum,
    /// Mean of the values (numeric only; keywords use `Min`).
    Avg,
    /// Median of the values (numeric only; keywords use `Min`).
    Median,
}

/// Where docs without a value sort (ES `missing`). `First`/`Last` hold
/// regardless of the sort direction.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum SortMissing {
    /// Before every doc with a value.
    First,
    /// After every doc with a value.
    #[default]
    Last,
    /// As if the doc had this value.
    Value(SortValue),
}

/// One sort criterion (Lucene `SortField`). Sorting on a field that no leaf
/// has (ES `unmapped_type`) treats every doc as missing.
#[derive(Clone, Debug)]
pub enum SortField {
    /// By score.
    Score { descending: bool },
    /// By shard-wide doc id: `(leaf << 32) | doc`.
    Doc,
    /// [`SortField::Doc`] descending.
    DocDesc,
    /// By a fast-field value.
    Field { field: u16, kind: SortKind, descending: bool, missing: SortMissing, mode: Option<SortMode> },
    /// By distance in meters from `(lat, lon)` to the doc's points of
    /// geo-point field `field` (ES `_geo_distance`; haversine `arc`, or
    /// `plane`). `mode` picks among several points (min ascending, max
    /// descending by default); docs without a point sort as `+inf`.
    GeoDistance { field: u16, lat: f64, lon: f64, descending: bool, mode: Option<SortMode>, plane: bool },
    /// ES `nested` sort: `sort` (a field or geo-distance sort) evaluated
    /// over the live nested docs of `path` in each hit's block (only those
    /// matching `filter`, if given); the children's values are combined by
    /// the inner sort's mode. A hit without such children is missing.
    Nested { path: String, filter: Option<Box<Query>>, sort: Box<SortField> },
}

impl SortField {
    /// Ascending sort on `field`, missing last, default mode.
    pub fn field(field: u16, kind: SortKind) -> Self {
        SortField::Field { field, kind, descending: false, missing: SortMissing::Last, mode: None }
    }

    /// Make a field / score sort descending.
    pub fn desc(mut self) -> Self {
        if matches!(self, SortField::Doc) {
            return SortField::DocDesc;
        }
        match &mut self {
            SortField::Field { descending, .. } | SortField::Score { descending } | SortField::GeoDistance { descending, .. } => *descending = true,
            SortField::Nested { sort, .. } => {
                let inner = std::mem::replace(&mut **sort, SortField::Doc);
                **sort = inner.desc();
            }
            SortField::Doc | SortField::DocDesc => {}
        }
        self
    }

    /// Whether the sort is descending.
    pub fn is_descending(&self) -> bool {
        match self {
            SortField::Field { descending, .. } | SortField::Score { descending } | SortField::GeoDistance { descending, .. } => *descending,
            SortField::Nested { sort, .. } => sort.is_descending(),
            SortField::Doc => false,
            SortField::DocDesc => true,
        }
    }

    /// Set how missing values sort.
    pub fn with_missing(mut self, m: SortMissing) -> Self {
        match &mut self {
            SortField::Field { missing, .. } => *missing = m,
            SortField::Nested { sort, .. } => {
                let inner = std::mem::replace(&mut **sort, SortField::Doc);
                **sort = inner.with_missing(m);
            }
            _ => {}
        }
        self
    }

    /// Set the multi-value mode.
    pub fn with_mode(mut self, m: SortMode) -> Self {
        match &mut self {
            SortField::Field { mode, .. } | SortField::GeoDistance { mode, .. } => *mode = Some(m),
            SortField::Nested { sort, .. } => {
                let inner = std::mem::replace(&mut **sort, SortField::Doc);
                **sort = inner.with_mode(m);
            }
            _ => {}
        }
        self
    }
}

/// ES `track_total_hits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrackTotalHits {
    /// Count every hit exactly.
    #[default]
    True,
    /// Do not count (`total_hits` is 0 with relation `Gte`).
    False,
    /// Count exactly up to this many; above it report the bound with
    /// relation `Gte` (counting, and for uniformly scored queries
    /// collection itself, stops early).
    UpTo(u64),
}

/// Whether `total_hits` is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TotalHitsRelation {
    /// Exact.
    #[default]
    Eq,
    /// A lower bound.
    Gte,
}

/// How the rescore query's weighted score combines with the weighted
/// original score (ES `QueryRescoreMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RescoreMode {
    #[default]
    Total,
    Multiply,
    Avg,
    Max,
    Min,
    /// The rescore score replaces the original one (ES script rescorer).
    Replace,
}

impl RescoreMode {
    /// Combine the weighted primary and secondary scores.
    pub fn combine(self, primary: f32, secondary: f32) -> f32 {
        match self {
            RescoreMode::Total => primary + secondary,
            RescoreMode::Multiply => primary * secondary,
            RescoreMode::Avg => (primary + secondary) / 2.0,
            RescoreMode::Max => primary.max(secondary),
            RescoreMode::Min => primary.min(secondary),
            RescoreMode::Replace => secondary,
        }
    }

    /// ES's explanation label (`sum`, `product`, `avg`, `max`, `min`).
    pub fn label(self) -> &'static str {
        match self {
            RescoreMode::Total => "sum",
            RescoreMode::Multiply => "product",
            RescoreMode::Avg => "avg",
            RescoreMode::Max => "max",
            RescoreMode::Min => "min",
            RescoreMode::Replace => "replace",
        }
    }
}

/// ES `rescore` (query rescorer, `score_mode: total`): the top
/// `window_size` hits get `query_weight * score + rescore_query_weight *
/// rescore_score` (docs the rescore query does not match keep
/// `query_weight * score`) and are re-sorted ahead of the rest. Only
/// applies to score-sorted searches.
#[derive(Clone, Debug)]
pub struct Rescore {
    /// Number of top hits to rescore.
    pub window_size: usize,
    /// The rescoring query.
    pub query: Query,
    /// Weight of the original score.
    pub query_weight: f32,
    /// Weight of the rescore query's score.
    pub rescore_query_weight: f32,
}

/// Search options (ES `from`, `size`, `sort`, `search_after`,
/// `track_total_hits`, `terminate_after`, `timeout`, `explain`, `collapse`,
/// `rescore`, `min_score`, similarity).
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// Number of hits to return (ES `size`).
    pub top_k: usize,
    /// Number of top hits to skip (ES `from`).
    pub from: usize,
    /// Sort criteria; empty = by score.
    pub sort: Vec<SortField>,
    /// Sort values of the last hit of the previous page, as `(leaf, doc, score, sort values)`.
    pub search_after: Option<TopHit>,
    /// Legacy BM25 parameters: honoured when `similarity` is the default
    /// BM25 and these differ from the defaults.
    pub params: Bm25Params,
    /// Default similarity (BM25).
    pub similarity: Arc<dyn Similarity>,
    /// Per-field similarity overrides.
    pub field_similarity: HashMap<u16, Arc<dyn Similarity>>,
    /// Run leaves concurrently (ignored with `terminate_after`, which
    /// needs a deterministic leaf order).
    pub parallel: bool,
    /// Drop hits scoring below this.
    pub min_score: Option<f32>,
    /// Total-hit counting.
    pub track_total_hits: TrackTotalHits,
    /// Stop after this many matching docs (over all leaves, in leaf/doc
    /// order) and set `terminated_early`.
    pub terminate_after: Option<u64>,
    /// Stop collecting once this much time has elapsed (checked every
    /// 1024 docs per leaf) and set `timed_out`.
    pub timeout: Option<Duration>,
    /// Attach an [`Explanation`] to every returned hit.
    pub explain: bool,
    /// Compute scores with field sorting. Core callers retain scores by
    /// default; ES can disable this when track_scores is not requested.
    pub track_scores: bool,
    /// Keep only the best hit per value of this field (ES `collapse`).
    pub collapse_field: Option<(u16, SortKind)>,
    /// Rescore the top window with a second query.
    pub rescore: Option<Rescore>,
    /// Score combination of the rescore (ES `score_mode`).
    pub rescore_mode: RescoreMode,
    /// Further rescorers applied after `rescore`, in order (ES `rescore`
    /// arrays). A step's query sees the previous pass's window scores in
    /// [`SearchContext::prev_pass`].
    pub rescore_chain: Vec<(Rescore, RescoreMode)>,
    /// Use the specialised single-field sort collector (tests compare it
    /// against the generic one).
    #[doc(hidden)]
    pub fast_sort: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            from: 0,
            sort: Vec::new(),
            search_after: None,
            params: Bm25Params::default(),
            similarity: default_similarity(),
            field_similarity: HashMap::new(),
            parallel: true,
            min_score: None,
            track_total_hits: TrackTotalHits::True,
            terminate_after: None,
            timeout: None,
            explain: false,
            track_scores: true,
            collapse_field: None,
            rescore: None,
            rescore_mode: RescoreMode::Total,
            rescore_chain: Vec::new(),
            fast_sort: true,
        }
    }
}

/// One result.
#[derive(Clone, Debug, PartialEq)]
pub struct TopHit {
    /// Leaf index.
    pub leaf: usize,
    /// Leaf-local doc id.
    pub doc: DocId,
    /// Score (after rescoring).
    pub score: f32,
    /// Sort values (`[score]` for score-sorted searches).
    pub sort: Vec<SortValue>,
    /// Collapse value, when collapsing.
    pub collapse: Option<SortValue>,
    /// Score explanation, when requested.
    pub explanation: Option<Explanation>,
}

impl TopHit {
    /// A hit with no collapse value or explanation.
    pub fn new(leaf: usize, doc: DocId, score: f32, sort: Vec<SortValue>) -> Self {
        Self { leaf, doc, score, sort, collapse: None, explanation: None }
    }
}

/// Search results.
#[derive(Clone, Debug, Default)]
pub struct TopDocs {
    /// Total matching docs (see `total_hits_relation`).
    pub total_hits: u64,
    /// Whether `total_hits` is exact or a lower bound.
    pub total_hits_relation: TotalHitsRelation,
    /// Best score among collected hits.
    pub max_score: f32,
    /// The page of hits.
    pub hits: Vec<TopHit>,
    /// `terminate_after` stopped collection.
    pub terminated_early: bool,
    /// `timeout` stopped collection.
    pub timed_out: bool,
    /// A `function_score` produced a negative / NaN score for this doc.
    pub invalid_score: Option<(DocId, f32)>,
    pub execution_error: Option<String>,
}

/// Hashable collapse key.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CollapseKey {
    I64(i64),
    F64(u64),
    Str(String),
    Missing,
}

impl From<&SortValue> for CollapseKey {
    fn from(v: &SortValue) -> Self {
        match v {
            SortValue::I64(i) => CollapseKey::I64(*i),
            SortValue::F64(f) => CollapseKey::F64(f.to_bits()),
            SortValue::Str(s) => CollapseKey::Str(s.clone()),
            SortValue::Missing => CollapseKey::Missing,
        }
    }
}

/// Leaves collected by [`Searcher::run_leaf_fast`] (tests check it runs).
#[cfg(test)]
static FAST_LEAVES: AtomicU64 = AtomicU64::new(0);

/// State shared by the per-leaf collectors of one search.
struct Shared {
    start: Instant,
    timed_out: AtomicBool,
    matched: AtomicU64,
    /// Hits counted so far by every leaf (published in batches): a uniformly
    /// scored search stops counting once the shard reached the
    /// `track_total_hits` limit, as Lucene's shard-wide hit counting does.
    counted: AtomicU64,
    /// Parallel score-sorted searches: the k-th best score some leaf holds
    /// (Lucene's `MaxScoreAccumulator`).
    floor: Option<SharedFloor>,
    /// The search task's cancellation: polled where the `timeout` is
    /// (every 1024 docs, and before each leaf). A cancelled search stops
    /// like a timed out one; its caller reports the cancellation.
    cancel: Option<Arc<crate::cancel::Token>>,
    /// Leaves no doc of which can match: a required i64 / date range lies
    /// outside the leaf's values ([`Searcher::range_excluded_leaves`]).
    excluded: Vec<bool>,
}

impl Shared {
    /// Whether the search task was cancelled (marks the search stopped, so
    /// the other leaves stop at their next check too).
    #[inline]
    fn cancelled(&self) -> bool {
        if self.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            self.timed_out.store(true, AtomicOrdering::Relaxed);
            return true;
        }
        false
    }
}

/// Per-leaf collection result.
#[derive(Default)]
struct LeafOut {
    hits: Vec<TopHit>,
    /// Score-sorted hits without per-hit options, kept compact (`hits`
    /// stays empty); they become [`TopHit`]s after the merge.
    scored: Vec<ScoredDoc>,
    total: u64,
    /// Counting stopped at the track_total_hits bound.
    saturated: bool,
    terminated_early: bool,
}

enum Gather {
    Score(TopKCollector),
    Field(TopFieldCollector),
    Collapse(HashMap<CollapseKey, TopHit>),
}

/// Runs queries over leaves.
pub struct Searcher<'a> {
    leaves: Vec<LeafReader<'a>>,
}

impl<'a> Searcher<'a> {
    /// A searcher over `leaves` (leaf index = position).
    pub fn new(leaves: Vec<LeafReader<'a>>) -> Self {
        Self { leaves }
    }

    /// The leaves.
    pub fn leaves(&self) -> &[LeafReader<'a>] {
        &self.leaves
    }

    /// Pick a value of a (possibly multi-valued) numeric field.
    fn pick_numeric(mut v: Vec<f64>, mode: SortMode) -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        Some(match mode {
            SortMode::Min => v.iter().copied().fold(f64::INFINITY, f64::min),
            SortMode::Max => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            SortMode::Sum => v.iter().sum(),
            SortMode::Avg => v.iter().sum::<f64>() / v.len() as f64,
            SortMode::Median => {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let n = v.len();
                if n % 2 == 1 {
                    v[n / 2]
                } else {
                    (v[n / 2 - 1] + v[n / 2]) / 2.0
                }
            }
        })
    }

    /// The sort value of one field of a doc.
    fn field_value(ctx:&SearchContext,leaf: &LeafReader<'_>, doc: DocId, field: u16, kind: SortKind, descending: bool, mode: Option<SortMode>) -> SortValue {
        Self::field_value_docs(ctx,leaf, std::slice::from_ref(&doc), field, kind, descending, mode, None)
    }

    /// Per sort field: the leaf's uninverted fielddata of a
    /// [`SortKind::Fielddata`] sort, fetched once per leaf (empty when no
    /// sort uses fielddata, so other sorts pay nothing).
    fn leaf_fielddata(ctx:&SearchContext,leaf: &LeafReader<'_>, sort: &[SortField]) -> Vec<Option<Arc<Uninverted>>> {
        fn fd_field(s: &SortField) -> Option<u16> {
            match s {
                SortField::Field { field, kind: SortKind::Fielddata, .. } => Some(*field),
                SortField::Nested { sort, .. } => fd_field(sort),
                _ => None,
            }
        }
        if !sort.iter().any(|s| fd_field(s).is_some()) {
            return Vec::new();
        }
        sort.iter().map(|s|fd_field(s).and_then(|f|match leaf.uninverted(f){Ok(value)=>Some(value),Err(error)=>{ctx.record_memory_error(error.to_string());None}})).collect()
    }

    /// Open a scalar primary numeric column once per leaf. Most candidates
    /// cannot beat the heap's primary key, so they need no secondary keys or
    /// per-hit allocations. Multi-valued/nested/custom sorts keep their path.
    fn primary_numeric<'r>(leaf: &'r LeafReader<'_>, sort: &[SortField]) -> Option<Box<dyn Fn(DocId) -> SortValue + 'r>> {
        let SortField::Field { field, kind, descending, missing, mode } = sort.first()? else { return None };
        let mode = mode.unwrap_or(if *descending { SortMode::Max } else { SortMode::Min });
        fn wrap<'a, T: 'a>(get: impl Fn(DocId) -> Option<T> + 'a, convert: impl Fn(T) -> SortValue + 'a, missing: SortMissing) -> Box<dyn Fn(DocId) -> SortValue + 'a> {
            Box::new(move |doc| get(doc).map(&convert).unwrap_or_else(|| match &missing {
                SortMissing::Value(value) => value.clone(), _ => SortValue::Missing,
            }))
        }
        let float = move |value: f64| SortValue::F64(match mode {
            // Match the existing folds, including NaN and signed-zero behavior.
            SortMode::Min => f64::INFINITY.min(value), SortMode::Max => f64::NEG_INFINITY.max(value),
            SortMode::Sum => std::iter::once(value).sum(), SortMode::Avg => std::iter::once(value).sum::<f64>() / 1.0, SortMode::Median => value,
        });
        let integer = move |value: i64| match mode {
            SortMode::Avg | SortMode::Median => SortValue::F64(value as f64), _ => SortValue::I64(value),
        };
        match (leaf, kind) {
            (LeafReader::Segment(s), SortKind::F64) => {
                let c = s.fast_fields().get_f64(*field)?;
                Some(wrap(move |d| c.get(d), float, missing.clone()))
            }
            (LeafReader::Memtable(w, ..), SortKind::F64) => {
                let c = w.fast_fields().get_f64(*field)?;
                Some(wrap(move |d| c.get(d), float, missing.clone()))
            }
            (LeafReader::Segment(s), SortKind::I64) => {
                if let Some(c) = s.fast_fields().get_i64(*field) {
                    Some(wrap(move |d| c.get(d), SortValue::I64, missing.clone()))
                } else {
                    let c = s.fast_fields().get_u64(*field)?;
                    Some(wrap(move |d| c.get(d).map(|v| (v ^ (1u64 << 63)) as i64), integer, missing.clone()))
                }
            }
            (LeafReader::Memtable(w, ..), SortKind::I64) => {
                if let Some(c) = w.fast_fields().get_i64(*field) {
                    Some(wrap(move |d| c.get(d), SortValue::I64, missing.clone()))
                } else {
                    let c = w.fast_fields().get_u64(*field)?;
                    Some(wrap(move |d| c.get(d).map(|v| (v ^ (1u64 << 63)) as i64), integer, missing.clone()))
                }
            }
            _ => None,
        }
    }

    /// Java's nearest-long rounding, including negative half ties toward +inf.
    fn round_long(value: f64) -> i64 {
        let floor = value.floor();
        (if value - floor >= 0.5 { floor + 1.0 } else { floor }) as i64
    }

    /// Open a nested scalar column and its block/filter metadata once per leaf.
    /// The hot path reduces child values without allocating per-parent vectors.
    fn nested_i64<'r>(leaf: &'r LeafReader<'_>, sort: &SortField, filter: Option<&'r [u64]>) -> Option<Box<dyn Fn(DocId) -> SortValue + 'r>> {
        let SortField::Nested { path, sort, .. } = sort else { return None };
        let SortField::Field { field, kind: SortKind::I64, descending, missing, mode } = &**sort else { return None };
        let mode = mode.unwrap_or(if *descending { SortMode::Max } else { SortMode::Min });
        let missing = missing.clone();
        let nested = leaf.nested_info()?;
        let ord = nested.path_ord(path)?;
        let deleted = leaf.deletions_only();
        let reduce = move |parent: DocId, get: &dyn Fn(DocId) -> Option<i64>| {
            let mut count = 0u64;
            let mut value = match mode { SortMode::Min => i64::MAX, SortMode::Max => i64::MIN, _ => 0 };
            let mut median = Vec::new();
            for child in nested.children_iter(parent, ord) {
                if deleted.as_ref().is_some_and(|bits| bits.get(child))
                    || filter.is_some_and(|bits| !bits.get(child as usize / 64).is_some_and(|w| (w >> (child % 64)) & 1 == 1)) { continue }
                if let Some(v) = get(child) {
                    count += 1;
                    match mode {
                        SortMode::Min => value = value.min(v),
                        SortMode::Max => value = value.max(v),
                        SortMode::Sum | SortMode::Avg => value = value.wrapping_add(v),
                        SortMode::Median => median.push(v),
                    }
                }
            }
            if count == 0 { return match &missing { SortMissing::Value(v) => v.clone(), _ => SortValue::Missing } }
            match mode {
                SortMode::Avg if count > 1 => SortValue::I64(Self::round_long(value as f64 / count as f64)),
                SortMode::Median => {
                    median.sort_unstable();
                    let m = median.len() / 2;
                    SortValue::I64(if median.len() % 2 == 0 { Self::round_long((median[m-1] as f64 + median[m] as f64) / 2.0) } else { median[m] })
                }
                _ => SortValue::I64(value),
            }
        };
        match leaf {
            LeafReader::Segment(segment) => {
                let fields = segment.fast_fields();
                if fields.get_u64(*field).is_some() || fields.get_multi_u64(*field).is_some() { return None }
                let column = fields.get_i64(*field)?;
                Some(Box::new(move |parent| reduce(parent, &|d| column.get(d))))
            }
            LeafReader::Memtable(writer, ..) => {
                let fields = writer.fast_fields();
                if fields.get_u64(*field).is_some() || fields.get_multi_u64(*field).is_some() { return None }
                let column = fields.get_i64(*field)?;
                Some(Box::new(move |parent| reduce(parent, &|d| column.get(d))))
            }
        }
    }

    /// The sort value of one field over the values of several docs (a
    /// nested sort combines all of a hit's children).
    /// `fd`: the leaf's uninverted `field` for [`SortKind::Fielddata`]
    /// (looked up when not given).
    fn field_value_docs(ctx:&SearchContext,leaf: &LeafReader<'_>, docs: &[DocId], field: u16, kind: SortKind, descending: bool, mode: Option<SortMode>, fd: Option<&Uninverted>) -> SortValue {
        let mode = mode.unwrap_or(if descending { SortMode::Max } else { SortMode::Min });
        match kind {
            SortKind::I64 => {
                let mut v: Vec<i64> = docs.iter().flat_map(|&d| leaf.i64_values(field, d)).collect();
                let unsigned = v.is_empty();
                if unsigned {
                    // u64 columns (unsigned_long) sort as signed longs shifted
                    // by Long.MIN_VALUE, like ES; callers un-shift for display.
                    v = docs.iter().flat_map(|&d| leaf.u64_values(field, d)).map(|x| (x ^ (1u64 << 63)) as i64).collect();
                }
                match (v.len(), mode) {
                    (0, _) => SortValue::Missing,
                    (_, SortMode::Min) => SortValue::I64(*v.iter().min().unwrap()),
                    (_, SortMode::Max) => SortValue::I64(*v.iter().max().unwrap()),
                    (_, SortMode::Sum) => SortValue::I64(v.iter().fold(0i64, |sum, &v| sum.wrapping_add(v))),
                    (1, _) if !unsigned => SortValue::I64(v[0]),
                    (_, SortMode::Avg) if !unsigned => SortValue::I64(Self::round_long(v.iter().fold(0i64, |sum, &v| sum.wrapping_add(v)) as f64 / v.len() as f64)),
                    (_, SortMode::Median) if !unsigned => {
                        v.sort_unstable();
                        let m = v.len()/2;
                        SortValue::I64(if v.len()%2==0 { Self::round_long((v[m-1] as f64 + v[m] as f64)/2.0) } else { v[m] })
                    }
                    (_, m) => Self::pick_numeric(v.into_iter().map(|x| x as f64).collect(), m).map(SortValue::F64).unwrap_or(SortValue::Missing),
                }
            }
            SortKind::F64 => {
                Self::pick_numeric(docs.iter().flat_map(|&d| leaf.f64_values(field, d)).collect(), mode).map(SortValue::F64).unwrap_or(SortValue::Missing)
            }
            SortKind::Keyword => {
                let mut v: Vec<String> = docs.iter().flat_map(|&d| leaf.keyword_values(field, d)).collect();
                if docs.len() > 1 {
                    v.sort();
                }
                let pick = if mode == SortMode::Max { v.last() } else { v.first() };
                pick.cloned().map(SortValue::Str).unwrap_or(SortValue::Missing)
            }
            SortKind::Fielddata => {
                let owned;
                let fd = match fd {
                    Some(f) => f,
                    None => {
                        owned = match leaf.uninverted(field){Ok(value)=>value,Err(error)=>{ctx.record_memory_error(error.to_string());return SortValue::Missing;}};
                        &*owned
                    }
                };
                // Ordinals follow term order: compare them, render only the pick.
                let ord = if mode == SortMode::Max {
                    docs.iter().filter_map(|&d| fd.max_ord(d)).max()
                } else {
                    docs.iter().filter_map(|&d| fd.min_ord(d)).min()
                };
                ord.map(|o| SortValue::Str(fd.term_string(o))).unwrap_or(SortValue::Missing)
            }
        }
    }

    /// Geo distance (meters) from a point over the points of several docs.
    fn geo_distance_docs(leaf: &LeafReader<'_>, docs: &[DocId], field: u16, (lat, lon): (f64, f64), descending: bool, mode: Option<SortMode>, plane: bool) -> SortValue {
        let mode = mode.unwrap_or(if descending { SortMode::Max } else { SortMode::Min });
        let d: Vec<f64> = docs
            .iter()
            .flat_map(|&d| leaf.geo_values(field, d))
            .map(|(a, b)| if plane { crate::geo::plane_meters(lat, lon, a, b) } else { crate::sloppy_math::haversin_meters(lat, lon, a, b) })
            .collect();
        SortValue::F64(Self::pick_numeric(d, mode).unwrap_or(f64::INFINITY))
    }

    /// The value of a field / geo-distance sort over `docs`.
    fn value_over(ctx:&SearchContext,leaf: &LeafReader<'_>, docs: &[DocId], s: &SortField, fd: Option<&Uninverted>) -> SortValue {
        match s {
            SortField::Field { field, kind, descending, missing, mode } => {
                match (Self::field_value_docs(ctx,leaf, docs, *field, *kind, *descending, *mode, fd), missing) {
                    (SortValue::Missing, SortMissing::Value(v)) => v.clone(),
                    (v, _) => v,
                }
            }
            SortField::GeoDistance { field, lat, lon, descending, mode, plane } => {
                Self::geo_distance_docs(leaf, docs, *field, (*lat, *lon), *descending, *mode, *plane)
            }
            SortField::Nested { sort, .. } => Self::value_over(ctx,leaf, docs, sort, fd),
            SortField::Score { .. } | SortField::Doc | SortField::DocDesc => SortValue::Missing,
        }
    }

    /// Per sort field: the filter bitset of a filtered nested sort on `leaf`.
    fn nested_filters(leaf: &LeafReader<'_>, ctx: &SearchContext, sort: &[SortField]) -> Vec<Option<Vec<u64>>> {
        sort.iter()
            .map(|s| match s {
                SortField::Nested { filter: Some(f), .. } => {
                    let mut bits = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                    if let Some(mut sc) = f.scorer(leaf, ctx) {
                        loop {
                            let d = sc.advance();
                            if d == TERMINATED {
                                break;
                            }
                            if let Some(w) = bits.get_mut(d as usize / 64) {
                                *w |= 1 << (d % 64);
                            }
                        }
                    }
                    Some(bits)
                }
                _ => None,
            })
            .collect()
    }

    /// Materialize the sort key of a doc. `nested_filters` comes from
    /// [`Searcher::nested_filters`] (empty when no sort is a filtered
    /// nested sort). `fielddata` comes from [`Searcher::leaf_fielddata`].
    fn sort_key(ctx:&SearchContext,li: usize, leaf: &LeafReader<'_>, doc: DocId, score: f32, sort: &[SortField], nested_filters: &[Option<Vec<u64>>], fielddata: &[Option<Arc<Uninverted>>], nested_numeric: &[Option<Box<dyn Fn(DocId) -> SortValue + '_>>]) -> Vec<SortValue> {
        sort.iter()
            .enumerate()
            .map(|(i, s)| {
                let fd = fielddata.get(i).and_then(|f| f.as_deref());
                match s {
                    SortField::Score { .. } => SortValue::F64(score as f64),
                    SortField::Doc | SortField::DocDesc => SortValue::I64(((li as i64) << 32) | doc as i64),
                    SortField::Field { .. } | SortField::GeoDistance { .. } => Self::value_over(ctx,leaf, std::slice::from_ref(&doc), s, fd),
                    SortField::Nested { path, sort: inner, .. } => {
                        if let Some(Some(get)) = nested_numeric.get(i) { return get(doc) }
                        let mut kids = leaf.nested_children(doc, path);
                        if let Some(Some(bits)) = nested_filters.get(i) {
                            kids.retain(|&d| bits.get(d as usize / 64).is_some_and(|w| (w >> (d % 64)) & 1 == 1));
                        }
                        if kids.is_empty() {
                            // No matching child: the inner sort's missing value.
                            return match &**inner {
                                SortField::Field { missing: SortMissing::Value(v), .. } => v.clone(),
                                SortField::GeoDistance { .. } => SortValue::F64(f64::INFINITY),
                                _ => SortValue::Missing,
                            };
                        }
                        Self::value_over(ctx,leaf, &kids, inner, fd)
                    }
                }
            })
            .collect()
    }

    fn specs(sort: &[SortField]) -> Vec<SortSpec> {
        sort.iter()
            .map(|s| match s {
                SortField::Score { descending } => SortSpec { descending: *descending, missing_first: false },
                SortField::Doc => SortSpec { descending: false, missing_first: false },
                SortField::DocDesc => SortSpec { descending: true, missing_first: false },
                SortField::Field { descending, missing, .. } => {
                    // The comparator reverses descending sorts after placing
                    // missing values, so flip to keep First/Last absolute.
                    let first = matches!(missing, SortMissing::First);
                    SortSpec { descending: *descending, missing_first: if *descending { !first } else { first } }
                }
                SortField::GeoDistance { descending, .. } => SortSpec { descending: *descending, missing_first: false },
                SortField::Nested { sort, .. } => Self::specs(std::slice::from_ref(&**sort)).remove(0),
            })
            .collect()
    }

    fn collapse_value(ctx:&SearchContext,leaf: &LeafReader<'_>, doc: DocId, (field, kind): (u16, SortKind)) -> SortValue {
        Self::field_value(ctx,leaf, doc, field, kind, false, Some(SortMode::Min))
    }

    /// Total order of hits: by score (desc) or by `specs`, then leaf, doc.
    fn cmp_hits(a: &TopHit, b: &TopHit, by_score: bool, specs: &[SortSpec]) -> Ordering {
        if by_score {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.leaf.cmp(&b.leaf))
                .then(a.doc.cmp(&b.doc))
        } else {
            // Ties on every sort value go by global doc order: leaf, then doc
            // (Lucene's doc-id tiebreak over the whole reader).
            compare_sort_keys(&a.sort, a.score, 0, &b.sort, b.score, 0, specs).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc))
        }
    }

    /// The statistics / similarity context `opts` asks for (what
    /// [`Searcher::search`] builds; pass it to
    /// [`Searcher::search_with_context`] to reuse it for further scoring).
    pub fn context(&self, query: &Query, opts: &SearchOptions) -> SearchContext {
        let d = Bm25Params::default();
        let legacy = opts.params.k1 != d.k1 || opts.params.b != d.b;
        let sim: Arc<dyn Similarity> = match opts.similarity.as_bm25() {
            Some(b) if legacy && b == Bm25::default() => Arc::new(Bm25::from_params(opts.params)),
            _ => opts.similarity.clone(),
        };
        SearchContext::with_similarity_deadline(query, &self.leaves, sim, opts.field_similarity.clone(),
            opts.timeout.and_then(|timeout| Instant::now().checked_add(timeout)))
    }

    /// Hits each leaf must return (`from + size`, or the rescore window).
    fn leaf_need(opts: &SearchOptions) -> usize {
        let need = opts.from.saturating_add(opts.top_k);
        if !opts.sort.is_empty() {
            return need;
        }
        opts.rescore.iter().chain(opts.rescore_chain.iter().map(|(r, _)| r)).fold(need, |n, r| n.max(r.window_size))
    }

    /// Whether leaves take the bulk (block-at-a-time) path.
    fn bulk_eligible(opts: &SearchOptions) -> bool {
        opts.sort.is_empty() && opts.min_score.is_none() && opts.collapse_field.is_none() && opts.terminate_after.is_none() && opts.timeout.is_none()
    }

    /// Rough number of postings the query walks, from the stats pass
    /// (`None` when the query has parts whose cost is not known up front).
    fn estimated_postings(&self, query: &Query, ctx: &SearchContext) -> Option<u64> {
        let mut cost = 0u64;
        let mut known = true;
        let mut docs = None;
        let mut num_docs = || *docs.get_or_insert_with(|| self.leaves.iter().map(|l| l.num_docs() as u64).sum::<u64>());
        query.visit(&mut |q| match q {
            Query::Term(t) => cost += ctx.stats(t).doc_freq,
            // Positions cost several times a doc-id walk.
            Query::Phrase { terms, .. } => cost += terms.iter().map(|t| ctx.stats(t).doc_freq * 4).sum::<u64>(),
            // A long / date range touches no doc of a sealed segment whose
            // values all lie outside it (its scorer is empty there, from
            // the segment's min / max): a `should` of a rare term and such
            // a range (big5 `range_field_conjunction_small_range_small_term_query`)
            // is cheap, not a column scan of every leaf.
            Query::RangeI64 { .. } | Query::RangeDate { .. } => {
                let mut ranges = Vec::new();
                q.required_i64_ranges(&mut ranges);
                cost += self.leaves.iter().filter(|leaf| match leaf {
                    LeafReader::Segment(seg) => !ranges.iter().any(|&(field, lo, hi)| seg.min_max_i64(field).is_some_and(|(min, max)| max < lo || min > hi)),
                    LeafReader::Memtable(..) => true,
                }).map(|l| l.num_docs() as u64).sum::<u64>();
            }
            // Column scans touch (at most) every doc.
            Query::RangeU64 { .. } | Query::RangeF64 { .. } | Query::MatchAll => cost += num_docs(),
            Query::Bool { .. } | Query::Boost { .. } | Query::ConstScore { .. } | Query::DisMax { .. } | Query::MatchNone => {}
            _ => known = false,
        });
        known.then_some(cost)
    }

    /// Per leaf: whether a numeric range every match of `query` satisfies
    /// ([`Query::required_i64_ranges`]) lies outside the sealed segment's
    /// min / max of the field (the bounds kept with the segment, as the
    /// range scorer tests them). Memtables are never excluded.
    fn range_excluded_leaves(&self, query: &Query) -> Vec<bool> {
        let mut ranges = Vec::new();
        query.required_i64_ranges(&mut ranges);
        if ranges.is_empty() {
            return Vec::new();
        }
        self.leaves.iter().map(|leaf| match leaf {
            LeafReader::Segment(seg) => ranges.iter().any(|&(field, lo, hi)| seg.min_max_i64(field).is_some_and(|(min, max)| max < lo || min > hi)),
            LeafReader::Memtable(..) => false,
        }).collect()
    }

    /// Leaves a search visits: those [`Searcher::range_excluded_leaves`]
    /// does not exclude (every leaf when it excludes none, an empty list).
    fn live_leaf_count(excluded: &[bool], leaves: usize) -> usize {
        if excluded.is_empty() { leaves } else { excluded.iter().filter(|x| !**x).count() }
    }

    /// Below this many estimated postings, leaves run sequentially (rayon
    /// task overhead exceeds the work, and later leaves can then skip blocks
    /// using the k-th best score of earlier ones).
    const PARALLEL_MIN_POSTINGS: u64 = 1 << 20;

    /// Collect one leaf. `floor`: the k-th best score of the leaves already
    /// collected (hits at or below it cannot make the merged top-k).
    fn run_leaf(&self, i: usize, query: &Query, ctx: &SearchContext, opts: &SearchOptions, shared: &Shared, floor: f32, cap: usize) -> LeafOut {
        use super::stats::{add, add_since, max_since, start, C};
        let t = start();
        // Between segments: a cancelled search visits no further leaf.
        if shared.cancelled() {
            return LeafOut::default();
        }
        // A required range outside the leaf's values: no doc matches, and
        // no scorer needs building (Lucene's point range weight returns no
        // scorer for such a segment).
        if shared.excluded.get(i).copied().unwrap_or(false) {
            return LeafOut::default();
        }
        let out = self.run_leaf_inner(i, query, ctx, opts, shared, floor, cap);
        add(C::leaves, 1);
        add_since(C::leaf_ns, t);
        max_since(C::leaf_max_ns, t);
        out
    }

    /// Whether a scored search may stop at its score bound once counting
    /// is done (the `score_bound` stop of [`Self::run_leaf_inner`]): its
    /// clauses' scorers may then be told it stops early.
    fn may_stop_on_score_bound(opts: &SearchOptions, by_score: bool, count_limit: u64) -> bool {
        by_score && opts.min_score.is_none() && opts.collapse_field.is_none() && opts.rescore.is_none() && opts.rescore_chain.is_empty()
            && opts.search_after.is_none() && opts.terminate_after.is_none() && count_limit != u64::MAX
    }

    fn run_leaf_inner(&self, i: usize, query: &Query, ctx: &SearchContext, opts: &SearchOptions, shared: &Shared, floor: f32, cap: usize) -> LeafOut {
        let leaf = &self.leaves[i];
        let sort = &opts.sort;
        let by_score = sort.is_empty();
        let need = Self::leaf_need(opts).min(cap);
        let count_limit = match opts.track_total_hits {
            TrackTotalHits::True => u64::MAX,
            TrackTotalHits::False => 0,
            TrackTotalHits::UpTo(n) => n.saturating_add(1),
        };
        // Only a count is wanted (no hits, nothing else per hit): a query
        // that knows the leaf's count without a scan (tree node counts, KD
        // leaf counts, `match_all`) answers it without a scorer.
        if need == 0 && by_score && count_limit > 0 && opts.min_score.is_none() && opts.collapse_field.is_none() && opts.terminate_after.is_none() && opts.timeout.is_none() && opts.rescore.is_none() && opts.rescore_chain.is_empty() && opts.search_after.is_none() {
            // A bounded total: only what the leaves counted so far leaves to
            // count, and a query that counts up to a limit stops there
            // (KD-tree or shape-tree nodes taken whole) instead of counting
            // every match of a leaf that adds no hit (a size-10 search's
            // later leaves).
            if count_limit != u64::MAX {
                let remaining = count_limit.saturating_sub(shared.counted.load(AtomicOrdering::Relaxed)).max(1);
                if let Some(n) = query.count_at_least(leaf, remaining) {
                    shared.counted.fetch_add(n, AtomicOrdering::Relaxed);
                    return LeafOut { total: n, saturated: n >= remaining, ..LeafOut::default() };
                }
            }
            if let Some(n) = query.exact_count(leaf) {
                return LeafOut { total: n.min(count_limit), saturated: n > count_limit, ..LeafOut::default() };
            }
        }
        // With uniform scores the best hits are the first ones in doc order,
        // so collection can stop once enough are held and counting is done.
        let uniform = by_score
            && query.scores_uniformly()
            && opts.min_score.is_none()
            && opts.collapse_field.is_none()
            && opts.rescore.is_none()
            && opts.rescore_chain.is_empty()
            && count_limit != u64::MAX;
        // A geo predicate collected that way only needs its first matches
        // in doc order: its scorer may test the doc values in order instead
        // of walking the whole KD tree (same docs, see `geo_scorer`). Hits
        // the leaves before it already counted lower what this leaf must
        // count before the shard's total reaches the limit.
        // Earlier leaves already hold every hit this search returns (`cap`
        // left no room: `need` is 0) and counted past the total limit. With
        // uniform scores this leaf's hits all lose the doc tie-break and its
        // count only adds to a total already reported as `gte`, so it is not
        // visited at all: building its scorer can be the whole cost (a
        // `terms` query of tens of thousands of values looks each up in
        // every leaf's dictionary).
        if uniform && need == 0 && shared.counted.load(AtomicOrdering::Relaxed) > count_limit {
            super::stats::add(super::stats::C::leaves_skipped, 1);
            return LeafOut { saturated: true, ..LeafOut::default() };
        }
        // A bounded total the query counts by itself (e.g. a geo_shape
        // tree's nodes): the scorer only has to produce the first `need`
        // hits in doc order, and collection stops there.
        let early_count = if uniform && need > 0 && count_limit != u64::MAX && opts.min_score.is_none() && opts.collapse_field.is_none()
            && opts.rescore.is_none() && opts.rescore_chain.is_empty() && opts.terminate_after.is_none() && opts.timeout.is_none() && opts.search_after.is_none() && query.takes_geo_hint() {
            // Only what the leaves counted so far leaves to count (their
            // capped counts still sum past the bound when the total does).
            let remaining = count_limit.saturating_sub(shared.counted.load(AtomicOrdering::Relaxed)).max(1);
            let c = query.count_at_least(leaf, remaining);
            if let Some(c) = c {
                shared.counted.fetch_add(c, AtomicOrdering::Relaxed);
            }
            c
        } else {
            None
        };
        // A field sort's best-block walk builds more scorers of the query
        // (one per batch of blocks): they share this one's dense filter
        // words.
        let _dense = (!by_score).then(|| ctx.dense_scope(query, leaf.id()));
        let built = if let Some(_) = early_count {
            super::geo_scorer::with_wanted(need as u64, || query.scorer(leaf, ctx))
        } else if (uniform || Self::may_stop_on_score_bound(opts, by_score, count_limit)) && query.takes_geo_hint() {
            let counted = shared.counted.load(AtomicOrdering::Relaxed);
            super::geo_scorer::with_wanted(count_limit.saturating_sub(counted).max(need as u64).max(1), || query.scorer(leaf, ctx))
        } else {
            query.scorer(leaf, ctx)
        };
        let Some(mut scorer) = built else {
            return LeafOut::default();
        };
        // Bounds are scorer-specific: e.g. a keyword term is constant only
        // when this leaf proves unit frequency and disabled norms. Unlike
        // syntactic uniformity, a Boolean OR can reach its maximum only on
        // documents matching every clause. Stop only once all retained hits
        // reach that bound and the requested hit count has been satisfied.
        let score_bound = if by_score && opts.min_score.is_none()
            && opts.collapse_field.is_none() && opts.rescore.is_none()
            && opts.rescore_chain.is_empty() && opts.search_after.is_none()
            && opts.terminate_after.is_none() && count_limit != u64::MAX {
            scorer.score_upper_bound().filter(|b| b.is_finite() && *b >= 0.0)
        } else { None };
        // Bulk fast path (block-at-a-time term / AND / OR scoring with exact
        // counts): score-sorted top-k without per-hit options.
        if early_count.is_none() && Self::bulk_eligible(opts) {
            let del_bits = leaf.deleted_bits();
            let del = del_bits.as_ref().map(|b| b.words());
            let floor = Floor::new(floor, shared.floor.as_ref()).with_hits(&shared.counted).at_leaf(i);
            // Bounded total (track_total_hits): count to the bound, then
            // skip what cannot compete (block-max / MaxScore pruning, e.g.
            // multi-field disjunctions of dis_max terms). Rescored searches
            // keep exact counts.
            let limit = if opts.rescore.is_none() && opts.rescore_chain.is_empty() { count_limit } else { u64::MAX };
            let mut unpublished = false;
            let out = scorer
                .collect_top_k_count_limited(need, del, limit, &floor)
                .or_else(|| { unpublished = true; scorer.collect_top_k_bulk_min(need, del, floor.get()) });
            if let Some((hits, total)) = out {
                // The count-limited paths publish their counts as they go;
                // publish this one's, so later uniform leaves can be skipped.
                if uniform && unpublished {
                    shared.counted.fetch_add(total, AtomicOrdering::Relaxed);
                }
                super::stats::add(super::stats::C::leaf_bulk, 1);
                return LeafOut {
                    hits: Vec::new(),
                    scored: hits,
                    total: total.min(count_limit),
                    saturated: total > count_limit,
                    terminated_early: false,
                };
            }
        }
        super::stats::add(super::stats::C::leaf_per_doc, 1);
        if opts.fast_sort {
            if let Some(out) = self.run_leaf_fast(ctx,i, query, &mut *scorer, opts, need, count_limit, &shared.counted) {
                return out;
            }
        }
        // match_all: the count is the live docs (capped at a bounded total,
        // e.g. the default 10,000) and the best hits are the first ones, so
        // collection stops once enough are held instead of visiting docs
        // only to count them.
        let precounted = (by_score && opts.min_score.is_none() && opts.collapse_field.is_none()
            && opts.rescore.is_none() && opts.rescore_chain.is_empty() && opts.terminate_after.is_none() && opts.timeout.is_none()
            && opts.search_after.is_none() && Self::is_match_all(query))
            .then(|| sort_fast::live_count(leaf.num_docs(), leaf.deleted_bits().as_ref().map(|b| b.words())))
            .or(early_count);
        let live = leaf.live_docs();
        let specs = Self::specs(sort);
        let nested_filters = if sort.iter().any(|s| matches!(s, SortField::Nested { filter: Some(_), .. })) {
            Self::nested_filters(leaf, ctx, sort)
        } else {
            Vec::new()
        };
        let mut gather = if opts.collapse_field.is_some() {
            Gather::Collapse(HashMap::new())
        } else if by_score {
            Gather::Score(TopKCollector::new(need))
        } else {
            let after = opts.search_after.as_ref().map(|a| {
                // search_after compares (sort values, score desc, doc asc). Across
                // leaves the doc id is not globally ordered, so the tiebreak is
                // neutralised for other leaves (Lucene uses global doc ids).
                let doc = if a.leaf == i {
                    a.doc
                } else if a.leaf < i {
                    0
                } else {
                    u32::MAX - 1
                };
                FieldDoc { doc, score: a.score, sort: a.sort.clone() }
            });
            let c = TopFieldCollector::new(need, specs.clone(), after);
            Gather::Field(if opts.search_after.as_ref().is_some_and(|a| a.leaf < i) { c.with_ties_after() } else { c })
        };
        let primary_numeric = if matches!(gather, Gather::Field(_)) { Self::primary_numeric(leaf, sort) } else { None };
        let fielddata = if by_score { Vec::new() } else { Self::leaf_fielddata(ctx,leaf, sort) };
        if ctx.execution_result().is_err(){return LeafOut::default();}
        let nested_numeric: Vec<_> = sort.iter().enumerate().map(|(i, s)| Self::nested_i64(leaf, s, nested_filters.get(i).and_then(|b| b.as_deref()))).collect();
        let after_hit = opts.search_after.as_ref();
        let mut out = LeafOut::default();
        let mut iter = 0u64;
        let mut held = 0usize;
        // A bounded count the query made was published when it was made.
        let mut published = early_count.unwrap_or(0);
        // Skip per-doc scoring when the sort doesn't need it: the score is
        // only used for min_score filtering, SortField::Score, and the
        // score tiebreaker (which is irrelevant when a field sort is present).
        let needs_score = by_score
            || sort.iter().any(|s| matches!(s, SortField::Score { .. }))
            || opts.min_score.is_some()
            || opts.track_scores || opts.explain
            || opts.rescore.is_some() || !opts.rescore_chain.is_empty();
        let polled = opts.timeout.is_some() || shared.cancel.is_some();
        let mut poll = crate::cancel::Poll::new();
        loop {
            let d = scorer.advance();
            if d == TERMINATED {
                break;
            }
            iter += 1;
            // The `timeout` and cancellation checks share one amortised
            // poll: every 1024 docs (more often while docs are slow), never
            // an atomic per doc.
            if polled && poll.due(iter) {
                if let Some(t) = opts.timeout {
                    if shared.timed_out.load(AtomicOrdering::Relaxed) || shared.start.elapsed() >= t {
                        shared.timed_out.store(true, AtomicOrdering::Relaxed);
                        break;
                    }
                }
                if shared.cancelled() {
                    break;
                }
            }
            if !live(d) {
                continue;
            }
            let s = if needs_score { scorer.score() } else { 0.0 };
            if let Some(ms) = opts.min_score {
                if s < ms {
                    continue;
                }
            }
            if let Some(ta) = opts.terminate_after {
                if shared.matched.fetch_add(1, AtomicOrdering::Relaxed) >= ta {
                    out.terminated_early = true;
                    break;
                }
            }
            if out.total < count_limit {
                out.total += 1;
            } else {
                out.saturated = true;
            }
            match &mut gather {
                Gather::Score(c) => c.collect(d, s),
                Gather::Field(c) => {
                    if primary_numeric.as_ref().or_else(|| nested_numeric.first().and_then(|get| get.as_ref())).is_none_or(|get| c.primary_competitive(&get(d))) {
                        c.collect(FieldDoc { doc: d, score: s, sort: Self::sort_key(ctx,i, leaf, d, s, sort, &nested_filters, &fielddata, &nested_numeric) });
                    }
                }
                Gather::Collapse(map) => {
                    let sv = if by_score { vec![SortValue::F64(s as f64)] } else { Self::sort_key(ctx,i, leaf, d, s, sort, &nested_filters, &fielddata, &nested_numeric) };
                    let mut hit = TopHit::new(i, d, s, sv);
                    if let (false, Some(a)) = (by_score, after_hit) {
                        let adoc = if a.leaf == i { a.doc } else if a.leaf < i { 0 } else { u32::MAX - 1 };
                        let o = compare_sort_keys(&hit.sort, s, d, &a.sort, a.score, adoc, &specs);
                        let tie_after = a.leaf < i && compare_sort_keys(&hit.sort, s, 0, &a.sort, a.score, 0, &specs) == Ordering::Equal;
                        if o != Ordering::Greater && !tie_after {
                            continue;
                        }
                    }
                    let key = Self::collapse_value(ctx,leaf, d, opts.collapse_field.unwrap());
                    hit.collapse = Some(key.clone());
                    let k = CollapseKey::from(&key);
                    match map.get(&k) {
                        Some(old) if Self::cmp_hits(&hit, old, by_score, &specs) != Ordering::Less => {}
                        _ => {
                            map.insert(k, hit);
                        }
                    }
                }
            }
            held += 1;
            if let (Some(n), true) = (precounted, held >= need) {
                out.total = n.min(count_limit);
                out.saturated = n > count_limit;
                break;
            }
            let bound_reached = score_bound.is_some_and(|bound| match &gather {
                Gather::Score(c) => c.min_score().is_some_and(|min| min >= bound),
                _ => false,
            });
            // Scorers visit ascending doc IDs, so later equal-score hits
            // cannot displace retained hits through the document tie-break.
            if (uniform || bound_reached) && held >= need && out.total >= count_limit {
                out.saturated = true;
                break;
            }
            // A leaf whose retained hits all reach the score bound keeps
            // them whatever it visits later (equal scores lose the doc
            // tie-break), so like a uniform one it only counts, and stops
            // once the leaves together are past the limit.
            if (uniform || bound_reached) && held >= need {
                // Other leaves' counts: published every 1024 hits.
                if out.total - published >= 1024 || published == 0 {
                    shared.counted.fetch_add(out.total - published, AtomicOrdering::Relaxed);
                    published = out.total;
                }
                // Past the limit, not at it: the summed total then exceeds it,
                // so the relation is `gte` without guessing whether more
                // documents exist.
                if shared.counted.load(AtomicOrdering::Relaxed) - published + out.total > count_limit {
                    break;
                }
            }
            if score_bound.is_some() && out.total >= count_limit {
                if let Gather::Score(c) = &gather {
                    if let Some(min) = c.min_score() {
                        // Counting is complete and later equal-score docs lose
                        // ties. Safe Boolean scorers may now require clauses
                        // whose absence makes a hit noncompetitive.
                        scorer.set_min_competitive_score(min);
                        out.saturated = true;
                    }
                }
            }
        }
        if uniform || score_bound.is_some() {
            shared.counted.fetch_add(out.total.saturating_sub(published), AtomicOrdering::Relaxed);
        }
        out.hits = match gather {
            Gather::Score(c) => {
                out.scored = c.into_sorted_vec();
                Vec::new()
            }
            Gather::Field(c) => c
                .into_sorted_vec()
                .into_iter()
                .map(|h| {
                    let mut sv = h.sort;
                    for (j, s) in sort.iter().enumerate() {
                        if matches!(s, SortField::Score { .. }) {
                            sv[j] = SortValue::F64(h.score as f64);
                        }
                    }
                    TopHit::new(i, h.doc, h.score, sv)
                })
                .collect(),
            Gather::Collapse(map) => {
                let mut v: Vec<TopHit> = map.into_values().collect();
                v.sort_by(|a, b| Self::cmp_hits(a, b, by_score, &specs));
                v.truncate(need);
                v
            }
        };
        out
    }

    /// Whether the merged hits are in (leaf, doc) order: an index-order
    /// (`_doc`) sort, or a uniformly scoring query sorted by score.
    fn leaf_ordered(query: &Query, opts: &SearchOptions) -> bool {
        if opts.collapse_field.is_some() || opts.min_score.is_some() || opts.rescore.is_some() {
            return false;
        }
        match opts.sort.as_slice() {
            [SortField::Doc] => true,
            [] | [SortField::Score { descending: true }] => query.scores_uniformly(),
            _ => false,
        }
    }

    /// [`Searcher::search_with_context`] may skip whole leaves of this
    /// sorted search by their best key (see [`Searcher::skip_sorted_leaf`]).
    fn sort_skips_leaves(query: &Query, opts: &SearchOptions) -> bool {
        matches!(query, Query::MatchAll)
            && matches!(opts.sort.as_slice(), [SortField::Field { kind: SortKind::I64 | SortKind::F64, .. }])
            && opts.search_after.is_none() && opts.collapse_field.is_none() && opts.min_score.is_none()
            && opts.rescore.is_none() && opts.rescore_chain.is_empty() && opts.terminate_after.is_none()
            && opts.timeout.is_none() && opts.fast_sort
    }

    /// A search sorted by one numeric field that reports no total
    /// (`track_total_hits: false`, e.g. EQL's stage searches): leaves whose
    /// keys all sort before `search_after`, or (visited best key first)
    /// after the page's last hit, cannot change the hits and are skipped
    /// ([`Searcher::run_sorted_leaves`]).
    fn sorted_skips_leaves(query: &Query, opts: &SearchOptions) -> bool {
        !Self::is_match_all(query)
            && matches!(opts.sort.as_slice(), [SortField::Field { kind: SortKind::I64 | SortKind::F64, .. }])
            && opts.track_total_hits == TrackTotalHits::False
            && opts.collapse_field.is_none() && opts.min_score.is_none()
            && opts.rescore.is_none() && opts.rescore_chain.is_empty() && opts.terminate_after.is_none()
            && opts.timeout.is_none() && opts.fast_sort
    }

    /// The leaves of a [`Searcher::sorted_skips_leaves`] search. A leaf is
    /// skipped when its worst key sorts before `search_after`'s; the others
    /// run best key first, a few in order until the page is full, then the
    /// rest (in parallel) except those whose best key sorts after the
    /// page's last hit. Lucene skips segments by their points bounds alike.
    /// Skipped leaves add no hits and, with no total tracked, no count.
    fn run_sorted_leaves(&self, query: &Query, ctx: &SearchContext, opts: &SearchOptions, shared: &Shared, parallel: bool) -> Vec<LeafOut> {
        let n = self.leaves.len();
        let need = Self::leaf_need(opts);
        let specs = Self::specs(&opts.sort);
        let sorts: Vec<Option<sort_fast::LeafSort<'_>>> = self.leaves.iter().map(|l| sort_fast::LeafSort::new(l, &opts.sort[0])).collect();
        let bounds: Vec<Option<(u128, u128)>> = sorts.iter().map(|ls| ls.as_ref().and_then(|ls| Some((ls.best_key()?, ls.worst_key()?)))).collect();
        let mut outs: Vec<Option<LeafOut>> = (0..n).map(|_| None).collect();
        let mut skipped = 0u64;
        let mut order: Vec<usize> = (0..n)
            .filter(|&i| {
                let before = match (&opts.search_after, &sorts[i], bounds[i]) {
                    (Some(a), Some(ls), Some((_, worst))) => a.sort.first().and_then(|v| ls.after_key(v)).is_some_and(|ak| worst < ak),
                    _ => false,
                };
                skipped += u64::from(before);
                !before
            })
            .collect();
        order.sort_by_key(|&i| bounds[i].map_or(0, |b| b.0));
        let mut page: Vec<TopHit> = Vec::new();
        let mut next = 0;
        while next < order.len() && next < 4 && !(need > 0 && page.len() >= need) {
            let i = order[next];
            next += 1;
            let o = self.run_leaf(i, query, ctx, opts, shared, f32::NEG_INFINITY, usize::MAX);
            page.extend(o.hits.iter().cloned());
            page.sort_by(|a, b| Self::cmp_hits(a, b, false, &specs));
            page.truncate(need);
            outs[i] = Some(o);
        }
        let bottom = (need > 0 && page.len() >= need).then(|| page[need - 1].clone());
        let rest: Vec<usize> = order[next..]
            .iter()
            .copied()
            .filter(|&i| {
                let after = match (&bottom, &sorts[i], bounds[i]) {
                    (Some(h), Some(ls), Some((best, _))) => h.sort.first().and_then(|v| ls.after_key(v)).is_some_and(|bk| best > bk),
                    _ => false,
                };
                skipped += u64::from(after);
                !after
            })
            .collect();
        if skipped > 0 {
            super::stats::add(super::stats::C::leaves_skipped, skipped);
        }
        let run = |i: usize| (i, self.run_leaf(i, query, ctx, opts, shared, f32::NEG_INFINITY, usize::MAX));
        let ran: Vec<(usize, LeafOut)> = if parallel && rest.len() > 1 {
            in_search_pool(|| rest.par_iter().map(|&i| run(i)).collect())
        } else {
            rest.iter().map(|&i| run(i)).collect()
        };
        for (i, o) in ran {
            outs[i] = Some(o);
        }
        outs.into_iter().map(Option::unwrap_or_default).collect()
    }

    /// The leaves of a [`Searcher::sort_skips_leaves`] search with a page
    /// to fill (`need > 0`): best key first (Lucene visits segments by
    /// their points bounds alike), one at a time until the page is full,
    /// then every other leaf that can still place a hit, together on the
    /// search pool when `parallel`. A leaf is skipped (its count from its
    /// live docs) when its best key sorts after the page's last hit, or
    /// equals it with a later leaf index (it loses the tie-break). Visiting
    /// the best leaves first fills the page with the hits that rule out
    /// most others; running the rest together does not serialise them.
    fn run_sort_skip_leaves(&self, query: &Query, ctx: &SearchContext, opts: &SearchOptions, shared: &Shared, parallel: bool) -> Vec<LeafOut> {
        let n = self.leaves.len();
        let need = Self::leaf_need(opts);
        let specs = Self::specs(&opts.sort);
        let best: Vec<Option<u128>> = self.leaves.iter().map(|l| sort_fast::LeafSort::new(l, &opts.sort[0]).and_then(|ls| ls.best_key())).collect();
        // Known bounds best first (ties by leaf), then the leaves without.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (best[i].is_none(), best[i].unwrap_or(0), i));
        let mut outs: Vec<Option<LeafOut>> = (0..n).map(|_| None).collect();
        let mut page: Vec<TopHit> = Vec::new();
        let mut next = 0;
        while next < order.len() && page.len() < need {
            let i = order[next];
            next += 1;
            let o = self.run_leaf(i, query, ctx, opts, shared, f32::NEG_INFINITY, usize::MAX);
            page.extend(o.hits.iter().cloned());
            page.sort_by(|a, b| Self::cmp_hits(a, b, false, &specs));
            page.truncate(need);
            outs[i] = Some(o);
        }
        let mut rest: Vec<usize> = Vec::new();
        for &i in &order[next..] {
            let skipped = match (page.len() >= need).then(|| &page[need - 1]) {
                Some(bottom) => self.skip_sorted_leaf_ordered(i, opts, bottom),
                None => None,
            };
            match skipped {
                Some(o) => outs[i] = Some(o),
                None => rest.push(i),
            }
        }
        let run = |i: usize| (i, self.run_leaf(i, query, ctx, opts, shared, f32::NEG_INFINITY, usize::MAX));
        let ran: Vec<(usize, LeafOut)> = if parallel && rest.len() > 1 {
            in_search_pool(|| rest.par_iter().map(|&i| run(i)).collect())
        } else {
            rest.iter().map(|&i| run(i)).collect()
        };
        for (i, o) in ran {
            outs[i] = Some(o);
        }
        outs.into_iter().map(Option::unwrap_or_default).collect()
    }

    /// [`Searcher::skip_sorted_leaf`] for leaves visited out of index
    /// order: a best key equal to `bottom`'s skips the leaf only when it
    /// comes after `bottom`'s leaf (otherwise its hit would win the tie).
    fn skip_sorted_leaf_ordered(&self, i: usize, opts: &SearchOptions, bottom: &TopHit) -> Option<LeafOut> {
        if i < bottom.leaf {
            let ls = sort_fast::LeafSort::new(&self.leaves[i], &opts.sort[0])?;
            let (best, bottom_key) = (ls.best_key()?, ls.after_key(bottom.sort.first()?)?);
            if best <= bottom_key {
                return None;
            }
        }
        self.skip_sorted_leaf(i, opts, bottom)
    }

    /// Leaf `i` of a [`Searcher::sort_skips_leaves`] search when no doc of
    /// it can enter the page: its best key sorts after `bottom` (the page's
    /// last hit, from an earlier leaf, so an equal key loses the leaf
    /// tie-break too). Its match_all count is its live docs. `None`: visit it.
    fn skip_sorted_leaf(&self, i: usize, opts: &SearchOptions, bottom: &TopHit) -> Option<LeafOut> {
        let leaf = &self.leaves[i];
        let ls = sort_fast::LeafSort::new(leaf, &opts.sort[0])?;
        let best = ls.best_key()?;
        let bottom_key = ls.after_key(bottom.sort.first()?)?;
        if best < bottom_key {
            return None;
        }
        let count_limit = match opts.track_total_hits {
            TrackTotalHits::True => u64::MAX,
            TrackTotalHits::False => 0,
            TrackTotalHits::UpTo(n) => n.saturating_add(1),
        };
        let n = sort_fast::live_count(leaf.num_docs(), leaf.deleted_bits().as_ref().map(|b| b.words()));
        super::stats::add(super::stats::C::leaves_skipped, 1);
        Some(LeafOut { total: n.min(count_limit), saturated: n > count_limit, ..LeafOut::default() })
    }

    /// Whether `q` matches every (live root) doc.
    fn is_match_all(q: &Query) -> bool {
        match q {
            Query::MatchAll => true,
            Query::Boost { query, .. } | Query::ConstScore { query, .. } => {
                Self::is_match_all(query)
            }
            _ => false,
        }
    }

    /// Specialised collection of one leaf, when `opts` allows it: a single
    /// field sort (typed keys, block skipping; see [`sort_fast`]) or a
    /// score sort resumed after a hit (scroll / `search_after` on `_score`).
    fn run_leaf_fast(
        &self,
        ctx:&SearchContext,
        i: usize,
        query: &Query,
        scorer: &mut dyn crate::query::scorer::Scorer,
        opts: &SearchOptions,
        need: usize,
        count_limit: u64,
        shard_counted: &std::sync::atomic::AtomicU64,
    ) -> Option<LeafOut> {
        let out = self.run_leaf_fast_inner(ctx,i, query, scorer, opts, need, count_limit, shard_counted);
        #[cfg(test)]
        if out.is_some() {
            FAST_LEAVES.fetch_add(1, AtomicOrdering::Relaxed);
        }
        out
    }

    fn run_leaf_fast_inner(
        &self,
        ctx:&SearchContext,
        i: usize,
        query: &Query,
        scorer: &mut dyn crate::query::scorer::Scorer,
        opts: &SearchOptions,
        need: usize,
        count_limit: u64,
        shard_counted: &std::sync::atomic::AtomicU64,
    ) -> Option<LeafOut> {
        if opts.sort.len() != 1
            || opts.collapse_field.is_some()
            || opts.terminate_after.is_some()
            || opts.timeout.is_some()
        {
            return None;
        }
        let needs_scores = opts.track_scores || opts.explain || opts.min_score.is_some() || opts.rescore.is_some() || !opts.rescore_chain.is_empty();
        let leaf = &self.leaves[i];
        // search_after's leaf-local tiebreak: hits of earlier leaves sort
        // before it, of later leaves after it.
        let after_doc = |a: &TopHit| {
            if a.leaf == i {
                a.doc as i64
            } else if a.leaf < i {
                -1
            } else {
                i64::MAX
            }
        };
        let del = leaf.deleted_bits();
        let deleted = del.as_ref().map(|b| b.words());
        let precounted = (count_limit == u64::MAX && opts.min_score.is_none() && Self::is_match_all(query))
            .then(|| sort_fast::live_count(leaf.num_docs(), deleted));
        match &opts.sort[0] {
            SortField::Score { descending: true } if opts.rescore.is_none() => {
                let a = opts.search_after.as_ref()?;
                let x = match a.sort.first()? {
                    SortValue::F64(x) => *x,
                    SortValue::I64(x) => *x as f64,
                    _ => return None,
                };
                Some(Self::collect_score_after(
                    i,
                    query,
                    scorer,
                    opts,
                    need,
                    count_limit,
                    precounted,
                    (x, after_doc(a)),
                    deleted,
                ))
            }
            SortField::Doc => {
                // Index order: the hits are the first live matches after the
                // cursor, so once counting is over collection seeks to it
                // and stops at `need` (ES scroll / search_after on `_doc`).
                let lo = (i as i64) << 32;
                let start: Option<DocId> = match &opts.search_after {
                    None => Some(0),
                    Some(a) => match a.sort.first()? {
                        SortValue::I64(g) if *g < lo => Some(0),
                        SortValue::I64(g) if *g >= lo + (1 << 32) - 1 => None,
                        SortValue::I64(g) => Some((g - lo + 1) as DocId),
                        _ => return None,
                    },
                };
                let live = |d: DocId| {
                    deleted.is_none_or(|w| {
                        w.get(d as usize >> 6)
                            .is_none_or(|x| (x >> (d & 63)) & 1 == 0)
                    })
                };
                let counting = precounted.is_none();
                let (mut total, mut saturated) = match precounted {
                    Some(n) => (n.min(count_limit), n > count_limit),
                    None => (0u64, false),
                };
                let mut hits: Vec<TopHit> = Vec::new();
                let mut d = scorer.advance();
                while d != TERMINATED {
                    if !counting || total >= count_limit {
                        match start {
                            Some(st) if hits.len() < need => {
                                if d < st {
                                    d = scorer.seek(st);
                                    continue;
                                }
                            }
                            _ => {
                                saturated |= counting;
                                break;
                            }
                        }
                    }
                    if live(d) {
                        if opts.min_score.is_some_and(|ms| scorer.score() < ms) {
                            d = scorer.advance();
                            continue;
                        }
                        if counting {
                            if total < count_limit {
                                total += 1;
                            } else {
                                saturated = true;
                            }
                        }
                        if start.is_some_and(|st| d >= st) && hits.len() < need {
                            let sc = if needs_scores {
                                scorer.score()
                            } else {
                                0.0
                            };
                            hits.push(TopHit::new(i, d, sc, vec![SortValue::I64(lo | d as i64)]));
                        }
                    }
                    d = scorer.advance();
                }
                Some(LeafOut {
                    hits,
                    scored: Vec::new(),
                    total,
                    saturated,
                    terminated_early: false,
                })
            }
            sf @ (SortField::Field { .. } | SortField::GeoDistance { .. }) => {
                if opts.min_score.is_some() {
                    return None;
                }
                let ls = sort_fast::LeafSort::new(leaf, sf)?;
                let after = match &opts.search_after {
                    Some(a) => Some((ls.after_key(a.sort.first()?)?, after_doc(a))),
                    None => None,
                };
                let p = sort_fast::Params {
                    need,
                    count_limit,
                    precounted,
                    after,
                    scores: needs_scores,
                    deleted,
                    shard_counted: Some(shard_counted),
                };
                // match_all: the hits come straight off the points index, or
                // (no points: a skipper column) off its blocks, best first.
                if Self::is_match_all(query) {
                    let n = precounted.unwrap_or_else(|| sort_fast::live_count(leaf.num_docs(), deleted));
                    let score = match query {
                        Query::MatchAll => Some(if needs_scores { 1.0 } else { 0.0 }),
                        _ => (!needs_scores).then_some(0.0),
                    };
                    if let Some(hits) = score.and_then(|sc| sort_fast::collect_all_docs(&ls, &p, sc).or_else(|| sort_fast::collect_all_blocks(&ls, &p, sc))) {
                        let hits = hits.into_iter().map(|(d, s)| TopHit::new(i, d, s, Self::sort_key(ctx,i, leaf, d, s, &opts.sort, &[], &[], &[]))).collect();
                        return Some(LeafOut { hits, scored: Vec::new(), total: n.min(count_limit), saturated: n > count_limit, terminated_early: false });
                    }
                }
                let fresh = || query.scorer(leaf, ctx);
                let c = sort_fast::collect(&ls, scorer, &p, Some(&fresh));
                let hits = c
                    .hits
                    .into_iter()
                    .map(|(d, s)| {
                        TopHit::new(i, d, s, Self::sort_key(ctx,i, leaf, d, s, &opts.sort, &[], &[], &[]))
                    })
                    .collect();
                Some(LeafOut {
                    hits,
                    scored: Vec::new(),
                    total: c.total,
                    saturated: c.saturated,
                    terminated_early: false,
                })
            }
            _ => None,
        }
    }

    /// Score-sorted collection of the hits after `(score, leaf-local doc)`
    /// (ties by doc). A uniformly scoring query yields its hits in doc
    /// order, so once counting is over it seeks straight past the cursor
    /// and stops after `need` hits (ES scroll over a constant-score query).
    #[allow(clippy::too_many_arguments)]
    fn collect_score_after(
        i: usize,
        query: &Query,
        scorer: &mut dyn crate::query::scorer::Scorer,
        opts: &SearchOptions,
        need: usize,
        count_limit: u64,
        precounted: Option<u64>,
        (x, adoc): (f64, i64),
        deleted: Option<&[u64]>,
    ) -> LeafOut {
        let uniform = query.scores_uniformly() && opts.min_score.is_none();
        let counting = precounted.is_none();
        let (mut total, mut saturated) = match precounted {
            Some(n) => (n.min(count_limit), n > count_limit),
            None => (0u64, false),
        };
        let live = |d: DocId| {
            deleted.is_none_or(|w| {
                w.get(d as usize >> 6)
                    .is_none_or(|x| (x >> (d & 63)) & 1 == 0)
            })
        };
        let mut c = TopKCollector::new(need);
        let mut held = 0usize;
        let mut d = scorer.advance();
        while d != TERMINATED {
            if !live(d) {
                d = scorer.advance();
                continue;
            }
            let s = scorer.score();
            if opts.min_score.is_some_and(|ms| s < ms) {
                d = scorer.advance();
                continue;
            }
            if counting {
                if total < count_limit {
                    total += 1;
                } else {
                    saturated = true;
                }
            }
            let sf = s as f64;
            let is_after = sf < x || (sf == x && d as i64 > adoc);
            if is_after {
                c.collect(d, s);
                held += 1;
            }
            if uniform && (!counting || total >= count_limit) {
                if held >= need {
                    saturated |= counting;
                    break;
                }
                if !is_after {
                    // Every doc scores `sf`: none is after the cursor if it
                    // scores higher or ties on an earlier leaf; on the
                    // cursor's leaf the next candidate is `adoc + 1`.
                    if sf > x || adoc == i64::MAX {
                        break;
                    }
                    if sf == x && (d as i64) < adoc {
                        d = scorer.seek((adoc + 1) as DocId);
                        continue;
                    }
                }
            }
            d = scorer.advance();
        }
        let hits = c
            .into_sorted_vec()
            .into_iter()
            .map(|h| TopHit::new(i, h.doc, h.score, vec![SortValue::F64(h.score as f64)]))
            .collect();
        LeafOut {
            hits,
            scored: Vec::new(),
            total,
            saturated,
            terminated_early: false,
        }
    }

    /// Run `query` and return the top hits per `opts`.
    pub fn search(&self, query: &Query, opts: &SearchOptions) -> TopDocs {
        let ctx = self.context(query, opts);
        self.search_with_context(query, opts, &ctx)
    }

    /// [`Searcher::search`] with a context already built by
    /// [`Searcher::context`] for the same query, options and leaves.
    pub fn search_with_context(&self, query: &Query, opts: &SearchOptions, ctx: &SearchContext) -> TopDocs {
        let started = super::stats::start();
        if let Err(error) = ctx.execution_result() { return TopDocs { execution_error:Some(error.to_string()), ..Default::default() }; }
        let mut out = self.search_with_context_inner(query, opts, ctx);
        if let Err(error) = ctx.execution_result() { out.execution_error=Some(error.to_string()); }
        super::stats::add_since(super::stats::C::search_ns, started);
        out
    }

    fn search_with_context_inner(&self, query: &Query, opts: &SearchOptions, ctx: &SearchContext) -> TopDocs {
        let excluded = self.range_excluded_leaves(query);
        // Leaves left to visit: with at most one, fanning the others out
        // only costs the hand-off to the pool (a `range` matching nothing on
        // a 50-segment write index woke every worker for empty leaves).
        let live_leaves = Self::live_leaf_count(&excluded, self.leaves.len());
        let mut shared = Shared { start: Instant::now(), timed_out: AtomicBool::new(false), matched: AtomicU64::new(0), counted: AtomicU64::new(0), floor: None,
            cancel: ctx.cancel.clone(), excluded };
        let cheap = || self.estimated_postings(query, &ctx).is_some_and(|c| c < Self::PARALLEL_MIN_POSTINGS);
        // Hits ordered by (leaf, doc): once earlier leaves hold enough, later
        // leaves only count.
        let leaf_ordered = Self::leaf_ordered(query, opts);
        // A match_all sorted by one numeric field: leaves run in order, and
        // one whose best possible key cannot beat the page held so far is
        // not visited (Lucene skips such segments by their points min/max).
        let sort_skips = Self::sort_skips_leaves(query, opts);
        let is_cheap = cheap();
        let in_pool = rayon::current_thread_index().is_some();
        let ordered_only = leaf_ordered && (opts.track_total_hits != TrackTotalHits::True || Self::is_match_all(query));
        let sorted_skips = !sort_skips && self.leaves.len() > 1 && Self::sorted_skips_leaves(query, opts);
        let sequential = !opts.parallel
            || self.leaves.len() <= 1
            || live_leaves <= 1
            || sort_skips
            || opts.terminate_after.is_some()
            || ordered_only
            || is_cheap || in_pool;
        // Leaf-ordered collection of costly leaves runs them in parallel
        // batches, in leaf order: later batches still see what earlier ones
        // hold and counted (and skip leaves that cannot add anything), but a
        // search that has to visit every leaf (its total stays under the
        // limit) is not serialised over all of them.
        let batch = if ordered_only && opts.parallel && !sort_skips && opts.terminate_after.is_none() && !is_cheap && !in_pool && self.leaves.len() > 1 && live_leaves > 1 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1)
        } else { 1 };
        if !sequential && Self::bulk_eligible(opts) && Self::leaf_need(opts) > 0 {
            shared.floor = Some(SharedFloor::new());
        }
        let shared = shared;
        let per_leaf: Vec<LeafOut> = if sorted_skips {
            self.run_sorted_leaves(query, ctx, opts, &shared, !sequential)
        } else if sort_skips && self.leaves.len() > 1 && Self::leaf_need(opts) > 0 {
            self.run_sort_skip_leaves(query, ctx, opts, &shared, opts.parallel && !is_cheap && !in_pool)
        } else if sequential {
            // Carry the merged k-th best score into later leaves so their
            // bulk scorers skip blocks that cannot compete.
            let share = Self::bulk_eligible(opts) && self.leaves.len() > 1;
            let need = Self::leaf_need(opts);
            let mut best: Vec<f32> = Vec::new();
            let mut floor = f32::NEG_INFINITY;
            let mut out = Vec::with_capacity(self.leaves.len());
            let mut cap = usize::MAX;
            // `sort_skips`: the best `need` hits so far, in sort order.
            let mut page: Vec<TopHit> = Vec::new();
            let specs = Self::specs(&opts.sort);
            let mut start = 0;
            while start < self.leaves.len() {
                let end = (start + batch).min(self.leaves.len());
                // Leaves of one batch run together with the cap and floor as
                // of the batch start (they may hold a few more hits than the
                // page needs; the merge truncates).
                let outs: Vec<LeafOut> = if end - start > 1 {
                    let (cap_now, floor_now) = (cap, floor);
                    in_search_pool(|| (start..end).into_par_iter().map(|i| self.run_leaf(i, query, &ctx, opts, &shared, floor_now, cap_now)).collect())
                } else {
                    Vec::new()
                };
                let mut outs = outs.into_iter();
                for i in start..end {
                if sort_skips && need > 0 && page.len() >= need {
                    if let Some(o) = self.skip_sorted_leaf(i, opts, &page[need - 1]) {
                        out.push(o);
                        continue;
                    }
                }
                let o = match outs.next() { Some(o) => o, None => self.run_leaf(i, query, &ctx, opts, &shared, floor, cap) };
                if sort_skips && need > 0 {
                    page.extend(o.hits.iter().cloned());
                    page.sort_by(|a, b| Self::cmp_hits(a, b, false, &specs));
                    page.truncate(need);
                }
                if leaf_ordered {
                    cap = cap.min(need).saturating_sub(o.hits.len() + o.scored.len());
                }
                if share && need > 0 && !o.scored.is_empty() {
                    best.extend(o.scored.iter().map(|h| h.score));
                    best.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                    best.truncate(need);
                    if best.len() == need && best[need - 1] > floor {
                        floor = best[need - 1];
                    }
                }
                out.push(o);
                }
                start = end;
            }
            out
        } else {
            // Leaves publish their k-th best score as they go (`shared.floor`);
            // bulk scorers of the others skip what cannot beat it.
            let need = Self::leaf_need(opts);
            in_search_pool(|| (0..self.leaves.len()).into_par_iter().map(|i| {
                let o = self.run_leaf(i, query, &ctx, opts, &shared, f32::NEG_INFINITY, usize::MAX);
                if let (Some(f), Some(h)) = (&shared.floor, o.scored.get(need.wrapping_sub(1))) {
                    f.raise(h.score);
                }
                o
            }).collect())
        };

        let by_score = opts.sort.is_empty();
        let specs = Self::specs(&opts.sort);
        let mut total = 0u64;
        let mut saturated = false;
        let mut terminated_early = false;
        let mut hits: Vec<TopHit> = Vec::with_capacity(per_leaf.iter().map(|o| o.hits.len()).sum());
        let mut scored: Vec<(f32, usize, DocId)> = Vec::with_capacity(per_leaf.iter().map(|o| o.scored.len()).sum());
        for (i, o) in per_leaf.into_iter().enumerate() {
            total += o.total;
            saturated |= o.saturated;
            terminated_early |= o.terminated_early;
            hits.extend(o.hits);
            scored.extend(o.scored.into_iter().map(|h| (h.score, i, h.doc)));
        }
        if !scored.is_empty() {
            // Merge the compact score-sorted hits (score desc, leaf, doc) and
            // materialize only what can be returned or rescored.
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            scored.truncate(Self::leaf_need(opts));
            hits.extend(scored.into_iter().map(|(s, i, d)| TopHit::new(i, d, s, Vec::new())));
        }
        let (total_hits, total_hits_relation) = match opts.track_total_hits {
            TrackTotalHits::True => (total, TotalHitsRelation::Eq),
            TrackTotalHits::False => (0, TotalHitsRelation::Gte),
            TrackTotalHits::UpTo(n) => {
                if saturated || total > n {
                    (n, TotalHitsRelation::Gte)
                } else {
                    (total, TotalHitsRelation::Eq)
                }
            }
        };

        hits.sort_by(|a, b| Self::cmp_hits(a, b, by_score, &specs));
        if opts.collapse_field.is_some() {
            let mut seen = std::collections::HashSet::new();
            hits.retain(|h| seen.insert(CollapseKey::from(h.collapse.as_ref().unwrap_or(&SortValue::Missing))));
        }
        let need = opts.from.saturating_add(opts.top_k);

        // Rescore the top window (score-sorted searches only). Hits beyond the
        // window keep their order and only get `query_weight` applied (ES).
        // value: Some(rescore score) when the rescore query matched the doc.
        // Per step: (leaf, doc) -> (Some(rescore score) when the step's query
        // matched, whether the doc was in the step's window).
        let steps: Vec<(&Rescore, RescoreMode)> = if by_score { opts.rescore.iter().map(|r| (r, opts.rescore_mode)).chain(opts.rescore_chain.iter().map(|(r, m)| (r, *m))).collect() } else { Vec::new() };
        let mut rescored: Vec<HashMap<(usize, DocId), (Option<f32>, bool)>> = Vec::with_capacity(steps.len());
        let mut rescore_ctxs: Vec<SearchContext> = Vec::with_capacity(steps.len());
        for (r, mode) in &steps {
            let mut step_map: HashMap<(usize, DocId), (Option<f32>, bool)> = HashMap::new();
            let window = r.window_size.min(hits.len());
            let mut rctx = SearchContext::with_similarity_deadline_limit(&r.query, &self.leaves, ctx.similarity.clone(), ctx.field_similarity.clone(), ctx.deadline, ctx.max_clause_count);
            rctx.cancel = ctx.cancel.clone();
            if rctx.execution_result().is_err() {
                ctx.inherit_errors(&rctx);
                return TopDocs::default();
            }
            let mut by_leaf: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, h) in hits[..window].iter().enumerate() {
                by_leaf.entry(h.leaf).or_default().push(idx);
            }
            for (leaf_idx, idxs) in &by_leaf {
                let mut prev: Vec<(DocId, f32)> = idxs.iter().map(|&j| (hits[j].doc, hits[j].score)).collect();
                prev.sort_by_key(|(d, _)| *d);
                rctx.prev_pass.insert(self.leaves[*leaf_idx].id(), prev);
            }
            for (leaf_idx, mut idxs) in by_leaf {
                idxs.sort_by_key(|&j| hits[j].doc);
                let mut scorer = r.query.scorer(&self.leaves[leaf_idx], &rctx);
                for j in idxs {
                    let doc = hits[j].doc;
                    let extra = match scorer.as_mut() {
                        Some(s) => {
                            let cur = s.doc();
                            let at = if cur != TERMINATED && cur >= doc { cur } else { s.seek(doc) };
                            if at == doc { Some(s.score()) } else { None }
                        }
                        None => None,
                    };
                    let primary = r.query_weight * hits[j].score;
                    let new = match extra {
                        Some(e) => mode.combine(primary, r.rescore_query_weight * e),
                        None => primary,
                    };
                    step_map.insert((leaf_idx, doc), (extra, true));
                    hits[j].score = new;
                    hits[j].sort = vec![SortValue::F64(new as f64)];
                }
            }
            for h in hits[window..].iter_mut() {
                step_map.insert((h.leaf, h.doc), (None, false));
                h.score *= r.query_weight;
                h.sort = vec![SortValue::F64(h.score as f64)];
            }
            if *mode == RescoreMode::Replace {
                // `ScriptRescorer.combine`: every hit is re-sorted.
                hits.sort_by(|a, b| Self::cmp_hits(a, b, true, &[]));
            } else {
                hits[..window].sort_by(|a, b| Self::cmp_hits(a, b, true, &[]));
            }
            rescored.push(step_map);
            ctx.inherit_errors(&rctx);
            rescore_ctxs.push(rctx);
        }
        hits.truncate(need);
        let max_score = hits.iter().map(|h| h.score).fold(0.0f32, f32::max);
        let mut hits: Vec<TopHit> = hits.into_iter().skip(opts.from).take(opts.top_k).collect();
        if by_score {
            for h in &mut hits {
                if h.sort.is_empty() {
                    h.sort = vec![SortValue::F64(h.score as f64)];
                }
            }
        }

        if opts.explain {
            for h in &mut hits {
                let leaf = &self.leaves[h.leaf];
                let mut e = query.explain(leaf, &ctx, h.doc);
                for (((r, mode), rctx), step_map) in steps.iter().zip(&rescore_ctxs).zip(&rescored) {
                    let Some((extra, in_window)) = step_map.get(&(h.leaf, h.doc)) else { continue };
                    if *mode == RescoreMode::Replace {
                        // `ScriptRescorer.explain`.
                        if let (Some(x), true) = (extra, in_window) {
                            let re = r.query.explain(leaf, rctx, h.doc);
                            let src_value = e.value;
                            e = Explanation::new(*x, format!("{}, computed as:", re.description), vec![Explanation::new(src_value, "_score: ", vec![e])]);
                        }
                        continue;
                    }
                    let pv = e.value * r.query_weight;
                    let prim = Explanation::new(pv, "product of:", vec![e, Explanation::leaf(r.query_weight, "primaryWeight")]);
                    e = match (extra, in_window) {
                        (Some(x), true) => {
                            let re = r.query.explain(leaf, rctx, h.doc);
                            let sv = x * r.rescore_query_weight;
                            let sec = Explanation::new(sv, "product of:", vec![re, Explanation::leaf(r.rescore_query_weight, "secondaryWeight")]);
                            Explanation::new(mode.combine(pv, sv), format!("{} of:", mode.label()), vec![prim, sec])
                        }
                        _ => prim,
                    };
                }
                h.explanation = Some(e);
            }
        }

        TopDocs {
            total_hits,
            total_hits_relation,
            max_score,
            hits,
            terminated_early,
            timed_out: shared.timed_out.load(AtomicOrdering::Relaxed) || ctx.vector_timed_out.load(AtomicOrdering::Relaxed),
            invalid_score: ctx.invalid_score.lock().take(),
            execution_error: ctx.execution_error.lock().clone(),
        }
    }

    /// Explain how `doc` on leaf `leaf` scores for `query` under `opts`'
    /// similarity configuration.
    pub fn explain(&self, query: &Query, opts: &SearchOptions, leaf: usize, doc: DocId) -> Explanation {
        let ctx = self.context(query, opts);
        query.explain(&self.leaves[leaf], &ctx, doc)
    }

    /// Number of live docs matching `query`.
    pub fn count(&self, query: &Query, params: Bm25Params) -> u64 {
        if matches!(query, Query::MatchAll) {
            return self.leaves.iter().map(match_all_count).sum();
        }
        let ctx = SearchContext::for_query(query, &self.leaves, params);
        self.count_in(query, &ctx)
    }

    /// [`Searcher::count`] of `query` in its context `ctx` (built for these
    /// leaves): a k-NN query's search, done when the context was built, is
    /// not redone.
    pub fn count_in(&self, query: &Query, ctx: &SearchContext) -> u64 {
        if matches!(query, Query::MatchAll) {
            return self.leaves.iter().map(match_all_count).sum();
        }
        let leaf_count = |leaf: &LeafReader<'_>| match query.scorer(leaf, ctx) {
            Some(mut s) => {
                let del = leaf.deleted_bits();
                match s.collect_top_k_bulk(0, del.as_ref().map(|b| b.words())) {
                    Some((_, n)) => n,
                    None => count(&mut *s, leaf.live_docs()),
                }
            }
            None => 0,
        };
        if self.leaves.len() <= 1 || self.estimated_postings(query, ctx).is_some_and(|c| c < Self::PARALLEL_MIN_POSTINGS) || rayon::current_thread_index().is_some() {
            self.leaves.iter().map(leaf_count).sum()
        } else {
            in_search_pool(|| self.leaves.par_iter().map(leaf_count).sum())
        }
    }
}

/// The docs a `match_all` counts in `leaf`: every doc but the deleted and
/// the nested children ([`LeafReader::deleted_bits`], what its scorer
/// skips), from the bits instead of walking every doc.
fn match_all_count(leaf: &LeafReader<'_>) -> u64 {
    let n = leaf.num_docs();
    let Some(bits) = leaf.deleted_bits() else { return n as u64 };
    let words = bits.words();
    let full = (n / 64) as usize;
    let mut hidden: u64 = words.iter().take(full).map(|w| w.count_ones() as u64).sum();
    let rest = n % 64;
    if rest > 0 {
        if let Some(w) = words.get(full) {
            hidden += (w & ((1u64 << rest) - 1)).count_ones() as u64;
        }
    }
    (n as u64).saturating_sub(hidden)
}

#[cfg(test)]
mod tests {
    #[test]
    fn nested_search_pool_work_does_not_enter_indexing_pool() {
        assert!(!super::on_search_pool_worker());
        let indexing = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        indexing.install(|| {
            assert!(rayon::current_thread_index().is_some());
            assert!(!super::on_search_pool_worker());
        });
        super::in_search_pool(|| {
            assert!(super::on_search_pool_worker());
            use rayon::prelude::*;
            super::in_search_pool(|| (0..64).into_par_iter().for_each(|_| {
                assert!(super::on_search_pool_worker());
            }));
        });
        assert!(!super::on_search_pool_worker());
    }

    #[test]
    fn long_sort_rounding_matches_java_ties_and_extremes() {
        for (input, expected) in [(1.5, 2), (-1.5, -1), (-0.5, 0), (0.5, 1),
            (4503599627370497.0, 4503599627370497), (f64::INFINITY, i64::MAX),
            (f64::NEG_INFINITY, i64::MIN), (f64::NAN, 0)] {
            assert_eq!(super::Searcher::round_long(input), expected);
        }
    }

    use super::*;
    use crate::codec::directory::RamDirectory;
    use crate::query::query::Bound;
    use crate::segment::SegmentWriter;
    use crate::types::{Document, Field, FieldType, Term};
    use std::sync::Arc;
    use std::time::Duration;

    struct Fixture {
        docs: Vec<(String, i64, String)>, // body, n, tag
        sealed: Vec<crate::segment::Segment>,
        mem: SegmentWriter,
    }

    fn fixture() -> Fixture {
        let schema = vec![
            Field::new(0, "body", FieldType::Text),
            Field::new(1, "n", FieldType::I64),
            Field::new(2, "tag", FieldType::Keyword),
            Field::new(3, "v", FieldType::Vector { dim: 4 }).with_similarity("l2_norm"),
        ];
        let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
        let mut docs = Vec::new();
        for i in 0..300u32 {
            let body = format!(
                "{} {} {}{}",
                words[(i % 6) as usize],
                words[((i / 6) % 6) as usize],
                if i % 5 == 0 { "rare " } else { "" },
                if i % 2 == 0 { "common even" } else { "common odd" }
            );
            let tag = format!("t{}", i % 4);
            docs.push((body, i as i64, tag));
        }
        let dir = RamDirectory::new();
        let mut sealed = Vec::new();
        let mut mem = SegmentWriter::new(9);
        for (chunk_idx, chunk) in docs.chunks(100).enumerate() {
            let mut w = SegmentWriter::new(chunk_idx as u32);
            for (j, (body, n, tag)) in chunk.iter().enumerate() {
                let mut d = Document::new();
                d.add_text(0, body);
                d.add_i64(1, *n);
                d.add_keyword(2, tag);
                let f = (chunk_idx * 100 + j) as f32;
                d.add_vector(3, vec![f, 1.0, 0.0, 0.0]);
                if chunk_idx < 2 {
                    w.add_document(&d, &schema);
                } else {
                    mem.add_document(&d, &schema);
                }
            }
            if chunk_idx < 2 {
                sealed.push(w.write_to(&dir).unwrap());
            }
        }
        // Delete global doc 7 (leaf 0 doc 7) and 250 (memtable doc 50).
        sealed[0].delete_doc(7);
        mem.delete_doc(50);
        Fixture { docs, sealed, mem }
    }

    /// `match_all` counts its live docs without visiting them for any
    /// `track_total_hits` (bounded included) and returns the first live docs:
    /// totals, relation and hits as the per-doc count gives them.
    #[test]
    fn match_all_precount_respects_bounded_totals() {
        let fixture = fixture();
        let live: Vec<(usize, u32)> = (0..300usize).filter(|&g| fixture.is_live(g)).map(|g| if g < 200 { (g / 100, (g % 100) as u32) } else { (2, (g - 200) as u32) }).collect();
        for tth in [TrackTotalHits::UpTo(5), TrackTotalHits::UpTo(297), TrackTotalHits::UpTo(298), TrackTotalHits::UpTo(1000), TrackTotalHits::True, TrackTotalHits::False] {
            for k in [0usize, 3, 150] {
                for q in [Query::MatchAll, Query::Boost { query: Box::new(Query::MatchAll), boost: 2.0 }] {
                    let opts = SearchOptions { top_k: k, track_total_hits: tth, ..Default::default() };
                    let top = fixture.searcher().search(&q, &opts);
                    let (total, rel) = match tth {
                        TrackTotalHits::UpTo(n) if 298 > n => (n, TotalHitsRelation::Gte),
                        TrackTotalHits::False => (0, top.total_hits_relation),
                        _ => (298, TotalHitsRelation::Eq),
                    };
                    assert_eq!((top.total_hits, top.total_hits_relation), (total, rel), "{tth:?} k={k}");
                    let got: Vec<(usize, u32)> = top.hits.iter().map(|h| (h.leaf, h.doc)).collect();
                    assert_eq!(got, live[..k.min(live.len())].to_vec(), "{tth:?} k={k}");
                }
            }
        }
    }

    /// Leaves whose `n` values all lie outside a required range are not
    /// visited (sealed segments only), with the same hits and totals as a
    /// search of every leaf: a range alone, in `must` beside a term, in a
    /// `filter`, nested in a required bool, and a range only in a `should`
    /// or `must_not` (which excludes nothing).
    #[test]
    fn required_ranges_skip_leaves_outside_their_values() {
        let f = fixture();
        let s = f.searcher();
        let range = |lo: i64, hi: i64| Query::RangeI64 { field: 1, lower: Bound::Included(lo), upper: Bound::Included(hi) };
        let even = Query::Term(Term::new_text(0, "even"));
        let bool_of = |must: Vec<Query>, should: Vec<Query>, must_not: Vec<Query>, filter: Vec<Query>| Query::Bool { must, should, must_not, filter, minimum_should_match: crate::query::boolean::MinShouldMatch::Count(0) };
        let cases: Vec<(Query, Vec<bool>, Box<dyn Fn(&(String, i64, String)) -> bool>)> = vec![
            (range(120, 150), vec![true, false, false], Box::new(|d| (120..=150).contains(&d.1))),
            (range(500, 600), vec![true, true, false], Box::new(|_| false)),
            (bool_of(vec![range(0, 40), even.clone()], vec![], vec![], vec![]), vec![false, true, false], Box::new(|d| d.1 <= 40 && d.0.contains("even"))),
            (bool_of(vec![even.clone()], vec![], vec![], vec![range(210, 260)]), vec![true, true, false], Box::new(|d| (210..=260).contains(&d.1) && d.0.contains("even"))),
            (bool_of(vec![bool_of(vec![], vec![], vec![], vec![Query::Boost { query: Box::new(range(100, 199)), boost: 2.0 }])], vec![], vec![], vec![]), vec![true, false, false], Box::new(|d| (100..=199).contains(&d.1))),
            (bool_of(vec![], vec![range(0, 10), even.clone()], vec![], vec![]), vec![], Box::new(|d| d.1 <= 10 || d.0.contains("even"))),
            (bool_of(vec![even.clone()], vec![], vec![range(0, 199)], vec![]), vec![], Box::new(|d| d.1 > 199 && d.0.contains("even"))),
        ];
        // No required range: every leaf stays live (and may run in parallel).
        assert_eq!(Searcher::live_leaf_count(&s.range_excluded_leaves(&even), 3), 3);
        assert_eq!(Searcher::live_leaf_count(&[true, false, true], 3), 1);
        // A range costs only the docs of leaves it can match (the memtable's
        // 100, never excluded, and the 100 of the sealed leaf it overlaps).
        let ctx = SearchContext::default();
        assert_eq!(s.estimated_postings(&range(500, 600), &ctx), Some(100));
        assert_eq!(s.estimated_postings(&range(150, 160), &ctx), Some(200));
        for (q, excluded, pred) in &cases {
            assert_eq!(&s.range_excluded_leaves(q), excluded, "{q:?}");
            let want = f.brute(|d| pred(d));
            for tth in [TrackTotalHits::True, TrackTotalHits::UpTo(5)] {
                for k in [0usize, 3, 1000] {
                    let opts = SearchOptions { top_k: k, track_total_hits: tth, ..Default::default() };
                    let td = s.search(q, &opts);
                    match tth {
                        TrackTotalHits::True => assert_eq!(td.total_hits, want.len() as u64, "{q:?} k={k}"),
                        _ => assert_eq!(td.total_hits, (want.len() as u64).min(5), "{q:?} k={k}"),
                    }
                    if k == 1000 {
                        assert_eq!(Fixture::hits_set(&td), want, "{q:?}");
                    } else {
                        assert_eq!(td.hits.len(), k.min(want.len()), "{q:?} k={k}");
                    }
                }
            }
        }
    }

    #[test]
    fn search_and_count_progress_when_write_workers_wait_on_readers() {
        let _test_gate = crate::global_rayon_test_gate();
        let fixture = fixture();
        let query = Query::prefix(0, "alpha"); // unknown cost: exercises parallel leaves
        let expected = fixture.searcher().count(&query, Bm25Params::default());
        let gate = Arc::new(std::sync::Mutex::new(()));
        let reader = gate.lock().unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let workers = rayon::current_num_threads();
        for _ in 0..workers {
            let gate = gate.clone();
            let started = started_tx.clone();
            rayon::spawn(move || {
                let _ = started.send(());
                let _writer = gate.lock().unwrap_or_else(|poison| poison.into_inner());
            });
        }
        for _ in 0..workers {
            if let Err(error) = started_rx.recv_timeout(Duration::from_secs(10)) {
                drop(reader);
                panic!("global Rayon blockers did not start: {error}");
            }
        }
        std::thread::scope(|scope| {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            scope.spawn(move || {
                let searcher = fixture.searcher();
                let count = searcher.count(&query, Bm25Params::default());
                let hits = searcher.search(&query, &SearchOptions::default());
                done_tx.send((count, hits.total_hits)).unwrap();
            });
            let result = done_rx.recv_timeout(Duration::from_secs(3));
            // Always release blocked writers before joining, including on failure.
            drop(reader);
            assert_eq!(result.unwrap(), (expected, expected));
        });
    }

    impl Fixture {
        fn searcher(&self) -> Searcher<'_> {
            let mut leaves: Vec<LeafReader<'_>> = self.sealed.iter().map(LeafReader::Segment).collect();
            leaves.push(LeafReader::Memtable(&self.mem, None));
            Searcher::new(leaves)
        }
        fn is_live(&self, g: usize) -> bool {
            g != 7 && g != 250
        }
        fn global(h: &TopHit) -> usize {
            h.leaf * 100 + h.doc as usize
        }
        fn brute(&self, pred: impl Fn(&(String, i64, String)) -> bool) -> Vec<usize> {
            self.docs.iter().enumerate().filter(|(g, d)| self.is_live(*g) && pred(d)).map(|(g, _)| g).collect()
        }
        fn hits_set(td: &TopDocs) -> Vec<usize> {
            let mut v: Vec<usize> = td.hits.iter().map(Self::global).collect();
            v.sort_unstable();
            v
        }
    }

    fn all(opts_k: usize) -> SearchOptions {
        SearchOptions { top_k: opts_k, ..Default::default() }
    }

    #[test]
    fn term_and_bool_vs_brute_force() {
        let f = fixture();
        let s = f.searcher();
        let q = Query::term(0, "alpha");
        let td = s.search(&q, &all(1000));
        let want = f.brute(|d| d.0.split(' ').any(|w| w == "alpha"));
        assert_eq!(Fixture::hits_set(&td), want);
        assert_eq!(td.total_hits, want.len() as u64);
        assert_eq!(s.count(&q, Bm25Params::default()), want.len() as u64);

        // must + must_not + filter + should(min 1)
        let q = Query::Bool {
            must: vec![Query::term(0, "common")],
            must_not: vec![Query::term(0, "rare")],
            filter: vec![Query::TermSet { field: 2, terms: vec![b"t1".to_vec(), b"t2".to_vec()], boost: 1.0 }],
            should: vec![Query::term(0, "alpha"), Query::term(0, "beta")],
            minimum_should_match: 1.into(),
        };
        let td = s.search(&q, &all(1000));
        let want = f.brute(|d| {
            let ws: Vec<&str> = d.0.split(' ').collect();
            ws.contains(&"common") && !ws.contains(&"rare") && (d.2 == "t1" || d.2 == "t2") && (ws.contains(&"alpha") || ws.contains(&"beta"))
        });
        assert_eq!(Fixture::hits_set(&td), want);

        // Rarer term scores higher than common one.
        let td = s.search(&Query::Bool { must: vec![], should: vec![Query::term(0, "rare"), Query::term(0, "common")], must_not: vec![], filter: vec![], minimum_should_match: 0.into() }, &all(5));
        assert!(td.hits.iter().all(|h| f.docs[Fixture::global(h)].0.contains("rare")));
    }

    #[test]
    fn small_k_with_cross_leaf_floor_matches_full_ranking() {
        // Later leaves skip with the k-th best score of earlier ones; the
        // page must equal the head of the full ranking (ties: leaf, doc).
        let f = fixture();
        let s = f.searcher();
        let queries = vec![
            Query::term(0, "common"),
            Query::term(0, "alpha"),
            Query::bool_query().must(Query::term(0, "common")).must(Query::term(0, "even")).build(),
            Query::bool_query().should(Query::term(0, "rare")).should(Query::term(0, "beta")).build(),
            Query::bool_query().should(Query::term(0, "rare")).should(Query::term(0, "beta")).should(Query::term(0, "odd")).build(),
            Query::term(0, "rare").boost(3.0),
        ];
        for q in &queries {
            let full = s.search(q, &all(1000));
            for k in [1usize, 3, 7, 20] {
                for parallel in [false, true] {
                    let opts = SearchOptions { top_k: k, parallel, ..Default::default() };
                    let td = s.search(q, &opts);
                    let got: Vec<(usize, DocId)> = td.hits.iter().map(|h| (h.leaf, h.doc)).collect();
                    let want: Vec<(usize, DocId)> = full.hits.iter().take(k).map(|h| (h.leaf, h.doc)).collect();
                    assert_eq!(got, want, "{q:?} k={k} parallel={parallel}");
                    assert_eq!(td.total_hits, full.total_hits);
                    assert!(td.hits.iter().all(|h| h.sort == vec![SortValue::F64(h.score as f64)]));
                }
            }
        }
    }

    #[test]
    fn multi_term_queries_vs_brute_force() {
        let f = fixture();
        let s = f.searcher();
        let cases: Vec<(Query, Box<dyn Fn(&str) -> bool>)> = vec![
            (Query::prefix(0, "ep").with_max_expansions(50), Box::new(|w| w.starts_with("ep"))),
            (Query::wildcard(0, "*ta").with_max_expansions(50), Box::new(|w| w.ends_with("ta"))),
            (Query::regexp(0, "[a-d]l.*").with_max_expansions(50), Box::new(|w| w.len() > 2 && "abcd".contains(&w[..1]) && &w[1..2] == "l")),
            (Query::Fuzzy { field: 0, text: "alpho".into(), fuzziness: crate::query::multi_term::Fuzziness::Edits(1), prefix_length: 0, transpositions: true, max_expansions: 50, rewrite: crate::query::multi_term::MultiTermRewrite::TopTermsBlendedFreqs(50) }, Box::new(|w| w == "alpha")),
            (Query::term_range(0, Bound::Included(b"beta".to_vec()), Bound::Excluded(b"delta".to_vec())), Box::new(|w| w >= "beta" && w < "delta")),
            (Query::Exists { field: 0 }, Box::new(|_| true)),
        ];
        for (q, pred) in cases {
            let td = s.search(&q, &all(1000));
            let want = f.brute(|d| d.0.split(' ').any(|w| pred(w)));
            assert_eq!(Fixture::hits_set(&td), want, "{q:?}");
        }
        let q = Query::Phrase { terms: vec![Term::new_text(0, "common"), Term::new_text(0, "odd")], slop: 0 };
        let td = s.search(&q, &all(1000));
        assert_eq!(Fixture::hits_set(&td), f.brute(|d| d.0.contains("common odd")));
    }

    #[test]
    fn numeric_ranges_and_sorting() {
        let f = fixture();
        let s = f.searcher();
        let q = Query::RangeI64 { field: 1, lower: Bound::Excluded(90), upper: Bound::Included(210) };
        let td = s.search(&q, &all(1000));
        assert_eq!(Fixture::hits_set(&td), f.brute(|d| d.1 > 90 && d.1 <= 210));

        // Sort by n desc, page with search_after across leaves.
        let opts = SearchOptions {
            top_k: 7,
            sort: vec![SortField::field(1, SortKind::I64).desc()],
            ..Default::default()
        };
        let p1 = s.search(&Query::MatchAll, &opts);
        let ns: Vec<i64> = p1.hits.iter().map(|h| f.docs[Fixture::global(h)].1).collect();
        assert_eq!(ns, vec![299, 298, 297, 296, 295, 294, 293]);
        let mut seen = ns.clone();
        let mut after = p1.hits.last().cloned();
        for _ in 0..50 {
            let opts2 = SearchOptions { search_after: after.clone(), ..opts.clone() };
            let p = s.search(&Query::MatchAll, &opts2);
            if p.hits.is_empty() {
                break;
            }
            seen.extend(p.hits.iter().map(|h| f.docs[Fixture::global(h)].1));
            after = p.hits.last().cloned();
        }
        let mut want: Vec<i64> = (0..300).filter(|n| *n != 7 && *n != 250).collect();
        want.sort_by(|a, b| b.cmp(a));
        assert_eq!(seen, want, "paging must visit every live doc exactly once in order");

        // Keyword sort asc then n desc.
        let opts = SearchOptions {
            top_k: 3,
            sort: vec![
                SortField::field(2, SortKind::Keyword),
                SortField::field(1, SortKind::I64).desc(),
            ],
            ..Default::default()
        };
        let td = s.search(&Query::MatchAll, &opts);
        let got: Vec<(String, i64)> = td.hits.iter().map(|h| (f.docs[Fixture::global(h)].2.clone(), f.docs[Fixture::global(h)].1)).collect();
        assert_eq!(got, vec![("t0".into(), 296), ("t0".into(), 292), ("t0".into(), 288)]);
    }

    #[test]
    fn dismax_boost_const_knn() {
        let f = fixture();
        let s = f.searcher();
        let a = s.search(&Query::term(0, "rare"), &all(1)).hits[0].score;
        let b = s.search(&Query::term(0, "rare").boost(2.0), &all(1)).hits[0].score;
        assert!((b - 2.0 * a).abs() < 1e-5);
        let c = s.search(&Query::ConstScore { query: Box::new(Query::term(0, "rare")), score: 0.5, point_set: false }, &all(1)).hits[0].score;
        assert_eq!(c, 0.5);
        let dm = s.search(
            &Query::DisMax { queries: vec![Query::term(0, "rare"), Query::ConstScore { query: Box::new(Query::term(0, "rare")), score: 0.1, point_set: false }], tie_breaker: 0.0 },
            &all(1),
        );
        assert!((dm.hits[0].score - a).abs() < 1e-5);
        let dm = s.search(
            &Query::DisMax { queries: vec![Query::term(0, "rare"), Query::ConstScore { query: Box::new(Query::term(0, "rare")), score: 0.1, point_set: false }], tie_breaker: 1.0 },
            &all(1),
        );
        assert!((dm.hits[0].score - (a + 0.1)).abs() < 1e-5);

        // knn: nearest to [123,1,0,0] is global doc 123; filter to odd docs → 123 still; filter to evens → 122/124.
        let knn = Query::Knn { field: 3, vector: vec![123.0, 1.0, 0.0, 0.0], k: 3, num_candidates: 50, filter: None, similarity_threshold: None, boost: 1.0, oversample: None, visit_percentage: None };
        let td = s.search(&knn, &all(3));
        assert_eq!(Fixture::global(&td.hits[0]), 123);
        let knn = Query::Knn {
            field: 3,
            vector: vec![123.0, 1.0, 0.0, 0.0],
            k: 2,
            num_candidates: 50,
            filter: Some(Box::new(Query::term(0, "even"))),
            similarity_threshold: None,
            boost: 1.0,
            oversample: None,
            visit_percentage: None,
        };
        let td = s.search(&knn, &all(2));
        let g: Vec<usize> = td.hits.iter().map(Fixture::global).collect();
        assert!(g.iter().all(|x| x % 2 == 0));
        assert!(g.contains(&122) || g.contains(&124));
        // Deleted doc never returned, even from knn on the memtable (250).
        let knn = Query::Knn { field: 3, vector: vec![250.0, 1.0, 0.0, 0.0], k: 1, num_candidates: 20, filter: None, similarity_threshold: None, boost: 1.0, oversample: None, visit_percentage: None };
        let td = s.search(&knn, &all(1));
        assert_ne!(Fixture::global(&td.hits[0]), 250);
        assert!(s.search(&Query::MatchNone, &all(5)).hits.is_empty());
        assert_eq!(s.search(&Query::MatchAll, &all(5)).total_hits, 298);
    }

    // ---------------------------------------------------------------
    // Similarities, rewrites, phrases, spans, function_score, options.
    // ---------------------------------------------------------------

    use crate::query::boolean::MinShouldMatch;
    use crate::query::function_score::{BoostMode, DecayKind, FieldValueModifier, ScoreFunction, ScoreMode};
    use crate::query::multi_term::{Fuzziness, MultiTermRewrite};
    use crate::query::reader::{field_stats, term_stats};
    use crate::query::similarity::{BooleanSim, Classic, LmDirichlet, Similarity};
    use crate::query::spans::{IntervalsRule, SpanQuery};
    use std::collections::HashMap;

    fn words(d: &(String, i64, String)) -> Vec<&str> {
        d.0.split(' ').collect()
    }

    #[test]
    fn similarity_is_threaded_through_search() {
        let f = fixture();
        let s = f.searcher();
        let q = Query::term(0, "rare");
        let bm25 = s.search(&q, &all(1000));
        let opts = SearchOptions { top_k: 1000, similarity: Arc::new(BooleanSim), ..Default::default() };
        let boolean = s.search(&q, &opts);
        assert_eq!(Fixture::hits_set(&bm25), Fixture::hits_set(&boolean));
        assert!(boolean.hits.iter().all(|h| h.score == 1.0));

        // Per-field override beats the default; check exact Classic scores.
        let mut fs = HashMap::new();
        fs.insert(0u16, Arc::new(Classic) as Arc<dyn Similarity>);
        let opts = SearchOptions { top_k: 1000, similarity: Arc::new(BooleanSim), field_similarity: fs, ..Default::default() };
        let classic = s.search(&q, &opts);
        let ts = term_stats(s.leaves(), &Term::new_text(0, "rare"));
        let fst = field_stats(s.leaves(), 0);
        let w = Classic.weight(ts, fst, 1.0);
        for h in &classic.hits {
            let len = words(&f.docs[Fixture::global(h)]).len() as u32;
            assert!((h.score - w.score(1, len)).abs() < 1e-4, "{} vs {}", h.score, w.score(1, len));
        }
        // Phrases and spans score through the similarity too.
        let lm = SearchOptions { top_k: 1000, similarity: Arc::new(LmDirichlet::default()), ..Default::default() };
        let p = s.search(&Query::phrase(0, &["common", "odd"]), &lm);
        assert_eq!(Fixture::hits_set(&p), f.brute(|d| d.0.contains("common odd")));
        assert!(p.hits.iter().all(|h| h.score >= 0.0));
        // Legacy params still apply when the similarity is the default BM25.
        let legacy = SearchOptions { top_k: 1, params: Bm25Params { k1: 2.0, b: 0.0 }, ..Default::default() };
        assert_ne!(s.search(&q, &legacy).hits[0].score, bm25.hits[0].score);
    }

    #[test]
    fn boolean_rewrite_limits_expand_deduplicate_and_preserve_constant_score() {
        let fixture = fixture();
        let searcher = fixture.searcher();
        let context = |query: &Query, max| SearchContext::with_similarity_deadline_limit(
            query, searcher.leaves(), Arc::new(crate::query::similarity::Bm25::default()), HashMap::new(), None, max);
        for rewrite in [MultiTermRewrite::ScoringBoolean, MultiTermRewrite::ConstantScoreBoolean] {
            let query = Query::wildcard(0, "*ta").with_rewrite(rewrite);
            assert!(matches!(context(&query, 1).execution_result(), Err(crate::RusticError::TooManyClauses { max:1,nested:false })));
        }
        let constant = Query::wildcard(0, "*ta");
        let ctx = context(&constant, 1);
        ctx.execution_result().unwrap();
        assert!(searcher.search_with_context(&constant, &all(1000), &ctx).total_hits > 0);
        // The same dictionary term appears in multiple leaves, but is one clause.
        let duplicate = Query::prefix(0, "rare").with_rewrite(MultiTermRewrite::ScoringBoolean);
        context(&duplicate, 1).execution_result().unwrap();
        // Lucene top-terms rewrites clamp their selected term count to the cap.
        let top = Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::TopTermsScoringBoolean(100));
        context(&top, 1).execution_result().unwrap();
        let nested = Query::bool_query().should(top.clone()).should(top).build();
        assert!(matches!(context(&nested, 1).execution_result(), Err(crate::RusticError::TooManyClauses { max:1,nested:true })));
        // An explicit per-request snapshot is independent of other contexts.
        context(&nested, 10).execution_result().unwrap();
        // Rescore uses the same immutable cap and returns its error to the main context.
        let main = Query::MatchAll;
        let ctx = context(&main, 1);
        let options = SearchOptions { rescore:Some(Rescore { window_size:10,
            query:Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::ScoringBoolean),
            query_weight:1.0, rescore_query_weight:1.0 }), ..all(10) };
        let result = searcher.search_with_context(&main, &options, &ctx);
        assert!(result.execution_error.is_some());
        assert!(matches!(ctx.execution_result(), Err(crate::RusticError::TooManyClauses { max:1,nested:false })));
    }

    #[test]
    fn span_multi_rewrite_limit_is_global_and_does_not_truncate() {
        let fixture = fixture();
        let searcher = fixture.searcher();
        let context = |query: &Query, max| SearchContext::with_similarity_deadline_limit(
            query, searcher.leaves(), Arc::new(crate::query::similarity::Bm25::default()), HashMap::new(), None, max);
        let query = Query::Span(crate::query::spans::SpanQuery::Wildcard { field:0, pattern:"*ta".into() });
        assert!(matches!(context(&query, 1).execution_result(), Err(crate::RusticError::TooManyClauses { max:1,nested:false })));
        let ctx = context(&query, 10);
        ctx.execution_result().unwrap();
        let span = searcher.search_with_context(&query, &all(1000), &ctx);
        let plain = searcher.search(&Query::wildcard(0, "*ta"), &all(1000));
        assert_eq!(Fixture::hits_set(&span), Fixture::hits_set(&plain));
        let duplicate = Query::Span(crate::query::spans::SpanQuery::Prefix { field:0, prefix:b"rare".to_vec() });
        context(&duplicate, 1).execution_result().unwrap();
    }

    #[test]
    fn interval_expansion_limit_is_per_leaf_and_reports_overflow() {
        use crate::query::spans::{IntervalsRule,SpanQuery};
        let schema = vec![Field::new(0,"body",FieldType::Text)];
        let mut first = SegmentWriter::new(1);
        let mut second = SegmentWriter::new(2);
        for (writer,text) in [(&mut first,"aa"),(&mut second,"ab")] {
            let mut doc = Document::new();
            doc.add_text(0,text);
            writer.add_document(&doc,&schema);
        }
        let query = Query::Intervals { field:0,rule:IntervalsRule::Prefix("a".into()) };
        {
            let leaves = vec![LeafReader::Memtable(&first,None),LeafReader::Memtable(&second,None)];
            let context = |query:&Query| SearchContext::with_similarity_deadline_limit(
                query,&leaves,Arc::new(crate::query::similarity::Bm25::default()),HashMap::new(),None,1);
            let ctx = context(&query);
            ctx.execution_result().unwrap();
            assert_eq!(Searcher::new(leaves.clone()).search_with_context(&query,&all(10),&ctx).total_hits,2);
            let span = Query::Span(SpanQuery::Prefix { field:0,prefix:b"a".to_vec() });
            assert!(matches!(context(&span).execution_result(),Err(crate::RusticError::TooManyClauses { .. })));
        }
        let mut doc = Document::new();
        doc.add_text(0,"ab");
        first.add_document(&doc,&schema);
        let leaves = vec![LeafReader::Memtable(&first,None)];
        let ctx = SearchContext::with_similarity_deadline_limit(&query,&leaves,
            Arc::new(crate::query::similarity::Bm25::default()),HashMap::new(),None,1);
        assert!(matches!(ctx.execution_result(),Err(crate::RusticError::IntervalExpansion(reason))
            if reason == "Automaton [a] expanded to too many terms (limit 1)"));
        assert!(Searcher::new(leaves).search_with_context(&query,&all(10),&ctx).execution_error.is_some());
    }

    #[test]
    fn multi_term_rewrites_agree_on_docs() {
        let f = fixture();
        let s = f.searcher();
        let rewrites = [
            MultiTermRewrite::ConstantScore,
            MultiTermRewrite::ScoringBoolean,
            MultiTermRewrite::ConstantScoreBoolean,
            MultiTermRewrite::TopTermsScoringBoolean(100),
            MultiTermRewrite::TopTermsBlendedFreqs(100),
            MultiTermRewrite::TopTermsBoost(100),
        ];
        let base: Vec<Query> = vec![
            Query::prefix(0, "ep"),
            Query::wildcard(0, "*ta"),
            Query::regexp(0, "[a-d]l.*"),
            Query::Fuzzy { field: 0, text: "alpho".into(), fuzziness: Fuzziness::Edits(1), prefix_length: 0, transpositions: true, max_expansions: 50, rewrite: MultiTermRewrite::ConstantScore },
            Query::term_range(0, Bound::Included(b"beta".to_vec()), Bound::Included(b"delta".to_vec())),
            Query::prefix(0, "EP").case_insensitive(),
        ];
        for q in &base {
            let want = Fixture::hits_set(&s.search(q, &all(1000)));
            assert!(!want.is_empty(), "{q:?}");
            for r in rewrites {
                let qr = q.clone().with_rewrite(r);
                let td = s.search(&qr, &all(1000));
                assert_eq!(Fixture::hits_set(&td), want, "{qr:?}");
                if !r.is_scoring() {
                    assert!(td.hits.iter().all(|h| h.score == 1.0));
                }
            }
        }
        // ScoringBoolean == a should-bool of the expanded terms, exactly.
        let wq = Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::ScoringBoolean);
        let bq = Query::bool_query().should(Query::term(0, "beta")).should(Query::term(0, "delta")).should(Query::term(0, "zeta")).build();
        let (a, b) = (s.search(&wq, &all(1000)), s.search(&bq, &all(1000)));
        let sb: HashMap<(usize, DocId), f32> = b.hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
        for h in &a.hits {
            assert!((h.score - sb[&(h.leaf, h.doc)]).abs() < 1e-5);
        }
        // TopTermsBoost: score = number of matched terms.
        let tb = s.search(&Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::TopTermsBoost(10)), &all(1000));
        for h in &tb.hits {
            let n = words(&f.docs[Fixture::global(h)]).iter().filter(|w| w.ends_with("ta")).collect::<std::collections::HashSet<_>>().len();
            assert_eq!(h.score, n as f32);
        }
        // TopTerms(1) keeps only the highest-df term (ties: smallest text);
        // doc freqs count deleted docs, as Lucene's do.
        let df = |t: &str| f.docs.iter().filter(|d| words(d).contains(&t)).count();
        let best = ["beta", "delta", "zeta"].into_iter().max_by(|a, b| df(a).cmp(&df(b)).then(b.cmp(a))).unwrap();
        let t1 = s.search(&Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::TopTermsScoringBoolean(1)), &all(1000));
        assert_eq!(Fixture::hits_set(&t1), f.brute(|d| words(d).contains(&best)));
        // Blended: every term scores with the same (max) df.
        let bl = s.search(&Query::wildcard(0, "*ta").with_rewrite(MultiTermRewrite::TopTermsBlendedFreqs(10)), &all(1000));
        assert_eq!(Fixture::hits_set(&bl), Fixture::hits_set(&a));
        // Fuzziness AUTO on a 5-char term allows 1 edit.
        let fz = s.search(&Query::fuzzy(0, "alpho"), &all(1000));
        assert_eq!(Fixture::hits_set(&fz), f.brute(|d| words(d).contains(&"alpha")));
    }

    #[test]
    fn multi_phrase_with_alternatives_vs_brute_force() {
        let f = fixture();
        let s = f.searcher();
        let q = Query::multi_phrase(0, &[&["alpha", "beta"], &["gamma", "delta"]], 0);
        let td = s.search(&q, &all(1000));
        let want = f.brute(|d| {
            let w = words(d);
            w.windows(2).any(|p| (p[0] == "alpha" || p[0] == "beta") && (p[1] == "gamma" || p[1] == "delta"))
        });
        assert!(!want.is_empty());
        assert_eq!(Fixture::hits_set(&td), want);
        // Single alternative per position == plain phrase, same scores.
        let mp = s.search(&Query::multi_phrase(0, &[&["common"], &["odd"]], 0), &all(1000));
        let ph = s.search(&Query::phrase(0, &["common", "odd"]), &all(1000));
        assert_eq!(Fixture::hits_set(&mp), Fixture::hits_set(&ph));
        assert!((mp.hits[0].score - ph.hits[0].score).abs() < 1e-5);
        // Three positions with a synonym in the middle.
        let q = Query::multi_phrase(0, &[&["rare"], &["common"], &["even", "odd"]], 0);
        assert_eq!(Fixture::hits_set(&s.search(&q, &all(1000))), f.brute(|d| d.0.contains("rare common")));
    }

    #[test]
    fn minimum_should_match_specs_in_search() {
        let f = fixture();
        let s = f.searcher();
        let should = || vec![Query::term(0, "alpha"), Query::term(0, "beta"), Query::term(0, "common"), Query::term(0, "rare")];
        for (spec, k) in [("75%", 3usize), ("-1", 3), ("2", 2), ("3<-25%", 3), ("1<-1", 3), ("100%", 4)] {
            let q = Query::Bool { must: vec![], should: should(), must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::from(spec) };
            let want = f.brute(|d| {
                let w = words(d);
                ["alpha", "beta", "common", "rare"].iter().filter(|t| w.contains(t)).count() >= k
            });
            assert_eq!(Fixture::hits_set(&s.search(&q, &all(1000))), want, "{spec}");
        }
        // With a must clause, shoulds only add score unless msm requires them.
        let q = Query::bool_query().must(Query::term(0, "even")).should_boosted(Query::term(0, "rare"), 3.0).build();
        let td = s.search(&q, &all(1000));
        assert_eq!(Fixture::hits_set(&td), f.brute(|d| words(d).contains(&"even")));
        assert!(f.docs[Fixture::global(&td.hits[0])].0.contains("rare"));
    }

    // --- spans -----------------------------------------------------------

    struct SpanFixture {
        docs: Vec<Vec<&'static str>>,
        sealed: crate::segment::Segment,
        mem: SegmentWriter,
    }

    fn span_fixture() -> SpanFixture {
        let schema = vec![Field::new(0, "body", FieldType::Text)];
        let vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut docs = Vec::new();
        for _ in 0..240 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let len = 3 + (x % 7) as usize;
            let mut w = Vec::new();
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.push(vocab[(x % 6) as usize]);
            }
            docs.push(w);
        }
        let dir = RamDirectory::new();
        let mut sw = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        for (i, w) in docs.iter().enumerate() {
            let mut d = Document::new();
            d.add_text(0, &w.join(" "));
            if i < 120 {
                sw.add_document(&d, &schema);
            } else {
                mem.add_document(&d, &schema);
            }
        }
        SpanFixture { docs, sealed: sw.write_to(&dir).unwrap(), mem }
    }

    impl SpanFixture {
        fn searcher(&self) -> Searcher<'_> {
            Searcher::new(vec![LeafReader::Segment(&self.sealed), LeafReader::Memtable(&self.mem, None)])
        }
        fn hits(&self, s: &Searcher<'_>, q: &Query) -> Vec<usize> {
            let mut v: Vec<usize> = s.search(q, &all(1000)).hits.iter().map(|h| h.leaf * 120 + h.doc as usize).collect();
            v.sort_unstable();
            v
        }
        fn brute(&self, pred: impl Fn(&[&str]) -> bool) -> Vec<usize> {
            (0..self.docs.len()).filter(|&i| pred(&self.docs[i])).collect()
        }
    }

    fn positions_of(w: &[&str], t: &str) -> Vec<i64> {
        w.iter().enumerate().filter(|(_, x)| **x == t).map(|(i, _)| i as i64).collect()
    }

    /// Exists p_0 < p_1 < ... with (p_last - p_0 - (n - 1)) <= slop.
    fn brute_ordered(w: &[&str], terms: &[&str], slop: i64) -> bool {
        fn rec(w: &[&str], terms: &[&str], i: usize, prev: i64, first: i64, slop: i64) -> bool {
            if i == terms.len() {
                return prev - first - (terms.len() as i64 - 1) <= slop;
            }
            positions_of(w, terms[i]).into_iter().filter(|&p| i == 0 || p > prev).any(|p| rec(w, terms, i + 1, p, if i == 0 { p } else { first }, slop))
        }
        rec(w, terms, 0, -1, 0, slop)
    }

    /// Exists one position per (distinct) term with max - min + 1 - n <= slop.
    fn brute_unordered(w: &[&str], terms: &[&str], slop: i64) -> bool {
        fn rec(w: &[&str], terms: &[&str], i: usize, lo: i64, hi: i64, slop: i64) -> bool {
            if i == terms.len() {
                return hi - lo + 1 - terms.len() as i64 <= slop;
            }
            positions_of(w, terms[i]).into_iter().any(|p| rec(w, terms, i + 1, lo.min(p), hi.max(p), slop))
        }
        rec(w, terms, 0, i64::MAX, i64::MIN, slop)
    }

    #[test]
    fn span_near_vs_brute_force_positions() {
        let f = span_fixture();
        let s = f.searcher();
        let sets: [&[&str]; 4] = [&["alpha", "beta"], &["gamma", "alpha"], &["alpha", "beta", "gamma"], &["zeta", "delta", "epsilon"]];
        for terms in sets {
            for slop in 0..5u32 {
                for in_order in [true, false] {
                    let q = Query::Span(SpanQuery::Near { clauses: terms.iter().map(|t| SpanQuery::term(0, t)).collect(), slop, in_order });
                    let want = f.brute(|w| if in_order { brute_ordered(w, terms, slop as i64) } else { brute_unordered(w, terms, slop as i64) });
                    assert_eq!(f.hits(&s, &q), want, "{terms:?} slop={slop} ordered={in_order}");
                    if slop == 2 {
                        assert!(!want.is_empty(), "{terms:?}");
                    }
                }
            }
        }
        // Nested: near(near(alpha beta, 0, ordered), gamma, 1, ordered).
        let inner = SpanQuery::Near { clauses: vec![SpanQuery::term(0, "alpha"), SpanQuery::term(0, "beta")], slop: 0, in_order: true };
        let q = Query::Span(SpanQuery::Near { clauses: vec![inner, SpanQuery::term(0, "gamma")], slop: 1, in_order: true });
        assert_eq!(f.hits(&s, &q), f.brute(|w| brute_ordered(w, &["alpha", "beta", "gamma"], 1) && {
            // the alpha-beta pair itself must be adjacent
            (0..w.len()).any(|p| w[p] == "alpha" && w.get(p + 1) == Some(&"beta") && (p + 2..=(p + 3).min(w.len() - 1)).any(|g| w.get(g) == Some(&"gamma")))
        }));
    }

    #[test]
    fn span_first_not_or_and_intervals() {
        let f = span_fixture();
        let s = f.searcher();
        let q = Query::Span(SpanQuery::First { inner: Box::new(SpanQuery::term(0, "delta")), end: 2 });
        assert_eq!(f.hits(&s, &q), f.brute(|w| w.iter().take(2).any(|x| *x == "delta")));
        let q = Query::Span(SpanQuery::Not {
            include: Box::new(SpanQuery::term(0, "alpha")),
            exclude: Box::new(SpanQuery::Near { clauses: vec![SpanQuery::term(0, "alpha"), SpanQuery::term(0, "beta")], slop: 0, in_order: true }),
            pre: 0,
            post: 0,
        });
        assert_eq!(f.hits(&s, &q), f.brute(|w| (0..w.len()).any(|p| w[p] == "alpha" && w.get(p + 1) != Some(&"beta"))));
        let q = Query::Span(SpanQuery::Or(vec![SpanQuery::term(0, "alpha"), SpanQuery::term(0, "zeta")]));
        assert_eq!(f.hits(&s, &q), f.brute(|w| w.contains(&"alpha") || w.contains(&"zeta")));
        let q = Query::Span(SpanQuery::Prefix { field: 0, prefix: b"ep".to_vec() });
        assert_eq!(f.hits(&s, &q), f.brute(|w| w.contains(&"epsilon")));

        // Intervals map onto spans.
        let q = Query::Intervals { field: 0, rule: IntervalsRule::Match { terms: vec!["alpha".into(), "beta".into()], max_gaps: Some(1), ordered: true } };
        assert_eq!(f.hits(&s, &q), f.brute(|w| brute_ordered(w, &["alpha", "beta"], 1)));
        let q = Query::Intervals {
            field: 0,
            rule: IntervalsRule::AllOf {
                rules: vec![IntervalsRule::Match { terms: vec!["gamma".into()], max_gaps: None, ordered: false }, IntervalsRule::AnyOf(vec![IntervalsRule::Prefix("alp".into()), IntervalsRule::Wildcard("z*a".into())])],
                max_gaps: Some(0),
                ordered: true,
            },
        };
        assert_eq!(f.hits(&s, &q), f.brute(|w| w.windows(2).any(|p| p[0] == "gamma" && (p[1] == "alpha" || p[1] == "zeta"))));
        // Unlimited gaps, unordered = all present.
        let q = Query::Intervals { field: 0, rule: IntervalsRule::Match { terms: vec!["beta".into(), "delta".into()], max_gaps: None, ordered: false } };
        assert_eq!(f.hits(&s, &q), f.brute(|w| w.contains(&"beta") && w.contains(&"delta")));
        // Span frequency drives the score: more matches, higher score at equal length.
        let td = s.search(&Query::Span(SpanQuery::term(0, "alpha")), &all(1000));
        assert!(td.hits.iter().all(|h| h.score > 0.0));
    }

    // --- function_score -----------------------------------------------------

    #[test]
    fn function_score_modes_on_known_values() {
        let f = fixture();
        let s = f.searcher();
        let base = Query::RangeI64 { field: 1, lower: Bound::Included(10), upper: Bound::Unbounded }; // score 1
        let fvf = ScoreFunction::FieldValueFactor { field: 1, factor: 1.0, modifier: FieldValueModifier::None, missing: None };
        let check = |q: Query, expect: &dyn Fn(f64) -> f64| {
            let td = s.search(&q, &all(1000));
            assert_eq!(td.hits.len(), 289, "{q:?}"); // 290 docs with n >= 10, minus deleted 250
            for h in &td.hits {
                let n = f.docs[Fixture::global(h)].1 as f64;
                assert!((h.score as f64 - expect(n)).abs() < 1e-3 * expect(n).abs().max(1.0), "n={n}: {} vs {}", h.score, expect(n));
            }
        };
        let fs = |functions: Vec<ScoreFunction>, sm: ScoreMode, bm: BoostMode| Query::FunctionScore {
            query: Box::new(base.clone()),
            functions,
            score_mode: sm,
            boost_mode: bm,
            max_boost: f32::MAX,
            min_score: None,
        };
        let two = || vec![ScoreFunction::Weight(2.0), fvf.clone()];
        check(fs(two(), ScoreMode::Multiply, BoostMode::Multiply), &|n| 2.0 * n);
        check(fs(two(), ScoreMode::Sum, BoostMode::Multiply), &|n| n + 2.0);
        // ES averages weighted: sum of values over sum of function weights (2 + 1).
        check(fs(two(), ScoreMode::Avg, BoostMode::Multiply), &|n| (n + 2.0) / 3.0);
        check(fs(two(), ScoreMode::First, BoostMode::Multiply), &|_| 2.0);
        check(fs(two(), ScoreMode::Max, BoostMode::Multiply), &|n| n.max(2.0));
        check(fs(two(), ScoreMode::Min, BoostMode::Multiply), &|n| n.min(2.0));
        check(fs(two(), ScoreMode::Sum, BoostMode::Sum), &|n| 1.0 + n + 2.0);
        check(fs(two(), ScoreMode::Sum, BoostMode::Replace), &|n| n + 2.0);
        check(fs(two(), ScoreMode::Sum, BoostMode::Avg), &|n| (1.0 + n + 2.0) / 2.0);
        check(fs(two(), ScoreMode::Sum, BoostMode::Min), &|_| 1.0);
        check(fs(two(), ScoreMode::Sum, BoostMode::Max), &|n| n + 2.0);
        let log = ScoreFunction::FieldValueFactor { field: 1, factor: 2.0, modifier: FieldValueModifier::Log1p, missing: None };
        check(fs(vec![log], ScoreMode::Multiply, BoostMode::Replace), &|n| (2.0 * n + 1.0).log10());
        let gauss = ScoreFunction::Decay { field: 1, origin: 150.0, scale: 20.0, offset: 5.0, decay: 0.5, kind: DecayKind::Gauss };
        check(fs(vec![gauss], ScoreMode::Multiply, BoostMode::Replace), &|n| {
            let d = ((n - 150.0).abs() - 5.0).max(0.0);
            let sigma2 = -400.0 / (2.0 * 0.5f64.ln());
            (-(d * d) / (2.0 * sigma2)).exp()
        });
        let filtered = ScoreFunction::Filtered { filter: Box::new(Query::term(0, "rare")), function: Box::new(ScoreFunction::Weight(10.0)) };
        let td = s.search(&fs(vec![filtered, ScoreFunction::Weighted { function: Box::new(ScoreFunction::Weight(1.0)), weight: 3.0 }], ScoreMode::Sum, BoostMode::Replace), &all(1000));
        for h in &td.hits {
            let rare = f.docs[Fixture::global(h)].0.contains("rare");
            assert_eq!(h.score, if rare { 13.0 } else { 3.0 });
        }
        // max_boost caps the function value; min_score drops docs.
        let mut q = fs(vec![fvf.clone()], ScoreMode::Multiply, BoostMode::Replace);
        if let Query::FunctionScore { max_boost, min_score, .. } = &mut q {
            *max_boost = 100.0;
            *min_score = Some(50.0);
        }
        let td = s.search(&q, &all(1000));
        assert_eq!(Fixture::hits_set(&td), f.brute(|d| d.1 >= 50));
        assert!(td.hits.iter().all(|h| h.score == 100.0 || (h.score as i64) == f.docs[Fixture::global(h)].1));
        assert!(td.hits.iter().all(|h| h.score <= 100.0));
        // Random scores are deterministic per seed and in [0, 1).
        let r = |seed| s.search(&fs(vec![ScoreFunction::RandomScore { seed }], ScoreMode::Multiply, BoostMode::Replace), &all(5));
        assert_eq!(r(7).hits, r(7).hits);
        assert!(r(7).hits.iter().all(|h| (0.0..1.0).contains(&h.score)));
    }

    // --- searcher options ---------------------------------------------------

    #[test]
    fn track_total_hits_relation_and_terminate_after() {
        let f = fixture();
        let s = f.searcher();
        let with = |t: TrackTotalHits, q: &Query| s.search(q, &SearchOptions { top_k: 5, track_total_hits: t, ..Default::default() });
        let td = with(TrackTotalHits::UpTo(50), &Query::MatchAll);
        assert_eq!((td.total_hits, td.total_hits_relation), (50, TotalHitsRelation::Gte));
        assert_eq!(td.hits.len(), 5);
        let full = with(TrackTotalHits::True, &Query::MatchAll);
        assert_eq!(td.hits, full.hits, "early termination must not change uniform-score hits");
        // An exact total over match_all (a scroll's first page) stops early
        // too: the count comes from the live docs.
        assert_eq!((full.total_hits, full.total_hits_relation), (298, TotalHitsRelation::Eq));
        let first = s.search(&Query::MatchAll, &SearchOptions { top_k: 298, track_total_hits: TrackTotalHits::True, ..Default::default() });
        assert_eq!(first.hits.iter().map(Fixture::global).take(5).collect::<Vec<_>>(), full.hits.iter().map(Fixture::global).collect::<Vec<_>>());
        assert_eq!((first.hits.len(), first.total_hits), (298, 298));
        let td = with(TrackTotalHits::UpTo(1000), &Query::MatchAll);
        assert_eq!((td.total_hits, td.total_hits_relation), (298, TotalHitsRelation::Eq));
        let td = with(TrackTotalHits::UpTo(298), &Query::MatchAll);
        assert_eq!((td.total_hits, td.total_hits_relation), (298, TotalHitsRelation::Eq));
        let td = with(TrackTotalHits::UpTo(297), &Query::MatchAll);
        assert_eq!((td.total_hits, td.total_hits_relation), (297, TotalHitsRelation::Gte));
        let td = with(TrackTotalHits::False, &Query::MatchAll);
        assert_eq!((td.total_hits, td.total_hits_relation), (0, TotalHitsRelation::Gte));
        assert_eq!(td.hits, full.hits);
        // Scored query: bound applies, hits unchanged.
        let q = Query::term(0, "common");
        let a = with(TrackTotalHits::UpTo(10), &q);
        let b = with(TrackTotalHits::True, &q);
        assert_eq!((a.total_hits, a.total_hits_relation), (10, TotalHitsRelation::Gte));
        assert_eq!(a.hits, b.hits);

        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 100, terminate_after: Some(5), ..Default::default() });
        assert!(td.terminated_early);
        assert_eq!(td.total_hits, 5);
        assert_eq!(td.hits.iter().map(Fixture::global).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 100, terminate_after: Some(150), ..Default::default() });
        assert_eq!(td.total_hits, 150);
        assert!(td.hits.iter().all(|h| Fixture::global(h) < 151));
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 1, terminate_after: Some(10_000), ..Default::default() });
        assert!(!td.terminated_early);
        assert_eq!(td.total_hits, 298);
        // Timeout of zero stops at the first check.
        let td = s.search(&Query::MatchAll, &SearchOptions { timeout: Some(Duration::ZERO), ..Default::default() });
        assert!(td.timed_out);
        assert!(!s.search(&Query::MatchAll, &SearchOptions { timeout: Some(Duration::from_secs(60)), ..Default::default() }).timed_out);
    }

    #[test]
    fn a_cancelled_search_task_stops_like_a_timeout() {
        let f = fixture();
        let s = f.searcher();
        let q = Query::term(0, "common");
        let full = s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::True, ..Default::default() });
        assert!(!full.timed_out && full.total_hits > 0);
        // A live token changes nothing.
        let live = crate::cancel::Token::new();
        let td = crate::cancel::with_search(Some(live), || s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::True, ..Default::default() }));
        assert_eq!((td.total_hits, td.hits.len(), td.timed_out), (full.total_hits, full.hits.len(), false));
        // A cancelled one visits no leaf.
        let gone = crate::cancel::Token::cancelled_with("by user request");
        let td = crate::cancel::with_search(Some(gone), || s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::True, ..Default::default() }));
        assert!(td.timed_out, "a cancelled search reports that it stopped");
        assert_eq!(td.total_hits, 0);
    }

    #[test]
    fn shard_wide_hit_counting_keeps_totals_and_hits() {
        // A uniformly scored query over several leaves: leaves stop counting
        // once the shard as a whole reached the limit; totals, relation and
        // hits match the exhaustive search.
        let f = fixture();
        let s = f.searcher();
        let q = Query::RangeI64 { field: 1, lower: Bound::Included(0), upper: Bound::Unbounded };
        for parallel in [false, true] {
            let with = |t: TrackTotalHits| s.search(&q, &SearchOptions { top_k: 5, track_total_hits: t, parallel, ..Default::default() });
            let full = with(TrackTotalHits::True);
            assert_eq!((full.total_hits, full.total_hits_relation), (298, TotalHitsRelation::Eq));
            for (limit, relation) in [(1, TotalHitsRelation::Gte), (150, TotalHitsRelation::Gte), (250, TotalHitsRelation::Gte), (297, TotalHitsRelation::Gte), (298, TotalHitsRelation::Eq), (1000, TotalHitsRelation::Eq)] {
                let td = with(TrackTotalHits::UpTo(limit));
                assert_eq!((td.total_hits, td.total_hits_relation), (limit.min(298), relation), "limit {limit}, parallel {parallel}");
                assert_eq!(td.hits, full.hits, "limit {limit}, parallel {parallel}");
            }
        }
    }

    #[test]
    fn from_size_paging_and_sorted_paging() {
        let f = fixture();
        let s = f.searcher();
        for (q, sort) in [
            (Query::term(0, "common"), vec![]),
            (Query::bool_query().should(Query::term(0, "rare")).should(Query::term(0, "alpha")).build(), vec![]),
            (Query::MatchAll, vec![SortField::field(2, SortKind::Keyword), SortField::field(1, SortKind::I64).desc()]),
        ] {
            let full = s.search(&q, &SearchOptions { top_k: 1000, sort: sort.clone(), ..Default::default() });
            let mut paged = Vec::new();
            for page in 0..30 {
                let p = s.search(&q, &SearchOptions { top_k: 7, from: page * 7, sort: sort.clone(), ..Default::default() });
                assert_eq!(p.total_hits, full.total_hits);
                paged.extend(p.hits);
            }
            assert_eq!(paged.len(), full.hits.len().min(210));
            assert_eq!(paged[..], full.hits[..paged.len()]);
        }
    }

    #[test]
    fn collapse_keeps_best_hit_per_value() {
        let f = fixture();
        let s = f.searcher();
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 10, sort: vec![SortField::field(1, SortKind::I64).desc()], collapse_field: Some((2, SortKind::Keyword)), ..Default::default() });
        let got: Vec<(String, i64)> = td.hits.iter().map(|h| (f.docs[Fixture::global(h)].2.clone(), f.docs[Fixture::global(h)].1)).collect();
        assert_eq!(got, vec![("t3".into(), 299), ("t2".into(), 298), ("t1".into(), 297), ("t0".into(), 296)]);
        assert_eq!(td.hits[0].collapse, Some(SortValue::Str("t3".into())));
        assert_eq!(td.total_hits, 298);
        // Score-sorted collapse: one hit per tag, each the best-scoring doc of its tag.
        let q = Query::bool_query().should(Query::term(0, "rare")).should(Query::term(0, "alpha")).build();
        let full = s.search(&q, &all(1000));
        let td = s.search(&q, &SearchOptions { top_k: 10, collapse_field: Some((2, SortKind::Keyword)), ..Default::default() });
        assert_eq!(td.hits.len(), 4);
        let mut seen = std::collections::HashSet::new();
        for h in &td.hits {
            let tag = &f.docs[Fixture::global(h)].2;
            assert!(seen.insert(tag.clone()));
            let best = full.hits.iter().find(|x| &f.docs[Fixture::global(x)].2 == tag).unwrap();
            assert_eq!((best.leaf, best.doc), (h.leaf, h.doc));
        }
        // Collapse on a numeric field (n is unique) changes nothing.
        let a = s.search(&q, &SearchOptions { top_k: 20, collapse_field: Some((1, SortKind::I64)), ..Default::default() });
        assert_eq!(a.hits.iter().map(|h| (h.leaf, h.doc)).collect::<Vec<_>>(), full.hits[..20].iter().map(|h| (h.leaf, h.doc)).collect::<Vec<_>>());
    }

    #[test]
    fn rescore_top_window() {
        let f = fixture();
        let s = f.searcher();
        let rare = |g: usize| f.docs[g].0.contains("rare");
        let rare_score: HashMap<usize, f32> = s.search(&Query::term(0, "rare"), &all(1000)).hits.iter().map(|h| (Fixture::global(h), h.score)).collect();
        let opts = SearchOptions {
            top_k: 20,
            rescore: Some(Rescore { window_size: 50, query: Query::term(0, "rare"), query_weight: 1.0, rescore_query_weight: 2.0 }),
            ..Default::default()
        };
        let td = s.search(&Query::MatchAll, &opts);
        // Window = the first 50 live docs (globals 0..=50 minus 7); rare ones rise to the top.
        let window_rare: Vec<usize> = (0..=50).filter(|&g| g != 7 && rare(g)).collect();
        let top: Vec<usize> = td.hits.iter().take(window_rare.len()).map(Fixture::global).collect();
        let mut sorted_top = top.clone();
        sorted_top.sort_unstable();
        assert_eq!(sorted_top, window_rare);
        for h in td.hits.iter().take(window_rare.len()) {
            assert!((h.score - (1.0 + 2.0 * rare_score[&Fixture::global(h)])).abs() < 1e-4);
        }
        // Non-rare window docs keep score 1 and follow.
        assert!(td.hits[window_rare.len()..].iter().all(|h| h.score == 1.0 && !rare(Fixture::global(h))));
        assert!((td.max_score - td.hits[0].score).abs() < 1e-6);
        // Explanation of a rescored hit has the original and rescore parts.
        let td = s.search(&Query::MatchAll, &SearchOptions { explain: true, ..opts });
        let e = td.hits[0].explanation.as_ref().unwrap();
        assert_eq!(e.details.len(), 2);
        assert!((e.value - td.hits[0].score).abs() < 1e-5);
    }

    #[test]
    fn explain_tree_shape() {
        let f = fixture();
        let s = f.searcher();
        let ex = |q: &Query| {
            let td = s.search(q, &SearchOptions { top_k: 5, explain: true, ..Default::default() });
            assert!(!td.hits.is_empty());
            for h in &td.hits {
                let e = h.explanation.as_ref().unwrap();
                assert!((e.value - h.score).abs() < 1e-4 * h.score.max(1.0), "{q:?}: {} vs {}\n{e}", e.value, h.score);
            }
            td.hits[0].explanation.clone().unwrap()
        };
        let e = ex(&Query::term(0, "rare"));
        assert!(e.description.contains("weight("));
        assert_eq!(e.details[0].details.len(), 3); // boost, idf, tf
        let e = ex(&Query::bool_query().must(Query::term(0, "common")).should(Query::term(0, "rare")).filter(Query::term(0, "even")).build());
        assert_eq!(e.description, "sum of:");
        assert_eq!(e.details.len(), 3); // must, should, filter
        assert_eq!(e.details[2].value, 0.0);
        // A boost is folded into the term weight (Lucene passes it down), not a product node.
        let base = ex(&Query::term(0, "rare")).value;
        let e = ex(&Query::term(0, "rare").boost(2.0));
        assert!((e.value - 2.0 * base).abs() < 1e-5);
        assert_eq!(e.details.len(), 1);
        let e = ex(&Query::ConstScore { query: Box::new(Query::term(0, "rare")), score: 0.5, point_set: false });
        assert_eq!(e.value, 0.5);
        let e = ex(&Query::phrase(0, &["common", "odd"]));
        assert!(format!("{e:?}").contains("phraseFreq=1.0"));
        let fsq = Query::function_score(Query::term(0, "rare"), vec![ScoreFunction::Weight(3.0)]);
        let e = ex(&fsq);
        assert_eq!(e.details.len(), 2);
        assert_eq!(e.details[1].value, 3.0);
        ex(&Query::DisMax { queries: vec![Query::term(0, "rare"), Query::term(0, "alpha")], tie_breaker: 0.3 });
        ex(&Query::Span(SpanQuery::Near { clauses: vec![SpanQuery::term(0, "common"), SpanQuery::term(0, "odd")], slop: 0, in_order: true }));
        // One-line explanation for other queries; non-match is a 0-valued node.
        let e = ex(&Query::RangeI64 { field: 1, lower: Bound::Included(5), upper: Bound::Unbounded });
        assert!(e.details.is_empty());
        let no = s.explain(&Query::term(0, "rare"), &SearchOptions::default(), 0, 1);
        assert_eq!(no.value, 0.0);
        assert!(no.description.starts_with("no match"));
        // Explain under another similarity reports that similarity.
        let e = s.explain(&Query::term(0, "rare"), &SearchOptions { similarity: Arc::new(Classic), ..Default::default() }, 0, 0);
        assert!(e.description.contains("Classic"));
    }

    #[test]
    fn knn_threshold_and_boost() {
        let f = fixture();
        let s = f.searcher();
        let base = s.search(&Query::knn(3, vec![123.0, 1.0, 0.0, 0.0], 5), &all(5));
        assert_eq!(Fixture::global(&base.hits[0]), 123);
        let boosted = s.search(&Query::Knn { field: 3, vector: vec![123.0, 1.0, 0.0, 0.0], k: 5, num_candidates: 50, filter: None, similarity_threshold: None, boost: 2.0, oversample: None, visit_percentage: None }, &all(5));
        assert!((boosted.hits[0].score - 2.0 * base.hits[0].score).abs() < 1e-5);
        let thr = base.hits[2].score;
        let cut = s.search(&Query::Knn { field: 3, vector: vec![123.0, 1.0, 0.0, 0.0], k: 5, num_candidates: 50, filter: None, similarity_threshold: Some(thr), boost: 1.0, oversample: None, visit_percentage: None }, &all(5));
        assert!(cut.hits.iter().all(|h| h.score >= thr));
        assert!(cut.hits.len() < base.hits.len());
    }

    #[test]
    fn knn_top_k_is_global_not_per_leaf() {
        let f = fixture();
        let s = f.searcher();
        assert_eq!(s.leaves().len(), 3);
        let target = [150.4f32, 1.0, 0.0, 0.0];
        let knn = Query::Knn { field: 3, vector: target.to_vec(), k: 5, num_candidates: 100, filter: None, similarity_threshold: None, boost: 1.0, oversample: None, visit_percentage: None };
        let td = s.search(&knn, &all(100));
        assert_eq!(td.hits.len(), 5);
        assert_eq!(td.total_hits, 5);
        // Brute force: vector of global doc g is [g, 1, 0, 0]; nearest live docs to 150.4.
        let mut brute: Vec<usize> = (0..300).filter(|&g| f.is_live(g)).collect();
        brute.sort_by(|a, b| (*a as f32 - 150.4).abs().partial_cmp(&(*b as f32 - 150.4).abs()).unwrap());
        let mut want = brute[..5].to_vec();
        want.sort_unstable();
        assert_eq!(Fixture::hits_set(&td), want);
        // Inside bool, knn contributes to at most k docs.
        let q = Query::bool_query().must(Query::term(0, "common")).should(knn.clone()).build();
        let td = s.search(&q, &all(1000));
        let plain = s.search(&Query::term(0, "common"), &all(1000));
        let base: HashMap<(usize, DocId), f32> = plain.hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
        let boosted: Vec<usize> = td.hits.iter().filter(|h| h.score > base[&(h.leaf, h.doc)] + 1e-6).map(Fixture::global).collect();
        assert!(boosted.len() <= 5);
        let mut boosted_sorted = boosted.clone();
        boosted_sorted.sort_unstable();
        assert_eq!(boosted_sorted, want);
        // With a filter and threshold the global k still holds.
        let filtered = Query::Knn { field: 3, vector: target.to_vec(), k: 3, num_candidates: 100, filter: Some(Box::new(Query::term(0, "odd"))), similarity_threshold: None, boost: 1.0, oversample: None, visit_percentage: None };
        let td = s.search(&filtered, &all(100));
        assert_eq!(td.hits.len(), 3);
        assert!(td.hits.iter().all(|h| Fixture::global(h) % 2 == 1));
    }

    // --- multi-valued sort, missing, exists -----------------------------------

    struct MvFixture {
        sealed: crate::segment::Segment,
        mem: SegmentWriter,
    }

    fn mv_vals(i: u32) -> Vec<i64> {
        // Unsorted on purpose so min/max/median differ from first/last.
        let all = [(i * 7 % 50) as i64 + 100, (i * 13 % 50) as i64, (i * 3 % 50) as i64 + 50];
        all[..(i % 4) as usize].to_vec()
    }

    fn mv_fixture() -> MvFixture {
        let schema = vec![Field::new(0, "body", FieldType::Text), Field::new(1, "n", FieldType::I64), Field::new(2, "tag", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let mut sw = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        for i in 0..40u32 {
            let mut d = Document::new();
            d.add_text(0, "doc");
            for v in mv_vals(i) {
                d.add_i64(1, v);
            }
            if i % 2 == 0 {
                d.add_keyword(2, &format!("k{}", i % 5));
            }
            if i < 20 {
                sw.add_document(&d, &schema);
            } else {
                mem.add_document(&d, &schema);
            }
        }
        MvFixture { sealed: sw.write_to(&dir).unwrap(), mem }
    }

    #[test]
    fn multi_valued_sort_modes_missing_and_exists() {
        let mv = mv_fixture();
        let s = Searcher::new(vec![LeafReader::Segment(&mv.sealed), LeafReader::Memtable(&mv.mem, None)]);
        let g = |h: &TopHit| h.leaf as u32 * 20 + h.doc;
        assert!((0..40).any(|i| mv_vals(i).len() >= 2) && (0..40).any(|i| mv_vals(i).is_empty()));

        let key = |i: u32, mode: SortMode| -> Option<f64> {
            let mut v: Vec<f64> = mv_vals(i).into_iter().map(|x| x as f64).collect();
            if v.is_empty() {
                return None;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            Some(match mode {
                SortMode::Min => v[0],
                SortMode::Max => v[n - 1],
                SortMode::Sum => v.iter().sum(),
                SortMode::Avg => v.iter().sum::<f64>() / n as f64,
                SortMode::Median => if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 },
            })
        };
        for mode in [SortMode::Min, SortMode::Max, SortMode::Sum, SortMode::Avg, SortMode::Median] {
            for desc in [false, true] {
                for missing in [SortMissing::Last, SortMissing::First] {
                    let mut sf = SortField::field(1, SortKind::I64).with_mode(mode).with_missing(missing.clone());
                    if desc {
                        sf = sf.desc();
                    }
                    let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 100, sort: vec![sf], ..Default::default() });
                    assert_eq!(td.hits.len(), 40);
                    let got: Vec<Option<f64>> = td.hits.iter().map(|h| key(g(h), mode)).collect();
                    let mut present: Vec<f64> = got.iter().flatten().copied().collect();
                    let n_missing = got.len() - present.len();
                    let mut want_present = present.clone();
                    want_present.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if desc {
                        want_present.reverse();
                    }
                    assert_eq!(present, want_present, "{mode:?} desc={desc}");
                    present.clear();
                    let missing_pos: Vec<usize> = got.iter().enumerate().filter(|(_, k)| k.is_none()).map(|(i, _)| i).collect();
                    let want_pos: Vec<usize> = if missing == SortMissing::First { (0..n_missing).collect() } else { (40 - n_missing..40).collect() };
                    assert_eq!(missing_pos, want_pos, "{mode:?} desc={desc} {missing:?}");
                }
            }
        }
        // Default mode: min for asc, max for desc.
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 1, sort: vec![SortField::field(1, SortKind::I64).desc()], ..Default::default() });
        let best = (0..40).filter_map(|i| key(i, SortMode::Max)).fold(f64::MIN, f64::max);
        assert_eq!(key(g(&td.hits[0]), SortMode::Max), Some(best));
        // Missing as a value: -1 sorts missing docs first ascending.
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 40, sort: vec![SortField::field(1, SortKind::I64).with_missing(SortMissing::Value(SortValue::I64(-1)))], ..Default::default() });
        let n_missing = (0..40).filter(|&i| mv_vals(i).is_empty()).count();
        assert!(td.hits[..n_missing].iter().all(|h| mv_vals(g(h)).is_empty()));
        assert_eq!(td.hits[0].sort[0], SortValue::I64(-1));
        // Unmapped sort field: everything missing, nothing fails.
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 40, sort: vec![SortField::field(77, SortKind::F64)], ..Default::default() });
        assert_eq!(td.hits.len(), 40);
        assert!(td.hits.iter().all(|h| h.sort[0] == SortValue::Missing));

        // Exists through fast-field presence (numeric multi + keyword) and postings.
        let ids = |td: &TopDocs| {
            let mut v: Vec<u32> = td.hits.iter().map(g).collect();
            v.sort_unstable();
            v
        };
        let td = s.search(&Query::Exists { field: 1 }, &all(100));
        assert_eq!(ids(&td), (0..40).filter(|&i| !mv_vals(i).is_empty()).collect::<Vec<_>>());
        let td = s.search(&Query::Exists { field: 2 }, &all(100));
        assert_eq!(ids(&td), (0..40).filter(|&i| i % 2 == 0).collect::<Vec<_>>());
        assert_eq!(s.search(&Query::Exists { field: 0 }, &all(100)).hits.len(), 40);
        assert!(s.search(&Query::Exists { field: 9 }, &all(100)).hits.is_empty());
        // Keyword multi-mode: max picks the last value (single-valued here).
        let td = s.search(&Query::MatchAll, &SearchOptions { top_k: 1, sort: vec![SortField::field(2, SortKind::Keyword).desc()], ..Default::default() });
        assert_eq!(td.hits[0].sort[0], SortValue::Str("k4".into()));
    }

    /// Randomised docs for comparing the single-field sort collector with
    /// the generic one: several sealed segments (each > 10 blocks, values
    /// clustered so blocks get skipped, many ties, sparse columns), plus a
    /// memtable leaf and deletions.
    struct SortFixture {
        sealed: Vec<crate::segment::Segment>,
        mem: SegmentWriter,
    }

    fn sort_fixture() -> SortFixture {
        let schema = vec![
            Field::new(0, "body", FieldType::Text),
            Field::new(1, "n", FieldType::I64),
            Field::new(2, "f", FieldType::F64),
            Field::new(3, "kw", FieldType::Keyword),
            Field::new(4, "mn", FieldType::I64),
            Field::new(5, "mk", FieldType::Keyword),
            Field::new(6, "u", FieldType::U64),
            // A skipper column (no points index), in runs as an index
            // sort's second key leaves it.
            Field::new(7, "ts", FieldType::I64).with_points(false),
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let dir = RamDirectory::new();
        let mut sealed = Vec::new();
        let mut mem = SegmentWriter::new(99);
        for seg in 0..4u32 {
            let mut w = SegmentWriter::new(seg);
            let n = if seg == 3 {
                300
            } else {
                1300 + rnd(700) as u32
            };
            for i in 0..n {
                let mut d = Document::new();
                let mut body = String::from("x");
                if rnd(10) < 3 {
                    body.push_str(" a");
                }
                if rnd(10) < 8 {
                    body.push_str(" b");
                }
                d.add_text(0, &body);
                if rnd(10) != 0 {
                    d.add_i64(1, (i as i64 / 37) * 3 - 200 + rnd(5) as i64);
                }
                if rnd(20) != 0 {
                    let x = [-0.0, 0.0, -1.5, 2.25][rnd(4) as usize]
                        + (rnd(50) as f64) * if i % 2 == 0 { 1.0 } else { -1.0 };
                    d.add_f64(2, x);
                }
                if rnd(20) != 0 {
                    d.add_keyword(3, &format!("t{:02}", (i / 64 + rnd(3) as u32) % 60));
                }
                for _ in 0..rnd(3) {
                    d.add_i64(4, rnd(1000) as i64 - 500);
                }
                for _ in 0..rnd(3) {
                    d.add_keyword(5, &format!("m{}", rnd(40)));
                }
                if rnd(8) != 0 {
                    d.add_u64(
                        6,
                        u64::MAX - rnd(5000) * (1 + i as u64 % 3) * 1_000_000_000_000,
                    );
                }
                if rnd(15) != 0 {
                    d.add_i64(7, (i % 97) as i64 * 10 + rnd(7) as i64);
                }
                if seg == 3 {
                    mem.add_document(&d, &schema);
                } else {
                    w.add_document(&d, &schema);
                }
            }
            if seg < 3 {
                sealed.push(w.write_to(&dir).unwrap());
            }
        }
        for s in &sealed {
            for _ in 0..40 {
                s.delete_doc(rnd(s.num_docs as u64) as DocId);
            }
        }
        mem.delete_doc(5);
        SortFixture { sealed, mem }
    }

    /// `cargo test -p rustic-core --release --lib term_query_microbench -- --ignored --nocapture`:
    /// the term query of Rally geonames (a keyword `country_code` over
    /// `TERM_SEGS` (default 10) segments of 230,000 docs: a ~1% and a ~12%
    /// code) and of pmc (a text `body` over 8 segments of long documents:
    /// a term in ~40% of them), size 10 with the default
    /// `track_total_hits` of 10,000, p50 / p99 over `TERM_RUNS` (default
    /// 3,000) runs each, one leaf set as one shard sees it.
    #[test]
    #[ignore]
    fn term_query_microbench() {
        let segs: u32 = std::env::var("TERM_SEGS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        let runs: usize = std::env::var("TERM_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3_000);
        let schema = vec![Field::new(0, "country_code", FieldType::Keyword), Field::new(1, "body", FieldType::Text)];
        // On disk: the RAM directory's budget holds less than these corpora.
        let tmp = tempfile::tempdir().unwrap();
        let dir = crate::codec::directory::MmapDirectory::open(tmp.path()).unwrap();
        let geo: Vec<_> = (0..segs).map(|seg| {
            let mut w = SegmentWriter::new(seg);
            for i in 0..230_000u32 {
                let g = (seg * 230_000 + i).wrapping_mul(2654435761);
                // Zipf-ish codes: code k with weight ~1/(k+1).
                let r = (g % 1_000_000) as f64 / 1_000_000.0;
                let k = ((250f64).powf(r) - 1.0) as u32;
                let mut d = Document::new();
                d.add_keyword(0, &format!("C{k}"));
                w.add_document(&d, &schema);
            }
            w.write_to(&dir).unwrap()
        }).collect();
        let words = ["the", "of", "patients", "study", "clinical", "physician", "results", "treatment", "health", "data", "cells", "protein"];
        let pmc: Vec<_> = (0..8u32).map(|seg| {
            let mut w = SegmentWriter::new(1000 + seg);
            for i in 0..30_000u32 {
                let mut x = ((seg * 30_000 + i) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
                let mut text = String::with_capacity(2400);
                for _ in 0..300 {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    let r = ((x % 1000) as f64 / 1000.0).powi(2);
                    // `physician` rarely; the common words often.
                    let word = if x % 700 == 0 { "physician" } else { words[(r * words.len() as f64) as usize % words.len()] };
                    text.push_str(word);
                    text.push(' ');
                }
                let mut d = Document::new();
                d.add_text(1, &text);
                w.add_document(&d, &schema);
            }
            w.write_to(&dir).unwrap()
        }).collect();
        let run = |label: &str, leaves: &[crate::segment::Segment], q: Query| {
            let s = Searcher::new(leaves.iter().map(LeafReader::Segment).collect());
            let opts = SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::UpTo(10_000), ..Default::default() };
            let first = s.search(&q, &opts);
            let mut t: Vec<f64> = (0..runs).map(|_| { let s0 = Instant::now(); std::hint::black_box(s.search(&q, &opts)); s0.elapsed().as_secs_f64() * 1e6 }).collect();
            t.sort_by(f64::total_cmp);
            eprintln!("{label:<34} p50 {:>8.1} us  p99 {:>8.1} us  (total {} {:?})", t[runs / 2], t[runs * 99 / 100], first.total_hits, first.total_hits_relation);
        };
        run("geonames term, ~1% code", &geo, Query::term(0, "C5"));
        run("geonames term, top code", &geo, Query::term(0, "C0"));
        run("geonames term, rare code", &geo, Query::term(0, "C200"));
        run("pmc term physician", &pmc, Query::term(1, "physician"));
        run("pmc term patients (frequent)", &pmc, Query::term(1, "patients"));
    }

    /// `cargo test -p rustic-core --release --lib many_leaves_microbench -- --ignored --nocapture`
    /// (`MANY_LEAVES`, default 72): geonames-shaped many-segment searches.
    /// A 5,000-value `terms` query (top 10, total up to 10,000) against the
    /// same query counted exactly (every leaf visited, the cost before leaf
    /// skipping), and match_all sorted by a long descending, best of 9.
    #[test]
    #[ignore]
    fn many_leaves_microbench() {
        let leaves_n: u32 = std::env::var("MANY_LEAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(72);
        let schema = vec![Field::new(0, "name", FieldType::Keyword), Field::new(1, "population", FieldType::I64)];
        let dir = RamDirectory::new();
        let sealed: Vec<_> = (0..leaves_n).map(|seg| {
            let mut w = SegmentWriter::new(seg);
            for i in 0..20_000u32 {
                let mut d = Document::new();
                let g = seg * 20_000 + i;
                d.add_keyword(0, &format!("name{:08}", g.wrapping_mul(2654435761) % 5_000_000));
                if i % 3 != 0 { d.add_i64(1, (g.wrapping_mul(40503) % 1_000_000) as i64); }
                w.add_document(&d, &schema);
            }
            w.write_to(&dir).unwrap()
        }).collect();
        let s = Searcher::new(sealed.iter().map(LeafReader::Segment).collect());
        let mut terms: Vec<Vec<u8>> = (0..5_000u32).map(|i| format!("name{:08}", (i * 997) % 5_000_000).into_bytes()).collect();
        terms.sort_unstable();
        terms.dedup();
        let q = Query::TermSet { field: 0, terms, boost: 1.0 };
        let time = |label: &str, f: &dyn Fn() -> u64| {
            let best = (0..9).map(|_| { let t = Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min);
            eprintln!("{label:<44} {:>9.3} ms", best * 1e3);
        };
        time("terms, total up to 10,000 (leaves skipped)", &|| s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::UpTo(10_000), ..Default::default() }).total_hits);
        time("terms, exact total (every leaf)", &|| s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::True, ..Default::default() }).total_hits);
        let sort = vec![SortField::Field { field: 1, kind: SortKind::I64, descending: true, missing: SortMissing::Last, mode: None }];
        time("match_all sort population desc (skipping)", &|| s.search(&Query::MatchAll, &SearchOptions { top_k: 10, sort: sort.clone(), ..Default::default() }).total_hits);
        time("same, generic collector", &|| s.search(&Query::MatchAll, &SearchOptions { top_k: 10, sort: sort.clone(), fast_sort: false, ..Default::default() }).total_hits);
    }

    /// match_all sorted by one numeric field: leaves whose best key cannot
    /// enter the page are skipped; hits, ties (equal values across leaves),
    /// missing values and totals match the generic collector.
    #[test]
    fn numeric_sort_leaf_skipping_matches_generic_collector() {
        let schema = vec![Field::new(0, "n", FieldType::I64), Field::new(1, "f", FieldType::F64)];
        let dir = RamDirectory::new();
        let sealed: Vec<_> = (0..10u32).map(|seg| {
            let mut w = SegmentWriter::new(seg);
            for i in 0..300u32 {
                let mut d = Document::new();
                // Leaf 0 holds the largest values; later leaves overlap and
                // repeat the page's boundary value (ties across leaves).
                if i % 13 != 0 { d.add_i64(0, 10_000 - (seg as i64) * 500 + (i % 40) as i64 * if seg % 3 == 0 { 1 } else { 20 }); }
                if i % 17 != 0 { d.add_f64(1, (seg * 7 % 10) as f64 + i as f64 / 1000.0); }
                w.add_document(&d, &schema);
            }
            let seg = w.write_to(&dir).unwrap();
            for d in (0..300).step_by(29) { seg.delete_doc(d).unwrap(); }
            seg
        }).collect();
        let s = Searcher::new(sealed.iter().map(LeafReader::Segment).collect());
        let key = |h: &TopHit| (h.leaf, h.doc, h.sort.clone());
        for (field, kind) in [(0u16, SortKind::I64), (1, SortKind::F64)] {
            for descending in [true, false] {
                for missing in [SortMissing::Last, SortMissing::First] {
                    for (top_k, tth) in [(10usize, TrackTotalHits::UpTo(10_000)), (3, TrackTotalHits::True), (50, TrackTotalHits::False), (0, TrackTotalHits::UpTo(5))] {
                        let sort = vec![SortField::Field { field, kind, descending, missing: missing.clone(), mode: None }];
                        // Best key first, one leaf at a time or the rest together.
                        for parallel in [true, false] {
                        let base = SearchOptions { sort: sort.clone(), top_k, track_total_hits: tth, parallel, ..Default::default() };
                        let fast = s.search(&Query::MatchAll, &base);
                        let generic = s.search(&Query::MatchAll, &SearchOptions { fast_sort: false, ..base.clone() });
                        let ctx = format!("{field} desc={descending} {missing:?} {top_k} {tth:?} parallel={parallel}");
                        assert_eq!(fast.hits.iter().map(key).collect::<Vec<_>>(), generic.hits.iter().map(key).collect::<Vec<_>>(), "{ctx}");
                        assert_eq!((fast.total_hits, fast.total_hits_relation), (generic.total_hits, generic.total_hits_relation), "{ctx}");
                        }
                    }
                }
            }
        }
    }

    /// A filtered search sorted by one numeric field with no total: leaves
    /// whose keys all sort before `search_after`, or after the page's last
    /// hit, are skipped; the hits match the generic collector (time-like
    /// disjoint leaves, overlapping ones, ties, missing values, asc/desc).
    #[test]
    fn filtered_numeric_sort_leaf_skipping_matches_generic_collector() {
        let schema = vec![Field::new(0, "ts", FieldType::I64), Field::new(1, "cat", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let sealed: Vec<_> = (0..14u32).map(|seg| {
            let mut w = SegmentWriter::new(seg);
            for i in 0..200i64 {
                let mut d = Document::new();
                // Mostly increasing (one leaf per time range), some leaves
                // spanning everything, values repeated across leaves.
                let ts = if seg % 5 == 4 { (i * 37) % 3000 } else { seg as i64 * 200 + i / 2 };
                if i % 23 != 0 { d.add_i64(0, ts); }
                d.add_keyword(1, ["a", "b", "c"][(i % 3) as usize]);
                w.add_document(&d, &schema);
            }
            w.write_to(&dir).unwrap()
        }).collect();
        let s = Searcher::new(sealed.iter().map(LeafReader::Segment).collect());
        let key = |h: &TopHit| (h.leaf, h.doc, h.sort.clone());
        let q = Query::term(1, "a");
        for descending in [false, true] {
            for top_k in [1usize, 7, 50, 400] {
                for after in [None, Some(-5i64), Some(450), Some(1200), Some(2799), Some(5000)] {
                    let sort = vec![SortField::Field { field: 0, kind: SortKind::I64, descending, missing: SortMissing::Last, mode: None }];
                    let search_after = after.map(|x| TopHit::new(usize::MAX, u32::MAX, f32::NEG_INFINITY, vec![SortValue::I64(x)]));
                    let base = SearchOptions { sort, top_k, track_total_hits: TrackTotalHits::False, search_after, ..Default::default() };
                    let fast = s.search(&q, &base);
                    let generic = s.search(&q, &SearchOptions { fast_sort: false, ..base.clone() });
                    let ctx = format!("desc={descending} {top_k} {after:?}");
                    assert_eq!(fast.hits.iter().map(key).collect::<Vec<_>>(), generic.hits.iter().map(key).collect::<Vec<_>>(), "{ctx}");
                }
            }
        }
    }

    /// A match_all sort whose key ties every doc (a constant column, or a
    /// field no doc has: an unmapped sort's missing value) takes the first
    /// docs and walks a few blocks, not the leaf (Rally geopoint
    /// `distanceSort` sorts on an unmapped field); any page size, asc and
    /// desc, the generic collector's hits.
    #[test]
    fn tied_match_all_sorts_walk_few_blocks() {
        let schema = vec![Field::new(0, "c", FieldType::I64), Field::new(1, "none", FieldType::I64)];
        let dir = RamDirectory::new();
        let mut w = SegmentWriter::new(0);
        for _ in 0..100_000 {
            let mut d = Document::new();
            d.add_i64(0, 7);
            w.add_document(&d, &schema);
        }
        let seg = w.write_to(&dir).unwrap();
        let s = Searcher::new(vec![LeafReader::Segment(&seg)]);
        let key = |h: &TopHit| (h.leaf, h.doc, h.sort.clone());
        for field in [0u16, 1] {
            for descending in [false, true] {
                for top_k in [1usize, 10, 300] {
                    let sort = vec![SortField::Field { field, kind: SortKind::I64, descending, missing: SortMissing::Last, mode: None }];
                    let base = SearchOptions { sort, top_k, track_total_hits: TrackTotalHits::UpTo(10_000), ..Default::default() };
                    let before = sort_fast::WALKED_BLOCKS.with(|w| w.get());
                    let fast = s.search(&Query::MatchAll, &base);
                    let walked = sort_fast::WALKED_BLOCKS.with(|w| w.get()) - before;
                    let generic = s.search(&Query::MatchAll, &SearchOptions { fast_sort: false, ..base.clone() });
                    assert_eq!(fast.hits.iter().map(key).collect::<Vec<_>>(), generic.hits.iter().map(key).collect::<Vec<_>>(), "field {field} desc={descending} {top_k}");
                    assert!(walked <= (top_k as u64).div_ceil(256) + 2, "field {field} desc={descending} {top_k}: {walked} blocks");
                }
            }
        }
    }

    /// A uniformly scoring query over many leaves: once earlier leaves hold
    /// the page and counted past the `track_total_hits` bound, later leaves
    /// are not visited (no scorer built), with the same hits and a `gte`
    /// total.
    #[test]
    fn uniform_queries_skip_leaves_after_hits_and_count_are_decided() {
        #[derive(Debug)]
        struct Counting(Query, std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl crate::query::query::CustomQuery for Counting {
            fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn crate::query::scorer::Scorer>> {
                self.1.fetch_add(1, AtomicOrdering::Relaxed);
                self.0.scorer(leaf, ctx)
            }
            fn children(&self) -> Vec<&Query> { vec![&self.0] }
        }
        let schema = vec![Field::new(0, "kw", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let sealed: Vec<_> = (0..12u32).map(|seg| {
            let mut w = SegmentWriter::new(seg);
            for i in 0..200 {
                let mut d = Document::new();
                d.add_keyword(0, &format!("t{}", i % 50));
                w.add_document(&d, &schema);
            }
            w.write_to(&dir).unwrap()
        }).collect();
        let s = Searcher::new(sealed.iter().map(LeafReader::Segment).collect());
        let terms: Vec<Vec<u8>> = (0..40).map(|i| format!("t{i}").into_bytes()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let built = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let q = Query::ConstScore { query: Box::new(Query::Custom(std::sync::Arc::new(Counting(Query::TermSet { field: 0, terms, boost: 1.0 }, built.clone())))), score: 1.0, point_set: false };
        let full = s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::True, ..Default::default() });
        assert_eq!(full.total_hits, 12 * 160);
        assert_eq!(built.swap(0, AtomicOrdering::Relaxed), 12);
        // In parallel batches: the same hits and total.
        let batched = s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::UpTo(300), parallel: true, ..Default::default() });
        assert_eq!(batched.hits.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>(), full.hits.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>());
        assert_eq!((batched.total_hits, batched.total_hits_relation), (300, TotalHitsRelation::Gte));
        built.store(0, AtomicOrdering::Relaxed);
        // One leaf at a time (parallel batches run the first ones together).
        let page = s.search(&q, &SearchOptions { top_k: 10, track_total_hits: TrackTotalHits::UpTo(300), parallel: false, ..Default::default() });
        let key = |h: &TopHit| (h.leaf, h.doc, h.score);
        assert_eq!(page.hits.iter().map(key).collect::<Vec<_>>(), full.hits.iter().map(key).collect::<Vec<_>>());
        assert_eq!((page.total_hits, page.total_hits_relation), (300, TotalHitsRelation::Gte));
        let visited = built.load(AtomicOrdering::Relaxed);
        assert!(visited < 12, "{visited} leaves built a scorer");
    }

    #[test]
    fn fast_field_sort_matches_generic_collector() {
        let fx = sort_fixture();
        let mut leaves: Vec<LeafReader<'_>> = fx.sealed.iter().map(LeafReader::Segment).collect();
        leaves.push(LeafReader::Memtable(&fx.mem, None));
        let s = Searcher::new(leaves);
        let queries = [
            Query::MatchAll,
            Query::Term(Term::new(0, "a")),
            Query::Term(Term::new(0, "b")),
        ];
        let mut sorts: Vec<SortField> = Vec::new();
        let missing_values = |kind: SortKind| match kind {
            SortKind::Keyword | SortKind::Fielddata => vec![
                SortMissing::Value(SortValue::Str("t30x".into())),
                SortMissing::Value(SortValue::Str("m7".into())),
            ],
            SortKind::F64 => vec![SortMissing::Value(SortValue::F64(3.0))],
            SortKind::I64 => vec![SortMissing::Value(SortValue::I64(40))],
        };
        for (field, kind) in [
            (1, SortKind::I64),
            (2, SortKind::F64),
            (3, SortKind::Keyword),
            (4, SortKind::I64),
            (5, SortKind::Keyword),
            (6, SortKind::I64),
            (7, SortKind::I64),
            (42, SortKind::I64),
        ] {
            for desc in [false, true] {
                let mut ms = vec![SortMissing::Last, SortMissing::First];
                ms.extend(missing_values(kind));
                for m in ms {
                    let modes: &[Option<SortMode>] = if field == 4 || field == 5 {
                        &[None, Some(SortMode::Min), Some(SortMode::Max)]
                    } else {
                        &[None]
                    };
                    for &mode in modes {
                        let mut sf = SortField::field(field, kind).with_missing(m.clone());
                        if let Some(mo) = mode {
                            sf = sf.with_mode(mo);
                        }
                        sorts.push(if desc { sf.desc() } else { sf });
                    }
                }
            }
        }
        let mut checked = 0;
        for q in &queries {
            for sf in &sorts {
                let all = s.search(
                    q,
                    &SearchOptions {
                        top_k: 2000,
                        sort: vec![sf.clone()],
                        fast_sort: false,
                        track_total_hits: TrackTotalHits::False,
                        ..Default::default()
                    },
                );
                // search_after: no cursor, cursors taken from hits (ties
                // across leaves), and bare sort values (no tiebreak).
                let mut afters: Vec<Option<TopHit>> = vec![None];
                for p in [3usize, 640, 1500] {
                    if let Some(h) = all.hits.get(p) {
                        afters.push(Some(h.clone()));
                        afters.push(Some(TopHit::new(
                            usize::MAX,
                            u32::MAX,
                            f32::NEG_INFINITY,
                            h.sort.clone(),
                        )));
                    }
                }
                for (k, tth) in [
                    (10, TrackTotalHits::UpTo(1_000)),
                    (1, TrackTotalHits::False),
                    (37, TrackTotalHits::True),
                ] {
                    let base = SearchOptions {
                        top_k: k,
                        sort: vec![sf.clone()],
                        track_total_hits: tth,
                        ..Default::default()
                    };
                    for after in afters.clone() {
                        let opts = SearchOptions {
                            search_after: after.clone(),
                            ..base.clone()
                        };
                        let fast = s.search(q, &opts);
                        let slow = s.search(
                            q,
                            &SearchOptions {
                                fast_sort: false,
                                ..opts.clone()
                            },
                        );
                        let ids = |td: &TopDocs| {
                            td.hits
                                .iter()
                                .map(|h| (h.leaf, h.doc, h.sort.clone(), h.score))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(
                            ids(&fast),
                            ids(&slow),
                            "{q:?} {sf:?} k={k} {tth:?} after={after:?}"
                        );
                        assert_eq!(
                            (fast.total_hits, fast.total_hits_relation),
                            (slow.total_hits, slow.total_hits_relation),
                            "{q:?} {sf:?} k={k} {tth:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 1000);
        assert!(FAST_LEAVES.load(AtomicOrdering::Relaxed) > 1000);
        let skipped = sort_fast::SKIPPED_BLOCKS.load(AtomicOrdering::Relaxed);
        assert!(skipped > 100, "{skipped}");
        assert!(sort_fast::POINT_WALKS.load(AtomicOrdering::Relaxed) > 100);
        assert!(sort_fast::BLOCK_WALKS.load(AtomicOrdering::Relaxed) > 10);
        assert!(sort_fast::CANDIDATE_LISTS.load(AtomicOrdering::Relaxed) > 10);
    }

    /// Leaves of a long column indexed in ascending order (http_logs'
    /// `@timestamp`), with runs of equal values, deletes and docs without a
    /// value, under a filter: the fast collector seeds its queue from the
    /// best blocks and returns exactly the generic collector's hits, desc
    /// and asc, for any size.
    #[test]
    fn seeded_sort_matches_generic_collector_on_ascending_columns() {
        let schema = vec![Field::new(0, "body", FieldType::Text), Field::new(1, "ts", FieldType::I64), Field::new(2, "f", FieldType::F64)];
        let dir = RamDirectory::new();
        let mut sealed = Vec::new();
        for seg in 0..3u32 {
            let mut w = SegmentWriter::new(seg);
            for i in 0..40_000u32 {
                let mut d = Document::new();
                d.add_text(0, if i % 7 < 4 { "x a" } else { "x b" });
                if i % 101 != 0 {
                    d.add_i64(1, (seg as i64 * 40_000 + i as i64) / 3);
                }
                d.add_f64(2, ((i * 2654435761u32) % 10_000) as f64 / 7.0);
                w.add_document(&d, &schema);
            }
            let s = w.write_to(&dir).unwrap();
            for k in 0..200u32 {
                s.delete_doc((k * 197 + seg * 13) % 40_000);
            }
            sealed.push(s);
        }
        let leaves: Vec<LeafReader<'_>> = sealed.iter().map(LeafReader::Segment).collect();
        let s = Searcher::new(leaves);
        // Seeded leaves, or (nothing to count) best-first block walks.
        let walked = || sort_fast::SEEDED_LEAVES.load(AtomicOrdering::Relaxed) + sort_fast::BEST_BLOCK_WALKS.load(AtomicOrdering::Relaxed);
        let before = walked();
        for q in [Query::Term(Term::new(0, "a")), Query::Term(Term::new(0, "x"))] {
            for (field, kind) in [(1u16, SortKind::I64), (2, SortKind::F64)] {
                for desc in [true, false] {
                    let sf = SortField::field(field, kind);
                    let sf = if desc { sf.desc() } else { sf };
                    for k in [1usize, 10, 100, 1000] {
                        let opts = SearchOptions { top_k: k, sort: vec![sf.clone()], track_total_hits: TrackTotalHits::False, ..Default::default() };
                        let fast = s.search(&q, &opts);
                        let slow = s.search(&q, &SearchOptions { fast_sort: false, ..opts.clone() });
                        let ids = |td: &TopDocs| td.hits.iter().map(|h| (h.leaf, h.doc, h.sort.clone())).collect::<Vec<_>>();
                        assert_eq!(ids(&fast), ids(&slow), "{q:?} {sf:?} k={k}");
                    }
                }
            }
        }
        assert!(walked() > before + 20);
    }

    #[test]
    fn best_block_walk_of_a_sparse_query_falls_back_to_doc_order() {
        // Descending on an ascending column: every block of the walk goes
        // back in doc order. A query matching one doc in 5,000 (an EQL stage
        // search) never fills the page, so the walk gives up after its
        // rewinds instead of building a scorer per block.
        let schema = vec![Field::new(0, "body", FieldType::Text), Field::new(1, "ts", FieldType::I64)];
        let dir = RamDirectory::new();
        let mut w = SegmentWriter::new(0);
        for i in 0..60_000u32 {
            let mut d = Document::new();
            d.add_text(0, if i % 5_000 == 17 { "x rare" } else { "x" });
            d.add_i64(1, i as i64);
            w.add_document(&d, &schema);
        }
        let seg = w.write_to(&dir).unwrap();
        let s = Searcher::new(vec![LeafReader::Segment(&seg)]);
        let before = sort_fast::BEST_BLOCK_BAILS.load(AtomicOrdering::Relaxed);
        for k in [1usize, 10, 1000] {
            let opts = SearchOptions { top_k: k, sort: vec![SortField::field(1, SortKind::I64).desc()], track_total_hits: TrackTotalHits::False, ..Default::default() };
            let q = Query::Term(Term::new(0, "rare"));
            let fast = s.search(&q, &opts);
            let slow = s.search(&q, &SearchOptions { fast_sort: false, ..opts.clone() });
            let ids = |td: &TopDocs| td.hits.iter().map(|h| (h.leaf, h.doc, h.sort.clone())).collect::<Vec<_>>();
            assert_eq!(ids(&fast), ids(&slow), "k={k}");
            assert_eq!(fast.hits.len(), k.min(12), "k={k}");
        }
        assert!(sort_fast::BEST_BLOCK_BAILS.load(AtomicOrdering::Relaxed) > before);
    }

    #[test]
    fn doc_sort_paging_matches_generic_collector() {
        let fx = sort_fixture();
        let mut leaves: Vec<LeafReader<'_>> = fx.sealed.iter().map(LeafReader::Segment).collect();
        leaves.push(LeafReader::Memtable(&fx.mem, None));
        let s = Searcher::new(leaves);
        for q in [Query::MatchAll, Query::Term(Term::new(0, "a"))] {
            for tth in [
                TrackTotalHits::UpTo(10_000),
                TrackTotalHits::False,
                TrackTotalHits::True,
                TrackTotalHits::UpTo(20),
            ] {
                for size in [1, 333, 1000] {
                    let mut after: Option<TopHit> = None;
                    let mut pages = 0;
                    loop {
                        let opts = SearchOptions {
                            top_k: size,
                            sort: vec![SortField::Doc],
                            search_after: after.clone(),
                            track_total_hits: tth,
                            ..Default::default()
                        };
                        let fast = s.search(&q, &opts);
                        let slow = s.search(
                            &q,
                            &SearchOptions {
                                fast_sort: false,
                                ..opts.clone()
                            },
                        );
                        let ids = |td: &TopDocs| {
                            td.hits
                                .iter()
                                .map(|h| (h.leaf, h.doc, h.sort.clone(), h.score))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(ids(&fast), ids(&slow), "{q:?} {tth:?} page {pages}");
                        assert_eq!(
                            (fast.total_hits, fast.total_hits_relation),
                            (slow.total_hits, slow.total_hits_relation),
                            "{q:?} {tth:?}"
                        );
                        let Some(last) = fast.hits.last() else { break };
                        after = Some(last.clone());
                        pages += 1;
                        if size == 1 && pages > 30 {
                            break;
                        }
                    }
                    assert!(pages >= 2);
                }
            }
            // A bare `_doc` value (no cursor hit): -1 and a mid-leaf value.
            for g in [-1i64, (1 << 32) | 17, (2 << 32) + (1 << 32) - 1] {
                let opts = SearchOptions {
                    top_k: 50,
                    sort: vec![SortField::Doc],
                    search_after: Some(TopHit::new(
                        usize::MAX,
                        u32::MAX,
                        0.0,
                        vec![SortValue::I64(g)],
                    )),
                    ..Default::default()
                };
                let fast = s.search(&q, &opts);
                let slow = s.search(
                    &q,
                    &SearchOptions {
                        fast_sort: false,
                        ..opts.clone()
                    },
                );
                assert_eq!(fast.hits, slow.hits);
            }
        }
    }

    #[test]
    fn leaf_ordered_pages_are_prefixes_of_the_full_ranking() {
        let fx = sort_fixture();
        let mut leaves: Vec<LeafReader<'_>> = fx.sealed.iter().map(LeafReader::Segment).collect();
        leaves.push(LeafReader::Memtable(&fx.mem, None));
        let s = Searcher::new(leaves);
        let queries = [Query::MatchAll, Query::ConstScore { query: Box::new(Query::Term(Term::new(0, "a"))), score: 2.0, point_set: false }, Query::Term(Term::new(0, "b"))];
        for q in &queries {
            for sort in [vec![SortField::Doc], vec![], vec![SortField::Score { descending: true }]] {
                for tth in [TrackTotalHits::UpTo(100), TrackTotalHits::True, TrackTotalHits::False] {
                    let base = SearchOptions { sort: sort.clone(), track_total_hits: tth, ..Default::default() };
                    let full = s.search(q, &SearchOptions { top_k: 100_000, ..base.clone() });
                    let key = |h: &TopHit| (h.leaf, h.doc, h.score);
                    for (from, size) in [(0usize, 37usize), (0, 1), (30, 1700), (0, 4000)] {
                        let page = s.search(q, &SearchOptions { top_k: size, from, ..base.clone() });
                        let want: Vec<_> = full.hits.iter().skip(from).take(size).map(key).collect();
                        assert_eq!(page.hits.iter().map(key).collect::<Vec<_>>(), want, "{q:?} {sort:?} {tth:?} {from}+{size}");
                        if tth == TrackTotalHits::True {
                            assert_eq!(page.total_hits, full.total_hits);
                        }
                    }
                    if !sort.is_empty() {
                        for p in [0usize, 777, 2500] {
                            let Some(a) = full.hits.get(p) else { continue };
                            let page = s.search(q, &SearchOptions { top_k: 50, search_after: Some(a.clone()), ..base.clone() });
                            let want: Vec<_> = full.hits.iter().skip(p + 1).take(50).map(key).collect();
                            assert_eq!(page.hits.iter().map(key).collect::<Vec<_>>(), want, "{q:?} {sort:?} after {p}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn score_sort_after_matches_generic_collector() {
        let fx = sort_fixture();
        let mut leaves: Vec<LeafReader<'_>> = fx.sealed.iter().map(LeafReader::Segment).collect();
        leaves.push(LeafReader::Memtable(&fx.mem, None));
        let s = Searcher::new(leaves);
        for q in [
            Query::MatchAll,
            Query::Term(Term::new(0, "a")),
            Query::ConstScore {
                query: Box::new(Query::Term(Term::new(0, "b"))),
                score: 2.0,
                point_set: false,
            },
        ] {
            for tth in [
                TrackTotalHits::UpTo(10_000),
                TrackTotalHits::False,
                TrackTotalHits::True,
                TrackTotalHits::UpTo(20),
            ] {
                // Page through everything like a scroll.
                let mut after: Option<TopHit> = None;
                let mut pages = 0;
                loop {
                    let opts = SearchOptions {
                        top_k: 700,
                        sort: vec![SortField::Score { descending: true }],
                        search_after: after.clone(),
                        track_total_hits: tth,
                        ..Default::default()
                    };
                    let fast = s.search(&q, &opts);
                    let slow = s.search(
                        &q,
                        &SearchOptions {
                            fast_sort: false,
                            ..opts.clone()
                        },
                    );
                    let ids = |td: &TopDocs| {
                        td.hits
                            .iter()
                            .map(|h| (h.leaf, h.doc, h.sort.clone(), h.score))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(ids(&fast), ids(&slow), "{q:?} {tth:?} page {pages}");
                    assert_eq!(
                        (fast.total_hits, fast.total_hits_relation),
                        (slow.total_hits, slow.total_hits_relation),
                        "{q:?} {tth:?}"
                    );
                    let Some(last) = fast.hits.last() else { break };
                    after = Some(last.clone());
                    pages += 1;
                }
                assert!(pages >= 2);
            }
        }
    }

    #[test]
    fn scalar_numeric_primary_pruning_matches_eager_sort_and_paging() {
        let schema = vec![Field::new(0, "price", FieldType::F64),
            Field::new(1, "id", FieldType::I64), Field::new(2, "unsigned", FieldType::U64)];
        let dir = RamDirectory::new();
        let mut persisted = SegmentWriter::new(0);
        let mut mem = SegmentWriter::new(1);
        for i in 0..1000u32 {
            let mut doc = Document::new();
            if i % 7 != 0 {
                doc.add_f64(0, if i % 11 == 0 { -0.0 } else { (i * 37 % 23) as f64 - 11.0 });
                doc.add_u64(2, if i % 2 == 0 { u64::MAX - i as u64 } else { i as u64 });
            }
            doc.add_i64(1, i as i64);
            if i < 400 { persisted.add_document(&doc, &schema); } else { mem.add_document(&doc, &schema); }
        }
        let segment = persisted.write_to(&dir).unwrap();
        let leaves = vec![LeafReader::Segment(&segment), LeafReader::Memtable(&mem, None)];
        let searcher = Searcher::new(leaves.clone());
        for (field, kind) in [(0, SortKind::F64), (1, SortKind::I64), (2, SortKind::I64)] {
            for mode in [SortMode::Min, SortMode::Max, SortMode::Sum, SortMode::Avg, SortMode::Median] {
                for desc in [false, true] {
                    for missing in [SortMissing::Last, SortMissing::First, SortMissing::Value(SortValue::I64(-5))] {
                        let mut first = SortField::field(field, kind).with_mode(mode).with_missing(missing);
                        if desc { first = first.desc(); }
                        let sort = vec![first, SortField::field(1, SortKind::I64)];
                        let specs = Searcher::specs(&sort);
                        let mut eager = Vec::new();
                        for (li, leaf) in leaves.iter().enumerate() {
                            let primary = Searcher::primary_numeric(leaf, &sort).expect("scalar column");
                            for doc in 0..leaf.num_docs() {
                                let key = Searcher::sort_key(&SearchContext::default(),li, leaf, doc, 1.0, &sort, &[], &[], &[]);
                                match (&primary(doc), &key[0]) {
                                    (SortValue::F64(a), SortValue::F64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                                    (a, b) => assert_eq!(a, b),
                                }
                                eager.push(TopHit::new(li, doc, 1.0, key));
                            }
                        }
                        eager.sort_by(|a, b| Searcher::cmp_hits(a, b, false, &specs));
                        let ids = |hits: &[TopHit]| hits.iter().map(|h| (h.leaf, h.doc)).collect::<Vec<_>>();
                        for k in [0, 1, 10, 100] {
                            let opts = SearchOptions { top_k: k, sort: sort.clone(), track_total_hits: TrackTotalHits::True, ..Default::default() };
                            let result = searcher.search(&Query::MatchAll, &opts);
                            assert_eq!(result.total_hits, 1000);
                            assert_eq!(ids(&result.hits), ids(&eager[..k]));
                            let page = searcher.search(&Query::MatchAll, &SearchOptions { search_after: Some(eager[7].clone()), ..opts });
                            assert_eq!(page.total_hits, 1000);
                            assert_eq!(ids(&page.hits), ids(&eager[8..8+k]));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn fast_sort_preserves_min_score_and_explanation_scores() {
        let fx = sort_fixture();
        let mut leaves: Vec<_> = fx.sealed.iter().map(LeafReader::Segment).collect();
        leaves.push(LeafReader::Memtable(&fx.mem, None));
        let searcher = Searcher::new(leaves);
        for query in [Query::MatchAll, Query::ConstScore { query: Box::new(Query::MatchAll), score: 2.0, point_set: false }, Query::Term(Term::new(0, "a"))] {
            for sort in [SortField::Doc, SortField::field(1, SortKind::I64), SortField::Score { descending: true }] {
                for min_score in [None, Some(0.0), Some(2.5)] {
                    for track_scores in [false, true] {
                        for explain in [false, true] {
                            for track_total_hits in [TrackTotalHits::True, TrackTotalHits::UpTo(20), TrackTotalHits::False] {
                                let search_after = matches!(sort, SortField::Score { .. }).then(|| TopHit::new(usize::MAX, u32::MAX, 100.0, vec![SortValue::F64(100.0)]));
                                let opts = SearchOptions { top_k: 7, sort: vec![sort.clone()], min_score, track_scores, explain, track_total_hits, search_after, ..Default::default() };
                                let fast = searcher.search(&query, &opts);
                                let generic = searcher.search(&query, &SearchOptions { fast_sort: false, ..opts.clone() });
                                assert_eq!(fast.hits, generic.hits, "{query:?} {sort:?} min={min_score:?} scores={track_scores} explain={explain}");
                                assert_eq!((fast.total_hits, fast.total_hits_relation), (generic.total_hits, generic.total_hits_relation), "{query:?} {sort:?} min={min_score:?} scores={track_scores} explain={explain}");
                            }
                        }
                    }
                }
            }
        }
    }

}
