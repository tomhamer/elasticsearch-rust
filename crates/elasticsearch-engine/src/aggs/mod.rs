//! Aggregations: bucket, metric and pipeline aggregations computed over the
//! matching doc set of a search, with sub-aggregations at any depth.
//!
//! A single-shard search computes every aggregation directly into its
//! final ES response JSON. Multi-shard searches compute per-shard partials
//! and reduce them (`hooks::agg_partial`); across nodes the cluster reduces
//! the rendered per-node results (`front/agg_reduce.rs`). Bucket aggregations partition the per-leaf doc lists and
//! recurse into their sub-aggregations; pipeline aggregations then run over
//! the produced JSON (sibling pipelines after their level, parent pipelines
//! over the bucket list of their multi-bucket parent).

// P5: analytics / ML aggregations and pipelines (registry: analytics::{AGGS, PARENT, SIBLING}).
mod analytics;
pub use analytics::frequent_items;
pub mod expr;
mod cmath;
mod extra;
pub mod auto_date;
pub mod pvalue;
mod geo_nested;
pub(crate) use geo_nested::{nested_whole_path_spec, DistanceCounts, PointGrid};
pub(crate) mod geo_hex;
mod geo_line;
pub(crate) mod geo_shape;
pub(crate) mod shape_tiler;
pub mod sketch;
pub mod significant_text;
// Hook 10 (partial aggregation results + reduce): `hooks/agg_partial.rs`.
#[path = "../hooks/agg_partial.rs"]
pub mod partial;

use crate::date::{add_units, parse_date_math, round_down, CalUnit, DateFormatter, Tz};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, ResolvedField};
use crate::query::dsl::{parse_query, QueryContext};
use crate::values;
use rustic_core::query::{Bm25Params, LeafReader, SearchContext, TERMINATED};
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use sketch::{Hdr, HyperLogLog, TDigest};
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The matching docs of one leaf of one index.
pub struct AggSeg<'a> {
    pub leaf: LeafReader<'a>,
    pub lookup: &'a FieldLookup,
    pub index: &'a str,
    /// Effective index.max_regex_length for this leaf's owning index.
    pub max_regex_length: usize,
    /// Matching live docs, ascending.
    pub docs: Vec<DocId>,
    /// Scores parallel to `docs` (empty when not scored).
    pub scores: Vec<f32>,
}

/// Fetches one hit for `top_hits`: `(seg index, doc, score, top_hits spec)` → hit JSON.
pub type FetchHit<'a> = dyn Fn(usize, DocId, f32, &Value) -> Result<Value> + Sync + 'a;

/// Loads only the source fields needed by source-based aggregations.
pub type FetchSource<'a> = dyn Fn(usize, &[(DocId, f32)], &[String], &mut dyn FnMut(DocId, Option<Value>) -> Result<()>) -> Result<()> + Sync + 'a;

pub struct AggEnv<'a> {
    /// Per-index mappings survive even when an empty index has no leaves.
    pub mapping_context: &'a [(&'a str, &'a FieldLookup)],
    pub now: i64,
    pub max_buckets: usize,
    pub fetch_hit: &'a FetchHit<'a>,
    pub fetch_source: Option<&'a FetchSource<'a>>,
    /// All visible live docs of segment i, independent of the request query
    /// (for `global` and aggregation background sets).
    pub all_docs: &'a (dyn Fn(usize) -> Vec<DocId> + Sync + 'a),
    /// Alias/DLS visibility restricts the background set. Significance uses
    /// maxDoc/docFreq only when this is false and no background_filter exists.
    pub background_filtered: bool,
    /// `index.top_metrics_max_size` (default 10): the max `top_metrics.size`.
    pub top_metrics_max_size: usize,
    /// `index.max_result_window` (the `ids` query's limit).
    pub max_result_window: usize,
    /// The search has `profile: true` (aggregations add debug info).
    pub profile: bool,
}

/// Per-segment `(doc, score)` lists.
type Docs = Vec<Vec<(DocId, f32)>>;

const PARENT_PIPELINES: &[&str] = &["bucket_sort", "bucket_selector", "bucket_script", "cumulative_sum", "derivative", "normalize", "cumulative_cardinality", "moving_percentiles", "serial_diff", "moving_fn"];
const SIBLING_PIPELINES: &[&str] = &["avg_bucket", "sum_bucket", "max_bucket", "min_bucket", "stats_bucket", "extended_stats_bucket", "percentiles_bucket", "bucket_correlation", "bucket_count_ks_test", "change_point"];
const KNOWN: &[&str] = &[
    "terms", "range", "date_range", "histogram", "date_histogram", "filter", "filters", "missing", "global", "composite", "cardinality", "avg", "sum",
    "min", "max", "value_count", "stats", "extended_stats", "percentiles", "percentile_ranks", "top_hits", "sampler", "diversified_sampler",
    "scripted_metric", "significant_text",
];

pub mod strict;
mod scripted;
pub use scripted::reduce_states as scripted_metric_reduce;
pub use scripted::{reduce_source_states as scripted_metric_reduce_source,render_values as render_scripted_values,has_raw_value as has_raw_scripted_value};

/// Request-level validation of the `aggs` object (errors ES raises while
/// parsing / validating the request, before executing it).
pub fn validate_request(spec: &Value) -> Result<()> {
    strict::check_tree(spec, None)?;
    significant_text::validate_contexts(spec)?;
    match spec.as_object() {
        Some(m) => {
            analytics::validate_tree(m, None)?;
            check_single_bucket_parents(m)
        }
        None => Ok(()),
    }
}

/// `PipelineAggregator.asMultiBucketAggregation`: parent pipelines that are
/// not checked at request time fail the coordinator reduce under a
/// single-bucket parent.
fn check_single_bucket_parents(aggs: &Map<String, Value>) -> Result<()> {
    const SINGLE: &[(&str, &str)] = &[
        ("filter", "InternalFilter"),
        ("nested", "InternalNested"),
        ("reverse_nested", "InternalReverseNested"),
        ("global", "InternalGlobal"),
        ("sampler", "InternalSampler"),
        ("diversified_sampler", "InternalSampler"),
        ("random_sampler", "InternalRandomSampler"),
        ("missing", "InternalMissing"),
        ("children", "InternalChildren"),
        ("parent", "InternalParent"),
    ];
    const UNCHECKED_PARENTS: &[&str] = &["normalize", "bucket_script", "bucket_selector", "bucket_sort", "cumulative_cardinality", "moving_percentiles", "inference"];
    for def in aggs.values().filter_map(Value::as_object) {
        let kind = def.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).map(String::as_str).unwrap_or("");
        let Some(sub) = def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object) else { continue };
        if let Some((_, class)) = SINGLE.iter().find(|(k, _)| *k == kind) {
            for (name, sd) in sub {
                let sk = sd.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta"))).map(String::as_str).unwrap_or("");
                if UNCHECKED_PARENTS.contains(&sk) {
                    // A coordinator reduce failure: `phase: fetch`, no shard root cause.
                    let cause = EsError::illegal_argument(format!("Expected a multi bucket aggregation but got [{class}] for aggregation [{name}]"));
                    return Err(EsError::new(400, "search_phase_execution_exception", "").with("phase", "fetch").with("grouped", true).with("failed_shards", json!([])).caused_by(cause));
                }
            }
        }
        check_single_bucket_parents(sub)?;
    }
    Ok(())
}

/// Run the `aggs` request object over the matching docs; returns the
/// `aggregations` response object.
pub fn run_aggregations(spec: &Value, segs: &[AggSeg], env: &AggEnv) -> Result<Value> {
    let ranges = QUERY_RANGES.with(|r| r.borrow().clone());
    run_aggregations_in_ranges(spec, segs, env, ranges)
}

thread_local! {
    /// Query ranges of the aggregation's segments set by [`with_query_ranges`].
    static QUERY_RANGES: std::cell::RefCell<Vec<Option<(u16, i64, i64)>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with `ranges` as the [`run_aggregations_in_ranges`] query ranges
/// of the aggregations it runs on this thread (by segment index).
pub fn with_query_ranges<R>(ranges: Vec<Option<(u16, i64, i64)>>, f: impl FnOnce() -> R) -> R {
    let prev = QUERY_RANGES.with(|r| r.replace(ranges));
    let out = f();
    QUERY_RANGES.with(|r| *r.borrow_mut() = prev);
    out
}

thread_local! {
    /// Segments whose query matches were left uncollected, set by
    /// [`with_elided_root`].
    static ELIDED_ROOT: std::cell::RefCell<Vec<bool>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The field of an `aggs` request whose every top-level aggregation is a
/// count-only `auto_date_histogram` on that one field (no sub-aggregations,
/// `missing`, `meta` or other options): Rally metricbeat's
/// `auto-date-histogram(-with-tz)`. Such buckets count from the field's
/// points index over the query's range, so a search whose query is that
/// range need not collect its matches ([`with_elided_root`]).
pub fn points_counted_field(spec: &Value) -> Option<&str> {
    let m = spec.as_object()?;
    let mut field: Option<&str> = None;
    for def in m.values() {
        let d = def.as_object()?;
        if d.len() != 1 {
            return None;
        }
        // Or a count-only `date_histogram` (Rally nyc_taxis
        // `date_histogram_agg`): its buckets count from the points index
        // over the query's range too (`date_histogram`'s elided path).
        let o = match (d.get("auto_date_histogram"), d.get("date_histogram")) {
            (Some(a), None) => {
                let o = a.as_object()?;
                if o.keys().any(|k| !matches!(k.as_str(), "field" | "buckets" | "time_zone" | "format" | "minimum_interval")) {
                    return None;
                }
                o
            }
            (None, Some(h)) => {
                let o = h.as_object()?;
                if o.keys().any(|k| !matches!(k.as_str(), "field" | "calendar_interval" | "fixed_interval" | "time_zone" | "format" | "min_doc_count" | "offset" | "keyed")) {
                    return None;
                }
                o
            }
            _ => return None,
        };
        let f = o.get("field")?.as_str()?;
        if field.is_some_and(|g| g != f) {
            return None;
        }
        field = Some(f);
    }
    field
}

/// [`points_counted_field`] when every aggregation is a `date_histogram`
/// (Rally tsdb `date-histo-entire-range`): over a `match_all` its buckets
/// count from the column's block min / max
/// ([`Ctx::count_whole_by_blocks`]), so the roots are not collected.
pub fn match_all_counted_field(spec: &Value) -> Option<&str> {
    let f = points_counted_field(spec)?;
    spec.as_object()?.values().all(|d| d.get("date_histogram").is_some()).then_some(f)
}

thread_local! {
    /// `double` ranges of uncollected segments, set by
    /// [`with_query_f64_ranges`].
    static QUERY_F64_RANGES: std::cell::RefCell<Vec<Option<(u16, f64, f64)>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with `ranges[s]` the `double` range ([`rustic_core::query::Query::sole_f64_range`])
/// whose docs are segment `s`'s uncollected matches ([`with_elided_root`]).
pub fn with_query_f64_ranges<R>(ranges: Vec<Option<(u16, f64, f64)>>, f: impl FnOnce() -> R) -> R {
    let prev = QUERY_F64_RANGES.with(|r| r.replace(ranges));
    let out = f();
    QUERY_F64_RANGES.with(|r| *r.borrow_mut() = prev);
    out
}

/// The histogram field of an `aggs` request that is one `histogram` whose
/// only sub-aggregations are `avg` / `sum` / `min` / `max` / `stats` of a
/// field (Rally nyc_taxis `distance_amount_agg`): a search whose query is a
/// bare `double` range on that field need not collect its matches; the
/// histogram folds them from the column ([`with_query_f64_ranges`]).
pub fn value_range_histogram_field(spec: &Value) -> Option<&str> {
    let m = spec.as_object()?;
    if m.len() != 1 {
        return None;
    }
    let d = m.values().next()?.as_object()?;
    let h = d.get("histogram")?.as_object()?;
    if d.keys().any(|k| k != "histogram" && k != "aggs" && k != "aggregations") || d.len() > 2 {
        return None;
    }
    if h.keys().any(|k| !matches!(k.as_str(), "field" | "interval" | "offset" | "min_doc_count" | "order" | "keyed" | "hard_bounds" | "extended_bounds")) {
        return None;
    }
    // No sub-aggregations: the fused pass counts only.
    let no_subs = Map::new();
    let subs = match d.get("aggs").or_else(|| d.get("aggregations")) {
        Some(s) => s.as_object()?,
        None => &no_subs,
    };
    for def in subs.values() {
        let x = def.as_object().filter(|x| x.len() == 1)?;
        let (t, body) = x.iter().next()?;
        if !matches!(t.as_str(), "avg" | "sum" | "min" | "max" | "stats") || body.as_object().is_none_or(|b| b.len() != 1 || !b.get("field").is_some_and(Value::is_string)) {
            return None;
        }
    }
    h.get("field")?.as_str()
}

/// The fields of an `aggs` request that is one `range` aggregation over a
/// field (`field` and `ranges`, maybe `keyed`) whose only sub-aggregation is
/// an `auto_date_histogram` (`field`, `buckets`, `time_zone`, `format`,
/// `minimum_interval`), maybe with plain `avg` / `sum` / `min` / `max`
/// sub-aggregations of a field (Rally big5 `range-auto-date-histo` and its
/// `-with-metrics` variant): `[range field, date field, metric fields..]`.
/// Over a `match_all` the range buckets fold every doc from the columns
/// ([`Ctx::range_auto_date_fused`]), so the roots are not collected
/// ([`with_elided_root`]).
pub fn range_auto_date_fields(spec: &Value) -> Option<Vec<&str>> {
    let m = spec.as_object().filter(|m| m.len() == 1)?;
    let d = m.values().next()?.as_object()?;
    let r = d.get("range")?.as_object()?;
    if d.keys().any(|k| !matches!(k.as_str(), "range" | "aggs" | "aggregations")) || d.len() != 2
        || r.keys().any(|k| !matches!(k.as_str(), "field" | "ranges" | "keyed")) {
        return None;
    }
    let mut fields = vec![r.get("field")?.as_str()?];
    let sub = d.get("aggs").or_else(|| d.get("aggregations"))?.as_object().filter(|s| s.len() == 1)?;
    let ad = sub.values().next()?.as_object()?;
    if ad.keys().any(|k| !matches!(k.as_str(), "auto_date_histogram" | "aggs" | "aggregations")) || ad.len() > 2 {
        return None;
    }
    let o = ad.get("auto_date_histogram")?.as_object()?;
    if o.keys().any(|k| !matches!(k.as_str(), "field" | "buckets" | "time_zone" | "format" | "minimum_interval")) {
        return None;
    }
    fields.push(o.get("field")?.as_str()?);
    if let Some(metrics) = ad.get("aggs").or_else(|| ad.get("aggregations")) {
        for def in metrics.as_object()?.values() {
            let x = def.as_object().filter(|x| x.len() == 1)?;
            let (t, body) = x.iter().next()?;
            let body = body.as_object().filter(|b| b.len() == 1)?;
            if !matches!(t.as_str(), "avg" | "sum" | "min" | "max") {
                return None;
            }
            fields.push(body.get("field")?.as_str()?);
        }
    }
    Some(fields)
}

/// Run `f` with `elided[s]` marking segment `s`'s doc list as left empty
/// because the query's matches there are exactly the docs whose value of
/// the [`points_counted_field`] lies in the query's range
/// ([`with_query_ranges`]; a bare range query, no deletions). The
/// aggregation counts them from the points index, or reads them back from
/// the column where it cannot.
pub fn with_elided_root<R>(elided: Vec<bool>, f: impl FnOnce() -> R) -> R {
    let prev = ELIDED_ROOT.with(|r| r.replace(elided));
    let out = f();
    ELIDED_ROOT.with(|r| *r.borrow_mut() = prev);
    out
}

/// [`run_aggregations`] where `query_ranges[s]` is a `(core field, lo, hi)`
/// long / date range every doc of segment `s`'s query matches satisfies
/// ([`rustic_core::query::Query::required_i64_range`]): top-level
/// aggregations on that field then count from the points index.
pub fn run_aggregations_in_ranges(spec: &Value, segs: &[AggSeg], env: &AggEnv, query_ranges: Vec<Option<(u16, i64, i64)>>) -> Result<Value> {
    let spec = spec.as_object().ok_or_else(|| EsError::parsing_at("Expected [START_OBJECT] under [aggregations], but got a [VALUE_STRING]", 1, 1))?;
    let docs: Docs = segs
        .iter()
        .map(|s| s.docs.iter().enumerate().map(|(i, d)| (*d, s.scores.get(i).copied().unwrap_or(0.0))).collect())
        .collect();
    let ctx = Ctx { segs, env, buckets: Cell::new(0), cols: Default::default(), rt_shapes: Default::default(), in_time_series: Cell::new(0), fielddata: Default::default(), memory_error:Default::default(), mem: crate::memory::RequestMemory::new(), bg_counts: Default::default(),
        root_docs: &docs as *const Docs as usize, query_ranges, elided: ELIDED_ROOT.with(|r| std::mem::take(&mut *r.borrow_mut())),
        query_f64_ranges: QUERY_F64_RANGES.with(|r| std::mem::take(&mut *r.borrow_mut())) };
    // The per-bucket doc lists copy every matching doc once.
    ctx.mem.add(docs.iter().map(|d| d.len() as u64 * 8 + 24).sum(), "<agg [_docs]>")?;
    let weights = doc_count_weights(segs);
    let prev = DOC_WEIGHTS.with(|w| w.replace(weights));
    let out = ctx.compute(spec, &docs);
    DOC_WEIGHTS.with(|w| *w.borrow_mut() = prev);
    if let Some(error)=ctx.memory_error.into_inner(){return Err(error);}
    Ok(Value::Object(out?))
}

#[cfg(test)]
thread_local! {
    /// Segments whose `date_histogram` counts came from the points index
    /// ([`Ctx::count_whole_segment_by_points`]), for tests to see the path taken.
    static POINTS_COUNTED_SEGMENTS: Cell<usize> = const { Cell::new(0) };
}

thread_local! {
    /// Set by a `nested` aggregation for its count-only `date_histogram`
    /// sub-aggregations: `(path, per segment Some(path ordinal))` where that
    /// segment's doc list was left empty because it is every doc of the
    /// nested path (all live roots, no deletes); see
    /// [`Ctx::nested_path_docs`].
    static NESTED_WHOLE_PATH: std::cell::RefCell<Option<(String, Vec<Option<u16>>)>> = const { std::cell::RefCell::new(None) };
}

/// Restores [`NESTED_WHOLE_PATH`] when dropped.
pub(super) struct WholePathGuard(Option<(String, Vec<Option<u16>>)>);

impl WholePathGuard {
    pub(super) fn set(v: (String, Vec<Option<u16>>)) -> Self {
        WholePathGuard(NESTED_WHOLE_PATH.with(|w| w.replace(Some(v))))
    }
}

impl Drop for WholePathGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        NESTED_WHOLE_PATH.with(|w| *w.borrow_mut() = prev);
    }
}

thread_local! {
    /// Per-segment `_doc_count` of each doc (`None`: every doc counts 1) of
    /// the aggregation running on this thread.
    static DOC_WEIGHTS: std::cell::RefCell<Option<Vec<Option<Vec<u32>>>>> = const { std::cell::RefCell::new(None) };
}

/// `_doc_count` values of the segments that have any (`DocCountProvider`).
fn doc_count_weights(segs: &[AggSeg]) -> Option<Vec<Option<Vec<u32>>>> {
    let mut any = false;
    let w: Vec<Option<Vec<u32>>> = segs
        .iter()
        .map(|s| {
            let id = s.lookup.doc_count_id?;
            let present = s.leaf.value_presence(id)?;
            if present.is_empty() {
                return None;
            }
            let mut v = vec![1u32; s.leaf.num_docs() as usize];
            for d in present {
                if let Some(n) = s.leaf.i64_value(id, d) {
                    if let Some(slot) = v.get_mut(d as usize) {
                        *slot = n.clamp(0, u32::MAX as i64) as u32;
                    }
                }
            }
            any = true;
            Some(v)
        })
        .collect();
    any.then_some(w)
}

/// Internal `significant_terms` option a shard partial sets: keep the
/// shard's best candidates whatever the sign of their shard-level score
/// (the cross-shard reduce filters them).
pub(crate) const SHARD_CANDIDATES: &str = "__rustic_shard_candidates";

struct Ctx<'a, 'b> {
    segs: &'b [AggSeg<'a>],
    env: &'b AggEnv<'a>,
    buckets: Cell<usize>,
    /// Doc-value columns of sealed segments, opened once per aggregation
    /// request instead of once per document (`(seg, core field id)`).
    cols: std::cell::RefCell<HashMap<(usize, u16), std::rc::Rc<Col<'a>>, ahash::RandomState>>,
    /// Value sources of `geo_shape` runtime fields (`(seg, field)`).
    rt_shapes: std::cell::RefCell<HashMap<(usize, String), crate::script::search::ValueSource>>,
    /// Nesting depth of `time_series` aggregations (`geo_line` defaults its sort there).
    in_time_series: Cell<u32>,
    /// Uninverted fielddata of text fields (`(seg, core field id)`), looked
    /// up once per request (the leaf itself caches it across requests).
    memory_error:std::cell::RefCell<Option<EsError>>,
    fielddata: std::cell::RefCell<HashMap<(usize, u16), Arc<rustic_core::fielddata::Uninverted>, ahash::RandomState>>,
    /// Request-breaker accounting of the buckets' state (ES
    /// `AggregatorBase.addRequestCircuitBreakerBytes`), released when the
    /// aggregation finishes or fails.
    mem: crate::memory::RequestMemory,
    /// `significant_terms` background counts per (field, background
    /// filter): the same for every parent bucket, so counted once per
    /// request (ES counts them once per shard too).
    bg_counts: std::cell::RefCell<HashMap<String, std::rc::Rc<(HashMap<Key, u64>, u64)>>>,
    /// Address of the top-level doc lists (the query's matches): a doc set
    /// at this address is the query's, not a bucket's or `global`'s.
    root_docs: usize,
    /// See [`run_aggregations_in_ranges`].
    query_ranges: Vec<Option<(u16, i64, i64)>>,
    /// See [`with_elided_root`].
    elided: Vec<bool>,
    /// See [`with_query_f64_ranges`].
    query_f64_ranges: Vec<Option<(u16, f64, f64)>>,
}

/// Bytes of a new bucket's grouping entry beyond its key: the hash slot
/// and the per-segment doc list headers.
const GROUP_BYTES: u64 = 64;
/// Bytes of a rendered bucket (counted by [`Ctx::add_buckets`]).
const BUCKET_BYTES: u64 = 256;

fn agg_label(name: &str) -> String {
    format!("<agg [{name}]>")
}

/// An opened doc-value column of a sealed segment.
/// `f(doc, score, value)` for each doc of `v` (ascending) with a value in
/// `c`. A long, dense doc list over a column without presence words is read
/// a block at a time ([`ColumnReader::block_values`]: a block-packed or
/// delta block decodes once, not per doc); otherwise doc by doc.
fn each_i64(c: &rustic_core::codec::fastfields::ColumnReader<'_, i64>, v: &[(DocId, f32)], mut f: impl FnMut(DocId, f32, i64)) {
    const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
    let span = match (v.first(), v.last()) { (Some(a), Some(b)) => (b.0 - a.0) as usize + 1, _ => 0 };
    if !c.all_present() || v.len() < 4096 || v.len() * 8 < span {
        for &(d, sc) in v {
            if let Some(x) = c.get(d) { f(d, sc, x); }
        }
        return;
    }
    let mut buf = [0i64; BS];
    let (mut cur, mut n) = (usize::MAX, 0usize);
    for &(d, sc) in v {
        let b = d as usize / BS;
        if b != cur {
            match c.block_values(b, &mut buf) {
                Some(m) => (cur, n) = (b, m),
                None => {
                    if let Some(x) = c.get(d) { f(d, sc, x); }
                    continue;
                }
            }
        }
        let i = d as usize % BS;
        if i < n { f(d, sc, buf[i]); }
    }
}

/// A single-valued keyword column with its segment's decoded terms when
/// the term table is front-coded and small
/// ([`Segment::keyword_terms_decoded`](rustic_core::segment::Segment::keyword_terms_decoded)):
/// [`Col::keys`] then reads a doc's term by ordinal instead of replaying
/// its front-coded block (big5's `terms` + `significant_terms` over
/// `process.name` and `log_stream` decoded a term per matching doc).
struct KwCol<'a> {
    reader: rustic_core::codec::fastfields::KeywordReader<'a>,
    decoded: Option<std::sync::Arc<rustic_core::admission::Allocation<Vec<Box<str>>>>>,
}

impl<'a> std::ops::Deref for KwCol<'a> {
    type Target = rustic_core::codec::fastfields::KeywordReader<'a>;
    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl KwCol<'_> {
    /// [`KeywordReader::get`](rustic_core::codec::fastfields::KeywordReader::get), by ordinal from the decoded terms when held
    /// (not `term`: that name is the reader's, by ordinal, which callers
    /// reach through the deref).
    /// The term of ordinal `ord` (decoded table when held).
    fn ord_term(&self, ord: u32) -> Option<std::borrow::Cow<'_, str>> {
        match self.decoded.as_ref().and_then(|terms| terms.get(ord as usize)) {
            Some(t) => Some(std::borrow::Cow::Borrowed(&**t)),
            None => self.reader.term(ord),
        }
    }

    fn doc_term(&self, doc: DocId) -> Option<std::borrow::Cow<'_, str>> {
        match &self.decoded {
            Some(terms) => {
                let ord = self.reader.ord(doc);
                if ord == rustic_core::codec::fastfields::MISSING_ORD {
                    return None;
                }
                match terms.get(ord as usize) {
                    Some(t) => Some(std::borrow::Cow::Borrowed(&**t)),
                    None => self.reader.term(ord),
                }
            }
            None => self.reader.get(doc),
        }
    }
}

enum Col<'a> {
    Kw(KwCol<'a>),
    MultiKw(rustic_core::codec::fastfields::MultiKeywordReader<'a>),
    I64(rustic_core::codec::fastfields::ColumnReader<'a, i64>),
    MultiI64(rustic_core::codec::fastfields::MultiColumnReader<'a, i64>),
    U64(rustic_core::codec::fastfields::ColumnReader<'a, u64>),
    MultiU64(rustic_core::codec::fastfields::MultiColumnReader<'a, u64>),
    F64(rustic_core::codec::fastfields::ColumnReader<'a, f64>),
    /// With a `scaled_float`'s factor and its inverse (`None`: values as
    /// stored), applied to each value read ([`es_scaled`]).
    MultiF64(rustic_core::codec::fastfields::MultiColumnReader<'a, f64>, Option<(f64, f64)>),
    /// No column (the field has no values in this segment).
    Empty,
}

impl<'a> Col<'a> {
    /// Open the column a [`LeafReader`] per-doc accessor of type `ty` reads
    /// (same single-then-multi lookup order). `None`: not a column type.
    fn open(seg: &'a rustic_core::segment::Segment, id: u16, ty: EsType, scaling_factor: f64) -> Option<Col<'a>> {
        // A scaled_float's values as ES aggregates them: the scaled long
        // times the inverse factor ([`es_scaled`]), not the stored quotient.
        let scale = (ty == EsType::ScaledFloat && scaling_factor.is_finite() && scaling_factor > 0.0).then(|| (scaling_factor, 1.0 / scaling_factor));
        let ff = seg.fast_fields();
        Some(match ty {
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened | EsType::Ip => match seg.keyword_reader(id) {
                Some(c) => Col::Kw(KwCol { reader: c, decoded: seg.keyword_terms_decoded(id).ok().flatten() }),
                None => seg.multi_keyword_reader(id).map(Col::MultiKw).unwrap_or(Col::Empty),
            },
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Date | EsType::DateNanos => match ff.get_i64(id) {
                Some(c) => Col::I64(c),
                None => ff.get_multi_i64(id).map(Col::MultiI64).unwrap_or(Col::Empty),
            },
            EsType::UnsignedLong => match ff.get_u64(id) {
                Some(c) => Col::U64(c),
                None => ff.get_multi_u64(id).map(Col::MultiU64).unwrap_or(Col::Empty),
            },
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => match ff.get_f64(id) {
                Some(c) => Col::F64(match scale { Some((f, _)) => c.with_es_scaled(f), None => c }),
                None => ff.get_multi_f64(id).map(|c| Col::MultiF64(c, scale)).unwrap_or(Col::Empty),
            },
            _ => return None,
        })
    }

    /// Same keys as the `LeafReader` path of [`Ctx::values`].
    fn keys(&self, ty: EsType, doc: DocId, mixed: bool) -> Vec<Key> {
        let kw = |s: &str| if ty == EsType::Ip { values::term_to_ip(s).map(Key::Ip).unwrap_or_else(|| Key::Str(s.to_string())) } else { Key::Str(s.to_string()) };
        let u = |v: u64| if mixed { Key::Dbl(v as f64) } else { Key::ULong(v) };
        match self {
            Col::Kw(c) => c.doc_term(doc).map(|v| kw(&v)).into_iter().collect(),
            Col::MultiKw(c) => c.values(doc).into_iter().map(kw).collect(),
            Col::I64(c) => c.get(doc).map(Key::Long).into_iter().collect(),
            Col::MultiI64(c) => c.values_vec(doc).into_iter().map(Key::Long).collect(),
            Col::U64(c) => c.get(doc).map(u).into_iter().collect(),
            Col::MultiU64(c) => c.values_vec(doc).into_iter().map(u).collect(),
            Col::F64(c) => c.get(doc).map(Key::Dbl).into_iter().collect(),
            Col::MultiF64(c, scale) => c.values_vec(doc).into_iter().map(|v| Key::Dbl(es_scaled(v, *scale))).collect(),
            Col::Empty => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Keys and values.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum Key {
    Null,
    Bool(bool),
    Long(i64),
    ULong(u64),
    Dbl(f64),
    Str(String),
    /// IP address (16-byte v4-mapped form); sorts by address order.
    Ip([u8; 16]),
}

impl PartialEq for Key {
    fn eq(&self, o: &Key) -> bool {
        cmp_key(self, o) == Ordering::Equal
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self {
            Key::Null => 0u8.hash(h),
            Key::Bool(b) => b.hash(h),
            Key::Long(i) => (*i as f64).to_bits().hash(h),
            Key::ULong(u) => u.hash(h),
            Key::Dbl(f) => f.to_bits().hash(h),
            Key::Str(s) => s.hash(h),
            Key::Ip(b) => b.hash(h),
        }
    }
}

impl Key {
    /// Heap bytes beyond the key itself.
    fn heap_bytes(&self) -> usize {
        match self {
            Key::Str(s) => s.capacity(),
            _ => 0,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Key::Long(i) => Some(*i as f64),
            Key::ULong(u) => Some(*u as f64),
            Key::Dbl(f) => Some(*f),
            Key::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    fn hash_bytes(&self) -> Vec<u8> {
        match self {
            Key::Null => vec![],
            Key::Bool(b) => vec![*b as u8],
            Key::Long(i) => i.to_le_bytes().to_vec(),
            Key::ULong(u) => { let mut b = vec![0u8]; b.extend_from_slice(&u.to_le_bytes()); b }
            Key::Dbl(f) => f.to_bits().to_le_bytes().to_vec(),
            Key::Str(s) => s.as_bytes().to_vec(),
            Key::Ip(b) => b.to_vec(),
        }
    }
}

fn cmp_key(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Null, Key::Null) => Ordering::Equal,
        (Key::Null, _) => Ordering::Less,
        (_, Key::Null) => Ordering::Greater,
        (Key::Str(x), Key::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
        (Key::Str(_), _) => Ordering::Greater,
        (_, Key::Str(_)) => Ordering::Less,
        (Key::Ip(x), Key::Ip(y)) => x.cmp(y),
        (Key::Ip(_), _) => Ordering::Greater,
        (_, Key::Ip(_)) => Ordering::Less,
        (Key::Long(x), Key::Long(y)) => x.cmp(y),
        (Key::ULong(x), Key::ULong(y)) => x.cmp(y),
        (Key::Long(x), Key::ULong(y)) => (*x as u128).cmp(&(*y as u128)),
        (Key::ULong(x), Key::Long(y)) => (*x as u128).cmp(&(*y as u128)),
        _ => a.as_f64().unwrap_or(0.0).total_cmp(&b.as_f64().unwrap_or(0.0)),
    }
}

/// Whether `x` lies in `lo..=hi` as a `double` range query (the points
/// index's total order) matches it: NaN never does.
#[inline]
fn f64_in_range(x: f64, lo: f64, hi: f64) -> bool {
    !x.is_nan() && lo.total_cmp(&x) != std::cmp::Ordering::Greater && x.total_cmp(&hi) != std::cmp::Ordering::Greater
}

/// A non-date `avg` / `sum` / `min` / `max` / `stats` result of `n` values
/// with sum `sum` (in collection order), minimum `min` and maximum `max`.
/// A stored `scaled_float` value (`long / factor`) as ES's aggregations
/// read it: `round(v * factor) * (1 / factor)` (`ScaledFloatLeafFieldData`
/// multiplies the scaled long by the inverse factor), e.g. 887.3 stored
/// reads as 887.3000000000001. `None`: not a scaled_float, `v` as is.
pub(crate) fn es_scaled(v: f64, scale: Option<(f64, f64)>) -> f64 {
    use rustic_core::codec::fastfields::FastValue;
    match scale {
        Some((f, inv)) => v.es_scaled(f, inv),
        None => v,
    }
}

fn simple_metric_json(t: &str, n: usize, sum: f64, min: f64, max: f64) -> Value {
    let single = |v: Option<f64>| json!({ "value": opt_num(v) });
    match t {
        "avg" => single((n > 0).then(|| sum / n as f64)),
        "sum" => json!({ "value": fnum(sum) }),
        "min" => single((n > 0).then_some(min)),
        "max" => single((n > 0).then_some(max)),
        _ => {
            let mut m = Map::new();
            m.insert("count".into(), json!(n));
            m.insert("min".into(), opt_num((n > 0).then_some(min)));
            m.insert("max".into(), opt_num((n > 0).then_some(max)));
            m.insert("avg".into(), opt_num((n > 0).then(|| sum / n as f64)));
            m.insert("sum".into(), fnum(sum));
            Value::Object(m)
        }
    }
}

/// Java `Double.toString`.
fn jfmt(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let a = v.abs();
    if a == 0.0 || (1e-3..1e7).contains(&a) {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
        format!("{m}E{e}")
    }
}

fn fnum(v: f64) -> Value {
    values::java_double(v)
}

fn opt_num(v: Option<f64>) -> Value {
    v.map(fnum).unwrap_or(Value::Null)
}

/// One aggregated field's resolution across all segments.
#[derive(Clone)]
struct FieldRef {
    per_seg: Vec<Option<Arc<ResolvedField>>>,
    ty: Option<EsType>,
    format: Option<DateFormatter>,
    /// Script / runtime-field value source (instead of doc values).
    src: Option<std::rc::Rc<ScriptSrc>>,
    /// The field is mapped as different numeric types across indices
    /// (e.g. `long` + `unsigned_long`): values merge as doubles.
    mixed: bool,
}

/// Values computed by a Painless script (`script`, value script over a
/// `field` as `_value`, or a runtime field).
struct ScriptSrc {
    script: Option<crate::script::CompiledScript>,
    runtime: Option<String>,
    base: Option<FieldRef>,
    /// Value source per segment index (bound executors).
    per_seg: std::cell::RefCell<Vec<Option<crate::script::search::ValueSource>>>,
}

fn val_key(v: &crate::script::Val) -> Option<Key> {
    use crate::script::Val;
    Some(match v {
        Val::Null => return None,
        Val::Bool(b) => Key::Bool(*b),
        Val::Int(i) => Key::Long(*i as i64),
        Val::Long(l) => Key::Long(*l),
        Val::Double(d) => Key::Dbl(*d),
        Val::Date(m) | Val::Instant(m) => Key::Long(*m),
        Val::Str(s) => Key::Str(s.to_string()),
        other => Key::Str(other.to_java_string()),
    })
}

fn key_val(k: &Key) -> crate::script::Val {
    use crate::script::Val;
    match k {
        Key::Null => Val::Null,
        Key::Bool(b) => Val::Bool(*b),
        Key::Long(l) => Val::Long(*l),
        Key::ULong(u) => Val::Long(*u as i64),
        Key::Dbl(d) => Val::Double(*d),
        Key::Str(s) => Val::str(s),
        Key::Ip(b) => Val::str(&values::ip_display(&values::ip_to_term(b))),
    }
}

impl<'a, 'b> Ctx<'a, 'b> {
    fn field(&self, name: &str) -> Result<FieldRef> {
        // Reserved `long` metadata fields (`_seq_no`, `_primary_term`) are
        // not in the mapping lookup; synthesize one resolved field for
        // every segment.
        if let Some(id) = crate::mapping::reserved_long_field_id(name) {
            let rf = Arc::new(crate::mapping::reserved_long_field(name, id));
            let per_seg: Vec<Option<Arc<ResolvedField>>> = self.segs.iter().map(|_| Some(rf.clone())).collect();
            return Ok(FieldRef { ty: Some(EsType::Long), format: Some(rf.date_format.clone()), per_seg, src: None, mixed: false });
        }
        // `_index` (ES `IndexFieldMapper`): not in the mapping lookup; each
        // segment's documents hold their concrete index name as a constant.
        // Unresolved, a `terms` on `_index` over several indices answered no
        // buckets at all.
        if name == "_index" {
            let mut by_index: Vec<(&str, Arc<ResolvedField>)> = Vec::new();
            let per_seg: Vec<Option<Arc<ResolvedField>>> = self
                .segs
                .iter()
                .map(|s| {
                    if let Some((_, rf)) = by_index.iter().find(|(i, _)| *i == s.index) {
                        return Some(rf.clone());
                    }
                    let rf = Arc::new(crate::mapping::index_name_field(s.index));
                    by_index.push((s.index, rf.clone()));
                    Some(rf)
                })
                .collect();
            return Ok(FieldRef { ty: Some(EsType::ConstantKeyword), format: None, per_seg, src: None, mixed: false });
        }
        let per_seg: Vec<Option<Arc<ResolvedField>>> = self.segs.iter().map(|s| s.lookup.get(name).cloned()).collect();
        let first = per_seg.iter().flatten().next().cloned();
        if let Some(f) = &first {
            if f.declared == EsType::PatternText {
                return Err(EsError::illegal_argument("pattern_text fields do not support sorting and aggregations"));
            }
            if f.ty.is_text() && !f.fielddata {
                return Err(EsError::illegal_argument(format!(
                    "Fielddata is disabled on [{name}] in [{}]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [{name}] in order to load field data by uninverting the inverted index. Note that this can use significant memory.",
                    self.segs.first().map(|s| s.index).unwrap_or("")
                )));
            }
            if matches!(f.ty, EsType::GeoPoint | EsType::DenseVector | EsType::Binary) {
                return Err(EsError::illegal_argument(format!("Field [{name}] of type [{}] is not supported for aggregation", f.ty.name())));
            }
            if f.ty.is_text() {
                for s in self.segs.iter() {
                    crate::fielddata_cache::mark(s.index, name);
                }
            }
        }
        let mixed = per_seg.iter().flatten().any(|f| f.ty != first.as_ref().map(|f| f.ty).unwrap_or(EsType::Keyword));
        Ok(FieldRef { ty: first.as_ref().map(|f| f.ty), format: first.as_ref().map(|f| f.date_format.clone()), per_seg, src: None, mixed })
    }

    /// The value source of an aggregation body: `script` (optionally over
    /// `field` as `_value`), a runtime field, or a mapped field.
    fn source(&self, o: &Map<String, Value>, name: &str, t: &str) -> Result<FieldRef> {
        let field = o.get("field").and_then(Value::as_str);
        if let Some(s) = o.get("script") {
            let cs = crate::script::compile_search_script(s, crate::script::ScriptContext::Aggs)?;
            let base = match field {
                Some(f) => Some(self.source_field(f)?),
                None => None,
            };
            let hint = match o.get("value_type").and_then(Value::as_str) {
                Some("date") => Some(EsType::Date),
                Some("long" | "number" | "numeric") => Some(EsType::Long),
                Some("double") => Some(EsType::Double),
                Some("string" | "keyword") => Some(EsType::Keyword),
                Some("boolean") => Some(EsType::Boolean),
                _ => None,
            };
            let ty = hint.or(base.as_ref().and_then(|b| b.ty));
            let format = match ty {
                Some(EsType::Date) => Some(base.as_ref().and_then(|b| b.format.clone()).unwrap_or_default()),
                _ => None,
            };
            let src = ScriptSrc { script: Some(cs), runtime: None, base, per_seg: Default::default() };
            return Ok(FieldRef { per_seg: vec![None; self.segs.len()], ty, format, src: Some(std::rc::Rc::new(src)), mixed: false });
        }
        let fname = field_name(o, name, t)?;
        self.source_field(fname).map_err(|mut e| {
            // ES names the aggregation (and a geo_point `position` metric).
            if e.reason.ends_with("is not supported for aggregation") {
                let position = self.segs.iter().find_map(|s| s.lookup.get(fname)).is_some_and(|f| f.time_series_metric.as_deref() == Some("position"));
                if position {
                    e.reason = e.reason.replace("of type [geo_point]", "of type [geo_point][position]");
                }
                e.reason = format!("{} [{t}]", e.reason);
            }
            e
        })
    }

    fn source_field(&self, name: &str) -> Result<FieldRef> {
        for seg in self.segs {
            if let Some((rt, i)) = crate::script::runtime_field(Some(seg.index), name) {
                let f = &rt.fields[i];
                if f.ty == crate::script::RtType::Lookup {
                    return Err(EsError::illegal_argument(format!("Fielddata is not supported on field [{name}] of type [lookup]")));
                }
                let ty = match f.ty {
                    crate::script::RtType::Long => EsType::Long,
                    crate::script::RtType::Double => EsType::Double,
                    crate::script::RtType::Keyword => EsType::Keyword,
                    crate::script::RtType::Boolean => EsType::Boolean,
                    crate::script::RtType::Date => EsType::Date,
                    crate::script::RtType::Ip => EsType::Ip,
                    crate::script::RtType::GeoShape => EsType::GeoShape,
                    crate::script::RtType::Lookup => unreachable!("rejected above"),
                };
                let src = ScriptSrc { script: None, runtime: Some(name.to_string()), base: None, per_seg: Default::default() };
                return Ok(FieldRef { per_seg: vec![None; self.segs.len()], ty: Some(ty), format: Some(f.format.clone()), src: Some(std::rc::Rc::new(src)), mixed: false });
            }
        }
        self.field(name)
    }

    fn script_values(&self, src: &ScriptSrc, seg: usize, doc: DocId) -> Vec<Key> {
        let s = &self.segs[seg];
        let mut map = src.per_seg.borrow_mut();
        if map.len() <= seg {
            map.resize_with(seg + 1, || None);
        }
        if map[seg].is_none() {
            let vs = match (&src.script, &src.runtime) {
                (Some(cs), _) => crate::script::search::ValueSource::script(cs.clone(), s.index, s.lookup),
                (None, Some(n)) => match crate::script::search::ValueSource::runtime(n, s.index) {
                    Some(v) => v,
                    None => return Vec::new(),
                },
                _ => return Vec::new(),
            };
            map[seg] = Some(vs);
        }
        let vs = map[seg].as_mut().unwrap();
        // `_score` of the doc in the search (matching docs are ascending).
        let score = if vs.uses_score() { s.docs.binary_search(&doc).ok().and_then(|i| s.scores.get(i)).copied().unwrap_or(0.0) } else { 0.0 };
        let r = match &src.base {
            Some(b) => {
                let mut out = Vec::new();
                let mut err = None;
                for k in self.values(b, seg, doc) {
                    match vs.values(seg, &s.leaf, doc, score, Some(key_val(&k))) {
                        Ok(v) => out.extend(v.iter().filter_map(val_key)),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                match err {
                    Some(e) => Err(e),
                    None => Ok(out),
                }
            }
            None => vs.with_values(seg, &s.leaf, doc, score, |vals| vals.iter().filter_map(val_key).collect()),
        };
        match r {
            Ok(keys) => keys,
            Err(e) => {
                if let Some(env) = crate::script::env() {
                    crate::script::report_error(&env.error, e);
                }
                Vec::new()
            }
        }
    }

    fn values(&self, f: &FieldRef, seg: usize, doc: DocId) -> Vec<Key> {
        let v = self.values_stored(f, seg, doc);
        // `_tsid` doc values are kept in an order-preserving encoding.
        if f.src.is_none() && f.per_seg.get(seg).and_then(Option::as_ref).is_some_and(|rf| rf.name == crate::tsdb::TSID_FIELD) {
            return v.into_iter().map(|k| if let Key::Str(s) = k { Key::Str(crate::tsdb::tsid_display(&s)) } else { k }).collect();
        }
        v
    }

    fn values_stored(&self, f: &FieldRef, seg: usize, doc: DocId) -> Vec<Key> {
        if let Some(src) = &f.src {
            return self.script_values(src, seg, doc);
        }
        let Some(rf) = &f.per_seg[seg] else { return Vec::new() };
        let leaf = &self.segs[seg].leaf;
        if let Some(v) = crate::fieldtypes::aggs::values(leaf, rf, doc) {
            return v
                .into_iter()
                .map(|v| match v {
                    crate::fieldtypes::aggs::AggValue::Str(s) => Key::Str(s),
                    crate::fieldtypes::aggs::AggValue::Long(i) => Key::Long(i),
                    crate::fieldtypes::aggs::AggValue::Dbl(d) => Key::Dbl(d),
                })
                .collect();
        }
        let Some(id) = rf.core_id else { return Vec::new() };
        if rf.ty.is_text() && rf.fielddata {
            // No doc-values column: the uninverted terms (ES fielddata).
            if self.memory_error.borrow().is_some(){return Vec::new();}
            let mut fields=self.fielddata.borrow_mut();
            let fd=match fields.entry((seg,id)) {
                std::collections::hash_map::Entry::Occupied(entry)=>entry.get().clone(),
                std::collections::hash_map::Entry::Vacant(entry)=>match leaf.uninverted(id){
                    Ok(value)=>entry.insert(value).clone(),
                    Err(error)=>{*self.memory_error.borrow_mut()=Some(error.into());return Vec::new();}
                },
            };
            return fd.ords(doc).iter().map(|&o| Key::Str(fd.term_string(o))).collect();
        }
        if let Some(c) = self.col(f, seg) {
            return c.keys(rf.ty, doc, f.mixed);
        }
        match rf.ty {
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened => leaf.keyword_values(id, doc).into_iter().map(Key::Str).collect(),
            EsType::Ip => leaf.keyword_values(id, doc).into_iter().map(|t| values::term_to_ip(&t).map(Key::Ip).unwrap_or_else(|| Key::Str(t))).collect(),
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Date | EsType::DateNanos => leaf.i64_values(id, doc).into_iter().map(Key::Long).collect(),
            EsType::UnsignedLong => leaf.u64_values(id, doc).into_iter().map(|v| if f.mixed { Key::Dbl(v as f64) } else { Key::ULong(v) }).collect(),
            EsType::Double | EsType::Float | EsType::HalfFloat => leaf.f64_values(id, doc).into_iter().map(Key::Dbl).collect(),
            EsType::ScaledFloat => {
                let scale = (rf.scaling_factor.is_finite() && rf.scaling_factor > 0.0).then(|| (rf.scaling_factor, 1.0 / rf.scaling_factor));
                leaf.f64_values(id, doc).into_iter().map(|v| Key::Dbl(es_scaled(v, scale))).collect()
            }
            EsType::Boolean => leaf.bool_value(id, doc).map(Key::Bool).into_iter().collect(),
            _ => Vec::new(),
        }
    }

    /// The single-valued keyword column [`Self::values`] reads `f` from in
    /// sealed segment `seg` (no script, flattened key, constant, `_tsid`
    /// or fielddata reading), for grouping and counting by ordinal: one
    /// key per distinct term instead of one per doc.
    fn ord_col(&self, f: &FieldRef, seg: usize) -> Option<std::rc::Rc<Col<'a>>> {
        if f.src.is_some() {
            return None;
        }
        let rf = f.per_seg.get(seg)?.as_ref()?;
        if rf.flattened_key().is_some() || rf.name == crate::tsdb::TSID_FIELD || (rf.ty.is_text() && rf.fielddata)
            || matches!(rf.declared, EsType::ConstantKeyword | EsType::DateNanos | EsType::AggregateMetricDouble)
            || !matches!(rf.ty, EsType::Keyword | EsType::Wildcard | EsType::Ip) {
            return None;
        }
        let c = self.col(f, seg)?;
        matches!(&*c, Col::Kw(_)).then_some(c)
    }

    /// The key [`Col::keys`] gives a keyword term of a field of type `ty`.
    fn term_key(ty: EsType, term: &str) -> Key {
        if ty == EsType::Ip { values::term_to_ip(term).map(Key::Ip).unwrap_or_else(|| Key::Str(term.to_string())) } else { Key::Str(term.to_string()) }
    }

    /// The docs of `docs` with a value of `f` (or `missing`), counted per
    /// key: [`Self::group_by`]'s groups' sizes without their doc lists.
    /// Single-valued keyword columns count by ordinal.
    pub(super) fn count_by(&self, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<HashMap<Key, u64>> {
        let mut counts: HashMap<Key, u64> = HashMap::new();
        // Segments counted by ordinal, on the search pool when there are
        // many docs: per segment its (ordinal, count) pairs and the docs
        // without a value; terms are looked up once per ordinal after.
        let mut ord_segs: Vec<(usize, std::rc::Rc<Col<'a>>)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            if let Some(c) = self.ord_col(f, s) {
                ord_segs.push((s, c));
                continue;
            }
            for &(d, _) in v {
                let mut ks = self.values_or(f, s, d, missing);
                ks.sort_by(cmp_key);
                ks.dedup();
                for k in ks { *counts.entry(k).or_default() += 1; }
            }
        }
        // Slices of each segment's docs (a large segment folds in parts).
        let slice = (rustic_core::query::scorer::part_split_min_docs() as usize / 2).max(1);
        let mut work: Vec<(usize, &KwCol<'a>, usize, usize)> = Vec::new();
        for (s, c) in &ord_segs {
            let Col::Kw(kc) = &**c else { unreachable!() };
            let n = docs[*s].len();
            let mut i0 = 0;
            while i0 < n { let i1 = (i0 + slice).min(n); work.push((*s, kc, i0, i1)); i0 = i1; }
        }
        let count_seg = |&(s, kc, i0, i1): &(usize, &KwCol<'a>, usize, usize)| -> (Vec<(u32, u64)>, u64) {
            let v = &docs[s][i0..i1];
            let n = kc.num_terms();
            let mut none = 0u64;
            if n <= v.len().saturating_mul(4).max(1 << 16) {
                let mut by_ord = vec![0u64; n];
                for &(d, _) in v {
                    match by_ord.get_mut(kc.ord(d) as usize) { Some(x) => *x += 1, None => none += 1 }
                }
                (by_ord.into_iter().enumerate().filter(|(_, k)| *k > 0).map(|(o, k)| (o as u32, k)).collect(), none)
            } else {
                let mut by_ord: HashMap<u32, u64> = HashMap::new();
                for &(d, _) in v {
                    let o = kc.ord(d);
                    if o == rustic_core::codec::fastfields::MISSING_ORD { none += 1 } else { *by_ord.entry(o).or_default() += 1 }
                }
                (by_ord.into_iter().collect(), none)
            }
        };
        let total: usize = work.iter().map(|w| w.3 - w.2).sum();
        let parts: Vec<(Vec<(u32, u64)>, u64)> = if total >= 2 * slice && work.len() > 1 {
            rustic_core::query::searcher::par_map_search(work.len(), |i| count_seg(&work[i]))
        } else {
            work.iter().map(count_seg).collect()
        };
        for ((s, kc, _, _), (ords, none)) in work.iter().zip(parts) {
            let ty = f.per_seg[*s].as_ref().map(|rf| rf.ty).unwrap_or(EsType::Keyword);
            for (o, k) in ords {
                if let Some(t) = kc.ord_term(o) { *counts.entry(Self::term_key(ty, &t)).or_default() += k; }
            }
            if none > 0 {
                if let Some(m) = missing { *counts.entry(m.clone()).or_default() += none; }
            }
        }
        Ok(counts)
    }

    /// [`Self::count_by`] of the background set of a `significant_terms`
    /// and its size (`key` names its field and background filter), once
    /// per request.
    pub(super) fn background_counts(&self, key: String, count: impl FnOnce() -> Result<(HashMap<Key, u64>, u64)>) -> Result<std::rc::Rc<(HashMap<Key, u64>, u64)>> {
        if let Some(c) = self.bg_counts.borrow().get(&key) {
            return Ok(c.clone());
        }
        let c = std::rc::Rc::new(count()?);
        self.bg_counts.borrow_mut().insert(key, c.clone());
        Ok(c)
    }

    /// The column of a mapped field in sealed segment `seg` (cached).
    fn col(&self, f: &FieldRef, seg: usize) -> Option<std::rc::Rc<Col<'a>>> {
        let rf = f.per_seg[seg].as_ref()?;
        let id = rf.core_id?;
        let LeafReader::Segment(sg) = &self.segs[seg].leaf else { return None };
        if let Some(c) = self.cols.borrow().get(&(seg, id)) {
            return Some(c.clone());
        }
        let c = std::rc::Rc::new(Col::open(sg, id, rf.ty, rf.scaling_factor)?);
        self.cols.borrow_mut().insert((seg, id), c.clone());
        Some(c)
    }

    /// Every numeric value of `f` in `docs` (`missing` for docs without
    /// one), as [`Self::values_or`] reads them; single-valued numeric
    /// columns are read once per segment with no per-document lookup.
    fn reserve_sketch(&self, bytes: u64, reserved: &mut u64) -> Result<()> {
        if bytes > *reserved {
            self.mem.charge(bytes - *reserved, "<agg [numeric sketch]>")?;
            *reserved = bytes;
        }
        Ok(())
    }

    /// Count, compensated sum, min and max of `f`'s values in `docs`, folded
    /// as [`Self::num_values`] lists them (same values, same order, so the
    /// same sum), without the list: single-valued numeric columns are read a
    /// block of docs at a time. `None` when a segment with docs has no such
    /// column (the caller lists the values).
    fn fold_simple(&self, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<Option<(usize, f64, f64, f64)>> {
        const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
        let missing_num = missing.as_ref().and_then(Key::as_f64);
        let mut cols = Vec::with_capacity(docs.len());
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                cols.push(None);
                continue;
            }
            match self.num_col(f, s) {
                Some(c) => cols.push(Some(c)),
                None => return Ok(None),
            }
        }
        let (mut n, mut sum, mut min, mut max) = (0usize, extra::Kahan::default(), f64::INFINITY, f64::NEG_INFINITY);
        let mut add = |x: f64| {
            n += 1;
            sum.add(x);
            min = min.min(x);
            max = max.max(x);
        };
        fn fold_col<T: rustic_core::codec::fastfields::FastValue>(c: &rustic_core::codec::fastfields::ColumnReader<'_, T>, v: &[(DocId, f32)], missing: Option<f64>, as_f64: impl Fn(T) -> f64, add: &mut impl FnMut(f64)) {
            let full = c.all_present();
            let mut buf = vec![T::zero(); BS];
            let mut cur = usize::MAX;
            // Blocks only pay for dense lists (a bucket's docs scattered
            // over the segment would decode a block per doc).
            let span = match (v.first(), v.last()) { (Some(a), Some(b)) => (b.0 - a.0) as usize + 1, _ => 0 };
            let mut blocked = v.len() * 8 >= span;
            for &(d, _) in v {
                let x = if blocked {
                    let b = d as usize / BS;
                    if b != cur {
                        match c.block_values(b, &mut buf) {
                            Some(_) => cur = b,
                            None => blocked = false,
                        }
                    }
                    if blocked {
                        if full || c.has_value(d) { Some(as_f64(buf[d as usize % BS])) } else { None }
                    } else {
                        c.get(d).map(&as_f64)
                    }
                } else {
                    c.get(d).map(&as_f64)
                };
                match x.or(missing) {
                    Some(x) => add(x),
                    None => {}
                }
            }
        }
        if cols.iter().flatten().any(|c| !matches!(&**c, Col::I64(_) | Col::F64(_) | Col::U64(_))) {
            return Ok(None);
        }
        let fold_part = |c: &Col<'a>, v: &[(DocId, f32)], add: &mut dyn FnMut(f64)| {
            let mut add = |x: f64| add(x);
            match c {
                Col::I64(c) => fold_col(c, v, missing_num, |x| x as f64, &mut add),
                Col::F64(c) => fold_col(c, v, missing_num, |x| x, &mut add),
                Col::U64(c) => fold_col(c, v, missing_num, |x| x as f64, &mut add),
                _ => unreachable!("checked above"),
            }
        };
        // Many docs: slices of each segment's docs fold on the search pool,
        // each into its own count / compensated sum / min / max, combined
        // in doc order (compensated sums combine as across segments).
        let part_docs = (rustic_core::query::scorer::part_split_min_docs() as usize / 2).max(1);
        let total: usize = docs.iter().map(Vec::len).sum();
        let mut work: Vec<(usize, usize, usize)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if cols[s].is_none() { continue; }
            let mut i0 = 0;
            while i0 < v.len() { let i1 = (i0 + part_docs).min(v.len()); work.push((s, i0, i1)); i0 = i1; }
        }
        if total >= 2 * part_docs && work.len() > 1 {
            let refs: Vec<Option<&Col<'a>>> = cols.iter().map(|c| c.as_deref()).collect();
            let parts = rustic_core::query::searcher::par_map_search(work.len(), |i| {
                let (s, i0, i1) = work[i];
                let (mut n, mut sum, mut min, mut max) = (0usize, extra::Kahan::default(), f64::INFINITY, f64::NEG_INFINITY);
                fold_part(refs[s].expect("cut from a column"), &docs[s][i0..i1], &mut |x: f64| { n += 1; sum.add(x); min = min.min(x); max = max.max(x); });
                (n, sum, min, max)
            });
            for (pn, psum, pmin, pmax) in parts {
                n += pn;
                sum.merge(&psum);
                min = min.min(pmin);
                max = max.max(pmax);
            }
            return Ok(Some((n, sum.value, min, max)));
        }
        for (s, v) in docs.iter().enumerate() {
            let Some(c) = &cols[s] else { continue };
            fold_part(c, v, &mut add);
        }
        Ok(Some((n, sum.value, min, max)))
    }

    fn num_values(&self, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<Vec<f64>> {
        let mut vals: Vec<f64> = Vec::new();
        let mut push = |value: f64| -> Result<()> {
            if vals.len() == vals.capacity() {
                let additional = vals.capacity().max(16);
                self.mem.charge((additional as u64).saturating_mul(8), "<agg [numeric values]>")?;
                vals.try_reserve_exact(additional).map_err(|_| EsError::new(429, "circuit_breaking_exception", "numeric aggregation allocation failed"))?;
            }
            vals.push(value);
            Ok(())
        };
        let missing_num = missing.as_ref().and_then(Key::as_f64);
        for (s, v) in docs.iter().enumerate() {
            if let Some(col) = self.num_col(f, s) {
                macro_rules! fill {
                    ($c:expr) => {
                        for &(d, _) in v {
                            match $c.get(d) {
                                Some(x) => push(x as f64)?,
                                None => if let Some(x) = missing_num { push(x)?; },
                            }
                        }
                    };
                }
                match &*col {
                    Col::I64(c) => fill!(c),
                    Col::F64(c) => fill!(c),
                    Col::U64(c) => fill!(c),
                    _ => unreachable!(),
                }
                continue;
            }
            for &(d, _) in v {
                for value in self.values_or(f, s, d, missing).iter().filter_map(Key::as_f64) {
                    push(value)?;
                }
            }
        }
        Ok(vals)
    }

    /// Add the HLL++ hash of every value of `f` in `docs` (`missing` for docs
    /// without one): what [`Self::values_or`] + [`Key::hash_bytes`] hash.
    /// Single-valued keyword columns hash the stored term bytes directly, with
    /// no string per document.
    fn add_value_hashes(&self, f: &FieldRef, docs: &Docs, missing: &Option<Key>, hll: &mut HyperLogLog) {
        let missing_hash = missing.as_ref().map(|k| sketch::murmur3_64(&k.hash_bytes()));
        for (s, v) in docs.iter().enumerate() {
            if let Some(c) = self.kw_col(f, s) {
                if let Col::Kw(r) = &*c {
                    // Many docs over few terms: mark the ordinals seen and
                    // hash each term once (a register keeps the largest
                    // rank of a hash: adding it again changes nothing).
                    let terms = r.num_terms();
                    if v.len() >= terms / 8 && terms > 0 {
                        let mut seen = vec![0u64; terms.div_ceil(64)];
                        let mut any_missing = false;
                        for &(d, _) in v {
                            let o = r.ord(d);
                            if o == rustic_core::codec::fastfields::MISSING_ORD {
                                any_missing = true;
                            } else if let Some(w) = seen.get_mut(o as usize / 64) {
                                *w |= 1 << (o % 64);
                            }
                        }
                        if let (true, Some(h)) = (any_missing, missing_hash) {
                            hll.add_hash(h);
                        }
                        for (w, &word) in seen.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let o = (w * 64) as u32 + bits.trailing_zeros();
                                bits &= bits - 1;
                                if let Some(b) = r.term_bytes(o) {
                                    hll.add_hash(sketch::murmur3_64(&b));
                                }
                            }
                        }
                        continue;
                    }
                    for &(d, _) in v {
                        match r.ord(d) {
                            rustic_core::codec::fastfields::MISSING_ORD => {
                                if let Some(h) = missing_hash {
                                    hll.add_hash(h);
                                }
                            }
                            o => {
                                if let Some(b) = r.term_bytes(o) {
                                    hll.add_hash(sketch::murmur3_64(&b));
                                }
                            }
                        }
                    }
                    continue;
                }
            }
            for &(d, _) in v {
                for k in self.values_or(f, s, d, missing) {
                    hll.add_bytes(&k.hash_bytes());
                }
            }
        }
    }

    /// The single-valued keyword column of `f` on sealed segment `seg`, when
    /// [`Self::values`] would read exactly its terms (as [`Self::num_col`]).
    fn kw_col(&self, f: &FieldRef, seg: usize) -> Option<std::rc::Rc<Col<'a>>> {
        if f.src.is_some() {
            return None;
        }
        let rf = f.per_seg.get(seg)?.as_ref()?;
        if !matches!(rf.ty, EsType::Keyword | EsType::Wildcard) || rf.flattened_key().is_some() || matches!(rf.declared, EsType::ConstantKeyword) || rf.name == crate::tsdb::TSID_FIELD {
            return None;
        }
        let c = self.col(f, seg)?;
        matches!(&*c, Col::Kw(_)).then_some(c)
    }

    /// Count-only `date_histogram` buckets of a doc set that is every doc of
    /// sealed segment `seg` (`n` docs; a doc set is ascending and unique, so
    /// `n == num_docs` means all of them): each bucket `[key, next)` is one
    /// range count on the column's points index, with no per-doc read
    /// (ES 9.5.5 adapts such a top-level histogram into filters counted
    /// by `Weight#count` on the points tree; from knowledge of
    /// `DateHistogramAggregator`/`FiltersAggregator`, not re-read here).
    /// Adds into `counts` and returns true; false (nothing added) when the
    /// shortcut does not apply, for the per-doc paths to count instead.
    fn count_whole_segment_by_points(&self, c: &rustic_core::codec::fastfields::ColumnReader<'a, i64>, seg: usize, docs: &[(DocId, f32)], rounding: &Rounding, counts: &mut HashMap<i64, u64, ahash::RandomState>) -> bool {
        let n = docs.len();
        let with_value = if n == c.num_docs() as usize && n == self.segs[seg].leaf.num_docs() as usize {
            if c.is_sparse() { c.points_len() as u64 } else { n as u64 }
        } else {
            // Or the docs hold every doc with a value: a `nested` agg's
            // children under a `match_all` (the field lives on them only).
            // The index holds exactly the docs with a value (deleted ones
            // too, which the docs never hold).
            if !c.counts_from_points() {
                return false;
            }
            let present = c.value_doc_count();
            if (n as u64) < present || docs.iter().filter(|&&(d, _)| c.has_value(d)).count() as u64 != present {
                return false;
            }
            present
        };
        Self::count_by_points(c, rounding, with_value, counts)
    }

    /// Count-only buckets of uncollected segment `seg` (its matches are the
    /// docs whose value of `f` lies in the query's range, no deletions):
    /// each bucket is one range count on the points index, clipped to the
    /// query's range. False (nothing added) when that does not apply.
    fn count_elided_by_points(&self, f: &FieldRef, seg: usize, rounding: &Rounding, counts: &mut HashMap<i64, u64, ahash::RandomState>) -> bool {
        let Some(&Some((field, lo, hi))) = self.query_ranges.get(seg) else { return false };
        if f.per_seg.get(seg).and_then(|x| x.as_ref()).and_then(|rf| rf.core_id) != Some(field) || self.segs[seg].leaf.deleted_bits().is_some() {
            return false;
        }
        let Some(col) = self.num_col(f, seg) else { return false };
        let Col::I64(c) = &*col else { return false };
        if !c.has_points() || !c.counts_from_points() {
            return false;
        }
        let Some((a, b)) = c.points_min_max_in(lo, hi) else { return true };
        let Some(prepared) = PreparedDateBuckets::new(rounding, a, b) else { return false };
        let mut per_bucket = Vec::with_capacity(prepared.keys.len() - 1);
        let mut total = 0u64;
        for w in prepared.keys.windows(2) {
            let k = c.count_range(w[0].max(lo), (w[1] - 1).min(hi));
            total += k;
            per_bucket.push(k);
        }
        if total != c.count_range(lo, hi) {
            return false;
        }
        for (i, k) in per_bucket.into_iter().enumerate() {
            if k != 0 {
                *counts.entry(prepared.keys[i]).or_default() += k;
            }
        }
        true
    }

    /// Count-only buckets of every doc of uncollected segment `seg` (a
    /// `match_all`'s roots, no deletions, no query range) from the column's
    /// per-block min / max: a block whose min and max round to one bucket
    /// adds its doc count without a per-doc read (rounding is monotonic),
    /// others decode their values. What ES's doc-values skipper gives a
    /// TSDB `@timestamp` without points (docs sorted by `_tsid`, then time,
    /// so a block mostly lies in one bucket). False (nothing added) when
    /// that does not apply.
    fn count_whole_by_blocks(&self, f: &FieldRef, seg: usize, rounding: &Rounding, counts: &mut HashMap<i64, u64, ahash::RandomState>) -> bool {
        if self.query_ranges.get(seg).is_some_and(Option::is_some) || self.query_f64_ranges.get(seg).is_some_and(Option::is_some) {
            return false;
        }
        let leaf = &self.segs[seg].leaf;
        if leaf.deleted_bits().is_some() {
            return false;
        }
        let Some(col) = self.num_col(f, seg) else { return false };
        let Col::I64(c) = &*col else { return false };
        let n = leaf.num_docs() as usize;
        if c.num_docs() as usize != n {
            return false;
        }
        const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
        let mut memo = Vec::new();
        let mut buf = [0i64; BS];
        // Consecutive blocks mostly share a bucket: one map update per run.
        let mut run: Option<(i64, u64)> = None;
        let mut add = |k: i64, by: u64, counts: &mut HashMap<i64, u64, ahash::RandomState>| match &mut run {
            Some((rk, rn)) if *rk == k => *rn += by,
            _ => {
                if let Some((rk, rn)) = run.replace((k, by)) {
                    *counts.entry(rk).or_default() += rn;
                }
            }
        };
        for b in 0..n.div_ceil(BS) {
            let (some, missing) = c.block_presence(b);
            if !some {
                continue;
            }
            let (start, end) = (b * BS, ((b + 1) * BS).min(n));
            let (mn, mx) = c.block_min_max(b);
            let k = rounding.round_memo(&mut memo, mn);
            if !missing && rounding.round_memo(&mut memo, mx) == k {
                add(k, (end - start) as u64, counts);
                continue;
            }
            match c.block_values(b, &mut buf) {
                Some(m) if !missing && m == end - start => {
                    for &x in &buf[..m] {
                        add(rounding.round_memo(&mut memo, x), 1, counts);
                    }
                }
                _ => {
                    for d in start..end {
                        if let Some(x) = c.get(d as DocId) {
                            add(rounding.round_memo(&mut memo, x), 1, counts);
                        }
                    }
                }
            }
        }
        if let Some((rk, rn)) = run {
            *counts.entry(rk).or_default() += rn;
        }
        true
    }

    /// The buckets of every doc holding a value of `c` (`with_value` of
    /// them) counted from its points index; false (nothing added) when the
    /// index does not cover exactly those docs.
    fn count_by_points(c: &rustic_core::codec::fastfields::ColumnReader<'a, i64>, rounding: &Rounding, with_value: u64, counts: &mut HashMap<i64, u64, ahash::RandomState>) -> bool {
        let Some((lo, hi)) = c.points_min_max() else { return false };
        let Some(prepared) = PreparedDateBuckets::new(rounding, lo, hi) else { return false };
        let mut per_bucket = Vec::with_capacity(prepared.keys.len() - 1);
        let mut total = 0u64;
        for w in prepared.keys.windows(2) {
            let k = c.count_range(w[0], w[1] - 1);
            total += k;
            per_bucket.push(k);
        }
        // Every doc with a value lies in exactly one bucket: anything else
        // means the index does not cover the doc set, so count per doc.
        if total != with_value {
            return false;
        }
        for (i, k) in per_bucket.into_iter().enumerate() {
            if k != 0 {
                *counts.entry(prepared.keys[i]).or_default() += k;
            }
        }
        #[cfg(test)]
        POINTS_COUNTED_SEGMENTS.with(|c| c.set(c.get() + 1));
        true
    }

    /// Every doc of nested path ordinal `ord` on segment `seg` (score 0):
    /// the doc list a `nested` aggregation over all live roots of a segment
    /// without deletes would have built (see [`NESTED_WHOLE_PATH`]).
    fn nested_path_docs(&self, seg: usize, ord: u16) -> Vec<(DocId, f32)> {
        let leaf = &self.segs[seg].leaf;
        let Some(n) = leaf.nested_info() else { return Vec::new() };
        (0..leaf.num_docs()).filter(|&d| n.ord(d) == ord).map(|d| (d, 0.0)).collect()
    }

    /// Count-only buckets of `f` over every doc of nested `path` on segment
    /// `seg` (no deletes), from the points index: valid when `f` lives on
    /// that path's docs only (not copied to parents or the root), so its
    /// docs with a value are all in the set. False when it does not apply.
    fn count_nested_path_by_points(&self, f: &FieldRef, seg: usize, path: &str, rounding: &Rounding, counts: &mut HashMap<i64, u64, ahash::RandomState>) -> bool {
        let Some(rf) = f.per_seg.get(seg).and_then(|x| x.as_ref()) else { return false };
        if rf.nested_path.as_deref() != Some(path) {
            return false;
        }
        match self.segs[seg].lookup.objects.get(path) {
            Some(o) if o.nested && !o.include_in_parent && !o.include_in_root => {}
            _ => return false,
        }
        let Some(col) = self.num_col(f, seg) else { return false };
        let Col::I64(c) = &*col else { return false };
        if !c.counts_from_points() {
            return false;
        }
        Self::count_by_points(c, rounding, c.value_doc_count(), counts)
    }

    /// The single-valued numeric column of `f` on sealed segment `seg`,
    /// when [`Self::values`] would read exactly that column (no script,
    /// no type-specific value conversion): per-doc reads then need no
    /// lookup and no allocation.
    fn num_col(&self, f: &FieldRef, seg: usize) -> Option<std::rc::Rc<Col<'a>>> {
        if f.src.is_some() {
            return None;
        }
        let rf = f.per_seg.get(seg)?.as_ref()?;
        if rf.flattened_key().is_some() || matches!(rf.declared, EsType::ConstantKeyword | EsType::DateNanos | EsType::AggregateMetricDouble) || rf.name == crate::tsdb::TSID_FIELD {
            return None;
        }
        let c = self.col(f, seg)?;
        matches!(&*c, Col::I64(_) | Col::F64(_) | Col::U64(_)).then_some(c)
    }

    /// Values with the `missing` substitution applied.
    fn values_or(&self, f: &FieldRef, seg: usize, doc: DocId, missing: &Option<Key>) -> Vec<Key> {
        let v = self.values(f, seg, doc);
        if v.is_empty() {
            if let Some(m) = missing {
                return vec![m.clone()];
            }
        }
        v
    }

    fn missing_key(&self, f: &FieldRef, v: Option<&Value>) -> Result<Option<Key>> {
        let Some(v) = v else { return Ok(None) };
        if v.is_null() {
            return Ok(None);
        }
        Ok(Some(match f.ty {
            Some(t) if t.is_date() => {
                let fmt = f.format.clone().unwrap_or_default();
                Key::Long(match v {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0) as i64,
                    Value::String(s) => parse_date_math(s, &fmt, Tz::UTC, self.env.now, false)?,
                    _ => 0,
                })
            }
            Some(t) if t.is_integer() => match v {
                Value::Number(n) if n.is_i64() => Key::Long(n.as_i64().unwrap()),
                Value::Number(n) if n.is_u64() => Key::ULong(n.as_u64().unwrap()),
                Value::Number(n) => Key::Dbl(n.as_f64().unwrap_or(0.0)),
                Value::String(s) => s.parse::<i64>().map(Key::Long).or_else(|_| s.parse::<f64>().map(Key::Dbl)).map_err(|_| EsError::illegal_argument(format!("For input string: \"{s}\"")))?,
                _ => Key::Null,
            },
            Some(t) if t.is_float() => Key::Dbl(match v {
                Value::Number(n) => n.as_f64().unwrap_or(0.0),
                Value::String(s) => s.parse().map_err(|_| EsError::illegal_argument(format!("For input string: \"{s}\"")))?,
                _ => 0.0,
            }),
            Some(EsType::Boolean) => Key::Bool(matches!(v, Value::Bool(true)) || v.as_str() == Some("true")),
            _ => Key::Str(match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            }),
        }))
    }

    /// Whether `docs` is the query's matches and segment `s`'s list was
    /// left uncollected ([`with_elided_root`]).
    fn is_elided(&self, docs: &Docs, s: usize) -> bool {
        self.elided.get(s).copied().unwrap_or(false) && std::ptr::eq(docs as *const Docs as *const u8, self.root_docs as *const u8)
    }

    /// `docs` with every uncollected segment's list read back: the docs
    /// with a value of `f` in the query's range (what the range matched).
    /// A copy, so not taken for the query's own doc lists again.
    fn collect_elided(&self, f: &FieldRef, docs: &Docs) -> Docs {
        let mut out = docs.clone();
        for (s, list) in out.iter_mut().enumerate() {
            if !self.is_elided(docs, s) {
                continue;
            }
            if let Some(&Some((id, lo, hi))) = self.query_f64_ranges.get(s) {
                // The range matched stored values (a scaled_float's
                // quotients, not the doc values aggregations read).
                let leaf = &self.segs[s].leaf;
                for d in 0..leaf.num_docs() {
                    if leaf.f64_values(id, d).into_iter().any(|x| f64_in_range(x, lo, hi)) {
                        list.push((d, 1.0));
                    }
                }
                continue;
            }
            let Some(&Some((_, lo, hi))) = self.query_ranges.get(s) else {
                // No range: a `match_all`'s roots, every doc of a segment
                // without deletions (`range_auto_date_fields`).
                list.extend((0..self.segs[s].leaf.num_docs()).map(|d| (d, 1.0)));
                continue;
            };
            // A single-valued long column: blocks outside the range are
            // skipped and blocks inside it taken whole, by their min / max.
            if let Some(col) = self.num_col(f, s) {
                if let Col::I64(c) = &*col {
                    const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
                    let n = self.segs[s].leaf.num_docs() as usize;
                    for b in 0..n.div_ceil(BS) {
                        let (some, missing) = c.block_presence(b);
                        if !some {
                            continue;
                        }
                        let (mn, mx) = c.block_min_max(b);
                        if mx < lo || mn > hi {
                            continue;
                        }
                        let (start, end) = (b * BS, ((b + 1) * BS).min(n));
                        if !missing && mn >= lo && mx <= hi {
                            list.extend((start..end).map(|d| (d as DocId, 1.0)));
                            continue;
                        }
                        for d in start..end {
                            if c.get(d as DocId).is_some_and(|x| (lo..=hi).contains(&x)) {
                                list.push((d as DocId, 1.0));
                            }
                        }
                    }
                    continue;
                }
            }
            for d in 0..self.segs[s].leaf.num_docs() {
                if self.values(f, s, d).iter().any(|k| matches!(k, Key::Long(x) if (lo..=hi).contains(x))) {
                    list.push((d, 1.0));
                }
            }
        }
        out
    }

    fn add_buckets(&self, n: usize) -> Result<()> {
        let total = self.buckets.get() + n;
        self.buckets.set(total);
        if total > self.env.max_buckets {
            return Err(EsError::too_many_buckets(self.env.max_buckets));
        }
        self.mem.add(n as u64 * BUCKET_BYTES, "allocated_buckets")?;
        self.mem.buckets(n as u64)
    }

    /// Charge `bytes` of aggregation `name`'s state to the request breaker.
    #[inline]
    fn charge(&self, name: &str, bytes: u64) -> Result<()> {
        self.mem.add_with(bytes, || agg_label(name))
    }

    /// A new grouping bucket of `name` (its key, `key_bytes` on the heap,
    /// with doc lists over `nsegs` segments): charged, and counted toward
    /// the breakers' periodic real-memory check.
    #[inline]
    fn charge_group(&self, name: &str, key_bytes: usize, nsegs: usize) -> Result<()> {
        self.charge(name, GROUP_BYTES + key_bytes as u64 + 24 * nsegs as u64)?;
        self.mem.buckets(1)
    }

    /// Add doc `(d, sc)` of segment `s` to bucket `key` of `groups`,
    /// creating (and charging) the bucket on first use.
    #[inline]
    fn push_grouped<K: Hash + Eq>(&self, name: &str, groups: &mut HashMap<K, Docs>, key: K, key_bytes: usize, nsegs: usize, s: usize, doc: (DocId, f32)) -> Result<()> {
        let docs = match groups.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                self.charge_group(name, key_bytes, nsegs)?;
                e.insert(vec![Vec::new(); nsegs])
            }
        };
        docs[s].push(doc);
        self.charge(name, 8)
    }

    /// Charge the growth of a grouping map (`groups` buckets now, `seen`
    /// at the last call) and `docs` doc entries about to be added.
    fn charge_grouping(&self, name: &str, groups: usize, seen: &mut usize, nsegs: usize, docs: usize) -> Result<()> {
        let new = groups.saturating_sub(*seen);
        *seen = groups;
        self.charge(name, new as u64 * (GROUP_BYTES + 24 * nsegs as u64) + docs as u64 * 8)?;
        self.mem.buckets(new as u64)
    }

    // -----------------------------------------------------------------------
    // Dispatcher.

    fn compute(&self, spec: &Map<String, Value>, docs: &Docs) -> Result<Map<String, Value>> {
        let mut results: HashMap<String, Value> = HashMap::new();
        // Borrowed from `spec` (this runs once per bucket of every parent).
        let mut parsed: Vec<(&String, &String, &Value, Option<&Map<String, Value>>, Option<&Value>)> = Vec::new();
        for (name, def) in spec {
            let def = def.as_object().ok_or_else(|| EsError::parsing_at(format!("Expected [START_OBJECT] under [{name}], but got a [VALUE_STRING] in [{name}]"), 1, 1))?;
            let mut ty: Option<(&String, &Value)> = None;
            let mut sub = None;
            let mut meta = None;
            for (k, v) in def {
                match k.as_str() {
                    "aggs" | "aggregations" => {
                        sub = Some(v.as_object().ok_or_else(|| EsError::parsing_at(format!("Expected [START_OBJECT] under [{k}], but got a [VALUE_STRING] in [{name}]"), 1, 1))?);
                    }
                    "meta" => meta = Some(v),
                    _ => {
                        if let Some((t0, _)) = ty {
                            return Err(EsError::parsing_at(format!("Found two aggregation type definitions in [{name}]: [{t0}] and [{k}]"), 1, 1));
                        }
                        ty = Some((k, v));
                    }
                }
            }
            let (t, body) = ty.ok_or_else(|| EsError::parsing_at(format!("Missing definition for aggregation [{name}]"), 1, 1))?;
            if !KNOWN.contains(&t.as_str()) && !extra::EXTRA.contains(&t.as_str()) && !geo_nested::GEO_NESTED.contains(&t.as_str()) && !crate::fieldtypes::aggs::AGGS.contains(&t.as_str()) && !analytics::AGGS.contains(&t.as_str()) && !PARENT_PIPELINES.contains(&t.as_str()) && !SIBLING_PIPELINES.contains(&t.as_str()) {
                return Err(EsError::unknown_named("Unknown aggregation type", t, Some(name), strict::AGG_NAMES));
            }
            if (SIBLING_PIPELINES.contains(&t.as_str()) || PARENT_PIPELINES.contains(&t.as_str())) && sub.is_some() {
                return Err(EsError::illegal_argument(format!("Aggregator [{name}] of type [{t}] cannot accept sub-aggregations")));
            }
            parsed.push((name, t, body, sub, meta));
        }
        for &(name, t, body, sub, _) in &parsed {
            if PARENT_PIPELINES.contains(&t.as_str()) || SIBLING_PIPELINES.contains(&t.as_str()) {
                continue;
            }
            // Per aggregation and bucket (never per doc): a cancelled
            // search task stops building buckets.
            crate::search::check_cancelled()?;
            let r = self.compute_one(name, t, body, sub, docs)?;
            results.insert(name.clone(), r);
        }
        // Only these pipelines read their siblings' definitions.
        if parsed.iter().any(|p| matches!(p.1.as_str(), "cumulative_cardinality" | "moving_percentiles")) {
            let entries: Vec<(String, String, Value, bool)> = parsed.iter().map(|p| (p.0.clone(), p.1.clone(), p.2.clone(), p.3.is_some())).collect();
            self.analytics_prepare(&entries, &mut results, docs)?;
        }
        for &(name, t, body, _, _) in &parsed {
            if SIBLING_PIPELINES.contains(&t.as_str()) && !partial::skip_pipeline() {
                let r = sibling_pipeline(name, t, body, &results)?;
                results.insert(name.clone(), r);
            }
        }
        let mut out = Map::new();
        for (name, t, _, _, meta) in parsed {
            if PARENT_PIPELINES.contains(&t.as_str()) {
                continue;
            }
            let mut v = results.remove(name).unwrap_or(Value::Null);
            if let (Some(m), Value::Object(o)) = (meta, &mut v) {
                let mut with_meta = Map::new();
                with_meta.insert("meta".into(), m.clone());
                for (k, x) in std::mem::take(o) {
                    with_meta.insert(k, x);
                }
                *o = with_meta;
            }
            out.insert(name.clone(), v);
        }
        Ok(out)
    }

    /// One aggregation. Its working state (grouping maps, doc lists,
    /// sketches) is charged as it is built and given back when it returns:
    /// only its rendered result stays charged, so the request breaker holds
    /// the live peak (the result tree plus the working state of the
    /// aggregations running now), never the sum over every bucket's
    /// sub-aggregations.
    fn compute_one(&self, name: &str, t: &str, body: &Value, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let mark = self.mem.level();
        let v = self.compute_one_inner(name, t, body, sub, docs)?;
        self.mem.release_to(mark);
        self.charge(name, crate::memory::json_bytes(&v))?;
        Ok(v)
    }

    fn compute_one_inner(&self, name: &str, t: &str, body: &Value, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let o = body.as_object().ok_or_else(|| EsError::parsing_at(format!("[{t}] aggregation [{name}] must be an object"), 1, 1))?;
        let _ts = geo_line::TimeSeriesScope::enter(&self.in_time_series, t == "time_series");
        // A coordinator reduce follows (cluster forwarding): sketch-based
        // aggregations also carry their mergeable state (`__p`).
        if partial::active() || (crate::search::aggregation_shard_response() && partial::SHARD_STATE_AGGS.contains(&t)) {
            if let Some(v) = self.partial_one(name, t, o, sub, docs)? {
                return Ok(v);
            }
        }
        let _enclosing = analytics::AggGuard::push(t);
        if let Some(v) = self.analytics_override(name, t, o, sub, docs)? {
            return Ok(v);
        }
        {
            let sub_fn = |d: &Docs| -> Result<Map<String, Value>> { self.subs(sub, d) };
            let host = crate::fieldtypes::aggs::Host { segs: self.segs, env: self.env, all_docs: self.env.all_docs, sub: &sub_fn };
            if let Some(v) = crate::fieldtypes::aggs::compute(name, t, o, docs, &host)? {
                return Ok(v);
            }
        }
        match t {
            "terms" => self.terms(name, o, sub, docs),
            "range" | "date_range" => self.range(name, o, sub, docs, t == "date_range"),
            "histogram" => self.histogram(name, o, sub, docs),
            "date_histogram" => self.date_histogram(name, o, sub, docs),
            "filter" => {
                let d = self.filter_docs(body, docs)?;
                self.single_bucket(d, sub)
            }
            "filters" => self.filters(name, o, sub, docs),
            "missing" => {
                let f = self.source(o, name, t)?;
                let d: Docs = docs.iter().enumerate().map(|(s, v)| v.iter().filter(|(doc, _)| self.values(&f, s, *doc).is_empty()).copied().collect()).collect();
                self.single_bucket(d, sub)
            }
            "global" => {
                let d: Docs = (0..self.segs.len()).map(|s| (self.env.all_docs)(s).into_iter().map(|d| (d, 0.0)).collect()).collect();
                self.single_bucket(d, sub)
            }
            "sampler" => self.sampler(name, o, sub, docs),
            "diversified_sampler" => self.diversified_sampler(name, o, sub, docs),
            "composite" => self.composite(name, o, sub, docs),
            "top_hits" => self.top_hits(body, docs),
            "scripted_metric" => self.scripted_metric(name, o, docs),
            "significant_text" => self.significant_text(name, o, sub, docs),
            t if extra::EXTRA.contains(&t) => self.extra_agg(name, t, o, sub, docs),
            t if geo_nested::GEO_NESTED.contains(&t) => self.geo_nested_agg(name, t, o, sub, docs),
            t if analytics::AGGS.contains(&t) => self.analytics_agg(name, t, o, sub, docs),
            _ => self.metric(name, t, o, docs),
        }
    }

    fn single_bucket(&self, d: Docs, sub: Option<&Map<String, Value>>) -> Result<Value> {
        let mut m = Map::new();
        m.insert("doc_count".into(), json!(count(&d)));
        if let Some(sub) = sub {
            m.extend(self.compute(sub, &d)?);
        }
        Ok(Value::Object(m))
    }

    fn subs(&self, sub: Option<&Map<String, Value>>, d: &Docs) -> Result<Map<String, Value>> {
        match sub {
            Some(s) => self.compute(s, d),
            None => Ok(Map::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Sampler.

    /// `sampler`: the best-scoring `shard_size` (default 100) docs of each
    /// shard (index), then the sub-aggregations over them.
    fn sampler(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        if sub.is_none() {
            return Err(EsError::illegal_argument("Sampler aggregation must be used with child aggregations."));
        }
        let shard_size = o.get("shard_size").and_then(Value::as_u64).unwrap_or(100) as usize;
        let _ = name;
        self.single_bucket(self.sample(name, docs, shard_size, None)?, sub)
    }

    /// `diversified_sampler`: like `sampler`, but at most `max_docs_per_value`
    /// (default 1) docs per distinct value of `field`.
    fn diversified_sampler(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        if sub.is_none() {
            return Err(EsError::illegal_argument("Sampler aggregation must be used with child aggregations."));
        }
        let f = self.source(o, name, "diversified_sampler")?;
        let shard_size = o.get("shard_size").and_then(Value::as_u64).unwrap_or(100) as usize;
        let per_value = o.get("max_docs_per_value").and_then(Value::as_u64).unwrap_or(1) as usize;
        self.single_bucket(self.sample(name, docs, shard_size, Some((&f, per_value)))?, sub)
    }

    /// Per index: best `limit` docs by score (ties: lower doc first), with an
    /// optional per-value cap. Each segment's list stays doc-ascending.
    ///
    /// Charged to the request breaker: the selection's working state (the
    /// bounded heap, or the diversified path's sorted copy of the index's
    /// docs and its per-value counts), given back once each index is
    /// sampled, and the sampled doc lists, held until the aggregation
    /// returns ([`Self::compute_one`]).
    fn sample(&self, name: &str, docs: &Docs, limit: usize, diversify: Option<(&FieldRef, usize)>) -> Result<Docs> {
        let mut out: Docs = vec![Vec::new(); docs.len()];
        let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            match groups.iter_mut().find(|(n, _)| *n == s.index) {
                Some((_, v)) => v.push(i),
                None => groups.push((s.index, vec![i])),
            }
        }
        // `(segment, doc, score)` entries of the heap and the sorted copy,
        // `(doc, score)` ones of the sampled lists.
        const ENTRY: u64 = std::mem::size_of::<(usize, DocId, f32)>() as u64;
        const SAMPLED: u64 = std::mem::size_of::<(DocId, f32)>() as u64;
        for (_, idxs) in groups {
            let total: usize = idxs.iter().map(|&s| docs[s].len()).sum();
            if diversify.is_none() {
                let working = total.min(limit) as u64 * ENTRY;
                self.charge(name, working)?;
                let top = sample_top(docs, &idxs, limit);
                self.charge(name, top.len() as u64 * SAMPLED)?;
                for (s, d, sc) in top { out[s].push((d, sc)); }
                for &s in &idxs { out[s].sort_by_key(|x| x.0); }
                self.mem.release(working);
                continue;
            }
            let mut working = total as u64 * ENTRY;
            self.charge(name, working)?;
            let mut all: Vec<(usize, DocId, f32)> = idxs.iter().flat_map(|&s| docs[s].iter().map(move |&(d, sc)| (s, d, sc))).collect();
            all.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
            let mut seen: HashMap<Key, usize> = HashMap::new();
            let mut taken = 0;
            for (s, d, sc) in all {
                if taken >= limit {
                    break;
                }
                if let Some((f, cap)) = diversify {
                    let k = self.values(f, s, d).into_iter().next().unwrap_or(Key::Null);
                    let c = match seen.entry(k) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            let bytes = GROUP_BYTES + e.key().heap_bytes() as u64;
                            self.charge(name, bytes)?;
                            working += bytes;
                            e.insert(0)
                        }
                    };
                    if *c >= cap {
                        continue;
                    }
                    *c += 1;
                }
                self.charge(name, SAMPLED)?;
                taken += 1;
                out[s].push((d, sc));
            }
            for &s in &idxs {
                out[s].sort_by_key(|x| x.0);
            }
            self.mem.release(working);
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Filters.

    /// `filter_docs` of a query that is one `exists` on a field read from
    /// doc values in every segment with docs: the docs whose column holds a
    /// value (no query, search context or scorer per bucket: a filter
    /// under a terms under a date_histogram runs once per bucket).
    /// `None`: not such a field (the query runs).
    fn filter_exists_by_column(&self, field: &str, docs: &Docs) -> Result<Option<Docs>> {
        if self.segs.iter().any(|s| crate::script::runtime_field(Some(s.index), field).is_some()) {
            return Ok(None);
        }
        let Ok(f) = self.source_field(field) else { return Ok(None) };
        if f.src.is_some() {
            return Ok(None);
        }
        let mut out: Docs = vec![Vec::new(); docs.len()];
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            // Unmapped in this index: nothing exists.
            let Some(rf) = f.per_seg[s].as_ref() else { continue };
            let plain = matches!(rf.declared, EsType::Keyword | EsType::Ip | EsType::Long | EsType::Integer | EsType::Short | EsType::Byte
                | EsType::UnsignedLong | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::Date);
            if !plain || !rf.doc_values || rf.flattened_key().is_some() || rf.name == crate::tsdb::TSID_FIELD {
                return Ok(None);
            }
            let Some(col) = self.col(&f, s) else { return Ok(None) };
            let has = |d: DocId| match &*col {
                Col::Kw(c) => c.get(d).is_some(),
                Col::MultiKw(c) => c.ords(d).next().is_some(),
                Col::I64(c) => c.get(d).is_some(),
                Col::MultiI64(c) => c.values(d).next().is_some(),
                Col::U64(c) => c.get(d).is_some(),
                Col::MultiU64(c) => c.values(d).next().is_some(),
                Col::F64(c) => c.get(d).is_some(),
                Col::MultiF64(c, _) => c.values(d).next().is_some(),
                Col::Empty => false,
            };
            let keep: Vec<(DocId, f32)> = v.iter().copied().filter(|&(d, _)| has(d)).collect();
            self.mem.charge((keep.len() * std::mem::size_of::<(DocId, f32)>()) as u64, "<agg [filter docs]>")?;
            out[s] = keep;
        }
        Ok(Some(out))
    }

    fn filter_docs(&self, query: &Value, docs: &Docs) -> Result<Docs> {
        if let Some(field) = exists_field(query) {
            if let Some(out) = self.filter_exists_by_column(field, docs)? {
                return Ok(out);
            }
        }
        let mut out: Docs = vec![Vec::new(); docs.len()];
        let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            match groups.iter_mut().find(|(n, _)| *n == s.index) {
                Some((_, v)) => v.push(i),
                None => groups.push((s.index, vec![i])),
            }
        }
        for (index, idxs) in groups {
            if idxs.iter().all(|i| docs[*i].is_empty()) {
                continue;
            }
            let leaves: Vec<LeafReader> = idxs.iter().map(|i| self.segs[*i].leaf).collect();
            let lookup = self.segs[idxs[0]].lookup;
            let qctx = QueryContext {
                deadline: None,
                lookup,
                index_name: index,
                index_uuid: "",
                aliases: Vec::new(),
                now: self.env.now,
                leaves: &leaves,
                default_fields: vec!["*".into()],
                terms_lookup: None,
                max_terms_count: 65536,
                max_regex_length: self.segs[idxs[0]].max_regex_length,
                max_result_window: self.env.max_result_window,
                seq_no_disabled: false,
                allow_unmapped_fields: true,
                knn_prefilter: None,
                knn_prefilter_lookup: None,
                allow_expensive_queries: true,
                mappings: None,
            };
            let q = parse_query(query, &qctx)?;
            let sctx = SearchContext::for_query(&q, &leaves, Bm25Params::default());
            sctx.execution_result()?;
            for (li, si) in idxs.iter().enumerate() {
                if docs[*si].is_empty() {
                    continue;
                }
                let Some(mut sc) = q.scorer(&leaves[li], &sctx) else { continue };
                // A bucket much smaller than its segment (a filter under a
                // terms under a date_histogram: hundreds of small buckets)
                // seeks the filter to each of its docs instead of listing
                // every match of the segment once per bucket (tsdb_k8s_queries'
                // tx_network_usage_per_pod: 20 s, each bucket walking all
                // 8.6M docs of an `exists` filter).
                let bucket = &docs[*si];
                if (bucket.len() as u64).saturating_mul(8) < leaves[li].num_docs() as u64 && bucket.windows(2).all(|w| w[0].0 <= w[1].0) {
                    let mut keep: Vec<(DocId, f32)> = Vec::new();
                    let mut cur: Option<DocId> = None;
                    for &(d, score) in bucket {
                        let at = match cur {
                            Some(c) if c >= d => c,
                            _ => sc.seek(d),
                        };
                        cur = Some(at);
                        if at == TERMINATED {
                            break;
                        }
                        if at == d {
                            if keep.len() == keep.capacity() {
                                let additional = keep.capacity().max(16);
                                self.mem.charge((additional * std::mem::size_of::<(DocId, f32)>()) as u64, "<agg [filter docs]>")?;
                                keep.try_reserve_exact(additional).map_err(|_| EsError::new(429, "circuit_breaking_exception", "filter allocation failed"))?;
                            }
                            keep.push((d, score));
                        }
                    }
                    out[*si] = keep;
                    continue;
                }
                let mut matched = Vec::new();
                loop {
                    let d = sc.advance();
                    if d == TERMINATED {
                        break;
                    }
                    if matched.len() == matched.capacity() {
                        let additional = matched.capacity().max(16);
                        self.mem.charge((additional * std::mem::size_of::<DocId>()) as u64, "<agg [filter matches]>")?;
                        matched.try_reserve_exact(additional).map_err(|_| EsError::new(429, "circuit_breaking_exception", "filter allocation failed"))?;
                    }
                    matched.push(d);
                }
                let count = docs[*si].iter().filter(|(d, _)| matched.binary_search(d).is_ok()).count();
                self.mem.charge((count * std::mem::size_of::<(DocId, f32)>()) as u64, "<agg [filter docs]>")?;
                out[*si] = Vec::with_capacity(count);
                out[*si].extend(docs[*si].iter().filter(|(d, _)| matched.binary_search(d).is_ok()).copied());
                self.mem.release((matched.capacity() * std::mem::size_of::<DocId>()) as u64);
            }
            sctx.execution_result()?;
        }
        Ok(out)
    }

    fn filters(&self, _name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let fs = o.get("filters").ok_or_else(|| EsError::parsing_at("[filters] cannot be empty.", 1, 1))?;
        let other_key = o.get("other_bucket_key").and_then(Value::as_str).map(str::to_string);
        let other = o.get("other_bucket").and_then(Value::as_bool).unwrap_or(other_key.is_some());
        let other_key = other_key.unwrap_or_else(|| "_other_".into());
        let named: Vec<(Option<String>, Value)> = match fs {
            Value::Object(m) => {
                let mut v: Vec<(Option<String>, Value)> = m.iter().map(|(k, q)| (Some(k.clone()), q.clone())).collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            }
            Value::Array(a) => a.iter().map(|q| (None, q.clone())).collect(),
            _ => return Err(EsError::parsing_at("[filters] must be an object or an array", 1, 1)),
        };
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(matches!(fs, Value::Object(_)));
        self.add_buckets(named.len() + other as usize)?;
        let mut matched_any: Vec<HashSet<DocId>> = if other {
            self.mem.charge((docs.len() * std::mem::size_of::<HashSet<DocId>>()) as u64, "<agg [filters other bucket]>")?;
            vec![HashSet::new(); docs.len()]
        } else { Vec::new() };
        let mut buckets: Vec<(String, Value)> = Vec::new();
        for (i, (k, q)) in named.iter().enumerate() {
            let d = self.filter_docs(q, docs)?;
            if other {
                for (s, v) in d.iter().enumerate() {
                    for &(doc, _) in v {
                        if !matched_any[s].contains(&doc) {
                            // Hash table slots/control bytes and growth slack, before insert.
                            self.mem.add(32, "<agg [filters other bucket]>")?;
                            matched_any[s].insert(doc);
                        }
                    }
                }
            }
            let doc_bytes = d.iter().map(|v| v.len() * std::mem::size_of::<(DocId, f32)>()).sum::<usize>() as u64;
            let bucket = self.single_bucket(d, sub)?;
            self.mem.release(doc_bytes);
            buckets.push((k.clone().unwrap_or_else(|| i.to_string()), bucket));
        }
        if other {
            let count = docs.iter().enumerate().map(|(s, v)| v.iter().filter(|(d, _)| !matched_any[s].contains(d)).count()).sum::<usize>();
            self.mem.charge((count * std::mem::size_of::<(DocId, f32)>()) as u64, "<agg [filter other docs]>")?;
            let d: Docs = docs.iter().enumerate().map(|(s, v)| {
                let count = v.iter().filter(|(d, _)| !matched_any[s].contains(d)).count();
                let mut out = Vec::with_capacity(count);
                out.extend(v.iter().filter(|(d, _)| !matched_any[s].contains(d)).copied());
                out
            }).collect();
            let bucket = self.single_bucket(d, sub)?;
            self.mem.release((count * std::mem::size_of::<(DocId, f32)>()) as u64);
            buckets.push((other_key, bucket));
        }
        let b = if keyed {
            Value::Object(buckets.into_iter().collect())
        } else {
            Value::Array(
                buckets
                    .into_iter()
                    .map(|(k, mut v)| {
                        if matches!(fs, Value::Object(_)) {
                            let mut m = Map::new();
                            m.insert("key".into(), json!(k));
                            m.extend(v.as_object_mut().map(std::mem::take).unwrap_or_default());
                            Value::Object(m)
                        } else {
                            v
                        }
                    })
                    .collect(),
            )
        };
        Ok(json!({ "buckets": b }))
    }

    // -----------------------------------------------------------------------
    // terms.

    fn terms(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.source(o, name, "terms")?;
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size <= 0 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0. Found [{size}] in [{name}]")));
        }
        let size = size as usize;
        let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(1);
        let missing = self.missing_key(&f, o.get("missing"))?;
        let include = Matcher::parse(o.get("include"))?;
        let exclude = Matcher::parse(o.get("exclude"))?;
        let order = parse_order(o.get("order"), &[("_count", false)])?;
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone());
        let show_err = o.get("show_term_doc_count_error").and_then(Value::as_bool).unwrap_or(false);

        let one_pass = if missing.is_none() && include.is_none() && exclude.is_none() {
            self.terms_one_pass(name, &f, sub, docs, &order, size, min_doc_count)?
        } else { None };
        let (total, bs) = match one_pass {
            Some(done) => done,
            None => self.terms_grouped(name, o, &f, sub, docs, &order, size, min_doc_count, &missing, &include, &exclude)?,
        };
        self.terms_output(&f, o, sub, fmt, show_err, total, bs)
    }

    /// [`Ctx::terms`]'s general collector: doc lists per term, then each
    /// bucket's sub-aggregations over its docs. Returns the total doc
    /// count of every bucket and the kept buckets, ranked.
    #[allow(clippy::too_many_arguments)]
    fn terms_grouped(&self, name: &str, o: &Map<String, Value>, f: &FieldRef, sub: Option<&Map<String, Value>>, docs: &Docs, order: &[(String, bool)],
        size: usize, min_doc_count: u64, missing: &Option<Key>, include: &Option<Matcher>, exclude: &Option<Matcher>) -> Result<(u64, Vec<Bucket>)> {
        // Docs per term: by ordinal for keyword columns (in parallel
        // slices for many docs), per document otherwise.
        let groups: HashMap<Key, Docs> = self.group_by(name, f, docs, missing)?;
        let mut bs: Vec<Bucket> = groups
            .into_iter()
            .filter(|(k, _)| include.as_ref().map(|m| m.matches(k)).unwrap_or(true) && !exclude.as_ref().map(|m| m.matches(k)).unwrap_or(false))
            .map(|(key, d)| Bucket { count: count(&d), key, docs: d, subs: None })
            .filter(|b| b.count >= min_doc_count)
            .collect();
        let total: u64 = bs.iter().map(|b| b.count).sum();
        let needs_subs_first = order.iter().any(|(k, _)| k != "_count" && k != "_key" && k != "_term");
        // Ranking needs only the sub-aggregations the order names: every
        // bucket computes those, and only the kept ones compute the rest (a
        // terms over a thousand pods ordered by an `avg` built each pod's
        // whole date_histogram, then kept a hundred).
        let ranking = if needs_subs_first { order_children(sub, order) } else { None };
        if needs_subs_first {
            for b in &mut bs {
                b.subs = Some(match &ranking {
                    Some(r) => self.compute(r, &b.docs)?,
                    None => self.subs(sub, &b.docs)?,
                });
            }
        }
        sort_buckets(&mut bs, order, o.get("field").and_then(Value::as_str) == Some(crate::tsdb::TSID_FIELD));
        bs.truncate(size);
        if ranking.is_some() {
            for b in &mut bs {
                b.subs = Some(self.subs(sub, &b.docs)?);
            }
        }
        Ok((total, bs))
    }

    #[allow(clippy::too_many_arguments)]
    fn terms_output(&self, f: &FieldRef, _o: &Map<String, Value>, sub: Option<&Map<String, Value>>, fmt: Option<DateFormatter>, show_err: bool,
        total: u64, bs: Vec<Bucket>) -> Result<Value> {
        self.add_buckets(bs.len())?;
        let kept: u64 = bs.iter().map(|b| b.count).sum();
        let mut out = Vec::new();
        for b in bs {
            let subs = match b.subs {
                Some(s) => s,
                None => self.subs(sub, &b.docs)?,
            };
            let mut m = Map::new();
            match (&b.key, f.ty) {
                (Key::Bool(v), _) => {
                    m.insert("key".into(), json!(*v as i64));
                    m.insert("key_as_string".into(), json!(v.to_string()));
                }
                (Key::Long(v), Some(t)) if t.is_date() => {
                    m.insert("key".into(), json!(v));
                    m.insert("key_as_string".into(), json!(fmt.clone().unwrap_or_default().format(*v, Tz::UTC)));
                }
                (k, _) => {
                    m.insert("key".into(), key_json(k));
                }
            }
            m.insert("doc_count".into(), json!(b.count));
            if show_err {
                m.insert("doc_count_error_upper_bound".into(), json!(0));
            }
            m.extend(subs);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({
            "doc_count_error_upper_bound": 0,
            "sum_other_doc_count": total - kept,
            "buckets": out,
        }))
    }

    /// [`Ctx::terms`] over a keyword's ordinals in one pass, for the shape
    /// TSDB dashboards send: every sub-aggregation a plain numeric metric
    /// (`avg`/`sum`/`min`/`max` of a field), or one `date_histogram` of
    /// such metrics. Each matched doc adds to its term's count and metrics
    /// and to its histogram bucket's, read from the columns once; no doc
    /// lists are built and no bucket reads its docs again (ES's per-bucket
    /// collectors). The kept terms' histograms are rendered by
    /// [`Ctx::date_histogram`] from the accumulated buckets
    /// ([`PREFUSED`]). Values accumulate in the order the general collector
    /// visits them (segment, then doc), so sums match it exactly. `None`:
    /// another shape, or a segment without plain columns.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn terms_one_pass(&self, name: &str, f: &FieldRef, sub: Option<&Map<String, Value>>, docs: &Docs, order: &[(String, bool)], size: usize,
        min_doc_count: u64) -> Result<Option<(u64, Vec<Bucket>)>> {
        let Some(sub) = sub.filter(|m| !m.is_empty()) else { return Ok(None) };
        if f.src.is_some() || NESTED_WHOLE_PATH.with(|w| w.borrow().is_some()) || (0..docs.len()).any(|s| self.is_elided(docs, s))
            || self.segs.iter().any(|s| s.lookup.registry.downsample_interval.is_some()) {
            return Ok(None);
        }
        // The sub-aggregations: metrics (by position in `sub`), at most
        // one histogram, and `filter`s on a field's existence over metrics
        // (index into `filters`).
        enum Part { Metric, Histogram, Filter(usize) }
        let mut metrics: Vec<FusedMetric> = Vec::new();
        let mut parts: Vec<(&String, Part)> = Vec::new();
        let mut filters: Vec<(FieldRef, Vec<FusedMetric>)> = Vec::new();
        let mut hist: Option<(&String, &Map<String, Value>, &Map<String, Value>, Vec<FusedMetric>, Rounding, FieldRef)> = None;
        for (sub_name, def) in sub {
            let Some(def) = def.as_object() else { return Ok(None) };
            if let Some(h) = def.get("date_histogram") {
                let Some(h) = h.as_object() else { return Ok(None) };
                let Some(hsub) = def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object) else { return Ok(None) };
                if hist.is_some() || def.len() != 2
                    || h.keys().any(|k| !matches!(k.as_str(), "field" | "fixed_interval" | "calendar_interval" | "time_zone" | "min_doc_count" | "format" | "offset")) {
                    return Ok(None);
                }
                let Some(hm) = self.fused_metrics(Some(hsub))?.filter(|m| !m.is_empty()) else { return Ok(None) };
                let Ok(hf) = self.source(h, sub_name, "date_histogram") else { return Ok(None) };
                if hf.src.is_some() || hf.ty == Some(EsType::Boolean) {
                    return Ok(None);
                }
                let rounding = Rounding::parse(h, sub_name)?;
                hist = Some((sub_name, h, hsub, hm, rounding, hf));
                parts.push((sub_name, Part::Histogram));
                continue;
            }
            if let Some(query) = def.get("filter") {
                // `{filter: exists, aggs: metrics}` on a plain doc-values field
                // (what [`Ctx::filter_exists_by_column`] reads).
                let fsub = def.get("aggs").or_else(|| def.get("aggregations"));
                if def.len() != 1 + fsub.is_some() as usize { return Ok(None); }
                let Some(field) = exists_field(query) else { return Ok(None) };
                if self.segs.iter().any(|s| crate::script::runtime_field(Some(s.index), field).is_some()) { return Ok(None); }
                let Ok(ff) = self.source_field(field) else { return Ok(None) };
                if ff.src.is_some() { return Ok(None); }
                let fm = match fsub {
                    None => Vec::new(),
                    Some(Value::Object(m)) if m.is_empty() => Vec::new(),
                    Some(Value::Object(m)) => match self.fused_metrics(Some(m))? { Some(fm) => fm, None => return Ok(None) },
                    Some(_) => return Ok(None),
                };
                parts.push((sub_name, Part::Filter(filters.len())));
                filters.push((ff, fm));
                continue;
            }
            let one: Map<String, Value> = std::iter::once((sub_name.clone(), Value::Object(def.clone()))).collect();
            let Some(mut m) = self.fused_metrics(Some(&one))? else { return Ok(None) };
            parts.push((sub_name, Part::Metric));
            metrics.push(m.remove(0));
        }
        // Ranking reads the metrics and the filters (their doc count, or a
        // metric of theirs) only.
        let filter_key = |k: &str| parts.iter().any(|(n, p)| match p {
            Part::Filter(i) => k == n.as_str() || k.strip_prefix(n.as_str()).and_then(|r| r.strip_prefix('>')).is_some_and(|m| filters[*i].1.iter().any(|fm| fm.name == m)),
            _ => false,
        });
        if order.iter().any(|(k, _)| !matches!(k.as_str(), "_count" | "_key" | "_term") && !metrics.iter().any(|m| &m.name == k) && !filter_key(k)) {
            return Ok(None);
        }
        struct Group {
            key: String,
            count: u64,
            accs: Vec<FusedAcc>,
            /// Per filter: its doc count and metric states.
            filtered: Vec<(u64, Vec<FusedAcc>)>,
            index: HashMap<i64, usize>,
            /// Per histogram slot: its doc count, and its metrics' states
            /// (`width` each, contiguous: docs land on random slots).
            counts: Vec<u64>,
            slots: Vec<FusedAcc>,
            /// The last histogram bucket added to: `[start, end)` and its slot.
            last: (i64, i64, usize),
        }
        let value = |c: &Option<std::rc::Rc<Col<'a>>>, d: DocId| -> Option<f64> {
            match c.as_deref()? {
                Col::I64(c) => c.get(d).map(|x| x as f64),
                Col::F64(c) => c.get(d),
                Col::U64(c) => c.get(d).map(|x| x as f64),
                _ => None,
            }
        };
        let metric_cols = |ms: &[FusedMetric], s: usize| -> Option<Vec<Option<std::rc::Rc<Col<'a>>>>> {
            let mut cols = Vec::with_capacity(ms.len());
            for m in ms {
                match self.num_col(&m.f, s) {
                    Some(c) => cols.push(Some(c)),
                    None if m.f.per_seg[s].is_none() || self.col(&m.f, s).is_some_and(|c| matches!(&*c, Col::Empty)) => cols.push(None),
                    None => return None,
                }
            }
            Some(cols)
        };
        let mut groups: Vec<Group> = Vec::new();
        let mut by_key: HashMap<String, usize> = HashMap::new();
        let mut memo = Vec::new();
        // Per segment: the histogram's keys over the segment's timestamps
        // (within the query's range on them) as a dense table, and per group
        // the slot of each key seen (TSDB docs of one term come in `_tsid`
        // order, not time order: every doc looks its bucket up).
        let mut dense: Vec<Vec<u32>> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let Some(rf) = &f.per_seg[s] else { return Ok(None) };
            if rf.ty != EsType::Keyword || rf.declared != EsType::Keyword || rf.flattened_key().is_some() || rf.name == crate::tsdb::TSID_FIELD {
                return Ok(None);
            }
            let Some(col) = self.col(f, s) else { return Ok(None) };
            let Col::Kw(c) = &*col else { return Ok(None) };
            let Some(cols) = metric_cols(&metrics, s) else { return Ok(None) };
            // Per filter: its existence column (`None`: unmapped here, so
            // nothing exists) and its metrics' columns.
            let mut fcols = Vec::with_capacity(filters.len());
            for (ff, fm) in &filters {
                let has = match ff.per_seg[s].as_ref() {
                    None => None,
                    Some(rf) => {
                        let plain = matches!(rf.declared, EsType::Keyword | EsType::Ip | EsType::Long | EsType::Integer | EsType::Short | EsType::Byte
                            | EsType::UnsignedLong | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::Date);
                        if !plain || !rf.doc_values || rf.flattened_key().is_some() || rf.name == crate::tsdb::TSID_FIELD { return Ok(None); }
                        let Some(col) = self.col(ff, s) else { return Ok(None) };
                        Some(col)
                    }
                };
                let Some(mc) = metric_cols(fm, s) else { return Ok(None) };
                fcols.push((has, mc));
            }
            let exists = |c: &Option<std::rc::Rc<Col<'a>>>, d: DocId| match c.as_deref() {
                None | Some(Col::Empty) => false,
                Some(Col::Kw(c)) => c.get(d).is_some(),
                Some(Col::MultiKw(c)) => c.ords(d).next().is_some(),
                Some(Col::I64(c)) => c.get(d).is_some(),
                Some(Col::MultiI64(c)) => c.values(d).next().is_some(),
                Some(Col::U64(c)) => c.get(d).is_some(),
                Some(Col::MultiU64(c)) => c.values(d).next().is_some(),
                Some(Col::F64(c)) => c.get(d).is_some(),
                Some(Col::MultiF64(c, _)) => c.values(d).next().is_some(),
            };
            let hcols = match &hist {
                Some((_, _, _, hm, _, hf)) => {
                    let Some(tc) = self.num_col(hf, s) else { return Ok(None) };
                    if !matches!(&*tc, Col::I64(_)) { return Ok(None); }
                    let Some(hc) = metric_cols(hm, s) else { return Ok(None) };
                    Some((tc, hc))
                }
                None => None,
            };
            let prepared = hcols.as_ref().and_then(|(tc, _)| {
                let Col::I64(tc) = &**tc else { return None };
                let (_, _, _, _, rounding, hf) = hist.as_ref()?;
                if tc.is_sparse() { return None; }
                let (mut lo, mut hi) = (tc.min()?, tc.max()?);
                if let Some(&Some((field, qlo, qhi))) = self.query_ranges.get(s) {
                    if hf.per_seg.get(s).and_then(|x| x.as_ref()).and_then(|rf| rf.core_id) == Some(field) { lo = lo.max(qlo); hi = hi.min(qhi); }
                }
                (lo <= hi).then(|| PreparedDateBuckets::new(rounding, lo, hi)).flatten()
            });
            dense.clear();
            dense.resize(groups.len(), Vec::new());
            let mut slot: Vec<u32> = vec![u32::MAX; c.num_terms()];
            for &(d, _) in v {
                let o = c.ord(d);
                let Some(at) = slot.get_mut(o as usize) else { continue };
                if *at == u32::MAX {
                    let Some(t) = c.term(o) else { continue };
                    let g = match by_key.get(&*t) {
                        Some(&g) => g,
                        None => {
                            self.charge_group(name, t.len(), docs.len())?;
                            groups.push(Group { key: t.to_string(), count: 0, accs: vec![FusedAcc::default(); metrics.len()],
                                filtered: filters.iter().map(|(_, fm)| (0, vec![FusedAcc::default(); fm.len()])).collect(), index: HashMap::new(),
                                counts: Vec::new(), slots: Vec::new(), last: (0, 0, 0) });
                            by_key.insert(t.into_owned(), groups.len() - 1);
                            dense.push(Vec::new());
                            groups.len() - 1
                        }
                    };
                    *at = g as u32;
                }
                let gi = *at as usize;
                let g = &mut groups[gi];
                g.count += 1;
                for (acc, c) in g.accs.iter_mut().zip(&cols) {
                    if let Some(x) = value(c, d) { acc.add(x); }
                }
                for ((n, accs), (has, mc)) in g.filtered.iter_mut().zip(&fcols) {
                    if !exists(has, d) { continue; }
                    *n += 1;
                    for (acc, c) in accs.iter_mut().zip(mc) {
                        if let Some(x) = value(c, d) { acc.add(x); }
                    }
                }
                let Some((tc, hc)) = &hcols else { continue };
                let Col::I64(tc) = &**tc else { continue };
                let Some(x) = tc.get(d) else { continue };
                let (start, end, b) = g.last;
                let in_table = prepared.as_ref().filter(|p| x >= p.keys[0] && x < p.keys[p.keys.len() - 1]);
                let b = if let Some(p) = in_table {
                    let i = p.bucket(x);
                    let slots = &mut dense[gi];
                    if slots.is_empty() {
                        self.charge(name, (p.keys.len() * 4) as u64)?;
                        slots.resize(p.keys.len() - 1, u32::MAX);
                    }
                    match slots[i] {
                        u32::MAX => {
                            let k = p.keys[i];
                            let b = match g.index.get(&k) {
                                Some(&b) => b,
                                None => {
                                    self.charge(name, BUCKET_BYTES)?;
                                    g.counts.push(0);
                                    g.slots.extend(std::iter::repeat_n(FusedAcc::default(), hc.len()));
                                    g.index.insert(k, g.counts.len() - 1);
                                    g.counts.len() - 1
                                }
                            };
                            slots[i] = b as u32;
                            b
                        }
                        b => b as usize,
                    }
                } else if x >= start && x < end { b } else {
                    let rounding = &hist.as_ref().expect("histogram columns").4;
                    let (k, end) = rounding.bucket_range(&mut memo, x);
                    let b = match g.index.get(&k) {
                        Some(&b) => b,
                        None => {
                            self.charge(name, BUCKET_BYTES)?;
                            g.counts.push(0);
                            g.slots.extend(std::iter::repeat_n(FusedAcc::default(), hc.len()));
                            g.index.insert(k, g.counts.len() - 1);
                            g.counts.len() - 1
                        }
                    };
                    g.last = (k, end, b);
                    b
                };
                g.counts[b] += 1;
                let width = hc.len();
                for (acc, c) in g.slots[b * width..(b + 1) * width].iter_mut().zip(hc) {
                    if let Some(x) = value(c, d) { acc.add(x); }
                }
            }
        }
        let metric_json = |ms: &[FusedMetric], accs: &[FusedAcc]| -> Map<String, Value> {
            ms.iter().zip(accs).map(|(m, a)| (m.name.clone(), simple_metric_json(m.t, a.n, a.sum.value, a.min, a.max))).collect()
        };
        // A filter renders as [`Ctx::single_bucket`] does.
        let filter_json = |i: usize, (n, accs): &(u64, Vec<FusedAcc>)| -> Value {
            let mut m = Map::new();
            m.insert("doc_count".into(), json!(n));
            m.extend(metric_json(&filters[i].1, accs));
            Value::Object(m)
        };
        let mut ranked: Vec<(Bucket, Group)> = groups.into_iter().filter(|g| g.count >= min_doc_count).map(|mut g| {
            let accs = std::mem::take(&mut g.accs);
            let mut subs = metric_json(&metrics, &accs);
            for (sub_name, part) in &parts {
                if let Part::Filter(i) = part { subs.insert((*sub_name).clone(), filter_json(*i, &g.filtered[*i])); }
            }
            (Bucket { key: Key::Str(std::mem::take(&mut g.key)), count: g.count, docs: Vec::new(), subs: Some(subs) }, g)
        }).collect();
        let total: u64 = ranked.iter().map(|(b, _)| b.count).sum();
        // Rank as the general collector does (ties break on the key).
        let mut bs: Vec<Bucket> = Vec::with_capacity(ranked.len());
        let mut states: HashMap<String, Group> = HashMap::new();
        for (b, g) in ranked.drain(..) {
            if let Key::Str(k) = &b.key { states.insert(k.clone(), g); }
            bs.push(b);
        }
        sort_buckets(&mut bs, order, false);
        bs.truncate(size);
        let empty: Docs = vec![Vec::new(); docs.len()];
        for b in &mut bs {
            let Key::Str(k) = &b.key else { continue };
            let metric_subs = b.subs.take().unwrap_or_default();
            let mut subs = Map::new();
            for (sub_name, part) in &parts {
                match part {
                    Part::Metric | Part::Filter(_) => { subs.insert((*sub_name).clone(), metric_subs.get(*sub_name).cloned().unwrap_or(Value::Null)); }
                    Part::Histogram => {
                        let (hname, h, hsub, _, _, _) = hist.as_ref().expect("histogram part");
                        let g = states.remove(k).expect("ranked group");
                        let width = hist.as_ref().map_or(0, |h| h.3.len()).max(1);
                        let buckets: Vec<(u64, Vec<FusedAcc>)> = g.counts.iter().zip(g.slots.chunks(width)).map(|(&n, accs)| (n, accs.to_vec())).collect();
                        PREFUSED.with(|p| *p.borrow_mut() = Some(((*hname).clone(), g.index, buckets)));
                        let rendered = self.compute_one(hname, "date_histogram", &Value::Object((*h).clone()), Some(hsub), &empty);
                        PREFUSED.with(|p| p.borrow_mut().take());
                        subs.insert((*hname).clone(), rendered?);
                    }
                }
            }
            b.subs = Some(subs);
        }
        #[cfg(test)]
        ONE_PASS_TERMS.with(|n| n.set(n.get() + 1));
        Ok(Some((total, bs)))
    }

    // -----------------------------------------------------------------------
    // range / date_range.

    fn range(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs, date: bool) -> Result<Value> {
        let f = self.source(o, name, if date { "date_range" } else { "range" })?;
        let is_date = date || f.ty.map(|t| t.is_date()).unwrap_or(false);
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
        let tz = match o.get("time_zone").and_then(Value::as_str) {
            Some(z) => Tz::parse(z)?,
            None => Tz::UTC,
        };
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(false);
        let missing = self.missing_key(&f, o.get("missing"))?;
        let ranges = o.get("ranges").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("No [ranges] specified for the [{name}] aggregation")))?;
        // ES `RangeAggregator` resolves the value source before parsing the
        // bounds: a `date` field (or a `date` value type) makes `from`/`to`
        // date-math expressions even for the plain `range` aggregation.
        let value_type = o.get("value_type").and_then(Value::as_str);
        let date_bounds = is_date || value_type == Some("date");
        let bound = |v: Option<&Value>| -> Result<Option<f64>> {
            match v {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Number(n)) => Ok(n.as_f64()),
                Some(Value::String(s)) => {
                    if date_bounds {
                        Ok(Some(parse_date_math(s, &fmt, tz, self.env.now, false)? as f64))
                    } else {
                        s.parse::<f64>().map(Some).map_err(|_| EsError::illegal_argument(format!("For input string: \"{s}\"")))
                    }
                }
                Some(_) => Err(EsError::parsing_at("[from]/[to] must be a number or a string", 1, 1)),
            }
        };
        let mut rs = Vec::new();
        for r in ranges {
            let from = bound(r.get("from"))?;
            let to = bound(r.get("to"))?;
            let key = r.get("key").and_then(Value::as_str).map(str::to_string);
            rs.push((from, to, key));
        }
        rs.sort_by(|a, b| a.0.unwrap_or(f64::NEG_INFINITY).total_cmp(&b.0.unwrap_or(f64::NEG_INFINITY)).then(a.1.unwrap_or(f64::INFINITY).total_cmp(&b.1.unwrap_or(f64::INFINITY))));
        // A `match_all`'s uncollected roots: the fused buckets read them
        // from the columns, anything else reads them back first.
        let any_elided = (0..docs.len()).any(|s| self.is_elided(docs, s));
        let bounds: Vec<(Option<f64>, Option<f64>)> = rs.iter().map(|r| (r.0, r.1)).collect();
        let fused = if missing.is_none() { self.range_auto_date_fused(&f, &bounds, sub, docs)? } else { None };
        if fused.is_none() && any_elided {
            return self.range(name, o, sub, &self.collect_elided(&f, docs), date);
        }
        self.add_buckets(rs.len())?;
        let mut fused = fused.map(|v| v.into_iter());
        // ES `RangeAggregator` renders the bounds with the value source's
        // formatter: a `date` field (or `value_type: date`) prints
        // `from_as_string` / `to_as_string` even for the plain `range` agg.
        let as_str = |v: f64| if date_bounds { fmt.format(v as i64, tz) } else { jfmt(v) };
        let mut out: Vec<(String, Map<String, Value>)> = Vec::new();
        for (from, to, key) in rs {
            // The fused buckets: this range's doc count and sub-aggregation.
            let fused_bucket = match fused.as_mut() {
                Some(it) => Some(it.next().ok_or_else(|| EsError::internal("fused range buckets"))?),
                None => None,
            };
            let d: Docs = if fused_bucket.is_some() { Vec::new() } else { docs
                .iter()
                .enumerate()
                .map(|(s, v)| {
                    v.iter()
                        .filter(|(doc, _)| {
                            self.values_or(&f, s, *doc, &missing).iter().filter_map(Key::as_f64).any(|x| from.map(|a| x >= a).unwrap_or(true) && to.map(|b| x < b).unwrap_or(true))
                        })
                        .copied()
                        .collect()
                })
                .collect() };
            let key = key.unwrap_or_else(|| format!("{}-{}", from.map(as_str).unwrap_or("*".into()), to.map(as_str).unwrap_or("*".into())));
            let mut m = Map::new();
            if !keyed {
                m.insert("key".into(), json!(key));
            }
            if let Some(a) = from {
                m.insert("from".into(), fnum(a));
                if date_bounds {
                    m.insert("from_as_string".into(), json!(as_str(a)));
                }
            }
            if let Some(b) = to {
                m.insert("to".into(), fnum(b));
                if date_bounds {
                    m.insert("to_as_string".into(), json!(as_str(b)));
                }
            }
            match fused_bucket {
                Some((n, subs, nb)) => {
                    // The histogram's buckets, accounted as its own
                    // aggregation accounts them.
                    if nb > self.env.max_buckets {
                        return Err(EsError::too_many_buckets(self.env.max_buckets));
                    }
                    if nb > 0 {
                        self.add_buckets(nb)?;
                    }
                    m.insert("doc_count".into(), json!(n));
                    m.extend(subs);
                }
                None => {
                    m.insert("doc_count".into(), json!(count(&d)));
                    m.extend(self.subs(sub, &d)?);
                }
            }
            out.push((key, m));
        }
        let b = if keyed {
            Value::Object(out.into_iter().map(|(k, m)| (k, Value::Object(m))).collect())
        } else {
            Value::Array(out.into_iter().map(|(_, m)| Value::Object(m)).collect())
        };
        Ok(json!({ "buckets": b }))
    }

    /// The buckets of a `range` over `f` (a single-valued numeric column)
    /// whose only sub-aggregation is a plain `auto_date_histogram`, maybe
    /// with plain numeric metrics ([`range_auto_date_fields`]), folded from
    /// the columns: for each range of `bounds` (in order), its doc count,
    /// its sub-aggregation's JSON and the histogram's bucket count. Two
    /// passes over the docs (each range's date bounds, then its buckets),
    /// in parts on the search pool, no per-range doc lists; `docs`' root
    /// segments left uncollected ([`with_elided_root`] over a `match_all`)
    /// are read whole. The same output as the per-range collection
    /// ([`Ctx::auto_date_histogram`]'s rounding choice, keys and metrics;
    /// compensated sums combined in doc order). `None`: compute it per
    /// range.
    #[allow(clippy::type_complexity)]
    fn range_auto_date_fused(&self, f: &FieldRef, bounds: &[(Option<f64>, Option<f64>)], sub: Option<&Map<String, Value>>, docs: &Docs)
        -> Result<Option<Vec<(u64, Map<String, Value>, usize)>>> {
        if f.src.is_some() || partial::active() || DOC_WEIGHTS.with(|w| w.borrow().is_some()) {
            return Ok(None);
        }
        #[cfg(test)]
        if NO_FUSED_RANGE.with(Cell::get) {
            return Ok(None);
        }
        let Some(sub) = sub.filter(|m| m.len() == 1) else { return Ok(None) };
        let (an, def) = sub.iter().next().expect("one entry");
        let Some(def) = def.as_object() else { return Ok(None) };
        if def.keys().any(|k| !matches!(k.as_str(), "auto_date_histogram" | "aggs" | "aggregations")) {
            return Ok(None);
        }
        let Some(o) = def.get("auto_date_histogram").and_then(Value::as_object) else { return Ok(None) };
        if o.keys().any(|k| !matches!(k.as_str(), "field" | "buckets" | "time_zone" | "format" | "minimum_interval")) {
            return Ok(None);
        }
        // Anything the histogram would reject: it reports it itself.
        let Some(target) = o.get("buckets").map_or(Some(10), Value::as_i64).filter(|t| *t > 0) else { return Ok(None) };
        let target = target as u64;
        let Ok(tf) = o.get("field").and_then(Value::as_str).ok_or(()).and_then(|n| self.field(n).map_err(|_| ())) else { return Ok(None) };
        if tf.src.is_some() {
            return Ok(None);
        }
        let tz = match o.get("time_zone").and_then(Value::as_str) {
            Some(z) => match Tz::parse(z) { Ok(tz) => tz, Err(_) => return Ok(None) },
            None => Tz::UTC,
        };
        let fmt = match o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose() {
            Ok(fmt) => fmt.or_else(|| tf.format.clone()).unwrap_or_default(),
            Err(_) => return Ok(None),
        };
        let Ok(min_unit) = auto_date::minimum_unit(o.get("minimum_interval").and_then(Value::as_str)) else { return Ok(None) };
        let metrics = match def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object).filter(|m| !m.is_empty()) {
            Some(m) => match self.fused_metrics(Some(m))? { Some(ms) => ms, None => return Ok(None) },
            None => Vec::new(),
        };
        // Every segment with docs: the range's column, the date's (or none:
        // unmapped or no values there), the metrics' columns (likewise).
        let absent = |g: &FieldRef, s: usize| g.per_seg.get(s).is_none_or(Option::is_none) || self.col(g, s).is_some_and(|c| matches!(&*c, Col::Empty));
        type SegCols<'a> = (Option<std::rc::Rc<Col<'a>>>, Option<std::rc::Rc<Col<'a>>>, Vec<Option<std::rc::Rc<Col<'a>>>>);
        let mut cols: Vec<Option<SegCols<'a>>> = Vec::with_capacity(docs.len());
        let mut whole: Vec<bool> = Vec::with_capacity(docs.len());
        for (s, v) in docs.iter().enumerate() {
            let elided = self.is_elided(docs, s);
            // Uncollected for a query range of another aggregation: not here.
            if elided && (self.query_ranges.get(s).is_some_and(Option::is_some) || self.query_f64_ranges.get(s).is_some_and(Option::is_some)) {
                return Ok(None);
            }
            whole.push(elided);
            if v.is_empty() && !elided {
                cols.push(None);
                continue;
            }
            let rc = match self.num_col(f, s) {
                Some(c) => Some(c),
                None if absent(f, s) => None,
                None => return Ok(None),
            };
            let tc = match self.num_col(&tf, s) {
                Some(c) if matches!(&*c, Col::I64(_)) => Some(c),
                None if absent(&tf, s) => None,
                _ => return Ok(None),
            };
            let mut mcs = Vec::with_capacity(metrics.len());
            for m in &metrics {
                mcs.push(match self.num_col(&m.f, s) {
                    Some(c) => Some(c),
                    None if absent(&m.f, s) => None,
                    None => return Ok(None),
                });
            }
            cols.push(Some((rc, tc, mcs)));
        }
        crate::search::check_cancelled()?;
        // Parts: runs of docs of one segment (a doc list's positions, or a
        // whole segment's doc ids).
        let part_docs = (rustic_core::query::scorer::part_split_min_docs() as usize / 2).max(1 << 14);
        let mut work: Vec<(usize, usize, usize)> = Vec::new();
        for (s, c) in cols.iter().enumerate() {
            let Some((Some(_), _, _)) = c else { continue };
            let n = if whole[s] { self.segs[s].leaf.num_docs() as usize } else { docs[s].len() };
            let mut i0 = 0;
            while i0 < n {
                let i1 = (i0 + part_docs).min(n);
                work.push((s, i0, i1));
                i0 = i1;
            }
        }
        let read = |c: &Col<'a>, d: DocId| -> Option<f64> {
            match c {
                Col::I64(c) => c.get(d).map(|x| x as f64),
                Col::F64(c) => c.get(d),
                Col::U64(c) => c.get(d).map(|x| x as f64),
                _ => None,
            }
        };
        let refs: Vec<Option<(&Col<'a>, Option<&Col<'a>>, Vec<Option<&Col<'a>>>)>> = cols.iter().map(|c| c.as_ref().and_then(|(rc, tc, ms)| {
            Some((rc.as_deref()?, tc.as_deref(), ms.iter().map(|m| m.as_deref()).collect()))
        })).collect();
        let nr = bounds.len();
        let inside = |x: f64, r: usize| bounds[r].0.is_none_or(|a| x >= a) && bounds[r].1.is_none_or(|b| x < b);
        // Visit part `w`'s docs: `f(doc, range value)` for each with one.
        let visit = |w: (usize, usize, usize), f: &mut dyn FnMut(&(&Col<'a>, Option<&Col<'a>>, Vec<Option<&Col<'a>>>), DocId, f64)| {
            let (s, i0, i1) = w;
            let Some(c) = &refs[s] else { return };
            let mut each = |d: DocId| if let Some(x) = read(c.0, d) { f(c, d, x) };
            if whole[s] {
                for d in i0 as DocId..i1 as DocId { each(d); }
            } else {
                for &(d, _) in &docs[s][i0..i1] { each(d); }
            }
        };
        let parallel = work.len() > 1;
        let run_parts = |job: &(dyn Fn(usize) -> Box<dyn std::any::Any + Send> + Sync)| -> Vec<Box<dyn std::any::Any + Send>> {
            if parallel { rustic_core::query::searcher::par_map_search(work.len(), job) } else { (0..work.len()).map(job).collect() }
        };
        // Pass 1: each range's doc count and date bounds.
        type Span = (u64, Option<(i64, i64)>);
        let firsts = run_parts(&|i| {
            let mut out: Vec<Span> = vec![(0, None); nr];
            visit(work[i], &mut |c, d, x| {
                let t = c.1.and_then(|tc| if let Col::I64(tc) = tc { tc.get(d) } else { None });
                for (r, o) in out.iter_mut().enumerate() {
                    if !inside(x, r) { continue; }
                    o.0 += 1;
                    if let Some(t) = t {
                        o.1 = Some(match o.1 { None => (t, t), Some((a, b)) => (a.min(t), b.max(t)) });
                    }
                }
            });
            Box::new(out)
        });
        crate::search::check_cancelled()?;
        let mut spans: Vec<Span> = vec![(0, None); nr];
        for p in firsts {
            let p = p.downcast::<Vec<Span>>().map_err(|_| EsError::internal("fused range part"))?;
            for (o, q) in spans.iter_mut().zip(p.iter()) {
                o.0 += q.0;
                if let Some((a, b)) = q.1 {
                    o.1 = Some(match o.1 { None => (a, b), Some((x, y)) => (x.min(a), y.max(b)) });
                }
            }
        }
        // Each range's rounding (as the histogram chooses it over the
        // range's docs), its keys, and a dense bucket lookup.
        let choices: Vec<Option<(auto_date::Choice, usize, Option<PreparedDateBuckets>)>> = spans.iter().map(|(_, span)| {
            let (lo, hi) = (*span)?;
            let choice = auto_date::choose(lo, hi, target, min_unit, 0, tz);
            let nb = (choice.bucket_of(hi) + 1) as usize;
            if nb > self.env.max_buckets {
                return Some((choice, nb, None));
            }
            let keys: Vec<i64> = (0..=nb).map(|b| choice.key_of(b as i64)).collect();
            let verified = keys[0] <= lo && keys[nb] > hi && keys.windows(2).enumerate().all(|(i, pair)| {
                pair[0] < pair[1] && choice.bucket_of(pair[0]) == i as i64 && choice.bucket_of(pair[1] - 1) == i as i64
            });
            Some((choice, nb, verified.then(|| PreparedDateBuckets::from_verified_keys(keys)).flatten()))
        }).collect();
        // Too many buckets: the histogram reports it.
        if choices.iter().flatten().any(|(_, nb, _)| *nb > self.env.max_buckets) {
            return Ok(Some(spans.iter().zip(&choices).map(|(sp, ch)| (sp.0, Map::new(), ch.as_ref().map_or(0, |c| c.1))).collect()));
        }
        // Pass 2: each range's bucket counts and metric states.
        let nm = metrics.len();
        type Fill = Vec<(Vec<u64>, Vec<FusedAcc>)>;
        let fresh = || -> Fill { choices.iter().map(|ch| ch.as_ref().map_or((Vec::new(), Vec::new()), |(_, nb, _)| (vec![0u64; *nb], vec![FusedAcc::default(); nb * nm]))).collect() };
        let seconds = run_parts(&|i| {
            let mut out = fresh();
            visit(work[i], &mut |c, d, x| {
                let Some(Col::I64(tc)) = c.1 else { return };
                let Some(t) = tc.get(d) else { return };
                for r in 0..nr {
                    if !inside(x, r) { continue; }
                    let Some((choice, _, prepared)) = &choices[r] else { continue };
                    let b = match prepared { Some(p) => p.bucket(t), None => choice.bucket_of(t) as usize };
                    let (counts, accs) = &mut out[r];
                    counts[b] += 1;
                    for (j, mc) in c.2.iter().enumerate() {
                        if let Some(y) = mc.and_then(|mc| read(mc, d)) { accs[b * nm + j].add(y); }
                    }
                }
            });
            Box::new(out)
        });
        crate::search::check_cancelled()?;
        let mut fill = fresh();
        for p in seconds {
            let p = p.downcast::<Fill>().map_err(|_| EsError::internal("fused range part"))?;
            for ((counts, accs), (pc, pa)) in fill.iter_mut().zip(p.iter()) {
                for (a, b) in counts.iter_mut().zip(pc) { *a += b; }
                for (a, b) in accs.iter_mut().zip(pa) {
                    a.n += b.n;
                    a.sum.merge(&b.sum);
                    a.min = a.min.min(b.min);
                    a.max = a.max.max(b.max);
                }
            }
        }
        let mut out = Vec::with_capacity(nr);
        for (r, (counts, accs)) in fill.into_iter().enumerate() {
            let mut subs = Map::new();
            let Some((choice, nb, _)) = &choices[r] else {
                subs.insert(an.clone(), json!({"buckets": [], "interval": format!("1{}", extra::unit_suffix(min_unit))}));
                out.push((spans[r].0, subs, 0));
                continue;
            };
            let buckets: Vec<Value> = counts.iter().enumerate().map(|(b, &count)| {
                let key = choice.key_of(b as i64);
                let mut m = Map::new();
                m.insert("key_as_string".into(), json!(fmt.format(key, tz)));
                m.insert("key".into(), json!(key));
                m.insert("doc_count".into(), json!(count));
                for (j, mt) in metrics.iter().enumerate() {
                    let a = &accs[b * nm + j];
                    m.insert(mt.name.clone(), simple_metric_json(mt.t, a.n, a.sum.value, a.min, a.max));
                }
                Value::Object(m)
            }).collect();
            subs.insert(an.clone(), json!({"buckets": buckets, "interval": choice.interval()}));
            out.push((spans[r].0, subs, *nb));
        }
        #[cfg(test)]
        FUSED_RANGES.with(|c| c.set(c.get() + 1));
        Ok(Some(out))
    }

    // -----------------------------------------------------------------------
    // histogram / date_histogram.

    fn histogram(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.source(o, name, "histogram")?;
        let interval = o.get("interval").and_then(Value::as_f64).ok_or_else(|| EsError::illegal_argument(format!("[interval] must be >0 for histogram aggregation [{name}]")))?;
        if interval <= 0.0 {
            return Err(EsError::illegal_argument(format!("[interval] must be >0 for histogram aggregation [{name}]")));
        }
        let offset = o.get("offset").and_then(Value::as_f64).unwrap_or(0.0);
        let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0);
        let missing = self.missing_key(&f, o.get("missing"))?;
        let hard = bounds(o.get("hard_bounds"));
        let ext = bounds(o.get("extended_bounds"));
        let round = |v: f64| ((v - offset) / interval).floor() * interval + offset;
        // Uncollected matches (a bare double range on this field): the
        // fused pass folds them from the column; anything else reads them
        // back first.
        let any_elided = (0..docs.len()).any(|s| self.is_elided(docs, s));
        if any_elided && missing.is_some() {
            return self.histogram(name, o, sub, &self.collect_elided(&f, docs));
        }
        if missing.is_none() {
            if let Some(out) = self.histogram_of_metrics(name, o, sub, docs, &f, &round, hard, ext, interval, offset, min_doc_count)? {
                return Ok(out);
            }
        }
        if any_elided {
            return self.histogram(name, o, sub, &self.collect_elided(&f, docs));
        }
        // Keyed by the bucket's bits (no DoS exposure: keys are rounded
        // doubles): aHash instead of SipHash, which was a third of the pass.
        let mut groups: HashMap<u64, Docs, ahash::RandomState> = HashMap::default();
        const DENSE: i64 = 1 << 16;
        let mut dense: Vec<u32> = Vec::new();
        let mut dense_base: Option<i64> = None;
        let mut dense_groups: Vec<(u64, Docs)> = Vec::new();
        let mut seen = 0;
        let in_hard = |x: f64| hard.map(|(a, b)| a.map(|a| x >= a).unwrap_or(true) && b.map(|b| x <= b).unwrap_or(true)).unwrap_or(true);
        let missing_num = missing.as_ref().and_then(Key::as_f64);
        for (s, v) in docs.iter().enumerate() {
            self.charge_grouping(name, groups.len() + dense_groups.len(), &mut seen, docs.len(), v.len())?;
            // A single-valued numeric column: each doc has at most one key,
            // read with no per-document allocation, sort or dedup (what
            // `values_or` + `Key::as_f64` give, as in `num_values`).
            if let Some(col) = self.num_col(&f, s) {
                // Bucket ordinals `floor((x - offset) / interval)` (what
                // `round` keys on) in a window of slots map straight to their
                // bucket's doc lists, with no hash per doc.
                // Dense lists read the column a block of docs at a time.
                const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
                let span = match (v.first(), v.last()) { (Some(a), Some(b)) => (b.0 - a.0) as usize + 1, _ => 0 };
                let dense_list = v.len() * 8 >= span;
                let mut fbuf = [0.0f64; BS];
                let mut fcur = usize::MAX;
                macro_rules! assign {
                    ($c:expr, $get:expr) => {
                        for &(d, sc) in v {
                            let Some(x) = ($get)($c, d).or(missing_num) else { continue };
                            if !in_hard(x) {
                                continue;
                            }
                            let q = ((x - offset) / interval).floor();
                            let slot = (q.abs() < 1e15).then(|| q as i64).and_then(|ord| {
                                let base = *dense_base.get_or_insert(ord - DENSE / 2);
                                let at = ord - base;
                                (0..DENSE).contains(&at).then_some(at as usize)
                            });
                            if let Some(at) = slot {
                                if dense.len() <= at {
                                    dense.resize(at + 1, 0);
                                }
                                let i = match dense[at] {
                                    0 => {
                                        dense_groups.push((round(x).to_bits(), vec![Vec::new(); docs.len()]));
                                        dense[at] = dense_groups.len() as u32;
                                        dense_groups.len() - 1
                                    }
                                    i => i as usize - 1,
                                };
                                dense_groups[i].1[s].push((d, sc));
                                continue;
                            }
                            let k = round(x).to_bits();
                            match groups.get_mut(&k) {
                                Some(per_seg) => per_seg[s].push((d, sc)),
                                None => {
                                    let mut per_seg = vec![Vec::new(); docs.len()];
                                    per_seg[s].push((d, sc));
                                    groups.insert(k, per_seg);
                                }
                            }
                        }
                    };
                }
                match &*col {
                    Col::I64(c) => assign!(c, |c: &rustic_core::codec::fastfields::ColumnReader<'_, i64>, d: DocId| c.get(d).map(|x| x as f64)),
                    Col::F64(c) if dense_list && c.block_values(0, &mut [0.0; 1]).is_some() => {
                        let full = c.all_present();
                        assign!(c, |c: &rustic_core::codec::fastfields::ColumnReader<'_, f64>, d: DocId| {
                            let b = d as usize / BS;
                            if b != fcur {
                                c.block_values(b, &mut fbuf);
                                fcur = b;
                            }
                            (full || c.has_value(d)).then(|| fbuf[d as usize % BS])
                        })
                    }
                    Col::F64(c) => assign!(c, |c: &rustic_core::codec::fastfields::ColumnReader<'_, f64>, d: DocId| c.get(d)),
                    Col::U64(c) => assign!(c, |c: &rustic_core::codec::fastfields::ColumnReader<'_, u64>, d: DocId| c.get(d).map(|x| x as f64)),
                    _ => unreachable!("num_col returns numeric columns"),
                }
                continue;
            }
            for &(d, sc) in v {
                let mut ks: Vec<f64> = self.values_or(&f, s, d, &missing).iter().filter_map(Key::as_f64).filter(|x| hard.map(|(a, b)| a.map(|a| *x >= a).unwrap_or(true) && b.map(|b| *x <= b).unwrap_or(true)).unwrap_or(true)).map(round).collect();
                ks.sort_by(|a, b| a.total_cmp(b));
                ks.dedup();
                for k in ks {
                    groups.entry(k.to_bits()).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc));
                }
            }
        }
        // The slotted buckets join the hashed ones (a key in both: lists
        // merged per segment, each in doc order).
        for (k, lists) in dense_groups {
            match groups.get_mut(&k) {
                None => { groups.insert(k, lists); }
                Some(g) => {
                    for (seg, l) in lists.into_iter().enumerate() {
                        if !l.is_empty() {
                            g[seg].extend(l);
                            g[seg].sort_unstable_by_key(|e| e.0);
                        }
                    }
                }
            }
        }
        self.charge_grouping(name, groups.len(), &mut seen, docs.len(), 0)?;
        let mut keys: Vec<f64> = groups.keys().map(|b| f64::from_bits(*b)).collect();
        if min_doc_count == 0 {
            let mut lo = keys.iter().copied().fold(f64::INFINITY, f64::min);
            let mut hi = keys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if let Some((a, b)) = ext {
                if let Some(a) = a {
                    lo = lo.min(round(a));
                }
                if let Some(b) = b {
                    hi = hi.max(round(b));
                }
            }
            if let Some((a, b)) = hard {
                if let Some(a) = a {
                    lo = lo.max(round(a));
                }
                if let Some(b) = b {
                    hi = hi.min(round(b));
                }
            }
            if lo.is_finite() && hi.is_finite() {
                let n = ((hi - lo) / interval).round() as i64;
                if n as usize > self.env.max_buckets {
                    return Err(EsError::too_many_buckets(self.env.max_buckets));
                }
                keys = (0..=n).map(|i| lo + i as f64 * interval).collect();
            }
        }
        let mut bs: Vec<Bucket> = keys
            .into_iter()
            .map(|k| {
                let d = groups.remove(&k.to_bits()).unwrap_or_else(|| vec![Vec::new(); docs.len()]);
                Bucket { key: Key::Dbl(k), count: count(&d), docs: d, subs: None }
            })
            .filter(|b| b.count >= min_doc_count)
            .collect();
        self.finish_histogram(o, sub, &mut bs, |b, m| {
            m.insert("key".into(), key_json(&b.key));
        })
    }

    /// A `histogram` of a single-valued numeric column whose only
    /// sub-aggregations are `avg` / `sum` / `min` / `max` / `stats` of other
    /// single-valued numeric columns (Rally nyc_taxis `distance_amount_agg`):
    /// one pass over the docs folds each bucket's count and metrics, with no
    /// per-bucket doc lists and no second read per metric. The same values
    /// in the same order as the general path (which collects each bucket's
    /// docs, then reads them per metric), so the same results. `None` when
    /// the request is anything else.
    #[allow(clippy::too_many_arguments)]
    fn histogram_of_metrics(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs, f: &FieldRef,
        round: &(dyn Fn(f64) -> f64 + Sync), hard: Option<(Option<f64>, Option<f64>)>, ext: Option<(Option<f64>, Option<f64>)>, interval: f64, offset: f64, min_doc_count: u64) -> Result<Option<Value>> {
        // No sub-aggregations: the same pass with no metrics (counts only),
        // instead of grouping every doc into per-bucket lists.
        let no_subs = Map::new();
        let sub = sub.unwrap_or(&no_subs);
        if DOC_WEIGHTS.with(|w| w.borrow().is_some()) {
            return Ok(None);
        }
        // An uncollected segment folds every doc whose value is in the
        // query's double range (no deletions: the search checked).
        let mut elided_ranges: Vec<Option<(f64, f64)>> = vec![None; docs.len()];
        for s in 0..docs.len() {
            if !self.is_elided(docs, s) {
                continue;
            }
            let Some(&Some((field, lo, hi))) = self.query_f64_ranges.get(s) else { return Ok(None) };
            if f.per_seg.get(s).and_then(|x| x.as_ref()).and_then(|rf| rf.core_id) != Some(field) || self.segs[s].leaf.deleted_bits().is_some() {
                return Ok(None);
            }
            elided_ranges[s] = Some((lo, hi));
        }
        if parse_order(o.get("order"), &[("_key", true)])?.iter().any(|(k, _)| k != "_count" && k != "_key") {
            return Ok(None);
        }
        let mut metrics: Vec<(&str, &str, FieldRef)> = Vec::new();
        for (sub_name, def) in sub {
            let Some(d) = def.as_object().filter(|d| d.len() == 1) else { return Ok(None) };
            let (t, body) = d.iter().next().expect("one entry");
            if !matches!(t.as_str(), "avg" | "sum" | "min" | "max" | "stats") {
                return Ok(None);
            }
            let Some(b) = body.as_object().filter(|b| b.len() == 1 && b.get("field").is_some_and(Value::is_string)) else { return Ok(None) };
            let _ = b;
            let mf = self.source(body.as_object().expect("object"), sub_name, t)?;
            if mf.ty.map(|t| t.is_date()).unwrap_or(false) {
                return Ok(None);
            }
            metrics.push((sub_name.as_str(), t.as_str(), mf));
        }
        // Every segment with docs reads plain numeric columns.
        let mut cols = Vec::with_capacity(docs.len());
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() && elided_ranges[s].is_none() {
                cols.push(None);
                continue;
            }
            let Some(c) = self.num_col(f, s) else { return Ok(None) };
            let mut ms = Vec::with_capacity(metrics.len());
            for (_, _, mf) in &metrics {
                let Some(mc) = self.num_col(mf, s) else { return Ok(None) };
                ms.push(mc);
            }
            cols.push(Some((c, ms)));
        }
        #[derive(Clone, Copy)]
        struct Acc { n: usize, sum: extra::Kahan, min: f64, max: f64 }
        let empty = Acc { n: 0, sum: extra::Kahan::default(), min: f64::INFINITY, max: f64::NEG_INFINITY };
        let read = |c: &Col<'a>, d: DocId| -> Option<f64> {
            match c {
                Col::I64(c) => c.get(d).map(|x| x as f64),
                Col::F64(c) => c.get(d),
                Col::U64(c) => c.get(d).map(|x| x as f64),
                _ => None,
            }
        };
        let in_hard = |x: f64| hard.map(|(a, b)| a.map(|a| x >= a).unwrap_or(true) && b.map(|b| x <= b).unwrap_or(true)).unwrap_or(true);
        // Bucket ordinals `floor((x - offset) / interval)` in a window of
        // DENSE slots map straight to their bucket (no hash per doc); the
        // key is computed from the same floor as `round`, so it is the same.
        const DENSE: i64 = 1 << 16;
        let inv = 1.0 / interval;
        let pow2 = interval.is_finite() && interval > 0.0 && interval.to_bits() & ((1u64 << 52) - 1) == 0 && inv.is_normal();
        let nm = metrics.len();
        /// One fold's buckets: `(key bits, count, metrics)` in first-seen
        /// order, their index, and the dense window of bucket ordinals.
        struct HState { index: HashMap<u64, usize, ahash::RandomState>, buckets: Vec<(u64, u64, Vec<Acc>)>, dense: Vec<u32>, dense_base: Option<i64> }
        #[inline(always)]
        fn bucket_of(st: &mut HState, x: f64, pow2: bool, inv: f64, interval: f64, offset: f64, round: &(dyn Fn(f64) -> f64 + Sync), nm: usize, empty: Acc) -> usize {
            // A power-of-two interval divides exactly as a multiply.
            let q = if pow2 { ((x - offset) * inv).floor() } else { ((x - offset) / interval).floor() };
            let slot = (q.abs() < 1e15).then(|| q as i64).and_then(|ord| {
                let base = *st.dense_base.get_or_insert(ord - DENSE / 2);
                let at = ord - base;
                (0..DENSE).contains(&at).then_some(at as usize)
            });
            if let Some(i) = slot.and_then(|at| st.dense.get(at).copied().filter(|&i| i != 0)) {
                return i as usize - 1;
            }
            let k = round(x).to_bits();
            let i = match st.index.get(&k) {
                Some(&i) => i,
                None => {
                    st.buckets.push((k, 0, vec![empty; nm]));
                    st.index.insert(k, st.buckets.len() - 1);
                    st.buckets.len() - 1
                }
            };
            if let Some(at) = slot {
                if st.dense.len() <= at {
                    st.dense.resize(at + 1, 0);
                }
                st.dense[at] = i as u32 + 1;
            }
            i
        }
        #[inline(always)]
        fn add_metric(a: &mut Acc, y: f64) {
            a.n += 1;
            a.sum.add(y);
            a.min = a.min.min(y);
            a.max = a.max.max(y);
        }
        // The fold, cut into work items in doc order: a large segment's doc
        // ranges (or slices of its collected docs) fold on the search pool,
        // each into its own buckets, merged in item order (the same buckets;
        // compensated sums combine as across segments). Few docs fold in one
        // pass, as before.
        let part_docs = (rustic_core::query::scorer::part_split_min_docs() as usize / 2).max(1);
        const BS: usize = rustic_core::codec::fastfields::BLOCK_SIZE;
        enum Work { Blocked { s: usize, lo: f64, hi: f64, b0: usize, b1: usize }, Scan { s: usize, lo: f64, hi: f64, d0: DocId, d1: DocId }, Docs { s: usize, i0: usize, i1: usize } }
        let mut work: Vec<Work> = Vec::new();
        let mut total = 0usize;
        let refs: Vec<Option<(&Col<'a>, Vec<&Col<'a>>)>> = cols.iter().map(|c| c.as_ref().map(|(c, ms)| (&**c, ms.iter().map(|m| &**m).collect()))).collect();
        for (s, v) in docs.iter().enumerate() {
            let Some((c, ms)) = &refs[s] else { continue };
            match (elided_ranges[s], *c) {
                (Some((lo, hi)), Col::F64(col)) => {
                    let n = self.segs[s].leaf.num_docs() as usize;
                    total += n;
                    if ms.iter().all(|mc| matches!(mc, Col::F64(_))) && col.block_values(0, &mut [0.0; 1]).is_some() {
                        let nb = n.div_ceil(BS);
                        let step = (part_docs / BS).max(1);
                        let mut b0 = 0;
                        while b0 < nb { let b1 = (b0 + step).min(nb); work.push(Work::Blocked { s, lo, hi, b0, b1 }); b0 = b1; }
                    } else {
                        let mut d0 = 0usize;
                        while d0 < n { let d1 = (d0 + part_docs).min(n); work.push(Work::Scan { s, lo, hi, d0: d0 as DocId, d1: d1 as DocId }); d0 = d1; }
                    }
                }
                (Some(_), _) => return Ok(None),
                (None, _) => {
                    total += v.len();
                    let mut i0 = 0;
                    while i0 < v.len() { let i1 = (i0 + part_docs).min(v.len()); work.push(Work::Docs { s, i0, i1 }); i0 = i1; }
                }
            }
        }
        let read = |c: &Col<'a>, d: DocId| -> Option<f64> {
            match c {
                Col::I64(c) => c.get(d).map(|x| x as f64),
                Col::F64(c) => c.get(d),
                Col::U64(c) => c.get(d).map(|x| x as f64),
                _ => None,
            }
        };
        let segs = self.segs;
        let run = |w: &Work, st: &mut HState| {
            match *w {
                // The query's matches, read from the column in doc order
                // (the order a collected list holds them in), a block of
                // docs at a time when every column decodes by block: the
                // same values folded in the same order.
                Work::Blocked { s, lo, hi, b0, b1 } => {
                    let Some((Col::F64(col), ms)) = &refs[s] else { unreachable!("checked when cut") };
                    let mcs: Vec<&rustic_core::codec::fastfields::ColumnReader<'_, f64>> = ms.iter().map(|mc| match mc { Col::F64(c) => c, _ => unreachable!("checked when cut") }).collect();
                    let mut xs = [0.0f64; BS];
                    let mut ys: Vec<[f64; BS]> = vec![[0.0; BS]; mcs.len()];
                    let mut blocked: Vec<bool> = mcs.iter().map(|c| c.block_values(0, &mut [0.0; 1]).is_some()).collect();
                    let x_full = col.all_present();
                    let y_full: Vec<bool> = mcs.iter().map(|c| c.all_present()).collect();
                    for b in b0..b1 {
                        // Stored values: the query's range is on them.
                        let Some(k) = col.block_values_stored(b, &mut xs) else { break };
                        for (j, c) in mcs.iter().enumerate() {
                            if blocked[j] && c.block_values(b, &mut ys[j]).is_none() {
                                blocked[j] = false;
                            }
                        }
                        let base = (b * BS) as DocId;
                        for i in 0..k {
                            let d = base + i as DocId;
                            if !x_full && !col.has_value(d) {
                                continue;
                            }
                            if !f64_in_range(xs[i], lo, hi) {
                                continue;
                            }
                            let x = col.es(xs[i]);
                            if !in_hard(x) {
                                continue;
                            }
                            let bi = bucket_of(st, x, pow2, inv, interval, offset, round, nm, empty);
                            let bucket = &mut st.buckets[bi];
                            bucket.1 += 1;
                            for (j, c) in mcs.iter().enumerate() {
                                let y = if blocked[j] {
                                    if !y_full[j] && !c.has_value(d) { continue }
                                    ys[j][i]
                                } else {
                                    match c.get(d) { Some(y) => y, None => continue }
                                };
                                add_metric(&mut bucket.2[j], y);
                            }
                        }
                    }
                }
                Work::Scan { s, lo, hi, d0, d1 } => {
                    let Some((Col::F64(col), ms)) = &refs[s] else { unreachable!("checked when cut") };
                    for d in d0..d1 {
                        if let Some(x) = col.get_stored(d) {
                            if f64_in_range(x, lo, hi) {
                                let x = col.es(x);
                                if in_hard(x) {
                                    let bi = bucket_of(st, x, pow2, inv, interval, offset, round, nm, empty);
                                    let b = &mut st.buckets[bi];
                                    b.1 += 1;
                                    for (j, mc) in ms.iter().enumerate() {
                                        if let Some(y) = read(mc, d) { add_metric(&mut b.2[j], y); }
                                    }
                                }
                            }
                        }
                    }
                }
                Work::Docs { s, i0, i1 } => {
                    let Some((c, ms)) = &refs[s] else { unreachable!("cut from a column") };
                    for &(d, _) in &docs[s][i0..i1] {
                        if let Some(x) = read(c, d) {
                            if in_hard(x) {
                                let bi = bucket_of(st, x, pow2, inv, interval, offset, round, nm, empty);
                                let b = &mut st.buckets[bi];
                                b.1 += 1;
                                for (j, mc) in ms.iter().enumerate() {
                                    if let Some(y) = read(mc, d) { add_metric(&mut b.2[j], y); }
                                }
                            }
                        }
                    }
                }
            }
        };
        let _ = segs;
        let fresh = || HState { index: HashMap::default(), buckets: Vec::new(), dense: Vec::new(), dense_base: None };
        let mut seen = 0;
        let merged = if total >= 2 * part_docs && work.len() > 1 {
            let parts: Vec<HState> = rustic_core::query::searcher::par_map_search(work.len(), |i| { let mut st = fresh(); run(&work[i], &mut st); st });
            let mut g = fresh();
            for st in parts {
                for (k, n, accs) in st.buckets {
                    let i = match g.index.get(&k) {
                        Some(&i) => i,
                        None => { g.buckets.push((k, 0, vec![empty; nm])); g.index.insert(k, g.buckets.len() - 1); g.buckets.len() - 1 }
                    };
                    let b = &mut g.buckets[i];
                    b.1 += n;
                    for (a, p) in b.2.iter_mut().zip(&accs) {
                        a.n += p.n;
                        a.sum.merge(&p.sum);
                        a.min = a.min.min(p.min);
                        a.max = a.max.max(p.max);
                    }
                }
                self.charge_grouping(name, g.buckets.len(), &mut seen, docs.len(), 0)?;
            }
            g
        } else {
            let mut st = fresh();
            for w in &work {
                run(w, &mut st);
                self.charge_grouping(name, st.buckets.len(), &mut seen, docs.len(), 0)?;
            }
            st
        };
        let HState { index, buckets, .. } = merged;
        self.charge_grouping(name, buckets.len(), &mut seen, docs.len(), 0)?;
        let mut keys: Vec<f64> = buckets.iter().map(|b| f64::from_bits(b.0)).collect();
        if min_doc_count == 0 {
            let mut lo = keys.iter().copied().fold(f64::INFINITY, f64::min);
            let mut hi = keys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if let Some((a, b)) = ext {
                if let Some(a) = a { lo = lo.min(round(a)); }
                if let Some(b) = b { hi = hi.max(round(b)); }
            }
            if let Some((a, b)) = hard {
                if let Some(a) = a { lo = lo.max(round(a)); }
                if let Some(b) = b { hi = hi.min(round(b)); }
            }
            if lo.is_finite() && hi.is_finite() {
                let n = ((hi - lo) / interval).round() as i64;
                if n as usize > self.env.max_buckets {
                    return Err(EsError::too_many_buckets(self.env.max_buckets));
                }
                keys = (0..=n).map(|i| lo + i as f64 * interval).collect();
            }
        }
        let no_values = vec![empty; metrics.len()];
        let mut bs: Vec<Bucket> = keys
            .into_iter()
            .map(|k| {
                let (count, accs) = match index.get(&k.to_bits()) {
                    Some(&i) => (buckets[i].1, &buckets[i].2),
                    None => (0, &no_values),
                };
                let mut subs = Map::new();
                for ((sub_name, t, _), a) in metrics.iter().zip(accs) {
                    subs.insert(sub_name.to_string(), simple_metric_json(t, a.n, a.sum.value, a.min, a.max));
                }
                Bucket { key: Key::Dbl(k), count, docs: Vec::new(), subs: Some(subs) }
            })
            .filter(|b| b.count >= min_doc_count)
            .collect();
        self.finish_histogram(o, Some(sub), &mut bs, |b, m| {
            m.insert("key".into(), key_json(&b.key));
        }).map(Some)
    }

    fn finish_histogram(&self, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, bs: &mut Vec<Bucket>, key_fn: impl Fn(&Bucket, &mut Map<String, Value>)) -> Result<Value> {
        let order = parse_order(o.get("order"), &[("_key", true)])?;
        if order.iter().any(|(k, _)| k != "_count" && k != "_key") {
            // (Buckets whose metrics were accumulated while bucketing have
            // them already, and no doc lists.)
            for b in bs.iter_mut().filter(|b| b.subs.is_none()) {
                b.subs = Some(self.subs(sub, &b.docs)?);
            }
        }
        sort_buckets(bs, &order, false);
        self.add_buckets(bs.len())?;
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(false);
        let mut out = Vec::new();
        for b in bs.drain(..) {
            let mut m = Map::new();
            key_fn(&b, &mut m);
            m.insert("doc_count".into(), json!(b.count));
            let subs = match b.subs {
                Some(s) => s,
                None => self.subs(sub, &b.docs)?,
            };
            m.extend(subs);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        if keyed {
            let mut km = Map::new();
            for m in out {
                let k = m.get("key_as_string").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| match &m["key"] {
                    Value::Number(n) => n.as_f64().map(jfmt).unwrap_or_default(),
                    v => v.to_string(),
                });
                km.insert(k, Value::Object(m));
            }
            return Ok(json!({ "buckets": km }));
        }
        Ok(json!({ "buckets": out }))
    }

    /// `sub` as plain numeric metrics a date_histogram can accumulate while
    /// it buckets (no per-bucket doc lists, no per-bucket dispatch): every
    /// sub-aggregation an `avg` / `sum` / `min` / `max` of a numeric
    /// (non-date) field, given by `field` alone (no script, `missing`,
    /// `format`, `meta` or sub-aggregations), outside partial (coordinator
    /// reduce) mode. `None`: compute them per bucket.
    fn fused_metrics(&self, sub: Option<&Map<String, Value>>) -> Result<Option<Vec<FusedMetric>>> {
        let Some(sub) = sub.filter(|m| !m.is_empty()) else { return Ok(None) };
        if partial::active() || DOC_WEIGHTS.with(|w| w.borrow().is_some()) {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(sub.len());
        for (name, def) in sub {
            let Some(def) = def.as_object().filter(|d| d.len() == 1) else { return Ok(None) };
            let (t, body) = def.iter().next().expect("one entry");
            let t: &'static str = match t.as_str() { "avg" => "avg", "sum" => "sum", "min" => "min", "max" => "max", _ => return Ok(None) };
            let Some(body) = body.as_object().filter(|b| b.len() == 1 && b.get("field").is_some_and(Value::is_string)) else { return Ok(None) };
            let field = body["field"].as_str().unwrap_or_default();
            if self.segs.iter().any(|s| crate::script::runtime_field(Some(s.index), field).is_some()) {
                return Ok(None);
            }
            let f = match self.source(body, name, t) {
                Ok(f) => f,
                Err(_) => return Ok(None),
            };
            let numeric = |ty: EsType| matches!(ty, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::UnsignedLong
                | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat);
            if f.src.is_some() || !f.ty.is_some_and(numeric) || f.per_seg.iter().flatten().any(|rf| !numeric(rf.declared) || rf.flattened_key().is_some()) {
                return Ok(None);
            }
            out.push(FusedMetric { name: name.clone(), t, f });
        }
        Ok(Some(out))
    }

    /// The buckets of a date_histogram over `f` (a plain i64 column in every
    /// segment with docs) with `metrics` accumulated per bucket: `key ->
    /// (doc count, one state per metric)`. `None` when some segment lacks a
    /// plain column (the general collector runs instead).
    #[allow(clippy::type_complexity)]
    fn fused_date_buckets(&self, name: &str, f: &FieldRef, rounding: &Rounding, metrics: &[FusedMetric], docs: &Docs)
        -> Result<Option<(HashMap<i64, usize>, Vec<(u64, Vec<FusedAcc>)>)>> {
        let mut index: HashMap<i64, usize> = HashMap::new();
        let mut buckets: Vec<(u64, Vec<FusedAcc>)> = Vec::new();
        let mut memo = Vec::new();
        let mut seen = 0;
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let Some(tc) = self.num_col(f, s) else { return Ok(None) };
            let Col::I64(tc) = &*tc else { return Ok(None) };
            let mut cols = Vec::with_capacity(metrics.len());
            for m in metrics {
                match self.num_col(&m.f, s) {
                    Some(c) => cols.push(Some(c)),
                    // Unmapped here, or no values in this segment.
                    None if m.f.per_seg[s].is_none() || self.col(&m.f, s).is_some_and(|c| matches!(&*c, Col::Empty)) => cols.push(None),
                    None => return Ok(None),
                }
            }
            self.charge_grouping(name, buckets.len(), &mut seen, docs.len(), 0)?;
            // Consecutive docs mostly share a bucket (index-sorted series).
            let mut last: Option<(i64, usize)> = None;
            for &(d, _) in v {
                let Some(x) = tc.get(d) else { continue };
                let k = rounding.round_memo(&mut memo, x);
                let at = match last {
                    Some((lk, at)) if lk == k => at,
                    _ => {
                        let at = *index.entry(k).or_insert_with(|| {
                            buckets.push((0, vec![FusedAcc::default(); metrics.len()]));
                            buckets.len() - 1
                        });
                        last = Some((k, at));
                        at
                    }
                };
                let bucket = &mut buckets[at];
                bucket.0 += 1;
                for (acc, c) in bucket.1.iter_mut().zip(&cols) {
                    let Some(c) = c else { continue };
                    let value = match &**c {
                        Col::I64(c) => c.get(d).map(|x| x as f64),
                        Col::F64(c) => c.get(d),
                        Col::U64(c) => c.get(d).map(|x| x as f64),
                        _ => None,
                    };
                    if let Some(value) = value {
                        acc.add(value);
                    }
                }
            }
        }
        self.charge_grouping(name, buckets.len(), &mut seen, docs.len(), 0)?;
        Ok(Some((index, buckets)))
    }

    fn date_histogram(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.source(o, name, "date_histogram")?;
        // ES 9 `DateHistogramAggregationBuilder` no longer registers the
        // boolean values source.
        if f.src.is_none() && f.ty == Some(EsType::Boolean) {
            let field = o.get("field").and_then(Value::as_str).unwrap_or("");
            return Err(EsError::illegal_argument(format!("Field [{field}] of type [boolean] is not supported for aggregation [date_histogram]")));
        }
        let mut rounding = Rounding::parse(o, name)?;
        // Downsampled indices (`DateHistogramAggregationBuilder.innerBuild`):
        // no calendar intervals; a time zone whose offset is not a multiple
        // of the downsampling interval shifts the buckets by the remainder.
        let mut ds_offset = false;
        let field_s = o.get("field").and_then(Value::as_str).unwrap_or("");
        for s in self.segs {
            let Some(iv) = s.lookup.registry.downsample_interval.as_deref() else { continue };
            if rounding.cal.is_some() {
                let ty = s.lookup.get(field_s).map(|x| x.declared.name()).unwrap_or("date");
                return Err(EsError::illegal_argument(format!("Field [{field_s}] of type [{ty}] is not supported for aggregation [date_histogram] with interval type [calendar_interval]")));
            }
            if !ds_offset && rounding.tz != crate::date::Tz::UTC && field_s == crate::datastream::TIMESTAMP_FIELD {
                if let Some(ds_res) = crate::date::parse_fixed_interval(iv).filter(|r| *r > 0) {
                    let off = (rounding.tz.offset_at(rounding.fixed) as i64 * 1000) % ds_res;
                    if off != 0 {
                        ds_offset = true;
                        rounding.offset += off;
                    }
                }
            }
        }
        let rounding = rounding;
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
        let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0);
        let missing = self.missing_key(&f, o.get("missing"))?;
        let date_bound = |v: &Value, up: bool| -> Result<i64> {
            match v {
                Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) as i64),
                Value::String(s) => parse_date_math(s, &fmt, rounding.tz, self.env.now, up),
                _ => Err(EsError::parsing_at("[extended_bounds] must be numbers or date strings", 1, 1)),
            }
        };
        let hard = match o.get("hard_bounds") {
            Some(h) => Some((h.get("min").map(|v| date_bound(v, false)).transpose()?, h.get("max").map(|v| date_bound(v, false)).transpose()?)),
            None => None,
        };
        let mut groups: HashMap<i64, Docs, ahash::RandomState> = HashMap::default();
        // Without sub-aggregations (and `_doc_count` weights) a bucket only
        // needs its count: no per-bucket doc lists.
        let count_only = sub.is_none_or(|m| m.is_empty()) && DOC_WEIGHTS.with(|w| w.borrow().is_none());
        let mut counts: HashMap<i64, u64, ahash::RandomState> = HashMap::default();
        let mut memo = Vec::new();
        let mut seen = 0;
        let whole = NESTED_WHOLE_PATH.with(|w| w.borrow().clone());
        // Segments whose query matches were left uncollected (a bare range
        // on this field, `with_elided_root`) count from the points index
        // over the query's range; anything else reads them back first.
        let elided: Vec<bool> = (0..docs.len()).map(|s| self.is_elided(docs, s)).collect();
        // Buckets a one-pass terms parent accumulated for this histogram
        // ([`Ctx::terms_one_pass`]).
        let prefused = PREFUSED.with(|p| { let mut p = p.borrow_mut(); if p.as_ref().is_some_and(|(n, _, _)| n == name) { p.take() } else { None } });
        if prefused.is_none() && elided.iter().any(|&e| e) && !(count_only && missing.is_none() && hard.is_none() && whole.is_none()) {
            return self.date_histogram(name, o, sub, &self.collect_elided(&f, docs));
        }
        // Plain numeric metric sub-aggregations accumulate while bucketing.
        let mut fused: Option<(Vec<FusedMetric>, HashMap<i64, usize>, Vec<(u64, Vec<FusedAcc>)>)> = None;
        if let Some((_, index, buckets)) = prefused {
            let metrics = self.fused_metrics(sub)?.ok_or_else(|| EsError::internal("one-pass histogram without plain metrics"))?;
            fused = Some((metrics, index, buckets));
        } else if !count_only && missing.is_none() && hard.is_none() && whole.is_none() && !elided.iter().any(|&e| e) {
            if let Some(metrics) = self.fused_metrics(sub)? {
                if let Some((index, buckets)) = self.fused_date_buckets(name, &f, &rounding, &metrics, docs)? {
                    #[cfg(test)]
                    FUSED_HISTOGRAMS.with(|n| n.set(n.get() + 1));
                    fused = Some((metrics, index, buckets));
                }
            }
        }
        for (s, v) in docs.iter().enumerate() {
            if fused.is_some() {
                break;
            }
            if elided[s] {
                if self.count_whole_by_blocks(&f, s, &rounding, &mut counts) {
                    continue;
                }
                if !self.count_elided_by_points(&f, s, &rounding, &mut counts) {
                    return self.date_histogram(name, o, sub, &self.collect_elided(&f, docs));
                }
                continue;
            }
            // A `nested` parent left this segment's list empty: it is every
            // doc of the path. Count from the points index, or build it.
            let built;
            let v = match whole.as_ref().and_then(|(path, ords)| Some((path, ords.get(s).copied().flatten()?))) {
                Some((path, ord)) if v.is_empty() => {
                    if count_only && missing.is_none() && hard.is_none() && self.count_nested_path_by_points(&f, s, path, &rounding, &mut counts) {
                        continue;
                    }
                    built = self.nested_path_docs(s, ord);
                    &built
                }
                _ => v,
            };
            self.charge_grouping(name, groups.len() + counts.len(), &mut seen, docs.len(), if count_only { 0 } else { v.len() })?;
            // Columnar fast-path: read the i64 column directly, avoiding
            // the per-doc Vec<Key> allocation in values_or.
            if missing.is_none() && hard.is_none() {
                if let Some(col) = self.num_col(&f, s) {
                    if let Col::I64(c) = &*col {
                        // Large scalar count-only requests can prepare the
                        // calendar boundaries once, then count by dense ordinal.
                        // Keep the general collector for small or irregular ranges.
                        if count_only && v.len() >= 4096 && self.count_whole_segment_by_points(c, s, v, &rounding, &mut counts) {
                            continue;
                        }
                        if v.len() >= 4096 {
                            let mut range: Option<(i64, i64)> = None;
                            each_i64(c, v, |_, _, x| range = Some(match range { None => (x, x), Some((lo, hi)) => (lo.min(x), hi.max(x)) }));
                            if let Some(prepared) = range.and_then(|(lo, hi)| PreparedDateBuckets::new(&rounding, lo, hi)) {
                                if count_only {
                                    let mut dense = vec![0u64; prepared.keys.len() - 1];
                                    each_i64(c, v, |_, _, x| dense[prepared.bucket(x)] += 1);
                                    for (i, n) in dense.into_iter().enumerate() {
                                        if n != 0 { *counts.entry(prepared.keys[i]).or_default() += n; }
                                    }
                                } else {
                                    // Each bucket's docs in doc order, by
                                    // dense bucket index (a TSDB segment's docs
                                    // come in `_tsid` order, so timestamps, and
                                    // buckets, interleave).
                                    let mut dense: Vec<Vec<(DocId, f32)>> = vec![Vec::new(); prepared.keys.len() - 1];
                                    each_i64(c, v, |d, sc, x| dense[prepared.bucket(x)].push((d, sc)));
                                    for (i, docs_here) in dense.into_iter().enumerate() {
                                        if !docs_here.is_empty() {
                                            groups.entry(prepared.keys[i]).or_insert_with(|| vec![Vec::new(); docs.len()])[s] = docs_here;
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                        each_i64(c, v, |d, sc, x| {
                            let k = rounding.round_memo(&mut memo, x);
                            if count_only { *counts.entry(k).or_default() += 1; }
                            else { groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc)); }
                        });
                        continue;
                    }
                }
            }
            if let Some(col) = missing.is_none().then(|| self.num_col(&f, s)).flatten() {
                macro_rules! fill {
                    ($c:expr) => {
                        for &(d, sc) in v {
                            let mut ks: Vec<i64> = $c
                                .get(d)
                                .into_iter()
                                .filter(|x| hard.map(|(a, b)| a.map(|a| *x >= a).unwrap_or(true) && b.map(|b| *x <= b).unwrap_or(true)).unwrap_or(true))
                                .map(|x| rounding.round_memo(&mut memo, x))
                                .collect();
                            ks.sort();
                            ks.dedup();
                            for k in ks {
                                if count_only { *counts.entry(k).or_default() += 1; } else { groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc)); }
                            }
                        }
                    };
                }
                match &*col {
                    Col::I64(c) => fill!(c),
                    Col::F64(c) => {
                        for &(d, sc) in v {
                            let mut ks: Vec<i64> = c
                                .get(d)
                                .into_iter()
                                .map(|x| x as i64)
                                .filter(|x| hard.map(|(a, b)| a.map(|a| *x >= a).unwrap_or(true) && b.map(|b| *x <= b).unwrap_or(true)).unwrap_or(true))
                                .map(|x| rounding.round_memo(&mut memo, x))
                                .collect();
                            ks.sort();
                            ks.dedup();
                            for k in ks {
                                if count_only { *counts.entry(k).or_default() += 1; } else { groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc)); }
                            }
                        }
                    }
                    Col::U64(c) => {
                        for &(d, sc) in v {
                            let mut ks: Vec<i64> = c
                                .get(d)
                                .into_iter()
                                .map(|x| x as i64)
                                .filter(|x| hard.map(|(a, b)| a.map(|a| *x >= a).unwrap_or(true) && b.map(|b| *x <= b).unwrap_or(true)).unwrap_or(true))
                                .map(|x| rounding.round_memo(&mut memo, x))
                                .collect();
                            ks.sort();
                            ks.dedup();
                            for k in ks {
                                if count_only { *counts.entry(k).or_default() += 1; } else { groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc)); }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                continue;
            }
            for &(d, sc) in v {
                let mut ks: Vec<i64> = self
                    .values_or(&f, s, d, &missing)
                    .iter()
                    .filter_map(|k| match k {
                        Key::Long(x) => Some(*x),
                        k => k.as_f64().map(|x| x as i64),
                    })
                    .filter(|x| hard.map(|(a, b)| a.map(|a| *x >= a).unwrap_or(true) && b.map(|b| *x <= b).unwrap_or(true)).unwrap_or(true))
                    .map(|x| rounding.round_memo(&mut memo, x))
                    .collect();
                ks.sort();
                ks.dedup();
                for k in ks {
                    groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc));
                }
            }
        }
        self.charge_grouping(name, groups.len() + counts.len(), &mut seen, docs.len(), 0)?;
        let mut keys: Vec<i64> = match &fused {
            Some((_, index, _)) => index.keys().copied().collect(),
            None => groups.keys().chain(counts.keys().filter(|k| !groups.contains_key(k))).copied().collect(),
        };
        keys.sort();
        if min_doc_count == 0 {
            let mut lo = keys.first().copied();
            let mut hi = keys.last().copied();
            if let Some(e) = o.get("extended_bounds") {
                if let Some(v) = e.get("min") {
                    let k = rounding.round(date_bound(v, false)?);
                    lo = Some(lo.map(|l| l.min(k)).unwrap_or(k));
                }
                if let Some(v) = e.get("max") {
                    let k = rounding.round(date_bound(v, false)?);
                    hi = Some(hi.map(|h| h.max(k)).unwrap_or(k));
                }
            }
            if let (Some(lo), Some(hi)) = (lo, hi) {
                let mut all = Vec::new();
                let mut k = lo;
                while k <= hi {
                    all.push(k);
                    if all.len() > self.env.max_buckets {
                        return Err(EsError::too_many_buckets(self.env.max_buckets));
                    }
                    let next = rounding.next(k);
                    if next <= k {
                        break;
                    }
                    k = next;
                }
                keys = all;
            }
        }
        let mut bs: Vec<Bucket> = match &fused {
            Some((metrics, index, buckets)) => keys
                .into_iter()
                .map(|k| match index.get(&k) {
                    Some(&at) => {
                        let (n, accs) = &buckets[at];
                        let mut subs = Map::new();
                        for (m, a) in metrics.iter().zip(accs) {
                            subs.insert(m.name.clone(), simple_metric_json(m.t, a.n, a.sum.value, a.min, a.max));
                        }
                        Bucket { key: Key::Long(k), count: *n, docs: vec![Vec::new(); docs.len()], subs: Some(subs) }
                    }
                    // An empty bucket (min_doc_count 0): its metrics over no docs.
                    None => Bucket { key: Key::Long(k), count: 0, docs: vec![Vec::new(); docs.len()], subs: None },
                })
                .filter(|b| b.count >= min_doc_count)
                .collect(),
            None => keys
                .into_iter()
                .map(|k| {
                    let d = groups.remove(&k).unwrap_or_else(|| vec![Vec::new(); docs.len()]);
                    Bucket { key: Key::Long(k), count: count(&d) + counts.get(&k).copied().unwrap_or(0), docs: d, subs: None }
                })
                .filter(|b| b.count >= min_doc_count)
                .collect(),
        };
        let tz = rounding.tz;
        let _rate_parent = extra::DhGuard::push(field_name(o, name, "date_histogram")?, &rounding);
        let mut out = self.finish_histogram(o, sub, &mut bs, |b, m| {
            if let Key::Long(k) = b.key {
                m.insert("key_as_string".into(), json!(fmt.format(k, tz)));
                m.insert("key".into(), json!(k));
            }
        })?;
        if ds_offset {
            if let Some(m) = out.as_object_mut() {
                m.insert("downsampled_results_offset".into(), json!(true));
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // composite.

    fn composite(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let size = o.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
        let sources = o.get("sources").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("Composite [sources] cannot be null or empty")))?;
        if sources.is_empty() {
            return Err(EsError::illegal_argument("Composite [sources] cannot be null or empty"));
        }
        struct Src {
            name: String,
            field: FieldRef,
            kind: u8, // 0 terms, 1 histogram, 2 date_histogram
            interval: f64,
            rounding: Option<Rounding>,
            desc: bool,
            /// ES `MissingOrder`: nulls first ascending, last descending,
            /// unless `missing_order` says.
            nulls_first: bool,
            missing_bucket: bool,
            fmt: Option<DateFormatter>,
            fname: String,
        }
        let mut srcs = Vec::new();
        for s in sources {
            let so = s.as_object().filter(|m| m.len() == 1).ok_or_else(|| EsError::parsing_at("[sources] entries must be single-key objects", 1, 1))?;
            let (sname, def) = so.iter().next().unwrap();
            let dobj = def.as_object().filter(|m| m.len() == 1).ok_or_else(|| EsError::parsing_at(format!("[{sname}] source must have exactly one type"), 1, 1))?;
            let (kind, spec) = dobj.iter().next().unwrap();
            let spec = spec.as_object().ok_or_else(|| EsError::parsing_at("source must be an object", 1, 1))?;
            let field = self.source(spec, name, "composite")?;
            let k = match kind.as_str() {
                "terms" => 0,
                "histogram" => 1,
                "date_histogram" => 2,
                other => return Err(EsError::parsing_at(format!("Unknown value source [{other}] in composite aggregation [{name}]"), 1, 1)),
            };
            srcs.push(Src {
                name: sname.clone(),
                interval: spec.get("interval").and_then(Value::as_f64).unwrap_or(1.0),
                rounding: if k == 2 { Some(Rounding::parse(spec, name)?) } else { None },
                desc: spec.get("order").and_then(Value::as_str) == Some("desc"),
                nulls_first: match spec.get("missing_order").and_then(Value::as_str) {
                    Some("first") => true,
                    Some("last") => false,
                    _ => spec.get("order").and_then(Value::as_str) != Some("desc"),
                },
                missing_bucket: spec.get("missing_bucket").and_then(Value::as_bool).unwrap_or(false),
                fmt: spec.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?,
                fname: spec.get("field").and_then(Value::as_str).unwrap_or_default().to_string(),
                field,
                kind: k,
            });
        }
        let cmp = |a: &Vec<Key>, b: &Vec<Key>| -> Ordering {
            for (i, s) in srcs.iter().enumerate() {
                let c = match (&a[i], &b[i]) {
                    (Key::Null, Key::Null) => Ordering::Equal,
                    (Key::Null, _) => if s.nulls_first { Ordering::Less } else { Ordering::Greater },
                    (_, Key::Null) => if s.nulls_first { Ordering::Greater } else { Ordering::Less },
                    (x, y) if s.desc => cmp_key(x, y).reverse(),
                    (x, y) => cmp_key(x, y),
                };
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        };
        let mut groups: HashMap<Vec<Key>, Docs> = HashMap::new();
        let mut memos: Vec<Vec<(i64, i64)>> = vec![Vec::new(); srcs.len()];
        for (s, v) in docs.iter().enumerate() {
            'doc: for &(d, sc) in v {
                let mut per: Vec<Vec<Key>> = Vec::new();
                for (si, src) in srcs.iter().enumerate() {
                    let memo = &mut memos[si];
                    let mut ks: Vec<Key> = self
                        .values(&src.field, s, d)
                        .into_iter()
                        .map(|k| match src.kind {
                            1 => Key::Dbl((k.as_f64().unwrap_or(0.0) / src.interval).floor() * src.interval),
                            2 => Key::Long(src.rounding.as_ref().unwrap().round_memo(memo, match k {
                                Key::Long(x) => x,
                                k => k.as_f64().unwrap_or(0.0) as i64,
                            })),
                            _ => k,
                        })
                        .collect();
                    ks.dedup();
                    if ks.is_empty() {
                        if src.missing_bucket {
                            ks.push(Key::Null);
                        } else {
                            continue 'doc;
                        }
                    }
                    per.push(ks);
                }
                let mut combos: Vec<Vec<Key>> = vec![Vec::new()];
                for ks in per {
                    combos = combos.into_iter().flat_map(|c| ks.iter().map(move |k| { let mut c = c.clone(); c.push(k.clone()); c })).collect();
                }
                for c in combos {
                    let kb = c.capacity() * std::mem::size_of::<Key>() + c.iter().map(Key::heap_bytes).sum::<usize>();
                    self.push_grouped(name, &mut groups, c, kb, docs.len(), s, (d, sc))?;
                }
            }
        }
        let key_of_json = |src: &Src, v: &Value| -> Key {
            match v {
                Value::Null => Key::Null,
                Value::Bool(b) => Key::Bool(*b),
                Value::Number(n) if src.kind == 1 => Key::Dbl(n.as_f64().unwrap_or(0.0)),
                Value::Number(n) if n.is_i64() => Key::Long(n.as_i64().unwrap()),
                Value::Number(n) if n.is_u64() => Key::ULong(n.as_u64().unwrap()),
                Value::Number(n) => Key::Dbl(n.as_f64().unwrap_or(0.0)),
                Value::String(s) if src.kind == 2 => Key::Long(src.fmt.clone().or_else(|| src.field.format.clone()).unwrap_or_default().parse(s).unwrap_or(0)),
                Value::String(s) => Key::Str(s.clone()),
                other => Key::Str(other.to_string()),
            }
        };
        let after: Option<Vec<Key>> = match o.get("after") {
            Some(Value::Object(a)) => Some(srcs.iter().map(|s| key_of_json(s, a.get(&s.name).unwrap_or(&Value::Null))).collect()),
            _ => None,
        };
        let mut keys: Vec<Vec<Key>> = groups.keys().cloned().filter(|k| after.as_ref().map(|a| cmp(k, a) == Ordering::Greater).unwrap_or(true)).collect();
        keys.sort_by(&cmp);
        keys.truncate(size);
        self.add_buckets(keys.len())?;
        let key_json_src = |src: &Src, k: &Key| -> Value {
            match (src.kind, k) {
                (2, Key::Long(x)) if src.fmt.is_some() => json!(src.fmt.as_ref().unwrap().format(*x, src.rounding.as_ref().map(|r| r.tz).unwrap_or(Tz::UTC))),
                (_, Key::Bool(b)) => json!(b),
                (_, k) => key_json(k),
            }
        };
        let mut out = Vec::new();
        let mut last = None;
        // `rate` under a composite with exactly one date_histogram source.
        let date_srcs: Vec<&Src> = srcs.iter().filter(|s| s.kind == 2).collect();
        let _rate_parent = match date_srcs.as_slice() {
            [s] => s.rounding.as_ref().map(|r| extra::DhGuard::push(&s.fname, r)),
            _ => None,
        };
        for k in keys {
            let d = groups.remove(&k).unwrap();
            let mut key = Map::new();
            for (i, s) in srcs.iter().enumerate() {
                key.insert(s.name.clone(), key_json_src(s, &k[i]));
            }
            let mut m = Map::new();
            m.insert("key".into(), Value::Object(key.clone()));
            m.insert("doc_count".into(), json!(count(&d)));
            m.extend(self.subs(sub, &d)?);
            out.push(m);
            last = Some(key);
        }
        apply_parent_pipelines(sub, &mut out)?;
        let out: Vec<Value> = out.into_iter().map(Value::Object).collect();
        let mut r = Map::new();
        if let Some(l) = last {
            r.insert("after_key".into(), Value::Object(l));
        }
        r.insert("buckets".into(), Value::Array(out));
        Ok(Value::Object(r))
    }

    // -----------------------------------------------------------------------
    // top_hits.

    fn top_hits(&self, spec: &Value, docs: &Docs) -> Result<Value> {
        let o = spec.as_object().cloned().unwrap_or_default();
        let size = o.get("size").and_then(Value::as_u64).unwrap_or(3) as usize;
        let from = o.get("from").and_then(Value::as_u64).unwrap_or(0) as usize;
        if from + size > 100 {
            return Err(EsError::illegal_argument(format!(
                "Top hits result window is too large, the top hits aggregator [top_hits]'s from + size must be less than or equal to: [100] but was [{}]. This limit can be set by changing the [index.max_inner_result_window] index level setting.",
                from + size
            )));
        }
        // Sort criteria: (field or _score, descending).
        let mut crit: Vec<(String, bool)> = Vec::new();
        if let Some(s) = o.get("sort") {
            let items: Vec<Value> = match s {
                Value::Array(a) => a.clone(),
                other => vec![other.clone()],
            };
            for it in items {
                match it {
                    Value::String(f) => crit.push((f.clone(), f == "_score")),
                    Value::Object(m) => {
                        for (f, v) in m {
                            let desc = match &v {
                                Value::String(s) => s == "desc",
                                Value::Object(x) => x.get("order").and_then(Value::as_str).map(|s| s == "desc").unwrap_or(f == "_score"),
                                _ => false,
                            };
                            crit.push((f.clone(), desc));
                        }
                    }
                    _ => {}
                }
            }
        }
        let by_score = crit.is_empty() || (crit.len() == 1 && crit[0].0 == "_score");
        let fields: Vec<Option<FieldRef>> = crit.iter().map(|(f, _)| if f == "_score" || f == "_doc" { None } else { self.field(f).ok() }).collect();
        let mut all: Vec<(usize, DocId, f32, Vec<Key>)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, sc) in v {
                let keys = crit
                    .iter()
                    .zip(&fields)
                    .map(|((f, desc), fr)| match (f.as_str(), fr) {
                        ("_score", _) => Key::Dbl(sc as f64),
                        ("_doc", _) => Key::Long(d as i64),
                        (_, Some(fr)) => {
                            let mut vs = self.values(fr, s, d);
                            vs.sort_by(cmp_key);
                            (if *desc { vs.pop() } else { vs.into_iter().next() }).unwrap_or(Key::Null)
                        }
                        _ => Key::Null,
                    })
                    .collect();
                all.push((s, d, sc, keys));
            }
        }
        let total = all.len();
        if by_score {
            all.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        } else {
            all.sort_by(|a, b| {
                for (i, (_, desc)) in crit.iter().enumerate() {
                    let (x, y) = (&a.3[i], &b.3[i]);
                    let c = match (x, y) {
                        (Key::Null, Key::Null) => Ordering::Equal,
                        (Key::Null, _) => Ordering::Greater,
                        (_, Key::Null) => Ordering::Less,
                        _ => {
                            let c = cmp_key(x, y);
                            if *desc { c.reverse() } else { c }
                        }
                    };
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                a.0.cmp(&b.0).then(a.1.cmp(&b.1))
            });
        }
        let max_score = if by_score { all.first().map(|h| fnum(h.2 as f64)).unwrap_or(Value::Null) } else { Value::Null };
        let mut hits = Vec::new();
        for (s, d, sc, keys) in all.into_iter().skip(from).take(size) {
            let mut h = (self.env.fetch_hit)(s, d, sc, spec)?;
            if !by_score {
                if let Value::Object(m) = &mut h {
                    m.insert("_score".into(), Value::Null);
                    m.insert("sort".into(), Value::Array(keys.iter().map(key_json).collect()));
                }
            }
            hits.push(h);
        }
        Ok(json!({ "hits": { "total": { "value": total, "relation": "eq" }, "max_score": max_score, "hits": hits } }))
    }

    // -----------------------------------------------------------------------
    // Metrics.

    fn metric(&self, name: &str, t: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        // Geo value_count counts point values, not decoded numeric coordinates.
        // Keep scripts/runtime overrides on their normal value-source path.
        if t == "value_count" && !o.contains_key("script") {
            if let Some(field) = o.get("field").and_then(Value::as_str) {
                let runtime = self.segs.iter().any(|s| crate::script::runtime_field(Some(s.index), field).is_some());
                let mapped_geo = self.segs.iter().any(|s| s.lookup.get(field).is_some_and(|f| f.ty == EsType::GeoPoint));
                if !runtime && mapped_geo {
                    return self.geo_value_count(field, o, docs);
                }
            }
        }
        let f = self.source(o, name, t)?;
        let missing = self.missing_key(&f, o.get("missing"))?;
        let is_date = f.ty.map(|t| t.is_date()).unwrap_or(false);
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
        let as_string = |v: f64| -> Option<Value> { if is_date { Some(json!(fmt.format(v as i64, Tz::UTC))) } else { None } };
        if t == "cardinality" {
            let thr = o.get("precision_threshold").and_then(Value::as_u64).unwrap_or(3000) as usize;
            let mut hll = HyperLogLog::new(thr);
            self.add_value_hashes(&f, docs, &missing, &mut hll);
            return Ok(json!({ "value": hll.count() }));
        }
        if t == "value_count" && missing.is_none() {
            // A single-valued numeric column: count the docs with a value,
            // without listing each doc's values (one per such doc).
            let mut n = 0u64;
            let mut plain = true;
            for (s, v) in docs.iter().enumerate() {
                if v.is_empty() {
                    continue;
                }
                let Some(c) = self.num_col(&f, s) else { plain = false; break };
                n += match &*c {
                    Col::I64(c) => if c.all_present() { v.len() as u64 } else { v.iter().filter(|&&(d, _)| c.has_value(d)).count() as u64 },
                    Col::F64(c) => if c.all_present() { v.len() as u64 } else { v.iter().filter(|&&(d, _)| c.has_value(d)).count() as u64 },
                    Col::U64(c) => if c.all_present() { v.len() as u64 } else { v.iter().filter(|&&(d, _)| c.has_value(d)).count() as u64 },
                    _ => { plain = false; break }
                };
            }
            if plain {
                return Ok(json!({ "value": n }));
            }
        }
        if t == "value_count" {
            let mut n = 0u64;
            let thr = o.get("precision_threshold").and_then(Value::as_u64).unwrap_or(3000) as usize;
            let mut hll = HyperLogLog::new(thr);
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    for k in self.values_or(&f, s, d, &missing) {
                        n += 1;
                        if t == "cardinality" {
                            hll.add_bytes(&k.hash_bytes());
                        }
                    }
                }
            }
            return Ok(json!({ "value": if t == "cardinality" { hll.count() } else { n } }));
        }
        if !is_date && matches!(t, "avg" | "sum" | "min" | "max" | "stats") {
            if let Some((n, sum, min, max)) = self.fold_simple(&f, docs, &missing)? {
                return Ok(simple_metric_json(t, n, sum, min, max));
            }
        }
        let vals = self.num_values(&f, docs, &missing)?;
        let n = vals.len();
        // ES `SumAggregator` / `StatsAggregator` / `AvgAggregator` sum with
        // `CompensatedSum` (Kahan): the plain sum of millions of doubles
        // drifted in the last digits (Rally nyc_taxis `distance_amount_agg`).
        let sum: f64 = {
            let mut k = extra::Kahan::default();
            for &v in &vals {
                k.add(v);
            }
            k.value
        };
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let with_str = |mut m: Map<String, Value>, key: &str, v: Option<f64>| {
            if let Some(v) = v {
                if let Some(s) = as_string(v) {
                    m.insert(format!("{key}_as_string"), s);
                }
            }
            m
        };
        let single = |v: Option<f64>| -> Value {
            let mut m = Map::new();
            m.insert("value".into(), opt_num(v));
            Value::Object(with_str(m, "value", v))
        };
        if !is_date && matches!(t, "avg" | "sum" | "min" | "max" | "stats") {
            return Ok(simple_metric_json(t, n, sum, min, max));
        }
        Ok(match t {
            "avg" => single((n > 0).then(|| sum / n as f64)),
            "sum" => {
                let mut m = Map::new();
                m.insert("value".into(), fnum(sum));
                Value::Object(with_str(m, "value", (n > 0).then_some(sum)))
            }
            "min" => single((n > 0).then_some(min)),
            "max" => single((n > 0).then_some(max)),
            "stats" | "extended_stats" => {
                let mut m = Map::new();
                m.insert("count".into(), json!(n));
                m.insert("min".into(), opt_num((n > 0).then_some(min)));
                m.insert("max".into(), opt_num((n > 0).then_some(max)));
                m.insert("avg".into(), opt_num((n > 0).then(|| sum / n as f64)));
                m.insert("sum".into(), fnum(sum));
                if t == "extended_stats" {
                    let sigma = o.get("sigma").and_then(Value::as_f64).unwrap_or(2.0);
                    if sigma < 0.0 {
                        return Err(EsError::illegal_argument(format!("[sigma] must be greater than or equal to 0. Found [{sigma}] in [{name}]")));
                    }
                    let sos: f64 = vals.iter().map(|x| x * x).sum();
                    let nf = n as f64;
                    let avg = sum / nf;
                    let var_p = (sos / nf - avg * avg).max(0.0);
                    let var_s = if n > 1 { (sos - nf * avg * avg).max(0.0) / (nf - 1.0) } else { f64::NAN };
                    let ok = n > 0;
                    let o2 = |v: f64| if ok && v.is_finite() { fnum(v) } else { Value::Null };
                    m.insert("sum_of_squares".into(), if ok { fnum(sos) } else { Value::Null });
                    m.insert("variance".into(), o2(var_p));
                    m.insert("variance_population".into(), o2(var_p));
                    m.insert("variance_sampling".into(), o2(var_s));
                    m.insert("std_deviation".into(), o2(var_p.sqrt()));
                    m.insert("std_deviation_population".into(), o2(var_p.sqrt()));
                    m.insert("std_deviation_sampling".into(), o2(var_s.sqrt()));
                    m.insert(
                        "std_deviation_bounds".into(),
                        json!({
                            "upper": o2(avg + sigma * var_p.sqrt()),
                            "lower": o2(avg - sigma * var_p.sqrt()),
                            "upper_population": o2(avg + sigma * var_p.sqrt()),
                            "lower_population": o2(avg - sigma * var_p.sqrt()),
                            "upper_sampling": o2(avg + sigma * var_s.sqrt()),
                            "lower_sampling": o2(avg - sigma * var_s.sqrt()),
                        }),
                    );
                }
                if n > 0 && is_date {
                    for k in ["min", "max", "avg", "sum"] {
                        let v = m[k].as_f64().unwrap_or(0.0);
                        m.insert(format!("{k}_as_string"), as_string(v).unwrap());
                    }
                }
                Value::Object(m)
            }
            "percentiles" | "percentile_ranks" => {
                let comp = o.get("tdigest").and_then(|t| t.get("compression")).and_then(Value::as_f64).unwrap_or(100.0);
                let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(true);
                let mut td = TDigest::new(comp);
                let mut hdr = match o.get("hdr") {
                    Some(h) => {
                        let d = h.get("number_of_significant_value_digits").and_then(Value::as_u64).unwrap_or(3);
                        if d > 5 {
                            return Err(EsError::illegal_argument("numberOfSignificantValueDigits must be between 0 and 5"));
                        }
                        Some(Hdr::new(d as u32))
                    }
                    None => None,
                };
                let mut sketch_bytes = 0;
                for v in &vals {
                    match &mut hdr {
                        Some(h) => {
                            self.reserve_sketch(h.working_bytes(), &mut sketch_bytes)?;
                            h.add(*v);
                        }
                        None => {
                            self.reserve_sketch(td.working_bytes(), &mut sketch_bytes)?;
                            td.add(*v);
                        }
                    }
                }
                let points: Vec<f64> = if t == "percentiles" {
                    match o.get("percents") {
                        Some(Value::Array(a)) => a.iter().filter_map(Value::as_f64).collect(),
                        _ => vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0],
                    }
                } else {
                    o.get("values").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).ok_or_else(|| EsError::illegal_argument(format!("[values] must not be null: [{name}]")))?
                };
                for p in &points {
                    if t == "percentiles" && !(0.0..=100.0).contains(p) {
                        return Err(EsError::illegal_argument(format!("percent must be in [0,100], got [{p}]: [{name}]")));
                    }
                }
                let mut entries: Vec<(f64, Option<f64>)> = Vec::new();
                for p in points {
                    let v = match (&hdr, t == "percentiles") {
                        (_, _) if n == 0 => None,
                        (Some(h), true) => h.value_at_percentile(p),
                        (Some(h), false) => h.percentile_at_or_below(p),
                        (None, true) => td.quantile(p / 100.0),
                        (None, false) => td.cdf(p).map(|c| c * 100.0),
                    };
                    entries.push((p, v));
                }
                if keyed {
                    let mut m = Map::new();
                    for (p, v) in &entries {
                        m.insert(jfmt(*p), opt_num(*v));
                        if let (Some(v), true) = (v, is_date && t == "percentiles") {
                            m.insert(format!("{}_as_string", jfmt(*p)), as_string(*v).unwrap());
                        }
                    }
                    json!({ "values": m })
                } else {
                    json!({ "values": entries.iter().map(|(p, v)| json!({"key": fnum(*p), "value": opt_num(*v)})).collect::<Vec<_>>() })
                }
            }
            _ => return Err(EsError::parsing_at(format!("Unknown aggregation type [{t}]"), 1, 1)),
        })
    }
}

fn field_name<'o>(o: &'o Map<String, Value>, name: &str, t: &str) -> Result<&'o str> {
    o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument(format!("Required one of fields [field, script], but none were specified. [{t}] aggregation [{name}]")))
}

/// Bucket doc count: the docs' `_doc_count` (default 1 each).
/// The field of a query that is exactly one `exists` (`{"exists":
/// {"field": F}}`, possibly wrapped in `bool`s holding nothing else: one
/// `must` / `filter` clause, or one `should` clause with
/// `minimum_should_match` 1 or unset, and no `must_not`).
fn exists_field(query: &Value) -> Option<&str> {
    let (kind, body) = query.as_object().filter(|q| q.len() == 1)?.iter().next()?;
    let body = body.as_object()?;
    match kind.as_str() {
        "exists" => (body.len() == 1).then(|| body.get("field")?.as_str()).flatten(),
        "bool" => {
            let clauses = |k: &str| -> Option<Vec<&Value>> {
                Some(match body.get(k) {
                    None => Vec::new(),
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(v @ Value::Object(_)) => vec![v],
                    Some(_) => return None,
                })
            };
            if body.keys().any(|k| !matches!(k.as_str(), "must" | "filter" | "should" | "must_not" | "minimum_should_match")) {
                return None;
            }
            let (must, filter, should, must_not) = (clauses("must")?, clauses("filter")?, clauses("should")?, clauses("must_not")?);
            if !must_not.is_empty() {
                return None;
            }
            let required: Vec<&Value> = must.into_iter().chain(filter).collect();
            let msm_one = match body.get("minimum_should_match") {
                None => true,
                Some(Value::Number(n)) => n.as_i64() == Some(1),
                Some(Value::String(s)) => s == "1",
                Some(_) => false,
            };
            match (required.as_slice(), should.as_slice()) {
                ([one], []) if body.get("minimum_should_match").is_none() => exists_field(one),
                ([], [one]) if msm_one => exists_field(one),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The sub-aggregations a bucket order names (each order key's first
/// path element), when they are fewer than all of `sub` and all of them are
/// plain aggregations (no pipelines: those read other siblings). `None`:
/// rank with every sub-aggregation.
fn order_children(sub: Option<&Map<String, Value>>, order: &[(String, bool)]) -> Option<Map<String, Value>> {
    let sub = sub?;
    let mut picked = Map::new();
    for (key, _) in order {
        if matches!(key.as_str(), "_count" | "_key" | "_term") {
            continue;
        }
        let head = key.split(['>', '.', '[']).next().unwrap_or(key);
        let def = sub.get(head)?;
        let is_pipeline = def.as_object()?.keys().any(|k| PARENT_PIPELINES.contains(&k.as_str()) || SIBLING_PIPELINES.contains(&k.as_str()));
        if is_pipeline {
            return None;
        }
        picked.insert(head.to_string(), def.clone());
    }
    (!picked.is_empty() && picked.len() < sub.len()).then_some(picked)
}

thread_local! {
    /// A histogram's buckets accumulated by its one-pass terms parent, for
    /// [`Ctx::date_histogram`] to render (name, key -> slot, slots).
    #[allow(clippy::type_complexity)]
    static PREFUSED: std::cell::RefCell<Option<(String, HashMap<i64, usize>, Vec<(u64, Vec<FusedAcc>)>)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Tests: date_histograms that accumulated their metrics while bucketing.
    static FUSED_HISTOGRAMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Tests: terms aggregations collected in one pass.
    static ONE_PASS_TERMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Tests: compute every `range` per range ([`Ctx::range_auto_date_fused`] off).
    pub(crate) static NO_FUSED_RANGE: Cell<bool> = const { Cell::new(false) };
    /// Tests: ranges [`Ctx::range_auto_date_fused`] folded.
    pub(crate) static FUSED_RANGES: Cell<usize> = const { Cell::new(0) };
}

/// One plain numeric metric sub-aggregation of a fused date_histogram
/// ([`Ctx::fused_metrics`]).
struct FusedMetric {
    name: String,
    t: &'static str,
    f: FieldRef,
}

/// A fused metric's running state over one bucket: what [`Ctx::metric`]
/// derives from the bucket's values (count, compensated sum in collection
/// order, min, max).
#[derive(Clone, Copy)]
struct FusedAcc {
    n: usize,
    sum: extra::Kahan,
    min: f64,
    max: f64,
}

impl Default for FusedAcc {
    fn default() -> Self {
        FusedAcc { n: 0, sum: extra::Kahan::default(), min: f64::INFINITY, max: f64::NEG_INFINITY }
    }
}

impl FusedAcc {
    #[inline]
    fn add(&mut self, v: f64) {
        self.n += 1;
        self.sum.add(v);
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
}

fn count(d: &Docs) -> u64 {
    DOC_WEIGHTS.with(|w| match &*w.borrow() {
        None => d.iter().map(|v| v.len() as u64).sum(),
        Some(ws) => d
            .iter()
            .enumerate()
            .map(|(i, v)| match ws.get(i).and_then(Option::as_ref) {
                None => v.len() as u64,
                Some(seg) => v.iter().map(|(doc, _)| seg.get(*doc as usize).copied().unwrap_or(1) as u64).sum(),
            })
            .sum(),
    })
}

fn key_json(k: &Key) -> Value {
    match k {
        Key::Null => Value::Null,
        Key::Bool(b) => json!(*b as i64),
        Key::Long(i) => json!(i),
        Key::ULong(u) => json!(*u),
        Key::Dbl(f) => fnum(*f),
        Key::Str(s) => json!(s),
        Key::Ip(b) => json!(values::ip_display(&values::ip_to_term(b))),
    }
}

fn bounds(v: Option<&Value>) -> Option<(Option<f64>, Option<f64>)> {
    let o = v?.as_object()?;
    Some((o.get("min").and_then(Value::as_f64), o.get("max").and_then(Value::as_f64)))
}

struct Bucket {
    key: Key,
    count: u64,
    docs: Docs,
    subs: Option<Map<String, Value>>,
}

/// `order` → `(path, ascending)` criteria; `_key` asc tiebreak appended.
fn parse_order(v: Option<&Value>, default: &[(&str, bool)]) -> Result<Vec<(String, bool)>> {
    let mut out = Vec::new();
    let items: Vec<&Value> = match v {
        None => return Ok(default.iter().map(|(k, a)| (k.to_string(), *a)).chain(std::iter::once(("_key".to_string(), true))).collect()),
        Some(Value::Array(a)) => a.iter().collect(),
        Some(o) => vec![o],
    };
    for it in items {
        let m = it.as_object().ok_or_else(|| EsError::parsing_at("[order] must be an object", 1, 1))?;
        for (k, dir) in m {
            let asc = match dir.as_str() {
                Some("asc") => true,
                Some("desc") => false,
                _ => return Err(EsError::parsing_at(format!("Unknown order direction [{dir}]"), 1, 1)),
            };
            let k = if k == "_term" { "_key".to_string() } else { k.clone() };
            out.push((k, asc));
        }
    }
    if !out.iter().any(|(k, _)| k == "_key") {
        out.push(("_key".into(), true));
    }
    Ok(out)
}

/// `_tsid` keys are binary terms: order by the decoded bytes, not the base64 text.
fn cmp_tsid_key(a: &Key, b: &Key) -> Ordering {
    use base64::Engine;
    match (a, b) {
        (Key::Str(x), Key::Str(y)) => {
            let d = |s: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).unwrap_or_else(|_| s.as_bytes().to_vec());
            d(x).cmp(&d(y))
        }
        _ => cmp_key(a, b),
    }
}

fn sort_buckets(bs: &mut [Bucket], order: &[(String, bool)], tsid: bool) {
    bs.sort_by(|a, b| {
        for (k, asc) in order {
            let c = match k.as_str() {
                "_count" => a.count.cmp(&b.count),
                "_key" if tsid => cmp_tsid_key(&a.key, &b.key),
                "_key" => cmp_key(&a.key, &b.key),
                path => {
                    let mk = |x: &Bucket| -> Option<Key> {
                        let mut m = x.subs.clone().unwrap_or_default();
                        m.insert("doc_count".into(), json!(x.count));
                        resolve_path(&m, path).map(Key::Dbl).or_else(|| resolve_path_str(&m, path).map(Key::Str))
                    };
                    match (mk(a), mk(b)) {
                        (Some(x), Some(y)) => cmp_key(&x, &y),
                        (None, Some(_)) => return Ordering::Greater,
                        (Some(_), None) => return Ordering::Less,
                        _ => Ordering::Equal,
                    }
                }
            };
            let c = if *asc { c } else { c.reverse() };
            if c != Ordering::Equal {
                return c;
            }
        }
        Ordering::Equal
    });
}

/// include/exclude: regex string (whole-term) or exact values.
enum Matcher {
    Re(regex::Regex),
    Exact(HashSet<String>),
}

impl Matcher {
    fn parse(v: Option<&Value>) -> Result<Option<Matcher>> {
        Ok(match v {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(Matcher::Re(regex::Regex::new(&format!("^(?:{s})$")).map_err(|e| EsError::illegal_argument(format!("invalid regex [{s}]: {e}")))?)),
            Some(Value::Array(a)) => Some(Matcher::Exact(a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect())),
            Some(Value::Object(_)) => None,
            Some(other) => return Err(EsError::parsing_at(format!("Unrecognized token for an include [{other}]"), 1, 1)),
        })
    }
    fn matches(&self, k: &Key) -> bool {
        let s = match k {
            Key::Str(s) => s.clone(),
            Key::Long(i) => i.to_string(),
            Key::ULong(u) => u.to_string(),
            Key::Dbl(f) => f.to_string(),
            Key::Bool(b) => b.to_string(),
            Key::Ip(b) => values::ip_display(&values::ip_to_term(b)),
            Key::Null => String::new(),
        };
        match self {
            Matcher::Re(r) => r.is_match(&s),
            Matcher::Exact(set) => set.contains(&s) || matches!(k, Key::Dbl(f) if set.contains(&jfmt(*f))),
        }
    }
}

/// Share exact bucket advancement with the distributed aggregation reducer.
pub fn date_histogram_stepper(config: &Map<String, Value>) -> Result<Box<dyn Fn(i64) -> i64 + Send + Sync>> {
    let rounding = Rounding::parse(config, "date_histogram")?;
    Ok(Box::new(move |key| rounding.next(key)))
}

/// Dense ordinals with a small uniform lookup table. A table cell can cross a
/// calendar boundary; those values use binary search, never an approximation.
struct PreparedDateBuckets {
    keys: Vec<i64>,
    slots: Vec<usize>,
    step: i64,
    uniform: bool,
}

impl PreparedDateBuckets {
    fn new(rounding: &Rounding, lo: i64, hi: i64) -> Option<Self> {
        let first = rounding.round(lo);
        if first > lo { return None; }
        let mut keys = vec![first];
        while *keys.last()? <= hi {
            if keys.len() > 4096 { return None; }
            let key = *keys.last()?;
            let next = rounding.next(key);
            // Same verified constant-range condition as round_memo. Some DST
            // transitions do not advance next(); leave those to the fallback.
            if next <= key || rounding.round(key) != key || rounding.round(next - 1) != key { return None; }
            keys.push(next);
        }
        Self::from_verified_keys(keys)
    }

    // Callers must establish that each [key, next) is one whole bucket.
    fn from_verified_keys(keys: Vec<i64>) -> Option<Self> {
        if keys.len() < 2 || keys.windows(2).any(|pair| pair[0] >= pair[1]) { return None; }
        let first = keys[0];
        let span = keys.last()?.checked_sub(first)?;
        let width = keys[1].checked_sub(first)?;
        if keys.windows(2).all(|pair| pair[1].checked_sub(pair[0]) == Some(width)) {
            return Some(Self { keys, slots: Vec::new(), step: width, uniform: true });
        }
        let step = span / 4096 + 1;
        let mut slots = Vec::with_capacity(4096);
        let mut key = 0;
        let mut x = first;
        while x < *keys.last()? {
            while keys[key + 1] <= x { key += 1; }
            slots.push(key);
            let Some(next) = x.checked_add(step) else { break; };
            x = next;
        }
        Some(Self { keys, slots, step, uniform: false })
    }

    fn bucket(&self, x: i64) -> usize {
        let slot = ((x - self.keys[0]) / self.step) as usize;
        if self.uniform { return slot; }
        let key = self.slots[slot];
        if x < self.keys[key + 1] { key }
        else { self.keys.partition_point(|&k| k <= x) - 1 }
    }
}

#[cfg(test)]
mod prepared_date_tests {
    use super::*;

    #[test]
    fn prepared_date_buckets_match_rounding_across_boundaries() {
        let lo = 1_704_067_200_000i64; // 2024-01-01, includes both DST transitions.
        let hi = lo + 366 * 86_400_000;
        let mut prepared_count = 0;
        for zone in ["UTC", "America/New_York", "Australia/Lord_Howe", "+05:30"] {
            for interval in ["day", "month", "year"] {
                for offset in ["0h", "6h", "-6h"] {
                    let config = json!({"calendar_interval":interval,"time_zone":zone,"offset":offset});
                    let rounding = Rounding::parse(config.as_object().unwrap(), "test").unwrap();
                    let Some(prepared) = PreparedDateBuckets::new(&rounding, lo, hi) else { continue };
                    prepared_count += 1;
                    for x in (lo..=hi).step_by(3_600_017).chain([lo, hi]) {
                        assert_eq!(prepared.keys[prepared.bucket(x)], rounding.round(x), "{config}/{x}");
                    }
                    for &key in &prepared.keys {
                        for x in [key - 1, key, key + 1] {
                            if (lo..=hi).contains(&x) {
                                assert_eq!(prepared.keys[prepared.bucket(x)], rounding.round(x), "{config}/{x}");
                            }
                        }
                    }
                }
            }
        }
        assert!(prepared_count >= 12);
        let rounding = Rounding { cal: None, fixed: 1, offset: 0, tz: Tz::UTC };
        assert!(PreparedDateBuckets::new(&rounding, 0, 10_000).is_none());
    }
}

#[derive(Clone)]
/// Date rounding for `date_histogram` (and composite sources).
struct Rounding {
    cal: Option<CalUnit>,
    fixed: i64,
    offset: i64,
    tz: Tz,
}

impl Rounding {
    fn parse(o: &Map<String, Value>, name: &str) -> Result<Rounding> {
        let tz = match o.get("time_zone").and_then(Value::as_str) {
            Some(z) => Tz::parse(z)?,
            None => Tz::UTC,
        };
        let offset = match o.get("offset") {
            Some(Value::String(s)) => {
                let (neg, rest) = match s.strip_prefix('-') {
                    Some(r) => (true, r),
                    None => (false, s.strip_prefix('+').unwrap_or(s)),
                };
                let v = crate::date::parse_time_value(rest).ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [date_histogram.offset] with value [{s}] as a time value")))?;
                if neg { -v } else { v }
            }
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            _ => 0,
        };
        let (cal, fixed) = if let Some(c) = o.get("calendar_interval").and_then(Value::as_str) {
            (Some(CalUnit::parse(c).ok_or_else(|| EsError::illegal_argument(format!("The supplied interval [{c}] could not be parsed as a calendar interval.")))?), 0)
        } else if let Some(f) = o.get("fixed_interval").and_then(Value::as_str) {
            (None, crate::date::parse_fixed_interval(f).ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [date_histogram.fixedInterval] with value [{f}] as a time value: unit is missing or unrecognized")))?)
        } else if let Some(i) = o.get("interval").and_then(Value::as_str) {
            match CalUnit::parse(i) {
                Some(u) => (Some(u), 0),
                None => (None, crate::date::parse_fixed_interval(i).ok_or_else(|| EsError::illegal_argument(format!("Unable to parse interval [{i}]")))?),
            }
        } else {
            return Err(EsError::illegal_argument(format!("Invalid interval specified, must be non-null and non-empty: [{name}]")));
        };
        Ok(Rounding { cal, fixed, offset, tz })
    }

    fn round(&self, v: i64) -> i64 {
        match self.cal {
            Some(u) => round_down(v - self.offset, u, self.tz) + self.offset,
            None => match self.tz {
                Tz::Named(ntz) => fixed_round_named(v - self.offset, self.fixed, ntz, self.tz, 0) + self.offset,
                Tz::Fixed(_) => {
                    let tzo = self.tz.offset_at(v) as i64 * 1000;
                    let local = v + tzo - self.offset;
                    local.div_euclid(self.fixed) * self.fixed + self.offset - tzo
                }
            },
        }
    }

    fn next(&self, k: i64) -> i64 {
        match self.cal {
            Some(u) => {
                let n = add_units(k - self.offset, u, 1, self.tz) + self.offset;
                // Re-round to absorb DST shifts.
                self.round(n)
            }
            None => {
                let next = self.round(k + self.fixed);
                if next > k || !matches!(self.tz, Tz::Named(_)) { return next; }
                // A fall-back transition can make this local-time bucket
                // longer than `fixed` UTC milliseconds. Rounding k+fixed
                // then returns k and used to truncate all following buckets.
                // Find the first instant rounded into a later bucket.
                let mut step = self.fixed.max(1);
                let mut hi = k.saturating_add(step);
                while self.round(hi) <= k {
                    step = step.saturating_mul(2);
                    let upper = k.saturating_add(step);
                    if upper <= hi { return next; }
                    hi = upper;
                }
                let mut lo = k;
                while (hi as i128 - lo as i128) > 1 {
                    let mid = (lo as i128 + (hi as i128 - lo as i128) / 2) as i64;
                    if self.round(mid) > k { hi = mid; } else { lo = mid; }
                }
                self.round(hi)
            }
        }
    }

    /// The bucket of `v` as `[key, end)`: `end` is past `v` only when the
    /// whole range is known to round to `key` (a fixed interval without
    /// zone transitions, or a range [`Rounding::round_memo`] verified);
    /// else `end == key` (callers round the next value again).
    fn bucket_range(&self, memo: &mut Vec<(i64, i64)>, v: i64) -> (i64, i64) {
        let k = self.round_memo(memo, v);
        if self.cal.is_none() && !matches!(self.tz, Tz::Named(_)) {
            return match k.checked_add(self.fixed) { Some(end) if self.fixed > 0 && v >= k && v < end => (k, end), _ => (k, k) };
        }
        let i = memo.partition_point(|&(start, _)| start <= v);
        match i.checked_sub(1).map(|i| memo[i]) {
            Some((start, end)) if start == k && v < end => (k, end),
            _ => (k, k),
        }
    }

    /// [`Rounding::round`] memoised over the buckets seen so far: calendar
    /// rounding (time-zone aware date arithmetic) costs far more than a
    /// binary search over a handful of `[key, next)` ranges. A range is only
    /// cached once `round` is verified constant over it (both ends), which
    /// with monotonic rounding holds for every value inside.
    fn round_memo(&self, memo: &mut Vec<(i64, i64)>, v: i64) -> i64 {
        // Fixed intervals round by arithmetic, except in a zone with
        // transitions: there each rounding looks the offsets up (two to
        // four zone binary searches), so their buckets are memoized too.
        if self.cal.is_none() && !matches!(self.tz, Tz::Named(_)) {
            return self.round(v);
        }
        let i = memo.partition_point(|&(start, _)| start <= v);
        if i > 0 && v < memo[i - 1].1 {
            return memo[i - 1].0;
        }
        let k = self.round(v);
        // A bounded memo (sorted inserts): many-bucket histograms past it
        // round directly.
        if k <= v && memo.len() < 8192 {
            let end = self.next(k);
            if end > v && self.round(k) == k && self.round(end - 1) == k {
                let at = memo.partition_point(|&(start, _)| start < k);
                memo.insert(at, (k, end));
            }
        }
        k
    }
}

/// A fixed interval rounded in a zone with transitions (ES
/// `TimeIntervalRounding.JavaTimeRounding.round`): floor the local time,
/// map it back through the offsets valid at that local time; a rounding
/// that would cross the previous transition rounds the instant just
/// before that transition instead.
fn fixed_round_named(utc: i64, interval: i64, ntz: chrono_tz::Tz, tz: Tz, depth: u32) -> i64 {
    use chrono::TimeZone;
    let off = tz.offset_at(utc) as i64 * 1000;
    let rounded_local = (utc + off).div_euclid(interval) * interval;
    let naive = crate::date::from_millis(rounded_local);
    // The latest transition in (a, b] (offsets differ at a and b).
    let transition = |mut a: i64, mut b: i64| -> i64 {
        let oa = tz.offset_at(a);
        while b - a > 1 {
            let m = a + (b - a) / 2;
            if tz.offset_at(m) == oa {
                a = m;
            } else {
                b = m;
            }
        }
        b
    };
    let cands: Vec<i64> = match ntz.from_local_datetime(&naive) {
        chrono::LocalResult::Single(d) => vec![d.timestamp_millis()],
        chrono::LocalResult::Ambiguous(a, b) => vec![b.timestamp_millis().max(a.timestamp_millis()), b.timestamp_millis().min(a.timestamp_millis())],
        chrono::LocalResult::None => {
            // In a gap: the gap's start instant.
            let lo = rounded_local - 26 * 3_600_000;
            let hi = rounded_local + 26 * 3_600_000;
            return transition(lo, hi);
        }
    };
    for &inst in &cands {
        if inst <= utc && tz.offset_at(inst) as i64 * 1000 != off && depth < 4 {
            let t = transition(inst, utc);
            if inst < t {
                return fixed_round_named(t - 1, interval, ntz, tz, depth + 1);
            }
        }
        if utc >= inst {
            return inst;
        }
    }
    *cands.last().unwrap_or(&utc)
}

// ---------------------------------------------------------------------------
// Pipelines.

/// Resolve a `buckets_path` against one bucket's JSON.
fn resolve_path(bucket: &Map<String, Value>, path: &str) -> Option<f64> {
    let mut node = bucket;
    let parts: Vec<&str> = path.split('>').collect();
    for p in &parts[..parts.len() - 1] {
        node = node.get(*p)?.as_object()?;
    }
    let last = parts[parts.len() - 1];
    match last {
        "_count" => return node.get("doc_count").and_then(Value::as_f64),
        "_key" => return node.get("key").and_then(Value::as_f64),
        _ => {}
    }
    let (name, metric) = if let Some(i) = last.find('[') {
        (&last[..i], Some(last[i + 1..].trim_end_matches(']')))
    } else if let Some((n, m)) = last.split_once('.') {
        (n, Some(m))
    } else {
        (last, None)
    };
    let agg = node.get(name)?.as_object()?;
    match metric {
        None => agg.get("value").and_then(Value::as_f64).or_else(|| agg.get("doc_count").and_then(Value::as_f64)),
        Some("_count") => agg.get("doc_count").and_then(Value::as_f64),
        // `top_metrics[m]`: the metric of the top document.
        Some(m) if agg.get("top").is_some_and(Value::is_array) => agg["top"].get(0).and_then(|t| t.get("metrics")).and_then(|x| x.get(m)).and_then(Value::as_f64),
        // `change_point` exposes the p-value of its change type (1.0 when stationary).
        Some("p_value") if agg.get("type").is_some_and(Value::is_object) => {
            Some(agg["type"].as_object().and_then(|t| t.values().next()).and_then(|v| v.get("p_value")).and_then(Value::as_f64).unwrap_or(1.0))
        }
        Some(m) => agg.get(m).and_then(Value::as_f64).or_else(|| {
            let vals = agg.get("values")?;
            let mf: Option<f64> = m.parse().ok();
            match vals {
                Value::Object(o) => o.get(m).or_else(|| mf.and_then(|f| o.get(&jfmt(f)))).and_then(Value::as_f64),
                Value::Array(a) => a.iter().find(|e| e.get("key").and_then(Value::as_f64) == mf).and_then(|e| e.get("value")).and_then(Value::as_f64),
                _ => None,
            }
        }),
    }
}

/// A string-valued `buckets_path` target (a keyword metric of `top_metrics`).
fn resolve_path_str(bucket: &Map<String, Value>, path: &str) -> Option<String> {
    let mut bucket = bucket;
    let parts: Vec<&str> = path.split('>').collect();
    for p in &parts[..parts.len() - 1] {
        bucket = bucket.get(*p)?.as_object()?;
    }
    let path = parts[parts.len() - 1];
    let (agg, metric) = path.split_once('.').or_else(|| path.split_once('['))?;
    let metric = metric.trim_end_matches(']');
    bucket.get(agg)?.get("top")?.get(0)?.get("metrics")?.get(metric)?.as_str().map(str::to_string)
}

fn buckets_of(v: &Value) -> Vec<Map<String, Value>> {
    match v.get("buckets") {
        Some(Value::Array(a)) => a.iter().filter_map(|b| b.as_object().cloned()).collect(),
        Some(Value::Object(o)) => o
            .iter()
            .filter_map(|(k, b)| {
                let mut m = b.as_object()?.clone();
                m.entry("key".to_string()).or_insert(json!(k));
                Some(m)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn bucket_key_string(b: &Map<String, Value>) -> String {
    if let Some(s) = b.get("key_as_string").and_then(Value::as_str) {
        return s.to_string();
    }
    match b.get("key") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) if n.is_f64() => jfmt(n.as_f64().unwrap()),
        // A `time_series` bucket's dimension map: Java `Map.toString`, as
        // ES's `getKeyAsString` renders it (`{a=x, b=y}`).
        Some(Value::Object(m)) => {
            let parts: Vec<String> = m.iter().map(|(k, v)| match v {
                Value::String(s) => format!("{k}={s}"),
                other => format!("{k}={other}"),
            }).collect();
            format!("{{{}}}", parts.join(", "))
        }
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn path_of(body: &Value, t: &str) -> Result<String> {
    body.get("buckets_path").and_then(Value::as_str).map(str::to_string).ok_or_else(|| EsError::illegal_argument(format!("Required parameter [buckets_path] missing for [{t}]")))
}

fn sibling_pipeline(name: &str, t: &str, body: &Value, results: &HashMap<String, Value>) -> Result<Value> {
    if analytics::SIBLING.contains(&t) {
        return analytics::sibling_pipeline(name, t, body, results);
    }
    let path = path_of(body, t)?;
    // `agg>metric`, or ES's `agg.metric` shorthand (e.g. `ids._count`, as Kibana sends).
    let (agg, rest) = match path.split_once('>') {
        Some(x) => x,
        None => match path.split_once('.') {
            Some(x) if results.contains_key(x.0) => x,
            _ => return Err(EsError::illegal_argument(format!("No aggregation found for path [{path}]"))),
        },
    };
    let target = results.get(agg).ok_or_else(|| EsError::illegal_argument(format!("No aggregation found for path [{path}]")))?;
    if target.get("buckets").is_none() {
        return Err(EsError::illegal_argument(format!("buckets_path must reference either a number value or a single value numeric metric aggregation, got: [{agg}] for [{name}]")));
    }
    let insert_zeros = body.get("gap_policy").and_then(Value::as_str) == Some("insert_zeros");
    let mut vals: Vec<(String, f64)> = Vec::new();
    for b in buckets_of(target) {
        match resolve_path(&b, rest) {
            Some(v) if v.is_finite() => vals.push((bucket_key_string(&b), v)),
            _ if insert_zeros => vals.push((bucket_key_string(&b), 0.0)),
            _ => {}
        }
    }
    let n = vals.len();
    let sum: f64 = vals.iter().map(|x| x.1).sum();
    Ok(match t {
        "avg_bucket" => json!({ "value": if n > 0 { fnum(sum / n as f64) } else { Value::Null } }),
        "sum_bucket" => json!({ "value": fnum(sum) }),
        "max_bucket" | "min_bucket" => {
            let best = vals.iter().map(|x| x.1).fold(None, |acc: Option<f64>, v| Some(match acc {
                None => v,
                Some(a) => if t == "max_bucket" { a.max(v) } else { a.min(v) },
            }));
            let keys: Vec<String> = match best {
                Some(b) => vals.iter().filter(|x| x.1 == b).map(|x| x.0.clone()).collect(),
                None => vec![],
            };
            json!({ "value": opt_num(best), "keys": keys })
        }
        "stats_bucket" | "extended_stats_bucket" => {
            let min = vals.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
            let max = vals.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            let mut m = json!({
                "count": n,
                "min": if n > 0 { fnum(min) } else { Value::Null },
                "max": if n > 0 { fnum(max) } else { Value::Null },
                "avg": if n > 0 { fnum(sum / n as f64) } else { Value::Null },
                "sum": fnum(sum),
            });
            if t == "extended_stats_bucket" && n > 0 {
                let sos: f64 = vals.iter().map(|x| x.1 * x.1).sum();
                let avg = sum / n as f64;
                let var = (sos / n as f64 - avg * avg).max(0.0);
                m["sum_of_squares"] = fnum(sos);
                m["variance"] = fnum(var);
                m["std_deviation"] = fnum(var.sqrt());
            }
            m
        }
        "percentiles_bucket" => {
            let mut sorted: Vec<f64> = vals.iter().map(|x| x.1).collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let percents: Vec<f64> = body.get("percents").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]);
            let mut m = Map::new();
            for p in percents {
                let v = if sorted.is_empty() { None } else { Some(sorted[((p / 100.0 * sorted.len() as f64).ceil() as usize).saturating_sub(1).min(sorted.len() - 1)]) };
                m.insert(jfmt(p), opt_num(v));
            }
            json!({ "values": m })
        }
        _ => unreachable!(),
    })
}

fn script_source(body: &Value) -> Result<String> {
    match body.get("script") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Object(o)) => o.get("source").or(o.get("inline")).and_then(Value::as_str).map(str::to_string).ok_or_else(|| EsError::illegal_argument("script source is required")),
        _ => Err(EsError::illegal_argument("[script] is required")),
    }
}

/// `bucket_selector` variables: every path resolves (ES
/// `BucketHelpers.resolveBucketValue` + gap policy); `None` only for a
/// malformed `buckets_path`.
fn selector_vars(body: &Value, b: &Map<String, Value>, insert_zeros: bool) -> Result<Option<HashMap<String, f64>>> {
    let doc_count = b.get("doc_count").and_then(Value::as_u64).unwrap_or(1);
    let gap = |p: &str| -> Result<f64> {
        let v = match resolve_path(b, p) {
            Some(v) => v,
            None => {
                // `top_metrics[m]` naming a metric it does not collect.
                if let Some((agg, m)) = p.rsplit('>').next().and_then(|l| l.split_once('[')) {
                    let m = m.trim_end_matches(']');
                    if let Some(metrics) = b.get(agg).and_then(|a| a.get("top")).and_then(|t| t.get(0)).and_then(|t| t.get("metrics")).and_then(Value::as_object) {
                        if !metrics.contains_key(m) {
                            return Err(EsError::illegal_argument(format!("path not supported for [{agg}]: [{m}]")));
                        }
                    }
                }
                f64::NAN
            }
        };
        if p == "_count" {
            return Ok(v);
        }
        Ok(if insert_zeros {
            if !v.is_finite() || doc_count == 0 { 0.0 } else { v }
        } else if v.is_infinite() || doc_count == 0 {
            f64::NAN
        } else {
            v
        })
    };
    let mut vars = HashMap::new();
    match body.get("buckets_path") {
        Some(Value::Object(paths)) => {
            for (k, p) in paths {
                let Some(p) = p.as_str() else { return Ok(None) };
                vars.insert(k.clone(), gap(p)?);
            }
        }
        Some(Value::String(p)) => {
            vars.insert("_value".into(), gap(p)?);
        }
        _ => {}
    }
    if let Some(Value::Object(params)) = body.get("script").and_then(|s| s.get("params")) {
        for (k, v) in params {
            if let Some(f) = v.as_f64() {
                vars.entry(k.clone()).or_insert(f);
            }
        }
    }
    Ok(Some(vars))
}

fn script_vars(body: &Value, b: &Map<String, Value>) -> Option<HashMap<String, f64>> {
    let mut vars = HashMap::new();
    match body.get("buckets_path") {
        Some(Value::Object(paths)) => {
            for (k, p) in paths {
                vars.insert(k.clone(), resolve_path(b, p.as_str()?)?);
            }
        }
        Some(Value::String(p)) => {
            vars.insert("_value".into(), resolve_path(b, p)?);
        }
        _ => {}
    }
    if let Some(Value::Object(params)) = body.get("script").and_then(|s| s.get("params")) {
        for (k, v) in params {
            if let Some(f) = v.as_f64() {
                vars.entry(k.clone()).or_insert(f);
            }
        }
    }
    Some(vars)
}

/// Apply parent pipelines declared in `sub` to the bucket list.
fn apply_parent_pipelines(sub: Option<&Map<String, Value>>, buckets: &mut Vec<Map<String, Value>>) -> Result<()> {
    let Some(sub) = sub else { return Ok(()) };
    if partial::skip_pipeline() {
        return Ok(());
    }
    for (name, def) in sub {
        let Some(obj) = def.as_object() else { continue };
        let Some((t, body)) = obj.iter().find(|(k, _)| PARENT_PIPELINES.contains(&k.as_str())) else { continue };
        match t.as_str() {
            "cumulative_sum" => {
                let path = path_of(body, t)?;
                let mut acc = 0.0;
                for b in buckets.iter_mut() {
                    acc += resolve_path(b, &path).filter(|v| v.is_finite()).unwrap_or(0.0);
                    b.insert(name.clone(), json!({ "value": fnum(acc) }));
                }
            }
            "derivative" => {
                let path = path_of(body, t)?;
                let mut prev: Option<f64> = None;
                for b in buckets.iter_mut() {
                    let cur = resolve_path(b, &path).filter(|v| v.is_finite());
                    if let (Some(p), Some(c)) = (prev, cur) {
                        b.insert(name.clone(), json!({ "value": fnum(c - p) }));
                    }
                    if cur.is_some() {
                        prev = cur;
                    }
                }
            }
            "bucket_script" => {
                let spec = body.get("script").ok_or_else(||EsError::illegal_argument("[script] is required"))?;
                for b in buckets.iter_mut() {
                    if let Some(vars) = script_vars(body, b) {
                        let v = expr::eval_search_script(spec, &vars, crate::script::ScriptContext::BucketScript)?;
                        if v.is_null() {
                            continue;
                        }
                        let x = v.as_f64().ok_or_else(|| EsError::illegal_argument(format!("buckets_path must reference either a number value or a single value numeric metric aggregation, but [{name}] returned a {}", v.type_name())))?;
                        b.insert(name.clone(), json!({ "value": fnum(x) }));
                    }
                }
            }
            "bucket_selector" => {
                let spec = body.get("script").ok_or_else(||EsError::illegal_argument("[script] is required"))?;
                let mut keep = Vec::new();
                // ES runs the selector on every bucket: a missing value is
                // NaN under `skip` (or 0 under `insert_zeros`).
                let zeros = body.get("gap_policy").and_then(Value::as_str) == Some("insert_zeros");
                for b in buckets.drain(..) {
                    match selector_vars(body, &b, zeros)? {
                        Some(vars) => {
                            let v = expr::eval_search_script(spec, &vars, crate::script::ScriptContext::BucketSelector)?;
                            let keep_it = match v {
                                crate::script::Val::Bool(x) => x,
                                other => return Err(EsError::illegal_argument(format!("Script for [{name}] must return a boolean, got [{}]", other.type_name()))),
                            };
                            if keep_it {
                                keep.push(b);
                            }
                        }
                        None => keep.push(b),
                    }
                }
                *buckets = keep;
            }
            "bucket_sort" => {
                let mut crit: Vec<(String, bool)> = Vec::new();
                if let Some(Value::Array(a)) = body.get("sort") {
                    for s in a {
                        match s {
                            Value::String(p) => crit.push((p.clone(), true)),
                            Value::Object(m) => {
                                for (p, d) in m {
                                    let asc = match d {
                                        Value::String(x) => x != "desc",
                                        Value::Object(x) => x.get("order").and_then(Value::as_str) != Some("desc"),
                                        _ => true,
                                    };
                                    crit.push((p.clone(), asc));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if !crit.is_empty() {
                    buckets.sort_by(|a, b| {
                        for (p, asc) in &crit {
                            let (x, y) = (resolve_path(a, p), resolve_path(b, p));
                            let c = match (x, y) {
                                (Some(x), Some(y)) => if *asc { x.total_cmp(&y) } else { y.total_cmp(&x) },
                                (None, Some(_)) => Ordering::Greater,
                                (Some(_), None) => Ordering::Less,
                                _ => Ordering::Equal,
                            };
                            if c != Ordering::Equal {
                                return c;
                            }
                        }
                        Ordering::Equal
                    });
                }
                let from = body.get("from").and_then(Value::as_u64).unwrap_or(0) as usize;
                let size = body.get("size").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(usize::MAX);
                *buckets = buckets.drain(..).skip(from).take(size).collect();
            }
            t if analytics::PARENT.contains(&t) => analytics::parent_pipeline(name, t, body, buckets)?,
            _ => {}
        }
    }
    // A coordinator reruns these pipelines over the merged buckets.
    if !crate::search::aggregation_shard_response() {
        analytics::strip_hidden(buckets);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

/// Validate bucket `order` paths that end in a `top_metrics` metric against
/// the metrics it collects. ES reports an unknown metric from the reduce
/// (`unknown metric [m]`) when terms run as filter-by-filter
/// (`search.aggs.rewrite_to_filter_by_filter`, the default), else from the
/// shard comparator.
pub fn validate_order_paths(aggs: &Map<String, Value>, filter_by_filter: bool) -> Result<()> {
    fn subs_of(spec: &Map<String, Value>) -> Option<&Map<String, Value>> {
        spec.get("aggs").or_else(|| spec.get("aggregations")).and_then(Value::as_object)
    }
    fn agg_type(spec: &Map<String, Value>) -> Option<(&str, &Map<String, Value>)> {
        spec.iter().find(|(k, _)| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).and_then(|(k, v)| v.as_object().map(|o| (k.as_str(), o)))
    }
    fn check(path: &str, subs: Option<&Map<String, Value>>, fbf: bool) -> Result<()> {
        let parts: Vec<&str> = path.split('>').collect();
        let mut cur = subs;
        for p in &parts[..parts.len() - 1] {
            let Some(spec) = cur.and_then(|s| s.get(*p)).and_then(Value::as_object) else { return Ok(()) };
            cur = subs_of(spec);
        }
        let last = parts[parts.len() - 1];
        let (name, metric) = match last.find('[') {
            Some(i) => (&last[..i], Some(last[i + 1..].trim_end_matches(']'))),
            None => match last.split_once('.') {
                Some((n, m)) => (n, Some(m)),
                None => (last, None),
            },
        };
        let Some(spec) = cur.and_then(|s| s.get(name)).and_then(Value::as_object) else { return Ok(()) };
        let (Some(("top_metrics", tm)), Some(m)) = (agg_type(spec), metric) else { return Ok(()) };
        let fields: Vec<&str> = match tm.get("metrics") {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.get("field").and_then(Value::as_str)).collect(),
            Some(Value::Object(o)) => o.get("field").and_then(Value::as_str).into_iter().collect(),
            _ => Vec::new(),
        };
        if fields.contains(&m) {
            return Ok(());
        }
        Err(EsError::illegal_argument(if fbf {
            format!("Invalid aggregation order path [{path}]. unknown metric [{m}]")
        } else {
            format!("Invalid aggregation order path [{path}]. Unknown metric name [{m}] on multi-value metrics aggregation [{name}]")
        }))
    }
    for spec in aggs.values() {
        let Some(spec) = spec.as_object() else { continue };
        let subs = subs_of(spec);
        if let Some((t, body)) = agg_type(spec) {
            if matches!(t, "terms" | "multi_terms" | "histogram" | "date_histogram" | "rare_terms" | "significant_terms") {
                let orders: Vec<&Map<String, Value>> = match body.get("order") {
                    Some(Value::Object(o)) => vec![o],
                    Some(Value::Array(a)) => a.iter().filter_map(Value::as_object).collect(),
                    _ => Vec::new(),
                };
                for o in orders {
                    for k in o.keys() {
                        if !matches!(k.as_str(), "_key" | "_count" | "_term") {
                            check(k, subs, filter_by_filter)?;
                        }
                    }
                }
            }
        }
        if let Some(s) = subs {
            validate_order_paths(s, filter_by_filter)?;
        }
    }
    Ok(())
}

/// Retain only the best K docs, with exactly the full-sort score/segment/doc tie order.
fn sample_top(docs: &Docs, segments: &[usize], limit: usize) -> Vec<(usize, DocId, f32)> {
    #[derive(Clone, Copy)]
    struct Hit(usize, DocId, f32);
    impl Ord for Hit {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.2.total_cmp(&self.2).then(self.0.cmp(&other.0)).then(self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for Hit { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
    impl PartialEq for Hit { fn eq(&self, other: &Self) -> bool { self.cmp(other).is_eq() } }
    impl Eq for Hit {}
    if limit == 0 { return Vec::new(); }
    let mut heap = std::collections::BinaryHeap::new();
    for &s in segments {
        for &(d, score) in &docs[s] {
            let h = Hit(s, d, score);
            if heap.len() < limit { heap.push(h); }
            else if heap.peek().is_some_and(|worst| h < *worst) { *heap.peek_mut().unwrap() = h; }
        }
    }
    heap.into_iter().map(|Hit(s, d, score)| (s, d, score)).collect()
}
