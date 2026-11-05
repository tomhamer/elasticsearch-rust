//! `_search` execution: request parsing, per-index query translation, hit
//! collection through `rustic_core::query::Searcher` (the engine visits its
//! shard targets one after another; the cluster fans shards out across
//! engines concurrently; parallel per-segment,
//! sort/search_after/track_total_hits/terminate_after/timeout/explain/
//! collapse/rescore), the aggregation pass, the fetch phase (`_source`
//! spliced raw unless filtered, `fields`, `docvalue_fields`, highlight) and
//! the ES response JSON.

use crate::aggs::{self, AggEnv, AggSeg};
use crate::error::{EsError, Result};
use crate::highlight::{self, QueryTerms};
use crate::index::IndexService;
use crate::mapping::{EsType, FieldLookup, ID_FIELD, IGNORED_FIELD, IGNORED_SOURCE_FIELD, PRIMARY_TERM_FIELD, ROUTING_FIELD, SEQ_NO_FIELD, SOURCE_FIELD, VERSION_FIELD};
use crate::query::dsl::{parse_knn_query, parse_query, QueryContext, TermsLookupFn};
use crate::values;
use rustic_core::query::{
    Bm25Params, LeafReader, MinShouldMatch, Query, Rescore, SearchContext, SearchOptions, Searcher, SortField, SortKind, SortMissing, SortMode,
    SortValue, TopDocs, TopHit, TotalHitsRelation, TrackTotalHits,
};
use rustic_core::ReaderView;
use rustic_core::types::{DocId, Document};
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Hook 9 (query phase / fetch phase split): `hooks/search_ext.rs`.
#[path = "hooks/search_ext.rs"]
pub mod phases;

pub(crate) fn add_typed_keys_pub(out: Value, spec: &Value) -> Value {
    add_typed_keys(out, spec)
}

/// One index to search (with an optional alias filter and an optional
/// point-in-time reader view of its shard copy).
#[derive(Clone)]
pub struct Target {
    pub svc: Arc<IndexService>,
    /// Qualified hit index name for a remote-cluster view of this target.
    pub remote_index: Option<String>,
    pub alias_filter: Option<Value>,
    /// A retained reader view (PIT / scroll / ES|QL capture, or a searcher
    /// handle's view); `None`: search the shard's current view.
    pub snapshot: Option<Arc<ReaderView>>,
    /// Security: the document-/field-level security the caller's role imposes
    /// on this index (`None` = unrestricted). Every read path honours it.
    pub access: Option<Arc<crate::security::roles::IndexAccess>>,
    /// The shard copy searched (`None` = every local shard the `routing`
    /// selects; [`expand_targets`] turns such a target into one target per
    /// shard before a search runs). `snapshot` belongs to this shard.
    pub shard: Option<Arc<crate::index::IndexShard>>,
    /// Search routing values (comma separated: request `routing`, alias
    /// `search_routing`): only the shards they route to are searched.
    pub routing: Option<String>,
}

thread_local! {
    static READ_SHARD: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    static READ_SHARD_COUNT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Request-scoped selection used by the cluster's shard search endpoints.
/// The guard restores worker-thread state before that worker is reused.
pub struct ReadShardScope(Option<u32>);
impl ReadShardScope {
    pub fn enter(shard: Option<u32>) -> Self { Self(READ_SHARD.with(|s| s.replace(shard))) }
}
impl Drop for ReadShardScope {
    fn drop(&mut self) { READ_SHARD.with(|s| s.set(self.0)); }
}
/// The coordinator's total selected shard count for the can-match threshold.
pub struct ReadShardCountScope(Option<usize>);
impl ReadShardCountScope {
    pub fn enter(count: Option<usize>) -> Self { Self(READ_SHARD_COUNT.with(|slot| slot.replace(count))) }
}
impl Drop for ReadShardCountScope {
    fn drop(&mut self) { READ_SHARD_COUNT.with(|slot| slot.set(self.0)); }
}

pub fn selected_read_shard() -> Option<u32> { READ_SHARD.with(|s| s.get()) }

/// The trusted cluster forwarding scope means aggregation candidates have a
/// final coordinator reduction still to come. Public search JSON cannot set it.
pub(crate) fn aggregation_shard_response() -> bool { READ_SHARD_COUNT.with(|s| s.get().is_some()) }

/// One target per shard copy to search: a whole-index target becomes one
/// target per local shard (in shard order) that its `routing` selects.
/// Already expanded targets are kept as they are. A sparse (cluster-managed)
/// index expands to the copies open on this node; when none of the shards
/// its routing selects is open here the search fails with
/// `shard_not_found_exception` (ES `ShardNotFoundException`) instead of
/// covering zero shards of it and answering as if it had none.
pub fn expand_targets(targets: &[Target]) -> Result<Vec<Target>> {
    let mut out = Vec::with_capacity(targets.len());
    for t in targets {
        if t.shard.is_some() {
            out.push(t.clone());
            continue;
        }
        let want = t.svc.search_shards_for(t.routing.as_deref());
        let before = out.len();
        for s in t.svc.local_shards() {
            if want.contains(&s.id) {
                out.push(Target { shard: Some(s), ..t.clone() });
            }
        }
        if out.len() == before && t.svc.is_sparse() {
            if let Some(&first) = want.first() {
                return Err(EsError::shard_not_found(&t.svc.name, &t.svc.uuid, first));
            }
        }
    }
    Ok(out)
}

/// A search request's own check (`_search`, `_count`, PIT open): every
/// shard a sparse index's routing selects must be open here, or the read
/// fails with `shard_not_found_exception`, never a 200 that covers some of
/// its shards and reports none failed. (Internal readers that pick the
/// local shards they need from [`expand_targets`] are not held to it.)
pub fn require_local_shards(targets: &[Target]) -> Result<()> {
    for t in targets.iter().filter(|t| t.shard.is_none() && t.svc.is_sparse()) {
        if let Some(missing) = t.svc.search_shards_for(t.routing.as_deref()).into_iter().find(|s| t.svc.shard(*s).is_none()) {
            return Err(EsError::shard_not_found(&t.svc.name, &t.svc.uuid, missing));
        }
    }
    Ok(())
}

/// Do these targets need expanding into shard targets?
fn needs_expansion(targets: &[Target]) -> bool {
    targets.iter().any(|t| t.shard.is_none())
}

impl Target {
    /// A whole-index target (no alias filter, snapshot or security).
    pub fn new(svc: Arc<IndexService>) -> Target {
        Target { svc, remote_index: None, alias_filter: None, snapshot: None, access: None, shard: None, routing: None }
    }

    pub fn hit_index(&self) -> &str {
        self.remote_index.as_deref().unwrap_or(&self.svc.name)
    }

    /// The shard copy of an expanded target ([`expand_targets`]).
    ///
    /// # Panics
    /// On a target that was not expanded.
    pub fn shard_copy(&self) -> &Arc<crate::index::IndexShard> {
        self.shard.as_ref().expect("search target not expanded to a shard copy")
    }

    /// The alias filter AND the caller's DLS filter.
    pub fn effective_filter(&self) -> Option<Value> {
        let dls = self.access.as_ref().and_then(|a| a.dls.clone());
        match (self.alias_filter.clone(), dls) {
            (Some(a), Some(d)) => Some(json!({"bool": {"filter": [a, d]}})),
            (a, d) => a.or(d),
        }
    }

    /// Field-level security of the caller on this index.
    pub fn field_access(&self) -> Option<&crate::security::roles::FieldAccess> {
        self.access.as_ref().and_then(|a| a.fields.as_ref())
    }

    /// Does the caller's role restrict reads of this index (DLS or FLS)?
    pub fn is_security_restricted(&self) -> bool {
        self.access.as_ref().is_some_and(|a| a.is_restricted())
    }

    /// The index's field lookup, minus the fields the caller may not see:
    /// hidden fields behave as unmapped for queries, sorts, aggregations,
    /// `fields`, `docvalue_fields`, highlighting and scripts.
    pub fn lookup(&self) -> Arc<FieldLookup> {
        let lk = self.svc.lookup();
        match self.field_access() {
            None => lk,
            Some(fa) => {
                let mut f = (*lk).clone();
                f.fields.retain(|n, _| fa.allows(n));
                Arc::new(f)
            }
        }
    }

    /// `_source` (raw JSON text) as the caller may see it (borrowed when
    /// unrestricted). Source that cannot be parsed is withheld entirely.
    pub fn visible_source<'a>(&self, raw: &'a str) -> std::borrow::Cow<'a, str> {
        match self.field_access() {
            None => std::borrow::Cow::Borrowed(raw),
            Some(fa) => match serde_json::from_str::<Value>(raw) {
                Ok(v) => std::borrow::Cow::Owned(fa.filter_source(&v).to_string()),
                Err(_) => crate::source_json::Source::parse(raw).map(|s|std::borrow::Cow::Owned(s.retain_leaves(&|p|fa.allows(p)).text())).unwrap_or(std::borrow::Cow::Borrowed("")),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SortSpec {
    pub field: String,
    pub descending: bool,
    pub missing: Option<Value>,
    pub mode: Option<String>,
    pub unmapped_type: Option<String>,
    pub format: Option<String>,
    /// `numeric_type` (`date` / `date_nanos`): unit date sort values are compared and rendered in.
    pub numeric_type: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceFilter {
    All,
    None,
    Filter { includes: Vec<String>, excludes: Vec<String> },
}

/// A parsed search request.
#[derive(Clone, Debug)]
pub struct SearchRequest {
    pub query: Option<Value>,
    pub post_filter: Option<Value>,
    pub knn: Vec<Value>,
    pub from: usize,
    pub size: usize,
    /// `size` was given (body or URL); ES keeps `-1` for "unset", which
    /// compound retrievers replace with their `rank_window_size`.
    pub size_explicit: bool,
    pub sort: Vec<SortSpec>,
    pub search_after: Option<Vec<Value>>,
    pub source: SourceFilter,
    /// `_source: {exclude_vectors: ..}` / `_source_exclude_vectors`: leave
    /// the vector fields out of `_source` (`None`: the index's
    /// `exclude_source_vectors` setting decides).
    pub exclude_vectors: Option<bool>,
    /// `exclude_vectors` unset means `false` (reindex / update-by-query
    /// searches: `AbstractAsyncBulkByPaginatedSearchAction` includes vectors
    /// unless told otherwise, `_inference_fields` still left out).
    pub vectors_by_default: bool,
    pub fields: Vec<(String, Option<String>)>,
    pub docvalue_fields: Vec<(String, Option<String>)>,
    pub highlight: Option<Map<String, Value>>,
    pub explain: bool,
    pub version: bool,
    pub seq_no_primary_term: bool,
    pub track_total_hits: Option<TrackTotalHits>,
    pub rest_total_hits_as_int: bool,
    /// `include_named_queries_score`: `matched_queries` as `{name: score}`.
    pub include_named_queries_score: bool,
    pub track_scores: bool,
    pub min_score: Option<f32>,
    pub terminate_after: Option<u64>,
    pub timeout: Option<Duration>,
    pub aggs: Option<Value>,
    pub typed_keys: bool,
    pub collapse: Option<String>,
    /// The raw `collapse` object (`inner_hits`, ...; see [`crate::collapse`]).
    pub collapse_spec: Option<Value>,
    pub rescore: Option<Value>,
    pub profile: bool,
    pub stored_fields_none: bool,
    /// `force_synthetic_source`: render `_source` through the synthetic loader even on a stored-source index.
    pub force_synthetic_source: bool,
    /// Cluster setting `search.allow_expensive_queries`.
    pub allow_expensive_queries: bool,
    /// `stored_fields` names (mapped fields with `store: true`).
    pub stored_fields: Vec<String>,
    /// `fields` patterns requested with `include_unmapped`.
    pub fields_unmapped: Vec<String>,
    /// Raw sort objects that need more than [`SortSpec`] holds, keyed by
    /// sort field: `_geo_distance` specs and sorts with a `nested` context.
    pub sort_extra: HashMap<String, Value>,
    /// `script_fields` (raw; compiled through the script cache).
    pub script_fields: Option<Value>,
    pub raw_script_fields: Option<crate::source_json::Source>,
    pub raw_runtime_mappings: Option<crate::source_json::Source>,
    pub raw_sort: Option<crate::source_json::Source>,
    pub raw_search_scripts: Vec<crate::source_json::Source>,
    /// Request-level `runtime_mappings`.
    pub runtime_mappings: Option<Value>,
    /// The `suggest` section (completion suggester).
    pub suggest: Option<Value>,
    /// `retriever` (raw; see [`crate::retriever`]).
    pub retriever: Option<Value>,
    /// Legacy `rank` (`{"rrf": {...}}`) and `sub_searches`.
    pub rank: Option<Value>,
    pub sub_searches: Option<Value>,
    /// Rank docs of a compound retriever (set by [`crate::retriever`]).
    pub rank_plan: Option<crate::retriever::SharedPlan>,
    /// `random_reranker` over a `standard` retriever, run as ES runs it (a
    /// rank builder, not a compound retriever): `(rank_window_size, seed)`.
    /// The query phase's top window hits are re-scored with
    /// `Random(seed + doc).nextFloat()` and re-sorted before the page is cut.
    pub random_rerank: Option<(usize, Option<i64>)>,
    /// `slice: {id, max}` (sliced scroll).
    pub slice: Option<(u32, u32)>,
    /// `indices_boost` resolved to concrete index names (first match wins).
    pub index_boosts: Vec<(String, f32)>,
    /// `search_type=dfs_query_then_fetch`: score with term statistics
    /// gathered over every shard searched ([`phases::DfsStats`]).
    pub dfs: bool,
    /// DFS statistics supplied by a cluster coordinator (merged over every
    /// shard of the search, [`phases::merge_dfs`]); scoring uses them
    /// instead of the statistics of the shards searched here. Never parsed
    /// from the request body.
    pub dfs_stats: Option<Arc<phases::DfsStats>>,
    pub admitted_dfs_stats: Option<phases::AdmittedDfsStats>,
    /// `pre_filter_shard_size`: with more shards than this, shards whose
    /// query rewrites to `match_none` are reported as `skipped`.
    pub pre_filter_shard_size: Option<usize>,
    /// Query phase of a cluster coordinator's query_then_fetch: every hit
    /// also carries `"_rustic_doc": [target, leaf, doc]`, its address in the
    /// (pinned, see [`pin_views`]) targets. Never parsed from the body.
    pub doc_refs: bool,
    /// With [`Self::doc_refs`]: a hit is only what the coordinator's merge
    /// reads (`_index`, `_score`, `sort` and its address), with no stored
    /// field read: the fetch phase renders the rest of the winners, as ES's
    /// query phase returns only doc ids, scores and sort values. Never
    /// parsed from the body.
    pub light_refs: bool,
    /// Fetch phase of a coordinator that matches the fetched hits to its
    /// winners by address: every hit carries its `"_rustic_doc"` too.
    /// Never parsed from the body.
    pub fetch_refs: bool,
    /// Fetch phase of a cluster coordinator's query_then_fetch: render
    /// these hits `(target, leaf, doc, score)` of the pinned targets the
    /// query phase searched instead of searching (the query still drives
    /// highlighting and `matched_queries`). Never parsed from the body.
    pub fetch_docs: Option<Arc<Vec<(usize, usize, u32, f32)>>>,
    /// A query_then_fetch shard whose query phase kept no context, searched
    /// again in one phase to render its winners: it counts as that shard's
    /// fetch phase only. Never parsed from the body.
    pub refetch: bool,
    /// A coordinator's single-round search of a small page: render the
    /// hits as usual but count only the query phase on each target; the
    /// coordinator counts the fetch ([`SearchOutput::fetch_nanos`]) on the
    /// shards whose hits made its merged page. Never parsed from the body.
    pub defer_fetch_stats: bool,
    /// Internal doc-values reads (EQL's stage searches): each hit is read
    /// as its `_index`, `_id` and `docvalue_fields` columns (dates as epoch
    /// millis) into [`SearchOutput::columns`], with no stored document,
    /// metadata or sort values, and none rendered ([`column_hit`]). Never
    /// parsed from the body.
    pub columns_only: bool,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: None,
            post_filter: None,
            knn: Vec::new(),
            from: 0,
            size: 10,
            size_explicit: false,
            sort: Vec::new(),
            search_after: None,
            source: SourceFilter::All,
            exclude_vectors: None,
            vectors_by_default: false,
            fields: Vec::new(),
            docvalue_fields: Vec::new(),
            highlight: None,
            explain: false,
            version: false,
            seq_no_primary_term: false,
            track_total_hits: Some(TrackTotalHits::UpTo(10_000)),
            rest_total_hits_as_int: false,
            include_named_queries_score: false,
            track_scores: false,
            min_score: None,
            terminate_after: None,
            timeout: None,
            aggs: None,
            typed_keys: false,
            collapse: None,
            collapse_spec: None,
            rescore: None,
            profile: false,
            stored_fields_none: false,
            force_synthetic_source: false,
            allow_expensive_queries: true,
            stored_fields: Vec::new(),
            fields_unmapped: Vec::new(),
            sort_extra: HashMap::new(),
            script_fields: None,
            raw_script_fields: None,
            raw_runtime_mappings: None, raw_sort: None, raw_search_scripts: Vec::new(),
            runtime_mappings: None,
            suggest: None,
            retriever: None,
            rank: None,
            sub_searches: None,
            rank_plan: None,
            random_rerank: None,
            slice: None,
            index_boosts: Vec::new(),
            dfs: false,
            dfs_stats: None,
            admitted_dfs_stats: None,
            pre_filter_shard_size: None,
            doc_refs: false,
            light_refs: false,
            fetch_refs: false,
            fetch_docs: None,
            refetch: false,
            defer_fetch_stats: false,
            columns_only: false,
        }
    }
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

pub fn parse_source_filter(v: &Value) -> Result<SourceFilter> {
    Ok(match v {
        Value::Bool(true) => SourceFilter::All,
        Value::Bool(false) => SourceFilter::None,
        Value::String(_) | Value::Array(_) => SourceFilter::Filter { includes: str_list(v), excludes: vec![] },
        Value::Object(o) => SourceFilter::Filter {
            includes: o.get("includes").or(o.get("include")).map(str_list).unwrap_or_default(),
            excludes: o.get("excludes").or(o.get("exclude")).map(str_list).unwrap_or_default(),
        },
        _ => return Err(EsError::parsing("[_source] must be a boolean, string, array or object")),
    })
}

fn parse_sort(v: &Value) -> Result<Vec<SortSpec>> {
    let items: Vec<Value> = match v {
        Value::Array(a) => a.clone(),
        Value::String(s) => s.split(',').map(|x| Value::String(x.to_string())).collect(),
        other => vec![other.clone()],
    };
    let mut out = Vec::new();
    for it in items {
        match it {
            Value::String(s) => {
                // URL form `field:desc`.
                let (f, ord) = match s.split_once(':') {
                    Some((f, o)) => (f.to_string(), Some(o.to_string())),
                    None => (s.clone(), None),
                };
                let desc = match ord.as_deref() {
                    Some("desc") => true,
                    Some("asc") => false,
                    _ => f == "_score",
                };
                out.push(SortSpec { field: f, descending: desc, missing: None, mode: None, unmapped_type: None, format: None, numeric_type: None });
            }
            Value::Number(_) | Value::Bool(_) => {
                let f = crate::query::dsl::scalar_string(&it);
                out.push(SortSpec { field: f, descending: false, missing: None, mode: None, unmapped_type: None, format: None, numeric_type: None });
            }
            Value::Object(o) => {
                for (f, spec) in o {
                    let mut s = SortSpec { field: f.clone(), descending: f == "_score", missing: None, mode: None, unmapped_type: None, format: None, numeric_type: None };
                    match spec {
                        Value::String(ord) => s.descending = ord == "desc",
                        Value::Object(so) => {
                            if let Some(ord) = so.get("order").and_then(Value::as_str) {
                                s.descending = match ord {
                                    "desc" => true,
                                    "asc" => false,
                                    other => return Err(EsError::illegal_argument(format!("Unknown SortOrder [{other}]"))),
                                };
                            }
                            s.missing = so.get("missing").cloned();
                            s.mode = so.get("mode").and_then(Value::as_str).map(str::to_string);
                            s.unmapped_type = so.get("unmapped_type").and_then(Value::as_str).map(str::to_string);
                            s.format = so.get("format").and_then(Value::as_str).map(str::to_string);
                            s.numeric_type = so.get("numeric_type").and_then(Value::as_str).map(str::to_string);
                        }
                        _ => return Err(EsError::parsing(format!("[{f}] sort must be a string or object"))),
                    }
                    out.push(s);
                }
            }
            _ => return Err(EsError::parsing("malformed sort")),
        }
    }
    Ok(out)
}

/// The sort objects [`SortSpec`] cannot represent (`_geo_distance`, a
/// `nested` context), keyed by sort field.
fn sort_extras(v: &Value) -> HashMap<String, Value> {
    let items: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut out = HashMap::new();
    for it in items {
        if let Value::Object(o) = it {
            for (f, spec) in o {
                if f == "_geo_distance" || f == "_script" || spec.get("nested").is_some() {
                    out.insert(f.clone(), spec.clone());
                }
            }
        }
    }
    out
}

fn field_list(v: &Value) -> Vec<(String, Option<String>)> {
    match v {
        Value::Array(a) => a
            .iter()
            .filter_map(|x| match x {
                Value::String(s) => Some((s.clone(), None)),
                Value::Object(o) => o.get("field").and_then(Value::as_str).map(|f| (f.to_string(), o.get("format").and_then(Value::as_str).map(str::to_string))),
                _ => None,
            })
            .collect(),
        Value::String(s) => s.split(',').map(|x| (x.to_string(), None)).collect(),
        _ => Vec::new(),
    }
}

fn track_total(v: &Value) -> Result<Option<TrackTotalHits>> {
    Ok(match v {
        Value::Bool(true) => Some(TrackTotalHits::True),
        Value::Bool(false) => None,
        Value::Number(n) => {
            let x = n.as_i64().unwrap_or(10_000);
            if x == -1 { None } else if x == i32::MAX as i64 { Some(TrackTotalHits::True) } else if x < 0 { return Err(EsError::illegal_argument(format!("[track_total_hits] parameter must be positive or equals to -1, got {x}"))) } else { Some(TrackTotalHits::UpTo(x as u64)) }
        }
        Value::String(s) => match s.as_str() {
            "true" => Some(TrackTotalHits::True),
            "false" => None,
            n => {
                let x: i64 = n.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse [track_total_hits] value [{n}]")))?;
                if x == -1 {
                    None
                } else if x == i32::MAX as i64 {
                    Some(TrackTotalHits::True)
                } else if x < 0 {
                    return Err(EsError::illegal_argument(format!("[track_total_hits] parameter must be positive or equals to -1, got {x}")));
                } else {
                    Some(TrackTotalHits::UpTo(x as u64))
                }
            }
        },
        _ => Some(TrackTotalHits::UpTo(10_000)),
    })
}

/// Jackson numeric accessor on a non-numeric token (`parser.intValue()`):
/// `x_content_parse_exception` over `json_parse_exception`, positioned just
/// past the token.
fn not_numeric(key: &str, v: &Value, text_hint: bool) -> EsError {
    let what = if text_hint { "not numeric, can not use numeric value accessors" } else { "not of boolean type" };
    EsError::jackson_type(key, None, v, what)
}

/// `parser.intValue()` on a body value (ES `SearchSourceBuilder`): strings go
/// through `Integer.parseInt`, floats are truncated, other tokens are a
/// Jackson error.
fn int_value(v: &Value, key: &str) -> Result<i64> {
    match v {
        Value::Number(n) => Ok(n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0)),
        Value::String(s) => s.trim().parse::<i64>().or_else(|_| s.trim().parse::<f64>().map(|f| f as i64).map_err(|_| ())).map_err(|_| EsError::number_format(s)),
        other => Err(not_numeric(key, other, true)),
    }
}

/// `parser.booleanValue()` on a body value.
fn bool_value(v: &Value, key: &str) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) => match s.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed."))),
        },
        other => Err(not_numeric(key, other, false)),
    }
}

fn usize_of(v: &Value, name: &str) -> Result<usize> {
    let n: i64 = match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).ok_or_else(|| EsError::illegal_argument(format!("[{name}] parameter cannot be negative, found [{n}]")))?,
        Value::String(s) => s.trim().parse().map_err(|_| EsError::number_format(s))?,
        _ => return Err(not_numeric(name, v, true)),
    };
    if n < 0 {
        // ES's exact wording differs per parameter.
        return Err(EsError::illegal_argument(match name {
            "from" => format!("[from] parameter cannot be negative but was [{n}]"),
            "terminate_after" => "terminateAfter must be > 0".to_string(),
            _ => format!("[{name}] parameter cannot be negative, found [{n}]"),
        }));
    }
    Ok(n as usize)
}

/// Parse a search body plus URL parameters into a [`SearchRequest`].
pub fn parse_request(body: &Value, params: &HashMap<String, String>) -> Result<SearchRequest> {
    let mut r = SearchRequest::default();
    let empty = Map::new();
    let o = match body {
        Value::Object(o) => o,
        Value::Null => &empty,
        other => return Err(EsError::parsing(format!("Expected [START_OBJECT] but found [{}]", crate::jsonpos::token_name(other))).cursor("root").with_lc()),
    };
    // ES `SearchSourceBuilder.parseXContent`: which token kinds each key accepts;
    // anything else is `Unknown key for a <TOKEN> in [key].` at the value.
    const SCALAR_KEYS: &[&str] = &["from", "size", "timeout", "terminate_after", "min_score", "version", "seq_no_primary_term", "explain", "track_scores", "track_total_hits", "_source", "stored_fields", "sort", "profile"];
    const OBJECT_KEYS: &[&str] = &["query", "post_filter", "_source", "script_fields", "aggregations", "aggs", "highlight", "suggest", "sort", "rescore", "ext", "slice", "collapse", "pit", "runtime_mappings", "knn", "rank", "retriever"];
    const ARRAY_KEYS: &[&str] = &["fields", "stored_fields", "docvalue_fields", "indices_boost", "sort", "rescore", "stats", "search_after", "knn", "sub_searches", "_source"];
    for (k, v) in o {
        let ok = match v {
            Value::Object(_) => OBJECT_KEYS.contains(&k.as_str()),
            Value::Array(_) => ARRAY_KEYS.contains(&k.as_str()),
            Value::Null => false,
            _ => SCALAR_KEYS.contains(&k.as_str()),
        };
        if !ok {
            return Err(EsError::unknown_key(k, v, None));
        }
    }
    for (k, v) in o {
        match k.as_str() {
            "query" => {
                crate::search_strict::check_query(v, "query", None)?;
                r.query = Some(v.clone())
            }
            "post_filter" => {
                crate::search_strict::check_query(v, "post_filter", None)?;
                r.post_filter = Some(v.clone())
            }
            "knn" => {
                crate::search_strict::check_knn(v)?;
                r.knn = match v {
                    Value::Array(a) => a.clone(),
                    other => vec![other.clone()],
                }
            }
            "from" => r.from = usize_of(v, "from")?,
            "size" => {
                r.size = usize_of(v, "size")?;
                r.size_explicit = true;
            }
            "sort" => {
                crate::search_strict::check_sort(v)?;
                r.sort = parse_sort(v)?;
                r.sort_extra = sort_extras(v);
            }
            "search_after" => {
                let a = v.as_array().cloned().ok_or_else(|| EsError::parsing("[search_after] must be an array"))?;
                if let Some(bad) = a.iter().find(|x| x.is_object() || x.is_array()) {
                    return Err(EsError::parsing(format!("Expected [VALUE_STRING] or [VALUE_NUMBER] or [VALUE_BOOLEAN] or [VALUE_NULL] but found [{}] inside search_after.", crate::jsonpos::token_name(bad))).at_key("search_after", None).with_lc());
                }
                r.search_after = Some(a)
            }
            "_source" => {
                crate::search_strict::check_source(v)?;
                r.source = parse_source_filter(v)?;
                r.exclude_vectors = v.get("exclude_vectors").and_then(Value::as_bool);
            }
            "fields" => {
                crate::search_strict::check_field_list("fields", v)?;
                r.fields = field_list(v);
                for (f, _) in &r.fields {
                    if matches!(f.as_str(), "_seq_no" | "_source" | "_feature" | "_nested_path" | "_field_names") {
                        return Err(EsError::illegal_argument(format!("error fetching [{f}]: Cannot fetch values for internal field [{f}].")));
                    }
                }
                if let Value::Array(a) = v {
                    for x in a {
                        if x.get("include_unmapped").and_then(Value::as_bool) == Some(true) {
                            if let Some(f) = x.get("field").and_then(Value::as_str) {
                                r.fields_unmapped.push(f.to_string());
                            }
                        }
                    }
                }
            }
            "docvalue_fields" => {
                crate::search_strict::check_field_list("docvalue_fields", v)?;
                r.docvalue_fields = field_list(v)
            }
            "stored_fields" => {
                if !v.is_string() && !v.is_array() {
                    return Err(EsError::parsing(format!("Expected [VALUE_STRING] or [START_ARRAY] in [stored_fields] but found [{}]", crate::jsonpos::token_name(v))).at_key("stored_fields", None).with_lc());
                }
                let sf = str_list(v);
                r.stored_fields = sf.iter().filter(|s| !s.starts_with('_') || s.as_str() == "_ignored_source").cloned().collect();
                if sf.iter().any(|s| s == "_none_") {
                    r.stored_fields_none = true;
                    r.source = SourceFilter::None;
                } else if !sf.iter().any(|s| s == "_source") && !o.contains_key("_source") {
                    // Explicit stored fields without `_source`: no source.
                    r.source = SourceFilter::None;
                }
            }
            "highlight" => {
                let h = v.as_object().cloned().ok_or_else(|| EsError::parsing("[highlight] must be an object"))?;
                crate::search_strict::check_highlight(&h)?;
                highlight::validate_spec(&h)?;
                r.highlight = Some(h);
            }
            "explain" => r.explain = bool_value(v, "explain")?,
            "version" => r.version = bool_value(v, "version")?,
            "seq_no_primary_term" => r.seq_no_primary_term = bool_value(v, "seq_no_primary_term")?,
            "track_total_hits" => {
                if let Value::String(s) = v {
                    if !matches!(s.as_str(), "true" | "false") && s.trim().parse::<i64>().is_err() {
                        return Err(EsError::number_format(s));
                    }
                }
                r.track_total_hits = track_total(v)?
            }
            "track_scores" => r.track_scores = bool_value(v, "track_scores")?,
            "min_score" => {
                r.min_score = Some(match v {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
                    Value::String(s) => s.trim().parse::<f32>().map_err(|_| EsError::number_format(s))?,
                    other => return Err(not_numeric("min_score", other, true)),
                })
            }
            "terminate_after" => r.terminate_after = Some(usize_of(v, "terminate_after")? as u64).filter(|x| *x > 0),
            "timeout" => {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => String::new(),
                };
                let ms = crate::date::parse_time_value(&s).ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [timeout] with value [{s}] as a time value: unit is missing or unrecognized")))?;
                r.timeout = Some(ms).filter(|x| *x > 0).map(|x| Duration::from_millis(x as u64))
            }
            "aggs" | "aggregations" => {
                crate::aggs::strict::check_tree(v, None)?;
                r.aggs = Some(v.clone())
            }
            "collapse" => {
                crate::search_strict::check_collapse(v)?;
                crate::collapse::validate(v)?;
                r.collapse = v.get("field").and_then(Value::as_str).map(str::to_string);
                r.collapse_spec = Some(v.clone());
            }
            "rescore" => {
                crate::search_strict::check_rescore(v)?;
                r.rescore = Some(v.clone())
            }
            "profile" => r.profile = bool_value(v, "profile")?,
            "runtime_mappings" => r.runtime_mappings = Some(v.clone()),
            "script_fields" => {
                if let Some(sf) = v.as_object() {
                    crate::search_strict::check_script_fields(sf)?;
                }
                r.script_fields = Some(v.clone());
                // ES omits `_source` when script_fields are asked for without it.
                if !o.contains_key("_source") {
                    r.source = SourceFilter::None;
                }
            }
            "suggest" => {
                if let Some(so) = v.as_object() {
                    crate::search_strict::check_suggest(so)?;
                }
                r.suggest = Some(v.clone())
            }
            "retriever" => {
                crate::search_strict::check_retriever(v)?;
                r.retriever = Some(v.clone())
            }
            "rank" => r.rank = Some(v.clone()),
            "sub_searches" => r.sub_searches = Some(v.clone()),
            "slice" => {
                if let Some(so) = v.as_object() {
                    for (sk, _) in so {
                        if !matches!(sk.as_str(), "id" | "max" | "field") {
                            return Err(EsError::unknown_field_xc("slice", sk, Some("slice"), &["id", "max", "field"]));
                        }
                    }
                }
                let n = |k: &str| v.get(k).and_then(|x| x.as_u64().or_else(|| x.as_str().and_then(|s| s.parse().ok())));
                if let (Some(id), Some(max)) = (n("id"), n("max")) {
                    if max < 2 || id >= max {
                        return Err(EsError::illegal_argument(if max < 2 { "max must be greater than 1".to_string() } else { "max must be greater than id".to_string() }));
                    }
                    r.slice = Some((id as u32, max as u32));
                }
            }
            "pit" => {
                if let Some(po) = v.as_object() {
                    crate::search_strict::check_pit(po)?;
                }
            }
            "indices_boost" => crate::search_strict::check_indices_boost(v)?,
            "stats" => {
                if let Some(bad) = v.as_array().into_iter().flatten().find(|x| !x.is_string()) {
                    return Err(EsError::parsing(format!("Expected [VALUE_STRING] in [stats] but found [{}]", crate::jsonpos::token_name(bad))).at_key("stats", None).with_lc());
                }
            }
            "ext" => {}
            other => return Err(EsError::unknown_key(other, v, None)),
        }
    }
    if o.contains_key("retriever") && o.contains_key("rank") && !crate::version::current_rest_api_version().is_v8() {
        // ES 9 turns `rank` into a retriever while parsing (`SearchSourceBuilder.parseXContent`).
        return Err(EsError::illegal_argument("Cannot specify both [rank] and [retriever]."));
    }
    if o.contains_key("retriever") {
        let mut specified = Vec::new();
        if o.contains_key("query") || o.contains_key("sub_searches") {
            specified.push("query");
        }
        for k in ["knn", "search_after", "terminate_after", "sort", "rescore", "rank"] {
            if o.contains_key(k) {
                specified.push(k);
            }
        }
        if !specified.is_empty() {
            return Err(EsError::action_validation(&[&format!("cannot specify [retriever] and [{}]", specified.join(", "))]));
        }
    }
    if o.contains_key("retriever") {
        crate::retriever::validate_top_level(o)?;
    }
    // URL parameters override the body.
    if let Some(q) = params.get("q") {
        let mut qs = Map::new();
        qs.insert("query".into(), json!(q));
        for (p, k) in [("df", "default_field"), ("default_operator", "default_operator"), ("analyzer", "analyzer"), ("lenient", "lenient"), ("analyze_wildcard", "analyze_wildcard")] {
            if let Some(v) = params.get(p) {
                qs.insert(k.into(), if k == "lenient" || k == "analyze_wildcard" { json!(v == "true") } else { json!(v) });
            }
        }
        r.query = Some(json!({ "query_string": qs }));
    }
    if let Some(v) = params.get("from") {
        r.from = usize_of(&json!(v), "from")?;
    }
    if let Some(v) = params.get("size") {
        r.size = usize_of(&json!(v), "size")?;
        r.size_explicit = true;
    }
    if let Some(v) = params.get("sort") {
        r.sort = parse_sort(&json!(v))?;
    }
    if let Some(v) = params.get("pre_filter_shard_size") {
        r.pre_filter_shard_size = v.parse::<usize>().ok();
    }
    // `SearchRequest.setBatchedReduceSize`.
    if let Some(v) = params.get("batched_reduce_size") {
        match v.parse::<i32>() {
            Ok(n) if n <= 1 => return Err(EsError::illegal_argument("batchedReduceSize must be >= 2")),
            Ok(_) => {}
            Err(_) => return Err(EsError::illegal_argument(format!("Failed to parse int parameter [batched_reduce_size] with value [{v}]"))),
        }
    }
    // `RestSearchAction` reads it with `paramAsInt` and
    // `SearchRequest.setMaxConcurrentShardRequests` rejects values below 1.
    if let Some(v) = params.get("max_concurrent_shard_requests") {
        match v.parse::<i32>() {
            Ok(n) if n < 1 => return Err(EsError::illegal_argument("maxConcurrentShardRequests must be >= 1")),
            Ok(_) => {}
            Err(_) => return Err(EsError::illegal_argument(format!("Failed to parse int parameter [max_concurrent_shard_requests] with value [{v}]"))),
        }
    }
    if let Some(v) = params.get("_source") {
        r.source = match v.as_str() {
            "true" => SourceFilter::All,
            "false" => SourceFilter::None,
            s => SourceFilter::Filter { includes: str_list(&json!(s)), excludes: vec![] },
        };
    }
    if let Some(v) = params.get("_source_exclude_vectors") {
        r.exclude_vectors = Some(v == "true" || v.is_empty());
    }
    let inc = params.get("_source_includes").or(params.get("_source_include")).map(|s| str_list(&json!(s)));
    let exc = params.get("_source_excludes").or(params.get("_source_exclude")).map(|s| str_list(&json!(s)));
    if inc.is_some() || exc.is_some() {
        r.source = SourceFilter::Filter { includes: inc.unwrap_or_default(), excludes: exc.unwrap_or_default() };
    }
    if let Some(v) = params.get("track_total_hits") {
        r.track_total_hits = track_total(&json!(v))?;
    }
    if params.get("force_synthetic_source").map(|s| s == "true" || s.is_empty()).unwrap_or(false) {
        r.force_synthetic_source = true;
    }
    if params.get("include_named_queries_score").map(|s| s == "true" || s.is_empty()).unwrap_or(false) {
        r.include_named_queries_score = true;
    }
    if params.get("rest_total_hits_as_int").map(|s| s == "true").unwrap_or(false) {
        r.rest_total_hits_as_int = true;
        // ES: accurate unless explicitly disabled (`-1` total); a limit is an error.
        let explicit = params.contains_key("track_total_hits") || o.contains_key("track_total_hits");
        match (explicit, r.track_total_hits) {
            (true, None) => {}
            (true, Some(TrackTotalHits::UpTo(n))) if n != i32::MAX as u64 => {
                return Err(EsError::illegal_argument(format!("[rest_total_hits_as_int] cannot be used if the tracking of total hits is not accurate, got {n}")));
            }
            _ => r.track_total_hits = Some(TrackTotalHits::True),
        }
    }
    for (p, f) in [("explain", &mut r.explain), ("version", &mut r.version), ("seq_no_primary_term", &mut r.seq_no_primary_term), ("typed_keys", &mut r.typed_keys), ("track_scores", &mut r.track_scores)] {
        if let Some(v) = params.get(p) {
            *f = v == "true" || v.is_empty();
        }
    }
    if params.get("search_type").map(|s| s == "dfs_query_then_fetch").unwrap_or(false) {
        r.dfs = true;
    }
    if let Some(v) = params.get("terminate_after") {
        r.terminate_after = Some(usize_of(&json!(v), "terminate_after")? as u64).filter(|x| *x > 0);
    }
    if let Some(v) = params.get("timeout") {
        r.timeout = crate::date::parse_time_value(v).filter(|x| *x > 0).map(|x| Duration::from_millis(x as u64));
    }
    if let Some(v) = params.get("docvalue_fields") {
        r.docvalue_fields = field_list(&json!(v));
    }
    if let Some(v) = params.get("stored_fields") {
        if v == "_none_" {
            r.stored_fields_none = true;
            r.source = SourceFilter::None;
        }
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// Execution.

/// Sort criterion resolved against one index.
struct ResolvedSort {
    field: SortField,
    /// `geo` sorts: meters per rendered unit.
    scale: f64,
    /// How to render sort values: `long`, `double`, `keyword`, `ip`, `date`, `score`, `doc`, `bool`.
    render: &'static str,
    descending: bool,
    missing_first: bool,
}

fn missing_value(v: &Value, render: &str, f: Option<&crate::mapping::ResolvedField>) -> Result<SortMissing> {
    Ok(match v {
        Value::String(s) if s == "_last" => SortMissing::Last,
        Value::String(s) if s == "_first" => SortMissing::First,
        other => SortMissing::Value(match render {
            "long" | "date" => SortValue::I64(match (other, f) {
                (Value::Number(n), _) => n.as_f64().unwrap_or(0.0) as i64,
                (Value::String(s), Some(f)) if f.ty.is_date() => f.date_format.parse(s)?,
                (Value::String(s), _) => s.parse::<f64>().map_err(|_| EsError::illegal_argument(format!("Invalid missing value [{s}]")))? as i64,
                _ => 0,
            }),
            "double" => SortValue::F64(other.as_f64().or_else(|| other.as_str().and_then(|s| s.parse().ok())).unwrap_or(0.0)),
            "unsigned_long" => SortValue::I64(unsigned_sort_key(other)),
            "version" => SortValue::Str(rustic_core::fieldtypes::version::encode(&crate::query::dsl::scalar_string(other))),
            "tsid" => SortValue::Str(crate::tsdb::tsid_stored(&crate::query::dsl::scalar_string(other))),
            _ => SortValue::Str(crate::query::dsl::scalar_string(other)),
        }),
    })
}

/// ES `_geo_distance` sort → core distance sort (meters; rendered in `unit`).
fn resolve_geo_sort(spec: &Map<String, Value>, s: &SortSpec, lookup: &FieldLookup, index: &str, uuid: &str) -> Result<ResolvedSort> {
    let (name, origin) = spec
        .iter()
        .find(|(k, _)| !matches!(k.as_str(), "order" | "unit" | "mode" | "distance_type" | "ignore_unmapped" | "validation_method" | "nested" | "missing"))
        .ok_or_else(|| EsError::parsing("[_geo_distance] requires a geo_point field and an origin"))?;
    let origin = match origin {
        Value::Array(a) if !values::is_point_array(a) => a.first().cloned().unwrap_or(Value::Null),
        other => other.clone(),
    };
    let (lat, lon) = values::parse_geo_point(&origin).ok_or_else(|| EsError::parsing(format!("failed to parse [_geo_distance] origin [{origin}]")))?;
    let unit = spec.get("unit").and_then(Value::as_str).unwrap_or("m");
    let scale = values::unit_meters(unit).ok_or_else(|| EsError::illegal_argument(format!("No distance unit match [{unit}]")))?;
    let plane = spec.get("distance_type").and_then(Value::as_str) == Some("plane");
    let ignore_unmapped = spec.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    let field = match lookup.get(name) {
        Some(f) if f.ty == EsType::GeoPoint => f.geo_id.ok_or_else(|| EsError::query_shard(index, uuid, format!("field [{name}] has no geo index; reindex to sort by distance")))?,
        Some(f) => return Err(EsError::query_shard(index, uuid, format!("field [{name}] of type [{}] is not supported for geo distance sorting", f.ty.name()))),
        None if ignore_unmapped => u16::MAX,
        None => return Err(EsError::query_shard(index, uuid, format!("failed to find mapper for [{name}] for geo distance based sort"))),
    };
    let mode = match s.mode.as_deref() {
        Some("min") => Some(SortMode::Min),
        Some("max") => Some(SortMode::Max),
        Some("avg") => Some(SortMode::Avg),
        Some("median") => Some(SortMode::Median),
        Some(o) => return Err(EsError::illegal_argument(format!("Unsupported sort mode [{o}] for geo distance sorting"))),
        None => None,
    };
    Ok(ResolvedSort { field: SortField::GeoDistance { field, lat, lon, descending: s.descending, mode, plane }, scale, render: "geo", descending: s.descending, missing_first: false })
}

/// Wrap a resolved field sort in a nested sort when the request gives a
/// `nested` context (or the field lives under a nested path).
fn wrap_nested_sort(rs: &mut ResolvedSort, nested: Option<&Value>, field_nested_path: Option<String>, qctx: &QueryContext) -> Result<()> {
    let (path, filter) = match nested {
        Some(n) => {
            let path = n.get("path").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[nested] sort requires a [path]"))?;
            let filter = match n.get("filter") {
                Some(f) => Some(Box::new(parse_query(f, qctx)?)),
                None => None,
            };
            if n.get("nested").is_some() {
                return Err(EsError::illegal_argument("multi-level nested sort contexts are not supported"));
            }
            (path.to_string(), filter)
        }
        None => match field_nested_path {
            Some(p) => (p, None),
            None => return Ok(()),
        },
    };
    let inner = std::mem::replace(&mut rs.field, SortField::Doc);
    rs.field = SortField::Nested { path, filter, sort: Box::new(inner) };
    Ok(())
}

fn resolve_sort(specs: &[SortSpec], lookup: &FieldLookup, index: &str, uuid: &str, extra: &HashMap<String, Value>, qctx: &QueryContext) -> Result<Vec<ResolvedSort>> {
    let mut out = Vec::new();
    for s in specs {
        match s.field.as_str() {
            "_score" => out.push(ResolvedSort { field: SortField::Score { descending: s.descending }, scale: 1.0, render: "score", descending: s.descending, missing_first: false }),
            "_doc" | "_shard_doc" => out.push(ResolvedSort { field: if s.descending { SortField::DocDesc } else { SortField::Doc }, scale: 1.0, render: "doc", descending: s.descending, missing_first: false }),
            // `_index`: one value per index (no doc carries it: every doc gets
            // it as its missing value).
            "_index" => out.push(ResolvedSort {
                field: SortField::Field { field: u16::MAX, kind: SortKind::Keyword, descending: s.descending, missing: SortMissing::Value(SortValue::Str(index.to_string())), mode: None },
                scale: 1.0,
                render: "keyword",
                descending: s.descending,
                missing_first: false,
            }),
            "_geo_distance" => {
                let spec = extra.get("_geo_distance").and_then(Value::as_object).cloned().unwrap_or_default();
                out.push(resolve_geo_sort(&spec, s, lookup, index, uuid)?);
                wrap_nested_sort(out.last_mut().unwrap(), spec.get("nested"), None, qctx)?;
            }
            // Time series indices: the 4-byte routing hash of `_id`.
            "_ts_routing_hash" if lookup.get(crate::tsdb::TSID_FIELD).is_some() => {
                out.push(ResolvedSort { field: SortField::Doc, scale: 1.0, render: "ts_routing_hash", descending: s.descending, missing_first: false });
            }
            "_script" => {
                let render = crate::script::search::script_sort_render(extra.get("_script"))?;
                out.push(ResolvedSort { field: SortField::Doc, scale: 1.0, render, descending: s.descending, missing_first: false });
            }
            "_id" => out.push(ResolvedSort {
                scale: 1.0,
                field: SortField::Field { field: ID_FIELD, kind: SortKind::Keyword, descending: s.descending, missing: SortMissing::Last, mode: None },
                render: "keyword",
                descending: s.descending,
                missing_first: false,
            }),
            "_seq_no" if qctx.seq_no_disabled => return Err(EsError::illegal_argument(SEQ_NO_DISABLED)),
            "_seq_no" => out.push(ResolvedSort {
                scale: 1.0,
                field: SortField::Field { field: SEQ_NO_FIELD, kind: SortKind::I64, descending: s.descending, missing: SortMissing::Last, mode: None },
                render: "long",
                descending: s.descending,
                missing_first: false,
            }),
            "_primary_term" => out.push(ResolvedSort {
                scale: 1.0,
                field: SortField::Field { field: PRIMARY_TERM_FIELD, kind: SortKind::I64, descending: s.descending, missing: SortMissing::Last, mode: None },
                render: "long",
                descending: s.descending,
                missing_first: false,
            }),
            name => {
                let f = lookup.get(name);
                let (id, kind, render) = match f {
                    Some(f) => {
                        let (kind, render) = match f.ty {
                            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => (SortKind::I64, "long"),
                            // Sorted as shifted longs (see rustic-core `field_value_docs`).
                            EsType::UnsignedLong => (SortKind::I64, "unsigned_long"),
                            EsType::Date | EsType::DateNanos => (SortKind::I64, "date"),
                            // Booleans sort (and render) as 0/1, like ES.
                            EsType::Boolean => (SortKind::I64, "long"),
                            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => (SortKind::F64, "double"),
                            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened => (SortKind::Keyword, "keyword"),
                            EsType::Ip => (SortKind::Keyword, "ip"),
                            EsType::MatchOnlyText if f.declared == EsType::PatternText => {
                                return Err(EsError::illegal_argument("pattern_text fields do not support sorting and aggregations"));
                            }
                            EsType::Text | EsType::MatchOnlyText if !f.fielddata => {
                                return Err(EsError::query_shard(index, uuid, format!(
                                    "Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [{name}] in order to load field data by uninverting the inverted index. Note that this can use significant memory."
                                )));
                            }
                            // No doc-values column: sort on the uninverted terms.
                            EsType::Text | EsType::MatchOnlyText => {
                                crate::fielddata_cache::mark(index, name);
                                (SortKind::Fielddata, "keyword")
                            }
                            EsType::AggregateMetricDouble => (SortKind::F64, "double"),
                            _ => return Err(EsError::query_shard(index, uuid, format!("can't sort on field [{name}] of type [{}]", f.ty.name()))),
                        };
                        // aggregate_metric_double: sorted on its default metric.
                        if f.declared == EsType::AggregateMetricDouble {
                            let Some(m) = crate::fieldtypes::aggs_default_metric(f) else {
                                return Err(EsError::query_shard(index, uuid, format!(
                                    "Sorting by field [{name}] of type [aggregate_metric_double] is not supported unless it has configured a single metric or the [sum, value_count] metrics"
                                )));
                            };
                            (f.aux_id(&m), SortKind::F64, "double")
                        } else if f.declared == EsType::Version {
                            if let Some(sid) = f.aux_id("sort") {
                                (Some(sid), kind, "version")
                            } else {
                                (f.core_id, kind, render)
                            }
                        } else if name == crate::tsdb::TSID_FIELD {
                            (f.core_id, kind, "tsid")
                        } else {
                            (f.core_id, kind, render)
                        }
                    }
                    None if crate::script::search::runtime_render(index, name).is_some() => {
                        crate::script::search::check_fielddata(index, name).map_err(|e| EsError::query_shard(index, uuid, e.reason))?;
                        (None, SortKind::Keyword, crate::script::search::runtime_render(index, name).unwrap_or("keyword"))
                    }
                    None => match &s.unmapped_type {
                        Some(t) => {
                            let render = match t.as_str() {
                                "long" | "integer" | "short" | "byte" => "long",
                                "date" => "date",
                                "double" | "float" => "double",
                                _ => "keyword",
                            };
                            (None, SortKind::Keyword, render)
                        }
                        None => return Err(EsError::query_shard(index, uuid, format!("No mapping found for [{name}] in order to sort on"))),
                    },
                };
                let missing = match (&s.missing, f.and_then(|f| f.constant_value.as_ref().filter(|_| f.declared == EsType::ConstantKeyword))) {
                    (Some(v), _) => missing_value(v, render, f.map(|x| &**x))?,
                    // constant_keyword: every doc of the index holds the constant.
                    (None, Some(c)) => SortMissing::Value(SortValue::Str(crate::query::dsl::scalar_string(c))),
                    (None, None) => SortMissing::Last,
                };
                let mode = match s.mode.as_deref() {
                    Some("min") => Some(SortMode::Min),
                    Some("max") => Some(SortMode::Max),
                    Some("sum") => Some(SortMode::Sum),
                    Some("avg") => Some(SortMode::Avg),
                    Some("median") => Some(SortMode::Median),
                    Some(o) => return Err(EsError::illegal_argument(format!("Unknown SortMode [{o}]"))),
                    None => None,
                };
                let missing_first = matches!(missing, SortMissing::First);
                out.push(ResolvedSort {
                    // Unmapped: a field id no leaf has → every doc missing.
                    field: SortField::Field { field: id.unwrap_or(u16::MAX), kind, descending: s.descending, missing, mode },
                    // Date sorts: 1e6 marks nanosecond values (`date_nanos` fields).
                    // Integer-typed fields (integer/short/byte) sort as 32-bit ints: -1.0 marks them.
                    scale: if render == "date" && f.is_some_and(|f| f.ty == EsType::DateNanos) {
                        1_000_000.0
                    } else if f.is_some_and(|f| matches!(f.ty, EsType::Integer | EsType::Short | EsType::Byte)) {
                        -1.0
                    } else {
                        1.0
                    },
                    render,
                    descending: s.descending,
                    missing_first,
                });
                let nested_path = f.and_then(|f| f.nested_path.clone());
                wrap_nested_sort(out.last_mut().unwrap(), extra.get(name).and_then(|e| e.get("nested")), nested_path, qctx)?;
            }
        }
    }
    Ok(out)
}

fn after_value(v: &Value, render: &str, lookup: &FieldLookup, spec: &SortSpec) -> Result<SortValue> {
    Ok(match (render, v) {
        (_, Value::Null) => SortValue::Missing,
        ("date", Value::Number(n)) if spec.numeric_type.is_some() => {
            let x = n.as_i64().unwrap_or(n.as_f64().unwrap_or(0.0) as i64);
            let nanos_field = lookup.get(&spec.field).is_some_and(|f| f.ty == EsType::DateNanos);
            SortValue::I64(match (spec.numeric_type.as_deref(), nanos_field) {
                (Some("date"), true) => x.saturating_mul(1_000_000),
                (Some("date_nanos"), false) => x / 1_000_000,
                _ => x,
            })
        }
        ("long" | "date" | "doc", Value::Number(n)) => SortValue::I64(n.as_i64().unwrap_or(n.as_f64().unwrap_or(0.0) as i64)),
        // The base64 routing hash bytes (ES parses the BytesRef).
        ("ts_routing_hash", Value::String(s)) => {
            use base64::Engine as _;
            let b = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(s.trim_end_matches('=').as_bytes())
                .map_err(|_| EsError::illegal_argument(format!("Failed to parse search_after value [{s}] for field [_ts_routing_hash]")))?;
            SortValue::Str(b.iter().map(|x| *x as char).collect())
        }
        ("date", Value::String(s)) => {
            let f = lookup.get(&spec.field);
            let fmt = match spec.format.as_deref() {
                Some(sf) => Some(crate::date::DateFormatter::new(sf)?),
                None => f.map(|f| f.date_format.clone()),
            };
            let m = match fmt.as_ref().map(|f| f.parse(s)) {
                Some(Ok(m)) => m,
                Some(Err(e)) => match s.parse::<i64>() {
                    Ok(n) => n,
                    Err(_) => return Err(e),
                },
                None => s.parse().unwrap_or(0),
            };
            match f {
                Some(f) if f.ty == EsType::DateNanos && spec.format.is_some() => SortValue::I64(m * 1_000_000),
                Some(f) if f.ty == EsType::DateNanos && s.parse::<i64>().is_err() => SortValue::I64(m * 1_000_000),
                _ => SortValue::I64(m),
            }
        }
        ("long", Value::String(s)) => SortValue::I64(s.parse().unwrap_or(0)),
        ("unsigned_long", v) => SortValue::I64(unsigned_sort_key(v)),
        ("double" | "score", v) => {
            let x = v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(0.0);
            // A scaled_float sorts on stored quotients but renders ES's doc
            // value (`render_sort_value`): a rendered value maps back to its
            // stored key, so the cursor ties the doc it came from.
            match lookup.get(&spec.field).filter(|f| render == "double" && f.ty == EsType::ScaledFloat && f.scaling_factor.is_finite() && f.scaling_factor > 0.0) {
                Some(f) => {
                    let l = (x * f.scaling_factor).round();
                    let stored = l / f.scaling_factor;
                    SortValue::F64(if crate::aggs::es_scaled(stored, Some((f.scaling_factor, 1.0 / f.scaling_factor))) == x { stored } else { x })
                }
                None => SortValue::F64(x),
            }
        }
        ("ip", Value::String(s)) => SortValue::Str(values::parse_ip(s).map(|b| values::ip_to_term(&b)).unwrap_or_else(|| s.clone())),
        ("version", v) => SortValue::Str(rustic_core::fieldtypes::version::encode(&crate::query::dsl::scalar_string(v))),
        ("tsid", v) => SortValue::Str(crate::tsdb::tsid_stored(&crate::query::dsl::scalar_string(v))),
        (_, v) => SortValue::Str(crate::query::dsl::scalar_string(v)),
    })
}

/// Missing values of integer/short/byte sorts are `Integer.MAX_VALUE` (not `Long.MAX_VALUE`),
/// which also orders them before the missing values of `long` fields in a cross-index sort.
/// ES `SortFieldValidation`: shards (indices) that returned hits must agree
/// on each sort field's Lucene sort type, except INT/LONG mixes and numeric
/// mixes involving floating point.
/// Whether the search is sorted by fields (ES `TopFieldCollector`) rather
/// than by relevance (no sort, or `_score` descending alone).
fn sorted_by_field(req: &SearchRequest) -> bool {
    !req.sort.is_empty() && !(req.sort.len() == 1 && req.sort[0].field == "_score" && req.sort[0].descending)
}

/// Whether hits show `_score`: relevance order, `track_scores`, or a
/// `_score` sort key (ES `SearchPhaseController` copies the score sort
/// value into the hit).
fn hit_scores_shown(req: &SearchRequest) -> bool {
    !sorted_by_field(req) || req.track_scores || req.sort.iter().any(|s| s.field == "_score")
}

/// Whether hits need their scores: shown (`_score` / `max_score`) unless
/// the search is sorted by fields without `track_scores` or `_score`.
pub(crate) fn scores_needed(req: &SearchRequest) -> bool {
    hit_scores_shown(req) || req.rank_plan.is_some() || req.min_score.is_some()
}

fn validate_sort_types(runs: &[IndexRun], req: &SearchRequest) -> Result<()> {
    #[derive(PartialEq, Clone, Copy)]
    enum T {
        Str,
        Int,
        Long,
        Double,
    }
    let name = |t: T| match t {
        T::Str => "STRING",
        T::Int => "INT",
        T::Long => "LONG",
        T::Double => "DOUBLE",
    };
    let with_hits: Vec<&IndexRun> = runs.iter().filter(|r| !r.top.hits.is_empty()).collect();
    if with_hits.len() < 2 {
        return Ok(());
    }
    let ty = |s: &ResolvedSort| -> Option<T> {
        let SortField::Field { field, kind, .. } = &s.field else { return None };
        if *field == u16::MAX {
            return None;
        }
        Some(match (kind, s.render) {
            (SortKind::Keyword, "keyword" | "ip" | "version") | (SortKind::Fielddata, "keyword") => T::Str,
            (SortKind::I64, "long") if s.scale < 0.0 => T::Int,
            (SortKind::I64, "long" | "date" | "unsigned_long") => T::Long,
            (SortKind::F64, "double") => T::Double,
            _ => return None,
        })
    };
    for i in 0..req.sort.len() {
        let mut first: Option<T> = None;
        for r in &with_hits {
            let Some(cur) = r.sort.get(i).and_then(ty) else { return Ok(()) };
            match first {
                None => first = Some(cur),
                Some(f) if f == cur => {}
                Some(f) => {
                    let numeric = |t: T| t != T::Str;
                    if numeric(f) && numeric(cur) {
                        continue;
                    }
                    return Err(EsError::illegal_argument(format!(
                        "Can't sort on field [{}]; the field has incompatible sort types: [{}] and [{}] across shards!",
                        req.sort[i].field,
                        name(f),
                        name(cur)
                    )));
                }
            }
        }
    }
    Ok(())
}

fn normalise_int_sorts(runs: &mut [IndexRun]) {
    for r in runs.iter_mut() {
        for i in 0..r.sort.len() {
            if r.sort[i].scale < 0.0 && !r.sort[i].descending && !r.sort[i].missing_first {
                for h in r.top.hits.iter_mut() {
                    if let Some(v @ SortValue::Missing) = h.sort.get_mut(i) {
                        *v = SortValue::I64(i32::MAX as i64);
                    }
                }
            }
        }
    }
}

/// Date sort values of every run in the `numeric_type` unit, so cross-index merging compares
/// like with like; the run's sort spec records the unit for rendering.
fn normalise_date_sorts(runs: &mut [IndexRun], req: &SearchRequest) {
    for (i, spec) in req.sort.iter().enumerate() {
        let is_date = |r: &IndexRun| r.sort.get(i).is_some_and(|s| s.render == "date");
        if !runs.iter().any(is_date) {
            continue;
        }
        // Without `numeric_type` ES compares the raw longs of each index.
        let target_nanos = match spec.numeric_type.as_deref() {
            Some("date") => false,
            Some("date_nanos") => true,
            _ => continue,
        };
        for r in runs.iter_mut() {
            if !is_date(r) {
                continue;
            }
            let run_nanos = r.sort[i].scale > 1.5;
            if run_nanos != target_nanos {
                for h in r.top.hits.iter_mut() {
                    if let Some(SortValue::I64(x)) = h.sort.get_mut(i) {
                        *x = if target_nanos { x.saturating_mul(1_000_000) } else { *x / 1_000_000 };
                    }
                }
            }
            r.sort[i].scale = if target_nanos { 1_000_000.0 } else { 1.0 };
        }
    }
}

/// An unsigned long as its sort key: shifted by Long.MIN_VALUE (ES
/// `UnsignedLongFieldMapper`), so it orders as a signed long.
fn unsigned_sort_key(v: &Value) -> i64 {
    let u = match v {
        Value::Number(n) => n.as_u64().unwrap_or_else(|| n.as_f64().map(|f| if f <= 0.0 { 0 } else { f as u64 }).unwrap_or(0)),
        Value::String(s) => s.trim().parse::<u64>().unwrap_or(0),
        _ => 0,
    };
    (u ^ (1u64 << 63)) as i64
}

/// Whether a missing value sorts as the lowest value (ascending space):
/// `_first` ascending or `_last` descending (ES's default is `_last`).
fn missing_sorts_low(spec: Option<&SortSpec>) -> bool {
    let first = spec.and_then(|s| s.missing.as_ref()).and_then(Value::as_str) == Some("_first");
    first != spec.is_some_and(|s| s.descending)
}

fn render_sort_value(v: &SortValue, render: &str, lookup: &FieldLookup, spec: Option<&SortSpec>, score: f32, nanos: bool) -> Value {
    match (v, render) {
        (_, "score") => values::java_double(score as f64),
        (SortValue::Missing, "date") if spec.is_some_and(|s| s.format.is_some()) => Value::Null,
        // ES renders a missing value as the one it sorts as
        // (`XFieldComparatorSource.missingObject`): the end it goes to,
        // Long.MIN / MAX, -/+Infinity (the comparator ties a real such value
        // with it, the tiebreaker deciding).
        (SortValue::Missing, "long" | "date") => Value::from(if missing_sorts_low(spec) { i64::MIN } else { i64::MAX }),
        // `numeric_type: long` on a floating-point field: ES casts the
        // field's values to longs (`(long) value`, toward zero), so a missing
        // value renders as the long end it sorts as.
        (SortValue::Missing, "double") if spec.is_some_and(|s| s.numeric_type.as_deref() == Some("long")) => Value::from(if missing_sorts_low(spec) { i64::MIN } else { i64::MAX }),
        (SortValue::Missing, "double") => Value::String(if missing_sorts_low(spec) { "-Infinity" } else { "Infinity" }.into()),
        // Long.MAX_VALUE shifted back: the largest unsigned long.
        (SortValue::Missing, "unsigned_long") => Value::from(u64::MAX),
        (SortValue::I64(i), "unsigned_long") => Value::from(*i as u64 ^ (1u64 << 63)),
        (SortValue::Missing, _) => Value::Null,
        (SortValue::I64(i), "date") => match spec.and_then(|s| s.format.as_deref()) {
            Some(fmt) => crate::date::DateFormatter::new(fmt).map(|f| Value::String(if nanos { f.format_nanos(*i, crate::date::Tz::UTC) } else { f.format(*i, crate::date::Tz::UTC) })).unwrap_or(Value::from(*i)),
            None => Value::from(*i),
        },
        (SortValue::I64(i), _) => Value::from(*i),
        (SortValue::F64(x), "double") if spec.is_some_and(|s| s.numeric_type.as_deref() == Some("long")) => {
            let x = match spec.and_then(|s| lookup.get(&s.field)).filter(|f| f.ty == EsType::ScaledFloat && f.scaling_factor.is_finite() && f.scaling_factor > 0.0) {
                Some(f) => crate::aggs::es_scaled(*x, Some((f.scaling_factor, 1.0 / f.scaling_factor))),
                None => *x,
            };
            // Java's `(long) value`: toward zero, saturating, NaN as 0.
            Value::from(x as i64)
        }
        (SortValue::F64(x), "double") => match spec.and_then(|s| lookup.get(&s.field)).filter(|f| f.ty == EsType::ScaledFloat && f.scaling_factor.is_finite() && f.scaling_factor > 0.0) {
            // ES sorts and renders a scaled_float's doc values: the scaled
            // long times the inverse factor, not the stored quotient.
            Some(f) => values::java_double(crate::aggs::es_scaled(*x, Some((f.scaling_factor, 1.0 / f.scaling_factor)))),
            None => values::java_double(*x),
        },
        (SortValue::F64(f), _) => values::java_double(*f),
        (SortValue::Str(s), "ip") => Value::String(values::ip_display(s)),
        (SortValue::Str(s), "version") => Value::String(rustic_core::fieldtypes::version::decode(s)),
        (SortValue::Str(s), "tsid") => Value::String(crate::tsdb::tsid_display(s)),
        (SortValue::Str(s), "ts_routing_hash") => {
            use base64::Engine as _;
            let bytes: Vec<u8> = s.chars().map(|c| c as u32 as u8).collect();
            Value::String(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
        }
        (SortValue::Str(s), _) => {
            let _ = lookup;
            Value::String(s.clone())
        }
    }
}

/// `_shard_doc` sort values: bits from here up hold the shard copy's
/// position among the request's shards, below them `(leaf << 32) | doc`
/// (ES `ShardDocSortField`).
const SHARD_DOC_SHIFT: u32 = 44;

/// A `_shard_doc` search_after value `g` seen from the shard at `ord`:
/// `Ok(local)` when `g` points into this shard, otherwise the sentinel that
/// keeps every (or no) document of this shard.
fn shard_doc_after(g: i64, ord: i64, desc: bool) -> std::result::Result<i64, i64> {
    let t = g >> SHARD_DOC_SHIFT;
    if t == ord {
        return Ok(g & ((1i64 << SHARD_DOC_SHIFT) - 1));
    }
    let (none, all) = if desc { (-1, i64::MAX) } else { (i64::MAX, -1) };
    Err(if (ord < t) != desc { none } else { all })
}

/// Whether `v` is the value a missing one sorts as, `low`: at the low end
/// (ascending space).
fn is_sentinel(v: &SortValue, low: bool) -> bool {
    match v {
        SortValue::I64(x) => *x == if low { i64::MIN } else { i64::MAX },
        SortValue::F64(x) => *x == if low { f64::NEG_INFINITY } else { f64::INFINITY },
        _ => false,
    }
}

fn cmp_sort(a: &SortValue, b: &SortValue, desc: bool, missing_first: bool) -> Ordering {
    use SortValue::*;
    let o = match (a, b) {
        (Missing, Missing) => return Ordering::Equal,
        // A real value equal to the one a missing value sorts as ties with
        // it (ES substitutes the sentinel).
        (Missing, v) | (v, Missing) if is_sentinel(v, missing_first != desc) => return Ordering::Equal,
        (Missing, _) => return if missing_first { Ordering::Less } else { Ordering::Greater },
        (_, Missing) => return if missing_first { Ordering::Greater } else { Ordering::Less },
        (I64(x), I64(y)) => x.cmp(y),
        (F64(x), F64(y)) => x.total_cmp(y),
        (I64(x), F64(y)) => (*x as f64).total_cmp(y),
        (F64(x), I64(y)) => x.total_cmp(&(*y as f64)),
        (Str(x), Str(y)) => x.cmp(y),
        (Str(_), _) => Ordering::Greater,
        (_, Str(_)) => Ordering::Less,
    };
    if desc { o.reverse() } else { o }
}


/// `_source` filtering with ES include/exclude path semantics.
pub fn filter_source(v: &Value, includes: &[String], excludes: &[String]) -> Value {
    fn matches(p: &str, path: &str) -> bool {
        crate::mapping::wildcard_match(p, path)
    }
    fn walk(v: &Value, path: &str, inc: &[String], exc: &[String]) -> Option<Value> {
        if !path.is_empty() && exc.iter().any(|e| matches(e, path) || path.starts_with(&format!("{e}."))) {
            return None;
        }
        match v {
            Value::Object(m) => {
                let mut out = Map::new();
                for (k, c) in m {
                    let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                    if let Some(x) = walk(c, &p, inc, exc) {
                        out.insert(k.clone(), x);
                    }
                }
                let self_included = !path.is_empty() && (inc.is_empty() || inc.iter().any(|i| matches(i, path) || path.starts_with(&format!("{i}."))));
                if out.is_empty() && !self_included && !path.is_empty() {
                    return None;
                }
                if out.is_empty() && !m.is_empty() && !self_included {
                    return if path.is_empty() { Some(Value::Object(out)) } else { None };
                }
                Some(Value::Object(out))
            }
            Value::Array(a) if a.iter().any(Value::is_object) => {
                let items: Vec<Value> = a.iter().filter_map(|x| if x.is_object() { walk(x, path, inc, exc) } else if inc.is_empty() || inc.iter().any(|i| matches(i, path) || path.starts_with(&format!("{i}."))) { Some(x.clone()) } else { None }).collect();
                let items: Vec<Value> = items.into_iter().filter(|x| !(x.is_object() && x.as_object().unwrap().is_empty()) || inc.is_empty()).collect();
                if items.is_empty() { None } else { Some(Value::Array(items)) }
            }
            leaf => {
                if inc.is_empty() || inc.iter().any(|i| matches(i, path) || path.starts_with(&format!("{i}."))) {
                    Some(leaf.clone())
                } else {
                    None
                }
            }
        }
    }
    walk(v, "", includes, excludes).unwrap_or_else(|| json!({}))
}

/// `{"a": {"b": v}}` for the dotted `path` `a.b`.
fn wrap_in_path(path: &str, v: Value) -> Value {
    path.rsplit('.').fold(v, |acc, seg| json!({ seg: acc }))
}


/// Like `highlight::source_values` but visiting object keys in document order
/// (dotted keys at any level), as the fields API does.
fn source_values_ordered<'v>(src: &'v Value, path: &str) -> Vec<&'v Value> {
    source_values_ordered_with(src, path, false)
}

/// [`source_values_ordered`], optionally keeping explicit `null`s (fields
/// with a `null_value` return it for them).
fn source_values_ordered_with<'v>(src: &'v Value, path: &str, nulls: bool) -> Vec<&'v Value> {
    fn walk<'v>(v: &'v Value, parts: &[&str], nulls: bool, out: &mut Vec<&'v Value>) {
        if parts.is_empty() {
            match v {
                Value::Array(a) => a.iter().for_each(|x| walk(x, parts, nulls, out)),
                Value::Null if !nulls => {}
                other => out.push(other),
            }
            return;
        }
        match v {
            Value::Object(m) => {
                for (k, c) in m {
                    if let Some(i) = (1..=parts.len()).find(|i| parts[..*i].join(".") == *k) {
                        walk(c, &parts[i..], nulls, out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, parts, nulls, out)),
            _ => {}
        }
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut out = Vec::new();
    walk(src, &parts, nulls, &mut out);
    out
}

/// Context of one `fields` API fetch.
struct FieldsCtx<'a> {
    lookup: &'a FieldLookup,
    pats: &'a [(String, Option<String>)],
    unmapped: &'a [String],
    meta: &'a HashMap<&'a str, Vec<Value>>,
}

fn under(k: &str, root: &str) -> bool {
    root.is_empty() || k.starts_with(&format!("{root}."))
}

/// Outermost `nested` object paths below `root`.
fn nested_children(lookup: &FieldLookup, root: &str) -> Vec<String> {
    let mut v: Vec<String> = lookup.objects.iter().filter(|(k, o)| o.nested && under(k, root)).map(|(k, _)| k.clone()).collect();
    let all = v.clone();
    v.retain(|k| !all.iter().any(|q| q != k && k.starts_with(&format!("{q}."))));
    v.sort();
    v
}

/// Fetch the fields of one (nested) object level into `out`, names relative
/// to `root`; nested children become arrays of per-object maps.
fn fetch_level(c: &FieldsCtx, src: &Value, root: &str, out: &mut Map<String, Value>) -> Result<()> {
    let strip = |n: &str| if root.is_empty() { n.to_string() } else { n[root.len() + 1..].to_string() };
    let mut names: Vec<String> = c.lookup.fields.keys().cloned().collect();
    if root.is_empty() {
        for (p, _) in c.pats {
            if !p.contains('*') && !c.lookup.fields.contains_key(p) && c.lookup.get(p).is_some_and(|f| f.flattened_key().is_some()) {
                names.push(p.clone());
            }
        }
    }
    names.sort();
    for n in &names {
        let Some(f) = c.lookup.fields.get(n).or_else(|| c.lookup.get(n)) else { continue };
        if n.starts_with('_') || f.nested_path.as_deref().unwrap_or("") != root || !under(n, root) {
            continue;
        }
        let Some((pat, fmt)) = c.pats.iter().find(|(p, _)| !p.contains('*') && p == n).or_else(|| c.pats.iter().find(|(p, _)| crate::mapping::wildcard_match(p, n))) else { continue };
        if fmt.as_deref() == Some("chunks") && f.declared.is_semantic() {
            if let Some(v) = c.meta.get(n.as_str()) {
                out.insert(strip(n), Value::Array(v.clone()));
            }
            continue;
        }
        if fmt.is_some() {
            let declared = match (f.ty, f.alias_path.as_deref().and_then(|t| c.lookup.get(t))) {
                (EsType::Alias, Some(target)) => target.declared,
                _ => f.declared,
            };
            let supports = matches!(declared, EsType::Date | EsType::DateNanos | EsType::GeoPoint | EsType::GeoShape | EsType::Shape | EsType::Point | EsType::DateRange);
            if !supports {
                let prefix = if pat.contains('*') { format!("error fetching [{n}] which matched [{pat}]") } else { format!("error fetching [{n}]") };
                let ty = if f.flattened_key().is_some() { EsType::Flattened } else { f.declared };
                return Err(EsError::illegal_argument(format!("{prefix}: Field [{n}] of type [{}] doesn't support formats.", ty.name())));
            }
        }
        let vals = if f.script.is_some() {
            c.meta.get(n.as_str()).cloned().unwrap_or_default()
        } else if f.ty == EsType::Alias {
            match f.alias_path.as_deref() {
                Some(t) if c.meta.contains_key(t) => c.meta[t].clone(),
                Some(t) => fields_api_values(c.lookup, t, &strip(t), src, fmt.as_deref()),
                None => vec![],
            }
        } else {
            // A pass-through alias (`host` for `attributes.host`) reads the leaf's own path.
            let base = if c.lookup.passthrough_alias.contains(n) { f.name.as_str() } else { n.as_str() };
            let path = if f.is_multi_field { base.rsplit_once('.').map(|x| x.0).unwrap_or(base) } else { base };
            fields_api_values(c.lookup, n, &strip(path), src, fmt.as_deref())
        };
        if !vals.is_empty() {
            out.insert(strip(n), Value::Array(vals));
        }
    }
    // `_index_mode` when asked for by name.
    if root.is_empty() && c.pats.iter().any(|(p, _)| p == crate::mapping::INDEX_MODE_FIELD) {
        if let Some(v) = c.lookup.index_mode_field.as_ref().and_then(|f| f.constant_value.clone()) {
            out.insert(crate::mapping::INDEX_MODE_FIELD.to_string(), Value::Array(vec![v]));
        }
    }
    for p in nested_children(c.lookup, root) {
        let rel = strip(&p);
        let mut arr = Vec::new();
        for e in source_values_ordered(src, &rel) {
            if e.is_object() {
                let mut m = Map::new();
                fetch_level(c, e, &p, &mut m)?;
                if !m.is_empty() {
                    arr.push(Value::Object(m));
                }
            }
        }
        if !arr.is_empty() {
            out.insert(rel, Value::Array(arr));
        }
    }
    if !c.unmapped.is_empty() {
        unmapped_level(c, src, root, "", out);
    }
    Ok(())
}

/// `include_unmapped`: source leaves with no mapping, as dotted names.
fn unmapped_level(c: &FieldsCtx, v: &Value, root: &str, prefix: &str, out: &mut Map<String, Value>) {
    fn leaves<'v>(v: &'v Value, scalars: &mut Vec<Value>, objs: &mut Vec<&'v Value>) {
        match v {
            Value::Array(a) => {
                for x in a.iter().filter(|x| x.is_array()) {
                    leaves(x, scalars, objs);
                }
                for x in a.iter().filter(|x| !x.is_array()) {
                    leaves(x, scalars, objs);
                }
            }
            Value::Object(_) => objs.push(v),
            Value::Null => {}
            other => scalars.push(other.clone()),
        }
    }
    let Value::Object(m) = v else { return };
    for (k, x) in m {
        let rel = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        let full = if root.is_empty() { rel.clone() } else { format!("{root}.{rel}") };
        let mapped_leaf = c.lookup.fields.contains_key(&full) || c.lookup.get(&full).is_some();
        if mapped_leaf {
            continue;
        }
        if c.lookup.objects.get(&full).is_some_and(|o| o.nested) {
            continue;
        }
        let mut scalars = Vec::new();
        let mut objs = Vec::new();
        leaves(x, &mut scalars, &mut objs);
        if !scalars.is_empty() && c.unmapped.iter().any(|p| crate::mapping::wildcard_match(p, &full)) {
            let e = out.entry(rel.clone()).or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(a) = e {
                a.extend(scalars);
            }
        }
        for o in objs {
            unmapped_level(c, o, root, &rel, out);
        }
    }
}

/// The fields API value of one mapped field from `_source`.
pub(crate) fn fields_api_values(lookup: &FieldLookup, name: &str, path: &str, src: &Value, format: Option<&str>) -> Vec<Value> {
    let Some(f) = lookup.get(name) else { return vec![] };
    // constant_keyword: every document holds the mapped value.
    if f.declared == EsType::ConstantKeyword {
        return f.constant_value.iter().map(|v| Value::String(crate::query::dsl::scalar_string(v))).collect();
    }
    // Geometries come back as GeoJSON (or WKT), parsed and normalized as ES fetches them.
    if crate::fieldtypes::is_shape(f.declared) || (f.declared == EsType::GeoPoint && f.null_value.is_none() && !lookup.copy_sources.contains_key(name)) {
        return crate::fieldtypes::geo::fetch_values(f.declared, src, path, format);
    }
    // rank_vectors: the list of vectors, each one an array.
    if f.declared == EsType::RankVectors {
        fn at<'v>(v: &'v Value, parts: &[&str]) -> Option<&'v Value> {
            if parts.is_empty() {
                return Some(v);
            }
            let m = v.as_object()?;
            (1..=parts.len()).rev().find_map(|i| m.get(&parts[..i].join(".")).and_then(|c| at(c, &parts[i..])))
        }
        let parts: Vec<&str> = path.split('.').collect();
        return match at(src, &parts) {
            Some(Value::Array(a)) => a.iter().filter(|x| x.is_array() || x.is_string()).cloned().collect(),
            _ => vec![],
        };
    }
    let mut raw = source_values_ordered_with(src, path, f.declared == EsType::UnsignedLong && f.null_value.is_some());
    // dense_vector: the whole vector (a hex string is its byte values).
    if f.declared == EsType::DenseVector {
        return match raw.as_slice() {
            [Value::String(h)] if matches!(crate::vectors::element_type(f), crate::vectors::ElementType::Byte | crate::vectors::ElementType::Bit) && h.len() % 2 == 0 && h.bytes().all(|b| b.is_ascii_hexdigit()) => {
                (0..h.len()).step_by(2).filter_map(|i| u8::from_str_radix(&h[i..i + 2], 16).ok()).map(|b| Value::from(b as i8 as i64)).collect()
            }
            _ => raw.into_iter().filter(|v| !v.is_null()).cloned().collect(),
        };
    }
    // Values copied into this field by `copy_to` are part of what it holds.
    if let Some(srcs) = lookup.copy_sources.get(name) {
        for s in srcs {
            raw.extend(source_values_ordered(src, s));
        }
    }
    let mut out = Vec::new();
    for v in raw {
        if matches!(f.declared, EsType::IntegerRange | EsType::LongRange | EsType::FloatRange | EsType::DoubleRange | EsType::IpRange) {
            out.push(v.clone());
            continue;
        }
        if f.declared == EsType::TokenCount {
            if let Value::String(text) = &v {
                let toks = match f.ext.as_ref().and_then(|x| x.analyzer.clone()) {
                    Some(a) => a.analyze(text),
                    None => rustic_core::analysis::Analyzer::standard().analyze(text),
                };
                let n = if f.ext.as_ref().and_then(|e| e.params.get("enable_position_increments")).and_then(Value::as_bool).unwrap_or(true) {
                    toks.last().map(|t| t.position as i64 + 1).unwrap_or(0)
                } else {
                    toks.len() as i64
                };
                out.push(Value::from(n));
                continue;
            }
        }
        let v = match (v, &f.null_value) {
            (Value::Null, Some(n)) => n.clone(),
            // unsigned_long indexes an empty string as its null_value.
            (Value::String(e), Some(n)) if e.is_empty() && f.declared == EsType::UnsignedLong => n.clone(),
            (v, _) => v.clone(),
        };
        // aggregate_metric_double comes back as its metric object.
        if f.declared == EsType::AggregateMetricDouble {
            if let Value::Object(m) = &v {
                if m.values().all(Value::is_number) {
                    out.push(Value::Object(m.clone()));
                }
            }
            continue;
        }
        // A `sparse_vector` comes back as its token -> weight object (a
        // stored one, as by default in an index that excludes source
        // vectors, with weights as floats).
        if f.declared == EsType::SparseVector {
            if let Value::Object(m) = &v {
                if f.stored {
                    out.push(Value::Object(m.iter().map(|(k, w)| (k.clone(), w.as_f64().map(|x| crate::values::java_float(x as f32)).unwrap_or(Value::Null))).collect()));
                } else {
                    out.push(Value::Object(m.clone()));
                }
                continue;
            }
        }
        let val = match f.ty {
            EsType::Date | EsType::DateNanos => {
                let ms = match &v {
                    Value::String(s) => f.date_format.parse(s).ok(),
                    Value::Number(n) => f.date_format.parse_number(n.as_f64().unwrap_or(0.0)).ok(),
                    _ => None,
                };
                let fmt = match format {
                    Some(spec) => crate::date::DateFormatter::new(spec).unwrap_or_else(|_| f.date_format.clone()),
                    None => f.date_format.clone(),
                };
                ms.map(|m| Value::String(fmt.format(m, crate::date::Tz::UTC)))
            }
            EsType::UnsignedLong => match &v {
                Value::Number(n) => Some(n.as_u64().map(Value::from).unwrap_or_else(|| Value::from(n.as_f64().unwrap_or(0.0).trunc() as u64))),
                Value::String(s) => s.trim().parse::<u64>().ok().map(Value::from),
                _ => None,
            },
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => match &v {
                Value::Number(n) => Some(n.as_i64().map(Value::from).unwrap_or_else(|| Value::from(n.as_f64().unwrap_or(0.0).trunc() as i64))),
                Value::String(s) => s.trim().parse::<f64>().ok().map(|x| Value::from(x.trunc() as i64)),
                _ => None,
            },
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
                let x = v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()));
                x.map(|x| match f.ty {
                    EsType::Float => values::java_double(values::to_float(x)),
                    EsType::HalfFloat => values::java_double(values::to_half_float(x)),
                    EsType::ScaledFloat => values::java_double((x * f.scaling_factor).round() / f.scaling_factor),
                    _ => values::java_double(x),
                })
            }
            EsType::Boolean => match &v {
                Value::Bool(b) => Some(Value::Bool(*b)),
                Value::String(s) => Some(Value::Bool(s == "true")),
                _ => None,
            },
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard => {
                let s = crate::query::dsl::scalar_string(&v);
                if f.ignore_above.map(|ia| s.chars().count() > ia).unwrap_or(false) {
                    None
                } else {
                    Some(Value::String(match &f.normalizer {
                        Some(n) => n.analyze(&s).into_iter().map(|t| t.text).collect(),
                        None => s,
                    }))
                }
            }
            EsType::GeoPoint => values::parse_geo_point(&v).map(|(lat, lon)| match format {
                Some("wkt") => Value::String(format!("POINT ({} {})", values::java_double(lon), values::java_double(lat))),
                _ => json!({"type": "Point", "coordinates": [lon, lat]}),
            }),
            EsType::Flattened if f.flattened_key().is_none() => match f.ignore_above {
                Some(ia) => filter_flattened(&v, ia),
                None => Some(v.clone()),
            },
            EsType::Ip => v.as_str().map(|s| Value::String(s.to_string())),
            // ES's dense_vector fetcher: the source list's elements (a
            // base64 / hex string as is).
            EsType::DenseVector => {
                if matches!(v, Value::Number(_) | Value::String(_)) {
                    out.push(v.clone());
                }
                continue;
            }
            _ => Some(match v {
                Value::String(_) => v.clone(),
                other => Value::String(other.to_string()),
            }),
        };
        if let Some(x) = val {
            out.push(x);
        }
    }
    out
}

/// A `flattened` value without the leaves longer than `ignore_above`
/// (`None` when nothing is left); a one-element array collapses to its value.
fn filter_flattened(v: &Value, ia: usize) -> Option<Value> {
    match v {
        Value::Object(m) => {
            let out: Map<String, Value> = m.iter().filter_map(|(k, x)| filter_flattened(x, ia).map(|y| (k.clone(), y))).collect();
            if out.is_empty() { None } else { Some(Value::Object(out)) }
        }
        Value::Array(a) => {
            let mut kept: Vec<Value> = a.iter().filter_map(|x| filter_flattened(x, ia)).collect();
            match kept.len() {
                0 => None,
                1 => kept.pop(),
                _ => Some(Value::Array(kept)),
            }
        }
        Value::Null => None,
        other => {
            let s = match other {
                Value::String(s) => s.chars().count(),
                o => o.to_string().chars().count(),
            };
            if s > ia { None } else { Some(other.clone()) }
        }
    }
}

/// Minimal `java.text.DecimalFormat`: `#,##0.00`-style integer/fraction digits.
fn decimal_format(pat: &str, x: f64) -> String {
    let (int_p, frac_p) = pat.split_once('.').unwrap_or((pat, ""));
    let min_frac = frac_p.chars().filter(|c| *c == '0').count();
    let max_frac = frac_p.chars().filter(|c| *c == '0' || *c == '#').count();
    let min_int = int_p.chars().filter(|c| *c == '0').count();
    let mut s = format!("{:.*}", max_frac, x.abs());
    if let Some((i, f)) = s.clone().split_once('.') {
        let mut f = f.to_string();
        while f.len() > min_frac && f.ends_with('0') {
            f.pop();
        }
        s = if f.is_empty() { i.to_string() } else { format!("{i}.{f}") };
    }
    let (i, f) = s.split_once('.').map(|(a, b)| (a.to_string(), Some(b.to_string()))).unwrap_or((s.clone(), None));
    let mut i = i.trim_start_matches('0').to_string();
    while i.len() < min_int {
        i.insert(0, '0');
    }
    if int_p.contains(',') {
        let b = i.as_bytes();
        let mut g = String::new();
        for (n, c) in b.iter().enumerate() {
            if n > 0 && (b.len() - n) % 3 == 0 {
                g.push(',');
            }
            g.push(*c as char);
        }
        i = g;
    }
    let mut out = if x < 0.0 { "-".to_string() } else { String::new() };
    if i.is_empty() && f.is_none() {
        i.push('0');
    }
    out.push_str(&i);
    if let Some(f) = f {
        out.push('.');
        out.push_str(&f);
    }
    out
}

pub(crate) fn docvalue_values(leaf: &LeafReader<'_>, lookup: &FieldLookup, name: &str, doc: DocId, format: Option<&str>) -> Result<Vec<Value>> {
    if let Some(field)=lookup.get(name).filter(|field|field.ty==EsType::Text&&field.fielddata) {
        if let Some(id)=field.core_id {
            let data=leaf.uninverted(id)?;
            return Ok(data.ords(doc).iter().map(|&ord|Value::String(data.term_string(ord))).collect());
        }
    }
    Ok(docvalue_values_plain(leaf,lookup,name,doc,format))
}
fn docvalue_values_plain(leaf: &LeafReader<'_>, lookup: &FieldLookup, name: &str, doc: DocId, format: Option<&str>) -> Vec<Value> {
    if name == "_id" {
        return leaf.keyword_value(ID_FIELD, doc).map(Value::String).into_iter().collect();
    }
    if name == crate::tsdb::TSID_FIELD {
        if let Some(id) = lookup.get(name).and_then(|f| f.core_id) {
            return leaf.keyword_values(id, doc).into_iter().map(|s| Value::String(crate::tsdb::tsid_display(&s))).collect();
        }
    }
    // Time series ids start with the 4-byte routing hash.
    if name == "_ts_routing_hash" && lookup.get(crate::tsdb::TSID_FIELD).is_some() {
        use base64::Engine as _;
        // A synthetic-id index keeps it in its own column; otherwise
        // standard ids start with it (LE), synthetic ids end with it (BE).
        let column = lookup.get(crate::tsdb::ROUTING_HASH_FIELD).and_then(|f| f.core_id).and_then(|id| leaf.i64_value(id, doc));
        let hash = column.map(|h| h as i32).or_else(|| crate::tsdb::routing_hash_of_id(&leaf.keyword_value(ID_FIELD, doc).unwrap_or_default()));
        return hash
            .map(|h| Value::String(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h.to_le_bytes())))
            .into_iter()
            .collect();
    }
    match name {
        "_seq_no" => return leaf.i64_values(SEQ_NO_FIELD, doc).into_iter().map(Value::from).collect(),
        "_version" => return leaf.i64_values(VERSION_FIELD, doc).into_iter().map(Value::from).collect(),
        _ => {}
    }
    let Some(f) = lookup.get(name) else { return vec![] };
    if let Some(v) = crate::fieldtypes::docvalues(leaf, f, doc, format) {
        return v;
    }
    let Some(id) = f.core_id else {
        let render = |(a, b): (f64, f64)| format!("{}, {}", crate::script::value::java_double_string(a), crate::script::value::java_double_string(b));
        // ES renders a geo_point doc value as `GeoPoint.toString()`: "lat,
        // lon" of the quantized point. `FieldData.toString` feeds those
        // strings through `SortingBinaryDocValues`, so the response lists
        // them in byte order (duplicates kept), not doc values order.
        let mut out: Vec<String> = if let Some(values) = f.geo_id.and_then(|g| match leaf {
            LeafReader::Segment(s) => s.geo_index(g).map(|i| i.values()),
            LeafReader::Memtable(w, ..) => w.geo_values(g),
        }) {
            crate::script::doc::sorted_geo_points(values, doc).into_iter().map(render).collect()
        } else if let (Some(la), Some(lo)) = (f.lat_id, f.lon_id) {
            leaf.f64_values(la, doc).into_iter().zip(leaf.f64_values(lo, doc)).map(|(a, b)| render(rustic_core::geo::quantize(a, b))).collect()
        } else {
            vec![]
        };
        out.sort_unstable();
        return out.into_iter().map(Value::String).collect();
    };
    // Sorted numeric doc values expose value order, independently of source
    // array order, and retain duplicates. Sort before display formatting.
    let sorted_i64 = || { let mut v = leaf.i64_values(id, doc); v.sort_unstable(); v };
    let sorted_f64 = || { let mut v = leaf.f64_values(id, doc); v.sort_by(f64::total_cmp); v };
    let sorted_u64 = || { let mut v = leaf.u64_values(id, doc); v.sort_unstable(); v };

    if let Some(pat) = format.filter(|p| p.contains('#') || p.contains('0')) {
        if f.ty.is_numeric() {
            let nums: Vec<f64> = if f.ty.is_integer() { sorted_i64().into_iter().map(|x| x as f64).collect() } else if f.ty == EsType::ScaledFloat {
                let scale = (f.scaling_factor.is_finite() && f.scaling_factor > 0.0).then(|| (f.scaling_factor, 1.0 / f.scaling_factor));
                sorted_f64().into_iter().map(|x| crate::aggs::es_scaled(x, scale)).collect()
            } else { sorted_f64() };
            return nums.into_iter().map(|x| Value::String(decimal_format(pat, x))).collect();
        }
    }
    match f.ty {
        EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened => leaf.keyword_values(id, doc).into_iter().map(Value::String).collect(),
        EsType::Ip => leaf.keyword_values(id, doc).iter().map(|s| Value::String(values::ip_display(s))).collect(),
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => sorted_i64().into_iter().map(Value::from).collect(),
        EsType::UnsignedLong => sorted_u64().into_iter().map(Value::from).collect(),
        EsType::Double | EsType::Float | EsType::HalfFloat => sorted_f64().into_iter().map(values::java_double).collect(),
        // ES reads a scaled_float's doc values as the scaled long times the
        // inverse factor (`ScaledFloatLeafFieldData`), not the stored quotient.
        EsType::ScaledFloat => {
            let scale = (f.scaling_factor.is_finite() && f.scaling_factor > 0.0).then(|| (f.scaling_factor, 1.0 / f.scaling_factor));
            sorted_f64().into_iter().map(|x| values::java_double(crate::aggs::es_scaled(x, scale))).collect()
        }
        EsType::Date | EsType::DateNanos => {
            let fmt = match format {
                Some(spec) if spec != "use_field_mapping" => crate::date::DateFormatter::new(spec).unwrap_or_else(|_| f.date_format.clone()),
                _ => f.date_format.clone(),
            };
            sorted_i64().into_iter().map(|m| Value::String(fmt.format(m, crate::date::Tz::UTC))).collect()
        }
        EsType::Boolean => leaf.bool_value(id, doc).map(Value::Bool).into_iter().collect(),
        EsType::DenseVector => {
            let Some(v) = leaf.vector(id, doc) else { return vec![] };
            let et = crate::vectors::element_type(f);
            if format == Some("binary") {
                // `DocValueFormat.BINARY`: base64 of the vector's bytes.
                return vec![Value::String(crate::vectors::binary_doc_value(&v, et, f.dims.unwrap_or(0)))];
            }
            let arr: Vec<Value> = match crate::vectors::Packing::of(f) {
                // Packed `byte` / `bit` rows: their signed bytes.
                Some(p) => p.bytes(&v).iter().map(|&b| Value::from(b as i8 as i64)).collect(),
                None => v.into_iter().map(values::java_float).collect(),
            };
            vec![Value::Array(arr)]
        }
        _ => vec![],
    }
}

/// Per-index state of a paginated search (scroll): last hit returned.
pub type Cursor = Option<TopHit>;

pub struct SearchOutput {
    /// Complete response JSON.
    pub json: String,
    /// Per target, the last hit returned from it (scroll continuation).
    pub cursors: Vec<Cursor>,
    pub hits: usize,
    pub total: u64,
    /// The returned hits as `(target, leaf, doc, score)` (retrievers).
    pub page: Vec<(usize, usize, DocId, f32)>,
    /// Time spent rendering the page (the fetch phase), in nanoseconds.
    pub fetch_nanos: u64,
    /// Where `hits.hits` lies in `json`: the end of its last hit and each
    /// hit's length (hits are separated by one comma); `None` once `json`
    /// was rewritten.
    pub hit_spans: Option<(usize, Vec<u32>)>,
    /// A [`SearchRequest::columns_only`] search's hits, in page order (its
    /// `json` then lists none).
    pub columns: Vec<ColumnHit>,
}

/// A [`SearchRequest::columns_only`] hit: its index (as a hit names it),
/// `_id`, and each `docvalue_fields` column's sorted doc values (dates as
/// epoch millis).
#[derive(Debug, Clone)]
pub struct ColumnHit {
    pub index: String,
    pub id: String,
    pub values: Vec<Vec<Value>>,
}

struct IndexRun<'a> {
    /// Position of the searched shard copy in the request (`_shard_doc`).
    shard_ord: i64,
    leaves: Vec<LeafReader<'a>>,
    lookup: Arc<FieldLookup>,
    query: Query,
    /// Alias/DLS visibility, independent of the user's query and post-filter.
    visibility_query: Option<Query>,
    visibility_context: Option<SearchContext>,
    top: TopDocs,
    sort: Vec<ResolvedSort>,
    terms: Option<QueryTerms>,
    /// `inner_hits` of the request's `nested` queries.
    inner: Vec<InnerHitsDef>,
    /// `_name`d queries of the request (`matched_queries`), in order.
    named: Vec<(String, Query, SearchContext)>,
    /// Fetch-phase script / runtime field executors (bound to `leaves`).
    scripts: crate::script::search::FetchScripts,
}

impl IndexRun<'_> {
    fn visibility_result(&self) -> Result<()> {
        if let Some(ctx) = &self.visibility_context { ctx.execution_result()?; }
        Ok(())
    }
    fn visible_docs(&self, leaf: &LeafReader<'_>) -> Result<Vec<DocId>> {
        let live = leaf.live_docs();
        let bits = self.visibility_query.as_ref().zip(self.visibility_context.as_ref())
            .map(|(q,ctx)| q.filter_bits(leaf, ctx)).transpose()?;
        Ok((0..leaf.num_docs()).filter(|&d| live(d) && bits.as_ref().is_none_or(|b|
            b.get(d as usize / 64).is_some_and(|w| w >> (d % 64) & 1 != 0))).collect())
    }
}

fn aggregation_visibility(t: &Target, ctx: &QueryContext, req: &SearchRequest) -> Result<Option<Query>> {
    if req.aggs.is_none() { return Ok(None) }
    t.effective_filter().map(|filter| {
        // A role/alias filter may reference a field hidden by FLS.
        let full_lookup = t.svc.lookup();
        parse_query(&filter, &QueryContext { lookup: &full_lookup, ..ctx.clone() })
            .map_err(|e| wrap_shard_err(&t.svc.name, e))
    }).transpose()
}

fn is_query_type(k: &str) -> bool {
    matches!(
        k,
        "match_all" | "match_none" | "match" | "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" | "multi_match" | "combined_fields" | "query_string"
            | "simple_query_string" | "term" | "terms" | "terms_set" | "range" | "exists" | "prefix" | "wildcard" | "regexp" | "fuzzy" | "ids" | "bool" | "boosting"
            | "constant_score" | "dis_max" | "function_score" | "knn" | "geo_bounding_box" | "geo_distance" | "geo_shape" | "geo_polygon" | "wrapper" | "nested"
            | "has_child" | "has_parent" | "parent_id" | "script" | "script_score" | "intervals" | "distance_feature" | "more_like_this" | "mlt" | "span_term"
            | "span_near" | "span_or" | "span_not" | "span_first" | "span_multi" | "span_containing" | "span_within" | "span_field_masking" | "percolate" | "rank_feature"
            | "sparse_vector" | "text_expansion" | "semantic" | "rule" | "pinned" | "shape" | "type"
    )
}

/// The `_name`d (sub)queries of a request body: top-level `knn` entries
/// first, then the `query` tree in document order. Each is the enclosing
/// query object (`{"term": {...}}`) with the `_name` left in place (the
/// parsers ignore it).
fn named_queries(query: Option<&Value>, knn: &[Value]) -> Vec<(String, Value, bool)> {
    fn walk(v: &Value, enclosing: Option<&Value>, out: &mut Vec<(String, Value, bool)>) {
        match v {
            Value::Object(m) => {
                if let Some(Value::String(n)) = m.get("_name") {
                    if let Some(e) = enclosing {
                        if !out.iter().any(|(x, _, _)| x == n) {
                            out.push((n.clone(), e.clone(), false));
                        }
                    }
                }
                // A single-key object whose value is an object is a query
                // node (`{"term": {...}}`); its children see it as enclosing.
                let is_query_node = m.len() == 1 && m.values().next().is_some_and(Value::is_object) && m.keys().next().is_some_and(|k| is_query_type(k));
                for (k, x) in m {
                    if k == "_name" {
                        continue;
                    }
                    if k == "functions" && enclosing.is_some_and(|e| e.get("function_score").is_some_and(|f| std::ptr::eq(f, v))) {
                        if let Value::Array(fs) = x {
                            fs.iter().for_each(|f| function_entry(f, out));
                            continue;
                        }
                    }
                    walk(x, if is_query_node { Some(v) } else { enclosing }, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, enclosing, out)),
            _ => {}
        }
    }
    /// A `function_score` function entry: a name on its filter (or the
    /// entry's `_name`) scores as the function alone over that filter
    /// (`FilterFunctionBuilder.toNamedScoreQuery`: `FunctionScoreQuery`
    /// with `boost_mode: replace`); names deeper in the filter are plain.
    fn function_entry(f: &Value, out: &mut Vec<(String, Value, bool)>) {
        let Some(fo) = f.as_object() else { return walk(f, None, out) };
        let mut entry = fo.clone();
        let filter = entry.remove("filter");
        let entry_name = entry.remove("_name").and_then(|n| n.as_str().map(str::to_string));
        let filter_q = filter.clone().unwrap_or_else(|| json!({"match_all": {}}));
        let scored = |name: String| (name, json!({"function_score": {"query": filter_q, "functions": [Value::Object(entry.clone())], "boost_mode": "replace"}}), false);
        let push = |item: (String, Value, bool), out: &mut Vec<(String, Value, bool)>| {
            if !out.iter().any(|(x, _, _)| *x == item.0) {
                out.push(item);
            }
        };
        let mut inner = Vec::new();
        if let Some(fv) = &filter {
            walk(fv, None, &mut inner);
        }
        for (n, q, k) in inner {
            if filter.as_ref() == Some(&q) {
                // The entry's `_name` replaces the filter's own name.
                if entry_name.is_none() {
                    push(scored(n), out);
                }
            } else {
                push((n, q, k), out);
            }
        }
        if let Some(n) = entry_name {
            push(scored(n), out);
        }
    }
    let mut out = Vec::new();
    for k in knn {
        if let Some(Value::String(n)) = k.get("_name") {
            out.push((n.clone(), k.clone(), true));
        }
    }
    if let Some(q) = query {
        walk(q, None, &mut out);
    }
    out
}

/// One `inner_hits` request: its (top-level) nested query and paging.
struct InnerHitsDef {
    name: String,
    query: Query,
    ctx: SearchContext,
    from: usize,
    size: usize,
    source: bool,
    version: bool,
    seq_no_primary_term: bool,
    docvalue_fields: Vec<(String, Option<String>)>,
    /// `inner_hits.fields` (fields API over the nested object's source).
    fields: Vec<(String, Option<String>)>,
    /// `inner_hits.highlight`.
    highlight: Option<Map<String, Value>>,
    /// `inner_hits` of nested queries inside this one (rendered per inner hit).
    children: Vec<InnerHitsDef>,
    /// The inner scorer of the last rendered parent (reused for later parents of the same leaf).
    cursor: std::sync::Mutex<Option<rustic_core::query::InnerHitsCursor>>,
    /// `inner_hits.sort`: `(field, descending)` keys (`_score` / `_doc` too).
    sort: Vec<(String, bool)>,
    /// Parent/child inner hits (`has_child` / `has_parent`) instead of nested ones.
    join: Option<crate::fieldtypes::join::InnerSide>,
}

/// `inner_hits.sort` entries (`"f"`, `{"f": "desc"}`, `{"f": {"order": ..}}`).
fn inner_sort(v: Option<&Value>) -> Vec<(String, bool)> {
    let items = match v {
        Some(Value::Array(a)) => a.clone(),
        Some(x) => vec![x.clone()],
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for it in items {
        match it {
            Value::String(f) => out.push((f.clone(), f == "_score")),
            Value::Object(m) => {
                for (f, o) in m {
                    let ord = o.as_str().or_else(|| o.get("order").and_then(Value::as_str));
                    let desc = match ord {
                        Some(x) => x.eq_ignore_ascii_case("desc"),
                        None => f == "_score",
                    };
                    out.push((f.clone(), desc));
                }
            }
            _ => {}
        }
    }
    out
}

/// Collect the `nested`, `has_child` and `has_parent` queries of the
/// request that ask for `inner_hits`; `nested` ones inside a `nested` one
/// become its `children`.
/// Mark every `knn` query under `v` as building inner hits (exact scoring
/// of all of each matched parent's children, see `vectors::nested_knn`).
fn mark_knn_inner_hits(v: &mut Value) {
    match v {
        Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                if k == "knn" {
                    if let Some(o) = x.as_object_mut() {
                        if o.contains_key("field") {
                            o.insert("_inner_hits_children".into(), Value::Bool(true));
                            continue;
                        }
                    }
                }
                mark_knn_inner_hits(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(mark_knn_inner_hits),
        _ => {}
    }
}

fn inner_hit_defs(query: Option<&Value>, qctx: &QueryContext, leaves: &[LeafReader<'_>]) -> Result<Vec<InnerHitsDef>> {
    fn walk(v: &Value, out: &mut Vec<(&'static str, Map<String, Value>)>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    let kind = match k.as_str() {
                        "nested" => "nested",
                        "has_child" => "has_child",
                        "has_parent" => "has_parent",
                        _ => "",
                    };
                    if !kind.is_empty() {
                        if let Some(o) = x.as_object() {
                            if o.contains_key("inner_hits") {
                                out.push((kind, o.clone()));
                            } else if let Some(q) = o.get("query") {
                                walk(q, out);
                            }
                            continue;
                        }
                    }
                    walk(x, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let Some(q) = query else { return Ok(Vec::new()) };
    let mut found = Vec::new();
    walk(q, &mut found);
    let mut out = Vec::new();
    for (kind, mut o) in found {
        let ih = o.remove("inner_hits").unwrap_or(Value::Null);
        let (path, children, query, join) = if kind == "nested" {
            let path = o.get("path").and_then(Value::as_str).unwrap_or("").to_string();
            let children = inner_hit_defs(o.get("query"), qctx, leaves)?;
            // ES rewrites a knn query for inner hits into an exact kNN over
            // the matched parent's children (`KnnVectorQueryBuilder.doRewrite`
            // with an `InnerHitsRewriteContext`), so inner hits are every child
            // passing the child filter and `similarity`, scored exactly, cut to
            // `inner_hits.size`; parents match as before.
            if let Some(q) = o.get_mut("query") {
                mark_knn_inner_hits(q);
            }
            let query = strip_boost(parse_query(&json!({ "nested": o }), qctx)?);
            let chunks = matches!(&query, Query::Custom(c) if c.as_any().is_some_and(|a| a.is::<crate::inference::chunk_query::ChunkQuery>()));
            if !matches!(query, Query::Nested { .. }) && !chunks {
                continue;
            }
            (path, children, query, None)
        } else {
            // ES `ParentChildInnerHitContextBuilder`: named after the relation.
            let Some(side) = crate::fieldtypes::join::inner_hits_side(kind, &o, qctx)? else { continue };
            (side.ty.clone(), Vec::new(), side.query.clone(), Some(side))
        };
        let ctx = SearchContext::for_query(&query, leaves, Bm25Params::default());
        ctx.execution_result()?;
        let num = |k: &str, d: usize| ih.get(k).and_then(Value::as_u64).map(|x| x as usize).unwrap_or(d);
        let flag = |k: &str| ih.get(k).and_then(Value::as_bool).unwrap_or(false);
        out.push(InnerHitsDef {
            name: ih.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or(path),
            query,
            ctx,
            from: num("from", 0),
            size: num("size", 3),
            source: ih.get("_source").and_then(Value::as_bool).unwrap_or(true),
            version: flag("version"),
            seq_no_primary_term: flag("seq_no_primary_term"),
            docvalue_fields: ih.get("docvalue_fields").map(field_list).unwrap_or_default(),
            fields: ih.get("fields").map(field_list).unwrap_or_default(),
            highlight: ih.get("highlight").and_then(Value::as_object).cloned(),
            children,
            cursor: Default::default(),
            sort: inner_sort(ih.get("sort")),
            join,
        });
    }
    Ok(out)
}

fn strip_boost(q: Query) -> Query {
    match q {
        Query::Boost { query, .. } => strip_boost(*query),
        other => other,
    }
}

/// `_nested` metadata and the nested object's source for nested doc `child`.
fn nested_meta(leaf: &LeafReader<'_>, child: DocId, parent_src: Option<&Value>) -> (Value, Option<Value>) {
    let mut chain: Vec<(String, u32)> = Vec::new();
    let mut d = child;
    while !leaf.is_root(d) {
        chain.push((leaf.nested_path(d).to_string(), leaf.nested_offset(d)));
        match leaf.nested_parent(d) {
            Some(p) => d = p,
            None => break,
        }
    }
    chain.reverse();
    let mut meta = Value::Null;
    let mut prev = String::new();
    // Walk borrowed source nodes; cloning every ancestor copies all siblings
    // once per inner hit. Only the selected nested object needs ownership.
    let mut src = parent_src;
    let mut rels = Vec::new();
    for (path, off) in &chain {
        let rel = if prev.is_empty() { path.clone() } else { path.strip_prefix(&format!("{prev}.")).unwrap_or(path).to_string() };
        src = src.and_then(|v| {
            let mut cur = Some(v);
            if let Some(Value::Object(m)) = cur {
                if let Some(x) = m.get(&rel) {
                    cur = Some(x);
                } else {
                    for part in rel.split('.') {
                        cur = cur.and_then(|c| c.get(part));
                    }
                }
            }
            match cur {
                Some(Value::Array(a)) => a.get(*off as usize),
                Some(o @ Value::Object(_)) if *off == 0 => Some(o),
                _ => None,
            }
        });
        rels.push((rel, *off));
        prev = path.clone();
    }
    for (rel, off) in rels.into_iter().rev() {
        let mut m = Map::new();
        m.insert("field".into(), json!(rel));
        m.insert("offset".into(), json!(off));
        if !meta.is_null() {
            m.insert("_nested".into(), meta);
        }
        meta = Value::Object(m);
    }
    (meta, src.cloned())
}

/// The `inner_hits` object of one hit (`doc` is the root document, or a
/// nested doc for the inner hits of nested-in-nested queries).
fn render_inner_hits(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, doc: DocId, defs: &[InnerHitsDef], total_as_int: bool, root_doc: Option<Document>) -> Result<Value> {
    let root = leaf.nested_root(doc);
    // The hit's own stored doc when the page's batch read it already (not
    // when an inner hit shows `_version` / `_seq_no`: the batch may leave
    // column-stored fields out).
    let stored: Document = match root_doc {
        Some(d) if root == doc && !defs.iter().any(|d| d.version || d.seq_no_primary_term) => d,
        _ => leaf.get_doc(root)?.unwrap_or_default(),
    };
    let id = stored.get(ID_FIELD).and_then(|v| v.as_text()).unwrap_or("").to_string();
    let src: Option<Value> = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).and_then(|s| serde_json::from_str(&t.visible_source(&crate::vector_source::load(s, &run.lookup, leaf, root))).ok());
    let mut out = Map::new();
    for def in defs {
        if let Query::Custom(c) = &def.query {
            if let Some(cq) = c.as_any().and_then(|a| a.downcast_ref::<crate::inference::chunk_query::ChunkQuery>()) {
                out.insert(def.name.clone(), chunk_inner_hits(t, cq, def, leaf, root, &id, src.as_ref(), total_as_int));
                continue;
            }
        }
        if let Some(side) = &def.join {
            out.insert(def.name.clone(), join_inner_hits(t, run, leaf, doc, def, side, total_as_int)?);
            continue;
        }
        let mut hits = {
            let mut cur = def.cursor.lock().unwrap_or_else(|e| e.into_inner());
            def.query.nested_inner_hits_with(&mut cur, leaf, &def.ctx, doc)
        };
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        let by_field = def.sort.iter().any(|(f, _)| f != "_score");
        if !def.sort.is_empty() {
            let failure=std::cell::RefCell::new(None);
            let key = |d: DocId, f: &str| match docvalue_values(leaf, &run.lookup, f, d, None) {
                Ok(values)=>values.into_iter().next().unwrap_or(Value::Null),
                Err(error)=>{failure.borrow_mut().get_or_insert(error);Value::Null}
            };
            hits.sort_by(|a, b| {
                for (f, desc) in &def.sort {
                    let o = match f.as_str() {
                        "_score" => a.score.total_cmp(&b.score),
                        "_doc" => a.doc.cmp(&b.doc),
                        _ => {
                            let (x, y) = (key(a.doc, f), key(b.doc, f));
                            match (&x, &y) {
                                (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
                                // Missing values sort last either way.
                                (Value::Null, _) => return std::cmp::Ordering::Greater,
                                (_, Value::Null) => return std::cmp::Ordering::Less,
                                (Value::Number(p), Value::Number(q)) => p.as_f64().unwrap_or(0.0).total_cmp(&q.as_f64().unwrap_or(0.0)),
                                _ => x.as_str().unwrap_or_default().cmp(y.as_str().unwrap_or_default()),
                            }
                        }
                    };
                    let o = if *desc { o.reverse() } else { o };
                    if o != std::cmp::Ordering::Equal {
                        return o;
                    }
                }
                a.doc.cmp(&b.doc)
            });
            if let Some(error)=failure.into_inner(){return Err(error);}
        }
        let max = if by_field { None } else { hits.first().map(|h| h.score) };
        let mut page = Vec::new();
        for h in hits.iter().skip(def.from).take(def.size) {
            let (meta, child_src) = nested_meta(leaf, h.doc, src.as_ref());
            let mut m = Map::new();
            m.insert("_index".into(), json!(t.svc.name));
            m.insert("_id".into(), json!(id));
            m.insert("_nested".into(), meta);
            if def.version {
                m.insert("_version".into(), json!(stored.get(VERSION_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)));
            }
            if def.seq_no_primary_term {
                m.insert("_seq_no".into(), json!(stored.get(SEQ_NO_FIELD).and_then(|v| v.as_i64()).unwrap_or(0)));
                m.insert("_primary_term".into(), json!(stored.get(PRIMARY_TERM_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)));
            }
            // ES scores are Java floats.
            m.insert("_score".into(), if by_field { Value::Null } else { serde_json::from_str(&crate::retriever::score_json(h.score)).unwrap_or(Value::Null) });
            if by_field {
                let sv: Vec<Value> = def
                    .sort
                    .iter()
                    .map(|(f, _)| Ok(match f.as_str() {
                        "_score" => serde_json::from_str(&crate::retriever::score_json(h.score)).unwrap_or(Value::Null),
                        "_doc" => json!(h.doc),
                        _ => docvalue_values(leaf, &run.lookup, f, h.doc, None)?.into_iter().next().unwrap_or(Value::Null),
                    }))
                    .collect::<Result<_>>()?;
                m.insert("sort".into(), Value::Array(sv));
            }
            let mut fields = Map::new();
            if !def.fields.is_empty() {
                if let Some(cs) = &child_src {
                    // The fields API over the nested object: values are
                    // wrapped in the nested path (`{"nested": [{"p": [v]}]}`).
                    let path = leaf.nested_path(h.doc);
                    let wrapped = wrap_in_path(path, cs.clone());
                    let mut inner = Map::new();
                    for (pat, fmt) in &def.fields {
                        let names = if pat.contains('*') { run.lookup.matching(pat) } else { vec![pat.clone()] };
                        for n in names {
                            let Some(rel) = n.strip_prefix(path).and_then(|r| r.strip_prefix('.')) else { continue };
                            let mpath = match run.lookup.get(&n) {
                                Some(f) if f.is_multi_field => n.rsplit_once('.').map(|x| x.0).unwrap_or(&n).to_string(),
                                _ => n.clone(),
                            };
                            let v = fields_api_values(&run.lookup, &n, &mpath, &wrapped, fmt.as_deref());
                            if !v.is_empty() {
                                inner.insert(rel.to_string(), Value::Array(v));
                            }
                        }
                    }
                    if !inner.is_empty() {
                        fields.insert(path.to_string(), json!([Value::Object(inner)]));
                    }
                }
            }
            if def.source {
                if let Some(cs) = child_src {
                    m.insert("_source".into(), cs);
                }
            }
            for (pat, fmt) in &def.docvalue_fields {
                let names = if pat.contains('*') { run.lookup.matching(pat) } else { vec![pat.clone()] };
                for n in names {
                    // Metadata doc values live on the root document.
                    let target = if n.starts_with('_') { root } else { h.doc };
                    let v = docvalue_values(leaf, &run.lookup, &n, target, fmt.as_deref())?;
                    if !v.is_empty() {
                        fields.insert(n, Value::Array(v));
                    }
                }
            }
            if !fields.is_empty() {
                m.insert("fields".into(), Value::Object(fields));
            }
            if let Some(hl) = &def.highlight {
                let hm = highlight::inner_hit(hl, &def.query, &t.svc, &run.lookup, leaf.nested_path(h.doc), crate::stored::overlay(m.get("_source").cloned().or_else(|| nested_meta(leaf, h.doc, src.as_ref()).1), leaf.get_doc(h.doc)?.and_then(|d| crate::stored::element_source(&d, leaf.nested_path(h.doc)))), h.doc)
                    .map_err(|e| EsError::search_phase(&t.svc.name, e))?;
                if !hm.is_empty() {
                    m.insert("highlight".into(), Value::Object(hm));
                }
            }
            if !def.children.is_empty() {
                m.insert("inner_hits".into(), render_inner_hits(t, run, leaf, h.doc, &def.children, total_as_int, None)?);
            }
            page.push(Value::Object(m));
        }
        let total = if total_as_int { json!(hits.len()) } else { json!({ "value": hits.len(), "relation": "eq" }) };
        out.insert(
            def.name.clone(),
            json!({ "hits": {
                "total": total,
                "max_score": max.map(|x| serde_json::from_str(&crate::retriever::score_json(x)).unwrap_or(Value::Null)).unwrap_or(Value::Null),
                "hits": page,
            }}),
        );
    }
    Ok(Value::Object(out))
}

/// Parent/child `inner_hits` of hit `doc`: its children (`has_child`) or its
/// parent (`has_parent`) matching the inner query, each a hit of its own
/// (ES `ParentChildInnerHitContextBuilder.JoinFieldInnerHitSubContext`).
fn join_inner_hits(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, doc: DocId, def: &InnerHitsDef, side: &crate::fieldtypes::join::InnerSide, total_as_int: bool) -> Result<Value> {
    let mut hits = crate::fieldtypes::join::inner_hits_of(side, &run.leaves, leaf, doc, run.visibility_query.as_ref())?;
    hits.sort_by(|a, b| b.2.total_cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1))));
    let by_field = def.sort.iter().any(|(f, _)| f != "_score");
    let failure=std::cell::RefCell::new(None);
    let key = |li: usize, d: DocId, f: &str| -> Value {
        match f {
            "_doc" => json!(d),
            _ => match docvalue_values(&run.leaves[li], &run.lookup, f, d, None) {
                Ok(values)=>values.into_iter().next().unwrap_or(Value::Null),
                Err(error)=>{failure.borrow_mut().get_or_insert(error);Value::Null},
            },
        }
    };
    if !def.sort.is_empty() {
        hits.sort_by(|a, b| {
            for (f, desc) in &def.sort {
                let o = match f.as_str() {
                    "_score" => a.2.total_cmp(&b.2),
                    "_doc" => (a.0, a.1).cmp(&(b.0, b.1)),
                    _ => {
                        let (x, y) = (key(a.0, a.1, f), key(b.0, b.1, f));
                        match (&x, &y) {
                            (Value::Null, Value::Null) => Ordering::Equal,
                            // Missing values sort last either way.
                            (Value::Null, _) => return Ordering::Greater,
                            (_, Value::Null) => return Ordering::Less,
                            (Value::Number(p), Value::Number(q)) => p.as_f64().unwrap_or(0.0).total_cmp(&q.as_f64().unwrap_or(0.0)),
                            _ => x.as_str().unwrap_or_default().cmp(y.as_str().unwrap_or_default()),
                        }
                    }
                };
                let o = if *desc { o.reverse() } else { o };
                if o != Ordering::Equal {
                    return o;
                }
            }
            (a.0, a.1).cmp(&(b.0, b.1))
        });
    }
    let score = |s: f32| serde_json::from_str(&crate::retriever::score_json(s)).unwrap_or(Value::Null);
    let max = if by_field { None } else { hits.first().map(|h| h.2) };
    let mut page = Vec::new();
    for &(li, d, s) in hits.iter().skip(def.from).take(def.size) {
        let other = &run.leaves[li];
        let stored: Document = other.get_doc(d)?.unwrap_or_default();
        let mut m = Map::new();
        m.insert("_index".into(), json!(t.svc.name));
        m.insert("_id".into(), json!(stored.get(ID_FIELD).and_then(|v| v.as_text()).unwrap_or("")));
        if def.version {
            m.insert("_version".into(), json!(stored.get(VERSION_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)));
        }
        if def.seq_no_primary_term {
            m.insert("_seq_no".into(), json!(stored.get(SEQ_NO_FIELD).and_then(|v| v.as_i64()).unwrap_or(0)));
            m.insert("_primary_term".into(), json!(stored.get(PRIMARY_TERM_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)));
        }
        m.insert("_score".into(), if by_field { Value::Null } else { score(s) });
        if let Some(r) = stored.get(ROUTING_FIELD).and_then(|v| v.as_text()) {
            m.insert("_routing".into(), json!(r));
        }
        if def.source {
            if let Some(src) = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).and_then(|x| serde_json::from_str::<Value>(&t.visible_source(&crate::vector_source::load(x, &run.lookup, other, d))).ok()) {
                m.insert("_source".into(), src);
            }
        }
        let mut fields = Map::new();
        for (pat, fmt) in &def.docvalue_fields {
            let names = if pat.contains('*') { run.lookup.matching(pat) } else { vec![pat.clone()] };
            for n in names {
                let v = docvalue_values(other, &run.lookup, &n, d, fmt.as_deref())?;
                if !v.is_empty() {
                    fields.insert(n, Value::Array(v));
                }
            }
        }
        if !fields.is_empty() {
            m.insert("fields".into(), Value::Object(fields));
        }
        if by_field {
            m.insert("sort".into(), Value::Array(def.sort.iter().map(|(f, _)| if f == "_score" { score(s) } else { key(li, d, f) }).collect()));
        }
        page.push(Value::Object(m));
    }
    let total = if total_as_int { json!(hits.len()) } else { json!({ "value": hits.len(), "relation": "eq" }) };
    if let Some(error)=failure.into_inner(){return Err(error);}
    Ok(json!({ "hits": {
        "total": total,
        "max_score": max.map(score).unwrap_or(Value::Null),
        "hits": page,
    }}))
}

/// `inner_hits` of a `nested` query over a semantic_text field's chunks:
/// the matching chunks as nested hits (`<field>.inference.chunks`).
#[allow(clippy::too_many_arguments)]
fn chunk_inner_hits(t: &Target, cq: &crate::inference::chunk_query::ChunkQuery, def: &InnerHitsDef, leaf: &LeafReader<'_>, doc: DocId, id: &str, src: Option<&Value>, total_as_int: bool) -> Value {
    let path = format!("{}.inference.chunks", cq.field);
    let mut hits: Vec<(usize, f32)> = cq.chunk_scores(leaf, doc).into_iter().enumerate().filter_map(|(i, s)| s.map(|s| (i, s))).collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let max = hits.first().map(|h| h.1);
    // Legacy format: the chunk objects are in the source.
    let legacy_chunks: Vec<Value> = src
        .and_then(|s| crate::highlight::source_values(s, &cq.field).into_iter().next().cloned())
        .and_then(|v| v.pointer("/inference/chunks").and_then(Value::as_array).cloned())
        .unwrap_or_default();
    let mut page = Vec::new();
    for (i, score) in hits.iter().skip(def.from).take(def.size) {
        let mut m = Map::new();
        m.insert("_index".into(), json!(t.svc.name));
        m.insert("_id".into(), json!(id));
        m.insert("_nested".into(), json!({"field": path, "offset": i}));
        m.insert("_score".into(), serde_json::from_str(&crate::retriever::score_json(*score)).unwrap_or(Value::Null));
        let chunk = legacy_chunks.get(*i);
        if def.source {
            if let Some(c) = chunk {
                m.insert("_source".into(), c.clone());
            }
        }
        let mut inner = Map::new();
        for (pat, _) in &def.fields {
            if let (Some(rel), Some(c)) = (pat.strip_prefix(&format!("{path}.")), chunk) {
                if let Some(v) = c.get(rel).filter(|v| !v.is_null()) {
                    inner.insert(rel.to_string(), json!([v]));
                }
            }
        }
        if !inner.is_empty() {
            m.insert("fields".into(), json!({ path.clone(): [Value::Object(inner)] }));
        }
        page.push(Value::Object(m));
    }
    let total = if total_as_int { json!(hits.len()) } else { json!({ "value": hits.len(), "relation": "eq" }) };
    json!({ "hits": {
        "total": total,
        "max_score": max.map(|x| serde_json::from_str(&crate::retriever::score_json(x)).unwrap_or(Value::Null)).unwrap_or(Value::Null),
        "hits": page,
    }})
}

pub struct ExecOpts<'a> {
    pub terms_lookup: Option<TermsLookupFn<'a>>,
    pub cursors: Option<&'a [Cursor]>,
    pub max_buckets: usize,
    pub scroll_id: Option<String>,
    pub pit_id: Option<String>,
}

/// `SeqNoFieldMapper`: queries / sorts on `_seq_no` of an index with
/// `index.disable_sequence_numbers`.
pub(crate) const SEQ_NO_DISABLED: &str = "Cannot query field [_seq_no]: _seq_no cannot be queried when [index.disable_sequence_numbers] is [true]";

fn build_query(req: &SearchRequest, t: &Target, ctx: &QueryContext) -> Result<(Query, Query)> {
    build_query_with(req, t, ctx, None, None)
}

/// [`build_query`] with DFS kNN winners (`knn_fixed`: the searcher id of
/// this run and the merged global top-k) and this run's position among
/// the shards of the request (`slice_pos`: ordinal, count; sliced
/// searches; default: the shard id and the index's shard count).
fn build_query_with(req: &SearchRequest, t: &Target, ctx: &QueryContext, knn_fixed: Option<(u64, &[phases::ShardKnnHit])>, slice_pos: Option<(u32, u32)>) -> Result<(Query, Query)> {
    let (q, hit) = build_query_inner(req, t, ctx, knn_fixed).map_err(|e| {
        // `toQuery` failures of the search query become a `QueryShardException`.
        if e.kind == "illegal_argument_exception" && e.reason.starts_with("Too many ids specified") {
            let reason = format!("failed to create query: {}", e.reason);
            ctx.shard_err_pub(reason).caused_by(e)
        } else {
            e
        }
    })?;
    let Some((id, max)) = req.slice else { return Ok((q, hit)) };
    let (shard_index, num_shards) = slice_pos.unwrap_or_else(|| (t.shard.as_ref().map(|s| s.id).unwrap_or(0), t.svc.number_of_shards()));
    let sq = Query::Custom(Arc::new(crate::slice::SliceQuery { id, max, shard_index, num_shards }));
    let wrap = |q: Query| Query::Bool { must: vec![q], should: vec![], must_not: vec![], filter: vec![sq.clone()], minimum_should_match: MinShouldMatch::Count(0) };
    Ok((wrap(q), wrap(hit)))
}

fn build_query_inner(req: &SearchRequest, t: &Target, ctx: &QueryContext, knn_fixed: Option<(u64, &[phases::ShardKnnHit])>) -> Result<(Query, Query)> {
    // Alias and role (DLS) filters are parsed against the full mapping, not the FLS-restricted one.
    let full_lookup = t.field_access().map(|_| t.svc.lookup());
    let dls_owned = full_lookup.as_ref().map(|l| QueryContext { lookup: l, ..ctx.clone() });
    let dls_ctx: &QueryContext = dls_owned.as_ref().unwrap_or(ctx);
    if let Some(plan) = &req.rank_plan {
        // Compound retriever: rank docs for the hits, rank docs + the
        // zero-scored union of the children for aggregations / total.
        let union = parse_query(plan.union(), ctx)?;
        // Hits in fused order (no sort, filtering or paging after the
        // ranking): the page's ranks, with as many again of slack (a rank
        // doc deleted since the plan leaves room for the next).
        let page = (req.sort.is_empty() && req.search_after.is_none() && req.post_filter.is_none() && req.min_score.is_none()
            && req.collapse.is_none() && req.rescore.is_none() && req.terminate_after.is_none()
            // The total comes from the union ([`crate::retriever::fix_total`]), not the hits.
            && req.track_total_hits.is_some())
            .then(|| req.from.saturating_add(req.size).saturating_mul(2));
        let (mut all, mut top) = crate::retriever::plan_queries(plan, &t.svc.name, union, page);
        // `DefaultSearchContext.buildFilteredQuery`: on an index with nested
        // documents (`semantic_text` chunks included) a query that might match
        // them is restricted to root documents (a filter clause).
        if req.explain {
            let lk = t.svc.lookup();
            if lk.objects.values().any(|o| o.nested) || lk.fields.values().any(|f| crate::inference::semantic::is_inference_field(f)) {
                top = Query::Bool { must: vec![top], should: vec![], must_not: vec![], filter: vec![Query::MatchAll], minimum_should_match: MinShouldMatch::Count(0) };
            }
        }
        if let Some(f) = &t.effective_filter() {
            let fq = parse_query(f, dls_ctx)?;
            all = Query::Bool { must: vec![all], should: vec![], must_not: vec![], filter: vec![fq.clone()], minimum_should_match: MinShouldMatch::Count(0) };
            top = Query::Bool { must: vec![top], should: vec![], must_not: vec![], filter: vec![fq], minimum_should_match: MinShouldMatch::Count(0) };
        }
        if let Some(pf) = &req.post_filter {
            top = Query::Bool { must: vec![top], should: vec![], must_not: vec![], filter: vec![parse_query(pf, ctx)?], minimum_should_match: MinShouldMatch::Count(0) };
        }
        return Ok((all, top));
    }
    let mut q = match &req.query {
        // Dense semantic queries size their kNN by the request's `size`.
        Some(v) => crate::inference::semantic_query::with_request_size(req.size_explicit.then_some(req.size), || parse_query(v, ctx))?,
        // ES `hasOnlySuggest`: a suggest-only body skips the query phase (no hits).
        None if req.suggest.is_some() && req.aggs.is_none() && req.knn.is_empty() && req.retriever.is_none() => Query::MatchNone,
        None => Query::MatchAll,
    };
    if !req.knn.is_empty() {
        let mut knns = Vec::new();
        match knn_fixed {
            // DFS: the clauses match exactly the global winners on this shard.
            Some((searcher, fixed)) => knns = phases::fixed_knn_queries(req, ctx.leaves, searcher, fixed),
            None => {
                for k in &req.knn {
                    let o = k.as_object().ok_or_else(|| EsError::parsing("[knn] must be an object"))?;
                    knns.push(parse_knn_query(o, ctx, Some(req.size.max(1)))?);
                }
            }
        }
        q = if req.query.is_some() {
            let mut should = vec![q];
            should.extend(knns);
            Query::Bool { must: vec![], should, must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(1) }
        } else if knns.len() == 1 {
            knns.pop().unwrap()
        } else {
            Query::Bool { must: vec![], should: knns, must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(1) }
        };
    }
    if let Some(f) = &t.effective_filter() {
        let fq = parse_query(f, dls_ctx)?;
        q = Query::Bool { must: vec![q], should: vec![], must_not: vec![], filter: vec![fq], minimum_should_match: MinShouldMatch::Count(0) };
    }
    let hit_q = match &req.post_filter {
        Some(pf) => Query::Bool { must: vec![q.clone()], should: vec![], must_not: vec![], filter: vec![parse_query(pf, ctx)?], minimum_should_match: MinShouldMatch::Count(0) },
        None => q.clone(),
    };
    Ok((q, hit_q))
}

// Bool/parser wrappers do not turn a mapping-dependent shard failure into
// a coordinating request parse failure. Keep genuine syntax errors intact.
fn shard_query_error(error: EsError) -> EsError {
    let mut cause = &error;
    while cause.kind == "x_content_parse_exception" {
        let Some(inner) = cause.caused_by.as_deref() else { break };
        cause = inner;
    }
    if cause.kind == "query_shard_exception" { cause.clone() } else { error }
}

/// Failures ES raises while a shard builds its query: they reach the client
/// as that shard's failure inside `search_phase_execution_exception`.
fn shard_query_failure(e: &EsError) -> bool {
    e.kind == "query_shard_exception"
        || matches!(e.kind.as_str(), "too_many_nested_clauses" | "too_many_clauses")
        || e.kind == "illegal_argument_exception" && (e.reason.starts_with("Result window") || e.reason.starts_with("The length of ") && e.reason.ends_with("[index.max_regex_length] index level setting."))
        || e.kind == "circuit_breaking_exception" && e.reason.starts_with("[script] Too many dynamic script compilations")
}

fn wrap_shard_err(index: &str, e: EsError) -> EsError {
    if shard_query_failure(&e) {
        EsError::search_phase(index, e)
    } else {
        e
    }
}

/// Run a search over `targets` (with the script environment of the
/// request installed: stored scripts, runtime fields, scorer errors).
/// JSON keys that run scripts (which could read `_source` or other fields directly).
const SCRIPT_KEYS: &[&str] = &["script", "script_fields", "runtime_mappings", "_script", "init_script", "map_script", "combine_script", "reduce_script", "script_score", "script_query"];

fn json_uses_scripts(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, c)| SCRIPT_KEYS.contains(&k.as_str()) || json_uses_scripts(c)),
        Value::Array(a) => a.iter().any(json_uses_scripts),
        _ => false,
    }
}

/// Default-deny guards for requests the security layer restricted: features that
/// read documents outside the filtered query / source / lookup paths are refused.
fn check_security_restrictions(targets: &[Target], req: &SearchRequest) -> Result<()> {
    let forbidden = |what: &str| EsError::new(403, "security_exception", format!("{what} are not permitted on indices with field or document level security"));
    if !targets.iter().any(|t| t.is_security_restricted()) {
        return Ok(());
    }
    if req.suggest.is_some() {
        return Err(forbidden("suggesters"));
    }
    if targets.iter().any(|t| t.field_access().is_some()) {
        let parts = [&req.query, &req.post_filter, &req.aggs, &req.rescore, &req.script_fields, &req.runtime_mappings, &req.retriever, &req.rank, &req.sub_searches, &req.collapse_spec];
        let in_sort = req.sort_extra.values().any(json_uses_scripts);
        if in_sort || req.knn.iter().any(json_uses_scripts) || parts.iter().any(|p| p.as_ref().is_some_and(json_uses_scripts)) {
            return Err(forbidden("scripts"));
        }
    }
    Ok(())
}

/// Phase times of shard searches for benchmarks (the cluster's
/// `fixed_cost_latency`): while [`search_phases::ENABLED`], each
/// [`execute`] records its setup (window checks, can-match, views, leaves),
/// query (every run's collection), merge, fetch and the rest (aggregations
/// and the response), in microseconds. One relaxed load otherwise.
pub mod search_phases {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    static RECORDED: parking_lot::Mutex<Vec<Vec<(&'static str, u128)>>> = parking_lot::const_mutex(Vec::new());
    thread_local! {
        pub(super) static CURRENT: std::cell::RefCell<Vec<(&'static str, u128)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    pub fn enabled() -> bool { ENABLED.load(Relaxed) }
    pub(super) fn record(phases: Vec<(&'static str, u128)>) {
        if enabled() { RECORDED.lock().push(phases); }
    }
    /// Every record so far, oldest first; the buffer is emptied.
    pub fn take() -> Vec<Vec<(&'static str, u128)>> { std::mem::take(&mut *RECORDED.lock()) }

    static NAMED: parking_lot::Mutex<Vec<(&'static str, Vec<(&'static str, u128)>)>> = parking_lot::const_mutex(Vec::new());
    /// Record `phases` under `kind` (the HTTP layer's: request handling,
    /// the executor hop), when enabled.
    pub fn record_named(kind: &'static str, phases: Vec<(&'static str, u128)>) {
        if enabled() { NAMED.lock().push((kind, phases)); }
    }
    /// Every [`record_named`] record so far; the buffer is emptied.
    pub fn take_named() -> Vec<(&'static str, Vec<(&'static str, u128)>)> { std::mem::take(&mut *NAMED.lock()) }
}

/// ES `TaskCancelledException` (400) of this thread's cancelled search
/// task: `task cancelled [<reason>]`.
pub fn task_cancelled() -> EsError {
    let reason = rustic_core::cancel::reason().unwrap_or_else(|| "by user request".to_string());
    EsError::new(400, "task_cancelled_exception", format!("task cancelled [{reason}]"))
}

/// `Err(task_cancelled)` once this thread's search task was cancelled
/// (per shard, segment, hit or bucket: a thread-local read and a load).
#[inline]
pub fn check_cancelled() -> Result<()> {
    if rustic_core::cancel::is_cancelled() {
        return Err(task_cancelled());
    }
    Ok(())
}

/// The `timeout` of the search whose aggregations collect on this thread:
/// collection stops at the deadline (checked every 1024 docs) and the
/// search answers its partial aggregations with `timed_out: true`, as ES's
/// `ContextIndexSearcher` time-exceeded collection does.
#[derive(Clone, Default)]
pub(crate) struct AggDeadline(Option<(Instant, Arc<std::sync::atomic::AtomicBool>)>);

thread_local! {
    static AGG_DEADLINE: std::cell::RefCell<AggDeadline> = std::cell::RefCell::new(AggDeadline::default());
}

/// Restores the previous [`AggDeadline`] when dropped.
pub(crate) struct AggDeadlineScope(Option<AggDeadline>);

impl Drop for AggDeadlineScope {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            AGG_DEADLINE.with(|d| *d.borrow_mut() = previous);
        }
    }
}

impl AggDeadline {
    pub(crate) fn new(deadline: Option<Instant>) -> Self {
        AggDeadline(deadline.map(|d| (d, Arc::new(std::sync::atomic::AtomicBool::new(false)))))
    }
    /// This thread's.
    pub(crate) fn current() -> Self {
        AGG_DEADLINE.with(|d| d.borrow().clone())
    }
    /// Make it this thread's until the scope drops.
    pub(crate) fn enter(&self) -> AggDeadlineScope {
        AggDeadlineScope(Some(AGG_DEADLINE.with(|d| d.replace(self.clone()))))
    }
    /// The deadline passed (recorded: the search reports `timed_out`).
    pub(crate) fn expired(&self) -> bool {
        match &self.0 {
            Some((deadline, hit)) => {
                if hit.load(std::sync::atomic::Ordering::Relaxed) || Instant::now() >= *deadline {
                    hit.store(true, std::sync::atomic::Ordering::Relaxed);
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }
    pub(crate) fn is_set(&self) -> bool {
        self.0.is_some()
    }
    /// Whether some collection stopped at the deadline.
    pub(crate) fn hit(&self) -> bool {
        self.0.as_ref().is_some_and(|(_, hit)| hit.load(std::sync::atomic::Ordering::Relaxed))
    }
}

pub fn execute(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<SearchOutput> {
    let out = execute_traced(targets, req, opts);
    // A cancelled search task answers its cancellation, whatever partial
    // result or error its interrupted phases left (ES `TaskCancelledException`).
    if rustic_core::cancel::is_cancelled() {
        return Err(task_cancelled());
    }
    out
}

fn execute_traced(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<SearchOutput> {
    if search_phases::enabled() {
        let t = Instant::now();
        let out = execute_timed(targets, req, opts);
        let total = t.elapsed().as_micros();
        let mut phases = search_phases::CURRENT.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let known: u128 = phases.iter().map(|p| p.1).sum();
        phases.push(("rest", total.saturating_sub(known)));
        phases.push(("total", total));
        search_phases::record(phases);
        return out;
    }
    execute_timed(targets, req, opts)
}

fn execute_timed(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<SearchOutput> {
    check_security_restrictions(targets, req)?;
    let requested = targets;
    // Fan out: one run per shard copy.
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    // Terms lookups resolve once per request, not once per shard run (ES
    // rewrites them on the coordinator).
    type LookupKey = (String, String, String, Option<String>);
    let memo: parking_lot::Mutex<HashMap<LookupKey, Result<Vec<Value>>>> = parking_lot::Mutex::new(HashMap::new());
    let inner = opts.terms_lookup;
    let cached = |i: &str, id: &str, p: &str, r: Option<&str>| -> Result<Vec<Value>> {
        let key = (i.to_string(), id.to_string(), p.to_string(), r.map(str::to_string));
        if let Some(v) = memo.lock().get(&key) {
            return v.clone();
        }
        let v = match inner {
            Some(f) => f(i, id, p, r),
            None => Ok(Vec::new()),
        };
        memo.lock().insert(key, v.clone());
        v
    };
    let cached_ref: &(dyn Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync) = &cached;
    let memo_opts = ExecOpts { terms_lookup: inner.map(|_| cached_ref), cursors: opts.cursors, max_buckets: opts.max_buckets, scroll_id: opts.scroll_id.clone(), pit_id: opts.pit_id.clone() };
    let opts = &memo_opts;
    // Shard-level aggregation validation (an aggregator factory failing on
    // some indices): those shards fail, the others answer.
    if let Some(aggs) = &req.aggs {
        let (mut ok, mut ok_at) = (Vec::new(), Vec::new());
        let mut failed: Vec<(String, u32, EsError)> = Vec::new();
        for (ti, t) in targets.iter().enumerate() {
            match shard_agg_check(aggs, t) {
                Ok(()) => {
                    ok.push(t.clone());
                    ok_at.push(ti);
                }
                Err(e) => failed.push((t.svc.name.clone(), t.shard.as_ref().map(|s| s.id).unwrap_or(0), e)),
            }
        }
        if !failed.is_empty() {
            if ok.is_empty() {
                let (index, _, e) = failed.swap_remove(0);
                return Err(EsError::search_phase(&index, e));
            }
            let mut out = execute(&ok, req, opts)?;
            // Hit addresses name a target of `targets` (a query_then_fetch
            // fetch phase reads all of them), not of the subset searched.
            for p in &mut out.page {
                p.0 = ok_at[p.0];
            }
            // Lazily: hit and `top_hits` sources pass through verbatim.
            let mut rs = crate::source_json::ResponseSources::default();
            if let Ok(mut v) = rs.parse_lazy(out.json.as_bytes()) {
                if req.doc_refs || req.fetch_refs {
                    for h in v.pointer_mut("/hits/hits").and_then(Value::as_array_mut).into_iter().flatten() {
                        if let Some(ti) = h.pointer_mut("/_rustic_doc/0") {
                            *ti = json!(ti.as_u64().and_then(|i| ok_at.get(i as usize)).copied().unwrap_or(usize::MAX));
                        }
                    }
                }
                if let Some(sh) = v.get_mut("_shards").and_then(Value::as_object_mut) {
                    let total = sh.get("total").and_then(Value::as_u64).unwrap_or(0) + failed.len() as u64;
                    sh.insert("total".into(), json!(total));
                    sh.insert("failed".into(), json!(failed.len()));
                    let failures: Vec<Value> = failed
                        .iter()
                        .map(|(index, shard, e)| {
                            let mut reason = e.to_json()["error"].clone();
                            if let Some(o) = reason.as_object_mut() {
                                o.remove("root_cause");
                            }
                            json!({"shard": shard, "index": index, "node": crate::node::NODE_ID, "reason": reason})
                        })
                        .collect();
                    sh.insert("failures".into(), Value::Array(failures));
                }
                // A response that needed the lossless tree stays as it was
                // (as before: `Value` could not read it either).
                if let Some(json) = rs.render_lazy(&v) {
                    out.json = String::from_utf8(json).expect("JSON text is UTF-8");
                }
                out.hit_spans = None;
            }
            return Ok(out);
        }
    }
    // Field usage counts searches that ran: not one every shard failed
    // (nothing was read), and not twice for a partial aggregation failure
    // (its own `execute` above records the targets it searches).
    let record = |out: Result<SearchOutput>| {
        if out.is_ok() {
            crate::field_usage::record(requested, req);
        }
        out
    };
    if req.retriever.is_some() || req.rank.is_some() || req.sub_searches.is_some() {
        return record(crate::retriever::execute(targets, req, opts));
    }
    crate::inference::semantic::check_query(req.query.as_ref(), targets)?;
    let stored = targets.first().and_then(|t| crate::script::stored::for_index_dir(t.svc.path()));
    let mut env = crate::script::ScriptEnv::new(stored, req.runtime_mappings.clone());
    env.raw_script_fields = req.raw_script_fields.clone();
    env.raw_runtime_mappings = req.raw_runtime_mappings.clone();
    env.raw_sort = req.raw_sort.clone();
    env.raw_search_scripts = req.raw_search_scripts.clone();
    env.cancel = req.timeout.map(crate::script::runtime::Cancel::new);
    let first = targets.first().map(|t| t.svc.name.clone()).unwrap_or_default();
    record(crate::script::with_terms_lookup(opts.terms_lookup, || crate::script::with_env(env, || {
        if let Some(sf) = &req.script_fields {
            for t in targets {
                crate::script::search::check_script_fields(sf, t.svc.setting_usize("index.max_script_fields", 32)).map_err(|e| wrap_script_err(&t.svc.name, e))?;
            }
        }
        // A query_then_fetch query phase checks the fetch-only parts (when
        // parsed, and the script fields just above, in this script
        // environment: stored scripts resolve) where ES checks them, but
        // renders none of them: its fetch phase does.
        let query_phase;
        let req = if req.doc_refs && (req.highlight.is_some() || req.script_fields.is_some() || !req.docvalue_fields.is_empty() || !req.fields.is_empty() || req.version || req.seq_no_primary_term) {
            let mut q = req.clone();
            (q.highlight, q.script_fields, q.raw_script_fields) = (None, None, None);
            q.docvalue_fields.clear();
            q.fields.clear();
            (q.version, q.seq_no_primary_term) = (false, false);
            query_phase = q;
            &query_phase
        } else {
            req
        };
        crate::collapse::check_request(req, opts.scroll_id.is_some())?;
        let r = execute_inner(targets, req, opts).and_then(|mut out| {
            let sub_opts = ExecOpts { terms_lookup: opts.terms_lookup, cursors: None, max_buckets: opts.max_buckets, scroll_id: None, pit_id: None };
            if let Some(j) = crate::collapse::expand(&out.json, req, &|sub| execute_inner(targets, sub, &sub_opts).map(|o| o.json))? {
                out.json = j;
                out.hit_spans = None;
            }
            Ok(out)
        });
        let err = crate::script::env().and_then(|e| e.take_error());
        match (r, err) {
            (_, Some(e)) => Err(EsError::search_phase(&first, e)),
            (Err(e), None) => Err(wrap_script_err(&first, e)),
            (ok, None) => ok,
        }
    })))
}

/// The request's rescorers in order (`rescore` object or array): query
/// rescorers and `script` rescorers (`RescoreMode::Replace`).
pub(crate) fn rescore_steps(r: &Value, qctx: &QueryContext) -> Result<Vec<(Rescore, rustic_core::query::RescoreMode)>> {
    use rustic_core::query::RescoreMode;
    let items: Vec<&Value> = match r {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut out = Vec::new();
    for r0 in items {
        let window_size = r0.get("window_size").and_then(|w| w.as_u64().or_else(|| w.as_str().and_then(|s| s.trim().parse().ok()))).unwrap_or(10) as usize;
        if let Some(qo) = r0.get("query") {
            let rq = parse_query(qo.get("rescore_query").unwrap_or(&Value::Null), qctx)?;
            let mode = match qo.get("score_mode").and_then(Value::as_str) {
                Some("multiply") => RescoreMode::Multiply,
                Some("avg") => RescoreMode::Avg,
                Some("max") => RescoreMode::Max,
                Some("min") => RescoreMode::Min,
                _ => RescoreMode::Total,
            };
            out.push((
                Rescore {
                    window_size,
                    query: rq,
                    query_weight: qo.get("query_weight").and_then(Value::as_f64).unwrap_or(1.0) as f32,
                    rescore_query_weight: qo.get("rescore_query_weight").and_then(Value::as_f64).unwrap_or(1.0) as f32,
                },
                mode,
            ));
        } else if let Some(so) = r0.get("script").and_then(Value::as_object) {
            let q = crate::script::search::script_rescore_query(so, qctx)?;
            out.push((Rescore { window_size, query: q, query_weight: 1.0, rescore_query_weight: 1.0 }, RescoreMode::Replace));
        }
    }
    Ok(out)
}

/// Per-index aggregation factory checks (`innerBuild` failures): a
/// `date_histogram` with a calendar interval on a downsampled index.
fn shard_agg_check(aggs: &Value, t: &Target) -> Result<()> {
    let Some(interval) = t.svc.setting("index.downsample.interval").filter(|s| !s.is_empty()) else { return Ok(()) };
    let _ = interval;
    fn walk(v: &Value, t: &Target) -> Result<()> {
        let Some(o) = v.as_object() else { return Ok(()) };
        for def in o.values() {
            let Some(d) = def.as_object() else { continue };
            if let Some(dh) = d.get("date_histogram").and_then(Value::as_object) {
                let cal = dh.get("calendar_interval").is_some() || dh.get("interval").and_then(Value::as_str).is_some_and(|i| crate::date::CalUnit::parse(i).is_some());
                if cal {
                    let field = dh.get("field").and_then(Value::as_str).unwrap_or("");
                    let ty = t.svc.lookup().get(field).map(|f| f.declared.name()).unwrap_or("date");
                    return Err(EsError::illegal_argument(format!("Field [{field}] of type [{ty}] is not supported for aggregation [date_histogram] with interval type [calendar_interval]")));
                }
            }
            for k in ["aggs", "aggregations"] {
                if let Some(sub) = d.get(k) {
                    walk(sub, t)?;
                }
            }
        }
        Ok(())
    }
    walk(aggs, t)
}

/// DFS phase of `dfs_query_then_fetch` over `targets` (every local shard
/// copy they select): the statistics of `req`'s query terms and fields,
/// summed over those shards. A coordinator merges the results of all
/// shards with [`phases::merge_dfs`] and passes them back as
/// [`SearchRequest::dfs_stats`].
pub fn dfs_phase(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<phases::DfsStats> {
    check_security_restrictions(targets, req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let stored = targets.first().and_then(|t| crate::script::stored::for_index_dir(t.svc.path()));
    let env = crate::script::ScriptEnv::new(stored, req.runtime_mappings.clone());
    let first = targets.first().map(|t| t.svc.name.clone()).unwrap_or_default();
    crate::script::with_terms_lookup(opts.terms_lookup, || crate::script::with_env(env, || {
        let mut parts = Vec::with_capacity(targets.len());
        for t in targets {
            if !t.svc.is_open() {
                return Err(EsError::index_closed(&t.svc.name, &t.svc.uuid));
            }
            // The same view the search would read (see `execute_inner`).
            if t.snapshot.is_none() {
                t.shard_copy().shard.before_search();
            }
            let view = target_view(t);
            let leaves: Vec<LeafReader<'_>> = view.leaves();
            parts.push(phases::dfs_stats_over(t, &leaves, req, now, opts.terms_lookup)?);
        }
        let r = Ok(phases::merge_dfs(&parts));
        match crate::script::env().and_then(|e| e.take_error()) {
            Some(e) => Err(EsError::search_phase(&first, e)),
            None => r,
        }
    }))
}

/// The targets of a cluster coordinator's query_then_fetch query phase,
/// one per shard copy, each pinned to the reader view it searches now, so
/// the fetch phase reads the very docs the query phase found (ES's
/// `ReaderContext`). Holding a published view takes no lock.
pub fn pin_views(targets: &[Target]) -> Result<Vec<Target>> {
    let mut out = Vec::new();
    for t in expand_targets(targets)? {
        if t.snapshot.is_none() {
            t.shard_copy().sync_schema_for_search()?;
            t.shard_copy().shard.before_search();
        }
        let view = target_view(&t);
        out.push(Target { snapshot: Some(view), ..t });
    }
    Ok(out)
}

/// The reader view a search over `t` reads: its retained view, else the
/// shard copy's current published view. Taking it holds no shard lock.
pub(crate) fn target_view(t: &Target) -> Arc<ReaderView> {
    match &t.snapshot {
        Some(view) => view.clone(),
        None => t.shard_copy().shard.reader_view(),
    }
}

fn wrap_script_err(index: &str, e: EsError) -> EsError {
    if e.kind == "script_exception" || (e.kind == "illegal_argument_exception" && e.reason.starts_with("Trying to retrieve too many script_fields"))
        || e.kind == "circuit_breaking_exception" && e.reason.starts_with("[script] Too many dynamic script compilations") {
        EsError::search_phase(index, e)
    } else {
        e
    }
}

/// A breaker trip in a phase the node runs over every shard of the
/// search (doc sets, fetch, reduce): ES fails the search with
/// `search_phase_execution_exception` around it (all shards failed, 429).
fn phase_trip(targets: &[Target], phase: &str, e: EsError) -> EsError {
    let index = targets.first().map(|t| t.svc.name.clone()).unwrap_or_default();
    let mut w = EsError::search_phase(&index, e);
    for (k, v) in w.extra.iter_mut() {
        if k == "phase" {
            *v = json!(phase);
        }
    }
    w
}

fn execute_inner(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<SearchOutput> {
    // Every search (PIT and scroll ones included) counts as search load,
    // once: not again for a query_then_fetch's fetch phase.
    if req.fetch_docs.is_none() && !req.refetch {
        rustic_core::note_search();
    }
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let start = Instant::now();
    let _current = crate::shard::CurrentGuard::new(
        targets.iter().map(|t| (if req.fetch_docs.is_some() || req.refetch { &t.shard_copy().shard.counters.fetch_current } else { &t.shard_copy().shard.counters.query_current }, 1)).collect(),
    );
    let now = crate::date::now_millis();
    // `DefaultSearchContext.preProcess` checks the result window after the
    // shard parsed the request (`parseSource`): query parse errors win.
    let mut window_err: Vec<Option<EsError>> = Vec::with_capacity(targets.len());
    for t in targets {
        let max = t.svc.setting_usize("index.max_result_window", 10_000);
        window_err.push((req.from + req.size > max && opts.scroll_id.is_none()).then(|| {
            EsError::search_phase(
                &t.svc.name,
                EsError::illegal_argument(format!(
                    "Result window is too large, from + size must be less than or equal to: [{max}] but was [{}]. See the scroll api for a more efficient way to request large data sets. This limit can be set by changing the [index.max_result_window] index level setting.",
                    req.from + req.size
                )),
            )
        }));
        if !t.svc.is_open() {
            return Err(EsError::index_closed(&t.svc.name, &t.svc.uuid));
        }
        if let Some(rt) = &req.runtime_mappings {
            let meta = t.svc.meta();
            crate::tsdb::check_runtime_shadow(&meta.settings, &meta.mappings, rt)?;
        }
        if t.svc.is_time_series() && (req.sort.iter().any(|s| s.field == "_id") || req.aggs.as_ref().is_some_and(crate::tsdb::aggs_terms_on_id)) {
            return Err(EsError::query_shard(&t.svc.name, &t.svc.uuid, "Fielddata is not supported on [_id] field in [time_series] indices"));
        }
    }
    // The can-match phase (`TransportSearchAction.shouldPreFilterSearchShards`):
    // with more shards than `pre_filter_shard_size` (default 128, or 1 when an
    // index has a write block), shards whose `range` clauses match no value
    // are skipped. Decided before the read guards below are taken.
    let range_skip: Vec<bool> = if prefilter_applies(req, targets) {
        targets.iter().map(|t| req.query.as_ref().is_some_and(|q| range_disjoint(t, q))).collect()
    } else {
        vec![false; targets.len()]
    };
    // Each live search reads its shard's published reader view (the shard
    // as of the last refresh) and holds no shard lock while it runs: writes,
    // refreshes, seals and merges proceed meanwhile, and searches over
    // several shards cannot deadlock on lock order.
    for t in targets.iter().filter(|t| t.snapshot.is_none()) {
        // Mapping changes (e.g. a vector field's index options) reach the core.
        t.shard_copy().sync_schema_for_search()?;
        t.shard_copy().shard.before_search();
    }
    let views: Vec<Arc<ReaderView>> = targets.iter().map(target_view).collect();
    // Compound retrievers: collect every rank doc so score ties keep the
    // fused order (see the rank re-sort of the merged hits).
    let k = match &req.rank_plan {
        Some(p) if req.sort.is_empty() => (req.from + req.size).max(p.len()),
        _ => req.from + req.size,
    };
    // `random_reranker`: the query phase collects the rank window.
    let k = match req.random_rerank {
        Some((window, _)) if req.sort.is_empty() => k.max(window),
        _ => k,
    };
    // Every run's leaves, in the order hits / doc refs address them.
    let all_leaves: Vec<Vec<LeafReader<'_>>> = targets
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            let mut leaves: Vec<LeafReader<'_>> = views[ti].leaves();
            if t.svc.segments_sorted_on_timestamp() {
                crate::segsort::order_leaves(&mut leaves, &t.lookup());
            }
            leaves
        })
        .collect();
    // Several shards: top-level kNN is a DFS phase (each shard's
    // candidates, merged into the global top k, then matched exactly), and
    // `dfs_query_then_fetch` scores with statistics summed over the shards.
    let multi = targets.len() > 1;
    let knn_fixed: Option<Vec<phases::ShardKnnHit>> = if multi && !req.knn.is_empty() && req.rank_plan.is_none() {
        let mut cands = Vec::with_capacity(targets.len());
        for (ti, t) in targets.iter().enumerate() {
            cands.push(phases::knn_candidates_over(t, &all_leaves[ti], ti as u64, req, now, opts.terms_lookup)?);
        }
        Some(phases::merge_knn_candidates(req, &cands))
    } else {
        None
    };
    let local_dfs: Option<phases::DfsStats> = if req.admitted_dfs_stats.is_none() && req.dfs_stats.is_none() && req.dfs && multi {
        let mut parts = Vec::with_capacity(targets.len());
        for (ti, t) in targets.iter().enumerate() {
            parts.push(phases::dfs_stats_over(t, &all_leaves[ti], req, now, opts.terms_lookup)?);
        }
        Some(phases::merge_dfs(&parts))
    } else {
        None
    };
    let dfs = req.admitted_dfs_stats.as_deref().or_else(||req.dfs_stats.as_deref()).or(local_dfs.as_ref());
    let setup_at = start.elapsed();
    let mut runs: Vec<IndexRun> = Vec::with_capacity(targets.len());
    let mut shard_failures: Vec<(String, EsError)> = Vec::new();
    let mut prefilter_skipped = 0usize;
    for (ti, t) in targets.iter().enumerate() {
        // Between shards: a cancelled search task searches no further shard.
        check_cancelled()?;
        let lookup = t.lookup();
        let leaves = all_leaves[ti].clone();
        crate::script::enter_index(&t.svc.name, &lookup, || t.svc.mapping_runtime())?;
        let aliases: Vec<String> = t.svc.aliases().keys().cloned().collect();
        let default_fields = t.svc.query_default_fields();
        let qctx = QueryContext {
            deadline: req.timeout.and_then(|timeout| Instant::now().checked_add(timeout)),
            lookup: &lookup,
            index_name: &t.svc.name,
            index_uuid: &t.svc.uuid,
            aliases,
            now,
            leaves: &leaves,
            default_fields,
            terms_lookup: opts.terms_lookup,
            max_terms_count: t.svc.setting_usize("index.max_terms_count", 65_536),
            max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000),
            max_result_window: t.svc.setting_usize("index.max_result_window", 10_000),
            seq_no_disabled: t.svc.seq_no_disabled(),
            allow_unmapped_fields: t.svc.setting("index.query.parse.allow_unmapped_fields").map(|v| v != "false").unwrap_or(true),
            knn_prefilter: t.effective_filter(),
            knn_prefilter_lookup: Some(t.svc.lookup()),
            allow_expensive_queries: req.allow_expensive_queries && t.svc.setting("index.search.allow_expensive_queries").map(|v| v != "false").unwrap_or(true),
            mappings: Some(&**t.svc),
        };
        let (q, hit_q) = match build_query_with(req, t, &qctx, knn_fixed.as_deref().map(|f| (ti as u64, f)), Some((ti as u32, targets.len() as u32))).map_err(shard_query_error) {
            Ok(x) => x,
            // A shard-level query failure on some of the searched shards:
            // partial results with the failure reported in `_shards`.
            Err(e) if e.kind == "query_shard_exception" && targets.len() > 1 => {
                shard_failures.push((t.svc.name.clone(), e));
                let sort = Vec::new();
                runs.push(IndexRun { shard_ord: ti as i64, leaves, lookup, query: Query::MatchNone, visibility_query: Some(Query::MatchNone), visibility_context: Some(SearchContext::default()), top: TopDocs::default(), sort, terms: None, inner: Vec::new(), named: Vec::new(), scripts: Default::default() });
                continue;
            }
            Err(e) => return Err(wrap_shard_err(&t.svc.name, e)),
        };
        let visibility_query = aggregation_visibility(t, &qctx, req)?;
        let visibility_context = visibility_query.as_ref().map(|q|SearchContext::for_query(q, &leaves, Bm25Params::default()));
        if let Some(context) = &visibility_context { context.execution_result()?; }
        if let Some(e) = window_err.get_mut(ti).and_then(Option::take) {
            if let Some(h) = &req.highlight {
                QueryTerms::for_index(&q, h, &t.svc, &|v| parse_query(v, &qctx)).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
            }
            return Err(e);
        }
        if prefilter_applies(req, targets) && (range_skip[ti] || rewrites_to_none(&q)) {
            prefilter_skipped += 1;
        }
        let (q, hit_q) = match req.index_boosts.iter().find(|(n, _)| n == &t.svc.name) {
            Some((_, b)) => (Query::Boost { query: Box::new(q), boost: *b }, Query::Boost { query: Box::new(hit_q), boost: *b }),
            None => (q, hit_q),
        };
        let sort = resolve_sort(&req.sort, &lookup, &t.svc.name, &t.svc.uuid, &req.sort_extra, &qctx).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
        let knn_ih = crate::vectors::inner_hits_query(req.query.as_ref(), &req.knn, req.size, &lookup);
        let inner = inner_hit_defs(knn_ih.as_ref().or(req.query.as_ref()).or(req.rank_plan.as_ref().map(|p| p.union())), &qctx, &leaves).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
        let mut so = SearchOptions {
            top_k: k,
            from: 0,
            sort: sort.iter().map(|s| s.field.clone()).collect(),
            min_score: req.min_score,
            track_total_hits: req.track_total_hits.unwrap_or(TrackTotalHits::False),
            terminate_after: req.terminate_after,
            timeout: req.timeout,
            explain: req.explain,
            track_scores: req.track_scores,
            ..Default::default()
        };
        (so.similarity, so.field_similarity) = crate::similarity::options_for(&t.svc, &lookup);
        so.track_scores = scores_needed(req);
        if req.aggs.is_some() && so.track_total_hits != TrackTotalHits::True {
            // aggregations see every hit anyway
        }
        if let Some(c) = &req.collapse {
            if let Some(f) = lookup.get(c) {
                let kind = if f.ty.is_keyword_like() { SortKind::Keyword } else if f.ty.is_float() { SortKind::F64 } else { SortKind::I64 };
                if let Some(id) = f.core_id {
                    so.collapse_field = Some((id, kind));
                }
            }
        }
        if let Some(r) = &req.rescore {
            let mut steps = rescore_steps(r, &qctx).map_err(|e| wrap_shard_err(&t.svc.name, e))?.into_iter();
            if let Some((first, mode)) = steps.next() {
                so.rescore = Some(first);
                so.rescore_mode = mode;
                so.rescore_chain = steps.collect();
            }
        }
        // search_after / scroll cursor.
        if let Some(cs) = opts.cursors {
            so.search_after = cs.get(ti).cloned().flatten().map(|mut h| {
                // An unsorted (score-ordered) scroll resumes after the last hit by (score, doc).
                if req.sort.is_empty() {
                    h.sort = vec![SortValue::F64(h.score as f64)];
                }
                h
            });
            if cs.get(ti).map(|c| c.is_none()).unwrap_or(false) && cs.iter().any(|c| c.is_some()) {
                // This index has not returned anything yet: start from scratch.
            }
        } else if let Some(after) = &req.search_after {
            if after.len() != req.sort.len() {
                return Err(EsError::search_phase(
                    &t.svc.name,
                    EsError::illegal_argument(format!("search_after has {} value(s) but sort has {}.", after.len(), req.sort.len())),
                ));
            }
            let mut vals = Vec::new();
            let mut leaf = usize::MAX;
            let mut doc = u32::MAX;
            let mut score = f32::NEG_INFINITY;
            for (i, (v, s)) in after.iter().zip(&sort).enumerate() {
                if s.render == "doc" {
                    let g = v.as_i64().unwrap_or(0);
                    if g < 0 {
                        // Composite PIT fanout uses MIN for "after every doc"
                        // and -1 for "before every doc", in either direction.
                        // Both lie outside the nonnegative real document IDs.
                        let after_all = g == i64::MIN;
                        vals.push(SortValue::I64(if s.descending == after_all { -1 } else { i64::MAX }));
                    } else {
                        let g = if req.sort[i].field == "_shard_doc" {
                            match shard_doc_after(g, ti as i64, s.descending) {
                                Ok(local) => local,
                                Err(sentinel) => {
                                    vals.push(SortValue::I64(sentinel));
                                    continue;
                                }
                            }
                        } else {
                            g
                        };
                        leaf = (g >> 32) as usize;
                        doc = (g & 0xffff_ffff) as u32;
                        vals.push(SortValue::I64(g));
                    }
                    continue;
                }
                if s.render == "score" {
                    score = v.as_f64().unwrap_or(0.0) as f32;
                }
                if s.render == "geo" {
                    vals.push(SortValue::F64(v.as_f64().or_else(|| v.as_str().and_then(|x| x.parse().ok())).unwrap_or(f64::INFINITY) * s.scale));
                    continue;
                }
                vals.push(after_value(v, s.render, &lookup, &req.sort[i]).map_err(|e| wrap_shard_err(&t.svc.name, e))?);
            }
            so.search_after = Some(TopHit::new(leaf, doc, score, vals));
        }
        let sort_names: Vec<(String, bool)> = req.sort.iter().map(|s| (s.field.clone(), s.descending)).collect();
        let script_keys = crate::script::search::sort_keys(&sort_names, &so.sort, &req.sort_extra, &t.svc.name, &lookup).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
        if so.sort.is_empty() && so.search_after.is_some() {
            // The core ignores `search_after` for plain score collection: use an explicit score sort.
            so.sort = vec![SortField::Score { descending: true }];
        }
        let top = if let Some(docs) = &req.fetch_docs {
            // A query_then_fetch fetch phase: the query phase's winners.
            let hits: Vec<TopHit> = docs
                .iter()
                .filter(|d| d.0 == ti && leaves.get(d.1).is_some_and(|l| d.2 < l.num_docs()))
                .map(|&(_, leaf, doc, score)| TopHit::new(leaf, doc, score, vec![SortValue::F64(score as f64)]))
                .collect();
            TopDocs { max_score: hits.iter().map(|h| h.score).fold(f32::NEG_INFINITY, f32::max), hits, ..Default::default() }
        } else if k == 0 && so.track_total_hits == TrackTotalHits::False {
            TopDocs::default()
        } else if let Some(mut keys) = script_keys {
            crate::script::search::collect_sorted(&leaves, &hit_q, &mut keys, k, req.min_score, so.search_after.as_ref())?
        } else {
            phases::search_with_dfs(&leaves, &hit_q, &so, dfs, &lookup)?
        };
        // A cancelled task stopped the collection (reported as a timeout,
        // or as a kNN execution error): its cancellation is the answer.
        check_cancelled()?;
        if let Some(error) = &top.execution_error {
            return Err(EsError::search_phase(&t.svc.name, EsError::illegal_argument(error.clone())));
        }
        if let Some((doc, score)) = top.invalid_score {
            // `FunctionScoreQuery.FunctionFactorScorer.score`.
            let shown = if score.is_nan() { "NaN".to_string() } else { format!("{score:?}") };
            return Err(EsError::search_phase(
                &t.svc.name,
                EsError::illegal_argument(format!("function score query returned an invalid score: {shown} for doc: {doc}; score must be a non-negative real number")),
            ));
        }
        let mut top = top;
        if req.rank_plan.is_some() {
            top = crate::retriever::fix_total(req, top, &leaves, &q)?;
        }
        if req.collapse.is_some() && req.track_scores && !req.sort.is_empty() {
            // max_score over every matching doc, not only the group heads
            top.max_score = phases::search_with_dfs(&leaves, &hit_q, &SearchOptions { top_k: 1, ..Default::default() }, None, &lookup)?.max_score;
        }
        let terms = match &req.highlight {
            Some(h) => Some(QueryTerms::for_index(&q, h, &t.svc, &|v| parse_query(v, &qctx)).map_err(|e| wrap_shard_err(&t.svc.name, e))?),
            None => None,
        };
        let mut named = Vec::new();
        for (name, spec, is_knn) in named_queries(req.query.as_ref(), &req.knn) {
            let nq = if is_knn {
                parse_knn_query(spec.as_object().unwrap(), &qctx, Some(req.size.max(1)))
            } else {
                parse_query(&spec, &qctx)
            };
            let Ok(nq) = nq else { continue };
            let nctx = SearchContext::for_query(&nq, &leaves, Bm25Params::default());
            nctx.execution_result()?;
            named.push((name, nq, nctx));
        }
        runs.push(IndexRun { shard_ord: ti as i64, leaves, lookup, query: q, visibility_query, visibility_context, top, sort, terms, inner, named, scripts: Default::default() });
    }
    if shard_failures.len() == targets.len() {
        crate::error::in_shard_order(&mut shard_failures);
        if let Some((ix, e)) = shard_failures.first() {
            return Err(wrap_shard_err(ix, e.clone()));
        }
    }

    validate_sort_types(&runs, req)?;
    normalise_date_sorts(&mut runs, req);
    normalise_int_sorts(&mut runs);
    let query_at = start.elapsed();
    // `random_reranker`'s re-scored window (see below).
    let reranked: Vec<(usize, TopHit)>;
    // Merge hits across indices.
    let mut merged: Vec<(usize, &TopHit)> = runs.iter().enumerate().flat_map(|(ti, r)| r.top.hits.iter().map(move |h| (ti, h))).collect();
    let sort0 = runs.first().map(|r| &r.sort);
    merged.sort_by(|(ta, a), (tb, b)| {
        match sort0 {
            Some(s) if !s.is_empty() => {
                for (i, sp) in s.iter().enumerate() {
                    let o = match sp.render {
                        "score" => b.score.total_cmp(&a.score),
                        "doc" => {
                            let o = ta.cmp(tb).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc));
                            if sp.descending { o.reverse() } else { o }
                        }
                        _ => cmp_sort(a.sort.get(i).unwrap_or(&SortValue::Missing), b.sort.get(i).unwrap_or(&SortValue::Missing), sp.descending, sp.missing_first),
                    };
                    if o != Ordering::Equal {
                        return o;
                    }
                }
                ta.cmp(tb).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc))
            }
            _ => b.score.total_cmp(&a.score).then(ta.cmp(tb)).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)),
        }
    });
    if let (Some(plan), true) = (&req.rank_plan, req.sort.is_empty()) {
        // Compound retrievers: hits in fused rank order.
        let rank = |ti: usize, h: &TopHit| {
            let id = runs[ti].leaves[h.leaf].keyword_value(ID_FIELD, h.doc).unwrap_or_default();
            plan.rank_of(&targets[ti].svc.name, &id)
        };
        let mut keyed: Vec<(usize, (usize, &TopHit))> = merged.into_iter().map(|(ti, h)| (rank(ti, h), (ti, h))).collect();
        keyed.sort_by(|a, b| b.1 .1.score.total_cmp(&a.1 .1.score).then(a.0.cmp(&b.0)));
        merged = keyed.into_iter().map(|x| x.1).collect();
    }
    if let (Some((window, seed)), true) = (req.random_rerank, req.sort.is_empty()) {
        // ES `RandomRankFeaturePhaseRankCoordinatorContext`: the top
        // `rank_window_size` hits scored `Random(seed + doc).nextFloat()`,
        // `doc` the shard-level doc id (the leaf's base plus its doc), then
        // stably sorted by that score (ties keep the query order).
        let mut v: Vec<(usize, TopHit)> = merged
            .iter()
            .take(window)
            .map(|(ti, h)| {
                let base: u64 = runs[*ti].leaves[..h.leaf].iter().map(|l| u64::from(l.num_docs())).sum();
                let mut h = (*h).clone();
                h.score = crate::retriever::random_rerank_score(seed, (base + u64::from(h.doc)) as i32);
                (*ti, h)
            })
            .collect();
        v.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        reranked = v;
        merged = reranked.iter().map(|(ti, h)| (*ti, h)).collect();
    }
    if let (Some(cf), true) = (&req.collapse, runs.len() > 1) {
        // groups are collapsed per index: keep the best head of each group overall
        let mut seen = std::collections::HashSet::new();
        let mut failure=None;
        merged.retain(|(ti,h)| {
            let r=&runs[*ti];match docvalue_values(&r.leaves[h.leaf],&r.lookup,cf,h.doc,None) {
                Ok(values)=>seen.insert(crate::collapse::group_key(&values)),
                Err(error)=>{failure.get_or_insert(error);false},
            }
        });
        if let Some(error)=failure{return Err(error);}

    }
    let page: Vec<(usize, &TopHit)> = merged.into_iter().skip(req.from).take(req.size).collect();
    let mut cursors: Vec<Cursor> = opts.cursors.map(|c| c.to_vec()).unwrap_or_else(|| vec![None; runs.len()]);
    for (ti, h) in &page {
        cursors[*ti] = Some((*h).clone());
    }

    let total: u64 = runs.iter().map(|r| r.top.total_hits).sum();
    let gte = runs.iter().any(|r| r.top.total_hits_relation == TotalHitsRelation::Gte);
    // Shard totals merge up to the tracking threshold (SearchPhaseController).
    let (total, gte) = match req.track_total_hits {
        Some(TrackTotalHits::UpTo(n)) if total > n => (n, true),
        _ => (total, gte),
    };
    let mut timed_out = runs.iter().any(|r| r.top.timed_out) || crate::script::env().is_some_and(|e| e.timed_out());
    let terminated_early = runs.iter().any(|r| r.top.terminated_early);
    let scores_shown = hit_scores_shown(req);
    // `max_score` is tracked under relevance order or `track_scores` only.
    let max_score_shown = !sorted_by_field(req) || req.track_scores;

    // Fetch phase. The rendered hits, aggregation state and the response
    // built from them are held on the request breaker until it is returned.
    let fetch_start = Instant::now();
    let mem = crate::memory::RequestMemory::new();
    let mut fetched = vec![false; targets.len()];
    let mut hits_json: Vec<String> = Vec::with_capacity(page.len());
    let mut columns: Vec<ColumnHit> = Vec::new();
    mem.add(page.len() as u64 * 24, "<fetch_source>").map_err(|e| phase_trip(targets, "fetch", e))?;
    let light = |run: &IndexRun| req.doc_refs && req.light_refs && req.collapse.is_none() && run.inner.is_empty();
    // The page's stored documents, read per leaf in one pass over its store
    // chunks (each decompressed once) instead of one cached read per hit.
    let mut pre: Vec<Option<Document>> = Vec::new();
    // Without stored fields to render, a hit still reads its `_id`, which
    // a segment without an `_id` column keeps in its stored fields only:
    // those leaves' hits are read in the same batch (a `_source: false`
    // page, e.g. EQL's doc-value stage searches, decoded each hit's chunk
    // from its start for one `_id`).
    let stored = fetch_needs_stored(req);
    // (A segment with an id column reads `_id`s from it, not the store.)
    let id_from_store = |leaf: &LeafReader<'_>| matches!(leaf, LeafReader::Segment(s) if s.keyword_from_postings(ID_FIELD) && s.uid_column_field() != Some(ID_FIELD));
    if page.len() > 1 && !req.columns_only && (stored || page.iter().any(|(ti, h)| id_from_store(&runs[*ti].leaves[h.leaf]))) {
        pre.resize_with(page.len(), || None);
        let mut groups: Vec<((usize, usize), Vec<usize>)> = Vec::new();
        for (p, (ti, h)) in page.iter().enumerate() {
            if light(&runs[*ti]) || (!stored && !id_from_store(&runs[*ti].leaves[h.leaf])) { continue; }
            match groups.iter_mut().find(|g| g.0 == (*ti, h.leaf)) {
                Some(g) => g.1.push(p),
                None => groups.push(((*ti, h.leaf), vec![p])),
            }
        }
        // Column-stored fields come back only for what reads them.
        let restore = req.version || req.seq_no_primary_term || req.force_synthetic_source || !req.fields.is_empty()
            || !req.stored_fields.is_empty() || req.highlight.is_some() || req.script_fields.is_some() || req.runtime_mappings.is_some()
            || !matches!(req.source, SourceFilter::All);
        // A large page reads its leaves' documents in parallel, in runs of
        // consecutive hits (each run decompresses its own store chunks).
        let jobs: Vec<(usize, usize, &[usize])> = groups.iter().filter(|(_, at)| at.len() >= 2)
            .flat_map(|((ti, leaf), at)| {
                let chunk = if page.len() >= PARALLEL_FETCH_MIN { PARALLEL_READ_CHUNK } else { at.len() };
                at.chunks(chunk).map(move |run| (*ti, *leaf, run))
            })
            .collect();
        if jobs.len() > 1 {
            let request_ctx = crate::memory::RequestContext::capture();
            let read: Vec<Result<Vec<(usize, Document)>>> = rustic_core::query::searcher::par_map_search(jobs.len(), |j| request_ctx.run(|| {
                let (ti, leaf, at) = jobs[j];
                let docs: Vec<DocId> = at.iter().map(|&p| page[p].1.doc).collect();
                let mut out = Vec::with_capacity(at.len());
                runs[ti].leaves[leaf].get_docs_batched(&docs, restore, |i, document| out.push((at[i], document)))?;
                Ok(out)
            }));
            for r in read {
                for (p, document) in r? {
                    pre[p] = Some(document);
                }
            }
        } else {
            for (ti, leaf, at) in jobs {
                let docs: Vec<DocId> = at.iter().map(|&p| page[p].1.doc).collect();
                runs[ti].leaves[leaf].get_docs_batched(&docs, restore, |i, document| pre[at[i]] = Some(document))?;
            }
        }
    }
    // Per target, once for the page (an index setting read per hit).
    let mut excludes: Vec<Option<bool>> = vec![None; runs.len()];
    // A large page of plain hits (a scroll's 1,000-hit pages, a big
    // `size`): each hit renders from its own stored document and the
    // columns, so the hits render on the search pool in chunks, under the
    // request's context as the per-shard aggregations run, and join in page
    // order. Same output; the first error in page order is returned.
    let parallel_fetch = page.len() >= PARALLEL_FETCH_MIN && !req.columns_only && !req.explain
        && req.script_fields.is_none() && req.runtime_mappings.is_none() && req.highlight.is_none()
        && runs.iter().all(|r| r.inner.is_empty() && !light(r));
    if parallel_fetch {
        check_cancelled().map_err(|e| phase_trip(targets, "fetch", e))?;
        for (ti, _) in &page {
            if excludes[*ti].is_none() {
                excludes[*ti] = Some(hit_excludes_vectors(&targets[*ti], &runs[*ti], req));
            }
        }
        let slots: Vec<parking_lot::Mutex<Option<Document>>> = (0..page.len()).map(|p| parking_lot::Mutex::new(pre.get_mut(p).and_then(Option::take))).collect();
        let request_ctx = crate::memory::RequestContext::capture();
        let coordinator = crate::coord::current();
        let inference = crate::inference::current();
        let read_shard = selected_read_shard();
        let read_count = READ_SHARD_COUNT.with(|s| s.get());
        let lookup = crate::script::terms_lookup_ptr();
        let chunks = page.len().div_ceil(PARALLEL_FETCH_CHUNK);
        let excludes = &excludes;
        let rendered: Vec<Vec<Result<String>>> = rustic_core::query::searcher::par_map_search(chunks, |c| {
            request_ctx.run(|| crate::coord::scoped(coordinator.clone(), || crate::inference::scoped(inference.clone(), || {
                let _shard = ReadShardScope::enter(read_shard);
                let _count = ReadShardCountScope::enter(read_count);
                crate::script::with_terms_lookup_ptr(lookup, || {
                    (c * PARALLEL_FETCH_CHUNK..((c + 1) * PARALLEL_FETCH_CHUNK).min(page.len()))
                        .map(|p| {
                            let (ti, h) = page[p];
                            fetch_hit_read(&targets[ti], &runs[ti], &runs[ti].leaves[h.leaf], h.doc, Some(h), req, scores_shown, slots[p].lock().take(), excludes[ti])
                        })
                        .collect()
                })
            })))
        });
        check_cancelled().map_err(|e| phase_trip(targets, "fetch", e))?;
        for (p, js) in rendered.into_iter().flatten().enumerate() {
            let mut js = js?;
            let (ti, h) = page[p];
            fetched[ti] = true;
            // A coordinator's fetch round matches its winners by address.
            if (req.doc_refs || req.fetch_refs) && js.ends_with('}') {
                js.pop();
                js.push_str(&format!(",\"_rustic_doc\":[{ti},{},{}]}}", h.leaf, h.doc));
            }
            mem.add(js.capacity() as u64, "<fetch_source>").map_err(|e| phase_trip(targets, "fetch", e))?;
            hits_json.push(js);
        }
    }
    for (p, (ti, h)) in page.iter().enumerate().filter(|_| !parallel_fetch) {
        // Per hit (not per doc): the fetch phase stops once cancelled.
        check_cancelled().map_err(|e| phase_trip(targets, "fetch", e))?;
        let t = &targets[*ti];
        let run = &runs[*ti];
        let leaf = &run.leaves[h.leaf];
        fetched[*ti] = true;
        if req.columns_only {
            let c = column_hit(t, run, leaf, h.doc, req)?;
            mem.add(96 + c.values.iter().map(|v| 24 + v.iter().map(crate::memory::json_bytes).sum::<u64>()).sum::<u64>(), "<fetch_source>").map_err(|e| phase_trip(targets, "fetch", e))?;
            columns.push(c);
            continue;
        }
        let mut inner_root: Option<Document> = None;
        let mut js = match () {
            _ if light(run) => query_phase_hit(t, run, h, req, scores_shown)?,
            _ => {
                let exclude = *excludes[*ti].get_or_insert_with(|| hit_excludes_vectors(t, run, req));
                let doc = pre.get_mut(p).and_then(Option::take);
                if !run.inner.is_empty() { inner_root = doc.clone(); }
                fetch_hit_read(t, run, leaf, h.doc, Some(*h), req, scores_shown, doc, Some(exclude))?
            }
        };
        if !run.inner.is_empty() && js.ends_with('}') {
            js.pop();
            js.push_str(",\"inner_hits\":");
            js.push_str(&serde_json::to_string(&render_inner_hits(t, run, leaf, h.doc, &run.inner, req.rest_total_hits_as_int, inner_root.take())?)?);
            js.push('}');
        }
        if (req.doc_refs || req.fetch_refs) && js.ends_with('}') {
            js.pop();
            js.push_str(&format!(",\"_rustic_doc\":[{ti},{},{}]}}", h.leaf, h.doc));
        }
        mem.add(js.capacity() as u64, "<fetch_source>").map_err(|e| phase_trip(targets, "fetch", e))?;
        hits_json.push(js);
    }
    let fetch_time = fetch_start.elapsed();
    if search_phases::enabled() {
        let merge = fetch_start.duration_since(start).saturating_sub(query_at);
        search_phases::CURRENT.with(|c| c.borrow_mut().extend([("setup", setup_at.as_micros()), ("query", (query_at - setup_at).as_micros()),
            ("merge", merge.as_micros()), ("fetch", fetch_time.as_micros())]));
    }
    rustic_core::query::stats::add(rustic_core::query::stats::C::fetch_ns, fetch_time.as_nanos() as u64);
    rustic_core::query::stats::add(rustic_core::query::stats::C::fetch_hits, page.len() as u64);

    // Aggregations: a search over several shards reduces per-shard
    // partials (ES `InternalAggregation.reduce`), whether the shards are
    // one multi-shard index's or several indices' (a data stream's backing
    // indices, each with one shard): each shard's terms are its top
    // `shard_size`, with ES's doc_count_error_upper_bound and
    // sum_other_doc_count.
    let sharded = multi && (targets.iter().any(|t| t.svc.number_of_shards() > 1) || aggs_fields_uniform(req.aggs.as_ref(), targets));
    // Request-level aggregation validation (ES fails these before any shard runs).
    if let Some(spec) = &req.aggs {
        aggs::validate_request(spec)?;
    }
    // The aggregations' collection keeps to the search's `timeout` (from
    // the search's start, as ES measures it): partial aggregations and
    // `timed_out: true` once it passed.
    let agg_deadline = AggDeadline::new(req.timeout.and_then(|t| start.checked_add(t)));
    let _agg_deadline = req.aggs.as_ref().filter(|_| agg_deadline.is_set()).map(|_| agg_deadline.enter());
    check_cancelled()?;
    let reduced = match &req.aggs {
        Some(spec) if (sharded && crate::hooks::agg_partial::reducible(spec)) || !shard_failures.is_empty() || aggs::significant_text::present(spec) => shard_reduce_aggs(targets, &runs, req, spec, now, opts.max_buckets, &mut shard_failures)?,
        _ => None,
    };
    let geo_counts = match (&req.aggs, &reduced) {
        (Some(spec), None) => geo_point_count_aggs(spec, targets, &runs, req, opts.max_buckets),
        _ => None,
    };
    let aggs_json = match &req.aggs {
        None => None,
        Some(_) if reduced.is_some() => reduced,
        // `value_count`s of the filtering geo_point field over a pure geo
        // filter, count-only `geo_distance` rings: counted from the KD
        // trees, no doc collected.
        Some(spec) if geo_counts.is_some() => {
            let mut out = geo_counts.clone().unwrap_or_default();
            mem.add(crate::memory::json_bytes(&out), "<reduce_aggs>").map_err(|e| phase_trip(targets, "query", e))?;
            if req.typed_keys {
                out = add_typed_keys(out, spec);
            }
            Some(out)
        }
        Some(spec) => {
            let mut segs: Vec<AggSeg> = Vec::new();
            let mut seg_owner: Vec<usize> = Vec::new();
            // Count-only `auto_date_histogram`s of the field a bare range
            // query filters on (Rally metricbeat): its matches are the
            // column's docs in the range, which the aggregation counts from
            // the points index, so they are not collected.
            let plain = req.min_score.is_none() && !req.profile && req.runtime_mappings.is_none();
            let points_field = if plain { aggs::points_counted_field(spec) } else { None };
            // A histogram (+ simple metrics) of the field a bare double range
            // filters on (Rally nyc_taxis `distance_amount_agg`) folds the
            // range's docs from the column: they are not collected either.
            let hist_field = if plain { aggs::value_range_histogram_field(spec) } else { None };
            // A `nested` aggregation's count-only date histograms over every
            // root (Rally nested `nested-date-histo`) count the path's docs
            // from the points index; a `match_all`'s roots are not collected.
            let nested_path = if plain { aggs::nested_whole_path_spec(spec) } else { None };
            // A `range` with an `auto_date_histogram` (Rally big5
            // `range-auto-date-histo`) over a `match_all` folds every doc
            // from the columns: the roots are not collected.
            let range_auto = if plain { aggs::range_auto_date_fields(spec) } else { None };
            let mut elided: Vec<bool> = Vec::new();
            for (ti, run) in runs.iter().enumerate() {
                let ctx = SearchContext::for_query(&run.query, &run.leaves, Bm25Params::default());
                ctx.execution_result()?;
                let elide_nested = nested_path.is_some_and(|p| {
                    run.visibility_query.is_none() && run.lookup.doc_count_id.is_none() && run.query.is_match_all()
                        && run.lookup.objects.get(p).is_some_and(|o| o.nested)
                });
                let plain_run = run.visibility_query.is_none()
                    && run.lookup.doc_count_id.is_none()
                    && !run.lookup.objects.values().any(|o| o.nested);
                let elide_hist = plain_run && hist_field.is_some_and(|f| {
                    crate::script::runtime_field(Some(targets[ti].svc.name.as_str()), f).is_none()
                        && run.query.sole_f64_range().is_some_and(|(fid, _, _)| {
                            run.lookup.get(f).is_some_and(|rf| rf.core_id == Some(fid) && rf.doc_values
                                && matches!(rf.ty, EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat))
                        })
                });
                let elide_range_auto = plain_run && run.query.is_match_all() && range_auto.as_ref().is_some_and(|fields| {
                    fields.iter().all(|f| crate::script::runtime_field(Some(targets[ti].svc.name.as_str()), f).is_none())
                });
                let elide_run = elide_nested || elide_hist || elide_range_auto || points_field.is_some_and(|f| {
                    let date_col = |fid: Option<u16>| run.lookup.get(f).is_some_and(|rf| (fid.is_none() || rf.core_id == fid) && rf.doc_values && rf.ty == EsType::Date && rf.declared == EsType::Date);
                    run.visibility_query.is_none()
                        && run.lookup.doc_count_id.is_none()
                        && !run.lookup.objects.values().any(|o| o.nested)
                        && crate::script::runtime_field(Some(targets[ti].svc.name.as_str()), f).is_none()
                        && (run.query.sole_i64_range().is_some_and(|(fid, _, _)| date_col(Some(fid)))
                            // A `match_all` under count-only date histograms
                            // (Rally tsdb `date-histo-entire-range`).
                            || (run.query.is_match_all() && aggs::match_all_counted_field(spec) == Some(f) && date_col(None)))
                });
                // A run whose doc sets trip the breaker fails alone (an
                // index of a multi-index search): its docs are dropped and
                // given back, the others aggregate.
                let (mark, first_seg) = (mem.level(), segs.len());
                let mut tripped = None;
                'collect: for leaf in &run.leaves {
                    // Only roots are hidden (`deleted_bits` also holds a nested
                    // index's child docs): the deletions themselves must be none.
                    let undeleted = if elide_nested { leaf.deletions_only().is_none() } else { leaf.deleted_bits().is_none() };
                    if elide_run && leaf.num_docs() > 0 && undeleted && matches!(leaf, LeafReader::Segment(_)) {
                        segs.push(AggSeg { leaf: *leaf, lookup: &run.lookup, index: &targets[ti].svc.name, max_regex_length: targets[ti].svc.setting_usize("index.max_regex_length", 1000), docs: Vec::new(), scores: Vec::new() });
                        seg_owner.push(ti);
                        elided.push(true);
                        continue;
                    }
                    let (docs, scores) = match agg_leaf_docs(&run.query, leaf, &ctx, req.min_score, &mem) {
                        Ok(found) => found,
                        Err(e) => {
                            tripped = Some(e);
                            break 'collect;
                        }
                    };
                    segs.push(AggSeg { leaf: *leaf, lookup: &run.lookup, index: &targets[ti].svc.name, max_regex_length: targets[ti].svc.setting_usize("index.max_regex_length", 1000), docs, scores });
                    seg_owner.push(ti);
                    elided.push(false);
                }
                if let Some(e) = tripped {
                    if runs.len() == 1 && shard_failures.is_empty() {
                        return Err(phase_trip(targets, "query", e));
                    }
                    segs.truncate(first_seg);
                    seg_owner.truncate(first_seg);
                    elided.truncate(first_seg);
                    mem.release_to(mark);
                    shard_failures.push((targets[ti].svc.name.clone(), e));
                }
            }
            // Every target failed: those failed in the query phase stay in
            // `runs` (matching nothing), so count the failures, not the runs.
            if !targets.is_empty() && shard_failures.len() >= targets.len() {
                return Err(EsError::search_phase_all(&shard_failures));
            }
            let fetch = |si: usize, doc: DocId, score: f32, spec: &Value| -> Result<Value> {
                let ti = seg_owner[si];
                let run = &runs[ti];
                let leaf = &segs[si].leaf;
                let mut sub = SearchRequest::default();
                if let Some(s) = spec.get("_source") {
                    sub.source = parse_source_filter(s)?;
                }
                sub.version = spec.get("version").and_then(Value::as_bool).unwrap_or(false);
                sub.seq_no_primary_term = spec.get("seq_no_primary_term").and_then(Value::as_bool).unwrap_or(false);
                if let Some(f) = spec.get("fields") {
                    sub.fields = field_list(f);
                }
                if let Some(f) = spec.get("docvalue_fields") {
                    sub.docvalue_fields = field_list(f);
                }
                let th = TopHit::new(usize::MAX, doc, score, vec![]);
                let s = fetch_hit(&targets[ti], run, leaf, doc, Some(&th), &sub, true)?;
                hit_value(&s)
            };
            let visibility_error=parking_lot::Mutex::new(None);
            let all = |si: usize| -> Vec<DocId> {
                match runs[seg_owner[si]].visible_docs(&segs[si].leaf) {Ok(docs)=>docs,Err(error)=>{*visibility_error.lock()=Some(error);Vec::new()}}
            };
            let source = |si: usize, docs: &[(DocId, f32)], paths: &[String], emit: &mut dyn FnMut(DocId, Option<Value>) -> Result<()>| {
                let ti = seg_owner[si];
                fetch_aggregation_sources(&targets[ti], &runs[ti], &segs[si].leaf, docs, paths, emit)
            };
            let mapping_context: Vec<_> = targets.iter().zip(&runs).map(|(t,r)|(t.svc.name.as_str(),r.lookup.as_ref())).collect();
            let env = AggEnv { mapping_context: &mapping_context, now, max_buckets: opts.max_buckets, fetch_hit: &fetch, fetch_source: Some(&source), all_docs: &all, background_filtered: runs.iter().any(|r| r.visibility_query.is_some()), top_metrics_max_size: targets.first().map(|t| t.svc.setting_usize("index.top_metrics_max_size", 10)).unwrap_or(10), max_result_window: targets.first().map(|t| t.svc.setting_usize("index.max_result_window", 10_000)).unwrap_or(10_000), profile: req.profile };
            let ranges = seg_owner.iter().map(|&ti| runs[ti].query.required_i64_range()).collect();
            let f64_ranges = seg_owner.iter().zip(&elided).map(|(&ti, &e)| if e { runs[ti].query.sole_f64_range() } else { None }).collect();
            let aggregated=aggs::with_query_f64_ranges(f64_ranges, || aggs::with_query_ranges(ranges, || aggs::with_elided_root(elided, || aggs::run_aggregations(spec,&segs,&env))));
            if let Some(error)=visibility_error.lock().take() {return Err(error);}
            let mut out = aggregated.map_err(|e| {
                let index = targets.first().map(|t| t.svc.name.clone()).unwrap_or_default();
                // A breaker trip fails the shard (ES: all shards failed, 429).
                if crate::memory::is_breaking(&e) && !shard_failures.is_empty() {
                    union_aggregation_failure(targets.iter().map(|t| t.svc.name.as_str()), &shard_failures, e)
                } else if crate::memory::is_breaking(&e) && e.kind != "search_phase_execution_exception" {
                    EsError::search_phase(&index, e)
                } else if e.kind == "parsing_exception" || e.kind == "x_content_parse_exception" || e.kind == "action_request_validation_exception" || e.kind == "search_phase_execution_exception" || e.status != 400 {
                    e
                } else {
                    EsError::search_phase(&index, e)
                }
            })?;
            for run in &runs { run.visibility_result()?; }
            mem.add(crate::memory::json_bytes(&out), "<reduce_aggs>").map_err(|e| phase_trip(targets, "query", e))?;
            if req.typed_keys {
                out = add_typed_keys(out, spec);
            }
            Some(out)
        }
    };

    // Aggregation collection that stopped at the deadline: partial
    // aggregations, reported as a timed out search.
    // Aggregation scripts that stopped at the deadline (value scripts,
    // runtime fields, `scripted_metric` map phases) left partial
    // aggregations too.
    timed_out |= agg_deadline.hit() || crate::script::env().is_some_and(|e| e.timed_out());
    check_cancelled()?;

    // Assemble the response.
    let mut s = String::with_capacity(256 + hits_json.iter().map(|h| h.len() + 1).sum::<usize>());
    s.push('{');
    if let Some(id) = &opts.scroll_id {
        s.push_str(&format!("\"_scroll_id\":{},", serde_json::to_string(id)?));
    }
    if let Some(id) = &opts.pit_id {
        s.push_str(&format!("\"pit_id\":{},", serde_json::to_string(id)?));
    }
    s.push_str(&format!("\"took\":{},\"timed_out\":{timed_out},", start.elapsed().as_millis()));
    // ES also reports `terminated_early: false` on the size-0 exact-count shortcut (no aggs, tth:true).
    let count_shortcut = req.size == 0 && req.aggs.is_none() && matches!(req.track_total_hits, Some(TrackTotalHits::True));
    if req.terminate_after.is_some() || count_shortcut {
        s.push_str(&format!("\"terminated_early\":{terminated_early},"));
    }
    // One run per shard copy searched.
    let n: usize = targets.len();
    // The can-match phase skips every shard unless the request has
    // aggregations (`requireAtLeastOneMatch`: one shard runs for them).
    // Part of a larger (cluster) search: the coordinator applies that rule
    // over every shard, so the raw count is reported.
    let partial = READ_SHARD_COUNT.with(|slot| slot.get()).is_some_and(|c| c > n);
    let skipped = if prefilter_skipped >= n && req.aggs.is_some() && !partial { n.saturating_sub(1) } else { prefilter_skipped.min(n) };
    if !shard_failures.is_empty() {
        crate::error::in_shard_order(&mut shard_failures);
        let failed = shard_failures.len();
        let failures: Vec<Value> = shard_failures
            .iter()
            .map(|(ix, e)| json!({"shard": 0, "index": ix, "node": crate::node::NODE_ID, "reason": e.shard_reason()}))
            .collect();
        s.push_str(&format!("\"_shards\":{{\"total\":{n},\"successful\":{},\"skipped\":{skipped},\"failed\":{failed},\"failures\":{}}},\"hits\":{{", n - failed, serde_json::to_string(&failures)?));
    } else {
        s.push_str(&format!("\"_shards\":{{\"total\":{n},\"successful\":{n},\"skipped\":{skipped},\"failed\":0}},\"hits\":{{"));
    }
    if req.track_total_hits.is_none() && req.rest_total_hits_as_int {
        s.push_str("\"total\":-1,");
    }
    if req.track_total_hits.is_some() {
        if req.rest_total_hits_as_int {
            s.push_str(&format!("\"total\":{total},"));
        } else {
            s.push_str(&format!("\"total\":{{\"value\":{total},\"relation\":\"{}\"}},", if gte { "gte" } else { "eq" }));
        }
    }
    let max_score: Option<f32> = if req.random_rerank.is_some() && req.sort.is_empty() {
        // ES `RankFeaturePhase.maxScore`: over the page's re-scored hits.
        page.iter().map(|(_, h)| h.score).reduce(f32::max)
    } else if max_score_shown && !page.is_empty() {
        runs.iter().map(|r| r.top.max_score).filter(|x| x.is_finite()).reduce(f32::max)
    } else {
        None
    };
    match max_score {
        Some(m) if !runs.iter().all(|r| r.top.hits.is_empty()) => s.push_str(&format!("\"max_score\":{},", crate::retriever::score_json(m))),
        _ => s.push_str("\"max_score\":null,"),
    }
    s.push_str("\"hits\":[");
    for (i, h) in hits_json.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(h);
    }
    let hits_end = s.len();
    let hit_lens: Vec<u32> = hits_json.iter().map(|h| u32::try_from(h.len()).unwrap_or(u32::MAX)).collect();
    s.push_str("]}");
    if let Some(a) = aggs_json {
        s.push_str(",\"aggregations\":");
        s.push_str(&if aggregation_shard_response()||!aggs::has_raw_scripted_value(&a){serde_json::to_string(&a)?}else{String::from_utf8(crate::source_json::render_aggregations(&a)).expect("JSON text is UTF-8")});
    }
    if let Some(sg) = &req.suggest {
        let st: Vec<crate::suggest::SuggestTarget> = runs
            .iter()
            .zip(targets)
            .map(|(r, t)| crate::suggest::SuggestTarget { index: &t.svc.name, leaves: &r.leaves, lookup: &r.lookup, svc: Some(&*t.svc) })
            .collect();
        s.push_str(",\"suggest\":");
        s.push_str(&serde_json::to_string(&crate::suggest::run_with_source(sg, &st, req.typed_keys, &req.source)?)?);
    }
    if req.profile {
        s.push_str(",\"profile\":{\"shards\":[]}");
    }
    s.push('}');
    for r in &runs {
        let _ = &r.terms;
    }
    let hits_n = page.len();
    let page_keys: Vec<(usize, usize, DocId, f32)> = page.iter().map(|(ti, h)| (*ti, h.leaf, h.doc, h.score)).collect();
    let total_time = start.elapsed();
    // A query_then_fetch phase counts as ES's phase does: the query phase
    // one query (its hits are only ids), the fetch phase one fetch.
    for (ti, t) in targets.iter().enumerate() {
        let c = &t.shard_copy().shard.counters;
        match (req.doc_refs, req.fetch_docs.is_some() || req.refetch) {
            (true, _) => c.record_search(total_time, Duration::ZERO, false),
            (_, true) => {
                if fetched[ti] {
                    c.record_fetch(total_time)
                }
            }
            // A single-round search: the coordinator counts the fetch on
            // the shards whose hits it returns.
            _ if req.defer_fetch_stats => c.record_search(total_time, fetch_time, false),
            _ => c.record_search(total_time, fetch_time, fetched[ti]),
        }
    }
    Ok(SearchOutput { json: s, cursors, hits: hits_n, total, page: page_keys, fetch_nanos: u64::try_from(fetch_time.as_nanos()).unwrap_or(u64::MAX), hit_spans: Some((hits_end, hit_lens)), columns })
}

/// The response of `aggs` when every aggregation's result is known from
/// the KD trees without collecting a doc (`None`: aggregate as usual).
/// Each aggregation (no options beyond those named, no sub-aggregations)
/// is one of:
///
/// - a `value_count` of the `geo_point` field a pure geo filter `query`
///   filters on (Rally `distanceFilterCount*`): every leaf counts its
///   matches from the KD tree ([`Query::exact_count`]: sealed, no
///   deletes, one point per doc), each match holding exactly one value;
/// - a count-only `arc` `geo_distance` of a `geo_point` field (Rally
///   `distanceRange`, `distanceFilterRange`): per bucket, the docs with a
///   point in its distance ring ([`aggs::DistanceCounts`], the
///   aggregation's own distance test), under a `match_all` (any point of
///   a doc) or a pure geo filter on the same single-valued field (its
///   predicate and the ring, per point);
/// - a `value_count` of a numeric field under any query that counts its
///   matches exactly ([`Query::exact_count`], e.g. a `geo_shape`
///   relation on a triangle-indexed segment: Rally geoshape
///   `polygon-*-count`), on leaves where every doc has exactly one value
///   of the field (a dense single-valued column): the values are the
///   matches;
/// - a `geo_centroid` of a `geo_point` field (Rally `polygon-centroid`,
///   `centroid`) over at least [`TREE_CENTROID_MIN_POINTS`] points, under
///   a `match_all` (every point) or a pure geo filter on the same
///   single-valued field (the points its KD walk accepts): compensated
///   sums of the points in doc order, chunk by chunk on several threads,
///   the chunk sums combined in order with their corrections (as ES
///   reduces its shards' compensated sums), instead of collecting the
///   matches and summing on one thread. Sums may differ from one
///   sequential sum in the last bits, as ES's do between shard layouts.
fn geo_point_count_aggs(spec: &Value, targets: &[Target], runs: &[IndexRun], req: &SearchRequest, max_buckets: usize) -> Option<Value> {
    use rustic_core::geo::GeoPred;
    enum Ask {
        Values(String),
        Distance(aggs::DistanceCounts),
        Centroid(String),
        Grid(aggs::PointGrid),
    }
    if req.min_score.is_some() || req.runtime_mappings.is_some() || req.profile || runs.len() != targets.len() {
        return None;
    }
    let m = spec.as_object()?;
    if m.is_empty() {
        return None;
    }
    let mut asks: Vec<(&String, Ask)> = Vec::with_capacity(m.len());
    let mut buckets = 0usize;
    for (name, def) in m {
        let d = def.as_object()?;
        if d.len() != 1 {
            return None;
        }
        if let Some(dc) = aggs::DistanceCounts::parse(d) {
            buckets += dc.buckets();
            asks.push((name, Ask::Distance(dc)));
            continue;
        }
        if let Some(pg) = aggs::PointGrid::parse(d) {
            asks.push((name, Ask::Grid(pg)));
            continue;
        }
        if let Some(c) = d.get("geo_centroid") {
            let c = c.as_object()?;
            if c.len() != 1 {
                return None;
            }
            asks.push((name, Ask::Centroid(c.get("field")?.as_str()?.to_string())));
            continue;
        }
        let vc = d.get("value_count")?.as_object()?;
        if vc.len() != 1 {
            return None;
        }
        asks.push((name, Ask::Values(vc.get("field")?.as_str()?.to_string())));
    }
    let mut centroid_points = 0usize;
    let mut sums: Vec<(crate::esql::partial::Kahan, crate::esql::partial::Kahan)> = vec![Default::default(); asks.len()];
    if buckets > max_buckets {
        return None;
    }
    let mut counts: Vec<Vec<u64>> = asks.iter().map(|(_, a)| vec![0; if let Ask::Distance(dc) = a { dc.buckets() } else { 1 }]).collect();
    for (ti, run) in runs.iter().enumerate() {
        if run.visibility_query.is_some() {
            return None;
        }
        let qpred = run.query.geo_pred();
        let all = qpred.is_none() && run.query.is_match_all();
        // A `match_all` also matches no nested child doc.
        if all && run.lookup.objects.values().any(|o| o.nested) {
            return None;
        }
        // Per ask: the geo field id, or (`numeric`) the numeric column a
        // value_count reads, counted as the query's exact matches where
        // every doc has one value.
        let mut ids: Vec<u16> = Vec::with_capacity(asks.len());
        let mut numeric: Vec<bool> = Vec::with_capacity(asks.len());
        for (_, ask) in &asks {
            let f = match ask {
                Ask::Values(f) | Ask::Centroid(f) => f.as_str(),
                Ask::Distance(dc) => dc.field.as_str(),
                Ask::Grid(pg) => pg.field.as_str(),
            };
            if crate::script::runtime_field(Some(targets[ti].svc.name.as_str()), f).is_some() {
                return None;
            }
            let rf = run.lookup.get(f)?;
            if let (Ask::Values(_), true) = (ask, rf.declared.is_numeric() && rf.ty == rf.declared) {
                if !rf.doc_values {
                    return None;
                }
                ids.push(rf.core_id?);
                numeric.push(true);
                continue;
            }
            // Every other ask needs a pure geo filter on its geo_point
            // field (value_count, any) or that or match_all.
            if qpred.is_none() && !all {
                return None;
            }
            let gid = rf.geo_id?;
            if rf.declared != EsType::GeoPoint || rf.ty != EsType::GeoPoint || !rf.doc_values || qpred.as_ref().is_some_and(|(q, _)| *q != gid) || (all && matches!(ask, Ask::Values(_))) {
                return None;
            }
            ids.push(gid);
            numeric.push(false);
        }
        // Grid asks' segment trees, celled after the walk over the segments,
        // in parallel (one fork of the grid per segment, merged).
        let mut grid_trees: Vec<(usize, &rustic_core::geo::GeoIndex)> = Vec::new();
        // Centroid asks' segment trees: summed after the walk, segments in
        // parallel, their sums then added in segment order (as one pass in
        // segment order adds them).
        let mut centroid_trees: Vec<(usize, &rustic_core::geo::GeoIndex, u32)> = Vec::new();
        // Values asks count the query's matches per segment: those counts
        // (KD-tree counts, no doc visited) taken for every segment at once,
        // in parallel, instead of one segment after another below.
        let exact: Vec<Option<u64>> = if asks.iter().any(|(_, ask)| matches!(ask, Ask::Values(_))) {
            use rayon::prelude::*;
            run.leaves.par_iter().map(|leaf| if leaf.num_docs() == 0 { Some(0) } else { run.query.exact_count(leaf) }).collect()
        } else {
            Vec::new()
        };
        for (li, leaf) in run.leaves.iter().enumerate() {
            if leaf.num_docs() == 0 {
                continue;
            }
            if leaf.deleted_bits().is_some() {
                return None;
            }
            let LeafReader::Segment(seg) = leaf else { return None };
            for (ai, (((((_, ask), out), &gid), sum), &num)) in asks.iter_mut().zip(&mut counts).zip(&ids).zip(&mut sums).zip(&numeric).enumerate() {
                match ask {
                    // One value per doc: the values are the matches.
                    Ask::Values(_) if num => {
                        if !leaf.numeric_one_value_per_doc(gid) {
                            return None;
                        }
                        out[0] += exact[li]?;
                    }
                    Ask::Values(_) => out[0] += exact[li]?,
                    Ask::Grid(_) => {
                        let Some(g) = seg.geo_index(gid) else { continue };
                        if !g.single_valued() {
                            return None;
                        }
                        grid_trees.push((ai, g));
                    }
                    Ask::Centroid(_) => {
                        let Some(g) = seg.geo_index(gid) else { continue };
                        centroid_points += g.values().len();
                        centroid_trees.push((ai, g, seg.num_docs));
                    }
                    Ask::Distance(dc) => {
                        let Some(g) = seg.geo_index(gid) else { continue };
                        for (i, n) in out.iter_mut().enumerate() {
                            let ring = GeoPred::Leaf(std::sync::Arc::new(dc.ring(i)));
                            *n += match &qpred {
                                None if g.single_valued() => g.pred_count(&ring)?,
                                None => g.any_point_hits(&ring, seg.num_docs).len() as u64,
                                Some((_, q)) => g.pred_count(&GeoPred::And(vec![q.clone(), ring]))?,
                            };
                        }
                    }
                }
            }
        }
        if !centroid_trees.is_empty() {
            use rayon::prelude::*;
            let parts: Vec<Option<(usize, (u64, crate::esql::partial::Kahan, crate::esql::partial::Kahan))>> = centroid_trees
                .par_iter()
                .map(|&(ai, g, num_docs)| {
                    let hits = match &qpred {
                        None => None,
                        Some((_, q)) => Some(g.pred_hits(q, num_docs)?),
                    };
                    Some((ai, tree_centroid_sums(g, hits)))
                })
                .collect();
            for part in parts {
                let (ai, (n, lat, lon)) = part?;
                counts[ai][0] += n;
                sums[ai].0.add(lat.value, lat.delta);
                sums[ai].1.add(lon.value, lon.delta);
            }
        }
        if !grid_trees.is_empty() {
            use rayon::prelude::*;
            let pred = match &qpred {
                Some((_, q)) => q.clone(),
                None => GeoPred::All,
            };
            let parts: Vec<(usize, aggs::PointGrid)> = grid_trees
                .par_iter()
                .map(|&(ai, g)| {
                    let Ask::Grid(base) = &asks[ai].1 else { unreachable!() };
                    let mut pg = base.fork();
                    g.visit_pred_leaves(&pred, |bx, len, inside, points| {
                        if inside && pg.add_leaf(bx, len as u64) {
                            return;
                        }
                        points(&mut |lat, lon| {
                            if inside || pred.test(lat, lon) {
                                pg.add_point(lat, lon);
                            }
                        });
                    });
                    (ai, pg)
                })
                .collect();
            for (ai, part) in parts {
                if let Ask::Grid(pg) = &mut asks[ai].1 {
                    pg.merge(part);
                }
            }
        }
    }
    // Too many buckets: the aggregation reports the error.
    if asks.iter().map(|(_, a)| if let Ask::Grid(pg) = a { pg.buckets() } else { 0 }).sum::<usize>() + buckets > max_buckets {
        return None;
    }
    // Small centroids keep the collected path's one sequential sum.
    if asks.iter().any(|(_, a)| matches!(a, Ask::Centroid(_))) && centroid_points < tree_centroid_min_points() {
        return None;
    }
    let mut out = Map::new();
    for (((name, ask), n), (lat, lon)) in asks.iter().zip(&counts).zip(&sums) {
        let v = match ask {
            Ask::Values(_) => json!({ "value": n[0] }),
            Ask::Distance(dc) => dc.render(n),
            Ask::Grid(pg) => pg.render(),
            Ask::Centroid(_) if n[0] == 0 => json!({ "count": 0 }),
            Ask::Centroid(_) => {
                let c = n[0] as f64;
                json!({ "location": { "lat": values::java_double(lat.value / c), "lon": values::java_double(lon.value / c) }, "count": n[0] })
            }
        };
        out.insert((*name).clone(), v);
    }
    Some(Value::Object(out))
}

/// Points a `geo_centroid` must sum to take the KD-tree path of
/// [`geo_point_count_aggs`].
const TREE_CENTROID_MIN_POINTS: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    /// Tests on this thread: the [`TREE_CENTROID_MIN_POINTS`] in force.
    pub(crate) static TEST_CENTROID_MIN_POINTS: std::cell::Cell<usize> = const { std::cell::Cell::new(TREE_CENTROID_MIN_POINTS) };
}

fn tree_centroid_min_points() -> usize {
    #[cfg(test)]
    return TEST_CENTROID_MIN_POINTS.with(|c| c.get());
    #[cfg(not(test))]
    TREE_CENTROID_MIN_POINTS
}

/// Points one thread sums in [`tree_centroid_sums`].
const CENTROID_CHUNK: usize = 1 << 20;

/// The count and compensated latitude / longitude sums of the points of
/// `g` whose doc is in `hits` (all points when `None`), in doc order:
/// chunks summed in parallel, combined in order with their corrections.
pub(crate) fn tree_centroid_sums(g: &rustic_core::geo::GeoIndex, hits: Option<rustic_core::geo::DocHits>) -> (u64, crate::esql::partial::Kahan, crate::esql::partial::Kahan) {
    use crate::esql::partial::Kahan;
    use rayon::prelude::*;
    use rustic_core::geo::{decode_lat, decode_lon, DocHits};
    let pts = g.values();
    let bits: Option<Vec<u64>> = hits.map(|h| match h {
        DocHits::Bits(b) => b,
        DocHits::Sorted(v) => {
            let mut b = vec![0u64; v.last().map_or(0, |&d| d as usize / 64 + 1)];
            for d in v {
                b[d as usize / 64] |= 1u64 << (d % 64);
            }
            b
        }
    });
    let chunks: Vec<std::ops::Range<usize>> = (0..pts.len()).step_by(CENTROID_CHUNK).map(|a| a..(a + CENTROID_CHUNK).min(pts.len())).collect();
    let parts: Vec<(u64, Kahan, Kahan)> = chunks
        .into_par_iter()
        .map(|r| {
            let (mut n, mut lat, mut lon) = (0u64, Kahan::default(), Kahan::default());
            for e in pts.iter_range(r) {
                if let Some(b) = &bits {
                    if !b.get(e.doc as usize / 64).is_some_and(|w| w >> (e.doc % 64) & 1 == 1) {
                        continue;
                    }
                }
                lat.add(decode_lat({ e.lat }), 0.0);
                lon.add(decode_lon({ e.lon }), 0.0);
                n += 1;
            }
            (n, lat, lon)
        })
        .collect();
    let (mut n, mut lat, mut lon) = (0u64, Kahan::default(), Kahan::default());
    for (k, a, b) in parts {
        n += k;
        lat.add(a.value, a.delta);
        lon.add(b.value, b.delta);
    }
    (n, lat, lon)
}

/// The docs of `leaf` an aggregation sees (live matches of `query` that
/// pass `min_score`, ascending) and their scores, each doc charged 8 bytes
/// on `mem` (a trip is returned; the caller drops the partial set).
///
/// A `match_all` (possibly wrapped: it scores uniformly) over a leaf without
/// deleted or nested child docs matches every doc `0..num_docs` with the
/// score of its first: those are filled directly instead of advancing and
/// scoring the scorer once per doc. ES collects such a shard through its
/// match-all weight's bulk scorer (`DefaultBulkScorer` over all docs) or,
/// for a top-level count-only histogram, not per doc at all.
fn agg_leaf_docs(query: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext, min_score: Option<f32>, mem: &crate::memory::RequestMemory) -> std::result::Result<(Vec<DocId>, Vec<f32>), EsError> {
    let mut docs = Vec::new();
    let mut scores = Vec::new();
    // Between segments: a cancelled search collects nothing more; one past
    // its `timeout` keeps what it collected (partial aggregations).
    check_cancelled()?;
    let deadline = AggDeadline::current();
    if deadline.expired() {
        return Ok((docs, scores));
    }
    let Some(mut s) = query.scorer(leaf, ctx) else { return Ok((docs, scores)) };
    let n = leaf.num_docs();
    if min_score.is_none() && n > 0 && query.is_match_all() && leaf.deleted_bits().is_none() {
        let first = s.advance();
        if first == 0 {
            let score = s.score();
            // Charged as the per-doc loop charges (8 bytes a doc), a page of
            // docs at a time.
            let page = (crate::memory::PAGE / 8) as u32;
            let mut at = 0u32;
            while at < n {
                let k = page.min(n - at);
                mem.add(8 * k as u64, "<agg [_docs]>")?;
                at += k;
            }
            docs = (0..n).collect();
            scores = vec![score; n as usize];
            return Ok((docs, scores));
        }
        // Not the expected first doc: collect generically from a fresh scorer.
        let Some(fresh) = query.scorer(leaf, ctx) else { return Ok((docs, scores)) };
        s = fresh;
    }
    // A constant-score set held whole (a geo / points query's bitset or
    // doc list): its live docs appended in bulk, all with its one score,
    // instead of advancing and scoring per doc. Same docs and scores.
    let deleted = leaf.deleted_bits();
    if let Some(score) = s.collect_all(deleted.as_ref().map(|b| b.words()), &mut docs) {
        if min_score.is_some_and(|m| !(score >= m)) {
            docs.clear();
            return Ok((docs, scores));
        }
        // Charged as the per-doc loop charges (8 bytes a doc).
        let page = (crate::memory::PAGE / 8) as usize;
        let mut at = 0usize;
        while at < docs.len() {
            let k = page.min(docs.len() - at);
            mem.add(8 * k as u64, "<agg [_docs]>")?;
            at += k;
        }
        scores = vec![score; docs.len()];
        return Ok((docs, scores));
    }
    let live = leaf.live_docs();
    // The `timeout` and cancellation polls share one check every 1024 docs.
    let cancel = rustic_core::cancel::current();
    let polled = deadline.is_set() || cancel.is_some();
    let mut poll = rustic_core::cancel::Poll::new();
    let mut iter = 0u64;
    loop {
        let d = s.advance();
        if d == rustic_core::query::TERMINATED {
            break;
        }
        iter += 1;
        if polled && poll.due(iter) {
            if cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(task_cancelled());
            }
            if deadline.expired() {
                break;
            }
        }
        if live(d) {
            let sc = s.score();
            if min_score.map(|m| sc >= m).unwrap_or(true) {
                mem.add(8, "<agg [_docs]>")?;
                docs.push(d);
                scores.push(sc);
            }
        }
    }
    Ok((docs, scores))
}

// A union fallback failed all still-active runs. Earlier query failures must
// remain in the envelope, including when only one aggregation run remained.
fn union_aggregation_failure<'a>(indices: impl IntoIterator<Item=&'a str>, prior: &[(String, EsError)], error: EsError) -> EsError {
    let mut failures = prior.to_vec();
    let mut pending = std::collections::HashMap::<&str, usize>::new();
    for (index, _) in prior { *pending.entry(index.as_str()).or_default() += 1; }
    for index in indices {
        if let Some(count) = pending.get_mut(index) {
            if *count > 0 { *count -= 1; continue; }
        }
        failures.push((index.to_string(), error.clone()));
    }
    EsError::search_phase_all(&failures)
}

/// The request of a `top_hits` aggregation's hit rendering.
fn agg_hit_request(spec: &Value) -> Result<SearchRequest> {
    let mut sub = SearchRequest::default();
    if let Some(s) = spec.get("_source") {
        sub.source = parse_source_filter(s)?;
    }
    sub.version = spec.get("version").and_then(Value::as_bool).unwrap_or(false);
    sub.seq_no_primary_term = spec.get("seq_no_primary_term").and_then(Value::as_bool).unwrap_or(false);
    if let Some(f) = spec.get("fields") {
        sub.fields = field_list(f);
    }
    if let Some(f) = spec.get("docvalue_fields") {
        sub.docvalue_fields = field_list(f);
    }
    Ok(sub)
}

/// Whether every field the aggregations name has one plain type in every
/// target: the shards of several single-shard indices (a data stream's
/// backing indices) then reduce per shard as ES's do. Pre-aggregated
/// fields (aggregate_metric_double, histogram, tdigest,
/// exponential_histogram), a field typed differently across the indices or
/// missing from some of them keep the one-pass aggregation over every
/// index, which reads each index's field as its own type (the per-shard
/// partials of such mixes did not reduce right: averages over
/// aggregate_metric_double + histograms, percentiles over tdigest +
/// exponential_histogram).
fn aggs_fields_uniform(spec: Option<&Value>, targets: &[Target]) -> bool {
    fn fields<'v>(v: &'v Value, out: &mut Vec<&'v str>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    if k == "field" {
                        if let Some(f) = x.as_str() { out.push(f); }
                    } else {
                        fields(x, out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| fields(x, out)),
            _ => {}
        }
    }
    let Some(spec) = spec else { return true };
    let mut names = Vec::new();
    fields(spec, &mut names);
    let pre_aggregated = |t: EsType| matches!(t, EsType::AggregateMetricDouble | EsType::Histogram | EsType::TDigest | EsType::ExponentialHistogram);
    names.iter().all(|name| {
        let mut seen: Option<(EsType, EsType)> = None;
        targets.iter().all(|t| match t.lookup().get(name) {
            Some(rf) => {
                let ty = (rf.ty, rf.declared);
                !pre_aggregated(rf.ty) && !pre_aggregated(rf.declared) && *seen.get_or_insert(ty) == ty
            }
            None => false,
        })
    })
}

/// Aggregations over several shard runs: one partial per shard
/// (`shard_size` buckets, mergeable metric state), reduced like ES's
/// coordinator. `None` when an aggregation of the request has no
/// cross-shard reduce yet (the caller aggregates over the union instead).
/// A shard whose aggregation trips a circuit breaker fails alone when
/// others succeed (ES reports it in
/// `_shards.failures` and returns the others' results); when all do, the
/// search fails with 429.
#[allow(clippy::too_many_arguments)]
fn shard_reduce_aggs(targets: &[Target], runs: &[IndexRun], req: &SearchRequest, spec: &Value, now: i64, max_buckets: usize, shard_failures: &mut Vec<(String, EsError)>) -> Result<Option<Value>> {
    let first = targets.first().map(|t| t.svc.name.clone()).unwrap_or_default();
    let wrap = |e: EsError| if crate::memory::is_breaking(&e) && e.kind != "search_phase_execution_exception" { EsError::search_phase(&first, e) } else if e.kind == "parsing_exception" || e.kind == "x_content_parse_exception" || e.kind == "action_request_validation_exception" || e.kind == "search_phase_execution_exception" || e.status != 400 { e } else { EsError::search_phase(&first, e) };
    let failed_before = shard_failures.len();
    let request = crate::hooks::agg_partial::AggRequest { aggs: spec.clone(), typed_keys: req.typed_keys, max_buckets, now };
    // Each shard's partial, on the search pool when there are several (ES
    // runs the shards concurrently), in this request's context (breaker
    // scope and admission policy, coordinator, inference service, read
    // shard selection, script environment and terms lookup); outcomes are
    // handled in shard order, as the sequential loop did.
    enum ShardOut { Part(crate::hooks::agg_partial::AggPartial), Tripped(EsError), Fatal(EsError), Raw(EsError) }
    let one = |ti: usize| -> ShardOut {
        let run = &runs[ti];
        let t = &targets[ti];
        let ctx = SearchContext::for_query(&run.query, &run.leaves, Bm25Params::default());
        if let Err(e) = ctx.execution_result() { return ShardOut::Raw(e.into()); }
        // This shard's doc sets, charged while it aggregates (freed, and
        // released, once its partial is built).
        let shard_mem = crate::memory::RequestMemory::new();
        let mut segs: Vec<AggSeg> = Vec::with_capacity(run.leaves.len());
        for leaf in &run.leaves {
            let (docs, scores) = match agg_leaf_docs(&run.query, leaf, &ctx, req.min_score, &shard_mem) {
                Ok(found) => found,
                Err(e) => return ShardOut::Tripped(e),
            };
            segs.push(AggSeg { leaf: *leaf, lookup: &run.lookup, index: &t.svc.name, max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000), docs, scores });
        }
        let fetch = |si: usize, doc: DocId, score: f32, spec: &Value| -> Result<Value> {
            let sub = agg_hit_request(spec)?;
            let th = TopHit::new(usize::MAX, doc, score, vec![]);
            let s = fetch_hit(t, run, &segs[si].leaf, doc, Some(&th), &sub, true)?;
            hit_value(&s)
        };
        let visibility_error=parking_lot::Mutex::new(None);
        let all = |si: usize| -> Vec<DocId> {
            match run.visible_docs(&segs[si].leaf) {Ok(docs)=>docs,Err(error)=>{*visibility_error.lock()=Some(error);Vec::new()}}
        };
        let source = |si: usize, docs: &[(DocId, f32)], paths: &[String], emit: &mut dyn FnMut(DocId, Option<Value>) -> Result<()>| {
            fetch_aggregation_sources(t, run, &segs[si].leaf, docs, paths, emit)
        };
        let mapping_context = [(t.svc.name.as_str(), run.lookup.as_ref())];
        let env = AggEnv { mapping_context: &mapping_context, now, max_buckets, fetch_hit: &fetch, fetch_source: Some(&source), all_docs: &all, background_filtered: run.visibility_query.is_some(), top_metrics_max_size: t.svc.setting_usize("index.top_metrics_max_size", 10), max_result_window: t.svc.setting_usize("index.max_result_window", 10_000), profile: req.profile };
        let ranges = vec![run.query.required_i64_range(); segs.len()];
        let partial=aggs::with_query_ranges(ranges, || crate::hooks::agg_partial::shard_partial(spec,&segs,&env,&request,true));
        if let Err(e) = run.visibility_result() { return ShardOut::Raw(e); }
        let partial=match visibility_error.lock().take() {Some(error)=>Err(error),None=>partial};
        match partial {
            Ok(p) => ShardOut::Part(p),
            Err(e) if crate::memory::is_breaking(&e) => ShardOut::Tripped(e),
            Err(e) => ShardOut::Fatal(e),
        }
    };
    let outs: Vec<ShardOut> = if runs.len() > 1 {
        let request_ctx = crate::memory::RequestContext::capture();
        let coordinator = crate::coord::current();
        let inference = crate::inference::current();
        let read_shard = selected_read_shard();
        let read_count = READ_SHARD_COUNT.with(|s| s.get());
        // The search's `timeout` reaches each shard's collection (the task
        // token travels with `par_map_search`).
        let agg_deadline = AggDeadline::current();
        let lookup = crate::script::terms_lookup_ptr();
        let envs: Vec<parking_lot::Mutex<Option<crate::script::ScriptEnv>>> = runs.iter().map(|_| parking_lot::Mutex::new(crate::script::env().map(|e| e.fork()))).collect();
        rustic_core::query::searcher::par_map_search(runs.len(), |ti| {
            request_ctx.run(|| crate::coord::scoped(coordinator.clone(), || crate::inference::scoped(inference.clone(), || {
                let _shard = ReadShardScope::enter(read_shard);
                let _count = ReadShardCountScope::enter(read_count);
                let _deadline = agg_deadline.enter();
                crate::script::with_terms_lookup_ptr(lookup, || match envs[ti].lock().take() {
                    Some(env) => crate::script::with_env(env, || one(ti)),
                    None => one(ti),
                })
            })))
        })
    } else {
        (0..runs.len()).map(one).collect()
    };
    let mut parts = Vec::with_capacity(runs.len());
    for (ti, out) in outs.into_iter().enumerate() {
        match out {
            ShardOut::Part(p) => parts.push(p),
            ShardOut::Tripped(e) if runs.len() > 1 => shard_failures.push((targets[ti].svc.name.clone(), e)),
            ShardOut::Tripped(e) | ShardOut::Fatal(e) => return Err(wrap(e)),
            ShardOut::Raw(e) => return Err(e),
        }
    }
    // Every target failed (query-phase failures included: their runs match
    // nothing but still produce empty partials).
    if shard_failures.len() > failed_before && (parts.is_empty() || shard_failures.len() >= targets.len()) {
        return Err(EsError::search_phase_all(&shard_failures));
    }
    match crate::hooks::agg_partial::AggPartial::reduce(parts, &request) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.reason.contains("does not support a cross-shard reduce") && !aggs::significant_text::present(spec) => Ok(None),
        Err(e) => Err(wrap(e)),
    }
}

fn add_typed_keys(out: Value, spec: &Value) -> Value {
    let Value::Object(m) = out else { return out };
    let specs = spec.as_object();
    let mut res = Map::new();
    for (k, v) in m {
        let def = specs.and_then(|s| s.get(&k)).and_then(Value::as_object);
        let ty = def.and_then(|d| d.keys().find(|x| !matches!(x.as_str(), "aggs" | "aggregations" | "meta")).cloned()).unwrap_or_default();
        let prefix = match ty.as_str() {
            "terms" => {
                let key_num = v.get("buckets").and_then(Value::as_array).and_then(|b| b.first()).map(|b| b["key"].is_number());
                match key_num {
                    Some(true) => if v.get("buckets").and_then(|b| b[0]["key"].as_i64()).is_some() { "lterms" } else { "dterms" },
                    _ => "sterms",
                }
            }
            "avg" | "sum" | "min" | "max" | "value_count" | "cardinality" | "stats" | "extended_stats" | "top_hits" | "filter" | "filters" | "range" | "histogram" | "global" | "missing" | "composite" => ty.as_str(),
            "date_histogram" => "date_histogram",
            "date_range" => "date_range",
            "percentiles" | "percentile_ranks" => {
                let hdr = def.and_then(|d| d.get(&ty)).and_then(|b| b.get("hdr")).is_some();
                match (ty.as_str(), hdr) {
                    ("percentiles", true) => "hdr_percentiles",
                    ("percentiles", false) => "tdigest_percentiles",
                    (_, true) => "hdr_percentile_ranks",
                    _ => "tdigest_percentile_ranks",
                }
            }
            "diversified_sampler" => "sampler",
            "significant_text" => "sigsterms",
            "cumulative_sum" | "bucket_script" | "serial_diff" | "moving_fn" | "moving_avg" | "normalize" | "avg_bucket" | "sum_bucket" => "simple_value",
            "max_bucket" | "min_bucket" => "bucket_metric_value",
            "cumulative_cardinality" => "simple_long_value",
            "rare_terms" | "significant_terms" => {
                let num = v.get("buckets").and_then(Value::as_array).and_then(|b| b.first()).map(|b| b["key"].is_number()).unwrap_or(false);
                match (ty.as_str(), num) {
                    ("rare_terms", true) => "lrareterms",
                    ("rare_terms", false) => "srareterms",
                    (_, true) => "siglterms",
                    _ => "sigsterms",
                }
            }
            _ => ty.as_str(),
        };
        let sub_spec = def.and_then(|d| d.get("aggs").or(d.get("aggregations"))).cloned();
        let v = match (v, sub_spec) {
            (Value::Object(mut o), Some(ss)) => {
                if let Some(Value::Array(bs)) = o.get_mut("buckets") {
                    for b in bs.iter_mut() {
                        *b = retype_bucket(b.take(), &ss);
                    }
                } else if let Some(Value::Object(bs)) = o.get_mut("buckets") {
                    for b in bs.values_mut() {
                        *b = retype_bucket(b.take(), &ss);
                    }
                } else {
                    let tmp = retype_bucket(Value::Object(o), &ss);
                    o = tmp.as_object().cloned().unwrap_or_default();
                }
                Value::Object(o)
            }
            (v, _) => v,
        };
        res.insert(format!("{prefix}#{k}"), v);
    }
    Value::Object(res)
}

fn retype_bucket(b: Value, sub: &Value) -> Value {
    let Value::Object(m) = b else { return b };
    let names: Vec<String> = sub.as_object().map(|s| s.keys().cloned().collect()).unwrap_or_default();
    let mut aggs = Map::new();
    let mut rest = Map::new();
    for (k, v) in m {
        if names.contains(&k) {
            aggs.insert(k, v);
        } else {
            rest.insert(k, v);
        }
    }
    if let Value::Object(t) = add_typed_keys(Value::Object(aggs), sub) {
        rest.extend(t);
    }
    Value::Object(rest)
}

/// Fetch only the ID from doc values when the response cannot need stored data.
/// Routing and ignored metadata are emitted even with `_source:false`, so a
/// leaf containing either must retain the normal fetch path. Both reserved
/// fields are indexed keywords; their collection statistics include deleted
/// docs, making the leaf-wide check conservative across refreshes and merges.
/// Whether a hit of `req` is rendered from its stored document
/// ([`fetch_document`]); `docvalue_fields` read columns, not the stored
/// document.
fn fetch_needs_stored(req: &SearchRequest) -> bool {
    !matches!(req.source, SourceFilter::None)
        || req.version || req.seq_no_primary_term || req.force_synthetic_source
        || !req.fields.is_empty() || !req.stored_fields.is_empty()
        || req.highlight.is_some()
        || req.script_fields.is_some() || req.runtime_mappings.is_some()
}

/// `pre`: the stored document, read already (a page's batch read).
fn fetch_document(leaf: &LeafReader<'_>, doc: DocId, req: &SearchRequest, pre: Option<Document>) -> Result<Document> {
    if !fetch_needs_stored(req) {
        let present = |field: u16| {
            let stats = match leaf {
                LeafReader::Segment(s) => s.field_agg(field),
                LeafReader::Memtable(w, ..) => w.field_agg(field),
            };
            stats.is_some_and(|a| a.doc_count > 0)
        };
        // `_routing` is stored only; `_ignored` also has a column.
        if !present(ROUTING_FIELD) {
            // The page's batch read the stored `_id` already ([`execute`]).
            let id = match &pre {
                Some(document) => document.get(ID_FIELD).and_then(|v| v.as_text()).map(str::to_string).or_else(|| leaf.keyword_value(ID_FIELD, doc)),
                None => leaf.keyword_value(ID_FIELD, doc),
            };
            if let Some(id) = id {
                let mut result = Document::new();
                result.add_keyword(ID_FIELD, &id);
                if present(IGNORED_FIELD) {
                    for f in leaf.keyword_values(IGNORED_FIELD, doc) {
                        result.add_keyword(IGNORED_FIELD, &f);
                    }
                }
                return Ok(result);
            }
        }
    }
    if let Some(document) = pre { return Ok(document); }
    Ok(leaf.get_doc(doc)?.unwrap_or_default())
}

/// Significant text needs source values, not a rendered search hit. Keep the
/// general fetch path for vector rehydration, and enforce FLS before selecting
/// any source fields. The ordinary path avoids hit metadata, request parsing,
/// source serialization and reparsing the rendered hit for every document.
fn fetch_aggregation_sources(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, docs: &[(DocId, f32)], paths: &[String], emit: &mut dyn FnMut(DocId, Option<Value>) -> Result<()>) -> Result<()> {
    if crate::vector_source::has_vector_fields(&run.lookup) {
        let req = SearchRequest { source: SourceFilter::Filter { includes: paths.to_vec(), excludes: vec![] }, ..Default::default() };
        for &(doc, _) in docs {
            let hit: Value = serde_json::from_str(&fetch_hit(t, run, leaf, doc, None, &req, true)?)?;
            emit(doc, hit.get("_source").cloned())?;
        }
        return Ok(());
    }
    // Only IDs are batched. Each document/source is consumed immediately,
    // bounding scratch space independently of the number of matched documents.
    for batch in docs.chunks(1024) {
        let ids: Vec<_> = batch.iter().map(|&(doc, _)| doc).collect();
        let mut error = None;
        leaf.try_for_each_doc(&ids, |doc, stored| {
            if error.is_some() { return; }
            let result = (|| {
                let raw = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).unwrap_or("");
                let loaded = if crate::synthetic::is_residual(raw) { crate::vector_source::load(raw, &t.svc.lookup(), leaf, doc) } else { std::borrow::Cow::Borrowed(raw) };
                let raw: &str = &loaded;
                let source = if raw.is_empty() { None } else {
                    let visible = t.visible_source(raw);
                    Some(aggs::significant_text::select_source(&visible, paths)?)
                };
                emit(doc, source)
            })();
            if let Err(e) = result { error = Some(e); }
        })?;
        if let Some(e) = error { return Err(e); }
    }
    Ok(())
}

/// Render one hit (fetch phase).
/// A rendered `top_hits` hit as the aggregation tree's `Value`. One `Value`
/// cannot hold (a lone UTF-16 surrogate in its source, kept raw) becomes
/// its other members plus its whole raw text (`__p.raw_json`): reduces read
/// its `_score` / `sort` as usual, and it is rendered verbatim at the end
/// (`aggs::render_scripted_values`, also on the coordinator).
pub(crate) fn hit_value(s: &str) -> Result<Value> {
    if let Ok(v) = serde_json::from_str(s) {
        return Ok(v);
    }
    let mut m = serde_json::Map::new();
    if let crate::source_json::Source::Object(o) = crate::source_json::Source::parse(s)? {
        for (k, v) in o.iter().filter(|(k, _)| k.as_str() != "_source") {
            if let Ok(x) = serde_json::from_str::<Value>(&v.text()) {
                m.insert(k.clone(), x);
            }
        }
    }
    m.insert("__p".into(), json!({ "raw_json": s }));
    Ok(Value::Object(m))
}

/// A [`SearchRequest::columns_only`] hit ([`ColumnHit`]): the `_id` and
/// the requested columns' doc values, nothing rendered (no metadata checks,
/// source, vector, script or sort handling).
fn column_hit(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, doc: DocId, req: &SearchRequest) -> Result<ColumnHit> {
    let id = match leaf.keyword_value(ID_FIELD, doc) {
        Some(id) => id,
        None => fetch_document(leaf, doc, req, None)?.get(ID_FIELD).and_then(|v| v.as_text()).unwrap_or("").to_string(),
    };
    let mut values = Vec::with_capacity(req.docvalue_fields.len());
    for (name, _) in &req.docvalue_fields {
        values.push(match run.lookup.get(name) {
            Some(f) if f.ty == EsType::Date => match f.core_id {
                Some(cid) => {
                    let mut v = leaf.i64_values(cid, doc);
                    v.sort_unstable();
                    v.into_iter().map(Value::from).collect()
                }
                None => Vec::new(),
            },
            _ => docvalue_values(leaf, &run.lookup, name, doc, None)?,
        });
    }
    Ok(ColumnHit { index: t.hit_index().to_string(), id, values })
}

/// A hit's `"sort"` member (when the search sorts), as [`fetch_hit`]
/// renders it.
fn push_hit_sort(s: &mut String, run: &IndexRun, h: &TopHit, req: &SearchRequest) -> Result<()> {
    if !req.sort.is_empty() {
        let vals: Vec<Value> = run
            .sort
            .iter()
            .enumerate()
            .map(|(i, sp)| {
                if sp.render == "doc" {
                    let local = ((h.leaf as i64) << 32) | h.doc as i64;
                    // `_shard_doc` also carries the shard's position in the request.
                    Value::from(if req.sort.get(i).is_some_and(|s| s.field == "_shard_doc") { (run.shard_ord << SHARD_DOC_SHIFT) | local } else { local })
                } else if sp.render == "geo" {
                    match h.sort.get(i) {
                        Some(SortValue::F64(m)) if m.is_finite() => values::java_double(m / sp.scale),
                        _ => json!("Infinity"),
                    }
                } else {
                    render_sort_value(h.sort.get(i).unwrap_or(&SortValue::Missing), sp.render, &run.lookup, req.sort.get(i), h.score, sp.scale > 1.5)
                }
            })
            .collect();
        s.push_str(",\"sort\":");
        s.push_str(&serde_json::to_string(&vals)?);
    }
    Ok(())
}

/// A query phase hit of [`SearchRequest::light_refs`]: `_index`, `_score`
/// and `sort` as [`fetch_hit`] renders them, nothing read from the
/// document.
fn query_phase_hit(t: &Target, run: &IndexRun, h: &TopHit, req: &SearchRequest, scores_shown: bool) -> Result<String> {
    let mut s = String::with_capacity(96);
    s.push_str("{\"_index\":");
    s.push_str(&serde_json::to_string(t.hit_index())?);
    if scores_shown {
        s.push_str(",\"_score\":");
        s.push_str(&crate::retriever::score_json(h.score));
    } else {
        s.push_str(",\"_score\":null");
    }
    push_hit_sort(&mut s, run, h, req)?;
    s.push('}');
    Ok(s)
}

fn fetch_hit(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, doc: DocId, h: Option<&TopHit>, req: &SearchRequest, scores_shown: bool) -> Result<String> {
    fetch_hit_read(t, run, leaf, doc, h, req, scores_shown, None, None)
}

/// Whether `_source` of `run`'s hits for `req` leaves the vectors out
/// (`exclude_source_vectors`), as [`fetch_hit`] decides per hit.
fn hit_excludes_vectors(t: &Target, run: &IndexRun, req: &SearchRequest) -> bool {
    // The mapping check first: an index without vector fields reads no
    // index settings (a lock and a copy each).
    !req.force_synthetic_source && crate::vector_source::has_vector_fields(&run.lookup) && crate::vector_source::exclude_for(req.exclude_vectors.or(req.vectors_by_default.then_some(false)), crate::vector_source::excludes_source_vectors(|k| t.svc.setting(k)))
}

/// Hits a page needs before they render in parallel (see the fetch phase).
const PARALLEL_FETCH_MIN: usize = 256;
/// Hits one search-pool task renders.
const PARALLEL_FETCH_CHUNK: usize = 64;
/// Consecutive hits of a leaf one search-pool task reads the stored
/// documents of, on a large page.
const PARALLEL_READ_CHUNK: usize = 256;

/// [`fetch_hit`] of a hit whose stored document was read already (`pre`),
/// and whose `exclude_vectors` the caller decided for its page.
#[allow(clippy::too_many_arguments)]
fn fetch_hit_read(t: &Target, run: &IndexRun, leaf: &LeafReader<'_>, doc: DocId, h: Option<&TopHit>, req: &SearchRequest, scores_shown: bool, pre: Option<Document>, exclude_vectors: Option<bool>) -> Result<String> {
    let stored = fetch_document(leaf, doc, req, pre)?;
    let id = stored.get(ID_FIELD).and_then(|v| v.as_text()).unwrap_or("").to_string();
    let raw_src = stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).unwrap_or("");
    // `exclude_source_vectors`: vectors stay out of `_source` unless asked for.
    // (`force_synthetic_source` loads every field, vectors included.)
    // Per hit: the mapping check first, so an index without vector fields
    // reads no index settings (a lock and a copy each) for every hit.
    let exclude_vectors = exclude_vectors.unwrap_or_else(|| hit_excludes_vectors(t, run, req));
    // The vectors the stored `_source` leaves to the vector store come back
    // unless the hit's `_source` is all there is to load and leaves them out
    // anyway (their marked keys are dropped like the other vector fields).
    let loaded;
    // `fields` loads them only when it can return one of them.
    let raw_src: &str = if crate::synthetic::is_residual(raw_src) || !raw_src.is_empty() && (!exclude_vectors || crate::vector_source::fields_load_vectors(&run.lookup, &req.fields) || !req.stored_fields.is_empty() || req.highlight.is_some() || matches!(req.source, SourceFilter::Filter { .. }) || req.force_synthetic_source) {
        loaded = crate::vector_source::load(raw_src, &run.lookup, leaf, doc);
        &loaded
    } else {
        raw_src
    };
    // The whole loaded source (the folded `_ignored_source` rebuilds from it).
    let full_src: &str = raw_src;
    // Field-level security: everything below sees only the permitted part of `_source`.
    let visible = t.visible_source(raw_src);
    let raw_src: &str = &visible;
    let forced = if req.force_synthetic_source && !raw_src.is_empty() { crate::synthetic::force(&t.svc, raw_src) } else { None };
    let raw_src: &str = forced.as_deref().unwrap_or(raw_src);
    let mut s = String::with_capacity(raw_src.len() + 128);
    s.push_str("{\"_index\":");
    s.push_str(&serde_json::to_string(t.hit_index())?);
    if !req.stored_fields_none {
        s.push_str(",\"_id\":");
        s.push_str(&serde_json::to_string(&id)?);
    }
    if req.version {
        s.push_str(&format!(",\"_version\":{}", stored.get(VERSION_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)));
    }
    if req.seq_no_primary_term {
        if t.svc.seq_no_disabled() {
            s.push_str(",\"_seq_no\":-2,\"_primary_term\":0");
        } else {
            s.push_str(&format!(
                ",\"_seq_no\":{},\"_primary_term\":{}",
                stored.get(SEQ_NO_FIELD).and_then(|v| v.as_i64()).unwrap_or(0),
                stored.get(PRIMARY_TERM_FIELD).and_then(|v| v.as_i64()).unwrap_or(1)
            ));
        }
    }
    let score = h.map(|h| h.score).unwrap_or(1.0);
    if scores_shown {
        s.push_str(",\"_score\":");
        // ES scores are Java floats: shortest f32 representation.
        s.push_str(&crate::retriever::score_json(score));
    } else {
        s.push_str(",\"_score\":null");
    }
    if let Some(r) = stored.get(ROUTING_FIELD).and_then(|v| v.as_text()) {
        s.push_str(",\"_routing\":");
        s.push_str(&serde_json::to_string(r)?);
    }
    let mut ignored: Vec<&str> = stored.get_all(IGNORED_FIELD).filter_map(|v| v.as_text()).collect();
    ignored.sort_unstable();
    ignored.dedup();
    if !ignored.is_empty() {
        s.push_str(",\"_ignored\":");
        s.push_str(&serde_json::to_string(&ignored)?);
    }
    // `_ignored_source` is only returned when explicitly requested (by name in
    // `fields` or `stored_fields`).
    // One binary value per ignored field (`mapper.ignored_source_as_top_level_metadata_array_field`).
    let want_ignored_source = req.fields.iter().any(|(p, _)| p == "_ignored_source") || req.stored_fields.iter().any(|p| p == "_ignored_source");
    if want_ignored_source {
        let mut entries: Vec<&str> = stored.get_all(IGNORED_SOURCE_FIELD).filter_map(|v| v.as_text()).collect();
        // A residual that folded its entries into its own leaves.
        let folded = if entries.is_empty() {
            let format = crate::synthetic::IgnoredSourceFormat::of(run.lookup.registry.created_version, run.lookup.registry.ts_doc_values_format);
            stored.get(SOURCE_FIELD).and_then(|v| v.as_text()).and_then(|raw| crate::synthetic::folded_ignored_source(raw, full_src, &run.lookup, format))
        } else {
            None
        };
        if let Some(f) = &folded {
            entries = f.iter().map(String::as_str).collect();
        }
        if run.lookup.registry.ts_doc_values_format && run.lookup.registry.created_version.on_or_after(crate::version::IndexVersion::IGNORED_SOURCE_AS_DOC_VALUES_NO_FF) {
            // ES returns binary doc values in encoded-byte order, whereas
            // legacy stored fields retain insertion order.
            use base64::Engine;
            entries.sort_by_cached_key(|s| base64::engine::general_purpose::STANDARD.decode(s).unwrap_or_default());
        }
        if !entries.is_empty() {
            s.push_str(",\"_ignored_source\":");
            s.push_str(&serde_json::to_string(&entries)?);
        }
    }
    // `fields: [_inference_fields]`: the semantic_text inference metadata, in `_source`.
    // Legacy semantic_text indices keep the inference results in `_source`.
    let legacy = || t.svc.setting("index.mapping.semantic_text.use_legacy_format").as_deref() == Some("true");
    let wanted = (req.fields.iter().any(|(p, _)| p == "_inference_fields") || crate::inference::semantic::rehydrate_inference_fields(&req.source, req.exclude_vectors)) && !legacy();
    let needs_parsed = wanted || !req.fields.is_empty() || !req.stored_fields.is_empty() || req.highlight.is_some() || matches!(req.source, SourceFilter::Filter { .. });
    let parsed: Option<Value> = if needs_parsed && !raw_src.is_empty() { serde_json::from_str(raw_src).ok() } else { None };
    let inference = if wanted { crate::inference::semantic::inference_fields(&run.lookup, leaf, doc, parsed.as_ref()) } else { None };
    let with_inference = |mut v: Value| {
        if let (Some(inf), Value::Object(m)) = (&inference, &mut v) {
            m.insert("_inference_fields".into(), inf.clone());
        }
        v
    };
    if !raw_src.is_empty() {
        match &req.source {
            SourceFilter::All if !exclude_vectors && inference.is_some() => {
                if let Some(p) = &parsed {
                    s.push_str(",\"_source\":");
                    s.push_str(&serde_json::to_string(&with_inference(p.clone()))?);
                }
            }
            SourceFilter::All if !exclude_vectors => {
                s.push_str(",\"_source\":");
                s.push_str(raw_src);
            }
            SourceFilter::None => {}
            SourceFilter::All if inference.is_some() => {
                if let Some(p) = &parsed {
                    let mut out = p.clone();
                    crate::vector_source::strip(&mut out, &run.lookup, &|_| false);
                    s.push_str(",\"_source\":");
                    s.push_str(&serde_json::to_string(&with_inference(out))?);
                }
            }
            // Vectors left out without parsing the rest of the source.
            SourceFilter::All => {
                s.push_str(",\"_source\":");
                match crate::vector_source::strip_source(raw_src, &run.lookup, &|_| false) {
                    Some(out) => s.push_str(&out),
                    None => s.push_str(raw_src),
                }
            }
            SourceFilter::Filter { includes, excludes } => {
                if let Some(p) = &parsed {
                    let mut out = filter_source(p, includes, excludes);
                    if exclude_vectors {
                        crate::vector_source::strip(&mut out, &run.lookup, &|f| crate::vector_source::explicitly_included(includes, f));
                    }
                    let out = if includes.is_empty() && excludes.is_empty() { out } else { crate::vector_source::finish_filtered(out, &run.lookup) };
                    s.push_str(",\"_source\":");
                    s.push_str(&serde_json::to_string(&with_inference(out))?);
                } else {
                    let source = if exclude_vectors {
                        crate::vector_source::strip_source(raw_src, &run.lookup, &|f| crate::vector_source::explicitly_included(includes, f)).unwrap_or_else(|| raw_src.to_owned())
                    } else { raw_src.to_owned() };
                    s.push_str(",\"_source\":");
                    s.push_str(&crate::source_json::filter_raw(&source, includes, excludes)?);
                }
            }
        }
    }
    let mut fields = Map::new();
    if !req.fields.is_empty() || !req.stored_fields.is_empty() {
        let empty = json!({});
        let src = parsed.as_ref().unwrap_or(&empty);
        // Index-time script fields are not in `_source`: fetched from doc values.
        let script_fields: Vec<(String, Option<String>)> = run
            .lookup
            .fields
            .values()
            .filter(|f| f.script.is_some())
            .filter_map(|f| req.fields.iter().find(|(p, _)| crate::mapping::wildcard_match(p, &f.name)).map(|(_, fmt)| (f.name.clone(), fmt.clone())))
            .collect();
        let mut meta: HashMap<&str, Vec<Value>> = HashMap::new();
        for (n, fmt) in &script_fields {
            let v = docvalue_values(leaf, &run.lookup, n, doc, fmt.as_deref())?;
            if !v.is_empty() {
                meta.insert(n.as_str(), v);
            }
        }
        // `semantic_text` fields fetched with `format: chunks` (9.2+): the
        // texts of their chunks, read back from the inference blob.
        let chunk_fields: Vec<String> = run
            .lookup
            .fields
            .values()
            .filter(|f| f.declared.is_semantic() && req.fields.iter().any(|(p, fmt)| fmt.as_deref() == Some("chunks") && crate::mapping::wildcard_match(p, &f.name)))
            .map(|f| f.name.clone())
            .collect();
        for n in &chunk_fields {
            let v = crate::inference::semantic::chunk_texts(&run.lookup, leaf, doc, n, src);
            if let Some(k) = run.lookup.fields.get_key_value(n.as_str()).map(|(k, _)| k.as_str()) {
                if !v.is_empty() {
                    meta.insert(k, v);
                }
            }
        }
        meta.insert("_id", vec![Value::String(id.clone())]);
        meta.insert("_index", vec![Value::String(t.svc.name.clone())]);
        meta.insert("_version", vec![json!(stored.get(VERSION_FIELD).and_then(|v| v.as_i64()).unwrap_or(1))]);
        if !ignored.is_empty() {
            meta.insert("_ignored", ignored.iter().map(|x| Value::String(x.to_string())).collect());
        }
        if req.fields.iter().any(|(p, _)| p == crate::tsdb::TSID_FIELD) {
            let m = t.svc.meta();
            if crate::tsdb::is_time_series(&m.settings) {
                if let Ok(ids) = crate::tsdb::build_ids(&m.mappings, &m.settings, src) {
                    meta.insert(crate::tsdb::TSID_FIELD, vec![Value::String(ids.tsid)]);
                }
            }
        }
        let c = FieldsCtx { lookup: &run.lookup, pats: &req.fields, unmapped: &req.fields_unmapped, meta: &meta };
        for (pat, _) in &req.fields {
            if let Some(v) = meta.get(pat.as_str()) {
                fields.insert(pat.clone(), Value::Array(v.clone()));
            }
        }
        fetch_level(&c, src, "", &mut fields)?;
        let stored_blob = crate::stored::from_doc(&stored);
        for pat in &req.stored_fields {
            for n in run.lookup.matching(pat) {
                if fields.contains_key(&n) || !run.lookup.get(&n).is_some_and(|f| f.stored) {
                    continue;
                }
                // ES returns no stored value for a sparse vector its index
                // keeps out of `_source` (served from term vectors).
                if run.lookup.registry.exclude_source_vectors && run.lookup.get(&n).is_some_and(|f| f.declared == EsType::SparseVector && crate::vector_source::is_store_backed(f)) {
                    continue;
                }
                // unsigned_long is stored as its decimal string.
                let ul = run.lookup.get(&n).is_some_and(|f| f.ty == EsType::UnsignedLong);
                let stored_form = |v: Vec<Value>| -> Value {
                    Value::Array(if ul { v.into_iter().map(|x| if let Value::Number(num) = &x { Value::String(num.to_string()) } else { x }).collect() } else { v })
                };
                // The stored-fields block (present even when `_source` is not).
                if let Some(Value::Array(a)) = stored_blob.get(&n) {
                    if !a.is_empty() {
                        fields.insert(n, stored_form(a.clone()));
                    }
                    continue;
                }
                let path = run.lookup.get(&n).map(|f| if f.is_multi_field { n.rsplit_once('.').map(|x| x.0).unwrap_or(&n).to_string() } else { n.clone() }).unwrap_or_else(|| n.clone());
                let v = fields_api_values(&run.lookup, &n, &path, src, None);
                if !v.is_empty() {
                    fields.insert(n, stored_form(v));
                }
            }
        }
    }
    for (pat, fmt) in &req.docvalue_fields {
        let names = if pat.contains('*') { run.lookup.matching(pat) } else { vec![pat.clone()] };
        for n in names {
            let v = docvalue_values(leaf, &run.lookup, &n, doc, fmt.as_deref())?;
            if !v.is_empty() {
                fields.insert(n, Value::Array(v));
            }
        }
    }
    if let Some(cf) = &req.collapse {
        if !fields.contains_key(cf) {
            let v = docvalue_values(leaf, &run.lookup, cf, doc, None)?;
            fields.insert(cf.clone(), if v.is_empty() { json!([null]) } else { Value::Array(v.into_iter().take(1).collect()) });
        }
    }
    let mut raw_fields = std::collections::BTreeMap::new();
    if req.script_fields.is_some() || crate::script::index_env(Some(&t.svc.name)).is_some_and(|e| e.runtime.is_some()) {
        crate::script::search::add_hit_fields(&mut fields, &mut raw_fields, &t.svc.name, &run.lookup, leaf, doc, score, req.script_fields.as_ref(), &req.fields, &run.scripts)?;
    }
    crate::fieldtypes::percolate::add_hit_fields(&run.query, leaf, doc, &mut fields);
    if !fields.is_empty() || !raw_fields.is_empty() {
        let mut all: std::collections::BTreeMap<_, _> = fields.iter().map(|(k,v)| (k.clone(), crate::source_json::Source::from_value(v))).collect();
        all.extend(raw_fields);
        s.push_str(",\"fields\":");
        s.push_str(&crate::source_json::Source::Object(all).text());
    }
    if let (false, Some(p)) = (req.fields.is_empty(), &parsed) {
        let mut igv = Map::new();
        for n in &ignored {
            if req.fields.iter().any(|(pat, _)| crate::mapping::wildcard_match(pat, n)) {
                let vals: Vec<Value> = source_values_ordered(p, n).into_iter().cloned().collect();
                if !vals.is_empty() {
                    igv.insert(n.to_string(), Value::Array(vals));
                }
            }
        }
        if !igv.is_empty() {
            s.push_str(",\"ignored_field_values\":");
            s.push_str(&serde_json::to_string(&igv)?);
        }
    }
    // Percolator highlights the submitted document using the matching stored
    // query; it must work even when the stored query's _source is not returned.
    let stored_src = if parsed.is_none() && req.highlight.is_some() { crate::stored::expanded_source(&stored) } else { None };
    if let Some(hl) = &req.highlight {
        let mut m = Map::new();
        if let (Some(p), Some(terms)) = (parsed.as_ref().or(stored_src.as_ref()), &run.terms) {
            m = highlight::try_highlight_hit(hl, p, &run.lookup, terms, doc).map_err(|e| EsError::search_phase(&t.svc.name, e))?;
            m.extend(crate::inference::highlight::highlight_hit(hl, &run.lookup, leaf, doc, terms, p));
        }
        m.extend(crate::fieldtypes::percolate::highlight_hit(&run.query, leaf, doc, hl, &t.svc)
            .map_err(|e| EsError::search_phase(&t.svc.name, e))?);
        if !m.is_empty() {
            s.push_str(",\"highlight\":");
            s.push_str(&serde_json::to_string(&m)?);
        }
    }
    if let Some(h) = h {
        push_hit_sort(&mut s, run, h, req)?;
        if !run.named.is_empty() {
            let mut matched: Vec<(&str, f32)> = Vec::new();
            for (name, q, ctx) in &run.named {
                if let Some(sc) = q.score_at(leaf, ctx, doc) {
                    matched.push((name.as_str(), sc));
                }
            }
            if !matched.is_empty() {
                if req.include_named_queries_score {
                    let m: Map<String, Value> = matched.iter().map(|(n, sc)| (n.to_string(), serde_json::from_str(&crate::retriever::score_json(*sc)).unwrap_or(Value::Null))).collect();
                    s.push_str(",\"matched_queries\":");
                    s.push_str(&serde_json::to_string(&m)?);
                } else {
                    let names: Vec<&str> = matched.iter().map(|(n, _)| *n).collect();
                    s.push_str(",\"matched_queries\":");
                    s.push_str(&serde_json::to_string(&names)?);
                }
            }
        }
        if let Some(e) = &h.explanation {
            s.push_str(&format!(",\"_shard\":\"[{}][{}]\",\"_node\":\"{}\",\"_explanation\":", t.svc.name, t.shard.as_ref().map(|x| x.id).unwrap_or(0), crate::node::NODE_ID));
            let names: HashMap<u16, String> = run.lookup.core_schema().into_iter().map(|f| (f.id, f.name)).collect();
            s.push_str(&serde_json::to_string(&crate::explain_fmt::to_json(&crate::explain_fmt::to_lucene(e, &names)))?);
        }
    }
    s.push('}');
    Ok(s)
}

/// `_count`.
pub fn count(targets: &[Target], query: Option<&Value>, terms_lookup: Option<TermsLookupFn>) -> Result<u64> {
    let req = SearchRequest { query: query.cloned(), size: 0, track_total_hits: Some(TrackTotalHits::True), ..Default::default() };
    let out = execute(targets, &req, &ExecOpts { terms_lookup, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None })?;
    Ok(out.total)
}

/// One page of [`for_each_match`]: ascending docs of one leaf of one shard.
pub struct MatchPage<'a> {
    /// Position of `target` in the searched targets.
    pub target_idx: usize,
    pub target: &'a Target,
    pub lookup: &'a FieldLookup,
    pub leaf: &'a LeafReader<'a>,
    /// Position of `leaf` among the target's leaves.
    pub leaf_idx: usize,
    pub docs: &'a [DocId],
    /// `docs` are every doc of the leaf (no deletes; the query matches all
    /// of them): a column of the leaf can be read whole.
    pub whole_leaf: bool,
    /// The page is every doc of the leaf, `0..leaf.num_docs()`, and `docs`
    /// is empty (a match-all over a leaf without deletes).
    pub all_docs: bool,
}

impl MatchPage<'_> {
    /// Number of docs in the page.
    pub fn len(&self) -> usize {
        if self.all_docs { self.leaf.num_docs() as usize } else { self.docs.len() }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The page's docs, ascending.
    pub fn doc_ids(&self) -> Box<dyn Iterator<Item = DocId> + '_> {
        if self.all_docs { Box::new(0..self.leaf.num_docs()) } else { Box::new(self.docs.iter().copied()) }
    }
}

/// The query of a search over one target, as the query phase builds it
/// (alias filters, DLS), and its leaves in search order: `f(leaves,
/// lookup, query, ctx)`.
fn with_target<R>(t: &Target, query: &Value, now: i64, f: &mut dyn FnMut(&[LeafReader<'_>], &FieldLookup, &Query, &SearchContext) -> Result<R>) -> Result<R> {
    if crate::script::env().is_some() { return with_target_in_env(t, query, now, f); }
    // Callers outside a search request (ES|QL filters) still need a request
    // error channel: lazy per-document failures (a percolated stored query
    // over the regex limit) are errors, never silent misses.
    crate::script::with_env(crate::script::ScriptEnv::new(None, None), || {
        let result = with_target_in_env(t, query, now, f);
        match crate::script::env().and_then(|env| env.take_error()) {
            Some(error) => Err(error),
            None => result,
        }
    })
}

fn with_target_in_env<R>(t: &Target, query: &Value, now: i64, f: &mut dyn FnMut(&[LeafReader<'_>], &FieldLookup, &Query, &SearchContext) -> Result<R>) -> Result<R> {
    if !t.svc.is_open() {
        return Err(EsError::index_closed(&t.svc.name, &t.svc.uuid));
    }
    if t.snapshot.is_none() {
        t.shard_copy().sync_schema_for_search()?;
        t.shard_copy().shard.before_search();
    }
    let view = target_view(t);
    let mut leaves: Vec<LeafReader<'_>> = view.leaves();
    if t.svc.segments_sorted_on_timestamp() {
        crate::segsort::order_leaves(&mut leaves, &t.lookup());
    }
    let lookup = t.lookup();
    crate::script::enter_index(&t.svc.name, &lookup, || t.svc.mapping_runtime())?;
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    let qctx = QueryContext {
        deadline: req.timeout.and_then(|timeout| Instant::now().checked_add(timeout)),
        lookup: &lookup,
        index_name: &t.svc.name,
        index_uuid: &t.svc.uuid,
        aliases: t.svc.aliases().keys().cloned().collect(),
        now,
        leaves: &leaves,
        default_fields: t.svc.query_default_fields(),
        terms_lookup: None,
        max_terms_count: t.svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: t.svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: t.svc.seq_no_disabled(),
        allow_unmapped_fields: t.svc.setting("index.query.parse.allow_unmapped_fields").map(|v| v != "false").unwrap_or(true),
        knn_prefilter: t.effective_filter(),
        knn_prefilter_lookup: Some(t.svc.lookup()),
        allow_expensive_queries: t.svc.setting("index.search.allow_expensive_queries").map(|v| v != "false").unwrap_or(true),
        mappings: Some(&**t.svc),
    };
    let (q, _) = build_query_with(&req, t, &qctx, None, None).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
    let ctx = SearchContext::for_query(&q, &leaves, Bm25Params::default());
    ctx.execution_result()?;
    let result = f(&leaves, &lookup, &q, &ctx);
    ctx.execution_result()?;
    result
}

/// The query-phase view of one target for [`for_each_match`]: `f` gets the
/// matching docs page by page; returns whether every page was accepted.
fn match_target(ti: usize, t: &Target, query: &Value, page: usize, lazy: Option<usize>, now: i64, f: &mut dyn FnMut(&MatchPage<'_>) -> Result<bool>) -> Result<bool> {
    with_target(t, query, now, &mut |leaves, lookup, q, ctx| {
        let page = page.max(1);
        let mut buf: Vec<DocId> = Vec::with_capacity(page.min(1 << 16));
        let geo = if lazy.is_some() { q.geo_pred() } else { None };
        for (li, leaf) in leaves.iter().enumerate() {
            // A match-all over a leaf without deletes: the whole leaf at once.
            if q.is_match_all() && leaf.deleted_bits().is_none() {
                if leaf.num_docs() == 0 {
                    continue;
                }
                let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &[], whole_leaf: true, all_docs: true };
                if !f(&p)? {
                    return Ok(false);
                }
                continue;
            }
            // Few rows wanted from a dense geo predicate: test the docs in
            // order against their doc values (Lucene's doc-values query for
            // a small lead cost) instead of collecting every match.
            if let (Some(n), Some((field, pred)), LeafReader::Segment(seg)) = (lazy, &geo, leaf) {
                if let Some(g) = seg.geo_index(*field).filter(|g| g.single_valued()) {
                    let est = g.pred_estimate(pred).max(1) as f64;
                    let nd = leaf.num_docs() as f64;
                    if leaf.deleted_bits().is_none() && (n as f64) * nd / est < est {
                        let dv = rustic_core::query::GeoDv::open(leaf, *field);
                        for e in dv.iter() {
                            if pred.test(rustic_core::geo::decode_lat(e.lat), rustic_core::geo::decode_lon(e.lon)) {
                                buf.push(e.doc);
                                if buf.len() >= page {
                                    let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &buf, whole_leaf: false, all_docs: false };
                                    let go = f(&p)?;
                                    buf.clear();
                                    if !go {
                                        return Ok(false);
                                    }
                                }
                            }
                        }
                        if !buf.is_empty() {
                            let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &buf, whole_leaf: false, all_docs: false };
                            let go = f(&p)?;
                            buf.clear();
                            if !go {
                                return Ok(false);
                            }
                        }
                        continue;
                    }
                }
            }
            // Few rows wanted from a geo predicate without core doc values
            // of its own (a `geo_shape` field's point tree): its scorer may
            // scan in doc order and stop early (see `geo_scorer`).
            let built = match lazy {
                Some(n) if q.takes_geo_hint() => rustic_core::query::geo_scorer::with_wanted(n.max(1) as u64, || q.scorer(leaf, ctx)),
                _ => q.scorer(leaf, ctx),
            };
            let Some(mut sc) = built else { continue };
            let live = leaf.live_docs();
            let mut seen = 0u64;
            let mut whole = true;
            let mut flush = |buf: &mut Vec<DocId>, whole_leaf: bool| -> Result<bool> {
                let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: buf, whole_leaf, all_docs: false };
                let go = f(&p)?;
                buf.clear();
                Ok(go)
            };
            loop {
                let d = rustic_core::query::DocSet::advance(&mut *sc);
                if d == rustic_core::query::TERMINATED {
                    break;
                }
                if live(d) {
                    buf.push(d);
                    seen += 1;
                    if buf.len() >= page {
                        whole = false;
                        if !flush(&mut buf, false)? {
                            return Ok(false);
                        }
                    }
                }
            }
            if !buf.is_empty() {
                // Every doc of the leaf (so none deleted): callers may read the
                // leaf's column whole.
                let whole_leaf = whole && seen == leaf.num_docs() as u64;
                if !flush(&mut buf, whole_leaf)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    })
}

/// One leaf of [`par_for_each_leaf`]: the compiled query and its context.
pub struct LeafView<'a> {
    pub target_idx: usize,
    pub target: &'a Target,
    pub lookup: &'a FieldLookup,
    pub leaf: &'a LeafReader<'a>,
    pub query: &'a Query,
    pub ctx: &'a SearchContext,
}

impl LeafView<'_> {
    /// Visit the live docs the query matches on this leaf, ascending.
    pub fn for_each_match(&self, mut f: impl FnMut(DocId)) {
        let Some(mut sc) = self.query.scorer(self.leaf, self.ctx) else { return };
        let live = self.leaf.live_docs();
        loop {
            let d = rustic_core::query::DocSet::advance(&mut *sc);
            if d == rustic_core::query::TERMINATED {
                break;
            }
            if live(d) {
                f(d);
            }
        }
    }
}

/// Every leaf of `targets` with the query compiled for it (search order
/// within a target), one thread per target: each target folds its leaves
/// into its own state; the states come back in target order.
pub fn par_for_each_leaf<S: Send>(targets: &[Target], query: &Value, init: &(dyn Fn(usize) -> S + Sync), f: &(dyn Fn(&mut S, &LeafView<'_>) -> Result<()> + Sync)) -> Result<Vec<S>> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let run = |ti: usize, t: &Target| -> Result<S> {
        let mut st = init(ti);
        with_target(t, query, now, &mut |leaves, lookup, q, ctx| {
            for leaf in leaves {
                f(&mut st, &LeafView { target_idx: ti, target: t, lookup, leaf, query: q, ctx })?;
            }
            Ok(())
        })?;
        Ok(st)
    };
    if targets.len() <= 1 {
        return targets.iter().enumerate().map(|(ti, t)| run(ti, t)).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = targets.iter().enumerate().map(|(ti, t)| scope.spawn(move || run(ti, t))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard scan panicked")))).collect()
    })
}

/// [`par_for_each_leaf`] keeping one state per leaf (`init` per leaf, in
/// leaf order within each target): when the query is a geo point predicate
/// or matches all (no script runs in either, so the thread-local request
/// environment is not needed), a target's leaves run concurrently on the
/// search pool; otherwise one after another, in place. The caller merges
/// the states in order.
pub fn par_for_each_leaf_states<S: Send>(targets: &[Target], query: &Value, init: &(dyn Fn(usize) -> S + Sync), f: &(dyn Fn(&mut S, &LeafView<'_>) -> Result<()> + Sync)) -> Result<Vec<Vec<S>>> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let run = |ti: usize, t: &Target| -> Result<Vec<S>> {
        with_target(t, query, now, &mut |leaves, lookup, q, ctx| {
            let leaf_state = |leaf: &LeafReader<'_>| -> Result<S> {
                let mut st = init(ti);
                f(&mut st, &LeafView { target_idx: ti, target: t, lookup, leaf, query: q, ctx })?;
                Ok(st)
            };
            if q.is_match_all() || q.geo_pred().is_some() || script_free(q) {
                rustic_core::query::searcher::par_map_search(leaves.len(), |i| leaf_state(&leaves[i])).into_iter().collect()
            } else {
                leaves.iter().map(leaf_state).collect()
            }
        })
    };
    if targets.len() <= 1 {
        return targets.iter().enumerate().map(|(ti, t)| run(ti, t)).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = targets.iter().enumerate().map(|(ti, t)| scope.spawn(move || run(ti, t))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard scan panicked")))).collect()
    })
}

/// Whether `q` is built only from leaves that run no script (terms,
/// ranges, exists, match all / none) combined by bool, boost, constant
/// score and dis_max: its leaves can be folded on the search pool, which
/// does not carry the request's thread-local environment (a script query
/// is a `Custom` leaf, and is never folded there).
fn script_free(q: &Query) -> bool {
    match q {
        Query::Term(_) | Query::TermSet { .. } | Query::RangeI64 { .. } | Query::RangeU64 { .. } | Query::RangeF64 { .. } | Query::RangeDate { .. }
        | Query::Exists { .. } | Query::MatchAll | Query::MatchNone => true,
        Query::ConstScore { query, .. } | Query::Boost { query, .. } => script_free(query),
        Query::Bool { must, should, must_not, filter, .. } => must.iter().chain(should).chain(must_not).chain(filter).all(script_free),
        Query::DisMax { queries, .. } => queries.iter().all(script_free),
        _ => false,
    }
}

/// The number of docs `query` matches over `targets` (as a search counts
/// them), shards in parallel: leaves whose count the query knows without a
/// scan (KD-tree leaf counts) are not visited.
pub fn par_count(targets: &[Target], query: &Value) -> Result<u64> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let run = |t: &Target| -> Result<u64> {
        with_target(t, query, now, &mut |leaves, _lookup, q, ctx| {
            let scan_leaf = |leaf: &LeafReader<'_>| -> u64 {
                let Some(mut sc) = q.scorer(leaf, ctx) else { return 0 };
                let live = leaf.live_docs();
                let mut n = 0u64;
                loop {
                    let d = rustic_core::query::DocSet::advance(&mut *sc);
                    if d == rustic_core::query::TERMINATED {
                        break;
                    }
                    if live(d) {
                        n += 1;
                    }
                }
                n
            };
            let count_leaf = |leaf: &LeafReader<'_>| -> u64 { q.exact_count(leaf).unwrap_or_else(|| scan_leaf(leaf)) };
            // Leaves counted concurrently on the search pool (a shard of
            // many segments otherwise counts them one after another; ES|QL
            // drivers and Lucene's concurrent search split by segment too).
            // A request script environment is thread-local (and `with_target`
            // always installs one): then only the index counts (KD-tree leaf
            // counts, leaf sizes; no script runs in them) are concurrent, and
            // leaves needing a scan are counted in place.
            if crate::script::env().is_some() {
                let exact = rustic_core::query::searcher::par_map_search(leaves.len(), |i| q.exact_count(&leaves[i]));
                return Ok(leaves.iter().zip(exact).map(|(leaf, n)| n.unwrap_or_else(|| scan_leaf(leaf))).sum());
            }
            Ok(rustic_core::query::searcher::par_map_search(leaves.len(), |i| count_leaf(&leaves[i])).into_iter().sum())
        })
    };
    if targets.len() <= 1 {
        return targets.iter().map(run).sum();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = targets.iter().map(|t| scope.spawn(move || run(t))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard count panicked")))).sum()
    })
}

/// Stream the documents a search with `query` matches over `targets`
/// (alias filters, document-level security and deletes applied, as the
/// query phase sees them), shard by shard and leaf by leaf (search order),
/// in pages of at most `page` ascending docs: callers needing every match
/// (ES|QL `STATS`) hold one page at a time instead of every hit. `f`
/// returns whether to continue.
pub fn for_each_match(targets: &[Target], query: &Value, page: usize, f: &mut dyn FnMut(&MatchPage<'_>) -> Result<bool>) -> Result<()> {
    for_each_match_wanting(targets, query, page, None, f)
}

/// [`for_each_match`] resumed at `from` = (target, leaf, doc): the matches
/// of leaf `from.1` of target `from.0` at or after doc `from.2`, then those
/// of every later leaf and target, in the same order. Pages hold at most
/// `page` ascending docs and always list them (`docs`; a match-all leaf is
/// paged too, never `all_docs`), so a caller that stops after a page
/// resumes after its last doc. Each call compiles the query again: callers
/// read many pages per call (a cursor page), not one call per page.
pub fn for_each_match_from(targets: &[Target], query: &Value, page: usize, from: (usize, usize, DocId), f: &mut dyn FnMut(&MatchPage<'_>) -> Result<bool>) -> Result<()> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let page = page.max(1);
    for (ti, t) in targets.iter().enumerate().skip(from.0) {
        let go = with_target(t, query, now, &mut |leaves, lookup, q, ctx| {
            let mut buf: Vec<DocId> = Vec::with_capacity(page.min(1 << 16));
            for (li, leaf) in leaves.iter().enumerate() {
                let start: DocId = if ti != from.0 {
                    0
                } else if li < from.1 {
                    continue;
                } else if li == from.1 {
                    from.2
                } else {
                    0
                };
                let n = leaf.num_docs();
                if start >= n {
                    continue;
                }
                if q.is_match_all() && leaf.deleted_bits().is_none() {
                    let mut d = start;
                    while d < n {
                        let end = d.saturating_add(page as DocId).min(n);
                        buf.clear();
                        buf.extend(d..end);
                        let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &buf, whole_leaf: false, all_docs: false };
                        if !f(&p)? {
                            return Ok(false);
                        }
                        d = end;
                    }
                    buf.clear();
                    continue;
                }
                let Some(mut sc) = q.scorer(leaf, ctx) else { continue };
                let live = leaf.live_docs();
                let mut d = rustic_core::query::DocSet::advance(&mut *sc);
                if d != rustic_core::query::TERMINATED && d < start {
                    d = rustic_core::query::DocSet::seek(&mut *sc, start);
                }
                while d != rustic_core::query::TERMINATED {
                    if live(d) {
                        buf.push(d);
                        if buf.len() >= page {
                            let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &buf, whole_leaf: false, all_docs: false };
                            let go = f(&p)?;
                            buf.clear();
                            if !go {
                                return Ok(false);
                            }
                        }
                    }
                    d = rustic_core::query::DocSet::advance(&mut *sc);
                }
                if !buf.is_empty() {
                    let p = MatchPage { target_idx: ti, target: t, lookup, leaf, leaf_idx: li, docs: &buf, whole_leaf: false, all_docs: false };
                    let go = f(&p)?;
                    buf.clear();
                    if !go {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        })?;
        if !go {
            break;
        }
    }
    Ok(())
}

/// [`for_each_match`] when the caller stops after about `wanted` docs
/// (`LIMIT`): dense geo predicates are tested doc by doc in order instead
/// of collecting all their matches first. Same docs, same order.
pub fn for_each_match_wanting(targets: &[Target], query: &Value, page: usize, wanted: Option<usize>, f: &mut dyn FnMut(&MatchPage<'_>) -> Result<bool>) -> Result<()> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    for (ti, t) in targets.iter().enumerate() {
        if !match_target(ti, t, query, page, wanted, now, f)? {
            break;
        }
    }
    Ok(())
}

/// Visit every leaf of `targets` (each target's snapshot when it has one,
/// else its current view) with the target's field lookup: `f` returning
/// false stops the walk, and the result says whether it ran to the end.
pub fn for_each_leaf(targets: &[Target], f: &mut dyn FnMut(&Target, &FieldLookup, &LeafReader<'_>) -> Result<bool>) -> Result<bool> {
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let all = json!({"match_all": {}});
    for t in targets {
        let complete = with_target(t, &all, now, &mut |leaves, lookup, _, _| {
            for leaf in leaves {
                if !f(t, lookup, leaf)? {
                    return Ok(false);
                }
            }
            Ok(true)
        })?;
        if !complete {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Run `f` over every leaf of the single target `targets` names (its
/// snapshot when it has one), all open at once, with its field lookup.
/// `None` when `targets` expands to no target or several.
pub fn with_leaves<R>(targets: &[Target], f: &mut dyn FnMut(&Target, &[LeafReader<'_>], &FieldLookup) -> Result<R>) -> Result<Option<R>> {
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let [t] = targets else { return Ok(None) };
    let all = json!({"match_all": {}});
    with_target(t, &all, crate::date::now_millis(), &mut |leaves, lookup, _, _| f(t, leaves, lookup)).map(Some)
}

/// [`for_each_match`] with one thread per target: each target folds its
/// pages into its own state (`init(target_idx)`); the states come back in
/// target order. `f` returning false stops that target only.
pub fn par_for_each_match<S: Send>(targets: &[Target], query: &Value, page: usize, init: &(dyn Fn(usize) -> S + Sync), f: &(dyn Fn(&mut S, &MatchPage<'_>) -> Result<bool> + Sync)) -> Result<Vec<S>> {
    let req = SearchRequest { query: Some(query.clone()), size: 0, ..Default::default() };
    check_security_restrictions(targets, &req)?;
    let expanded;
    let targets = if needs_expansion(targets) {
        expanded = expand_targets(targets)?;
        &expanded[..]
    } else {
        targets
    };
    let now = crate::date::now_millis();
    let run = |ti: usize, t: &Target| -> Result<S> {
        let mut st = init(ti);
        match_target(ti, t, query, page, None, now, &mut |p| f(&mut st, p))?;
        Ok(st)
    };
    if targets.len() <= 1 {
        return targets.iter().enumerate().map(|(ti, t)| run(ti, t)).collect();
    }
    // Script / index context is thread-local: each worker enters its index.
    std::thread::scope(|scope| {
        let handles: Vec<_> = targets.iter().enumerate().map(|(ti, t)| scope.spawn(move || run(ti, t))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard scan panicked")))).collect()
    })
}

/// The can-match check (`field_caps` `index_filter`): false when `query`
/// rewrites to `match_none` on the target (an unmapped field, an `_index` /
/// `_index_mode` / `constant_keyword` mismatch), true otherwise even if no
/// document matches.
/// Does `q` rewrite to `match_none` (the can-match phase skips the shard)?
fn rewrites_to_none(q: &Query) -> bool {
    match q {
        Query::MatchNone => true,
        Query::Boost { query, .. } | Query::ConstScore { query, .. } => rewrites_to_none(query),
        Query::Bool { must, should, filter, .. } => {
            must.iter().chain(filter).any(rewrites_to_none) || (must.is_empty() && filter.is_empty() && !should.is_empty() && should.iter().all(rewrites_to_none))
        }
        _ => false,
    }
}

/// Lucene's shard-global doc number of `(leaf, doc)` of a search over `t`
/// (the leaves' doc counts before `leaf`, in the order searches see them).
pub fn global_doc(t: &Target, leaf: usize, doc: DocId) -> u64 {
    let view = target_view(t);
    let mut leaves: Vec<LeafReader<'_>> = view.leaves();
    if t.svc.segments_sorted_on_timestamp() {
        crate::segsort::order_leaves(&mut leaves, &t.lookup());
    }
    leaves.iter().take(leaf).map(|l| l.num_docs() as u64).sum::<u64>() + doc as u64
}

/// Does the request run the can-match phase over `targets`?
fn prefilter_applies(req: &SearchRequest, targets: &[Target]) -> bool {
    if req.dfs {
        return false;
    }
    let threshold = req.pre_filter_shard_size.unwrap_or_else(|| {
        let write_blocked = |t: &Target| ["index.blocks.write", "index.blocks.read_only", "index.blocks.read_only_allow_delete"].iter().any(|k| t.svc.setting(k).as_deref() == Some("true"));
        if targets.iter().any(write_blocked) { 1 } else { 128 }
    });
    threshold < READ_SHARD_COUNT.with(|slot| slot.get()).unwrap_or(targets.len()) && can_rewrite_to_match_none(req)
}

/// `SearchService.canRewriteToMatchNone`: a query other than `match_all`,
/// no suggest, and no aggregation that must visit every document.
fn can_rewrite_to_match_none(req: &SearchRequest) -> bool {
    let Some(q) = &req.query else { return false };
    if q.get("match_all").is_some() || req.suggest.is_some() {
        return false;
    }
    // `AggregatorFactories.mustVisitAllDocs` (top-level aggregations only).
    let must_visit_all = req.aggs.as_ref().and_then(Value::as_object).is_some_and(|aggs| {
        aggs.values().any(|a| a.get("global").is_some() || a.get("terms").is_some_and(|t| t.get("min_doc_count").and_then(Value::as_i64) == Some(0)))
    });
    !must_visit_all
}

/// Does a required `range` clause of `query` match no value on the shard
/// (ES rewrites such a range to `match_none` from the field's min/max)?
fn range_disjoint(t: &Target, query: &Value) -> bool {
    // A whole-index target (field_caps' `index_filter`): disjoint only when
    // it is on every shard copy here (an index is kept if any shard can
    // match); none open here, or an unexpandable target, is not disjoint.
    if t.shard.is_none() {
        return match expand_targets(std::slice::from_ref(t)) {
            Ok(shards) if !shards.is_empty() => shards.iter().all(|s| range_disjoint(s, query)),
            _ => false,
        };
    }
    let Some((kind, body)) = query.as_object().and_then(|o| o.iter().next()) else { return false };
    match kind.as_str() {
        "range" => {
            // Only an indexed date field rewrites by its min/max
            // (`DateFieldType.isFieldWithinQuery`; doc-values-only fields cannot).
            let field = body.as_object().and_then(|o| o.keys().next()).cloned().unwrap_or_default();
            let lookup = t.lookup();
            let Some(f) = lookup.get(&field) else { return false };
            if !f.indexed || !matches!(f.ty, crate::mapping::EsType::Date | crate::mapping::EsType::DateNanos) {
                return false;
            }
            // ES `RangeQueryBuilder.getRelation`: DISJOINT from the shard's
            // min / max of the field (no value at all included), not "no
            // hit": a range between two values the shard holds is searched.
            let Some(id) = f.core_id else { return false };
            let Some((lower, upper)) = with_query_context(t, |qctx| crate::query::dsl::parse_query(query, qctx).ok()).and_then(|q| range_bounds(&q)) else { return false };
            let view = target_view(t);
            let bounds = view.leaves().iter().filter_map(|l| l.min_max_i64(id)).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
            match bounds {
                None => true,
                Some((min, max)) => lower > max || upper < min || lower > upper,
            }
        }
        "bool" => {
            let list = |k: &str| -> Vec<Value> {
                match body.get(k) {
                    Some(Value::Array(a)) => a.clone(),
                    Some(v @ Value::Object(_)) => vec![v.clone()],
                    _ => Vec::new(),
                }
            };
            let required: Vec<Value> = list("must").into_iter().chain(list("filter")).collect();
            if required.iter().any(|c| range_disjoint(t, c)) {
                return true;
            }
            let should = list("should");
            required.is_empty() && !should.is_empty() && should.iter().all(|c| range_disjoint(t, c))
        }
        "constant_score" => body.get("filter").is_some_and(|f| range_disjoint(t, f)),
        _ => false,
    }
}

/// A range query's inclusive `[lower, upper]` (`Excluded` bounds moved in),
/// under the boosts and constant scores the parse wraps it in.
fn range_bounds(q: &Query) -> Option<(i64, i64)> {
    use rustic_core::query::Bound;
    let incl = |b: &Bound<i64>, low: bool| match b {
        Bound::Included(v) => *v,
        Bound::Excluded(v) if low => v.saturating_add(1),
        Bound::Excluded(v) => v.saturating_sub(1),
        Bound::Unbounded if low => i64::MIN,
        Bound::Unbounded => i64::MAX,
    };
    match q {
        Query::RangeI64 { lower, upper, .. } | Query::RangeDate { lower, upper, .. } => Some((incl(lower, true), incl(upper, false))),
        Query::Boost { query, .. } | Query::ConstScore { query, .. } => range_bounds(query),
        _ => None,
    }
}

/// Run `f` with the query context of a can-match check on `t`.
fn with_query_context<R>(t: &Target, f: impl FnOnce(&QueryContext) -> R) -> R {
    let lookup = t.lookup();
    let aliases: Vec<String> = t.svc.aliases().keys().cloned().collect();
    let qctx = QueryContext {
        deadline: None,
        lookup: &lookup,
        index_name: &t.svc.name,
        index_uuid: &t.svc.uuid,
        aliases,
        now: chrono::Utc::now().timestamp_millis(),
        leaves: &[],
        default_fields: vec!["*".to_string()],
        terms_lookup: None,
        max_terms_count: t.svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: t.svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: t.svc.seq_no_disabled(),
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: true,
        mappings: Some(&**t.svc),
    };
    f(&qctx)
}

pub fn can_match(t: &Target, query: &Value) -> Result<bool> {
    let none = rewrites_to_none;
    let lookup = t.lookup();
    let aliases: Vec<String> = t.svc.aliases().keys().cloned().collect();
    let qctx = QueryContext {
        deadline: None,
        lookup: &lookup,
        index_name: &t.svc.name,
        index_uuid: &t.svc.uuid,
        aliases,
        now: chrono::Utc::now().timestamp_millis(),
        leaves: &[],
        default_fields: vec!["*".to_string()],
        terms_lookup: None,
        max_terms_count: t.svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: t.svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: t.svc.seq_no_disabled(),
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: true,
        mappings: Some(&**t.svc),
    };
    if none(&parse_query(query, &qctx)?) {
        return Ok(false);
    }
    // ES rewrites a `range` against the shard's min / max values (no
    // overlap, or no values at all, is `match_none`), as the search
    // pre-filter does.
    Ok(!range_disjoint(t, query))
}

/// Document-level security for single-document reads: is document `id` of
/// `svc` visible to a caller restricted by `access`? Refreshes first so a
/// realtime-visible document is judged by the same query as a search.
/// `refresh = false` is for callers already inside a search (they hold the shard read
/// lock, so a refresh would deadlock): an unrefreshed document is then treated as hidden.
pub fn doc_visible(svc: &Arc<IndexService>, id: &str, access: &Arc<crate::security::roles::IndexAccess>, refresh: bool) -> Result<bool> {
    if access.dls.is_none() {
        return Ok(true);
    }
    if refresh {
        svc.refresh()?;
    }
    let t = Target { access: Some(access.clone()), ..Target::new(svc.clone()) };
    Ok(count(std::slice::from_ref(&t), Some(&json!({"ids": {"values": [id]}})), None)? > 0)
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Target({})", self.svc.name)
    }
}

#[cfg(test)]
mod fetch_document_tests {
    use super::*;

    #[test]
    fn coordinator_prefilter_count_is_scoped_and_restored() {
        let request = SearchRequest { query: Some(json!({"match_none":{}})), pre_filter_shard_size: Some(1), ..Default::default() };
        assert!(!prefilter_applies(&request, &[]));
        {
            let _scope = ReadShardCountScope::enter(Some(3));
            assert!(prefilter_applies(&request, &[]));
            let dfs = SearchRequest { dfs: true, ..request.clone() };
            assert!(!prefilter_applies(&dfs, &[]));
        }
        assert!(!prefilter_applies(&request, &[]));
    }

    use rustic_core::{Index, IndexSettings};

    /// A light query phase (`light_refs`) renders what the merge reads
    /// (`_index`, `_score`, `sort`, the address) and nothing it reads from
    /// the documents, with the full query phase's scores, sort values and
    /// order; an addressed fetch (`fetch_refs`) renders the same hits as
    /// a plain fetch, each followed by its address.
    #[test]
    fn light_query_phase_and_addressed_fetch() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("t", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"body": {"type": "text"}, "n": {"type": "long"}}}})).unwrap();
        let w = (0..60u32).map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"body": format!("apple {}", "x ".repeat(i as usize % 7)), "n": i % 9}).to_string(), routing: Some(format!("r{i}")), cond: WriteCond::default() }).collect();
        for r in svc.write(w, true) { r.unwrap(); }
        svc.refresh().unwrap();
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let pinned = pin_views(&node.resolve("t", ResolveOpts::default()).unwrap()).unwrap();
        for body in [json!({"query": {"match": {"body": "apple"}}, "size": 9, "_source": false}),
            json!({"query": {"match": {"body": "apple"}}, "size": 9, "_source": false, "sort": [{"n": "desc"}, "_score"], "track_scores": true}),
            json!({"query": {"match": {"body": "apple"}}, "size": 9, "_source": false, "sort": [{"n": "asc"}]})] {
            let mut full = parse_request(&body, &HashMap::new()).unwrap();
            full.doc_refs = true;
            let mut light = full.clone();
            light.light_refs = true;
            let full: Value = serde_json::from_str(&execute(&pinned, &full, &opts).unwrap().json).unwrap();
            let light: Value = serde_json::from_str(&execute(&pinned, &light, &opts).unwrap().json).unwrap();
            let (fh, lh) = (full["hits"]["hits"].as_array().unwrap(), light["hits"]["hits"].as_array().unwrap());
            assert_eq!(fh.len(), 9);
            assert_eq!(fh.len(), lh.len());
            for (f, l) in fh.iter().zip(lh) {
                let keys: Vec<&String> = l.as_object().unwrap().keys().collect();
                let want: Vec<&str> = if body.get("sort").is_some() { vec!["_index", "_score", "sort", "_rustic_doc"] } else { vec!["_index", "_score", "_rustic_doc"] };
                assert_eq!(keys, want, "{l}");
                for k in want { assert_eq!(l[k], f[k], "{k}: {l} vs {f}"); }
                assert!(f.get("_id").is_some() && f.get("_routing").is_some());
            }
            // The fetch of the light winners, addressed and plain.
            let docs: Vec<(usize, usize, u32, f32)> = lh.iter().map(|h| {
                let r = &h["_rustic_doc"];
                (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize, r[2].as_u64().unwrap() as u32, h["_score"].as_f64().unwrap_or(0.0) as f32)
            }).collect();
            let mut f = parse_request(&json!({"query": {"match": {"body": "apple"}}}), &HashMap::new()).unwrap();
            (f.size, f.track_total_hits, f.fetch_docs) = (docs.len(), Some(TrackTotalHits::False), Some(Arc::new(docs)));
            let plain: Value = serde_json::from_str(&execute(&pinned, &f, &opts).unwrap().json).unwrap();
            f.fetch_refs = true;
            let addressed: Value = serde_json::from_str(&execute(&pinned, &f, &opts).unwrap().json).unwrap();
            let (ph, ah) = (plain["hits"]["hits"].as_array().unwrap(), addressed["hits"]["hits"].as_array().unwrap());
            assert_eq!(ph.len(), 9);
            for (p, a) in ph.iter().zip(ah) {
                let mut a = a.clone();
                let at = a.as_object_mut().unwrap().shift_remove("_rustic_doc").unwrap();
                assert_eq!(&a, p);
                assert!(lh.iter().any(|w| w["_rustic_doc"] == at), "{at}");
            }
        }
    }

    /// query_then_fetch on pinned views: the fetch phase renders the query
    /// phase's docs from the view it searched, after the shard changed.
    #[test]
    fn fetch_phase_renders_query_phase_docs_from_the_pinned_view() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("t", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"body": {"type": "text"}, "n": {"type": "long"}}}})).unwrap();
        let put = |ids: std::ops::Range<u32>, word: &str| {
            let w = ids.map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"body": format!("{word} {}", "x ".repeat(i as usize % 7)), "n": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
            for r in svc.write(w, true) {
                r.unwrap();
            }
            svc.refresh().unwrap();
        };
        put(0..40, "apple");
        put(40..80, "apple pie");
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let body = json!({"query": {"match": {"body": "apple"}}, "size": 7, "highlight": {"fields": {"body": {}}}});
        let want: Value = serde_json::from_str(&execute(&node.resolve("t", ResolveOpts::default()).unwrap(), &parse_request(&body, &HashMap::new()).unwrap(), &opts).unwrap().json).unwrap();
        let pinned = pin_views(&node.resolve("t", ResolveOpts::default()).unwrap()).unwrap();
        let counts = || {
            use std::sync::atomic::Ordering::Relaxed;
            pinned.iter().map(|t| &t.shard_copy().shard.counters).fold((0, 0), |(q, f), c| (q + c.search_total.load(Relaxed), f + c.fetch_total.load(Relaxed)))
        };
        let before = counts();
        let mut q = parse_request(&json!({"query": {"match": {"body": "apple"}}, "size": 7, "_source": false}), &HashMap::new()).unwrap();
        q.doc_refs = true;
        let phase1: Value = serde_json::from_str(&execute(&pinned, &q, &opts).unwrap().json).unwrap();
        let docs: Vec<(usize, usize, u32, f32)> = phase1["hits"]["hits"].as_array().unwrap().iter().map(|h| {
            let r = &h["_rustic_doc"];
            (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize, r[2].as_u64().unwrap() as u32, h["_score"].as_f64().unwrap() as f32)
        }).collect();
        assert_eq!(docs.len(), 7);
        let shards_hit = docs.iter().map(|d| d.0).collect::<std::collections::BTreeSet<_>>().len() as u64;
        // The query phase counts one query per shard, no fetch.
        assert_eq!(counts(), (before.0 + 2, before.1));
        // Updates, deletes and a merge after the query phase.
        put(0..80, "pear");
        svc.force_merge(1, false).unwrap();
        let mut f = parse_request(&json!({"query": {"match": {"body": "apple"}}, "highlight": {"fields": {"body": {}}}}), &HashMap::new()).unwrap();
        (f.size, f.track_total_hits, f.fetch_docs) = (docs.len(), Some(TrackTotalHits::False), Some(Arc::new(docs)));
        let got: Value = serde_json::from_str(&execute(&pinned, &f, &opts).unwrap().json).unwrap();
        let ids = |v: &Value| v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].clone(), h["_source"].clone(), h["highlight"].clone())).collect::<Vec<_>>();
        assert_eq!(ids(&got), ids(&want));
        assert!(got["hits"]["hits"][0].get("_rustic_doc").is_none());
        // The fetch phase one fetch per shard it rendered hits of, no query.
        assert_eq!(counts(), (before.0 + 2, before.1 + shards_hit));
    }

    /// A coordinator's single round (`defer_fetch_stats`): the same response
    /// as a plain search, one query per shard and no fetch counted (the
    /// coordinator counts it on the shards whose hits it keeps).
    #[test]
    fn deferred_fetch_stats_render_the_plain_response_and_count_queries_only() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("t", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"body": {"type": "text"}, "n": {"type": "long"}}}})).unwrap();
        let w = (0..40u32).map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"body": format!("apple {}", "x ".repeat(i as usize % 5)), "n": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in svc.write(w, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let targets = node.resolve("t", ResolveOpts::default()).unwrap();
        let counts = || {
            use std::sync::atomic::Ordering::Relaxed;
            crate::search::expand_targets(&targets).unwrap().iter().map(|t| &t.shard_copy().shard.counters).fold((0, 0), |(q, f), c| (q + c.search_total.load(Relaxed), f + c.fetch_total.load(Relaxed)))
        };
        for body in [json!({"query": {"match": {"body": "apple"}}, "size": 7}), json!({"sort": [{"n": "desc"}], "size": 5, "version": true, "_source": ["n"]})] {
            let plain_req = parse_request(&body, &HashMap::new()).unwrap();
            let before = counts();
            let plain = execute(&targets, &plain_req, &opts).unwrap();
            let after_plain = counts();
            // A plain search counts a query and a fetch on each shard with hits.
            assert_eq!(after_plain.0, before.0 + 2, "{body}");
            assert!(after_plain.1 > before.1, "{body}");
            let mut deferred_req = plain_req.clone();
            deferred_req.defer_fetch_stats = true;
            let deferred = execute(&targets, &deferred_req, &opts).unwrap();
            // Byte for byte but `took`.
            let untimed = |s: &str| -> String {
                let Some(at) = s.find("\"took\":") else { return s.to_string() };
                let start = at + "\"took\":".len();
                let end = start + s[start..].find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
                format!("{}{}", &s[..start], &s[end..])
            };
            assert_eq!(untimed(&deferred.json), untimed(&plain.json), "{body}");
            assert_eq!(counts(), (after_plain.0 + 2, after_plain.1), "{body}: no fetch counted");
        }
    }

    /// A `top_hits` hit `Value` cannot hold (a lone surrogate in its
    /// source) keeps its metadata for the reduce and renders verbatim.
    #[test]
    fn top_hits_hits_with_lone_surrogates_render_verbatim() {
        let raw = r#"{"_index":"i","_id":"1","_score":1.5,"_source":{"p":1.50,"t":"x\ud800"},"sort":[3]}"#;
        let v = hit_value(raw).unwrap();
        assert_eq!((v["_id"].as_str(), v["_score"].as_f64(), v["sort"][0].as_u64()), (Some("1"), Some(1.5), Some(3)));
        assert!(v.get("_source").is_none());
        let aggs = json!({"top": {"hits": {"total": {"value": 1, "relation": "eq"}, "max_score": 1.5, "hits": [v, hit_value(r#"{"_id":"2","_source":{"a":1}}"#).unwrap()]}}});
        assert!(crate::aggs::has_raw_scripted_value(&aggs));
        let text = crate::aggs::render_scripted_values(&aggs).text();
        assert!(text.contains(r#""_source":{"p":1.50,"t":"x\ud800"}"#) && text.contains(r#""_source":{"a":1}"#) && !text.contains("raw_json"), "{text}");
        // The single-node render (in key order) writes the raw hit verbatim.
        let one = String::from_utf8(crate::source_json::render_aggregations(&aggs)).unwrap();
        assert!(one.contains(&format!(r#""hits":[{raw},{{"_id":"2","_source":{{"a":1}}}}]"#)) && !one.contains("__p"), "{one}");
    }

    /// A query phase whose aggregations fail on some targets searches the
    /// others only: its hit addresses still name the pinned list's targets.
    #[test]
    fn query_phase_doc_refs_survive_partial_agg_failures() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        for (name, extra) in [("a", json!({})), ("b", json!({"index.downsample.interval": "1h"}))] {
            let mut settings = json!({"number_of_shards": 1});
            settings.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let svc = node.create_index(name, &json!({"settings": settings, "mappings": {"properties": {"d": {"type": "date"}}}})).unwrap();
            let w = (0..5).map(|i| WriteReq::Index { id: Some(format!("{name}{i}")), source: json!({"d": 1_600_000_000_000i64 + i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
            for r in svc.write(w, true) {
                r.unwrap();
            }
            svc.refresh().unwrap();
        }
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        // `b` first: the subset searched is `a` alone, at another position.
        let mut targets = node.resolve("a,b", ResolveOpts::default()).unwrap();
        targets.sort_by(|x, y| y.svc.name.cmp(&x.svc.name));
        let pinned = pin_views(&targets).unwrap();
        let mut q = parse_request(&json!({"size": 10, "_source": false, "aggs": {"h": {"date_histogram": {"field": "d", "calendar_interval": "1d"}}}}), &HashMap::new()).unwrap();
        q.doc_refs = true;
        let out = execute(&pinned, &q, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        assert_eq!(v["_shards"]["failed"], 1, "{v}");
        let hits = v["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 5, "{v}");
        for h in hits {
            let ti = h["_rustic_doc"][0].as_u64().unwrap() as usize;
            assert_eq!(pinned[ti].svc.name, h["_index"].as_str().unwrap(), "{h}");
        }
        assert!(out.page.iter().all(|p| pinned[p.0].svc.name == "a"));
    }

    /// The can-match check skips a shard only when a date range is disjoint
    /// from the field's min / max on it (ES `RangeQueryBuilder.getRelation`),
    /// not when the range merely matches no document between them.
    #[test]
    fn can_match_tests_ranges_against_the_field_bounds() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("cm", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"d": {"type": "date", "format": "yyyy-MM-dd"}, "e": {"type": "date"}}}})).unwrap();
        let w = ["2015-01-01", "2019-05-01"].iter().enumerate().map(|(i, d)| WriteReq::Index { id: Some(i.to_string()), source: json!({"d": d}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in svc.write(w, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        // A whole-index target (field_caps' index_filter): expanded, kept
        // when any shard can match.
        let whole = Target::new(svc.clone());
        assert!(whole.shard.is_none());
        assert!(can_match(&whole, &json!({"range": {"d": {"gte": "2016-01-01"}}})).unwrap());
        assert!(!can_match(&whole, &json!({"range": {"d": {"gte": "2030-01-01"}}})).unwrap());
        // can_match runs per shard copy: expand the index target first.
        let t = &expand_targets(&node.resolve("cm", ResolveOpts::default()).unwrap()).unwrap()[0];
        let range = |gte: &str, lt: &str| json!({"bool": {"filter": [{"range": {"d": {"gte": gte, "lt": lt}}}]}});
        assert!(can_match(t, &range("2016-01-01", "2018-01-01")).unwrap(), "between the bounds, no hit: searched");
        assert!(can_match(t, &range("2019-01-01", "2020-01-01")).unwrap());
        assert!(!can_match(t, &range("2020-01-01", "2021-01-01")).unwrap(), "after the max: skipped");
        assert!(!can_match(t, &range("2010-01-01", "2014-12-31")).unwrap(), "before the min: skipped");
        assert!(!can_match(t, &json!({"range": {"e": {"gte": "now-1d"}}})).unwrap(), "no value at all: skipped");
    }

    /// A cluster's searched-shard count above `pre_filter_shard_size` makes
    /// a one-shard engine pre-filter; its range probe must not recurse.
    #[test]
    fn coordinator_prefilter_count_skips_a_disjoint_shard_without_recursing() {
        use crate::index::WriteReq;
        use crate::node::{Node, ResolveOpts};
        use crate::shard::WriteCond;
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("t", &json!({
            "settings": {"number_of_shards": 1, "index.blocks.write": true},
            "mappings": {"properties": {"created_at": {"type": "date", "format": "yyyy-MM-dd"}}}
        })).unwrap();
        let w = vec![WriteReq::Index { id: Some("1".into()), source: json!({"created_at": "2019-05-01"}).to_string(), routing: None, cond: WriteCond::default() }];
        let _ = svc.write(w, true);
        svc.refresh().unwrap();
        let targets = node.resolve("t", ResolveOpts::default()).unwrap();
        let request = parse_request(&json!({"size": 0, "query": {"range": {"created_at": {"gte": "2016-01-01", "lt": "2018-02-01"}}}}), &HashMap::new()).unwrap();
        let _scope = ReadShardCountScope::enter(Some(2));
        let out = execute(&targets, &request, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        assert_eq!(v["_shards"]["skipped"], 1);
        // With aggregations, a part of a cluster search reports the raw count
        // (the coordinator keeps one shard of the whole search).
        let aggs = parse_request(&json!({"size": 0, "query": {"range": {"created_at": {"gte": "2016-01-01", "lt": "2018-02-01"}}}, "aggs": {"c": {"value_count": {"field": "created_at"}}}}), &HashMap::new()).unwrap();
        let out = execute(&targets, &aggs, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        assert_eq!(v["_shards"]["skipped"], 1);
    }

    #[test]
    fn global_aggregations_keep_alias_and_role_visibility_on_all_shards() {
        use crate::index::WriteReq;
        use crate::shard::WriteCond;
        use crate::node::{Node, ResolveOpts};
        use crate::security::roles::{FieldAccess, FieldPermissions, IndexAccess};
        for shards in [1, 3] {
            let dir = tempfile::tempdir().unwrap();
            let node = Node::open(dir.path()).unwrap();
            let svc = node.create_index("docs", &json!({
                "settings": {"number_of_shards": shards},
                "mappings": {"properties": {"id": {"type": "long"}, "tag": {"type": "keyword"}}},
                "aliases": {"visible": {"filter": {"term": {"tag": "t1"}}}}
            })).unwrap();
            let writes = (0..30).map(|i| WriteReq::Index { id: Some(i.to_string()),
                source: json!({"id": i, "tag": format!("t{}", i % 3)}).to_string(),
                routing: None, cond: WriteCond::default() }).collect();
            assert!(svc.write(writes, true).iter().all(Result::is_ok));
            svc.refresh().unwrap();
            let request = parse_request(&json!({"size": 0, "query": {"match_none": {}},
                "aggs": {"all": {"global": {}, "aggs": {"ids": {"value_count": {"field": "id"}}}}}
            }), &HashMap::new()).unwrap();
            let mut targets = node.resolve("visible", ResolveOpts::default()).unwrap();
            let check = |targets: &[Target], expected: u64| {
                let output = execute(targets, &request, &ExecOpts { terms_lookup: None, cursors: None,
                    max_buckets: 65536, scroll_id: None, pit_id: None }).unwrap();
                let value: Value = serde_json::from_str(&output.json).unwrap();
                assert_eq!(value["hits"]["total"]["value"], 0);
                assert_eq!(value["aggregations"]["all"]["doc_count"], expected);
                assert_eq!(value["aggregations"]["all"]["ids"]["value"], expected);
            };
            check(&targets, 10);
            targets[0].access = Some(Arc::new(IndexAccess {
                dls: Some(json!({"range": {"id": {"gte": 5, "lte": 20}}})),
                fields: Some(FieldAccess::from_layer(vec![FieldPermissions {
                    grant: vec!["id".into()], except: Vec::new()
                }]))
            }));
            check(&targets, 5);
            let mut direct = node.resolve("docs", ResolveOpts::default()).unwrap();
            check(&direct, 30);
            direct[0].access = targets[0].access.clone();
            check(&direct, 16);
        }
    }

    #[test]
    fn knn_role_prefilter_resolves_fls_hidden_fields_without_exposing_them() {
        use crate::index::WriteReq;
        use crate::shard::WriteCond;
        use crate::node::Node;
        use crate::security::roles::{FieldAccess, FieldPermissions, IndexAccess};
        for mutable in [false, true] {
            #[cfg(not(feature = "mutable-hnsw"))]
            if mutable { continue; }
            let dir = tempfile::tempdir().unwrap();
            let node = Node::open(dir.path()).unwrap();
            let svc = node.create_index("vectors", &json!({
                "settings": {"number_of_shards": 2}, "mappings": {"properties": {
                    "tenant": {"type": "keyword"}, "label": {"type": "keyword"},
                    "v": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm",
                        "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100}}
                }}
            })).unwrap();
            #[cfg(feature = "mutable-hnsw")]
            if mutable {
                for shard in svc.local_shards() { shard.shard.core.write().enable_mutable_vectors().unwrap(); }
            }
            let writes = (0..20).map(|i| WriteReq::Index { id: Some(i.to_string()),
                source: json!({"v": [i as f32, 0.0], "label": i.to_string(),
                    "tenant": if i >= 10 { "allowed" } else { "denied" }}).to_string(),
                routing: None, cond: WriteCond::default() }).collect();
            assert!(svc.write(writes, true).iter().all(Result::is_ok));
            svc.refresh().unwrap();
            let mut target = Target::new(svc.clone());
            target.access = Some(Arc::new(IndexAccess {
                dls: Some(json!({"term": {"tenant": "allowed"}})),
                fields: Some(FieldAccess::from_layer(vec![FieldPermissions {
                    grant: vec!["v".into(), "label".into()], except: Vec::new()
                }]))
            }));
            for persisted in [false, true] {
                if persisted { svc.flush().unwrap(); }
                for top_level in [false, true] {
                    for user_hidden_filter in [false, true] {
                        let mut knn = json!({"field": "v", "query_vector": [0.0, 0.0], "k": 3, "num_candidates": 20});
                        if user_hidden_filter { knn["filter"] = json!({"term": {"tenant": "allowed"}}); }
                        let body = if top_level { json!({"size": 3, "knn": knn}) }
                            else { json!({"size": 3, "query": {"knn": knn}}) };
                        let request = parse_request(&body, &HashMap::new()).unwrap();
                        let output = execute(std::slice::from_ref(&target), &request, &ExecOpts {
                            terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None }).unwrap();
                        let value: Value = serde_json::from_str(&output.json).unwrap();
                        let hits = value["hits"]["hits"].as_array().unwrap();
                        let ids: Vec<_> = hits.iter().map(|h| h["_id"].as_str().unwrap()).collect();
                        let expected = if user_hidden_filter { vec![] } else { vec!["10", "11", "12"] };
                        assert_eq!(ids, expected, "mutable={mutable}, persisted={persisted}, top_level={top_level}");
                        assert!(hits.iter().all(|h| h["_source"].get("tenant").is_none()));
                    }
                }
            }
        }
    }

    #[test]
    fn metadata_only_fetch_keeps_stored_fallbacks_on_memtables_and_segments() {
        for metadata in [None, Some(ROUTING_FIELD), Some(IGNORED_FIELD)] {
            let dir = tempfile::tempdir().unwrap();
            let mut index = Index::create(dir.path(), "fetch", crate::mapping::reserved_core_fields(), IndexSettings::default()).unwrap();
            let mut d = Document::new();
            d.add_keyword(ID_FIELD, "doc-id");
            d.add_keyword(SOURCE_FIELD, "{\"value\":1}");
            d.add_i64(VERSION_FIELD, 7);
            if let Some(field) = metadata { d.add_keyword(field, "metadata"); }
            index.add_document(&d).unwrap();
            index.refresh().unwrap();
            // Read the refresh memtable, then commit and reopen the segment.
            for persisted in [false, true] {
                if persisted { index.commit().unwrap(); index = Index::open(dir.path()).unwrap(); }
                let leaves = index.leaves();
                let leaf = &leaves[0];
                let req = SearchRequest { source: SourceFilter::None, ..Default::default() };
                let fetched = fetch_document(leaf, 0, &req, None).unwrap();
                assert_eq!(fetched.get(ID_FIELD).and_then(|v| v.as_text()), Some("doc-id"));
                // `_ignored` comes from its column; `_routing` is stored only.
                assert_eq!(fetched.get(SOURCE_FIELD).is_some(), metadata == Some(ROUTING_FIELD));
                if let Some(field) = metadata { assert_eq!(fetched.get(field).and_then(|v| v.as_text()), Some("metadata")); }
                // Doc values do not need the stored document either.
                let dv = SearchRequest { docvalue_fields: vec![("value".into(), None)], ..req.clone() };
                assert_eq!(fetch_document(leaf, 0, &dv, None).unwrap().get(SOURCE_FIELD).is_some(), metadata == Some(ROUTING_FIELD));
                let requests = [
                    SearchRequest::default(),
                    SearchRequest { version: true, ..req.clone() },
                    SearchRequest { seq_no_primary_term: true, ..req.clone() },
                    SearchRequest { stored_fields: vec!["_source".into()], ..req.clone() },
                    SearchRequest { fields: vec![("value".into(), None)], ..req.clone() },
                    SearchRequest { highlight: Some(Map::new()), ..req.clone() },
                    SearchRequest { script_fields: Some(json!({})), ..req.clone() },
                    SearchRequest { runtime_mappings: Some(json!({})), ..req.clone() },
                    SearchRequest { force_synthetic_source: true, ..req.clone() },
                ];
                for request in requests { assert!(fetch_document(leaf, 0, &request, None).unwrap().get(SOURCE_FIELD).is_some()); }
            }
        }
    }
}

#[cfg(test)]
mod score_display_tests {
    use super::*;
    fn req(sort: Value) -> SearchRequest {
        parse_request(&serde_json::json!({ "sort": sort }), &HashMap::new()).unwrap()
    }
    // ES 9.5.5: a `_score` sort key shows hit scores; `max_score` stays
    // null unless the order is relevance (`_score` desc alone).
    #[test]
    fn score_sort_keys_show_hit_scores() {
        for (sort, hits, max) in [
            (serde_json::json!(["_score", "n"]), true, false),
            (serde_json::json!(["n", "_score"]), true, false),
            (serde_json::json!(["n"]), false, false),
            (serde_json::json!([{ "_score": "asc" }]), true, false),
            (serde_json::json!(["_score"]), true, true),
        ] {
            let r = req(sort.clone());
            assert_eq!(hit_scores_shown(&r), hits, "{sort}");
            assert_eq!(!sorted_by_field(&r), max, "{sort}");
            assert_eq!(scores_needed(&r), hits, "{sort}");
        }
    }
}

#[cfg(test)]
mod shard_query_error_tests {
    use super::*;
    #[test]
    fn mapping_failures_remain_shard_local_through_bool_wrappers() {
        let mapping = EsError::query_shard("i", "uuid", "bad numeric term");
        let wrapped = EsError::x_content_parse("[bool] failed to parse field [should]").caused_by(mapping.clone());
        assert_eq!(shard_query_error(wrapped), mapping);
        let syntax = EsError::x_content_parse("[bool] failed to parse field [should]")
            .caused_by(EsError::new(400, "parsing_exception", "unknown query"));
        assert_eq!(shard_query_error(syntax.clone()), syntax);
    }
}

#[cfg(test)]
mod mixed_aggregation_failure_tests {
    use super::*;
    #[test]
    fn union_fallback_keeps_query_failures_and_shard_order_status() {
        for (query_index, aggregate_index, status) in [("qa", "qb", 400), ("rb", "ra", 429)] {
            let prior = vec![(query_index.to_string(), EsError::query_shard(query_index, "uuid", "bad numeric term"))];
            let error = union_aggregation_failure([query_index, aggregate_index], &prior,
                EsError::new(429, "circuit_breaking_exception", "request limit"));
            assert_eq!(error.status, status);
            let body = error.to_json();
            let failed = body["error"]["failed_shards"].as_array().unwrap();
            assert_eq!(failed.len(), 2);
            assert!(failed.iter().any(|f| f["index"] == query_index && f["reason"]["type"] == "query_shard_exception"));
            assert!(failed.iter().any(|f| f["index"] == aggregate_index && f["reason"]["type"] == "circuit_breaking_exception"));
        }
    }
}

#[cfg(test)]
mod shard_query_failure_tests {
    use super::*;
    #[test]
    fn query_construction_limits_fail_as_shard_failures() {
        for e in [
            EsError::new(400, "too_many_nested_clauses", "Query contains too many nested clauses; maxClauseCount is set to 5"),
            crate::query::limits::check_regex("regexp", "abc", 2).unwrap_err(),
            EsError::new(429, "circuit_breaking_exception", "[script] Too many dynamic script compilations within, max: [1/1m]; please use indexed, or scripts with parameters instead; this limit can be changed by the [script.max_compilations_rate] setting"),
        ] {
            let status = e.status;
            let wrapped = wrap_shard_err("idx", e);
            assert_eq!((wrapped.kind.as_str(), wrapped.status), ("search_phase_execution_exception", status));
        }
        assert_eq!(wrap_shard_err("idx", EsError::illegal_argument("other")).kind, "illegal_argument_exception");
    }
}

#[cfg(test)]
mod agg_leaf_docs_tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use crate::mapping::Mappings;
    use rustic_core::types::{DocAddress, IndexSettings};

    fn index(n: usize) -> rustic_core::Index {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&json!({"properties": {"n": {"type": "long"}}}), &reg).unwrap();
        m.assign_core_ids();
        let lookup = FieldLookup::build(&m, &reg);
        let mut index = rustic_core::Index::new("idx", lookup.core_schema(), IndexSettings::default());
        for i in 0..n {
            let (pd, _) = crate::document::parse_with_dynamic(&json!({"n": i}), &i.to_string(), &m, &lookup, 1000).unwrap();
            index.add_document(&pd.doc).unwrap();
            if i == n * 2 / 3 {
                index.refresh().unwrap();
            }
        }
        index
    }

    /// The per-doc collection [`agg_leaf_docs`] replaces for whole matches.
    fn per_doc(query: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext) -> (Vec<DocId>, Vec<f32>) {
        let (mut docs, mut scores) = (Vec::new(), Vec::new());
        if let Some(mut s) = query.scorer(leaf, ctx) {
            let live = leaf.live_docs();
            loop {
                let d = s.advance();
                if d == rustic_core::query::TERMINATED { break; }
                if live(d) { docs.push(d); scores.push(s.score()); }
            }
        }
        (docs, scores)
    }

    /// A cancelled search task collects nothing more (ES's
    /// `task_cancelled_exception`); past its `timeout`, collection keeps
    /// what it has and the search is marked timed out.
    #[test]
    fn collection_stops_once_cancelled_or_past_the_timeout() {
        let idx = index(3000);
        let leaves = idx.leaves();
        let q = Query::MatchAll;
        let ctx = SearchContext::for_query(&q, &leaves, Bm25Params::default());
        let mem = crate::memory::RequestMemory::new();
        let gone = rustic_core::cancel::Token::cancelled_with("by user request");
        let e = rustic_core::cancel::with_search(Some(gone), || agg_leaf_docs(&q, &leaves[0], &ctx, None, &mem)).unwrap_err();
        assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "task_cancelled_exception", "task cancelled [by user request]"));
        let live = rustic_core::cancel::Token::new();
        let all = rustic_core::cancel::with_search(Some(live), || agg_leaf_docs(&q, &leaves[0], &ctx, None, &mem)).unwrap();
        assert_eq!(all.0.len(), leaves[0].num_docs() as usize);
        let passed = AggDeadline::new(Some(Instant::now()));
        let got = {
            let _scope = passed.enter();
            agg_leaf_docs(&q, &leaves[0], &ctx, None, &mem).unwrap()
        };
        assert!(got.0.is_empty() && passed.hit(), "collection past the deadline");
        let ahead = AggDeadline::new(Some(Instant::now() + Duration::from_secs(600)));
        let got = {
            let _scope = ahead.enter();
            agg_leaf_docs(&q, &leaves[0], &ctx, None, &mem).unwrap()
        };
        assert_eq!(got.0.len(), leaves[0].num_docs() as usize);
        assert!(!ahead.hit());
        assert!(!AggDeadline::current().is_set(), "the scope restores the thread's deadline");
    }

    /// Wrapped and plain `match_all`s, on a segment and a memtable, with and
    /// without deletes: the filled doc sets and scores equal the per-doc
    /// collection, and the breaker is charged the same 8 bytes a doc.
    #[test]
    fn match_all_fill_equals_per_doc_collection() {
        let mut idx = index(7000);
        let queries = [
            Query::MatchAll,
            Query::Boost { query: Box::new(Query::MatchAll), boost: 2.5 },
            Query::ConstScore { query: Box::new(Query::MatchAll), score: 0.75, point_set: false },
        ];
        for deletes in [false, true] {
            if deletes {
                idx.delete_doc(&DocAddress { segment_id: 0, doc_id: 17 }).unwrap();
                idx.delete_doc(&DocAddress { segment_id: 1, doc_id: 3 }).unwrap();
            }
            let leaves = idx.leaves();
            assert_eq!(leaves.len(), 2);
            for q in &queries {
                assert!(q.is_match_all());
                let ctx = SearchContext::for_query(q, &leaves, Bm25Params::default());
                for leaf in &leaves {
                    let mem = crate::memory::RequestMemory::new();
                    let got = agg_leaf_docs(q, leaf, &ctx, None, &mem).unwrap();
                    assert_eq!(got, per_doc(q, leaf, &ctx), "{q:?} deletes={deletes}");
                    assert_eq!(mem.level(), got.0.len() as u64 * 8, "{q:?}");
                    if deletes {
                        assert!(got.0.len() < leaf.num_docs() as usize);
                    } else {
                        assert_eq!(got.0, (0..leaf.num_docs()).collect::<Vec<_>>());
                    }
                }
            }
        }
        // `min_score` keeps the per-doc filter.
        let leaves = idx.leaves();
        let ctx = SearchContext::for_query(&Query::MatchAll, &leaves, Bm25Params::default());
        let mem = crate::memory::RequestMemory::new();
        assert!(agg_leaf_docs(&Query::MatchAll, &leaves[0], &ctx, Some(2.0), &mem).unwrap().0.is_empty());
    }

    /// Constant-score sets held whole (geo distance, its constant-score
    /// wrapper, a distance ring; the KD walk's bitset or doc list), on
    /// segments and a memtable, with and without deletes and `min_score`:
    /// the bulk collection equals the per-doc one, charged 8 bytes a doc.
    #[test]
    fn constant_set_bulk_collection_equals_per_doc_collection() {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&json!({"properties": {"g": {"type": "geo_point"}}}), &reg).unwrap();
        m.assign_core_ids();
        let lookup = FieldLookup::build(&m, &reg);
        let mut idx = rustic_core::Index::new("idx", lookup.core_schema(), IndexSettings::default());
        for i in 0..9000u32 {
            let (lat, lon) = (52.0 + (i % 97) as f64 * 0.07, 3.0 + (i % 89) as f64 * 0.09);
            let (pd, _) = crate::document::parse_with_dynamic(&json!({"g": [lon, lat]}), &i.to_string(), &m, &lookup, 1000).unwrap();
            idx.add_document(&pd.doc).unwrap();
            if i == 3000 || i == 6000 {
                idx.refresh().unwrap();
            }
        }
        let g = lookup.get("g").unwrap().geo_id.unwrap();
        let dist = |m: f64| Query::GeoDistance { field: g, lat: 55.0, lon: 7.0, meters: m };
        let zero = |q: Query| Query::ConstScore { query: Box::new(q), score: 0.0, point_set: false };
        let queries = [
            dist(200_000.0),
            Query::ConstScore { query: Box::new(dist(300_000.0)), score: 0.5, point_set: false },
            Query::Bool { must: vec![zero(dist(200_000.0)), zero(Query::Bool { must: vec![], should: vec![], minimum_should_match: rustic_core::query::MinShouldMatch::Count(0), filter: vec![Query::GeoBoundingBox { field: g, top: 90.0, bottom: -90.0, left: -180.0, right: 180.0 }], must_not: vec![dist(100_000.0)] })], should: vec![], minimum_should_match: rustic_core::query::MinShouldMatch::Count(0), filter: vec![], must_not: vec![] },
        ];
        for deletes in [false, true] {
            if deletes {
                idx.delete_doc(&DocAddress { segment_id: 0, doc_id: 17 }).unwrap();
                idx.delete_doc(&DocAddress { segment_id: 1, doc_id: 3 }).unwrap();
            }
            let leaves = idx.leaves();
            for q in &queries {
                let ctx = SearchContext::for_query(q, &leaves, Bm25Params::default());
                for leaf in &leaves {
                    let mem = crate::memory::RequestMemory::new();
                    let got = agg_leaf_docs(q, leaf, &ctx, None, &mem).unwrap();
                    let want = per_doc(q, leaf, &ctx);
                    assert_eq!(got, want, "{q:?} deletes={deletes}");
                    assert_eq!(mem.level(), got.0.len() as u64 * 8, "{q:?}");
                    for ms in [0.25f32, 1.0, 2.0] {
                        let mem = crate::memory::RequestMemory::new();
                        let got = agg_leaf_docs(q, leaf, &ctx, Some(ms), &mem).unwrap();
                        let keep: Vec<usize> = (0..want.0.len()).filter(|&i| want.1[i] >= ms).collect();
                        assert_eq!(got.0, keep.iter().map(|&i| want.0[i]).collect::<Vec<_>>(), "{q:?} min_score={ms}");
                    }
                }
            }
        }
    }

    /// Timing of the aggregation doc collection over a pmc shard
    /// (`PMC_AGG_DOCS`, default 115,000): fill against per-doc scoring.
    /// Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn match_all_fill_timing() {
        let n: usize = std::env::var("PMC_AGG_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(115_000);
        let mut idx = index(n);
        idx.refresh().unwrap();
        let leaves = idx.leaves();
        let ctx = SearchContext::for_query(&Query::MatchAll, &leaves, Bm25Params::default());
        let time = |f: &dyn Fn() -> usize| {
            let mut t: Vec<_> = (0..21).map(|_| { let s = std::time::Instant::now(); assert!(f() > 0); s.elapsed() }).collect();
            t.sort();
            t[10]
        };
        let fill = time(&|| leaves.iter().map(|l| agg_leaf_docs(&Query::MatchAll, l, &ctx, None, &crate::memory::RequestMemory::new()).unwrap().0.len()).sum());
        let loop_ = time(&|| leaves.iter().map(|l| per_doc(&Query::MatchAll, l, &ctx).0.len()).sum());
        eprintln!("agg doc collection over {n} docs: fill {fill:?}, per-doc {loop_:?}");
    }
}
