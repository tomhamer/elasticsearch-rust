//! Hook 9: the shard-level query phase and fetch phase of `_search`, so a
//! coordinator can run `query_phase` on every shard copy, merge the shard
//! results, then `fetch` only the winning hits (ES's query-then-fetch).
//!
//! * [`IndexService::open_searcher`] pins the index's segments; doc refs
//!   returned by [`IndexService::query_phase`] / `knn_candidates` are valid
//!   for [`IndexService::fetch`] on the same [`SearcherHandle`] until it is
//!   released (also usable as a scroll / PIT context).
//! * [`QueryPhaseResult`] carries hits **without** `_source` / `fields`
//!   rendering, totals, the aggregation partial ([`AggPartial`], hook 10)
//!   and the rendered `suggest` section.
//! * [`merge_query_phases`] is the coordinator's sorted merge (sort values,
//!   collapse, `from` / `size`, total tracking).
//! * [`IndexService::fetch`] renders exactly what the single-node `_search`
//!   renders for a hit (source filtering, `fields`, highlight, `inner_hits`,
//!   `matched_queries`, `sort`).
//!
//! Compiled as a private child module of `crate::search` so it shares the
//! single-node path's query translation and hit rendering.

use super::*;
use crate::hooks::agg_partial::{AggPartial, AggRequest};
use crate::index::IndexShard;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static SEARCHER_SEQ: AtomicU64 = AtomicU64::new(1);

/// A sort value as the shard computed it (serializable twin of
/// `rustic_core::query::SortValue`; `F64` keeps non-finite values exact).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ShardSortValue {
    I64(i64),
    /// IEEE-754 bits, so `Infinity` / `NaN` survive JSON.
    F64(u64),
    Str(String),
    Missing,
}

impl ShardSortValue {
    pub fn f64(v: f64) -> Self {
        ShardSortValue::F64(v.to_bits())
    }
}

impl From<&SortValue> for ShardSortValue {
    fn from(v: &SortValue) -> Self {
        match v {
            SortValue::I64(i) => ShardSortValue::I64(*i),
            SortValue::F64(f) => ShardSortValue::F64(f.to_bits()),
            SortValue::Str(s) => ShardSortValue::Str(s.clone()),
            SortValue::Missing => ShardSortValue::Missing,
        }
    }
}

impl From<&ShardSortValue> for SortValue {
    fn from(v: &ShardSortValue) -> Self {
        match v {
            ShardSortValue::I64(i) => SortValue::I64(*i),
            ShardSortValue::F64(b) => SortValue::F64(f64::from_bits(*b)),
            ShardSortValue::Str(s) => SortValue::Str(s.clone()),
            ShardSortValue::Missing => SortValue::Missing,
        }
    }
}

/// Address of a document within a pinned searcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DocRef {
    /// [`SearcherHandle::id`] the ref belongs to.
    pub searcher: u64,
    /// Leaf (segment) index in the pinned searcher.
    pub leaf: u32,
    /// Leaf-local doc id.
    pub doc: u32,
}

/// One shard-level hit: no rendering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardHit {
    pub doc_ref: DocRef,
    pub score: f32,
    /// One value per `sort` criterion (`[]` when the request has no sort).
    pub sort_values: Vec<ShardSortValue>,
    /// `collapse` group key (`group_key` of the collapse field's values).
    pub collapse_key: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TotalRelation {
    Eq,
    Gte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TotalHits {
    pub value: u64,
    pub relation: TotalRelation,
}

/// What a shard returns from the query phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryPhaseResult {
    pub hits: Vec<ShardHit>,
    /// Matching docs (`Gte` when a limit stopped counting). Meaningless when
    /// the request disables total tracking (`track_total_hits: false`).
    pub total: TotalHits,
    /// Best score over the collected hits (`None` when no hits / no scores).
    pub max_score: Option<f32>,
    /// Aggregation partial (when the request has `aggs`).
    pub aggs: Option<AggPartial>,
    /// The rendered `suggest` section (merge by ES `Suggest.reduce` rules).
    pub suggest: Option<Value>,
    pub profile: Option<Value>,
    pub timed_out: bool,
    pub terminated_early: bool,
}

/// Options of [`IndexService::query_phase`].
#[derive(Clone, Copy)]
pub struct QueryPhaseOpts<'a> {
    pub terms_lookup: Option<TermsLookupFn<'a>>,
    pub max_buckets: usize,
    /// `now` for date math; give every shard of one search the same value.
    pub now: Option<i64>,
    /// Skip pipeline aggregations (the coordinator runs them after reduce).
    pub skip_pipeline: bool,
    /// DFS kNN: the globally merged kNN winners
    /// ([`merge_knn_candidates`]); the request's `knn` clauses then match
    /// exactly these docs (those of this shard's searcher) with these scores.
    pub knn_fixed: Option<&'a [ShardKnnHit]>,
    /// `dfs_query_then_fetch`: term / field statistics merged over every
    /// shard ([`merge_dfs`] of each shard's [`IndexShard::dfs_stats`]);
    /// scoring uses them instead of this shard's own.
    pub dfs: Option<&'a DfsStats>,
}

impl Default for QueryPhaseOpts<'_> {
    fn default() -> Self {
        QueryPhaseOpts { terms_lookup: None, max_buckets: 65_536, now: None, skip_pipeline: true, knn_fixed: None, dfs: None }
    }
}

/// Statistics of one term over the shards a DFS phase saw.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DfsTerm {
    /// Field name (shards and indices may number fields differently).
    pub field: String,
    pub term: Vec<u8>,
    pub doc_freq: u64,
    pub total_term_freq: u64,
    /// Docs with the term's field (Lucene field `docCount`).
    pub field_doc_count: u64,
}

/// Statistics of one field over the shards a DFS phase saw.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DfsField {
    pub field: String,
    pub doc_count: u64,
    pub sum_total_term_freq: u64,
}

/// The DFS phase result of `dfs_query_then_fetch` (ES `DfsSearchResult`):
/// the statistics of the query's terms and fields on one shard, or merged
/// over all shards with [`merge_dfs`]. Serializable.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DfsStats {
    pub terms: Vec<DfsTerm>,
    pub fields: Vec<DfsField>,
}

/// Immutable admitted statistics supplied by a coordinator. This owner is
/// retained in SearchRequest/reader contexts; converting to a naked Arc would
/// detach the allocation charge while a scroll still uses its fields.
#[derive(Clone,Debug)]
pub struct AdmittedDfsStats(Arc<OwnedDfsStats>);
#[derive(Debug)]
struct OwnedDfsStats { value:DfsStats, _memory:crate::memory::Reservation }
impl std::ops::Deref for AdmittedDfsStats {type Target=DfsStats;fn deref(&self)->&DfsStats{&self.0.value}}
impl PartialEq for AdmittedDfsStats {fn eq(&self,other:&Self)->bool{self.0.value==other.0.value}}
impl serde::Serialize for AdmittedDfsStats {fn serialize<S:serde::Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{self.0.value.serialize(s)}}
impl AdmittedDfsStats {
    /// The caller admitted every string/array before constructing value.
    pub fn from_reserved(value:DfsStats,mut memory:crate::memory::Reservation)->Result<Self>{
        memory.grow((std::mem::size_of::<OwnedDfsStats>()+2*std::mem::size_of::<usize>())as u64,"DFS result owner")?;
        Ok(Self(Arc::new(OwnedDfsStats{value,_memory:memory.into_result_owner()})))
    }
}

/// Sum shard statistics (coordinator side of the DFS phase).
pub fn merge_dfs(parts: &[DfsStats]) -> DfsStats {
    let mut terms: std::collections::BTreeMap<(String, Vec<u8>), (u64, u64, u64)> = Default::default();
    let mut fields: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
    for p in parts {
        for t in &p.terms {
            let e = terms.entry((t.field.clone(), t.term.clone())).or_default();
            e.0 += t.doc_freq;
            e.1 += t.total_term_freq;
            e.2 += t.field_doc_count;
        }
        for f in &p.fields {
            let e = fields.entry(f.field.clone()).or_default();
            e.0 += f.doc_count;
            e.1 += f.sum_total_term_freq;
        }
    }
    DfsStats {
        terms: terms.into_iter().map(|((field, term), (df, ttf, dc))| DfsTerm { field, term, doc_freq: df, total_term_freq: ttf, field_doc_count: dc }).collect(),
        fields: fields.into_iter().map(|(field, (dc, sum))| DfsField { field, doc_count: dc, sum_total_term_freq: sum }).collect(),
    }
}

/// The statistics a search context gathered (one shard's DFS result).
pub(super) fn dfs_of_ctx(ctx: &SearchContext, lookup: &FieldLookup) -> DfsStats {
    let names: HashMap<u16, String> = lookup.core_schema().into_iter().map(|f| (f.id, f.name)).collect();
    let mut out = DfsStats::default();
    for (t, st, _) in &ctx.term_infos {
        if let Some(n) = names.get(&t.field) {
            out.terms.push(DfsTerm { field: n.clone(), term: t.text.clone(), doc_freq: st.doc_freq, total_term_freq: st.total_term_freq, field_doc_count: st.num_docs });
        }
    }
    for (t, st) in &ctx.term_stats {
        if let Some(n) = names.get(&t.field) {
            if !out.terms.iter().any(|x| x.field == *n && x.term == t.text) {
                out.terms.push(DfsTerm { field: n.clone(), term: t.text.clone(), doc_freq: st.doc_freq, total_term_freq: st.total_term_freq, field_doc_count: st.num_docs });
            }
        }
    }
    for (f, st) in &ctx.field_stats {
        if let Some(n) = names.get(f) {
            out.fields.push(DfsField { field: n.clone(), doc_count: st.num_docs, sum_total_term_freq: st.sum_field_len });
        }
    }
    out.terms.sort_by(|a, b| a.field.cmp(&b.field).then_with(|| a.term.cmp(&b.term)));
    out.fields.sort_by(|a, b| a.field.cmp(&b.field));
    out
}

/// Replace a search context's shard-local statistics with the merged DFS
/// statistics (terms / fields the DFS phase did not see keep local ones).
pub(super) fn apply_dfs(ctx: &mut SearchContext, dfs: &DfsStats, lookup: &FieldLookup) {
    let ids: HashMap<String, u16> = lookup.core_schema().into_iter().map(|f| (f.name, f.id)).collect();
    let terms: HashMap<(u16, &[u8]), &DfsTerm> = dfs.terms.iter().filter_map(|t| ids.get(&t.field).map(|id| ((*id, t.term.as_slice()), t))).collect();
    let conv = |d: &DfsTerm| rustic_core::query::reader::TermStats { doc_freq: d.doc_freq, total_term_freq: d.total_term_freq, num_docs: d.field_doc_count };
    for (t, st, _) in ctx.term_infos.iter_mut() {
        if let Some(d) = terms.get(&(t.field, t.text.as_slice())) {
            *st = conv(d);
        }
    }
    for (t, st) in ctx.term_stats.iter_mut() {
        if let Some(d) = terms.get(&(t.field, t.text.as_slice())) {
            *st = conv(d);
        }
    }
    for f in &dfs.fields {
        if let Some(id) = ids.get(&f.field) {
            if let Some(st) = ctx.field_stats.get_mut(id) {
                *st = rustic_core::query::reader::FieldStats { num_docs: f.doc_count, sum_field_len: f.sum_total_term_freq };
            }
        }
    }
}

/// Run `hit_q` over `leaves`, scoring with `dfs` statistics when given.
pub(super) fn search_with_dfs(leaves: &[LeafReader<'_>], hit_q: &Query, so: &SearchOptions, dfs: Option<&DfsStats>, lookup: &FieldLookup) -> Result<TopDocs> {
    let sr = Searcher::new(leaves.to_vec());
    let mut ctx = sr.context(hit_q, so);
    ctx.execution_result()?;
    if let Some(d) = dfs { apply_dfs(&mut ctx, d, lookup); }
    let top = sr.search_with_context(hit_q, so, &ctx);
    ctx.execution_result()?;
    Ok(top)
}

/// One kNN candidate of a shard (hook 11, DFS phase).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardKnnHit {
    /// Index of the `knn` clause in the request (`knn` array order).
    pub clause: u32,
    pub doc_ref: DocRef,
    /// Final kNN score (similarity mapped to the ES score, `boost` applied).
    pub score: f32,
}

/// Docs with fixed scores (the global kNN winners living on this shard).
#[derive(Debug)]
pub(super) struct FixedDocs {
    by_leaf: HashMap<usize, Vec<(DocId, f32)>>,
}

impl rustic_core::query::query::CustomQuery for FixedDocs {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn rustic_core::query::Scorer>> {
        let live = leaf.live_docs();
        let docs: Vec<(DocId, f32)> = self.by_leaf.get(&leaf.id())?.iter().copied().filter(|(d, _)| live(*d)).collect();
        if docs.is_empty() {
            None
        } else {
            Some(Box::new(rustic_core::query::query::ScoredDocs::new(docs)))
        }
    }
    fn name(&self) -> &'static str {
        "knn_fixed"
    }
}

/// One rendered hit (`{"_index": ..., "_id": ..., ...}`), exactly the JSON
/// the single-node `_search` puts in `hits.hits`.
pub type HitJson = Box<serde_json::value::RawValue>;

/// A pinned view of one shard copy's segments (point-in-time searcher).
pub struct SearcherHandle {
    id: u64,
    /// The shard copy the segments belong to.
    shard: Arc<IndexShard>,
    /// A view service over exactly that copy (what search targets hold).
    svc: Arc<IndexService>,
    alias_filter: Option<Value>,
    /// The pinned reader view (`None` once released).
    segs: Mutex<Option<Arc<ReaderView>>>,
}

use parking_lot::Mutex;

impl SearcherHandle {
    /// Unique (per process) id; stored in every [`DocRef`] it produces.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The index (a view exposing only this handle's shard copy).
    pub fn index(&self) -> &Arc<IndexService> {
        &self.svc
    }

    /// The shard copy the handle pins.
    pub fn shard(&self) -> &Arc<IndexShard> {
        &self.shard
    }

    /// Number of pinned segments.
    pub fn num_leaves(&self) -> usize {
        self.segs.lock().as_ref().map(|s| s.num_leaves()).unwrap_or(0)
    }

    /// Release the pinned segments now. Later phases on this handle fail.
    pub fn release(&self) {
        *self.segs.lock() = None;
    }

    pub fn is_released(&self) -> bool {
        self.segs.lock().is_none()
    }

    fn target(&self) -> Result<Target> {
        let segs = self.segs.lock().clone().ok_or_else(|| EsError::new(404, "search_context_missing_exception", format!("No search context found for id [{}]", self.id)))?;
        Ok(Target { snapshot: Some(segs), alias_filter: self.alias_filter.clone(), shard: Some(self.shard.clone()), ..Target::new(self.svc.clone()) })
    }

    /// A [`Target`] over the pinned segments (runs the single-node path
    /// against exactly this view).
    pub fn as_target(&self) -> Result<Target> {
        self.target()
    }
}

impl Drop for SearcherHandle {
    fn drop(&mut self) {
        self.release();
    }
}

impl std::fmt::Debug for SearcherHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearcherHandle({}, [{}][{}], {} leaves)", self.id, self.shard.name, self.shard.id, self.num_leaves())
    }
}

impl IndexShard {
    /// Pin this copy's current reader view (refreshing first when a
    /// search-idle refresh was skipped), with an optional alias filter
    /// applied to every phase run through the handle.
    pub fn open_searcher(self: &Arc<Self>, alias_filter: Option<Value>) -> Result<SearcherHandle> {
        let _p = self.permit()?;
        self.sync_schema_for_search()?;
        // Pin what searches see now: the published reader view (segments and
        // published memtables). No seal and no shard lock.
        self.shard.before_search();
        let view = self.shard.reader_view();
        Ok(SearcherHandle { id: SEARCHER_SEQ.fetch_add(1, AtomicOrdering::Relaxed), shard: self.clone(), svc: IndexService::view_of(self), alias_filter, segs: Mutex::new(Some(view)) })
    }

    /// Hook 9 query phase on this copy (see [`IndexService::query_phase`]).
    pub fn query_phase(&self, handle: &SearcherHandle, req: &SearchRequest, opts: &QueryPhaseOpts) -> Result<QueryPhaseResult> {
        self.check_handle(handle)?;
        handle.svc.query_phase(handle, req, opts)
    }

    /// Hook 11 DFS kNN candidates on this copy.
    pub fn knn_candidates(&self, handle: &SearcherHandle, req: &SearchRequest, opts: &QueryPhaseOpts) -> Result<Vec<ShardKnnHit>> {
        self.check_handle(handle)?;
        handle.svc.knn_candidates(handle, req, opts)
    }

    /// Hook 9 fetch phase on this copy.
    pub fn fetch(&self, handle: &SearcherHandle, hits: &[ShardHit], req: &SearchRequest) -> Result<Vec<HitJson>> {
        self.check_handle(handle)?;
        handle.svc.fetch(handle, hits, req)
    }

    /// DFS phase of `dfs_query_then_fetch` on this copy: the statistics of
    /// the request's query terms and fields over the pinned segments.
    /// Merge every shard's with [`merge_dfs`] and pass the result as
    /// [`QueryPhaseOpts::dfs`].
    pub fn dfs_stats(&self, handle: &SearcherHandle, req: &SearchRequest) -> Result<DfsStats> {
        self.check_handle(handle)?;
        self.check_usable()?;
        let t = handle.target()?;
        with_request_env(&t, req, || {
            let leaves = leaves_of(&t);
            dfs_stats_over(&t, &leaves, req, crate::date::now_millis(), None)
        })
    }

    fn check_handle(&self, h: &SearcherHandle) -> Result<()> {
        if h.shard.uuid != self.uuid || h.shard.id != self.id {
            return Err(EsError::illegal_argument(format!("searcher [{}] belongs to shard [{}][{}], not [{}][{}]", h.id, h.shard.name, h.shard.id, self.name, self.id)));
        }
        Ok(())
    }
}

/// The DFS statistics of `req`'s query over `leaves` (one shard).
pub(super) fn dfs_stats_over(t: &Target, leaves: &[LeafReader<'_>], req: &SearchRequest, now: i64, terms_lookup: Option<TermsLookupFn<'_>>) -> Result<DfsStats> {
    let lookup = t.lookup();
    crate::script::enter_index(&t.svc.name, &lookup, || t.svc.mapping_runtime())?;
    let qctx = query_ctx(t, &lookup, leaves, now, terms_lookup, req);
    let (_, hit_q) = build_query(req, t, &qctx).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
    let ctx = SearchContext::for_query(&hit_q, leaves, Bm25Params::default());
    ctx.execution_result()?;
    Ok(dfs_of_ctx(&ctx, &lookup))
}

/// The query context of one run (as the single-node search builds it).
pub(super) fn query_ctx<'a>(t: &'a Target, lookup: &'a FieldLookup, leaves: &'a [LeafReader<'a>], now: i64, terms_lookup: Option<TermsLookupFn<'a>>, req: &SearchRequest) -> QueryContext<'a> {
    QueryContext {
        deadline: req.timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout)),
        lookup,
        index_name: &t.svc.name,
        index_uuid: &t.svc.uuid,
        aliases: t.svc.aliases().keys().cloned().collect(),
        now,
        leaves,
        default_fields: t.svc.query_default_fields(),
        terms_lookup,
        max_terms_count: t.svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: t.svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: t.svc.seq_no_disabled(),
        allow_unmapped_fields: t.svc.setting("index.query.parse.allow_unmapped_fields").map(|v| v != "false").unwrap_or(true),
        knn_prefilter: t.effective_filter(),
        knn_prefilter_lookup: Some(t.svc.lookup()),
        allow_expensive_queries: req.allow_expensive_queries && t.svc.setting("index.search.allow_expensive_queries").map(|v| v != "false").unwrap_or(true),
        mappings: Some(&**t.svc),
    }
}

impl IndexService {
    /// Pin the segments of a single-shard index (see
    /// [`IndexShard::open_searcher`]; an index with several shards needs
    /// one searcher per shard copy).
    pub fn open_searcher(self: &Arc<Self>) -> Result<SearcherHandle> {
        self.open_searcher_filtered(None)
    }

    /// [`open_searcher`](Self::open_searcher) with an alias filter applied to
    /// every phase run through the handle.
    pub fn open_searcher_filtered(self: &Arc<Self>, alias_filter: Option<Value>) -> Result<SearcherHandle> {
        let shards = self.local_shards();
        match shards.as_slice() {
            [one] if self.number_of_shards() == 1 => one.open_searcher(alias_filter),
            _ => Err(EsError::illegal_argument(format!("index [{}] has [{}] shards: open a searcher per shard copy (IndexShard::open_searcher)", self.name, self.number_of_shards()))),
        }
    }

    /// Hook 9, shard side of the query phase: run `req` over the pinned
    /// segments and return the top `from + size` hits (no `_source` / `fields`
    /// rendering), totals, the aggregation partial and the `suggest` section.
    pub fn query_phase(&self, handle: &SearcherHandle, req: &SearchRequest, opts: &QueryPhaseOpts) -> Result<QueryPhaseResult> {
        self.check_handle(handle)?;
        handle.shard.check_usable()?;
        let t = handle.target()?;
        with_request_env(&t, req, || query_phase_inner(&t, handle, req, opts))
    }

    /// Hook 9, fetch phase: render `hits` (from [`query_phase`](Self::query_phase)
    /// or `knn_candidates`) exactly as the single-node search renders them.
    /// `req` is the original request. Output order follows `hits`.
    pub fn fetch(&self, handle: &SearcherHandle, hits: &[ShardHit], req: &SearchRequest) -> Result<Vec<HitJson>> {
        self.check_handle(handle)?;
        handle.shard.check_usable()?;
        let t = handle.target()?;
        with_request_env(&t, req, || fetch_inner(&t, handle, hits, req))
    }

    fn check_handle(&self, h: &SearcherHandle) -> Result<()> {
        if h.svc.uuid != self.uuid {
            return Err(EsError::illegal_argument(format!("searcher [{}] belongs to index [{}], not [{}]", h.id, h.svc.name, self.name)));
        }
        Ok(())
    }
}

/// The script environment of a request around `f` (stored scripts, runtime
/// mappings, scorer errors), as `execute` does.
fn with_request_env<R>(t: &Target, req: &SearchRequest, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let stored = crate::script::stored::for_index_dir(t.svc.path());
    let env = crate::script::ScriptEnv::new(stored, req.runtime_mappings.clone());
    let first = t.svc.name.clone();
    crate::script::with_env(env, || {
        if let Some(sf) = &req.script_fields {
            crate::script::search::check_script_fields(sf, t.svc.setting_usize("index.max_script_fields", 32)).map_err(|e| wrap_script_err(&t.svc.name, e))?;
        }
        crate::collapse::check_request(req, false)?;
        let r = f();
        let err = crate::script::env().and_then(|e| e.take_error());
        match (r, err) {
            (_, Some(e)) => Err(EsError::search_phase(&first, e)),
            (Err(e), None) => Err(wrap_script_err(&first, e)),
            (ok, None) => ok,
        }
    })
}

fn leaves_of<'a>(t: &'a Target) -> Vec<LeafReader<'a>> {
    match &t.snapshot {
        Some(s) => s.leaves(),
        None => Vec::new(),
    }
}

/// Build the per-index run (query, sort, inner hits, named queries,
/// highlight terms) over `leaves`; with `collect_top` also the top docs.
#[allow(clippy::too_many_arguments)]
fn prepare<'a>(t: &Target, leaves: Vec<LeafReader<'a>>, req: &SearchRequest, terms_lookup: Option<TermsLookupFn<'_>>, now: i64, collect_top: bool, knn_fixed: Option<(u64, &[ShardKnnHit])>, dfs: Option<&DfsStats>) -> Result<IndexRun<'a>> {
    let lookup = t.svc.lookup();
    crate::script::enter_index(&t.svc.name, &lookup, || t.svc.mapping_runtime())?;
    let aliases: Vec<String> = t.svc.aliases().keys().cloned().collect();
    let default_fields = t.svc.query_default_fields();
    // Compound retrievers: collect every rank doc so score ties keep the
    // fused order (see the rank re-sort of the merged hits).
    let k = match &req.rank_plan {
        Some(p) if req.sort.is_empty() => (req.from + req.size).max(p.len()),
        _ => req.from + req.size,
    };
    let qctx = QueryContext {
        deadline: req.timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout)),
        lookup: &lookup,
        index_name: &t.svc.name,
        index_uuid: &t.svc.uuid,
        aliases,
        now,
        leaves: &leaves,
        default_fields,
        terms_lookup,
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
    let (q, hit_q) = build_query_with(req, t, &qctx, knn_fixed, None).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
    let (q, hit_q) = match req.index_boosts.iter().find(|(n, _)| n == &t.svc.name) {
        Some((_, b)) => (Query::Boost { query: Box::new(q), boost: *b }, Query::Boost { query: Box::new(hit_q), boost: *b }),
        None => (q, hit_q),
    };
    let sort = resolve_sort(&req.sort, &lookup, &t.svc.name, &t.svc.uuid, &req.sort_extra, &qctx).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
    let knn_ih = crate::vectors::inner_hits_query(req.query.as_ref(), &req.knn, req.size, &lookup);
    let inner = inner_hit_defs(knn_ih.as_ref().or(req.query.as_ref()).or(req.rank_plan.as_ref().map(|p| p.union())), &qctx, &leaves).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
    let top = if collect_top {
        let mut so = SearchOptions {
            top_k: k,
            from: 0,
            sort: sort.iter().map(|s| s.field.clone()).collect(),
            min_score: req.min_score,
            track_total_hits: req.track_total_hits.unwrap_or(TrackTotalHits::False),
            terminate_after: req.terminate_after,
            timeout: req.timeout,
            explain: false,
            ..Default::default()
        };
        so.track_scores = scores_needed(req);
        if let Some(c) = &req.collapse {
            if let Some(f) = lookup.get(c) {
                let kind = if f.ty.is_keyword_like() { SortKind::Keyword } else if f.ty.is_float() { SortKind::F64 } else { SortKind::I64 };
                if let Some(id) = f.core_id {
                    so.collapse_field = Some((id, kind));
                }
            }
        }
        if let Some(r) = &req.rescore {
            let mut steps = crate::search::rescore_steps(r, &qctx).map_err(|e| wrap_shard_err(&t.svc.name, e))?.into_iter();
            if let Some((first, mode)) = steps.next() {
                so.rescore = Some(first);
                so.rescore_mode = mode;
                so.rescore_chain = steps.collect();
            }
        }
        if let Some(after) = &req.search_after {
            if after.len() != req.sort.len() {
                return Err(EsError::search_phase(&t.svc.name, EsError::illegal_argument(format!("search_after has {} value(s) but sort has {}.", after.len(), req.sort.len()))));
            }
            let mut vals = Vec::new();
            let mut leaf = usize::MAX;
            let mut doc = u32::MAX;
            let mut score = f32::NEG_INFINITY;
            for (i, (v, s)) in after.iter().zip(&sort).enumerate() {
                if s.render == "doc" {
                    let g = v.as_i64().unwrap_or(0);
                    if g < 0 {
                        // A negative value is "before every doc" in sort order.
                        vals.push(SortValue::I64(if s.descending { i64::MAX } else { -1 }));
                    } else {
                        let g = if req.sort[i].field == "_shard_doc" {
                            match shard_doc_after(g, t.shard.as_ref().map(|s| s.id as i64).unwrap_or(0), s.descending) {
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
        let mut top = if k == 0 && so.track_total_hits == TrackTotalHits::False {
            TopDocs::default()
        } else if let Some(mut keys) = script_keys {
            crate::script::search::collect_sorted(&leaves, &hit_q, &mut keys, k, req.min_score, so.search_after.as_ref())?
        } else {
            search_with_dfs(&leaves, &hit_q, &so, dfs, &lookup)?
        };
        if req.rank_plan.is_some() {
            top = crate::retriever::fix_total(req, top, &leaves, &q)?;
        }
        if req.collapse.is_some() && req.track_scores && !req.sort.is_empty() {
            top.max_score = search_with_dfs(&leaves, &hit_q, &SearchOptions { top_k: 1, ..Default::default() }, None, &lookup)?.max_score;
        }
        top
    } else {
        TopDocs::default()
    };
    if let Some(error) = &top.execution_error {
        return Err(EsError::search_phase(&t.svc.name, EsError::illegal_argument(error.clone())));
    }
    let terms = match &req.highlight {
        Some(h) => Some(QueryTerms::for_index(&q, h, &t.svc, &|v| parse_query(v, &qctx)).map_err(|e| wrap_shard_err(&t.svc.name, e))?),
        None => None,
    };
    let mut named = Vec::new();
    for (name, spec, is_knn) in named_queries(req.query.as_ref(), &req.knn) {
        let nq = if is_knn { parse_knn_query(spec.as_object().unwrap(), &qctx, Some(req.size.max(1))) } else { parse_query(&spec, &qctx) };
        let Ok(nq) = nq else { continue };
        let nctx = SearchContext::for_query(&nq, &leaves, Bm25Params::default());
        nctx.execution_result()?;
        named.push((name, nq, nctx));
    }
    let visibility_query = aggregation_visibility(t, &qctx, req)?;
    let visibility_context = visibility_query.as_ref().map(|q|SearchContext::for_query(q, &leaves, Bm25Params::default()));
    if let Some(context) = &visibility_context { context.execution_result()?; }
    Ok(IndexRun { shard_ord: t.shard.as_ref().map(|s| s.id as i64).unwrap_or(0), leaves, lookup, query: q, visibility_query, visibility_context, top, sort, terms, inner, named, scripts: Default::default() })
}

/// The request's `knn` clauses as fixed docs: clause `i` matches exactly
/// the global winners of clause `i` that live in `leaves` of searcher
/// `searcher`, with their DFS scores.
pub(super) fn fixed_knn_queries(req: &SearchRequest, leaves: &[LeafReader<'_>], searcher: u64, fixed: &[ShardKnnHit]) -> Vec<Query> {
    let mut knns = Vec::new();
    for i in 0..req.knn.len() {
        let mut by_leaf: HashMap<usize, Vec<(DocId, f32)>> = HashMap::new();
        for h in fixed.iter().filter(|h| h.clause as usize == i && h.doc_ref.searcher == searcher) {
            if let Some(l) = leaves.get(h.doc_ref.leaf as usize) {
                by_leaf.entry(l.id()).or_default().push((h.doc_ref.doc, h.score));
            }
        }
        knns.push(if by_leaf.is_empty() { Query::MatchNone } else { Query::Custom(Arc::new(FixedDocs { by_leaf })) });
    }
    knns
}

/// `(k, num_candidates)` of a `knn` clause as the single-node path resolves them.
fn knn_sizes(clause: &Value, req: &SearchRequest) -> (usize, usize) {
    let int = |key: &str| clause.get(key).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).map(|x| x as usize);
    let k = int("k").unwrap_or(req.size.max(1));
    let nc = int("num_candidates").unwrap_or_else(|| ((k as f64 * 1.5).round() as usize).clamp(k.min(10_000), 10_000));
    (k, nc)
}

impl IndexService {
    /// Hook 11, DFS phase on one shard: for every `knn` clause of `req`, the
    /// shard's candidates (pre-filtered kNN, scores final). Float HNSW
    /// retains only the shard's top `k` from the same candidate pool.
    /// The coordinator merges all shards with [`merge_knn_candidates`], then
    /// runs [`query_phase`](Self::query_phase) with
    /// [`QueryPhaseOpts::knn_fixed`] and fetches as usual.
    pub fn knn_candidates(&self, handle: &SearcherHandle, req: &SearchRequest, opts: &QueryPhaseOpts) -> Result<Vec<ShardKnnHit>> {
        self.check_handle(handle)?;
        handle.shard.check_usable()?;
        let t = handle.target()?;
        with_request_env(&t, req, || {
            let leaves = leaves_of(&t);
            knn_candidates_over(&t, &leaves, handle.id, req, opts.now.unwrap_or_else(crate::date::now_millis), opts.terms_lookup)
        })
    }
}

/// The DFS kNN candidates of one shard run over `leaves` (doc refs carry
/// `searcher`).
pub(super) fn knn_candidates_over(t: &Target, leaves: &[LeafReader<'_>], searcher: u64, req: &SearchRequest, now: i64, terms_lookup: Option<TermsLookupFn<'_>>) -> Result<Vec<ShardKnnHit>> {
    if req.knn.is_empty() {
        return Ok(Vec::new());
    }
    let lookup = t.lookup();
    crate::script::enter_index(&t.svc.name, &lookup, || t.svc.mapping_runtime())?;
    let qctx = query_ctx(t, &lookup, leaves, now, terms_lookup, req);
    let mut out = Vec::new();
    for (i, clause) in req.knn.iter().enumerate() {
        let o = clause.as_object().ok_or_else(|| EsError::parsing("[knn] must be an object"))?;
        let (k, nc) = knn_sizes(clause, req);
        // A field searched without rescoring (no `rescore_vector.oversample`
        // in the request or mapping; not nested) returns final scores: the
        // shard's top `k` of the `num_candidates` beam is all the global top
        // `k` can use, as ES's DFS phase keeps (the beam is the same, only
        // the returned hits shrink: no owner lookup, sort or merge of the
        // other `num_candidates - k` per shard). Otherwise the shard asks for
        // `num_candidates` docs (a superset of the global top k), which a
        // rescoring field's gather depends on.
        let field = o.get("field").and_then(Value::as_str).and_then(|name| lookup.get(name));
        let final_scores = field.is_some_and(|field| field.nested_path.is_none() && {
            let requested = o.get("rescore_vector").and_then(|r| r.get("oversample")).and_then(Value::as_f64);
            requested.or_else(|| crate::vectors::mapped_oversample(field)).unwrap_or(0.0) == 0.0
        });
        let mut shard_clause = o.clone();
        shard_clause.insert("k".into(), json!(if final_scores { k } else { nc }));
        shard_clause.insert("num_candidates".into(), json!(nc.max(k)));
        let q = parse_knn_query(&shard_clause, &qctx, Some(req.size.max(1))).map_err(|e| wrap_shard_err(&t.svc.name, e))?;
        // A global top-k winner must be in its shard's top k. Keep the
        // graph/rescoring pool unchanged; bound just the shard collector.
        let keep = if final_scores || (matches!(&q, Query::Knn { boost, .. } if *boost == 1.0)
            && field.is_some_and(|field|
                crate::vectors::element_type(field) == crate::vectors::ElementType::Float
                    && crate::vectors::index_type(field).as_deref() == Some("hnsw"))) {
            k
        } else { nc };
        let top = Searcher::new(leaves.to_vec()).search(&q, &SearchOptions { top_k: keep, track_total_hits: TrackTotalHits::False, timeout: req.timeout, ..Default::default() });
        if let Some(error) = top.execution_error {
            return Err(EsError::search_phase(&t.svc.name, EsError::illegal_argument(error)));
        }
        if top.timed_out {
            return Err(EsError::new(408, "timeout_exception", "kNN candidate collection timed out"));
        }
        for h in top.hits {
            out.push(ShardKnnHit { clause: i as u32, doc_ref: DocRef { searcher, leaf: h.leaf as u32, doc: h.doc }, score: h.score });
        }
    }
    Ok(out)
}

/// Coordinator side of the DFS phase: merge the shards' candidates into the
/// global top `k` of every `knn` clause (score descending; ties by shard
/// searcher, leaf, doc). `req` is the original request.
pub fn merge_knn_candidates(req: &SearchRequest, shards: &[Vec<ShardKnnHit>]) -> Vec<ShardKnnHit> {
    let mut out = Vec::new();
    for (i, clause) in req.knn.iter().enumerate() {
        let (k, _) = knn_sizes(clause, req);
        let mut all: Vec<&ShardKnnHit> = shards.iter().flatten().filter(|h| h.clause as usize == i).collect();
        all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_ref.cmp(&b.doc_ref)));
        out.extend(all.into_iter().take(k).cloned());
    }
    out
}

fn check_index_request(t: &Target, req: &SearchRequest) -> Result<()> {
    let max = t.svc.setting_usize("index.max_result_window", 10_000);
    if req.from + req.size > max {
        return Err(EsError::search_phase(
            &t.svc.name,
            EsError::illegal_argument(format!(
                "Result window is too large, from + size must be less than or equal to: [{max}] but was [{}]. See the scroll api for a more efficient way to request large data sets. This limit can be set by changing the [index.max_result_window] index level setting.",
                req.from + req.size
            )),
        ));
    }
    if !t.svc.is_open() {
        return Err(EsError::index_closed(&t.svc.name, &t.svc.uuid));
    }
    if t.svc.is_time_series() && (req.sort.iter().any(|s| s.field == "_id") || req.aggs.as_ref().is_some_and(crate::tsdb::aggs_terms_on_id)) {
        return Err(EsError::query_shard(&t.svc.name, &t.svc.uuid, "Fielddata is not supported on [_id] field in [time_series] indices"));
    }
    Ok(())
}

fn query_phase_inner(t: &Target, handle: &SearcherHandle, req: &SearchRequest, opts: &QueryPhaseOpts) -> Result<QueryPhaseResult> {
    if req.retriever.is_some() || req.rank.is_some() || req.sub_searches.is_some() {
        return Err(EsError::illegal_argument("[retriever] and [rank] searches are coordinated by the caller: run each child as its own query phase"));
    }
    let start = Instant::now();
    check_index_request(t, req)?;
    let _query_current = crate::shard::CurrentGuard::new(vec![(&t.shard_copy().shard.counters.query_current, 1)]);
    let now = opts.now.unwrap_or_else(crate::date::now_millis);
    let run = prepare(t, leaves_of(t), req, opts.terms_lookup, now, true, opts.knn_fixed.map(|f| (handle.id, f)), opts.dfs)?;
    let sorted_by_field = !req.sort.is_empty() && !(req.sort.len() == 1 && req.sort[0].field == "_score");
    let scores_shown = !sorted_by_field || req.track_scores;
    let hits: Vec<ShardHit> = run
        .top
        .hits
        .iter()
        .map(|h| Ok(ShardHit {
            doc_ref: DocRef { searcher: handle.id, leaf: h.leaf as u32, doc: h.doc },
            score: h.score,
            sort_values: h.sort.iter().map(ShardSortValue::from).collect(),
            collapse_key: req.collapse.as_ref().map(|cf| docvalue_values(&run.leaves[h.leaf],&run.lookup,cf,h.doc,None).map(|values|crate::collapse::group_key(&values))).transpose()?,
        }))
        .collect::<Result<_>>()?;
    let gte = run.top.total_hits_relation == TotalHitsRelation::Gte;
    let total = TotalHits { value: run.top.total_hits, relation: if gte { TotalRelation::Gte } else { TotalRelation::Eq } };
    let max_score = if scores_shown && !run.top.hits.is_empty() { Some(run.top.max_score).filter(|x| x.is_finite()) } else { None };
    let aggs = match &req.aggs {
        None => None,
        Some(spec) => Some(shard_aggs(t, &run, req, spec, now, opts.max_buckets, opts.skip_pipeline)?),
    };
    let suggest = match &req.suggest {
        Some(sg) => {
            let st = [crate::suggest::SuggestTarget { index: &t.svc.name, leaves: &run.leaves, lookup: &run.lookup, svc: Some(&*t.svc) }];
            Some(crate::suggest::run(sg, &st, req.typed_keys)?)
        }
        None => None,
    };
    let profile = req.profile.then(|| json!({"shards": []}));
    t.shard_copy().shard.counters.record_search(start.elapsed(), Duration::ZERO, false);
    Ok(QueryPhaseResult { hits, total, max_score, aggs, suggest, profile, timed_out: run.top.timed_out, terminated_early: run.top.terminated_early })
}

/// Render `hits` (fetch phase).
fn fetch_inner(t: &Target, handle: &SearcherHandle, hits: &[ShardHit], req: &SearchRequest) -> Result<Vec<HitJson>> {
    let now = crate::date::now_millis();
    let run = prepare(t, leaves_of(t), req, None, now, false, None, None)?;
    let sorted_by_field = !req.sort.is_empty() && !(req.sort.len() == 1 && req.sort[0].field == "_score");
    let scores_shown = !sorted_by_field || req.track_scores;
    let fetch_start = Instant::now();
    let mut out = Vec::with_capacity(hits.len());
    for h in hits {
        if h.doc_ref.searcher != handle.id {
            return Err(EsError::illegal_argument(format!("doc ref belongs to searcher [{}], not [{}]", h.doc_ref.searcher, handle.id)));
        }
        let leaf = run.leaves.get(h.doc_ref.leaf as usize).ok_or_else(|| EsError::illegal_argument(format!("doc ref leaf [{}] out of range", h.doc_ref.leaf)))?;
        let th = TopHit::new(h.doc_ref.leaf as usize, h.doc_ref.doc, h.score, h.sort_values.iter().map(SortValue::from).collect());
        let mut js = fetch_hit(t, &run, leaf, h.doc_ref.doc, Some(&th), req, scores_shown)?;
        if !run.inner.is_empty() && js.ends_with('}') {
            js.pop();
            js.push_str(",\"inner_hits\":");
            js.push_str(&serde_json::to_string(&render_inner_hits(t, &run, leaf, h.doc_ref.doc, &run.inner, req.rest_total_hits_as_int, None)?)?);
            js.push('}');
        }
        out.push(serde_json::value::RawValue::from_string(js)?);
    }
    t.shard_copy().shard.counters.record_search(fetch_start.elapsed(), fetch_start.elapsed(), !hits.is_empty());
    Ok(out)
}

/// The coordinator's merge of shard query results.
#[derive(Clone, Debug)]
pub struct MergedQuery {
    /// The requested page (`from`..`from + size`) of the global order:
    /// `(index into the input results, hit)`.
    pub page: Vec<(usize, ShardHit)>,
    /// Merged total (capped at the `track_total_hits` limit); `None` when the
    /// request disables total tracking.
    pub total: Option<TotalHits>,
    /// `max_score` to render (`None` → `null`).
    pub max_score: Option<f32>,
    pub timed_out: bool,
    pub terminated_early: bool,
}

fn cmp_hits(req: &SearchRequest, (sa, a): (usize, &ShardHit), (sb, b): (usize, &ShardHit)) -> Ordering {
    let tie = || sa.cmp(&sb).then(a.doc_ref.leaf.cmp(&b.doc_ref.leaf)).then(a.doc_ref.doc.cmp(&b.doc_ref.doc));
    if req.sort.is_empty() {
        return b.score.total_cmp(&a.score).then_with(tie);
    }
    for (i, sp) in req.sort.iter().enumerate() {
        let o = match sp.field.as_str() {
            "_score" => {
                let o = b.score.total_cmp(&a.score);
                if sp.descending { o } else { o.reverse() }
            }
            "_doc" | "_shard_doc" => {
                let o = sa.cmp(&sb).then(a.doc_ref.leaf.cmp(&b.doc_ref.leaf)).then(a.doc_ref.doc.cmp(&b.doc_ref.doc));
                if sp.descending { o.reverse() } else { o }
            }
            _ => {
                let (x, y) = (a.sort_values.get(i).map(SortValue::from).unwrap_or(SortValue::Missing), b.sort_values.get(i).map(SortValue::from).unwrap_or(SortValue::Missing));
                let missing_first = sp.missing.as_ref().and_then(Value::as_str) == Some("_first");
                cmp_sort(&x, &y, sp.descending, missing_first)
            }
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    tie()
}

/// Merge shard query results the way the single-node `_search` merges
/// indices: global order by the request's sort (then score), collapse
/// groups deduplicated across shards, `from` / `size` applied, totals summed.
pub fn merge_query_phases(req: &SearchRequest, results: &[QueryPhaseResult]) -> MergedQuery {
    let mut all: Vec<(usize, &ShardHit)> = results.iter().enumerate().flat_map(|(i, r)| r.hits.iter().map(move |h| (i, h))).collect();
    all.sort_by(|a, b| cmp_hits(req, *a, *b));
    if req.collapse.is_some() && results.len() > 1 {
        let mut seen = std::collections::HashSet::new();
        all.retain(|(_, h)| seen.insert(h.collapse_key.clone().unwrap_or_default()));
    }
    let page: Vec<(usize, ShardHit)> = all.into_iter().skip(req.from).take(req.size).map(|(i, h)| (i, h.clone())).collect();
    let sum: u64 = results.iter().map(|r| r.total.value).sum();
    let gte = results.iter().any(|r| r.total.relation == TotalRelation::Gte);
    let (value, gte) = match req.track_total_hits {
        Some(TrackTotalHits::UpTo(n)) if sum > n => (n, true),
        _ => (sum, gte),
    };
    let total = req.track_total_hits.map(|_| TotalHits { value, relation: if gte { TotalRelation::Gte } else { TotalRelation::Eq } });
    let sorted_by_field = !req.sort.is_empty() && !(req.sort.len() == 1 && req.sort[0].field == "_score");
    let scores_shown = !sorted_by_field || req.track_scores;
    let max_score = if scores_shown && !page.is_empty() && results.iter().any(|r| !r.hits.is_empty()) {
        results.iter().filter_map(|r| r.max_score).filter(|x| x.is_finite()).reduce(f32::max)
    } else {
        None
    };
    MergedQuery {
        page,
        total,
        max_score,
        timed_out: results.iter().any(|r| r.timed_out),
        terminated_early: results.iter().any(|r| r.terminated_early),
    }
}

/// Fetch a merged page from the shards it came from (`handles[i]` is the
/// searcher of input result `i`) and return the rendered hits in page order.
pub fn fetch_page(handles: &[&SearcherHandle], merged: &MergedQuery, req: &SearchRequest) -> Result<Vec<HitJson>> {
    let mut by_shard: Vec<Vec<(usize, &ShardHit)>> = vec![Vec::new(); handles.len()];
    for (pos, (shard, hit)) in merged.page.iter().enumerate() {
        by_shard[*shard].push((pos, hit));
    }
    let mut out: Vec<Option<HitJson>> = vec![None; merged.page.len()];
    for (shard, items) in by_shard.iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        let hits: Vec<ShardHit> = items.iter().map(|(_, h)| (*h).clone()).collect();
        let rendered = handles[shard].svc.fetch(handles[shard], &hits, req)?;
        for ((pos, _), r) in items.iter().zip(rendered) {
            out[*pos] = Some(r);
        }
    }
    Ok(out.into_iter().flatten().collect())
}

/// Run the request's aggregations over the shard's matching docs and return
/// the shard partial (hook 10).
fn shard_aggs(t: &Target, run: &IndexRun, req: &SearchRequest, spec: &Value, now: i64, max_buckets: usize, skip_pipeline: bool) -> Result<AggPartial> {
    let mut segs: Vec<AggSeg> = Vec::new();
    let ctx = SearchContext::for_query(&run.query, &run.leaves, Bm25Params::default());
    ctx.execution_result()?;
    for leaf in &run.leaves {
        let mut docs = Vec::new();
        let mut scores = Vec::new();
        if let Some(mut s) = run.query.scorer(leaf, &ctx) {
            let live = leaf.live_docs();
            loop {
                let d = s.advance();
                if d == rustic_core::query::TERMINATED {
                    break;
                }
                if live(d) {
                    let sc = s.score();
                    if req.min_score.map(|m| sc >= m).unwrap_or(true) {
                        docs.push(d);
                        scores.push(sc);
                    }
                }
            }
        }
        segs.push(AggSeg { leaf: *leaf, lookup: &run.lookup, index: &t.svc.name, max_regex_length: t.svc.setting_usize("index.max_regex_length", 1000), docs, scores });
    }
    let fetch = |si: usize, doc: DocId, score: f32, spec: &Value| -> Result<Value> {
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
        let s = fetch_hit(t, run, leaf, doc, Some(&th), &sub, true)?;
        crate::search::hit_value(&s)
    };
    let visibility_error = parking_lot::Mutex::new(None);
    let all = |si: usize| -> Vec<DocId> {
        match run.visible_docs(&segs[si].leaf) {
            Ok(docs) => docs,
            Err(error) => { *visibility_error.lock() = Some(error); Vec::new() }
        }
    };
    let mapping_context = [(t.svc.name.as_str(), run.lookup.as_ref())];
    let env = AggEnv { mapping_context: &mapping_context, now, max_buckets, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: run.visibility_query.is_some(), top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
    let request = AggRequest { aggs: spec.clone(), typed_keys: req.typed_keys, max_buckets, now };
    let result = crate::hooks::agg_partial::shard_partial(spec, &segs, &env, &request, skip_pipeline);
    run.visibility_result()?;
    if let Some(error) = visibility_error.lock().take() { return Err(wrap_agg_err(&t.svc.name, error)); }
    result.map_err(|e| wrap_agg_err(&t.svc.name, e))
}

fn wrap_agg_err(index: &str, e: EsError) -> EsError {
    if e.kind == "parsing_exception" || e.kind == "x_content_parse_exception" || e.status != 400 {
        e
    } else {
        EsError::search_phase(index, e)
    }
}

impl AggRequest {
    /// The aggregation part of a search request.
    pub fn from_search(req: &SearchRequest, max_buckets: usize, now: i64) -> Option<AggRequest> {
        req.aggs.as_ref().map(|a| AggRequest { aggs: a.clone(), typed_keys: req.typed_keys, max_buckets, now })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    fn setup() -> (tempfile::TempDir, Arc<Node>, Arc<IndexService>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node
            .create_index(
                "t",
                &json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {
                    "title": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"},
                    "comments": {"type": "nested", "properties": {"who": {"type": "keyword"}, "text": {"type": "text"}}}
                }}}),
            )
            .unwrap();
        for i in 0..30 {
            let body = json!({
                "title": format!("quick brown fox number {i} {}", if i % 3 == 0 { "jumps" } else { "sleeps" }),
                "tag": format!("t{}", i % 4), "n": i,
                "comments": [{"who": format!("u{}", i % 5), "text": "nice fox"}, {"who": "z", "text": format!("meh {i}")}]
            });
            put_doc(&svc, &format!("d{i}"), body);
            if i == 14 {
                svc.shard(0).unwrap().shard.refresh().unwrap(); // two segments
            }
        }
        svc.refresh().unwrap();
        (tmp, node, svc)
    }

    fn put_doc(svc: &IndexService, id: &str, body: Value) {
        let r = svc.write(vec![crate::index::WriteReq::Index { id: Some(id.into()), source: body.to_string(), routing: None, cond: crate::shard::WriteCond::default() }], true);
        r.into_iter().next().unwrap().unwrap();
    }

    /// Single-node reference output over the same pinned segments.
    fn reference(h: &SearcherHandle, req: &SearchRequest) -> Value {
        let t = h.as_target().unwrap();
        let out = execute(&[t], req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
        serde_json::from_str(&out.json).unwrap()
    }

    /// query_phase on each handle + merge + fetch, shaped like `response["hits"]`.
    fn two_phase(handles: &[&SearcherHandle], req: &SearchRequest) -> Value {
        let results: Vec<QueryPhaseResult> = handles.iter().map(|h| h.svc.query_phase(h, req, &QueryPhaseOpts::default()).unwrap()).collect();
        // Results survive serialization.
        let results: Vec<QueryPhaseResult> = results.into_iter().map(|r| serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap()).collect();
        let merged = merge_query_phases(req, &results);
        let hits = fetch_page(handles, &merged, req).unwrap();
        let mut h = Map::new();
        if let Some(t) = merged.total {
            if req.rest_total_hits_as_int {
                h.insert("total".into(), json!(t.value));
            } else {
                h.insert("total".into(), json!({"value": t.value, "relation": if t.relation == TotalRelation::Gte { "gte" } else { "eq" }}));
            }
        }
        h.insert("max_score".into(), merged.max_score.map(|m| serde_json::from_str(&crate::retriever::score_json(m)).unwrap()).unwrap_or(Value::Null));
        h.insert("hits".into(), Value::Array(hits.iter().map(|x| serde_json::from_str(x.get()).unwrap()).collect()));
        Value::Object(h)
    }

    fn parse(body: Value) -> SearchRequest {
        parse_request(&body, &HashMap::new()).unwrap()
    }

    #[test]
    fn query_then_fetch_equals_single_node_search() {
        let (_tmp, _node, svc) = setup();
        let h = svc.open_searcher().unwrap();
        assert!(h.num_leaves() >= 2);
        let cases = vec![
            json!({"query": {"match": {"title": "fox"}}, "size": 5}),
            json!({"query": {"match": {"title": "jumps"}}, "from": 2, "size": 4}),
            json!({"query": {"match_all": {}}, "sort": [{"n": "desc"}], "size": 7}),
            json!({"query": {"match_all": {}}, "sort": [{"tag": "asc"}, {"n": "desc"}], "from": 3, "size": 6, "_source": ["n", "tag"]}),
            json!({"query": {"match": {"title": "fox"}}, "highlight": {"fields": {"title": {}}}, "size": 3}),
            json!({"query": {"nested": {"path": "comments", "query": {"match": {"comments.text": "nice"}}, "inner_hits": {}}}, "size": 3}),
            json!({"query": {"match_all": {}}, "collapse": {"field": "tag"}, "sort": [{"n": "asc"}], "size": 10}),
            json!({"query": {"match_all": {}}, "fields": ["tag", "n"], "docvalue_fields": ["n"], "version": true, "seq_no_primary_term": true, "size": 4, "sort": ["_doc"]}),
            json!({"query": {"bool": {"must": {"match": {"title": "fox"}}, "should": {"term": {"tag": {"value": "t1", "_name": "is_t1"}}}}}, "size": 6, "track_total_hits": true}),
            json!({"query": {"match_all": {}}, "track_scores": true, "sort": [{"n": "asc"}], "size": 2}),
        ];
        for body in cases {
            let req = parse(body.clone());
            let want = reference(&h, &req)["hits"].clone();
            let got = two_phase(&[&h], &req);
            assert_eq!(serde_json::to_string(&want).unwrap(), serde_json::to_string(&got).unwrap(), "request {body}");
        }
    }

    #[test]
    fn released_handles_fail_and_refs_are_checked() {
        let (_tmp, _node, svc) = setup();
        let h = svc.open_searcher().unwrap();
        let req = parse(json!({"query": {"match_all": {}}, "size": 2}));
        let r = svc.query_phase(&h, &req, &QueryPhaseOpts::default()).unwrap();
        assert_eq!(r.hits.len(), 2);
        assert_eq!(r.total, TotalHits { value: 30, relation: TotalRelation::Eq });
        assert!(r.hits.iter().all(|x| x.doc_ref.searcher == h.id()));
        let h2 = svc.open_searcher().unwrap();
        assert_eq!(svc.fetch(&h2, &r.hits, &req).unwrap_err().kind, "illegal_argument_exception");
        // The pinned view does not see later writes; a new searcher does
        // (once refreshed).
        put_doc(&svc, "late", json!({"title": "late", "n": 99}));
        svc.refresh().unwrap();
        let r2 = svc.query_phase(&h, &req, &QueryPhaseOpts::default()).unwrap();
        assert_eq!(r2.total.value, 30);
        assert_eq!(svc.query_phase(&svc.open_searcher().unwrap(), &req, &QueryPhaseOpts::default()).unwrap().total.value, 31);
        h.release();
        assert!(h.is_released());
        assert_eq!(svc.query_phase(&h, &req, &QueryPhaseOpts::default()).unwrap_err().kind, "search_context_missing_exception");
        assert_eq!(svc.fetch(&h, &r.hits, &req).unwrap_err().kind, "search_context_missing_exception");
    }

    #[test]
    fn merge_across_shards_matches_one_index() {
        // Two indices stand in for two shards of one index.
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let mapping = json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}, "title": {"type": "text"}}}});
        let a = node.create_index("a", &mapping).unwrap();
        let b = node.create_index("b", &mapping).unwrap();
        let all = node.create_index("all", &mapping).unwrap();
        for i in 0..40 {
            let body = json!({"tag": format!("t{}", i % 3), "n": i, "title": format!("fox {i}")});
            put_doc(if i % 2 == 0 { &a } else { &b }, &format!("d{i}"), body.clone());
            put_doc(&all, &format!("d{i}"), body);
        }
        // Writes are searchable after a refresh (as in ES), not on return.
        [&a, &b, &all].iter().for_each(|s| s.refresh().unwrap());
        let (ha, hb, hall) = (a.open_searcher().unwrap(), b.open_searcher().unwrap(), all.open_searcher().unwrap());
        for body in [
            json!({"query": {"match_all": {}}, "sort": [{"n": "desc"}], "from": 5, "size": 8}),
            json!({"query": {"match_all": {}}, "sort": [{"tag": "asc"}, {"n": "asc"}], "size": 12}),
            json!({"query": {"match_all": {}}, "collapse": {"field": "tag"}, "sort": [{"n": "desc"}], "size": 10}),
        ] {
            let req = parse(body.clone());
            let want = two_phase(&[&hall], &req);
            let got = two_phase(&[&ha, &hb], &req);
            let ids = |v: &Value| v["hits"].as_array().unwrap().iter().map(|h| h["_id"].clone()).collect::<Vec<_>>();
            assert_eq!(ids(&want), ids(&got), "{body}");
            assert_eq!(want["total"], got["total"], "{body}");
        }
    }

    #[test]
    fn aggregations_through_query_phase_and_reduce_match_single_node() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let mapping = json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}, "p": {"type": "double"}}}});
        let shards: Vec<_> = (0..3).map(|i| node.create_index(&format!("s{i}"), &mapping).unwrap()).collect();
        let all = node.create_index("all", &mapping).unwrap();
        for i in 0..90 {
            let body = json!({"tag": format!("t{}", i % 6), "n": i, "p": (i * 3 % 20) as f64});
            put_doc(&shards[i % 3], &format!("d{i}"), body.clone());
            put_doc(&all, &format!("d{i}"), body);
        }
        shards.iter().chain([&all]).for_each(|s| s.refresh().unwrap());
        let hs: Vec<SearcherHandle> = shards.iter().map(|s| s.open_searcher().unwrap()).collect();
        let hall = all.open_searcher().unwrap();
        let aggs = json!({
            "t": {"terms": {"field": "tag"}, "aggs": {"a": {"avg": {"field": "p"}}, "d": {"cumulative_sum": {"buckets_path": "a"}}}},
            "h": {"histogram": {"field": "n", "interval": 25}, "aggs": {"s": {"sum": {"field": "p"}}, "dv": {"derivative": {"buckets_path": "s"}}}},
            "c": {"cardinality": {"field": "tag"}}, "st": {"stats": {"field": "n"}}, "sb": {"sum_bucket": {"buckets_path": "t>_count"}}
        });
        let req = parse(json!({"size": 0, "query": {"range": {"n": {"gte": 10}}}, "aggs": aggs.clone(), "track_total_hits": true}));
        let want = reference(&hall, &req)["aggregations"].clone();
        let opts = QueryPhaseOpts { now: Some(0), ..Default::default() };
        let results: Vec<QueryPhaseResult> = hs.iter().map(|h| h.index().query_phase(h, &req, &opts).unwrap()).collect();
        let parts: Vec<AggPartial> = results.iter().map(|r| AggPartial::from_bytes(&r.aggs.as_ref().unwrap().to_bytes()).unwrap()).collect();
        let got = AggPartial::reduce(parts, &AggRequest::from_search(&req, 65_536, 0).unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&want).unwrap(), serde_json::to_string(&got).unwrap());
        assert_eq!(merge_query_phases(&req, &results).total.unwrap().value, 80);
    }

    fn vec_of(i: usize) -> Vec<f64> {
        // Deterministic, well-separated vectors on a spiral.
        let a = i as f64 * 0.37;
        vec![a.cos(), a.sin(), (i % 7) as f64 / 7.0, 1.0 + (i % 3) as f64]
    }

    /// N `IndexService`s created from one body with distinct uuids stand in for
    /// N shards: DFS kNN + query + reduce + fetch equals the same data in one index.
    #[test]
    fn dfs_knn_over_n_services_matches_one_index() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let body = json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {
            "tag": {"type": "keyword"}, "n": {"type": "long"}, "title": {"type": "text"},
            "v": {"type": "dense_vector", "dims": 4, "index": true, "similarity": "cosine", "index_options": {"type": "hnsw"}}
        }}});
        let shards: Vec<Arc<IndexService>> = (0..3).map(|i| node.create_index_with_uuid(&format!("shard{i}"), &format!("cluster-uuid-{i}"), &body).unwrap()).collect();
        assert!(shards.iter().all(|s| s.uuid.starts_with("cluster-uuid-")));
        let all = node.create_index("all", &body).unwrap();
        for i in 0..60usize {
            let doc = json!({"tag": format!("t{}", i % 3), "n": i, "title": format!("doc {i} fox"), "v": vec_of(i)});
            put_doc(&shards[i % 3], &format!("d{i}"), doc.clone());
            put_doc(&all, &format!("d{i}"), doc);
        }
        shards.iter().chain([&all]).for_each(|s| s.refresh().unwrap());
        let hs: Vec<SearcherHandle> = shards.iter().map(|s| s.open_searcher().unwrap()).collect();
        let hall = all.open_searcher().unwrap();
        let qv = vec_of(10);
        let cases = vec![
            json!({"knn": {"field": "v", "query_vector": qv, "k": 5, "num_candidates": 30}, "size": 5}),
            json!({"knn": {"field": "v", "query_vector": qv, "k": 4, "num_candidates": 30, "filter": {"term": {"tag": "t1"}}}, "size": 10}),
            json!({"knn": [{"field": "v", "query_vector": qv, "k": 3, "num_candidates": 30}, {"field": "v", "query_vector": vec_of(33), "k": 3, "num_candidates": 30, "boost": 2.0}], "size": 10}),
            json!({"query": {"match": {"title": "fox"}}, "knn": {"field": "v", "query_vector": qv, "k": 3, "num_candidates": 30}, "size": 8}),
            json!({"knn": {"field": "v", "query_vector": qv, "k": 6, "num_candidates": 30}, "size": 3, "from": 2, "aggs": {"t": {"terms": {"field": "tag"}}}}),
        ];
        for body in cases {
            let req = parse(body.clone());
            // Reference: the same two-phase flow over the single combined index.
            let ref_cands = vec![hall.index().knn_candidates(&hall, &req, &QueryPhaseOpts::default()).unwrap()];
            let ref_fixed = merge_knn_candidates(&req, &ref_cands);
            let ref_opts = QueryPhaseOpts { knn_fixed: Some(&ref_fixed), ..Default::default() };
            let ref_qp = hall.index().query_phase(&hall, &req, &ref_opts).unwrap();
            let want = two_phase_with(&[&hall], &req, &[ref_qp.clone()]);
            // Sharded flow, with the candidates travelling as JSON.
            let cands: Vec<Vec<ShardKnnHit>> = hs
                .iter()
                .map(|h| serde_json::from_slice(&serde_json::to_vec(&h.index().knn_candidates(h, &req, &QueryPhaseOpts::default()).unwrap()).unwrap()).unwrap())
                .collect();
            let fixed = merge_knn_candidates(&req, &cands);
            let k_total: usize = req.knn.iter().map(|c| knn_sizes(c, &req).0).sum();
            assert!(fixed.len() <= k_total, "{body}");
            let opts = QueryPhaseOpts { knn_fixed: Some(&fixed), ..Default::default() };
            let qps: Vec<QueryPhaseResult> = hs.iter().map(|h| h.index().query_phase(h, &req, &opts).unwrap()).collect();
            let got = two_phase_with(&hs.iter().collect::<Vec<_>>(), &req, &qps);
            let ids = |v: &Value| v["hits"].as_array().unwrap().iter().map(|h| (h["_id"].clone(), h["_score"].clone())).collect::<Vec<_>>();
            if req.query.is_some() {
                // Text scores use per-shard statistics (no DFS_QUERY_THEN_FETCH): compare the kNN winners only.
                assert_eq!(ids(&want)[..3].iter().map(|x| x.0.clone()).collect::<Vec<_>>(), ids(&got)[..3].iter().map(|x| x.0.clone()).collect::<Vec<_>>(), "{body}");
            } else {
                assert_eq!(ids(&want), ids(&got), "{body}");
            }
            assert_eq!(want["total"], got["total"], "{body}");
            assert!(!ids(&got).is_empty());
            if req.aggs.is_some() {
                let r = AggRequest::from_search(&req, 65_536, 0).unwrap();
                let a = AggPartial::reduce(qps.iter().map(|q| q.aggs.clone().unwrap()).collect(), &r).unwrap();
                let b = AggPartial::reduce(vec![ref_qp.aggs.clone().unwrap()], &r).unwrap();
                assert_eq!(a, b, "{body}");
                // Only the global top-k docs feed the aggregations.
                let docs: u64 = a["t"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
                assert_eq!(docs, 6);
            }
        }
    }

    /// Merge + fetch for already computed query phases.
    fn two_phase_with(handles: &[&SearcherHandle], req: &SearchRequest, results: &[QueryPhaseResult]) -> Value {
        let merged = merge_query_phases(req, results);
        let hits = fetch_page(handles, &merged, req).unwrap();
        let mut h = Map::new();
        if let Some(t) = merged.total {
            h.insert("total".into(), json!({"value": t.value, "relation": if t.relation == TotalRelation::Gte { "gte" } else { "eq" }}));
        }
        h.insert("hits".into(), Value::Array(hits.iter().map(|x| serde_json::from_str(x.get()).unwrap()).collect()));
        Value::Object(h)
    }

    #[test]
    fn n_services_from_one_create_body_query_reduce_and_fetch() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let body = json!({"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}}}});
        let shards: Vec<_> = (0..3).map(|i| node.create_index_with_uuid(&format!("sh{i}"), &format!("u{i}"), &body).unwrap()).collect();
        let all = node.create_index_with_uuid("whole", "uwhole", &body).unwrap();
        for i in 0..45 {
            let d = json!({"tag": format!("t{}", i % 4), "n": i});
            put_doc(&shards[i % 3], &format!("d{i}"), d.clone());
            put_doc(&all, &format!("d{i}"), d);
        }
        // Writes are searchable after a refresh (as in ES), not on return.
        shards.iter().chain([&all]).for_each(|s| s.refresh().unwrap());
        let hs: Vec<SearcherHandle> = shards.iter().map(|s| s.open_searcher().unwrap()).collect();
        let hall = all.open_searcher().unwrap();
        let req = parse(json!({"query": {"range": {"n": {"gte": 5}}}, "sort": [{"n": "desc"}], "size": 6, "from": 2,
            "aggs": {"t": {"terms": {"field": "tag"}, "aggs": {"m": {"max": {"field": "n"}}}}, "c": {"cardinality": {"field": "tag"}}}}));
        let opts = QueryPhaseOpts { now: Some(0), ..Default::default() };
        let qps: Vec<QueryPhaseResult> = hs.iter().map(|h| h.index().query_phase(h, &req, &opts).unwrap()).collect();
        let want_qp = hall.index().query_phase(&hall, &req, &opts).unwrap();
        let got = two_phase_with(&hs.iter().collect::<Vec<_>>(), &req, &qps);
        let want = two_phase_with(&[&hall], &req, &[want_qp.clone()]);
        let ids = |v: &Value| v["hits"].as_array().unwrap().iter().map(|h| h["_id"].clone()).collect::<Vec<_>>();
        assert_eq!(ids(&want), ids(&got));
        assert_eq!(want["total"], got["total"]);
        let r = AggRequest::from_search(&req, 65_536, 0).unwrap();
        let a = AggPartial::reduce(qps.into_iter().map(|q| q.aggs.unwrap()).collect(), &r).unwrap();
        assert_eq!(a, AggPartial::reduce(vec![want_qp.aggs.unwrap()], &r).unwrap());
        // Distinct cluster uuids per service.
        assert_eq!(shards.iter().map(|s| s.uuid.clone()).collect::<Vec<_>>(), vec!["u0", "u1", "u2"]);
    }
}
