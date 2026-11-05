//! ES `query_then_fetch` for cross-node searches.
//!
//! The single-phase fan-out asks every shard for its top `from + size`
//! hits fully fetched (`_source`, highlights, fields, ...), so all but the
//! global top `size` are fetched, serialized and shipped for nothing. Here
//! the query phase asks each shard for ids, scores and sort values only
//! (plus aggregations, suggestions and totals): the shard searches a pinned
//! reader view, registers it as a reader context (ES `ReaderContext`) and
//! answers with the context id (`_rustic_ctx`) and each hit's address in it
//! (`_rustic_doc`: target, leaf, doc). The coordinator merges as before,
//! then asks each shard, on the copy that answered its query phase, to
//! render just its winners from that context; this fetch frees it, a shard
//! without winners is told to free it, as is every context still open when
//! the search ends any other way ([`OpenContexts`], also when the request
//! future is dropped); contexts the coordinator never learned of expire on
//! the shard (`search.default_keep_alive`, reaped by node maintenance). A shard
//! whose context admission fails returns a failure before scoring. Older
//! peers use the single-phase protocol selected before dispatch.
//! The fetch phase does not search: the
//! query only drives highlighting and `matched_queries`. Hits keep the query
//! phase's `_score` and `sort`; their sources pass through as raw JSON.
//!
//! Both phases run on the search's copy plan ([`copies::Plan`]), as the
//! single-phase fan-out does: a query phase goes to the shard's copies in
//! order and moves on after any failure but a request error; the copy that
//! answers is pinned, and its context alone is fetched. A shard no copy
//! answered, or whose fetch failed (phase `fetch`, not retried), is a shard
//! failure: partial results, or ES's `search_phase_execution_exception`.
//!
//! Request features listed neither in [`BODY_KEYS`] nor in [`PARAMS`] use
//! the single-phase path. Unreadable shard answers are shard failures. `RUSTIC_SINGLE_PHASE_SEARCH` in the environment disables the
//! two-phase path.

use super::fanout::member;
use super::{copies, copy_request, fanout, label_part_failures, stateless_read, Dfs, RestReq, RestResp, ShardUnit, Tried};
use crate::node::ClusterNode;
use elasticsearch_engine::error::Result as EsResult;

macro_rules! optional {
    ($value:expr) => { match $value { Some(value) => value, None => return Ok(None) } };
}
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Header naming a shard search's query_then_fetch phase (`query` /
/// `fetch`): internal, stripped from client requests.
pub(super) const HEADER: &str = "x-rustic-internal-qtf";

/// Header naming the reader context id a query phase registers (chosen by
/// the coordinator; internal, stripped from client requests). Apart from
/// [`HEADER`] so a node of an earlier build, which reads only `query`
/// there, still runs the phase (registering an id of its own, which its
/// response names).
pub(super) const CTX_HEADER: &str = "x-rustic-internal-qtf-ctx";
/// Marks a query phase from a pinned query_then_fetch coordinator (see
/// `elasticsearch_http::InternalQtfPinned`).
pub(super) const PINNED_HEADER: &str = "x-rustic-internal-qtf-pinned";
/// Marks a query or fetch phase from a coordinator that merges light query
/// hits and matches fetched hits by address (see
/// `elasticsearch_http::InternalQtfLight`). A node of an earlier build
/// ignores it: its full query hits and address-less fetched hits are
/// matched by `_index` and `_id`, as before.
pub(super) const LIGHT_HEADER: &str = "x-rustic-internal-qtf-light";

/// Tests only: send no [`LIGHT_HEADER`] (the full query hits and plain
/// fetch answers of a node of an earlier build).
#[doc(hidden)]
pub static LIGHT_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Mark `r` with [`LIGHT_HEADER`].
fn light(r: &mut RestReq) -> EsResult<()> {
    if !LIGHT_DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
        r.replace_header(LIGHT_HEADER, "1")?;
    }
    Ok(())
}

/// Top-level body keys whose semantics both phases reproduce exactly.
const BODY_KEYS: &[&str] = &[
    "query", "from", "size", "sort", "track_total_hits", "track_scores", "_source", "aggs", "aggregations", "highlight",
    "docvalue_fields", "fields", "script_fields", "post_filter", "min_score", "search_after", "timeout", "version",
    "seq_no_primary_term", "suggest", "stats", "runtime_mappings", "indices_boost",
];

/// Only needed by the query phase (the fetch phase renders given docs).
const QUERY_ONLY: &[&str] = &["aggs", "aggregations", "suggest", "search_after", "post_filter", "min_score", "stats", "track_total_hits", "sort"];

/// URL parameters of a shard sub-request (`from`, `size`, `preference` and
/// rendering parameters never reach it) that do not change what a hit renders.
const PARAMS: &[&str] = &[
    "typed_keys", "rest_total_hits_as_int", "track_total_hits", "search_type", "routing", "ignore_unavailable", "allow_no_indices",
    "expand_wildcards", "ignore_throttled", "request_cache", "batched_reduce_size", "pre_filter_shard_size",
    "max_concurrent_shard_requests", "allow_partial_search_results", "ccs_minimize_roundtrips", "timeout", "from", "size",
    "preference", "filter_path", "pretty", "human", "error_trace",
];

/// Hit tag: the index of the sub-request that returned it.
const TAG: &str = "_qtf_part";
/// A query-phase hit's address in its reader context (engine output).
const DOC_REF: &str = "_rustic_doc";
/// A query-phase response's reader context id (engine output).
const CTX: &str = "_rustic_ctx";

/// Can this cross-node `_search` over `n_subs` shard requests run as
/// query_then_fetch?
pub(super) fn eligible(req: &RestReq, body: &Value, params: &HashMap<String, String>, n_subs: usize) -> bool {
    if n_subs < 2 || req.dfs == Dfs::Collect || std::env::var_os("RUSTIC_SINGLE_PHASE_SEARCH").is_some() {
        return false;
    }
    let Some(o) = body.as_object() else { return false };
    if !o.keys().all(|k| BODY_KEYS.contains(&k.as_str())) || !params.keys().all(|k| PARAMS.contains(&k.as_str())) {
        return false;
    }
    // Completion suggestion options carry their document's `_source`, which
    // the query phase leaves out: such searches stay single-phase.
    let completion = body.get("suggest").and_then(Value::as_object).is_some_and(|s| s.values().any(|e| e.get("completion").is_some()));
    let (from, size) = fanout::from_size(body, params);
    // A light fetch (no `_source`, stored fields, highlighting or script
    // fields, few hits): the second round trip costs more than returning
    // every shard's few hits from the query phase. kNN searches typically.
    let light = o.get("_source") == Some(&Value::Bool(false))
        && !["highlight", "stored_fields", "script_fields"].iter().any(|k| o.contains_key(*k))
        && from.saturating_add(size) <= 100;
    // Nothing to fetch; `inner_hits` would need their own fetch per hit.
    size > 0 && !light && !completion && from.saturating_add(size) <= 10_000 && !has_key(body, "inner_hits")
}

/// Whether any object in `v` has key `key` (without serializing `v`: a
/// kNN body carries thousands of floats).
fn has_key(v: &Value, key: &str) -> bool {
    match v {
        Value::Object(o) => o.contains_key(key) || o.values().any(|v| has_key(v, key)),
        Value::Array(a) => a.iter().any(|v| matches!(v, Value::Object(_) | Value::Array(_)) && has_key(v, key)),
        _ => false,
    }
}

/// [`HEADER`] value of a single-round shard search ([`single_round`]).
pub(super) const SINGLE_ROUND: &str = "query_fetch";

/// The largest page (`from + size`) answered in a single round: each shard
/// renders its top `from + size` hits, `(shards - 1) * (from + size)` more
/// than the page, which for a page this small costs far less than the
/// fetch round it saves (a sub-request per shard with winners and a free
/// per shard without, each parsed, dispatched and answered in full).
const SINGLE_ROUND_WINDOW: usize = 20;

/// Top-level body keys a single round answers exactly as query_then_fetch
/// does. Left out: what renders differently or fails in a different phase
/// when the shard renders the hits itself (highlighting counts offsets in
/// the fetch phase; `fields`, `docvalue_fields`, `script_fields` and
/// runtime fields fail in phase `fetch`; `timeout` would also bound the
/// rendering; `stats` groups count fetches of their own).
const SINGLE_ROUND_KEYS: &[&str] = &[
    "query", "from", "size", "sort", "track_total_hits", "track_scores", "_source", "aggs", "aggregations", "post_filter",
    "min_score", "search_after", "version", "seq_no_primary_term", "suggest", "indices_boost",
];

/// Hit tag of a single round: the position of the shard answer it came
/// from in that round ([`round_answer`]).
const ROUND_TAG: &str = "_rustic_round_part";

/// Tests only: answer every search [`single_round`] would take in two
/// phases, to compare the two.
#[doc(hidden)]
pub static SINGLE_ROUND_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static TWO_PHASE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SINGLE_ROUNDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Searches this process answered in a single round ([`single_round`]).
pub fn single_rounds() -> u64 {
    SINGLE_ROUNDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Can a search [`eligible`] for query_then_fetch answer in a single round
/// instead (ES `query_and_fetch`): every shard returns its top `from +
/// size` hits rendered, the coordinator merges them, and no fetch round
/// follows? For a small page of plain hits (see [`SINGLE_ROUND_KEYS`]) whose
/// every shard copy is on this node (`all_local`: the coordinator counts
/// the fetch phase the shards leave to it, [`finish_single_round`]).
/// `RUSTIC_TWO_PHASE_SMALL_SEARCH` in the environment disables it.
///
/// The response is the query_then_fetch one: the same shard requests and
/// merge, hits rendered by the same code from the same reader (in two
/// phases the query phase's `_score` and `sort` replace the fetch phase's,
/// which are the same values). Except one order: there the coordinator
/// appends `sort` after `matched_queries`, a shard renders it before them,
/// so a sorted search with named queries stays in two phases.
pub(super) fn single_round(body: &Value, params: &HashMap<String, String>, all_local: bool) -> bool {
    static ENV_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !all_local || SINGLE_ROUND_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
        || *ENV_DISABLED.get_or_init(|| std::env::var_os("RUSTIC_TWO_PHASE_SMALL_SEARCH").is_some())
    {
        return false;
    }
    let Some(o) = body.as_object() else { return false };
    if !o.keys().all(|k| SINGLE_ROUND_KEYS.contains(&k.as_str())) || params.contains_key("timeout") {
        return false;
    }
    if o.contains_key("sort") && o.get("query").is_some_and(|q| has_key(q, "_name")) {
        return false;
    }
    let (from, size) = fanout::from_size(body, params);
    size > 0 && from.saturating_add(size) <= SINGLE_ROUND_WINDOW
}

/// A single round's shard answer ([`round_answer`]): its shard, the copy
/// that answered and the fetch time it left to the coordinator (`None`: it
/// counted its fetch itself).
pub(super) struct RoundAnswer {
    k: u32,
    node: String,
    fetch_nanos: Option<u64>,
}

/// Take a single round's shard answer `part` (the `i`th): its deferred
/// fetch time ([`elasticsearch_http::QTF_FETCH_NS_KEY`]) comes off, its
/// hits are tagged with `i` for [`finish_single_round`].
pub(super) fn round_answer(part: &mut Value, i: usize, k: u32, node: &str) -> RoundAnswer {
    let fetch_nanos = part.as_object_mut().and_then(|o| o.shift_remove(elasticsearch_http::QTF_FETCH_NS_KEY)).and_then(|v| v.as_u64());
    for h in part.get_mut("hits").and_then(|h| h.get_mut("hits")).and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(o) = h.as_object_mut() {
            o.insert(ROUND_TAG.into(), json!(i));
        }
    }
    RoundAnswer { k, node: node.to_string(), fetch_nanos }
}

/// Untag the merged page of a single round and return the shard copies
/// (answer position, `_index`) its hits came from, each once.
fn untag_round_page(merged: &mut Value) -> BTreeSet<(usize, String)> {
    let mut from = BTreeSet::new();
    for h in merged.get_mut("hits").and_then(|h| h.get_mut("hits")).and_then(Value::as_array_mut).into_iter().flatten() {
        let Some(o) = h.as_object_mut() else { continue };
        // The tag went in last: removing it moves no other member.
        let Some(i) = o.shift_remove(ROUND_TAG).and_then(|v| v.as_u64()) else { continue };
        if let Some(index) = o.get("_index").and_then(Value::as_str) {
            from.insert((i as usize, index.to_string()));
        }
    }
    from
}

/// The end of a single round: untag the merged page and count the fetch
/// phase on each shard copy whose hits made it, once, with the time that
/// shard spent rendering, as ES counts query_then_fetch fetches (a shard
/// without winners is never fetched).
pub(super) fn finish_single_round(node: &ClusterNode, merged: &mut Value, answers: &[RoundAnswer]) {
    for (i, index) in untag_round_page(merged) {
        let Some(a) = answers.get(i) else { continue };
        let Some(nanos) = a.fetch_nanos else { continue };
        if a.node != node.local.id {
            continue;
        }
        if let Some(copy) = node.engine_for(a.k).and_then(|e| e.get_index(&index)).and_then(|svc| svc.shard(a.k)) {
            copy.shard.counters.record_fetch(Duration::from_nanos(nanos));
        }
    }
    SINGLE_ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Searches this process answered as query_then_fetch, and eligible ones
/// that selected single-phase execution before sending any query.
pub fn counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (TWO_PHASE.load(Relaxed), FALLBACKS.load(Relaxed))
}

/// Kept for telemetry compatibility. Two-phase searches never reexecute.
pub fn reruns() -> u64 { 0 }

/// A sub-request with its JSON body edited and the phase header set.
fn phase_req(sub: &RestReq, phase: &str, edit: impl FnOnce(&mut Map<String, Value>)) -> EsResult<Option<RestReq>> {
    phase_req_with(sub, phase, edit, |request, body| request.set_json(body))
}

fn phase_req_with(sub: &RestReq, phase: &str, edit: impl FnOnce(&mut Map<String, Value>), encode: impl FnOnce(&mut RestReq, &Value) -> EsResult<()>) -> EsResult<Option<RestReq>> {
    let mut b: Value = optional!(serde_json::from_slice(&sub.body).ok());
    edit(optional!(b.as_object_mut()));
    let mut s = sub.clone();
    encode(&mut s, &b)?;
    s.replace_header(HEADER, phase)?;
    if matches!(phase, "query" | "fetch") {
        light(&mut s)?;
    }
    Ok(Some(s))
}

/// The parts of a shard request body the fetch phase drops: the
/// query-only keys, and the query unless it renders something (highlights,
/// `_name`d queries for `matched_queries`) and there are docs to render.
fn fetch_template(b: &mut Map<String, Value>, has_docs: bool) {
    for key in QUERY_ONLY {
        b.remove(*key);
    }
    // The fetch phase does not search: its query only drives highlights
    // and `matched_queries`. Without them the shard need not parse and
    // rebuild it (rule lookups, `query_string` field expansion, ...).
    let named = || b.get("query").is_some_and(|q| has_key(q, "_name"));
    if !has_docs || (!b.contains_key("highlight") && !named()) {
        b.remove("query");
    }
}

/// Fill a [`fetch_template`] with the page and the context to render.
fn fetch_fill(b: &mut Map<String, Value>, ctx: &str, docs: &[Value]) {
    b.insert("from".into(), json!(0));
    b.insert("size".into(), json!(docs.len()));
    b.insert(elasticsearch_http::QTF_FETCH_KEY.into(), json!({"ctx": ctx, "docs": docs}));
}

/// Fetch-phase request rendering `docs` of reader context `ctx` (none:
/// only free it).
fn fetch_req(sub: &RestReq, ctx: &str, docs: &[Value]) -> EsResult<Option<RestReq>> {
    let mut f = optional!(phase_req(sub, "fetch", |b| {
        fetch_template(b, !docs.is_empty());
        fetch_fill(b, ctx, docs);
    })?);
    if !docs.is_empty() {
        f.replace_header(elasticsearch_http::HIT_SPANS_HEADER, "1")?;
    }
    fetch_headers(f).map(Some)
}

fn fetch_headers(mut f: RestReq) -> EsResult<RestReq> {
    // It reads the context's pinned views, not the copy's current state:
    // no read-copies guard (a free to a copy that just became unreadable
    // must still free; the engine refuses a context whose copy was replaced).
    f.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(copies::READ_COPIES_HEADER))?;
    Ok(f)
}

/// A shard request body parsed once for every phase request built from
/// it. All shard requests of a search carry the same body; a `terms` query
/// with tens of thousands of values (Rally geonames `large_terms`) is
/// hundreds of KB, and parsing and re-encoding it for each shard's query,
/// free and fetch request cost more than the shards' searches.
struct PhaseBodies {
    /// The body these were derived from.
    source: axum::body::Bytes,
    /// The query-phase request (`_source: false`, header set), whose body
    /// every shard's query request copies.
    query: RestReq,
    /// The fetch body without docs (a free) and with docs, before
    /// [`fetch_fill`].
    free: Map<String, Value>,
    fetch: Map<String, Value>,
}

impl PhaseBodies {
    fn new(base: &RestReq) -> EsResult<Option<Arc<Self>>> {
        let query = optional!(phase_req(base, "query", |b| {
            b.insert("_source".into(), json!(false));
        })?);
        let parsed: Value = optional!(serde_json::from_slice(&base.body).ok());
        let mut free = optional!(parsed.as_object()).clone();
        let mut fetch = free.clone();
        fetch_template(&mut free, false);
        fetch_template(&mut fetch, true);
        Ok(Some(Arc::new(Self { source: base.body.clone(), query, free, fetch })))
    }

    /// The bodies of `base`, reusing `known` ones of an identical body.
    fn of(base: &RestReq, known: &mut Vec<Arc<Self>>) -> EsResult<Option<Arc<Self>>> {
        if let Some(b) = known.iter().find(|b| b.source == base.body) {
            return Ok(Some(b.clone()));
        }
        let b = optional!(Self::new(base)?);
        known.push(b.clone());
        Ok(Some(b))
    }

    /// [`fetch_req`] of `base` (whose body is [`Self::source`]).
    fn fetch_req(&self, base: &RestReq, ctx: &str, docs: &[Value]) -> EsResult<RestReq> {
        let mut b = if docs.is_empty() { self.free.clone() } else { self.fetch.clone() };
        // The body `fetch_fill` makes, its docs (a page's worth, thousands)
        // written straight out rather than as one tree serialized twice
        // (counted, then written): the rest with no docs, the docs spliced in.
        use std::io::Write as _;
        fetch_fill(&mut b, ctx, &[]);
        b.insert("size".into(), json!(docs.len()));
        let head = serde_json::to_vec(&Value::Object(b)).map_err(|e| elasticsearch_engine::error::EsError::internal(e.to_string()))?;
        let mut f = match head.strip_suffix(b"[]}}") {
            Some(head) if !docs.is_empty() => base.try_with_encoded_body(|w| {
                w.write_all(head)?;
                w.write_all(b"[")?;
                for (n, doc) in docs.iter().enumerate() {
                    if n > 0 {
                        w.write_all(b",")?;
                    }
                    serde_json::to_writer(&mut *w, doc)?;
                }
                w.write_all(b"]}}")
            })?,
            _ => {
                let mut b = if docs.is_empty() { self.free.clone() } else { self.fetch.clone() };
                fetch_fill(&mut b, ctx, docs);
                let mut f = base.clone();
                f.set_json(&Value::Object(b))?;
                f
            }
        };
        f.replace_header(HEADER, "fetch")?;
        light(&mut f)?;
        if !docs.is_empty() {
            // Where the answer's hits are (`spanned_fetch_hits`).
            f.replace_header(elasticsearch_http::HIT_SPANS_HEADER, "1")?;
        }
        fetch_headers(f)
    }
}

/// Reader contexts the coordinator still owes a fetch or a free, by
/// answered shard request: `(free request, copy holding it)`. Dropping the
/// set frees every one still open, whatever ends the search (an early
/// return, a failed phase, the client going away and the request future
/// being dropped); a context its fetch already freed is no longer in the
/// set (a repeated free would only find it missing).
struct OpenContexts<F: FnMut(RestReq, String)> {
    open: Vec<Option<(RestReq, String)>>,
    free: F,
}

impl<F: FnMut(RestReq, String)> OpenContexts<F> {
    fn new(n: usize, free: F) -> Self {
        Self { open: (0..n).map(|_| None).collect(), free }
    }

    /// Context `ctx` of shard request `i` (`sub`), held by `answered`.
    #[cfg(test)]
    fn opened(&mut self, i: usize, sub: &RestReq, ctx: &str, answered: String, fallback: &RestReq) -> EsResult<()> {
        self.opened_with(i, || fetch_req(sub, ctx, &[]), answered, fallback)
    }

    /// [`Self::opened`] with the free request built by `free`.
    fn opened_with(&mut self, i: usize, free: impl FnOnce() -> EsResult<Option<RestReq>>, answered: String, fallback: &RestReq) -> EsResult<()> {
        if self.open.len() <= i {
            self.open.resize_with(i + 1, || None);
        }
        // Retain the prebuilt cleanup first: admission failure must not
        // abandon the context we just learned about.
        self.open[i] = Some((fallback.clone(), answered.clone()));
        if let Some(f) = free()? {
            self.open[i] = Some((f, answered));
        }
        Ok(())
    }

    fn is_open(&self, i: usize) -> bool {
        self.open.get(i).is_some_and(Option::is_some)
    }

    /// Context `i` was fetched (the fetch freed it on the shard).
    fn fetched(&mut self, i: usize) {
        if let Some(c) = self.open.get_mut(i) {
            *c = None;
        }
    }

    /// Free context `i` now.
    fn free_now(&mut self, i: usize) {
        if let Some((f, answered)) = self.open.get_mut(i).and_then(Option::take) {
            (self.free)(f, answered);
        }
    }
}

impl<F: FnMut(RestReq, String)> Drop for OpenContexts<F> {
    fn drop(&mut self) {
        for i in 0..self.open.len() {
            self.free_now(i);
        }
    }
}

/// The reader context id a query-phase response opens, read off its first
/// member (the engine writes `{"_rustic_ctx":"<id>",...`) without parsing
/// the rest: also for a response the coordinator cannot read.
pub(super) fn ctx_of(body: &[u8]) -> Option<String> {
    let rest = body.strip_prefix(format!("{{\"{CTX}\":").as_bytes())?;
    serde_json::Deserializer::from_slice(rest).into_iter::<String>().next()?.ok()
}

/// One query-phase shard request: shard `k` of `idx`. `base` is the
/// single-phase sub-request the phases edit; `query` names the reader
/// context id the coordinator chose for it, which `free` frees.
struct Sub {
    k: u32,
    idx: Vec<String>,
    base: RestReq,
    query: RestReq,
    free: RestReq,
    bodies: Arc<PhaseBodies>,
}

impl Sub {
    #[cfg(test)]
    fn new(base: RestReq, k: u32, idx: Vec<String>) -> EsResult<Option<Sub>> {
        Self::new_shared(base, k, idx, &mut Vec::new())
    }

    /// A query-phase request for `base`, its phase bodies shared with the
    /// other shard requests of the search carrying the same body (`known`).
    fn new_shared(base: RestReq, k: u32, idx: Vec<String>, known: &mut Vec<Arc<PhaseBodies>>) -> EsResult<Option<Sub>> {
        let bodies = optional!(PhaseBodies::of(&base, known)?);
        // The context id is ours, so a copy we give up on can be told to
        // free it (the shard drops a free that arrives first).
        let id = elasticsearch_engine::index::new_uuid();
        // Query phase: no sources; fetch-only parts are checked, not
        // rendered, by the shard. Same body as `bodies.query`, this
        // request's URI and headers.
        let mut query = base.clone();
        query.set_body(&bodies.query.body)?;
        query.replace_header(HEADER, "query")?;
        query.replace_header(CTX_HEADER, &id)?;
        query.replace_header(PINNED_HEADER, "1")?;
        light(&mut query)?;
        // Where its hits are: a large page keeps them as text (`RawQuery`).
        query.replace_header(elasticsearch_http::HIT_SPANS_HEADER, "1")?;
        let free = bodies.fetch_req(&base, &id, &[])?;
        Ok(Some(Sub { k, idx, base, query, free, bodies }))
    }

    /// [`fetch_req`] of this request's base, from the shared bodies.
    fn fetch_req(&self, ctx: &str, docs: &[Value]) -> EsResult<RestReq> {
        self.bodies.fetch_req(&self.base, ctx, docs)
    }
}

/// A query-phase shard request as the fan-out runs one
/// ([`super::run_unit`]): `whole` over every index of the shard group, on
/// `cands`; when no copy answered it, `singles` (one per index, empty for a
/// single index) go to their own copies.
struct Unit {
    whole: Sub,
    cands: Vec<String>,
    singles: Vec<Sub>,
}

/// What became of a query-phase shard request.
enum QueryOut {
    /// Copy `node`'s answer (a failed one is a shard failure).
    /// `parsed`: the body, parsed on the task that received it.
    Answer { sub: Sub, node: String, resp: RestResp, parsed: Option<Box<Parsed>> },
    Failed(Vec<copies::ShardFailure>),
    /// A request error: the search's answer as it is.
    Request(RestResp),
}

/// One batch of query answers (a split unit's per-index answers come
/// together). Dropped before every answer was taken (a phase error
/// returned mid-batch), it frees the context of each successful answer
/// left, as ES releases every shard result on failure; neither the pending
/// requests nor the open set know of them. Failed answers were sent their
/// free already ([`query_copies`]).
struct Batch<F: FnMut(RestReq, String)> {
    outs: std::vec::IntoIter<QueryOut>,
    free: F,
}

impl<F: FnMut(RestReq, String)> Iterator for Batch<F> {
    type Item = QueryOut;
    fn next(&mut self) -> Option<QueryOut> {
        self.outs.next()
    }
}

impl<F: FnMut(RestReq, String)> Drop for Batch<F> {
    fn drop(&mut self) {
        for out in self.outs.by_ref() {
            if let QueryOut::Answer { sub, node, resp, .. } = out {
                if resp.is_success() {
                    (self.free)(sub.free, node);
                }
            }
        }
    }
}

/// The query-phase requests of a search still in flight. If the search
/// ends before they all answer (the client went away, a timeout dropped the
/// request future), the rest move to a task that waits for them and frees
/// every context they open: the shards register them all the same. Before
/// the first poll nothing was sent, and nothing is (`started`).
struct Pending<F: std::future::Future<Output = Vec<QueryOut>> + Send + 'static> {
    q: futures::stream::FuturesUnordered<F>,
    started: bool,
    node: Arc<ClusterNode>,
    /// Set when the search is abandoned: a request still running tries no
    /// further copy (it would only be freed).
    abandoned: Arc<std::sync::atomic::AtomicBool>,
}

impl<F: std::future::Future<Output = Vec<QueryOut>> + Send + 'static> Drop for Pending<F> {
    fn drop(&mut self) {
        if self.q.is_empty() || !self.started {
            return;
        }
        self.abandoned.store(true, std::sync::atomic::Ordering::Relaxed);
        let (mut q, node) = (std::mem::take(&mut self.q), self.node.clone());
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                while let Some(outs) = futures::StreamExt::next(&mut q).await {
                    for out in outs {
                        // Failed answers were sent their free already.
                        let QueryOut::Answer { sub, node: answered, resp, .. } = out else { continue };
                        if resp.is_success() {
                            let f = ctx_of(&resp.body).and_then(|ctx| fetch_req(&sub.base, &ctx, &[]).ok().flatten()).unwrap_or(sub.free);
                            copy_request(&node, &f, &answered, None).await;
                        }
                    }
                }
            });
        }
    }
}

/// `sub`'s query phase on each of `cands` in turn, as [`super::try_copies`]
/// runs a shard request: any failure but a request error (a 429 breaker
/// trip, a script error, a timeout, a 5xx, a copy gone) moves on to the next
/// copy, and the most meaningful failure is the shard's. Each attempt waits
/// at most `bound` and gives up when the copy's node leaves
/// ([`copy_request`]). A copy whose attempt failed server side or gave no
/// answer (5xx: a timeout, a node gone) may still run the phase and
/// register its context, so it is sent `sub.free` (which it also honours
/// before the phase arrives) without waiting; one that refused the request
/// (4xx) opened none. Once the search is `abandoned`, no further copy is
/// tried.
async fn query_copies(node: &Arc<ClusterNode>, sub: &Sub, cands: &[String], bound: Option<Duration>, abandoned: &std::sync::atomic::AtomicBool) -> Tried {
    let mut best: Option<(RestResp, String)> = None;
    let mut dead = Vec::new();
    for c in cands {
        if best.is_some() && abandoned.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let r = super::copy_query_request(node, &sub.query, c, bound).await;
        if r.status >= 500 {
            spawn_free(node)(sub.free.clone(), c.clone());
        }
        if copies::answered(&r) {
            return Tried::Answered(r, c.clone());
        }
        if copies::is_request_error(&r) {
            return Tried::Request(r);
        }
        if copies::node_failure(&r) {
            dead.push(c.clone());
        }
        if best.as_ref().map_or(true, |(prev, _)| copies::replaces(prev, &r)) {
            best = Some((r, c.clone()));
        }
    }
    let (resp, node) = best.unwrap_or_else(|| (RestResp::error(503, "no_shard_available_action_exception", "no active shard copy"), String::new()));
    Tried::Failed { resp, node, dead }
}

/// One shard request's query phase (see [`super::run_unit`]): the copy
/// that answers is pinned in `plan` for the fetch phase. A group of indices
/// that no copy answered is split per index, each on its own copies,
/// skipping nodes that failed as nodes.
async fn query_unit(node: Arc<ClusterNode>, plan: Arc<copies::Plan>, unit: Unit, bound: Option<Duration>, abandoned: Arc<std::sync::atomic::AtomicBool>) -> Vec<QueryOut> {
    let answer = |sub: Sub, n: String, resp: RestResp| {
        for i in &sub.idx {
            plan.pin(&format!("{i}#{}", sub.k), &n);
        }
        QueryOut::Answer { sub, node: n, resp, parsed: None }
    };
    let Unit { whole, cands, singles } = unit;
    let tried = query_copies(&node, &whole, &cands, bound, &abandoned).await;
    match tried {
        Tried::Answered(r, n) => vec![answer(whole, n, r)],
        Tried::Request(r) => vec![QueryOut::Request(r)],
        Tried::Failed { resp, node: n, dead } if !singles.is_empty() => {
            let (node, resp, n, answer, abandoned) = (&node, &resp, &n, &answer, &abandoned);
            let singles = singles.into_iter().map(|one| {
                let cands: Vec<String> = plan.candidates(&format!("{}#{}", one.idx[0], one.k)).into_iter().filter(|c| !dead.contains(c)).collect();
                async move {
                    if cands.is_empty() || abandoned.load(std::sync::atomic::Ordering::Relaxed) {
                        return QueryOut::Failed(copies::ShardFailure::from_response(resp, &one.idx, one.k, n));
                    }
                    let tried = query_copies(node, &one, &cands, bound, abandoned).await;
                    match tried {
                        Tried::Answered(r, c) => answer(one, c, r),
                        Tried::Request(r) => QueryOut::Request(r),
                        Tried::Failed { resp, node: c, .. } => QueryOut::Failed(copies::ShardFailure::from_response(&resp, &one.idx, one.k, &c)),
                    }
                }
            });
            futures::future::join_all(singles).await
        }
        Tried::Failed { resp, node: n, .. } => vec![QueryOut::Failed(copies::ShardFailure::from_response(&resp, &whole.idx, whole.k, &n))],
    }
}

/// A shard answer parsed on the task that received it, into a source
/// table of its own part of the search
/// ([`elasticsearch_engine::source_json::ResponseSources::for_part`]);
/// `value` is `None` when it is unreadable.
struct Parsed {
    value: Option<Value>,
    sources: elasticsearch_engine::source_json::ResponseSources,
    /// A query answer's [`prepare_query_part`], done on its task too.
    query: Option<(Option<String>, bool, u32)>,
    /// A large page's query hits kept as text ([`RawQuery`]); `value`'s
    /// hit list is then empty.
    raw: Option<RawQuery>,
}

impl Parsed {
    fn parse(part: u32, body: &[u8]) -> Self {
        let mut sources = elasticsearch_engine::source_json::ResponseSources::for_part(part);
        let value = sources.parse_lazy(body).ok();
        Parsed { value, sources, query: None, raw: None }
    }

    /// A query answer of a [`raw_query_eligible`] search whose hit spans
    /// the shard reported, its hits kept as text; `None` when the answer is
    /// not shaped as expected (it is then parsed whole).
    fn query_raw(part: u32, body: &axum::body::Bytes, spans: &str) -> Option<Self> {
        let (meta, hits) = spanned_hits(body, spans)?;
        let mut names: Vec<String> = Vec::new();
        let mut keys = Vec::with_capacity(hits.len());
        for hit in &hits {
            let index = leading_index(body, hit.text.start)?;
            let at = match names.iter().position(|n| *n == *index) {
                Some(at) => at,
                None => { names.push(index.into_owned()); names.len() - 1 }
            };
            let score = match body.get(hit.score.clone())? {
                b"null" => None,
                text => Some(serde_json::from_slice::<f64>(text).ok()?),
            };
            keys.push((score, at as u32));
        }
        Some(Parsed { value: Some(meta), sources: elasticsearch_engine::source_json::ResponseSources::for_part(part), query: None,
            raw: Some(RawQuery { body: body.clone(), hits, keys, names }) })
    }
}

/// A query answer's hits kept as text: a search over many shards with a
/// large page (Rally's `search_limit_10000`: 5 shard requests of 10,000
/// hits each) needs only `from + size` of them, so the others never become
/// trees ([`select_raw_winners`]). `keys`: each hit's `_score` and
/// `_index` (a position in `names`), what the merge orders them by.
struct RawQuery {
    body: axum::body::Bytes,
    hits: Vec<RawHit>,
    keys: Vec<(Option<f64>, u32)>,
    names: Vec<String>,
}

/// Whether the merge of a search orders its hits by `_score` and `_index`
/// alone (no sort, collapse, kNN, rank or aggregation, which read more of
/// each hit), with a page large enough for [`RawQuery`] to pay.
fn raw_query_eligible(body: &Value, params: &HashMap<String, String>) -> bool {
    const KEYS: &[&str] = &["query", "from", "size", "track_total_hits", "track_scores", "_source", "stored_fields", "docvalue_fields",
        "fields", "version", "seq_no_primary_term", "min_score", "post_filter", "timeout", "terminate_after"];
    let Some(o) = body.as_object() else { return false };
    if !o.keys().all(|k| KEYS.contains(&k.as_str())) || params.contains_key("scroll") || params.contains_key("sort") {
        return false;
    }
    let (from, size) = fanout::from_size(body, params);
    from.saturating_add(size) >= RAW_QUERY_MIN_HITS
}

/// Pages from which [`RawQuery`] keeps the query hits as text.
const RAW_QUERY_MIN_HITS: usize = 1000;

/// Turn the [`RawQuery`] hits of `parts` (in shard order) into the hit
/// lists the merge reads: when every part is raw, only the `keep` best by
/// the merge's order (`_score` descending, `_index`, then shard and hit
/// order, as its stable sort leaves ties), each in its part's hit order;
/// else all of them. Each is tagged with its part's `tag`.
fn select_raw_winners(parts: &mut [(u32, Value, Option<RawQuery>, u32)], keep: usize) -> Option<()> {
    let all_raw = parts.iter().all(|p| p.2.is_some());
    let mut chosen: Vec<(usize, usize)> = {
        let mut items: Vec<(Option<f64>, &str, usize, usize)> = Vec::new();
        for (pi, (_, _, raw, _)) in parts.iter().enumerate() {
            if let Some(raw) = raw {
                items.extend(raw.keys.iter().enumerate().map(|(hi, (score, name))| (*score, raw.names[*name as usize].as_str(), pi, hi)));
            }
        }
        let order = |a: &(Option<f64>, &str, usize, usize), b: &(Option<f64>, &str, usize, usize)| fanout::score_order(a.0, b.0).then_with(|| a.1.cmp(b.1)).then_with(|| (a.2, a.3).cmp(&(b.2, b.3)));
        if all_raw && items.len() > keep {
            if keep == 0 {
                items.clear();
            } else {
                items.select_nth_unstable_by(keep - 1, order);
                items.truncate(keep);
            }
        }
        items.into_iter().map(|(_, _, pi, hi)| (pi, hi)).collect()
    };
    chosen.sort_unstable();
    // Built in parallel (a 10,000-hit page): each from its text as its
    // parse would give it. A light query hit is `{"_index":..,"_score":..,
    // "_rustic_doc":[t,l,d]}`; any other shape is parsed.
    let built: Vec<Option<Value>> = {
        use rayon::prelude::*;
        let parts = &*parts;
        chosen.par_iter().with_min_len(512).map(|&(pi, hi)| {
            let (_, _, raw, tag) = &parts[pi];
            let raw = raw.as_ref()?;
            let (hit, key) = (&raw.hits[hi], raw.keys[hi]);
            let mut v = if hit.score.end == hit.cut {
                let mut m = Map::with_capacity(4);
                m.insert("_index".into(), Value::String(raw.names[key.1 as usize].clone()));
                m.insert("_score".into(), serde_json::from_slice(raw.body.get(hit.score.clone())?).ok()?);
                m.insert(DOC_REF.into(), Value::Array(hit.at.iter().map(|&n| Value::from(n)).collect()));
                Value::Object(m)
            } else {
                serde_json::from_slice(raw.body.get(hit.text.clone())?).ok()?
            };
            v.as_object_mut()?.insert(TAG.into(), json!(*tag));
            Some(v)
        }).collect()
    };
    let mut built = built.into_iter();
    let mut next = 0;
    for (pi, (_, part, raw, _)) in parts.iter_mut().enumerate() {
        if raw.take().is_none() { continue };
        let mut hits = Vec::new();
        while next < chosen.len() && chosen[next].0 == pi {
            hits.push(built.next()??);
            next += 1;
        }
        *part.pointer_mut("/hits/hits")? = Value::Array(hits);
    }
    Some(())
}

/// A readable query answer `part` of `sub` from copy `answered`, made
/// ready to merge: its reader context id taken off, and when it is valid
/// ([`valid_query_part`]) its shard counts normalized, its failures
/// labelled and its hits tagged with `tag` (its part of the search).
/// Returns the context id, whether it is valid, and `tag`.
fn prepare_query_part(part: &mut Value, sub: &Sub, answered: &str, tag: u32) -> (Option<String>, bool, u32) {
    let ctx = part.as_object_mut().and_then(|o| o.remove(CTX)).and_then(|c| c.as_str().map(str::to_string));
    if !valid_query_part(part, ctx.as_deref()) {
        return (ctx, false, tag);
    }
    fanout::normalize_part_shards(part, sub.idx.len());
    label_part_failures(part, sub.k, &sub.idx, answered);
    for h in part.get_mut("hits").and_then(|h| h.get_mut("hits")).and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(o) = h.as_object_mut() {
            o.insert(TAG.into(), json!(tag));
        }
    }
    (ctx, true, tag)
}

fn protocol_failure(reason: &str) -> RestResp {
    RestResp::error(500, "search_phase_execution_exception", reason)
}

fn valid_query_part(part: &Value, context: Option<&str>) -> bool {
    part.pointer("/hits/hits").and_then(Value::as_array).is_some_and(|hits|
            (hits.is_empty() || context.is_some_and(|context| !context.is_empty())) && hits.iter().all(|hit| member(hit, DOC_REF).and_then(Value::as_array).is_some_and(|address| address.len() == 3 && address.iter().all(|value| value.as_u64().is_some()))
                && hit.get("_index").and_then(Value::as_str).is_some()
                && hit.get("_id").is_none_or(Value::is_string)))
}

/// Whether shard request `ordinal`'s fetch answer holds exactly its
/// winners: as many hits as winners, each with a string `_index` and `_id`,
/// and every winner's `(_index, _id)` among them exactly once.
///
/// Linear in winners + hits (one hash count of the hits). Comparing every
/// winner against every hit was quadratic: a `size: 10000` page whose
/// winners all come from one shard (uniform scores tie-break by shard) cost
/// 10^8 JSON comparisons, ~10 s, in the geopoint `distanceFilter10kGTE`.
fn valid_fetch_part(part: &Value, winners: &[Value], ordinal: usize) -> bool {
    let Some(hits) = part.pointer("/hits/hits").and_then(Value::as_array) else { return false; };
    let wanted = || winners.iter().filter(|hit| member(hit, TAG).and_then(Value::as_u64) == Some(ordinal as u64));
    if hits.len() != wanted().count() {
        return false;
    }
    if by_address(part) {
        // Hits matched to winners by address ([`LIGHT_HEADER`]).
        let mut counts: HashMap<[u64; 3], usize> = HashMap::with_capacity(hits.len());
        for hit in hits {
            let Some(at) = address(hit) else { return false };
            *counts.entry(at).or_insert(0) += 1;
        }
        return wanted().all(|winner| address(winner).is_some_and(|at| counts.get(&at) == Some(&1)));
    }
    let mut counts: HashMap<(&str, &str), usize> = HashMap::with_capacity(hits.len());
    for hit in hits {
        let (Some(index), Some(id)) = (hit.get("_index").and_then(Value::as_str), hit.get("_id").and_then(Value::as_str)) else { return false };
        *counts.entry((index, id)).or_insert(0) += 1;
    }
    // Every hit has a string `_index` and `_id`, so a winner without them
    // matches none (as `Value` equality against a string never holds).
    wanted().all(|winner| match (winner.get("_index").and_then(Value::as_str), winner.get("_id").and_then(Value::as_str)) {
        (Some(index), Some(id)) => counts.get(&(index, id)) == Some(&1),
        _ => false,
    })
}

/// Drop `parts` (every shard's query hits) off the request's path when
/// there are many of them: freeing a `size: 10000` search's 250,000 hit
/// trees takes milliseconds the response need not wait for.
fn drop_later(parts: Vec<Value>) {
    let hits: usize = parts.iter().map(|p| p.pointer("/hits/hits").and_then(Value::as_array).map_or(0, Vec::len)).sum();
    if hits < 4096 {
        return;
    }
    if let Ok(rt) = tokio::runtime::Handle::try_current() {
        drop(rt.spawn_blocking(move || drop(parts)));
    }
}

/// A fetch answer, parsed on the task that received it.
enum FetchParsed {
    /// Its hits kept as text ([`raw_fetch_hits`]) in its body.
    Raw(axum::body::Bytes, Vec<RawHit>),
    Value(Box<Parsed>),
}

/// A fetched hit as the page takes it.
enum Fetched {
    Value(Value),
    /// A hit kept as text in a fetch answer's body.
    Raw(axum::body::Bytes, RawHit),
}

/// A fetched hit kept as text: its address, and where in its answer's body
/// it is (`text`), its `_score` value is, and its address member starts
/// (`cut`: the comma before it; the address is its last member).
struct RawHit {
    at: [u64; 3],
    text: std::ops::Range<usize>,
    score: std::ops::Range<usize>,
    cut: usize,
}

/// The hits of a fetch answer whose every hit ends with its address (a
/// shard that read [`LIGHT_HEADER`]), each kept as text in `body`:
/// neither its `_source` nor anything else becomes a tree. `None` for any
/// other answer (parsed as a `Value` instead).
fn raw_fetch_hits(body: &[u8]) -> Option<Vec<RawHit>> {
    use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
    type Hit<'a> = RawMembers<'a>;
    /// `hits` (an object): its `hits` list, each hit as its members' texts.
    struct Hits;
    impl<'de> serde::de::DeserializeSeed<'de> for Hits {
        type Value = Option<Vec<Hit<'de>>>;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> { d.deserialize_map(self) }
    }
    impl<'de> Visitor<'de> for Hits {
        type Value = Option<Vec<Hit<'de>>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("hits") }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = None;
            while let Some(k) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                if k == "hits" && out.is_none() { out = Some(map.next_value::<Vec<Hit<'de>>>()?); } else { map.next_value::<IgnoredAny>()?; }
            }
            Ok(out)
        }
    }
    struct Answer;
    impl<'de> Visitor<'de> for Answer {
        type Value = Option<Vec<Hit<'de>>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a fetch answer") }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = None;
            while let Some(k) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                if k == "hits" && out.is_none() { out = map.next_value_seed(Hits)?; } else { map.next_value::<IgnoredAny>()?; }
            }
            Ok(out)
        }
    }
    let mut d = serde_json::Deserializer::from_slice(body);
    let hits = d.deserialize_map(Answer).ok()??;
    d.end().ok()?;
    if hits.is_empty() {
        return None;
    }
    let base = body.as_ptr() as usize;
    let off = |t: &str| t.as_ptr() as usize - base;
    hits.into_iter().map(|RawMembers(members)| {
        let (last, at_text) = members.last()?;
        if last != DOC_REF {
            return None;
        }
        let at: [u64; 3] = serde_json::from_str(at_text.get()).ok()?;
        // The hit's text: from the `{` before its first member's key (the
        // engine writes hits without whitespace) to the `}` after its last
        // value.
        const ADDRESS: &str = ",\"_rustic_doc\":";
        let end = off(at_text.get()) + at_text.get().len();
        let cut = off(at_text.get()).checked_sub(ADDRESS.len())?;
        let first = members.first()?;
        let start = off(first.1.get()).checked_sub(serde_json::to_string(&*first.0).ok()?.len() + 2)?;
        if body.get(start) != Some(&b'{') || body.get(end) != Some(&b'}') || body.get(cut..cut + ADDRESS.len()) != Some(ADDRESS.as_bytes()) {
            return None;
        }
        let end = end + 1;
        // A hit without a `_score` or with a `sort` of its own (a fetch
        // phase renders neither) takes the parsed path.
        let (_, score) = members.iter().find(|(k, _)| k == "_score")?;
        if members.iter().any(|(k, _)| k == "sort") {
            return None;
        }
        Some(RawHit { at, text: start..end, score: off(score.get())..off(score.get()) + score.get().len(), cut })
    }).collect()
}

/// [`raw_fetch_hits`] of a fetch answer whose hits' spans the shard
/// reported ([`elasticsearch_http::encode_hit_spans`]): each hit is found
/// from its span and read only up to its `_score` and from its address, not
/// through its `_source`. The same hits [`raw_fetch_hits`] reads; `None`
/// when the spans do not fit the body or a hit is not shaped as the engine
/// renders a light fetch hit (`{"_index":..,"_id":..,"_score":..,...,
/// "_rustic_doc":[t,l,d]}`), when the answer is read whole.
fn spanned_fetch_hits(body: &[u8], spans: &str) -> Option<Vec<RawHit>> {
    spanned_hits(body, spans).map(|(_, hits)| hits)
}

/// [`spanned_fetch_hits`] with the answer's other members (its hit list
/// left empty).
fn spanned_hits(body: &[u8], spans: &str) -> Option<(Value, Vec<RawHit>)> {
    const ADDRESS: &[u8] = b",\"_rustic_doc\":";
    let mut nums = spans.split(',').map(|n| n.parse::<usize>().ok());
    let tail = nums.next()??;
    let lens: Vec<usize> = nums.collect::<Option<_>>()?;
    if lens.is_empty() {
        return None;
    }
    let end = body.len().checked_sub(tail)?;
    let mut start = end.checked_sub(lens.iter().sum::<usize>() + lens.len() - 1)?;
    // `[` before the first hit, `]` after the last, a comma between two.
    if body.get(start.checked_sub(1)?) != Some(&b'[') || body.get(end) != Some(&b']') {
        return None;
    }
    // The answer's other members (before the hits) as `raw_fetch_hits`
    // requires: one `hits.hits` list, the rest of the body well formed.
    let mut rest = Vec::with_capacity(start + body.len() - end);
    rest.extend_from_slice(&body[..start]);
    rest.extend_from_slice(&body[end..]);
    let meta: Value = serde_json::from_slice(&rest).ok()?;
    if !meta.pointer("/hits/hits").and_then(Value::as_array).is_some_and(Vec::is_empty) {
        return None;
    }
    let mut out = Vec::with_capacity(lens.len());
    for (n, len) in lens.iter().enumerate() {
        let text = body.get(start..start + len)?;
        if text.first() != Some(&b'{') || text.last() != Some(&b'}') {
            return None;
        }
        // The address: the last member, `[t,l,d]`.
        let cut = start + text.windows(ADDRESS.len()).rposition(|w| w == ADDRESS)?;
        let at: [u64; 3] = serde_json::from_slice(body.get(cut + ADDRESS.len()..start + len - 1)?).ok()?;
        out.push(RawHit { at, text: start..start + len, score: leading_score(body, start)?, cut });
        start += len;
        if n + 1 < lens.len() {
            if body.get(start) != Some(&b',') {
                return None;
            }
            start += 1;
        }
    }
    Some((meta, out))
}

/// The `_index` of the hit at `at` (its `{`), its first member as the
/// engine renders a hit; `None` otherwise.
fn leading_index(body: &[u8], at: usize) -> Option<std::borrow::Cow<'_, str>> {
    const KEY: &[u8] = b"{\"_index\":\"";
    if body.get(at..at + KEY.len())? != KEY {
        return None;
    }
    let start = at + KEY.len() - 1;
    let mut i = start + 1;
    loop {
        match body.get(i)? {
            b'\\' => i += 2,
            b'"' => break,
            _ => i += 1,
        }
    }
    let text = body.get(start..=i)?;
    if !text.contains(&b'\\') {
        return std::str::from_utf8(&text[1..text.len() - 1]).ok().map(std::borrow::Cow::Borrowed);
    }
    serde_json::from_slice::<String>(text).ok().map(std::borrow::Cow::Owned)
}

/// The text range of the `_score` value of the hit at `at` (its `{`), read
/// over the members before it, each a string or a scalar (`_index`, `_id`):
/// `None` when another kind of member comes first.
fn leading_score(body: &[u8], at: usize) -> Option<std::ops::Range<usize>> {
    // The end (after the closing quote) of the string starting at `i`.
    let string_end = |mut i: usize| -> Option<usize> {
        if body.get(i) != Some(&b'"') {
            return None;
        }
        i += 1;
        loop {
            match body.get(i)? {
                b'\\' => i += 2,
                b'"' => return Some(i + 1),
                _ => i += 1,
            }
        }
    };
    // The end of the scalar (number, literal) starting at `i`.
    let scalar_end = |mut i: usize| -> Option<usize> {
        while !matches!(body.get(i)?, b',' | b'}') {
            if matches!(body[i], b'{' | b'[' | b'"') {
                return None;
            }
            i += 1;
        }
        Some(i)
    };
    let mut i = at + 1;
    loop {
        let key_end = string_end(i)?;
        let key = &body[i + 1..key_end - 1];
        if body.get(key_end) != Some(&b':') {
            return None;
        }
        let value = key_end + 1;
        if key == b"_score" {
            return Some(value..scalar_end(value)?);
        }
        let value_end = if body.get(value) == Some(&b'"') { string_end(value)? } else { scalar_end(value)? };
        if body.get(value_end) != Some(&b',') {
            return None;
        }
        i = value_end + 1;
    }
}

/// An object's members, borrowed from its text.
struct RawMembers<'a>(Vec<(std::borrow::Cow<'a, str>, &'a serde_json::value::RawValue)>);
impl<'de> serde::Deserialize<'de> for RawMembers<'de> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = RawMembers<'de>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("an object") }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(8);
                while let Some(entry) = map.next_entry()? { out.push(entry); }
                Ok(RawMembers(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// [`valid_fetch_part`] of hits kept as text: as many as the winners of
/// shard request `ordinal`, each winner's address among them exactly once.
fn valid_raw_fetch_part(hits: &[RawHit], winners: &[Value], ordinal: usize) -> bool {
    let wanted = || winners.iter().filter(|hit| member(hit, TAG).and_then(Value::as_u64) == Some(ordinal as u64));
    if hits.len() != wanted().count() {
        return false;
    }
    let mut counts: HashMap<[u64; 3], usize> = HashMap::with_capacity(hits.len());
    for hit in hits {
        *counts.entry(hit.at).or_insert(0) += 1;
    }
    wanted().all(|winner| address(winner).is_some_and(|at| counts.get(&at) == Some(&1)))
}

/// Fetched hit `hit` (text in `body`) as the page shows it: the query
/// hit `winner`'s `_score` in place of its own, the winner's `sort` (if
/// any) appended, its address left out, as the `Value` edits of a parsed
/// hit do. `None` when its text is not what it was read as.
fn raw_page_hit(body: &[u8], hit: &RawHit, winner: &Value) -> EsResult<Option<String>> {
    // UTF-8: the answer was read as JSON text.
    let Some(text) = body.get(hit.text.clone()).and_then(|t| std::str::from_utf8(t).ok()) else { return Ok(None) };
    let score = match member(winner, "_score").unwrap_or(&Value::Null) {
        // The common case, written as `to_raw_value` writes it, without
        // allocating it apart.
        Value::Number(n) => std::borrow::Cow::Owned(n.to_string()),
        Value::Null => std::borrow::Cow::Borrowed("null"),
        other => std::borrow::Cow::Owned(raw_text(other)?.get().to_string()),
    };
    let sort = member(winner, "sort").map(raw_text).transpose()?;
    let local = |at: usize| at - hit.text.start;
    let (head, middle) = (&text[..local(hit.score.start)], &text[local(hit.score.end)..local(hit.cut)]);
    // Exactly sized: the page keeps it as a boxed `str` (no reallocation).
    let mut out = String::with_capacity(head.len() + score.len() + middle.len() + sort.as_ref().map_or(0, |s| s.get().len() + 8) + 1);
    out.push_str(head);
    out.push_str(&score);
    out.push_str(middle);
    if let Some(sort) = sort {
        out.push_str(",\"sort\":");
        out.push_str(sort.get());
    }
    out.push('}');
    Ok(Some(out))
}

/// `v` as JSON text.
fn raw_text(v: &impl serde::Serialize) -> EsResult<Box<serde_json::value::RawValue>> {
    serde_json::value::to_raw_value(v).map_err(|e| elasticsearch_engine::error::EsError::internal(e.to_string()))
}

/// A hit's address (`[target, leaf, doc]`, [`DOC_REF`]).
fn address(hit: &Value) -> Option<[u64; 3]> {
    let a = member(hit, DOC_REF)?.as_array()?;
    match a.as_slice() {
        [t, l, d] => Some([t.as_u64()?, l.as_u64()?, d.as_u64()?]),
        _ => None,
    }
}

/// Whether a fetch answer's hits carry their addresses (a shard that read
/// [`LIGHT_HEADER`]): they are matched to the winners by address, else by
/// `_index` and `_id`.
fn by_address(part: &Value) -> bool {
    part.pointer("/hits/hits").and_then(Value::as_array).is_some_and(|hits| !hits.is_empty() && hits.iter().all(|h| h.get(DOC_REF).is_some()))
}

/// How a fetched hit is matched to its winner: by address, or by `_index`
/// and `_id`.
#[derive(PartialEq, Eq, Hash)]
enum HitKey {
    At([u64; 3]),
    Id(String, String),
}

/// `allow_partial_search_results`, else the cluster's
/// `search.default_allow_partial_results` ([`copies::allow_partial`]).
fn allow_partial(node: &ClusterNode, params: &HashMap<String, String>) -> bool {
    copies::allow_partial(node, params)
}

/// Send a free request to the copy holding the context, without waiting.
fn spawn_free(node: &Arc<ClusterNode>) -> impl FnMut(RestReq, String) {
    let node = node.clone();
    move |f, answered| {
        let node = node.clone();
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                copy_request(&node, &f, &answered, None).await;
            });
        }
    }
}

/// Run the search in two phases over `plan`; `None` falls back to the
/// single-phase fan-out. `units` are the fan-out's shard requests (after a
/// DFS round, each on the copy that answered it), `make` builds one's
/// sub-request, and `failures` are the shards already failed (no copy, DFS).
#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    node: &Arc<ClusterNode>,
    req: &RestReq,
    body: &Value,
    params: &HashMap<String, String>,
    plan: &Arc<copies::Plan>,
    units: &[ShardUnit],
    make: &(dyn Fn(&[String], u32) -> EsResult<RestReq> + Sync),
    failures: &[copies::ShardFailure],
    started: Instant,
) -> Option<RestResp> {
    use std::sync::atomic::Ordering::Relaxed;
    // Choose compatibility fallback before dispatching any query work.
    if !node.state().nodes.values().all(|peer| crate::internal_capabilities::supports(peer, crate::internal_capabilities::Capability::QtfPinnedV1)) {
        FALLBACKS.fetch_add(1, Relaxed);
        return None;
    }
    let out = match run_phases(node, req, body, params, plan, units, make, failures, started).await {
        Ok(out) => out,
        Err(error) => Some(RestResp::json(error.status, &error.to_json())),
    };
    match &out {
        Some(r) if r.is_success() => TWO_PHASE.fetch_add(1, Relaxed),
        Some(_) => 0,
        None => FALLBACKS.fetch_add(1, Relaxed),
    };
    out
}

#[allow(clippy::too_many_arguments)]
async fn run_phases(
    node: &Arc<ClusterNode>,
    req: &RestReq,
    body: &Value,
    params: &HashMap<String, String>,
    plan: &Arc<copies::Plan>,
    units: &[ShardUnit],
    make: &(dyn Fn(&[String], u32) -> EsResult<RestReq> + Sync),
    failures: &[copies::ShardFailure],
    started: Instant,
) -> EsResult<Option<RestResp>> {
    // The fan-out's own per-copy bound ([`copies::copy_bound`]); a hung
    // node is given up when it leaves ([`copy_request`]'s node_left), not
    // after a deadline derived from the keep-alive (a slow but live copy is
    // waited for, as ES waits).
    let bound = copies::copy_bound(params, body);
    let abandoned = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut queries = Pending { q: futures::stream::FuturesUnordered::new(), started: false, node: node.clone(), abandoned: abandoned.clone() };
    // Every shard request usually carries the same body: parse it once.
    let mut bodies: Vec<Arc<PhaseBodies>> = Vec::new();
    // Every request is built before any is sent: a body that cannot be
    // split per shard falls back with nothing in flight.
    let mut built = Vec::with_capacity(units.len());
    for u in units {
        let whole = optional!(Sub::new_shared(make(u.idx.as_slice(), u.k)?, u.k, u.idx.clone(), &mut bodies)?);
        let singles = if u.idx.len() > 1 {
            optional!(u.idx.iter().map(|i| Sub::new_shared(make(std::slice::from_ref(i), u.k)?, u.k, vec![i.clone()], &mut bodies)).collect::<EsResult<Option<Vec<_>>>>()?)
        } else {
            Vec::new()
        };
        built.push(Unit { whole, cands: u.cands.clone(), singles });
    }
    // Answers are parsed where they arrive, each into a source table of its
    // own part (`parts`: the next part number).
    let part_numbers = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let raw_ok = raw_query_eligible(body, params);
    for unit in built {
        // Each shard's query on a task of its own (see `spawn_in_request`):
        // polled together, their synchronous parts ran one after another.
        let (unit, numbers) = (query_unit(node.clone(), plan.clone(), unit, bound, abandoned.clone()), part_numbers.clone());
        let unit = super::spawn_in_request(async move {
            let mut outs = unit.await;
            for out in &mut outs {
                if let QueryOut::Answer { sub, node, resp, parsed } = out {
                    if resp.is_success() {
                        let tag = numbers.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let spans = if raw_ok { resp.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(elasticsearch_http::HIT_SPANS_HEADER)).map(|(_, v)| v.clone()) } else { None };
                        let mut p = spans.and_then(|spans| Parsed::query_raw(tag, &resp.body, &spans)).unwrap_or_else(|| Parsed::parse(tag, &resp.body));
                        if let Some(part) = &mut p.value {
                            let mut prepared = prepare_query_part(part, sub, node, tag);
                            // Hits kept as text need the context too.
                            if p.raw.as_ref().is_some_and(|r| !r.hits.is_empty()) && prepared.0.as_deref().is_none_or(str::is_empty) {
                                prepared.1 = false;
                            }
                            p.query = Some(prepared);
                        }
                        *parsed = Some(Box::new(p));
                    }
                }
            }
            outs
        });
        queries.q.push(async move { unit.await.unwrap_or_else(|e| vec![QueryOut::Request(protocol_failure(&format!("shard query task failed: {e}")))]) });
    }
    // The spawned requests are in flight from here: dropping `queries`
    // frees whatever contexts they open.
    queries.started = true;
    let t_sent = started.elapsed().as_micros();
    // One side table for both phases: `top_hits` sources of the query
    // phase pass through raw, as the fetched hits' do.
    let mut sources = elasticsearch_engine::source_json::ResponseSources::default();
    let mut parts: Vec<(u32, Value, Option<RawQuery>, u32)> = Vec::with_capacity(units.len());
    // Per answered shard request (a hit's tag): the request, the copy that
    // answered (pinned: its context is the one fetched), its context id.
    let mut answers: Vec<(Sub, String, Option<String>)> = Vec::with_capacity(units.len());
    let mut open = OpenContexts::new(units.len(), spawn_free(node));
    let mut failures: Vec<copies::ShardFailure> = failures.to_vec();
    let mut warnings = BTreeSet::new();
    // Each context is owed a free as soon as its shard answers: a search
    // cancelled or dropped while other shards still run frees it too.
    // Once any query has started, a malformed protocol value is an error,
    // never permission to dispatch the whole search again.
    macro_rules! optional {
        ($value:expr) => { match $value { Some(value) => value, None => return Err(elasticsearch_engine::error::EsError::new(500, "search_phase_execution_exception", "invalid query_then_fetch response")) } };
    }
    // Shard request (`answers` position) of each part's hit tag.
    let mut tagged: HashMap<u64, usize> = HashMap::with_capacity(units.len());
    while let Some(outs) = futures::StreamExt::next(&mut queries.q).await {
        let mut request_error = None;
        for out in (Batch { outs: outs.into_iter(), free: spawn_free(node) }) {
            let (sub, answered, r, parsed) = match out {
                QueryOut::Request(r) => {
                    request_error.get_or_insert(r);
                    continue;
                }
                QueryOut::Failed(f) => {
                    failures.extend(f);
                    continue;
                }
                QueryOut::Answer { sub, node, resp, parsed } if resp.is_success() => (sub, node, resp, parsed),
                QueryOut::Answer { sub, node, resp, .. } => {
                    failures.extend(copies::ShardFailure::from_response(&resp, &sub.idx, sub.k, &node));
                    continue;
                }
            };
            let i = answers.len();
            // Preserve completed peers and account this copy as failed. Never
            // repeat its query or the successful peers after protocol failure.
            let (part, prepared, raw) = match parsed {
                Some(p) => {
                    let Parsed { value, sources: table, query, raw } = *p;
                    sources.absorb(table);
                    (value, query, raw)
                }
                None => (sources.parse_lazy(&r.body).ok(), None, None),
            };
            let Some(mut part) = part else {
                if let Some(ctx) = ctx_of(&r.body) {
                    open.opened_with(i, || sub.fetch_req(&ctx, &[]).map(Some), answered.clone(), &sub.free)?;
                } else {
                    // The coordinator-selected ID is still known even when
                    // corrupt JSON hides the returned ID.
                    spawn_free(node)(sub.free.clone(), answered.clone());
                }
                failures.extend(copies::ShardFailure::from_response(&protocol_failure("unreadable query response"), &sub.idx, sub.k, &answered));
                open.free_now(i);
                continue;
            };
            let (ctx, valid, tag) = match prepared {
                Some(prepared) => prepared,
                None => {
                    let tag = part_numbers.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    prepare_query_part(&mut part, &sub, &answered, tag)
                }
            };
            if let Some(ctx) = &ctx {
                open.opened_with(i, || sub.fetch_req(ctx, &[]).map(Some), answered.clone(), &sub.free)?;
            }
            if !valid {
                failures.extend(copies::ShardFailure::from_response(&protocol_failure("query response has no pinned context or invalid hit addresses"), &sub.idx, sub.k, &answered));
                if !open.is_open(i) { spawn_free(node)(sub.free.clone(), answered.clone()); }
                open.free_now(i);
                continue;
            }
            tagged.insert(u64::from(tag), i);
            warnings.extend(r.headers.iter().filter(|(name, _)| name.eq_ignore_ascii_case("warning")).map(|(_, value)| value.clone()));
            parts.push((sub.k, part, raw, tag));
            answers.push((sub, answered, ctx));
        }
        // A request error is the search's answer (the contexts opened so
        // far, and those of requests still running, are freed).
        if let Some(r) = request_error {
            return Ok(Some(r));
        }
    }
    let t_queried = started.elapsed().as_micros();
    let partial_allowed = allow_partial(node, params);
    if !failures.is_empty() {
        if parts.is_empty() {
            return Ok(Some(copies::phase_failure("query", "all shards failed", &failures)));
        }
        if !partial_allowed {
            if failures.len() == plan.unavailable.len() {
                return Ok(Some(stateless_read::missing_shards_rejected(&plan.unavailable)));
            }
            return Ok(Some(copies::phase_failure("query", "Partial shards failure", &failures)));
        }
    }
    // Shard order, not node order: the merge's stable sort then breaks ties
    // by shard as ES does.
    parts.sort_by_key(|p| p.0);
    if parts.iter().any(|p| p.2.is_some()) {
        let (from, size) = fanout::from_size(body, params);
        optional!(select_raw_winners(&mut parts, from.saturating_add(size)));
    }
    let parts: Vec<Value> = parts.into_iter().map(|p| p.1).collect();
    let mut merged = fanout::merge_search_with_source(&parts, body, params, node.engine.max_buckets(), &req.body);
    drop_later(parts);
    // A reduce trip first, as the single-phase fan-out does: ES's 429
    // `search_phase_execution_exception` around the whole breaker error.
    if let Some(e) = fanout::take_agg_breaking() {
        return Ok(Some(RestResp::json(e.status, &e.to_json())));
    }
    if let Some(e) = fanout::take_agg_error() {
        return Ok(Some(RestResp::json(e.status, &e.to_json())));
    }
    copies::add_failures(&mut merged, &failures);
    // Each page hit's tag (its part) becomes its shard request's position.
    for h in merged.pointer_mut("/hits/hits").and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(t) = h.get_mut(TAG) {
            *t = json!(optional!(t.as_u64().and_then(|t| tagged.get(&t))));
        }
    }
    // Taken out: the page is rebuilt from the fetched hits below.
    let page: Vec<Value> = merged.pointer_mut("/hits/hits").and_then(Value::as_array_mut).map(std::mem::take).unwrap_or_default();
    // Winners per answered shard request: `[target, leaf, doc, score]`.
    let mut wanted: Vec<Vec<Value>> = vec![Vec::new(); answers.len()];
    for h in &page {
        let (Some(i), Some(r)) = (member(h, TAG).and_then(Value::as_u64), member(h, DOC_REF).and_then(Value::as_array)) else {
            return Err(elasticsearch_engine::error::EsError::new(500,"search_phase_execution_exception","invalid merged query hit"));
        };
        let mut d = r.clone();
        d.push(h.get("_score").cloned().filter(Value::is_number).unwrap_or(json!(0)));
        optional!(wanted.get_mut(i as usize)).push(Value::Array(d));
    }
    let t_merged = started.elapsed().as_micros();
    // Fetch phase, from each context on the (pinned) copy that holds it.
    let mut fetches = Vec::new();
    for (i, docs) in wanted.iter().enumerate() {
        let (sub, answered, ctx) = &answers[i];
        let f = match (open.is_open(i), ctx) {
            (true, Some(ctx)) => {
                if docs.is_empty() {
                    open.free_now(i);
                    continue;
                }
                sub.fetch_req(ctx, docs)?
            }
            _ if docs.is_empty() => continue,
            _ => return Err(elasticsearch_engine::error::EsError::new(500, "search_context_missing_exception", "query context unavailable before fetch")),
        };
        fetches.push((i, f, answered.clone()));
    }
    // All built before any is sent; each on a task of its own.
    let fetches: Vec<_> = fetches
        .into_iter()
        .map(|(i, f, answered)| {
            let node = node.clone();
            let part = part_numbers.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let fetch = super::spawn_in_request(async move {
                let r = copy_request(&node, &f, &answered, bound).await;
                let spans = r.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(elasticsearch_http::HIT_SPANS_HEADER)).map(|(_, v)| v.as_str());
                let parsed = (r.is_success() && !copies::is_request_error(&r)).then(|| match spans.and_then(|spans| spanned_fetch_hits(&r.body, spans)).or_else(|| raw_fetch_hits(&r.body)) {
                    Some(hits) => FetchParsed::Raw(r.body.clone(), hits),
                    None => FetchParsed::Value(Box::new(Parsed::parse(part, &r.body))),
                });
                (r, parsed)
            });
            async move { (i, fetch.await.unwrap_or_else(|e| (protocol_failure(&format!("shard fetch task failed: {e}")), None))) }
        })
        .collect();
    let mut fetched: HashMap<(usize, HitKey), Fetched> = HashMap::with_capacity(page.len());
    let mut addressed = vec![false; answers.len()];
    let mut lost: Vec<copies::ShardFailure> = Vec::new();
    let mut lost_parts = BTreeSet::new();
    for (i, (r, parsed)) in futures::future::join_all(fetches).await {
        // A request error (one the query phase did not see) is the
        // request's answer, as the shards would all give it (ES: all
        // shards failed), not a lost shard.
        if copies::is_request_error(&r) {
            return Ok(Some(r));
        }
        if !r.is_success() {
            // ES `FetchSearchPhase`: a failed fetch is that shard's failure
            // (not retried on another copy; its context, still open, is
            // freed), and its winners leave the page.
            let (sub, answered, _) = &answers[i];
            lost.extend(copies::ShardFailure::from_response(&r, &sub.idx, sub.k, answered));
            lost_parts.insert(i);
            continue;
        }
        open.fetched(i);
        warnings.extend(r.headers.iter().filter(|(name, _)| name.eq_ignore_ascii_case("warning")).map(|(_, value)| value.clone()));
        let part = match parsed {
            Some(FetchParsed::Raw(body, raw)) => {
                // Hits kept as text, matched by address.
                if !valid_raw_fetch_part(&raw, &page, i) {
                    let (sub, answered, _) = &answers[i];
                    lost.extend(copies::ShardFailure::from_response(&protocol_failure("fetch response omitted or malformed winning hits"), &sub.idx, sub.k, answered));
                    lost_parts.insert(i);
                    continue;
                }
                addressed[i] = true;
                for hit in raw {
                    fetched.insert((i, HitKey::At(hit.at)), Fetched::Raw(body.clone(), hit));
                }
                continue;
            }
            Some(FetchParsed::Value(p)) => {
                let Parsed { value, sources: table, .. } = *p;
                sources.absorb(table);
                value.ok_or(())
            }
            None => sources.parse_lazy(&r.body).map_err(|_| ()),
        };
        let mut part = match part {
            Ok(part) => part,
            Err(_) => {
                let (sub, answered, _) = &answers[i];
                lost.extend(copies::ShardFailure::from_response(&protocol_failure("unreadable fetch response"), &sub.idx, sub.k, answered));
                lost_parts.insert(i);
                continue;
            }
        };
        if !valid_fetch_part(&part, &page, i) {
            let (sub, answered, _) = &answers[i];
            lost.extend(copies::ShardFailure::from_response(&protocol_failure("fetch response omitted or malformed winning hits"), &sub.idx, sub.k, answered));
            lost_parts.insert(i);
            continue;
        }
        addressed[i] = by_address(&part);
        for mut h in part.get_mut("hits").and_then(|h| h.get_mut("hits")).and_then(Value::as_array_mut).map(std::mem::take).into_iter().flatten() {
            let key = if addressed[i] {
                let at = optional!(address(&h));
                // The address went in last: removing it moves no other member.
                optional!(h.as_object_mut()).shift_remove(DOC_REF);
                HitKey::At(at)
            } else {
                HitKey::Id(optional!(h["_index"].as_str()).to_string(), optional!(h["_id"].as_str()).to_string())
            };
            fetched.insert((i, key), Fetched::Value(h));
        }
    }
    let t_fetched = started.elapsed().as_micros();
    if !lost.is_empty() {
        if !partial_allowed {
            let all: Vec<copies::ShardFailure> = failures.iter().chain(&lost).cloned().collect();
            return Ok(Some(copies::phase_failure("fetch", "Partial shards failure", &all)));
        }
        copies::add_lost_failures(&mut merged, &lost);
        // Every shard lost (in either phase): ES's all shards failed, in
        // phase fetch.
        if merged.pointer("/_shards/successful").and_then(Value::as_u64) == Some(0) {
            let all: Vec<copies::ShardFailure> = failures.iter().chain(&lost).cloned().collect();
            return Ok(Some(copies::phase_failure("fetch", "all shards failed", &all)));
        }
    }
    let mut found = Vec::with_capacity(page.len());
    for h in page {
        let i = optional!(member(&h, TAG).and_then(Value::as_u64)) as usize;
        if lost_parts.contains(&i) {
            continue;
        }
        let key = if addressed.get(i).copied().unwrap_or(false) {
            HitKey::At(optional!(address(&h)))
        } else {
            HitKey::Id(optional!(h["_index"].as_str()).to_string(), optional!(h["_id"].as_str()).to_string())
        };
        let f = optional!(fetched.remove(&(i, key)));
        found.push((h, f));
    }
    // A hit kept as text takes the page's edits on its text, in parallel
    // on a large page (each a copy of the hit's text).
    let texts: Vec<EsResult<Option<String>>> = {
        use rayon::prelude::*;
        found.par_iter().with_min_len(256).map(|(h, f)| match f {
            Fetched::Raw(body, hit) => raw_page_hit(body, hit, h),
            Fetched::Value(_) => Ok(None),
        }).collect()
    };
    let mut hits = Vec::with_capacity(found.len());
    for ((h, f), text) in found.into_iter().zip(texts) {
        match f {
            Fetched::Value(mut f) => {
                let o = optional!(f.as_object_mut());
                o.insert("_score".into(), h["_score"].clone());
                match h.get("sort") {
                    Some(s) => o.insert("sort".into(), s.clone()),
                    None => o.remove("sort"),
                };
                hits.push(f);
            }
            // The same edits on the hit's text.
            Fetched::Raw(..) => hits.push(sources.push_hit_text(optional!(text?))),
        }
    }
    merged["hits"]["hits"] = Value::Array(hits);
    // `took`: the coordinating node's time for the whole request.
    if merged.get("took").is_some() {
        merged["took"] = json!(started.elapsed().as_millis() as u64);
    }
    let t_assembled = started.elapsed().as_micros();
    let mut raw = RestResp::json(200, &Value::Null);
    raw.replace_body_vec(sources.render(&merged));
    let mut response = crate::rest::util::postprocess(req, raw);
    response.extend_headers(warnings.into_iter().map(|value| ("Warning".into(), value)));
    if super::stats_phases::ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        // Coordinator phases of a two-phase search: building and sending the
        // query requests, their answers (parsed as they come), the merge,
        // the fetch round, the page's assembly and its render.
        let t_done = started.elapsed().as_micros();
        super::stats_phases::record("search_two_phase", vec![("setup", t_sent), ("query", t_queried - t_sent), ("merge", t_merged - t_queried),
            ("fetch", t_fetched - t_merged), ("assemble", t_assembled - t_fetched), ("render", t_done - t_assembled), ("total", t_done)]);
    }
    Ok(Some(response))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_hit_score_text_matches_raw_value() {
        for v in [json!(1.0), json!(0.5), json!(2.0e-5), json!(1.0e21), json!(3), json!(-7), json!(f32::MAX as f64), json!(0.1f32 as f64)] {
            let Value::Number(n) = &v else { unreachable!() };
            assert_eq!(n.to_string(), raw_text(&v).unwrap().get(), "{v}");
        }
    }

    #[test]
    fn raw_query_winners_merge_as_parsed_parts() {
        // Light query answers of three shard requests: equal and distinct
        // scores, two indices in one answer, a null score, an escaped name.
        let answers: Vec<Vec<(&str, &str, [u64; 3])>> = vec![
            vec![("b", "2.0", [0, 0, 1]), ("a", "1.0", [1, 0, 2]), ("b", "1.0", [0, 0, 3]), ("a", "1.0", [1, 1, 0]), ("b", "null", [0, 2, 9])],
            vec![("a", "2.0", [0, 0, 4]), ("a", "1.5", [0, 0, 5]), ("a", "1.0", [0, 0, 6])],
            vec![("x\\\"y", "1.0", [0, 0, 7]), ("b", "2.0", [0, 0, 8]), ("a", "1.0", [0, 3, 1])],
        ];
        let bodies: Vec<(axum::body::Bytes, String)> = answers.iter().map(|hits| {
            let texts: Vec<String> = hits.iter().map(|(i, s, at)| format!(r#"{{"_index":"{i}","_score":{s},"_rustic_doc":[{},{},{}]}}"#, at[0], at[1], at[2])).collect();
            let tail = r#"]},"_rustic_ctx":"c"}"#;
            let body = format!(r#"{{"took":1,"timed_out":false,"_shards":{{"total":1,"successful":1,"skipped":0,"failed":0}},"hits":{{"total":{{"value":{},"relation":"eq"}},"max_score":2.0,"hits":[{}{tail}"#, hits.len(), texts.join(","));
            let lens: Vec<u32> = texts.iter().map(|t| t.len() as u32).collect();
            (axum::body::Bytes::from(body), elasticsearch_http::encode_hit_spans(tail.len(), &lens))
        }).collect();
        let params = HashMap::new();
        for size in [1usize, 2, 4, 6, 9, 11, 20] {
            let body = json!({"size": size});
            let mut raw: Vec<(u32, Value, Option<RawQuery>, u32)> = Vec::new();
            let mut full: Vec<Value> = Vec::new();
            for (tag, (b, spans)) in bodies.iter().enumerate() {
                let p = Parsed::query_raw(tag as u32, b, spans).expect("spans fit");
                assert_eq!(p.raw.as_ref().map(|r| r.hits.len()), Some(answers[tag].len()));
                raw.push((tag as u32, p.value.unwrap(), p.raw, tag as u32));
                let mut v: Value = serde_json::from_slice(b).unwrap();
                for h in v.pointer_mut("/hits/hits").unwrap().as_array_mut().unwrap() {
                    h.as_object_mut().unwrap().insert(TAG.into(), json!(tag));
                }
                full.push(v);
            }
            select_raw_winners(&mut raw, size).unwrap();
            let raw: Vec<Value> = raw.into_iter().map(|p| p.1).collect();
            let (a, b) = (fanout::merge_search_with_limit(&raw, &body, &params, 100), fanout::merge_search_with_limit(&full, &body, &params, 100));
            assert_eq!(a["hits"], b["hits"], "size {size}");
        }
    }

    #[test]
    fn spanned_fetch_hits_match_scanned() {
        let hits = [
            r#"{"_index":"i","_id":"a\"b,\"_score\":","_score":1.5,"_source":{"x":{"_rustic_doc":[9,9,9]},"s":"q,\"_rustic_doc\":[1,1,1]"},"_rustic_doc":[0,1,2]}"#,
            r#"{"_index":"j","_id":"2","_score":null,"_source":{},"fields":{"f":[1]},"_rustic_doc":[3,0,17]}"#,
            r#"{"_index":"k\\","_id":"3","_score":2.0E-5,"_rustic_doc":[4,5,6]}"#,
        ];
        let head = r#"{"_rustic_x":1,"took":3,"timed_out":false,"hits":{"max_score":null,"hits":["#;
        let body = format!("{head}{}]}}}}", hits.join(","));
        let lens: Vec<u32> = hits.iter().map(|h| h.len() as u32).collect();
        let tail = 3; // `]}}`
        let spans = elasticsearch_http::encode_hit_spans(tail, &lens);
        let key = |h: &RawHit| (h.at, h.text.clone(), h.score.clone(), h.cut);
        let scanned: Vec<_> = raw_fetch_hits(body.as_bytes()).unwrap().iter().map(key).collect();
        let spanned: Vec<_> = spanned_fetch_hits(body.as_bytes(), &spans).unwrap().iter().map(key).collect();
        assert_eq!(spanned, scanned);
        assert_eq!(&body[spanned[1].2.clone()], "null");
        // Spans that do not fit the body: read whole.
        assert!(spanned_fetch_hits(body.as_bytes(), &elasticsearch_http::encode_hit_spans(tail + 1, &lens)).is_none());
        let mut short = lens.clone();
        short[0] -= 1;
        assert!(spanned_fetch_hits(body.as_bytes(), &elasticsearch_http::encode_hit_spans(tail, &short)).is_none());
        // A hit whose `_score` comes after its `_source`: read whole.
        let late = r#"{"_index":"i","_id":"1","_source":{"_score":3},"_score":1.0,"_rustic_doc":[0,0,1]}"#;
        let body = format!("{head}{late}]}}}}");
        assert!(spanned_fetch_hits(body.as_bytes(), &elasticsearch_http::encode_hit_spans(tail, &[late.len() as u32])).is_none());
    }

    fn sub() -> RestReq {
        RestReq::try_new("POST", "/i/_search", &[], br#"{"query":{"match_all":{}}}"#, 0, Dfs::None, None).unwrap()
    }

    #[test]
    fn malformed_or_unpinned_query_hits_are_failed_without_rerun() {
        let part=json!({"hits":{"hits":[{"_index":"i","_id":"a","_rustic_doc":[0,1,2]}]}});
        assert!(valid_query_part(&part,Some("pinned")));
        assert!(!valid_query_part(&part,None));
        assert!(valid_query_part(&json!({"hits":{"hits":[]}}),None));
        assert!(!valid_query_part(&json!({"hits":{"hits":[{"_index":"i","_id":"a"}]}}),Some("pinned")));
        let failures=copies::ShardFailure::from_response(&protocol_failure("unreadable query response"),&["i".into()],3,"node");
        let mut successful=json!({"_shards":{"total":1,"successful":1,"failed":0},"hits":{"hits":[{"_id":"kept"}]}});
        copies::add_failures(&mut successful,&failures);
        assert_eq!(successful["hits"]["hits"][0]["_id"],"kept");
        assert_eq!(successful["_shards"]["failed"],1);
    }

    #[test]
    fn incomplete_fetch_cannot_trigger_reexecution() {
        let winners=vec![json!({"_qtf_part":0,"_index":"i","_id":"a"}),json!({"_qtf_part":1,"_index":"j","_id":"b"})];
        assert!(valid_fetch_part(&json!({"hits":{"hits":[{"_index":"i","_id":"a"}]}}),&winners,0));
        assert!(!valid_fetch_part(&json!({"hits":{"hits":[]}}),&winners,0));
        assert!(!valid_fetch_part(&json!({"hits":{"hits":[{"_index":"i","_id":"a"},{"_index":"i","_id":"a"}]}}),&winners,0));
        assert!(!valid_fetch_part(&json!({"hits":{"hits":[{"_index":"i","_id":"extra"}]}}),&winners,0));
        assert!(!valid_fetch_part(&json!({"hits":{"hits":[{"_id":"a"}]}}),&winners,0));
    }

    /// The hashed check agrees with the pairwise definition: duplicates,
    /// misses, other parts' winners, winners or hits without string ids.
    #[test]
    fn fetch_part_validation_matches_pairwise_definition() {
        fn pairwise(part: &Value, winners: &[Value], ordinal: usize) -> bool {
            let Some(hits) = part.pointer("/hits/hits").and_then(Value::as_array) else { return false; };
            let wanted = || winners.iter().filter(|hit| member(hit, TAG).and_then(Value::as_u64) == Some(ordinal as u64));
            hits.len() == wanted().count()
                && hits.iter().all(|hit| hit.get("_index").and_then(Value::as_str).is_some() && hit.get("_id").and_then(Value::as_str).is_some())
                && wanted().all(|winner| hits.iter().filter(|hit| hit["_index"] == winner["_index"] && hit["_id"] == winner["_id"]).count() == 1)
        }
        let w = |p: u64, i: &str, id: &str| json!({TAG: p, "_index": i, "_id": id});
        let h = |i: &str, id: &str| json!({"_index": i, "_id": id});
        let winners = vec![w(0, "i", "a"), w(0, "j", "a"), w(1, "i", "b"), json!({TAG: 0, "_index": "i", "_id": 7}), w(0, "i", "c")];
        let cases = vec![
            json!({"hits": {"hits": [h("i", "a"), h("j", "a"), h("i", "c")]}}),
            json!({"hits": {"hits": [h("i", "a"), h("j", "a"), h("i", "c"), h("i", "7")]}}),
            json!({"hits": {"hits": [h("i", "a"), h("i", "a"), h("i", "c"), h("j", "a")]}}),
            json!({"hits": {"hits": [h("i", "a"), h("j", "a"), h("i", "c"), {"_index": "i", "_id": 7}]}}),
            json!({"hits": {"hits": [h("i", "a"), h("j", "a"), h("i", "c"), h("i", "b")]}}),
            json!({"hits": {}}),
        ];
        let simple = vec![w(0, "i", "a"), w(0, "j", "a"), w(1, "i", "b")];
        for part in &cases {
            for ordinal in 0..3 {
                assert_eq!(valid_fetch_part(part, &winners, ordinal), pairwise(part, &winners, ordinal), "{part} #{ordinal}");
                assert_eq!(valid_fetch_part(part, &simple, ordinal), pairwise(part, &simple, ordinal), "{part} #{ordinal} simple");
            }
        }
        assert!(valid_fetch_part(&json!({"hits": {"hits": [h("j", "a"), h("i", "a")]}}), &simple, 0));
        assert!(valid_fetch_part(&json!({"hits": {"hits": [h("i", "b")]}}), &simple, 1));
        assert!(valid_fetch_part(&json!({"hits": {"hits": []}}), &simple, 2));
    }

    /// geopoint `distanceFilter10kGTE`: a 10,000-hit page all from one
    /// shard. Quadratic validation took seconds; this must stay well under
    /// a second even in a debug build. Run with `--ignored` to time it.
    #[test]
    #[ignore]
    fn fetch_part_validation_of_a_10k_page_is_linear() {
        let n = 10_000;
        let winners: Vec<Value> = (0..n).map(|d| json!({TAG: 0, "_index": "osmgeopoints", "_id": format!("id{d}")})).collect();
        let part = json!({"hits": {"hits": (0..n).rev().map(|d| json!({"_index": "osmgeopoints", "_id": format!("id{d}")})).collect::<Vec<_>>()}});
        let started = Instant::now();
        assert!(valid_fetch_part(&part, &winners, 0));
        let took = started.elapsed();
        eprintln!("valid_fetch_part over {n} winners: {took:?}");
        assert!(took < Duration::from_secs(1), "{took:?}");
    }

    /// The shared bodies build byte-identical phase requests to parsing the
    /// base body for each one (query, free, fetch), and are parsed once per
    /// distinct body.
    #[test]
    fn shared_phase_bodies_match_per_request_parsing() {
        let bodies = [
            r#"{"query":{"terms":{"name.raw":["a","b","c"]}},"size":10,"sort":["_doc"],"aggs":{"x":{"terms":{"field":"f"}}},"track_total_hits":true}"#,
            r#"{"query":{"match":{"t":{"query":"x","_name":"n"}}},"size":3}"#,
            r#"{"query":{"match":{"t":"x"}},"highlight":{"fields":{"t":{}}},"from":0,"size":3,"_source":["a"]}"#,
            r#"{"size":2}"#,
        ];
        let docs = vec![json!([0, 1, 2, 1.5]), json!([0, 0, 7, 1.0])];
        let mut known = Vec::new();
        for body in bodies {
            let base = RestReq::try_new("POST", "/i/_search", &[("content-type".into(), "application/json".into()), (copies::READ_COPIES_HEADER.into(), "i".into())], body.as_bytes(), 0, Dfs::None, None).unwrap();
            let shared = Sub::new_shared(base.clone(), 0, vec!["i".into()], &mut known).unwrap().unwrap();
            let again = Sub::new_shared(base.clone(), 1, vec!["i".into()], &mut known).unwrap().unwrap();
            assert!(Arc::ptr_eq(&shared.bodies, &again.bodies), "one parse per distinct body");
            let old_query = phase_req(&base, "query", |b| { b.insert("_source".into(), json!(false)); }).unwrap().unwrap();
            assert_eq!(shared.query.body, old_query.body, "{body}");
            let header = |r: &RestReq, k: &str| r.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.clone());
            assert_eq!(header(&shared.query, HEADER).as_deref(), Some("query"));
            assert!(header(&shared.query, CTX_HEADER).is_some() && header(&shared.query, PINNED_HEADER).is_some());
            for d in [&docs[..], &[][..]] {
                let old = fetch_req(&base, "ctx-1", d).unwrap().unwrap();
                let new = shared.fetch_req("ctx-1", d).unwrap();
                assert_eq!(new.body, old.body, "{body} docs={}", d.len());
                assert_eq!(new.headers, old.headers);
                assert_eq!(new.uri, old.uri);
            }
            let free = fetch_req(&base, &ctx_id(&shared.free), &[]).unwrap().unwrap();
            assert_eq!(shared.free.body, free.body);
        }
        assert_eq!(known.len(), bodies.len());
    }

    /// Coordinator cost of a geonames `large_terms`-sized body (45,586
    /// terms) over 5 shards: building every phase request from shared
    /// bodies against parsing per request. Run with `--ignored` to time it.
    #[test]
    #[ignore]
    fn large_terms_phase_requests_parse_the_body_once() {
        let terms: Vec<String> = (0..45_586).map(|i| format!("Place name {i:06}")).collect();
        let body = serde_json::to_vec(&json!({"query": {"terms": {"name.raw": terms}}, "size": 10})).unwrap();
        let base = RestReq::try_new("POST", "/geonames/_search", &[("content-type".into(), "application/json".into())], &body, 0, Dfs::None, None).unwrap();
        let docs = vec![json!([0, 0, 1, 1.0])];
        let started = Instant::now();
        for _ in 0..5 {
            let q = phase_req(&base, "query", |b| { b.insert("_source".into(), json!(false)); }).unwrap().unwrap();
            let f = fetch_req(&base, "c", &[]).unwrap().unwrap();
            let o = fetch_req(&base, "c", &[]).unwrap().unwrap();
            let d = fetch_req(&base, "c", &docs).unwrap().unwrap();
            assert!(!q.body.is_empty() && !f.body.is_empty() && !o.body.is_empty() && !d.body.is_empty());
        }
        let per_request = started.elapsed();
        let started = Instant::now();
        let mut known = Vec::new();
        for k in 0..5 {
            let sub = Sub::new_shared(base.clone(), k, vec!["geonames".into()], &mut known).unwrap().unwrap();
            let o = sub.fetch_req("c", &[]).unwrap();
            let d = sub.fetch_req("c", &docs).unwrap();
            assert!(!o.body.is_empty() && !d.body.is_empty());
        }
        let shared = started.elapsed();
        eprintln!("5 shards, {} byte body: per-request parsing {per_request:?}, shared bodies {shared:?}", body.len());
        assert!(shared < per_request, "{shared:?} vs {per_request:?}");
    }

    #[test]
    fn phase_admission_error_is_not_optional_fallback() {
        let request = sub();
        let original = request.body.clone();
        let error = phase_req_with(&request, "query", |_| {}, |_, _| {
            Err(elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "injected serialization admission refusal"))
        }).unwrap_err();
        assert_eq!(error.status, 429);
        assert_eq!(request.body, original);
        assert!(!request.headers.iter().any(|(name, _)| name == HEADER));
        let malformed = request.try_with_body(b"not JSON").unwrap();
        assert!(phase_req_with(&malformed, "query", |_| {}, |_, _| panic!("must not encode malformed input")).unwrap().is_none());
    }

    #[test]
    fn ctx_of_reads_the_first_member_only() {
        assert_eq!(ctx_of(br#"{"_rustic_ctx":"a-1","hits":{"x":"\ud800"}}"#).as_deref(), Some("a-1"));
        assert_eq!(ctx_of(br#"{"hits":{},"_rustic_ctx":"a-1"}"#), None);
        assert_eq!(ctx_of(b"{}"), None);
    }

    /// A registration failing on the first answer of a multi-answer batch
    /// (an admission error) returns early: that answer's context is freed
    /// by the open set, the answers left in the batch by the batch.
    #[test]
    fn early_return_mid_batch_frees_every_context() {
        let freed = std::cell::RefCell::new(Vec::new());
        let record = |f: RestReq, node: String| {
            let b: Value = serde_json::from_slice(&f.body).unwrap();
            freed.borrow_mut().push((b[elasticsearch_http::QTF_FETCH_KEY]["ctx"].as_str().unwrap().to_string(), node));
        };
        let subs: Vec<Sub> = (0..3).map(|i| Sub::new(sub(), 0, vec![format!("i{i}")]).unwrap().unwrap()).collect();
        let ids: Vec<String> = subs.iter().map(|s| ctx_id(&s.free)).collect();
        let outs: Vec<QueryOut> = subs.into_iter().enumerate().map(|(i, sub)| QueryOut::Answer {
            sub, node: format!("n{i}"), resp: RestResp::json(200, &json!({"hits": {}})), parsed: None,
        }).collect();
        let run = || -> EsResult<()> {
            let mut open = OpenContexts::new(3, record);
            for out in (Batch { outs: outs.into_iter(), free: record }) {
                let QueryOut::Answer { sub, node, .. } = out else { unreachable!() };
                let ctx = ctx_id(&sub.free);
                // Registration keeps the prebuilt free, then fails.
                open.opened(0, &sub.base, &ctx, node, &sub.free)?;
                return Err(elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "injected"));
            }
            Ok(())
        };
        assert_eq!(run().unwrap_err().status, 429);
        let mut got = freed.borrow().clone();
        got.sort();
        let mut want: Vec<(String, String)> = ids.into_iter().enumerate().map(|(i, id)| (id, format!("n{i}"))).collect();
        want.sort();
        assert_eq!(got, want, "no context outlives the failed phase");
    }

    /// Single rounds: the geonames `default` / `term` / `phrase` bodies and
    /// other small plain pages, all shards local; not over the window, nor
    /// for what renders or fails differently in one round.
    #[test]
    fn single_round_takes_small_plain_pages_only() {
        let none = HashMap::new();
        let yes = [
            json!({"query": {"match_all": {}}}),
            json!({"query": {"term": {"country_code.raw": "AT"}}}),
            json!({"query": {"match_phrase": {"name": "Sankt Georgen"}}}),
            json!({"query": {"match": {"name": "x"}}, "from": 10, "size": 10}),
            json!({"sort": [{"population": "desc"}], "size": 5, "track_scores": true, "_source": ["name"], "version": true}),
            json!({"size": 3, "aggs": {"c": {"terms": {"field": "f"}}}, "track_total_hits": true}),
            json!({"query": {"match": {"name": {"query": "x", "_name": "n"}}}}),
        ];
        for body in &yes {
            assert!(single_round(body, &none, true), "{body}");
            assert!(!single_round(body, &none, false), "{body}: a remote copy keeps two phases");
        }
        let no = [
            json!({"query": {"match_all": {}}, "size": 21}),
            json!({"query": {"match_all": {}}, "from": 15, "size": 10}),
            json!({"query": {"match_all": {}}, "size": 0}),
            json!({"query": {"match": {"t": "x"}}, "highlight": {"fields": {"t": {}}}}),
            json!({"docvalue_fields": ["n"]}),
            json!({"fields": ["n"]}),
            json!({"script_fields": {"s": {"script": "1"}}}),
            json!({"runtime_mappings": {"r": {"type": "long"}}}),
            json!({"timeout": "1s"}),
            json!({"stats": ["g"]}),
            json!({"sort": ["n"], "query": {"match": {"name": {"query": "x", "_name": "n"}}}}),
        ];
        for body in &no {
            assert!(!single_round(body, &none, true), "{body}");
        }
        let small = json!({"query": {"match_all": {}}});
        assert!(!single_round(&small, &HashMap::from([("size".to_string(), "50".to_string())]), true), "URL size counts");
        assert!(!single_round(&small, &HashMap::from([("timeout".to_string(), "1s".to_string())]), true));
        assert!(single_round(&small, &HashMap::from([("size".to_string(), "20".to_string())]), true));
    }

    /// A single round's answers lose their deferred fetch time; the merged
    /// page loses its tags (and nothing else moves), naming each answer
    /// and index its hits came from once.
    #[test]
    fn single_round_tags_come_off_the_page_in_place() {
        let mut a = json!({"_rustic_fetch_ns": 1234, "took": 1, "hits": {"hits": [
            {"_index": "g", "_id": "1", "_score": 2.0, "_source": {"x": 1}},
            {"_index": "g", "_id": "2", "_score": 1.0, "_source": {"x": 2}}]}});
        let mut b = json!({"took": 1, "hits": {"hits": [{"_index": "h", "_id": "3", "_score": 1.5, "_source": {"x": 3}}]}});
        let ra = round_answer(&mut a, 0, 0, "n0");
        let rb = round_answer(&mut b, 1, 3, "n0");
        assert_eq!((ra.k, ra.fetch_nanos, rb.k, rb.fetch_nanos), (0, Some(1234), 3, None));
        assert_eq!(ra.node, "n0");
        assert!(a.get(elasticsearch_http::QTF_FETCH_NS_KEY).is_none());
        assert_eq!(a.as_object().unwrap().keys().next().map(String::as_str), Some("took"));
        assert_eq!(a["hits"]["hits"][1][ROUND_TAG], 0);
        assert_eq!(b["hits"]["hits"][0][ROUND_TAG], 1);
        let mut merged = json!({"took": 2, "hits": {"max_score": 2.0, "hits": [a["hits"]["hits"][0].clone(), b["hits"]["hits"][0].clone(), a["hits"]["hits"][1].clone()]}});
        let from = untag_round_page(&mut merged);
        assert_eq!(from.into_iter().collect::<Vec<_>>(), vec![(0, "g".to_string()), (1, "h".to_string())]);
        assert_eq!(merged.to_string(), json!({"took": 2, "hits": {"max_score": 2.0, "hits": [
            {"_index": "g", "_id": "1", "_score": 2.0, "_source": {"x": 1}},
            {"_index": "h", "_id": "3", "_score": 1.5, "_source": {"x": 3}},
            {"_index": "g", "_id": "2", "_score": 1.0, "_source": {"x": 2}}]}}).to_string());
    }

    fn ctx_id(free: &RestReq) -> String {
        let b: Value = serde_json::from_slice(&free.body).unwrap();
        b[elasticsearch_http::QTF_FETCH_KEY]["ctx"].as_str().unwrap().to_string()
    }

    /// Every context still open when the set goes away is freed once, on
    /// the copy holding it; fetched ones are not.
    #[test]
    fn open_contexts_free_what_is_left_on_drop() {
        let freed = std::cell::RefCell::new(Vec::new());
        {
            let mut open = OpenContexts::new(4, |f: RestReq, node: String| {
                let b: Value = serde_json::from_slice(&f.body).unwrap();
                assert!(f.headers.iter().any(|(k, v)| k == HEADER && v == "fetch"));
                assert_eq!(b[elasticsearch_http::QTF_FETCH_KEY]["docs"], json!([]));
                freed.borrow_mut().push((b[elasticsearch_http::QTF_FETCH_KEY]["ctx"].as_str().unwrap().to_string(), node));
            });
            for i in 0..3 {
                let fallback = fetch_req(&sub(), &format!("c{i}"), &[]).unwrap().unwrap();
                open.opened(i, &sub(), &format!("c{i}"), format!("n{i}"), &fallback).unwrap();
            }
            open.fetched(0);
            open.free_now(1);
            assert_eq!(freed.borrow().len(), 1);
            // An early return / dropped future: the rest go now.
        }
        assert_eq!(*freed.borrow(), vec![("c1".to_string(), "n1".to_string()), ("c2".to_string(), "n2".to_string())]);
    }
}
