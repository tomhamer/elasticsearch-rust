//! ES|QL reads of shards held by other nodes.
//!
//! `FROM` loads documents with one search over its shard targets
//! ([`super::exec`]'s `load`); every later command runs over the loaded
//! table. In a cluster, the node running the query may lack a started copy
//! of some shards (indices without replicas spread over several nodes).
//! A [`ShardSource`] splits the resolved targets into an ordered list of
//! [`Slot`]s, one per wanted shard: read locally, or fetched from a node
//! holding a started copy. Each slot runs the same per-shard search
//! ([`shard_result`]) wherever it lives, and [`merge`] combines the slot
//! results in slot order exactly like a single search over all the shards
//! would: hits ordered by score (then shard, then document) and cut to the
//! pushed-down size, or, for a pushed-down sort, every shard's top hits in
//! shard order (the `SORT | LIMIT` that follows in the pipeline picks the
//! global top rows with the same tie order).
//!
//! When every wanted shard is local, [`Split::slots`] is `None` and `load`
//! keeps its single multi-shard search.

use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use crate::search::Target;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// One wanted shard of a `FROM`, in the order a single node would search it.
#[derive(Clone, Debug, PartialEq)]
pub enum Slot {
    /// Read here: the index into [`Split::targets`].
    Local(usize),
    /// Read on another node.
    Remote(RemoteShard),
}

/// A shard without a readable local copy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteShard {
    pub index: String,
    pub shard: u32,
    /// The alias filter of the target the shard belongs to.
    #[serde(default)]
    pub alias_filter: Option<Value>,
}

/// Resolved targets after the shard source looked at them.
pub struct Split {
    /// Targets for the schema (mappings, field access); with `slots`,
    /// local reads refer into this list, and targets no slot refers to are
    /// mapping-only.
    pub targets: Vec<Target>,
    /// `None`: every target is read locally (the single-search path).
    pub slots: Option<Vec<Slot>>,
}

/// Bounded, constant-score grouped limit. The complete schema is checked on
/// each serving node so missing fields retain the same null semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupLimitPlan {
    pub indices: Vec<String>,
    pub columns: Vec<(String, String)>,
    pub keys: Vec<String>,
    pub per_group: usize,
    pub cap: usize,
}

/// What a remote node runs for each shard it is asked for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FetchSpec {
    pub shards: Vec<RemoteShard>,
    /// The pushed-down query (counted per shard).
    pub query: Value,
    /// The search body, without `size`.
    pub body: Value,
    /// The pushed-down size (`None`: every matching document).
    pub size: Option<usize>,
    /// Full-text probe queries (each counted with `query` as a filter).
    #[serde(default)]
    pub probes: Vec<Value>,
    /// Aggregate per shard instead of returning hits ([`super::partial`]).
    #[serde(default)]
    pub agg: Option<super::partial::AggPlan>,
    #[serde(default)]
    pub group_limit: Option<GroupLimitPlan>,
    /// Read through a point in time held on the serving node (pages of one
    /// fold see one view of each shard; plain hit reads only).
    #[serde(default)]
    pub pit: Option<FetchPit>,
    /// Run a columnar fast path per shard and return its partial state
    /// ([`FastPart`]) instead of hits. A node that does not know it returns
    /// hits without a part; the caller then takes the table path.
    #[serde(default)]
    pub fast: Option<FastPlan>,
}

/// A row command a shard runs before its rows are combined.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RowCmd {
    Where(super::parse::Expr, usize),
    Eval(Vec<super::parse::Field>),
    Keep(Vec<String>, usize),
    Drop(Vec<String>, usize),
    Rename(Vec<(String, String)>, usize),
}

impl RowCmd {
    pub fn of(c: &super::parse::Cmd) -> Option<RowCmd> {
        use super::parse::Cmd;
        Some(match c {
            Cmd::Where(e, p) => RowCmd::Where(e.clone(), *p),
            Cmd::Eval(f) => RowCmd::Eval(f.clone()),
            Cmd::Keep(n, p) => RowCmd::Keep(n.clone(), *p),
            Cmd::Drop(n, p) => RowCmd::Drop(n.clone(), *p),
            Cmd::Rename(n, p) => RowCmd::Rename(n.clone(), *p),
            _ => return None,
        })
    }

    pub fn cmd(&self) -> super::parse::Cmd {
        use super::parse::Cmd;
        match self {
            RowCmd::Where(e, p) => Cmd::Where(e.clone(), *p),
            RowCmd::Eval(f) => Cmd::Eval(f.clone()),
            RowCmd::Keep(n, p) => Cmd::Keep(n.clone(), *p),
            RowCmd::Drop(n, p) => Cmd::Drop(n.clone(), *p),
            RowCmd::Rename(n, p) => Cmd::Rename(n.clone(), *p),
        }
    }
}

/// One aggregate of a doc-values geo `STATS`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GeoAggPlan {
    Count,
    Centroid(String),
    Extent(String),
}

/// A columnar fast path run on each shard (the same per-shard work the
/// node running the query does for its local shards); the parts are
/// merged in shard (search) order like the single-node path merges them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FastPlan {
    /// `STATS` of `COUNT(*)` / `ST_CENTROID_AGG` / `ST_EXTENT_AGG` over geo doc values.
    GeoStats { aggs: Vec<GeoAggPlan> },
    /// `STATS COUNT(*) BY` a geohash / geotile / geohex cell of a geo doc value.
    GridStats { field: String, kind: u8, precision: i64, bounds: Option<(f64, f64, f64, f64)> },
    /// Each shard's rows within its own `n` nearest points to `(lat, lon)`.
    Distance { indices: Vec<String>, columns: Vec<(String, String)>, field: String, lat: f64, lon: f64, n: usize },
    /// Each shard's rows through `residual` and `pre`: the first `n`, or
    /// with `sort` the first `n` in that order.
    Rows {
        indices: Vec<String>,
        columns: Vec<(String, String)>,
        /// The query text (warning locations) and its `NOW()`.
        query_text: String,
        now: i64,
        residual: Vec<(super::parse::Expr, usize)>,
        pre: Vec<RowCmd>,
        sort: Option<Vec<super::parse::SortKey>>,
        n: usize,
    },
}

/// One shard's part of a [`FastPlan`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FastPart {
    Geo {
        docs: u64,
        /// Per aggregate: compensated x / y sums (value and delta bits) and point count.
        sums: Vec<((u64, u64), (u64, u64), u64)>,
        bounds: Vec<Option<(f64, f64, f64, f64)>>,
    },
    Grid {
        docs: u64,
        /// Cells in first-seen order (`None`: no point) and their counts.
        groups: Vec<(Option<u64>, i64)>,
        /// A point fell outside the cell function's bounds.
        outside: bool,
    },
    Rows {
        docs: u64,
        loaded: u64,
        /// Documents with a point ([`FastPlan::Distance`]).
        with_point: u64,
        /// Columns of the rows (`None`: shards gave different column types).
        cols: Option<Vec<(String, String)>>,
        rows: Vec<Vec<super::partial::WireVal>>,
    },
}

/// A point in time over a fetch's shards, retained on the serving node
/// between the fetches of one fold (ES: `search_after` pages are
/// consistent only against a PIT). The coordinating node picks the id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FetchPit {
    pub id: String,
    /// Kept this long after each fetch, then released by the node's
    /// context reaper (a coordinator that never closes it). Held in
    /// `Node::fold_contexts`, apart from user PITs.
    pub keep_alive_ms: u64,
    pub phase: PitPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PitPhase {
    /// Retain the shards' current views under the id, then read them.
    Open,
    /// Read the views retained under the id (extending its keep-alive).
    Read,
    /// Release them (no results).
    Close,
}

/// One shard's part of a `FROM`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardResult {
    pub index: String,
    pub shard: u32,
    /// Documents matching the query.
    pub count: u64,
    /// The shard's top hits (search hit JSON), in search order.
    pub hits: Vec<Value>,
    /// For each probe, the `(index, id)` of the documents it matches.
    #[serde(default)]
    pub probes: Vec<Vec<(String, String)>>,
    /// With an aggregation plan: the shard's partial groups.
    #[serde(default)]
    pub groups: Option<Vec<super::partial::PartialGroup>>,
    /// Non-null values read (profile).
    #[serde(default)]
    pub loaded: u64,
    /// Served through the request's point in time ([`FetchSpec::pit`]): a
    /// node that does not know points in time leaves it `false`.
    #[serde(default)]
    pub pit: bool,
    /// With a [`FetchSpec::fast`] plan: the shard's part.
    #[serde(default)]
    pub fast: Option<FastPart>,
    /// Evaluation warnings raised serving the fetch.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Splits targets into local and remote shards and fetches remote ones.
pub trait ShardSource: Send + Sync {
    fn split(&self, targets: Vec<Target>) -> Result<Split>;
    /// Run `spec` on the nodes holding its shards: one result per shard, in any order.
    /// Decode/copy allocations must be admitted through `memory` before they
    /// occur. The caller holds this guard until it consumes this fetched page
    /// or transfers its values into separately accounted output ownership.
    fn fetch(&self, spec: &FetchSpec, memory: &mut crate::memory::Reservation) -> Result<Vec<ShardResult>>;
}

pub type SharedShardSource = Arc<dyn ShardSource>;

/// Request extension: this `POST /_query` is a remote node's fetch of
/// shards for another node's query (set by the cluster layer only).
#[derive(Clone, Debug)]
pub struct FetchRequest(pub FetchSpec);

/// Request extension: the cluster's shard source for ES|QL.
#[derive(Clone)]
pub struct ShardSourceExt(pub SharedShardSource);

/// Error type a coordinating run stops with once its shard source decided to
/// forward the whole query to another node: the query runs (and is counted)
/// there, not here.
pub const FORWARD_ERROR_TYPE: &str = "rustic_esql_forward";

// Search is an internal row-fetch implementation detail of ES|QL. A
// breaker remains an ES|QL execution refusal, not a Search API phase envelope.
fn fetch_error(error: EsError) -> EsError {
    if error.kind == "search_phase_execution_exception" {
        let mut cause = error.caused_by.as_deref();
        while let Some(inner) = cause {
            if inner.kind == "circuit_breaking_exception" { return inner.clone(); }
            cause = inner.caused_by.as_deref();
        }
    }
    error
}

fn exec_opts(node: &Node) -> crate::search::ExecOpts<'static> {
    crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: Some(String::new()), pit_id: None }
}

/// Run a search body over `targets`; the hits, in search order.
pub(crate) fn search_hits(node: &Node, targets: &[Target], body: &Value) -> Result<Vec<Value>> {
    let req = crate::search::parse_request(body, &HashMap::new())?;
    let out = crate::search::execute(targets, &req, &exec_opts(node)).map_err(fetch_error)?;
    let mut v: Value = serde_json::from_str(&out.json).map_err(|e| EsError::internal(format!("bad search output: {e}")))?;
    Ok(match v["hits"]["hits"].take() {
        Value::Array(hits) => hits,
        _ => Vec::new(),
    })
}

/// `(index, id)` of the documents of `targets` matching probe `pq` within `query`.
pub(crate) fn probe_set(node: &Node, targets: &[Target], pq: &Value, query: &Value) -> Result<Vec<(String, String)>> {
    let q = json!({"bool": {"filter": [pq, query]}});
    let n = crate::search::count(targets, Some(&q), None)? as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let hits = search_hits(node, targets, &json!({"query": q, "size": n, "_source": false, "track_total_hits": false}))?;
    Ok(hits.iter().map(|h| (h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string())).collect())
}

/// One shard's part of a `FROM` (the same on the node running the query and
/// on a node serving a fetch).
pub fn shard_result(node: &Node, t: &Target, query: &Value, body: &Value, size: Option<usize>, probes: &[Value]) -> Result<ShardResult> {
    let targets = std::slice::from_ref(t);
    let count = crate::search::count(targets, Some(query), None)?;
    let n = size.map_or(count as usize, |s| s.min(count as usize));
    let mut hits = Vec::new();
    let mut sets = Vec::with_capacity(probes.len());
    if n > 0 {
        let mut b = body.clone();
        b["size"] = json!(n);
        hits = search_hits(node, targets, &b)?;
    }
    for pq in probes {
        sets.push(if count > 0 { probe_set(node, targets, pq, query)? } else { Vec::new() });
    }
    Ok(ShardResult { index: t.svc.name.clone(), shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0), count, hits, probes: sets, groups: None, loaded: 0, pit: false, fast: None, warnings: Vec::new() })
}

/// Slot results combined like one search over every shard.
#[derive(Debug, Default, PartialEq)]
pub struct Merged {
    pub count: u64,
    pub hits: Vec<Value>,
    pub probes: Vec<HashSet<(String, String)>>,
}

/// Combine per-shard results given in slot order. `sorted`: the search had
/// a field sort (each shard's hits are kept, in shard order); otherwise hits
/// are ordered by descending score, ties in shard order, and cut to the
/// size.
pub fn merge(results: Vec<ShardResult>, sorted: bool, size: Option<usize>, probes: usize) -> Merged {
    let count: u64 = results.iter().map(|r| r.count).sum();
    let total = size.map_or(count as usize, |s| s.min(count as usize));
    let mut merged = Merged { count, hits: Vec::new(), probes: vec![HashSet::new(); probes] };
    if total == 0 {
        return merged;
    }
    for r in results {
        merged.hits.extend(r.hits);
        for (set, ids) in merged.probes.iter_mut().zip(r.probes) {
            set.extend(ids);
        }
    }
    if !sorted {
        let score = |h: &Value| h["_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        // Stable: equal scores keep shard order, then each shard's order.
        merged.hits.sort_by(|a, b| score(b).total_cmp(&score(a)));
        merged.hits.truncate(total);
    }
    merged
}

/// The shards of a fetch, grouped by the node chosen to read each one:
/// the first node `holders(index, shard)` lists (nodes holding a started
/// copy, in preference order). Each shard is read exactly once; shards
/// nobody holds are returned in `missing`.
pub fn assign<F>(shards: &[RemoteShard], mut holders: F) -> (Vec<(String, Vec<RemoteShard>)>, Vec<RemoteShard>)
where
    F: FnMut(&str, u32) -> Vec<String>,
{
    let mut by_node: Vec<(String, Vec<RemoteShard>)> = Vec::new();
    let mut missing = Vec::new();
    for s in shards {
        let Some(node) = holders(&s.index, s.shard).into_iter().next() else {
            missing.push(s.clone());
            continue;
        };
        match by_node.iter_mut().find(|(n, _)| *n == node) {
            Some((_, list)) => list.push(s.clone()),
            None => by_node.push((node, vec![s.clone()])),
        }
    }
    (by_node, missing)
}

/// Serve a fetch: read the requested shards from this node's started copies.
pub fn serve(node: &Node, spec: &FetchSpec, source: Option<&SharedShardSource>, restrict: Option<&super::exec::Restrict>) -> Result<Vec<ShardResult>> {
    if let Some(pit) = &spec.pit {
        if spec.agg.is_some() || spec.group_limit.is_some() || spec.fast.is_some() {
            return Err(EsError::internal("a point-in-time fetch reads hits only"));
        }
        match pit.phase {
            PitPhase::Open => {}
            PitPhase::Read => {
                let keep_alive = std::time::Duration::from_millis(pit.keep_alive_ms);
                let targets = {
                    let mut pits = node.fold_contexts.lock();
                    let ctx = pits.get_mut(&pit.id).ok_or_else(|| EsError::new(404, "search_context_missing_exception", format!("No search context found for id [{}]", pit.id)))?;
                    ctx.keep_alive = keep_alive;
                    ctx.expires = std::time::Instant::now() + keep_alive;
                    ctx.targets.clone()
                };
                return pit_results(node, spec, &targets);
            }
            PitPhase::Close => {
                node.fold_contexts.lock().remove(&pit.id);
                return Ok(Vec::new());
            }
        }
    }
    let mut names: Vec<String> = spec.shards.iter().map(|s| s.index.clone()).collect();
    names.sort();
    names.dedup();
    let mut opts = ResolveOpts::default();
    opts.expand_hidden = true;
    let targets = node.resolve(&names.join(","), opts)?;
    let targets = match source {
        Some(s) => {
            let split = s.split(targets)?;
            match split.slots {
                None => split.targets,
                Some(slots) => slots.iter().filter_map(|slot| match slot { Slot::Local(i) => Some(split.targets[*i].clone()), Slot::Remote(_) => None }).collect(),
            }
        }
        None => crate::search::expand_targets(&targets)?,
    };
    let mut picked = Vec::with_capacity(spec.shards.len());
    for s in &spec.shards {
        let Some(t) = targets.iter().find(|t| t.svc.name == s.index && t.shard.as_ref().is_some_and(|c| c.id == s.shard)) else {
            return Err(EsError::new(503, "no_shard_available_action_exception", format!("[{}][{}] no started copy on this node", s.index, s.shard)));
        };
        let mut t = t.clone();
        t.alias_filter = s.alias_filter.clone();
        t.routing = None;
        picked.push(t);
    }
    let picked = match restrict {
        Some(r) => r(picked)?,
        None => picked,
    };
    if let Some(pit) = &spec.pit {
        // `PitPhase::Open`: every requested copy's current view, retained
        // until closed or expired.
        let keep_alive = std::time::Duration::from_millis(pit.keep_alive_ms);
        // The cap is checked and the context inserted under one lock (the
        // snapshot is taken before: it takes the copies' locks, and the
        // contexts are a leaf lock), so concurrent opens never pass it.
        let snapshot = node.snapshot(&picked)?;
        let accounting = Some(crate::node::context_charge(&snapshot, 0));
        {
            let mut pits = node.fold_contexts.lock();
            if !pits.contains_key(&pit.id) && pits.len() >= MAX_FOLD_CONTEXTS {
                return Err(EsError::new(429, "es_rejected_execution_exception", format!("too many open fold contexts [{MAX_FOLD_CONTEXTS}] on this node")));
            }
            pits.insert(pit.id.clone(), crate::node::PitCtx { targets: snapshot.clone(), keep_alive, expires: std::time::Instant::now() + keep_alive, accounting, lost: Vec::new() });
        }
        return pit_results(node, spec, &snapshot);
    }
    if let Some(plan) = &spec.fast {
        return super::exec::serve_fast(node, spec, plan, &picked, source, restrict);
    }
    if let Some(plan) = &spec.group_limit {
        return super::exec::serve_group_limit(node, spec, plan, &picked, source, restrict);
    }
    if let Some(plan) = &spec.agg {
        return super::exec::serve_partial(node, spec, plan, &picked, source, restrict);
    }
    picked.iter().map(|t| shard_result(node, t, &spec.query, &spec.body, spec.size, &spec.probes)).collect()
}

/// [`shard_result`] of each requested shard over a point in time's targets.
fn pit_results(node: &Node, spec: &FetchSpec, targets: &[Target]) -> Result<Vec<ShardResult>> {
    spec.shards.iter().map(|s| {
        let t = targets.iter().find(|t| t.svc.name == s.index && t.shard.as_ref().is_some_and(|c| c.id == s.shard))
            .ok_or_else(|| EsError::new(404, "search_context_missing_exception", format!("[{}][{}] is not in the point in time", s.index, s.shard)))?;
        pit_shard_result(node, t, spec)
    }).collect()
}

/// One page of a point-in-time read: the hits of `spec.body` (its
/// `search_after` page) on `t`, `spec.size` at most. A point-in-time fetch
/// reads hits only, so the shard does not count the whole query again for
/// every page (`count` is the page's).
fn pit_shard_result(node: &Node, t: &Target, spec: &FetchSpec) -> Result<ShardResult> {
    if !spec.probes.is_empty() || spec.size.is_none() {
        return shard_result(node, t, &spec.query, &spec.body, spec.size, &spec.probes).map(|r| ShardResult { pit: true, ..r });
    }
    let mut b = spec.body.clone();
    b["size"] = json!(spec.size);
    b["track_total_hits"] = json!(false);
    let hits = if spec.size == Some(0) { Vec::new() } else { search_hits(node, std::slice::from_ref(t), &b)? };
    Ok(ShardResult { index: t.svc.name.clone(), shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0), count: hits.len() as u64, hits, probes: Vec::new(), groups: None, loaded: 0, pit: true, fast: None, warnings: Vec::new() })
}

/// Most fold points in time one node holds at once.
pub const MAX_FOLD_CONTEXTS: usize = 500;

/// Order fetched results by slot: local results come from `local(i)`,
/// remote ones from `fetched` (keyed by index and shard).
pub fn in_slot_order(slots: &[Slot], mut local: impl FnMut(usize) -> Result<ShardResult>, fetched: Vec<ShardResult>) -> Result<Vec<ShardResult>> {
    let mut by_key: HashMap<(String, u32), ShardResult> = fetched.into_iter().map(|r| ((r.index.clone(), r.shard), r)).collect();
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        out.push(match slot {
            Slot::Local(i) => local(*i)?,
            Slot::Remote(s) => by_key
                .remove(&(s.index.clone(), s.shard))
                .ok_or_else(|| EsError::new(503, "no_shard_available_action_exception", format!("[{}][{}] was not read", s.index, s.shard)))?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs(index: &str, shard: u32) -> RemoteShard {
        RemoteShard { index: index.into(), shard, alias_filter: None }
    }

    fn res(index: &str, shard: u32, count: u64, hits: Vec<(&str, f64)>) -> ShardResult {
        ShardResult { index: index.into(), shard, count, hits: hits.into_iter().map(|(id, s)| json!({"_index": index, "_id": id, "_score": s})).collect(), probes: vec![], groups: None, loaded: 0, pit: false, fast: None, warnings: vec![] }
    }

    fn ids(m: &Merged) -> Vec<&str> {
        m.hits.iter().map(|h| h["_id"].as_str().unwrap()).collect()
    }

    #[test]
    fn unsorted_merge_orders_by_score_then_shard_and_cuts_to_size() {
        let a = res("i", 0, 3, vec![("a1", 1.0), ("a2", 1.0), ("a3", 0.5)]);
        let b = res("i", 1, 2, vec![("b1", 2.0), ("b2", 1.0)]);
        let m = merge(vec![a.clone(), b.clone()], false, Some(4), 0);
        assert_eq!(m.count, 5);
        assert_eq!(ids(&m), ["b1", "a1", "a2", "b2"]);
        // No size: every hit.
        assert_eq!(ids(&merge(vec![a, b], false, None, 0)), ["b1", "a1", "a2", "b2", "a3"]);
    }

    #[test]
    fn constant_scores_keep_shard_order() {
        let a = res("x", 0, 2, vec![("x1", 1.0), ("x2", 1.0)]);
        let b = res("y", 0, 2, vec![("y1", 1.0), ("y2", 1.0)]);
        assert_eq!(ids(&merge(vec![a, b], false, Some(3), 0)), ["x1", "x2", "y1"]);
    }

    #[test]
    fn sorted_merge_keeps_every_shard_top_in_shard_order() {
        let a = res("i", 0, 9, vec![("a1", 0.0), ("a2", 0.0)]);
        let b = res("i", 1, 9, vec![("b1", 0.0), ("b2", 0.0)]);
        let m = merge(vec![a, b], true, Some(2), 0);
        assert_eq!(ids(&m), ["a1", "a2", "b1", "b2"]);
        assert_eq!(m.count, 18);
    }

    #[test]
    fn nothing_matching_loads_nothing() {
        let mut a = res("i", 0, 0, vec![]);
        a.probes = vec![vec![]];
        let m = merge(vec![a], false, Some(10), 1);
        assert!(m.hits.is_empty());
        assert_eq!(m.probes, vec![HashSet::new()]);
    }

    #[test]
    fn probe_sets_union_over_shards() {
        let mut a = res("i", 0, 1, vec![("a", 1.0)]);
        a.probes = vec![vec![("i".into(), "a".into())], vec![]];
        let mut b = res("i", 1, 1, vec![("b", 1.0)]);
        b.probes = vec![vec![], vec![("i".into(), "b".into())]];
        let m = merge(vec![a, b], false, None, 2);
        assert!(m.probes[0].contains(&("i".to_string(), "a".to_string())));
        assert!(m.probes[1].contains(&("i".to_string(), "b".to_string())));
        assert_eq!(m.probes[0].len() + m.probes[1].len(), 2);
    }

    #[test]
    fn assign_reads_each_shard_once_on_its_first_holder() {
        let shards = vec![rs("i", 0), rs("i", 1), rs("j", 0), rs("k", 0)];
        let holders = |index: &str, shard: u32| -> Vec<String> {
            match (index, shard) {
                ("i", 0) => vec!["n2".into()],
                ("i", 1) => vec!["n3".into(), "n2".into()],
                ("j", 0) => vec!["n2".into(), "n1".into()],
                _ => vec![],
            }
        };
        let (by_node, missing) = assign(&shards, holders);
        assert_eq!(by_node, vec![("n2".to_string(), vec![rs("i", 0), rs("j", 0)]), ("n3".to_string(), vec![rs("i", 1)])]);
        assert_eq!(missing, vec![rs("k", 0)]);
        let read: usize = by_node.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(read + missing.len(), shards.len());
    }

    #[test]
    fn slot_order_interleaves_local_and_remote() {
        let slots = vec![Slot::Remote(rs("i", 0)), Slot::Local(0), Slot::Remote(rs("i", 2))];
        let fetched = vec![res("i", 2, 1, vec![("c", 1.0)]), res("i", 0, 1, vec![("a", 1.0)])];
        let out = in_slot_order(&slots, |_| Ok(res("i", 1, 1, vec![("b", 1.0)])), fetched).unwrap();
        assert_eq!(out.iter().map(|r| r.shard).collect::<Vec<_>>(), [0, 1, 2]);
        // A remote shard the fetch did not return is an error, not a partial read.
        let err = in_slot_order(&slots, |_| Ok(res("i", 1, 0, vec![])), vec![]).unwrap_err();
        assert_eq!(err.status, 503);
    }

    #[test]
    fn fetch_spec_round_trips() {
        let spec = FetchSpec { shards: vec![RemoteShard { index: "i".into(), shard: 3, alias_filter: Some(json!({"term": {"a": "é"}})) }], query: json!({"match_all": {}}), body: json!({"_source": true}), size: Some(5), probes: vec![], agg: None, group_limit: None, pit: None, fast: None };
        let back: FetchSpec = serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(back, spec);
    }
}

#[cfg(test)]
mod fetch_error_tests {
    use super::*;
    #[test]
    fn internal_search_breaker_keeps_esql_error_contract() {
        let breaker = EsError::new(429, "circuit_breaking_exception", "request limit")
            .with("bytes_wanted", 4096).with("bytes_limit", 1024).with("durability", "TRANSIENT");
        assert_eq!(fetch_error(EsError::search_phase("i", breaker.clone())), breaker);
        let other = EsError::search_phase("i", EsError::query_shard("i", "uuid", "invalid query"));
        assert_eq!(fetch_error(other.clone()), other);
    }
}
