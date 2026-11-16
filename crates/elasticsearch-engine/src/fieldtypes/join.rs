//! Parent/child (`join` field): `has_child`, `has_parent`, `parent_id`
//! queries and the `children` / `parent` aggregations.
//!
//! Each doc stores its relation name in the join field and, for child
//! docs, the parent's `_id` in `<join>#parent` (ES `<join>#<parent>`
//! global ordinals). Join queries run their inner query over the searched
//! leaves first, then match the other side by `_id` / parent id.

use super::aggs::Docs;
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, ResolvedField, ID_FIELD};
use crate::query::dsl::{parse_query, QueryContext};
use rustic_core::fieldtypes::filter::FilterQuery;
use rustic_core::query::{Bm25Params, LeafReader, MinShouldMatch, Query, SearchContext, TERMINATED};
use rustic_core::types::{DocId, Term};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// The `relations` of a join field.
#[derive(Clone, Debug, Default)]
pub struct Relations {
    /// `(parent, children)`.
    pub pairs: Vec<(String, Vec<String>)>,
}

impl Relations {
    pub fn of(f: &ResolvedField) -> Relations {
        let mut pairs = Vec::new();
        if let Some(rel) = f.ext.as_ref().and_then(|e| e.params.get("relations")).and_then(Value::as_object) {
            for (p, c) in rel {
                let ch = match c {
                    Value::String(s) => vec![s.clone()],
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                    _ => vec![],
                };
                pairs.push((p.clone(), ch));
            }
        }
        Relations { pairs }
    }
    pub fn knows(&self, name: &str) -> bool {
        self.pairs.iter().any(|(p, c)| p == name || c.iter().any(|x| x == name))
    }
    pub fn parent_of(&self, child: &str) -> Option<&str> {
        self.pairs.iter().find(|(_, c)| c.iter().any(|x| x == child)).map(|(p, _)| p.as_str())
    }
    pub fn children_of(&self, parent: &str) -> Vec<String> {
        self.pairs.iter().filter(|(p, _)| p == parent).flat_map(|(_, c)| c.clone()).collect()
    }
}

/// The index's join field.
pub fn join_field(lk: &FieldLookup) -> Option<Arc<ResolvedField>> {
    lk.fields.values().find(|f| f.declared == EsType::Join && !f.is_multi_field).cloned()
}

fn shard_err(ctx: &QueryContext, reason: impl Into<String>) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, reason)
}

/// Run `q` over the context's leaves: `(leaf, doc, score)` of live matches.
pub fn run(q: &Query, leaves: &[LeafReader<'_>]) -> Result<Vec<(usize, DocId, f32)>> {
    let sctx = SearchContext::for_query(q, leaves, Bm25Params::default());
    sctx.execution_result()?;
    let mut out = Vec::new();
    for (li, leaf) in leaves.iter().enumerate() {
        let Some(mut s) = q.scorer(leaf, &sctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == TERMINATED {
                break;
            }
            if live(d) {
                out.push((li, d, s.score()));
            }
        }
    }
    sctx.execution_result()?;
    Ok(out)
}

/// Matches polled between deadline / cancellation checks.
const POLL_EVERY: u64 = 1024;

/// Whether a join's inner traversal must stop: a cancelled task fails as
/// cancelled; a search past its `timeout` stops (`Ok(true)`) and the join
/// is built from the inner matches read so far, as ES's collection stops at
/// the deadline: the search itself, already past its deadline, then stops
/// at its first check and answers `timed_out: true` with partial results.
fn check_stopped(sctx: &SearchContext) -> Result<bool> {
    crate::search::check_cancelled()?;
    Ok(sctx.stopped())
}

/// [`run`] for a join query being built: `f` sees each live match of `q`
/// over the context's leaves. The search's deadline and its task's
/// cancellation are polled every [`POLL_EVERY`] matches.
/// `Ok(true)`: the search's deadline stopped it (partial matches).
fn each_match(q: &Query, ctx: &QueryContext, mut f: impl FnMut(usize, DocId, f32) -> Result<()>) -> Result<bool> {
    let sctx = SearchContext::for_query_deadline(q, ctx.leaves, Bm25Params::default(), ctx.deadline);
    sctx.execution_result()?;
    let mut seen = 0u64;
    for (li, leaf) in ctx.leaves.iter().enumerate() {
        let Some(mut s) = q.scorer(leaf, &sctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == TERMINATED {
                break;
            }
            seen += 1;
            if seen % POLL_EVERY == 0 && check_stopped(&sctx)? {
                return Ok(true);
            }
            if live(d) {
                f(li, d, s.score())?;
            }
        }
    }
    // A cancellation that stopped the scorer early still fails the search.
    let stopped = check_stopped(&sctx)?;
    sctx.execution_result()?;
    Ok(stopped)
}

/// A join built from the partial matches of a search past its deadline:
/// the search that runs it reports `timed_out: true` (its context's
/// timeout flag), however few documents it then collects.
#[derive(Debug)]
struct PartialJoin(Query);

impl rustic_core::query::CustomQuery for PartialJoin {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn rustic_core::query::Scorer>> {
        ctx.vector_timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
        self.0.scorer(leaf, ctx)
    }
    fn children(&self) -> Vec<&Query> {
        vec![&self.0]
    }
}

fn mark_partial(q: Query, partial: bool) -> Query {
    if partial { Query::Custom(Arc::new(PartialJoin(q))) } else { q }
}

/// Request-breaker bytes of one retained join map entry keyed by `id`.
fn entry_bytes(id: &str) -> u64 {
    // The key's heap bytes, the map slot (key, value, control) and the
    // copied candidate `_id` term.
    2 * id.len() as u64 + 96
}

/// ES rejects every joining query when expensive queries are disallowed.
fn check_expensive(ctx: &QueryContext) -> Result<()> {
    if !ctx.allow_expensive_queries {
        return Err(shard_err(ctx, "[joining] queries cannot be executed when 'search.allow_expensive_queries' is set to false."));
    }
    Ok(())
}

/// Child scores of one parent, combined as they arrive (no per-child list).
#[derive(Clone, Copy)]
struct ChildScores {
    n: usize,
    sum: f32,
    min: f32,
    max: f32,
}

fn term(field: u16, s: &str) -> Query {
    Query::Term(Term::new_text(field, s))
}

fn filter2(a: Query, b: Query) -> Query {
    Query::Bool { must: vec![], should: vec![], must_not: vec![], filter: vec![a, b], minimum_should_match: MinShouldMatch::Count(0) }
}

fn boost_of(o: &Map<String, Value>) -> f32 {
    o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32
}

fn with_boost(q: Query, boost: f32) -> Query {
    if (boost - 1.0).abs() > f32::EPSILON {
        Query::Boost { query: Box::new(q), boost }
    } else {
        q
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ScoreMode {
    None,
    Avg,
    Sum,
    Max,
    Min,
}

pub fn has_child(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    for k in o.keys() {
        if !matches!(k.as_str(), "type" | "query" | "score_mode" | "min_children" | "max_children" | "ignore_unmapped" | "inner_hits" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[has_child] query does not support [{k}]")));
        }
    }
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[has_child] requires 'type' field"))?;
    let inner = o.get("query").ok_or_else(|| EsError::illegal_argument("[has_child] requires 'query' field"))?;
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    let mode = match o.get("score_mode").and_then(Value::as_str).unwrap_or("none") {
        "none" => ScoreMode::None,
        "avg" => ScoreMode::Avg,
        "sum" | "total" => ScoreMode::Sum,
        "max" => ScoreMode::Max,
        "min" => ScoreMode::Min,
        other => return Err(EsError::illegal_argument(format!("No score mode for child query [{other}] found"))),
    };
    let min_children = o.get("min_children").and_then(Value::as_u64).unwrap_or(1) as usize;
    let max_children = o.get("max_children").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(usize::MAX);
    if min_children > max_children {
        return Err(EsError::illegal_argument("[has_child] 'max_children' is less than 'min_children'"));
    }
    check_expensive(ctx)?;
    let Some(jf) = join_field(ctx.lookup) else {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, format!("[has_child] join field has not been configured")));
    };
    let rel = Relations::of(&jf);
    let Some(parent_ty) = rel.parent_of(ty).map(str::to_string) else {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, format!("[has_child] join field [{}] doesn't hold [{ty}] as a child", jf.name)));
    };
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(Query::MatchNone) };
    let inner_q = parse_query(inner, ctx)?;
    // Document-level security: children the caller cannot see do not count.
    let mut filter = vec![term(jid, ty)];
    filter.extend(ctx.dls_query()?);
    let child_q = Query::Bool { must: vec![inner_q], should: vec![], must_not: vec![], filter, minimum_should_match: MinShouldMatch::Count(0) };
    // Every parent with a matching child, charged to the request breaker
    // as the map grows (child scores are combined, not kept).
    let mem = crate::memory::RequestMemory::new();
    let mut per_parent: HashMap<String, ChildScores> = HashMap::new();
    let partial = each_match(&child_q, ctx, |li, d, s| {
        for p in ctx.leaves[li].keyword_values(pid, d) {
            match per_parent.get_mut(&p) {
                Some(c) => {
                    c.n += 1;
                    c.sum += s;
                    c.min = c.min.min(s);
                    c.max = c.max.max(s);
                }
                None => {
                    mem.add(entry_bytes(&p), "[has_child] matching parents")?;
                    per_parent.insert(p, ChildScores { n: 1, sum: s, min: f32::MAX.min(s), max: f32::MIN.max(s) });
                }
            }
        }
        Ok(())
    })?;
    let scores: HashMap<String, f32> = per_parent
        .into_iter()
        .filter(|(_, c)| c.n >= min_children && c.n <= max_children)
        .map(|(k, c)| {
            let s = match mode {
                ScoreMode::None => 1.0,
                ScoreMode::Avg => c.sum / c.n as f32,
                ScoreMode::Sum => c.sum,
                ScoreMode::Max => c.max,
                ScoreMode::Min => c.min,
            };
            (k, s)
        })
        .collect();
    if scores.is_empty() {
        return Ok(mark_partial(Query::MatchNone, partial));
    }
    let ids: Vec<Vec<u8>> = scores.keys().map(|k| k.clone().into_bytes()).collect();
    let cand = filter2(Query::TermSet { field: ID_FIELD, terms: ids, boost: 1.0 }, term(jid, &parent_ty));
    let scores = Arc::new(scores);
    let q = FilterQuery::new("has_child", Some(cand), Arc::new(move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_value(ID_FIELD, d).and_then(|id| scores.get(&id).copied())));
    Ok(mark_partial(with_boost(q.into_query(), boost_of(o)), partial))
}

pub fn has_parent(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    for k in o.keys() {
        if !matches!(k.as_str(), "parent_type" | "query" | "score" | "ignore_unmapped" | "inner_hits" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[has_parent] query does not support [{k}]")));
        }
    }
    let pty = o.get("parent_type").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[has_parent] requires 'parent_type' field"))?;
    let inner = o.get("query").ok_or_else(|| EsError::parsing("[has_parent] requires 'query' field"))?;
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    let score = o.get("score").and_then(Value::as_bool).unwrap_or(false);
    check_expensive(ctx)?;
    let Some(jf) = join_field(ctx.lookup) else {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, "[has_parent] join field has not been configured"));
    };
    let rel = Relations::of(&jf);
    let children = rel.children_of(pty);
    if children.is_empty() {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, format!("[has_parent] join field [{}] doesn't hold [{pty}] as a parent", jf.name)));
    }
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(Query::MatchNone) };
    // Document-level security: a parent the caller cannot see does not match.
    let mut filter = vec![term(jid, pty)];
    filter.extend(ctx.dls_query()?);
    let parent_q = Query::Bool { must: vec![parse_query(inner, ctx)?], should: vec![], must_not: vec![], filter, minimum_should_match: MinShouldMatch::Count(0) };
    let mem = crate::memory::RequestMemory::new();
    let mut scores: HashMap<String, f32> = HashMap::new();
    let partial = each_match(&parent_q, ctx, |li, d, s| {
        if let Some(id) = ctx.leaves[li].keyword_value(ID_FIELD, d) {
            if !scores.contains_key(&id) {
                mem.add(entry_bytes(&id), "[has_parent] matching parents")?;
            }
            scores.insert(id, if score { s } else { 1.0 });
        }
        Ok(())
    })?;
    if scores.is_empty() {
        return Ok(mark_partial(Query::MatchNone, partial));
    }
    let ids: Vec<Vec<u8>> = scores.keys().map(|k| k.clone().into_bytes()).collect();
    let types = Query::TermSet { field: jid, terms: children.iter().map(|c| c.clone().into_bytes()).collect(), boost: 1.0 };
    let cand = filter2(Query::TermSet { field: pid, terms: ids, boost: 1.0 }, types);
    let scores = Arc::new(scores);
    let q = FilterQuery::new("has_parent", Some(cand), Arc::new(move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_values(pid, d).iter().find_map(|p| scores.get(p).copied())));
    Ok(mark_partial(with_boost(q.into_query(), boost_of(o)), partial))
}

/// The other side of a `has_child` / `has_parent` query that asks for
/// `inner_hits` (ES `ParentChildInnerHitContextBuilder`).
pub struct InnerSide {
    /// The inner query, restricted to docs of relation `ty`.
    pub query: Query,
    /// The inner hits' relation (the child `type` / the `parent_type`), also
    /// the default `inner_hits` name.
    pub ty: String,
    /// `has_child`: the inner hits are the hit's children; `has_parent`: its parent.
    pub children: bool,
    /// The relation a hit must have for any inner hits (the parent type of
    /// `has_child`; the child types of `has_parent`).
    pub hit_types: Vec<String>,
    /// The join field.
    pub join_field: u16,
    /// `<join>#parent` (a child's parent `_id`).
    pub parent_field: u16,
}

/// `kind` is `has_child` or `has_parent`; `None` when the join field or the
/// relation does not exist (the query itself is `ignore_unmapped` then).
pub fn inner_hits_side(kind: &str, o: &Map<String, Value>, ctx: &QueryContext) -> Result<Option<InnerSide>> {
    let Some(jf) = join_field(ctx.lookup) else { return Ok(None) };
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(None) };
    let rel = Relations::of(&jf);
    let children = kind == "has_child";
    let (ty, hit_types) = if children {
        let Some(ty) = o.get("type").and_then(Value::as_str) else { return Ok(None) };
        let Some(p) = rel.parent_of(ty) else { return Ok(None) };
        (ty.to_string(), vec![p.to_string()])
    } else {
        let Some(ty) = o.get("parent_type").and_then(Value::as_str) else { return Ok(None) };
        let ch = rel.children_of(ty);
        if ch.is_empty() {
            return Ok(None);
        }
        (ty.to_string(), ch)
    };
    let Some(inner) = o.get("query") else { return Ok(None) };
    // Document-level security: inner hits never include a document the
    // caller cannot see (whether or not the request has aggregations).
    let mut filter = vec![term(jid, &ty)];
    filter.extend(ctx.dls_query()?);
    let query = Query::Bool { must: vec![parse_query(inner, ctx)?], should: vec![], must_not: vec![], filter, minimum_should_match: MinShouldMatch::Count(0) };
    Ok(Some(InnerSide { query, ty, children, hit_types, join_field: jid, parent_field: pid }))
}

/// The inner hits of hit `doc` (on `leaf`): `(leaf, doc, score)` of the
/// docs of `side` joined to it, over all `leaves` of the shard, also
/// matching `visibility` (alias filter / DLS) when given.
pub fn inner_hits_of(side: &InnerSide, leaves: &[LeafReader<'_>], leaf: &LeafReader<'_>, doc: DocId, visibility: Option<&Query>) -> Result<Vec<(usize, DocId, f32)>> {
    if !leaf.keyword_value(side.join_field, doc).is_some_and(|n| side.hit_types.contains(&n)) {
        return Ok(Vec::new());
    }
    let join = if side.children {
        let Some(id) = leaf.keyword_value(ID_FIELD, doc) else { return Ok(Vec::new()) };
        term(side.parent_field, &id)
    } else {
        let Some(p) = leaf.keyword_values(side.parent_field, doc).into_iter().next() else { return Ok(Vec::new()) };
        term(ID_FIELD, &p)
    };
    let mut filter = vec![join];
    filter.extend(visibility.cloned());
    let q = Query::Bool { must: vec![side.query.clone()], should: vec![], must_not: vec![], filter, minimum_should_match: MinShouldMatch::Count(0) };
    run(&q, leaves)
}

pub fn parent_id(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[parent_id] requires 'type' field"))?;
    let id = o.get("id").map(|v| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    let id = id.ok_or_else(|| EsError::parsing("[parent_id] requires 'id' field"))?;
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    check_expensive(ctx)?;
    let Some(jf) = join_field(ctx.lookup) else {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, "[parent_id] join field has not been configured"));
    };
    if Relations::of(&jf).parent_of(ty).is_none() {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, format!("[parent_id] no relation found for child [{ty}]")));
    }
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(Query::MatchNone) };
    let q = Query::ConstScore { query: Box::new(filter2(term(jid, ty), term(pid, &id))), score: 1.0, point_set: false };
    Ok(with_boost(q, boost_of(o)))
}

/// The segments of `host` per physical shard (index and shard): a
/// parent and its children live on one shard, and `_id`s are only unique
/// within one index (#603).
fn shard_groups(host: &super::aggs::Host) -> Vec<Vec<usize>> {
    let mut groups: Vec<((&str, usize), Vec<usize>)> = Vec::new();
    for (i, s) in host.segs.iter().enumerate() {
        match groups.iter_mut().find(|(k, _)| *k == (s.index, s.shard)) {
            Some((_, v)) => v.push(i),
            None => groups.push(((s.index, s.shard), vec![i])),
        }
    }
    groups.into_iter().map(|(_, v)| v).collect()
}

/// `children` aggregation: the child docs (of `type`) of the parent docs,
/// joined within each shard with that shard's join field.
pub fn children_docs(o: &Map<String, Value>, docs: &Docs, host: &super::aggs::Host) -> Result<Docs> {
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing_at("Missing [type] field for children aggregation", 1, 1))?;
    let mut out: Docs = vec![Vec::new(); docs.len()];
    if host.segs.first().and_then(|s| join_field(s.lookup)).is_none() {
        return Ok(out);
    }
    let mut related = false;
    for group in shard_groups(host) {
        let Some(jf) = join_field(host.segs[group[0]].lookup) else { continue };
        let rel = Relations::of(&jf);
        let Some(parent_ty) = rel.parent_of(ty) else { continue };
        related = true;
        let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { continue };
        let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for &s in &group {
            let leaf = &host.segs[s].leaf;
            for &(d, _) in &docs[s] {
                if leaf.keyword_value(jid, d).as_deref() == Some(parent_ty) {
                    if let Some(id) = leaf.keyword_value(ID_FIELD, d) {
                        ids.insert(id);
                    }
                }
            }
        }
        if ids.is_empty() {
            continue;
        }
        for &s in &group {
            let seg = &host.segs[s];
            for d in (host.all_docs)(s) {
                if seg.leaf.keyword_value(jid, d).as_deref() == Some(ty) && seg.leaf.keyword_values(pid, d).iter().any(|p| ids.contains(p)) {
                    out[s].push((d, 0.0));
                }
            }
        }
    }
    if !related {
        return Err(EsError::illegal_argument(format!("[children] no join field with parent/child relation was found for child type [{ty}]")));
    }
    Ok(out)
}

/// `parent` aggregation: the parent docs of the child docs (of `type`),
/// joined within each shard with that shard's join field.
pub fn parent_docs(o: &Map<String, Value>, docs: &Docs, host: &super::aggs::Host) -> Result<Docs> {
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing_at("Missing [type] field for parent aggregation", 1, 1))?;
    let mut out: Docs = vec![Vec::new(); docs.len()];
    if host.segs.first().and_then(|s| join_field(s.lookup)).is_none() {
        return Ok(out);
    }
    let mut related = false;
    for group in shard_groups(host) {
        let Some(jf) = join_field(host.segs[group[0]].lookup) else { continue };
        let rel = Relations::of(&jf);
        let Some(parent_ty) = rel.parent_of(ty).map(str::to_string) else { continue };
        related = true;
        let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { continue };
        let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for &s in &group {
            let leaf = &host.segs[s].leaf;
            for &(d, _) in &docs[s] {
                if leaf.keyword_value(jid, d).as_deref() == Some(ty) {
                    ids.extend(leaf.keyword_values(pid, d));
                }
            }
        }
        if ids.is_empty() {
            continue;
        }
        for &s in &group {
            let seg = &host.segs[s];
            for d in (host.all_docs)(s) {
                if seg.leaf.keyword_value(jid, d).as_deref() == Some(parent_ty.as_str()) && seg.leaf.keyword_value(ID_FIELD, d).map(|i| ids.contains(&i)).unwrap_or(false) {
                    out[s].push((d, 0.0));
                }
            }
        }
    }
    if !related {
        return Err(EsError::illegal_argument(format!("[parent] no join field with parent/child relation was found for child type [{ty}]")));
    }
    Ok(out)
}
