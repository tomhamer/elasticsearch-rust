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
    let child_q = Query::Bool { must: vec![inner_q], should: vec![], must_not: vec![], filter: vec![term(jid, ty)], minimum_should_match: MinShouldMatch::Count(0) };
    let mut per_parent: HashMap<String, Vec<f32>> = HashMap::new();
    for (li, d, s) in run(&child_q, ctx.leaves)? {
        for p in ctx.leaves[li].keyword_values(pid, d) {
            per_parent.entry(p).or_default().push(s);
        }
    }
    let scores: HashMap<String, f32> = per_parent
        .into_iter()
        .filter(|(_, v)| v.len() >= min_children && v.len() <= max_children)
        .map(|(k, v)| {
            let s = match mode {
                ScoreMode::None => 1.0,
                ScoreMode::Avg => v.iter().sum::<f32>() / v.len() as f32,
                ScoreMode::Sum => v.iter().sum(),
                ScoreMode::Max => v.iter().copied().fold(f32::MIN, f32::max),
                ScoreMode::Min => v.iter().copied().fold(f32::MAX, f32::min),
            };
            (k, s)
        })
        .collect();
    if scores.is_empty() {
        return Ok(Query::MatchNone);
    }
    let ids: Vec<Vec<u8>> = scores.keys().map(|k| k.clone().into_bytes()).collect();
    let cand = filter2(Query::TermSet { field: ID_FIELD, terms: ids, boost: 1.0 }, term(jid, &parent_ty));
    let scores = Arc::new(scores);
    let q = FilterQuery::new("has_child", Some(cand), Arc::new(move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_value(ID_FIELD, d).and_then(|id| scores.get(&id).copied())));
    Ok(with_boost(q.into_query(), boost_of(o)))
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
    let parent_q = Query::Bool { must: vec![parse_query(inner, ctx)?], should: vec![], must_not: vec![], filter: vec![term(jid, pty)], minimum_should_match: MinShouldMatch::Count(0) };
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (li, d, s) in run(&parent_q, ctx.leaves)? {
        if let Some(id) = ctx.leaves[li].keyword_value(ID_FIELD, d) {
            scores.insert(id, if score { s } else { 1.0 });
        }
    }
    if scores.is_empty() {
        return Ok(Query::MatchNone);
    }
    let ids: Vec<Vec<u8>> = scores.keys().map(|k| k.clone().into_bytes()).collect();
    let types = Query::TermSet { field: jid, terms: children.iter().map(|c| c.clone().into_bytes()).collect(), boost: 1.0 };
    let cand = filter2(Query::TermSet { field: pid, terms: ids, boost: 1.0 }, types);
    let scores = Arc::new(scores);
    let q = FilterQuery::new("has_parent", Some(cand), Arc::new(move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_values(pid, d).iter().find_map(|p| scores.get(p).copied())));
    Ok(with_boost(q.into_query(), boost_of(o)))
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
    let query = Query::Bool { must: vec![parse_query(inner, ctx)?], should: vec![], must_not: vec![], filter: vec![term(jid, &ty)], minimum_should_match: MinShouldMatch::Count(0) };
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

/// `children` aggregation: the child docs (of `type`) of the parent docs in
/// `docs`, over all docs of the searched leaves.
pub fn children_docs(o: &Map<String, Value>, docs: &Docs, host: &super::aggs::Host) -> Result<Docs> {
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing_at("Missing [type] field for children aggregation", 1, 1))?;
    let mut out: Docs = vec![Vec::new(); docs.len()];
    let Some(jf) = host.segs.first().and_then(|s| join_field(s.lookup)) else { return Ok(out) };
    let rel = Relations::of(&jf);
    let Some(parent_ty) = rel.parent_of(ty) else {
        return Err(EsError::illegal_argument(format!("[children] no join field with parent/child relation was found for child type [{ty}]")));
    };
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(out) };
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (s, v) in docs.iter().enumerate() {
        let leaf = &host.segs[s].leaf;
        for &(d, _) in v {
            if leaf.keyword_value(jid, d).as_deref() == Some(parent_ty) {
                if let Some(id) = leaf.keyword_value(ID_FIELD, d) {
                    ids.insert(id);
                }
            }
        }
    }
    for (s, seg) in host.segs.iter().enumerate() {
        for d in (host.all_docs)(s) {
            if seg.leaf.keyword_value(jid, d).as_deref() == Some(ty) && seg.leaf.keyword_values(pid, d).iter().any(|p| ids.contains(p)) {
                out[s].push((d, 0.0));
            }
        }
    }
    Ok(out)
}

/// `parent` aggregation: the parent docs of the child docs (of `type`).
pub fn parent_docs(o: &Map<String, Value>, docs: &Docs, host: &super::aggs::Host) -> Result<Docs> {
    let ty = o.get("type").and_then(Value::as_str).ok_or_else(|| EsError::parsing_at("Missing [type] field for parent aggregation", 1, 1))?;
    let mut out: Docs = vec![Vec::new(); docs.len()];
    let Some(jf) = host.segs.first().and_then(|s| join_field(s.lookup)) else { return Ok(out) };
    let rel = Relations::of(&jf);
    let Some(parent_ty) = rel.parent_of(ty).map(str::to_string) else {
        return Err(EsError::illegal_argument(format!("[parent] no join field with parent/child relation was found for child type [{ty}]")));
    };
    let (Some(jid), Some(pid)) = (jf.core_id, jf.aux_id("parent")) else { return Ok(out) };
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (s, v) in docs.iter().enumerate() {
        let leaf = &host.segs[s].leaf;
        for &(d, _) in v {
            if leaf.keyword_value(jid, d).as_deref() == Some(ty) {
                ids.extend(leaf.keyword_values(pid, d));
            }
        }
    }
    for (s, seg) in host.segs.iter().enumerate() {
        for d in (host.all_docs)(s) {
            if seg.leaf.keyword_value(jid, d).as_deref() == Some(parent_ty.as_str()) && seg.leaf.keyword_value(ID_FIELD, d).map(|i| ids.contains(&i)).unwrap_or(false) {
                out[s].push((d, 0.0));
            }
        }
    }
    Ok(out)
}
