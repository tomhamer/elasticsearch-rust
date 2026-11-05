//! Retrievers (`retriever` in `_search`, ES 8.14+ semantics) and the legacy
//! `rank: { rrf: {...} }` + `sub_searches` form (ES 8.8+, converted to an
//! `rrf` retriever since 8.16).
//!
//! * `standard` / `knn` at the top level are plain rewrites of the request
//!   (`query` + `filter`, `sort`, `search_after`, `terminate_after`,
//!   `min_score`, `collapse` / a top-level `knn` section).
//! * Compound retrievers (`rrf`, `linear`, `rescorer`, `rule`, `pinned`) run
//!   every child as its own search for its top `rank_window_size` hits,
//!   fuse them (`rustic_core::vector::fusion::{rrf, linear}`), and finish
//!   with one search whose hit query is a [`RankDocsQuery`] (the fused docs
//!   with their fused scores, keyed by `_id`) and whose aggregation /
//!   total-hits query is the union of the children's queries (ES
//!   `RankDocsQuery`: rank docs + a zero-scored tail).
//! * `text_similarity_reranker` needs an inference endpoint: 404 as ES does
//!   without one.

use crate::error::{EsError, Result};
use crate::mapping::ID_FIELD;
use crate::search::{self, ExecOpts, SearchOutput, SearchRequest, SourceFilter, Target};
use rustic_core::query::query::{CustomQuery, ScoredDocs, SearchContext};
use rustic_core::query::{Explanation, LeafReader, MinShouldMatch, Query, Scorer, TopDocs, TotalHitsRelation, TrackTotalHits};
use rustic_core::types::{DocId, Term};
use rustic_core::vector::fusion::{self, Normalization};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_RANK_WINDOW_SIZE: usize = 10;

/// The search body key carrying a serialized [`RankPlan`] between a
/// cluster coordinator and the shards (accepted only from the coordinator).
pub const RANK_PLAN_KEY: &str = "_rank_plan";

/// A score as ES renders it (a Java `float`: shortest `f32` repr);
/// non-finite scores render as `null`.
pub fn score_json(x: f32) -> String {
    if x.is_finite() {
        serde_json::to_string(&x).unwrap_or_else(|_| "null".into())
    } else {
        "null".into()
    }
}
pub const DEFAULT_RANK_CONSTANT: f32 = 60.0;

/// Parameters every retriever accepts (`RetrieverBuilder`).
#[derive(Clone, Debug, Default)]
pub struct Common {
    /// Pre-filters (`filter`), applied to every child.
    pub filter: Vec<Value>,
    pub min_score: Option<f32>,
    pub name: Option<String>,
    /// The legacy `rank: {rrf}` form (its own explanation wording).
    pub legacy: bool,
}

#[derive(Clone, Debug)]
pub enum Normalizer {
    None,
    MinMax,
    L2,
}

impl Normalizer {
    fn parse(s: &str) -> Result<Normalizer> {
        Ok(match s {
            "none" => Normalizer::None,
            "minmax" => Normalizer::MinMax,
            "l2_norm" => Normalizer::L2,
            other => return Err(EsError::illegal_argument(format!("Unknown normalizer [{other}]"))),
        })
    }
    fn name(&self) -> &'static str {
        match self {
            Normalizer::None => "none",
            Normalizer::MinMax => "minmax",
            Normalizer::L2 => "l2_norm",
        }
    }
    fn core(&self) -> Normalization {
        match self {
            Normalizer::None => Normalization::None,
            Normalizer::MinMax => Normalization::MinMax,
            Normalizer::L2 => Normalization::L2,
        }
    }
}

/// A pinned / excluded document (`ids` or `docs` of query rules / pinned).
#[derive(Clone, Debug, PartialEq)]
pub struct DocRef {
    pub index: Option<String>,
    pub id: String,
}

#[derive(Clone, Debug)]
pub enum Retriever {
    Standard { c: Common, query: Option<Value>, sort: Option<Value>, search_after: Option<Value>, terminate_after: Option<u64>, collapse: Option<Value> },
    Knn { c: Common, spec: Map<String, Value> },
    Rrf { c: Common, children: Vec<(Retriever, f32)>, window: usize, rank_constant: f32 },
    Linear { c: Common, children: Vec<(Retriever, f32, Normalizer)>, window: usize },
    Rescorer { c: Common, child: Box<Retriever>, rescore: Value, window: usize },
    /// `rule` and `pinned`: `pinned` docs first, `excluded` docs removed.
    /// `rule_info`: a `rule` retriever's `ruleset_ids` and `match_criteria` (explanations).
    Pinned { c: Common, kind: &'static str, child: Box<Retriever>, pinned: Vec<DocRef>, excluded: Vec<DocRef>, window: usize, sort_checked: bool, rule_info: Option<(Vec<String>, Value)> },
    /// `text_similarity_reranker`: the child's top `window` hits re-scored by
    /// an `_inference` rerank endpoint over `field`.
    /// `chunk`: the `chunk_rescorer` (`size` best chunks per doc, chunking
    /// settings; `None` settings resolve to the endpoint's window).
    Rerank { c: Common, child: Box<Retriever>, field: String, inference_id: String, inference_text: String, window: usize, allow_failures: bool, chunk: Option<(Option<i64>, Option<Value>)> },
    /// `random_reranker` (the inference plugin's test retriever): the
    /// child's top `window` hits re-scored with `java.util.Random(seed + doc)`.
    Random { c: Common, child: Box<Retriever>, window: usize, seed: Option<i64> },
    /// `rrf` / `linear` in the multi-field query format (`fields` + `query`):
    /// rewritten against the target index's mapping before it runs
    /// (`MultiFieldsInnerRetrieverUtils.generateInnerRetrievers`).
    MultiField { c: Common, kind: &'static str, fields: Option<Vec<String>>, query: String, window: usize, rank_constant: f32, normalizer: Normalizer },
    /// `diversify` (ES 9.2+ `DiversifyRetrieverBuilder`, type `mmr`): the
    /// child's top `window` hits re-selected by maximal marginal relevance
    /// over the vectors of `field`, `size` of them kept in child order.
    Diversify { c: Common, child: Box<Retriever>, field: String, window: usize, size: i64, lambda: f32, query_vector: Option<Value> },
}

impl Retriever {
    pub fn name(&self) -> &'static str {
        match self {
            Retriever::Standard { .. } => "standard",
            Retriever::Knn { .. } => "knn",
            Retriever::Rrf { .. } => "rrf",
            Retriever::Linear { .. } => "linear",
            Retriever::Rescorer { .. } => "rescorer",
            Retriever::Pinned { kind, .. } => *kind,
            Retriever::Rerank { .. } => "text_similarity_reranker",
            Retriever::Random { .. } => "random_reranker",
            Retriever::MultiField { kind, .. } => *kind,
            Retriever::Diversify { .. } => "diversify",
        }
    }
    fn common(&self) -> &Common {
        match self {
            Retriever::Standard { c, .. }
            | Retriever::Knn { c, .. }
            | Retriever::Rrf { c, .. }
            | Retriever::Linear { c, .. }
            | Retriever::Rescorer { c, .. }
            | Retriever::Pinned { c, .. }
            | Retriever::Rerank { c, .. }
            | Retriever::Random { c, .. }
            | Retriever::MultiField { c, .. }
            | Retriever::Diversify { c, .. } => c,
        }
    }
    /// Rank window of a compound retriever.
    fn window(&self) -> Option<usize> {
        match self {
            Retriever::Rrf { window, .. }
            | Retriever::Linear { window, .. }
            | Retriever::Rescorer { window, .. }
            | Retriever::Pinned { window, .. }
            | Retriever::Rerank { window, .. }
            | Retriever::Random { window, .. }
            | Retriever::MultiField { window, .. }
            | Retriever::Diversify { window, .. } => Some(*window),
            _ => None,
        }
    }
    fn window_field(&self) -> &'static str {
        match self {
            Retriever::Rescorer { .. } => "window_size",
            _ => "rank_window_size",
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing.

fn xparse(name: &str, reason: impl std::fmt::Display) -> EsError {
    EsError::x_content_parse(format!("[1:1] [{name}] {reason}"))
}

fn unknown_field(name: &str, k: &str) -> EsError {
    xparse(name, format!("unknown field [{k}]"))
}

fn f32_of(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|x| x as f32),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn window_of(name: &str, o: &Map<String, Value>, key: &str) -> Result<usize> {
    match o.get(key) {
        None => Ok(DEFAULT_RANK_WINDOW_SIZE),
        Some(v) => match v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())) {
            Some(n) if n < 0 => Err(EsError::illegal_argument(format!("[{key}] parameter cannot be negative, found [{n}]"))),
            Some(n) => Ok(n as usize),
            None => Err(xparse(name, format!("failed to parse field [{key}]"))),
        },
    }
}

fn filters_of(v: Option<&Value>) -> Vec<Value> {
    match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(other) => vec![other.clone()],
    }
}

/// Common fields; `true` when `k` was one of them.
fn common_field(c: &mut Common, k: &str, v: &Value) -> bool {
    match k {
        "filter" => c.filter = filters_of(Some(v)),
        "min_score" => c.min_score = f32_of(v),
        "_name" => c.name = v.as_str().map(str::to_string),
        _ => return false,
    }
    true
}

/// Parse `{ "<type>": { ... } }`.
fn str_list(name: &str, k: &str, v: &Value) -> Result<Vec<String>> {
    v.as_array()
        .and_then(|a| a.iter().map(|x| x.as_str().map(str::to_string)).collect::<Option<Vec<_>>>())
        .ok_or_else(|| xparse(name, format!("failed to parse field [{k}]")))
}

/// `MultiFieldsInnerRetrieverUtils.validateParams`.
fn multi_field_checks(name: &str, has_retrievers: bool, fields: &Option<Vec<String>>, query: &Option<String>, errs: &mut Vec<String>) {
    if fields.is_some() || query.is_some() {
        let Some(q) = query else {
            errs.push(format!("[{name}] [query] must be provided when [fields] is specified"));
            return;
        };
        if q.is_empty() {
            errs.push(format!("[{name}] [query] cannot be empty"));
        }
        if fields.as_ref().is_some_and(Vec::is_empty) {
            errs.push(format!("[{name}] [fields] cannot be empty"));
        }
        if has_retrievers {
            errs.push(format!("[{name}] cannot combine [retrievers] and [query]"));
        }
    } else if !has_retrievers {
        errs.push(format!("[{name}] must provide [retrievers] or [query]"));
    }
}

pub fn parse(v: &Value) -> Result<Retriever> {
    let o = v.as_object().ok_or_else(|| EsError::parsing("retriever malformed, must start with [START_OBJECT]"))?;
    let mut it = o.iter();
    let (name, body) = match it.next() {
        Some(x) => x,
        None => return Err(EsError::parsing("retriever malformed, empty clause found")),
    };
    if let Some((extra, _)) = it.next() {
        return Err(EsError::parsing(format!("retriever malformed, found [{extra}] after the closing bracket of [{name}]")));
    }
    let b = body.as_object().ok_or_else(|| EsError::parsing(format!("retriever malformed, [{name}] must be an object")))?;
    let mut c = Common::default();
    match name.as_str() {
        "standard" => {
            let (mut query, mut sort, mut search_after, mut terminate_after, mut collapse) = (None, None, None, None, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "query" => query = Some(v.clone()),
                    "sort" => sort = Some(v.clone()),
                    "search_after" => search_after = Some(v.clone()),
                    "terminate_after" => terminate_after = v.as_u64(),
                    "collapse" => collapse = Some(v.clone()),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            Ok(Retriever::Standard { c, query, sort, search_after, terminate_after, collapse })
        }
        "knn" => {
            let mut spec = Map::new();
            for (k, v) in b {
                match k.as_str() {
                    "filter" => c.filter = filters_of(Some(v)),
                    "_name" => c.name = v.as_str().map(str::to_string),
                    "min_score" => c.min_score = f32_of(v),
                    "field" | "query_vector" | "query_vector_builder" | "k" | "num_candidates" | "similarity" | "rescore_vector" | "boost" | "visit_percentage" => {
                        spec.insert(k.clone(), v.clone());
                    }
                    _ => return Err(unknown_field(name, k)),
                }
            }
            if !spec.contains_key("field") {
                return Err(xparse(name, "Required [field]"));
            }
            if spec.contains_key("query_vector") && spec.contains_key("query_vector_builder") {
                return Err(EsError::illegal_argument("cannot provide both [query_vector_builder] and [query_vector]"));
            }
            if let Some(k) = spec.get("k").and_then(Value::as_i64) {
                if k < 1 {
                    return Err(EsError::illegal_argument("[k] must be greater than 0"));
                }
            }
            Ok(Retriever::Knn { c, spec })
        }
        "rrf" => {
            let mut children = Vec::new();
            let mut window = DEFAULT_RANK_WINDOW_SIZE;
            let mut rank_constant = DEFAULT_RANK_CONSTANT;
            let (mut fields, mut query) = (None, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "retrievers" => {
                        let a = v.as_array().ok_or_else(|| xparse(name, "[retrievers] must be an array"))?;
                        for x in a {
                            match x.get("retriever") {
                                Some(r) if x.as_object().map(|m| m.len() <= 2).unwrap_or(false) => {
                                    let w = x.get("weight").and_then(f32_of).unwrap_or(1.0);
                                    if w < 0.0 {
                                        // `RRFRetrieverComponent`'s constructor, under the
                                        // `retrievers` array parser.
                                        return Err(xparse(name, "failed to parse field [retrievers]").caused_by(EsError::illegal_argument(format!("[weight] must be non-negative, found [{w}]"))));
                                    }
                                    children.push((parse(r)?, w));
                                }
                                _ => children.push((parse(x)?, 1.0)),
                            }
                        }
                    }
                    "rank_window_size" => window = window_of(name, b, k)?,
                    "rank_constant" => {
                        let rc = v.as_f64().ok_or_else(|| xparse(name, "failed to parse field [rank_constant]"))?;
                        if rc < 1.0 {
                            return Err(EsError::illegal_argument("[rank_constant] must be greater or equal to [1] for [rrf]"));
                        }
                        rank_constant = rc as f32;
                    }
                    "fields" => fields = Some(str_list(name, k, v)?),
                    "query" => query = Some(v.as_str().map(str::to_string).ok_or_else(|| xparse(name, "failed to parse field [query]"))?),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            let mut errs = Vec::new();
            multi_field_checks(name, !children.is_empty(), &fields, &query, &mut errs);
            if !errs.is_empty() {
                let r: Vec<&str> = errs.iter().map(String::as_str).collect();
                return Err(EsError::action_validation(&r));
            }
            if let Some(query) = query {
                return Ok(Retriever::MultiField { c, kind: "rrf", fields, query, window, rank_constant, normalizer: Normalizer::None });
            }
            Ok(Retriever::Rrf { c, children, window, rank_constant })
        }
        "linear" => {
            let mut raw = Vec::new();
            let mut window = DEFAULT_RANK_WINDOW_SIZE;
            let mut top_norm: Option<Normalizer> = None;
            let (mut fields, mut query) = (None, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "retrievers" => raw = v.as_array().cloned().ok_or_else(|| xparse(name, "[retrievers] must be an array"))?,
                    "rank_window_size" => window = window_of(name, b, k)?,
                    "normalizer" => top_norm = Some(Normalizer::parse(v.as_str().unwrap_or(""))?),
                    "fields" => fields = Some(str_list(name, k, v)?),
                    "query" => query = Some(v.as_str().map(str::to_string).ok_or_else(|| xparse(name, "failed to parse field [query]"))?),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            let mut errs = Vec::new();
            multi_field_checks(name, !raw.is_empty(), &fields, &query, &mut errs);
            if query.is_some() && top_norm.is_none() {
                errs.push("[linear] [normalizer] must be provided when [query] is specified".into());
            } else if !raw.is_empty() && top_norm.is_some() && crate::version::current_rest_api_version().is_v8() {
                // 8.x refused a top-level normalizer next to `retrievers`;
                // 9.x makes it the components' default.
                errs.push("[linear] [normalizer] cannot be provided when [retrievers] is specified".into());
            }
            if !errs.is_empty() {
                let r: Vec<&str> = errs.iter().map(String::as_str).collect();
                return Err(EsError::action_validation(&r));
            }
            if let Some(query) = query {
                return Ok(Retriever::MultiField { c, kind: "linear", fields, query, window, rank_constant: DEFAULT_RANK_CONSTANT, normalizer: top_norm.unwrap_or(Normalizer::None) });
            }
            // `LinearRetrieverBuilder.resolveNormalizer`: a component's own
            // normalizer, else the top-level one, else `none`.
            let default_norm = top_norm.clone().unwrap_or(Normalizer::None);
            let mut children = Vec::new();
            for x in &raw {
                let xo = x.as_object().ok_or_else(|| xparse(name, "[retrievers] entries must be objects"))?;
                if let Some(r) = xo.get("retriever") {
                    let mut w = 1.0f32;
                    let mut norm = default_norm.clone();
                    for (k, v) in xo {
                        match k.as_str() {
                            "retriever" => {}
                            "weight" => {
                                w = f32_of(v).ok_or_else(|| xparse("linear", "failed to parse field [weight]"))?;
                                if w < 0.0 {
                                    return Err(EsError::illegal_argument(format!("[weight] must be non-negative, found [{w}]")));
                                }
                            }
                            "normalizer" => norm = Normalizer::parse(v.as_str().unwrap_or(""))?,
                            _ => return Err(unknown_field("linear", k)),
                        }
                    }
                    children.push((parse(r)?, w, norm));
                } else {
                    children.push((parse(x)?, 1.0, default_norm.clone()));
                }
            }
            Ok(Retriever::Linear { c, children, window })
        }
        "rescorer" => {
            let (mut child, mut rescore) = (None, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "retriever" => child = Some(parse(v)?),
                    "rescore" => rescore = Some(v.clone()),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            let child = child.ok_or_else(|| xparse(name, "Required [retriever]"))?;
            let rescore = rescore.ok_or_else(|| xparse(name, "Required [rescore]"))?;
            let rs: Vec<&Value> = match &rescore {
                Value::Array(a) => a.iter().collect(),
                o => vec![o],
            };
            if rs.is_empty() {
                return Err(xparse(name, "[rescore] must not be empty"));
            }
            let window = rs.iter().map(|r| r.get("window_size").and_then(Value::as_u64).unwrap_or(10) as usize).max().unwrap_or(10);
            Ok(Retriever::Rescorer { c, child: Box::new(child), rescore, window })
        }
        "rule" | "pinned" => {
            let kind: &'static str = if name == "rule" { "rule" } else { "pinned" };
            let mut child = None;
            let mut window = DEFAULT_RANK_WINDOW_SIZE;
            let (mut pinned, mut excluded) = (Vec::new(), Vec::new());
            let (mut n_ids, mut n_docs) = (None, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match (kind, k.as_str()) {
                    (_, "retriever") => child = Some(parse(v)?),
                    (_, "rank_window_size") => window = window_of(name, b, k)?,
                    ("rule", "ruleset_ids") | ("rule", "match_criteria") => {}
                    ("rule", "_resolved") => {
                        pinned = doc_refs(v.get("pinned"));
                        excluded = doc_refs(v.get("excluded"));
                    }
                    ("pinned", "ids") => {
                        let ids: Vec<DocRef> = v.as_array().into_iter().flatten().map(|x| DocRef { index: None, id: crate::query::dsl::scalar_string(x) }).collect();
                        if let Some(d) = ids.iter().enumerate().find(|(i, d)| ids[..*i].iter().any(|o| o.id == d.id)) {
                            return Err(EsError::illegal_argument(format!("duplicate id found [{}]", d.1.id)));
                        }
                        n_ids = Some(ids.len());
                        pinned.extend(ids);
                    }
                    ("pinned", "docs") => {
                        let docs = doc_refs(Some(v));
                        if let Some(d) = docs.iter().enumerate().find(|(i, d)| docs[..*i].iter().any(|o| o.id == d.id && o.index == d.index)) {
                            return Err(EsError::illegal_argument(format!("duplicate doc found [{}]", d.1.id)));
                        }
                        n_docs = Some(docs.len());
                        pinned.extend(docs);
                    }
                    _ => return Err(unknown_field(name, k)),
                }
            }
            if kind == "rule" {
                if !b.contains_key("ruleset_ids") {
                    return Err(xparse(name, "Required [ruleset_ids]"));
                }
                if !b.contains_key("match_criteria") {
                    return Err(xparse(name, "Required [match_criteria]"));
                }
            } else if n_ids.is_some() && n_docs.is_some() && (n_ids != Some(0) || n_docs != Some(0)) {
                return Err(EsError::illegal_argument("Both 'ids' and 'docs' cannot be specified at the same time"));
            } else if n_ids.unwrap_or(0) + n_docs.unwrap_or(0) == 0 {
                return Err(EsError::illegal_argument("Either 'ids' or 'docs' must be provided and non-empty for pinned retriever"));
            }
            let child = child.ok_or_else(|| xparse(name, "Required [retriever]"))?;
            if kind == "pinned" {
                if let Retriever::Standard { query: None, .. } = &child {
                    return Err(EsError::illegal_argument("pinned retriever requires retriever with associated query"));
                }
            }
            let rule_info = (kind == "rule").then(|| {
                let ids: Vec<String> = match b.get("ruleset_ids") {
                    Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect(),
                    Some(Value::String(s)) => vec![s.clone()],
                    _ => Vec::new(),
                };
                (ids, b.get("match_criteria").cloned().unwrap_or(Value::Null))
            });
            Ok(Retriever::Pinned { c, kind, child: Box::new(child), pinned, excluded, window, sort_checked: false, rule_info })
        }
        "text_similarity_reranker" => {
            let (mut child, mut field, mut inference_id, mut text, mut window) = (None, None, None, None, DEFAULT_RANK_WINDOW_SIZE);
            let mut allow_failures = false;
            let mut chunk = None;
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "retriever" => child = Some(parse(v)?),
                    "field" => field = v.as_str().map(str::to_string),
                    "inference_id" => inference_id = v.as_str().map(str::to_string),
                    "inference_text" => text = v.as_str().map(str::to_string),
                    "rank_window_size" => window = v.as_u64().ok_or_else(|| xparse(name, "[rank_window_size] must be a number"))? as usize,
                    "allow_rerank_failures" => allow_failures = v.as_bool().ok_or_else(|| xparse(name, "failed to parse field [allow_rerank_failures]"))?,
                    "chunk_rescorer" => chunk = Some(parse_chunk_rescorer(v)?),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            let child = child.ok_or_else(|| xparse(name, "Required [retriever]"))?;
            let field = field.ok_or_else(|| xparse(name, "Required [field]"))?;
            let inference_text = text.ok_or_else(|| xparse(name, "Required [inference_text]"))?;
            let inference_id = inference_id.unwrap_or_else(|| ".rerank-v1-elasticsearch".to_string());
            match crate::inference::current().and_then(|s| s.get_endpoint(&inference_id)) {
                Some(ep) if ep.task_type == crate::inference::providers::Task::Rerank => {}
                Some(ep) => return Err(EsError::illegal_argument(format!("Inference endpoint [{inference_id}] does not support the task type [rerank]; it is a [{}] endpoint", ep.task_type.name()))),
                // With `allow_rerank_failures` a missing endpoint is a rerank
                // failure: the child's ranking is kept.
                None if allow_failures => {}
                None => return Err(EsError::resource_not_found(format!("Inference endpoint not found [{inference_id}]"))),
            }
            if window == 0 {
                return Err(EsError::illegal_argument("[rank_window_size] must be greater than [0]"));
            }
            Ok(Retriever::Rerank { c, child: Box::new(child), field, inference_id, inference_text, window, allow_failures, chunk })
        }
        "random_reranker" => {
            let (mut child, mut window, mut seed) = (None, DEFAULT_RANK_WINDOW_SIZE, None);
            for (k, v) in b {
                if common_field(&mut c, k, v) {
                    continue;
                }
                match k.as_str() {
                    "retriever" => child = Some(parse(v)?),
                    "field" => {}
                    "rank_window_size" => window = v.as_u64().ok_or_else(|| xparse(name, "[rank_window_size] must be a number"))? as usize,
                    "seed" => seed = Some(v.as_i64().ok_or_else(|| xparse(name, "failed to parse field [seed]"))?),
                    _ => return Err(unknown_field(name, k)),
                }
            }
            let child = child.ok_or_else(|| xparse(name, "Required [retriever]"))?;
            if b.get("field").is_none() {
                return Err(xparse(name, "Required [field]"));
            }
            Ok(Retriever::Random { c, child: Box::new(child), window, seed })
        }
        "diversify" => parse_diversify(c, b),
        other => Err(EsError::parsing(format!("unknown retriever [{other}]"))
            .with("line", 1)
            .with("col", 1)
            .caused_by(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{other}]")))),
    }
}

/// `RerankingInferenceService.rerankerWindowSize`: the default chunk size
/// (in words) of a chunk rescorer (the test reranking service: 333).
fn reranker_window_size(ep: Option<&crate::inference::Endpoint>) -> usize {
    match ep.map(|e| e.service.as_str()) {
        Some(crate::inference::mock::RERANK) => 333,
        _ => 300,
    }
}

/// `MemoryIndexChunkScorer.scoreChunks(chunks, text, size, backfill)`: the
/// `size` chunks best matching `text` (BM25 over the chunks, standard
/// analyzer, OR of the terms), else the first `size` chunks.
fn best_chunks(chunks: &[String], text: &str, size: usize) -> Vec<String> {
    if chunks.is_empty() || text.trim().is_empty() {
        return Vec::new();
    }
    let an = rustic_core::analysis::Analyzer::standard();
    let docs: Vec<Vec<String>> = chunks.iter().map(|c| an.analyze(c).into_iter().map(|t| t.text).collect()).collect();
    let mut q: Vec<String> = an.analyze(text).into_iter().map(|t| t.text).collect();
    q.dedup();
    let n = docs.len() as f32;
    let avgdl = docs.iter().map(Vec::len).sum::<usize>() as f32 / n.max(1.0);
    let mut scored: Vec<(usize, f32)> = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        let mut s = 0f32;
        let mut hit = false;
        for t in &q {
            let tf = d.iter().filter(|x| *x == t).count() as f32;
            if tf == 0.0 {
                continue;
            }
            hit = true;
            let df = docs.iter().filter(|x| x.contains(t)).count() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            s += idf * tf / (tf + 1.2 * (0.25 + 0.75 * d.len() as f32 / avgdl.max(f32::MIN_POSITIVE)));
        }
        if hit {
            scored.push((i, s));
        }
    }
    if scored.is_empty() {
        return chunks.iter().take(size).cloned().collect();
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(size).map(|(i, _)| chunks[i].clone()).collect()
}

/// `TextSimilarityRankRetrieverBuilder.CHUNK_SCORER_PARSER` (lenient):
/// `(size, chunking_settings)`; chunking settings are validated here
/// (`ChunkScorerConfig.chunkingSettingsFromMap(map, true)`).
fn parse_chunk_rescorer(v: &Value) -> Result<(Option<i64>, Option<Value>)> {
    let o = v.as_object().ok_or_else(|| xparse("text_similarity_reranker", "[chunk_rescorer] must be an object"))?;
    let size = match o.get("size") {
        None | Some(Value::Null) => None,
        Some(x) => Some(x.as_i64().ok_or_else(|| xparse("chunk_rescorer", "failed to parse field [size]"))?),
    };
    if let Some(n) = size.filter(|n| *n < 1) {
        return Err(xparse("text_similarity_reranker", "failed to parse field [chunk_rescorer]").caused_by(EsError::illegal_argument(format!("size must be greater than 0, was: {n}"))));
    }
    let settings = match o.get("chunking_settings") {
        None | Some(Value::Null) => None,
        Some(Value::Object(m)) if m.len() == 1 && m.contains_key("max_chunk_size") => {
            let n = m["max_chunk_size"].as_i64().ok_or_else(|| EsError::illegal_argument("max_chunk_size must not be null"))?;
            Some(crate::inference::semantic_map::chunking_settings(&json!({"strategy": "sentence", "max_chunk_size": n, "sentence_overlap": 0}))?.unwrap_or(Value::Null))
        }
        Some(other) => crate::inference::semantic_map::chunking_settings(other)?,
    };
    Ok((size, settings))
}

/// `DiversifyRetrieverBuilder.PARSER` + `validate`.
fn parse_diversify(mut c: Common, b: &Map<String, Value>) -> Result<Retriever> {
    let name = "diversify";
    let (mut child, mut kind, mut field, mut window, mut size, mut lambda, mut qv, mut qvb) = (None, None, None, DEFAULT_RANK_WINDOW_SIZE, None, None, None, None);
    for (k, v) in b {
        if common_field(&mut c, k, v) {
            continue;
        }
        match k.as_str() {
            "retriever" => child = Some(parse(v)?),
            "type" => kind = Some(v.as_str().map(str::to_string).ok_or_else(|| xparse(name, "failed to parse field [type]"))?),
            "field" => field = Some(v.as_str().map(str::to_string).ok_or_else(|| xparse(name, "failed to parse field [field]"))?),
            "rank_window_size" => window = window_of(name, b, k)?,
            "query_vector" => qv = Some(v.clone()),
            "query_vector_builder" => qvb = Some(v.clone()),
            "lambda" => lambda = Some(f32_of(v).ok_or_else(|| xparse(name, "failed to parse field [lambda]"))?),
            "size" => size = Some(v.as_i64().or_else(|| v.as_str().and_then(|x| x.parse().ok())).ok_or_else(|| xparse(name, "failed to parse field [size]"))?),
            _ => return Err(unknown_field(name, k)),
        }
    }
    let missing: Vec<&str> = [("retriever", child.is_none()), ("type", kind.is_none()), ("field", field.is_none())].iter().filter(|x| x.1).map(|x| x.0).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    let kind = kind.unwrap_or_default();
    if !kind.eq_ignore_ascii_case("mmr") {
        return Err(EsError::illegal_argument(format!("unknown result diversification type [{kind}]")));
    }
    let size = size.unwrap_or_else(|| (window as i64).min(10));
    let mut errs: Vec<String> = Vec::new();
    if qv.is_some() && qvb.is_some() {
        errs.push(format!("[{name}] MMR result diversification can have one of [query_vector] or [query_vector_builder], but not both"));
    }
    if size <= 0 {
        errs.push(format!("[{name}] MMR result diversification [size] of {size} must be greater than zero"));
    }
    if size > window as i64 {
        errs.push(format!("[{name}] MMR result diversification [size] of {size} cannot be greater than the [rank_window_size] of {window}"));
    }
    match lambda {
        Some(l) if (0.0..=1.0).contains(&l) => {}
        other => errs.push(format!(
            "[{name}] MMR result diversification must have a [lambda] between 0.0 and 1.0. The value provided was {}",
            other.map(crate::ingest::jtext::java_float).unwrap_or_else(|| "null".into())
        )),
    }
    if qv.as_ref().is_some_and(Value::is_string) {
        errs.push(format!("[{name}] retriever cannot have a [query_vector] that is string encoded"));
    }
    if !errs.is_empty() {
        let r: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&r));
    }
    // `query_vector_builder` resolves (inference) while rewriting.
    let query_vector = match qvb {
        Some(b) => Some(crate::inference::semantic::resolve_knn_builder(&Map::new(), &b)?.get("query_vector").cloned().unwrap_or(Value::Null)),
        None => qv,
    };
    Ok(Retriever::Diversify { c, child: Box::new(child.unwrap_or_else(|| std_child(json!({"match_all": {}})))), field: field.unwrap_or_default(), window, size, lambda: lambda.unwrap_or(0.0), query_vector })
}

fn doc_refs(v: Option<&Value>) -> Vec<DocRef> {
    v.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|d| match d {
            Value::Object(o) => o.get("_id").map(|id| DocRef { index: o.get("_index").and_then(Value::as_str).map(str::to_string), id: crate::query::dsl::scalar_string(id) }),
            Value::String(s) => Some(DocRef { index: None, id: s.clone() }),
            _ => None,
        })
        .collect()
}

/// Query rules need the node's rulesets: resolve every `rule` retriever of
/// `retriever` into `_resolved: {pinned, excluded}` (called by the search
/// preprocessing, which has the node).
pub fn resolve_rules(node: &crate::node::Node, v: &Value) -> Result<Value> {
    match v {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, x) in m {
                if k == "rule" && x.get("ruleset_ids").is_some() {
                    let mut r = resolve_rules(node, x)?;
                    let ids: Vec<String> = match x.get("ruleset_ids") {
                        Some(Value::Array(a)) => a.iter().filter_map(|i| i.as_str().map(str::to_string)).collect(),
                        Some(Value::String(s)) => vec![s.clone()],
                        _ => Vec::new(),
                    };
                    let criteria = x.get("match_criteria").and_then(Value::as_object).cloned().unwrap_or_default();
                    let matched = crate::search_ext::matching_rules(node, &ids, &criteria)?;
                    let (mut pinned, mut excluded) = (Vec::new(), Vec::new());
                    for (_, rule) in &matched {
                        let a = &rule["actions"];
                        let refs: Vec<Value> = match a.get("ids").and_then(Value::as_array) {
                            Some(ids) => ids.iter().map(|i| json!({ "_id": i })).collect(),
                            None => a.get("docs").and_then(Value::as_array).cloned().unwrap_or_default(),
                        };
                        if rule["type"] == "exclude" {
                            excluded.extend(refs);
                        } else {
                            for p in refs {
                                if !pinned.contains(&p) {
                                    pinned.push(p);
                                }
                            }
                        }
                    }
                    r["_resolved"] = json!({ "pinned": pinned, "excluded": excluded });
                    out.insert(k.clone(), r);
                } else {
                    out.insert(k.clone(), resolve_rules(node, x)?);
                }
            }
            Ok(Value::Object(out))
        }
        Value::Array(a) => Ok(Value::Array(a.iter().map(|x| resolve_rules(node, x)).collect::<Result<_>>()?)),
        other => Ok(other.clone()),
    }
}

/// The top-level keys ES refuses next to `retriever`
/// (`SearchSourceBuilder.parseXContent`).
pub fn validate_top_level(body: &Map<String, Value>) -> Result<()> {
    // ES 9 turns `rank` into a retriever while parsing and refuses both
    // first (`SearchSourceBuilder.parseXContent`); 8.x listed it below.
    if body.contains_key("rank") && !crate::version::current_rest_api_version().is_v8() {
        return Err(EsError::illegal_argument("Cannot specify both [rank] and [retriever]."));
    }
    let mut specified = Vec::new();
    if body.contains_key("query") || body.contains_key("sub_searches") {
        specified.push("query");
    }
    for k in ["knn", "search_after", "terminate_after", "sort", "rescore", "rank"] {
        if body.contains_key(k) {
            specified.push(k);
        }
    }
    if !specified.is_empty() {
        return Err(EsError::illegal_argument(format!("cannot specify [retriever] and [{}]", specified.join(", "))));
    }
    Ok(())
}

/// Legacy `rank` / `sub_searches` → an `rrf` retriever, with
/// `SearchRequest.validate`'s checks.
fn legacy_rank(req: &SearchRequest, scroll: bool) -> Result<Retriever> {
    let rank = req.rank.as_ref().ok_or_else(|| EsError::action_validation(&["[sub_searches] requires [rank]"]))?;
    let ro = rank.as_object().ok_or_else(|| EsError::parsing("[rank] must be an object"))?;
    let (kind, spec) = ro.iter().next().ok_or_else(|| EsError::parsing("[rank] malformed, empty clause found"))?;
    if kind != "rrf" {
        return Err(EsError::parsing(format!("unknown rank [{kind}]")));
    }
    let so = spec.as_object().cloned().unwrap_or_default();
    let mut window = DEFAULT_RANK_WINDOW_SIZE;
    let mut rank_constant = DEFAULT_RANK_CONSTANT;
    for (k, v) in &so {
        match k.as_str() {
            "rank_window_size" | "window_size" => window = window_of("rrf", &so, k)?,
            "rank_constant" => {
                let rc = v.as_f64().unwrap_or(0.0);
                if rc < 1.0 {
                    return Err(EsError::illegal_argument("[rank_constant] must be greater or equal to [1] for [rrf]"));
                }
                rank_constant = rc as f32;
            }
            _ => return Err(unknown_field("rrf", k)),
        }
    }
    // ES 9 rewrites `rank` into a plain `rrf` retriever while parsing
    // (`RRFRankBuilder.toRetriever`): retriever semantics (kNN children keep
    // their top `rank_window_size` docs, retriever explanations). 8.x (and
    // `compatible-with=8`) kept the legacy rank behaviour.
    let legacy = crate::version::current_rest_api_version().is_v8();
    let mut children: Vec<(Retriever, f32)> = Vec::new();
    if req.query.is_some() && req.sub_searches.is_some() {
        return Err(EsError::illegal_argument("cannot specify both [query] and [sub_searches]"));
    }
    let named = |q: Value| {
        let mut r = std_child(q.clone());
        if let Retriever::Standard { c, .. } = &mut r {
            c.name = query_name(&q);
            c.legacy = legacy;
        }
        r
    };
    if let Some(q) = &req.query {
        children.push((named(q.clone()), 1.0));
    }
    if let Some(Value::Array(subs)) = &req.sub_searches {
        for s in subs {
            let q = s.get("query").cloned().ok_or_else(|| EsError::parsing("[sub_searches] entries require [query]"))?;
            children.push((named(q), 1.0));
        }
    }
    for k in &req.knn {
        let mut c = Common { legacy, ..Common::default() };
        let mut spec = k.as_object().cloned().unwrap_or_default();
        if let Some(f) = spec.remove("filter") {
            c.filter = filters_of(Some(&f));
        }
        c.name = spec.get("_name").and_then(Value::as_str).map(str::to_string);
        children.push((Retriever::Knn { c, spec }, 1.0));
    }
    let mut errs: Vec<String> = Vec::new();
    if req.size == 0 {
        errs.push("[rank] requires [size] greater than [0]".into());
    }
    if req.size > window {
        // ES 9 validates the converted `rrf` retriever (`CompoundRetrieverBuilder.validate`).
        let who = if crate::version::current_rest_api_version().is_v8() { "rank" } else { "rrf" };
        errs.push(format!("[{who}] requires [rank_window_size: {window}] be greater than or equal to [size: {}]", req.size));
    }
    if children.len() < 2 {
        errs.push("[rank] requires a minimum of [2] result sets using a combination of sub searches and/or knn searches".into());
    }
    if scroll {
        errs.push("[rank] cannot be used in a scroll context".into());
    }
    if req.rescore.is_some() {
        errs.push("[rank] cannot be used with [rescore]".into());
    }
    if !req.sort.is_empty() {
        errs.push("[rank] cannot be used with [sort]".into());
    }
    if req.collapse.is_some() {
        errs.push("[rank] cannot be used with [collapse]".into());
    }
    if req.suggest.is_some() {
        errs.push("[rank] cannot be used with [suggest]".into());
    }
    if !errs.is_empty() {
        let r: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&r));
    }
    Ok(Retriever::Rrf { c: Common { legacy, ..Common::default() }, children, window, rank_constant })
}

/// `QueryBuilder.queryName()` of a query (`{"term": {"f": {.., "_name": n}}}`
/// or `{"knn": {.., "_name": n}}`).
fn query_name(q: &Value) -> Option<String> {
    let (_, body) = q.as_object()?.iter().next()?;
    let b = body.as_object()?;
    if let Some(n) = b.get("_name").and_then(Value::as_str) {
        return Some(n.to_string());
    }
    if b.len() == 1 {
        return b.values().next()?.get("_name").and_then(Value::as_str).map(str::to_string);
    }
    None
}

fn std_child(q: Value) -> Retriever {
    Retriever::Standard { c: Common::default(), query: Some(q), sort: None, search_after: None, terminate_after: None, collapse: None }
}

// ---------------------------------------------------------------------------
// Validation (`CompoundRetrieverBuilder.validate`,
// `StandardRetrieverBuilder.extractToSearchSourceBuilder`).

fn validate(r: &Retriever, size: usize, scroll: bool, in_compound: bool, errs: &mut Vec<String>) -> Result<()> {
    match r {
        Retriever::Standard { search_after, terminate_after, collapse, .. } => {
            let _ = collapse;
            if in_compound {
                if search_after.is_some() {
                    return Err(EsError::illegal_argument("[search_after] cannot be used in children of compound retrievers"));
                }
                if terminate_after.is_some() {
                    return Err(EsError::illegal_argument("[terminate_after] cannot be used in children of compound retrievers"));
                }
            }
        }
        Retriever::Knn { .. } => {}
        _ => {
            let w = r.window().unwrap_or(DEFAULT_RANK_WINDOW_SIZE);
            // `rule` / `pinned` only cap the hits at their window.
            if size > w && !matches!(r, Retriever::Pinned { .. }) {
                errs.push(format!("[{}] requires [{}: {w}] be greater than or equal to [size: {size}]", r.name(), r.window_field()));
            }
            if scroll {
                errs.push(format!("cannot specify [{}] and [scroll]", r.name()));
            }
            let kids: Vec<&Retriever> = match r {
                Retriever::Rrf { children, .. } => children.iter().map(|x| &x.0).collect(),
                Retriever::Linear { children, .. } => children.iter().map(|x| &x.0).collect(),
                Retriever::Rescorer { child, .. } | Retriever::Pinned { child, .. } | Retriever::Rerank { child, .. } | Retriever::Random { child, .. } | Retriever::Diversify { child, .. } => vec![&**child],
                _ => vec![],
            };
            for k in kids {
                validate(k, size, scroll, true, errs)?;
                if let Some(cw) = k.window() {
                    if w > cw {
                        errs.push(format!(
                            "[{}] requires [{}: {w}] to be smaller than or equal to its sub retriever's {} [{}: {cw}]",
                            r.name(),
                            r.window_field(),
                            k.name(),
                            k.window_field()
                        ));
                    }
                }
            }
            if let Retriever::Pinned { kind, child, .. } = r {
                if let Retriever::Standard { sort: Some(s), .. } = &**child {
                    if !sort_starts_with_score(s) {
                        return Err(EsError::illegal_argument(if *kind == "rule" { "[rule] retriever only supports sort by score" } else { "[pinned] retriever only supports sorting by score" }));
                    }
                }
            }
        }
    }
    Ok(())
}

fn sort_starts_with_score(s: &Value) -> bool {
    let first = match s {
        Value::Array(a) => match a.first() {
            Some(x) => x,
            None => return true,
        },
        other => other,
    };
    match first {
        Value::String(f) => f == "_score",
        Value::Object(o) => o.keys().next().map(|k| k == "_score").unwrap_or(true),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Execution.

/// One ranked document of a retriever's result list.
#[derive(Clone, Debug)]
struct RankDoc {
    ti: usize,
    index: String,
    id: String,
    score: f32,
    /// `(leaf, doc)` of the search that produced it (tie-break order).
    leaf: usize,
    doc: DocId,
    /// Set by a `pinned` retriever: whether the doc was pinned (ES
    /// `PinnedRankDoc.isPinned`); `None` for other retrievers.
    pinned: Option<bool>,
    /// How the doc got its score (requests with `explain`).
    expl: Option<Explanation>,
}

impl RankDoc {
    fn key(&self) -> (usize, String) {
        (self.ti, self.id.clone())
    }
}

/// A retriever's result: its top docs (best first) and a query matching
/// every doc it "matches" (total hits / aggregations).
struct RankList {
    docs: Vec<RankDoc>,
    union: Value,
}

/// One rank doc of a [`RankPlan`]: `(_id, score, rank (1-based), pinned,
/// explanation)`.
type PlanDoc = (String, f32, usize, Option<bool>, Option<Explanation>);

/// The rank docs of a compound retriever, handed to the final search.
#[derive(Debug)]
pub struct RankPlan {
    /// index name → (`_id`, score, rank (1-based), pinned, explanation).
    docs: HashMap<String, Vec<PlanDoc>>,
    /// Union of the children's queries (aggregations, total hits).
    union: Value,
    /// (index, `_id`) → position in the fused ranking (ES sorts the final
    /// hits by rank: `RankDocsSortBuilder`, so score ties keep fused order).
    order: HashMap<(String, String), usize>,
}

/// `SearchRequest.rank_plan` holder (Clone + Debug for the request).
pub type SharedPlan = Arc<RankPlan>;

struct Ctx<'a, 'b> {
    targets: &'a [Target],
    opts: &'a ExecOpts<'b>,
    base: &'a SearchRequest,
    /// Children run here instead of on `targets` (see [`plan`]).
    remote: Option<&'a dyn ChildSearch>,
}

/// Runs the child searches of a compound retriever somewhere else: a
/// cluster coordinator runs each one over every shard of the targets.
/// Search body → search response.
pub trait ChildSearch {
    fn search(&self, body: Value) -> Result<Value>;
    /// Several child searches, answered in order; by default one after
    /// another. A cluster coordinator runs them at once (ES runs a compound
    /// retriever's child searches as one multi-search).
    fn search_all(&self, bodies: Vec<Value>) -> Vec<Result<Value>> {
        bodies.into_iter().map(|b| self.search(b)).collect()
    }
}

/// A leaf child (`standard`, `knn`) of a compound retriever, ready to run:
/// its search request, its raw sort (a `standard` child's) and what its
/// rank list keeps of the answer.
enum Leaf {
    Standard { req: SearchRequest, sort: Option<Value>, union: Value },
    Knn { req: SearchRequest, window: usize, legacy: bool },
}

/// [`Leaf`] of `r` under `inherited` filters, or `None` for a compound `r`.
fn leaf_of(r: &Retriever, cx: &Ctx, inherited: &[Value], window: usize) -> Option<Result<Leaf>> {
    let mut filters: Vec<Value> = inherited.to_vec();
    filters.extend(r.common().filter.iter().cloned());
    let min_score = r.common().min_score;
    match r {
        Retriever::Standard { query, sort, collapse, .. } => Some((|| {
            let q = bool_filter(query.clone(), &filters);
            let mut req = child_request(cx.base, window);
            req.query = Some(q.clone());
            req.min_score = min_score;
            if let Some(col) = collapse {
                req.collapse = col.get("field").and_then(Value::as_str).map(str::to_string);
            }
            if let Some(s) = sort {
                let tmp = search::parse_request(&json!({ "sort": s }), &HashMap::new())?;
                req.sort = tmp.sort;
                req.sort_extra = tmp.sort_extra;
            }
            req.explain = cx.base.explain;
            Ok(Leaf::Standard { req, sort: sort.clone(), union: q })
        })()),
        Retriever::Knn { spec, .. } => {
            let s = knn_spec_with(spec, &filters, window);
            // A retriever child keeps its top `rank_window_size` hits; the
            // legacy `rank` form matches every one of the `k` winners
            // (fusion still reads the first `rank_window_size`).
            let legacy = r.common().legacy;
            let k = if legacy { s.get("k").and_then(Value::as_u64).map(|k| k as usize).unwrap_or(window).max(window) } else { window };
            let mut req = child_request(cx.base, k);
            req.knn = vec![Value::Object(s)];
            req.min_score = min_score;
            Some(Ok(Leaf::Knn { req, window, legacy }))
        }
        _ => None,
    }
}

/// A leaf's rank list from its docs.
fn leaf_list(cx: &Ctx, leaf: Leaf, mut docs: Vec<RankDoc>) -> RankList {
    match leaf {
        Leaf::Standard { union, .. } => RankList { docs, union },
        Leaf::Knn { window, legacy, .. } => {
            let union = ids_union(&docs);
            docs.truncate(window);
            if cx.base.explain {
                for (i, d) in docs.iter_mut().enumerate() {
                    d.expl = Some(if legacy { Explanation::leaf(d.score, "within top k documents") } else { knn_explain(d, i + 1) });
                }
            }
            // `KnnRetrieverBuilder.topDocsQuery`: its rank docs (the kNN
            // winners), not a re-run of the (per-shard) kNN query.
            RankList { docs, union }
        }
    }
}

/// The rank lists of a compound retriever's children when they are all
/// leaves run remotely: their searches sent at once ([`ChildSearch::search_all`]).
/// `None` otherwise (each child is evaluated in turn).
fn remote_leaves(children: &[&Retriever], cx: &Ctx, inherited: &[Value], window: usize) -> Option<Vec<Result<RankList>>> {
    let remote = cx.remote?;
    if children.len() < 2 {
        return None;
    }
    let mut leaves = Vec::with_capacity(children.len());
    for c in children {
        leaves.push(leaf_of(c, cx, inherited, window)?);
    }
    // A leaf that cannot be built: evaluated in turn, which fails as the
    // first failing child does.
    let ready = leaves.into_iter().collect::<Result<Vec<Leaf>>>().ok()?;
    let bodies = ready.iter().map(|l| match l {
        Leaf::Standard { req, sort, .. } => child_body(req, sort.as_ref()),
        Leaf::Knn { req, .. } => child_body(req, None),
    }).collect();
    let answers = remote.search_all(bodies);
    Some(ready.into_iter().zip(answers).map(|(l, a)| a.map(|v| leaf_list(cx, l, remote_hits(cx, &v)))).collect())
}

/// A child search request as a `_search` body (what [`child_request`] and
/// the retriever left set); `sort` is the raw sort of a `standard` child.
fn child_body(req: &SearchRequest, sort: Option<&Value>) -> Value {
    let mut b = Map::new();
    b.insert("size".into(), json!(req.size));
    if let Some(q) = &req.query {
        b.insert("query".into(), q.clone());
    }
    if req.knn.len() == 1 && req.query.is_none() {
        // A lone kNN child (a `knn` retriever): one clause, as an object,
        // runs as the cluster's lone kNN search (each shard's top `k`,
        // merged and cut to the global `k`). As an array it took the DFS
        // kNN route (a lone kNN search for the global top `k`, a term-stats
        // round with no text to score, then a query round over those
        // docs): the same hits, at three fan-outs instead of one.
        b.insert("knn".into(), req.knn[0].clone());
    } else if !req.knn.is_empty() {
        b.insert("knn".into(), Value::Array(req.knn.clone()));
    }
    if let Some(m) = req.min_score {
        b.insert("min_score".into(), json!(m));
    }
    if let Some(c) = &req.collapse {
        b.insert("collapse".into(), json!({ "field": c }));
    }
    if let Some(s) = sort {
        b.insert("sort".into(), s.clone());
    }
    if req.explain {
        b.insert("explain".into(), json!(true));
    }
    if let Some(r) = &req.rescore {
        b.insert("rescore".into(), r.clone());
    }
    if let Some(p) = &req.rank_plan {
        b.insert(RANK_PLAN_KEY.into(), p.to_json());
    }
    b.insert(
        "_source".into(),
        match &req.source {
            SourceFilter::All => json!(true),
            SourceFilter::None => json!(false),
            SourceFilter::Filter { includes, excludes } => json!({ "includes": includes, "excludes": excludes }),
        },
    );
    if !req.fields.is_empty() {
        let f: Vec<Value> = req.fields.iter().map(|(n, fmt)| match fmt {
            Some(fmt) => json!({ "field": n, "format": fmt }),
            None => json!(n),
        }).collect();
        b.insert("fields".into(), Value::Array(f));
    }
    if let Some(rm) = &req.runtime_mappings {
        b.insert("runtime_mappings".into(), rm.clone());
    }
    b.insert("track_scores".into(), json!(true));
    b.insert("track_total_hits".into(), json!(false));
    Value::Object(b)
}

/// The rank docs of a remote child search response (hit order is the
/// tie-break order).
fn remote_hits(cx: &Ctx, v: &Value) -> Vec<RankDoc> {
    let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            let index = h["_index"].as_str().unwrap_or_default().to_string();
            let ti = cx.targets.iter().position(|t| t.svc.name == index).unwrap_or(0);
            let score = h["_score"].as_f64().map(|x| x as f32).unwrap_or(f32::NAN);
            let expl = h.get("_explanation").map(expl_from_json);
            RankDoc { ti, index, id: h["_id"].as_str().unwrap_or_default().to_string(), score, leaf: 0, doc: i as DocId, pinned: None, expl }
        })
        .collect()
}

/// A child search response: on the targets, or remotely.
fn exec_json(cx: &Ctx, targets: &[Target], req: &SearchRequest) -> Result<Value> {
    match cx.remote {
        Some(r) => r.search(child_body(req, None)),
        None => {
            let out = search::execute(targets, req, &child_opts(cx.opts))?;
            Ok(serde_json::from_str(&out.json)?)
        }
    }
}

fn bool_filter(q: Option<Value>, filters: &[Value]) -> Value {
    if filters.is_empty() {
        return q.unwrap_or_else(|| json!({ "match_all": {} }));
    }
    let mut b = Map::new();
    b.insert("filter".into(), Value::Array(filters.to_vec()));
    if let Some(q) = q {
        b.insert("must".into(), Value::Array(vec![q]));
    }
    json!({ "bool": b })
}

fn knn_spec_with(spec: &Map<String, Value>, filters: &[Value], default_k: usize) -> Map<String, Value> {
    let mut s = spec.clone();
    if !filters.is_empty() {
        s.insert("filter".into(), Value::Array(filters.to_vec()));
    }
    if !s.contains_key("k") {
        s.insert("k".into(), json!(default_k));
    }
    s
}

/// A child search: the base request (runtime mappings etc.) stripped down
/// to identifying its top `size` hits.
fn child_request(base: &SearchRequest, size: usize) -> SearchRequest {
    let mut r = base.clone();
    r.retriever = None;
    r.rank = None;
    r.sub_searches = None;
    r.rank_plan = None;
    r.query = None;
    r.knn = Vec::new();
    r.post_filter = None;
    r.from = 0;
    r.size = size;
    r.sort = Vec::new();
    r.sort_extra = HashMap::new();
    r.search_after = None;
    r.source = SourceFilter::None;
    r.fields = Vec::new();
    r.docvalue_fields = Vec::new();
    r.highlight = None;
    r.explain = false;
    r.version = false;
    r.seq_no_primary_term = false;
    r.track_total_hits = None;
    r.track_scores = true;
    r.min_score = None;
    r.terminate_after = None;
    r.aggs = None;
    r.collapse = None;
    r.collapse_spec = None;
    r.rescore = None;
    r.profile = false;
    // `_id` is read back from the child's hits, so it must not be suppressed.
    r.stored_fields_none = false;
    r.script_fields = None;
    r.suggest = None;
    r
}

fn child_opts<'a>(opts: &ExecOpts<'a>) -> ExecOpts<'a> {
    ExecOpts { terms_lookup: opts.terms_lookup, cursors: None, max_buckets: opts.max_buckets, scroll_id: None, pit_id: None }
}

/// Read `(index, id, score)` back from a search response plus the hit keys.
fn read_hits(cx: &Ctx, out: &SearchOutput) -> Result<Vec<RankDoc>> {
    let v: Value = serde_json::from_str(&out.json)?;
    let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let mut docs = Vec::with_capacity(hits.len());
    for (i, h) in hits.iter().enumerate() {
        let (ti, leaf, doc, score) = out.page.get(i).copied().unwrap_or((0, 0, 0, 0.0));
        let index = h["_index"].as_str().unwrap_or_default().to_string();
        let ti = cx.targets.iter().position(|t| t.svc.name == index).unwrap_or(ti);
        let expl = h.get("_explanation").map(expl_from_json);
        docs.push(RankDoc { ti, index, id: h["_id"].as_str().unwrap_or_default().to_string(), score, leaf, doc, pinned: None, expl });
    }
    Ok(docs)
}

/// An `_explanation` of a hit back into an [`Explanation`].
fn expl_from_json(v: &Value) -> Explanation {
    Explanation::new(
        v["value"].as_f64().unwrap_or(0.0) as f32,
        v["description"].as_str().unwrap_or_default(),
        v["details"].as_array().map(|a| a.iter().map(expl_from_json).collect()).unwrap_or_default(),
    )
}

/// `" [name]"` of a named child retriever (`queryNames[i]` in ES rank docs).
fn alias(r: &Retriever) -> String {
    r.common().name.as_ref().map(|n| format!(" [{n}]")).unwrap_or_default()
}

/// `Arrays.toString(float[])`.
fn floats_str(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| crate::ingest::jtext::java_float(*x)).collect::<Vec<_>>().join(", "))
}

/// `RRFRankDoc.explain`: `ranks[i]` is the 1-based rank of the doc in child
/// `i` (0 = not found) with that child's explanation of it.
fn rrf_explain(score: f32, rank_constant: f32, children: &[(String, usize, Option<Explanation>)], legacy: bool) -> Explanation {
    let k = rank_constant as i64;
    let mut details = Vec::with_capacity(children.len());
    for (i, (alias, rank, src)) in children.iter().enumerate() {
        // `RRFRankBuilder.explainHit` (legacy `rank`) names a query by its
        // `_name` alone.
        let qid = if !legacy {
            format!("at index [{i}]{alias}")
        } else if alias.is_empty() {
            format!("at index [{i}]")
        } else {
            alias.trim_start().to_string()
        };
        if *rank == 0 {
            details.push(Explanation::leaf(0.0, format!("rrf score: [0], result not found in query {qid}")));
        } else {
            let s = crate::ingest::jtext::java_float(1f32 / (*rank as i64 + k) as f32);
            let desc = if legacy {
                format!("rrf score: [{s}], for rank [{rank}] in query {qid} computed as [1 / ({rank} + {k}]), for matching query with score: ")
            } else {
                format!("rrf score: [{s}], for rank [{rank}] in query {qid} computed as [1 / ({rank} + {k})], for matching query with score")
            };
            details.push(Explanation::new(*rank as f32, desc, src.iter().cloned().collect()));
        }
    }
    let ranks: Vec<String> = children.iter().map(|c| c.1.to_string()).collect();
    Explanation::new(
        score,
        format!("rrf score: [{}] computed for initial ranks [{}] with rankConstant: [{k}] as sum of [1 / (rank + rankConstant)] for each query", crate::ingest::jtext::java_float(score), ranks.join(", ")),
        details,
    )
}

/// `RankDoc.explain` of a kNN child: the exact-kNN explanation under the
/// doc's rank in the child's top docs.
fn knn_explain(d: &RankDoc, rank: usize) -> Explanation {
    let sc = crate::ingest::jtext::java_float(d.score);
    Explanation::new(
        rank as f32,
        format!("doc [{}] with an original score of [{sc}] is at rank [{rank}] from the following source queries.", d.doc),
        vec![Explanation::leaf(d.score, format!("found vector with calculated similarity: {sc}"))],
    )
}

fn run_child(cx: &Ctx, req: &SearchRequest) -> Result<Vec<RankDoc>> {
    run_child_sorted(cx, req, None)
}

fn run_child_sorted(cx: &Ctx, req: &SearchRequest, sort: Option<&Value>) -> Result<Vec<RankDoc>> {
    if let Some(r) = cx.remote {
        let v = r.search(child_body(req, sort))?;
        return Ok(remote_hits(cx, &v));
    }
    let out = search::execute(cx.targets, req, &child_opts(cx.opts))?;
    read_hits(cx, &out)
}

fn cmp_ranked(a: &RankDoc, b: &RankDoc) -> std::cmp::Ordering {
    b.score.total_cmp(&a.score).then(a.ti.cmp(&b.ti)).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)).then(a.id.cmp(&b.id))
}

fn union_of(parts: Vec<Value>) -> Value {
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap_or(Value::Null);
    }
    json!({ "bool": { "should": parts, "minimum_should_match": 1 } })
}

fn eval(r: &Retriever, cx: &Ctx, inherited: &[Value], window: usize) -> Result<RankList> {
    let mut filters: Vec<Value> = inherited.to_vec();
    filters.extend(r.common().filter.iter().cloned());
    let min_score = r.common().min_score;
    match r {
        Retriever::Standard { .. } | Retriever::Knn { .. } => {
            let leaf = leaf_of(r, cx, inherited, window).unwrap_or_else(|| Err(EsError::internal("not a leaf retriever")))?;
            let docs = match &leaf {
                Leaf::Standard { req, sort, .. } => run_child_sorted(cx, req, sort.as_ref())?,
                Leaf::Knn { req, .. } => run_child(cx, req)?,
            };
            Ok(leaf_list(cx, leaf, docs))
        }
        Retriever::Rrf { children, window: w, rank_constant, .. } => {
            let mut lists = Vec::with_capacity(children.len());
            let mut unions = Vec::with_capacity(children.len());
            let mut by_key: HashMap<(usize, String), RankDoc> = HashMap::new();
            let mut weights = Vec::with_capacity(children.len());
            let mut ranked: Vec<HashMap<(usize, String), (usize, Option<Explanation>)>> = Vec::new();
            let kids: Vec<&Retriever> = children.iter().map(|(c, _)| c).collect();
            let mut pre = remote_leaves(&kids, cx, &filters, *w).map(Vec::into_iter);
            for (child, weight) in children {
                let l = match pre.as_mut().and_then(Iterator::next) { Some(l) => l, None => eval(child, cx, &filters, *w) }.map_err(|e| child_failure("rrf", child.name(), e))?;
                unions.push(l.union);
                let list: Vec<((usize, String), f32)> = l.docs.iter().map(|d| (d.key(), d.score)).collect();
                if cx.base.explain {
                    ranked.push(l.docs.iter().take(*w).enumerate().map(|(i, d)| (d.key(), (i + 1, d.expl.clone()))).collect());
                }
                for d in l.docs {
                    by_key.entry(d.key()).or_insert(d);
                }
                lists.push(list);
                weights.push(*weight);
            }
            let fused: Vec<((usize, String), f32)> = if weights.iter().all(|w| *w == 1.0) {
                fusion::rrf(&lists, *rank_constant, *w)
            } else {
                let mut acc: HashMap<(usize, String), f32> = HashMap::new();
                for (list, wt) in lists.iter().zip(&weights) {
                    for (rank, (k, _)) in list.iter().take(*w).enumerate() {
                        *acc.entry(k.clone()).or_insert(0.0) += wt / (rank_constant + rank as f32 + 1.0);
                    }
                }
                acc.into_iter().collect()
            };
            let mut docs = finish(fused, &by_key, min_score, usize::MAX);
            if r.common().legacy {
                // `RRFQueryPhaseRankCoordinatorContext`: ties go to the doc
                // with the larger score in the first query that tells them
                // apart (a doc a query found beats one it did not).
                let per: Vec<HashMap<&(usize, String), f32>> = lists.iter().map(|l| l.iter().take(*w).map(|(k, s)| (k, *s)).collect()).collect();
                docs.sort_by(|a, b| {
                    b.score.total_cmp(&a.score).then_with(|| {
                        let (ka, kb) = (a.key(), b.key());
                        for m in &per {
                            match (m.get(&ka), m.get(&kb)) {
                                (Some(x), Some(y)) if x != y => return y.total_cmp(x),
                                (Some(_), None) => return std::cmp::Ordering::Less,
                                (None, Some(_)) => return std::cmp::Ordering::Greater,
                                _ => {}
                            }
                        }
                        cmp_ranked(a, b)
                    })
                });
            }
            docs.truncate(*w);
            if cx.base.explain {
                for d in docs.iter_mut() {
                    let per: Vec<(String, usize, Option<Explanation>)> = children
                        .iter()
                        .zip(&ranked)
                        .map(|((c, _), m)| match m.get(&d.key()) {
                            Some((r, e)) => (alias(c), *r, e.clone()),
                            None => (alias(c), 0, None),
                        })
                        .collect();
                    d.expl = Some(rrf_explain(d.score, *rank_constant, &per, r.common().legacy));
                }
            }
            let union = if min_score.is_some() { ids_union(&docs) } else { union_of(unions) };
            Ok(RankList { docs, union })
        }
        Retriever::Linear { children, window: w, .. } => {
            let mut acc: HashMap<(usize, String), f32> = HashMap::new();
            let mut unions = Vec::with_capacity(children.len());
            let mut by_key: HashMap<(usize, String), RankDoc> = HashMap::new();
            let mut normed: Vec<HashMap<(usize, String), (f32, Option<Explanation>)>> = Vec::new();
            let kids: Vec<&Retriever> = children.iter().map(|(c, _, _)| c).collect();
            let mut pre = remote_leaves(&kids, cx, &filters, *w).map(Vec::into_iter);
            for (child, weight, norm) in children {
                let l = match pre.as_mut().and_then(Iterator::next) { Some(l) => l, None => eval(child, cx, &filters, *w) }.map_err(|e| child_failure("linear", child.name(), e))?;
                unions.push(l.union);
                let list: Vec<((usize, String), f32)> = l.docs.iter().take(*w).map(|d| (d.key(), if d.score.is_finite() { d.score } else { 0.0 })).collect();
                let mut m = HashMap::new();
                for (k, s) in fusion::linear(std::slice::from_ref(&list), &[1.0], norm.core()) {
                    *acc.entry(k.clone()).or_insert(0.0) += *weight * s;
                    if cx.base.explain {
                        let e = l.docs.iter().find(|d| d.key() == k).and_then(|d| d.expl.clone());
                        m.insert(k, (s, e));
                    }
                }
                normed.push(m);
                for d in l.docs {
                    by_key.entry(d.key()).or_insert(d);
                }
            }
            let mut docs = finish(acc.into_iter().collect(), &by_key, min_score, *w);
            if cx.base.explain {
                for d in docs.iter_mut() {
                    let mut details = Vec::new();
                    let mut scores = Vec::new();
                    let mut ws = Vec::new();
                    for (i, ((c, wt, norm), m)) in children.iter().zip(&normed).enumerate() {
                        let qid = format!("at index [{i}]{}", alias(c));
                        let (ns, e) = m.get(&d.key()).cloned().unwrap_or((0.0, None));
                        scores.push(ns);
                        ws.push(*wt);
                        if ns > 0.0 {
                            let jf = crate::ingest::jtext::java_float;
                            details.push(Explanation::new(
                                wt * ns,
                                format!("weighted score: [{}] in query {qid} computed as [{} * {}] using score normalizer [{}] for original matching query with score:", jf(wt * ns), jf(*wt), jf(ns), norm.name()),
                                e.into_iter().collect(),
                            ));
                        } else {
                            details.push(Explanation::leaf(0.0, format!("weighted score: [0], result not found in query {qid}")));
                        }
                    }
                    d.expl = Some(Explanation::new(
                        d.score,
                        format!(
                            "weighted linear combination score: [{}] computed for normalized scores {} and weights {} as sum of (weight[i] * score[i]) for each query.",
                            crate::ingest::jtext::java_float(d.score),
                            floats_str(&scores),
                            floats_str(&ws)
                        ),
                        details,
                    ));
                }
            }
            let union = if min_score.is_some() { ids_union(&docs) } else { union_of(unions) };
            Ok(RankList { docs, union })
        }
        Retriever::Rescorer { child, rescore, window: w, .. } => {
            let l = eval(child, cx, &filters, *w)?;
            let union = l.union.clone();
            let plan = plan_of(&l);
            let mut req = child_request(cx.base, *w);
            req.rank_plan = Some(Arc::new(plan));
            req.rescore = Some(rescore.clone());
            req.min_score = min_score;
            let docs = run_child(cx, &req)?;
            Ok(RankList { docs, union })
        }
        Retriever::Rerank { child, field, inference_id, inference_text, window: w, allow_failures, chunk, .. } => {
            let l = eval(child, cx, &filters, *w)?;
            let union = l.union.clone();
            let mut docs: Vec<RankDoc> = l.docs.into_iter().take(*w).collect();
            if docs.is_empty() {
                return Ok(RankList { docs, union });
            }
            // The rank feature phase fetches the field (`hit.field(field).getValue()`).
            let ids: Vec<&str> = docs.iter().map(|d| d.id.as_str()).collect();
            let mut req = child_request(cx.base, docs.len() * cx.targets.len().max(1));
            req.query = Some(json!({ "ids": { "values": ids } }));
            req.source = SourceFilter::Filter { includes: vec![field.clone()], excludes: vec![] };
            req.fields = vec![(field.clone(), None)];
            let v = exec_json(cx, cx.targets, &req)?;
            let mut texts: HashMap<(String, String), String> = HashMap::new();
            for h in v["hits"]["hits"].as_array().into_iter().flatten() {
                let from_fields = h["fields"][field.as_str()].as_array().and_then(|a| a.first()).cloned();
                let value = from_fields.or_else(|| {
                    let mut cur = &h["_source"];
                    for part in field.split('.') {
                        cur = &cur[part];
                    }
                    match cur {
                        Value::Array(a) => a.first().cloned(),
                        Value::Null => None,
                        other => Some(other.clone()),
                    }
                });
                let Some(value) = value else { continue };
                let t = match value {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                texts.insert((h["_index"].as_str().unwrap_or_default().to_string(), h["_id"].as_str().unwrap_or_default().to_string()), t);
            }
            let svc = crate::inference::current();
            let ep = svc.as_ref().and_then(|s| s.get_endpoint(inference_id));
            // Per doc: the texts sent to the reranker (`featureData`).
            let mut feature_docs: Vec<(usize, Vec<String>)> = Vec::new();
            for (i, d) in docs.iter().enumerate() {
                let Some(t) = texts.get(&(d.index.clone(), d.id.clone())) else { continue };
                let feats = match chunk {
                    None => vec![t.clone()],
                    Some((size, settings)) => {
                        let size = size.unwrap_or(1).max(1) as usize;
                        let settings = settings.clone().unwrap_or_else(|| json!({"strategy": "sentence", "max_chunk_size": reranker_window_size(ep.as_ref()), "sentence_overlap": 0}));
                        let chunking = crate::inference::local::Chunking::from_json(Some(&settings));
                        let chunks: Vec<String> = crate::inference::local::chunk(t, &chunking).into_iter().map(|(a, b)| t[a..b].to_string()).collect();
                        best_chunks(&chunks, inference_text, size)
                    }
                };
                feature_docs.push((i, feats));
            }
            let failure = |e: EsError| -> EsError {
                let mut spe = EsError::new(e.status, "search_phase_execution_exception", "Computing updated ranks for results failed").with("phase", "rank-feature").with("grouped", true).with("failed_shards", json!([]));
                spe = spe.caused_by(e);
                child_failure("text_similarity_reranker", child.name(), spe)
            };
            let ranked: Result<Vec<(usize, f32)>> = (|| {
                // `TopNProvider`: a reported `top_n` below the window cannot rerank it.
                if let Some(ep) = &ep {
                    let hidden = ep.task_settings.get("hide_top_n").is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"));
                    if let Some(n) = ep.task_settings.get("top_n").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).filter(|_| !hidden) {
                        if (n as usize) < *w {
                            return Err(failure(EsError::illegal_argument(format!(
                                "Inference endpoint [{inference_id}] is configured to return the top [{n}] results, but rank_window_size is [{w}]. Reduce rank_window_size to be less than or equal to the configured top N value."
                            ))));
                        }
                    }
                }
                let inputs: Vec<String> = feature_docs.iter().flat_map(|(_, f)| f.iter().cloned()).collect();
                let owner: Vec<usize> = feature_docs.iter().enumerate().flat_map(|(k, (_, f))| std::iter::repeat_n(k, f.len())).collect();
                if inputs.is_empty() {
                    return Ok(Vec::new());
                }
                let ranked = svc
                    .as_ref()
                    .ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{inference_id}]")))
                    .and_then(|svc| svc.rerank(inference_id, inference_text, &inputs))?;
                if ranked.len() != inputs.len() {
                    return Err(failure(EsError::new(500, "illegal_state_exception", format!(
                        "Expected ranked doc size to be {}, got {}. Is the reranker service using an unreported top N task setting?",
                        inputs.len(),
                        ranked.len()
                    ))));
                }
                // A doc's score is its best feature's (`extractScoresFromRankedDocs`).
                let mut best: Vec<Option<f32>> = vec![None; feature_docs.len()];
                for (idx, sc) in ranked {
                    if let Some(&k) = owner.get(idx) {
                        let sc = sc as f32;
                        best[k] = Some(best[k].map_or(sc, |b| b.max(sc)));
                    }
                }
                Ok(feature_docs.iter().zip(best).filter_map(|((i, _), b)| b.map(|b| (*i, b))).collect())
            })();
            let ranked = match ranked {
                Ok(r) => r,
                // `allow_rerank_failures`: keep the child's ranking.
                Err(_) if *allow_failures => {
                    let union = if min_score.is_some() { ids_union(&docs) } else { union };
                    return Ok(RankList { docs, union });
                }
                Err(e) => return Err(e),
            };
            let mut scored: Vec<RankDoc> = Vec::with_capacity(ranked.len());
            for (i, sc) in ranked {
                // `preprocess`: `min_score` on the raw score, then
                // `max(score, 0) + min(exp(score), 1)`.
                if min_score.is_some_and(|m| sc < m) {
                    continue;
                }
                if let Some(d) = docs.get(i) {
                    let mut d = d.clone();
                    d.score = sc.max(0.0) + (sc.exp()).min(1.0);
                    if cx.base.explain {
                        let a = child.common().name.as_ref().map(|n| format!("[{n}]")).unwrap_or_default();
                        let chunks = match chunk {
                            Some((size, _)) => format!("and rescoring considering only top [{}] best chunks", size.unwrap_or(1)),
                            None => String::new(),
                        };
                        d.expl = Some(Explanation::new(
                            d.score,
                            format!("text_similarity_reranker match using inference endpoint: [{inference_id}] on document field: [{field}] matching on source query {a}{chunks}"),
                            d.expl.take().into_iter().collect(),
                        ));
                    }
                    scored.push(d);
                }
            }
            docs.clear();
            scored.sort_by(cmp_ranked);
            scored.truncate(*w);
            let union = if min_score.is_some() { ids_union(&scored) } else { union };
            Ok(RankList { docs: scored, union })
        }
        Retriever::Random { child, window: w, seed, .. } => {
            let l = eval(child, cx, &filters, *w)?;
            let union = l.union.clone();
            let mut docs: Vec<RankDoc> = l.docs.into_iter().take(*w).collect();
            for d in docs.iter_mut() {
                // `RandomRankFeaturePhaseRankCoordinatorContext`: Random(seed + doc).nextFloat().
                d.score = random_rerank_score(*seed, d.doc as i32);
            }
            docs.sort_by(cmp_ranked);
            if let Some(m) = min_score {
                docs.retain(|d| d.score >= m);
            }
            Ok(RankList { docs, union })
        }
        Retriever::Diversify { child, field, window: w, size, lambda, query_vector, .. } => {
            let l = eval(child, cx, &filters, *w)?;
            let docs: Vec<RankDoc> = l.docs.into_iter().take(*w).collect();
            let qv = query_vector.as_ref().and_then(vector_of);
            // `combineInnerRetrieverResults` failures fail the compound retriever.
            let vectors = diversify_vectors(cx, &docs, field, qv.as_ref()).map_err(|e| {
                let status = e.status;
                EsError::new(status, "status_exception", "[diversify] search failed - retrievers '[diversify]' returned errors. All failures are attached as suppressed exceptions.").with("suppressed", json!([e.error_object(false)]))
            })?;
            let keep = mmr_select(&vectors, qv.as_ref(), *lambda, *size as usize);
            let mut out: Vec<RankDoc> = docs.into_iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, d)| d).collect();
            if let Some(m) = min_score {
                out.retain(|d| d.score >= m);
            }
            Ok(RankList { docs: out, union: l.union })
        }
        Retriever::MultiField { kind, .. } => Err(EsError::illegal_argument(format!("[{kind}] the multi-field query format was not rewritten"))),
        Retriever::Pinned { child, pinned, excluded, window: w, kind, rule_info, .. } => {
            let l = eval(child, cx, &filters, *w)?;
            let explain_pins = *kind == "pinned";
            let is_excluded = |d: &RankDoc| excluded.iter().any(|e| e.id == d.id && e.index.as_ref().map(|i| *i == d.index).unwrap_or(true));
            // Which pinned docs exist (and pass the filters)?
            let mut out: Vec<RankDoc> = Vec::new();
            let mut pin_query = Vec::new();
            if !pinned.is_empty() {
                let ids: Vec<&str> = pinned.iter().map(|p| p.id.as_str()).collect();
                let q = bool_filter(Some(json!({ "ids": { "values": ids } })), &filters);
                pin_query.push(q.clone());
                let mut req = child_request(cx.base, pinned.len().max(1) * cx.targets.len().max(1));
                req.query = Some(q);
                let found = run_child(cx, &req)?;
                let n = pinned.len();
                // ES pinned scores: just above `Float.MAX_VALUE / 2`, best first.
                let base_bits = 0x7F00_0001u32;
                for (i, p) in pinned.iter().enumerate() {
                    for d in found.iter().filter(|d| d.id == p.id && p.index.as_ref().map(|ix| *ix == d.index).unwrap_or(true)) {
                        if is_excluded(d) || out.iter().any(|o| o.key() == d.key()) {
                            continue;
                        }
                        let mut d = d.clone();
                        d.score = f32::from_bits(base_bits + (n - i) as u32);
                        d.pinned = explain_pins.then_some(true);
                        out.push(d);
                    }
                }
            }
            for mut d in l.docs {
                if is_excluded(&d) || out.iter().any(|o| o.key() == d.key()) {
                    continue;
                }
                if explain_pins {
                    d.pinned = Some(false);
                }
                out.push(d);
            }
            if let Some(m) = min_score {
                out.retain(|d| d.score >= m);
            }
            out.truncate(*w);
            let mut parts = pin_query;
            parts.push(l.union);
            let mut union = union_of(parts);
            if !excluded.is_empty() {
                let ex: Vec<Value> = excluded
                    .iter()
                    .map(|e| match &e.index {
                        Some(ix) => json!({ "bool": { "filter": [{ "ids": { "values": [e.id] } }, { "term": { "_index": ix } }] } }),
                        None => json!({ "ids": { "values": [e.id] } }),
                    })
                    .collect();
                union = json!({ "bool": { "must": [union], "must_not": ex } });
            }
            // `explain` of a `rule` retriever (`QueryRuleRetrieverBuilderWrapper.explainQuery`):
            // `RuleQueryRankDoc` over the child's `RankDoc` over the explanation
            // of the child query as the rule query rewrites it.
            if let (true, Some((rulesets, criteria)), Retriever::Standard { query, .. }) = (cx.base.explain, rule_info, &**child) {
                let q = rule_rewritten_query(query.clone(), pinned, excluded);
                let mut req = child_request(cx.base, *w);
                req.query = Some(bool_filter(Some(q), &filters));
                req.explain = true;
                let explained: HashMap<(usize, String), Explanation> = run_child(cx, &req)?.into_iter().filter_map(|d| d.expl.clone().map(|e| (d.key(), e))).collect();
                let desc = format!("query rules evaluated rules from rulesets [{}] and match criteria {}", rulesets.join(", "), java_string(criteria));
                for (i, d) in out.iter_mut().enumerate() {
                    let gdoc = cx.targets.get(d.ti).map(|t| search::global_doc(t, d.leaf, d.doc)).unwrap_or(d.doc as u64);
                    let rank_doc = Explanation::new(
                        (i + 1) as f32,
                        format!("doc [{gdoc}] with an original score of [{}] is at rank [{}] from the following source queries.", crate::ingest::jtext::java_float(d.score), i + 1),
                        explained.get(&d.key()).cloned().into_iter().collect(),
                    );
                    d.expl = Some(Explanation::new(d.score, desc.clone(), vec![rank_doc]));
                }
            }
            Ok(RankList { docs: out, union })
        }
    }
}

/// `RuleQueryBuilder.doRewrite`: the organic query pinned / excluded as the
/// matched rules say.
/// A vector as MMR compares it (`VectorData`: float or byte).
#[derive(Clone, Debug)]
enum Vector {
    Float(Vec<f32>),
    Byte(Vec<i8>),
}

/// `VectorData.parseXContent`-like reading of a fetched / supplied vector:
/// numbers (floats) or a hex string (bytes).
fn vector_of(v: &Value) -> Option<Vector> {
    match v {
        Value::Array(a) if !a.is_empty() => a.iter().map(|x| x.as_f64().map(|f| f as f32)).collect::<Option<Vec<f32>>>().map(Vector::Float),
        Value::String(h) if h.len() % 2 == 0 => (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|b| b as i8)).collect::<Option<Vec<i8>>>().map(Vector::Byte),
        _ => None,
    }
}

/// Lucene `VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT`.
fn mip(a: &Vector, b: &Vector) -> f32 {
    let dot = match (a, b) {
        (Vector::Float(x), Vector::Float(y)) => x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>(),
        (Vector::Byte(x), Vector::Byte(y)) => x.iter().zip(y).map(|(p, q)| *p as i32 * *q as i32).sum::<i32>() as f32,
        (Vector::Float(x), Vector::Byte(y)) | (Vector::Byte(y), Vector::Float(x)) => x.iter().zip(y).map(|(p, q)| p * *q as f32).sum::<f32>(),
    };
    if dot < 0.0 { 1.0 / (1.0 - dot) } else { dot + 1.0 }
}

/// The diversification vector of each doc (`DiversifyRetrieverBuilder.getFieldVectorForSearchHit`):
/// a `semantic_text` field's best-matching chunk embedding (against the
/// query vector), else the field's fetched `dense_vector` value.
fn diversify_vectors(cx: &Ctx, docs: &[RankDoc], field: &str, qv: Option<&Vector>) -> Result<Vec<Option<Vector>>> {
    let mut out: Vec<Option<Vector>> = vec![None; docs.len()];
    let mut by_index: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        match by_index.iter_mut().find(|(n, _)| *n == d.index) {
            Some((_, v)) => v.push(i),
            None => by_index.push((d.index.clone(), vec![i])),
        }
    }
    for (index, positions) in by_index {
        let targets: Vec<Target> = cx.targets.iter().filter(|t| t.svc.name == index).cloned().collect();
        if targets.is_empty() {
            continue;
        }
        let semantic = targets[0].svc.lookup().fields.get(field).is_some_and(|f| crate::inference::semantic::is_inference_field(f));
        let ids: Vec<&str> = positions.iter().map(|i| docs[*i].id.as_str()).collect();
        let body = json!({ "query": { "ids": { "values": ids } }, "size": ids.len(), "_source": semantic, "fields": [field, "_inference_fields"] });
        let mut req = search::parse_request(&body, &HashMap::new())?;
        if cx.remote.is_some() {
            // Every target answers remotely: keep this index's hits.
            req.size = ids.len() * cx.targets.len().max(1);
        }
        let v = exec_json(cx, &targets, &req)?;
        for h in v["hits"]["hits"].as_array().into_iter().flatten() {
            if h["_index"].as_str().is_some_and(|i| i != index) {
                continue;
            }
            let id = h["_id"].as_str().unwrap_or_default();
            let Some(&pos) = positions.iter().find(|i| docs[**i].id == id) else { continue };
            if semantic {
                let chunks = &h["_source"]["_inference_fields"][field]["inference"]["chunks"];
                let embeddings: Vec<Vector> = chunks.as_object().into_iter().flat_map(|m| m.values()).flat_map(|a| a.as_array().into_iter().flatten()).filter_map(|c| vector_of(&c["embeddings"])).collect();
                if embeddings.is_empty() {
                    continue;
                }
                let Some(q) = qv else {
                    return Err(EsError::illegal_argument("[query_vector] or [query_vector_builder] must be supplied when diversifying on a [semantic_text] field."));
                };
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (i, e) in embeddings.iter().enumerate() {
                    let sc = mip(e, q);
                    if sc > best_score {
                        best = i;
                        best_score = sc;
                    }
                }
                out[pos] = embeddings.into_iter().nth(best);
            } else {
                let vals = &h["fields"][field];
                // `fields` renders a vector as its list of values.
                out[pos] = match vals.as_array().map(|a| a.as_slice()) {
                    Some([Value::String(s)]) => vector_of(&Value::String(s.clone())),
                    Some([Value::Array(inner)]) => vector_of(&Value::Array(inner.clone())),
                    _ => vector_of(vals),
                };
            }
        }
    }
    Ok(out)
}

/// `MMRResultDiversification.diversify`: positions (in child order) of the
/// `size` docs picked by maximal marginal relevance.
fn mmr_select(vectors: &[Option<Vector>], qv: Option<&Vector>, lambda: f32, size: usize) -> Vec<usize> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }
    let has = |i: usize| vectors[i].as_ref().is_some_and(|v| match v {
        Vector::Float(x) => !x.is_empty(),
        Vector::Byte(x) => !x.is_empty(),
    });
    let query_sim: Vec<f32> = (0..n).map(|i| match (qv, &vectors[i]) {
        (Some(q), Some(v)) => mip(v, q),
        _ => 0.0,
    }).collect();
    let Some(best) = (0..n).filter(|i| vectors[*i].is_some()).reduce(|a, b| if query_sim[a] >= query_sim[b] { a } else { b }) else {
        return Vec::new();
    };
    let mut selected = vec![best];
    let mut prev = best;
    let mut max_sim = vec![f32::NEG_INFINITY; n];
    let mut x = 0;
    while x < size && selected.len() < size && selected.len() < n {
        let mut pick: Option<usize> = None;
        let mut pick_score = f32::NEG_INFINITY;
        for i in 0..n {
            if selected.contains(&i) || !has(i) {
                continue;
            }
            let (Some(v), Some(p)) = (&vectors[i], &vectors[prev]) else { continue };
            max_sim[i] = max_sim[i].max(mip(v, p));
            let mmr = lambda * query_sim[i] - (1.0 - lambda) * max_sim[i];
            if mmr > pick_score {
                pick_score = mmr;
                pick = Some(i);
            }
        }
        if let Some(p) = pick {
            selected.push(p);
            prev = p;
        }
        x += 1;
    }
    selected.sort_unstable();
    selected
}

fn rule_rewritten_query(organic: Option<Value>, pinned: &[DocRef], excluded: &[DocRef]) -> Value {
    let mut q = organic.unwrap_or_else(|| json!({ "match_all": {} }));
    if !pinned.is_empty() {
        let mut pq = if pinned.iter().any(|p| p.index.is_some()) {
            json!({ "docs": pinned.iter().map(|p| match &p.index { Some(ix) => json!({ "_index": ix, "_id": p.id }), None => json!({ "_id": p.id }) }).collect::<Vec<_>>() })
        } else {
            json!({ "ids": pinned.iter().map(|p| p.id.clone()).collect::<Vec<_>>() })
        };
        pq["organic"] = q;
        q = json!({ "pinned": pq });
    }
    if !excluded.is_empty() {
        let ex: Vec<Value> = excluded
            .iter()
            .map(|e| match &e.index {
                Some(ix) => json!({ "bool": { "filter": [{ "ids": { "values": [e.id] } }, { "term": { "_index": ix } }] } }),
                None => json!({ "ids": { "values": [e.id] } }),
            })
            .collect();
        q = json!({ "bool": { "must": [q], "must_not": ex } });
    }
    q
}

/// Java `Object.toString` of a parsed JSON value (`List`/`Map` toString).
fn java_string(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => i.to_string(),
            (None, Some(f)) => crate::ingest::jtext::java_double(f),
            _ => n.to_string(),
        },
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(java_string).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => java_map_string(o),
    }
}

/// Java `HashMap.toString`: entries in the map's bucket order (spread
/// `String.hashCode` of the key; default capacity, growing at 0.75 load).
fn java_map_string(o: &Map<String, Value>) -> String {
    let hash = |s: &str| -> u32 {
        let h = s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)) as u32;
        h ^ (h >> 16)
    };
    let mut cap = 16usize;
    while o.len() as f64 > cap as f64 * 0.75 {
        cap *= 2;
    }
    let mut keys: Vec<(usize, usize, &String)> = o.keys().enumerate().map(|(i, k)| ((hash(k) as usize) & (cap - 1), i, k)).collect();
    keys.sort();
    format!("{{{}}}", keys.iter().map(|(_, _, k)| format!("{k}={}", java_string(&o[k.as_str()]))).collect::<Vec<_>>().join(", "))
}

/// A query matching exactly `docs` (union of a compound retriever whose
/// `min_score` dropped docs: only the survivors count as matches).
fn ids_union(docs: &[RankDoc]) -> Value {
    let mut by_index: HashMap<&str, Vec<&str>> = HashMap::new();
    for d in docs {
        by_index.entry(d.index.as_str()).or_default().push(d.id.as_str());
    }
    let parts: Vec<Value> = by_index
        .into_iter()
        .map(|(ix, ids)| json!({ "bool": { "filter": [{ "ids": { "values": ids } }, { "term": { "_index": ix } }] } }))
        .collect();
    if parts.is_empty() {
        return json!({ "match_none": {} });
    }
    union_of(parts)
}

/// Fused `(key, score)` → ranked docs: ES order (score desc, then doc order),
/// `min_score`, window.
fn finish(fused: Vec<((usize, String), f32)>, by_key: &HashMap<(usize, String), RankDoc>, min_score: Option<f32>, window: usize) -> Vec<RankDoc> {
    let mut docs: Vec<RankDoc> = fused
        .into_iter()
        .filter_map(|(k, s)| {
            by_key.get(&k).map(|d| {
                let mut d = d.clone();
                d.score = s;
                d.pinned = None;
                d
            })
        })
        .filter(|d| min_score.map(|m| d.score >= m).unwrap_or(true))
        .collect();
    docs.sort_by(cmp_ranked);
    docs.truncate(window);
    docs
}

fn plan_of(l: &RankList) -> RankPlan {
    let mut docs: HashMap<String, Vec<PlanDoc>> = HashMap::new();
    for (i, d) in l.docs.iter().enumerate() {
        docs.entry(d.index.clone()).or_default().push((d.id.clone(), d.score, i + 1, d.pinned, d.expl.clone()));
    }
    let order = l.docs.iter().enumerate().map(|(i, d)| ((d.index.clone(), d.id.clone()), i)).collect();
    RankPlan { docs, union: l.union.clone(), order }
}

/// Entry point from [`search::execute`] for requests with a `retriever` or
/// a legacy `rank`.
// ---------------------------------------------------------------------------
// Multi-field query format rewrite (`RRFRetrieverBuilder.doRewrite`,
// `LinearRetrieverBuilder.doRewrite`, `MultiFieldsInnerRetrieverUtils`).

/// `QueryParserHelper.parseFieldsAndWeights`: `field^weight` entries
/// (a repeated bare name multiplies into the earlier weight, as ES's
/// `containsKey(field)` check does).
fn fields_and_weights(fields: &[String]) -> Result<Vec<(String, f32)>> {
    let mut out: Vec<(String, f32)> = Vec::new();
    for f in fields {
        let (name, mut w) = match f.find('^') {
            Some(i) => (&f[..i], f[i + 1..].parse::<f32>().map_err(|_| EsError::illegal_argument(format!("failed to parse weight of field [{f}]")))?),
            None => (f.as_str(), 1.0),
        };
        if let Some(prev) = out.iter().find(|(n, _)| n == f).map(|e| e.1) {
            w *= prev;
        }
        match out.iter_mut().find(|(n, _)| n == name) {
            Some(e) => e.1 = w,
            None => out.push((name.to_string(), w)),
        }
    }
    Ok(out)
}

/// `index.query.default_field` of an index (`*` when unset).
fn default_fields(t: &Target) -> Vec<String> {
    t.svc.query_default_fields()
}

fn std_query(q: Value) -> Retriever {
    std_child(q)
}

/// Rewrite every multi-field-format `rrf` / `linear` against the targets.
fn expand(r: Retriever, targets: &[Target]) -> Result<Retriever> {
    Ok(match r {
        Retriever::MultiField { c, kind, fields, query, window, rank_constant, normalizer } => {
            let v8 = crate::version::current_rest_api_version().is_v8();
            let mut indices: Vec<&Target> = Vec::new();
            for t in targets {
                if !indices.iter().any(|x| x.svc.name == t.svc.name) {
                    indices.push(t);
                }
            }
            if v8 && indices.len() > 1 {
                return Err(EsError::illegal_argument(format!("[{kind}] cannot specify [query] when querying multiple indices")));
            }
            let check = |w: f32| -> Result<()> {
                if v8 && kind == "rrf" && w != 1.0 {
                    return Err(EsError::illegal_argument("[rrf] does not support per-field weights in [fields]"));
                }
                if w < 0.0 {
                    return Err(EsError::illegal_argument(format!("[{kind}] per-field weights must be non-negative")));
                }
                Ok(())
            };
            let explicit = match &fields {
                Some(f) => fields_and_weights(f)?,
                None => Vec::new(),
            };
            for (_, w) in &explicit {
                check(*w)?;
            }
            // `MultiFieldsInnerRetrieverUtils.generateInnerRetrievers`.
            let all: Vec<String> = indices.iter().map(|t| t.svc.name.clone()).collect();
            let with_index_filter = |q: Value, names: &[String]| -> Value {
                if names.len() == all.len() {
                    q
                } else {
                    json!({ "bool": { "must": [q], "filter": [{ "terms": { "_index": names } }] } })
                }
            };
            // Lexical: the non-inference fields per index, grouped.
            let mut lexical_groups: Vec<(Vec<(String, f32)>, Vec<String>)> = Vec::new();
            // Semantic: (field, inference_id) -> (weight, indices).
            let mut semantic_groups: Vec<((String, String), f32, Vec<String>)> = Vec::new();
            for t in &indices {
                let wanted = if explicit.is_empty() {
                    let d = fields_and_weights(&default_fields(t))?;
                    for (_, w) in &d {
                        check(*w)?;
                    }
                    d
                } else {
                    explicit.clone()
                };
                let lookup = t.svc.lookup();
                let mut inference_fields: Vec<(String, String)> = lookup
                    .fields
                    .iter()
                    .filter(|(_, f)| crate::inference::semantic::is_inference_field(f))
                    .map(|(n, f)| (n.clone(), f.ext.as_ref().and_then(|e| e.params.get("inference_id")).and_then(Value::as_str).unwrap_or("").to_string()))
                    .collect();
                inference_fields.sort();
                // `IndexMetadata.getMatchingInferenceFields` (wildcards resolved, repeated matches multiply).
                let mut matched: Vec<(String, String, f32)> = Vec::new();
                let mut add = |name: &str, id: &str, w: f32| match matched.iter_mut().find(|(n, _, _)| n == name) {
                    Some(e) => e.2 *= w,
                    None => matched.push((name.to_string(), id.to_string(), w)),
                };
                for (f, w) in &wanted {
                    if let Some((n, id)) = inference_fields.iter().find(|(n, _)| n == f) {
                        add(n, id, *w);
                    } else if f == "*" {
                        inference_fields.iter().for_each(|(n, id)| add(n, id, *w));
                    } else if f.contains('*') {
                        inference_fields.iter().filter(|(n, _)| crate::mapping::wildcard_match(f, n)).for_each(|(n, id)| add(n, id, *w));
                    }
                }
                for (n, id, w) in matched {
                    match semantic_groups.iter_mut().find(|(k, _, _)| k.0 == n && k.1 == id) {
                        Some((_, gw, names)) => {
                            if *gw != w {
                                return Err(EsError::illegal_argument(format!("field [{n}] has different weights in indices [{}] and [{}]", names[0], t.svc.name)));
                            }
                            names.push(t.svc.name.clone());
                        }
                        None => semantic_groups.push(((n, id), w, vec![t.svc.name.clone()])),
                    }
                }
                let lexical: Vec<(String, f32)> = wanted.into_iter().filter(|(f, _)| !inference_fields.iter().any(|(n, _)| n == f)).collect();
                if lexical.is_empty() {
                    continue;
                }
                let same = |a: &Vec<(String, f32)>| a.len() == lexical.len() && a.iter().all(|x| lexical.contains(x));
                match lexical_groups.iter_mut().find(|(k, _)| same(k)) {
                    Some((_, names)) => names.push(t.svc.name.clone()),
                    None => lexical_groups.push((lexical, vec![t.svc.name.clone()])),
                }
            }
            let mut kids: Vec<Retriever> = Vec::new();
            if !lexical_groups.is_empty() {
                let qs: Vec<Value> = lexical_groups
                    .iter()
                    .map(|(fw, names)| {
                        let fl: Vec<Value> = fw.iter().map(|(f, w)| if *w == 1.0 { json!(f) } else { json!(format!("{f}^{w}")) }).collect();
                        with_index_filter(json!({ "multi_match": { "query": query, "fields": fl, "type": "most_fields" } }), names)
                    })
                    .collect();
                kids.push(std_query(if qs.len() == 1 { qs.into_iter().next().unwrap_or(Value::Null) } else { json!({ "bool": { "should": qs } }) }));
            }
            if !semantic_groups.is_empty() {
                let sem: Vec<(Retriever, f32)> = semantic_groups.iter().map(|((f, _), w, names)| (std_query(with_index_filter(json!({ "match": { f.clone(): { "query": query } } }), names)), *w)).collect();
                kids.push(if kind == "rrf" {
                    Retriever::Rrf { c: Common::default(), children: sem, window, rank_constant }
                } else {
                    Retriever::Linear { c: Common::default(), children: sem.into_iter().map(|(r, w)| (r, w, normalizer.clone())).collect(), window }
                });
            }
            if kids.is_empty() {
                // No index matched the pattern: nothing to search.
                return Ok(Retriever::Standard { c: Common::default(), query: Some(json!({ "match_none": {} })), sort: None, search_after: None, terminate_after: None, collapse: None });
            }
            // The rewritten retriever keeps only the pre-filters (as ES does).
            let c = Common { filter: c.filter, min_score: None, name: None, legacy: false };
            if kind == "rrf" {
                Retriever::Rrf { c, children: kids.into_iter().map(|k| (k, 1.0)).collect(), window, rank_constant }
            } else {
                Retriever::Linear { c, children: kids.into_iter().map(|k| (k, 1.0, normalizer.clone())).collect(), window }
            }
        }
        Retriever::Rrf { c, children, window, rank_constant } => Retriever::Rrf { c, children: children.into_iter().map(|(k, w)| Ok((expand(k, targets)?, w))).collect::<Result<_>>()?, window, rank_constant },
        Retriever::Linear { c, children, window } => Retriever::Linear { c, children: children.into_iter().map(|(k, w, n)| Ok((expand(k, targets)?, w, n))).collect::<Result<_>>()?, window },
        Retriever::Rescorer { c, child, rescore, window } => Retriever::Rescorer { c, child: Box::new(expand(*child, targets)?), rescore, window },
        Retriever::Pinned { c, kind, child, pinned, excluded, window, sort_checked, rule_info } => {
            Retriever::Pinned { c, kind, child: Box::new(expand(*child, targets)?), pinned, excluded, window, sort_checked, rule_info }
        }
        Retriever::Rerank { c, child, field, inference_id, inference_text, window, allow_failures, chunk } => {
            Retriever::Rerank { c, child: Box::new(expand(*child, targets)?), field, inference_id, inference_text, window, allow_failures, chunk }
        }
        Retriever::Random { c, child, window, seed } => Retriever::Random { c, child: Box::new(expand(*child, targets)?), window, seed },
        Retriever::Diversify { c, child, field, window, size, lambda, query_vector } => Retriever::Diversify { c, child: Box::new(expand(*child, targets)?), field, window, size, lambda, query_vector },
        other => other,
    })
}

/// `CompoundRetrieverBuilder`: a failing child search fails the compound
/// retriever with the child failures attached as `suppressed`.
fn child_failure(name: &str, kid: &str, e: EsError) -> EsError {
    if !matches!(e.kind.as_str(), "search_phase_execution_exception" | "query_shard_exception") {
        return e;
    }
    let status = e.status;
    let mut sup = e.error_object(false);
    if e.kind == "query_shard_exception" {
        // A shard-level failure reaches the coordinator as a search phase failure.
        let reason = sup.clone();
        sup = json!({ "type": "search_phase_execution_exception", "reason": "all shards failed", "phase": "query", "grouped": true, "failed_shards": [{ "shard": 0, "index": Value::Null, "reason": reason }] });
    }
    EsError::new(status, "status_exception", format!("[{name}] search failed - retrievers '[{kid}]' returned errors. All failures are attached as suppressed exceptions.")).with("suppressed", json!([sup]))
}

/// A cluster coordinator's first phase of a `retriever` / `rank` search:
/// run the children through `remote` (over every shard) and return the
/// body of the final search, a plain search the shards can run and the
/// coordinator can merge: `standard` / `knn` roots rewritten into
/// `query` / `knn`, compound roots into their fused [`RankPlan`]
/// (under [`RANK_PLAN_KEY`]).
pub fn plan(targets: &[Target], req: &SearchRequest, body: &Value, opts: &ExecOpts, remote: &dyn ChildSearch) -> Result<Value> {
    let root = match &req.retriever {
        Some(rv) => parse(rv)?,
        None => legacy_rank(req, false)?,
    };
    let mut errs = Vec::new();
    let compound = !matches!(root, Retriever::Standard { .. } | Retriever::Knn { .. });
    validate(&root, if req.size_explicit || !compound || req.rank.is_some() { req.size } else { 0 }, false, false, &mut errs)?;
    if !errs.is_empty() {
        let r: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&r));
    }
    let root = expand(root, targets)?;
    let mut out = body.as_object().cloned().unwrap_or_default();
    for k in ["retriever", "rank", "sub_searches"] {
        out.remove(k);
    }
    let mut base = req.clone();
    base.retriever = None;
    base.rank = None;
    base.sub_searches = None;
    if req.rank.is_some() {
        base.query = None;
        base.knn = Vec::new();
        out.remove("query");
        out.remove("knn");
    }
    match &root {
        Retriever::Standard { c, query, sort, search_after, terminate_after, collapse } => {
            out.insert("query".into(), bool_filter(query.clone(), &c.filter));
            if let Some(s) = sort {
                out.insert("sort".into(), s.clone());
            }
            if let Some(sa) = search_after {
                out.insert("search_after".into(), sa.clone());
            }
            match terminate_after {
                Some(t) if *t > 0 => {
                    out.insert("terminate_after".into(), json!(t));
                }
                Some(_) => {
                    out.remove("terminate_after");
                }
                None => {}
            }
            if let Some(m) = c.min_score {
                out.insert("min_score".into(), json!(m));
            }
            if let Some(col) = collapse {
                out.insert("collapse".into(), col.clone());
            }
        }
        Retriever::Knn { c, spec } => {
            out.remove("query");
            out.insert("knn".into(), Value::Object(knn_spec_with(spec, &c.filter, req.size.max(1))));
            if let Some(m) = c.min_score {
                out.insert("min_score".into(), json!(m));
            }
        }
        _ => {
            let cx = Ctx { targets, opts, base: &base, remote: Some(remote) };
            let window = root.window().unwrap_or(DEFAULT_RANK_WINDOW_SIZE);
            let list = eval(&root, &cx, &[], window)?;
            if !req.size_explicit && req.rank.is_none() {
                out.insert("size".into(), json!(window));
            }
            out.remove("query");
            out.remove("knn");
            out.insert(RANK_PLAN_KEY.into(), plan_of(&list).to_json());
            if let Some(m) = root.common().min_score {
                out.insert("min_score".into(), json!(base.min_score.map_or(m, |x| x.max(m))));
            }
        }
    }
    Ok(Value::Object(out))
}

pub fn execute(targets: &[Target], req: &SearchRequest, opts: &ExecOpts) -> Result<SearchOutput> {
    let scroll = opts.scroll_id.is_some();
    let root = match &req.retriever {
        Some(rv) => parse(rv)?,
        None => legacy_rank(req, scroll)?,
    };
    let mut errs = Vec::new();
    // ES validates against the request's own `size` (`-1` when unset: no
    // window check), and a compound retriever then defaults it to its
    // `rank_window_size` (`RankDocsRetrieverBuilder.extractToSearchSourceBuilder`).
    let compound = !matches!(root, Retriever::Standard { .. } | Retriever::Knn { .. });
    validate(&root, if req.size_explicit || !compound || req.rank.is_some() { req.size } else { 0 }, scroll, false, &mut errs)?;
    if !errs.is_empty() {
        let r: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&r));
    }
    let root = expand(root, targets)?;
    let mut base = req.clone();
    base.retriever = None;
    base.rank = None;
    base.sub_searches = None;
    if req.rank.is_some() {
        base.query = None;
        base.knn = Vec::new();
    }
    match &root {
        Retriever::Standard { c, query, sort, search_after, terminate_after, collapse } => {
            base.query = Some(bool_filter(query.clone(), &c.filter));
            if let Some(s) = sort {
                let tmp = search::parse_request(&json!({ "sort": s }), &HashMap::new())?;
                base.sort = tmp.sort;
                base.sort_extra = tmp.sort_extra;
            }
            if let Some(sa) = search_after {
                base.search_after = Some(sa.as_array().cloned().ok_or_else(|| EsError::parsing("[search_after] must be an array"))?);
            }
            if terminate_after.is_some() {
                base.terminate_after = terminate_after.filter(|x| *x > 0);
            }
            if c.min_score.is_some() {
                base.min_score = c.min_score;
            }
            if let Some(col) = collapse {
                base.collapse = col.get("field").and_then(Value::as_str).map(str::to_string);
                base.collapse_spec = Some(col.clone());
            }
            search::execute(targets, &base, opts)
        }
        Retriever::Knn { c, spec } => {
            let s = knn_spec_with(spec, &c.filter, req.size.max(1));
            base.query = None;
            base.knn = vec![Value::Object(s)];
            if c.min_score.is_some() {
                base.min_score = c.min_score;
            }
            search::execute(targets, &base, opts)
        }
        // `random_reranker` over a plain `standard` retriever: ES adds a
        // `RandomRankBuilder` to the one search (no compound retriever, no
        // second query): its window is re-scored before the page is cut
        // (see `SearchRequest::random_rerank`). An unset `size` keeps the
        // compound path's default (the window).
        Retriever::Random { c, child, window, seed }
            if c.filter.is_empty() && c.min_score.is_none() && c.name.is_none() && req.size_explicit && !req.explain && req.sort.is_empty() && req.rank.is_none()
                && matches!(&**child, Retriever::Standard { sort: None, search_after: None, terminate_after: None, collapse: None, .. }) =>
        {
            let Retriever::Standard { c: cc, query, .. } = &**child else { unreachable!("matched above") };
            base.query = Some(bool_filter(query.clone(), &cc.filter));
            if cc.min_score.is_some() {
                base.min_score = cc.min_score;
            }
            base.random_rerank = Some((*window, *seed));
            search::execute(targets, &base, opts)
        }
        _ => {
            let cx = Ctx { targets, opts, base: &base, remote: None };
            let window = root.window().unwrap_or(DEFAULT_RANK_WINDOW_SIZE);
            let list = eval(&root, &cx, &[], window)?;
            if !req.size_explicit && req.rank.is_none() {
                base.size = window;
            }
            base.query = None;
            base.knn = Vec::new();
            base.rank_plan = Some(Arc::new(plan_of(&list)));
            if let Some(m) = root.common().min_score {
                base.min_score = Some(base.min_score.map_or(m, |x| x.max(m)));
            }
            search::execute(targets, &base, opts)
        }
    }
}

// ---------------------------------------------------------------------------
// The final search's queries.

/// Rank docs of one index with fixed scores (ES `RankDocsQuery` top part).
#[derive(Debug)]
pub struct RankDocsQuery {
    docs: Vec<PlanDoc>,
    /// Each leaf's matches, looked up once: a search reads them twice
    /// (its hits, then the union's total in [`fix_total`]), and each is an
    /// `_id` term lookup per rank doc in every leaf.
    found: parking_lot::Mutex<Vec<((usize, u32), Arc<Vec<(DocId, f32)>>)>>,
}

impl RankDocsQuery {
    fn new(docs: Vec<PlanDoc>) -> RankDocsQuery {
        RankDocsQuery { docs, found: parking_lot::Mutex::new(Vec::new()) }
    }

    /// `leaf`'s live rank docs with their scores.
    fn matches(&self, leaf: &LeafReader<'_>) -> Arc<Vec<(DocId, f32)>> {
        let key = (leaf.id(), leaf.num_docs());
        if let Some((_, m)) = self.found.lock().iter().find(|(k, _)| *k == key) {
            return m.clone();
        }
        let live = leaf.live_docs();
        let mut out: Vec<(DocId, f32)> = Vec::new();
        for (id, s, ..) in &self.docs {
            if let Some(mut p) = leaf.postings(&Term::new_text(ID_FIELD, id)) {
                while let Some(d) = p.next() {
                    if live(d) {
                        out.push((d, *s));
                    }
                }
            }
        }
        let out = Arc::new(out);
        self.found.lock().push((key, out.clone()));
        out
    }
}

impl CustomQuery for RankDocsQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let out = self.matches(leaf);
        if out.is_empty() {
            None
        } else {
            Some(Box::new(ScoredDocs::new((*out).clone())))
        }
    }
    fn name(&self) -> &'static str {
        "rank_docs"
    }
    /// ES `RankDoc.explain` / `PinnedRankDoc.explain`; compound retrievers
    /// (`rrf`, `linear`, rerankers) carry their own per-doc explanation.
    fn explain(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        let id = leaf.keyword_value(ID_FIELD, doc).unwrap_or_default();
        let Some((_, s, rank, pinned, expl)) = self.docs.iter().find(|d| d.0 == id) else {
            return Explanation::leaf(score, "rank doc: score of the compound retriever".to_string());
        };
        let rank_doc = |score: f32| {
            Explanation::new(
                *rank as f32,
                format!("doc [{doc}] with an original score of [{}] is at rank [{rank}] from the following source queries.", crate::ingest::jtext::java_float(score)),
                Vec::new(),
            )
        };
        match (pinned, expl) {
            (Some(true), _) => Explanation::new(score, "Pinned document, original explanation:", vec![rank_doc(*s)]),
            (Some(false), _) => rank_doc(*s),
            (None, Some(e)) => e.clone(),
            (None, None) => Explanation::leaf(score, "rank doc: score of the compound retriever".to_string()),
        }
    }
}

/// `(aggregation/total query, hit query)` of a rank plan for one index.
/// `page`: when the hits are the fused ranking's first ranks (a score-
/// ordered page), only rank docs up to this rank are looked up for them:
/// a shard renders its share of the page, not its share of the whole
/// window (eight shards rendered up to `size` hits each, `_id` reads
/// included, of which the merge kept `size`). The union for totals and
/// aggregations keeps every rank doc.
pub fn plan_queries(plan: &RankPlan, index: &str, union: Query, page: Option<usize>) -> (Query, Query) {
    let docs = plan.docs.get(index).cloned().unwrap_or_default();
    let every = Query::Custom(Arc::new(RankDocsQuery::new(docs.clone())));
    let top = match page {
        Some(page) if docs.iter().any(|d| d.2 > page) => Query::Custom(Arc::new(RankDocsQuery::new(docs.into_iter().filter(|d| d.2 <= page).collect()))),
        _ => every.clone(),
    };
    let all = Query::Bool {
        must: vec![],
        should: vec![every, Query::ConstScore { query: Box::new(union), score: 0.0, point_set: false }],
        must_not: vec![],
        filter: vec![],
        minimum_should_match: MinShouldMatch::Count(1),
    };
    (all, top)
}

impl RankPlan {
    /// The plan as JSON (see [`RANK_PLAN_KEY`]).
    pub fn to_json(&self) -> Value {
        let docs: Map<String, Value> = self
            .docs
            .iter()
            .map(|(index, list)| {
                let l: Vec<Value> = list.iter().map(|(id, score, rank, pinned, expl)| json!([id, score, rank, pinned, expl.as_ref().map(crate::explain_fmt::to_json)])).collect();
                (index.clone(), Value::Array(l))
            })
            .collect();
        let mut order: Vec<(&(String, String), &usize)> = self.order.iter().collect();
        order.sort_by_key(|(_, pos)| **pos);
        let order: Vec<Value> = order.into_iter().map(|((index, id), _)| json!([index, id])).collect();
        json!({ "docs": docs, "union": self.union, "order": order })
    }

    /// A plan from [`RankPlan::to_json`].
    pub fn from_json(v: &Value) -> Result<RankPlan> {
        let bad = || EsError::parsing(format!("[{RANK_PLAN_KEY}] is malformed"));
        let mut docs: HashMap<String, Vec<PlanDoc>> = HashMap::new();
        for (index, list) in v["docs"].as_object().ok_or_else(bad)? {
            let mut out = Vec::new();
            for d in list.as_array().ok_or_else(bad)? {
                let id = d[0].as_str().ok_or_else(bad)?.to_string();
                let score = d[1].as_f64().map(|x| x as f32).unwrap_or(f32::NAN);
                let rank = d[2].as_u64().ok_or_else(bad)? as usize;
                let pinned = d[3].as_bool();
                let expl = d.get(4).filter(|e| e.is_object()).map(expl_from_json);
                out.push((id, score, rank, pinned, expl));
            }
            docs.insert(index.clone(), out);
        }
        let mut order = HashMap::new();
        for (pos, e) in v["order"].as_array().ok_or_else(bad)?.iter().enumerate() {
            order.insert((e[0].as_str().ok_or_else(bad)?.to_string(), e[1].as_str().ok_or_else(bad)?.to_string()), pos);
        }
        Ok(RankPlan { docs, union: v["union"].clone(), order })
    }

    pub fn union(&self) -> &Value {
        &self.union
    }
    /// Number of rank docs.
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    /// Fused rank of a doc (`usize::MAX` when not a rank doc).
    pub fn rank_of(&self, index: &str, id: &str) -> usize {
        self.order.get(&(index.to_string(), id.to_string())).copied().unwrap_or(usize::MAX)
    }
}

/// ES counts every doc of the union (rank docs + tail) as a hit; the hit
/// query only returned the rank docs.
pub fn fix_total(req: &SearchRequest, mut top: TopDocs, leaves: &[LeafReader<'_>], all: &Query) -> Result<TopDocs> {
    let limit = match req.track_total_hits {
        None | Some(TrackTotalHits::False) => return Ok(top),
        Some(TrackTotalHits::True) => u64::MAX,
        Some(TrackTotalHits::UpTo(n)) => n,
    };
    let ctx = SearchContext::for_query(all, leaves, rustic_core::query::Bm25Params::default());
    ctx.execution_result()?;
    let mut n = 0u64;
    if req.min_score.is_none() {
        // Membership only: each leaf's matches are OR-ed into a doc bitset
        // clause by clause (the union of the rank docs and the children's
        // queries), not walked through the union's scorer, whose per-doc
        // heap over every clause (a sparse child's 100+ tokens) dominated
        // compound retriever searches with exact totals.
        for leaf in leaves {
            let mut words = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
            fill_matches(all, leaf, &ctx, &mut words);
            if let Some(skip) = leaf.deleted_bits() {
                for (w, d) in words.iter_mut().zip(skip.words()) {
                    *w &= !d;
                }
            }
            n += words.iter().map(|w| w.count_ones() as u64).sum::<u64>();
            if n > limit {
                break;
            }
        }
        ctx.execution_result()?;
        if n > limit {
            top.total_hits = limit;
            top.total_hits_relation = TotalHitsRelation::Gte;
        } else {
            top.total_hits = n;
            top.total_hits_relation = TotalHitsRelation::Eq;
        }
        return Ok(top);
    }
    'outer: for leaf in leaves {
        let Some(mut s) = all.scorer(leaf, &ctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == rustic_core::query::TERMINATED {
                break;
            }
            if !live(d) {
                continue;
            }
            if let Some(m) = req.min_score {
                if s.score() < m {
                    continue;
                }
            }
            n += 1;
            if n > limit {
                break 'outer;
            }
        }
    }
    if n > limit {
        top.total_hits = limit;
        top.total_hits_relation = TotalHitsRelation::Gte;
    } else {
        top.total_hits = n;
        top.total_hits_relation = TotalHitsRelation::Eq;
    }
    ctx.execution_result()?;
    Ok(top)
}

/// Set the bit of every doc of `leaf` that `q` matches (deleted ones too):
/// a should-only disjunction (and constant-score / boost wrappers) clause
/// by clause, anything else by walking its scorer.
fn fill_matches(q: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext, words: &mut [u64]) {
    match q {
        Query::Bool { must, should, must_not, filter, minimum_should_match }
            if must.is_empty() && must_not.is_empty() && filter.is_empty() && !should.is_empty() && minimum_should_match.resolve(should.len()) <= 1 =>
        {
            for c in should {
                fill_matches(c, leaf, ctx, words);
            }
        }
        Query::ConstScore { query, .. } | Query::Boost { query, .. } => fill_matches(query, leaf, ctx, words),
        _ => {
            if let Some(mut s) = q.scorer(leaf, ctx) {
                loop {
                    let d = s.advance();
                    if d == rustic_core::query::TERMINATED {
                        break;
                    }
                    if let Some(w) = words.get_mut(d as usize / 64) {
                        *w |= 1u64 << (d % 64);
                    }
                }
            }
        }
    }
}

/// `new java.util.Random(seed).nextFloat()`.
/// `random_reranker`'s score of shard-level doc `doc`: ES
/// `new Random(seed != null ? seed + doc : doc).nextFloat()` (`seed + doc`
/// in `int` arithmetic).
pub(crate) fn random_rerank_score(seed: Option<i64>, doc: i32) -> f32 {
    java_random_next_float(seed.map_or(i64::from(doc), |s| i64::from((s as i32).wrapping_add(doc))))
}

fn java_random_next_float(seed: i64) -> f32 {
    const MULT: i64 = 0x5DEECE66D;
    const MASK: i64 = (1 << 48) - 1;
    let s = (seed ^ MULT) & MASK;
    let s = (s.wrapping_mul(MULT).wrapping_add(0xB)) & MASK;
    let next24 = (s >> (48 - 24)) as i32;
    next24 as f32 / (1i32 << 24) as f32
}

#[cfg(test)]
mod cluster_plan_tests {
    use super::*;

    #[test]
    fn rank_plan_json_round_trip() {
        let mut docs = HashMap::new();
        docs.insert("i".to_string(), vec![("a".to_string(), 0.5f32, 1usize, None, Some(Explanation::leaf(0.5, "x"))), ("b".to_string(), 0.25f32, 2usize, Some(true), None)]);
        let order = [(("i".to_string(), "a".to_string()), 0usize), (("i".to_string(), "b".to_string()), 1usize)].into_iter().collect();
        let p = RankPlan { docs, union: json!({"match_all": {}}), order };
        let back = RankPlan::from_json(&p.to_json()).unwrap();
        assert_eq!(back.rank_of("i", "a"), 0);
        assert_eq!(back.rank_of("i", "b"), 1);
        assert_eq!(back.union(), &json!({"match_all": {}}));
        let d = &back.docs["i"];
        assert_eq!((d[0].1, d[0].2, d[0].3), (0.5, 1, None));
        assert_eq!(d[0].4.as_ref().map(|e| e.description.clone()), Some("x".to_string()));
        assert_eq!((d[1].1, d[1].3), (0.25, Some(true)));
        assert!(RankPlan::from_json(&json!({"docs": 1})).is_err());
    }

    #[test]
    fn child_body_keeps_what_the_child_search_needs() {
        let base = search::parse_request(&json!({"runtime_mappings": {"r": {"type": "long"}}}), &HashMap::new()).unwrap();
        let mut req = child_request(&base, 7);
        req.query = Some(json!({"term": {"f": "v"}}));
        req.min_score = Some(1.5);
        req.collapse = Some("c".into());
        let b = child_body(&req, Some(&json!([{"f": "desc"}])));
        assert_eq!(b["size"], 7);
        assert_eq!(b["query"], json!({"term": {"f": "v"}}));
        assert_eq!(b["min_score"], 1.5);
        assert_eq!(b["collapse"], json!({"field": "c"}));
        assert_eq!(b["sort"], json!([{"f": "desc"}]));
        assert_eq!(b["_source"], false);
        assert_eq!(b["track_scores"], true);
        assert_eq!(b["runtime_mappings"], json!({"r": {"type": "long"}}));
        assert!(b.get("knn").is_none() && b.get(RANK_PLAN_KEY).is_none());
    }

    /// The final search's hits look up only the page's rank docs; its
    /// union (totals, aggregations) keeps every one; each leaf's lookups
    /// are made once.
    #[test]
    fn final_hits_read_the_page_of_rank_docs() {
        let docs: Vec<PlanDoc> = (1..=5).map(|r| (format!("d{r}"), 1.0 / r as f32, r, None, None)).collect();
        let plan = RankPlan { docs: HashMap::from([("i".to_string(), docs)]), union: json!({"match_all": {}}), order: HashMap::new() };
        // The rank docs a query holds, read off its debug form.
        let ranks = |q: &Query| -> Vec<usize> {
            let Query::Custom(c) = q else { panic!("rank docs query expected") };
            let shown = format!("{c:?}");
            (1..=5).filter(|r| shown.contains(&format!("\"d{r}\""))).collect()
        };
        let (all, top) = plan_queries(&plan, "i", Query::MatchAll, Some(2));
        assert_eq!(ranks(&top), vec![1, 2]);
        let Query::Bool { should, .. } = &all else { panic!("union expected") };
        assert_eq!(ranks(&should[0]), vec![1, 2, 3, 4, 5]);
        let (_, top) = plan_queries(&plan, "i", Query::MatchAll, None);
        assert_eq!(ranks(&top), vec![1, 2, 3, 4, 5]);
    }

    /// A lone kNN child goes out as one clause object (the cluster's lone
    /// kNN search, cut to its global `k`), not the DFS kNN array route.
    #[test]
    fn a_lone_knn_child_is_one_clause_object() {
        let base = search::parse_request(&json!({}), &HashMap::new()).unwrap();
        let mut req = child_request(&base, 100);
        let clause = json!({"field": "v", "query_vector": [1.0], "k": 10, "num_candidates": 50});
        req.knn = vec![clause.clone()];
        assert_eq!(child_body(&req, None)["knn"], clause);
        req.knn = vec![clause.clone(), clause.clone()];
        assert!(child_body(&req, None)["knn"].is_array());
        req.knn = vec![clause.clone()];
        req.query = Some(json!({"match_all": {}}));
        assert!(child_body(&req, None)["knn"].is_array());
    }

    /// Leaf children of a compound retriever run remotely are sent at once,
    /// in order, and fused as one at a time would be.
    #[test]
    fn leaf_children_are_searched_together() {
        struct Fake(std::cell::RefCell<Vec<usize>>);
        impl ChildSearch for Fake {
            fn search(&self, _body: Value) -> Result<Value> {
                self.0.borrow_mut().push(1);
                Ok(json!({"hits": {"hits": [{"_index": "i", "_id": "a", "_score": 1.0}]}}))
            }
            fn search_all(&self, bodies: Vec<Value>) -> Vec<Result<Value>> {
                self.0.borrow_mut().push(bodies.len());
                bodies.iter().enumerate().map(|(n, b)| {
                    let id = if b.get("knn").is_some() { "k" } else { "q" };
                    Ok(json!({"hits": {"hits": [{"_index": "i", "_id": format!("{id}{n}"), "_score": 1.0}]}}))
                }).collect()
            }
        }
        let fake = Fake(Default::default());
        let base = search::parse_request(&json!({}), &HashMap::new()).unwrap();
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 10, scroll_id: None, pit_id: None };
        let cx = Ctx { targets: &[], opts: &opts, base: &base, remote: Some(&fake) };
        let std = parse(&json!({"standard": {"query": {"match_all": {}}}})).unwrap();
        let knn = parse(&json!({"knn": {"field": "v", "query_vector": [1.0], "k": 1, "num_candidates": 2}})).unwrap();
        let lists = remote_leaves(&[&std, &knn], &cx, &[], 10).expect("leaves").into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(*fake.0.borrow(), vec![2], "one multi-search, no single searches");
        assert_eq!(lists[0].docs[0].id, "q0");
        assert_eq!(lists[1].docs[0].id, "k1");
        let rrf = parse(&json!({"rrf": {"retrievers": [{"rrf": {"retrievers": [{"standard": {"query": {"match_all": {}}}}]}}, {"standard": {"query": {"match_all": {}}}}]}})).unwrap();
        let Retriever::Rrf { children, .. } = &rrf else { panic!() };
        let kids: Vec<&Retriever> = children.iter().map(|(c, _)| c).collect();
        assert!(remote_leaves(&kids, &cx, &[], 10).is_none(), "a compound child: evaluated in turn");
    }
}

#[cfg(test)]
mod random_rerank_tests {
    use super::*;
    use crate::index::{IndexService, WriteReq};
    use crate::shard::WriteCond;
    use std::collections::BTreeMap;

    const TEXTS: [&str; 5] = [
        "Pugs are proof that even nature has a sense of humor.",
        "A pugs snore can rival a chainsaw, but it's somehow adorable.",
        "Pugs are like potato chips; you can't have just one wrinkle.",
        "Pugs don't walk; pugs waddle majestically.",
        "A pugs life goal: be the ultimate couch potato, and they're crushing it.",
    ];

    /// The `80_random_rerank_retriever` index; `split`: docs 1-3 and 4-5 in
    /// two segments (shard-level doc ids still 0-4).
    fn index(dir: &std::path::Path, split: bool) -> Arc<IndexService> {
        let s = Arc::new(
            IndexService::create(
                dir,
                "test-index",
                crate::settings::normalize_index_settings(&json!({"number_of_shards": 1, "number_of_replicas": 0})),
                &json!({"properties": {"text": {"type": "text"}, "topic": {"type": "keyword"}}}),
                BTreeMap::new(),
            )
            .unwrap(),
        );
        let w = |from: usize, to: usize| {
            let reqs: Vec<WriteReq> = (from..to).map(|i| WriteReq::Index { id: Some(format!("doc_{}", i + 1)), source: json!({"text": TEXTS[i]}).to_string(), routing: None, cond: WriteCond::default() }).collect();
            assert!(s.write(reqs, true).iter().all(|r| r.is_ok()));
        };
        if split {
            w(0, 3);
            s.flush().unwrap();
            w(3, 5);
        } else {
            w(0, 5);
        }
        s.refresh().unwrap();
        s
    }

    fn search(s: &Arc<IndexService>, body: Value) -> Value {
        let req = search::parse_request(&body, &HashMap::new()).unwrap();
        let targets = vec![Target::new(s.clone())];
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
        serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap()
    }

    fn body(seed: Option<i64>, size: Option<usize>) -> Value {
        let mut r = json!({"retriever": {"standard": {"query": {"query_string": {"query": "pugs"}}}}, "field": "text", "rank_window_size": 10});
        if let Some(s) = seed {
            r["seed"] = json!(s);
        }
        let mut b = json!({"retriever": {"random_reranker": r}});
        if let Some(n) = size {
            b["size"] = json!(n);
        }
        b
    }

    /// ES's `80_random_rerank_retriever.yml` (single search, as ES runs it):
    /// `Random(seed + shard doc)` scores, also with the docs over two
    /// segments, and the same hits as the compound path on one segment.
    #[test]
    fn random_reranker_matches_es() {
        for split in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let s = index(dir.path(), split);
            for (seed, top, score) in [(Some(42), "doc_1", 0.727), (None, "doc_3", 0.731)] {
                let v = search(&s, body(seed, Some(10)));
                assert_eq!(v["hits"]["total"]["value"], 5, "{v}");
                let hits = v["hits"]["hits"].as_array().unwrap();
                assert_eq!(hits.len(), 5);
                assert_eq!(hits[0]["_id"], top, "split {split} seed {seed:?}: {v}");
                assert!((hits[0]["_score"].as_f64().unwrap() - score).abs() < 0.001, "{v}");
                // Every score is its shard-level doc's (doc_N is doc N-1).
                let mut last = f64::INFINITY;
                for h in hits {
                    let n: i32 = h["_id"].as_str().unwrap()["doc_".len()..].parse().unwrap();
                    let sc = h["_score"].as_f64().unwrap();
                    assert_eq!(sc as f32, random_rerank_score(seed, n - 1), "{v}");
                    assert!(sc <= last);
                    last = sc;
                }
                assert_eq!(v["hits"]["max_score"].as_f64().unwrap(), hits[0]["_score"].as_f64().unwrap());
                // A page of the window.
                let mut b = body(seed, Some(2));
                b["from"] = json!(1);
                let p = search(&s, b);
                let ids = |v: &Value| v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].clone()).collect::<Vec<_>>();
                assert_eq!(ids(&p), ids(&v)[1..3].to_vec(), "{p}");
                assert_eq!(p["hits"]["max_score"], hits[1]["_score"]);
                if !split {
                    // The compound path (no explicit size) on one segment.
                    let c = search(&s, body(seed, None));
                    assert_eq!(ids(&c), ids(&v));
                }
            }
        }
    }
}
