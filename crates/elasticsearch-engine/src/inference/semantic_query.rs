//! Queries on `semantic_text` fields: the `semantic` query (and `match`),
//! and ES's query rewrite interceptors that turn `sparse_vector` / `knn`
//! queries on a `semantic_text` field, or `nested` queries over its
//! `<field>.inference.chunks`, into chunk queries ([`ChunkQuery`]).

use super::chunk_query::{ChunkQuery, ChunkVec};
use super::semantic::DEFAULT_INFERENCE_ID;
use super::Embedded;
use crate::error::{EsError, Result};
use crate::mapping::ResolvedField;
use crate::query::dsl::QueryContext;
use rustic_core::nested::NestedScoreMode;
use rustic_core::query::Query;
use rustic_core::vector::Similarity;
use serde_json::{Map, Value};
use std::sync::Arc;

fn params(f: &ResolvedField) -> Map<String, Value> {
    f.ext.as_ref().map(|e| (*e.params).clone()).unwrap_or_default()
}

fn inference_id(p: &Map<String, Value>) -> String {
    p.get("inference_id").and_then(Value::as_str).unwrap_or(DEFAULT_INFERENCE_ID).to_string()
}

fn search_inference_id(p: &Map<String, Value>) -> String {
    p.get("search_inference_id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| inference_id(p))
}

/// ES 9 semantics for this request.
fn v9() -> bool {
    !crate::version::current_rest_api_version().is_v8()
}

/// The task type of the field's model: its `model_settings`, else its
/// inference endpoint's.
fn field_task(p: &Map<String, Value>) -> Option<&'static str> {
    let t = super::semantic_map::resolved_model_settings(p)?.get("task_type").and_then(Value::as_str).map(str::to_string)?;
    super::providers::Task::parse(&t).map(super::providers::Task::name)
}

fn model_settings(p: &Map<String, Value>) -> Option<Value> {
    p.get("model_settings").filter(|v| !v.is_null()).cloned()
}

fn similarity(ms: &Value) -> Similarity {
    match ms.get("similarity").and_then(Value::as_str).unwrap_or("cosine") {
        "dot_product" => Similarity::DotProduct,
        "l2_norm" => Similarity::L2,
        "max_inner_product" => Similarity::MaxInnerProduct,
        _ => Similarity::Cosine,
    }
}

fn aux(f: &ResolvedField) -> Option<u16> {
    f.aux_id("inference")
}

/// `generateInvalidQueryInferenceResultsMessage`.
fn mismatch_suffix(p: &Map<String, Value>) -> String {
    let id = inference_id(p);
    match p.get("search_inference_id").and_then(Value::as_str) {
        Some(s) if s != id => format!(" Is the search inference endpoint [{s}] compatible with the inference endpoint [{id}]?"),
        _ => format!(" Has the configuration for inference endpoint [{id}] changed?"),
    }
}

/// Embed `text` with endpoint `id` (`Inference endpoint not found` when unknown).
fn embed(id: &str, text: &str) -> Result<Embedded> {
    let svc = super::current().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
    if svc.get_endpoint(id).is_none() {
        return Err(EsError::resource_not_found(format!("Inference endpoint not found [{id}]")));
    }
    svc.embed_typed(id, &[text.to_string()], "search")
}

fn sparse_tokens(m: &Map<String, Value>) -> Vec<(String, f32)> {
    m.iter().filter_map(|(k, v)| v.as_f64().map(|w| (k.clone(), w as f32))).collect()
}

thread_local! {
    /// The search request's `size` when given (`SearchExecutionContext.requestSize`).
    static REQUEST_SIZE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with the request size dense semantic queries size their kNN by.
pub fn with_request_size<T>(size: Option<usize>, f: impl FnOnce() -> T) -> T {
    let prev = REQUEST_SIZE.with(|c| c.replace(size));
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            REQUEST_SIZE.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(prev);
    f()
}

/// `AutoPrefilteringUtils.pruneQuery` with the nested types ES prunes
/// (`nested`, and the semantic queries its interceptors turn into nested
/// queries); `None` when nothing is left.
fn prune_prefilter(v: &Value, ctx: &QueryContext) -> Option<Value> {
    let Some((kind, body)) = v.as_object().and_then(|o| o.iter().next()) else { return Some(v.clone()) };
    let on_inference = |f: &str| ctx.lookup.get(f).is_some_and(|x| x.declared.is_semantic());
    match kind.as_str() {
        "nested" => None,
        "match" if body.as_object().and_then(|m| m.keys().next()).is_some_and(|f| on_inference(f)) => None,
        "semantic" | "sparse_vector" | "knn" if body.get("field").and_then(Value::as_str).is_some_and(on_inference) => None,
        "bool" => {
            let b = body.as_object()?;
            let mut out = serde_json::Map::new();
            let mut removed_should = 0usize;
            for k in ["must", "should", "filter", "must_not"] {
                let list: Vec<&Value> = match b.get(k) {
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(Value::Null) | None => continue,
                    Some(q) => vec![q],
                };
                let kept: Vec<Value> = list.iter().filter_map(|q| prune_prefilter(q, ctx)).collect();
                if k == "should" {
                    removed_should = list.len() - kept.len();
                }
                if !kept.is_empty() {
                    out.insert(k.into(), Value::Array(kept));
                }
            }
            if out.is_empty() {
                return None;
            }
            if let Some(m) = b.get("minimum_should_match") {
                // `adjustMinimumShouldMatchForPrunedShouldClauses`.
                match (m.as_i64(), removed_should) {
                    (Some(n), r) if r > 0 => {
                        out.insert("minimum_should_match".into(), serde_json::json!((n - r as i64).max(0)));
                    }
                    _ => {
                        out.insert("minimum_should_match".into(), m.clone());
                    }
                }
            }
            Some(serde_json::json!({ "bool": out }))
        }
        "constant_score" => {
            let inner = prune_prefilter(body.get("filter")?, ctx)?;
            Some(serde_json::json!({ "constant_score": { "filter": inner } }))
        }
        "boosting" => prune_prefilter(body.get("positive")?, ctx),
        "dis_max" => {
            let kept: Vec<Value> = body.get("queries")?.as_array()?.iter().filter_map(|q| prune_prefilter(q, ctx)).collect();
            (!kept.is_empty()).then(|| serde_json::json!({ "dis_max": { "queries": kept } }))
        }
        "function_score" => {
            let mut b = body.as_object()?.clone();
            if let Some(q) = b.get("query").cloned() {
                b.insert("query".into(), prune_prefilter(&q, ctx)?);
            }
            Some(serde_json::json!({ "function_score": b }))
        }
        _ => Some(v.clone()),
    }
}

/// The auto-prefilters of a dense semantic query (ES 9): the enclosing
/// bools' sibling clauses, nested / semantic branches pruned.
fn auto_prefilter(ctx: &QueryContext) -> Result<Option<Query>> {
    let mut filters: Vec<Query> = Vec::new();
    for v in crate::query::dsl::auto_prefilters() {
        if let Some(p) = prune_prefilter(&v, ctx) {
            let q = crate::query::dsl::parse_query(&p, ctx)?;
            if !matches!(q, Query::MatchAll) {
                filters.push(q);
            }
        }
    }
    if let Some(filter) = ctx.knn_prefilter_query()? {
        filters.push(filter);
    }
    Ok(match filters.len() {
        0 => None,
        1 => filters.pop(),
        _ => Some(Query::Bool { must: vec![], should: vec![], must_not: vec![], filter: filters, minimum_should_match: rustic_core::query::MinShouldMatch::Count(0) }),
    })
}

/// `SemanticTextFieldType.semanticQuery`: the field's chunks scored against
/// the search inference results of `text` (`nested`, `score_mode: max`).
pub fn semantic_query(f: &ResolvedField, text: &str, ctx: Option<&QueryContext>) -> Result<Query> {
    let p = params(f);
    let search_id = search_inference_id(&p);
    let emb = embed(&search_id, text)?;
    let Some(ms) = model_settings(&p) else { return Ok(Query::MatchNone) };
    let Some(aux) = aux(f) else { return Ok(Query::MatchNone) };
    let task = ms.get("task_type").and_then(Value::as_str).unwrap_or("");
    let vec = match (task, emb) {
        ("sparse_embedding", Embedded::Sparse(mut v)) => {
            let tokens = sparse_tokens(&v.pop().unwrap_or_default());
            match ctx {
                Some(c) => ChunkVec::Sparse(Arc::new(pruned(&p, None, aux, c, tokens)?)),
                None => ChunkVec::Sparse(Arc::new(tokens)),
            }
        }
        ("sparse_embedding", Embedded::Dense(_)) => {
            return Err(EsError::illegal_argument(format!(
                "Field [{}] expected query inference results to be of type [text_expansion_result], got [text_embedding_result].{}",
                f.name,
                mismatch_suffix(&p)
            )))
        }
        ("text_embedding" | "embedding", Embedded::Dense(mut v)) => {
            let v = v.pop().unwrap_or_default();
            let et = ms.get("element_type").and_then(Value::as_str).unwrap_or("float");
            let dims = if et == "bit" { v.len() * 8 } else { v.len() };
            let want = ms.get("dimensions").and_then(Value::as_u64).unwrap_or(0) as usize;
            if dims != want {
                return Err(EsError::illegal_argument(format!(
                    "Field [{}] expected query inference results with {want} dimensions, got {dims} dimensions.{}",
                    f.name,
                    mismatch_suffix(&p)
                )));
            }
            // `SemanticFieldType.semanticQuery`: a kNN over the chunks with
            // `k = max(request size, 10)` (10 when unset), auto-prefiltered (9.x).
            let k = REQUEST_SIZE.with(|c| c.get()).map(|s| s.max(10)).unwrap_or(10);
            let filter = match ctx {
                Some(c) if v9() => auto_prefilter(c)?,
                Some(c) => c.knn_prefilter_query()?,
                None => None,
            };
            let q = ChunkQuery { field: f.name.clone(), aux, vec: ChunkVec::Dense { v: Arc::new(v), sim: similarity(&ms), threshold: None }, mode: NestedScoreMode::Max, k: Some(k), filter: filter.map(Box::new), fixed: None };
            return Ok(Query::Custom(Arc::new(match ctx {
                Some(c) => q.with_top_k(c.leaves)?,
                None => q,
            })));
        }
        ("text_embedding" | "embedding", Embedded::Sparse(_)) => {
            return Err(EsError::illegal_argument(format!(
                "Field [{}] expected query inference results to be of type [text_embedding_result], got [text_expansion_result].{}",
                f.name,
                mismatch_suffix(&p)
            )))
        }
        _ => return Ok(Query::MatchNone),
    };
    Ok(Query::Custom(Arc::new(ChunkQuery { field: f.name.clone(), aux, vec, mode: NestedScoreMode::Max, k: None, filter: None, fixed: None })))
}

/// `SemanticSparseVectorQueryRewriteInterceptor`: `sparse_vector` on a
/// `semantic_text` field.
/// `SparseVectorFieldType.finalizeSparseVectorQuery` on the chunk
/// embeddings: the query's `prune` / `pruning_config`, else the field's
/// `index_options.sparse_vector` (default: prune with the default config).
/// No tokens when the field holds none (`EMPTY_FIELD`; kept a chunk query,
/// not `match_none`, so can-match phases still see a semantic query).
fn pruned(p: &Map<String, Value>, spec: Option<&Map<String, Value>>, aux: u16, ctx: &QueryContext, tokens: Vec<(String, f32)>) -> Result<Vec<(String, f32)>> {
    use crate::fieldtypes::query::pruning_config;
    let mut prune = spec.and_then(|s| s.get("prune")).and_then(Value::as_bool);
    let mut config = spec.and_then(|s| s.get("pruning_config")).filter(|v| !v.is_null()).map(pruning_config).transpose()?;
    let io = p.get("index_options").and_then(|io| io.get("sparse_vector"));
    let (ip, ic) = match io.and_then(Value::as_object) {
        None => (Some(true), None),
        Some(o) => (o.get("prune").and_then(Value::as_bool), o.get("pruning_config").filter(|v| !v.is_null()).map(pruning_config).transpose()?),
    };
    if prune.is_none() {
        prune = ip;
    }
    if config.is_none() {
        config = ic;
    }
    if prune != Some(true) || ctx.leaves.is_empty() {
        return Ok(tokens);
    }
    let c = config.unwrap_or_default();
    Ok(super::chunk_query::prune_tokens(ctx.leaves, aux, tokens, c.freq_ratio, c.weight, c.only_pruned).unwrap_or_default())
}

pub fn sparse_vector(f: &ResolvedField, spec: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let p = params(f);
    // ES 9 (`InterceptedInferenceQueryBuilder`): the field's model is checked
    // before any inference.
    if v9() && field_task(&p).is_some_and(|t| t != "sparse_embedding") {
        return Err(EsError::illegal_argument(format!("Field [{}] does not use a [sparse_embedding] model", f.name)));
    }
    let Some(aux) = aux(f) else { return Ok(Query::MatchNone) };
    let tokens = match spec.get("query_vector").and_then(Value::as_object) {
        Some(v) => sparse_tokens(v),
        None => {
            let id = spec.get("inference_id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| search_inference_id(&p));
            let text = spec.get("query").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default();
            match embed(&id, &text)? {
                Embedded::Sparse(mut v) => sparse_tokens(&v.pop().unwrap_or_default()),
                Embedded::Dense(_) => return Err(EsError::illegal_argument(format!("Field [{}] expected query inference results to be of type [text_expansion_result], got [text_embedding_result].{}", f.name, mismatch_suffix(&p)))),
            }
        }
    };
    if model_settings(&p).is_none() {
        return Ok(Query::MatchNone);
    }
    let tokens = pruned(&p, Some(spec), aux, ctx, tokens)?;
    Ok(Query::Custom(Arc::new(ChunkQuery { field: f.name.clone(), aux, vec: ChunkVec::Sparse(Arc::new(tokens)), mode: NestedScoreMode::Max, k: None, filter: None, fixed: None })))
}

/// The query vector of a `knn` on a `semantic_text` field: `query_vector`,
/// or `query_vector_builder.text_embedding` (`model_id` defaults to the
/// field's search inference id).
fn knn_vector(f: &ResolvedField, o: &Map<String, Value>) -> Result<Vec<f32>> {
    if v9() {
        let p = params(f);
        if let Some(t) = field_task(&p).filter(|t| !matches!(*t, "text_embedding" | "embedding")) {
            return Err(EsError::illegal_argument(format!(
                "Field [{}] requires an embedding or text embedding model, but the configured model type is [{t}] which is not compatible with knn queries",
                f.name
            )));
        }
    }
    if let Some(v) = o.get("query_vector").and_then(Value::as_array) {
        return Ok(v.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect());
    }
    if let Some(e) = o.get("query_vector_builder").and_then(|b| b.get("embedding")) {
        // `InferenceQueryUtils`: the field's endpoint embeds the input.
        let (id, inputs) = super::semantic::parse_embedding_builder(e)?;
        let p = params(f);
        let id = id.unwrap_or_else(|| search_inference_id(&p));
        let task = super::current().and_then(|s| s.get_endpoint(&id)).map(|ep| ep.task_type);
        if let Some(t) = task.filter(|t| matches!(t, super::providers::Task::TextEmbedding | super::providers::Task::SparseEmbedding)) {
            if inputs.iter().any(|(ty, _)| ty != "text") {
                return Err(EsError::illegal_argument(format!("Non-text input is not supported for [{}] inference endpoints for inference_id [{id}]", t.name())));
            }
            if inputs.len() > 1 {
                return Err(EsError::illegal_argument(format!("Multiple text inputs are not supported for [{}] inference endpoints for inference_id [{id}]", t.name())));
            }
            return match embed(&id, &inputs[0].1)? {
                Embedded::Dense(mut v) => Ok(v.pop().unwrap_or_default()),
                Embedded::Sparse(_) => Err(EsError::illegal_argument(format!("Field [{}] expected query inference results to be of type [text_embedding_result], got [text_expansion_result].{}", f.name, mismatch_suffix(&p)))),
            };
        }
        return super::semantic::embedding_vector(&id, &inputs);
    }
    let te = o.get("query_vector_builder").and_then(|b| b.get("text_embedding")).and_then(Value::as_object);
    let Some(te) = te else {
        return Err(EsError::illegal_argument("either [query_vector] or [query_vector_builder] must be provided"));
    };
    let p = params(f);
    let id = te.get("model_id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| search_inference_id(&p));
    let text = te.get("model_text").and_then(Value::as_str).unwrap_or_default();
    match embed(&id, text)? {
        Embedded::Dense(mut v) => Ok(v.pop().unwrap_or_default()),
        Embedded::Sparse(_) => Err(EsError::illegal_argument(format!("Field [{}] expected query inference results to be of type [text_embedding_result], got [text_expansion_result].{}", f.name, mismatch_suffix(&p)))),
    }
}

/// `SemanticKnnVectorQueryRewriteInterceptor`: `knn` on a `semantic_text`
/// field (a nested kNN over its chunks: the `k` best docs by best chunk).
pub fn knn(f: &ResolvedField, o: &Map<String, Value>, ctx: &QueryContext, k: usize) -> Result<Query> {
    let p = params(f);
    let v = knn_vector(f, o)?;
    let Some(ms) = model_settings(&p) else { return Ok(Query::MatchNone) };
    let Some(aux) = aux(f) else { return Ok(Query::MatchNone) };
    if !matches!(ms.get("task_type").and_then(Value::as_str), Some("text_embedding" | "embedding")) {
        return Err(EsError::illegal_argument(format!("Field [{}] is not a text_embedding semantic_text field", f.name)));
    }
    let dims = ms.get("dimensions").and_then(Value::as_u64).unwrap_or(0) as usize;
    if dims != v.len() {
        return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("failed to create query: The query vector has a different number of dimensions [{}] than the document vectors [{dims}].", v.len())));
    }
    let sim_name = ms.get("similarity").and_then(Value::as_str).unwrap_or("cosine").to_string();
    let threshold = o.get("similarity").and_then(Value::as_f64).map(|s| crate::vectors::score_of_similarity(s as f32, &sim_name, crate::vectors::ElementType::Float, dims));
    let filter = knn_filter(o, ctx)?;
    let q = ChunkQuery { field: f.name.clone(), aux, vec: ChunkVec::Dense { v: Arc::new(v), sim: similarity(&ms), threshold }, mode: NestedScoreMode::Max, k: Some(k), filter: filter.map(Box::new), fixed: None };
    Ok(Query::Custom(Arc::new(q.with_top_k(ctx.leaves)?)))
}

/// Whether a query (JSON) holds a query the semantic interceptors turn into
/// a `nested` query over `semantic_text` chunks (`match`, `knn`,
/// `sparse_vector`, `semantic` on an inference field).
fn has_semantic_query(v: &Value, ctx: &QueryContext) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, b)| {
            let on_inference = |f: &str| ctx.lookup.get(f).is_some_and(|x| x.declared.is_semantic());
            let hit = match k.as_str() {
                "match" => b.as_object().and_then(|m| m.keys().next()).is_some_and(|f| on_inference(f)),
                "knn" | "sparse_vector" | "semantic" => b.get("field").and_then(Value::as_str).is_some_and(on_inference),
                _ => false,
            };
            hit || has_semantic_query(b, ctx)
        }),
        Value::Array(a) => a.iter().any(|x| has_semantic_query(x, ctx)),
        _ => false,
    }
}

fn knn_filter(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Option<Query>> {
    // ES 9 intercepts the filters too (`SEMANTIC_KNN_VECTOR_QUERY_FILTERS_REWRITE_INTERCEPTION_SUPPORTED`):
    // a semantic (nested) filter of the nested chunk kNN applies at the
    // chunk level and matches nothing (elastic/elasticsearch#138184).
    if !crate::version::current_rest_api_version().is_v8() && o.get("filter").is_some_and(|f| has_semantic_query(f, ctx)) {
        return Ok(Some(Query::MatchNone));
    }
    let mut filters: Vec<Query> = match o.get("filter") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().map(|q| crate::query::dsl::parse_query(q, ctx)).collect::<Result<Vec<_>>>()?,
        Some(q) => vec![crate::query::dsl::parse_query(q, ctx)?],
    };
    if let Some(filter) = ctx.knn_prefilter_query()? {
        filters.push(filter);
    }
    Ok(match filters.len() {
        0 => None,
        1 => filters.pop(),
        _ => Some(Query::Bool { must: vec![], should: vec![], must_not: vec![], filter: filters, minimum_should_match: rustic_core::query::MinShouldMatch::Count(0) }),
    })
}

/// The `semantic_text` field whose chunks `path` names
/// (`<field>.inference.chunks`).
pub fn chunks_field<'a>(ctx: &'a QueryContext, path: &str) -> Option<&'a Arc<ResolvedField>> {
    let field = path.strip_suffix(".inference.chunks")?;
    ctx.lookup.get(field).filter(|f| f.declared.is_semantic())
}

/// A `nested` query over a `semantic_text` field's chunks: its inner
/// `sparse_vector` / `knn` / `match_all` on `<field>.inference.chunks.embeddings`.
pub fn nested(f: &ResolvedField, inner: &Value, mode: NestedScoreMode, ctx: &QueryContext) -> Result<Query> {
    let p = params(f);
    let Some(aux) = aux(f) else { return Ok(Query::MatchNone) };
    let emb_field = format!("{}.inference.chunks.embeddings", f.name);
    let Some((kind, body)) = inner.as_object().and_then(|o| o.iter().next()) else {
        return Err(EsError::parsing("[nested] query malformed"));
    };
    let body = body.as_object().cloned().unwrap_or_default();
    let field_of = |b: &Map<String, Value>| b.get("field").and_then(Value::as_str).map(str::to_string);
    let wrap = |vec: ChunkVec, k: Option<usize>, filter: Option<Query>| -> Result<Query> {
        Ok(Query::Custom(Arc::new(ChunkQuery { field: f.name.clone(), aux, vec, mode, k, filter: filter.map(Box::new), fixed: None }.with_top_k(ctx.leaves)?)))
    };
    let boost = body.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let boosted = |q: Query| if boost != 1.0 { Query::Boost { query: Box::new(q), boost } } else { q };
    if model_settings(&p).is_none() {
        return Ok(Query::MatchNone);
    }
    match kind.as_str() {
        "match_all" => Ok(boosted(wrap(ChunkVec::All, None, None)?)),
        "sparse_vector" if field_of(&body).as_deref() == Some(emb_field.as_str()) => {
            let tokens = match body.get("query_vector").and_then(Value::as_object) {
                Some(v) => sparse_tokens(v),
                None => {
                    let id = body.get("inference_id").and_then(Value::as_str).map(str::to_string).ok_or_else(|| EsError::illegal_argument("inference_id required to perform vector search on query string"))?;
                    let text = body.get("query").and_then(Value::as_str).unwrap_or_default();
                    match embed(&id, text)? {
                        Embedded::Sparse(mut v) => sparse_tokens(&v.pop().unwrap_or_default()),
                        Embedded::Dense(_) => return Err(EsError::illegal_argument(format!("Inference endpoint [{id}] is not a sparse_embedding endpoint"))),
                    }
                }
            };
            Ok(boosted(wrap(ChunkVec::Sparse(Arc::new(tokens)), None, None)?))
        }
        "knn" if field_of(&body).as_deref() == Some(emb_field.as_str()) => {
            let ms = model_settings(&p).unwrap_or(Value::Null);
            let v: Vec<f32> = match body.get("query_vector").and_then(Value::as_array) {
                Some(a) => a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect(),
                None => {
                    let resolved = super::semantic::resolve_knn_builder(&body, body.get("query_vector_builder").unwrap_or(&Value::Null))?;
                    resolved.get("query_vector").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()).unwrap_or_default()
                }
            };
            let k = body.get("k").and_then(Value::as_u64).map(|k| k as usize).or_else(|| body.get("num_candidates").and_then(Value::as_u64).map(|n| (n as usize).min(10))).unwrap_or(10);
            let sim_name = ms.get("similarity").and_then(Value::as_str).unwrap_or("cosine").to_string();
            let threshold = body.get("similarity").and_then(Value::as_f64).map(|s| crate::vectors::score_of_similarity(s as f32, &sim_name, crate::vectors::ElementType::Float, v.len()));
            let filter = knn_filter(&body, ctx)?;
            Ok(boosted(wrap(ChunkVec::Dense { v: Arc::new(v), sim: similarity(&ms), threshold }, Some(k), filter)?))
        }
        _ => Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("failed to create query: [{kind}] queries are not supported on the chunks of semantic_text field [{}]", f.name))),
    }
}
