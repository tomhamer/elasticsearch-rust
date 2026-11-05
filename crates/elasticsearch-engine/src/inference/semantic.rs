//! `semantic_text` support: document-level inference (the
//! `ShardBulkInferenceActionFilter` step: chunk + embed every value of the
//! field and of its `copy_to` sources), the per-field inference blob (the
//! field's `#inference` doc-values / stored companion) and query-time
//! scoring of those blobs.
//!
//! Blob (one per field per doc): `{"k":"d"|"s","c":[[source_field,start,end],..],
//! "v":[chunk vectors | {token:w}]}`; offsets are UTF-16 (Java) offsets into
//! the source field's values joined with one separator character.

use super::{local, Embedded, Endpoint};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup};
use rustic_core::fieldtypes::filter::FilterQuery;
use rustic_core::query::{LeafReader, Query};
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub const DEFAULT_INFERENCE_ID: &str = ".elser-2-elasticsearch";

/// A field the index's inference metadata lists (`IndexMetadata.getInferenceFields`):
/// a top-level (non multi-field) `semantic_text` field.
pub fn is_inference_field(f: &crate::mapping::ResolvedField) -> bool {
    f.declared.is_semantic() && !f.is_multi_field
}

fn status(reason: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", reason)
}

/// Java `String.length()` (UTF-16 code units).
fn jlen(s: &str) -> usize {
    s.encode_utf16().count()
}

/// Java `String.isBlank()`.
fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// `SemanticTextUtils.nodeStringValues`: the scalar values of a field as
/// strings (objects / nested arrays are rejected).
fn node_string_values(field: &str, v: &Value) -> Result<Vec<String>> {
    check_input(field, v)?;
    Ok(match v {
        Value::Array(a) => a.iter().filter(|x| !x.is_null()).map(scalar_string).collect(),
        Value::Null => Vec::new(),
        other => vec![scalar_string(other)],
    })
}

fn scalar_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        o => o.to_string(),
    }
}

/// The value at a dotted path of a source (objects walked by path parts or
/// dotted keys); `Some(Value::Null)` for an explicit null.
fn extract_value<'a>(src: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    if let Some(v) = src.get(path) {
        return Some(v);
    }
    for (i, _) in path.match_indices('.') {
        if let Some(Value::Object(o)) = src.get(&path[..i]) {
            if let Some(v) = extract_value(o, &path[i + 1..]) {
                return Some(v);
            }
        }
    }
    None
}

/// The field's source paths: for a multi-field, its parent field.
fn source_path(lookup: &FieldLookup, name: &str) -> String {
    match lookup.fields.get(name) {
        Some(f) if f.is_multi_field => name.rsplit_once('.').map(|x| x.0.to_string()).unwrap_or_else(|| name.to_string()),
        _ => name.to_string(),
    }
}

/// UTF-16 offset (Java `String` index) of byte offset `b` in `s`.
fn utf16_at(s: &str, b: usize) -> usize {
    s.get(..b.min(s.len())).map(|p| p.encode_utf16().count()).unwrap_or(b)
}

/// Run the inference of every `semantic_text` field of a document that does
/// not carry its results yet: the `_inference_fields` value of each field
/// (`{inference: {inference_id, model_settings, chunks: {source: [..]}}}`).
pub fn infer_document(root: &Value, lookup: &FieldLookup, legacy: bool) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    if !lookup.may_have_semantic_fields { return Ok(out); }
    let Some(root) = root.as_object() else { return Ok(out) };
    let mut names: Vec<&String> = lookup.fields.iter().filter(|(_, f)| f.declared.is_semantic()).map(|(n, _)| n).collect();
    if names.is_empty() {
        return Ok(out);
    }
    names.sort();
    let explicit = supplied_of(root);
    for name in names {
        if explicit.contains_key(name.as_str()) {
            continue;
        }
        let f = &lookup.fields[name];
        let own = source_path(lookup, name);
        let own_value = extract_value(root, &own);
        if legacy && (own_value.is_some_and(Value::is_object) || own_value.is_some_and(is_legacy_value)) {
            continue;
        }
        let mut sources: Vec<String> = vec![own.clone()];
        if let Some(cs) = lookup.copy_sources.get(name) {
            sources.extend(cs.iter().cloned());
        }
        sources.sort();
        sources.dedup();
        // (source field, input, offset adjustment)
        let mut requests: Vec<(String, String, usize)> = Vec::new();
        // Each request's value index in its source field (input order).
        let mut req_index: Vec<usize> = Vec::new();
        // `semantic` media inputs: (source field, value, input index).
        let mut media: Vec<(String, String, usize)> = Vec::new();
        let mut any = false;
        for src in &sources {
            let Some(v) = extract_value(root, src) else { continue };
            if v.is_null() {
                continue;
            }
            any = true;
            let mut adj = 0usize;
            if f.declared == EsType::Semantic && *src == own {
                for (i, (ty, t)) in semantic_inputs(v).into_iter().enumerate() {
                    if ty == "text" {
                        if !is_blank(&t) {
                            requests.push((src.clone(), t.clone(), adj));
                            req_index.push(i);
                        }
                        adj += jlen(&t) + 1;
                    } else {
                        if t.starts_with("data:") && t.split_once("base64,").is_some_and(|(_, p)| p.is_empty()) {
                            return Err(EsError::new(400, "status_exception", format!("Input for field [{name}] from source field [{src}] has an empty base64 payload.")));
                        }
                        media.push((src.clone(), t, i));
                    }
                }
                continue;
            }
            for (i, t) in node_string_values(name, v)?.into_iter().enumerate() {
                if !is_blank(&t) {
                    requests.push((src.clone(), t.clone(), adj));
                    req_index.push(i);
                }
                adj += jlen(&t) + 1;
            }
        }
        if !any {
            continue;
        }
        let params = f.ext.as_ref().map(|e| (*e.params).clone()).unwrap_or_default();
        let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(DEFAULT_INFERENCE_ID).to_string();
        let Some(svc) = super::current() else { continue };
        let Some(ep) = svc.get_endpoint(&id) else {
            return Err(EsError::resource_not_found(format!("Inference id [{id}] not found for field [{name}]")));
        };
        let field_chunking = params.get("chunking_settings").filter(|v| !v.is_null());
        let chunking = local::Chunking::from_json(field_chunking.or(ep.chunking_settings.as_ref()));
        // The test services embed each value whole unless chunking is set.
        let whole = field_chunking.is_none() && ep.chunking_settings.is_none() && super::mock::is_mock(&ep.service);
        // (source field, start, end) or, for media, (source field, input index, usize::MAX).
        let mut spans: Vec<(String, usize, usize)> = Vec::new();
        let mut order: Vec<usize> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        for ((src, t, adj), vi) in requests.iter().zip(&req_index) {
            let offs = if whole { vec![(0, t.len())] } else { local::chunk(t, &chunking) };
            for (s, e) in offs {
                inputs.push(t[s..e].to_string());
                spans.push((src.clone(), adj + utf16_at(t, s), adj + utf16_at(t, e)));
                order.push(*vi);
            }
        }
        for (src, value, i) in &media {
            inputs.push(value.clone());
            spans.push((src.clone(), *i, usize::MAX));
            order.push(*i);
        }
        let mut chunks = Map::new();
        for src in &sources {
            if extract_value(root, src).is_some_and(|v| !v.is_null()) {
                chunks.insert(src.clone(), json!([]));
            }
        }
        let mut settings = Value::Null;
        if !inputs.is_empty() {
            let emb = svc.embed_typed(&id, &inputs, "ingest").map_err(|e| {
                EsError::new(e.status, "inference_exception", format!("Exception when running inference id [{id}] on field [{name}]")).caused_by(e)
            })?;
            let vecs: Vec<Value> = match emb {
                Embedded::Dense(v) => v.into_iter().map(|x| json!(x)).collect(),
                Embedded::Sparse(v) => v.into_iter().map(Value::Object).collect(),
            };
            let mut rows: Vec<((String, usize, usize), Value, usize)> = spans.into_iter().zip(vecs).zip(order).map(|((sp, v), o)| (sp, v, o)).collect();
            // Chunks in input order: a `semantic` field's text and media
            // inputs interleave by value position.
            if !media.is_empty() {
                rows.sort_by_key(|r| r.2);
            }
            for ((src, s, e), v, _) in rows {
                if let Some(Value::Array(a)) = chunks.get_mut(&src) {
                    if e == usize::MAX {
                        a.push(json!({"input_index": s, "embeddings": v}));
                    } else {
                        a.push(json!({"start_offset": s, "end_offset": e, "embeddings": v}));
                    }
                }
            }
            settings = super::model_settings(&ep).unwrap_or(Value::Null);
        }
        out.insert(name.clone(), json!({"inference": {"inference_id": id, "model_settings": settings, "chunks": chunks}}));
    }
    Ok(out)
}

fn dense_sim(sim: &str) -> rustic_core::vector::Similarity {
    use rustic_core::vector::Similarity::*;
    match sim {
        "dot_product" => DotProduct,
        "l2_norm" => L2,
        "max_inner_product" => MaxInnerProduct,
        _ => Cosine,
    }
}

/// A scoring query over the `#inference` doc-values field `aux_id`.
pub fn query(aux_id: u16, search_ep: &Endpoint, text: &str) -> Result<Query> {
    let svc = super::current().ok_or_else(|| status("inference service is not available"))?;
    let emb = svc.embed_typed(&search_ep.inference_id, &[text.to_string()], "search")?;
    match emb {
        Embedded::Dense(mut v) => {
            let q = v.pop().unwrap_or_default();
            let sim = dense_sim(search_ep.service_settings.get("similarity").and_then(Value::as_str).unwrap_or(if matches!(search_ep.service.as_str(), "elasticsearch" | "elser") || super::mock::is_mock(&search_ep.service) { "cosine" } else { "dot_product" }));
            let q = Arc::new(q);
            Ok(FilterQuery::new(
                "semantic",
                None,
                Arc::new(move |leaf: &LeafReader<'_>, d: DocId| {
                    let mut best: Option<f32> = None;
                    for raw in leaf.keyword_values(aux_id, d) {
                        let Ok(b) = serde_json::from_str::<Value>(&raw) else { continue };
                        if b["k"] != "d" {
                            continue;
                        }
                        for c in b["v"].as_array().into_iter().flatten() {
                            let v: Vec<f32> = c.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()).unwrap_or_default();
                            if v.len() == q.len() {
                                let s = sim.score(&q, &v);
                                best = Some(best.map_or(s, |m| m.max(s)));
                            }
                        }
                    }
                    best
                }),
            )
            .into_query())
        }
        Embedded::Sparse(mut v) => {
            let q: Map<String, Value> = v.pop().unwrap_or_default();
            let q: Vec<(String, f32)> = q.into_iter().filter_map(|(k, w)| w.as_f64().map(|w| (k, w as f32))).collect();
            let q = Arc::new(q);
            Ok(FilterQuery::new(
                "semantic",
                None,
                Arc::new(move |leaf: &LeafReader<'_>, d: DocId| {
                    let mut best: Option<f32> = None;
                    for raw in leaf.keyword_values(aux_id, d) {
                        let Ok(b) = serde_json::from_str::<Value>(&raw) else { continue };
                        if b["k"] != "s" {
                            continue;
                        }
                        for c in b["v"].as_array().into_iter().flatten() {
                            let mut s = 0f32;
                            let mut hit = false;
                            for (t, qw) in q.iter() {
                                if let Some(w) = c.get(t).and_then(Value::as_f64) {
                                    // Lucene `FeatureField` keeps 9 significant bits of the weight.
                                    s += qw * f32::from_bits(((w as f32).to_bits() >> 15) << 15);
                                    hit = true;
                                }
                            }
                            if hit {
                                best = Some(best.map_or(s, |m| m.max(s)));
                            }
                        }
                    }
                    best
                }),
            )
            .into_query())
        }
    }
}

/// One `InferenceString` of an `embedding` query vector builder: `(type, value)`.
pub type InferenceInput = (String, String);

/// `EmbeddingQueryVectorBuilder.PARSER` (ES 9.4+ `query_vector_builder.embedding`):
/// `(inference_id, input)` with the parser's failure chain.
pub fn parse_embedding_builder(v: &Value) -> Result<(Option<String>, Vec<InferenceInput>)> {
    let qvb = |e: EsError| EsError::x_content_parse("[1:1] [knn] failed to parse field [query_vector_builder]").caused_by(EsError::x_content_parse("[1:1] [query_vector_builder] failed to parse field [embedding]").caused_by(e));
    let input_err = |e: EsError| qvb(EsError::x_content_parse("[1:1] [embedding] failed to parse field [input]").caused_by(e));
    let o = v.as_object().ok_or_else(|| qvb(EsError::illegal_argument("[embedding] must be an object")))?;
    let id = o.get("inference_id").and_then(Value::as_str).map(str::to_string);
    let Some(input) = o.get("input").filter(|v| !v.is_null()) else {
        return Err(qvb(EsError::illegal_argument("Required [input]")));
    };
    let one = |x: &Value| -> Result<InferenceInput> {
        match x {
            Value::String(s) => Ok(("text".into(), s.clone())),
            Value::Object(m) => {
                let ty = m.get("type").and_then(Value::as_str);
                let val = m.get("value").and_then(Value::as_str);
                let field = |f: &str, e: EsError| input_err(EsError::x_content_parse(format!("[1:1] [InferenceString] failed to parse field [{f}]")).caused_by(e));
                if let Some(t) = ty {
                    if !matches!(t, "text" | "image" | "audio" | "video" | "pdf") {
                        return Err(field("type", EsError::illegal_argument(format!("Unrecognized type [{t}], must be one of [text, image, audio, video, pdf]"))));
                    }
                }
                if let Some(f) = m.get("format").and_then(Value::as_str) {
                    if !matches!(f, "text" | "base64") {
                        return Err(field("format", EsError::illegal_argument(format!("Unrecognized format [{f}], must be one of [text, base64]"))));
                    }
                    let supported: &[&str] = if ty == Some("text") { &["text"] } else { &["base64"] };
                    if let Some(t) = ty.filter(|_| !supported.contains(&f)) {
                        return Err(field("value", EsError::x_content_parse("Failed to build [InferenceString] after last required field arrived").caused_by(EsError::illegal_argument(format!(
                            "Data type [{t}] does not support data format [{f}], supported formats are [{}]",
                            supported.join(", ")
                        )))));
                    }
                }
                let missing: Vec<&str> = [("type", ty.is_none()), ("value", val.is_none())].iter().filter(|x| x.1).map(|x| x.0).collect();
                if !missing.is_empty() {
                    return Err(input_err(EsError::illegal_argument(format!("Required [{}]", missing.join(", ")))));
                }
                Ok((ty.unwrap_or("text").to_string(), val.unwrap_or("").to_string()))
            }
            _ => Err(input_err(EsError::illegal_argument("[input] must be a string, an object or an array of objects"))),
        }
    };
    let inputs = match input {
        Value::Array(a) => {
            if a.is_empty() {
                return Err(input_err(EsError::x_content_parse("[1:1] Must provide at least one value in input")));
            }
            a.iter().map(one).collect::<Result<Vec<_>>>()?
        }
        other => vec![one(other)?],
    };
    Ok((id, inputs))
}

/// The query vector of an `embedding` builder against `id`'s endpoint (an
/// `embedding` task endpoint; several inputs average into one vector, as
/// the test service does).
pub fn embedding_vector(id: &str, inputs: &[InferenceInput]) -> Result<Vec<f32>> {
    let svc = super::current().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
    let ep = svc.get_endpoint(id).ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
    if ep.task_type != super::providers::Task::Embedding {
        return Err(status(format!("Incompatible task_type for embedding API, the inference endpoint [{id}] has task type [{}], expected [embedding]", ep.task_type.name())));
    }
    let texts: Vec<String> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let rows = match svc.embed_typed(id, &texts, "search")? {
        Embedded::Dense(v) => v,
        Embedded::Sparse(_) => return Err(status(format!("Inference endpoint [{id}] is not a dense embedding endpoint"))),
    };
    let n = rows.len().max(1) as f32;
    let dims = rows.first().map(Vec::len).unwrap_or(0);
    let mut out = vec![0f32; dims];
    for r in &rows {
        for (o, x) in out.iter_mut().zip(r) {
            *o += x;
        }
    }
    if rows.len() > 1 {
        out.iter_mut().for_each(|x| *x /= n);
    }
    Ok(out)
}

/// Resolve the knn `query_vector_builder` of `o` into a `query_vector`.
pub fn resolve_knn_builder(o: &Map<String, Value>, builder: &Value) -> Result<Map<String, Value>> {
    if let Some(e) = builder.get("embedding") {
        let (id, inputs) = parse_embedding_builder(e)?;
        let id = id.ok_or_else(|| EsError::illegal_argument("[inference_id] must be specified"))?;
        let v = embedding_vector(&id, &inputs)?;
        let mut out = o.clone();
        out.shift_remove("query_vector_builder");
        out.insert("query_vector".into(), json!(v));
        return Ok(out);
    }
    let b = builder.as_object().ok_or_else(|| EsError::x_content_parse("[1:1] [knn] failed to parse field [query_vector_builder]"))?;
    let Some(te) = b.get("text_embedding").and_then(Value::as_object) else {
        let k = b.keys().next().cloned().unwrap_or_default();
        return Err(EsError::x_content_parse(format!("[1:1] [knn] failed to parse field [query_vector_builder]")).caused_by(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{k}]"))));
    };
    let model_id = te.get("model_id").and_then(Value::as_str).ok_or_else(|| {
        // `TextEmbeddingQueryVectorBuilder.buildVector` (9.x wording; 8.x: "must not be null.").
        EsError::illegal_argument(if crate::version::current_rest_api_version().is_v8() { "[model_id] must not be null." } else { "[model_id] must be specified" })
    })?;
    let text = te.get("model_text").and_then(Value::as_str).ok_or_else(|| EsError::x_content_parse("[1:1] [text_embedding] failed to parse field [model_text]: Required [model_text]"))?;
    let svc = super::current().ok_or_else(|| status("inference service is not available"))?;
    if svc.get_endpoint(model_id).is_none() {
        // Not an inference endpoint: a deployed trained model
        // (`TransportCoordinatedInferenceAction.forNlp`).
        let (res, field) = super::models::infer_text_with_field(svc.state_dir(), model_id, text).map_err(|e| {
            if e.kind == "resource_not_found_exception" {
                EsError::new(404, "status_exception", format!("[{model_id}] is not an inference service model or a deployed ml model"))
            } else {
                e
            }
        })?;
        let v = res.get(&field).cloned().unwrap_or(Value::Null);
        if !v.is_array() {
            return Err(status(format!("Trained model [{model_id}] is not a text_embedding model")));
        }
        let mut out = o.clone();
        out.shift_remove("query_vector_builder");
        out.insert("query_vector".into(), v);
        return Ok(out);
    }
    let v = match svc.embed_typed(model_id, &[text.to_string()], "search")? {
        Embedded::Dense(mut v) => v.pop().unwrap_or_default(),
        Embedded::Sparse(_) => return Err(status(format!("Inference endpoint [{model_id}] is not a text_embedding endpoint"))),
    };
    let mut out = o.clone();
    out.shift_remove("query_vector_builder");
    out.insert("query_vector".into(), json!(v));
    Ok(out)
}

/// `ShardGetService.shouldExcludeInferenceFieldsFromSource` negated: whether
/// `_inference_fields` is rehydrated into `_source` (excluded by default;
/// a source filter naming it explicitly, or `exclude_vectors: false`, brings it back).
pub fn rehydrate_inference_fields(source: &crate::search::SourceFilter, exclude_vectors: Option<bool>) -> bool {
    const NAME: &str = "_inference_fields";
    match source {
        crate::search::SourceFilter::None => false,
        crate::search::SourceFilter::All => exclude_vectors == Some(false),
        crate::search::SourceFilter::Filter { includes, excludes } => {
            let hit = |p: &String| crate::mapping::wildcard_match(p, NAME) || NAME.starts_with(&format!("{p}."));
            if excludes.iter().any(hit) || (!includes.is_empty() && !includes.iter().any(hit)) {
                false
            } else if includes.iter().any(|p| p == NAME) {
                true
            } else {
                exclude_vectors == Some(false)
            }
        }
    }
}

/// The inference blob of `field` in a doc (the `#inference` companion).
pub fn blob_of(leaf: &LeafReader<'_>, aux: u16, doc: DocId) -> Option<Value> {
    leaf.keyword_values(aux, doc).into_iter().find_map(|raw| serde_json::from_str::<Value>(&raw).ok())
}

/// The `chunks` map (`{source_field: [{start_offset, end_offset, embeddings}]}`)
/// of a blob, sources in first-appearance order.
pub fn blob_chunks(b: &Value) -> Map<String, Value> {
    let mut out: Map<String, Value> = Map::new();
    let vecs = b["v"].as_array().cloned().unwrap_or_default();
    for (c, emb) in b["c"].as_array().into_iter().flatten().zip(vecs) {
        let src = c[0].as_str().unwrap_or_default().to_string();
        let entry = out.entry(src).or_insert_with(|| json!([]));
        if let Value::Array(a) = entry {
            match c.get(3).filter(|x| x.is_u64()) {
                Some(i) => a.push(json!({"input_index": i, "embeddings": emb})),
                None => a.push(json!({"start_offset": c[1], "end_offset": c[2], "embeddings": emb})),
            }
        }
    }
    out
}

/// The chunk texts of a `semantic_text` field in a doc (`fields` with
/// `format: chunks`): each blob chunk's `[start, end)` UTF-16 span of its
/// source field's values joined by one separator.
pub fn chunk_texts(lookup: &crate::mapping::FieldLookup, leaf: &LeafReader<'_>, doc: DocId, field: &str, src: &Value) -> Vec<Value> {
    let Some(f) = lookup.fields.get(field) else { return Vec::new() };
    let Some(aux) = f.aux_id("inference") else { return Vec::new() };
    let Some(b) = blob_of(leaf, aux, doc) else { return Vec::new() };
    let Some(root) = src.as_object() else { return Vec::new() };
    let mut joined: std::collections::HashMap<String, Vec<u16>> = std::collections::HashMap::new();
    let mut out = Vec::new();
    let semantic = f.declared == EsType::Semantic;
    for c in b["c"].as_array().into_iter().flatten() {
        let s = c[0].as_str().unwrap_or_default();
        let value = extract_value(root, &source_path(lookup, s));
        // A `semantic` media chunk: its input value.
        if let Some(i) = c.get(3).and_then(Value::as_u64) {
            let inputs = value.map(semantic_inputs).unwrap_or_default();
            if let Some((_, v)) = inputs.into_iter().nth(i as usize) {
                out.push(Value::String(v));
            }
            continue;
        }
        let text = joined.entry(s.to_string()).or_insert_with(|| {
            let vals = match value {
                Some(v) if semantic => semantic_inputs(v).into_iter().filter(|(ty, _)| ty == "text").map(|(_, t)| t).collect(),
                Some(v) => node_string_values(field, v).unwrap_or_default(),
                None => Vec::new(),
            };
            vals.join("\u{0}").encode_utf16().collect()
        });
        let (a, z) = (c[1].as_u64().unwrap_or(0) as usize, c[2].as_u64().unwrap_or(0) as usize);
        if a <= z && z <= text.len() {
            out.push(Value::String(String::from_utf16_lossy(&text[a..z])));
        }
    }
    out
}

/// `_inference_fields` metadata of a hit (requested with
/// `fields: [_inference_fields]`): per `semantic_text` field with indexed
/// chunks, its endpoint, model settings and chunk embeddings by source field.
pub fn inference_fields(lookup: &crate::mapping::FieldLookup, leaf: &LeafReader<'_>, doc: DocId, _src: Option<&Value>) -> Option<Value> {
    let mut names: Vec<&String> = lookup.fields.iter().filter(|(_, f)| f.declared.is_semantic()).map(|(n, _)| n).collect();
    names.sort();
    let svc = super::current();
    let mut out = Map::new();
    for name in names {
        let Some(f) = lookup.fields.get(name) else { continue };
        let Some(aux) = f.aux_id("inference") else { continue };
        let Some(b) = blob_of(leaf, aux, doc) else { continue };
        let chunks = blob_chunks(&b);
        if chunks.is_empty() {
            continue;
        }
        let params = f.ext.as_ref().map(|e| (*e.params).clone()).unwrap_or_default();
        let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(DEFAULT_INFERENCE_ID).to_string();
        let mut inf = Map::new();
        inf.insert("inference_id".into(), json!(id));
        let ms = params.get("model_settings").filter(|v| !v.is_null()).cloned().or_else(|| svc.as_ref().and_then(|s| s.get_endpoint(&id)).and_then(|ep| super::model_settings(&ep)));
        if let Some(ms) = ms {
            inf.insert("model_settings".into(), ms);
        }
        if let Some(c) = params.get("chunking_settings").filter(|v| !v.is_null()) {
            inf.insert("chunking_settings".into(), c.clone());
        }
        inf.insert("chunks".into(), Value::Object(chunks));
        out.insert(name.clone(), json!({ "inference": inf }));
    }
    if out.is_empty() { None } else { Some(Value::Object(out)) }
}

// ---------------------------------------------------------------------------
// Document-supplied inference results (`_inference_fields` in the source,
// `SemanticTextFieldMapper.parseCreateFieldFromContext`): used as is instead
// of calling the endpoint, validated against the field's mapping.

thread_local! {
    static SUPPLIED: std::cell::RefCell<Option<Map<String, Value>>> = const { std::cell::RefCell::new(None) };
    static INFERRED: std::cell::RefCell<Option<Map<String, Value>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` (a document parse) with the inference results computed for the
/// document ([`infer_document`]) in scope: they are used like supplied
/// `_inference_fields` for the fields the source does not carry.
pub fn with_inferred<R>(m: Map<String, Value>, f: impl FnOnce() -> R) -> R {
    let prev = INFERRED.with(|c| c.replace(if m.is_empty() { None } else { Some(m) }));
    let r = f();
    INFERRED.with(|c| *c.borrow_mut() = prev);
    r
}

/// The `_inference_fields` entries of a source root, by field
/// (`{"_inference_fields": {f: ..}}` or dotted `"_inference_fields.f"` keys).
fn supplied_of(root: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, v) in root {
        if k == "_inference_fields" {
            if let Some(o) = v.as_object() {
                for (f, x) in o {
                    out.insert(f.clone(), x.clone());
                }
            }
        } else if let Some(f) = k.strip_prefix("_inference_fields.") {
            out.insert(f.to_string(), v.clone());
        }
    }
    out
}

/// Whether a root key of a document is the `_inference_fields` metadata.
pub fn is_inference_key(k: &str) -> bool {
    k == "_inference_fields" || k.starts_with("_inference_fields.")
}

/// Run a document parse with the source's `_inference_fields` in scope.
pub fn with_supplied<R>(root: &Map<String, Value>, f: impl FnOnce() -> R) -> R {
    let mut s = supplied_of(root);
    INFERRED.with(|c| {
        if let Some(m) = c.borrow().as_ref() {
            for (k, v) in m {
                s.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    });
    let prev = SUPPLIED.with(|c| c.replace(if s.is_empty() { None } else { Some(s) }));
    let r = f();
    SUPPLIED.with(|c| *c.borrow_mut() = prev);
    r
}

/// The supplied `_inference_fields` entry of `field`, if any.
pub fn supplied(field: &str) -> Option<Value> {
    SUPPLIED.with(|c| c.borrow().as_ref().and_then(|m| m.get(field).cloned()))
}

/// Every supplied `_inference_fields` entry of the document being parsed.
pub fn supplied_all() -> Map<String, Value> {
    SUPPLIED.with(|c| c.borrow().clone().unwrap_or_default())
}

fn bad_metadata(id: &str, v: &Value, cause: EsError) -> EsError {
    let preview: String = v.to_string().chars().take(100).collect();
    EsError::document_parsing(format!("[1:1] failed to parse field [_inference_fields] of type [_inference_fields] in document with id '{id}'. Preview of field's value: '{preview}'")).caused_by(cause)
}

/// `canMergeModelSettings`: the settings a document reports must equal the mapped ones.
fn settings_conflict(mapped: &Value, reported: &Value) -> Option<String> {
    for k in ["task_type", "dimensions", "similarity", "element_type"] {
        if mapped.get(k) != reported.get(k) {
            return Some(format!("Cannot update parameter [model_settings] from [{mapped}] to [{reported}]"));
        }
    }
    None
}

/// Validate a field's supplied inference results and turn them into the
/// `#inference` blob. Returns the blob and the reported model settings.
pub fn supplied_blob(field: &str, doc_id: &str, mapped_id: &str, mapped_settings: Option<&Value>, v: &Value) -> Result<(String, Option<Value>)> {
    let missing = |what: &str| bad_metadata(doc_id, v, EsError::illegal_argument(format!("Required [{what}]")));
    let inf = v.get("inference").and_then(Value::as_object).ok_or_else(|| missing("inference"))?;
    let id = inf.get("inference_id").and_then(Value::as_str).ok_or_else(|| missing("inference_id"))?;
    if id != mapped_id {
        return Err(EsError::document_parsing(format!(
            "[1:1] The configured inference_id [{id}] for field [{field}] doesn't match the inference_id [{mapped_id}] reported in the document."
        )));
    }
    let settings = inf.get("model_settings").filter(|x| !x.is_null()).cloned();
    if let (Some(m), Some(r)) = (mapped_settings, &settings) {
        if let Some(c) = settings_conflict(m, r) {
            return Err(EsError::document_parsing(format!("[1:1] Incompatible model settings for field [{field}]. Check that the inference_id is not using different model settings"))
                .caused_by(EsError::illegal_argument(format!("Mapper for [{field}] conflicts with existing mapper:\n\t{c}"))));
        }
    }
    let effective = mapped_settings.cloned().or_else(|| settings.clone());
    let chunks = inf.get("chunks").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut offsets = Vec::new();
    let mut vecs = Vec::new();
    for (src, list) in &chunks {
        for c in list.as_array().into_iter().flatten() {
            let emb = c.get("embeddings").filter(|x| !x.is_null()).ok_or_else(|| missing("embeddings"))?;
            // `semantic` media chunks reference their input by index.
            if let Some(i) = c.get("input_index").and_then(Value::as_u64) {
                offsets.push(json!([src, Value::Null, Value::Null, i]));
                vecs.push(emb.clone());
                continue;
            }
            let s = c.get("start_offset").and_then(Value::as_u64).ok_or_else(|| missing("start_offset"))?;
            let e = c.get("end_offset").and_then(Value::as_u64).ok_or_else(|| missing("end_offset"))?;
            offsets.push(json!([src, s, e]));
            vecs.push(emb.clone());
        }
    }
    if !vecs.is_empty() && effective.is_none() {
        return Err(EsError::document_parsing(format!("[1:1] [model_settings] must be set for field [{field}] when chunks are provided")));
    }
    let dense = matches!(effective.as_ref().and_then(|s| s.get("task_type")).and_then(Value::as_str), Some("text_embedding" | "embedding"));
    for x in &vecs {
        if dense != x.is_array() {
            return Err(bad_metadata(doc_id, v, EsError::illegal_argument("embeddings do not match the field's model settings")));
        }
    }
    Ok((json!({"k": if dense { "d" } else { "s" }, "c": offsets, "v": vecs}).to_string(), settings))
}

/// `SemanticQueryBuilder.doToQuery` across the searched indices: the field
/// must exist in every index and carry a single inference id.
pub fn check_query(q: Option<&Value>, targets: &[crate::search::Target]) -> Result<()> {
    let Some(q) = q else { return Ok(()) };
    let Some(o) = q.as_object() else { return Ok(()) };
    let Some((name, body)) = o.iter().find(|(k, _)| k.as_str() != "boost" && k.as_str() != "_name") else { return Ok(()) };
    if name != "semantic" {
        return Ok(());
    }
    let Some(field) = body.get("field").and_then(Value::as_str) else { return Ok(()) };
    let mut ids: Vec<String> = Vec::new();
    for t in targets {
        let lk = t.svc.lookup();
        let f = lk.fields.get(field);
        // Missing or not a semantic_text field: the shard query decides.
        let Some(id) = f.filter(|f| f.declared.is_semantic()).and_then(|f| f.ext.as_ref().and_then(|e| e.str("inference_id")).map(String::from)) else {
            continue;
        };
        if !ids.iter().any(|x| *x == id) {
            ids.push(id.to_string());
        }
    }
    // ES 9 (`SEMANTIC_QUERY_MULTIPLE_INFERENCE_IDS`) embeds the query once per
    // inference id; 8.x refused.
    if ids.len() > 1 && crate::version::current_rest_api_version().is_v8() {
        return Err(EsError::illegal_argument(format!("Field [{field}] has multiple inference IDs associated with it")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Legacy format (`index.mapping.semantic_text.use_legacy_format: true`): the
// field's `_source` value becomes `{text, inference: {inference_id,
// model_settings, chunks: [{text, embeddings}]}}`.

/// Whether a semantic_text value is the legacy object form.
pub fn is_legacy_value(v: &Value) -> bool {
    v.as_object().is_some_and(|o| o.contains_key("text") || o.contains_key("inference"))
}

/// The input texts of a legacy-form value.
pub fn legacy_texts(v: &Value) -> Vec<String> {
    match v.get("text") {
        Some(Value::Array(a)) => a.iter().filter(|x| !x.is_null()).map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(o) => vec![o.to_string()],
    }
}

/// The reported model settings of a legacy-form or `_inference_fields` value.
pub fn reported_settings(v: &Value) -> Option<Value> {
    v.pointer("/inference/model_settings").filter(|x| !x.is_null()).cloned()
}

/// Validate a legacy-form value's inference results into the `#inference`
/// blob (`err` wraps a structural failure as the field's parse error).
pub fn legacy_blob(field: &str, mapped_id: &str, mapped_settings: Option<&Value>, v: &Value, err: impl Fn(EsError) -> EsError) -> Result<String> {
    let missing = |what: &str| err(EsError::illegal_argument(format!("Required [{what}]")));
    let inf = v.get("inference").and_then(Value::as_object).ok_or_else(|| missing("inference"))?;
    let id = inf.get("inference_id").and_then(Value::as_str).ok_or_else(|| missing("inference_id"))?;
    if id != mapped_id {
        return Err(EsError::document_parsing(format!(
            "[1:1] The configured inference_id [{id}] for field [{field}] doesn't match the inference_id [{mapped_id}] reported in the document."
        )));
    }
    let settings = reported_settings(v);
    if let (Some(m), Some(r)) = (mapped_settings, &settings) {
        if let Some(c) = settings_conflict(m, r) {
            return Err(EsError::document_parsing(format!("[1:1] Incompatible model settings for field [{field}]. Check that the inference_id is not using different model settings"))
                .caused_by(EsError::illegal_argument(format!("Mapper for [{field}] conflicts with existing mapper:\n\t{c}"))));
        }
    }
    let effective = mapped_settings.cloned().or(settings);
    let texts = legacy_texts(v);
    let joined = texts.join("\n");
    let mut offsets = Vec::new();
    let mut vecs = Vec::new();
    let mut from = 0usize;
    for c in inf.get("chunks").and_then(Value::as_array).into_iter().flatten() {
        let t = c.get("text").and_then(Value::as_str).ok_or_else(|| missing("text"))?;
        let emb = c.get("embeddings").filter(|x| !x.is_null()).ok_or_else(|| missing("embeddings"))?;
        let s = joined.get(from..).and_then(|r| r.find(t)).map(|p| p + from).or_else(|| joined.find(t)).unwrap_or(0);
        from = s;
        offsets.push(json!([field, utf16_at(&joined, s), utf16_at(&joined, s + t.len())]));
        vecs.push(emb.clone());
    }
    if !vecs.is_empty() && effective.is_none() {
        return Err(EsError::document_parsing(format!("[1:1] [model_settings] must be set for field [{field}] when chunks are provided")));
    }
    let dense = matches!(effective.as_ref().and_then(|s| s.get("task_type")).and_then(Value::as_str), Some("text_embedding" | "embedding"));
    if vecs.iter().any(|x| dense != x.is_array()) {
        return Err(err(EsError::illegal_argument("embeddings do not match the field's model settings")));
    }
    Ok(json!({"k": if dense { "d" } else { "s" }, "c": offsets, "v": vecs}).to_string())
}

fn get_path<'a>(src: &'a Value, path: &str) -> Option<&'a Value> {
    let o = src.as_object()?;
    if let Some(v) = o.get(path) {
        return Some(v);
    }
    let (head, rest) = path.split_once('.')?;
    get_path(o.get(head)?, rest)
}

fn set_path(src: &mut Value, path: &str, v: Value) -> bool {
    let Some(o) = src.as_object_mut() else { return false };
    if o.contains_key(path) {
        o.insert(path.to_string(), v);
        return true;
    }
    match path.split_once('.') {
        Some((head, rest)) => o.get_mut(head).is_some_and(|c| set_path(c, rest, v)),
        None => false,
    }
}

/// `SemanticTextUtils.insertValue`: set `path`, creating the objects on the way.
fn insert_path(src: &mut Value, path: &str, v: Value) -> bool {
    let Some(o) = src.as_object_mut() else { return false };
    match path.split_once('.') {
        None => {
            o.insert(path.to_string(), v);
            true
        }
        Some((head, rest)) => {
            let child = o.entry(head.to_string()).or_insert_with(|| json!({}));
            insert_path(child, rest, v)
        }
    }
}

/// `TransportUpdateAction.deleteInferenceResults` (legacy format): a
/// partial update that sets a `copy_to` source of a semantic_text field
/// resets the field to its original text, so its inference runs again.
pub fn strip_stale_legacy(lookup: &FieldLookup, partial: &Value, merged: &mut Value) {
    let Some(p) = partial.as_object() else { return };
    for (name, f) in &lookup.fields {
        if !f.declared.is_semantic() || f.is_multi_field {
            continue;
        }
        let touched = lookup.copy_sources.get(name).into_iter().flatten().any(|s| s != name && extract_value(p, s).is_some_and(|v| !v.is_null()));
        if !touched {
            continue;
        }
        let orig = match get_path(merged, name) {
            Some(Value::Object(o)) => o.get("text").cloned().unwrap_or(Value::Null),
            Some(v) => v.clone(),
            None => continue,
        };
        set_path(merged, name, orig);
    }
}

/// `ShardBulkInferenceActionFilter` for a legacy-format bulk update: every
/// source of a semantic_text field the partial doc touches must be present.
pub fn check_legacy_bulk_update(lookup: &FieldLookup, partial: &Value) -> Result<()> {
    let Some(p) = partial.as_object() else { return Ok(()) };
    let mut names: Vec<&String> = lookup.fields.iter().filter(|(_, f)| f.declared.is_semantic() && !f.is_multi_field).map(|(n, _)| n).collect();
    names.sort();
    for name in names {
        let own = extract_value(p, name).filter(|v| !v.is_null());
        let mut sources: Vec<String> = vec![name.clone()];
        sources.extend(lookup.copy_sources.get(name).into_iter().flatten().cloned());
        sources.sort();
        sources.dedup();
        if own.is_some_and(Value::is_object) || (own.is_none() && sources.len() == 1) {
            continue;
        }
        for src in &sources {
            if extract_value(p, src).is_none_or(Value::is_null) {
                return Err(EsError::new(400, "status_exception", format!("Field [{src}] must be specified on an update request to calculate inference for field [{name}]")));
            }
        }
    }
    Ok(())
}

/// The `_source` of a legacy-format document: each semantic_text value
/// rewritten into its legacy object from the embeddings in `doc`.
pub fn legacy_source(src: &Value, lookup: &crate::mapping::FieldLookup, doc: &rustic_core::types::Document) -> Option<Value> {
    let mut out = src.clone();
    let mut changed = false;
    let svc = super::current();
    for (name, f) in &lookup.fields {
        if !f.declared.is_semantic() || f.is_multi_field {
            continue;
        }
        let Some(aux) = f.aux_id("inference") else { continue };
        let blob = doc.fields.iter().filter(|(id, _)| *id == aux).find_map(|(_, v)| match v {
            rustic_core::types::FieldValue::Keyword(s) => serde_json::from_str::<Value>(s).ok(),
            _ => None,
        });
        // A value already in the legacy form (supplied results) is kept.
        let own = get_path(src, name).filter(|v| !v.is_null());
        if own.is_some_and(is_legacy_value) {
            continue;
        }
        let texts: Vec<String> = own.map(|v| node_string_values(name, v).unwrap_or_default()).unwrap_or_default();
        // Without a value of its own the field still gets the results of
        // its `copy_to` sources (with a null `text`).
        let Some(b) = blob.filter(|b| own.is_some() || b["c"].as_array().is_some_and(|a| !a.is_empty())) else { continue };
        let id = f.ext.as_ref().and_then(|e| e.str("inference_id")).unwrap_or(DEFAULT_INFERENCE_ID).to_string();
        // Chunk text: the UTF-16 span of the source field's values joined
        // with one separator.
        let mut chunks = Vec::new();
        let vecs = b["v"].as_array().cloned().unwrap_or_default();
        for (c, emb) in b["c"].as_array().into_iter().flatten().zip(vecs) {
            let field = c[0].as_str().unwrap_or(name);
            let vals: Vec<String> = get_path(src, field).map(|v| node_string_values(name, v).unwrap_or_default()).unwrap_or_default();
            let joined: Vec<u16> = vals.join("\n").encode_utf16().collect();
            let (s, e) = (c[1].as_u64().unwrap_or(0) as usize, c[2].as_u64().unwrap_or(0) as usize);
            let t = joined.get(s..e.min(joined.len())).map(String::from_utf16_lossy).unwrap_or_default();
            chunks.push(json!({"text": t, "embeddings": emb}));
        }
        let settings = f.ext.as_ref().and_then(|x| x.params.get("model_settings").cloned()).filter(|x| !x.is_null())
            .or_else(|| svc.as_ref().and_then(|s| s.get_endpoint(&id)).and_then(|ep| super::model_settings(&ep)));
        let text = match texts.len() {
            0 if own.is_none() => Value::Null,
            1 => json!(texts[0]),
            _ => json!(texts),
        };
        let v = json!({"text": text, "inference": {"inference_id": id, "model_settings": settings, "chunks": chunks}});
        changed |= set_path(&mut out, name, v.clone()) || insert_path(&mut out, name, v);
    }
    changed.then_some(out)
}

/// Mapping checks of `SemanticTextFieldMapper.Builder.build` / its type
/// parser: never nested or a multi-field; the legacy format also forbids
/// `copy_to` and multi-fields.
pub fn validate_mapping(settings: &crate::settings::FlatSettings, lk: &crate::mapping::FieldLookup) -> Result<()> {
    let legacy = settings.get("index.mapping.semantic_text.use_legacy_format").is_some_and(|v| v.as_str() == Some("true") || v.as_bool() == Some(true));
    // ES 9.4+ (`SEMANTIC_TEXT_LEGACY_FORMAT_FORBIDDEN`): new indices cannot
    // use the legacy format (8.x clients under `compatible-with=8` still can).
    let created = crate::version::IndexVersion::from_setting(settings.get("index.version.created").or(Some(&json!(crate::version::IndexVersion::CURRENT.to_string()))));
    if legacy && !crate::version::current_rest_api_version().is_v8() && created.on_or_after(crate::version::IndexVersion::SEMANTIC_TEXT_LEGACY_FORMAT_FORBIDDEN) {
        return Err(EsError::illegal_argument(format!(
            "The setting [index.mapping.semantic_text.use_legacy_format] cannot be set to [true] for indices created on or after version [{}]. The legacy semantic text format is not supported for new indices.",
            crate::version::IndexVersion::SEMANTIC_TEXT_LEGACY_FORMAT_FORBIDDEN.0
        )));
    }
    let mut names: Vec<&String> = lk.fields.iter().filter(|(_, f)| f.declared.is_semantic()).map(|(n, _)| n).collect();
    names.sort();
    let fail = |msg: String| EsError::mapper_parsing(format!("Failed to parse mapping: {msg}")).caused_by(EsError::illegal_argument(msg));
    for name in names {
        let f = &lk.fields[name];
        let leaf = name.rsplit('.').next().unwrap_or(name);
        if legacy && f.is_multi_field {
            return Err(fail(format!("Field [{leaf}] of type [semantic_text] can't be used in multifields")));
        }
        if legacy && !f.copy_to.is_empty() {
            return Err(fail(format!("semantic_text field [{leaf}] does not support [copy_to]")));
        }
        if legacy && lk.fields.iter().any(|(n, x)| x.is_multi_field && n.strip_prefix(name.as_str()).is_some_and(|r| r.starts_with('.'))) {
            return Err(fail(format!("semantic_text field [{leaf}] does not support multi-fields")));
        }
        if f.nested_path.is_some() {
            return Err(fail(format!("semantic_text field [{name}] cannot be nested")));
        }
    }
    Ok(())
}

/// The inputs of a `semantic` field value: `(type, value)` per element
/// (scalars are text; `{type, value}` objects are media or text).
pub fn semantic_inputs(v: &Value) -> Vec<(String, String)> {
    let one = |x: &Value| -> Option<(String, String)> {
        match x {
            Value::Null => None,
            Value::Object(o) => Some((o.get("type").and_then(Value::as_str).unwrap_or("text").to_string(), o.get("value").map(scalar_string).unwrap_or_default())),
            other => Some(("text".to_string(), scalar_string(other))),
        }
    };
    match v {
        Value::Array(a) => a.iter().filter_map(one).collect(),
        other => one(other).into_iter().collect(),
    }
}

/// `SemanticOriginalValueEncoder` round trip: a `semantic` input as a
/// rebuilt `_source` shows it. Scalars become strings; a media data URI
/// becomes `{type, format: base64, value}` with its payload re-encoded in
/// canonical (padded) base64.
pub fn original_value(v: &Value) -> Value {
    use base64::Engine;
    match v {
        Value::String(_) => v.clone(),
        Value::Object(o) => {
            let ty = o.get("type").and_then(Value::as_str).unwrap_or("text");
            let value = o.get("value").map(scalar_string).unwrap_or_default();
            let canon = match value.find(',') {
                Some(i) if value[..=i].ends_with(";base64,") => {
                    let payload = value[i + 1..].trim_end_matches('=');
                    match base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload) {
                        Ok(bytes) => format!("{}{}", &value[..=i], base64::engine::general_purpose::STANDARD.encode(bytes)),
                        Err(_) => value.clone(),
                    }
                }
                _ => value.clone(),
            };
            json!({"type": ty, "format": "base64", "value": canon})
        }
        other => Value::String(scalar_string(other)),
    }
}

/// `semantic` values: strings, numbers, booleans or `{type, value}` objects.
pub fn check_semantic_input(field: &str, v: &Value) -> Result<()> {
    let check = |x: &Value| -> Result<()> {
        match x {
            Value::Object(o) => {
                let ty = o.get("type").and_then(Value::as_str);
                if ty.is_none() || !o.contains_key("value") {
                    return Err(EsError::new(400, "status_exception", format!("Invalid format for field [{field}], expected an object with [type] and [value]")));
                }
                if !matches!(ty, Some("text" | "image" | "audio" | "video" | "pdf")) {
                    return Err(EsError::illegal_argument(format!("Unrecognized type [{}], must be one of [text, image, audio, video, pdf]", ty.unwrap_or(""))));
                }
                Ok(())
            }
            Value::Array(_) => Err(EsError::new(400, "status_exception", format!("Invalid format for field [{field}], expected [String|Number|Boolean|Object] got [ArrayList]"))),
            _ => Ok(()),
        }
    };
    match v {
        Value::Array(a) => a.iter().try_for_each(check),
        other => check(other),
    }
}

/// `ShardBulkInferenceActionFilter`: semantic_text inputs must be scalars.
pub fn check_input(field: &str, v: &Value) -> Result<()> {
    let bad = match v {
        Value::Array(a) => a.iter().find(|x| x.is_object() || x.is_array()),
        Value::Object(_) => Some(v),
        _ => None,
    };
    match bad {
        Some(x) => Err(EsError::new(400, "status_exception", format!(
            "Invalid format for field [{field}], expected [String|Number|Boolean] got [{}]",
            if x.is_object() { "LinkedHashMap" } else { "ArrayList" }
        ))),
        None => Ok(()),
    }
}
