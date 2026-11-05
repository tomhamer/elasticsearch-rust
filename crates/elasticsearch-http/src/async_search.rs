//! `_async_search` (submit / get / status / delete) and `_field_caps`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::async_search::AsyncSearch;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::field_caps::{field_caps as build_field_caps, FieldCapsOpts};
use elasticsearch_engine::node::{Node, ResolveOpts};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

fn time_param(req: &Req, name: &str, default: &str) -> Result<Duration> {
    let s = req.param(name).unwrap_or(default);
    let ms = elasticsearch_engine::date::parse_time_value(s)
        .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{name}] with value [{s}] as a time value: unit is missing or unrecognized")))?;
    Ok(Duration::from_millis(ms.max(0) as u64))
}

/// `/_async_search/...` (global) and `/{index}/_async_search`.
pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let off = if index.is_some() { 1 } else { 0 };
    let a = req.seg(off + 1).to_string();
    let b = req.seg(off + 2).to_string();
    let m = req.method.clone();
    match (a.as_str(), b.as_str(), &m) {
        ("", "", &Method::POST) => blocking(move || submit(&node, &req, index.as_deref())).await,
        ("", "", _) => method_not_allowed(&req, &["POST"]),
        ("status", id, &Method::GET) if index.is_none() && !id.is_empty() => {
            let id = id.to_string();
            blocking(move || status(&node, &id)).await
        }
        (id, "", &Method::GET) if index.is_none() => {
            let id = id.to_string();
            blocking(move || get(&node, &req, &id)).await
        }
        (id, "", &Method::DELETE) if index.is_none() => {
            let id = id.to_string();
            blocking(move || delete(&node, &id)).await
        }
        _ => crate::not_found_handler(&req),
    }
}

fn submit(node: &Arc<Node>, req: &Req, index: Option<&str>) -> R {
    if req.param("scroll").is_some() {
        return Err(EsError::action_validation(&["[request_cache] cannot be used in a scroll context", "[scroll] queries are not supported"]));
    }
    let wait = time_param(req, "wait_for_completion_timeout", "1s")?;
    let keep_alive = time_param(req, "keep_alive", "5d")?;
    if keep_alive < Duration::from_secs(1) {
        return Err(EsError::action_validation(&[&format!("[keep_alive] must be greater or equals than 1 second, got:{}", req.param("keep_alive").unwrap_or("0ms"))]));
    }
    let keep_on_completion = req.flag("keep_on_completion")?;
    let body = req.json()?;
    let mut params = req.params.clone();
    for k in ["wait_for_completion_timeout", "keep_alive", "keep_on_completion", "batched_reduce_size", "ccs_minimize_roundtrips"] {
        params.remove(k);
    }
    // Validate eagerly so request errors come back as plain ES errors.
    {
        let _modules = crate::search_api::test_modules(node);
        elasticsearch_engine::search::parse_request(&body, &params)?;
    }
    // ES keeps the internal response and renders it per request: store it
    // typed, strip the types for requests without `typed_keys`.
    let typed = req.flag("typed_keys")?;
    params.insert("typed_keys".into(), "true".into());
    let resolve = ResolveOpts::from_params(&req.params);
    let shards = {
        let mut o = resolve;
        if index.is_none() {
            o.allow_no_indices = true;
        }
        // ES runs the search asynchronously: a missing index surfaces as the task's failure.
        // One task per shard of every target index.
        node.resolve(index.unwrap_or("_all"), o)
            .map_err(|e| EsError::new(e.status, "status_exception", "error while executing search").caused_by(e))?
            .iter()
            .map(|t| t.svc.setting_usize("index.number_of_shards", 1).max(1))
            .sum()
    };
    let s = AsyncSearch::new(shards, keep_alive);
    let (n, s2, idx, access) = (node.clone(), s.clone(), index.map(str::to_string), req.access.clone());
    let worker_scope=elasticsearch_engine::memory::RequestScope::new("async search worker",elasticsearch_engine::memory::breakers());
    std::thread::Builder::new()
        .name("es-async-search".into())
        .spawn(move || elasticsearch_engine::memory::scoped(Some(worker_scope),|| {
            let r = crate::search_api::run_search(&n, &body, &params, idx.as_deref(), resolve, access.as_deref());
            *s2.warnings.lock() = elasticsearch_engine::warnings::take();
            s2.complete(r);
        }))
        .map_err(|e| EsError::internal(format!("failed to start async search: {e}")))?;
    let done = s.wait(wait);
    let store = !done || keep_on_completion;
    if store {
        node.async_searches.lock().insert(s.id.clone(), s.clone());
    }
    let (st, mut v) = s.response(store)?;
    if let (false, Some(r)) = (typed, v.get_mut("response")) {
        untype_keys(r);
    }
    Ok(with_warnings(Resp::json(st, v), &s))
}

/// The search's response warnings as `Warning` headers (once it finished).
fn with_warnings(mut r: Resp, s: &AsyncSearch) -> Resp {
    for w in s.warnings.lock().iter() {
        r.headers.push(crate::search_api::warning_header(w));
    }
    r
}

/// Drop the `type#` prefixes `typed_keys` puts on aggregation (and nested
/// sub-aggregation) and suggestion names.
pub fn untype_keys(resp: &mut Value) {
    fn strip(k: &str) -> String {
        k.split_once('#').map(|(_, n)| n.to_string()).unwrap_or_else(|| k.to_string())
    }
    fn agg_body(v: &mut Value) {
        match v {
            Value::Object(m) => {
                let old = std::mem::take(m);
                for (k, mut x) in old {
                    // `top_hits` documents are not aggregation trees.
                    if k != "hits" {
                        agg_body(&mut x);
                    }
                    m.insert(if k.contains('#') { strip(&k) } else { k }, x);
                }
            }
            Value::Array(a) => a.iter_mut().for_each(agg_body),
            _ => {}
        }
    }
    if let Some(a) = resp.get_mut("aggregations") {
        agg_body(a);
    }
    if let Some(Value::Object(m)) = resp.get_mut("suggest") {
        let old = std::mem::take(m);
        m.extend(old.into_iter().map(|(k, x)| (strip(&k), x)));
    }
}

/// `AsyncExecutionId.decode`: malformed ids are `invalid id: [x]` 400s.
fn check_id(id: &str) -> std::result::Result<(), EsError> {
    elasticsearch_engine::esql::async_id::decode(id).map(|_| ())
}

fn get(node: &Node, req: &Req, id: &str) -> R {
    if req.param("keep_alive").is_some() {
        time_param(req, "keep_alive", "5d")?;
    }
    if req.param("wait_for_completion_timeout").is_some() {
        time_param(req, "wait_for_completion_timeout", "0s")?;
    }
    check_id(id)?;
    let s = node.async_search(id).ok_or_else(|| Node::async_search_missing(id))?;
    if let Some(_) = req.param("keep_alive") {
        s.extend(time_param(req, "keep_alive", "5d")?);
    }
    if req.param("wait_for_completion_timeout").is_some() {
        s.wait(time_param(req, "wait_for_completion_timeout", "0s")?);
    }
    let (st, mut v) = s.response(true)?;
    if let (false, Some(r)) = (req.flag("typed_keys")?, v.get_mut("response")) {
        untype_keys(r);
    }
    Ok(with_warnings(Resp::json(st, v), &s))
}

fn status(node: &Node, id: &str) -> R {
    check_id(id)?;
    let s = node.async_search(id).ok_or_else(|| Node::async_search_missing(id))?;
    Ok(Resp::ok(s.status()))
}

fn delete(node: &Node, id: &str) -> R {
    check_id(id)?;
    let _ = node.async_search(id);
    match node.async_searches.lock().remove(id) {
        Some(_) => Ok(Resp::ack()),
        None => Err(Node::async_search_missing(id)),
    }
}

/// `GET|POST /_field_caps`, `/{index}/_field_caps`.
pub async fn field_caps(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    if req.method != Method::GET && req.method != Method::POST {
        return method_not_allowed(&req, &["GET", "POST"]);
    }
    blocking(move || {
        let body = req.json()?;
        let mut fields: Vec<String> = req.param("fields").map(|f| f.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
        if let Some(bf) = body.get("fields") {
            match bf {
                Value::Array(a) => fields.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
                Value::String(s) => fields.extend(s.split(',').map(|x| x.trim().to_string())),
                _ => {}
            }
        }
        if let Some(o) = body.as_object() {
            for k in o.keys() {
                if !matches!(k.as_str(), "fields" | "index_filter" | "runtime_mappings") {
                    return Err(EsError::x_content_parse(format!("[1:2] [field_caps_request] unknown field [{k}]")));
                }
            }
        }
        let list = |p: &str| -> Vec<String> { req.param(p).map(|f| f.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default() };
        let filters = list("filters");
        let types = list("types");
        let include_unmapped = req.flag("include_unmapped")?;
        let include_empty_fields = match req.param("include_empty_fields") {
            Some(v) => v != "false",
            None => true,
        };
        let mut o = ResolveOpts::from_params(&req.params);
        if index.is_none() {
            o.allow_no_indices = true;
        }
        // Field caps accept cross-cluster expressions: plan the remote parts.
        let mut targets = elasticsearch_engine::ccr::remote::resolve_local_part(&node, index.as_deref().unwrap_or("_all"), o)?.0;
        if let Some(a) = req.access.as_deref() {
            targets = a.apply(targets)?;
        }
        if let Some(f) = body.get("index_filter") {
            let mut keep = Vec::new();
            for t in targets {
                // ES's can-match phase: an index is dropped only when the
                // filter rewrites to `match_none` there.
                if elasticsearch_engine::search::can_match(&t, f)? {
                    keep.push(t);
                }
            }
            targets = keep;
        }
        let runtime_mappings = body.get("runtime_mappings");
        let v = build_field_caps(&targets, &FieldCapsOpts { fields: &fields, include_unmapped, filters: &filters, types: &types, runtime_mappings: runtime_mappings, include_empty_fields })?;
        Ok(Resp::ok(v))
    })
    .await
}

