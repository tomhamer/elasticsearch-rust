//! Stored scripts (`_scripts/{id}`), `_scripts/painless/_execute`,
//! `_script_context` and `_script_language`.

use crate::{blocking, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::script;
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    blocking(move || handle_sync(&node, &req)).await
}

fn handle_sync(node: &Node, req: &Req) -> R {
    let m = req.method.clone();
    let stored = script::stored::for_data_dir(&node.data);
    match (req.seg(0), req.seg(1), req.seg(2), &m) {
        ("_script_context", "", _, &Method::GET) => Ok(Resp::ok(script_contexts())),
        ("_script_language", "", _, &Method::GET) => Ok(Resp::ok(json!({
            "types_allowed": ["inline", "stored"],
            "language_contexts": [
                {"language": "expression", "contexts": ["aggregation_selector", "aggs", "bucket_aggregation", "field", "filter", "number_sort", "score", "terms_set"]},
                {"language": "mustache", "contexts": ["template"]},
                {"language": "painless", "contexts": script::CONTEXT_NAMES.iter().map(|c| c.0).collect::<Vec<_>>()},
            ]
        }))),
        ("_scripts", "painless", "_execute", &Method::POST) | ("_scripts", "painless", "_execute", &Method::GET) => {
            let (body, raw) = match req.json() {
                Ok(body) => (body, None),
                Err(_) => elasticsearch_engine::source_json::update_request(
                    std::str::from_utf8(req.body_text()).map_err(|_| EsError::parsing("request body is not valid UTF-8"))?)?,
            };
            let lookup = |name: &str| -> Result<Arc<elasticsearch_engine::mapping::FieldLookup>, EsError> {
                let svc = node.get_index(name).ok_or_else(|| EsError::index_not_found(name))?;
                Ok(svc.lookup())
            };
            Ok(Resp::raw(200, script::painless_execute_source(&body, raw.as_ref(), &lookup, Some(&stored))?.text()))
        }
        ("_scripts", id, ctx, &Method::PUT) | ("_scripts", id, ctx, &Method::POST) if !id.is_empty() => {
            let (body, raw) = match req.json() {
                Ok(body) => (body, None),
                Err(_) => elasticsearch_engine::source_json::update_request(
                    std::str::from_utf8(req.body_text()).map_err(|_| EsError::parsing("request body is not valid UTF-8"))?)?,
            };
            stored.put_source(id, &body, raw.as_ref(), if ctx.is_empty() { None } else { Some(ctx) })?;
            Ok(Resp::ack())
        }
        ("_scripts", id, "", &Method::GET) if !id.is_empty() => {
            let (found, v) = stored.get_raw_json(id);
            Ok(Resp::raw(if found { 200 } else { 404 }, v))
        }
        ("_scripts", id, "", &Method::DELETE) if !id.is_empty() => {
            stored.delete(id)?;
            Ok(Resp::ack())
        }
        _ => crate::not_found_handler(req),
    }
}

fn script_contexts() -> Value {
    let contexts: Vec<Value> = script::CONTEXT_NAMES
        .iter()
        .map(|(name, ret)| {
            json!({
                "name": name,
                "methods": [
                    {"name": "execute", "return_type": ret, "params": []},
                ]
            })
        })
        .collect();
    json!({ "contexts": contexts })
}
