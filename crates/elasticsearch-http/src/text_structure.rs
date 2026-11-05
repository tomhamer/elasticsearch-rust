//! `_text_structure` REST handlers: `find_structure`, `find_message_structure`,
//! `find_field_structure`, `test_grok_pattern` (logic in
//! `elasticsearch_engine::text_structure`).

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ResolveOpts};
use elasticsearch_engine::search::{self, ExecOpts};
use elasticsearch_engine::text_structure::{self as ts, Overrides};
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let m = req.method.clone();
    match (req.seg(1), req.segs.len()) {
        ("find_structure", 2) => match m {
            Method::POST | Method::PUT => blocking(move || {
                let ov = Overrides::from_params(&req.params)?;
                Ok(Resp::ok(ts::find_structure(&req.body, &ov)?))
            })
            .await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        ("find_message_structure", 2) => match m {
            Method::GET | Method::POST => blocking(move || {
                let ov = Overrides::from_params(&req.params)?;
                let body = req.json()?;
                let msgs = body
                    .get("messages")
                    .and_then(Value::as_array)
                    .ok_or_else(|| EsError::action_validation(&["[messages] must be specified"]))?
                    .iter()
                    .map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))
                    .collect::<Vec<_>>();
                Ok(Resp::ok(ts::find_message_structure(&msgs, &ov)?))
            })
            .await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        ("find_field_structure", 2) => match m {
            Method::GET => blocking(move || find_field_structure(&node, &req)).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("test_grok_pattern", 2) => match m {
            Method::GET | Method::POST => blocking(move || {
                let body = req.json()?;
                Ok(Resp::ok(ts::test_grok_pattern(&body, req.param("ecs_compatibility"))?))
            })
            .await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

/// Values of `field` from up to `documents_to_sample` documents of `index`.
fn find_field_structure(node: &Node, req: &Req) -> R {
    let ov = Overrides::from_params(&req.params)?;
    let index = req.param("index").filter(|s| !s.is_empty()).ok_or_else(|| EsError::action_validation(&["index must be specified"]))?;
    let field = req.param("field").filter(|s| !s.is_empty()).ok_or_else(|| EsError::action_validation(&["field must be specified"]))?;
    let n: usize = match req.param("documents_to_sample") {
        None => 1000,
        Some(v) => v.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [documents_to_sample] with value [{v}]")))?,
    };
    if n < 2 {
        return Err(EsError::action_validation(&["[documents_to_sample] must be at least [2] if specified"]));
    }
    let body = json!({"size": n, "_source": [field], "query": {"exists": {"field": field}}, "sort": ["_doc"]});
    // On a cluster the sample comes from every shard, through the coordinator.
    let v: Value = match req.extensions.get::<crate::ClusterRest>() {
        Some(rest) => (rest.0)("POST", &crate::search_api::coordinator_search_uri(index, &std::collections::HashMap::new()), Some(body))?,
        None => {
            let targets = node.resolve(index, ResolveOpts::default())?;
            let sreq = search::parse_request(&body, &std::collections::HashMap::new())?;
            let out = search::execute(&targets, &sreq, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
            serde_json::from_str(&out.json)?
        }
    };
    let mut msgs = Vec::new();
    for h in v["hits"]["hits"].as_array().cloned().unwrap_or_default() {
        let src = &h["_source"];
        let val = src.get(field).cloned().or_else(|| {
            let mut cur = src;
            for p in field.split('.') {
                cur = cur.get(p)?;
            }
            Some(cur.clone())
        });
        match val {
            Some(Value::String(s)) => msgs.push(s),
            Some(Value::Array(a)) => msgs.extend(a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))),
            Some(Value::Null) | None => {}
            Some(other) => msgs.push(other.to_string()),
        }
    }
    Ok(Resp::ok(ts::find_message_structure(&msgs, &ov)?))
}
