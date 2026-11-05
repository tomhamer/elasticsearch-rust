//! `/_ml/data_frame/analytics/...` REST handlers over
//! `elasticsearch_engine::ml::dfa`, and `_cat/ml/data_frame/analytics`.

use crate::cat::{render, Table};
use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::ml::dfa;
use elasticsearch_engine::node::{Node, NODE_ID};
use serde_json::Value;
use std::sync::Arc;

fn int_param(req: &Req, k: &str, default: i64) -> Result<i64, EsError> {
    match req.param(k) {
        None | Some("") => Ok(default),
        Some(v) => v.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{v}]"))),
    }
}

/// Body as JSON, `None` when absent / empty.
fn body(req: &Req) -> Result<Option<Value>, EsError> {
    let b = req.json()?;
    Ok((!b.is_null()).then_some(b))
}

/// `id` / a flag given in a start / stop body (ES lets the body repeat them).
fn body_str(b: &Option<Value>, k: &str) -> Option<String> {
    b.as_ref().and_then(|v| v.get(k)).map(|v| match v {
        Value::String(s) => s.clone(),
        x => x.to_string(),
    })
}

fn bool_from(req: &Req, b: &Option<Value>, k: &str, default: bool) -> Result<bool, EsError> {
    if let Some(v) = req.bool_param(k)? {
        return Ok(v);
    }
    Ok(match b.as_ref().and_then(|v| v.get(k)) {
        Some(Value::Bool(x)) => *x,
        Some(Value::String(s)) => s == "true",
        _ => default,
    })
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    if req.seg(2) != "analytics" {
        return crate::not_found_handler(&req);
    }
    let rest: Vec<String> = req.segs.iter().skip(3).cloned().collect();
    let m = req.method.clone();
    let get_like = m == Method::GET || m == Method::POST;
    match rest.len() {
        0 => match m {
            Method::GET => get(node, &req, "_all".into()).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        1 => match rest[0].as_str() {
            "_stats" => match m {
                Method::GET => stats(node, &req, "_all".into()).await,
                _ => method_not_allowed(&req, &["GET"]),
            },
            "_explain" if get_like => {
                let b = body(&req)?;
                blocking(move || Ok(Resp::ok(dfa::explain(&node, None, b.as_ref())?))).await
            }
            "_preview" if get_like => {
                let b = body(&req)?;
                blocking(move || Ok(Resp::ok(dfa::preview(&node, None, b.as_ref())?))).await
            }
            "_explain" | "_preview" => method_not_allowed(&req, &["GET", "POST"]),
            id => {
                let id = id.to_string();
                match m {
                    Method::GET => get(node, &req, id).await,
                    Method::PUT => {
                        let b = req.json()?;
                        blocking(move || Ok(Resp::ok(dfa::put(&node, &id, &b)?))).await
                    }
                    Method::DELETE => {
                        let force = req.flag("force")?;
                        blocking(move || Ok(Resp::ok(dfa::delete(&node, &id, force)?))).await
                    }
                    _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
                }
            }
        },
        2 => {
            let id = rest[0].clone();
            match (rest[1].as_str(), &m) {
                ("_stats", &Method::GET) => stats(node, &req, id).await,
                ("_start", &Method::POST) => {
                    let b = body(&req)?;
                    let body_id = body_str(&b, "id");
                    blocking(move || Ok(Resp::ok(dfa::start(&node, &id, body_id.as_deref())?))).await
                }
                ("_stop", &Method::POST) => {
                    let b = body(&req)?;
                    let body_id = body_str(&b, "id");
                    let allow = bool_from(&req, &b, "allow_no_match", true)?;
                    let force = bool_from(&req, &b, "force", false)?;
                    blocking(move || Ok(Resp::ok(dfa::stop(&node, &id, body_id.as_deref(), allow, force)?))).await
                }
                ("_update", &Method::POST) => {
                    let b = req.json()?;
                    blocking(move || Ok(Resp::ok(dfa::update(&node, &id, &b)?))).await
                }
                ("_explain", _) if get_like => {
                    let b = body(&req)?;
                    blocking(move || Ok(Resp::ok(dfa::explain(&node, Some(&id), b.as_ref())?))).await
                }
                ("_preview", _) if get_like => {
                    let b = body(&req)?;
                    blocking(move || Ok(Resp::ok(dfa::preview(&node, Some(&id), b.as_ref())?))).await
                }
                ("_stats", _) => method_not_allowed(&req, &["GET"]),
                ("_start" | "_stop" | "_update", _) => method_not_allowed(&req, &["POST"]),
                ("_explain" | "_preview", _) => method_not_allowed(&req, &["GET", "POST"]),
                _ => crate::not_found_handler(&req),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

async fn get(node: Arc<Node>, req: &Req, id: String) -> R {
    let allow = req.bool_param("allow_no_match")?.unwrap_or(true);
    let from = int_param(req, "from", 0)?;
    let size = int_param(req, "size", 100)?;
    let excl = req.flag("exclude_generated")?;
    blocking(move || Ok(Resp::ok(dfa::get(&node, &id, allow, from, size, excl)?))).await
}

async fn stats(node: Arc<Node>, req: &Req, id: String) -> R {
    let allow = req.bool_param("allow_no_match")?.unwrap_or(true);
    let from = int_param(req, "from", 0)?;
    let size = int_param(req, "size", 100)?;
    let verbose = req.flag("verbose")?;
    blocking(move || Ok(Resp::ok(dfa::stats(&node, &id, allow, from, size, verbose)?))).await
}

/// `GET /_cat/ml/data_frame/analytics[/{id}]` (ES `RestCatDataFrameAnalyticsAction`).
pub fn cat(node: &Node, req: &Req, id: Option<&str>) -> R {
    let allow = req.bool_param("allow_no_match")?.unwrap_or(true);
    let rows = dfa::cat_rows(node, id.unwrap_or("_all"), allow)?;
    let rows = rows
        .into_iter()
        .map(|r| {
            let (nid, nname, neph, naddr) = if r.assigned { (NODE_ID.to_string(), node.name.clone(), NODE_ID.to_string(), "127.0.0.1:9300".to_string()) } else { Default::default() };
            vec![
                r.id,
                r.kind,
                r.create_time.map(elasticsearch_engine::ilm::iso_date).unwrap_or_default(),
                r.version,
                r.source_index,
                r.dest_index,
                r.description,
                r.model_memory_limit,
                r.state,
                r.failure_reason,
                r.progress,
                r.assignment_explanation,
                nid,
                nname,
                neph,
                naddr,
            ]
        })
        .collect();
    render(
        req,
        Table {
            cols: vec![
                ("id", &[], true),
                ("type", &["t"], true),
                ("create_time", &["ct", "createTime"], true),
                ("version", &["v"], false),
                ("source_index", &["si", "sourceIndex"], false),
                ("dest_index", &["di", "destIndex"], false),
                ("description", &["d"], false),
                ("model_memory_limit", &["mml", "modelMemoryLimit"], false),
                ("state", &["s"], true),
                ("failure_reason", &["fr", "failureReason"], false),
                ("progress", &["p"], false),
                ("assignment_explanation", &["ae", "assignmentExplanation"], false),
                ("node.id", &["ni", "nodeId"], false),
                ("node.name", &["nn", "nodeName"], false),
                ("node.ephemeral_id", &["ne", "nodeEphemeralId"], false),
                ("node.address", &["na", "nodeAddress"], false),
            ],
            rows,
            numeric: vec![],
        },
    )
}
