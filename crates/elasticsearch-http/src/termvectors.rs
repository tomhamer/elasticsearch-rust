//! `/{index}/_termvectors[/{id}]` and `[/{index}]/_mtermvectors`
//! (engine: `elasticsearch_engine::termvectors`).

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::termvectors::{mterm_vectors, term_vectors, TvRequest};
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let off = if index.is_some() { 1 } else { 0 };
    let api = req.seg(off).to_string();
    let id = req.segs.get(off + 1).cloned();
    if !matches!(req.method, Method::GET | Method::POST) {
        return method_not_allowed(&req, &["GET", "POST"]);
    }
    match (api.as_str(), index) {
        ("_termvectors", Some(index)) => blocking(move || {
            let body = req.json()?;
            let tv = TvRequest::from_parts(&req.params, &body)?;
            // `TermVectorsRequest.validate` runs before the index is resolved.
            let body_id = body.get("_id").map(elasticsearch_engine::query::dsl::scalar_string);
            let id = id.or(body_id);
            if id.is_some() && tv.doc.is_some() {
                return Err(EsError::new(400, "parse_exception", "failed to parse term vectors request. either [id] or [doc] can be specified, but not both!"));
            }
            if id.is_none() && tv.doc.is_none() {
                return Err(EsError::action_validation(&["id or doc is missing"]));
            }
            Ok(Resp::ok(term_vectors(&node, &index, id.as_deref(), &tv)?))
        })
        .await,
        ("_mtermvectors", index) if id.is_none() => blocking(move || {
            let body = req.json()?;
            Ok(Resp::ok(mterm_vectors(&node, index.as_deref(), &req.params, &body)?))
        })
        .await,
        _ => crate::not_found_handler(&req),
    }
}
