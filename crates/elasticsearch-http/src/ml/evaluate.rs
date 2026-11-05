//! `POST /_ml/data_frame/_evaluate`.

use crate::{Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::node::Node;
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    if req.method != Method::POST {
        return crate::method_not_allowed(&req, &["POST"]);
    }
    let body = req.json()?;
    crate::blocking(move || Ok(Resp::ok(elasticsearch_engine::ml::evaluate::evaluate(&node, &body)?))).await
}
