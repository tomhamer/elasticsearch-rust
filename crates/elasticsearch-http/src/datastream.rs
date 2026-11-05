//! `_data_stream` create/get/delete and `_rollover`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::node::Node;
use std::sync::Arc;

/// `/_data_stream[/{name}]`.
pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let name = req.seg(1).to_string();
    if !req.seg(2).is_empty() {
        return crate::not_found_handler(&req);
    }
    let m = req.method.clone();
    match m {
        Method::PUT if !name.is_empty() => blocking(move || {
            node.create_data_stream(&name)?;
            Ok(Resp::ack())
        })
        .await,
        Method::GET => blocking(move || Ok(Resp::ok(node.get_data_streams(if name.is_empty() { None } else { Some(&name) })?))).await,
        Method::DELETE if !name.is_empty() => blocking(move || {
            node.delete_data_streams(&name)?;
            Ok(Resp::ack())
        })
        .await,
        _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
    }
}

/// `PUT|POST /_create_from/{source}/{dest}`.
pub async fn create_from(node: Arc<Node>, req: Req) -> R {
    let (source, dest) = (req.seg(1).to_string(), req.seg(2).to_string());
    if source.is_empty() || dest.is_empty() || !req.seg(3).is_empty() {
        return crate::not_found_handler(&req);
    }
    if req.method != Method::PUT && req.method != Method::POST {
        return method_not_allowed(&req, &["PUT", "POST"]);
    }
    blocking(move || {
        let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
        Ok(Resp::ok(elasticsearch_engine::migrate::create_from(&node, &source, &dest, &body)?))
    })
    .await
}

/// `POST /_migration/reindex`, `GET /_migration/reindex/{index}/_status`,
/// `POST /_migration/reindex/{index}/_cancel`.
pub async fn migration_reindex(node: Arc<Node>, req: Req) -> R {
    let (name, action) = (req.seg(2).to_string(), req.seg(3).to_string());
    if !req.seg(4).is_empty() {
        return crate::not_found_handler(&req);
    }
    match (name.is_empty(), action.as_str()) {
        (true, "") => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || {
                let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
                Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex(&node, &body)?))
            })
            .await
        }
        (false, "_status") => {
            if req.method != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            blocking(move || Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex_status(&node, &name)?))).await
        }
        (false, "_cancel") => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex_cancel(&node, &name)?))).await
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `POST /{index}/_downsample/{target_index}`.
pub async fn downsample(node: Arc<Node>, req: Req, index: String) -> R {
    let target = req.seg(2).to_string();
    if target.is_empty() || !req.seg(3).is_empty() {
        return crate::not_found_handler(&req);
    }
    if req.method != Method::POST {
        return method_not_allowed(&req, &["POST"]);
    }
    // ES downsamples under DOWNSAMPLE_ORIGIN: the target needs no caller privileges.
    let rest = req.internal_cluster_rest();
    blocking(move || {
        let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
        let out = match rest {
            Some(rest) => elasticsearch_engine::migrate::downsample_with(&node, &index, &target, &body, &RestDownsampleOps(rest))?,
            None => elasticsearch_engine::migrate::downsample(&node, &index, &target, &body)?,
        };
        Ok(Resp::ok(out))
    })
    .await
}

/// Downsample index operations through the cluster coordinator: the source
/// is read from every shard, the target is created in the cluster metadata
/// and its documents go to its primaries.
struct RestDownsampleOps(crate::ClusterRest);

impl RestDownsampleOps {
    fn call(&self, method: &str, uri: &str, body: Option<serde_json::Value>) -> Result<serde_json::Value, elasticsearch_engine::error::EsError> {
        (self.0 .0)(method, uri, body)
    }
}

/// The NDJSON `_bulk` body indexing `docs` with generated ids.
fn bulk_index_body(docs: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for d in docs {
        out.push_str("{\"index\":{}}\n");
        out.push_str(&d.to_string());
        out.push('\n');
    }
    out
}

/// The first item failure of a `_bulk` response, as an error.
fn bulk_failure(resp: &serde_json::Value) -> Option<elasticsearch_engine::error::EsError> {
    if resp["errors"].as_bool() != Some(true) {
        return None;
    }
    resp["items"].as_array()?.iter().find_map(|item| {
        let r = item.as_object()?.values().next()?;
        let e = r.get("error")?;
        let status = r["status"].as_u64().unwrap_or(500) as u16;
        Some(elasticsearch_engine::error::EsError::new(status, e["type"].as_str().unwrap_or("exception"), e["reason"].as_str().unwrap_or("bulk item failed")))
    })
}

impl elasticsearch_engine::migrate::DownsampleOps for RestDownsampleOps {
    fn create_index(&self, name: &str, body: &serde_json::Value) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("PUT", &format!("/{name}"), Some(body.clone())).map(|_| ())
    }
    fn scan(&self, index: &str, f: &mut dyn FnMut(&serde_json::Value)) -> Result<(), elasticsearch_engine::error::EsError> {
        let mut resp = self.call("POST", &format!("/{index}/_search?scroll=1m"), Some(serde_json::json!({"size": 1000, "sort": ["_doc"]})))?;
        let mut scroll_id = resp["_scroll_id"].as_str().map(str::to_string);
        let result = loop {
            let hits = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
            if hits.is_empty() {
                break Ok(());
            }
            for h in &hits {
                if let Some(s) = h.get("_source") {
                    f(s);
                }
            }
            let Some(id) = scroll_id.clone() else { break Ok(()) };
            match self.call("POST", "/_search/scroll", Some(serde_json::json!({"scroll": "1m", "scroll_id": id}))) {
                Ok(r) => {
                    resp = r;
                    if let Some(id) = resp["_scroll_id"].as_str() {
                        scroll_id = Some(id.to_string());
                    }
                }
                Err(e) => break Err(e),
            }
        };
        if let Some(id) = scroll_id {
            let _ = self.call("DELETE", "/_search/scroll", Some(serde_json::json!({"scroll_id": [id]})));
        }
        result
    }
    fn index_docs(&self, index: &str, docs: Vec<serde_json::Value>) -> Result<(), elasticsearch_engine::error::EsError> {
        let resp = self.call("POST", &format!("/{index}/_bulk"), Some(serde_json::Value::String(bulk_index_body(&docs))))?;
        match bulk_failure(&resp) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
    fn refresh(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("POST", &format!("/{index}/_refresh"), None).map(|_| ())
    }
    fn block_writes(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("PUT", &format!("/{index}/_settings"), Some(serde_json::json!({"index": {"blocks": {"write": true}}}))).map(|_| ())
    }
    fn delete_index(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("DELETE", &format!("/{index}"), None).map(|_| ())
    }
}

#[cfg(test)]
mod downsample_ops_tests {
    use super::*;

    #[test]
    fn bulk_body_is_ndjson_with_auto_ids() {
        let body = bulk_index_body(&[serde_json::json!({"a": 1}), serde_json::json!({"b": "x"})]);
        assert_eq!(body, "{\"index\":{}}\n{\"a\":1}\n{\"index\":{}}\n{\"b\":\"x\"}\n");
    }

    #[test]
    fn bulk_failure_reports_first_item_error() {
        let ok = serde_json::json!({"errors": false, "items": [{"index": {"status": 201}}]});
        assert!(bulk_failure(&ok).is_none());
        let bad = serde_json::json!({"errors": true, "items": [{"index": {"status": 201}}, {"index": {"status": 400, "error": {"type": "document_parsing_exception", "reason": "boom"}}}]});
        let e = bulk_failure(&bad).expect("failure");
        assert_eq!(e.status, 400);
    }
}

/// `POST /{target}/_rollover[/{new_index}]`.
pub async fn rollover(node: Arc<Node>, req: Req, target: String) -> R {
    if req.method != Method::POST {
        return method_not_allowed(&req, &["POST"]);
    }
    blocking(move || {
        let body = req.json()?;
        let new_index = req.segs.get(2).cloned();
        let dry_run = req.flag("dry_run")?;
        let lazy = req.flag("lazy")?;
        let v = node.rollover_ext(&target, new_index.as_deref(), &body, dry_run, lazy)?;
        let mut r = Resp::ok(v);
        // `RestRolloverIndexAction`: `max_size` is deprecated.
        if body.pointer("/conditions/max_size").is_some_and(|v| !v.is_null()) {
            r.headers.push(crate::search_api::warning_header(
                "Use of the [max_size] rollover condition has been deprecated in favour of the [max_primary_shard_size] condition and will be removed in a later version",
            ));
        }
        Ok(r)
    })
    .await
}
