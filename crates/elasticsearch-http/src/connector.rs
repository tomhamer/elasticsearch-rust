//! Connector APIs (`/_connector/...`): connectors, sync jobs and connector
//! secrets over `elasticsearch_engine::connector`.
//!
//! Routing works on the raw path so an empty id (`/_connector//_check_in`)
//! reaches the handler and fails request validation, as in ES.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::connector::{self as conn, JobListParams, ListParams, Src};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use serde_json::Value;
use std::sync::Arc;

fn int_param(req: &Req, k: &str, d: i64) -> Result<i64, EsError> {
    match req.param(k) {
        None => Ok(d),
        Some(v) => v.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{v}]"))),
    }
}

/// Path segments after `/_connector`, keeping empty inner segments.
fn parts(path: &str) -> Vec<String> {
    let mut v: Vec<String> = path.split('/').skip(1).map(crate::decode_path_segment).collect();
    while v.last().is_some_and(String::is_empty) {
        v.pop();
    }
    v.into_iter().skip(1).collect()
}

fn reply(r: conn::Out) -> R {
    let (status, v) = r?;
    Ok(Resp::json(status, v))
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    blocking(move || route(&node, &req)).await
}

fn route(node: &Node, req: &Req) -> R {
    let p = parts(&req.path);
    let s: Vec<&str> = p.iter().map(String::as_str).collect();
    let m = &req.method;
    let body = || -> Result<(Value, Src), EsError> { Ok((req.json()?, Src::new(&req.body))) };
    match s.as_slice() {
        [] => match *m {
            Method::GET => reply(conn::list(node, &ListParams {
                from: int_param(req, "from", 0)?,
                size: int_param(req, "size", 100)?,
                index_name: req.param("index_name"),
                connector_name: req.param("connector_name"),
                service_type: req.param("service_type"),
                query: req.param("query"),
                include_deleted: req.flag("include_deleted")?,
            })),
            Method::POST | Method::PUT => {
                let (b, src) = body()?;
                reply(conn::put(node, None, &b, &src))
            }
            _ => method_not_allowed(req, &["GET", "POST", "PUT"]),
        },
        ["_sync_job"] => match *m {
            Method::GET => reply(conn::sync_job_list(node, &JobListParams {
                from: int_param(req, "from", 0)?,
                size: int_param(req, "size", 100)?,
                connector_id: req.param("connector_id"),
                status: req.param("status"),
                job_type: req.param("job_type"),
            })),
            Method::POST => {
                let (b, src) = body()?;
                reply(match req.extensions.get::<crate::ConnectorSourceLookup>() {
                    Some(reader) => conn::sync_job_post_with_reader(node, &b, &src, &*reader.0),
                    None => conn::sync_job_post(node, &b, &src),
                })
            }
            _ => method_not_allowed(req, &["GET", "POST"]),
        },
        ["_sync_job", id] => match *m {
            Method::GET => reply(conn::sync_job_get(node, id)),
            Method::DELETE => reply(conn::sync_job_delete(node, id)),
            _ => method_not_allowed(req, &["GET", "DELETE"]),
        },
        ["_sync_job", id, action] => {
            if *m != Method::PUT {
                return method_not_allowed(req, &["PUT"]);
            }
            match *action {
                "_check_in" => reply(conn::sync_job_check_in(node, id)),
                "_cancel" => reply(conn::sync_job_cancel(node, id)),
                "_error" | "_stats" | "_claim" => {
                    let (b, src) = body()?;
                    reply(match *action {
                        "_error" => conn::sync_job_error(node, id, &b, &src),
                        "_stats" => conn::sync_job_stats(node, id, &b, &src),
                        _ => conn::sync_job_claim(node, id, &b, &src),
                    })
                }
                _ => crate::not_found_handler(req),
            }
        }
        ["_secret"] => match *m {
            Method::POST => {
                let (b, src) = body()?;
                reply(conn::secret_post(node, &b, &src))
            }
            _ => method_not_allowed(req, &["POST"]),
        },
        ["_secret", id] => match *m {
            Method::GET => reply(conn::secret_get(node, id)),
            Method::DELETE => reply(conn::secret_delete(node, id)),
            Method::PUT => {
                let (b, src) = body()?;
                reply(conn::secret_put(node, id, &b, &src))
            }
            _ => method_not_allowed(req, &["GET", "PUT", "DELETE"]),
        },
        [id] => match *m {
            Method::GET => reply(conn::get(node, id, req.flag("include_deleted")?)),
            Method::PUT => {
                let (b, src) = body()?;
                reply(conn::put(node, Some(id), &b, &src))
            }
            Method::DELETE => reply(conn::delete_connector(node, id, req.flag("delete_sync_jobs")?, req.flag("hard")?)),
            _ => method_not_allowed(req, &["GET", "PUT", "DELETE"]),
        },
        [id, "_check_in"] if *m == Method::PUT => reply(conn::check_in(node, id)),
        [id, "_filtering", "_activate"] if *m == Method::PUT => reply(conn::activate_filtering(node, id)),
        [id, "_filtering", "_validation"] if *m == Method::PUT => {
            let (b, src) = body()?;
            reply(conn::update_filtering_validation(node, id, &b, &src))
        }
        [id, action] if *m == Method::PUT => {
            let f: fn(&Node, &str, &Value, &Src) -> conn::Out = match *action {
                "_error" => conn::update_error,
                "_name" => conn::update_name,
                "_service_type" => conn::update_service_type,
                "_index_name" => conn::update_index_name,
                "_native" => conn::update_native,
                "_status" => conn::update_status,
                "_api_key_id" => conn::update_api_key_id,
                "_features" => conn::update_features,
                "_pipeline" => conn::update_pipeline,
                "_scheduling" => conn::update_scheduling,
                "_last_sync" => conn::last_sync,
                "_configuration" => conn::update_configuration,
                "_filtering" => conn::update_filtering,
                _ => return crate::not_found_handler(req),
            };
            let (b, src) = body()?;
            reply(f(node, id, &b, &src))
        }
        _ => crate::not_found_handler(req),
    }
}
