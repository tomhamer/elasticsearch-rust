//! Watcher REST APIs: `_watcher/watch/*`, `_watcher/_start|_stop`,
//! `_watcher/stats`, `_watcher/_query/watches`, `_watcher/settings`.

use crate::{blocking, method_not_allowed, not_found_handler, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::watcher::api::{self, PutParams};
use elasticsearch_engine::watcher::model;
use serde_json::{json, Value};
use std::sync::Arc;

fn num(req: &Req, k: &str) -> Result<Option<i64>, EsError> {
    match req.param(k) {
        None => Ok(None),
        Some(s) => s
            .parse::<i64>()
            .map(Some)
            .map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{s}]"))),
    }
}

fn with_warnings(mut r: Resp, warnings: Vec<String>) -> Resp {
    let mut seen = std::collections::HashSet::new();
    for w in warnings {
        if seen.insert(w.clone()) {
            r.headers.push(("Warning", format!("299 Elasticsearch-{}-{} \"{w}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH)));
        }
    }
    r
}

/// The effective user of the request (security on), as watcher records it.
fn caller(node: &Node, req: &Req) -> Option<String> {
    if !node.security.enabled() {
        return None;
    }
    req.access.as_ref().map(|a| a.user().effective().username.clone())
}

fn body_or_empty(req: &Req) -> Result<Value, EsError> {
    let b = req.json()?;
    Ok(if b.is_null() { json!({}) } else { b })
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    if !elasticsearch_engine::watcher::watcher_enabled(&node) {
        return not_found_handler(&req);
    }
    // On a cluster, watches read and write indices through the coordinator.
    elasticsearch_engine::watcher::set_remote(req.extensions.get::<crate::ClusterRest>().map(|r| -> elasticsearch_engine::watcher::RestCall { r.0.clone() }));
    let b = req.seg(1).to_string();
    let c = req.seg(2).to_string();
    let d = req.seg(3).to_string();
    let e = req.seg(4).to_string();
    let m = req.method.clone();
    let put_or_post = m == Method::PUT || m == Method::POST;
    // Watch APIs need a standard+ license (ES's `WatcherTransportAction`);
    // stats, settings, start / stop and delete do not.
    if (b == "watch" && !(m == Method::DELETE && d.is_empty() && c != "_execute")) || b == "_query" {
        elasticsearch_engine::license::check(&node, "watcher", elasticsearch_engine::license::Mode::Standard)?;
    }
    match b.as_str() {
        "watch" => {
            // _watcher/watch/_execute (inline)
            if c == "_execute" {
                if !put_or_post {
                    return method_not_allowed(&req, &["POST", "PUT"]);
                }
                let debug = req.flag("debug")?;
                let user = caller(&node, &req);
                return blocking(move || Ok(Resp::ok(api::execute_watch(&node, None, &body_or_empty(&req)?, debug, user)?))).await;
            }
            if c.is_empty() {
                return not_found_handler(&req);
            }
            match d.as_str() {
                "" => match m {
                    Method::GET => blocking(move || {
                        let (s, v) = api::get_watch(&node, &c)?;
                        let w = model::parse_warnings(&v["watch"]);
                        Ok(with_warnings(Resp::json(s, v), w))
                    })
                    .await,
                    Method::DELETE => blocking(move || {
                        let (s, v) = api::delete_watch(&node, &c)?;
                        Ok(Resp::json(s, v))
                    })
                    .await,
                    Method::PUT | Method::POST => {
                        let p = PutParams {
                            active: req.bool_param("active")?.unwrap_or(true),
                            version: num(&req, "version")?,
                            if_seq_no: num(&req, "if_seq_no")?,
                            if_primary_term: num(&req, "if_primary_term")?.map(|x| x as u64),
                            user: caller(&node, &req),
                        };
                        blocking(move || {
                            let body = req.json()?;
                            let w = model::parse_warnings(&body);
                            let v = api::put_watch(&node, &c, &body, &p)?;
                            let status = if v["created"] == true { 201 } else { 200 };
                            Ok(with_warnings(Resp::json(status, v), w))
                        })
                        .await
                    }
                    _ => method_not_allowed(&req, &["GET", "PUT", "POST", "DELETE"]),
                },
                "_execute" => {
                    if !put_or_post {
                        return method_not_allowed(&req, &["POST", "PUT"]);
                    }
                    let debug = req.flag("debug")?;
                    let user = caller(&node, &req);
                    blocking(move || {
                        let body = body_or_empty(&req)?;
                        let w = api::cached(&node, &c).map(|src| model::parse_warnings(&src)).unwrap_or_default();
                        Ok(with_warnings(Resp::ok(api::execute_watch(&node, Some(&c), &body, debug, user)?), w))
                    })
                    .await
                }
                "_ack" => {
                    if !put_or_post {
                        return method_not_allowed(&req, &["POST", "PUT"]);
                    }
                    blocking(move || Ok(Resp::ok(api::ack_watch(&node, &c, if e.is_empty() { None } else { Some(&e) })?))).await
                }
                "_activate" | "_deactivate" => {
                    if !put_or_post {
                        return method_not_allowed(&req, &["POST", "PUT"]);
                    }
                    let active = d == "_activate";
                    blocking(move || Ok(Resp::ok(api::set_active(&node, &c, active)?))).await
                }
                _ => not_found_handler(&req),
            }
        }
        "_start" | "_stop" => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let start = b == "_start";
            blocking(move || Ok(Resp::ok(if start { api::start(&node) } else { api::stop(&node) }))).await
        }
        "stats" | "_stats" => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            let metric = if c.is_empty() { req.param("metric").map(str::to_string) } else { Some(c.clone()) };
            let metric = metric.or_else(|| {
                if req.flag("emit_stacktraces").unwrap_or(false) || req.flag("all").unwrap_or(false) {
                    Some("_all".to_string())
                } else {
                    None
                }
            });
            let w = if metric.as_deref().map(|m| m.split(',').any(|x| x == "pending_watches")).unwrap_or(false) {
                vec!["The pending_watches parameter is deprecated, use queued_watches instead".to_string()]
            } else {
                Vec::new()
            };
            blocking(move || Ok(with_warnings(Resp::ok(api::stats(&node, metric.as_deref())?), w))).await
        }
        "_query" if c == "watches" => {
            if !(m == Method::GET || m == Method::POST) {
                return method_not_allowed(&req, &["GET", "POST"]);
            }
            blocking(move || {
                let v = api::query_watches(&node, &body_or_empty(&req)?)?;
                let w = v["watches"].as_array().map(|a| a.iter().flat_map(|x| model::parse_warnings(&x["watch"])).collect()).unwrap_or_default();
                Ok(with_warnings(Resp::ok(v), w))
            })
            .await
        }
        "settings" => match m {
            Method::GET => blocking(move || Ok(Resp::ok(api::get_settings(&node)))).await,
            Method::PUT => blocking(move || Ok(Resp::ok(api::put_settings(&node, &body_or_empty(&req)?)?))).await,
            _ => method_not_allowed(&req, &["GET", "PUT"]),
        },
        _ => not_found_handler(&req),
    }
}

/// The watcher section of `GET _xpack/usage`.
pub fn xpack_usage(node: &Node) -> Value {
    api::usage(node)
}
