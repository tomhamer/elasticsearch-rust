//! Cross-cluster replication REST handlers (x-pack `ccr`): `/_ccr/stats`,
//! `/_ccr/auto_follow[/{name}[/pause|/resume]]` and
//! `/{index}/_ccr/{follow,pause_follow,resume_follow,unfollow,info,stats,forget_follower}`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::ccr;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use serde_json::{json, Map, Value};
use std::sync::Arc;

fn human(req: &Req) -> bool {
    matches!(req.param("human"), Some("") | Some("true"))
}

/// `?human`: the ES-specific human-readable twins of the shard follow
/// status fields (`ShardFollowNodeTaskStatus.humanReadableField`).
fn humanize_status(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if m.contains_key("follower_index") && m.contains_key("time_since_last_read_millis") {
                let mut out = Map::new();
                for (k, x) in std::mem::take(m) {
                    let h = match k.as_str() {
                        "total_read_time_millis" => Some(("total_read_time", time(&x))),
                        "total_read_remote_exec_time_millis" => Some(("total_read_remote_exec_time", time(&x))),
                        "bytes_read" => Some(("total_read", crate::util::human_bytes(x.as_u64().unwrap_or(0)))),
                        "total_write_time_millis" => Some(("total_write_time", time(&x))),
                        "time_since_last_read_millis" => Some(("time_since_last_read", time(&x))),
                        _ => None,
                    };
                    if let Some((hk, hv)) = h {
                        out.insert(hk.into(), json!(hv));
                    }
                    out.insert(k, x);
                }
                *m = out;
            } else {
                for x in m.values_mut() {
                    humanize_status(x);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(humanize_status),
        _ => {}
    }
}

fn time(v: &Value) -> String {
    match v.as_i64() {
        Some(ms) if ms < 0 => "-1".into(),
        Some(ms) => crate::util::human_time_nanos(ms as u64 * 1_000_000),
        None => "0s".into(),
    }
}

/// `/_ccr/...`
pub async fn handle_global(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let s = |i: usize| segs.get(i).map(String::as_str).unwrap_or("");
    match (s(1), segs.len()) {
        ("stats", 2) => match req.method {
            Method::GET => blocking(move || {
                let mut v = ccr::ccr_stats(&node)?;
                if human(&req) {
                    humanize_status(&mut v);
                }
                Ok(Resp::ok(v))
            })
            .await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("auto_follow", 2) => match req.method {
            Method::GET => blocking(move || Ok(Resp::ok(ccr::get_auto_follow_patterns(&node, None)?))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("auto_follow", 3) => {
            let name = s(2).to_string();
            match req.method {
                Method::GET => blocking(move || Ok(Resp::ok(ccr::get_auto_follow_patterns(&node, Some(&name))?))).await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    ccr::put_auto_follow_pattern(&node, &name, &body)?;
                    Ok(Resp::ack())
                })
                .await,
                Method::DELETE => blocking(move || {
                    ccr::delete_auto_follow_pattern(&node, &name)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("auto_follow", 4) if matches!(s(3), "pause" | "resume") => {
            let name = s(2).to_string();
            let active = s(3) == "resume";
            match req.method {
                Method::POST => blocking(move || {
                    ccr::activate_auto_follow_pattern(&node, &name, active)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["POST"]),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `ActiveShardCount` of `wait_for_active_shards`: whether put-follow waits.
fn waits_for_shards(req: &Req) -> Result<bool, EsError> {
    match req.param("wait_for_active_shards") {
        None | Some("0") | Some("") | Some("index-setting") => Ok(false),
        Some("all") => Ok(true),
        Some(v) => match v.parse::<i64>() {
            Ok(n) if n >= 0 => Ok(n > 0),
            _ => Err(EsError::illegal_argument(format!("cannot parse ActiveShardCount[{v}]"))),
        },
    }
}

/// `/{index}/_ccr/...`
pub async fn handle_index(node: Arc<Node>, req: Req, index: String) -> R {
    if req.segs.len() != 3 {
        return crate::not_found_handler(&req);
    }
    let action = req.seg(2).to_string();
    let m = req.method.clone();
    match action.as_str() {
        "follow" => match m {
            Method::PUT => blocking(move || {
                let body = req.json()?;
                let mut r = ccr::PutFollowRequest::parse(&index, &body)?;
                r.wait_for_active_shards = waits_for_shards(&req)?;
                let (created, acked, started) = ccr::put_follow(&node, &r)?;
                Ok(Resp::ok(json!({"follow_index_created": created, "follow_index_shards_acked": acked, "index_following_started": started})))
            })
            .await,
            _ => method_not_allowed(&req, &["PUT"]),
        },
        "pause_follow" => match m {
            Method::POST => blocking(move || {
                ccr::pause_follow(&node, &index)?;
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        "resume_follow" => match m {
            Method::POST => blocking(move || {
                let body = req.json()?;
                let p = ccr::parse_resume_body(&body)?;
                ccr::resume_follow(&node, &index, &p)?;
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        "unfollow" => match m {
            Method::POST => blocking(move || {
                ccr::unfollow(&node, &index)?;
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        "info" => match m {
            Method::GET => blocking(move || Ok(Resp::ok(ccr::follow_info(&node, &index)?))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        "stats" => match m {
            Method::GET => blocking(move || {
                let mut v = ccr::follow_stats(&node, &index)?;
                if human(&req) {
                    humanize_status(&mut v);
                }
                Ok(Resp::ok(v))
            })
            .await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        "forget_follower" => match m {
            Method::POST => blocking(move || {
                let body = req.json()?;
                Ok(Resp::ok(ccr::forget_follower(&node, &index, &body)?))
            })
            .await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

fn following(svc: &elasticsearch_engine::index::IndexService) -> bool {
    svc.setting(ccr::FOLLOWING_INDEX_SETTING).as_deref() == Some("true")
}

/// `CcrRequests.CCR_PUT_MAPPING_REQUEST_VALIDATOR`: the mapping of a
/// following index is replicated from its leader.
pub(crate) fn check_put_mapping(names: &[&elasticsearch_engine::index::IndexService]) -> Result<(), EsError> {
    let f: Vec<&str> = names.iter().filter(|s| following(s)).map(|s| s.name.as_str()).collect();
    if f.is_empty() {
        return Ok(());
    }
    Err(EsError::new(
        403,
        "status_exception",
        format!("can't put mapping to the following indices [{}]; the mapping of the following indices are self-replicated from its leader indices", f.join(", ")),
    ))
}

/// `CcrRequests.CCR_INDICES_ALIASES_REQUEST_VALIDATOR`: aliases of
/// following indices are replicated from their leaders.
pub(crate) fn check_alias_actions(node: &Node, actions: &[Value]) -> Result<(), EsError> {
    let mut f: Vec<String> = Vec::new();
    for a in actions {
        let Some(o) = a.as_object() else { continue };
        for body in o.values() {
            let mut names: Vec<String> = body.get("index").and_then(Value::as_str).map(|s| vec![s.to_string()]).unwrap_or_default();
            if let Some(arr) = body.get("indices").and_then(Value::as_array) {
                names.extend(arr.iter().filter_map(|x| x.as_str().map(str::to_string)));
            }
            for n in names {
                let opts = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: true, expand_closed: true, ..Default::default() };
                for t in node.resolve(&n, opts).unwrap_or_default() {
                    if following(&t.svc) && !f.contains(&t.svc.name) {
                        f.push(t.svc.name.clone());
                    }
                }
            }
        }
    }
    if f.is_empty() {
        return Ok(());
    }
    Err(EsError::new(
        403,
        "status_exception",
        format!("can't modify aliases on indices [{}]; aliases of following indices are self-replicated from their leader indices", f.join(", ")),
    ))
}
