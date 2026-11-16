//! Transform REST handlers (`/_transform/...`) over
//! `elasticsearch_engine::transform`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ES_VERSION};
use elasticsearch_engine::security::{Authn, Permission, Principal};
use elasticsearch_engine::transform;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

fn with_warnings(mut r: Resp, warnings: Vec<String>) -> Resp {
    for w in warnings {
        r.headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"{w}\"", crate::cluster::BUILD_HASH)));
    }
    r
}

fn usize_param(req: &Req, k: &str, d: usize) -> Result<usize, EsError> {
    match req.param(k) {
        None => Ok(d),
        Some(v) => v.parse::<i64>().map(|x| x.max(0) as usize).map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{v}]"))),
    }
}

fn timeout(req: &Req) -> Duration {
    req.param("timeout").and_then(elasticsearch_engine::date::parse_time_value).filter(|v| *v > 0).map(|v| Duration::from_millis(v as u64)).unwrap_or(Duration::from_secs(30))
}

fn body_obj(req: &Req) -> Result<Value, EsError> {
    let b = req.json()?;
    Ok(if b.is_null() { json!({}) } else { b })
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let segs: Vec<String> = req.segs.clone();
    let s = |i: usize| segs.get(i).map(String::as_str).unwrap_or("");
    let m = req.method.clone();
    let n = segs.len();
    let svc = transform::service(&node);
    install_source_access();
    // On a cluster the elected master serves transforms from the internal
    // index (loaded once per master term).
    {
        let svc = svc.clone();
        blocking(move || svc.ensure_current().map(|_| Resp::ack())).await?;
    }
    let ops = cluster_ops(&req);
    match (n, s(1), s(2)) {
        (1, _, _) => match m {
            Method::GET => blocking(move || get(&svc, &req, "_all")).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        (2, "_preview", _) => match m {
            Method::GET | Method::POST => blocking(move || preview(&node, &svc, &ops, &req, None)).await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        (2, "_upgrade", _) => match m {
            Method::POST => Ok(Resp::ok(svc.upgrade())),
            _ => method_not_allowed(&req, &["POST"]),
        },
        // `GET /_transform/_stats`: stats of every transform.
        (2, "_stats", _) => match m {
            Method::GET => blocking(move || {
                let from = usize_param(&req, "from", 0)?;
                let size = usize_param(&req, "size", 100)?;
                let anm = req.bool_param("allow_no_match")?.unwrap_or(true);
                Ok(Resp::ok(svc.stats(&node, "_all", from, size, anm)?))
            })
            .await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        (2, "_node_stats", _) => match m {
            Method::GET => Ok(Resp::ok(svc.node_stats())),
            _ => method_not_allowed(&req, &["GET"]),
        },
        (2, "set_upgrade_mode", _) => match m {
            Method::POST => {
                let enabled = req.flag("enabled")?;
                blocking(move || {
                    svc.set_upgrade_mode(enabled)?;
                    Ok(Resp::ack())
                })
                .await
            }
            _ => method_not_allowed(&req, &["POST"]),
        },
        (2, id, _) => {
            let id = id.to_string();
            match m {
                Method::GET => blocking(move || get(&svc, &req, &id)).await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    if body.is_null() {
                        return Err(EsError::new(400, "parse_exception", "request body is required"));
                    }
                    let defer = req.flag("defer_validation")?;
                    let auth = check_privileges(&node, &req, "create", &id, &body, !defer)?;
                    let w = svc.put(&node, &ops, &id, &body, defer, auth.as_ref()).map_err(|e| fix_pos(&req, e))?;
                    Ok(with_warnings(Resp::ack(), w))
                })
                .await,
                Method::DELETE => blocking(move || {
                    svc.delete(&node, &ops, &id, req.flag("force")?, req.flag("delete_dest_index")?)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        (3, id, action) => {
            let id = id.to_string();
            match (action, m.clone()) {
                ("_preview", Method::GET | Method::POST) => blocking(move || preview(&node, &svc, &ops, &req, Some(&id))).await,
                ("_start", Method::POST) => blocking(move || {
                    if let Some(cfg) = svc.stored_config(&id) {
                        check_privileges(&node, &req, "start", &id, &cfg, false)?;
                    }
                    svc.start(&node, &ops, &id, req.param("from"))?;
                    Ok(Resp::ack())
                })
                .await,
                ("_stop", Method::POST) => blocking(move || {
                    let force = req.flag("force")?;
                    let wfc = req.flag("wait_for_checkpoint")?;
                    let wait = req.flag("wait_for_completion")?;
                    let anm = req.bool_param("allow_no_match")?.unwrap_or(true);
                    svc.stop(&id, force, wfc, wait, anm, timeout(&req))?;
                    Ok(Resp::ack())
                })
                .await,
                ("_reset", Method::POST) => blocking(move || {
                    svc.reset(&node, &ops, &id, req.flag("force")?)?;
                    Ok(Resp::ack())
                })
                .await,
                ("_schedule_now", Method::POST) => blocking(move || {
                    svc.schedule_now(&id)?;
                    Ok(Resp::ack())
                })
                .await,
                ("_update", Method::POST) => blocking(move || {
                    let body = body_obj(&req)?;
                    let defer = req.flag("defer_validation")?;
                    let auth = match svc.stored_config(&id) {
                        Some(mut cfg) => {
                            for k in ["source", "dest"] {
                                if let Some(v) = body.get(k) {
                                    cfg[k] = v.clone();
                                }
                            }
                            check_privileges(&node, &req, "update", &id, &cfg, !defer)?
                        }
                        None => None,
                    };
                    let (cfg, w) = svc.update(&node, &ops, &id, &body, defer, auth.as_ref()).map_err(|e| fix_pos(&req, e))?;
                    Ok(with_warnings(Resp::ok(cfg), w))
                })
                .await,
                ("_stats", Method::GET) => blocking(move || {
                    let from = usize_param(&req, "from", 0)?;
                    let size = usize_param(&req, "size", 100)?;
                    let anm = req.bool_param("allow_no_match")?.unwrap_or(true);
                    Ok(Resp::ok(svc.stats(&node, &id, from, size, anm)?))
                })
                .await,
                ("_preview", _) => method_not_allowed(&req, &["GET", "POST"]),
                ("_stats", _) => method_not_allowed(&req, &["GET"]),
                ("_start" | "_stop" | "_reset" | "_schedule_now" | "_update", _) => method_not_allowed(&req, &["POST"]),
                _ => crate::not_found_handler(&req),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

fn get(svc: &transform::TransformService, req: &Req, id: &str) -> R {
    let from = usize_param(req, "from", 0)?;
    let size = usize_param(req, "size", 100)?;
    let anm = req.bool_param("allow_no_match")?.unwrap_or(true);
    let eg = req.flag("exclude_generated")?;
    Ok(Resp::ok(svc.get(id, from, size, anm, eg)?))
}

/// On a cluster, transform index work runs through the coordinator.
fn cluster_ops(req: &Req) -> transform::Ops {
    transform::Ops(req.extensions.get::<crate::ClusterRest>().map(|r| -> transform::RestCall { r.0.clone() }), None)
}

fn preview(node: &Node, svc: &transform::TransformService, ops: &transform::Ops, req: &Req, id: Option<&str>) -> R {
    // An absent body (`Null`) differs from `{}`: the latter is a config.
    let body = req.json()?;
    let cfg = match id {
        Some(id) => svc.stored_config(id),
        None => Some(body.clone()),
    };
    let auth = match &cfg {
        Some(cfg) if cfg.is_object() => check_privileges(node, req, "preview", id.unwrap_or("_preview"), cfg, false)?,
        _ => None,
    };
    let run_as = auth.map(|a| transform::RunAs::Identity(a.identity));
    let (v, w) = svc.preview(node, ops, id, &body, run_as.as_ref()).map_err(|e| fix_pos(req, e))?;
    Ok(with_warnings(Resp::ok(v), w))
}

/// 1-based `(line, col)` of byte offset `at`.
fn line_col(text: &str, at: usize) -> (usize, usize) {
    let before = &text[..at.min(text.len())];
    (before.matches('\n').count() + 1, before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1)
}

/// Byte offsets of top-level member `key`: (key, value start, value's last byte).
fn top_member(text: &str, key: &str) -> Option<(usize, usize, usize)> {
    let b = text.as_bytes();
    let skip_string = |mut i: usize| -> usize {
        i += 1;
        while i < b.len() && b[i] != b'"' {
            if b[i] == b'\\' {
                i += 1;
            }
            i += 1;
        }
        i
    };
    let skip_ws = |mut i: usize| -> usize {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    };
    let value_end = |start: usize| -> usize {
        match b.get(start) {
            Some(b'{') | Some(b'[') => {
                let mut depth = 0i32;
                let mut i = start;
                while i < b.len() {
                    match b[i] {
                        b'"' => i = skip_string(i),
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth == 0 {
                                return i;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                start
            }
            _ => start,
        }
    };
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let end = skip_string(i);
                if depth == 1 && text.get(i + 1..end) == Some(key) {
                    let colon = skip_ws(end + 1);
                    if b.get(colon) == Some(&b':') {
                        let vs = skip_ws(colon + 1);
                        return Some((i, vs, value_end(vs)));
                    }
                }
                i = end;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

/// Give a `[1:1]` transform config parse error the position ES reports
/// (the unknown key; the end of a sub-object missing a required field).
fn fix_pos(req: &Req, mut e: EsError) -> EsError {
    const P: &str = "[1:1] [data_frame_transform_config] ";
    if e.kind != "x_content_parse_exception" || !e.reason.starts_with(P) {
        return e;
    }
    let Ok(text) = req.body_str() else { return e };
    let rest = e.reason[P.len()..].to_string();
    let field = |prefix: &str| rest.strip_prefix(prefix).and_then(|r| r.strip_suffix(']')).map(str::to_string);
    let at = if let Some(k) = field("unknown field [") {
        top_member(text, &k).map(|(ks, _, _)| ks)
    } else if let Some(k) = field("failed to parse field [") {
        let required = e.caused_by.as_ref().is_some_and(|c| c.reason.starts_with("Required ["));
        top_member(text, &k).map(|(_, vs, ve)| if required { ve } else { vs })
    } else {
        None
    };
    if let Some(at) = at {
        let (l, c) = line_col(text, at);
        e.reason = format!("[{l}:{c}] [data_frame_transform_config] {rest}");
    }
    e
}

/// Rows of `_cat/transforms`.
pub(crate) fn cat_rows(node: &Node, req: &Req, id: Option<&str>) -> Result<Vec<Vec<String>>, EsError> {
    let svc = transform::service_ref(node);
    svc.ensure_current()?;
    let anm = req.bool_param("allow_no_match")?.unwrap_or(true);
    let rows = svc.cat_rows(node, id, anm)?;
    let fmt_time = |ms: Option<i64>| ms.map(|m| elasticsearch_engine::date::DateFormatter::default().format(m, elasticsearch_engine::date::Tz::UTC)).unwrap_or_default();
    let mut out = Vec::new();
    for (cfg, st) in rows {
        let stats = &st["stats"];
        let cp = &st["checkpointing"];
        let u = |k: &str| stats[k].as_u64().unwrap_or(0).to_string();
        let f2 = |k: &str| format!("{:.2}", stats[k].as_f64().unwrap_or(0.0));
        let progress = cp["next"]["checkpoint_progress"]["percent_complete"].as_f64().map(|p| format!("{p:.2}")).unwrap_or_default();
        let src = cfg["source"]["index"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(",")).unwrap_or_default();
        let kind = if cfg.get("sync").is_some() { "continuous" } else { "batch" };
        let page = cfg["settings"]["max_page_search_size"].as_u64().or_else(|| cfg["pivot"]["max_page_search_size"].as_u64()).unwrap_or(500);
        out.push(vec![
            cfg["id"].as_str().unwrap_or("").to_string(),
            st["state"].as_str().unwrap_or("").to_string(),
            cp["last"]["checkpoint"].as_u64().unwrap_or(0).to_string(),
            u("documents_processed"),
            progress,
            fmt_time(cp["last_search_time"].as_i64()),
            fmt_time(cp["changes_last_detected_at"].as_i64()),
            fmt_time(cfg["create_time"].as_i64()),
            cfg["version"].as_str().unwrap_or("").to_string(),
            src,
            cfg["dest"]["index"].as_str().unwrap_or("").to_string(),
            cfg["dest"]["pipeline"].as_str().unwrap_or("").to_string(),
            cfg["description"].as_str().unwrap_or("").to_string(),
            kind.to_string(),
            cfg["frequency"].as_str().unwrap_or("1m").to_string(),
            page.to_string(),
            cfg["settings"]["docs_per_second"].as_f64().map(|d| d.to_string()).unwrap_or_default(),
            st["reason"].as_str().unwrap_or("").to_string(),
            u("search_total"),
            u("search_failures"),
            u("search_time_in_ms"),
            u("index_total"),
            u("index_failures"),
            u("index_time_in_ms"),
            u("documents_indexed"),
            u("delete_time_in_ms"),
            u("documents_deleted"),
            u("trigger_count"),
            u("pages_processed"),
            u("processing_time_in_ms"),
            f2("exponential_avg_checkpoint_duration_ms"),
            f2("exponential_avg_documents_indexed"),
            f2("exponential_avg_documents_processed"),
        ]);
    }
    Ok(out)
}

// ---------------------------------------------------------------- security

/// The caller of a transform API when security is enabled: who they are
/// (the run-as target when impersonating) and what they may do.
fn caller(node: &Node, req: &Req) -> Option<(Authn, Permission)> {
    if !node.security.enabled() {
        return None;
    }
    let access = req.access.as_ref()?;
    let user = access.user().clone();
    let perm = node.security.permission(&user);
    Some((user, perm))
}

/// Who a transform created or updated by `user` runs as (the whole
/// authentication, as ES stores it: an API key keeps its own role
/// descriptors and its owner's limit), and the `authorization` ES reports.
fn caller_auth(user: &Authn) -> transform::CallerAuth {
    let effective = user.run_as.as_deref().unwrap_or(user);
    let authorization = match &effective.principal {
        Principal::ApiKey { id, name, .. } => json!({"api_key": {"id": id, "name": name}}),
        _ => json!({"roles": effective.roles}),
    };
    transform::CallerAuth { identity: elasticsearch_engine::security::owner::StoredAuthn::of(user), authorization }
}

/// The index names of a config's `source.index` (a string, comma lists
/// included, or an array).
fn source_names(cfg: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match &cfg["source"]["index"] {
        Value::String(s) => out.extend(s.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)),
        Value::Array(a) => {
            for v in a.iter().filter_map(Value::as_str) {
                out.extend(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string));
            }
        }
        _ => {}
    }
    out
}

/// ES `TransformPrivilegeChecker`: the caller needs `read` and
/// `view_index_metadata` on every source, and (`check_dest`) `read`,
/// `index` and, for a new destination, `create_index` on the destination.
/// Returns the identity the transform runs as (`None`: security disabled).
fn check_privileges(node: &Node, req: &Req, verb: &str, id: &str, cfg: &Value, check_dest: bool) -> Result<Option<transform::CallerAuth>, EsError> {
    let Some((user, perm)) = caller(node, req) else { return Ok(None) };
    if !perm.is_superuser {
        let mut missing: Vec<String> = Vec::new();
        for s in source_names(cfg) {
            let lacks: Vec<&str> = ["read", "view_index_metadata"].into_iter().filter(|p| !perm.has_index_privilege(&s, p)).collect();
            if !lacks.is_empty() {
                missing.push(format!("{s}:[{}]", lacks.join(", ")));
            }
        }
        if check_dest {
            if let Some(d) = cfg["dest"]["index"].as_str() {
                let mut need = vec!["index", "read"];
                if node.get_index(d).is_none() {
                    need.insert(0, "create_index");
                }
                let lacks: Vec<&str> = need.into_iter().filter(|p| !perm.has_index_privilege(d, p)).collect();
                if !lacks.is_empty() {
                    missing.push(format!("{d}:[{}]", lacks.join(", ")));
                }
            }
        }
        if !missing.is_empty() {
            let name = user.run_as.as_deref().unwrap_or(&user).username.clone();
            return Err(EsError::new(403, "security_exception", format!("Cannot {verb} transform [{id}] because user {name} lacks the required permissions [{}]", missing.join(", "))));
        }
    }
    Ok(Some(caller_auth(&user)))
}

/// Standalone checkpoints read their sources as the transform's user:
/// targets are limited to the indices the user may search, with the user's
/// DLS/FLS stamped on them (a cluster runs those searches through its
/// coordinator as the user instead).
fn install_source_access() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        transform::set_source_access(Some(Arc::new(|node: &Node, subject: &transform::RunAs, names: &[String], targets| {
            if !node.security.enabled() {
                return Ok(targets);
            }
            let authn = match subject {
                // The stored authentication: roles (an API key's descriptors
                // limited by its owner's) resolve now, as ES resolves them.
                transform::RunAs::Identity(stored) => stored.to_authn(),
                transform::RunAs::User(user) => node.security.lookup_user(user).ok_or_else(|| EsError::new(403, "security_exception", format!("transform runs as user [{user}], who cannot be found")))?,
            };
            let perm = node.security.permission(&authn);
            const SEARCH: &str = "indices:data/read/search";
            let explicit: Vec<String> = names.iter().filter(|n| !n.contains('*')).cloned().collect();
            let access = crate::security::access::ReadAccess::build(node, &authn, &perm, SEARCH, names)?;
            access.authorize_names(node, SEARCH, &explicit)?;
            let allowed: Vec<_> = targets.into_iter().filter(|t| perm.is_superuser || crate::security::authz::allowed(node, &perm, SEARCH, &t.svc.name)).collect();
            access.apply(allowed)
        })));
    });
}
