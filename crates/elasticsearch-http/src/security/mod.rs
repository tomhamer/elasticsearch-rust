//! Security layer (only when `xpack.security.enabled`): authenticate every
//! request, serve `/_security/*`, authorize the transport action behind the
//! route, then hand the (possibly index-rewritten) request to the router.
//! With security disabled the router never enters this module except for
//! `/_security/*`, which answers like ES without security (no handler).

pub mod access;
pub mod actions;
mod api;
pub(crate) mod authz;
mod query;

use crate::{Req, Resp, R};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{AuthFail, Authn, Credentials, Permission, Principal};
use serde_json::{json, Value};
use std::sync::Arc;

/// `/_security/*` with security disabled: ES registers no handlers.
pub fn disabled(req: &Req) -> R {
    // Real ES (8.19, security disabled) answers `GET|PUT|DELETE /_security/<x>`
    // with 405 `Allow: POST` (a generic `POST /{a}/{b}` route matches the
    // two-segment path); deeper paths and POST are plain "no handler".
    if req.segs.len() == 2 && matches!(req.method.as_str(), "GET" | "PUT" | "DELETE") {
        return crate::method_not_allowed(req, &["POST"]);
    }
    crate::not_found_handler(req)
}

/// Entry point for every request when security is enabled.
pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let authn = if req.extensions.get::<crate::SystemIdentity>().is_some() {
        elasticsearch_engine::security::Security::system()
    } else {
        match node.security.authenticate(credentials(&req)) {
            Ok(a) => a,
            Err(f) => return Ok(auth_failure(&node, f, &req)),
        }
    };
    let authn = match req.headers.get("es-security-runas-user").and_then(|v| v.to_str().ok()) {
        Some(target) if !target.is_empty() => {
            let target = target.to_string();
            match node.security.run_as(authn.clone(), &target) {
                Ok(a) => a,
                Err(()) => {
                    let perm = node.security.permission(&authn);
                    let action = actions::classify(&req).action;
                    let msg = format!(
                        "action [{action}] is unauthorized for {}{}, because {} is unauthorized to run as [{target}]",
                        subject_text(&authn),
                        roles_text(&authn, &perm),
                        subject_text(&authn)
                    );
                    return Err(EsError::new(403, "security_exception", msg));
                }
            }
        }
        _ => authn,
    };
    // An API key restricted to workflows may only call the REST handlers of
    // those workflows (`WorkflowResolver`; `RBACEngine.resolveAuthorizationInfo`).
    if let Some(workflows) = node.security.workflow_restriction(&authn) {
        if !request_workflow(&req).is_some_and(|w| workflows.iter().any(|x| x == w)) {
            let action = actions::classify(&req).action;
            let msg = format!("action [{action}] is unauthorized for {}", subject_text(&authn));
            return Err(EsError::new(403, "security_exception", msg).caused_by(EsError::new(403, "role_restriction_exception", "access restricted by workflow")));
        }
    }
    // ES runs `BaseRestHandler` validation after authentication.
    crate::spec_validate::validate(&req, true)?;
    if req.seg(0) == "_security" {
        // The cluster frontend routes writes to the master and publishes
        // system-index metadata. The final engine API handler still performs
        // its route-specific authorization before executing the operation.
        if req.extensions.get::<crate::AuthorizedDispatch>().is_some() {
            return crate::dispatch(node, req).await;
        }
        return api::handle(node, req, authn).await;
    }
    authz::authorize_and_dispatch(node, req, authn).await
}

/// The workflow the request's REST handler belongs to (ES 8.19 defines one:
/// `search_application_query` = `search_application_query_action`).
fn request_workflow(req: &Req) -> Option<&'static str> {
    let s = &req.segs;
    let search_app_query = s.len() == 4 && s[0] == "_application" && s[1] == "search_application" && s[3] == "_search" && matches!(req.method.as_str(), "GET" | "POST");
    search_app_query.then_some("search_application_query")
}

fn credentials(req: &Req) -> Credentials {
    credentials_from(req.headers.get("authorization").and_then(|v| v.to_str().ok()))
}

/// Whether a request with this `Authorization` header authenticates (always,
/// with security off). ES authenticates before `RestController` charges the
/// in-flight breaker, so an unauthenticated request gets its 401, not a 429.
pub fn authenticated(node: &Node, authorization: Option<&str>) -> bool {
    !node.security.enabled() || node.security.authenticate(credentials_from(authorization)).is_ok()
}

fn credentials_from(header: Option<&str>) -> Credentials {
    let Some(h) = header else { return Credentials::None };
    let h = h.trim();
    let (scheme, rest) = h.split_once(' ').unwrap_or((h, ""));
    let rest = rest.trim();
    match scheme.to_ascii_lowercase().as_str() {
        "basic" => {
            use base64::Engine;
            match base64::engine::general_purpose::STANDARD.decode(rest).ok().and_then(|b| String::from_utf8(b).ok()) {
                Some(s) => match s.split_once(':') {
                    Some((u, p)) => Credentials::Basic(u.to_string(), p.to_string()),
                    None => Credentials::None,
                },
                None => Credentials::None,
            }
        }
        "apikey" => Credentials::ApiKey(rest.to_string()),
        "bearer" => Credentials::Bearer(rest.to_string()),
        _ => Credentials::None,
    }
}

/// The request URI as ES prints it in authentication errors.
fn request_uri(req: &Req) -> String {
    if req.params.is_empty() {
        return req.path.clone();
    }
    let mut ps: Vec<String> = req.params.iter().map(|(k, v)| if v.is_empty() { k.clone() } else { format!("{k}={v}") }).collect();
    ps.sort();
    format!("{}?{}", req.path, ps.join("&"))
}

/// `WWW-Authenticate` values, most preferred first (ES `authSchemePriority`).
fn www_authenticate(node: &Node) -> Vec<String> {
    let cfg = node.security.cfg.read();
    let mut v = Vec::new();
    if cfg.token_enabled {
        v.push("Bearer realm=\"security\"".to_string());
    }
    if cfg.api_key_enabled {
        v.push("ApiKey".to_string());
    }
    v.push("Basic realm=\"security\", charset=\"UTF-8\"".to_string());
    v
}

fn unauthorized(node: &Node, reason: String, extra: Option<Value>, headers: Vec<String>) -> Resp {
    let headers = if headers.is_empty() { www_authenticate(node) } else { headers };
    let hv = if headers.len() == 1 { json!(headers[0]) } else { json!(headers) };
    let mut e = EsError::new(401, "security_exception", reason);
    if let Some(x) = extra {
        e = e.with("additional_unsuccessful_credentials", x);
    }
    e = e.with("header", json!({"WWW-Authenticate": hv}));
    let mut r = Resp::json(401, e.to_json());
    for h in headers {
        r.headers.push(("WWW-Authenticate", h));
    }
    r
}

fn auth_failure(node: &Node, f: AuthFail, req: &Req) -> Resp {
    let uri = request_uri(req);
    match f {
        AuthFail::Missing => unauthorized(node, format!("missing authentication credentials for REST request [{uri}]"), None, vec![]),
        AuthFail::User(u) => unauthorized(node, format!("unable to authenticate user [{u}] for REST request [{uri}]"), None, vec![]),
        AuthFail::Other(msgs) if msgs.is_empty() => unauthorized(node, format!("missing authentication credentials for REST request [{uri}]"), None, vec![]),
        AuthFail::Other(msgs) => {
            let extra = if msgs.len() == 1 { json!(msgs[0]) } else { json!(msgs) };
            unauthorized(node, "unable to authenticate with provided credentials and anonymous access is not allowed for this request".into(), Some(extra), vec![])
        }
        AuthFail::Reason(m) => unauthorized(node, m, None, vec![]),
        AuthFail::TokenExpired => unauthorized(
            node,
            "token expired".into(),
            None,
            vec!["Bearer realm=\"security\", error=\"invalid_token\", error_description=\"The access token expired\"".into()],
        ),
    }
}

/// `user [x]` / `API key id [id] of user [x]` / `service account [x]` (+ run as).
pub(crate) fn subject_text(a: &Authn) -> String {
    let mut t = match &a.principal {
        Principal::ServiceAccount { .. } => format!("service account [{}]", a.username),
        Principal::ApiKey { id, .. } => format!("API key id [{id}] of user [{}]", a.username),
        _ => format!("user [{}]", a.username),
    };
    if let Some(r) = &a.run_as {
        t.push_str(&format!(" run as [{}]", r.username));
    }
    t
}

pub(crate) fn roles_text(a: &Authn, perm: &Permission) -> String {
    if matches!(a.principal, Principal::ApiKey { .. } | Principal::ServiceAccount { .. }) {
        return String::new();
    }
    let mut s = format!(" with effective roles [{}]", perm.effective_roles.join(","));
    if !perm.missing_roles.is_empty() {
        let mut m = perm.missing_roles.clone();
        m.sort();
        s.push_str(&format!(" (assigned roles [{}] were not found)", m.join(",")));
    }
    s
}

/// The 403 (or, for anonymous without `authz_exception`, 401) for a denied action.
pub(crate) fn denied(node: &Node, a: &Authn, perm: &Permission, action: &str, indices: Option<&[String]>) -> EsError {
    use elasticsearch_engine::security::privilege;
    let mut msg = format!("action [{action}] is unauthorized for {}{}", subject_text(a), roles_text(a, perm));
    if let Some(ix) = indices {
        if !ix.is_empty() {
            let restricted = ix.iter().all(|n| elasticsearch_engine::security::roles::is_restricted(n));
            msg.push_str(&format!(" on {}indices [{}]", if restricted { "restricted " } else { "" }, ix.join(",")));
        }
    }
    if privilege::is_cluster_action(action) {
        let p = privilege::cluster_privs_granting(action);
        if !p.is_empty() {
            msg.push_str(&format!(", this action is granted by the cluster privileges [{}]", p.join(",")));
        }
    } else if privilege::is_index_action(action) {
        let p = privilege::index_privs_granting(action);
        if !p.is_empty() {
            msg.push_str(&format!(", this action is granted by the index privileges [{}]", p.join(",")));
        }
    }
    if matches!(a.principal, Principal::Anonymous) && !node.security.cfg.read().anonymous_authz_exception {
        let hv = www_authenticate(node);
        return EsError::new(401, "security_exception", format!("action [{action}] requires authentication")).with("header", json!({"WWW-Authenticate": hv}));
    }
    EsError::new(403, "security_exception", msg)
}

/// Cluster-privilege check for an action; `Err` is the denial.
pub(crate) fn check_cluster(node: &Node, a: &Authn, perm: &Permission, action: &str) -> Result<(), EsError> {
    if perm.check_cluster(action) {
        Ok(())
    } else {
        Err(denied(node, a, perm, action, None))
    }
}
