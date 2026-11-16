//! Ownership of retained results: async search, ES|QL / SQL / EQL async
//! results and SQL cursors are bound to the subject that created them
//! (ES stores the creating request's security headers with the result and
//! checks them on every access: `AsyncTaskIndexService`,
//! `AsyncSearchSecurity`, `SecuritySearchOperationListener`).
//!
//! * Get, keep-alive extension and stop: the creator only (user + realm,
//!   or the same API key; run-as aware).
//! * Status: the creator, or a subject granted the API's status action
//!   (the `monitor` cluster privilege).
//! * Delete: the creator, or a subject with the `cancel_task` privilege
//!   (`manage` / superuser include it).
//!
//! Anyone else gets the same answer as for a missing result, so an id
//! reveals nothing about another subject's results.

use crate::Req;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::owner::ResourceOwner;
use elasticsearch_engine::security::{Authn, Permission};

/// The action whose privilege (`cancel_task`) lets a subject delete others' results.
pub const CANCEL_TASK: &str = "cluster:admin/tasks/cancel";

/// What a request does with a retained result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// Get (with keep-alive extension), stop, cursor continuation.
    Read,
    /// Status: also allowed with this cluster action (`cluster:monitor/...`).
    Status(&'static str),
    /// Delete / close: also allowed with `cancel_task`.
    Delete,
}

/// The owner a result created by `req` is bound to (`None` with security off).
pub fn owner_of(node: &Node, req: &Req) -> Option<ResourceOwner> {
    if !node.security.enabled() {
        return None;
    }
    req.access.as_ref().map(|a| ResourceOwner::of(a.user()))
}

/// May `req`'s subject `use_` a result created by `stored`? `Err(missing())` if not.
pub fn check(node: &Node, req: &Req, stored: Option<&ResourceOwner>, use_: Use, missing: impl FnOnce() -> EsError) -> Result<(), EsError> {
    if !node.security.enabled() {
        return Ok(());
    }
    // In-process requests the engine builds itself carry no subject.
    let Some(access) = req.access.as_deref() else { return Ok(()) };
    check_authn(node, access.user(), Some(access.permission()), stored, use_, missing)
}

/// [`check`] for an authenticated subject (`perm` computed when `None`).
pub fn check_authn(node: &Node, caller: &Authn, perm: Option<&Permission>, stored: Option<&ResourceOwner>, use_: Use, missing: impl FnOnce() -> EsError) -> Result<(), EsError> {
    // Created with security off: not bound to anyone.
    let Some(stored) = stored else { return Ok(()) };
    if caller.auth_type == "internal" || ResourceOwner::of(caller).can_access(stored) {
        return Ok(());
    }
    let computed;
    let perm = match perm {
        Some(p) => p,
        None => {
            computed = node.security.permission(caller);
            &computed
        }
    };
    let granted = match use_ {
        Use::Read => false,
        Use::Status(action) => perm.check_cluster(action),
        Use::Delete => perm.check_cluster(CANCEL_TASK),
    };
    if granted {
        Ok(())
    } else {
        Err(missing())
    }
}

/// The subject of a request served outside the engine's security layer
/// (the cluster front's own async-search contexts): its `Authorization`
/// header, then `es-security-runas-user`. `None` if it does not
/// authenticate (such a request was already refused at the edge).
/// Expensive password verification runs on the crypto pool
/// ([`super::edge::verify`]), never on the calling runtime thread.
pub async fn authenticate_headers(node: &std::sync::Arc<Node>, authorization: Option<&str>, run_as: Option<&str>) -> Option<Authn> {
    let a = match super::edge::verify(node, super::credentials_from(authorization), None).await {
        super::edge::Verified::Authenticated(a) => a,
        _ => return None,
    };
    match run_as.filter(|t| !t.is_empty()) {
        Some(target) => node.security.run_as(a, target).ok(),
        None => Some(a),
    }
}
