//! Document- and field-level security (DLS / FLS) enforcement glue.
//!
//! The authorization layer builds a [`ReadAccess`] for every request of a
//! non-superuser and attaches it to the [`crate::Req`]. Read paths pass it
//! to [`ReadAccess::apply`], which stamps each target index with the caller's
//! [`IndexAccess`]; the engine then ANDs the DLS filter into every query,
//! filters `_source`, and hides ungranted fields from the field lookup.
//!
//! Default-deny: when the access of an index cannot be computed, the request
//! fails; actions this layer cannot filter are refused on restricted indices
//! (see [`gate`]).

use super::{roles_text, subject_text};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::Target;
use elasticsearch_engine::security::roles::{FieldAccess, IndexAccess};
use elasticsearch_engine::security::{Authn, Permission};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

const SEARCH: &str = "indices:data/read/search";

/// What the caller may read, per concrete index, for one request.
pub struct ReadAccess {
    perm: Permission,
    user: Authn,
    by_index: HashMap<String, Arc<IndexAccess>>,
    /// The subject (role or API key) carries DLS/FLS somewhere.
    restricted: bool,
}

fn security_denied(user: &Authn, perm: &Permission, action: &str, indices: &[String]) -> EsError {
    EsError::new(
        403,
        "security_exception",
        format!("action [{action}] is unauthorized for {}{} on indices [{}]", subject_text(user), roles_text(user, perm), indices.join(",")),
    )
}

/// Concrete indices a requested name (index, alias or data stream) stands for.
fn concretes(node: &Node, name: &str) -> Vec<String> {
    if node.get_index(name).is_some() {
        return vec![name.to_string()];
    }
    if let Some(b) = node.data_stream_indices(name) {
        return b;
    }
    node.all_indices().iter().filter(|s| s.aliases().contains_key(name)).map(|s| s.name.clone()).collect()
}

impl ReadAccess {
    /// Compute the access for `names` (the authorized index/alias/data stream
    /// names of the request) under `action`. Errors deny the request.
    pub fn build(node: &Node, user: &Authn, perm: &Permission, action: &str, names: &[String]) -> Result<ReadAccess, EsError> {
        let restricted = perm.has_dls_fls();
        let mut by_index = HashMap::new();
        if restricted {
            let mut vias: HashMap<String, Vec<String>> = HashMap::new();
            for n in names {
                for c in concretes(node, n) {
                    let v = vias.entry(c.clone()).or_default();
                    if *n != c && !v.contains(n) {
                        v.push(n.clone());
                    }
                }
            }
            for (c, v) in vias {
                let mut ns: Vec<&str> = vec![c.as_str()];
                ns.extend(v.iter().map(String::as_str));
                let acc = perm.index_access(action, &ns, user).map_err(|_| security_denied(user, perm, action, std::slice::from_ref(&c)))?;
                by_index.insert(c, Arc::new(acc));
            }
        }
        Ok(ReadAccess { perm: perm.clone(), user: user.clone(), by_index, restricted })
    }

    /// Merge another request item's access: every restriction of both holds
    /// (used for `_msearch`, whose items share one request).
    pub fn merge(&mut self, other: ReadAccess) {
        for (k, v) in other.by_index {
            match self.by_index.remove(&k) {
                Some(mine) => {
                    self.by_index.insert(k, Arc::new((*mine).clone().and((*v).clone())));
                }
                None => {
                    self.by_index.insert(k, v);
                }
            }
        }
        self.restricted |= other.restricted;
    }

    /// Authorize a sub-request the handler executes as the user (ES runs it
    /// through the security filter): `action` on every explicit name.
    pub fn authorize_names(&self, node: &Node, action: &str, names: &[String]) -> Result<(), EsError> {
        let bad: Vec<String> = names
            .iter()
            .filter(|n| !(self.perm.is_superuser && !elasticsearch_engine::security::roles::is_restricted(n)) && !super::authz::allowed(node, &self.perm, action, n))
            .cloned()
            .collect();
        if bad.is_empty() {
            Ok(())
        } else {
            Err(super::denied(node, &self.user, &self.perm, action, Some(&bad)))
        }
    }

    pub fn is_restricted(&self) -> bool {
        self.restricted
    }

    /// The authenticated subject (the run-as target when impersonating).
    pub fn user(&self) -> &Authn {
        &self.user
    }

    /// Does any index of this request carry DLS/FLS?
    pub fn any_index_restricted(&self) -> bool {
        self.by_index.values().any(|a| a.is_restricted())
    }

    fn access_of(&self, index: &str, action: &str) -> Result<Arc<IndexAccess>, EsError> {
        if let Some(a) = self.by_index.get(index) {
            return Ok(a.clone());
        }
        // Not part of the authorized request (e.g. a point-in-time over other indices):
        // fall back to the groups matching the concrete name, or deny.
        self.perm.index_access(action, &[index], &self.user).map(Arc::new).map_err(|_| security_denied(&self.user, &self.perm, action, &[index.to_string()]))
    }

    /// Stamp each target with the caller's access. Default-deny: a target whose
    /// access cannot be computed fails the whole request.
    pub fn apply(&self, targets: Vec<Target>) -> Result<Vec<Target>, EsError> {
        if !self.restricted {
            return Ok(targets);
        }
        targets
            .into_iter()
            .map(|mut t| {
                let a = self.access_of(&t.svc.name, SEARCH)?;
                // Keep a stricter access a previous stage (point in time, scroll) already attached.
                t.access = Some(match t.access.take() {
                    Some(prev) => Arc::new((*prev).clone().and((*a).clone())),
                    None => a,
                }).filter(|a| a.is_restricted());
                Ok(t)
            })
            .collect()
    }

    /// The access for single-document reads of one index (`action` e.g. `indices:data/read/get`).
    pub fn doc_access(&self, index: &str, action: &str) -> Result<Option<Arc<IndexAccess>>, EsError> {
        if !self.restricted {
            return Ok(None);
        }
        let a = self.access_of(index, action)?;
        Ok(if a.is_restricted() { Some(a) } else { None })
    }

    /// Terms-lookup / more-like-this document fetches are reads of the lookup
    /// index: they need the read privilege and honour its DLS/FLS.
    pub fn lookup_access(&self, concrete: &str, requested: &str) -> Result<Option<Arc<IndexAccess>>, EsError> {
        let action = "indices:data/read/get";
        if !self.perm.check_index(action, concrete) && !self.perm.check_index(action, requested) {
            return Err(security_denied(&self.user, &self.perm, action, &[requested.to_string()]));
        }
        if !self.restricted {
            return Ok(None);
        }
        let a = self.perm.index_access(action, &[concrete, requested], &self.user).map(Arc::new).map_err(|_| security_denied(&self.user, &self.perm, action, &[requested.to_string()]))?;
        Ok(if a.is_restricted() { Some(a) } else { None })
    }

    /// FLS of an index's mappings (`None` = show everything).
    pub fn field_access(&self, index: &str) -> Option<FieldAccess> {
        self.by_index.get(index).and_then(|a| a.fields.clone())
    }

    /// Indices of this request whose mappings must be filtered.
    pub fn fls_by_index(&self) -> HashMap<String, FieldAccess> {
        self.by_index.iter().filter_map(|(k, a)| a.fields.clone().map(|f| (k.clone(), f))).collect()
    }
}

/// How the security layer treats a transport action on DLS/FLS-restricted indices.
pub enum Gate {
    /// Not affected by DLS/FLS.
    Ignore,
    /// Enforced by [`ReadAccess::apply`] / document filtering / response filtering.
    Enforce,
    /// Refused with this error when any resolved index is restricted for the action.
    Deny(fn(&[String]) -> EsError),
}

fn deny_generic(indices: &[String]) -> EsError {
    EsError::new(403, "security_exception", format!("this action is not permitted on indices [{}] with field or document level security", indices.join(",")))
}

fn deny_update(_: &[String]) -> EsError {
    EsError::new(400, "security_exception", "Can't execute an update request if field or document level security is enabled")
}

fn deny_rollup(indices: &[String]) -> EsError {
    EsError::new(403, "security_exception", format!("Rollup forbidden for index [{}] with document level or field level security settings.", indices.join(",")))
}

fn deny_resize(_: &[String]) -> EsError {
    EsError::new(403, "security_exception", "Resize requests are not allowed for users when field or document level security is enabled on the source index")
}

fn deny_alias(_: &[String]) -> EsError {
    EsError::new(403, "security_exception", "Alias requests are not allowed for users who have field or document level security enabled on one of the indices")
}

pub fn deny_bulk_update() -> EsError {
    EsError::new(400, "security_exception", "Can't execute a bulk request with update requests embedded if field or document level security is enabled")
}

/// Classify a transport action. Every `indices:data/read/*` action that is not
/// explicitly enforced is refused on restricted indices.
pub fn gate(action: &str) -> Gate {
    match action {
        "indices:data/read/search"
        | "indices:data/read/search/template"
        | "indices:data/read/msearch"
        | "indices:data/read/msearch/template"
        | "indices:data/read/get"
        | "indices:data/read/mget"
        | "indices:data/read/field_caps"
        | "indices:data/read/open_point_in_time"
        | "indices:data/read/knn_search"
        // Answers only from shards whose DLS rewrites to `match_all`, without FLS-hidden fields.
        | "indices:data/read/xpack/termsenum/list" => Gate::Enforce,
        "indices:admin/mappings/get" | "indices:admin/mappings/fields/get" | "indices:admin/get" => Gate::Enforce,
        "indices:data/write/update" => Gate::Deny(deny_update),
        "indices:admin/xpack/downsample" => Gate::Deny(deny_rollup),
        "indices:admin/resize" => Gate::Deny(deny_resize),
        "indices:admin/aliases" => Gate::Deny(deny_alias),
        "indices:data/write/update/byquery" | "indices:data/write/delete/byquery" | "indices:data/write/reindex" => Gate::Deny(deny_generic),
        a if a.starts_with("indices:data/read/") => Gate::Deny(deny_generic),
        _ => Gate::Ignore,
    }
}

pub fn deny_reindex(indices: &[String]) -> EsError {
    EsError::new(403, "security_exception", format!("reindex from indices [{}] with field or document level security is not permitted", indices.join(",")))
}

/// Concrete indices among `names` on which `action` is restricted for the subject.
/// An access that cannot be computed counts as restricted (default-deny).
fn restricted_among(node: &Node, user: &Authn, perm: &Permission, action: &str, names: &[String]) -> Vec<String> {
    let mut hit: Vec<String> = Vec::new();
    for n in names {
        for c in concretes(node, n) {
            let restricted = perm.index_access(action, &[c.as_str(), n.as_str()], user).map(|a| a.is_restricted()).unwrap_or(true);
            if restricted && !hit.contains(&c) {
                hit.push(c);
            }
        }
    }
    hit
}

/// Is `action` on any of `names` DLS/FLS restricted?
pub fn is_restricted(node: &Node, user: &Authn, perm: &Permission, action: &str, names: &[String]) -> bool {
    perm.has_dls_fls() && !restricted_among(node, user, perm, action, names).is_empty()
}

/// Refuse `action` when `err` applies: some index among `names` is restricted for it.
pub fn check_gate_with(node: &Node, user: &Authn, perm: &Permission, action: &str, names: &[String], err: fn(&[String]) -> EsError) -> Result<(), EsError> {
    if !perm.has_dls_fls() {
        return Ok(());
    }
    let hit = restricted_among(node, user, perm, action, names);
    if hit.is_empty() {
        Ok(())
    } else {
        Err(err(&hit))
    }
}

/// Refuse `action` when it is not enforceable on a restricted index among `names`.
pub fn check_gate(node: &Node, user: &Authn, perm: &Permission, action: &str, names: &[String]) -> Result<(), EsError> {
    match gate(action) {
        Gate::Deny(err) => check_gate_with(node, user, perm, action, names, err),
        _ => Ok(()),
    }
}

/// Actions without an index expression (ES|QL, SQL, EQL, async results...) that read
/// documents cannot be filtered: refuse them for any subject carrying DLS/FLS.
pub fn check_unscoped_read(perm: &Permission, action: &str) -> Result<(), EsError> {
    // ES|QL stamps every index it reads with the caller's DLS/FLS (`EsqlOptions::restrict`).
    let esql = action == "indices:data/read/esql" || action.starts_with("indices:data/read/esql/");
    if perm.has_dls_fls() && !esql && action.starts_with("indices:data/read/") && !matches!(action, "indices:data/read/scroll" | "indices:data/read/close_point_in_time" | "indices:data/read/msearch" | "indices:data/read/mget") {
        return Err(EsError::new(403, "security_exception", format!("action [{action}] is not permitted for subjects with field or document level security")));
    }
    Ok(())
}

// ----------------------------------------------------------------- mappings

fn join(prefix: &str, k: &str) -> String {
    if prefix.is_empty() {
        k.to_string()
    } else {
        format!("{prefix}.{k}")
    }
}

/// Keep only granted fields of a `properties` map; objects survive when a child does.
fn filter_properties(fa: &FieldAccess, props: &mut Map<String, Value>, prefix: &str) {
    let keys: Vec<String> = props.keys().cloned().collect();
    for k in keys {
        let path = join(prefix, &k);
        let keep = match props.get_mut(&k) {
            Some(Value::Object(def)) => {
                let had_props = def.get("properties").is_some_and(Value::is_object);
                if had_props {
                    if let Some(Value::Object(p)) = def.get_mut("properties") {
                        filter_properties(fa, p, &path);
                        !p.is_empty() || fa.allows(&path)
                    } else {
                        false
                    }
                } else if fa.allows(&path) {
                    if let Some(Value::Object(f)) = def.get_mut("fields") {
                        f.retain(|sub, _| fa.allows(&join(&path, sub)));
                        if f.is_empty() {
                            def.remove("fields");
                        }
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !keep {
            props.remove(&k);
        }
    }
}

fn filter_mappings_obj(fa: &FieldAccess, mappings: &mut Value) {
    if let Some(Value::Object(m)) = mappings.get_mut("properties") {
        filter_properties(fa, m, "");
    }
    // `runtime` fields are named at the top level.
    if let Some(Value::Object(r)) = mappings.get_mut("runtime") {
        r.retain(|n, _| fa.allows(n));
    }
}

/// Remove ungranted fields from a mapping-bearing response
/// (`GET /{idx}`, `GET /{idx}/_mapping`, `GET /{idx}/_mapping/field/{f}`).
pub fn filter_mapping_response(resp: &mut Value, fls: &HashMap<String, FieldAccess>) {
    let Some(top) = resp.as_object_mut() else { return };
    for (index, body) in top.iter_mut() {
        let Some(fa) = fls.get(index) else { continue };
        let Some(m) = body.get_mut("mappings") else { continue };
        match m {
            Value::Object(o) if o.contains_key("properties") || o.is_empty() || !o.values().any(|v| v.get("full_name").is_some()) => filter_mappings_obj(fa, m),
            Value::Object(o) => {
                // field mapping API: {field: {full_name, mapping}}
                o.retain(|_, v| v.get("full_name").and_then(Value::as_str).is_some_and(|n| fa.allows(n)));
            }
            _ => {}
        }
    }
}
