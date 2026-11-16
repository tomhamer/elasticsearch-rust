//! Index-level authorization: ES `IndicesAndAliasesResolver` semantics
//! (wildcards expand to authorized names only, explicit names must be
//! authorized), plus per-item handling for `_bulk`, `_mget`, `_msearch`.

use super::access::{self, ReadAccess};
use super::actions::{self, Act, Target};
use super::{check_cluster, denied};
use crate::{Req, Resp, RespBody, R};
use axum::body::Bytes;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{resolve_date_math, Node, ResolveOpts};
use elasticsearch_engine::security::privilege::{glob, is_cluster_action};
use elasticsearch_engine::security::{Authn, Permission};
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// ES's "no indices" replacement for an empty authorized expansion.
const NO_INDICES: &str = "*,-*";

pub async fn authorize_and_dispatch(node: Arc<Node>, mut req: Req, authn: Authn) -> R {
    let act = actions::classify(&req);
    let perm = node.security.permission(&authn);
    // Names (indices, aliases, data streams) the request was authorized on.
    let mut names: Vec<String> = Vec::new();
    // Every request carries the subject (ingest's `set_security_user`); read paths refine it below.
    req.access = Some(Arc::new(ReadAccess::build(&node, &authn, &perm, &act.action, &[])?));
    if !perm.is_superuser {
        req.extensions.insert(mapping_authz(&node, &authn, &perm));
        let catalog = req.extensions.get::<crate::AuthorizationIndexCatalog>().map(|c| Arc::new(c.0.clone()));
        req.extensions.insert(ingest_dest_authz(&node, &authn, &perm, catalog));
    }
    match act.target.clone() {
        Target::Authenticated => access::check_unscoped_read(&perm, &act.action)?,
        Target::None => {
            if is_cluster_action(&act.action) {
                check_cluster(&node, &authn, &perm, &act.action)?;
            } else if !perm.check_index_action(&act.action) {
                return Err(denied(&node, &authn, &perm, &act.action, None));
            }
            access::check_unscoped_read(&perm, &act.action)?;
        }
        Target::AnyIndex => {
            // ES|QL / SQL of a caller granted only remote clusters' indices
            // (`remote_indices`): its remote parts are authorized there, its
            // local ones by the query's index authorization below.
            let remote_reader = query_read_action(&act.action).is_some() && has_remote_indices(&node, &authn);
            if !perm.check_index_action(&act.action) && !remote_reader {
                return Err(denied(&node, &authn, &perm, &act.action, None));
            }
            // ES|QL, SQL, analyze...: names live in the body, so reads cannot be filtered.
            access::check_unscoped_read(&perm, &act.action)?;
            // ES|QL and SQL: the engine authorizes every index expression the query resolves.
            if let Some((read, strict)) = query_read_action(&act.action) {
                req.extensions.insert(query_index_authz(&node, &authn, &perm, read, strict));
            }
        }
        Target::Fixed(fixed) => {
            let fixed: Vec<String> = fixed.iter().map(|n| n.to_string()).collect();
            if !perm.check_index_action(&act.action) {
                return Err(denied(&node, &authn, &perm, &act.action, Some(&fixed)));
            }
            let bad: Vec<String> = fixed.iter().filter(|n| !perm.is_superuser && !allowed(&node, &perm, &act.action, n)).cloned().collect();
            if !bad.is_empty() {
                return Err(denied(&node, &authn, &perm, &act.action, Some(&bad)));
            }
            names = fixed;
        }
        Target::Seg(i) => names = authorize_seg(&node, &authn, &perm, &mut req, &act, i)?,
        Target::Bulk => return bulk(node, req, authn, perm).await,
        Target::Mget => return mget(node, req, authn, perm).await,
        Target::Msearch => return msearch(node, req, authn, perm).await,
        Target::Reindex => reindex(&node, &authn, &perm, &mut req)?,
        Target::AliasesBody => aliases_body(&node, &authn, &perm, &mut req)?,
        Target::DataStreamModifyBody => names = data_stream_modify_body(&node, &authn, &perm, &mut req, &act.action)?,
    }
    // Document- and field-level security (superusers are never restricted).
    if !perm.is_superuser {
        access::check_gate(&node, &authn, &perm, &act.action, &names)?;
        req.access = Some(Arc::new(ReadAccess::build(&node, &authn, &perm, &act.action, &names)?));
    }
    let access = req.access.clone();
    let resp = crate::dispatch(node, req).await?;
    filter_mapping_resp(&act.action, access.as_deref(), resp)
}

/// Field-level security for mapping-bearing responses (`GET /{idx}`, `_mapping`, `_mapping/field`).
fn filter_mapping_resp(action: &str, access: Option<&ReadAccess>, resp: Resp) -> R {
    let Some(a) = access else { return Ok(resp) };
    if !matches!(action, "indices:admin/mappings/get" | "indices:admin/mappings/fields/get" | "indices:admin/get") {
        return Ok(resp);
    }
    let fls = a.fls_by_index();
    if fls.is_empty() {
        return Ok(resp);
    }
    match resp_json(resp) {
        Ok((owner, status, mut v, headers)) => {
            access::filter_mapping_response(&mut v, &fls);
            Ok(Resp { status, body: RespBody::retained_json(v, owner), headers })
        }
        // A body we cannot parse could leak hidden fields: refuse it.
        Err(_) => Err(EsError::new(403, "security_exception", "unable to apply field level security to the response")),
    }
}

/// Http request extension: the writing user's authorization of dynamic
/// mapping updates (`indices:admin/mapping/auto_put` on the concrete index),
/// entered around document writes by the handlers.
#[derive(Clone)]
pub struct MappingAuthz(pub elasticsearch_engine::index::MappingUpdateAuthz);

fn mapping_authz(node: &Arc<Node>, a: &Authn, perm: &Permission) -> MappingAuthz {
    let (node, a, perm) = (node.clone(), a.clone(), perm.clone());
    MappingAuthz(Arc::new(move |index: &str| {
        const ACTION: &str = "indices:admin/mapping/auto_put";
        // Backing indices inherit their data stream's privileges; an index
        // written through an alias is authorized through that alias.
        if allowed(&node, &perm, ACTION, index) || node.get_index(index).is_some_and(|svc| svc.aliases().keys().any(|al| perm.check_index(ACTION, al))) {
            Ok(())
        } else {
            Err(denied(&node, &a, &perm, ACTION, Some(&[index.to_string()])))
        }
    }))
}

/// Http request extension: authorizes the destination an ingest pipeline
/// reroutes a document to (`reroute` processor, or `_index` set by any
/// processor) as the writing user's bulk item on that index (`create`: an
/// `op_type=create` write), as ES re-resolves a rerouted request and
/// authorizes it through the shard-level bulk request.
#[derive(Clone)]
pub struct IngestDestAuthz(pub Arc<dyn Fn(&str, bool) -> Result<(), EsError> + Send + Sync>);

fn ingest_dest_authz(node: &Arc<Node>, a: &Authn, perm: &Permission, catalog: Option<Arc<std::collections::BTreeSet<String>>>) -> IngestDestAuthz {
    let (node, a, perm) = (node.clone(), a.clone(), perm.clone());
    IngestDestAuthz(Arc::new(move |dest: &str, create: bool| {
        let ix = resolve_date_math(dest).unwrap_or_else(|_| dest.to_string());
        let names = vec![ix.clone()];
        let item = if create { "indices:data/write/index:op_type/create" } else { "indices:data/write/index:op_type/index" };
        if !allowed(&node, &perm, "indices:data/write/bulk[s]", &ix) {
            return Err(denied(&node, &a, &perm, "indices:data/write/bulk[s]", Some(&names)));
        }
        if !allowed(&node, &perm, item, &ix) || unauthorized_write_alias(&node, &perm, item, &ix).is_some() {
            return Err(denied(&node, &a, &perm, item, Some(&names)));
        }
        let present = match &catalog {
            Some(c) => c.contains(&ix),
            None => node.get_index(&ix).is_some() || node.data_streams.read().contains_key(&ix) || node.all_indices().iter().any(|s| s.aliases().contains_key(&ix)),
        };
        if !present && !perm.check_index("indices:admin/auto_create", &ix) {
            return Err(denied(&node, &a, &perm, "indices:admin/auto_create", Some(&names)));
        }
        Ok(())
    }))
}

/// For an ES|QL or SQL query's action: the action ES runs on the indices
/// the query reads (ES|QL `ComputeService`'s data node request, SQL's
/// search) and whether the query's index resolution is strict (ES|QL).
fn query_read_action(action: &str) -> Option<(&'static str, bool)> {
    if action == "indices:data/read/esql" {
        Some(("indices:data/read/esql/data", true))
    } else if action == "indices:data/read/sql" || action.starts_with("indices:data/read/sql/") {
        Some(("indices:data/read/search", false))
    } else {
        None
    }
}

const FIELD_CAPS: &str = "indices:data/read/field_caps";

/// One of the caller's roles grants indices of a remote cluster.
fn has_remote_indices(node: &Node, authn: &Authn) -> bool {
    let sec = &node.security;
    sec.assigned_roles(authn.effective()).iter().any(|name| sec.role_descriptor(name).is_some_and(|d| d["remote_indices"].as_array().is_some_and(|a| !a.is_empty())))
}

/// The caller's authorization of the index expressions an ES|QL or SQL
/// query resolves, as ES resolves them under the caller's privileges.
/// ES|QL (`strict`): wildcards expand to the indices the caller may read,
/// an explicit name the caller may not read is an unknown index. SQL: the
/// field-caps resolution is lenient (wildcards expand to the names the
/// caller may describe, unauthorized explicit names are dropped, leaving an
/// unknown index when none is left), then the indices read need
/// `read_action` (else a `security_exception`).
fn query_index_authz(node: &Arc<Node>, a: &Authn, perm: &Permission, read_action: &'static str, strict: bool) -> crate::QueryIndexAuthz {
    crate::QueryIndexAuthz(Arc::new(QueryAuthz { node: node.clone(), a: a.clone(), perm: perm.clone(), read_action, strict }))
}

struct QueryAuthz {
    node: Arc<Node>,
    a: Authn,
    perm: Permission,
    read_action: &'static str,
    strict: bool,
}

impl elasticsearch_engine::esql::exec::QueryAuthorizer for QueryAuthz {
    fn indices(&self, expr: &str, opts: &ResolveOpts, read: bool) -> Result<Option<elasticsearch_engine::esql::exec::AuthorizedIndices>, EsError> {
        let (node, perm) = (&self.node, &self.perm);
        // Superusers can read anything outside restricted (.security*) indices.
        if perm.is_superuser && !expr.contains(".security") {
            return Ok(None);
        }
        let mut opts = *opts;
        opts.ignore_unavailable = !self.strict;
        let r = resolve(node, perm, if self.strict { self.read_action } else { FIELD_CAPS }, expr, &opts);
        if read && !self.strict {
            let bad: Vec<String> = r.names.iter().filter(|n| !is_remote(n) && !allowed_name(node, perm, self.read_action, n)).cloned().collect();
            if !bad.is_empty() {
                return Err(denied(node, &self.a, perm, self.read_action, Some(&bad)));
            }
        }
        if self.strict && !r.denied.is_empty() {
            // An unauthorized index the authorized names resolve to anyway
            // (an alias and its own index): ES resolves it through the
            // alias, then refuses the explicit name's read (403).
            let reached: std::collections::HashSet<String> = r.names.iter().filter(|n| !is_remote(n)).flat_map(|n| access::concretes(node, n)).collect();
            let explicit: Vec<String> = r.denied.iter().filter(|d| access::concretes(node, d).iter().any(|c| reached.contains(c))).cloned().collect();
            if !explicit.is_empty() {
                return Err(denied(node, &self.a, perm, "indices:data/read/esql", Some(&explicit)));
            }
        }
        Ok(Some(elasticsearch_engine::esql::exec::AuthorizedIndices { names: r.names, unknown: r.denied }))
    }

    fn enrich(&self, _policy: &str) -> Result<(), EsError> {
        if self.perm.has_cluster_privilege("monitor_enrich") {
            return Ok(());
        }
        Err(EsError::new(403, "security_exception", format!("user [{}] doesn't have sufficient privileges to perform enrich lookup: privilege [monitor_enrich] is missing", self.a.run_as.as_deref().unwrap_or(&self.a).username)))
    }
}

/// A remote cluster expression (`cluster:index`), authorized by the remote
/// cluster (as the engine splits them: `name::selector` is local).
fn is_remote(p: &str) -> bool {
    elasticsearch_engine::ccr::remote::is_remote_part(p)
}

/// `name::selector` (index component selector) split into the name and
/// its selector (`None`: no selector).
fn split_selector(p: &str) -> (&str, Option<&str>) {
    match p.split_once("::") {
        Some((name, selector)) => (name, Some(selector)),
        None => (p, None),
    }
}

/// The component an expression or resolved name reads (`::data` is the default).
fn component(selector: Option<&str>) -> &str {
    selector.unwrap_or("data")
}

fn has_wildcard(p: &str) -> bool {
    p == "_all" || p.contains('*')
}

/// Every name the expression could expand to: indices, aliases, data streams.
fn candidates(node: &Node, opts: &ResolveOpts) -> Vec<String> {
    let mut out = Vec::new();
    for svc in node.all_indices() {
        let hidden = svc.name.starts_with('.') || svc.setting("index.hidden").map(|s| s == "true").unwrap_or(false);
        let open = svc.is_open();
        if (open && opts.expand_open) || (!open && opts.expand_closed) {
            out.push((svc.name.clone(), hidden));
        }
        for (a, def) in svc.aliases() {
            let h = a.starts_with('.') || def.get("is_hidden").and_then(Value::as_bool).unwrap_or(false);
            out.push((a, h));
        }
    }
    for (n, ds) in node.data_streams.read().iter() {
        out.push((n.clone(), ds.hidden || n.starts_with('.')));
    }
    let mut names: Vec<String> = out.into_iter().filter(|(_, h)| !*h || opts.expand_hidden).map(|(n, _)| n).collect();
    names.sort();
    names.dedup();
    names
}

/// Data stream names only (for `indices:*/data_stream/*` actions).
fn ds_candidates(node: &Node, opts: &ResolveOpts) -> Vec<String> {
    node.data_streams.read().iter().filter(|(n, ds)| opts.expand_hidden || !(ds.hidden || n.starts_with('.'))).map(|(n, _)| n.clone()).collect()
}

/// `perm` grants `action` on `name`, directly or through the data stream owning it.
pub fn allowed(node: &Node, perm: &Permission, action: &str, name: &str) -> bool {
    allowed_component(node, perm, action, name, false)
}

/// [`allowed`] on `name`'s `::failures` component (`failures`, which needs
/// a failure-store privilege) or on its data.
fn allowed_component(node: &Node, perm: &Permission, action: &str, name: &str, failures: bool) -> bool {
    if perm.check_index_component(action, name, failures) {
        return true;
    }
    if name.starts_with(".ds-") || name.starts_with(".fs-") {
        let dss = node.data_streams.read();
        // ES: privileges on a data stream cover its backing indices, and
        // its failure-store privileges its failure indices.
        if let Some((ds, d)) = dss.iter().find(|(_, d)| d.indices.iter().chain(d.failure_indices.iter()).any(|i| i == name)) {
            return perm.check_index_component(action, ds, d.failure_indices.iter().any(|i| i == name));
        }
    }
    false
}

/// [`allowed`] of a resolved name, by the component its selector reads.
fn allowed_name(node: &Node, perm: &Permission, action: &str, name: &str) -> bool {
    let (base, selector) = split_selector(name);
    allowed_component(node, perm, action, base, component(selector) == "failures")
}

fn hidden_candidates(node: &Node) -> Vec<String> {
    let o = ResolveOpts { expand_hidden: true, ..ResolveOpts::default() };
    candidates(node, &o)
}

pub struct Resolved {
    pub names: Vec<String>,
    pub denied: Vec<String>,
    pub had_wildcard: bool,
    /// The expression's exclusions (`-name`, after the first part). They
    /// remove authorized names here, and stay in a rewritten expression: an
    /// excluded index behind a kept alias (`alias,-index`, as a cluster
    /// shard request scopes an alias to the indices of one shard group) is
    /// only removed once the engine expands the alias.
    pub excluded: Vec<String>,
}

/// Resolve `expr` for `action`: wildcards expand to authorized names,
/// explicit names are checked (dropped with `ignore_unavailable`).
pub fn resolve(node: &Node, perm: &Permission, action: &str, expr: &str, opts: &ResolveOpts) -> Resolved {
    let mut names: Vec<String> = Vec::new();
    let mut denied_names: Vec<String> = Vec::new();
    let mut had_wildcard = false;
    let mut excluded: Vec<String> = Vec::new();
    let mut cands: Option<Vec<String>> = None;
    let parts: Vec<&str> = if expr.is_empty() { vec!["_all"] } else { expr.split(',').map(str::trim).filter(|p| !p.is_empty()).collect() };
    for (i, raw) in parts.iter().enumerate() {
        let (exclude, p) = match raw.strip_prefix('-') {
            Some(r) if i > 0 => (true, r),
            _ => (false, *raw),
        };
        if is_remote(p) {
            // remote cluster expression: not ours to authorize
            names.push(raw.to_string());
            continue;
        }
        let full = resolve_date_math(p).unwrap_or_else(|_| p.to_string());
        // A component selector (`logs::failures`) names its index, alias or
        // data stream: that name is authorized, the selector kept.
        let (p, selector) = split_selector(&full);
        let p = p.to_string();
        let failures = component(selector) == "failures";
        let with_selector = |n: &str| match selector {
            Some(s) => format!("{n}::{s}"),
            None => n.to_string(),
        };
        if exclude {
            excluded.push(raw.to_string());
            names.retain(|n| {
                let (base, sel) = split_selector(n);
                !(glob(&p, base) && component(sel) == component(selector))
            });
            continue;
        }
        if has_wildcard(&p) {
            had_wildcard = true;
            let pat = if p == "_all" { "*".to_string() } else { p.clone() };
            let c = cands.get_or_insert_with(|| {
                if action.contains("/data_stream/") {
                    ds_candidates(node, opts)
                } else if pat.starts_with('.') {
                    // `.foo*` patterns match hidden names too
                    hidden_candidates(node)
                } else {
                    candidates(node, opts)
                }
            });
            for n in c.iter() {
                if glob(&pat, n) && allowed_component(node, perm, action, n, failures) {
                    let n = with_selector(n);
                    if !names.contains(&n) {
                        names.push(n);
                    }
                }
            }
        } else if allowed_component(node, perm, action, &p, failures) {
            let n = with_selector(&p);
            if !names.contains(&n) {
                names.push(n);
            }
        } else if !opts.ignore_unavailable {
            denied_names.push(full.clone());
        }
    }
    Resolved { names, denied: denied_names, had_wildcard, excluded }
}

/// The authorized expression a request is rewritten to: the names, and,
/// after a kept alias or data stream, the exclusions (they apply to what it
/// expands to: `alias,-index` keeps its other indices).
fn rewritten_expr(node: &Node, r: &Resolved) -> String {
    if r.names.is_empty() {
        NO_INDICES.to_string()
    } else if !r.excluded.is_empty() && r.names.iter().any(|n| !node_has_index(node, n)) {
        format!("{},{}", r.names.join(","), r.excluded.join(","))
    } else {
        r.names.join(",")
    }
}

/// `name` is a concrete index (not an alias or data stream) of this node.
fn node_has_index(node: &Node, name: &str) -> bool {
    node.get_index(split_selector(name).0).is_some()
}

fn exists(node: &Node, req: &Req, name: &str) -> bool {
    if let Some(catalog) = req.extensions.get::<crate::AuthorizationIndexCatalog>() {
        return catalog.0.contains(name);
    }
    if node.get_index(name).is_some() || node.data_streams.read().contains_key(name) {
        return true;
    }
    node.all_indices().iter().any(|s| s.aliases().contains_key(name))
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    /// An alias scoped by an exclusion (a cluster shard request through an
    /// alias, `alias,-other`) keeps the exclusion for a non-superuser too:
    /// the engine then resolves only the indices of this shard group.
    #[test]
    fn exclusions_after_an_alias_survive_authorization() {
        use elasticsearch_engine::security::roles::Role;
        let directory = tempfile::tempdir().unwrap();
        let node = Node::open(directory.path()).unwrap();
        for i in ["i1", "i2"] {
            node.create_index(i, &json!({"aliases": {"al": {}}})).unwrap();
        }
        let role = Role::compile(&[&json!({"indices": [{"names": ["*"], "privileges": ["all"]}]})]);
        let perm = Permission { role: Arc::new(role), limit: None, effective_roles: vec!["admin".into()], missing_roles: vec![], is_superuser: false };
        let r = resolve(&node, &perm, "indices:data/read/search", "al,-i2", &ResolveOpts::default());
        assert_eq!(r.names, vec!["al".to_string()]);
        let expr = rewritten_expr(&node, &r);
        assert_eq!(expr, "al,-i2");
        let got: Vec<String> = node.resolve(&expr, ResolveOpts::default()).unwrap().into_iter().map(|t| t.svc.name.clone()).collect();
        assert_eq!(got, vec!["i1".to_string()]);
        // Concrete names only: unchanged (the exclusions already applied).
        let r = resolve(&node, &perm, "indices:data/read/search", "i*,-i2", &ResolveOpts::default());
        assert_eq!(rewritten_expr(&node, &r), "i1");
    }

    #[test]
    fn applied_metadata_controls_existence_during_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let node = Node::open(directory.path()).unwrap();
        node.create_index("stale-local", &json!({})).unwrap();
        let mut request = Req { method: axum::http::Method::POST, path: "/_bulk".into(), segs: vec!["_bulk".into()],
            params: Default::default(), body: Bytes::new(), headers: Default::default(), access: None,
            extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
        assert!(exists(&node, &request, "stale-local"));
        assert!(!exists(&node, &request, "recovering"));
        request.extensions.insert(crate::AuthorizationIndexCatalog(std::collections::BTreeSet::from(["recovering".into()])));
        assert!(exists(&node, &request, "recovering"), "the applied index must not require auto-create privileges during recovery");
        assert!(!exists(&node, &request, "stale-local"), "a deleted index's stale engine shell cannot authorize auto-creation");
    }
}

fn authorize_seg(node: &Node, a: &Authn, perm: &Permission, req: &mut Req, act: &Act, i: usize) -> Result<Vec<String>, EsError> {
    let present = req.segs.get(i).map(|s| !s.starts_with('_') || s == "_all").unwrap_or(false);
    // Only remote clusters (`alias:index`, or a point in time held by remote
    // clusters only): each remote cluster authorizes its part as this user;
    // ES requires no privilege on the local cluster for them.
    let remote_only = if present {
        let parts: Vec<&str> = req.segs[i].split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
        !parts.is_empty() && parts.iter().all(|p| is_remote(p.strip_prefix('-').unwrap_or(p)))
    } else {
        act.action == "indices:data/read/search"
            && req.json().ok().and_then(|b| b.get("pit").and_then(|p| p.get("id")).and_then(|v| v.as_str()).map(elasticsearch_engine::ccr::remote::remote_only_pit)).unwrap_or(false)
    };
    if remote_only {
        return Ok(Vec::new());
    }
    if !perm.check_index_action(&act.action) {
        // ES names the requested indices in the denial.
        let named: Vec<String> = if present && !has_wildcard(&req.segs[i]) { req.segs[i].split(',').map(str::to_string).collect() } else { Vec::new() };
        return Err(denied(node, a, perm, &act.action, if named.is_empty() { None } else { Some(&named) }));
    }
    let expr = if present { req.segs[i].clone() } else { "_all".to_string() };
    // Superusers can do anything outside restricted (.security*) indices.
    if perm.is_superuser && !expr.contains(".security") {
        return Ok(Vec::new());
    }
    let mut opts = ResolveOpts::from_params(&req.params);
    if req.seg(0) == "_cat" && req.seg(1) == "indices" && req.param("expand_wildcards").is_none() {
        // cat indices: strictExpand (open + closed)
        opts.expand_closed = true;
    }
    let r = resolve(node, perm, &act.action, &expr, &opts);
    if !r.denied.is_empty() {
        return Err(denied(node, a, perm, &act.action, Some(&r.denied)));
    }
    // Second action (document write `op_type`) and auto-create on the same names.
    if let Some(item) = &act.item {
        let bad: Vec<String> = r.names.iter().filter(|n| !allowed(node, perm, item, n)).cloned().collect();
        if !bad.is_empty() {
            return Err(denied(node, a, perm, item, Some(&bad)));
        }
    }
    if act.auto_create {
        let bad: Vec<String> = r.names.iter().filter(|n| !exists(node, req, n) && !perm.check_index("indices:admin/auto_create", n)).cloned().collect();
        if !bad.is_empty() {
            return Err(denied(node, a, perm, "indices:admin/auto_create", Some(&bad)));
        }
    }
    if let Some(ai) = act.alias_seg {
        if let Some(aliases) = req.segs.get(ai) {
            let bad: Vec<String> = aliases.split(',').filter(|n| !perm.check_index(&act.action, n)).map(str::to_string).collect();
            if !bad.is_empty() {
                return Err(denied(node, a, perm, &act.action, Some(&bad)));
            }
        }
    }
    let joined = rewritten_expr(node, &r);
    if present {
        if r.had_wildcard || joined != expr {
            req.segs[i] = joined;
        }
    } else if i <= req.segs.len() && !(act.action == "indices:data/read/search" && req.json().map(|b| b.get("pit").is_some()).unwrap_or(false)) {
        // (a point-in-time search names no index: its targets come from the PIT)
        req.segs.insert(i, joined);
    }
    Ok(r.names)
}

// ------------------------------------------------------------------- bulk

fn resp_json(r: Resp) -> Result<(Option<axum::body::Bytes>, u16, Value, Vec<(&'static str, String)>), Resp> {
    match r.body {
        RespBody::Json(v) => Ok((None, r.status, v, r.headers)),
        RespBody::RetainedJson {value,owner} => Ok((Some(owner),r.status,value,r.headers)),
        RespBody::Raw(s) => match serde_json::from_str(&s) {
            Ok(v) => Ok((None, r.status, v, r.headers)),
            Err(_) => Err(Resp { status: r.status, body: RespBody::Raw(s), headers: r.headers }),
        },
        RespBody::Owned {bytes,json:true} => match serde_json::from_slice(&bytes) {
            Ok(v)=>Ok((Some(bytes),r.status,v,r.headers)),
            Err(_)=>Err(Resp{status:r.status,body:RespBody::Owned{bytes,json:true},headers:r.headers}),
        },
        other => Err(Resp { status: r.status, body: other, headers: r.headers }),
    }
}

fn item_error(e: &EsError) -> Value {
    json!({"type": e.kind, "reason": e.reason})
}

/// A bulk item addressed to an alias resolves to the alias's write index; ES
/// then also requires the privilege on every other alias for which that
/// concrete index is the write index. Returns the first alias lacking it.
fn unauthorized_write_alias(node: &Node, perm: &Permission, action: &str, name: &str) -> Option<String> {
    let all = node.all_indices();
    if all.iter().any(|s| s.name == name) {
        return None;
    }
    let is_write = |def: &Value, n_indices: usize| def.get("is_write_index").and_then(Value::as_bool).unwrap_or(n_indices == 1);
    let holders: Vec<_> = all.iter().filter(|s| s.aliases().contains_key(name)).collect();
    // A data stream alias writes to its write data stream's current write
    // index, not to the first backing index carrying the flag.
    let stream_write = node.alias_write_data_stream(name).and_then(|d| all.iter().find(|s| s.name == d.write_index()));
    let write_idx = match stream_write {
        Some(w) => w,
        None => *(if holders.len() == 1 { holders.first() } else { holders.iter().find(|s| s.aliases().get(name).map(|d| is_write(d, holders.len())).unwrap_or(false)) })?,
    };
    if perm.check_index(action, &write_idx.name) {
        return None;
    }
    for (alias, def) in write_idx.aliases() {
        let n = all.iter().filter(|s| s.aliases().contains_key(&alias)).count();
        let flagged = def.get("is_write_index").and_then(Value::as_bool);
        let writes_here = match flagged {
            Some(b) => b,
            None => n == 1,
        };
        if writes_here && !perm.check_index(action, &alias) {
            return Some(alias);
        }
    }
    None
}

/// The per-item authorization of a bulk item (`op` with its action-line
/// parameters `meta`): the failure (`index`, `_id`, error) it answers with,
/// or `None` when it may run.
fn authorize_bulk_item(node: &Node, req: &Req, a: &Authn, perm: &Permission, op: &str, meta: &Map<String, Value>, default_index: &Option<String>) -> Option<(String, Value, EsError)> {
    let index = meta.get("_index").and_then(Value::as_str).map(str::to_string).or_else(|| default_index.clone());
    let id = meta.get("_id").cloned().unwrap_or(Value::Null);
    let ix = index?;
    let ix = resolve_date_math(&ix).unwrap_or_else(|_| ix.clone());
    // An `index` without `_id` gets an auto-generated id and runs as a
    // create, except on time series indices (the id derives from the
    // dimensions, so it stays an `op_type=index` that may overwrite).
    let tsdb = || node.write_index(&ix, false).map(|s| s.setting("index.mode").as_deref() == Some("time_series")).unwrap_or(false);
    let op_create = op == "create" || (op == "index" && (meta.get("op_type").and_then(Value::as_str) == Some("create") || (id.is_null() && !tsdb())));
    let item_action = match op {
        "update" => "indices:data/write/update",
        "delete" => "indices:data/write/delete",
        _ if op_create => "indices:data/write/index:op_type/create",
        _ => "indices:data/write/index:op_type/index",
    };
    let names = vec![ix.clone()];
    // The cluster front end hands data stream writes over with `_index`
    // already resolved to the write backing index (`.ds-...`): it is
    // authorized through its data stream, as ES does.
    let err = if !allowed(node, perm, "indices:data/write/bulk[s]", &ix) {
        Some(denied(node, a, perm, "indices:data/write/bulk[s]", Some(&names)))
    } else if !allowed(node, perm, item_action, &ix) {
        Some(denied(node, a, perm, item_action, Some(&names)))
    } else if unauthorized_write_alias(node, perm, item_action, &ix).is_some() {
        Some(denied(node, a, perm, item_action, Some(&names)))
    } else if op == "update" && access::is_restricted(node, a, perm, item_action, &names) {
        // Document/field level security: an update would read and rewrite hidden data.
        Some(access::deny_bulk_update())
    } else if op != "delete" && !exists(node, req, &ix) && !perm.check_index("indices:admin/auto_create", &ix) {
        Some(denied(node, a, perm, "indices:admin/auto_create", Some(&names)))
    } else {
        None
    };
    err.map(|e| (ix, id, e))
}

async fn bulk(node: Arc<Node>, mut req: Req, a: Authn, perm: Permission) -> R {
    if !perm.check_index_action("indices:data/write/bulk") {
        return Err(denied(&node, &a, &perm, "indices:data/write/bulk", None));
    }
    let default_index = if !req.seg(0).starts_with('_') { Some(req.seg(0).to_string()) } else { None };
    // (item position, op, index, id, error)
    let mut failures: Vec<(usize, String, String, Value, EsError)> = Vec::new();
    let mut item = 0usize;
    // Items the cluster coordinator handed over pre-parsed (empty body):
    // authorized from their actions; the denied ones are dropped from the
    // items the engine takes.
    if let Some(prepared) = req.extensions.get::<crate::PreparedBulk>().cloned() {
        if let Some(items) = prepared.take() {
            let mentions_security = |it: &crate::PreparedBulkItem| it.source.as_deref().is_some_and(|s| s.contains(".security"))
                || it.meta.values().any(|v| v.as_str().is_some_and(|s| s.contains(".security")))
                || default_index.as_deref().is_some_and(|i| i.contains(".security"));
            let mut kept = Vec::with_capacity(items.len());
            if perm.is_superuser && !items.iter().any(mentions_security) {
                kept = items;
                item = kept.len();
            } else {
                for it in items {
                    match authorize_bulk_item(&node, &req, &a, &perm, &it.action, &it.meta, &default_index) {
                        Some((ix, id, e)) => failures.push((item, it.action.clone(), ix, id, e)),
                        None => kept.push(it),
                    }
                    item += 1;
                }
            }
            if !kept.is_empty() || failures.is_empty() {
                req.extensions.insert(crate::PreparedBulk::new(kept, prepared.bytes));
            }
        }
    } else {
        if perm.is_superuser && !req.body.windows(9).any(|w| w == b".security") {
            return crate::dispatch(node, req).await;
        }
        let text = req.body_str()?.to_string();
        let lines: Vec<&str> = text.split('\n').collect();
        let mut kept = String::with_capacity(text.len());
        let mut i = 0usize;
        while i < lines.len() {
            let line = lines[i];
            if line.trim().is_empty() {
                i += 1;
                continue;
            }
            let parsed: Option<(String, Map<String, Value>)> = serde_json::from_str::<Value>(line).ok().and_then(|v| v.as_object().and_then(|o| o.iter().next().map(|(k, m)| (k.clone(), m.as_object().cloned().unwrap_or_default()))));
            let Some((op, meta)) = parsed else {
                // let the engine report the malformed line
                kept.push_str(line);
                kept.push('\n');
                i += 1;
                continue;
            };
            let has_source = op != "delete";
            let src = if has_source { lines.get(i + 1).copied() } else { None };
            if let Some((ix, id, e)) = authorize_bulk_item(&node, &req, &a, &perm, &op, &meta, &default_index) {
                failures.push((item, op.clone(), ix, id, e));
                item += 1;
                i += if has_source { 2 } else { 1 };
                continue;
            }
            kept.push_str(line);
            kept.push('\n');
            if let Some(s) = src {
                kept.push_str(s);
                kept.push('\n');
            }
            item += 1;
            i += if has_source { 2 } else { 1 };
        }
        if !failures.is_empty() && item != failures.len() {
            req.body = Bytes::from(kept);
        }
    }
    if failures.is_empty() {
        return crate::dispatch(node, req).await;
    }
    let fail_json = |(_, op, ix, id, e): &(usize, String, String, Value, EsError)| -> Value {
        let mut m = Map::new();
        m.insert("_index".into(), json!(ix));
        m.insert("_id".into(), id.clone());
        m.insert("status".into(), json!(e.status));
        m.insert("error".into(), item_error(e));
        json!({ op.clone(): m })
    };
    if item == failures.len() {
        let items: Vec<Value> = failures.iter().map(fail_json).collect();
        return Ok(Resp::ok(json!({"errors": true, "took": 0, "items": items})));
    }
    let r = crate::dispatch(node, req).await?;
    let (owner, status, mut v, headers) = match resp_json(r) {
        Ok(x) => x,
        Err(r) => return Ok(r),
    };
    if let Some(arr) = v.get_mut("items").and_then(Value::as_array_mut) {
        for f in &failures {
            let pos = f.0.min(arr.len());
            arr.insert(pos, fail_json(f));
        }
        v["errors"] = json!(true);
    }
    Ok(Resp { status, body: RespBody::retained_json(v, owner), headers })
}

// ------------------------------------------------------------------- mget

async fn mget(node: Arc<Node>, mut req: Req, a: Authn, perm: Permission) -> R {
    if !perm.check_index_action("indices:data/read/mget") {
        return Err(denied(&node, &a, &perm, "indices:data/read/mget", None));
    }
    if perm.is_superuser {
        return crate::dispatch(node, req).await;
    }
    let default_index = if !req.seg(0).starts_with('_') { Some(req.seg(0).to_string()) } else { None };
    let mut body = req.json()?;
    let action = "indices:data/read/mget[shard]";
    let mut failures: Vec<(usize, Value)> = Vec::new();
    let mut total = 0usize;
    if let Some(docs) = body.get_mut("docs").and_then(Value::as_array_mut) {
        let mut keep = Vec::new();
        for (pos, d) in docs.drain(..).enumerate() {
            total += 1;
            let ix = d.get("_index").and_then(index_text).or_else(|| default_index.clone());
            match ix {
                Some(ix) if !perm.check_index(action, &ix) => {
                    let e = denied(&node, &a, &perm, action, Some(&[ix.clone()]));
                    failures.push((pos, json!({"_index": ix, "_id": d.get("_id").cloned().unwrap_or(Value::Null), "error": item_error(&e)})));
                }
                _ => keep.push(d),
            }
        }
        *docs = keep;
    } else if let (Some(ids), Some(ix)) = (body.get("ids").and_then(Value::as_array).cloned(), default_index.clone()) {
        total = ids.len();
        if !perm.check_index(action, &ix) {
            let e = denied(&node, &a, &perm, action, Some(&[ix.clone()]));
            let docs: Vec<Value> = ids.iter().map(|id| json!({"_index": ix, "_id": id, "error": item_error(&e)})).collect();
            return Ok(Resp::ok(json!({"docs": docs})));
        }
    }
    // DLS/FLS over every index the (remaining) documents read.
    let mut read_names: Vec<String> = Vec::new();
    match body.get("docs").and_then(Value::as_array) {
        Some(docs) => {
            for d in docs {
                if let Some(ix) = d.get("_index").and_then(index_text).or_else(|| default_index.clone()) {
                    if !read_names.contains(&ix) {
                        read_names.push(ix);
                    }
                }
            }
        }
        None => read_names.extend(default_index.clone()),
    }
    req.access = Some(Arc::new(ReadAccess::build(&node, &a, &perm, "indices:data/read/get", &read_names)?));
    if failures.is_empty() {
        return crate::dispatch(node, req).await;
    }
    if failures.len() == total {
        return Ok(Resp::ok(json!({"docs": failures.into_iter().map(|(_, v)| v).collect::<Vec<_>>()})));
    }
    req.body = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    let r = crate::dispatch(node, req).await?;
    let (owner, status, mut v, headers) = match resp_json(r) {
        Ok(x) => x,
        Err(r) => return Ok(r),
    };
    if let Some(arr) = v.get_mut("docs").and_then(Value::as_array_mut) {
        for (pos, f) in failures {
            let p = pos.min(arr.len());
            arr.insert(p, f);
        }
    }
    Ok(Resp { status, body: RespBody::retained_json(v, owner), headers })
}

/// A document's `_index` as the engine reads it (any scalar's text; `None` for `null`).
fn index_text(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        x => Some(x.to_string()),
    }
}

// ---------------------------------------------------------------- msearch

async fn msearch(node: Arc<Node>, mut req: Req, a: Authn, perm: Permission) -> R {
    let top = if req.seg(1) == "template" || req.seg(2) == "template" { "indices:data/read/msearch/template" } else { "indices:data/read/msearch" };
    if !perm.check_index_action(top) {
        return Err(denied(&node, &a, &perm, top, None));
    }
    if perm.is_superuser {
        return crate::dispatch(node, req).await;
    }
    let default_index = if !req.seg(0).starts_with('_') { Some(req.seg(0).to_string()) } else { None };
    let text = req.body_str()?.to_string();
    let lines: Vec<&str> = text.split('\n').filter(|l| !l.trim().is_empty()).collect();
    let mut out = String::with_capacity(text.len());
    let mut failures: Vec<(usize, Value)> = Vec::new();
    let mut n = 0usize;
    let mut i = 0usize;
    let action = "indices:data/read/search";
    // Every sub-search's restrictions hold for the whole request (strictest wins).
    let mut merged: Option<ReadAccess> = None;
    while i < lines.len() {
        let mut header: Value = serde_json::from_str(lines[i]).unwrap_or_else(|_| json!({}));
        let body = lines.get(i + 1).copied().unwrap_or("{}");
        i += 2;
        let expr = match header.get("index") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
            _ => default_index.clone().unwrap_or_else(|| "_all".into()),
        };
        let mut params = req.params.clone();
        for k in ["ignore_unavailable", "allow_no_indices", "expand_wildcards"] {
            if let Some(v) = header.get(k) {
                params.insert(k.to_string(), match v {
                    Value::String(s) => s.clone(),
                    Value::Array(a) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
                    other => other.to_string(),
                });
            }
        }
        let opts = ResolveOpts::from_params(&params);
        let r = resolve(&node, &perm, action, &expr, &opts);
        if !r.denied.is_empty() {
            let e = denied(&node, &a, &perm, action, Some(&r.denied));
            failures.push((n, json!({"error": e.error_object(false), "status": e.status})));
        } else {
            let acc = ReadAccess::build(&node, &a, &perm, action, &r.names)?;
            match merged.as_mut() {
                Some(m) => m.merge(acc),
                None => merged = Some(acc),
            }
            let names = if r.names.is_empty() { vec!["*".to_string(), "-*".to_string()] } else { r.names };
            header["index"] = json!(names);
            out.push_str(&header.to_string());
            out.push('\n');
            out.push_str(body);
            out.push('\n');
        }
        n += 1;
    }
    if failures.len() == n && n > 0 {
        return Ok(Resp::ok(json!({"took": 0, "responses": failures.into_iter().map(|(_, v)| v).collect::<Vec<_>>()})));
    }
    req.body = Bytes::from(out);
    req.access = Some(Arc::new(match merged {
        Some(m) => m,
        None => ReadAccess::build(&node, &a, &perm, action, &[])?,
    }));
    let r = crate::dispatch(node, req).await?;
    if failures.is_empty() {
        return Ok(r);
    }
    let (owner, status, mut v, headers) = match resp_json(r) {
        Ok(x) => x,
        Err(r) => return Ok(r),
    };
    if let Some(arr) = v.get_mut("responses").and_then(Value::as_array_mut) {
        for (pos, f) in failures {
            let p = pos.min(arr.len());
            arr.insert(p, f);
        }
    }
    Ok(Resp { status, body: RespBody::retained_json(v, owner), headers })
}

// ---------------------------------------------------------------- reindex

fn reindex(node: &Node, a: &Authn, perm: &Permission, req: &mut Req) -> Result<(), EsError> {
    if !perm.check_index_action("indices:data/write/reindex") {
        return Err(denied(node, a, perm, "indices:data/write/reindex", None));
    }
    if perm.is_superuser {
        return Ok(());
    }
    let mut body = req.json()?;
    let src = match &body["source"]["index"] {
        Value::String(s) => s.clone(),
        Value::Array(v) => v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
        _ => return Ok(()),
    };
    let r = resolve(node, perm, "indices:data/read/search", &src, &ResolveOpts::default());
    if !r.denied.is_empty() {
        return Err(denied(node, a, perm, "indices:data/read/search", Some(&r.denied)));
    }
    access::check_gate_with(node, a, perm, "indices:data/read/search", &r.names, access::deny_reindex)?;
    body["source"]["index"] = json!(if r.names.is_empty() { vec!["*".to_string(), "-*".to_string()] } else { r.names });
    if let Some(dest) = body["dest"]["index"].as_str().map(str::to_string) {
        let names = vec![dest.clone()];
        let create = body["dest"]["op_type"].as_str() == Some("create");
        let item = if create { "indices:data/write/index:op_type/create" } else { "indices:data/write/index:op_type/index" };
        for act in ["indices:data/write/bulk[s]", item] {
            if !perm.check_index(act, &dest) {
                return Err(denied(node, a, perm, act, Some(&names)));
            }
        }
        if !exists(node, req, &dest) && !perm.check_index("indices:admin/auto_create", &dest) {
            return Err(denied(node, a, perm, "indices:admin/auto_create", Some(&names)));
        }
    }
    req.body = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    Ok(())
}

// ---------------------------------------------------------------- aliases

/// `POST /_data_stream/_modify`: the action on every data stream and index
/// the body names (a backing index through its stream, a failure index
/// through its stream's failure store), all checked before anything
/// changes: a grant scoped to one data stream never modifies another.
fn data_stream_modify_body(node: &Node, a: &Authn, perm: &Permission, req: &mut Req, action: &str) -> Result<Vec<String>, EsError> {
    if !perm.check_index_action(action) {
        return Err(denied(node, a, perm, action, None));
    }
    let body = req.json()?;
    let mut names: Vec<String> = Vec::new();
    for act in body["actions"].as_array().into_iter().flatten() {
        let Some((_, spec)) = act.as_object().and_then(|o| o.iter().next()) else { continue };
        let failures = spec["failure_store"].as_bool() == Some(true);
        if let Some(ds) = spec["data_stream"].as_str() {
            names.push(ds.to_string());
            if !perm.is_superuser && !allowed_component(node, perm, action, ds, failures) {
                return Err(denied(node, a, perm, action, Some(&[ds.to_string()])));
            }
        }
        if let Some(ix) = spec["index"].as_str() {
            names.push(ix.to_string());
            if !perm.is_superuser && !allowed(node, perm, action, ix) {
                return Err(denied(node, a, perm, action, Some(&[ix.to_string()])));
            }
        }
    }
    Ok(names)
}

fn aliases_body(node: &Node, a: &Authn, perm: &Permission, req: &mut Req) -> Result<(), EsError> {
    let action = "indices:admin/aliases";
    if !perm.check_index_action(action) {
        return Err(denied(node, a, perm, action, None));
    }
    if perm.is_superuser {
        return Ok(());
    }
    let body = req.json()?;
    let mut bad: Vec<String> = Vec::new();
    for act in body["actions"].as_array().into_iter().flatten() {
        let Some((kind, spec)) = act.as_object().and_then(|o| o.iter().next()) else { continue };
        let names = |k1: &str, k2: &str| -> Vec<String> {
            let mut v = Vec::new();
            for k in [k1, k2] {
                match &spec[k] {
                    Value::String(s) => v.push(s.clone()),
                    Value::Array(a) => v.extend(a.iter().filter_map(Value::as_str).map(str::to_string)),
                    _ => {}
                }
            }
            v
        };
        let need = if kind == "remove_index" { "indices:admin/delete" } else { action };
        for ix in names("index", "indices") {
            if has_wildcard(&ix) {
                let r = resolve(node, perm, need, &ix, &ResolveOpts::default());
                if r.names.is_empty() {
                    bad.push(ix);
                }
            } else if !perm.check_index(need, &ix) {
                bad.push(ix);
            }
        }
        if kind != "remove_index" {
            for al in names("alias", "aliases") {
                if !has_wildcard(&al) && !perm.check_index(action, &al) {
                    bad.push(al);
                }
            }
        }
    }
    if !bad.is_empty() {
        bad.dedup();
        return Err(denied(node, a, perm, action, Some(&bad)));
    }
    // DLS/FLS: aliases (which could carry a filter around them) on restricted indices are refused.
    let mut targets: Vec<String> = Vec::new();
    for act in body["actions"].as_array().into_iter().flatten() {
        let Some((_, spec)) = act.as_object().and_then(|o| o.iter().next()) else { continue };
        for k in ["index", "indices"] {
            match &spec[k] {
                Value::String(s) => targets.push(s.clone()),
                Value::Array(v) => targets.extend(v.iter().filter_map(Value::as_str).map(str::to_string)),
                _ => {}
            }
        }
    }
    let mut resolved: Vec<String> = Vec::new();
    for t in targets {
        if has_wildcard(&t) {
            resolved.extend(resolve(node, perm, action, &t, &ResolveOpts::default()).names);
        } else {
            resolved.push(t);
        }
    }
    access::check_gate(node, a, perm, action, &resolved)?;
    Ok(())
}

#[cfg(test)]
mod coordinator_owner_tests {
    use super::*;
    #[test]
    fn parsed_security_response_retains_header_owner_until_rebuilt_response_drops() {
        struct Owner { marker: std::sync::Arc<()>, bytes: &'static [u8] }
        impl AsRef<[u8]> for Owner { fn as_ref(&self) -> &[u8] { self.bytes } }
        let marker = std::sync::Arc::new(());
        let weak = std::sync::Arc::downgrade(&marker);
        let bytes = axum::body::Bytes::from_owner(Owner { marker, bytes: br#"{"hits":[]}"# });
        let response = Resp { status: 200, headers: vec![("warning", "retained header".into())], body: RespBody::Owned { bytes, json: true } };
        let (owner, status, mut value, headers) = resp_json(response).ok().unwrap();
        assert!(weak.upgrade().is_some());
        value["filtered"] = serde_json::json!(true);
        let response = Resp { status, headers, body: RespBody::retained_json(value, owner) };
        assert!(weak.upgrade().is_some());
        // A second FLS/response transformation must preserve the same owner.
        let (owner, status, value, headers) = resp_json(response).ok().unwrap();
        let response = Resp { status, headers, body: RespBody::retained_json(value, owner) };
        assert!(weak.upgrade().is_some());
        drop(response);
        assert!(weak.upgrade().is_none());
    }
}
