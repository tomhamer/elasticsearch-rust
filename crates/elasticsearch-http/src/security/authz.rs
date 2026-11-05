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
            if !perm.check_index_action(&act.action) {
                return Err(denied(&node, &authn, &perm, &act.action, None));
            }
            // ES|QL, SQL, analyze...: names live in the body, so reads cannot be filtered.
            access::check_unscoped_read(&perm, &act.action)?;
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
    if perm.check_index(action, name) {
        return true;
    }
    if name.starts_with(".ds-") || name.starts_with(".fs-") {
        let dss = node.data_streams.read();
        // ES: privileges on a data stream cover its backing and failure-store indices.
        if let Some((ds, _)) = dss.iter().find(|(_, d)| d.indices.iter().chain(d.failure_indices.iter()).any(|i| i == name)) {
            return perm.check_index(action, ds);
        }
    }
    false
}

fn hidden_candidates(node: &Node) -> Vec<String> {
    let o = ResolveOpts { expand_hidden: true, ..ResolveOpts::default() };
    candidates(node, &o)
}

pub struct Resolved {
    pub names: Vec<String>,
    pub denied: Vec<String>,
    pub had_wildcard: bool,
}

/// Resolve `expr` for `action`: wildcards expand to authorized names,
/// explicit names are checked (dropped with `ignore_unavailable`).
pub fn resolve(node: &Node, perm: &Permission, action: &str, expr: &str, opts: &ResolveOpts) -> Resolved {
    let mut names: Vec<String> = Vec::new();
    let mut denied_names: Vec<String> = Vec::new();
    let mut had_wildcard = false;
    let mut cands: Option<Vec<String>> = None;
    let parts: Vec<&str> = if expr.is_empty() { vec!["_all"] } else { expr.split(',').map(str::trim).filter(|p| !p.is_empty()).collect() };
    for (i, raw) in parts.iter().enumerate() {
        let (exclude, p) = match raw.strip_prefix('-') {
            Some(r) if i > 0 => (true, r),
            _ => (false, *raw),
        };
        if p.contains(':') && !p.starts_with('<') {
            // remote cluster expression: not ours to authorize
            names.push(raw.to_string());
            continue;
        }
        let p = resolve_date_math(p).unwrap_or_else(|_| p.to_string());
        if exclude {
            names.retain(|n| !glob(&p, n));
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
                if glob(&pat, n) && allowed(node, perm, action, n) && !names.contains(n) {
                    names.push(n.clone());
                }
            }
        } else if allowed(node, perm, action, &p) {
            if !names.contains(&p) {
                names.push(p);
            }
        } else if !opts.ignore_unavailable {
            denied_names.push(p);
        }
    }
    Resolved { names, denied: denied_names, had_wildcard }
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
    let joined = if r.names.is_empty() { NO_INDICES.to_string() } else { r.names.join(",") };
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
    let write_idx = if holders.len() == 1 { holders.first() } else { holders.iter().find(|s| s.aliases().get(name).map(|d| is_write(d, holders.len())).unwrap_or(false)) }?;
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
            let ix = d.get("_index").and_then(Value::as_str).map(str::to_string).or_else(|| default_index.clone());
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
                if let Some(ix) = d.get("_index").and_then(Value::as_str).map(str::to_string).or_else(|| default_index.clone()) {
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
