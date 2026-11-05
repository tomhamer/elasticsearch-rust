//! Index-level APIs answered by the cluster layer because they span nodes:
//! `_field_caps` (per-index fan-out + merge), `_segments`, `_cat/segments`,
//! `_cache/clear`, `_resolve/cluster` and
//! the deprecated `_knn_search`.

use super::cat::{self, cols, Row};
use super::cat_columns as C;
use super::cat_tables::{cat_resolve_opts, resolve};
use super::{gather_shard_stats, LocalShardStats};
use crate::front::{self, fanout, local_engine, RestReq, RestResp};
use crate::node::{ClusterNode, ES_VERSION};
use elasticsearch_cluster::state::{ClusterState, ShardRouting};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::ResolveOpts;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Build hash the engine reports in `/` and in deprecation `Warning` headers.
use elasticsearch_engine::version::{BUILD_HASH, LUCENE_VERSION};

fn es_err(e: &EsError) -> RestResp {
    RestResp::json(e.status, &e.to_json_with_trace(false))
}

/// The index expression of `/{index}/_api` (empty for `/_api`).
fn index_expr(req: &RestReq) -> Option<String> {
    req.segments().first().filter(|s| !s.starts_with('_')).cloned()
}

/// Copies of `index`, primary first, each with the stats its node reported
/// (`None` for unassigned / unreachable copies).
fn copies<'a>(st: &'a ClusterState, stats: &'a BTreeMap<crate::front::SharedText, LocalShardStats>, index: &str) -> Vec<(&'a ShardRouting, Option<&'a LocalShardStats>)> {
    let mut v: Vec<&ShardRouting> = st.routing_table.all_shards().filter(|s| s.index == index).collect();
    v.sort_by_key(|s| (s.shard, !s.primary));
    v.into_iter().map(|s| (s, s.allocation_id().and_then(|a| stats.get(a)))).collect()
}

fn active_ids(st: &ClusterState, index: &str) -> BTreeSet<String> {
    st.routing_table.all_shards().filter(|c| c.index == index && c.active()).filter_map(|c| c.allocation_id().map(str::to_string)).collect()
}

// ------------------------------------------------------------- field caps

/// Run `/{index}/{api}` once per concrete index when no single node holds a
/// copy of every target. `Err` is the finished response: the plain read
/// (zero/one index, a common holder, or a resolution error the engine
/// renders) or the first failing per-index response.
async fn per_index(node: &Arc<ClusterNode>, req: &RestReq, api: &str) -> Result<Vec<(String, Value)>, RestResp> {
    let targets = index_expr(req).unwrap_or_default();
    let expr = if targets.is_empty() { "_all".to_string() } else { targets.clone() };
    let opts = ResolveOpts::from_params(&req.params());
    let engine = node.engine.clone();
    let names: Vec<String> = match tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|t| t.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<String>>())).await {
        Ok(Ok(n)) => n.into_iter().collect(),
        _ => return Err(front::read_route(node, req, &targets).await),
    };
    if names.len() < 2 {
        return Err(front::read_route(node, req, &targets).await);
    }
    // One node holding a copy of everything answers with the engine's merge.
    let st = node.state();
    let mut common: Option<BTreeSet<String>> = None;
    for n in &names {
        let holders: BTreeSet<String> = st.routing_table.shard_table(n, 0).map(|t| t.searchable_copies(&st.nodes).filter_map(|c| c.node_id.clone()).collect()).unwrap_or_default();
        common = Some(match common {
            None => holders,
            Some(c) => c.intersection(&holders).cloned().collect(),
        });
    }
    if common.map(|c| !c.is_empty()).unwrap_or(true) {
        return Err(front::read_route(node, req, &targets).await);
    }
    let query = req.query();
    let subs = names
        .iter()
        .map(|n| {
            let mut r = req.clone();
            r.set_uri_fmt(format_args!("/{n}/{api}{}{query}",if query.is_empty(){""}else{"?"}))?;
            Ok(r)
        })
        .collect::<elasticsearch_engine::error::Result<Vec<_>>>().map_err(|e|es_err(&e))?;
    let resps = futures::future::join_all(subs.iter().zip(&names).map(|(r, n)| front::read_route(node, r, n))).await;
    let mut parts: Vec<(String, Value)> = Vec::with_capacity(names.len());
    for (n, r) in names.iter().zip(resps) {
        if !r.is_success() {
            return Err(r);
        }
        parts.push((n.clone(), r.body_json().unwrap_or(Value::Null)));
    }
    Ok(parts)
}

pub async fn field_caps(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let parts = match per_index(node, req, "_field_caps").await {
        Ok(p) => p,
        Err(r) => return r,
    };
    let include_unmapped = req.params().get("include_unmapped").map(|v| v == "true" || v.is_empty()).unwrap_or(false);
    RestResp::json(200, &fanout::merge_field_caps(&parts, include_unmapped))
}

/// `_shard_stores`: the per-index `indices` entries are disjoint, so the
/// cross-node answer is their union.
pub async fn shard_stores(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let parts = match per_index(node, req, "_shard_stores").await {
        Ok(p) => p,
        Err(r) => return r,
    };
    let mut indices = BTreeMap::new();
    for (_, body) in parts {
        if let Some(Value::Object(m)) = body.get("indices") {
            indices.extend(m.clone());
        }
    }
    RestResp::json(200, &json!({"indices": indices}))
}

// --------------------------------------------------------------- segments

fn resolve_from_params(node: &ClusterNode, req: &RestReq) -> Result<Vec<String>, RestResp> {
    let expr = index_expr(req);
    resolve(node, expr.as_deref(), ResolveOpts::from_params(&req.params()))
}

pub async fn segments(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let names = match resolve_from_params(node, req) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let st = node.state();
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let (mut total, mut successful) = (0u64, 0u64);
    let mut indices = Map::new();
    for name in names.iter().collect::<BTreeSet<_>>() {
        let active = active_ids(&st, name);
        let mut shards: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for (c, s) in copies(&st, &stats, name) {
            total += 1;
            let (Some(s), true) = (s, c.allocation_id().map(|a| active.contains(a)).unwrap_or(false)) else { continue };
            successful += 1;
            let mut segs = Map::new();
            let details = s.segment_details();
            for entry in &details {
                let i = entry.id;
                // ES `num_docs`: live documents (maxDoc minus deletions).
                let docs = entry.docs.saturating_sub(entry.deleted);
                segs.insert(
                    format!("_{i}"),
                    json!({"generation": i, "num_docs": docs, "deleted_docs": entry.deleted, "size_in_bytes": entry.bytes, "committed": entry.committed, "search": entry.search, "version": LUCENE_VERSION, "compound": false, "attributes": {}}),
                );
            }
            shards.entry(c.shard.to_string()).or_default().push(json!({
                "routing": {"state": "STARTED", "primary": c.primary, "node": c.node_id},
                "num_committed_segments": details.iter().filter(|s| s.committed).count(),
                "num_search_segments": details.iter().filter(|s| s.search).count(),
                "segments": segs,
            }));
        }
        indices.insert(name.clone(), json!({"shards": shards}));
    }
    RestResp::json(200, &json!({"_shards": {"total": total, "successful": successful, "failed": 0}, "indices": indices}))
}

pub async fn cat_segments(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let opts = match cat_resolve_opts(req, false) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let expr = req.segments().get(2).cloned();
    let mut names = match resolve(node, expr.as_deref(), opts) {
        Ok(n) => n,
        Err(r) => return r,
    };
    names.sort();
    names.dedup();
    let st = node.state();
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let unit = req.params().get("bytes").cloned();
    let mut rows: Vec<Row> = Vec::new();
    for name in &names {
        let active = active_ids(&st, name);
        for (c, s) in copies(&st, &stats, name) {
            let (Some(s), true) = (s, c.allocation_id().map(|a| active.contains(a)).unwrap_or(false)) else { continue };
            let n = c.node_id.as_ref().and_then(|id| st.nodes.get(id));
            let details = s.segment_details();
            for entry in &details {
                let i = entry.id;
                let docs = entry.docs.saturating_sub(entry.deleted);
                rows.push(vec![
                    ("index", Some(name.clone())),
                    ("shard", Some(c.shard.to_string())),
                    ("prirep", Some(if c.primary { "p" } else { "r" }.to_string())),
                    ("ip", n.map(|n| n.ip.clone())),
                    ("id", c.node_id.clone()),
                    ("segment", Some(format!("_{i}"))),
                    ("generation", Some(i.to_string())),
                    ("docs.count", Some(docs.to_string())),
                    ("docs.deleted", Some(entry.deleted.to_string())),
                    // ES reports `size.memory` as 0 since 8.0.
                    ("size", Some(cat::bytes(entry.bytes, unit.as_deref()))),
                    ("size.memory", Some("0".into())),
                    ("committed", Some(entry.committed.to_string())),
                    ("searchable", Some(entry.search.to_string())),
                    ("version", Some(LUCENE_VERSION.into())),
                    ("compound", Some("false".into())),
                ]);
            }
        }
    }
    cat::render(req, &cols(C::SEGMENTS_COLS), rows)
}

// ------------------------------------------------------------ clear cache

pub fn clear_cache(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    // No body (ES `RestClearIndicesCacheAction`): refused with ES's 400,
    // not answered with an empty-cache 200.
    if let Some(e) = crate::front::broadcast_body_rejection(req) {
        return RestResp::error(e.status, &e.kind, &e.reason);
    }
    let names = match resolve_from_params(node, req) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let st = node.state();
    let (mut total, mut successful) = (0usize, 0usize);
    for name in names.iter().collect::<BTreeSet<_>>() {
        total += st.routing_table.all_shards().filter(|s| &s.index == name).count();
        successful += active_ids(&st, name).len();
    }
    RestResp::json(200, &json!({"_shards": {"total": total, "successful": successful, "failed": 0}}))
}

// -------------------------------------------------------- resolve cluster

/// `GET /_resolve/cluster[/{expr}]`: the local cluster only when the
/// expression has local parts, plus an entry per named (or, without an
/// expression, configured) remote cluster, none of them connected.
pub async fn resolve_cluster(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let name = req.segments().get(2).cloned().filter(|s| !s.is_empty());
    let (local_expr, mut out) = match elasticsearch_engine::ccr::remote::resolve_cluster_remotes(&node.engine, name.as_deref()) {
        Ok(planned) => planned,
        Err(e) => return es_err(&e),
    };
    let Some(local_expr) = local_expr else { return RestResp::json(200, &Value::Object(out)); };
    let query: Vec<&str> = req.query().split('&').filter(|kv| !kv.is_empty() && !kv.starts_with("timeout")).collect();
    let mut sub = req.clone();
    if let Err(e)=sub.set_method("GET").and_then(|_|sub.set_body(&[])).and_then(|_|sub.set_uri_fmt(format_args!("/_resolve/index/{local_expr}{}{}",if query.is_empty(){""}else{"?"},front::UriJoin(query.iter(),"&")))) { return es_err(&e); }
    let r = local_engine(node, &sub).await;
    let body = r.body_json().unwrap_or(Value::Null);
    let info = if r.is_success() {
        let nonempty = |k: &str| body.get(k).and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false);
        let open = body.get("indices").and_then(Value::as_array).map(|a| a.iter().any(|i| i.get("attributes").and_then(Value::as_array).map(|at| at.iter().any(|x| x.as_str() == Some("open"))).unwrap_or(false))).unwrap_or(false);
        json!({
            "connected": true,
            "skip_unavailable": false,
            "matching_indices": open || nonempty("aliases") || nonempty("data_streams"),
            "version": {"number": ES_VERSION, "build_flavor": elasticsearch_engine::version::BUILD_FLAVOR, "minimum_wire_compatibility_version": elasticsearch_engine::version::MIN_WIRE_COMPAT, "minimum_index_compatibility_version": elasticsearch_engine::version::MIN_INDEX_COMPAT},
        })
    } else {
        let reason = body.pointer("/error/reason").and_then(Value::as_str).unwrap_or("").to_string();
        json!({"connected": true, "skip_unavailable": false, "error": reason})
    };
    out.insert("(local)".into(), info);
    RestResp::json(200, &Value::Object(out))
}

// ------------------------------------------------------------- knn search

/// Deprecated `_knn_search`: `{knn, filter, fields, _source, docvalue_fields,
/// stored_fields}` run as a `_search` with a top-level `knn` section
/// (`size` = `k`), with the deprecation warning ES emits.
pub async fn knn_search(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let index = index_expr(req).unwrap_or_default();
    let body = req.json().unwrap_or(Value::Null);
    let Some(o) = body.as_object() else {
        return es_err(&EsError::parsing("[knn_search] request body must be an object"));
    };
    let mut knn = match o.get("knn") {
        Some(Value::Object(k)) => k.clone(),
        Some(_) => return es_err(&EsError::parsing("[knn] must be an object")),
        None => return es_err(&EsError::action_validation(&["[knn] section must be specified"])),
    };
    let mut search = Map::new();
    for (k, v) in o {
        match k.as_str() {
            "knn" => {}
            "filter" => {
                let filters: Vec<Value> = match v {
                    Value::Array(a) => a.clone(),
                    other => vec![other.clone()],
                };
                if !filters.is_empty() {
                    knn.insert("filter".into(), Value::Array(filters));
                }
            }
            "fields" | "_source" | "docvalue_fields" | "stored_fields" => {
                search.insert(k.clone(), v.clone());
            }
            other => return es_err(&EsError::parsing(format!("[1:1] [knn_search] unknown field [{other}]"))),
        }
    }
    if let Some(k) = knn.get("k").and_then(Value::as_u64) {
        search.insert("size".into(), Value::from(k));
    }
    search.insert("knn".into(), Value::Object(knn));
    let mut sub = req.clone();
    if let Err(e)=sub.set_method("POST") { return es_err(&e); }
    let query = req.query();
    if let Err(e)=sub.set_uri_fmt(format_args!("/{index}/_search{}{query}",if query.is_empty(){""}else{"?"})).and_then(|_|sub.set_json(&Value::Object(search))) { return es_err(&e); }
    if !sub.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        if let Err(e)=sub.push_header("content-type",&req.json_content_type()) { return es_err(&e); }
    }
    let mut r = front::read_route(node, &sub, &index).await;
    if r.is_success() {
        r.push_header(("Warning".into(), format!("299 Elasticsearch-{ES_VERSION}-{BUILD_HASH} \"The kNN search API has been replaced by the `knn` option in the search API.\"")));
    }
    r
}
