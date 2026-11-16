//! Write-path APIs whose targets span several indices, and data stream
//! health taken from the cluster state.
//!
//! * `/{target}/_update_by_query|_delete_by_query`: a wildcard, alias or
//!   data stream target resolves to several concrete indices whose primaries
//!   may live on different nodes. Each concrete index runs on its own
//!   primary (the normal replicated write path) and the per-index
//!   `BulkByScrollResponse`s are merged, as ES's single task reports them.
//! * `GET /_data_stream[/{name}]`: the engine only sees local copies, so its
//!   `status` is replaced by the cluster health of the backing indices.

use super::{await_mirror, encode_component, local_canonical, write, RestReq, RestResp};
use crate::node::ClusterNode;
use axum::http::Method;
use elasticsearch_cluster::health::{index_health, HealthStatus};
use elasticsearch_engine::node::ResolveOpts;
use elasticsearch_engine::reindex::{parse_by_query, Kind};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Is this an `/{target}/_update_by_query|_delete_by_query` request?
pub(super) fn is_by_query(req: &RestReq) -> bool {
    let segs = req.segments();
    segs.len() == 2 && req.method() == Method::POST && matches!(segs[1].as_str(), "_update_by_query" | "_delete_by_query")
}

pub(super) async fn by_query(node: &Arc<ClusterNode>, req: &RestReq, index: &str) -> RestResp {
    // A per-shard sub-request of this fan-out (forwarded to the shard's
    // primary node): run it on that shard, never fan out again. Client
    // requests have the header stripped.
    if crate::shard_engines::shard_of(&req.headers).is_some() {
        return write(node, req, index).await;
    }
    let params = req.params();
    let opts = ResolveOpts::from_params(&params);
    await_mirror(node).await;
    let engine = node.engine.clone();
    let expr = index.to_string();
    let routing = params.get("routing").cloned();
    // Every concrete index with what the expression reached it through: the
    // alias filter (`None` when named directly or through an unfiltered
    // alias) and the search routing (alias `search_routing` intersected
    // with the request's `routing`, ES `IndexNameExpressionResolver`).
    let resolved = tokio::task::spawn_blocking(move || {
        engine.resolve(&expr, opts).map(|ts| {
            let any = !ts.is_empty();
            let routed = elasticsearch_http::apply_search_routing(ts, routing.as_deref());
            (any, routed.into_iter().map(|t| (t.svc.name.clone(), (t.alias_filter.clone(), t.routing.clone()))).collect::<BTreeMap<_, _>>())
        })
    })
    .await
    .ok();
    let names = match resolved {
        Some(Ok((true, n))) => n,
        // index_not_found, empty wildcard matches, ...: exactly as the
        // engine renders them (never auto-creates an index).
        _ => return local_canonical(node, req).await,
    };
    // (index, shard, alias filter) targets: every shard of a multi-shard
    // index the routing selects runs the by-query on its own primary
    // (documents never move between shards, as the op keeps `_id` /
    // `_routing`), under the filter of the alias that selected it.
    let targets: Vec<Target> = {
        let st = node.state();
        names
            .iter()
            .flat_map(|(n, (filter, routing))| {
                let k = crate::shard_engines::shards_of(&st.metadata, n);
                crate::shard_engines::search_shards(&st.metadata, n, routing.as_deref()).into_iter().map(move |s| Target { index: n.clone(), shard: (k > 1).then_some(s), filter: filter.clone() })
            })
            .collect()
    };
    if targets.len() == 1 && targets[0].shard.is_none() && targets[0].filter.is_none() && names.contains_key(index) {
        return write(node, req, index).await;
    }
    let wait = params.get("wait_for_completion").map(|v| v != "false").unwrap_or(true);
    let op = req.segments()[1].clone();
    let body = req.json().or_else(||std::str::from_utf8(&req.body).ok().and_then(|text|elasticsearch_engine::source_json::update_request(text).ok()).map(|(body,_)|body));
    let kind = if op == "_delete_by_query" { Kind::DeleteByQuery } else { Kind::UpdateByQuery };
    // The query an alias filter is combined with (the body's, or `q`).
    let query = parse_by_query(kind, index, body.as_ref().unwrap_or(&json!({})), &params).ok().and_then(|r| r.query);
    let body_max = body.as_ref().and_then(|b| b.get("max_docs").or_else(|| b.get("size"))).and_then(Value::as_u64);
    let remaining: Option<u64> = params.get("max_docs").and_then(|v| v.parse().ok()).or(body_max);
    let scroll_size = params.get("scroll_size").and_then(|v| v.parse::<u64>().ok()).filter(|s| *s > 0).unwrap_or(1000);
    if !wait && targets.len() == 1 && names.len() == 1 && targets[0].shard.is_none() {
        // One single-shard index: the engine's own task on its primary (the
        // engine resolves the expression itself, alias filter included).
        return write(node, req, index).await;
    }
    if !wait {
        // One cluster task for the whole fan-out (ES runs a single
        // BulkByScroll task over every shard of every target).
        let parsed = parse_by_query(kind, index, body.as_ref().unwrap_or(&json!({})), &params);
        let Ok(r) = parsed else {
            // Parameter / body errors render as the engine renders them.
            return write(node, req, index).await;
        };
        let base_query: Vec<String> = req.query().split('&').filter(|p| !p.is_empty() && !p.starts_with("max_docs=") && !p.starts_with("wait_for_completion=")).map(str::to_string).collect();
        let (n, sub) = (node.clone(), req.clone());
        let work = async move { fan_out(&n, &sub, &targets, &op, body, query, remaining, &base_query, scroll_size).await };
        return super::bg_task::spawn(node, req, kind.action(), r.description(), r.rps, work);
    }
    let base_query: Vec<String> = req.query().split('&').filter(|p| !p.is_empty() && !p.starts_with("max_docs=")).map(str::to_string).collect();
    fan_out(node, req, &targets, &op, body, query, remaining, &base_query, scroll_size).await
}

/// One shard a by-query runs on.
pub(super) struct Target {
    index: String,
    /// `None`: the index's only shard.
    shard: Option<u32>,
    /// The filter of the alias the expression reached the index through.
    filter: Option<Value>,
}

/// The by-query `body` restricted to `filter` (ES adds the alias filter to
/// the search of a BulkByScroll request): the request's `query` (the body's
/// or `q`'s, `match_all` without one) as a `bool` with the filter.
fn filtered_body(body: Option<&Value>, query: Option<&Value>, filter: &Value) -> Value {
    let mut out = match body {
        Some(Value::Object(o)) => Value::Object(o.clone()),
        _ => json!({}),
    };
    let q = query.cloned().unwrap_or_else(|| json!({"match_all": {}}));
    out["query"] = json!({"bool": {"must": [q], "filter": [filter.clone()]}});
    out
}

/// The `BulkByScrollResponse` of a by-query that ran on no shard (its
/// routing selects none).
fn empty_response(kind: Kind, rps: f64) -> Value {
    let mut m = serde_json::Map::new();
    m.insert("took".into(), json!(0));
    m.insert("timed_out".into(), json!(false));
    m.extend(elasticsearch_engine::reindex::Stats::default().status_json(kind, rps, 0, None));
    m.insert("failures".into(), json!([]));
    Value::Object(m)
}

/// Is `v` a `BulkByScrollResponse` (a by-query that ran and reported its
/// failures), rather than an error that stopped it?
fn is_bulk_by_scroll(v: &Value) -> bool {
    v.get("failures").is_some_and(Value::is_array) && v.get("total").is_some()
}

/// ES `ScrollableHitSource.SearchFailure` for a shard whose by-query did
/// not run (its error response).
fn shard_failure(index: &str, shard: Option<u32>, status: u16, v: &Value) -> (u16, Value) {
    let reason = match v.get("error") {
        Some(Value::Object(e)) => {
            let mut e = e.clone();
            e.remove("root_cause");
            Value::Object(e)
        }
        Some(other) => json!({"type": "exception", "reason": other}),
        None => json!({"type": "exception", "reason": v.to_string()}),
    };
    let status = v.get("status").and_then(Value::as_u64).map(|s| s as u16).unwrap_or(status).max(400);
    (status, json!({"index": index, "shard": shard.unwrap_or(0), "node": null, "status": status, "reason": reason}))
}

/// Run the by-query on every (index, shard) target in turn, carrying the
/// remaining `max_docs`, and merge the responses.
#[allow(clippy::too_many_arguments)]
async fn fan_out(node: &Arc<ClusterNode>, req: &RestReq, targets: &[Target], op: &str, mut body: Option<Value>, query: Option<Value>, mut remaining: Option<u64>, base_query: &[String], scroll_size: u64) -> RestResp {
    let raw_script=std::str::from_utf8(&req.body).ok().and_then(|text|elasticsearch_engine::source_json::update_request(text).ok()).and_then(|(_,raw)|raw).and_then(|raw|raw.get("script").cloned());
    if let Some(b) = body.as_mut().and_then(Value::as_object_mut) {
        if remaining.is_some() {
            b.remove("max_docs");
            b.remove("size");
        }
    }
    let quota = remaining;
    let mut parts: Vec<Value> = Vec::new();
    // Under a cluster task (`wait_for_completion=false`): stop between
    // targets once it is cancelled, and run each target as its child.
    let task = super::bg_task::current();
    let mut canceled = false;
    // A shard whose by-query did not run: the run stops there, reporting it
    // with what the shards before it did (ES fails a BulkByScroll on any
    // search failure and never claims a complete run).
    let mut stopped: Option<(u16, Value)> = None;
    for Target { index: name, shard, filter } in targets {
        if remaining == Some(0) {
            break;
        }
        if task.as_ref().map(|t| t.is_cancelled()).unwrap_or(false) {
            canceled = true;
            break;
        }
        let mut q = match &task {
            // The task's current rate: `_rethrottle` applies to the targets
            // still to run.
            Some(t) => base_query.iter().filter(|p| !p.starts_with("requests_per_second=")).cloned().chain(std::iter::once(format!("requests_per_second={}", t.rps()))).collect::<Vec<_>>().join("&"),
            None => base_query.join("&"),
        };
        if let Some(m) = remaining {
            if !q.is_empty() {
                q.push('&');
            }
            q.push_str(&format!("max_docs={m}"));
        }
        let mut sub = req.clone();
        if let Some(t) = &task {
            if let Err(e)=sub.push_header(crate::rest::tasks::PARENT_HEADER,&format!("{}:{}",node.local.id,t.id)) {return RestResp::json(e.status,&e.to_json())}
        }
        if let Err(e)=sub.set_uri(&format!("/{}/{op}{}{q}", encode_component(name), if q.is_empty() { "" } else { "?" })) {return RestResp::json(e.status,&e.to_json())}
        if let Some(f) = filter {
            let b = filtered_body(body.as_ref(), query.as_ref(), f);
            let encoded = if let Some(script)=&raw_script {
                let mut source=elasticsearch_engine::source_json::Source::from_value(&b);
                if let elasticsearch_engine::source_json::Source::Object(m)=&mut source {m.insert("script".into(),script.clone());}
                source.text().into_bytes()
            } else {serde_json::to_vec(&b).unwrap_or_default()};
            if let Err(e)=sub.retain_headers(|(k,_)| !k.eq_ignore_ascii_case("content-type") && !k.eq_ignore_ascii_case("content-length")).and_then(|_| sub.push_header("content-type","application/json")).and_then(|_| sub.set_body(&encoded)) {return RestResp::json(e.status,&e.to_json())}
        } else if let Some(b) = &body {
            let encoded = if let Some(script)=&raw_script {
                let mut source=elasticsearch_engine::source_json::Source::from_value(b);
                if let elasticsearch_engine::source_json::Source::Object(m)=&mut source {m.insert("script".into(),script.clone());}
                source.text().into_bytes()
            } else {serde_json::to_vec(b).unwrap_or_default()};
            if let Err(e)=sub.set_body(&encoded) {return RestResp::json(e.status,&e.to_json())}
        }
        if let Some(k) = shard {
            if let Err(e)=sub.set_shard(*k) {return RestResp::json(e.status,&e.to_json())}
        }
        let r = write_send(node.clone(), sub, name.clone()).await;
        let v = r.body_json().unwrap_or(Value::Null);
        if !r.is_success() && !is_bulk_by_scroll(&v) {
            // Nothing ran yet: the error as the engine renders it.
            if parts.is_empty() {
                return r;
            }
            stopped = Some(shard_failure(name, *shard, r.status, &v));
            break;
        }
        let stop = !r.is_success() || !v["failures"].as_array().is_none_or(Vec::is_empty) || v["timed_out"].as_bool() == Some(true);
        // ES counts the successfully processed documents toward
        // `max_docs` (conflicts skipped with `conflicts: proceed` and noops
        // leave it for the next shards).
        if let Some(m) = remaining.as_mut() {
            let done: u64 = ["created", "updated", "deleted"].iter().filter_map(|k| v[*k].as_u64()).sum();
            *m = m.saturating_sub(done);
        }
        if v["canceled"].is_string() {
            canceled = true;
        }
        parts.push(v);
        // A shard that reported failures or timed out ended the run (ES
        // stops a BulkByScroll at its first failed batch or source page).
        if canceled || stop {
            break;
        }
    }
    let mut out = if parts.is_empty() && stopped.is_none() {
        let kind = if op == "_delete_by_query" { Kind::DeleteByQuery } else { Kind::UpdateByQuery };
        let rps = req.params().get("requests_per_second").and_then(|v| elasticsearch_engine::reindex::parse_rps(v).ok()).unwrap_or(-1.0);
        empty_response(kind, rps)
    } else {
        merge(&parts, scroll_size)
    };
    if canceled {
        if let Some(o) = out.as_object_mut() {
            o.insert("canceled".into(), json!("by user request"));
        }
    }
    // ES `total`: the matches, at most `max_docs`.
    if let (Some(q), Some(t)) = (quota, out["total"].as_u64()) {
        out["total"] = json!(t.min(q));
    }
    if let (Some((_, f)), Some(Value::Array(failures))) = (&stopped, out.get_mut("failures")) {
        failures.push(f.clone());
    }
    // ES `BulkByScrollResponse` status: 408 when timed out, raised to the
    // worst failure's.
    let statuses: Vec<(u16, Value)> = out["failures"].as_array().into_iter().flatten()
        .map(|f| (f["status"].as_u64().map(|s| s as u16).unwrap_or(500), Value::Null)).collect();
    let status = elasticsearch_engine::reindex::response_status(&statuses, out["timed_out"].as_bool() == Some(true));
    RestResp::json(status, &out)
}

/// `write` as a nameable `Send` future: `fan_out` runs inside a spawned
/// task and reaches `dispatch` recursively, whose auto-trait inference
/// cannot see through the cycle.
fn write_send(node: Arc<ClusterNode>, req: RestReq, name: String) -> std::pin::Pin<Box<dyn std::future::Future<Output = RestResp> + Send>> {
    Box::pin(async move { write(&node, &req, &name).await })
}

/// One `BulkByScrollResponse` for the per-index ones: counters add up,
/// `batches` counts scroll pages of the combined scan.
fn merge(parts: &[Value], scroll_size: u64) -> Value {
    let Some(first) = parts.first() else { return json!({}) };
    let mut out = first.clone();
    let Some(o) = out.as_object_mut() else { return out };
    for p in &parts[1..] {
        let Some(po) = p.as_object() else { continue };
        for (k, v) in po {
            match (k.as_str(), o.get_mut(k)) {
                ("requests_per_second" | "throttled_until_millis", _) => {}
                ("timed_out", Some(cur)) => *cur = json!(cur.as_bool().unwrap_or(false) || v.as_bool().unwrap_or(false)),
                ("failures", Some(Value::Array(cur))) => cur.extend(v.as_array().cloned().unwrap_or_default()),
                ("retries", Some(Value::Object(cur))) => {
                    for (rk, rv) in v.as_object().into_iter().flatten() {
                        let sum = cur.get(rk).and_then(Value::as_u64).unwrap_or(0) + rv.as_u64().unwrap_or(0);
                        cur.insert(rk.clone(), json!(sum));
                    }
                }
                (_, Some(cur)) if cur.is_u64() && v.is_u64() => *cur = json!(cur.as_u64().unwrap_or(0) + v.as_u64().unwrap_or(0)),
                (_, None) => {
                    o.insert(k.clone(), v.clone());
                }
                _ => {}
            }
        }
    }
    if parts.len() > 1 && o.contains_key("batches") {
        let total = o.get("total").and_then(Value::as_u64).unwrap_or(0);
        o.insert("batches".into(), json!(total.div_ceil(scroll_size)));
    }
    out
}

/// The cluster-wide doc stats request of a rollover's write index: a client
/// request (`hops` 0, so the broadcast covers every node's primaries rather
/// than running on this node's engine as a forwarded request would), as the
/// caller.
fn docs_stats_request(req: &RestReq, index: &str) -> elasticsearch_engine::error::Result<RestReq> {
    let mut out=req.clone();
    out.set_method("GET")?;
    out.set_uri(&format!("/{}/_stats/docs?level=shards",encode_component(index)))?;
    out.retain_headers(|(key,_)| key.eq_ignore_ascii_case("authorization") || key.eq_ignore_ascii_case("es-security-runas-user") || key.eq_ignore_ascii_case(super::SYSTEM_HEADER) || key.eq_ignore_ascii_case(super::IDENTITY_HEADER))?;
    out.set_body(&[])?;out.set_dfs(Default::default())?;out.set_app_plan(None)?;out.hops=0;
    Ok(out)
}

/// `(primary docs, largest primary shard's docs)` of `index` in a
/// `_stats/docs?level=shards` response.
fn primary_docs(stats: &Value, index: &str) -> Option<(u64, u64)> {
    let ix = &stats["indices"][index];
    let docs = ix["primaries"]["docs"]["count"].as_u64()?;
    let shard_docs = ix["shards"].as_object().into_iter().flat_map(|m| m.values()).flat_map(|c| c.as_array().into_iter().flatten()).filter(|c| c["routing"]["primary"].as_bool() == Some(true)).filter_map(|c| c["docs"]["count"].as_u64()).max().unwrap_or(docs);
    Some((docs, shard_docs))
}

/// Rollover stats of `index` from its primaries across the cluster
/// (`_stats/docs,store?level=shards`, as ES's `TransportRolloverAction`
/// reads `IndicesStats`): what the master's engine judges rollover
/// conditions by (its own copy may hold no data). `None` when the stats
/// cannot be gathered.
pub(super) async fn cluster_rollover_stats(node: &Arc<ClusterNode>, index: &str) -> Option<elasticsearch_engine::datastream::RolloverStats> {
    let req = match RestReq::try_new("GET",&format!("/{}/_stats/docs,store?level=shards",encode_component(index)),&super::system_headers(),&[],0,Default::default(),None) {
        Ok(req)=>req,
        Err(error)=>{tracing::warn!(reason=%error.reason,"rollover statistics request admission refused");return None;}
    };
    let st = dispatch_boxed(node, req).await.body_json()?;
    rollover_stats_of(&st, index)
}

/// [`cluster_rollover_stats`] from a `_stats/docs,store?level=shards` body.
fn rollover_stats_of(st: &Value, index: &str) -> Option<elasticsearch_engine::datastream::RolloverStats> {
    let ix = &st["indices"][index];
    let docs = ix["primaries"]["docs"]["count"].as_u64()?;
    // The shard store includes state and translog files even before a
    // segment contains a document. Those files must not satisfy ES's
    // segment-size rollover conditions. Keep the reported bytes when deleted
    // documents remain: a delete-only segment still occupies index storage.
    let segment_size = |stats: &Value| {
        if stats["docs"]["count"].as_u64() == Some(0)
            && stats["docs"]["deleted"].as_u64().unwrap_or(0) == 0 {
            0
        } else {
            stats["store"]["size_in_bytes"].as_u64().unwrap_or(0)
        }
    };
    let mut size = 0u64;
    let mut primary_shard_docs = 0;
    let mut primary_shard_size = 0;
    let mut found_primary = false;
    for primary in ix["shards"].as_object().into_iter().flat_map(|m| m.values())
        .flat_map(|copies| copies.as_array().into_iter().flatten())
        .filter(|copy| copy["routing"]["primary"].as_bool() == Some(true)) {
        found_primary = true;
        let bytes = segment_size(primary);
        size = size.saturating_add(bytes);
        primary_shard_docs = primary_shard_docs.max(primary["docs"]["count"].as_u64().unwrap_or(0));
        primary_shard_size = primary_shard_size.max(bytes);
    }
    if !found_primary {
        size = segment_size(&ix["primaries"]);
        primary_shard_docs = docs;
        primary_shard_size = size;
    }
    Some(elasticsearch_engine::datastream::RolloverStats { docs, size, primary_shard_docs, primary_shard_size })
}

/// `dispatch` behind a boxed future (this module runs inside `dispatch`).
fn dispatch_boxed(node: &Arc<ClusterNode>, req: RestReq) -> futures::future::BoxFuture<'static, RestResp> {
    let node = node.clone();
    Box::pin(async move { super::dispatch(&node, req).await })
}

/// `POST /{target}/_rollover[/{new}]` with a doc-count condition, on the
/// master. The engine judges conditions on its local copy, which is an empty
/// shell when the write index's primary lives elsewhere: doc conditions are
/// judged here on the cluster-wide primary doc stats instead (as ES's
/// `TransportRolloverAction` does from `IndicesStats`), and the engine then
/// rolls over unconditionally, or not at all. `None`: no doc condition, a
/// lazy rollover, or the dry run failed (the normal path renders the error).
#[allow(dead_code)]
pub(super) async fn rollover_on_cluster_docs(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let segs = req.segments();
    if req.method() != Method::POST || !(2..=3).contains(&segs.len()) || segs[1] != "_rollover" || segs[0].starts_with('_') {
        return None;
    }
    let params = req.params();
    let flag = |k: &str| params.get(k).map(|v| v.is_empty() || v == "true").unwrap_or(false);
    if flag("lazy") {
        return None;
    }
    let mut body = req.json()?;
    if !body.get("conditions")?.as_object()?.keys().any(|k| k.ends_with("_docs")) {
        return None;
    }
    let dry_run = flag("dry_run");
    let mut dry = req.clone();
    if !dry_run {
        let suffix=if dry.uri.contains('?') { "&dry_run=true" } else { "?dry_run=true" };
        let uri=match super::SharedText::try_format(format_args!("{}{suffix}",dry.uri)) {Ok(uri)=>uri,Err(error)=>return Some(super::err(error))};
        if let Err(error)=dry.set_uri(&uri) {return Some(super::err(error));}
    }
    let dr = local_canonical(node, &dry).await;
    if !dr.is_success() {
        return None;
    }
    let mut v = dr.body_json()?;
    let old = v["old_index"].as_str()?.to_string();
    let stats_req=match docs_stats_request(req,&old) {Ok(r)=>r,Err(e)=>return Some(RestResp::json(e.status,&e.to_json()))};
    let st = dispatch_boxed(node, stats_req).await.body_json()?;
    let (docs, shard_docs) = primary_docs(&st, &old)?;
    let conds = v["conditions"].as_object_mut()?;
    for (k, met) in conds.iter_mut() {
        let Some((name, n)) = k.trim_start_matches('[').trim_end_matches(']').split_once(": ") else { continue };
        let Ok(n) = n.trim().parse::<u64>() else { continue };
        *met = json!(match name {
            "max_docs" | "min_docs" => docs >= n,
            "max_primary_shard_docs" | "min_primary_shard_docs" => shard_docs >= n,
            _ => continue,
        });
    }
    let max: Vec<bool> = conds.iter().filter(|(k, _)| k.starts_with("[max_")).map(|(_, m)| m.as_bool() == Some(true)).collect();
    let min_ok = conds.iter().filter(|(k, _)| k.starts_with("[min_")).all(|(_, m)| m.as_bool() == Some(true));
    let met = (max.is_empty() || max.contains(&true)) && min_ok;
    let conds = Value::Object(conds.clone());
    if dry_run || !met {
        v["dry_run"] = json!(dry_run);
        return Some(RestResp::json(200, &v));
    }
    body.as_object_mut()?.remove("conditions");
    let mut real = req.clone();
    if let Err(e)=real.set_json(&body) {return Some(RestResp::json(e.status,&e.to_json()))}
    if let Err(e)=real.retain_headers(|(k,_)|!k.eq_ignore_ascii_case("content-length")) {return Some(RestResp::json(e.status,&e.to_json()))}
    let mut resp = local_canonical(node, &real).await;
    if let Some(mut out) = resp.body_json().filter(|_| resp.is_success()) {
        out["conditions"] = conds;
        resp.replace_json(&out);
        resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
    }
    Some(resp)
}

/// `PUT /_data_stream/{name}` waits for the primaries of the new backing
/// indices, as index creation does (`wait_for_active_shards` 1); otherwise an
/// immediate `GET _data_stream` reports an initializing primary.
pub(super) async fn wait_data_stream_primaries(node: &Arc<ClusterNode>, name: &str) {
    let engine = node.engine.clone();
    let expr = name.to_string();
    let Ok(Ok(targets)) = tokio::task::spawn_blocking(move || engine.resolve(&expr, ResolveOpts::default()).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<Vec<_>>())).await else { return };
    for index in targets {
        super::wait_for_primary(node, &index, std::time::Duration::from_secs(30)).await;
    }
}

/// Is this a `GET /_data_stream[/{name}]` request?
pub(super) fn is_data_stream_get(req: &RestReq) -> bool {
    let segs = req.segments();
    req.method() == Method::GET && segs.first().map(|s| s == "_data_stream").unwrap_or(false) && segs.len() <= 2 && !segs.get(1).map(|s| s.starts_with('_')).unwrap_or(false)
}

/// Timestamp statistics must include backing indices on other nodes, including
/// older generations after rollover. Search resolves the stream to all copies
/// and selects one active copy per shard.
pub(super) async fn data_stream_stats(node: &Arc<ClusterNode>, req: &RestReq, mut resp: RestResp) -> RestResp {
    if !resp.is_success() { return resp; }
    let Some(mut body) = resp.body_json() else { return resp };
    for stream in body["data_streams"].as_array_mut().into_iter().flatten() {
        let Some(name) = stream["data_stream"].as_str().map(str::to_string) else { continue };
        let mut search=req.clone();
        if let Err(e)=search.set_json(&json!({"size":0,"aggs":{"maximum_timestamp":{"max":{"field":"@timestamp"}}}})) {return RestResp::json(e.status,&e.to_json())}
        if let Err(e)=search.set_method("POST") {return RestResp::json(e.status,&e.to_json())}
        if let Err(e)=search.set_uri(&format!("/{}/_search?expand_wildcards=all&ignore_unavailable=true&allow_no_indices=true", encode_component(&name))) {return RestResp::json(e.status,&e.to_json())}
        let result = super::read(node, &search, &name).await;
        if !result.is_success() { return result; }
        let value = result.body_json().and_then(|v| v["aggregations"]["maximum_timestamp"]["value"].as_f64()).unwrap_or(0.0);
        stream["maximum_timestamp"] = json!(value as i64);
    }
    resp.replace_json(&body);
    resp.retain_headers(|(key, _)| !key.eq_ignore_ascii_case("content-length"));
    resp
}

/// The engine's answer with each data stream's `status` taken from the
/// cluster health of its backing indices (the worst one wins, as in ES).
pub(super) async fn data_stream_get(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let mut resp = local_canonical(node, req).await;
    if !resp.is_success() {
        return resp;
    }
    let Some(mut v) = resp.body_json() else { return resp };
    let state = node.state();
    let mut changed = false;
    for ds in v["data_streams"].as_array_mut().into_iter().flatten() {
        if ds.get("status").is_none() {
            continue;
        }
        let mut worst: Option<HealthStatus> = None;
        for i in ds["indices"].as_array().into_iter().flatten() {
            let Some(name) = i["index_name"].as_str() else { continue };
            let (Some(m), Some(irt)) = (state.metadata.indices.get(name), state.routing_table.indices.get(name)) else { continue };
            let s = index_health(m, irt).status;
            worst = Some(worst.map_or(s, |w| w.max(s)));
        }
        if let Some(s) = worst {
            ds["status"] = json!(s.name().to_ascii_uppercase());
            changed = true;
        }
        if ds.get("maximum_timestamp").is_some() {
            let indices = ds["indices"].as_array().into_iter().flatten().filter_map(|i| i["index_name"].as_str()).collect::<Vec<_>>().join(",");
            if !indices.is_empty() {
                let mut search=req.clone();
        if let Err(e)=search.set_json(&json!({"size":0,"aggs":{"m":{"max":{"field":"@timestamp"}}}})) {return RestResp::json(e.status,&e.to_json())}
                if let Err(e)=search.set_method("POST") {return RestResp::json(e.status,&e.to_json())}
                if let Err(e)=search.set_uri(&format!("/{}/_search?expand_wildcards=all&ignore_unavailable=true&allow_no_indices=true", encode_component(&indices))) {return RestResp::json(e.status,&e.to_json())}
                let result = super::read(node, &search, &indices).await;
                if !result.is_success() { return result; }
                let value = result.body_json().and_then(|v| v["aggregations"]["m"]["value"].as_f64()).unwrap_or(0.0);
                ds["maximum_timestamp"] = json!(value as i64);
                changed = true;
            }
        }
    }
    if changed {
        resp.replace_json(&v);
        resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_sums_counters_and_recounts_batches() {
        let a = json!({"took": 3, "timed_out": false, "total": 1, "updated": 1, "deleted": 0, "batches": 1, "version_conflicts": 0, "noops": 0, "retries": {"bulk": 0, "search": 0}, "throttled_millis": 0, "requests_per_second": -1.0, "throttled_until_millis": 0, "failures": []});
        let b = json!({"took": 2, "timed_out": false, "total": 1, "updated": 1, "deleted": 0, "batches": 1, "version_conflicts": 0, "noops": 0, "retries": {"bulk": 1, "search": 0}, "throttled_millis": 0, "requests_per_second": -1.0, "throttled_until_millis": 0, "failures": []});
        let m = merge(&[a, b], 1000);
        assert_eq!(m["updated"], 2);
        assert_eq!(m["total"], 2);
        assert_eq!(m["batches"], 1);
        assert_eq!(m["took"], 5);
        assert_eq!(m["retries"]["bulk"], 1);
        assert_eq!(m["failures"], json!([]));
        assert_eq!(m["requests_per_second"], -1.0);
    }

    #[test]
    fn alias_filter_wraps_the_request_query() {
        let f = json!({"term": {"tag": "a"}});
        let body = json!({"query": {"match": {"x": 1}}, "conflicts": "proceed"});
        let q = body["query"].clone();
        assert_eq!(filtered_body(Some(&body), Some(&q), &f), json!({"query": {"bool": {"must": [{"match": {"x": 1}}], "filter": [f.clone()]}}, "conflicts": "proceed"}));
        // No body / no query (`_update_by_query` of everything): match_all.
        assert_eq!(filtered_body(None, None, &f), json!({"query": {"bool": {"must": [{"match_all": {}}], "filter": [f.clone()]}}}));
        // `q`: the parsed query_string.
        let qs = json!({"query_string": {"query": "v:1"}});
        assert_eq!(filtered_body(Some(&json!({})), Some(&qs), &f)["query"]["bool"]["must"][0], qs);
    }

    #[test]
    fn a_shard_that_did_not_run_is_a_search_failure() {
        let err = json!({"error": {"root_cause": [], "type": "unavailable_shards_exception", "reason": "[i][1] primary shard is not active"}, "status": 503});
        let (status, f) = shard_failure("i", Some(1), 503, &err);
        assert_eq!(status, 503);
        assert_eq!(f, json!({"index": "i", "shard": 1, "node": null, "status": 503, "reason": {"type": "unavailable_shards_exception", "reason": "[i][1] primary shard is not active"}}));
        assert!(is_bulk_by_scroll(&json!({"total": 1, "failures": []})));
        assert!(!is_bulk_by_scroll(&err));
    }

    #[test]
    fn empty_response_is_a_zero_run() {
        let v = empty_response(Kind::DeleteByQuery, -1.0);
        assert_eq!((v["total"].as_u64(), v["deleted"].as_u64(), v["timed_out"].as_bool()), (Some(0), Some(0), Some(false)));
        assert_eq!(v["failures"], json!([]));
        assert!(v.get("updated").is_none());
    }
}

#[cfg(test)]
mod rollover_docs_tests {
    use super::*;

    #[test]
    fn stats_request_is_a_client_request_as_the_caller() {
        let req = RestReq::try_new("POST","/logs/_rollover",&[("authorization".into(),"Basic dTpw".into()),("x-other".into(),"1".into())],&[],2,Default::default(),None).unwrap();
        let s = docs_stats_request(&req, "logs-000001").unwrap();
        assert_eq!(s.hops, 0);
        assert_eq!(s.uri, "/logs-000001/_stats/docs?level=shards");
        assert_eq!(s.headers, vec![("authorization".to_string(), "Basic dTpw".to_string())]);
    }

    #[test]
    fn rollover_stats_read_cluster_primaries() {
        let st = json!({"indices": {"i": {"primaries": {"docs": {"count": 5}, "store": {"size_in_bytes": 500}}, "shards": {
            "0": [{"routing": {"primary": true}, "docs": {"count": 3}, "store": {"size_in_bytes": 300}}, {"routing": {"primary": false}, "docs": {"count": 9}, "store": {"size_in_bytes": 900}}],
            "1": [{"routing": {"primary": true}, "docs": {"count": 2}, "store": {"size_in_bytes": 200}}]}}}});
        let s = rollover_stats_of(&st, "i").unwrap();
        assert_eq!((s.docs, s.size, s.primary_shard_docs, s.primary_shard_size), (5, 500, 3, 300));
        assert!(rollover_stats_of(&st, "missing").is_none());
    }

    #[test]
    fn primary_docs_reads_totals_and_largest_primary() {
        let st = json!({"indices": {"i": {"primaries": {"docs": {"count": 5}}, "shards": {
            "0": [{"routing": {"primary": true}, "docs": {"count": 3}}, {"routing": {"primary": false}, "docs": {"count": 9}}],
            "1": [{"routing": {"primary": true}, "docs": {"count": 2}}]}}}});
        assert_eq!(primary_docs(&st, "i"), Some((5, 3)));
        assert_eq!(primary_docs(&st, "missing"), None);
    }
}
