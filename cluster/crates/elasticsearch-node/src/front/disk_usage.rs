//! `POST /{index}/_disk_usage` under the cluster.
//!
//! ES analyses one copy of every shard. The engine measures the segments of
//! the copy it holds, so the request must run where the copies are: a
//! single-shard target goes to a node holding it (like any read), a
//! multi-shard index runs once per shard on a holder of that shard's engine
//! and the per-shard usages are summed.

use super::{await_mirror, exec_shard, read, RestReq, RestResp};
use crate::node::ClusterNode;
use crate::stats_shards::{add_json, fix_human};
use elasticsearch_engine::node::ResolveOpts;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::sync::Arc;

pub(super) async fn disk_usage(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    await_mirror(node).await;
    let opts = ResolveOpts::from_params(&req.params());
    let engine = node.engine.clone();
    let expr = targets.to_string();
    let resolved = tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok();
    let names = match resolved {
        Some(Ok(n)) if !n.is_empty() => n,
        // Errors and parameter validation render as the engine renders them.
        _ => return read(node, req, targets).await,
    };
    let st = node.state();
    if !names.iter().any(|n| crate::shard_engines::shards_of(&st.metadata, n) > 1) {
        return read(node, req, targets).await;
    }
    let query = req.query().to_string();
    let mut futs = Vec::new();
    for name in &names {
        for k in 0..crate::shard_engines::shards_of(&st.metadata, name) {
            // Holders of an active copy: this node first, then primaries.
            let mut cands: Vec<(bool, bool, String)> = st
                .routing_table
                .shard_table(name, k)
                .map(|t| t.searchable_copies(&st.nodes).filter_map(|c| c.node_id.clone().map(|n| (n != node.local.id, !c.primary, n))).collect())
                .unwrap_or_default();
            cands.sort();
            let mut sub = req.clone();
            if let Err(error) = sub.set_uri(&(format!("/{}/_disk_usage{}{}", super::encode_component(name), if query.is_empty() { "" } else { "?" }, query))) { return RestResp::json(error.status, &error.to_json()); }
            let node = node.clone();
            let name = name.clone();
            futs.push(async move {
                let mut last = RestResp::error(503, "no_shard_available_action_exception", &format!("[{name}][{k}] no active shard copy"));
                for (_, _, n) in cands {
                    let r = exec_shard(&node, &n, &sub, k).await;
                    if !matches!(r.status, 502 | 503) {
                        return r;
                    }
                    last = r;
                }
                last
            });
        }
    }
    let (mut total, mut ok) = (0u64, 0u64);
    let mut out = Map::new();
    for r in futures::future::join_all(futs).await {
        total += 1;
        if !r.is_success() {
            if r.status == 400 || r.status == 404 {
                return r;
            }
            continue;
        }
        let Some(Value::Object(body)) = r.body_json() else { continue };
        ok += 1;
        for (k, v) in body {
            if k == "_shards" {
                continue;
            }
            match out.get_mut(&k) {
                Some(cur) => add_json(cur, &v),
                None => {
                    out.insert(k, v);
                }
            }
        }
    }
    let mut body = Map::new();
    body.insert("_shards".into(), json!({"total": total, "successful": ok, "failed": total - ok}));
    for (k, mut v) in out {
        fix_human(&mut v);
        body.insert(k, v);
    }
    RestResp::json(200, &Value::Object(body))
}

/// `GET /{index}/_field_usage_stats` under the cluster.
///
/// Usage is recorded where searches execute, i.e. on the nodes holding
/// copies. The engine tracks it per index per process (all shard engines of
/// a node share one counter), so the request runs once on every node holding
/// an active copy and each node contributes one `shards` entry attributed to
/// it.
pub(super) async fn field_usage(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    await_mirror(node).await;
    let opts = ResolveOpts::from_params(&req.params());
    let engine = node.engine.clone();
    let expr = targets.to_string();
    let resolved = tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok();
    let names = match resolved {
        Some(Ok(n)) if !n.is_empty() => n,
        _ => return read(node, req, targets).await,
    };
    let st = node.state();
    let query = req.query().to_string();
    let mut futs = Vec::new();
    for name in &names {
        // node -> (a shard it holds, holds a primary)
        let mut holders: std::collections::BTreeMap<String, (u32, bool)> = Default::default();
        for k in 0..crate::shard_engines::shards_of(&st.metadata, name) {
            if let Some(t) = st.routing_table.shard_table(name, k) {
                for c in t.active_copies() {
                    if let Some(n) = c.node_id.clone() {
                        let e = holders.entry(n).or_insert((k, false));
                        e.1 |= c.primary;
                    }
                }
            }
        }
        if holders.is_empty() {
            return read(node, req, targets).await;
        }
        for (n, (k, primary)) in holders {
            let mut sub = req.clone();
            if let Err(error) = sub.set_uri(&(format!("/{}/_field_usage_stats{}{}", super::encode_component(name), if query.is_empty() { "" } else { "?" }, query))) { return RestResp::json(error.status, &error.to_json()); }
            let node = node.clone();
            let name = name.clone();
            futs.push(async move { (name, n.clone(), primary, exec_shard(&node, &n, &sub, k).await) });
        }
    }
    let (mut total, mut ok) = (0u64, 0u64);
    let mut out: Map<String, Value> = Map::new();
    for (name, n, primary, r) in futures::future::join_all(futs).await {
        total += 1;
        if !r.is_success() {
            if r.status == 400 || r.status == 404 {
                return r;
            }
            continue;
        }
        let Some(body) = r.body_json() else { continue };
        ok += 1;
        let entries = body.get(&name).and_then(|v| v.get("shards")).and_then(Value::as_array).cloned().unwrap_or_default();
        let slot = out.entry(name).or_insert_with(|| json!({"shards": []}));
        for mut e in entries {
            if let Some(rt) = e.get_mut("routing").and_then(Value::as_object_mut) {
                rt.insert("node".into(), json!(n));
                rt.insert("primary".into(), json!(primary));
            }
            if let Some(a) = slot["shards"].as_array_mut() {
                a.push(e);
            }
        }
    }
    let mut body = Map::new();
    body.insert("_shards".into(), json!({"total": total, "successful": ok, "failed": total - ok}));
    body.extend(out);
    RestResp::json(200, &Value::Object(body))
}
