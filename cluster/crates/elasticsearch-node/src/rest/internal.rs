//! `_internal/desired_balance` (GET/DELETE) and `_internal/prevalidate_node_removal`.
//!
//! The allocator here moves shards directly towards its balanced target, so
//! the desired assignment is the current one: every assigned copy is
//! "desired" and there are no undesired allocations.

use super::nodes::fetch_disk;
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::health::{self, HealthInputs, HealthStatus};
use elasticsearch_cluster::state::{ClusterState, ShardState};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

fn stat_block(values: &[f64]) -> Value {
    let n = values.len().max(1) as f64;
    let total: f64 = values.iter().sum();
    let avg = total / n;
    let var = values.iter().map(|v| (v - avg) * (v - avg)).sum::<f64>() / n;
    json!({
        "total": total,
        "min": if values.is_empty() { 0.0 } else { values.iter().cloned().fold(f64::INFINITY, f64::min) },
        "max": values.iter().cloned().fold(0.0, f64::max),
        "average": avg,
        "std_dev": var.sqrt(),
    })
}

fn tier_preference(st: &ClusterState, index: &str) -> Vec<String> {
    st.metadata
        .indices
        .get(index)
        .and_then(|m| m.setting("index.routing.allocation.include._tier_preference"))
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_else(|| vec!["data_content".into()])
}

pub async fn desired_balance(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if req.method() == axum::http::Method::DELETE {
        return RestResp::new(200,vec![],vec![]);
    }
    let st = node.state();
    let stats = match super::gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let size_of = |c: &elasticsearch_cluster::state::ShardRouting| c.allocation_id().and_then(|a| stats.get(a)).map(|s| s.store_bytes).unwrap_or(0);

    let mut routing = Map::new();
    let (mut total_alloc, mut unassigned_total) = (0u64, 0u64);
    for (name, irt) in &st.routing_table.indices {
        let tiers = tier_preference(&st, name);
        let mut shards = Map::new();
        for t in &irt.shards {
            let current: Vec<Value> = t
                .copies
                .iter()
                .filter(|c| c.node_id.is_some())
                .map(|c| {
                    json!({
                        "state": match c.state { ShardState::Started => "STARTED", ShardState::Initializing => "INITIALIZING", ShardState::Relocating => "RELOCATING", ShardState::Unassigned => "UNASSIGNED" },
                        "primary": c.primary,
                        "node": c.node_id,
                        "node_is_desired": true,
                        "relocating_node": c.relocating_node_id,
                        "relocating_node_is_desired": false,
                        "shard_id": c.shard,
                        "index": c.index,
                        "forecast_write_load": Value::Null,
                        "forecast_shard_size_in_bytes": Value::Null,
                        "tier_preference": tiers,
                    })
                })
                .collect();
            let node_ids: Vec<String> = t.copies.iter().filter_map(|c| c.node_id.clone()).collect();
            let unassigned = t.copies.iter().filter(|c| c.node_id.is_none()).count() as u64;
            total_alloc += node_ids.len() as u64;
            unassigned_total += unassigned;
            shards.insert(
                t.shard.to_string(),
                json!({"current": current, "desired": {"node_ids": node_ids, "total": t.copies.len(), "unassigned": unassigned, "ignored": 0}}),
            );
        }
        routing.insert(name.clone(), Value::Object(shards));
    }

    // Per-node balance stats.
    let data: Vec<_> = st.data_nodes().cloned().collect();
    // ES9 WeightFunction.calculateNodeWeight: normalize all four balance
    // factors, then compare shard count and forecast disk to cluster averages.
    // This allocator currently has no write-load forecasts (reported as zero).
    let factor = |key: &str, default: f32| {
        st.metadata.transient_settings.get(key).or_else(|| st.metadata.persistent_settings.get(key)).or_else(|| node.config.settings.get(key))
            .and_then(|v| v.as_f64().map(|n| n as f32).or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(default)
    };
    let shard_factor = factor("cluster.routing.allocation.balance.shard", 0.45);
    let index_factor = factor("cluster.routing.allocation.balance.index", 0.55);
    let write_factor = factor("cluster.routing.allocation.balance.write_load", 10.0);
    let disk_factor = factor("cluster.routing.allocation.balance.disk_usage", 2e-11);
    let factor_sum = shard_factor + index_factor + write_factor + disk_factor;
    let total_shards: u64 = st.metadata.indices.values().map(|i| i.number_of_shards as u64 * (i.number_of_replicas as u64 + 1)).sum();
    let mut forecast_sizes: BTreeMap<(String, u32), u64> = BTreeMap::new();
    for copy in st.routing_table.all_shards() {
        let entry = forecast_sizes.entry((copy.index.clone(), copy.shard)).or_default();
        *entry = (*entry).max(size_of(copy));
    }
    let total_disk: u64 = forecast_sizes.iter().map(|((index, _), size)| size * st.metadata.indices.get(index).map(|i| i.number_of_replicas as u64 + 1).unwrap_or(1)).sum();
    let n_nodes = data.len().max(1) as f32;
    let disks = futures::future::join_all(data.iter().map(|n| fetch_disk(node, &n.id))).await;
    let mut nodes = Map::new();
    let mut tier_vals: BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut cluster_info_nodes = Map::new();
    for (n, d) in data.iter().zip(disks) {
        let copies: Vec<_> = st.routing_table.shards_on_node(&n.id).collect();
        let disk: u64 = copies.iter().map(|c| size_of(c)).sum();
        let node_weight = if factor_sum > 0.0 {
            (shard_factor / factor_sum) * (copies.len() as f32 - total_shards as f32 / n_nodes)
                + (disk_factor / factor_sum) * (disk as f32 - total_disk as f32 / n_nodes)
        } else { 0.0 };
        nodes.insert(
            n.name.clone(),
            json!({
                "node_id": n.id,
                "roles": n.roles.iter().map(|r| r.name()).collect::<Vec<_>>(),
                "shard_count": copies.len(),
                "undesired_shard_allocation_count": 0,
                "forecast_write_load": 0.0,
                "forecast_disk_usage_bytes": disk,
                "actual_disk_usage_bytes": disk,
                "node_weight": node_weight,
            }),
        );
        for r in n.roles.iter().filter(|r| r.is_data()) {
            let name = r.name();
            let name = if name == "data" { "data_content" } else { name };
            let e = tier_vals.entry(name.to_string()).or_default();
            e.0.push(copies.len() as f64);
            e.1.push(0.0);
            e.2.push(disk as f64);
            e.3.push(0.0);
        }
        if let Ok(Some(d)) = d {
            let du = json!({"path": d.path, "total_bytes": d.total_bytes, "used_bytes": d.total_bytes.saturating_sub(d.free_bytes), "free_bytes": d.free_bytes, "free_disk_percent": if d.total_bytes == 0 { 0.0 } else { 100.0 * d.free_bytes as f64 / d.total_bytes as f64 }, "used_disk_percent": if d.total_bytes == 0 { 0.0 } else { 100.0 * d.total_bytes.saturating_sub(d.free_bytes) as f64 / d.total_bytes as f64 }});
            cluster_info_nodes.insert(n.id.clone(), json!({"node_name": n.name, "least_available": du, "most_available": du}));
        }
    }
    let tiers: Map<String, Value> = tier_vals
        .into_iter()
        .map(|(k, (sc, wl, du, und))| {
            (k, json!({"shard_count": stat_block(&sc), "undesired_shard_allocation_count": stat_block(&und), "forecast_write_load": stat_block(&wl), "forecast_disk_usage": stat_block(&du), "actual_disk_usage": stat_block(&du)}))
        })
        .collect();
    let shard_sizes: Map<String, Value> = st
        .routing_table
        .all_shards()
        .filter(|c| c.node_id.is_some())
        .map(|c| (format!("[{}][{}][{}]", c.index, c.shard, if c.primary { "p" } else { "r" }), json!(size_of(c))))
        .collect();
    let body = json!({
        "stats": {
            "computation_converged_index": st.version,
            "computation_active": false,
            "computation_submitted": st.version,
            "computation_executed": st.version,
            "computation_converged": st.version,
            "computation_iterations": st.version,
            "computed_shard_movements": 0,
            "computation_time_in_millis": 0,
            "reconciliation_time_in_millis": 0,
            "unassigned_shards": unassigned_total,
            "total_allocations": total_alloc,
            "undesired_allocations": 0,
            "undesired_allocations_ratio": 0.0,
        },
        "cluster_balance_stats": {
            "shard_count": st.routing_table.all_shards().filter(|c| c.node_id.is_some()).count(),
            "undesired_shard_allocation_count": 0,
            "tiers": tiers,
            "nodes": nodes,
        },
        "routing_table": routing,
        "cluster_info": {
            "nodes": cluster_info_nodes,
            "shard_sizes": shard_sizes,
            "shard_data_set_sizes": {},
            "shard_paths": {},
            "reserved_sizes": [],
        },
    });
    RestResp::json(200, &body)
}

pub fn prevalidate_node_removal(node: &ClusterNode, req: &RestReq) -> RestResp {
    let p = req.params();
    let given: Vec<(&str, &str)> = [("names", "names"), ("ids", "IDs"), ("external_ids", "external IDs")]
        .into_iter()
        .filter(|(k, _)| p.get(*k).map(|v| !v.is_empty()).unwrap_or(false))
        .collect();
    if given.is_empty() {
        return RestResp::error(400, "action_request_validation_exception", "Validation Failed: 1: request must contain one of the parameters 'names', 'ids', or 'external_ids';");
    }
    if given.len() > 1 {
        return RestResp::error(400, "action_request_validation_exception", "Validation Failed: 1: request must contain only one of the parameters 'names', 'ids', or 'external_ids';");
    }
    let (key, label) = given[0];
    let st = node.state();
    let wanted: Vec<String> = p[key].split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let mut found = Vec::new();
    let mut missing = Vec::new();
    for w in &wanted {
        let hit = st.nodes.values().find(|n| match key {
            "names" => &n.name == w,
            "ids" => &n.id == w,
            _ => n.attributes.get("external_id").map(|e| e == w).unwrap_or(false) || &n.name == w,
        });
        match hit {
            Some(n) => found.push(n.clone()),
            None => missing.push(w.clone()),
        }
    }
    if !missing.is_empty() {
        return RestResp::error(404, "resource_not_found_exception", &format!("could not resolve node {label} [{}]", missing.join(", ")));
    }
    // Removing nodes is unsafe if a red index has a copy only on them.
    let h = health::cluster_health(&st, None, HealthInputs::default());
    let red: Vec<&str> = h.indices.iter().filter(|i| i.status == HealthStatus::Red).map(|i| i.index.as_str()).collect();
    let mut overall_safe = true;
    let mut results = Vec::new();
    for n in &found {
        let blocking = red.iter().any(|idx| st.routing_table.shards_on_node(&n.id).any(|c| &c.index == idx));
        let safe = red.is_empty() || !blocking;
        overall_safe &= safe;
        let (reason, msg) = if red.is_empty() {
            ("no_problems", String::new())
        } else if safe {
            ("no_red_shards_on_node", String::new())
        } else {
            ("red_shards_on_node", format!("node [{}] contains copies of the following red shards: {:?}", n.id, red))
        };
        results.push(json!({"name": n.name, "id": n.id, "external_id": n.name, "result": {"is_safe": safe, "reason": reason, "message": msg}}));
    }
    let msg = if overall_safe { "" } else { "cluster health is RED" };
    RestResp::json(200, &json!({"is_safe": overall_safe, "message": msg, "nodes": results}))
}
