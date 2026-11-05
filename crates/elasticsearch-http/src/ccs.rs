//! CCS planning with explicit remote handling. No local index may stand in
//! for a remote cluster: optional remotes are reported as skipped and required
//! remotes fail, so a search never silently returns partial data. Each cluster
//! is designed to be self-contained; scale out by adding nodes to it.

use elasticsearch_engine::ccr::remote;
use elasticsearch_engine::error::Result;
use elasticsearch_engine::node::{Node, ResolveOpts};
use elasticsearch_engine::search::Target;
use serde_json::{json, Value};

pub struct Plan {
    pub targets: Vec<Target>,
    pub aliases: Vec<String>,
    pub skipped: Vec<String>,
    pub plan: remote::CcsPlan,
}

pub fn has_remote(expr: &str) -> bool {
    remote::has_remote_expression(expr)
}

/// The shared engine planner ([`remote::ccs_plan`]) plus the local targets.
pub fn resolve(node: &Node, expr: &str, opts: ResolveOpts) -> Result<Plan> {
    let (targets, plan) = remote::resolve_local_part(node, expr, opts)?;
    let plan = plan.unwrap_or_default();
    let aliases = plan.aliases();
    Ok(Plan { targets, skipped: aliases.clone(), aliases, plan })
}

/// `_clusters` with per-cluster `details`, as ES 8.10+ renders it.
pub fn finish(raw: String, plan: &Plan) -> Result<String> {
    let mut v: Value = serde_json::from_str(&raw)?;
    let local = plan.plan.local_expr().is_some().then(|| v.clone());
    v["_clusters"] = plan.plan.clusters(local.as_ref());
    Ok(serde_json::to_string(&v)?)
}

pub fn stats(node: &Node, include_remotes: bool, local_stats: &Value) -> Value {
    let mut out = json!({"_search": node.ccs_stats.lock().json()});
    if include_remotes {
        let info = remote::remote_info(node);
        let mut clusters = serde_json::Map::new();
        for alias in remote::remote_aliases(node) {
            if info[&alias]["connected"] != true { continue; }
            clusters.insert(alias.clone(), json!({
                "cluster_uuid": node.cluster_uuid, "mode": info[&alias]["mode"], "skip_unavailable": info[&alias]["skip_unavailable"],
                "version": elasticsearch_engine::node::ES_VERSION, "status": local_stats["status"],
                "nodes_count": local_stats["nodes"]["count"]["total"], "shards_count": local_stats["indices"]["shards"]["total"],
                "indices_count": local_stats["indices"]["count"], "indices_total_size_in_bytes": local_stats["indices"]["store"]["size_in_bytes"],
                "max_heap_in_bytes": local_stats["nodes"]["jvm"]["mem"]["heap_max_in_bytes"], "mem_total_in_bytes": local_stats["nodes"]["os"]["mem"]["total_in_bytes"]
            }));
        }
        if !clusters.is_empty() { out["clusters"] = Value::Object(clusters); }
    }
    out
}
