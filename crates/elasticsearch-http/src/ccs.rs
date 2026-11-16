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

/// [`resolve`] keeping connected remote clusters out of the local targets
/// without failing (a retriever's plan phase: its child searches read the
/// remote clusters through the cluster's cross-cluster search).
pub fn resolve_keeping_remotes(node: &Node, expr: &str, opts: ResolveOpts) -> Result<Plan> {
    let (targets, plan) = remote::resolve_parts(node, expr, opts)?;
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

/// `_cluster/stats` `ccs`: this node's CCS usage. The cluster layer adds
/// `clusters` (each connected remote's own stats) for `include_remotes`;
/// this engine has no remote connections of its own.
pub fn stats(node: &Node, _include_remotes: bool, _local_stats: &Value) -> Value {
    json!({"_search": node.ccs_stats.lock().json()})
}
