//! `GET _cluster/allocation/explain` (ES `ClusterAllocationExplanation`).
//!
//! The explanation strings are the `AllocateUnassignedDecision` /
//! `MoveDecision#getExplanation` texts of ES 8.0–8.5 (later 8.x releases
//! replaced them with longer prose; the decision values and the response
//! shape are unchanged).

use super::ApiError;
use super::balancer::{NodeResult, Outcome};
use super::context::{Allocation, DebugMode, is_relocation_target};
use super::deciders::DecisionType;
use super::settings::{fmt_date_millis, fmt_time_millis};
use crate::state::*;
use serde_json::{Map, Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplainOptions {
    pub include_yes_decisions: bool,
    pub include_disk_info: bool,
}

const NO_SHARD_SPECIFIED: &str = "No shard was specified in the explain API request, so this response explains a randomly chosen unassigned shard. There may be other unassigned shards in this cluster which cannot be assigned for different reasons. It may not be possible to assign this shard until one of the other shards is assigned correctly. To explain the allocation of other shards (whether assigned or unassigned) you must specify the target shard in the request to this API. See https://www.elastic.co/guide/en/elasticsearch/reference/8.15/cluster-allocation-explain.html for more information.";

const NO_UNASSIGNED: &str = "No shard was specified in the request which means the response should explain a randomly-chosen unassigned shard, but there are no unassigned shards in this cluster. To explain the allocation of an assigned shard you must specify the target shard in the request. See https://www.elastic.co/guide/en/elasticsearch/reference/8.15/cluster-allocation-explain.html for more information.";

fn bad(reason: impl Into<String>) -> ApiError {
    (400, "illegal_argument_exception".into(), reason.into())
}

pub fn resolve_node<'a>(state: &'a ClusterState, id_or_name: &str) -> Option<&'a DiscoveryNode> {
    state.nodes.get(id_or_name).or_else(|| state.nodes.values().find(|n| n.name == id_or_name))
}

/// `UnassignedInfo.AllocationStatus` → `AllocationDecision` value.
fn last_status(s: &str) -> &str {
    match s {
        "deciders_no" => "no",
        "deciders_throttled" => "throttled",
        "fetching_shard_data" | "awaiting_info" => "awaiting_info",
        "delayed_allocation" => "allocation_delayed",
        other => other,
    }
}

fn roles(n: &DiscoveryNode) -> Value {
    Value::Array(n.roles.iter().map(|r| Value::String(r.name().into())).collect())
}

fn attrs(n: &DiscoveryNode) -> Value {
    Value::Object(n.attributes.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect())
}

/// `discoveryNodeToXContent(node, outerObjectWritten = true)`.
fn node_outer(n: &DiscoveryNode) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("id".into(), json!(n.id));
    m.insert("name".into(), json!(n.name));
    m.insert("transport_address".into(), json!(n.transport_address));
    m.insert("roles".into(), roles(n));
    if !n.attributes.is_empty() {
        m.insert("attributes".into(), attrs(n));
    }
    m
}

fn node_results(rows: &[NodeResult]) -> Value {
    Value::Array(
        rows.iter()
            .map(|r| {
                let mut m = Map::new();
                m.insert("node_id".into(), json!(r.node.id));
                m.insert("node_name".into(), json!(r.node.name));
                m.insert("transport_address".into(), json!(r.node.transport_address));
                m.insert("roles".into(), roles(&r.node));
                m.insert("node_attributes".into(), attrs(&r.node));
                m.insert("node_decision".into(), json!(r.node_decision));
                if let Some(s) = &r.store {
                    m.insert("store".into(), s.clone());
                }
                if let Some(w) = r.weight_ranking {
                    m.insert("weight_ranking".into(), json!(w));
                }
                if !r.decision.singles.is_empty() {
                    m.insert("deciders".into(), Value::Array(r.decision.to_json()));
                }
                Value::Object(m)
            })
            .collect(),
    )
}

pub fn explain(mut a: Allocation, req: &Value, opts: ExplainOptions) -> Result<Value, ApiError> {
    a.debug = if opts.include_yes_decisions { DebugMode::On } else { DebugMode::ExcludeYes };
    let get = |k: &str| req.get(k).filter(|v| !v.is_null());
    let (index, shard, primary, current) = (get("index"), get("shard"), get("primary"), get("current_node"));
    let random = index.is_none() && shard.is_none() && primary.is_none() && current.is_none();
    let copy: ShardRouting = if random {
        let all: Vec<&ShardRouting> = a.state.routing_table.all_shards().filter(|c| c.state == ShardState::Unassigned).collect();
        all.iter().find(|c| c.primary).or_else(|| all.first()).map(|c| (*c).clone()).ok_or_else(|| bad(NO_UNASSIGNED))?
    } else {
        let mut errs = Vec::new();
        if index.is_none() {
            errs.push("index must be specified");
        }
        if shard.is_none() {
            errs.push("shard must be specified");
        }
        if primary.is_none() {
            errs.push("primary must be specified");
        }
        if !errs.is_empty() {
            let msg: String = errs.iter().enumerate().map(|(i, e)| format!("{}: {e};", i + 1)).collect();
            return Err((400, "action_request_validation_exception".into(), format!("Validation Failed: {msg}")));
        }
        let index = index.and_then(Value::as_str).ok_or_else(|| bad("[index] must be a string"))?;
        let shard = shard.and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).ok_or_else(|| bad("[shard] must be a non-negative integer"))? as u32;
        let primary = primary.and_then(|v| v.as_bool().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).ok_or_else(|| bad("[primary] must be a boolean"))?;
        let current = match current.map(|v| v.as_str().unwrap_or_default().to_string()) {
            None => None,
            Some(c) => Some(resolve_node(&a.state, &c).map(|n| n.id.clone()).ok_or_else(|| bad(format!("failed to resolve [{c}], no matching nodes")))?),
        };
        if !a.state.metadata.indices.contains_key(index) {
            return Err((404, "index_not_found_exception".into(), format!("no such index [{index}]")));
        }
        let t = a.table(index, shard).ok_or_else(|| (404, "shard_not_found_exception".into(), "no such shard".to_string()))?;
        let cands: Vec<&ShardRouting> = t.copies.iter().filter(|c| c.primary == primary && !is_relocation_target(c)).filter(|c| current.is_none() || c.node_id == current).collect();
        let found = if primary || current.is_some() { cands.first() } else { cands.iter().find(|c| c.state == ShardState::Unassigned).or_else(|| cands.first()) };
        found.map(|c| (*c).clone()).ok_or_else(|| {
            let mut r = format!("ClusterAllocationExplainRequest[index={index},shard={shard},primary?={primary}");
            if let Some(c) = &current {
                r.push_str(&format!(",currentNode={c}"));
            }
            bad(format!("unable to find any shards to explain [{r},includeYesDecisions?={}]] in the routing table", opts.include_yes_decisions))
        })?
    };

    let mut out = Map::new();
    if random {
        out.insert("note".into(), json!(NO_SHARD_SPECIFIED));
    }
    out.insert("index".into(), json!(copy.index));
    out.insert("shard".into(), json!(copy.shard));
    out.insert("primary".into(), json!(copy.primary));
    out.insert("current_state".into(), json!(format!("{:?}", copy.state).to_lowercase()));
    if let Some(u) = &copy.unassigned_info {
        let mut m = Map::new();
        m.insert("reason".into(), json!(super::deciders::reason_name(u.reason)));
        m.insert("at".into(), json!(fmt_date_millis(u.at_millis)));
        if u.failed_allocations > 0 {
            m.insert("failed_allocation_attempts".into(), json!(u.failed_allocations));
        }
        if let Some(d) = &u.details {
            m.insert("details".into(), json!(d));
        }
        m.insert("last_allocation_status".into(), json!(last_status(&u.last_allocation_status)));
        out.insert("unassigned_info".into(), Value::Object(m));
    }
    let current_node = copy.node_id.as_ref().and_then(|n| a.node(n)).cloned();
    let move_decision = if copy.state == ShardState::Started { Some(a.decide_move(&copy, true)) } else { None };
    if let Some(n) = &current_node {
        let mut m = node_outer(n);
        if let Some(md) = &move_decision {
            if md.current_ranking > 0 {
                m.insert("weight_ranking".into(), json!(md.current_ranking));
            }
        }
        out.insert("current_node".into(), Value::Object(m));
    }
    if opts.include_disk_info {
        out.insert("cluster_info".into(), a.info.to_json(&a.state));
    }

    match copy.state {
        ShardState::Unassigned => {
            let d = a.decide_unassigned(&copy, true);
            let any_yes = d.nodes.iter().any(|r| r.decision.kind == DecisionType::Yes);
            let explanation = match &d.outcome {
                Outcome::Yes => "can allocate the shard".to_string(),
                Outcome::Throttled => "allocation temporarily throttled".to_string(),
                Outcome::AwaitingInfo => "cannot allocate because information about existing shard data is still being retrieved from some of the nodes".to_string(),
                Outcome::NoValidCopy { found_stale: true } => "cannot allocate because all found copies of the shard are either stale or corrupt".to_string(),
                Outcome::NoValidCopy { found_stale: false } => "cannot allocate because a previous copy of the primary shard existed but can no longer be found on the nodes in the cluster".to_string(),
                Outcome::Delayed { remaining, .. } => format!(
                    "cannot allocate because the cluster is still waiting {} for the departed node holding a replica to rejoin{}",
                    fmt_time_millis(*remaining),
                    if any_yes { ", despite being allowed to allocate the shard to at least one other node" } else { "" }
                ),
                Outcome::No if d.gateway_primary => "cannot allocate because allocation is not permitted to any of the nodes that hold an in-sync shard copy".to_string(),
                Outcome::No => "cannot allocate because allocation is not permitted to any of the nodes".to_string(),
            };
            out.insert("can_allocate".into(), json!(d.outcome.allocation_decision()));
            out.insert("allocate_explanation".into(), json!(explanation));
            if let Some(t) = d.target.as_ref().and_then(|t| a.node(t)) {
                out.insert("target_node".into(), Value::Object(node_outer(t)));
            }
            if let Some(aid) = &d.allocation_id {
                out.insert("allocation_id".into(), json!(aid));
            }
            if let Outcome::Delayed { remaining, configured } = d.outcome {
                out.insert("configured_delay_in_millis".into(), json!(configured));
                out.insert("remaining_delay_in_millis".into(), json!(remaining));
            }
            if !d.nodes.is_empty() {
                out.insert("node_allocation_decisions".into(), node_results(&d.nodes));
            }
        }
        ShardState::Started => {
            let md = move_decision.unwrap_or_else(|| a.decide_move(&copy, true));
            if let Some(t) = md.target.as_ref().and_then(|t| a.node(t)) {
                out.insert("target_node".into(), Value::Object(node_outer(t)));
            }
            let can_remain = md.can_remain.kind != DecisionType::No;
            out.insert("can_remain_on_current_node".into(), json!(if can_remain { "yes" } else { "no" }));
            if !can_remain && !md.can_remain.singles.is_empty() {
                out.insert("can_remain_decisions".into(), Value::Array(md.can_remain.to_json()));
            }
            let any_yes = md.nodes.iter().any(|r| r.decision.kind == DecisionType::Yes);
            if let Some(rb) = &md.rebalance {
                out.insert("can_rebalance_cluster".into(), json!(rb.kind.allocation_decision()));
                if rb.kind != DecisionType::Yes && !rb.singles.is_empty() {
                    out.insert("can_rebalance_cluster_decisions".into(), Value::Array(rb.to_json()));
                }
                let pending = a.stores.has_pending();
                let decision = if pending { "awaiting_info" } else if rb.kind != DecisionType::Yes { rb.kind.allocation_decision() } else { md.best.allocation_decision() };
                let explanation = if pending {
                    "cannot rebalance as information about existing copies of this shard in the cluster is still being gathered".to_string()
                } else {
                    match rb.kind {
                        DecisionType::No => format!("rebalancing is not allowed{}", if any_yes { ", even though there is at least one node on which the shard can be allocated" } else { "" }),
                        DecisionType::Throttle => "rebalancing is throttled".to_string(),
                        DecisionType::Yes => match (&md.target, md.best) {
                            (Some(_), DecisionType::Throttle) => "shard rebalancing throttled".to_string(),
                            (Some(_), _) => "can rebalance shard".to_string(),
                            (None, _) => "cannot rebalance as no target node exists that can both allocate this shard and improve the cluster balance".to_string(),
                        },
                    }
                };
                out.insert("can_rebalance_to_other_node".into(), json!(decision));
                out.insert("rebalance_explanation".into(), json!(explanation));
            } else {
                let decision = md.best.allocation_decision();
                out.insert("can_move_to_other_node".into(), json!(if md.target.is_some() { "yes" } else { "no" }));
                let explanation = match (md.target.is_some(), decision) {
                    (true, _) => "shard cannot remain on this node and is force-moved to another node",
                    (false, "throttled") => "shard cannot remain on this node but is throttled on moving to another node",
                    _ => "cannot move shard to another node, even though it is not allowed to remain on its current node",
                };
                out.insert("move_explanation".into(), json!(explanation));
            }
            if !md.nodes.is_empty() {
                out.insert("node_allocation_decisions".into(), node_results(&md.nodes));
            }
        }
        ShardState::Relocating => {
            let from = current_node.as_ref().map(|n| n.name.clone()).unwrap_or_default();
            let to = copy.relocating_node_id.as_ref().and_then(|n| a.node(n)).map(|n| n.name.clone()).unwrap_or_default();
            out.insert("explanation".into(), json!(format!("the shard is in the process of relocating from node [{from}] to node [{to}], wait until relocation has completed")));
        }
        ShardState::Initializing => {
            let on = current_node.as_ref().map(|n| n.name.clone()).unwrap_or_default();
            out.insert("explanation".into(), json!(format!("the shard is in the process of initializing on node [{on}], wait until initialization has completed")));
        }
    }
    Ok(Value::Object(out))
}
