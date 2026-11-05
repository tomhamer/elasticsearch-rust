//! `GET _nodes/{id}/shutdown` shard-migration status (ES
//! `TransportGetShutdownStatusAction.shardMigrationStatus`).

use super::context::{Allocation, all_shutdowns};
use super::deciders::DecisionType;
use super::explain::{ExplainOptions, explain};
use crate::state::*;
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

/// ES `ShutdownShardMigrationStatus`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShardMigrationStatus {
    /// `NOT_STARTED`, `IN_PROGRESS`, `STALLED`, `COMPLETE`.
    pub status: &'static str,
    /// `(started, relocating, initializing)`; `None` renders no counts (ES -1).
    pub counts: Option<(u64, u64, u64)>,
    pub remaining: u64,
    pub explanation: Option<String>,
    /// `node_allocation_decision` (ES `ShardAllocationDecision`).
    pub decision: Option<Value>,
}

impl ShardMigrationStatus {
    fn simple(status: &'static str, remaining: u64, explanation: impl Into<String>, decision: Option<Value>) -> Self {
        ShardMigrationStatus { status, counts: None, remaining, explanation: Some(explanation.into()), decision }
    }

    fn counted(status: &'static str, started: u64, relocating: u64, initializing: u64, explanation: Option<String>, decision: Option<Value>) -> Self {
        ShardMigrationStatus { status, counts: Some((started, relocating, initializing)), remaining: started + relocating + initializing, explanation, decision }
    }

    /// ES `ShutdownShardMigrationStatus.toXContent`.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("status".into(), json!(self.status));
        if let Some((s, r, i)) = self.counts {
            m.insert("started_shards".into(), json!(s));
            m.insert("relocating_shards".into(), json!(r));
            m.insert("initializing_shards".into(), json!(i));
        }
        m.insert("shard_migrations_remaining".into(), json!(self.remaining));
        if let Some(e) = &self.explanation {
            m.insert("explanation".into(), json!(e));
        }
        if let Some(d) = &self.decision {
            m.insert("node_allocation_decision".into(), d.clone());
        }
        Value::Object(m)
    }
}

const DECISION_KEY: &str = "node_allocation_decision";

fn role(c: &ShardRouting) -> &'static str {
    if c.primary { "primary" } else { "replica" }
}

/// ES `hasShardCopyOnAnotherNode`: any STARTED copy of the shard (stateful
/// shards all share the DEFAULT role) on a node that is not shutting down.
fn has_copy_elsewhere(state: &ClusterState, c: &ShardRouting, shutting_down: &BTreeSet<String>) -> bool {
    state
        .routing_table
        .shard_table(&c.index, c.shard)
        .is_some_and(|t| t.copies.iter().any(|o| o.state == ShardState::Started && o.node_id.as_ref().is_some_and(|n| !shutting_down.contains(n))))
}

/// The `ShardAllocationDecision` part of an allocation explanation (the
/// explain response minus the `ClusterAllocationExplanation` header).
fn shard_decision(a: Allocation, c: &ShardRouting) -> Option<Value> {
    let mut req = json!({"index": c.index, "shard": c.shard, "primary": c.primary});
    if let Some(n) = &c.node_id {
        req["current_node"] = json!(n);
    }
    let mut v = explain(a, &req, ExplainOptions::default()).ok()?;
    if let Some(m) = v.as_object_mut() {
        for k in ["note", "index", "shard", "primary", "current_state", "unassigned_info", "current_node", "cluster_info"] {
            m.remove(k);
        }
    }
    Some(v)
}

/// `can_remain == NO` move-decision outcome: `yes` (target found),
/// `throttled`, or `no`.
fn move_outcome(a: &Allocation, c: &ShardRouting) -> Option<&'static str> {
    let md = a.decide_move(c, false);
    if md.can_remain.kind != DecisionType::No {
        return None;
    }
    Some(if md.target.is_some() {
        "yes"
    } else if md.best == DecisionType::Throttle {
        "throttled"
    } else {
        "no"
    })
}

pub(super) fn migration_status<'a>(mk: impl Fn() -> Allocation<'a>, node_id: &str, shutdown_type: &str, node_seen: bool) -> ShardMigrationStatus {
    if shutdown_type.eq_ignore_ascii_case("restart") {
        return ShardMigrationStatus::simple("COMPLETE", 0, "no shard relocation is necessary for a node restart", None);
    }
    let a = mk();
    let state = &a.state;
    if !state.nodes.contains_key(node_id) && !node_seen {
        return ShardMigrationStatus::simple("NOT_STARTED", 0, "node is not currently part of the cluster", None);
    }
    let shutting_down: BTreeSet<String> = all_shutdowns(&state.metadata).into_iter().map(|(id, _)| id).collect();

    let unassigned: Vec<ShardRouting> = state
        .routing_table
        .all_shards()
        .filter(|c| c.state == ShardState::Unassigned)
        .filter(|c| c.unassigned_info.as_ref().and_then(|u| u.last_node.as_deref()) == Some(node_id))
        .filter(|c| c.primary || !has_copy_elsewhere(state, c, &shutting_down))
        .cloned()
        .collect();
    if let Some(c) = unassigned.first() {
        let msg = format!("shard [{}] [{}] of index [{}] is unassigned, see [{DECISION_KEY}] for details or use the cluster allocation explain API", c.shard, role(c), c.index);
        return ShardMigrationStatus::simple("STALLED", unassigned.len() as u64, msg, shard_decision(mk(), c));
    }

    let on_node: Vec<ShardRouting> = state.routing_table.shards_on_node(node_id).cloned().collect();
    let count = |s: ShardState| on_node.iter().filter(|c| c.state == s).count() as u64;
    let (started, relocating, initializing) = (count(ShardState::Started), count(ShardState::Relocating), count(ShardState::Initializing));
    let total = started + relocating + initializing;
    if total == 0 {
        return ShardMigrationStatus::counted("COMPLETE", 0, 0, 0, None, None);
    }
    if relocating > 0 {
        return ShardMigrationStatus::counted("IN_PROGRESS", started, relocating, initializing, None, None);
    }
    if started == 0 {
        return ShardMigrationStatus::counted(
            "IN_PROGRESS",
            started,
            relocating,
            initializing,
            Some("all remaining shards are currently INITIALIZING and must finish before they can be moved off this node".into()),
            None,
        );
    }

    // Started shards that may not remain and have no YES target.
    let unmovable: Vec<(&ShardRouting, &'static str)> =
        on_node.iter().filter(|c| c.state == ShardState::Started).filter_map(|c| move_outcome(&a, c).map(|o| (c, o))).filter(|(_, o)| *o != "yes").collect();
    let mut ignored = 0u64;
    let mut stuck = None;
    for (c, o) in &unmovable {
        if *o == "throttled" {
            continue;
        }
        if has_copy_elsewhere(state, c, &shutting_down) {
            ignored += 1;
            continue;
        }
        if stuck.is_none() {
            stuck = Some(*c);
        }
    }
    let throttled: Vec<&ShardRouting> = unmovable.iter().filter(|(_, o)| *o == "throttled").map(|(c, _)| *c).collect();

    if total == ignored && stuck.is_none() {
        return ShardMigrationStatus::simple("COMPLETE", 0, format!("[{ignored}] shards cannot be moved away from this node but have at least one copy on another node in the cluster"), None);
    }
    if let Some(c) = stuck {
        let msg = format!("shard [{}] [{}] of index [{}] cannot move, see [{DECISION_KEY}] for details or use the cluster allocation explain API", c.shard, role(c), c.index);
        return ShardMigrationStatus::simple("STALLED", total, msg, shard_decision(mk(), c));
    }
    if initializing == 0 && throttled.len() as u64 == started {
        let c = throttled[0];
        let msg = format!("shard [{}] [{}] of index [{}] is waiting to be moved, see [{DECISION_KEY}] for details or use the cluster allocation explain API", c.shard, role(c), c.index);
        return ShardMigrationStatus::counted("IN_PROGRESS", started, relocating, initializing, Some(msg), shard_decision(mk(), c));
    }
    ShardMigrationStatus::counted("IN_PROGRESS", started, relocating, initializing, None, None)
}
