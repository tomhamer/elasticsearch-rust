//! `POST _cluster/reroute` commands (ES `AllocationCommands`): `move`,
//! `cancel`, `allocate_replica`, `allocate_stale_primary`,
//! `allocate_empty_primary`, with ES's parse errors, rejection messages and
//! `explanations` shape. Commands run with all deciders recorded and
//! `enable` ignored; afterwards a normal reroute runs (ES does the same).
//!
//! `dry_run` does not change the computation: the caller simply does not
//! publish the returned state.

use super::ApiError;
use super::context::{Allocation, DebugMode, is_relocation_target, shard_id_str};
use super::deciders::{Decision, DecisionType, Single, node_str, reason_name, state_name};
use super::explain::resolve_node;
use super::lifecycle::reset_failed_counters;
use crate::state::*;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RerouteFlags {
    pub dry_run: bool,
    pub explain: bool,
    pub retry_failed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RerouteCommandsResult {
    pub state: ClusterState,
    /// `explanations` array of the response (filled when `explain`).
    pub explanations: Vec<Value>,
}

#[derive(Clone, Debug)]
enum Command {
    Move { index: String, shard: u32, from: String, to: String },
    Cancel { index: String, shard: u32, node: String, allow_primary: bool },
    AllocateReplica { index: String, shard: u32, node: String },
    AllocateStalePrimary { index: String, shard: u32, node: String, accept_data_loss: bool },
    AllocateEmptyPrimary { index: String, shard: u32, node: String, accept_data_loss: bool },
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Move { .. } => "move",
            Command::Cancel { .. } => "cancel",
            Command::AllocateReplica { .. } => "allocate_replica",
            Command::AllocateStalePrimary { .. } => "allocate_stale_primary",
            Command::AllocateEmptyPrimary { .. } => "allocate_empty_primary",
        }
    }

    fn parameters(&self) -> Value {
        match self {
            Command::Move { index, shard, from, to } => json!({"index": index, "shard": shard, "from_node": from, "to_node": to}),
            Command::Cancel { index, shard, node, allow_primary } => json!({"index": index, "shard": shard, "node": node, "allow_primary": allow_primary}),
            Command::AllocateReplica { index, shard, node } => json!({"index": index, "shard": shard, "node": node}),
            Command::AllocateStalePrimary { index, shard, node, accept_data_loss } | Command::AllocateEmptyPrimary { index, shard, node, accept_data_loss } => {
                json!({"index": index, "shard": shard, "node": node, "accept_data_loss": accept_data_loss})
            }
        }
    }
}

fn parse_err(msg: String) -> ApiError {
    (400, "parse_exception".into(), msg)
}

fn parse_command(name: &str, body: &Value) -> Result<Command, ApiError> {
    let obj = body.as_object().ok_or_else(|| parse_err(format!("[{name}] command must be an object")))?;
    let allowed: &[&str] = match name {
        "move" => &["index", "shard", "from_node", "to_node"],
        "cancel" => &["index", "shard", "node", "allow_primary"],
        "allocate_replica" => &["index", "shard", "node"],
        "allocate_stale_primary" | "allocate_empty_primary" => &["index", "shard", "node", "accept_data_loss"],
        _ => return Err((400, "x_content_parse_exception".into(), format!("unknown allocation command [{name}]"))),
    };
    for k in obj.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(parse_err(format!("[{name}] command does not support field [{k}]")));
        }
    }
    let s = |k: &str| -> Result<String, ApiError> { obj.get(k).and_then(|v| v.as_str().map(String::from)).ok_or_else(|| parse_err(format!("[{name}] command missing the {k} parameter"))) };
    let shard = || -> Result<u32, ApiError> {
        obj.get("shard").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|x| x.parse().ok()))).map(|v| v as u32).ok_or_else(|| parse_err(format!("[{name}] command missing the shard parameter")))
    };
    let b = |k: &str| obj.get(k).and_then(|v| v.as_bool().or_else(|| v.as_str().and_then(|x| x.parse().ok()))).unwrap_or(false);
    let index = s("index")?;
    let shard = shard()?;
    Ok(match name {
        "move" => Command::Move { index, shard, from: s("from_node")?, to: s("to_node")? },
        "cancel" => Command::Cancel { index, shard, node: s("node")?, allow_primary: b("allow_primary") },
        "allocate_replica" => Command::AllocateReplica { index, shard, node: s("node")? },
        "allocate_stale_primary" => Command::AllocateStalePrimary { index, shard, node: s("node")?, accept_data_loss: b("accept_data_loss") },
        _ => Command::AllocateEmptyPrimary { index, shard, node: s("node")?, accept_data_loss: b("accept_data_loss") },
    })
}

fn parse_commands(body: &Value) -> Result<Vec<Command>, ApiError> {
    let Some(cmds) = body.get("commands") else { return Ok(vec![]) };
    let arr = cmds.as_array().ok_or_else(|| parse_err("[commands] must be an array".into()))?;
    let mut out = Vec::new();
    for c in arr {
        let obj = c.as_object().filter(|o| o.len() == 1).ok_or_else(|| parse_err("each command must be an object with a single command name".into()))?;
        for (name, body) in obj {
            out.push(parse_command(name, body)?);
        }
    }
    Ok(out)
}

fn explanation(cmd: &Command, d: &Decision) -> Value {
    json!({"command": cmd.name(), "parameters": cmd.parameters(), "decisions": d.to_json()})
}

/// Rejection: an explanation in explain mode, else an error.
fn reject(cmd: &Command, explain: bool, err: ApiError) -> Result<Value, ApiError> {
    if explain {
        Ok(explanation(cmd, &Decision::of(Single::new(cmd.name(), DecisionType::No, err.2))))
    } else {
        Err(err)
    }
}

fn bad(msg: String) -> ApiError {
    (400, "illegal_argument_exception".into(), msg)
}

pub fn execute(mut a: Allocation, body: &Value, flags: RerouteFlags) -> Result<(ClusterState, Vec<Value>), ApiError> {
    let commands = parse_commands(body)?;
    a.debug = DebugMode::On;
    a.ignore_disable = true;
    if flags.retry_failed {
        reset_failed_counters(&mut a.state, a.now);
        a.refresh();
    }
    let mut explanations = Vec::new();
    for cmd in &commands {
        let e = run(&mut a, cmd, flags.explain)?;
        explanations.push(e);
    }
    a.ignore_disable = false;
    a.debug = DebugMode::Off;
    a.run_reroute();
    Ok((a.state, if flags.explain { explanations } else { vec![] }))
}

fn resolve(a: &Allocation, node: &str) -> Result<DiscoveryNode, ApiError> {
    resolve_node(&a.state, node).cloned().ok_or_else(|| bad(format!("failed to resolve [{node}], no matching nodes")))
}

fn shard_table(a: &Allocation, index: &str, shard: u32) -> Result<IndexShardRoutingTable, ApiError> {
    if !a.state.metadata.indices.contains_key(index) {
        return Err((404, "index_not_found_exception".into(), format!("no such index [{index}]")));
    }
    a.table(index, shard).cloned().ok_or_else(|| (404, "shard_not_found_exception".into(), "no such shard".into()))
}

fn run(a: &mut Allocation, cmd: &Command, explain: bool) -> Result<Value, ApiError> {
    match cmd {
        Command::Move { index, shard, from, to } => {
            let sid = shard_id_str(index, *shard);
            let from_n = match resolve(a, from) {
                Ok(n) => n,
                Err(e) => return reject(cmd, explain, e),
            };
            let to_n = match resolve(a, to) {
                Ok(n) => n,
                Err(e) => return reject(cmd, explain, e),
            };
            let found = a.state.routing_table.shards_on_node(&from_n.id).find(|c| &c.index == index && c.shard == *shard && !is_relocation_target(c)).cloned();
            let Some(c) = found else {
                if explain {
                    return Ok(explanation(cmd, &Decision::of(Single::new("move_allocation_command", DecisionType::No, format!("shard {sid} not found")))));
                }
                return Err(bad(format!("[move_allocation] can't move {sid}, failed to find it on node {}", node_str(&from_n))));
            };
            if c.state != ShardState::Started {
                if explain {
                    return Ok(explanation(cmd, &Decision::of(Single::new("move_allocation_command", DecisionType::No, format!("shard {sid} has not been started")))));
                }
                return Err(bad(format!("[move_allocation] can't move {sid}, shard is not started (state = {}]", state_name(c.state))));
            }
            let d = a.can_allocate(&c, &to_n);
            if d.kind == DecisionType::No {
                if explain {
                    return Ok(explanation(cmd, &d));
                }
                return Err(bad(format!("[move_allocation] can't move {sid}, from {}, to {}, since its not allowed, reason: {}", node_str(&from_n), node_str(&to_n), d.describe())));
            }
            let aid = c.allocation_id().unwrap_or_default().to_string();
            a.relocate(index, *shard, &aid, &to_n.id);
            Ok(explanation(cmd, &d))
        }
        Command::Cancel { index, shard, node, allow_primary } => {
            let sid = shard_id_str(index, *shard);
            let n = match resolve(a, node) {
                Ok(n) => n,
                Err(e) => return reject(cmd, explain, e),
            };
            if !a.state.metadata.indices.contains_key(index) {
                return Err((404, "index_not_found_exception".into(), format!("no such index [{index}]")));
            }
            let found = a.state.routing_table.shards_on_node(&n.id).find(|c| &c.index == index && c.shard == *shard).cloned();
            let Some(c) = found else {
                if explain {
                    return Ok(explanation(cmd, &Decision::of(Single::new("cancel_allocation_command", DecisionType::No, format!("can't cancel {sid}, failed to find it on node {}", node_str(&n))))));
                }
                return Err(bad(format!("[cancel_allocation] can't cancel {sid}, failed to find it on node {}", node_str(&n))));
            };
            if c.primary && !allow_primary && !is_relocation_target(&c) {
                let st = format!("{:?}", c.state).to_lowercase();
                if explain {
                    return Ok(explanation(cmd, &Decision::of(Single::new("cancel_allocation_command", DecisionType::No, format!("can't cancel {sid} on node {}, shard is primary and {st}", node_str(&n))))));
                }
                return Err(bad(format!("[cancel_allocation] can't cancel {sid} on node {}, shard is primary and {st}", node_str(&n))));
            }
            let info = UnassignedInfo { reason: UnassignedReason::RerouteCancelled, at_millis: a.now, failed_allocations: 0, delayed: false, details: None, last_allocation_status: "no_attempt".into(), last_node: c.node_id.clone() };
            let aid = c.allocation_id().unwrap_or_default().to_string();
            a.fail_shard(index, *shard, &aid, info, true);
            Ok(explanation(cmd, &Decision::of(Single::new("cancel_allocation_command", DecisionType::Yes, format!("shard {sid} on node {} can be cancelled", node_str(&n))))))
        }
        Command::AllocateReplica { index, shard, node } => {
            let name = cmd.name();
            let n = match resolve(a, node) {
                Ok(n) => n,
                Err(e) => return reject(cmd, explain, e),
            };
            if !n.is_data() {
                return reject(cmd, explain, bad(format!("[{name}] allocation can only be done on data nodes, not [{node}]")));
            }
            let t = match shard_table(a, index, *shard) {
                Ok(t) => t,
                Err(e) => return reject(cmd, explain, e),
            };
            if t.copies.iter().find(|c| c.primary && !is_relocation_target(c)).is_none_or(|p| p.state == ShardState::Unassigned) {
                return reject(cmd, explain, bad(format!("[{name}] trying to allocate a replica shard [{index}][{shard}], while corresponding primary shard is still unassigned")));
            }
            let Some(pos) = t.copies.iter().position(|c| !c.primary && c.state == ShardState::Unassigned) else {
                return reject(cmd, explain, bad(format!("[{name}] all copies of [{index}][{shard}] are already assigned. Use the move allocation command instead")));
            };
            let c = t.copies[pos].clone();
            let d = a.can_allocate(&c, &n);
            if d.kind == DecisionType::No {
                if explain {
                    return Ok(explanation(cmd, &d));
                }
                return Err(bad(format!("[{name}] allocation of [{index}][{shard}] on node {} is not allowed, reason: {}", node_str(&n), d.describe())));
            }
            a.initialize(index, *shard, pos, &n.id, None, None, None);
            Ok(explanation(cmd, &d))
        }
        Command::AllocateStalePrimary { index, shard, node, accept_data_loss } | Command::AllocateEmptyPrimary { index, shard, node, accept_data_loss } => {
            let name = cmd.name();
            let empty = matches!(cmd, Command::AllocateEmptyPrimary { .. });
            if empty && !accept_data_loss {
                return reject(cmd, explain, bad(format!("[{name}] allocating an empty primary for [{index}][{shard}] can result in data loss. Please confirm by setting the accept_data_loss parameter to true")));
            }
            let n = match resolve(a, node) {
                Ok(n) => n,
                Err(e) => return reject(cmd, explain, e),
            };
            if !n.is_data() {
                return reject(cmd, explain, bad(format!("[{name}] allocation can only be done on data nodes, not [{node}]")));
            }
            let t = match shard_table(a, index, *shard) {
                Ok(t) => t,
                Err(e) => return reject(cmd, explain, e),
            };
            let Some(pos) = t.copies.iter().position(|c| c.primary && !is_relocation_target(c)) else {
                return reject(cmd, explain, bad(format!("[{name}] primary [{index}][{shard}] is already assigned")));
            };
            let p = t.copies[pos].clone();
            if p.state != ShardState::Unassigned {
                return reject(cmd, explain, bad(format!("[{name}] primary [{index}][{shard}] is already assigned")));
            }
            if !empty && !accept_data_loss {
                return reject(cmd, explain, bad(format!("[{name}] allocating an empty primary for [{index}][{shard}] can result in data loss. Please confirm by setting the accept_data_loss parameter to true").replace("an empty primary", "a stale primary")));
            }
            if !empty && !matches!(p.recovery_source, Some(RecoverySource::ExistingStore { .. })) {
                return reject(cmd, explain, bad(format!("[{name}] trying to allocate an existing primary shard [{index}][{shard}], while no such shard has ever been active")));
            }
            if t.copies.iter().any(|c| c.node_id.as_deref() == Some(n.id.as_str())) {
                return reject(cmd, explain, bad(format!("[{name}] a copy of [{index}][{shard}] is already allocated to node [{}]", n.id)));
            }
            let info = if empty {
                p.unassigned_info.as_ref().filter(|u| u.reason != UnassignedReason::ForcedEmptyPrimary).map(|u| UnassignedInfo {
                    reason: UnassignedReason::ForcedEmptyPrimary,
                    at_millis: a.now,
                    failed_allocations: 0,
                    delayed: false,
                    details: Some(format!("force empty allocation from previous reason {}, {}", reason_name(u.reason), u.details.clone().unwrap_or_else(|| "null".into()))),
                    last_allocation_status: "no_attempt".into(),
                    last_node: None,
                })
            } else {
                None
            };
            let source = if empty { RecoverySource::EmptyStore } else { RecoverySource::ExistingStore { bootstrap_new_history_uuid: true } };
            a.initialize(index, *shard, pos, &n.id, None, Some(source), info);
            Ok(explanation(cmd, &Decision::of(Single::new(&format!("{name} (allocation command)"), DecisionType::Yes, "ignore deciders"))))
        }
    }
}
