//! Allocation deciders, in ES's order (`ClusterModule.createAllocationDeciders`)
//! with ES's decider names and explanation strings.
//!
//! Each decider answers some of: `can_allocate(shard, node)`,
//! `can_remain(shard, node)`, `can_rebalance(shard)`, the cluster-wide
//! `can_rebalance()`, the index-level `can_allocate_index(index, node)` and
//! `should_auto_expand_to_node(index, node)`. A [`Decision`] aggregates the
//! single results: NO if any NO, else THROTTLE if any THROTTLE, else YES.
//! With [`DebugMode::Off`] evaluation stops at the first NO and YES results
//! are not recorded, exactly like ES outside of explain.

use super::context::{Allocation, DebugMode, all_shutdowns, is_relocation_target, parse_version, shard_id_str, shutdown_info};
use super::settings::{EnableAllocation, EnableRebalance, AllowRebalance, NodeFilters, format1, fmt_bytes, fmt_date_millis, index_i64, index_setting};
use crate::state::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecisionType {
    No,
    Throttle,
    Yes,
}

impl DecisionType {
    pub fn name(self) -> &'static str {
        match self {
            DecisionType::No => "NO",
            DecisionType::Throttle => "THROTTLE",
            DecisionType::Yes => "YES",
        }
    }
    /// `AllocationDecision.fromDecisionType` (lowercase, explain API).
    pub fn allocation_decision(self) -> &'static str {
        match self {
            DecisionType::No => "no",
            DecisionType::Throttle => "throttled",
            DecisionType::Yes => "yes",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Single {
    pub decider: String,
    pub kind: DecisionType,
    pub explanation: String,
}

impl Single {
    pub fn new(decider: &str, kind: DecisionType, explanation: impl Into<String>) -> Single {
        Single { decider: decider.to_string(), kind, explanation: explanation.into() }
    }
    pub fn to_json(&self) -> Value {
        json!({"decider": self.decider, "decision": self.kind.name(), "explanation": self.explanation})
    }
}

/// `Decision.Multi`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub kind: DecisionType,
    pub singles: Vec<Single>,
}

impl Decision {
    pub fn yes() -> Decision {
        Decision { kind: DecisionType::Yes, singles: vec![] }
    }
    pub fn of(single: Single) -> Decision {
        Decision { kind: single.kind, singles: vec![single] }
    }
    pub fn add(&mut self, s: Single, mode: DebugMode) {
        self.kind = self.kind.min(s.kind);
        if s.kind != DecisionType::Yes || mode == DebugMode::On {
            self.singles.push(s);
        }
    }
    pub fn merge(&mut self, other: Decision) {
        self.kind = self.kind.min(other.kind);
        self.singles.extend(other.singles);
    }
    pub fn to_json(&self) -> Vec<Value> {
        self.singles.iter().map(Single::to_json).collect()
    }
    /// `Decision.Multi#toString`: `[NO(explanation)][YES(...)]`.
    pub fn describe(&self) -> String {
        if self.singles.is_empty() {
            return format!("{}()", self.kind.name());
        }
        self.singles.iter().map(|s| format!("[{}({})]", s.kind.name(), s.explanation)).collect()
    }
}

type ShardNodeFn = fn(&Allocation, &ShardRouting, &DiscoveryNode) -> Option<Single>;
type ShardFn = fn(&Allocation, &ShardRouting) -> Option<Single>;
type GlobalFn = fn(&Allocation) -> Option<Single>;
type IndexNodeFn = fn(&Allocation, &IndexMetadata, &DiscoveryNode) -> Option<Single>;

const CAN_ALLOCATE: &[ShardNodeFn] = &[
    max_retry,
    replica_after_primary_active,
    enable_allocate,
    node_version,
    segment_format,
    restore_in_progress,
    node_shutdown_allocate,
    node_replacement_allocate,
    filter_allocate,
    same_shard,
    disk_allocate,
    throttling,
    shards_limit_allocate,
    awareness_allocate,
    data_tier,
    stateless_shard_role,
];

const CAN_REMAIN: &[ShardNodeFn] = &[node_shutdown_allocate, node_replacement_remain, filter_allocate, disk_remain, shards_limit_remain, awareness_remain, data_tier, stateless_shard_role];

const CAN_REBALANCE_SHARD: &[ShardFn] = &[rebalance_only_when_active, cluster_rebalance_shard, concurrent_rebalance_shard, enable_rebalance];

const CAN_REBALANCE: &[GlobalFn] = &[cluster_rebalance, concurrent_rebalance, enable_rebalance_global];

const CAN_ALLOCATE_INDEX: &[IndexNodeFn] = &[filter_index, data_tier_index];

const AUTO_EXPAND: &[IndexNodeFn] = &[filter_index, node_shutdown_auto_expand, node_replacement_auto_expand, data_tier_index];

impl Allocation<'_> {
    fn collect<T: Copy>(&self, fns: &[T], mut call: impl FnMut(T) -> Option<Single>) -> Decision {
        let mut d = Decision::yes();
        for f in fns {
            if let Some(s) = call(*f) {
                let no = s.kind == DecisionType::No;
                d.add(s, self.debug);
                if no && self.debug == DebugMode::Off {
                    break;
                }
            }
        }
        d
    }

    pub fn can_allocate(&self, shard: &ShardRouting, node: &DiscoveryNode) -> Decision {
        self.collect(CAN_ALLOCATE, |f| f(self, shard, node))
    }

    pub fn can_remain(&self, shard: &ShardRouting, node: &DiscoveryNode) -> Decision {
        self.collect(CAN_REMAIN, |f| f(self, shard, node))
    }

    pub fn can_rebalance_shard(&self, shard: &ShardRouting) -> Decision {
        // The target allocation carries the move time in its opaque ID so
        // ordinary count rebalance and a successor master respect cooldown.
        // Mandatory can-remain evacuations are intentionally unaffected.
        if let Some(start)=shard.allocation_id().and_then(|id|id.strip_prefix("capacity-")).and_then(|tail|tail.split_once('-')).and_then(|(time,_)|time.parse::<i64>().ok()) {
            if self.now.saturating_sub(start)<600_000{return Decision::of(Single::new("memory_pressure_cooldown",DecisionType::No,"recent admitted pressure relocation is cooling down"));}
        }
        self.collect(CAN_REBALANCE_SHARD, |f| f(self, shard))
    }

    pub fn can_rebalance(&self) -> Decision {
        self.collect(CAN_REBALANCE, |f| f(self))
    }

    /// May a rebalance move a shard onto `node`? Not while the node is under
    /// sustained memory pressure: the count balancer would refill the node a
    /// pressure relocation just relieved (the cooldown protects only the
    /// moved copy), the next pressure move would take another shard off it,
    /// and so on forever. Unassigned shards and mandatory moves are unaffected.
    pub fn can_rebalance_to(&self, node: &DiscoveryNode) -> Decision {
        if self.info.memory_pressure.contains(&node.id) {
            return Decision::of(Single::new("memory_pressure", DecisionType::No, format!("node [{}] is under sustained memory pressure, rebalancing does not move shards onto it", node.id)));
        }
        Decision::yes()
    }

    pub fn can_allocate_index(&self, meta: &IndexMetadata, node: &DiscoveryNode) -> Decision {
        self.collect(CAN_ALLOCATE_INDEX, |f| f(self, meta, node))
    }

    pub fn should_auto_expand_to_node(&self, meta: &IndexMetadata, node: &DiscoveryNode) -> Decision {
        self.collect(AUTO_EXPAND, |f| f(self, meta, node))
    }

    /// `canForceAllocatePrimary`: by default a NO becomes YES (the copy's data
    /// is on that node); `max_retry`, `node_shutdown`, `node_replacement` and
    /// `segment_format` and `stateless_shard_role` keep their answer (a stateless primary is never
    /// forced onto a search node holding a copy: ES's stateless shards
    /// allocator recovers primaries on index nodes only).
    pub fn can_force_allocate_primary(&self, shard: &ShardRouting, node: &DiscoveryNode) -> Decision {
        let mut d = Decision::yes();
        // Use decider names, not positions: inserting a new decider must not
        // accidentally turn a safety refusal into force-allocation permission.
        for f in CAN_ALLOCATE {
            if let Some(mut s) = f(self, shard, node) {
                let keep = matches!(s.decider.as_str(), "max_retry" | "node_shutdown" | "node_replacement" | "segment_format" | "stateless_shard_role");
                if s.kind == DecisionType::No && !keep {
                    s = Single::new(&s.decider, DecisionType::Yes, format!("primary shard [{}] allowed to force allocate on node [{}]", shard_id_str(&shard.index, shard.shard), node.id));
                }
                d.add(s, self.debug);
            }
        }
        d
    }
}

// ------------------------------------------------------- string helpers --

fn recovery_source_str(r: &RecoverySource) -> String {
    match r {
        RecoverySource::EmptyStore => "new shard recovery".into(),
        RecoverySource::ExistingStore { bootstrap_new_history_uuid } => format!("existing store recovery; bootstrap_history_uuid={bootstrap_new_history_uuid}"),
        RecoverySource::Peer => "peer recovery".into(),
        RecoverySource::Snapshot { repository, snapshot, .. } => format!("snapshot recovery from {repository}:{snapshot}"),
        RecoverySource::LocalShards => "local shards recovery".into(),
    }
}

pub fn state_name(s: ShardState) -> &'static str {
    match s {
        ShardState::Unassigned => "UNASSIGNED",
        ShardState::Initializing => "INITIALIZING",
        ShardState::Started => "STARTED",
        ShardState::Relocating => "RELOCATING",
    }
}

pub fn reason_name(r: UnassignedReason) -> String {
    serde_json::to_value(r).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()
}

/// `UnassignedInfo#toString`.
pub fn unassigned_info_str(u: &UnassignedInfo) -> String {
    let mut s = format!("unassigned_info[[reason={}], at[{}]", reason_name(u.reason), fmt_date_millis(u.at_millis));
    if u.failed_allocations > 0 {
        s.push_str(&format!(", failed_attempts[{}]", u.failed_allocations));
    }
    s.push_str(&format!(", delayed={}", u.delayed));
    if let Some(n) = &u.last_node {
        s.push_str(&format!(", last_node[{n}]"));
    }
    if let Some(d) = &u.details {
        s.push_str(&format!(", details[{d}]"));
    }
    s.push_str(&format!(", allocation_status[{}]]", u.last_allocation_status));
    s
}

/// `ShardRouting#shortSummary`.
pub fn shard_str(c: &ShardRouting) -> String {
    let mut s = format!("{}, node[{}], ", shard_id_str(&c.index, c.shard), c.node_id.as_deref().unwrap_or("null"));
    if let Some(r) = &c.relocating_node_id {
        s.push_str(&format!("relocating [{r}], "));
    }
    s.push_str(if c.primary { "[P]" } else { "[R]" });
    if let Some(r) = &c.recovery_source {
        s.push_str(&format!(", recovery_source[{}]", recovery_source_str(r)));
    }
    s.push_str(&format!(", s[{}]", state_name(c.state)));
    if let Some(a) = &c.allocation_id {
        match &a.relocation_id {
            Some(r) => s.push_str(&format!(", a[id={}, rId={r}]", a.id)),
            None => s.push_str(&format!(", a[id={}]", a.id)),
        }
    }
    if let Some(u) = &c.unassigned_info {
        s.push_str(&format!(", {}", unassigned_info_str(u)));
    }
    if let Some(e) = c.expected_shard_size_in_bytes {
        s.push_str(&format!(", expected_shard_size[{e}]"));
    }
    s
}

/// `DiscoveryNode#toString`.
pub fn node_str(n: &DiscoveryNode) -> String {
    let mut s = format!("{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}", n.name, n.id, n.ephemeral_id, n.host, n.ip, n.transport_address, n.role_abbreviations());
    if !n.attributes.is_empty() {
        s.push_str(&format!("{{{}}}", n.attributes.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", ")));
    }
    s
}

fn y(name: &str, msg: impl Into<String>) -> Option<Single> {
    Some(Single::new(name, DecisionType::Yes, msg))
}
fn no(name: &str, msg: impl Into<String>) -> Option<Single> {
    Some(Single::new(name, DecisionType::No, msg))
}
fn throttle(name: &str, msg: impl Into<String>) -> Option<Single> {
    Some(Single::new(name, DecisionType::Throttle, msg))
}

// ------------------------------------------------------------ max_retry --

const RETRY_API: &str = "POST /_cluster/reroute?retry_failed&metric=none";

fn max_retry(a: &Allocation, shard: &ShardRouting, _node: &DiscoveryNode) -> Option<Single> {
    let max = a.meta(&shard.index).map(|m| index_i64(m, "index.allocation.max_retries", 5)).unwrap_or(5);
    match &shard.unassigned_info {
        // A refused segment format is not a failure of the copy: the
        // `segment_format` decider keeps it off nodes that cannot read it.
        // An older master counted it; it must not block upgraded targets.
        Some(u) if u.details.as_deref().is_some_and(crate::compatibility::is_segment_format_refusal) => {
            y("max_retry", "the last failed allocation was a segment format refusal, which is not counted")
        }
        Some(u) if u.failed_allocations > 0 => {
            if u.failed_allocations as i64 >= max {
                no("max_retry", format!("shard has exceeded the maximum number of retries [{max}] on failed allocation attempts - manually call [{RETRY_API}] to retry, [{}]", unassigned_info_str(u)))
            } else {
                y("max_retry", format!("shard has failed allocating [{}] times but [{max}] retries are allowed", u.failed_allocations))
            }
        }
        _ => y("max_retry", "shard has no previous failures"),
    }
}

// ------------------------------------------- replica_after_primary_active --

fn replica_after_primary_active(a: &Allocation, shard: &ShardRouting, _node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "replica_after_primary_active";
    if shard.primary {
        return y(N, "shard is primary and can be allocated");
    }
    match a.active_primary(&shard.index, shard.shard) {
        None => no(N, "primary shard for this replica is not yet active"),
        Some(_) => y(N, "primary shard for this replica is already active"),
    }
}

// --------------------------------------------------------------- enable --

fn enable_setting_desc(index_level: bool, key_index: &str, key_cluster: &str, value: &str) -> String {
    if index_level { format!("index setting [{key_index}={value}]") } else { format!("cluster setting [{key_cluster}={value}]") }
}

fn enable_allocate(a: &Allocation, shard: &ShardRouting, _node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "enable";
    if a.ignore_disable {
        return y(N, "explicitly ignoring any disabling of allocation due to manual allocation commands via the reroute API");
    }
    let idx = a.meta(&shard.index).and_then(|m| index_setting(m, "index.routing.allocation.enable")).and_then(|s| EnableAllocation::parse(&s));
    let (enable, index_level) = match idx {
        Some(e) => (e, true),
        None => (a.s.enable, false),
    };
    let desc = || enable_setting_desc(index_level, "index.routing.allocation.enable", "cluster.routing.allocation.enable", enable.name());
    match enable {
        EnableAllocation::All => y(N, "all allocations are allowed"),
        EnableAllocation::None => no(N, format!("no allocations are allowed due to {}", desc())),
        EnableAllocation::NewPrimaries => {
            let new_primary = shard.primary && !shard.active() && !matches!(shard.recovery_source, Some(RecoverySource::ExistingStore { .. }));
            if new_primary { y(N, "new primary allocations are allowed") } else { no(N, format!("non-new primary allocations are forbidden due to {}", desc())) }
        }
        EnableAllocation::Primaries => {
            if shard.primary { y(N, "primary allocations are allowed") } else { no(N, format!("replica allocations are forbidden due to {}", desc())) }
        }
    }
}

fn enable_rebalance(a: &Allocation, shard: &ShardRouting) -> Option<Single> {
    const N: &str = "enable";
    if a.ignore_disable {
        return y(N, "allocation is explicitly ignoring any disabling of rebalancing");
    }
    let idx = a.meta(&shard.index).and_then(|m| index_setting(m, "index.routing.rebalance.enable")).and_then(|s| EnableRebalance::parse(&s));
    let (enable, index_level) = match idx {
        Some(e) => (e, true),
        None => (a.s.rebalance_enable, false),
    };
    let desc = || enable_setting_desc(index_level, "index.routing.rebalance.enable", "cluster.routing.rebalance.enable", enable.name());
    match enable {
        EnableRebalance::All => y(N, "all rebalancing is allowed"),
        EnableRebalance::None => no(N, format!("no rebalancing is allowed due to {}", desc())),
        EnableRebalance::Primaries => {
            if shard.primary { y(N, "primary rebalancing is allowed") } else { no(N, format!("replica rebalancing is forbidden due to {}", desc())) }
        }
        EnableRebalance::Replicas => {
            if shard.primary { no(N, format!("primary rebalancing is forbidden due to {}", desc())) } else { y(N, "replica rebalancing is allowed") }
        }
    }
}

fn enable_rebalance_global(a: &Allocation) -> Option<Single> {
    const N: &str = "enable";
    if a.ignore_disable {
        return y(N, "allocation is explicitly ignoring any disabling of rebalancing");
    }
    if a.s.rebalance_enable == EnableRebalance::None {
        let any_index = a.state.metadata.indices.values().any(|m| index_setting(m, "index.routing.rebalance.enable").and_then(|s| EnableRebalance::parse(&s)).is_some_and(|e| e != EnableRebalance::None));
        if any_index {
            return y(N, "rebalancing is permitted on one or more indices");
        }
        return no(N, "no rebalancing is allowed due to cluster setting [cluster.routing.rebalance.enable=none]");
    }
    y(N, "rebalancing is not globally disabled")
}

// --------------------------------------------------------- node_version --

fn node_version(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "node_version";
    if shard.primary {
        let Some(cur) = shard.node_id.as_ref().and_then(|n| a.node(n)) else {
            return y(N, "the primary shard is new or already existed on the node");
        };
        return if parse_version(&node.version) >= parse_version(&cur.version) {
            y(N, format!("can relocate primary shard from a node with version [{}] to a node with equal-or-newer version [{}]", cur.version, node.version))
        } else {
            no(N, format!("cannot relocate primary shard from a node with version [{}] to a node with older version [{}]", cur.version, node.version))
        };
    }
    let Some(p) = a.active_primary(&shard.index, shard.shard).and_then(|p| p.node_id.as_ref()).and_then(|n| a.node(n)) else {
        return y(N, "no active primary shard yet");
    };
    if parse_version(&node.version) >= parse_version(&p.version) {
        y(N, format!("can allocate replica shard to a node with version [{}] since this is equal-or-newer than the primary version [{}]", node.version, p.version))
    } else {
        no(N, format!("cannot allocate replica shard to a node with version [{}] since this is older than the primary version [{}]", node.version, p.version))
    }
}

// The product version is constant across internal builds. Check the actual
// writer/reader formats before assigning either a replica or relocation target.
fn segment_format(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "segment_format";
    let source = if shard.primary {
        shard.node_id.as_ref().and_then(|id| a.node(id))
    } else {
        a.active_primary(&shard.index, shard.shard).and_then(|p| p.node_id.as_ref()).and_then(|id| a.node(id))
    };
    let checked = (|| -> Result<(), String> {
        let (_, target_write) = crate::compatibility::segment_formats(node)?;
        crate::compatibility::require_segment_reader(node, target_write)?;
        // File kinds the copy holds whatever its writers' versions (DiskBBQ
        // IVF files), and what an earlier recovery to a node like this one
        // was refused for: the source would refuse it again.
        if let Some(meta) = a.meta(&shard.index) {
            crate::compatibility::require_segment_reader(node, crate::compatibility::index_required_reader(meta))?;
        }
        if let Some(refused) = shard.unassigned_info.as_ref().and_then(|u| u.details.as_deref()).and_then(crate::compatibility::refused_segment_format) {
            crate::compatibility::require_segment_reader(node, refused)?;
        }
        if let Some(source) = source {
            let (_, write) = crate::compatibility::segment_formats(source)?;
            crate::compatibility::require_segment_reader(node, write)?;
            // Replicas write too; any future primary must read those files.
            crate::compatibility::require_segment_reader(source, target_write)?;
        }
        Ok(())
    })();
    match checked { Ok(()) => y(N, "segment reader and writer formats are compatible"), Err(reason) => no(N, reason) }
}

// -------------------------------------------------- restore_in_progress --

/// Node attribute (`rustic.internal.*`, generated by the process) of a build
/// that restores a primary by downloading and adopting the snapshot's shard
/// files: an older build would open a `Snapshot`-sourced primary empty.
pub const SNAPSHOT_FILE_RESTORE_ATTR: &str = "rustic.internal.snapshot_file_restore_v1";

/// Whether `node` advertises [`SNAPSHOT_FILE_RESTORE_ATTR`] under the
/// capability schema this build recognizes.
pub fn restores_snapshot_files(node: &DiscoveryNode) -> bool {
    node.attributes.get(crate::compatibility::CAPABILITIES_SCHEMA).is_some_and(|v| v == "1")
        && node.attributes.get(SNAPSHOT_FILE_RESTORE_ATTR).is_some_and(|v| v == "1")
}

/// ES `RestoreInProgressAllocationDecider`: a primary restored from a
/// snapshot is not allocated again once its restore failed (ES marks the
/// shard's restore FAILURE on the first failure and refuses it from then
/// on; closing or deleting the index, or forcing an empty primary, is the
/// way out). A failed allocation (`failed_allocations > 0`) is that
/// failure here; a copy unassigned because its node left is restored
/// elsewhere. Only nodes that adopt snapshot files may hold such a copy.
fn restore_in_progress(_a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "restore_in_progress";
    let Some(RecoverySource::Snapshot { repository, snapshot, .. }) = shard.recovery_source.as_ref().filter(|_| shard.primary) else {
        return y(N, "ignored as shard is not being recovered from a snapshot");
    };
    if let Some(u) = shard.unassigned_info.as_ref().filter(|u| u.failed_allocations > 0) {
        return no(N, format!(
            "shard has failed to be restored from the snapshot [{repository}:{snapshot}] - manually close or delete the index [{}] in order to retry to restore the snapshot again or use the reroute API to force the allocation of an empty primary shard. Details: [{}]",
            shard.index,
            u.details.as_deref().unwrap_or_default()
        ));
    }
    if !restores_snapshot_files(node) {
        return no(N, format!("node [{}] cannot restore shard files from a snapshot (it does not advertise [{SNAPSHOT_FILE_RESTORE_ATTR}])", node.name));
    }
    y(N, "shard is currently being restored")
}

// -------------------------------------------------------- node_shutdown --

fn shutdown_decision(a: &Allocation, node: &DiscoveryNode, auto_expand: bool) -> Option<Single> {
    const N: &str = "node_shutdown";
    if all_shutdowns(&a.state.metadata).is_empty() {
        return y(N, "no nodes are shutting down");
    }
    match shutdown_info(&a.state.metadata, &node.id) {
        None => y(N, "this node is not shutting down"),
        Some(s) if s.kind == "restart" => y(N, format!("node [{}] is preparing to restart, but will remain in the cluster", node.id)),
        Some(s) if s.kind == "replace" && auto_expand && !has_node_named(a, s.target_node_name.as_deref()) => {
            y(N, format!("node [{}] is preparing to be removed from the cluster, but replacement is not yet present", node.id))
        }
        Some(_) => no(N, format!("node [{}] is preparing to be removed from the cluster", node.id)),
    }
}

fn node_shutdown_allocate(a: &Allocation, _shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    shutdown_decision(a, node, false)
}

fn node_shutdown_auto_expand(a: &Allocation, _meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    shutdown_decision(a, node, true)
}

// ----------------------------------------------------- node_replacement --

const NR: &str = "node_replacement";

fn has_node_named(a: &Allocation, name: Option<&str>) -> bool {
    name.is_some_and(|n| a.state.nodes.values().any(|d| d.name == n))
}

/// (source node id, target node name) of every ongoing `replace` shutdown.
fn replacements(a: &Allocation) -> Vec<(String, String)> {
    all_shutdowns(&a.state.metadata).into_iter().filter(|(_, s)| s.kind == "replace").filter_map(|(id, s)| s.target_node_name.map(|t| (id, t))).collect()
}

fn replacement_target_of(r: &[(String, String)], source: &str) -> Option<String> {
    r.iter().find(|(s, _)| s == source).map(|(_, t)| t.clone())
}

fn node_replacement_allocate(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    let r = replacements(a);
    if r.is_empty() {
        return y(NR, "there are no ongoing node replacements");
    }
    let cur = shard.node_id.as_deref().unwrap_or("");
    if r.iter().any(|(s, t)| s == cur && *t == node.name) {
        return y(NR, format!("node [{}] is replacing node [{}], and may receive shards from it", node.id, cur));
    }
    if let Some(t) = replacement_target_of(&r, cur) {
        return no(NR, format!("node [{cur}] is being replaced, and its shards may only be allocated to the replacement target [{t}]"));
    }
    if let Some(t) = replacement_target_of(&r, &node.id) {
        return no(NR, format!("node [{}] is being replaced by [{t}], so no data may be allocated to it", node.id));
    }
    if let Some((src, _)) = r.iter().find(|(_, t)| *t == node.name) {
        return no(NR, format!("node [{}] is replacing the vacating node [{src}], only data currently allocated to the source node may be allocated to it until the replacement is complete", node.id));
    }
    y(NR, "none of the ongoing node replacements relate to the allocation of this shard")
}

fn node_replacement_remain(a: &Allocation, _shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    let r = replacements(a);
    if r.is_empty() {
        return y(NR, "there are no ongoing node replacements");
    }
    match replacement_target_of(&r, &node.id) {
        Some(t) => no(NR, format!("node [{}] is being replaced by node [{t}], so no data may remain on it", node.id)),
        None => y(NR, "none of the ongoing node replacements relate to the allocation of this shard"),
    }
}

fn node_replacement_auto_expand(a: &Allocation, meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    let r = replacements(a);
    if r.is_empty() {
        return y(NR, "there are no ongoing node replacements");
    }
    let has_on = |nid: &str| a.state.routing_table.shards_on_node(nid).any(|c| c.index == meta.name);
    if let Some((src, _)) = r.iter().find(|(_, t)| *t == node.name) {
        return if has_on(&node.id) || has_on(src) {
            y(NR, format!("node [{}] is a node replacement target for node [{src}], shard can auto expand to it as it was already present on the source node", node.id))
        } else {
            no(NR, format!("node [{}] is a node replacement target for node [{src}], shards cannot auto expand to be on it until the replacement is complete", node.id))
        };
    }
    if let Some(t) = replacement_target_of(&r, &node.id) {
        return if has_on(&node.id) && !has_node_named(a, Some(&t)) {
            y(NR, format!("node [{}] is being replaced by [{t}], shards can auto expand to be on it while replacement node has not joined the cluster", node.id))
        } else {
            no(NR, format!("node [{}] is being replaced by [{t}], shards cannot auto expand to be on it", node.id))
        };
    }
    y(NR, "none of the ongoing node replacements relate to the allocation of this shard")
}

// --------------------------------------------------------------- filter --

fn simple_match(pattern: &str, value: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == value;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !value.starts_with(first) || value.len() < first.len() + last.len() || !value.ends_with(last) {
        return false;
    }
    let mut rest = &value[first.len()..value.len() - last.len()];
    for p in &parts[1..parts.len() - 1] {
        match rest.find(p) {
            Some(i) => rest = &rest[i + p.len()..],
            None => return false,
        }
    }
    true
}

fn attr_matches(attr: &str, values: &[String], node: &DiscoveryNode) -> bool {
    let any = |cands: &[&str]| values.iter().any(|v| cands.iter().any(|c| simple_match(v, c)));
    match attr {
        "_ip" | "_host_ip" | "_publish_ip" => any(&[&node.ip]),
        "_host" => any(&[&node.host, &node.ip]),
        "_id" => values.iter().any(|v| v == &node.id),
        "_name" | "name" => any(&[&node.name]),
        "_tier" => values.iter().any(|v| node.roles.iter().any(|r| simple_match(v, r.name()))),
        _ => node.attributes.get(attr).is_some_and(|nv| any(&[nv])),
    }
}

/// `DiscoveryNodeFilters.match`: AND for `require`, OR for include/exclude.
pub fn filters_match(f: &NodeFilters, and: bool, node: &DiscoveryNode) -> bool {
    if and { f.0.iter().all(|(k, v)| attr_matches(k, v, node)) } else { f.0.iter().any(|(k, v)| attr_matches(k, v, node)) }
}

fn index_filters(meta: &IndexMetadata, kind: &str) -> NodeFilters {
    let prefix = format!("index.routing.allocation.{kind}.");
    NodeFilters::from_entries(&prefix, meta.settings.iter().filter_map(|(k, v)| super::settings::value_str(v).map(|s| (k, s))))
}

fn filter_check(a: &Allocation, meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "filter";
    let s = &a.s;
    if !s.require.is_empty() && !filters_match(&s.require, true, node) {
        return no(N, format!("node does not match cluster setting [cluster.routing.allocation.require] filters [{}]", s.require.describe()));
    }
    if !s.include.is_empty() && !filters_match(&s.include, false, node) {
        return no(N, format!("node does not match cluster setting [cluster.routing.allocation.include] filters [{}]", s.include.describe()));
    }
    if !s.exclude.is_empty() && filters_match(&s.exclude, false, node) {
        return no(N, format!("node matches cluster setting [cluster.routing.allocation.exclude] filters [{}]", s.exclude.describe()));
    }
    let req = index_filters(meta, "require");
    if !req.is_empty() && !filters_match(&req, true, node) {
        return no(N, format!("node does not match index setting [index.routing.allocation.require] filters [{}]", req.describe()));
    }
    let inc = index_filters(meta, "include");
    if !inc.is_empty() && !filters_match(&inc, false, node) {
        return no(N, format!("node does not match index setting [index.routing.allocation.include] filters [{}]", inc.describe()));
    }
    let exc = index_filters(meta, "exclude");
    if !exc.is_empty() && filters_match(&exc, false, node) {
        return no(N, format!("node matches index setting [index.routing.allocation.exclude] filters [{}]", exc.describe()));
    }
    y(N, "node passes include/exclude/require filters")
}

fn filter_allocate(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    let meta = a.meta(&shard.index)?;
    filter_check(a, meta, node)
}

fn filter_index(a: &Allocation, meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    filter_check(a, meta, node)
}

// ----------------------------------------------------------- same_shard --

fn same_shard(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "same_shard";
    let t = a.table(&shard.index, shard.shard)?;
    for c in &t.copies {
        if c.node_id.as_deref() == Some(node.id.as_str()) {
            if c.allocation_id.is_some() && c.allocation_id() == shard.allocation_id() {
                return no(N, format!("this shard is already allocated to this node [{}]", shard_str(shard)));
            }
            return no(N, format!("a copy of this shard is already allocated to this node [{}]", shard_str(c)));
        }
    }
    y(N, "this node does not hold a copy of this shard")
}

// -------------------------------------------------------- disk_threshold --

/// (total, free) for a node, falling back to the average of known nodes
/// (ES `getDiskUsage` with no entry for the node).
fn node_disk(a: &Allocation, node: &str) -> Option<(u64, u64)> {
    if let Some(u) = a.info.nodes.get(node) {
        return Some((u.total_bytes, u.free_bytes.min(u.total_bytes)));
    }
    if a.info.nodes.is_empty() {
        return None;
    }
    let n = a.info.nodes.len() as u64;
    let total = a.info.nodes.values().map(|u| u.total_bytes).sum::<u64>() / n;
    let free = a.info.nodes.values().map(|u| u.free_bytes).sum::<u64>() / n;
    Some((total, free.min(total)))
}

fn used_pct(total: u64, free: u64) -> String {
    let used = if total == 0 { 0.0 } else { 100.0 * (total - free.min(total)) as f64 / total as f64 };
    format1(used, "%")
}

fn disk_allocate(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "disk_threshold";
    if !a.s.disk_threshold_enabled {
        return y(N, "the disk threshold decider is disabled");
    }
    let Some((total, free0)) = node_disk(a, &node.id) else {
        return y(N, "disk usages are unavailable");
    };
    // shards relocating onto this node are not reflected in the usage yet
    let free = free0.saturating_sub(a.stats.incoming_relocation_bytes.get(&node.id).copied().unwrap_or(0));
    let (low_req, low_hr) = a.s.low.required_free(total);
    let (high_req, high_hr) = a.s.high.required_free(total);
    let new_primary = shard.primary && !shard.active() && shard.recovery_source == Some(RecoverySource::EmptyStore);
    if free < low_req {
        if !new_primary {
            return no(N, format!(
                "the node is above the low watermark cluster setting [{}], having less than the minimum required [{}] free space, actual free: [{}], actual used: [{}]",
                a.s.low.describe(low_hr), fmt_bytes(low_req), fmt_bytes(free), used_pct(total, free)
            ));
        }
        if free < high_req {
            return no(N, format!(
                "the node is above the high watermark cluster setting [{}], having less than the minimum required [{}] free space, actual free: [{}], actual used: [{}]",
                a.s.high.describe(high_hr), fmt_bytes(high_req), fmt_bytes(free), used_pct(total, free)
            ));
        }
    }
    let size = if shard.assigned() { a.shard_size(shard) } else { a.expected_size(shard) };
    let after = free.saturating_sub(size);
    if after < high_req {
        return no(N, format!(
            "allocating the shard to this node will bring the node above the high watermark cluster setting [{}] and cause it to have less than the minimum required [{}] of free space (free: [{}], used: [{}], estimated shard size: [{}])",
            a.s.high.describe(high_hr), fmt_bytes(high_req), fmt_bytes(free), used_pct(total, free), fmt_bytes(size)
        ));
    }
    y(N, format!("enough disk for shard on node, free: [{}], used: [{}], shard size: [{}], free after allocating shard: [{}]", fmt_bytes(free), used_pct(total, free), fmt_bytes(size), fmt_bytes(after)))
}

fn disk_remain(a: &Allocation, _shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "disk_threshold";
    if !a.s.disk_threshold_enabled {
        return y(N, "the disk threshold decider is disabled");
    }
    let Some((total, free0)) = node_disk(a, &node.id) else {
        return y(N, "disk usages are unavailable");
    };
    // shards already leaving free their space once relocated
    let free = (free0 + a.stats.outgoing_relocation_bytes.get(&node.id).copied().unwrap_or(0)).saturating_sub(a.stats.incoming_relocation_bytes.get(&node.id).copied().unwrap_or(0)).min(total);
    let (high_req, high_hr) = a.s.high.required_free(total);
    if free < high_req {
        return no(N, format!(
            "the shard cannot remain on this node because it is above the high watermark cluster setting [{}] and there is less than the required [{}] free space on node, actual free: [{}], actual used: [{}]",
            a.s.high.describe(high_hr), fmt_bytes(high_req), fmt_bytes(free), used_pct(total, free)
        ));
    }
    y(N, format!("there is enough disk on this node for the shard to remain, free: [{}]", fmt_bytes(free)))
}

// ----------------------------------------------------------- throttling --

fn throttling(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "throttling";
    let s = &a.s;
    if shard.primary && shard.state == ShardState::Unassigned {
        let in_recovery = a.stats.initial_primaries.get(&node.id).copied().unwrap_or(0) as i64;
        let limit = s.node_initial_primaries_recoveries;
        return if in_recovery >= limit {
            throttle(N, format!("reached the limit of ongoing initial primary recoveries [{in_recovery}], cluster setting [cluster.routing.allocation.node_initial_primaries_recoveries={limit}]"))
        } else {
            y(N, format!("below primary recovery limit of [{limit}]"))
        };
    }
    let incoming = a.stats.incoming.get(&node.id).copied().unwrap_or(0) as i64;
    if incoming >= s.node_concurrent_incoming {
        return throttle(N, format!(
            "reached the limit of incoming shard recoveries [{incoming}], cluster setting [cluster.routing.allocation.node_concurrent_incoming_recoveries={}] (can also be set via [cluster.routing.allocation.node_concurrent_recoveries])",
            s.node_concurrent_incoming
        ));
    }
    let Some(primary_node) = a.active_primary(&shard.index, shard.shard).and_then(|p| p.node_id.clone()) else {
        return no(N, "primary shard for this replica is not yet active");
    };
    let outgoing = a.stats.outgoing.get(&primary_node).copied().unwrap_or(0) as i64;
    if outgoing >= s.node_concurrent_outgoing {
        return throttle(N, format!(
            "reached the limit of outgoing shard recoveries [{outgoing}] on the node [{primary_node}] which holds the primary, cluster setting [cluster.routing.allocation.node_concurrent_outgoing_recoveries={}] (can also be set via [cluster.routing.allocation.node_concurrent_recoveries])",
            s.node_concurrent_outgoing
        ));
    }
    y(N, format!("below shard recovery limit of outgoing: [{outgoing} < {}] incoming: [{incoming} < {}]", s.node_concurrent_outgoing, s.node_concurrent_incoming))
}

// --------------------------------------------------------- shards_limit --

fn shards_limit(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode, remain: bool) -> Option<Single> {
    const N: &str = "shards_limit";
    let index_limit = a.meta(&shard.index).map(|m| index_i64(m, "index.routing.allocation.total_shards_per_node", -1)).unwrap_or(-1);
    let cluster_limit = a.s.cluster_total_shards_per_node;
    if index_limit <= 0 && cluster_limit <= 0 {
        return y(N, format!("total shard limits are disabled: [index: {index_limit}, cluster: {cluster_limit}] <= 0"));
    }
    let node_count = a.stats.node_shards.get(&node.id).copied().unwrap_or(0) as i64;
    let index_count = a.stats.node_index_shards.get(&(node.id.clone(), shard.index.clone())).copied().unwrap_or(0) as i64;
    let breached = |count: i64, limit: i64| if remain { count > limit } else { count >= limit };
    if cluster_limit > 0 && breached(node_count, cluster_limit) {
        return no(N, format!("too many shards [{node_count}] allocated to this node, cluster setting [cluster.routing.allocation.total_shards_per_node={cluster_limit}]"));
    }
    if index_limit > 0 && breached(index_count, index_limit) {
        return no(N, format!("too many shards [{index_count}] allocated to this node for index [{}], index setting [index.routing.allocation.total_shards_per_node={index_limit}]", shard.index));
    }
    y(N, format!("the shard count [{node_count}] for this node is under the index limit [{index_limit}] and cluster level node limit [{cluster_limit}]"))
}

fn shards_limit_allocate(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    shards_limit(a, shard, node, false)
}

fn shards_limit_remain(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    shards_limit(a, shard, node, true)
}

// ------------------------------------------------------------ awareness --

fn awareness(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode, move_to_node: bool) -> Option<Single> {
    const N: &str = "awareness";
    let attrs = &a.s.awareness_attributes;
    if attrs.is_empty() {
        return y(N, "allocation awareness is not enabled, set cluster setting [cluster.routing.allocation.awareness.attributes] to enable it");
    }
    let meta = a.meta(&shard.index)?;
    if index_setting(meta, "index.auto_expand_replicas").is_some_and(|s| s.trim().ends_with("-all")) {
        return y(N, "allocation awareness is ignored, this index is set to auto-expand to all nodes");
    }
    let shard_count = meta.number_of_replicas as usize + 1;
    for attr in attrs {
        let Some(target_value) = node.attributes.get(attr) else {
            return no(N, format!("node does not contain the awareness attribute [{attr}]; required attributes cluster setting [cluster.routing.allocation.awareness.attributes={}]", attrs.join(",")));
        };
        let actual: BTreeSet<&String> = a.state.data_nodes().filter_map(|n| n.attributes.get(attr)).collect();
        let mut count = 0usize;
        if let Some(t) = a.table(&shard.index, shard.shard) {
            for c in t.copies.iter().filter(|c| matches!(c.state, ShardState::Started | ShardState::Initializing)) {
                if c.node_id.as_ref().and_then(|n| a.node(n)).and_then(|n| n.attributes.get(attr)) == Some(target_value) {
                    count += 1;
                }
            }
        }
        if move_to_node {
            if shard.assigned() {
                let cur = if shard.state == ShardState::Relocating { shard.relocating_node_id.as_ref() } else { shard.node_id.as_ref() };
                if cur.and_then(|n| a.node(n)).and_then(|n| n.attributes.get(attr)) != Some(target_value) {
                    count += 1;
                }
            } else {
                count += 1;
            }
        }
        let forced = a.s.forced_awareness.get(attr);
        let mut all_values: BTreeSet<&String> = actual.clone();
        if let Some(f) = forced {
            all_values.extend(f.iter());
        }
        let value_count = all_values.len().max(1);
        let max_per_value = shard_count.div_ceil(value_count);
        if count > max_per_value {
            let actual_str = format!("[{}]", actual.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
            let forced_str = match forced {
                None => "no forced awareness".to_string(),
                Some(f) => format!("[{}] from forced awareness", f.join(", ")),
            };
            return no(N, format!(
                "there are [{shard_count}] copies of this shard and [{value_count}] values for attribute [{attr}] ({actual_str} from nodes in the cluster and {forced_str}) so there may be at most [{max_per_value}] copies of this shard allocated to nodes with each value, but (including this copy) there would be [{count}] copies allocated to nodes with [node.attr.{attr}: {target_value}]"
            ));
        }
    }
    y(N, "node meets all awareness attribute requirements")
}

fn awareness_allocate(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    awareness(a, shard, node, true)
}

fn awareness_remain(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    awareness(a, shard, node, false)
}

// ------------------------------------------------------------ data_tier --

/// `index.routing.allocation.include._tier_preference`; regular indices get
/// `data_content` at creation in ES, so an absent key means `data_content`.
pub fn tier_preference(meta: &IndexMetadata) -> Vec<String> {
    match meta.settings.get("index.routing.allocation.include._tier_preference") {
        None => vec!["data_content".into()],
        Some(v) => super::settings::value_str(v).map(|s| super::settings::split_list(&s)).unwrap_or_default(),
    }
}

fn node_has_tier(node: &DiscoveryNode, tier: &str) -> bool {
    // Stateless nodes have no data tiers.
    node.is_stateless() || node.roles.iter().any(|r| *r == Role::Data || r.name() == tier)
}

// ------------------------------------------------- stateless_shard_role --

/// `StatelessAllocationDecider`: on stateless nodes the indexing copy (the
/// primary, promotable) goes to `index` nodes and the searchable copies
/// (replicas) to `search` nodes. Nodes without a stateless role are not
/// this decider's business.
fn stateless_shard_role(_a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "stateless_shard_role";
    if !node.is_stateless() {
        return None;
    }
    let ok = if shard.primary { node.roles.contains(&Role::Index) } else { node.roles.contains(&Role::Search) };
    if ok {
        y(N, "shard role matches stateless node role")
    } else {
        let shard_role = if shard.primary { "INDEX_ONLY" } else { "SEARCH_ONLY" };
        let node_role: Vec<&str> = node.roles.iter().filter(|r| r.is_stateless()).map(|r| r.name()).collect();
        no(N, format!("shard role [{shard_role}] does not match stateless node role [{}]", node_role.join(",")))
    }
}

fn tier_check(a: &Allocation, meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    const N: &str = "data_tier";
    let pref = tier_preference(meta);
    if pref.is_empty() {
        return y(N, "index has no tier preference");
    }
    let joined = pref.join(",");
    let Some(tier) = pref.iter().find(|t| a.state.nodes.values().any(|n| node_has_tier(n, t))) else {
        return no(N, format!("index has a preference for tiers [{joined}], but no nodes for any of those tiers are available in the cluster"));
    };
    if node_has_tier(node, tier) {
        y(N, format!("index has a preference for tiers [{joined}] and node has tier [{tier}]"))
    } else {
        no(N, format!("index has a preference for tiers [{joined}] and node does not meet the required [{tier}] tier"))
    }
}

fn data_tier(a: &Allocation, shard: &ShardRouting, node: &DiscoveryNode) -> Option<Single> {
    let meta = a.meta(&shard.index)?;
    tier_check(a, meta, node)
}

fn data_tier_index(a: &Allocation, meta: &IndexMetadata, node: &DiscoveryNode) -> Option<Single> {
    tier_check(a, meta, node)
}

// ------------------------------------------------------------ rebalance --

fn rebalance_only_when_active(a: &Allocation, shard: &ShardRouting) -> Option<Single> {
    const N: &str = "rebalance_only_when_active";
    let all_active = match (a.table(&shard.index, shard.shard), a.meta(&shard.index)) {
        (Some(t), Some(m)) => {
            let copies: Vec<&ShardRouting> = t.copies.iter().filter(|c| !is_relocation_target(c)).collect();
            copies.len() == m.number_of_replicas as usize + 1 && copies.iter().all(|c| c.active())
        }
        _ => false,
    };
    if all_active { y(N, "rebalancing is allowed as all replicas are active in the cluster") } else { no(N, "rebalancing is not allowed until all replicas in the cluster are active") }
}

fn cluster_rebalance(a: &Allocation) -> Option<Single> {
    const N: &str = "cluster_rebalance";
    const KEY: &str = "cluster.routing.allocation.allow_rebalance";
    let mode = a.s.allow_rebalance;
    let unassigned = if a.unassigned_ignored { 0 } else { a.stats.unassigned };
    let unassigned_primaries = if a.unassigned_ignored { 0 } else { a.stats.unassigned_primaries };
    match mode {
        AllowRebalance::IndicesPrimariesActive => {
            if unassigned_primaries > 0 {
                return no(N, format!("the cluster has unassigned primary shards and cluster setting [{KEY}] is set to [{}]", mode.name()));
            }
            if a.stats.inactive_primaries > 0 {
                return no(N, format!("the cluster has inactive primary shards and cluster setting [{KEY}] is set to [{}]", mode.name()));
            }
            y(N, "all primary shards are active")
        }
        AllowRebalance::IndicesAllActive => {
            if unassigned > 0 {
                return no(N, format!("the cluster has unassigned shards and cluster setting [{KEY}] is set to [{}]", mode.name()));
            }
            if a.stats.inactive_shards > 0 {
                return no(N, format!("the cluster has inactive shards and cluster setting [{KEY}] is set to [{}]", mode.name()));
            }
            y(N, "all shards are active")
        }
        AllowRebalance::Always => y(N, "all shards are active"),
    }
}

fn cluster_rebalance_shard(a: &Allocation, _shard: &ShardRouting) -> Option<Single> {
    cluster_rebalance(a)
}

fn concurrent_rebalance(a: &Allocation) -> Option<Single> {
    const N: &str = "concurrent_rebalance";
    let limit = a.s.cluster_concurrent_rebalance;
    if limit == -1 {
        return y(N, "unlimited concurrent rebalances are allowed");
    }
    let relocating = a.stats.relocating as i64;
    if relocating >= limit {
        return throttle(N, format!("reached the limit of concurrently rebalancing shards [{relocating}], cluster setting [cluster.routing.allocation.cluster_concurrent_rebalance={limit}]"));
    }
    y(N, format!("below threshold [{limit}] for concurrent rebalances, current rebalance shard count [{relocating}]"))
}

fn concurrent_rebalance_shard(a: &Allocation, _shard: &ShardRouting) -> Option<Single> {
    concurrent_rebalance(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matching() {
        assert!(simple_match("node-*", "node-1"));
        assert!(simple_match("*-1", "node-1"));
        assert!(simple_match("n*e*1", "node-1"));
        assert!(!simple_match("node-*", "data-1"));
        assert!(simple_match("*", "anything"));
        assert!(simple_match("exact", "exact"));
        assert!(!simple_match("a*a", "a"));
    }

    #[test]
    fn decision_aggregation() {
        let mut d = Decision::yes();
        d.add(Single::new("a", DecisionType::Yes, "ok"), DebugMode::ExcludeYes);
        assert!(d.singles.is_empty());
        d.add(Single::new("b", DecisionType::Throttle, "busy"), DebugMode::ExcludeYes);
        assert_eq!(d.kind, DecisionType::Throttle);
        d.add(Single::new("c", DecisionType::No, "nope"), DebugMode::ExcludeYes);
        assert_eq!(d.kind, DecisionType::No);
        assert_eq!(d.describe(), "[THROTTLE(busy)][NO(nope)]");
    }
}
