//! Shard allocation: ES's `AllocationService` as pure functions over
//! [`ClusterState`].
//!
//! The master runs every routing change through [`AllocationService`]:
//!
//! * index lifecycle: [`add_index_routing`](AllocationService::add_index_routing)
//!   (create), [`add_restored_index_routing`](AllocationService::add_restored_index_routing),
//!   [`delete_index`](AllocationService::delete_index), [`close_index`](AllocationService::close_index) /
//!   [`open_index`](AllocationService::open_index), [`update_number_of_replicas`](AllocationService::update_number_of_replicas);
//! * shard lifecycle reported by data nodes: [`apply_started_shards`](AllocationService::apply_started_shards),
//!   [`apply_failed_shards`](AllocationService::apply_failed_shards), and node membership:
//!   [`disassociate_dead_nodes`](AllocationService::disassociate_dead_nodes);
//! * [`reroute`](AllocationService::reroute): delayed-allocation expiry,
//!   `auto_expand_replicas`, allocate unassigned shards, move shards that
//!   may not remain, rebalance — see `balancer`.
//!
//! Like ES, the transition functions do not reroute on their own: the master
//! applies a transition and then `reroute`s in the same cluster-state update.
//! None of them bump `ClusterState.version`; the master does that with
//! `ClusterState::next()` and [`patch_versions`] (ES `MasterService.patchVersions`).
//! `IndexMetadata.version` *is* bumped here whenever allocation changes an
//! index's metadata (in-sync ids, primary terms, replicas), as ES does.
//!
//! Data-safety rules (enforced in `context` / `balancer`):
//! * a primary is only ever created on a copy whose allocation id is in
//!   `in_sync_allocation_ids` (promotion of an in-sync active replica, or an
//!   existing-store copy reported by [`ShardStoreInfo`] keeping its id), except
//!   through `allocate_stale_primary` / `allocate_empty_primary` with
//!   `accept_data_loss`;
//! * copies leave the in-sync set only when failed with `mark_as_stale`
//!   (failed by the primary while replicating), never because their node
//!   left; the set is never emptied once populated;
//! * every new primary (initialization or promotion) increments
//!   `primary_terms[shard]`.

mod balancer;
mod commands;
mod context;
mod deciders;
mod explain;
mod lifecycle;
pub mod settings;
mod shutdown;

#[cfg(test)]
mod tests;

pub use commands::{RerouteCommandsResult, RerouteFlags};
pub use deciders::{restores_snapshot_files, Decision, DecisionType, Single, SNAPSHOT_FILE_RESTORE_ATTR};
pub use explain::ExplainOptions;
pub use settings::AllocSettings;
pub use shutdown::ShardMigrationStatus;

use crate::state::*;
use context::Allocation;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// `(http status, error type, reason)` — rendered by the node crate as ES's
/// `{"error":{"root_cause":[...],"type":..,"reason":..},"status":..}`.
pub type ApiError = (u16, String, String);

// ----------------------------------------------------------- cluster info --

/// Immutable path whose actual producer allocation can retain an external
/// admission owner without introducing a dependency on the engine crate.
#[derive(Clone)]
pub struct DiskPath(std::sync::Arc<dyn DiskPathSource>);
pub trait DiskPathSource: Send+Sync { fn as_str(&self)->&str; }
impl DiskPathSource for String {fn as_str(&self)->&str{self}}
impl DiskPath {
    pub fn from_owner(owner:std::sync::Arc<dyn DiskPathSource>)->Self{Self(owner)}
}
impl std::ops::Deref for DiskPath{type Target=str;fn deref(&self)->&str{self.0.as_str()}}
impl std::fmt::Debug for DiskPath{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{self.0.as_str().fmt(f)}}
impl std::fmt::Display for DiskPath{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(self)}}
impl PartialEq for DiskPath{fn eq(&self,o:&Self)->bool{**self==**o}}
impl Default for DiskPath{fn default()->Self{Self(std::sync::Arc::new(String::new()))}}
impl From<String> for DiskPath{fn from(value:String)->Self{Self(std::sync::Arc::new(value))}}
impl From<&str> for DiskPath{fn from(value:&str)->Self{value.to_owned().into()}}
impl serde::Serialize for DiskPath{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{s.serialize_str(self)}}

/// Disk usage of one node's data path (ES `DiskUsage`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DiskUsage {
    pub path: DiskPath,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// Advisory process memory utilization. Constructed only from fresh, comparable
/// observations by the master; never a recovery reservation or a hard decider.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacementMemory {
    pub measured_bytes: u64,
    pub budget_bytes: u64,
}

/// ES `ClusterInfo`: what the master knows about disks and shard sizes,
/// refreshed periodically by the node crate (nodes stats).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterInfo {
    /// node id → usage
    pub nodes: BTreeMap<String, DiskUsage>,
    /// (index, shard, primary) → store size in bytes
    pub shard_sizes: BTreeMap<(String, u32, bool), u64>,
    /// Ephemeral, non-wire hint used only to break equal-weight unassigned ties.
    pub placement_memory: BTreeMap<String, PlacementMemory>,
    /// Ephemeral, non-wire: nodes whose fresh observations show sustained
    /// high memory pressure. Rebalancing moves nothing onto them.
    pub memory_pressure: BTreeSet<String>,
}

impl ClusterInfo {
    pub fn shard_size(&self, index: &str, shard: u32, primary: bool) -> Option<u64> {
        self.shard_sizes.get(&(index.to_string(), shard, primary)).copied()
    }

    /// `cluster_info` of `_cluster/allocation/explain?include_disk_info`.
    pub fn to_json(&self, state: &ClusterState) -> Value {
        let mut nodes = Map::new();
        for (id, u) in &self.nodes {
            let used = u.total_bytes.saturating_sub(u.free_bytes);
            let pct = |x: u64| if u.total_bytes == 0 { 0.0 } else { 100.0 * x as f64 / u.total_bytes as f64 };
            let usage = json!({"path": u.path, "total_bytes": u.total_bytes, "used_bytes": used, "free_bytes": u.free_bytes, "free_disk_percent": pct(u.free_bytes), "used_disk_percent": pct(used)});
            nodes.insert(id.clone(), json!({"node_name": state.nodes.get(id).map(|n| n.name.clone()).unwrap_or_default(), "least_available": usage, "most_available": usage}));
        }
        let mut sizes = Map::new();
        for ((i, s, p), b) in &self.shard_sizes {
            sizes.insert(format!("[{i}][{s}][{}]_bytes", if *p { "p" } else { "r" }), json!(b));
        }
        json!({"nodes": nodes, "shard_sizes": sizes})
    }
}

// ----------------------------------------------------------- shard stores --

/// One on-disk copy of a shard reported by a data node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreCopy {
    pub node_id: String,
    /// Allocation id recorded in the copy's shard state file.
    pub allocation_id: Option<String>,
    /// Whether the copy was a primary (shard state file); ES prefers these.
    pub primary: bool,
}

/// What data nodes hold on disk (ES's async-fetched `NodeGatewayStartedShards`).
/// The node crate fetches it for shards whose primary must recover from an
/// existing store (and for delayed / existing replicas) and passes it to
/// `reroute`. A shard marked pending is "fetching": it is not allocated yet,
/// rebalancing pauses, and health reports it in `number_of_in_flight_fetch`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShardStoreInfo {
    copies: BTreeMap<(String, u32), Vec<StoreCopy>>,
    pending: BTreeSet<(String, u32)>,
}

impl ShardStoreInfo {
    pub fn new() -> ShardStoreInfo {
        ShardStoreInfo::default()
    }

    pub fn add_copy(&mut self, index: &str, shard: u32, node_id: &str, allocation_id: Option<&str>, primary: bool) {
        let v = self.copies.entry((index.to_string(), shard)).or_default();
        v.retain(|c| c.node_id != node_id);
        v.push(StoreCopy { node_id: node_id.to_string(), allocation_id: allocation_id.map(String::from), primary });
    }

    pub fn mark_pending(&mut self, index: &str, shard: u32) {
        self.pending.insert((index.to_string(), shard));
    }

    pub fn copies(&self, index: &str, shard: u32) -> &[StoreCopy] {
        self.copies.get(&(index.to_string(), shard)).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn is_pending(&self, index: &str, shard: u32) -> bool {
        self.pending.contains(&(index.to_string(), shard))
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// `number_of_in_flight_fetch`.
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Every copy the routing table currently has assigned, as if each data
    /// node reported its started copies (handy for tests and for a master
    /// that tracks stores from shard-started events).
    pub fn from_routing(state: &ClusterState) -> ShardStoreInfo {
        let mut s = ShardStoreInfo::default();
        for c in state.routing_table.all_shards() {
            if let (Some(n), Some(a)) = (&c.node_id, c.allocation_id()) {
                if c.active() {
                    s.add_copy(&c.index, c.shard, n, Some(a), c.primary);
                }
            }
        }
        s
    }
}

// ------------------------------------------------------- transition input --

/// A `shard-started` report from a data node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartedShard {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
}

/// A `shard-failed` report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedShard {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
    pub message: String,
    pub failure: Option<String>,
    /// ES `markAsStale`: the primary failed this copy while replicating, so
    /// it misses acknowledged writes and must leave the in-sync set.
    pub mark_as_stale: bool,
}

/// Indices to block / unblock after a disk check (ES `DiskThresholdMonitor`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FloodStageActions {
    /// Need `index.blocks.read_only_allow_delete: true`.
    pub block: BTreeSet<String>,
    /// Carry the block but every node holding them is below the high watermark.
    pub release: BTreeSet<String>,
}

// -------------------------------------------------------------- service --

/// Stateless apart from the node's static settings (`elasticsearch.yml` /
/// `-E`), which sit below cluster settings in precedence.
#[derive(Clone, Debug, Default)]
pub struct AllocationService {
    node_settings: Map<String, Value>,
}

static EMPTY_INFO: std::sync::LazyLock<ClusterInfo> = std::sync::LazyLock::new(ClusterInfo::default);
static EMPTY_STORES: std::sync::LazyLock<ShardStoreInfo> = std::sync::LazyLock::new(ShardStoreInfo::default);

impl AllocationService {
    pub fn new(node_settings: Map<String, Value>) -> AllocationService {
        AllocationService { node_settings }
    }

    /// Effective allocation settings for `state`.
    pub fn settings(&self, state: &ClusterState) -> AllocSettings {
        AllocSettings::resolve(&settings::SettingsView { metadata: &state.metadata, node: &self.node_settings })
    }

    fn alloc<'a>(&self, state: &ClusterState, now: i64, info: &'a ClusterInfo, stores: &'a ShardStoreInfo) -> Allocation<'a> {
        Allocation::new(state.clone(), self.settings(state), info, stores, now)
    }

    fn bare<'a>(&self, state: &ClusterState, now: i64) -> Allocation<'a> {
        self.alloc(state, now, &EMPTY_INFO, &EMPTY_STORES)
    }

    // ---------------------------------------------------- shard lifecycle --

    /// INITIALIZING → STARTED; finishes relocations; adds the copy to the
    /// in-sync set. Unknown or stale reports are ignored.
    pub fn apply_started_shards(&self, state: &ClusterState, started: &[StartedShard]) -> ClusterState {
        let mut a = self.bare(state, 0);
        for s in started {
            a.start(&s.index, s.shard, &s.allocation_id);
        }
        finish(state, a.state)
    }

    /// Fails copies (ES `applyFailedShards`): `failed_allocations` + 1, a
    /// failed primary promotes the most suitable in-sync active replica
    /// (primary term + 1), relocations are cancelled as ES does.
    pub fn apply_failed_shards(&self, state: &ClusterState, failed: &[FailedShard], now_millis: i64) -> ClusterState {
        let mut a = self.bare(state, now_millis);
        for f in failed {
            let Some(pos) = a.pos_by_aid(&f.index, f.shard, &f.allocation_id) else {
                // ES `removeStaleIdsWithoutRoutings`: the primary marks an
                // in-sync copy that has no routing entry (its node left) as
                // stale; never empty the set.
                if f.mark_as_stale {
                    if let Some(set) = a.state.metadata.indices.get_mut(&f.index).and_then(|m| m.in_sync_allocation_ids.get_mut(&f.shard)) {
                        if set.len() > 1 {
                            set.remove(&f.allocation_id);
                        }
                    }
                }
                continue;
            };
            let Some(c) = a.table(&f.index, f.shard).map(|t| t.copies[pos].clone()) else { continue };
            let node = c.node_id.clone().unwrap_or_default();
            // A segment format refusal is not counted (see `max_retry`).
            let refused = crate::compatibility::refused_segment_format(&f.message).is_some()
                || f.failure.as_deref().is_some_and(|e| crate::compatibility::refused_segment_format(e).is_some());
            let previous = c.unassigned_info.as_ref().and_then(|u| u.details.as_deref());
            // The first failure after a refusal starts the count afresh: an
            // older master counted every refusal, and those counts must not
            // add up with this one to block the copy everywhere. Only a
            // refusal itself resets it (a carried tag does not), so real
            // failures still reach max_retry.
            let base = if previous.is_some_and(crate::compatibility::is_segment_format_refusal) { 0 }
                else { c.unassigned_info.as_ref().map(|u| u.failed_allocations).unwrap_or(0) };
            let failed_allocations = base + u32::from(!refused);
            let mut details = format!("failed shard on node [{node}]: {}", f.message);
            if let Some(e) = &f.failure {
                details.push_str(&format!(", failure {e}"));
            }
            let details = crate::compatibility::carry_segment_format_refusal(previous, details);
            let info = UnassignedInfo { reason: UnassignedReason::AllocationFailed, at_millis: now_millis, failed_allocations, delayed: false, details: Some(details), last_allocation_status: "no_attempt".into(), last_node: Some(node) };
            a.fail_shard(&f.index, f.shard, &f.allocation_id, info, f.mark_as_stale);
        }
        finish(state, a.state)
    }

    /// Copies on nodes that are gone (or no longer data nodes) become
    /// unassigned with `NODE_LEFT` (`NODE_RESTARTING` for a node registered
    /// for restart); they stay in the in-sync set. Replicas are delayed by
    /// `index.unassigned.node_left.delayed_timeout` (default 1m).
    pub fn disassociate_dead_nodes(&self, state: &ClusterState, now_millis: i64) -> ClusterState {
        let mut a = self.bare(state, now_millis);
        let mut dead: Vec<(String, String, u32, String)> = Vec::new();
        for c in state.routing_table.all_shards() {
            if let (Some(n), Some(aid)) = (&c.node_id, c.allocation_id()) {
                if !state.nodes.get(n).is_some_and(|x| x.is_data()) {
                    dead.push((n.clone(), c.index.clone(), c.shard, aid.to_string()));
                }
            }
        }
        dead.sort();
        for (node, index, shard, aid) in dead {
            let Some(pos) = a.pos_by_aid(&index, shard, &aid) else { continue };
            let Some(c) = a.table(&index, shard).map(|t| t.copies[pos].clone()) else { continue };
            if c.node_id.as_deref() != Some(node.as_str()) {
                continue;
            }
            let restarting = context::shutdown_info(&state.metadata, &node).is_some_and(|s| s.kind == "restart");
            let probe = UnassignedInfo { reason: if restarting { UnassignedReason::NodeRestarting } else { UnassignedReason::NodeLeft }, at_millis: now_millis, failed_allocations: 0, delayed: false, details: Some(format!("node_left [{node}]")), last_allocation_status: "no_attempt".into(), last_node: Some(node.clone()) };
            let mut info = probe.clone();
            let mut tmp = c.clone();
            tmp.unassigned_info = Some(probe);
            info.delayed = a.configured_delay(&tmp) > 0;
            a.fail_shard(&index, shard, &aid, info, false);
        }
        finish(state, a.state)
    }

    /// `retry_failed`: reset `failed_allocations` of unassigned copies
    /// (reason becomes `MANUAL_ALLOCATION`), so `max_retry` lets them through.
    pub fn reset_failed_allocations(&self, state: &ClusterState, now_millis: i64) -> ClusterState {
        let mut next = state.clone();
        lifecycle::reset_failed_counters(&mut next, now_millis);
        next
    }

    // ------------------------------------------------------------ reroute --

    /// ES `AllocationService.reroute`.
    pub fn reroute(&self, state: &ClusterState, now_millis: i64, info: &ClusterInfo, stores: &ShardStoreInfo) -> ClusterState {
        let mut a = self.alloc(state, now_millis, info, stores);
        a.run_reroute();
        finish(state, a.state)
    }

    /// Prepare one pressure-driven move with ordinary automatic-allocation
    /// deciders (unlike an explicit reroute command, enable is not ignored).
    /// The caller must reserve target memory before publishing this proposal.
    pub fn pressure_relocation(&self,state:&ClusterState,index:&str,shard:u32,source_aid:&str,target:&str,now_millis:i64,info:&ClusterInfo,stores:&ShardStoreInfo)->Option<ClusterState>{
        let mut a=self.alloc(state,now_millis,info,stores);
        if a.can_rebalance().kind!=deciders::DecisionType::Yes{return None;}
        if state.routing_table.all_shards().any(|copy|copy.state==crate::state::ShardState::Relocating){return None;}
        let copy=a.table(index,shard)?.copies.iter().find(|copy|copy.allocation_id()==Some(source_aid))?.clone();
        if copy.state!=crate::state::ShardState::Started || a.can_rebalance_shard(&copy).kind!=deciders::DecisionType::Yes{return None;}
        let node=a.node(target)?;
        if a.can_allocate(&copy,node).kind!=deciders::DecisionType::Yes{return None;}
        let id=a.relocate(index,shard,source_aid,target)?;
        // Persist the admission-required marker in the existing opaque ID
        // field. A target restart/expired grant cannot silently recover this
        // allocation as ordinary unreserved work.
        let admitted_id=format!("capacity-{}-{id}",now_millis.max(0));
        let table=a.state.routing_table.indices.get_mut(index)?.shards.iter_mut().find(|table|table.shard==shard)?;
        for copy in &mut table.copies{if let Some(allocation)=&mut copy.allocation_id{
            if allocation.id==id{allocation.id=admitted_id.clone();}
            if allocation.relocation_id.as_deref()==Some(id.as_str()){allocation.relocation_id=Some(admitted_id.clone());}
        }}
        Some(finish(state,a.state))
    }

    /// When the next delayed replica's delay expires (absolute millis), so
    /// the master can schedule a reroute (ES `DelayedAllocationService`).
    pub fn next_delayed_allocation_millis(&self, state: &ClusterState, now_millis: i64) -> Option<i64> {
        let a = self.bare(state, now_millis);
        state.routing_table.all_shards().filter(|c| c.state == ShardState::Unassigned && c.unassigned_info.as_ref().is_some_and(|u| u.delayed)).map(|c| now_millis + a.remaining_delay(c)).min()
    }

    /// `_cluster/reroute` with commands.
    pub fn reroute_commands(&self, state: &ClusterState, body: &Value, flags: RerouteFlags, now_millis: i64, info: &ClusterInfo, stores: &ShardStoreInfo) -> Result<RerouteCommandsResult, ApiError> {
        let a = self.alloc(state, now_millis, info, stores);
        let (next, explanations) = commands::execute(a, body, flags)?;
        Ok(RerouteCommandsResult { state: finish(state, next), explanations })
    }

    /// `_cluster/allocation/explain`.
    pub fn explain(&self, state: &ClusterState, request: &Value, opts: ExplainOptions, now_millis: i64, info: &ClusterInfo, stores: &ShardStoreInfo) -> Result<Value, ApiError> {
        let a = self.alloc(state, now_millis, info, stores);
        explain::explain(a, request, opts)
    }

    /// Shard-migration part of `GET _nodes/{id}/shutdown` (ES
    /// `TransportGetShutdownStatusAction.shardMigrationStatus`).
    pub fn shutdown_migration_status(&self, state: &ClusterState, node_id: &str, shutdown_type: &str, node_seen: bool, now_millis: i64, info: &ClusterInfo) -> ShardMigrationStatus {
        shutdown::migration_status(|| self.alloc(state, now_millis, info, &EMPTY_STORES), node_id, shutdown_type, node_seen)
    }

    // ----------------------------------------------------- index lifecycle --

    /// Routing for a newly created index (metadata already in `state`): every
    /// copy UNASSIGNED / `INDEX_CREATED`, primaries `EMPTY_STORE`, replicas `PEER`.
    pub fn add_index_routing(&self, state: &ClusterState, index: &str, now_millis: i64) -> Result<ClusterState, String> {
        lifecycle::add_routing(state, index, UnassignedReason::IndexCreated, None, now_millis)
    }

    /// Routing for an index restored from a snapshot: primaries recover from
    /// `source` (a `RecoverySource::Snapshot`), reason `NEW_INDEX_RESTORED`
    /// or `EXISTING_INDEX_RESTORED`.
    pub fn add_restored_index_routing(&self, state: &ClusterState, index: &str, source: RecoverySource, is_new_index: bool, now_millis: i64) -> Result<ClusterState, String> {
        let reason = if is_new_index { UnassignedReason::NewIndexRestored } else { UnassignedReason::ExistingIndexRestored };
        let mut s = state.clone();
        s.routing_table.indices.remove(index);
        lifecycle::add_routing(&s, index, reason, Some(source), now_millis)
    }

    /// Removes the index's routing, metadata and blocks.
    pub fn delete_index(&self, state: &ClusterState, index: &str) -> ClusterState {
        let mut s = state.clone();
        s.routing_table.indices.remove(index);
        s.metadata.indices.remove(index);
        s.blocks.indices.remove(index);
        s
    }

    /// Close: `IndexState::Close`, `index.verified_before_close`, the
    /// index-closed block, and routing re-created as `INDEX_CLOSED` (shards
    /// re-allocate onto their in-sync copies, ES 7.2+ replicated closed indices).
    pub fn close_index(&self, state: &ClusterState, index: &str, now_millis: i64) -> Result<ClusterState, String> {
        lifecycle::set_index_state(state, index, IndexState::Close, now_millis)
    }

    pub fn open_index(&self, state: &ClusterState, index: &str, now_millis: i64) -> Result<ClusterState, String> {
        lifecycle::set_index_state(state, index, IndexState::Open, now_millis)
    }

    /// `index.number_of_replicas` update (adds `REPLICA_ADDED` copies or
    /// removes replicas, unassigned ones first).
    pub fn update_number_of_replicas(&self, state: &ClusterState, indices: &[String], replicas: u32, now_millis: i64) -> Result<ClusterState, String> {
        let mut s = state.clone();
        for i in indices {
            if !s.metadata.indices.contains_key(i) {
                return Err(format!("no such index [{i}]"));
            }
            lifecycle::set_replicas(&mut s, i, replicas, now_millis);
        }
        Ok(finish(state, s))
    }

    // ------------------------------------------------------------- disk --

    /// Flood-stage check (ES `DiskThresholdMonitor`).
    pub fn disk_flood_stage(&self, state: &ClusterState, info: &ClusterInfo) -> FloodStageActions {
        let s = self.settings(state);
        let mut out = FloodStageActions::default();
        if !s.disk_threshold_enabled {
            return out;
        }
        let above = |node: &str, wm: &settings::Watermark| info.nodes.get(node).is_some_and(|u| u.free_bytes < wm.required_free(u.total_bytes).0);
        let blocked = |m: &IndexMetadata| settings::index_bool(m, "index.blocks.read_only_allow_delete", false);
        let mut nodes_of: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for c in state.routing_table.all_shards() {
            if let Some(n) = &c.node_id {
                nodes_of.entry(c.index.clone()).or_default().insert(n.clone());
            }
        }
        for (index, nodes) in &nodes_of {
            let Some(m) = state.metadata.indices.get(index) else { continue };
            if nodes.iter().any(|n| above(n, &s.flood)) {
                if !blocked(m) {
                    out.block.insert(index.clone());
                }
            } else if blocked(m) && nodes.iter().all(|n| info.nodes.contains_key(n) && !above(n, &s.high)) {
                out.release.insert(index.clone());
            }
        }
        out
    }
}

/// `UnassignedInfo.Reason` name (`NODE_LEFT`, ...).
pub fn reason_str(r: UnassignedReason) -> String {
    deciders::reason_name(r)
}

/// Sets or clears `index.blocks.read_only_allow_delete` (setting + block).
pub fn apply_read_only_allow_delete(state: &ClusterState, indices: &BTreeSet<String>, block: bool) -> ClusterState {
    let mut s = state.clone();
    for i in indices {
        let Some(m) = s.metadata.indices.get_mut(i) else { continue };
        if block {
            m.settings.insert("index.blocks.read_only_allow_delete".into(), Value::String("true".into()));
            s.blocks.indices.entry(i.clone()).or_default().insert(ClusterBlock::read_only_allow_delete());
        } else {
            m.settings.remove("index.blocks.read_only_allow_delete");
            if let Some(b) = s.blocks.indices.get_mut(i) {
                b.retain(|x| x.id != 12);
                if b.is_empty() {
                    s.blocks.indices.remove(i);
                }
            }
        }
        m.settings_version += 1;
        m.version += 1;
    }
    s
}

/// ES `MasterService.patchVersions`: bump `routing_table.version` /
/// `metadata.version` of `next` if they changed relative to `previous`.
pub fn patch_versions(previous: &ClusterState, next: &mut ClusterState) {
    let mut rt = next.routing_table.clone();
    rt.version = previous.routing_table.version;
    if rt != previous.routing_table {
        next.routing_table.version = previous.routing_table.version + 1;
    }
    let mut md = next.metadata.clone();
    md.version = previous.metadata.version;
    if md != previous.metadata {
        next.metadata.version = previous.metadata.version + 1;
    }
}

/// Bump `IndexMetadata.version` of indices whose metadata allocation changed.
fn finish(before: &ClusterState, mut after: ClusterState) -> ClusterState {
    for (name, m) in after.metadata.indices.iter_mut() {
        if let Some(old) = before.metadata.indices.get(name) {
            if old != m && old.version == m.version {
                m.version += 1;
            }
        }
    }
    after
}
