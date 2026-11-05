//! The mutable working copy one allocation round operates on (ES's
//! `RoutingAllocation` + `RoutingNodes`), and the routing-table mutations
//! with ES's exact semantics (`RoutingNodes.initializeShard`,
//! `relocateShard`, `startShard`, `failShard`, promotion) plus the in-sync /
//! primary-term bookkeeping ES does in `IndexMetadataUpdater`.
//!
//! Copies are addressed by allocation id once assigned (positions shift when
//! relocation targets are removed) and by position while unassigned.

use super::settings::{AllocSettings, index_setting, parse_time_millis};
use super::{ClusterInfo, ShardStoreInfo};
use crate::state::*;
use std::collections::{BTreeSet, HashMap};

/// Which decider results are recorded (ES `RoutingAllocation.DebugMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugMode {
    /// Short-circuit on the first NO; record only non-YES decisions.
    Off,
    /// Evaluate every decider; record only non-YES decisions (explain default).
    ExcludeYes,
    /// Evaluate and record everything (`include_yes_decisions`, commands).
    On,
}

/// Per-node counters derived from the routing table; recomputed after every
/// mutation (cheap relative to decider evaluation, and impossible to drift).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Shards per node, not counting RELOCATING sources (their target counts).
    pub node_shards: HashMap<String, usize>,
    pub node_index_shards: HashMap<(String, String), usize>,
    /// Highest primary shard id of an index on a node (balancer tie-break).
    pub highest_primary: HashMap<(String, String), i64>,
    pub incoming: HashMap<String, usize>,
    pub outgoing: HashMap<String, usize>,
    pub initial_primaries: HashMap<String, usize>,
    pub incoming_relocation_bytes: HashMap<String, u64>,
    pub outgoing_relocation_bytes: HashMap<String, u64>,
    pub relocating: usize,
    pub unassigned: usize,
    pub unassigned_primaries: usize,
    pub inactive_primaries: usize,
    pub inactive_shards: usize,
}

pub struct Allocation<'a> {
    pub state: ClusterState,
    pub s: AllocSettings,
    pub info: &'a ClusterInfo,
    pub stores: &'a ShardStoreInfo,
    pub now: i64,
    pub debug: DebugMode,
    /// Manual reroute commands bypass `enable` (ES `ignoreDisable`).
    pub ignore_disable: bool,
    /// Set once the unassigned-allocation pass has run: shards still
    /// unassigned are then on ES's "ignored" list and no longer count as
    /// unassigned for `cluster.routing.allocation.allow_rebalance`.
    pub unassigned_ignored: bool,
    pub stats: Stats,
}

pub fn is_relocation_target(c: &ShardRouting) -> bool {
    c.state == ShardState::Initializing && c.relocating_node_id.is_some()
}

/// `[index][shard]`
pub fn shard_id_str(index: &str, shard: u32) -> String {
    format!("[{index}][{shard}]")
}

impl<'a> Allocation<'a> {
    pub fn new(state: ClusterState, s: AllocSettings, info: &'a ClusterInfo, stores: &'a ShardStoreInfo, now: i64) -> Allocation<'a> {
        let mut a = Allocation { state, s, info, stores, now, debug: DebugMode::Off, ignore_disable: false, unassigned_ignored: false, stats: Stats::default() };
        a.refresh();
        a
    }

    // ------------------------------------------------------------ lookups --

    pub fn meta(&self, index: &str) -> Option<&IndexMetadata> {
        self.state.metadata.indices.get(index)
    }

    pub fn node(&self, id: &str) -> Option<&DiscoveryNode> {
        self.state.nodes.get(id)
    }

    /// Data nodes in id order: ES's `RoutingNodes` (the allocation model).
    pub fn data_nodes(&self) -> Vec<DiscoveryNode> {
        self.state.data_nodes().cloned().collect()
    }

    pub fn table(&self, index: &str, shard: u32) -> Option<&IndexShardRoutingTable> {
        self.state.routing_table.shard_table(index, shard)
    }

    fn table_mut(&mut self, index: &str, shard: u32) -> Option<&mut IndexShardRoutingTable> {
        self.state.routing_table.indices.get_mut(index).and_then(|i| i.shards.get_mut(shard as usize))
    }

    pub fn active_primary(&self, index: &str, shard: u32) -> Option<&ShardRouting> {
        self.table(index, shard).and_then(|t| t.copies.iter().find(|c| c.primary && c.active()))
    }

    pub fn pos_by_aid(&self, index: &str, shard: u32, aid: &str) -> Option<usize> {
        self.table(index, shard)?.copies.iter().position(|c| c.allocation_id() == Some(aid))
    }

    pub fn shard_size(&self, c: &ShardRouting) -> u64 {
        c.expected_shard_size_in_bytes.or_else(|| self.info.shard_size(&c.index, c.shard, c.primary)).unwrap_or(0)
    }

    /// Expected size of a shard about to be allocated (ES `getExpectedShardSize`):
    /// new empty shards are 0, otherwise the primary's known size.
    pub fn expected_size(&self, c: &ShardRouting) -> u64 {
        match c.recovery_source {
            Some(RecoverySource::EmptyStore) => 0,
            _ => self.info.shard_size(&c.index, c.shard, c.primary).or_else(|| self.info.shard_size(&c.index, c.shard, true)).unwrap_or(0),
        }
    }

    /// `index.unassigned.node_left.delayed_timeout` (default 1m), extended
    /// to a known restart's `allocation_delay` (ES `getRemainingDelay`).
    pub fn configured_delay(&self, c: &ShardRouting) -> i64 {
        let index_delay = self.meta(&c.index).and_then(|m| index_setting(m, "index.unassigned.node_left.delayed_timeout")).and_then(|s| parse_time_millis(&s)).unwrap_or(60_000);
        let restart = c.unassigned_info.as_ref().filter(|u| u.reason == UnassignedReason::NodeRestarting).and_then(|u| u.last_node.as_deref()).and_then(|n| shutdown_info(&self.state.metadata, n)).filter(|s| s.kind == "restart").map(|s| s.allocation_delay_millis);
        match restart {
            Some(d) => d.max(index_delay),
            None => index_delay,
        }
    }

    pub fn remaining_delay(&self, c: &ShardRouting) -> i64 {
        let Some(u) = &c.unassigned_info else { return 0 };
        (self.configured_delay(c) - (self.now - u.at_millis)).max(0)
    }

    // ------------------------------------------------------------- stats --

    pub fn refresh(&mut self) {
        let mut st = Stats::default();
        for irt in self.state.routing_table.indices.values() {
            for t in &irt.shards {
                let primary_node = t.copies.iter().find(|c| c.primary && c.active()).and_then(|c| c.node_id.clone());
                for c in &t.copies {
                    let size = c.expected_shard_size_in_bytes.or_else(|| self.info.shard_size(&c.index, c.shard, c.primary)).unwrap_or(0);
                    match c.state {
                        ShardState::Unassigned => {
                            st.unassigned += 1;
                            if c.primary {
                                st.unassigned_primaries += 1;
                            }
                            continue;
                        }
                        ShardState::Initializing => {
                            let Some(n) = &c.node_id else { continue };
                            *st.incoming.entry(n.clone()).or_default() += 1;
                            if is_relocation_target(c) {
                                *st.incoming_relocation_bytes.entry(n.clone()).or_default() += size;
                            } else {
                                st.inactive_shards += 1;
                                if c.primary {
                                    st.inactive_primaries += 1;
                                    *st.initial_primaries.entry(n.clone()).or_default() += 1;
                                }
                            }
                            if c.recovery_source == Some(RecoverySource::Peer) {
                                if let Some(p) = &primary_node {
                                    *st.outgoing.entry(p.clone()).or_default() += 1;
                                }
                            }
                        }
                        ShardState::Relocating => {
                            st.relocating += 1;
                            if let Some(n) = &c.node_id {
                                *st.outgoing_relocation_bytes.entry(n.clone()).or_default() += size;
                            }
                            continue;
                        }
                        ShardState::Started => {}
                    }
                    let Some(n) = &c.node_id else { continue };
                    *st.node_shards.entry(n.clone()).or_default() += 1;
                    *st.node_index_shards.entry((n.clone(), c.index.clone())).or_default() += 1;
                    if c.primary {
                        let e = st.highest_primary.entry((n.clone(), c.index.clone())).or_insert(-1);
                        *e = (*e).max(c.shard as i64);
                    }
                }
            }
        }
        self.stats = st;
    }

    // --------------------------------------------------------- mutations --

    fn bump_term(&mut self, index: &str, shard: u32) {
        if let Some(m) = self.state.metadata.indices.get_mut(index) {
            let n = m.number_of_shards as usize;
            if m.primary_terms.len() < n {
                m.primary_terms.resize(n, 0);
            }
            if let Some(t) = m.primary_terms.get_mut(shard as usize) {
                *t += 1;
            }
        }
    }

    fn in_sync_mut(&mut self, index: &str, shard: u32) -> Option<&mut BTreeSet<String>> {
        self.state.metadata.indices.get_mut(index).map(|m| m.in_sync_allocation_ids.entry(shard).or_default())
    }

    /// `RoutingNodes.initializeShard` for the unassigned copy at `pos`.
    /// `aid` reuses an existing store's allocation id (primary from existing
    /// store); otherwise a fresh id is generated.
    pub fn initialize(&mut self, index: &str, shard: u32, pos: usize, node: &str, aid: Option<String>, source: Option<RecoverySource>, info: Option<UnassignedInfo>) -> Option<String> {
        let in_sync_empty = self.meta(index).map(|m| m.in_sync_allocation_ids.get(&shard).is_none_or(|s| s.is_empty())).unwrap_or(true);
        let expected = {
            let c = self.table(index, shard)?.copies.get(pos)?;
            if c.state != ShardState::Unassigned {
                return None;
            }
            self.expected_size(c)
        };
        let id = aid.unwrap_or_else(random_id);
        let c = self.table_mut(index, shard)?.copies.get_mut(pos)?;
        c.state = ShardState::Initializing;
        c.node_id = Some(node.to_string());
        c.relocating_node_id = None;
        c.allocation_id = Some(AllocationId { id: id.clone(), relocation_id: None });
        if let Some(src) = source {
            c.recovery_source = Some(src);
        }
        if c.recovery_source.is_none() {
            c.recovery_source = Some(if !c.primary {
                RecoverySource::Peer
            } else if in_sync_empty {
                RecoverySource::EmptyStore
            } else {
                RecoverySource::ExistingStore { bootstrap_new_history_uuid: false }
            });
        }
        if let Some(i) = info {
            c.unassigned_info = Some(i);
        }
        if let Some(u) = c.unassigned_info.as_mut() {
            u.delayed = false;
        }
        c.expected_shard_size_in_bytes = (expected > 0).then_some(expected);
        let primary = c.primary;
        let src = c.recovery_source.clone();
        if primary {
            self.bump_term(index, shard);
            // IndexMetadataUpdater: forced empty / stale / restored primaries
            // replace the in-sync set.
            match src {
                // A forced empty primary (in-sync ids exist) replaces the set
                // with itself (ES clears it; we keep it non-empty so no older
                // copy can ever be promoted again). New indices have an empty
                // set and get their first id when the primary starts.
                Some(RecoverySource::EmptyStore) => {
                    if let Some(s) = self.in_sync_mut(index, shard) {
                        if !s.is_empty() {
                            *s = BTreeSet::from([id.clone()]);
                        }
                    }
                }
                Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: true }) | Some(RecoverySource::Snapshot { .. }) | Some(RecoverySource::LocalShards) => {
                    if let Some(s) = self.in_sync_mut(index, shard) {
                        *s = BTreeSet::from([id.clone()]);
                    }
                }
                _ => {}
            }
        }
        self.refresh();
        Some(id)
    }

    /// `RoutingNodes.relocateShard`: STARTED source → RELOCATING plus an
    /// INITIALIZING target on `target_node`.
    pub fn relocate(&mut self, index: &str, shard: u32, aid: &str, target_node: &str) -> Option<String> {
        let pos = self.pos_by_aid(index, shard, aid)?;
        let size = {
            let c = &self.table(index, shard)?.copies[pos];
            if c.state != ShardState::Started {
                return None;
            }
            self.shard_size(c)
        };
        let target_id = random_id();
        let t = self.table_mut(index, shard)?;
        let src = &mut t.copies[pos];
        src.state = ShardState::Relocating;
        src.relocating_node_id = Some(target_node.to_string());
        src.allocation_id = Some(AllocationId { id: aid.to_string(), relocation_id: Some(target_id.clone()) });
        src.expected_shard_size_in_bytes = Some(size);
        let target = ShardRouting {
            index: index.to_string(),
            shard,
            primary: src.primary,
            state: ShardState::Initializing,
            node_id: Some(target_node.to_string()),
            relocating_node_id: src.node_id.clone(),
            allocation_id: Some(AllocationId { id: target_id.clone(), relocation_id: Some(aid.to_string()) }),
            recovery_source: Some(RecoverySource::Peer),
            unassigned_info: None,
            expected_shard_size_in_bytes: Some(size),
        };
        t.copies.push(target);
        self.refresh();
        Some(target_id)
    }

    /// Turn a RELOCATING source back into a plain STARTED copy (target removed separately).
    fn cancel_relocation_at(&mut self, index: &str, shard: u32, pos: usize) {
        if let Some(c) = self.table_mut(index, shard).and_then(|t| t.copies.get_mut(pos)) {
            c.state = ShardState::Started;
            c.relocating_node_id = None;
            if let Some(a) = c.allocation_id.as_mut() {
                a.relocation_id = None;
            }
            c.expected_shard_size_in_bytes = None;
        }
    }

    fn remove_at(&mut self, index: &str, shard: u32, pos: usize) {
        if let Some(t) = self.table_mut(index, shard) {
            if pos < t.copies.len() {
                t.copies.remove(pos);
            }
        }
    }

    /// `ShardRouting.moveToUnassigned`.
    fn move_to_unassigned(&mut self, index: &str, shard: u32, pos: usize, info: UnassignedInfo) {
        if let Some(c) = self.table_mut(index, shard).and_then(|t| t.copies.get_mut(pos)) {
            let was_active = c.active();
            if was_active || c.recovery_source.is_none() {
                c.recovery_source = Some(if c.primary { RecoverySource::ExistingStore { bootstrap_new_history_uuid: false } } else { RecoverySource::Peer });
            }
            c.state = ShardState::Unassigned;
            c.node_id = None;
            c.relocating_node_id = None;
            c.allocation_id = None;
            c.unassigned_info = Some(info);
            c.expected_shard_size_in_bytes = None;
        }
    }

    /// `RoutingNodes.startShard` + the in-sync bookkeeping of `shardStarted`
    /// / `relocationCompleted`. Returns false for unknown / non-initializing ids.
    pub fn start(&mut self, index: &str, shard: u32, aid: &str) -> bool {
        let Some(pos) = self.pos_by_aid(index, shard, aid) else { return false };
        let Some(c) = self.table(index, shard).map(|t| t.copies[pos].clone()) else { return false };
        if c.state != ShardState::Initializing {
            return false;
        }
        let mut removed_source = None;
        if is_relocation_target(&c) {
            if let Some(src_aid) = c.allocation_id.as_ref().and_then(|a| a.relocation_id.clone()) {
                if let Some(sp) = self.pos_by_aid(index, shard, &src_aid) {
                    self.remove_at(index, shard, sp);
                    removed_source = Some(src_aid);
                }
            }
        }
        let Some(pos) = self.pos_by_aid(index, shard, aid) else { return false };
        if let Some(t) = self.table_mut(index, shard) {
            let s = &mut t.copies[pos];
            s.state = ShardState::Started;
            s.relocating_node_id = None;
            if let Some(a) = s.allocation_id.as_mut() {
                a.relocation_id = None;
            }
            s.recovery_source = None;
            s.unassigned_info = None;
            s.expected_shard_size_in_bytes = None;
        }
        let max_active = self.meta(index).map(|m| m.number_of_replicas as usize + 1).unwrap_or(1);
        let assigned: BTreeSet<String> = self.table(index, shard).map(|t| t.copies.iter().filter(|x| !is_relocation_target(x)).filter_map(|x| x.allocation_id().map(String::from)).collect()).unwrap_or_default();
        if let Some(set) = self.in_sync_mut(index, shard) {
            let old_len = set.len();
            set.insert(aid.to_string());
            if let Some(r) = &removed_source {
                set.remove(r);
            }
            // Bound the set to the possible number of active copies,
            // keeping ids that still have routing entries.
            if set.len() > old_len && set.len() > max_active {
                let mut ids: Vec<String> = set.iter().cloned().collect();
                ids.sort_by_key(|i| !assigned.contains(i));
                *set = ids.into_iter().take(max_active).collect();
            }
        }
        // A finished primary relocation re-initialises replicas that were
        // recovering from the old primary.
        if c.primary && removed_source.is_some() {
            let reinit: Vec<String> = self.table(index, shard).map(|t| t.copies.iter().filter(|x| !x.primary && x.state == ShardState::Initializing).filter_map(|x| x.allocation_id().map(String::from)).collect()).unwrap_or_default();
            for old in reinit {
                let new_id = random_id();
                if let Some(t) = self.table_mut(index, shard) {
                    for x in t.copies.iter_mut() {
                        if let Some(a) = x.allocation_id.as_mut() {
                            if a.id == old {
                                a.id = new_id.clone();
                            } else if a.relocation_id.as_deref() == Some(old.as_str()) {
                                a.relocation_id = Some(new_id.clone());
                            }
                        }
                    }
                }
            }
        }
        normalize_order(self.table_mut(index, shard));
        self.refresh();
        true
    }

    /// `RoutingNodes.failShard` (ES semantics, including failing initializing
    /// replicas of a failed primary, cancelling relocations and promoting the
    /// most suitable in-sync active replica). `mark_stale` removes the copy
    /// from the in-sync set unless that would leave no active copy.
    pub fn fail_shard(&mut self, index: &str, shard: u32, aid: &str, info: UnassignedInfo, mark_stale: bool) -> bool {
        let Some(pos) = self.pos_by_aid(index, shard, aid) else { return false };
        let Some(failed) = self.table(index, shard).map(|t| t.copies[pos].clone()) else { return false };
        if failed.state == ShardState::Unassigned {
            return false;
        }
        if failed.primary && !is_relocation_target(&failed) {
            let inits: Vec<(String, Option<String>)> = self.table(index, shard).map(|t| t.copies.iter().filter(|c| !c.primary && c.state == ShardState::Initializing).filter_map(|c| c.allocation_id().map(|a| (a.to_string(), c.node_id.clone()))).collect()).unwrap_or_default();
            for (rid, node) in inits {
                let pi = UnassignedInfo {
                    reason: UnassignedReason::PrimaryFailed,
                    at_millis: info.at_millis,
                    failed_allocations: 0,
                    delayed: false,
                    details: Some("primary failed while replica initializing".into()),
                    last_allocation_status: "no_attempt".into(),
                    last_node: node,
                };
                self.fail_shard(index, shard, &rid, pi, false);
            }
        }
        if failed.state == ShardState::Relocating {
            if let Some(tid) = failed.allocation_id.as_ref().and_then(|a| a.relocation_id.clone()) {
                if let Some(tp) = self.pos_by_aid(index, shard, &tid) {
                    if failed.primary {
                        self.remove_at(index, shard, tp);
                    } else if let Some(t) = self.table_mut(index, shard).and_then(|t| t.copies.get_mut(tp)) {
                        // the target carries on as a plain initializing replica
                        t.relocating_node_id = None;
                        if let Some(a) = t.allocation_id.as_mut() {
                            a.relocation_id = None;
                        }
                    }
                }
            }
        }
        let Some(pos) = self.pos_by_aid(index, shard, aid) else { return false };
        let mut failed_active_primary = false;
        match failed.state {
            ShardState::Initializing if is_relocation_target(&failed) => {
                if let Some(src) = failed.allocation_id.as_ref().and_then(|a| a.relocation_id.clone()) {
                    if let Some(sp) = self.pos_by_aid(index, shard, &src) {
                        self.cancel_relocation_at(index, shard, sp);
                    }
                }
                if let Some(pos) = self.pos_by_aid(index, shard, aid) {
                    self.remove_at(index, shard, pos);
                }
            }
            ShardState::Initializing => {
                if failed.primary {
                    self.unassign_primary_and_promote(index, shard, pos, info);
                } else {
                    self.move_to_unassigned(index, shard, pos, info);
                }
            }
            _ => {
                if failed.primary {
                    failed_active_primary = true;
                    self.unassign_primary_and_promote(index, shard, pos, info);
                } else if failed.state == ShardState::Relocating {
                    self.remove_at(index, shard, pos);
                } else {
                    self.move_to_unassigned(index, shard, pos, info);
                }
            }
        }
        // Only a copy failed by the primary while replicating (`mark_stale`)
        // leaves the in-sync set; node-left / recovery failures keep it, as
        // the data may come back. The set is never emptied, and a failed
        // active primary with no active copy left stays in it (ES).
        if mark_stale {
            let any_active = self.table(index, shard).is_some_and(|t| t.copies.iter().any(|c| c.active()));
            if let Some(set) = self.in_sync_mut(index, shard) {
                let keep = (!any_active && failed_active_primary) || (set.len() == 1 && set.contains(aid));
                if !keep {
                    set.remove(aid);
                }
            }
        }
        normalize_order(self.table_mut(index, shard));
        self.refresh();
        true
    }

    fn unassign_primary_and_promote(&mut self, index: &str, shard: u32, pos: usize, info: UnassignedInfo) {
        let in_sync = self.meta(index).and_then(|m| m.in_sync_allocation_ids.get(&shard).cloned()).unwrap_or_default();
        let candidate = self.table(index, shard).and_then(|t| {
            t.copies
                .iter()
                .filter(|c| !c.primary && c.active())
                .filter(|c| c.node_id.as_ref().is_some_and(|n| self.state.nodes.contains_key(n)))
                // A copy on a stateless search-only node is not promotable
                // (ES `ShardRouting.isPromotableToPrimary`: `SEARCH_ONLY`
                // copies never become primary): the shard waits for an
                // index node instead.
                .filter(|c| c.node_id.as_ref().and_then(|n| self.node(n)).is_some_and(|n| !n.is_stateless() || n.roles.contains(&Role::Index)))
                // only an in-sync copy may ever become primary
                .filter(|c| c.allocation_id().is_some_and(|a| in_sync.contains(a)))
                .max_by(|a, b| {
                    let va = a.node_id.as_ref().and_then(|n| self.node(n)).map(|n| parse_version(&n.version)).unwrap_or_default();
                    let vb = b.node_id.as_ref().and_then(|n| self.node(n)).map(|n| parse_version(&n.version)).unwrap_or_default();
                    // max_by keeps the last max; reverse ties to prefer the first copy
                    va.cmp(&vb).then(std::cmp::Ordering::Greater)
                })
                .and_then(|c| c.allocation_id().map(String::from))
        });
        self.move_to_unassigned(index, shard, pos, info);
        let Some(promote) = candidate else { return };
        if let Some(t) = self.table_mut(index, shard) {
            // demote the failed copy to an unassigned replica
            let failed = &mut t.copies[pos];
            failed.primary = false;
            failed.recovery_source = Some(RecoverySource::Peer);
            let target = t.copies.iter().find(|c| c.allocation_id() == Some(promote.as_str())).and_then(|c| c.allocation_id.as_ref()).and_then(|a| a.relocation_id.clone());
            for c in t.copies.iter_mut() {
                let id = c.allocation_id().map(String::from);
                if id.as_deref() == Some(promote.as_str()) || (target.is_some() && id == target && is_relocation_target(c)) {
                    c.primary = true;
                }
            }
        }
        self.bump_term(index, shard);
    }

    /// Unassigned → keep `last_allocation_status` current (ES `ignoreShard`).
    pub fn set_allocation_status(&mut self, index: &str, shard: u32, pos: usize, status: &str) {
        if let Some(u) = self.table_mut(index, shard).and_then(|t| t.copies.get_mut(pos)).and_then(|c| c.unassigned_info.as_mut()) {
            if u.last_allocation_status != status {
                u.last_allocation_status = status.to_string();
            }
        }
    }
}

/// Primary first, then replicas, then relocation targets (stable).
pub fn normalize_order(t: Option<&mut IndexShardRoutingTable>) {
    if let Some(t) = t {
        t.copies.sort_by_key(|c| if is_relocation_target(c) { 2 } else if c.primary { 0 } else { 1 });
    }
}

/// `x.y.z` → comparable triple (missing parts are 0).
pub fn parse_version(v: &str) -> (u32, u32, u32) {
    let mut it = v.split(|c: char| c == '.' || c == '-').map(|p| p.parse::<u32>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0))
}

/// One entry of the `node_shutdown` metadata custom.
#[derive(Clone, Debug)]
pub struct ShutdownInfo {
    /// `remove`, `restart`, `replace` or `sigterm` (lowercase).
    pub kind: String,
    pub allocation_delay_millis: i64,
    /// `target_node_name` of a `replace` shutdown.
    pub target_node_name: Option<String>,
}

/// Reads `metadata.customs["node_shutdown"]["nodes"][node_id]` (also
/// accepts the map without the `nodes` wrapper).
pub fn shutdown_info(meta: &Metadata, node_id: &str) -> Option<ShutdownInfo> {
    let custom = meta.customs.get("node_shutdown")?;
    let entry = custom.get("nodes").and_then(|n| n.get(node_id)).or_else(|| custom.get(node_id))?;
    Some(parse_shutdown(entry))
}

/// Every `node_shutdown` entry as (node id, info).
pub fn all_shutdowns(meta: &Metadata) -> Vec<(String, ShutdownInfo)> {
    let Some(custom) = meta.customs.get("node_shutdown") else { return vec![] };
    let map = custom.get("nodes").and_then(|n| n.as_object()).or_else(|| custom.as_object());
    map.map(|m| m.iter().filter(|(_, v)| v.is_object()).map(|(k, v)| (k.clone(), parse_shutdown(v))).collect()).unwrap_or_default()
}

fn parse_shutdown(entry: &serde_json::Value) -> ShutdownInfo {
    let kind = entry.get("type").and_then(|t| t.as_str()).unwrap_or("remove").to_ascii_lowercase();
    let allocation_delay_millis = entry.get("allocation_delay").and_then(super::settings::value_str).and_then(|s| parse_time_millis(&s)).unwrap_or(300_000);
    let target_node_name = entry.get("target_node_name").and_then(|t| t.as_str()).map(str::to_string);
    ShutdownInfo { kind, allocation_delay_millis, target_node_name }
}
