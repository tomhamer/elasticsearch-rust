//! Index-level routing transitions (ES `RoutingTable.Builder`
//! `addAsNew` / `addAsFromOpenToClose` / `addAsFromCloseToOpen` /
//! `updateNumberOfReplicas`), `auto_expand_replicas`, and the reroute
//! pipeline.

use super::context::{Allocation, is_relocation_target, normalize_order};
use super::deciders::DecisionType;
use super::settings::index_setting;
use crate::state::*;
use serde_json::Value;

fn new_info(reason: UnassignedReason, now: i64) -> UnassignedInfo {
    UnassignedInfo { reason, at_millis: now, failed_allocations: 0, delayed: false, details: None, last_allocation_status: "no_attempt".into(), last_node: None }
}

/// ES `IndexRoutingTable.Builder.initializeEmpty`: primaries recover from
/// an existing store when the shard has in-sync copies, otherwise empty.
fn initialize_empty(meta: &IndexMetadata, reason: UnassignedReason, primary_source: Option<RecoverySource>, now: i64) -> IndexRoutingTable {
    let shards = (0..meta.number_of_shards)
        .map(|s| {
            let has_in_sync = meta.in_sync_allocation_ids.get(&s).is_some_and(|ids| !ids.is_empty());
            let psrc = primary_source.clone().unwrap_or(if has_in_sync { RecoverySource::ExistingStore { bootstrap_new_history_uuid: false } } else { RecoverySource::EmptyStore });
            let mut copies = vec![ShardRouting::unassigned(&meta.name, s, true, new_info(reason, now), Some(psrc))];
            for _ in 0..meta.number_of_replicas {
                copies.push(ShardRouting::unassigned(&meta.name, s, false, new_info(reason, now), Some(RecoverySource::Peer)));
            }
            IndexShardRoutingTable { shard: s, copies }
        })
        .collect();
    IndexRoutingTable { index_uuid: meta.uuid.clone(), shards }
}

pub fn add_routing(state: &ClusterState, index: &str, reason: UnassignedReason, source: Option<RecoverySource>, now: i64) -> Result<ClusterState, String> {
    let mut s = state.clone();
    let meta = s.metadata.indices.get_mut(index).ok_or_else(|| format!("no such index [{index}]"))?;
    if s.routing_table.indices.contains_key(index) {
        return Err(format!("index [{index}] already exists in the routing table"));
    }
    if meta.number_of_shards == 0 {
        return Err(format!("index [{index}] must have at least one shard"));
    }
    let n = meta.number_of_shards as usize;
    if meta.primary_terms.len() < n {
        meta.primary_terms.resize(n, 0);
    }
    if meta.routing_num_shards == 0 {
        meta.routing_num_shards = crate::routing::calculate_num_routing_shards(meta.number_of_shards);
    }
    let irt = initialize_empty(meta, reason, source, now);
    s.routing_table.indices.insert(index.to_string(), irt);
    Ok(s)
}

pub fn set_index_state(state: &ClusterState, index: &str, to: IndexState, now: i64) -> Result<ClusterState, String> {
    let mut s = state.clone();
    let meta = s.metadata.indices.get_mut(index).ok_or_else(|| format!("no such index [{index}]"))?;
    if meta.state == to {
        return Ok(s);
    }
    meta.state = to;
    meta.version += 1;
    let reason = match to {
        IndexState::Close => {
            meta.settings.insert("index.verified_before_close".into(), Value::String("true".into()));
            meta.settings_version += 1;
            s.blocks.indices.entry(index.to_string()).or_default().insert(ClusterBlock::index_closed());
            UnassignedReason::IndexClosed
        }
        IndexState::Open => {
            if meta.settings.remove("index.verified_before_close").is_some() {
                meta.settings_version += 1;
            }
            if let Some(b) = s.blocks.indices.get_mut(index) {
                b.retain(|x| x.id != ClusterBlock::index_closed().id);
                if b.is_empty() {
                    s.blocks.indices.remove(index);
                }
            }
            UnassignedReason::IndexReopened
        }
    };
    let meta = s.metadata.indices.get(index).cloned().ok_or_else(|| format!("no such index [{index}]"))?;
    s.routing_table.indices.insert(index.to_string(), initialize_empty(&meta, reason, None, now));
    Ok(s)
}

/// Set `number_of_replicas` of one index in metadata and routing.
pub fn set_replicas(s: &mut ClusterState, index: &str, replicas: u32, now: i64) {
    let Some(meta) = s.metadata.indices.get_mut(index) else { return };
    if meta.number_of_replicas != replicas {
        meta.number_of_replicas = replicas;
        meta.settings.insert("index.number_of_replicas".into(), Value::String(replicas.to_string()));
        meta.settings_version += 1;
    }
    let Some(irt) = s.routing_table.indices.get_mut(index) else { return };
    let mut removed_ids = Vec::new();
    for t in irt.shards.iter_mut() {
        let current = t.copies.iter().filter(|c| !c.primary && !is_relocation_target(c)).count();
        if replicas as usize > current {
            for _ in current..replicas as usize {
                t.copies.push(ShardRouting::unassigned(index, t.shard, false, new_info(UnassignedReason::ReplicaAdded, now), Some(RecoverySource::Peer)));
            }
        } else {
            for _ in replicas as usize..current {
                // prefer unassigned, then initializing, then the last active replica
                let rank = |c: &ShardRouting| match c.state {
                    ShardState::Unassigned => 0,
                    ShardState::Initializing => 1,
                    _ => 2,
                };
                let Some(pos) = t.copies.iter().enumerate().filter(|(_, c)| !c.primary && !is_relocation_target(c)).min_by_key(|(i, c)| (rank(c), usize::MAX - i)).map(|(i, _)| i) else { break };
                let c = t.copies.remove(pos);
                if let Some(a) = &c.allocation_id {
                    removed_ids.push((t.shard, a.id.clone()));
                    if let Some(target) = &a.relocation_id {
                        t.copies.retain(|x| x.allocation_id() != Some(target.as_str()));
                    }
                }
            }
        }
        normalize_order(Some(t));
    }
    // the removed copies no longer exist; drop them from in-sync (never emptying it)
    if let Some(meta) = s.metadata.indices.get_mut(index) {
        for (shard, id) in removed_ids {
            if let Some(set) = meta.in_sync_allocation_ids.get_mut(&shard) {
                if set.len() > 1 {
                    set.remove(&id);
                }
            }
        }
    }
}

pub fn reset_failed_counters(s: &mut ClusterState, now: i64) {
    for irt in s.routing_table.indices.values_mut() {
        for t in irt.shards.iter_mut() {
            for c in t.copies.iter_mut() {
                if c.state != ShardState::Unassigned {
                    continue;
                }
                if let Some(u) = c.unassigned_info.as_mut() {
                    if u.failed_allocations > 0 {
                        u.reason = UnassignedReason::ManualAllocation;
                        u.failed_allocations = 0;
                        u.at_millis = now;
                        u.last_allocation_status = "no_attempt".into();
                    }
                }
            }
        }
    }
}

/// `index.auto_expand_replicas`: `"0-5"`, `"1-all"`, `"false"`.
pub fn parse_auto_expand(s: &str) -> Option<(u32, Option<u32>)> {
    let s = s.trim();
    if s == "false" || s.is_empty() {
        return None;
    }
    let (min, max) = s.split_once('-')?;
    let min = min.trim().parse().ok()?;
    let max = if max.trim() == "all" { None } else { Some(max.trim().parse().ok()?) };
    Some((min, max))
}

impl Allocation<'_> {
    /// ES `AutoExpandReplicas.getDesiredNumberOfReplicas` for every index.
    pub fn adapt_auto_expand_replicas(&mut self) {
        let mut changes = Vec::new();
        for (name, meta) in &self.state.metadata.indices {
            let Some((min, max)) = index_setting(meta, "index.auto_expand_replicas").and_then(|s| parse_auto_expand(&s)) else { continue };
            let matching = self.state.data_nodes().filter(|n| self.should_auto_expand_to_node(meta, n).kind != DecisionType::No).count() as i64;
            let max_eff = (max.map(|m| m as i64).unwrap_or(i64::MAX)).min(matching - 1);
            let mut n = matching - 1;
            if n < min as i64 {
                n = min as i64;
            } else if n > max_eff {
                n = max_eff;
            }
            if n >= min as i64 && n <= max_eff && n != meta.number_of_replicas as i64 {
                changes.push((name.clone(), n as u32));
            }
        }
        if changes.is_empty() {
            return;
        }
        for (name, n) in changes {
            set_replicas(&mut self.state, &name, n, self.now);
        }
        self.refresh();
    }

    pub fn run_reroute(&mut self) {
        self.unassigned_ignored = false;
        self.remove_delay_markers();
        self.adapt_auto_expand_replicas();
        self.allocate_unassigned();
        self.move_shards();
        self.rebalance();
    }
}
