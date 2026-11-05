//! Master-side state transitions (ES's `MasterService` task executors):
//! node joins/leaves, shard started / failed, metadata changes, reroute.
//!
//! Every task is a pure function of the current state run by the
//! coordinator on the elected master; the result is published through the
//! two-phase commit. Each transition is followed by a reroute, as in ES.

use elasticsearch_cluster::allocation::{AllocationService, ClusterInfo, FailedShard, ShardStoreInfo, StartedShard};
use elasticsearch_cluster::coordination::{StateTask, StateUpdater};
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode, IndexState, Metadata};
use crate::store_records::StoreCache;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// What the master knows beyond the cluster state.
pub struct MasterCtx {
    pub allocation: AllocationService,
    pub cluster_info: Mutex<ClusterInfo>,
    pub(crate) capacity: Mutex<crate::capacity::CapacityCache>,
    /// Copies reported by `shard_stores` fetches (full-cluster restart),
    /// by reporting node id.
    pub stores: Mutex<StoreCache>,
}

impl MasterCtx {
    pub fn new(node_settings: serde_json::Map<String, serde_json::Value>) -> MasterCtx {
        MasterCtx { allocation: AllocationService::new(node_settings), cluster_info: Mutex::new(ClusterInfo::default()), capacity: Mutex::new(Default::default()), stores: Mutex::new(StoreCache::default()) }
    }

    /// Reroute with the best store knowledge: fetched copies plus what the
    /// routing table itself says.
    pub fn reroute(&self, state: &ClusterState) -> ClusterState {
        // Declare the placement owner first so the copied info/map drops first.
        let now=std::time::Instant::now();
        let (mut placement,mut pressured)={let capacity=self.capacity.lock();(capacity.placement(state,now),capacity.sustained_pressure(state,now))};
        let mut info = self.cluster_info.lock().clone();
        info.placement_memory=std::mem::take(&mut placement.values);
        info.memory_pressure=std::mem::take(&mut pressured.values);
        let mut stores = ShardStoreInfo::from_routing(state);
        for (node_id, copies) in self.stores.lock().iter() {
            // Only copies of the index generation the state knows about.
            for c in copies.iter().filter(|c| state.metadata.indices.get(c.index.as_str()).map(|m| m.uuid == c.index_uuid).unwrap_or(false)) {
                if !state.nodes.contains_key(node_id) {
                    continue;
                }
                stores.add_copy(&c.index, c.shard, node_id, Some(&c.allocation_id), c.primary);
            }
        }
        let mut next = self.allocation.reroute(state, now_millis(), &info, &stores);
        // DiskThresholdMonitor changes belong to the same ordered publication
        // as allocation. Allocation alone only moves shards; it does not stop
        // writes on a node that has crossed the flood watermark.
        let flood = self.allocation.disk_flood_stage(state, &info);
        if !flood.block.is_empty() {
            next = elasticsearch_cluster::allocation::apply_read_only_allow_delete(&next, &flood.block, true);
        }
        if !flood.release.is_empty() {
            next = elasticsearch_cluster::allocation::apply_read_only_allow_delete(&next, &flood.release, false);
        }
        elasticsearch_cluster::allocation::patch_versions(state, &mut next);
        next
    }
}

/// Node join / leave hook run by the coordinator on the leader.
pub struct Updater(pub Arc<MasterCtx>);

impl StateUpdater for Updater {
    fn nodes_changed(&mut self, state: &ClusterState, joined: &[DiscoveryNode], left: &[(String, String)]) -> ClusterState {
        if !joined.is_empty() || !left.is_empty() {
            tracing::info!(joined = ?joined.iter().map(|n| &n.name).collect::<Vec<_>>(), left = ?left, "cluster membership changed");
        }
        let mut s = if left.is_empty() { state.clone() } else { self.0.allocation.disassociate_dead_nodes(state, now_millis()) };
        // A node that just joined is a fresh process with no open shards:
        // any copy the routing table still assigns to it (the node
        // restarted before the master noticed it leave, or the whole
        // cluster restarted) is gone. Fail those copies now, in the same
        // publication that admits the node, so no state ever names a
        // restarted node as an active primary; in-sync ids are kept and
        // the reroute re-opens the copies from their existing stores.
        // Replicas first, so a primary's failure does not promote a
        // replica that is itself about to be failed.
        let mut stale: Vec<(bool, FailedShard)> = s
            .routing_table
            .all_shards()
            .filter(|c| c.node_id.as_ref().is_some_and(|n| joined.iter().any(|j| &j.id == n)))
            .filter_map(|c| {
                let aid = c.allocation_id()?.to_string();
                Some((c.primary, FailedShard { index: c.index.clone(), shard: c.shard, allocation_id: aid, message: "node restarted".into(), failure: None, mark_as_stale: false }))
            })
            .collect();
        if !stale.is_empty() {
            stale.sort_by_key(|(primary, _)| *primary);
            let failed: Vec<FailedShard> = stale.into_iter().map(|(_, f)| f).collect();
            tracing::info!(copies = failed.len(), "failing shard copies assigned to restarted nodes");
            s = self.0.allocation.apply_failed_shards(&s, &failed, now_millis());
        }
        // `JoinTaskExecutor` → `DesiredNodes.updateDesiredNodesStatusIfNeeded`:
        // desired nodes matching a member flip to ACTUALIZED. Leaves revert
        // nothing (ACTUALIZED = "joined at least once"), so running it on
        // every membership change is equivalent and also covers the master's
        // own restart-join.
        crate::rest::desired_nodes::refresh_status(&mut s);
        // `NodeSeenService`: shutdown records whose node has joined.
        crate::rest::node_shutdown::mark_seen(&mut s);
        self.0.reroute(&s)
    }

    /// A claim above the baseline counts only with a current-process proof:
    /// an old binary can echo a configured `rustic.internal.*` attribute.
    fn segment_read_max(&self, cluster_name: &str, local: &DiscoveryNode, node: &DiscoveryNode) -> u32 {
        crate::capability_probe::verified_segment_read_max(cluster_name, local, node)
    }
}

pub fn reroute_task(ctx: Arc<MasterCtx>) -> StateTask {
    Box::new(move |s: &ClusterState| Ok(ctx.reroute(s)))
}

pub fn shards_started_task(ctx: Arc<MasterCtx>, started: Vec<StartedShard>) -> StateTask {
    Box::new(move |s: &ClusterState| {
        let next = ctx.allocation.apply_started_shards(s, &started);
        Ok(ctx.reroute(&next))
    })
}

/// A copy failed. `reporter_term` is the primary term of the primary that
/// reported it (None when the copy reports itself): a report from a stale
/// primary is rejected, so a deposed primary cannot evict in-sync copies
/// (and therefore cannot ack writes; `docs/CONSISTENCY.md` G4).
pub fn shard_failed_task(ctx: Arc<MasterCtx>, failed: FailedShard, reporter_term: Option<u64>) -> StateTask {
    Box::new(move |s: &ClusterState| {
        if let Some(t) = reporter_term {
            let current = s.metadata.indices.get(&failed.index).map(|m| m.primary_term(failed.shard)).unwrap_or(0);
            if t != current {
                return Err(format!(
                    "primary term [{t}] did not match current primary term [{current}] for shard [{}][{}]",
                    failed.index, failed.shard
                ));
            }
        }
        let next = ctx.allocation.apply_failed_shards(s, std::slice::from_ref(&failed), now_millis());
        Ok(ctx.reroute(&next))
    })
}

/// Publish metadata computed by the master's engine (see `mirror::extract`)
/// and adjust the routing table: new / deleted / opened / closed indices
/// and replica-count changes.
pub fn metadata_task(ctx: Arc<MasterCtx>, md: Metadata) -> StateTask {
    Box::new(move |s: &ClusterState| {
        let now = now_millis();
        let mut next = s.clone();
        for name in s.metadata.indices.keys() {
            if !md.indices.contains_key(name) {
                next = ctx.allocation.delete_index(&next, name);
            }
        }
        let mut new_md = md.clone();
        // Routing-owned fields come from the live state, not the snapshot
        // the master's engine was extracted against.
        for (name, im) in new_md.indices.iter_mut() {
            if let Some(cur) = s.metadata.indices.get(name).filter(|c| c.uuid == im.uuid) {
                im.primary_terms = cur.primary_terms.clone();
                im.in_sync_allocation_ids = cur.in_sync_allocation_ids.clone();
                im.routing_num_shards = cur.routing_num_shards;
            } else {
                // `routing_num_shards` is the explicit
                // `index.number_of_routing_shards` or 0 (derived on
                // allocation).
                im.primary_terms = vec![0; im.number_of_shards as usize];
            }
        }
        new_md.cluster_uuid = s.metadata.cluster_uuid.clone();
        new_md.cluster_uuid_committed = s.metadata.cluster_uuid_committed;
        new_md.version = s.metadata.version + 1;
        let old = std::mem::replace(&mut next.metadata, new_md);
        for (name, im) in md.indices.iter() {
            match old.indices.get(name).filter(|o| o.uuid == im.uuid) {
                None => {
                    next.routing_table.indices.remove(name);
                    // A restore's target index gets ES's restore routing.
                    next = crate::snapshot_api::file_restore::new_index_routing(&ctx.allocation, next, name, now)?;
                }
                Some(o) => {
                    if o.state != im.state {
                        next = match im.state {
                            IndexState::Close => ctx.allocation.close_index(&next, name, now)?,
                            IndexState::Open => ctx.allocation.open_index(&next, name, now)?,
                        };
                    }
                    if o.number_of_replicas != im.number_of_replicas {
                        next = ctx.allocation.update_number_of_replicas(&next, std::slice::from_ref(name), im.number_of_replicas, now)?;
                    }
                }
            }
        }
        Ok(ctx.reroute(&next))
    })
}

/// Error prefix of a metadata-service task whose base changed meanwhile.
pub const STALE_BASE: &str = "stale metadata base";

/// The pure half of a metadata-service batch (`master_service`): apply the
/// batch's [`MetadataDelta`](crate::mirror::MetadataDelta) to the *current*
/// master state, then adjust the routing table for the indices it names
/// (new / deleted / re-created, opened / closed, replica count). Indices,
/// templates, settings and customs the batch did not touch keep whatever
/// the current state has (e.g. a custom another task changed meanwhile),
/// and are not visited: the work is proportional to the change (plus the
/// reroute), not to the cluster's metadata.
pub fn engine_metadata_task(ctx: Arc<MasterCtx>, delta: crate::mirror::MetadataDelta) -> StateTask {
    Box::new(move |s: &ClusterState| {
        // Compare-and-set: every entry the delta touches must still be as
        // the batch saw it (ES runs its tasks against the current state; a
        // change computed on an older base is retried, never applied over
        // a newer one). In particular a create never replaces another
        // incarnation of the name unless the batch deleted it.
        delta.expect.check(&s.metadata).map_err(|e| format!("{STALE_BASE}: {e}"))?;
        let now = now_millis();
        let mut next = s.clone();
        for (name, change) in &delta.indices {
            // A delete, or a re-create under a new uuid (the batch deleted
            // the old incarnation): the old one's routing and index blocks
            // go with it.
            let replaced = match (change, s.metadata.indices.get(name)) {
                (_, None) => false,
                (None, Some(_)) => true,
                (Some(im), Some(cur)) => cur.uuid != im.uuid,
            };
            if replaced {
                next = ctx.allocation.delete_index(&next, name);
            }
        }
        for (name, change) in &delta.indices {
            match change {
                None => {
                    next.metadata.indices.remove(name);
                    next.routing_table.indices.remove(name);
                }
                Some(im) => {
                    let mut im = im.clone();
                    // Routing-owned fields come from the live state, not
                    // the state the change was computed against.
                    match s.metadata.indices.get(name).filter(|c| c.uuid == im.uuid) {
                        Some(cur) => {
                            im.primary_terms = cur.primary_terms.clone();
                            im.in_sync_allocation_ids = cur.in_sync_allocation_ids.clone();
                            im.routing_num_shards = cur.routing_num_shards;
                        }
                        // `routing_num_shards` is the explicit
                        // `index.number_of_routing_shards` or 0 (derived on
                        // allocation).
                        None => im.primary_terms = vec![0; im.number_of_shards as usize],
                    }
                    next.metadata.indices.insert(name.clone(), im);
                }
            }
        }
        delta.apply_globals_to(&mut next.metadata);
        next.metadata.version = s.metadata.version + 1;
        for (name, change) in &delta.indices {
            let Some(im) = change else { continue };
            match s.metadata.indices.get(name).filter(|o| o.uuid == im.uuid) {
                None => {
                    next.routing_table.indices.remove(name);
                    // A restore's target index gets ES's restore routing.
                    next = crate::snapshot_api::file_restore::new_index_routing(&ctx.allocation, next, name, now)?;
                }
                Some(o) => {
                    if o.state != im.state {
                        next = match im.state {
                            IndexState::Close => ctx.allocation.close_index(&next, name, now)?,
                            IndexState::Open => ctx.allocation.open_index(&next, name, now)?,
                        };
                    }
                    if o.number_of_replicas != im.number_of_replicas {
                        next = ctx.allocation.update_number_of_replicas(&next, std::slice::from_ref(name), im.number_of_replicas, now)?;
                    }
                }
            }
        }
        Ok(ctx.reroute(&next))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mirror::MetadataDelta;

    fn im(name: &str, uuid: &str) -> elasticsearch_cluster::state::IndexMetadata {
        let mut m: elasticsearch_cluster::state::IndexMetadata = serde_json::from_value(serde_json::json!({
            "name": name, "uuid": uuid, "version": 1, "mapping_version": 1, "settings_version": 1,
            "aliases_version": 1, "state": "open", "settings": {"index.number_of_shards": "1", "index.number_of_replicas": "0"}, "mappings": {}, "aliases": {},
            "number_of_shards": 1, "number_of_replicas": 0, "routing_num_shards": 1,
            "primary_terms": [0], "in_sync_allocation_ids": {}, "creation_date": 0
        })).unwrap();
        m.uuid = uuid.into();
        m
    }

    /// Review round 2: a published task computed on a base that changed
    /// meanwhile is rejected with the stale-base error (the batch retries),
    /// never applied over the newer state.
    #[test]
    fn a_task_computed_on_a_stale_base_is_rejected() {
        let ctx = Arc::new(MasterCtx::new(Default::default()));
        let mut base = ClusterState::empty("c");
        base.metadata.indices.insert("a".into(), im("a", "u1"));
        let mut changed = base.metadata.indices["a"].clone();
        changed.mapping_version += 1;
        changed.version += 1;
        let mut delta = MetadataDelta::default();
        delta.indices.insert("a".into(), Some(changed));
        delta.expect_base(&base.metadata);
        // `a` re-created meanwhile (another uuid).
        let mut recreated = base.clone();
        recreated.metadata.indices.insert("a".into(), im("a", "u2"));
        let err = engine_metadata_task(ctx.clone(), delta.clone())(&recreated).unwrap_err();
        assert!(err.starts_with(STALE_BASE), "{err}");
        // `a`'s mapping changed meanwhile.
        let mut remapped = base.clone();
        remapped.metadata.indices.get_mut("a").unwrap().mapping_version += 1;
        assert!(engine_metadata_task(ctx.clone(), delta.clone())(&remapped).unwrap_err().starts_with(STALE_BASE));
        // A create computed where `b` was absent never replaces a `b` that
        // appeared meanwhile.
        let mut create = MetadataDelta::default();
        create.indices.insert("b".into(), Some(im("b", "mine")));
        create.expect_base(&base.metadata);
        let mut raced = base.clone();
        raced.metadata.indices.insert("b".into(), im("b", "theirs"));
        assert!(engine_metadata_task(ctx.clone(), create)(&raced).unwrap_err().starts_with(STALE_BASE));
        // A template changed meanwhile.
        let mut tdelta = MetadataDelta::default();
        tdelta.index_templates.insert("t".into(), Some(serde_json::json!({"index_patterns": ["x*"]})));
        tdelta.expect_base(&base.metadata);
        let mut templated = base.clone();
        templated.metadata.index_templates.insert("t".into(), serde_json::json!({"index_patterns": ["y*"]}));
        assert!(engine_metadata_task(ctx.clone(), tdelta)(&templated).unwrap_err().starts_with(STALE_BASE));
        // On the unchanged base the task applies.
        let next = engine_metadata_task(ctx, delta)(&base).expect("applies on its base");
        assert_eq!(next.metadata.indices["a"].mapping_version, 2);
        assert_eq!(next.metadata.version, base.metadata.version + 1);
    }
}
