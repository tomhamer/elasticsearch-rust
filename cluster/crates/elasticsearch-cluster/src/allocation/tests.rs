//! Allocation scenario tests. A small simulated cluster ([`World`]) tracks
//! which shard data each node holds on disk (surviving restarts), applies
//! transitions through [`AllocationService`] and checks `state.validate()`
//! plus the data-safety invariants after every step.

use super::*;
use crate::health::{HealthInputs, HealthStatus, cluster_health};
use serde_json::json;
use std::collections::BTreeMap;

const MIN: i64 = 60_000;

fn node(id: &str, attrs: &[(&str, &str)]) -> DiscoveryNode {
    DiscoveryNode {
        id: id.into(),
        name: id.into(),
        ephemeral_id: format!("eph-{id}"),
        transport_address: format!("10.0.0.{}:9300", id.len()),
        http_address: "127.0.0.1:9200".into(),
        host: format!("host-{id}"),
        ip: format!("10.0.0.{}", id.len()),
        roles: Role::defaults(),
        attributes: attrs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        version: "8.15.0".into(),
    }
}

fn index_meta(name: &str, shards: u32, replicas: u32, settings: &[(&str, &str)]) -> IndexMetadata {
    let mut s = Map::new();
    s.insert("index.number_of_shards".into(), json!(shards.to_string()));
    s.insert("index.number_of_replicas".into(), json!(replicas.to_string()));
    for (k, v) in settings {
        s.insert(k.to_string(), json!(v));
    }
    IndexMetadata {
        name: name.into(),
        uuid: format!("uuid-{name}"),
        version: 1,
        mapping_version: 1,
        settings_version: 1,
        aliases_version: 1,
        state: IndexState::Open,
        settings: s.into(),
        mappings: json!({}).into(),
        aliases: BTreeMap::new(),
        number_of_shards: shards,
        number_of_replicas: replicas,
        routing_num_shards: 0,
        primary_terms: vec![],
        in_sync_allocation_ids: BTreeMap::new(),
        creation_date: 0,
        lifecycle: None,
    }
}

struct World {
    svc: AllocationService,
    state: ClusterState,
    now: i64,
    info: ClusterInfo,
    /// (index, shard, node) → (allocation id, was primary) of the data on that node's disk
    disk: BTreeMap<(String, u32, String), (String, bool)>,
    /// highest primary term seen per (index, shard)
    terms: BTreeMap<(String, u32), u64>,
    /// last primary (allocation id, relocation target id, term) per (index, shard)
    primaries: BTreeMap<(String, u32), (String, Option<String>, u64)>,
}

impl World {
    fn new(nodes: &[DiscoveryNode]) -> World {
        let mut state = ClusterState::empty("test-cluster");
        state.blocks.global.clear();
        for n in nodes {
            state.nodes.insert(n.id.clone(), n.clone());
        }
        state.master_node_id = nodes.first().map(|n| n.id.clone());
        World { svc: AllocationService::default(), state, now: 1_000_000, info: ClusterInfo::default(), disk: BTreeMap::new(), terms: BTreeMap::new(), primaries: BTreeMap::new() }
    }

    fn check(&mut self) {
        self.state.validate().unwrap();
        for (name, irt) in &self.state.routing_table.indices {
            let meta = &self.state.metadata.indices[name];
            for t in &irt.shards {
                let in_sync = meta.in_sync_allocation_ids.get(&t.shard).cloned().unwrap_or_default();
                for c in &t.copies {
                    // an active primary is always an in-sync copy
                    if c.primary && c.active() {
                        assert!(in_sync.contains(c.allocation_id().unwrap()), "active primary {c:?} not in-sync {in_sync:?}");
                    }
                    assert_eq!(c.assigned(), c.state != ShardState::Unassigned);
                }
                assert!(t.copies.iter().filter(|c| c.primary && !context::is_relocation_target(c)).count() == 1, "exactly one primary: {t:?}");
                let term = meta.primary_term(t.shard);
                let prev = self.terms.entry((name.clone(), t.shard)).or_insert(0);
                assert!(term >= *prev, "primary term went backwards");
                *prev = term;
                // every new primary copy (other than a relocation hand-off) has a higher term
                if let Some(p) = t.copies.iter().find(|c| c.primary && !context::is_relocation_target(c) && c.assigned()) {
                    let aid = p.allocation_id().unwrap().to_string();
                    let reloc = p.allocation_id.as_ref().and_then(|a| a.relocation_id.clone());
                    let key = (name.clone(), t.shard);
                    if let Some((paid, ptarget, pterm)) = self.primaries.get(&key) {
                        if *paid != aid && ptarget.as_deref() != Some(aid.as_str()) {
                            assert!(term > *pterm, "primary of {key:?} changed {paid} -> {aid} without a term bump");
                        }
                    }
                    self.primaries.insert(key, (aid, reloc, term));
                }
            }
        }
    }

    fn stores(&self) -> ShardStoreInfo {
        let mut s = ShardStoreInfo::new();
        for ((i, sh, n), (aid, primary)) in &self.disk {
            if self.state.nodes.contains_key(n) {
                s.add_copy(i, *sh, n, Some(aid), *primary);
            }
        }
        s
    }

    fn create(&mut self, name: &str, shards: u32, replicas: u32, settings: &[(&str, &str)]) {
        self.state.metadata.indices.insert(name.into(), index_meta(name, shards, replicas, settings));
        self.state = self.svc.add_index_routing(&self.state, name, self.now).unwrap();
        self.check();
    }

    fn reroute(&mut self) {
        let stores = self.stores();
        self.state = self.svc.reroute(&self.state, self.now, &self.info, &stores);
        self.check();
    }

    fn initializing(&self) -> Vec<StartedShard> {
        self.state.routing_table.all_shards().filter(|c| c.state == ShardState::Initializing).map(|c| StartedShard { index: c.index.clone(), shard: c.shard, allocation_id: c.allocation_id().unwrap().into() }).collect()
    }

    fn start_all(&mut self) -> usize {
        let started = self.initializing();
        for c in self.state.routing_table.all_shards().filter(|c| c.state == ShardState::Initializing) {
            self.disk.insert((c.index.clone(), c.shard, c.node_id.clone().unwrap()), (c.allocation_id().unwrap().into(), c.primary));
        }
        self.state = self.svc.apply_started_shards(&self.state, &started);
        self.check();
        // shard data is deleted from nodes that no longer hold a copy once the shard is fully active
        let st = self.state.clone();
        self.disk.retain(|(i, s, n), _| {
            let Some(t) = st.routing_table.shard_table(i, *s) else { return false };
            let fully_active = t.copies.iter().all(|c| c.state == ShardState::Started);
            !st.nodes.contains_key(n) || !fully_active || t.copies.iter().any(|c| c.node_id.as_deref() == Some(n.as_str()))
        });
        started.len()
    }

    /// reroute + start until nothing moves
    fn converge(&mut self) -> usize {
        let mut rounds = 0;
        loop {
            self.reroute();
            if self.start_all() == 0 {
                return rounds;
            }
            rounds += 1;
            assert!(rounds < 200, "did not converge");
        }
    }

    fn kill(&mut self, id: &str) {
        self.state.nodes.remove(id);
        self.state = self.svc.disassociate_dead_nodes(&self.state, self.now);
        self.check();
    }

    fn add_node(&mut self, n: DiscoveryNode) {
        self.state.nodes.insert(n.id.clone(), n);
    }

    fn health(&self) -> crate::health::ClusterHealth {
        cluster_health(&self.state, None, HealthInputs::default())
    }

    fn copies(&self, index: &str, shard: u32) -> Vec<ShardRouting> {
        self.state.routing_table.shard_table(index, shard).unwrap().copies.clone()
    }

    fn primary(&self, index: &str, shard: u32) -> ShardRouting {
        self.copies(index, shard).into_iter().find(|c| c.primary).unwrap()
    }

    fn in_sync(&self, index: &str, shard: u32) -> BTreeSet<String> {
        self.state.metadata.indices[index].in_sync_allocation_ids.get(&shard).cloned().unwrap_or_default()
    }

    fn term(&self, index: &str, shard: u32) -> u64 {
        self.state.metadata.indices[index].primary_term(shard)
    }

    fn set_cluster(&mut self, k: &str, v: &str) {
        self.state.metadata.persistent_settings.insert(k.into(), json!(v));
    }

    fn set_index(&mut self, index: &str, k: &str, v: &str) {
        self.state.metadata.indices.get_mut(index).unwrap().settings.insert(k.into(), json!(v));
    }

    fn per_node(&self) -> BTreeMap<String, usize> {
        let mut m: BTreeMap<String, usize> = self.state.data_nodes().map(|n| (n.id.clone(), 0)).collect();
        for c in self.state.routing_table.all_shards() {
            if let Some(n) = &c.node_id {
                *m.entry(n.clone()).or_default() += 1;
            }
        }
        m
    }

    fn explain(&self, req: Value) -> Result<Value, ApiError> {
        self.svc.explain(&self.state, &req, ExplainOptions::default(), self.now, &self.info, &self.stores())
    }
}

fn three() -> World {
    World::new(&[node("n1", &[]), node("n2", &[]), node("n3", &[])])
}

// ------------------------------------------------------------ lifecycle --

#[test]
fn create_start_replicate_green() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    let c = w.copies("idx", 0);
    assert!(c.iter().all(|c| c.state == ShardState::Unassigned));
    assert_eq!(c[0].unassigned_info.as_ref().unwrap().reason, UnassignedReason::IndexCreated);
    assert_eq!(c[0].recovery_source, Some(RecoverySource::EmptyStore));
    assert_eq!(c[1].recovery_source, Some(RecoverySource::Peer));
    assert_eq!(w.health().status, HealthStatus::Yellow); // brand new primary
    assert_eq!(w.state.metadata.indices["idx"].routing_num_shards, 1024);

    w.reroute();
    let p = w.primary("idx", 0);
    assert_eq!(p.state, ShardState::Initializing);
    assert_eq!(w.term("idx", 0), 1);
    // replica waits for the primary
    assert_eq!(w.copies("idx", 0)[1].state, ShardState::Unassigned);
    assert_eq!(w.copies("idx", 0)[1].unassigned_info.as_ref().unwrap().last_allocation_status, "deciders_no");
    assert_eq!(w.health().status, HealthStatus::Yellow);

    w.start_all();
    assert_eq!(w.in_sync("idx", 0), BTreeSet::from([p.allocation_id().unwrap().to_string()]));
    w.reroute();
    let r = w.copies("idx", 0)[1].clone();
    assert_eq!(r.state, ShardState::Initializing);
    assert_ne!(r.node_id, p.node_id);
    w.start_all();
    let h = w.health();
    assert_eq!(h.status, HealthStatus::Green);
    assert_eq!((h.active_shards, h.active_primary_shards), (2, 1));
    assert_eq!(w.in_sync("idx", 0).len(), 2);
    assert_eq!(w.term("idx", 0), 1);
    // stable: another reroute changes nothing
    let before = w.state.clone();
    w.reroute();
    assert_eq!(before.routing_table, w.state.routing_table);
}

#[test]
fn kill_primary_promotes_replica_and_reallocates_after_delay() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let p = w.primary("idx", 0);
    let r = w.copies("idx", 0)[1].clone();
    let old_in_sync = w.in_sync("idx", 0);
    w.kill(p.node_id.as_deref().unwrap());
    // replica promoted, term bumped, old primary demoted to a delayed unassigned replica
    let np = w.primary("idx", 0);
    assert_eq!(np.allocation_id(), r.allocation_id());
    assert_eq!(np.state, ShardState::Started);
    assert_eq!(w.term("idx", 0), 2);
    let old = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    let u = old.unassigned_info.clone().unwrap();
    assert_eq!(u.reason, UnassignedReason::NodeLeft);
    assert!(u.delayed);
    assert_eq!(u.details.as_deref(), Some(format!("node_left [{}]", p.node_id.as_deref().unwrap()).as_str()));
    // node-left never removes in-sync ids
    assert_eq!(w.in_sync("idx", 0), old_in_sync);
    let h = w.health();
    assert_eq!(h.status, HealthStatus::Yellow);
    assert_eq!(h.delayed_unassigned_shards, 1);
    // still delayed: not reallocated
    w.reroute();
    assert_eq!(w.copies("idx", 0).iter().filter(|c| c.assigned()).count(), 1);
    assert_eq!(w.svc.next_delayed_allocation_millis(&w.state, w.now), Some(w.now + MIN));
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": false})).unwrap();
    assert_eq!(ex["can_allocate"], json!("allocation_delayed"));
    assert!(ex["allocate_explanation"].as_str().unwrap().starts_with("cannot allocate because the cluster is still waiting 1m for the departed node"));
    assert_eq!(ex["configured_delay_in_millis"], json!(MIN));
    // after the delay it goes to the third node
    w.now += MIN + 1;
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
    let nodes: BTreeSet<_> = w.copies("idx", 0).iter().map(|c| c.node_id.clone().unwrap()).collect();
    assert!(!nodes.contains(p.node_id.as_deref().unwrap()));
    // in-sync bounded to the active copies (old id trimmed)
    assert_eq!(w.in_sync("idx", 0).len(), 2);
    assert!(!w.in_sync("idx", 0).contains(p.allocation_id().unwrap()));
    assert_eq!(w.term("idx", 0), 2);
}

#[test]
fn delayed_replica_returns_to_its_node() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let r = w.copies("idx", 0)[1].clone();
    let rn = r.node_id.clone().unwrap();
    let saved = w.state.nodes[&rn].clone();
    w.kill(&rn);
    w.reroute();
    assert_eq!(w.health().delayed_unassigned_shards, 1);
    w.add_node(saved);
    w.reroute();
    let back = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(back.node_id.as_deref(), Some(rn.as_str()));
    assert_eq!(back.state, ShardState::Initializing);
    w.start_all();
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn partition_old_primary_becomes_stale_replica() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let p = w.primary("idx", 0);
    let pn = p.node_id.clone().unwrap();
    let saved = w.state.nodes[&pn].clone();
    let paid = p.allocation_id().unwrap().to_string();
    // old primary is cut off: replica promoted with term + 1
    w.kill(&pn);
    assert_eq!(w.term("idx", 0), 2);
    // the new primary's first write marks the unavailable in-sync copy stale
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: paid.clone(), message: "mark copy as stale".into(), failure: None, mark_as_stale: true }], w.now);
    w.check();
    assert!(!w.in_sync("idx", 0).contains(&paid));
    assert_eq!(w.in_sync("idx", 0).len(), 1);
    // old node comes back: its copy recovers as a replica with a fresh allocation id
    w.add_node(saved);
    w.reroute();
    let back = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(back.node_id.as_deref(), Some(pn.as_str()));
    assert_eq!(back.recovery_source, Some(RecoverySource::Peer));
    assert_ne!(back.allocation_id(), Some(paid.as_str()));
    // if the new primary dies before that recovery finishes, the stale copy must not become primary
    let np = w.primary("idx", 0).node_id.clone().unwrap();
    w.kill(&np);
    w.reroute();
    let p2 = w.primary("idx", 0);
    assert_eq!(p2.state, ShardState::Unassigned);
    assert_eq!(p2.unassigned_info.as_ref().unwrap().last_allocation_status, "no_valid_shard_copy");
    assert_eq!(w.health().status, HealthStatus::Red);
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["can_allocate"], json!("no_valid_shard_copy"));
    assert_eq!(ex["allocate_explanation"], json!("cannot allocate because all found copies of the shard are either stale or corrupt"));
    let stale_node = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|n| n["node_id"] == json!(pn)).unwrap().clone();
    assert_eq!(stale_node["store"]["in_sync"], json!(false));
    assert_eq!(stale_node["node_decision"], json!("no"));
}

#[test]
fn full_cluster_restart_uses_only_in_sync_copies() {
    let mut w = three();
    w.create("idx", 2, 1, &[]);
    w.converge();
    let terms: Vec<u64> = (0..2).map(|s| w.term("idx", s)).collect();
    let in_sync: Vec<_> = (0..2).map(|s| w.in_sync("idx", s)).collect();
    let saved: Vec<DiscoveryNode> = w.state.nodes.values().cloned().collect();
    w.state.nodes.clear();
    w.state = w.svc.disassociate_dead_nodes(&w.state, w.now);
    w.check();
    for s in 0..2 {
        let p = w.primary("idx", s);
        assert_eq!(p.state, ShardState::Unassigned);
        assert!(matches!(p.recovery_source, Some(RecoverySource::ExistingStore { .. })));
        assert_eq!(w.in_sync("idx", s), in_sync[s as usize], "node-left keeps in-sync ids");
    }
    assert_eq!(w.health().status, HealthStatus::Red);
    // data on disk is pending fetch: nothing allocated, in-flight fetch reported
    for n in &saved {
        w.add_node(n.clone());
    }
    let mut pending = w.stores();
    pending.mark_pending("idx", 0);
    w.state = w.svc.reroute(&w.state, w.now, &w.info, &pending);
    w.check();
    assert_eq!(w.primary("idx", 0).state, ShardState::Unassigned);
    assert_eq!(w.primary("idx", 0).unassigned_info.as_ref().unwrap().last_allocation_status, "fetching_shard_data");
    // fetch done: primaries reopen their own in-sync copies, keeping their allocation ids
    w.reroute();
    for s in 0..2 {
        let p = w.primary("idx", s);
        assert_eq!(p.state, ShardState::Initializing);
        assert!(in_sync[s as usize].contains(p.allocation_id().unwrap()));
        assert_eq!(w.disk[&("idx".to_string(), s, p.node_id.clone().unwrap())].0, p.allocation_id().unwrap());
        assert_eq!(w.term("idx", s), terms[s as usize] + 1);
    }
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn restart_with_only_stale_copy_stays_red() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let r = w.copies("idx", 0)[1].clone();
    let rn = r.node_id.clone().unwrap();
    // replica failed by the primary while replicating → stale
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: r.allocation_id().unwrap().into(), message: "replication failed".into(), failure: Some("IOException".into()), mark_as_stale: true }], w.now);
    w.check();
    assert_eq!(w.in_sync("idx", 0).len(), 1);
    let failed = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(failed.unassigned_info.as_ref().unwrap().reason, UnassignedReason::AllocationFailed);
    assert_eq!(failed.unassigned_info.as_ref().unwrap().failed_allocations, 1);
    assert_eq!(failed.unassigned_info.as_ref().unwrap().details.as_deref(), Some(format!("failed shard on node [{rn}]: replication failed, failure IOException").as_str()));
    // everybody goes down; only the stale replica's node comes back
    let saved = w.state.nodes[&rn].clone();
    w.state.nodes.clear();
    w.state = w.svc.disassociate_dead_nodes(&w.state, w.now);
    w.add_node(saved);
    w.reroute();
    assert_eq!(w.primary("idx", 0).state, ShardState::Unassigned);
    assert_eq!(w.health().status, HealthStatus::Red);
    // explicit stale-primary allocation is the only way; it bumps the term and resets in-sync
    let term = w.term("idx", 0);
    let err = w.svc.reroute_commands(&w.state, &json!({"commands": [{"allocate_stale_primary": {"index": "idx", "shard": 0, "node": rn}}]}), RerouteFlags::default(), w.now, &w.info, &w.stores()).unwrap_err();
    assert_eq!(err.0, 400);
    assert!(err.2.contains("can result in data loss. Please confirm by setting the accept_data_loss parameter to true"), "{}", err.2);
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"allocate_stale_primary": {"index": "idx", "shard": 0, "node": rn, "accept_data_loss": true}}]}), RerouteFlags::default(), w.now, &w.info, &w.stores()).unwrap();
    w.state = res.state;
    w.check();
    let p = w.primary("idx", 0);
    assert_eq!(p.state, ShardState::Initializing);
    assert_eq!(p.recovery_source, Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: true }));
    assert_eq!(w.in_sync("idx", 0), BTreeSet::from([p.allocation_id().unwrap().to_string()]));
    assert_eq!(w.term("idx", 0), term + 1);
}

#[test]
fn primary_failure_never_promotes_non_in_sync_replica() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.reroute();
    w.start_all(); // primary started, replica not yet allocated
    w.reroute(); // replica initializing (not in-sync)
    let p = w.primary("idx", 0);
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: p.allocation_id().unwrap().into(), message: "disk broke".into(), failure: None, mark_as_stale: true }], w.now);
    w.check();
    let c = w.copies("idx", 0);
    // the initializing replica was failed (PRIMARY_FAILED), the primary is unassigned and kept in-sync
    assert_eq!(c.iter().filter(|c| c.assigned()).count(), 0);
    let rep = c.iter().find(|c| !c.primary).unwrap();
    assert_eq!(rep.unassigned_info.as_ref().unwrap().reason, UnassignedReason::PrimaryFailed);
    assert_eq!(w.in_sync("idx", 0), BTreeSet::from([p.allocation_id().unwrap().to_string()]));
    assert_eq!(w.health().status, HealthStatus::Red);
    // it reallocates only onto its own in-sync copy
    w.reroute();
    let np = w.primary("idx", 0);
    assert_eq!(np.node_id, p.node_id);
    assert_eq!(np.allocation_id(), p.allocation_id());
    assert_eq!(w.term("idx", 0), 2);
}

#[test]
fn replica_failure_without_stale_keeps_in_sync() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let r = w.copies("idx", 0)[1].clone();
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: r.allocation_id().unwrap().into(), message: "x".into(), failure: None, mark_as_stale: false }], w.now);
    w.check();
    assert!(w.in_sync("idx", 0).contains(r.allocation_id().unwrap()));
    // unknown / duplicate reports are ignored
    let before = w.state.clone();
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: "nope".into(), message: "x".into(), failure: None, mark_as_stale: false }], w.now);
    assert_eq!(before, w.state);
}

#[test]
fn max_retry_and_retry_failed() {
    let mut w = three();
    w.create("idx", 1, 0, &[]);
    for i in 1..=5 {
        w.reroute();
        let p = w.primary("idx", 0);
        assert_eq!(p.state, ShardState::Initializing, "attempt {i}");
        w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: p.allocation_id().unwrap().into(), message: "boom".into(), failure: None, mark_as_stale: false }], w.now);
        w.check();
        assert_eq!(w.primary("idx", 0).unassigned_info.as_ref().unwrap().failed_allocations, i);
    }
    w.reroute();
    let p = w.primary("idx", 0);
    assert_eq!(p.state, ShardState::Unassigned);
    assert_eq!(w.health().status, HealthStatus::Red);
    let ex = w.explain(json!({})).unwrap();
    assert!(ex["note"].as_str().unwrap().starts_with("No shard was specified in the explain API request"));
    let d = &ex["node_allocation_decisions"][0]["deciders"][0];
    assert_eq!(d["decider"], json!("max_retry"));
    assert!(d["explanation"].as_str().unwrap().starts_with("shard has exceeded the maximum number of retries [5] on failed allocation attempts - manually call [POST /_cluster/reroute?retry_failed&metric=none] to retry, [unassigned_info[[reason=ALLOCATION_FAILED]"));
    let res = w.svc.reroute_commands(&w.state, &json!({}), RerouteFlags { retry_failed: true, ..Default::default() }, w.now, &w.info, &w.stores()).unwrap();
    w.state = res.state;
    w.check();
    assert_eq!(w.primary("idx", 0).state, ShardState::Initializing);
    assert_eq!(w.primary("idx", 0).unassigned_info.as_ref().unwrap().reason, UnassignedReason::ManualAllocation);
}

// --------------------------------------------------------------- deciders --

#[test]
fn same_shard_single_node_stays_yellow() {
    let mut w = World::new(&[node("n1", &[])]);
    w.create("idx", 1, 1, &[]);
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Yellow);
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": false})).unwrap();
    assert_eq!(ex["can_allocate"], json!("no"));
    assert_eq!(ex["allocate_explanation"], json!("cannot allocate because allocation is not permitted to any of the nodes"));
    let d = &ex["node_allocation_decisions"][0];
    assert_eq!(d["node_decision"], json!("no"));
    assert_eq!(d["weight_ranking"], json!(1));
    assert_eq!(d["deciders"][0]["decider"], json!("same_shard"));
    assert!(d["deciders"][0]["explanation"].as_str().unwrap().starts_with("a copy of this shard is already allocated to this node [[idx][0], node[n1], [P], s[STARTED], a[id="));
}

#[test]
fn awareness_spreads_copies_across_zones() {
    let mut w = World::new(&[node("a1", &[("zone", "a")]), node("a2", &[("zone", "a")]), node("b1", &[("zone", "b")]), node("b2", &[("zone", "b")])]);
    w.set_cluster("cluster.routing.allocation.awareness.attributes", "zone");
    w.create("idx", 3, 1, &[]);
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
    for s in 0..3 {
        let zones: BTreeSet<String> = w.copies("idx", s).iter().map(|c| w.state.nodes[c.node_id.as_ref().unwrap()].attributes["zone"].clone()).collect();
        assert_eq!(zones.len(), 2, "shard {s} copies in both zones");
    }
}

#[test]
fn forced_awareness_leaves_replica_unassigned() {
    let mut w = World::new(&[node("a1", &[("zone", "a")]), node("a2", &[("zone", "a")])]);
    w.set_cluster("cluster.routing.allocation.awareness.attributes", "zone");
    w.set_cluster("cluster.routing.allocation.awareness.force.zone.values", "a,b");
    w.create("idx", 1, 1, &[]);
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Yellow);
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": false})).unwrap();
    let decs: Vec<Value> = ex["node_allocation_decisions"].as_array().unwrap().iter().flat_map(|n| n["deciders"].as_array().cloned().unwrap_or_default()).collect();
    let aw = decs.iter().find(|d| d["decider"] == json!("awareness")).unwrap();
    assert_eq!(aw["explanation"], json!("there are [2] copies of this shard and [2] values for attribute [zone] ([a] from nodes in the cluster and [a, b] from forced awareness) so there may be at most [1] copies of this shard allocated to nodes with each value, but (including this copy) there would be [2] copies allocated to nodes with [node.attr.zone: a]"));
    // a node lacking the attribute is refused outright
    let mut w2 = World::new(&[node("x", &[])]);
    w2.set_cluster("cluster.routing.allocation.awareness.attributes", "zone");
    w2.create("idx", 1, 0, &[]);
    w2.reroute();
    let ex = w2.explain(json!({"index": "idx", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["node_allocation_decisions"][0]["deciders"][0]["explanation"], json!("node does not contain the awareness attribute [zone]; required attributes cluster setting [cluster.routing.allocation.awareness.attributes=zone]"));
}

#[test]
fn index_and_cluster_filters() {
    let mut w = three();
    w.create("idx", 2, 0, &[("index.routing.allocation.require._name", "n2")]);
    w.converge();
    assert!(w.copies("idx", 0).iter().chain(w.copies("idx", 1).iter()).all(|c| c.node_id.as_deref() == Some("n2")));
    // include by attribute that matches nothing → NO with ES's message
    w.create("other", 1, 0, &[("index.routing.allocation.include.rack", "r9")]);
    w.reroute();
    let ex = w.explain(json!({"index": "other", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["node_allocation_decisions"][0]["deciders"][0]["explanation"], json!("node does not match index setting [index.routing.allocation.include] filters [rack:\"r9\"]"));
    assert_eq!(w.health().status, HealthStatus::Red);
    // cluster-level exclude drains a node (moveShards)
    let mut w = three();
    w.create("idx", 3, 0, &[]);
    w.converge();
    assert!(w.per_node()["n1"] > 0);
    w.set_cluster("cluster.routing.allocation.exclude._name", "n1");
    w.reroute();
    assert!(w.state.routing_table.all_shards().any(|c| c.state == ShardState::Relocating));
    let ex = w.explain(json!({"index": "idx", "shard": w.state.routing_table.shards_on_node("n1").next().unwrap().shard, "primary": true})).unwrap();
    assert_eq!(ex["explanation"].as_str().unwrap().split(" to node").next().unwrap(), "the shard is in the process of relocating from node [n1]");
    w.converge();
    assert_eq!(w.per_node()["n1"], 0);
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn enable_allocation_settings() {
    let mut w = three();
    w.set_cluster("cluster.routing.allocation.enable", "none");
    w.create("idx", 1, 1, &[]);
    w.reroute();
    assert!(w.copies("idx", 0).iter().all(|c| c.state == ShardState::Unassigned));
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": true})).unwrap();
    let ds = ex["node_allocation_decisions"][0]["deciders"].as_array().unwrap();
    assert!(ds.iter().any(|d| d["explanation"] == json!("no allocations are allowed due to cluster setting [cluster.routing.allocation.enable=none]")));
    w.set_cluster("cluster.routing.allocation.enable", "primaries");
    w.converge();
    assert_eq!(w.primary("idx", 0).state, ShardState::Started);
    assert!(w.copies("idx", 0).iter().any(|c| !c.primary && c.state == ShardState::Unassigned));
    // index-level setting overrides
    w.set_index("idx", "index.routing.allocation.enable", "all");
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn shards_limit() {
    let mut w = World::new(&[node("n1", &[]), node("n2", &[])]);
    w.create("idx", 3, 0, &[("index.routing.allocation.total_shards_per_node", "1")]);
    w.converge();
    assert_eq!(w.health().unassigned_shards, 1);
    let unassigned_shard = (0..3).find(|s| w.primary("idx", *s).state == ShardState::Unassigned).unwrap();
    let ex = w.explain(json!({"index": "idx", "shard": unassigned_shard, "primary": true})).unwrap();
    let ds: Vec<String> = ex["node_allocation_decisions"].as_array().unwrap().iter().map(|n| n["deciders"][0]["explanation"].as_str().unwrap().to_string()).collect();
    assert!(ds.iter().all(|d| d == "too many shards [1] allocated to this node for index [idx], index setting [index.routing.allocation.total_shards_per_node=1]"), "{ds:?}");
}

#[test]
fn throttling_initial_primaries_and_recoveries() {
    let mut w = World::new(&[node("n1", &[])]);
    w.create("idx", 6, 0, &[]);
    w.reroute();
    let init = w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Initializing).count();
    assert_eq!(init, 4);
    let throttled: Vec<_> = w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Unassigned).collect();
    assert_eq!(throttled.len(), 2);
    assert!(throttled.iter().all(|c| c.unassigned_info.as_ref().unwrap().last_allocation_status == "deciders_throttled"));
    assert_eq!(w.health().status, HealthStatus::Yellow); // new primaries throttled: yellow, not red
    let s = throttled[0].shard;
    let ex = w.explain(json!({"index": "idx", "shard": s, "primary": true})).unwrap();
    assert_eq!(ex["can_allocate"], json!("throttled"));
    assert_eq!(ex["node_allocation_decisions"][0]["deciders"][0]["explanation"], json!("reached the limit of ongoing initial primary recoveries [4], cluster setting [cluster.routing.allocation.node_initial_primaries_recoveries=4]"));
    w.converge();
    // peer recoveries: at most 2 incoming per node
    let mut w = World::new(&[node("n1", &[]), node("n2", &[])]);
    w.create("idx", 4, 0, &[("index.routing.allocation.require._name", "n1")]);
    w.converge();
    w.state = w.svc.update_number_of_replicas(&w.state, &["idx".into()], 1, w.now).unwrap();
    w.check();
    w.reroute();
    assert_eq!(w.state.routing_table.shards_on_node("n2").filter(|c| c.state == ShardState::Initializing).count(), 0, "filter keeps replicas off n2 too");
    w.set_index("idx", "index.routing.allocation.require._name", "");
    w.reroute();
    assert_eq!(w.state.routing_table.shards_on_node("n2").filter(|c| c.state == ShardState::Initializing).count(), 2);
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn disk_watermarks() {
    let gb = 1u64 << 30;
    let mut w = three();
    // n1 above low (88% used), n2 above high (92%), n3 fine
    w.info.nodes.insert("n1".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 12 * gb });
    w.info.nodes.insert("n2".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 8 * gb });
    w.info.nodes.insert("n3".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 50 * gb });
    w.create("idx", 3, 1, &[]);
    w.converge();
    // new primaries may go above low (n1) but not above high (n2); replicas only below low
    assert_eq!(w.per_node()["n2"], 0);
    for c in w.state.routing_table.shards_on_node("n1") {
        assert!(c.primary);
    }
    let r = w.state.routing_table.all_shards().find(|c| !c.primary && c.state == ShardState::Unassigned).cloned();
    if let Some(r) = r {
        let ex = w.explain(json!({"index": "idx", "shard": r.shard, "primary": false})).unwrap();
        let n1 = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|n| n["node_id"] == json!("n1")).unwrap().clone();
        let d = n1["deciders"].as_array().unwrap().iter().find(|d| d["decider"] == json!("disk_threshold")).unwrap().clone();
        assert_eq!(d["explanation"], json!("the node is above the low watermark cluster setting [cluster.routing.allocation.disk.watermark.low=85%], having less than the minimum required [15gb] free space, actual free: [12gb], actual used: [88%]"));
    }
    // n3 fills past high: its shards must move off (to below-low nodes only)
    w.info.nodes.insert("n2".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 60 * gb });
    w.converge();
    w.info.nodes.insert("n3".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 4 * gb });
    let on_n3 = w.per_node()["n3"];
    assert!(on_n3 > 0);
    let some = w.state.routing_table.shards_on_node("n3").next().unwrap().clone();
    let ex = w.explain(json!({"index": "idx", "shard": some.shard, "primary": some.primary})).unwrap();
    assert_eq!(ex["can_remain_on_current_node"], json!("no"));
    assert_eq!(ex["can_remain_decisions"][0]["explanation"], json!("the shard cannot remain on this node because it is above the high watermark cluster setting [cluster.routing.allocation.disk.watermark.high=90%] and there is less than the required [10gb] free space on node, actual free: [4gb], actual used: [96%]"));
    // flood stage → read-only-allow-delete for indices with shards there
    let fs = w.svc.disk_flood_stage(&w.state, &w.info);
    assert_eq!(fs.block, BTreeSet::from(["idx".to_string()]));
    w.state = apply_read_only_allow_delete(&w.state, &fs.block, true);
    assert!(w.state.blocks.index_blocked("idx", "write").is_some());
    // A failed disk probe is unknown, not evidence that the disk recovered.
    let saved_info = w.info.clone();
    w.info.nodes.clear();
    assert!(w.svc.disk_flood_stage(&w.state, &w.info).release.is_empty());
    w.info = saved_info;
    w.converge();
    // free space recovers → release
    w.info.nodes.insert("n3".into(), DiskUsage { path: "/d".into(), total_bytes: 100 * gb, free_bytes: 50 * gb });
    let fs = w.svc.disk_flood_stage(&w.state, &w.info);
    assert!(fs.block.is_empty());
    assert_eq!(fs.release, BTreeSet::from(["idx".to_string()]));
}

#[test]
fn data_tiers_and_node_version() {
    let mut hot = node("hot", &[]);
    hot.roles = BTreeSet::from([Role::DataHot, Role::Master]);
    let mut w = World::new(&[hot.clone()]);
    w.create("content", 1, 0, &[]);
    w.create("hotidx", 1, 0, &[("index.routing.allocation.include._tier_preference", "data_hot,data_content")]);
    w.converge();
    assert_eq!(w.primary("content", 0).state, ShardState::Unassigned);
    assert_eq!(w.primary("hotidx", 0).state, ShardState::Started);
    let ex = w.explain(json!({"index": "content", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["node_allocation_decisions"][0]["deciders"][0]["explanation"], json!("index has a preference for tiers [data_content], but no nodes for any of those tiers are available in the cluster"));
    // replicas never go to an older node than the primary's
    let mut old = node("old", &[]);
    old.version = "8.14.0".into();
    let mut w = World::new(&[node("new", &[]), old]);
    w.create("idx", 1, 1, &[("index.routing.allocation.include._name", "new")]);
    w.converge();
    w.set_index("idx", "index.routing.allocation.include._name", "");
    w.converge();
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": false})).unwrap();
    let ds = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|n| n["node_id"] == json!("old")).unwrap()["deciders"].clone();
    assert!(ds.as_array().unwrap().iter().any(|d| d["explanation"] == json!("cannot allocate replica shard to a node with version [8.14.0] since this is older than the primary version [8.15.0]")));
}

#[test]
fn node_shutdown_remove_and_restart() {
    let mut w = three();
    w.create("idx", 3, 1, &[]);
    w.converge();
    w.state.metadata.customs.insert("node_shutdown".into(), json!({"nodes": {"n1": {"node_id": "n1", "type": "remove", "reason": "scale down"}, "n2": {"node_id": "n2", "type": "restart", "allocation_delay": "10m"}}}));
    w.converge();
    assert_eq!(w.per_node()["n1"], 0, "drained");
    assert_eq!(w.health().status, HealthStatus::Green);
    // restart: replicas wait the shutdown's allocation_delay (10m > 1m index delay)
    let on_n2: Vec<ShardRouting> = w.state.routing_table.shards_on_node("n2").cloned().collect();
    assert!(!on_n2.is_empty());
    w.kill("n2");
    let delayed: Vec<_> = w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Unassigned).collect();
    assert!(delayed.iter().all(|c| c.unassigned_info.as_ref().unwrap().reason == UnassignedReason::NodeRestarting && c.unassigned_info.as_ref().unwrap().delayed));
    w.now += 2 * MIN;
    w.reroute();
    assert!(w.state.routing_table.all_shards().any(|c| c.state == ShardState::Unassigned), "still delayed after 2m");
    w.now += 9 * MIN;
    w.reroute();
    // n1 is being removed and n2 is gone: nowhere to put the second copy; stays yellow, not delayed
    assert_eq!(w.health().delayed_unassigned_shards, 0);
}

#[test]
fn node_shutdown_replace_moves_only_to_target() {
    let mut w = three();
    w.create("idx", 3, 1, &[]);
    w.converge();
    let before = w.per_node()["n1"];
    assert!(before > 0);
    w.state.metadata.customs.insert("node_shutdown".into(), json!({"nodes": {"n1": {"node_id": "n1", "type": "replace", "reason": "r", "target_node_name": "n4"}}}));
    // target not yet in the cluster: copies elsewhere -> COMPLETE (ignored)
    w.converge();
    assert_eq!(w.per_node()["n1"], before, "no target: nothing moves");
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "replace", true, w.now, &w.info);
    assert_eq!(st.status, "COMPLETE");
    assert_eq!(st.explanation.as_deref(), Some(format!("[{before}] shards cannot be moved away from this node but have at least one copy on another node in the cluster").as_str()));
    assert_eq!(st.to_json().get("started_shards"), None);
    w.add_node(node("n4", &[]));
    w.reroute();
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "replace", true, w.now, &w.info);
    assert_eq!(st.status, "IN_PROGRESS");
    assert!(st.to_json()["relocating_shards"].as_u64().unwrap() > 0);
    // every relocation goes to the replacement target
    for c in w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Relocating) {
        assert_eq!(c.relocating_node_id.as_deref(), Some("n4"));
    }
    w.converge();
    assert_eq!(w.per_node()["n1"], 0);
    assert_eq!(w.per_node()["n4"], before);
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "replace", true, w.now, &w.info);
    assert_eq!(st.to_json(), json!({"status": "COMPLETE", "started_shards": 0, "relocating_shards": 0, "initializing_shards": 0, "shard_migrations_remaining": 0}));
}

#[test]
fn node_shutdown_migration_status() {
    let mut w = three();
    // restart / unknown node
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "restart", false, w.now, &w.info);
    assert_eq!(st.to_json(), json!({"status": "COMPLETE", "shard_migrations_remaining": 0, "explanation": "no shard relocation is necessary for a node restart"}));
    let st = w.svc.shutdown_migration_status(&w.state, "nx", "remove", false, w.now, &w.info);
    assert_eq!(st.to_json(), json!({"status": "NOT_STARTED", "shard_migrations_remaining": 0, "explanation": "node is not currently part of the cluster"}));
    // a single-copy index pinned to n1 cannot move
    w.create("pinned", 1, 0, &[("index.routing.allocation.require._name", "n1")]);
    w.converge();
    w.state.metadata.customs.insert("node_shutdown".into(), json!({"nodes": {"n1": {"node_id": "n1", "type": "remove", "reason": "r"}}}));
    w.converge();
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "remove", true, w.now, &w.info);
    assert_eq!(st.status, "STALLED");
    assert_eq!(st.remaining, 1);
    assert_eq!(st.explanation.as_deref(), Some("shard [0] [primary] of index [pinned] cannot move, see [node_allocation_decision] for details or use the cluster allocation explain API"));
    let d = st.decision.clone().unwrap();
    assert_eq!(d["can_remain_on_current_node"], json!("no"));
    assert_eq!(d["can_move_to_other_node"], json!("no"));
    assert!(d.get("index").is_none() && d.get("current_node").is_none());
    assert!(st.to_json().get("started_shards").is_none());
    // the node leaves: the primary is unassigned with last_node n1
    w.kill("n1");
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "remove", true, w.now, &w.info);
    assert_eq!(st.status, "STALLED");
    assert_eq!(st.explanation.as_deref(), Some("shard [0] [primary] of index [pinned] is unassigned, see [node_allocation_decision] for details or use the cluster allocation explain API"));
    // the only copy was on n1: ES reports no_valid_shard_copy
    assert_eq!(st.decision.unwrap()["can_allocate"], json!("no_valid_shard_copy"));
    // node gone and never seen
    let st = w.svc.shutdown_migration_status(&w.state, "n1", "remove", false, w.now, &w.info);
    assert_eq!(st.status, "NOT_STARTED");
}

// ------------------------------------------------------------- balancing --

#[test]
fn rebalance_converges_and_does_not_oscillate() {
    let mut w = World::new(&[node("n1", &[])]);
    w.create("a", 4, 0, &[]);
    w.create("b", 2, 1, &[]);
    w.converge();
    w.add_node(node("n2", &[]));
    w.add_node(node("n3", &[]));
    w.reroute();
    // at most cluster_concurrent_rebalance (2) relocations at once, plus replica recoveries
    assert!(w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Relocating).count() <= 2);
    w.converge();
    let per = w.per_node();
    let (min, max) = (per.values().min().unwrap(), per.values().max().unwrap());
    assert!(max - min <= 1, "{per:?}");
    assert_eq!(w.health().status, HealthStatus::Green);
    // stable
    for _ in 0..3 {
        let before = w.state.routing_table.clone();
        w.reroute();
        assert_eq!(before, w.state.routing_table, "no oscillation");
    }
    // explain of a balanced started shard
    let ex = w.explain(json!({"index": "a", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["can_remain_on_current_node"], json!("yes"));
    assert_eq!(ex["can_rebalance_cluster"], json!("yes"));
    assert_eq!(ex["can_rebalance_to_other_node"], json!("no"));
    assert_eq!(ex["rebalance_explanation"], json!("cannot rebalance as no target node exists that can both allocate this shard and improve the cluster balance"));
    assert!(ex["current_node"]["weight_ranking"].as_u64().is_some());
}

#[test]
fn rebalance_respects_settings() {
    let mut w = World::new(&[node("n1", &[])]);
    w.create("a", 4, 0, &[]);
    w.converge();
    w.add_node(node("n2", &[]));
    w.set_cluster("cluster.routing.rebalance.enable", "none");
    w.converge();
    assert_eq!(w.per_node()["n2"], 0);
    w.set_cluster("cluster.routing.rebalance.enable", "all");
    w.set_cluster("cluster.routing.allocation.cluster_concurrent_rebalance", "1");
    w.reroute();
    assert_eq!(w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Relocating).count(), 1);
    // the concurrent rebalance limit throttles further moves
    let ex = w.explain(json!({"index": "a", "shard": w.state.routing_table.shards_on_node("n1").find(|c| c.state == ShardState::Started).unwrap().shard, "primary": true})).unwrap();
    assert_eq!(ex["can_rebalance_cluster"], json!("throttled"));
    assert_eq!(ex["can_rebalance_cluster_decisions"][0]["decider"], json!("concurrent_rebalance"));
    assert_eq!(ex["rebalance_explanation"], json!("rebalancing is throttled"));
    w.converge();
    assert_eq!(w.per_node()["n2"], 2);
}

#[test]
fn primary_relocation_completes() {
    let mut w = World::new(&[node("n1", &[]), node("n2", &[]), node("n3", &[])]);
    w.create("idx", 1, 1, &[("index.routing.allocation.exclude._name", "n3")]);
    w.converge();
    let p = w.primary("idx", 0);
    let pn = p.node_id.clone().unwrap();
    let term = w.term("idx", 0);
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"move": {"index": "idx", "shard": 0, "from_node": pn, "to_node": "n3"}}]}), RerouteFlags { explain: true, ..Default::default() }, w.now, &w.info, &w.stores());
    // n3 excluded: rejected with the filter decision in explain mode
    let res = res.unwrap();
    assert_eq!(res.explanations[0]["decisions"].as_array().unwrap().iter().find(|d| d["decision"] == json!("NO")).unwrap()["decider"], json!("filter"));
    w.set_index("idx", "index.routing.allocation.exclude._name", "");
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"move": {"index": "idx", "shard": 0, "from_node": pn, "to_node": "n3"}}]}), RerouteFlags { explain: true, ..Default::default() }, w.now, &w.info, &w.stores()).unwrap();
    assert_eq!(res.explanations[0]["command"], json!("move"));
    assert_eq!(res.explanations[0]["parameters"], json!({"index": "idx", "shard": 0, "from_node": pn, "to_node": "n3"}));
    w.state = res.state;
    w.check();
    let src = w.primary("idx", 0);
    assert_eq!(src.state, ShardState::Relocating);
    assert_eq!(src.relocating_node_id.as_deref(), Some("n3"));
    assert_eq!(w.health().relocating_shards, 1);
    w.start_all();
    let np = w.primary("idx", 0);
    assert_eq!(np.node_id.as_deref(), Some("n3"));
    assert_eq!(np.state, ShardState::Started);
    assert!(w.in_sync("idx", 0).contains(np.allocation_id().unwrap()));
    assert!(!w.in_sync("idx", 0).contains(p.allocation_id().unwrap()));
    assert_eq!(w.term("idx", 0), term, "a relocation hands off the primary within the same term");
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn relocation_target_failure_cancels_relocation() {
    let mut w = three();
    w.create("idx", 1, 0, &[]);
    w.converge();
    let p = w.primary("idx", 0);
    let to = ["n1", "n2", "n3"].into_iter().find(|n| Some(*n) != p.node_id.as_deref()).unwrap();
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"move": {"index": "idx", "shard": 0, "from_node": p.node_id, "to_node": to}}]}), RerouteFlags::default(), w.now, &w.info, &w.stores()).unwrap();
    w.state = res.state;
    w.check();
    let target = w.copies("idx", 0).into_iter().find(|c| c.state == ShardState::Initializing).unwrap();
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: target.allocation_id().unwrap().into(), message: "recovery failed".into(), failure: None, mark_as_stale: false }], w.now);
    w.check();
    let c = w.copies("idx", 0);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].state, ShardState::Started);
    assert_eq!(c[0].allocation_id(), p.allocation_id());
}

// ---------------------------------------------------------- replicas API --

#[test]
fn auto_expand_replicas() {
    let mut w = three();
    w.create("all", 1, 0, &[("index.auto_expand_replicas", "0-all")]);
    w.create("one", 1, 0, &[("index.auto_expand_replicas", "0-1")]);
    w.create("min1", 1, 0, &[("index.auto_expand_replicas", "1-5")]);
    w.converge();
    assert_eq!(w.state.metadata.indices["all"].number_of_replicas, 2);
    assert_eq!(w.state.metadata.indices["all"].settings["index.number_of_replicas"], json!("2"));
    assert_eq!(w.state.metadata.indices["one"].number_of_replicas, 1);
    assert_eq!(w.state.metadata.indices["min1"].number_of_replicas, 2);
    assert_eq!(w.health().status, HealthStatus::Green);
    w.kill("n3");
    w.converge();
    assert_eq!(w.state.metadata.indices["all"].number_of_replicas, 1);
    assert_eq!(w.health().status, HealthStatus::Green);
    w.kill("n2");
    w.converge();
    assert_eq!(w.state.metadata.indices["all"].number_of_replicas, 0);
    // "1-5" cannot go below 1 with a single node: unchanged
    assert_eq!(w.state.metadata.indices["min1"].number_of_replicas, 1);
    assert_eq!(w.health().status, HealthStatus::Yellow);
}

#[test]
fn update_replicas_and_close_open_delete() {
    let mut w = three();
    w.create("idx", 2, 2, &[]);
    w.converge();
    assert_eq!(w.health().active_shards, 6);
    w.state = w.svc.update_number_of_replicas(&w.state, &["idx".into()], 0, w.now).unwrap();
    w.check();
    assert_eq!(w.state.routing_table.all_shards().count(), 2);
    assert_eq!(w.in_sync("idx", 0).len(), 1);
    w.state = w.svc.update_number_of_replicas(&w.state, &["idx".into()], 1, w.now).unwrap();
    w.check();
    let added = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(added.unassigned_info.unwrap().reason, UnassignedReason::ReplicaAdded);
    w.converge();
    let placement: BTreeMap<(u32, bool), (String, String)> = (0..2).flat_map(|s| w.copies("idx", s)).map(|c| ((c.shard, c.primary), (c.node_id.clone().unwrap(), c.allocation_id().unwrap().to_string()))).collect();
    // close: replicated closed index re-allocates onto the same in-sync copies
    w.state = w.svc.close_index(&w.state, "idx", w.now).unwrap();
    w.check();
    assert_eq!(w.state.metadata.indices["idx"].state, IndexState::Close);
    assert!(w.state.blocks.index_blocked("idx", "read").is_some());
    let p = w.primary("idx", 0);
    assert_eq!(p.unassigned_info.as_ref().unwrap().reason, UnassignedReason::IndexClosed);
    assert!(matches!(p.recovery_source, Some(RecoverySource::ExistingStore { .. })));
    w.converge();
    for s in 0..2 {
        let p = w.primary("idx", s);
        assert_eq!(placement[&(s, true)], (p.node_id.clone().unwrap(), p.allocation_id().unwrap().to_string()));
    }
    assert_eq!(w.health().status, HealthStatus::Green);
    w.state = w.svc.open_index(&w.state, "idx", w.now).unwrap();
    assert_eq!(w.primary("idx", 0).unassigned_info.as_ref().unwrap().reason, UnassignedReason::IndexReopened);
    assert!(w.state.blocks.index_blocked("idx", "read").is_none());
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
    w.state = w.svc.delete_index(&w.state, "idx");
    w.check();
    assert!(w.state.routing_table.indices.is_empty());
    assert_eq!(w.health().status, HealthStatus::Green);
}

/// Nodes that restore snapshot files (`restore_in_progress` decider). A
/// node making any capability claim must advertise its segment formats too
/// (`segment_format` decider), as every advertising build does.
fn three_restoring() -> World {
    use crate::compatibility::{CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let attrs = [(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, "11"), (SEGMENT_WRITE, "11"), (SNAPSHOT_FILE_RESTORE_ATTR, "1")];
    World::new(&[node("n1", &attrs), node("n2", &attrs), node("n3", &attrs)])
}

#[test]
fn restored_index_uses_snapshot_source() {
    let mut w = three_restoring();
    w.state.metadata.indices.insert("r".into(), index_meta("r", 1, 1, &[]));
    let src = RecoverySource::Snapshot { repository: "repo".into(), snapshot: "snap".into(), index: "r".into() };
    w.state = w.svc.add_restored_index_routing(&w.state, "r", src.clone(), true, w.now).unwrap();
    w.check();
    let p = w.primary("r", 0);
    assert_eq!(p.recovery_source, Some(src));
    assert_eq!(p.unassigned_info.as_ref().unwrap().reason, UnassignedReason::NewIndexRestored);
    assert_eq!(w.health().status, HealthStatus::Yellow);
    w.reroute();
    let p = w.primary("r", 0);
    assert_eq!(w.in_sync("r", 0), BTreeSet::from([p.allocation_id().unwrap().to_string()]));
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

/// ES `RestoreInProgressAllocationDecider`: a restored primary goes only to
/// a node that adopts snapshot files, and once its restore failed it is not
/// allocated again (the replica waits for it); node loss is not a failure.
#[test]
fn restored_primary_is_not_retried_after_a_failed_restore() {
    let src = RecoverySource::Snapshot { repository: "repo".into(), snapshot: "snap".into(), index: "r".into() };
    // No node advertises file restore: the primary stays unassigned.
    let mut old = three();
    old.state.metadata.indices.insert("r".into(), index_meta("r", 1, 0, &[]));
    old.state = old.svc.add_restored_index_routing(&old.state, "r", src.clone(), true, old.now).unwrap();
    old.reroute();
    assert_eq!(old.primary("r", 0).state, ShardState::Unassigned);
    let e = old.explain(json!({"index": "r", "shard": 0, "primary": true})).unwrap();
    assert!(e.to_string().contains("cannot restore shard files from a snapshot"), "{e}");

    let mut w = three_restoring();
    w.state.metadata.indices.insert("r".into(), index_meta("r", 1, 1, &[]));
    w.state = w.svc.add_restored_index_routing(&w.state, "r", src.clone(), true, w.now).unwrap();
    w.reroute();
    let p = w.primary("r", 0);
    assert_eq!(p.state, ShardState::Initializing);
    // Node loss: the primary is restored elsewhere.
    w.kill(p.node_id.as_deref().unwrap());
    w.reroute();
    let p = w.primary("r", 0);
    assert_eq!((p.state, p.recovery_source.clone()), (ShardState::Initializing, Some(src.clone())));
    // A failed restore: never allocated again.
    let failed = FailedShard { index: "r".into(), shard: 0, allocation_id: p.allocation_id().unwrap().into(), message: "failed to restore from snapshot".into(), failure: Some("checksum failed".into()), mark_as_stale: false };
    w.state = w.svc.apply_failed_shards(&w.state, &[failed], w.now);
    w.check();
    for _ in 0..3 {
        w.reroute();
        assert_eq!(w.start_all(), 0);
    }
    let p = w.primary("r", 0);
    assert_eq!((p.state, p.recovery_source.clone()), (ShardState::Unassigned, Some(src)));
    assert!(w.copies("r", 0).iter().all(|c| c.state == ShardState::Unassigned), "the replica waits for the primary");
    let e = w.explain(json!({"index": "r", "shard": 0, "primary": true})).unwrap();
    assert!(e.to_string().contains("shard has failed to be restored from the snapshot [repo:snap]"), "{e}");
    assert_eq!(w.health().status, HealthStatus::Red);
}

// --------------------------------------------------------------- commands --

#[test]
fn reroute_command_errors_and_cancel() {
    let mut w = three();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let run = |w: &World, body: Value| w.svc.reroute_commands(&w.state, &body, RerouteFlags::default(), w.now, &w.info, &w.stores());
    let err = run(&w, json!({"commands": [{"allocate_empty_primary": {"index": "idx", "shard": 0, "node": "n1"}}]})).unwrap_err();
    assert_eq!(err.2, "[allocate_empty_primary] allocating an empty primary for [idx][0] can result in data loss. Please confirm by setting the accept_data_loss parameter to true");
    let err = run(&w, json!({"commands": [{"allocate_empty_primary": {"index": "idx", "shard": 0, "node": "n1", "accept_data_loss": true}}]})).unwrap_err();
    assert_eq!(err.2, "[allocate_empty_primary] primary [idx][0] is already assigned");
    let err = run(&w, json!({"commands": [{"allocate_replica": {"index": "idx", "shard": 0, "node": "n1"}}]})).unwrap_err();
    assert_eq!(err.2, "[allocate_replica] all copies of [idx][0] are already assigned. Use the move allocation command instead");
    let err = run(&w, json!({"commands": [{"move": {"index": "idx", "shard": 0, "from_node": "zz", "to_node": "n1"}}]})).unwrap_err();
    assert_eq!(err.2, "failed to resolve [zz], no matching nodes");
    let err = run(&w, json!({"commands": [{"move": {"index": "idx", "shard": 0, "to_node": "n1"}}]})).unwrap_err();
    assert_eq!((err.0, err.1.as_str(), err.2.as_str()), (400, "parse_exception", "[move] command missing the from_node parameter"));
    let err = run(&w, json!({"commands": [{"allocate_replica": {"index": "nope", "shard": 0, "node": "n1"}}]})).unwrap_err();
    assert_eq!((err.0, err.2.as_str()), (404, "no such index [nope]"));
    let p = w.primary("idx", 0);
    let pn = p.node_id.clone().unwrap();
    let err = run(&w, json!({"commands": [{"cancel": {"index": "idx", "shard": 0, "node": pn}}]})).unwrap_err();
    assert!(err.2.starts_with("[cancel_allocation] can't cancel [idx][0] on node {"), "{}", err.2);
    assert!(err.2.ends_with(", shard is primary and started"));
    // cancel the replica: it is re-created and reallocated by the follow-up reroute
    let r = w.copies("idx", 0)[1].clone();
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"cancel": {"index": "idx", "shard": 0, "node": r.node_id}}]}), RerouteFlags { explain: true, ..Default::default() }, w.now, &w.info, &w.stores()).unwrap();
    assert_eq!(res.explanations[0]["decisions"][0]["decider"], json!("cancel_allocation_command"));
    assert_eq!(res.explanations[0]["decisions"][0]["decision"], json!("YES"));
    w.state = res.state;
    w.check();
    let rr = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(rr.state, ShardState::Initializing);
    assert_ne!(rr.allocation_id(), r.allocation_id());
    // explain mode turns a rejection into a NO explanation
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"allocate_empty_primary": {"index": "idx", "shard": 0, "node": "n1"}}]}), RerouteFlags { explain: true, ..Default::default() }, w.now, &w.info, &w.stores()).unwrap();
    assert_eq!(res.explanations[0]["decisions"][0]["decision"], json!("NO"));
}

#[test]
fn allocate_replica_and_empty_primary() {
    let mut w = three();
    w.set_cluster("cluster.routing.allocation.enable", "primaries");
    w.create("idx", 1, 1, &[]);
    w.converge();
    let target = ["n1", "n2", "n3"].into_iter().find(|n| Some(*n) != w.primary("idx", 0).node_id.as_deref()).unwrap();
    // manual commands ignore `enable`
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"allocate_replica": {"index": "idx", "shard": 0, "node": target}}]}), RerouteFlags { explain: true, ..Default::default() }, w.now, &w.info, &w.stores()).unwrap();
    assert!(res.explanations[0]["decisions"].as_array().unwrap().iter().any(|d| d["explanation"] == json!("explicitly ignoring any disabling of allocation due to manual allocation commands via the reroute API")));
    w.state = res.state;
    w.check();
    assert_eq!(w.copies("idx", 0)[1].node_id.as_deref(), Some(target));
    // lose every copy → empty primary on request
    w.start_all();
    let mut nodes: Vec<String> = w.state.nodes.keys().cloned().collect();
    nodes.retain(|n| w.copies("idx", 0).iter().any(|c| c.node_id.as_deref() == Some(n.as_str())));
    for n in &nodes {
        w.kill(n);
        w.disk.retain(|(_, _, dn), _| dn != n);
    }
    w.reroute();
    assert_eq!(w.health().status, HealthStatus::Red);
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": true})).unwrap();
    assert_eq!(ex["allocate_explanation"], json!("cannot allocate because a previous copy of the primary shard existed but can no longer be found on the nodes in the cluster"));
    let term = w.term("idx", 0);
    let live = w.state.nodes.keys().next().unwrap().clone();
    let res = w.svc.reroute_commands(&w.state, &json!({"commands": [{"allocate_empty_primary": {"index": "idx", "shard": 0, "node": live, "accept_data_loss": true}}]}), RerouteFlags::default(), w.now, &w.info, &w.stores()).unwrap();
    w.state = res.state;
    w.check();
    let p = w.primary("idx", 0);
    assert_eq!(p.recovery_source, Some(RecoverySource::EmptyStore));
    assert_eq!(p.unassigned_info.as_ref().unwrap().reason, UnassignedReason::ForcedEmptyPrimary);
    assert_eq!(w.in_sync("idx", 0), BTreeSet::from([p.allocation_id().unwrap().to_string()]));
    assert_eq!(w.term("idx", 0), term + 1);
}

// ---------------------------------------------------------------- explain --

#[test]
fn explain_errors_and_shapes() {
    let mut w = three();
    w.create("idx", 1, 0, &[]);
    w.converge();
    let err = w.explain(json!({})).unwrap_err();
    assert_eq!(err.0, 400);
    assert_eq!(err.1, "illegal_argument_exception");
    assert!(err.2.starts_with("No shard was specified in the request which means the response should explain a randomly-chosen unassigned shard, but there are no unassigned shards in this cluster."));
    let err = w.explain(json!({"index": "idx"})).unwrap_err();
    assert_eq!(err.2, "Validation Failed: 1: shard must be specified;2: primary must be specified;");
    let err = w.explain(json!({"index": "nope", "shard": 0, "primary": true})).unwrap_err();
    assert_eq!((err.0, err.1.as_str()), (404, "index_not_found_exception"));
    let err = w.explain(json!({"index": "idx", "shard": 0, "primary": false})).unwrap_err();
    assert!(err.2.starts_with("unable to find any shards to explain"));
    let ex = w.explain(json!({"index": "idx", "shard": 0, "primary": true})).unwrap();
    let keys: Vec<&str> = ex.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    assert_eq!(keys, vec!["index", "shard", "primary", "current_state", "current_node", "can_remain_on_current_node", "can_rebalance_cluster", "can_rebalance_to_other_node", "rebalance_explanation", "node_allocation_decisions"]);
    assert_eq!(ex["current_state"], json!("started"));
    let nd = &ex["node_allocation_decisions"][0];
    let keys: Vec<&str> = nd.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    assert_eq!(&keys[..6], &["node_id", "node_name", "transport_address", "roles", "node_attributes", "node_decision"]);
    // include_yes_decisions lists every decider
    let ex = w.svc.explain(&w.state, &json!({"index": "idx", "shard": 0, "primary": true}), ExplainOptions { include_yes_decisions: true, include_disk_info: true }, w.now, &w.info, &w.stores()).unwrap();
    assert!(ex["node_allocation_decisions"][0]["deciders"].as_array().unwrap().len() >= 10);
    assert!(ex.get("cluster_info").is_some());
    // unassigned shape
    w.create("u", 1, 0, &[("index.routing.allocation.require._name", "none")]);
    w.reroute();
    let ex = w.explain(json!({})).unwrap();
    let keys: Vec<&str> = ex.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    assert_eq!(keys, vec!["note", "index", "shard", "primary", "current_state", "unassigned_info", "can_allocate", "allocate_explanation", "node_allocation_decisions"]);
    assert_eq!(ex["unassigned_info"]["reason"], json!("INDEX_CREATED"));
    assert_eq!(ex["unassigned_info"]["last_allocation_status"], json!("no"));
    assert_eq!(ex["node_allocation_decisions"].as_array().unwrap().len(), 3);
    assert_eq!(w.health().status, HealthStatus::Red);
}

#[test]
fn randomized_churn_keeps_invariants_and_converges() {
    let mut greens = 0;
    for s0 in 0..20u64 {
        greens += churn(s0) as u32;
    }
    // most runs heal fully; the rest are red only because every in-sync copy was lost
    eprintln!("churn: {greens}/20 runs healed to green");
    assert!(greens > 0);
}

fn churn(s0: u64) -> bool {
    // deterministic LCG so failures reproduce
    let mut seed: u64 = 0x9e3779b97f4a7c15 ^ s0.wrapping_mul(0xbf58476d1ce4e5b9);
    let mut rnd = move |n: u64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) % n
    };
    let all: Vec<DiscoveryNode> = (1..=5).map(|i| node(&format!("n{i}"), &[("zone", if i % 2 == 0 { "a" } else { "b" })])).collect();
    let mut w = World::new(&all[..3]);
    w.create("a", 3, 1, &[]);
    w.create("b", 2, 2, &[]);
    w.create("c", 1, 1, &[("index.auto_expand_replicas", "0-all")]);
    w.converge();
    for step in 0..60 {
        match rnd(6) {
            0 => {
                // a node leaves (keep at least one)
                if w.state.nodes.len() > 1 {
                    let ids: Vec<String> = w.state.nodes.keys().cloned().collect();
                    let id = ids[rnd(ids.len() as u64) as usize].clone();
                    w.kill(&id);
                }
            }
            1 => {
                // a node (re)joins with whatever it has on disk
                let missing: Vec<&DiscoveryNode> = all.iter().filter(|n| !w.state.nodes.contains_key(&n.id)).collect();
                if !missing.is_empty() {
                    let n = missing[rnd(missing.len() as u64) as usize].clone();
                    w.add_node(n);
                }
            }
            2 => {
                // a started replica is failed by its primary (stale)
                let reps: Vec<ShardRouting> = w.state.routing_table.all_shards().filter(|c| !c.primary && c.state == ShardState::Started).cloned().collect();
                if !reps.is_empty() {
                    let r = &reps[rnd(reps.len() as u64) as usize];
                    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: r.index.clone(), shard: r.shard, allocation_id: r.allocation_id().unwrap().into(), message: "stale".into(), failure: None, mark_as_stale: true }], w.now);
                    w.check();
                }
            }
            3 => {
                // an initializing copy fails its recovery
                let init: Vec<ShardRouting> = w.state.routing_table.all_shards().filter(|c| c.state == ShardState::Initializing).cloned().collect();
                if !init.is_empty() {
                    let r = &init[rnd(init.len() as u64) as usize];
                    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: r.index.clone(), shard: r.shard, allocation_id: r.allocation_id().unwrap().into(), message: "recovery".into(), failure: None, mark_as_stale: false }], w.now);
                    w.check();
                }
            }
            4 => w.now += 2 * MIN,
            _ => {
                w.reroute();
                w.start_all();
            }
        }
        let _ = step;
    }
    // everyone back, failures retried: the cluster must heal to green and stay put
    for n in &all {
        w.add_node(n.clone());
    }
    w.now += 10 * MIN;
    w.state = w.svc.reset_failed_allocations(&w.state, w.now);
    w.converge();
    let h = w.health();
    // a shard whose every in-sync copy was lost stays red (never silently re-created)
    for ih in &h.indices {
        for sh in &ih.shards {
            if sh.status != HealthStatus::Green {
                let p = w.primary(&ih.index, sh.shard);
                assert_eq!(p.state, ShardState::Unassigned, "{} {} not green but primary assigned", ih.index, sh.shard);
            }
        }
    }
    for _ in 0..3 {
        let before = w.state.routing_table.clone();
        w.reroute();
        assert_eq!(before, w.state.routing_table, "no oscillation after convergence (seed {s0})");
    }
    h.status == HealthStatus::Green
}

// ------------------------------------------------------------ stateless --

fn stateless_node(id: &str, roles: &[Role]) -> DiscoveryNode {
    DiscoveryNode { roles: roles.iter().copied().collect(), ..node(id, &[]) }
}

/// The stateless test cluster (`DefaultStatelessLocalConfigProvider`): an
/// index node `[master,remote_cluster_client,ingest,index]` and a search
/// node `[remote_cluster_client,search]`.
fn stateless_pair() -> World {
    World::new(&[
        stateless_node("index", &[Role::Master, Role::RemoteClusterClient, Role::Ingest, Role::Index]),
        stateless_node("search", &[Role::RemoteClusterClient, Role::Search]),
    ])
}

#[test]
fn stateless_roles() {
    assert_eq!((Role::parse("index"), Role::parse("search")), (Some(Role::Index), Some(Role::Search)));
    assert_eq!((Role::Index.abbreviation(), Role::Search.abbreviation()), ('I', 'S'));
    assert!(Role::Index.is_data() && Role::Search.is_data(), "both can contain data (`canContainData`)");
    assert!(Role::Index.is_stateless() && Role::Search.is_stateless() && !Role::Data.is_stateless());
    // Canonical (alphabetical) order, as ES lists roles.
    let names: Vec<&str> = Role::ALL.iter().map(|r| r.name()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    // Not enabled by default on a regular node; on a stateless node `index`
    // replaces the data roles and `search` stays off.
    let d = Role::defaults();
    assert!(!d.contains(&Role::Index) && !d.contains(&Role::Search) && d.contains(&Role::Data));
    let s = Role::stateless_defaults();
    assert!(s.contains(&Role::Index) && s.contains(&Role::Master) && s.contains(&Role::Ingest));
    assert!(!s.contains(&Role::Search) && !s.iter().any(|r| r.is_data() && *r != Role::Index) && !s.contains(&Role::VotingOnly));
    let sn = stateless_node("s", &[Role::Search]);
    assert_eq!(sn.role_abbreviations(), "S");
    assert!(sn.is_data() && sn.is_stateless());
}

#[test]
fn stateless_places_primaries_on_index_nodes_and_replicas_on_search_nodes() {
    let mut w = stateless_pair();
    w.create("idx", 4, 1, &[]);
    w.converge();
    for shard in 0..4 {
        let copies = w.copies("idx", shard);
        assert!(copies.iter().all(|c| c.state == ShardState::Started), "{copies:?}");
        let p = copies.iter().find(|c| c.primary).unwrap();
        let r = copies.iter().find(|c| !c.primary).unwrap();
        assert_eq!((p.node_id.as_deref(), r.node_id.as_deref()), (Some("index"), Some("search")));
    }
    assert_eq!(w.health().status, HealthStatus::Green);
    // A second replica has no search node to go to.
    w.create("two", 1, 2, &[]);
    w.converge();
    assert_eq!(w.copies("two", 0).iter().filter(|c| c.assigned()).count(), 2);
    let ex = w.explain(json!({"index": "two", "shard": 0, "primary": false})).unwrap();
    let text = ex.to_string();
    assert!(text.contains("stateless_shard_role") || text.contains("same_shard"), "{text}");
}

#[test]
fn stateless_decider_explains_role_mismatches() {
    let mut w = stateless_pair();
    // Two replicas: the second has nowhere to go.
    w.create("two", 1, 2, &[]);
    w.converge();
    let ex = w.explain(json!({"index": "two", "shard": 0, "primary": false})).unwrap().to_string();
    assert!(ex.contains("stateless_shard_role"), "{ex}");
    assert!(ex.contains("shard role [SEARCH_ONLY] does not match stateless node role [index]"), "{ex}");
    // An unassigned primary would only be placed on the index node.
    let mut w = World::new(&[stateless_node("search", &[Role::Search]), stateless_node("m", &[Role::Master])]);
    w.create("p", 1, 0, &[]);
    w.converge();
    assert!(!w.primary("p", 0).assigned());
    let ex = w.explain(json!({"index": "p", "shard": 0, "primary": true})).unwrap().to_string();
    assert!(ex.contains("shard role [INDEX_ONLY] does not match stateless node role [search]"), "{ex}");
}

#[test]
fn stateless_never_promotes_a_search_replica() {
    let mut w = stateless_pair();
    w.create("idx", 1, 1, &[]);
    w.converge();
    let r = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert_eq!(r.node_id.as_deref(), Some("search"));
    let term = w.term("idx", 0);
    w.kill("index");
    // The search copy stays a replica; the primary waits for an index node.
    let copies = w.copies("idx", 0);
    let p = copies.iter().find(|c| c.primary).unwrap();
    assert_eq!(p.state, ShardState::Unassigned, "{copies:?}");
    let still = copies.iter().find(|c| c.allocation_id() == r.allocation_id()).unwrap();
    assert!(!still.primary && still.active());
    assert_eq!(w.term("idx", 0), term, "no promotion, no term bump");
    assert_eq!(w.health().status, HealthStatus::Red);
    w.reroute();
    assert!(!w.copies("idx", 0).iter().any(|c| c.primary && c.node_id.as_deref() == Some("search")));
}

#[test]
fn stateless_reads_go_to_search_copies_only() {
    let mut w = stateless_pair();
    w.create("idx", 2, 1, &[]);
    w.converge();
    for k in 0..2 {
        let t = w.state.routing_table.shard_table("idx", k).unwrap();
        let readers: Vec<&str> = t.searchable_copies(&w.state.nodes).filter_map(|c| c.node_id.as_deref()).collect();
        assert_eq!(readers, ["search"], "the indexing copy serves no reads");
        assert!(!t.stateless_without_search_copy(&w.state.nodes));
    }
    // No replicas: the shard has no search copy at all.
    w.create("solo", 1, 0, &[]);
    w.converge();
    let t = w.state.routing_table.shard_table("solo", 0).unwrap();
    assert!(t.active_copies().count() == 1 && t.searchable_copies(&w.state.nodes).next().is_none());
    assert!(t.stateless_without_search_copy(&w.state.nodes));
    // Regular clusters read from every active copy.
    let mut r = three();
    r.create("idx", 1, 1, &[]);
    r.converge();
    let t = r.state.routing_table.shard_table("idx", 0).unwrap();
    assert_eq!(t.searchable_copies(&r.state.nodes).count(), 2);
    assert!(!t.stateless_without_search_copy(&r.state.nodes));
}

#[test]
fn stateless_primary_moves_to_another_index_node_with_the_object_store() {
    let mut w = World::new(&[
        stateless_node("m", &[Role::Master]),
        stateless_node("i1", &[Role::Index]),
        stateless_node("i2", &[Role::Index]),
        stateless_node("s", &[Role::Search]),
    ]);
    w.create("idx", 1, 1, &[]);
    w.converge();
    let p = w.primary("idx", 0);
    let old = p.node_id.clone().unwrap();
    let other = if old == "i1" { "i2" } else { "i1" };
    let replica = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    let term = w.term("idx", 0);
    w.kill(&old);
    w.reroute();
    // No local copy anywhere else, yet the primary is allocated to the other
    // index node (it recovers from the object store), never to the search node.
    let np = w.primary("idx", 0);
    assert_eq!(np.node_id.as_deref(), Some(other), "{:?}", w.copies("idx", 0));
    assert!(matches!(np.recovery_source, Some(RecoverySource::ExistingStore { .. })), "{np:?}");
    assert_ne!(np.allocation_id(), p.allocation_id(), "a new copy");
    let r = w.copies("idx", 0).into_iter().find(|c| c.allocation_id() == replica.allocation_id()).unwrap();
    assert!(!r.primary && r.active(), "the search copy stays a replica");
    w.converge();
    assert_eq!(w.primary("idx", 0).state, ShardState::Started);
    assert!(w.term("idx", 0) > term);
    assert_eq!(w.health().status, HealthStatus::Green);
}

#[test]
fn stateless_wait_for_one_active_shard_means_a_search_shard() {
    let mut w = stateless_pair();
    w.create("idx", 1, 1, &[]);
    w.reroute();
    // Start only the primary: active, but no search shard yet.
    let p = w.primary("idx", 0);
    w.state = w.svc.apply_started_shards(&w.state, &[StartedShard { index: "idx".into(), shard: 0, allocation_id: p.allocation_id().unwrap().into() }]);
    let t = w.state.routing_table.shard_table("idx", 0).unwrap().clone();
    assert!(t.active_primary().is_some() && !t.one_shard_active(&w.state.nodes), "{:?}", t.copies);
    w.converge();
    assert!(w.state.routing_table.shard_table("idx", 0).unwrap().one_shard_active(&w.state.nodes));
    // Without replicas, or off stateless nodes, the primary is enough.
    w.create("solo", 1, 0, &[]);
    w.converge();
    assert!(w.state.routing_table.shard_table("solo", 0).unwrap().one_shard_active(&w.state.nodes));
    let mut r = three();
    r.create("idx", 1, 1, &[]);
    r.reroute();
    let p = r.primary("idx", 0);
    r.state = r.svc.apply_started_shards(&r.state, &[StartedShard { index: "idx".into(), shard: 0, allocation_id: p.allocation_id().unwrap().into() }]);
    assert!(r.state.routing_table.shard_table("idx", 0).unwrap().one_shard_active(&r.state.nodes));
}

#[test]
fn segment_format_blocks_replica_until_its_reader_is_upgraded() {
    use crate::compatibility::{CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let writer = node("writer", &[(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, "12"), (SEGMENT_WRITE, "12")]);
    let mut w = World::new(&[writer]);
    w.create("idx", 1, 1, &[]);
    w.converge();
    w.state.nodes.insert("legacy".into(), node("legacy", &[]));
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Yellow);
    let ex = w.explain(json!({"index":"idx","shard":0,"primary":false})).unwrap();
    let target = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|d| d["node_id"] == "legacy").unwrap();
    assert!(target["deciders"].as_array().unwrap().iter().any(|d| d["decider"] == "segment_format" && d["decision"] == "NO"));
    let upgraded = node("legacy", &[(CAPABILITIES_SCHEMA,"1"), (SEGMENT_READ_MAX,"12"), (SEGMENT_WRITE,"11")]);
    w.state.nodes.insert(upgraded.id.clone(), upgraded);
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

/// Upgraded nodes still advertise writes below v13 until the floor rises,
/// but a `bbq_disk` copy holds IVF files from the start: no copy of it goes
/// to a node that reads less than v13 (the recovery source would refuse),
/// and it is placed once such a node is upgraded.
#[test]
fn segment_format_keeps_disk_bbq_copies_off_older_readers() {
    use crate::compatibility::{CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let upgraded = |id: &str| node(id, &[(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, "13"), (SEGMENT_WRITE, "12")]);
    let previous = |id: &str| node(id, &[(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, "12"), (SEGMENT_WRITE, "12")]);
    let mut w = World::new(&[upgraded("upgraded"), previous("old")]);
    w.create("disk", 1, 1, &[]);
    w.state.metadata.indices.get_mut("disk").unwrap().mappings = json!({"properties": {"v": {"type": "dense_vector", "dims": 64, "index_options": {"type": "bbq_disk"}}}}).into();
    w.create("plain", 1, 1, &[]);
    w.converge();
    assert_eq!(w.primary("disk", 0).node_id.as_deref(), Some("upgraded"));
    assert!(w.copies("disk", 0).iter().all(|c| c.node_id.as_deref() != Some("old")), "{:?}", w.copies("disk", 0));
    assert!(w.copies("plain", 0).iter().all(|c| c.assigned()), "other indices still use the older node");
    let ex = w.explain(json!({"index":"disk","shard":0,"primary":false})).unwrap();
    let target = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|d| d["node_id"] == "old").unwrap();
    let decider = target["deciders"].as_array().unwrap().iter().find(|d| d["decider"] == "segment_format").unwrap();
    assert_eq!(decider["decision"], "NO");
    assert!(decider["explanation"].as_str().unwrap().contains("requires v13"), "{decider}");
    w.state.nodes.insert("old".into(), upgraded("old"));
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green);
}

/// A recovery the source refused for the target's segment format is not an
/// allocation failure: it is not counted toward `max_retry`, the copy is
/// kept off nodes reading less from then on, and a count an older master
/// made does not block an upgraded target.
#[test]
fn segment_format_refusal_is_not_an_allocation_failure() {
    use crate::compatibility::{segment_format_refusal, CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let reader = |id: &str, read: &str| node(id, &[(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, read), (SEGMENT_WRITE, "12")]);
    let mut w = World::new(&[reader("primary", "13")]);
    w.create("idx", 1, 1, &[]);
    w.converge();
    w.add_node(reader("r", "12"));
    w.reroute();
    let replica = w.copies("idx", 0).into_iter().find(|c| !c.primary && c.node_id.as_deref() == Some("r")).expect("the replica initializes on r");
    let message = format!("failed recovery: file recovery source start [idx][0]: [recovery_failed_exception] {}",
        segment_format_refusal(13, "file [core-1/ivf_0000_0000000000000001.rivc] requires segment format v13, target reads through v12"));
    w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: replica.allocation_id().unwrap().into(), message, failure: None, mark_as_stale: false }], w.now);
    w.check();
    w.converge();
    let failed = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap();
    assert!(!failed.assigned(), "the refused node is not retried: {failed:?}");
    assert_eq!(failed.unassigned_info.as_ref().unwrap().failed_allocations, 0, "a refusal is not counted");
    let ex = w.explain(json!({"index":"idx","shard":0,"primary":false})).unwrap();
    let target = ex["node_allocation_decisions"].as_array().unwrap().iter().find(|d| d["node_id"] == "r").unwrap();
    assert!(target["deciders"].as_array().unwrap().iter().any(|d| d["decider"] == "segment_format" && d["decision"] == "NO"), "{target}");
    // An older master counted every refusal up to max_retry.
    for copy in w.state.routing_table.indices.get_mut("idx").unwrap().shards[0].copies.iter_mut().filter(|c| !c.primary) {
        copy.unassigned_info.as_mut().unwrap().failed_allocations = 5;
    }
    w.state.nodes.insert("r".into(), reader("r", "13"));
    w.converge();
    assert_eq!(w.health().status, HealthStatus::Green, "no _cluster/reroute?retry_failed needed");
}

/// The rolling upgrade's sequence: an older master counted four refusals
/// (tagged) and put a fifth attempt on its own node, which then restarted
/// on the upgraded binary. The upgraded master fails that copy ("node
/// restarted"): that must neither count on top of the refusals nor drop
/// their tag, so the copy is placed on the upgraded node without
/// retry_failed. Real failures after it still count up to max_retry.
#[test]
fn a_failure_after_counted_refusals_starts_the_count_afresh() {
    use crate::compatibility::{refused_segment_format, segment_format_refusal, CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let reader = |id: &str, read: &str| node(id, &[(CAPABILITIES_SCHEMA, "1"), (SEGMENT_READ_MAX, read), (SEGMENT_WRITE, "11")]);
    let mut w = World::new(&[reader("primary", "13")]);
    w.create("idx", 1, 1, &[]);
    w.converge();
    w.add_node(reader("r", "13"));
    w.reroute();
    let tagged = format!("failed shard on node [r]: failed recovery: [recovery_failed_exception] {}", segment_format_refusal(13, "node [r] reads segment formats through v12"));
    let fail = |w: &mut World, message: &str| {
        let copy = w.copies("idx", 0).into_iter().find(|c| !c.primary && c.node_id.as_deref() == Some("r")).expect("the replica initializes on r");
        w.state = w.svc.apply_failed_shards(&w.state, &[FailedShard { index: "idx".into(), shard: 0, allocation_id: copy.allocation_id().unwrap().into(), message: message.into(), failure: None, mark_as_stale: false }], w.now);
        w.check();
    };
    // What the older master left on the copy it was initializing on r.
    for copy in w.state.routing_table.indices.get_mut("idx").unwrap().shards[0].copies.iter_mut().filter(|c| !c.primary) {
        let info = copy.unassigned_info.get_or_insert_with(|| UnassignedInfo { reason: UnassignedReason::AllocationFailed, at_millis: 0, failed_allocations: 0, delayed: false, details: None, last_allocation_status: "no_attempt".into(), last_node: None });
        info.failed_allocations = 4;
        info.details = Some(tagged.clone());
    }
    fail(&mut w, "node restarted");
    let info = w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap().unassigned_info.unwrap();
    assert_eq!(info.failed_allocations, 1, "the refusals the older master counted are not added: {info:?}");
    assert_eq!(refused_segment_format(info.details.as_deref().unwrap()), Some(13), "the tag survives: {info:?}");
    w.reroute();
    // Real failures still count, and a carried tag never resets the count.
    for expected in 2..=5 {
        fail(&mut w, "disk broke");
        assert_eq!(w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap().unassigned_info.unwrap().failed_allocations, expected);
        w.reroute();
    }
    assert!(!w.copies("idx", 0).into_iter().find(|c| !c.primary).unwrap().assigned(), "max_retry still applies to real failures");
    // The fresh count admits the upgraded target.
    let mut fresh = World::new(&[reader("primary", "13")]);
    fresh.create("idx", 1, 1, &[]);
    fresh.converge();
    fresh.add_node(reader("r", "13"));
    fresh.reroute();
    for copy in fresh.state.routing_table.indices.get_mut("idx").unwrap().shards[0].copies.iter_mut().filter(|c| !c.primary) {
        let info = copy.unassigned_info.get_or_insert_with(|| UnassignedInfo { reason: UnassignedReason::AllocationFailed, at_millis: 0, failed_allocations: 0, delayed: false, details: None, last_allocation_status: "no_attempt".into(), last_node: None });
        info.failed_allocations = 4;
        info.details = Some(tagged.clone());
    }
    fail(&mut fresh, "node restarted");
    fresh.converge();
    assert_eq!(fresh.health().status, HealthStatus::Green, "no _cluster/reroute?retry_failed needed");
}

#[test]
fn segment_advertisements_fail_closed_and_never_change_writer_selection() {
    use crate::compatibility::*;
    let legacy = node("old", &[]);
    assert_eq!(segment_formats(&legacy).unwrap(), (11, 11));
    for attrs in [
        vec![(SEGMENT_READ_MAX,"12")],
        vec![(CAPABILITIES_SCHEMA,"2"), (SEGMENT_READ_MAX,"12"), (SEGMENT_WRITE,"12")],
        vec![(CAPABILITIES_SCHEMA,"1"), (SEGMENT_READ_MAX,"eleven"), (SEGMENT_WRITE,"12")],
        vec![(CAPABILITIES_SCHEMA,"1"), (SEGMENT_READ_MAX,"11"), (SEGMENT_WRITE,"12")],
    ] { assert!(segment_formats(&node("bad", &attrs)).is_err()); }
    let writer = node("new", &[(CAPABILITIES_SCHEMA,"1"), (SEGMENT_READ_MAX,"12"), (SEGMENT_WRITE,"12")]);
    let mut w = World::new(&[legacy, writer]);
    assert!(validate_segment_members(&w.state).is_err());
    assert_eq!(segment_formats(&w.state.nodes["new"]).unwrap().1, 12);
    w.state.nodes.remove("old");
    assert!(validate_segment_members(&w.state).is_ok());
}

#[test]
fn force_allocation_cannot_override_segment_format_refusal() {
    use crate::compatibility::{CAPABILITIES_SCHEMA, SEGMENT_READ_MAX, SEGMENT_WRITE};
    let mut w = World::new(&[node("store", &[])]);
    w.create("idx", 1, 0, &[]);
    w.converge();
    w.kill("store");
    // The same in-sync store returns, but its process cannot read its selected
    // write format. Existing-store force allocation must preserve this NO.
    w.add_node(node("store", &[(CAPABILITIES_SCHEMA,"1"), (SEGMENT_READ_MAX,"11"), (SEGMENT_WRITE,"12")]));
    w.converge();
    assert!(!w.primary("idx",0).assigned());
    assert_eq!(w.health().status,HealthStatus::Red);
}

#[test]
fn measured_memory_only_breaks_equal_weight_unassigned_ties() {
    let mut w = World::new(&[node("n1", &[]), node("n2", &[])]);
    // n2 has more resident bytes but a lower fraction of its effective budget.
    w.info.placement_memory.insert("n1".into(), PlacementMemory { measured_bytes: 90, budget_bytes: 100 });
    w.info.placement_memory.insert("n2".into(), PlacementMemory { measured_bytes: 120, budget_bytes: 400 });
    w.create("idx", 2, 0, &[]);
    w.reroute();
    let shards = &w.state.routing_table.indices["idx"].shards;
    assert_eq!(shards[0].copies[0].node_id.as_deref(), Some("n2"));
    // The second shard goes to the lighter shard-count node despite memory.
    assert_eq!(shards[1].copies[0].node_id.as_deref(), Some("n1"));
    w.start_all();
    w.info.placement_memory.get_mut("n2").unwrap().measured_bytes = u64::MAX;
    w.reroute();
    assert!(w.state.routing_table.all_shards().all(|s| s.state == ShardState::Started));
    assert_eq!(w.state.routing_table.indices["idx"].shards[0].copies[0].node_id.as_deref(), Some("n2"));
}

#[test]
fn measured_memory_missing_observations_and_deciders_preserve_existing_behavior() {
    fn destination(samples: BTreeMap<String, PlacementMemory>, settings: &[(&str, &str)]) -> String {
        let mut w = World::new(&[node("n1", &[]), node("n2", &[])]);
        w.info.placement_memory = samples;
        w.create("idx", 1, 0, settings);
        w.reroute();
        w.state.routing_table.indices["idx"].shards[0].copies[0].node_id.clone().unwrap()
    }
    let baseline = destination(BTreeMap::new(), &[]);
    let partial = BTreeMap::from([("n2".into(), PlacementMemory { measured_bytes: 0, budget_bytes: 100 })]);
    assert_eq!(destination(partial, &[]), baseline);
    let all = BTreeMap::from([
        ("n1".into(), PlacementMemory { measured_bytes: u64::MAX, budget_bytes: 1 }),
        ("n2".into(), PlacementMemory { measured_bytes: 0, budget_bytes: u64::MAX }),
    ]);
    // Even extreme observations cannot override allocation filtering.
    assert_eq!(destination(all.clone(), &[("index.routing.allocation.require._name", "n1")]), "n1");
    let mut invalid = all;
    invalid.get_mut("n1").unwrap().budget_bytes = 0;
    assert_eq!(destination(invalid, &[]), baseline);
}

#[test]
fn pressure_moves_preserve_deciders_and_refuse_concurrent_relocations(){
    let mut w=World::new(&[node("n1",&[]),node("n2",&[]),node("n3",&[])]);
    w.create("idx",1,0,&[]);w.converge();
    let copy=w.state.routing_table.indices["idx"].shards[0].copies[0].clone();
    let target=if copy.node_id.as_deref()==Some("n1"){"n2"}else{"n1"};
    let stores=ShardStoreInfo::from_routing(&w.state);
    let moved=w.svc.pressure_relocation(&w.state,"idx",0,copy.allocation_id().unwrap(),target,w.now,&w.info,&stores).unwrap();
    assert!(moved.routing_table.all_shards().any(|copy|copy.state==ShardState::Initializing&&copy.allocation_id().unwrap().starts_with("capacity-")));
    assert!(w.svc.pressure_relocation(&moved,"idx",0,copy.allocation_id().unwrap(),"n3",w.now,&w.info,&stores).is_none());
    let disabled=AllocationService::new(serde_json::from_value(json!({"cluster.routing.allocation.enable":"none"})).unwrap());
    assert!(disabled.pressure_relocation(&w.state,"idx",0,copy.allocation_id().unwrap(),target,w.now,&w.info,&stores).is_none());
}

/// Review finding 7: after a pressure move off a hot node, the count
/// rebalancer must not move another shard back onto it while its pressure
/// is sustained (else the next pressure move follows, forever). Once the
/// pressure is gone the ordinary balance returns.
#[test]
fn rebalance_moves_nothing_onto_a_node_under_sustained_memory_pressure(){
    let mut w=World::new(&[node("n1",&[]),node("n2",&[])]);
    w.create("idx",4,0,&[]);w.converge();
    assert_eq!(w.per_node(),BTreeMap::from([("n1".to_string(),2),("n2".to_string(),2)]));
    let copy=w.state.routing_table.all_shards().find(|c|c.node_id.as_deref()==Some("n1")).unwrap().clone();
    let stores=ShardStoreInfo::from_routing(&w.state);
    w.state=w.svc.pressure_relocation(&w.state,&copy.index,copy.shard,copy.allocation_id().unwrap(),"n2",w.now,&w.info,&stores).unwrap();
    w.info.memory_pressure.insert("n1".into());
    w.converge();
    assert_eq!(w.per_node()["n1"],1,"nothing rebalanced back onto the hot node: {:?}",w.per_node());
    for _ in 0..3{w.reroute();assert!(w.state.routing_table.all_shards().all(|c|c.relocating_node_id.is_none()));}
    let ex=w.explain(json!({"index":"idx","shard":w.state.routing_table.all_shards().find(|c|c.node_id.as_deref()==Some("n2")&&!c.allocation_id().unwrap().starts_with("capacity-")).unwrap().shard,"primary":true})).unwrap();
    assert!(ex.to_string().contains("memory_pressure"),"{ex}");
    w.info.memory_pressure.clear();
    w.converge();
    assert_eq!(w.per_node()["n1"],2,"balance returns once the pressure is gone");
}

#[test]
fn pressure_ranking_skips_larger_decider_ineligible_copy(){
    let mut w=World::new(&[node("n1",&[])]);
    w.create("large",1,0,&[("index.routing.allocation.require._name","n1")]);
    w.create("small",1,0,&[]);w.converge();
    w.state.nodes.insert("n2".into(),node("n2",&[]));
    let stores=ShardStoreInfo::from_routing(&w.state);let mut best=None;
    for (index,bytes) in [("large",4096u64),("small",1024)]{
        let copy=&w.state.routing_table.indices[index].shards[0].copies[0];
        let next=w.svc.pressure_relocation(&w.state,index,0,copy.allocation_id().unwrap(),"n2",w.now,&w.info,&stores);
        if index=="large"{assert!(next.is_none(),"filter must veto the larger copy");}
        if let Some(next)=next{if best.as_ref().is_none_or(|(_,prior,_)|bytes>*prior){best=Some((index,bytes,next));}}
    }
    let(index,_,next)=best.expect("smaller eligible shard must remain selectable");assert_eq!(index,"small");
    assert!(next.routing_table.indices["small"].shards[0].copies.iter().any(|copy|copy.node_id.as_deref()==Some("n2")&&copy.state==ShardState::Initializing));
    assert!(next.routing_table.indices["large"].shards[0].copies.iter().all(|copy|copy.state==ShardState::Started&&copy.node_id.as_deref()==Some("n1")));
}
