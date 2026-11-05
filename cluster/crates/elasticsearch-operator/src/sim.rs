//! Closed-loop simulation tests: the planners + executor drive an in-memory
//! Elasticsearch (shard allocator, voting exclusions, node shutdown API)
//! and an in-memory StatefulSet controller, round after round, and safety
//! invariants are checked after every step.

use crate::es_client::*;
use crate::executor::{self, K8sAction};
use crate::rolling::{self, PodView};
use crate::scale::{self, NodeSetScale};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::Mutex;

// ---------------------------------------------------------------- fake ES

#[derive(Clone, Debug)]
struct Copy {
    index: String,
    shard: u32,
    primary: bool,
    node: Option<String>,
}

#[derive(Default)]
struct EsState {
    /// name -> (id, master, data, up)
    nodes: BTreeMap<String, (String, bool, bool, bool)>,
    copies: Vec<Copy>,
    excludes: Vec<String>,
    voting_exclusions: Vec<String>,
    shutdowns: Vec<ShutdownRecord>,
    max_voting_exclusions_seen: usize,
}

#[derive(Default)]
pub(crate) struct FakeEs {
    s: Mutex<EsState>,
}

impl EsState {
    fn up(&self, n: &str) -> bool {
        self.nodes.get(n).is_some_and(|x| x.3)
    }
    fn health(&self) -> Option<ClusterHealth> {
        if !self.nodes.values().any(|n| n.1 && n.3) {
            return None; // no master
        }
        let unassigned = self.copies.iter().filter(|c| !c.node.as_deref().is_some_and(|n| self.up(n))).count() as i64;
        let primaries_ok = self.copies.iter().filter(|c| c.primary).all(|c| c.node.as_deref().is_some_and(|n| self.up(n)));
        let status = if unassigned == 0 { Health::Green } else if primaries_ok { Health::Yellow } else { Health::Red };
        Some(ClusterHealth { status, number_of_nodes: self.nodes.values().filter(|n| n.3).count() as i64, relocating_shards: 0, initializing_shards: 0, unassigned_shards: unassigned })
    }
    fn restarting(&self, name: &str) -> bool {
        let id = self.nodes.get(name).map(|n| n.0.clone());
        self.shutdowns.iter().any(|r| Some(&r.node_id) == id.as_ref() && r.type_ == ShutdownType::Restart)
    }
    /// One allocator round.
    fn allocate(&mut self) {
        for r in &mut self.shutdowns {
            r.status = "COMPLETE".into();
        }
        let eligible: Vec<String> = self.nodes.iter().filter(|(n, x)| x.2 && x.3 && !self.excludes.contains(n)).map(|(n, _)| n.clone()).collect();
        let n = self.copies.len();
        for i in 0..n {
            let c = self.copies[i].clone();
            let on_down = c.node.as_deref().is_some_and(|nd| !self.up(nd));
            // node gone or down without a restart record -> unassign (promote a replica if primary)
            if c.node.as_deref().is_some_and(|nd| !self.nodes.contains_key(nd)) || (on_down && !self.restarting(c.node.as_deref().unwrap())) {
                self.copies[i].node = None;
                if c.primary {
                    if let Some(j) = (0..n).find(|&j| j != i && self.copies[j].index == c.index && self.copies[j].shard == c.shard && self.copies[j].node.as_deref().is_some_and(|nd| self.up(nd))) {
                        self.copies[j].primary = true;
                        self.copies[i].primary = false;
                    }
                }
            }
            let c = self.copies[i].clone();
            let needs_move = match c.node.as_deref() {
                None => true,
                Some(nd) => self.excludes.iter().any(|e| e == nd) && self.up(nd),
            };
            if needs_move {
                let taken: Vec<String> = self.copies.iter().filter(|o| o.index == c.index && o.shard == c.shard).filter_map(|o| o.node.clone()).collect();
                if let Some(target) = eligible.iter().find(|e| !taken.contains(e)) {
                    // primaries are never unassigned while a copy survives (relocation keeps data)
                    if c.node.is_some() || !c.primary || !self.copies.iter().any(|o| o.index == c.index && o.shard == c.shard && o.node.is_some()) {
                        self.copies[i].node = Some(target.clone());
                    }
                }
            }
        }
    }
}

impl FakeEs {
    fn with<R>(&self, f: impl FnOnce(&mut EsState) -> R) -> R {
        f(&mut self.s.lock().unwrap())
    }
}

fn unreachable() -> EsError {
    EsError::Status { status: 503, body: "master_not_discovered_exception".into() }
}

#[async_trait]
impl EsClient for FakeEs {
    async fn health(&self) -> Result<ClusterHealth, EsError> {
        self.with(|s| s.health()).ok_or_else(unreachable)
    }
    async fn cluster_uuid(&self) -> Result<Option<String>, EsError> {
        Ok(Some("uuid".into()))
    }
    async fn nodes(&self) -> Result<Vec<EsNode>, EsError> {
        Ok(self.with(|s| s.nodes.iter().filter(|n| n.1.3).map(|(n, x)| EsNode { id: x.0.clone(), name: n.clone(), roles: String::new(), is_master: false }).collect()))
    }
    async fn shards(&self) -> Result<Vec<ShardRow>, EsError> {
        Ok(self.with(|s| {
            s.copies
                .iter()
                .map(|c| ShardRow { index: c.index.clone(), shard: c.shard.to_string(), primary: c.primary, state: if c.node.is_some() { "STARTED" } else { "UNASSIGNED" }.into(), node: c.node.clone() })
                .collect()
        }))
    }
    async fn allocation_exclude_names(&self) -> Result<Vec<String>, EsError> {
        Ok(self.with(|s| s.excludes.clone()))
    }
    async fn set_allocation_exclude_names(&self, names: &[String]) -> Result<(), EsError> {
        self.with(|s| s.excludes = names.to_vec());
        Ok(())
    }
    async fn voting_exclusions(&self) -> Result<Vec<String>, EsError> {
        Ok(self.with(|s| s.voting_exclusions.clone()))
    }
    async fn add_voting_exclusions(&self, names: &[String]) -> Result<(), EsError> {
        self.with(|s| {
            s.voting_exclusions.extend(names.iter().cloned());
            s.max_voting_exclusions_seen = s.max_voting_exclusions_seen.max(s.voting_exclusions.len());
        });
        Ok(())
    }
    async fn clear_voting_exclusions(&self) -> Result<(), EsError> {
        self.with(|s| s.voting_exclusions.clear());
        Ok(())
    }
    async fn shutdowns(&self) -> Result<Vec<ShutdownRecord>, EsError> {
        Ok(self.with(|s| s.shutdowns.clone()))
    }
    async fn put_shutdown(&self, node_id: &str, type_: ShutdownType, reason: &str) -> Result<(), EsError> {
        self.with(|s| {
            s.shutdowns.retain(|r| r.node_id != node_id);
            s.shutdowns.push(ShutdownRecord { node_id: node_id.into(), type_, reason: reason.into(), status: "IN_PROGRESS".into() });
        });
        Ok(())
    }
    async fn delete_shutdown(&self, node_id: &str) -> Result<(), EsError> {
        self.with(|s| s.shutdowns.retain(|r| r.node_id != node_id));
        Ok(())
    }
}

// ---------------------------------------------------------------- fake k8s

struct Sts {
    replicas: i32,
    master: bool,
    data: bool,
    revision: u32,
}

#[derive(Clone)]
struct SimPod {
    sts: String,
    ordinal: i32,
    revision: u32,
    ready: bool,
}

struct Sim {
    es: FakeEs,
    sts: BTreeMap<String, Sts>,
    pods: BTreeMap<String, SimPod>,
    desired: BTreeMap<String, i32>,
    deleted_pods: Vec<String>,
}

impl Sim {
    fn new(sets: &[(&str, i32, bool, bool)], indices: u32, replicas: u32) -> Sim {
        let mut sim = Sim { es: FakeEs::default(), sts: BTreeMap::new(), pods: BTreeMap::new(), desired: BTreeMap::new(), deleted_pods: vec![] };
        for (name, n, m, d) in sets {
            sim.sts.insert(name.to_string(), Sts { replicas: *n, master: *m, data: *d, revision: 1 });
            sim.desired.insert(name.to_string(), *n);
        }
        // start everything up and ready
        for _ in 0..2 {
            sim.k8s_tick();
        }
        sim.es.with(|s| {
            for i in 0..indices {
                for r in 0..=replicas {
                    s.copies.push(Copy { index: format!("idx{i}"), shard: 0, primary: r == 0, node: None });
                }
            }
            for _ in 0..5 {
                s.allocate();
            }
        });
        assert_eq!(sim.es.with(|s| s.health()).unwrap().status, Health::Green);
        sim
    }

    fn k8s_tick(&mut self) {
        // pods created last tick become ready and join
        for (name, p) in self.pods.iter_mut() {
            if !p.ready {
                p.ready = true;
                let sts = &self.sts[&p.sts];
                let (m, d) = (sts.master, sts.data);
                self.es.with(|s| {
                    let id = s.nodes.get(name).map(|n| n.0.clone()).unwrap_or_else(|| format!("id-{name}"));
                    s.nodes.insert(name.clone(), (id, m, d, true));
                });
            }
        }
        // reconcile pod count
        for (sname, sts) in &self.sts {
            for o in 0..sts.replicas {
                let name = format!("{sname}-{o}");
                self.pods.entry(name).or_insert(SimPod { sts: sname.clone(), ordinal: o, revision: sts.revision, ready: false });
            }
            let extra: Vec<String> = self.pods.iter().filter(|(_, p)| &p.sts == sname && p.ordinal >= sts.replicas).map(|(n, _)| n.clone()).collect();
            for n in extra {
                self.pods.remove(&n);
                // pod and its PVC are gone: the node leaves for good
                self.es.with(|s| {
                    s.nodes.remove(&n);
                });
            }
        }
    }

    fn delete_pod(&mut self, pod: &str) {
        let p = self.pods.remove(pod).expect("pod exists");
        self.deleted_pods.push(pod.to_string());
        self.es.with(|s| {
            if let Some(n) = s.nodes.get_mut(pod) {
                n.3 = false;
            }
        });
        let rev = self.sts[&p.sts].revision;
        self.pods.insert(pod.to_string(), SimPod { revision: rev, ready: false, ..p });
    }

    fn all_ready(&self) -> bool {
        let expected: i32 = self.sts.values().map(|s| s.replicas).sum();
        self.pods.values().filter(|p| p.ready).count() as i32 == expected
    }

    /// One reconcile round + one tick of the world. Returns true when converged.
    async fn round(&mut self, max_unavailable: Option<i32>) -> bool {
        let obs = observe(&self.es).await.unwrap();
        let sets: Vec<NodeSetScale> = self
            .sts
            .iter()
            .map(|(n, s)| NodeSetScale { sts: n.clone(), current: s.replicas, desired: self.desired.get(n).copied().unwrap_or(0), master: s.master, data: s.data, in_spec: true })
            .collect();
        let plan = scale::plan(&sets, &obs, true, None);
        for a in executor::execute_scale(&self.es, &plan).await.unwrap() {
            if let K8sAction::SetReplicas { sts, replicas } = a {
                let s = self.sts.get_mut(&sts).unwrap();
                // invariant: a leaving master must be vote-excluded, a leaving data node empty
                for o in replicas..s.replicas {
                    let pod = format!("{sts}-{o}");
                    self.es.with(|e| {
                        if s.master {
                            assert!(e.voting_exclusions.contains(&pod), "{pod} removed without voting exclusion");
                        }
                        assert!(!e.copies.iter().any(|c| c.node.as_deref() == Some(&pod)), "{pod} removed while holding shards");
                    });
                }
                s.replicas = replicas;
            }
        }
        let topology_stable = plan.waiting.is_empty() && plan.actions.iter().all(|a| matches!(a, scale::ScaleAction::SetAllocationExcludes(_) | scale::ScaleAction::ClearVotingExclusions));
        let mut outdated = 0;
        if topology_stable {
            let views: Vec<PodView> = self
                .pods
                .iter()
                .map(|(n, p)| {
                    let s = &self.sts[&p.sts];
                    PodView { name: n.clone(), sts: p.sts.clone(), ordinal: p.ordinal, master: s.master, data: s.data, outdated: p.revision != s.revision, ready: p.ready, terminating: false }
                })
                .collect();
            let expected: i32 = self.sts.values().map(|s| s.replicas).sum();
            let rplan = rolling::plan(&views, expected as usize, &obs, max_unavailable);
            outdated = rplan.remaining;
            for a in executor::execute_rolling(&self.es, &rplan).await.unwrap() {
                if let K8sAction::DeletePod { pod } = a {
                    // invariant: stateful pods restart only from a green, fully ready cluster
                    assert_eq!(self.es.with(|s| s.health()).map(|h| h.status), Some(Health::Green), "deleting {pod} while not green");
                    assert!(self.all_ready(), "deleting {pod} while another pod is unavailable");
                    self.delete_pod(&pod);
                }
            }
        }
        self.k8s_tick();
        self.es.with(|s| s.allocate());
        // invariant: no shard ever loses its last copy
        self.es.with(|s| {
            let mut by_shard: BTreeMap<(String, u32), usize> = BTreeMap::new();
            for c in &s.copies {
                *by_shard.entry((c.index.clone(), c.shard)).or_default() += c.node.is_some() as usize;
            }
            assert!(by_shard.values().all(|&n| n > 0), "data loss: {by_shard:?}");
        });
        let es_clean = self.es.with(|s| s.excludes.is_empty() && s.voting_exclusions.is_empty() && s.shutdowns.is_empty());
        topology_stable && outdated == 0 && self.all_ready() && es_clean && self.sts.iter().all(|(n, s)| self.desired.get(n) == Some(&s.replicas))
    }

    async fn run_until_converged(&mut self, max_unavailable: Option<i32>, max_rounds: usize) -> usize {
        for r in 0..max_rounds {
            if self.round(max_unavailable).await {
                return r;
            }
        }
        panic!("did not converge in {max_rounds} rounds");
    }
}

#[tokio::test]
async fn sim_data_downscale_migrates_before_removal() {
    let mut sim = Sim::new(&[("c-es-m", 3, true, false), ("c-es-d", 4, false, true)], 6, 1);
    sim.desired.insert("c-es-d".into(), 2);
    sim.run_until_converged(Some(1), 40).await;
    assert_eq!(sim.sts["c-es-d"].replicas, 2);
    assert_eq!(sim.es.with(|s| s.health()).unwrap().status, Health::Green);
}

#[tokio::test]
async fn sim_master_downscale_one_voter_at_a_time() {
    let mut sim = Sim::new(&[("c-es-m", 5, true, true)], 4, 1);
    sim.desired.insert("c-es-m".into(), 3);
    sim.run_until_converged(Some(1), 40).await;
    assert_eq!(sim.sts["c-es-m"].replicas, 3);
    assert_eq!(sim.es.with(|s| s.max_voting_exclusions_seen), 1);
}

#[tokio::test]
async fn sim_rolling_restart_one_pod_at_a_time_green_between() {
    let mut sim = Sim::new(&[("c-es-m", 3, true, false), ("c-es-d", 3, false, true)], 5, 1);
    for s in sim.sts.values_mut() {
        s.revision = 2;
    }
    sim.run_until_converged(Some(1), 80).await;
    assert!(sim.pods.values().all(|p| p.revision == 2));
    assert_eq!(sim.deleted_pods.len(), 6);
    // data nodes first, masters last, highest ordinal first
    assert_eq!(sim.deleted_pods, vec!["c-es-d-2", "c-es-d-1", "c-es-d-0", "c-es-m-2", "c-es-m-1", "c-es-m-0"]);
}

#[tokio::test]
async fn sim_scale_up_then_down_round_trip() {
    let mut sim = Sim::new(&[("c-es-all", 3, true, true)], 6, 1);
    sim.desired.insert("c-es-all".into(), 5);
    sim.run_until_converged(Some(1), 20).await;
    assert_eq!(sim.pods.len(), 5);
    sim.desired.insert("c-es-all".into(), 3);
    sim.run_until_converged(Some(1), 60).await;
    assert_eq!(sim.pods.len(), 3);
    assert_eq!(sim.es.with(|s| s.max_voting_exclusions_seen), 1);
}
