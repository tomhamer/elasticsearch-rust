//! Topology changes: upscales (bounded by `changeBudget.maxSurge`) and safe
//! downscales.
//!
//! The planner is a pure function of the observed state, so the reconciler
//! can be restarted at any point: every step is re-derived from what
//! Kubernetes and Elasticsearch report, never from in-memory progress.
//!
//! Downscale of a nodeSet from `current` to `desired` replicas (the
//! StatefulSet always removes the highest ordinals first):
//!
//! 1. **Data nodes**: the leaving pods are added to
//!    `cluster.routing.allocation.exclude._name` (other, user-provided names
//!    in that setting are preserved). A pod may be removed only once the
//!    setting *as observed* contains it and `_cat/shards` shows no shard
//!    copy on it.
//! 2. **Master-eligible nodes**: removed **one at a time** cluster-wide.
//!    The next leaving master gets a voting config exclusion
//!    (`POST /_cluster/voting_config_exclusions`); the StatefulSet is shrunk
//!    only once the observed exclusions contain it. When every excluded
//!    node has left the cluster the exclusions are cleared.
//! 3. The StatefulSet's `replicas` is lowered to the lowest ordinal that is
//!    safe to remove; the remaining leaving pods wait for the next round.
//!
//! A cluster that has never formed holds no data: it is resized directly.
//! A formed cluster that is unreachable is never downscaled.

use crate::es_client::EsObservation;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct NodeSetScale {
    /// StatefulSet name (pods are `<sts>-<ordinal>`).
    pub sts: String,
    /// Current `spec.replicas` of the StatefulSet (0 if it does not exist).
    pub current: i32,
    /// Target from the spec (0 for a nodeSet removed from the spec).
    pub desired: i32,
    pub master: bool,
    pub data: bool,
    /// False for a StatefulSet whose nodeSet was removed from the spec.
    pub in_spec: bool,
}

impl NodeSetScale {
    pub fn pod(&self, ordinal: i32) -> String {
        format!("{}-{ordinal}", self.sts)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScaleAction {
    /// Replace `cluster.routing.allocation.exclude._name` with this list.
    SetAllocationExcludes(Vec<String>),
    ClearVotingExclusions,
    AddVotingExclusion(String),
    SetReplicas { sts: String, replicas: i32 },
    DeleteStatefulSet { sts: String },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScalePlan {
    /// Ordered: Elasticsearch-side actions come before the replica changes
    /// they enable.
    pub actions: Vec<ScaleAction>,
    /// Why some change could not be made yet (empty = topology converged
    /// or converging with no blocker).
    pub waiting: Vec<String>,
    /// True while shards are being moved off leaving data nodes.
    pub migrating_data: bool,
}

impl ScalePlan {
    pub fn replicas_for(&self, sts: &str) -> Option<i32> {
        self.actions.iter().rev().find_map(|a| match a {
            ScaleAction::SetReplicas { sts: s, replicas } if s == sts => Some(*replicas),
            _ => None,
        })
    }
}

pub fn plan(sets: &[NodeSetScale], es: &EsObservation, bootstrapped: bool, max_surge: Option<i32>) -> ScalePlan {
    let mut p = ScalePlan::default();

    // ---- upscales, bounded by maxSurge over the total desired count
    let total_desired: i32 = sets.iter().map(|s| s.desired).sum();
    let total_current: i32 = sets.iter().map(|s| s.current).sum();
    let mut allowance = max_surge.map(|s| (total_desired + s - total_current).max(0));
    for s in sets.iter().filter(|s| s.desired > s.current) {
        let need = s.desired - s.current;
        let grant = allowance.map_or(need, |a| a.min(need));
        if grant > 0 {
            p.actions.push(ScaleAction::SetReplicas { sts: s.sts.clone(), replicas: s.current + grant });
        }
        if let Some(a) = allowance.as_mut() {
            *a -= grant;
        }
        if grant < need {
            p.waiting.push(format!("{}: upscale limited by changeBudget.maxSurge", s.sts));
        }
    }

    let downscaling: Vec<&NodeSetScale> = sets.iter().filter(|s| s.current > s.desired).collect();

    // ---- cluster never seen formed. Missing bookkeeping is not proof it
    // never formed (the annotation is written after formation): existing
    // pods are only resized directly when Elasticsearch answers and reports
    // no formed cluster (production review #297).
    if !bootstrapped {
        if !es.reachable() && downscaling.iter().any(|s| s.current > 0) {
            p.waiting.push("downscale postponed: whether the cluster formed is unknown while Elasticsearch is unreachable".into());
            finish_deletes(sets, &mut p);
            return p;
        }
        for s in &downscaling {
            p.actions.push(ScaleAction::SetReplicas { sts: s.sts.clone(), replicas: s.desired });
        }
        finish_deletes(sets, &mut p);
        return p;
    }

    if !es.reachable() {
        if !downscaling.is_empty() {
            p.waiting.push("downscale postponed: Elasticsearch is unreachable".into());
        }
        finish_deletes(sets, &mut p);
        return p;
    }

    let our_pod = |name: &str| sets.iter().any(|s| name.strip_prefix(&s.sts).and_then(|r| r.strip_prefix('-')).is_some_and(|n| n.parse::<i32>().is_ok()));

    // ---- 1. allocation excludes for leaving data pods
    let leaving_data: BTreeSet<String> = downscaling.iter().filter(|s| s.data).flat_map(|s| (s.desired..s.current).map(|o| s.pod(o))).collect();
    let mut wanted: Vec<String> = es.allocation_excludes.iter().filter(|n| !our_pod(n)).cloned().collect();
    wanted.extend(leaving_data.iter().cloned());
    let observed: BTreeSet<&String> = es.allocation_excludes.iter().collect();
    if wanted.iter().collect::<BTreeSet<_>>() != observed {
        p.actions.push(ScaleAction::SetAllocationExcludes(wanted));
    }

    let has_shards = |pod: &str| es.shards.iter().any(|r| r.node.as_deref() == Some(pod));
    let in_cluster = |pod: &str| es.nodes.iter().any(|n| n.name == pod);
    // A data pod is drained only with positive evidence: it is in the
    // cluster holding no copy, or it left and every shard has an assigned
    // copy (green). A red cluster may have its only copy on a departed
    // pod's volume (production review #287: an unavailable pod has no
    // `_cat/shards` rows, which proved nothing).
    let red = es.health.as_ref().is_some_and(|h| h.status == crate::es_client::Health::Red);
    let green = es.health.as_ref().is_some_and(|h| h.status == crate::es_client::Health::Green);
    let drained = |pod: &str| !red && !has_shards(pod) && (in_cluster(pod) || green);

    // ---- 2. voting exclusions bookkeeping
    let excluded: BTreeSet<&String> = es.voting_exclusions.iter().collect();
    let excluded_present: Vec<&String> = es.voting_exclusions.iter().filter(|n| in_cluster(n)).collect();
    let mut add_voting: Option<String> = None;
    if !es.voting_exclusions.is_empty() && excluded_present.is_empty() {
        p.actions.push(ScaleAction::ClearVotingExclusions);
    }
    // a master removal is in flight while an excluded master is still around
    let mut master_shrunk = false;

    // ---- 3. shrink what is safe
    for s in &downscaling {
        let mut r = s.current;
        while r > s.desired {
            let pod = s.pod(r - 1);
            if s.data && (!observed.contains(&pod) || !drained(&pod)) {
                p.migrating_data = true;
                p.waiting.push(format!("{pod}: waiting for shards to migrate away"));
                break;
            }
            if s.master {
                if master_shrunk {
                    p.waiting.push(format!("{pod}: another master-eligible node is being removed"));
                } else if excluded.contains(&pod) {
                    // The exclusion record alone is not the proof: the node
                    // must be out of the committed and accepted voting
                    // configurations (production review #289).
                    if es.voters.as_ref().is_some_and(|v| !v.contains(&pod)) {
                        r -= 1;
                        master_shrunk = true;
                    } else {
                        p.waiting.push(format!("{pod}: waiting for the voting configuration to drop it"));
                    }
                } else {
                    if excluded_present.is_empty() && add_voting.is_none() {
                        add_voting = Some(pod.clone());
                    }
                    p.waiting.push(format!("{pod}: waiting for voting config exclusion"));
                }
                break; // one master at a time
            }
            r -= 1;
        }
        if r < s.current {
            p.actions.push(ScaleAction::SetReplicas { sts: s.sts.clone(), replicas: r });
        }
    }
    if let Some(pod) = add_voting {
        p.actions.push(ScaleAction::AddVotingExclusion(pod));
    }
    finish_deletes(sets, &mut p);
    p
}

/// The roles a nodeSet's existing pods may still hold while its spec moves
/// to new roles: the desired roles, plus those its StatefulSet's pod
/// template labels and its pods' Elasticsearch roles show (production
/// review #286: a downscale combined with a role change was planned from
/// the desired roles alone, deleting data and voters without migration).
pub fn conservative_roles(desired_master: bool, desired_data: bool, labels: Option<&std::collections::BTreeMap<String, String>>, observed_roles: &[&str]) -> (bool, bool) {
    let flag = |k: &str| labels.is_some_and(|l| l.get(k).is_none_or(|v| v == "true"));
    let es_master = observed_roles.iter().any(|r| r.contains('m'));
    let es_data = observed_roles.iter().any(|r| r.chars().any(|c| matches!(c, 'd' | 'h' | 'w' | 'c' | 'f' | 's')));
    (
        desired_master || flag(crate::names::LABEL_NODE_MASTER) || es_master,
        desired_data || flag(crate::names::LABEL_NODE_DATA) || es_data,
    )
}

fn finish_deletes(sets: &[NodeSetScale], p: &mut ScalePlan) {
    for s in sets.iter().filter(|s| !s.in_spec) {
        let replicas = p.replicas_for(&s.sts).unwrap_or(s.current);
        if replicas == 0 && s.current == 0 {
            p.actions.push(ScaleAction::DeleteStatefulSet { sts: s.sts.clone() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::es_client::{ClusterHealth, EsNode, Health, ShardRow};

    fn set(sts: &str, current: i32, desired: i32, master: bool, data: bool) -> NodeSetScale {
        NodeSetScale { sts: sts.into(), current, desired, master, data, in_spec: true }
    }

    fn es(nodes: &[&str]) -> EsObservation {
        EsObservation {
            health: Some(ClusterHealth { status: Health::Green, number_of_nodes: nodes.len() as i64, relocating_shards: 0, initializing_shards: 0, unassigned_shards: 0 }),
            nodes: nodes.iter().map(|n| EsNode { id: format!("id-{n}"), name: n.to_string(), roles: "dm".into(), is_master: false }).collect(),
            voters: Some(BTreeSet::new()),
            ..Default::default()
        }
    }

    fn shard(node: &str) -> ShardRow {
        ShardRow { index: "i".into(), shard: "0".into(), primary: true, state: "STARTED".into(), node: Some(node.into()) }
    }

    #[test]
    fn upscale_unbounded_and_with_max_surge() {
        let sets = [set("c-es-a", 1, 3, true, true), set("c-es-b", 0, 2, false, true)];
        let p = plan(&sets, &es(&[]), true, None);
        assert_eq!(p.replicas_for("c-es-a"), Some(3));
        assert_eq!(p.replicas_for("c-es-b"), Some(2));
        let p = plan(&sets, &es(&[]), true, Some(0));
        assert_eq!(p.replicas_for("c-es-a"), Some(3));
        assert_eq!(p.replicas_for("c-es-b"), Some(2));
        // rename a->b with maxSurge 1: only one extra pod may exist
        let sets = [NodeSetScale { in_spec: false, ..set("c-es-a", 3, 0, true, true) }, set("c-es-b", 0, 3, true, true)];
        let p = plan(&sets, &es(&["c-es-a-0", "c-es-a-1", "c-es-a-2"]), true, Some(1));
        assert_eq!(p.replicas_for("c-es-b"), Some(1));
        assert!(p.waiting.iter().any(|w| w.contains("maxSurge")));
    }

    #[test]
    fn unbootstrapped_cluster_resizes_directly() {
        let sets = [set("c-es-d", 5, 2, true, true)];
        // Elasticsearch answers and no cluster formed: nothing to protect.
        let p = plan(&sets, &es(&[]), false, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-d".into(), replicas: 2 }]);
        // Never seen formed but unreachable: existing pods may hold a formed
        // cluster's data (the annotation is written after formation).
        let p = plan(&sets, &EsObservation::default(), false, None);
        assert!(p.actions.is_empty(), "{:?}", p.actions);
        assert!(p.waiting.iter().any(|w| w.contains("unknown")), "{:?}", p.waiting);
        // A new set without pods still resizes.
        let p = plan(&[set("c-es-new", 0, 0, true, true)], &EsObservation::default(), false, None);
        assert!(p.waiting.is_empty());
    }

    /// Production review #287: a red cluster, or a departed data pod
    /// whose shards may have no other copy, keeps its volume.
    #[test]
    fn unavailable_data_pod_with_unassigned_copies_is_kept() {
        let sets = [set("c-es-d", 3, 2, false, true)];
        let mut o = es(&["c-es-d-0", "c-es-d-1"]);
        o.allocation_excludes = vec!["c-es-d-2".into()];
        o.shards = vec![ShardRow { index: "i".into(), shard: "0".into(), primary: true, state: "UNASSIGNED".into(), node: None }];
        o.health.as_mut().unwrap().status = Health::Red;
        let p = plan(&sets, &o, true, None);
        assert!(!p.actions.iter().any(|a| matches!(a, ScaleAction::SetReplicas { .. })), "{:?}", p.actions);
        // Yellow and the pod gone: not proof either.
        o.health.as_mut().unwrap().status = Health::Yellow;
        let p = plan(&sets, &o, true, None);
        assert!(!p.actions.iter().any(|a| matches!(a, ScaleAction::SetReplicas { .. })), "{:?}", p.actions);
        // Green with the pod gone: every shard has an assigned copy.
        o.health.as_mut().unwrap().status = Health::Green;
        o.shards.clear();
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.replicas_for("c-es-d"), Some(2));
    }

    /// Production review #289: an exclusion record without the node out of
    /// the voting configuration never shrinks the set.
    #[test]
    fn master_removal_waits_for_the_voting_configuration() {
        let sets = [set("c-es-m", 3, 2, true, false)];
        let mut o = es(&["c-es-m-0", "c-es-m-1", "c-es-m-2"]);
        o.voting_exclusions = vec!["c-es-m-2".into()];
        o.voters = Some(["c-es-m-0", "c-es-m-1", "c-es-m-2"].iter().map(|s| s.to_string()).collect());
        let p = plan(&sets, &o, true, None);
        assert!(p.actions.is_empty(), "{:?}", p.actions);
        o.voters = None;
        assert!(plan(&sets, &o, true, None).actions.is_empty(), "unobserved configuration: no shrink");
        o.voters = Some(["c-es-m-0", "c-es-m-1"].iter().map(|s| s.to_string()).collect());
        assert_eq!(plan(&sets, &o, true, None).replicas_for("c-es-m"), Some(2));
    }

    /// Production review #286: roles still held by the existing pods count.
    #[test]
    fn existing_roles_are_kept_during_a_role_change() {
        let labels: std::collections::BTreeMap<String, String> = [(crate::names::LABEL_NODE_MASTER.to_string(), "true".to_string()), (crate::names::LABEL_NODE_DATA.to_string(), "true".to_string())].into();
        assert_eq!(conservative_roles(false, false, Some(&labels), &[]), (true, true));
        assert_eq!(conservative_roles(false, false, None, &["dim"]), (true, true));
        assert_eq!(conservative_roles(false, false, None, &["-"]), (false, false));
        // Planned with the kept roles, the old master/data pods drain first.
        let (master, data) = conservative_roles(false, false, Some(&labels), &[]);
        let sets = [NodeSetScale { master, data, ..set("c-es-old", 3, 1, false, false) }];
        let mut o = es(&["c-es-old-0", "c-es-old-1", "c-es-old-2"]);
        o.shards = vec![shard("c-es-old-2")];
        let p = plan(&sets, &o, true, None);
        assert!(!p.actions.iter().any(|a| matches!(a, ScaleAction::SetReplicas { .. })), "{:?}", p.actions);
    }

    #[test]
    fn unreachable_formed_cluster_never_downscales() {
        let sets = [set("c-es-d", 3, 2, false, true)];
        let p = plan(&sets, &EsObservation::default(), true, None);
        assert!(p.actions.is_empty());
        assert!(!p.waiting.is_empty());
    }

    #[test]
    fn data_downscale_excludes_then_waits_for_migration_then_shrinks() {
        let sets = [set("c-es-d", 3, 1, false, true)];
        // round 1: set excludes; nothing shrinks because the setting isn't observed yet
        let mut o = es(&["c-es-d-0", "c-es-d-1", "c-es-d-2"]);
        o.allocation_excludes = vec!["user-node".into()];
        o.shards = vec![shard("c-es-d-1"), shard("c-es-d-2")];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetAllocationExcludes(vec!["user-node".into(), "c-es-d-1".into(), "c-es-d-2".into()])]);
        assert!(p.migrating_data);
        // round 2: excluded; d-2 drained, d-1 still has a shard -> shrink to 2 only
        o.allocation_excludes = vec!["user-node".into(), "c-es-d-1".into(), "c-es-d-2".into()];
        o.shards = vec![shard("c-es-d-1"), shard("c-es-d-0")];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-d".into(), replicas: 2 }]);
        // round 3: everything drained -> shrink to 1
        let sets = [set("c-es-d", 2, 1, false, true)];
        o.shards = vec![shard("c-es-d-0")];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetAllocationExcludes(vec!["user-node".into(), "c-es-d-1".into()]), ScaleAction::SetReplicas { sts: "c-es-d".into(), replicas: 1 }]);
        // round 4: converged -> our names are removed from the setting, user's kept
        let sets = [set("c-es-d", 1, 1, false, true)];
        o.allocation_excludes = vec!["user-node".into(), "c-es-d-1".into()];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetAllocationExcludes(vec!["user-node".into()])]);
        assert!(p.waiting.is_empty() && !p.migrating_data);
    }

    #[test]
    fn non_data_non_master_shrinks_immediately() {
        let sets = [set("c-es-coord", 3, 1, false, false)];
        let p = plan(&sets, &es(&["c-es-coord-0"]), true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-coord".into(), replicas: 1 }]);
    }

    #[test]
    fn master_downscale_one_at_a_time_with_voting_exclusions() {
        let sets = [set("c-es-m", 5, 3, true, false)];
        let mut o = es(&["c-es-m-0", "c-es-m-1", "c-es-m-2", "c-es-m-3", "c-es-m-4"]);
        // round 1: exclude m-4 only
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::AddVotingExclusion("c-es-m-4".into())]);
        // round 2: exclusion observed -> shrink by one
        o.voting_exclusions = vec!["c-es-m-4".into()];
        o.voters = Some(["c-es-m-0", "c-es-m-1", "c-es-m-2"].iter().map(|s| s.to_string()).collect());
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-m".into(), replicas: 4 }]);
        // round 3: m-4 still in the cluster (terminating): no new exclusion yet
        let sets = [set("c-es-m", 4, 3, true, false)];
        let p = plan(&sets, &o, true, None);
        assert!(p.actions.is_empty(), "{:?}", p.actions);
        // round 4: m-4 left -> clear, then exclude m-3
        o.nodes.retain(|n| n.name != "c-es-m-4");
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::ClearVotingExclusions, ScaleAction::AddVotingExclusion("c-es-m-3".into())]);
        // round 5: m-3 excluded -> shrink to 3
        o.voting_exclusions = vec!["c-es-m-3".into()];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-m".into(), replicas: 3 }]);
        // round 6: m-3 left -> clear; converged
        o.nodes.retain(|n| n.name != "c-es-m-3");
        let sets = [set("c-es-m", 3, 3, true, false)];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::ClearVotingExclusions]);
    }

    #[test]
    fn two_master_sets_never_lose_two_voters_at_once() {
        let sets = [set("c-es-m1", 2, 1, true, false), set("c-es-m2", 2, 1, true, false)];
        let o = es(&["c-es-m1-0", "c-es-m1-1", "c-es-m2-0", "c-es-m2-1"]);
        let p = plan(&sets, &o, true, None);
        let adds: Vec<_> = p.actions.iter().filter(|a| matches!(a, ScaleAction::AddVotingExclusion(_))).collect();
        assert_eq!(adds.len(), 1);
    }

    #[test]
    fn master_data_node_drains_before_voting_exclusion() {
        let sets = [set("c-es-all", 3, 2, true, true)];
        let mut o = es(&["c-es-all-0", "c-es-all-1", "c-es-all-2"]);
        o.shards = vec![shard("c-es-all-2")];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetAllocationExcludes(vec!["c-es-all-2".into()])]);
        o.allocation_excludes = vec!["c-es-all-2".into()];
        o.shards.clear();
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::AddVotingExclusion("c-es-all-2".into())]);
        o.voting_exclusions = vec!["c-es-all-2".into()];
        let p = plan(&sets, &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-all".into(), replicas: 2 }]);
    }

    #[test]
    fn removed_node_set_is_drained_then_deleted() {
        let gone = NodeSetScale { in_spec: false, ..set("c-es-old", 1, 0, false, true) };
        let mut o = es(&["c-es-old-0"]);
        o.allocation_excludes = vec!["c-es-old-0".into()];
        let p = plan(&[gone.clone()], &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetReplicas { sts: "c-es-old".into(), replicas: 0 }]);
        let gone0 = NodeSetScale { current: 0, ..gone };
        o.nodes.clear();
        let p = plan(&[gone0], &o, true, None);
        assert_eq!(p.actions, vec![ScaleAction::SetAllocationExcludes(vec![]), ScaleAction::DeleteStatefulSet { sts: "c-es-old".into() }]);
    }
}
