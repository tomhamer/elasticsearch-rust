//! Rolling-restart planner.
//!
//! StatefulSets use `updateStrategy: OnDelete`; the operator decides which
//! outdated pod (its `controller-revision-hash` differs from the
//! StatefulSet's `updateRevision`) to delete next:
//!
//! * a restart starts only when the cluster is **green** and every expected
//!   pod is ready, so data and master-eligible pods restart **one at a time
//!   with green in between**;
//! * pods holding no shards and no vote (coordinating-only) may restart in
//!   parallel up to `changeBudget.maxUnavailable`;
//! * non-master pods go first, master-eligible pods last (highest ordinal
//!   first within a StatefulSet, like the StatefulSet controller);
//! * before deleting a pod that is part of the cluster the operator
//!   registers `PUT /_nodes/{id}/shutdown {"type":"restart"}` and waits for
//!   the record's status to be `COMPLETE`; after the pod is back (updated
//!   and ready) the record is deleted;
//! * if Elasticsearch is unreachable and *no* pod is ready (e.g. a broken
//!   configuration), all outdated pods are deleted at once so the fix can
//!   roll out — the same escape hatch ECK has.

use crate::es_client::{EsObservation, Health, ShutdownType};
use crate::names::SHUTDOWN_REASON;

#[derive(Clone, Debug, PartialEq)]
pub struct PodView {
    pub name: String,
    pub sts: String,
    pub ordinal: i32,
    pub master: bool,
    pub data: bool,
    /// Running an older StatefulSet revision than `updateRevision`.
    pub outdated: bool,
    pub ready: bool,
    /// `metadata.deletionTimestamp` set.
    pub terminating: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RollingAction {
    PrepareShutdown { pod: String, node_id: String },
    DeletePod { pod: String },
    ClearShutdown { node_id: String },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RollingPlan {
    pub actions: Vec<RollingAction>,
    pub waiting: Vec<String>,
    /// Outdated pods left (including those being restarted now).
    pub remaining: usize,
}

/// `expected` = total number of pods the StatefulSets currently want
/// (sum of `spec.replicas`). `max_unavailable` = `None` for unbounded.
pub fn plan(pods: &[PodView], expected: usize, es: &EsObservation, max_unavailable: Option<i32>) -> RollingPlan {
    let mut p = RollingPlan::default();
    let ours = |r: &&crate::es_client::ShutdownRecord| r.type_ == ShutdownType::Restart && r.reason == SHUTDOWN_REASON;
    let pod_by_node_id = |id: &str| es.nodes.iter().find(|n| n.id == id).and_then(|n| pods.iter().find(|pd| pd.name == n.name));
    let ready_count = pods.iter().filter(|pd| pd.ready && !pd.terminating).count();
    let all_ready = ready_count >= expected;

    // 1. clean up records of completed restarts
    for r in es.shutdowns.iter().filter(ours) {
        let done = match pod_by_node_id(&r.node_id) {
            Some(pd) => !pd.outdated && pd.ready && !pd.terminating,
            // node no longer part of the cluster and nothing is restarting
            None => all_ready && !es.nodes.iter().any(|n| n.id == r.node_id),
        };
        if done {
            p.actions.push(RollingAction::ClearShutdown { node_id: r.node_id.clone() });
        }
    }

    let outdated: Vec<&PodView> = pods.iter().filter(|pd| pd.outdated).collect();
    p.remaining = outdated.len();
    if outdated.is_empty() {
        return p;
    }

    // 2. unreachable cluster
    if !es.reachable() {
        if ready_count == 0 {
            for pd in outdated.iter().filter(|pd| !pd.terminating) {
                p.actions.push(RollingAction::DeletePod { pod: pd.name.clone() });
            }
        } else {
            p.waiting.push("rolling restart postponed: Elasticsearch is unreachable".into());
        }
        return p;
    }

    // 3. continue restarts already prepared
    let mut in_progress = 0usize;
    let mut prepared_ready = 0usize;
    for pd in outdated.iter().filter(|pd| !pd.terminating) {
        let Some(id) = es.node_id(&pd.name) else { continue };
        if let Some(rec) = es.shutdowns.iter().filter(ours).find(|r| r.node_id == id) {
            in_progress += 1;
            if pd.ready {
                prepared_ready += 1;
            }
            if rec.status == "COMPLETE" {
                p.actions.push(RollingAction::DeletePod { pod: pd.name.clone() });
            } else {
                p.waiting.push(format!("{}: waiting for node shutdown preparation ({})", pd.name, rec.status));
            }
        }
    }
    let terminating = outdated.iter().filter(|pd| pd.terminating).count();
    in_progress += terminating;

    // 4. start new restarts
    let unavailable = expected.saturating_sub(ready_count);
    let mut budget: i64 = match max_unavailable {
        None => i64::MAX,
        Some(m) => m as i64 - unavailable as i64 - prepared_ready as i64,
    };
    if budget <= 0 {
        p.waiting.push("rolling restart: changeBudget.maxUnavailable reached".into());
        return p;
    }
    let green = es.health.as_ref().is_some_and(|h| h.status == Health::Green);
    let quiet = unavailable == 0 && in_progress == 0;

    let mut candidates: Vec<&PodView> = outdated
        .iter()
        .copied()
        .filter(|pd| !pd.terminating && es.node_id(&pd.name).is_none_or(|id| !es.shutdowns.iter().filter(ours).any(|r| r.node_id == id)))
        .collect();
    candidates.sort_by(|a, b| a.master.cmp(&b.master).then(a.sts.cmp(&b.sts)).then(b.ordinal.cmp(&a.ordinal)));

    let mut stateful_started = false;
    for pd in candidates {
        if budget <= 0 {
            break;
        }
        let stateful = pd.master || pd.data;
        if stateful {
            if stateful_started || !quiet || !green {
                if !green {
                    p.waiting.push(format!("{}: waiting for green health before restart", pd.name));
                } else if !quiet {
                    p.waiting.push(format!("{}: waiting for the previous restart to complete", pd.name));
                }
                continue;
            }
            stateful_started = true;
        }
        match es.node_id(&pd.name) {
            Some(id) => p.actions.push(RollingAction::PrepareShutdown { pod: pd.name.clone(), node_id: id.to_string() }),
            None => p.actions.push(RollingAction::DeletePod { pod: pd.name.clone() }),
        }
        budget -= 1;
    }
    p.waiting.dedup();
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::es_client::{ClusterHealth, EsNode, ShutdownRecord};

    fn pod(sts: &str, ord: i32, master: bool, data: bool, outdated: bool) -> PodView {
        PodView { name: format!("{sts}-{ord}"), sts: sts.into(), ordinal: ord, master, data, outdated, ready: true, terminating: false }
    }

    fn obs(pods: &[PodView], status: Health) -> EsObservation {
        EsObservation {
            health: Some(ClusterHealth { status, number_of_nodes: pods.len() as i64, relocating_shards: 0, initializing_shards: 0, unassigned_shards: 0 }),
            nodes: pods.iter().map(|p| EsNode { id: format!("id-{}", p.name), name: p.name.clone(), roles: String::new(), is_master: false }).collect(),
            ..Default::default()
        }
    }

    fn rec(pod: &str, status: &str) -> ShutdownRecord {
        ShutdownRecord { node_id: format!("id-{pod}"), type_: ShutdownType::Restart, reason: SHUTDOWN_REASON.into(), status: status.into() }
    }

    #[test]
    fn nothing_outdated_nothing_to_do() {
        let pods = vec![pod("c-es-d", 0, true, true, false)];
        let p = plan(&pods, 1, &obs(&pods, Health::Green), Some(1));
        assert!(p.actions.is_empty() && p.remaining == 0);
    }

    #[test]
    fn data_nodes_first_one_at_a_time_highest_ordinal_masters_last() {
        let pods = vec![pod("c-es-m", 0, true, false, true), pod("c-es-d", 0, false, true, true), pod("c-es-d", 1, false, true, true)];
        let p = plan(&pods, 3, &obs(&pods, Health::Green), Some(5));
        assert_eq!(p.actions, vec![RollingAction::PrepareShutdown { pod: "c-es-d-1".into(), node_id: "id-c-es-d-1".into() }]);
        assert_eq!(p.remaining, 3);
    }

    #[test]
    fn prepared_shutdown_completes_then_pod_deleted() {
        let pods = vec![pod("c-es-d", 0, false, true, true), pod("c-es-d", 1, false, true, true)];
        let mut o = obs(&pods, Health::Green);
        o.shutdowns = vec![rec("c-es-d-1", "IN_PROGRESS")];
        let p = plan(&pods, 2, &o, Some(1));
        assert!(p.actions.is_empty(), "{:?}", p.actions);
        o.shutdowns = vec![rec("c-es-d-1", "COMPLETE")];
        let p = plan(&pods, 2, &o, Some(1));
        assert_eq!(p.actions, vec![RollingAction::DeletePod { pod: "c-es-d-1".into() }]);
    }

    #[test]
    fn waits_for_green_and_for_restarted_pod_then_clears_record() {
        // d-1 restarted (updated) but not ready yet, cluster yellow
        let mut pods = vec![pod("c-es-d", 0, false, true, true), pod("c-es-d", 1, false, true, false)];
        pods[1].ready = false;
        let mut o = obs(&pods, Health::Yellow);
        o.shutdowns = vec![rec("c-es-d-1", "COMPLETE")];
        let p = plan(&pods, 2, &o, Some(1));
        assert!(p.actions.is_empty());
        // ready again, still yellow: clear the record, but do not start d-0
        pods[1].ready = true;
        let p = plan(&pods, 2, &o, Some(1));
        assert_eq!(p.actions, vec![RollingAction::ClearShutdown { node_id: "id-c-es-d-1".into() }]);
        // green: next pod
        let mut o2 = obs(&pods, Health::Green);
        o2.shutdowns.clear();
        let p = plan(&pods, 2, &o2, Some(1));
        assert_eq!(p.actions, vec![RollingAction::PrepareShutdown { pod: "c-es-d-0".into(), node_id: "id-c-es-d-0".into() }]);
    }

    #[test]
    fn coordinating_nodes_restart_in_parallel_within_budget() {
        let pods = vec![pod("c-es-c", 0, false, false, true), pod("c-es-c", 1, false, false, true), pod("c-es-c", 2, false, false, true)];
        let p = plan(&pods, 3, &obs(&pods, Health::Green), Some(2));
        assert_eq!(p.actions.len(), 2);
        let p = plan(&pods, 3, &obs(&pods, Health::Green), None);
        assert_eq!(p.actions.len(), 3);
        let p = plan(&pods, 3, &obs(&pods, Health::Green), Some(0));
        assert!(p.actions.is_empty());
    }

    #[test]
    fn unavailable_pod_consumes_budget() {
        let mut pods = vec![pod("c-es-d", 0, false, true, true), pod("c-es-d", 1, false, true, false)];
        pods[1].ready = false;
        let p = plan(&pods, 2, &obs(&pods, Health::Green), Some(1));
        assert!(p.actions.is_empty());
        // a missing pod (expected 3, only 2 exist) too
        let pods = vec![pod("c-es-d", 0, false, true, true), pod("c-es-d", 1, false, true, false)];
        let p = plan(&pods, 3, &obs(&pods, Health::Green), Some(1));
        assert!(p.actions.is_empty());
    }

    #[test]
    fn unreachable_cluster_with_no_ready_pod_deletes_all_outdated() {
        let mut pods = vec![pod("c-es-d", 0, true, true, true), pod("c-es-d", 1, true, true, true)];
        for p in &mut pods {
            p.ready = false;
        }
        let p = plan(&pods, 2, &EsObservation::default(), Some(1));
        assert_eq!(p.actions.len(), 2);
        pods[0].ready = true;
        let p = plan(&pods, 2, &EsObservation::default(), Some(1));
        assert!(p.actions.is_empty());
    }

    #[test]
    fn foreign_shutdown_records_are_left_alone() {
        let pods = vec![pod("c-es-d", 0, false, true, false)];
        let mut o = obs(&pods, Health::Green);
        o.shutdowns = vec![ShutdownRecord { reason: "maintenance by a human".into(), ..rec("c-es-d-0", "COMPLETE") }];
        let p = plan(&pods, 1, &o, Some(1));
        assert!(p.actions.is_empty());
    }

    #[test]
    fn pod_not_in_cluster_is_deleted_without_shutdown_record() {
        let pods = vec![pod("c-es-d", 0, false, true, true)];
        let mut o = obs(&pods, Health::Green);
        o.nodes.clear();
        let p = plan(&pods, 1, &o, Some(1));
        assert_eq!(p.actions, vec![RollingAction::DeletePod { pod: "c-es-d-0".into() }]);
    }
}
