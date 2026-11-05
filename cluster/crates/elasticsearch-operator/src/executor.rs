//! Executes the Elasticsearch side of planner output through [`EsClient`],
//! returning the Kubernetes side for the caller (the controller, or the
//! simulation in tests) to apply.

use crate::es_client::{EsClient, EsError, ShutdownType};
use crate::names::SHUTDOWN_REASON;
use crate::rolling::{RollingAction, RollingPlan};
use crate::scale::{ScaleAction, ScalePlan};

/// Kubernetes changes requested by the planners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum K8sAction {
    SetReplicas { sts: String, replicas: i32 },
    DeleteStatefulSet { sts: String },
    DeletePod { pod: String },
}

/// Run ES-side scale actions in order. If one fails, later ones (and the
/// replica changes that depend on them) are not returned: a StatefulSet is
/// never shrunk unless every ES action before it succeeded.
pub async fn execute_scale(es: &dyn EsClient, plan: &ScalePlan) -> Result<Vec<K8sAction>, EsError> {
    let mut out = Vec::new();
    for a in &plan.actions {
        match a {
            ScaleAction::SetAllocationExcludes(names) => es.set_allocation_exclude_names(names).await?,
            ScaleAction::ClearVotingExclusions => es.clear_voting_exclusions().await?,
            ScaleAction::AddVotingExclusion(n) => es.add_voting_exclusions(std::slice::from_ref(n)).await?,
            ScaleAction::SetReplicas { sts, replicas } => out.push(K8sAction::SetReplicas { sts: sts.clone(), replicas: *replicas }),
            ScaleAction::DeleteStatefulSet { sts } => out.push(K8sAction::DeleteStatefulSet { sts: sts.clone() }),
        }
    }
    Ok(out)
}

pub async fn execute_rolling(es: &dyn EsClient, plan: &RollingPlan) -> Result<Vec<K8sAction>, EsError> {
    let mut out = Vec::new();
    for a in &plan.actions {
        match a {
            RollingAction::ClearShutdown { node_id } => es.delete_shutdown(node_id).await?,
            RollingAction::PrepareShutdown { node_id, .. } => es.put_shutdown(node_id, ShutdownType::Restart, SHUTDOWN_REASON).await?,
            RollingAction::DeletePod { pod } => out.push(K8sAction::DeletePod { pod: pod.clone() }),
        }
    }
    Ok(out)
}
