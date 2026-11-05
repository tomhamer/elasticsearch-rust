//! Status computation (pure).

use crate::crd::{Condition, ElasticsearchStatus};
use crate::es_client::EsObservation;

pub const PHASE_READY: &str = "Ready";
pub const PHASE_APPLYING: &str = "ApplyingChanges";
pub const PHASE_MIGRATING: &str = "MigratingData";
pub const PHASE_INVALID: &str = "Invalid";

#[derive(Clone, Debug, Default)]
pub struct StatusInputs {
    pub generation: Option<i64>,
    pub version: String,
    pub invalid: Vec<String>,
    pub es: EsObservation,
    pub ready_pods: i32,
    pub desired_pods: i32,
    pub scale_pending: bool,
    pub migrating_data: bool,
    pub outdated_pods: usize,
    pub waiting: Vec<String>,
}

fn cond(prev: Option<&ElasticsearchStatus>, type_: &str, ok: bool, message: Option<String>, now: &str) -> Condition {
    let status = if ok { "True" } else { "False" }.to_string();
    let last = prev
        .and_then(|p| p.conditions.iter().find(|c| c.type_ == type_))
        .filter(|c| c.status == status)
        .and_then(|c| c.last_transition_time.clone())
        .unwrap_or_else(|| now.to_string());
    Condition { type_: type_.into(), status, message, last_transition_time: Some(last) }
}

pub fn compute(prev: Option<&ElasticsearchStatus>, i: &StatusInputs, now: &str) -> ElasticsearchStatus {
    let health = i.es.health.as_ref().map(|h| h.status.as_str().to_string()).unwrap_or_else(|| "unknown".into());
    let phase = if !i.invalid.is_empty() {
        PHASE_INVALID
    } else if i.migrating_data {
        PHASE_MIGRATING
    } else if i.scale_pending || i.outdated_pods > 0 || i.ready_pods < i.desired_pods || !i.es.reachable() {
        PHASE_APPLYING
    } else {
        PHASE_READY
    };
    let version = if i.outdated_pods == 0 && i.invalid.is_empty() { Some(i.version.clone()) } else { prev.and_then(|p| p.version.clone()) };
    let msg = |v: &[String]| if v.is_empty() { None } else { Some(v.join("; ")) };
    ElasticsearchStatus {
        health: Some(health),
        phase: Some(phase.into()),
        available_nodes: i.ready_pods,
        version,
        observed_generation: i.generation,
        conditions: vec![
            cond(prev, "ReconciliationComplete", phase == PHASE_READY, msg(if i.invalid.is_empty() { &i.waiting } else { &i.invalid }), now),
            cond(prev, "ElasticsearchIsReachable", i.es.reachable(), None, now),
            cond(prev, "RunningDesiredVersion", i.outdated_pods == 0, if i.outdated_pods > 0 { Some(format!("{} pod(s) pending restart", i.outdated_pods)) } else { None }, now),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::es_client::{ClusterHealth, Health};

    fn reachable() -> EsObservation {
        EsObservation { health: Some(ClusterHealth { status: Health::Green, number_of_nodes: 3, relocating_shards: 0, initializing_shards: 0, unassigned_shards: 0 }), ..Default::default() }
    }

    #[test]
    fn phases() {
        let base = StatusInputs { version: "9.1.0".into(), es: reachable(), ready_pods: 3, desired_pods: 3, generation: Some(4), ..Default::default() };
        let s = compute(None, &base, "t0");
        assert_eq!(s.phase.as_deref(), Some(PHASE_READY));
        assert_eq!(s.health.as_deref(), Some("green"));
        assert_eq!(s.version.as_deref(), Some("9.1.0"));
        assert_eq!(s.observed_generation, Some(4));
        let s2 = compute(Some(&s), &StatusInputs { migrating_data: true, ..base.clone() }, "t1");
        assert_eq!(s2.phase.as_deref(), Some(PHASE_MIGRATING));
        // transition time kept for unchanged conditions, updated for changed ones
        let reach = s2.conditions.iter().find(|c| c.type_ == "ElasticsearchIsReachable").unwrap();
        assert_eq!(reach.last_transition_time.as_deref(), Some("t0"));
        let rc = s2.conditions.iter().find(|c| c.type_ == "ReconciliationComplete").unwrap();
        assert_eq!(rc.last_transition_time.as_deref(), Some("t1"));
        let s3 = compute(Some(&s), &StatusInputs { outdated_pods: 1, version: "9.2.0".into(), ..base.clone() }, "t2");
        assert_eq!(s3.phase.as_deref(), Some(PHASE_APPLYING));
        assert_eq!(s3.version.as_deref(), Some("9.1.0"));
        let s4 = compute(None, &StatusInputs { invalid: vec!["bad".into()], ..base.clone() }, "t");
        assert_eq!(s4.phase.as_deref(), Some(PHASE_INVALID));
        let s5 = compute(None, &StatusInputs { es: EsObservation::default(), ..base }, "t");
        assert_eq!(s5.health.as_deref(), Some("unknown"));
        assert_eq!(s5.phase.as_deref(), Some(PHASE_APPLYING));
    }
}
