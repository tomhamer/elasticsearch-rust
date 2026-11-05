//! Unit tests: the safety core (ported from ES's `CoordinationStateTests`),
//! wire format, settings and single-coordinator behaviour.

use super::simulation::mk_node;
use super::*;
use crate::state::{ClusterState, DiscoveryNode, VotingConfiguration};

fn cfg(ids: &[&str]) -> VotingConfiguration {
    VotingConfiguration::new(ids.iter().map(|s| s.to_string()))
}

fn node(id: &str) -> DiscoveryNode {
    let mut n = mk_node("", 0, true, 0);
    n.id = id.to_string();
    n.name = id.to_string();
    n
}

fn state_with(term: u64, version: u64, committed: &[&str], accepted: &[&str]) -> ClusterState {
    let mut s = PersistedState::empty("c").last_accepted_state;
    s.coordination.term = term;
    s.version = version;
    s.state_uuid = format!("uuid-{term}-{version}");
    s.coordination.last_committed_config = cfg(committed);
    s.coordination.last_accepted_config = cfg(accepted);
    s
}

fn cs(id: &str) -> CoordinationState {
    CoordinationState::new(id, PersistedState::empty("c"))
}

fn sj(source: &str, term: u64) -> StartJoin {
    StartJoin { source_node: node(source), term }
}

#[test]
fn set_initial_state_only_once() {
    let mut a = cs("a");
    assert!(a.set_initial_state(state_with(0, 0, &[], &[])).is_err());
    a.set_initial_state(state_with(0, 0, &["a"], &["a"])).unwrap();
    assert!(a.take_persist().unwrap().1.is_some());
    assert!(a.set_initial_state(state_with(0, 0, &["a"], &["a"])).is_err());
}

#[test]
fn start_join_requires_higher_term_and_persists() {
    let mut a = cs("a");
    assert!(a.handle_start_join(&sj("b", 0)).is_err());
    let j = a.handle_start_join(&sj("b", 2)).unwrap();
    assert_eq!((j.source_node.as_str(), j.target_node.as_str(), j.term), ("a", "b", 2));
    assert_eq!(a.take_persist(), Some((2, None)));
    assert!(a.handle_start_join(&sj("b", 2)).is_err(), "votes at most once per term");
    assert!(a.take_persist().is_none());
}

#[test]
fn join_rules() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a", "b", "c"], &["a", "b", "c"])).unwrap();
    let own = a.handle_start_join(&sj("a", 1)).unwrap();
    // Wrong term.
    let mut stale = own.clone();
    stale.term = 0;
    assert!(a.handle_join(&stale).is_err());
    // Voter with a fresher state is refused.
    let fresher = Join { source_node: "b".into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 5 };
    assert!(a.handle_join(&fresher).is_err());
    // Wrong target.
    let wrong = Join { source_node: "b".into(), target_node: "c".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 };
    assert!(a.handle_join(&wrong).is_err());
    assert!(a.handle_join(&own).unwrap());
    assert!(!a.election_won());
    assert!(!a.handle_join(&own).unwrap(), "duplicate vote not new");
    let b = Join { source_node: "b".into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 };
    assert!(a.handle_join(&b).unwrap());
    assert!(a.election_won());
}

#[test]
fn join_ignored_after_reboot_until_term_bumped() {
    let mut p = PersistedState::empty("c");
    p.current_term = 3;
    p.last_accepted_state = state_with(2, 4, &["a"], &["a"]);
    let mut a = CoordinationState::new("a", p);
    let j = Join { source_node: "a".into(), target_node: "a".into(), term: 3, last_accepted_term: 2, last_accepted_version: 4 };
    assert!(a.handle_join(&j).unwrap_err().0.contains("after reboot"));
}

#[test]
fn join_rejected_before_bootstrap() {
    let mut a = cs("a");
    let j = a.handle_start_join(&sj("a", 1)).unwrap();
    assert!(a.handle_join(&j).unwrap_err().0.contains("initial configuration"));
}

#[test]
fn election_needs_quorum_in_both_configs() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a", "b", "c"], &["a", "d", "e"])).unwrap();
    a.handle_start_join(&sj("a", 1)).unwrap();
    for v in ["a", "b"] {
        a.handle_join(&Join { source_node: v.into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 }).unwrap();
    }
    assert!(!a.election_won(), "quorum of committed config only");
    a.handle_join(&Join { source_node: "d".into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 }).unwrap();
    assert!(a.election_won());
}

fn elected_three() -> CoordinationState {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a", "b", "c"], &["a", "b", "c"])).unwrap();
    a.handle_start_join(&sj("a", 1)).unwrap();
    for v in ["a", "b"] {
        a.handle_join(&Join { source_node: v.into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 }).unwrap();
    }
    assert!(a.election_won());
    a
}

#[test]
fn client_value_rules() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a"], &["a"])).unwrap();
    assert!(a.handle_client_value(&state_with(1, 1, &["a"], &["a"])).unwrap_err().0.contains("election not won"));

    let mut a = elected_three();
    assert!(a.handle_client_value(&state_with(2, 1, &["a", "b", "c"], &["a", "b", "c"])).is_err(), "wrong term");
    assert!(a.handle_client_value(&state_with(1, 0, &["a", "b", "c"], &["a", "b", "c"])).is_err(), "not newer");
    assert!(a.handle_client_value(&state_with(1, 1, &["a"], &["a", "b", "c"])).is_err(), "committed config must not change");
    // Reconfiguration needs join votes to be a quorum of the new config.
    assert!(a.handle_client_value(&state_with(1, 1, &["a", "b", "c"], &["c", "d", "e"])).unwrap_err().0.contains("joinVotes"));
    a.handle_client_value(&state_with(1, 1, &["a", "b", "c"], &["a", "b", "d"])).unwrap();
    // Must accept own publication before publishing the next.
    assert!(a.handle_client_value(&state_with(1, 2, &["a", "b", "c"], &["a", "b", "d"])).unwrap_err().0.contains("before accepting"));
    a.handle_publish_request(state_with(1, 1, &["a", "b", "c"], &["a", "b", "d"])).unwrap();
    // Not yet committed: no further reconfiguration.
    assert!(a.handle_client_value(&state_with(1, 2, &["a", "b", "c"], &["a", "b", "c"])).unwrap_err().0.contains("already reconfiguring"));
}

#[test]
fn publish_request_rules() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a"], &["a"])).unwrap();
    a.take_persist();
    assert!(a.handle_publish_request(state_with(1, 1, &["a"], &["a"])).is_err(), "term mismatch");
    a.handle_start_join(&sj("b", 1)).unwrap();
    let r = a.handle_publish_request(state_with(1, 3, &["a"], &["a"])).unwrap();
    assert_eq!(r, PublishResponse { term: 1, version: 3 });
    assert!(a.take_persist().unwrap().1.is_some());
    assert!(a.handle_publish_request(state_with(1, 3, &["a"], &["a"])).is_err(), "same version");
    assert!(a.handle_publish_request(state_with(1, 2, &["a"], &["a"])).is_err(), "older version");
    a.handle_start_join(&sj("b", 2)).unwrap();
    // A new term may publish a lower version than an uncommitted one.
    a.handle_publish_request(state_with(2, 2, &["a"], &["a"])).unwrap();
}

#[test]
fn publish_response_and_commit() {
    let mut a = elected_three();
    let s = state_with(1, 1, &["a", "b", "c"], &["a", "b", "c"]);
    a.handle_client_value(&s).unwrap();
    let r = a.handle_publish_request(s).unwrap();
    assert!(a.handle_publish_response("a", PublishResponse { term: 1, version: 2 }).is_err(), "wrong version");
    assert_eq!(a.handle_publish_response("a", r).unwrap(), None);
    assert_eq!(a.handle_publish_response("a", r).unwrap(), None, "same voter twice");
    let commit = a.handle_publish_response("c", r).unwrap().expect("quorum");
    assert_eq!((commit.term, commit.version), (1, 1));
    a.take_persist();
    a.handle_commit(&commit).unwrap();
    let (_, st) = a.take_persist().unwrap();
    let st = st.unwrap();
    assert!(st.metadata.cluster_uuid_committed);
    assert!(a.handle_commit(&ApplyCommit { source_node: "a".into(), term: 1, version: 2 }).is_err());
    // Idempotent re-commit does not persist again.
    a.handle_commit(&commit).unwrap();
    assert!(a.take_persist().is_none());
}

#[test]
fn commit_requires_accepted_state_in_current_term() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a"], &["a"])).unwrap();
    a.handle_start_join(&sj("b", 1)).unwrap();
    a.handle_publish_request(state_with(1, 1, &["a"], &["a"])).unwrap();
    a.handle_start_join(&sj("b", 2)).unwrap();
    // Commit from the old term: refused (current term moved on).
    assert!(a.handle_commit(&ApplyCommit { source_node: "b".into(), term: 1, version: 1 }).is_err());
    // Commit in the new term for a state accepted in the old one: refused.
    assert!(a.handle_commit(&ApplyCommit { source_node: "b".into(), term: 2, version: 1 }).is_err());
}

#[test]
fn reconfiguration_publish_quorum_uses_both_configs() {
    let mut a = cs("a");
    a.set_initial_state(state_with(0, 0, &["a", "b", "c"], &["a", "b", "c"])).unwrap();
    a.handle_start_join(&sj("a", 1)).unwrap();
    for v in ["a", "b", "d"] {
        let _ = a.handle_join(&Join { source_node: v.into(), target_node: "a".into(), term: 1, last_accepted_term: 0, last_accepted_version: 0 });
    }
    let s = state_with(1, 1, &["a", "b", "c"], &["a", "d", "e"]);
    a.handle_client_value(&s).unwrap();
    let r = a.handle_publish_request(s).unwrap();
    assert_eq!(a.handle_publish_response("a", r).unwrap(), None);
    assert_eq!(a.handle_publish_response("b", r).unwrap(), None, "quorum of old config only");
    assert!(a.handle_publish_response("d", r).unwrap().is_some());
}

#[test]
fn messages_roundtrip_through_serde() {
    let n = mk_node("", 1, true, 0);
    let msgs = vec![
        Message::PeersRequest { request_id: 1, cluster_name: "c".into(), source: n.clone(), known_peers: vec![n.clone()] },
        Message::StartJoin(StartJoin { source_node: n.clone(), term: 4 }),
        Message::JoinRequest {
            request_id: 2,
            cluster_name: "c".into(),
            source: n.clone(),
            min_term: 3,
            join: Some(Join { source_node: "a".into(), target_node: "b".into(), term: 3, last_accepted_term: 1, last_accepted_version: 7 }),
            cluster_uuid: "u".into(),
            cluster_uuid_committed: true,
        },
        Message::PublishRequest { request_id: 3, state: Box::new(state_with(1, 2, &["a"], &["a"])) },
        Message::PublishResponse { request_id: 3, result: Err("nope".into()) },
        Message::ApplyCommit { request_id: 4, commit: ApplyCommit { source_node: "a".into(), term: 1, version: 2 } },
        Message::FollowerCheck { request_id: 5, source: n.clone(), term: 1, target_ephemeral_id: "e".into() },
    ];
    for m in msgs {
        let json = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert!(m.action().starts_with("internal:"));
    }
}

#[test]
fn settings_parse() {
    let mut m = std::collections::BTreeMap::new();
    m.insert("cluster.name".to_string(), "prod".to_string());
    m.insert("discovery.seed_hosts".to_string(), "a:9300, b:9300".to_string());
    m.insert("cluster.initial_master_nodes".to_string(), "[n1,n2,n3]".to_string());
    m.insert("cluster.election.max_timeout".to_string(), "20s".to_string());
    m.insert("cluster.fault_detection.leader_check.interval".to_string(), "500ms".to_string());
    m.insert("cluster.no_master_block".to_string(), "all".to_string());
    let s = CoordinationSettings::from_flat(&m).unwrap();
    assert_eq!(s.cluster_name, "prod");
    assert_eq!(s.seed_hosts, vec!["a:9300", "b:9300"]);
    assert_eq!(s.initial_master_nodes, vec!["n1", "n2", "n3"]);
    assert_eq!(s.election_max_timeout_ms, 20_000);
    assert_eq!(s.leader_check_interval_ms, 500);
    assert_eq!(s.no_master_block, NoMasterBlock::All);
    assert_eq!(s.publish_timeout_ms, 30_000);
    assert_eq!(s.follower_lag_timeout_ms, 90_000);
    m.insert("cluster.no_master_block".to_string(), "bogus".to_string());
    assert!(CoordinationSettings::from_flat(&m).is_err());
    assert_eq!(parse_time_millis("2m").unwrap(), 120_000);
}

#[test]
fn settings_enforce_es_bounds() {
    let parse = |k: &str, v: &str| {
        let mut m = std::collections::BTreeMap::new();
        m.insert(k.to_string(), v.to_string());
        CoordinationSettings::from_flat(&m)
    };
    assert!(parse("cluster.fault_detection.leader_check.interval", "0ms").is_err());
    assert!(parse("cluster.fault_detection.leader_check.interval", "99ms").is_err());
    assert!(parse("cluster.fault_detection.leader_check.interval", "100ms").is_ok());
    assert!(parse("cluster.fault_detection.follower_check.retry_count", "0").is_err());
    assert!(parse("cluster.fault_detection.leader_check.retry_count", "1").is_ok());
    assert!(parse("cluster.election.max_timeout", "100ms").is_err());
    assert!(parse("cluster.election.initial_timeout", "11s").is_err());
    assert!(parse("cluster.publish.timeout", "0").is_err());
}

// ----------------------------------------------------- coordinator --

fn single(initial: bool) -> Coordinator {
    let n = mk_node("", 0, true, 1);
    let mut s = CoordinationSettings::new("c");
    if initial {
        s.initial_master_nodes = vec![n.name.clone()];
    }
    Coordinator::new(s, n, PersistedState::empty("c"), 7, Box::new(NoopStateUpdater))
}

/// Drive a lone coordinator through its timers until idle or `until`.
fn run_alone(c: &mut Coordinator, mut now: u64, until: u64) -> (u64, Vec<Output>) {
    let mut all = Vec::new();
    while let Some(d) = c.next_deadline() {
        if d > until {
            break;
        }
        now = now.max(d);
        all.extend(c.tick(now));
    }
    (now, all)
}

#[test]
fn single_node_bootstraps_and_elects_itself() {
    let mut c = single(true);
    let out = c.start(0);
    // Bootstrap persists before anything else.
    assert!(matches!(out.iter().find(|o| matches!(o, Output::Persist(_))), Some(Output::Persist(p)) if p.last_accepted_state.is_some()));
    let (_, out) = run_alone(&mut c, 0, 1_000);
    assert_eq!(c.mode(), Mode::Leader);
    let applied = c.applied_state();
    assert_eq!(applied.master_node_id.as_deref(), Some(c.local_node().id.as_str()));
    assert!(applied.blocks.global.iter().all(|b| b.id != 1 && b.id != 2));
    assert!(applied.metadata.cluster_uuid_committed);
    assert_eq!(applied.coordination.last_committed_config, cfg(&[c.local_node().id.as_str()]));
    // Every Send/ApplyState that follows a term bump is preceded by its Persist.
    let first_persist_term2 = out.iter().position(|o| matches!(o, Output::Persist(p) if p.current_term >= 1));
    let first_apply = out.iter().position(|o| matches!(o, Output::ApplyState(_)));
    assert!(first_persist_term2.unwrap() < first_apply.unwrap());
}

#[test]
fn node_without_initial_master_nodes_never_elects() {
    let mut c = single(false);
    c.start(0);
    run_alone(&mut c, 0, 60_000);
    assert_eq!(c.mode(), Mode::Candidate);
    assert_eq!(c.current_term(), 0);
    assert!(c.applied_state().blocks.global.iter().any(|b| b.id == 2));
}

#[test]
fn bootstrap_never_repeats_on_restart() {
    let mut c = single(true);
    let mut persisted = PersistedState::empty("c");
    for o in c.start(0) {
        if let Output::Persist(p) = o {
            p.apply_to(&mut persisted);
        }
    }
    let (_, out) = run_alone(&mut c, 0, 2_000);
    for o in out {
        if let Output::Persist(p) = o {
            p.apply_to(&mut persisted);
        }
    }
    let uuid = persisted.last_accepted_state.metadata.cluster_uuid.clone();
    assert!(persisted.last_accepted_state.metadata.cluster_uuid_committed);
    let mut n = c.local_node().clone();
    n.ephemeral_id = "restarted".into();
    let mut s = CoordinationSettings::new("c");
    s.initial_master_nodes = vec![n.name.clone()];
    let mut c2 = Coordinator::new(s, n, persisted, 9, Box::new(NoopStateUpdater));
    c2.start(10_000);
    run_alone(&mut c2, 10_000, 12_000);
    assert_eq!(c2.mode(), Mode::Leader);
    assert_eq!(c2.applied_state().metadata.cluster_uuid, uuid, "re-bootstrapping would have changed the cluster uuid");
    assert!(c2.current_term() >= 2);
}

#[test]
fn leader_tasks_batch_and_report() {
    let mut c = single(true);
    c.start(0);
    run_alone(&mut c, 0, 1_000);
    let v0 = c.applied_state().version;
    let ok = c
        .submit_task(
            1_000,
            Box::new(|s: &ClusterState| {
                let mut s = s.clone();
                s.metadata.persistent_settings.insert("k".into(), "v".into());
                Ok(s)
            }),
        )
        .unwrap();
    assert!(ok.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { task_id, result: Ok(v) } if *task_id == ok.id && *v == v0 + 1)));
    let bad = c.submit_task(1_000, Box::new(|_: &ClusterState| Err("rejected".into()))).unwrap();
    assert!(bad.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { result: Err(e), .. } if e == "rejected")));
    // No-op task: completes without a publication.
    let noop = c.submit_task(1_000, Box::new(|s: &ClusterState| Ok(s.clone()))).unwrap();
    assert!(noop.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { result: Ok(v), .. } if *v == v0 + 1)));
    assert!(!noop.outputs.iter().any(|o| matches!(o, Output::PublicationCompleted { .. })));
    // Driver-computed state.
    let mut next = c.master_state().clone();
    next.metadata.persistent_settings.insert("x".into(), "y".into());
    let p = c.publish(1_000, next).unwrap();
    assert!(p.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { result: Ok(v), .. } if *v == v0 + 2)));
    assert_eq!(c.applied_state().metadata.persistent_settings.get("x").and_then(|v| v.as_str()), Some("y"));
    // Stale driver state is refused.
    let mut stale = c.master_state().clone();
    stale.state_uuid = "old".into();
    let p = c.publish(1_000, stale).unwrap();
    assert!(p.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { result: Err(e), .. } if e.contains("stale"))));
}

#[test]
fn non_leader_rejects_tasks_and_exclusion_validation() {
    let mut c = single(false);
    c.start(0);
    let refused = c.submit_task(0, Box::new(|s: &ClusterState| Ok(s.clone()))).err().expect("refused");
    assert!(matches!(refused, CoordinationError::NotMaster(_)) && CoordinationError::is_not_master(&refused.to_string()), "{refused}");
    // A queued or publishing task failed on stepping down may have been computed: not this refusal.
    assert!(!CoordinationError::is_not_master("no longer master: term bumped"));
    let mut c = single(true);
    c.start(0);
    run_alone(&mut c, 0, 1_000);
    assert!(matches!(c.add_voting_config_exclusions(0, vec![], vec![]), Err(CoordinationError::Invalid(_))));
    let ids: Vec<String> = (0..11).map(|i| format!("ghost-{i}")).collect();
    let r = c.add_voting_config_exclusions(1_000, ids, vec![]).unwrap();
    assert!(r.outputs.iter().any(|o| matches!(o, Output::TaskCompleted { result: Err(e), .. } if e.contains("max_voting_config_exclusions"))));
}

#[test]
fn no_master_block_setting_all() {
    let n = mk_node("", 0, true, 1);
    let mut s = CoordinationSettings::new("c");
    s.no_master_block = NoMasterBlock::All;
    let mut c = Coordinator::new(s, n, PersistedState::empty("c"), 1, Box::new(NoopStateUpdater));
    c.start(0);
    let b = c.applied_state().blocks.global.iter().find(|b| b.id == 2).unwrap();
    assert!(b.blocks("read") && b.blocks("write"));
}
