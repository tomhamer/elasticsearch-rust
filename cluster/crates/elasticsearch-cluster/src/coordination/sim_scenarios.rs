//! Simulation scenarios. Seeded families run `seed_count(default)` seeds
//! (`COORD_SEEDS` overrides); `soak` runs every family with `COORD_SEEDS`
//! seeds (default 100000) and is `#[ignore]`d.
//!
//! Virtual-time bounds asserted below are derived from ES's defaults:
//! leader checks 1s/10s×3, follower checks 1s/10s×3, publish timeout 30s,
//! election back-off 100ms..10s.

use super::simulation::*;
use super::*;
use crate::state::ClusterState;
use rand::Rng;
use std::collections::BTreeSet;

fn master_ids(c: &Cluster) -> Vec<String> {
    c.nodes.iter().filter(|n| n.node.is_master_eligible()).map(|n| n.node.id.clone()).collect()
}

fn voting_config(c: &Cluster, l: usize) -> Vec<String> {
    c.nodes[l].c().applied_state().coordination.last_accepted_config.0.iter().cloned().collect()
}

/// A node in `members` leads in a term above `term` and a majority of
/// `members` applied a state it published: a new master is elected and
/// serving.
fn new_master_among(c: &Cluster, members: &[usize], term: u64) -> Option<usize> {
    let l = members.iter().copied().find(|i| c.nodes[*i].alive() && c.nodes[*i].c().mode() == Mode::Leader && c.nodes[*i].c().current_term() > term)?;
    let id = &c.nodes[l].node.id;
    // The master itself applies at the end of its publication (which can
    // wait for unreachable targets), so it counts as serving already.
    let n = members.iter().filter(|i| **i == l || (c.nodes[**i].alive() && c.nodes[**i].c().applied_state().master_node_id.as_ref() == Some(id))).count();
    (n * 2 > members.len()).then_some(l)
}

fn has_no_master_block(s: &ClusterState) -> bool {
    s.master_node_id.is_none() && s.blocks.global.iter().any(|b| b.id == 2)
}

/// Randomise the network for "realistic" seeds.
fn jitter_network(c: &mut Cluster) {
    c.min_delay = c.rng.gen_range(0..5);
    c.max_delay = c.min_delay + c.rng.gen_range(1..200);
}

// =============================================================== families ==

/// Fresh 1/3/5-node clusters (sometimes plus a data-only node) form and
/// stabilise with every master-eligible node in the voting configuration.
fn stabilise(seed: u64, trace: bool) -> Cluster {
    let masters = [1, 3, 5][(seed % 3) as usize];
    let data = (seed / 3 % 2) as usize;
    let mut c = Cluster::empty(seed);
    if trace {
        c.trace = Some(Default::default());
    }
    jitter_network(&mut c);
    c.max_disk_delay = c.rng.gen_range(0..20);
    let seeds: Vec<String> = (0..masters).map(|i| mk_node("", i, true, 0).transport_address).collect();
    let initial: Vec<String> = (0..masters).map(|i| mk_node("", i, true, 0).name).collect();
    for i in 0..masters + data {
        c.add_node("", i, i < masters, settings::with(seeds.clone(), if i < masters { initial.clone() } else { vec![] }), None);
    }
    // Staggered starts.
    let order: Vec<usize> = (0..c.nodes.len()).collect();
    for i in order {
        c.start(i);
        let gap = c.rng.gen_range(0..300);
        c.run_for(gap);
    }
    let t = c.run_until(30_000, |c| c.stable_leader().is_some());
    assert!(t.is_some(), "{masters}+{data} nodes did not stabilise within 30s");
    let l = c.stable_leader().unwrap();
    // Immediately: odd-sized, masters (or bootstrap placeholders) only, at
    // least min(3, masters). Masters that voted for a losing candidate in
    // the winning term join the configuration once the master has re-run
    // its election to collect their votes (see `Coordinator`), so within a
    // bounded time every master votes.
    let vc = voting_config(&c, l);
    let ms = master_ids(&c);
    assert!(vc.iter().all(|id| ms.contains(id) || id.starts_with(BOOTSTRAP_PLACEHOLDER_PREFIX)) && vc.len() % 2 == 1 && vc.len() >= masters.min(3), "bad voting config {vc:?}");
    assert!(
        c.run_until(30_000, |c| c.stable_leader().is_some_and(|l| voting_config(c, l) == master_ids(c))).is_some(),
        "not every master became a voter: {}",
        c.describe_pending()
    );
    let l = c.stable_leader().unwrap();
    let s = c.nodes[l].c().applied_state().clone();
    assert!(s.metadata.cluster_uuid_committed);
    assert!(s.blocks.global.is_empty());
    for _ in 0..3 {
        assert!(c.submit_log_task());
    }
    assert!(c.run_until(5_000, |c| c.acked_log_values.len() == 3).is_some(), "tasks not acked");
    c
}

/// Master crash (fail-fast) or isolation (black hole): a new master is
/// elected within bounded time.
fn master_failure(seed: u64, trace: bool) -> Cluster {
    let masters = if seed % 2 == 0 { 3 } else { 5 };
    let mut c = Cluster::new(seed, masters, 0);
    if trace {
        c.trace = Some(Default::default());
    }
    // Every master must be a voter first: with e.g. `{a, b, placeholder}`
    // losing `a` leaves no quorum, for any protocol.
    assert!(c.run_until(30_000, |c| c.stable_leader().is_some_and(|l| voting_config(c, l) == master_ids(c))).is_some(), "not all masters voting: {}", c.describe_pending());
    c.submit_log_task();
    c.run_for(1_000);
    let old = c.stable_leader().unwrap();
    let old_term = c.nodes[old].c().current_term();
    let blackhole = seed % 4 >= 2;
    let rest: Vec<usize> = (0..masters).filter(|i| *i != old).collect();
    if blackhole {
        c.partition(&[old], true);
    } else {
        c.crash(old);
    }
    // Fail-fast: leader check fails on the next check (<= 1s) + election.
    // Black hole: 3 leader-check timeouts (3 x (10s + 1s)) + election.
    let bound = if blackhole { 40_000 } else { 5_000 };
    let t = c.run_until(bound, |c| new_master_among(c, &rest, old_term).is_some());
    assert!(t.is_some(), "no new master within {bound}ms (blackhole={blackhole}, old leader {old}): {}", c.describe_pending());
    // Fully stable (old master removed from the state) once its follower
    // checks fail and, if needed, a publication to it times out.
    assert!(c.run_until(120_000, |c| c.stable_among(&rest).is_some()).is_some(), "survivors did not stabilise");
    c.submit_log_task();
    c.run_for(2_000);
    if blackhole {
        // The isolated old master must have stepped down by now (publish
        // timeout after its follower checks fail).
        assert!(c.run_until(60_000, |c| c.nodes[old].c().mode() != Mode::Leader).is_some(), "isolated master never stepped down");
        assert!(has_no_master_block(c.nodes[old].c().applied_state()));
        c.heal();
    } else {
        c.start(old);
    }
    assert!(c.run_until(60_000, |c| c.stable_leader().is_some() && c.stable_leader().map(|l| c.nodes[l].c().applied_state().nodes.len()) == Some(masters)).is_some(), "did not reform after recovery");
    c
}

/// A minority partition (with or without the master) never elects a
/// master and applies the no-master block; the majority stays available;
/// healing yields a single master.
fn minority_partition(seed: u64, trace: bool) -> Cluster {
    let mut c = Cluster::new(seed, 5, 0);
    if trace {
        c.trace = Some(Default::default());
    }
    jitter_network(&mut c);
    // "Minority" is relative to the voting configuration: wait until all 5
    // masters vote (a 3-voter config would let 2 nodes form a quorum).
    assert!(c.run_until(60_000, |c| c.stable_leader().is_some_and(|l| voting_config(c, l).len() == 5)).is_some(), "config never grew to 5: {}", c.describe_pending());
    let leader = c.stable_leader().unwrap();
    let mut minority: Vec<usize> = Vec::new();
    if seed % 2 == 0 {
        minority.push(leader);
    }
    while minority.len() < 2 {
        let i = c.rng.gen_range(0..5);
        if !minority.contains(&i) && i != leader {
            minority.push(i);
        }
    }
    let majority: Vec<usize> = (0..5).filter(|i| !minority.contains(i)).collect();
    let blackhole = seed % 4 >= 2;
    c.partition(&minority, blackhole);
    let terms_before: Vec<u64> = minority.iter().map(|i| c.nodes[*i].c().current_term()).collect();
    assert!(c.run_until(45_000, |c| new_master_among(c, &majority, 0).is_some() || c.stable_among(&majority).is_some()).is_some(), "majority has no master");
    assert!(c.run_until(120_000, |c| c.stable_among(&majority).is_some()).is_some(), "majority did not stabilise: {}", c.describe_pending());
    c.submit_log_task();
    // Leader-in-minority steps down after follower checks + publish timeout.
    c.run_for(60_000);
    for (k, &i) in minority.iter().enumerate() {
        let n = c.nodes[i].c();
        assert_ne!(n.mode(), Mode::Leader, "minority node {} is leader", n.local_node().name);
        assert!(has_no_master_block(n.applied_state()), "minority node lacks no-master block");
        // Pre-voting: no term inflation in the minority (beyond voting for
        // the majority's new master before the partition was noticed).
        assert!(n.current_term() <= terms_before[k] + 1, "minority term inflated: {} -> {}", terms_before[k], n.current_term());
    }
    let l = c.stable_among(&majority).expect("majority still stable");
    assert_eq!(c.nodes[l].c().applied_state().nodes.len(), 3);
    c.heal();
    assert!(c.run_until(60_000, |c| c.stable_leader().is_some()).is_some(), "no single master after heal");
    c
}

/// Jepsen's "bridge" partition: {0,1} | 2 | {3,4} with 2 seeing everyone.
/// Safety invariants (one committing master per term, linear history) are
/// checked throughout; the cluster keeps committing; after heal it reforms.
fn bridge(seed: u64, trace: bool) -> Cluster {
    let mut c = Cluster::new(seed, 5, 0);
    if trace {
        c.trace = Some(Default::default());
    }
    jitter_network(&mut c);
    assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
    c.drop_prob = 0.01;
    c.dup_prob = 0.01;
    let blackhole = seed % 2 == 0;
    for a in [0, 1] {
        for b in [3, 4] {
            c.cut_link(a, b, blackhole);
            c.cut_link(b, a, blackhole);
        }
    }
    let acked_before = c.acked_log_values.len();
    // Writes stall until the master drops the unreachable side: follower
    // checks (~33s) plus a publish timeout (30s) for each publication
    // stuck on an unreachable target or a lost message (with silent loss
    // each lost publish/commit message costs a publish timeout; over TCP it
    // would be a fast connection failure). Keep writing for at least 60s
    // and until something is acked, bounded by 240s.
    let start = c.now;
    while c.now < start + 60_000 || (c.acked_log_values.len() == acked_before && c.now < start + 240_000) {
        c.submit_log_task();
        let gap = c.rng.gen_range(100..1_500);
        c.run_for(gap);
    }
    assert!(c.acked_log_values.len() > acked_before, "bridged cluster never committed anything: {}", c.describe_pending());
    c.heal();
    c.drop_prob = 0.0;
    c.dup_prob = 0.0;
    // Worst case after the heal (seen with seed 463): a lost publish
    // request leaves a follower behind; its publication completes at the
    // publish timeout (30s), the lag detector removes it after
    // cluster.follower_lag.timeout (90s), it fails 3 leader checks (~3s)
    // and rejoins. Bound: 30 + 90 + 33 (black-holed checks) + election.
    assert!(c.run_until(200_000, |c| c.stable_leader().is_some()).is_some(), "no single master after bridge heal");
    let l = c.stable_leader().unwrap();
    let log = sim_log(c.nodes[l].c().applied_state());
    for v in &c.acked_log_values {
        assert!(log.contains(v), "acknowledged write {v} lost");
    }
    c
}

/// Random faults of every kind; safety checked continuously; liveness
/// after all faults are removed.
fn chaos(seed: u64, trace: bool) -> Cluster {
    let masters = if seed % 3 == 0 { 3 } else { 5 };
    let data = (seed % 2) as usize;
    let mut c = Cluster::new(seed, masters, data);
    if trace {
        c.trace = Some(Default::default());
    }
    jitter_network(&mut c);
    c.drop_prob = c.rng.gen_range(0.0..0.1);
    c.dup_prob = c.rng.gen_range(0.0..0.05);
    c.max_disk_delay = c.rng.gen_range(0..100);
    c.crash_mid_output_prob = c.rng.gen_range(0.0..0.002);
    let n = c.nodes.len();
    let max_excluded = masters / 2;
    let end = c.now + 60_000;
    while c.now < end {
        let gap = c.rng.gen_range(50..2_000);
        c.run_for(gap);
        match c.rng.gen_range(0..100) {
            0..=9 => {
                let i = c.rng.gen_range(0..n);
                if c.nodes[i].alive() {
                    c.crash(i);
                }
            }
            10..=24 => {
                let i = c.rng.gen_range(0..n);
                if !c.nodes[i].alive() {
                    c.start(i);
                }
            }
            25..=31 => {
                let group: Vec<usize> = (0..n).filter(|_| c.rng.gen_bool(0.5)).collect();
                let bh = c.rng.gen_bool(0.5);
                c.partition(&group, bh);
            }
            32..=37 => {
                let (a, b) = (c.rng.gen_range(0..n), c.rng.gen_range(0..n));
                let bh = c.rng.gen_bool(0.5);
                if a != b {
                    c.cut_link(a, b, bh);
                }
            }
            38..=49 => c.heal(),
            50..=53 => {
                // Keep the total number of excluded masters a minority, as
                // ES requires: excluding a majority can leave only stale
                // nodes eligible, which no election can fix (ES needs
                // `elasticsearch-node unsafe-bootstrap` then).
                if let Some(l) = c.any_leader() {
                    let mut excl: BTreeSet<String> = c.nodes[l].c().master_state().coordination.voting_config_exclusions.iter().map(|e| e.node_id.clone()).collect();
                    let k = c.rng.gen_range(1..=max_excluded.max(1));
                    let mut ids = Vec::new();
                    for _ in 0..k {
                        let id = mk_node("", c.rng.gen_range(0..masters), true, 0).id;
                        if excl.len() < max_excluded && excl.insert(id.clone()) {
                            ids.push(id);
                        }
                    }
                    if !ids.is_empty() {
                        c.leader_op(|co, now| co.add_voting_config_exclusions(now, ids, vec![]));
                    }
                }
            }
            54..=56 => {
                c.leader_op(|co, now| co.clear_voting_config_exclusions(now, false));
            }
            _ => {
                c.submit_log_task();
            }
        }
    }
    // Remove all faults; the cluster must recover.
    c.heal();
    c.drop_prob = 0.0;
    c.dup_prob = 0.0;
    c.crash_mid_output_prob = 0.0;
    for i in 0..n {
        if !c.nodes[i].alive() {
            c.start(i);
        }
    }
    let t = c.run_until(200_000, |c| c.stable_leader().is_some());
    assert!(t.is_some(), "cluster did not recover within 200s after faults stopped: {}", c.describe_pending());
    c.leader_op(|co, now| co.clear_voting_config_exclusions(now, false));
    assert!(c.run_until(120_000, |c| c.stable_leader().is_some()).is_some(), "not stable after clearing exclusions: {}", c.describe_pending());
    c.submit_log_task();
    assert!(
        c.run_until(120_000, |c| c.stable_leader().is_some() && c.pending_log_tasks_len() == 0).is_some(),
        "log tasks never completed: {}",
        c.describe_pending()
    );
    let l = c.stable_leader().unwrap();
    let log = sim_log(c.nodes[l].c().applied_state());
    for v in &c.acked_log_values {
        assert!(log.contains(v), "acknowledged write {v} lost");
    }
    c
}

mod settings {
    use super::super::CoordinationSettings;
    pub fn with(seeds: Vec<String>, initial: Vec<String>) -> CoordinationSettings {
        super::super::simulation::settings_for("sim", seeds, initial)
    }
}


// ========================================================== seed tests ==

#[test]
fn sim_stabilise() {
    run_seeds("stabilise", first_seed(), seed_count(1_000), stabilise);
}

#[test]
fn sim_master_failure() {
    run_seeds("master_failure", first_seed(), seed_count(1_000), master_failure);
}

#[test]
fn sim_minority_partition() {
    run_seeds("minority_partition", first_seed(), seed_count(1_000), minority_partition);
}

#[test]
fn sim_bridge_partition() {
    run_seeds("bridge", first_seed(), seed_count(1_000), bridge);
}

#[test]
fn sim_chaos() {
    run_seeds("chaos", first_seed(), seed_count(1_000), chaos);
}

/// Long soak: `COORD_SEEDS=100000 cargo test -p elasticsearch-cluster soak -- --ignored --nocapture`.
/// Uses seeds disjoint from the default runs.
#[test]
#[ignore]
fn soak() {
    let n = std::env::var("COORD_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let first = std::env::var("COORD_FIRST_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    run_seeds("soak/chaos", first, n, chaos);
    run_seeds("soak/bridge", first, n, bridge);
    run_seeds("soak/minority_partition", first, n, minority_partition);
    run_seeds("soak/master_failure", first, n, master_failure);
    run_seeds("soak/stabilise", first, n, stabilise);
}

#[test]
fn sim_is_deterministic() {
    let a = chaos(42, false);
    let b = chaos(42, false);
    assert_eq!(a.stats.steps, b.stats.steps);
    assert_eq!(a.stats.messages, b.stats.messages);
    assert_eq!(a.acked_log_values, b.acked_log_values);
}

// ====================================================== targeted tests ==

#[test]
fn rolling_restart_keeps_a_master_most_of_the_time() {
    for seed in 0..20 {
        let mut c = Cluster::new(seed, 5, 0);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        let (mut samples, mut with_master) = (0u64, 0u64);
        for i in 0..5 {
            c.crash(i);
            for _ in 0..50 {
                c.run_for(100);
                samples += 1;
                with_master += c.any_leader().is_some_and(|l| c.alive().iter().filter(|j| c.nodes[**j].c().applied_state().master_node_id.as_deref() == Some(c.nodes[l].node.id.as_str())).count() >= 3) as u64;
            }
            c.start(i);
            let t = c.run_until(60_000, |c| c.stable_leader().is_some());
            assert!(t.is_some(), "seed {seed}: no stable cluster after restarting node {i}");
            for _ in 0..50 {
                c.run_for(100);
                samples += 1;
                with_master += c.any_leader().is_some() as u64;
            }
        }
        let ratio = with_master as f64 / samples as f64;
        assert!(ratio >= 0.8, "seed {seed}: master available only {:.0}% of the rolling restart", ratio * 100.0);
    }
}

#[test]
fn voting_configuration_grows_and_shrinks() {
    for seed in 0..20 {
        let mut c = Cluster::empty(seed);
        let first = mk_node("", 0, true, 0);
        for i in 0..5 {
            let initial = if i == 0 { vec![first.name.clone()] } else { vec![] };
            c.add_node("", i, true, settings_for("sim", vec![first.transport_address.clone()], initial), None);
        }
        c.start(0);
        assert!(c.run_until(5_000, |c| c.stable_leader().is_some()).is_some());
        assert_eq!(voting_config(&c, 0).len(), 1);
        c.start(1);
        assert!(c.run_until(10_000, |c| c.stable_leader().is_some()).is_some());
        assert_eq!(voting_config(&c, 0).len(), 1, "two masters: stay at one voter (odd)");
        c.start(2);
        assert!(c.run_until(10_000, |c| c.stable_leader().is_some() && voting_config(c, 0).len() == 3).is_some(), "seed {seed}: did not grow to 3");
        c.start(3);
        c.start(4);
        assert!(c.run_until(10_000, |c| c.stable_leader().is_some() && voting_config(c, 0).len() == 5).is_some(), "seed {seed}: did not grow to 5");
        // Two (non-master) nodes die: removed, configuration shrinks to 3.
        let l = c.stable_leader().unwrap();
        let victims: Vec<usize> = (0..5).filter(|i| *i != l).take(2).collect();
        for v in &victims {
            c.crash(*v);
        }
        let alive = c.alive();
        assert!(
            c.run_until(30_000, |c| c.stable_among(&alive).is_some() && voting_config(c, l).len() == 3).is_some(),
            "seed {seed}: did not shrink to 3 (config {:?})",
            voting_config(&c, l)
        );
        // One more dies: 3-node minimum keeps the dead voter, quorum held.
        let v = *alive.iter().find(|i| **i != l).unwrap();
        c.crash(v);
        let alive = c.alive();
        assert!(c.run_until(30_000, |c| c.stable_among(&alive).is_some()).is_some());
        assert_eq!(voting_config(&c, l).len(), 3);
        assert!(voting_config(&c, l).contains(&c.nodes[v].node.id));
        assert_eq!(c.stable_among(&alive), Some(l));
    }
}

#[test]
fn exclusions_allow_shutting_down_two_of_five_masters() {
    for seed in 0..40 {
        let mut c = Cluster::new(seed, 5, 0);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        let l = c.stable_leader().unwrap();
        // Half the seeds exclude the current master too.
        let mut excluded: Vec<usize> = if seed % 2 == 0 { vec![l] } else { vec![] };
        while excluded.len() < 2 {
            let i = c.rng.gen_range(0..5);
            if !excluded.contains(&i) && (seed % 2 == 0 || i != l) {
                excluded.push(i);
            }
        }
        let ids: Vec<String> = excluded.iter().map(|i| c.nodes[*i].node.id.clone()).collect();
        let (leader, task) = c.leader_op(|co, now| co.add_voting_config_exclusions(now, ids.clone(), vec![])).unwrap();
        assert!(
            c.run_until(30_000, |c| c.stable_leader().is_some_and(|l| c.nodes[l].c().voting_config_excludes(&ids)) && c.task_results.get(&(leader, task)).is_some_and(|r| r.is_ok())).is_some(),
            "seed {seed}: exclusions not applied"
        );
        let l2 = c.stable_leader().unwrap();
        assert!(!excluded.contains(&l2), "seed {seed}: excluded master did not abdicate");
        assert_eq!(voting_config(&c, l2).len(), 3);
        for e in &excluded {
            c.crash(*e);
        }
        let rest = c.alive();
        assert!(c.run_until(40_000, |c| c.stable_among(&rest).is_some()).is_some(), "seed {seed}: lost the master after shutting down excluded nodes");
        assert_eq!(c.stable_among(&rest), Some(l2), "seed {seed}: master changed although it was not excluded");
        // Excluded nodes are gone: clearing with wait_for_removal succeeds.
        let (leader, task) = c.leader_op(|co, now| co.clear_voting_config_exclusions(now, true)).unwrap();
        assert!(c.run_until(5_000, |c| c.task_results.get(&(leader, task)).is_some()).is_some());
        assert!(c.task_results[&(leader, task)].is_ok());
        assert!(c.nodes[l2].c().applied_state().coordination.voting_config_exclusions.is_empty());
    }
}

#[test]
fn clearing_exclusions_waits_for_removal() {
    let mut c = Cluster::new(3, 3, 0);
    assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
    let l = c.stable_leader().unwrap();
    let other = (0..3).find(|i| *i != l).unwrap();
    let id = c.nodes[other].node.id.clone();
    c.leader_op(|co, now| co.add_voting_config_exclusions(now, vec![id], vec![])).unwrap();
    c.run_for(2_000);
    let (leader, task) = c.leader_op(|co, now| co.clear_voting_config_exclusions(now, true)).unwrap();
    c.run_for(2_000);
    assert!(c.task_results[&(leader, task)].as_ref().unwrap_err().contains("still in the cluster"));
}

#[test]
fn node_from_another_cluster_cannot_join() {
    for seed in 0..10 {
        // Cluster X forms; one of its nodes is later pointed at cluster A.
        let mut x = Cluster::empty(seed + 1000);
        let xseeds: Vec<String> = (0..3).map(|i| mk_node("x-", i, true, 0).transport_address).collect();
        let xinit: Vec<String> = (0..3).map(|i| mk_node("x-", i, true, 0).name).collect();
        for i in 0..3 {
            x.add_node("x-", i, true, settings_for("sim", xseeds.clone(), xinit.clone()), None);
            x.start(i);
        }
        assert!(x.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        let foreign = x.nodes[0].storage.clone();
        assert!(foreign.last_accepted_state.metadata.cluster_uuid_committed);

        let mut a = Cluster::new(seed, 3, 0);
        assert!(a.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        let uuid = a.nodes[a.stable_leader().unwrap()].c().applied_state().metadata.cluster_uuid.clone();
        assert_ne!(uuid, foreign.last_accepted_state.metadata.cluster_uuid);
        let aseeds: Vec<String> = (0..3).map(|i| mk_node("", i, true, 0).transport_address).collect();
        let idx = a.add_node("x-", 0, true, settings_for("sim", aseeds, vec![]), Some(foreign));
        a.start(idx);
        a.run_for(30_000);
        let members: Vec<usize> = (0..3).collect();
        let l = a.stable_among(&members).expect("cluster A unaffected");
        let s = a.nodes[l].c().applied_state();
        assert!(!s.nodes.contains_key(&a.nodes[idx].node.id), "foreign node joined");
        assert_eq!(s.metadata.cluster_uuid, uuid);
        let f = a.nodes[idx].c();
        assert_eq!(f.mode(), Mode::Candidate);
        assert_ne!(f.last_accepted_state().metadata.cluster_uuid, uuid);
        assert!(has_no_master_block(f.applied_state()));
    }
}

#[test]
fn full_cluster_restart_keeps_cluster_uuid_and_data() {
    for seed in 0..20 {
        let mut c = Cluster::new(seed, 3, 1);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        for _ in 0..5 {
            c.submit_log_task();
            c.run_for(200);
        }
        assert!(c.run_until(5_000, |c| c.acked_log_values.len() == 5).is_some());
        let uuid = c.nodes[0].c().applied_state().metadata.cluster_uuid.clone();
        for i in 0..4 {
            c.crash(i);
        }
        c.run_for(1_000);
        for i in 0..4 {
            c.start(i);
        }
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some(), "seed {seed}");
        let l = c.stable_leader().unwrap();
        let s = c.nodes[l].c().applied_state();
        assert_eq!(s.metadata.cluster_uuid, uuid, "seed {seed}: re-bootstrapped");
        assert_eq!(sim_log(s).len(), 5);
    }
}

#[test]
fn no_master_block_all_in_minority() {
    let mut c = Cluster::empty(5);
    let seeds: Vec<String> = (0..3).map(|i| mk_node("", i, true, 0).transport_address).collect();
    let initial: Vec<String> = (0..3).map(|i| mk_node("", i, true, 0).name).collect();
    for i in 0..3 {
        let mut s = settings_for("sim", seeds.clone(), initial.clone());
        s.no_master_block = NoMasterBlock::All;
        c.add_node("", i, true, s, None);
        c.start(i);
    }
    assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
    let l = c.stable_leader().unwrap();
    let f = (0..3).find(|i| *i != l).unwrap();
    c.partition(&[f], false);
    assert!(c.run_until(10_000, |c| c.nodes[f].c().mode() == Mode::Candidate).is_some());
    let s = c.nodes[f].c().applied_state();
    let b = s.blocks.global.iter().find(|b| b.id == 2).unwrap();
    assert!(b.blocks("read") && b.blocks("write") && s.master_node_id.is_none());
}

#[test]
fn asymmetric_partition_leader_cannot_hear_followers() {
    for seed in 0..50 {
        let mut c = Cluster::new(seed, 5, 0);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some()).is_some());
        let l = c.stable_leader().unwrap();
        // The leader can still send but receives nothing.
        for i in 0..5 {
            if i != l {
                c.cut_link(i, l, true);
            }
        }
        let rest: Vec<usize> = (0..5).filter(|i| *i != l).collect();
        for _ in 0..30 {
            c.submit_log_task();
            c.run_for(500);
        }
        assert!(c.run_until(120_000, |c| c.stable_among(&rest).is_some()).is_some(), "seed {seed}: majority never re-formed");
        assert!(c.run_until(60_000, |c| c.nodes[l].c().mode() != Mode::Leader).is_some(), "seed {seed}: deaf leader never stepped down");
        c.heal();
        assert!(c.run_until(60_000, |c| c.stable_leader().is_some()).is_some(), "seed {seed}");
    }
}

/// A closed connection is acted on at once, not after check timeouts: the
/// peer black-holes every message (so nothing ever fails fast and the
/// check-timeout path would need 3 x (10s + 1s)), but the survivors are
/// told their connection to it closed.
#[test]
fn disconnect_notification_fails_leader_and_follower_immediately() {
    for seed in 0..20u64 {
        // Followers lose the leader: new master without waiting for checks.
        let mut c = Cluster::new(seed, 3, 0);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some_and(|l| voting_config(c, l) == master_ids(c))).is_some(), "seed {seed}");
        c.run_for(1_000);
        let old = c.stable_leader().unwrap();
        let old_term = c.nodes[old].c().current_term();
        let rest: Vec<usize> = (0..3).filter(|i| *i != old).collect();
        c.partition(&[old], true);
        for &i in &rest {
            c.disconnect(i, old);
            assert_eq!(c.nodes[i].c().mode(), Mode::Candidate, "seed {seed}: follower kept a disconnected leader");
        }
        assert!(c.run_until(5_000, |c| new_master_among(c, &rest, old_term).is_some()).is_some(), "seed {seed}: no new master within 5s of the disconnect: {}", c.describe_pending());

        // The leader loses a follower: removed without waiting for checks.
        let mut c = Cluster::new(seed, 3, 0);
        assert!(c.run_until(30_000, |c| c.stable_leader().is_some_and(|l| voting_config(c, l) == master_ids(c))).is_some(), "seed {seed}");
        c.run_for(1_000);
        let l = c.stable_leader().unwrap();
        let term = c.nodes[l].c().current_term();
        let gone = (0..3).find(|i| *i != l).unwrap();
        let gone_id = c.nodes[gone].node.id.clone();
        c.partition(&[gone], true);
        c.disconnect(l, gone);
        // The removal publication cannot reach the black-holed node; it
        // commits on the remaining quorum.
        assert!(c.run_until(5_000, |c| !c.nodes[l].c().applied_state().nodes.contains_key(&gone_id)).is_some(), "seed {seed}: disconnected follower not removed within 5s: {}", c.describe_pending());
        assert_eq!(c.nodes[l].c().mode(), Mode::Leader, "seed {seed}");
        assert_eq!(c.nodes[l].c().current_term(), term, "seed {seed}: removal must not need a new term");

        // A disconnect from a node that is neither leader nor follower of
        // ours changes nothing.
        let f = (0..3).find(|i| *i != l && *i != gone).unwrap();
        c.disconnect(f, gone);
        assert_eq!(c.nodes[f].c().mode(), Mode::Follower, "seed {seed}");
    }
}
