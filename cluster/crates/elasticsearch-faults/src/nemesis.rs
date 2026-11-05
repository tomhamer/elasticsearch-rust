//! Fault schedules. A schedule is driven by a seeded RNG, so a seed fixes
//! the *shape* of the schedule (which role is targeted, partition layout,
//! durations); the concrete node numbers follow from which node holds that
//! role at run time and are logged as events.

use crate::api::Api;
use crate::cluster::{Cluster, ProcState};
use crate::net::{self, Drops, Net};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SCENARIOS: &[(&str, &str)] = &[
    ("kill-primary", "kill -9 the node holding a primary, restart it a few seconds later; repeat"),
    ("pause-primary", "SIGSTOP the primary's node 15-40s (GC-pause / VM-freeze; long enough to be removed), then SIGCONT"),
    ("isolate-master", "cut the elected master off from every other node, heal; repeat"),
    ("majority-minority", "split into majority | minority, the minority biased to contain the master and/or a primary"),
    ("bridge", "{a,b} | c | {d,e} where c sees everyone (the 2014 Jepsen split-brain topology)"),
    ("asymmetric", "one-way loss: master/primary cannot send, or cannot receive"),
    ("flapping", "a new random partition (or heal) every 1-4s"),
    ("rolling-restart", "restart every node in turn (graceful or kill -9), waiting for green between nodes"),
    ("full-restart", "kill -9 every node at once, restart all"),
    ("kill-during-recovery", "wipe a replica's node so it must fully recover, then kill the recovery source (primary) or target mid-recovery"),
    ("chaos", "random overlapping mix of all of the above (+ data wipe, + netem when enabled)"),
];

#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub t: f64,
    pub kind: String,
    pub nodes: Vec<usize>,
    /// Roles the targeted node(s) held when targeted (`master`, `primary:idx`, ...).
    pub roles: Vec<String>,
    pub detail: String,
}

/// A period during which a fault was in effect.
#[derive(Clone, Debug, Serialize)]
pub struct Window {
    pub kind: String,
    pub start: f64,
    pub end: Option<f64>,
    pub nodes: Vec<usize>,
    /// For partitions: nodes that could not reach a quorum of all nodes.
    pub minority: Vec<usize>,
    /// For partitions: the dropped `(from, to)` links.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub drops: Vec<(usize, usize)>,
}

pub struct Params {
    pub dur: Duration,
    pub short: bool,
    pub netem: bool,
    /// Expected seconds for the cluster to detect a failed/unreachable
    /// node (leader/follower checks: retry_count x (interval + timeout)).
    pub detect: f64,
}

pub struct Nemesis {
    pub cluster: Arc<Cluster>,
    pub net: Arc<Net>,
    pub api: Api,
    pub t0: Instant,
    pub indices: Vec<String>,
    pub events: Mutex<Vec<Event>>,
    pub windows: Mutex<Vec<Window>>,
    pub params: Params,
}

fn names(v: &[usize]) -> String {
    v.iter().map(|i| format!("n{}", i + 1)).collect::<Vec<_>>().join(",")
}

impl Nemesis {
    pub fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    pub fn log(&self, kind: &str, nodes: Vec<usize>, roles: Vec<String>, detail: String) {
        let e = Event { t: self.now(), kind: kind.into(), nodes, roles, detail };
        println!("    [{:7.2}s] nemesis {} {} {} {}", e.t, e.kind, names(&e.nodes), e.roles.join(","), e.detail);
        self.events.lock().push(e);
    }

    fn open(&self, kind: &str, nodes: Vec<usize>, minority: Vec<usize>, drops: Vec<(usize, usize)>) {
        self.windows.lock().push(Window { kind: kind.into(), start: self.now(), end: None, nodes, minority, drops });
    }

    fn close(&self, kind: &str, node: Option<usize>) {
        let t = self.now();
        for w in self.windows.lock().iter_mut() {
            if w.end.is_none() && w.kind == kind && node.map(|n| w.nodes.contains(&n)).unwrap_or(true) {
                w.end = Some(t);
            }
        }
    }

    async fn roles_of(&self, i: usize) -> Vec<String> {
        let mut r = Vec::new();
        if self.api.master().await == Some(i) {
            r.push("master".into());
        }
        if let Some(rt) = self.api.routing().await {
            for (ix, (p, reps)) in &rt.started {
                if *p == Some(i) {
                    r.push(format!("primary:{ix}"));
                } else if reps.contains(&i) {
                    r.push(format!("replica:{ix}"));
                }
            }
        }
        r
    }

    pub async fn kill(&self, i: usize, why: &str) {
        let roles = self.roles_of(i).await;
        self.cluster.kill9(i);
        self.log("kill", vec![i], roles, why.into());
        self.open("down", vec![i], vec![], vec![]);
    }

    pub async fn stop_graceful(&self, i: usize) {
        let roles = self.roles_of(i).await;
        self.log("stop", vec![i], roles, "SIGTERM".into());
        self.open("down", vec![i], vec![], vec![]);
        self.cluster.stop(i, Duration::from_secs(20)).await;
    }

    pub fn start(&self, i: usize) {
        if self.cluster.state(i) != ProcState::Down {
            return;
        }
        match self.cluster.start(i) {
            Ok(()) => self.log("start", vec![i], vec![], String::new()),
            Err(e) => self.log("start-failed", vec![i], vec![], e),
        }
        self.close("down", Some(i));
    }

    pub async fn pause(&self, i: usize) {
        let roles = self.roles_of(i).await;
        if self.cluster.pause(i) {
            self.log("pause", vec![i], roles, "SIGSTOP".into());
            self.open("pause", vec![i], vec![], vec![]);
        }
    }

    pub fn resume(&self, i: usize) {
        if self.cluster.resume(i) {
            self.log("resume", vec![i], vec![], "SIGCONT".into());
            self.close("pause", Some(i));
        }
    }

    pub fn wipe(&self, i: usize) {
        match self.cluster.wipe(i) {
            Ok(()) => self.log("wipe", vec![i], vec![], "data dir deleted".into()),
            Err(e) => self.log("wipe-failed", vec![i], vec![], e),
        }
    }

    pub async fn partition(&self, drops: Drops, label: &str, roles: Vec<String>) {
        self.heal_net();
        let minority = net::minority_nodes(self.cluster.n, &drops);
        let pairs: Vec<String> = drops.iter().map(|(a, b)| format!("{}>{}", a + 1, b + 1)).collect();
        if let Err(e) = self.net.apply(&drops) {
            self.log("partition-failed", vec![], vec![], e);
            return;
        }
        let affected: Vec<usize> = {
            let mut s: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            for (a, b) in &drops {
                s.insert(*a);
                s.insert(*b);
            }
            s.into_iter().collect()
        };
        self.log("partition", affected.clone(), roles, format!("{label}; minority=[{}]; drop {}", names(&minority), pairs.join(" ")));
        self.open("partition", affected, minority, drops.iter().copied().collect());
    }

    pub fn heal_net(&self) {
        if self.net.current().is_empty() {
            return;
        }
        match self.net.heal() {
            Ok(()) => self.log("heal", vec![], vec![], String::new()),
            Err(e) => self.log("heal-failed", vec![], vec![], e),
        }
        self.close("partition", None);
    }

    /// Undo every fault: network, netem, pauses, downed nodes.
    pub fn heal_all(&self) {
        self.heal_net();
        if self.params.netem {
            self.net.netem_off();
            self.close("netem", None);
        }
        for i in 0..self.cluster.n {
            match self.cluster.state(i) {
                ProcState::Paused => self.resume(i),
                ProcState::Down => self.start(i),
                ProcState::Running => {}
            }
        }
    }

    async fn sleep_s(&self, end: f64, lo: f64, hi: f64, rng: &mut StdRng) {
        let d = if hi > lo { rng.gen_range(lo..hi) } else { lo };
        let left = (end - self.now()).max(0.0);
        tokio::time::sleep(Duration::from_secs_f64(d.min(left.max(0.0)))).await;
    }

    /// Sleep `lo..hi` seconds regardless of the phase end (fault durations
    /// are not truncated: a fault started is always held for its duration).
    async fn hold(&self, lo: f64, hi: f64, rng: &mut StdRng) {
        let d = if hi > lo { rng.gen_range(lo..hi) } else { lo };
        tokio::time::sleep(Duration::from_secs_f64(d)).await;
    }

    fn random_index(&self, rng: &mut StdRng) -> String {
        self.indices[rng.gen_range(0..self.indices.len())].clone()
    }

    async fn primary_or_random(&self, rng: &mut StdRng) -> (usize, Vec<String>) {
        let ix = self.random_index(rng);
        match self.api.primary(&ix).await {
            Some(p) => (p, vec![format!("primary:{ix}")]),
            None => (rng.gen_range(0..self.cluster.n), vec!["random(no primary found)".into()]),
        }
    }

    async fn master_or_random(&self, rng: &mut StdRng) -> (usize, Vec<String>) {
        match self.api.master().await {
            Some(m) => (m, vec!["master".into()]),
            None => (rng.gen_range(0..self.cluster.n), vec!["random(no master found)".into()]),
        }
    }

    /// Partition hold times. Short mode always outlasts fault detection
    /// (so every partition forces a reaction); long mode also includes
    /// partitions that heal before anyone notices.
    fn partition_lengths(&self) -> (f64, f64) {
        let d = self.params.detect;
        if self.params.short { (d + 12.0, d + 25.0) } else { (5.0, d + 60.0) }
    }

    pub async fn run(&self, scenario: &str, rng: &mut StdRng) {
        let end = self.now() + self.params.dur.as_secs_f64();
        let n = self.cluster.n;
        let minority_size = (n - 1) / 2;
        let (plo, phi) = self.partition_lengths();
        match scenario {
            "kill-primary" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 4.0, rng).await;
                    let (p, roles) = self.primary_or_random(rng).await;
                    let _ = roles;
                    self.kill(p, "kill -9 primary").await;
                    self.hold(3.0, 10.0, rng).await;
                    self.start(p);
                }
            }
            "pause-primary" => {
                while self.now() < end {
                    self.sleep_s(end, 2.0, 4.0, rng).await;
                    let (p, _) = self.primary_or_random(rng).await;
                    self.pause(p).await;
                    if self.params.short { self.hold(20.0, 45.0, rng).await } else { self.hold(15.0, 60.0, rng).await }
                    self.resume(p);
                    self.sleep_s(end, 5.0, 8.0, rng).await;
                }
            }
            "isolate-master" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 3.0, rng).await;
                    let (m, roles) = self.master_or_random(rng).await;
                    self.partition(net::isolate(n, m), "isolate", roles).await;
                    self.hold(plo, phi, rng).await;
                    self.heal_net();
                    self.sleep_s(end, 5.0, 8.0, rng).await;
                }
            }
            "majority-minority" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 3.0, rng).await;
                    let mut minority: Vec<usize> = Vec::new();
                    let mut roles = Vec::new();
                    if rng.gen_bool(0.5) {
                        if let Some(m) = self.api.master().await {
                            minority.push(m);
                            roles.push(format!("master=n{}", m + 1));
                        }
                    }
                    if rng.gen_bool(0.6) && minority.len() < minority_size {
                        let ix = self.random_index(rng);
                        if let Some(p) = self.api.primary(&ix).await {
                            if !minority.contains(&p) {
                                minority.push(p);
                                roles.push(format!("primary:{ix}=n{}", p + 1));
                            }
                        }
                    }
                    let mut rest: Vec<usize> = (0..n).filter(|i| !minority.contains(i)).collect();
                    rest.shuffle(rng);
                    while minority.len() < minority_size {
                        minority.push(rest.pop().expect("node"));
                    }
                    minority.sort();
                    let majority: Vec<usize> = (0..n).filter(|i| !minority.contains(i)).collect();
                    self.partition(net::split(&[minority.clone(), majority]), "majority|minority", roles).await;
                    self.hold(plo, phi, rng).await;
                    self.heal_net();
                    self.sleep_s(end, 5.0, 8.0, rng).await;
                }
            }
            "bridge" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 3.0, rng).await;
                    let mut perm: Vec<usize> = (0..n).collect();
                    perm.shuffle(rng);
                    // Half the time, the master is NOT the bridge (it ends up on a side).
                    let master = self.api.master().await;
                    if let (Some(m), true) = (master, rng.gen_bool(0.5)) {
                        perm.retain(|x| *x != m);
                        perm.insert(0, m);
                    }
                    let k = (n - 1) / 2;
                    let left = perm[..k].to_vec();
                    let bridge = perm[k];
                    let right = perm[k + 1..].to_vec();
                    let roles = vec![format!("bridge=n{}", bridge + 1), format!("master={}", master.map(|m| format!("n{}", m + 1)).unwrap_or("?".into()))];
                    self.partition(net::bridge(&left, &right), "bridge", roles).await;
                    self.hold(plo, phi, rng).await;
                    self.heal_net();
                    self.sleep_s(end, 5.0, 8.0, rng).await;
                }
            }
            "asymmetric" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 3.0, rng).await;
                    let v = rng.gen_range(0..4);
                    let (x, mut roles) = if v < 2 { self.master_or_random(rng).await } else { self.primary_or_random(rng).await };
                    let others: Vec<usize> = (0..n).filter(|i| *i != x).collect();
                    let (drops, label) = if v % 2 == 0 { (net::one_way(&[x], &others), "x cannot send") } else { (net::one_way(&others, &[x]), "x cannot receive") };
                    roles.push(format!("x=n{}", x + 1));
                    self.partition(drops, label, roles).await;
                    self.hold(plo, phi, rng).await;
                    self.heal_net();
                    self.sleep_s(end, 5.0, 8.0, rng).await;
                }
            }
            "flapping" => {
                while self.now() < end {
                    let choice = rng.gen_range(0..5);
                    let mut perm: Vec<usize> = (0..n).collect();
                    perm.shuffle(rng);
                    match choice {
                        0 => self.heal_net(),
                        1 => self.partition(net::isolate(n, perm[0]), "flap-isolate", vec![]).await,
                        2 => {
                            let k = rng.gen_range(1..n);
                            self.partition(net::split(&[perm[..k].to_vec(), perm[k..].to_vec()]), "flap-split", vec![]).await
                        }
                        3 => {
                            let k = (n - 1) / 2;
                            self.partition(net::bridge(&perm[..k], &perm[k + 1..]), "flap-bridge", vec![]).await
                        }
                        _ => {
                            let k = rng.gen_range(1..n);
                            self.partition(net::one_way(&perm[..k], &perm[k..]), "flap-oneway", vec![]).await
                        }
                    }
                    self.hold(1.0, 4.0, rng).await;
                }
                self.heal_net();
            }
            "rolling-restart" => {
                let mut order: Vec<usize> = (0..n).collect();
                order.shuffle(rng);
                for i in order {
                    if rng.gen_bool(0.5) {
                        self.stop_graceful(i).await;
                    } else {
                        self.kill(i, "kill -9 (rolling)").await;
                    }
                    self.hold(1.0, 3.0, rng).await;
                    self.start(i);
                    match self.api.wait_converged(n, Duration::from_secs(90)).await {
                        Ok(s) => self.log("green", vec![], vec![], format!("after {s:.1}s")),
                        Err(e) => self.log("not-green", vec![], vec![], e),
                    }
                }
            }
            "full-restart" => {
                let rounds = if self.params.short { 1 } else { 3 };
                for _ in 0..rounds {
                    self.sleep_s(f64::MAX, 3.0, 6.0, rng).await;
                    let graceful = !self.params.short && rng.gen_bool(0.3);
                    for i in 0..n {
                        if graceful {
                            self.stop_graceful(i).await;
                        } else {
                            self.cluster.kill9(i);
                            self.open("down", vec![i], vec![], vec![]);
                        }
                    }
                    self.log(if graceful { "stop-all" } else { "kill-all" }, (0..n).collect(), vec![], String::new());
                    self.hold(1.0, 4.0, rng).await;
                    for i in 0..n {
                        self.start(i);
                    }
                    match self.api.wait_converged(n, Duration::from_secs(120)).await {
                        Ok(s) => self.log("green", vec![], vec![], format!("after {s:.1}s")),
                        Err(e) => self.log("not-green", vec![], vec![], e),
                    }
                }
            }
            "kill-during-recovery" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 2.0, rng).await;
                    let ix = self.random_index(rng);
                    let Some(rt) = self.api.routing().await else { continue };
                    let Some((Some(p), reps)) = rt.started.get(&ix).cloned() else { continue };
                    let Some(&r) = reps.choose(rng) else { continue };
                    self.kill(r, &format!("replica of {ix}, will be wiped")).await;
                    self.wipe(r);
                    self.start(r);
                    // Wait for the wiped node's copy of `ix` to start recovering.
                    let t_wait = self.now();
                    let mut recovering = false;
                    while self.now() - t_wait < 30.0 {
                        if let Some(rt) = self.api.routing().await {
                            if rt.initializing.get(&ix).map(|v| v.contains(&r)).unwrap_or(false) {
                                recovering = true;
                                break;
                            }
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    let kill_source = rng.gen_bool(0.6);
                    let victim = if kill_source { p } else { r };
                    self.log("recovery", vec![r], vec![], format!("observed INITIALIZING copy of {ix} on n{}: {recovering}", r + 1));
                    self.kill(victim, if kill_source { "recovery source (primary) mid-recovery" } else { "recovery target mid-recovery" }).await;
                    self.hold(3.0, 8.0, rng).await;
                    self.start(victim);
                    match self.api.wait_converged(n, Duration::from_secs(120)).await {
                        Ok(s) => self.log("green", vec![], vec![], format!("after {s:.1}s")),
                        Err(e) => self.log("not-green", vec![], vec![], e),
                    }
                }
            }
            "chaos" => {
                while self.now() < end {
                    self.sleep_s(end, 1.0, 4.0, rng).await;
                    let faults = if rng.gen_bool(0.35) { 2 } else { 1 };
                    for _ in 0..faults {
                        self.random_fault(rng).await;
                    }
                    self.hold(3.0, self.params.detect + if self.params.short { 10.0 } else { 40.0 }, rng).await;
                    self.heal_all();
                }
            }
            other => self.log("unknown-scenario", vec![], vec![], other.into()),
        }
        self.heal_all();
    }

    async fn random_fault(&self, rng: &mut StdRng) {
        let n = self.cluster.n;
        let budget = (n - 1) / 2;
        let impaired = self.cluster.states().iter().filter(|s| **s != ProcState::Running).count();
        let choice = rng.gen_range(0..if self.params.netem { 8 } else { 7 });
        let (target, roles) = match rng.gen_range(0..3) {
            0 => self.master_or_random(rng).await,
            1 => self.primary_or_random(rng).await,
            _ => (rng.gen_range(0..n), vec![]),
        };
        let _ = roles;
        match choice {
            0 if impaired < budget => self.kill(target, "chaos").await,
            1 if impaired < budget => self.pause(target).await,
            2 if impaired < budget => self.stop_graceful(target).await,
            3 => {
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(rng);
                let k = rng.gen_range(1..=budget.max(1));
                self.partition(net::split(&[perm[..k].to_vec(), perm[k..].to_vec()]), "chaos-split", vec![]).await
            }
            4 => {
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(rng);
                let k = (n - 1) / 2;
                self.partition(net::bridge(&perm[..k], &perm[k + 1..]), "chaos-bridge", vec![]).await
            }
            5 => {
                let others: Vec<usize> = (0..n).filter(|i| *i != target).collect();
                let d = if rng.gen_bool(0.5) { net::one_way(&[target], &others) } else { net::one_way(&others, &[target]) };
                self.partition(d, "chaos-oneway", vec![format!("x=n{}", target + 1)]).await
            }
            6 if impaired == 0 => {
                // Data loss on one node: only while everything else is
                // healthy, so other in-sync copies exist (G3's precondition).
                if self.api.wait_converged(n, Duration::from_secs(5)).await.is_ok() {
                    self.kill(target, "chaos wipe").await;
                    self.wipe(target);
                    self.start(target);
                }
            }
            7 => {
                let (d, j, l) = (rng.gen_range(5..100u32), rng.gen_range(0..50u32), [0.0, 0.5, 2.0, 5.0][rng.gen_range(0..4)]);
                match self.net.netem_on(d, j, l) {
                    Ok(()) => {
                        self.log("netem", (0..n).collect(), vec![], format!("delay {d}ms±{j}ms loss {l}%"));
                        self.open("netem", (0..n).collect(), vec![], vec![]);
                    }
                    Err(e) => self.log("netem-failed", vec![], vec![], e),
                }
            }
            _ => {}
        }
    }
}
