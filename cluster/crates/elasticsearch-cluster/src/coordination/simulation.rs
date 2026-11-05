//! Deterministic simulation of a cluster of coordinators (the equivalent of
//! ES's `AbstractCoordinatorTestCase`): virtual time, a message network with
//! random delays, drops, black-hole and fail-fast partitions, node crashes
//! (including crashes half-way through executing a batch of outputs) and
//! restarts from persisted state. Safety invariants are checked on every
//! output; liveness scenarios assert bounded convergence.
//!
//! Invariants (checked continuously):
//! * I1 at most one leader per term;
//! * I2 at most one state per (term, version) is ever accepted;
//! * I3 states accepted by a publish quorum ("committed") are unique per
//!   version and linear: every committed state's task log is a prefix of
//!   every later committed state's log (committed data is never lost
//!   across master changes);
//! * I4 every applied state (version > 0) is a committed state;
//! * I5 applied versions are monotonic per node incarnation;
//! * I6 the `last_committed_config` of every persisted or published state
//!   is a bootstrap configuration or the accepted configuration of a
//!   committed state (configurations only change via commits);
//! * I7 the persisted term never decreases;
//! * I8 every committed state descends (via the publishing leader's
//!   parent links) from the previously committed state, and all states
//!   committed in one term were published by the same master;
//! * I9 each node's applied sequence is an ordered subsequence of the
//!   single global committed history (no forks).
//!
//! Fault model: message loss, duplication, reordering and arbitrary delay;
//! symmetric, asymmetric (one-way) and bridged partitions, each either
//! black-hole (silent) or fail-fast (send failure); crashes that lose
//! everything not yet fsynced (only completed `Persist` outputs survive),
//! including crashes part-way through a batch of outputs; restarts with a
//! fresh ephemeral id; slow disks (a `Persist` completes after a random
//! delay and every later output of that node waits for it, as the driver
//! contract requires); per-node clock offsets (no synchronised clocks).
//!
//! Reproducibility: everything derives from the seed. `run_seeds` re-runs
//! a failing seed with tracing enabled and prints the seed and the last
//! events before the failure.

use super::*;
use crate::state::{ClusterState, DiscoveryNode, Role, VotingConfiguration};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const LOG_KEY: &str = "sim_log";

pub fn sim_log(s: &ClusterState) -> Vec<u64> {
    s.metadata.customs.get(LOG_KEY).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default()
}

pub fn mk_node(prefix: &str, i: usize, master: bool, incarnation: u64) -> DiscoveryNode {
    let roles = if master { Role::defaults() } else { [Role::Data, Role::Ingest].into_iter().collect() };
    DiscoveryNode {
        id: format!("{prefix}id-{i}"),
        name: format!("{prefix}node-{i}"),
        ephemeral_id: format!("{prefix}eph-{i}-{incarnation}"),
        transport_address: format!("{prefix}10.0.0.{i}:9300"),
        http_address: format!("{prefix}10.0.0.{i}:9200"),
        host: format!("{prefix}10.0.0.{i}"),
        ip: format!("{prefix}10.0.0.{i}"),
        roles,
        attributes: BTreeMap::new(),
        version: "8.15.0".into(),
    }
}

/// Settings for a test cluster: fast enough defaults are ES's own.
pub fn settings_for(cluster: &str, seeds: Vec<String>, initial_masters: Vec<String>) -> CoordinationSettings {
    let mut s = CoordinationSettings::new(cluster);
    s.seed_hosts = seeds;
    s.initial_master_nodes = initial_masters;
    s
}

pub struct SimNode {
    pub settings: CoordinationSettings,
    pub node: DiscoveryNode,
    pub prefix: String,
    pub index_in_prefix: usize,
    pub storage: PersistedState,
    pub coord: Option<Coordinator>,
    pub incarnation: u64,
    pub applied_version: u64,
    /// Offset added to the virtual clock: nodes do not share a clock.
    pub clock_offset: u64,
    /// Outputs not yet executed (blocked behind an in-progress fsync).
    queue: VecDeque<Output>,
    disk_busy: bool,
    /// Applied (version, uuid) sequence of this incarnation.
    pub applied_seq: Vec<(u64, String)>,
}

impl SimNode {
    pub fn alive(&self) -> bool {
        self.coord.is_some()
    }
    pub fn c(&self) -> &Coordinator {
        self.coord.as_ref().expect("node is alive")
    }
}

enum Event {
    Deliver { from: usize, to: usize, msg: Message },
    SendFailure { node: usize, incarnation: u64, request_id: u64 },
    PersistDone { node: usize, incarnation: u64, update: PersistUpdate },
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub steps: u64,
    pub messages: u64,
    pub invariant_checks: u64,
    pub elections: u64,
    pub commits: u64,
}

struct Accepted {
    uuid: String,
    acceptors: BTreeSet<String>,
    committed_config: VotingConfiguration,
    accepted_config: VotingConfiguration,
    log: Vec<u64>,
    committed: bool,
}

pub struct Cluster {
    pub rng: StdRng,
    pub now: u64,
    pub nodes: Vec<SimNode>,
    addr_index: BTreeMap<String, usize>,
    events: BTreeMap<(u64, u64), Event>,
    seq: u64,
    pub min_delay: u64,
    pub max_delay: u64,
    pub drop_prob: f64,
    /// Probability that a node crashes part-way through a batch of outputs.
    pub crash_mid_output_prob: f64,
    pub dup_prob: f64,
    /// Max fsync latency (0 = synchronous disk).
    pub max_disk_delay: u64,
    pub trace: Option<VecDeque<String>>,
    blackholed: BTreeSet<(usize, usize)>,
    disconnected: BTreeSet<(usize, usize)>,
    // Invariant bookkeeping.
    leaders_by_term: BTreeMap<u64, String>,
    accepted: BTreeMap<(u64, u64), Accepted>,
    committed_by_version: BTreeMap<u64, (u64, String, Vec<u64>)>,
    committed_configs: BTreeSet<Vec<String>>,
    bootstrap_configs: BTreeSet<Vec<String>>,
    pub stats: Stats,
    next_log_value: u64,
    pub acked_log_values: BTreeSet<u64>,
    pending_log_tasks: BTreeMap<(usize, u64), u64>,
    pub task_results: BTreeMap<(usize, u64), Result<u64, String>>,
    /// state uuid -> (parent uuid, version), recorded when the publishing
    /// master persists its own publication.
    parents: BTreeMap<String, (String, u64)>,
    committed_master_by_term: BTreeMap<u64, Option<String>>,
}

fn cfg_key(c: &VotingConfiguration) -> Vec<String> {
    c.0.iter().cloned().collect()
}

fn is_prefix(a: &[u64], b: &[u64]) -> bool {
    a.len() <= b.len() && a == &b[..a.len()]
}

impl Cluster {
    pub fn empty(seed: u64) -> Cluster {
        Cluster {
            rng: StdRng::seed_from_u64(seed),
            now: 0,
            nodes: Vec::new(),
            addr_index: BTreeMap::new(),
            events: BTreeMap::new(),
            seq: 0,
            min_delay: 1,
            max_delay: 30,
            drop_prob: 0.0,
            crash_mid_output_prob: 0.0,
            dup_prob: 0.0,
            max_disk_delay: 0,
            trace: None,
            blackholed: BTreeSet::new(),
            disconnected: BTreeSet::new(),
            leaders_by_term: BTreeMap::new(),
            accepted: BTreeMap::new(),
            committed_by_version: BTreeMap::new(),
            committed_configs: BTreeSet::new(),
            bootstrap_configs: BTreeSet::new(),
            stats: Stats::default(),
            next_log_value: 1,
            acked_log_values: BTreeSet::new(),
            pending_log_tasks: BTreeMap::new(),
            task_results: BTreeMap::new(),
            parents: BTreeMap::new(),
            committed_master_by_term: BTreeMap::new(),
        }
    }

    fn tr(&mut self, f: impl FnOnce() -> String) {
        let now = self.now;
        if let Some(t) = self.trace.as_mut() {
            if t.len() >= trace_lines() {
                t.pop_front();
            }
            t.push_back(format!("[{now:>8}] {}", f()));
        }
    }

    /// `masters` master-eligible nodes (all listed in
    /// `cluster.initial_master_nodes` and `discovery.seed_hosts`) plus
    /// `data` data-only nodes, all started.
    pub fn new(seed: u64, masters: usize, data: usize) -> Cluster {
        let mut c = Cluster::empty(seed);
        let seeds: Vec<String> = (0..masters).map(|i| mk_node("", i, true, 0).transport_address).collect();
        let initial: Vec<String> = (0..masters).map(|i| mk_node("", i, true, 0).name).collect();
        for i in 0..masters + data {
            let s = settings_for("sim", seeds.clone(), if i < masters { initial.clone() } else { vec![] });
            c.add_node("", i, i < masters, s, None);
        }
        for i in 0..c.nodes.len() {
            c.start(i);
        }
        c
    }

    /// Add a (not yet started) node. `storage` defaults to empty.
    pub fn add_node(&mut self, prefix: &str, i: usize, master: bool, settings: CoordinationSettings, storage: Option<PersistedState>) -> usize {
        let node = mk_node(prefix, i, master, 0);
        let idx = self.nodes.len();
        self.addr_index.insert(node.transport_address.clone(), idx);
        let storage = storage.unwrap_or_else(|| PersistedState::empty(&settings.cluster_name));
        let clock_offset = self.rng.gen_range(0..1_000_000_000u64);
        self.nodes.push(SimNode {
            settings,
            node,
            prefix: prefix.to_string(),
            index_in_prefix: i,
            storage,
            coord: None,
            incarnation: 0,
            applied_version: 0,
            clock_offset,
            queue: VecDeque::new(),
            disk_busy: false,
            applied_seq: Vec::new(),
        });
        idx
    }

    pub fn start(&mut self, idx: usize) {
        assert!(!self.nodes[idx].alive());
        let n = &mut self.nodes[idx];
        n.incarnation += 1;
        n.node = mk_node(&n.prefix, n.index_in_prefix, n.node.is_master_eligible(), n.incarnation);
        n.applied_version = 0;
        n.applied_seq.clear();
        n.queue.clear();
        n.disk_busy = false;
        let seed = self.rng.r#gen::<u64>();
        let mut coord = Coordinator::new(n.settings.clone(), n.node.clone(), n.storage.clone(), seed, Box::new(NoopStateUpdater));
        let out = coord.start(self.now + n.clock_offset);
        let name = n.node.name.clone();
        n.coord = Some(coord);
        self.tr(|| format!("START {name}"));
        self.process(idx, out);
    }

    /// Crash: everything not yet fsynced (queued outputs, in-progress
    /// persists) is lost.
    pub fn crash(&mut self, idx: usize) {
        let n = &mut self.nodes[idx];
        n.coord = None;
        n.queue.clear();
        n.disk_busy = false;
        let name = n.node.name.clone();
        // Tasks submitted to this incarnation can never complete.
        self.pending_log_tasks.retain(|(i, _), _| *i != idx);
        self.task_results.retain(|(i, _), _| *i != idx);
        self.tr(|| format!("CRASH {name}"));
    }

    /// The transport connection `idx -> peer` closed: deliver the
    /// disconnect notification the node driver gets from its transport.
    pub fn disconnect(&mut self, idx: usize, peer: usize) {
        let addr = self.nodes[peer].node.transport_address.clone();
        let now = self.local_now(idx);
        let out = self.nodes[idx].coord.as_mut().expect("alive").handle_disconnect(now, &addr);
        self.process(idx, out);
    }

    /// Local clock of a node.
    fn local_now(&self, idx: usize) -> u64 {
        self.now + self.nodes[idx].clock_offset
    }

    /// Cut the directed link `from -> to` only (asymmetric partition).
    pub fn cut_link(&mut self, from: usize, to: usize, blackhole: bool) {
        if blackhole {
            self.blackholed.insert((from, to));
        } else {
            self.disconnected.insert((from, to));
        }
    }

    pub fn alive(&self) -> Vec<usize> {
        (0..self.nodes.len()).filter(|i| self.nodes[*i].alive()).collect()
    }

    /// Cut every link between `group` and the other nodes.
    pub fn partition(&mut self, group: &[usize], blackhole: bool) {
        for a in 0..self.nodes.len() {
            for b in 0..self.nodes.len() {
                if group.contains(&a) != group.contains(&b) {
                    if blackhole {
                        self.blackholed.insert((a, b));
                    } else {
                        self.disconnected.insert((a, b));
                    }
                }
            }
        }
    }

    pub fn heal(&mut self) {
        self.blackholed.clear();
        self.disconnected.clear();
    }

    fn schedule(&mut self, at: u64, e: Event) {
        self.seq += 1;
        self.events.insert((at, self.seq), e);
    }

    fn delay(&mut self) -> u64 {
        self.rng.gen_range(self.min_delay..=self.max_delay)
    }

    fn fail_request(&mut self, from: usize, msg: &Message) {
        if let Some(request_id) = msg.request_id() {
            let at = self.now + self.delay();
            let incarnation = self.nodes[from].incarnation;
            self.schedule(at, Event::SendFailure { node: from, incarnation, request_id });
        }
    }

    fn route(&mut self, from: usize, to_addr: &str, msg: Message) {
        self.stats.messages += 1;
        if let Message::PublishRequest { state, .. } = &msg {
            self.check_config_committed(state, "published");
        }
        let Some(&to) = self.addr_index.get(to_addr) else {
            return self.fail_request(from, &msg);
        };
        if self.blackholed.contains(&(from, to)) || (self.drop_prob > 0.0 && self.rng.gen_bool(self.drop_prob)) {
            return;
        }
        if self.disconnected.contains(&(from, to)) || !self.nodes[to].alive() {
            return self.fail_request(from, &msg);
        }
        if self.trace.is_some() && !is_chatter(&msg) {
            let (a, b, m) = (self.nodes[from].node.name.clone(), self.nodes[to].node.name.clone(), summarize(&msg));
            self.tr(|| format!("SEND {a} -> {b}: {m}"));
        }
        if self.dup_prob > 0.0 && self.rng.gen_bool(self.dup_prob) {
            let at = self.now + self.delay();
            self.schedule(at, Event::Deliver { from, to, msg: msg.clone() });
        }
        let at = self.now + self.delay();
        self.schedule(at, Event::Deliver { from, to, msg });
    }

    /// Queue a batch of outputs and execute as many as the disk allows.
    fn process(&mut self, idx: usize, outputs: Vec<Output>) {
        if !self.nodes[idx].alive() {
            return;
        }
        self.nodes[idx].queue.extend(outputs);
        self.drain(idx);
    }

    fn drain(&mut self, idx: usize) {
        while self.nodes[idx].alive() && !self.nodes[idx].disk_busy {
            let Some(o) = self.nodes[idx].queue.pop_front() else { return };
            if self.crash_mid_output_prob > 0.0 && self.rng.gen_bool(self.crash_mid_output_prob) {
                self.crash(idx);
                return;
            }
            self.stats.invariant_checks += 1;
            match o {
                Output::Persist(u) => {
                    if self.max_disk_delay > 0 {
                        self.nodes[idx].disk_busy = true;
                        let at = self.now + self.rng.gen_range(0..=self.max_disk_delay);
                        let incarnation = self.nodes[idx].incarnation;
                        self.schedule(at, Event::PersistDone { node: idx, incarnation, update: u });
                        return;
                    }
                    self.persist_done(idx, u);
                }
                Output::Send { to, msg } => self.route(idx, &to, msg),
                Output::ApplyState(s) => self.on_applied(idx, &s),
                Output::ModeChanged { mode, term, leader } => {
                    let (name, l) = (self.nodes[idx].node.name.clone(), leader.map(|n| n.name));
                    self.tr(|| format!("MODE {name}: {mode:?} term {term} leader {l:?}"));
                    if mode == Mode::Leader {
                        self.stats.elections += 1;
                        let id = self.nodes[idx].node.id.clone();
                        let prev = self.leaders_by_term.entry(term).or_insert_with(|| id.clone());
                        assert_eq!(*prev, id, "I1: two leaders in term {term}");
                    }
                }
                Output::PublicationCompleted { .. } => {}
                Output::TaskCompleted { task_id, result } => {
                    if let Some(v) = self.pending_log_tasks.remove(&(idx, task_id)) {
                        if result.is_ok() {
                            self.acked_log_values.insert(v);
                        }
                    }
                    self.task_results.insert((idx, task_id), result);
                }
            }
        }
    }

    /// An fsync completed: the update is now durable.
    fn persist_done(&mut self, idx: usize, u: PersistUpdate) {
        let n = &mut self.nodes[idx];
        assert!(u.current_term >= n.storage.current_term, "I7: term went backwards on {}", n.node.name);
        let prev_uuid = n.storage.last_accepted_state.state_uuid.clone();
        u.apply_to(&mut n.storage);
        if let Some(s) = u.last_accepted_state {
            let id = self.nodes[idx].node.id.clone();
            // The publishing master persists its own publication on top of
            // its previous accepted state: that is the state's parent.
            if s.master_node_id.as_deref() == Some(id.as_str()) && prev_uuid != s.state_uuid && s.coordination.term > 0 {
                self.parents.entry(s.state_uuid.clone()).or_insert((prev_uuid, s.version));
            }
            let (name, t, v) = (self.nodes[idx].node.name.clone(), s.coordination.term, s.version);
            self.tr(|| format!("PERSIST {name}: term {} state ({t},{v})", u.current_term));
            self.on_persisted_state(&id, &s);
        } else {
            let name = self.nodes[idx].node.name.clone();
            self.tr(|| format!("PERSIST {name}: term {}", u.current_term));
        }
    }

    fn check_config_committed(&self, s: &ClusterState, what: &str) {
        if s.coordination.term == 0 {
            return;
        }
        let k = cfg_key(&s.coordination.last_committed_config);
        assert!(
            self.committed_configs.contains(&k) || self.bootstrap_configs.contains(&k),
            "I6: {what} state (term {}, version {}) has last_committed_config {k:?} which was never committed",
            s.coordination.term,
            s.version
        );
    }

    fn on_persisted_state(&mut self, node_id: &str, s: &ClusterState) {
        if s.coordination.term == 0 {
            if !s.coordination.last_accepted_config.is_empty() {
                self.bootstrap_configs.insert(cfg_key(&s.coordination.last_accepted_config));
            }
            return;
        }
        self.check_config_committed(s, "persisted");
        let key = (s.coordination.term, s.version);
        let entry = self.accepted.entry(key).or_insert_with(|| Accepted {
            uuid: s.state_uuid.clone(),
            acceptors: BTreeSet::new(),
            committed_config: s.coordination.last_committed_config.clone(),
            accepted_config: s.coordination.last_accepted_config.clone(),
            log: sim_log(s),
            committed: false,
        });
        assert_eq!(entry.uuid, s.state_uuid, "I2: two different states accepted for (term, version) {key:?}");
        entry.acceptors.insert(node_id.to_string());
        if !entry.committed && entry.committed_config.has_quorum(entry.acceptors.iter()) && entry.accepted_config.has_quorum(entry.acceptors.iter()) {
            entry.committed = true;
            let (uuid, log, cfg) = (entry.uuid.clone(), entry.log.clone(), cfg_key(&entry.accepted_config));
            self.committed_configs.insert(cfg);
            self.on_committed(key.0, key.1, uuid, log, s.master_node_id.clone());
        }
    }

    fn on_committed(&mut self, term: u64, version: u64, uuid: String, log: Vec<u64>, master: Option<String>) {
        self.stats.commits += 1;
        self.tr(|| format!("COMMITTED ({term},{version}) {uuid}"));
        let m = self.committed_master_by_term.entry(term).or_insert_with(|| master.clone());
        assert_eq!(*m, master, "I8: states committed in term {term} by two different masters");
        if let Some((t, u, _)) = self.committed_by_version.get(&version) {
            assert!(*u == uuid && *t == term, "I3: two different states committed at version {version}: term {t} vs {term}");
            return;
        }
        if let Some((v, (_, pu, prev))) = self.committed_by_version.range(..version).next_back() {
            assert!(is_prefix(prev, &log), "I3: committed version {version} (term {term}) lost data of committed version {v}: {prev:?} vs {log:?}");
            assert!(self.descends(&uuid, pu, *v), "I8: committed ({term},{version}) does not descend from committed version {v}");
        }
        if let Some((v, (_, nu, next))) = self.committed_by_version.range(version + 1..).next() {
            assert!(is_prefix(&log, next), "I3: committed version {v} does not extend committed version {version}");
            assert!(self.descends(nu, &uuid, version), "I8: committed version {v} does not descend from committed ({term},{version})");
        }
        self.committed_by_version.insert(version, (term, uuid, log));
    }

    /// Whether `uuid` has `ancestor` (at `ancestor_version`) on its parent
    /// chain.
    fn descends(&self, uuid: &str, ancestor: &str, ancestor_version: u64) -> bool {
        let mut cur = uuid.to_string();
        loop {
            if cur == ancestor {
                return true;
            }
            match self.parents.get(&cur) {
                Some((p, v)) if *v > ancestor_version => cur = p.clone(),
                _ => return false,
            }
        }
    }

    fn on_applied(&mut self, idx: usize, s: &ClusterState) {
        let n = &mut self.nodes[idx];
        assert!(s.version >= n.applied_version, "I5: {} applied version {} after {}", n.node.name, s.version, n.applied_version);
        n.applied_version = s.version;
        if s.version == 0 {
            return;
        }
        if let Some(m) = &s.master_node_id {
            assert!(s.nodes.contains_key(m), "applied state's master is not in its nodes");
        }
        match self.committed_by_version.get(&s.version) {
            Some((_, uuid, _)) => assert_eq!(*uuid, s.state_uuid, "I4: {} applied an uncommitted state at version {}", self.nodes[idx].node.name, s.version),
            None => panic!("I4: {} applied version {} which was never committed", self.nodes[idx].node.name, s.version),
        }
        // I9: the applied sequence is an ordered subsequence of the global
        // committed history (consecutive applied states are linked by
        // descent, so no node ever applies a fork).
        let n = &mut self.nodes[idx];
        if n.applied_seq.last().is_none_or(|(_, u)| *u != s.state_uuid) {
            let prev = n.applied_seq.last().cloned();
            n.applied_seq.push((s.version, s.state_uuid.clone()));
            if let Some((pv, pu)) = prev {
                assert!(self.descends(&s.state_uuid, &pu, pv), "I9: applied ({},{}) does not descend from previously applied version {pv}", s.coordination.term, s.version);
            }
        }
        let (name, v, m) = (self.nodes[idx].node.name.clone(), s.version, s.master_node_id.clone());
        self.tr(|| format!("APPLY {name}: version {v} master {m:?}"));
    }

    // --------------------------------------------------------- driving --

    /// Execute the next event or timer. Returns false when idle.
    pub fn step(&mut self) -> bool {
        self.stats.steps += 1;
        let ev_time = self.events.keys().next().map(|k| k.0);
        let mut tick: Option<(u64, usize)> = None;
        for (i, n) in self.nodes.iter().enumerate() {
            if let Some(d) = n.coord.as_ref().and_then(|c| c.next_deadline()) {
                let d = d.saturating_sub(n.clock_offset).max(self.now);
                if tick.is_none_or(|(t, _)| d < t) {
                    tick = Some((d, i));
                }
            }
        }
        match (ev_time, tick) {
            (None, None) => return false,
            (Some(e), t) if t.is_none_or(|(tt, _)| e <= tt) => {
                let (_, ev) = self.events.pop_first().expect("present");
                self.now = self.now.max(e);
                match ev {
                    Event::Deliver { from, to, msg } => {
                        if self.blackholed.contains(&(from, to)) {
                            return true;
                        }
                        if !self.nodes[to].alive() || self.disconnected.contains(&(from, to)) {
                            self.fail_request(from, &msg);
                            return true;
                        }
                        let from_addr = self.nodes[from].node.transport_address.clone();
                        if self.trace.is_some() && !is_chatter(&msg) {
                            let (a, b, m) = (self.nodes[from].node.name.clone(), self.nodes[to].node.name.clone(), summarize(&msg));
                            self.tr(|| format!("RECV {a} -> {b}: {m}"));
                        }
                        let now = self.local_now(to);
                        let out = self.nodes[to].coord.as_mut().expect("alive").handle_message(now, &from_addr, msg);
                        self.process(to, out);
                    }
                    Event::SendFailure { node, incarnation, request_id } => {
                        if self.nodes[node].incarnation == incarnation && self.nodes[node].alive() {
                            let now = self.local_now(node);
                            let out = self.nodes[node].coord.as_mut().expect("alive").handle_send_failure(now, request_id, "connection refused");
                            self.process(node, out);
                        }
                    }
                    Event::PersistDone { node, incarnation, update } => {
                        if self.nodes[node].incarnation == incarnation && self.nodes[node].alive() {
                            self.nodes[node].disk_busy = false;
                            self.persist_done(node, update);
                            self.drain(node);
                        }
                    }
                }
            }
            _ => {
                let (t, i) = tick.expect("a timer is due when no event is");
                self.now = self.now.max(t);
                let now = self.local_now(i);
                let out = self.nodes[i].coord.as_mut().expect("alive").tick(now);
                let after = self.nodes[i].coord.as_ref().and_then(|c| c.next_deadline());
                assert!(after.is_none_or(|d| d > now) || !out.is_empty() || !self.nodes[i].alive(), "tick at {now} made no progress on {}", self.nodes[i].node.name);
                self.process(i, out);
            }
        }
        true
    }

    pub fn run_for(&mut self, ms: u64) {
        let end = self.now + ms;
        while self.next_time().is_some_and(|t| t <= end) {
            self.step();
        }
        self.now = self.now.max(end);
    }

    fn next_time(&self) -> Option<u64> {
        let e = self.events.keys().next().map(|k| k.0);
        let t = self.nodes.iter().filter_map(|n| n.coord.as_ref().and_then(|c| c.next_deadline()).map(|d| d.saturating_sub(n.clock_offset))).min();
        match (e, t) {
            (Some(a), Some(b)) => Some(a.min(b).max(self.now)),
            (a, b) => a.or(b).map(|x| x.max(self.now)),
        }
    }

    /// Run until `pred` holds (checked every 50ms of virtual time) or
    /// `max_ms` elapse. Returns the elapsed virtual millis on success.
    pub fn run_until(&mut self, max_ms: u64, pred: impl Fn(&Cluster) -> bool) -> Option<u64> {
        let start = self.now;
        loop {
            if pred(self) {
                return Some(self.now - start);
            }
            if self.now >= start + max_ms {
                return None;
            }
            self.run_for(50);
        }
    }

    // ------------------------------------------------------- queries --

    /// The leader if `members` form a stable cluster: exactly one leader,
    /// not publishing, whose applied state contains exactly `members`
    /// (current incarnations), and every member follows it and has applied
    /// the same version.
    pub fn stable_among(&self, members: &[usize]) -> Option<usize> {
        let leaders: Vec<usize> = members.iter().copied().filter(|i| self.nodes[*i].alive() && self.nodes[*i].c().mode() == Mode::Leader).collect();
        let [l] = leaders[..] else { return None };
        let lc = self.nodes[l].c();
        if lc.is_publishing() {
            return None;
        }
        let ls = lc.applied_state();
        if ls.master_node_id.as_deref() != Some(lc.local_node().id.as_str()) {
            return None;
        }
        let expected: BTreeMap<String, String> = members.iter().map(|i| (self.nodes[*i].node.id.clone(), self.nodes[*i].node.ephemeral_id.clone())).collect();
        let actual: BTreeMap<String, String> = ls.nodes.values().map(|n| (n.id.clone(), n.ephemeral_id.clone())).collect();
        if expected != actual {
            return None;
        }
        for &i in members {
            let n = &self.nodes[i];
            if !n.alive() {
                return None;
            }
            let c = n.c();
            if i != l && (c.mode() != Mode::Follower || c.leader().map(|x| &x.id) != Some(&lc.local_node().id)) {
                return None;
            }
            if c.applied_state().version != ls.version || c.applied_state().master_node_id != ls.master_node_id {
                return None;
            }
        }
        Some(l)
    }

    pub fn stable_leader(&self) -> Option<usize> {
        self.stable_among(&self.alive())
    }

    /// Any alive node currently in LEADER mode (prefers the highest term).
    pub fn any_leader(&self) -> Option<usize> {
        self.alive().into_iter().filter(|i| self.nodes[*i].c().mode() == Mode::Leader).max_by_key(|i| self.nodes[*i].c().current_term())
    }

    /// Run a leader-only operation on the current leader.
    pub fn leader_op(&mut self, f: impl FnOnce(&mut Coordinator, u64) -> Result<Submitted, CoordinationError>) -> Option<(usize, u64)> {
        self.leader_op_with(f, |_, _, _| {})
    }

    /// Like `leader_op`, calling `before` with (leader, task id) before the
    /// submission's outputs are executed (they may crash the node).
    fn leader_op_with(&mut self, f: impl FnOnce(&mut Coordinator, u64) -> Result<Submitted, CoordinationError>, before: impl FnOnce(&mut Cluster, usize, u64)) -> Option<(usize, u64)> {
        let l = self.any_leader()?;
        let now = self.local_now(l);
        let s = f(self.nodes[l].coord.as_mut().expect("alive"), now).ok()?;
        before(self, l, s.id);
        self.process(l, s.outputs);
        Some((l, s.id))
    }

    /// Submit a task appending a fresh value to the replicated log.
    pub fn submit_log_task(&mut self) -> bool {
        let v = self.next_log_value;
        self.next_log_value += 1;
        self.leader_op_with(
            move |c, now| {
                c.submit_task(
                    now,
                    Box::new(move |s: &ClusterState| {
                        let mut s = s.clone();
                        let mut log = sim_log(&s);
                        log.push(v);
                        s.metadata.customs.insert(LOG_KEY.into(), Value::from(log));
                        Ok(s)
                    }),
                )
            },
            move |c, l, id| {
                c.pending_log_tasks.insert((l, id), v);
            },
        )
        .is_some()
    }

    pub fn describe_pending(&self) -> String {
        let mut out = format!("pending (node, task) -> value: {:?}; ", self.pending_log_tasks);
        for (i, n) in self.nodes.iter().enumerate() {
            if let Some(c) = &n.coord {
                let a = c.applied_state();
                out += &format!(
                    "[{i} {:?} term {} leader {:?} publishing {} queue {} disk_busy {} accepted ({},{}) cfg {:?}/{:?} excl {:?} applied v{} nodes {:?}] ",
                    c.mode(),
                    c.current_term(),
                    c.leader().map(|l| l.name.clone()),
                    c.is_publishing(),
                    n.queue.len(),
                    n.disk_busy,
                    c.last_accepted_state().coordination.term,
                    c.last_accepted_state().version,
                    c.last_accepted_state().coordination.last_committed_config.0,
                    c.last_accepted_state().coordination.last_accepted_config.0,
                    c.last_accepted_state().coordination.voting_config_exclusions.iter().map(|e| e.node_id.clone()).collect::<Vec<_>>(),
                    a.version,
                    a.nodes.keys().collect::<Vec<_>>()
                );
            }
        }
        out
    }

    pub fn pending_log_tasks_len(&self) -> usize {
        self.pending_log_tasks.len()
    }

}

/// Trace ring-buffer size (`COORD_TRACE_LINES`, default 600).
fn trace_lines() -> usize {
    std::env::var("COORD_TRACE_LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(600)
}

/// Periodic fault-detection and discovery traffic, kept out of traces.
fn is_chatter(m: &Message) -> bool {
    std::env::var("COORD_TRACE_CHATTER").is_err() && matches!(
        m,
        Message::LeaderCheck { .. } | Message::LeaderCheckResponse { result: Ok(()), .. } | Message::FollowerCheck { .. } | Message::FollowerCheckResponse { result: Ok(()), .. } | Message::PeersRequest { .. } | Message::PeersResponse { .. }
    )
}

/// One-line description of a message for traces.
pub fn summarize(m: &Message) -> String {
    match m {
        Message::PeersRequest { request_id, .. } => format!("PeersRequest#{request_id}"),
        Message::PeersResponse { request_id, result } => format!("PeersResponse#{request_id} {:?}", result.as_ref().map(|r| (r.master_node.as_ref().map(|n| n.name.clone()), r.term))),
        Message::PreVoteRequest { request_id, current_term, .. } => format!("PreVoteRequest#{request_id} term {current_term}"),
        Message::PreVoteResponse { request_id, result } => format!("PreVoteResponse#{request_id} {result:?}"),
        Message::StartJoin(sj) => format!("StartJoin for {} term {}", sj.source_node.name, sj.term),
        Message::JoinRequest { request_id, min_term, join, .. } => format!("JoinRequest#{request_id} min_term {min_term} join {join:?}"),
        Message::JoinResponse { request_id, result } => format!("JoinResponse#{request_id} {result:?}"),
        Message::PublishRequest { request_id, state } => format!("PublishRequest#{request_id} ({},{}) cfg {:?}", state.coordination.term, state.version, state.coordination.last_accepted_config.0),
        Message::PublishResponse { request_id, result } => format!("PublishResponse#{request_id} {:?}", result.as_ref().map(|r| (r.term, r.version, r.join.is_some()))),
        Message::ApplyCommit { request_id, commit } => format!("ApplyCommit#{request_id} ({},{})", commit.term, commit.version),
        Message::ApplyCommitResponse { request_id, result } => format!("ApplyCommitResponse#{request_id} {result:?}"),
        Message::LeaderCheck { request_id, .. } => format!("LeaderCheck#{request_id}"),
        Message::LeaderCheckResponse { request_id, result } => format!("LeaderCheckResponse#{request_id} {result:?}"),
        Message::FollowerCheck { request_id, term, .. } => format!("FollowerCheck#{request_id} term {term}"),
        Message::FollowerCheckResponse { request_id, result } => format!("FollowerCheckResponse#{request_id} {result:?}"),
    }
}

/// Totals over a seed family.
#[derive(Default, Debug, Clone)]
pub struct FamilyStats {
    pub seeds: u64,
    pub steps: u64,
    pub messages: u64,
    pub invariant_checks: u64,
    pub commits: u64,
    pub elections: u64,
}

/// Run `scenario(seed, trace)` for `count` seeds starting at `first`,
/// spread over all cores (each seed is an independent, deterministic
/// simulation). A panicking seed is re-run deterministically with tracing
/// on, and the seed plus the last events are printed before failing the
/// test.
pub fn run_seeds(name: &str, first: u64, count: u64, scenario: fn(u64, bool) -> Cluster) -> FamilyStats {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    let started = std::time::Instant::now();
    let next = AtomicU64::new(first);
    let total = Mutex::new(FamilyStats::default());
    let failure: Mutex<Option<(u64, String, Vec<String>, bool)>> = Mutex::new(None);
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(count.max(1) as usize);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    if failure.lock().unwrap().is_some() {
                        return;
                    }
                    let seed = next.fetch_add(1, Ordering::SeqCst);
                    if seed >= first + count {
                        return;
                    }
                    match std::panic::catch_unwind(|| scenario(seed, false)) {
                        Ok(c) => {
                            let mut t = total.lock().unwrap();
                            t.seeds += 1;
                            t.steps += c.stats.steps;
                            t.messages += c.stats.messages;
                            t.invariant_checks += c.stats.invariant_checks;
                            t.commits += c.stats.commits;
                            t.elections += c.stats.elections;
                        }
                        Err(e) => {
                            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                            // Deterministic replay with tracing.
                            TRACE_SINK.with(|t| t.borrow_mut().clear());
                            let replay = std::panic::catch_unwind(|| scenario(seed, true));
                            let lines = TRACE_SINK.with(|t| t.borrow().clone());
                            let mut f = failure.lock().unwrap();
                            if f.as_ref().is_none_or(|(s, ..)| seed < *s) {
                                *f = Some((seed, msg, lines, replay.is_err()));
                            }
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some((seed, msg, lines, reproduced)) = failure.into_inner().unwrap() {
        eprintln!("==== {name}: seed {seed} FAILED: {msg}\n---- last {} events of the replay ({}):", lines.len(), if reproduced { "reproduced" } else { "NOT reproduced" });
        for l in &lines {
            eprintln!("{l}");
        }
        panic!("{name}: seed {seed} failed: {msg} (re-run with COORD_FIRST={seed} COORD_SEEDS=1 to reproduce)");
    }
    let total = total.into_inner().unwrap();
    eprintln!(
        "{name}: {} seeds, {} steps, {} messages, {} invariant checks, {} commits, {} elections in {:.1}s ({workers} threads)",
        total.seeds,
        total.steps,
        total.messages,
        total.invariant_checks,
        total.commits,
        total.elections,
        started.elapsed().as_secs_f64()
    );
    total
}

/// First seed of a family: `COORD_FIRST` env var or 0.
pub fn first_seed() -> u64 {
    std::env::var("COORD_FIRST").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

thread_local! {
    /// Trace of the last traced cluster (filled when a traced cluster is dropped or panics).
    pub static TRACE_SINK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for Cluster {
    fn drop(&mut self) {
        if let Some(t) = &self.trace {
            let lines: Vec<String> = t.iter().cloned().collect();
            TRACE_SINK.with(|s| *s.borrow_mut() = lines);
        }
    }
}

/// Number of seeds per family: `COORD_SEEDS` env var or the default.
pub fn seed_count(default: u64) -> u64 {
    std::env::var("COORD_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
