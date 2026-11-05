//! The liveness layer of coordination: ES's `Coordinator` together with
//! `PeerFinder`, `ClusterBootstrapService`, `PreVoteCollector`,
//! `ElectionSchedulerFactory`, `JoinHelper`, `Publication`,
//! `LeaderChecker`, `FollowersChecker` and `LagDetector`, as one
//! single-threaded, sans-IO state machine.
//!
//! See the parent module docs for the protocol and the driver contract.
//! Method names follow ES (`become_candidate`, `ensure_term_at_least`,
//! `join_leader_in_term`, `update_max_term_seen`, ...).

use super::coordination_state::{ApplyCommit, CoordinationState, Join, PersistedState, PublishResponse, StartJoin};
use super::messages::{ApplyCommitAck, Message, PeersResponse, PreVoteResponse, PublishWithJoinResponse};
use super::reconfigurator::reconfigure;
use super::{BOOTSTRAP_PLACEHOLDER_PREFIX, CoordinationSettings, NoMasterBlock};
use crate::state::{ClusterState, DiscoveryNode, UNKNOWN_UUID, VotingConfigExclusion, VotingConfiguration};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// ES's `Coordinator.Mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Mode {
    Candidate,
    Leader,
    Follower,
}

/// Durable-write instruction. `last_accepted_state` is `None` when only the
/// term changed.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistUpdate {
    pub current_term: u64,
    pub last_accepted_state: Option<Box<ClusterState>>,
}

impl PersistUpdate {
    /// Apply to an in-memory copy of the persisted state.
    pub fn apply_to(&self, p: &mut PersistedState) {
        p.current_term = self.current_term;
        if let Some(s) = &self.last_accepted_state {
            p.last_accepted_state = (**s).clone();
        }
    }
}

/// Everything the driver must do, in order. See the module docs for the
/// persistence-ordering contract.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// Durably write before acting on any later output.
    Persist(PersistUpdate),
    /// Send `msg` to the node listening on transport address `to`. If a
    /// request (`msg.request_id()` is `Some`) cannot be delivered, report
    /// it with `handle_send_failure`.
    Send { to: String, msg: Message },
    /// Make this the node's applied cluster state (`ClusterApplierService`).
    ApplyState(Box<ClusterState>),
    /// Mode or known leader changed.
    ModeChanged { mode: Mode, term: u64, leader: Option<DiscoveryNode> },
    /// A publication finished: committed and applied on the master, or
    /// failed (the master stepped down).
    PublicationCompleted { publication_id: u64, version: u64, result: Result<(), String> },
    /// A submitted task finished: `Ok(version)` of the committed state that
    /// includes it, or the error (task rejected, publication failed, master
    /// stepped down).
    TaskCompleted { task_id: u64, result: Result<u64, String> },
}

/// A master-service task: compute a new state from the current one. The
/// coordinator overwrites `version`, `state_uuid`, `master_node_id`,
/// `coordination` and the master/recovery blocks.
pub type StateTask = Box<dyn FnOnce(&ClusterState) -> Result<ClusterState, String> + Send>;

/// Hook through which the driver adjusts the state when nodes join or
/// leave (e.g. fail shards on departed nodes and reroute). Called on the
/// leader with the node changes already applied to `state.nodes`.
pub trait StateUpdater: Send {
    fn nodes_changed(&mut self, state: &ClusterState, joined: &[DiscoveryNode], left: &[(String, String)]) -> ClusterState;
    /// The segment format `node` is trusted to read when the leader
    /// negotiates the cluster's segment format floor
    /// ([`crate::compatibility::negotiated_segment_format`]). The driver may
    /// demand more than the advertisement (a live process proof); `local`
    /// is this leader's own node.
    fn segment_read_max(&self, _cluster_name: &str, _local: &DiscoveryNode, node: &DiscoveryNode) -> u32 {
        crate::compatibility::advertised_read_max(node)
    }
}

/// A `StateUpdater` that changes nothing beyond membership.
pub struct NoopStateUpdater;

impl StateUpdater for NoopStateUpdater {
    fn nodes_changed(&mut self, state: &ClusterState, _: &[DiscoveryNode], _: &[(String, String)]) -> ClusterState {
        state.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinationError {
    /// ES's `NotMasterException`.
    NotMaster(String),
    Invalid(String),
}

impl std::fmt::Display for CoordinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinationError::NotMaster(s) | CoordinationError::Invalid(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for CoordinationError {}

impl CoordinationError {
    /// `node`'s refusal of a task it never queued (it is not the leader).
    pub fn not_master(node: &str) -> Self {
        CoordinationError::NotMaster(format!("node [{node}] is not the elected master"))
    }

    /// Whether a task error is that refusal: nothing was computed or
    /// published, so the caller may retry on the next master.
    pub fn is_not_master(reason: &str) -> bool {
        reason.starts_with("node [") && reason.ends_with("] is not the elected master")
    }
}

/// Result of a leader-only submission: the task id and the outputs the
/// submission produced (it may start a publication immediately).
#[derive(Debug)]
pub struct Submitted {
    pub id: u64,
    pub outputs: Vec<Output>,
}

// ------------------------------------------------------------ internals --

type JoinKey = (String, u64, Option<Join>);

#[derive(Clone, Debug)]
enum Pending {
    Peers,
    PreVote { round: u64, node_id: String },
    Join { key: JoinKey },
    Publish { publication: u64, node_id: String },
    Commit { publication: u64, node_id: String },
    LeaderCheck,
    FollowerCheck { node_id: String },
}

#[derive(Clone, Debug)]
struct PendingRequest {
    to: String,
    kind: Pending,
    deadline: Option<u64>,
}

#[derive(Clone, Debug, Default)]
struct Peer {
    node: Option<DiscoveryNode>,
    in_flight: Option<u64>,
}

#[derive(Clone, Debug)]
struct ElectionScheduler {
    attempt: u64,
    next_at: u64,
}

#[derive(Clone, Debug)]
struct PreVoteRound {
    id: u64,
    votes: BTreeSet<String>,
    started: bool,
}

/// A join request held until it can be answered (after the publication
/// that adds the node). `reply` is `None` for the local node's own join.
#[derive(Clone, Debug)]
struct IncomingJoin {
    node: DiscoveryNode,
    reply: Option<(String, u64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// PublishRequest sent, awaiting acceptance.
    Publishing,
    /// Accepted, awaiting commit.
    Accepted,
    /// ApplyCommit sent, awaiting the apply ack.
    Committing,
    Done,
    Failed,
}

impl Phase {
    fn active(self) -> bool {
        matches!(self, Phase::Publishing | Phase::Accepted | Phase::Committing)
    }
}

struct Publication {
    id: u64,
    state: ClusterState,
    deadline: u64,
    targets: BTreeMap<String, Phase>,
    commit: Option<ApplyCommit>,
    joins: Vec<IncomingJoin>,
    tasks: Vec<u64>,
}

struct QueuedTask {
    id: u64,
    task: StateTask,
}

#[derive(Clone, Debug)]
struct Checker {
    node: DiscoveryNode,
    /// Time of the last successful check.
    last_ok: Option<u64>,
    failures: u32,
    in_flight: Option<u64>,
    next_at: u64,
}

pub struct Coordinator {
    settings: CoordinationSettings,
    local: DiscoveryNode,
    rng: StdRng,
    state: CoordinationState,
    mode: Mode,
    /// Known leader (FOLLOWER) or the local node (LEADER).
    leader: Option<DiscoveryNode>,
    last_join: Option<Join>,
    max_term_seen: u64,
    applied: ClusterState,
    outputs: Vec<Output>,
    next_request_id: u64,
    requests: BTreeMap<u64, PendingRequest>,
    // PeerFinder
    peer_finder_active: bool,
    peers: BTreeMap<String, Peer>,
    next_peer_wakeup: u64,
    // ClusterBootstrapService
    bootstrapping_permitted: bool,
    // Elections
    election: Option<ElectionScheduler>,
    prevote: Option<PreVoteRound>,
    prevote_seq: u64,
    // JoinHelper
    pending_joins_out: BTreeSet<JoinKey>,
    incoming_joins: Vec<IncomingJoin>,
    // Leader
    publication: Option<Publication>,
    next_publication_id: u64,
    pending_node_left: BTreeMap<String, String>,
    tasks: VecDeque<QueuedTask>,
    next_task_id: u64,
    follower_checks: BTreeMap<String, Checker>,
    applied_versions: BTreeMap<String, u64>,
    lag_checks: Vec<(String, u64, u64)>,
    /// When this node last became a candidate.
    candidate_since: u64,
    /// Time of the last re-election started to collect missing votes, and
    /// how many have been tried since the votes were last complete.
    last_vote_reelection: Option<u64>,
    vote_reelections: u32,
    // Follower
    leader_checker: Option<Checker>,
    updater: Box<dyn StateUpdater>,
}

impl Coordinator {
    /// Create a coordinator from the node's persisted state. `seed` seeds
    /// the RNG used for election jitter and generated ids. Call
    /// [`start`](Self::start) next.
    pub fn new(settings: CoordinationSettings, local: DiscoveryNode, persisted: PersistedState, seed: u64, updater: Box<dyn StateUpdater>) -> Coordinator {
        let mut rng = StdRng::seed_from_u64(seed);
        let state = CoordinationState::new(&local.id, persisted);
        let max_term_seen = state.current_term();
        // ES's initial applied state: empty, only the local node, with the
        // state-not-recovered and no-master blocks.
        let mut applied = ClusterState::empty(&settings.cluster_name);
        applied.state_uuid = UNKNOWN_UUID.to_string();
        applied.nodes.insert(local.id.clone(), local.clone());
        applied.blocks.global.insert(settings.no_master_block.block());
        let next_request_id = rng.next_u64() >> 16;
        Coordinator {
            settings,
            local,
            rng,
            state,
            mode: Mode::Candidate,
            leader: None,
            last_join: None,
            max_term_seen,
            applied,
            outputs: Vec::new(),
            next_request_id,
            requests: BTreeMap::new(),
            peer_finder_active: false,
            peers: BTreeMap::new(),
            next_peer_wakeup: 0,
            bootstrapping_permitted: true,
            election: None,
            prevote: None,
            prevote_seq: 0,
            pending_joins_out: BTreeSet::new(),
            incoming_joins: Vec::new(),
            publication: None,
            next_publication_id: 1,
            pending_node_left: BTreeMap::new(),
            tasks: VecDeque::new(),
            next_task_id: 1,
            follower_checks: BTreeMap::new(),
            applied_versions: BTreeMap::new(),
            lag_checks: Vec::new(),
            leader_checker: None,
            last_vote_reelection: None,
            vote_reelections: 0,
            candidate_since: 0,
            updater,
        }
    }

    // ============================================================ API ==

    /// Start as a CANDIDATE: apply the initial state, start discovery and
    /// (if configured) bootstrapping.
    pub fn start(&mut self, now: u64) -> Vec<Output> {
        let s = self.applied.clone();
        self.emit(Output::ApplyState(Box::new(s)));
        self.emit_mode();
        self.activate_candidate(now);
        self.finish()
    }

    /// An inbound message from the node at transport address `from`.
    pub fn handle_message(&mut self, now: u64, from: &str, msg: Message) -> Vec<Output> {
        match msg {
            Message::PeersRequest { request_id, cluster_name, source, known_peers } => self.on_peers_request(now, from, request_id, &cluster_name, source, known_peers),
            Message::PreVoteRequest { request_id, source, current_term } => self.on_prevote_request(now, from, request_id, source, current_term),
            Message::StartJoin(sj) => self.on_start_join(now, sj),
            Message::JoinRequest { request_id, cluster_name, source, min_term, join, cluster_uuid, cluster_uuid_committed } => {
                self.on_join_request(now, from, request_id, &cluster_name, source, min_term, join, &cluster_uuid, cluster_uuid_committed)
            }
            Message::PublishRequest { request_id, state } => self.on_publish_request(now, from, request_id, *state),
            Message::ApplyCommit { request_id, commit } => self.on_apply_commit(now, from, request_id, commit),
            Message::LeaderCheck { request_id, source, target_ephemeral_id } => self.on_leader_check(from, request_id, source, &target_ephemeral_id),
            Message::FollowerCheck { request_id, source, term, target_ephemeral_id } => self.on_follower_check(now, from, request_id, source, term, &target_ephemeral_id),
            response => {
                let id = response.response_to().expect("non-request message is a response");
                // Discovery requests go to seed host *names* (Kubernetes headless
        // services, compose service names) but replies arrive from the
        // peer's IP, so peers responses are matched by request id alone.
        let matches = self.requests.get(&id).is_some_and(|p| p.to == from || matches!(p.kind, Pending::Peers));
                if matches {
                    let pending = self.requests.remove(&id).expect("checked");
                    self.on_response(now, id, pending, response);
                }
            }
        }
        self.finish()
    }

    /// The transport could not deliver request `request_id` (connection
    /// refused, node disconnected). Treated like ES's
    /// `ConnectTransportException`: fault detection fails immediately.
    pub fn handle_send_failure(&mut self, now: u64, request_id: u64, reason: &str) -> Vec<Output> {
        if let Some(p) = self.requests.remove(&request_id) {
            self.on_request_failed(now, request_id, p, format!("disconnected: {reason}"), true);
        }
        self.finish()
    }

    /// A request failed while the connection remained established, for
    /// example a response timeout or a remote handler error. Fault detection
    /// counts these failures against its configured retry budget.
    pub fn handle_request_failure(&mut self, now: u64, request_id: u64, reason: &str) -> Vec<Output> {
        if let Some(p) = self.requests.remove(&request_id) {
            self.on_request_failed(now, request_id, p, reason.to_string(), false);
        }
        self.finish()
    }

    /// Our transport connection to `address` closed (ES's
    /// `TransportConnectionListener.onNodeDisconnected`). Every request in
    /// flight to that address fails as disconnected, and fault detection
    /// reacts at once even between checks, as ES's `LeaderChecker` /
    /// `FollowersChecker` do: a follower whose leader disconnected becomes a
    /// candidate, a leader removes the disconnected follower. Electing a
    /// replacement still needs a quorum; no safety rule is involved.
    pub fn handle_disconnect(&mut self, now: u64, address: &str) -> Vec<Output> {
        let ids: Vec<u64> = self.requests.iter().filter(|(_, p)| p.to == address).map(|(id, _)| *id).collect();
        for id in ids {
            if let Some(p) = self.requests.remove(&id) {
                self.on_request_failed(now, id, p, "disconnected: connection closed".to_string(), true);
            }
        }
        if self.mode == Mode::Follower && self.leader_checker.as_ref().is_some_and(|c| c.node.transport_address == address) {
            self.become_candidate(now, "leader failed: disconnected");
        }
        let followers: Vec<String> = self.follower_checks.iter().filter(|(_, c)| c.node.transport_address == address).map(|(id, _)| id.clone()).collect();
        for node_id in followers {
            self.fail_node(now, &node_id, "disconnected");
        }
        self.finish()
    }

    /// Advance timers. Must be called at (or soon after) `next_deadline()`.
    pub fn tick(&mut self, now: u64) -> Vec<Output> {
        // Request timeouts.
        let expired: Vec<u64> = self.requests.iter().filter(|(_, p)| p.deadline.is_some_and(|d| d <= now)).map(|(id, _)| *id).collect();
        for id in expired {
            if let Some(p) = self.requests.remove(&id) {
                self.on_request_failed(now, id, p, "request timed out".into(), false);
            }
        }
        if self.peer_finder_active && now >= self.next_peer_wakeup {
            self.peers_wakeup(now);
        }
        if let Some(e) = &self.election {
            if now >= e.next_at {
                let grace = self.settings.election_duration_ms;
                self.schedule_next_election(now, grace);
                if self.mode == Mode::Candidate && self.may_stand_for_election(now) {
                    self.start_prevote(now);
                }
            }
        }
        self.leader_check_wakeup(now);
        self.follower_checks_wakeup(now);
        self.idle_vote_check(now);
        if self.publication.as_ref().is_some_and(|p| now >= p.deadline) {
            self.on_publication_timeout(now);
        }
        if self.lag_checks.iter().any(|l| l.2 <= now) {
            let (due, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.lag_checks).into_iter().partition(|l| l.2 <= now);
            self.lag_checks = keep;
            for (node_id, version, _) in due {
                let applied = self.applied_versions.get(&node_id).copied().unwrap_or(0);
                if applied < version && self.mode == Mode::Leader && self.master_state().nodes.contains_key(&node_id) {
                    self.fail_node(now, &node_id, "lagging");
                }
            }
        }
        self.finish()
    }

    /// The earliest time at which `tick` has work to do.
    pub fn next_deadline(&self) -> Option<u64> {
        let mut d: Option<u64> = None;
        let mut add = |t: u64| d = Some(d.map_or(t, |x: u64| x.min(t)));
        for p in self.requests.values() {
            if let Some(t) = p.deadline {
                add(t);
            }
        }
        if self.peer_finder_active {
            add(self.next_peer_wakeup);
        }
        if let Some(e) = &self.election {
            add(e.next_at);
        }
        if let Some(c) = &self.leader_checker {
            if c.in_flight.is_none() {
                add(c.next_at);
            }
        }
        for c in self.follower_checks.values() {
            if c.in_flight.is_none() {
                add(c.next_at);
            }
        }
        if let Some(p) = &self.publication {
            add(p.deadline);
        }
        for l in &self.lag_checks {
            add(l.2);
        }
        d
    }

    /// Queue a master-service task (leader only). Tasks are batched into
    /// the next publication; the result arrives as `TaskCompleted`.
    pub fn submit_task(&mut self, now: u64, task: StateTask) -> Result<Submitted, CoordinationError> {
        if self.mode != Mode::Leader {
            return Err(CoordinationError::not_master(&self.local.name));
        }
        let id = self.next_task_id;
        self.next_task_id += 1;
        self.tasks.push_back(QueuedTask { id, task });
        self.maybe_publish(now);
        Ok(Submitted { id, outputs: self.finish() })
    }

    /// Publish a state the driver computed from [`master_state`](Self::master_state)
    /// (leader only), leaving its `version`, `state_uuid` and `nodes`
    /// untouched: they identify the base state. If the master state changed
    /// before the task runs (another publication, a node joining or
    /// leaving) it fails with `TaskCompleted(Err("stale ..."))` and the
    /// driver recomputes. Version, uuid, master id and coordination
    /// metadata of the published state are set by the coordinator.
    pub fn publish(&mut self, now: u64, new_state: ClusterState) -> Result<Submitted, CoordinationError> {
        self.submit_task(
            now,
            Box::new(move |cur: &ClusterState| {
                if cur.state_uuid != new_state.state_uuid || cur.version != new_state.version || cur.nodes != new_state.nodes {
                    return Err("stale cluster state: the master state changed since this state was computed".into());
                }
                Ok(new_state)
            }),
        )
    }

    /// `POST /_cluster/voting_config_exclusions?node_ids=..|node_names=..`.
    /// Takes effect through a published state; the reconfiguration that
    /// removes the nodes from the voting configuration is part of the same
    /// publication when possible.
    pub fn add_voting_config_exclusions(&mut self, now: u64, node_ids: Vec<String>, node_names: Vec<String>) -> Result<Submitted, CoordinationError> {
        if node_ids.is_empty() == node_names.is_empty() {
            return Err(CoordinationError::Invalid("Please set node identifiers correctly. One and only one of [node_name], [node_names] and [node_ids] has to be set".into()));
        }
        let max = self.settings.max_voting_config_exclusions;
        self.submit_task(
            now,
            Box::new(move |cur: &ClusterState| {
                let mut s = cur.clone();
                let mut add: Vec<VotingConfigExclusion> = Vec::new();
                for id in &node_ids {
                    match cur.nodes.get(id) {
                        Some(n) if n.is_master_eligible() => add.push(VotingConfigExclusion { node_id: n.id.clone(), node_name: n.name.clone() }),
                        Some(_) => {}
                        None => add.push(VotingConfigExclusion { node_id: id.clone(), node_name: "_absent_".into() }),
                    }
                }
                for name in &node_names {
                    match cur.nodes.values().find(|n| &n.name == name) {
                        Some(n) if n.is_master_eligible() => add.push(VotingConfigExclusion { node_id: n.id.clone(), node_name: n.name.clone() }),
                        Some(_) => {}
                        None => add.push(VotingConfigExclusion { node_id: "_absent_".into(), node_name: name.clone() }),
                    }
                }
                for e in add {
                    if !s.coordination.voting_config_exclusions.contains(&e) {
                        s.coordination.voting_config_exclusions.push(e);
                    }
                }
                if s.coordination.voting_config_exclusions.len() > max {
                    return Err(format!(
                        "add voting config exclusions request for {:?} would add [{}] exclusions to the existing [{}] which would exceed the maximum of [{max}] set by [cluster.max_voting_config_exclusions]",
                        node_ids.iter().chain(node_names.iter()).collect::<Vec<_>>(),
                        s.coordination.voting_config_exclusions.len() - cur.coordination.voting_config_exclusions.len(),
                        cur.coordination.voting_config_exclusions.len()
                    ));
                }
                Ok(s)
            }),
        )
    }

    /// `DELETE /_cluster/voting_config_exclusions?wait_for_removal=..`.
    pub fn clear_voting_config_exclusions(&mut self, now: u64, wait_for_removal: bool) -> Result<Submitted, CoordinationError> {
        self.submit_task(
            now,
            Box::new(move |cur: &ClusterState| {
                if wait_for_removal {
                    let still: Vec<&String> = cur.coordination.voting_config_exclusions.iter().map(|e| &e.node_id).filter(|id| cur.nodes.contains_key(*id)).collect();
                    if !still.is_empty() {
                        return Err(format!("clear voting config exclusions failed: nodes {still:?} are still in the cluster (use wait_for_removal=false to clear anyway)"));
                    }
                }
                let mut s = cur.clone();
                s.coordination.voting_config_exclusions.clear();
                Ok(s)
            }),
        )
    }

    /// Replace the seed addresses (ES re-reads its seed hosts providers,
    /// e.g. the `file` provider's `unicast_hosts.txt`, on every discovery
    /// round). The next peer-finder round probes the new addresses.
    pub fn set_seed_hosts(&mut self, seed_hosts: Vec<String>) {
        self.settings.seed_hosts = seed_hosts;
    }

    pub fn seed_hosts(&self) -> &[String] {
        &self.settings.seed_hosts
    }

    // ------------------------------------------------------ accessors --

    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn current_term(&self) -> u64 {
        self.state.current_term()
    }
    pub fn local_node(&self) -> &DiscoveryNode {
        &self.local
    }
    /// The known elected master (`Some(local)` on the leader).
    pub fn leader(&self) -> Option<&DiscoveryNode> {
        self.leader.as_ref()
    }
    /// The state most recently handed out with `ApplyState`.
    pub fn applied_state(&self) -> &ClusterState {
        &self.applied
    }
    pub fn last_accepted_state(&self) -> &ClusterState {
        self.state.last_accepted_state()
    }
    pub fn coordination_state(&self) -> &CoordinationState {
        &self.state
    }
    /// The master's newest state (in-flight publication included): the
    /// base against which `publish` states must be computed.
    pub fn master_state(&self) -> &ClusterState {
        self.publication.as_ref().map(|p| &p.state).unwrap_or_else(|| self.state.last_accepted_state())
    }
    pub fn is_publishing(&self) -> bool {
        self.publication.is_some()
    }
    /// Master-eligible peers found by discovery (CANDIDATE mode).
    pub fn discovered_peers(&self) -> Vec<DiscoveryNode> {
        let mut by_id: BTreeMap<String, DiscoveryNode> = BTreeMap::new();
        for p in self.peers.values() {
            if let Some(n) = &p.node {
                if n.is_master_eligible() && n.id != self.local.id {
                    by_id.insert(n.id.clone(), n.clone());
                }
            }
        }
        by_id.into_values().collect()
    }
    /// Whether none of `node_ids` is in the last accepted voting
    /// configuration any more (what `POST _cluster/voting_config_exclusions`
    /// waits for).
    pub fn voting_config_excludes(&self, node_ids: &[String]) -> bool {
        let c = &self.state.last_accepted_state().coordination;
        node_ids.iter().all(|id| !c.last_accepted_config.0.contains(id) && !c.last_committed_config.0.contains(id))
    }

    // ======================================================= plumbing ==

    fn flush_persist(&mut self) {
        if let Some((term, state)) = self.state.take_persist() {
            self.outputs.push(Output::Persist(PersistUpdate { current_term: term, last_accepted_state: state.map(Box::new) }));
        }
    }

    /// Every output goes through here so that pending persistence is
    /// always emitted first.
    fn emit(&mut self, o: Output) {
        self.flush_persist();
        self.outputs.push(o);
    }

    fn finish(&mut self) -> Vec<Output> {
        self.flush_persist();
        std::mem::take(&mut self.outputs)
    }

    fn send(&mut self, to: &str, msg: Message) {
        self.emit(Output::Send { to: to.to_string(), msg });
    }

    fn send_request(&mut self, to: &str, kind: Pending, deadline: Option<u64>, build: impl FnOnce(u64) -> Message) -> u64 {
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let id = self.next_request_id;
        self.requests.insert(id, PendingRequest { to: to.to_string(), kind, deadline });
        let msg = build(id);
        self.send(to, msg);
        id
    }

    fn emit_mode(&mut self) {
        let o = Output::ModeChanged { mode: self.mode, term: self.state.current_term(), leader: self.leader.clone() };
        self.emit(o);
    }

    fn random_uuid(&mut self) -> String {
        use base64::Engine;
        let mut b = [0u8; 16];
        self.rng.fill_bytes(&mut b);
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
    }

    fn set_applied(&mut self, s: ClusterState) {
        if s == self.applied {
            return;
        }
        if s.version < self.applied.version {
            debug_assert!(false, "applied version went backwards: {} -> {}", self.applied.version, s.version);
            return;
        }
        self.applied = s.clone();
        self.emit(Output::ApplyState(Box::new(s)));
    }

    fn no_master_block(&self, s: &ClusterState) -> NoMasterBlock {
        s.metadata.setting("cluster.no_master_block").and_then(|v| v.as_str()).and_then(NoMasterBlock::parse).unwrap_or(self.settings.no_master_block)
    }

    /// ES's `clusterStateWithNoMasterBlock`.
    fn with_no_master_block(&self, s: &ClusterState) -> ClusterState {
        let mut s = s.clone();
        s.master_node_id = None;
        let block = self.no_master_block(&s).block();
        s.blocks.global.retain(|b| b.id != block.id);
        s.blocks.global.insert(block);
        s
    }

    // ===================================================== responses ==

    fn on_response(&mut self, now: u64, id: u64, pending: PendingRequest, msg: Message) {
        match (pending.kind, msg) {
            (Pending::Peers, Message::PeersResponse { result, .. }) => self.on_peers_response(now, &pending.to, id, result),
            (Pending::PreVote { round, node_id }, Message::PreVoteResponse { result, .. }) => {
                if let Ok(r) = result {
                    self.on_prevote_response(now, round, &node_id, r);
                }
            }
            (Pending::Join { key }, Message::JoinResponse { .. }) => {
                self.pending_joins_out.remove(&key);
            }
            (Pending::Publish { publication, node_id }, Message::PublishResponse { result, .. }) => self.on_publish_response(now, publication, &node_id, result),
            (Pending::Commit { publication, node_id }, Message::ApplyCommitResponse { result, .. }) => self.on_commit_response(now, publication, &node_id, result),
            (Pending::LeaderCheck, Message::LeaderCheckResponse { result, .. }) => self.on_leader_check_result(now, id, result.map_err(|e| (e, false))),
            (Pending::FollowerCheck { node_id }, Message::FollowerCheckResponse { result, .. }) => self.on_follower_check_result(now, &node_id, id, result.map_err(|e| (e, false))),
            _ => {}
        }
    }

    fn on_request_failed(&mut self, now: u64, id: u64, pending: PendingRequest, reason: String, disconnected: bool) {
        match pending.kind {
            Pending::Peers => self.on_peers_response(now, &pending.to, id, Err(reason)),
            Pending::PreVote { .. } => {}
            Pending::Join { key } => {
                self.pending_joins_out.remove(&key);
            }
            Pending::Publish { publication, node_id } => self.on_publish_response(now, publication, &node_id, Err(reason)),
            Pending::Commit { publication, node_id } => self.on_commit_response(now, publication, &node_id, Err(reason)),
            Pending::LeaderCheck => self.on_leader_check_result(now, id, Err((reason, disconnected))),
            Pending::FollowerCheck { node_id } => self.on_follower_check_result(now, &node_id, id, Err((reason, disconnected))),
        }
    }

    // ================================================ mode transitions ==

    /// ES's `becomeCandidate`.
    fn become_candidate(&mut self, now: u64, reason: &str) {
        if self.mode == Mode::Candidate {
            return;
        }
        tracing::debug!(node = %self.local.name, term = self.current_term(), reason, "becoming candidate");
        self.mode = Mode::Candidate;
        self.leader = None;
        let msg = format!("no longer master: {reason}");
        if let Some(p) = self.publication.take() {
            self.fail_publication_listeners(p, &msg);
        }
        self.fail_incoming_joins(&format!("join target became candidate: {reason}"));
        self.fail_queued_tasks(&msg);
        self.pending_node_left.clear();
        self.follower_checks.clear();
        self.lag_checks.clear();
        self.applied_versions.clear();
        self.leader_checker = None;
        self.emit_mode();
        self.activate_candidate(now);
    }

    fn activate_candidate(&mut self, now: u64) {
        self.candidate_since = now;
        self.peer_finder_active = true;
        self.peers.clear();
        self.next_peer_wakeup = now;
        if self.applied.master_node_id.is_some() {
            let s = self.with_no_master_block(&self.applied);
            self.set_applied(s);
        }
        self.peers_wakeup(now);
        self.on_found_peers_updated(now);
    }

    /// ES's `becomeLeader`: only from CANDIDATE, once the election is won.
    fn become_leader(&mut self, now: u64) {
        debug_assert_eq!(self.mode, Mode::Candidate);
        tracing::debug!(node = %self.local.name, term = self.current_term(), "elected as master");
        self.mode = Mode::Leader;
        self.leader = Some(self.local.clone());
        self.peer_finder_active = false;
        self.peers.clear();
        self.election = None;
        self.prevote = None;
        self.leader_checker = None;
        self.emit_mode();
        // The first publication in the term establishes leadership (ES's
        // "elected-as-master" task) and carries the accumulated joins.
        self.maybe_publish(now);
    }

    /// ES's `becomeFollower`.
    fn become_follower(&mut self, now: u64, leader: DiscoveryNode) {
        if self.mode == Mode::Leader {
            // Cannot happen (a leader moves to a higher term, and so to
            // CANDIDATE, first) but be defensive.
            self.become_candidate(now, "following another master");
        }
        let same_leader = self.mode == Mode::Follower && self.leader.as_ref() == Some(&leader);
        if self.mode == Mode::Candidate {
            self.fail_incoming_joins("join target is a follower");
            self.mode = Mode::Follower;
        }
        self.peer_finder_active = false;
        self.peers.clear();
        self.election = None;
        self.prevote = None;
        self.follower_checks.clear();
        self.lag_checks.clear();
        if !same_leader {
            self.leader = Some(leader.clone());
            self.leader_checker = Some(Checker { node: leader, last_ok: None, failures: 0, in_flight: None, next_at: now + self.settings.leader_check_interval_ms });
            self.emit_mode();
        }
    }

    fn fail_incoming_joins(&mut self, reason: &str) {
        for j in std::mem::take(&mut self.incoming_joins) {
            self.reply_join(j.reply, Err(reason.to_string()));
        }
    }

    fn fail_queued_tasks(&mut self, reason: &str) {
        for t in std::mem::take(&mut self.tasks) {
            self.emit(Output::TaskCompleted { task_id: t.id, result: Err(reason.to_string()) });
        }
    }

    fn reply_join(&mut self, reply: Option<(String, u64)>, result: Result<(), String>) {
        if let Some((to, request_id)) = reply {
            self.send(&to, Message::JoinResponse { request_id, result });
        }
    }

    // ================================================ terms and joins ==

    /// ES's `joinLeaderInTerm`: vote for `sj.source_node` in `sj.term`.
    fn join_leader_in_term(&mut self, now: u64, sj: &StartJoin) -> Option<Join> {
        let join = self.state.handle_start_join(sj).ok()?;
        self.last_join = Some(join.clone());
        self.max_term_seen = self.max_term_seen.max(join.term);
        if self.mode != Mode::Candidate {
            self.become_candidate(now, "joinLeaderInTerm");
        }
        Some(join)
    }

    /// ES's `ensureTermAtLeast`.
    fn ensure_term_at_least(&mut self, now: u64, source: &DiscoveryNode, term: u64) -> Option<Join> {
        if self.state.current_term() < term { self.join_leader_in_term(now, &StartJoin { source_node: source.clone(), term }) } else { None }
    }

    /// ES's `updateMaxTermSeen`: a leader that learns of a newer term
    /// re-runs an election in an even newer one (deferred while
    /// publishing).
    fn update_max_term_seen(&mut self, now: u64, term: u64) {
        self.max_term_seen = self.max_term_seen.max(term);
        if self.mode == Mode::Leader && self.max_term_seen > self.state.current_term() && self.publication.is_none() {
            let local = self.local.clone();
            let t = self.max_term_seen;
            let _ = self.ensure_term_at_least(now, &local, t);
            self.start_election(now);
        }
    }

    fn join_with_destination(&self, dest: &DiscoveryNode, term: u64) -> Option<Join> {
        self.last_join.clone().filter(|j| j.target_node == dest.id && j.term == term)
    }

    fn send_join_request(&mut self, now: u64, dest: &DiscoveryNode, join: Option<Join>) {
        if !dest.is_master_eligible() {
            return;
        }
        let min_term = self.state.current_term();
        if dest.id == self.local.id {
            let local = self.local.clone();
            let _ = self.process_join_request(now, None, local, min_term, join);
            return;
        }
        let key: JoinKey = (dest.transport_address.clone(), min_term, join.clone());
        if !self.pending_joins_out.insert(key.clone()) {
            return;
        }
        let meta = &self.state.last_accepted_state().metadata;
        let (uuid, committed) = (meta.cluster_uuid.clone(), meta.cluster_uuid_committed);
        let (cluster_name, source) = (self.settings.cluster_name.clone(), self.local.clone());
        let deadline = now + self.settings.join_timeout_ms;
        self.send_request(&dest.transport_address, Pending::Join { key }, Some(deadline), |request_id| Message::JoinRequest {
            request_id,
            cluster_name,
            source,
            min_term,
            join,
            cluster_uuid: uuid,
            cluster_uuid_committed: committed,
        });
    }

    fn on_start_join(&mut self, now: u64, sj: StartJoin) {
        if sj.term <= self.state.current_term() {
            return;
        }
        if let Some(join) = self.join_leader_in_term(now, &sj) {
            self.send_join_request(now, &sj.source_node, Some(join));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_join_request(&mut self, now: u64, from: &str, request_id: u64, cluster_name: &str, source: DiscoveryNode, min_term: u64, join: Option<Join>, uuid: &str, committed: bool) {
        let reply = Some((from.to_string(), request_id));
        if cluster_name != self.settings.cluster_name {
            let msg = format!("join validation failed: node [{}] has cluster name [{cluster_name}] but this cluster is [{}]", source.name, self.settings.cluster_name);
            return self.reply_join(reply, Err(msg));
        }
        let ours = self.state.last_accepted_state().metadata.cluster_uuid.clone();
        if committed && uuid != ours {
            let msg = format!(
                "join validation on cluster state with a different cluster uuid [{ours}] than local cluster uuid [{uuid}], rejecting: node [{}] previously joined a different cluster and its data path must be wiped before it can join this one",
                source.name
            );
            return self.reply_join(reply, Err(msg));
        }
        if let Err(e) = self.process_join_request(now, reply.clone(), source, min_term, join) {
            self.reply_join(reply, Err(e));
        }
    }

    /// ES's `processJoinRequest`. On `Err` the caller replies.
    fn process_join_request(&mut self, now: u64, reply: Option<(String, u64)>, source: DiscoveryNode, min_term: u64, join: Option<Join>) -> Result<(), String> {
        self.update_max_term_seen(now, min_term);
        let prev_won = self.state.election_won();
        if let Some(j) = &join {
            self.handle_join(now, j)?;
        }
        match self.mode {
            Mode::Follower => return Err("join target is a follower".into()),
            Mode::Candidate | Mode::Leader => {
                if let Some(pos) = self.incoming_joins.iter().position(|j| j.node.id == source.id) {
                    let old = self.incoming_joins.remove(pos);
                    self.reply_join(old.reply, Err(format!("received a newer join from {}", source.name)));
                }
                self.incoming_joins.push(IncomingJoin { node: source, reply });
            }
        }
        if !prev_won && self.state.election_won() && self.mode == Mode::Candidate {
            self.become_leader(now);
        } else if self.mode == Mode::Leader {
            self.maybe_publish(now);
        }
        Ok(())
    }

    /// ES's `handleJoin`.
    fn handle_join(&mut self, now: u64, join: &Join) -> Result<(), String> {
        let local = self.local.clone();
        if let Some(own) = self.ensure_term_at_least(now, &local, join.term) {
            let _ = self.handle_join(now, &own);
        }
        if self.state.election_won() {
            // Already won: the vote only matters for reconfiguration.
            let _ = self.state.handle_join(join);
            Ok(())
        } else {
            self.state.handle_join(join).map(|_| ()).map_err(|e| e.0)
        }
    }

    // ================================================= peer discovery ==

    fn peers_wakeup(&mut self, now: u64) {
        if !self.peer_finder_active {
            return;
        }
        self.next_peer_wakeup = now + self.settings.find_peers_interval_ms;
        let mut addrs: BTreeSet<String> = self.settings.seed_hosts.iter().cloned().collect();
        addrs.extend(self.state.last_accepted_state().master_eligible_nodes().map(|n| n.transport_address.clone()));
        addrs.extend(self.peers.keys().cloned());
        for a in addrs {
            self.probe(now, &a);
        }
    }

    fn probe(&mut self, now: u64, addr: &str) {
        if !self.peer_finder_active || addr == self.local.transport_address {
            return;
        }
        if self.peers.get(addr).is_some_and(|p| p.in_flight.is_some()) {
            return;
        }
        let (cluster_name, source, known_peers) = (self.settings.cluster_name.clone(), self.local.clone(), self.discovered_peers());
        let deadline = now + self.settings.request_peers_timeout_ms;
        let id = self.send_request(addr, Pending::Peers, Some(deadline), |request_id| Message::PeersRequest { request_id, cluster_name, source, known_peers });
        self.peers.entry(addr.to_string()).or_default().in_flight = Some(id);
    }

    fn on_peers_request(&mut self, now: u64, from: &str, request_id: u64, cluster_name: &str, source: DiscoveryNode, known_peers: Vec<DiscoveryNode>) {
        if cluster_name != self.settings.cluster_name {
            let result = Err(format!("cluster name mismatch: [{cluster_name}] vs [{}]", self.settings.cluster_name));
            return self.send(from, Message::PeersResponse { request_id, result });
        }
        let response = if self.peer_finder_active {
            if source.is_master_eligible() && source.id != self.local.id {
                let addr = source.transport_address.clone();
                self.peers.entry(addr).or_default().node = Some(source);
                for p in known_peers.iter().filter(|p| p.is_master_eligible()) {
                    self.probe(now, &p.transport_address);
                }
                self.on_found_peers_updated(now);
            }
            PeersResponse { source: self.local.clone(), master_node: None, known_peers: self.discovered_peers(), term: self.current_term() }
        } else {
            let master = match self.mode {
                Mode::Leader => Some(self.local.clone()),
                _ => self.leader.clone(),
            };
            PeersResponse { source: self.local.clone(), master_node: master, known_peers: Vec::new(), term: self.current_term() }
        };
        self.send(from, Message::PeersResponse { request_id, result: Ok(response) });
    }

    fn on_peers_response(&mut self, now: u64, addr: &str, id: u64, result: Result<PeersResponse, String>) {
        if !self.peer_finder_active {
            return;
        }
        match self.peers.get_mut(addr) {
            Some(p) if p.in_flight == Some(id) => p.in_flight = None,
            _ => return,
        }
        match result {
            Ok(r) => {
                let is_peer = r.source.is_master_eligible() && r.source.id != self.local.id;
                self.peers.get_mut(addr).expect("present").node = is_peer.then(|| r.source.clone());
                for p in r.known_peers.iter().filter(|p| p.is_master_eligible()) {
                    self.probe(now, &p.transport_address);
                }
                if let Some(m) = r.master_node {
                    if m.id == r.source.id && m.id != self.local.id {
                        self.on_active_master_found(now, m, r.term);
                    } else {
                        self.probe(now, &m.transport_address);
                    }
                }
            }
            Err(_) => self.peers.get_mut(addr).expect("present").node = None,
        }
        self.on_found_peers_updated(now);
    }

    fn on_active_master_found(&mut self, now: u64, master: DiscoveryNode, term: u64) {
        if self.mode != Mode::Candidate {
            return;
        }
        let _ = self.ensure_term_at_least(now, &master, term);
        let join = self.join_with_destination(&master, term);
        self.send_join_request(now, &master, join);
    }

    /// ES's `onFoundPeersUpdated`: bootstrap if possible; run the election
    /// scheduler only while the discovered peers could win an election.
    fn on_found_peers_updated(&mut self, now: u64) {
        self.check_bootstrap(now);
        if self.mode != Mode::Candidate || !self.local.is_master_eligible() {
            return;
        }
        let mut votes: BTreeSet<String> = self.discovered_peers().into_iter().map(|n| n.id).collect();
        votes.insert(self.local.id.clone());
        if self.state.is_election_quorum(&votes) {
            if self.election.is_none() {
                self.election = Some(ElectionScheduler { attempt: 0, next_at: now });
                self.schedule_next_election(now, 0);
            }
        } else {
            self.election = None;
            self.prevote = None;
        }
    }

    /// ES's `ClusterBootstrapService`.
    fn check_bootstrap(&mut self, _now: u64) {
        if !self.bootstrapping_permitted || self.settings.initial_master_nodes.is_empty() || !self.local.is_master_eligible() || !self.state.last_accepted_config().is_empty() {
            return;
        }
        let mut nodes = self.discovered_peers();
        nodes.push(self.local.clone());
        let mut reqs: Vec<String> = Vec::new();
        for r in &self.settings.initial_master_nodes {
            if !reqs.contains(r) {
                reqs.push(r.clone());
            }
        }
        let mut matched: Vec<String> = Vec::new();
        let mut unsatisfied: Vec<String> = Vec::new();
        for r in &reqs {
            let m = nodes.iter().find(|n| !matched.contains(&n.id) && (n.name == *r || n.transport_address == *r || n.ip == *r));
            match m {
                Some(n) => matched.push(n.id.clone()),
                None => unsatisfied.push(r.clone()),
            }
        }
        if matched.len() * 2 <= reqs.len() {
            return;
        }
        self.bootstrapping_permitted = false;
        let config = VotingConfiguration::new(matched.into_iter().chain(unsatisfied.iter().map(|r| format!("{BOOTSTRAP_PLACEHOLDER_PREFIX}{r}"))));
        let mut init = self.state.last_accepted_state().clone();
        init.coordination.last_accepted_config = config.clone();
        init.coordination.last_committed_config = config;
        if init.metadata.cluster_uuid == UNKNOWN_UUID {
            init.metadata.cluster_uuid = self.random_uuid();
            init.metadata.cluster_uuid_committed = false;
        }
        if let Err(e) = self.state.set_initial_state(init) {
            tracing::warn!(error = %e, "bootstrapping failed");
        } else {
            tracing::info!(node = %self.local.name, "bootstrapped cluster with initial voting configuration {:?}", self.state.last_accepted_config().0);
        }
    }

    // ====================================================== elections ==

    /// ES's `ElectionSchedulerFactory`: delay uniformly random in
    /// `[1, min(initial + attempt * back_off, max)]` plus `grace`.
    fn schedule_next_election(&mut self, now: u64, grace: u64) {
        let s = &self.settings;
        let (initial, back_off, max) = (s.election_initial_timeout_ms, s.election_back_off_time_ms, s.election_max_timeout_ms);
        let Some(e) = self.election.as_mut() else { return };
        let max_delay = initial.saturating_add(e.attempt.saturating_mul(back_off)).min(max).max(1);
        e.attempt += 1;
        let delay = self.rng.gen_range(1..=max_delay);
        e.next_at = now + grace + delay;
    }

    fn local_prevote_response(&self) -> PreVoteResponse {
        PreVoteResponse { current_term: self.state.current_term(), last_accepted_term: self.state.last_accepted_term(), last_accepted_version: self.state.last_accepted_version() }
    }

    fn start_prevote(&mut self, now: u64) {
        self.prevote_seq += 1;
        let round = self.prevote_seq;
        self.prevote = Some(PreVoteRound { id: round, votes: BTreeSet::new(), started: false });
        let peers = self.discovered_peers();
        let local_id = self.local.id.clone();
        let own = self.local_prevote_response();
        self.on_prevote_response(now, round, &local_id, own);
        for n in peers {
            if self.prevote.as_ref().is_none_or(|p| p.id != round || p.started) {
                break;
            }
            let (source, current_term) = (self.local.clone(), self.current_term());
            let deadline = now + self.settings.request_peers_timeout_ms;
            self.send_request(&n.transport_address, Pending::PreVote { round, node_id: n.id.clone() }, Some(deadline), |request_id| Message::PreVoteRequest { request_id, source, current_term });
        }
    }

    fn on_prevote_request(&mut self, now: u64, from: &str, request_id: u64, source: DiscoveryNode, term: u64) {
        self.update_max_term_seen(now, term);
        let leader = match self.mode {
            Mode::Leader => Some(self.local.clone()),
            Mode::Follower => self.leader.clone(),
            Mode::Candidate => None,
        };
        let result = match leader {
            // A follower still offers its pre-vote to its own leader, which
            // may have stepped down and wants to be re-elected.
            Some(l) if l.id != source.id => Err(format!("rejecting pre-vote request from [{}] as there is already a leader [{}]", source.name, l.name)),
            _ => Ok(self.local_prevote_response()),
        };
        self.send(from, Message::PreVoteResponse { request_id, result });
    }

    fn on_prevote_response(&mut self, now: u64, round: u64, node_id: &str, r: PreVoteResponse) {
        self.update_max_term_seen(now, r.current_term);
        if self.mode != Mode::Candidate {
            return;
        }
        let (lat, lav) = (self.state.last_accepted_term(), self.state.last_accepted_version());
        if r.last_accepted_term > lat || (r.last_accepted_term == lat && r.last_accepted_version > lav) {
            return; // the responder is fresher than us
        }
        let Some(pv) = self.prevote.as_mut() else { return };
        if pv.id != round || pv.started {
            return;
        }
        pv.votes.insert(node_id.to_string());
        let votes = pv.votes.clone();
        if self.state.is_election_quorum(&votes) {
            self.prevote.as_mut().expect("present").started = true;
            self.start_election(now);
        }
    }

    /// ES's `startElection`: ask every known master-eligible node (and
    /// ourselves) to vote for us in a fresh term.
    /// ES's `localNodeMayWinElection`, plus an excluded-node fallback: an
    /// excluded master-eligible node may stand once it has been a candidate
    /// for `excluded_node_election_fallback_ms` (default 30s, configurable).
    /// Example it covers: 3 masters, `c` excluded and reconfigured out while
    /// leading; `c` accepts a newer state that nobody else accepted, then
    /// restarts. `a` and `b` still hold the older state whose committed
    /// configuration includes `c`, so they need `c`'s vote, which they must
    /// refuse because `c` is fresher. Standing still needs a quorum of votes
    /// (every election is checked by `CoordinationState`), and an excluded
    /// master that wins abdicates when another eligible node can take over.
    fn may_stand_for_election(&self, now: u64) -> bool {
        may_win_election(self.state.last_accepted_state(), &self.local)
            || (self.local.is_master_eligible() && !self.local.is_voting_only() && now >= self.candidate_since + self.settings.excluded_node_election_fallback_ms)
    }

    fn start_election(&mut self, now: u64) {
        if self.mode != Mode::Candidate || !self.may_stand_for_election(now) {
            return;
        }
        let term = self.state.current_term().max(self.max_term_seen) + 1;
        let sj = StartJoin { source_node: self.local.clone(), term };
        let mut targets: BTreeSet<String> = self.discovered_peers().into_iter().map(|n| n.transport_address).collect();
        targets.extend(self.state.last_accepted_state().master_eligible_nodes().filter(|n| n.id != self.local.id).map(|n| n.transport_address.clone()));
        targets.remove(&self.local.transport_address);
        self.on_start_join(now, sj.clone());
        for t in targets {
            self.send(&t, Message::StartJoin(sj.clone()));
        }
    }

    // ================================================ follower duties ==

    fn on_publish_request(&mut self, now: u64, from: &str, request_id: u64, state: ClusterState) {
        let result = self.handle_publish_request(now, state);
        self.send(from, Message::PublishResponse { request_id, result });
    }

    fn handle_publish_request(&mut self, now: u64, state: ClusterState) -> Result<PublishWithJoinResponse, String> {
        if state.cluster_name != self.settings.cluster_name {
            return Err(format!("received publication from cluster [{}] but this node is in cluster [{}]", state.cluster_name, self.settings.cluster_name));
        }
        let master = state.master().cloned().ok_or("published state has no master")?;
        if master.id == self.local.id {
            return Err("received own publication".into());
        }
        let meta = &self.state.last_accepted_state().metadata;
        if meta.cluster_uuid_committed && state.metadata.cluster_uuid != meta.cluster_uuid {
            return Err(format!(
                "This node previously joined a cluster with UUID [{}] and is now trying to join a different cluster with UUID [{}]",
                meta.cluster_uuid, state.metadata.cluster_uuid
            ));
        }
        if state.nodes.get(&self.local.id) != Some(&self.local) {
            return Err("published state does not contain this node".into());
        }
        let term = state.coordination.term;
        let _ = self.ensure_term_at_least(now, &master, term);
        let resp = self.state.handle_publish_request(state).map_err(|e| e.0)?;
        self.become_follower(now, master.clone());
        Ok(PublishWithJoinResponse { term: resp.term, version: resp.version, join: self.join_with_destination(&master, term) })
    }

    fn on_apply_commit(&mut self, _now: u64, from: &str, request_id: u64, commit: ApplyCommit) {
        let result = match self.state.handle_commit(&commit) {
            Err(e) => Err(e.0),
            Ok(()) => {
                let committed = self.state.last_accepted_state().clone();
                let s = if self.mode == Mode::Candidate { self.with_no_master_block(&committed) } else { committed };
                self.set_applied(s);
                Ok(ApplyCommitAck { node_id: self.local.id.clone(), term: commit.term, version: commit.version })
            }
        };
        self.send(from, Message::ApplyCommitResponse { request_id, result });
    }

    fn on_leader_check(&mut self, from: &str, request_id: u64, source: DiscoveryNode, target_ephemeral_id: &str) {
        let result = if target_ephemeral_id != self.local.ephemeral_id {
            Err("leader check addressed to a previous incarnation of this node".to_string())
        } else if self.mode != Mode::Leader {
            Err(format!("rejecting leader check since [{}] is not the elected master", self.local.name))
        } else if !self.master_state().nodes.contains_key(&source.id) {
            Err(format!("rejecting leader check from [{}] which has been removed from the cluster", source.name))
        } else {
            Ok(())
        };
        self.send(from, Message::LeaderCheckResponse { request_id, result });
    }

    /// ES's `onFollowerCheckRequest`.
    fn on_follower_check(&mut self, now: u64, from: &str, request_id: u64, source: DiscoveryNode, term: u64, target_ephemeral_id: &str) {
        let result = if target_ephemeral_id != self.local.ephemeral_id {
            Err("follower check addressed to a previous incarnation of this node".to_string())
        } else {
            let _ = self.ensure_term_at_least(now, &source, term);
            if self.current_term() != term {
                Err(format!("current term is {}, rejecting follower check in term {term}", self.current_term()))
            } else if self.mode == Mode::Leader {
                Err("rejecting follower check: this node is the leader".to_string())
            } else if self.state.last_accepted_term() < self.current_term() {
                self.become_follower(now, source);
                Ok(())
            } else if self.mode == Mode::Follower || !self.pending_joins_out.is_empty() {
                Ok(())
            } else {
                Err("received follower check from faulty master".to_string())
            }
        };
        self.send(from, Message::FollowerCheckResponse { request_id, result });
    }

    fn leader_check_wakeup(&mut self, now: u64) {
        let Some(c) = &self.leader_checker else { return };
        if c.in_flight.is_some() || now < c.next_at {
            return;
        }
        let (addr, eph) = (c.node.transport_address.clone(), c.node.ephemeral_id.clone());
        let source = self.local.clone();
        let deadline = now + self.settings.leader_check_timeout_ms;
        let id = self.send_request(&addr, Pending::LeaderCheck, Some(deadline), |request_id| Message::LeaderCheck { request_id, source, target_ephemeral_id: eph });
        self.leader_checker.as_mut().expect("present").in_flight = Some(id);
    }

    fn on_leader_check_result(&mut self, now: u64, id: u64, result: Result<(), (String, bool)>) {
        let interval = self.settings.leader_check_interval_ms;
        let retry = self.settings.leader_check_retry_count;
        let Some(c) = self.leader_checker.as_mut() else { return };
        if c.in_flight != Some(id) {
            return;
        }
        c.in_flight = None;
        c.next_at = now + interval;
        let failed = match result {
            Ok(()) => {
                c.failures = 0;
                None
            }
            Err((reason, true)) => Some(reason),
            Err((reason, false)) => {
                c.failures += 1;
                (c.failures >= retry).then(|| format!("leader check failed {} times: {reason}", c.failures))
            }
        };
        if let Some(reason) = failed {
            if self.mode == Mode::Follower {
                self.become_candidate(now, &format!("leader failed: {reason}"));
            }
        }
    }

    // ================================================== leader duties ==

    fn follower_checks_wakeup(&mut self, now: u64) {
        let due: Vec<String> = self.follower_checks.iter().filter(|(_, c)| c.in_flight.is_none() && now >= c.next_at).map(|(id, _)| id.clone()).collect();
        for node_id in due {
            let c = &self.follower_checks[&node_id];
            let (addr, eph) = (c.node.transport_address.clone(), c.node.ephemeral_id.clone());
            let (source, term) = (self.local.clone(), self.current_term());
            let deadline = now + self.settings.follower_check_timeout_ms;
            let id = self.send_request(&addr, Pending::FollowerCheck { node_id: node_id.clone() }, Some(deadline), |request_id| Message::FollowerCheck {
                request_id,
                source,
                term,
                target_ephemeral_id: eph,
            });
            self.follower_checks.get_mut(&node_id).expect("present").in_flight = Some(id);
        }
    }

    fn on_follower_check_result(&mut self, now: u64, node_id: &str, id: u64, result: Result<(), (String, bool)>) {
        let interval = self.settings.follower_check_interval_ms;
        let retry = self.settings.follower_check_retry_count;
        let Some(c) = self.follower_checks.get_mut(node_id) else { return };
        if c.in_flight != Some(id) {
            return;
        }
        c.in_flight = None;
        c.next_at = now + interval;
        let failed = match result {
            Ok(()) => {
                c.failures = 0;
                c.last_ok = Some(now);
                None
            }
            Err((_, true)) => Some("disconnected".to_string()),
            Err((reason, false)) => {
                c.failures += 1;
                (c.failures >= retry).then(|| format!("followers check retry count exceeded [{}]: {reason}", c.failures))
            }
        };
        if let Some(reason) = failed {
            self.fail_node(now, node_id, &reason);
        }
    }

    /// Track the nodes of a state being published: follower checks for new
    /// nodes, forget removed ones.
    fn set_follower_check_nodes(&mut self, now: u64, s: &ClusterState) {
        self.follower_checks.retain(|id, c| s.nodes.get(id) == Some(&c.node));
        for n in s.nodes.values() {
            if n.id != self.local.id && !self.follower_checks.contains_key(&n.id) {
                let next_at = now + self.settings.follower_check_interval_ms;
                self.follower_checks.insert(n.id.clone(), Checker { node: n.clone(), last_ok: None, failures: 0, in_flight: None, next_at });
            }
        }
        self.applied_versions.retain(|id, _| s.nodes.contains_key(id));
        self.lag_checks.retain(|l| s.nodes.contains_key(&l.0));
    }

    /// Remove a faulty node from the cluster (ES's `removeNode`).
    fn fail_node(&mut self, now: u64, node_id: &str, reason: &str) {
        if self.mode != Mode::Leader {
            return;
        }
        tracing::info!(node = %self.local.name, removed = node_id, reason, "removing node");
        self.follower_checks.remove(node_id);
        self.pending_node_left.insert(node_id.to_string(), reason.to_string());
        if let Some(p) = self.publication.as_mut() {
            if let Some(ph) = p.targets.get_mut(node_id) {
                if ph.active() {
                    *ph = Phase::Failed;
                }
            }
        }
        self.check_publication(now);
        self.maybe_publish(now);
    }

    /// Voting configuration the leader should move to, if it differs and
    /// a change is currently allowed (ES's `improveConfiguration`).
    fn improved_config(&self, s: &ClusterState) -> Option<VotingConfiguration> {
        let current = &s.coordination.last_accepted_config;
        let live: BTreeSet<String> =
            s.nodes.values().filter(|n| n.is_master_eligible() && (n.id == self.local.id || self.state.contains_join_vote_for(&n.id))).map(|n| n.id.clone()).collect();
        let retired: BTreeSet<String> = s.coordination.voting_config_exclusions.iter().map(|e| e.node_id.clone()).collect();
        let cfg = reconfigure(&live, &retired, &self.local.id, current, self.settings.auto_shrink_voting_configuration);
        if cfg == *current || self.state.last_committed_config() != self.state.last_accepted_config() || !self.state.join_votes_have_quorum_for(&cfg) {
            return None;
        }
        Some(cfg)
    }

    /// The master service: if idle, batch all queued work (joins, removals,
    /// driver tasks, reconfiguration) into one new state and publish it.
    fn maybe_publish(&mut self, now: u64) {
        if self.mode != Mode::Leader || self.publication.is_some() {
            return;
        }
        let established = self.state.last_accepted_term() == self.state.current_term();
        let base = self.state.last_accepted_state().clone();
        let mut new = base.clone();

        // 1. Membership.
        let joins = std::mem::take(&mut self.incoming_joins);
        let mut joined: Vec<DiscoveryNode> = Vec::new();
        let mut left: Vec<(String, String)> = Vec::new();
        for (id, reason) in std::mem::take(&mut self.pending_node_left) {
            if id != self.local.id && new.nodes.remove(&id).is_some() {
                left.push((id, reason));
            }
        }
        for j in &joins {
            let n = &j.node;
            if n.id == self.local.id {
                continue;
            }
            match new.nodes.get(&n.id) {
                Some(existing) if existing == n => {}
                Some(_) => {
                    // Same node id, new incarnation (restart): replace.
                    new.nodes.insert(n.id.clone(), n.clone());
                    left.push((n.id.clone(), "node restarted".into()));
                    joined.push(n.clone());
                }
                None => {
                    // A different node at the same address (data wiped):
                    // the old one is gone.
                    let stale: Vec<String> = new.nodes.values().filter(|o| o.transport_address == n.transport_address && o.id != self.local.id).map(|o| o.id.clone()).collect();
                    for s in stale {
                        new.nodes.remove(&s);
                        left.push((s, "replaced by a new node at the same address".into()));
                    }
                    new.nodes.insert(n.id.clone(), n.clone());
                    joined.push(n.clone());
                }
            }
        }
        // Resolve name-only exclusions for nodes that joined.
        for e in new.coordination.voting_config_exclusions.iter_mut() {
            if e.node_id == "_absent_" {
                if let Some(n) = joined.iter().find(|n| n.name == e.node_name) {
                    e.node_id = n.id.clone();
                }
            }
        }
        // The state read from disk names this node's previous incarnation:
        // this process restarted, so (like any restarted node) it holds no
        // open shard copies whatever the routing table says.
        if new.nodes.get(&self.local.id).is_some_and(|e| e != &self.local) {
            left.push((self.local.id.clone(), "node restarted".into()));
            joined.push(self.local.clone());
        }
        new.nodes.insert(self.local.id.clone(), self.local.clone());
        if !joined.is_empty() || !left.is_empty() {
            new = self.updater.nodes_changed(&new, &joined, &left);
        }

        // 2. Driver tasks, in submission order.
        let mut task_ids = Vec::new();
        for t in std::mem::take(&mut self.tasks) {
            match (t.task)(&new) {
                Ok(s) => {
                    new = s;
                    task_ids.push(t.id);
                }
                Err(e) => self.emit(Output::TaskCompleted { task_id: t.id, result: Err(e) }),
            }
        }

        // 3. Fields owned by coordination.
        new.cluster_name = self.settings.cluster_name.clone();
        new.master_node_id = Some(self.local.id.clone());
        new.nodes.insert(self.local.id.clone(), self.local.clone());
        new.coordination.term = self.state.current_term();
        new.coordination.last_committed_config = base.coordination.last_committed_config.clone();
        new.coordination.last_accepted_config = base.coordination.last_accepted_config.clone();
        new.metadata.cluster_uuid = base.metadata.cluster_uuid.clone();
        new.metadata.cluster_uuid_committed = base.metadata.cluster_uuid_committed;
        // A master exists and the state is recovered (it is read from the
        // master's persisted state): drop the no-master and
        // state-not-recovered blocks.
        new.blocks.global.retain(|b| b.id != 1 && b.id != 2);

        // The segment format floor follows the final membership: raised to
        // what every member reads, never lowered (an old member keeps it).
        // Nothing to do once it reached what this leader reads itself.
        if crate::compatibility::effective_segment_format_floor(&new.metadata) < crate::compatibility::advertised_read_max(&self.local) {
            let floor = crate::compatibility::negotiated_segment_format(&new, |n| self.updater.segment_read_max(&self.settings.cluster_name, &self.local, n));
            if crate::compatibility::raise_segment_format(&mut new, floor) {
                tracing::info!(floor, "raising the cluster's segment format");
            }
        }

        // 4. Reconfiguration.
        if let Some(cfg) = self.improved_config(&new) {
            new.coordination.last_accepted_config = cfg;
        }

        // 5. Publish if anything changed. Joins always publish so that the
        // joining node (possibly already a member) learns of its leader.
        let mut cmp = new.clone();
        cmp.version = base.version;
        cmp.state_uuid = base.state_uuid.clone();
        if established && joins.is_empty() && cmp == base {
            for id in task_ids {
                self.emit(Output::TaskCompleted { task_id: id, result: Ok(base.version) });
            }
            return;
        }
        new.version = base.version + 1;
        new.state_uuid = self.random_uuid();
        self.start_publication(now, new, joins, task_ids);
    }

    fn start_publication(&mut self, now: u64, new_state: ClusterState, joins: Vec<IncomingJoin>, tasks: Vec<u64>) {
        let id = self.next_publication_id;
        self.next_publication_id += 1;
        if let Err(e) = self.state.handle_client_value(&new_state) {
            let p = Publication { id, state: new_state, deadline: now, targets: BTreeMap::new(), commit: None, joins, tasks };
            self.fail_publication_listeners(p, &format!("publication failed: {e}"));
            self.become_candidate(now, &format!("cannot publish: {e}"));
            return;
        }
        self.set_follower_check_nodes(now, &new_state);
        let targets: BTreeMap<String, Phase> = new_state.nodes.keys().map(|k| (k.clone(), Phase::Publishing)).collect();
        let remote: Vec<DiscoveryNode> = new_state.nodes.values().filter(|n| n.id != self.local.id).cloned().collect();
        let deadline = now + self.settings.publish_timeout_ms;
        self.publication = Some(Publication { id, state: new_state.clone(), deadline, targets, commit: None, joins, tasks });

        // The leader accepts its own publication first (persisted before
        // any message below is sent).
        let local_id = self.local.id.clone();
        match self.state.handle_publish_request(new_state.clone()) {
            Ok(resp) => self.on_publish_ack(now, id, &local_id, resp, None),
            Err(e) => self.target_failed(now, id, &local_id, &e.0),
        }
        for n in remote {
            if self.publication.as_ref().is_none_or(|p| p.id != id) {
                return;
            }
            let state = Box::new(new_state.clone());
            self.send_request(&n.transport_address, Pending::Publish { publication: id, node_id: n.id.clone() }, None, |request_id| Message::PublishRequest { request_id, state });
        }
        self.check_publication(now);
    }

    fn on_publish_response(&mut self, now: u64, publication: u64, node_id: &str, result: Result<PublishWithJoinResponse, String>) {
        if self.publication.as_ref().is_none_or(|p| p.id != publication) {
            return;
        }
        match result {
            Ok(r) => {
                let resp = PublishResponse { term: r.term, version: r.version };
                self.on_publish_ack(now, publication, node_id, resp, r.join);
            }
            Err(e) => self.target_failed(now, publication, node_id, &e),
        }
        self.check_publication(now);
    }

    fn on_publish_ack(&mut self, now: u64, publication: u64, node_id: &str, resp: PublishResponse, join: Option<Join>) {
        if let Some(j) = join {
            if j.term == self.current_term() {
                let _ = self.handle_join(now, &j);
            }
        }
        if self.mode != Mode::Leader || self.publication.as_ref().is_none_or(|p| p.id != publication) {
            return;
        }
        match self.state.handle_publish_response(node_id, resp) {
            Err(e) => self.target_failed(now, publication, node_id, &e.0),
            Ok(commit) => {
                let p = self.publication.as_mut().expect("checked");
                if p.targets.get(node_id) != Some(&Phase::Publishing) {
                    return;
                }
                p.targets.insert(node_id.to_string(), Phase::Accepted);
                if p.commit.is_none() {
                    if let Some(c) = commit {
                        p.commit = Some(c);
                        let accepted: Vec<String> = p.targets.iter().filter(|(_, ph)| **ph == Phase::Accepted).map(|(k, _)| k.clone()).collect();
                        // Commit locally first so the leader's persisted
                        // state is committed before anyone applies it.
                        let mut ordered: Vec<String> = accepted.iter().filter(|k| **k == self.local.id).cloned().collect();
                        ordered.extend(accepted.into_iter().filter(|k| *k != self.local.id));
                        for k in ordered {
                            self.send_commit(now, publication, &k);
                        }
                    }
                } else {
                    self.send_commit(now, publication, node_id);
                }
            }
        }
    }

    fn send_commit(&mut self, now: u64, publication: u64, node_id: &str) {
        let Some(p) = self.publication.as_mut() else { return };
        let Some(commit) = p.commit.clone() else { return };
        if p.id != publication || p.targets.get(node_id) != Some(&Phase::Accepted) {
            return;
        }
        if node_id == self.local.id {
            match self.state.handle_commit(&commit) {
                Ok(()) => {
                    p.targets.insert(node_id.to_string(), Phase::Done);
                    self.applied_versions.insert(node_id.to_string(), commit.version);
                }
                Err(e) => self.target_failed(now, publication, node_id, &e.0),
            }
            return;
        }
        p.targets.insert(node_id.to_string(), Phase::Committing);
        let addr = p.state.nodes[node_id].transport_address.clone();
        self.send_request(&addr, Pending::Commit { publication, node_id: node_id.to_string() }, None, |request_id| Message::ApplyCommit { request_id, commit });
    }

    fn on_commit_response(&mut self, now: u64, publication: u64, node_id: &str, result: Result<ApplyCommitAck, String>) {
        match result {
            Ok(ack) => {
                if ack.term == self.current_term() {
                    let v = self.applied_versions.entry(node_id.to_string()).or_insert(0);
                    *v = (*v).max(ack.version);
                }
                if let Some(p) = self.publication.as_mut() {
                    if p.id == publication && p.targets.get(node_id) == Some(&Phase::Committing) {
                        p.targets.insert(node_id.to_string(), Phase::Done);
                    }
                }
            }
            Err(e) => self.target_failed(now, publication, node_id, &e),
        }
        self.check_publication(now);
    }

    fn target_failed(&mut self, _now: u64, publication: u64, node_id: &str, reason: &str) {
        if let Some(p) = self.publication.as_mut() {
            if p.id == publication {
                if let Some(ph) = p.targets.get_mut(node_id) {
                    if ph.active() {
                        tracing::debug!(node_id, reason, "publication target failed");
                        *ph = Phase::Failed;
                    }
                }
            }
        }
    }

    /// Complete the publication once every target is done (committed) or
    /// when a commit quorum has become impossible.
    fn check_publication(&mut self, now: u64) {
        let Some(p) = &self.publication else { return };
        if p.commit.is_some() {
            if p.targets.values().all(|ph| !ph.active()) {
                self.complete_publication(now, Ok(()));
            }
        } else {
            let possible: BTreeSet<String> = p.targets.iter().filter(|(_, ph)| **ph != Phase::Failed).map(|(k, _)| k.clone()).collect();
            if !self.state.is_publish_quorum(&possible) {
                self.complete_publication(now, Err("publication failed: a quorum of master-eligible nodes can no longer accept it".into()));
            }
        }
    }

    fn on_publication_timeout(&mut self, now: u64) {
        let Some(p) = self.publication.as_mut() else { return };
        if p.commit.is_some() {
            for ph in p.targets.values_mut() {
                if ph.active() {
                    *ph = Phase::Failed;
                }
            }
            self.complete_publication(now, Ok(()));
        } else {
            self.complete_publication(now, Err(format!("publication failed: not committed within [cluster.publish.timeout] {}ms", self.settings.publish_timeout_ms)));
        }
    }

    /// Forget outstanding publish/commit requests of a finished
    /// publication (they have no timeout of their own).
    fn forget_publication_requests(&mut self, id: u64) {
        self.requests.retain(|_, r| !matches!(&r.kind, Pending::Publish { publication, .. } | Pending::Commit { publication, .. } if *publication == id));
    }

    fn fail_publication_listeners(&mut self, p: Publication, reason: &str) {
        self.forget_publication_requests(p.id);
        for j in p.joins {
            self.reply_join(j.reply, Err(reason.to_string()));
        }
        for t in p.tasks {
            self.emit(Output::TaskCompleted { task_id: t, result: Err(reason.to_string()) });
        }
        self.emit(Output::PublicationCompleted { publication_id: p.id, version: p.state.version, result: Err(reason.to_string()) });
    }

    fn complete_publication(&mut self, now: u64, result: Result<(), String>) {
        let p = self.publication.take().expect("publication in flight");
        if let Err(e) = result {
            self.fail_publication_listeners(p, &e);
            self.become_candidate(now, &e);
            return;
        }
        let version = p.state.version;
        self.forget_publication_requests(p.id);
        // The master applies the committed state last (ES applies on the
        // master at the end of the publication).
        let committed = self.state.last_accepted_state().clone();
        debug_assert_eq!(committed.version, version);
        self.set_applied(committed);
        // Lag detector: nodes that have not applied this version within
        // `cluster.follower_lag.timeout` are removed.
        for id in p.targets.keys() {
            if *id != self.local.id && self.applied_versions.get(id).copied().unwrap_or(0) < version {
                self.lag_checks.push((id.clone(), version, now + self.settings.follower_lag_timeout_ms));
            }
        }
        for j in &p.joins {
            self.reply_join(j.reply.clone(), Ok(()));
        }
        for t in &p.tasks {
            self.emit(Output::TaskCompleted { task_id: *t, result: Ok(version) });
        }
        self.emit(Output::PublicationCompleted { publication_id: p.id, version, result: Ok(()) });

        // ES's removePublicationAndPossiblyBecomeCandidate.
        let term = self.current_term();
        self.update_max_term_seen(now, term);
        if self.mode != Mode::Leader {
            return;
        }
        let st = self.state.last_accepted_state().clone();
        if !may_win_election(&st, &self.local) {
            let completed: BTreeSet<String> = p.targets.iter().filter(|(_, ph)| **ph == Phase::Done).map(|(k, _)| k.clone()).collect();
            if self.state.is_election_quorum(&completed) {
                let candidates: Vec<DiscoveryNode> = st.master_eligible_nodes().filter(|n| n.id != self.local.id && completed.contains(&n.id) && may_win_election(&st, n)).cloned().collect();
                if !candidates.is_empty() {
                    let target = candidates[self.rng.gen_range(0..candidates.len())].clone();
                    self.abdicate_to(now, target);
                    return;
                }
            }
        }
        let done: BTreeSet<String> = p.targets.iter().filter(|(_, ph)| **ph == Phase::Done).map(|(k, _)| k.clone()).collect();
        if self.reelect_for_missing_votes(now, &st, &done) {
            return;
        }
        self.maybe_publish(now);
    }

    /// Periodic form of the check below for an idle leader: nodes whose
    /// last follower check succeeded count as responsive.
    fn idle_vote_check(&mut self, now: u64) {
        if self.mode != Mode::Leader || self.publication.is_some() {
            return;
        }
        // Responsive = answered a follower check recently (a black-holed
        // node has no failures yet while its check is still in flight).
        let window = 3 * self.settings.follower_check_interval_ms;
        let healthy: BTreeSet<String> = self.follower_checks.iter().filter(|(_, c)| c.last_ok.is_some_and(|t| now <= t + window)).map(|(id, _)| id.clone()).collect();
        let st = self.state.last_accepted_state().clone();
        self.reelect_for_missing_votes(now, &st, &healthy);
    }

    /// Deviation from ES (liveness improvement): a master-eligible node that
    /// joined without voting for the leader in its term (it had voted for a
    /// losing candidate, or restarted in the same term) cannot be added to
    /// the voting configuration until the next election, because the
    /// reconfiguration safety rule requires a quorum of join votes in the
    /// new configuration. ES simply waits; e.g. a 3-node cluster can then
    /// sit with configuration `{a, b, placeholder}` and tolerate no
    /// failure. When this blocks a configuration the reconfigurator would
    /// otherwise choose, and every node involved has just acknowledged our
    /// publication, the leader re-runs its own election in a fresh term
    /// (exactly what ES does when it learns of a higher term). Rate-limited
    /// to once per `cluster.election.max_timeout`. Safety is unaffected: it
    /// is an ordinary election.
    fn reelect_for_missing_votes(&mut self, now: u64, st: &ClusterState, responsive: &BTreeSet<String>) -> bool {
        // Retry with exponential back-off (1s, 2s, 4s, ... up to
        // `cluster.election.max_timeout`): the re-election can race with
        // the missing voter's own election attempt.
        let backoff = 1_000u64.saturating_mul(1 << self.vote_reelections.min(20)).min(self.settings.election_max_timeout_ms);
        if self.last_vote_reelection.is_some_and(|t| now < t + backoff) {
            return false;
        }
        if self.state.last_committed_config() != self.state.last_accepted_config() || self.improved_config(st).is_some() {
            return false;
        }
        if self.mode != Mode::Leader || self.publication.is_some() || st.coordination.term != self.current_term() {
            return false;
        }
        let live: BTreeSet<String> = st.nodes.values().filter(|n| n.is_master_eligible() && (n.id == self.local.id || responsive.contains(&n.id))).map(|n| n.id.clone()).collect();
        let retired: BTreeSet<String> = st.coordination.voting_config_exclusions.iter().map(|e| e.node_id.clone()).collect();
        let current = &st.coordination.last_accepted_config;
        let wanted = reconfigure(&live, &retired, &self.local.id, current, self.settings.auto_shrink_voting_configuration);
        if wanted == *current || wanted.0.iter().all(|id| self.state.contains_join_vote_for(id)) {
            self.vote_reelections = 0;
            return false;
        }
        tracing::info!(node = %self.local.name, "re-running election to collect votes for voting configuration {:?}", wanted.0);
        self.last_vote_reelection = Some(now);
        self.vote_reelections += 1;
        let local = self.local.clone();
        let term = self.current_term() + 1;
        let _ = self.ensure_term_at_least(now, &local, term);
        self.start_election(now);
        true
    }

    /// ES's `abdicateTo`: ask every master-eligible node to vote for
    /// `target` in a new term, then step down.
    fn abdicate_to(&mut self, now: u64, target: DiscoveryNode) {
        tracing::info!(node = %self.local.name, to = %target.name, "abdicating");
        let term = self.state.current_term().max(self.max_term_seen) + 1;
        let sj = StartJoin { source_node: target, term };
        let targets: Vec<String> = self.state.last_accepted_state().master_eligible_nodes().filter(|n| n.id != self.local.id).map(|n| n.transport_address.clone()).collect();
        self.on_start_join(now, sj.clone());
        for t in targets {
            self.send(&t, Message::StartJoin(sj.clone()));
        }
    }
}

/// ES's `nodeMayWinElection`: master-eligible, not voting-only, and either
/// in a voting configuration or not excluded.
fn may_win_election(s: &ClusterState, n: &DiscoveryNode) -> bool {
    if !n.is_master_eligible() || n.is_voting_only() {
        return false;
    }
    let c = &s.coordination;
    c.last_committed_config.0.contains(&n.id) || c.last_accepted_config.0.contains(&n.id) || !c.voting_config_exclusions.iter().any(|e| e.node_id == n.id)
}

#[cfg(test)]
mod transport_failure_tests {
    use super::*;
    use super::super::simulation::mk_node;

    fn leader_with_follower() -> (Coordinator, String) {
        let local = mk_node("", 0, true, 1);
        let follower = mk_node("", 1, true, 1);
        let id = follower.id.clone();
        let mut settings = CoordinationSettings::new("c");
        settings.follower_check_retry_count = 3;
        let mut c = Coordinator::new(settings, local, PersistedState::empty("c"), 7, Box::new(NoopStateUpdater));
        c.mode = Mode::Leader;
        c.follower_checks.insert(id.clone(), Checker { node: follower, last_ok: None, failures: 0, in_flight: None, next_at: 0 });
        (c, id)
    }

    #[test]
    fn response_failures_retry_but_disconnect_fails_immediately() {
        let (mut c, id) = leader_with_follower();
        for attempt in 1..=3 {
            let now = attempt * c.settings.follower_check_interval_ms;
            c.follower_checks_wakeup(now);
            let request = c.follower_checks[&id].in_flight.unwrap();
            c.handle_request_failure(now, request, "response timeout");
            if attempt < 3 {
                assert_eq!(c.follower_checks[&id].failures, attempt as u32);
            } else {
                assert!(!c.follower_checks.contains_key(&id));
            }
        }
        let (mut c, id) = leader_with_follower();
        c.follower_checks_wakeup(0);
        let request = c.follower_checks[&id].in_flight.unwrap();
        c.handle_send_failure(0, request, "connection closed");
        assert!(!c.follower_checks.contains_key(&id));
    }
}
