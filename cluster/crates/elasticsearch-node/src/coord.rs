//! Drives the sans-IO [`Coordinator`] over the transport.
//!
//! Three independent execution contexts, as in ES (`Coordinator` mutex,
//! `ClusterApplierService` thread, transport-thread fault detection):
//!
//! * **The coordination actor** (one task) owns the coordinator and a
//!   ticker (the `cluster-persist` thread owns the gateway). Every call returns
//!   an ordered list of outputs, which the actor executes in order:
//!   - `Persist` is handed to the `cluster-persist` thread, which writes
//!     and fsyncs in order (a backlog in one write); every later output
//!     waits until it is durable (the Zen2 persistence-ordering contract:
//!     votes and acks may only leave the node once the term / state they
//!     vouch for is durable), but the actor itself never waits on disk;
//!   - `Send` is handed to the transport (fire-and-forget; a delivery
//!     failure of a request comes back as `handle_send_failure`, a closed
//!     connection as `handle_disconnect`);
//!   - `ApplyState` is *queued* to the cluster applier and never runs on the
//!     actor: a slow application (opening shards, a big mapping) cannot
//!     hold up joins, publications or fault detection;
//!   - completions resolve the callers' oneshots.
//!
//!   Outputs that vouch for an application are held back until the applier
//!   reports it done ([`deferred_until_applied`]): a follower's
//!   apply-commit ack (ES sends it after `ClusterApplierService` applied the
//!   state, and the lag detector depends on it), task completions (callers
//!   expect `ClusterNode::state()` to include their change) and publication
//!   completions. Holding back an ack is equivalent to a slow network, so
//!   the safety argument is unchanged.
//! * **The cluster applier** (one OS thread, `cluster-applier`) applies
//!   committed states in order. Newer queued states supersede older ones
//!   (it applies the newest and reports all as done), like ES's applier
//!   batching of pending states.
//! * **Fault detection responses** are answered directly by the transport
//!   handler from a durable [`HeartbeatAuthority`] snapshot the actor
//!   publishes after every step, without queuing behind the actor. Only
//!   the unambiguous "yes" answers are given there (a follower answering
//!   its own leader in the current term, the leader answering a current
//!   member); everything else (term changes, rejections, the first check
//!   after an election) goes through the coordinator.
//!
//! Wire format. Two transport actions; both handlers only enqueue (or
//! answer a heartbeat) and return an empty ack:
//!
//! * `internal:coordination/message_v2` (current): bincode
//!   [`EnvelopeV2`]. Cluster states are binary (their JSON-valued parts
//!   travel as pre-serialized bytes, `elasticsearch_cluster::wire`), and a
//!   publication to a node that accepted the previous one carries a
//!   [`ClusterStateDiff`] instead of the full state. The receiver applies
//!   the diff to its last accepted state if that is the diff's base and
//!   hands the coordinator the rebuilt full `PublishRequest`; otherwise it
//!   answers `ResendFull` and the master sends the full state for that
//!   request. Nodes that just joined (or missed the previous publication)
//!   get the full state, as in ES.
//! * `internal:coordination/message` (legacy, rolling upgrades): JSON
//!   `{"from": <sender transport address>, "msg": Message}`, full states
//!   only. A node sends v2 to every peer until a peer answers
//!   `action_not_found` (an older node) or [`UNSUPPORTED_FORMAT`] (a v2
//!   body of a format version or with fields it cannot decode; the body
//!   starts with [`WIRE_FORMAT`]): the message is re-sent as v1 at once and
//!   that peer gets v1 until it reconnects or [`V1_RECHECK`] passes, then
//!   the binary format is probed again. Old nodes only send v1, which new
//!   nodes still accept, so a mixed cluster keeps working; diffs and binary
//!   states are used between upgraded nodes only. A starting node answers
//!   coordination messages with a retryable "not ready" until its
//!   coordinator runs (never `action_not_found`). The persisted
//!   state stays JSON (`gateway`, written incrementally).

use crate::gateway::Gateway;
use bytes::Bytes;
use elasticsearch_cluster::coordination::{CoordinationError, Coordinator, Message, Mode, Output, PersistUpdate, StateTask};
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode};
use elasticsearch_cluster::wire::ClusterStateDiff;
use elasticsearch_transport::{ConnectionEvent, RemoteError, TransportService};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot};

/// Legacy JSON coordination action (see the module docs).
pub const ACTION: &str = "internal:coordination/message";
/// Binary coordination action with publication diffs.
pub const ACTION_V2: &str = "internal:coordination/message_v2";

#[derive(Serialize, Deserialize)]
struct Envelope {
    from: String,
    msg: Message,
}

/// What travels on [`ACTION_V2`].
#[derive(Serialize, Deserialize)]
enum Body {
    Message(Message),
    /// A `PublishRequest` expressed against the previously published state.
    PublishDiff { request_id: u64, diff: Box<ClusterStateDiff> },
    /// The receiver cannot apply the diff of publish request `request_id`
    /// (its last accepted state is not the base): send the full state.
    ResendFull { request_id: u64 },
}

#[derive(Serialize, Deserialize)]
struct EnvelopeV2 {
    from: String,
    body: Body,
}

fn encode_v1(from: String, msg: Message) -> Result<Bytes, String> {
    serde_json::to_vec(&Envelope { from, msg }).map(Bytes::from).map_err(|e| e.to_string())
}

/// Version of the binary envelope (the first byte of a v2 body). A node
/// that cannot decode a body (another format version, a field it does not
/// know) answers [`UNSUPPORTED_FORMAT`]: the sender falls back to the JSON
/// action for that message and peer, as for `action_not_found`.
pub const WIRE_FORMAT: u8 = 1;
/// Error type of an undecodable v2 body.
pub const UNSUPPORTED_FORMAT: &str = "coordination_wire_format_unsupported_exception";

fn encode_v2(env: &EnvelopeV2) -> Result<Bytes, String> {
    let mut out = vec![WIRE_FORMAT];
    bincode::serialize_into(&mut out, env).map_err(|e| e.to_string())?;
    Ok(Bytes::from(out))
}

fn decode_v2(body: &[u8]) -> Result<EnvelopeV2, RemoteError> {
    let unsupported = |why: String| RemoteError::new(UNSUPPORTED_FORMAT, why, 400);
    match body.split_first() {
        // Trailing bytes are an error: a sender's newer schema (a field
        // appended at the end) is refused and answered in JSON instead of
        // silently misread.
        Some((&WIRE_FORMAT, rest)) => {
            use bincode::Options;
            bincode::DefaultOptions::new().with_fixint_encoding().reject_trailing_bytes().deserialize(rest).map_err(|e| unsupported(format!("bad coordination message: {e}")))
        }
        Some((v, _)) => Err(unsupported(format!("coordination wire format [{v}] is not supported (this node speaks [{WIRE_FORMAT}])"))),
        None => Err(unsupported("empty coordination message".into())),
    }
}

/// Stop the process: a node that cannot persist or apply cluster state must
/// not keep running. Logged and written to stderr (a test harness shows no
/// tracing output, and exit code 1 alone does not say why).
fn fatal(why: &str) -> ! {
    tracing::error!("{why}; exiting");
    eprintln!("FATAL [{}]: {why}; exiting", std::thread::current().name().unwrap_or("?"));
    std::process::exit(1)
}

/// How long a peer stays on the JSON format after it answered
/// `action_not_found` / [`UNSUPPORTED_FORMAT`] before the binary format is
/// tried again (a peer upgraded meanwhile; a restart probed too early).
const V1_RECHECK: Duration = Duration::from_secs(60);

/// A coordination-state write outstanding longer than this makes the node
/// fail fault-detection checks (ES `FsHealthService`,
/// `monitor.fs.health.unhealthy_timeout_threshold` 60s): a node whose disk
/// hangs must not keep vouching for itself.
const PERSIST_STALL: Duration = Duration::from_secs(60);

/// Whether a persist outstanding since `since` is stalled at `now`.
fn persist_stalled_at(since: Option<Instant>, now: Instant) -> bool {
    since.is_some_and(|t| now.saturating_duration_since(t) >= PERSIST_STALL)
}

/// Whether a peer marked legacy (JSON only) at `since` still is at `now`
/// (after [`V1_RECHECK`] the binary format is probed again).
fn still_v1(since: Option<&Instant>, now: Instant) -> bool {
    since.is_some_and(|t| now.saturating_duration_since(*t) < V1_RECHECK)
}

/// Fault injection (tests and the fault matrix).
#[derive(Clone, Default)]
pub struct Faults {
    /// Send no publish requests (publications time out while fault
    /// detection stays healthy).
    pub drop_publications: Arc<std::sync::atomic::AtomicBool>,
    /// Speak only the legacy JSON action (an un-upgraded node; the test also
    /// unregisters the binary handler).
    pub legacy_wire: Arc<std::sync::atomic::AtomicBool>,
}

/// What went over the wire (tests assert diffs and fallbacks were used).
#[derive(Default)]
pub struct WireStats {
    pub full_states: std::sync::atomic::AtomicU64,
    pub diffs: std::sync::atomic::AtomicU64,
    pub resent_full: std::sync::atomic::AtomicU64,
    pub legacy_messages: std::sync::atomic::AtomicU64,
    pub downgrades: std::sync::atomic::AtomicU64,
}

/// Publications the leader sent most recently, for diffs: the newest
/// state, which nodes accepted it, and the one before.
#[derive(Default)]
struct PublishTracker {
    current: Option<Arc<ClusterState>>,
    /// Transport addresses that accepted `current`.
    current_acked: std::collections::HashSet<String>,
    previous: Option<Arc<ClusterState>>,
    previous_acked: std::collections::HashSet<String>,
    /// `previous` → `current`, computed once per publication.
    diff: Option<Arc<ClusterStateDiff>>,
    /// Publish requests of `current` by request id (for `ResendFull`).
    requests: HashMap<u64, String>,
}

impl PublishTracker {
    /// Note a publish request for `state`; starts a new publication when it
    /// is not the current one.
    fn sending(&mut self, state: &ClusterState, request_id: u64, to: &str) {
        let same = self.current.as_ref().is_some_and(|c| c.version == state.version && c.state_uuid == state.state_uuid);
        if !same {
            self.previous = self.current.take();
            self.previous_acked = std::mem::take(&mut self.current_acked);
            self.current = Some(Arc::new(state.clone()));
            self.diff = None;
            self.requests.clear();
        }
        self.requests.insert(request_id, to.to_string());
    }

    /// The diff to send to `to` for the current publication, if it accepted
    /// the previous one.
    fn diff_for(&mut self, to: &str) -> Option<Arc<ClusterStateDiff>> {
        let (prev, cur) = (self.previous.as_ref()?, self.current.as_ref()?);
        if !self.previous_acked.contains(to) || prev.version >= cur.version {
            return None;
        }
        if self.diff.is_none() {
            self.diff = Some(Arc::new(ClusterStateDiff::between(prev, cur)));
        }
        self.diff.clone()
    }

    fn accepted(&mut self, from: &str, version: u64) {
        if self.current.as_ref().is_some_and(|c| c.version == version) {
            self.current_acked.insert(from.to_string());
        }
    }
}

/// Read-only fault-detection authority, published by the actor after every
/// step once the term it names is durable. It cannot vote, advance a term,
/// or acknowledge a publication: it only confirms liveness.
#[derive(Clone)]
struct HeartbeatAuthority {
    mode: Mode,
    term: u64,
    local_ephemeral_id: String,
    leader: Option<DiscoveryNode>,
    members: HashMap<String, DiscoveryNode>,
}

impl HeartbeatAuthority {
    fn response(&self, from: &str, msg: &Message) -> Option<Message> {
        let same_peer = |known: &DiscoveryNode, source: &DiscoveryNode| {
            known.id == source.id && known.ephemeral_id == source.ephemeral_id
                && known.transport_address == from && source.transport_address == from
        };
        match msg {
            Message::FollowerCheck { request_id, source, term, target_ephemeral_id }
                if self.mode == Mode::Follower && *term == self.term
                    && *target_ephemeral_id == self.local_ephemeral_id
                    && self.leader.as_ref().is_some_and(|leader| same_peer(leader, source)) =>
                Some(Message::FollowerCheckResponse { request_id: *request_id, result: Ok(()) }),
            Message::LeaderCheck { request_id, source, target_ephemeral_id }
                if self.mode == Mode::Leader && *target_ephemeral_id == self.local_ephemeral_id
                    && self.members.get(&source.id).is_some_and(|peer| same_peer(peer, source)) =>
                Some(Message::LeaderCheckResponse { request_id: *request_id, result: Ok(()) }),
            _ => None,
        }
    }
}

type Heartbeat = Arc<parking_lot::RwLock<Option<HeartbeatAuthority>>>;

/// What the actor reports about itself (`_cat/master`, discovery stats).
#[derive(Clone, Debug)]
pub struct CoordInfo {
    pub mode: Mode,
    pub term: u64,
    pub leader: Option<DiscoveryNode>,
    pub discovered: Vec<DiscoveryNode>,
    pub publishing: bool,
}

pub type TaskResult = Result<u64, String>;

enum Cmd {
    Inbound { from: String, msg: Message },
    SendFailed { request_id: u64, reason: String, disconnected: bool },
    Disconnected { addr: SocketAddr },
    Submit { task: StateTask, reply: oneshot::Sender<TaskResult> },
    AddExclusions { ids: Vec<String>, names: Vec<String>, reply: oneshot::Sender<TaskResult> },
    ClearExclusions { wait_for_removal: bool, reply: oneshot::Sender<TaskResult> },
    Info { reply: oneshot::Sender<CoordInfo> },
    /// New seed addresses (the `file` seed hosts provider re-read).
    SetSeedHosts { hosts: Vec<String> },
    /// The applier finished every queued state up to `seq` (`ok`: without a
    /// panic).
    Applied { seq: u64, ok: bool },
    /// Every `Persist` up to `seq` is durable.
    Persisted { seq: u64 },
    /// A publish request sent as a diff (v2).
    InboundDiff { from: String, request_id: u64, diff: Box<ClusterStateDiff> },
    /// A follower could not apply our diff for `request_id`.
    ResendFull { from: String, request_id: u64 },
    /// The peer at this address has no v2 handler (an older node).
    PeerIsV1 { addr: String },
}

/// Applies committed (or no-master) states on this node. Called on the
/// cluster applier thread, in order; must not wait for the coordinator.
pub trait Applier: Send + Sync + 'static {
    fn apply(&self, state: Arc<ClusterState>);
    /// Called on the actor: keep it cheap.
    fn mode_changed(&self, _mode: Mode, _term: u64, _leader: Option<&DiscoveryNode>) {}
    /// A publication this node led failed (it stepped down). Called on the
    /// actor: keep it cheap.
    fn publication_failed(&self) {}
}

#[derive(Clone)]
pub struct CoordHandle {
    tx: mpsc::UnboundedSender<Cmd>,
    stats: Arc<WireStats>,
}

/// Fail before the coordinator can persist a vote, accept a publication or
/// add a member that cannot preserve activated lifecycle metadata.
fn validate_capability_message(current: &ClusterState, msg: &Message) -> Result<(), String> {
    use crate::internal_capabilities as caps;
    match msg {
        Message::PublishRequest { state, .. } => caps::validate_lifecycle_publication(current, state),
        // Refuse up front any member the followers would refuse in the
        // publication that adds it (lifecycle, pending pressure relocation,
        // segment formats); admitting it would fail the quorum and make the
        // leader step down on every retry of the same join.
        Message::JoinRequest { source, .. } => {
            caps::require_member(current, source)?;
            let mut joined = current.clone();
            joined.nodes.insert(source.id.clone(), source.clone());
            elasticsearch_cluster::compatibility::validate_segment_members(&joined)
        }
        Message::PreVoteRequest { source, .. } => caps::require_member(current, source),
        Message::StartJoin(start) => caps::require_member(current, &start.source_node),
        _ => Ok(()),
    }
}

async fn verify_message_capabilities(transport: &TransportService, from: &str, msg: &Message) -> Result<(), RemoteError> {
    let result = match msg {
        Message::JoinRequest { source, .. } | Message::PreVoteRequest { source, .. } => crate::capability_probe::verify(transport, source, true).await,
        Message::StartJoin(start) => crate::capability_probe::verify(transport, &start.source_node, true).await,
        Message::PublishRequest { state, .. } => {
            crate::capability_probe::verify_sender(transport, from).await
                .map_err(|reason| RemoteError::new("unsupported_operation_exception", reason, 503))?;
            crate::capability_probe::observe_members(transport, &state.nodes).await;
            crate::capability_probe::require_publication_proof(transport, state)
        },
        _ => Ok(()),
    };
    result.map_err(|reason| RemoteError::new("unsupported_operation_exception", reason, 503))
}

fn capability_failure_response(msg: &Message, reason: String) -> Option<Message> {
    match msg {
        Message::PublishRequest { request_id, .. } => Some(Message::PublishResponse { request_id: *request_id, result: Err(reason) }),
        Message::JoinRequest { request_id, .. } => Some(Message::JoinResponse { request_id: *request_id, result: Err(reason) }),
        Message::PreVoteRequest { request_id, .. } => Some(Message::PreVoteResponse { request_id: *request_id, result: Err(reason) }),
        // StartJoin is one-way; dropping it refuses the vote without advancing
        // the term or manufacturing a JoinRequest for an unsupported candidate.
        _ => None,
    }
}

fn not_master(e: CoordinationError) -> String {
    e.to_string()
}

impl CoordHandle {
    /// What this node sent over the wire.
    pub fn wire_stats(&self) -> &WireStats {
        &self.stats
    }

    /// Run a master-service task; resolves with the committed version, once
    /// this node has applied it.
    pub async fn submit(&self, task: StateTask) -> TaskResult {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Submit { task, reply }).map_err(|_| "node is shutting down".to_string())?;
        rx.await.map_err(|_| "node is shutting down".to_string())?
    }

    pub async fn add_voting_config_exclusions(&self, ids: Vec<String>, names: Vec<String>) -> TaskResult {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::AddExclusions { ids, names, reply }).map_err(|_| "node is shutting down".to_string())?;
        rx.await.map_err(|_| "node is shutting down".to_string())?
    }

    pub async fn clear_voting_config_exclusions(&self, wait_for_removal: bool) -> TaskResult {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::ClearExclusions { wait_for_removal, reply }).map_err(|_| "node is shutting down".to_string())?;
        rx.await.map_err(|_| "node is shutting down".to_string())?
    }

    /// Replace the discovery seed addresses (the static
    /// `discovery.seed_hosts` plus what the `file` provider read).
    pub fn set_seed_hosts(&self, hosts: Vec<String>) {
        let _ = self.tx.send(Cmd::SetSeedHosts { hosts });
    }

    pub async fn info(&self) -> Option<CoordInfo> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Info { reply }).ok()?;
        rx.await.ok()
    }
}

/// Outputs that must not run before every `ApplyState` emitted ahead of
/// them has been applied (see the module docs).
fn deferred_until_applied(o: &Output) -> bool {
    matches!(o, Output::TaskCompleted { .. } | Output::PublicationCompleted { .. } | Output::Send { msg: Message::ApplyCommitResponse { .. }, .. })
}

/// The identity of a heartbeat authority, to rebuild it only on change.
type AuthorityKey = (Mode, u64, Option<String>, u64, String, bool, bool);

struct Actor {
    coord: Coordinator,
    persist_tx: std::sync::mpsc::Sender<(u64, PersistUpdate)>,
    persist_enqueued: u64,
    persist_done: u64,
    /// Term of each in-flight `Persist`, by sequence number.
    persist_terms: VecDeque<(u64, u64)>,
    /// The newest term known durable.
    durable_term: u64,
    /// Outputs waiting for a `Persist` sequence number to be durable.
    persist_wait: VecDeque<(u64, Output)>,
    transport: TransportService,
    applier: Arc<dyn Applier>,
    tx: mpsc::UnboundedSender<Cmd>,
    epoch: Instant,
    tasks: HashMap<u64, oneshot::Sender<TaskResult>>,
    heartbeat: Heartbeat,
    authority_key: Option<AuthorityKey>,
    apply_tx: std::sync::mpsc::Sender<(u64, Arc<ClusterState>)>,
    /// Sequence number of the last state queued to the applier.
    apply_enqueued: u64,
    /// Sequence number of the last state the applier finished.
    apply_done: u64,
    /// Held-back outputs with the apply sequence number they wait for.
    deferred: VecDeque<(u64, Output)>,
    publications: PublishTracker,
    /// Peers that answered for the legacy JSON action only, and since when.
    v1_peers: HashMap<String, Instant>,
    faults: Faults,
    stats: Arc<WireStats>,
    /// Since when a coordination-state write has been outstanding.
    persist_since: Option<Instant>,
    /// The segment format floor of the newest committed state this node
    /// applied (or recorded before a restart): `(cluster uuid, floor)`. A
    /// publication may not lower it; an accepted but uncommitted floor does
    /// not bind (`validate_committed_segment_floor`).
    committed_floor: (String, u32),
}

impl Drop for Actor {
    fn drop(&mut self) {
        // No answers on behalf of an actor that is gone.
        *self.heartbeat.write() = None;
    }
}

impl Actor {
    fn now(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// Only committed states are applied: follow their segment format floor.
    fn note_committed_floor(&mut self, state: &ClusterState) {
        let uuid = &state.metadata.cluster_uuid;
        if uuid.is_empty() { return; }
        let floor = elasticsearch_cluster::compatibility::effective_segment_format_floor(&state.metadata);
        if self.committed_floor.0 != *uuid || floor > self.committed_floor.1 {
            self.committed_floor = (uuid.clone(), floor);
        }
    }

    fn validate_committed_floor(&self, next: &ClusterState) -> Result<(), String> {
        elasticsearch_cluster::compatibility::validate_committed_segment_floor(&self.committed_floor.0, self.committed_floor.1, next)
    }

    fn run_outputs(&mut self, outputs: Vec<Output>) {
        for out in outputs {
            self.route(out);
        }
        self.publish_authority();
    }

    /// `Persist` goes to the persistence thread at once (it writes in
    /// order). Every other output waits until every `Persist` emitted before
    /// it is durable: votes, publish acks and commit acks never leave before
    /// the term / state they vouch for is on disk (the Zen2 contract), and
    /// the actor never blocks on an fsync meanwhile.
    fn route(&mut self, out: Output) {
        if let Output::Persist(update) = out {
            if self.persist_done >= self.persist_enqueued {
                self.persist_since = Some(Instant::now());
            }
            self.persist_enqueued += 1;
            self.persist_terms.push_back((self.persist_enqueued, update.current_term));
            if self.persist_tx.send((self.persist_enqueued, update)).is_err() {
                fatal("coordination persistence thread is gone");
            }
            return;
        }
        if self.persist_done < self.persist_enqueued || !self.persist_wait.is_empty() {
            self.persist_wait.push_back((self.persist_enqueued, out));
            return;
        }
        self.dispatch(out);
    }

    /// Run an output whose persistence prerequisites are durable.
    fn dispatch(&mut self, out: Output) {
            match out {
                Output::ApplyState(state) => {
                    self.note_committed_floor(&state);
                    self.apply_enqueued += 1;
                    let s: Arc<ClusterState> = Arc::from(state);
                    if self.apply_tx.send((self.apply_enqueued, s)).is_err() {
                        fatal("cluster applier thread is gone");
                    }
                }
                out if deferred_until_applied(&out) && (self.apply_done < self.apply_enqueued || !self.deferred.is_empty()) => {
                    self.deferred.push_back((self.apply_enqueued, out));
                }
                out => self.execute(out),
            }
    }

    /// The persistence thread made every `Persist` up to `seq` durable.
    fn on_persisted(&mut self, seq: u64) {
        self.persist_done = self.persist_done.max(seq);
        self.persist_since = (self.persist_done < self.persist_enqueued).then(Instant::now);
        while self.persist_terms.front().is_some_and(|(s, _)| *s <= self.persist_done) {
            let (_, term) = self.persist_terms.pop_front().expect("checked");
            self.durable_term = term;
        }
        while self.persist_wait.front().is_some_and(|(wait, _)| *wait <= self.persist_done) {
            let (_, out) = self.persist_wait.pop_front().expect("checked");
            self.dispatch(out);
        }
        self.publish_authority();
    }

    /// Run one output that is neither a `Persist` nor an `ApplyState`.
    fn execute(&mut self, out: Output) {
        match out {
            Output::Persist(_) => debug_assert!(false, "Persist goes to the persistence thread"),
            Output::Send { to, msg } => self.send(to, msg),
            Output::ApplyState(_) => debug_assert!(false, "ApplyState is queued, never executed inline"),
            Output::ModeChanged { mode, term, leader } => {
                tracing::info!(?mode, term, leader = leader.as_ref().map(|l| l.name.as_str()), "coordinator mode changed");
                self.applier.mode_changed(mode, term, leader.as_ref());
            }
            Output::PublicationCompleted { publication_id, version, result } => {
                if let Err(e) = &result {
                    tracing::warn!(publication_id, version, "publication failed: {e}");
                    self.applier.publication_failed();
                }
            }
            Output::TaskCompleted { task_id, result } => {
                if let Some(r) = self.tasks.remove(&task_id) {
                    let _ = r.send(result);
                }
            }
        }
    }

    fn on_applied(&mut self, seq: u64, ok: bool) {
        self.apply_done = self.apply_done.max(seq);
        if !ok {
            // The application panicked: the apply-commit acks it vouched for
            // report a failure instead.
            for (wait, out) in self.deferred.iter_mut() {
                if *wait <= seq {
                    if let Output::Send { msg: Message::ApplyCommitResponse { result, .. }, .. } = out {
                        *result = Err("failed to apply the committed cluster state".into());
                    }
                }
            }
        }
        while self.deferred.front().is_some_and(|(wait, _)| *wait <= self.apply_done) {
            let (_, out) = self.deferred.pop_front().expect("checked");
            self.execute(out);
        }
    }

    /// Publish the fault-detection authority for the transport handler.
    /// Only a term that is already durable is vouched for.
    fn persist_stalled(&self) -> bool {
        persist_stalled_at(self.persist_since, Instant::now())
    }

    fn publish_authority(&mut self) {
        let durable = self.durable_term == self.coord.current_term();
        let stalled = self.persist_stalled();
        let ms = self.coord.master_state();
        let key: AuthorityKey = (self.coord.mode(), self.coord.current_term(), self.coord.leader().map(|l| l.ephemeral_id.clone()), ms.version, ms.state_uuid.clone(), durable, stalled);
        if self.authority_key.as_ref() == Some(&key) {
            return;
        }
        let authority = (durable && !stalled).then(|| HeartbeatAuthority {
            mode: self.coord.mode(),
            term: self.coord.current_term(),
            local_ephemeral_id: self.coord.local_node().ephemeral_id.clone(),
            leader: self.coord.leader().cloned(),
            members: ms.nodes.iter().map(|(id, node)| (id.clone(), node.clone())).collect(),
        });
        *self.heartbeat.write() = authority;
        self.authority_key = Some(key);
    }

    /// Whether `to` is spoken to in the legacy JSON format.
    fn is_v1(&self, to: &str) -> bool {
        self.faults.legacy_wire.load(std::sync::atomic::Ordering::SeqCst) || still_v1(self.v1_peers.get(to), Instant::now())
    }

    fn send(&mut self, to: String, msg: Message) {
        if matches!(msg, Message::PublishRequest { .. }) && self.faults.drop_publications.load(std::sync::atomic::Ordering::SeqCst) {
            tracing::debug!(%to, "fault injection: publish request dropped");
            return;
        }
        let request_id = msg.request_id();
        let (body, fallback) = match msg {
            Message::PublishRequest { request_id: rid, state } => {
                self.publications.sending(&state, rid, &to);
                match self.publications.diff_for(&to).filter(|_| !self.is_v1(&to)) {
                    // The full state stays the v1 fallback.
                    Some(diff) => (Body::PublishDiff { request_id: rid, diff: Box::new((*diff).clone()) }, Some(Message::PublishRequest { request_id: rid, state })),
                    None => (Body::Message(Message::PublishRequest { request_id: rid, state }), None),
                }
            }
            other => (Body::Message(other), None),
        };
        self.send_body(to, body, request_id, fallback);
    }

    /// Send on the peer's wire version. A v2 send the peer has no handler
    /// for is re-sent as v1 (`fallback` replaces a diff) and the peer is
    /// remembered as v1.
    fn send_body(&self, to: String, body: Body, request_id: Option<u64>, fallback: Option<Message>) {
        use std::sync::atomic::Ordering;
        let from = self.transport.publish_address().to_string();
        let transport = self.transport.clone();
        let tx = self.tx.clone();
        let stats = self.stats.clone();
        if self.is_v1(&to) {
            stats.legacy_messages.fetch_add(1, Ordering::Relaxed);
            if matches!(fallback, Some(Message::PublishRequest { .. })) || matches!(body, Body::Message(Message::PublishRequest { .. })) {
                stats.full_states.fetch_add(1, Ordering::Relaxed);
            }
            let msg = match (body, fallback) {
                (_, Some(full)) => full,
                (Body::Message(m), None) => m,
                (_, None) => return,
            };
            let bytes = match encode_v1(from, msg) {
                Ok(b) => b,
                Err(e) => return tracing::error!("cannot encode coordination message: {e}"),
            };
            tokio::spawn(async move {
                if let (Err(e), Some(request_id)) = (send_to(&transport, &to, ACTION, bytes).await, request_id) {
                    let _ = tx.send(Cmd::SendFailed { request_id, reason: e.reason, disconnected: e.disconnected });
                }
            });
            return;
        }
        match &body {
            Body::PublishDiff { .. } => {
                stats.diffs.fetch_add(1, Ordering::Relaxed);
            }
            Body::Message(Message::PublishRequest { .. }) => {
                stats.full_states.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let env = EnvelopeV2 { from, body };
        let bytes = match encode_v2(&env) {
            Ok(b) => b,
            Err(e) => return tracing::error!("cannot encode coordination message: {e}"),
        };
        tokio::spawn(async move {
            let mut res = send_to(&transport, &to, ACTION_V2, bytes).await;
            if res.as_ref().is_err_and(|e| e.downgrade) {
                // An older node: legacy JSON for a while.
                stats.downgrades.fetch_add(1, Ordering::Relaxed);
                stats.legacy_messages.fetch_add(1, Ordering::Relaxed);
                let _ = tx.send(Cmd::PeerIsV1 { addr: to.clone() });
                let msg = match (env.body, fallback) {
                    (_, Some(full)) => Some(full),
                    (Body::Message(m), None) => Some(m),
                    (_, None) => None,
                };
                if let Some(msg) = msg {
                    res = match encode_v1(env.from, msg) {
                        Ok(b) => send_to(&transport, &to, ACTION, b).await,
                        Err(reason) => Err(DeliveryError { reason, downgrade: false, disconnected: false }),
                    };
                }
            }
            if let (Err(e), Some(request_id)) = (res, request_id) {
                let _ = tx.send(Cmd::SendFailed { request_id, reason: e.reason, disconnected: e.disconnected });
            }
        });
    }

    fn handle(&mut self, cmd: Cmd) {
        let now = self.now();
        let capability_transport = self.transport.clone();
        match cmd {
            Cmd::Inbound { from, msg } => {
                let floor = match &msg {
                    Message::PublishRequest { state, .. } => self.validate_committed_floor(state),
                    _ => Ok(()),
                };
                if let Err(reason) = floor.and_then(|()| validate_capability_message(self.coord.last_accepted_state(), &msg)) {
                    tracing::warn!(%from, %reason, "refusing incompatible coordination operation");
                    if let Some(response) = capability_failure_response(&msg, reason) { self.send(from, response); }
                    return;
                }
                if let Message::PublishResponse { result: Ok(r), .. } = &msg {
                    self.publications.accepted(&from, r.version);
                }
                // A node whose coordination-state writes hang fails checks.
                if self.persist_stalled() {
                    let unhealthy = || Err("node is unhealthy: coordination state writes have stalled".to_string());
                    match &msg {
                        Message::LeaderCheck { request_id, .. } => return self.send(from, Message::LeaderCheckResponse { request_id: *request_id, result: unhealthy() }),
                        Message::FollowerCheck { request_id, .. } => return self.send(from, Message::FollowerCheckResponse { request_id: *request_id, result: unhealthy() }),
                        _ => {}
                    }
                }
                let out = self.coord.handle_message(now, &from, msg);
                self.run_outputs(out);
            }
            Cmd::InboundDiff { from, request_id, diff } => {
                let full = diff.apply(self.coord.last_accepted_state());
                match full {
                    Ok(state) => {
                        if let Err(reason) = self.validate_committed_floor(&state)
                            .and_then(|()| crate::internal_capabilities::validate_lifecycle_publication(self.coord.last_accepted_state(), &state)) {
                            self.send(from, Message::PublishResponse { request_id, result: Err(reason) });
                            return;
                        }
                        // The live publication authority is verified separately
                        // from relayed/offline members. Only the master task
                        // that activates a feature requires every member proof.
                        if let Err(reason) = crate::capability_probe::require_publication_proof(&self.transport, &state) {
                            self.send(from, Message::PublishResponse { request_id, result: Err(reason) });
                            return;
                        }
                        let out = self.coord.handle_message(now, &from, Message::PublishRequest { request_id, state: Box::new(state) });
                        self.run_outputs(out);
                    }
                    Err(e) => {
                        tracing::debug!(%from, request_id, "cannot apply the published diff, asking for the full state: {e}");
                        self.send_body(from, Body::ResendFull { request_id }, None, None);
                    }
                }
            }
            Cmd::ResendFull { from, request_id } => {
                let full = self.publications.requests.get(&request_id).cloned().zip(self.publications.current.clone());
                match full {
                    Some((to, state)) => {
                        let msg = Message::PublishRequest { request_id, state: Box::new((*state).clone()) };
                        self.stats.resent_full.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.send_body(to, Body::Message(msg), Some(request_id), None);
                    }
                    None => tracing::debug!(%from, request_id, "full state requested for a finished publication"),
                }
            }
            Cmd::PeerIsV1 { addr } => {
                if self.v1_peers.insert(addr.clone(), Instant::now()).is_none() {
                    tracing::info!(%addr, "peer speaks the legacy coordination wire format");
                }
            }
            Cmd::SendFailed { request_id, reason, disconnected } => {
                let out = if disconnected {
                    self.coord.handle_send_failure(now, request_id, &reason)
                } else {
                    self.coord.handle_request_failure(now, request_id, &reason)
                };
                self.run_outputs(out);
            }
            Cmd::Disconnected { addr } => {
                // The event is asynchronous: ignore it if a later send
                // already re-established the connection.
                if !self.transport.is_connected(addr) {
                    tracing::debug!(%addr, "transport connection closed");
                    // A peer that reconnects (restarted, maybe upgraded) is
                    // probed in the binary format again.
                    self.v1_peers.remove(&addr.to_string());
                    let out = self.coord.handle_disconnect(now, &addr.to_string());
                    self.run_outputs(out);
                }
            }
            Cmd::Submit { task, reply } => match self.coord.submit_task(now, Box::new(move |current| {
                let next = task(current)?;
                crate::internal_capabilities::validate_lifecycle_publication(current, &next)?;
                elasticsearch_cluster::compatibility::validate_segment_floor_publication(current, &next)?;
                crate::capability_probe::require_activation_proofs(&capability_transport, &next)?;
                Ok(next)
            })) {
                Ok(s) => {
                    self.tasks.insert(s.id, reply);
                    self.run_outputs(s.outputs);
                }
                Err(e) => {
                    let _ = reply.send(Err(not_master(e)));
                }
            },
            Cmd::AddExclusions { ids, names, reply } => match self.coord.add_voting_config_exclusions(now, ids, names) {
                Ok(s) => {
                    self.tasks.insert(s.id, reply);
                    self.run_outputs(s.outputs);
                }
                Err(e) => {
                    let _ = reply.send(Err(not_master(e)));
                }
            },
            Cmd::ClearExclusions { wait_for_removal, reply } => match self.coord.clear_voting_config_exclusions(now, wait_for_removal) {
                Ok(s) => {
                    self.tasks.insert(s.id, reply);
                    self.run_outputs(s.outputs);
                }
                Err(e) => {
                    let _ = reply.send(Err(not_master(e)));
                }
            },
            Cmd::Info { reply } => {
                let _ = reply.send(CoordInfo {
                    mode: self.coord.mode(),
                    term: self.coord.current_term(),
                    leader: self.coord.leader().cloned(),
                    discovered: self.coord.discovered_peers(),
                    publishing: self.coord.is_publishing(),
                });
            }
            Cmd::SetSeedHosts { hosts } => {
                if self.coord.seed_hosts() != hosts.as_slice() {
                    tracing::info!(?hosts, "seed hosts changed");
                    self.coord.set_seed_hosts(hosts);
                }
            }
            Cmd::Applied { seq, ok } => self.on_applied(seq, ok),
            Cmd::Persisted { seq } => self.on_persisted(seq),
        }
    }
}

/// The coordination persistence thread: writes `Persist` outputs in order
/// (fsynced), coalescing a backlog into one write, and reports each
/// durable sequence number. A node that cannot persist must die: acting on
/// later outputs without durability would break safety.
fn spawn_persister(mut gateway: Gateway, rx: std::sync::mpsc::Receiver<(u64, PersistUpdate)>, tx: mpsc::UnboundedSender<Cmd>) {
    std::thread::Builder::new()
        .name("cluster-persist".into())
        .spawn(move || {
            while let Ok((mut seq, update)) = rx.recv() {
                let mut updates = vec![update];
                while let Ok((s, u)) = rx.try_recv() {
                    seq = s;
                    updates.push(u);
                }
                // A transient failure (no file descriptor left, say) is retried
                // briefly; the state stays unacknowledged meanwhile. A write
                // that keeps failing stops the node, as ES fails a node whose
                // coordination state cannot be persisted.
                let mut attempt = 0;
                while let Err(e) = gateway.persist_all(&updates) {
                    attempt += 1;
                    if attempt >= 5 {
                        fatal(&format!("failed to persist coordination state: {e}"));
                    }
                    tracing::warn!("failed to persist coordination state (attempt {attempt}): {e}; retrying");
                    std::thread::sleep(Duration::from_millis(100 << attempt));
                }
                if tx.send(Cmd::Persisted { seq }).is_err() {
                    break;
                }
            }
        })
        .expect("spawn the coordination persistence thread");
}

/// The cluster applier thread: applies queued states in order, the newest
/// of a backlog only, and reports each finished sequence number.
fn spawn_applier(applier: Arc<dyn Applier>, rx: std::sync::mpsc::Receiver<(u64, Arc<ClusterState>)>, tx: mpsc::UnboundedSender<Cmd>) {
    let runtime = tokio::runtime::Handle::current();
    std::thread::Builder::new()
        .name("cluster-applier".into())
        .spawn(move || {
            // State application spawns recovery / report tasks.
            let _runtime = runtime.enter();
            while let Ok((mut seq, mut state)) = rx.recv() {
                // A committed state supersedes the ones queued before it.
                while let Ok((s, st)) = rx.try_recv() {
                    seq = s;
                    state = st;
                }
                let version = state.version;
                let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| applier.apply(state))).is_ok();
                if !ok {
                    tracing::error!(version, "cluster state application panicked");
                }
                if tx.send(Cmd::Applied { seq, ok }).is_err() {
                    break;
                }
            }
        })
        .expect("spawn the cluster applier thread");
}

/// How long a coordination message may wait for a connection to a peer we
/// are not connected to. ES never dials inside a send: a request to an
/// unconnected node fails at once with `NodeNotConnectedException`, which
/// fault detection treats as a disconnect. Here connections are opened on
/// demand, so the dial gets a short bound instead of the 30s
/// `transport.connect_timeout`: a dead peer whose address black-holes
/// connects (killed container, powered-off host) must not stall leader /
/// follower checks and publications for 30s each. 2s tolerates one lost SYN.
/// Replica operations use the same bound (`node.rs`, `TransportNet`).
pub(crate) const CONNECT_GRACE: Duration = Duration::from_secs(2);

/// Why a coordination message was not delivered.
struct DeliveryError {
    reason: String,
    /// Only a connection failure bypasses the fault detector retry count.
    disconnected: bool,
    /// The peer cannot take the binary format (no handler for the action,
    /// an older node, or a format version it does not decode): retry in
    /// the legacy JSON format.
    downgrade: bool,
}

impl From<elasticsearch_transport::TransportError> for DeliveryError {
    fn from(e: elasticsearch_transport::TransportError) -> Self {
        let downgrade = match &e {
            elasticsearch_transport::TransportError::ActionNotFound { .. } => true,
            elasticsearch_transport::TransportError::Remote(r) => r.error_type == UNSUPPORTED_FORMAT,
            _ => false,
        };
        let disconnected = matches!(&e,
            elasticsearch_transport::TransportError::ConnectTransport { .. }
            | elasticsearch_transport::TransportError::HandshakeFailed { .. }
            | elasticsearch_transport::TransportError::NodeDisconnected { .. }
            | elasticsearch_transport::TransportError::Closed
        );
        DeliveryError { reason: e.to_string(), downgrade, disconnected }
    }
}

/// Deliver one coordination message. An established connection is used as
/// is (a hung peer is caught by the coordinator's own request timeouts);
/// otherwise the dial + delivery is bounded by [`CONNECT_GRACE`].
async fn deliver(transport: &TransportService, addr: SocketAddr, action: &str, body: Bytes) -> Result<(), DeliveryError> {
    let connected = transport.is_connected(addr);
    let send = transport.send_request_bytes(addr, action, body, Duration::from_secs(30));
    if connected {
        return send.await.map(|_| ()).map_err(DeliveryError::from);
    }
    match tokio::time::timeout(CONNECT_GRACE, send).await {
        Ok(r) => r.map(|_| ()).map_err(DeliveryError::from),
        Err(_) => Err(DeliveryError { reason: format!("[{addr}] connect_exception: not connected, connect timed out after {CONNECT_GRACE:?}"), downgrade: false, disconnected: true }),
    }
}

/// Deliver to a transport address, or to every address a seed host name
/// resolves to.
async fn send_to(transport: &TransportService, to: &str, action: &str, body: Bytes) -> Result<(), DeliveryError> {
    match to.parse::<SocketAddr>() {
        Ok(addr) => deliver(transport, addr, action, body).await,
        Err(_) => resolve_and_send(transport, to, action, body).await,
    }
}

/// Seed hosts may be DNS names (Kubernetes headless services resolve to
/// one A record per pod); try every address.
async fn resolve_and_send(transport: &TransportService, to: &str, action: &str, body: Bytes) -> Result<(), DeliveryError> {
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host(to).await.map_err(|e| DeliveryError { reason: format!("failed to resolve [{to}]: {e}"), downgrade: false, disconnected: true })?.collect();
    // Probe every address (ES does): a headless service resolves to one
    // A record per pod. Success if any delivery succeeded.
    let sends = addrs.into_iter().filter(|a| !transport.is_local_address(*a)).map(|a| {
        let body = body.clone();
        async move { deliver(transport, a, action, body).await }
    });
    let results = futures::future::join_all(sends).await;
    if results.iter().any(|r| r.is_ok()) {
        return Ok(());
    }
    Err(results.into_iter().find_map(|r| r.err()).unwrap_or_else(|| DeliveryError { reason: format!("no addresses for [{to}]"), downgrade: false, disconnected: true }))
}

/// Start the actor and the cluster applier; registers the transport handler.
pub fn start(coord: Coordinator, gateway: Gateway, transport: TransportService, applier: Arc<dyn Applier>, faults: Faults) -> CoordHandle {
    let (tx, mut rx) = mpsc::unbounded_channel::<Cmd>();
    let stats = Arc::new(WireStats::default());
    let handle = CoordHandle { tx: tx.clone(), stats: stats.clone() };
    let heartbeat: Heartbeat = Arc::new(parking_lot::RwLock::new(None));
    {
        // Legacy JSON (older nodes during a rolling upgrade).
        let tx = tx.clone();
        let heartbeat = heartbeat.clone();
        let responder = transport.clone();
        transport.register_raw_handler(ACTION, move |body: Bytes| {
            let tx = tx.clone();
            let heartbeat = heartbeat.clone();
            let responder = responder.clone();
            async move {
                let env: Envelope = serde_json::from_slice(&body).map_err(|e| RemoteError::new("illegal_argument_exception", format!("bad coordination message: {e}"), 400))?;
                let response = heartbeat.read().as_ref().and_then(|a| a.response(&env.from, &env.msg));
                if let Some(msg) = response {
                    let body = encode_v1(responder.publish_address().to_string(), msg).map_err(|e| RemoteError::new("internal_exception", e, 500))?;
                    // Use the normal reply transport, independently of the
                    // actor. A send failure is not a successful check.
                    tokio::spawn(async move {
                        let _ = send_to(&responder, &env.from, ACTION, body).await;
                    });
                    return Ok(Bytes::new());
                }
                verify_message_capabilities(&responder, &env.from, &env.msg).await?;
                tx.send(Cmd::Inbound { from: env.from, msg: env.msg }).map_err(|_| RemoteError::new("node_closed_exception", "node closed", 503))?;
                Ok(Bytes::new())
            }
        });
    }
    {
        let tx = tx.clone();
        let heartbeat = heartbeat.clone();
        let responder = transport.clone();
        transport.register_raw_handler(ACTION_V2, move |body: Bytes| {
            let tx = tx.clone();
            let heartbeat = heartbeat.clone();
            let responder = responder.clone();
            async move {
                let env: EnvelopeV2 = decode_v2(&body)?;
                let closed = |_| RemoteError::new("node_closed_exception", "node closed", 503);
                match env.body {
                    Body::Message(msg) => {
                        let response = heartbeat.read().as_ref().and_then(|a| a.response(&env.from, &msg));
                        if let Some(reply) = response {
                            let from = responder.publish_address().to_string();
                            let body = encode_v2(&EnvelopeV2 { from: from.clone(), body: Body::Message(reply.clone()) })
                                .map_err(|e| RemoteError::new("internal_exception", e, 500))?;
                            tokio::spawn(async move {
                                // A peer without the binary action gets the
                                // legacy format (as the actor's sends do).
                                if send_to(&responder, &env.from, ACTION_V2, body).await.is_err_and(|e| e.downgrade) {
                                    if let Ok(body) = encode_v1(from, reply) {
                                        let _ = send_to(&responder, &env.from, ACTION, body).await;
                                    }
                                }
                            });
                            return Ok(Bytes::new());
                        }
                        verify_message_capabilities(&responder, &env.from, &msg).await?;
                        tx.send(Cmd::Inbound { from: env.from, msg }).map_err(closed)?;
                    }
                    Body::PublishDiff { request_id, diff } => {
                        crate::capability_probe::verify_sender(&responder, &env.from).await
                            .map_err(|reason| RemoteError::new("unsupported_operation_exception", reason, 503))?;
                        if let Some(nodes) = &diff.nodes {
                            crate::capability_probe::observe_members(&responder, nodes).await;
                        }
                        tx.send(Cmd::InboundDiff { from: env.from, request_id, diff }).map_err(closed)?;
                    }
                    Body::ResendFull { request_id } => tx.send(Cmd::ResendFull { from: env.from, request_id }).map_err(closed)?,
                }
                Ok(Bytes::new())
            }
        });
    }
    // A closed connection fails fault detection at once (ES's disconnect
    // listeners) instead of waiting for the next check to fail.
    {
        let tx = tx.clone();
        let mut events = transport.subscribe();
        tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(ConnectionEvent::Disconnected { addr, .. }) => {
                        if tx.send(Cmd::Disconnected { addr }).is_err() {
                            break;
                        }
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }
    let (apply_tx, apply_rx) = std::sync::mpsc::channel::<(u64, Arc<ClusterState>)>();
    spawn_applier(applier.clone(), apply_rx, tx.clone());
    let durable_term = gateway.persisted().current_term;
    // Unreadable is refused at node start, before the coordinator exists.
    let committed_floor = gateway.committed_segment_format().ok().flatten()
        .unwrap_or_else(|| (String::new(), elasticsearch_cluster::compatibility::BASELINE_SEGMENT_FORMAT));
    let (persist_tx, persist_rx) = std::sync::mpsc::channel::<(u64, PersistUpdate)>();
    spawn_persister(gateway, persist_rx, tx.clone());
    let mut actor = Actor {
        coord,
        persist_tx,
        persist_enqueued: 0,
        persist_done: 0,
        persist_terms: VecDeque::new(),
        durable_term,
        persist_wait: VecDeque::new(),
        transport,
        applier,
        tx,
        epoch: Instant::now(),
        tasks: HashMap::new(),
        heartbeat,
        authority_key: None,
        apply_tx,
        apply_enqueued: 0,
        apply_done: 0,
        deferred: VecDeque::new(),
        publications: PublishTracker::default(),
        v1_peers: Default::default(),
        faults,
        stats,
        persist_since: None,
        committed_floor,
    };
    tokio::spawn(async move {
        let out = actor.coord.start(actor.now());
        actor.run_outputs(out);
        // Tick at the coordinator's next deadline, at least every 50ms.
        let mut tick = tokio::time::interval(Duration::from_millis(50));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut formation = FormationWarning::default();
        loop {
            tokio::select! {
                cmd = rx.recv() => match cmd {
                    Some(c) => actor.handle(c),
                    None => break,
                },
                _ = tick.tick() => {
                    // A stalled persist can start or end without any message.
                    actor.publish_authority();
                    let now = actor.now();
                    if actor.coord.next_deadline().map(|d| d <= now).unwrap_or(false) {
                        let out = actor.coord.tick(now);
                        actor.run_outputs(out);
                    }
                    formation.check(&actor.coord);
                }
            }
        }
    });
    handle
}

/// ES's `ClusterFormationFailureHelper`: while the node has no master, say
/// every 10 s why (a candidate that can never win an election, e.g. one
/// whose persisted voting configuration does not name it, used to sit
/// silently in "master not discovered").
#[derive(Default)]
struct FormationWarning {
    since: Option<Instant>,
    last: Option<Instant>,
}

impl FormationWarning {
    const AFTER: Duration = Duration::from_secs(10);

    fn check(&mut self, coord: &Coordinator) {
        if coord.mode() != Mode::Candidate {
            self.since = None;
            self.last = None;
            return;
        }
        let now = Instant::now();
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since) < Self::AFTER || self.last.is_some_and(|l| now.duration_since(l) < Self::AFTER) {
            return;
        }
        self.last = Some(now);
        let state = coord.last_accepted_state();
        let local = coord.local_node();
        let config: Vec<String> = state.coordination.last_accepted_config.0.iter().cloned().collect();
        let discovered: Vec<String> = coord.discovered_peers().iter().map(|n| format!("{}{{{}}}", n.name, n.id)).collect();
        let in_config = config.is_empty() || config.iter().any(|id| id == &local.id);
        tracing::warn!(
            term = coord.current_term(), local = %format!("{}{{{}}}", local.name, local.id), voting_config = ?config, ?discovered,
            "master not discovered or elected yet{}",
            if in_config { "" } else { "; this node is not in the last accepted voting configuration and cannot elect itself" }
        );
    }
}

#[cfg(test)]
mod heartbeat_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn delivery_timeout_preserves_fault_detection_retries() {
        use elasticsearch_transport::{RemoteError, TransportError};
        let addr = "127.0.0.1:9300".parse().unwrap();
        let timeout = DeliveryError::from(TransportError::ReceiveTimeout {
            addr, action: ACTION.into(), request_id: 1, timeout_ms: 30_000,
        });
        assert!(!timeout.disconnected);
        assert!(!DeliveryError::from(TransportError::Remote(RemoteError::internal("busy"))).disconnected);
        assert!(DeliveryError::from(TransportError::NodeDisconnected { addr, action: ACTION.into() }).disconnected);
        assert!(DeliveryError::from(TransportError::ConnectTransport { addr, reason: "refused".into() }).disconnected);
    }

    fn peer(id: &str) -> DiscoveryNode {
        DiscoveryNode {
            id: id.into(), name: id.into(), ephemeral_id: format!("{id}-boot"),
            transport_address: format!("{id}:9300"), http_address: String::new(),
            host: id.into(), ip: String::new(), roles: Default::default(),
            attributes: Default::default(), version: String::new(),
        }
    }
    fn authority(mode: Mode) -> HeartbeatAuthority {
        HeartbeatAuthority { mode, term: 7, local_ephemeral_id: "local-boot".into(),
            leader: Some(peer("master")), members: [("member".into(), peer("member"))].into() }
    }
    fn check(term: u64) -> Message {
        Message::FollowerCheck { request_id: 42, source: peer("master"), term,
            target_ephemeral_id: "local-boot".into() }
    }

    #[test]
    fn heartbeat_only_accepts_established_term_and_incarnation() {
        let a = authority(Mode::Follower);
        assert!(matches!(a.response("master:9300", &check(7)),
            Some(Message::FollowerCheckResponse { request_id: 42, result: Ok(()) })));
        for term in [6, 8] { assert!(a.response("master:9300", &check(term)).is_none()); }
        assert!(a.response("other:9300", &check(7)).is_none());
        for mode in [Mode::Candidate, Mode::Leader] {
            assert!(authority(mode).response("master:9300", &check(7)).is_none());
        }
        let mut request = check(7);
        if let Message::FollowerCheck { source, .. } = &mut request { source.ephemeral_id = "old-boot".into(); }
        assert!(a.response("master:9300", &request).is_none());
        let mut request = check(7);
        if let Message::FollowerCheck { target_ephemeral_id, .. } = &mut request { *target_ephemeral_id = "old-local".into(); }
        assert!(a.response("master:9300", &request).is_none());
        assert!(a.response("master:9300", &Message::FollowerCheckResponse { request_id: 1, result: Ok(()) }).is_none());
    }

    #[test]
    fn leader_heartbeat_requires_current_member() {
        let a = authority(Mode::Leader);
        let mut request = Message::LeaderCheck { request_id: 5, source: peer("member"), target_ephemeral_id: "local-boot".into() };
        assert!(matches!(a.response("member:9300", &request), Some(Message::LeaderCheckResponse { request_id: 5, result: Ok(()) })));
        if let Message::LeaderCheck { source, .. } = &mut request { *source = peer("removed"); }
        assert!(a.response("removed:9300", &request).is_none());
    }

    #[test]
    fn apply_acks_and_completions_wait_for_the_applier() {
        let ack = Output::Send { to: "m:9300".into(), msg: Message::ApplyCommitResponse { request_id: 1, result: Err("x".into()) } };
        assert!(deferred_until_applied(&ack));
        assert!(deferred_until_applied(&Output::TaskCompleted { task_id: 1, result: Ok(3) }));
        let check = Output::Send { to: "m:9300".into(), msg: Message::LeaderCheckResponse { request_id: 1, result: Ok(()) } };
        assert!(!deferred_until_applied(&check), "fault detection never waits for state application");
    }

    fn member(id: &str) -> DiscoveryNode {
        let mut n = peer(id);
        n.transport_address = format!("127.0.0.1:{}", 9300 + id.len());
        n
    }

    fn published(version: u64) -> ClusterState {
        let mut s = ClusterState::empty("c");
        s.version = version;
        s.state_uuid = format!("uuid-{version}");
        s.nodes.insert("a".into(), member("a"));
        s.metadata.customs.insert("k".into(), serde_json::json!({"v": version}));
        s
    }

    #[test]
    fn v2_envelopes_round_trip_with_binary_states_and_diffs() {
        let base = published(3);
        let next = published(4);
        let env = EnvelopeV2 { from: "127.0.0.1:9300".into(), body: Body::Message(Message::PublishRequest { request_id: 9, state: Box::new(next.clone()) }) };
        let back: EnvelopeV2 = decode_v2(&encode_v2(&env).unwrap()).unwrap();
        match back.body {
            Body::Message(Message::PublishRequest { request_id: 9, state }) => assert_eq!(*state, next),
            _ => panic!("publish request expected"),
        }
        let diff = ClusterStateDiff::between(&base, &next);
        let env = EnvelopeV2 { from: "x".into(), body: Body::PublishDiff { request_id: 1, diff: Box::new(diff) } };
        let back: EnvelopeV2 = decode_v2(&encode_v2(&env).unwrap()).unwrap();
        match back.body {
            Body::PublishDiff { request_id: 1, diff } => assert_eq!(diff.apply(&base).unwrap(), next),
            _ => panic!("diff expected"),
        }
        // The legacy envelope is unchanged JSON.
        let v1: serde_json::Value = serde_json::from_slice(&encode_v1("x".into(), Message::LeaderCheckResponse { request_id: 2, result: Ok(()) }).unwrap()).unwrap();
        assert_eq!(v1["from"], "x");
    }

    /// The v2 body is the format byte, then bincode (fixint, little
    /// endian): pinned, so a change of the encoding is a deliberate format
    /// bump (an older node answers it with `UNSUPPORTED_FORMAT` and gets
    /// JSON) rather than an accident.
    #[test]
    fn v2_envelope_bytes_are_pinned_and_versioned() {
        let env = EnvelopeV2 { from: "a".into(), body: Body::Message(Message::LeaderCheckResponse { request_id: 7, result: Ok(()) }) };
        let bytes = encode_v2(&env).unwrap();
        let golden: Vec<u8> = [
            vec![WIRE_FORMAT],
            vec![1, 0, 0, 0, 0, 0, 0, 0, b'a'], // from: u64 length, bytes
            vec![0, 0, 0, 0],                   // Body::Message
            vec![12, 0, 0, 0],                  // Message::LeaderCheckResponse
            vec![7, 0, 0, 0, 0, 0, 0, 0],       // request_id
            vec![0, 0, 0, 0],                   // Ok(())
        ].concat();
        assert_eq!(bytes.to_vec(), golden);
        let mut future = golden.clone();
        future[0] = WIRE_FORMAT + 1;
        assert_eq!(decode_v2(&future).err().map(|e| e.error_type.clone()), Some(UNSUPPORTED_FORMAT.to_string()), "another format version is refused, not misread");
        let mut garbage = golden.clone();
        garbage.truncate(5);
        assert_eq!(decode_v2(&garbage).err().map(|e| e.error_type.clone()), Some(UNSUPPORTED_FORMAT.to_string()));
        let remote = elasticsearch_transport::TransportError::Remote(RemoteError::new(UNSUPPORTED_FORMAT, "x", 400));
        assert!(DeliveryError::from(remote).downgrade, "the sender falls back to JSON");
    }

    /// Review round 2: the persist-stall gate and the legacy re-probe.
    #[test]
    fn stall_gate_and_legacy_reprobe_follow_their_clocks() {
        let now = Instant::now();
        let ago = |d: Duration| now.checked_sub(d).expect("monotonic clock far enough from its origin");
        assert!(!persist_stalled_at(None, now), "nothing outstanding");
        assert!(!persist_stalled_at(Some(ago(Duration::from_secs(1))), now));
        assert!(persist_stalled_at(Some(ago(PERSIST_STALL + Duration::from_secs(1))), now), "a stuck write fails the checks");
        assert!(!still_v1(None, now), "an unknown peer gets the binary format");
        assert!(still_v1(Some(&ago(Duration::from_secs(1))), now));
        assert!(!still_v1(Some(&ago(V1_RECHECK + Duration::from_secs(1))), now), "a legacy peer is probed again");
    }

    /// Review NM8: the whole binary schema of a publication is pinned. The
    /// fixture fills every part of a cluster state (nodes, coordination,
    /// metadata with an index, templates, settings and customs, routing
    /// with copies, blocks) and a diff between two of them. Any change to
    /// these types changes the bytes: bump [`WIRE_FORMAT`] and update the
    /// pin (an older node then refuses the new format and is spoken to in
    /// JSON, instead of misreading it). Trailing bytes are refused.
    #[test]
    fn the_publication_schema_is_pinned_to_the_wire_format() {
        use elasticsearch_cluster::state::{AllocationId, ClusterBlock, IndexRoutingTable, IndexShardRoutingTable, RecoverySource, ShardRouting, ShardState, UnassignedInfo, UnassignedReason, VotingConfigExclusion, VotingConfiguration};
        // Every Option set, every collection non-empty; `rich(4)` differs
        // from `rich(3)` in every part a diff carries (nodes, templates,
        // settings, customs, an index removed and one changed, routing,
        // blocks), so every diff variant is encoded.
        fn rich(version: u64) -> ClusterState {
            let mut s = published(version);
            s.master_node_id = Some("a".into());
            s.nodes.insert(format!("n{version}"), member(&format!("n{version}")));
            s.coordination.term = version;
            s.coordination.last_committed_config = VotingConfiguration::new(["a".to_string()]);
            s.coordination.last_accepted_config = VotingConfiguration::new(["a".to_string(), format!("n{version}")]);
            s.coordination.voting_config_exclusions = vec![VotingConfigExclusion { node_id: "x".into(), node_name: "x".into() }];
            s.metadata.cluster_uuid = "cu".into();
            s.metadata.cluster_uuid_committed = true;
            s.metadata.version = version;
            let mut im: elasticsearch_cluster::state::IndexMetadata = serde_json::from_value(serde_json::json!({
                "name": "i", "uuid": "iu", "version": version, "mapping_version": 2, "settings_version": 3, "aliases_version": 4,
                "state": "open", "settings": {"index.number_of_shards": "1"}, "mappings": {"properties": {"f": {"type": "keyword"}}},
                "aliases": {"al": {"is_write_index": true}}, "number_of_shards": 1, "number_of_replicas": 1, "routing_num_shards": 1,
                "primary_terms": [5], "in_sync_allocation_ids": {"0": ["p"]}, "creation_date": 7,
                "lifecycle": {"ilm": {"phase": "hot"}}
            })).unwrap();
            im.version = version;
            s.metadata.indices.insert("i".into(), im.clone());
            // Present in one state only: an index removal / addition.
            let mut gone = im;
            gone.name = format!("only-{version}");
            s.metadata.indices.insert(gone.name.clone(), gone);
            s.metadata.index_templates.insert(format!("t{version}"), serde_json::json!({"index_patterns": ["i*"], "v": version}));
            s.metadata.component_templates.insert(format!("c{version}"), serde_json::json!({"template": {}}));
            s.metadata.legacy_templates.insert(format!("l{version}"), serde_json::json!({"order": version}));
            s.metadata.persistent_settings.insert("p".into(), serde_json::json!(version.to_string()));
            s.metadata.transient_settings.insert("t".into(), serde_json::json!(version.to_string()));
            // Every variant of each routing enum, one copy each.
            let sources = [
                RecoverySource::EmptyStore,
                RecoverySource::ExistingStore { bootstrap_new_history_uuid: true },
                RecoverySource::Peer,
                RecoverySource::Snapshot { repository: "repo".into(), snapshot: "snap".into(), index: "i".into() },
                RecoverySource::LocalShards,
            ];
            let reasons = [
                UnassignedReason::IndexCreated, UnassignedReason::ClusterRecovered, UnassignedReason::IndexReopened,
                UnassignedReason::DanglingIndexImported, UnassignedReason::NewIndexRestored, UnassignedReason::ExistingIndexRestored,
                UnassignedReason::ReplicaAdded, UnassignedReason::AllocationFailed, UnassignedReason::NodeLeft,
                UnassignedReason::RerouteCancelled, UnassignedReason::Reinitialized, UnassignedReason::ReallocatedReplica,
                UnassignedReason::PrimaryFailed, UnassignedReason::ForcedEmptyPrimary, UnassignedReason::ManualAllocation,
                UnassignedReason::IndexClosed, UnassignedReason::NodeRestarting,
            ];
            // A new variant fails to compile here until listed above.
            let _sources_exhaustive = |r: &RecoverySource| match r {
                RecoverySource::EmptyStore | RecoverySource::ExistingStore { .. } | RecoverySource::Peer | RecoverySource::Snapshot { .. } | RecoverySource::LocalShards => (),
            };
            let _states_exhaustive = |s: ShardState| match s {
                ShardState::Unassigned | ShardState::Initializing | ShardState::Started | ShardState::Relocating => (),
            };
            let _exhaustive = |r: UnassignedReason| match r {
                UnassignedReason::IndexCreated | UnassignedReason::ClusterRecovered | UnassignedReason::IndexReopened
                | UnassignedReason::DanglingIndexImported | UnassignedReason::NewIndexRestored | UnassignedReason::ExistingIndexRestored
                | UnassignedReason::ReplicaAdded | UnassignedReason::AllocationFailed | UnassignedReason::NodeLeft
                | UnassignedReason::RerouteCancelled | UnassignedReason::Reinitialized | UnassignedReason::ReallocatedReplica
                | UnassignedReason::PrimaryFailed | UnassignedReason::ForcedEmptyPrimary | UnassignedReason::ManualAllocation
                | UnassignedReason::IndexClosed | UnassignedReason::NodeRestarting => (),
            };
            let states = [ShardState::Unassigned, ShardState::Initializing, ShardState::Started, ShardState::Relocating];
            let copy = |n: usize| ShardRouting {
                index: "i".into(), shard: 0, primary: n == 0, state: states[n % states.len()], node_id: Some("a".into()), relocating_node_id: Some("n".into()),
                allocation_id: Some(AllocationId { id: format!("p{version}-{n}"), relocation_id: Some("r".into()) }),
                recovery_source: Some(sources[n % sources.len()].clone()),
                unassigned_info: Some(UnassignedInfo { reason: reasons[n % reasons.len()], at_millis: 1, failed_allocations: 2, delayed: true,
                    details: Some("d".into()), last_allocation_status: "no_attempt".into(), last_node: Some("a".into()) }),
                expected_shard_size_in_bytes: Some(9),
            };
            s.routing_table.version = version;
            s.routing_table.indices.insert("i".into(), IndexRoutingTable { index_uuid: "iu".into(), shards: vec![IndexShardRoutingTable { shard: 0, copies: (0..reasons.len()).map(copy).collect() }] });
            // A global block besides the default one.
            s.blocks.global.insert(ClusterBlock { id: 7 + version as u32, description: "fixture".into(), retryable: true, disable_state_persistence: false, levels: vec!["write".into(), "metadata_write".into()] });
            s.routing_table.indices.insert(format!("only-{version}"), IndexRoutingTable { index_uuid: "ou".into(), shards: vec![] });
            s.blocks.indices.insert(format!("only-{version}"), [ClusterBlock::state_not_recovered()].into_iter().collect());
            s
        }
        let (base, next) = (rich(3), rich(4));
        let full = encode_v2(&EnvelopeV2 { from: "a".into(), body: Body::Message(Message::PublishRequest { request_id: 1, state: Box::new(next.clone()) }) }).unwrap();
        let diff = encode_v2(&EnvelopeV2 { from: "a".into(), body: Body::PublishDiff { request_id: 2, diff: Box::new(ClusterStateDiff::between(&base, &next)) } }).unwrap();
        // FNV-1a over both envelopes.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in full.iter().chain(diff.iter()) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(WIRE_FORMAT, 1);
        eprintln!("publication schema hash: {h:#018x}");
        assert_ne!(PUBLICATION_SCHEMA_PIN, 0, "PUBLICATION_SCHEMA_PIN is unset: set it to {h:#018x}");
        assert_eq!(h, PUBLICATION_SCHEMA_PIN, "the binary publication schema changed (pin {h:#018x}): bump WIRE_FORMAT and update PUBLICATION_SCHEMA_PIN");
        // Round trips, and a byte appended (a newer sender's field) is refused.
        assert!(decode_v2(&full).is_ok() && decode_v2(&diff).is_ok());
        let mut longer = full.to_vec();
        longer.push(0);
        assert_eq!(decode_v2(&longer).err().map(|e| e.error_type.clone()), Some(UNSUPPORTED_FORMAT.to_string()));
    }

    /// See [`the_publication_schema_is_pinned_to_the_wire_format`].
    const PUBLICATION_SCHEMA_PIN: u64 = 0xcec1_1899_3f14_88b3;

    #[test]
    fn diffs_go_only_to_nodes_that_accepted_the_previous_publication() {
        let mut t = PublishTracker::default();
        let (s3, s4) = (published(3), published(4));
        t.sending(&s3, 1, "a:1");
        t.sending(&s3, 2, "b:1");
        assert!(t.diff_for("a:1").is_none(), "no previous publication: full state");
        t.accepted("a:1", 3);
        t.accepted("b:1", 2); // a stale answer does not count
        t.sending(&s4, 3, "a:1");
        t.sending(&s4, 4, "b:1");
        t.sending(&s4, 5, "joiner:1");
        let d = t.diff_for("a:1").expect("a accepted version 3");
        assert_eq!(d.base_version, 3);
        assert!(t.diff_for("b:1").is_none(), "b did not accept the base");
        assert!(t.diff_for("joiner:1").is_none(), "a joining node gets the full state");
        assert_eq!(t.requests.get(&4).map(String::as_str), Some("b:1"), "full state can be re-sent per request");
    }

    struct Slow {
        release: parking_lot::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
        applied: AtomicBool,
    }
    impl Applier for Slow {
        fn apply(&self, _state: Arc<ClusterState>) {
            if let Some(rx) = self.release.lock().take() {
                let _ = rx.recv_timeout(Duration::from_secs(10));
            }
            self.applied.store(true, Ordering::SeqCst);
        }
    }

    /// A blocked application runs on its own thread: the caller (the
    /// actor) is never blocked, and completion is reported back in order.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_application_runs_off_the_actor() {
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let slow = Arc::new(Slow { release: parking_lot::Mutex::new(Some(release_rx)), applied: AtomicBool::new(false) });
        let (apply_tx, apply_rx) = std::sync::mpsc::channel();
        let (tx, mut rx) = mpsc::unbounded_channel();
        spawn_applier(slow.clone(), apply_rx, tx);
        let started = Instant::now();
        apply_tx.send((1, Arc::new(ClusterState::empty("c")))).unwrap();
        apply_tx.send((2, Arc::new(ClusterState::empty("c")))).unwrap();
        assert!(started.elapsed() < Duration::from_millis(100), "queuing never waits for application");
        assert!(tokio::time::timeout(Duration::from_millis(200), rx.recv()).await.is_err(), "still applying");
        release_tx.send(()).unwrap();
        let mut last = 0;
        while last < 2 {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                Ok(Some(Cmd::Applied { seq, .. })) => { assert!(seq > last); last = seq; }
                _ => panic!("applier did not report"),
            }
        }
        assert!(slow.applied.load(Ordering::SeqCst));
    }
    #[test]
    fn activated_lifecycle_refuses_unrecognized_join_and_election_before_coordination() {
        let current = crate::internal_capabilities::tests::lifecycle_state();
        // The pre-advertisement baseline (master) already preserves lifecycle
        // state: rolling upgrades with ILM indices must keep admitting it.
        let baseline = peer("baseline");
        let baseline_join = Message::JoinRequest { request_id: 8, cluster_name: "c".into(), source: baseline.clone(),
            min_term: 9, join: None, cluster_uuid: String::new(), cluster_uuid_committed: false };
        assert!(validate_capability_message(&current, &baseline_join).is_ok());
        assert!(validate_capability_message(&current, &Message::PreVoteRequest { request_id: 8, source: baseline, current_term: 10 }).is_ok());
        // An advertising build under an unrecognized schema fails closed.
        let mut source = peer("old");
        crate::internal_capabilities::advertise(&mut source.attributes);
        source.attributes.insert("rustic.internal.capabilities_schema".into(), "2".into());
        let join = Message::JoinRequest { request_id: 9, cluster_name: "c".into(), source: source.clone(),
            min_term: 9, join: None, cluster_uuid: String::new(), cluster_uuid_committed: false };
        let reason = validate_capability_message(&current, &join).unwrap_err();
        assert!(matches!(capability_failure_response(&join, reason), Some(Message::JoinResponse { request_id: 9, result: Err(_) })));
        let vote = Message::PreVoteRequest { request_id: 10, source: source.clone(), current_term: 10 };
        assert!(validate_capability_message(&current, &vote).is_err());
        let start = Message::StartJoin(elasticsearch_cluster::coordination::StartJoin { source_node: source, term: 11 });
        let reason = validate_capability_message(&current, &start).unwrap_err();
        assert!(capability_failure_response(&start, reason).is_none());
        let empty = ClusterState::empty("c");
        assert!(validate_capability_message(&empty, &start).is_ok(), "no lifecycle activation means no capability restriction");
        let mut upgraded = peer("new");
        crate::internal_capabilities::advertise(&mut upgraded.attributes);
        let start = Message::StartJoin(elasticsearch_cluster::coordination::StartJoin { source_node: upgraded, term: 11 });
        assert!(validate_capability_message(&current, &start).is_ok());
    }

    #[test]
    fn pending_pressure_relocation_refuses_an_old_join_before_the_leader_publishes() {
        let pending = crate::internal_capabilities::tests::pressure_state(false);
        let old = peer("old");
        let join = Message::JoinRequest { request_id: 7, cluster_name: "c".into(), source: old.clone(),
            min_term: 9, join: None, cluster_uuid: String::new(), cluster_uuid_committed: false };
        let reason = validate_capability_message(&pending, &join).unwrap_err();
        assert!(reason.contains("capacity-pressure"), "{reason}");
        assert!(matches!(capability_failure_response(&join, reason), Some(Message::JoinResponse { request_id: 7, result: Err(_) })));
        assert!(validate_capability_message(&pending, &Message::PreVoteRequest { request_id: 8, source: old.clone(), current_term: 10 }).is_err());
        // The relocated copy started: the latch is gone and the join proceeds.
        let started = crate::internal_capabilities::tests::pressure_state(true);
        assert!(validate_capability_message(&started, &join).is_ok());
    }

    /// After a full upgrade raised the segment format floor, an old build
    /// (no advertisement: a v11 reader) is refused at join, pre-vote and
    /// start-join, with the downgrade named; before, it is admitted.
    #[test]
    fn segment_format_floor_refuses_an_old_join_and_election() {
        let mut current = ClusterState::empty("c");
        current.metadata.cluster_uuid = "cu".into();
        let mut member = peer("new");
        crate::internal_capabilities::advertise(&mut member.attributes);
        current.nodes.insert(member.id.clone(), member.clone());
        let old = peer("old");
        let join = Message::JoinRequest { request_id: 7, cluster_name: "c".into(), source: old.clone(),
            min_term: 9, join: None, cluster_uuid: String::new(), cluster_uuid_committed: false };
        assert!(validate_capability_message(&current, &join).is_ok(), "mixed clusters admit old builds");
        let floor = elasticsearch_cluster::compatibility::negotiated_segment_format(&current, elasticsearch_cluster::compatibility::advertised_read_max);
        assert_eq!(floor, elasticsearch_engine::rustic_core::codec::VERSION);
        assert!(elasticsearch_cluster::compatibility::raise_segment_format(&mut current, floor));
        let reason = validate_capability_message(&current, &join).unwrap_err();
        assert!(reason.contains("downgrading") && reason.contains(&format!("v{floor}")), "{reason}");
        assert!(matches!(capability_failure_response(&join, reason), Some(Message::JoinResponse { request_id: 7, result: Err(_) })));
        assert!(validate_capability_message(&current, &Message::PreVoteRequest { request_id: 8, source: old.clone(), current_term: 10 }).is_err());
        let start = Message::StartJoin(elasticsearch_cluster::coordination::StartJoin { source_node: old, term: 11 });
        assert!(validate_capability_message(&current, &start).is_err());
        let rejoin = Message::JoinRequest { request_id: 9, cluster_name: "c".into(), source: member,
            min_term: 9, join: None, cluster_uuid: String::new(), cluster_uuid_committed: false };
        assert!(validate_capability_message(&current, &rejoin).is_ok(), "a current build still joins");
    }

}
