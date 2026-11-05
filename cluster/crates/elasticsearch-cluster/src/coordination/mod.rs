//! Cluster coordination: master election and cluster-state publication.
//!
//! This is a sans-IO, deterministic re-implementation of Elasticsearch's
//! "Zen2" subsystem (`org.elasticsearch.cluster.coordination`). It owns no
//! sockets, threads or clocks: the driver (the `elasticsearch-node` crate,
//! or the simulation in `simulation.rs`) feeds it messages, transport
//! failures and the current time, and executes the [`Output`]s it returns.
//!
//! # Layers
//!
//! * [`CoordinationState`] (`coordination_state.rs`) is the *safety core*, a
//!   line-by-line port of ES's `CoordinationState` (itself the
//!   implementation of the TLA+ model `ZenWithTerms`). It decides whether
//!   to grant a vote, whether an election is won, whether a publication may
//!   start, whether to accept a published state and when a state is
//!   committed. Nothing outside it may change the persisted term or
//!   last-accepted state, so every safety property rests on this file.
//! * [`Coordinator`] (`coordinator.rs`) provides *liveness*: modes
//!   (CANDIDATE / LEADER / FOLLOWER), peer discovery, pre-voting, the
//!   randomised election scheduler, bootstrapping, join handling, the
//!   two-phase publication pipeline, leader/follower fault detection, the
//!   lag detector, voting-config reconfiguration and the master-service
//!   task queue.
//! * [`reconfigurator`] computes the "best" voting configuration (ES's
//!   `Reconfigurator`).
//! * [`messages`] is the wire protocol (serde).
//!
//! # Protocol summary
//!
//! A *term* is an election epoch. Every node persists `current_term` and
//! the `last_accepted_state` (whose coordination metadata carries the term
//! it was published in, the last *committed* voting configuration and the
//! last *accepted* voting configuration).
//!
//! 1. **Discovery.** A CANDIDATE probes `discovery.seed_hosts` plus the
//!    master-eligible nodes of its last accepted state every
//!    `discovery.find_peers_interval` with `PeersRequest`s, learning
//!    `DiscoveryNode` identities and further peer addresses. If a peer says
//!    *it* is the elected master, the candidate joins it.
//! 2. **Bootstrap.** A brand-new cluster has an empty voting configuration
//!    and nobody can win an election. A master-eligible node with
//!    `cluster.initial_master_nodes` waits until it has discovered a
//!    majority of the listed nodes, then writes the initial configuration
//!    (unfound entries become placeholder ids) and a fresh cluster UUID —
//!    exactly once, and never if it already has a configuration.
//! 3. **Pre-vote.** Once the discovered peers could form an election
//!    quorum, the election scheduler fires after a random delay
//!    (`cluster.election.*`) and runs a pre-voting round: no term is
//!    bumped unless a quorum of nodes (with no leader of their own and no
//!    fresher state) would vote for us.
//! 4. **Election.** The candidate sends `StartJoin(term+1)`; every
//!    receiver bumps its term (persisted!) and replies with a `Join` vote.
//!    The election is won when the votes form a quorum in *both* the last
//!    committed and last accepted configurations.
//! 5. **Publication.** The leader publishes each new state in two phases:
//!    `PublishRequest` → nodes persist it as "accepted" and respond; once a
//!    quorum (again in both configurations) has accepted, `ApplyCommit` is
//!    sent; nodes then mark it committed (persisted) and apply it. Only one
//!    publication is in flight; queued work is batched into the next one.
//!    A publication that cannot commit within `cluster.publish.timeout`
//!    makes the leader step down.
//! 6. **Fault detection.** Followers check the leader
//!    (`cluster.fault_detection.leader_check.*`) and become candidates when
//!    it fails; the leader checks every follower
//!    (`cluster.fault_detection.follower_check.*`) and removes failed ones,
//!    plus any node that has not applied a committed state within
//!    `cluster.follower_lag.timeout`. A leader that cannot commit (lost its
//!    quorum) steps down. Candidates apply their last state with the
//!    `cluster.no_master_block`.
//! 7. **Reconfiguration.** With `cluster.auto_shrink_voting_configuration`
//!    the leader keeps an odd number of live master-eligible voters,
//!    honouring voting-config exclusions; configuration changes only take
//!    effect through committed publications.
//!
//! # Driver contract
//!
//! * Inputs: [`Coordinator::start`] once, then [`Coordinator::handle_message`]
//!   for every inbound message (with the sender's transport address),
//!   [`Coordinator::handle_send_failure`] when the transport could not
//!   deliver a request we sent (connection refused / disconnected),
//!   [`Coordinator::handle_disconnect`] when an established connection to
//!   a peer closed, and
//!   [`Coordinator::tick`] whenever `now >= next_deadline()` (calling it more
//!   often, e.g. every 50ms, is fine: timer precision is the tick period).
//!   Time is a monotonic `now_millis: u64` supplied by the driver.
//! * Leader-only master-service inputs: [`Coordinator::submit_task`],
//!   [`Coordinator::publish`], [`Coordinator::add_voting_config_exclusions`],
//!   [`Coordinator::clear_voting_config_exclusions`].
//! * Outputs are returned as an ordered `Vec<Output>` and **must be executed
//!   in order**.
//!
//! ## Persistence-ordering contract
//!
//! [`Output::Persist`] carries a new `current_term` and/or
//! `last_accepted_state`. The driver must make it **durable (fsync)
//! before acting on any later output** of the same batch or of any later
//! call — in particular before sending any later message. Votes (`Join`),
//! publish acknowledgements and commit acknowledgements are only safe to
//! send once the term / accepted state they vouch for is on disk; the
//! coordinator guarantees that every such `Send` is preceded by the
//! `Persist` it depends on. If the process crashes after a `Persist` was
//! written but before later outputs ran, simply restart from the persisted
//! state: un-sent messages are equivalent to dropped messages.
//! `ApplyState` outputs should likewise be applied before acting on later
//! outputs (a follower's apply-ack follows its `ApplyState`).
//!
//! On restart, rebuild with [`Coordinator::new`] from the last
//! [`PersistedState`] and a `DiscoveryNode` with the same `id` and a fresh
//! `ephemeral_id`.

//! # Deviations from ES
//!
//! Safety rules are ES's exactly. Liveness and plumbing differ where a
//! sans-IO design needs it or where simulation found ES's behaviour weak:
//!
//! * **Explicit timeouts.** ES relies on TCP disconnects; here every
//!   request without its own protocol timeout has one
//!   (`cluster.join.timeout` 10s for joins, `discovery.request_peers_timeout`
//!   for peers and pre-votes). Publish/commit requests are bounded by the
//!   publication (`cluster.publish.timeout`).
//! * **Join validation** runs on the master using the joiner's
//!   `cluster_uuid`/`cluster_uuid_committed` carried in the `JoinRequest`
//!   (ES sends a `ValidateJoinRequest` back to the joiner); a follower also
//!   refuses publications from a different committed cluster uuid.
//! * **Every join triggers a publication**, even from a node that is
//!   already a member, so a member that dropped to CANDIDATE becomes a
//!   follower at once instead of waiting to be removed by follower checks.
//! * **Restarts:** a join from a known node id with a new ephemeral id
//!   replaces the old incarnation in one publication; a new node id at a
//!   known transport address replaces the stale node.
//! * **Vote collection:** a master-eligible member that never voted for the
//!   current master (it voted for a losing candidate in that term, or
//!   restarted mid-term) cannot enter the voting configuration until the
//!   next election. ES just waits, which can leave e.g. a 3-node cluster
//!   with configuration `{a, b, placeholder}` that tolerates no failure.
//!   The master here re-runs its own election (rate-limited) when missing
//!   votes are all that blocks a better configuration.
//! * **Excluded-node fallback:** a node excluded from voting may stand for
//!   election after `excluded_node_election_fallback_ms` (default 30s,
//!   configurable) as a candidate, so a cluster whose only up-to-date node
//!   is excluded can still elect a master. The election still needs a
//!   quorum, and the node abdicates once another eligible node can lead.
//! * The coordinator publishes full states; the driver sends a node that
//!   accepted the previous publication a diff instead and rebuilds the
//!   full state on the receiving side (`wire::ClusterStateDiff`), so the
//!   safety core only ever sees full states. `state_not_recovered` is cleared
//!   by the first publication of a master;
//!   voting-only nodes never stand for election but have no special
//!   vote-counting rule; the master handles its own publish/commit inline
//!   rather than over the transport.
//!
//! # Simulation
//!
//! `simulation.rs` + `sim_scenarios.rs` run seeded, deterministic clusters
//! under message loss, duplication, reordering, delay, symmetric,
//! asymmetric and bridged partitions (black-hole or fail-fast), crashes
//! that lose un-fsynced state (including mid-batch), restarts with new
//! ephemeral ids, slow disks and unsynchronised clocks, checking the safety
//! invariants listed there after every output. `COORD_SEEDS` sets the
//! number of seeds per family (default 1000); the `#[ignore]`d `soak`
//! test runs every family with `COORD_SEEDS` (default 100000) seeds from
//! `COORD_FIRST_SEED`. A failing seed is replayed with tracing and printed;
//! reproduce with `COORD_FIRST=<seed> COORD_SEEDS=1`.

pub mod coordination_state;
mod coordinator;
pub mod messages;
pub mod reconfigurator;

#[cfg(test)]
mod sim_scenarios;
#[cfg(test)]
mod simulation;
#[cfg(test)]
mod tests;

pub use coordination_state::{ApplyCommit, CoordinationState, CoordinationStateRejected, Join, PersistedState, PublishResponse, StartJoin};
pub use coordinator::{CoordinationError, Coordinator, Mode, NoopStateUpdater, Output, PersistUpdate, StateTask, StateUpdater, Submitted};
pub use messages::Message;

use crate::state::ClusterBlock;
use std::collections::BTreeMap;

/// Prefix ES uses for voting-config entries standing in for
/// `cluster.initial_master_nodes` entries not yet discovered at bootstrap.
pub const BOOTSTRAP_PLACEHOLDER_PREFIX: &str = "{bootstrap-placeholder}-";

/// `cluster.max_voting_config_exclusions` default.
pub const MAX_VOTING_CONFIG_EXCLUSIONS: usize = 10;

/// `cluster.no_master_block`: what a node without an elected master blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoMasterBlock {
    /// `write` (default): reads still work, writes and metadata writes fail.
    Write,
    /// `metadata_write`: only metadata writes fail.
    MetadataWrite,
    /// `all`: reads and writes fail.
    All,
}

impl NoMasterBlock {
    pub fn parse(s: &str) -> Option<NoMasterBlock> {
        match s {
            "write" => Some(NoMasterBlock::Write),
            "metadata_write" => Some(NoMasterBlock::MetadataWrite),
            "all" => Some(NoMasterBlock::All),
            _ => None,
        }
    }

    /// The cluster block (id 2) a node adds to its applied state.
    pub fn block(self) -> ClusterBlock {
        match self {
            NoMasterBlock::Write => ClusterBlock::no_master_writes(),
            NoMasterBlock::All => ClusterBlock::no_master_all(),
            NoMasterBlock::MetadataWrite => {
                let mut b = ClusterBlock::no_master_writes();
                b.levels = vec!["metadata_write".to_string()];
                b
            }
        }
    }
}

/// All coordination settings, with ES's defaults. Durations in millis.
#[derive(Clone, Debug)]
pub struct CoordinationSettings {
    /// `cluster.name`.
    pub cluster_name: String,
    /// `discovery.seed_hosts` (transport addresses).
    pub seed_hosts: Vec<String>,
    /// `cluster.initial_master_nodes` (node names or transport addresses /
    /// IPs).
    pub initial_master_nodes: Vec<String>,
    /// `discovery.find_peers_interval` (1s).
    pub find_peers_interval_ms: u64,
    /// `discovery.request_peers_timeout` (3s).
    pub request_peers_timeout_ms: u64,
    /// `cluster.election.initial_timeout` (100ms).
    pub election_initial_timeout_ms: u64,
    /// `cluster.election.back_off_time` (100ms).
    pub election_back_off_time_ms: u64,
    /// `cluster.election.max_timeout` (10s).
    pub election_max_timeout_ms: u64,
    /// `cluster.election.duration` (500ms).
    pub election_duration_ms: u64,
    /// `cluster.publish.timeout` (30s).
    pub publish_timeout_ms: u64,
    /// `cluster.follower_lag.timeout` (90s).
    pub follower_lag_timeout_ms: u64,
    /// `cluster.join.timeout`: how long an outgoing join request stays
    /// pending before it may be re-sent. ES relies on transport-level
    /// disconnect detection instead; a sans-IO state machine needs an
    /// explicit bound (10s).
    pub join_timeout_ms: u64,
    /// `cluster.fault_detection.leader_check.interval` (1s).
    pub leader_check_interval_ms: u64,
    /// `cluster.fault_detection.leader_check.timeout` (10s).
    pub leader_check_timeout_ms: u64,
    /// `cluster.fault_detection.leader_check.retry_count` (3).
    pub leader_check_retry_count: u32,
    /// `cluster.fault_detection.follower_check.interval` (1s).
    pub follower_check_interval_ms: u64,
    /// `cluster.fault_detection.follower_check.timeout` (10s).
    pub follower_check_timeout_ms: u64,
    /// `cluster.fault_detection.follower_check.retry_count` (3).
    pub follower_check_retry_count: u32,
    /// `cluster.auto_shrink_voting_configuration` (true).
    pub auto_shrink_voting_configuration: bool,
    /// `cluster.no_master_block` (write). A value of the same setting in
    /// the cluster state's persistent/transient settings overrides it.
    pub no_master_block: NoMasterBlock,
    /// `cluster.max_voting_config_exclusions` (10).
    pub max_voting_config_exclusions: usize,
    /// Not in ES: how long a master-eligible node that is excluded from
    /// voting (and so normally never stands for election) must have been
    /// a candidate before it stands anyway (`cluster.election.excluded_node_fallback`,
    /// default 30s). It lets a cluster whose only up-to-date node is excluded
    /// still elect a master (see `Coordinator`).
    pub excluded_node_election_fallback_ms: u64,
}

impl CoordinationSettings {
    pub fn new(cluster_name: &str) -> CoordinationSettings {
        CoordinationSettings {
            cluster_name: cluster_name.to_string(),
            seed_hosts: Vec::new(),
            initial_master_nodes: Vec::new(),
            find_peers_interval_ms: 1_000,
            request_peers_timeout_ms: 3_000,
            election_initial_timeout_ms: 100,
            election_back_off_time_ms: 100,
            election_max_timeout_ms: 10_000,
            election_duration_ms: 500,
            publish_timeout_ms: 30_000,
            follower_lag_timeout_ms: 90_000,
            join_timeout_ms: 10_000,
            leader_check_interval_ms: 1_000,
            leader_check_timeout_ms: 10_000,
            leader_check_retry_count: 3,
            follower_check_interval_ms: 1_000,
            follower_check_timeout_ms: 10_000,
            follower_check_retry_count: 3,
            auto_shrink_voting_configuration: true,
            no_master_block: NoMasterBlock::Write,
            max_voting_config_exclusions: MAX_VOTING_CONFIG_EXCLUSIONS,
            excluded_node_election_fallback_ms: 30_000,
        }
    }

    /// Build from flat node settings (`elasticsearch.yml` keys, string
    /// values; lists comma-separated). Unknown keys are ignored.
    pub fn from_flat(settings: &BTreeMap<String, String>) -> Result<CoordinationSettings, String> {
        let mut s = CoordinationSettings::new(settings.get("cluster.name").map(String::as_str).unwrap_or("elasticsearch"));
        let list = |v: &str| v.split(',').map(|x| x.trim().trim_matches(|c| c == '[' || c == ']' || c == '"').to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>();
        for (k, v) in settings {
            match k.as_str() {
                "discovery.seed_hosts" => s.seed_hosts = list(v),
                "cluster.initial_master_nodes" => s.initial_master_nodes = list(v),
                "discovery.find_peers_interval" => s.find_peers_interval_ms = parse_time_millis(v)?,
                "discovery.request_peers_timeout" => s.request_peers_timeout_ms = parse_time_millis(v)?,
                "cluster.election.initial_timeout" => s.election_initial_timeout_ms = parse_time_millis(v)?,
                "cluster.election.back_off_time" => s.election_back_off_time_ms = parse_time_millis(v)?,
                "cluster.election.max_timeout" => s.election_max_timeout_ms = parse_time_millis(v)?,
                "cluster.election.duration" => s.election_duration_ms = parse_time_millis(v)?,
                "cluster.publish.timeout" => s.publish_timeout_ms = parse_time_millis(v)?,
                "cluster.follower_lag.timeout" => s.follower_lag_timeout_ms = parse_time_millis(v)?,
                "cluster.join.timeout" => s.join_timeout_ms = parse_time_millis(v)?,
                "cluster.fault_detection.leader_check.interval" => s.leader_check_interval_ms = parse_time_millis(v)?,
                "cluster.fault_detection.leader_check.timeout" => s.leader_check_timeout_ms = parse_time_millis(v)?,
                "cluster.fault_detection.leader_check.retry_count" => s.leader_check_retry_count = v.parse().map_err(|_| format!("invalid [{k}]: [{v}]"))?,
                "cluster.fault_detection.follower_check.interval" => s.follower_check_interval_ms = parse_time_millis(v)?,
                "cluster.fault_detection.follower_check.timeout" => s.follower_check_timeout_ms = parse_time_millis(v)?,
                "cluster.fault_detection.follower_check.retry_count" => s.follower_check_retry_count = v.parse().map_err(|_| format!("invalid [{k}]: [{v}]"))?,
                "cluster.auto_shrink_voting_configuration" => s.auto_shrink_voting_configuration = v.parse().map_err(|_| format!("invalid [{k}]: [{v}]"))?,
                "cluster.no_master_block" => s.no_master_block = NoMasterBlock::parse(v).ok_or_else(|| format!("invalid [{k}]: [{v}]"))?,
                "cluster.election.excluded_node_fallback" => s.excluded_node_election_fallback_ms = parse_time_millis(v)?,
                "cluster.max_voting_config_exclusions" => s.max_voting_config_exclusions = v.parse().map_err(|_| format!("invalid [{k}]: [{v}]"))?,
                _ => {}
            }
        }
        // ES's setting bounds: a zero interval or retry count would spin
        // the checkers or fail a node on its first missed check.
        let bounds: [(&str, u64, u64, u64); 13] = [
            ("discovery.find_peers_interval", s.find_peers_interval_ms, 1, u64::MAX),
            ("discovery.request_peers_timeout", s.request_peers_timeout_ms, 1, u64::MAX),
            ("cluster.election.initial_timeout", s.election_initial_timeout_ms, 1, 10_000),
            ("cluster.election.back_off_time", s.election_back_off_time_ms, 1, 60_000),
            ("cluster.election.max_timeout", s.election_max_timeout_ms, 200, 300_000),
            ("cluster.election.duration", s.election_duration_ms, 1, 300_000),
            ("cluster.publish.timeout", s.publish_timeout_ms, 1, u64::MAX),
            ("cluster.follower_lag.timeout", s.follower_lag_timeout_ms, 1, u64::MAX),
            ("cluster.join.timeout", s.join_timeout_ms, 1, u64::MAX),
            ("cluster.fault_detection.leader_check.interval", s.leader_check_interval_ms, 100, u64::MAX),
            ("cluster.fault_detection.leader_check.timeout", s.leader_check_timeout_ms, 1, 60_000),
            ("cluster.fault_detection.follower_check.interval", s.follower_check_interval_ms, 100, u64::MAX),
            ("cluster.fault_detection.follower_check.timeout", s.follower_check_timeout_ms, 1, 60_000),
        ];
        let human = |ms: u64| if ms % 1000 == 0 { format!("{}s", ms / 1000) } else { format!("{ms}ms") };
        for (key, value, min, max) in bounds {
            if value < min {
                return Err(format!("failed to parse value [{}] for setting [{key}], must be >= [{}]", human(value), human(min)));
            }
            if value > max {
                return Err(format!("failed to parse value [{}] for setting [{key}], must be <= [{}]", human(value), human(max)));
            }
        }
        for (key, value) in [
            ("cluster.fault_detection.leader_check.retry_count", s.leader_check_retry_count as u64),
            ("cluster.fault_detection.follower_check.retry_count", s.follower_check_retry_count as u64),
            ("cluster.max_voting_config_exclusions", s.max_voting_config_exclusions as u64),
        ] {
            if value < 1 {
                // ES `Setting.parseInt`'s message.
                return Err(format!("Failed to parse value [{value}] for setting [{key}] must be >= 1"));
            }
        }
        Ok(s)
    }
}

/// Parse an ES time value (`500ms`, `1s`, `2m`, `1h`, `1d`, bare millis).
pub fn parse_time_millis(v: &str) -> Result<u64, String> {
    let v = v.trim();
    let (num, mult) = if let Some(n) = v.strip_suffix("ms") {
        (n, 1)
    } else if let Some(n) = v.strip_suffix('s') {
        (n, 1_000)
    } else if let Some(n) = v.strip_suffix('m') {
        (n, 60_000)
    } else if let Some(n) = v.strip_suffix('h') {
        (n, 3_600_000)
    } else if let Some(n) = v.strip_suffix('d') {
        (n, 86_400_000)
    } else {
        (v, 1)
    };
    num.trim().parse::<u64>().map(|n| n * mult).map_err(|_| format!("failed to parse time value [{v}]"))
}
