//! The safety core of cluster coordination: a port of ES's
//! `CoordinationState`, which implements the TLA+ specification
//! `ZenWithTerms`.
//!
//! The state consists of a persisted part — `current_term` and
//! `last_accepted_state` — and a volatile part (votes collected in the
//! current term, the last version this node published as leader). Every
//! method either performs a transition allowed by the model or returns
//! [`CoordinationStateRejected`] without changing anything.
//!
//! Persisted fields are never written behind the caller's back: every
//! mutation sets a dirty flag that the [`Coordinator`](super::Coordinator)
//! turns into an `Output::Persist` *before* it emits any output that
//! depends on the mutation (see the module docs for the contract).
//!
//! The safety argument, informally:
//! * A node votes (`handle_start_join`) at most once per term, because the
//!   term strictly increases and is persisted before the vote is sent.
//! * An election needs a quorum of votes in both the last committed and the
//!   last accepted configuration, and a vote is only counted if the voter's
//!   last accepted state is not fresher than the candidate's, so the winner
//!   holds every state that might have been committed (any committed state
//!   was accepted by a quorum which intersects the election quorum).
//! * A state is only committed once a quorum (again in both the committed
//!   and the published configuration) accepted it in the current term, and
//!   a node only accepts publications in its current term.
//! * The leader may only change the voting configuration when the previous
//!   change has been committed and its votes already form a quorum in the
//!   new configuration, so consecutive configurations' quorums intersect.

use crate::state::{ClusterState, DiscoveryNode, VotingConfiguration};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// What a node must keep on disk across restarts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedState {
    pub current_term: u64,
    pub last_accepted_state: ClusterState,
}

impl PersistedState {
    /// A brand-new node: term 0, empty state, no voting configuration.
    pub fn empty(cluster_name: &str) -> PersistedState {
        let mut s = ClusterState::empty(cluster_name);
        // Deterministic placeholder: the real uuid comes with the first
        // published state.
        s.state_uuid = crate::state::UNKNOWN_UUID.to_string();
        PersistedState { current_term: 0, last_accepted_state: s }
    }
}

/// A vote: `source_node` votes for `target_node` to become master in `term`,
/// declaring its own last accepted (term, version).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Join {
    pub source_node: String,
    pub target_node: String,
    pub term: u64,
    pub last_accepted_term: u64,
    pub last_accepted_version: u64,
}

/// Request to vote for `source_node` in `term` (sent by a candidate, or by
/// an abdicating master on behalf of its successor).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StartJoin {
    pub source_node: DiscoveryNode,
    pub term: u64,
}

/// Acknowledgement that a published state was accepted (persisted).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishResponse {
    pub term: u64,
    pub version: u64,
}

/// Instruction to mark the accepted state (term, version) as committed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyCommit {
    pub source_node: String,
    pub term: u64,
    pub version: u64,
}

/// ES's `CoordinationStateRejectedException`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinationStateRejected(pub String);

impl std::fmt::Display for CoordinationStateRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CoordinationStateRejected {}

fn reject<T>(msg: String) -> Result<T, CoordinationStateRejected> {
    Err(CoordinationStateRejected(msg))
}

/// Which persisted fields changed since the last `take_persist`.
#[derive(Clone, Copy, Debug, Default)]
struct Dirty {
    term: bool,
    state: bool,
}

#[derive(Debug)]
pub struct CoordinationState {
    local_node_id: String,
    persisted: PersistedState,
    // Volatile state (reset on restart).
    election_won: bool,
    join_votes: BTreeMap<String, Join>,
    started_join_since_last_reboot: bool,
    last_published_version: u64,
    last_published_config: VotingConfiguration,
    publish_votes: BTreeSet<String>,
    dirty: Dirty,
}

impl CoordinationState {
    pub fn new(local_node_id: &str, persisted: PersistedState) -> CoordinationState {
        let last_published_config = persisted.last_accepted_state.coordination.last_accepted_config.clone();
        CoordinationState {
            local_node_id: local_node_id.to_string(),
            persisted,
            election_won: false,
            join_votes: BTreeMap::new(),
            started_join_since_last_reboot: false,
            last_published_version: 0,
            last_published_config,
            publish_votes: BTreeSet::new(),
            dirty: Dirty::default(),
        }
    }

    // ------------------------------------------------------- accessors --

    pub fn current_term(&self) -> u64 {
        self.persisted.current_term
    }
    pub fn last_accepted_state(&self) -> &ClusterState {
        &self.persisted.last_accepted_state
    }
    pub fn persisted(&self) -> &PersistedState {
        &self.persisted
    }
    /// Term of the last accepted state (its coordination metadata term).
    pub fn last_accepted_term(&self) -> u64 {
        self.persisted.last_accepted_state.coordination.term
    }
    pub fn last_accepted_version(&self) -> u64 {
        self.persisted.last_accepted_state.version
    }
    pub fn last_committed_config(&self) -> &VotingConfiguration {
        &self.persisted.last_accepted_state.coordination.last_committed_config
    }
    pub fn last_accepted_config(&self) -> &VotingConfiguration {
        &self.persisted.last_accepted_state.coordination.last_accepted_config
    }
    pub fn last_published_version(&self) -> u64 {
        self.last_published_version
    }
    pub fn election_won(&self) -> bool {
        self.election_won
    }
    pub fn join_votes(&self) -> &BTreeMap<String, Join> {
        &self.join_votes
    }
    pub fn contains_join_vote_for(&self, node_id: &str) -> bool {
        self.join_votes.contains_key(node_id)
    }

    /// Quorum in both the last committed and last accepted configuration.
    pub fn is_election_quorum(&self, votes: &BTreeSet<String>) -> bool {
        self.last_committed_config().has_quorum(votes.iter()) && self.last_accepted_config().has_quorum(votes.iter())
    }

    /// Quorum in both the last committed and the last published
    /// configuration.
    pub fn is_publish_quorum(&self, votes: &BTreeSet<String>) -> bool {
        self.last_committed_config().has_quorum(votes.iter()) && self.last_published_config.has_quorum(votes.iter())
    }

    pub fn join_votes_have_quorum_for(&self, config: &VotingConfiguration) -> bool {
        config.has_quorum(self.join_votes.keys())
    }

    /// Takes the pending persistence work: `(current_term, Some(state) if
    /// the accepted state changed)`. `None` when nothing changed.
    pub fn take_persist(&mut self) -> Option<(u64, Option<ClusterState>)> {
        let d = std::mem::take(&mut self.dirty);
        if !d.term && !d.state {
            return None;
        }
        Some((self.persisted.current_term, d.state.then(|| self.persisted.last_accepted_state.clone())))
    }

    // ----------------------------------------------------- transitions --

    /// Bootstrap: install the initial voting configuration. Only allowed on
    /// a node that has never had one.
    pub fn set_initial_state(&mut self, initial: ClusterState) -> Result<(), CoordinationStateRejected> {
        if !self.last_accepted_config().is_empty() {
            return reject(format!("initial state already set: last-accepted configuration now {:?}", self.last_accepted_config().0));
        }
        if initial.coordination.last_accepted_config.is_empty() || initial.coordination.last_committed_config.is_empty() {
            return reject("initial state must have a voting configuration".into());
        }
        if initial.coordination.term != 0 {
            return reject(format!("initial state should have term 0, has {}", initial.coordination.term));
        }
        debug_assert!(!self.election_won && self.join_votes.is_empty() && self.publish_votes.is_empty());
        self.persisted.last_accepted_state = initial;
        self.last_published_config = self.last_accepted_config().clone();
        self.dirty.state = true;
        Ok(())
    }

    /// Vote for `start_join.source_node` in `start_join.term`, which must be
    /// newer than the current term. The returned `Join` may only be sent
    /// after the new term has been persisted.
    pub fn handle_start_join(&mut self, start_join: &StartJoin) -> Result<Join, CoordinationStateRejected> {
        if start_join.term <= self.current_term() {
            return reject(format!("incoming term {} not greater than current term {}", start_join.term, self.current_term()));
        }
        self.persisted.current_term = start_join.term;
        self.dirty.term = true;
        self.last_published_version = 0;
        self.last_published_config = self.last_accepted_config().clone();
        self.started_join_since_last_reboot = true;
        self.election_won = false;
        self.join_votes.clear();
        self.publish_votes.clear();
        Ok(Join {
            source_node: self.local_node_id.clone(),
            target_node: start_join.source_node.id.clone(),
            term: self.current_term(),
            last_accepted_term: self.last_accepted_term(),
            last_accepted_version: self.last_accepted_version(),
        })
    }

    /// Count a vote for this node. Returns whether the vote was new.
    pub fn handle_join(&mut self, join: &Join) -> Result<bool, CoordinationStateRejected> {
        if join.target_node != self.local_node_id {
            return reject(format!("join {join:?} is not for this node [{}]", self.local_node_id));
        }
        if join.term != self.current_term() {
            return reject(format!("incoming term {} does not match current term {}", join.term, self.current_term()));
        }
        if !self.started_join_since_last_reboot {
            return reject("ignored join as term has not been incremented yet after reboot".into());
        }
        let last_accepted_term = self.last_accepted_term();
        if join.last_accepted_term > last_accepted_term {
            return reject(format!("incoming last accepted term {} of join higher than current last accepted term {}", join.last_accepted_term, last_accepted_term));
        }
        if join.last_accepted_term == last_accepted_term && join.last_accepted_version > self.last_accepted_version() {
            return reject(format!(
                "incoming last accepted version {} of join higher than current last accepted version {} in term {}",
                join.last_accepted_version,
                self.last_accepted_version(),
                last_accepted_term
            ));
        }
        if self.last_accepted_config().is_empty() {
            // Joins received before bootstrapping could make the election
            // "won" the moment the configuration is set; ES rejects them.
            return reject("rejecting join since this node has not received its initial configuration yet".into());
        }
        let added = self.join_votes.insert(join.source_node.clone(), join.clone()).is_none();
        let prev_won = self.election_won;
        let votes: BTreeSet<String> = self.join_votes.keys().cloned().collect();
        self.election_won = self.is_election_quorum(&votes);
        debug_assert!(!prev_won || self.election_won, "cannot go from won to not won");
        if self.election_won && !prev_won {
            self.last_published_version = self.last_accepted_version();
        }
        Ok(added)
    }

    /// The leader wants to publish `state`. Checks it may; afterwards the
    /// caller sends `PublishRequest(state)` to every node (itself included).
    pub fn handle_client_value(&mut self, state: &ClusterState) -> Result<(), CoordinationStateRejected> {
        if !self.election_won {
            return reject("election not won".into());
        }
        if self.last_published_version != self.last_accepted_version() {
            return reject("cannot start publishing next value before accepting previous one".into());
        }
        if state.coordination.term != self.current_term() {
            return reject(format!("incoming term {} does not match current term {}", state.coordination.term, self.current_term()));
        }
        if state.version <= self.last_published_version {
            return reject(format!("incoming cluster state version {} lower or equal to last published version {}", state.version, self.last_published_version));
        }
        if state.coordination.last_accepted_config != *self.last_accepted_config() && self.last_committed_config() != self.last_accepted_config() {
            return reject("only allow reconfiguration while not already reconfiguring".into());
        }
        if !self.join_votes_have_quorum_for(&state.coordination.last_accepted_config) {
            return reject("only allow reconfiguration if joinVotes have quorum for new config".into());
        }
        if state.coordination.last_committed_config != *self.last_committed_config() {
            return reject("last committed configuration should not change".into());
        }
        self.last_published_version = state.version;
        self.last_published_config = state.coordination.last_accepted_config.clone();
        self.publish_votes.clear();
        Ok(())
    }

    /// Accept a published state (persisting it). The response may only be
    /// sent once the state is durable.
    pub fn handle_publish_request(&mut self, state: ClusterState) -> Result<PublishResponse, CoordinationStateRejected> {
        if state.coordination.term != self.current_term() {
            return reject(format!("incoming term {} does not match current term {}", state.coordination.term, self.current_term()));
        }
        if state.coordination.term == self.last_accepted_term() && state.version <= self.last_accepted_version() {
            return reject(format!("incoming version {} lower or equal to current version {}", state.version, self.last_accepted_version()));
        }
        let resp = PublishResponse { term: state.coordination.term, version: state.version };
        self.persisted.last_accepted_state = state;
        self.dirty.state = true;
        Ok(resp)
    }

    /// Count an acceptance; returns the commit to broadcast once the
    /// accepting nodes form a publish quorum (every time, the caller
    /// de-duplicates).
    pub fn handle_publish_response(&mut self, source_node_id: &str, resp: PublishResponse) -> Result<Option<ApplyCommit>, CoordinationStateRejected> {
        if !self.election_won {
            return reject("election not won".into());
        }
        if resp.term != self.current_term() {
            return reject(format!("incoming term {} does not match current term {}", resp.term, self.current_term()));
        }
        if resp.version != self.last_published_version {
            return reject(format!("incoming version {} does not match current version {}", resp.version, self.last_published_version));
        }
        self.publish_votes.insert(source_node_id.to_string());
        if self.is_publish_quorum(&self.publish_votes.clone()) {
            return Ok(Some(ApplyCommit { source_node: self.local_node_id.clone(), term: resp.term, version: resp.version }));
        }
        Ok(None)
    }

    /// Mark the accepted state committed (persisting the committed
    /// configuration and `cluster_uuid_committed`).
    pub fn handle_commit(&mut self, commit: &ApplyCommit) -> Result<(), CoordinationStateRejected> {
        if commit.term != self.current_term() {
            return reject(format!("incoming term {} does not match current term {}", commit.term, self.current_term()));
        }
        if commit.term != self.last_accepted_term() {
            return reject(format!("incoming term {} does not match last accepted term {}", commit.term, self.last_accepted_term()));
        }
        if commit.version != self.last_accepted_version() {
            return reject(format!("incoming version {} does not match current version {}", commit.version, self.last_accepted_version()));
        }
        let s = &mut self.persisted.last_accepted_state;
        if s.coordination.last_committed_config != s.coordination.last_accepted_config || !s.metadata.cluster_uuid_committed {
            s.coordination.last_committed_config = s.coordination.last_accepted_config.clone();
            s.metadata.cluster_uuid_committed = true;
            self.dirty.state = true;
        }
        Ok(())
    }
}
