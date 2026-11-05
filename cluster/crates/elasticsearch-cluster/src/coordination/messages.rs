//! Coordination wire protocol. Every message is addressed by transport
//! address; the driver owns connections. Requests carry a `request_id`
//! chosen by the sender and responses echo it. Failures travel as
//! `Err(reason)` inside responses; transport-level failures (could not
//! connect, connection closed) are reported to the sender with
//! [`Coordinator::handle_send_failure`](super::Coordinator::handle_send_failure).

use super::coordination_state::{ApplyCommit, Join, StartJoin};
use crate::state::{ClusterState, DiscoveryNode};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeersResponse {
    /// The responding node (its identity, learnt as part of the exchange).
    pub source: DiscoveryNode,
    /// `Some(responder)` if the responder is the elected master, or the
    /// master it follows.
    pub master_node: Option<DiscoveryNode>,
    /// Master-eligible peers the responder knows (only while it is itself
    /// looking for a master).
    pub known_peers: Vec<DiscoveryNode>,
    pub term: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreVoteResponse {
    pub current_term: u64,
    pub last_accepted_term: u64,
    pub last_accepted_version: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublishWithJoinResponse {
    pub term: u64,
    pub version: u64,
    /// The vote the follower gave this leader in this term, if any (it may
    /// only just have bumped its term to accept the publication).
    pub join: Option<Join>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyCommitAck {
    pub node_id: String,
    pub term: u64,
    pub version: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Message {
    /// `internal:discovery/request_peers`.
    PeersRequest { request_id: u64, cluster_name: String, source: DiscoveryNode, known_peers: Vec<DiscoveryNode> },
    PeersResponse { request_id: u64, result: Result<PeersResponse, String> },
    /// `internal:cluster/request_pre_vote`.
    PreVoteRequest { request_id: u64, source: DiscoveryNode, current_term: u64 },
    PreVoteResponse { request_id: u64, result: Result<PreVoteResponse, String> },
    /// `internal:cluster/coordination/start_join` (one-way: the answer is a
    /// `JoinRequest` carrying the vote).
    StartJoin(StartJoin),
    /// `internal:cluster/coordination/join`. `cluster_uuid*` let the master
    /// refuse a node that belongs to another cluster.
    JoinRequest {
        request_id: u64,
        cluster_name: String,
        source: DiscoveryNode,
        min_term: u64,
        join: Option<Join>,
        cluster_uuid: String,
        cluster_uuid_committed: bool,
    },
    JoinResponse { request_id: u64, result: Result<(), String> },
    /// `internal:cluster/coordination/publish_state` (full state; the
    /// master is `state.master_node_id`).
    PublishRequest { request_id: u64, state: Box<ClusterState> },
    PublishResponse { request_id: u64, result: Result<PublishWithJoinResponse, String> },
    /// `internal:cluster/coordination/commit_state`.
    ApplyCommit { request_id: u64, commit: ApplyCommit },
    ApplyCommitResponse { request_id: u64, result: Result<ApplyCommitAck, String> },
    /// `internal:coordination/fault_detection/leader_check`.
    LeaderCheck { request_id: u64, source: DiscoveryNode, target_ephemeral_id: String },
    LeaderCheckResponse { request_id: u64, result: Result<(), String> },
    /// `internal:coordination/fault_detection/follower_check`.
    FollowerCheck { request_id: u64, source: DiscoveryNode, term: u64, target_ephemeral_id: String },
    FollowerCheckResponse { request_id: u64, result: Result<(), String> },
}

impl Message {
    /// The id of a request (`None` for responses and one-way messages).
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Message::PeersRequest { request_id, .. }
            | Message::PreVoteRequest { request_id, .. }
            | Message::JoinRequest { request_id, .. }
            | Message::PublishRequest { request_id, .. }
            | Message::ApplyCommit { request_id, .. }
            | Message::LeaderCheck { request_id, .. }
            | Message::FollowerCheck { request_id, .. } => Some(*request_id),
            _ => None,
        }
    }

    /// The id a response answers (`None` for requests).
    pub fn response_to(&self) -> Option<u64> {
        match self {
            Message::PeersResponse { request_id, .. }
            | Message::PreVoteResponse { request_id, .. }
            | Message::JoinResponse { request_id, .. }
            | Message::PublishResponse { request_id, .. }
            | Message::ApplyCommitResponse { request_id, .. }
            | Message::LeaderCheckResponse { request_id, .. }
            | Message::FollowerCheckResponse { request_id, .. } => Some(*request_id),
            _ => None,
        }
    }

    /// ES transport action name.
    pub fn action(&self) -> &'static str {
        match self {
            Message::PeersRequest { .. } | Message::PeersResponse { .. } => "internal:discovery/request_peers",
            Message::PreVoteRequest { .. } | Message::PreVoteResponse { .. } => "internal:cluster/request_pre_vote",
            Message::StartJoin(_) => "internal:cluster/coordination/start_join",
            Message::JoinRequest { .. } | Message::JoinResponse { .. } => "internal:cluster/coordination/join",
            Message::PublishRequest { .. } | Message::PublishResponse { .. } => "internal:cluster/coordination/publish_state",
            Message::ApplyCommit { .. } | Message::ApplyCommitResponse { .. } => "internal:cluster/coordination/commit_state",
            Message::LeaderCheck { .. } | Message::LeaderCheckResponse { .. } => "internal:coordination/fault_detection/leader_check",
            Message::FollowerCheck { .. } | Message::FollowerCheckResponse { .. } => "internal:coordination/fault_detection/follower_check",
        }
    }
}

#[cfg(test)]
mod schema_field_guards {
    use super::Message;

    /// No `..` patterns: adding even a serde-default/skipped field must fail
    /// compilation here and receive a deliberate compatibility review. Byte
    /// ordering/types are additionally covered by the node publication pin.
    #[test]
    fn coordination_schema_fields_require_explicit_review() {
        let _discoverynode = |value: &crate::state::DiscoveryNode| {
            let crate::state::DiscoveryNode { id: _, name: _, ephemeral_id: _, transport_address: _, http_address: _, host: _, ip: _, roles: _, attributes: _, version: _ } = value;
        };
        let _votingconfigexclusion = |value: &crate::state::VotingConfigExclusion| {
            let crate::state::VotingConfigExclusion { node_id: _, node_name: _ } = value;
        };
        let _coordinationmetadata = |value: &crate::state::CoordinationMetadata| {
            let crate::state::CoordinationMetadata { term: _, last_committed_config: _, last_accepted_config: _, voting_config_exclusions: _ } = value;
        };
        let _indexmetadata = |value: &crate::state::IndexMetadata| {
            let crate::state::IndexMetadata { name: _, uuid: _, version: _, mapping_version: _, settings_version: _, aliases_version: _, state: _, settings: _, mappings: _, aliases: _, number_of_shards: _, number_of_replicas: _, routing_num_shards: _, primary_terms: _, in_sync_allocation_ids: _, creation_date: _, lifecycle: _ } = value;
        };
        let _metadata = |value: &crate::state::Metadata| {
            let crate::state::Metadata { cluster_uuid: _, cluster_uuid_committed: _, version: _, persistent_settings: _, transient_settings: _, indices: _, index_templates: _, component_templates: _, legacy_templates: _, customs: _ } = value;
        };
        let _allocationid = |value: &crate::state::AllocationId| {
            let crate::state::AllocationId { id: _, relocation_id: _ } = value;
        };
        let _unassignedinfo = |value: &crate::state::UnassignedInfo| {
            let crate::state::UnassignedInfo { reason: _, at_millis: _, failed_allocations: _, delayed: _, details: _, last_allocation_status: _, last_node: _ } = value;
        };
        let _shardrouting = |value: &crate::state::ShardRouting| {
            let crate::state::ShardRouting { index: _, shard: _, primary: _, state: _, node_id: _, relocating_node_id: _, allocation_id: _, recovery_source: _, unassigned_info: _, expected_shard_size_in_bytes: _ } = value;
        };
        let _indexshardroutingtable = |value: &crate::state::IndexShardRoutingTable| {
            let crate::state::IndexShardRoutingTable { shard: _, copies: _ } = value;
        };
        let _indexroutingtable = |value: &crate::state::IndexRoutingTable| {
            let crate::state::IndexRoutingTable { index_uuid: _, shards: _ } = value;
        };
        let _routingtable = |value: &crate::state::RoutingTable| {
            let crate::state::RoutingTable { version: _, indices: _ } = value;
        };
        let _clusterblock = |value: &crate::state::ClusterBlock| {
            let crate::state::ClusterBlock { id: _, description: _, retryable: _, disable_state_persistence: _, levels: _ } = value;
        };
        let _clusterblocks = |value: &crate::state::ClusterBlocks| {
            let crate::state::ClusterBlocks { global: _, indices: _ } = value;
        };
        let _clusterstate = |value: &crate::state::ClusterState| {
            let crate::state::ClusterState { cluster_name: _, version: _, state_uuid: _, master_node_id: _, nodes: _, coordination: _, metadata: _, routing_table: _, blocks: _ } = value;
        };
        let _persistedstate = |value: &super::super::coordination_state::PersistedState| {
            let super::super::coordination_state::PersistedState { current_term: _, last_accepted_state: _ } = value;
        };
        let _join = |value: &super::super::coordination_state::Join| {
            let super::super::coordination_state::Join { source_node: _, target_node: _, term: _, last_accepted_term: _, last_accepted_version: _ } = value;
        };
        let _startjoin = |value: &super::super::coordination_state::StartJoin| {
            let super::super::coordination_state::StartJoin { source_node: _, term: _ } = value;
        };
        let _publishresponse = |value: &super::super::coordination_state::PublishResponse| {
            let super::super::coordination_state::PublishResponse { term: _, version: _ } = value;
        };
        let _applycommit = |value: &super::super::coordination_state::ApplyCommit| {
            let super::super::coordination_state::ApplyCommit { source_node: _, term: _, version: _ } = value;
        };
        let _peersresponse = |value: &super::PeersResponse| {
            let super::PeersResponse { source: _, master_node: _, known_peers: _, term: _ } = value;
        };
        let _prevoteresponse = |value: &super::PreVoteResponse| {
            let super::PreVoteResponse { current_term: _, last_accepted_term: _, last_accepted_version: _ } = value;
        };
        let _publishwithjoinresponse = |value: &super::PublishWithJoinResponse| {
            let super::PublishWithJoinResponse { term: _, version: _, join: _ } = value;
        };
        let _applycommitack = |value: &super::ApplyCommitAck| {
            let super::ApplyCommitAck { node_id: _, term: _, version: _ } = value;
        };
        let _message = |value: &Message| match value {
            Message::PeersRequest { request_id: _, cluster_name: _, source: _, known_peers: _ } => (),
            Message::PeersResponse { request_id: _, result: _ } => (),
            Message::PreVoteRequest { request_id: _, source: _, current_term: _ } => (),
            Message::PreVoteResponse { request_id: _, result: _ } => (),
            Message::JoinRequest { request_id: _, cluster_name: _, source: _, min_term: _, join: _, cluster_uuid: _, cluster_uuid_committed: _ } => (),
            Message::JoinResponse { request_id: _, result: _ } => (),
            Message::PublishRequest { request_id: _, state: _ } => (),
            Message::PublishResponse { request_id: _, result: _ } => (),
            Message::ApplyCommit { request_id: _, commit: _ } => (),
            Message::ApplyCommitResponse { request_id: _, result: _ } => (),
            Message::LeaderCheck { request_id: _, source: _, target_ephemeral_id: _ } => (),
            Message::LeaderCheckResponse { request_id: _, result: _ } => (),
            Message::FollowerCheck { request_id: _, source: _, term: _, target_ephemeral_id: _ } => (),
            Message::FollowerCheckResponse { request_id: _, result: _ } => (),
            Message::StartJoin(_) => (),
        };
    }
}
