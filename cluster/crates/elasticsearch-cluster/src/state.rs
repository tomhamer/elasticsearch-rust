//! The cluster state: the single, versioned, master-published document that
//! every node applies. Mirrors ES's `ClusterState` (nodes, metadata,
//! routing table, blocks, coordination metadata) closely enough that
//! `GET /_cluster/state` can be rendered in ES's exact shape.
//!
//! Invariants (enforced by the master, checked by `validate`):
//! - `version` strictly increases with every published state; `term` never
//!   decreases.
//! - every `ShardRouting` with a `node_id` names a node in `nodes`.
//! - at most one primary per shard; primary and replica of the same shard
//!   never share a node.
//! - `metadata.indices[i].in_sync_allocation_ids[s]` is the set of copies
//!   allowed to become primary for shard `s`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

pub use crate::wire::{json_bytes, Shared};

/// ES's placeholder for "no cluster uuid yet".
pub const UNKNOWN_UUID: &str = "_na_";

/// A 22-char URL-safe base64 id, the shape ES uses for node ids, index
/// uuids, allocation ids and the cluster uuid.
pub fn random_id() -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&uuid::Uuid::new_v4().as_bytes()[..16])
}

// ---------------------------------------------------------------- nodes ----

/// `node.roles` values, in ES's canonical order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Data,
    DataCold,
    DataContent,
    DataFrozen,
    DataHot,
    DataWarm,
    /// Stateless `index` node: holds the indexing (primary) shard copies.
    Index,
    Ingest,
    Master,
    Ml,
    RemoteClusterClient,
    /// Stateless `search` node: holds the searchable (replica) copies.
    Search,
    Transform,
    VotingOnly,
}

impl Role {
    pub const ALL: [Role; 14] = [
        Role::Data,
        Role::DataCold,
        Role::DataContent,
        Role::DataFrozen,
        Role::DataHot,
        Role::DataWarm,
        Role::Index,
        Role::Ingest,
        Role::Master,
        Role::Ml,
        Role::RemoteClusterClient,
        Role::Search,
        Role::Transform,
        Role::VotingOnly,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Role::Data => "data",
            Role::DataCold => "data_cold",
            Role::DataContent => "data_content",
            Role::DataFrozen => "data_frozen",
            Role::DataHot => "data_hot",
            Role::DataWarm => "data_warm",
            Role::Index => "index",
            Role::Ingest => "ingest",
            Role::Master => "master",
            Role::Ml => "ml",
            Role::RemoteClusterClient => "remote_cluster_client",
            Role::Search => "search",
            Role::Transform => "transform",
            Role::VotingOnly => "voting_only",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        Role::ALL.iter().copied().find(|r| r.name() == s)
    }

    /// The one-letter abbreviation `_cat/nodes` prints in `node.role`.
    pub fn abbreviation(self) -> char {
        match self {
            Role::Data => 'd',
            Role::DataCold => 'c',
            Role::DataContent => 's',
            Role::DataFrozen => 'f',
            Role::DataHot => 'h',
            Role::DataWarm => 'w',
            Role::Index => 'I',
            Role::Ingest => 'i',
            Role::Master => 'm',
            Role::Ml => 'l',
            Role::RemoteClusterClient => 'r',
            Role::Search => 'S',
            Role::Transform => 't',
            Role::VotingOnly => 'v',
        }
    }

    /// Whether the role makes a node hold shard data
    /// (`DiscoveryNodeRole.canContainData`: the data roles, and the
    /// stateless `index` and `search` roles).
    pub fn is_data(self) -> bool {
        matches!(self, Role::Data | Role::DataCold | Role::DataContent | Role::DataFrozen | Role::DataHot | Role::DataWarm | Role::Index | Role::Search)
    }

    /// The stateless roles (`StatelessPlugin.STATELESS_ROLES`).
    pub fn is_stateless(self) -> bool {
        matches!(self, Role::Index | Role::Search)
    }

    /// ES's default role set: every role except `voting_only` and the
    /// stateless `index` / `search` roles.
    pub fn defaults() -> BTreeSet<Role> {
        Role::ALL.iter().copied().filter(|r| *r != Role::VotingOnly && !r.is_stateless()).collect()
    }

    /// The default role set of a stateless node (`index` replaces the
    /// data roles: `INDEX_ROLE.isEnabledByDefault` on stateless nodes, the
    /// data roles are not).
    pub fn stateless_defaults() -> BTreeSet<Role> {
        Role::ALL.iter().copied().filter(|r| *r != Role::VotingOnly && *r != Role::Search && !(r.is_data() && *r != Role::Index)).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryNode {
    /// Persistent node id (22 chars), stable across restarts.
    pub id: String,
    pub name: String,
    /// Changes on every process start; lets peers tell a restart apart.
    pub ephemeral_id: String,
    /// `host:port` of the transport listener.
    pub transport_address: String,
    /// `host:port` of the HTTP listener (`_nodes/http`, client sniffing).
    pub http_address: String,
    pub host: String,
    pub ip: String,
    pub roles: BTreeSet<Role>,
    /// `node.attr.*` (zone, rack, `xpack.installed`, ...).
    pub attributes: BTreeMap<String, String>,
    pub version: String,
}

impl DiscoveryNode {
    pub fn is_master_eligible(&self) -> bool {
        self.roles.contains(&Role::Master)
    }
    pub fn is_data(&self) -> bool {
        self.roles.iter().any(|r| r.is_data())
    }
    /// A stateless node (`index` or `search` role).
    pub fn is_stateless(&self) -> bool {
        self.roles.iter().any(|r| r.is_stateless())
    }
    pub fn is_voting_only(&self) -> bool {
        self.roles.contains(&Role::VotingOnly)
    }
    /// `_cat/nodes` `node.role` column, e.g. `cdfhilmrstw`.
    pub fn role_abbreviations(&self) -> String {
        let mut s: Vec<char> = self.roles.iter().map(|r| r.abbreviation()).collect();
        s.sort_unstable();
        if s.is_empty() { "-".into() } else { s.into_iter().collect() }
    }
}

// ---------------------------------------------------------- coordination ---

/// A set of master-eligible node ids; a quorum is a strict majority.
/// The two special ids ES uses are not needed: bootstrapping is explicit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VotingConfiguration(pub BTreeSet<String>);

impl VotingConfiguration {
    pub fn new<I: IntoIterator<Item = String>>(ids: I) -> Self {
        VotingConfiguration(ids.into_iter().collect())
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn has_quorum<'a, I: IntoIterator<Item = &'a String>>(&self, votes: I) -> bool {
        let n = votes.into_iter().filter(|v| self.0.contains(*v)).collect::<BTreeSet<_>>().len();
        n * 2 > self.0.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VotingConfigExclusion {
    pub node_id: String,
    pub node_name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinationMetadata {
    pub term: u64,
    pub last_committed_config: VotingConfiguration,
    pub last_accepted_config: VotingConfiguration,
    pub voting_config_exclusions: Vec<VotingConfigExclusion>,
}

// -------------------------------------------------------------- metadata ---

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexState {
    Open,
    Close,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub name: String,
    pub uuid: String,
    pub version: u64,
    pub mapping_version: u64,
    pub settings_version: u64,
    pub aliases_version: u64,
    pub state: IndexState,
    /// Flat settings (`index.number_of_shards: "1"`), ES's stored form.
    /// Shared between successive states like `mappings`.
    pub settings: Shared<Map<String, Value>>,
    /// `{"properties": ...}` as `GET _mapping` returns it (without the
    /// index-name wrapper). Shared between successive states (cloning a
    /// state does not copy mappings) and serialized once for the wire.
    pub mappings: Shared<Value>,
    #[serde(with = "json_bytes")]
    pub aliases: BTreeMap<String, Value>,
    pub number_of_shards: u32,
    pub number_of_replicas: u32,
    /// `index.number_of_routing_shards` (routing hash space).
    pub routing_num_shards: u32,
    /// One entry per shard; bumped whenever a new primary is chosen.
    pub primary_terms: Vec<u64>,
    /// Shard number → allocation ids of copies that are in sync.
    pub in_sync_allocation_ids: BTreeMap<u32, BTreeSet<String>>,
    pub creation_date: i64,
    /// The index's lifecycle execution state and rollover info (ES keeps
    /// `LifecycleExecutionState` and `rollover_info` here) as the master's
    /// lifecycle left it: published, and reverted, with the step's other
    /// effects; a new master continues from it. `None`: no lifecycle state.
    #[serde(default)]
    pub lifecycle: Option<Shared<Value>>,
}

impl IndexMetadata {
    pub fn setting(&self, key: &str) -> Option<&str> {
        self.settings.get(key).and_then(Value::as_str)
    }
    pub fn primary_term(&self, shard: u32) -> u64 {
        self.primary_terms.get(shard as usize).copied().unwrap_or(1)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub cluster_uuid: String,
    pub cluster_uuid_committed: bool,
    pub version: u64,
    #[serde(with = "json_bytes")]
    pub persistent_settings: Map<String, Value>,
    #[serde(with = "json_bytes")]
    pub transient_settings: Map<String, Value>,
    pub indices: BTreeMap<String, IndexMetadata>,
    /// Composable index templates, component templates, legacy templates.
    #[serde(with = "json_bytes")]
    pub index_templates: BTreeMap<String, Value>,
    #[serde(with = "json_bytes")]
    pub component_templates: BTreeMap<String, Value>,
    #[serde(with = "json_bytes")]
    pub legacy_templates: BTreeMap<String, Value>,
    /// Everything else ES keeps as metadata customs: `repositories`,
    /// `ingest`, `index_lifecycle`, `snapshot_lifecycle`, `data_stream`,
    /// `persistent_tasks`, `index-graveyard`, ...
    #[serde(with = "json_bytes")]
    pub customs: BTreeMap<String, Value>,
}

impl Metadata {
    /// A cluster setting, transient first (ES precedence), then persistent.
    pub fn setting(&self, key: &str) -> Option<&Value> {
        self.transient_settings.get(key).or_else(|| self.persistent_settings.get(key))
    }
}

// --------------------------------------------------------------- routing ---

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ShardState {
    Unassigned,
    Initializing,
    Started,
    Relocating,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationId {
    pub id: String,
    /// Set on both halves of a relocation.
    pub relocation_id: Option<String>,
}

/// Where an initializing copy gets its data (`recovery_source.type`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoverySource {
    /// New index: start empty.
    EmptyStore,
    /// Primary re-opening its own data (restart, stale primary).
    ExistingStore { bootstrap_new_history_uuid: bool },
    /// Replica / relocation target copying from the primary.
    Peer,
    Snapshot { repository: String, snapshot: String, index: String },
    /// Shrink / split / clone.
    LocalShards,
}

/// `unassigned_info.reason` values ES uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UnassignedReason {
    IndexCreated,
    ClusterRecovered,
    IndexReopened,
    DanglingIndexImported,
    NewIndexRestored,
    ExistingIndexRestored,
    ReplicaAdded,
    AllocationFailed,
    NodeLeft,
    RerouteCancelled,
    Reinitialized,
    ReallocatedReplica,
    PrimaryFailed,
    ForcedEmptyPrimary,
    ManualAllocation,
    IndexClosed,
    NodeRestarting,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnassignedInfo {
    pub reason: UnassignedReason,
    pub at_millis: i64,
    pub failed_allocations: u32,
    pub delayed: bool,
    pub details: Option<String>,
    /// `allocation_status`: `no_attempt`, `deciders_no`, `deciders_throttled`, `no_valid_shard_copy`, `awaiting_info`.
    pub last_allocation_status: String,
    /// Node the copy was on when it became unassigned (delayed allocation).
    pub last_node: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardRouting {
    pub index: String,
    pub shard: u32,
    pub primary: bool,
    pub state: ShardState,
    pub node_id: Option<String>,
    pub relocating_node_id: Option<String>,
    pub allocation_id: Option<AllocationId>,
    /// Internally tagged (ES's JSON shape), which binary formats cannot
    /// decode: carried as JSON bytes there.
    #[serde(with = "json_bytes")]
    pub recovery_source: Option<RecoverySource>,
    pub unassigned_info: Option<UnassignedInfo>,
    pub expected_shard_size_in_bytes: Option<u64>,
}

impl ShardRouting {
    pub fn unassigned(index: &str, shard: u32, primary: bool, info: UnassignedInfo, source: Option<RecoverySource>) -> ShardRouting {
        ShardRouting {
            index: index.to_string(),
            shard,
            primary,
            state: ShardState::Unassigned,
            node_id: None,
            relocating_node_id: None,
            allocation_id: None,
            recovery_source: source,
            unassigned_info: Some(info),
            expected_shard_size_in_bytes: None,
        }
    }
    pub fn active(&self) -> bool {
        matches!(self.state, ShardState::Started | ShardState::Relocating)
    }
    pub fn assigned(&self) -> bool {
        self.node_id.is_some()
    }
    pub fn allocation_id(&self) -> Option<&str> {
        self.allocation_id.as_ref().map(|a| a.id.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexShardRoutingTable {
    pub shard: u32,
    /// Primary first, then replicas. A relocation is represented as the
    /// RELOCATING source copy plus an INITIALIZING target copy whose
    /// `relocating_node_id` points back at the source.
    pub copies: Vec<ShardRouting>,
}

impl IndexShardRoutingTable {
    pub fn primary(&self) -> Option<&ShardRouting> {
        self.copies.iter().find(|c| c.primary && c.relocating_node_id.is_none() || c.primary && c.state == ShardState::Relocating)
    }
    pub fn active_primary(&self) -> Option<&ShardRouting> {
        self.copies.iter().find(|c| c.primary && c.active())
    }
    pub fn replicas(&self) -> impl Iterator<Item = &ShardRouting> {
        self.copies.iter().filter(|c| !c.primary)
    }
    /// Copies that can serve reads (`STARTED` or `RELOCATING`).
    pub fn active_copies(&self) -> impl Iterator<Item = &ShardRouting> {
        self.copies.iter().filter(|c| c.active())
    }
    /// The active copies searches, gets and other reads go to (ES
    /// `ShardIterator.allSearchableShards` / `OperationRouting.searchShards`
    /// keep only `ShardRouting.isSearchable` copies): every active copy,
    /// except that on stateless nodes the indexing copy (the primary,
    /// `INDEX_ONLY`) serves no reads, only the search copies do.
    pub fn searchable_copies<'a>(&'a self, nodes: &'a BTreeMap<String, DiscoveryNode>) -> impl Iterator<Item = &'a ShardRouting> + 'a {
        self.active_copies().filter(move |c| copy_searchable(c, nodes))
    }
    /// ES `ActiveShardCount.ONE.enoughShardsActive` (the default
    /// `wait_for_active_shards`): a shard with search shards needs an active
    /// search shard, others an active copy. Without stateless nodes every
    /// copy is searchable, so this is "an active primary"; on a stateless
    /// cluster the search shards are the replicas (the primary is
    /// `INDEX_ONLY`), so a shard with replicas waits for one of them.
    pub fn one_shard_active(&self, nodes: &BTreeMap<String, DiscoveryNode>) -> bool {
        if self.is_stateless(nodes) && self.copies.iter().any(|c| !c.primary) {
            self.searchable_copies(nodes).next().is_some()
        } else {
            self.active_primary().is_some()
        }
    }
    /// A stateless shard (some copy is on a stateless node).
    pub fn is_stateless(&self, nodes: &BTreeMap<String, DiscoveryNode>) -> bool {
        self.copies.iter().any(|c| c.node_id.as_ref().and_then(|n| nodes.get(n)).is_some_and(|n| n.is_stateless()))
    }
    /// A stateless shard whose active copies include none that serves
    /// reads (its search copies are unassigned or still recovering): ES
    /// treats it like an unassigned shard for reads.
    pub fn stateless_without_search_copy(&self, nodes: &BTreeMap<String, DiscoveryNode>) -> bool {
        let stateless = self.copies.iter().any(|c| c.node_id.as_ref().and_then(|n| nodes.get(n)).is_some_and(|n| n.is_stateless()));
        stateless && self.searchable_copies(nodes).next().is_none()
    }
}

/// See [`IndexShardRoutingTable::searchable_copies`].
pub fn copy_searchable(c: &ShardRouting, nodes: &BTreeMap<String, DiscoveryNode>) -> bool {
    match c.node_id.as_ref().and_then(|n| nodes.get(n)) {
        Some(n) if n.is_stateless() => !c.primary,
        _ => true,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRoutingTable {
    pub index_uuid: String,
    pub shards: Vec<IndexShardRoutingTable>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub version: u64,
    pub indices: BTreeMap<String, IndexRoutingTable>,
}

impl RoutingTable {
    pub fn all_shards(&self) -> impl Iterator<Item = &ShardRouting> {
        self.indices.values().flat_map(|i| i.shards.iter()).flat_map(|s| s.copies.iter())
    }
    pub fn shards_on_node<'a>(&'a self, node_id: &'a str) -> impl Iterator<Item = &'a ShardRouting> + 'a {
        self.all_shards().filter(move |s| s.node_id.as_deref() == Some(node_id))
    }
    pub fn shard_table(&self, index: &str, shard: u32) -> Option<&IndexShardRoutingTable> {
        self.indices.get(index).and_then(|i| i.shards.get(shard as usize))
    }
}

// ---------------------------------------------------------------- blocks ---

/// One cluster block (`GET _cluster/state/blocks`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClusterBlock {
    pub id: u32,
    pub description: String,
    pub retryable: bool,
    pub disable_state_persistence: bool,
    /// `read`, `write`, `metadata_read`, `metadata_write`.
    pub levels: Vec<String>,
}

impl ClusterBlock {
    /// `STATE_NOT_RECOVERED_BLOCK` (id 1).
    pub fn state_not_recovered() -> ClusterBlock {
        ClusterBlock { id: 1, description: "state not recovered / initialized".into(), retryable: true, disable_state_persistence: true, levels: levels(&["read", "write", "metadata_read", "metadata_write"]) }
    }
    /// `NO_MASTER_BLOCK_WRITES` (id 2, default `cluster.no_master_block: write`).
    pub fn no_master_writes() -> ClusterBlock {
        ClusterBlock { id: 2, description: "no master".into(), retryable: true, disable_state_persistence: false, levels: levels(&["write", "metadata_write"]) }
    }
    /// `NO_MASTER_BLOCK_ALL` (id 2, `cluster.no_master_block: all`).
    pub fn no_master_all() -> ClusterBlock {
        ClusterBlock { id: 2, description: "no master".into(), retryable: true, disable_state_persistence: false, levels: levels(&["read", "write", "metadata_read", "metadata_write"]) }
    }
    /// `index.blocks.read_only_allow_delete` set by the flood-stage watermark (id 12).
    pub fn read_only_allow_delete() -> ClusterBlock {
        ClusterBlock { id: 12, description: "disk usage exceeded flood-stage watermark, index has read-only-allow-delete block".into(), retryable: false, disable_state_persistence: false, levels: levels(&["write", "metadata_write"]) }
    }
    pub fn index_closed() -> ClusterBlock {
        ClusterBlock { id: 4, description: "index closed".into(), retryable: false, disable_state_persistence: false, levels: levels(&["read", "write"]) }
    }
    pub fn blocks(&self, level: &str) -> bool {
        self.levels.iter().any(|l| l == level)
    }
}

fn levels(l: &[&str]) -> Vec<String> {
    l.iter().map(|s| s.to_string()).collect()
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterBlocks {
    pub global: BTreeSet<ClusterBlock>,
    pub indices: BTreeMap<String, BTreeSet<ClusterBlock>>,
}

impl ClusterBlocks {
    pub fn global_blocked(&self, level: &str) -> Option<&ClusterBlock> {
        self.global.iter().find(|b| b.blocks(level))
    }
    pub fn index_blocked(&self, index: &str, level: &str) -> Option<&ClusterBlock> {
        self.global_blocked(level).or_else(|| self.indices.get(index).and_then(|s| s.iter().find(|b| b.blocks(level))))
    }
}

// --------------------------------------------------------- cluster state ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterState {
    pub cluster_name: String,
    pub version: u64,
    /// Random per published state (`state_uuid`).
    pub state_uuid: String,
    pub master_node_id: Option<String>,
    pub nodes: BTreeMap<String, DiscoveryNode>,
    pub coordination: CoordinationMetadata,
    pub metadata: Metadata,
    pub routing_table: RoutingTable,
    pub blocks: ClusterBlocks,
}

impl ClusterState {
    /// The state a node starts with before it has joined a cluster.
    pub fn empty(cluster_name: &str) -> ClusterState {
        let mut blocks = ClusterBlocks::default();
        blocks.global.insert(ClusterBlock::state_not_recovered());
        ClusterState {
            cluster_name: cluster_name.to_string(),
            version: 0,
            state_uuid: random_id(),
            master_node_id: None,
            nodes: BTreeMap::new(),
            coordination: CoordinationMetadata::default(),
            metadata: Metadata { cluster_uuid: UNKNOWN_UUID.into(), ..Default::default() },
            routing_table: RoutingTable::default(),
            blocks,
        }
    }

    pub fn term(&self) -> u64 {
        self.coordination.term
    }

    pub fn master(&self) -> Option<&DiscoveryNode> {
        self.master_node_id.as_ref().and_then(|id| self.nodes.get(id))
    }

    pub fn data_nodes(&self) -> impl Iterator<Item = &DiscoveryNode> {
        self.nodes.values().filter(|n| n.is_data())
    }

    pub fn master_eligible_nodes(&self) -> impl Iterator<Item = &DiscoveryNode> {
        self.nodes.values().filter(|n| n.is_master_eligible())
    }

    /// Successor state: `version + 1` and a fresh `state_uuid`. The master
    /// builds every new state through this.
    pub fn next(&self) -> ClusterState {
        let mut s = self.clone();
        s.version += 1;
        s.state_uuid = random_id();
        s
    }

    /// Check the structural invariants listed in the module docs.
    pub fn validate(&self) -> Result<(), String> {
        for (name, irt) in &self.routing_table.indices {
            let meta = self.metadata.indices.get(name).ok_or_else(|| format!("routing table has unknown index [{name}]"))?;
            if irt.shards.len() != meta.number_of_shards as usize {
                return Err(format!("index [{name}] has {} shard tables, metadata says {}", irt.shards.len(), meta.number_of_shards));
            }
            for st in &irt.shards {
                let primaries = st.copies.iter().filter(|c| c.primary && c.relocating_node_id.is_none()).count();
                if primaries > 1 {
                    return Err(format!("[{name}][{}] has {primaries} primaries", st.shard));
                }
                let mut seen = BTreeSet::new();
                for c in &st.copies {
                    if let Some(n) = &c.node_id {
                        if !self.nodes.contains_key(n) {
                            return Err(format!("[{name}][{}] assigned to unknown node [{n}]", st.shard));
                        }
                        if !seen.insert(n.clone()) {
                            return Err(format!("[{name}][{}] has two copies on node [{n}]", st.shard));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quorum() {
        let c = VotingConfiguration::new(["a".to_string(), "b".into(), "c".into()]);
        assert!(!c.has_quorum([&"a".to_string()]));
        assert!(c.has_quorum([&"a".to_string(), &"c".to_string()]));
        assert!(!c.has_quorum([&"a".to_string(), &"x".to_string(), &"y".to_string()]));
    }

    #[test]
    fn ids_are_es_shaped() {
        let id = random_id();
        assert_eq!(id.len(), 22);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }

    #[test]
    fn role_abbreviations() {
        let mut roles = Role::defaults();
        let n = DiscoveryNode { id: "x".into(), name: "n".into(), ephemeral_id: "e".into(), transport_address: "127.0.0.1:9300".into(), http_address: "127.0.0.1:9200".into(), host: "h".into(), ip: "127.0.0.1".into(), roles: roles.clone(), attributes: BTreeMap::new(), version: "8.15.0".into() };
        assert_eq!(n.role_abbreviations(), "cdfhilmrstw");
        roles.clear();
        assert_eq!(DiscoveryNode { roles, ..n }.role_abbreviations(), "-");
    }
}
