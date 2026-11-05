//! The cluster node: wires config, engine, transport, coordination, the
//! cluster-state applier, shard roles and the master's background work.

#[path = "recovery_control.rs"]
mod recovery_control;
pub use recovery_control::{RecoveryStartReq,RecoveryReuseReq,RecoveryBatchReq,RecoveryFinalizeReq,BoundRecoveryReq};

use crate::config::NodeConfig;
use crate::coord::{self, Applier, CoordHandle};
use crate::gateway::{self, Gateway};
use crate::master::{self, MasterCtx, Updater};
use crate::mirror::{self, UuidMap};
use crate::replication::ops::{ReplicationRequest, ReplicationResponse, ResyncRequest};
use crate::replication::recovery::{RecoveryStartInfo, RecoveryState};
use crate::replication::replica::{ReplicaApplier, ReplicaPrefix};
use crate::shards::{self, CopyRecord, LocalShard, PrimaryCtx, RecoveryBatch, ReplicationNet, Role, ShardManager, Target};
use crate::stats::CpuSampler;
use elasticsearch_cluster::allocation::{FailedShard, StartedShard};
use elasticsearch_cluster::coordination::{CoordinationSettings, Coordinator, Mode, NoMasterBlock};
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode, RecoverySource, ShardRouting, ShardState};
use elasticsearch_engine::index::{IndexService, RecoverySource as EngineRecovery, ShardOpenSpec};
use elasticsearch_engine::node::Node as Engine;
use elasticsearch_transport::{RemoteError, TransportConfig, TransportService};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub const ES_VERSION: &str = elasticsearch_engine::node::ES_VERSION;

pub const ACTION_SHARD_STARTED: &str = "internal:cluster/shard/started";
pub const ACTION_SHARD_FAILED: &str = "internal:cluster/shard/failure";
pub const ACTION_MAPPING_UPDATE: &str = "internal:cluster/mapping/update";
pub const ACTION_MAPPING_UPDATE_V2: &str = "internal:cluster/mapping/update_uuid_v2";

/// The dynamic-mapping action `master` handles: the UUID-fenced one when it
/// advertises it, else the original one (which also carries and fences on
/// the index UUID).
pub(crate) fn mapping_update_action(master: &DiscoveryNode) -> &'static str {
    if crate::internal_capabilities::supports(master, crate::internal_capabilities::Capability::MappingUuidV2) { ACTION_MAPPING_UPDATE_V2 } else { ACTION_MAPPING_UPDATE }
}
pub const ACTION_PUBLISH_METADATA: &str = "internal:cluster/metadata/publish";
pub const ACTION_APPLIED_VERSION: &str = "internal:cluster/state/applied_version";
/// Does this node really run the primary of a shard right now? (Asked by a
/// recovery target before it discards an in-sync local copy.)
pub const ACTION_PRIMARY_PROBE: &str = "internal:index/shard/recovery/probe_primary";

#[path = "node_control.rs"]
mod control;
pub use control::{ShardStartedReq,ShardFailedReq,FollowerMetadataReq};
/// A CCR follower's metadata change, sent to the elected master.
pub const ACTION_FOLLOWER_METADATA: &str = "internal:ccr/follower_metadata";

/// On the master: a follower's metadata change as a dry run, published.
async fn commit_follower_change(node: &Arc<ClusterNode>, req: FollowerMetadataReq) -> Result<u64, RemoteError> {
    let parse=|text:&str|crate::owned_json::OwnedJson::parse(text).map_err(|e|RemoteError::new(e.kind,e.reason,e.status));
    let mapping=req.mapping_json.as_deref().map(parse).transpose()?;
    let settings=parse(&req.settings_json)?;
    let aliases=req.aliases_json.as_deref().map(parse).transpose()?;
    let reopen=req.reopen;
    let failure: Arc<Mutex<Option<elasticsearch_engine::error::EsError>>> = Arc::new(Mutex::new(None));
    let failed = failure.clone();
    let follower = req.follower;
    let run: Box<dyn FnOnce(&ClusterNode, crate::master_service::On) -> bool + Send> = Box::new(move |n: &ClusterNode, on: crate::master_service::On| {
        let _envelope=req.memory;
        // Tree owners stay in this queued job even if its submitter cancels.
        // BTreeMap nodes are admitted before transferring parsed object entries.
        fn object(value:serde_json::Value,memory:&mut elasticsearch_engine::memory::Reservation)->elasticsearch_engine::error::Result<BTreeMap<String,serde_json::Value>>{
            let serde_json::Value::Object(values)=value else{return Err(elasticsearch_engine::error::EsError::new(400,"illegal_argument_exception","follower metadata must be an object"));};
            // Conservative two internal B-tree nodes per inserted pair. Bound the
            // concrete key/value slots and edge/header words, not encoded bytes.
            let node_bytes=2*(11*std::mem::size_of::<(String,serde_json::Value)>()+20*std::mem::size_of::<usize>());
            let bytes=values.len().checked_mul(node_bytes).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","follower map capacity overflow"))?;
            memory.grow(bytes as u64,"follower metadata map nodes")?;
            Ok(values.into_iter().collect())
        }
        fn optional(value:Option<crate::owned_json::OwnedJson>,f:impl FnOnce(Option<serde_json::Value>,Option<&mut elasticsearch_engine::memory::Reservation>)->elasticsearch_engine::error::Result<()>)->elasticsearch_engine::error::Result<()>{
            match value{Some(value)=>value.consume(|value,memory|f(Some(value),Some(memory))),None=>f(None,None)}
        }
        let result=optional(mapping,|mapping,_|settings.consume(|settings,settings_memory|optional(aliases,|aliases,aliases_memory|{
            let settings=object(settings,settings_memory)?;
            let aliases=match (aliases,aliases_memory){(Some(value),Some(memory))=>Some(object(value,memory)?),_=>None};
            elasticsearch_engine::ccr::apply_follower_change(&n.engine_on(on), &follower, &elasticsearch_engine::ccr::FollowerMetadataChange{mapping,settings,reopen,aliases})
        })));
        match result {
            Ok(()) => true,
            Err(e) => {
                *failed.lock() = Some(e);
                false
            }
        }
    });
    let published = node.metadata_service().run(run, true).await;
    if let Some(e) = failure.lock().take() {
        return Err(RemoteError::new(&e.kind, e.reason, e.status));
    }
    published.map_err(|e| RemoteError::new("failed_to_commit_cluster_state_exception", e, 503))
}

#[derive(Serialize)]
pub struct MappingUpdateReq {
    pub index: String,
    /// The dynamic update: only the mappers it adds (`mapping_update`).
    pub mapping_json: String,
    /// The index's cluster uuid (ES's `MappingUpdatedAction` carries the
    /// `Index`): a proposal from a deleted incarnation never merges into a
    /// re-created index of the same name. Appended: absent from an older
    /// sender's request.
    pub index_uuid: Option<String>,
    /// Exact parser delta; absent on legacy full-mapping senders.
    pub dynamic_delta: Option<String>,
    #[serde(skip)]
    pub(crate) memory: crate::replication::decode::DecodeMemory,
}

impl MappingUpdateReq {
    pub(crate) fn try_new(index: &str, mapping_json: &str, index_uuid: Option<&str>) -> elasticsearch_engine::error::Result<Self> {
        let memory = crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index,mapping_json,index_uuid.unwrap_or("")])?;
        Ok(Self { index: crate::mapping_update::copy_wire_text(index)?, mapping_json: crate::mapping_update::copy_wire_text(mapping_json)?,
            index_uuid: index_uuid.map(crate::mapping_update::copy_wire_text).transpose()?, dynamic_delta: None, memory })
    }
}
impl<'de> Deserialize<'de> for MappingUpdateReq {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self,D::Error> {
        use serde::de::{Error, MapAccess, SeqAccess, Visitor};
        use crate::replication::decode::{Text, OptionalText, DecodeMemory};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=MappingUpdateReq;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result { f.write_str("mapping update") }
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error> {
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"mapping envelope").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::invalid_length(0,&self))?;
                let mapping_json=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::invalid_length(1,&self))?;
                let index_uuid=match seq.next_element_seed(OptionalText(&mut memory)) {
                    Ok(value)=>value.flatten(),
                    Err(error) if error.to_string().to_ascii_lowercase().contains("unexpected end of file")=>None,
                    Err(error)=>return Err(error),
                };
                let dynamic_delta=match seq.next_element_seed(OptionalText(&mut memory)) {
                    Ok(value)=>value.flatten(),
                    Err(error) if error.to_string().to_ascii_lowercase().contains("unexpected end of file")=>None,
                    Err(error)=>return Err(error),
                };
                Ok(MappingUpdateReq{index,mapping_json,index_uuid,dynamic_delta,memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error> {
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"mapping envelope").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let (mut index,mut mapping_json,mut index_uuid)=(None,None,None);
                let mut uuid_seen=false;let mut dynamic_delta=None;let mut delta_seen=false;
                while let Some(key)=map.next_key_seed(Text(&mut memory))? {
                    match key.as_str() {
                        "index"=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
                        "mapping_json"=>{if mapping_json.is_some(){return Err(A::Error::duplicate_field("mapping_json"));} mapping_json=Some(map.next_value_seed(Text(&mut memory))?);},
                        "index_uuid"=>{if uuid_seen{return Err(A::Error::duplicate_field("index_uuid"));} uuid_seen=true;index_uuid=map.next_value_seed(OptionalText(&mut memory))?;},
                        "dynamic_delta"=>{if delta_seen{return Err(A::Error::duplicate_field("dynamic_delta"));} delta_seen=true;dynamic_delta=map.next_value_seed(OptionalText(&mut memory))?;},
                        _=>{map.next_value::<serde::de::IgnoredAny>()?;},
                    }
                }
                Ok(MappingUpdateReq{index:index.ok_or_else(||A::Error::missing_field("index"))?, mapping_json:mapping_json.ok_or_else(||A::Error::missing_field("mapping_json"))?,index_uuid,dynamic_delta,memory:DecodeMemory::new(memory)})
            }
        }
        d.deserialize_struct("MappingUpdateReq", &["index","mapping_json","index_uuid","dynamic_delta"], V)
    }
}



#[derive(Serialize, Deserialize)]
pub struct RecoveryStartResp {
    pub recovery_id: u64,
    pub info: RecoveryStartInfo,
}

/// Only offered for a committed replica prefix. The old allocation id is
/// the history identity; the newly assigned id identifies this recovery.


/// Recovery sources on this node predate file-based recovery.
const FILE_RECOVERY_UNSUPPORTED: &str = "recovery source does not support file-based recovery";

/// How a recovering copy gets its data (see `start_peer_recovery_with`).
enum PeerMode {
    /// Phase 1 copies the primary's commit as files.
    Files,
    /// Operations only: a reused local prefix, or (`None`) a document
    /// snapshot of the primary into an empty copy.
    Ops(Option<RecoveryReuseReq>),
}

/// An unbound recovery start is still restricted to this exact active primary
/// and an initializing target. Re-evaluate after every slow preparation phase.
fn recovery_start_matches(state: &ClusterState, node_id: &str, p: &PrimaryCtx, target_aid: &str, target_node: Option<&str>) -> bool {
    let Some(im) = state.metadata.indices.get(&p.index) else { return false };
    if im.uuid != p.svc.uuid || im.primary_term(p.shard) != p.term { return false; }
    let Some(table) = state.routing_table.shard_table(&p.index, p.shard) else { return false };
    table.active_primary().is_some_and(|source| source.allocation_id() == Some(p.allocation_id.as_str()) && source.node_id.as_deref() == Some(node_id))
        && table.copies.iter().any(|target| target.state == ShardState::Initializing
            && target.allocation_id() == Some(target_aid)
            && target.node_id.as_deref().is_some_and(|id| id != node_id && target_node.is_none_or(|expected| id == expected)))
}

fn reuse_request_matches(state: &ClusterState, node_id: &str, p: &PrimaryCtx, r: &RecoveryReuseReq) -> bool {
    if state.master_node_id.is_none() || state.metadata.cluster_uuid != r.cluster_uuid
        || p.index != r.index || p.shard != r.shard || p.allocation_id != r.source_allocation_id || p.term != r.primary_term
        || r.old_allocation_id == r.target_allocation_id || r.old_allocation_id == r.source_allocation_id
        || r.target_node_id == node_id
    { return false; }
    let Some(im) = state.metadata.indices.get(&r.index) else { return false };
    // ES `isTargetSameHistory`: the old copy shares this primary's history
    // uuid, or (copies recorded before history uuids) is still in sync.
    let same_history = r.history_uuid.as_ref().is_some_and(|h| shards::read_copy_record(p.svc.path())
        .is_some_and(|c| c.allocation_id == p.allocation_id && c.history_uuid.as_ref() == Some(h)));
    if im.uuid != r.index_uuid || im.primary_term(r.shard) != r.primary_term
        || !(same_history || im.in_sync_allocation_ids.get(&r.shard).is_some_and(|ids| ids.contains(&r.old_allocation_id)))
    { return false; }
    let Some(table) = state.routing_table.shard_table(&r.index, r.shard) else { return false };
    table.active_primary().is_some_and(|s| s.allocation_id() == Some(r.source_allocation_id.as_str()) && s.node_id.as_deref() == Some(node_id))
        && table.copies.iter().any(|s| !s.primary && s.state == ShardState::Initializing
            && s.allocation_id() == Some(r.target_allocation_id.as_str()) && s.node_id.as_deref() == Some(r.target_node_id.as_str()))
}

const PREFIX_REUSE_REJECTED: &str = "replica prefix requires full snapshot recovery";







fn remote(kind: &str, reason: impl Into<String>, status: u16) -> RemoteError {
    RemoteError::new(kind, reason.into(), status)
}

/// Own the node-wide optimizer slot until the blocking round finishes. Drop
/// also runs on panic/cancellation, allowing the next tick to retry.
struct VectorOptimizerLease(Arc<std::sync::atomic::AtomicBool>);

impl VectorOptimizerLease {
    fn claim(running: &Arc<std::sync::atomic::AtomicBool>) -> Option<Self> {
        use std::sync::atomic::Ordering;
        running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).ok().map(|_| Self(running.clone()))
    }
}

impl Drop for VectorOptimizerLease {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

pub struct ClusterNode {
    pub config: NodeConfig,
    pub local: DiscoveryNode,
    pub engine: Arc<Engine>,
    pub router: axum::Router,
    /// `xpack.security.http.ssl.*`: the REST listener's TLS server config,
    /// `None` = plain HTTP ([`crate::front::serve`]). Reloaded when its
    /// files change ([`watch_ssl_files`]).
    pub http_tls: Option<Arc<elasticsearch_ssl::Reloadable<elasticsearch_ssl::rustls::ServerConfig>>>,
    pub transport: TransportService,
    coord: OnceLock<CoordHandle>,
    state_tx: watch::Sender<Arc<ClusterState>>,
    pub uuids: Mutex<UuidMap>,
    /// Engines for shards 1.. of multi-shard indices (`shard_engines`).
    pub shard_engines: crate::shard_engines::ShardEngines,
    pub shards: ShardManager,
    pub master: Arc<MasterCtx>,
    pub started: Instant,
    pub cpu: CpuSampler,
    pub(crate) capacity_leases: Mutex<crate::capacity::lease::Registry>,
    pub mode: Mutex<(Mode, u64)>,
    me: OnceLock<Weak<ClusterNode>>,
    apply_lock: Mutex<()>,
    last_metadata_version: Mutex<Option<(String, u64)>>,
    /// In-flight per-copy background work (source probes, master reports),
    /// so each state application does not start another one.
    pending: Mutex<std::collections::BTreeSet<String>>,
    /// Serializes local shard-copy opens retried after a memory refusal
    /// (a breaker or the shared vector budget refused the open): under
    /// memory pressure the node opens such copies one at a time instead of
    /// all at once ([`ClusterNode::open_local_with_backoff`]).
    local_open_gate: Arc<tokio::sync::Mutex<()>>,
    /// Stale local copies kept because search contexts still read them
    /// (a scroll over a shard that relocated away): deleted once the last
    /// one is released (`release_retained_copies`).
    retained: Mutex<std::collections::BTreeSet<(String, u32)>>,
    /// A vector optimizer pass is running (one at a time; a seal can build
    /// HNSW graphs for many seconds).
    optimizing: Arc<std::sync::atomic::AtomicBool>,
    /// Per local shard copy (keyed by engine copy identity): the max seq_no
    /// last seen and since when, for the idle flush in `periodic_flush`.
    write_activity: Mutex<std::collections::HashMap<usize, (i64, Instant)>>,
    /// The stateless object store (`stateless.enabled` only).
    pub object_store: Option<Arc<crate::stateless::ObjectStore>>,
    /// The master's metadata service (`master_service`).
    metadata_service: OnceLock<crate::master_service::MetadataService>,
    /// Held while the master's engine metadata is mutated or re-synced by
    /// the metadata service, and while the applier mirrors state metadata
    /// into the engines. Lock order: `metadata_mutex`, then `apply_lock`.
    pub metadata_mutex: Mutex<()>,
    /// A metadata batch has changed the master's engine and its state is
    /// not yet committed and applied: the applier leaves engine 0 alone.
    engine_metadata_pending: std::sync::atomic::AtomicBool,
    /// A publication this node led failed: the master's engine must be
    /// rolled back to the applied state.
    metadata_resync: std::sync::atomic::AtomicBool,
    /// The master service's scratch engine (`scratch`), opened on first use.
    scratch: OnceLock<Result<crate::scratch::ScratchEngine, String>>,
    /// The state whose metadata engine 0 / the shard engines last mirrored
    /// (the base of the next incremental application).
    mirrored: Mutex<Option<Arc<ClusterState>>>,
    shards_mirrored: Mutex<Option<Arc<ClusterState>>>,
    /// Indices whose last application to engine 0 failed (retried).
    apply_retry: Mutex<std::collections::BTreeSet<String>>,
    /// The last application to a shard engine failed somewhere (retried).
    shard_apply_failed: std::sync::atomic::AtomicBool,
    /// Engine APIs changing engine 0 whose kept job has not published yet
    /// (`EngineApiGuard`).
    engine_api_inflight: std::sync::atomic::AtomicUsize,
    /// A kept change ended unpublished: engine 0 is re-synced to the
    /// committed state once no engine API is in flight.
    rollback_wanted: std::sync::atomic::AtomicBool,
    /// Engine APIs enter, and a rollback starts, under this lock: no API
    /// takes its snapshot while a rollback is wanted or running.
    api_gate: Mutex<()>,
    /// Engine APIs in flight that change one custom family only, by family
    /// (only that family's application is deferred).
    engine_api_families: Mutex<std::collections::HashMap<&'static str, usize>>,
    /// Per engine API in flight (by guard id): the index incarnations
    /// (name, cluster uuid) engine 0 held at its `before` snapshot. The
    /// applier never re-creates one engine 0 no longer holds while the API
    /// is in flight (`mirror::HeldDeletes`): that is the API's delete.
    engine_api_held: Mutex<std::collections::HashMap<u64, Vec<(String, String)>>>,
    engine_api_seq: std::sync::atomic::AtomicU64,
    /// Indices the applier left absent for an engine API in flight: once it
    /// is done they are applied again (the retry) if its kept job did not
    /// publish their delete (a failed API keeps no unpublished delete).
    engine_api_withheld: Mutex<std::collections::BTreeSet<String>>,
    /// A lifecycle pass changed (or is changing) the master's engine and
    /// its publication is pending: the applier leaves engine 0 alone.
    /// The cluster-wide rollover stats hook (engine 0 and the scratch copy).
    rollover_stats: OnceLock<elasticsearch_engine::datastream::RolloverStatsHook>,
    /// Rollover stats gathered before a metadata batch (inside a batch the
    /// hook reads only these: a fan-out there would hold every metadata
    /// change and dynamic mapping update behind it).
    rollover_cache: Mutex<std::collections::HashMap<String, (Instant, elasticsearch_engine::datastream::RolloverStats)>>,
    /// The cluster's ILM data / placement steps (`lifecycle_steps`).
    lifecycle_steps: OnceLock<Arc<dyn elasticsearch_engine::ilm::ClusterLifecycleSteps>>,
    /// Independent background duties (see `spawn_background`).
    background: Background,
    /// Fault injection (tests, fault matrix): see [`coord::Faults`].
    pub faults: coord::Faults,
    /// Background flush counters.
    pub flush_stats: FlushStats,
    pub(crate) adaptive_selection: crate::front::adaptive::Selection,
    /// The newest cluster state this node applied completely (metadata and
    /// shard copies): what an acknowledged request waits for.
    applied_version: std::sync::atomic::AtomicU64,
    /// The newest state whose application left no custom deferred for an
    /// engine API in flight (see `CUSTOMS_DEFERRED`): while a deferral is
    /// pending, an acknowledgement of a newer state waits for this to reach
    /// it (an acked change is visible to the requests that follow it).
    customs_settled: std::sync::atomic::AtomicU64,
    /// The committed segment format floor last recorded in the gateway
    /// (`gateway::store_committed_segment_format`): `(cluster uuid, floor)`.
    committed_segment_format: Mutex<Option<(String, u32)>>,
}

tokio::task_local! {
    /// Set for the task running an engine API and its kept job: waits for
    /// an acknowledgement there do not wait for the customs deferred for
    /// that very API (see `ClusterNode::applied_here`).
    pub static IN_ENGINE_API: ();
}

/// See [`ClusterNode::engine_api_in_flight`].
pub struct EngineApiGuard(Arc<ClusterNode>, Option<&'static str>, u64);

impl Drop for EngineApiGuard {
    fn drop(&mut self) {
        // The API's kept job has run (published or failed): its deletes are
        // no longer held. One the state still has (not published: the API
        // or its publication failed) is applied to engine 0 again.
        let tracked = self.0.engine_api_held.lock().remove(&self.2).is_some();
        if tracked {
            let withheld = std::mem::take(&mut *self.0.engine_api_withheld.lock());
            if !withheld.is_empty() {
                let state = self.0.state();
                let again: Vec<String> = withheld.into_iter().filter(|n| state.metadata.indices.contains_key(n)).collect();
                self.0.apply_retry.lock().extend(again);
            }
        }
        match self.1 {
            None => {
                self.0.engine_api_inflight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }
            Some(family) => {
                let mut families = self.0.engine_api_families.lock();
                if let Some(n) = families.get_mut(family) {
                    *n -= 1;
                    if *n == 0 {
                        families.remove(family);
                    }
                }
            }
        }
    }
}

/// Background flush counters (node-wide).
#[derive(Default)]
pub struct FlushStats {
    /// Flushes for the translog size threshold.
    pub size: std::sync::atomic::AtomicU64,
    /// Periodic flushes (translog age, idle copy).
    pub periodic: std::sync::atomic::AtomicU64,
    /// Primary flushes skipped because the replication round before them
    /// did not complete.
    pub skipped_replication: std::sync::atomic::AtomicU64,
}

/// State of the independent background tasks: which shard copies have a
/// replication top-up or flush in flight, the concurrency caps, and the
/// master's in-flight side work.
struct Background {
    replicating: Arc<Mutex<std::collections::HashSet<usize>>>,
    flushing: Arc<Mutex<std::collections::HashSet<usize>>>,
    replication_permits: Arc<tokio::sync::Semaphore>,
    flush_permits: Arc<tokio::sync::Semaphore>,
    /// Periodic (age, idle) flushes: one at a time node-wide, as the
    /// single maintenance loop ran them (a burst of copies all due on one
    /// tick must not seal at once next to searches).
    periodic_flush_permit: Arc<tokio::sync::Semaphore>,
    /// Per primary (by context identity): size-due flushes skipped in a
    /// row because their replication round failed.
    size_flush_skips: Arc<Mutex<std::collections::HashMap<usize, u32>>>,
    reroute: Arc<std::sync::atomic::AtomicBool>,
    registry: Arc<std::sync::atomic::AtomicBool>,
    stores: Arc<std::sync::atomic::AtomicBool>,
    cluster_info: Arc<std::sync::atomic::AtomicBool>,
    capacity_info: Arc<std::sync::atomic::AtomicBool>,
    lifecycle: Arc<std::sync::atomic::AtomicBool>,
    apply_retry: Arc<std::sync::atomic::AtomicBool>,
}

/// Replication top-ups running at once (each holds only its own shard).
const REPLICATION_TOPUP_CONCURRENCY: usize = 8;
/// At most this often a replica commits locally on its WAL size while the
/// global checkpoint does not let it trim the WAL.
const REPLICA_LOCAL_FLUSH_INTERVAL: Duration = Duration::from_secs(30);

/// `apply_retry` entry: customs not applied while an engine API was in
/// flight (applied again by the retry timer).
const CUSTOMS_DEFERRED: &str = "_customs_deferred";

/// How long gathered rollover stats are used inside metadata batches.
const ROLLOVER_STATS_FRESH: Duration = Duration::from_secs(60);

/// The indices a rollover may be judged on: every data stream's write
/// index and every index with a rollover alias.
fn rollover_candidates(state: &ClusterState) -> Vec<String> {
    let mut out: Vec<String> = state.metadata.customs.get(crate::mirror::DATA_STREAMS_CUSTOM).and_then(serde_json::Value::as_object).map(|m| {
        m.values().filter_map(|d| d["indices"].as_array().and_then(|a| a.last()).and_then(serde_json::Value::as_str).map(str::to_string)).collect()
    }).unwrap_or_default();
    out.extend(state.metadata.indices.iter().filter(|(_, im)| im.setting("index.lifecycle.rollover_alias").is_some_and(|a| !a.is_empty())).map(|(n, _)| n.clone()));
    out.sort();
    out.dedup();
    out
}

/// The write index a `_rollover` of `target` (a data stream or an alias)
/// judges its conditions on.
pub(crate) fn rollover_write_index(state: &ClusterState, target: &str) -> Option<String> {
    if let Some(d) = state.metadata.customs.get(crate::mirror::DATA_STREAMS_CUSTOM).and_then(|m| m.get(target)) {
        return d["indices"].as_array().and_then(|a| a.last()).and_then(serde_json::Value::as_str).map(str::to_string);
    }
    let holders: Vec<(&String, &serde_json::Value)> = state.metadata.indices.iter().filter_map(|(n, im)| im.aliases.get(target).map(|a| (n, a))).collect();
    holders.iter().find(|(_, a)| a.get("is_write_index").and_then(serde_json::Value::as_bool) == Some(true)).or(if holders.len() == 1 { holders.first() } else { None }).map(|(n, _)| (*n).clone())
}

/// Bound of a background replication round (top-up, pre-flush): a replica
/// on a stalled node delays its own shard's next round, nothing else.
const BACKGROUND_REPLICATION_TIMEOUT: Duration = Duration::from_secs(10);

impl Background {
    fn new() -> Background {
        // ES's `flush` pool: half the processors, at most 5.
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        Background {
            replicating: Default::default(),
            flushing: Default::default(),
            replication_permits: Arc::new(tokio::sync::Semaphore::new(REPLICATION_TOPUP_CONCURRENCY)),
            flush_permits: Arc::new(tokio::sync::Semaphore::new((cpus / 2).clamp(1, 5))),
            periodic_flush_permit: Arc::new(tokio::sync::Semaphore::new(1)),
            size_flush_skips: Default::default(),
            reroute: Default::default(),
            registry: Default::default(),
            stores: Default::default(),
            cluster_info: Default::default(),
            capacity_info: Default::default(),
            lifecycle: Default::default(),
            apply_retry: Default::default(),
        }
    }
}

/// One in-flight background job for one shard copy; the key is released
/// when the job ends (also on panic or cancellation).
struct CopyLease {
    set: Arc<Mutex<std::collections::HashSet<usize>>>,
    key: usize,
}

impl CopyLease {
    fn claim(set: &Arc<Mutex<std::collections::HashSet<usize>>>, key: usize) -> Option<CopyLease> {
        set.lock().insert(key).then(|| CopyLease { set: set.clone(), key })
    }
}

impl Drop for CopyLease {
    fn drop(&mut self) {
        self.set.lock().remove(&self.key);
    }
}

/// Bind the first free port of a range.
/// `xpack.security.transport.ssl.*` from the node settings (paths relative
/// to `path.conf`). `secure_key_passphrase` is a keystore setting in ES; here
/// it may also come from `-E` (it is filtered from every settings output).
/// `xpack.security.http.ssl.*` (ES `SSLService` + the security plugin's
/// HTTP transport): `Ok(None)` when disabled; invalid settings or
/// unreadable key material fail the start. As in ES, HTTPS is served only
/// by the security layer: with `xpack.security.enabled: false` the settings
/// are still validated but the REST layer stays plain HTTP.
fn http_tls(settings: &crate::config::Settings) -> Result<Option<Arc<elasticsearch_ssl::Reloadable<elasticsearch_ssl::rustls::ServerConfig>>>, String> {
    let get = |k: &str| match settings.get(k) {
        Some(serde_json::Value::String(v)) => Some(v.clone()),
        Some(serde_json::Value::Null) | None => None,
        Some(other) => Some(other.to_string()),
    };
    let conf = std::path::PathBuf::from(crate::config::get_str(settings, "path.conf").unwrap_or("config"));
    let tls = elasticsearch_ssl::http_server_reloadable(&get, &|k| crate::config::get_list(settings, k), &conf)?;
    if tls.is_some() && get("xpack.security.enabled").as_deref() == Some("false") {
        tracing::warn!("[xpack.security.http.ssl.enabled] is true but security is disabled ([xpack.security.enabled] is false): HTTP is served without TLS");
        return Ok(None);
    }
    Ok(tls)
}

/// `reindex.ssl.*` is validated at start (ES builds the reindex SSL context
/// when the module loads): the engine's context for these settings
/// (`elasticsearch_engine::remote::reindex_context`, which remote reindexing
/// connects with) is built here, so the SSL watcher reloads it.
fn reindex_ssl(settings: &crate::config::Settings) -> Result<Option<Arc<elasticsearch_ssl::Reloadable<elasticsearch_ssl::rustls::ClientConfig>>>, String> {
    if !settings.keys().any(|k| k.starts_with(elasticsearch_ssl::REINDEX_SSL)) {
        return Ok(None);
    }
    let map: serde_json::Map<String, serde_json::Value> = settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    elasticsearch_engine::remote::reindex_context(&map).map(Some)
}

/// `resource.reload.enabled` / `resource.reload.interval.high`: `None` when
/// SSL files are not watched.
fn ssl_reload_interval(settings: &crate::config::Settings) -> Result<Option<Duration>, String> {
    let get = |k: &str| match settings.get(k) {
        Some(serde_json::Value::String(v)) => Some(v.clone()),
        Some(serde_json::Value::Null) | None => None,
        Some(other) => Some(other.to_string()),
    };
    elasticsearch_ssl::reload_interval(&get, &elasticsearch_engine::date::parse_time_value)
}

/// ES `SSLConfigurationReloader`: poll every SSL context's files (HTTP,
/// transport, reindex) and swap a rebuilt context in when one changes; new
/// connections use it, established ones keep theirs, a failed rebuild is
/// logged and the old context stays.
fn watch_ssl_files(interval: Option<Duration>, contexts: Vec<Arc<dyn elasticsearch_ssl::Reload>>) {
    let Some(interval) = interval else { return };
    if let Err(error) = elasticsearch_ssl::watch(contexts, interval) {
        tracing::warn!(%error, "cannot watch the SSL configuration files");
    }
}

fn transport_tls(settings: &crate::config::Settings) -> Result<Option<elasticsearch_transport::TlsConfig>, String> {
    let get = |k: &str| match settings.get(k) {
        Some(serde_json::Value::String(v)) => Some(v.clone()),
        Some(serde_json::Value::Null) | None => None,
        Some(other) => Some(other.to_string()),
    };
    let conf = std::path::PathBuf::from(crate::config::get_str(settings, "path.conf").unwrap_or("config"));
    elasticsearch_transport::TlsConfig::from_settings(&get, &|k| crate::config::get_list(settings, k), &conf)
}

/// ES `PersistedClusterStateService` refuses an on-disk state written under
/// another node id. Here: a persisted voting configuration that does not
/// name this node, when this node's own entry in it carries another id (or
/// single-node discovery, where this node is the only possible voter),
/// means the node id and the cluster state come from different nodes (a
/// data path emptied and reused while a previous process kept writing into
/// it). Such a node could never elect itself or be elected: it sat in
/// "master not discovered" forever. Fail the start with the reason instead.
fn check_state_owner(state: &ClusterState, node_id: &str, config: &NodeConfig) -> std::io::Result<()> {
    let voters = &state.coordination.last_accepted_config.0;
    if voters.is_empty() || voters.contains(node_id) {
        return Ok(());
    }
    let previous = state.nodes.values().find(|n| n.name == config.node_name && n.id != node_id);
    if !config.single_node && previous.is_none() {
        return Ok(());
    }
    Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
        "the cluster state in [{}] belongs to node id {} (voting configuration {:?}) but this data path's node id is [{node_id}]: \
         the node id and the cluster state were written by different nodes (was the data path emptied or replaced while another \
         process was still using it?); this node could never be elected. Restore the data path, or remove it to start a new node",
        config.data_path.join("_cluster").display(),
        previous.map(|n| format!("[{}]", n.id)).unwrap_or_else(|| "[unknown]".into()),
        voters)))
}

async fn bind_first(ip: std::net::IpAddr, ports: &[u16]) -> std::io::Result<tokio::net::TcpListener> {
    let mut last = None;
    for p in ports {
        match tokio::net::TcpListener::bind(SocketAddr::new(ip, *p)).await {
            Ok(l) => return Ok(l),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("empty port range")))
}

impl ClusterNode {
    /// Start everything except the HTTP listener; returns the node and the
    /// bound HTTP listener.
    pub async fn start(config: NodeConfig) -> Result<(Arc<ClusterNode>, tokio::net::TcpListener), Box<dyn std::error::Error + Send + Sync>> {
        let writer_setting = std::env::var("RUSTIC_SEGMENT_WRITE_VERSION");
        let writer_setting = match &writer_setting {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string()).into()),
        };
        crate::internal_capabilities::validate_writer_setting(writer_setting)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
        // TLS settings fail the start before anything is bound or written.
        let http_tls = http_tls(&config.settings)?;
        let reindex_ssl = reindex_ssl(&config.settings)?;
        let ssl_reload = ssl_reload_interval(&config.settings)?;
        std::fs::create_dir_all(&config.data_path)?;
        gateway::lock_data_path(&config.data_path, Duration::from_secs(10))?;
        // The listeners first: a second process started while the previous
        // one still serves (ports in use) fails here, before it writes
        // anything (it used to create a fresh node id in a data path the
        // live process was still writing).
        let http = bind_first(config.http_bind, &config.http_ports).await?;
        let http_port = http.local_addr()?.port();
        let transport_probe = bind_first(config.transport_bind, &config.transport_ports).await?;
        let transport_addr = transport_probe.local_addr()?;
        // The probe keeps the port until the transport binds it below: freed
        // earlier, the gateway/settings/TLS work in between left a window in
        // which another socket (a parallel process's ephemeral port) could
        // take it, and the start failed with "address already in use".
        let node_id = gateway::load_or_create_node_id(&config.data_path)?;
        // Local files are read/written by this process. Peer capability checks
        // belong at live membership/write and file-transfer boundaries; an
        // offline historical peer must not prevent a quorum from restarting.
        let gateway = Gateway::open(&config.data_path, &config.cluster_name)?;
        check_state_owner(&gateway.persisted().last_accepted_state, &node_id, &config)?;
        // Until a master publishes, write in the floor this node last
        // applied from a committed state (never one it only accepted: an
        // uncommitted raise may be superseded): the cluster's floor never
        // lowers, and a node of a cluster whose floor this build cannot
        // read must not start.
        let committed_segment_format = gateway.committed_segment_format()?;
        if let Some((_, floor)) = committed_segment_format.clone() {
            let read = elasticsearch_engine::rustic_core::codec::VERSION;
            if floor > read {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                    "the cluster's segment format is v{floor} but this build reads segment formats through v{read}; downgrading this node is not supported")).into());
            }
            elasticsearch_engine::rustic_core::codec::apply_cluster_segment_format(floor);
        }
        // The engine reads the cluster name from the environment.
        // SAFETY: called once at startup before other threads read the env.
        unsafe { std::env::set_var("ES_CLUSTER_NAME", &config.cluster_name) };
        let node_settings: serde_json::Map<String, serde_json::Value> = config.settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        let mut attributes = config.attributes.clone();
        crate::internal_capabilities::advertise(&mut attributes);
        crate::rest::nodes::advertise_gpu(&mut attributes, elasticsearch_engine::gpu::supported());
        let local = DiscoveryNode {
            id: node_id,
            name: config.node_name.clone(),
            ephemeral_id: elasticsearch_cluster::state::random_id(),
            transport_address: SocketAddr::new(config.transport_publish, transport_addr.port()).to_string(),
            http_address: SocketAddr::new(config.http_publish, http_port).to_string(),
            host: config.transport_publish.to_string(),
            ip: config.transport_publish.to_string(),
            roles: config.roles.clone(),
            attributes,
            version: ES_VERSION.to_string(),
        };
        let mut tcfg = TransportConfig::new(config.cluster_name.clone(), local.clone(), SocketAddr::new(config.transport_bind, transport_addr.port()));
        tcfg.publish_addr = Some(SocketAddr::new(config.transport_publish, transport_addr.port()));
        tcfg.settings = crate::config::transport_settings(&config.settings)?;
        tcfg.tls = transport_tls(&config.settings)?;
        let frame_admission: elasticsearch_transport::FrameAdmission = Arc::new(|bytes| {
            elasticsearch_engine::memory::breakers().reserve_native(bytes as u64, "transport protocol buffer")
                .map(|reservation| Box::new(reservation.into_result_owner()) as Box<dyn Send + Sync>)
                .map_err(|error| RemoteError::new(error.kind, error.reason, error.status))
        });
        drop(transport_probe);
        let transport = TransportService::bind_with_admission(tcfg, Some(frame_admission)).await?;
        {
            let mut contexts: Vec<Arc<dyn elasticsearch_ssl::Reload>> = Vec::new();
            if let Some(http) = &http_tls {
                contexts.push(http.clone());
            }
            contexts.extend(transport.ssl_reloader());
            if let Some(reindex) = &reindex_ssl {
                contexts.push(reindex.clone());
            }
            watch_ssl_files(ssl_reload, contexts);
        }
        crate::capability_probe::register(&transport)?;
        // The transport accepts connections from here on, the coordinator
        // registers its actions later: until then a peer's coordination
        // message gets a retryable "not ready" (never `action_not_found`,
        // which would make it speak the legacy format to this node).
        for action in [coord::ACTION, coord::ACTION_V2] {
            transport.register_raw_handler(action, |_body: bytes::Bytes| async {
                Err::<bytes::Bytes, RemoteError>(RemoteError::new("node_not_ready_exception", "the node is starting", 503))
            });
        }
        let local = transport.local_node().clone();
        let opened = (|| -> Result<_, Box<dyn std::error::Error + Send + Sync>> {
            let engine = Engine::open_named(&config.data_path.join("engine"), &config.node_name)?;
            // `elasticsearch.yml` / `-E` settings the engine reads as node
            // settings (e.g. `xpack.license.self_generated.type`).
            *engine.node_settings.write() = node_settings.clone();
            crate::install_breaker_release_hook();
            elasticsearch_engine::memory::validate_node_settings(&node_settings)?;
            elasticsearch_engine::memory::configure(&engine);
            elasticsearch_engine::script::rate_limit::configure(&engine)?;
            elasticsearch_http::configure_processors(&node_settings);
            // `indices.query.bool.max_nested_depth` (static; invalid fails the start).
            elasticsearch_engine::query::limits::configure_nested_depth(&node_settings)?;
            engine.security.configure(elasticsearch_engine::security::SecConfig::from_node_settings(&node_settings)?);
            // What the REST layer actually serves, not just the setting.
            engine.security.cfg.write().http_ssl = http_tls.is_some();
            // The file realm follows changes to its files (ES reloads them;
            // `elasticsearch-reset-password` relies on it).
            if engine.security.enabled() {
                let conf = std::path::PathBuf::from(crate::config::get_str(&config.settings, "path.conf").unwrap_or("config"));
                engine.security.watch_files(conf, Duration::from_secs(1));
            }
            // An invalid `indexing_pressure.memory.limit` fails the start (ES).
            elasticsearch_engine::indexing_pressure::configure(&node_settings)?;
            // `indices.memory.index_buffer_size` (bytes or a bounded %).
            elasticsearch_engine::memory::configure_indexing_buffer(&node_settings)?;
            // `GPUPlugin.GpuModeBootstrapCheck` (enforced as ES enforces
            // bootstrap checks: not single-node and not published on loopback).
            let enforce = !config.single_node && !config.transport_publish.is_loopback();
            elasticsearch_engine::gpu::bootstrap_check(&node_settings, enforce)?;
            // ES `BootstrapChecks` (process limits, memory lock, discovery,
            // and transport TLS while security is enabled).
            crate::bootstrap_checks::run(&config.settings, engine.security.enabled(), &[config.transport_bind], config.transport_publish, config.single_node)?;
            // The cluster drives engine 0's lifecycle (`lifecycle_tick`): on the
            // master its passes are published like any metadata change.
            engine.ilm.drive_externally();
            let router = elasticsearch_http::router(engine.clone());
            Ok((engine, router))
        })();
        let (engine, router) = match opened {
            Ok(opened) => opened,
            Err(error) => { transport.close().await; return Err(error); }
        };

        *engine.transport_address.write() = local.transport_address.clone();

        // `cluster.publish.timeout`, `cluster.fault_detection.*`,
        // `cluster.election.*`, `cluster.auto_shrink_voting_configuration`,
        // ... from the node settings (they were ignored: every node ran
        // with the defaults); discovery comes from the parsed config below.
        let coordination_settings: BTreeMap<String, String> = config.settings.iter()
            .filter(|(k, _)| !matches!(k.as_str(), "discovery.seed_hosts" | "cluster.initial_master_nodes"))
            .map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())))
            .collect();
        let mut cs = CoordinationSettings::from_flat(&coordination_settings)?;
        let find_peers_interval_ms = cs.find_peers_interval_ms;
        cs.cluster_name = config.cluster_name.clone();
        cs.seed_hosts = config.seed_addresses();
        cs.initial_master_nodes = if config.single_node || (config.discovery_unconfigured() && config.is_master_eligible()) {
            vec![local.name.clone()]
        } else {
            config.initial_master_nodes.clone()
        };
        if let Some(b) = config.settings.get("cluster.no_master_block").and_then(|v| v.as_str()).and_then(NoMasterBlock::parse) {
            cs.no_master_block = b;
        }
        let master_ctx = Arc::new(MasterCtx::new(node_settings.clone()));
        // The last accepted state is not an applied state of this process:
        // until a master publishes (or this node wins an election), the node
        // knows no master. Exposing the persisted master id (this node's own
        // after a single-node restart) made `_cluster/health` answer green
        // from the previous run's routing table before the election failed
        // those copies (ES starts every node with no master and the state
        // not recovered).
        let initial = {
            let mut s = gateway.persisted().last_accepted_state.clone();
            s.master_node_id = None;
            Arc::new(s)
        };
        let (state_tx, _) = watch::channel(initial);
        let uuids = UuidMap::load(&config.data_path.join("_cluster/uuids.json"));
        let shard_engines = crate::shard_engines::ShardEngines::open(&config.data_path, node_settings, engine.clone());
        let object_store = {
            let flat: BTreeMap<String, String> = config.settings.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
            let ctx = elasticsearch_snapshots::RepositoryContext {
                path_repo: config.repo_paths.clone(),
                allowed_urls: crate::config::get_list(&config.settings, "repositories.url.allowed_urls"),
                s3_clients: elasticsearch_snapshots::S3ClientSettings::from_node_settings(&flat),
            };
            crate::stateless::ObjectStore::open(&config.settings, &ctx).await?
        };

        let node = Arc::new(ClusterNode {
            config,
            local: local.clone(),
            engine,
            router,
            http_tls,
            transport: transport.clone(),
            coord: OnceLock::new(),
            state_tx,
            uuids: Mutex::new(uuids),
            shard_engines,
            shards: ShardManager::default(),
            master: master_ctx.clone(),
            started: Instant::now(),
            cpu: CpuSampler::default(),
            capacity_leases: Mutex::new(Default::default()),
            mode: Mutex::new((Mode::Candidate, 0)),
            me: OnceLock::new(),
            apply_lock: Mutex::new(()),
            last_metadata_version: Mutex::new(None),
            pending: Mutex::new(Default::default()),
            local_open_gate: Arc::new(tokio::sync::Mutex::new(())),
            retained: Mutex::new(Default::default()),
            optimizing: Default::default(),
            write_activity: Mutex::new(Default::default()),
            object_store,
            metadata_service: OnceLock::new(),
            metadata_mutex: Mutex::new(()),
            engine_metadata_pending: Default::default(),
            metadata_resync: Default::default(),
            scratch: OnceLock::new(),
            rollover_stats: OnceLock::new(),
            rollover_cache: Mutex::new(Default::default()),
            lifecycle_steps: OnceLock::new(),
            mirrored: Mutex::new(None),
            shards_mirrored: Mutex::new(None),
            apply_retry: Mutex::new(Default::default()),
            shard_apply_failed: Default::default(),
            engine_api_inflight: Default::default(),
            engine_api_families: Default::default(),
            engine_api_held: Mutex::new(Default::default()),
            engine_api_seq: Default::default(),
            engine_api_withheld: Mutex::new(Default::default()),
            rollback_wanted: Default::default(),
            api_gate: Mutex::new(()),
            background: Background::new(),
            faults: Default::default(),
            flush_stats: Default::default(),
            adaptive_selection: Default::default(),
            applied_version: Default::default(),
            customs_settled: Default::default(),
            committed_segment_format: Mutex::new(committed_segment_format),
        });
        let _ = node.me.set(Arc::downgrade(&node));
        // Capability proofs are otherwise only collected when membership
        // changes: re-probe members that still lack one, in the background.
        tokio::spawn(crate::capability_probe::reprobe_missing(node.transport.clone(), node.subscribe()));
        let _ = node.metadata_service.set(crate::master_service::MetadataService::start(Arc::downgrade(&node)));
        // Dynamic mapping updates are committed by the master before the
        // documents that need them are indexed (`mapping_update`).
        // Rollover conditions (REST and lifecycle) on this node's engine 0 are
        // judged by the cluster-wide primary stats: the local copy of the
        // index may be a metadata-only shell.
        {
            let weak = Arc::downgrade(&node);
            let runtime = tokio::runtime::Handle::current();
            let stats: elasticsearch_engine::datastream::RolloverStatsHook = Arc::new(move |index: &str| {
                let node = weak.upgrade()?;
                if crate::master_service::on_metadata_service() {
                    return node.cached_rollover_stats(index);
                }
                let index = index.to_string();
                let (tx, rx) = std::sync::mpsc::channel();
                runtime.spawn(async move {
                    let _ = tx.send(crate::front::cluster_rollover_stats(&node, &index).await);
                });
                // Engine threads block here (never an async worker).
                rx.recv_timeout(Duration::from_secs(10)).ok().flatten()
            });
            let _ = node.rollover_stats.set(stats.clone());
            node.engine.set_rollover_stats_hook(Some(stats));
            let steps: Arc<dyn elasticsearch_engine::ilm::ClusterLifecycleSteps> = Arc::new(crate::lifecycle_steps::ClusterSteps::new(Arc::downgrade(&node)));
            node.engine.ilm.set_cluster_steps(Some(steps.clone()));
            let _ = node.lifecycle_steps.set(steps);
        }
        // CCR follower metadata (the leader's mapping, settings, aliases)
        // is a metadata change like any other: a dry run on the master's
        // copy, published, then applied by every node's applier (never
        // written onto an engine before its commit). A follow task on any
        // node sends it to the elected master; the task's worker never waits
        // for it: the change is reported pending, and the next sync (a
        // second later) finds it applied or sends it again.
        {
            let weak = Arc::downgrade(&node);
            let runtime = tokio::runtime::Handle::current();
            let pending: Arc<Mutex<std::collections::BTreeSet<control::FollowerKey>>> = Default::default();
            // The last failed commit per follower: returned to the next sync
            // (a read exception, fatal per ES's retry rules), and resends
            // paced (doubling, at most a minute).
            let failures: Arc<Mutex<std::collections::BTreeMap<control::FollowerKey, (elasticsearch_engine::error::EsError, Instant, u32, bool)>>> = Default::default();
            let hook: elasticsearch_engine::ccr::FollowerMetadataHook = Arc::new(move |follower: &str, change: elasticsearch_engine::ccr::FollowerMetadataChange| {
                use elasticsearch_engine::error::EsError;
                let Some(node) = weak.upgrade() else { return Ok(()) };
                if crate::master_service::on_metadata_service() {
                    return Err(EsError::new(503, "illegal_state_exception", "follower metadata cannot be changed inside a metadata batch"));
                }
                let pending_reply = || Err(EsError::new(503, elasticsearch_engine::ccr::FOLLOWER_METADATA_PENDING, format!("the metadata change of follower [{follower}] is being committed")));
                {
                    let mut f = failures.lock();
                    if let Some((e, at, n, reported)) = f.get_mut(follower) {
                        if !*reported {
                            *reported = true;
                            return Err(e.try_clone_owned().unwrap_or_else(|refusal| refusal));
                        }
                        if at.elapsed() < Duration::from_secs((1u64 << (*n).min(6)).min(60)) {
                            return pending_reply();
                        }
                    }
                }
                let req = FollowerMetadataReq::of(follower, &change)?;
                let follower=control::FollowerKey::try_new(follower)?;
                if !pending.lock().insert(follower.clone()) {
                    return pending_reply();
                }
                let pending = pending.clone();
                let failures = failures.clone();
                runtime.spawn(async move {
                    let r: Result<u64, RemoteError> = if node.is_master() {
                        commit_follower_change(&node, req).await
                    } else {
                        node.to_master::<_, u64>(ACTION_FOLLOWER_METADATA, &req, Duration::from_secs(30)).await
                    };
                    match r {
                        // Pending until this node applied it (the next sync
                        // then sees no change, instead of sending it again).
                        Ok(version) => {
                            let mut rx = node.subscribe();
                            let _ = tokio::time::timeout(Duration::from_secs(30), async {
                                while rx.borrow_and_update().version < version {
                                    if rx.changed().await.is_err() {
                                        break;
                                    }
                                }
                            })
                            .await;
                        }
                        Err(e) => {
                            tracing::warn!(%follower, "follower metadata change not committed: {}", e.reason);
                            let mut f = failures.lock();
                            let n = f.get(&follower).map_or(0, |(_, _, n, _)| *n) + 1;
                            f.insert(follower.clone(), (elasticsearch_engine::memory::native_control(||EsError::try_owned(e.status, &e.error_type, &e.reason)).unwrap_or_else(|refusal| refusal), Instant::now(), n, false));
                            drop(f);
                            control::remove_pending(&mut pending.lock(),&follower);
                            return;
                        }
                    }
                    control::remove_failure(&mut failures.lock(),&follower);
                    control::remove_pending(&mut pending.lock(),&follower);
                });
                pending_reply()
            });
            elasticsearch_engine::ccr::set_follower_metadata_hook(&node.engine, Some(hook));
        }
        let hook: Arc<dyn elasticsearch_engine::index::MappingUpdateHook> = Arc::new(crate::mapping_update::ClusterMappingHook::new(Arc::downgrade(&node)));
        node.engine.set_mapping_update_hook(Some(hook.clone()));
        node.shard_engines.set_mapping_update_hook(hook);
        node.register_handlers();

        let coordinator = Coordinator::new(cs, local, gateway.persisted().clone(), rand::random(), Box::new(Updater(master_ctx)));
        let handle = coord::start(coordinator, gateway, transport, node.clone() as Arc<dyn Applier>, node.faults.clone());
        if node.config.seed_hosts_file().is_some() {
            // ES's `FileBasedSeedHostsProvider` re-reads `unicast_hosts.txt`
            // on every discovery round (`discovery.find_peers_interval`).
            let (handle, config, node) = (handle.clone(), node.config.clone(), Arc::downgrade(&node));
            let interval = Duration::from_millis(find_peers_interval_ms.max(100));
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tick.tick().await;
                    if node.upgrade().is_none() {
                        break;
                    }
                    let config = config.clone();
                    if let Ok(hosts) = tokio::task::spawn_blocking(move || config.current_seed_addresses()).await {
                        handle.set_seed_hosts(hosts);
                    }
                }
            });
        }
        let _ = node.coord.set(handle);
        node.clone().spawn_background();
        Ok((node, http))
    }

    pub fn me(&self) -> Arc<ClusterNode> {
        self.me.get().and_then(|w| w.upgrade()).expect("node alive")
    }

    pub fn coord(&self) -> &CoordHandle {
        self.coord.get().expect("coordinator started")
    }

    /// The master's metadata service (`master_service`).
    pub fn metadata_service(&self) -> &crate::master_service::MetadataService {
        self.metadata_service.get().expect("metadata service started")
    }

    pub fn state(&self) -> Arc<ClusterState> {
        self.state_tx.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<Arc<ClusterState>> {
        self.state_tx.subscribe()
    }

    /// Elected, and the applied state is of the current term (a node
    /// re-elected while its applier still holds an older state is not yet
    /// master for anything that reads that state).
    pub fn is_master(&self) -> bool {
        let (mode, term) = *self.mode.lock();
        leads(&self.state(), &self.local.id, mode, term)
    }

    pub fn address_of(&self, node_id: &str) -> Option<SocketAddr> {
        self.state().nodes.get(node_id).and_then(|n| n.transport_address.parse().ok())
    }

    /// Send a typed request to the elected master (or fail with ES's
    /// `master_not_discovered_exception`).
    pub async fn to_master<Req: Serialize + Sync, Resp: serde::de::DeserializeOwned>(&self, action: &str, req: &Req, timeout: Duration) -> Result<Resp, RemoteError> {
        let deadline = Instant::now() + timeout;
        loop {
            let st = self.state();
            if let Some(addr) = st.master().and_then(|m| m.transport_address.parse::<SocketAddr>().ok()) {
                if action == ACTION_MAPPING_UPDATE_V2 && !st.master().is_some_and(|m|
                    crate::internal_capabilities::supports(m, crate::internal_capabilities::Capability::MappingUuidV2)) {
                    return Err(remote("unsupported_operation_exception",
                        "the elected master does not advertise UUID-fenced dynamic mapping updates; upgrade the master before adding dynamic fields", 503));
                }
                if action == ACTION_MAPPING_UPDATE_V2 {
                    crate::capability_probe::verify(&self.transport, st.master().unwrap(), true).await
                        .map_err(|e| remote("unsupported_operation_exception", e, 503))?;
                }
                match self.transport.send_request(addr, action, req, deadline.saturating_duration_since(Instant::now())).await {
                    Ok(r) => return Ok(r),
                    Err(elasticsearch_transport::TransportError::Remote(e)) if e.error_type != "not_master_exception" => return Err(e),
                    Err(e) if Instant::now() >= deadline => return Err(e.to_remote_error()),
                    Err(_) => {}
                }
            } else if Instant::now() >= deadline {
                return Err(remote("master_not_discovered_exception", "", 503));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Send a dynamic mapping update to the elected master, in the action
    /// that master understands. Masters upgrade last (ES order): one that
    /// does not advertise UUID-fenced updates still takes the original
    /// action, whose request carries the index UUID and whose handler fences
    /// on it, so adding a field keeps working for the whole upgrade. The
    /// choice is made per attempt, against the master being addressed.
    pub(crate) async fn mapping_update_to_master(&self, legacy: &MappingUpdateReq, fenced: &crate::mapping_update::UuidMappingUpdate, timeout: Duration) -> Result<u64, RemoteError> {
        let deadline = Instant::now() + timeout;
        loop {
            let st = self.state();
            if let Some((master, addr)) = st.master().and_then(|m| m.transport_address.parse::<SocketAddr>().ok().map(|a| (m, a))) {
                let left = deadline.saturating_duration_since(Instant::now());
                let sent = if mapping_update_action(master) == ACTION_MAPPING_UPDATE_V2 {
                    crate::capability_probe::verify(&self.transport, master, true).await
                        .map_err(|e| remote("unsupported_operation_exception", e, 503))?;
                    self.transport.send_request(addr, ACTION_MAPPING_UPDATE_V2, fenced, left).await
                } else {
                    self.transport.send_request(addr, ACTION_MAPPING_UPDATE, legacy, left).await
                };
                match sent {
                    Ok(r) => return Ok(r),
                    Err(elasticsearch_transport::TransportError::Remote(e)) if e.error_type != "not_master_exception" => return Err(e),
                    Err(e) if Instant::now() >= deadline => return Err(e.to_remote_error()),
                    Err(_) => {}
                }
            } else if Instant::now() >= deadline {
                return Err(remote("master_not_discovered_exception", "", 503));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    // ------------------------------------------------------------ handlers

    fn register_handlers(self: &Arc<Self>) {
        let t = &self.transport;
        t.set_inbound_gate(Arc::new(inbound_gate));
        let n = self.clone();
        t.register_handler(ACTION_SHARD_STARTED, move |r: ShardStartedReq| {
            let n = n.clone();
            async move {
                n.ensure_master()?;
                let task = master::shards_started_task(n.master.clone(), vec![StartedShard { index: r.index, shard: r.shard, allocation_id: r.allocation_id }]);
                let memory=r.memory;
                let task:elasticsearch_cluster::coordination::StateTask=Box::new(move |state|{let _owner=memory;task(state)});
                n.coord().submit(task).await.map(|_| ()).map_err(|e| remote("illegal_state_exception", e, 500))
            }
        });
        let n = self.clone();
        t.register_handler(ACTION_SHARD_FAILED, move |r: ShardFailedReq| {
            let n = n.clone();
            async move {
                n.ensure_master()?;
                let failed = FailedShard { index: r.index, shard: r.shard, allocation_id: r.allocation_id, message: r.message, failure: None, mark_as_stale: r.mark_as_stale };
                let task = master::shard_failed_task(n.master.clone(), failed, r.reporter_term);
                let memory=r.memory;
                let task:elasticsearch_cluster::coordination::StateTask=Box::new(move |state|{let _owner=memory;task(state)});
                n.coord().submit(task).await.map(|_| ()).map_err(|e| remote("shard_state_action_exception", e, 500))
            }
        });
        let n = self.clone();
        t.register_handler(ACTION_FOLLOWER_METADATA, move |r: FollowerMetadataReq| {
            let n = n.clone();
            async move {
                if !n.is_master() {
                    return Err(RemoteError::new("not_master_exception", "node is not the elected master", 503));
                }
                commit_follower_change(&n, r).await
            }
        });
        let n = self.clone();
        t.register_handler(ACTION_MAPPING_UPDATE, move |r: MappingUpdateReq| {
            let n = n.clone();
            async move {
                crate::mapping_update::handle(&n, r).await
            }
        });
        let n = self.clone();
        t.register_handler(ACTION_MAPPING_UPDATE_V2, move |r: crate::mapping_update::UuidMappingUpdate| {
            let n = n.clone();
            async move {
                crate::mapping_update::handle(&n, r.into_request()?).await
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_REPLICATE, move |r: ReplicationRequest| {
            let n = n.clone();
            async move {
                crate::internal_capabilities::require_local_writer(&n.state(), &n.local)
                    .map_err(|e| remote("unsupported_operation_exception", e, 503))?;
                // Replica bytes count in indexing pressure, rejected (429)
                // only past 1.5x its limit, as ES does.
                let bytes: u64 = r.ops.iter().map(|op| match op {
                    crate::replication::ops::ReplicaOp::Index { source_json, .. } => source_json.len() as u64,
                    _ => 0,
                }).sum();
                let _pressure = elasticsearch_engine::indexing_pressure::global().replica(bytes)
                    .map_err(|e| remote(&e.kind, e.reason.clone(), 429))?;
                // Replica work has its own executor: it never queues behind
                // user searches or writes.
                elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Replica, move || shards::handle_replicate(&n.shards, r))
                    .await
                    .map_err(|e| remote("exception", e.reason, 500))?
                    .map_err(|(t, m)| remote(&t, m, 500))
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_REPLICA_REFRESH, move |r: shards::ReplicaRefreshReq| {
            let n = n.clone();
            async move {
                // Replica work has its own executor: it never queues behind
                // user searches or writes.
                elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Replica, move || shards::handle_replica_refresh(&n.shards, &r))
                    .await
                    .map_err(|e| remote("exception", e.reason, 500))?
                    .map_err(|(t, m)| remote(&t, m, 500))
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RESYNC, move |r: ResyncRequest| {
            let n = n.clone();
            async move {
                // Replica work has its own executor: it never queues behind
                // user searches or writes.
                elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Replica, move || shards::handle_resync(&n.shards, &r))
                    .await
                    .map_err(|e| remote("exception", e.reason, 500))?
                    .map_err(|(t, m)| remote(&t, m, 500))
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_START, move |r: RecoveryStartReq| {
            let n = n.clone();
            async move {
                let p = n.shards.primary(&r.index, r.shard).ok_or_else(|| remote("shard_not_found_exception", format!("[{}][{}] is not a primary on this node", r.index, r.shard), 404))?;
                let current = || n.shards.primary(&r.index, r.shard).is_some_and(|held| Arc::ptr_eq(&held, &p))
                    && recovery_start_matches(&n.state(), &n.local.id, &p, &r.target_allocation_id, None);
                let (recovery_id, info) = shards::start_recovery_source_checked(&p, &r.target_allocation_id, r.from_seq_no, current).await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                Ok(RecoveryStartResp { recovery_id, info })
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_REUSE, move |r: RecoveryReuseReq| {
            let n = n.clone();
            async move {
                let Some(p) = n.shards.primary(&r.index, r.shard) else { return Ok(None::<RecoveryStartResp>) };
                let current = || {
                    n.shards.primary(&r.index, r.shard).is_some_and(|held| Arc::ptr_eq(&held, &p))
                        && reuse_request_matches(&n.state(), &n.local.id, &p, &r)
                };
                let resumed = shards::resume_recovery_source(&p, &r.target_allocation_id, &r.prefix, current)
                    .await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                Ok(resumed.map(|(recovery_id, info)| RecoveryStartResp { recovery_id, info }))
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_REUSE_BATCH, move |r: BoundRecoveryReq<RecoveryBatchReq>| {
            let n = n.clone();
            async move {
                let q = r.request;
                let p = n.bound_recovery_primary(&q.index, q.shard, &r.source_allocation_id, r.primary_term)?;
                let target = Target::try_new(&q.target_allocation_id,&q.target_node_id,&q.target_address).map_err(|e|remote("circuit_breaking_exception",e.reason,e.status))?;
                let batch = shards::next_recovery_batch(&p, q.recovery_id, &q.target_allocation_id, target)
                    .await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                // Prefix reuse ships only operations: not throttled (ES
                // throttles file data, not an operations-based recovery).
                let held = n.bound_recovery_primary(&q.index, q.shard, &r.source_allocation_id, r.primary_term)?;
                if !Arc::ptr_eq(&held, &p) { return Err(remote("recovery_failed_exception", "recovery primary was replaced", 409)); }
                Ok(batch)
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_REUSE_FINALIZE, move |r: BoundRecoveryReq<RecoveryFinalizeReq>| {
            let n = n.clone();
            async move {
                let q = r.request;
                if q.relocation { return Err(remote("recovery_failed_exception", "replica-prefix reuse does not support relocation", 400)); }
                let p = n.bound_recovery_primary(&q.index, q.shard, &r.source_allocation_id, r.primary_term)?;
                // Fence writes while establishing the final checkpoint, and
                // revalidate authority after waiting for the permit.
                let _gate = p.gate.write().await;
                let held = n.bound_recovery_primary(&q.index, q.shard, &r.source_allocation_id, r.primary_term)?;
                if !Arc::ptr_eq(&held, &p) { return Err(remote("recovery_failed_exception", "recovery primary was replaced", 409)); }
                shards::current_recovery(&p, q.recovery_id, &q.target_allocation_id)
                    .map_err(|e| remote("recovery_failed_exception", e, 500))?;
                if p.replicate_all(&TransportNet(n.clone())).await.is_err() { return Ok(false); }
                let held = n.bound_recovery_primary(&q.index, q.shard, &r.source_allocation_id, r.primary_term)?;
                if !Arc::ptr_eq(&held, &p) { return Err(remote("recovery_failed_exception", "recovery primary was replaced", 409)); }
                let cp = p.tracker.lock().local_checkpoint(&q.target_allocation_id).unwrap_or(-1).max(q.local_checkpoint);
                shards::finalize_recovery_source(&p, q.recovery_id, &q.target_allocation_id, cp)
                    .map_err(|e| remote("recovery_failed_exception", e, 500))
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_BATCH, move |r: RecoveryBatchReq| {
            let n = n.clone();
            async move {
                let p = n.shards.primary(&r.index, r.shard).ok_or_else(|| remote("shard_not_found_exception", "primary moved", 404))?;
                let target = Target::try_new(&r.target_allocation_id,&r.target_node_id,&r.target_address).map_err(|e|remote("circuit_breaking_exception",e.reason,e.status))?;
                let batch = shards::next_recovery_batch(&p, r.recovery_id, &r.target_allocation_id, target).await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                n.throttle_recovery_ops(&batch).await;
                Ok(batch)
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_RECOVERY_FINALIZE, move |r: RecoveryFinalizeReq| {
            let n = n.clone();
            async move {
                let p = n.shards.primary(&r.index, r.shard).ok_or_else(|| remote("shard_not_found_exception", "primary moved", 404))?;
                if r.relocation {
                    shards::current_recovery(&p, r.recovery_id, &r.target_allocation_id).map_err(|e| remote("recovery_failed_exception", e, 500))?;
                    p.tracker.lock().update_local_checkpoint(&r.target_allocation_id, r.local_checkpoint);
                    return shards::handoff_primary(&p, r.recovery_id, &r.target_allocation_id, &TransportNet(n.clone())).await.map_err(|e| remote("recovery_failed_exception", e, 500));
                }
                let fin = |cp: i64| shards::finalize_recovery_source(&p, r.recovery_id, &r.target_allocation_id, cp).map_err(|e| remote("recovery_failed_exception", e, 500));
                if fin(r.local_checkpoint)? {
                    return Ok(true);
                }
                // Behind the primary's newest operations: block writes
                // briefly and ship everything (as the relocation handoff
                // does), so a copy under continuous writes still catches up.
                let _g = p.gate.write().await;
                if p.replicate_all(&TransportNet(n.clone())).await.is_err() {
                    return Ok(false);
                }
                let cp = p.tracker.lock().local_checkpoint(&r.target_allocation_id).unwrap_or(-1).max(r.local_checkpoint);
                fin(cp)
            }
        });
        let n = self.clone();
        t.register_handler(crate::replication::file_recovery::ACTION_FILE_RECOVERY_START, move |r: crate::replication::file_recovery::FileRecoveryStartReq| {
            let n = n.clone();
            async move {
                use crate::replication::file_recovery as fr;
                let p = n.shards.primary(&r.index, r.shard).ok_or_else(|| remote("shard_not_found_exception", format!("[{}][{}] is not a primary on this node", r.index, r.shard), 404))?;
                // ES `DelayRecoveryException`: the target must be an
                // initializing copy of this shard on the requesting node in
                // this node's state (retried by the target until it is).
                let state = n.state();
                let known = state.routing_table.shard_table(&r.index, r.shard).is_some_and(|t| t.copies.iter().any(|c| c.state == ShardState::Initializing
                    && c.allocation_id() == Some(r.target_allocation_id.as_str()) && c.node_id.as_deref() == Some(r.target_node_id.as_str())));
                if !known {
                    return Err(remote("shard_not_found_exception", format!("source node does not have the target [{}] of [{}][{}] in its routing table yet", r.target_allocation_id, r.index, r.shard), 404));
                }
                let target = state.nodes.get(&r.target_node_id).ok_or_else(|| remote("recovery_failed_exception", "recovery target is not a cluster member", 409))?;
                crate::capability_probe::verify(&n.transport, target, true).await
                    .map_err(|e| remote("recovery_failed_exception", e, 409))?;
                let (target_read_max, _) = elasticsearch_cluster::compatibility::segment_formats(target)
                    .map_err(|e| remote("recovery_failed_exception", e, 409))?;
                // Refused before any flush or graph save, tagged so the
                // master neither counts it toward max_retry nor picks such a
                // node for this copy again (`segment_format_refusal`).
                {
                    use elasticsearch_cluster::compatibility::{index_required_reader, require_segment_reader, segment_format_refusal, BASELINE_SEGMENT_FORMAT};
                    let write = elasticsearch_engine::rustic_core::codec::write_version();
                    let files = state.metadata.indices.get(&r.index).map_or(BASELINE_SEGMENT_FORMAT, index_required_reader);
                    for required in [write, files] {
                        require_segment_reader(target, required)
                            .map_err(|e| remote("recovery_failed_exception", segment_format_refusal(required, &e), 409))?;
                    }
                }
                let lag = n.effective_setting(fr::GRAPH_LAG_OPS).and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(50_000);
                let current = || n.shards.primary(&r.index, r.shard).is_some_and(|held| Arc::ptr_eq(&held, &p))
                    && recovery_start_matches(&n.state(), &n.local.id, &p, &r.target_allocation_id, Some(&r.target_node_id));
                let (recovery_id, info, manifest) = shards::start_file_recovery_source_checked(&p, &r.target_allocation_id, lag, target_read_max, current).await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                let history_uuid = crate::store_records::read(p.svc.path()).map_err(|e|remote(&e.kind,e.reason,e.status))?.filter(|c| c.primary && c.allocation_id == p.allocation_id).and_then(|c| c.history_uuid);
                tracing::info!(index = %r.index, shard = r.shard, target = %r.target_allocation_id, files = manifest.files.len(), bytes = manifest.total_bytes(), from = info.start_seq_no, "serving a file-based recovery");
                fr::FileRecoveryStartResp::try_new(recovery_id,info,manifest,history_uuid.as_deref(),&p.allocation_id,p.term)
                    .map_err(|e|remote("circuit_breaking_exception",e.reason,429))
            }
        });
        let n = self.clone();
        t.register_handler(crate::replication::file_recovery::ACTION_FILE_CHUNK, move |r: crate::replication::file_recovery::FileChunkReq| {
            let n = n.clone();
            async move {
                use crate::replication::file_recovery as fr;
                // Fenced: this node still runs the primary the recovery
                // started from, under the same term, and the recovery is
                // the target's current one.
                let p = n.bound_recovery_primary(&r.index, r.shard, &r.source_allocation_id, r.primary_term)?;
                let (files, throttle) = shards::recovery_files(&p, r.recovery_id, &r.target_allocation_id).map_err(|e| remote("recovery_failed_exception", e, 409))?;
                let chunk_memory = fr::ChunkBytes::reserve(r.len.min(crate::replication::throttle::MAX_CHUNK).saturating_add(r.path.len() as u64))
                    .map_err(|e| remote("circuit_breaking_exception", e.reason, 429))?;
                let data = shards::file_chunk(files, r.path.clone(), r.offset, r.len, chunk_memory).await.map_err(|e| remote("recovery_failed_exception", e, 500))?;
                // ES `RemoteRecoveryTargetHandler.writeFileChunk`: phase-1
                // file data is paced on the source's node-wide limiter.
                let pause = crate::replication::throttle::node_limiter(&n.local.id).delay_capped(data.len() as u64, n.recovery_settings().max_bytes_per_sec, crate::replication::throttle::MAX_PAUSE);
                if !pause.is_zero() {
                    tokio::time::sleep(pause).await;
                    throttle.fetch_add(pause.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                let crc32 = elasticsearch_engine::rustic_core::codec::crc32(&data);
                Ok(fr::FileChunkResp { offset: r.offset, data, crc32, source_throttle_nanos: throttle.load(std::sync::atomic::Ordering::Relaxed) })
            }
        });
        let n = self.clone();
        t.register_handler(shards::ACTION_SHARD_STORES, move |_: ()| {
            let n = n.clone();
            async move { n.local_copy_records().map_err(|e|RemoteError::new(e.kind,e.reason,e.status)) }
        });
        let n = self.clone();
        t.register_handler(ACTION_PRIMARY_PROBE, move |r: (crate::replication::recovery::RecoveryText, u32)| {
            let n = n.clone();
            async move { Ok::<_, RemoteError>(n.shards.primary(&r.0, r.1).is_some()) }
        });
        let n = self.clone();
        t.register_handler(ACTION_APPLIED_VERSION, move |_: ()| {
            let n = n.clone();
            async move { Ok::<_, RemoteError>(n.applied_version()) }
        });
        crate::front::register_forward_handler(self);
        crate::front::register_past_tsdb_handler(self);
    }

    /// The effective value of a node-level dynamic setting: cluster
    /// transient, then persistent, then the node's own.
    pub(crate) fn effective_setting(&self, key: &str) -> Option<String> {
        let state = self.state();
        let v = state.metadata.setting(key).cloned().or_else(|| self.config.settings.get(key).cloned())?;
        match v {
            serde_json::Value::String(s) => Some(s),
            serde_json::Value::Null => None,
            other => Some(other.to_string()),
        }
    }

    /// `indices.recovery.*` as of now (dynamic: read per served batch, so a
    /// change applies to recoveries in flight).
    pub fn recovery_settings(&self) -> crate::replication::throttle::RecoverySettings {
        crate::replication::throttle::RecoverySettings::resolve(|k| self.effective_setting(k))
    }

    /// `indices.recovery.max_bytes_per_sec` paces a recovery's data on the
    /// source, on the node-wide limiter: ES throttles the file chunks of
    /// phase 1 (`RemoteRecoveryTargetHandler.writeFileChunk`), the bytes
    /// put on the wire, and neither the translog phase nor an
    /// operations-based recovery. Here the document snapshot plays phase
    /// 1's part (relocations included): its batches' bytes are waited for
    /// before they are sent; translog batches are not paced. One pause is
    /// capped well below the batch request's timeout, so a batch at a low
    /// rate never fails its recovery, and a stalled recovery holds the
    /// limiter for at most that long.
    async fn throttle_recovery_ops(&self, batch: &RecoveryBatch) {
        if !batch.snapshot {
            return;
        }
        let bytes: u64 = batch.ops.iter().map(|o| o.estimated_bytes() as u64).sum();
        let pause = crate::replication::throttle::node_limiter(&self.local.id).delay_capped(bytes, self.recovery_settings().max_bytes_per_sec, crate::replication::throttle::MAX_PAUSE);
        if !pause.is_zero() {
            tokio::time::sleep(pause).await;
        }
    }

    fn ensure_master(&self) -> Result<(), RemoteError> {
        if self.is_master() { Ok(()) } else { Err(remote("not_master_exception", format!("node [{}] is not the elected master", self.local.name), 503)) }
    }

    /// The copies this node holds data for (for full-restart allocation).
    pub fn local_copy_records(&self) -> elasticsearch_engine::error::Result<crate::store_records::StoreRecords> { crate::store_records::collect(self) }

    /// ES's acknowledgement: wait until every node in the state has applied
    /// a cluster state of at least `version` (the ack timeout). Returns
    /// false on timeout, which ES reports as `"acknowledged": false`.
    /// The newest state this node applied completely (see `applied_version`).
    pub fn applied_version(&self) -> u64 {
        self.applied_version.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Whether a custom is deferred for an engine API in flight.
    fn customs_deferred(&self) -> bool {
        self.apply_retry.lock().iter().any(|e| e.starts_with(CUSTOMS_DEFERRED))
    }

    /// After an application of `version`: record it as settled unless a
    /// custom is (still) deferred.
    fn note_customs_settled(&self, version: u64) {
        if !self.customs_deferred() {
            self.customs_settled.fetch_max(version, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether this node applied `version` completely, its customs
    /// included. A custom deferred for an engine API in flight is applied
    /// once that API is done; an acknowledgement waits for it (ES applies
    /// each committed state whole before acking, so a request that follows
    /// an acked change sees it). Inside an engine API's own task the
    /// deferral is not waited for: it is that API's guard that holds it.
    fn applied_here(&self, version: u64) -> bool {
        if self.applied_version() < version {
            return false;
        }
        if self.customs_settled.load(std::sync::atomic::Ordering::SeqCst) >= version || IN_ENGINE_API.try_with(|_| ()).is_ok() || !self.customs_deferred() {
            return true;
        }
        // Applied as soon as nothing holds it, not on the retry timer.
        if !self.engine_api_busy() {
            if let Some(me) = self.me.get().and_then(Weak::upgrade) {
                if let Some(lease) = VectorOptimizerLease::claim(&me.background.apply_retry) {
                    tokio::task::spawn_blocking(move || {
                        let _lease = lease;
                        me.retry_failed_application();
                    });
                }
            }
        }
        false
    }

    /// Whether any engine API is in flight on engine 0.
    fn engine_api_busy(&self) -> bool {
        self.engine_api_inflight.load(std::sync::atomic::Ordering::SeqCst) > 0 || !self.engine_api_families.lock().is_empty()
    }

    /// An engine API is done (its guard dropped after its kept job): apply
    /// what was deferred for it now, before its response, rather than on
    /// the retry timer.
    pub async fn settle_deferred_customs(self: &Arc<Self>) {
        if !self.customs_deferred() || self.engine_api_busy() {
            return;
        }
        let me = self.clone();
        let _ = tokio::task::spawn_blocking(move || me.retry_failed_application()).await;
    }

    /// Wait (bounded) until this node applied `version` completely.
    pub async fn wait_applied_here(&self, version: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.applied_here(version) {
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        true
    }

    pub async fn wait_applied_everywhere(&self, version: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            let st = self.state();
            let mut all = self.applied_here(version);
            for n in st.nodes.values().filter(|n| n.id != self.local.id) {
                if !all {
                    break;
                }
                let Ok(addr) = n.transport_address.parse::<SocketAddr>() else { continue };
                match self.transport.send_request::<_, u64>(addr, ACTION_APPLIED_VERSION, &(), Duration::from_secs(5)).await {
                    Ok(v) if v >= version => {}
                    _ => all = false,
                }
            }
            if all {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Apply a synchronous engine metadata mutation on the master's engine
    /// and publish the result, through the metadata service: the mutation
    /// and the extraction of its metadata run in one batch, off the
    /// coordination actor, and are rolled back if the publication fails.
    pub async fn mutate_engine_metadata<T, F>(&self, mutation: F) -> Result<(T, u64), elasticsearch_engine::error::EsError>
    where T: Send + 'static,
          F: FnOnce(&Engine) -> Result<T, elasticsearch_engine::error::EsError> + Send + 'static,
    {
        use elasticsearch_engine::error::EsError;
        let slot = Arc::new(Mutex::new(None));
        let captured = slot.clone();
        let run: Box<dyn FnOnce(&ClusterNode, crate::master_service::On) -> bool + Send> = Box::new(move |node: &ClusterNode, on: crate::master_service::On| {
            let engine = node.engine_on(on);
            // Recovery uses the live map's lock while replacing index shells.
            let _uuids = node.uuids_on(on).lock();
            let result = mutation(&engine);
            let ok = result.is_ok();
            *captured.lock() = Some(result);
            ok
        });
        // A dry run: undone at once, applied here once committed.
        let publication = self.metadata_service().run(run, true).await;
        let value = slot.lock().take();
        match value {
            Some(Err(error)) => Err(error),
            Some(Ok(value)) => match publication {
                Ok(version) => Ok((value, version)),
                Err(error) => Err(EsError::new(503, "failed_to_commit_cluster_state_exception", error)),
            },
            None => Err(EsError::new(503, "failed_to_commit_cluster_state_exception", publication.err().unwrap_or_else(|| "metadata task did not execute".into()))),
        }
    }

    /// Master only: publish the metadata the engine holds (after an engine
    /// API that ran outside the metadata service changed it).
    /// Publish what an engine API changed in engine 0 since `before`
    /// ([`Self::engine_snapshot`] taken before the API ran).
    pub async fn publish_engine_metadata_since(&self, before: crate::scratch::Snapshot) -> Result<u64, String> {
        self.metadata_service().publish_kept_since(before).await
    }

    /// See `rollback_wanted`.
    pub(crate) fn request_engine_rollback(&self) {
        self.rollback_wanted.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Enter an engine API (see [`Self::engine_api_in_flight`]), first
    /// waiting out a wanted or running rollback: an API never snapshots, or
    /// changes, an engine 0 a rollback is about to reset (a client retrying
    /// a request that failed unapplied sees the rolled-back engine).
    pub async fn engine_api_enter(self: &Arc<Self>, family: Option<&'static str>) -> EngineApiGuard {
        loop {
            {
                let _gate = self.api_gate.lock();
                if !self.rollback_wanted.load(std::sync::atomic::Ordering::SeqCst) {
                    return self.engine_api_in_flight(family);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Run a wanted rollback if no engine API is in flight (blocking).
    /// New APIs wait while it runs (`rollback_wanted` stays set until done).
    fn run_due_rollback(&self) {
        {
            let _gate = self.api_gate.lock();
            if !self.rollback_due() {
                return;
            }
        }
        self.resync_engine_metadata();
        self.rollback_wanted.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether a requested rollback can run now (no engine API in flight).
    fn rollback_due(&self) -> bool {
        self.rollback_wanted.load(std::sync::atomic::Ordering::SeqCst)
            && self.engine_api_inflight.load(std::sync::atomic::Ordering::SeqCst) == 0
            && self.engine_api_families.lock().is_empty()
    }

    /// Mark an engine API as in flight until the guard drops (after its
    /// kept job): appliers leave engine 0's customs alone meanwhile (only
    /// `family`, when the API changes that one family).
    pub fn engine_api_in_flight(self: &Arc<Self>, family: Option<&'static str>) -> EngineApiGuard {
        match family {
            None => {
                self.engine_api_inflight.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            Some(f) => {
                *self.engine_api_families.lock().entry(f).or_insert(0) += 1;
            }
        }
        let id = self.engine_api_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        EngineApiGuard(self.clone(), family, id)
    }

    /// Engine 0 as it is now, between applications (the base of
    /// [`Self::publish_engine_metadata_since`]).
    pub async fn engine_snapshot(self: &Arc<Self>, family: Option<&'static str>) -> Option<crate::scratch::Snapshot> {
        let me = self.clone();
        tokio::task::spawn_blocking(move || {
            let _metadata = me.metadata_mutex.lock();
            crate::scratch::snapshot_scoped(&me.engine, Some(family))
        })
        .await
        .ok()
    }

    /// [`Self::engine_snapshot`] for the engine API of `guard`, which also
    /// holds the index incarnations of the snapshot until the guard drops
    /// (see `engine_api_held`). Taken under the metadata lock, as every
    /// application: no application falls between the snapshot and the hold.
    pub async fn engine_api_snapshot(self: &Arc<Self>, guard: &EngineApiGuard) -> Option<crate::scratch::Snapshot> {
        let me = self.clone();
        let (family, id) = (guard.1, guard.2);
        tokio::task::spawn_blocking(move || {
            let _metadata = me.metadata_mutex.lock();
            let snapshot = crate::scratch::snapshot_scoped(&me.engine, Some(family));
            let held: Vec<(String, String)> = {
                let uuids = me.uuids.lock();
                snapshot.index_names().filter_map(|name| uuids.cluster_uuid(name).map(|u| (name.clone(), u.to_string()))).collect()
            };
            me.engine_api_held.lock().insert(id, held);
            snapshot
        })
        .await
        .ok()
    }

    /// The index incarnations engine APIs in flight hold (see
    /// `engine_api_held`).
    fn held_deletes(&self) -> mirror::HeldDeletes {
        self.engine_api_held.lock().values().flatten().cloned().collect()
    }

    pub async fn publish_engine_metadata(&self) -> Result<u64, String> {
        // Kept: the engine API already changed the engine.
        self.metadata_service().run(Box::new(|node: &ClusterNode, _: crate::master_service::On| {
            elasticsearch_engine::registry::ensure(&node.engine);
            true
        }), false).await
    }

    /// The applied state engine 0 last mirrored (the base kept changes are
    /// diffed against); `None` before the first application.
    pub(crate) fn mirrored_state(&self) -> Option<Arc<ClusterState>> {
        self.mirrored.lock().clone()
    }

    /// Wait (bounded) until this node leads and its applier has applied a
    /// state of its own term; whether it did.
    pub async fn await_own_master_state(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut rx = self.subscribe();
        loop {
            if self.is_master() {
                return true;
            }
            if self.mode.lock().0 != Mode::Leader {
                return false;
            }
            if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
                return self.is_master();
            }
        }
    }

    /// The master service's scratch engine, opened on first use.
    pub(crate) fn scratch(&self) -> Result<&crate::scratch::ScratchEngine, String> {
        self.scratch.get_or_init(|| {
            crate::scratch::ScratchEngine::open(&self.config.data_path, &self.engine).map(|scratch| {
                // Rollovers (REST and lifecycle) on the copy judge the
                // cluster-wide stats, as on engine 0.
                if let Some(hook) = self.rollover_stats.get() {
                    scratch.engine.set_rollover_stats_hook(Some(hook.clone()));
                }
                if let Some(steps) = self.lifecycle_steps.get() {
                    scratch.engine.ilm.set_cluster_steps(Some(steps.clone()));
                }
                scratch
            })
        }).as_ref().map_err(Clone::clone)
    }

    pub(crate) fn set_engine_metadata_pending(&self, pending: bool) {
        self.engine_metadata_pending.store(pending, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether a rollback of the master's engine was requested (and clear it).
    pub(crate) fn take_metadata_resync(&self) -> bool {
        self.metadata_resync.swap(false, std::sync::atomic::Ordering::SeqCst)
    }

    /// Make the master's engines follow the applied state again (the
    /// metadata service, after each batch): rolls back the changes of a
    /// batch whose publication failed, and catches up state metadata the
    /// applier skipped while the batch was pending. Shards are reconciled
    /// against the result (an index a pure task added, e.g. a restore, can
    /// only be opened once its shell exists).
    /// After a committed kept batch: engine 0 applies, incrementally, what
    /// the applier left to it while the batch was pending.
    pub(crate) fn catch_up_engine_metadata(&self) {
        let _metadata = self.metadata_mutex.lock();
        let _g = self.apply_lock.lock();
        self.set_engine_metadata_pending(false);
        let state = self.state();
        if state.version == 0 || state.master_node_id.is_none() {
            return;
        }
        self.apply_state_metadata(&state, false);
    }

    pub(crate) fn resync_engine_metadata(&self) {
        let _metadata = self.metadata_mutex.lock();
        let _g = self.apply_lock.lock();
        self.set_engine_metadata_pending(false);
        let state = self.state();
        if state.version == 0 {
            return;
        }
        self.apply_state_metadata(&state, true);
        if state.master_node_id.is_some() {
            self.reconcile_shards(&state);
        }
    }

    /// Mirror the state's metadata into this node's engines. `force` applies
    /// even an already applied metadata version and a no-master state (a
    /// rollback). While a metadata batch is pending, engine 0 is left to
    /// the metadata service.
    fn apply_state_metadata(&self, state: &Arc<ClusterState>, force: bool) {
        let mv = (state.metadata.cluster_uuid.clone(), state.metadata.version);
        if !force && self.last_metadata_version.lock().as_ref() == Some(&mv) && self.apply_retry.lock().is_empty() && !self.shard_apply_failed.load(std::sync::atomic::Ordering::SeqCst) {
            self.note_customs_settled(state.version);
            return;
        }
        let engine0 = force || !self.engine_metadata_pending.load(std::sync::atomic::Ordering::SeqCst);
        let local_master = state.master_node_id.as_deref() == Some(self.local.id.as_str());
        let mut engine0_mirrored: Option<Arc<ClusterState>> = None;
        if engine0 {
            // Endpoints first: semantic_text copies opened or recovered
            // for the applied indices resolve them.
            if !local_master {
                if let Some(e) = crate::meta_extra::apply_inference(&self.engine, &state.metadata) {
                    tracing::error!("failed to apply inference endpoints: {e}");
                }
                if let Some(e) = crate::meta_extra::apply_stored_scripts(&self.engine, &state.metadata) {
                    tracing::error!("failed to apply stored scripts: {e}");
                }
                if let Some(e) = crate::meta_extra::apply_esql_views(&self.engine, &state.metadata) {
                    tracing::error!("failed to apply ES|QL views: {e}");
                }
            }
            // Incremental from the state last mirrored here (unchanged index
            // metadata is shared, so this is O(changed)); a full pass when
            // forced (re-sync, rollback) or with nothing to diff against.
            let prev = self.mirrored.lock().clone().filter(|p| !force && p.metadata.cluster_uuid == state.metadata.cluster_uuid);
            let retry = std::mem::take(&mut *self.apply_retry.lock());
            // An engine API in flight on this node's engine 0 (its change is
            // not published yet): its customs are not overwritten now; they
            // are applied once it has published (the retry below).
            // Only customs the engine consumes count (data streams are
            // applied with the globals, other customs have no engine form),
            // and only those engine 0 does not already hold as the state has
            // them: an API's own publication carries exactly what it wrote
            // into engine 0, so there is nothing to hold back for it.
            let engine_customs_changed: Vec<&'static str> = crate::meta_extra::ENGINE_APPLIED_CUSTOMS.iter().copied()
                .filter(|k| prev.as_ref().is_none_or(|p| p.metadata.customs.get(*k) != state.metadata.customs.get(*k)))
                .collect();
            let api_inflight = self.engine_api_inflight.load(std::sync::atomic::Ordering::SeqCst) > 0;
            let engine_holds_all = api_inflight && engine_customs_changed.iter().all(|k| crate::meta_extra::engine_holds_custom(&self.engine, &state.metadata, k));
            let families: Vec<&'static str> = if api_inflight { Vec::new() } else { self.engine_api_families.lock().keys().copied().collect() };
            // Only those families held back: the state applied keeps their
            // previous form (recorded as deferred below).
            // A held family is applied as engine 0 holds it now (a no-op), never
            // as absent: an absent family means "empty" to a full apply.
            // A family engine 0 already holds as the state has it (the API's
            // own publication) is not deferred.
            let mut families_held: Vec<&'static str> = Vec::new();
            let held: Option<elasticsearch_cluster::state::Metadata> = (!families.is_empty()).then(|| {
                let mut md = state.metadata.clone();
                for f in &families {
                    let mut own = elasticsearch_cluster::state::Metadata::default();
                    crate::meta_extra::extract_only(&self.engine, &mut own, Some(*f));
                    let own = own.customs.remove(*f);
                    if crate::meta_extra::same_custom(f, own.as_ref(), state.metadata.customs.get(*f)) {
                        families_held.push(*f);
                    }
                    match own {
                        Some(v) => md.customs.insert(f.to_string(), v),
                        None => md.customs.remove(*f),
                    };
                }
                md
            });
            let applied_md = held.as_ref().unwrap_or(&state.metadata);
            // A retry of failed globals re-applies every custom wholesale:
            // not while an engine API is in flight (it would wipe its
            // unpublished change); kept for a later pass.
            let (retry, postponed): (std::collections::BTreeSet<String>, Vec<String>) = if api_inflight || !families.is_empty() {
                let (keep, later): (Vec<String>, Vec<String>) = retry.into_iter().partition(|n| state.metadata.indices.contains_key(n) || prev.as_ref().is_some_and(|p| p.metadata.indices.contains_key(n)));
                (keep.into_iter().collect(), later)
            } else {
                (retry, Vec::new())
            };
            // A deferral this pass resolves is not postponed again: `prev`
            // (the mirrored state) keeps a deferred custom in its previous
            // form, so the entry-by-entry application below applies it once
            // its API is done, and a custom engine 0 already holds as the
            // state has it needs nothing.
            let resolved = |n: &str| -> bool {
                if n == CUSTOMS_DEFERRED {
                    return !api_inflight || engine_holds_all;
                }
                match n.strip_prefix(CUSTOMS_DEFERRED).and_then(|r| r.strip_prefix(':')) {
                    Some(f) => !api_inflight && (families_held.contains(&f) || !families.contains(&f)),
                    None => false,
                }
            };
            self.apply_retry.lock().extend(postponed.into_iter().filter(|n| !resolved(n)));
            // An index an engine API in flight deleted from engine 0 is not
            // re-created: its kept job publishes the delete against the state
            // mirrored here (the index stays in `mirrored`, as applied).
            let held = self.held_deletes();
            let mut rep = match &prev {
                Some(p) => mirror::apply_changes_holding(&self.engine, &p.metadata, applied_md, &mut self.uuids.lock(), !api_inflight, &retry, &held),
                None => mirror::apply_scoped(&self.engine, applied_md, &mut self.uuids.lock(), !api_inflight),
            };
            if !rep.withheld.is_empty() {
                tracing::debug!(indices = ?rep.withheld, "left absent from engine 0: deleted by an engine API in flight");
                self.engine_api_withheld.lock().extend(rep.withheld.iter().cloned());
            }
            let retry_globals = retry.iter().any(|n| !state.metadata.indices.contains_key(n));
            let customs_differ = (!engine_customs_changed.is_empty() && !engine_holds_all) || retry_globals;
            if api_inflight && customs_differ {
                rep.errors.push((CUSTOMS_DEFERRED.into(), "an engine API is in flight; customs applied after it".into()));
            }
            for f in &families {
                if families_held.contains(f) {
                    continue;
                }
                if prev.as_ref().is_none_or(|p| p.metadata.customs.get(*f) != state.metadata.customs.get(*f)) {
                    rep.errors.push((format!("{CUSTOMS_DEFERRED}:{f}"), "an engine API of this family is in flight; applied after it".into()));
                }
            }
            for (index, reason) in &rep.failed_mappings { self.fail_mapping_copy(index, 0, reason); }
            for (i, e) in &rep.errors {
                // A deferral is expected (applied once its API is done), not
                // a failure.
                if i.starts_with(CUSTOMS_DEFERRED) {
                    tracing::debug!(index = %i, "metadata application deferred: {e}");
                } else {
                    tracing::error!(index = %i, "failed to apply metadata: {e}");
                }
            }
            // Applied again at the next application or by the retry timer:
            // the failed parts stay in their previous form in `mirrored`, so
            // the next incremental pass sees them as changed (a failed
            // delete included).
            self.apply_retry.lock().extend(rep.errors.iter().map(|(i, _)| i.clone()));
            engine0_mirrored = Some(mirrored_after(state, prev.as_deref(), &rep.errors));
            if !local_master {
                if let Some(e) = crate::meta_extra::apply_enrich(&self.engine, &state.metadata) {
                    tracing::error!("failed to apply enrich policies: {e}");
                }
            }
        }
        let prev_shards = self.shards_mirrored.lock().clone().filter(|p| !force && p.metadata.cluster_uuid == state.metadata.cluster_uuid);
        let mut shard_errors: Vec<(String, String)> = Vec::new();
        for (k, rep) in self.apply_shard_engines(prev_shards.as_ref().map(|p| &p.metadata), &state.metadata) {
            for (index, reason) in &rep.failed_mappings { self.fail_mapping_copy(index, k, reason); }
            for (i, e) in rep.errors {
                tracing::error!(index = %i, shard = k, "failed to apply metadata to shard engine: {e}");
                shard_errors.push((i, e));
            }
        }
        // Only what applied advances: a failed part is retried from its
        // previous form (the rest is not re-diffed from an ever older base).
        self.shard_apply_failed.store(!shard_errors.is_empty(), std::sync::atomic::Ordering::SeqCst);
        *self.shards_mirrored.lock() = Some(mirrored_after(state, prev_shards.as_deref(), &shard_errors));
        if let Some(m) = engine0_mirrored {
            *self.last_metadata_version.lock() = Some(mv);
            *self.mirrored.lock() = Some(m);
            self.note_customs_settled(state.version);
        }
    }

    /// Re-apply the current state where its last application failed (the
    /// retry timer: a failure is not left for the next state change).
    /// Delete the retained stale copies (see `retained`) no context reads
    /// any more, as `reconcile_shards` would have (ES deletes a shard's
    /// data once its shard lock is released), without waiting for the next
    /// cluster state change.
    fn release_retained_copies(&self) {
        let keys: Vec<(String, u32)> = self.retained.lock().iter().cloned().collect();
        if keys.is_empty() {
            return;
        }
        let _metadata = self.metadata_mutex.lock();
        let _g = self.apply_lock.lock();
        let state = self.state();
        let me = self.local.id.as_str();
        for (index, k) in keys {
            let Some(engine) = self.engine_for(k) else {
                self.retained.lock().remove(&(index, k));
                continue;
            };
            if engine.copy_has_contexts(&index, k) {
                continue;
            }
            self.retained.lock().remove(&(index.clone(), k));
            let assigned_here = state.routing_table.shard_table(&index, k).is_some_and(|t| t.copies.iter().any(|c| c.node_id.as_deref() == Some(me) && c.state != ShardState::Unassigned));
            if assigned_here || self.shards.get(&index, k).is_some() {
                continue;
            }
            let Some(svc) = engine.get_index(&index) else { continue };
            if shards::read_copy_record(svc.path()).is_none() {
                continue;
            }
            let fully_active = state.routing_table.shard_table(&index, k).map(|t| !t.copies.is_empty() && t.copies.iter().all(|c| c.state == ShardState::Started)).unwrap_or(false);
            if fully_active && !self.local_copy_in_sync(&state, &index, k) {
                tracing::info!(%index, shard = k, "deleting stale local shard data (its last search context was released)");
                self.wipe_local_copy(&index, k, &state);
            }
        }
    }

    fn retry_failed_application(&self) {
        let _metadata = self.metadata_mutex.lock();
        let _g = self.apply_lock.lock();
        let state = self.state();
        if state.version == 0 || state.master_node_id.is_none() {
            return;
        }
        self.apply_state_metadata(&state, false);
    }

    fn application_failed(&self) -> bool {
        !self.apply_retry.lock().is_empty() || self.shard_apply_failed.load(std::sync::atomic::Ordering::SeqCst)
    }

    // ------------------------------------------------------------- applier

    fn reconcile_shards(&self, state: &ClusterState) {
        self.capacity_leases.lock().reconcile(self,state,Instant::now());
        let me = &self.local.id;
        let mut wanted: BTreeMap<(String, u32), ShardRouting> = BTreeMap::new();
        for irt in state.routing_table.indices.values() {
            for st in &irt.shards {
                // A relocation target sits next to its source; prefer the
                // copy that is ours.
                if let Some(c) = st.copies.iter().find(|c| c.node_id.as_deref() == Some(me.as_str()) && c.state != ShardState::Unassigned) {
                    wanted.insert((c.index.clone(), c.shard), c.clone());
                }
            }
        }
        // Copies no longer assigned here.
        for local in self.shards.all() {
            let (index, shard) = {
                let g = local.lock();
                (g.index.clone(), g.shard)
            };
            let keep = wanted.get(&(index.clone(), shard)).map(|c| c.allocation_id() == Some(local.lock().allocation_id.as_str())).unwrap_or(false);
            if !keep {
                self.remove_copy(&index, shard);
                tracing::info!(%index, shard, "shard copy removed from this node");
            }
        }
        // Data of unassigned copies is deleted once the shard is fully
        // active elsewhere (ES `IndicesStore`); until then it may still be
        // the only copy an operator could force-allocate.
        for (k, engine) in self.all_shard_engines() {
            for svc in engine.all_indices() {
                let key = (svc.name.clone(), k);
                if wanted.contains_key(&key) || self.shards.get(&svc.name, k).is_some()
                    || self.pending.lock().contains(&primary_open_pending_key(&svc.name, k)) {
                    continue;
                }
                if shards::read_copy_record(svc.path()).is_none() {
                    // A file recovery's leftovers (staged files, the old
                    // store set aside) of a copy no longer assigned here.
                    use crate::replication::file_recovery as fr;
                    if fr::staging_dir(&svc, k).exists() || fr::reuse_dir(&svc, k).exists() || fr::install_dir(&svc, k).exists() {
                        crate::replication::file_recovery::remove_recovery_dirs(&svc, k);
                    }
                    continue;
                }
                let fully_active = state.routing_table.shard_table(&svc.name, k).map(|t| !t.copies.is_empty() && t.copies.iter().all(|c| c.state == ShardState::Started)).unwrap_or(false);
                // Never on the routing table's word alone while this copy is
                // still in-sync: after a full restart the table says "started"
                // for copies no node has opened yet.
                if fully_active && !self.local_copy_in_sync(state, &svc.name, k) {
                    // A context still pinning the copy (a scroll over a shard
                    // that relocated away) keeps its data until it is cleared
                    // or expires; a later state change deletes it.
                    if engine.copy_has_contexts(&svc.name, k) {
                        tracing::debug!(index = %svc.name, shard = k, "stale local shard data kept: search contexts still read it");
                        self.retained.lock().insert((svc.name.clone(), k));
                        continue;
                    }
                    tracing::info!(index = %svc.name, shard = k, "deleting stale local shard data (shard is fully active elsewhere)");
                    self.wipe_local_copy(&svc.name, k, state);
                }
            }
        }
        for ((index, shard), routing) in wanted {
            self.reconcile_one(state, &index, shard, &routing);
        }
    }

    fn fail_mapping_copy(&self, index: &str, shard: u32, reason: &str) {
        if let Some(slot) = self.shards.get(index, shard) {
            let aid = slot.lock().allocation_id.clone();
            self.report_failed(index, shard, &aid, None, &format!("persistent committed mapping conflict: {reason}"), true);
        } else if let Some(engine) = self.engine_for(shard) {
            // A metadata-only shell has no data copy to fail or recover.
            // Recreate only empty shells; populated copies follow allocation.
            if engine.get_index(index).is_some_and(|svc| svc.active_shards() == 0) {
                let state = self.state();
                self.with_uuids(shard, |uuids| self.wipe_copy_in(&engine, uuids, index, &state));
            }
        }
    }

    /// Delete the engine index (data) and recreate the empty metadata shell.
    ///
    /// The uuid map is locked for the whole delete + create: the master's
    /// `mirror::extract` (run under the same lock) must never see the engine
    /// without the index, or it would publish the index as deleted and every
    /// node would drop its data (FAULT_RESULTS.md bug #3). The map is saved
    /// before returning so a crash right after does not leave a stale local
    /// uuid behind (which `mirror::apply` would answer by deleting the copy).
    fn wipe_local_copy(&self, index: &str, shard: u32, state: &ClusterState) -> Option<Arc<IndexService>> {
        self.wipe_local_copy_mode(index, shard, state, false)
    }

    /// [`Self::wipe_local_copy`]; `keep_files`: a file-based recovery
    /// follows, so the same incarnation's store is set aside for reuse
    /// (`shard-<n>.reuse/`) instead of deleted.
    fn wipe_local_copy_mode(&self, index: &str, shard: u32, state: &ClusterState, keep_files: bool) -> Option<Arc<IndexService>> {
        let engine = self.engine_for(shard)?;
        // The same incarnation: reset only this shard's data, in place. The
        // index service (on the master: the metadata source of truth, with
        // every shard and any change not yet published) stays as it is.
        if let Some(svc) = self.same_incarnation_index(index, shard, state).filter(|svc| !svc.is_detached()) {
            // Same index incarnation: retain scroll/PIT reader generations.
            // The old writer closes before its pathname is reused; retained
            // FileSlices survive unlink and cannot open replacement files.
            let reset = if keep_files { shards::set_aside_copy_for_recovery(&svc, shard) } else { shards::reset_copy_for_recovery(&svc, shard) };
            match reset {
                Ok(()) => return Some(svc),
                Err(e) => {
                    tracing::warn!(%index, shard, "failed to reset the local shard copy in place: {e}");
                    return None;
                }
            }
        }
        // The master's engine 0 is the metadata source of truth: an index it
        // no longer has was deleted after `state` was published. Re-creating
        // it from that state would resurrect it at the next publish.
        if !may_recreate_from_state(self.is_master(), shard, engine.get_index(index).is_some()) {
            return None;
        }
        self.with_uuids(shard, |uuids| self.wipe_copy_in(&engine, uuids, index, state)).flatten()
    }

    /// The local index service of `index` when it is the incarnation
    /// `state` describes (its copy of `shard` can be reset in place).
    fn same_incarnation_index(&self, index: &str, shard: u32, state: &ClusterState) -> Option<Arc<IndexService>> {
        let (svc, im) = (self.engine_for(shard)?.get_index(index)?, state.metadata.indices.get(index)?);
        let mapped = self.with_uuids(shard, |uuids| uuids.indices.get(index).cloned()).flatten();
        same_incarnation(mapped.as_ref(), &im.uuid, &svc.uuid).then_some(svc)
    }

    fn wipe_copy_in(&self, engine: &Engine, uuids: &mut UuidMap, index: &str, state: &ClusterState) -> Option<Arc<IndexService>> {
        let im = state.metadata.indices.get(index)?;
        let mut last = String::new();
        for _ in 0..5 {
            let _ = crate::meta_extra::delete_local_copy(engine, index);
            // Sparse: the caller opens the one copy it holds (`ensure_copy_open`).
            match mirror::create_sparse(engine, index, im) {
                Ok(s) => {
                    uuids.indices.insert(index.to_string(), (im.uuid.clone(), s.uuid.clone()));
                    s.set_cluster_uuid(&im.uuid);
                    if let Err(e) = uuids.save() {
                        tracing::warn!(%index, "failed to persist the index uuid map: {e}");
                    }
                    return Some(s);
                }
                // Someone (e.g. an auto-create racing on this engine) made
                // the index between our delete and create: go again.
                Err(e) => last = e.reason,
            }
        }
        tracing::error!(%index, "failed to recreate index shell: {last}");
        // Something else (an auto-create racing the wipe) re-created the
        // index: adopt it, so the uuid map never goes stale. A stale map
        // makes `mirror::extract` on the master mistake the index for a new
        // one and re-create it cluster-wide, dropping every copy.
        let existing = engine.get_index(index)?;
        uuids.indices.insert(index.to_string(), (im.uuid.clone(), existing.uuid.clone()));
        existing.set_cluster_uuid(&im.uuid);
        if let Err(e) = uuids.save() {
            tracing::warn!(%index, "failed to persist the index uuid map: {e}");
        }
        None
    }

    /// Drop copy `shard` of `index` from the shard manager and close its
    /// engine copy (sparse data plane: the node only keeps open the copies
    /// it holds). Closing fences new operations with
    /// `index_shard_closed_exception`, drains in-flight ones and makes
    /// acknowledged operations durable; a primary context still in flight
    /// then fails its write (never acknowledges it) instead of writing to a
    /// copy that no longer belongs here. The data stays on disk until
    /// `wipe_local_copy` (or a reopen with `ExistingStore`).
    fn remove_copy(&self, index: &str, shard: u32) {
        let Some(local) = self.shards.remove(index, shard) else { return };
        let (svc, copy) = {
            let g = local.lock();
            let copy = match &g.role {
                Role::Primary(p) => p.copy(),
                Role::Replica(r) => r.copy(),
            };
            (g.svc.clone(), copy)
        };
        if !svc.is_sparse() {
            // A full-open index (the master's own engine-0 copies created by
            // the metadata REST layer) keeps its copies: other code paths
            // (resize) read them from there.
            return;
        }
        // Fence and detach here (new operations fail at once), drain and
        // close in the background: an in-flight write may hold a permit
        // while it waits for a mapping commit only this applier would apply,
        // and the applier must not wait for it.
        // The copy's scroll / PIT / qtf contexts stay (ES 9.5.5 frees
        // reader contexts only when the index is deleted, closed or
        // reopened, `SearchService.afterIndexRemoved`): a scroll over a copy
        // that relocated away keeps reading its pinned views until it is
        // cleared or expires, and the copy's data is deleted only after that
        // (`reconcile_shards`). A same-incarnation recovery also preserves
        // them: the retired directory rejects new opens, and immutable
        // reader mappings stay valid even when that store is replaced.
        let Some(detached) = svc.detach_shard_if_same(&copy) else { return };
        let (index, runtime) = (index.to_string(), tokio::runtime::Handle::try_current());
        let close = move || {
            if let Err(e) = detached.close() {
                if e.kind != "shard_not_found_exception" {
                    tracing::warn!(%index, shard, "failed to close the engine shard copy: {}", e.reason);
                }
            }
        };
        match runtime {
            Ok(rt) => {
                rt.spawn_blocking(close);
            }
            Err(_) => close(),
        }
    }

    /// Make sure the engine copy of `shard` is open on `svc` (sparse data
    /// plane: indices are created with no copies open). Opens from the
    /// store when one is on disk, else as a new empty copy. Always opened as
    /// a primary-capable copy at term 1: the cluster layer owns roles and
    /// primary terms (replica WAL / `PrimaryCtx`), exactly as with the full
    /// opens `create_index` used to do.
    fn ensure_copy_open(svc: &IndexService, shard: u32, aid: &str) -> Result<(), String> {
        if svc.shard(shard).is_some() {
            return Ok(());
        }
        svc.reload_search_analyzers().map_err(|error| format!("failed analyzer recovery: [{}] {}", error.kind, error.reason))?;
        let recovery = if shards::shard_path(svc, shard).join("shard_state.json").exists() { EngineRecovery::ExistingStore } else { EngineRecovery::Empty };
        match svc.open_shard(shard, ShardOpenSpec { allocation_id: aid.to_string(), primary: true, primary_term: 1, recovery }) {
            Ok(_) => Ok(()),
            Err(_) if svc.shard(shard).is_some() => Ok(()),
            Err(e) => Err(format!("failed to open shard copy: [{}] {}", e.kind, e.reason)),
        }
    }

    /// The engine's resize (executed on the master's engine 0) reads every
    /// source shard of its local index. When that index is sparse (the
    /// master mirrored it rather than creating it), open the source shards
    /// it lacks for the duration of the resize; [`Self::close_resize_sources`]
    /// closes them again. The cluster copies the documents itself
    /// (`resize::copy_docs`), so these engine-0 shells only need to exist.
    pub fn open_resize_sources(&self, index: &str) -> Vec<u32> {
        let Some(svc) = self.engine_for(0).and_then(|e| e.get_index(index)) else { return Vec::new() };
        if !svc.is_sparse() {
            return Vec::new();
        }
        let n: u32 = svc.setting("index.number_of_shards").and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut opened = Vec::new();
        for k in 0..n {
            if svc.shard(k).is_some() {
                continue;
            }
            match Self::ensure_copy_open(&svc, k, "resize-source") {
                Ok(()) => opened.push(k),
                Err(e) => tracing::warn!(%index, shard = k, "resize: {e}"),
            }
        }
        opened
    }

    /// Close the engine-0 source shells [`Self::open_resize_sources`]
    /// opened, except one the cluster meanwhile assigned here (engine 0
    /// holds cluster shard 0).
    pub fn close_resize_sources(&self, index: &str, opened: &[u32]) {
        let Some(svc) = self.engine_for(0).and_then(|e| e.get_index(index)) else { return };
        for &k in opened {
            if k == 0 && self.shards.get(index, 0).is_some() {
                continue;
            }
            if let Err(e) = svc.close_shard(k) {
                if e.kind != "shard_not_found_exception" {
                    tracing::warn!(%index, shard = k, "resize: failed to close a source shell: {}", e.reason);
                }
            }
        }
    }

    /// Is the copy on disk one of the shard's in-sync copies (i.e. may it
    /// hold acknowledged writes no other reachable copy has)?
    fn local_copy_in_sync(&self, state: &ClusterState, index: &str, shard: u32) -> bool {
        let Some(im) = state.metadata.indices.get(index) else { return false };
        let Some(svc) = self.engine_for(shard).and_then(|e| e.get_index(index)) else { return false };
        let Some(rec) = shards::read_copy_record(svc.path()) else { return false };
        rec.index_uuid == im.uuid && rec.shard == shard && im.in_sync_allocation_ids.get(&shard).map(|s| s.contains(&rec.allocation_id)).unwrap_or(false)
    }

    fn reconcile_one(&self, state: &ClusterState, index: &str, shard: u32, routing: &ShardRouting) {
        let Some(im) = state.metadata.indices.get(index) else { return };
        let Some(aid) = routing.allocation_id().map(str::to_string) else { return };
        let term = im.primary_term(shard);
        let existing = self.shards.get(index, shard);
        if let Some(local) = &existing {
            let mut g = local.lock();
            if g.allocation_id == aid {
                match (&g.role, routing.primary) {
                    (Role::Primary(p), true) => {
                        // Engine 0 re-created the index shell beneath a held
                        // primary (a rollback or catch-up of the master's
                        // engine to a state that still names the index): the
                        // context keeps its old service and can never write
                        // again. Fail the copy so a replica is promoted or
                        // the shard reallocated, instead of every write
                        // answering "primary moved" until it times out. A
                        // missing shell is a delete still being published.
                        if self.engine_for(shard).and_then(|e| e.get_index(index)).is_some_and(|svc| !Arc::ptr_eq(&svc, &p.svc)) {
                            tracing::warn!(%index, shard, "failing a primary whose engine copy was replaced beneath it");
                            self.report_failed(index, shard, &aid, None, "the primary's engine copy was replaced", false);
                            return;
                        }
                        self.refresh_primary_targets(state, p);
                        return;
                    }
                    (Role::Replica(r), false) => {
                        if term > r.primary_term() {
                            if let Err(e) = r.bump_term(term, r.global_checkpoint()) {
                                tracing::warn!(%index, shard, "term bump failed: {e}");
                            }
                        }
                        if let Some(reason) = r.needs_full_recovery() {
                            self.report_failed(index, shard, &aid, None, &format!("copy diverged from the new primary: {reason}"), true);
                        }
                        return;
                    }
                    // A stateless primary being rebuilt from the object store
                    // (`recover_primary_from_object_store`) starts itself once
                    // the replay is done; promoting (or term-bumping) its
                    // placeholder here would abort the running replay.
                    (Role::Replica(_), true) if routing.state == ShardState::Initializing && routing.relocating_node_id.is_none() && g.recovery.is_some() && self.object_store.is_some() && self.local.is_stateless() => {
                        return;
                    }
                    (Role::Replica(r), true) if !replica_may_promote(routing.primary, &routing.state) => {
                        // A primary relocation target that is still
                        // recovering. Its routing entry already says
                        // `primary`, but it must keep the replica role until
                        // the source handed off and the master started it:
                        // promoting here would accept writes on a partly
                        // recovered copy and acknowledge them without the
                        // real primary ever seeing them.
                        if term > r.primary_term() {
                            if let Err(e) = r.bump_term(term, r.global_checkpoint()) {
                                tracing::warn!(%index, shard, "term bump failed: {e}");
                            }
                        }
                        return;
                    }
                    (Role::Replica(r), true) => {
                        // Promotion (failover or relocation handoff).
                        let r = r.clone();
                        match self.promote(index, shard, &aid, term, &g.svc, &r) {
                            Ok((p, resync)) => {
                                self.refresh_primary_targets(state, &p);
                                g.role = Role::Primary(p.clone());
                                g.term = term;
                                self.shards.mark_primary(index, shard);
                                if let Some(resync) = resync {
                                    let me = self.me();
                                    tokio::spawn(async move { me.resync_replicas(p, resync).await });
                                }
                            }
                            Err(e) => self.report_failed(index, shard, &aid, None, &format!("failed to promote replica: {e}"), false),
                        }
                        return;
                    }
                    (Role::Primary(_), false) => {
                        // Demoted (shouldn't happen for the same allocation id).
                        drop(g);
                        self.remove_copy(index, shard);
                    }
                }
            } else {
                drop(g);
                self.remove_copy(index, shard);
            }
        }
        // The shard's old store is being set aside (an earlier allocation's
        // file recovery): that task reconciles this allocation once done.
        if self.pending.lock().contains(&files_pending_key(index, shard)) {
            return;
        }
        // An existing-store primary of this shard is being opened off the
        // applier: that task starts it, or reconciles whatever allocation
        // replaced it once its open is done.
        if self.pending.lock().contains(&primary_open_pending_key(index, shard)) {
            return;
        }
        let Some(svc) = self.engine_for(shard).and_then(|e| e.get_index(index)) else { return };
        // A primary relocation target recovers like a replica; it becomes
        // primary only after the source handed off and the master started it.
        let relocation_target = routing.primary && routing.state == ShardState::Initializing && routing.relocating_node_id.is_some() && matches!(routing.recovery_source, Some(RecoverySource::Peer));
        if routing.primary && !relocation_target {
            if routing.state != ShardState::Initializing {
                // We were told we hold an active primary we know nothing
                // about (restart before the master noticed): fail it.
                self.report_failed(index, shard, &aid, None, "shard copy is not present on this node", false);
                return;
            }
            // ES `StoreRecovery.restore`: a primary restored from a snapshot
            // adopts the snapshot's files (off the applier).
            if matches!(routing.recovery_source, Some(RecoverySource::Snapshot { .. })) {
                self.restore_primary_from_snapshot(state, index, shard, &aid, term);
                return;
            }
            let from_existing = matches!(routing.recovery_source, Some(RecoverySource::ExistingStore { .. }));
            // Stateless: an existing shard's primary always comes back from
            // the object store (the local disk is only a cache in ES).
            if from_existing && self.object_store.is_some() && self.local.is_stateless() {
                self.recover_primary_from_object_store(state, index, shard, routing, im.uuid.clone(), term);
                return;
            }
            // G6: an existing-store primary must open a copy that is in the
            // in-sync set. The master's store listing can be stale (the copy
            // may have been wiped since), so check what is really on disk.
            let in_sync_known = im.in_sync_allocation_ids.get(&shard).map(|s| !s.is_empty()).unwrap_or(false);
            if from_existing && in_sync_known && !self.local_copy_in_sync(state, index, shard) {
                tracing::warn!(%index, shard, "refusing to start a primary from a local copy that is not in-sync");
                self.report_failed(index, shard, &aid, None, "local copy is not an in-sync copy of this shard", false);
                return;
            }
            // ES's `IndicesClusterStateService` creates the shard and starts
            // its recovery asynchronously, reporting it started through a
            // master task. Opening an existing store (a WAL replay, vector
            // graphs: 75 s for a large vector shard) on the applier held up
            // every later state and heartbeat until the master removed the
            // node as lagging. An existing-store primary opens on the
            // recovery pool; an empty one (a new index) opens here.
            if from_existing {
                self.open_primary_then_start(index, shard, &aid, term, svc);
                return;
            }
            let start = || -> Result<Arc<PrimaryCtx>, String> {
                Self::ensure_copy_open(&svc, shard, &aid)?;
                PrimaryCtx::new(index, shard, &aid, term, svc.clone()).map(Arc::new).map_err(|e| e.to_string())
            };
            match start() {
                Ok(p) => self.install_started_primary(state, index, shard, &aid, term, svc, p, routing),
                Err(e) => {
                    // Leave no orphan open copy behind (data stays on disk).
                    if svc.is_sparse() {
                        let _ = svc.close_shard(shard);
                    }
                    self.report_failed(index, shard, &aid, None, &format!("failed to start primary: {e}"), false)
                }
            }
            return;
        }
        if routing.state != ShardState::Initializing {
            self.report_failed(index, shard, &aid, None, "shard copy is not present on this node", false);
            return;
        }
        self.start_peer_recovery(state, index, shard, routing, 0, false);
    }

    /// Record and register a primary whose copy is open, and report it
    /// started to the master. Under the apply lock.
    #[allow(clippy::too_many_arguments)]
    fn install_started_primary(&self, state: &ClusterState, index: &str, shard: u32, aid: &str, term: u64, svc: Arc<IndexService>, p: Arc<PrimaryCtx>, routing: &ShardRouting) {
        let Some(im) = state.metadata.indices.get(index) else { return };
        // ES `bootstrapNewHistoryUUID`: a forced stale or empty primary
        // starts a new history; an existing-store primary continues its.
        let continues = matches!(routing.recovery_source, Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: false }));
        let history_uuid = shards::primary_history_uuid(&svc, &im.uuid, shard, continues);
        self.attach_object_store(&p, &im.uuid);
        self.refresh_primary_targets(state, &p);
        let checkpoint = shards::primary_checkpoint(&svc, &im.uuid, shard, &history_uuid, -1);
        let _ = shards::write_copy_record(&svc, &CopyRecord { index: index.into(), index_uuid: im.uuid.clone(), shard, allocation_id: aid.to_string(), primary_term: term, primary: true,
            history_uuid: Some(history_uuid), global_checkpoint: Some(checkpoint) });
        self.shards.insert(LocalShard { index: index.into(), shard, allocation_id: aid.to_string(), term, svc, role: Role::Primary(p), recovery: None });
        self.report_started(index, shard, aid);
    }

    /// Start an existing-store primary without blocking the cluster
    /// applier (ES: the applier creates the shard, its store recovery runs
    /// on the generic pool, and `shard-started` goes to the master as a
    /// task). The copy is opened, its WAL replayed and the copy promoted on
    /// the recovery pool ([`Self::open_local_with_backoff`]: a memory
    /// refusal is retried, never answered by discarding the copy). Then,
    /// under the apply lock, the primary is installed and reported started
    /// only if the applied state still wants exactly this allocation here
    /// (same index incarnation, still an initializing existing-store
    /// primary, same primary term); otherwise the open copy is closed (the
    /// data stays) and whatever the state assigns here now is reconciled.
    /// While the open runs, the pending key keeps every other
    /// reconciliation (and the stale-data cleanup) away from this shard.
    fn open_primary_then_start(&self, index: &str, shard: u32, aid: &str, term: u64, svc: Arc<IndexService>) {
        let key = primary_open_pending_key(index, shard);
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());
        tokio::spawn(async move {
            let (open_svc, open_index, open_aid) = (svc.clone(), index.clone(), aid.clone());
            let (wanted_index, wanted_aid) = (index.clone(), aid.clone());
            let opened = Self::open_local_with_backoff(&me, &index, shard, &svc,
                // A failure after the copy opened closes it again (whatever
                // the allocation does meanwhile): no exit path but the
                // install keeps an open copy.
                move || release_open_on_failure(&open_svc, shard, || -> Result<Arc<PrimaryCtx>, String> {
                    Self::ensure_copy_open(&open_svc, shard, &open_aid)?;
                    // open() replays any replica WAL left by a previous replica
                    // role. Do not replay it again before promotion.
                    let r = ReplicaApplier::open(open_svc.clone(), shard).map_err(|e| e.to_string())?;
                    r.promote_to_primary(term).map_err(|e| e.to_string())?;
                    PrimaryCtx::new(&open_index, shard, &open_aid, term, open_svc.clone()).map(Arc::new).map_err(|e| e.to_string())
                }),
                move |n: &ClusterNode| n.shards.get(&wanted_index, shard).is_none()
                    && n.my_routing(&n.state(), &wanted_index, shard, &wanted_aid).is_some_and(|r| r.primary && r.state == ShardState::Initializing)).await;
            // The apply lock is held across copy closes and shard retries:
            // wait for it on a blocking thread, not a runtime worker or the
            // recovery pool (see `launch_peer_recovery`).
            let done = tokio::task::spawn_blocking(move || {
                let _g = me.apply_lock.lock();
                me.pending.lock().remove(&key);
                let state = me.state();
                let same_svc = me.engine_for(shard).and_then(|e| e.get_index(&index)).is_some_and(|current| Arc::ptr_eq(&current, &svc));
                let free = me.shards.get(&index, shard).is_none();
                let routing = me.my_routing(&state, &index, shard, &aid)
                    .filter(|r| r.primary && r.state == ShardState::Initializing && r.relocating_node_id.is_none()
                        && matches!(r.recovery_source, Some(RecoverySource::ExistingStore { .. })));
                let same_term = state.metadata.indices.get(&index).is_some_and(|im| im.primary_term(shard) == term);
                match (opened, routing) {
                    (Ok(p), Some(routing)) if same_svc && free && same_term => {
                        me.install_started_primary(&state, &index, shard, &aid, term, svc, p, &routing);
                        return;
                    }
                    (Err(e), Some(_)) if same_svc && free => {
                        // Leave no orphan open copy behind (data stays on disk).
                        if svc.is_sparse() {
                            let _ = svc.close_shard(shard);
                        }
                        me.report_failed(&index, shard, &aid, None, &format!("failed to start primary: {e}"), false);
                        return;
                    }
                    (Ok(p), _) => {
                        // The allocation moved on (or its term did) while the
                        // copy opened: close it; the data stays.
                        let _ = svc.close_shard_if_same(&p.copy());
                    }
                    // The open closed what it had opened before failing
                    // (`release_open_on_failure`).
                    (Err(e), _) => tracing::info!(%index, shard, "stale primary open stopped: {e}"),
                }
                let me_id = me.local.id.clone();
                let assigned = state.routing_table.shard_table(&index, shard)
                    .and_then(|t| t.copies.iter().find(|c| c.node_id.as_deref() == Some(me_id.as_str()) && c.state != ShardState::Unassigned).cloned());
                if let Some(routing) = assigned.filter(|_| me.shards.get(&index, shard).is_none()) {
                    me.reconcile_one(&state, &index, shard, &routing);
                }
            }).await;
            if let Err(e) = done {
                tracing::warn!(shard, "primary start completion task failed: {e}");
            }
        });
    }

    /// A primary allocated here with a `Snapshot` recovery source (see
    /// `snapshot_api::file_restore`): the copy is reset, the snapshot's shard
    /// files are downloaded, verified and adopted on the recovery pool, and
    /// the copy is promoted. Then, under the apply lock, it is installed and
    /// reported started if the applied state still wants exactly this
    /// allocation here (as [`Self::open_primary_then_start`]); a failure is
    /// reported (the master never allocates the restore again, so nothing
    /// partial starts). The restored data of an allocation that moved on is
    /// deleted. The pending key keeps every other reconciliation (and the
    /// stale-data cleanup) away from the shard meanwhile.
    fn restore_primary_from_snapshot(&self, state: &ClusterState, index: &str, shard: u32, aid: &str, term: u64) {
        use crate::snapshot_api::file_restore;
        let key = primary_open_pending_key(index, shard);
        if self.pending.lock().contains(&key) {
            return;
        }
        let Some(im) = state.metadata.indices.get(index) else { return };
        let uuid = im.uuid.clone();
        let Some((restore_uuid, spec)) = file_restore::file_restore_of(state, index, &uuid) else {
            // ES `RestoreInProgressAllocationDecider`: no restore in progress
            // for this incarnation (it was interrupted).
            self.report_failed(index, shard, aid, None, "failed to restore from snapshot: the restore of this index is no longer in progress", false);
            return;
        };
        // An earlier attempt's store, or the master's own full-open copy:
        // the snapshot's files replace it.
        let Some(svc) = self.wipe_local_copy(index, shard, state) else {
            self.report_failed(index, shard, aid, None, "failed to restore from snapshot: cannot reset the local shard copy", false);
            return;
        };
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());
        tokio::spawn(async move {
            let restored = me.restore_primary_files(&index, shard, &aid, term, &svc, &restore_uuid, &spec).await;
            // The apply lock is held across copy closes: wait for it on a
            // blocking thread (see `open_primary_then_start`).
            let done = tokio::task::spawn_blocking(move || {
                let _g = me.apply_lock.lock();
                me.pending.lock().remove(&key);
                let state = me.state();
                let same_svc = me.engine_for(shard).and_then(|e| e.get_index(&index)).is_some_and(|current| Arc::ptr_eq(&current, &svc));
                let free = me.shards.get(&index, shard).is_none();
                let routing = me.my_routing(&state, &index, shard, &aid)
                    .filter(|r| r.primary && r.state == ShardState::Initializing && matches!(r.recovery_source, Some(RecoverySource::Snapshot { .. })));
                let same = state.metadata.indices.get(&index).is_some_and(|im| im.uuid == uuid && im.primary_term(shard) == term);
                // A restored copy that does not start here: its store goes.
                let discard = |svc: &Arc<IndexService>| {
                    if let Err(e) = svc.remove_shard(shard, true) {
                        tracing::warn!(%index, shard, "failed to remove a restored copy that did not start: {}", e.reason);
                    }
                };
                match (restored, routing) {
                    (Ok((p, rec)), Some(routing)) if same_svc && free && same => {
                        me.install_started_primary(&state, &index, shard, &aid, term, svc, p, &routing);
                        if let Some(local) = me.shards.get(&index, shard) {
                            local.lock().recovery = Some(Arc::new(Mutex::new(Some(rec))));
                        }
                        return;
                    }
                    (Err(e), Some(_)) if same_svc && free => {
                        discard(&svc);
                        tracing::warn!(%index, shard, "{e}");
                        me.report_failed(&index, shard, &aid, None, &e, false);
                        return;
                    }
                    // Another copy holds the slot: close only the restored one.
                    (Ok((p, _)), _) if !free => {
                        let _ = svc.close_shard_if_same(&p.copy());
                    }
                    (Ok(_), _) => discard(&svc),
                    (Err(e), _) => {
                        if free {
                            discard(&svc);
                        }
                        tracing::info!(%index, shard, "stale snapshot restore stopped: {e}");
                    }
                }
                let me_id = me.local.id.clone();
                let assigned = state.routing_table.shard_table(&index, shard)
                    .and_then(|t| t.copies.iter().find(|c| c.node_id.as_deref() == Some(me_id.as_str()) && c.state != ShardState::Unassigned).cloned());
                if let Some(routing) = assigned.filter(|_| me.shards.get(&index, shard).is_none()) {
                    me.reconcile_one(&state, &index, shard, &routing);
                }
            }).await;
            if let Err(e) = done {
                tracing::warn!(shard, "snapshot restore completion task failed: {e}");
            }
        });
    }

    /// The work of [`Self::restore_primary_from_snapshot`]: download and
    /// verify the shard's snapshot files into its staging directory, open
    /// them as the copy (`FromRecoveredFiles`) and promote it under `term`.
    /// Returns the primary context and the copy's `_recovery` record.
    #[allow(clippy::too_many_arguments)]
    async fn restore_primary_files(&self, index: &str, shard: u32, aid: &str, term: u64, svc: &Arc<IndexService>, restore_uuid: &str,
        spec: &crate::snapshot_api::file_restore::FileRestoreSpec) -> Result<(Arc<PrimaryCtx>, RecoveryState), String>
    {
        use crate::replication::file_recovery as fr;
        use crate::snapshot_api::file_restore;
        use elasticsearch_http::pools::{self, Pool};
        let now = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        let start = now();
        let staging = fr::staging_dir(svc, shard);
        let _staging_lock = fr::StagingLock::try_acquire(&staging)?.ok_or(fr::STAGING_BUSY)?;
        // The segment format every member reads (replicas, any later
        // primary): the lowest advertised read maximum, never below the
        // cluster's floor.
        let readable = {
            use elasticsearch_cluster::compatibility::{advertised_read_max, effective_segment_format_floor, negotiated_segment_format};
            let state = self.state();
            negotiated_segment_format(&state, advertised_read_max).max(effective_segment_format_floor(&state.metadata))
        };
        let staged = match file_restore::stage_shard(self, svc, shard, spec, &staging, readable).await {
            Ok(staged) => staged,
            Err(e) => {
                // Nothing of a failed download is kept (a retry downloads again).
                let spare = staging.clone();
                let _ = pools::run(Pool::Recovery, move || std::fs::remove_dir_all(&spare)).await;
                return Err(e.message());
            }
        };
        let (open_svc, open_index, open_aid, open_staging, manifest) = (svc.clone(), index.to_string(), aid.to_string(), staging.clone(), staged.manifest.clone());
        let opened = pools::run(Pool::Recovery, move || release_open_on_failure(&open_svc, shard, || -> Result<(Arc<PrimaryCtx>, u64, i64), String> {
            // The engine moves the staged shard into `shard-<n>/` after
            // checking every manifest file is there with its length (each was
            // verified against the snapshot as it was written), then replays
            // the translog of a snapshot staged without a flush.
            let copy = open_svc.open_shard(shard, ShardOpenSpec { allocation_id: open_aid.clone(), primary: true, primary_term: 1,
                recovery: EngineRecovery::FromRecoveredFiles { staged: open_staging.clone(), manifest } })
                .map_err(|e| format!("cannot open the restored files: [{}] {}", e.kind, e.reason))?;
            let replayed_to = copy.shard.local_checkpoint();
            // Promoted as `promote` does: seq_no gaps a replayed translog left
            // are filled with no-ops of the new term.
            let r = ReplicaApplier::open(open_svc.clone(), shard).map_err(|e| e.to_string())?;
            if term > r.primary_term() || r.promotion_pending() == Some(term) {
                r.promote_to_primary(term).map_err(|e| e.to_string())?;
            }
            let p = PrimaryCtx::new(&open_index, shard, &open_aid, term, open_svc.clone()).map(Arc::new).map_err(|e| e.to_string())?;
            Ok((p, copy.doc_count().0, replayed_to))
        })).await.map_err(|e| e.reason).and_then(|r| r);
        // What the engine did not move in (the source's index metadata).
        let spare = staging.clone();
        let _ = pools::run(Pool::Recovery, move || std::fs::remove_dir_all(&spare)).await;
        let (p, docs, checkpoint) = opened.map_err(|e| format!("failed to restore from snapshot: {e}"))?;
        let replayed = (checkpoint - staged.commit_checkpoint).max(0) as u64;
        let rec = file_restore::recovery_state(index, shard, spec, restore_uuid, &self.local.id, &staged, docs, replayed, start, now())?;
        tracing::info!(index, shard, docs, files = staged.manifest.files.len(), bytes = staged.bytes, translog_ops = replayed, "restored the primary from the snapshot's files");
        Ok((p, rec))
    }

    /// Replica / relocation target: recover from the active primary.
    ///
    /// A recovery that fails because the source is unreachable (or has not
    /// caught up with the routing table yet) is retried in place with a
    /// backoff instead of failing the shard, like ES's
    /// `indices.recovery.retry_delay_network`. Reporting every such failure
    /// straight away let a just-killed primary burn through the `max_retry`
    /// allocation budget (5) in a few hundred ms, before the master had even
    /// noticed the node was gone, leaving the replica unassigned for good
    /// (FAULT_RESULTS.md bug #2).
    ///
    /// A local copy sharing the primary's history catches up from
    /// operations; otherwise the primary's commit is copied as files, the
    /// local copy's identical files reused (see `start_peer_recovery_with`).
    /// When the local copy is still in-sync it may be the only one holding
    /// some acknowledged write (after a full restart the routing table calls
    /// copies "started" that no node has opened), so it is only replaced
    /// once the source confirmed that it really runs the primary
    /// (`confirmed`).
    fn start_peer_recovery(&self, state: &ClusterState, index: &str, shard: u32, routing: &ShardRouting, attempt: u32, confirmed: bool) {
        self.start_peer_recovery_mode(state, index, shard, routing, attempt, confirmed, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn start_peer_recovery_mode(&self, state: &ClusterState, index: &str, shard: u32, routing: &ShardRouting, attempt: u32, confirmed: bool, allow_reuse: bool) {
        self.start_peer_recovery_with(state, index, shard, routing, attempt, confirmed, allow_reuse, self.file_based_recovery());
    }

    /// `indices.recovery.file_based` (node setting, default `true`).
    fn file_based_recovery(&self) -> bool {
        !matches!(self.config.settings.get(crate::replication::file_recovery::FILE_BASED), Some(serde_json::Value::Bool(false)))
            && !matches!(self.config.settings.get(crate::replication::file_recovery::FILE_BASED).and_then(|v| v.as_str()), Some("false"))
    }

    /// How a peer recovery gets the data (ES `RecoverySourceHandler`
    /// chooses between phase 1 and an operations-based recovery the same
    /// way: operations when the target already holds the history, files
    /// otherwise).
    ///
    /// 1. The local copy may catch up by operations ([`Self::local_copy_reusable`]):
    ///    it is opened and its WAL replayed off the applier, and its prefix
    ///    offered to the primary ([`Self::prepare_reuse_then_recover`]).
    /// 2. Otherwise (or when the primary declines) files: the old store is
    ///    set aside for reuse of its identical files, an empty placeholder
    ///    holds the slot, and the primary's commit is copied
    ///    ([`Self::recover_replica_files`]).
    /// 3. `files == false` (node setting, or a source that predates file
    ///    recovery): the store is wiped and every document re-indexed from a
    ///    snapshot of the primary.
    #[allow(clippy::too_many_arguments)]
    fn start_peer_recovery_with(&self, state: &ClusterState, index: &str, shard: u32, routing: &ShardRouting, attempt: u32, confirmed: bool, allow_reuse: bool, files: bool) {
        let Some(im) = state.metadata.indices.get(index) else { return };
        let Some(aid) = routing.allocation_id().map(str::to_string) else { return };
        // The shard's old store is being set aside off the applier: that
        // task starts the recovery (or reconciles the allocation that
        // replaced it).
        if self.pending.lock().contains(&files_pending_key(index, shard)) {
            return;
        }
        let term = im.primary_term(shard);
        let relocation_target = routing.primary && routing.state == ShardState::Initializing && routing.relocating_node_id.is_some() && matches!(routing.recovery_source, Some(RecoverySource::Peer));
        let Some(primary_node) = state.routing_table.shard_table(index, shard).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone()) else { return };
        if !confirmed && self.local_copy_in_sync(state, index, shard) {
            self.probe_then_recover(index, shard, &aid, attempt, allow_reuse);
            return;
        }
        // Never trust index name or equal document counts as history proof:
        // the copy must belong to this index incarnation and share the
        // primary's history, which the primary checks again. Retained reader
        // views own immutable mapped bytes and a captured vector generation;
        // they do not reopen files through the retired directory. Closing the
        // old writer fences both new file opens and writes before reuse.
        if allow_reuse && !relocation_target && self.local_copy_reusable(state, index, shard) {
            self.prepare_reuse_then_recover(index, shard, &aid, attempt, files);
            return;
        }
        let stateless = self.object_store.is_some() && self.local.is_stateless();
        let file_based = files && !stateless;
        if file_based {
            // A quarantined service retains the rejected immutable mapping.
            // Reusing only its files/slot would reopen that same fenced core
            // on every allocation attempt. Fall through to recreation from
            // committed metadata instead.
            if let Some(svc) = self.same_incarnation_index(index, shard, state).filter(|svc| !svc.is_detached()) {
                self.set_aside_then_recover_files(index, shard, &aid, svc, attempt);
                return;
            }
        }
        let Some(svc) = self.wipe_local_copy_mode(index, shard, state, file_based) else {
            // Never a silent stop: nothing would retry this allocation (it
            // stays INITIALIZING) until some other state change arrives.
            if self.my_routing(state, index, shard, &aid).is_some() {
                self.report_failed(index, shard, &aid, None, "failed recovery: the local shard copy could not be reset", false);
            }
            return;
        };
        if let Err(e) = Self::ensure_copy_open(&svc, shard, &aid) {
            self.report_failed(index, shard, &aid, None, &format!("failed recovery: {e}"), false);
            return;
        }
        if file_based {
            let _ = std::fs::write(shards::shard_path(&svc, shard).join(crate::replication::file_recovery::PLACEHOLDER_MARKER), b"");
        }
        let applier = match ReplicaApplier::open(svc.clone(), shard) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                self.report_failed(index, shard, &aid, None, &format!("failed recovery: {e}"), false);
                return;
            }
        };
        let mode = if file_based { PeerMode::Files } else { PeerMode::Ops(None) };
        self.launch_peer_recovery(index, shard, &aid, &im.uuid, term, svc, applier, &primary_node, relocation_target, attempt, mode);
    }

    /// The file-based branch of [`Self::start_peer_recovery_with`] for this
    /// incarnation's copy. Setting the old store aside
    /// (`shards::set_aside_copy_for_recovery`: it may delete an older store
    /// set aside, tens of GB, and wait up to 5 s for a closing copy's shard
    /// lock) and opening the placeholder run on the recovery pool, never
    /// under the apply lock or on a runtime worker; the recovery is then
    /// launched under the apply lock if the routing table still wants this
    /// copy here (term, source and relocation as of then). Meanwhile the
    /// pending key keeps any other reconciliation of this allocation from
    /// starting one (or touching the store being moved).
    fn set_aside_then_recover_files(&self, index: &str, shard: u32, aid: &str, svc: Arc<IndexService>, attempt: u32) {
        use crate::replication::file_recovery as fr;
        use elasticsearch_http::pools::{self, Pool};
        let key = files_pending_key(index, shard);
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        // Keep reader contexts for this incarnation. Closing/moving the old
        // store fences pathname access, while each retained view owns its
        // immutable mapped file and vector generation until it expires.
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());
        tokio::spawn(async move {
            let (prep_svc, prep_aid) = (svc.clone(), aid.clone());
            let prepared = pools::run(Pool::Recovery, move || -> Result<Arc<ReplicaApplier>, String> {
                // The copy this allocation replaces (removed from this node
                // just before) closes in the background and holds its shard
                // lock until then: wait for it (`set_aside_copy_for_recovery`
                // itself waits 5 s at a time) rather than fail the copy.
                let deadline = Instant::now() + SET_ASIDE_LOCK_WAIT;
                loop {
                    match shards::set_aside_copy_for_recovery(&prep_svc, shard) {
                        Ok(()) => break,
                        Err(e) if e.contains("shard_lock_obtain_failed_exception") && Instant::now() < deadline => {
                            tracing::info!(index = %prep_svc.name, shard, "waiting for the replaced copy to close: {e}");
                        }
                        Err(e) => return Err(format!("failed to reset the local shard copy in place: {e}")),
                    }
                }
                Self::ensure_copy_open(&prep_svc, shard, &prep_aid)?;
                let _ = std::fs::write(shards::shard_path(&prep_svc, shard).join(fr::PLACEHOLDER_MARKER), b"");
                ReplicaApplier::open(prep_svc.clone(), shard).map(Arc::new).map_err(|e| e.to_string())
            }).await.map_err(|e| e.reason).and_then(|r| r);
            let rt = tokio::runtime::Handle::current();
            let _ = pools::run(Pool::Recovery, move || {
                // `launch_peer_recovery` spawns the recovery task.
                let _rt = rt.enter();
                let _g = me.apply_lock.lock();
                me.pending.lock().remove(&key);
                let state = me.state();
                let routing = me.my_routing(&state, &index, shard, &aid).filter(|r| r.state == ShardState::Initializing && me.shards.get(&index, shard).is_none());
                let launch = routing.and_then(|r| {
                    let im = state.metadata.indices.get(&index)?;
                    let primary_node = state.routing_table.shard_table(&index, shard)?.active_primary()?.node_id.clone()?;
                    let relocation_target = r.primary && r.relocating_node_id.is_some() && matches!(r.recovery_source, Some(RecoverySource::Peer));
                    Some((im.uuid.clone(), im.primary_term(shard), primary_node, relocation_target))
                });
                match (prepared, launch) {
                    (Ok(applier), Some((uuid, term, primary_node, relocation_target))) => {
                        me.launch_peer_recovery(&index, shard, &aid, &uuid, term, svc, applier, &primary_node, relocation_target, attempt, PeerMode::Files);
                    }
                    (Err(e), Some(_)) => me.report_failed(&index, shard, &aid, None, &format!("failed recovery: {e}"), false),
                    // The allocation moved on: close the placeholder (an
                    // empty store; the old one stays set aside), then start
                    // whatever the routing table assigns here now, which
                    // waited for this task.
                    (prepared, None) => {
                        match prepared {
                            Ok(applier) => {
                                let _ = svc.close_shard_if_same(&applier.copy());
                            }
                            Err(e) => tracing::info!(%index, shard, "stale file recovery preparation stopped: {e}"),
                        }
                        let me_id = me.local.id.clone();
                        let assigned = state.routing_table.shard_table(&index, shard)
                            .and_then(|t| t.copies.iter().find(|c| c.node_id.as_deref() == Some(me_id.as_str()) && c.state != ShardState::Unassigned).cloned());
                        if let Some(routing) = assigned.filter(|_| me.shards.get(&index, shard).is_none()) {
                            me.reconcile_one(&state, &index, shard, &routing);
                        }
                    }
                }
            }).await;
        });
    }

    /// Insert the recovering copy (its applier: the reused copy's, or the
    /// empty placeholder's) and run its recovery, retrying in place on
    /// network-level failures (see [`Self::start_peer_recovery`]).
    #[allow(clippy::too_many_arguments)]
    fn launch_peer_recovery(&self, index: &str, shard: u32, aid: &str, uuid: &str, term: u64, svc: Arc<IndexService>, applier: Arc<ReplicaApplier>, primary_node: &str, relocation_target: bool, attempt: u32, mode: PeerMode) {
        match self.capacity_leases.lock().has_allocation(self,&self.state(),index,shard,aid,Instant::now()) {
            Ok(false)=>{},
            Ok(true)=>{self.report_failed(index,shard,aid,None,"recovery admission attempt already exists",false);return;},
            Err(error)=>{self.report_failed(index,shard,aid,None,error,false);return;}
        }
        let admission:crate::capacity::lease::TaskSlot=Arc::new(Mutex::new(None));
        let slot: Arc<Mutex<Option<RecoveryState>>> = Arc::new(Mutex::new(None));
        self.shards.insert(LocalShard { index: index.into(), shard, allocation_id: aid.into(), term, svc: svc.clone(), role: Role::Replica(applier.clone()), recovery: Some(slot.clone()) });
        let me = self.me();
        let (index, aid, uuid, primary_node) = (index.to_string(), aid.to_string(), uuid.to_string(), primary_node.to_string());
        let reusing = matches!(mode, PeerMode::Ops(Some(_)));
        let object_store = self.object_store.as_ref().filter(|_| self.local.is_stateless() && !relocation_target && attempt == 0 && !reusing).map(|o| o.shard(&uuid, shard));
        let target_copy = applier.copy();
        tokio::spawn(async move {
            let r = match (object_store, mode) {
                // Stateless search copy: the object store first, then only
                // the primary's translog above it.
                (Some(os), _) => me.recover_replica_from_object_store(&index, shard, &aid, &uuid, term, svc.clone(), applier.clone(), &primary_node, slot, os, &admission).await,
                (None, PeerMode::Files) => me.recover_replica_files(&index, shard, &aid, &uuid, term, svc.clone(), applier.clone(), &primary_node, slot, relocation_target, &admission).await,
                (None, PeerMode::Ops(reuse)) => me.recover_replica(&index, shard, &aid, &uuid, term, svc.clone(), applier.clone(), &primary_node, slot, relocation_target, reuse, &admission).await,
            };
            let admission=admission.lock().take();
            // The applier holds `apply_lock` across copy closes/wipes and
            // shard retries that sleep for seconds: wait for it (and the
            // capacity leases under it) on a blocking thread, never on a
            // runtime worker, or a burst of completions after a restart
            // parks the workers HTTP and transport run on. Not the recovery
            // pool either: in-flight recoveries run their steps there, and
            // the applier may be waiting on one of them while it holds the
            // lock.
            let (finish_me, finish_index, finish_aid, finish_svc) = (me.clone(), index.clone(), aid.clone(), svc.clone());
            let finished = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let (me, index, aid, svc) = (finish_me, finish_index, finish_aid, finish_svc);
                match (r,admission) {
                    (Ok(finished_copy),admission)=>{
                        let _apply=me.apply_lock.lock();
                        let current=me.recovery_current(&index,shard,&aid,&svc,&finished_copy);
                        if !current {
                            if let Some(envelope)=admission {me.capacity_leases.lock().cancel(envelope.epoch);}
                            Err(STALE_RECOVERY.into())
                        } else {
                            let completion=match admission {
                                Some(envelope)=>me.capacity_leases.lock().complete(&me,&me.state(),envelope.epoch,&envelope.identity,Instant::now()),
                                None=>Err("recovery completed without a target admission envelope".into()),
                            };
                            completion.map(|()|me.report_started(&index,shard,&aid))
                        }
                    },
                    (Err(error),Some(envelope))=>{me.capacity_leases.lock().cancel(envelope.epoch);Err(error)},
                    (Err(error),None)=>Err(error),
                }
            }).await;
            let r = finished.unwrap_or_else(|e| Err(format!("recovery completion task failed: {e}")));

            let Err(e) = r else { return };
            // A task whose copy was removed meanwhile has nothing to retry
            // or report: its allocation is gone, its directory may already
            // belong to a newer copy (FAULT_RESULTS.md §10). (A file recovery
            // swapped its placeholder for the received copy: check either.
            // One whose open of the received copy failed holds the retired
            // placeholder: still current, it fails the copy.)
            let current = me.recovery_current(&index, shard, &aid, &svc, &target_copy)
                || me.recovery_slot_held(&index, shard, &aid, &svc, &target_copy)
                || me.shards.replica(&index, shard, &aid).is_some_and(|r| Arc::ptr_eq(r.index(), &svc) && svc.shard(shard).is_some_and(|c| Arc::ptr_eq(&c, &r.copy())));
            if !current {
                tracing::info!(%index, shard, "stale recovery task stopped: {e}");
                return;
            }
            if e.starts_with(crate::replication::file_recovery::INSTALL_FAILED) {
                // Retried in place already: ES fails the shard, and the
                // master reallocates it (the received files are kept for a
                // copy allocated here again).
                tracing::warn!(%index, shard, "recovery failed: {e}");
                me.report_failed(&index, shard, &aid, None, &format!("failed recovery: {e}"), false);
                return;
            }
            if e == crate::stateless::TAIL_UNAVAILABLE {
                // The restored snapshot may contain documents deleted since its
                // checkpoint. A full live-document snapshot cannot remove them:
                // restart on a fresh physical copy and bypass the object store.
                me.retry_peer_recovery(&index, shard, &aid, attempt + 1);
                return;
            }
            if e == PREFIX_REUSE_REJECTED {
                tracing::info!(%index, shard, "replica prefix cannot be reused; falling back to a file-based recovery");
                me.retry_peer_recovery(&index, shard, &aid, attempt);
                return;
            }
            if reusing && !recovery_retryable(&e) && attempt < RECOVERY_MAX_RETRIES {
                // An operations-based reuse that failed (e.g. a deposed
                // primary's ops above its recorded global checkpoint are not
                // in the new primary's history): recover from files in place.
                // Reporting it instead let the master allocate the copy here
                // again, where it offered the same prefix again
                // (`open_demoted` starts a fresh WAL) and failed the same way.
                tracing::info!(%index, shard, "the reused local copy failed its recovery, recovering from files: {e}");
                me.retry_peer_recovery(&index, shard, &aid, attempt + 1);
                return;
            }
            if e == FILE_RECOVERY_UNSUPPORTED {
                tracing::info!(%index, shard, "the recovery source predates file-based recovery; re-indexing from a document snapshot");
                me.retry_peer_recovery_with(&index, shard, &aid, attempt, false);
                return;
            }
            if recovery_retryable(&e) && attempt < RECOVERY_MAX_RETRIES {
                let delay = Duration::from_millis((250u64 << attempt.min(5)).min(5000));
                tracing::warn!(%index, shard, attempt, "recovery failed, retrying in {delay:?}: {e}");
                tokio::time::sleep(delay).await;
                me.retry_peer_recovery(&index, shard, &aid, attempt + 1);
                return;
            }
            tracing::warn!(%index, shard, "recovery failed: {e}");
            me.report_failed(&index, shard, &aid, None, &format!("failed recovery: {e}"), false);
        });
    }

    /// May the local copy catch up from operations instead of files (ES
    /// `isTargetSameHistory` + a retention lease covering it)? It must be
    /// this index incarnation's copy of the shard and share the primary's
    /// history: carry its history uuid (copies record it since file-based
    /// recovery; a copy that left the in-sync set, e.g. failed while its
    /// node restarted, still qualifies, as in ES), or be in the in-sync set.
    /// A deposed primary also needs the global checkpoint it recorded.
    fn local_copy_reusable(&self, state: &ClusterState, index: &str, shard: u32) -> bool {
        let Some(im) = state.metadata.indices.get(index) else { return false };
        let Some(svc) = self.engine_for(shard).and_then(|e| e.get_index(index)) else { return false };
        if svc.is_detached() { return false; }
        let Some(rec) = shards::read_copy_record(svc.path()) else { return false };
        rec.index_uuid == im.uuid && rec.shard == shard && rec.index == index
            && (rec.history_uuid.is_some() || im.in_sync_allocation_ids.get(&shard).is_some_and(|ids| ids.contains(&rec.allocation_id)))
            && (!rec.primary || rec.global_checkpoint.is_some())
    }

    /// Open the local copy, replay its WAL (a deposed primary: seed it from
    /// its engine history above its recorded global checkpoint) and offer
    /// its prefix to the primary. All of it runs on the recovery pool: a
    /// large WAL replay on the cluster applier held up heartbeats until the
    /// node was removed and its copies fully re-recovered (memory review,
    /// 6.35M restart). Falls back to a file recovery when the copy has no
    /// reusable prefix.
    fn prepare_reuse_then_recover(&self, index: &str, shard: u32, aid: &str, attempt: u32, files: bool) {
        let key = format!("reuse:{index}:{shard}:{aid}");
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());
        tokio::spawn(async move {
            let Some(svc) = me.engine_for(shard).and_then(|e| e.get_index(&index)) else {
                me.pending.lock().remove(&key);
                return;
            };
            let (open_svc, open_aid, rec) = (svc.clone(), aid.clone(), shards::read_copy_record(svc.path()));
            let (wanted_index, wanted_aid) = (index.clone(), aid.clone());
            let prepared = Self::open_local_with_backoff(&me, &index, shard, &svc,
                move || release_open_on_failure(&open_svc, shard, || Self::open_for_reuse(&open_svc, shard, &open_aid, rec.clone())),
                move |n: &ClusterNode| n.shards.get(&wanted_index, shard).is_none()
                    && n.my_routing(&n.state(), &wanted_index, shard, &wanted_aid).is_some_and(|r| r.state == ShardState::Initializing)).await;
            let prepared = match prepared {
                Ok(p) => p,
                Err(e) if memory_refusal(&e) => {
                    // Memory, not the copy: keep it. Fail this allocation
                    // (the data stays on disk and is offered again when the
                    // master allocates the shard here again) rather than
                    // set it aside for a full file recovery.
                    let _g = me.apply_lock.lock();
                    me.pending.lock().remove(&key);
                    let state = me.state();
                    if me.shards.get(&index, shard).is_none() && me.my_routing(&state, &index, shard, &aid).is_some_and(|r| r.state == ShardState::Initializing) {
                        tracing::warn!(%index, shard, "the local copy could not be opened for memory; failing the allocation and keeping the copy: {e}");
                        me.report_failed(&index, shard, &aid, None, &format!("failed recovery: opening the local copy was refused for memory: {e}"), false);
                    }
                    return;
                }
                Err(e) => {
                    tracing::info!(%index, shard, "local copy cannot be reused, recovering from files: {e}");
                    None
                }
            };
            let _g = me.apply_lock.lock();
            me.pending.lock().remove(&key);
            let state = me.state();
            let routing = me.my_routing(&state, &index, shard, &aid).filter(|r| r.state == ShardState::Initializing);
            let (Some(routing), true) = (routing, me.shards.get(&index, shard).is_none()) else {
                // The allocation moved on (or another task took the slot):
                // close what was opened; the data stays.
                if prepared.is_some() && me.shards.get(&index, shard).is_none() {
                    let _ = svc.close_shard(shard);
                }
                return;
            };
            let mut admission_error = None;
            let offer = prepared.and_then(|(applier, prefix, rec)| {
                let im = state.metadata.indices.get(&index)?;
                let primary = state.routing_table.shard_table(&index, shard)?.active_primary()?;
                let source_allocation_id = primary.allocation_id()?;
                // Recovery control (ES `StartRecoveryRequest` is not subject
                // to the request breaker): accounted, never refused. A
                // refusal used to return here with the copy initializing and
                // nothing left to recover it until some later state.
                let offer = crate::control_memory::unrefused("peer recovery reuse offer", || RecoveryReuseReq::try_new(&index, shard, &state.metadata.cluster_uuid, &im.uuid,
                    &rec.allocation_id, &aid, &me.local.id, source_allocation_id, im.primary_term(shard), prefix,
                    rec.history_uuid.as_deref())).map_err(|e| admission_error=Some(e)).ok()?;
                Some((applier, offer, im.uuid.clone(), im.primary_term(shard), primary.node_id.clone()?))
            });
            if let Some(error) = admission_error {
                // Not a breaker (forced above): the offer could not be
                // allocated. Do not turn that into the more expensive full
                // snapshot path, and never leave the copy initializing with
                // no recovery: close what was opened (the data stays, and is
                // offered again when the master allocates the shard here
                // again) and fail the allocation.
                tracing::warn!(%index, shard, reason=%error.reason, "recovery reuse offer could not be built; failing the allocation and keeping the copy");
                if me.shards.get(&index, shard).is_none() {
                    let _ = svc.close_shard(shard);
                }
                me.report_failed(&index, shard, &aid, None, &format!("failed recovery: the reuse offer could not be built: {}", error.reason), false);
                return;
            }
            match offer {
                Some((applier, offer, uuid, term, primary_node)) => {
                    me.launch_peer_recovery(&index, shard, &aid, &uuid, term, svc, applier, &primary_node, false, attempt, PeerMode::Ops(Some(offer)));
                }
                None => me.start_peer_recovery_with(&state, &index, shard, &routing, attempt, true, false, files),
            }
        });
    }

    /// Run `open` (a blocking open of the local copy of `shard` on `svc`) on
    /// the recovery pool. When memory refuses it (a breaker, the shared
    /// mutable-vector budget: [`memory_refusal`]), what it opened is closed
    /// again (the data stays), and it is retried with a backoff while
    /// `wanted` holds, the retries serialized on the node's open gate so a
    /// node under memory pressure opens such copies one at a time. A memory
    /// refusal says nothing about the copy itself: the callers keep the copy
    /// rather than replace it from another node.
    async fn open_local_with_backoff<T: Send + 'static>(
        me: &Arc<ClusterNode>, index: &str, shard: u32, svc: &Arc<IndexService>,
        open: impl Fn() -> Result<T, String> + Send + Sync + 'static,
        wanted: impl Fn(&ClusterNode) -> bool + Send,
    ) -> Result<T, String> {
        use elasticsearch_http::pools::{self, Pool};
        let open = Arc::new(open);
        let mut attempt = 0;
        loop {
            let gate = if attempt > 0 { Some(me.local_open_gate.clone().lock_owned().await) } else { None };
            let (f, close_me, close_svc, close_index) = (open.clone(), me.clone(), svc.clone(), index.to_string());
            let r = pools::run(Pool::Recovery, move || {
                let r = f();
                if let Err(e) = &r {
                    // Free what a partial open holds before waiting; never a
                    // copy the shard manager holds (another task's).
                    if memory_refusal(e) && close_me.shards.get(&close_index, shard).is_none() {
                        let _ = close_svc.close_shard(shard);
                    }
                }
                r
            }).await.map_err(|e| e.reason).and_then(|r| r);
            drop(gate);
            match r {
                // A store file that does not fit (`node.store.allow_mmap:
                // false`) is a capacity limit, not passing pressure: fail
                // the copy now and leave retries to the master.
                Err(e) if memory_refusal(&e) && !store_file_refusal(&e) && attempt < LOCAL_OPEN_MEMORY_RETRIES && wanted(me.as_ref()) => {
                    let delay = local_open_backoff(attempt);
                    tracing::warn!(%index, shard, attempt, "opening the local shard copy was refused for memory, retrying in {delay:?} (the copy is kept): {e}");
                    tokio::time::sleep(delay).await;
                    if !wanted(me.as_ref()) {
                        return Err(e);
                    }
                    attempt += 1;
                }
                r => return r,
            }
        }
    }

    /// Blocking part of [`Self::prepare_reuse_then_recover`]: `None` when
    /// the copy offers no prefix (it then recovers from files).
    fn open_for_reuse(svc: &Arc<IndexService>, shard: u32, aid: &str, rec: Option<CopyRecord>) -> Result<Option<(Arc<ReplicaApplier>, ReplicaPrefix, CopyRecord)>, String> {
        let Some(rec) = rec else { return Ok(None) };
        Self::ensure_copy_open(svc, shard, aid)?;
        let applier = if rec.primary {
            // A deposed primary: its own ops above the global checkpoint it
            // recorded are verified against the new primary's.
            let Some(gcp) = rec.global_checkpoint else { return Ok(None) };
            ReplicaApplier::open_demoted(svc.clone(), shard, gcp, rec.primary_term).map_err(|e| e.to_string())?
        } else {
            let applier = ReplicaApplier::open(svc.clone(), shard).map_err(|e| e.to_string())?;
            // A `_flush` / force merge may have committed this copy outside
            // the applier: keep its committed segments reusable.
            applier.adopt_engine_commit().map_err(|e| e.to_string())?;
            applier
        };
        let Some(prefix) = applier.reusable_prefix() else { return Ok(None) };
        // The WAL's term only moves up from the recorded one.
        if prefix.primary_term < rec.primary_term {
            return Ok(None);
        }
        Ok(Some((Arc::new(applier), prefix, rec)))
    }

    /// Restart a failed peer recovery from scratch, provided the routing
    /// table still wants this very copy (same allocation id, still
    /// initializing here). If it does not, the master has already moved on
    /// and there is nothing to retry or report.
    fn retry_peer_recovery(&self, index: &str, shard: u32, aid: &str, attempt: u32) {
        self.retry_peer_recovery_with(index, shard, aid, attempt, self.file_based_recovery());
    }

    /// [`Self::retry_peer_recovery`]; `files == false` re-indexes from a
    /// document snapshot.
    fn retry_peer_recovery_with(&self, index: &str, shard: u32, aid: &str, attempt: u32, files: bool) {
        let _g = self.apply_lock.lock();
        let state = self.state();
        let Some(local) = self.shards.get(index, shard) else { return };
        if local.lock().allocation_id != aid {
            return;
        }
        let me = self.local.id.as_str();
        let routing = state.routing_table.shard_table(index, shard).and_then(|t| t.copies.iter().find(|c| c.allocation_id() == Some(aid) && c.node_id.as_deref() == Some(me) && c.state == ShardState::Initializing).cloned());
        let Some(routing) = routing else { return };
        let old_copy = {
            let local = local.lock();
            match &local.role { Role::Primary(p) => p.copy(), Role::Replica(r) => r.copy() }
        };
        self.remove_copy(index, shard);
        // Same routing allocation may retry only with a newly detached physical
        // copy. Full-open services that retained their copy cannot clear the
        // admission tombstone and accidentally resume it without its lease.
        if !old_copy.is_closed() || self.shards.get(index, shard).is_some() {
            self.report_failed(index, shard, aid, None, "recovery retry could not fence its previous physical copy", false);
            return;
        }
        if let Err(error) = self.capacity_leases.lock().retire_detached_attempt(index, shard, aid) {
            self.report_failed(index, shard, aid, None, error, false);
            return;
        }
        self.start_peer_recovery_with(&state, index, shard, &routing, attempt, false, false, files);
    }

    /// The routing entry of copy `aid` on this node, if still current.
    fn my_routing(&self, state: &ClusterState, index: &str, shard: u32, aid: &str) -> Option<ShardRouting> {
        let me = self.local.id.as_str();
        state.routing_table.shard_table(index, shard).and_then(|t| t.copies.iter().find(|c| c.allocation_id() == Some(aid) && c.node_id.as_deref() == Some(me) && c.state != ShardState::Unassigned).cloned())
    }

    /// Ask the recovery source whether it really runs the primary; start the
    /// (wiping) recovery only once it says yes. Gives up quietly when the
    /// routing table no longer wants this copy, and fails the copy (keeping
    /// the data) when the source never comes up.
    fn probe_then_recover(&self, index: &str, shard: u32, aid: &str, attempt: u32, allow_reuse: bool) {
        let key = format!("probe:{index}:{shard}:{aid}");
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());
        tokio::spawn(async move {
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                let state = me.state();
                let Some(routing) = me.my_routing(&state, &index, shard, &aid).filter(|r| r.state == ShardState::Initializing) else { break };
                let primary = state.routing_table.shard_table(&index, shard).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone());
                let live = match primary.as_deref().and_then(|n| me.address_of(n)) {
                    Some(addr) => me.transport.send_request::<_, bool>(addr, ACTION_PRIMARY_PROBE, &(&index, shard), Duration::from_secs(10)).await.unwrap_or(false),
                    None => false,
                };
                if live {
                    let _g = me.apply_lock.lock();
                    let state = me.state();
                    if me.shards.get(&index, shard).is_none() && me.my_routing(&state, &index, shard, &aid).map(|r| r.state == ShardState::Initializing).unwrap_or(false) {
                        me.start_peer_recovery_mode(&state, &index, shard, &routing, attempt, true, allow_reuse);
                    }
                    break;
                }
                if Instant::now() > deadline {
                    tracing::warn!(%index, shard, "recovery source never confirmed an active primary; failing the copy (local data kept)");
                    me.report_failed(&index, shard, &aid, None, "recovery source is not an active primary", false);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            me.pending.lock().remove(&key);
        });
    }

    /// Promote a replica copy. A failover (higher term) returns the resync
    /// for the other copies; a relocation handoff (same term) has nothing
    /// to resync because the source shipped everything before handing off.
    /// `promote_to_primary` must run exactly once: it empties the WAL. One
    /// that adopted `term` but did not finish (refused or failed after
    /// saving the term, or a crash) left a durable marker and runs again.
    fn promote(&self, index: &str, shard: u32, aid: &str, term: u64, svc: &Arc<IndexService>, r: &ReplicaApplier) -> Result<(Arc<PrimaryCtx>, Option<crate::replication::replica::PromotionResync>), String> {
        let owed = term > r.primary_term() || r.promotion_pending() == Some(term);
        let resync = if owed { Some(r.promote_to_primary(term).map_err(|e| e.to_string())?) } else { None };
        let p = PrimaryCtx::new(index, shard, aid, term, svc.clone()).map_err(|e| e.to_string())?;
        let p = Arc::new(p);
        if let Some(im) = self.state().metadata.indices.get(index) {
            self.attach_object_store(&p, &im.uuid);
            // A promoted replica continues the history it recovered into.
            let history_uuid = shards::primary_history_uuid(svc, &im.uuid, shard, true);
            let checkpoint = shards::primary_checkpoint(svc, &im.uuid, shard, &history_uuid, r.global_checkpoint());
            let _ = shards::write_copy_record(svc, &CopyRecord { index: index.into(), index_uuid: im.uuid.clone(), shard, allocation_id: aid.into(), primary_term: term, primary: true,
                history_uuid: Some(history_uuid), global_checkpoint: Some(checkpoint) });
        }
        Ok((p, resync))
    }

    /// After a promotion, send every op above the global checkpoint to the
    /// other copies; any that diverged report `NeedsFullRecovery` and are
    /// failed (stale) so they get re-recovered.
    ///
    /// The resync is never skipped for memory (ES's `PrimaryReplicaSyncer`
    /// is not subject to the request breaker): its target list and requests
    /// are accounted but not refused ([`crate::control_memory::unrefused`];
    /// bounded by the copies and by the ops the promotion already holds,
    /// one request at a time). A skipped resync left diverged copies
    /// serving, never failed or re-recovered. A copy whose request still
    /// cannot be built, or that the request does not reach, is failed.
    async fn resync_replicas(self: Arc<Self>, p: Arc<PrimaryCtx>, resync: crate::replication::replica::PromotionResync) {
        let targets = match crate::control_memory::unrefused("primary-replica resync", || p.targets.lock().try_snapshot()) {
            Ok(targets) => targets,
            Err(error) => {
                // Not a breaker (forced above): an allocation failure. Fail
                // every other copy rather than leave one unsynced.
                tracing::error!(index = %p.index, shard = p.shard, %error, "primary-replica resync could not list its targets; failing the other copies");
                let ids: Vec<String> = p.targets.lock().keys().filter(|aid| **aid != p.allocation_id).cloned().collect();
                for aid in ids {
                    self.report_failed(&p.index, p.shard, &aid, Some(p.term), &format!("primary-replica resync failed: {error}"), true);
                }
                return;
            }
        };
        for (aid, t) in targets.iter() {
            if aid == &p.allocation_id {
                continue;
            }
            let req = match crate::control_memory::unrefused("primary-replica resync", || resync.to_request(&p.index, p.shard, aid)) {
                Ok(req) => req,
                Err(error) => {
                    tracing::warn!(index = %p.index, copy = %aid, "resync request could not be built: {error}");
                    self.report_failed(&p.index, p.shard, aid, Some(p.term), &format!("primary-replica resync failed: {error}"), true);
                    continue;
                }
            };
            let Ok(addr) = t.address.parse::<SocketAddr>() else {
                tracing::warn!(index = %p.index, copy = %aid, address = %t.address, "resync target has no valid address");
                self.report_failed(&p.index, p.shard, aid, Some(p.term), "primary-replica resync failed: invalid target address", true);
                continue;
            };
            let res: Result<ReplicationResponse, _> = self.transport.send_request(addr, shards::ACTION_RESYNC, &req, Duration::from_secs(60)).await;
            match res {
                Ok(resp) => {
                    p.tracker.lock().update_local_checkpoint(&aid, resp.local_checkpoint);
                }
                Err(e) => {
                    tracing::warn!(index = %p.index, copy = %aid, "resync failed: {e}");
                    self.report_failed(&p.index, p.shard, &aid, Some(p.term), &format!("primary-replica resync failed: {e}"), true);
                }
            }
        }
    }

    fn bound_recovery_primary(&self, index: &str, shard: u32, allocation_id: &str, term: u64) -> Result<Arc<PrimaryCtx>, RemoteError> {
        let p = self.shards.primary(index, shard).ok_or_else(|| remote("recovery_failed_exception", "recovery primary moved", 409))?;
        let state = self.state();
        let current = state.master_node_id.is_some() && state.metadata.indices.get(index).is_some_and(|im| im.primary_term(shard) == term)
            && state.routing_table.shard_table(index, shard).and_then(|t| t.active_primary()).is_some_and(|s|
                s.allocation_id() == Some(allocation_id) && s.node_id.as_deref() == Some(self.local.id.as_str()));
        if !current || p.allocation_id != allocation_id || p.term != term || p.relocated.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(remote("recovery_failed_exception", "recovery primary allocation or term changed", 409));
        }
        Ok(p)
    }

    pub(crate) fn prepare_pressure_recovery(&self,request:&crate::capacity::pressure::Grant)->Result<(),String>{
        let _apply=self.apply_lock.lock();
        self.capacity_leases.lock().prepare_pressure(self,&self.state(),request,Instant::now())
    }

    fn admit_recovery_memory(&self,index:&str,shard:u32,aid:&str,uuid:&str,term:u64,primary_node:&str,source_generation:u64,slot:&crate::capacity::lease::TaskSlot)->Result<elasticsearch_engine::rustic_core::admission::Context,String> {
        if slot.lock().is_some(){return Err("recovery task already negotiated memory".into());}
        let text=self.local.id.len()+self.local.ephemeral_id.len()+index.len()+uuid.len()+aid.len();
        let _identity_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(text as u64,"recovery negotiation identity").map_err(|e|e.to_string())?;
        let identity=crate::capacity::lease::Identity{node_id:self.local.id.clone(),ephemeral_id:self.local.ephemeral_id.clone(),index:index.into(),index_uuid:uuid.into(),shard,allocation_id:aid.into(),primary_term:term};
        let _apply=self.apply_lock.lock();
        let envelope=self.capacity_leases.lock().grant_envelope(self,&self.state(),&identity,source_generation,primary_node,Instant::now())?;
        let context=elasticsearch_engine::rustic_core::admission::Context::with_lease(envelope.lease.clone());
        *slot.lock()=Some(envelope);
        Ok(context)
    }

    #[allow(clippy::too_many_arguments)]
    async fn recover_replica(&self, index: &str, shard: u32, aid: &str, uuid: &str, term: u64, svc: Arc<IndexService>, applier: Arc<ReplicaApplier>, primary_node: &str, slot: Arc<Mutex<Option<RecoveryState>>>, relocation: bool, reuse: Option<RecoveryReuseReq>, envelope:&crate::capacity::lease::TaskSlot) -> Result<Arc<elasticsearch_engine::index::IndexShard>, String> {
        let addr = self.address_of(primary_node).ok_or("primary node not in cluster state")?;
        let source_binding = reuse.as_ref().map(|r| (r.source_allocation_id.clone(), r.primary_term));
        // An operations-based reuse keeps this copy's history identity.
        let history_uuid = reuse.as_ref().and_then(|r| r.history_uuid.clone());
        let start: RecoveryStartResp = if let Some(offer) = reuse {
            let accepted: Option<RecoveryStartResp> = match self.transport.send_request(addr, shards::ACTION_RECOVERY_REUSE, &offer, Duration::from_secs(60)).await {
                Ok(response) => response,
                Err(e) => {
                    // Unknown action on an older peer, a lost response, or
                    // failed negotiation cannot authorize retaining a copy.
                    // The normal fallback re-probes authority before wiping.
                    tracing::info!(index, shard, "replica-prefix negotiation failed; using snapshot fallback: {e}");
                    return Err(PREFIX_REUSE_REJECTED.into());
                }
            };
            let Some(start) = accepted else { return Err(PREFIX_REUSE_REJECTED.into()) };
            if start.info.primary_term != term || start.info.snapshot_docs != 0
                || start.info.snapshot_max_seq_no != offer.prefix.checkpoint || start.info.start_seq_no != offer.prefix.checkpoint + 1
            { return Err("invalid replica-prefix recovery response".into()); }
            tracing::info!(index, shard, checkpoint = offer.prefix.checkpoint, "reusing the local copy: operations-based recovery above its checkpoint");
            start
        } else {
            self.transport.send_request(addr, shards::ACTION_RECOVERY_START, &RecoveryStartReq::try_new(index, shard, aid, None).map_err(|e| e.to_string())?, Duration::from_secs(60))
                .await.map_err(|e| format!("recovery source start [{index}][{shard}]: {e}"))?
        };
        if start.info.primary_term!=term{return Err("recovery source term changed".into());}
        let context=self.admit_recovery_memory(index,shard,aid,uuid,term,primary_node,start.recovery_id,envelope)?;
        context.scope_future(self.finish_peer_recovery(index, shard, aid, uuid, term, svc, applier, addr, slot, relocation, start, source_binding, history_uuid, None)).await
    }

    /// Phase 2 of a peer recovery (the translog phase from `start`, through
    /// the bound reuse actions when `source_binding` is set), then mark the
    /// copy in sync on the primary, persist its copy record and report it
    /// started. `files`: the file phase that preceded it.
    #[allow(clippy::too_many_arguments)]
    async fn finish_peer_recovery(&self, index: &str, shard: u32, aid: &str, uuid: &str, term: u64, svc: Arc<IndexService>, applier: Arc<ReplicaApplier>, addr: SocketAddr,
        slot: Arc<Mutex<Option<RecoveryState>>>, relocation: bool, start: RecoveryStartResp, source_binding: Option<(String, u64)>, history_uuid: Option<String>,
        files: Option<crate::replication::recovery::FileRecoveryStats>) -> Result<Arc<elasticsearch_engine::index::IndexShard>, String>
    {
        let target_copy = applier.copy();
        let recovery_id = start.recovery_id;
        let me_node = self.local.id.clone();
        let me_addr = self.local.transport_address.clone();
        let register = |a: Arc<ReplicaApplier>| {
            self.recovery_current(index, shard, aid, &svc, &target_copy)
                && self.shards.register_recovery_applier(index, shard, aid, &svc, a)
        };
        let transport = self.transport.clone();
        let pull = || {
            let transport = transport.clone();
            let req = RecoveryBatchReq::try_new(index, shard, recovery_id, aid, &me_node, &me_addr);
            let binding = &source_binding;
            async move {
                let req=req.map_err(|e| e.to_string())?;
                let response = if let Some((source_allocation_id, primary_term)) = binding {
                    transport.send_request::<_, RecoveryBatch>(addr, shards::ACTION_RECOVERY_REUSE_BATCH,
                        &BoundRecoveryReq::try_new(source_allocation_id, *primary_term, req).map_err(|e|e.to_string())?, Duration::from_secs(120)).await
                } else {
                    transport.send_request::<_, RecoveryBatch>(addr, shards::ACTION_RECOVERY_BATCH, &req, Duration::from_secs(120)).await
                };
                response.map_err(|e| format!("recovery source batch: {e}"))
            }
        };
        // The node recovered from: the one whose transport address is `addr`.
        let source_node = self.state().nodes.values().find(|n| n.transport_address.parse::<SocketAddr>().ok() == Some(addr)).map(|n| crate::replication::recovery::RecoveryText::try_new(&n.id)).transpose().map_err(|e|e.to_string())?;
        let nodes = source_node.map(|source|crate::replication::recovery::RecoveryText::try_new(&self.local.id).map(|target|(source,target))).transpose().map_err(|e|e.to_string())?;
        let (applier, cp) = shards::run_recovery_target_with(applier, start.info, files, nodes, slot, &register, pull).await
            .map_err(|e| format!("recovery target [{index}][{shard}] at [{}]: {e}", shards::shard_path(&svc, shard).display()))?;
        if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
            return Err(STALE_RECOVERY.into());
        }
        if !register(applier.clone()) { return Err(STALE_RECOVERY.into()); }
        // Mark in sync on the primary (retry while the global checkpoint
        // is ahead of us; live ops keep flowing meanwhile).
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
                return Err(STALE_RECOVERY.into());
            }
            let req = RecoveryFinalizeReq::try_new(index, shard, recovery_id, aid, applier.local_checkpoint().max(cp), relocation).map_err(|e|e.to_string())?;
            let ok: bool = if let Some((source_allocation_id, primary_term)) = &source_binding {
                self.transport.send_request(addr, shards::ACTION_RECOVERY_REUSE_FINALIZE,
                    &BoundRecoveryReq::try_new(source_allocation_id, *primary_term, req).map_err(|e|e.to_string())?, Duration::from_secs(60)).await
            } else {
                self.transport.send_request(addr, shards::ACTION_RECOVERY_FINALIZE, &req, Duration::from_secs(60)).await
            }.map_err(|e| e.to_string())?;
            if ok {
                break;
            }
            if Instant::now() > deadline {
                return Err("timed out waiting to be marked in sync".into());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // The copy may have been removed (and its directory rebuilt for
        // another allocation) while this task ran: never stamp or start it.
        tokio::task::block_in_place(|| {
            let _permit = target_copy.operation_permit().map_err(|e| e.to_string())?;
            if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
                return Err(STALE_RECOVERY.into());
            }
            // Disk persistence and refresh can wait on merges or publish a large
            // recovered reader. Let Tokio replace this worker so coordination
            // and transport heartbeats continue while the operation permit fences
            // this exact copy against replacement.
            shards::write_copy_record(&svc, &CopyRecord { index: index.into(), index_uuid: uuid.into(), shard, allocation_id: aid.into(), primary_term: term, primary: false,
                history_uuid: history_uuid.clone(), global_checkpoint: None })
                .map_err(|e| format!("persisting recovered copy failed: {e}"))?;
            // A copy cannot become searchable/STARTED after a failed refresh.
            applier.copy().shard.refresh().map_err(|e| format!("refresh after recovery failed: {e}"))?;
            // Return this task's physical copy, never resolve its replacement
            // later by allocation id when admission completion runs.
            Ok(target_copy.clone())
        })
    }

    /// Free bytes a recovery must leave on the filesystem of `path`: the
    /// high disk watermark's (0 with the disk threshold decider disabled).
    pub(crate) fn disk_reserve(&self, path: &std::path::Path) -> u64 {
        use elasticsearch_cluster::allocation::settings::{AllocSettings, SettingsView};
        let state = self.state();
        let node: serde_json::Map<String, serde_json::Value> = self.config.settings.iter()
            .filter(|(k, _)| k.starts_with("cluster.routing.allocation.disk.")).map(|(k, v)| (k.clone(), v.clone())).collect();
        let s = AllocSettings::resolve(&SettingsView { metadata: &state.metadata, node: &node });
        if !s.disk_threshold_enabled {
            return 0;
        }
        s.high.required_free(crate::stats::disk_usage(path).total_bytes).0
    }

    /// File-based recovery of a replica or relocation target (ES
    /// `PeerRecoveryTargetService` + `RecoveryTarget`, see
    /// `replication::file_recovery`): the primary pins its commit; files
    /// identical in the copy's old store are reused, files an interrupted
    /// attempt received are kept or resumed, the rest is fetched in chunks
    /// (`indices.recovery.chunk_size`, `max_concurrent_file_chunks` in
    /// flight, paced to `max_bytes_per_sec` on the source), each
    /// verified before it is renamed into place. Then the received copy
    /// replaces `placeholder` and only the operations above the commit are
    /// replayed. Every chunk is fenced: the source checks its allocation id
    /// and term, this side that the copy is still the current allocation
    /// under the same term.
    #[allow(clippy::too_many_arguments)]
    async fn recover_replica_files(&self, index: &str, shard: u32, aid: &str, uuid: &str, term: u64, svc: Arc<IndexService>, placeholder: Arc<ReplicaApplier>, primary_node: &str, slot: Arc<Mutex<Option<RecoveryState>>>, relocation: bool, envelope:&crate::capacity::lease::TaskSlot) -> Result<Arc<elasticsearch_engine::index::IndexShard>, String> {
        use crate::replication::file_recovery::{self as fr, FileAction, FileRecoveryStartReq, FileRecoveryStartResp, PartWriter, Progress};
        use elasticsearch_http::pools::{self, Pool};
        let started = Instant::now();
        let placeholder_copy = placeholder.copy();
        let addr = self.address_of(primary_node).ok_or("primary node not in cluster state")?;
        let (staging, reuse) = (fr::admitted_recovery_dir(&svc, shard, false)?, fr::admitted_recovery_dir(&svc, shard, true)?);
        // A superseded attempt of this shard stops at its next chunk.
        let _staging_lock = fr::StagingLock::try_acquire(&staging)?.ok_or(fr::STAGING_BUSY)?;
        let req = FileRecoveryStartReq::try_new(index, shard, aid, &self.local.id).map_err(|e|e.to_string())?;
        // The source flushes and checksums its commit first (minutes for a
        // large never-recovered shard; cached afterwards).
        let start: FileRecoveryStartResp = match self.transport.send_request(addr, fr::ACTION_FILE_RECOVERY_START, &req, Duration::from_secs(1800)).await {
            Ok(start) => start,
            Err(e) if e.to_string().contains("No handler for action") => return Err(FILE_RECOVERY_UNSUPPORTED.into()),
            Err(e) => return Err(format!("file recovery source start [{index}][{shard}]: {e}")),
        };
        if start.primary_term != term || start.info.primary_term != term || start.info.snapshot_docs != 0 || start.info.start_seq_no != start.manifest.local_checkpoint + 1 {
            return Err("invalid file recovery response".into());
        }
        let context=self.admit_recovery_memory(index,shard,aid,uuid,term,primary_node,start.recovery_id,envelope)?;
        context.scope_future(async {
        let settings = self.recovery_settings();
        let reserve = self.disk_reserve(svc.path());
        // Diff, reuse, resume and the disk check read files: off the runtime.
        let plan_path_bytes=staging.as_os_str().as_encoded_bytes().len().checked_add(reuse.as_os_str().as_encoded_bytes().len()).ok_or("recovery plan paths overflow")?;
        let plan_path_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(plan_path_bytes as u64,"recovery plan worker paths").map_err(|e|e.to_string())?;
        let (manifest, plan_staging, plan_reuse) = (start.manifest.clone(), staging.clone(), reuse.clone());
        // The bytes to fetch stay reserved (node-wide, per filesystem) until
        // this attempt ends: concurrent recoveries each count the others'.
        let (mut plan, local_tails) = pools::run(Pool::Recovery, move || -> Result<(fr::Plan, Vec<(usize, Option<Vec<u8>>)>), String> {
            let _memory=plan_path_memory;
            let (manifest,plan_staging,plan_reuse)=(manifest,plan_staging,plan_reuse);
            let plan = fr::plan(&plan_staging, &plan_reuse, &manifest)?;
            // The tails of the files the plan would take locally (see
            // `fr::TAIL_CHECK_BYTES`): name, length and whole-file CRC-32
            // do not identify a self-checksummed file.
            let mut tails = Vec::new();
            for (i, f) in plan.files.iter().enumerate() {
                if let Some(path) = fr::local_candidate(&plan_staging, f)? {
                    tails.push((i, fr::local_tail(&path, f.entry.len)?));
                }
            }
            Ok((plan, tails))
        }).await.map_err(|e| e.reason)??;
        // Ask the source for the same tails and fetch every file whose tail
        // differs (a stale copy's file under the source's name and length).
        {
            use futures::StreamExt;
            let _tail_memory=elasticsearch_engine::rustic_core::admission::shared()
                .reserve((local_tails.len() as u64).saturating_mul(fr::TAIL_CHECK_BYTES.saturating_mul(2).saturating_add(512)),"recovery reuse tail checks").map_err(|e|e.to_string())?;
            let transport = self.transport.clone();
            let checks = local_tails.into_iter().map(|(i, local)| {
                let entry = plan.files[i].entry.clone();
                let transport = transport.clone();
                let (source_aid, recovery_id) = (start.source_allocation_id.clone(), start.recovery_id);
                async move {
                    let Some(local) = local else { return Ok::<_, String>((i, false)) };
                    let (offset, len) = fr::tail_range(entry.len);
                    let req = fr::FileChunkReq::try_new(index, shard, recovery_id, aid, &source_aid, term, &entry.path, offset, len).map_err(|e| e.to_string())?;
                    let resp = fetch_chunk(transport, addr, req).await?;
                    Ok((i, &*resp.data == local.as_slice()))
                }
            });
            let mut checked = futures::stream::iter(checks).buffer_unordered(settings.max_concurrent_file_chunks.max(1));
            let mut differ = Vec::new();
            while let Some(result) = checked.next().await {
                let (i, same) = result?;
                if !same { differ.push(i); }
            }
            drop(checked);
            let demoted = differ.len();
            for i in differ { fr::demote_to_fetch(&mut plan.files[i]); }
            if demoted > 0 {
                tracing::info!(index, shard, files = demoted, "old-store or staged files differ from the source's despite name, length and checksum: fetching them");
            }
        }
        let (plan_staging, plan_reserve) = (staging.clone(), reserve);
        let (plan, reserved) = pools::run(Pool::Recovery, move || -> Result<(fr::Plan, fr::DiskReservation), String> {
            let reserved = fr::reserve_disk(&plan_staging, plan.bytes_to_fetch(), plan_reserve)?;
            let mut progress = Progress::load(&plan_staging)?;
            for f in &plan.files {
                if let FileAction::Reuse(src) = &f.action {
                    fr::link_reused(&plan_staging, &f.entry, src, &mut progress)?;
                }
            }
            Ok((plan, reserved))
        }).await.map_err(|e| e.reason)??;
        let mut stats = plan.stats()?;
        let source_node=Some(crate::replication::recovery::RecoveryText::try_new(primary_node).map_err(|e|e.to_string())?);
        let target_node=Some(crate::replication::recovery::RecoveryText::try_new(&self.local.id).map_err(|e|e.to_string())?);
        let telemetry=RecoveryState::file_phase(&start.info,stats.clone(),source_node,target_node).map_err(|e|e.to_string())?;
        let publish = |stats: &crate::replication::recovery::FileRecoveryStats| {
            let mut s = stats.clone();
            s.time_millis = started.elapsed().as_millis() as u64;
            let mut state=telemetry.clone();
            state.total_time_millis=s.time_millis;state.index_time_millis=s.time_millis;state.files=s;
            *slot.lock() = Some(state);
        };
        publish(&stats);
        tracing::info!(index, shard, files = stats.files_total, reused = stats.files_reused, reused_bytes = stats.bytes_reused, fetch_bytes = plan.bytes_to_fetch(), "file-based recovery started");
        let current = || self.recovery_current(index, shard, aid, &svc, &placeholder_copy)
            && self.state().metadata.indices.get(index).is_some_and(|im| im.primary_term(shard) == term);
        // The progress map, loaded once and saved in batches (see
        // `ProgressBatch`); held by this loop between files.
        let batch_staging = staging.clone();
        let mut batch = Some(pools::run(Pool::Recovery, move || fr::ProgressBatch::load(&batch_staging)).await.map_err(|e| e.reason)??);
        // The files received before an error (an interrupted transfer, a
        // source that moved) are saved below either way, so the retry
        // resumes after them instead of fetching them again.
        let files_received: Result<(), String> = async {
        for (i, f) in plan.files.iter().enumerate() {
            let mut resume = match f.action {
                FileAction::Fetch => None,
                FileAction::Resume { done, running } => Some((done, running)),
                FileAction::Staged | FileAction::Reuse(_) => continue,
            };
            // A file whose whole-file checksum fails is fetched once more
            // from zero (ES fails the recovery; the retry then re-sends it).
            for attempt in 0.. {
                let open_bytes=staging.as_os_str().as_encoded_bytes().len().checked_add(f.entry.path.len()).ok_or("recovery file-open metadata overflow")?;
                let open_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(open_bytes as u64,"recovery file-open worker metadata").map_err(|e|e.to_string())?;
                let (open_staging, entry) = (staging.clone(), f.entry.clone());
                let opened = pools::run(Pool::Recovery, move || {
                    let _memory=open_memory;let (open_staging,entry)=(open_staging,entry);
                    PartWriter::open(&open_staging,&entry,resume)
                }).await.map_err(|e| e.reason)?;
                let writer = match opened {
                    Ok(writer) => writer,
                    // The resumed prefix changed on disk: it was discarded.
                    Err(e) if e.starts_with(fr::FILE_CORRUPT) && attempt == 0 => {
                        tracing::warn!(index, shard, file = %f.entry.path, "{e}; fetching it again");
                        stats.bytes_recovered = stats.bytes_recovered.saturating_sub(stats.details[i].recovered);
                        stats.details.set_recovered(i, 0)?;
                        resume = None;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                stats.details.set_recovered(i, writer.done())?;
                let base = FileChunkBase { index, shard, recovery_id: start.recovery_id, aid, source_allocation_id: &start.source_allocation_id, term, addr };
                match self.fetch_file(&base, writer, &settings, &staging, &reserved, &mut stats, i, &current, &publish).await {
                    Ok((entry, dir)) => {
                        let (mut b, record_staging) = (batch.take().expect("the progress batch is held between files"), staging.clone());
                        let (b, recorded) = pools::run(Pool::Recovery, move || { let r = b.record(&record_staging, &entry, dir); (b, r) }).await.map_err(|e| e.reason)?;
                        batch = Some(b);
                        recorded?;
                        break;
                    }
                    Err(e) if e.starts_with(fr::FILE_CORRUPT) && attempt == 0 => {
                        tracing::warn!(index, shard, file = %f.entry.path, "{e}; fetching it again");
                        stats.bytes_recovered = stats.bytes_recovered.saturating_sub(stats.details[i].recovered);
                        stats.details.set_recovered(i, 0)?;
                        resume = None;
                    }
                    Err(e) => return Err(e),
                }
            }
            stats.files_recovered += 1;
            publish(&stats);
        }
        Ok(())
        }.await;
        if let Some(mut b) = batch.take().filter(|b| b.pending() > 0) {
            let save_staging = staging.clone();
            let saved = pools::run(Pool::Recovery, move || b.save(&save_staging)).await.map_err(|e| e.reason).and_then(|r| r);
            match (&files_received, saved) {
                (Ok(()), saved) => saved?,
                (Err(_), Err(e)) => tracing::warn!(index, shard, "saving the recovery progress after a failed transfer failed (its files are fetched again): {e}"),
                (Err(_), Ok(())) => {}
            }
        }
        files_received?;
        stats.time_millis = started.elapsed().as_millis() as u64;
        tracing::info!(index, shard, millis = stats.time_millis, bytes = stats.bytes_recovered, source_throttle_millis = stats.source_throttle_millis, target_throttle_millis = stats.target_throttle_millis, "recovery files received");
        // Install the received copy in place of the placeholder: opened off
        // the apply lock, retried (the files are kept) and then reported.
        let applier = self.install_recovered_files(index, shard, aid, &svc, &staging, &start.manifest, &placeholder_copy).await?;
        drop(reserved);
        // Hard-linked into the shard directory: the staged files are spare.
        let spare = staging.clone();
        let _ = pools::run(Pool::Recovery, move || std::fs::remove_dir_all(&spare)).await;
        let info = start.info.clone();
        let resp = RecoveryStartResp { recovery_id: start.recovery_id, info };
        let history_uuid = start.history_uuid.clone();
        let finished_copy=self.finish_peer_recovery(index, shard, aid, uuid, term, svc.clone(), applier, addr, slot.clone(), relocation, resp, None, history_uuid, Some(stats)).await?;
        // The old store's reusable files are hard links in the new one now.
        let old = reuse.clone();
        let _ = pools::run(Pool::Recovery, move || std::fs::remove_dir_all(&old)).await;
        Ok(finished_copy)
        }).await
    }

    /// Replace the placeholder of a file recovery with the received copy
    /// (see `replication::file_recovery`, step 4): the node's apply lock is
    /// held only to retire the placeholder and to publish the opened copy,
    /// never while it opens (its graph checkpoint reconciliation can take
    /// long, and the cluster applier and its heartbeats wait for that lock).
    /// An open that fails is retried [`INSTALL_ATTEMPTS`](crate::replication::file_recovery::INSTALL_ATTEMPTS)
    /// times with a backoff, the received files kept; then the error
    /// (prefixed `INSTALL_FAILED`) fails the copy. `STALE_RECOVERY` when
    /// the slot no longer belongs to this recovery.
    #[allow(clippy::too_many_arguments)]
    async fn install_recovered_files(&self, index: &str, shard: u32, aid: &str, svc: &Arc<IndexService>, staging: &std::path::Path,
        manifest: &elasticsearch_engine::hooks::IndexManifest, placeholder: &Arc<elasticsearch_engine::index::IndexShard>) -> Result<Arc<ReplicaApplier>, String>
    {
        use crate::replication::file_recovery as fr;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let e = match self.try_install_recovered_files(index, shard, aid, svc, staging, manifest, placeholder).await {
                Ok(applier) => return Ok(applier),
                Err(e) if e == STALE_RECOVERY => return Err(e),
                Err(e) => e,
            };
            if attempt >= fr::INSTALL_ATTEMPTS {
                let e = if e.starts_with(fr::INSTALL_FAILED) { e } else { format!("{}: {e}", fr::INSTALL_FAILED) };
                return Err(format!("{e} (after {attempt} attempts)"));
            }
            let delay = Duration::from_millis(500u64 << attempt.min(4));
            tracing::warn!(index, shard, attempt, "installing the recovered copy failed, retrying in {delay:?}: {e}");
            tokio::time::sleep(delay).await;
        }
    }

    /// One attempt of [`Self::install_recovered_files`].
    #[allow(clippy::too_many_arguments)]
    async fn try_install_recovered_files(&self, index: &str, shard: u32, aid: &str, svc: &Arc<IndexService>, staging: &std::path::Path,
        manifest: &elasticsearch_engine::hooks::IndexManifest, placeholder: &Arc<elasticsearch_engine::index::IndexShard>) -> Result<Arc<ReplicaApplier>, String>
    {
        use crate::replication::file_recovery as fr;
        use elasticsearch_http::pools::{self, Pool};
        // The three worker submissions may outlive this future when canceled.
        // Retain their path/identity-copy allowance inside each actual worker.
        let manifest_bytes=(manifest.files.len() as u64)
            .checked_mul(std::mem::size_of::<elasticsearch_engine::hooks::files::IndexFileEntry>() as u64)
            .and_then(|bytes|bytes.checked_add(manifest.index.len() as u64))
            .and_then(|bytes|bytes.checked_add(manifest.uuid.len() as u64))
            .and_then(|bytes|manifest.files.iter().try_fold(bytes,|total,file|total.checked_add(file.path.len() as u64)))
            .and_then(|bytes|bytes.checked_mul(2)).ok_or("recovery manifest copy size overflow")?;
        let metadata_bytes=(index.len() as u64).checked_add(aid.len() as u64)
            .and_then(|bytes|bytes.checked_mul(3))
            .and_then(|bytes|bytes.checked_add((staging.as_os_str().as_encoded_bytes().len() as u64).saturating_mul(2)))
            .and_then(|bytes|bytes.checked_add(manifest_bytes))
            .and_then(|bytes|bytes.checked_add(512)).ok_or("recovery installation metadata overflow")?;
        let metadata_memory=Arc::new(elasticsearch_engine::rustic_core::admission::shared()
            .reserve(metadata_bytes,"recovery installation worker metadata").map_err(|error|error.to_string())?);
        let (me, index, aid, svc, placeholder) = (self.me(), index.to_string(), aid.to_string(), svc.clone(), placeholder.clone());
        // Validate compatibility before removing even the placeholder store.
        let (check_staging, check_manifest) = (staging.to_path_buf(), manifest.clone());
        let validation_memory=metadata_memory.clone();
        pools::run(Pool::Recovery, move || {let _memory=validation_memory;let (check_staging,check_manifest)=(check_staging,check_manifest);fr::validate_staged_formats(&check_staging, &check_manifest)}).await.map_err(|e| e.reason)??;
        // 1. Apply lock, short: still this recovery's slot; the placeholder
        // and its empty store go.
        let (me1, index1, aid1, svc1, placeholder1) = (me.clone(), index.clone(), aid.clone(), svc.clone(), placeholder.clone());
        let release_memory=metadata_memory.clone();
        pools::run(Pool::Recovery, move || -> Result<(), String> {
            let _memory=release_memory;
            let (me1,index1,aid1,svc1,placeholder1)=(me1,index1,aid1,svc1,placeholder1);
            let _g = me1.apply_lock.lock();
            if !me1.recovery_slot_held(&index1, shard, &aid1, &svc1, &placeholder1) {
                return Err(STALE_RECOVERY.into());
            }
            fr::release_placeholder(&svc1, shard, &placeholder1)
        }).await.map_err(|e| e.reason)??;
        // The index takes the field ids the received files were written
        // with: this node's shell numbers fields as its mapping arrived here
        // (built later from the final mapping, or in other steps), which
        // need not be how the primary's did.
        let (ids_svc, ids_staging) = (svc.clone(), staging.to_path_buf());
        let ids_memory=metadata_memory.clone();
        pools::run(Pool::Recovery, move || -> Result<(), String> {
            let _memory=ids_memory;
            let ids = fr::staged_core_ids(&ids_staging)?;
            ids_svc.adopt_core_ids(&ids).map(|_| ()).map_err(|e| format!("cannot take the recovered files' field ids: [{}] {}", e.kind, e.reason))
        }).await.map_err(|e| e.reason)??;
        // 2. No lock: open the received copy, unpublished.
        let (svc2, aid2, staging2, manifest2) = (svc.clone(), aid.clone(), staging.to_path_buf(), manifest.clone());
        let open_memory=metadata_memory.clone();
        let (prepared, applier) = pools::run(Pool::Recovery, move || {let _memory=open_memory;let (svc2,aid2,staging2,manifest2)=(svc2,aid2,staging2,manifest2);fr::open_recovered(&svc2, shard, &aid2, &staging2, &manifest2)}).await.map_err(|e| e.reason)??;
        // 3. Apply lock, short: still current, then publish and swap the
        // applier in. Dropping `prepared` (stale) closes the copy.
        pools::run(Pool::Recovery, move || -> Result<Arc<ReplicaApplier>, String> {
            let _memory=metadata_memory;
            let (me,index,aid,svc,placeholder)=(me,index,aid,svc,placeholder);
            let mut prepared = prepared;
            let _g = me.apply_lock.lock();
            if !me.recovery_slot_held(&index, shard, &aid, &svc, &placeholder) {
                return Err(STALE_RECOVERY.into());
            }
            fr::activate(&svc, &mut prepared)?;
            let applier = Arc::new(applier);
            if !me.shards.replace_recovery_applier(&index, shard, &aid, &svc, &placeholder, applier.clone()) {
                let _ = svc.close_shard_if_same(&applier.copy());
                return Err(STALE_RECOVERY.into());
            }
            Ok(applier)
        }).await.map_err(|e| e.reason)?
    }

    /// Receive one file in chunks (see [`Self::recover_replica_files`]):
    /// up to `max_concurrent_file_chunks` requests in flight, applied in
    /// offset order. On any error the progress made is kept (the next
    /// attempt resumes after the last verified chunk).
    #[allow(clippy::too_many_arguments)]
    async fn fetch_file(&self, base: &FileChunkBase<'_>, writer: crate::replication::file_recovery::PartWriter, settings: &crate::replication::throttle::RecoverySettings,
        staging: &std::path::Path, reserved: &crate::replication::file_recovery::DiskReservation, stats: &mut crate::replication::recovery::FileRecoveryStats, i: usize,
        current: &(dyn Fn() -> bool + Send + Sync), publish: &(dyn Fn(&crate::replication::recovery::FileRecoveryStats) + Send + Sync))
        -> Result<(elasticsearch_engine::hooks::files::IndexFileEntry, Option<std::path::PathBuf>), String>
    {
        use crate::replication::file_recovery::{self as fr, FileChunkReq};
        use elasticsearch_http::pools::{self, Pool};
        use futures::StreamExt;
        let _path_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(writer.entry().path.len() as u64,"recovery file-fetch path").map_err(|e|e.to_string())?;
        let (len, path, from) = (writer.entry().len, writer.entry().path.clone(), writer.done());
        let chunk = settings.chunk_size.max(1);
        // Only the bounded in-flight window needs descriptors; file size must
        // not allocate a proportional offsets vector before any chunk runs.
        let offsets = (from..len).step_by(chunk as usize);
        let transport = self.transport.clone();
        let requests = offsets.map(|offset| {
            let req = FileChunkReq::try_new(base.index,base.shard,base.recovery_id,base.aid,
                base.source_allocation_id,base.term,&path,offset,chunk.min(len-offset));
            let transport=transport.clone();
            async move {fetch_chunk(transport,base.addr,req.map_err(|e|e.to_string())?).await}
        });
        let mut chunks = futures::stream::iter(requests).buffered(settings.max_concurrent_file_chunks.max(1));
        let mut writer = Some(writer);
        let suspend = |w: Option<fr::PartWriter>| async move {
            if let Some(w) = w {
                let _ = pools::run(Pool::Recovery, move || w.suspend()).await;
            }
        };
        let mut published = Instant::now();
        while let Some(resp) = chunks.next().await {
            let resp = match resp {
                Ok(resp) => resp,
                Err(e) => {
                    suspend(writer.take()).await;
                    return Err(e);
                }
            };
            if !current() {
                suspend(writer.take()).await;
                return Err(RECOVERY_SOURCE_CHANGED.into());
            }
            let n = resp.data.len() as u64;
            // Paced on the source only (ES 9 `RemoteRecoveryTargetHandler`);
            // the target reports its throttle time as 0.
            stats.source_throttle_millis = stats.source_throttle_millis.max(resp.source_throttle_nanos / 1_000_000);
            let w = writer.take().expect("writer is held between chunks");
            let (w, written) = pools::run(Pool::Recovery, move || {
                let mut w = w;
                let r = w.write_chunk(resp.offset, &resp.data, resp.crc32);
                (w, r)
            }).await.map_err(|e| e.reason)?;
            writer = Some(w);
            if let Err(e) = written {
                suspend(writer.take()).await;
                return Err(e);
            }
            FILE_CHUNK_BYTES_RECEIVED.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            reserved.written(n);
            stats.details.set_recovered(i, stats.details[i].recovered + n)?;
            stats.bytes_recovered += n;
            if published.elapsed() >= Duration::from_millis(200) {
                publish(stats);
                published = Instant::now();
            }
        }
        let w = writer.take().expect("writer is held after the last chunk");
        // Durable and in place; the caller records it (`ProgressBatch`).
        let _ = staging;
        pools::run(Pool::Recovery, move || w.finish_deferred()).await.map_err(|e| e.reason)?
    }

    /// Stateless: give a new primary context its object store directory and
    /// upload a first commit (so a later recovery need not replay the whole
    /// translog).
    fn attach_object_store(&self, p: &Arc<PrimaryCtx>, index_uuid: &str) {
        let Some(os) = self.object_store.as_ref().filter(|_| self.local.is_stateless()) else { return };
        let shard_store = os.shard(index_uuid, p.shard);
        if p.object_store.set(shard_store.clone()).is_err() {
            return;
        }
        let (svc, shard, term, gate) = (p.svc.clone(), p.shard, p.term, p.gate.clone());
        tokio::spawn(async move {
            if let Err(e) = shard_store.commit(svc, shard, term, gate).await {
                tracing::warn!(shard, "initial object store commit failed: {e}");
            }
        });
    }

    /// Stateless primary allocated to this index node (the shard already
    /// exists): rebuild the copy from the object store, then start it as
    /// the primary under the new term.
    fn recover_primary_from_object_store(&self, state: &ClusterState, index: &str, shard: u32, routing: &ShardRouting, uuid: String, term: u64) {
        let Some(aid) = routing.allocation_id().map(str::to_string) else { return };
        let Some(os) = self.object_store.as_ref().map(|o| o.shard(&uuid, shard)) else { return };
        let Some(svc) = self.wipe_local_copy(index, shard, state) else { return };
        if let Err(e) = Self::ensure_copy_open(&svc, shard, &aid) {
            self.report_failed(index, shard, &aid, None, &format!("failed recovery: {e}"), false);
            return;
        }
        let placeholder = match ReplicaApplier::open(svc.clone(), shard) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                self.report_failed(index, shard, &aid, None, &format!("failed recovery: {e}"), false);
                return;
            }
        };
        let slot: Arc<Mutex<Option<RecoveryState>>> = Arc::new(Mutex::new(None));
        self.shards.insert(LocalShard { index: index.into(), shard, allocation_id: aid.clone(), term, svc: svc.clone(), role: Role::Replica(placeholder.clone()), recovery: Some(slot.clone()) });
        let me = self.me();
        let index = index.to_string();
        let target_copy = placeholder.copy();
        tokio::spawn(async move {
            if let Err(e) = me.primary_from_object_store(&index, shard, &aid, term, svc.clone(), placeholder, os, slot).await {
                if !me.recovery_current(&index, shard, &aid, &svc, &target_copy) {
                    tracing::info!(%index, shard, "stale object store recovery stopped: {e}");
                    return;
                }
                tracing::warn!(%index, shard, "primary recovery from the object store failed: {e}");
                me.report_failed(&index, shard, &aid, None, &format!("failed to recover from the object store: {e}"), false);
            }
        });
    }

    async fn primary_from_object_store(&self, index: &str, shard: u32, aid: &str, term: u64, svc: Arc<IndexService>, applier: Arc<ReplicaApplier>, os: Arc<crate::stateless::ShardObjectStore>, slot: Arc<Mutex<Option<RecoveryState>>>) -> Result<(), String> {
        let target_copy = applier.copy();
        let plan = os.recovery_plan().await?;
        let register = |a: Arc<ReplicaApplier>| {
            self.recovery_current(index, shard, aid, &svc, &target_copy)
                && self.shards.register_recovery_applier(index, shard, aid, &svc, a)
        };
        let info = crate::stateless::plan_info(&plan, &svc, shard)?;
        let pull = crate::stateless::plan_puller(os.clone(), plan.clone(), info.primary_term);
        let (applier, cp) = shards::run_recovery_target(applier, info, slot, &register, pull).await?;
        if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
            return Err(STALE_RECOVERY.into());
        }
        let (p, resync) = {
            let _g = self.apply_lock.lock();
            if !self.recovery_current(index, shard, aid, &svc, &target_copy) { return Err(STALE_RECOVERY.into()); }
            let state = self.state();
            if self.my_routing(&state, index, shard, aid).filter(|r| r.state == ShardState::Initializing && r.primary).is_none() {
                return Err(STALE_RECOVERY.into());
            }
            let (p, resync) = self.promote(index, shard, aid, term, &svc, &applier)?;
            self.refresh_primary_targets(&state, &p);
            if let Some(local) = self.shards.get(index, shard) {
                let mut g = local.lock();
                g.role = Role::Primary(p.clone());
                g.term = term;
                self.shards.mark_primary(index, shard);
            }
            (p, resync)
        };
        if let Some(resync) = resync {
            let me = self.me();
            let p = p.clone();
            tokio::spawn(async move { me.resync_replicas(p, resync).await });
        }
        tracing::info!(index, shard, local_checkpoint = cp, commit = ?plan.commit.as_ref().map(|c| c.generation), translog_blobs = plan.translog.len(), "recovered the primary from the object store");
        self.report_started(index, shard, aid);
        Ok(())
    }

    /// Stateless search copy: replay the object store, then catch up from
    /// the primary's translog above what the object store held.
    #[allow(clippy::too_many_arguments)]
    async fn recover_replica_from_object_store(&self, index: &str, shard: u32, aid: &str, uuid: &str, term: u64, svc: Arc<IndexService>, applier: Arc<ReplicaApplier>, primary_node: &str, slot: Arc<Mutex<Option<RecoveryState>>>, os: Arc<crate::stateless::ShardObjectStore>, envelope:&crate::capacity::lease::TaskSlot) -> Result<Arc<elasticsearch_engine::index::IndexShard>, String> {
        use crate::replication::recovery::RecoveryTarget;
        let addr = self.address_of(primary_node).ok_or("primary node not in cluster state")?;
        let target_copy = applier.copy();
        let plan = os.recovery_plan().await?;
        let context=self.admit_recovery_memory(index,shard,aid,uuid,term,primary_node,plan.commit.as_ref().map_or(0,|commit|commit.generation),envelope)?;
        context.scope_future(async {
        let register = |a: Arc<ReplicaApplier>| {
            self.recovery_current(index, shard, aid, &svc, &target_copy)
                && self.shards.register_recovery_applier(index, shard, aid, &svc, a)
        };
        // Under the current term: live replication keeps arriving with it.
        let mut info = crate::stateless::plan_info(&plan, &svc, shard)?;
        info.primary_term = info.primary_term.max(term);
        let mut target = tokio::task::block_in_place(|| RecoveryTarget::with_applier(applier, info)).map_err(|e| e.to_string())?;
        if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
        if let Some(c) = &plan.commit {
            for i in 0..c.batches {
                let batch = os.commit_batch(c, i).await?;
                if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
                tokio::task::block_in_place(|| target.apply_snapshot_batch(&batch)).map_err(|e| e.to_string())?;
                *slot.lock() = Some(target.state());
            }
        }
        if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
        tokio::task::block_in_place(|| target.snapshot_done()).map_err(|e| e.to_string())?;
        for name in &plan.translog {
            let batch = os.translog_batch(name).await?;
            if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
            tokio::task::block_in_place(|| target.apply_translog_batch(&batch, None)).map_err(|e| e.to_string())?;
            *slot.lock() = Some(target.state());
        }
        let from = target.applier().local_checkpoint();
        let restored = from;
        // The primary's translog above the object store.
        let start: RecoveryStartResp = self
            .transport
            .send_request(addr, shards::ACTION_RECOVERY_START, &RecoveryStartReq::try_new(index, shard, aid, Some(from)).map_err(|e| e.to_string())?, Duration::from_secs(60))
            .await
            .map_err(|e| e.to_string())?;
        if start.info.snapshot_docs > 0 || start.info.start_seq_no != from + 1 {
            // The primary no longer has that translog: full peer recovery.
            return Err(crate::stateless::TAIL_UNAVAILABLE.into());
        }
        let recovery_id = start.recovery_id;
        let mut gcp = -1;
        loop {
            let req = RecoveryBatchReq::try_new(index, shard, recovery_id, aid, &self.local.id, &self.local.transport_address).map_err(|e|e.to_string())?;
            let b: RecoveryBatch = self.transport.send_request(addr, shards::ACTION_RECOVERY_BATCH, &req, Duration::from_secs(120)).await.map_err(|e| e.to_string())?;
            if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
            gcp = gcp.max(b.global_checkpoint);
            if b.snapshot {
                if !b.ops.is_empty() {
                    return Err(crate::stateless::TAIL_UNAVAILABLE.into());
                }
                continue;
            }
            if !b.ops.is_empty() {
                let mapping: Option<serde_json::Value> = serde_json::from_str(&b.mapping_json).ok();
                tokio::task::block_in_place(|| target.apply_translog_batch(&b.ops, mapping.as_ref())).map_err(|e| e.to_string())?;
            }
            *slot.lock() = Some(target.state());
            if b.phase_done {
                break;
            }
        }
        if !register(target.applier().clone()) { return Err(STALE_RECOVERY.into()); }
        let cp = tokio::task::block_in_place(|| target.finalize(gcp)).map_err(|e| e.to_string())?;
        *slot.lock() = Some(target.state());
        let applier = target.into_applier();
        if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
            return Err(STALE_RECOVERY.into());
        }
        if !register(applier.clone()) { return Err(STALE_RECOVERY.into()); }
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            if !self.recovery_current(index, shard, aid, &svc, &target_copy) { return Err(STALE_RECOVERY.into()); }
            let req = RecoveryFinalizeReq::try_new(index, shard, recovery_id, aid, applier.local_checkpoint().max(cp), false).map_err(|e|e.to_string())?;
            let ok: bool = self.transport.send_request(addr, shards::ACTION_RECOVERY_FINALIZE, &req, Duration::from_secs(60)).await.map_err(|e| e.to_string())?;
            if ok {
                break;
            }
            if Instant::now() > deadline {
                return Err("timed out waiting to be marked in sync".into());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if !self.recovery_current(index, shard, aid, &svc, &target_copy) {
            return Err(STALE_RECOVERY.into());
        }
        tokio::task::block_in_place(|| -> Result<(), String> {
            let _permit = target_copy.operation_permit().map_err(|e| e.to_string())?;
            if !self.recovery_current(index, shard, aid, &svc, &target_copy) { return Err(STALE_RECOVERY.into()); }
            shards::write_copy_record(&svc, &CopyRecord { index: index.into(), index_uuid: uuid.into(), shard, allocation_id: aid.into(), primary_term: term, primary: false,
                history_uuid: None, global_checkpoint: None })
                .map_err(|e| format!("persisting recovered copy failed: {e}"))?;
            target_copy.shard.refresh().map_err(|e| format!("refresh after recovery failed: {e}"))?;
            // launch_peer_recovery reports STARTED only after admission completion.
            Ok(())
        })?;
        tracing::info!(index, shard, restored_to = restored, local_checkpoint = applier.local_checkpoint(), commit = ?plan.commit.as_ref().map(|c| c.generation), translog_blobs = plan.translog.len(), "recovered the search copy from the object store");
        Ok(target_copy)
        }).await
    }

    /// Does this node still hold this exact physical recovery target?
    /// `wipe_local_copy` may retain both the service and allocation id.
    /// The slot still belongs to the recovery of `aid` onto `copy`: the
    /// local shard entry holds that allocation, this incarnation of the
    /// index and an applier of `copy`. Unlike [`Self::recovery_current`],
    /// `copy` may be closed and out of the engine slot: a file recovery
    /// retires its placeholder before it opens the received copy, and an
    /// open that failed there is still this recovery's to retry or report.
    fn recovery_slot_held(&self, index: &str, shard: u32, aid: &str, svc: &Arc<IndexService>, copy: &Arc<elasticsearch_engine::index::IndexShard>) -> bool {
        self.shards.get(index, shard).is_some_and(|s| {
            let g = s.lock();
            g.allocation_id == aid && Arc::ptr_eq(&g.svc, svc) && matches!(&g.role, Role::Replica(r) if Arc::ptr_eq(&r.copy(), copy))
        }) && self.engine_for(shard).and_then(|e| e.get_index(index)).is_some_and(|e| Arc::ptr_eq(&e, svc))
    }

    fn recovery_current(&self, index: &str, shard: u32, aid: &str, svc: &Arc<IndexService>, copy: &Arc<elasticsearch_engine::index::IndexShard>) -> bool {
        let held = self.shards.get(index, shard).map(|s| {
            let g = s.lock();
            g.allocation_id == aid && Arc::ptr_eq(&g.svc, svc)
                && matches!(&g.role, Role::Replica(r) if Arc::ptr_eq(&r.copy(), copy))
        });
        held.unwrap_or(false) && !copy.is_closed()
            && svc.shard(shard).is_some_and(|current| Arc::ptr_eq(&current, copy))
            && self.engine_for(shard).and_then(|e| e.get_index(index)).map(|e| Arc::ptr_eq(&e, svc)).unwrap_or(false)
    }

    fn refresh_primary_targets(&self, state: &ClusterState, p: &PrimaryCtx) {
        let Some(st) = state.routing_table.shard_table(&p.index, p.shard) else { return };
        let in_sync: Vec<String> = state.metadata.indices.get(&p.index).and_then(|m| m.in_sync_allocation_ids.get(&p.shard)).map(|s| s.iter().cloned().collect()).unwrap_or_default();
        // Every initializing copy other than us, including a primary
        // relocation target (which is flagged `primary`).
        let initializing: Vec<String> = st.copies.iter().filter(|c| c.state == ShardState::Initializing && c.allocation_id() != Some(p.allocation_id.as_str())).filter_map(|c| c.allocation_id().map(str::to_string)).collect();
        {
            let mut tracker = p.tracker.lock();
            tracker.update_from_cluster_state(in_sync.iter().map(String::as_str), initializing.iter().map(String::as_str));
            // ES `ReplicationGroup`: in-sync copies the routing table lost are
            // marked stale by the next write; every listed copy (unassigned
            // ones too) counts in a write's `_shards.total`.
            tracker.set_routing(st.copies.iter().filter_map(|c| Some((c.allocation_id()?, c.node_id.as_deref()?))), st.copies.len() as u32);
        }
        // File recoveries of targets that failed, moved or were removed
        // release their pinned commit and retention lock.
        if !p.recovery_files.lock().is_empty() {
            let ids: std::collections::HashSet<String> = initializing.iter().cloned().collect();
            shards::release_abandoned_recoveries(p, &ids);
        }
        let mut targets = p.targets.lock();
        let assigned: BTreeMap<String, String> = st.copies.iter().filter_map(|c| Some((c.allocation_id()?.to_string(), c.node_id.clone()?))).collect();
        targets.retain(|aid, _| assigned.contains_key(aid));
        for (aid, node_id) in assigned {
            if aid == p.allocation_id {
                continue;
            }
            // Recovering copies get an address on their first batch pull.
            let started = st.copies.iter().any(|c| c.allocation_id() == Some(aid.as_str()) && c.state != ShardState::Initializing);
            if !started && !targets.contains_key(&aid) {
                continue;
            }
            if let Some(n) = state.nodes.get(&node_id) {
                // Never refused for memory: a started copy left without a
                // target (a promoted primary starts with none) is skipped by
                // every write and global checkpoint sync until a later state
                // happens to retry, though the master counts it in sync. The
                // metadata is a few identifiers per copy, still accounted.
                let added = crate::control_memory::unrefused("replication target metadata", ||
                    Target::try_new(&aid, &node_id, &n.transport_address).and_then(|target| targets.try_insert(target)));
                if let Err(error) = added {
                    tracing::error!(%error, %aid, index = %p.index, shard = p.shard, "replication target metadata could not be allocated; keeping the previous target");
                }
            }
        }
    }

    /// Tell the master a copy started. Retried across master changes for
    /// as long as the routing table still shows the copy initializing here:
    /// a report lost to an election would otherwise leave it stuck.
    ///
    /// A report the breaker refuses to build waits for memory instead of
    /// being dropped (ES's `ShardStateAction` never drops one): with no
    /// report the master never learns the copy started.
    pub fn report_started(&self, index: &str, shard: u32, aid: &str) {
        let key = format!("started:{index}:{shard}:{aid}");
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid) = (index.to_string(), aid.to_string());

        tokio::spawn(async move {
            let relevant = || {
                me.my_routing(&me.state(), &index, shard, &aid).is_some_and(|r| r.state == ShardState::Initializing)
                    && me.shards.get(&index, shard).is_some_and(|l| l.lock().allocation_id == aid)
            };
            let Some(req) = admit_report(&index, || ShardStartedReq::try_new(&index, shard, &aid), relevant).await else {
                me.pending.lock().remove(&key);
                return;
            };
            for attempt in 0..REPORT_MAX_RETRIES {
                match me.to_master::<_, ()>(ACTION_SHARD_STARTED, &req, Duration::from_secs(60)).await {
                    Ok(()) => break,
                    Err(e) => tracing::warn!(index = %req.index, attempt, "shard-started report failed: {}", e.reason),
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
                let still = me.my_routing(&me.state(), &req.index, req.shard, &req.allocation_id).map(|r| r.state == ShardState::Initializing).unwrap_or(false);
                let ours = me.shards.get(&req.index, req.shard).map(|l| l.lock().allocation_id == req.allocation_id).unwrap_or(false);
                if !still || !ours {
                    break;
                }
            }
            me.pending.lock().remove(&key);
        });
    }

    /// Tell the master a copy failed; retried like `report_started` while
    /// the routing table still assigns the copy to this node.
    pub fn report_failed(&self, index: &str, shard: u32, aid: &str, reporter_term: Option<u64>, message: &str, mark_as_stale: bool) {
        let key = format!("failed:{index}:{shard}:{aid}");
        if !self.pending.lock().insert(key.clone()) {
            return;
        }
        let me = self.me();
        let (index, aid, message) = (index.to_string(), aid.to_string(), message.to_string());

        tokio::spawn(async move {
            // The copy may be another node's (a primary failing a replica
            // its resync did not reach): relevance is the routing table's,
            // not this node's.
            let relevant = || failure_report_relevant(&me.state(), &index, shard, &aid, reporter_term);
            let Some(req) = admit_report(&index, || ShardFailedReq::try_new(&index, shard, &aid, reporter_term, &message, mark_as_stale), relevant).await else {
                me.pending.lock().remove(&key);
                return;
            };
            for attempt in 0..REPORT_MAX_RETRIES {
                match me.to_master::<_, ()>(ACTION_SHARD_FAILED, &req, Duration::from_secs(60)).await {
                    Ok(()) => break,
                    Err(e) => {
                        tracing::warn!(index = %req.index, attempt, "shard-failed report failed: {}", e.reason);
                        // A term-checked report the master rejected for its
                        // term stays rejected; any other failure (a breaker,
                        // a lost master, a failed publication) is retried,
                        // as ES's `ShardStateAction` does.
                        if e.reason.contains(STALE_REPORTER_TERM) {
                            break;
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
                if !failure_report_relevant(&me.state(), &req.index, req.shard, &req.allocation_id, req.reporter_term) {
                    break;
                }
            }
            me.pending.lock().remove(&key);
        });
    }

    // ------------------------------------------------------------ background

    /// Independent background duties, each on its own interval and never
    /// awaited one after another (a slow flush or a slow replica used to
    /// hold up allocation on the master for as long as it took):
    ///
    /// * engine maintenance (context expiry, vector optimizers) every 1s;
    /// * the master round (reroute, registry reconciliation, store and disk
    ///   info fetches) every 1s on the elected master, each piece fired off
    ///   at most once at a time, so the round itself never waits on I/O;
    /// * replication top-up every 1s: one task per primary, at most
    ///   [`REPLICATION_TOPUP_CONCURRENCY`] at once, skipped for a shard whose
    ///   previous top-up is still running (a slow replica holds up only its
    ///   own shard);
    /// * periodic flush every 5s (ES `indices.memory.interval`): one task
    ///   per copy, at most the flush pool size at once, the engine flush on
    ///   the blocking pool.
    fn spawn_background(self: Arc<Self>) {
        fn every(period: Duration) -> tokio::time::Interval {
            let mut tick = tokio::time::interval(period);
            // A late tick is dropped, not replayed in a burst.
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            tick
        }
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut tick = every(Duration::from_secs(1));
            let mut n: u64 = 0;
            loop {
                tick.tick().await;
                n += 1;
                let Some(me) = weak.upgrade() else { break };
                me.engine_maintenance();
                // Copies kept for their search contexts go once released.
                if n % 2 == 0 && !me.retained.lock().is_empty() {
                    let me = me.clone();
                    tokio::task::spawn_blocking(move || me.release_retained_copies());
                }
                // A failed metadata application is retried every 5s, not
                // only at the next state change.
                if me.rollback_due() {
                    if let Some(lease) = VectorOptimizerLease::claim(&me.background.apply_retry) {
                        let me = me.clone();
                        tokio::task::spawn_blocking(move || {
                            let _lease = lease;
                            me.run_due_rollback();
                        });
                    }
                }
                if n % 5 == 0 && me.application_failed() {
                    if let Some(lease) = VectorOptimizerLease::claim(&me.background.apply_retry) {
                        tokio::task::spawn_blocking(move || {
                            let _lease = lease;
                            me.retry_failed_application();
                        });
                    }
                }
            }
        });
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut tick = every(Duration::from_secs(1));
            let mut n: u64 = 0;
            loop {
                tick.tick().await;
                n += 1;
                let Some(me) = weak.upgrade() else { break };
                // Target leases expire on every node even without new cluster
                // publications; unfinished allocations keep their own charge.
                me.capacity_leases.lock().reconcile(&me,&me.state(),Instant::now());
                if me.is_master() {
                    me.master_round(n);
                }
            }
        });
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut tick = every(Duration::from_secs(1));
            loop {
                tick.tick().await;
                let Some(me) = weak.upgrade() else { break };
                me.replicate_idle();
            }
        });
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut tick = every(Duration::from_secs(1));
            let clock = Arc::new(Mutex::new(elasticsearch_engine::ilm::TickClock::default()));
            loop {
                tick.tick().await;
                let Some(me) = weak.upgrade() else { break };
                if let Some(lease) = VectorOptimizerLease::claim(&me.background.lifecycle) {
                    let clock = clock.clone();
                    tokio::spawn(async move {
                        let _lease = lease;
                        me.lifecycle_tick(clock).await;
                    });
                }
            }
        });
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut tick = every(Duration::from_secs(5));
            // The first tick fires at once; flushing starts one period in.
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(me) = weak.upgrade() else { break };
                me.periodic_flush();
            }
        });
    }

    /// The engine duties the cluster takes over from `Node::maintenance`
    /// (expiring scroll / PIT contexts, vector optimizers). Flushes go
    /// through the gated path.
    fn engine_maintenance(&self) {
        let now = Instant::now();
        let expired = elasticsearch_engine::search_ext::take_expired_sql_cursors(&self.engine, now);
        if !expired.is_empty() {
            // A cursor can own many rows: deallocation must not stall heartbeats.
            tokio::task::spawn_blocking(move || drop(expired));
        }
        let shard_engines = self.shard_engines.all();
        reap_reader_contexts(std::iter::once(&*self.engine).chain(shard_engines.iter().map(|(_, se)| &*se.engine)), now);
        self.engine.reap_async_searches();
        crate::front::reap_expired_contexts();
        self.spawn_lease_expiry();
        self.spawn_vector_optimizer();
        self.fail_tragic_copies();
    }

    /// A shard copy whose engine hit a translog tragic event (append or
    /// fsync failure: the engine has failed itself and refuses every op) is
    /// reported failed to the master once, which reallocates it (a replica
    /// is promoted when it was the primary), as ES fails the shard on
    /// `Translog.tragedy`.
    fn fail_tragic_copies(&self) {
        for slot in self.shards.all() {
            let (index, shard, aid, copy) = {
                let local = slot.lock();
                let copy = match &local.role {
                    Role::Primary(p) => p.copy(),
                    Role::Replica(r) => r.copy(),
                };
                (local.index.clone(), local.shard, local.allocation_id.clone(), copy)
            };
            if let Some(error) = copy.shard.take_tragic_report() {
                tracing::error!(index = %index, shard, "failing the shard copy after a translog tragic event: {error}");
                self.report_failed(&index, shard, &aid, None, &format!("translog tragic event: {error}"), true);
            }
        }
    }

    /// Expire stale durable retention leases (ES
    /// `index.soft_deletes.retention_lease.period`) on every engine, off the
    /// async runtime, about once a minute.
    fn spawn_lease_expiry(&self) {
        static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
        {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            if last.is_some_and(|t| t.elapsed() < Duration::from_secs(60)) {
                return;
            }
            *last = Some(Instant::now());
        }
        let mut engines = vec![self.engine.clone()];
        engines.extend(self.shard_engines.all().into_iter().map(|(_, e)| e.engine.clone()));
        tokio::task::spawn_blocking(move || {
            for engine in engines {
                engine.expire_retention_leases();
            }
        });
    }

    /// Seal appendable vector chunks into HNSW-indexed segments (and run the
    /// vacuum / merge optimizers) on every engine, off the async runtime.
    /// Never trims the translog, so it needs no flush gate.
    fn spawn_vector_optimizer(&self) {
        let Some(running) = VectorOptimizerLease::claim(&self.optimizing) else { return };
        let mut engines = vec![self.engine.clone()];
        engines.extend(self.shard_engines.all().into_iter().map(|(_, e)| e.engine.clone()));
        tokio::task::spawn_blocking(move || {
            let _running = running;
            Engine::optimize_vectors_background(&engines);
        });
    }

    /// Ship ops written by paths that do not replicate inline (e.g. by-query
    /// background tasks), so replicas never drift far, and deliver an idle
    /// shard's final global checkpoint. One task per primary (calling the
    /// shard's own replication entry points); a primary whose previous
    /// top-up is still running is skipped this round.
    fn replicate_idle(&self) {
        let primaries: Vec<Arc<PrimaryCtx>> = self.shards.all().into_iter().filter_map(|s| match &s.lock().role {
            Role::Primary(p) => Some(p.clone()),
            Role::Replica(_) => None,
        }).collect();
        for p in primaries {
            let Some(lease) = CopyLease::claim(&self.background.replicating, Arc::as_ptr(&p) as usize) else { continue };
            let me = self.me();
            let permits = self.background.replication_permits.clone();
            tokio::spawn(async move {
                // Force-merge holds the shard writer for the entire merge:
                // read engine state behind that lock off the async workers.
                let copy = p.clone();
                let max = match tokio::task::spawn_blocking(move || copy.copy().shard.max_seq_no()).await {
                    Ok(max) => max,
                    Err(_) => return,
                };
                let behind = max > p.tracker.lock().global_checkpoint();
                // The round runs to its end: a replica that does not answer
                // is failed by the replication request timeout (as ES fails
                // the copy), which a cancelled round would never reach. The
                // shard's lease stays with the round (no second round for
                // it meanwhile); the permit is held at most the bound, so a
                // stalled replica does not hold up other shards' top-ups.
                let permit = if behind {
                    match permits.acquire_owned().await {
                        Ok(permit) => Some(permit),
                        Err(_) => return,
                    }
                } else {
                    None
                };
                let round = tokio::spawn(async move {
                    let _lease = lease;
                    let net = TransportNet(me.clone());
                    if behind {
                        let _g = p.gate.read().await;
                        if let Err(e) = p.replicate_all(&net).await {
                            tracing::warn!(index = %p.index, "background replication failed: {e}");
                        }
                    }
                    // The global checkpoint sync: no permit (its own path).
                    p.sync_global_checkpoint(&net).await;
                });
                if tokio::time::timeout(BACKGROUND_REPLICATION_TIMEOUT, round).await.is_err() {
                    tracing::debug!("background replication round still running; its permit is released");
                }
                drop(permit);
            });
        }
    }

    /// `indices.memory.shard_inactive_time` (ES default 5m).
    fn shard_inactive_time(&self) -> Duration {
        self.config
            .settings
            .get("indices.memory.shard_inactive_time")
            .and_then(|v| v.as_str())
            .and_then(|v| elasticsearch_cluster::coordination::parse_time_millis(v).ok())
            .map_or(Duration::from_secs(300), Duration::from_millis)
    }

    /// Bound translog / replica WAL growth (`index.translog.flush_threshold_size`
    /// 10 GiB in ES9, capped at 1% of disk), and flush copies that hold uncommitted ops or vector
    /// segment swaps but saw no write for `indices.memory.shard_inactive_time`
    /// (ES `flushOnIdle`), so a restart neither replays (re-indexes) an idle
    /// tail nor redoes the seals and merges made since the last commit.
    /// Primaries flush without a write fence: their replication retention
    /// (installed by `PrimaryCtx::new`) keeps every op a copy still needs.
    ///
    /// One task per copy, at most the flush pool size at once; a copy whose
    /// previous flush is still running is skipped this round. Nothing else
    /// waits for a flush.
    fn periodic_flush(&self) {
        let inactive = self.shard_inactive_time();
        let now = Instant::now();
        let copies: Vec<Result<Arc<PrimaryCtx>, Arc<ReplicaApplier>>> = self.shards.all().into_iter().map(|s| match &s.lock().role {
            Role::Primary(p) => Ok(p.clone()),
            Role::Replica(r) => Err(r.clone()),
        }).collect();
        // Forget the write activity of copies that are gone.
        let live: std::collections::HashSet<usize> = copies.iter().map(|role| match role {
            Ok(p) => Arc::as_ptr(&p.copy()) as usize,
            Err(r) => Arc::as_ptr(&r.copy()) as usize,
        }).collect();
        self.write_activity.lock().retain(|k, _| live.contains(k));
        for role in copies {
            let key = match &role {
                Ok(p) => Arc::as_ptr(p) as usize,
                Err(r) => Arc::as_ptr(r) as usize,
            };
            let Some(lease) = CopyLease::claim(&self.background.flushing, key) else { continue };
            let me = self.me();
            tokio::spawn(async move {
                let _lease = lease;
                match role {
                    Ok(p) => me.flush_primary(p, inactive, now).await,
                    Err(r) => me.flush_replica(r, inactive, now).await,
                }
            });
        }
    }

    /// One lifecycle iteration (ILM, data stream lifecycle), on the master
    /// only, as a dry run of the metadata service: it executes on the
    /// scratch copy synced to the committed state, its result (rollovers,
    /// deletes, settings / alias changes, the indices' execution state) is
    /// published like any metadata change, and the applier applies it on
    /// every node. As in ES, where ILM steps are cluster-state update tasks:
    /// a failed publication reverts the step with its execution state, and
    /// nothing reaches a live engine before the commit (a delete destroys no
    /// data before it).
    /// Stats gathered by [`Self::prefetch_rollover_stats`] (fresh ones only).
    fn cached_rollover_stats(&self, index: &str) -> Option<elasticsearch_engine::datastream::RolloverStats> {
        self.rollover_cache.lock().get(index).filter(|(at, _)| at.elapsed() < ROLLOVER_STATS_FRESH).map(|(_, s)| *s)
    }

    /// Gather the cluster-wide rollover stats of `indices` before a batch
    /// that judges rollover conditions (bounded: 8 at a time, 10 s each).
    pub(crate) async fn prefetch_rollover_stats(self: &Arc<Self>, indices: Vec<String>) {
        use futures::StreamExt;
        let fetched: Vec<(String, Option<elasticsearch_engine::datastream::RolloverStats>)> = futures::stream::iter(indices)
            .map(|index| {
                let node = self.clone();
                async move {
                    let stats = tokio::time::timeout(Duration::from_secs(10), crate::front::cluster_rollover_stats(&node, &index)).await.ok().flatten();
                    (index, stats)
                }
            })
            .buffer_unordered(8)
            .collect()
            .await;
        let now = Instant::now();
        let mut cache = self.rollover_cache.lock();
        cache.retain(|_, (at, _)| at.elapsed() < ROLLOVER_STATS_FRESH);
        for (index, stats) in fetched {
            if let Some(s) = stats {
                cache.insert(index, (now, s));
            }
        }
    }

    /// `_ilm/stop` completes: STOPPING becomes STOPPED, published, once no
    /// index is inside an unsafe action (a shrink), as ES's
    /// `OperationModeUpdateTask` does on the master.
    async fn settle_lifecycle_stop(self: &Arc<Self>) {
        let st = self.state();
        let Some(c) = st.metadata.customs.get(crate::meta_extra::ILM_CUSTOM) else { return };
        if c["running"].as_bool().unwrap_or(true) || c["mode"].as_str() == Some("STOPPED") {
            return;
        }
        let busy = st.metadata.indices.values().any(|im| im.lifecycle.as_deref().is_some_and(|l| l["ilm"]["action"] == "shrink" && l["ilm"]["step"] != "ERROR"));
        if busy {
            return;
        }
        let compute: Box<dyn FnOnce(&ClusterNode, &ClusterState, &mut crate::mirror::MetadataDelta) + Send> = Box::new(|_: &ClusterNode, base: &ClusterState, delta: &mut crate::mirror::MetadataDelta| {
            if let Some(c) = base.metadata.customs.get(crate::meta_extra::ILM_CUSTOM) {
                if c["running"].as_bool() == Some(false) && c["mode"].as_str() != Some("STOPPED") {
                    let mut c = c.clone();
                    c["mode"] = serde_json::Value::String("STOPPED".into());
                    delta.customs.insert(crate::meta_extra::ILM_CUSTOM.to_string(), Some(c));
                }
            }
        });
        if let Err(e) = self.metadata_service().pure(compute).await {
            tracing::debug!("lifecycle stop not published: {e}");
        }
    }

    async fn lifecycle_tick(self: Arc<Self>, clock: Arc<Mutex<elasticsearch_engine::ilm::TickClock>>) {
        if !self.is_master() {
            return;
        }
        // Rollover conditions are judged inside the batch on stats gathered
        // here (only when a periodic ILM or data stream lifecycle poll is due).
        if clock.lock().polls_due(&self.engine) {
            let candidates = rollover_candidates(&self.state());
            if !candidates.is_empty() {
                self.prefetch_rollover_stats(candidates).await;
            }
        }
        self.settle_lifecycle_stop().await;
        let run: Box<dyn FnOnce(&ClusterNode, crate::master_service::On) -> bool + Send> = Box::new(move |node: &ClusterNode, on: crate::master_service::On| {
            let engine = node.engine_on(on);
            elasticsearch_engine::ilm::tick_lifecycle(&engine, &mut clock.lock());
            true
        });
        if let Err(e) = self.metadata_service().run(run, true).await {
            tracing::debug!("lifecycle pass not published: {e}");
        }
    }

    /// Whether `copy` saw no write for `inactive` while holding uncommitted
    /// work; records its activity. Blocking (reads engine state).
    fn copy_idle(&self, copy: &Arc<elasticsearch_engine::index::IndexShard>, inactive: Duration, now: Instant) -> bool {
        let key = Arc::as_ptr(copy) as usize;
        let max = copy.shard.max_seq_no();
        // Merged-away files neither readers nor the last commit need go now.
        copy.shard.sweep_retired();
        let pending = max > copy.shard.committed_seq_no() || copy.shard.has_uncommitted_vector_changes() || copy.shard.has_uncommitted_merges();
        let mut activity = self.write_activity.lock();
        let since = match activity.get(&key) {
            Some(&(m, t)) if m == max => t,
            _ => now,
        };
        activity.insert(key, (max, since));
        pending && now.duration_since(since) >= inactive
    }

    /// The permit a background flush of this kind runs under: size-due
    /// flushes take the flush pool's permits; periodic ones go one at a
    /// time node-wide.
    ///
    /// A periodic flush queued behind others re-checks `size_due` every
    /// second: once the copy is over its translog threshold it takes a size
    /// permit instead (a queue of periodic flushes never holds back a
    /// size-due one). Returns whether the flush is size-due.
    async fn flush_permit(&self, mut size_due: bool, size_check: Arc<dyn Fn() -> bool + Send + Sync>) -> Option<(tokio::sync::OwnedSemaphorePermit, bool)> {
        let permit = loop {
            if size_due {
                break self.background.flush_permits.clone().acquire_owned().await.ok()?;
            }
            match tokio::time::timeout(Duration::from_secs(1), self.background.periodic_flush_permit.clone().acquire_owned()).await {
                Ok(permit) => break permit.ok()?,
                Err(_) => {
                    let check = size_check.clone();
                    size_due = tokio::task::spawn_blocking(move || check()).await.unwrap_or(false);
                }
            }
        };
        let counter = if size_due { &self.flush_stats.size } else { &self.flush_stats.periodic };
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some((permit, size_due))
    }

    async fn flush_primary(self: Arc<Self>, p: Arc<PrimaryCtx>, inactive: Duration, now: Instant) {
        // Settings, disk statistics and sequence fences can block.
        let (me, copy) = (self.clone(), p.clone());
        let due = tokio::task::spawn_blocking(move || {
            let c = copy.copy();
            let size = c.shard.translog_size() >= translog_flush_threshold(&c);
            let periodic = me.copy_idle(&c, inactive, now) || translog_age_flush_slot(&c) || merge_flush_due(&c);
            (size, periodic)
        }).await;
        let Ok((size_due, periodic_due)) = due else { return };
        if !(size_due || periodic_due) || p.relocated.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // Ops through `max` are shipped and acknowledged below.
        let copy = p.clone();
        let Ok(max) = tokio::task::spawn_blocking(move || copy.copy().shard.max_seq_no()).await else { return };
        // The replication round before the flush shares the top-up lease
        // (one round per shard at a time): a top-up in flight is waited for,
        // then this round runs to its end (a replica that does not answer
        // is failed by the request timeout), as the flush always waited for
        // it. A round that does not complete skips the flush, counted.
        // Waited for at most the replication request timeout (a top-up
        // ends within it); no flush past it, counted.
        let wait_until = Instant::now() + Duration::from_secs(70);
        let lease = loop {
            if let Some(lease) = CopyLease::claim(&self.background.replicating, Arc::as_ptr(&p) as usize) {
                break lease;
            }
            if Instant::now() >= wait_until || p.relocated.load(std::sync::atomic::Ordering::SeqCst) {
                self.flush_stats.skipped_replication.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(index = %p.index, shard = p.shard, "periodic flush skipped: a replication round of the shard did not end");
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        // Like a top-up: under the replication permits and the shard's gate.
        let Ok(round_permit) = self.background.replication_permits.clone().acquire_owned().await else { return };
        let (round_p, net) = (p.clone(), TransportNet(self.clone()));
        let round = tokio::spawn(async move {
            let _lease = lease;
            let _permit = round_permit;
            let _g = round_p.gate.read().await;
            round_p.replicate_all(&net).await
        });
        let key = Arc::as_ptr(&p) as usize;
        // A size-due flush is never skipped twice in a row for replication:
        // the commit is local (the replication retention keeps every op a
        // copy still needs), so it goes ahead once the global checkpoint
        // covers the ops, or at the second failed round regardless.
        let mut force = false;
        match round.await {
            Ok(Ok(_)) => {}
            failed => {
                let why = match failed {
                    Ok(Err(e)) => e,
                    Err(e) => e.to_string(),
                    Ok(Ok(_)) => unreachable!("matched above"),
                };
                let skipped = self.background.size_flush_skips.lock().get(&key).copied().unwrap_or(0);
                let covered = p.global_checkpoint() >= max;
                if size_due && (covered || skipped >= 1) {
                    tracing::warn!(index = %p.index, shard = p.shard, "the replication round before a translog-size flush failed ({why}); flushing locally");
                    force = true;
                } else {
                    self.flush_stats.skipped_replication.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if size_due {
                        self.background.size_flush_skips.lock().insert(key, skipped + 1);
                    }
                    tracing::warn!(index = %p.index, shard = p.shard, "periodic flush skipped: the replication round before it failed: {why}");
                    return;
                }
            }
        }
        let check_p = p.clone();
        let size_check: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            let c = check_p.copy();
            c.shard.translog_size() >= translog_flush_threshold(&c)
        });
        let Some((_permit, size_due)) = self.flush_permit(size_due, size_check).await else { return };
        let skips = self.background.size_flush_skips.clone();
        let stats_skipped = self.me();
        // Peer-recovery retention leases are renewed before the flush trims.
        let assigned: std::collections::HashSet<String> = self.state().routing_table.shard_table(&p.index, p.shard)
            .map(|t| t.copies.iter().filter_map(|c| c.node_id.clone()).collect()).unwrap_or_default();
        // No write fence (ES flushes without blocking indexing): the commit
        // point stays below every in-flight write's claim, and the
        // replication retention `PrimaryCtx::new` installed keeps every
        // translog op a copy or the tailer still needs. A copy still behind
        // is waited out: its retained ops would keep the translog over the
        // threshold and re-trigger this flush each round (writes after
        // `max` are no reason to wait).
        let flushed = tokio::task::spawn_blocking(move || {
            shards::sync_peer_recovery_leases(&p, &assigned, elasticsearch_engine::hooks::retention::DEFAULT_LEASE_PERIOD_MILLIS);
            // A lower bound of every in-sync copy's checkpoint, for this
            // copy's own operations-based recovery if it is ever deposed (and
            // inherited by it as a cold primary), recorded once durable here.
            shards::note_durable_primary_global_checkpoint(&p);
            let lagging = p.tracker.lock().lagging(max).iter().any(|(a, _)| a != &p.allocation_id);
            if lagging && !force {
                // Counted and logged (never a silent skip).
                stats_skipped.flush_stats.skipped_replication.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if size_due {
                    // A lagging copy counts as a skip too (never twice in a row).
                    let mut skips = skips.lock();
                    let n = skips.entry(key).or_insert(0);
                    if *n == 0 {
                        *n = 1;
                        tracing::warn!(index = %p.index, shard = p.shard, "translog-size flush deferred once: a copy lags behind seq_no [{max}]");
                        return Ok(());
                    }
                } else {
                    tracing::warn!(index = %p.index, shard = p.shard, "periodic flush skipped: a copy lags behind seq_no [{max}]");
                    return Ok(());
                }
            }
            skips.lock().remove(&key);
            let copy = p.copy();
            // Nothing to commit: no op above the commit and no merge or
            // vector swap since it. (Skipping on ops alone left a merge that
            // finished after the last write uncommitted, its inputs on disk.)
            if !copy.shard.has_uncommitted_changes() {
                return Ok(());
            }
            copy.shard.flush().map_err(|e| (p.index.clone(), e.reason))
        }).await;
        if let Ok(Err((index, reason))) = flushed {
            tracing::warn!(%index, "periodic flush failed: {reason}");
        }
    }

    async fn flush_replica(self: Arc<Self>, r: Arc<ReplicaApplier>, inactive: Duration, now: Instant) {
        let (me, replica) = (self.clone(), r.clone());
        let due = tokio::task::spawn_blocking(move || {
            let copy = replica.copy();
            let threshold = translog_flush_threshold(&copy);
            // Its own WAL size: reclaimable up to the global checkpoint it
            // received, or (a checkpoint that does not arrive) a local
            // commit of what it applied, at most every 30 s (its WAL then
            // stays until the checkpoint moves).
            let size = replica.needs_periodic_flush(threshold, false)
                || (replica.stats().wal_size_bytes >= threshold
                    && copy.shard.max_seq_no() > copy.shard.committed_seq_no()
                    && copy.shard.elapsed_since_flush() >= REPLICA_LOCAL_FLUSH_INTERVAL);
            let periodic = me.copy_idle(&copy, inactive, now) || (translog_age_flush_slot(&copy) && replica.needs_periodic_flush(u64::MAX, true)) || merge_flush_due(&copy);
            (size, periodic)
        }).await;
        let Ok((size_due, periodic_due)) = due else { return };
        if !(size_due || periodic_due) {
            return;
        }
        let check_r = r.clone();
        let size_check: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            let copy = check_r.copy();
            check_r.needs_periodic_flush(translog_flush_threshold(&copy), false)
        });
        let Some((_permit, _)) = self.flush_permit(size_due, size_check).await else { return };
        let flushed = tokio::task::spawn_blocking(move || r.flush().map(|_| ())).await;
        if let Ok(Err(e)) = flushed {
            tracing::warn!("replica flush failed: {e}");
        }
    }

    /// The master's periodic duties. Pure state computation goes to the
    /// coordinator; engine work to the metadata service; I/O (store and
    /// disk info fetches) to its own task. Each piece runs at most once at
    /// a time and nothing here is awaited, so a slow publication, flush or
    /// node cannot delay the next round.
    fn master_round(&self, n: u64) {
        // Startup resources can be replaced by the first published metadata.
        // Reconcile on the elected master in the metadata service, which
        // publishes templates and pipelines together to every node.
        // Every 10s (each check snapshots engine 0 around its kept job).
        if n % 10 == 1 {
            if let Some(lease) = VectorOptimizerLease::claim(&self.background.registry) {
                let me = self.me();
                tokio::spawn(async move {
                    let _lease = lease;
                    let run: Box<dyn FnOnce(&ClusterNode, crate::master_service::On) -> bool + Send> = Box::new(|node: &ClusterNode, _: crate::master_service::On| elasticsearch_engine::registry::ensure(&node.engine));
                    if let Err(e) = me.metadata_service().run(run, false).await {
                        tracing::warn!("reconciling built-in registries failed: {e}");
                    }
                });
            }
        }
        let state = self.state();
        // Full-cluster restart: find which nodes hold which copies.
        let needs_stores = state.routing_table.all_shards().any(|s| s.primary && s.state == ShardState::Unassigned && matches!(s.recovery_source, Some(RecoverySource::ExistingStore { .. })));
        if needs_stores || n % 30 == 0 {
            if let Some(lease) = VectorOptimizerLease::claim(&self.background.stores) {
                let (me, state) = (self.me(), state.clone());
                tokio::spawn(async move {
                    let _lease = lease;
                    me.fetch_stores(&state).await;
                });
            }
        }
        if n % 30 == 1 {
            if let Some(lease) = VectorOptimizerLease::claim(&self.background.cluster_info) {
                let (me, state) = (self.me(), state.clone());
                tokio::spawn(async move {
                    let _lease = lease;
                    me.refresh_cluster_info(&state).await;
                });
            }
            // Capacity probes have their own lease: slow/legacy peers must
            // not delay existing disk-watermark observations.
            if let Some(lease) = VectorOptimizerLease::claim(&self.background.capacity_info) {
                let (me, state) = (self.me(), state.clone());
                tokio::spawn(async move {
                    let _lease = lease;
                    crate::capacity::collect(&me, &state).await;
                });
            }
        }
        let has_unassigned = state.routing_table.all_shards().any(|s| s.state == ShardState::Unassigned);
        if has_unassigned || n % 10 == 0 {
            if let Some(lease) = VectorOptimizerLease::claim(&self.background.reroute) {
                let me = self.me();
                let ctx = self.master.clone();
                tokio::spawn(async move {
                    let _lease = lease;
                    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
                        let next = ctx.reroute(s);
                        if next.routing_table == s.routing_table && next.metadata.indices == s.metadata.indices { Err("no change".into()) } else { Ok(next) }
                    });
                    let _ = me.coord().submit(task).await;
                });
            }
        }
    }

    /// Ask every data node for the copies it holds, concurrently.
    async fn fetch_stores(&self, state: &ClusterState) {
        async fn fetch<'a>(node:&'a ClusterNode,id:&'a str,addr:SocketAddr)->(&'a str,Result<crate::store_records::StoreRecords,elasticsearch_transport::TransportError>){
            let recs = node.transport.send_request::<_, crate::store_records::StoreRecords>(addr, shards::ACTION_SHARD_STORES, &(), Duration::from_secs(10)).await;
            (id, recs)
        }
        // Fixed stack futures bound fanout without depending on JoinAll's
        // private queue/task allocation layout. IDs borrow the retained state.
        let mut nodes=state.data_nodes().filter_map(|n|n.transport_address.parse::<SocketAddr>().ok().map(|addr|(n.id.as_str(),addr)));
        let mut out = crate::store_records::StoreCache::default();
        async fn optional<'a>(node:&'a ClusterNode,next:Option<(&'a str,SocketAddr)>)->Option<(&'a str,Result<crate::store_records::StoreRecords,elasticsearch_transport::TransportError>)>{
            match next {Some((id,addr))=>Some(fetch(node,id,addr).await),None=>None}
        }
        loop {
            let first=nodes.next();if first.is_none(){break}
            let results=tokio::join!(optional(self,first),optional(self,nodes.next()),optional(self,nodes.next()),optional(self,nodes.next()));
            for result in [results.0,results.1,results.2,results.3].into_iter().flatten(){
                if let (id,Ok(recs))=result {
                    if let Err(error)=out.try_insert(id,recs){tracing::warn!(%error,"store cache admission refused");return;}
                }
            }
        }
        *self.master.stores.lock() = out;
    }

    async fn refresh_cluster_info(&self, state: &ClusterState) {
        let mut info = elasticsearch_cluster::allocation::ClusterInfo::default();
        for n in state.data_nodes() {
            if let Ok(Some(d)) = crate::rest::nodes::fetch_disk(self, &n.id).await {
                info.nodes.insert(n.id.clone(), d);
            }
        }
        *self.master.cluster_info.lock() = info;
    }
}

/// In-place retries of a peer recovery before the shard is failed to the
/// master (about five minutes with the 5s backoff cap).
const RECOVERY_MAX_RETRIES: u32 = 60;
/// Error of a recovery task whose target copy was removed or rebuilt.
const STALE_RECOVERY: &str = "recovery target copy was removed from this node";
/// Attempts for a shard-started / shard-failed report (500 ms apart).
const REPORT_MAX_RETRIES: u32 = 40;

/// What the master answers a term-checked failure report from a reporter
/// whose primary term is no longer current (`master::shard_failed_task`).
const STALE_REPORTER_TERM: &str = "did not match current primary term";

/// Whether a shard-failed report about copy `aid` still matters: the copy is
/// assigned (on any node: a primary reports its replicas) and, for a
/// term-checked report, the reporter's primary term is still the shard's.
fn failure_report_relevant(state: &ClusterState, index: &str, shard: u32, aid: &str, reporter_term: Option<u64>) -> bool {
    let assigned = state.routing_table.shard_table(index, shard)
        .is_some_and(|t| t.copies.iter().any(|c| c.allocation_id() == Some(aid) && c.state != ShardState::Unassigned));
    assigned && reporter_term.is_none_or(|term| state.metadata.indices.get(index).is_some_and(|im| im.primary_term(shard) == term))
}

/// Build a shard report, waiting while the breaker refuses its memory (a
/// recovery's transient reservations, a node near its limit) and the copy
/// still needs it (`relevant`). `None`: the copy no longer needs a report.
async fn admit_report<R>(index: &str, mut build: impl FnMut() -> elasticsearch_engine::error::Result<R>, relevant: impl Fn() -> bool) -> Option<R> {
    let mut wait = Duration::from_millis(100);
    let mut warned = false;
    loop {
        match build() {
            Ok(req) => return Some(req),
            Err(error) => {
                if !warned {
                    tracing::warn!(%index, %error, "shard report admission refused; retrying until memory frees");
                    warned = true;
                }
            }
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(Duration::from_secs(2));
        if !relevant() {
            return None;
        }
    }
}

/// Recovery failures that say "the source is unreachable or not ready", as
/// opposed to "this recovery cannot work": the transport's connect /
/// disconnect / timeout errors, the source not (yet, or no longer) holding
/// the primary, and a breaker refusing a batch in transit (the transport
/// decoding a recovery request or response: the replay itself is never
/// refused). Those are retried in place with backoff rather than failing
/// the copy, or making a reused copy fall back to a full file recovery.
fn recovery_retryable(e: &str) -> bool {
    use crate::replication::file_recovery as fr;
    ["connect_exception", "] disconnected", "] timed out after [", "circuit_breaking_exception", "transport service is closed", "shard_not_found_exception", "primary node not in cluster state",
        fr::STAGING_BUSY, fr::INSUFFICIENT_DISK, fr::CHUNK_CORRUPT, fr::FILE_CORRUPT, fr::INJECTED_INTERRUPTION, RECOVERY_SOURCE_CHANGED,
        // A file recovery's source failed over or moved: start again from
        // the current primary (the received files are kept).
        "recovery primary allocation or term changed", "recovery primary moved"].iter().any(|p| e.contains(p))
}

/// How long a file recovery waits for the copy it replaces to release its
/// shard lock (it closes in the background) before failing the new copy.
const SET_ASIDE_LOCK_WAIT: Duration = Duration::from_secs(120);

/// `pending` key of a shard whose old store is being set aside for a file
/// recovery (see `ClusterNode::set_aside_then_recover_files`). Per shard,
/// not per allocation: a re-allocation must not reset the same store
/// (deleting the store set aside, or closing the other's placeholder)
/// while the earlier set-aside runs.
fn files_pending_key(index: &str, shard: u32) -> String {
    format!("files:{index}:{shard}")
}

/// Pending key of an existing-store primary being opened off the applier
/// ([`ClusterNode::open_primary_then_start`]): while it is held, no other
/// reconciliation starts or wipes this shard's copy here.
fn primary_open_pending_key(index: &str, shard: u32) -> String {
    format!("primary-open:{index}:{shard}")
}

/// Whether opening a local copy failed because memory was refused (a
/// circuit breaker, the shared mutable-vector budget): transient, so the
/// open is retried. It never proves the copy unusable, and must never make
/// the node discard a valid local copy for a full file recovery.
fn memory_refusal(e: &str) -> bool {
    ["circuit_breaking_exception", "exceeds shared budget", "Data too large"].iter().any(|p| e.contains(p))
}

/// A store file refused for memory (`node.store.allow_mmap: false`; see
/// `rustic_core::codec::directory::read_whole`).
fn store_file_refusal(e: &str) -> bool {
    e.contains("cannot read store file") && e.contains("[node.store.allow_mmap] is false")
}

/// Run `open` (a local open of copy `shard` of `svc`). When it fails after
/// opening the copy itself (an engine copy that was not open before it ran),
/// close that copy again: its data stays on disk, its memory and file
/// handles are released, and no later exit path (the allocation moved on,
/// a retry, a fallback recovery) is left holding an orphan open copy. A
/// copy that was already open before (another owner's) is never closed.
fn release_open_on_failure<T>(svc: &IndexService, shard: u32, open: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let before = svc.shard(shard);
    let result = open();
    if result.is_err() {
        if let Some(copy) = svc.shard(shard) {
            if before.as_ref().is_none_or(|b| !Arc::ptr_eq(b, &copy)) {
                if let Err(e) = svc.close_shard_if_same(&copy) {
                    tracing::warn!(index = %svc.name, shard, "failed to close a copy whose open failed: {}", e.reason);
                }
            }
        }
    }
    result
}

/// Attempts of a local open refused for memory before the copy is
/// reported failed (its data kept; the master allocates it again):
/// backoff 1 s doubling to 30 s, about 10 minutes in all.
const LOCAL_OPEN_MEMORY_RETRIES: u32 = 24;

fn local_open_backoff(attempt: u32) -> Duration {
    Duration::from_millis((1000u64 << attempt.min(5)).min(30_000))
}

/// A file recovery noticed that its copy's primary term or allocation
/// changed between chunks.
const RECOVERY_SOURCE_CHANGED: &str = "recovery source changed (primary term or allocation)";

/// What every chunk request of one file recovery carries.
struct FileChunkBase<'a> {
    index: &'a str,
    shard: u32,
    recovery_id: u64,
    aid: &'a str,
    source_allocation_id: &'a str,
    term: u64,
    addr: SocketAddr,
}

/// Chunk bytes this process received in file recoveries (tests and
/// benchmarks: a resumed recovery receives each byte once).
#[doc(hidden)]
pub static FILE_CHUNK_BYTES_RECEIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One chunk, re-requested (at most 3 times) when it arrives damaged: a
/// wrong length, offset or checksum (ES fails the recovery; a transient
/// corruption on the wire should not restart it).
async fn fetch_chunk(transport: TransportService, addr: SocketAddr, req: crate::replication::file_recovery::FileChunkReq) -> Result<crate::replication::file_recovery::FileChunkResp, String> {
    use crate::replication::file_recovery::{self as fr, FileChunkResp};
    fr::inject_interruption()?;
    for attempt in 0..3 {
        let mut resp: FileChunkResp = transport.send_request(addr, fr::ACTION_FILE_CHUNK, &req, Duration::from_secs(120)).await
            .map_err(|e| format!("file chunk [{}] at [{}]: {e}", req.path, req.offset))?;
        fr::inject_corruption(resp.data.as_mut_slice());
        if resp.offset == req.offset && resp.data.len() as u64 == req.len && elasticsearch_engine::rustic_core::codec::crc32(&resp.data) == resp.crc32 {
            return Ok(resp);
        }
        tracing::warn!(file = %req.path, offset = req.offset, attempt, "a file chunk arrived damaged; requesting it again");
    }
    Err(format!("{}: [{}] at offset [{}] after 3 attempts", fr::CHUNK_CORRUPT, req.path, req.offset))
}

/// May a local copy in the replica role take the primary role under this
/// routing entry? Only once the master *started* it as primary. A primary
/// relocation target is routed `primary` + `INITIALIZING` for the whole
/// recovery; promoting it then makes a partly recovered copy assign sequence
/// numbers the real primary already used and acknowledge writes the real
/// primary never sees (FAULT_RESULTS.md section 10, majority-minority seed 1).
fn replica_may_promote(routing_primary: bool, state: &ShardState) -> bool {
    routing_primary && *state != ShardState::Initializing
}

/// Drop the expired reader contexts (scrolls, PITs, folds,
/// query_then_fetch) of `engines`: they live in the engine of their shard,
/// so the cluster's engine maintenance passes engine 0 and every shard
/// engine.
///
/// The reaped contexts are dropped off the maintenance task and outside the
/// engines' locks: the reader views they pin can be the last references to
/// whole segments.
fn reap_reader_contexts<'a>(engines: impl Iterator<Item = &'a Engine>, now: Instant) {
    use elasticsearch_engine::node::take_dead;
    let mut dead: Vec<Box<dyn Send>> = Vec::new();
    for engine in engines {
        dead.push(Box::new(take_dead(&mut engine.scrolls.lock(), |s| s.expires > now)));
        dead.push(Box::new(take_dead(&mut engine.pits.lock(), |p| p.expires > now)));
        dead.push(Box::new(take_dead(&mut engine.fold_contexts.lock(), |p| p.expires > now)));
        dead.push(Box::new(engine.reap_fetch_contexts(now)));
    }
    match tokio::runtime::Handle::try_current() {
        Ok(rt) => drop(rt.spawn_blocking(move || drop(dead))),
        Err(_) => drop(dead),
    }
}

#[cfg(test)]
mod reader_context_tests {
    use super::*;

    /// Every engine handed in is reaped, not only the first.
    #[test]
    fn reap_reader_contexts_reaps_every_engine() {
        let dirs = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
        let engines: Vec<Arc<Engine>> = dirs.iter().map(|d| Engine::open(d.path()).unwrap()).collect();
        let past = Instant::now();
        let owner = elasticsearch_engine::node::FetchOwner { principal: None, index: None, shard: None };
        // A shard engine charges its contexts to engine 0's budget.
        *engines[1].fetch_budget.lock() = engines[0].fetch_budget.lock().clone();
        for e in &engines {
            e.scrolls.lock().insert("s".into(), elasticsearch_engine::node::ScrollCtx {
                targets: Vec::new(), req: elasticsearch_engine::search::SearchRequest::default(), cursors: Vec::new(), keep_alive: Duration::from_secs(1), expires: past, accounting: None, lost: Vec::new(),
            });
            e.pits.lock().insert("p".into(), elasticsearch_engine::node::PitCtx { targets: Vec::new(), keep_alive: Duration::from_secs(1), expires: past, accounting: None, lost: Vec::new() });
            e.fold_contexts.lock().insert("f".into(), elasticsearch_engine::node::PitCtx { targets: Vec::new(), keep_alive: Duration::from_secs(1), expires: past, accounting: None, lost: Vec::new() });
            assert!(e.register_fetch_context("q".into(), Vec::new(), owner.clone()));
        }
        assert_eq!(engines[0].fetch_budget.lock().held().0, 2);
        reap_reader_contexts(engines.iter().map(|e| &**e), past + elasticsearch_engine::node::FETCH_CTX_KEEP_ALIVE + Duration::from_secs(1));
        assert_eq!(engines[0].fetch_budget.lock().held(), (0, 0));
        for e in &engines {
            assert!(e.scrolls.lock().is_empty() && e.pits.lock().is_empty() && e.fold_contexts.lock().is_empty() && e.fetch_contexts.lock().is_empty());
        }
    }
}

#[cfg(test)]
mod replica_prefix_authority_tests {
    use super::*;
    use elasticsearch_cluster::state::{AllocationId, IndexRoutingTable, IndexShardRoutingTable};

    #[test]
    fn replica_prefix_recovery_requires_history_incarnation_and_current_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        engine.create_index("idx", &serde_json::json!({})).unwrap();
        let svc = engine.get_index("idx").unwrap();
        let p = PrimaryCtx::new("idx", 0, "primary", 1, svc).unwrap();
        let mut state = ClusterState::empty("test");
        state.master_node_id = Some("source-node".into());
        state.metadata = mirror::extract(&engine, &state.metadata, &mut UuidMap::default());
        state.metadata.cluster_uuid = "cluster-history".into();
        let im = state.metadata.indices.get_mut("idx").unwrap();
        im.in_sync_allocation_ids.insert(0, ["primary".to_string(), "old-replica".to_string()].into_iter().collect());
        let index_uuid = im.uuid.clone();
        let route = |primary: bool, aid: &str, node: &str, state| ShardRouting {
            index: "idx".into(), shard: 0, primary, state, node_id: Some(node.into()), relocating_node_id: None,
            allocation_id: Some(AllocationId { id: aid.into(), relocation_id: None }), recovery_source: None,
            unassigned_info: None, expected_shard_size_in_bytes: None,
        };
        state.routing_table.indices.insert("idx".into(), IndexRoutingTable { index_uuid: index_uuid.clone(), shards: vec![IndexShardRoutingTable {
            shard: 0, copies: vec![route(true, "primary", "source-node", ShardState::Started), route(false, "new-replica", "target-node", ShardState::Initializing)],
        }] });
        let mut request = RecoveryReuseReq::try_new("idx", 0, "cluster-history", &index_uuid,
            "old-replica", "new-replica", "target-node", "primary", 1,
            ReplicaPrefix { checkpoint: -1, global_checkpoint: -1, primary_term: 1 }, None).unwrap();
        assert!(reuse_request_matches(&state, "source-node", &p, &request));
        assert!(recovery_start_matches(&state, "source-node", &p, "new-replica", Some("target-node")));
        assert!(!recovery_start_matches(&state, "source-node", &p, "new-replica", Some("wrong-node")));
        let mut moved = state.clone();
        moved.routing_table.indices.get_mut("idx").unwrap().shards[0].copies[1].allocation_id.as_mut().unwrap().id = "replacement-target".into();
        assert!(!recovery_start_matches(&moved, "source-node", &p, "new-replica", None));
        let mut promoted = state.clone();
        promoted.routing_table.indices.get_mut("idx").unwrap().shards[0].copies[0].allocation_id.as_mut().unwrap().id = "replacement-primary".into();
        assert!(!recovery_start_matches(&promoted, "source-node", &p, "new-replica", None));
        let mut stale = state.clone();
        stale.metadata.indices.get_mut("idx").unwrap().in_sync_allocation_ids.get_mut(&0).unwrap().remove("old-replica");
        assert!(!reuse_request_matches(&stale, "source-node", &p, &request), "same index UUID does not prove history");
        // ES `isTargetSameHistory`: a copy out of the in-sync set (its node
        // restarted) shares the primary's history uuid.
        shards::write_copy_record(&p.svc, &CopyRecord { index: "idx".into(), index_uuid: request.index_uuid.clone(), shard: 0, allocation_id: "primary".into(),
            primary_term: 1, primary: true, history_uuid: Some("history-1".into()), global_checkpoint: None }).unwrap();
        request.history_uuid = Some("history-1".into());
        assert!(reuse_request_matches(&stale, "source-node", &p, &request), "same history uuid");
        request.history_uuid = Some("history-0".into());
        assert!(!reuse_request_matches(&stale, "source-node", &p, &request), "a forced stale primary starts a new history");
        request.history_uuid = None;
        let mut changed = state.clone();
        changed.metadata.indices.get_mut("idx").unwrap().uuid = "recreated-index".into();
        assert!(!reuse_request_matches(&changed, "source-node", &p, &request));
        request.primary_term = 2;
        assert!(!reuse_request_matches(&state, "source-node", &p, &request));
        request.primary_term = 1;
        request.source_allocation_id = "replacement-primary".into();
        assert!(!reuse_request_matches(&state, "source-node", &p, &request));
        request.source_allocation_id = "primary".into();
        request.target_allocation_id = "stale-target".into();
        assert!(!reuse_request_matches(&state, "source-node", &p, &request));
    }
}

#[cfg(test)]
mod vector_optimizer_lease_tests {
    use super::VectorOptimizerLease;
    use std::sync::{atomic::{AtomicBool, Ordering}, Arc};

    #[test]
    fn vector_optimizer_slot_is_exclusive_and_released_after_panic() {
        let running = Arc::new(AtomicBool::new(false));
        let failure = std::panic::catch_unwind(|| {
            let _lease = VectorOptimizerLease::claim(&running).expect("first round owns the slot");
            assert!(running.load(Ordering::Acquire));
            assert!(VectorOptimizerLease::claim(&running).is_none(), "rounds must not overlap");
            panic!("simulate failed graph build");
        });
        assert!(failure.is_err());
        assert!(!running.load(Ordering::Acquire));
        let lease = VectorOptimizerLease::claim(&running).expect("next tick can retry");
        assert!(running.load(Ordering::Acquire));
        drop(lease);
        assert!(!running.load(Ordering::Acquire));
    }
}

#[cfg(test)]
mod promotion_tests {
    use super::{replica_may_promote, ShardState};

    #[test]
    fn initializing_relocation_target_keeps_the_replica_role() {
        assert!(!replica_may_promote(true, &ShardState::Initializing));
    }

    #[test]
    fn started_primary_is_promoted_and_replicas_never_are() {
        assert!(replica_may_promote(true, &ShardState::Started));
        assert!(!replica_may_promote(false, &ShardState::Started));
        assert!(!replica_may_promote(false, &ShardState::Initializing));
    }
}

#[cfg(test)]
mod recovery_retry_tests {
    use super::recovery_retryable;

    #[test]
    fn failed_synonym_recovery_cannot_open_a_cluster_copy() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        let service = engine.create_index_sparse("broken", &elasticsearch_engine::index::new_uuid(), &serde_json::json!({
            "settings": {"analysis": {
                "filter": {"syn": {"type":"synonym", "synonyms_set":"missing", "updateable":true, "lenient":false}},
                "analyzer": {"query": {"type":"custom", "tokenizer":"whitespace", "filter":["syn"]}}
            }},
            "mappings": {"properties": {"text": {"type":"text", "search_analyzer":"query"}}}
        })).unwrap();
        assert!(super::ClusterNode::ensure_copy_open(&service, 0, "allocation").is_err());
        assert!(service.shard(0).is_none());
    }

    #[test]
    fn network_and_stale_source_errors_are_retried() {
        // exactly what seed 2 kill-primary produced, five times in 270ms
        assert!(recovery_retryable("[127.0.96.2:9701] connect_exception: Connection refused (os error 111)"));
        assert!(recovery_retryable("[127.0.96.2:9701][internal:index/shard/recovery/start] disconnected"));
        assert!(recovery_retryable("[127.0.96.2:9701][a] request_id [7] timed out after [60000ms]"));
        assert!(recovery_retryable("[shard_not_found_exception] primary moved"));
        assert!(recovery_retryable("primary node not in cluster state"));
        // A batch refused by a breaker while decoded on either side.
        assert!(recovery_retryable("[circuit_breaking_exception] [request] Data too large, data for [decoded replica operations] would be [9000], which is larger than the limit of [8000]"));
        assert!(recovery_retryable("translog batch: [circuit_breaking_exception] [parent] Data too large"));
    }

    #[test]
    fn real_recovery_failures_are_reported() {
        assert!(!recovery_retryable("[recovery_failed_exception] unknown recovery"));
        assert!(!recovery_retryable("timed out waiting to be marked in sync"));
        assert!(!recovery_retryable("io: No such file or directory (os error 2)"));
    }
}

/// Whether recovery may (re-)create a missing local index from cluster
/// state: not on the master's engine 0, whose indices *are* the metadata.
/// The base of the next incremental application after applying `state`
/// with `errors`: the failed indices keep their form in `prev` (absent
/// there: absent here, so the next pass creates them). A failure that names
/// no index (stored scripts, pipelines, ...) leaves the globals as applied:
/// reverting them would make every kept job's base stale; `apply_retry`
/// re-applies them instead. The next application, or the retry timer,
/// then applies the failed parts again.
fn mirrored_after(state: &Arc<ClusterState>, prev: Option<&ClusterState>, errors: &[(String, String)]) -> Arc<ClusterState> {
    if errors.is_empty() {
        return state.clone();
    }
    let mut s = (**state).clone();
    let none = elasticsearch_cluster::state::Metadata::default();
    let pm = prev.map_or(&none, |p| &p.metadata);
    let mut touched = false;
    let revert_custom = |s: &mut ClusterState, key: &str| {
        match pm.customs.get(key) {
            Some(v) => s.metadata.customs.insert(key.to_string(), v.clone()),
            None => s.metadata.customs.remove(key),
        };
    };
    for (name, _) in errors {
        if state.metadata.indices.contains_key(name) || pm.indices.contains_key(name) {
            touched = true;
            match pm.indices.get(name) {
                Some(im) => {
                    s.metadata.indices.insert(name.clone(), im.clone());
                }
                None => {
                    s.metadata.indices.remove(name);
                }
            }
        } else if name == CUSTOMS_DEFERRED {
            // Not applied at all: every custom stays at its previous form
            // (the next pass applies them; a kept job meanwhile fails safe
            // with a stale base instead of publishing engine 0's form).
            touched = true;
            // Data streams are applied with the globals regardless (never
            // deferred): their mirrored form stays as applied.
            let keys: Vec<String> = s.metadata.customs.keys().chain(pm.customs.keys()).filter(|k| k.as_str() != crate::mirror::DATA_STREAMS_CUSTOM).cloned().collect();
            for key in keys {
                revert_custom(&mut s, &key);
            }
        } else if let Some(family) = name.strip_prefix(CUSTOMS_DEFERRED).and_then(|r| r.strip_prefix(':')) {
            touched = true;
            revert_custom(&mut s, family);
        } else {
            // That family failed to apply: recorded as not applied.
            for key in crate::meta_extra::custom_of_error(name) {
                touched = true;
                revert_custom(&mut s, key);
            }
        }
    }
    if !touched {
        return state.clone();
    }
    Arc::new(s)
}

#[cfg(test)]
mod mirrored_tests {
    use super::*;

    fn im(name: &str, version: u64) -> elasticsearch_cluster::state::IndexMetadata {
        let mut m: elasticsearch_cluster::state::IndexMetadata = serde_json::from_value(serde_json::json!({
            "name": name, "uuid": format!("u-{name}"), "version": version, "mapping_version": 1, "settings_version": 1,
            "aliases_version": 1, "state": "open", "settings": {}, "mappings": {}, "aliases": {},
            "number_of_shards": 1, "number_of_replicas": 0, "routing_num_shards": 1,
            "primary_terms": [1], "in_sync_allocation_ids": {}, "creation_date": 0
        })).unwrap();
        m.version = version;
        m
    }

    /// Review NB2/NM6: a failed index application does not advance the
    /// mirrored base (a failed delete keeps the index there, a failed create
    /// stays absent), so the next incremental pass retries it.
    #[test]
    fn failed_parts_keep_their_previous_form() {
        let mut prev = ClusterState::empty("c");
        prev.metadata.indices.insert("updated".into(), im("updated", 1));
        prev.metadata.indices.insert("deleted".into(), im("deleted", 1));
        prev.metadata.indices.insert("fine".into(), im("fine", 1));
        let mut next = prev.clone();
        next.metadata.indices.insert("updated".into(), im("updated", 2));
        next.metadata.indices.remove("deleted");
        next.metadata.indices.insert("created".into(), im("created", 1));
        next.metadata.indices.insert("fine".into(), im("fine", 2));
        next.metadata.index_templates.insert("t".into(), serde_json::json!({}));
        let next = Arc::new(next);
        assert!(Arc::ptr_eq(&mirrored_after(&next, Some(&prev), &[]), &next));
        let errors = vec![("updated".to_string(), "x".to_string()), ("deleted".to_string(), "x".to_string()), ("created".to_string(), "x".to_string())];
        let m = mirrored_after(&next, Some(&prev), &errors);
        assert_eq!(m.metadata.indices["updated"].version, 1);
        assert!(m.metadata.indices.contains_key("deleted"), "a failed delete is retried");
        assert!(!m.metadata.indices.contains_key("created"), "a failed create is retried");
        assert_eq!(m.metadata.indices["fine"].version, 2, "what applied advances");
        assert!(m.metadata.index_templates.contains_key("t"));
        // A failure naming no index or custom family leaves the globals as
        // applied; a failed custom family is recorded as not applied.
        let m = mirrored_after(&next, Some(&prev), &[("stored scripts".to_string(), "x".to_string())]);
        assert!(Arc::ptr_eq(&m, &next));
        let mut with_pipeline = (*next).clone();
        with_pipeline.metadata.customs.insert("ingest".into(), serde_json::json!({"pipeline": [{"id": "x", "config": {}}]}));
        let with_pipeline = Arc::new(with_pipeline);
        let m = mirrored_after(&with_pipeline, Some(&prev), &[("pipeline x".to_string(), "x".to_string())]);
        assert!(m.metadata.customs.get("ingest").is_none(), "a failed pipeline apply is not recorded as applied");
    }
}

fn may_recreate_from_state(is_master: bool, shard: u32, engine_has_index: bool) -> bool {
    engine_has_index || !(is_master && shard == 0)
}

/// Whether the local engine index (uuid `local`) is the cluster's index
/// `cluster` (the uuid map entry `(cluster uuid, local uuid)` links them).
fn same_incarnation(mapped: Option<&(String, String)>, cluster: &str, local: &str) -> bool {
    mapped.is_some_and(|(c, l)| c == cluster && l == local)
}

#[cfg(test)]
mod release_open_tests {
    use super::{release_open_on_failure, ClusterNode};
    use elasticsearch_engine::index::IndexService;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn index(path: &std::path::Path) -> Arc<IndexService> {
        let settings = elasticsearch_engine::settings::normalize_index_settings(&serde_json::json!({}));
        Arc::new(IndexService::create(path, "idx", settings, &serde_json::json!({}), BTreeMap::new()).unwrap())
    }

    /// Review LOW (c19): an open that fails after opening the copy (the
    /// allocation moved on, then a non-memory failure) closes that copy; a
    /// successful open keeps it, and a copy open before is never closed.
    #[test]
    fn a_failed_open_closes_only_the_copy_it_opened() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = index(&tmp.path().join("idx"));
        svc.close_shard(0).unwrap();
        assert!(svc.shard(0).is_none());
        let failed: Result<(), String> = release_open_on_failure(&svc, 0, || {
            ClusterNode::ensure_copy_open(&svc, 0, "a")?;
            assert!(svc.shard(0).is_some());
            Err("promotion failed".into())
        });
        assert_eq!(failed.unwrap_err(), "promotion failed");
        assert!(svc.shard(0).is_none(), "the copy the failed open opened is closed");

        let opened: Result<(), String> = release_open_on_failure(&svc, 0, || ClusterNode::ensure_copy_open(&svc, 0, "a"));
        assert!(opened.is_ok());
        let held = svc.shard(0).expect("a successful open keeps its copy");
        let failed: Result<(), String> = release_open_on_failure(&svc, 0, || Err("refused".into()));
        assert!(failed.is_err());
        assert!(svc.shard(0).is_some_and(|c| Arc::ptr_eq(&c, &held)), "a copy open before the attempt stays open");
    }
}

#[cfg(test)]
mod incarnation_tests {
    use super::same_incarnation;

    #[test]
    fn the_master_never_resurrects_an_index_it_deleted() {
        use super::may_recreate_from_state;
        // Master, engine 0, index gone (deleted after the state was published).
        assert!(!may_recreate_from_state(true, 0, false));
        // Other engines/nodes mirror the published metadata.
        assert!(may_recreate_from_state(true, 1, false));
        assert!(may_recreate_from_state(false, 0, false));
        assert!(may_recreate_from_state(true, 0, true));
    }

    #[test]
    fn only_the_mapped_incarnation_is_reset_in_place() {
        let map = ("cluster-u1".to_string(), "local-a".to_string());
        assert!(same_incarnation(Some(&map), "cluster-u1", "local-a"));
        // Re-created under the same name (new cluster uuid) or a different local copy.
        assert!(!same_incarnation(Some(&map), "cluster-u2", "local-a"));
        assert!(!same_incarnation(Some(&map), "cluster-u1", "local-b"));
        assert!(!same_incarnation(None, "cluster-u1", "local-a"));
    }
}

impl ClusterNode {
    /// Follow a committed state's segment format floor: new files are
    /// written in it from here on (each file keeps the version its writer
    /// captured). Every member reads the floor, so a raise is safe to act on
    /// at once. The floor is recorded in the gateway for the next start
    /// (only states the applier receives are committed); if that write
    /// fails the next start uses the older committed floor, which is still
    /// safe, and the next applied state retries it.
    fn apply_segment_format(&self, state: &ClusterState) {
        use elasticsearch_engine::rustic_core::codec;
        let Some(floor) = elasticsearch_cluster::compatibility::segment_format_floor(&state.metadata) else { return };
        let uuid = &state.metadata.cluster_uuid;
        let mut recorded = self.committed_segment_format.lock();
        if recorded.as_ref().is_some_and(|(at, applied)| at == uuid && *applied >= floor) { return; }
        let write = codec::apply_cluster_segment_format(floor);
        match gateway::store_committed_segment_format(&self.config.data_path, uuid, floor) {
            Ok(()) => *recorded = Some((uuid.clone(), floor)),
            Err(error) => tracing::warn!(floor, %error, "could not record the committed segment format floor"),
        }
        tracing::info!(floor, segment_write_version = write, "applied the cluster's segment format");
    }
}

impl Applier for ClusterNode {
    /// Runs on the cluster applier thread (`coord`), never on the
    /// coordination actor.
    fn apply(&self, state: Arc<ClusterState>) {
        let started = std::time::Instant::now();
        let _metadata = self.metadata_mutex.lock();
        let _g = self.apply_lock.lock();
        let acquired = started.elapsed();
        let has_master = state.master_node_id.is_some();
        // A remote cluster seeded with any member's address is this cluster.
        let members: Vec<String> = state.nodes.values().map(|n| n.transport_address.clone()).collect();
        if *self.engine.cluster_transport_addresses.read() != members {
            *self.engine.cluster_transport_addresses.write() = members.clone();
            for (_, se) in self.shard_engines.all() {
                *se.engine.cluster_transport_addresses.write() = members.clone();
            }
        }
        if has_master {
            self.apply_segment_format(&state);
            self.apply_state_metadata(&state, false);
        }
        drop(_metadata);
        // `send` would drop the value when nobody is subscribed.
        self.state_tx.send_replace(state.clone());
        let metadata_applied = started.elapsed();
        if has_master {
            self.reconcile_shards(&state);
        }
        // Fully applied (metadata and shards): what acknowledgements wait for.
        self.applied_version.fetch_max(state.version, std::sync::atomic::Ordering::SeqCst);
        if started.elapsed() >= Duration::from_secs(5) {
            tracing::warn!(version=state.version,
                wait_ms=acquired.as_millis() as u64,
                metadata_ms=metadata_applied.saturating_sub(acquired).as_millis() as u64,
                reconcile_ms=started.elapsed().saturating_sub(metadata_applied).as_millis() as u64,
                "slow cluster state application");
        }
    }

    fn mode_changed(&self, mode: Mode, term: u64, _leader: Option<&DiscoveryNode>) {
        *self.mode.lock() = (mode, term);
    }

    fn publication_failed(&self) {
        // Roll the master's engine back to the applied state, on the
        // metadata service (it holds the engine's metadata lock).
        self.metadata_resync.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(service) = self.metadata_service.get() {
            service.submit(crate::master_service::Job::Run { run: Box::new(|_: &ClusterNode, _: crate::master_service::On| false), done: Box::new(|_: Result<u64, String>| {}), dry_run: false, before: None, unpublished: None });
        }
    }
}

/// [`ReplicationNet`] over the transport.
pub struct TransportNet(pub Arc<ClusterNode>);

#[async_trait::async_trait]
impl ReplicationNet for TransportNet {
    async fn replicate(&self, target: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
        let addr: SocketAddr = target.address.parse().map_err(|_| format!("bad address [{}]", target.address))?;
        let transport = &self.0.transport;
        // ES sends replica operations only over the connection opened when
        // the node joined; a node that is not connected fails the op at once
        // (`NodeNotConnectedException`) and the copy goes through the
        // fail-shard path. Connections here are opened on demand, so the
        // dial gets the same short bound as coordination messages instead
        // of the 30s `transport.connect_timeout`: the caller holds the
        // shard's replication pump, and a dead replica whose address
        // black-holes SYNs (killed container, vanished host) must not stall
        // every write on the shard. Once connected, a hung replica is still
        // bounded only by the request timeout below.
        if !transport.is_connected(addr) {
            match tokio::time::timeout(coord::CONNECT_GRACE, transport.connect(addr)).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(e.to_remote_error().to_string()),
                Err(_) => {
                    return Err(format!(
                        "[connect_transport_exception] [{}][{addr}] connect_exception: node not connected, connect timed out after {:?}",
                        target.node_id,
                        coord::CONNECT_GRACE
                    ))
                }
            }
        }
        // "[type] reason": `_shards.failures` reports the type and reason.
        transport.send_request(addr, shards::ACTION_REPLICATE, &req, Duration::from_secs(60)).await.map_err(|e| e.to_remote_error().to_string())
    }

    async fn refresh_replica(&self, target: &Target, req: shards::ReplicaRefreshReq) -> Result<(), String> {
        let addr: SocketAddr = target.address.parse().map_err(|_| format!("bad address [{}]", target.address))?;
        self.0.transport.send_request::<_, ()>(addr, shards::ACTION_REPLICA_REFRESH, &req, Duration::from_secs(60)).await.map_err(|e| e.to_remote_error().to_string())
    }

    async fn fail_copy(&self, index: &str, shard: u32, allocation_id: &str, primary_term: u64, reason: &str) -> Result<(), String> {
        let req = ShardFailedReq::try_new(index,shard,allocation_id,Some(primary_term),reason,true).map_err(|error|error.to_string())?;
        self.0.to_master::<_, ()>(ACTION_SHARD_FAILED, &req, Duration::from_secs(30)).await.map_err(|e| format!("{}: {}", e.error_type, e.reason))
    }
}

// ES9's dynamic flush policy is separate from request fsync durability.
fn configured_translog_flush_threshold(value: Option<&str>, disk_bytes: u64) -> u64 {
    let configured = value.and_then(|v| elasticsearch_engine::ingest::util::parse_byte_size_value(v, "index.translog.flush_threshold_size").ok())
        .and_then(|v| u64::try_from(v).ok()).filter(|v| *v >= 56).unwrap_or(10 << 30);
    if disk_bytes == 0 { configured } else { configured.min((disk_bytes / 100).max(56)) }
}
fn translog_flush_threshold(copy: &elasticsearch_engine::index::IndexShard) -> u64 {
    configured_translog_flush_threshold(copy.setting("index.translog.flush_threshold_size").as_deref(),
        crate::stats::disk_usage(copy.shard.path()).total_bytes)
}
fn translog_age_threshold_millis(copy: &elasticsearch_engine::index::IndexShard) -> u64 {
    copy.setting("index.translog.flush_threshold_age")
        .and_then(|v| elasticsearch_engine::date::parse_time_value(&v)).unwrap_or(60_000).max(1000) as u64
}

/// ES's `index.translog.flush_threshold_age`: the copy is due once its
/// uncommitted operations are older than the threshold.
fn translog_age_flush_due(copy: &elasticsearch_engine::index::IndexShard) -> bool {
    let millis = translog_age_threshold_millis(copy);
    copy.shard.elapsed_since_flush() >= std::time::Duration::from_millis(millis)
        && copy.shard.max_seq_no() > copy.shard.committed_seq_no()
}

/// When the background flush takes up an age-due copy: its own slot
/// after the threshold ([`age_flush_jitter`]), so copies written together
/// do not all flush on one tick. Whether a copy is due stays ES's
/// [`translog_age_flush_due`].
fn translog_age_flush_slot(copy: &elasticsearch_engine::index::IndexShard) -> bool {
    let millis = translog_age_threshold_millis(copy);
    translog_age_flush_due(copy) && copy.shard.elapsed_since_flush() >= std::time::Duration::from_millis(millis + age_flush_jitter(&copy.uuid, copy.id, millis))
}

/// ES `index.flush_after_merge` (default 512mb): once merges replaced at
/// least this much of the last commit, the copy flushes (ES flushes on the
/// next operation after such a merge), freeing the merged-away files.
fn merge_flush_due(copy: &elasticsearch_engine::index::IndexShard) -> bool {
    const FLUSH_AFTER_MERGE_BYTES: u64 = 512 * 1024 * 1024;
    copy.shard.merged_away_committed_bytes() >= FLUSH_AFTER_MERGE_BYTES
}

/// A per-copy delay (up to a quarter of the age threshold, fixed for the
/// copy) so copies written together do not all reach their age flush on
/// the same tick.
fn age_flush_jitter(index_uuid: &str, shard: u32, threshold_millis: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in index_uuid.bytes().chain(shard.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h % (threshold_millis / 4).max(1)
}

#[cfg(test)]
mod translog_flush_policy_tests {
    use super::*;
    #[test]
    fn age_flush_jitter_is_fixed_per_copy_and_bounded() {
        let a = age_flush_jitter("u1", 0, 60_000);
        assert_eq!(a, age_flush_jitter("u1", 0, 60_000), "deterministic");
        assert!(a < 15_000);
        let spread: std::collections::BTreeSet<u64> = (0..16).map(|k| age_flush_jitter("u1", k, 60_000)).collect();
        assert!(spread.len() > 8, "copies are spread: {spread:?}");
    }

    #[test]
    fn es9_translog_threshold_default_override_and_disk_cap() {
        assert_eq!(configured_translog_flush_threshold(None, 2 << 40), 10 << 30);
        assert_eq!(configured_translog_flush_threshold(Some("64mb"), 2 << 40), 64 << 20);
        assert_eq!(configured_translog_flush_threshold(Some("2GB"), 100 << 30), (100u64 << 30) / 100);
        assert_eq!(configured_translog_flush_threshold(None, 0), 10 << 30);
    }
    #[test]
    fn es9_translog_threshold_tracks_dynamic_settings_and_reset() {
        use elasticsearch_engine::settings::normalize_index_settings;
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let svc = engine.create_index("flush-policy", &serde_json::json!({})).unwrap();
        let copy = svc.shard(0).unwrap();
        let initial = translog_flush_threshold(&copy);
        for bytes in ["2kb", "4kb"] {
            svc.put_settings(normalize_index_settings(&serde_json::json!({"index.translog.flush_threshold_size":bytes})), false).unwrap();
            assert_eq!(translog_flush_threshold(&copy), if bytes == "2kb" { 2048 } else { 4096 });
        }
        assert!(svc.put_settings(normalize_index_settings(&serde_json::json!({"index.translog.flush_threshold_size":"55b"})), false).is_err());
        assert!(svc.put_settings(normalize_index_settings(&serde_json::json!({"index.translog.flush_threshold_age":"0s"})), false).is_err());
        svc.put_settings(normalize_index_settings(&serde_json::json!({"index.translog.flush_threshold_size":null,"index.translog.flush_threshold_age":"1s"})), false).unwrap();
        assert_eq!(translog_flush_threshold(&copy), initial);
        assert!(!translog_age_flush_due(&copy)); // An empty shard never needs age flushing.
        svc.write(vec![elasticsearch_engine::index::WriteReq::Index {
            id: Some("age-probe".into()), source: "{\"x\":1}".into(), routing: None,
            cond: elasticsearch_engine::shard::WriteCond::default(),
        }], true).remove(0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(translog_age_flush_due(&copy));
        copy.shard.flush().unwrap();
        assert!(!translog_age_flush_due(&copy));
    }
}


/// Whether a node leads for state-dependent work: it is the applied state's
/// master, in leader mode, and the applied state is of its current term.
fn leads(state: &ClusterState, local_id: &str, mode: Mode, term: u64) -> bool {
    mode == Mode::Leader && state.master_node_id.as_deref() == Some(local_id) && state.coordination.term == term
}

#[cfg(test)]
mod leadership_tests {
    use super::*;

    #[test]
    fn a_re_elected_node_does_not_lead_on_a_state_of_an_older_term() {
        let mut st = ClusterState::empty("c");
        st.master_node_id = Some("me".into());
        st.coordination.term = 5;
        assert!(leads(&st, "me", Mode::Leader, 5));
        assert!(!leads(&st, "me", Mode::Leader, 7), "elected in term 7, applier still on term 5");
        assert!(!leads(&st, "me", Mode::Candidate, 5));
        assert!(!leads(&st, "other", Mode::Leader, 5));
    }
}

#[cfg(test)]
mod wire_compat_tests {
    use super::*;

    /// Review P2-3: `MappingUpdateReq` across versions, both directions
    /// (bincode, the transport's codec).
    #[test]
    fn mapping_update_requests_decode_across_versions() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct Old {
            index: String,
            mapping_json: String,
        }
        let old = Old { index: "i".into(), mapping_json: "{\"properties\":{}}".into() };
        // Old sender, new receiver: no uuid (merged without the incarnation check).
        let new: MappingUpdateReq = bincode::deserialize(&bincode::serialize(&old).unwrap()).unwrap();
        assert_eq!((new.index.as_str(), new.mapping_json.as_str(), new.index_uuid), ("i", "{\"properties\":{}}", None));
        // New sender, old receiver: the appended uuid is ignored.
        let sent = MappingUpdateReq::try_new("i", "{}", Some("u")).unwrap();
        let seen: Old = bincode::deserialize(&bincode::serialize(&sent).unwrap()).unwrap();
        assert_eq!(seen, Old { index: "i".into(), mapping_json: "{}".into() });
        // New to new keeps it.
        let back: MappingUpdateReq = bincode::deserialize(&bincode::serialize(&sent).unwrap()).unwrap();
        assert_eq!(back.index_uuid.as_deref(), Some("u"));
        // JSON (the coordination v1 path) both ways.
        let back: MappingUpdateReq = serde_json::from_str(&serde_json::to_string(&old).unwrap()).unwrap();
        assert_eq!(back.index_uuid, None);
    }

    /// Masters upgrade last: while the elected master does not advertise
    /// UUID-fenced mapping updates, a dynamic field goes out in the original
    /// action, still carrying the index UUID (and the exact delta) for that
    /// master to fence on, instead of failing every new field with a 503.
    #[test]
    fn dynamic_mapping_updates_fall_back_to_the_fenced_original_action_for_an_older_master() {
        let mut master = DiscoveryNode { id: "m".into(), name: "m".into(), ephemeral_id: "e".into(),
            transport_address: "127.0.0.1:9300".into(), http_address: "127.0.0.1:9200".into(),
            host: "localhost".into(), ip: "127.0.0.1".into(), roles: Default::default(),
            attributes: Default::default(), version: "9.5.5".into() };
        assert_eq!(mapping_update_action(&master), ACTION_MAPPING_UPDATE);
        crate::internal_capabilities::advertise(&mut master.attributes);
        assert_eq!(mapping_update_action(&master), ACTION_MAPPING_UPDATE_V2);
        master.attributes.remove("rustic.internal.mapping_uuid_v2");
        assert_eq!(mapping_update_action(&master), ACTION_MAPPING_UPDATE, "a master without the capability takes the original action");
        // What the original action's handler decodes keeps the fence and the delta.
        let mut sent = MappingUpdateReq::try_new("i", "{\"properties\":{\"f\":{\"type\":\"keyword\"}}}", Some("u")).unwrap();
        sent.dynamic_delta = Some("{\"properties\":{\"f\":{\"type\":\"keyword\"}}}".into());
        let seen: MappingUpdateReq = bincode::deserialize(&bincode::serialize(&sent).unwrap()).unwrap();
        assert_eq!(seen.index_uuid.as_deref(), Some("u"));
        assert_eq!(seen.dynamic_delta, sent.dynamic_delta);
    }
}

/// ES's transport `inflight_requests` accounting: every inbound request
/// payload is held on the breaker while its handler runs, without
/// tripping: internal cluster actions (coordination, replication,
/// recovery, monitoring) never trip, as ES registers them, and a forwarded
/// REST request trips by its own REST policy in the forward handler
/// (`front::forward_trips`), not by action name.
fn inbound_gate(_action: &str, bytes: usize) -> Result<Box<dyn std::any::Any + Send>, elasticsearch_transport::RemoteError> {
    use elasticsearch_engine::memory::{process_breakers, Kind};
    Ok(Box::new(process_breakers().reserve_without_breaking(Kind::InflightRequests, bytes as u64)))
}

#[cfg(test)]
mod engine_api_held_delete_tests {
    use super::*;
    use serde_json::json;

    async fn single_node(dir: &std::path::Path) -> Arc<ClusterNode> {
        let mut settings = crate::config::Settings::new();
        for (key, value) in [
            ("cluster.name", json!("held-delete")),
            ("node.name", json!("held-delete")),
            ("path.data", json!(dir.join("data"))),
            ("path.logs", json!(dir.join("logs"))),
            ("network.host", json!("127.0.0.1")),
            ("http.port", json!("0")),
            ("transport.port", json!("0")),
            ("discovery.type", json!("single-node")),
            ("xpack.security.enabled", json!("false")),
        ] { settings.insert(key.into(), value); }
        let (node, listener) = ClusterNode::start(NodeConfig::from_settings(settings).unwrap()).await.unwrap();
        std::mem::forget(listener);
        // `await_own_master_state` answers false at once while the node is
        // not yet leading (it is for a node that already leads), so wait for
        // the election first, as the other single-node tests do.
        for _ in 0..300 {
            if node.is_master() && node.state().nodes.contains_key(&node.local.id) { break; }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(node.await_own_master_state(Duration::from_secs(20)).await, "the single node never became master");
        node
    }

    /// An index created through the metadata service, applied here.
    async fn create(node: &Arc<ClusterNode>, name: &'static str) {
        let ((), version) = node.mutate_engine_metadata(move |engine| engine.create_index(name, &json!({"settings": {"number_of_replicas": 0}})).map(|_| ())).await.unwrap();
        assert!(node.wait_applied_here(version, Duration::from_secs(20)).await);
        assert!(node.engine.get_index(name).is_some());
    }

    /// The `POST /_features/_reset` race: an engine API deletes an index
    /// from engine 0, and a state application (a mirror catch-up of an
    /// unrelated change, here a create; on the cluster, a replica started)
    /// runs before its kept job. The application used to re-create the
    /// shell, so the kept job saw no change and the delete was never
    /// published (the master then held a stale primary). Now the shell stays
    /// absent and the delete is published.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn engine_api_delete_survives_an_application_before_its_kept_job() {
        // The node outlives the test (its runtime is not shut down), so keep
        // its data directory: a deleted one makes cluster-persist exit the process.
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = single_node(dir.path()).await;
        create(&node, "reset-target").await;

        let guard = node.engine_api_enter(None).await;
        let before = node.engine_api_snapshot(&guard).await.expect("snapshot");
        // The engine API (the reset) deletes the index on engine 0 only.
        node.engine.delete_index("reset-target").unwrap();
        // An application lands before the API's kept job.
        create(&node, "unrelated").await;
        let n = node.clone();
        tokio::task::spawn_blocking(move || n.retry_failed_application()).await.unwrap();
        assert!(node.engine.get_index("reset-target").is_none(), "an application re-created the shell an engine API in flight deleted");
        assert!(node.state().metadata.indices.contains_key("reset-target"), "the delete is not published before the kept job");

        let version = node.publish_engine_metadata_since(before).await.expect("the kept job publishes the delete");
        assert!(node.wait_applied_here(version, Duration::from_secs(20)).await);
        drop(guard);
        assert!(!node.state().metadata.indices.contains_key("reset-target"), "the engine API's delete was lost");
        assert!(node.state().metadata.indices.contains_key("unrelated"));
        assert!(node.engine.get_index("reset-target").is_none());
        assert!(!node.apply_retry.lock().contains("reset-target"));
    }

    /// An engine API that deletes an index and then fails publishes
    /// nothing: once it is done the index is applied to engine 0 again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unpublished_engine_api_delete_is_reapplied_after_the_api() {
        // The node outlives the test (its runtime is not shut down), so keep
        // its data directory: a deleted one makes cluster-persist exit the process.
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = single_node(dir.path()).await;
        create(&node, "kept").await;

        let guard = node.engine_api_enter(None).await;
        let _before = node.engine_api_snapshot(&guard).await.expect("snapshot");
        node.engine.delete_index("kept").unwrap();
        create(&node, "unrelated").await;
        assert!(node.engine.get_index("kept").is_none());
        // The API failed: no kept job.
        drop(guard);
        // (The retry timer may already have taken it.)
        assert!(node.apply_retry.lock().contains("kept") || node.engine.get_index("kept").is_some(), "the withheld index is retried once the API is done");
        let n = node.clone();
        tokio::task::spawn_blocking(move || n.retry_failed_application()).await.unwrap();
        assert!(node.engine.get_index("kept").is_some(), "the committed index is back in engine 0");
        assert!(node.state().metadata.indices.contains_key("kept"));
    }

    fn ingest_custom(ids: &[&str]) -> serde_json::Value {
        json!({"pipeline": ids.iter().map(|id| json!({"id": id, "config": {"processors": [{"set": {"field": "via", "value": id}}]}})).collect::<Vec<_>>()})
    }

    fn engine_pipelines(node: &ClusterNode) -> Vec<String> {
        node.engine.ingest().get(None).into_iter().map(|(id, _)| id).collect()
    }

    fn deferred_markers(node: &ClusterNode) -> Vec<String> {
        node.apply_retry.lock().iter().filter(|e| e.starts_with(CUSTOMS_DEFERRED)).cloned().collect()
    }

    /// Publish custom `key` as a pure job (a change made off engine 0, as
    /// the lifecycle passes, the licence startup and the scratch copy do).
    async fn publish_custom(node: &Arc<ClusterNode>, key: &'static str, value: serde_json::Value) -> u64 {
        node.metadata_service().pure(Box::new(move |_, state, delta| {
            // Pipelines are added to the committed ones (the registries' too).
            let value = match (key, state.metadata.customs.get(key)) {
                (crate::meta_extra::INGEST_CUSTOM, Some(current)) => {
                    let mut merged = current.clone();
                    merged["pipeline"].as_array_mut().unwrap().extend(value["pipeline"].as_array().unwrap().iter().cloned());
                    merged
                }
                _ => value,
            };
            delta.customs.insert(key.to_string(), Some(value));
        })).await.expect("published")
    }

    /// An engine API's own publication carries what it already wrote into
    /// engine 0: nothing is deferred for it (this used to log "failed to
    /// apply metadata: an engine API of this family is in flight" at every
    /// licence / pipeline / security API and leave a retry pending).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn own_family_publication_is_not_deferred() {
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = single_node(dir.path()).await;
        let guard = node.engine_api_enter(Some(crate::meta_extra::INGEST_CUSTOM)).await;
        let before = node.engine_api_snapshot(&guard).await.expect("snapshot");
        node.engine.ingest().put("own", json!({"processors": [{"set": {"field": "via", "value": "own"}}]}), None).unwrap();
        let version = IN_ENGINE_API.scope((), node.publish_engine_metadata_since(before)).await.expect("kept job");
        assert!(IN_ENGINE_API.scope((), node.wait_applied_here(version, Duration::from_secs(10))).await);
        assert!(deferred_markers(&node).is_empty(), "the API's own change was deferred: {:?}", deferred_markers(&node));
        assert!(node.customs_settled.load(std::sync::atomic::Ordering::SeqCst) >= version);
        drop(guard);
        // Also without the scope: the acknowledgement does not wait.
        assert!(node.wait_applied_here(version, Duration::from_millis(200)).await);
        assert!(node.state().metadata.customs[crate::meta_extra::INGEST_CUSTOM].to_string().contains("own"));
    }

    /// A custom change committed while an engine API of the same family is
    /// in flight cannot be written into engine 0 yet (it would wipe the
    /// API's unpublished change). Its acknowledgement waits until it is
    /// applied, which happens as soon as the API is done (not on the 5 s
    /// retry timer), and both changes end up in engine 0 and the state.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn custom_change_racing_a_family_engine_api_is_visible_once_acked() {
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = single_node(dir.path()).await;
        let guard = node.engine_api_enter(Some(crate::meta_extra::INGEST_CUSTOM)).await;
        let before = node.engine_api_snapshot(&guard).await.expect("snapshot");
        // The API (PUT _ingest/pipeline/a) has changed engine 0, unpublished.
        node.engine.ingest().put("a", json!({"processors": [{"set": {"field": "via", "value": "a"}}]}), None).unwrap();
        // A change of the same family made off engine 0 commits meanwhile.
        let vb = publish_custom(&node, crate::meta_extra::INGEST_CUSTOM, ingest_custom(&["b"])).await;
        assert!(!engine_pipelines(&node).contains(&"b".to_string()), "applied over an engine API in flight");
        assert!(engine_pipelines(&node).contains(&"a".to_string()), "the API's unpublished change was wiped");
        let n = node.clone();
        let waiter = tokio::spawn(async move { n.wait_applied_here(vb, Duration::from_secs(20)).await });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!waiter.is_finished(), "acknowledged before the change is visible on this node");
        // The API publishes and finishes.
        let va = IN_ENGINE_API.scope((), node.publish_engine_metadata_since(before)).await.expect("kept job");
        drop(guard);
        let started = Instant::now();
        assert!(waiter.await.unwrap(), "the acknowledgement timed out");
        assert!(started.elapsed() < Duration::from_secs(3), "waited for the retry timer: {:?}", started.elapsed());
        assert!(node.wait_applied_here(va, Duration::from_secs(5)).await);
        let ids = engine_pipelines(&node);
        assert!(ids.contains(&"a".to_string()) && ids.contains(&"b".to_string()), "{ids:?}");
        let published = node.state().metadata.customs[crate::meta_extra::INGEST_CUSTOM].to_string();
        assert!(published.contains("\"a\"") && published.contains("\"b\""), "{published}");
        assert!(deferred_markers(&node).is_empty());
    }

    /// An unscoped engine API in flight holds back only customs the engine
    /// consumes; any other custom (here a plain one) is not deferred, and an
    /// engine custom that is deferred is applied when the API finishes
    /// (`settle_deferred_customs`), before its response.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unscoped_engine_api_defers_only_engine_customs_and_settles_on_exit() {
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = single_node(dir.path()).await;
        let guard = node.engine_api_enter(None).await;
        let v1 = publish_custom(&node, "rustic_test_plain_custom", json!({"x": 1})).await;
        assert!(node.wait_applied_here(v1, Duration::from_millis(500)).await, "a custom with no engine form waited for an engine API");
        assert!(deferred_markers(&node).is_empty(), "{:?}", deferred_markers(&node));
        let v2 = publish_custom(&node, crate::meta_extra::INGEST_CUSTOM, ingest_custom(&["c"])).await;
        assert!(!node.wait_applied_here(v2, Duration::from_millis(300)).await, "acknowledged while deferred");
        assert!(!engine_pipelines(&node).contains(&"c".to_string()));
        drop(guard);
        node.settle_deferred_customs().await;
        assert!(engine_pipelines(&node).contains(&"c".to_string()), "not applied when the API finished");
        assert!(deferred_markers(&node).is_empty());
        assert!(node.wait_applied_here(v2, Duration::from_millis(100)).await);
    }
}

#[cfg(test)]
mod state_owner_tests {
    use super::*;
    use serde_json::json;

    fn config(single: bool) -> NodeConfig {
        let mut settings = crate::config::Settings::new();
        settings.insert("node.name".into(), json!("geo-0"));
        settings.insert("path.data".into(), json!("/tmp/rustic-state-owner-test"));
        if single {
            settings.insert("discovery.type".into(), json!("single-node"));
        }
        NodeConfig::from_settings(settings).unwrap()
    }

    fn state(voters: &[&str], nodes: &[(&str, &str)]) -> ClusterState {
        let mut s = ClusterState::empty("test");
        s.coordination.last_accepted_config = elasticsearch_cluster::state::VotingConfiguration::new(voters.iter().map(|v| v.to_string()));
        for (id, name) in nodes {
            let n: DiscoveryNode = serde_json::from_value(json!({"id": id, "name": name, "ephemeral_id": "e", "transport_address": "127.0.0.1:9300",
                "http_address": "127.0.0.1:9200", "host": "127.0.0.1", "ip": "127.0.0.1", "roles": [], "attributes": {}, "version": "9.5.5"})).unwrap();
            s.nodes.insert(id.to_string(), n);
        }
        s
    }

    /// The geo restart: the data path was emptied and reused while the old
    /// process kept writing, leaving a fresh node id next to the old node's
    /// cluster state. The start now fails with the reason instead of
    /// sitting in "master not discovered".
    #[test]
    fn a_state_written_under_another_node_id_fails_the_start() {
        let s = state(&["Bk"], &[("Bk", "geo-0")]);
        let e = check_state_owner(&s, "Va2", &config(true)).unwrap_err().to_string();
        assert!(e.contains("[Bk]") && e.contains("[Va2]"), "{e}");
        // Not single-node, but the state names this node under another id.
        assert!(check_state_owner(&s, "Va2", &config(false)).is_err());
        // Single-node: the only voter must be this node, whatever the name.
        assert!(check_state_owner(&state(&["Bk"], &[("Bk", "other")]), "Va2", &config(true)).is_err());
    }

    #[test]
    fn own_state_and_ordinary_cluster_members_start() {
        assert!(check_state_owner(&state(&[], &[]), "Va2", &config(true)).is_ok(), "fresh node");
        assert!(check_state_owner(&state(&["Va2"], &[("Va2", "geo-0")]), "Va2", &config(true)).is_ok());
        // A data node outside the voting configuration of a larger cluster.
        assert!(check_state_owner(&state(&["m1", "m2", "m3"], &[("m1", "m-1"), ("Va2", "geo-0")]), "Va2", &config(false)).is_ok());
    }
}

#[cfg(test)]
mod report_admission_tests {
    use super::admit_report;
    use elasticsearch_engine::error::EsError;
    use std::cell::Cell;

    fn refused() -> EsError {
        EsError::new(429, "circuit_breaking_exception", "[request] Data too large")
    }

    /// A report the breaker refuses is built once memory frees, not dropped.
    #[tokio::test]
    async fn a_refused_report_is_built_once_memory_frees() {
        let tries = Cell::new(0);
        let built = admit_report("brk", || {
            tries.set(tries.get() + 1);
            if tries.get() < 4 { Err(refused()) } else { Ok(tries.get()) }
        }, || true).await;
        assert_eq!(built, Some(4));
    }

    /// A failure report about another node's copy (a primary failing a
    /// replica its resync did not reach) stays relevant while that copy is
    /// assigned anywhere and the reporter's term is current; it is never
    /// judged by whether the copy is on the reporting node.
    #[test]
    fn a_failure_report_about_a_remote_copy_stays_relevant() {
        use super::{failure_report_relevant, mirror, ClusterState, Engine, ShardRouting, ShardState, UuidMap};
        use elasticsearch_cluster::state::{AllocationId, IndexRoutingTable, IndexShardRoutingTable};
        let route = |primary: bool, aid: &str, node: Option<&str>, state| ShardRouting {
            index: "idx".into(), shard: 0, primary, state, node_id: node.map(Into::into), relocating_node_id: None,
            allocation_id: Some(AllocationId { id: aid.into(), relocation_id: None }), recovery_source: None,
            unassigned_info: None, expected_shard_size_in_bytes: None,
        };
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        engine.create_index("idx", &serde_json::json!({})).unwrap();
        let mut state = ClusterState::empty("test");
        state.metadata = mirror::extract(&engine, &state.metadata, &mut UuidMap::default());
        state.metadata.indices.get_mut("idx").unwrap().primary_terms = vec![3];
        state.routing_table.indices.insert("idx".into(), IndexRoutingTable { index_uuid: "u".into(), shards: vec![IndexShardRoutingTable {
            shard: 0, copies: vec![route(true, "p", Some("here"), ShardState::Started), route(false, "r", Some("elsewhere"), ShardState::Started),
                route(false, "gone", None, ShardState::Unassigned)],
        }] });
        assert!(failure_report_relevant(&state, "idx", 0, "r", Some(3)), "a remote copy of the current term");
        assert!(failure_report_relevant(&state, "idx", 0, "r", None));
        assert!(!failure_report_relevant(&state, "idx", 0, "r", Some(2)), "the reporter was deposed");
        assert!(!failure_report_relevant(&state, "idx", 0, "gone", None), "an unassigned copy");
        assert!(!failure_report_relevant(&state, "idx", 0, "other", None), "a copy the table no longer has");
        assert!(!failure_report_relevant(&state, "idx", 1, "r", None));
    }

    /// Recovery control and shard-state envelopes are accounted but never
    /// refused under `unrefused`, on a breaker that refuses them otherwise;
    /// their memory is released with them.
    #[test]
    fn control_envelopes_are_accounted_not_refused() {
        use super::{RecoveryReuseReq, ReplicaPrefix, ShardFailedReq, ShardStartedReq};
        use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};
        let b = Breakers::leak_for_tests(1 << 20);
        b.apply_settings(&|k| match k {
            "indices.breaker.total.use_real_memory" => Some("false".into()),
            "indices.breaker.request.limit" => Some("16b".into()),
            _ => None,
        });
        let scope = RequestScope::new("control envelopes", b);
        let long = "x".repeat(4096);
        let refused = scoped(Some(scope.clone()), || ShardFailedReq::try_new("idx", 0, "aid", Some(3), &long, true));
        assert!(refused.is_err(), "the breaker refuses it outside a control scope");
        let (started, failed, offer) = scoped(Some(scope.clone()), || crate::control_memory::unrefused("test", || (
            ShardStartedReq::try_new("idx", 0, "aid"),
            ShardFailedReq::try_new("idx", 0, "aid", Some(3), &long, true),
            RecoveryReuseReq::try_new("idx", 0, "c", "u", "old", "new", "node", "src", 3,
                ReplicaPrefix { checkpoint: -1, global_checkpoint: -1, primary_term: 3 }, None),
        )));
        let (started, failed, offer) = (started.unwrap(), failed.unwrap(), offer.unwrap());
        assert_eq!(failed.message, long);
        assert!(b.get(Kind::Request).used() >= long.len() as u64, "accounted: {}", b.get(Kind::Request).used());
        // Decoding a report on the master is unrefused too.
        let wire = bincode::serialize(&failed).unwrap();
        let decoded: ShardFailedReq = scoped(Some(scope.clone()), || bincode::deserialize(&wire)).unwrap();
        assert_eq!(decoded.message, long);
        drop((started, failed, offer, decoded));
        drop(scope);
        assert_eq!(b.get(Kind::Request).used(), 0);
    }

    /// The wait ends once the copy no longer needs the report.
    #[tokio::test]
    async fn the_wait_ends_when_the_copy_no_longer_needs_it() {
        let checks = Cell::new(0);
        let built: Option<()> = admit_report("brk", || Err(refused()), || {
            checks.set(checks.get() + 1);
            checks.get() < 3
        }).await;
        assert_eq!(built, None);
        assert_eq!(checks.get(), 3);
    }
}
