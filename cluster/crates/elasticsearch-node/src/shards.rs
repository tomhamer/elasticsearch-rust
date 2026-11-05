//! Local shard copies and their replication roles.
//!
//! Every node mirrors every index in its engine (see `mirror`), but a copy
//! only *holds data* when the routing table assigns it here. This module
//! keeps, per locally assigned copy, its role:
//!
//! * **Primary** — a [`PrimaryCtx`]: front-level write gate, the
//!   replication pump (translog tailer → `ReplicationRequest`s), the
//!   `ReplicationTracker` (checkpoints, in-sync set, retention) and the
//!   recovery sources it is serving.
//! * **Replica** — a [`ReplicaApplier`] fed by the primary's pump.
//!
//! The write protocol (see `replication` module docs, and
//! `docs/CONSISTENCY.md` G3–G5): the front executes a write on the primary
//! under a gate permit, then [`PrimaryCtx::replicate_through`] ships every
//! op up to the write's seq_no to every replication target and waits until
//! all in-sync copies acked it. A copy that fails is reported to the master
//! (`mark_as_stale`) and the write is acked only once the master committed
//! that failure. A replica rejects ops from an older primary term, so a
//! deposed primary can never collect the acks it needs.
//!
//! Each locally assigned copy is recorded in `<engine index>/cluster-copy.json`
//! (index uuid, shard, allocation id, primary term) so that after a full
//! cluster restart the master can find in-sync copies (`shard_stores`).

use crate::replication::checkpoint::{LocalCheckpointTracker, ReplicationTracker};
use crate::replication::ops::{ReplicaOp, ReplicationRequest, ReplicationResponse, ResyncRequest};
use crate::replication::recovery::{RecoveryOptions, RecoverySource, RecoveryStartInfo, RecoveryState, RecoveryTarget};
use crate::replication::replica::ReplicaApplier;
use crate::replication::tailer::TranslogTailer;
use crate::replication::ReplicationError;
use elasticsearch_engine::index::{IndexService, IndexShard};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const ACTION_REPLICATE: &str = "indices:data/write/replica";
pub const ACTION_RESYNC: &str = "internal:index/seq_no/resync";
/// Refresh a replica copy (`indices:admin/refresh[s][r]`): a write with
/// `refresh=true|wait_for` is visible on every copy before it is acked.
pub const ACTION_REPLICA_REFRESH: &str = "indices:admin/refresh[s][r]";
pub const ACTION_RECOVERY_START: &str = "internal:index/shard/recovery/start_recovery";
/// Separate negotiation action: older peers retain the full-snapshot wire
/// protocol and cannot accidentally interpret a reuse proposal as a wipe.
pub const ACTION_RECOVERY_REUSE: &str = "internal:index/shard/recovery/reuse_prefix";
pub const ACTION_RECOVERY_REUSE_BATCH: &str = "internal:index/shard/recovery/reuse_batch";
pub const ACTION_RECOVERY_REUSE_FINALIZE: &str = "internal:index/shard/recovery/reuse_finalize";
pub const ACTION_RECOVERY_BATCH: &str = "internal:index/shard/recovery/batch";
pub const ACTION_RECOVERY_FINALIZE: &str = "internal:index/shard/recovery/finalize";
pub const ACTION_SHARD_STORES: &str = "internal:gateway/local/started_shards";

/// The engine directory of shard `n` of an index service. Cluster shard `n`
/// lives in shard engine `n`, whose copy of the index routes it to engine
/// shard `n` (same routing), so the cluster shard number is the engine one.
pub fn shard_path(svc: &IndexService, n: u32) -> PathBuf {
    svc.path().join(format!("shard-{n}"))
}

/// The open engine copy of shard `n` of `svc`, or `shard_not_found_exception`
/// when it is not open. Shard engines create their indices sparse
/// (`IndexService::is_sparse`) and the node opens only the copies it holds
/// (`ClusterNode::ensure_copy_open`), but a copy can close at any moment
/// under a caller that does not hold the apply lock: an index deleted or
/// closed (or re-created under the same name), a failed or superseded
/// allocation, or a file recovery between retiring its placeholder and
/// publishing the received copy (`try_install_recovered_files`). Callers
/// fail their operation (a recovery fails, allocation retries it), never
/// the node. Long-lived holders (`PrimaryCtx`, `ReplicaApplier`) capture
/// the copy once and keep it after the node closes it (operations on it
/// then fail with `index_shard_closed_exception`).
pub fn try_copy_of(svc: &IndexService, n: u32) -> Result<Arc<IndexShard>, ReplicationError> {
    svc.shard(n).ok_or_else(|| ReplicationError::Engine(elasticsearch_engine::error::EsError::new(404, "shard_not_found_exception",
        format!("shard [{}][{n}] is not open in this engine", svc.name))))
}

/// Tests: [`try_copy_of`] of a copy the test holds open.
#[cfg(test)]
pub fn copy_of(svc: &IndexService, n: u32) -> Arc<IndexShard> {
    try_copy_of(svc, n).unwrap_or_else(|e| panic!("{e}"))
}

/// What a node remembers about the copy it holds (for `shard_stores`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CopyRecord {
    pub index: String,
    pub index_uuid: String,
    pub shard: u32,
    pub allocation_id: String,
    pub primary_term: u64,
    pub primary: bool,
    /// The shard's history identity (ES `history_uuid`): set when a
    /// primary starts a new history (a new shard, or a stale or empty
    /// primary forced with `accept_data_loss`), inherited by every copy
    /// recovered from it. Copies sharing it share the operations up to
    /// any global checkpoint either saw, so a copy that left the in-sync
    /// set may still catch up from operations. `None`: recorded by an older
    /// version (history reuse then needs in-sync membership).
    #[serde(default)]
    pub history_uuid: Option<String>,
    /// A primary's global checkpoint as of its last flush (a lower bound):
    /// everything at or below it is on every in-sync copy, so a deposed
    /// primary can verify its operations above it against the new primary
    /// instead of being re-recovered from scratch.
    #[serde(default)]
    pub global_checkpoint: Option<i64>,
}

/// The history identity a new primary of `index` starts with: the local
/// copy's when it continues that history (`ExistingStore` without
/// `bootstrap_new_history_uuid`), else a new one.
pub fn primary_history_uuid(svc: &IndexService, index_uuid: &str, shard: u32, continues: bool) -> String {
    read_copy_record(svc.path()).filter(|r| continues && r.index_uuid == index_uuid && r.shard == shard).and_then(|r| r.history_uuid)
        .unwrap_or_else(elasticsearch_cluster::state::random_id)
}

/// Preserve the durable lower bound when the same history changes primary
/// role or restarts. A new/forced history must never inherit that checkpoint.
/// `observed` is a replica's already durable GCP, or -1 for a cold primary.
pub fn primary_checkpoint(svc: &IndexService, index_uuid: &str, shard: u32, history_uuid: &str, observed: i64) -> i64 {
    let inherited = read_copy_record(svc.path()).filter(|r|
        r.index_uuid == index_uuid && r.shard == shard && r.history_uuid.as_deref() == Some(history_uuid))
        .and_then(|r| r.global_checkpoint).unwrap_or(-1);
    observed.max(-1).max(inherited)
}

/// Record `p`'s global checkpoint in its copy record (before each periodic
/// flush) once it is durable here: every op at or below it was appended to
/// this copy's translog, but with `durability: async` maybe not yet synced,
/// and a crash would leave the copy below the value it claims (ES computes
/// the global checkpoint from persisted local checkpoints). The translog is
/// synced first, and the value recorded only when the sync succeeds.
/// Returns the checkpoint recorded (`None`: the sync failed).
pub fn note_durable_primary_global_checkpoint(p: &PrimaryCtx) -> Option<i64> {
    let gcp = p.global_checkpoint();
    match p.copy().shard.sync_translog() {
        Ok(()) => {
            note_primary_global_checkpoint(&p.svc, &p.allocation_id, gcp);
            Some(gcp)
        }
        Err(e) => {
            tracing::debug!(index = %p.index, shard = p.shard, "global checkpoint not recorded: translog sync failed: {}", e.reason);
            None
        }
    }
}

/// Raise the global checkpoint recorded for a primary copy (best effort;
/// called around its flushes).
pub fn note_primary_global_checkpoint(svc: &IndexService, allocation_id: &str, gcp: i64) {
    let Some(mut rec) = read_copy_record(svc.path()) else { return };
    if !rec.primary || rec.allocation_id != allocation_id || rec.global_checkpoint.is_some_and(|g| g >= gcp) {
        return;
    }
    rec.global_checkpoint = Some(gcp);
    if let Err(e) = write_copy_record(svc, &rec) {
        tracing::debug!(index = %rec.index, "could not record the primary's global checkpoint: {e}");
    }
}

const COPY_FILE: &str = "cluster-copy.json";

pub fn write_copy_record(svc: &IndexService, rec: &CopyRecord) -> std::io::Result<()> {
    let p = svc.path().join(COPY_FILE);
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(rec).map_err(std::io::Error::other)?)?;
    elasticsearch_engine::rustic_core::platform::sync_file(&tmp)?;
    std::fs::rename(tmp, p)?;
    elasticsearch_engine::rustic_core::platform::sync_dir(svc.path())
}

pub fn read_copy_record(index_path: &Path) -> Option<CopyRecord> {
    serde_json::from_slice(&std::fs::read(index_path.join(COPY_FILE)).ok()?).ok()
}

pub fn remove_copy_record(svc: &IndexService) -> std::io::Result<()> {
    match std::fs::remove_file(svc.path().join(COPY_FILE)) {
        Ok(()) => {},
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
        Err(e) => return Err(e),
    }
    elasticsearch_engine::rustic_core::platform::sync_dir(svc.path())
}

/// Discard history proof durably before replacing its physical data. Otherwise
/// an interrupted recovery can advertise the old in-sync allocation while
/// actually holding only a partial snapshot of the new copy.
pub fn reset_copy_for_recovery(svc: &IndexService, shard: u32) -> Result<(), String> {
    // Drain old recovery finalizers before unlinking their proof; a finalizer
    // holding an operation permit must not recreate the old record afterward.
    if svc.shard(shard).is_some() {
        svc.close_shard(shard).map_err(|e| e.to_string())?;
    }
    remove_copy_record(svc).map_err(|e| format!("cannot invalidate old shard history: {e}"))?;
    svc.remove_shard(shard, true).map_err(|e| e.to_string())?;
    // Nothing of this copy is kept for a later file recovery either.
    crate::replication::file_recovery::remove_recovery_dirs(svc, shard);
    Ok(())
}

/// [`reset_copy_for_recovery`] for a file-based recovery: the old store is
/// moved aside (`shard-<n>.reuse/`) instead of deleted, so the recovery
/// reuses its files identical to the source's. Its history proof goes
/// first (durably), exactly as when it is deleted. An earlier attempt's
/// empty placeholder (marked) is deleted and the store it set aside kept;
/// a real store replaces an older one set aside.
pub fn set_aside_copy_for_recovery(svc: &IndexService, shard: u32) -> Result<(), String> {
    use crate::replication::file_recovery as fr;
    if svc.shard(shard).is_some() {
        svc.close_shard(shard).map_err(|e| e.to_string())?;
    }
    remove_copy_record(svc).map_err(|e| format!("cannot invalidate old shard history: {e}"))?;
    let (dir, reuse) = (shard_path(svc, shard), fr::reuse_dir(svc, shard));
    if dir.join("shard_state.json").exists() && !dir.join(fr::PLACEHOLDER_MARKER).exists() {
        if reuse.exists() {
            std::fs::remove_dir_all(&reuse).map_err(|e| format!("cannot replace the store set aside: {e}"))?;
        }
        retry_shard_locked(|| svc.set_aside_shard_store(shard, &reuse).map(|_| ()))?;
    }
    retry_shard_locked(|| svc.remove_shard(shard, true))
}

/// A copy detached by a recovery retry is closed in the background: wait
/// (up to 5 s) for its shard lock instead of giving the recovery up.
fn retry_shard_locked(mut f: impl FnMut() -> elasticsearch_engine::error::Result<()>) -> Result<(), String> {
    for _ in 0..50 {
        match f() {
            Err(e) if e.kind == "shard_lock_obtain_failed_exception" => std::thread::sleep(std::time::Duration::from_millis(100)),
            r => return r.map_err(|e| e.to_string()),
        }
    }
    f().map_err(|e| e.to_string())
}

/// A replica refresh error that says the copy is not there (ES
/// `TransportActions.isShardNotAvailableException`).
fn refresh_target_missing(e: &str) -> bool {
    ["[shard_not_found_exception]", "[index_shard_closed_exception]", "[index_not_found_exception]", "[illegal_index_shard_state_exception]"].iter().any(|t| e.starts_with(t))
}

/// ES `TransportActions.isShardNotAvailableException` on a replica
/// operation's failure ("[type] reason"): `ShardNotFoundException`,
/// `IndexNotFoundException`, `IllegalIndexShardStateException` (and its
/// subclasses), `NoShardAvailableActionException`,
/// `UnavailableShardsException`, `AlreadyClosedException`. ES still fails
/// such a copy (write actions always do) but does not report it in
/// `_shards.failures`.
pub fn is_shard_not_available(e: &str) -> bool {
    const TYPES: [&str; 9] = ["shard_not_found_exception", "index_not_found_exception", "illegal_index_shard_state_exception",
        "index_shard_closed_exception", "index_shard_not_started_exception", "index_shard_not_recovering_exception",
        "no_shard_available_action_exception", "unavailable_shards_exception", "already_closed_exception"];
    split_error(e).is_some_and(|(t, _)| TYPES.contains(&t))
}

/// "[type] reason" (how replica failures are rendered, `RemoteError`'s
/// display) split into its ES exception type and reason.
fn split_error(e: &str) -> Option<(&str, &str)> {
    let rest = e.strip_prefix('[')?;
    let (t, reason) = rest.split_once(']')?;
    if t.is_empty() || !t.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
        return None;
    }
    Some((t, reason.strip_prefix(' ').unwrap_or(reason)))
}

/// The REST status ES gives the replica failures a replicated write can
/// report (`ExceptionsHelper.status`); anything else is a 500.
fn replica_failure_status(error_type: &str) -> u16 {
    match error_type {
        "circuit_breaking_exception" | "es_rejected_execution_exception" => 429,
        "connect_transport_exception" | "node_disconnected_exception" | "node_not_connected_exception" | "send_request_transport_exception" => 502,
        "version_conflict_engine_exception" => 409,
        "mapper_parsing_exception" | "document_parsing_exception" | "illegal_argument_exception" | "strict_dynamic_mapping_exception" => 400,
        "shard_not_found_exception" | "index_not_found_exception" => 404,
        "unavailable_shards_exception" | "no_shard_available_action_exception" | "mapper_exception" => 503,
        _ => 500,
    }
}

/// One `_shards.failures` entry of a replicated write (ES
/// `ReplicationResponse.ShardInfo.Failure`: the copy's shard, the node it
/// was on and the replica operation's exception; `primary` is always false).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaFailure {
    pub index: String,
    pub shard: u32,
    /// The failed copy's node id (ES `ShardRouting.currentNodeId()`).
    pub node: String,
    pub error_type: String,
    pub reason: String,
    pub status: u16,
}

impl ReplicaFailure {
    /// From a replica operation's error, "[type] reason" (anything else is
    /// a generic `exception`).
    pub fn new(index: &str, shard: u32, node: &str, error: &str) -> Self {
        let (error_type, reason) = split_error(error).unwrap_or(("exception", error));
        ReplicaFailure { index: index.to_string(), shard, node: node.to_string(), error_type: error_type.to_string(),
            reason: reason.to_string(), status: replica_failure_status(error_type) }
    }
}

/// `_shards` accounting of one replicated write.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShardsInfo {
    pub total: u32,
    pub successful: u32,
    pub failures: Vec<ReplicaFailure>,
}

/// Immutable target strings remain admitted through every pump snapshot.
#[derive(Clone, Debug)]
pub struct Target(Arc<TargetData>);
#[derive(Debug)]
pub struct TargetData {
    pub allocation_id:String,
    pub node_id:String,
    pub address:String,
    _memory:elasticsearch_engine::memory::Reservation,
}
impl std::ops::Deref for Target {type Target=TargetData;fn deref(&self)->&TargetData{&self.0}}
impl Target {
    pub fn try_new(allocation_id:&str,node_id:&str,address:&str)->Result<Self,elasticsearch_engine::error::EsError>{
        let bytes=allocation_id.len().checked_add(node_id.len()).and_then(|n|n.checked_add(address.len())).and_then(|n|n.checked_add(std::mem::size_of::<TargetData>()+2*std::mem::size_of::<usize>())).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","target metadata overflow"))?;
        let memory=elasticsearch_engine::memory::breakers().reserve_native(bytes as u64,"recovery target metadata")?;
        Ok(Self(Arc::new(TargetData{allocation_id:crate::mapping_update::copy_wire_text(allocation_id)?,node_id:crate::mapping_update::copy_wire_text(node_id)?,address:crate::mapping_update::copy_wire_text(address)?,_memory:memory.into_result_owner()})))
    }
}
/// Structural admission lives with the map, including any retained empty root.
#[derive(Default)]
pub struct TargetMap { values:BTreeMap<String,Target>, memory:Option<elasticsearch_engine::memory::Reservation>, slots:usize, key_live:usize, key_high:usize }
impl std::ops::Deref for TargetMap {type Target=BTreeMap<String,Target>;fn deref(&self)->&Self::Target{&self.values}}
impl TargetMap {
    pub fn try_insert(&mut self,target:Target)->Result<(),elasticsearch_engine::error::EsError>{
        if self.memory.is_none(){self.memory=Some(elasticsearch_engine::memory::breakers().reserve_native(0,"recovery target map")?.into_result_owner());}
        // Retain conservative structural/key high-water charge until map drop.
        // Replacement borrows the existing key and allocates no new key/node.
        if let Some(old)=self.values.get_mut(&target.allocation_id){*old=target;return Ok(())}
        let key_live=self.key_live.checked_add(target.allocation_id.len()).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","target map overflow"))?;
        let node_bytes=if self.values.len()>=self.slots {2*(11*std::mem::size_of::<(String,Target)>()+20*std::mem::size_of::<usize>())}else{0};
        let bytes=key_live.saturating_sub(self.key_high).checked_add(node_bytes).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","target map overflow"))?;
        self.memory.as_mut().unwrap().grow(bytes as u64,"recovery target map growth")?;
        if node_bytes!=0{self.slots+=1}self.key_high=self.key_high.max(key_live);
        let key=crate::mapping_update::copy_wire_text(&target.allocation_id)?;
        self.key_live=key_live;self.values.insert(key,target);Ok(())
    }
    pub fn try_snapshot(&self)->Result<Self,elasticsearch_engine::error::EsError>{let mut out=Self::default();for target in self.values.values(){out.try_insert(target.clone())?}Ok(out)}
    pub fn remove(&mut self,key:&str){if let Some((key,_))=self.values.remove_entry(key){self.key_live-=key.len();}}
    pub fn retain(&mut self,mut f:impl FnMut(&String,&mut Target)->bool){self.values.retain(|k,v|{if f(k,v){true}else{self.key_live-=k.len();false}});}
    #[cfg(test)]
    fn insert(&mut self,_key:String,target:Target){self.try_insert(target).unwrap();}
}

/// Sends replica requests and reports failed copies; implemented by the
/// node over the transport (and by tests in-process).
#[async_trait::async_trait]
pub trait ReplicationNet: Send + Sync {
    async fn replicate(&self, target: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String>;
    /// Report `allocation_id` failed (mark stale) to the master and wait
    /// until the state removing it is committed. Errors mean the write
    /// must not be acked.
    async fn fail_copy(&self, index: &str, shard: u32, allocation_id: &str, primary_term: u64, reason: &str) -> Result<(), String>;
    /// Refresh the replica copy `target` of `index`/`shard`.
    async fn refresh_replica(&self, _target: &Target, _req: ReplicaRefreshReq) -> Result<(), String> {
        Ok(())
    }
}

/// [`ACTION_REPLICA_REFRESH`] request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaRefreshReq {
    pub index: crate::front::SharedText,
    pub shard: u32,
    pub target_allocation_id: crate::front::SharedText,
}

pub fn handle_replica_refresh(mgr: &ShardManager, req: &ReplicaRefreshReq) -> Result<(), (String, String)> {
    let Some(r) = mgr.replica(&req.index, req.shard, &req.target_allocation_id) else {
        return Err(("shard_not_found_exception".into(), format!("no such shard copy [{}][{}] with allocation id [{}]", req.index, req.shard, req.target_allocation_id)));
    };
    r.copy().shard.refresh().map(|_| ()).map_err(|e| ("refresh_failed_engine_exception".to_string(), e.to_string()))
}

struct Pump {
    tailer: TranslogTailer,
    local: LocalCheckpointTracker,
    /// Mapping JSON last sent per target allocation id.
    mapping_sent: HashMap<String, String>,
    /// Global checkpoint last delivered per target allocation id.
    gcp_sent: HashMap<String, i64>,
    max_seq_no_of_updates: i64,
    /// The tailer was left behind while no copy needed it (see
    /// `replicate_through`): reposition it before the next poll.
    tailer_stale: bool,
    /// Running average of the bytes per shipped op: sizes the next batch
    /// to about [`REPLICATION_BATCH_BYTES`].
    op_bytes: f64,
}

/// Target bytes of one replication batch. Batches were bounded by count
/// only (10,000 ops): with large documents (a 1536-d vector's JSON source is
/// ~35 KB) one batch and its encoded and compressed copies held hundreds of
/// MB per shard, and a lagging replica let them pile up until the parent
/// breaker tripped.
const REPLICATION_BATCH_BYTES: f64 = 8.0 * 1024.0 * 1024.0;
/// Pause after a transient breaker refusal before shipping again; doubles
/// up to [`BACKPRESSURE_MAX_DELAY`] while it keeps refusing.
const BACKPRESSURE_DELAY: std::time::Duration = std::time::Duration::from_millis(50);
const BACKPRESSURE_MAX_DELAY: std::time::Duration = std::time::Duration::from_secs(1);
/// How long a copy may keep refusing a batch with breaker errors before it is
/// failed. Bounded: a persistent refusal (an op larger than the replica's
/// breaker, sustained pressure) would otherwise hold every waiting writer
/// forever.
const TRANSIENT_REFUSAL_WINDOW: std::time::Duration = std::time::Duration::from_secs(5);

/// A replication failure worth a short, bounded retry before the copy is
/// failed: a breaker refusal while encoding, sending or decoding a batch.
/// The replica refuses only before its WAL append (everything after it is
/// forced), so a repeat appends nothing twice. An indexing-pressure rejection
/// (`es_rejected_execution_exception`) is not retried: like ES, it fails the
/// replica copy at once, as does any other error.
fn is_transient_refusal(error: &str) -> bool {
    error.contains("circuit_breaking_exception")
}

/// An in-sync copy whose probed checkpoint has not advanced for this long,
/// with nothing in flight to it, is failed (it cannot catch up through us).
const LAG_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// How soon a lagging copy with nothing in flight is probed again.
const LAG_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Ops of a round trip that ended before its replicas answered (the
/// request was cancelled or failed early): they left the tailer, so the
/// next writer ships them (ES never relies on the originating request to
/// deliver an op: later replication sends it from history).
struct Requeued {
    ops: Option<elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>>,
    uploaded: bool,
    next: Option<Box<Requeued>>,
    // The node is allocated before polling; cancellation never allocates.
    _memory: elasticsearch_engine::memory::Reservation,
}
impl Drop for Requeued {
    fn drop(&mut self) {
        // Cancellation backlogs must not recurse once per queued batch.
        let mut next = self.next.take();
        while let Some(mut node) = next { next = node.next.take(); }
    }
}
struct RoundTrip<'a> {
    ctx: &'a PrimaryCtx,
    node: Option<Box<Requeued>>,
    started: bool,
    delivered: bool,
}
impl RoundTrip<'_> {
    fn ops(&self) -> &elasticsearch_engine::translog::AdmittedBatch<ReplicaOp> {
        self.node.as_ref().unwrap().ops.as_ref().unwrap()
    }
    fn ops_mut(&mut self) -> &mut elasticsearch_engine::translog::AdmittedBatch<ReplicaOp> {
        self.node.as_mut().unwrap().ops.as_mut().unwrap()
    }
    fn start(&mut self) {
        self.ctx.round_trips.fetch_add(1, Ordering::SeqCst);
        self.started = true;
    }
}
impl Drop for RoundTrip<'_> {
    fn drop(&mut self) {
        if let Some(mut node) = self.node.take() {
            if !self.delivered && node.ops.as_ref().is_some_and(|ops| !ops.is_empty()) {
                let mut queue = self.ctx.requeued.lock();
                node.next = queue.take();
                *queue = Some(node);
            }
        }
        if self.started {
            self.ctx.round_trips.fetch_sub(1, Ordering::SeqCst);
            self.ctx.round_trip_done.send_modify(|n| *n = n.wrapping_add(1));
        }
    }
}

pub struct PrimaryCtx {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
    pub term: u64,
    pub svc: Arc<IndexService>,
    /// The engine copy captured at construction (see [`try_copy_of`]).
    copy: Arc<IndexShard>,
    /// Writes take `read`, flush / recovery start take `write`.
    pub gate: Arc<tokio::sync::RwLock<()>>,
    pump: tokio::sync::Mutex<Pump>,
    /// Replica round trips in flight: batches that left the tailer (counted
    /// under the pump lock) whose responses are not processed yet.
    round_trips: std::sync::atomic::AtomicUsize,
    /// Bumped each time a round trip's responses have been processed, so a
    /// writer whose ops ride on an in-flight batch waits for it.
    round_trip_done: tokio::sync::watch::Sender<u64>,
    /// Batches whose round trip ended undelivered (see [`Requeued`]).
    requeued: Mutex<Option<Box<Requeued>>>,
    /// A [`Self::sync_global_checkpoint`] round trip is in flight.
    gcp_sync_running: std::sync::atomic::AtomicBool,
    /// Shared (weakly) with the engine copy's replication retention.
    pub tracker: Arc<Mutex<ReplicationTracker>>,
    /// Recovery sources being served, by recovery id, with the target
    /// allocation id each one belongs to (at most one live recovery per id:
    /// a newer start supersedes the older one).
    pub recoveries: Mutex<HashMap<u64, (String, Arc<tokio::sync::Mutex<RecoverySource>>)>>,
    /// The pinned commits of the file-based recoveries being served, by
    /// recovery id (released with their recovery).
    pub recovery_files: Mutex<HashMap<u64, FileServe>>,
    /// Pending START ordering only; entries disappear when the last preparing
    /// attempt returns. A late older completion cannot replace a newer start.
    recovery_attempts: Mutex<HashMap<String, (u64, usize)>>,
    /// Allocation id → current target address (refreshed from the state).
    pub targets: Mutex<TargetMap>,
    /// Set once primary authority was handed to a relocation target: every
    /// later write on this copy must be re-routed to the new primary.
    pub relocated: std::sync::atomic::AtomicBool,
    /// Stateless: the shard's object store directory. Every batch the pump
    /// takes from the translog is uploaded before it is shipped, so a write
    /// is acknowledged only once it is durable there.
    pub object_store: std::sync::OnceLock<Arc<crate::stateless::ShardObjectStore>>,
}

impl PrimaryCtx {
    /// A fresh primary over `svc`; tails the translog from the current end.
    pub fn new(index: &str, shard: u32, allocation_id: &str, term: u64, svc: Arc<IndexService>) -> Result<PrimaryCtx, ReplicationError> {
        let copy = try_copy_of(&svc, shard)?;
        // The engine assigns its ops the copy's term (`_primary_term`, the
        // version map, `if_primary_term`): the cluster's, not the term 1 a
        // copy is opened at.
        if term > copy.shard.primary_term() {
            copy.shard.set_primary_term(term)?;
        }
        let max = copy.shard.max_seq_no();
        let tailer = TranslogTailer::open_at(&shard_path(&svc, shard), max + 1)?;
        let tracker = Arc::new(Mutex::new(ReplicationTracker::new(allocation_id, term, max)));
        // Every engine flush of this copy keeps the translog ops some copy
        // (or the tailer: the primary's own entry) still needs, so flushes
        // need no write fence. A dropped context retains nothing.
        let retained = Arc::downgrade(&tracker);
        copy.shard.set_replication_retention(Some(Arc::new(move || retained.upgrade().map(|t| t.lock().min_retained_seq_no()))));
        Ok(PrimaryCtx {
            index: index.to_string(),
            shard,
            allocation_id: allocation_id.to_string(),
            term,
            svc,
            copy,
            gate: Arc::new(tokio::sync::RwLock::new(())),
            pump: tokio::sync::Mutex::new(Pump { tailer, local: LocalCheckpointTracker::new(max), mapping_sent: HashMap::new(), gcp_sent: HashMap::new(), max_seq_no_of_updates: max, tailer_stale: false, op_bytes: 0.0 }),
            round_trips: std::sync::atomic::AtomicUsize::new(0),
            round_trip_done: tokio::sync::watch::channel(0).0,
            requeued: Mutex::new(None),
            gcp_sync_running: std::sync::atomic::AtomicBool::new(false),
            tracker,
            recoveries: Mutex::new(HashMap::new()),
            recovery_files: Mutex::new(HashMap::new()),
            recovery_attempts: Mutex::new(HashMap::new()),
            targets: Mutex::new(TargetMap::default()),
            relocated: std::sync::atomic::AtomicBool::new(false),
            object_store: std::sync::OnceLock::new(),
        })
    }

    /// The engine copy this primary serves.
    pub fn copy(&self) -> Arc<IndexShard> {
        self.copy.clone()
    }

    fn mapping_json(&self) -> Result<(String, elasticsearch_engine::memory::Reservation), String> {
        self.svc.mapping_json_admitted().map_err(|e| e.to_string())
    }

    /// Count a round trip in flight (call under the pump lock, before the
    /// batch's ops become unobservable to other writers).
    fn prepare_round_trip(&self) -> Result<RoundTrip<'_>, String> {
        let memory = elasticsearch_engine::memory::breakers().reserve_request(
            std::mem::size_of::<Requeued>() as u64, "replication cancellation queue node").map_err(|e| e.to_string())?;
        let node = Box::new(Requeued { ops: None, uploaded: false, next: None, _memory: memory });
        Ok(RoundTrip { ctx: self, node: Some(node), started: false, delivered: false })
    }

    /// Ship every op up to `seq_no` and wait until all in-sync copies have
    /// it (failing copies through the master as needed).
    pub async fn replicate_through(&self, seq_no: i64, net: &dyn ReplicationNet) -> Result<ShardsInfo, String> {
        let mut failures: Vec<ReplicaFailure> = Vec::new();
        let mut empty_polls = 0;
        // Lagging copy → (checkpoint last reported, when it last advanced).
        let mut lag_seen: HashMap<String, (i64, std::time::Instant)> = HashMap::new();
        // Copy → when it first refused (breaker) without acking since.
        let mut refused_since: HashMap<String, std::time::Instant> = HashMap::new();
        let mut backoff = BACKPRESSURE_DELAY;
        loop {
            let mut pump = self.pump.lock().await;
            // A primary engine failed by a translog tragic event (append or
            // fsync) replicates nothing more: its later ops are not durable
            // here and are never acknowledged, even ones every copy already
            // has (checked first); the copy is failed and a replica promoted
            // (ES failEngine).
            if let Some(error) = self.copy.shard.tragic_error() {
                return Err(format!("primary engine failed: {error}"));
            }
            if self.tracker.lock().all_in_sync_acked(seq_no) && pump.local.checkpoint() >= seq_no {
                break;
            }
            // Subscribed before the requeue and the in-flight count are read
            // (both below, under the pump lock): a round trip ending in
            // between re-queues before it signals, so it cannot be missed.
            // No copy to replicate to (no tracked replica or recovering
            // copy, nothing re-queued, no object store): the ops would be
            // read back from the translog only to be shipped nowhere. The
            // local checkpoint comes from the engine instead, and the tailer
            // is repositioned past it once a copy is tracked (ops processed
            // before that are its recovery's).
            let alone = self.object_store.get().is_none() && self.requeued.lock().is_none()
                && self.round_trips.load(Ordering::SeqCst) == 0 && self.tracker.lock().replication_targets().is_empty();
            if alone {
                let cp = self.copy.shard.processed_checkpoint();
                pump.local.fast_forward(cp);
                pump.max_seq_no_of_updates = pump.max_seq_no_of_updates.max(self.copy.shard.max_seq_no_of_updates());
                pump.tailer_stale = true;
                let local_cp = pump.local.checkpoint();
                self.tracker.lock().update_local_checkpoint(&self.allocation_id, local_cp);
                if local_cp >= seq_no {
                    continue;
                }
                // An earlier op of a concurrent batch is still being appended.
                empty_polls += 1;
                if empty_polls >= 15_000 {
                    return Err(format!("timed out waiting for primary WAL through seq_no {seq_no}; checkpoint {local_cp}"));
                }
                drop(pump);
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                continue;
            }
            if pump.tailer_stale {
                let from = pump.local.checkpoint() + 1;
                // A seek reads translog frames (blocking file I/O). From
                // above the engine's processed checkpoint it never takes the
                // extended scan (that is only for seeks at or below the
                // commit point), but still keep the read off the async
                // worker where the runtime allows it.
                let tailer = &mut pump.tailer;
                let seeked = if tokio::runtime::Handle::current().runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
                    tokio::task::block_in_place(|| tailer.seek_to_seq_no(from))
                } else {
                    tailer.seek_to_seq_no(from)
                };
                match seeked {
                    Ok(()) => pump.tailer_stale = false,
                    // The ops some copy needs are no longer in the translog:
                    // it can never catch up through replication. ES fails
                    // it (marked stale, it recovers from a file copy); the
                    // pump then goes on from what the translog still holds,
                    // never stuck retrying a seek that cannot succeed.
                    Err(ReplicationError::OpsTrimmed { needed_from, available_from }) => {
                        drop(pump);
                        self.fail_copies_missing_history(needed_from, available_from, net, &mut failures).await?;
                        let mut pump = self.pump.lock().await;
                        let processed = self.copy.shard.processed_checkpoint();
                        if processed < available_from - 1 {
                            return Err(format!("operations trimmed from translog: needed from seq_no [{needed_from}], oldest available [{available_from}], processed only through [{processed}]"));
                        }
                        pump.local.fast_forward(available_from - 1);
                        let local_cp = pump.local.checkpoint();
                        self.tracker.lock().update_local_checkpoint(&self.allocation_id, local_cp);
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            let mut done = self.round_trip_done.subscribe();
            // The watermark the tailer may prune its jumped ranges to: the
            // processed checkpoint sampled before the poll.
            let processed = self.copy.shard.processed_checkpoint();
            let mut round_trip = self.prepare_round_trip()?;
            let max_ops = if pump.op_bytes > 0.0 { ((REPLICATION_BATCH_BYTES / pump.op_bytes) as usize).clamp(16, 10_000) } else { 64 };
            let ops = match pump.tailer.poll_mapped(max_ops, |op| ReplicaOp::from_translog(op, self.term)) {
                Ok(ops) => ops,
                // Across a generation switch the tailer cannot tell trimmed
                // ops from ones that reach the translog late (a batch
                // sequenced before a roll appends after it, behind later
                // batches). Reposition with a seek, which looks through
                // every retained generation; only a seek that still finds
                // them gone fails the copies that need them (above). The
                // ops this poll consumed are read again.
                Err(ReplicationError::OpsTrimmed { needed_from, available_from }) => {
                    tracing::warn!(index = %self.index, shard = self.shard, needed_from, available_from, "replication tailer lost contiguity; repositioning it");
                    pump.tailer_stale = true;
                    continue;
                }
                Err(error) => return Err(error.to_string()),
            };
            if !ops.is_empty() {
                let bytes: usize = ops.iter().map(|op| match op { ReplicaOp::Index { source_json, id, .. } => source_json.len() + id.len() + 64, _ => 64 }).sum();
                let per_op = bytes as f64 / ops.len() as f64;
                pump.op_bytes = if pump.op_bytes == 0.0 { per_op } else { 0.8 * pump.op_bytes + 0.2 * per_op };
            }
            round_trip.node.as_mut().unwrap().ops = Some(ops);
            pump.tailer.prune_jumped_through(processed);
            let mut uploaded_memory = elasticsearch_engine::memory::breakers().reserve_request(0, "uploaded sequence slots").map_err(|e| e.to_string())?;
            let mut uploaded_seq_nos = Vec::<i64>::new();
            let mut requeued = self.requeued.lock().take();
            while let Some(mut node) = requeued.take() {
                requeued = node.next.take();
                let batch = node.ops.take().expect("queued batch");
                let prepare = (|| -> elasticsearch_engine::error::Result<()> {
                    if node.uploaded {
                        let required = uploaded_seq_nos.len().checked_add(batch.len()).ok_or_else(|| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "uploaded sequence overflow"))?;
                        if required > uploaded_seq_nos.capacity() {
                            let bytes = required.checked_mul(std::mem::size_of::<i64>()).ok_or_else(|| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "uploaded sequence overflow"))?;
                            uploaded_memory.grow(bytes as u64, "uploaded sequence replacement")?;
                            uploaded_seq_nos.try_reserve_exact(required - uploaded_seq_nos.len()).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("uploaded sequence allocation: {e}")))?;
                        }
                        uploaded_seq_nos.extend(batch.iter().map(ReplicaOp::seq_no));
                    }
                    Ok(())
                })();
                let appended = match prepare { Ok(()) => round_trip.ops_mut().try_append(batch), Err(e) => Err((e, batch)) };
                if let Err((error, batch)) = appended {
                    node.ops = Some(batch);
                    node.next = requeued;
                    // Restore the entire unconsumed queue without allocation.
                    let mut queue = self.requeued.lock();
                    let mut end = &mut node.next;
                    while let Some(next) = end { end = &mut next.next; }
                    *end = queue.take();
                    *queue = Some(node);
                    return Err(error.to_string()); // round_trip retains/requeues its prefix
                }
            }
            uploaded_seq_nos.sort_unstable();
            round_trip.ops_mut().sort_by_key(ReplicaOp::seq_no);
            let ops = round_trip.ops();
            if ops.is_empty() && pump.local.checkpoint() < seq_no {
                // Concurrent append-only batches reserve sequence numbers before
                // their private text writers reach WAL publication. A temporary
                // gap is NOT evidence of a failed operation: synthesizing no-ops
                // here lets replicas acknowledge and then discard the real row.
                // Only durable primary history may supply a no-op. Wait for the
                // missing record, or fail the request without advancing its ack.
                empty_polls += 1;
                if empty_polls >= 15_000 {
                    return Err(format!("timed out waiting for primary WAL through seq_no {seq_no}; checkpoint {}", pump.local.checkpoint()));
                }
                drop(pump);
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                continue;
            }
            empty_polls = 0;
            if ops.is_empty() {
                // Every op through `seq_no` has left the tailer, but some
                // in-sync copy has not acknowledged it yet. The ops may be
                // riding on another writer's round trip: wait for that one
                // to be processed instead of judging the copy (ES fails a
                // copy only on a failure response or a timeout of an actual
                // request, never because a concurrent request is in flight).
                if self.round_trips.load(Ordering::SeqCst) > 0 {
                    drop(pump);
                    let _ = done.changed().await;
                    continue;
                }
            }
            // From here the batch is owned by its round trip: if this request
            // ends before the replicas answer, the next writer ships it.
            // With no ops this is a probe: the copies report their checkpoint.
            round_trip.start();
            // Stateless: durable in the object store before anything else
            // (ES acknowledges a write once its translog is uploaded). The
            // upload is retried until it succeeds: the batch has left the
            // translog tailer, so giving up would leave a hole.
            if let (Some(os), false) = (self.object_store.get(), round_trip.ops().is_empty()) {
                let upload_owned: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>;
                let upload: &[ReplicaOp] = if uploaded_seq_nos.is_empty() { round_trip.ops() } else {
                    upload_owned = ReplicaOp::try_copy_selected(round_trip.ops(), |o| uploaded_seq_nos.binary_search(&o.seq_no()).is_err()).map_err(|e| e.to_string())?;
                    &upload_owned
                };
                let mut delay = std::time::Duration::from_millis(50);
                while let Err(e) = os.append(upload, self.term).await {
                    tracing::warn!(index = %self.index, shard = self.shard, "object store upload failed, retrying in {delay:?}: {e}");
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(std::time::Duration::from_secs(5));
                }
                round_trip.node.as_mut().unwrap().uploaded = true;
                if os.wants_commit() {
                    let (os, svc, shard, term, gate) = (os.clone(), self.svc.clone(), self.shard, self.term, self.gate.clone());
                    tokio::spawn(async move {
                        if let Err(e) = os.commit(svc, shard, term, gate).await {
                            tracing::warn!(shard, "object store commit failed: {e}");
                        }
                    });
                }
            }
            let ops = round_trip.ops();
            let had_ops = !ops.is_empty();
            for op in ops {
                pump.local.mark_processed(op.seq_no());
                if matches!(op, ReplicaOp::Delete { .. }) {
                    pump.max_seq_no_of_updates = pump.max_seq_no_of_updates.max(op.seq_no());
                }
            }
            let local_cp = pump.local.checkpoint();
            self.tracker.lock().update_local_checkpoint(&self.allocation_id, local_cp);
            // Targets without an address are recovering copies that have not
            // pulled their first batch yet; the recovery's translog phase
            // covers the ops they miss here.
            let target_ids: Vec<String> = self.tracker.lock().replication_targets().into_iter().filter(|a| a != &self.allocation_id).collect();
            let gcp = self.tracker.lock().global_checkpoint();
            // Version first: a version older than the mapping read after it
            // is safe (the replica may re-apply), a newer one is not.
            let mapping_version = self.svc.mapping_version();
            let (mapping, _mapping_memory) = self.mapping_json()?;
            let targets = self.targets.lock().try_snapshot().map_err(|e|e.to_string())?;
            let msu = pump.max_seq_no_of_updates.max(self.copy.shard.max_seq_no_of_updates());
            let mut sends = Vec::new();
            for aid in &target_ids {
                let Some(t) = targets.get(aid).cloned() else { continue };
                let send_mapping = pump.mapping_sent.get(aid) != Some(&mapping);
                let req = ReplicationRequest::try_new(&self.index, self.shard, aid, self.term, gcp, msu,
                    send_mapping.then_some(mapping.as_str()), ReplicaOp::try_clone_batch(ops).map_err(|e| e.to_string())?, mapping_version)
                    .map_err(|e| e.to_string())?;
                sends.push(async move { (t.clone(), net.replicate(&t, req).await) });
            }
            // The batch has left the tailer in order; replicas apply batches
            // in any order (their WAL / checkpoint tracker fill gaps). Release
            // the pump for the round trip (up to the replicate timeout), so
            // later writes keep polling and shipping their own batches meanwhile.
            drop(pump);
            let results = futures::future::join_all(sends).await;
            let mut stale = Vec::new();
            let mut acked = Vec::new();
            let mut pushed_back = Vec::new();
            for (t, r) in results {
                // A replica that has not got the committed mapping yet (it
                // never derives one) is sent the batch again with the
                // mapping, before it is failed (ES retries the replica op).
                let r = match r {
                    Err(e) if e.contains(crate::mapping_update::MAPPING_NOT_ARRIVED) => {
                        let (mapping, _mapping_memory) = self.mapping_json()?;
                        let req = ReplicationRequest::try_new(&self.index, self.shard, &t.allocation_id, self.term, gcp, msu,
                            Some(&mapping), ReplicaOp::try_clone_batch(ops).map_err(|e| e.to_string())?, self.svc.mapping_version())
                            .map_err(|e| e.to_string())?;
                        net.replicate(&t, req).await
                    }
                    other => other,
                };
                match r {
                    Ok(resp) => {
                        self.tracker.lock().update_local_checkpoint(&t.allocation_id, resp.local_checkpoint);
                        refused_since.remove(&t.allocation_id);
                        acked.push(t.allocation_id.clone());
                    }
                    // A breaker refusal: the batch goes again shortly
                    // (replicas apply by seq_no, a repeat is harmless), but
                    // only for a bounded window; then the copy is failed.
                    Err(e) if is_transient_refusal(&e) => {
                        let since = *refused_since.entry(t.allocation_id.clone()).or_insert_with(std::time::Instant::now);
                        if since.elapsed() < TRANSIENT_REFUSAL_WINDOW {
                            pushed_back.push((t, e));
                        } else {
                            refused_since.remove(&t.allocation_id);
                            tracing::warn!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "replica on [{}] kept refusing for {TRANSIENT_REFUSAL_WINDOW:?}", t.node_id);
                            stale.push((t, e));
                        }
                    }
                    Err(e) => stale.push((t, e)),
                }
            }
            round_trip.delivered = pushed_back.is_empty();
            if !acked.is_empty() {
                let mut pump = self.pump.lock().await;
                for aid in acked {
                    pump.mapping_sent.insert(aid.clone(), mapping.clone());
                    let sent = pump.gcp_sent.entry(aid).or_insert(gcp);
                    *sent = (*sent).max(gcp);
                }
            }
            for (t, e) in stale {
                // ES `ReplicationOperation.performOnReplica` failure: the
                // copy is always failed (marked stale) through the master;
                // `_shards.failures` names it unless the copy simply was not
                // there (`isShardNotAvailableException`).
                tracing::warn!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "replication to [{}] failed: {e}", t.node_id);
                net.fail_copy(&self.index, self.shard, &t.allocation_id, self.term, &format!("failed to perform indices:data/write/replica on replica: {e}")).await?;
                self.tracker.lock().remove(&t.allocation_id);
                self.targets.lock().remove(&t.allocation_id);
                if !is_shard_not_available(&e) {
                    failures.push(ReplicaFailure::new(&self.index, self.shard, &t.node_id, &e));
                }
            }
            drop(round_trip);
            if !pushed_back.is_empty() {
                for (t, e) in &pushed_back {
                    tracing::debug!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "replication to [{}] refused, retrying in {backoff:?}: {e}", t.node_id);
                }
                // Writers waiting for this ack hold their requests (and
                // their indexing-pressure bytes) meanwhile, at most for
                // `TRANSIENT_REFUSAL_WINDOW` per copy.
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(BACKPRESSURE_MAX_DELAY);
                continue;
            }
            backoff = BACKPRESSURE_DELAY;
            if !had_ops {
                // Nothing new to send and nothing in flight, yet an in-sync
                // copy answered the probe below `seq_no`. Judge it on
                // checkpoint progress over time: fail it only once its
                // checkpoint has not moved for `LAG_STALL_TIMEOUT` (it cannot
                // catch up through us), and otherwise probe again shortly.
                let lagging: Vec<(String, i64)> = self.tracker.lock().lagging(seq_no).into_iter().filter(|(a, _)| a != &self.allocation_id).collect();
                let targets = self.targets.lock().try_snapshot().map_err(|e|e.to_string())?;
                let mut still_lagging = false;
                for (aid, cp) in lagging {
                    if !self.tracker.lock().is_in_sync(&aid) {
                        continue;
                    }
                    // An in-sync copy the routing table no longer has (its
                    // node left or restarted: the copy there, if any, has a
                    // new allocation id) can never be reached. ES marks it
                    // stale before acknowledging (`markUnavailableShardsAsStale`):
                    // no failure of this write, and not one of its copies.
                    if self.tracker.lock().is_unavailable_in_sync(&aid) {
                        net.fail_copy(&self.index, self.shard, &aid, self.term, "mark copy as stale").await?;
                        self.tracker.lock().remove(&aid);
                        lag_seen.remove(&aid);
                        continue;
                    }
                    // A routed copy without an address yet (the state that
                    // gives it one is being applied) is judged like any
                    // lagging copy, on checkpoint progress over time.
                    {
                        let now = std::time::Instant::now();
                        let (seen, since) = lag_seen.entry(aid.clone()).or_insert((cp, now));
                        if cp > *seen {
                            (*seen, *since) = (cp, now);
                        }
                        if now.duration_since(*since) < LAG_STALL_TIMEOUT {
                            still_lagging = true;
                            continue;
                        }
                    }
                    let node = targets.get(&aid).map(|t| t.node_id.clone()).or_else(|| self.tracker.lock().routed_node(&aid)).unwrap_or_default();
                    let error = format!("[illegal_state_exception] replica [{aid}] did not acknowledge operations through seq_no [{seq_no}] (checkpoint [{cp}]) for [{LAG_STALL_TIMEOUT:?}]");
                    tracing::warn!(index = %self.index, shard = self.shard, copy = %aid, "failing the replica on [{node}]: {error}");
                    net.fail_copy(&self.index, self.shard, &aid, self.term, &format!("failed to perform indices:data/write/replica on replica: {error}")).await?;
                    self.tracker.lock().remove(&aid);
                    self.targets.lock().remove(&aid);
                    lag_seen.remove(&aid);
                    failures.push(ReplicaFailure::new(&self.index, self.shard, &node, &error));
                }
                if still_lagging {
                    tokio::time::sleep(LAG_PROBE_INTERVAL).await;
                }
            }
        }
        // Fleet and persisted engine metadata read the physical copy's
        // checkpoint, not this replication tracker. Publish only the
        // acknowledged global checkpoint, never the primary's local maximum.
        // Also covers the already-acked fast path above.
        let tracker = self.tracker.lock();
        self.copy.update_global_checkpoint(tracker.global_checkpoint());
        // ES `ReplicationOperation`: successful = the primary and every
        // replication target that acknowledged (in-sync copies all have, or
        // were failed above; a recovering copy may have too); total = every
        // copy the routing table lists (replication targets and skipped
        // shards: unassigned or not yet tracked); failed copies are among
        // them. Stateless search copies are not promotable: not counted.
        let acked = tracker.acked_copies(seq_no) as u32;
        let failed = failures.len() as u32;
        let total = match tracker.routing_copies() {
            Some(copies) if self.object_store.get().is_none() => copies.max(acked + failed),
            _ => tracker.in_sync_ids().len() as u32 + failed,
        };
        let successful = if self.object_store.get().is_none() { acked } else { tracker.in_sync_ids().len() as u32 };
        Ok(ShardsInfo { total, successful, failures })
    }

    /// Refresh every replica copy this primary replicates to (ES
    /// `WriteReplicaResult` with `RefreshPolicy.IMMEDIATE` / `WAIT_UNTIL`):
    /// after [`Self::replicate_through`] the ops are on the replicas; this
    /// makes them searchable there before the write is acknowledged, and
    /// moves into `info` what did not: in ES the refresh is part of the
    /// replica operation, so an in-sync copy that failed to refresh failed
    /// the write's replica operation. It is no longer successful, is
    /// reported in `_shards.failures` and is failed (marked stale), as a copy
    /// that failed the replication itself. A copy that was not there to
    /// refresh (gone, or not yet recovering: ES
    /// `isShardNotAvailableException`) is no failure, but not successful.
    /// An error: a copy that did not refresh could not be failed either (the
    /// master did not commit its removal); the write must not be acked, as
    /// for a replication failure.
    pub async fn refresh_replicas(&self, net: &dyn ReplicationNet, info: &mut ShardsInfo) -> Result<(), String> {
        let (target_ids, in_sync): (Vec<String>, Vec<String>) = {
            let tracker = self.tracker.lock();
            (tracker.replication_targets().into_iter().filter(|a| a != &self.allocation_id).collect(), tracker.in_sync_ids())
        };
        let targets = self.targets.lock().try_snapshot().map_err(|e|e.to_string())?;
        // Build every request before dispatch: local admission denial must not
        // be reported as a remote replica failure or mark a healthy copy stale.
        let mut prepared=crate::shared_sequence::Builder::new().map_err(|e|e.to_string())?;
        for aid in &target_ids {
            let Some(target)=targets.get(aid) else{continue};
            let req=ReplicaRefreshReq{index:crate::front::SharedText::try_new(&self.index).map_err(|e|e.to_string())?,shard:self.shard,target_allocation_id:crate::front::SharedText::try_new(&target.allocation_id).map_err(|e|e.to_string())?};
            prepared.push((target.clone(),req)).map_err(|e|e.to_string())?;
        }
        let prepared=prepared.finish().map_err(|e|e.to_string())?;
        let mut sends=prepared.iter().map(|(target,req)|{let target=target.clone();let req=req.clone();async move{let result=net.refresh_replica(&target,req).await;(target,result)}});
        async fn optional<F:std::future::Future>(future:Option<F>)->Option<F::Output>{match future{Some(f)=>Some(f.await),None=>None}}
        loop {
        let first=sends.next();if first.is_none(){break}
        let results=tokio::join!(optional(first),optional(sends.next()),optional(sends.next()),optional(sends.next()));
        for (t,r) in [results.0,results.1,results.2,results.3].into_iter().flatten() {
            let Err(e) = r else { continue };
            tracing::warn!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "refresh of the replica on [{}] failed: {e}", t.node_id);
            if !in_sync.contains(&t.allocation_id) {
                continue;
            }
            info.successful = info.successful.saturating_sub(1);
            if refresh_target_missing(&e) {
                continue;
            }
            if let Err(f) = net.fail_copy(&self.index, self.shard, &t.allocation_id, self.term, &format!("failed to refresh the replica after a write: {e}")).await {
                tracing::warn!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "could not fail the copy that did not refresh: {f}");
                return Err(format!("the copy on [{}] did not refresh ({e}) and could not be failed: {f}", t.node_id));
            }
            self.tracker.lock().remove(&t.allocation_id);
            self.targets.lock().remove(&t.allocation_id);
            info.failures.push(ReplicaFailure::new(&self.index, self.shard, &t.node_id, &e));
        }
        }
        Ok(())
    }

    /// Ship everything currently in the translog (used before flushes and
    /// by the background loop so replicas never fall far behind).
    /// The shard's writer lock can be held by a long seal/merge. Reading its
    /// sequence number must not pin an async request/coordination worker.
    pub async fn max_seq_no_async(&self) -> i64 {
        let copy = self.copy();
        tokio::task::spawn_blocking(move || copy.shard.max_seq_no()).await.expect("sequence-number worker panicked")
    }

    /// ES `GlobalCheckpointSyncAction`: replicas learn the global checkpoint
    /// piggybacked on the next batch, so after the last write they keep the
    /// value sent before that batch was acknowledged. Send copies that are
    /// behind an operation-free request carrying the current one (they
    /// persist it in their WAL), so a restarted replica can prove its
    /// committed history acknowledged and reuse its files instead of being
    /// rebuilt. Runs only between batches (pump lock); a failed send is
    /// retried on the next tick, and replication itself reports failures.
    pub async fn sync_global_checkpoint(&self, net: &dyn ReplicationNet) {
        // One sync at a time (a tick finding one in flight skips).
        if self.gcp_sync_running.swap(true, Ordering::AcqRel) { return; }
        struct Running<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Running<'_> {
            fn drop(&mut self) { self.0.store(false, Ordering::Release); }
        }
        let _running = Running(&self.gcp_sync_running);
        // A batch in flight carries the checkpoint itself: never wait for it.
        let Ok(pump) = self.pump.try_lock() else { return };
        let gcp = self.tracker.lock().global_checkpoint();
        let target_ids: Vec<String> = self.tracker.lock().replication_targets().into_iter().filter(|a| a != &self.allocation_id).collect();
        let Ok(targets) = self.targets.lock().try_snapshot() else{return};
        let mut sends = Vec::new();
        for aid in target_ids {
            if pump.gcp_sent.get(&aid).is_some_and(|sent| *sent >= gcp) {
                continue;
            }
            let Some(t) = targets.get(&aid).cloned() else { continue };
            let Ok(ops) = elasticsearch_engine::translog::AdmittedBatch::new(elasticsearch_engine::memory::breakers()) else { return };
            let Ok(req) = ReplicationRequest::try_new(&self.index, self.shard, &aid, self.term, gcp,
                pump.max_seq_no_of_updates.max(self.copy.shard.max_seq_no_of_updates()), None, ops, 0) else { return };
            sends.push(async move { (t.clone(), net.replicate(&t, req).await) });
        }
        // Released for the round trip (up to the replicate timeout): writers
        // keep shipping their batches meanwhile.
        drop(pump);
        if sends.is_empty() { return; }
        let results = futures::future::join_all(sends).await;
        let mut pump = self.pump.lock().await;
        for (t, r) in results {
            match r {
                Ok(resp) => {
                    let sent = pump.gcp_sent.entry(t.allocation_id.clone()).or_insert(gcp);
                    *sent = (*sent).max(gcp);
                    self.tracker.lock().update_local_checkpoint(&t.allocation_id, resp.local_checkpoint);
                }
                Err(e) => tracing::debug!(index = %self.index, shard = self.shard, copy = %t.allocation_id, "global checkpoint sync to [{}] failed: {e}", t.node_id),
            }
        }
    }

    /// Fail every tracked copy that needs an op below `available_from`, the
    /// oldest op the translog still holds (ES: a replica whose history the
    /// primary no longer retains is failed and recovers from a file copy;
    /// retained ops never go missing silently). Each is reported in
    /// `failures` and removed from the tracker and the targets.
    async fn fail_copies_missing_history(&self, needed_from: i64, available_from: i64, net: &dyn ReplicationNet, failures: &mut Vec<ReplicaFailure>) -> Result<(), String> {
        let behind: Vec<(String, i64)> = self.tracker.lock().lagging(available_from - 1).into_iter().filter(|(a, _)| a != &self.allocation_id).collect();
        let targets = self.targets.lock().try_snapshot().map_err(|e| e.to_string())?;
        for (aid, cp) in behind {
            let node = targets.get(&aid).map(|t| t.node_id.clone()).or_else(|| self.tracker.lock().routed_node(&aid)).unwrap_or_default();
            let error = format!("[retention_lease_not_found_exception] replica [{aid}] (checkpoint [{cp}]) needs operations from seq_no [{}], but the primary's translog holds them only from [{available_from}] (the replication tailer needed [{needed_from}])", cp + 1);
            tracing::warn!(index = %self.index, shard = self.shard, copy = %aid, "failing the replica on [{node}]: {error}");
            net.fail_copy(&self.index, self.shard, &aid, self.term, &format!("failed to perform indices:data/write/replica on replica: {error}")).await?;
            self.tracker.lock().remove(&aid);
            self.targets.lock().remove(&aid);
            failures.push(ReplicaFailure::new(&self.index, self.shard, &node, &error));
        }
        Ok(())
    }

    pub async fn replicate_all(&self, net: &dyn ReplicationNet) -> Result<ShardsInfo, String> {
        let max = self.max_seq_no_async().await;
        self.replicate_through(max, net).await
    }

    /// Block writes for a snapshot of this primary: returns the write gate
    /// (held), the max seq_no it covers and whether a tracked copy other
    /// than this one is still behind it. Every in-sync copy is caught up
    /// while the gate is held, as recovery finalize does (no write can land
    /// meanwhile), so under continuous writes an in-sync replica never looks
    /// behind; only a copy that is not in sync (still recovering) is waited
    /// for, at most `wait`, with the gate released between checks so its
    /// recovery can finish.
    pub async fn freeze_for_snapshot(&self, net: &dyn ReplicationNet, wait: std::time::Duration) -> (tokio::sync::RwLockWriteGuard<'_, ()>, i64, bool) {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let gate = self.gate.write().await;
            if let Err(e) = self.replicate_all(net).await {
                tracing::debug!(index = %self.index, shard = self.shard, "replication before snapshot failed: {e}");
            }
            let max = self.copy.shard.max_seq_no();
            let lagging = self.tracker.lock().lagging(max).iter().any(|(a, _)| a != &self.allocation_id);
            if !lagging || tokio::time::Instant::now() >= deadline {
                return (gate, max, lagging);
            }
            drop(gate);
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    pub fn global_checkpoint(&self) -> i64 {
        self.tracker.lock().global_checkpoint()
    }
}

static RECOVERY_IDS: AtomicU64 = AtomicU64::new(1);

/// A source START generation allocated before any awaited work. The map lock
/// serializes beginning and publishing attempts. The latest ID remains until
/// every overlapping attempt returns, even if the newest finishes first.
struct RecoveryAttempt<'a> {
    primary: &'a PrimaryCtx,
    target: String,
    id: u64,
}

impl<'a> RecoveryAttempt<'a> {
    fn begin(primary: &'a PrimaryCtx, target: &str) -> Self {
        let mut pending = primary.recovery_attempts.lock();
        let id = RECOVERY_IDS.fetch_add(1, Ordering::Relaxed);
        let entry = pending.entry(target.to_string()).or_insert((id, 0));
        entry.0 = id;
        entry.1 += 1;
        Self { primary, target: target.to_string(), id }
    }

    fn publish(&self, source: RecoverySource, files: Option<elasticsearch_engine::hooks::IndexFileSet>) -> Result<(), String> {
        let pending = self.primary.recovery_attempts.lock();
        if pending.get(&self.target).map(|entry| entry.0) != Some(self.id) {
            return Err("recovery START superseded by a newer attempt".into());
        }
        let mut recoveries = self.primary.recoveries.lock();
        recoveries.retain(|_, (target, _)| target != &self.target);
        recoveries.insert(self.id, (self.target.clone(), Arc::new(tokio::sync::Mutex::new(source))));
        let mut served = self.primary.recovery_files.lock();
        served.retain(|id, file| file.target != self.target && recoveries.contains_key(id));
        if let Some(files) = files { served.insert(self.id, FileServe::new(&self.target, files)); }
        Ok(())
    }
}

impl Drop for RecoveryAttempt<'_> {
    fn drop(&mut self) {
        let mut pending = self.primary.recovery_attempts.lock();
        let entry = pending.get_mut(&self.target).expect("active recovery attempt is registered");
        entry.1 -= 1;
        if entry.1 == 0 { pending.remove(&self.target); }
    }
}

/// Start serving a recovery to `target_aid`: track it (retention), then
/// snapshot under the gate so no write slips between the snapshot and the
/// translog position.
pub async fn start_recovery_source(p: &PrimaryCtx, target_aid: &str, from_seq_no: Option<i64>) -> Result<(u64, RecoveryStartInfo), String> {
    start_recovery_source_checked(p, target_aid, from_seq_no, || true).await
}

/// Revalidate routing and primary identity after asynchronous snapshot work.
/// Insert before the last check: a state change either sees and releases the
/// inserted source, or this check releases it itself. No orphaned retention pin.
pub async fn start_recovery_source_checked(p: &PrimaryCtx, target_aid: &str, from_seq_no: Option<i64>, current: impl Fn() -> bool) -> Result<(u64, RecoveryStartInfo), String> {
    if !current() { return Err("recovery source or target allocation changed".into()); }
    let attempt = RecoveryAttempt::begin(p, target_aid);

    // Seal the bulk of the shard before fencing writes, so the gated
    // snapshot below only seals what was written meanwhile. Best effort:
    // skipped when the core or seal lock is busy right now, so the write
    // gate is never delayed behind a long holder (the gated snapshot seals
    // everything anyway).
    let pre = p.copy();
    let _ = elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || pre.shard.try_seal_all()).await;
    let gate = p.gate.clone();
    let tracker = p.tracker.clone();
    let target = target_aid.to_string();
    let svc = p.svc.clone();
    let copy = p.copy();
    let shard = p.shard;
    let term = p.term;
    // Snapshot creation can flush/seal a large shard. The write gate is
    // awaited here (no thread waits for in-flight writes to drain) and the
    // snapshot then runs at once on a blocking thread holding it (never
    // queued behind other work: writes are fenced only for the snapshot).
    // The guard moves into that task, so it survives cancellation of this
    // async caller: the snapshot finishes before writes can enter the
    // protected gap.
    let guard = gate.write_owned().await;
    let src = tokio::task::spawn_blocking(move || {
        let _gate = guard;
        tracker.lock().init_tracking(&target);
        // This primary's own copy, never whatever the index service holds
        // by now (see `RecoverySource::copy`).
        match from_seq_no.and_then(|from| RecoverySource::start_from_copy(svc.clone(), copy.clone(), shard, term, RecoveryOptions::default(), from).ok()) {
            Some(source) => Ok(source),
            None => RecoverySource::start_copy(svc, copy, shard, term, RecoveryOptions::default()),
        }
    }).await.map_err(|e| format!("recovery start failed: {e}"))?;
    let src = src.map_err(|e| e.to_string())?;
    let mut info = src.info().clone();
    info.shard = p.shard;
    let id = attempt.id;
    attempt.publish(src, None)?;
    if !current() {
        forget_recovery(p, id);
        return Err("recovery source or target allocation changed".into());
    }
    Ok((id, info))
}

/// The pinned commit of one file-based recovery being served.
pub struct FileServe {
    pub target: String,
    pub files: Arc<elasticsearch_engine::hooks::IndexFileSet>,
    /// Time this recovery's chunk reads waited for the node's rate limiter.
    pub throttle_nanos: Arc<AtomicU64>,
}

impl FileServe {
    pub fn new(target: &str, files: elasticsearch_engine::hooks::IndexFileSet) -> FileServe {
        FileServe { target: target.to_string(), files: Arc::new(files), throttle_nanos: Arc::new(AtomicU64::new(0)) }
    }
}

/// Forget recovery `id` (finalized or superseded): its pinned commit,
/// retention lock and mappings go with it.
fn forget_recovery(p: &PrimaryCtx, id: u64) {
    p.recoveries.lock().remove(&id);
    p.recovery_files.lock().remove(&id);
}

/// Release the file recoveries this primary serves whose target is no
/// longer an initializing copy of the shard (`initializing`: its
/// initializing allocation ids now): ES `PeerRecoverySourceService`
/// cancels a recovery when its target shard leaves the routing table. Each
/// pins a whole commit (mappings of segments a later merge deleted, the
/// graph checkpoint) and a retention lock that blocks translog trimming.
/// Never on silence: a target still initializing may be opening the
/// received copy (a long graph catch-up) and sends nothing meanwhile.
/// Returns how many were released.
pub fn release_abandoned_recoveries(p: &PrimaryCtx, initializing: &std::collections::HashSet<String>) -> usize {
    let abandoned: Vec<(u64, String)> = p.recovery_files.lock().iter()
        .filter(|(_, f)| !initializing.contains(&f.target))
        .map(|(id, f)| (*id, f.target.clone())).collect();
    for (id, target) in &abandoned {
        tracing::info!(index = %p.index, shard = p.shard, target = %target, recovery_id = id, "released the pinned commit of an abandoned file recovery");
        forget_recovery(p, *id);
    }
    abandoned.len()
}

/// Start serving a file-based recovery to `target_aid` (ES
/// `RecoverySourceHandler.recoverToTarget`, phase 1): track it, flush (a
/// graph save first when the last commit's graph checkpoint lags more than
/// `graph_lag_ops` behind it), pin that commit (mapped files, a retention
/// lock above its local checkpoint, checksums), and position the translog
/// phase above the commit's local checkpoint. No write is fenced: every op
/// at or below that checkpoint is in the commit, every later one in the
/// retained translog. Returns the recovery id, the phase-2 start and the
/// file manifest.
pub async fn start_file_recovery_source(p: &PrimaryCtx, target_aid: &str, graph_lag_ops: i64, target_read_max: u32) -> Result<(u64, RecoveryStartInfo, crate::replication::file_recovery::AdmittedManifest), String> {
    start_file_recovery_source_checked(p, target_aid, graph_lag_ops, target_read_max, || true).await
}

pub async fn start_file_recovery_source_checked(p: &PrimaryCtx, target_aid: &str, graph_lag_ops: i64, target_read_max: u32, current: impl Fn() -> bool) -> Result<(u64, RecoveryStartInfo, crate::replication::file_recovery::AdmittedManifest), String> {
    if !current() { return Err("recovery source or target allocation changed".into()); }
    let attempt = RecoveryAttempt::begin(p, target_aid);

    if p.relocated.load(Ordering::SeqCst) {
        return Err("primary relocated: no recovery from this copy".into());
    }
    let (copy, tracker, target, svc, shard, term) = (p.copy(), p.tracker.clone(), target_aid.to_string(), p.svc.clone(), p.shard, p.term);
    let (src, files) = elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || -> Result<_, String> {
        tracker.lock().init_tracking(&target);
        // A fresh commit: the translog phase replays only what was written
        // since (ES ships its safe commit; ours is flushed for this).
        flush_within_graph_lag(&copy, graph_lag_ops)?;
        let files = copy.snapshot_files(false).map_err(|e| format!("pinning the commit: {}", e.reason))?;
        crate::replication::file_recovery::validate_file_formats(&files.manifest, target_read_max,
            |path, offset, len| files.read_range(path, offset, len).map_err(|e| e.reason))?;
        // The translog phase tails the copy whose commit was just pinned.
        let src = RecoverySource::start_from_copy(svc, copy.clone(), shard, term, RecoveryOptions::default(), files.manifest.local_checkpoint).map_err(|e| e.to_string())?;
        Ok((src, files))
    }).await.map_err(|e| format!("file recovery start failed: {}", e.reason))??;
    let mut info = src.info().clone();
    info.shard = p.shard;
    let manifest = crate::replication::file_recovery::AdmittedManifest::try_copy(&files.manifest).map_err(|e|e.to_string())?;
    let id = attempt.id;
    attempt.publish(src, Some(files))?;
    if !current() {
        forget_recovery(p, id);
        return Err("recovery source or target allocation changed".into());
    }
    Ok((id, info, manifest))
}

/// Flushes a file recovery source makes at most to bring the commit's graph
/// checkpoint within the lag limit. A requested graph save captures
/// without waiting for in-flight graph ingestion (that ingestion is left out
/// of the saved graph and reconciled at the target's open), so the first
/// flush normally saves; the others cover a failed capture.
const GRAPH_SAVE_FLUSHES: usize = 5;

/// Flush `copy` for a file recovery, so that the commit it ships holds at
/// most `graph_lag_ops` ops above its graph checkpoint (what the target's
/// open inserts into the graphs, B9; a large lag turned that open into a
/// long graph catch-up). A save is requested and the flush repeated while
/// the lag is over the limit (a save is skipped while graph ingestion is
/// in flight); a lag still over it after [`GRAPH_SAVE_FLUSHES`] is logged
/// and the commit shipped as it is.
/// Also the snapshot freeze's flush (`snapshot::shard::stage`): a file
/// restore opens the snapshotted commit as a file recovery target does.
/// Returns whether it flushed.
pub(crate) fn flush_within_graph_lag(copy: &IndexShard, graph_lag_ops: i64) -> Result<bool, String> {
    let over = || copy.shard.graph_save_lag().filter(|&lag| lag > graph_lag_ops);
    // Nothing since the last commit: it is shipped as it is (ES ships its
    // last safe commit). A flush would write a new commit point anyway (a
    // new `meta.json`), so a retried attempt would re-fetch it although
    // the shard did not change.
    if copy.shard.uncommitted_ops() <= 0 && !copy.shard.has_uncommitted_vector_changes() && over().is_none() {
        return Ok(false);
    }
    for _ in 0..GRAPH_SAVE_FLUSHES {
        if over().is_some() {
            copy.shard.request_graph_save();
        }
        copy.shard.flush().map_err(|e| format!("flush before the file copy: {}", e.reason))?;
        if over().is_none() {
            return Ok(true);
        }
    }
    if let Some(lag) = over() {
        tracing::warn!(shard = copy.id, lag, limit = graph_lag_ops, "file recovery source: the graph checkpoint still lags its commit after {GRAPH_SAVE_FLUSHES} flushes; the target catches the graphs up at open");
    }
    Ok(true)
}

/// The pinned commit of file recovery `recovery_id` of `target_aid` (and
/// its throttle counter), if it is the current one.
pub fn recovery_files(p: &PrimaryCtx, recovery_id: u64, target_aid: &str) -> Result<(Arc<elasticsearch_engine::hooks::IndexFileSet>, Arc<AtomicU64>), String> {
    current_recovery(p, recovery_id, target_aid)?;
    match p.recovery_files.lock().get(&recovery_id) {
        Some(f) if f.target == target_aid => Ok((f.files.clone(), f.throttle_nanos.clone())),
        _ => Err(format!("no file recovery [{recovery_id}] for allocation [{target_aid}] on this primary")),
    }
}

/// One chunk of a pinned file, read at its offset from the mapping (at
/// most [`MAX_CHUNK`](crate::replication::throttle::MAX_CHUNK) bytes;
/// a page fault on a cold mapping blocks only a recovery thread).
pub async fn file_chunk(files: Arc<elasticsearch_engine::hooks::IndexFileSet>, path: String, offset: u64, len: u64, memory: elasticsearch_engine::memory::Reservation) -> Result<crate::replication::file_recovery::ChunkBytes, String> {
    // The actual worker owns admission even if its awaiting caller is canceled.
    let memory = memory.into_result_owner();
    let len = len.min(crate::replication::throttle::MAX_CHUNK);
    elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || {
        let entry = files.entry(&path).ok_or_else(|| format!("no file [{path}] in the recovery's commit"))?;
        if offset > entry.len {
            return Err(format!("offset [{offset}] is past the end of [{path}] ([{}] bytes)", entry.len));
        }
        files.read_range(&path, offset, len).map(|bytes|crate::replication::file_recovery::ChunkBytes::from_admitted(bytes,memory)).map_err(|e|e.reason)
    }).await.map_err(|e| e.reason)?
}

/// ES peer-recovery retention leases (`ReplicationTracker`
/// `peer_recovery/<node id>`): for every copy this primary tracks, retain
/// the history above `min(its checkpoint, global checkpoint)` durably, so a
/// copy that restarts (or briefly leaves) catches up from operations even
/// after this primary flushed or restarted. Leases of nodes holding no
/// copy any more expire after `index.soft_deletes.retention_lease.period`
/// (12h). Blocking (durable lease file).
pub fn sync_peer_recovery_leases(p: &PrimaryCtx, assigned_nodes: &std::collections::HashSet<String>, period_millis: i64) {
    use elasticsearch_engine::hooks::retention::PEER_RECOVERY_SOURCE;
    let copy = p.copy();
    // Accounted, never refused: a skipped round renews no lease (the copies'
    // history stays retained only in memory) and expires none.
    let Ok(targets) = crate::control_memory::unrefused("peer recovery retention leases", || p.targets.lock().try_snapshot()) else{return};
    let (gcp, cps): (i64, Vec<(String, i64)>) = {
        let t = p.tracker.lock();
        (t.global_checkpoint(), targets.iter().filter(|(aid, _)| **aid != p.allocation_id).filter_map(|(aid, target)| Some((target.node_id.clone(), t.local_checkpoint(aid)?))).collect())
    };
    for (node, cp) in cps {
        let id = format!("peer_recovery/{node}");
        let retain = cp.min(gcp) + 1;
        if copy.shard.renew_retention_lease(&id, retain, PEER_RECOVERY_SOURCE).is_err() {
            // A copy rebuilt below its old lease: leases never go backwards.
            let _ = copy.shard.remove_retention_lease(&id);
            let _ = copy.shard.add_retention_lease(&id, retain, PEER_RECOVERY_SOURCE);
        }
    }
    let now = elasticsearch_engine::date::now_millis();
    for lease in copy.shard.retention_leases() {
        let Some(node) = lease.id.strip_prefix("peer_recovery/") else { continue };
        if lease.source == PEER_RECOVERY_SOURCE && !assigned_nodes.contains(node) && now.saturating_sub(lease.timestamp_millis) > period_millis {
            tracing::info!(index = %p.index, shard = p.shard, lease = %lease.id, "peer recovery retention lease expired");
            let _ = copy.shard.remove_retention_lease(&lease.id);
        }
    }
}

/// Negotiate operations-only recovery of a proven replica prefix. `current`
/// checks authoritative routing/incarnation and source identity both after
/// acquiring the write gate and after the blocking history validation.
/// `None` means the caller must use a fresh full-snapshot recovery.
pub async fn resume_recovery_source(
    p: &PrimaryCtx,
    target_aid: &str,
    prefix: &crate::replication::replica::ReplicaPrefix,
    current: impl Fn() -> bool,
) -> Result<Option<(u64, RecoveryStartInfo)>, String> {
    // The write gate is taken by the recovery task once it runs (below),
    // never while it is queued; authority is checked again after it.
    let gate = p.gate.clone();
    if !current() || prefix.primary_term == 0 || prefix.primary_term > p.term
        || prefix.checkpoint < -1 || prefix.checkpoint > prefix.global_checkpoint
        || p.relocated.load(Ordering::SeqCst)
        || p.recoveries.lock().values().any(|(aid, _)| aid == target_aid)
    {
        return Ok(None);
    }
    let attempt = RecoveryAttempt::begin(p, target_aid);
    // Keep the new copy below the required range until it actually applies
    // it. Seeding this tracker with the source's maximum would allow flush
    // to discard precisely the operations this recovery is about to read.
    p.tracker.lock().init_tracking(target_aid);
    if p.tracker.lock().local_checkpoint(target_aid).is_some_and(|cp| cp > prefix.checkpoint) {
        return Ok(None);
    }
    let (svc, copy, shard, term, checkpoint) = (p.svc.clone(), p.copy(), p.shard, p.term, prefix.checkpoint);
    // As `start_recovery_source`: the gate is awaited, not waited for on a
    // pool thread.
    let guard = gate.write_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        let _gate = guard;
        RecoverySource::resume_copy(svc, copy, shard, term, checkpoint, RecoveryOptions::default())
    }).await.map_err(|e| format!("recovery resume failed: {e}"))?;
    let src = match result {
        Ok(src) => src,
        Err(ReplicationError::OpsTrimmed { .. } | ReplicationError::IllegalState(_)) => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    if !current() || p.relocated.load(Ordering::SeqCst) {
        return Err("primary changed during replica-prefix negotiation".into());
    }
    let info = src.info().clone();
    let id = attempt.id;
    attempt.publish(src, None)?;
    if !current() || p.relocated.load(Ordering::SeqCst) {
        forget_recovery(p, id);
        return Err("primary changed during replica-prefix negotiation".into());
    }
    Ok(Some((id, info)))
}

/// The live recovery `recovery_id`, only if it is the current one for
/// `target_aid` (unknown, superseded or foreign ids are rejected).
pub fn current_recovery(p: &PrimaryCtx, recovery_id: u64, target_aid: &str) -> Result<Arc<tokio::sync::Mutex<RecoverySource>>, String> {
    match p.recoveries.lock().get(&recovery_id) {
        Some((aid, src)) if aid == target_aid => Ok(src.clone()),
        _ => Err(format!("no recovery [{recovery_id}] for allocation [{target_aid}] on this primary")),
    }
}

/// One pulled recovery batch.
#[derive(Debug, Serialize)]
pub struct RecoveryBatch {
    /// `true` while still in the snapshot phase.
    pub snapshot: bool,
    pub ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>,
    /// No more ops in this phase (snapshot exhausted / translog caught up).
    pub phase_done: bool,
    /// The source's mapping, read after the batch's ops (so it covers
    /// them); installed before the ops in both phases.
    pub mapping_json: String,
    pub global_checkpoint: i64,
    /// The cluster `mapping_version` of `mapping_json` (0 = unknown).
    /// Appended: absent from an older source's batch.
    pub mapping_version: u64,
    #[serde(skip)]
    pub memory: crate::replication::decode::DecodeMemory,
}

/// Source response owns its snapshot-production reservation through transport
/// serialization. Wire representation remains exactly RecoveryBatch.
#[derive(Serialize)]
#[serde(transparent)]
pub struct SourceRecoveryBatch {
    batch: RecoveryBatch,
}
impl std::ops::Deref for SourceRecoveryBatch {
    type Target = RecoveryBatch;
    fn deref(&self) -> &Self::Target { &self.batch }
}

/// Replay admission owns the decoded batch until all blocking work releases
/// it. This covers simultaneous target replay, not the transport decoder's
/// earlier allocation or source-side materialization (separate boundaries).
struct AdmittedRecoveryBatch {
    batch: RecoveryBatch,
    _memory: elasticsearch_engine::memory::Reservation,
}

impl RecoveryBatch {
    pub fn try_new(snapshot: bool, ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>, phase_done: bool,
        mapping: &str, global_checkpoint: i64, mapping_version: u64) -> elasticsearch_engine::error::Result<Self> {
        let memory = crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[mapping])?;
        let mut mapping_json = String::new();
        mapping_json.try_reserve_exact(mapping.len()).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("recovery mapping allocation refused: {e}")))?;
        mapping_json.push_str(mapping);
        Ok(Self { snapshot, ops, phase_done, mapping_json, global_checkpoint, mapping_version, memory })
    }
    fn retained_bytes(&self) -> u64 {
        let strings = self.ops.iter().fold(0u64, |bytes, op| bytes.saturating_add(match op {
            ReplicaOp::Index { id, source_json, routing, .. } => (id.capacity() as u64)
                .saturating_add(source_json.capacity() as u64).saturating_add(routing.as_ref().map_or(0, |r| r.capacity() as u64)),
            ReplicaOp::Delete { id, .. } => id.capacity() as u64,
            ReplicaOp::NoOp { reason, .. } => reason.capacity() as u64,
        }));
        (std::mem::size_of::<Self>() as u64)
            .saturating_add((self.ops.capacity() as u64).saturating_mul(std::mem::size_of::<ReplicaOp>() as u64))
            .saturating_add(self.mapping_json.capacity() as u64).saturating_add(strings)
    }
    /// Account the batch for its replay. Peer-recovery ops are not subject
    /// to indexing pressure or a circuit breaker on the target (ES applies
    /// them with origin `PEER_RECOVERY`): the reservation is forced under
    /// `replay`, never refused; the source's batch size bounds it.
    fn admit_replay(self, replay: &elasticsearch_engine::memory::Replay) -> Result<AdmittedRecoveryBatch, String> {
        let bytes = self.retained_bytes();
        let memory = replay.run(|| elasticsearch_engine::memory::breakers().reserve_request(bytes, "recovery replay batch"))
            .map_err(|error| error.to_string())?;
        Ok(AdmittedRecoveryBatch { batch: self, _memory: memory })
    }
}

impl std::ops::Deref for AdmittedRecoveryBatch {
    type Target = RecoveryBatch;
    fn deref(&self) -> &Self::Target { &self.batch }
}

impl<'de> Deserialize<'de> for RecoveryBatch {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Visitor, SeqAccess, MapAccess};
        use crate::replication::decode::{Text, OptionalText, Operations, DecodeMemory};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RecoveryBatch;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("RecoveryBatch") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let snapshot = seq.next_element::<bool>()?.ok_or_else(|| A::Error::invalid_length(0, &"RecoveryBatch"))?;
                let ops = seq.next_element_seed(Operations(breakers))?.ok_or_else(|| A::Error::invalid_length(1, &"RecoveryBatch"))?;
                let phase_done = seq.next_element::<bool>()?.ok_or_else(|| A::Error::invalid_length(2, &"RecoveryBatch"))?;
                let mapping_json = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(3, &"RecoveryBatch"))?;
                let global_checkpoint = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(4, &"RecoveryBatch"))?;
                let mapping_version = seq.next_element::<u64>().ok().flatten().unwrap_or(0);
                Ok(RecoveryBatch { snapshot, ops, phase_done, mapping_json, global_checkpoint, mapping_version, memory: DecodeMemory::new(memory) })
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                #[derive(serde::Deserialize)]
                #[serde(field_identifier, rename_all = "snake_case")]
                enum Field { Snapshot, Ops, PhaseDone, MappingJson, GlobalCheckpoint, MappingVersion }
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut snapshot = None;
                let mut ops = None;
                let mut phase_done = None;
                let mut mapping_json = None;
                let mut global_checkpoint = None;
                let mut mapping_version = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::Snapshot => {
                            if snapshot.is_some() { return Err(A::Error::duplicate_field("snapshot")); }
                            snapshot = Some(map.next_value::<bool>()?);
                        }
                        Field::Ops => {
                            if ops.is_some() { return Err(A::Error::duplicate_field("ops")); }
                            ops = Some(map.next_value_seed(Operations(breakers))?);
                        }
                        Field::PhaseDone => {
                            if phase_done.is_some() { return Err(A::Error::duplicate_field("phase_done")); }
                            phase_done = Some(map.next_value::<bool>()?);
                        }
                        Field::MappingJson => {
                            if mapping_json.is_some() { return Err(A::Error::duplicate_field("mapping_json")); }
                            mapping_json = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::GlobalCheckpoint => {
                            if global_checkpoint.is_some() { return Err(A::Error::duplicate_field("global_checkpoint")); }
                            global_checkpoint = Some(map.next_value::<i64>()?);
                        }
                        Field::MappingVersion => {
                            if mapping_version.is_some() { return Err(A::Error::duplicate_field("mapping_version")); }
                            mapping_version = Some(map.next_value::<u64>()?);
                        }
                    }
                }
                Ok(RecoveryBatch {
                    snapshot: snapshot.ok_or_else(|| A::Error::missing_field("snapshot"))?,
                    ops: ops.ok_or_else(|| A::Error::missing_field("ops"))?,
                    phase_done: phase_done.ok_or_else(|| A::Error::missing_field("phase_done"))?,
                    mapping_json: mapping_json.ok_or_else(|| A::Error::missing_field("mapping_json"))?,
                    global_checkpoint: global_checkpoint.ok_or_else(|| A::Error::missing_field("global_checkpoint"))?,
                    mapping_version: mapping_version.unwrap_or(0),
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("RecoveryBatch", &["snapshot", "ops", "phase_done", "mapping_json", "global_checkpoint", "mapping_version"], V)
    }
}

/// The target pulls batches only once its applier exists, so the first
/// pull is when live ops may start flowing to it (`target`).
pub async fn next_recovery_batch(p: &PrimaryCtx, recovery_id: u64, target_aid: &str, target: Target) -> Result<SourceRecoveryBatch, String> {
    let src = current_recovery(p, recovery_id, target_aid)?;
    if !p.targets.lock().contains_key(target_aid) {p.targets.lock().try_insert(target).map_err(|e|e.to_string())?;}
    let mut src = src.lock_owned().await;
    let global_checkpoint = p.global_checkpoint();
    let svc = p.svc.clone();
    // The mapping is read after the ops (it covers every field they hold),
    // its version before the mapping (an older version is safe).
    // Keep a decoded response pending if the later mapping allocation refuses;
    // the source cursor must never advance past a batch that was not handed off.
    // Reading acknowledged ops for a recovery is a replay (ES recovery
    // actions trip no breaker): accounted, never refused; the source's batch
    // size bounds it. A refusal would fail the recovering copy.
    elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || elasticsearch_engine::memory::replaying("peer recovery source", || -> Result<SourceRecoveryBatch, String> {
        let (snapshot, ops) = if let Some(pending) = src.pending_response.take() { pending } else {
            let mut memory = elasticsearch_engine::memory::breakers().reserve_request(0, "recovery source snapshot")
                .map_err(|error| error.to_string())?;
            let snapshot = src.next_snapshot_batch_guarded(|peak| {
                if peak > memory.bytes() { memory.grow(peak - memory.bytes(), "recovery source snapshot")?; }
                Ok(())
            }).map_err(|error| error.to_string())?;
            match snapshot {
                Some(ops) => (true, elasticsearch_engine::translog::AdmittedBatch::from_reserved(ops, memory)),
                None => (false, src.next_translog_batch().map_err(|e| e.to_string())?),
            }
        };
        let mapping_version = svc.mapping_version();
        let (mapping_json, memory) = match svc.mapping_json_admitted() {
            Ok(mapping) => mapping,
            Err(error) => { src.pending_response = Some((snapshot, ops)); return Err(error.to_string()); }
        };
        let phase_done = !snapshot && ops.is_empty();
        Ok(SourceRecoveryBatch { batch: RecoveryBatch { snapshot, ops, phase_done, mapping_json, global_checkpoint, mapping_version,
            memory: crate::replication::decode::DecodeMemory::new(memory) } })
    })).await.map_err(|e| e.to_string())?
}

/// Target finished: mark it in sync (retried by the caller until true).
pub fn finalize_recovery_source(p: &PrimaryCtx, recovery_id: u64, target_aid: &str, local_checkpoint: i64) -> Result<bool, String> {
    finalize_checked(p, recovery_id, target_aid, local_checkpoint, true)
}

/// Validate before touching the tracker: checkpoints are monotonic, so a
/// stale finalize reporting the old store's (higher) checkpoint would
/// otherwise mark a freshly wiped copy in sync.
fn finalize_checked(p: &PrimaryCtx, recovery_id: u64, target_aid: &str, local_checkpoint: i64, remove: bool) -> Result<bool, String> {
    current_recovery(p, recovery_id, target_aid)?;
    p.tracker.lock().update_local_checkpoint(target_aid, local_checkpoint);
    // A recovered copy joins the in-sync set only once it holds every
    // operation this primary has (not just up to the global checkpoint):
    // writes acknowledged while it recovered did not wait for it, and its
    // post-recovery refresh must cover them, or a search it serves right
    // after starting misses them (ES hands off with every op replayed).
    // The target retries finalize while live operations catch it up.
    if remove && local_checkpoint < p.copy().shard.max_seq_no() {
        return Ok(false);
    }
    let ok = p.tracker.lock().mark_in_sync(target_aid)?;
    if ok && remove {
        forget_recovery(p, recovery_id);
    }
    Ok(ok)
}

/// Primary relocation handoff (ES `relocated()`): with every write blocked,
/// ship all ops to every copy (including the relocation target), require
/// the target to be in sync, then stop accepting writes here for good.
/// After this the master may start the target as primary; writes that
/// still reach this copy are re-routed.
pub async fn handoff_primary(p: &PrimaryCtx, recovery_id: u64, target_aid: &str, net: &dyn ReplicationNet) -> Result<bool, String> {
    let _g = p.gate.write().await;
    p.replicate_all(net).await?;
    let cp = p.tracker.lock().local_checkpoint(target_aid).unwrap_or(-1);
    // Keep the recovery registered until the handoff fully succeeds so the
    // caller's retry is still recognised.
    if !finalize_checked(p, recovery_id, target_aid, cp, false)? {
        return Ok(false);
    }
    let max = p.max_seq_no_async().await;
    if p.tracker.lock().local_checkpoint(target_aid).unwrap_or(-1) < max {
        return Ok(false);
    }
    forget_recovery(p, recovery_id);
    p.relocated.store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(true)
}

/// The local role of one shard copy.
pub enum Role {
    Primary(Arc<PrimaryCtx>),
    Replica(Arc<ReplicaApplier>),
}

pub struct LocalShard {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
    pub term: u64,
    pub svc: Arc<IndexService>,
    pub role: Role,
    /// Last recovery (kept after completion for `_recovery`).
    pub recovery: Option<Arc<Mutex<Option<RecoveryState>>>>,
}

/// All local copies, keyed by (index, shard).
#[derive(Default)]
pub struct ShardManager {
    pub shards: parking_lot::RwLock<BTreeMap<(String, u32), Arc<Mutex<LocalShard>>>>,
    /// The copies in the primary role, readable without a slot's mutex
    /// (held across promotions): kept by `insert`, `remove`, `mark_primary`.
    primaries: parking_lot::RwLock<std::collections::HashSet<(String, u32)>>,
}

impl ShardManager {
    pub fn get(&self, index: &str, shard: u32) -> Option<Arc<Mutex<LocalShard>>> {
        self.shards.read().get(&(index.to_string(), shard)).cloned()
    }

    pub fn primary(&self, index: &str, shard: u32) -> Option<Arc<PrimaryCtx>> {
        let s = self.get(index, shard)?;
        let g = s.lock();
        match &g.role {
            Role::Primary(p) => Some(p.clone()),
            _ => None,
        }
    }

    pub fn replica(&self, index: &str, shard: u32, allocation_id: &str) -> Option<Arc<ReplicaApplier>> {
        let s = self.get(index, shard)?;
        let g = s.lock();
        if g.allocation_id != allocation_id {
            return None;
        }
        match &g.role {
            Role::Replica(r) => Some(r.clone()),
            _ => None,
        }
    }

    /// Install a recovering copy's applier, only if the slot still holds
    /// that exact copy (same allocation id *and* same store instance). A
    /// stale recovery task whose store was replaced must not install its
    /// applier on the new copy.
    pub fn register_recovery_applier(&self, index: &str, shard: u32, allocation_id: &str, svc: &Arc<IndexService>, a: Arc<ReplicaApplier>) -> bool {
        let Some(s) = self.get(index, shard) else { return false };
        let mut g = s.lock();
        if g.allocation_id != allocation_id || !Arc::ptr_eq(&g.svc, svc)
            || !Arc::ptr_eq(a.index(), svc) || a.copy().is_closed()
            || !svc.shard(shard).is_some_and(|copy| Arc::ptr_eq(&copy, &a.copy()))
            || !matches!(&g.role, Role::Replica(current) if Arc::ptr_eq(&current.copy(), &a.copy()))
        {
            return false;
        }
        g.role = Role::Replica(a);
        true
    }

    /// A file recovery installed its received copy: swap its applier in for
    /// the placeholder's, only if the slot still holds this recovery's
    /// placeholder (same allocation id, store instance and placeholder
    /// copy).
    pub fn replace_recovery_applier(&self, index: &str, shard: u32, allocation_id: &str, svc: &Arc<IndexService>, placeholder: &Arc<IndexShard>, a: Arc<ReplicaApplier>) -> bool {
        let Some(s) = self.get(index, shard) else { return false };
        let mut g = s.lock();
        if g.allocation_id != allocation_id || !Arc::ptr_eq(&g.svc, svc) || !Arc::ptr_eq(a.index(), svc)
            || !matches!(&g.role, Role::Replica(current) if Arc::ptr_eq(&current.copy(), placeholder))
            || !svc.shard(shard).is_some_and(|copy| Arc::ptr_eq(&copy, &a.copy()))
        {
            return false;
        }
        g.role = Role::Replica(a);
        true
    }

    pub fn insert(&self, s: LocalShard) {
        let key = (s.index.clone(), s.shard);
        let primary = matches!(s.role, Role::Primary(_));
        self.shards.write().insert(key.clone(), Arc::new(Mutex::new(s)));
        let mut primaries = self.primaries.write();
        if primary {
            primaries.insert(key);
        } else {
            primaries.remove(&key);
        }
    }

    pub fn remove(&self, index: &str, shard: u32) -> Option<Arc<Mutex<LocalShard>>> {
        self.primaries.write().remove(&(index.to_string(), shard));
        self.shards.write().remove(&(index.to_string(), shard))
    }

    /// The copy of (`index`, `shard`) took the primary role (call with the
    /// slot's new role set).
    pub fn mark_primary(&self, index: &str, shard: u32) {
        self.primaries.write().insert((index.to_string(), shard));
    }

    /// Whether the copy of (`index`, `shard`) is in the primary role, without
    /// the slot's mutex.
    pub fn is_primary(&self, index: &str, shard: u32) -> bool {
        self.primaries.read().contains(&(index.to_string(), shard))
    }

    /// Retained slot ownership for the stats worker's registry snapshot.
    pub(crate) fn admitted_all(&self)->elasticsearch_engine::error::Result<crate::shared_sequence::SharedSequence<Arc<Mutex<LocalShard>>>>{
        crate::shared_sequence::SharedSequence::try_collect(self.shards.read().values().cloned())
    }

    pub fn all(&self) -> Vec<Arc<Mutex<LocalShard>>> {
        self.shards.read().values().cloned().collect()
    }
}

/// Replica-side handling of a replication request.
pub fn handle_replicate(mgr: &ShardManager, req: ReplicationRequest) -> Result<ReplicationResponse, (String, String)> {
    let Some(r) = mgr.replica(&req.index, req.shard, &req.target_allocation_id) else {
        return Err(("shard_not_found_exception".into(), format!("no such shard copy [{}][{}] with allocation id [{}]", req.index, req.shard, req.target_allocation_id)));
    };
    r.handle_owned(req).map_err(|e| (e.es_type().to_string(), e.to_string()))
}

pub fn handle_resync(mgr: &ShardManager, req: &ResyncRequest) -> Result<ReplicationResponse, (String, String)> {
    let Some(r) = mgr.replica(&req.index, req.shard, &req.target_allocation_id) else {
        return Err(("shard_not_found_exception".into(), format!("no such shard copy [{}][{}]", req.index, req.shard)));
    };
    r.resync(req).map_err(|e| (e.es_type().to_string(), e.to_string()))
}

/// Drive a recovery on the target side. `pull` fetches the next batch from
/// the primary. Returns the started replica applier.
pub async fn run_recovery_target<F, Fut>(applier: Arc<ReplicaApplier>, info: RecoveryStartInfo, state_slot: Arc<Mutex<Option<RecoveryState>>>, applier_slot: &(dyn Fn(Arc<ReplicaApplier>) -> bool + Send + Sync), pull: F) -> Result<(Arc<ReplicaApplier>, i64), String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<RecoveryBatch, String>>,
{
    run_recovery_target_with(applier, info, None, None, state_slot, applier_slot, pull).await
}

/// [`run_recovery_target`] after a file phase (`files`: its progress, for
/// `_recovery`). `nodes`: the (source, target) node ids the recovery
/// reports (ES `RecoveryState.sourceNode` / `targetNode`).
#[allow(clippy::too_many_arguments)]
pub async fn run_recovery_target_with<F, Fut>(applier: Arc<ReplicaApplier>, info: RecoveryStartInfo, files: Option<crate::replication::recovery::FileRecoveryStats>, nodes: Option<(crate::replication::recovery::RecoveryText, crate::replication::recovery::RecoveryText)>, state_slot: Arc<Mutex<Option<RecoveryState>>>, applier_slot: &(dyn Fn(Arc<ReplicaApplier>) -> bool + Send + Sync), mut pull: F) -> Result<(Arc<ReplicaApplier>, i64), String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<RecoveryBatch, String>>,
{
    let ensure_current = |a: &Arc<ReplicaApplier>| {
        if applier_slot(a.clone()) { Ok(()) } else { Err("recovery target physical copy was replaced".to_string()) }
    };
    ensure_current(&applier)?;
    // Applying what the source ships is a replay: no circuit breaker
    // refuses or delays it (one warning per recovery past a limit). That
    // covers the target's initialization and finalization too: a refusal
    // there would fail the copy (and a reused copy would fall back to a
    // full file recovery).
    let replay = elasticsearch_engine::memory::Replay::new("peer recovery replay");
    let init_replay = replay.clone();
    let mut target = elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || init_replay.run(|| RecoveryTarget::with_applier(applier, info)))
        .await.map_err(|e| format!("target initialization task: {e}"))?.map_err(|e| format!("target initialization: {e}"))?;
    if let Some(files) = files {
        target.set_file_stats(files);
    }
    if let Some((source, dest)) = nodes {
        target.set_nodes(Some(source), Some(dest));
    }
    // Register before the first pull: from then on the primary's pump
    // ships live ops to this applier.
    ensure_current(target.applier())?;
    let mut gcp = -1;
    let mut snapshot_finished = false;
    loop {
        let b = pull().await?;
        // The index service/allocation may survive an in-place shard
        // reset. Revalidate the captured IndexShard after every network
        // wait, before even an empty batch can fast-forward/flush it.
        ensure_current(target.applier())?;
        gcp = gcp.max(b.global_checkpoint);
        // Limit each non-cancellable blocking job. Re-check allocation/store
        // ownership between chunks so a large translog response cannot keep
        // applying after the target has been removed. Dropping this future may
        // finish the currently admitted chunk, never the whole network batch.
        let done = !b.snapshot && b.phase_done;
        // Admit before mapping parsing or shard application. The reservation
        // travels with the Arc into each non-cancellable pool job, so dropping
        // the outer async future cannot release bytes still held by that job.
        let batch = Arc::new(b.admit_replay(&replay)?);
        let mapping: Option<serde_json::Value> = serde_json::from_str(&batch.mapping_json).ok();
        let mapping = Arc::new(mapping);
        for start in (0..batch.ops.len().max(1)).step_by(256) {
            // Even an immediately-ready pool job must hand control back so
            // aborting this async recovery cannot run every chunk in one poll.
            tokio::task::yield_now().await;
            ensure_current(target.applier())?;
            let batch = batch.clone();
            let mapping = mapping.clone();
            let replay = replay.clone();
            let (next, finished) = elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || replay.run(|| -> Result<_, String> {
                let ops = &batch.ops[start..(start + 256).min(batch.ops.len())];
                let mut finished = snapshot_finished;
                if batch.snapshot {
                    target.apply_snapshot_batch_mapped(ops, mapping.as_ref().as_ref(), batch.mapping_version).map_err(|e| format!("snapshot batch: {e}"))?;
                } else {
                    if !finished {
                        target.snapshot_done().map_err(|e| format!("snapshot completion: {e}"))?;
                        finished = true;
                    }
                    if !ops.is_empty() {
                        target.apply_translog_batch_mapped(ops, mapping.as_ref().as_ref(), batch.mapping_version).map_err(|e| format!("translog batch: {e}"))?;
                    }
                }
                Ok((target, finished))
            })).await.map_err(|e| e.to_string())??;
            target = next;
            snapshot_finished = finished;
            *state_slot.lock() = Some(target.state());
        }
        if done { break; }
    }
    ensure_current(target.applier())?;
    let (target, cp) = elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery, move || replay.run(|| -> Result<_, String> {
        // The translog phase delivered every op the primary has: ops this
        // copy held above its global checkpoint that it did not verify are
        // not in the primary's history (an operations-based reuse of a copy
        // that diverged). The copy fails, and recovers from files.
        target.applier().fail_unverified_suspects().map_err(|e| format!("translog phase: {e}"))?;
        let cp = target.finalize(gcp).map_err(|e| format!("target finalization: {e}"))?;
        Ok((target, cp))
    })).await.map_err(|e| e.to_string())??;
    ensure_current(target.applier())?;
    *state_slot.lock() = Some(target.state());
    Ok((target.into_applier(), cp))
}

#[cfg(test)]
mod stale_recovery_tests {
    use std::time::Duration;
    use super::*;
    use elasticsearch_engine::index::WriteReq;
    use elasticsearch_engine::shard::WriteCond;
    use serde_json::json;

    #[test]
    fn recovery_reset_removes_old_in_sync_identity_before_replacing_data() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, _p) = primary(dir.path(), "reset-proof");
        let rec = CopyRecord { index: svc.name.clone(), index_uuid: svc.uuid.clone(),
            shard: 0, allocation_id: "old-in-sync".into(), primary_term: 1, primary: false, history_uuid: None, global_checkpoint: None };
        write_copy_record(&svc, &rec).unwrap();
        assert_eq!(read_copy_record(svc.path()), Some(rec));
        reset_copy_for_recovery(&svc, 0).unwrap();
        assert!(read_copy_record(svc.path()).is_none());
        assert!(!shard_path(&svc, 0).exists());
        // An interrupted transfer must not regain the old proof just because
        // it has recreated the shard directory and written a partial batch.
        std::fs::create_dir_all(shard_path(&svc, 0)).unwrap();
        std::fs::write(shard_path(&svc, 0).join("partial-snapshot"), b"partial").unwrap();
        assert!(read_copy_record(svc.path()).is_none());
    }

    #[test]
    fn recovery_reset_preserves_data_if_history_invalidation_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, _p) = primary(dir.path(), "reset-failure");
        std::fs::create_dir(svc.path().join(COPY_FILE)).unwrap();
        assert!(reset_copy_for_recovery(&svc, 0).is_err());
        assert!(shard_path(&svc, 0).exists());
    }

    #[test]
    fn recovery_reset_drains_old_finalizer_before_invalidating_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "reset-finalizer");
        let copy = p.copy();
        let old_svc = svc.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _permit = copy.operation_permit().unwrap();
            ready_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(30));
            write_copy_record(&old_svc, &CopyRecord { index: old_svc.name.clone(),
                index_uuid: old_svc.uuid.clone(), shard: 0, allocation_id: "old-finalizer".into(),
                primary_term: 1, primary: false, history_uuid: None, global_checkpoint: None }).unwrap();
        });
        ready_rx.recv().unwrap();
        reset_copy_for_recovery(&svc, 0).unwrap();
        writer.join().unwrap();
        assert!(read_copy_record(svc.path()).is_none());
    }

    fn primary(dir: &Path, name: &str) -> (Arc<elasticsearch_engine::node::Node>, Arc<IndexService>, PrimaryCtx) {
        let engine = elasticsearch_engine::node::Node::open(dir).unwrap();
        engine.create_index(name, &json!({})).unwrap();
        let svc = engine.get_index(name).unwrap();
        let r = svc.write(vec![WriteReq::Index { id: Some("1".into()), source: r#"{"f":1}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert!(r[0].is_ok());
        let p = PrimaryCtx::new(name, 0, "aid-p", 1, svc.clone()).unwrap();
        (engine, svc, p)
    }

    fn target(aid: &str) -> Target {
        Target::try_new(aid,"n","127.0.0.1:1").unwrap()
    }

    struct CheckpointNet {
        primary_copy: Arc<IndexShard>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl ReplicationNet for CheckpointNet {
        async fn replicate(&self, _target: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
            assert_eq!(self.primary_copy.global_checkpoint(), -1,
                "the primary must not publish an unacknowledged local checkpoint");
            if self.fail { return Err("replica unavailable".into()); }
            let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap();
            Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
        }
        async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> {
            Err("master has not committed replica removal".into())
        }
    }

    /// `refresh=true|wait_for`: an in-sync replica that did not refresh is
    /// reported failed and failed (ES: the refresh is part of the replica
    /// operation); one that is not there is not counted successful.
    #[tokio::test]
    async fn a_replica_that_did_not_refresh_is_reported_and_failed() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "refresh-fail");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r", "aid-q", "aid-ok"], []);
        for a in ["aid-r", "aid-q", "aid-ok"] {
            p.targets.lock().insert(a.into(), target(a));
        }
        #[derive(Default)]
        struct Net(Mutex<Vec<String>>);
        #[async_trait::async_trait]
        impl ReplicationNet for Net {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(0);
                Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, aid: &str, _: u64, _: &str) -> Result<(), String> {
                self.0.lock().push(aid.to_string());
                Ok(())
            }
            async fn refresh_replica(&self, t: &Target, _: ReplicaRefreshReq) -> Result<(), String> {
                match t.allocation_id.as_str() {
                    "aid-r" => Err("[refresh_failed_engine_exception] disk full".into()),
                    "aid-q" => Err("[shard_not_found_exception] no such shard copy".into()),
                    _ => Ok(()),
                }
            }
        }
        let net = Net::default();
        let mut info = ShardsInfo { total: 4, successful: 4, failures: vec![] };
        p.refresh_replicas(&net, &mut info).await.unwrap();
        assert_eq!(info.successful, 2, "the primary and the copy that refreshed");
        assert_eq!(info.failures.len(), 1, "{:?}", info.failures);
        assert_eq!((info.failures[0].error_type.as_str(), info.failures[0].reason.as_str(), info.failures[0].node.as_str()), ("refresh_failed_engine_exception", "disk full", "n"));
        assert_eq!(*net.0.lock(), vec!["aid-r".to_string()]);
        assert!(!p.tracker.lock().in_sync_ids().contains(&"aid-r".to_string()));
    }

    /// M17: a copy that did not refresh and cannot be failed (the master
    /// did not commit it) means the write is not acked.
    #[tokio::test]
    async fn a_replica_that_did_not_refresh_and_cannot_be_failed_is_not_acked() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "refresh-unfailable");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        struct Net;
        #[async_trait::async_trait]
        impl ReplicationNet for Net {
            async fn replicate(&self, _: &Target, _: ReplicationRequest) -> Result<ReplicationResponse, String> {
                Ok(ReplicationResponse { local_checkpoint: 0, max_seq_no: 0, applied: 0 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> {
                Err("master not reachable".into())
            }
            async fn refresh_replica(&self, _: &Target, _: ReplicaRefreshReq) -> Result<(), String> {
                Err("[refresh_failed_engine_exception] disk full".into())
            }
        }
        let mut info = ShardsInfo { total: 2, successful: 2, failures: vec![] };
        assert!(p.refresh_replicas(&Net, &mut info).await.is_err());
        assert!(p.tracker.lock().in_sync_ids().contains(&"aid-r".to_string()), "still in sync: not failed");
    }

    #[tokio::test]
    async fn idle_primary_delivers_the_final_global_checkpoint() {
        // The batch carries the global checkpoint from before it was
        // acknowledged; the sync delivers the advanced one, once.
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "gcp-sync");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        #[derive(Default)]
        struct Recording(Mutex<Vec<(usize, i64)>>);
        #[async_trait::async_trait]
        impl ReplicationNet for Recording {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                self.0.lock().push((req.ops.len(), req.global_checkpoint));
                let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(0);
                Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Ok(()) }
        }
        let net = Recording::default();
        p.replicate_all(&net).await.unwrap();
        let r = _svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert!(r[0].is_ok());
        net.0.lock().clear();
        p.replicate_all(&net).await.unwrap();
        let gcp = p.global_checkpoint();
        let sent: Vec<(usize, i64)> = net.0.lock().clone();
        assert!(sent.len() == 1 && sent[0].0 == 1 && sent[0].1 < gcp, "the batch carried the checkpoint from before its ack: {sent:?}, now {gcp}");
        p.sync_global_checkpoint(&net).await;
        assert_eq!(net.0.lock().last(), Some(&(0, gcp)), "an op-free request delivers the acknowledged checkpoint");
        p.sync_global_checkpoint(&net).await;
        assert_eq!(net.0.lock().len(), 2, "nothing is resent once delivered");
    }

    /// H6: the global checkpoint sync does not hold the pump across its
    /// round trip: a write is shipped and acknowledged while the sync's
    /// request to the replica is still unanswered.
    #[tokio::test]
    async fn a_held_global_checkpoint_sync_does_not_block_writes() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "gcp-sync-held");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        struct HoldSync { held: std::sync::atomic::AtomicBool, entered: tokio::sync::Notify, release: tokio::sync::Notify }
        #[async_trait::async_trait]
        impl ReplicationNet for HoldSync {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                if req.ops.is_empty() && !self.held.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    self.entered.notify_one();
                    self.release.notified().await;
                }
                let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(0);
                Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Err("unexpected".into()) }
        }
        let net = HoldSync { held: Default::default(), entered: tokio::sync::Notify::new(), release: tokio::sync::Notify::new() };
        let sync = p.sync_global_checkpoint(&net);
        tokio::pin!(sync);
        tokio::select! {
            biased;
            _ = net.entered.notified() => {}
            _ = &mut sync => panic!("the held sync cannot complete"),
        }
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        let write = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(1, &net)).await
            .expect("a write waited for the global checkpoint sync's round trip");
        assert!(write.unwrap().failures.is_empty());
        net.release.notify_one();
        sync.await;
    }

    /// The pump is not held across replica round trips: while one batch's
    /// replicate call stalls, a later write polls, ships and is acknowledged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stalled_replica_round_trip_does_not_hold_back_later_batches() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-stall");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        struct StallFirst(std::sync::atomic::AtomicBool);
        #[async_trait::async_trait]
        impl ReplicationNet for StallFirst {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                if !self.0.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
                let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(-1);
                Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Ok(()) }
        }
        let net = StallFirst(std::sync::atomic::AtomicBool::new(false));
        let later = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
            assert!(r[0].is_ok());
            let started = std::time::Instant::now();
            p.replicate_through(1, &net).await.unwrap();
            started.elapsed()
        };
        let (first, elapsed) = tokio::join!(p.replicate_through(0, &net), later);
        first.unwrap();
        assert!(elapsed < Duration::from_secs(2), "a later batch waited {elapsed:?} behind a stalled round trip");
    }

    /// Review PR3 R6: under continuous writes a snapshot freeze catches the
    /// in-sync replica up with the write gate held, so it does not look
    /// behind and the freeze neither waits out its deadline nor falls back
    /// to an unflushed image.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_snapshot_freeze_under_continuous_writes_does_not_wait_for_in_sync_copies() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "freeze-writes");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        struct Acking(Mutex<i64>);
        #[async_trait::async_trait]
        impl ReplicationNet for Acking {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                let mut applied = self.0.lock();
                *applied = (*applied).max(req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(-1));
                Ok(ReplicationResponse { local_checkpoint: *applied, max_seq_no: *applied, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Err("unexpected".into()) }
        }
        // The replica holds op 0 (written before the primary context opened,
        // so never shipped: an in-sync copy has it from its recovery).
        let net = Acking(Mutex::new(0));
        p.replicate_all(&net).await.unwrap();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let writes = async {
            let mut i = 0;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                {
                    let _write = p.gate.read().await;
                    assert!(svc.write(vec![WriteReq::Index { id: Some(format!("w{i}")), source: "{}".into(), routing: None, cond: WriteCond::default() }], false)[0].is_ok());
                }
                i += 1;
                tokio::task::yield_now().await;
            }
        };
        let freeze = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let started = std::time::Instant::now();
            let (gate, max, lagging) = p.freeze_for_snapshot(&net, Duration::from_secs(5)).await;
            let elapsed = started.elapsed();
            assert_eq!(max, p.copy().shard.max_seq_no(), "no write lands while the gate is held");
            drop(gate);
            stop.store(true, std::sync::atomic::Ordering::Release);
            (elapsed, max, lagging)
        };
        let ((), (elapsed, max, lagging)) = tokio::join!(writes, freeze);
        assert!(!lagging, "the in-sync replica looked behind");
        assert!(max > 0, "writes ran before the freeze");
        assert!(elapsed < Duration::from_secs(2), "the freeze waited {elapsed:?} for an in-sync copy");
        assert!(*net.0.lock() >= max, "the replica holds every op the image covers");
    }

    /// A replica that refuses its batches: `refusals` times with `error`
    /// (`usize::MAX`: always), then applies. Records the copies failed.
    struct Refusing { error: &'static str, refusals: usize, calls: std::sync::atomic::AtomicUsize, failed: Mutex<Vec<String>> }
    impl Refusing {
        fn new(error: &'static str, refusals: usize) -> Self {
            Refusing { error, refusals, calls: Default::default(), failed: Mutex::new(Vec::new()) }
        }
    }
    #[async_trait::async_trait]
    impl ReplicationNet for Refusing {
        async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
            if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < self.refusals {
                return Err(format!("[{}] refused", self.error));
            }
            let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(-1);
            Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
        }
        async fn fail_copy(&self, _: &str, _: u32, aid: &str, _: u64, _: &str) -> Result<(), String> {
            self.failed.lock().push(aid.to_string());
            Ok(())
        }
    }

    /// ES parity: an indexing-pressure rejection on the replica fails that
    /// copy at once (it is not retried), and the write completes without it.
    #[tokio::test]
    async fn a_replica_indexing_pressure_rejection_fails_the_copy() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "replica-rejected");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = Refusing::new("es_rejected_execution_exception", usize::MAX);
        let info = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(0, &net)).await
            .expect("a rejecting replica held the write").unwrap();
        assert_eq!(net.calls.load(std::sync::atomic::Ordering::SeqCst), 1, "a rejection is not retried");
        assert_eq!(*net.failed.lock(), vec!["aid-r".to_string()]);
        // ES `ShardInfo.Failure`: the copy's index, shard and node, the
        // replica's exception and its status.
        assert_eq!(info.failures, vec![ReplicaFailure { index: "replica-rejected".into(), shard: 0, node: "n".into(),
            error_type: "es_rejected_execution_exception".into(), reason: "refused".into(), status: 429 }]);
        assert!(!p.tracker.lock().in_sync_ids().contains(&"aid-r".to_string()));
    }

    /// ES `isShardNotAvailableException`: a replica that answers that it
    /// has no such copy is still failed (marked stale), but the write does
    /// not report it.
    #[tokio::test]
    async fn a_replica_without_the_copy_is_failed_but_not_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "replica-missing");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.tracker.lock().set_routing([("aid-p", "n-p"), ("aid-r", "n")], 2);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        let net = Refusing::new("shard_not_found_exception", usize::MAX);
        let info = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(1, &net)).await.unwrap().unwrap();
        assert_eq!(*net.failed.lock(), vec!["aid-r".to_string()]);
        assert!(info.failures.is_empty(), "{:?}", info.failures);
        assert_eq!((info.total, info.successful), (2, 1));
    }

    /// ES `markUnavailableShardsAsStale`: an in-sync copy the routing table
    /// no longer lists (its node restarted, and the copy recovering there
    /// has a new allocation id) is marked stale before the write is
    /// acknowledged, without a `_shards.failures` entry (it used to be one
    /// with an empty `_node`). `_shards.total` counts every routed copy.
    #[tokio::test]
    async fn an_unavailable_in_sync_copy_is_marked_stale_without_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "replica-gone");
        // The old copy is in sync; the routing table has the primary and a
        // new, still initializing copy (not tracked: its recovery has not
        // started, so it has no address).
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-old"], ["aid-new"]);
        p.tracker.lock().set_routing([("aid-p", "n-p"), ("aid-new", "n")], 2);
        let net = Refusing::new("unused", 0);
        let info = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(0, &net)).await
            .expect("an unavailable copy held the write").unwrap();
        assert_eq!(*net.failed.lock(), vec!["aid-old".to_string()]);
        assert!(info.failures.is_empty(), "{:?}", info.failures);
        assert_eq!((info.total, info.successful), (2, 1));
        assert_eq!(p.tracker.lock().in_sync_ids(), vec!["aid-p".to_string()]);
    }

    /// A breaker refusal is retried only for a bounded window: a replica
    /// that keeps refusing is failed instead of holding the writers (and
    /// re-shipping the batch) forever.
    #[tokio::test]
    async fn a_replica_that_keeps_refusing_is_failed_after_a_bounded_retry() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "replica-refusing");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = Refusing::new("circuit_breaking_exception", usize::MAX);
        let started = std::time::Instant::now();
        let info = tokio::time::timeout(TRANSIENT_REFUSAL_WINDOW + Duration::from_secs(10), p.replicate_through(0, &net)).await
            .expect("a refusing replica held the write past the retry window").unwrap();
        assert!(started.elapsed() >= TRANSIENT_REFUSAL_WINDOW, "failed before the retry window: {:?}", started.elapsed());
        let calls = net.calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(calls > 1 && calls < 64, "a bounded number of re-ships: {calls}");
        assert_eq!(*net.failed.lock(), vec!["aid-r".to_string()]);
        assert_eq!(info.failures.len(), 1, "{:?}", info.failures);
        assert!(!p.tracker.lock().in_sync_ids().contains(&"aid-r".to_string()));
    }

    /// A transient breaker refusal is retried and the copy stays in sync.
    /// The refused batch carries a write made after the primary context
    /// opened: op 0 predates it (the tailer starts past it, the in-sync copy
    /// holds it from its recovery), so a refused probe would re-ship nothing
    /// and `Refusing` would report -1 for it.
    #[tokio::test]
    async fn a_transient_replica_refusal_is_retried_without_failing_the_copy() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "replica-transient");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        let net = Refusing::new("circuit_breaking_exception", 2);
        let info = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(1, &net)).await.unwrap().unwrap();
        assert!(info.failures.is_empty(), "{:?}", info.failures);
        assert!(net.failed.lock().is_empty());
        assert!(net.calls.load(std::sync::atomic::Ordering::SeqCst) >= 3, "both refusals were re-shipped");
        assert!(p.tracker.lock().in_sync_ids().contains(&"aid-r".to_string()));
    }

    /// The replica applies the first batch only once released and
    /// reports its applied checkpoint, also to op-free probes.
    struct HeldReplica {
        applied: Mutex<i64>,
        held: std::sync::atomic::AtomicBool,
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
        failed: Mutex<Vec<String>>,
    }
    impl HeldReplica {
        fn new() -> Self {
            HeldReplica { applied: Mutex::new(0), held: Default::default(), entered: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(), failed: Mutex::new(Vec::new()) }
        }
    }

    #[async_trait::async_trait]
    impl ReplicationNet for HeldReplica {
        async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
            if !req.ops.is_empty() && !self.held.swap(true, std::sync::atomic::Ordering::SeqCst) {
                self.entered.notify_one();
                self.release.notified().await;
            }
            let cp = {
                let mut applied = self.applied.lock();
                if let Some(max) = req.ops.iter().map(ReplicaOp::seq_no).max() {
                    *applied = (*applied).max(max);
                }
                *applied
            };
            Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
        }
        async fn fail_copy(&self, _: &str, _: u32, aid: &str, _: u64, _: &str) -> Result<(), String> {
            self.failed.lock().push(aid.to_string());
            Ok(())
        }
    }

    /// A writer whose ops already left the tailer on another writer's
    /// in-flight round trip rides on it: it never fails the healthy replica
    /// as lagging just because that round trip has not answered yet.
    #[tokio::test]
    async fn a_writer_with_nothing_to_send_waits_for_the_in_flight_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-ride");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = HeldReplica::new();
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        // Drive the first writer until its batch (seq_no 1) is in flight.
        let first = p.replicate_through(1, &net);
        tokio::pin!(first);
        tokio::select! {
            biased;
            _ = net.entered.notified() => {}
            r = &mut first => panic!("the held round trip cannot complete: {r:?}"),
        }
        // A second writer for the same seq_no finds nothing new to send.
        let second = p.replicate_through(1, &net);
        tokio::pin!(second);
        assert!(futures::poll!(&mut second).is_pending(), "a writer with nothing new to send must wait for the in-flight round trip, not judge the replica");
        assert!(net.failed.lock().is_empty(), "the healthy replica was failed as lagging: {:?}", net.failed.lock());
        net.release.notify_one();
        let (a, b) = tokio::join!(first, second);
        assert!(a.unwrap().failures.is_empty());
        assert!(b.unwrap().failures.is_empty());
        assert!(net.failed.lock().is_empty(), "the healthy replica was failed: {:?}", net.failed.lock());
        assert!(p.tracker.lock().is_in_sync("aid-r"));
        assert_eq!(p.global_checkpoint(), 1);
    }

    /// A request cancelled mid-round-trip (client gone) does not strand its
    /// batch: the ops left the tailer, so the next writer ships them and the
    /// copy stays in sync instead of being failed as lagging.
    #[tokio::test]
    async fn a_cancelled_round_trip_is_delivered_by_the_next_writer() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-cancel");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = HeldReplica::new();
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        {
            let first = p.replicate_through(1, &net);
            tokio::pin!(first);
            tokio::select! {
                biased;
                _ = net.entered.notified() => {}
                r = &mut first => panic!("the held round trip cannot complete: {r:?}"),
            }
            // The request is dropped here, its batch never answered.
        }
        let second = tokio::time::timeout(Duration::from_secs(10), p.replicate_through(1, &net)).await
            .expect("the next writer waited for the stall timeout instead of shipping the stranded batch");
        assert!(second.unwrap().failures.is_empty());
        assert!(net.failed.lock().is_empty(), "the replica was failed: {:?}", net.failed.lock());
        assert!(p.tracker.lock().is_in_sync("aid-r"));
        assert_eq!(p.global_checkpoint(), 1);
    }

    /// B5: a primary that ran without replicas (tailer left stale, its
    /// translog trimmed by a flush) gains a replica while the frame of
    /// seq_no M+2 precedes the frame of M+1: the tailer seek finds M+1 and
    /// writes keep being acknowledged (it reported M+1 trimmed, and every
    /// later write was not acknowledged).
    #[tokio::test]
    async fn gaining_a_replica_after_a_trim_with_out_of_order_frames_keeps_acking() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-stale-seek");
        let net = HeldReplica::new();
        net.held.store(true, std::sync::atomic::Ordering::SeqCst); // never hold
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        // Alone: nothing is shipped, the tailer goes stale.
        p.replicate_through(1, &net).await.unwrap();
        p.copy().shard.flush().unwrap();
        // Precondition: the flush trimmed the generation holding seq_nos 0
        // and 1, so the seek below targets an op behind a trimmed range.
        assert!(matches!(crate::replication::tailer::TranslogTailer::open_at(&shard_path(&svc, 0), 1),
            Err(ReplicationError::OpsTrimmed { .. })), "the flush did not trim the old translog generation");
        // A replica (holding everything through 1) is added.
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        *net.applied.lock() = 1;
        // Seq_nos 2 and 3 reach the translog as 3 first, then 2.
        let translog_dir = shard_path(&svc, 0).join("translog");
        let generation = std::fs::read_dir(&translog_dir).unwrap()
            .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("translog-")?.strip_suffix(".tlog")?.parse::<u64>().ok())
            .max().unwrap();
        let mut log = elasticsearch_engine::translog::Translog::open(&translog_dir, generation).unwrap();
        for seq_no in [3u64, 2] {
            log.append_many(&[elasticsearch_engine::translog::TranslogOp::Index { id: format!("x{seq_no}"), source_json: "{}".into(), seq_no, version: 1, routing: None }]).unwrap();
        }
        log.fsync().unwrap();
        let info = p.replicate_through(3, &net).await.expect("the stale tailer reported a retained op trimmed");
        assert!(info.failures.is_empty());
        assert_eq!(*net.applied.lock(), 3);
        assert!(p.tracker.lock().is_in_sync("aid-r"));
        assert_eq!(p.global_checkpoint(), 3);
    }

    /// A quiet shard right after a load: every op acknowledged, then two
    /// flushes with no write between them (the middle translog generation
    /// never holds an op and is trimmed before the tailer reaches it), then
    /// writes resume with a later batch's frame first. The replica stays in
    /// sync and every write is acknowledged. (A 2-node mixed load froze the
    /// replica here: every round reported the ops trimmed, every write to
    /// the shard was unacknowledged.)
    #[tokio::test]
    async fn writes_after_two_quiet_flushes_with_out_of_order_frames_keep_acking() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-quiet-flushes");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = HeldReplica::new();
        net.held.store(true, std::sync::atomic::Ordering::SeqCst); // never hold
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        assert!(p.replicate_through(1, &net).await.unwrap().failures.is_empty());
        assert_eq!(*net.applied.lock(), 1);
        // Quiet: two flushes, nothing written between them.
        p.copy().shard.flush().unwrap();
        p.copy().shard.flush().unwrap();
        let translog_dir = shard_path(&svc, 0).join("translog");
        let generations: Vec<u64> = std::fs::read_dir(&translog_dir).unwrap()
            .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("translog-")?.strip_suffix(".tlog")?.parse::<u64>().ok()).collect();
        assert_eq!(generations.len(), 1, "precondition: the empty middle generation was trimmed: {generations:?}");
        // Writes resume: seq_nos 2 and 3 reach the translog as 3 first.
        let mut log = elasticsearch_engine::translog::Translog::open(&translog_dir, generations[0]).unwrap();
        for seq_no in [3u64, 2] {
            log.append_many(&[elasticsearch_engine::translog::TranslogOp::Index { id: format!("x{seq_no}"), source_json: "{}".into(), seq_no, version: 1, routing: None }]).unwrap();
        }
        log.fsync().unwrap();
        let info = tokio::time::timeout(Duration::from_secs(20), p.replicate_through(3, &net)).await
            .expect("the write was never acknowledged").expect("ops above the commit point were reported trimmed");
        assert!(info.failures.is_empty(), "{info:?}");
        assert!(net.failed.lock().is_empty(), "the healthy replica was failed: {:?}", net.failed.lock());
        assert_eq!(*net.applied.lock(), 3);
        assert!(p.tracker.lock().is_in_sync("aid-r"));
        assert_eq!(p.global_checkpoint(), 3);
    }

    /// ES: a copy that needs ops the primary's translog no longer holds is
    /// failed (marked stale; the master re-allocates it and it recovers
    /// from a file copy) and replication goes on for the other copies. A
    /// 2-node mixed load saw the pump report `operations trimmed` on every
    /// round for minutes instead: every write to the shard unacknowledged,
    /// periodic flushes skipped, the replica never failed.
    #[tokio::test]
    async fn a_replica_behind_trimmed_history_is_failed_and_recovers_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-trimmed");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = HeldReplica::new();
        net.held.store(true, std::sync::atomic::Ordering::SeqCst); // never hold
        // Trim past the replica's need: no replication retention, two
        // flushes with writes between them and nothing shipped meanwhile.
        // The tailer still holds the first generation open; the second is
        // gone before it got there.
        p.copy().shard.set_replication_retention(None);
        let write = |ids: std::ops::Range<u32>| {
            let r = svc.write(ids.map(|i| WriteReq::Index { id: Some(format!("t{i}")), source: r#"{"f":1}"#.into(), routing: None, cond: WriteCond::default() }).collect(), false);
            assert!(r.iter().all(|r| r.is_ok()));
        };
        write(0..5); // seq_nos 1..=5
        p.copy().shard.flush().unwrap();
        write(5..10); // 6..=10
        p.copy().shard.flush().unwrap();
        write(10..11); // 11
        assert_eq!(p.copy().shard.max_seq_no(), 11);
        assert!(matches!(crate::replication::tailer::TranslogTailer::open_at(&shard_path(&svc, 0), 1),
            Err(ReplicationError::OpsTrimmed { .. })), "precondition: the flushes trimmed seq_nos 1..=10");
        let info = tokio::time::timeout(Duration::from_secs(20), p.replicate_through(11, &net)).await
            .expect("the pump kept retrying the trimmed range")
            .expect("replication did not go on without the copy missing history");
        assert_eq!(*net.failed.lock(), vec!["aid-r".to_string()], "the copy missing history is failed through the master");
        assert_eq!(info.failures.len(), 1, "{info:?}");
        assert!(!p.tracker.lock().replication_targets().contains(&"aid-r".to_string()));
        assert_eq!(*net.applied.lock(), 0, "nothing was shipped to the copy with a hole");
        // The master re-allocates the copy: the new one recovers from the
        // primary's files and catches up through the translog phase.
        p.tracker.lock().update_from_cluster_state(["aid-p"], ["aid-r2"]);
        let version = elasticsearch_engine::rustic_core::codec::VERSION;
        let (id, _, manifest) = start_file_recovery_source(&p, "aid-r2", 0, version).await.unwrap();
        assert!(manifest.local_checkpoint >= 10, "the shipped commit holds the trimmed ops: {}", manifest.local_checkpoint);
        let mut replayed = Vec::new();
        loop {
            let b = next_recovery_batch(&p, id, "aid-r2", target("aid-r2")).await.unwrap();
            replayed.extend(b.ops.iter().map(|op| op.seq_no()));
            if !b.snapshot && b.phase_done { break; }
        }
        assert_eq!(replayed, ((manifest.local_checkpoint + 1)..=11).collect::<Vec<_>>(), "the translog phase replays the rest");
        assert_eq!(finalize_recovery_source(&p, id, "aid-r2", 11), Ok(true));
        assert!(p.tracker.lock().is_in_sync("aid-r2"));
    }

    /// R4: once the primary engine failed (translog tragic event), the pump
    /// ships nothing more: no op of a failed primary reaches a replica
    /// through it, and the write is not acknowledged.
    #[tokio::test]
    async fn a_failed_primary_engine_replicates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "pump-tragic");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        let net = HeldReplica::new();
        net.held.store(true, std::sync::atomic::Ordering::SeqCst); // never hold
        let r = svc.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"f":2}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert_eq!(r[0].as_ref().unwrap().seq_no, 1);
        p.copy().shard.fail_next_translog_appends(1);
        let r = svc.write(vec![WriteReq::Index { id: Some("3".into()), source: r#"{"f":3}"#.into(), routing: None, cond: WriteCond::default() }], false);
        assert!(r[0].is_err());
        assert!(p.copy().shard.tragic_error().is_some());
        assert!(p.replicate_through(1, &net).await.is_err(), "a failed primary engine replicated");
        assert_eq!(*net.applied.lock(), 0, "an op of the failed primary reached the replica");
    }

    #[tokio::test]
    async fn replication_waits_for_delayed_wal_without_inventing_noops() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "delayed-wal");
        p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
        p.targets.lock().insert("aid-r".into(), target("aid-r"));
        struct NoInventedOps;
        #[async_trait::async_trait]
        impl ReplicationNet for NoInventedOps {
            async fn replicate(&self, _: &Target, req: ReplicationRequest) -> Result<ReplicationResponse, String> {
                assert!(req.ops.iter().all(|op| matches!(op, ReplicaOp::Index { .. })),
                    "a delayed real operation must never become a synthesized no-op");
                let cp = req.ops.iter().map(ReplicaOp::seq_no).max().unwrap();
                Ok(ReplicationResponse { local_checkpoint: cp, max_seq_no: cp, applied: req.ops.len() as u64 })
            }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Err("unexpected failure".into()) }
        }
        let delayed = async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let result = svc.write(vec![WriteReq::Index { id: Some("later".into()), source: "{}".into(), routing: None, cond: Default::default() }], false);
            assert_eq!(result[0].as_ref().unwrap().seq_no, 1);
        };
        let (result, ()) = tokio::join!(p.replicate_through(1, &NoInventedOps), delayed);
        result.unwrap();
        assert_eq!(p.copy().global_checkpoint(), 1);
    }

    #[tokio::test]
    async fn fleet_checkpoint_published_only_after_replica_acknowledgement() {
        for fail in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
            engine.create_index("fleet-cp", &json!({})).unwrap();
            let svc = engine.get_index("fleet-cp").unwrap();
            let p = PrimaryCtx::new("fleet-cp", 0, "aid-p", 1, svc.clone()).unwrap();
            p.tracker.lock().update_from_cluster_state(["aid-p", "aid-r"], []);
            p.targets.lock().insert("aid-r".into(), target("aid-r"));
            for id in ["a", "b"] {
                assert!(svc.write(vec![WriteReq::Index { id: Some(id.into()), source: "{}".into(), routing: None, cond: Default::default() }], false)[0].is_ok());
            }
            let net = CheckpointNet { primary_copy: p.copy(), fail };
            let result = p.replicate_through(1, &net).await;
            assert_eq!(result.is_err(), fail);
            assert_eq!(p.copy().global_checkpoint(), if fail { -1 } else { 1 });
            if !fail {
                // Repeated acknowledgements take the fast path and remain monotonic.
                p.replicate_through(1, &net).await.unwrap();
                assert_eq!(p.copy().global_checkpoint(), 1);
            }
        }
    }

    #[tokio::test]
    async fn replica_prefix_recovery_retains_missing_ops_and_checks_final_advancement() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, p) = primary(dir.path(), "reuse");
        let prefix = crate::replication::replica::ReplicaPrefix { checkpoint: 0, global_checkpoint: 0, primary_term: 1 };
        let (id, info) = resume_recovery_source(&p, "new-r", &prefix, || true).await.unwrap().unwrap();
        assert_eq!((info.snapshot_docs, info.snapshot_max_seq_no), (0, 0));
        assert!(!p.tracker.lock().all_tracked_acked(0), "negotiating cannot fabricate a target ack");
        {
            let _write = p.gate.read().await;
            assert!(svc.write(vec![WriteReq::Index { id: Some("late".into()), source: "{}".into(), routing: None, cond: Default::default() }], false)[0].is_ok());
        }
        assert_eq!(finalize_recovery_source(&p, id, "new-r", 0), Ok(false));
        let batch = next_recovery_batch(&p, id, "new-r", target("new-r")).await.unwrap();
        assert!(!batch.snapshot);
        assert_eq!(batch.ops.len(), 1);
        assert_eq!(batch.ops[0].seq_no(), 1);
        assert_eq!(finalize_recovery_source(&p, id, "new-r", 1), Ok(true));
    }

    #[tokio::test]
    async fn replica_prefix_recovery_declines_trimmed_history_and_changed_authority() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "reuse-trimmed");
        p.copy().shard.flush().unwrap();
        let prefix = crate::replication::replica::ReplicaPrefix { checkpoint: -1, global_checkpoint: -1, primary_term: 1 };
        assert!(resume_recovery_source(&p, "new-r", &prefix, || true).await.unwrap().is_none());
        assert!(p.recoveries.lock().is_empty(), "refusal cannot install a partial recovery session");
        let prefix = crate::replication::replica::ReplicaPrefix { checkpoint: 0, global_checkpoint: 0, primary_term: 1 };
        assert!(resume_recovery_source(&p, "new-r", &prefix, || false).await.unwrap().is_none());
        let newer = crate::replication::replica::ReplicaPrefix { primary_term: 2, ..prefix.clone() };
        assert!(resume_recovery_source(&p, "new-r", &newer, || true).await.unwrap().is_none());
        // The unchanged full-snapshot path remains available after refusal.
        let (id, info) = start_recovery_source(&p, "new-r", None).await.unwrap();
        assert_eq!(info.snapshot_docs, 1);
        assert!(next_recovery_batch(&p, id, "new-r", target("new-r")).await.unwrap().snapshot);
    }

    #[tokio::test]
    async fn replica_prefix_recovery_rechecks_authority_after_history_validation() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "reuse-authority");
        let prefix = crate::replication::replica::ReplicaPrefix { checkpoint: 0, global_checkpoint: 0, primary_term: 1 };
        let checks = std::sync::atomic::AtomicUsize::new(0);
        let result = resume_recovery_source(&p, "new-r", &prefix, || checks.fetch_add(1, Ordering::SeqCst) == 0).await;
        assert!(result.is_err(), "a source superseded during blocking validation must not approve reuse");
        assert!(p.recoveries.lock().is_empty());
        assert!(p.gate.try_write().is_ok());
    }

    #[tokio::test]
    async fn replica_prefix_lost_reply_allows_full_snapshot_to_supersede_session() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "reuse-lost-reply");
        let prefix = crate::replication::replica::ReplicaPrefix { checkpoint: 0, global_checkpoint: 0, primary_term: 1 };
        let (lost_id, _) = resume_recovery_source(&p, "new-r", &prefix, || true).await.unwrap().unwrap();
        // Source accepted, but the target never received the reply. A
        // repeated offer cannot silently share that unknown session.
        assert!(resume_recovery_source(&p, "new-r", &prefix, || true).await.unwrap().is_none());
        let (fallback_id, info) = start_recovery_source(&p, "new-r", None).await.unwrap();
        assert_ne!(lost_id, fallback_id);
        assert_eq!(info.snapshot_docs, 1);
        assert!(current_recovery(&p, lost_id, "new-r").is_err());
        assert!(next_recovery_batch(&p, lost_id, "new-r", target("new-r")).await.is_err());
        assert!(finalize_recovery_source(&p, lost_id, "new-r", 0).is_err());
        assert!(!p.tracker.lock().is_in_sync("new-r"));
        let batch = next_recovery_batch(&p, fallback_id, "new-r", target("new-r")).await.unwrap();
        assert!(batch.snapshot && batch.ops.len() == 1);
        assert_eq!(finalize_recovery_source(&p, fallback_id, "new-r", 0), Ok(true));
    }

    /// A blocked shard snapshot must not stop the runtime from handling
    /// timers/heartbeats. The holder has a bounded escape so the old path
    /// fails this regression rather than hanging the test process.
    #[tokio::test(flavor = "current_thread")]
    async fn recovery_snapshot_wait_does_not_block_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "snapshot-runtime");
        let copy = p.copy();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = copy.shard.core.write();
            ready_tx.send(()).unwrap();
            release_rx.recv_timeout(std::time::Duration::from_secs(2)).is_ok()
        });
        ready_rx.recv().unwrap();
        let heartbeat = async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let _ = release_tx.send(());
        };
        let (result, ()) = tokio::join!(start_recovery_source(&p, "aid-r", None), heartbeat);
        assert!(result.is_ok(), "{:?}", result.err());
        assert!(holder.join().unwrap(), "snapshot lock wait blocked the runtime heartbeat");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_snapshot_keeps_write_gate_until_blocking_work_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "snapshot-cancel");
        let p = Arc::new(p);
        let copy = p.copy();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = copy.shard.core.write();
            ready_tx.send(()).unwrap();
            release_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        });
        ready_rx.recv().unwrap();
        let source = p.clone();
        let task = tokio::spawn(async move { start_recovery_source(&source, "aid-r", None).await });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while p.gate.try_read().is_ok() { tokio::task::yield_now().await; }
        }).await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(p.gate.try_write().is_err(), "cancel released the gate while snapshot work still runs");
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        drop(tokio::time::timeout(std::time::Duration::from_secs(2), p.gate.write()).await.unwrap());
    }

    /// R5: a recovery start queued on a busy recovery pool does not hold
    /// the primary's write gate meanwhile (writes are fenced only while its
    /// snapshot runs).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn queued_recovery_start_does_not_hold_the_write_gate() {
        use elasticsearch_http::pools::{executor, run, Pool};
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "queued-recovery");
        let p = Arc::new(p);
        // Occupy every recovery thread.
        let threads = executor(Pool::Recovery).stats().threads;
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
        let mut blockers = Vec::new();
        for _ in 0..threads {
            let rx = release_rx.clone();
            blockers.push(tokio::spawn(run(Pool::Recovery, move || { let _ = rx.lock().unwrap().recv_timeout(std::time::Duration::from_secs(20)); })));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let source = p.clone();
        let task = tokio::spawn(async move { start_recovery_source(&source, "aid-r", None).await });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        // One more blocker queued behind the pre-seal: the snapshot then
        // queues behind it.
        let rx = release_rx.clone();
        blockers.push(tokio::spawn(run(Pool::Recovery, move || { let _ = rx.lock().unwrap().recv_timeout(std::time::Duration::from_secs(20)); })));
        release_tx.send(()).unwrap(); // frees a thread: the pre-seal runs, then the extra blocker
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let free = p.gate.try_read().is_ok();
        for _ in 0..threads + 1 { let _ = release_tx.send(()); }
        for b in blockers { let _ = b.await; }
        task.await.unwrap().unwrap();
        assert!(free, "the write gate was held while the recovery start waited for a recovery thread");
    }

    /// M7: recovery starts waiting for the write gate (in-flight writes)
    /// hold no recovery thread meanwhile.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recovery_starts_waiting_for_the_gate_hold_no_recovery_thread() {
        use elasticsearch_http::pools::{executor, run, Pool};
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "gate-wait");
        let p = Arc::new(p);
        // An in-flight write holds the gate.
        let write = p.gate.clone().read_owned().await;
        let threads = executor(Pool::Recovery).stats().threads;
        let starts: Vec<_> = (0..threads + 1).map(|i| {
            let source = p.clone();
            tokio::spawn(async move { start_recovery_source(&source, &format!("aid-{i}"), None).await })
        }).collect();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let probe = tokio::time::timeout(std::time::Duration::from_secs(2), run(Pool::Recovery, || 7)).await;
        drop(write);
        for s in starts { s.await.unwrap().unwrap(); }
        assert_eq!(probe.expect("a recovery thread was free").unwrap(), 7);
    }

    /// A restarted recovery supersedes the old one: the stale task can no
    /// longer pull, and its finalize (which would report the old store's
    /// checkpoint) neither bumps the checkpoint nor marks the copy in sync.
    #[tokio::test]
    async fn superseded_recovery_cannot_pull_or_finalize() {
        let dir = tempfile::tempdir().unwrap();
        let (_e, _svc, p) = primary(dir.path(), "sr");
        let (old, _) = start_recovery_source(&p, "aid-r", None).await.unwrap();
        let (new, _) = start_recovery_source(&p, "aid-r", None).await.unwrap();
        assert_ne!(old, new);
        assert!(next_recovery_batch(&p, old, "aid-r", target("aid-r")).await.is_err());
        let before = p.tracker.lock().local_checkpoint("aid-r");
        assert!(finalize_recovery_source(&p, old, "aid-r", 1_000).is_err());
        assert_eq!(p.tracker.lock().local_checkpoint("aid-r"), before, "stale finalize must not bump the checkpoint");
        assert!(!p.tracker.lock().is_in_sync("aid-r"), "stale finalize must not mark the copy in sync");
        // The live recovery still works end to end.
        assert!(next_recovery_batch(&p, new, "aid-r", target("aid-r")).await.is_ok());
        assert_eq!(finalize_recovery_source(&p, new, "aid-r", 0), Ok(true));
        assert!(p.tracker.lock().is_in_sync("aid-r"));
    }

    #[tokio::test]
    async fn stale_recovery_start_after_preparation_releases_every_pin() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, p) = primary(dir.path(), "late-start");
        for files in [false, true] {
            let checks = std::sync::atomic::AtomicUsize::new(0);
            // Initially admitted, then allocation/primary changes during the
            // awaited source preparation (before final admission).
            let current = || checks.fetch_add(1, Ordering::SeqCst) == 0;
            let result = if files {
                start_file_recovery_source_checked(&p, "aid-late", 0, elasticsearch_engine::rustic_core::codec::VERSION, current).await.map(|_| ())
            } else {
                start_recovery_source_checked(&p, "aid-late", None, current).await.map(|_| ())
            };
            assert!(result.unwrap_err().contains("allocation changed"));
            assert_eq!(checks.load(Ordering::SeqCst), 2);
            assert!(p.recoveries.lock().is_empty(), "stale start retained translog history");
            assert!(p.recovery_files.lock().is_empty(), "stale start retained a file commit");
        }
    }

    #[tokio::test]
    async fn older_slow_start_cannot_supersede_a_newer_file_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, _svc, primary) = primary(dir.path(), "start-order");
        let primary = Arc::new(primary);
        // Keep the old snapshot START waiting while a new file START finishes.
        let gate = primary.gate.clone().read_owned().await;
        let old_primary = primary.clone();
        let old = tokio::spawn(async move { start_recovery_source(&old_primary, "aid-r", None).await });
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if primary.recovery_attempts.lock().contains_key("aid-r") { break; }
                tokio::task::yield_now().await;
            }
        }).await.unwrap();
        let (new_id, _, _) = tokio::time::timeout(Duration::from_secs(5),
            start_file_recovery_source(&primary, "aid-r", 0, elasticsearch_engine::rustic_core::codec::VERSION)).await.unwrap().unwrap();
        drop(gate);
        let error = old.await.unwrap().unwrap_err();
        assert!(error.contains("superseded"), "{error}");
        assert!(current_recovery(&primary, new_id, "aid-r").is_ok());
        assert!(primary.recovery_files.lock().contains_key(&new_id));
        assert_eq!(primary.recoveries.lock().len(), 1);
        assert!(primary.recovery_attempts.lock().is_empty(), "completed attempt metadata must not accumulate");
    }

    /// A file recovery nobody will finish releases its pinned commit once
    /// its target left the initializing copies (failed, moved, removed);
    /// one still initializing keeps it however long it stays silent (it may
    /// be opening the received copy).
    #[tokio::test]
    async fn abandoned_file_recoveries_release_their_pinned_commit() {
        let dir = tempfile::tempdir().unwrap();
        let (_e, _svc, p) = primary(dir.path(), "fr-abandon");
        let (gone, _, _) = start_file_recovery_source(&p, "aid-gone", 0, elasticsearch_engine::rustic_core::codec::VERSION).await.unwrap();
        let (live, _, _) = start_file_recovery_source(&p, "aid-live", 0, elasticsearch_engine::rustic_core::codec::VERSION).await.unwrap();
        let initializing: std::collections::HashSet<String> = ["aid-live".to_string()].into();
        assert_eq!(release_abandoned_recoveries(&p, &initializing), 1);
        assert!(recovery_files(&p, gone, "aid-gone").is_err(), "the cancelled target's commit is released");
        assert!(current_recovery(&p, gone, "aid-gone").is_err());
        assert_eq!(release_abandoned_recoveries(&p, &initializing), 0, "an initializing target keeps its recovery");
        assert!(recovery_files(&p, live, "aid-live").is_ok());
        assert_eq!(release_abandoned_recoveries(&p, &std::collections::HashSet::new()), 1);
        assert!(p.recovery_files.lock().is_empty() && p.recoveries.lock().is_empty());
    }

    /// A recovering copy is not marked in sync until its checkpoint covers
    /// every operation the primary holds (its post-recovery refresh then
    /// makes all of them searchable).
    #[tokio::test]
    async fn finalize_waits_for_every_primary_operation() {
        let dir = tempfile::tempdir().unwrap();
        let (_e, svc, p) = primary(dir.path(), "fw");
        let w = svc.write(vec![elasticsearch_engine::index::WriteReq::Index { id: Some("1".into()), source: "{}".into(), routing: None, cond: Default::default() }], true);
        assert!(w[0].is_ok());
        let (id, _) = start_recovery_source(&p, "aid-r", None).await.unwrap();
        // The primary holds seq_nos 0 (the helper's document) and 1.
        assert_eq!(finalize_recovery_source(&p, id, "aid-r", 0), Ok(false), "copy missing seq_no 1 must not join the in-sync set");
        assert!(!p.tracker.lock().is_in_sync("aid-r"));
        assert_eq!(finalize_recovery_source(&p, id, "aid-r", 1), Ok(true));
        assert!(p.tracker.lock().is_in_sync("aid-r"));
    }

    #[test]
    fn recovery_replay_charge_includes_retained_capacities_and_follows_batch_lifetime() {
        let mut source = String::with_capacity(4096);
        source.push_str("{}");
        let mut ops = Vec::with_capacity(8);
        ops.push(ReplicaOp::Index { id: "a".into(), source_json: source, routing: Some("r".into()),
            seq_no: 0, version: 1, primary_term: 1 });
        let batch = RecoveryBatch::try_new(true, crate::replication::ops::admitted_ops_for_test(ops), false, "{}", -1, 1).unwrap();
        let bytes = batch.retained_bytes();
        assert!(bytes >= 4096 + (8 * std::mem::size_of::<ReplicaOp>()) as u64);
        let batch = Arc::new(batch.admit_replay(&elasticsearch_engine::memory::Replay::new("test")).unwrap());
        let held_by_blocking_job = batch.clone();
        let weak = Arc::downgrade(&batch);
        drop(batch); // async cancellation
        assert!(weak.upgrade().is_some(), "blocking job still owns its admission and payload");
        assert_eq!(held_by_blocking_job.retained_bytes(), bytes);
        drop(held_by_blocking_job);
        assert!(weak.upgrade().is_none());
    }

    /// The translog files of a copy (all generations) and their sizes.
    fn translog_sizes(dir: &Path) -> BTreeMap<PathBuf, u64> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let e = e.unwrap();
                let path = e.path();
                let name = e.file_name().to_string_lossy().into_owned();
                if e.file_type().unwrap().is_dir() { stack.push(path); }
                else if name.starts_with("translog-") && name.ends_with(".tlog") { out.insert(path, e.metadata().unwrap().len()); }
            }
        }
        out
    }

    /// A crash that loses every translog byte not yet fsynced: each file is
    /// cut back to its size at the last durable point (`durable`, taken while
    /// nothing was unsynced), and generations created after it lose all.
    /// Appends are plain write(2): without this a dropped process loses
    /// nothing, and a recorded checkpoint above the fsynced translog would
    /// still be recovered.
    fn crash_losing_unsynced(dir: &Path, durable: &BTreeMap<PathBuf, u64>) {
        for (path, len) in translog_sizes(dir) {
            let keep = durable.get(&path).copied().unwrap_or(0);
            if len > keep {
                std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(keep).unwrap();
            }
        }
    }

    /// One run of the global checkpoint durability scenario: async writes,
    /// the global checkpoint recorded (`sync_first`: as
    /// [`note_durable_primary_global_checkpoint`] does, after a translog
    /// fsync; otherwise recorded straight away, the old ordering), more
    /// async writes, then a crash losing every unsynced translog byte and a
    /// restart. Returns (recorded, recovered local checkpoint).
    async fn recorded_and_recovered_checkpoints(sync_first: bool) -> (i64, i64) {
        struct Alone;
        #[async_trait::async_trait]
        impl ReplicationNet for Alone {
            async fn replicate(&self, _: &Target, _: ReplicationRequest) -> Result<ReplicationResponse, String> { Err("no replica to replicate to".into()) }
            async fn fail_copy(&self, _: &str, _: u32, _: &str, _: u64, _: &str) -> Result<(), String> { Ok(()) }
        }
        let write = |svc: &IndexService, i: usize| {
            let r = svc.write(vec![WriteReq::Index { id: Some(format!("d{i}")), source: format!(r#"{{"n":{i}}}"#), routing: None, cond: WriteCond::default() }], false);
            assert!(r[0].is_ok());
        };
        let dir = tempfile::tempdir().unwrap();
        let (engine, svc, p) = primary(dir.path(), "gcp-durable");
        write_copy_record(&svc, &CopyRecord { index: svc.name.clone(), index_uuid: svc.uuid.clone(), shard: 0,
            allocation_id: "aid-p".into(), primary_term: 1, primary: true, history_uuid: Some("history-a".into()), global_checkpoint: None }).unwrap();
        let copy = p.copy();
        let shard_dir = copy.shard.path().to_path_buf();
        copy.shard.sync_translog().unwrap();
        assert!(!copy.shard.translog_has_unsynced());
        let mut durable = translog_sizes(&shard_dir);
        for i in 0..20 { write(&svc, i); }
        p.replicate_all(&Alone).await.unwrap();
        let gcp = p.global_checkpoint();
        assert_eq!(gcp, 20, "the global checkpoint advanced over the async writes");
        // A crash before the record loses nothing it claims: nothing yet.
        assert_eq!(read_copy_record(svc.path()).unwrap().global_checkpoint, None);
        if sync_first {
            assert_eq!(note_durable_primary_global_checkpoint(&p), Some(gcp));
            // A crash right after the record: no op at or below it is unsynced.
            assert!(!copy.shard.translog_has_unsynced(), "a global checkpoint was recorded above the durable translog");
            durable = translog_sizes(&shard_dir);
        } else {
            note_primary_global_checkpoint(&p.svc, &p.allocation_id, gcp);
        }
        assert_eq!(read_copy_record(svc.path()).unwrap().global_checkpoint, Some(gcp));
        // More async writes advance the global checkpoint in memory only.
        for i in 20..25 { write(&svc, i); }
        p.replicate_all(&Alone).await.unwrap();
        assert!(p.global_checkpoint() > gcp);
        assert_eq!(read_copy_record(svc.path()).unwrap().global_checkpoint, Some(gcp), "nothing recorded past the last sync");
        let persisted = read_copy_record(svc.path()).unwrap().global_checkpoint.unwrap();
        // Crash (no flush, no further sync, unsynced bytes lost) and restart.
        drop((copy, p, svc, engine));
        crash_losing_unsynced(&shard_dir, &durable);
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        let svc = engine.get_index("gcp-durable").unwrap();
        let recovered = svc.shard(0).unwrap().shard.local_checkpoint();
        if sync_first {
            assert_eq!(primary_checkpoint(&svc, &svc.uuid, 0, "history-a", -1), persisted);
        }
        (persisted, recovered)
    }

    /// Commit 07cbbb4d: a primary's global checkpoint is recorded (in its
    /// copy record, trusted after a restart) only once every op at or below
    /// it is durable in its translog. With async durability (writes without
    /// a per-request fsync) the ops under an advanced global checkpoint are
    /// not synced yet: a crash right after the record must not leave the
    /// copy below the value it claims, and after a crash that loses every
    /// unsynced byte the recorded checkpoint never exceeds what the
    /// translog replays.
    #[tokio::test]
    async fn a_recorded_global_checkpoint_never_exceeds_the_durable_translog() {
        let (persisted, recovered) = recorded_and_recovered_checkpoints(true).await;
        assert!(recovered >= persisted, "recorded global checkpoint [{persisted}] above the recovered local checkpoint [{recovered}]");
    }

    /// Negative control for the test above: recording the global checkpoint
    /// before the translog fsync (the ordering before 07cbbb4d) is caught,
    /// because the simulated crash really loses the unsynced ops.
    #[tokio::test]
    async fn recording_the_global_checkpoint_before_the_sync_is_caught_by_the_crash() {
        let (persisted, recovered) = recorded_and_recovered_checkpoints(false).await;
        assert!(recovered < persisted, "the crash lost no unsynced op: recorded [{persisted}], recovered [{recovered}]");
    }

    #[test]
    fn primary_restart_preserves_checkpoint_only_for_the_same_history() {
        let dir = tempfile::tempdir().unwrap();
        let (_engine, svc, _p) = primary(dir.path(), "checkpoint-restart");
        let record = CopyRecord { index: svc.name.clone(), index_uuid: svc.uuid.clone(), shard: 0,
            allocation_id: "previous-primary".into(), primary_term: 1, primary: true,
            history_uuid: Some("history-a".into()), global_checkpoint: Some(42) };
        write_copy_record(&svc, &record).unwrap();
        assert_eq!(primary_checkpoint(&svc, &svc.uuid, 0, "history-a", -1), 42);
        assert_eq!(primary_checkpoint(&svc, &svc.uuid, 0, "history-a", 50), 50);
        assert_eq!(primary_checkpoint(&svc, &svc.uuid, 0, "forced-history", -1), -1);
        assert_eq!(primary_checkpoint(&svc, "other-index", 0, "history-a", -1), -1);
        assert_eq!(primary_checkpoint(&svc, &svc.uuid, 1, "history-a", -1), -1);
        let checkpoint = primary_checkpoint(&svc, &svc.uuid, 0, "history-a", 7);
        write_copy_record(&svc, &CopyRecord { allocation_id: "new-primary".into(), primary_term: 2,
            global_checkpoint: Some(checkpoint), ..record }).unwrap();
        assert_eq!(read_copy_record(svc.path()).unwrap().global_checkpoint, Some(42));
    }

    /// Unknown ids and ids belonging to another copy are rejected before
    /// the tracker is touched.
    #[tokio::test]
    async fn unknown_or_foreign_recovery_id_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (_e, _svc, p) = primary(dir.path(), "sr2");
        let (id, _) = start_recovery_source(&p, "aid-a", None).await.unwrap();
        p.tracker.lock().init_tracking("aid-b");
        assert!(finalize_recovery_source(&p, id, "aid-b", 0).is_err());
        assert!(!p.tracker.lock().is_in_sync("aid-b"));
        assert!(finalize_recovery_source(&p, u64::MAX, "aid-a", 0).is_err());
        assert!(!p.tracker.lock().is_in_sync("aid-a"));
        assert!(current_recovery(&p, id, "aid-a").is_ok());
    }

    /// A stale target task can only install its applier on the exact copy
    /// (allocation id and store instance) it was recovering.
    #[test]
    fn stale_applier_cannot_activate_replaced_copy() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("old", &json!({})).unwrap();
        engine.create_index("new", &json!({})).unwrap();
        let old_svc = engine.get_index("old").unwrap();
        let new_svc = engine.get_index("new").unwrap();
        let mgr = ShardManager::default();
        let placeholder = Arc::new(ReplicaApplier::open(new_svc.clone(), 0).unwrap());
        mgr.insert(LocalShard { index: "i".into(), shard: 0, allocation_id: "aid-r".into(), term: 1, svc: new_svc.clone(), role: Role::Replica(placeholder.clone()), recovery: None });
        let stale = Arc::new(ReplicaApplier::open(old_svc.clone(), 0).unwrap());
        // Same allocation id, replaced store: rejected.
        assert!(!mgr.register_recovery_applier("i", 0, "aid-r", &old_svc, stale.clone()));
        // Different allocation id: rejected.
        assert!(!mgr.register_recovery_applier("i", 0, "aid-x", &new_svc, stale.clone()));
        assert!(Arc::ptr_eq(&mgr.replica("i", 0, "aid-r").unwrap(), &placeholder));
        // The current copy's own applier is accepted.
        let fresh = Arc::new(ReplicaApplier::open(new_svc.clone(), 0).unwrap());
        assert!(mgr.register_recovery_applier("i", 0, "aid-r", &new_svc, fresh.clone()));
        assert!(Arc::ptr_eq(&mgr.replica("i", 0, "aid-r").unwrap(), &fresh));
    }

    fn empty_recovery_info(svc: &IndexService) -> RecoveryStartInfo {
        RecoveryStartInfo::try_new(&svc.name,&svc.uuid,0,1,"{}","{}",0,-1,0,0,0).unwrap()
    }

    #[tokio::test]
    async fn recovery_rechecks_target_between_translog_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("cancel-replay", &json!({})).unwrap();
        let svc = engine.get_index("cancel-replay").unwrap();
        let applier = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        let result = run_recovery_target(applier.clone(), empty_recovery_info(&svc), Arc::new(Mutex::new(None)),
            // Model ownership revoked after the first admitted replay chunk.
            &|a| a.local_checkpoint() < 255, || async {
                RecoveryBatch::try_new(false, crate::replication::ops::admitted_ops_for_test(
                    (0..1024).map(|seq_no| ReplicaOp::NoOp { seq_no, primary_term: 1, reason: "test".into() }).collect()),
                    true, "{}", -1, 0).map_err(|e| e.to_string())
            }).await;
        assert!(result.unwrap_err().contains("physical copy was replaced"));
        assert_eq!(applier.local_checkpoint(), 255, "no remaining chunks may run after ownership is revoked");
    }

    /// A removed shard can be reopened in the SAME IndexService with the
    /// SAME allocation id. The old task must not flush its empty final
    /// batch into that replacement's directory, even after registration
    /// succeeded before the network wait.
    #[tokio::test]
    async fn recovery_empty_tail_cannot_overwrite_in_place_replacement() {
        use elasticsearch_engine::index::{RecoverySource as EngineRecovery, ShardOpenSpec};
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("reuse-lifetime", &json!({})).unwrap();
        let svc = engine.get_index("reuse-lifetime").unwrap();
        let old = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        old.bump_term(1, -1).unwrap();
        old.flush().unwrap();
        let info = empty_recovery_info(&svc);
        let manager = Arc::new(ShardManager::default());
        manager.insert(LocalShard { index: svc.name.clone(), shard: 0, allocation_id: "same-aid".into(),
            term: 1, svc: svc.clone(), role: Role::Replica(old.clone()), recovery: None });
        let slot = Arc::new(Mutex::new(None));
        let (pulled_tx, pulled_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let (task_manager, task_svc, task_old) = (manager.clone(), svc.clone(), old.clone());
        let task = tokio::spawn(async move {
            let register = |a| task_manager.register_recovery_applier(&task_svc.name, 0, "same-aid", &task_svc, a);
            let mut signals = Some((pulled_tx, release_rx));
            run_recovery_target(task_old, info, slot, &register, move || {
                let (pulled, release) = signals.take().expect("only one empty tail is expected");
                async move {
                    let _ = pulled.send(());
                    release.await.map_err(|e| e.to_string())?;
                    RecoveryBatch::try_new(false, crate::replication::ops::admitted_ops_for_test(vec![]), true, "{}", -1, 0).map_err(|e| e.to_string())
                }
            }).await
        });
        tokio::time::timeout(Duration::from_secs(5), pulled_rx).await.unwrap().unwrap();
        svc.remove_shard(0, true).unwrap();
        let replacement = svc.open_shard(0, ShardOpenSpec {
            allocation_id: "same-aid".into(), primary: true, primary_term: 1, recovery: EngineRecovery::Empty,
        }).unwrap();
        assert!(!Arc::ptr_eq(&replacement, &old.copy()));
        let written = replacement.write(vec![WriteReq::Index { id: Some("sentinel".into()),
            source: r#"{"f":"replacement must survive"}"#.into(), routing: None, cond: Default::default() }], true);
        assert!(written[0].is_ok());
        replacement.shard.flush().unwrap();
        let fresh = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        manager.insert(LocalShard { index: svc.name.clone(), shard: 0, allocation_id: "same-aid".into(),
            term: 1, svc: svc.clone(), role: Role::Replica(fresh.clone()), recovery: None });
        assert!(!manager.register_recovery_applier(&svc.name, 0, "same-aid", &svc, old.clone()));
        let before = std::fs::read(replacement.shard.path().join("shard_state.json")).unwrap();
        // The same lifecycle guard also fences stale raw-WAL/flush callers
        // that are outside run_recovery_target's registration checks.
        assert!(matches!(old.flush(), Err(crate::replication::ReplicationError::Engine(e)) if e.is_already_closed()));
        assert!(old.update_global_checkpoint(100).is_err());
        assert!(old.fast_forward_checkpoint(100).is_err());
        assert!(old.recover_from_wal().is_err());
        release_tx.send(()).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), task).await.unwrap().unwrap();
        assert!(result.unwrap_err().contains("physical copy was replaced"));
        assert!(Arc::ptr_eq(&manager.replica(&svc.name, 0, "same-aid").unwrap(), &fresh));
        assert_eq!(std::fs::read(replacement.shard.path().join("shard_state.json")).unwrap(), before);
        svc.close_shard(0).unwrap();
        let reopened = svc.open_shard(0, ShardOpenSpec {
            allocation_id: "same-aid".into(), primary: true, primary_term: 1, recovery: EngineRecovery::ExistingStore,
        }).unwrap();
        assert!(reopened.get("sentinel", None, true).unwrap().is_some());
        assert_eq!(reopened.shard.max_seq_no(), 0);
    }

    #[tokio::test]
    async fn recovery_registration_rejection_precedes_term_adoption_and_pull() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("rejected-target", &json!({})).unwrap();
        let svc = engine.get_index("rejected-target").unwrap();
        let applier = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        let result = run_recovery_target(applier.clone(), empty_recovery_info(&svc),
            Arc::new(Mutex::new(None)), &|_| false, || async { panic!("rejected registration must not pull") }).await;
        assert!(result.unwrap_err().contains("physical copy was replaced"));
        assert_eq!(applier.primary_term(), 0, "rejected copy must not mutate the WAL term");
    }

    #[tokio::test]
    async fn recovery_empty_tail_aborts_when_term_changes_during_pull() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("changed-source-term", &json!({})).unwrap();
        let svc = engine.get_index("changed-source-term").unwrap();
        let applier = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        let copy = applier.clone();
        let result = run_recovery_target(applier.clone(), empty_recovery_info(&svc), Arc::new(Mutex::new(None)),
            &|_| true, move || {
                let copy = copy.clone();
                async move {
                // Models the state applier observing a new source term
                // while the old source's final network response is pending.
                copy.bump_term(2, -1).unwrap();
                RecoveryBatch::try_new(false, crate::replication::ops::admitted_ops_for_test(vec![]), true, "{}", -1, 0).map_err(|e| e.to_string())
                }
            }).await;
        assert!(result.unwrap_err().contains("aborted: current term is [2]"));
        assert_eq!(applier.primary_term(), 2);
        assert_eq!(applier.local_checkpoint(), -1);
    }

    #[test]
    fn replica_operation_permit_drains_before_store_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index("permit-lifetime", &json!({})).unwrap();
        let svc = engine.get_index("permit-lifetime").unwrap();
        let copy = svc.shard(0).unwrap();
        let permit = copy.operation_permit().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let remover = std::thread::spawn(move || { done_tx.send(svc.remove_shard(0, true)).unwrap(); });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !copy.is_closed() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let fenced = copy.is_closed();
        let early = done_rx.try_recv();
        // An already admitted compound operation may still finish raw
        // shard/WAL work; the store remains present until it drops permit.
        let store_present = copy.shard.path().join("shard_state.json").exists();
        drop(permit);
        remover.join().unwrap();
        assert!(fenced);
        assert!(matches!(early, Err(std::sync::mpsc::TryRecvError::Empty)));
        assert!(store_present);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        assert!(copy.operation_permit().is_err());
        assert!(!copy.shard.path().exists());
    }
}

#[cfg(test)]
mod target_ownership_tests {
    use super::*;
    use elasticsearch_engine::memory::{scoped,Breakers,Kind,RequestScope};
    fn budget(n:u64)->&'static Breakers{let b=Breakers::leak_for_tests(n);b.apply_settings(&|k|(k=="indices.breaker.total.use_real_memory").then(||"false".into()));b}
    #[test]
    fn recovery_target_and_snapshot_retain_actual_backing(){
        let b=budget(1<<20);let scope=RequestScope::new("target snapshot",b);
        let (mut targets,snapshot)=scoped(Some(scope.clone()),||{
            let target=Target::try_new("aid","node","127.0.0.1:9300").unwrap();
            let mut targets=TargetMap::default();targets.try_insert(target.clone()).unwrap();
            let snapshot=targets.try_snapshot().unwrap();
            assert!(Arc::ptr_eq(&target.0,&snapshot.get("aid").unwrap().0));
            (targets,snapshot)
        });
        drop(scope);targets.remove("aid");
        assert!(b.get(Kind::Accounting).used()>0);
        drop(targets);assert_eq!(snapshot.get("aid").unwrap().node_id,"node");
        assert!(b.get(Kind::Accounting).used()>0);drop(snapshot);
        assert_eq!(b.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn denied_target_does_not_publish_partial_metadata(){
        let b=budget(256);let scope=RequestScope::new("target denial",b);
        assert!(scoped(Some(scope.clone()),||Target::try_new("aid",&"n".repeat(1024),"addr")).is_err());
        drop(scope);assert_eq!(b.get(Kind::Accounting).used(),0);
    }
    /// Replication target metadata is control-plane work: under
    /// `unrefused` the same denied target is admitted, accounted, inserted
    /// and snapshotted, and its memory is released with it.
    #[test]
    fn control_target_metadata_is_accounted_not_refused(){
        let b=budget(256);let scope=RequestScope::new("target control",b);
        let (targets,snapshot)=scoped(Some(scope.clone()),||crate::control_memory::unrefused("replication target metadata",||{
            let target=Target::try_new("aid",&"n".repeat(1024),"addr").unwrap();
            let mut targets=TargetMap::default();targets.try_insert(target).unwrap();
            let snapshot=targets.try_snapshot().unwrap();
            (targets,snapshot)
        }));
        assert!(b.get(Kind::Accounting).used()>=1024);
        assert_eq!(snapshot.get("aid").unwrap().node_id.len(),1024);
        drop(targets);drop(snapshot);drop(scope);
        assert_eq!(b.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn replica_refresh_preserves_legacy_wire_and_owned_decode(){
        let b=budget(1<<20);let scope=RequestScope::new("refresh wire",b);
        let wire=bincode::serialize(&("index",3u32,"allocation")).unwrap();
        let req:ReplicaRefreshReq=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();
        assert_eq!(bincode::serialize(&req).unwrap(),wire);drop(wire);drop(scope);
        assert!(b.get(Kind::Request).used()>0);assert_eq!(&*req.index,"index");drop(req);
        assert_eq!(b.get(Kind::Request).used(),0);
    }
}
