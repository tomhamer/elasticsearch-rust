//! Applying replicated operations on a replica shard copy.
//!
//! A [`ReplicaApplier`] owns, for one engine `IndexService` acting as a
//! replica:
//!
//! * the **replica WAL** (`<shard path>/cluster-replica-wal/`): every batch
//!   is appended and fsynced *before* it is applied and before anything is
//!   acked, because the engine's `replay_index`/`replay_delete` do not
//!   write the engine translog;
//! * the durable **state** (`state.json`, atomic + fsynced): highest
//!   accepted primary term, durable checkpoint floor, global checkpoint at
//!   the last trim, pending-resync marker, sticky "needs full recovery";
//! * the [`LocalCheckpointTracker`] (which seq_nos were processed);
//! * **delete tombstones** (`_id → seq_no` of the delete) for deletes above
//!   the local checkpoint. The engine forgets deletes at its next flush, so
//!   it cannot answer "was this id deleted at a newer seq_no?" itself.
//!
//! # Applying an op
//! 1. ops whose seq_no was already processed are dropped (duplicates,
//!    retries, the overlap between recovery phases);
//! 2. an `Index`/`Delete` whose seq_no is not newer than the seq_no the copy
//!    holds for that `_id` (live doc or tombstone) is **stale**: it is only
//!    marked processed (out-of-order delivery);
//! 3. otherwise it is applied with `IndexShard::replay_source` (parses
//!    against the copy's current mapping, persisting dynamic additions like
//!    the engine's own write path, then `Shard::replay_index`) /
//!    `Shard::replay_delete`, which store the
//!    primary's `_seq_no` / `_version` / `_routing`, so realtime GET and
//!    search on the replica see the same values as on the primary;
//! 4. `NoOp`s only mark their seq_no processed.
//!
//! Tombstones at or below the local checkpoint are dropped: any op for that
//! id older than the tombstone has a seq_no below the checkpoint and is
//! discarded as a duplicate in step 1.
//!
//! # Durability and checkpoints
//! The local checkpoint is never persisted separately: it is *recomputed*
//! on open from the durable floor plus the fsynced WAL records, so after a
//! crash a copy never claims an op it does not durably hold (an op whose
//! WAL append was fsynced but whose engine apply was lost is re-applied).
//! The global checkpoint travels in the same WAL record as the ops it was
//! received with (or a dedicated record), and in `state.json` at trim, so
//! it never goes backwards across restarts. The WAL is trimmed only by
//! [`ReplicaApplier::flush`], below `min(local checkpoint, global
//! checkpoint) + 1`: ops above the global checkpoint are always kept, they
//! are what a post-failover resync must compare.
//!
//! # Primary-term fencing
//! Every request carries the sender's primary term. The copy durably
//! records (fsync, before applying or acking anything) the highest term it
//! accepted and rejects lower ones with
//! [`ReplicationError::StalePrimaryTerm`], so a deposed primary that is
//! partitioned away can never collect acks and never acks a client write.
//!
//! # Failover: primary-replica resync (ES `PrimaryReplicaSyncer`)
//! The engine cannot roll back, so divergent history is *detected*, never
//! silently kept:
//! 1. When the copy first learns of a higher term (first request from the
//!    new primary, or [`ReplicaApplier::bump_term`]) it takes `G` = the
//!    highest global checkpoint it knows (its own or the one supplied) and
//!    collects every op it holds above `G` from its WAL ([`ops_above`]).
//!    These are *suspects*: the old primary may have replicated them here
//!    without the new primary ever getting them. If it holds seq_nos above
//!    `G` it cannot account for op by op (covered only by a snapshot
//!    fast-forward), it fails immediately with
//!    [`ReplicationError::NeedsFullRecovery`]. Otherwise the new term and
//!    a "resync pending at G" marker are persisted (one fsync) and while
//!    the resync is pending the copy reports `min(local checkpoint, G)` as
//!    its checkpoint, so the new primary's global checkpoint cannot move
//!    past unverified ops.
//! 2. The new primary (see [`ReplicaApplier::promote_to_primary`]) sends a
//!    [`ResyncRequest`] with all its ops in `(G, max_seq_no]`, gaps filled
//!    with `NoOp`s.
//! 3. [`ReplicaApplier::resync`] requires every suspect to appear in the
//!    resync with the same kind, seq_no, primary term, id and version
//!    ([`ReplicaOp::same_op`]); any mismatch or absence →
//!    `NeedsFullRecovery`. Ordinary requests arriving before the resync
//!    are checked against the suspects the same way. On success the resync
//!    ops are applied (missing ones fill in) and the marker is cleared.
//! 4. `NeedsFullRecovery` is sticky (persisted): every later call fails
//!    with it until the caller wipes the copy and peer-recovers it.
//!
//! This is conservative: a copy whose ops arrived through a different path
//! than the new primary's (e.g. a snapshot recovery stamping the source's
//! current term) may be re-recovered although its data was identical.
//! A *deposed primary* that rejoins has its own ops in its engine translog,
//! not in a replica WAL: the node crate must always fully re-recover it.
//!
//! [`ops_above`]: ReplicaApplier::ops_above

use super::checkpoint::{LocalCheckpointTracker, UNASSIGNED_SEQ_NO};
use super::ops::{ReplicaOp, ReplicationRequest, ReplicationResponse, ResyncRequest};
use super::wal::{ReplicaWal, WalRecord};
use super::{ReplicationError, Result};
use elasticsearch_engine::index::IndexService;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use elasticsearch_engine::translog::AdmittedBatch;
enum ApplyBatch<'a> { Borrowed(&'a [ReplicaOp]), Owned(AdmittedBatch<ReplicaOp>) }
impl std::ops::Deref for ApplyBatch<'_> {
    type Target = [ReplicaOp];
    fn deref(&self) -> &[ReplicaOp] { match self { Self::Borrowed(ops) => ops, Self::Owned(ops) => ops } }
}
use std::collections::BTreeMap;
use std::sync::Arc;

/// Counters of one replica copy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaStats {
    pub batches: u64,
    pub ops_applied: u64,
    /// Older than what the copy holds for the id (out-of-order delivery).
    pub ops_stale: u64,
    /// Already processed (duplicate / retried delivery).
    pub ops_duplicate: u64,
    pub noops: u64,
    /// Primary mappings merged from a batch (dynamic fields learned while
    /// replaying ops are persisted by the engine and not counted).
    pub mapping_updates: u64,
    /// Ops re-applied from the WAL by the last `recover_from_wal`.
    pub wal_ops_replayed: u64,
    pub tombstones: u64,
    /// Reported local checkpoint (capped while a resync is pending).
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    pub global_checkpoint: i64,
    pub max_seq_no_of_updates: i64,
    pub primary_term: u64,
    /// Ops above the resync global checkpoint awaiting verification.
    pub resync_suspects: u64,
    pub needs_full_recovery: Option<String>,
    pub wal_generations: u64,
    pub wal_size_bytes: u64,
}

struct Tombstone {
    seq_no: i64,
    _memory: elasticsearch_engine::rustic_core::admission::Reservation,
}
type PreparedTombstone = (String, Tombstone);

/// Admission and key allocation happen before WAL append / engine mutation.
/// Each entry covers a BTreeMap node split (eleven entries plus child edges);
/// retaining sixteen entry units per key also bounds sparse nodes after prune.
const TOMBSTONE_NODE_BYTES: usize=16*(std::mem::size_of::<(String,Tombstone)>()+std::mem::size_of::<usize>());
fn prepare_tombstone(op: &ReplicaOp) -> Result<Option<PreparedTombstone>> {
    prepare_tombstone_on(op,&elasticsearch_engine::rustic_core::admission::shared())
}
fn prepare_tombstone_on(op: &ReplicaOp,budget: &Arc<elasticsearch_engine::rustic_core::admission::Budget>) -> Result<Option<PreparedTombstone>> {
    let ReplicaOp::Delete{id,seq_no,..}=op else {return Ok(None)};
    let bytes=id.len().checked_add(TOMBSTONE_NODE_BYTES)
        .ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","replica tombstone size overflow"))?;
    let memory=budget.reserve(bytes as u64,"replica resident tombstone")?;
    let mut key=String::new();
    key.try_reserve_exact(id.len()).map_err(|e|elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
    key.push_str(id);
    Ok(Some((key,Tombstone{seq_no:*seq_no,_memory:memory})))
}

struct Inner {
    wal: ReplicaWal,
    tracker: LocalCheckpointTracker,
    tombstones: BTreeMap<String, Tombstone>,
    // BTreeMap may retain its empty root after removing the final entry.
    // This owner outlives the map; per-entry owners cover every other node.
    _tombstone_root: elasticsearch_engine::rustic_core::admission::Reservation,
    primary_term: u64,
    global_checkpoint: i64,
    max_seq_no_of_updates: i64,
    /// Pending resync: (global checkpoint G, ops held above G by seq_no).
    resync: Option<(i64, AdmittedBatch<ReplicaOp>)>,
    failed: Option<String>,
    stats: ReplicaStats,
    /// Concurrent batches past the term fence whose engine apply (with
    /// `inner` released) has not been marked processed yet. Promotion
    /// drains them before it reads the max seq_no.
    applying: usize,
}

impl Inner {
    fn reported_checkpoint(&self) -> i64 {
        match &self.resync {
            Some((g, _)) => self.tracker.checkpoint().min(*g),
            None => self.tracker.checkpoint(),
        }
    }
}

/// One replica shard copy.
pub struct ReplicaApplier {
    svc: Arc<IndexService>,
    /// Engine shard (= cluster shard number) this applier writes.
    shard: u32,
    /// The engine copy captured at open (see `shards::copy_of`).
    copy: Arc<elasticsearch_engine::index::IndexShard>,
    inner: Mutex<Inner>,
    /// Shared by replicated batches while they apply to the engine with
    /// `inner` released (they apply concurrently: the engine orders each id
    /// by seq_no); exclusive for state transitions that must not overlap an
    /// engine apply (resync, WAL recovery). Taken before `inner`.
    apply_gate: RwLock<()>,
    /// Signalled (with `inner`) whenever `Inner::applying` drops.
    drained: parking_lot::Condvar,
    /// Test seam: the next concurrent batch waits twice on this barrier
    /// between its WAL append and its engine apply (entered, released).
    #[cfg(test)]
    pub(crate) apply_pause: Mutex<Option<Arc<std::sync::Barrier>>>,
    /// Test seam: the next memory admission with this label is refused
    /// (once), as the request breaker would refuse it.
    #[cfg(test)]
    pub(crate) refuse_admission: Mutex<Option<&'static str>>,
    /// Test seam: the next promotion fails (once) after it adopted its
    /// term and applied its held ops, before its gap no-ops.
    #[cfg(test)]
    pub(crate) interrupt_promotion: std::sync::atomic::AtomicBool,
    // Cluster-state reconciliation must not wait behind a replica batch or
    // flush: it runs on the coordinator actor that also handles heartbeats.
    // Publish only after durable term/failure changes; never hold this lock
    // while parsing, indexing, or doing I/O.
    status: RwLock<(u64, Option<String>)>,
}

impl std::fmt::Debug for ReplicaApplier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReplicaApplier({})", self.svc.name)
    }
}

/// What `recover_from_wal` did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalRecoveryStats {
    pub records: u64,
    pub ops_replayed: u64,
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    pub global_checkpoint: i64,
}

/// A replica prefix that is wholly committed and was acknowledged by the
/// old in-sync set. This is not a general history-resync token: the source
/// must additionally validate the copy's old allocation id and incarnation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaPrefix {
    pub checkpoint: i64,
    pub global_checkpoint: i64,
    pub primary_term: u64,
}

/// What a promoted copy must send to the other copies
/// (see [`ReplicaApplier::promote_to_primary`]).
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionResync {
    pub primary_term: u64,
    pub global_checkpoint: i64,
    pub max_seq_no: i64,
    /// Every op in `(global_checkpoint, max_seq_no]`, seq_no order, gaps
    /// filled with `NoOp`s of the new term.
    #[serde(deserialize_with = "super::wal::decode_ops")]
    pub ops: AdmittedBatch<ReplicaOp>,
}

impl PromotionResync {
    pub fn to_request(&self, index: &str, shard: u32, target_allocation_id: &str) -> elasticsearch_engine::error::Result<ResyncRequest> {
        ResyncRequest::try_new(index, shard, target_allocation_id, self.primary_term,
            self.global_checkpoint, self.max_seq_no, ReplicaOp::try_copy_selected(&self.ops, |_| true)?)
    }
}

/// Ops above `above` in a WAL, one per seq_no (first occurrence), plus the
/// seq_nos above `above` covered only by fast-forwards (unaccounted). Fed
/// one record at a time, so only the selected ops are ever held.
struct OpsAbove {
    above: i64,
    ops: AdmittedBatch<ReplicaOp>,
    ff_max: i64,
}

impl OpsAbove {
    fn new(above: i64) -> Result<OpsAbove> {
        Ok(OpsAbove { above, ops: AdmittedBatch::new(elasticsearch_engine::memory::breakers())?, ff_max: i64::MIN })
    }

    fn add(&mut self, record: &WalRecord) -> Result<()> {
        match record {
            WalRecord::Ops { ops: batch, .. } => {
                let above = self.above;
                // Fed by a WAL read (a replay of acknowledged ops): the
                // copy's reservations are forced, never refused.
                let copied = ReplicaOp::try_copy_selected(batch, |op| op.seq_no() > above)?;
                self.ops.try_append(copied).map_err(|(e, _)| e)?;
            }
            WalRecord::FastForward(to) => self.ff_max = self.ff_max.max(*to),
            WalRecord::GlobalCheckpoint(_) => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<(AdmittedBatch<ReplicaOp>, Vec<i64>)> {
        let OpsAbove { above, mut ops, ff_max } = self;
        ops.try_stable_sort_by_key(elasticsearch_engine::memory::breakers(), ReplicaOp::seq_no)?;
        ops.dedup_by_key(|op| op.seq_no());
        // Diagnostics are bounded independently of history size.
        let unaccounted = if ff_max > above { (above + 1..=ff_max).filter(|seq| ops.binary_search_by_key(seq, ReplicaOp::seq_no).is_err()).take(16).collect() } else { Vec::new() };
        Ok((ops, unaccounted))
    }
}

/// [`OpsAbove`] over every record still in `wal`, streamed.
fn ops_above_wal(wal: &ReplicaWal, above: i64) -> Result<(AdmittedBatch<ReplicaOp>, Vec<i64>)> {
    let mut acc = OpsAbove::new(above)?;
    wal.for_each_record(|record| acc.add(&record))?;
    acc.finish()
}

#[cfg(test)]
fn ops_above_in(records: &[WalRecord], above: i64) -> Result<(AdmittedBatch<ReplicaOp>, Vec<i64>)> {
    let mut acc = OpsAbove::new(above)?;
    for record in records { acc.add(record)?; }
    acc.finish()
}

/// Sorted sequence/index scratch with explicit slot ownership. Indices preserve
/// first/last duplicate selection without an unbounded hash allocation.
fn sequence_positions(ops: &[ReplicaOp]) -> elasticsearch_engine::error::Result<AdmittedBatch<(i64, usize)>> {
    let bytes = ops.len().checked_mul(std::mem::size_of::<(i64, usize)>()).ok_or_else(|| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "sequence scratch overflow"))?;
    let memory = elasticsearch_engine::memory::breakers().reserve_request(bytes as u64, "replica sequence positions")?;
    let mut values = Vec::new();
    values.try_reserve_exact(ops.len()).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("sequence scratch: {e}")))?;
    values.extend(ops.iter().enumerate().map(|(i, op)| (op.seq_no(), i)));
    values.sort_unstable();
    Ok(AdmittedBatch::from_reserved(values, memory))
}

fn replay_container_admission(count: usize, concurrent: bool) -> elasticsearch_engine::error::Result<elasticsearch_engine::memory::Reservation> {
    // Only these concrete caller-owned arrays; parsed document heaps and engine
    // staging are independent producers, never estimated from source byte size.
    let per = std::mem::size_of::<(&str, &str)>() + if concurrent {
        std::mem::size_of::<elasticsearch_engine::index::ReplayOp<'static>>() + std::mem::size_of::<usize>() + 2 * std::mem::size_of::<bool>()
    } else { std::mem::size_of::<elasticsearch_engine::shard::ReplayIndex<'static>>() };
    let bytes = count.checked_mul(per).ok_or_else(|| elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","replica replay workspace overflow"))?;
    elasticsearch_engine::memory::breakers().reserve_request(bytes as u64,"replica replay caller containers")
}
/// Ops per WAL record when a deposed primary seeds its WAL from history.
const WAL_SEED_CHUNK: usize = 256;
fn replay_vec<T>(count: usize) -> elasticsearch_engine::error::Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|e| elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",format!("replica replay allocation: {e}")))?;
    Ok(out)
}
/// The engine term records of a batch, admitted and built before its WAL
/// append (see [`ReplicaApplier::stage_terms`]).
struct StagedTerms {
    terms: Vec<(i64, u64)>,
    _memory: elasticsearch_engine::memory::Reservation,
}
fn replica_sources(ops: &[ReplicaOp]) -> elasticsearch_engine::error::Result<Option<Vec<(&str,&str)>>> {
    let mut out = replay_vec(ops.len())?;
    for op in ops {
        match op { ReplicaOp::Index{id,source_json,..} => out.push((id.as_str(),source_json.as_str())), _ => return Ok(None) }
    }
    Ok(Some(out))
}

impl ReplicaApplier {
    /// Attach to the replica copy `svc` (freshly created or just reopened
    /// with `IndexService::open`) and replay its WAL. A copy marked
    /// "needs full recovery" opens fine but refuses every request.
    pub fn open(svc: Arc<IndexService>, shard: u32) -> Result<ReplicaApplier> {
        let copy = crate::shards::try_copy_of(&svc, shard)?;
        Self::open_copy(svc, copy)
    }

    /// Replay a pending physical copy without publishing an engine slot or
    /// cluster role. The caller must retain the PreparedShard until installation
    /// and revalidate allocation/term authority before making the applier live.
    pub fn open_prepared(svc: Arc<IndexService>, prepared: &elasticsearch_engine::index::PreparedShard) -> Result<ReplicaApplier> {
        let copy = svc.prepared_shard_copy(prepared)?;
        Self::open_copy(svc, copy)
    }

    fn open_copy(svc: Arc<IndexService>, copy: Arc<elasticsearch_engine::index::IndexShard>) -> Result<ReplicaApplier> {
        let shard = copy.id;
        let _permit = copy.operation_permit()?;
        let wal = ReplicaWal::open(copy.shard.path())?;
        let st = wal.state().clone();
        let applier = ReplicaApplier {
            svc,
            shard,
            copy: copy.clone(),
            status: RwLock::new((st.primary_term, st.needs_full_recovery.clone())),
            apply_gate: RwLock::new(()),
            drained: parking_lot::Condvar::new(),
            #[cfg(test)]
            apply_pause: Mutex::new(None),
            #[cfg(test)]
            refuse_admission: Mutex::new(None),
            #[cfg(test)]
            interrupt_promotion: std::sync::atomic::AtomicBool::new(false),
            inner: Mutex::new(Inner {
                wal,
                tracker: LocalCheckpointTracker::new(st.checkpoint_floor),
                tombstones: BTreeMap::new(),
                _tombstone_root: elasticsearch_engine::rustic_core::admission::shared().reserve(TOMBSTONE_NODE_BYTES as u64,"replica tombstone root")?,
                primary_term: st.primary_term,
                global_checkpoint: st.global_checkpoint,
                max_seq_no_of_updates: -1,
                resync: None,
                failed: st.needs_full_recovery.clone(),
                stats: ReplicaStats::default(),
                applying: 0,
            }),
        };
        applier.replay_open_wal(&mut applier.inner.lock())?;
        Ok(applier)
    }

    pub fn index(&self) -> &Arc<IndexService> {
        &self.svc
    }

    /// The engine shard copy this applier writes.
    pub fn copy(&self) -> Arc<elasticsearch_engine::index::IndexShard> {
        self.copy.clone()
    }

    /// Re-apply every WAL op above what is durably committed, on top of
    /// the engine's current state, and restore term / global checkpoint /
    /// pending-resync state. Called by [`open`](Self::open); idempotent.
    pub fn recover_from_wal(&self) -> Result<WalRecoveryStats> {
        let _permit = self.copy.operation_permit()?;
        let _exclusive = self.apply_gate.write();
        let mut g = self.inner.lock();
        g.wal = ReplicaWal::open(self.copy().shard.path())?;
        self.replay_open_wal(&mut g)
    }

    /// Replay the just opened WAL, streaming: one record is read at a time
    /// and ops already processed are skipped one by one. The ops were
    /// acknowledged: reading and applying them (and the refresh that ends
    /// the recovery) is an `elasticsearch_engine::memory::Replay`, never
    /// refused or delayed by a circuit breaker (ES translog recovery). Only
    /// real failures (I/O, corruption) fail it, and with it the copy, which
    /// then recovers from its primary. A single pass also collects the
    /// pending resync's suspects.
    fn replay_open_wal(&self, g: &mut Inner) -> Result<WalRecoveryStats> {
        let what = format!("replica WAL recovery of [{}][{}]", self.svc.name, self.shard);
        elasticsearch_engine::memory::replaying(&what, || self.replay_open_wal_locked(g))
    }

    fn replay_open_wal_locked(&self, g: &mut Inner) -> Result<WalRecoveryStats> {
        let st = g.wal.state().clone();
        g.primary_term = g.primary_term.max(st.primary_term);
        g.global_checkpoint = g.global_checkpoint.max(st.global_checkpoint);
        g.failed = g.failed.take().or(st.needs_full_recovery.clone());
        self.publish_status(g);
        g.tracker = LocalCheckpointTracker::new(st.checkpoint_floor);
        g.tombstones.clear();
        let mut stats = WalRecoveryStats::default();
        let mut suspects = st.resync_pending_gcp.map(OpsAbove::new).transpose()?;
        let plan = g.wal.replay_plan();
        let outcome = plan.run(|rec| {
            if let Some(acc) = &mut suspects { acc.add(&rec)?; }
            match rec {
                WalRecord::Ops { ops, global_checkpoint } => {
                    g.global_checkpoint = g.global_checkpoint.max(global_checkpoint);
                    for op in ops.iter() {
                        if g.tracker.has_processed(op.seq_no()) {
                            continue;
                        }
                        self.apply_one(g, op)?;
                        g.tracker.mark_processed(op.seq_no());
                        stats.ops_replayed += 1;
                    }
                }
                WalRecord::FastForward(to) => {
                    g.tracker.fast_forward(to);
                }
                WalRecord::GlobalCheckpoint(gcp) => g.global_checkpoint = g.global_checkpoint.max(gcp),
            }
            Ok(())
        })?;
        g.wal.replayed(&outcome);
        stats.records = outcome.records;
        g.resync = match (st.resync_pending_gcp, suspects) {
            (Some(gcp), Some(acc)) => Some((gcp, acc.finish()?.0)),
            _ => None,
        };
        Self::prune_tombstones(g);
        g.stats.wal_ops_replayed = stats.ops_replayed;
        stats.local_checkpoint = g.reported_checkpoint();
        stats.max_seq_no = g.tracker.max_seq_no();
        stats.global_checkpoint = g.global_checkpoint;
        self.publish_global_checkpoint(g.global_checkpoint);
        if stats.ops_replayed > 0 {
            tracing::info!(index = %self.svc.name, ops = stats.ops_replayed, checkpoint = stats.local_checkpoint, "replayed replica WAL");
            // Recovered ops are searchable once recovery ends (ES refreshes).
            self.copy().shard.refresh()?;
        }
        Ok(stats)
    }

    /// Apply a batch from the primary (in-process form of [`handle`]).
    /// The batch's highest op term is taken as the sender's term for
    /// fencing. `mapping` is the primary's mapping (`GET _mapping` body
    /// form); when it differs from this copy's it is merged in first.
    /// Returns the (reported) local checkpoint.
    ///
    /// [`handle`]: Self::handle
    pub fn apply(&self, batch: &[ReplicaOp], mapping: Option<&Value>) -> Result<i64> {
        self.apply_versioned(batch, mapping, 0)
    }

    /// [`apply`](Self::apply) with the cluster `mapping_version` of
    /// `mapping` (0 = unknown): a copy that already holds that version or
    /// a newer one keeps its mapping.
    pub fn apply_versioned(&self, batch: &[ReplicaOp], mapping: Option<&Value>, mapping_version: u64) -> Result<i64> {
        self.apply_under(batch, batch.iter().map(ReplicaOp::primary_term).max(), mapping, mapping_version)
    }

    /// [`apply_versioned`](Self::apply_versioned) fenced on `term`, the
    /// sender's term, not its ops' own: a recovery under term 2 ships the
    /// documents and ops term 1 wrote with term 1 (each copy stores an op's
    /// own `_primary_term`), and must not be fenced as a term-1 sender.
    pub fn apply_under(&self, batch: &[ReplicaOp], term: Option<u64>, mapping: Option<&Value>, mapping_version: u64) -> Result<i64> {
        // One bounded wait for seals per batch, before any lock.
        self.copy.shard.admit_replicated();
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        if let Some(term) = term {
            self.fence(&mut g, term, UNASSIGNED_SEQ_NO)?;
        } else if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        self.apply_locked(&mut g, ApplyBatch::Borrowed(batch), mapping.map(|m| (m, mapping_version)), None, true)
    }

    /// Transport entry point (`indices:data/write/replica`): term fencing
    /// (durable before anything else), mapping, ops, global checkpoint.
    pub fn handle(&self, req: &ReplicationRequest) -> Result<ReplicationResponse> {
        self.handle_ops(req, ApplyBatch::Borrowed(&req.ops[..]))
    }

    /// [`handle`](Self::handle) of a request it owns: the ops move into the
    /// WAL record and the engine apply instead of being copied.
    pub fn handle_owned(&self, mut req: ReplicationRequest) -> Result<ReplicationResponse> {
        let ops = std::mem::replace(&mut req.ops, AdmittedBatch::new(elasticsearch_engine::memory::breakers())?);
        self.handle_ops(&req, ApplyBatch::Owned(ops))
    }

    fn handle_ops(&self, req: &ReplicationRequest, ops: ApplyBatch<'_>) -> Result<ReplicationResponse> {
        let started = std::time::Instant::now();
        let rows = ops.len();
        // One bounded wait for seals per transport batch, before the permit,
        // `inner` or the applying count (a promotion or flush never waits
        // behind it). Never a rejection for the buffer (it would fail the
        // copy); the node's replica indexing pressure rejects instead.
        self.copy.shard.admit_replicated();
        let _permit = self.copy.operation_permit()?;
        let mapping: Option<Value> = req.mapping_json.as_deref().map(serde_json::from_str).transpose()?;
        let mut g = self.inner.lock();
        let waited = started.elapsed();
        self.fence(&mut g, req.primary_term, req.global_checkpoint)?;
        g.max_seq_no_of_updates = g.max_seq_no_of_updates.max(req.max_seq_no_of_updates);
        // The engine applies ops above the primary's MSU as first copies (ES).
        self.copy.shard.advance_max_seq_no_of_updates_from_primary(g.max_seq_no_of_updates);
        let before = g.stats.ops_applied + g.stats.ops_stale + g.stats.noops;
        self.apply_locked(&mut g, ops, mapping.as_ref().map(|m| (m, req.mapping_version)), Some(req.global_checkpoint), true)?;
        let after = g.stats.ops_applied + g.stats.ops_stale + g.stats.noops;
        if started.elapsed() >= std::time::Duration::from_secs(5) {
            tracing::warn!(index=%req.index, shard=req.shard, rows,
                wait_ms=waited.as_millis() as u64, total_ms=started.elapsed().as_millis() as u64,
                "slow replica request (includes operation permit and apply-lock wait)");
        }
        Ok(ReplicationResponse { local_checkpoint: g.reported_checkpoint(), max_seq_no: g.tracker.max_seq_no(), applied: after - before })
    }

    /// Learn of a new primary term out of band (cluster state applied:
    /// this copy's primary changed). See the module docs, "Failover".
    pub fn bump_term(&self, new_term: u64, global_checkpoint: i64) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        self.fence(&mut g, new_term, global_checkpoint)
    }

    /// Primary-replica resync from a newly promoted primary.
    pub fn resync(&self, req: &ResyncRequest) -> Result<ReplicationResponse> {
        let _permit = self.copy.operation_permit()?;
        // No replicated batch is applying to the engine meanwhile.
        let _exclusive = self.apply_gate.write();
        let mut g = self.inner.lock();
        self.fence(&mut g, req.primary_term, req.global_checkpoint)?;
        let positions = if g.resync.is_some() { Some(sequence_positions(&req.ops)?) } else { None };
        let mismatch = g.resync.as_ref().and_then(|(gcp, suspects)| {
            suspects.iter().find_map(|ours| {
                let seq = ours.seq_no();
                // Preserve prior duplicate precedence (the last incoming op).
                let positions = positions.as_ref().expect("pending resync positions");
                let end = positions.partition_point(|(s, _)| *s <= seq);
                let theirs = end.checked_sub(1).and_then(|at| (positions[at].0 == seq).then(|| &req.ops[positions[at].1]));
                match theirs {
                    Some(theirs) if theirs.same_op(ours) => None,
                    Some(theirs) => Some(format!("op [{seq}] differs from the new primary's: ours {ours:?}, theirs {theirs:?}")),
                    None => Some(format!("op [{seq}] above global checkpoint [{gcp}] is not on the new primary (max_seq_no [{}])", req.max_seq_no)),
                }
            })
        });
        if let Some(reason) = mismatch { return Err(self.fail(&mut g, reason)); }
        let before = g.stats.ops_applied + g.stats.ops_stale + g.stats.noops;
        self.apply_locked(&mut g, ApplyBatch::Borrowed(&req.ops[..]), None, Some(req.global_checkpoint), false)?;
        if g.resync.take().is_some() {
            g.wal.update_state(|s| s.resync_pending_gcp = None)?;
        }
        let after = g.stats.ops_applied + g.stats.ops_stale + g.stats.noops;
        Ok(ReplicationResponse { local_checkpoint: g.reported_checkpoint(), max_seq_no: g.tracker.max_seq_no(), applied: after - before })
    }

    /// Every op this copy holds above `seq_no` (from its WAL), one per
    /// seq_no, in seq_no order.
    pub fn ops_above(&self, seq_no: i64) -> Result<AdmittedBatch<ReplicaOp>> {
        let _permit = self.copy.operation_permit()?;
        let g = self.inner.lock();
        Ok(ops_above_wal(&g.wal, seq_no)?.0)
    }

    /// Term fencing. Lower term → reject. Higher term → durably adopt it
    /// and set up the resync check (module docs) before anything is acked.
    fn fence(&self, g: &mut Inner, term: u64, global_checkpoint: i64) -> Result<()> {
        if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        if term < g.primary_term {
            return Err(ReplicationError::StalePrimaryTerm { ours: g.primary_term, theirs: term });
        }
        if term == g.primary_term {
            return Ok(());
        }
        let first_term = g.primary_term == 0;
        let gcp = g.global_checkpoint.max(global_checkpoint);
        let (mut suspects, unaccounted) = ops_above_wal(&g.wal, gcp)?;
        if !first_term && g.wal.state().checkpoint_floor > gcp.max(super::checkpoint::NO_OPS_PERFORMED) {
            return Err(self.fail(g, format!("ops up to [{}] were trimmed above global checkpoint [{gcp}]", g.wal.state().checkpoint_floor)));
        }
        if !first_term && !unaccounted.is_empty() {
            return Err(self.fail(g, format!("seq_nos {unaccounted:?} above global checkpoint [{gcp}] are only covered by a snapshot fast-forward and cannot be verified")));
        }
        let pending = (!first_term && !suspects.is_empty()).then_some(gcp);
        if pending.is_some() {
            if let Some((_, old)) = &g.resync {
                let previous = ReplicaOp::try_clone_batch(old)?;
                // New WAL values take precedence, matching BTreeMap::extend.
                suspects.try_append(previous).map_err(|(e, _)| e)?;
                suspects.try_stable_sort_by_key(elasticsearch_engine::memory::breakers(), ReplicaOp::seq_no)?;
                suspects.dedup_by_key(|op| op.seq_no());
            }
        }
        g.wal.update_state(|s| {
            s.primary_term = term;
            s.global_checkpoint = s.global_checkpoint.max(gcp);
            if pending.is_some() {
                s.resync_pending_gcp = pending;
            }
        })?;
        g.primary_term = term;
        self.publish_status(g);
        g.global_checkpoint = gcp;
        self.publish_global_checkpoint(gcp);
        // The engine's term history follows (ops of the new term stamp it
        // without a per-op record; `note_terms` covers the rest).
        if term > self.copy.shard.primary_term() {
            if let Err(e) = self.copy.shard.set_primary_term(term) {
                tracing::warn!(index = %self.svc.name, term, "could not record the new primary term in the engine: {}", e.reason);
            }
        }
        if let Some(p) = pending {
            let oldest = g.resync.as_ref().map_or(p, |(old, _)| (*old).min(p));
            g.resync = Some((oldest, suspects));
            tracing::warn!(index = %self.svc.name, term, gcp, "new primary term with ops above the global checkpoint: resync required");
        }
        Ok(())
    }

    /// The global checkpoint this copy knows of, on its engine copy (ES
    /// `getLastKnownGlobalCheckpoint`: replicas advance it from replication
    /// requests and checkpoint syncs): readers of the copy (a fleet search's
    /// checkpoint wait, `_fleet/global_checkpoints`, stats) see it. Only
    /// ever raised.
    fn publish_global_checkpoint(&self, gcp: i64) {
        self.copy.update_global_checkpoint(gcp);
    }

    fn publish_status(&self, g: &Inner) {
        *self.status.write() = (g.primary_term, g.failed.clone());
    }

    fn fail(&self, g: &mut Inner, reason: String) -> ReplicationError {
        tracing::error!(index = %self.svc.name, %reason, "replica diverged; needs full recovery");
        if let Err(e) = g.wal.update_state(|s| s.needs_full_recovery = Some(reason.clone())) {
            tracing::error!("could not persist needs_full_recovery: {e}");
        }
        g.failed = Some(reason.clone());
        self.publish_status(g);
        ReplicationError::NeedsFullRecovery { reason }
    }

    /// `concurrent`: apply to the engine with `inner` released (under the
    /// shared `apply_gate`), so replicated batches apply in parallel; the
    /// engine decides each op's staleness by seq_no per id, and the WAL
    /// record above is already durable. Otherwise apply under `inner`, in
    /// batch order (resync).
    fn apply_locked(&self, g: &mut parking_lot::MutexGuard<'_, Inner>, batch: ApplyBatch<'_>, mapping: Option<(&Value, u64)>, global_checkpoint: Option<i64>, concurrent: bool) -> Result<i64> {
        let profile = tracing::enabled!(target: "replica_profile", tracing::Level::DEBUG).then(std::time::Instant::now);
        if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        // While a resync is pending, an op at a suspect seq_no must be the
        // very same op. A matching one verifies that suspect: the primary's
        // op at a seq_no is unique, so a recovery's translog phase (or live
        // replication) that delivers every suspect's seq_no resolves the
        // resync as the resync request itself would.
        let mut verified = 0;
        if let Some((_, suspects)) = &g.resync {
            for op in batch.iter() {
                if let Ok(at) = suspects.binary_search_by_key(&op.seq_no(), ReplicaOp::seq_no) {
                    let ours = &suspects[at];
                    if !ours.same_op(op) {
                        let reason = format!("op [{}] differs from the new primary's: ours {ours:?}, theirs {op:?}", op.seq_no());
                        return Err(self.fail(g, reason));
                    }
                    verified += 1;
                }
            }
        }
        if verified > 0 {
            let verified_positions = sequence_positions(&batch)?;
            let emptied = match &mut g.resync {
                Some((_, suspects)) => {
                    suspects.retain(|suspect| verified_positions.binary_search_by_key(&suspect.seq_no(), |entry| entry.0).is_err());
                    suspects.is_empty()
                }
                None => false,
            };
            if emptied {
                g.resync = None;
                g.wal.update_state(|s| s.resync_pending_gcp = None)?;
            }
        }
        if let Some((m, version)) = mapping {
            self.sync_mapping(g, m, version)?;
        }
        let new_gcp = global_checkpoint.filter(|x| *x > g.global_checkpoint);
        let total = batch.len();
        // Exact scratch slots, sorted without allocating. Keep the first
        // occurrence of each seq_no, preserving the previous dedup semantics.
        let bytes = total.checked_mul(std::mem::size_of::<(i64, usize)>()).ok_or_else(|| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", "replica dedup overflow"))?;
        let _dedup_memory = elasticsearch_engine::memory::breakers().reserve_request(bytes as u64, "replica dedup slots")?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(total).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("replica dedup allocation: {e}")))?;
        seen.extend(batch.iter().enumerate().map(|(i, op)| (op.seq_no(), i)));
        seen.sort_unstable(); seen.dedup_by_key(|entry| entry.0);
        let fresh = {
            let tracker = &g.tracker;
            let mut position = 0usize;
            let mut keep = |op: &ReplicaOp| {
                let i = position; position += 1;
                !tracker.has_processed(op.seq_no()) && seen.binary_search_by_key(&op.seq_no(), |entry| entry.0).is_ok_and(|found| seen[found].1 == i)
            };
            match batch {
                ApplyBatch::Owned(mut ops) => { ops.retain(&mut keep); ops },
                ApplyBatch::Borrowed(ops) => ReplicaOp::try_copy_selected(ops, keep)?,
            }
        };
        g.stats.ops_duplicate += (total - fresh.len()) as u64;
        if fresh.is_empty() {
            if let Some(gcp) = new_gcp {
                g.wal.append(&WalRecord::GlobalCheckpoint(gcp))?;
                g.global_checkpoint = gcp;
                self.publish_global_checkpoint(gcp);
            }
            return Ok(g.reported_checkpoint());
        }
        let _tombstone_slots=elasticsearch_engine::memory::breakers().reserve_request(
            fresh.len().checked_mul(std::mem::size_of::<Option<PreparedTombstone>>())
                .ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","replica tombstone slots overflow"))? as u64,
            "replica tombstone preparation slots")?;
        let mut tombstones=replay_vec(fresh.len())?;
        for op in fresh.iter() {tombstones.push(prepare_tombstone(op)?);}
        // Everything the batch must admit between its WAL append and marking
        // its ops processed is admitted here, before the append: a refusal
        // after it leaves durable ops unmarked, so the primary's retry passes
        // the dedup above, appends them again and can be refused again (the
        // checkpoint stalls and the WAL grows until the copy is failed).
        let staged_terms = self.stage_terms(&fresh)?;
        // The replay containers of one chunk (at most 256 ops), held for
        // the whole apply and reused by every chunk.
        let _containers = replay_container_admission(fresh.len().min(256), concurrent)?;
        // Counted until marked processed (or failed), so a promotion
        // waits for this batch instead of computing its max seq_no and
        // trimming the WAL underneath it.
        let applied_slots = if concurrent {
            Some((self.admit(fresh.len() as u64, "replica apply result slots")?, replay_vec::<bool>(fresh.len())?))
        } else { None };
        g.stats.batches += 1;
        let gcp = new_gcp.unwrap_or(g.global_checkpoint);
        let setup_us = profile.map(|t|t.elapsed().as_micros() as u64).unwrap_or(0);
        // The WAL serializes synchronously: lend it the owned batch, then
        // recover that same vector for replay instead of cloning every ID
        // and source string a second time.
        let record = WalRecord::Ops { ops: fresh, global_checkpoint: gcp };
        // Concurrent batches share the WAL fsync (group commit) outside
        // `inner`; the ordered path keeps the synchronous append.
        let pending_sync = if concurrent { Some(g.wal.append_deferred(&record)?) } else { g.wal.append(&record)?; None };
        let WalRecord::Ops { ops: fresh, .. } = record else { unreachable!() };
        // The batch is durable in this copy's WAL from here: everything until
        // its ops are marked processed is forced (charged, never refused). A
        // refusal here would leave durable ops unmarked, so the primary's
        // retry would pass the dedup above and append them again. The
        // batch's memory was admitted before the append (indexing pressure,
        // the staged admissions above), where a refusal fails the copy
        // cleanly. It is a live replay, not a recovery: it forces none of
        // the copy's background seals and stays under the copy's write
        // throttle (a batch a peer recovery applies joins the recovery).
        elasticsearch_engine::memory::replaying_live("replicated batch after its WAL append", || -> Result<i64> {
            g.global_checkpoint = gcp;
            self.publish_global_checkpoint(gcp);
            self.note_staged_terms(staged_terms, &fresh)?;
            let wal_done_us = profile.map(|t|t.elapsed().as_micros() as u64).unwrap_or(0);
            let (mut prepare_us, mut apply_us, mut prepared_rows, mut parallel_rows) = (0u64,0u64,0usize,0usize);
            if let Some((_applied_memory, mut applied)) = applied_slots {
                let copy = self.copy();
                g.applying += 1;
                let applied = parking_lot::MutexGuard::unlocked(g, || -> Result<Vec<bool>> {
                    let _shared = self.apply_gate.read();
                    // Durable before anything is applied or reported processed.
                    if let Some(sync) = &pending_sync { sync.wait()?; }
                    #[cfg(test)]
                    if let Some(pause) = self.apply_pause.lock().take() {
                        pause.wait();
                        pause.wait();
                    }
                    // Bound prepared-document memory per chunk.
                    for chunk in fresh.chunks(256) {
                        let sources = replica_sources(chunk)?;
                        let mut prepared = sources.as_ref().and_then(|s| copy.prepare_replica_sources(s)).map(|p| p.into_iter());
                        let mut ops = replay_vec(chunk.len())?;
                        let mut positions = replay_vec(chunk.len())?;
                        for (position, op) in chunk.iter().enumerate() {
                            match op {
                                ReplicaOp::Index { id, source_json, routing, seq_no, version, .. } => {
                                    ops.push(elasticsearch_engine::index::ReplayOp::Index { id, source: source_json, routing: routing.as_deref(),
                                        seq_no: *seq_no, version: *version, prepared: prepared.as_mut().and_then(Iterator::next) });
                                    positions.push(position);
                                }
                                ReplicaOp::Delete { id, seq_no, version, .. } => {
                                    ops.push(elasticsearch_engine::index::ReplayOp::Delete { id, seq_no: *seq_no, version: *version });
                                    positions.push(position);
                                }
                                ReplicaOp::NoOp { .. } => {}
                            }
                        }
                        let outcomes = copy.replay_ops(ops)?;
                        let mut chunk_applied = replay_vec(chunk.len())?;
                        chunk_applied.resize(chunk.len(), false);
                        for (position, outcome) in positions.into_iter().zip(outcomes.iter().copied()) { chunk_applied[position] = outcome; }
                        applied.extend(chunk_applied);
                    }
                    Ok(applied)
                });
                if let Ok(applied) = &applied {
                    for ((op, applied), tombstone) in fresh.iter().zip(applied.iter().copied()).zip(tombstones.iter_mut()) {
                        match op {
                            ReplicaOp::Index { id, .. } if applied => { g.tombstones.remove(id); g.stats.ops_applied += 1; }
                            ReplicaOp::Delete { .. } if applied => { let (id,entry)=tombstone.take().expect("delete admission prepared before WAL"); g.tombstones.insert(id,entry); g.stats.ops_applied += 1; }
                            ReplicaOp::NoOp { .. } => g.stats.noops += 1,
                            _ => g.stats.ops_stale += 1,
                        }
                        g.tracker.mark_processed(op.seq_no());
                    }
                }
                g.applying -= 1;
                self.drained.notify_all();
                applied?;
                Self::prune_tombstones(g);
                return Ok(g.reported_checkpoint());
            }
            // Bound prepared-document memory and keep each chunk's application
            // order. Preparation cannot publish dynamic mappings or fail a stale
            // op: an unsupported/invalid chunk uses the original replay path.
            for (chunk,tombstones) in fresh.chunks(256).zip(tombstones.chunks_mut(256)) {
                let chunk_start = profile.map(|_|std::time::Instant::now());
                let sources = replica_sources(chunk)?;
                let mut prepared = sources.as_ref().and_then(|s| self.copy().prepare_replica_sources(s));
                if let Some(t) = chunk_start { prepare_us += t.elapsed().as_micros() as u64; }
                let apply_start = profile.map(|_|std::time::Instant::now());
                if let Some(docs) = &mut prepared {
                    prepared_rows += docs.len();
                    let mut metadata = replay_vec(chunk.len())?;
                    let mut eligible = true;
                    for op in chunk {
                        match op {
                            ReplicaOp::Index { id, source_json, routing, seq_no, version, .. } if !g.tombstones.contains_key(id) =>
                                metadata.push(elasticsearch_engine::shard::ReplayIndex { id, source: source_json, routing: routing.as_deref(), seq_no: *seq_no, version: *version }),
                            _ => { eligible = false; break; }
                        }
                    }
                    let metadata = eligible.then_some(metadata);
                    if let Some(metadata) = metadata {
                        if self.copy().try_replay_new_batch(&metadata, docs)? {
                            for op in chunk { g.tracker.mark_processed(op.seq_no()); }
                            g.stats.ops_applied += chunk.len() as u64;
                            parallel_rows += chunk.len();
                            if let Some(t) = apply_start { apply_us += t.elapsed().as_micros() as u64; }
                            continue;
                        }
                    }
                }
                let mut prepared = prepared.map(|p| p.into_iter());
                for (op,tombstone) in chunk.iter().zip(tombstones.iter_mut()) {
                    self.apply_one_prepared(g, op, prepared.as_mut().and_then(Iterator::next), tombstone.take())?;
                    g.tracker.mark_processed(op.seq_no());
                }
                if let Some(t) = apply_start { apply_us += t.elapsed().as_micros() as u64; }
            }
            if let Some(t) = profile {
                tracing::debug!(target: "replica_profile", index=%self.svc.name, rows=fresh.len(), prepared_rows, parallel_rows,
                    setup_us, wal_us=wal_done_us.saturating_sub(setup_us), prepare_us, apply_us,
                    total_us=t.elapsed().as_micros() as u64, "replica write stages");
            }
            Self::prune_tombstones(g);
            Ok(g.reported_checkpoint())
        })
    }

    /// Test hook: durably append a batch to the WAL *without* applying it
    /// (simulates a crash between the WAL fsync and the engine apply).
    #[cfg(test)]
    pub(crate) fn wal_append_only(&self, ops: &[ReplicaOp]) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        let gcp = g.global_checkpoint;
        g.wal.append(&WalRecord::Ops { ops: ReplicaOp::try_copy_selected(ops, |_| true)?, global_checkpoint: gcp })
    }

    /// Declare every seq_no `<= to` processed (durably, via the WAL). Used
    /// after a snapshot recovery; see [`super::recovery`].
    pub fn fast_forward_checkpoint(&self, to: i64) -> Result<i64> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        if to > g.tracker.checkpoint() {
            g.wal.append(&WalRecord::FastForward(to))?;
            g.tracker.fast_forward(to);
            Self::prune_tombstones(&mut g);
        }
        Ok(g.reported_checkpoint())
    }

    /// The local checkpoint this copy reports (capped at the resync global
    /// checkpoint while a resync is pending).
    pub fn local_checkpoint(&self) -> i64 {
        self.inner.lock().reported_checkpoint()
    }

    pub fn max_seq_no(&self) -> i64 {
        self.inner.lock().tracker.max_seq_no()
    }

    pub fn primary_term(&self) -> u64 {
        self.status.read().0
    }

    pub fn global_checkpoint(&self) -> i64 {
        self.inner.lock().global_checkpoint
    }

    pub fn needs_full_recovery(&self) -> Option<String> {
        self.status.read().1.clone()
    }

    /// The term of a promotion this copy adopted but has not finished
    /// (see [`promote_to_primary`](Self::promote_to_primary)).
    pub fn promotion_pending(&self) -> Option<u64> {
        self.inner.lock().wal.state().promotion_pending
    }

    pub fn resync_pending(&self) -> bool {
        self.inner.lock().resync.is_some()
    }

    /// Only the already committed, globally acknowledged prefix may be
    /// reused without comparing divergent operation histories. Keep this
    /// atomic with replica apply/term changes, and never derive the proof
    /// from the new primary's (possibly just initialized) checkpoint.
    /// Record in the WAL an engine commit made outside [`flush`](Self::flush).
    /// An explicit `_flush` or force merge commits the replica's engine
    /// directly: the commit then holds every op up to the checkpoint while
    /// the WAL floor stays below it, so [`reusable_prefix`](Self::reusable_prefix)
    /// refused the copy and a full restart wiped and re-replayed it (losing
    /// the merged segments). When the commit covers the whole processed and
    /// globally acknowledged history, trim the WAL below it, exactly as a
    /// flush through this applier would have. Otherwise (uncommitted or
    /// unacknowledged suffix, failure, pending resync) nothing changes.
    pub fn adopt_engine_commit(&self) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        let cp = g.tracker.checkpoint();
        if g.failed.is_some() || g.resync.is_some() || g.primary_term == 0
            || cp < 0 || cp != g.tracker.max_seq_no() || cp > g.global_checkpoint
            || cp <= g.wal.state().checkpoint_floor
            || cp != self.copy.shard.committed_seq_no() || cp != self.copy.shard.max_seq_no()
        {
            return Ok(());
        }
        let gcp = g.global_checkpoint;
        g.wal.trim_below(cp + 1, gcp)
    }

    /// The prefix this copy may catch up from by operations (ES
    /// `recoverLocallyUpToGlobalCheckpoint` + `startingSeqNo`): every op
    /// up to `min(local checkpoint, global checkpoint)` is common history
    /// (any later primary holds every op at or below a global checkpoint),
    /// so the recovery's translog phase starts right above it. Ops this copy
    /// holds above that point need no commit and no acknowledgement: they
    /// were either sent by the current primary in its own term (re-sent,
    /// dropped as duplicates), or are verified one by one against the
    /// primary's ops at their seq_nos when the recovery adopts a newer term
    /// (the resync suspects of [`bump_term`](Self::bump_term)); a mismatch
    /// fails the copy, which then recovers from files. The engine must not
    /// hold an op the checkpoint tracker does not account for.
    pub fn reusable_prefix(&self) -> Option<ReplicaPrefix> {
        let _permit = self.copy.operation_permit().ok()?;
        let g = self.inner.lock();
        let cp = g.reported_checkpoint();
        let start = cp.min(g.global_checkpoint);
        if g.failed.is_some() || g.primary_term == 0 || start < -1
            || self.copy.shard.max_seq_no() > g.tracker.max_seq_no()
            || g.wal.state().checkpoint_floor > g.global_checkpoint.max(super::checkpoint::NO_OPS_PERFORMED)
        {
            tracing::info!(index=%self.svc.name, shard=self.shard,
                checkpoint=cp, max_seq_no=g.tracker.max_seq_no(),
                durable_floor=g.wal.state().checkpoint_floor, global_checkpoint=g.global_checkpoint,
                engine_max_seq_no=self.copy.shard.max_seq_no(),
                primary_term=g.primary_term, pending_resync=g.resync.is_some(), failed=g.failed.is_some(),
                "replica prefix not reusable");
            return None;
        }
        Some(ReplicaPrefix { checkpoint: start, global_checkpoint: g.global_checkpoint, primary_term: g.primary_term })
    }

    /// Ops of a pending resync not verified yet (see `apply_locked`). Once a
    /// recovery's translog phase caught up with the primary, any left are
    /// not in its history.
    pub fn unverified_suspects(&self) -> Vec<i64> {
        self.inner.lock().resync.as_ref().map(|(_, s)| s.iter().map(ReplicaOp::seq_no).take(16).collect()).unwrap_or_default()
    }

    /// Fail the copy (it needs a file recovery) when ops above the global
    /// checkpoint it holds were not delivered by the primary's history:
    /// called once the recovery's translog phase caught up.
    pub fn fail_unverified_suspects(&self) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        let left: Vec<i64> = g.resync.as_ref().map(|(_, s)| s.iter().map(ReplicaOp::seq_no).take(16).collect()).unwrap_or_default();
        if left.is_empty() {
            return Ok(());
        }
        Err(self.fail(&mut g, format!("ops {left:?} above the global checkpoint are not in the primary's history")))
    }

    /// Open a copy that last ran as **primary** (deposed, restarted) as a
    /// replica that may catch up from operations: its own engine history
    /// above `global_checkpoint` (the global checkpoint it recorded, a lower
    /// bound of every in-sync copy's) becomes the WAL content above a floor
    /// at that checkpoint, under the term it held. The recovery then adopts
    /// the new primary's term and verifies those ops one by one (resync
    /// suspects); anything not verified fails the copy, which recovers from
    /// files. Any older WAL (from before it was promoted) is discarded.
    pub fn open_demoted(svc: Arc<IndexService>, shard: u32, global_checkpoint: i64, term: u64) -> Result<ReplicaApplier> {
        let copy = crate::shards::try_copy_of(&svc, shard)?;
        {
            let _permit = copy.operation_permit()?;
            if copy.shard.local_checkpoint() < global_checkpoint {
                return Err(ReplicationError::IllegalState(format!("local checkpoint [{}] is below the recorded global checkpoint [{global_checkpoint}]", copy.shard.local_checkpoint())));
            }
            let mut changes = copy.shard.changes_since(global_checkpoint + 1)?.peekable();
            super::wal::remove(copy.shard.path())?;
            let mut wal = ReplicaWal::open(copy.shard.path())?;
            wal.replay(|_| Ok(()))?;
            wal.update_state(|s| {
                s.primary_term = term;
                s.checkpoint_floor = global_checkpoint;
                s.global_checkpoint = global_checkpoint;
                s.resync_pending_gcp = None;
                s.needs_full_recovery = None;
            })?;
            // Bounded records: the replay on open streams one at a time.
            while changes.peek().is_some() {
                let mut ops = AdmittedBatch::new(elasticsearch_engine::memory::breakers())?;
                for item in changes.by_ref().take(WAL_SEED_CHUNK) {
                    ops.reserve_one()?;
                    let (value, memory) = ReplicaOp::try_copy_change(&item)?;
                    ops.push_reserved(value, memory).map_err(|(e, _, _)| e)?;
                }
                wal.append(&WalRecord::Ops { ops, global_checkpoint })?;
            }
        }
        Self::open_copy(svc, copy)
    }

    /// Advance the global checkpoint (durably; never goes backwards).
    pub fn update_global_checkpoint(&self, gcp: i64) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        if gcp > g.global_checkpoint {
            g.wal.append(&WalRecord::GlobalCheckpoint(gcp))?;
            g.global_checkpoint = gcp;
            self.publish_global_checkpoint(gcp);
        }
        Ok(())
    }

    pub fn stats(&self) -> ReplicaStats {
        let g = self.inner.lock();
        let mut s = g.stats.clone();
        s.tombstones = g.tombstones.len() as u64;
        s.local_checkpoint = g.reported_checkpoint();
        s.max_seq_no = g.tracker.max_seq_no();
        s.global_checkpoint = g.global_checkpoint;
        s.max_seq_no_of_updates = g.max_seq_no_of_updates;
        s.primary_term = g.primary_term;
        s.resync_suspects = g.resync.as_ref().map_or(0, |(_, s)| s.len() as u64);
        s.needs_full_recovery = g.failed.clone();
        s.wal_generations = g.wal.num_generations() as u64;
        s.wal_size_bytes = g.wal.size_bytes();
        s
    }

    /// A size-triggered flush is useful only if it can advance the safe WAL
    /// floor. During snapshot recovery the global checkpoint may stay unset
    /// until finalization. Repeatedly checkpointing the growing vector graph
    /// in that interval cannot trim a byte and can starve recovery itself.
    /// Explicit, idle and recovery-finalization flushes remain independent.
    pub fn needs_size_flush(&self, threshold: u64) -> bool {
        self.needs_periodic_flush(threshold, false)
    }

    /// Age-triggered flushes obey the same reclaimability fence as size
    /// flushes. Recovery retains its WAL until the global checkpoint moves;
    /// repeatedly sealing that tail cannot retire history and delays replay.
    pub fn needs_periodic_flush(&self, threshold: u64, age_due: bool) -> bool {
        let g = self.inner.lock();
        g.tracker.checkpoint().min(g.global_checkpoint) > g.wal.state().checkpoint_floor
            && (age_due || g.wal.size_bytes() >= threshold)
    }

    /// Flush the engine, then trim the WAL below `min(checkpoint captured
    /// before the flush, global checkpoint) + 1`. Applies are blocked
    /// meanwhile. Returns the durable checkpoint.
    pub fn flush(&self) -> Result<i64> {
        let started = std::time::Instant::now();
        let _permit = self.copy.operation_permit()?;
        // Capture the local checkpoint, then release `inner` for the engine
        // flush: replicated batches keep applying (`handle` needs `inner`)
        // while it seals and commits. Every op <= cp was applied before the
        // capture, so the commit the flush makes holds it, whatever was
        // applied meanwhile; the WAL is trimmed only below that point.
        let cp = self.inner.lock().tracker.checkpoint();
        let waited = started.elapsed();
        self.copy().shard.flush()?;
        let committed = started.elapsed();
        let mut g = self.inner.lock();
        let gcp = g.global_checkpoint;
        // The engine's flush is detached: it can commit below `cp` (below
        // memtables another seal is still encoding). Never trim the WAL past
        // what the engine committed.
        let committed_seq_no = self.copy.shard.committed_seq_no();
        g.wal.trim_below(cp.min(gcp).min(committed_seq_no) + 1, gcp)?;
        if started.elapsed() >= std::time::Duration::from_secs(5) {
            tracing::warn!(index=%self.svc.name, shard=self.shard,
                wait_ms=waited.as_millis() as u64,
                commit_ms=committed.saturating_sub(waited).as_millis() as u64,
                trim_ms=started.elapsed().saturating_sub(committed).as_millis() as u64,
                "slow replica flush");
        }
        Ok(cp)
    }

    /// Trim WAL generations holding only ops `< below_seq_no` (clamped to
    /// `min(local checkpoint, global checkpoint) + 1`). The caller
    /// guarantees every op `< below_seq_no` is committed in the engine
    /// (prefer [`flush`](Self::flush), which captures that point itself).
    pub fn trim_wal(&self, below_seq_no: i64) -> Result<()> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        let gcp = g.global_checkpoint;
        let below = below_seq_no.min(g.tracker.checkpoint().min(gcp) + 1).min(self.copy.shard.committed_seq_no() + 1);
        g.wal.trim_below(below, gcp)
    }

    /// Promotion of this copy to primary under `new_term` (ES
    /// `PrimaryReplicaSyncer`): collects the resync for the other copies —
    /// every op held in `(global checkpoint, max_seq_no]`, gaps (and seq_nos
    /// only covered by a snapshot fast-forward) filled with `NoOp`s of the
    /// new term — then commits everything into the engine (flush), treats
    /// the gaps as filled (`fillSeqNoGaps`) and empties the WAL. From now on
    /// the engine translog is the source of truth: open the primary's
    /// tailer at `max_seq_no + 1` and send the returned resync to every
    /// replica ([`PromotionResync::to_request`]).
    pub fn promote_to_primary(&self, new_term: u64) -> Result<PromotionResync> {
        // Promotion applies held, acknowledged ops and seals this copy's
        // history (gap no-ops, flush, WAL trim): it is a replay, never refused
        // by a circuit breaker. A refusal would fail what may be the last
        // in-sync copy (a red shard) for memory it cannot free by failing.
        elasticsearch_engine::memory::replaying("primary promotion", || self.promote_to_primary_replayed(new_term))
    }

    fn promote_to_primary_replayed(&self, new_term: u64) -> Result<PromotionResync> {
        let _permit = self.copy.operation_permit()?;
        let mut g = self.inner.lock();
        if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        // A promotion to this very term that adopted it but did not finish
        // (a refusal or I/O error after the term was saved, or a crash) runs
        // again: the term alone does not prove gaps were filled, the resync
        // collected and the WAL trimmed.
        let resuming = new_term == g.primary_term && g.wal.state().promotion_pending == Some(new_term);
        if new_term <= g.primary_term && !resuming {
            return Err(ReplicationError::StalePrimaryTerm { ours: g.primary_term, theirs: new_term });
        }
        // Adopt the new term first: from here on every batch of the old
        // primary is rejected at the fence. Then drain the batches already
        // past it (ES: the primary-term bump acquires all operation
        // permits), so the max seq_no below covers every op this copy took
        // and none lands in the engine after the WAL is trimmed. The pending
        // marker is written with the term, so no failure (or crash) between
        // here and the end can leave a copy that holds the new term but
        // skips the rest of its promotion.
        g.wal.update_state(|s| {
            s.primary_term = new_term;
            s.resync_pending_gcp = None;
            s.promotion_pending = Some(new_term);
        })?;
        g.primary_term = new_term;
        self.publish_status(&g);
        while g.applying > 0 {
            self.drained.wait(&mut g);
        }
        if let Some(r) = &g.failed {
            return Err(ReplicationError::NeedsFullRecovery { reason: r.clone() });
        }
        let gcp = g.global_checkpoint.max(-1);
        let max = g.tracker.max_seq_no();
        let held = ops_above_wal(&g.wal, gcp)?.0;
        let mut ops = AdmittedBatch::new(elasticsearch_engine::memory::breakers())?;
        for seq in gcp + 1..=max {
            // The held batch owns all source strings while each independently
            // admitted outgoing operation is constructed. No naked clone escapes.
            let next = match held.binary_search_by_key(&seq, ReplicaOp::seq_no) {
                Ok(at) => ReplicaOp::try_copy_selected(&held[at..at + 1], |_| true)?,
                Err(_) => ReplicaOp::try_noop(seq, new_term, "filling seq_no gap on primary promotion")?,
            };
            if !matches!(&next[0], ReplicaOp::NoOp { .. }) && !g.tracker.has_processed(seq) {
                // A held op may have been acknowledged by its old primary
                // (every other in-sync copy took it): it is applied (a replay,
                // never refused), never replaced. Resyncing a no-op in its
                // place would drop an acknowledged write here and make every
                // copy holding the real op diverge. One that cannot apply
                // fails the promotion instead; the pending marker makes the
                // next promotion under this term run it again.
                let applied = crate::mapping_update::without_derived_mappings(|| self.apply_one(&mut g, &next[0]));
                if let Err(e) = applied {
                    tracing::error!(seq_no = seq, "a held op could not be applied on promotion, failing the promotion: {e}");
                    return Err(e);
                }
                g.tracker.mark_processed(seq);
            }
            ops.try_append(next).map_err(|(e, _)| e)?;
        }
        // Preserve grouped durable no-op writes. One fsync per missing seq_no
        // would make a large promotion gap quadratic in IO latency.
        let slots = ops.len().checked_mul(std::mem::size_of::<(u64,i64)>() + std::mem::size_of::<i64>())
            .ok_or_else(|| elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","promotion no-op slots overflow"))?;
        #[cfg(test)]
        if self.interrupt_promotion.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return Err(ReplicationError::Io(std::io::Error::other("injected promotion interruption")));
        }
        let _noop_memory = self.admit(slots as u64,"promotion grouped no-op slots")?;
        let mut noops = replay_vec(ops.len())?;
        let mut seq_nos = replay_vec(ops.len())?;
        for op in &ops {
            if let ReplicaOp::NoOp { seq_no, primary_term, .. } = op { noops.push((*primary_term,*seq_no)); }
        }
        noops.sort_unstable();
        g.resync = None;
        let mut at = 0;
        while at < noops.len() {
            let term = noops[at].0;
            seq_nos.clear();
            while at < noops.len() && noops[at].0 == term { seq_nos.push(noops[at].1); at += 1; }
            self.copy().shard.fill_seq_no_gaps(&seq_nos, term, "filling seq_no gap on primary promotion")?;
        }
        self.copy().shard.flush()?;
        g.tracker.fast_forward(max);
        // Everything up to `max` is committed in the engine: the promotion
        // is done once the marker clears. Cleared before the trim, never
        // after it: a promotion re-run over a trimmed WAL would no longer
        // find the held ops and would resync them as no-ops.
        g.wal.update_state(|s| s.promotion_pending = None)?;
        // Only below what that flush committed (see `flush`).
        g.wal.trim_below(max.min(self.copy.shard.committed_seq_no()) + 1, gcp)?;
        g.tombstones.clear();
        Ok(PromotionResync { primary_term: new_term, global_checkpoint: gcp, max_seq_no: max, ops })
    }

    fn prune_tombstones(g: &mut Inner) {
        if !g.tombstones.is_empty() {
            let cp = g.tracker.checkpoint();
            g.tombstones.retain(|_, s| s.seq_no > cp);
        }
    }

    /// seq_no this copy holds for `id` (live doc or tombstone).
    fn current_seq_no(&self, g: &Inner, id: &str) -> Option<i64> {
        // The engine's newest knowledge of the id: live copy, pooled
        // (not yet checked-in) op, delete tombstone.
        let live = self.copy().shard.known_seq_no(id);
        match (live, g.tombstones.get(id).map(|entry|entry.seq_no)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    fn apply_one(&self, g: &mut Inner, op: &ReplicaOp) -> Result<()> {
        let tombstone=prepare_tombstone(op)?;
        self.note_terms(std::slice::from_ref(op))?;
        self.apply_one_prepared(g, op, None, tombstone)
    }

    /// A request-breaker admission (see the `refuse_admission` test seam).
    fn admit(&self, bytes: u64, label: &'static str) -> elasticsearch_engine::error::Result<elasticsearch_engine::memory::Reservation> {
        #[cfg(test)]
        {
            let mut refuse = self.refuse_admission.lock();
            // As the breaker: nothing is refused during a replay.
            if *refuse == Some(label) && !elasticsearch_engine::memory::is_replaying() {
                *refuse = None;
                return Err(elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("injected refusal of [{label}]")));
            }
        }
        elasticsearch_engine::memory::breakers().reserve_request(bytes, label)
    }

    /// Test hook: the number of records in this copy's WAL.
    #[cfg(test)]
    pub(crate) fn wal_record_count(&self) -> usize {
        let mut count = 0;
        self.inner.lock().wal.for_each_record(|_| { count += 1; Ok(()) }).map(|()| count).unwrap_or(0)
    }

    /// Each op's own primary term for the engine to stamp on its document
    /// (`_primary_term`, the version map): the engine otherwise stamps the
    /// term it knows for a seq_no, and a copy that stayed a replica through
    /// a failover, or was recovered, would disagree with the primary.
    fn note_terms(&self, ops: &[ReplicaOp]) -> Result<()> {
        let staged = self.stage_terms(ops)?;
        self.note_staged_terms(staged, ops)
    }

    /// The admitted half of [`note_terms`](Self::note_terms): a batch stages
    /// its terms before its WAL append, so no admission can refuse it
    /// between the append and marking its ops processed.
    fn stage_terms(&self, ops: &[ReplicaOp]) -> Result<StagedTerms> {
        let bytes=ops.len().checked_mul(std::mem::size_of::<(i64,u64)>()).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","replica term slots overflow"))?;
        let _memory=self.admit(bytes as u64,"replica term staging")?;
        let mut terms=replay_vec(ops.len())?;
        terms.extend(ops.iter().filter(|o| !matches!(o, ReplicaOp::NoOp { .. })).map(|o| (o.seq_no(), o.primary_term())));
        Ok(StagedTerms { terms, _memory })
    }

    fn note_staged_terms(&self, staged: StagedTerms, ops: &[ReplicaOp]) -> Result<()> {
        if !staged.terms.is_empty() {
            self.copy().shard.note_op_terms(&staged.terms)?;
        }
        // A replica no-op is part of the engine's processed history too.
        // Updating only the replication tracker leaves a permanent engine
        // checkpoint hole; later promotion/flush can trim the WAL while file
        // recovery still asks to replay from before that hole.
        for op in ops {
            if let ReplicaOp::NoOp { seq_no, primary_term, reason } = op {
                self.copy().shard.fill_seq_no_gaps(&[*seq_no], *primary_term, reason)?;
            }
        }
        Ok(())
    }

    fn apply_one_prepared(&self, g: &mut Inner, op: &ReplicaOp, prepared: Option<elasticsearch_engine::document::ParsedDoc>, tombstone: Option<PreparedTombstone>) -> Result<()> {
        match op {
            ReplicaOp::Index { id, source_json, routing, seq_no, version, .. } => {
                if self.current_seq_no(g, id).is_some_and(|cur| cur >= *seq_no) {
                    // The durable replica WAL already contains this operation.
                    // It must advance engine history even though its document
                    // mutation loses to a newer op (same as replay_staged).
                    self.copy().shard.replay_staged(vec![elasticsearch_engine::shard::StagedReplay::Skip { seq_no: *seq_no }])?;
                    g.stats.ops_stale += 1;
                    return Ok(());
                }
                // The engine's full replay path: parse against the copy's
                // mapping (persisting dynamic updates), schema sync, stored
                // fields / synthetic `_source`, then `Shard::replay_index`.
                // Fencing stays here: it does no term check.
                match prepared {
                    Some(pd) => self.copy().replay_prepared(id, source_json, pd, routing.clone(), *seq_no, *version)?,
                    None => self.copy().replay_source(id, source_json, routing.clone(), *seq_no, *version)?,
                }
                g.tombstones.remove(id);
                g.stats.ops_applied += 1;
            }
            ReplicaOp::Delete { id, seq_no, version, .. } => {
                if self.current_seq_no(g, id).is_some_and(|cur| cur >= *seq_no) {
                    // The durable replica WAL already contains this operation.
                    // It must advance engine history even though its document
                    // mutation loses to a newer op (same as replay_staged).
                    self.copy().shard.replay_staged(vec![elasticsearch_engine::shard::StagedReplay::Skip { seq_no: *seq_no }])?;
                    g.stats.ops_stale += 1;
                    return Ok(());
                }
                self.copy().shard.replay_delete(id, *seq_no, *version)?;
                let (key,entry)=tombstone.expect("delete admission prepared before engine mutation");
                g.tombstones.insert(key,entry);
                g.stats.ops_applied += 1;
            }
            ReplicaOp::NoOp { .. } => g.stats.noops += 1,
        }
        Ok(())
    }

    /// Merge the primary's mapping when it differs from ours, unless this
    /// copy already holds its cluster mapping version (`version`, 0 =
    /// unknown) or a newer one: this node may have applied a newer
    /// committed mapping (a raised `ignore_above`, say) before a request
    /// carrying the older one arrives, and that must not be reverted.
    fn sync_mapping(&self, g: &mut Inner, primary: &Value, version: u64) -> Result<()> {
        let have = self.svc.mapping_version();
        let ours = self.svc.mapping_json();
        // Not newer than ours: the primary may have read newer content under
        // its label (its engine installs a mapping before recording its
        // version), so take what it adds, never what it would revert. An
        // unknown version (0: a primary from before versions, or one that
        // has not applied a state) is treated the same: never a revert.
        if version == 0 || have >= version {
            if let Some(added) = crate::mapping_update::added_mappers(&ours, primary) {
                self.svc.put_mapping(&added)?;
                g.stats.mapping_updates += 1;
            }
            return Ok(());
        }
        if &ours != primary {
            self.svc.put_mapping(primary)?;
            g.stats.mapping_updates += 1;
        }
        self.svc.advance_mapping_version(version);
        Ok(())
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tombstone_key_and_owner_are_prepared_before_publication() {
        use elasticsearch_engine::rustic_core::admission::Budget;
        let budget=Budget::new(1<<20);
        let op=ReplicaOp::Delete{id:"deleted-key".into(),seq_no:7,version:2,primary_term:1};
        let (key,entry)=prepare_tombstone_on(&op,&budget).unwrap().unwrap();
        drop(op);
        assert_eq!(key,"deleted-key");
        assert_eq!(entry.seq_no,7);
        assert!(budget.used()>=TOMBSTONE_NODE_BYTES as u64);
        drop((key,entry));
        assert_eq!(budget.used(),0);
        let tiny=Budget::new(1);
        let op=ReplicaOp::Delete{id:"denied".into(),seq_no:8,version:3,primary_term:1};
        assert!(prepare_tombstone_on(&op,&tiny).is_err());
        assert_eq!(op.seq_no(),8);
        assert_eq!(tiny.used(),0);
    }


    #[test]
    fn admitted_promotion_history_preserves_first_duplicate_and_legacy_wire() {
        let first = ReplicaOp::NoOp { seq_no: 4, primary_term: 2, reason: "first".into() };
        let later = ReplicaOp::NoOp { seq_no: 4, primary_term: 3, reason: "later".into() };
        let records = vec![
            WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![first.clone()]), global_checkpoint: 2 },
            WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![later]), global_checkpoint: 2 },
            WalRecord::FastForward(5),
        ];
        let (ops, missing) = ops_above_in(&records, 2).unwrap();
        drop(records);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], first);
        assert_eq!(missing, vec![3, 5]);
        let plan = PromotionResync { primary_term: 7, global_checkpoint: 2, max_seq_no: 4, ops };
        #[derive(Serialize)]
        struct Legacy { primary_term: u64, global_checkpoint: i64, max_seq_no: i64, ops: Vec<ReplicaOp> }
        let legacy = Legacy { primary_term: 7, global_checkpoint: 2, max_seq_no: 4, ops: vec![first] };
        let encoded = bincode::serialize(&legacy).unwrap();
        assert_eq!(bincode::serialize(&plan).unwrap(), encoded);
        let decoded: PromotionResync = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn coordination_status_stays_available_while_replica_work_is_locked() {
        let dir = tempfile::tempdir().unwrap();
        let svc = Arc::new(IndexService::create(dir.path(), "idx",
            elasticsearch_engine::settings::normalize_index_settings(&json!({})),
            &json!({}), BTreeMap::new()).unwrap());
        let replica = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
        replica.bump_term(7, -1).unwrap();
        let guard = replica.inner.lock(); // models a long indexing batch/flush
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = replica.clone();
        let thread = std::thread::spawn(move || {
            tx.send((reader.primary_term(), reader.needs_full_recovery())).unwrap();
        });
        let observed = rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(guard); // release before asserting so a regression can exit cleanly
        thread.join().unwrap();
        assert_eq!(observed.unwrap(), (7, None));
        {
            let mut guard = replica.inner.lock();
            replica.fail(&mut guard, "diverged".into());
        }
        assert_eq!(replica.needs_full_recovery().as_deref(), Some("diverged"));
        drop(replica);
        let reopened = ReplicaApplier::open(svc, 0).unwrap();
        assert_eq!(reopened.primary_term(), 7);
        assert_eq!(reopened.needs_full_recovery().as_deref(), Some("diverged"));
    }
}
