//! Per-shard replication and peer-recovery primitives (no networking).
//!
//! Everything here works on one shard copy backed by an engine
//! [`IndexService`](elasticsearch_engine::index::IndexService). All inputs
//! and outputs that cross node boundaries are plain serde structs
//! ([`ReplicaOp`], [`ReplicationRequest`], [`ReplicationResponse`],
//! [`recovery::RecoveryStartInfo`], [`recovery::RecoveryState`], ...) so
//! the integrator can ship them over the transport with bincode. JSON
//! payloads (mappings, settings) travel as JSON *strings* because
//! `serde_json::Value` is not bincode-decodable.
//!
//! | module | role |
//! |---|---|
//! | [`tailer`] | reads the primary engine's translog files incrementally |
//! | [`ops`] | the wire form of an operation ([`ReplicaOp`]) |
//! | [`replica`] | applies op batches on a replica copy, with its own WAL |
//! | [`checkpoint`] | local checkpoints, global checkpoint, retention |
//! | [`recovery`] | operation-based peer recovery (source + target) |
//! | [`file_recovery`] | file-based peer recovery: commit files in chunks, then the translog phase |
//! | [`flush_gate`] | the write gate and the safe-flush protocol |
//!
//! # Write path (primary node), exact sequence
//!
//! 1. Take `WriteGate::write_permit()` for the shard (shared; many writers).
//! 2. Execute the write through the engine (`IndexService::write`,
//!    `update`, bulk, by-query ...). The engine assigns `_seq_no`s and
//!    appends to its translog. Release the permit.
//! 3. The shard's single replication pump polls its [`TranslogTailer`]
//!    (`poll_limited`), converts each record with
//!    [`ReplicaOp::from_translog`] (current primary term) and marks each
//!    seq_no processed in the primary's own [`LocalCheckpointTracker`];
//!    it then calls `ReplicationTracker::update_local_checkpoint(primary_aid, cp)`.
//! 4. It sends one [`ReplicationRequest`] (ops, primary term, global
//!    checkpoint, the primary's mapping JSON when it changed since the
//!    last request to that copy) to every id in
//!    `ReplicationTracker::replication_targets()` (in-sync copies **and**
//!    copies being recovered).
//! 5. Each replica calls [`ReplicaApplier::handle`] and answers with a
//!    [`ReplicationResponse`] (`local_checkpoint`, `max_seq_no`); the
//!    primary feeds that into `update_local_checkpoint(aid, cp)`.
//! 6. The client is acked once `ReplicationTracker::all_in_sync_acked(seq)`
//!    holds for the write's highest seq_no, or the lagging copies were
//!    failed through the master (`shard-failed`, which removes them from
//!    the in-sync set; apply that state via `update_from_cluster_state`).
//!    `_shards.{total,successful,failed}` = 1 + in-sync replicas, acks, fails.
//!
//! Requests may be delivered out of order, duplicated or retried: replicas
//! skip ops they already processed (checkpoint tracker) and ops that are
//! older than what they hold for the same `_id` (seq_no comparison against
//! the live doc or a delete tombstone).
//!
//! # Recovery, exact sequence (target copy `T` of primary `P`)
//!
//! 1. Master allocates `T` (INITIALIZING). On `P`'s node:
//!    `tracker.init_tracking(T)` — from now on `safe_flush` will not trim
//!    ops `T` has not acked (retention).
//! 2. `P` calls [`recovery::RecoverySource::start`] and sends
//!    `info()` ([`recovery::RecoveryStartInfo`]) to `T`'s node.
//! 3. `T` creates the empty engine index from `info` (e.g.
//!    `Node::create_index(name, &RecoveryTarget::create_index_body(&info)?)`
//!    or [`recovery::RecoveryTarget::create_index`]), builds
//!    [`recovery::RecoveryTarget::new`] and exposes its `applier()` to
//!    replication requests.
//! 4. `P`'s pump adds `T` to the copies it ships live ops to, and records
//!    `L` = the highest seq_no it shipped *before* adding `T` (every op
//!    `> L` reaches `T` through live replication).
//! 5. `P` streams `next_snapshot_batch()` until `None`;
//!    `T` → `apply_snapshot_batch` each; then `T.snapshot_done()`
//!    (verifies the doc count and fast-forwards the checkpoint to
//!    `snapshot_max_seq_no`, see [`recovery`] for why that is safe).
//! 6. `P` streams `next_translog_batch()` until `last_seq_no_sent() >= L`;
//!    `T` → `apply_translog_batch`. (Snapshot, translog phase and live ops
//!    overlap freely; the applier makes the overlap idempotent.)
//! 7. `T.finalize(global_checkpoint)` (flushes, trims the WAL, stage DONE;
//!    retry while it reports the copy is still below the gcp).
//!    `P`: `tracker.update_local_checkpoint(T, cp)`, then
//!    `tracker.mark_in_sync(T)` (retry while it returns `false`); the
//!    master marks the copy STARTED and adds it to the in-sync set.
//!
//! On `OpsTrimmed` anywhere, fall back to a fresh full recovery (delete the
//! target index and restart from step 2). A recovery started under term `T`
//! fails with `RecoveryAborted` once the target has seen a newer term.
//!
//! # Failover, exact sequence (new primary `B`, term `T+1`)
//!
//! 1. Every copy applying a cluster state where its primary changed calls
//!    `ReplicaApplier::bump_term(T+1, gcp)` (also implied by the first
//!    request carrying `T+1`). From then on requests with a lower term fail
//!    with `StalePrimaryTerm` — the deposed primary can never collect acks,
//!    so it can never ack a client write.
//! 2. `B` calls `ReplicaApplier::promote_to_primary(T+1)` → a
//!    [`PromotionResync`]; it stops using its applier, opens its
//!    [`TranslogTailer`] at `max_seq_no + 1`, builds a fresh
//!    [`ReplicationTracker`] with term `T+1`, and sends
//!    `resync.to_request(..)` to every other copy (`ReplicaApplier::resync`).
//! 3. A copy answering `NeedsFullRecovery` (it held ops above the gcp that
//!    `B` does not have) is failed via the master, wiped and peer-recovered.
//!    The old primary, when it comes back, is always re-recovered.
//! See [`replica`] ("Failover") for the exact divergence rules.

pub mod decode;
pub mod checkpoint;
pub mod file_recovery;
pub mod flush_gate;
pub mod ops;
pub mod recovery;
pub mod replica;
pub mod tailer;
pub mod throttle;
mod wal;

#[cfg(test)]
mod tests;

pub use checkpoint::{LocalCheckpointTracker, ReplicationTracker, NO_OPS_PERFORMED, UNASSIGNED_SEQ_NO};
pub use flush_gate::{safe_flush, FlushOutcome, WriteGate};
pub use ops::{ReplicaOp, ReplicationRequest, ReplicationResponse, ResyncRequest};
pub use recovery::{RecoverySource, RecoveryStage, RecoveryStartInfo, RecoveryState, RecoveryTarget};
pub use replica::{PromotionResync, ReplicaApplier, ReplicaStats};
pub use tailer::{TailerPosition, TranslogTailer};

use elasticsearch_engine::error::EsError;

/// Errors of the replication layer.
#[derive(Debug, thiserror::Error)]
pub enum ReplicationError {
    #[error("{0}")]
    Engine(#[from] EsError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("codec: {0}")]
    Codec(String),
    /// The ops a reader needs were already trimmed from the primary's
    /// translog: the caller must fall back to a full (snapshot) recovery.
    #[error("operations trimmed from translog: needed from seq_no [{needed_from}], oldest available [{available_from}]")]
    OpsTrimmed { needed_from: i64, available_from: i64 },
    #[error("corrupt {what}: {detail}")]
    Corrupt { what: String, detail: String },
    /// A request from a deposed primary (its term is below the highest
    /// term this copy has durably accepted) — rejected, never acked.
    #[error("operation primary term [{theirs}] is too old (current [{ours}])")]
    StalePrimaryTerm { ours: u64, theirs: u64 },
    /// This copy holds operations the current primary does not have (or
    /// cannot prove it has). The engine cannot roll back: the caller must
    /// wipe the copy and run a full peer recovery. Sticky (persisted).
    #[error("shard copy diverged from the primary and needs a full recovery: {reason}")]
    NeedsFullRecovery { reason: String },
    /// A recovery started under `recovery_term` saw a newer primary term.
    #[error("recovery started under primary term [{recovery_term}] aborted: current term is [{current_term}]")]
    RecoveryAborted { recovery_term: u64, current_term: u64 },
    #[error("illegal state: {0}")]
    IllegalState(String),
}

impl ReplicationError {
    /// The ES exception type this maps to on the wire / in REST errors.
    pub fn es_type(&self) -> &'static str {
        match self {
            ReplicationError::Engine(error) if error.kind == "circuit_breaking_exception" => "circuit_breaking_exception",
            ReplicationError::Engine(_) => "engine_exception",
            ReplicationError::Io(_) | ReplicationError::Codec(_) | ReplicationError::Corrupt { .. } => "translog_corrupted_exception",
            ReplicationError::OpsTrimmed { .. } => "retention_lease_not_found_exception",
            ReplicationError::StalePrimaryTerm { .. } | ReplicationError::IllegalState(_) => "illegal_state_exception",
            ReplicationError::NeedsFullRecovery { .. } => "shard_not_in_primary_mode_exception",
            ReplicationError::RecoveryAborted { .. } => "recovery_failed_exception",
        }
    }
}

// Preserve the typed engine breaker error through guarded source decoding.
impl From<elasticsearch_engine::rustic_core::RusticError> for ReplicationError {
    fn from(error: elasticsearch_engine::rustic_core::RusticError) -> Self {
        Self::Engine(EsError::from(error))
    }
}

impl From<bincode::Error> for ReplicationError {
    fn from(e: bincode::Error) -> Self {
        ReplicationError::Codec(e.to_string())
    }
}

impl From<serde_json::Error> for ReplicationError {
    fn from(e: serde_json::Error) -> Self {
        ReplicationError::Codec(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ReplicationError>;

/// Record framing shared with the engine translog:
/// `[len u32 LE][crc32 u32 LE][body]`.
pub(crate) mod frame {
    pub const HEADER_LEN: usize = 8;

    pub fn encode(body: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&elasticsearch_engine::translog::crc32(body).to_le_bytes());
        out.extend_from_slice(body);
    }

    /// Outcome of decoding one record at the start of `data`.
    pub enum Decoded<'a> {
        /// A complete, checksum-valid record: body and total length.
        Record(&'a [u8], usize),
        /// Not enough bytes yet (a torn or in-progress tail).
        Incomplete,
        /// Complete length but bad checksum.
        Corrupt,
    }

    pub fn decode(data: &[u8]) -> Decoded<'_> {
        if data.len() < HEADER_LEN {
            return Decoded::Incomplete;
        }
        let len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let Some(end) = HEADER_LEN.checked_add(len) else { return Decoded::Corrupt };
        if end > data.len() {
            return Decoded::Incomplete;
        }
        let body = &data[HEADER_LEN..end];
        if elasticsearch_engine::translog::crc32(body) != crc {
            return Decoded::Corrupt;
        }
        Decoded::Record(body, end)
    }
}

/// Write `bytes` to `dir/name` atomically (tmp + fsync + rename + dir fsync).
pub(crate) fn write_atomic(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = dir.join(format!("{name}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(name))?;
    sync_dir(dir);
    Ok(())
}

pub(crate) fn sync_dir(dir: &std::path::Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

mod ops_copy;
