//! Safe flushing in cluster mode.
//!
//! An engine flush (`Shard::flush`, also done by `Shard::force_merge` and by
//! `IndexService::set_open(false)`) rolls the translog and **deletes every
//! older generation**. On a primary that would destroy ops not yet read by
//! the replication tailer or not yet acked by a copy (or needed by a
//! recovery's translog phase), so in cluster mode every engine operation
//! that trims the translog must go through this protocol:
//!
//! * Per primary shard there is one [`WriteGate`] (a `RwLock<()>`). Every
//!   engine write (index/bulk/update/delete/by-query) holds
//!   [`WriteGate::write_permit`] (shared) while it executes. Flush, force
//!   merge and close hold [`WriteGate::flush_permit`] (exclusive), so no op
//!   can enter the translog while a flush decides what is safe to trim.
//! * [`safe_flush`] then, with the exclusive permit held:
//!   1. drains the tailer and hands the ops to `ship`, which must deliver
//!      them synchronously to the replication targets and record their
//!      acks in the tracker (or fail lagging copies through the master);
//!   2. checks `ReplicationTracker::all_tracked_acked(max_seq_no)`: every
//!      tracked copy — in-sync replicas *and* copies under recovery — has
//!      processed everything. If not, the flush is **skipped** (retry on
//!      the next maintenance tick): trimming would drop ops someone needs,
//!      ES retention-lease semantics;
//!   3. calls the engine flush, then trims this copy's replica WAL if it
//!      has one (a promoted ex-replica still holds ops in its WAL).
//!
//! Replicas flush through [`ReplicaApplier::flush`], which serialises with
//! `apply` and trims their WAL below the checkpoint captured before the
//! engine flush. A replica's own engine translog is empty (replayed ops are
//! not logged by the engine), so its flush trims nothing anyone needs.
//!
//! # Engine background duties the node crate must take over
//!
//! In cluster mode the node does **not** call `Node::start_background`
//! (its `maintenance` tick would flush primaries behind the tailer's back).
//! The node crate's own ~1 s maintenance loop must therefore do what
//! `Node::maintenance` does, minus the unsafe part:
//! * expire scroll and PIT contexts (`node.scrolls` / `node.pits`: retain
//!   entries whose `expires > now`);
//! * flush shards whose translog exceeds 512 MB (`shard.translog_size()`),
//!   **via [`safe_flush`] for primaries / [`ReplicaApplier::flush`] for
//!   replicas** instead of `shard.flush()`; replicas should also flush when
//!   their WAL (`ReplicaApplier::stats().wal_size_bytes`) grows past the
//!   same threshold;
//! * `core.optimize_vectors()` for every open index (safe as is);
//! * on shutdown, `Node::flush_all` must likewise be replaced by gated
//!   flushes (or happen after replication stopped).
//! Force merge and close requests on a primary must hold the flush permit
//! and drain/ack like [`safe_flush`] before calling the engine.

use super::checkpoint::ReplicationTracker;
use super::replica::ReplicaApplier;
use super::tailer::TranslogTailer;
use super::Result;

use elasticsearch_engine::translog::TranslogOp;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};

/// Per-shard write gate: writes share it, flushes take it exclusively.
#[derive(Default)]
pub struct WriteGate(RwLock<()>);

impl WriteGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Held (shared) for the duration of an engine write.
    pub fn write_permit(&self) -> RwLockReadGuard<'_, ()> {
        self.0.read()
    }

    /// Held (exclusive) while deciding on and performing a flush.
    pub fn flush_permit(&self) -> RwLockWriteGuard<'_, ()> {
        self.0.write()
    }
}

/// Result of [`safe_flush`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlushOutcome {
    Flushed {
        /// Every op `<=` this is committed and acked by every tracked copy.
        max_seq_no: i64,
        /// Ops drained from the tailer and shipped during the flush.
        drained: usize,
    },
    Skipped {
        max_seq_no: i64,
        /// Tracked copies below `max_seq_no` (allocation id, checkpoint).
        lagging: Vec<(String, i64)>,
    },
}

/// Flush a primary without trimming translog ops any copy still needs.
/// See the module docs for the protocol. `ship` receives the drained ops
/// (possibly empty) and must ship them and update `tracker` before
/// returning; `tracker` is not locked while `ship` runs.
pub fn safe_flush(
    svc: &elasticsearch_engine::index::IndexShard,
    gate: &WriteGate,
    tailer: &mut TranslogTailer,
    tracker: &Mutex<ReplicationTracker>,
    replica_wal: Option<&ReplicaApplier>,
    ship: &mut dyn FnMut(elasticsearch_engine::translog::AdmittedBatch<TranslogOp>) -> Result<()>,
) -> Result<FlushOutcome> {
    let _permit = gate.flush_permit();
    let max_seq_no = svc.shard.max_seq_no();
    let mut drained = 0;
    loop {
        let ops = tailer.poll()?;
        if ops.is_empty() {
            break;
        }
        drained += ops.len();
        ship(ops)?;
    }
    let lagging = tracker.lock().lagging(max_seq_no);
    if !lagging.is_empty() {
        tracing::debug!(index = %svc.name, max_seq_no, ?lagging, "flush skipped: copies have not acked every op");
        return Ok(FlushOutcome::Skipped { max_seq_no, lagging });
    }
    match replica_wal {
        Some(r) => {
            r.flush()?;
        }
        None => svc.shard.flush()?,
    }
    Ok(FlushOutcome::Flushed { max_seq_no, drained })
}
