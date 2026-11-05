//! Operation-based peer recovery.
//!
//! **Source** (on the primary): [`RecoverySource::start`]
//! 1. reads `s0 = Shard::max_seq_no()` — the engine assigns seq_nos and
//!    applies a whole batch under its write lock, so every op `<= s0` is in
//!    the core index once this returns;
//! 2. takes a point-in-time snapshot under the core write lock: seals the
//!    memtable (`refresh`) and clones every segment `Arc` together with its
//!    deletion bitmap `Arc` (deletions are copy-on-write, so later deletes
//!    do not show through);
//! 3. opens a [`TranslogTailer`] at `s0 + 1`.
//!
//! It then streams (a) every live root document of the snapshot as
//! [`ReplicaOp::Index`] with its stored `_id`, `_source`, `_routing`,
//! `_version` and `_seq_no`, in batches of `batch_docs` docs or
//! `batch_bytes` bytes, and (b) the translog ops `> s0` while writes
//! continue. Phases (a) and (b) overlap (the snapshot may already contain
//! ops `> s0`); that is harmless because the replica applier skips ops that
//! are not newer than what it holds for the id.
//!
//! **Why the target may jump its local checkpoint to
//! `snapshot_max_seq_no`.** Deleted documents leave no trace in the
//! snapshot and overwritten versions of a document are gone too, so the
//! seq_nos of those ops never reach the target and per-op accounting would
//! leave permanent gaps — the local checkpoint would stall forever. But the
//! snapshot is an atomic image of the engine after *some prefix* of the op
//! history: it reflects every op `<= s1` (the engine's max seq_no at
//! snapshot time), with `s1 >= s0` and `s1 >=` the largest `_seq_no` of any
//! live doc in it. So once all snapshot docs are applied, every op `<=
//! snapshot_max_seq_no = max(s0, max live _seq_no)` is reflected on the
//! target, which is exactly the definition of "processed" — the target
//! fast-forwards its checkpoint there (durably, through its WAL). Ops in
//! `(s0, snapshot_max_seq_no]` that arrive later through the translog
//! phase or live replication are then dropped as already processed, which
//! is correct for the same reason.
//!
//! **Target**: [`RecoveryTarget`] wraps a [`ReplicaApplier`] over the newly
//! created (empty) engine index and records ES `_recovery` progress
//! ([`RecoveryState`]).
//!
//! A file-based recovery ([`super::file_recovery`]) uses the same source
//! and target for its phase 2: the source starts with no snapshot docs and
//! its translog phase at the shipped commit's local checkpoint + 1
//! ([`RecoverySource::start_from`]); the target fast-forwards to that
//! checkpoint (every op at or below it is in the shipped commit) and records
//! the file phase's progress ([`RecoveryTarget::set_file_stats`]).

use super::ops::ReplicaOp;
use super::replica::ReplicaApplier;
use super::tailer::TranslogTailer;
use super::{ReplicationError, Result};
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::mapping::{ID_FIELD, PRIMARY_TERM_FIELD, ROUTING_FIELD, SEQ_NO_FIELD, SOURCE_FIELD, VERSION_FIELD};
use elasticsearch_engine::rustic_core::postings::BitSet;
use elasticsearch_engine::rustic_core::query::LeafReader;
use elasticsearch_engine::rustic_core::segment::Segment;
use elasticsearch_engine::settings::{self, FlatSettings};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Settings the engine fills in per copy; not copied to the target.
const PER_COPY_SETTINGS: &[&str] = &["index.uuid", "index.provided_name", "index.creation_date", "index.version.created"];

fn snapshot_vec_growth_bytes(len: usize, capacity: usize) -> u64 {
    if len < capacity { 0 } else {
        (len.saturating_add(1) as u64).saturating_mul(std::mem::size_of::<ReplicaOp>() as u64)
    }
}

#[cfg(test)]
mod snapshot_growth_tests {
    use super::*;
    #[test]
    fn full_operation_vector_requires_old_plus_entire_replacement() {
        let slot = std::mem::size_of::<ReplicaOp>() as u64;
        let old = 4096 * slot;
        let available = old + slot;
        let required = old + snapshot_vec_growth_bytes(4096, 4096);
        assert!(required > available, "old admission incorrectly allowed this allocation");
        assert_eq!(required, (4096 + 4097) * slot);
        assert_eq!(snapshot_vec_growth_bytes(4095, 4096), 0);
    }
}

fn now_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Batching knobs of a recovery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryOptions {
    /// Max docs per snapshot batch.
    pub batch_docs: usize,
    /// Max (approximate) bytes per snapshot batch.
    pub batch_bytes: usize,
    /// Max ops per translog batch.
    pub translog_batch_ops: usize,
    /// A snapshot batch never spans two segments (a restore re-creates the
    /// source's segments, whose order a timestamp-sorted index searches by).
    #[serde(default)]
    pub segment_batches: bool,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self { batch_docs: 1000, batch_bytes: 512 * 1024, translog_batch_ops: 1000, segment_batches: false }
    }
}

/// What the source tells the target before streaming.
#[path="start_info.rs"]
mod start_info;
pub use start_info::RecoveryStartInfo;

/// Primary side of one recovery.
pub struct RecoverySource {
    svc: Arc<IndexService>,
    /// The engine copy this recovery serves, captured when it started (the
    /// primary's own copy): every later step reads it, never whatever copy
    /// the index service holds by then. A primary whose copy the node
    /// detached (relocated away, closed for a reset) keeps serving from the
    /// copy it pinned, or fails with that copy's closed error; it never
    /// panics on an empty slot or tails another incarnation's translog.
    copy: Arc<elasticsearch_engine::index::IndexShard>,
    info: RecoveryStartInfo,
    opts: RecoveryOptions,
    segments: Vec<(Arc<Segment>, Arc<BitSet>)>,
    cursor: (usize, u32),
    tailer: TranslogTailer,
    docs_sent: u64,
    bytes_sent: u64,
    translog_ops_sent: u64,
    last_seq_no_sent: i64,
    pub(crate) pending_response: Option<(bool, elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>)>,
}

impl RecoverySource {
    /// Resume above an independently verified committed replica prefix.
    /// The caller holds the primary write gate and has registered target
    /// retention. Prove every missing operation is still available before
    /// accepting reuse; a gap (including an unlogged no-op) requires the
    /// existing snapshot fallback. No document snapshot or checkpoint jump
    /// to the source's newer maximum is permitted here.
    pub fn resume(svc: Arc<IndexService>, shard: u32, primary_term: u64, checkpoint: i64, opts: RecoveryOptions) -> Result<RecoverySource> {
        let copy = crate::shards::try_copy_of(&svc, shard)?;
        Self::resume_copy(svc, copy, shard, primary_term, checkpoint, opts)
    }

    /// [`Self::resume`] from the primary's own engine copy `copy`.
    pub fn resume_copy(svc: Arc<IndexService>, copy: Arc<elasticsearch_engine::index::IndexShard>, shard: u32, primary_term: u64, checkpoint: i64, opts: RecoveryOptions) -> Result<RecoverySource> {
        let max = copy.shard.max_seq_no();
        if checkpoint < -1 || checkpoint > max {
            return Err(ReplicationError::IllegalState(format!("replica prefix [{checkpoint}] is outside primary history ending at [{max}]")));
        }
        // Every seq_no in (checkpoint, max] must be retained. Frames are not
        // strictly seq_no-ordered (concurrent batches append in any order
        // within a bounded window), so coverage is tracked as a checkpoint,
        // not by expecting each next seq_no in frame order (which forced a
        // full recovery on any reordering).
        let mut verify = TranslogTailer::open_at(copy.shard.path(), checkpoint + 1)?;
        let mut covered = super::checkpoint::LocalCheckpointTracker::new(checkpoint);
        while covered.checkpoint() < max {
            let ops = verify.poll_limited(opts.translog_batch_ops.max(1))?;
            if ops.is_empty() {
                return Err(ReplicationError::OpsTrimmed { needed_from: covered.checkpoint() + 1, available_from: max + 1 });
            }
            // Borrow the admitted batch through verification: its payload and
            // slot reservations remain alive until all sequence numbers are read.
            for op in &ops {
                covered.mark_processed(op.seq_no() as i64);
            }
        }
        let tailer = TranslogTailer::open_at(copy.shard.path(), checkpoint + 1)?;
        let meta = svc.meta();
        let flat: FlatSettings = meta.settings.iter().filter(|(k, _)| !PER_COPY_SETTINGS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
        let info = RecoveryStartInfo::try_new(&svc.name,&svc.uuid,shard,primary_term,&serde_json::to_string(&settings::as_flat_json(&flat))?,&serde_json::to_string(&meta.mappings.to_json())?,checkpoint + 1,checkpoint,0,0,(max - checkpoint) as u64).map_err(|e|super::ReplicationError::Engine(e))?;
        Ok(RecoverySource { svc, copy, info, opts, segments: Vec::new(), cursor: (0, 0), tailer,
            docs_sent: 0, bytes_sent: 0, translog_ops_sent: 0, last_seq_no_sent: checkpoint, pending_response: None })
    }

    /// Snapshot the primary and position the translog phase. The caller
    /// must already be tracking the target in its `ReplicationTracker`
    /// (`init_tracking`) so a concurrent `safe_flush` cannot trim the ops
    /// the translog phase needs.
    /// `shard` is the engine shard (= cluster shard number) to copy.
    pub fn start(svc: Arc<IndexService>, shard: u32, primary_term: u64, opts: RecoveryOptions) -> Result<RecoverySource> {
        let copy = crate::shards::try_copy_of(&svc, shard)?;
        Self::start_copy(svc, copy, shard, primary_term, opts)
    }

    /// [`Self::start`] from the primary's own engine copy `copy`.
    pub fn start_copy(svc: Arc<IndexService>, copy: Arc<elasticsearch_engine::index::IndexShard>, shard: u32, primary_term: u64, opts: RecoveryOptions) -> Result<RecoverySource> {
        let s0 = copy.shard.max_seq_no();
        // Every write so far, published to searches or not (the caller's
        // gate fences writes), sealed detached: the core lock is held only to
        // begin / install the seal and to collect the segments.
        copy.shard.seal_all()?;
        let segments: Vec<(Arc<Segment>, Arc<BitSet>)> = {
            let mut core = copy.shard.core.write();
            // Anything the detached seal could not take (nothing normally).
            core.seal().map_err(elasticsearch_engine::error::EsError::from)?;
            core.segments().iter().chain(core.hidden_segments()).map(|s| Ok((s.clone(), s.latest_deletions()?))).collect::<Result<Vec<_>>>()?
        };
        let mut docs = 0u64;
        let mut max_live_seq = -1i64;
        for (seg, dels) in &segments {
            let leaf = LeafReader::Segment(seg);
            let fields = seg.fast_fields();
            let seq_nos = fields.get_i64(SEQ_NO_FIELD);
            for d in 0..seg.num_docs {
                if dels.get(d) || !leaf.is_root(d) {
                    continue;
                }
                docs += 1;
                if let Some(s) = seq_nos.as_ref().and_then(|column| column.get(d)) {
                    max_live_seq = max_live_seq.max(s);
                }
            }
        }
        let tailer = TranslogTailer::open_at(copy.shard.path(), s0 + 1)?;
        // Count (without consuming) the translog ops present now.
        let pos_ops = TranslogTailer::open_at(copy.shard.path(), s0 + 1).and_then(|mut t| t.poll()).map(|v| v.len() as u64).unwrap_or(0);
        let meta = svc.meta();
        let flat: FlatSettings = meta.settings.iter().filter(|(k, _)| !PER_COPY_SETTINGS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
        let info = RecoveryStartInfo::try_new(&svc.name,&svc.uuid,shard,primary_term,&serde_json::to_string(&settings::as_flat_json(&flat))?,&serde_json::to_string(&meta.mappings.to_json())?,s0 + 1,s0.max(max_live_seq),docs,copy.shard.store_size(),pos_ops).map_err(|e|super::ReplicationError::Engine(e))?;
        Ok(RecoverySource { svc, copy, info, opts, segments, cursor: (0, 0), tailer, docs_sent: 0, bytes_sent: 0, translog_ops_sent: 0, last_seq_no_sent: s0, pending_response: None })
    }

    /// A recovery for a target that already reflects every op `<= from`
    /// (a stateless search copy restored from the object store): no
    /// snapshot, the translog phase starts at `from + 1`. Fails with
    /// `OpsTrimmed` when the translog no longer reaches back that far.
    pub fn start_from(svc: Arc<IndexService>, shard: u32, primary_term: u64, opts: RecoveryOptions, from: i64) -> Result<RecoverySource> {
        let copy = crate::shards::try_copy_of(&svc, shard)?;
        Self::start_from_copy(svc, copy, shard, primary_term, opts, from)
    }

    /// [`Self::start_from`] from the primary's own engine copy `copy` (the
    /// one whose commit a file recovery pinned).
    pub fn start_from_copy(svc: Arc<IndexService>, copy: Arc<elasticsearch_engine::index::IndexShard>, shard: u32, primary_term: u64, opts: RecoveryOptions, from: i64) -> Result<RecoverySource> {
        let tailer = TranslogTailer::open_at(copy.shard.path(), from + 1)?;
        let pos_ops = TranslogTailer::open_at(copy.shard.path(), from + 1).and_then(|mut t| t.poll()).map(|v| v.len() as u64).unwrap_or(0);
        let meta = svc.meta();
        let flat: FlatSettings = meta.settings.iter().filter(|(k, _)| !PER_COPY_SETTINGS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
        let info = RecoveryStartInfo::try_new(&svc.name,&svc.uuid,shard,primary_term,&serde_json::to_string(&settings::as_flat_json(&flat))?,&serde_json::to_string(&meta.mappings.to_json())?,from + 1,from,0,0,pos_ops).map_err(|e|super::ReplicationError::Engine(e))?;
        Ok(RecoverySource { svc, copy, info, opts, segments: Vec::new(), cursor: (0, 0), tailer, docs_sent: 0, bytes_sent: 0, translog_ops_sent: 0, last_seq_no_sent: from, pending_response: None })
    }

    pub fn info(&self) -> &RecoveryStartInfo {
        &self.info
    }

    pub fn index(&self) -> &Arc<IndexService> {
        &self.svc
    }

    /// Lowest seq_no this recovery still needs from the translog.
    pub fn retention_seq_no(&self) -> i64 {
        self.last_seq_no_sent + 1
    }

    /// Highest seq_no sent by the translog phase (`s0` before any).
    pub fn last_seq_no_sent(&self) -> i64 {
        self.last_seq_no_sent
    }

    pub fn docs_sent(&self) -> u64 {
        self.docs_sent
    }

    pub fn translog_ops_sent(&self) -> u64 {
        self.translog_ops_sent
    }

    /// The last snapshot batch ended a segment (the next starts a new one).
    pub fn at_segment_boundary(&self) -> bool {
        self.cursor.1 == 0
    }

    /// The next batch of snapshot documents; `None` when the snapshot is
    /// exhausted.
    pub fn next_snapshot_batch(&mut self) -> Result<Option<Vec<ReplicaOp>>> {
        self.next_snapshot_batch_guarded(|_| Ok(()))
    }

    /// Admission receives the batch's total retained plus transient peak bound.
    /// A failed admission/decode leaves the cursor at the start of this batch,
    /// so a retry cannot silently omit documents already visited in the attempt.
    pub fn next_snapshot_batch_guarded(&mut self, mut before: impl FnMut(u64) -> Result<()>) -> Result<Option<Vec<ReplicaOp>>> {
        let cursor = self.cursor;
        let result = self.read_snapshot_batch(&mut before);
        if result.is_err() { self.cursor = cursor; }
        result
    }

    fn read_snapshot_batch(&mut self, before: &mut impl FnMut(u64) -> Result<()>) -> Result<Option<Vec<ReplicaOp>>> {
        let mut out = Vec::new();
        let mut bytes = 0usize;
        // Logical byte limits and retained heap are different: reconstructed
        // source strings commonly retain spare output capacity after encoding.
        let mut retained_strings = 0u64;
        let term = self.info.primary_term;
        'outer: while self.cursor.0 < self.segments.len() {
            let (seg, dels) = &self.segments[self.cursor.0];
            let leaf = LeafReader::Segment(seg);
            // Sparse-column validation is proportional to segment size. Reuse
            // readers throughout the batch instead of opening them per document.
            let fields = seg.fast_fields();
            let seq_nos = fields.get_i64(SEQ_NO_FIELD);
            let versions = fields.get_i64(VERSION_FIELD);
            let terms = fields.get_i64(PRIMARY_TERM_FIELD);
            while self.cursor.1 < seg.num_docs {
                if out.len() >= self.opts.batch_docs || bytes >= self.opts.batch_bytes {
                    break 'outer;
                }
                let d = self.cursor.1;
                self.cursor.1 += 1;
                if dels.get(d) || !leaf.is_root(d) {
                    continue;
                }
                let retained = retained_strings.saturating_add((out.capacity() as u64).saturating_mul(std::mem::size_of::<ReplicaOp>() as u64));
                let mut decode_peak = 0u64;
                let doc = seg.get_doc_guarded(d, |peak| {
                    decode_peak = decode_peak.max(peak);
                    before(retained.saturating_add(decode_peak))
                })?
                    .ok_or_else(|| ReplicationError::Corrupt { what: "segment".into(), detail: format!("stored fields of doc {d} in segment {} missing", seg.id) })?;
                let text = |f: u16| doc.get(f).and_then(|v| v.as_text()).map(str::to_string);
                // Stored text copies and the owned op slot coexist with the
                // decoded document until the next loop iteration.
                let copied = [ID_FIELD, SOURCE_FIELD, ROUTING_FIELD].into_iter()
                    .filter_map(|field| doc.get(field).and_then(|value| value.as_text()))
                    .fold(0u64, |sum, text| sum.saturating_add(text.len() as u64));
                // Reallocation can keep the complete old allocation alive
                // while allocating its complete replacement, not just one slot.
                before(retained.saturating_add(decode_peak).saturating_add(copied)
                    .saturating_add(snapshot_vec_growth_bytes(out.len(), out.capacity())))?;
                out.try_reserve_exact(1).map_err(|_| ReplicationError::IllegalState("recovery batch allocation failed".into()))?;
                let retained = retained_strings.saturating_add((out.capacity() as u64)
                    .saturating_mul(std::mem::size_of::<ReplicaOp>() as u64));
                let source_json = match doc.get(SOURCE_FIELD).and_then(|value| value.as_text()) {
                    Some(raw) => elasticsearch_engine::vector_source::load_guarded(raw, &self.svc.lookup(), &leaf, d,
                        |peak| before(retained.saturating_add(decode_peak).saturating_add(copied).saturating_add(peak)))?.into_owned(),
                    None => String::new(),
                };
                let id = text(ID_FIELD).ok_or_else(|| ReplicationError::Corrupt { what: "segment".into(), detail: format!("doc {d} in segment {} has no _id", seg.id) })?;
                let op = ReplicaOp::Index {
                    id,
                    // With the vectors the stored `_source` leaves to the vector store.
                    source_json,
                    routing: text(ROUTING_FIELD),
                    seq_no: seq_nos.as_ref().and_then(|column| column.get(d)).unwrap_or(0),
                    version: versions.as_ref().and_then(|column| column.get(d)).unwrap_or(1) as u64,
                    // The document's own term: the target stores it as is.
                    primary_term: terms.as_ref().and_then(|column| column.get(d)).filter(|t| *t > 0).map_or(term, |t| t as u64),
                };
                if let ReplicaOp::Index { id, source_json, routing, .. } = &op {
                    retained_strings = retained_strings.saturating_add(id.capacity() as u64)
                        .saturating_add(source_json.capacity() as u64)
                        .saturating_add(routing.as_ref().map_or(0, |value| value.capacity() as u64));
                }
                bytes += op.estimated_bytes();
                out.push(op);
            }
            self.cursor = (self.cursor.0 + 1, 0);
            if self.opts.segment_batches && !out.is_empty() {
                break;
            }
        }
        if out.is_empty() {
            // Release the snapshot's segments as soon as it is done.
            self.segments.clear();
            self.cursor = (0, 0);
            return Ok(None);
        }
        self.docs_sent += out.len() as u64;
        self.bytes_sent += bytes as u64;
        Ok(Some(out))
    }

    /// Translog ops `> s0` written since the last call (empty when caught
    /// up; call again later while writes continue).
    pub fn next_translog_batch(&mut self) -> Result<elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>> {
        let copy = self.copy.clone();
        let ops = self.tailer.poll_mapped(self.opts.translog_batch_ops, |o| {
            let term = copy.shard.primary_term_of(o.seq_no() as i64);
            ReplicaOp::from_translog(o, term)
        })?;
        if let Some(m) = ops.iter().map(ReplicaOp::seq_no).max() {
            self.last_seq_no_sent = self.last_seq_no_sent.max(m);
        }
        self.translog_ops_sent += ops.len() as u64;
        Ok(ops)
    }
}

/// ES recovery stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecoveryStage {
    Init,
    Index,
    VerifyIndex,
    Translog,
    Finalize,
    Done,
}

impl RecoveryStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            RecoveryStage::Init => "INIT",
            RecoveryStage::Index => "INDEX",
            RecoveryStage::VerifyIndex => "VERIFY_INDEX",
            RecoveryStage::Translog => "TRANSLOG",
            RecoveryStage::Finalize => "FINALIZE",
            RecoveryStage::Done => "DONE",
        }
    }
}

#[path = "recovery_details.rs"]
mod recovery_details;
pub use recovery_details::FileDetails;

/// One file of a file-based recovery (ES `RecoveryState.FileDetail`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDetail {
    pub name: String,
    pub length: u64,
    pub recovered: u64,
    pub reused: bool,
}

/// The file phase of a recovery (ES `RecoveryState.Index`): every file of
/// the shipped commit, those reused from the target's own store, the bytes
/// received so far and the time spent throttled on either side.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecoveryStats {
    pub files_total: u64,
    pub files_reused: u64,
    pub files_recovered: u64,
    pub bytes_total: u64,
    pub bytes_reused: u64,
    pub bytes_recovered: u64,
    pub source_throttle_millis: u64,
    pub target_throttle_millis: u64,
    /// Wall time of the file phase.
    pub time_millis: u64,
    pub details: FileDetails,
}

impl FileRecoveryStats {
    /// ES `recoveredFilesPercent`: of the files not reused; 0% before the
    /// file list is known, 100% when nothing had to be copied.
    pub fn files_percent(&self) -> String {
        let total = self.files_total - self.files_reused.min(self.files_total);
        if self.files_total == 0 { "0.0%".into() } else if self.files_recovered >= total { "100.0%".into() } else { pct(self.files_recovered, total) }
    }

    /// ES `recoveredBytesPercent` (same rules, in bytes).
    pub fn bytes_percent(&self) -> String {
        let total = self.bytes_total - self.bytes_reused.min(self.bytes_total);
        if self.files_total == 0 { "0.0%".into() } else if self.bytes_recovered >= total { "100.0%".into() } else { pct(self.bytes_recovered, total) }
    }
}

#[path="recovery_text.rs"]
mod recovery_text;
pub use recovery_text::RecoveryText;

/// Progress of one recovery, in ES `_recovery` / `_cat/recovery` terms.
///
/// Operation-based recovery ships everything as operations (like ES's
/// ops-based peer recovery, which has no file phase): `files` stay 0 and
/// the snapshot docs are counted as translog operations; the doc and byte
/// counters of the snapshot phase are kept separately as well. A
/// file-based recovery reports its file phase in `files`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryState {
    pub index: RecoveryText,
    pub shard: u32,
    pub stage: RecoveryStage,
    pub primary: bool,
    pub source_node: Option<RecoveryText>,
    pub target_node: Option<RecoveryText>,
    pub start_time_millis: u64,
    pub stop_time_millis: Option<u64>,
    pub total_time_millis: u64,
    pub snapshot_docs_total: u64,
    pub snapshot_docs_recovered: u64,
    pub snapshot_bytes_estimate: u64,
    pub snapshot_bytes_recovered: u64,
    pub index_time_millis: u64,
    pub verify_index_time_millis: u64,
    pub translog_ops_recovered: u64,
    /// Snapshot docs + translog ops recovered so far.
    pub ops_recovered: u64,
    /// Snapshot docs + translog ops known at start.
    pub ops_total_on_start: u64,
    pub translog_time_millis: u64,
    pub finalize_time_millis: u64,
    /// The file phase (all zero for an operations-based recovery).
    #[serde(default)]
    pub files: FileRecoveryStats,
    /// A primary restored from a snapshot (ES `RecoverySource.SNAPSHOT`):
    /// rendered as `type: SNAPSHOT` with this source; `None` is a peer
    /// recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<SnapshotRecoverySource>,
}

/// ES `RecoverySource.SnapshotRecoverySource` as `_recovery` renders it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRecoverySource {
    pub repository: RecoveryText,
    pub snapshot: RecoveryText,
    pub version: Option<RecoveryText>,
    /// The snapshotted index (before any rename).
    pub index: RecoveryText,
    pub restore_uuid: RecoveryText,
}

fn pct(done: u64, total: u64) -> String {
    if total == 0 {
        "0.0%".into()
    } else {
        format!("{:.1}%", (done as f64 * 100.0 / total as f64).min(100.0))
    }
}

impl RecoveryState {
    fn new(info: &RecoveryStartInfo) -> Result<Self> {
        Ok(Self {
            index: RecoveryText::try_new(&info.index)?,
            shard: info.shard,
            stage: RecoveryStage::Init,
            primary: false,
            source_node: None,
            target_node: None,
            start_time_millis: now_millis(),
            stop_time_millis: None,
            total_time_millis: 0,
            snapshot_docs_total: info.snapshot_docs,
            snapshot_docs_recovered: 0,
            snapshot_bytes_estimate: info.snapshot_bytes_estimate,
            snapshot_bytes_recovered: 0,
            index_time_millis: 0,
            verify_index_time_millis: 0,
            translog_ops_recovered: 0,
            ops_recovered: 0,
            ops_total_on_start: info.snapshot_docs + info.translog_ops_on_start,
            translog_time_millis: 0,
            finalize_time_millis: 0,
            files: FileRecoveryStats::default(),
            snapshot: None,
        })
    }

    /// The finished restore of primary `shard` of `index` from a snapshot:
    /// the files it adopted, then `translog_ops` replayed above their commit
    /// (a snapshot staged without a flush).
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot_restore(index: &str, shard: u32, source: SnapshotRecoverySource, target_node: &str, files: FileRecoveryStats, translog_ops: u64, start_time_millis: u64, stop_time_millis: u64) -> Result<Self> {
        let total = stop_time_millis.saturating_sub(start_time_millis);
        Ok(Self {
            index: RecoveryText::try_new(index)?,
            shard,
            stage: RecoveryStage::Done,
            primary: true,
            source_node: None,
            target_node: Some(RecoveryText::try_new(target_node)?),
            start_time_millis,
            stop_time_millis: Some(stop_time_millis),
            total_time_millis: total,
            snapshot_docs_total: 0,
            snapshot_docs_recovered: 0,
            snapshot_bytes_estimate: 0,
            snapshot_bytes_recovered: 0,
            index_time_millis: files.time_millis,
            verify_index_time_millis: 0,
            translog_ops_recovered: translog_ops,
            ops_recovered: translog_ops,
            ops_total_on_start: translog_ops,
            translog_time_millis: total.saturating_sub(files.time_millis),
            finalize_time_millis: 0,
            files,
            snapshot: Some(source),
        })
    }

    /// The state of a file-based recovery during its file phase (stage
    /// `INDEX`, nothing replayed yet).
    pub fn file_phase(info: &RecoveryStartInfo, files: FileRecoveryStats, source: Option<RecoveryText>, target: Option<RecoveryText>) -> Result<Self> {
        let mut s = Self::new(info)?;
        s.start_time_millis = s.start_time_millis.saturating_sub(files.time_millis);
        s.stage = RecoveryStage::Index;
        s.total_time_millis = files.time_millis;
        s.index_time_millis = files.time_millis;
        s.files = files;
        s.source_node = source;
        s.target_node = target;
        Ok(s)
    }

    /// `ops total` as ES reports it: unknown (-1) until done.
    fn ops_total(&self) -> i64 {
        if self.stage == RecoveryStage::Done { self.ops_recovered as i64 } else { -1 }
    }

    /// One shard entry of `GET /{index}/_recovery` (`shards[]`).
    pub fn to_recovery_json(&self) -> Value {
        let node = |n: &Option<RecoveryText>| json!({"name": n.as_deref().unwrap_or_default(), "id": n.as_deref().unwrap_or_default()});
        let total = self.ops_total();
        // ES `SnapshotRecoverySource` x-content.
        let (ty, source) = match &self.snapshot {
            Some(s) => ("SNAPSHOT", json!({"repository": &*s.repository, "snapshot": &*s.snapshot, "version": s.version.as_deref(), "index": &*s.index, "restoreUUID": &*s.restore_uuid})),
            None => ("PEER", node(&self.source_node)),
        };
        let mut v = json!({
            "id": self.shard,
            "type": ty,
            "stage": self.stage.as_str(),
            "primary": self.primary,
            "start_time_in_millis": self.start_time_millis,
            "total_time_in_millis": self.total_time_millis,
            "source": source,
            "target": node(&self.target_node),
            "index": {
                "size": {"total_in_bytes": self.files.bytes_total, "reused_in_bytes": self.files.bytes_reused, "recovered_in_bytes": self.files.bytes_recovered,
                    "recovered_from_snapshot_in_bytes": 0, "percent": self.files.bytes_percent()},
                "files": {"total": self.files.files_total, "reused": self.files.files_reused, "recovered": self.files.files_recovered, "percent": self.files.files_percent()},
                "total_time_in_millis": self.index_time_millis,
                "source_throttle_time_in_millis": self.files.source_throttle_millis,
                "target_throttle_time_in_millis": self.files.target_throttle_millis
            },
            "translog": {
                "recovered": self.ops_recovered,
                "total": total,
                "percent": if total < 0 { pct(self.ops_recovered, self.ops_total_on_start.max(self.ops_recovered)) } else { "100.0%".into() },
                "total_on_start": self.ops_total_on_start,
                "total_time_in_millis": self.translog_time_millis
            },
            "verify_index": {"check_index_time_in_millis": self.verify_index_time_millis, "total_time_in_millis": self.verify_index_time_millis}
        });
        if let Some(stop) = self.stop_time_millis {
            v["stop_time_in_millis"] = json!(stop);
        }
        // `?detailed=true` (the REST layer drops them otherwise).
        if !self.files.details.is_empty() {
            v["index"]["files"]["details"] = self.files.details.iter().map(|f| json!({
                "name": f.name, "length_in_bytes": f.length, "reused": f.reused,
                "recovered_in_bytes": f.recovered, "recovered_from_snapshot_in_bytes": 0
            })).collect();
        }
        v
    }

    /// One `_cat/recovery` row (column name → value).
    pub fn cat_row(&self) -> BTreeMap<&'static str, String> {
        let s = |x: &Option<RecoveryText>| x.as_deref().unwrap_or("n/a").to_owned();
        let total = self.ops_total();
        let mut r = BTreeMap::new();
        r.insert("index", self.index.to_string());
        r.insert("shard", self.shard.to_string());
        r.insert("start_time_millis", self.start_time_millis.to_string());
        r.insert("stop_time_millis", self.stop_time_millis.map(|x| x.to_string()).unwrap_or_else(|| "0".into()));
        r.insert("time", format!("{}ms", self.total_time_millis));
        r.insert("type", if self.snapshot.is_some() { "snapshot" } else { "peer" }.into());
        r.insert("stage", self.stage.as_str().to_lowercase());
        r.insert("source_host", s(&self.source_node));
        r.insert("source_node", s(&self.source_node));
        r.insert("target_host", s(&self.target_node));
        r.insert("target_node", s(&self.target_node));
        r.insert("repository", self.snapshot.as_ref().map(|x| x.repository.to_string()).unwrap_or_else(|| "n/a".into()));
        r.insert("snapshot", self.snapshot.as_ref().map(|x| x.snapshot.to_string()).unwrap_or_else(|| "n/a".into()));
        let f = &self.files;
        r.insert("files", (f.files_total - f.files_reused.min(f.files_total)).to_string());
        r.insert("files_recovered", f.files_recovered.to_string());
        r.insert("files_percent", f.files_percent());
        r.insert("files_total", f.files_total.to_string());
        r.insert("bytes", (f.bytes_total - f.bytes_reused.min(f.bytes_total)).to_string());
        r.insert("bytes_recovered", f.bytes_recovered.to_string());
        r.insert("bytes_percent", f.bytes_percent());
        r.insert("bytes_total", f.bytes_total.to_string());
        r.insert("translog_ops", total.to_string());
        r.insert("translog_ops_recovered", self.ops_recovered.to_string());
        r.insert("translog_ops_percent", if total < 0 { pct(self.ops_recovered, self.ops_total_on_start.max(self.ops_recovered)) } else { "100.0%".into() });
        r
    }
}

/// Target side of one recovery.
pub struct RecoveryTarget {
    applier: Arc<ReplicaApplier>,
    info: RecoveryStartInfo,
    state: RecoveryState,
    started: Instant,
    stage_started: Instant,
}

impl RecoveryTarget {
    fn flat_settings(info: &RecoveryStartInfo) -> Result<FlatSettings> {
        let v: Value = serde_json::from_str(&info.settings_json)?;
        Ok(v.as_object().map(|m| m.iter().filter(|(k, _)| !PER_COPY_SETTINGS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default())
    }

    /// `PUT /{index}` body for `Node::create_index` (settings + mappings).
    pub fn create_index_body(info: &RecoveryStartInfo) -> Result<Value> {
        let flat = Self::flat_settings(info)?;
        let mapping: Value = serde_json::from_str(&info.mapping_json)?;
        Ok(json!({"settings": settings::unflatten(&flat), "mappings": mapping}))
    }

    /// Create the target's engine index directly at `path` (outside an
    /// engine `Node`; tests and phase-2 per-shard indices).
    pub fn create_index(path: &Path, info: &RecoveryStartInfo) -> Result<Arc<IndexService>> {
        let flat = Self::flat_settings(info)?;
        let (archived, active): (FlatSettings, FlatSettings) = flat.into_iter().partition(|(key, _)| key.starts_with("archived."));
        let mapping: Value = serde_json::from_str(&info.mapping_json)?;
        let service = Arc::new(IndexService::create(path, &info.index, active, &mapping, BTreeMap::new())?);
        if !archived.is_empty() { service.apply_settings_delta(archived)?; }
        Ok(service)
    }

    /// Start recovering into `svc` (normally empty, created from `info`).
    pub fn new(svc: Arc<IndexService>, info: RecoveryStartInfo) -> Result<RecoveryTarget> {
        let (live, _) = svc.doc_count_realtime();
        if live > 0 {
            tracing::warn!(index = %info.index, live, "recovery target is not empty; ops are applied idempotently");
        }
        let applier = Arc::new(ReplicaApplier::open(svc, info.shard)?);
        Self::with_applier(applier, info)
    }

    /// Reuse the registered copy's WAL owner. Opening another applier here
    /// races cluster-state term updates against WAL state persistence and
    /// creates independent checkpoint/term state for the same replica.
    pub fn with_applier(applier: Arc<ReplicaApplier>, info: RecoveryStartInfo) -> Result<RecoveryTarget> {
        // Adopt the recovery's term durably; a target that already saw a
        // newer term refuses the (stale) recovery.
        if let Err(e) = applier.bump_term(info.primary_term, super::checkpoint::UNASSIGNED_SEQ_NO) {
            return Err(match e {
                ReplicationError::StalePrimaryTerm { ours, .. } => ReplicationError::RecoveryAborted { recovery_term: info.primary_term, current_term: ours },
                e => e,
            });
        }
        let mut state = RecoveryState::new(&info)?;
        state.stage = RecoveryStage::Index;
        let now = Instant::now();
        Ok(RecoveryTarget { applier, info, state, started: now, stage_started: now })
    }

    /// The applier; register it for live replication requests as soon as
    /// the recovery starts.
    pub fn applier(&self) -> &Arc<ReplicaApplier> {
        &self.applier
    }

    pub fn info(&self) -> &RecoveryStartInfo {
        &self.info
    }

    pub fn set_nodes(&mut self, source: Option<RecoveryText>, target: Option<RecoveryText>) {
        self.state.source_node = source;
        self.state.target_node = target;
    }

    /// The file phase this recovery's operations follow (file-based
    /// recovery): reported in `_recovery`'s `index` section.
    pub fn set_file_stats(&mut self, files: FileRecoveryStats) {
        self.state.start_time_millis = now_millis().saturating_sub(files.time_millis);
        self.state.index_time_millis = files.time_millis;
        self.state.files = files;
    }

    pub fn state(&self) -> RecoveryState {
        let mut s = self.state.clone();
        if s.stop_time_millis.is_none() {
            s.total_time_millis = self.state.files.time_millis + self.started.elapsed().as_millis() as u64;
        }
        s
    }

    /// Recovery fencing: a recovery started under term `T` is aborted as
    /// soon as this copy has seen a newer primary term (the source may be a
    /// deposed primary). The caller wipes the target and starts over from
    /// the new primary.
    fn check_term(&self) -> Result<()> {
        let current = self.applier.primary_term();
        if current > self.info.primary_term {
            return Err(ReplicationError::RecoveryAborted { recovery_term: self.info.primary_term, current_term: current });
        }
        Ok(())
    }

    fn fenced<T>(&self, r: Result<T>) -> Result<T> {
        match r {
            Err(ReplicationError::StalePrimaryTerm { ours, .. }) => Err(ReplicationError::RecoveryAborted { recovery_term: self.info.primary_term, current_term: ours }),
            other => other,
        }
    }

    fn expect(&self, stage: RecoveryStage) -> Result<()> {
        self.check_term()?;
        if self.state.stage != stage {
            return Err(ReplicationError::IllegalState(format!("recovery of [{}] is in stage {} (expected {})", self.info.index, self.state.stage.as_str(), stage.as_str())));
        }
        Ok(())
    }

    fn enter(&mut self, stage: RecoveryStage) -> u64 {
        let ms = self.stage_started.elapsed().as_millis() as u64;
        self.stage_started = Instant::now();
        self.state.stage = stage;
        ms
    }

    /// Apply one snapshot batch. Returns the local checkpoint.
    pub fn apply_snapshot_batch(&mut self, ops: &[ReplicaOp]) -> Result<i64> {
        self.apply_snapshot_batch_mapped(ops, None, 0)
    }

    /// [`apply_snapshot_batch`](Self::apply_snapshot_batch) with the
    /// source's mapping as of the batch (and its cluster mapping version,
    /// 0 = unknown), installed first: the snapshot's documents can hold
    /// fields this copy's mapping does not have yet.
    pub fn apply_snapshot_batch_mapped(&mut self, ops: &[ReplicaOp], mapping: Option<&Value>, mapping_version: u64) -> Result<i64> {
        self.expect(RecoveryStage::Index)?;
        // Fenced on the recovery's term; the ops keep their own.
        let cp = self.fenced(self.applier.apply_under(ops, Some(self.info.primary_term), mapping, mapping_version))?;
        self.state.snapshot_docs_recovered += ops.len() as u64;
        self.state.snapshot_bytes_recovered += ops.iter().map(|o| o.estimated_bytes() as u64).sum::<u64>();
        self.state.ops_recovered += ops.len() as u64;
        Ok(cp)
    }

    /// The snapshot is complete: verify it and fast-forward the local
    /// checkpoint to `snapshot_max_seq_no` (see the module docs).
    pub fn snapshot_done(&mut self) -> Result<i64> {
        self.expect(RecoveryStage::Index)?;
        // A file phase ran before this target existed: its time counts.
        self.state.index_time_millis = self.state.files.time_millis + self.enter(RecoveryStage::VerifyIndex);
        if self.state.snapshot_docs_recovered != self.info.snapshot_docs {
            return Err(ReplicationError::IllegalState(format!(
                "recovery of [{}]: received {} snapshot docs, source announced {}",
                self.info.index, self.state.snapshot_docs_recovered, self.info.snapshot_docs
            )));
        }
        self.check_term()?;
        let cp = self.applier.fast_forward_checkpoint(self.info.snapshot_max_seq_no)?;
        self.state.verify_index_time_millis = self.enter(RecoveryStage::Translog);
        Ok(cp)
    }

    /// Apply one translog-phase batch (`mapping` = the primary's current
    /// mapping, if it changed). Returns the local checkpoint.
    pub fn apply_translog_batch(&mut self, ops: &[ReplicaOp], mapping: Option<&Value>) -> Result<i64> {
        self.apply_translog_batch_mapped(ops, mapping, 0)
    }

    /// [`apply_translog_batch`](Self::apply_translog_batch) with the cluster
    /// mapping version of `mapping` (0 = unknown).
    pub fn apply_translog_batch_mapped(&mut self, ops: &[ReplicaOp], mapping: Option<&Value>, mapping_version: u64) -> Result<i64> {
        self.expect(RecoveryStage::Translog)?;
        // The MSU safeguard (ES max_seq_no_of_updates_or_deletes): the engine
        // appends an index op above the MSU it learned from the primary as
        // the first copy of its id, without looking for a live one. A
        // recovery batch can carry an overwrite of a document the snapshot
        // installed before any live request taught this copy an MSU that
        // covers it, so raise the MSU over the batch first: its ops are
        // applied with the full lookup (ES ships the source's MSU with every
        // recovery batch for the same reason). Only raised: this is not the
        // primary's MSU, so it must not switch the fast path on for a copy
        // that never received one (live ops above the batch, updates among
        // them, would then be appended as first copies: two live copies).
        if let Some(max) = ops.iter().map(ReplicaOp::seq_no).max() {
            self.applier.copy().shard.raise_max_seq_no_of_updates(max);
        }
        // Fenced on the recovery's term; the ops keep their own.
        let cp = self.fenced(self.applier.apply_under(ops, Some(self.info.primary_term), mapping, mapping_version))?;
        self.state.translog_ops_recovered += ops.len() as u64;
        self.state.ops_recovered += ops.len() as u64;
        Ok(cp)
    }

    /// Finish: record the primary's global checkpoint, flush (so the WAL
    /// can be trimmed) and move to DONE. Fails while the local checkpoint
    /// is still below `global_checkpoint` (keep shipping ops, then retry).
    /// Returns the local checkpoint the primary should record before
    /// `mark_in_sync`.
    pub fn finalize(&mut self, global_checkpoint: i64) -> Result<i64> {
        // Flush + refresh below touch raw Shard state. Fence the physical
        // copy for the entire action, including an empty recovery tail.
        let copy = self.applier.copy();
        let _permit = copy.operation_permit()?;
        if self.state.stage != RecoveryStage::Finalize {
            self.expect(RecoveryStage::Translog)?;
            self.state.translog_time_millis = self.enter(RecoveryStage::Finalize);
        }
        self.check_term()?;
        self.applier.update_global_checkpoint(global_checkpoint)?;
        let cp = self.applier.local_checkpoint();
        if cp < global_checkpoint {
            return Err(ReplicationError::IllegalState(format!(
                "recovery of [{}]: local checkpoint {cp} is below the global checkpoint {global_checkpoint}",
                self.info.index
            )));
        }
        self.applier.flush()?;
        // ES `IndexShard.postRecovery`: refresh so everything recovered is
        // searchable as soon as the copy starts (a flush does not change
        // what searches see).
        self.applier.copy().shard.refresh()?;
        self.check_term()?;
        self.state.finalize_time_millis = self.enter(RecoveryStage::Done);
        self.state.stop_time_millis = Some(now_millis());
        self.state.total_time_millis = self.state.files.time_millis + self.started.elapsed().as_millis() as u64;
        Ok(self.applier.local_checkpoint())
    }

    pub fn into_applier(self) -> Arc<ReplicaApplier> {
        self.applier
    }
}
