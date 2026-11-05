//! The single shard of an index: a `rustic_core::Index` plus what ES wraps
//! around Lucene — the translog, sequence numbers, per-document versions,
//! optimistic concurrency, realtime get, flush/recovery and force merge.
//!
//! Visibility (ES near-real-time search): the core runs in NRT mode, so
//! searches see the shard as of the last [`Shard::refresh`]; documents and
//! deletes written since are visible only to realtime reads (`get`,
//! version / seq_no checks: [`Shard::lookup`] reads every leaf with the
//! latest deletes). A refresh is cheap: it publishes the memtables as they
//! are (no segment is written). Memtables are sealed into segments when
//! they grow past [`SEAL_THRESHOLD_DOCS`] or on flush, without changing
//! what searches see.
//!
//! Durability: ops are applied to the core under the shard write mutex,
//! then appended to the translog (still under that mutex, core lock
//! released) and fsynced outside both locks; the auto-id append-only path
//! appends before applying.
//! `flush` = seal + `Index::commit` + `shard_state.json` (committed seq_no)
//! + translog roll/trim. Opening replays translog ops above the committed
//! seq_no (done by the index service, which owns the mapping).

use crate::error::{EsError, Result};
use crate::mapping::{ID_FIELD, IGNORED_FIELD, PRIMARY_TERM_FIELD, ROUTING_FIELD, SEQ_NO_FIELD, SOURCE_FIELD, VERSION_FIELD};
use crate::translog::{Translog, TranslogOp};
use parking_lot::{Mutex, RwLock};
use rustic_core::types::{DocAddress, Document, Field as CoreField, FieldValue, IndexSettings as CoreSettings, Term};
use rustic_core::Index as CoreIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const SEAL_THRESHOLD_DOCS: u32 = 100_000;

/// Longest a replicated batch waits for seals to make room
/// ([`Shard::admit_replicated`], once per transport batch): a replica keeps
/// applying (its primary's round trip has a timeout), as ES replicas do.
const REPLICA_STALL: std::time::Duration = std::time::Duration::from_secs(1);
/// A refresh that would make more than this many memtable bytes
/// searchable seals them into segments first ([`Shard::seal_unpublished`]):
/// as a Lucene refresh, searches then read segments, never a huge buffer
/// (a search-idle copy's backlog).
const LARGE_PUBLISH_BYTES: u64 = 64 << 20;
/// Longest a large publish of a served copy waits for what it cannot seal
/// itself: a seal-job permit or the seal lock (about the default
/// `index.refresh_interval`). Past it the rest is published as memtables.
/// The refresh seals everything else itself ([`Shard::seal_unpublished`]),
/// so it never waits on the background seal queue: a 30 s wait there let
/// a node whose seals fell behind spend every refresh waiting out the
/// timeout while its backlog compounded.
const PUBLISH_SEAL_WAIT: std::time::Duration = std::time::Duration::from_secs(1);
/// Longest a refresh waits for background seal jobs already encoding
/// memtables it publishes (they make progress on their own workers; the
/// backlog cap keeps each small). Waiting for them, rather than publishing
/// their memtables, keeps searches on segments: a search over a large
/// published memtable builds its read structures first.
const IN_FLIGHT_SEAL_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
/// Longest the refresh a search on a search-idle copy the scheduler serves
/// runs waits for the core and for completed writes' hand-off, before the
/// search reads the last published view
/// ([`Shard::before_search`]). A large backlog's seal is the refresh's own
/// work and is not bounded by it. As a background worker's deferred refresh
/// ([`DEFERRED_REFRESH_WAIT`]): at 1 s, searches on copies whose core a
/// merge install or a write batch held kept taking just over a second
/// (the c28q wiki mixed load with a replica: idle phase 146 searches over
/// 1 s, max 1.8 s).
const IDLE_SEARCH_REFRESH_WAIT: std::time::Duration = std::time::Duration::from_millis(100);
/// A merge install, or a search-idle search's refresh, taking at least this
/// long is logged with its parts ("slow merge install", "slow search-idle
/// refresh").
const SLOW_INSTALL: std::time::Duration = std::time::Duration::from_millis(200);
/// Longest a search on a search-idle copy the scheduler serves waits for
/// the refresh another search is running (which may be sealing a backlog)
/// before reading the last published view ([`Shard::before_search`]).
const IDLE_SEARCH_OTHER_REFRESH_WAIT: std::time::Duration = std::time::Duration::from_secs(1);
/// Seal work a refresh may find waiting on a copy: the copy's unsealed
/// backlog is capped at what its measured seal throughput writes in this
/// long ([`backlog_limit`]), so the first refresh after a large ingest with
/// `refresh_interval: -1` seals seconds of work, not the whole memory share.
const REFRESH_SEAL_TARGET: std::time::Duration = std::time::Duration::from_secs(5);
/// The backlog cap never goes below this (segments of a reasonable size;
/// a slow first measurement must not starve ingest).
const BACKLOG_FLOOR_BYTES: u64 = 256 << 20;

/// The unsealed-bytes limit of a copy (the write stall; write-outs start at
/// half, the shard throttle at three quarters): the memory bound (twice its
/// share of the indexing buffer) or, when lower, [`REFRESH_SEAL_TARGET`] of
/// its measured seal throughput (`seal_rate`, bytes per second, `0`: not
/// measured yet), at least [`BACKLOG_FLOOR_BYTES`].
fn backlog_limit(memory: u64, seal_rate: u64) -> u64 {
    if seal_rate == 0 {
        return memory;
    }
    let target = (seal_rate as u128 * REFRESH_SEAL_TARGET.as_millis() / 1000).min(u64::MAX as u128) as u64;
    memory.min(target.max(BACKLOG_FLOOR_BYTES))
}

/// Graph preparation's 64-document task size is not a text-segment boundary.
/// Keep private writers for WAL failure isolation, but assemble larger chunks.
#[cfg(feature = "mutable-hnsw")]
const MUTABLE_TEXT_CHUNK_DOCS: usize = 4096;
const STATE_FILE: &str = "shard_state.json";
const TRANSLOG_DIR: &str = "translog";

// Hooks 3/4/5 (primary terms, replica apply, history, retention): see
// `hooks/shard_ext.rs` and `docs/ENGINE_HOOKS_IMPL.md`.
#[path = "hooks/shard_ext.rs"]
mod hooks_ext;
mod registry;
pub(crate) use hooks_ext::PreparedReplicaOp;
pub use hooks_ext::ChangesCursor;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ShardState {
    committed_seq_no: i64,
    #[serde(with = "translog_generation_format")]
    translog_generation: u64,
    core_generation: u64,
    /// Local checkpoint, processed seq_nos and tombstones at this commit
    /// (absent in shards committed before the hooks existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seqno: Option<crate::hooks::history::SeqNoState>,
    /// The commit seq_no the mutable graphs' checkpoint was saved at (graph
    /// saves are scheduled apart from text commits; `None`: none yet, or a
    /// shard without mutable vector fields).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    graph_checkpoint_seq_no: Option<i64>,
}

/// Version the required generation field so an older binary rejects upgraded
/// shards before opening/truncating an unfamiliar primary translog.
mod translog_generation_format {
    use serde::{Deserialize, Serialize, Deserializer, Serializer};
    #[derive(Serialize, Deserialize)]
    struct Current { format: u32, generation: u64 }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Read { Legacy(u64), Current(Current) }
    pub fn serialize<S: Serializer>(generation: &u64, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        Current { format: 2, generation: *generation }.serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<u64, D::Error> {
        match Read::deserialize(deserializer)? {
            Read::Legacy(n) => Ok(n),
            Read::Current(v) if v.format == 2 => Ok(v.generation),
            _ => Err(serde::de::Error::custom("unsupported primary translog format")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VersionEntry {
    pub version: u64,
    pub seq_no: i64,
    pub deleted: bool,
    /// The primary term of the op (`0`: not known here, e.g. replayed):
    /// `if_primary_term` compares with it, never with an older copy's.
    pub primary_term: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionType {
    Internal,
    External,
    ExternalGte,
}

impl VersionType {
    pub fn parse(s: &str) -> Result<VersionType> {
        match s {
            "internal" => Ok(VersionType::Internal),
            "external" => Ok(VersionType::External),
            "external_gte" => Ok(VersionType::ExternalGte),
            "force" => Err(EsError::illegal_argument("version type [force] may no longer be used")),
            other => Err(EsError::illegal_argument(format!("No version type match [{other}]"))),
        }
    }
}

/// Concurrency-control parameters of a write.
#[derive(Clone, Debug, Default)]
pub struct WriteCond {
    pub create: bool,
    pub if_seq_no: Option<i64>,
    pub if_primary_term: Option<u64>,
    pub version: Option<u64>,
    pub version_type: Option<VersionType>,
    /// Hook 12: this op is a retry of an op the coordinator already sent
    /// (ES's retry bit). With `coordinator_id` it makes the write idempotent.
    pub is_retry: bool,
    /// Hook 12: marker identifying one logical client op across retries.
    /// Recorded on every successful op that carries it; a later `is_retry`
    /// op for the same `_id` with the same marker returns the original result.
    pub coordinator_id: Option<String>,
    /// The `_id` was generated by the coordinator for this request and this
    /// is the op's first, in-process attempt: no earlier copy of the
    /// document can exist, so the write skips the version lookup (ES's
    /// `canOptimizeAddDocument` for auto-generated ids).
    pub append_only: bool,
}

impl WriteCond {
    pub fn validate(&self, _is_delete: bool) -> Result<()> {
        match (self.if_seq_no, self.if_primary_term) {
            (Some(_), None) => return Err(EsError::action_validation(&["ifSeqNo is set, but primary term is [0]"])),
            (None, Some(t)) => return Err(EsError::action_validation(&[&format!("ifSeqNo is unassigned, but primary term is [{t}]")])),
            _ => {}
        }
        let vt = self.version_type.unwrap_or(VersionType::Internal);
        if self.version.is_some() && vt == VersionType::Internal {
            return Err(EsError::action_validation(&[
                "internal versioning can not be used for optimistic concurrency control. Please use `if_seq_no` and `if_primary_term` instead",
            ]));
        }
        if self.version.is_none() && vt != VersionType::Internal {
            return Err(EsError::action_validation(&["an external version must be provided when using external versioning"]));
        }
        if self.create && vt != VersionType::Internal {
            return Err(EsError::action_validation(&["create operations only support internal versioning. use index instead"]));
        }
        if self.if_seq_no.is_some() && self.version.is_some() {
            return Err(EsError::action_validation(&["compare and write operations can not use versioning"]));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WriteResult {
    pub id: String,
    pub version: u64,
    pub seq_no: i64,
    /// The primary term the op was assigned under (the document's own
    /// term for a write; the shard's current term for a delete / noop).
    pub primary_term: u64,
    /// `created` / `updated` / `deleted` / `not_found`.
    pub result: &'static str,
}

#[derive(Clone, Debug)]
pub struct GetResult {
    pub source: Option<String>,
    pub version: u64,
    pub seq_no: i64,
    /// The document's stored `_primary_term`.
    pub primary_term: u64,
    pub routing: Option<String>,
    pub doc: Document,
}

/// One replicated op for [`Shard::replay_staged`], its document parsed.
pub enum StagedReplay {
    Index { id: String, source: String, doc: Document, ignored: Vec<String>, routing: Option<String>, seq_no: i64, version: u64 },
    Delete { id: String, seq_no: i64, version: u64 },
    /// Known stale before parsing: only marked processed.
    Skip { seq_no: i64 },
}

/// Metadata for documents already durably recorded by the replication layer.
pub struct ReplayIndex<'a> {
    pub id: &'a str,
    pub source: &'a str,
    pub routing: Option<&'a str>,
    pub seq_no: i64,
    pub version: u64,
}

/// One write in a batch.
pub enum WriteOp {
    Index {
        id: String,
        source: String,
        doc: Document,
        ignored: Vec<String>,
        routing: Option<String>,
        cond: WriteCond,
        /// Auto-generated id: skip the existence lookup.
        auto_id: bool,
    },
    Delete {
        id: String,
        cond: WriteCond,
    },
}

/// Per-shard operation statistics (`_stats`). Times are accumulated in
/// nanoseconds and reported in millis; `*_current` are live gauges.
#[derive(Default)]
pub struct Counters {
    pub index_total: AtomicU64,
    pub index_time_nanos: AtomicU64,
    pub index_current: AtomicU64,
    pub index_failed: AtomicU64,
    pub delete_total: AtomicU64,
    pub delete_time_nanos: AtomicU64,
    pub delete_current: AtomicU64,
    pub noop_update_total: AtomicU64,
    pub get_total: AtomicU64,
    pub get_time_nanos: AtomicU64,
    pub get_exists_total: AtomicU64,
    pub get_exists_time_nanos: AtomicU64,
    pub get_missing_total: AtomicU64,
    pub get_missing_time_nanos: AtomicU64,
    pub get_current: AtomicU64,
    /// Query phases run (ES `search.query_total`).
    pub search_total: AtomicU64,
    pub query_time_nanos: AtomicU64,
    pub query_current: AtomicU64,
    pub fetch_total: AtomicU64,
    pub fetch_time_nanos: AtomicU64,
    pub fetch_current: AtomicU64,
    pub refresh_total: AtomicU64,
    pub refresh_time_nanos: AtomicU64,
    pub flush_total: AtomicU64,
    pub flush_time_nanos: AtomicU64,
    pub merge_total: AtomicU64,
    pub merge_time_nanos: AtomicU64,
    pub merge_docs: AtomicU64,
    pub merge_size_bytes: AtomicU64,
    /// Merges running now (background merges), and their docs.
    pub merge_current: AtomicU64,
    pub merge_current_docs: AtomicU64,
    /// ES `indexing.throttle_time_in_millis`: time writes waited for the
    /// indexing throttle ([`Shard::merge_throttle`]), and whether it is on.
    pub index_throttle_nanos: AtomicU64,
    pub index_throttled: AtomicU64,
}

impl Counters {
    /// The field-wise sum of several shards' counters (index-level stats).
    pub fn sum<'a>(all: impl IntoIterator<Item = &'a Counters>) -> Counters {
        let out = Counters::default();
        macro_rules! add {
            ($c:expr, $($f:ident),*) => { $( out.$f.fetch_add($c.$f.load(Ordering::Relaxed), Ordering::Relaxed); )* };
        }
        for c in all {
            add!(c, index_total, index_time_nanos, index_current, index_failed, delete_total, delete_time_nanos, delete_current, noop_update_total,
                get_total, get_time_nanos, get_exists_total, get_exists_time_nanos, get_missing_total, get_missing_time_nanos, get_current,
                search_total, query_time_nanos, query_current, fetch_total, fetch_time_nanos, fetch_current, refresh_total, refresh_time_nanos,
                flush_total, flush_time_nanos, merge_total, merge_time_nanos, merge_docs, merge_size_bytes, merge_current, merge_current_docs,
                index_throttle_nanos, index_throttled);
        }
        out
    }

    pub fn add_time(c: &AtomicU64, d: std::time::Duration) {
        c.fetch_add(d.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    }

    /// A nanosecond accumulator in millis.
    pub fn millis(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed) / 1_000_000
    }

    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }

    /// Record one search over this shard: total wall time, of which `fetch`
    /// was spent in the fetch phase (`fetched` = whether it fetched hits).
    pub fn record_search(&self, total: std::time::Duration, fetch: std::time::Duration, fetched: bool) {
        self.search_total.fetch_add(1, Ordering::Relaxed);
        Self::add_time(&self.query_time_nanos, total.saturating_sub(fetch));
        if fetched {
            self.fetch_total.fetch_add(1, Ordering::Relaxed);
            Self::add_time(&self.fetch_time_nanos, fetch);
        }
    }

    /// A query_then_fetch fetch phase: one fetch, no query.
    pub fn record_fetch(&self, fetch: std::time::Duration) {
        self.fetch_total.fetch_add(1, Ordering::Relaxed);
        Self::add_time(&self.fetch_time_nanos, fetch);
    }
}

/// Increments a set of `*_current` gauges for its lifetime.
pub struct CurrentGuard<'a>(Vec<(&'a AtomicU64, u64)>);

impl<'a> CurrentGuard<'a> {
    pub fn new(gauges: Vec<(&'a AtomicU64, u64)>) -> CurrentGuard<'a> {
        for (g, n) in &gauges {
            g.fetch_add(*n, Ordering::Relaxed);
        }
        CurrentGuard(gauges)
    }
}

impl Drop for CurrentGuard<'_> {
    fn drop(&mut self) {
        for (g, n) in &self.0 {
            g.fetch_sub(*n, Ordering::Relaxed);
        }
    }
}

/// The translog behind its own lock (ES appends under the translog's lock,
/// not a shard-wide one): writers append after leaving the write state,
/// holding only their ids' locks ([`KeyedLocks`]). Every method
/// takes the lock for its own duration.
#[derive(Clone)]
struct SharedTranslog(Arc<Mutex<Translog>>);

impl SharedTranslog {
    fn new(translog: Translog) -> Self { SharedTranslog(Arc::new(Mutex::new(translog))) }
    fn generation(&self) -> u64 { self.0.lock().generation() }
    fn fail_next_appends(&self, count: usize) { self.0.lock().fail_next_appends(count) }
    fn append_many(&self, ops: &[TranslogOp]) -> Result<()> { self.0.lock().append_many(ops) }
    /// Append and take the sync point covering it, under one hold.
    fn append_many_synced(&self, ops: &[TranslogOp]) -> Result<(Arc<std::fs::File>, u64)> {
        // Serialize and compress before the lock: only the write is serialized.
        let encoded = Translog::encode_batch(ops)?;
        let mut translog = self.0.lock();
        translog.append_encoded(&encoded)?;
        Ok(translog.sync_point())
    }
    fn sync_point(&self) -> (Arc<std::fs::File>, u64) { self.0.lock().sync_point() }
    fn fsync(&self) -> Result<()> { self.0.lock().fsync() }
    fn roll(&self) -> Result<u64> { self.0.lock().roll() }
    fn fail(&self, reason: &str) { self.0.lock().fail(reason) }
    fn failed(&self) -> Option<String> { self.0.lock().failed().map(str::to_string) }
    #[cfg(test)]
    fn fail_next_fsync(&self) { self.0.lock().fail_next_fsync() }
    fn trim_to(&self, generation: u64) -> Result<()> { self.0.lock().trim_to(generation) }
    fn size_bytes(&self) -> u64 { self.0.lock().size_bytes() }
}

struct WriteState {
    translog: SharedTranslog,
    /// Highest seq_no assigned (-1 = none).
    seq_no: i64,
    committed_seq_no: i64,
    /// Versions of docs written since the last flush (incl. deletes).
    versions: registry::Registry<VersionEntry>,
    version_inflight: usize,
    core_generation: u64,
    /// Seq_no tracker, tombstones, history sidecar (hooks).
    hist: crate::hooks::history::ShardHistory,
    /// Staged bulks ([`Shard::apply_staged`]) sequenced but not installed
    /// yet, by first seq_no: their version-map entries are the latest state
    /// of their ids until installation, so pruning keeps them.
    staged: std::collections::BTreeMap<i64, (usize, u64, rustic_core::admission::Reservation)>,
    /// Pending documents by id (in pooled memtables, not checked in yet):
    /// the latest op of the id while it is one.
    /// Keyed by document id: ahash (keyed per map, so still resistant to
    /// crafted ids), several times faster than SipHash on ids.
    pending_docs: HashMap<String, Arc<PendingDoc>, ahash::RandomState>,
    pending_inflight: usize,
    pending_memory: Option<rustic_core::admission::Reservation>,
    /// Bumped by every version-map prune: a live-copy lookup made before
    /// one may predate an op whose entry it dropped ([`Shard::apply_staged`]).
    prunes: u64,
}

impl WriteState {
    fn prepare_version_key(&mut self, id: &str) -> Result<registry::Key> {
        let count = self.versions.len().checked_add(self.version_inflight).and_then(|n| n.checked_add(1))
            .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "version map capacity overflow"))?;
        self.versions.reserve_capacity(count)?;
        registry::Key::prepare(id)
    }
    /// An op batch sequenced from `first` on whose documents are not all in
    /// the core yet: it holds the commit point below `first` (a capture
    /// does not contain it) and its translog generation (never trimmed
    /// while claimed), and keeps its version-map entries.
    fn claim(&mut self, first: i64, memory: rustic_core::admission::Reservation) {
        let generation = self.translog.generation();
        let entry = self.staged.entry(first).or_insert((0, generation, memory));
        entry.0 += 1;
        entry.1 = entry.1.min(generation);
    }

    /// seq_nos assigned to ops that never reached the translog (its append
    /// failed): recorded as no-ops in the history sidecar and marked
    /// processed, so no hole stalls the local / global checkpoint or
    /// retention (ES fills such gaps with no-ops too).
    /// With `durable` (gap fills: the no-ops are history other copies
    /// and a restart rely on) the records are fsynced and the seq_nos are
    /// marked processed only once they are; a failure is returned (the
    /// caller fails the engine). Without it (the seq_nos of an append that
    /// failed: the engine is failed already) a failure is only logged.
    fn fill_noops(&mut self, seq_nos: &[i64], term: u64, reason: &str, durable: bool) -> Result<()> {
        if seq_nos.is_empty() { return Ok(()); }
        let generation = self.translog.generation();
        let records: Vec<crate::hooks::history::SideRec> = seq_nos.iter()
            .map(|&seq_no| crate::hooks::history::SideRec { seq_no, term, noop: Some(reason.to_string()) }).collect();
        if let Err(error) = self.hist.side.append(generation, &records, durable) {
            if durable {
                return Err(error);
            }
            tracing::error!("recording no-ops for failed translog appends failed: {error}");
        }
        for &seq_no in seq_nos { self.hist.mark(seq_no); }
        Ok(())
    }

    fn release_claim(&mut self, first: i64) {
        if let Some(entry) = self.staged.get_mut(&first) {
            entry.0 -= 1;
            if entry.0 == 0 { self.staged.remove(&first); }
        }
    }

    /// What a capture taken now covers: the commit's local checkpoint (the
    /// highest seq_no below every claimed batch, i.e. the lowest seq_no not
    /// in the capture minus one) and the oldest translog generation a
    /// claimed batch still needs.
    fn capture_floor(&self) -> (i64, Option<u64>) {
        match self.staged.keys().next() {
            Some(first) => (first - 1, self.staged.values().map(|entry| entry.1).min()),
            None => (self.seq_no, None),
        }
    }

    /// Drop live version-map entries (they mirror what the id lookup finds
    /// in the core), keeping delete tombstones and every entry a staged
    /// bulk still relies on.
    /// `unresolved`: ids with a buffered older-copy delete, kept (the version
    /// map answers for them until their older copies are gone).
    fn prune_versions(&mut self, unresolved: &registry::Deletes) {
        let keep_from = self.staged.keys().next().copied().unwrap_or(i64::MAX);
        self.versions.retain(|id, e| e.deleted || e.seq_no >= keep_from || unresolved.contains_key(id.as_str()));
        self.prunes += 1;
    }

    /// [`WriteState::prune_versions`], handing the dropped entries back to
    /// be freed after the locks.
    fn prune_versions_deferred(&mut self, unresolved: &registry::Deletes) -> Vec<registry::Removed<VersionEntry>> {
        let keep_from = self.staged.keys().next().copied().unwrap_or(i64::MAX);
        self.prunes += 1;
        self.versions.extract(|id, e| e.deleted || e.seq_no >= keep_from || unresolved.contains_key(id.as_str()))
    }
}

impl Drop for Shard {
    fn drop(&mut self) {
        // No `refresh=wait_for` request waits on a dropped copy.
        self.listeners.close();
        rustic_core::merge::note_pending_merges(self.bg.merge_backlog.swap(0, Ordering::AcqRel), 0);
        let old = self.buffered.swap(0, Ordering::Relaxed);
        crate::memory::IndexingBuffer::global().update(old, 0);
    }
}

/// Translog and seq_no figures reported by stats APIs ([`Shard::write_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteStats {
    pub translog_bytes: u64,
    pub uncommitted_ops: i64,
    pub max_seq_no: i64,
    pub local_checkpoint: i64,
}

pub struct Shard {
    /// Owned core reservations observed for this shard. Explicitly partial: worker
    /// propagation and non-core categories must be covered before placement can
    /// treat it as a complete recovery cost.
    pub admitted_owner: Arc<rustic_core::admission::Owner>,
    path: PathBuf,
    pub core: RwLock<CoreIndex>,
    /// The core's published reader views (shared with the core, see
    /// [`rustic_core::reader_view`]): searches read [`Shard::reader_view`]
    /// without taking `core` at all.
    views: Arc<rustic_core::ViewSlot>,
    /// A copy of the core schema as last applied (replaced under the core
    /// write lock): the search path decides whether the schema needs a sync
    /// without taking the core lock.
    core_schema: Mutex<Arc<Vec<CoreField>>>,
    /// Stored-field compression of new memtables (`index.codec`), read
    /// lock-free by the write paths that fill pooled / private memtables;
    /// the core index holds the same mode for its own writers. 0 = default,
    /// 1 = `best_compression` (see [`Shard::set_stored_compression`]).
    stored_compression: std::sync::atomic::AtomicU8,
    write: Mutex<WriteState>,
    /// Translog append mark known to be fsync'd (group commit, see
    /// [`Shard::group_sync`]).
    synced: Mutex<u64>,
    /// Newest translog sync point (file, mark), published after each append
    /// so the syncer never waits on the write lock.
    tail: Mutex<Option<(Arc<std::fs::File>, u64)>>,
    pub counters: Counters,
    last_flush_started: Mutex<std::time::Instant>,
    /// Mutable-graph save schedule ([`Shard::graph_save_due`]).
    graph_saves: Mutex<GraphSaves>,
    ext: hooks_ext::ShardExt,
    /// Refresh scheduling state (see [`crate::refresh`]).
    pub(crate) sched: crate::refresh::RefreshState,
    /// `refresh=wait_for` listeners ([`Shard::wait_for_refresh`]).
    pub(crate) listeners: RefreshListeners,
    /// Idle pooled memtables ([`Shard::apply_staged`]).
    pool: Mutex<std::collections::LinkedList<PoolWriter>>,
    /// Pooled memtables handed off at a freeze, waiting for their check-in.
    frozen: Mutex<std::collections::LinkedList<PoolWriter>>,
    /// Serializes the check-ins that take memtables out of `pool` /
    /// `frozen` ([`Shard::checkin_completed_writes`], [`Shard::checkin_pool`],
    /// a flush's capture). Between taking them and checking them into the
    /// core they are held by that one caller only; without this a blocking
    /// refresh starting meanwhile found both lists empty and returned
    /// without writes acknowledged before it (a scheduled / deferred
    /// refresh or an idle seal held them). Blocking callers wait for it
    /// (its holders only wait for bounded freezes and the write state / the
    /// core, whose holders never take it); non-blocking ones try it within
    /// their own wait. Lock order: after `merge_lock`, before `pool`,
    /// `holders`, `frozen`, the write state and the core (see
    /// [`Background`]). Never held across encoding, I/O or a commit.
    checkin_lock: Mutex<()>,
    /// Freeze generation ([`Shard::freeze_pool`]).
    freeze_gen: AtomicU64,
    /// Requests holding a pooled memtable: token -> freeze generation seen.
    holders: Mutex<std::collections::BTreeMap<u64, (u64, rustic_core::admission::Reservation)>>,
    holders_changed: parking_lot::Condvar,
    next_holder: AtomicU64,
    /// Test seam ([`Shard::pause_next_staged_write`]).
    pause_armed: std::sync::atomic::AtomicBool,
    pause_hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Test seam ([`Shard::pause_next_flush_commit`]).
    flush_pause: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Test seam ([`Shard::panic_next_inversion`]).
    inversion_panic: std::sync::atomic::AtomicBool,
    /// Test seam ([`Shard::fail_next_translog_syncs`]): fsyncs still to fail.
    fail_syncs: std::sync::atomic::AtomicUsize,
    /// [`Shard::write_stats`] as last read, or as published by a long holder
    /// of the write state (force merge, flush) when it took it.
    write_stats: Mutex<WriteStats>,
    /// This shard's last reported bytes in the memtables being written (its
    /// share of the node's indexing buffer, [`crate::memory::IndexingBuffer`]).
    buffered: AtomicU64,
    /// Puts back what the stored `_source` leaves to other storage (the
    /// vectors of an index that excludes them, see
    /// [`crate::vector_source::rehydrate_stored`]); set by the index.
    patch: RwLock<Option<SourcePatch>>,
    /// Background flush / merge state (see [`Shard::run_background`]).
    pub(crate) bg: Background,
    /// Replication retention of a cluster primary
    /// ([`Shard::set_replication_retention`]).
    retention: Mutex<Option<Arc<dyn Fn() -> Option<i64> + Send + Sync>>>,
    /// Buffered older-copy deletes: id -> seq_no of its newest checked-in
    /// copy ([`Shard::buffer_delete`]). Leaf lock (ws -> core -> this).
    unresolved: Mutex<registry::Deletes>,
    /// During a sorted refresh: the buffered deletes as of its seal point,
    /// i.e. those whose new copies it seals and publishes. Only these are
    /// resolved until it publishes (a delete whose new copy was checked in
    /// later would become visible without it). Leaf lock.
    sorted_window: Mutex<Option<rustic_core::admission::Allocation<registry::Deletes>>>,
    /// The translog's tragic error (ES `Translog.tragedy`): set by the
    /// first append or fsync failure; the engine then fails every op fast
    /// and the shard copy is to be failed ([`Shard::take_tragic_report`]).
    tragic: Mutex<Option<String>>,
    /// Per-id locks of staged writes ([`KeyedLocks`]).
    id_locks: KeyedLocks,
    /// Test seam ([`Shard::fail_next_older_deletes`]).
    fail_older_deletes: std::sync::atomic::AtomicUsize,
    /// Test seam ([`Shard::probe_next_sequencing`]): called while a batch
    /// holds its id locks, between its lookups and its version checks.
    sequencing_probe: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    /// ES `max_seq_no_of_updates_or_deletes` (MSU): the highest seq_no of
    /// an op that updated or deleted an id this copy (or, for a replica, its
    /// primary) knew. An op above it is the first copy of its id.
    msu: std::sync::atomic::AtomicI64,
    /// A primary supplied its MSU (replicas): only then does replay take
    /// the append path for ops above it.
    msu_from_primary: std::sync::atomic::AtomicBool,
    tragic_reported: std::sync::atomic::AtomicBool,
    /// Append-only batches indexing into a checked-out memtable hold this
    /// shared ([`Shard::apply_append_only`]); a commit (flush, force merge)
    /// holds it exclusively, so every op below the commit point is in the
    /// core it commits. Lock order: `bg` locks -> `batches` -> write state -> core.
    batches: RwLock<()>,
    /// Bumped by everything that can change the shard's files (writes,
    /// seals, merges, commits): [`Shard::store_size`] reuses its last walk
    /// of the shard directory while it is unchanged (at most
    /// [`STORE_STATS_MAX_AGE`], ES's `index.store.stats_refresh_interval`).
    files_gen: AtomicU64,
    store_cache: Mutex<Option<(u64, std::time::Instant, u64)>>,
    /// [`Shard::sampled_source_bytes`] of a reader view generation.
    source_sample: Mutex<Option<((u64, usize), Option<u64>)>>,
}

/// ES refresh listeners (`refresh=wait_for`): every refresh takes a ticket
/// when it captures what becomes visible (under the core write lock) and
/// completes it once its view is installed; a listener waits for the first
/// ticket taken after its write was installed.
#[derive(Default)]
pub(crate) struct RefreshListeners {
    started: AtomicU64,
    completed: Mutex<u64>,
    done: parking_lot::Condvar,
    waiting: std::sync::atomic::AtomicUsize,
    /// Registered listeners (ES `RefreshListeners`): called by the refresh
    /// that completes their ticket, not waited for by any thread. Locked
    /// after `completed`.
    parked: Mutex<Vec<ParkedListener>>,
    /// The copy closed ([`RefreshListeners::close`]): listeners fire at once.
    closed: std::sync::atomic::AtomicBool,
}

/// A `refresh=wait_for` listener: called with whether a refresh had to be
/// forced for it (`forced_refresh`).
pub type RefreshCallback = Box<dyn FnOnce(Result<bool>) + Send>;

struct ParkedListener {
    target: u64,
    deadline: std::time::Instant,
    /// Its wait expired: the refresh that completes it was forced.
    forced: bool,
    callback: RefreshCallback,
}

impl RefreshListeners {
    /// A refresh captures visibility now (call under the core write lock).
    fn begin(&self) -> u64 {
        self.started.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The refresh of `ticket` installed its view.
    fn complete(&self, ticket: u64) {
        let due: Vec<ParkedListener> = {
            let mut completed = self.completed.lock();
            if ticket > *completed { *completed = ticket; }
            self.done.notify_all();
            let mut parked = self.parked.lock();
            let (due, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut *parked).into_iter().partition(|l| l.target <= *completed);
            *parked = rest;
            due
        };
        if !due.is_empty() {
            self.waiting.fetch_sub(due.len(), Ordering::AcqRel);
        }
        // Outside the locks: a callback sends the response.
        for listener in due {
            (listener.callback)(Ok(listener.forced));
        }
    }

    /// Mark the listeners whose wait expired (they need a forced refresh);
    /// whether any did.
    fn expire(&self, now: std::time::Instant) -> bool {
        let mut any = false;
        for listener in self.parked.lock().iter_mut() {
            if listener.deadline <= now {
                listener.forced = true;
                any = true;
            }
        }
        any
    }

    /// Whether a listener is waiting (a search-idle copy still refreshes).
    pub(crate) fn pending(&self) -> bool {
        self.waiting.load(Ordering::Acquire) > 0
    }

    /// The copy closes (ES `RefreshListeners.close`): every waiting
    /// listener fires now, as not forced; later ones fire at registration.
    /// No request waits on a copy that will never refresh again.
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let all = {
            let _completed = self.completed.lock();
            std::mem::take(&mut *self.parked.lock())
        };
        if !all.is_empty() {
            self.waiting.fetch_sub(all.len(), Ordering::AcqRel);
        }
        for listener in all {
            (listener.callback)(Ok(false));
        }
    }
}

/// Vespa-style graph save scheduling: the mutable graphs are checkpointed
/// with a commit only when catching them up at open would be expensive
/// (ops since the last graph checkpoint, or its age); other commits keep
/// referencing the previous graph checkpoint.
struct GraphSaves {
    /// Commit seq_no of the last graph checkpoint (`None`: none yet).
    seq_no: Option<i64>,
    saved_at: std::time::Instant,
    /// Fewest ops between graph checkpoints (`RUSTIC_GRAPH_SAVE_OPS`, 50,000).
    max_ops: i64,
    /// Ops between graph checkpoints grow with the graph: a save is also
    /// held back until the ops since the last one reach this fraction of
    /// that save's seq_no (`RUSTIC_GRAPH_SAVE_GROWTH`, 0.05: at 1.3M
    /// vectors a shard copy reopens with at most 65k ops to re-insert, about
    /// 30 s, where 0.5 left up to 650k). A checkpoint
    /// writes the whole graph's topology (HNSW back-links touch nearly
    /// every node between saves), so a geometric schedule keeps the total
    /// checkpoint I/O linear in the ops (amortised `1/growth` graph bytes
    /// per op), and bounds the open-time catch-up to that fraction.
    growth: f64,
    /// Longest interval between graph checkpoints
    /// (`RUSTIC_GRAPH_SAVE_INTERVAL_SECS`, 10 minutes).
    max_age: std::time::Duration,
    /// Consecutive due graph saves skipped because graph ingestion was in
    /// flight (see [`GraphSaves::overdue`]).
    skipped: u32,
    /// The next flush saves the graphs whatever the schedule says (a file
    /// recovery source bounds the target's catch-up, see
    /// [`Shard::request_graph_save`]).
    forced: bool,
}

/// Due graph saves a flush skips (graph ingestion in flight) before the
/// next one quiesces graph ingestion for its capture: under sustained
/// vector ingestion something is always in flight, and the graph would
/// never be saved (the reopen catch-up and the pinned rows grow).
const GRAPH_SAVE_MAX_SKIPS: u32 = 3;

impl GraphSaves {
    fn new(seq_no: Option<i64>) -> Self {
        let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<i64>().ok());
        GraphSaves { seq_no, saved_at: std::time::Instant::now(),
            max_ops: env("RUSTIC_GRAPH_SAVE_OPS").unwrap_or(50_000).max(0),
            growth: std::env::var("RUSTIC_GRAPH_SAVE_GROWTH").ok().and_then(|v| v.parse::<f64>().ok()).filter(|g| g.is_finite() && *g >= 0.0).unwrap_or(0.05),
            max_age: std::time::Duration::from_secs(env("RUSTIC_GRAPH_SAVE_INTERVAL_SECS").unwrap_or(600).max(0) as u64), skipped: 0, forced: false }
    }

    /// The save is due and was skipped too often: the next flush quiesces
    /// graph ingestion briefly (as for a field's first save).
    fn overdue(&self) -> bool {
        self.skipped >= GRAPH_SAVE_MAX_SKIPS
    }

    fn due(&self, commit_seq: i64) -> bool {
        if self.forced {
            return true;
        }
        match self.seq_no {
            None => true,
            Some(saved) => {
                let since = commit_seq.saturating_sub(saved);
                let geometric = (saved.max(0) as f64 * self.growth).min(i64::MAX as f64) as i64;
                since >= self.max_ops.max(geometric)
                    // Idle shards still checkpoint what changed, eventually.
                    || (since > 0 && self.saved_at.elapsed() >= self.max_age)
            }
        }
    }
}

/// A pooled per-thread memtable ([`Shard::apply_staged`]): taken by one
/// request at a time, invisible (realtime GET answers from the version
/// map's pending entries) until checked into the index.
struct PoolCheckin { value: std::sync::atomic::AtomicBool, _memory: Option<rustic_core::admission::Reservation> }
impl std::ops::Deref for PoolCheckin {type Target=std::sync::atomic::AtomicBool;fn deref(&self)->&Self::Target {&self.value}}

struct PoolWriter {
    writer: rustic_core::segment::SegmentWriter,
    /// Its documents in order: id, seq_no, local doc id.
    docs: Vec<PoolDoc>,
    /// The version-map / commit-point claims (`WriteState::staged`) of the
    /// requests whose documents it holds: every memtable a request wrote
    /// into holds a reference, and the claim is released when the last of
    /// them is checked in ([`Shard::checkin_pool_writer`]).
    registered: Vec<Arc<ClaimRef>>,
    docs_memory: Option<rustic_core::admission::Reservation>,
    registered_memory: Option<rustic_core::admission::Reservation>,
    /// Set (under the write state) when it is checked in.
    checked_in: Arc<PoolCheckin>,
    /// Its memory on node-wide accounting while outside the index (idle,
    /// frozen, or filled by a request); released when it is checked in
    /// (the indexing buffer counts it then) or dropped, panics included.
    charge: rustic_core::accounting::Charge,
    /// `docs.len()` when the charge was last brought up to date.
    accounted_docs: usize,
    queue_memory: Option<rustic_core::admission::Reservation>,
}

/// Documents between two refreshes of a held pooled memtable's charge.
const POOL_ACCOUNT_DOCS: usize = 64;

impl PoolWriter {
    fn queue_bytes()->usize {std::mem::size_of::<Self>()+2*std::mem::size_of::<usize>()+Self::flag_bytes()}
    fn flag_bytes()->usize {std::mem::size_of::<PoolCheckin>()+2*std::mem::size_of::<usize>()}

    fn fresh() -> PoolWriter { Self::fresh_with_credit(None) }
    fn fresh_with_credit(mut memory:Option<rustic_core::admission::Reservation>)->PoolWriter {
        let flag_memory=memory.as_mut().map(|credit|credit.split(Self::flag_bytes() as u64).expect("pre-WAL checkin flag credit"));
        PoolWriter { writer: rustic_core::segment::SegmentWriter::new(0), docs: Vec::new(), registered: Vec::new(), docs_memory: None, registered_memory: None,
            checked_in: Arc::new(PoolCheckin{value:std::sync::atomic::AtomicBool::new(false),_memory:flag_memory}),
            charge: rustic_core::accounting::Charge::new(rustic_core::accounting::Category::PooledMemtables, 0), accounted_docs: 0, queue_memory: memory }
    }

    fn prepare_registry(&mut self,documents:usize)->Result<()> {
        fn prepare<T>(values:&mut Vec<T>, memory:&mut Option<rustic_core::admission::Reservation>, required:usize,label:&str)->Result<()> {
            if required<=values.capacity(){return Ok(())}
            let bytes=required.checked_mul(std::mem::size_of::<T>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","pool capacity overflow"))?;
            let next=rustic_core::admission::shared().reserve(bytes as u64,label)?;
            values.try_reserve_exact(required-values.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
            *memory=Some(next);Ok(())
        }
        let required=self.docs.len().checked_add(documents).ok_or_else(||EsError::new(429,"circuit_breaking_exception","pool document capacity overflow"))?;
        prepare(&mut self.docs,&mut self.docs_memory,required,"pooled document slots")?;
        let required=self.registered.len().checked_add(1).ok_or_else(||EsError::new(429,"circuit_breaking_exception","pool claim capacity overflow"))?;
        prepare(&mut self.registered,&mut self.registered_memory,required,"pooled claim slots")
    }
    fn admit_claim_slot(&mut self,holding:&mut Holding) {
        if self.registered.len()<self.registered.capacity(){return}
        if holding.registry_credits.is_none(){return} // test-only legacy holder
        let bytes=std::mem::size_of::<Arc<ClaimRef>>() as u64;
        let next=holding.handoff_credit(bytes,"pooled claim slot handoff");
        self.registered.reserve_exact(1);
        // Without credit the previous charge stays: under-counted, never lost.
        if next.is_some() { self.registered_memory=next; }
    }
    fn admit_next_doc(&mut self,holding:&mut Holding) {
        if self.docs.len()<self.docs.capacity(){return}
        let target=self.docs.capacity().saturating_mul(2).max(1);
        let bytes=target.saturating_mul(std::mem::size_of::<PoolDoc>()) as u64;
        let next=holding.handoff_credit(bytes,"pooled document slots handoff");
        self.docs.reserve_exact(target-self.docs.len());
        if next.is_some() { self.docs_memory=next; }
    }

    /// Bring the charge up to the memtable's current size.
    fn account(&mut self) {
        // Its read caches carry their own charge (and are not locked here).
        let bytes = self.writer.indexing_bytes() as u64;
        self.charge.resize(bytes);
        self.accounted_docs = self.docs.len();
    }
}

/// A request holding a pooled memtable: its token in `Shard::holders` and
/// the freeze generation it has observed.
struct Holding {
    token: u64,
    seen: u64,
    /// The request's claim (taken at sequencing): each memtable it hands
    /// off gets a reference, the one it ends in gets this one.
    claim: Option<Arc<ClaimRef>>,
    registry_credits: Option<rustic_core::admission::Reservation>,
}

impl Holding {
    /// `bytes` of the pre-WAL handoff credits. These run after the WAL
    /// append, where failing the request would leave sequenced operations
    /// half inverted, so a shortfall (not expected: the preflight sizes the
    /// credits for every boundary) never panics or fails: ordinary admission
    /// is tried, and if that is refused too the allocation proceeds
    /// unadmitted (recorded as unplanned) rather than lose acknowledged work.
    fn handoff_credit(&mut self, bytes: u64, label: &str) -> Option<rustic_core::admission::Reservation> {
        if let Some(credit) = self.registry_credits.as_mut().and_then(|credits| credits.split(bytes)) {
            return Some(credit);
        }
        match handoff_budget().reserve(bytes, label) {
            Ok(memory) => Some(memory),
            Err(error) => {
                rustic_core::admission::Context::capture().record_unplanned();
                tracing::warn!("pre-WAL handoff credits short and [{label}] refused ({error}); continuing unadmitted");
                None
            }
        }
    }
}

/// The budget a short handoff falls back to (the shared one; a test may
/// substitute its own on this thread to reach the refused path).
fn handoff_budget() -> Arc<rustic_core::admission::Budget> {
    #[cfg(test)]
    if let Some(budget) = TEST_HANDOFF_BUDGET.with(|b| b.borrow().clone()) { return budget; }
    rustic_core::admission::shared()
}
#[cfg(test)]
thread_local! {static TEST_HANDOFF_BUDGET: std::cell::RefCell<Option<Arc<rustic_core::admission::Budget>>> = const { std::cell::RefCell::new(None) };}

/// One reference to a request's claim (`WriteState::claim`, from its first
/// seq_no). References are created only by the request holding one and
/// dropped only under the write state (at check-in), so the check-in that
/// drops the last reference releases the claim.
struct ClaimRef(i64, Option<rustic_core::admission::Reservation>);

/// A request's pooled memtable while it inverts ([`Shard::apply_staged`],
/// [`Shard::replay_staged`]). If the inversion unwinds (a panic), the drop
/// hands the memtable (with the documents other requests acknowledged in
/// it, and the request's claim) to the frozen list, releases the holding
/// (no freeze waits for a dead holder), marks the request's seq_nos
/// processed (no checkpoint hole) and fails the engine (its translog holds
/// ops the index may now lack: ES fails the engine on an indexing failure
/// it cannot roll back).
struct Inverting<'a> {
    shard: &'a Shard,
    holding: Option<Holding>,
    pooled: Option<PoolWriter>,
    seq_nos: Vec<i64>,
}

impl<'a> Inverting<'a> {
    fn new(shard: &'a Shard, (holding, pooled): (Holding, PoolWriter), seq_nos: Vec<i64>) -> Self {
        Inverting { shard, holding: Some(holding), pooled: Some(pooled), seq_nos }
    }
    fn parts(&mut self) -> (&mut Holding, &mut PoolWriter) {
        (self.holding.as_mut().expect("held"), self.pooled.as_mut().expect("held"))
    }
    /// The inversion completed: the caller completes the request.
    fn finish(mut self) -> (Holding, PoolWriter) {
        (self.holding.take().expect("held"), self.pooled.take().expect("held"))
    }
}

impl Drop for Inverting<'_> {
    fn drop(&mut self) {
        let (Some(mut holding), Some(mut pooled)) = (self.holding.take(), self.pooled.take()) else { return };
        if let Some(claim) = holding.claim.take() { pooled.registered.push(claim); }
        if !pooled.docs.is_empty() || !pooled.registered.is_empty() {
            pooled.account();
            self.shard.frozen.lock().push_back(pooled);
        }
        self.shard.release_holding(holding);
        {
            let mut ws = self.shard.write.lock();
            for &seq_no in &self.seq_nos { ws.hist.mark(seq_no); }
        }
        self.shard.mark_tragic("indexing failed mid-request (panic)");
    }
}

/// Pre-WAL resources without an active refresh holder. Older acknowledged
/// pooled writers remain visible to freezes while sequencing or WAL work waits.
struct PreparedPool<'a> {
    shard: &'a Shard,
    writer: PoolWriter,
    holder_memory: rustic_core::admission::Reservation,
    registry_credits: rustic_core::admission::Reservation,
    credits: rustic_core::postings::BitmapCredits,
    prepared_claim: Arc<ClaimRef>,
    documents: usize,
}
/// Reserve pending-map slots before sequencing. Concurrent uninstalled
/// batches count against capacity too, so none can consume another batch's
/// reserved slots and force a map reallocation after its WAL append.
struct PendingMapSlots<'a> { shard: &'a Shard, count: usize }
impl<'a> PendingMapSlots<'a> {
    fn new(shard: &'a Shard, count: usize) -> Result<Self> {
        let mut ws = shard.write.lock();
        let inflight = ws.pending_inflight.checked_add(count).ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending map claims overflow"))?;
        let required = ws.pending_docs.len().checked_add(inflight).ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending map capacity overflow"))?;
        if required > ws.pending_docs.capacity() {
            // HashMap's load factor and power-of-two buckets, including
            // control bytes. The old admitted table stays live through grow.
            let buckets = required.checked_next_power_of_two().and_then(|n| n.checked_mul(2))
                .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending map buckets overflow"))?;
            let bytes = buckets.checked_mul(std::mem::size_of::<(String, Arc<PendingDoc>)>() + 1)
                .and_then(|n| n.checked_add(32)).ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending map bytes overflow"))?;
            let memory = rustic_core::admission::shared().reserve(bytes as u64, "pending document table")?;
            let additional = required - ws.pending_docs.len();
            ws.pending_docs.try_reserve(additional).map_err(|e| EsError::new(429, "circuit_breaking_exception", e.to_string()))?;
            ws.pending_memory = Some(memory);
        }
        ws.pending_inflight = inflight;
        Ok(Self { shard, count })
    }
}
impl Drop for PendingMapSlots<'_> {
    fn drop(&mut self) {
        let mut ws = self.shard.write.lock();
        ws.pending_inflight = ws.pending_inflight.checked_sub(self.count).expect("pending map claim belongs to this batch");
    }
}

struct VersionMapSlots<'a> { shard: &'a Shard, count: usize }
impl<'a> VersionMapSlots<'a> {
    fn new(shard: &'a Shard, count: usize) -> Result<Self> {
        let mut ws = shard.write.lock();
        let inflight = ws.version_inflight.checked_add(count).ok_or_else(|| EsError::new(429,"circuit_breaking_exception","version claims overflow"))?;
        let required = ws.versions.len().checked_add(inflight).ok_or_else(|| EsError::new(429,"circuit_breaking_exception","version capacity overflow"))?;
        ws.versions.reserve_capacity(required)?;
        ws.version_inflight = inflight;
        Ok(Self { shard, count })
    }
}
impl Drop for VersionMapSlots<'_> {
    fn drop(&mut self) { let mut ws=self.shard.write.lock(); ws.version_inflight=ws.version_inflight.checked_sub(self.count).expect("version claim belongs to batch"); }
}

impl<'a> PreparedPool<'a> {
    fn new(shard: &'a Shard, documents: usize) -> Result<Self> {
        let holder_memory=rustic_core::admission::shared().reserve(2048,"refresh holder node")?;
        let queue_memory=rustic_core::admission::shared().reserve(PoolWriter::queue_bytes() as u64,"pooled queue node")?;
        let mut writer=PoolWriter::fresh_with_credit(Some(queue_memory));
        let credits=writer.writer.prepare_bitmap_batch(documents)?;
        writer.prepare_registry(documents)?;
        let claim_memory=rustic_core::admission::shared().reserve((std::mem::size_of::<ClaimRef>()+2*std::mem::size_of::<usize>()) as u64,"pooled claim owner")?;
        let prepared_claim=Arc::new(ClaimRef(0,Some(claim_memory)));
        // Geometric buffers across arbitrary document-boundary handoffs.
        let bytes=documents.checked_add(1).and_then(|n|n.checked_mul(4*(std::mem::size_of::<PoolDoc>()+std::mem::size_of::<Arc<ClaimRef>>())+PoolWriter::queue_bytes()))
            .ok_or_else(||EsError::new(429,"circuit_breaking_exception","pooled registry credits overflow"))?;
        let registry_credits=rustic_core::admission::shared().reserve(bytes as u64,"pooled registry handoff credits")?;
        // Preserve reuse, but do not check the writer out: a freeze can still
        // take it, and another request can use it before this one inverts.
        // The newest few are prepared (the newest must be): concurrent
        // requests each take one, and a request finding none prepared
        // inverts into a fresh memtable that joins the pool, so under three
        // touch clients a refresh checked in ~50 small memtables, each
        // sealed and its postings unioned.
        {
            let mut pool=shard.pool.lock();
            for (i,pooled) in pool.iter_mut().rev().take(POOL_PREPARED).enumerate() {
                let prepared=(|| -> Result<()> { pooled.writer.prepare_bitmap_capacity(documents)?; pooled.prepare_registry(documents) })();
                if i==0 { prepared?; }
            }
        }
        Ok(Self {shard,writer,holder_memory,registry_credits,credits,prepared_claim,documents})
    }
    fn into_inverting(self, claim: Option<i64>, seq_nos: Vec<i64>) -> (Inverting<'a>, rustic_core::postings::BitmapCredits) {
        let Self {shard,writer,holder_memory,registry_credits,credits,mut prepared_claim,documents}=self;
        let token=shard.next_holder.fetch_add(1,Ordering::Relaxed);
        let seen=shard.freeze_gen.load(Ordering::Acquire);
        // Register before taking the pool lock: a freeze either takes the
        // idle writer or waits for this now-active inversion to hand it off.
        shard.holders.lock().insert(token,(seen,holder_memory));
        let pooled={
            let mut pool=shard.pool.lock();
            let usable=|p:&PoolWriter| p.docs.len().checked_add(documents).is_some_and(|n|n<=p.docs.capacity())
                && p.registered.len()<p.registered.capacity()
                && p.writer.has_prepared_bitmap_capacity(documents);
            // The newest prepared one ([`POOL_PREPARED`]).
            match pool.iter().rev().take(POOL_PREPARED).position(usable) {
                Some(from_back) => {
                    let at=pool.len()-1-from_back;
                    let mut tail=pool.split_off(at);
                    let taken=tail.pop_front().expect("usable pooled writer");
                    pool.append(&mut tail);
                    taken
                }
                None => writer,
            }
        };
        let claim=claim.map(|first| {Arc::get_mut(&mut prepared_claim).unwrap().0=first;prepared_claim});
        let holding=Holding {token,seen,claim,registry_credits:Some(registry_credits)};
        (Inverting::new(shard,(holding,pooled),seq_nos),credits)
    }
}

/// A graph capture refused because graph ingestion is in flight.
fn ingestion_in_flight(error: &rustic_core::error::RusticError) -> bool {
    let message = error.to_string();
    message.contains("in-flight ingestion") || message.contains("pending graph reservations")
}

/// Longest a blocking refresh / flush waits for holders to hand their
/// pooled memtables off (each does at its next document boundary).
const FREEZE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// An explicit refresh slower than this logs its phases
/// ("slow engine refresh phases").
const SLOW_REFRESH: std::time::Duration = std::time::Duration::from_secs(1);

/// Where an explicit refresh ([`Shard::refresh`]) spent its time, recorded
/// on its own thread only while it runs (logging; no behavior).
#[derive(Default, Clone, Copy)]
struct RefreshPhases {
    checkin_lock: std::time::Duration,
    /// Of a check-in: starting the freeze (the pool lock).
    begin_freeze: std::time::Duration,
    /// Of a check-in: checking the memtables in, and `after_checkin`.
    checkin_writers: std::time::Duration,
    after_checkin: std::time::Duration,
    checkin_memtables: u32,
    checkin_docs: u32,
    checkin_docs_loop: std::time::Duration,
    checkin_core: std::time::Duration,
    checkin_older: u32,
    handoff: std::time::Duration,
    write_lock: std::time::Duration,
    core_lock: std::time::Duration,
    seal_first: bool,
    seal_lock: std::time::Duration,
    encode: std::time::Duration,
    encoded_bytes: u64,
    in_flight_seal: std::time::Duration,
    publish_core_lock: std::time::Duration,
    publish: std::time::Duration,
    /// Of `publish`: a sorted refresh waiting for every in-flight seal job
    /// and the seal lock ([`Shard::seal_fences`]).
    sorted_fence: std::time::Duration,
}

thread_local! {
    static REFRESH_PHASES: std::cell::Cell<Option<RefreshPhases>> = const { std::cell::Cell::new(None) };
}

impl RefreshPhases {
    /// Start recording on this thread until the guard drops.
    fn begin() -> impl Drop {
        struct Recording(Option<RefreshPhases>);
        impl Drop for Recording { fn drop(&mut self) { REFRESH_PHASES.with(|p| p.set(self.0)); } }
        Recording(REFRESH_PHASES.with(|p| p.replace(Some(RefreshPhases::default()))))
    }
    fn current() -> RefreshPhases { REFRESH_PHASES.with(|p| p.get()).unwrap_or_default() }
}

/// Add to the running refresh's phases (no-op off a refresh).
fn note_refresh_phase(f: impl FnOnce(&mut RefreshPhases)) {
    REFRESH_PHASES.with(|cell| if let Some(mut phases) = cell.get() { f(&mut phases); cell.set(Some(phases)); });
}

/// A completing request polls for a freeze this often while it waits for
/// the write state ([`Shard::complete_pooled`]).
const HANDOFF_POLL: std::time::Duration = std::time::Duration::from_millis(2);

/// A flush asks pooled memtables to freeze but barely waits for the hand-off:
/// a write still in flight stays above the commit point (its claim holds it
/// down) and is replayed from the translog, so the commit needs none of it.
const FLUSH_FREEZE_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

/// Backoff of a refused background seal admission: doubling from the
/// minimum to the maximum while refusals continue.
const SEAL_REFUSAL_BACKOFF_MIN: std::time::Duration = std::time::Duration::from_millis(10);
const SEAL_REFUSAL_BACKOFF_MAX: std::time::Duration = std::time::Duration::from_millis(250);

/// How long after a recovery replay's last write this copy's background
/// seals stay forced ([`Shard::replay_seals_forced`]).
const REPLAY_SEAL_FORCED_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

/// Monotonic nanoseconds since the first call (seal backoff deadlines).
fn seal_clock_nanos() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

/// Waits between retries of an encoded seal whose install admission was
/// refused ([`Shard::install_seal_retrying`]), before its work is given up.
const SEAL_INSTALL_RETRY_DELAYS: [std::time::Duration; 3] =
    [std::time::Duration::from_millis(5), std::time::Duration::from_millis(20), std::time::Duration::from_millis(80)];

struct PoolDoc {
    id: registry::DeleteKey,
    seq_no: i64,
    local: rustic_core::types::DocId,
    /// An older copy of the id may exist (its sequencing found a version
    /// entry or a live copy): the check-in deletes it. A new id has none
    /// (every op sequenced before it on that id is in the version map until
    /// checked in, and a live copy was found by the lookup), so its
    /// check-in skips the per-document term lookup over every leaf, which
    /// is most of a check-in's time on a large shard.
    older: bool,
}

/// The latest op of an id whose document is in a pooled memtable
/// (realtime GET serves it until the check-in).
/// Check-ins of at least this many documents by a refresh (a search-idle
/// search's, the refresh API's, a scheduled one) free their pooled copies on
/// [`defer_drop`]'s thread. Only there: other check-ins (write threads'
/// hand-offs) free them in place, where the memory was allocated (freeing it
/// from another thread cost the write path about 8% of touch-bulk
/// throughput).
const DEFERRED_DROP_DOCS: usize = 1024;

thread_local! {
    /// Set while this thread refreshes ([`DeferCheckinDrops`]).
    static DEFER_CHECKIN_DROPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The pooled copies a refresh's check-ins released, and their document
    /// count: freed together when the refresh ends, on [`defer_drop`]'s
    /// thread once they reach [`DEFERRED_DROP_DOCS`]. A refresh under touch
    /// writes checks in dozens of small memtables (about 150 documents
    /// each); freeing each in place cost about 40 ms of an 80 ms
    /// search-idle refresh (10k TSDB docs).
    static CHECKIN_DROPS: std::cell::RefCell<(Vec<Box<dyn Send>>, usize)> = const { std::cell::RefCell::new((Vec::new(), 0)) };
}

/// Check-ins on this thread defer their large drops until dropped (a
/// refresh: [`Shard::publish`], [`Shard::try_publish`], [`Shard::try_publish_for`]).
struct DeferCheckinDrops(bool);

impl DeferCheckinDrops {
    fn enter() -> Self {
        DeferCheckinDrops(DEFER_CHECKIN_DROPS.with(|d| d.replace(true)))
    }
}

impl Drop for DeferCheckinDrops {
    fn drop(&mut self) {
        DEFER_CHECKIN_DROPS.with(|d| d.set(self.0));
        if !self.0 {
            let (drops, docs) = CHECKIN_DROPS.with(|c| std::mem::take(&mut *c.borrow_mut()));
            if docs >= DEFERRED_DROP_DOCS { defer_drop(Box::new(drops)) } else { drop(drops) }
        }
    }
}

/// A write thread's check-ins (a request completing its pooled memtable)
/// collect what they release while it holds the write state and the core,
/// and free it in place once those are released (where it was allocated:
/// freeing on another thread costs the write path more). Inside a refresh
/// ([`DeferCheckinDrops`]) the refresh's own rule applies.
struct LocalCheckinDrops(bool);

impl LocalCheckinDrops {
    fn enter() -> Self {
        LocalCheckinDrops(DEFER_CHECKIN_DROPS.with(|d| d.replace(true)))
    }
}

impl Drop for LocalCheckinDrops {
    fn drop(&mut self) {
        DEFER_CHECKIN_DROPS.with(|d| d.set(self.0));
        if !self.0 {
            let (drops, _) = CHECKIN_DROPS.with(|c| std::mem::take(&mut *c.borrow_mut()));
            drop(drops);
        }
    }
}

/// Free memtables a seal install replaced: with the running refresh's
/// pooled copies ([`CHECKIN_DROPS`]), else in place.
fn retire_checked_in(writers: Vec<Arc<rustic_core::segment::SegmentWriter>>) {
    if writers.is_empty() { return; }
    if DEFER_CHECKIN_DROPS.with(|d| d.get()) {
        let n = writers.iter().map(|w| w.num_docs() as usize).sum::<usize>();
        CHECKIN_DROPS.with(|c| { let mut c = c.borrow_mut(); c.0.push(Box::new(writers)); c.1 += n; });
    }
}

/// Drop `value` on a background thread (one shared, low-priority thread;
/// inline if it cannot be started). Memory charged to what is dropped is
/// released when that thread gets to it.
fn defer_drop(value: Box<dyn Send>) {
    static DROPPER: std::sync::OnceLock<Option<Mutex<std::sync::mpsc::Sender<Box<dyn Send>>>>> = std::sync::OnceLock::new();
    let sender = DROPPER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn Send>>();
        std::thread::Builder::new().name("es-dropper".into()).spawn(move || {
            rustic_core::lower_thread_priority(5);
            for value in rx { drop(value); }
        }).ok().map(|_| Mutex::new(tx))
    });
    match sender {
        Some(tx) => if let Err(std::sync::mpsc::SendError(value)) = tx.lock().send(value) { drop(value) },
        None => drop(value),
    }
}

pub(crate) struct PendingDoc {
    seq_no: i64,
    version: u64,
    primary_term: u64,
    doc: Document,
}

impl PendingDoc {
    /// All metadata growth and the pending Arc itself are admitted while the
    /// source document still belongs to the unsequenced request. Moving the
    /// source/routing/ignored strings below transfers their retained ownership
    /// to the Document; only the ID field needs a new string allocation.
    fn prepare_document(doc: &mut Document, id: &String, source: &String, routing: Option<&String>, ignored: &[String]) -> Result<()> {
        let source_needed = !doc.fields.iter().any(|(field, _)| *field == SOURCE_FIELD);
        let slots = 4usize.checked_add(usize::from(source_needed))
            .and_then(|n| n.checked_add(usize::from(routing.is_some())))
            .and_then(|n| n.checked_add(ignored.len()))
            .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending metadata slots overflow"))?;
        doc.reserve_field_slots(slots)?;
        let mut bytes = std::mem::size_of::<PendingDoc>() + 2 * std::mem::size_of::<usize>();
        // One copy becomes the ID field; one becomes the pending-map key.
        // The map value owns this Document, so the key cannot outlive its
        // charge; replica input can retain capacity beyond its logical length.
        // This does not admit the HashMap backing capacity.
        for len in [id.len(), id.capacity()]
            .into_iter().chain(source_needed.then_some(source.capacity()))
            .chain(routing.map(String::capacity))
            .chain(ignored.iter().map(String::capacity)) {
            bytes = bytes.checked_add(len).ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "pending metadata bytes overflow"))?;
        }
        doc.admit_allocation(bytes)?;
        // The document is complete: what remains charged is exactly what
        // its producers admitted (chunked admission's spare credit returns).
        doc.settle_allocation();
        Ok(())
    }
}

thread_local! {
    /// Set while a refresh publishes an index-sorted core's memtables
    /// unsealed ([`Shard::try_publish_for`]).
    static MEMTABLES_FIRST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a search-idle or scheduled refresh of an index-sorted core
/// publishes its memtables without sealing them first
/// (`RUSTIC_SORTED_REFRESH_SEALS=1` keeps the sorted seal on the refresh).
fn publish_sorted_memtables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTIC_SORTED_REFRESH_SEALS").map_or(true, |v| v != "1"))
}

/// Pooled memtables a request prepares for its batch, newest first
/// ([`PreparedPool::new`]); it inverts into the newest of them still usable.
const POOL_PREPARED: usize = 8;

/// A pooled memtable holding this many documents is checked in by the
/// request that filled it.
const POOL_CHECKIN_DOCS: usize = 16_384;

/// Shards of the per-id lock table ([`KeyedLocks`]).
const ID_LOCK_SHARDS: usize = 64;

/// ES `KeyedLock` on the uid: one lock per id some writer holds or waits
/// for, created on demand and removed when its last holder and waiter
/// leave. The table is sharded by a per-copy randomly keyed hash, so ids
/// of concurrent bulks only meet when they are the same id. A batch takes
/// its ids in sorted order, so batches never deadlock.
pub(crate) struct KeyedLocks {
    shards: Box<[(Mutex<HashMap<String, KeyState>>, parking_lot::Condvar)]>,
    hasher: std::collections::hash_map::RandomState,
}

#[derive(Default)]
struct KeyState {
    held: bool,
    waiters: usize,
}

/// The ids a batch holds ([`KeyedLocks::lock_all`]); released on drop.
pub(crate) struct KeyGuard<'a> {
    locks: &'a KeyedLocks,
    ids: Vec<String>,
}

impl KeyedLocks {
    fn new() -> Self {
        KeyedLocks { shards: (0..ID_LOCK_SHARDS).map(|_| (Mutex::new(HashMap::new()), parking_lot::Condvar::new())).collect(), hasher: Default::default() }
    }

    fn shard(&self, id: &str) -> &(Mutex<HashMap<String, KeyState>>, parking_lot::Condvar) {
        use std::hash::BuildHasher;
        &self.shards[(self.hasher.hash_one(id) as usize) % self.shards.len()]
    }

    /// Lock every id of `ids` (duplicates once), in sorted order.
    pub(crate) fn lock_all<'a>(&self, ids: impl Iterator<Item = &'a str>) -> KeyGuard<'_> {
        let sorted: std::collections::BTreeSet<&str> = ids.collect();
        let mut guard = KeyGuard { locks: self, ids: Vec::with_capacity(sorted.len()) };
        for id in sorted {
            let (map, cv) = self.shard(id);
            let mut map = map.lock();
            loop {
                let state = map.entry(id.to_string()).or_default();
                if !state.held {
                    state.held = true;
                    break;
                }
                state.waiters += 1;
                cv.wait(&mut map);
                if let Some(state) = map.get_mut(id) {
                    state.waiters -= 1;
                }
            }
            guard.ids.push(id.to_string());
        }
        guard
    }

    /// Ids locked or waited for now (the table stays as small as that).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.shards.iter().map(|(map, _)| map.lock().len()).sum()
    }
}

impl Drop for KeyGuard<'_> {
    fn drop(&mut self) {
        for id in &self.ids {
            let (map, cv) = self.locks.shard(id);
            let mut map = map.lock();
            if let Some(state) = map.get_mut(id.as_str()) {
                state.held = false;
                if state.waiters == 0 {
                    map.remove(id.as_str());
                } else {
                    cv.notify_all();
                }
            }
        }
    }
}

/// Buffered deletes resolved per core read guard ([`Shard::resolve_buffered_deletes`]).
const RESOLVE_CHUNK: usize = 256;

/// [`Shard::reader_pressure`]: how much old readers hold back.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReaderPressure {
    /// Age of the oldest superseded reader view still in use (zero: none).
    pub oldest_live_view_age: std::time::Duration,
    /// Mutable-graph repair backlog and retired memory held for readers.
    pub vectors: rustic_core::vector::store::MutablePressure,
}

static REFRESH_HOLDS: AtomicU64 = AtomicU64::new(0);
static REFRESH_HOLD_TOTAL_NANOS: AtomicU64 = AtomicU64::new(0);
static REFRESH_HOLD_MAX_NANOS: AtomicU64 = AtomicU64::new(0);

/// Core write-lock holds of refresh publications on this node (scheduled,
/// explicit and search-idle refreshes): `(count, total, max)`.
pub fn refresh_lock_hold() -> (u64, std::time::Duration, std::time::Duration) {
    (
        REFRESH_HOLDS.load(Ordering::Relaxed),
        std::time::Duration::from_nanos(REFRESH_HOLD_TOTAL_NANOS.load(Ordering::Relaxed)),
        std::time::Duration::from_nanos(REFRESH_HOLD_MAX_NANOS.load(Ordering::Relaxed)),
    )
}

fn note_refresh_hold(held: std::time::Duration) {
    let nanos = u64::try_from(held.as_nanos()).unwrap_or(u64::MAX);
    REFRESH_HOLDS.fetch_add(1, Ordering::Relaxed);
    REFRESH_HOLD_TOTAL_NANOS.fetch_add(nanos, Ordering::Relaxed);
    REFRESH_HOLD_MAX_NANOS.fetch_max(nanos, Ordering::Relaxed);
}

/// Queued mutable-graph deletes physically repaired per field per
/// background pass ([`Shard::repair_vectors`]).
const VECTOR_REPAIR_BATCH: usize = 4096;

/// Fence releases waiting for the next flush capture of a shard copy (by
/// path); see [`release_at_next_capture`].
type CaptureRelease = (PathBuf, Box<dyn FnOnce() + Send>);
static CAPTURE_RELEASES: parking_lot::Mutex<Vec<CaptureRelease>> = parking_lot::const_mutex(Vec::new());

/// Run `release` (once) when the next flush of the shard copy at
/// `shard_path` has captured its commit point: a caller fencing writes
/// around a flush it does not run itself (the cluster's explicit `_flush`
/// holds each primary's replication gate across the engine flush request)
/// needs the fence only until then. [`cancel_capture_releases`] runs what
/// no flush consumed.
pub fn release_at_next_capture(shard_path: &Path, release: Box<dyn FnOnce() + Send>) {
    CAPTURE_RELEASES.lock().push((shard_path.to_path_buf(), release));
}

/// Run (release) every pending [`release_at_next_capture`] of `shard_path`.
pub fn cancel_capture_releases(shard_path: &Path) {
    run_capture_releases(shard_path);
}

fn run_capture_releases(shard_path: &Path) {
    let due: Vec<CaptureRelease> = {
        let mut all = CAPTURE_RELEASES.lock();
        let (due, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut *all).into_iter().partition(|(path, _)| path == shard_path);
        *all = rest;
        due
    };
    for (_, release) in due {
        release();
    }
}

/// How long a background worker waits for a busy core before re-arming a
/// deferred refresh (see [`Shard::refresh_deferred`]).
const DEFERRED_REFRESH_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

/// ES caches store stats for `index.store.stats_refresh_interval` (10s).
const STORE_STATS_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(10);

/// Background work of a shard copy (Lucene's flush of full in-memory
/// segments and `ConcurrentMergeScheduler`): memtables are sealed and
/// segments merged on the background workers ([`crate::refresh`]), never
/// under the write path's locks.
///
/// Lock order: `force_lock` -> `seal_running` -> `seal_lock` -> `merge_running` -> `merge_lock` ->
/// `checkin_lock` (the shard's pool check-in, see `Shard::checkin_lock`) -> write state ->
/// core (-> the core's view-slot pin lock, taken by a publish; the view
/// slot's own mutex is a leaf held only to clone or swap the view).
/// Searches hold none of these while they run: they read
/// [`Shard::reader_view`]. Every foreground operation that seals or commits
/// (flush, force merge) holds `seal_lock`, so it never races a background
/// seal (point in time snapshots retain a reader view and seal nothing,
/// except a memtable-only view of mutable vectors, see
/// [`Shard::retain_view`]). A force merge holds `force_lock` throughout and
/// `merge_lock` only to claim its inputs: background merges of other
/// segments (written meanwhile) run concurrently with its encoding.
/// `merge_lock` pins the segment set (no merge install) and is held by a
/// merge only to install (a force merge: also to claim), never across its
/// encoding: a flush (which holds
/// it through its commit) never waits for a running merge, only for an
/// install. `merge_running` admits one merge (background or a force merge's
/// claim) at a time and is held across the encoding.
#[derive(Default)]
pub(crate) struct Background {
    pub(crate) seal_lock: BgLock,
    /// One seal job encoding at a time (held across the encoding by
    /// background seals and by every foreground path that needs its own
    /// seal to run: sorted refresh, seal_all, seal_published, force merge,
    /// files_pinned). `seal_lock` only pins the segment set (held by a
    /// background seal to claim and to install, never across its encoding),
    /// so a flush never waits for a background seal's encoding: it commits
    /// what is sealed with its commit point below the memtables still
    /// encoding ([`Shard::flush_then`]).
    pub(crate) seal_running: BgSemaphore,
    pub(crate) merge_lock: BgLock,
    /// Held by a flush across its commit's writes ([`CommitJob::run`]: the
    /// graph checkpoints, deletion files, `meta.json`, and the deletes of
    /// files the previous commit listed), which run with `merge_lock`
    /// released so merges install meanwhile. Taken after `merge_lock`
    /// (never the other way round); [`Shard::files_pinned`] takes it too.
    ///
    /// [`CommitJob::run`]: rustic_core::index::CommitJob::run
    pub(crate) commit_lock: BgLock,
    /// One merge at a time, held across its encoding (see above).
    pub(crate) merge_running: MergeSlots,
    /// The merge scheduler / policy settings ([`Shard::set_merge_config`]).
    pub(crate) merge_config: RwLock<MergeConfig>,
    /// Merges the policy would start now beyond the running ones (the
    /// merge backlog, ES pending merges): writes stall past
    /// [`max_merge_count`].
    pub(crate) merge_backlog: std::sync::atomic::AtomicUsize,
    /// ES `IndexingThrottle`: one indexing request at a time while merges are
    /// behind ([`Shard::stall_if_behind`]).
    pub(crate) indexing_throttle: BgLock,
    /// Serializes explicit force merges of the copy.
    pub(crate) force_lock: BgLock,
    /// Serializes flushes: a flush encodes its seal with `seal_lock`
    /// released, and commits must not interleave (a later capture's commit
    /// may never overwrite a higher commit point).
    pub(crate) flush_lock: BgLock,
    /// Sealing is due (memtables over the threshold, or many published
    /// memtables after a refresh).
    pub(crate) want_seal: std::sync::atomic::AtomicBool,
    /// No background seal starts before this [`seal_clock_nanos`] (a refused
    /// seal admission's backoff).
    pub(crate) seal_backoff_until: std::sync::atomic::AtomicU64,
    /// Consecutive refused seal admissions (sets the backoff; reset when a
    /// seal starts).
    pub(crate) seal_refusals: std::sync::atomic::AtomicU32,
    /// [`seal_clock_nanos`] of the last write a replay of acknowledged ops
    /// made to this copy (0: none): see [`Shard::replay_seals_forced`].
    pub(crate) replay_write_nanos: std::sync::atomic::AtomicU64,
    /// The buffer is to be written out by the next background seal
    /// ([`Shard::write_indexing_buffer`]).
    pub(crate) idle_seal: std::sync::atomic::AtomicBool,
    /// A buffer write-out left memtables unclaimed: the next seal jobs
    /// take them (one coalesced segment each) until none is left.
    pub(crate) draining: std::sync::atomic::AtomicBool,
    /// Searches on a search-idle copy wait here for the one refresh the
    /// first of them runs.
    pub(crate) idle_refresh: parking_lot::Mutex<()>,
    /// The refresh a first search on a search-idle copy runs is under way.
    pub(crate) idle_refreshing: std::sync::atomic::AtomicBool,
    /// Refreshes not run by a search (the refresh API, a scheduled or
    /// deferred refresh) under way: a search never waits on one
    /// ([`Shard::before_search`]).
    pub(crate) refreshing: std::sync::atomic::AtomicUsize,
    /// [`Shard::close_files`] ran: no flush may write this copy's files.
    pub(crate) files_closed: std::sync::atomic::AtomicBool,
    /// The node's indexing buffer is past 1.5x and this copy is written
    /// out: indexing runs one request at a time (ES IMC throttling).
    pub(crate) memory_throttled: std::sync::atomic::AtomicBool,
    /// Test seam: the byte stall's limit (`0`: the indexing-buffer share).
    pub(crate) stall_limit_override: std::sync::atomic::AtomicU64,
    /// Measured seal throughput, bytes of memtables per second (`0`: none
    /// yet): caps the unsealed backlog ([`Shard::stall_limit_bytes`]).
    pub(crate) seal_rate: std::sync::atomic::AtomicU64,
    /// Test seam: [`LARGE_PUBLISH_BYTES`] (`0`: the constant).
    pub(crate) large_publish_override: std::sync::atomic::AtomicU64,
    /// A merge may be due (segments were added).
    pub(crate) want_merge: std::sync::atomic::AtomicBool,
    /// A scheduled refresh found the core busy: a background worker
    /// refreshes the copy (blocking), so the scheduler never waits on it.
    pub(crate) want_refresh: std::sync::atomic::AtomicBool,
    /// A refresh made deletes visible: queued mutable-graph deletes may
    /// need physical repair ([`Shard::repair_vectors`]).
    pub(crate) want_repair: std::sync::atomic::AtomicBool,
    /// Queued on (or running on) a background refresh / seal worker.
    pub(crate) queued: std::sync::atomic::AtomicBool,
    /// Queued on (or running on) a background merge worker.
    pub(crate) merge_queued: std::sync::atomic::AtomicBool,
    /// Background workers serve this copy ([`crate::refresh::register`]);
    /// otherwise the write path seals inline.
    pub(crate) served: std::sync::atomic::AtomicBool,
    /// Docs in the memtables being written that make a seal due (`0`:
    /// [`SEAL_THRESHOLD_DOCS`]; tests lower it).
    pub(crate) threshold: std::sync::atomic::AtomicU32,
}

/// A lock whose guard owns its state, so a background worker can hold it
/// across a job without keeping the shard itself alive (a dropped shard
/// releases its shard lock at once; the job's result is then discarded).
#[derive(Default, Clone)]
pub(crate) struct BgLock(Arc<(Mutex<bool>, parking_lot::Condvar)>);

/// Held [`BgLock`]; released on drop.
pub(crate) struct BgGuard(Arc<(Mutex<bool>, parking_lot::Condvar)>);

impl BgLock {
    pub(crate) fn lock(&self) -> BgGuard {
        let (m, cv) = &*self.0;
        let mut held = m.lock();
        while *held {
            cv.wait(&mut held);
        }
        *held = true;
        BgGuard(self.0.clone())
    }

    pub(crate) fn try_lock(&self) -> Option<BgGuard> {
        let mut held = self.0 .0.lock();
        if *held {
            return None;
        }
        *held = true;
        Some(BgGuard(self.0.clone()))
    }

    /// [`BgLock::lock`], waiting at most until `deadline` (`None`: still held).
    pub(crate) fn lock_until(&self, deadline: std::time::Instant) -> Option<BgGuard> {
        let (m, cv) = &*self.0;
        let mut held = m.lock();
        while *held {
            if cv.wait_until(&mut held, deadline).timed_out() && *held {
                return None;
            }
        }
        *held = true;
        Some(BgGuard(self.0.clone()))
    }
}

impl Drop for BgGuard {
    fn drop(&mut self) {
        let (m, cv) = &*self.0;
        *m.lock() = false;
        cv.notify_all();
    }
}

thread_local! {
    /// The shard indexing throttle this thread's write holds
    /// ([`HeldThrottle`]).
    static HELD_THROTTLE: std::cell::RefCell<Option<BgGuard>> = const { std::cell::RefCell::new(None) };
}

/// A write's hold on its shard's indexing throttle ([`Shard::merge_throttle`]),
/// kept where [`Shard::group_sync`] can let it go: the throttle serializes
/// the indexing of throttled writes, never their translog fsync (which
/// would serialize the fsyncs too, one per request instead of one per
/// group commit). Released at the latest when this drops.
struct HeldThrottle(bool);

impl HeldThrottle {
    fn hold(guard: Option<BgGuard>) -> HeldThrottle {
        let held = guard.is_some();
        if held {
            HELD_THROTTLE.with(|slot| *slot.borrow_mut() = guard);
        }
        HeldThrottle(held)
    }

    /// Whether this thread's write holds the indexing throttle.
    fn held() -> bool {
        HELD_THROTTLE.with(|slot| slot.borrow().is_some())
    }

    /// Release the throttle this thread's write holds, if any.
    fn release() {
        let guard = HELD_THROTTLE.with(|slot| slot.borrow_mut().take());
        drop(guard);
    }
}

impl Drop for HeldThrottle {
    fn drop(&mut self) {
        if self.0 {
            HeldThrottle::release();
        }
    }
}

/// A counting [`BgLock`]: up to `max` holders of one permit each, or one
/// exclusive holder of all of them (waits for the permits out and keeps new
/// ones off meanwhile).
pub(crate) struct BgSemaphore {
    state: Arc<(Mutex<SemState>, parking_lot::Condvar)>,
    /// Permits (dynamic: [`BgSemaphore::set_max`]).
    max: std::sync::atomic::AtomicUsize,
}

#[derive(Default)]
struct SemState {
    held: usize,
    exclusive: bool,
}

/// Held permit(s) of a [`BgSemaphore`]; released on drop.
pub(crate) struct SemGuard {
    state: Arc<(Mutex<SemState>, parking_lot::Condvar)>,
    exclusive: bool,
}

impl BgSemaphore {
    pub(crate) fn new(max: usize) -> Self {
        BgSemaphore { state: Arc::new((Mutex::new(SemState::default()), parking_lot::Condvar::new())), max: std::sync::atomic::AtomicUsize::new(max.max(1)) }
    }

    /// Change the number of permits (held ones stay until released).
    pub(crate) fn set_max(&self, max: usize) {
        self.max.store(max.max(1), Ordering::Release);
        self.state.1.notify_all();
    }

    fn limit(&self) -> usize {
        self.max.load(Ordering::Acquire)
    }

    /// One permit if one is free now and nobody holds all of them.
    pub(crate) fn try_acquire(&self) -> Option<SemGuard> {
        let mut st = self.state.0.lock();
        if st.exclusive || st.held >= self.limit() { return None; }
        st.held += 1;
        Some(SemGuard { state: self.state.clone(), exclusive: false })
    }

    /// One permit, waiting for it.
    pub(crate) fn acquire(&self) -> SemGuard {
        let (m, cv) = &*self.state;
        let mut st = m.lock();
        while st.exclusive || st.held >= self.limit() { cv.wait(&mut st); }
        st.held += 1;
        SemGuard { state: self.state.clone(), exclusive: false }
    }

    /// Permits held now.
    pub(crate) fn held(&self) -> usize {
        self.state.0.lock().held
    }

    /// [`BgSemaphore::acquire`], waiting at most until `deadline`.
    pub(crate) fn acquire_until(&self, deadline: std::time::Instant) -> Option<SemGuard> {
        let (m, cv) = &*self.state;
        let mut st = m.lock();
        while st.exclusive || st.held >= self.limit() {
            if cv.wait_until(&mut st, deadline).timed_out() && (st.exclusive || st.held >= self.limit()) {
                return None;
            }
        }
        st.held += 1;
        Some(SemGuard { state: self.state.clone(), exclusive: false })
    }

    /// Every permit: waits for the held ones; no new one is granted meanwhile.
    pub(crate) fn acquire_all(&self) -> SemGuard {
        let (m, cv) = &*self.state;
        let mut st = m.lock();
        while st.exclusive { cv.wait(&mut st); }
        st.exclusive = true;
        while st.held > 0 { cv.wait(&mut st); }
        SemGuard { state: self.state.clone(), exclusive: true }
    }

    /// [`BgSemaphore::acquire_all`] if no permit is held now.
    pub(crate) fn try_acquire_all(&self) -> Option<SemGuard> {
        let mut st = self.state.0.lock();
        if st.exclusive || st.held > 0 { return None; }
        st.exclusive = true;
        Some(SemGuard { state: self.state.clone(), exclusive: true })
    }
}

impl Default for BgSemaphore {
    /// The seal-job semaphore ([`seal_jobs`] permits).
    fn default() -> Self {
        BgSemaphore::new(seal_jobs())
    }
}

impl Drop for SemGuard {
    fn drop(&mut self) {
        let (m, cv) = &*self.state;
        let mut st = m.lock();
        if self.exclusive { st.exclusive = false; } else { st.held -= 1; }
        cv.notify_all();
    }
}

/// Background seal jobs a shard copy encodes concurrently: `RUSTIC_SEAL_JOBS`,
/// else `min(4, encode threads / 2)`, at least 1.
fn seal_jobs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RUSTIC_SEAL_JOBS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0)
        .unwrap_or_else(|| (rustic_core::encode_threads() / 2).clamp(1, 4)))
}


/// Concurrent background merges of one shard copy (ES
/// `index.merge.scheduler.max_thread_count`): `RUSTIC_MERGE_THREADS`, else
/// `max(1, min(4, cores / 2))` (the merge pool's size).
fn merge_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RUSTIC_MERGE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0)
        // ES `index.merge.scheduler.max_thread_count` default, per copy.
        .unwrap_or_else(|| (crate::processors::get() / 2).clamp(1, 4)))
}

/// Background merges the node runs at once, over every copy:
/// `RUSTIC_NODE_MERGES`, else half the processors (at least 1); halved
/// while the node searches.
fn node_merge_cap() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RUSTIC_NODE_MERGES").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0)
        .unwrap_or_else(|| (crate::processors::get() / 2).max(1)))
}

/// The node-wide merge permits ([`node_merge_cap`]).
fn node_merges() -> &'static BgSemaphore {
    static S: std::sync::OnceLock<BgSemaphore> = std::sync::OnceLock::new();
    S.get_or_init(|| BgSemaphore::new(node_merge_cap()))
}

/// ES `ThreadPoolMergeScheduler.checkMergeTaskThrottling`: indexing is
/// throttled while the merges in flight (running and queued) exceed
/// `max_merge_count`, and, with the auto IO throttle on, only once the merge
/// IO rate is at its maximum (merges are no longer held back by the rate,
/// and still cannot keep up).
fn merge_throttle_due(in_flight: usize, max_merge_count: usize, auto_throttle: bool, io_rate_at_max: bool) -> bool {
    in_flight > max_merge_count && (!auto_throttle || io_rate_at_max)
}

/// A shard copy's merge scheduler and policy (ES `MergeSchedulerConfig` and
/// `MergePolicyConfig`, from the dynamic `index.merge.*` settings: see
/// [`MergeConfig::from_settings`]).
#[derive(Clone, Debug, PartialEq)]
pub struct MergeConfig {
    /// `index.merge.scheduler.max_thread_count`: concurrent merges.
    pub max_thread_count: usize,
    /// `index.merge.scheduler.max_merge_count`: pending merges beyond which
    /// writes stall.
    pub max_merge_count: usize,
    /// `index.merge.scheduler.auto_throttle`.
    pub auto_throttle: bool,
    /// `index.merge.policy.segments_per_tier`.
    pub segments_per_tier: f64,
    /// `index.merge.policy.max_merge_at_once`.
    pub max_merge_at_once: usize,
    /// `index.merge.policy.floor_segment`, bytes.
    pub floor_segment_bytes: u64,
    /// `index.merge.policy.max_merged_segment`, bytes.
    pub max_merged_segment_bytes: u64,
    /// `index.merge.policy.deletes_pct_allowed`, percent.
    pub deletes_pct_allowed: f64,
    /// `index.merge.policy.expunge_deletes_allowed`, percent.
    pub expunge_deletes_allowed: f64,
    /// `index.merge.policy.type` (ES `MergePolicyConfig.Type`).
    pub policy_type: MergePolicyType,
    /// `index.merge.policy.merge_factor`: the time-based policy's (ES 32).
    pub merge_factor: usize,
    /// The time-based policy's max merged segment: an explicit
    /// `index.merge.policy.max_merged_segment`, else ES's 100gb
    /// (`indices.merge.policy.max_time_based_merged_segment`).
    pub time_based_max_merged_segment_bytes: u64,
    /// The copy's mapping had a `@timestamp` date field when it opened (ES
    /// decides at engine creation: `IndexShard.newEngineConfig` passes
    /// `MappingLookup.getTimestampFieldType() != null`). Not a setting.
    pub timestamp_field: bool,
}

/// ES `index.merge.policy.type`: `unset` (time-based when the mapping has
/// a `@timestamp` date field, else tiered), `tiered` or `time_based`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergePolicyType {
    #[default]
    Unset,
    Tiered,
    TimeBased,
}

impl Default for MergeConfig {
    /// ES 9 defaults (the settings registry's): 8 segments per tier, 16 at
    /// once, 16mb floor, 5gb max merged segment, 20% / 10% deletes.
    fn default() -> Self {
        let threads = merge_threads();
        MergeConfig { max_thread_count: threads, max_merge_count: threads + 5, auto_throttle: true,
            segments_per_tier: 8.0, max_merge_at_once: 16, floor_segment_bytes: 16 << 20, max_merged_segment_bytes: 5 << 30,
            deletes_pct_allowed: 20.0, expunge_deletes_allowed: 10.0,
            policy_type: MergePolicyType::Unset, merge_factor: 32, time_based_max_merged_segment_bytes: 100 << 30, timestamp_field: false }
    }
}

impl MergeConfig {
    /// Parse the `index.merge.*` settings `get` returns (ES names, units and
    /// ranges); an inverted thread / merge count pair is clamped
    /// (`max_merge_count` raised to `max_thread_count`) as ES 9 does at
    /// apply time. `RUSTIC_MERGE_THREADS` overrides the thread count.
    pub fn from_settings(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let d = MergeConfig::default();
        let int = |k: &str, min: usize| -> Result<Option<usize>> {
            match get(k) {
                None => Ok(None),
                Some(v) => {
                    let n: usize = v.trim().parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [{k}]")))?;
                    if n < min { return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [{k}] must be >= {min}"))); }
                    Ok(Some(n))
                }
            }
        };
        let float = |k: &str, min: f64, max: f64| -> Result<Option<f64>> {
            match get(k) {
                None => Ok(None),
                Some(v) => {
                    let n: f64 = v.trim().parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [{k}]")))?;
                    if !(min..=max).contains(&n) { return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [{k}] must be between [{min}] and [{max}]"))); }
                    Ok(Some(n))
                }
            }
        };
        let bytes = |k: &str| -> Result<Option<u64>> {
            match get(k) {
                None => Ok(None),
                Some(v) => Ok(Some(crate::ingest::util::parse_byte_size_value(&v, k)?.max(0) as u64)),
            }
        };
        let threads_env = std::env::var("RUSTIC_MERGE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0);
        let max_thread_count = threads_env.or(int("index.merge.scheduler.max_thread_count", 1)?).unwrap_or(d.max_thread_count);
        let max_merge_count = int("index.merge.scheduler.max_merge_count", 1)?.unwrap_or(max_thread_count + 5).max(max_thread_count);
        let auto_throttle = match get("index.merge.scheduler.auto_throttle") {
            None => d.auto_throttle,
            Some(v) => match v.trim() {
                "true" => true,
                "false" => false,
                _ => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
            },
        };
        // ES `Setting.enumSetting`: the name upper-cased (`Enum.valueOf`).
        let policy_type = match get("index.merge.policy.type") {
            None => d.policy_type,
            Some(v) => match v.trim().to_ascii_uppercase().as_str() {
                "UNSET" => MergePolicyType::Unset,
                "TIERED" => MergePolicyType::Tiered,
                "TIME_BASED" => MergePolicyType::TimeBased,
                other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.MergePolicyConfig.Type.{other}"))),
            },
        };
        // `0b` (ES's registered default) is unset: each policy's own default.
        let explicit_max_merged = bytes("index.merge.policy.max_merged_segment")?.filter(|b| *b > 0);
        Ok(MergeConfig {
            policy_type,
            merge_factor: int("index.merge.policy.merge_factor", 2)?.unwrap_or(d.merge_factor),
            time_based_max_merged_segment_bytes: explicit_max_merged.unwrap_or(d.time_based_max_merged_segment_bytes),
            timestamp_field: false,
            max_thread_count, max_merge_count, auto_throttle,
            segments_per_tier: float("index.merge.policy.segments_per_tier", 2.0, f64::MAX)?.unwrap_or(d.segments_per_tier),
            max_merge_at_once: int("index.merge.policy.max_merge_at_once", 2)?.unwrap_or(d.max_merge_at_once),
            floor_segment_bytes: bytes("index.merge.policy.floor_segment")?.unwrap_or(d.floor_segment_bytes),
            // `0b` (ES 9's registered default): the default 5gb.
            max_merged_segment_bytes: explicit_max_merged.unwrap_or(d.max_merged_segment_bytes),
            deletes_pct_allowed: float("index.merge.policy.deletes_pct_allowed", 5.0, 50.0)?.unwrap_or(d.deletes_pct_allowed),
            expunge_deletes_allowed: float("index.merge.policy.expunge_deletes_allowed", 0.0, 100.0)?.unwrap_or(d.expunge_deletes_allowed),
        })
    }

    /// The document-count policy of a core whose segments average
    /// `avg_doc_bytes` (byte sizes converted; at least one byte per doc).
    fn policy(&self, avg_doc_bytes: Option<f64>) -> rustic_core::merge::MergePolicy {
        let per_doc = avg_doc_bytes.unwrap_or(1024.0).max(1.0);
        let docs = |bytes: u64| ((bytes as f64 / per_doc).round() as u64).clamp(1, u64::from(u32::MAX)) as u32;
        rustic_core::merge::MergePolicy {
            max_segments_per_tier: (self.segments_per_tier.round() as usize).max(2),
            max_merge_at_once: self.max_merge_at_once.max(2),
            floor_segment_docs: docs(self.floor_segment_bytes),
            max_merged_docs: docs(self.max_merged_segment_bytes),
            deletes_pct_allowed: (self.deletes_pct_allowed / 100.0) as f32,
            // ES's TieredMergePolicy, in bytes (`find_merges_tiered`): no
            // segment budget, no cascade into one segment.
            max_segment_count: 0,
            tiered: Some(rustic_core::merge::TieredBytes {
                segments_per_tier: self.segments_per_tier,
                floor_segment_bytes: self.floor_segment_bytes,
                max_merged_segment_bytes: self.max_merged_segment_bytes,
            }),
            log_bytes: self.time_based().then_some(rustic_core::merge::LogBytes {
                merge_factor: self.merge_factor,
                min_merge_bytes: self.floor_segment_bytes,
                max_merge_bytes: self.time_based_max_merged_segment_bytes,
            }),
        }
    }

    /// Whether this copy merges with ES's time-based policy (Lucene's
    /// `LogByteSizeMergePolicy`; `MergePolicyConfig.Type.getMergePolicy`).
    pub fn time_based(&self) -> bool {
        match self.policy_type {
            MergePolicyType::TimeBased => true,
            MergePolicyType::Tiered => false,
            MergePolicyType::Unset => self.timestamp_field,
        }
    }
}

/// The merge slots of a shard copy ([`merge_threads`] permits).
pub(crate) struct MergeSlots(BgSemaphore);

impl Default for MergeSlots {
    fn default() -> Self { MergeSlots(BgSemaphore::new(merge_threads())) }
}

impl std::ops::Deref for MergeSlots {
    type Target = BgSemaphore;
    fn deref(&self) -> &BgSemaphore { &self.0 }
}

/// Held [`Shard::files_pinned`].
pub struct FilesPinned {
    _commit: BgGuard,
    _merge: BgGuard,
    _seal: BgGuard,
}

/// Own every not-yet-run foreground claim, including across fallible flush
/// preparation after the capture. Drop only cancels these unrun jobs: it never
/// touches an encoded/installed chunk or another background job's claims.
struct PendingSealChunks<'a> {
    shard: &'a Shard,
    jobs: std::iter::Peekable<std::vec::IntoIter<rustic_core::index::SealJob>>,
    /// What the capture held when claimed (reported by a slow flush).
    stats: SealCaptureStats,
}

#[derive(Clone, Copy, Default)]
struct SealCaptureStats {
    jobs: usize,
    memtables: usize,
    docs: u64,
    bytes: u64,
    coalesced: usize,
}

impl<'a> PendingSealChunks<'a> {
    fn new(shard: &'a Shard, jobs: Vec<rustic_core::index::SealJob>) -> Self {
        let stats = SealCaptureStats {
            jobs: jobs.len(),
            memtables: jobs.iter().map(|job| job.num_memtables()).sum(),
            docs: jobs.iter().map(|job| job.num_docs()).sum(),
            bytes: jobs.iter().map(|job| job.memtable_bytes()).fold(0, u64::saturating_add),
            coalesced: jobs.iter().filter(|job| job.coalesces()).count(),
        };
        Self { shard, jobs: jobs.into_iter().peekable(), stats }
    }

    /// Encode the chunks concurrently (at most
    /// [`CoreIndex::MAX_PARALLEL_SEALS`] encode threads together, as one
    /// job over the whole capture used), handing each to `install` (on
    /// this thread) as soon as it is encoded, so its memtables are released
    /// per chunk. An unstarted chunk is cancelled if this unwinds (`Drop`).
    fn run(&mut self, mut install: impl FnMut(rustic_core::index::SealedMemtables)) {
        let budget = CoreIndex::MAX_PARALLEL_SEALS;
        let mut panicked = None;
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut running = 0usize;
            loop {
                while let Some(width) = self.jobs.peek().map(|job| job.encode_parallelism()) {
                    if running > 0 && running + width > budget { break; }
                    let job = self.jobs.next().expect("peeked");
                    let tx = tx.clone();
                    running += width;
                    scope.spawn(move || {
                        // Always report (a panic too), so this loop never
                        // waits for a chunk that will not arrive.
                        // A flush or refresh waits for these chunks itself:
                        // encoded at normal priority, not on the niced
                        // background encode pool.
                        let sealed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rustic_core::foreground_encode(|| job.run())));
                        let _ = tx.send((width, sealed));
                    });
                }
                if running == 0 { break; }
                let (width, sealed) = rx.recv().expect("a running seal chunk reports");
                running -= width;
                match sealed {
                    Ok(sealed) => install(sealed),
                    Err(payload) => { panicked.get_or_insert(payload); }
                }
            }
        });
        if let Some(payload) = panicked { std::panic::resume_unwind(payload); }
    }
}

impl Drop for PendingSealChunks<'_> {
    fn drop(&mut self) {
        if self.jobs.len() == 0 { return; }
        let mut core = self.shard.core.write();
        for job in self.jobs.by_ref() { core.cancel_unrun_seal(job); }
    }
}

/// A background seal detached from its shard ([`Shard::start_seal`]).
pub(crate) struct SealTask {
    _guard: SemGuard,
    job: rustic_core::index::SealJob,
    /// Started for a replay's memtables: installed forced as well.
    forced: bool,
    /// Seal jobs of the copy encoding when this one started, itself
    /// included (throughput measurement).
    concurrent: usize,
}

impl SealTask {
    /// Encode (no shard lock, no shard reference needed).
    pub(crate) fn run(self) -> SealDone {
        let bytes = self.job.memtable_bytes();
        let t0 = std::time::Instant::now();
        let sealed = self.job.run();
        SealDone { _guard: self._guard, sealed, forced: self.forced, bytes, encoded: t0.elapsed(), concurrent: self.concurrent }
    }
}

/// An encoded [`SealTask`], to install with [`Shard::finish_seal`].
pub(crate) struct SealDone {
    _guard: SemGuard,
    sealed: rustic_core::index::SealedMemtables,
    forced: bool,
    /// Memtable bytes it sealed, its encoding time and concurrency.
    bytes: u64,
    encoded: std::time::Duration,
    concurrent: usize,
}

/// A background merge detached from its shard ([`Shard::start_merge`]).
pub(crate) struct MergeTask {
    /// The node-wide merge permit ([`node_merges`]).
    _node: SemGuard,
    _guard: SemGuard,
    job: rustic_core::index::MergeJob,
    docs: u64,
    t0: std::time::Instant,
}

impl MergeTask {
    pub(crate) fn run(self) -> MergeDone {
        let merged = self.job.run();
        MergeDone { task: self, merged }
    }
}

/// A finished [`MergeTask`], to install with [`Shard::finish_merge`].
pub(crate) struct MergeDone {
    task: MergeTask,
    merged: rustic_core::error::Result<rustic_core::segment::Segment>,
}

/// Memtables being written that make a seal due: every append-only batch
/// fills a fresh memtable (see `rustic_core::Index::checkout_writer`), so
/// concurrent auto-id bulks add memtables; past this many they are closed
/// and sealed (Lucene flushes each `DocumentsWriterPerThread` the same way)
/// and merged by the merge policy.
const MAX_OPEN_MEMTABLES: usize = 32;
/// Longest a force merge without a segment count waits for the policy's
/// merges (ES's request timeout bounds the same wait).
const POLICY_MERGE_WAIT: std::time::Duration = std::time::Duration::from_secs(2 * 60 * 60);

/// Writes of at least this many ops index into pooled memtables
/// ([`Shard::apply_staged`]): every write by default (pooled memtables are
/// reused across requests, so single-document writes do not create
/// memtables). `RUSTIC_STAGED_MIN_OPS` raises it (diagnostics).
fn staged_min_ops() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("RUSTIC_STAGED_MIN_OPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}


/// ES `TieredMergePolicy` defaults in docs (10 segments per tier).
fn merge_policy() -> rustic_core::merge::MergePolicy {
    rustic_core::merge::MergePolicy::default()
}

/// `(stored _source, vector of a core field) -> loaded _source` (`None`:
/// unchanged).
pub type SourcePatch = Arc<dyn Fn(&str, &dyn Fn(u16) -> Option<Vec<f32>>, &dyn crate::synthetic::DocColumns) -> Option<String> + Send + Sync>;

/// The dedicated pool of vector preparation (graph rows of bulk documents).
///
/// One bulk's graph inserts fan out over every core here, where ES indexes
/// a shard's bulk on one write thread: with a search running beside them,
/// equal-priority insert threads took half of each core the search ran on
/// (kNN p50 during indexing about twice its p50 alone). The threads run at
/// a lower priority (`RUSTIC_VECTOR_PREPARE_NICE`, default 5, `0` keeps
/// normal priority): searches take the cores they need first, inserts use
/// the rest.
#[cfg(feature = "mutable-hnsw")]
fn vector_prepare_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .num_threads(std::env::var("RUSTIC_VECTOR_PREPARE_THREADS").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from)))
        .start_handler(|_| lower_vector_prepare_priority())
        .thread_name(|i| format!("vector-prepare-{i}")).build().ok()).as_ref();
    if let Some(pool) = pool { rustic_core::vector::mutable_ingest::set_prepare_pool(pool); }
    pool
}

/// See [`vector_prepare_pool`].
#[cfg(feature = "mutable-hnsw")]
fn lower_vector_prepare_priority() {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        }
        let nice = std::env::var("RUSTIC_VECTOR_PREPARE_NICE").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(5);
        if nice != 0 {
            // PRIO_PROCESS with who = 0 addresses the calling thread on Linux.
            // SAFETY: a plain syscall wrapper; a failure leaves the priority as is.
            unsafe { let _ = setpriority(0, 0, nice.clamp(-20, 19)); }
        }
    }
}

/// The graph rows of a bulk's documents (`ops` are `WriteOp::Index`), on the
/// vector prepare pool: staged per field for the whole bulk
/// (`SegmentWriter::prepare_vector_documents`: one sequencer hold, one
/// parallel prepare pass and one commit per field, where staging each
/// document alone serialized a sequencer hold and a source commit per
/// document); if that fails, document by document, so each document gets
/// its own result.
#[cfg(feature = "mutable-hnsw")]
/// A core failure as a write item's error (shared by every item of a
/// batch, or carried by one item's prepared vectors): a refused memory
/// admission is a 429 `circuit_breaking_exception` with ES's fields, so
/// bulk clients (Rally, beats) back off and retry as from an ES breaker;
/// anything else stays a 500 with the core's message.
fn core_write_error(error: &rustic_core::RusticError) -> EsError {
    match error {
        rustic_core::RusticError::Memory(reason) => crate::memory::core_refusal(reason.clone()),
        other => EsError::internal(other.to_string()),
    }
}

fn prepare_bulk_vectors(ops: &[WriteOp], schema: &[CoreField], store: &Arc<rustic_core::vector::store::VectorStore>,
    admission: &rustic_core::admission::Context)
    -> Vec<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>
{
    let docs: Vec<&rustic_core::types::Document> = ops.iter().map(|op| match op {
        WriteOp::Index { doc, .. } => doc,
        _ => unreachable!("append-only bulks hold index operations"),
    }).collect();
    prepare_documents_vectors(&docs, schema, store, admission)
}

/// [`prepare_bulk_vectors`] of documents (each with vectors): staged per
/// field for all of them at once, else document by document. Every write
/// path that prepares a batch's graph rows goes through here: staging each
/// document by itself (an explicit-id bulk, a replica batch, a replay) took
/// the field sequencer, the graph writer twice (completion and a source
/// commit with its generation scan and reclamation) and a parallel pass of
/// one row per document, so a node's prepare threads queued on each shard's
/// locks (a 1B BIGANN load with explicit ids: 32 prepare threads at ~26%
/// CPU each, 67% idle, ~180k context switches/s, a third of ES's docs/s).
fn prepare_documents_vectors(docs: &[&rustic_core::types::Document], schema: &[CoreField],
    store: &Arc<rustic_core::vector::store::VectorStore>, admission: &rustic_core::admission::Context)
    -> Vec<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>
{
    use rayon::prelude::*;
    // Staged on this thread (a write thread): only the graph preparation
    // tasks run on the vector prepare pool (registered with rustic-core by
    // `vector_prepare_pool`), so this thread completes its own slices.
    vector_prepare_pool();
    {
        let _admission = admission.enter();
        if docs.len() > 1 {
            if let Ok(tokens) = rustic_core::segment::SegmentWriter::prepare_vector_documents(docs, schema, store) {
                return tokens.into_iter().map(Ok).collect();
            }
        }
    }
    on_vector_pool(|| docs.par_iter().map(|doc| {
        let _admission = admission.enter();
        rustic_core::segment::SegmentWriter::prepare_vector_document(doc, schema, store)
    }).collect())
}

/// The pool a large append-only batch inverts its parts on
/// ([`Shard::apply_append_only`]), through [`install_without_stealing`]:
/// inverting on the write pool itself let a writer waiting for its parts
/// steal another bulk's shard write, which then waited for the indexing
/// throttle (or `batches`) this same thread held: every bulk of the shard
/// stalled behind it (an eql load at 4.6M docs on a slow disk, once merges
/// fell behind and the throttle engaged).
fn append_invert_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, usize::from))
        .thread_name(|i| format!("append-invert-{i}")).build().ok()).as_ref()
}

/// Run `f` on `pool`. A caller on another rayon pool waits on a plain
/// scoped thread instead of `install`ing directly: a cross-pool install
/// steals the caller pool's work while waiting, which can re-enter a shard
/// lock the caller holds (`batches`, the write state, the core).
fn install_without_stealing<R: Send>(pool: &rayon::ThreadPool, f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() && pool.current_thread_index().is_none() {
        std::thread::scope(|scope| scope.spawn(|| pool.install(f)).join().expect("pool work panicked"))
    } else {
        pool.install(f)
    }
}

/// Run `f` (a replica batch's parallel source parsing) on a pool of its
/// own, never stealing ([`install_without_stealing`]); inline when the
/// pool could not be built.
pub(crate) fn on_replica_parse_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .num_threads((std::thread::available_parallelism().map_or(2, usize::from) / 2).max(1))
        .thread_name(|i| format!("replica-ops-parse-{i}")).build().ok());
    match pool {
        Some(pool) => install_without_stealing(pool, f),
        None => f(),
    }
}

/// [`install_without_stealing`] on the vector preparation pool (inline when
/// it could not be built).
#[cfg(feature = "mutable-hnsw")]
fn on_vector_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match vector_prepare_pool() {
        Some(pool) => install_without_stealing(pool, f),
        None => f(),
    }
}

fn core_dir(path: &Path, generation: u64) -> PathBuf {
    path.join(format!("core-{generation}"))
}

impl Shard {
    /// Create a new empty shard at `path` with the given core schema.
    pub fn create(path: &Path, name: &str, schema: Vec<CoreField>) -> Result<Shard> {
        let admitted_owner=rustic_core::admission::Owner::new();
        let _admission_scope=admitted_owner.enter();
        // ES 8.x BM25 carries the legacy (k1 + 1) factor.
        rustic_core::query::bm25::set_legacy_k1_boost(true);
        std::fs::create_dir_all(path)?;
        let mut core = CoreIndex::create(&core_dir(path, 0), name, schema, CoreSettings::default())?;
        core.set_nrt(true);
        core.set_auto_seal(false);
        #[cfg(feature = "mutable-hnsw")]
        core.enable_new_mutable_vector_fields()?;
        let translog = Translog::open(&path.join(TRANSLOG_DIR), 1)?;
        let st = ShardState { committed_seq_no: -1, translog_generation: 1, core_generation: 0, seqno: None, graph_checkpoint_seq_no: None };
        write_state(path, &st)?;
        Ok(Shard {
            admitted_owner,
            path: path.to_path_buf(),
            views: core.view_slot().clone(),
            core_schema: Mutex::new(Arc::new(core.schema.clone())),
            stored_compression: std::sync::atomic::AtomicU8::new(0),
            core: RwLock::new(core),
            write: Mutex::new(WriteState { translog: SharedTranslog::new(translog), seq_no: -1, committed_seq_no: -1, versions: registry::Registry::default(), version_inflight: 0, core_generation: 0, hist: hooks_ext::new_history(path, None, -1)?, staged: Default::default(), pending_docs: HashMap::default(), pending_inflight: 0, pending_memory: None, prunes: 0 }),
            synced: Mutex::new(0),
            tail: Mutex::new(None),
            counters: Counters::default(),
            last_flush_started: Mutex::new(std::time::Instant::now()),
            graph_saves: Mutex::new(GraphSaves::new(None)),
            ext: hooks_ext::ShardExt::open(path)?,
            sched: Default::default(),
            listeners: Default::default(),
            pool: Mutex::new(Default::default()),
            frozen: Mutex::new(Default::default()),
            checkin_lock: Mutex::new(()),
            freeze_gen: AtomicU64::new(0),
            holders: Mutex::new(Default::default()),
            holders_changed: parking_lot::Condvar::new(),
            next_holder: AtomicU64::new(0),
            pause_armed: std::sync::atomic::AtomicBool::new(false),
            pause_hook: Mutex::new(None),
            flush_pause: Mutex::new(None),
            inversion_panic: std::sync::atomic::AtomicBool::new(false),
            fail_syncs: std::sync::atomic::AtomicUsize::new(0),
            write_stats: Default::default(),
            buffered: Default::default(),
            patch: RwLock::new(None),
            bg: Default::default(),
            retention: Mutex::new(None),
            unresolved: Mutex::new(registry::Deletes::new()),
            sorted_window: Mutex::new(None),
            tragic: Mutex::new(None),
            id_locks: KeyedLocks::new(),
            sequencing_probe: Mutex::new(None),
            fail_older_deletes: std::sync::atomic::AtomicUsize::new(0),
            msu: std::sync::atomic::AtomicI64::new(-1),
            msu_from_primary: std::sync::atomic::AtomicBool::new(false),
            tragic_reported: std::sync::atomic::AtomicBool::new(false),
            batches: RwLock::new(()),
            files_gen: AtomicU64::new(0),
            store_cache: Mutex::new(None),
            source_sample: Mutex::new(None),
        })
    }

    /// Open an existing shard; returns it with the translog ops that must be
    /// replayed (seq_no above the commit point), all in memory at once
    /// (tests; store recovery streams them, [`Shard::open_for_replay`]).
    pub fn open(path: &Path, schema: Vec<CoreField>) -> Result<(Shard, crate::translog::AdmittedBatch<TranslogOp>)> {
        let (shard, replay) = Shard::open_for_replay(path, schema)?;
        let ops = replay.collect()?;
        Ok((shard, ops))
    }

    /// Open an existing shard; returns it with the translog ops that must be
    /// replayed (seq_no above the commit point), to be read a chunk at a time.
    pub fn open_for_replay(path: &Path, schema: Vec<CoreField>) -> Result<(Shard, crate::translog::TranslogReplay)> {
        let admitted_owner=rustic_core::admission::Owner::new();
        let _admission_scope=admitted_owner.enter();
        rustic_core::query::bm25::set_legacy_k1_boost(true);
        let st: ShardState = serde_json::from_slice(&std::fs::read(path.join(STATE_FILE))?)
            .map_err(|e| EsError::internal(format!("corrupt shard state: {e}")))?;
        let mut core = CoreIndex::open(&core_dir(path, st.core_generation))?;
        // Files no commit refers to (segments sealed or merged after the last
        // commit, interrupted writes) are dropped now: nothing else uses the
        // directory before this copy opens, and they would otherwise stay
        // on disk for good.
        let swept = core.sweep_orphan_files()?;
        if swept > 0 {
            tracing::info!(path = %path.display(), swept, "removed files no commit refers to");
        }
        core.set_nrt(true);
        core.set_auto_seal(false);
        // The mapping may know fields the last commit did not.
        for f in schema {
            if !core.schema.iter().any(|x| x.id == f.id) {
                core.schema.push(f);
            }
        }
        #[cfg(feature = "mutable-hnsw")]
        core.enable_new_mutable_vector_fields()?;
        let translog = Translog::open(&path.join(TRANSLOG_DIR), st.translog_generation)?;
        let hist = hooks_ext::new_history(path, st.seqno.as_ref(), st.committed_seq_no)?;
        let ops = hooks_ext::ops_to_replay(&translog, st.seqno.as_ref(), st.committed_seq_no)?;
        let mut max_seq = st.committed_seq_no;
        // Highest seq_no in the committed docs.
        for leaf in core.leaves() {
            if let Some(s) = leaf.max_i64_value(SEQ_NO_FIELD) {
                max_seq = max_seq.max(s);
            }
        }
        if let Some(&s) = hist.side.noops.keys().next_back() {
            max_seq = max_seq.max(s);
        }
        max_seq = max_seq.max(hist.checkpoint);
        let shard = Shard {
            admitted_owner,
            path: path.to_path_buf(),
            views: core.view_slot().clone(),
            core_schema: Mutex::new(Arc::new(core.schema.clone())),
            stored_compression: std::sync::atomic::AtomicU8::new(0),
            core: RwLock::new(core),
            write: Mutex::new(WriteState {
                translog: SharedTranslog::new(translog),
                seq_no: max_seq,
                committed_seq_no: st.committed_seq_no,
                versions: registry::Registry::default(), version_inflight: 0,
                core_generation: st.core_generation,
                hist,
                staged: Default::default(),
                pending_docs: HashMap::default(),
                pending_inflight: 0,
                pending_memory: None,
                prunes: 0,
            }),
            synced: Mutex::new(0),
            tail: Mutex::new(None),
            counters: Counters::default(),
            last_flush_started: Mutex::new(std::time::Instant::now()),
            graph_saves: Mutex::new(GraphSaves::new(st.graph_checkpoint_seq_no)),
            ext: hooks_ext::ShardExt::open(path)?,
            sched: Default::default(),
            listeners: Default::default(),
            pool: Mutex::new(Default::default()),
            frozen: Mutex::new(Default::default()),
            checkin_lock: Mutex::new(()),
            freeze_gen: AtomicU64::new(0),
            holders: Mutex::new(Default::default()),
            holders_changed: parking_lot::Condvar::new(),
            next_holder: AtomicU64::new(0),
            pause_armed: std::sync::atomic::AtomicBool::new(false),
            pause_hook: Mutex::new(None),
            flush_pause: Mutex::new(None),
            inversion_panic: std::sync::atomic::AtomicBool::new(false),
            fail_syncs: std::sync::atomic::AtomicUsize::new(0),
            write_stats: Default::default(),
            buffered: Default::default(),
            patch: RwLock::new(None),
            bg: Default::default(),
            retention: Mutex::new(None),
            unresolved: Mutex::new(registry::Deletes::new()),
            sorted_window: Mutex::new(None),
            tragic: Mutex::new(None),
            id_locks: KeyedLocks::new(),
            sequencing_probe: Mutex::new(None),
            fail_older_deletes: std::sync::atomic::AtomicUsize::new(0),
            // Unknown history: every op up to the last one may be an update.
            msu: std::sync::atomic::AtomicI64::new(max_seq),
            msu_from_primary: std::sync::atomic::AtomicBool::new(false),
            tragic_reported: std::sync::atomic::AtomicBool::new(false),
            batches: RwLock::new(()),
            files_gen: AtomicU64::new(0),
            store_cache: Mutex::new(None),
            source_sample: Mutex::new(None),
        };
        // Existing small segments need compaction even if this copy receives
        // no further writes (for example a retained benchmark or replica).
        shard.bg.want_merge.store(true, Ordering::Release);
        // Persist the format gate before this opened shard accepts new writes.
        write_state(path, &st)?;
        Ok((shard, ops))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What searches see now: the core's last published reader view. Takes
    /// no shard lock (the view slot's mutex is held only to clone an `Arc`);
    /// hold the view, not a lock, for the whole search.
    pub fn reader_view(&self) -> Arc<rustic_core::ReaderView> {
        self.views.load()
    }

    /// The view slot searches read (test fixtures that install their own
    /// core adopt it: `Index::adopt_view_slot`).
    #[doc(hidden)]
    pub fn view_slot(&self) -> &Arc<rustic_core::ViewSlot> {
        &self.views
    }

    /// A retained copy of the current view for a point in time, scroll or
    /// ES|QL capture (see [`rustic_core::ViewSlot::retain`]): no seal and no
    /// shard lock.
    pub fn retain_view(&self) -> Result<rustic_core::ReaderView> {
        if self.views.load().needs_sealed_capture() {
            // Mutable-graph points only in published memtables: the retained
            // view would search the live graph. Seal them first (a detached
            // seal: the core is held only to begin and install it).
            self.seal_published();
        }
        Ok(self.views.retain()?)
    }

    /// Add new core fields (mapping grew). The core schema is published as
    /// an immutable snapshot ([`Shard::core_schema`], swapped under the core
    /// write lock, which is held only for the in-memory schema update): a
    /// mapping change never takes `batches` or `seal_lock` and never waits
    /// for in-flight bulks. Bulks that started on the old snapshot finish on
    /// it (their documents simply lack the new fields); new bulks pick up
    /// the new one; fields added meanwhile reach checked-out memtables at
    /// check-in. Mapping changes are additive (as in ES), so segments and
    /// memtables written under an older schema stay readable.
    ///
    /// Vector backends: a new graph field gets an empty graph (no document
    /// can hold rows for a field id it did not know); an `index_options`
    /// update of a field with data only schedules its graph's rebuild
    /// (built in the background, see `apply_schema`).
    ///
    /// A failed application is not noted as synced: the next write or
    /// search applies it again (it is idempotent).
    pub fn sync_schema(&self, schema: &[CoreField]) -> Result<()> {
        if self.schema_needs_sync(schema) {
            let mut core = self.core.write();
            Self::apply_schema(&mut core, schema)?;
            self.note_core_schema(&core);
        }
        Ok(())
    }

    /// [`Shard::sync_schema`] for the search path: never blocks on a shard
    /// lock. Whether a sync is needed is read from the lock-free schema
    /// snapshot; the sync itself only try-locks the core. Returns whether
    /// the core schema is now in sync; when not, the search reads its view
    /// as is (searches read published leaves only: a field the core does
    /// not know yet simply has no data in them) and the next write or
    /// search syncs it.
    pub fn try_sync_schema(&self, schema: &[CoreField]) -> Result<bool> {
        if !self.schema_needs_sync(schema) {
            return Ok(true);
        }
        let Some(mut core) = self.core.try_write() else { return Ok(false) };
        Self::apply_schema(&mut core, schema)?;
        self.note_core_schema(&core);
        Ok(true)
    }

    /// New fields, or vector fields whose index options changed (against
    /// the lock-free copy of the core schema).
    fn schema_needs_sync(&self, schema: &[CoreField]) -> bool {
        let current = self.core_schema.lock().clone();
        schema.iter().any(|f| match current.iter().find(|x| x.id == f.id) {
            None => true,
            Some(x) => x.vector_options != f.vector_options || x.shape != f.shape,
        })
    }

    /// The core schema as last applied (lock-free copy).
    pub fn core_schema(&self) -> Arc<Vec<CoreField>> {
        self.core_schema.lock().clone()
    }

    /// Frame and compress the stored fields of memtables filled from now on
    /// as `mode` (`index.codec`; `best_compression` is ES's large zstd
    /// blocks). Applied to the core index's writers and to the pooled and
    /// private memtables the write paths fill.
    pub fn set_stored_compression(&self, mode: rustic_core::codec::store::StoredCompression) {
        let v = matches!(mode, rustic_core::codec::store::StoredCompression::Best) as u8;
        if self.stored_compression.swap(v, Ordering::AcqRel) != v || self.core.read().stored_compression() != mode {
            self.core.write().set_stored_compression(mode);
        }
    }

    /// See [`Shard::set_stored_compression`].
    pub fn stored_compression(&self) -> rustic_core::codec::store::StoredCompression {
        if self.stored_compression.load(Ordering::Acquire) == 1 {
            rustic_core::codec::store::StoredCompression::Best
        } else {
            rustic_core::codec::store::StoredCompression::Fast
        }
    }

    /// Refresh the lock-free copy of the core schema (call under the core
    /// write lock, after changing it).
    fn note_core_schema(&self, core: &CoreIndex) {
        *self.core_schema.lock() = Arc::new(core.schema.clone());
    }

    /// ES applies a `dense_vector` `index_options` update to the data a
    /// field already holds (`flat` → `hnsw` → `int8_hnsw` → `int4_hnsw` →
    /// `bbq_hnsw`, ...; the mapping layer rejects the paths ES disallows) and
    /// does no work at mapping time (`DenseVectorFieldMapper.updatableTo`).
    /// Here the field's new graph is only scheduled
    /// (`request_vector_migrations`): the background vector maintenance
    /// builds it off every shard lock from the current graph, catches it up
    /// with the writes that arrive meanwhile and swaps it in briefly; the
    /// current graph keeps admitting rows and serving searches until then.
    /// A new vector field gets its mapped options before its graph.
    fn apply_schema(core: &mut CoreIndex, schema: &[CoreField]) -> Result<()> {
        #[cfg(feature = "mutable-hnsw")]
        core.prepare_vector_schema_change(schema)?;
        #[cfg(feature = "mutable-hnsw")]
        let migrate = core.vector_fields_to_migrate(schema);
        let mut options_changed = false;
        for f in schema {
            match core.schema.iter_mut().find(|x| x.id == f.id) {
                None => {
                    options_changed |= f.vector_options.is_some();
                    core.schema.push(f.clone());
                }
                Some(x) if x.vector_options != f.vector_options => {
                    x.vector_options = f.vector_options;
                    x.shape = f.shape;
                    options_changed = true;
                }
                // A schema from before the shape flag: segments sealed from
                // now on build triangle trees for the field.
                Some(x) if x.shape != f.shape => x.shape = f.shape,
                Some(_) => {}
            }
        }
        if options_changed {
            core.sync_all_vector_field_options();
        }
        #[cfg(feature = "mutable-hnsw")]
        core.request_vector_migrations(&migrate)?;
        #[cfg(feature = "mutable-hnsw")]
        core.enable_new_mutable_vector_fields()?;
        Ok(())
    }

    /// A value that differs whenever the searchable data differs (shard
    /// request cache key): every sealed segment (ids are never reused; live
    /// counts only fall) and every memtable (doc counts only grow until it
    /// is sealed into a new segment), over a published reader view.
    pub fn data_fingerprint(view: &rustic_core::ReaderView) -> Vec<u64> {
        let segs = view.segments();
        let mut v = Vec::with_capacity(2 * segs.len() + 4);
        for s in segs {
            v.push(((s.id as u64) << 32) | s.num_docs as u64);
            v.push(s.num_live_docs() as u64);
        }
        for leaf in view.leaves().iter().skip(segs.len()) {
            v.push(u64::MAX);
            v.push(((leaf.num_docs() as u64) << 32) | leaf.num_live_docs() as u64);
        }
        v
    }

    pub fn max_seq_no(&self) -> i64 {
        self.write.lock().seq_no
    }

    /// The local checkpoint: every op at or below it is in the translog
    /// (appended) and applied.
    pub fn processed_checkpoint(&self) -> i64 {
        self.write.lock().hist.checkpoint
    }

    /// The refresh opening a copy published every op processed before it
    /// (`processed`, sampled first): its refreshed checkpoint starts there,
    /// whether or not that refresh found anything new to publish.
    pub(crate) fn note_opened_refresh(&self, processed: i64) {
        self.note_refreshed_checkpoint(processed);
    }

    /// Vector segments were sealed / merged since the last commit (a
    /// flush saves redoing that on reopen).
    pub fn has_uncommitted_vector_changes(&self) -> bool {
        self.core.read().vector_store().has_uncommitted_changes()
    }

    /// Bytes of committed segments that merges replaced since the last
    /// commit: they stay on disk until the next commit.
    pub fn merged_away_committed_bytes(&self) -> u64 {
        self.core.read().merged_away_committed_bytes()
    }

    /// Whether a commit would change anything: ops above the last commit,
    /// committed segments a merge replaced (their files stay until the next
    /// commit), or vector segment swaps (Lucene `hasUncommittedChanges`). A
    /// periodic flush skips only when none holds: a merge that finished after
    /// the last write must still be committed to free its inputs.
    pub fn has_uncommitted_changes(&self) -> bool {
        self.max_seq_no() > self.committed_seq_no() || self.merged_away_committed_bytes() > 0 || self.has_uncommitted_vector_changes()
    }

    /// Merges the policy wants that have not started, as last measured.
    pub fn queued_merges(&self) -> u64 {
        self.bg.merge_backlog.load(Ordering::Acquire) as u64
    }

    /// Whether no merge runs and the merge policy wants none (measured now).
    fn merges_settled(&self) -> bool {
        if self.counters.merge_current.load(Ordering::Acquire) > 0 {
            return false;
        }
        let config = self.bg.merge_config.read().clone();
        let pending = {
            let core = self.core.read();
            core.num_pending_merges(&config.policy(core.avg_doc_bytes()))
        };
        self.note_merge_backlog(pending);
        pending == 0 && self.counters.merge_current.load(Ordering::Acquire) == 0
    }

    /// Whether merges replaced committed segments and none is planned or
    /// running: ES `maybeFlushAfterMerge` / `flushOnIdle` commit then (Lucene
    /// `hasUncommittedChanges`), which frees the merged-away files.
    pub fn has_uncommitted_merges(&self) -> bool {
        self.bg.merge_backlog.load(Ordering::Acquire) == 0
            && self.counters.merge_current.load(Ordering::Relaxed) == 0
            && self.merged_away_committed_bytes() > 0
    }

    /// Delete merged-away segment files that neither a reader nor the last
    /// commit references (Lucene deletes these without a commit). Run
    /// periodically: a merge's inputs are still read when it installs.
    pub fn sweep_retired(&self) {
        let sweep = self.core.read().retired_sweep();
        match sweep() {
            Ok(0) => {}
            Ok(_) => self.files_changed(),
            Err(e) => tracing::warn!("deleting merged-away segment files of [{}] failed: {e}", self.path.display()),
        }
    }

    /// Find the live doc with `id`, realtime (every write so far, published
    /// or not): address (in [`CoreIndex::all_leaves`] order) plus its
    /// stored version/seq_no.
    pub fn lookup(core: &CoreIndex, id: &str) -> Option<(DocAddress, u64, i64)> {
        Self::lookup_in(&core.all_leaves(), id, true)
    }

    /// Find the live doc with `id` among `leaves`; `latest`: unpublished
    /// deletes count (realtime), else only what searches see.
    pub fn lookup_in(leaves: &[rustic_core::query::LeafReader<'_>], id: &str, latest: bool) -> Option<(DocAddress, u64, i64)> {
        Self::lookup_in_impl(leaves, id, latest, false)
    }

    fn lookup_in_impl(leaves: &[rustic_core::query::LeafReader<'_>], id: &str, latest: bool, append_check: bool) -> Option<(DocAddress, u64, i64)> {
        // The newest live copy by seq_no. Realtime (`latest`) lookups can
        // meet an older copy whose buffered delete is not resolved yet
        // (`Shard::buffer_delete`), in any leaf order, so every leaf is
        // looked at; visible lookups never do (a refresh resolves them
        // first), and one live copy is the common case.
        let term = Term::new_text(ID_FIELD, id);
        // Hashed once for every leaf's id filter.
        let hash = rustic_core::codec::terms::negative_filter_hash(&term.text);
        let mut best: Option<(DocAddress, u64, i64)> = None;
        for (i, leaf) in leaves.iter().enumerate().rev() {
            match leaf {
                rustic_core::query::LeafReader::Segment(segment) => {
                    if (append_check || latest) && !segment.may_contain_unique(ID_FIELD, hash) { continue; }
                }
                // A memtable's own id filter: one word instead of a probe
                // of its whole term table (most ids are in no memtable).
                rustic_core::query::LeafReader::Memtable(writer, ..) => {
                    if !writer.may_contain_unique_hash(ID_FIELD, hash) { continue; }
                }
            }
            if let Some(mut p) = leaf.postings(&term) {
                while let Some(d) = p.next() {
                    let dead = if latest { leaf.is_deleted_latest(d) } else { leaf.is_deleted(d) };
                    if dead { continue; }
                    let s = leaf.i64_value(SEQ_NO_FIELD, d).unwrap_or(0);
                    if best.as_ref().is_none_or(|b| s > b.2) {
                        let v = leaf.i64_value(VERSION_FIELD, d).unwrap_or(1) as u64;
                        best = Some((DocAddress::new(i as u32, d), v, s));
                    }
                }
            }
            if best.is_some() && !latest { break; }
        }
        best
    }

    /// Apply a batch of writes. Each op gets its own result; the batch is
    /// logged with one translog write and one fsync (when `sync`).
    pub fn apply(&self, index: &str, uuid: &str, ops: Vec<WriteOp>, sync: bool) -> Vec<Result<WriteResult>> {
        let _admission_scope=self.admitted_owner.enter();
        // A failed engine fails every op at once (ES `ensureOpen`).
        if let Some(error) = self.tragic_error() {
            return ops.iter().map(|_| Err(Self::tragic_exception(&error))).collect();
        }
        // Merge throttling adapts only while the node indexes.
        rustic_core::merge::note_indexing();
        // Held for the indexing, released before the translog fsync.
        let stall_started = std::time::Instant::now();
        let _throttle = HeldThrottle::hold(self.stall_if_behind());
        let stalled = stall_started.elapsed();
        if stalled > std::time::Duration::from_millis(50) {
            tracing::debug!(target: "write_stall", stall_ms = stalled.as_millis() as u64, unsealed = self.unsealed_bytes(), limit = self.stall_limit_bytes(),
                memory_throttled = self.bg.memory_throttled.load(Ordering::Acquire), merge_backlog = self.bg.merge_backlog.load(Ordering::Acquire), "write waited before indexing");
        }
        self.files_changed();
        #[cfg(feature = "mutable-hnsw")]
        if ops.len() > 1
            && ops.iter().any(|op| matches!(op, WriteOp::Index { doc, .. } if doc.has_vectors()))
            && ops.iter().all(|op| matches!(op, WriteOp::Index { auto_id: true, .. })) {
            return self.apply_mutable_append_only(ops, sync);
        }
        if ops.len() > 1 && ops.iter().all(|o| matches!(o, WriteOp::Index { auto_id: true, doc, .. } if !doc.has_vectors())) {
            return self.apply_append_only(ops, sync);
        }
        if Self::staged_eligible(&ops) {
            return self.apply_staged(index, uuid, ops, sync);
        }
        let n_idx_ops = ops.iter().filter(|o| matches!(o, WriteOp::Index { .. })).count() as u64;
        let _current = CurrentGuard::new(vec![(&self.counters.index_current, n_idx_ops), (&self.counters.delete_current, ops.len() as u64 - n_idx_ops)]);
        let preparation_started = std::time::Instant::now();
        // Explicit-id bulks of new vector documents: prepare the graph rows
        // before taking the write state and the core (under `batches`, like
        // append-only batches, so a schema change waits for them); the
        // locked section below only verifies the ids are still new and
        // installs in seq_no order. Tokens of a batch that turns out not to
        // be new are dropped, which rolls their reservations back.
        #[cfg(feature = "mutable-hnsw")]
        let _vector_batch = Self::fresh_mutable_batch_candidate(&ops).then(|| self.batches.read());
        #[cfg(feature = "mutable-hnsw")]
        let early = if _vector_batch.is_some() { self.prepare_mutable_tokens(&ops) } else { None };
        let mut ws = self.write.lock();
        let mut core = self.core.write();
        #[cfg(feature = "mutable-hnsw")]
        let mut vector_prepared = early.filter(|_| Self::fresh_ids(&ws, &core, &ops)).into_iter().flatten();
        let mut results = Vec::with_capacity(ops.len());
        let mut log = Vec::with_capacity(ops.len());
        let mut n_index = 0u64;
        let mut n_delete = 0u64;
        let mut index_time = preparation_started.elapsed();
        let mut delete_time = std::time::Duration::ZERO;
        for op in ops {
            let t0 = std::time::Instant::now();
            let sequence_before = ws.seq_no;
            let r = match op {
                WriteOp::Index { id, source, doc, ignored, routing, cond, auto_id } => {
                    let r = self.apply_index(&mut ws, &mut core, index, uuid, id, source, doc, ignored, routing, cond, auto_id, &mut log,
                        #[cfg(feature = "mutable-hnsw")]
                        vector_prepared.next());
                    index_time += t0.elapsed();
                    if r.is_ok() {
                        n_index += 1;
                    } else {
                        self.counters.index_failed.fetch_add(1, Ordering::Relaxed);
                    }
                    r
                }
                WriteOp::Delete { id, cond } => {
                    let r = self.apply_delete(&mut ws, &mut core, index, uuid, id, cond, &mut log);
                    delete_time += t0.elapsed();
                    if r.is_ok() {
                        n_delete += 1;
                    }
                    r
                }
            };
            if r.is_err() { ws.seq_no = sequence_before; }
            results.push(r);
        }
        let t_log = std::time::Instant::now();
        if self.buffer_full(core.writing_bytes() as u64) || core.num_writing_docs() >= self.seal_threshold() || core.num_open_memtables() > MAX_OPEN_MEMTABLES {
            if self.bg.served.load(Ordering::Acquire) {
                // Sealed by a background worker (see `Background`).
                self.bg.want_seal.store(true, Ordering::Release);
            } else if let Err(e) = core.seal() {
                tracing::error!("seal failed: {e}");
            } else {
                self.note_buffered(core.writing_bytes() as u64);
            }
            // Live entries of the version map only mirror what the id lookup
            // returns from the core (every leaf, memtables included; ES prunes
            // its `LiveVersionMap` on refresh): keep the delete tombstones, as
            // a flush does, so the map stays bounded by the memtables rather
            // than growing with every id written since the last flush.
            ws.prune_versions(&self.unresolved.lock());
        }
        if n_index + n_delete > 0 {
            self.sched.note_write();
        }
        // Searches only need the core: release it before the translog write
        // (the write lock still orders appends by seq_no).
        drop(core);
        let mut sync_point = None;
        let mut durability_error = None;
        if !log.is_empty() {
            match ws.translog.append_many(&log) {
                Err(error) => {
                    self.mark_tragic(&format!("translog append failed: {error}"));
                    durability_error = Some(error.to_string());
                }
                Ok(()) => {
                    let sp = ws.translog.sync_point();
                    *self.tail.lock() = Some(sp.clone());
                    for op in &log { ws.hist.mark(op.seq_no() as i64); }
                    if sync { sync_point = Some(sp); }
                }
            }
        }
        drop(ws);
        if let Some((file, mark)) = sync_point {
            if let Err(error) = self.group_sync(&file, mark) {
                durability_error = Some(error.to_string());
            }
        }
        // A failed durable write can have reached memory or a partial WAL;
        // never acknowledge it as safely persisted. Preserve per-item parse /
        // admission failures rather than replacing them with this batch error.
        if let Some(error) = durability_error {
            for result in &mut results {
                if result.is_ok() { *result = Err(EsError::internal(format!("translog durability failure: {error}"))); }
            }
        }
        // Translog/fsync/seal cost is shared by the batch's ops pro rata.
        let shared = t_log.elapsed();
        let n = (n_idx_ops + n_delete).max(1) as u32;
        if n_idx_ops > 0 {
            index_time += shared * n_idx_ops as u32 / n;
        }
        if n_delete > 0 {
            delete_time += shared * n_delete as u32 / n;
        }
        self.counters.index_total.fetch_add(n_index, Ordering::Relaxed);
        self.counters.delete_total.fetch_add(n_delete, Ordering::Relaxed);
        Counters::add_time(&self.counters.index_time_nanos, index_time);
        Counters::add_time(&self.counters.delete_time_nanos, delete_time);
        self.files_changed();
        results
    }

    /// Record this shard's bytes in the memtables being written; returns the
    /// node-wide total and the number of shards holding any.
    fn note_buffered(&self, bytes: u64) -> (u64, usize) {
        let old = self.buffered.swap(bytes, Ordering::Relaxed);
        crate::memory::IndexingBuffer::global().update(old, bytes)
    }

    /// ES `IndexingMemoryController`: the memtables share the node's
    /// indexing buffer (10% of the memory budget). A copy the scheduler
    /// serves never seals itself for it: a write that sees the node's
    /// memtables being written over the buffer asks the scheduler's memory
    /// check ([`crate::refresh::check_indexing_buffer`]) to run now, which
    /// writes out the largest copies' buffers. A copy no scheduler serves
    /// seals when it holds at least its share.
    fn buffer_full(&self, writing: u64) -> bool {
        self.buffer_full_with(writing, 0)
    }

    /// [`Shard::buffer_full`] with `pooled` bytes of pooled memtables on
    /// top: they count toward the decision, but only `writing` is reported
    /// to the node-wide indexing buffer (pooled memtables carry their own
    /// accounting charge until checked in).
    fn buffer_full_with(&self, writing: u64, pooled: u64) -> bool {
        let (total, active) = self.note_buffered(writing);
        let total = total + pooled;
        let writing = writing + pooled;
        if total <= crate::memory::indexing_buffer() {
            return false;
        }
        if self.bg.served.load(Ordering::Acquire) {
            crate::refresh::request_indexing_buffer_check();
            return false;
        }
        writing >= total / active.max(1) as u64
    }

    /// This copy's indexing-buffer bytes without waiting on the core: every
    /// memtable not sealed yet (being written, published, parked, pooled),
    /// and the part of it no seal job claimed. `None`: the core is busy.
    pub(crate) fn indexing_buffer_bytes(&self) -> Option<(u64, u64)> {
        let core = self.core.try_read()?;
        let (unsealed, unclaimed) = (core.num_unsealed_bytes(), core.num_unclaimed_bytes());
        drop(core);
        let pooled = self.pooled_bytes();
        Some((unsealed + pooled, unclaimed + pooled))
    }

    /// ES IMC `activateThrottling` / `deactivateThrottling`.
    pub(crate) fn set_memory_throttled(&self, on: bool) {
        self.bg.memory_throttled.store(on, Ordering::Release);
    }

    /// Write this copy's whole indexing buffer out in the background (ES
    /// `IndexShard::writeIndexingBuffer`): completed writes checked in, the
    /// memtables being written closed, everything sealed in coalesced jobs
    /// (one segment per job). Visibility is unchanged.
    pub(crate) fn write_indexing_buffer(&self) {
        self.bg.idle_seal.store(true, Ordering::Release);
        self.bg.want_seal.store(true, Ordering::Release);
    }

    /// Docs in the memtables being written that make a seal due.
    pub(crate) fn seal_threshold(&self) -> u32 {
        match self.bg.threshold.load(Ordering::Relaxed) {
            0 => SEAL_THRESHOLD_DOCS,
            n => n,
        }
    }

    /// Write backpressure, as ES applies it, before a write runs:
    /// - Lucene's flush-control stall: while the memtables not yet sealed
    ///   (being written, published, closed for a seal, pooled) hold more than
    ///   twice this shard's share of the indexing buffer
    ///   (`indices.memory.index_buffer_size`), the write waits (holding no
    ///   lock) for the seals to catch up;
    /// - ES `IndexingThrottle`: while merges are more than
    ///   `index.merge.scheduler.max_merge_count` behind, indexing on this
    ///   shard runs one request at a time (the returned guard, held for the
    ///   write) instead of stalling.
    pub(crate) fn stall_if_behind(&self) -> Option<BgGuard> {
        self.stall_if_behind_for(std::time::Duration::from_secs(60))
    }

    /// Admission of one replicated batch (a transport request or a
    /// recovery batch), before the replica takes any lock or counts the
    /// batch as applying: the byte stall waits at most [`REPLICA_STALL`]
    /// once per batch (the engine calls that apply it do not stall), so
    /// local sealing never makes a primary's round trip time out. A
    /// replica never rejects for its indexing buffer (ES: only the node's
    /// in-flight replica bytes past 1.5x the indexing-pressure limit
    /// reject a replica op).
    pub fn admit_replicated(&self) {
        // A buffer past its share is written out before the batch's WAL
        // append; the shard throttle is taken by the engine call that
        // applies the batch ([`Shard::merge_throttle`]).
        self.buffer_pressure();
        if self.stalls_writes() {
            self.wait_for_seals(self.stall_limit_bytes(), REPLICA_STALL);
        }
    }

    /// Whether the byte stall applies (a copy that serves searches, or a
    /// test's stall limit).
    fn stalls_writes(&self) -> bool {
        self.bg.served.load(Ordering::Acquire) || self.bg.stall_limit_override.load(Ordering::Acquire) > 0
    }

    /// Wait (holding no lock) at most `max_wait` while the unsealed bytes
    /// are over `limit`, asking for the whole buffer to be written out.
    fn wait_for_seals(&self, limit: u64, max_wait: std::time::Duration) {
        let t0 = std::time::Instant::now();
        while self.unsealed_bytes() > limit && t0.elapsed() < max_wait {
            // Write the whole buffer out (memtables being written too).
            self.bg.idle_seal.store(true, Ordering::Release);
            self.bg.want_seal.store(true, Ordering::Release);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// ES `IndexingThrottle`: one indexing call at a time on this shard
    /// while merges are more than `max_merge_count` behind, while the node's
    /// indexing-buffer check throttles the copy, or while this copy's own
    /// buffer is past 1.5x its share ([`Shard::buffer_pressure`]) (the guard
    /// is held for the call). Taken by every live write: a primary's
    /// ([`Shard::stall_if_behind`]) and a replica's engine apply.
    pub(crate) fn merge_throttle(&self) -> Option<BgGuard> {
        let (max_merge_count, auto_throttle) = { let c = self.bg.merge_config.read(); (c.max_merge_count, c.auto_throttle) };
        let memory = self.bg.memory_throttled.load(Ordering::Acquire);
        let in_flight = self.bg.merge_backlog.load(Ordering::Acquire) + self.counters.merge_current.load(Ordering::Acquire) as usize;
        let merges = merge_throttle_due(in_flight, max_merge_count, auto_throttle, rustic_core::merge::merge_auto_rate_at_max());
        if merges {
            self.bg.want_merge.store(true, Ordering::Release);
        }
        let buffer = self.buffer_pressure();
        let on = memory || merges || buffer;
        self.counters.index_throttled.store(u64::from(on), Ordering::Relaxed);
        on.then(|| {
            let waited = std::time::Instant::now();
            let guard = self.bg.indexing_throttle.lock();
            Counters::add_time(&self.counters.index_throttle_nanos, waited.elapsed());
            guard
        })
    }

    /// This copy's share of the node's indexing buffer
    /// (`indices.memory.index_buffer_size` over the copies holding buffered
    /// writes, at least 16 MB): half the write stall's limit.
    fn buffer_share_bytes(&self) -> u64 {
        self.stall_limit_bytes() / 2
    }

    /// ES `IndexingMemoryController`, per copy, on the live write path of
    /// primaries and replicas (before each write, never waiting on the
    /// core). Once the copy's unclaimed buffer (memtables no seal job has
    /// claimed, pooled ones included) is past its share of the node's
    /// buffer, the whole buffer is written out now (ES writes out the
    /// largest buffers first; a copy past its share is one of them) rather
    /// than at the next node-wide check, about a second later, on every copy
    /// of a busy node. Returns whether the copy's writes are throttled to
    /// one at a time (ES `activateThrottling`): its unsealed bytes, claimed
    /// ones included, are past 1.5x its share, so its seals are behind;
    /// lifted once they catch up. The write stall
    /// ([`Shard::stall_if_behind`], twice the share) stays the hard bound.
    /// A recovery replay (WAL at open, peer recovery, promotion) writes its
    /// buffer out the same way but is never throttled (ES exempts recovery
    /// origins; its seals are forced). Copies no scheduler serves seal on
    /// their own write path ([`Shard::buffer_full`]).
    fn buffer_pressure(&self) -> bool {
        if !self.bg.served.load(Ordering::Acquire) {
            return false;
        }
        let Some((unsealed, unclaimed)) = self.indexing_buffer_bytes() else { return false };
        let share = self.buffer_share_bytes();
        if unclaimed > share && !self.bg.idle_seal.load(Ordering::Acquire) {
            self.write_indexing_buffer();
            crate::refresh::request_indexing_buffer_check();
        }
        unsealed > share.saturating_add(share / 2) && !crate::memory::is_recovery_replay()
    }

    fn stall_if_behind_for(&self, max_wait: std::time::Duration) -> Option<BgGuard> {
        if !self.stalls_writes() {
            return None;
        }
        self.wait_for_seals(self.stall_limit_bytes(), max_wait);
        self.merge_throttle()
    }

    /// Unsealed bytes at which writes stall: twice this shard's share of the
    /// node's indexing buffer (Lucene stalls at twice the RAM buffer of
    /// flush-pending segments), or, when lower, what the copy's seals write
    /// in [`REFRESH_SEAL_TARGET`] ([`backlog_limit`]): a copy whose seals
    /// are slower than its ingest (a slow disk) holds a few seconds of seal
    /// work, so its next refresh (the first after a refresh-disabled load)
    /// seals that and publishes segments in seconds, and its writes wait
    /// in shorter, more even stalls instead of long ones at a multi-GB
    /// limit. Ingest throughput is the seal throughput either way.
    fn stall_limit_bytes(&self) -> u64 {
        let set = self.bg.stall_limit_override.load(Ordering::Acquire);
        if set > 0 {
            return set;
        }
        let active = crate::memory::IndexingBuffer::global().active().max(1) as u64;
        let memory = (crate::memory::indexing_buffer() / active).max(16 << 20).saturating_mul(2);
        backlog_limit(memory, self.bg.seal_rate.load(Ordering::Acquire))
    }

    /// Bytes not in a segment yet: the core's memtables of every kind (being
    /// written, published, closed for a seal) and the pooled ones. Every
    /// refresh moves the memtables being written to the published ones, so
    /// counting only those never stalled.
    fn unsealed_bytes(&self) -> u64 {
        self.core.read().num_unsealed_bytes() + self.pooled_bytes()
    }

    /// Background work is due (see [`Shard::run_background`]).
    pub(crate) fn background_due(&self) -> bool {
        self.bg.want_seal.load(Ordering::Acquire) || self.bg.want_merge.load(Ordering::Acquire) || self.bg.want_refresh.load(Ordering::Acquire)
            || self.bg.want_repair.load(Ordering::Acquire)
    }

    /// Detach a due seal for a background worker (the core write lock is
    /// held only to detach). `None` when nothing is due, a foreground seal
    /// holds the lock (it seals everything anyway), or the memtables seal
    /// in place (vectors / index sort: done here, as before).
    /// A recovery replay of acknowledged ops (replica WAL or local translog
    /// replay at open, peer recovery, promotion) wrote to this copy: noted
    /// so its background seals run forced. A live replicated batch (forced
    /// reservations past its WAL append, `memory::Replay::live`) is not
    /// noted: it is ordinary replication, throttled at the primary by
    /// indexing pressure and on this copy by its write throttle and stall,
    /// with ordinary seals (forcing them for 30 s after every batch kept a
    /// bulk-loaded replica's seals forced for the whole load).
    fn note_replay_write(&self) {
        if crate::memory::is_recovery_replay() {
            self.bg.replay_write_nanos.store(seal_clock_nanos().max(1), Ordering::Release);
        }
    }

    /// Whether this copy's background seals run forced: a recovery replay
    /// wrote to it within [`REPLAY_SEAL_FORCED_WINDOW`]. Its seals run on
    /// the refresh workers, outside the replay's thread, so they would
    /// otherwise capture an ordinary context and be refused over the limit
    /// while the forced replay keeps filling memtables (unbounded memory).
    pub fn replay_seals_forced(&self) -> bool {
        let at = self.bg.replay_write_nanos.load(Ordering::Acquire);
        at != 0 && seal_clock_nanos().saturating_sub(at) < REPLAY_SEAL_FORCED_WINDOW.as_nanos() as u64
    }

    pub(crate) fn start_seal(&self) -> Option<SealTask> {
        // One of `seal_jobs()` permits: background seals of disjoint memtable
        // sets encode concurrently; half as many while the node searches
        // (their encoding then also runs on the half-size encode pool).
        self.bg.seal_running.set_max(if rustic_core::searches_active() { (seal_jobs() / 2).max(1) } else { seal_jobs() });
        // A refused seal admission backs off (retried after the delay, not on
        // every tick and every stalled write's nudge): another attempt now
        // would only take the core lock to be refused again.
        if seal_clock_nanos() < self.bg.seal_backoff_until.load(Ordering::Acquire) {
            self.bg.want_seal.store(true, Ordering::Release);
            return None;
        }
        let Some(guard) = self.bg.seal_running.try_acquire() else {
            // Every job slot is encoding: a buffer write-out still pending
            // is retried (the scheduler re-dispatches due copies each tick).
            if self.bg.idle_seal.load(Ordering::Acquire) || self.bg.draining.load(Ordering::Acquire) {
                self.bg.want_seal.store(true, Ordering::Release);
            }
            return None;
        };
        // Not while a flush pins the segment set (it seals itself); retried.
        let Some(_pin) = self.bg.seal_lock.try_lock() else {
            self.bg.want_seal.store(true, Ordering::Release);
            return None;
        };
        // A search-idle copy's buffer: its completed writes join the core
        // (visibility unchanged; never waiting long on a holder) and the
        // memtables being written are closed into this seal.
        let idle = self.bg.idle_seal.swap(false, Ordering::AcqRel);
        let seal_started = std::time::Instant::now();
        if idle {
            self.checkin_completed_writes(Some(std::time::Duration::from_millis(50)));
        }
        let checked_in = seal_started.elapsed();
        let draining = idle || self.bg.draining.swap(false, Ordering::AcqRel);
        let mut core = self.core.write();
        let core_locked = seal_started.elapsed();
        let close = idle || core.num_writing_docs() >= self.seal_threshold() || self.buffer_full(core.writing_bytes() as u64) || core.num_open_memtables() > MAX_OPEN_MEMTABLES
            // Writes are stalled on the backlog: seal all of it.
            || core.num_unsealed_bytes() + self.pooled_bytes() > self.stall_limit_bytes();
        if !close && !draining && !core.frozen_over_limit() {
            return None;
        }
        // One coalesced segment per job (however many small memtables the
        // buffer holds); the rest goes to the next job.
        // Memtables a replay of acknowledged ops is filling are sealed as
        // part of it: the job captures a forced context (its admission, the
        // encode and the install are charged, never refused), or the replay
        // would keep growing memtables nothing can write out.
        let forced = self.replay_seals_forced();
        let begun = if forced { rustic_core::admission::forced(|| core.begin_seal_bytes(close, CoreIndex::COALESCE_MAX_BYTES)) }
            else { core.begin_seal_bytes(close, CoreIndex::COALESCE_MAX_BYTES) };
        if seal_started.elapsed() > std::time::Duration::from_millis(20) {
            tracing::debug!(target: "seal_profile", idle, close, checkin_ms = checked_in.as_millis() as u64, lock_ms = (core_locked - checked_in).as_millis() as u64,
                begin_ms = (seal_started.elapsed() - core_locked).as_millis() as u64, "seal start");
        }
        match begun {
            Err(error) => {
                // Retried after a growing delay; a buffer write-out stays due.
                let refusals = self.bg.seal_refusals.fetch_add(1, Ordering::AcqRel).min(16);
                let delay = SEAL_REFUSAL_BACKOFF_MIN.saturating_mul(1 << refusals.min(5)).min(SEAL_REFUSAL_BACKOFF_MAX);
                self.bg.seal_backoff_until.store(seal_clock_nanos().saturating_add(delay.as_nanos() as u64), Ordering::Release);
                if idle { self.bg.idle_seal.store(true, Ordering::Release); }
                if draining { self.bg.draining.store(true, Ordering::Release); }
                self.bg.want_seal.store(true, Ordering::Release);
                if refusals == 0 {
                    tracing::warn!("seal admission for [{}] failed: {error}",self.path.display());
                }
                None
            }
            Ok(Some(job)) => {
                self.bg.seal_refusals.store(0, Ordering::Release);
                self.note_buffered(core.writing_bytes() as u64);
                // More left unclaimed: another worker starts the next job
                // (a buffer write-out takes all of it).
                if core.num_unclaimed_memtables() > 0 {
                    if draining { self.bg.draining.store(true, Ordering::Release); }
                    self.bg.want_seal.store(true, Ordering::Release);
                }
                Some(SealTask { _guard: guard, job, forced, concurrent: self.bg.seal_running.held().max(1) })
            }
            // Everything is claimed by in-flight jobs: never seal inline
            // under the core lock beside them.
            Ok(None) if core.seal_in_flight() => None,
            Ok(None) => {
                let seal = |core: &mut CoreIndex| if close { core.seal() } else { core.seal_frozen() };
                let r = if forced { rustic_core::admission::forced(|| seal(&mut *core)) } else { seal(&mut *core) };
                self.note_buffered(core.writing_bytes() as u64);
                drop(core);
                if let Err(e) = r {
                    tracing::error!("seal of [{}] failed: {e}", self.path.display());
                }
                self.bg.want_merge.store(true, Ordering::Release);
                None
            }
        }
    }

    /// The fences of a foreground seal that must run its own job: no seal
    /// encoding (`seal_running`) and the segment set pinned (`seal_lock`).
    fn seal_fences(&self) -> (SemGuard, BgGuard) {
        let running = self.bg.seal_running.acquire_all();
        (running, self.bg.seal_lock.lock())
    }

    /// [`Shard::seal_fences`] if both are free now.
    fn try_seal_fences(&self) -> Option<(SemGuard, BgGuard)> {
        let running = self.bg.seal_running.try_acquire_all()?;
        let seal = self.bg.seal_lock.try_lock()?;
        Some((running, seal))
    }

    /// Install an encoded seal.
    pub(crate) fn finish_seal(&self, done: SealDone) {
        self.files_changed();
        // The install alone excludes a flush's capture / commit.
        let install = || self.install_seal_retrying(done.sealed, |core, sealed| {
            core.try_finish_seal_captured(sealed).map(|r| r.map(|_| ()))
        }, true);
        let (bytes, encoded, concurrent) = (done.bytes, done.encoded, done.concurrent);
        let installed = if done.forced { rustic_core::admission::forced(install) } else { install() };
        match installed {
            // Jobs of the copy encode concurrently: the copy's throughput is
            // about this job's rate times the jobs running beside it.
            Ok(()) => self.note_seal_rate(bytes.saturating_mul(concurrent as u64), encoded),
            Err(e) => tracing::error!("background seal of [{}] failed: {e}", self.path.display()),
        }
        self.bg.want_merge.store(true, Ordering::Release);
    }

    /// Record a seal throughput sample (`bytes` of memtables in `took`):
    /// an exponentially weighted average, in bytes per second, that caps the
    /// copy's unsealed backlog ([`backlog_limit`]). Tiny samples are
    /// skipped (timer noise).
    fn note_seal_rate(&self, bytes: u64, took: std::time::Duration) {
        if bytes < (256 << 10) || took.is_zero() {
            return;
        }
        let sample = (bytes as u128 * 1_000_000 / took.as_micros().max(1)).min(u64::MAX as u128) as u64;
        let old = self.bg.seal_rate.load(Ordering::Acquire);
        let new = if old == 0 { sample } else { (old / 4).saturating_mul(3).saturating_add(sample / 4) };
        self.bg.seal_rate.store(new.max(1), Ordering::Release);
    }

    /// Install an encoded seal with `attempt` (under the core write lock,
    /// and `seal_lock` when `pin`), retrying a refused install's admission a
    /// few times, briefly, off both locks before giving the encoded work up
    /// ([`CoreIndex::abandon_seal`]: its memtables stay for a later seal).
    /// The refused job keeps its claims meanwhile, so the retries are short.
    fn install_seal_retrying<R>(&self, mut sealed: rustic_core::index::SealedMemtables,
        mut attempt: impl FnMut(&mut CoreIndex, rustic_core::index::SealedMemtables) -> std::result::Result<rustic_core::error::Result<R>, rustic_core::index::SealInstallRefused>,
        pin: bool) -> rustic_core::error::Result<R> {
        let mut delays = SEAL_INSTALL_RETRY_DELAYS.iter();
        loop {
            let refused = {
                let t = std::time::Instant::now();
                let (_pin, mut core) = self.pinned_core(pin);
                let locked = t.elapsed();
                // The memtables an install replaces are freed after the
                // locks, with the refresh's other pooled copies when a
                // refresh runs it ([`CHECKIN_DROPS`]).
                let (attempted, retired) = rustic_core::index::collect_retired_writers(|| attempt(&mut core, sealed));
                let installed = t.elapsed();
                drop(core);
                drop(_pin);
                retire_checked_in(retired);
                match attempted {
                    Ok(result) => {
                        tracing::debug!(target: "seal_profile", lock_ms = locked.as_millis() as u64, install_ms = (installed - locked).as_millis() as u64,
                            unlock_ms = (t.elapsed() - installed).as_millis() as u64, "seal install");
                        return result;
                    }
                    Err(refused) => refused,
                }
            };
            match delays.next() {
                Some(delay) => {
                    sealed = refused.sealed;
                    std::thread::sleep(*delay);
                }
                None => {
                    let _pin = pin.then(|| self.bg.seal_lock.lock());
                    self.core.write().abandon_seal(refused.sealed);
                    return Err(refused.error);
                }
            }
        }
    }

    /// The seal lock (when `pin`) and the core write lock, in that order,
    /// without holding the seal lock while waiting long for the core: a
    /// seal install that waited for the core (a merge install, a flush
    /// capture, a write batch holding it) kept the seal lock meanwhile, so
    /// a flush waiting for the seal lock waited for the core too (4.4 s on
    /// the c28p wiki mixed load). The pin is retried once the core is free.
    fn pinned_core(&self, pin: bool) -> (Option<BgGuard>, parking_lot::RwLockWriteGuard<'_, CoreIndex>) {
        if !pin {
            return (None, self.core.write());
        }
        loop {
            let pinned = self.bg.seal_lock.lock();
            if let Some(core) = self.core.try_write_for(std::time::Duration::from_millis(5)) {
                return (Some(pinned), core);
            }
            drop(pinned);
            drop(self.core.write());
        }
    }

    /// Detach the next merge the merge policy finds, for a background
    /// worker. `None` when there is none or a merge is running.
    pub(crate) fn start_merge(&self) -> Option<MergeTask> {
        // One of `merge_threads()` slots (ES ConcurrentMergeScheduler): the
        // policy picks merges of disjoint, unclaimed segments. Claiming
        // installs nothing, so it may overlap a flush (merges encode on their
        // own pool and never compete for the encode threads seals use).
        // A node-wide merge permit first: background merges of all copies
        // together use at most half the cores, and half of that while the
        // node searches (search latency first, as the encode pool).
        let node = node_merges();
        node.set_max(if rustic_core::searches_active() { (node_merge_cap() / 2).max(1) } else { node_merge_cap() });
        let Some((node_guard, guard)) = node.try_acquire().and_then(|n| self.bg.merge_running.try_acquire().map(|g| (n, g))) else {
            // Every merge slot is busy: the request stays (a finishing merge
            // or the scheduler's next dispatch picks it up), and the backlog
            // is still measured: it drives the merge throttle and the
            // indexing throttle (`max_merge_count`) while merges fall behind.
            self.bg.want_merge.store(true, Ordering::Release);
            let config = self.bg.merge_config.read().clone();
            if let Some(core) = self.core.try_read() {
                let backlog = core.num_pending_merges(&config.policy(core.avg_doc_bytes()));
                drop(core);
                self.note_merge_backlog(backlog);
            }
            return None;
        };
        let config = self.bg.merge_config.read().clone();
        let (job, backlog) = {
            let mut core = self.core.write();
            let policy = config.policy(core.avg_doc_bytes());
            let job = core.begin_merge(&policy);
            // Counted running before the claim is visible: claimed inputs
            // are no longer pending, so a waiter for "no merge pending or
            // running" (a force merge before its commit, Rally's
            // wait-until-merges-finish) must already see this one.
            if let Some(job) = &job {
                self.counters.merge_current.fetch_add(1, Ordering::Relaxed);
                self.counters.merge_current_docs.fetch_add(job.num_docs(), Ordering::Relaxed);
            }
            (job, core.num_pending_merges(&policy))
        };
        self.note_merge_backlog(backlog);
        // More to merge: another worker starts the next merge meanwhile. A
        // backlog without a job means planning admission was denied: the
        // request stays for the scheduler's next dispatch.
        if backlog > 0 { self.bg.want_merge.store(true, Ordering::Release); }
        let mut job = job?;
        job.set_throttled(config.auto_throttle);
        let docs = job.num_docs();
        Some(MergeTask { _guard: guard, _node: node_guard, job, docs, t0: std::time::Instant::now() })
    }

    /// This copy's merges the policy wants that have not started.
    fn note_merge_backlog(&self, backlog: usize) {
        let old = self.bg.merge_backlog.swap(backlog, Ordering::AcqRel);
        rustic_core::merge::note_pending_merges(old, backlog);
    }

    /// Install a finished merge (counted in the merge stats). Returns
    /// whether another merge may be due.
    pub(crate) fn finish_merge(&self, done: MergeDone) -> bool {
        self.files_changed();
        let MergeDone { task, merged } = done;
        // Running until installed (or released): a merge counted done before
        // its install let a force merge commit first and return with the
        // merge then installing uncommitted.
        let _current = CurrentGuard(vec![(&self.counters.merge_current, 1), (&self.counters.merge_current_docs, task.docs)]);
        let merged = match merged {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("background merge of [{}] failed: {e}", self.path.display());
                self.core.write().release_merge(&task.job);
                return false;
            }
        };
        let bytes = merged.file_bytes() as u64;
        // The install alone excludes a flush's commit (encoding did not).
        let wait_start = std::time::Instant::now();
        let installed = {
            let _pin = self.bg.merge_lock.lock();
            let pinned = wait_start.elapsed();
            let mut core = self.core.write();
            let locked = wait_start.elapsed();
            let installed = core.finish_merge(task.job, merged);
            let policy = self.bg.merge_config.read().policy(core.avg_doc_bytes());
            self.note_merge_backlog(core.num_pending_merges(&policy));
            let held = wait_start.elapsed() - locked;
            // Searches read the published view without the core lock, but
            // refreshes, writes and seal installs wait while it is held.
            if held >= SLOW_INSTALL || locked >= SLOW_INSTALL {
                tracing::warn!(path = %self.path.display(), merge_lock_wait_ms = pinned.as_millis() as u64, core_lock_wait_ms = (locked - pinned).as_millis() as u64,
                    core_lock_held_ms = held.as_millis() as u64, merged_bytes = bytes, docs = task.docs, "slow merge install");
            }
            installed
        };
        match installed {
            Ok(true) => {
                self.counters.merge_total.fetch_add(1, Ordering::Relaxed);
                self.counters.merge_docs.fetch_add(task.docs, Ordering::Relaxed);
                self.counters.merge_size_bytes.fetch_add(bytes, Ordering::Relaxed);
                Counters::add_time(&self.counters.merge_time_nanos, task.t0.elapsed());
                true
            }
            Ok(false) => false,
            Err(e) => {
                tracing::error!("background merge of [{}] failed: {e}", self.path.display());
                false
            }
        }
    }

    /// Background passes until `done` holds, for at most 60 s (tests). A
    /// merge needs one of the node-wide merge permits every copy of the
    /// process shares: in a parallel test run other tests' merges can hold
    /// them all (and the cap halves while any test searches), so a pass may
    /// start nothing for a while; the request stays set, as in a node, where
    /// the scheduler's tick retries it. Waiting on the state, not on a count
    /// of passes, keeps such tests deterministic under load.
    #[cfg(test)]
    pub(crate) fn background_until(&self, mut done: impl FnMut(&Shard) -> bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if done(self) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            if self.background_due() {
                self.run_background();
                if done(self) {
                    return true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// [`Shard::background_until`] nothing is due and no merge runs.
    #[cfg(test)]
    pub(crate) fn settle_background(&self) -> bool {
        self.background_until(|s| !s.background_due() && Counters::get(&s.counters.merge_current) == 0)
    }

    /// [`Shard::start_merge`], retried for up to 60 s while the node-wide
    /// merge permits are busy (see [`Shard::background_until`]).
    #[cfg(test)]
    pub(crate) fn start_merge_waiting(&self) -> Option<MergeTask> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if let Some(t) = self.start_merge() {
                return Some(t);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Merge until no merge runs and the policy wants none (bounded by
    /// [`POLICY_MERGE_WAIT`]: writes may keep making merges due). Merges this
    /// thread can claim run here; while background workers hold the merge
    /// slots it sleeps, holding no lock.
    fn run_policy_merges(&self) {
        let deadline = std::time::Instant::now() + POLICY_MERGE_WAIT;
        loop {
            while let Some(t) = self.start_merge() {
                if !self.finish_merge(t.run()) {
                    break;
                }
            }
            if self.merges_settled() || std::time::Instant::now() >= deadline {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// One background pass run in place (tests, and copies no worker
    /// serves): seal what is due, then merge until the policy is satisfied.
    pub(crate) fn run_background(&self) {
        if self.bg.want_refresh.swap(false, Ordering::AcqRel) {
            self.refresh_deferred();
        }
        if self.bg.want_repair.swap(false, Ordering::AcqRel) {
            self.repair_vectors();
        }
        if self.bg.want_seal.swap(false, Ordering::AcqRel) {
            if let Some(t) = self.start_seal() {
                self.finish_seal(t.run());
            }
        }
        if self.bg.want_merge.swap(false, Ordering::AcqRel) {
            while let Some(t) = self.start_merge() {
                if !self.finish_merge(t.run()) {
                    break;
                }
            }
        }
    }

    /// A batch of new documents with auto-generated ids (no earlier copy
    /// can exist: no version lookup, nothing to delete), indexed like a
    /// Lucene `DocumentsWriterPerThread`: seq_nos are assigned and the ops
    /// logged under the write lock, then the documents are added to a
    /// memtable checked out of the core, without any lock, so concurrent
    /// batches to one shard index in parallel; the memtable is checked
    /// back in before the batch is acknowledged (realtime from then on,
    /// searchable at the next refresh, as any write).
    fn prepare_claim_node()->Result<rustic_core::admission::Reservation> {
        // One B-tree node per prospective key is a conservative bound even
        // during a split; the guard remains with the installed claim.
        Ok(rustic_core::admission::shared().reserve(2048,"resident staged claim node")?)
    }

    fn apply_append_only(&self, ops: Vec<WriteOp>, sync: bool) -> Vec<Result<WriteResult>> {
        /// Fewest docs per memtable a batch is split into, and most memtables.
        const APPEND_PART_DOCS: usize = 250;
        const APPEND_PARTS: usize = 4;
        /// Fewest source bytes per memtable a batch is split into.
        const APPEND_PART_BYTES: usize = 1 << 20;
        static APPEND_PARTS_ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        static APPEND_PART_BYTES_ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let n = ops.len();
        let claim_memory=match Self::prepare_claim_node() {
            Ok(memory)=>memory,Err(error)=>return (0..n).map(|_|Err(error.clone())).collect(),
        };
        let _current = CurrentGuard::new(vec![(&self.counters.index_current, n as u64)]);
        let t0 = std::time::Instant::now();
        let _batch = self.batches.read();
        // Sequence and log (cheap, ordered).
        let primary_term = self.primary_term();
        let mut docs: Vec<(String, String, Document, Vec<String>, Option<String>)> = Vec::with_capacity(n);
        for op in ops {
            if let WriteOp::Index { id, source, doc, ignored, routing, .. } = op {
                docs.push((id, source, doc, ignored, routing));
            }
        }
        // A large batch inverts into several private memtables at once (its
        // documents in contiguous runs, one per memtable), as Lucene's
        // DWPTs spread one shard's indexing over threads: the inversion is
        // the batch's serial part otherwise (k8s_metrics: a sixth of its
        // CPU, on the critical path of every bulk). Seals coalesce them.
        //
        // Only batches with enough work per memtable split: a split batch
        // checks in several small memtables, and the open-memtable cap then
        // seals them into small segments that merges rewrite again. A
        // shard's share of a bulk of tiny documents (http_logs: ~1,000
        // documents of ~140 bytes per shard, five shards a bulk, eight
        // bulks at once) already runs beside the other shards' batches, so
        // splitting it adds no parallelism, only 4x the memtables and
        // segments. Batches of at least a megabyte a part (metricbeat:
        // 10,000 x ~1 KB; k8s_metrics: 1,000 x ~10 KB) keep their split.
        let max_parts = *APPEND_PARTS_ENV.get_or_init(|| std::env::var("RUSTIC_APPEND_PARTS").ok().and_then(|v| v.parse().ok()).unwrap_or(APPEND_PARTS));
        let part_bytes = (*APPEND_PART_BYTES_ENV.get_or_init(|| std::env::var("RUSTIC_APPEND_PART_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(APPEND_PART_BYTES))).max(1);
        let bytes: usize = docs.iter().map(|(_, source, ..)| source.len()).sum();
        // A throttled write indexes on its own thread only (ES
        // `IndexingThrottle`: one thread indexes while merges are behind),
        // so the merges it waits on get the cores back.
        let parts = if n >= 2 * APPEND_PART_DOCS && !HeldThrottle::held() { (n / APPEND_PART_DOCS).min(bytes / part_bytes).min(max_parts).max(1) } else { 1 };
        let per = n.div_ceil(parts);
        // Checkout returns only empty writers, so a refusal drops no
        // acknowledged documents. Capacity admission precedes sequencing.
        let (mut writers, schema) = {
            let mut core = self.core.write();
            let mut writers = Vec::with_capacity(parts);
            let mut schema = Vec::new();
            for _ in 0..parts {
                let (w, s) = core.checkout_writer();
                writers.push(w);
                schema = s;
            }
            (writers, schema)
        };
        for (w, chunk) in writers.iter_mut().zip(docs.chunks(per)) {
            let bitmap_admission = chunk.iter().try_fold(0usize, |count, (_, _, doc, _, _)| {
                count.checked_add(rustic_core::segment::SegmentWriter::document_slots(doc)?)
                    .ok_or_else(|| rustic_core::error::RusticError::Memory("batch document count overflow".into()))
            }).and_then(|count| w.prepare_bitmap_capacity(count));
            if let Err(error) = bitmap_admission {
                let mut core = self.core.write();
                for w in writers {
                    core.checkin_writer(w);
                }
                let error = EsError::from(error);
                return (0..n).map(|_| Err(error.clone())).collect();
            }
        }
        let (first, sync_point) = {
            let mut ws = self.write.lock();
            let first = ws.seq_no + 1;
            ws.seq_no += n as i64;
            let log: Vec<TranslogOp> = docs
                .iter()
                .enumerate()
                .map(|(i, (id, source, _, _, routing))| TranslogOp::Index { id: id.clone(), source_json: source.clone(), seq_no: (first + i as i64) as u64, version: 1, routing: routing.clone() })
                .collect();
            if let Err(error) = ws.translog.append_many(&log) {
                // Never in the translog: nothing is indexed, replicated (the
                // tailer reads the translog) or acknowledged; the seq_nos
                // become no-ops (no hole) and the engine fails itself (ES
                // translog tragic event -> failEngine).
                self.mark_tragic(&format!("translog append failed: {error}"));
                let seq_nos: Vec<i64> = (first..first + n as i64).collect();
                let _ = ws.fill_noops(&seq_nos, primary_term, "translog append failed", false);
                drop(ws);
                self.counters.index_failed.fetch_add(n as u64, Ordering::Relaxed);
                let error = error.to_string();
                return (0..n).map(|_| Err(EsError::internal(format!("translog durability failure: {error}")))).collect();
            }
            let sp = ws.translog.sync_point();
            *self.tail.lock() = Some(sp.clone());
            // Not in the core until the check-in below.
            ws.claim(first, claim_memory);
            (first, sp)
        };
        // Index into the private, preadmitted memtables (one run each).
        let index_run = |w: &mut rustic_core::segment::SegmentWriter, start: usize, run: Vec<(String, String, Document, Vec<String>, Option<String>)>| {
            let mut out = Vec::with_capacity(run.len());
            for (i, (id, source, mut doc, ignored, routing)) in run.into_iter().enumerate() {
                let seq_no = first + (start + i) as i64;
                doc.fields.push((ID_FIELD, FieldValue::Keyword(id.clone())));
                // The WAL keeps the original source even when source storage
                // is disabled. Respect the parser's explicit empty-source marker.
                if !doc.fields.iter().any(|(field, _)| *field == SOURCE_FIELD) {
                    doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
                }
                doc.fields.push((VERSION_FIELD, FieldValue::I64(1)));
                doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
                doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(primary_term as i64)));
                if let Some(r) = routing {
                    doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r)));
                }
                for f in ignored {
                    doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
                }
                w.add_document(&doc, &schema);
                out.push(Ok(WriteResult { id, version: 1, seq_no, primary_term, result: "created" }));
            }
            out
        };
        let mut runs: Vec<Vec<(String, String, Document, Vec<String>, Option<String>)>> = Vec::with_capacity(parts);
        let mut docs = docs.into_iter();
        for _ in 0..parts {
            runs.push(docs.by_ref().take(per).collect());
        }
        let results: Vec<Result<WriteResult>> = if parts == 1 {
            index_run(&mut writers[0], 0, runs.pop().unwrap_or_default())
        } else {
            use rayon::prelude::*;
            let admission = rustic_core::admission::Context::capture();
            // Never on the write pool: this thread holds `batches` and maybe
            // the indexing throttle while it waits (see `append_invert_pool`).
            let invert = || writers.par_iter_mut().zip(runs.into_par_iter()).enumerate()
                .map(|(p, (w, run))| admission.run(|| index_run(w, p * per, run)))
                .collect::<Vec<_>>();
            match append_invert_pool() {
                Some(pool) => install_without_stealing(pool, invert),
                // No pool: a plain thread's parallel work goes to the global
                // pool, and this thread only waits for it.
                None => std::thread::scope(|scope| scope.spawn(invert).join().expect("append inversion panicked")),
            }.into_iter().flatten().collect()
        };
        {
            let mut core = self.core.write();
            for w in writers {
                core.checkin_writer(w);
            }
            if self.buffer_full(core.writing_bytes() as u64) || core.num_writing_docs() >= self.seal_threshold() || core.num_open_memtables() > MAX_OPEN_MEMTABLES {
                if self.bg.served.load(Ordering::Acquire) {
                    self.bg.want_seal.store(true, Ordering::Release);
                } else if let Err(e) = core.seal() {
                    tracing::error!("seal failed: {e}");
                }
            }
        }
        {
            let mut ws = self.write.lock();
            for s in first..first + n as i64 {
                ws.hist.mark(s);
            }
            ws.release_claim(first);
        }
        drop(_batch);
        self.sched.note_write();
        if sync {
            // A failed fsync fails the engine (`group_sync` marks it
            // tragic): the ops are not durable here, so none is acknowledged.
            if let Err(error) = self.group_sync(&sync_point.0, sync_point.1) {
                self.counters.index_failed.fetch_add(n as u64, Ordering::Relaxed);
                self.files_changed();
                let error = error.to_string();
                return (0..n).map(|_| Err(EsError::internal(format!("translog durability failure: {error}")))).collect();
            }
        }
        self.counters.index_total.fetch_add(n as u64, Ordering::Relaxed);
        Counters::add_time(&self.counters.index_time_nanos, t0.elapsed());
        self.files_changed();
        results
    }

    /// Whether [`Shard::apply_staged`] takes `ops`: any write of at least
    /// [`staged_min_ops`] ops (repeated ids and retried ops included).
    /// Vector documents need the mutable graph backend (their rows are
    /// prepared before sequencing); without it they index in place.
    fn staged_eligible(ops: &[WriteOp]) -> bool {
        if ops.is_empty() || ops.len() < staged_min_ops() { return false; }
        #[cfg(not(feature = "mutable-hnsw"))]
        if ops.iter().any(|op| matches!(op, WriteOp::Index { doc, .. } if doc.has_vectors())) { return false; }
        true
    }

    /// Concurrent DWPT-style indexing (Lucene `DocumentsWriterPerThread` /
    /// ES `InternalEngine`): every staged write request indexes into a
    /// pooled per-thread memtable ([`PoolWriter`]) it holds exclusively, so
    /// concurrent requests on one shard invert in parallel; no shard-wide
    /// lock is held across a request.
    ///
    /// 1. Vector rows are prepared (mutable graphs; tokens own rollback),
    ///    and the current live copy of each explicit id is looked up under
    ///    a core *read*.
    /// 2. Sequencing, under the write state only (short: hash lookups and
    ///    the translog append): version checks against the latest state
    ///    (version map first, so a concurrent op on the same id sequenced
    ///    earlier is seen), seq_no assignment, version-map update, one
    ///    translog append in seq_no order. This is the per-op serialization
    ///    point ES gets from its per-uid locks.
    /// 3. Off every lock: each sequenced document gets its metadata fields
    ///    and is inverted into a pooled memtable taken from the pool.
    /// 4. Completion (write state, briefly the core for deletes): realtime
    ///    GET serves the new documents from the version map's pending
    ///    entries until their memtable is checked in; deletes apply to the
    ///    core at once; the seq_nos are marked processed (local checkpoint,
    ///    completing out of order across requests); the memtable returns to
    ///    the pool (checked in once it is large).
    ///
    /// Pooled memtables are checked into the index by every refresh, flush
    /// and realtime GET of a pending id ([`Shard::checkin_pool`]); the
    /// check-in deletes the previous live copies of each id in seq_no order
    /// and installs a document superseded by a later-sequenced op on its id
    /// deleted, so there is never more than one live copy and no acked
    /// document is unaddressable (#63). `batches` is held shared
    /// throughout, so a commit never fixes a seq_no whose document is not
    /// in the core.
    fn apply_staged(&self, index: &str, uuid: &str, mut ops: Vec<WriteOp>, sync: bool) -> Vec<Result<WriteResult>> {
        let _shard_owner=self.admitted_owner.enter();
        let admission_context=rustic_core::admission::Context::capture();
        enum Planned {
            // The complete document (metadata fields included), shared with
            // the version map's pending entry from sequencing on.
            Index { delete_key: registry::DeleteKey, id: String, seq_no: i64, older: bool, pending: Arc<PendingDoc>,
                #[cfg(feature = "mutable-hnsw")]
                token: Option<rustic_core::segment::PreparedDocumentVectors> },
            Delete { id: String, seq_no: i64 },
        }
        let n = ops.len();
        let claim_memory=match Self::prepare_claim_node() {
            Ok(memory)=>memory,Err(error)=>return (0..n).map(|_|Err(error.clone())).collect(),
        };
        let n_idx_ops = ops.iter().filter(|o| matches!(o, WriteOp::Index { .. })).count() as u64;
        let _current = CurrentGuard::new(vec![(&self.counters.index_current, n_idx_ops), (&self.counters.delete_current, n as u64 - n_idx_ops)]);
        let started = std::time::Instant::now();
        let _batch = self.batches.read();
        for op in &mut ops {
            if let WriteOp::Index { doc, id, source, routing, ignored, .. } = op {
                if let Err(error) = PendingDoc::prepare_document(doc, id, source, routing.as_ref(), ignored) {
                    return (0..n).map(|_| Err(error.clone())).collect();
                }
            }
        }
        let _pending_slots = match PendingMapSlots::new(self, n_idx_ops as usize) {
            Ok(slots) => slots,
            Err(error) => return (0..n).map(|_| Err(error.clone())).collect(),
        };
        let _version_slots = match VersionMapSlots::new(self, n) {
            Ok(slots) => slots, Err(error) => return (0..n).map(|_| Err(error.clone())).collect(),
        };
        let version_keys = match registry::Keys::prepare(ops.iter().map(|op| match op {
            WriteOp::Index { id, .. } | WriteOp::Delete { id, .. } => id.as_str(),
        })) { Ok(keys) => keys, Err(error) => return (0..n).map(|_| Err(error.clone())).collect() };
        let delete_keys = match registry::DeleteKeys::prepare(ops.iter().map(|op| match op { WriteOp::Index { id, .. } | WriteOp::Delete { id, .. } => id.as_str() })) {
            Ok(keys) => keys, Err(error) => return (0..n).map(|_|Err(error.clone())).collect(),
        };
        let documents = ops.iter().try_fold(0usize, |count, op| {
            let slots = match op { WriteOp::Index { doc, .. } => rustic_core::segment::SegmentWriter::document_slots(doc)?, _ => 0 };
            count.checked_add(slots).ok_or_else(|| rustic_core::error::RusticError::Memory("batch document count overflow".into()))
        });
        let prepared_pool = match documents.map_err(EsError::from).and_then(|count| PreparedPool::new(self, count)) {
            Ok(pool) => pool,
            Err(error) => return (0..n).map(|_| Err(error.clone())).collect(),
        };
        let schema = self.core_schema();
        // 1a. Graph rows, before sequencing (only admitted rows enter the WAL).
        #[cfg(feature = "mutable-hnsw")]
        let mut tokens: Vec<Option<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>> = {
            let vectors = ops.iter().any(|op| matches!(op, WriteOp::Index { doc, .. } if doc.has_vectors()));
            if vectors {
                // Shared access in the common case (every graph field has
                // its graph): an exclusive core lock per vector batch made
                // every concurrent search wait behind each bulk. Only a new
                // graph field takes the core exclusively.
                let ready = {
                    let core = self.core.read();
                    core.mutable_vector_fields_ready().then(|| core.vector_store().clone())
                };
                let store = match ready {
                    Some(store) => store,
                    None => {
                        let core = self.core.write();
                        if let Err(error) = core.enable_new_mutable_vector_fields() {
                            return (0..n).map(|_| Err(core_write_error(&error))).collect();
                        }
                        core.vector_store().clone()
                    }
                };
                let schema = &schema;
                let store = &store;
                // On the dedicated pool, never stealing (this thread holds
                // `batches.read()`; see `apply_mutable_append_only`).
                // The bulk's vector documents staged together
                // (`prepare_documents_vectors`), not one by one.
                let (positions, docs): (Vec<usize>, Vec<&rustic_core::types::Document>) = ops.iter().enumerate()
                    .filter_map(|(i, op)| match op { WriteOp::Index { doc, .. } if doc.has_vectors() => Some((i, doc)), _ => None })
                    .unzip();
                let prepared = prepare_documents_vectors(&docs, schema, store, &admission_context);
                let mut tokens: Vec<Option<_>> = (0..n).map(|_| None).collect();
                for (i, token) in positions.into_iter().zip(prepared) { tokens[i] = Some(token); }
                tokens
            } else {
                (0..n).map(|_| None).collect()
            }
        };
        // 1b. The live copy of each explicit id (and its primary term when a
        // condition needs it), under a core read: the version map overrides
        // it in step 2 when a later op on the id was sequenced meanwhile.
        let lookup_all = |core: &CoreIndex, ops: &[WriteOp]| -> Vec<(Option<(DocAddress, u64, i64)>, u64)> {
            let leaves = core.all_leaves();
            ops.iter().map(|op| {
                let (id, cond, auto) = match op {
                    WriteOp::Index { id, cond, auto_id, .. } => (id, cond, *auto_id),
                    WriteOp::Delete { id, cond } => (id, cond, false),
                };
                if auto { return (None, 1); }
                let hit = Self::lookup_in(&leaves, id, true);
                let term = match (&hit, cond.if_primary_term) {
                    (Some((addr, _, _)), Some(_)) => hooks_ext::doc_term(core, addr),
                    _ => 1,
                };
                (hit, term)
            }).collect()
        };
        // Per-id locks (ES `KeyedLock` on the uid), held from the version
        // checks through the translog append and the realtime entries:
        // each id's ops are sequenced and appended in order, while batches
        // of other ids proceed concurrently.
        let stripes = self.id_locks.lock_all(ops.iter().map(|op| match op {
            WriteOp::Index { id, .. } | WriteOp::Delete { id, .. } => id.as_str(),
        }));
        let mut prunes = self.write.lock().prunes;
        let mut found = { let core = self.core.read(); lookup_all(&core, &ops) };
        // A check-in pruned the version map meanwhile (every refresh): look
        // the live copies up again off the write state, so the locked
        // re-lookup below is rare (5,000 time series ids looked up under it
        // serialised every concurrent bulk of the shard: Rally tsdb).
        for _ in 0..3 {
            let now = self.write.lock().prunes;
            if now == prunes {
                break;
            }
            prunes = now;
            found = lookup_all(&self.core.read(), &ops);
        }
        let probe = self.sequencing_probe.lock().clone();
        if let Some(probe) = probe { probe(); }
        // 2. Sequencing.
        let mut results: Vec<Result<WriteResult>> = Vec::with_capacity(n);
        let mut planned = Vec::with_capacity(n);
        let mut log = Vec::with_capacity(n);
        let primary_term = self.primary_term();
        // Sequenced but not yet built: the document gets its metadata fields
        // after the write state is released.
        struct Raw {
            delete_key: registry::DeleteKey,
            id: String, seq_no: i64, older: bool, doc: Document, version: u64,
            // This op's place in `log`, and whether its `_source` is also
            // the stored source: once appended, the id and source are moved
            // out of the translog op rather than copied.
            at: usize, source: bool, routing: Option<String>, ignored: Vec<String>,
            #[cfg(feature = "mutable-hnsw")]
            token: Option<rustic_core::segment::PreparedDocumentVectors>,
        }
        let mut raw: Vec<Option<Raw>> = Vec::with_capacity(n);
        // 2a. Version checks and seq_nos, under the write state (hash-map
        // work only: no document is built and nothing is appended here).
        let (first, undo, translog) = {
            let mut ws = self.write.lock();
            if ws.prunes != prunes {
                // The version map dropped entries since the lookups (their
                // ops were checked in): look the live copies up again.
                found = lookup_all(&self.core.read(), &ops);
            }
            let first = ws.seq_no + 1;
            // Version-map entries as they were before this batch (undone if
            // its append fails), one per `log` op: the op carries the id.
            let mut undo: Vec<Option<VersionEntry>> = Vec::new();
            for (i, (((op, (hit, cur_term)), version_key), delete_key)) in ops.into_iter().zip(found).zip(version_keys).zip(delete_keys).enumerate() {
                #[cfg(feature = "mutable-hnsw")]
                let token = match tokens[i].take() {
                    Some(Err(error)) => { results.push(Err(core_write_error(&error))); continue; }
                    Some(Ok(token)) => Some(token),
                    None => None,
                };
                #[cfg(not(feature = "mutable-hnsw"))]
                let _ = i;
                let current = |ws: &WriteState, id: &str| -> Option<VersionEntry> {
                    match ws.versions.get(id) {
                        Some(e) => Some(*e),
                        None => hit.as_ref().map(|&(_, v, s)| VersionEntry { version: v, seq_no: s, deleted: false, primary_term: 0 }),
                    }
                };
                // `if_primary_term` compares with the term of the id's latest
                // document, which may still be pending in a pooled memtable.
                // The version map's entry carries its op's term: the copy the
                // lookup found may be an older one (the newest was checked in
                // after the lookup), whose term is not the latest.
                let pending_term = |ws: &WriteState, id: &str| -> Option<u64> {
                    let latest = ws.versions.get(id)?;
                    ws.pending_docs.get(id).filter(|p| p.seq_no == latest.seq_no).map(|p| p.primary_term)
                        .or_else(|| (latest.primary_term > 0).then_some(latest.primary_term))
                };
                match op {
                    WriteOp::Index { id, source, doc, ignored, routing, cond, auto_id } => {
                        let cur = if auto_id { None } else { current(&ws, &id) };
                        let cur_term = if cond.if_primary_term.is_some() { pending_term(&ws, &id).unwrap_or(cur_term) } else { cur_term };
                        // A coordinator retry of an op already applied.
                        if let Some(orig) = self.retry_hit(&cond, &id, cur) {
                            results.push(Ok(orig));
                            continue;
                        }
                        let version = match Self::check(&cond, index, uuid, &id, cur, cur_term) {
                            Ok(version) => version,
                            Err(error) => { results.push(Err(error)); continue; }
                        };
                        let existed = cur.map(|e| !e.deleted).unwrap_or(false);
                        ws.seq_no += 1;
                        let seq_no = ws.seq_no;
                        // The parse may have set the stored form (`vector_source::stored_form`).
                        // The `_source` both the translog and the stored
                        // field need: one copy, moved to the document after
                        // the append (2c) instead of cloned here.
                        let stored_source = !doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD);
                        let at = log.len();
                        let doc_routing = routing.clone();
                        // Auto ids too: until its memtable is checked in, a
                        // delete, overwrite or realtime GET of the returned id
                        // finds the document only here (ES keeps the version
                        // map safe for auto ids the same way once one can be
                        // targeted).
                        undo.push(ws.versions.get(&id).copied());
                        // An overwrite (of a live copy, or of a delete's
                        // tombstone) is an update for the MSU.
                        if cur.is_some() { self.note_update(seq_no); }
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: false, primary_term });
                        log.push(TranslogOp::Index { id: id.clone(), source_json: source, seq_no: seq_no as u64, version, routing });
                        let res = WriteResult { id: id.clone(), version, seq_no, primary_term, result: if existed { "updated" } else { "created" } };
                        if !auto_id {
                            self.retry_note(&cond, &res, false);
                        }
                        raw.push(Some(Raw { delete_key, id, seq_no, older: cur.is_some(), doc, version, at, source: stored_source, routing: doc_routing, ignored,
                            #[cfg(feature = "mutable-hnsw")]
                            token }));
                        results.push(Ok(res));
                    }
                    WriteOp::Delete { id, cond } => {
                        let cur = current(&ws, &id);
                        let cur_term = if cond.if_primary_term.is_some() { pending_term(&ws, &id).unwrap_or(cur_term) } else { cur_term };
                        if let Some(orig) = self.retry_hit(&cond, &id, cur) {
                            results.push(Ok(orig));
                            continue;
                        }
                        let version = match Self::check(&cond, index, uuid, &id, cur, cur_term) {
                            Ok(version) => version,
                            Err(error) => { results.push(Err(error)); continue; }
                        };
                        // ES: a delete of a missing document is still an op.
                        let existed = cur.map(|e| !e.deleted).unwrap_or(false);
                        ws.seq_no += 1;
                        let seq_no = ws.seq_no;
                        undo.push(ws.versions.get(&id).copied());
                        self.note_update(seq_no);
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: true, primary_term });
                        log.push(TranslogOp::Delete { id: id.clone(), seq_no: seq_no as u64, version });
                        let res = WriteResult { id: id.clone(), version, seq_no, primary_term, result: if existed { "deleted" } else { "not_found" } };
                        self.retry_note(&cond, &res, true);
                        raw.push(None);
                        planned.push(Planned::Delete { id, seq_no });
                        results.push(Ok(res));
                    }
                }
            }
            // The version map keeps these entries until the documents are
            // checked in (staleness at check-in, realtime GET).
            if !log.is_empty() {
                ws.claim(first, claim_memory);
            }
            (first, undo, ws.translog.clone())
        };
        // 2b. The translog append, under the translog's own lock only (ES):
        // batches of other ids append concurrently with this one's
        // sequencing; this batch's ids stay locked, so each id's ops reach
        // the translog in seq_no order.
        let mut sync_point = None;
        let mut durability_error = None;
        let appended = if log.is_empty() { None } else { Some(translog.append_many_synced(&log)) };
        let appended_ok = matches!(appended, Some(Ok(_)));
        // 2c. The documents, with their metadata fields in the order the
        // in-place path writes them (off the write state; this batch's ids
        // stay locked). Deletes keep their place in `planned`. Built after
        // the append so each stored `_source` is moved out of its encoded
        // translog op rather than copied; `log` keeps only seq_nos in use.
        let mut sequenced_pending: Vec<(String, Arc<PendingDoc>)> = Vec::new();
        let deletes_planned = std::mem::take(&mut planned);
        let mut deletes_planned = deletes_planned.into_iter();
        for item in raw {
            match item {
                Some(Raw { delete_key, id, seq_no, older, mut doc, version, at, source, routing, ignored, #[cfg(feature = "mutable-hnsw")] token }) => {
                    let (logged_id, logged_source) = match log.get_mut(at) {
                        Some(TranslogOp::Index { id, source_json, .. }) => (Some(id), Some(source_json)),
                        _ => (None, None),
                    };
                    // A failed append undoes the version map by the logged
                    // ids (2d): keep them then.
                    let id_value = match logged_id {
                        Some(logged) if appended_ok => std::mem::take(logged),
                        _ => id.clone(),
                    };
                    doc.fields.push((ID_FIELD, FieldValue::Keyword(id_value)));
                    if let Some(source_json) = logged_source.filter(|_| source) {
                        doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(std::mem::take(source_json))));
                    }
                    doc.fields.push((VERSION_FIELD, FieldValue::I64(version as i64)));
                    doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
                    doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(primary_term as i64)));
                    if let Some(r) = routing {
                        doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r)));
                    }
                    for f in ignored {
                        doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
                    }
                    let pending = Arc::new(PendingDoc { seq_no, version, primary_term, doc });
                    sequenced_pending.push((id.clone(), pending.clone()));
                    planned.push(Planned::Index { delete_key, id, seq_no, older, pending,
                        #[cfg(feature = "mutable-hnsw")]
                        token });
                }
                None => planned.extend(deletes_planned.next()),
            }
        }
        // 2d. Under the write state again (short): the realtime entries, or
        // undoing the batch when its append failed.
        if let Some(appended) = appended {
            let mut ws = self.write.lock();
            match appended {
                Err(error) => {
                    // Never in the translog: nothing of the batch is
                    // installed or replicated (the tailer reads the
                    // translog), the version map is as before the batch
                    // (so a retry is sequenced afresh, never acknowledged
                    // from it), and their seq_nos become no-ops (no hole);
                    // the ops fail to the client below.
                    for (op, before) in log.iter().zip(undo).rev() {
                        let (TranslogOp::Index { id, .. } | TranslogOp::Delete { id, .. }) = op;
                        match before {
                            Some(entry) => { ws.versions.replace_existing(id, entry); }
                            None => { ws.versions.remove(id); }
                        }
                    }
                    ws.release_claim(first);
                    planned.clear();
                    self.mark_tragic(&format!("translog append failed: {error}"));
                    let seq_nos: Vec<i64> = log.iter().map(|op| op.seq_no() as i64).collect();
                    let _ = ws.fill_noops(&seq_nos, primary_term, "translog append failed", false);
                    durability_error = Some(error.to_string());
                }
                Ok(sp) => {
                    {
                        // The newest sync point (appends of other batches may
                        // have landed meanwhile; keep the furthest).
                        let mut tail = self.tail.lock();
                        let newer = tail.as_ref().is_none_or(|(file, mark)| !Arc::ptr_eq(file, &sp.0) || *mark <= sp.1);
                        if newer { *tail = Some(sp.clone()); }
                    }
                    if sync { sync_point = Some(sp); }
                    // Realtime GET / update read these ops from now on
                    // (the latest op of each id; a delete's id has none).
                    for (id, pending) in sequenced_pending.drain(..) {
                        if ws.versions.get(&id).is_some_and(|e| e.seq_no == pending.seq_no) {
                            ws.pending_docs.insert(id, pending);
                        }
                    }
                    for item in &planned {
                        if let Planned::Delete { id, seq_no } = item {
                            if ws.pending_docs.get(id).is_some_and(|p| p.seq_no < *seq_no) { ws.pending_docs.remove(id); }
                        }
                    }
                }
            }
        }
        drop(stripes);
        // Test seam: one write pauses here, sequenced (in the translog, its
        // claim holding the commit point) but not inverted yet.
        if self.pause_armed.load(Ordering::Acquire) {
            let hook = self.pause_hook.lock().take();
            if let Some(hook) = hook {
                self.pause_armed.store(false, Ordering::Release);
                hook();
            }
        }
        // 3. Inversion into a pooled memtable, off every lock, handing it
        // off at a document boundary whenever a refresh / flush freezes.
        // Processed in any case (an append failure filled no-ops already;
        // marking again is a no-op): no seq_no stays a hole.
        let processed: Vec<i64> = log.iter().map(|op| op.seq_no() as i64).collect();
        let (mut inverting, mut bitmap_credits) = prepared_pool.into_inverting((!planned.is_empty()).then_some(first), processed.clone());
        // Pending entries were installed at sequencing.
        let pending = Vec::new();
        let mut deletes = Vec::new();
        let mut indexed = 0u64;
        let stored_compression = self.stored_compression();
        for item in planned {
            let (holding, pooled) = inverting.parts();
            self.observe_freeze(holding, pooled);
            self.maybe_injected_panic();
            match item {
                Planned::Index { delete_key, id, seq_no, older, pending: entry, #[cfg(feature = "mutable-hnsw")] token } => {
                    pooled.writer.set_stored_compression(stored_compression);
                    pooled.writer.ensure_document_bitmap(&entry.doc, &mut bitmap_credits);
                    #[cfg(feature = "mutable-hnsw")]
                    match token {
                        Some(token) => pooled.writer.append_prepared_document(&entry.doc, &schema, token),
                        None => pooled.writer.add_document(&entry.doc, &schema),
                    }
                    #[cfg(not(feature = "mutable-hnsw"))]
                    pooled.writer.add_document(&entry.doc, &schema);
                    pooled.admit_next_doc(holding);
                    pooled.docs.push(PoolDoc { id: delete_key, seq_no, local: pooled.writer.num_docs() - 1, older });
                    indexed += 1;
                }
                Planned::Delete { id, seq_no } => deletes.push((id, seq_no)),
            }
        }
        let (mut holding, mut pooled) = inverting.finish();
        self.observe_freeze(&mut holding, &mut pooled);
        // 4. Completion.
        let deleted = deletes.len() as u64;
        if let Err(error) = self.complete_pooled(holding, pooled, pending, &deletes, &processed, false) {
            tracing::error!("staged write completion failed: {error}");
        }
        drop(_batch);
        let mut durability_error = durability_error;
        if let Some((file, mark)) = sync_point {
            if let Err(error) = self.group_sync(&file, mark) {
                durability_error = Some(error.to_string());
            }
        }
        if let Some(error) = durability_error {
            for result in &mut results {
                if result.is_ok() { *result = Err(EsError::internal(format!("translog durability failure: {error}"))); }
            }
        }
        self.counters.index_total.fetch_add(indexed, Ordering::Relaxed);
        self.counters.delete_total.fetch_add(deleted, Ordering::Relaxed);
        self.counters.index_failed.fetch_add(n_idx_ops.saturating_sub(indexed), Ordering::Relaxed);
        let elapsed = started.elapsed();
        let total = (indexed + deleted).max(1) as u32;
        Counters::add_time(&self.counters.index_time_nanos, elapsed * indexed as u32 / total);
        Counters::add_time(&self.counters.delete_time_nanos, elapsed * deleted as u32 / total);
        if indexed + deleted > 0 { self.sched.note_write(); }
        self.files_changed();
        results
    }

    /// Test seam: the next staged write runs `hook` between its sequencing
    /// and its inversion (it is in the translog and holds the commit point,
    /// its documents are not in any memtable yet). Production never arms it.
    #[doc(hidden)]
    pub fn pause_next_staged_write(&self, hook: impl FnOnce() + Send + 'static) {
        *self.pause_hook.lock() = Some(Box::new(hook));
        self.pause_armed.store(true, Ordering::Release);
    }

    /// Test seam: the next flush runs `hook` right after its capture point
    /// (writers released, cluster fence released), before it seals and
    /// commits: a slow seal / commit. Production never arms it.
    #[doc(hidden)]
    pub fn pause_next_flush_commit(&self, hook: impl FnOnce() + Send + 'static) {
        *self.flush_pause.lock() = Some(Box::new(hook));
    }

    /// Test seam: the next staged write panics in its inversion (before
    /// its next document). Production never arms it.
    #[doc(hidden)]
    pub fn panic_next_inversion(&self) {
        self.inversion_panic.store(true, Ordering::Release);
    }

    fn maybe_injected_panic(&self) {
        if self.inversion_panic.load(Ordering::Relaxed) && self.inversion_panic.swap(false, Ordering::AcqRel) {
            panic!("injected indexing panic");
        }
    }

    /// Whether this shard's graph save is overdue: due, and skipped too
    /// often (captures no longer wait for in-flight ingestion, so this
    /// only follows repeated capture failures). For node statistics.
    pub fn graph_save_overdue(&self) -> bool {
        self.graph_saves.lock().overdue()
    }

    /// Ops the last commit holds above its graph checkpoint (B9: what a
    /// copy opened from that commit inserts into its graphs at open);
    /// `None` without a graph checkpoint.
    pub fn graph_save_lag(&self) -> Option<i64> {
        let saved = self.graph_saves.lock().seq_no?;
        Some(self.committed_seq_no().saturating_sub(saved).max(0))
    }

    /// Make the next flush save the mutable graphs (whatever the save
    /// schedule says; skipped as usual while graph ingestion is in flight).
    /// A file recovery source asks for it before pinning a commit whose
    /// graph checkpoint lags far behind, so the target's open-time catch-up
    /// stays small.
    pub fn request_graph_save(&self) {
        self.graph_saves.lock().forced = true;
    }

    /// Test seam: the next `count` translog appends fail.
    #[doc(hidden)]
    pub fn fail_next_translog_appends(&self, count: usize) {
        self.write.lock().translog.fail_next_appends(count);
    }

    /// Test seam: the next `count` history side-log appends fail.
    #[cfg(test)]
    pub(crate) fn fail_next_side_log_appends(&self, count: usize) {
        self.write.lock().hist.side.fail_next_appends(count);
    }

    /// Test seam: the next `count` translog fsyncs (group commits) fail.
    #[doc(hidden)]
    pub fn fail_next_translog_syncs(&self, count: usize) {
        self.fail_syncs.store(count, Ordering::Release);
    }

    /// A pooled memtable held by one request ([`Shard::apply_staged`]):
    /// an idle one from the pool, else a new one. The request is registered
    /// as a holder first, so a freeze that starts now waits for it to hand
    /// the memtable off ([`Shard::observe_freeze`]).
    #[cfg(test)]
    fn hold_pool_writer(&self, claim: Option<i64>) -> Result<(Holding, PoolWriter)> {
        let holder_memory=rustic_core::admission::shared().reserve(2048,"refresh holder node")?;
        let queue_memory=rustic_core::admission::shared().reserve(PoolWriter::queue_bytes() as u64,"pooled queue node")?;
        let token = self.next_holder.fetch_add(1, Ordering::Relaxed);
        let seen = self.freeze_gen.load(Ordering::Acquire);
        self.holders.lock().insert(token, (seen,holder_memory));
        let pooled = self.pool.lock().pop_back().unwrap_or_else(||PoolWriter::fresh_with_credit(Some(queue_memory)));
        Ok((Holding { token, seen, claim: claim.map(|first| Arc::new(ClaimRef(first,None))), registry_credits: None }, pooled))
    }

    /// At a document boundary: when a refresh or flush has asked for a
    /// freeze since (ES DWPT flush-pending), hand the memtable (every
    /// document so far, other requests' included) to it and continue in a
    /// fresh one. Never waits.
    fn observe_freeze(&self, holding: &mut Holding, pooled: &mut PoolWriter) {
        // Every POOL_ACCOUNT_DOCS documents added (a delete-heavy request
        // adds none and never re-estimates).
        // (and at its first document, so a pooled memtable of a few large
        // documents is never invisible).
        if pooled.docs.len() >= pooled.accounted_docs + POOL_ACCOUNT_DOCS || (pooled.accounted_docs == 0 && !pooled.docs.is_empty()) {
            pooled.account();
        }
        let current = self.freeze_gen.load(Ordering::Acquire);
        if current == holding.seen {
            return;
        }
        // Empty writers need no handoff and retain their already admitted
        // claim slot. This also bounds fresh allocations by actual documents.
        if pooled.docs.is_empty() && pooled.registered.is_empty() {
            holding.seen=current;self.holders.lock().get_mut(&holding.token).expect("registered holder").0=current;self.holders_changed.notify_all();return;
        }
        let memory=if holding.registry_credits.is_some() { holding.handoff_credit(PoolWriter::queue_bytes() as u64,"pooled queue node handoff") } else { None };
        let mut next=PoolWriter::fresh_with_credit(memory);
        next.admit_claim_slot(holding);
        let mut full = std::mem::replace(pooled, next);
        if !full.docs.is_empty() || !full.registered.is_empty() {
            // Its documents may include this request's: they stay claimed
            // until this memtable is checked in too.
            if let Some(claim) = &holding.claim {
                if !full.docs.is_empty() { full.registered.push(claim.clone()); }
            }
            full.account();
            self.frozen.lock().push_back(full);
        }
        holding.seen = current;
        self.holders.lock().get_mut(&holding.token).expect("registered holder").0=current;
        self.holders_changed.notify_all();
    }

    fn release_holding(&self, holding: Holding) {
        self.holders.lock().remove(&holding.token);
        self.holders_changed.notify_all();
    }

    /// Freeze the pooled memtables (ES DWPT flush-pending): every request
    /// holding one hands it off at its next document boundary; waits (at
    /// most `wait`) until all have, so the check-in that follows contains
    /// every write acknowledged before the freeze. Never waits for a whole
    /// request. Returns whether every holder handed off in time.
    fn freeze_pool(&self, wait: std::time::Duration) -> bool {
        let target = self.freeze_gen.fetch_add(1, Ordering::AcqRel) + 1;
        self.wait_freeze(target, wait)
    }

    /// Start a freeze and take the idle pooled memtables, atomically under
    /// the pool lock: a memtable is either taken here or returned to the
    /// pool by a request that already sees this freeze (and so hands it to
    /// the frozen list instead, [`Shard::complete_pooled`]); a request
    /// taking a memtable before it holds and is waited for. Returns the
    /// freeze generation and the idle memtables.
    fn begin_freeze(&self) -> (u64, std::collections::LinkedList<PoolWriter>) {
        let mut pool = self.pool.lock();
        let target = self.freeze_gen.fetch_add(1, Ordering::AcqRel) + 1;
        (target, std::mem::take(&mut *pool))
    }

    /// Wait (at most `wait`) until every holder handed off for freeze `target`.
    fn wait_freeze(&self, target: u64, wait: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + wait;
        let mut holders = self.holders.lock();
        while holders.values().any(|(seen,_)| *seen < target) {
            if self.holders_changed.wait_until(&mut holders, deadline).timed_out() {
                return !holders.values().any(|(seen,_)| *seen < target);
            }
        }
        true
    }

    /// Bytes of the pooled memtables (idle and frozen): they count toward
    /// the indexing buffer.
    fn pooled_bytes(&self) -> u64 {
        let idle: usize = self.pool.lock().iter().map(|p| p.writer.estimated_bytes()).sum();
        let frozen: usize = self.frozen.lock().iter().map(|p| p.writer.estimated_bytes()).sum();
        (idle + frozen) as u64
    }

    /// Completion of a staged request (primary or replica): pending
    /// documents for realtime GET (unless already checked in with a frozen
    /// memtable), deletes applied to the core, seq_nos marked processed,
    /// the request's claim moved to the memtable it ends in, the memtable
    /// returned to the pool (or checked in when large or when the indexing
    /// buffer is full, even with refreshes disabled), then the holding
    /// released.
    #[allow(clippy::too_many_arguments)]
    fn complete_pooled(&self, mut holding: Holding, mut pooled: PoolWriter, pending: Vec<(String, Arc<PendingDoc>, Arc<PoolCheckin>)>,
        deletes: &[(String, i64)], processed: &[i64], replica: bool) -> Result<()> {
        let mut failure = None;
        {
            // Declared first: frees what check-ins below release after the
            // write state and the core are unlocked (a check-in of every
            // idle pooled memtable freed their parsed documents under both,
            // stalling every write of the shard for about 400 ms).
            let _drops = LocalCheckinDrops::enter();
            // Waiting for the write state is not a document boundary a
            // freeze should sit through: a refresh / flush that asks
            // meanwhile gets the memtable at once (its claim then moves to
            // the fresh one, as for a hand-off during inversion).
            let mut ws = loop {
                if let Some(ws) = self.write.try_lock_for(HANDOFF_POLL) { break ws; }
                self.observe_freeze(&mut holding, &mut pooled);
            };
            for (id, doc, checked_in) in pending {
                if checked_in.load(Ordering::Acquire) { continue; }
                // Only while it is the latest op of its id.
                if ws.versions.get(&id).is_some_and(|e| e.seq_no == doc.seq_no) {
                    ws.pending_docs.insert(id, doc);
                }
            }
            if !deletes.is_empty() {
                // Deleting older copies only needs the leaf set stable
                // (`delete_doc_shared`, as the buffered-delete resolution
                // does): a shared core lock, so searches keep running while
                // an update bulk deletes its thousands of older copies.
                let core = self.core.read();
                for (id, seq_no) in deletes {
                    if ws.pending_docs.get(id).is_some_and(|p| p.seq_no < *seq_no) { ws.pending_docs.remove(id); }
                    if let Err(error) = Self::delete_older(&core, id, *seq_no) {
                        tracing::error!("delete of [{id}] failed: {error}");
                        failure.get_or_insert(error);
                    }
                }
            }
            for seq_no in processed { ws.hist.mark(*seq_no); }
            if replica { ws.hist.prune_tombstones(); }
            if let Some(claim) = holding.claim.take() {
                // The memtable now carries the request's version-map and
                // commit-point claim (with any memtable it handed off) until
                // its check-in.
                pooled.registered.push(claim);
            }
            let core_bytes = self.core.read().writing_bytes() as u64;
            let pooled_bytes = pooled.writer.estimated_bytes() as u64 + self.pooled_bytes();
            if pooled.docs.len() >= POOL_CHECKIN_DOCS || self.buffer_full_with(core_bytes, pooled_bytes) {
                let mut core = self.core.write();
                self.checkin_pool_writer(&mut ws, &mut core, pooled);
                // Under memory pressure every idle pooled memtable goes too.
                let idle = std::mem::take(&mut *self.pool.lock());
                for writer in idle { self.checkin_pool_writer(&mut ws, &mut core, writer); }
                self.after_checkin(&mut ws, &mut core);
            } else {
                // A freeze that started since this request's last document
                // boundary (and so did not find this memtable in the pool)
                // gets it through the frozen list: it may hold writes
                // acknowledged before that refresh. Decided under the pool
                // lock, which a freeze holds while it starts.
                pooled.account();
                let mut pool = self.pool.lock();
                if self.freeze_gen.load(Ordering::Acquire) != holding.seen {
                    drop(pool);
                    self.frozen.lock().push_back(pooled);
                } else {
                    pool.push_back(pooled);
                }
            }
        }
        self.release_holding(holding);
        failure.map_or(Ok(()), Err)
    }

    /// Delete every live copy of `id` in the core sequenced before
    /// `seq_no` (all of them: no copy older than the newest op survives).
    fn delete_older(core: &CoreIndex, id: &str, seq_no: i64) -> Result<()> {
        let older: Vec<DocAddress> = {
            let term = Term::new_text(ID_FIELD, id);
            let leaves = core.all_leaves();
            let mut out = Vec::new();
            for (i, leaf) in leaves.iter().enumerate() {
                if let rustic_core::query::LeafReader::Segment(segment) = leaf {
                    if !segment.may_contain_unique_term(&term) { continue; }
                }
                if let Some(mut postings) = leaf.postings(&term) {
                    while let Some(d) = postings.next() {
                        if leaf.is_deleted_latest(d) { continue; }
                        if leaf.i64_value(SEQ_NO_FIELD, d).unwrap_or(0) < seq_no {
                            out.push(DocAddress::new(i as u32, d));
                        }
                    }
                }
            }
            out
        };
        for addr in &older {
            core.delete_doc_shared(addr)?;
        }
        Ok(())
    }

    /// ES `max_seq_no_of_updates_or_deletes` of this copy (shipped to
    /// replicas with every replication request).
    pub fn max_seq_no_of_updates(&self) -> i64 {
        self.msu.load(Ordering::Acquire)
    }

    /// A replica learns its primary's MSU (every replication request
    /// carries it): from now on replay applies an op above it as the first
    /// copy of its id, without looking for an older one.
    pub fn advance_max_seq_no_of_updates_from_primary(&self, msu: i64) {
        self.msu.fetch_max(msu, Ordering::AcqRel);
        self.msu_from_primary.store(true, Ordering::Release);
    }

    /// Raise the MSU to at least `seq_no` without learning it from the
    /// primary: ops at or below it are applied with the full lookup, and
    /// the first-copy fast path stays as it was (off until a primary's
    /// MSU arrives with a replication request).
    pub fn raise_max_seq_no_of_updates(&self, seq_no: i64) {
        self.msu.fetch_max(seq_no, Ordering::AcqRel);
    }

    /// An op updated or deleted an id: raise the MSU.
    fn note_update(&self, seq_no: i64) {
        self.msu.fetch_max(seq_no, Ordering::AcqRel);
    }

    /// Apply the index's merge settings (dynamic: the scheduler re-reads them;
    /// see [`MergeConfig::from_settings`]).
    pub fn set_merge_config(&self, config: MergeConfig) {
        if *self.bg.merge_config.read() == config { return; }
        self.bg.merge_running.set_max(config.max_thread_count);
        *self.bg.merge_config.write() = config;
        // A larger budget or a different policy may start merges now.
        self.bg.want_merge.store(true, Ordering::Release);
    }

    /// Test seam: `probe` runs in every staged batch while it holds its id
    /// locks, after its lookups and before its version checks (`None`
    /// disarms). Production never arms it.
    #[doc(hidden)]
    pub fn probe_next_sequencing(&self, probe: Option<Arc<dyn Fn() + Send + Sync>>) {
        *self.sequencing_probe.lock() = probe;
    }

    /// Record that every copy of `id` sequenced before `seq_no` is to be
    /// deleted (Lucene's buffered delete terms): a check-in records it
    /// instead of looking the id up in every leaf under the write state and
    /// the core lock. Until it is resolved the version map keeps the id
    /// (`prune_versions`), realtime lookups read the newest copy
    /// (`lookup`), and nothing becomes visible before it is resolved
    /// (`resolve_all_buffered_deletes` in the refresh critical section and
    /// before a commit).
    fn buffer_delete(&self, key: registry::DeleteKey, seq_no: i64) {
        let mut unresolved = self.unresolved.lock();
        let entry = unresolved.entry(key).or_insert(seq_no);
        *entry = (*entry).max(seq_no);
    }

    /// Resolve the buffered deletes under short core *read* guards (chunks
    /// of [`RESOLVE_CHUNK`] ids; leaves keep their deletions behind their
    /// own locks): searches, realtime reads and writes' sequencing proceed,
    /// and check-ins (core write) get in between chunks. Deletes land as
    /// pending; the next refresh publishes them together with the copies
    /// that replace them.
    fn resolve_buffered_deletes(&self) {
        // A sorted refresh in flight: only what it will publish. The window
        // and the buffered deletes are read under one hold (window, then
        // unresolved), so a sorted refresh starting meanwhile cannot have
        // its seal point overtaken by this snapshot.
        let pending = {
            let window = self.sorted_window.lock();
            match window.as_ref() {
                Some(window) => registry::delete_snapshot(window),
                None => registry::delete_snapshot(&self.unresolved.lock()),
            }
        };
        // This is only a lock-shortening optimization. Required publication
        // still resolves the retained map under its core lock on refusal.
        let Ok(pending) = pending else { return; };
        for chunk in pending.chunks(RESOLVE_CHUNK) {
            {
                let core = self.core.read();
                for (id, seq_no) in chunk {
                    self.delete_older_or_fail(&core, id, *seq_no);
                }
            }
            // Only what was resolved: a later op on the id buffered meanwhile stays.
            let mut unresolved = self.unresolved.lock();
            for (id, seq_no) in chunk {
                if unresolved.get(id.as_ref()) == Some(seq_no) { unresolved.remove(id.as_ref()); }
            }
        }
    }

    /// Resolve whatever is still buffered, under the core write lock the
    /// caller holds (no check-in can add more meanwhile): the refresh
    /// critical section and a commit's capture, after
    /// [`Shard::resolve_buffered_deletes`] did the bulk.
    fn resolve_all_buffered_deletes(&self, core: &CoreIndex) {
        let all = std::mem::take(&mut *self.unresolved.lock());
        for (id, seq_no) in all {
            self.delete_older_or_fail(core, &id, seq_no);
        }
    }

    /// The sorted refresh's publication (core write lock held): resolve the
    /// buffered deletes of its seal point only; deletes of copies checked
    /// in since stay buffered until a refresh publishes those copies.
    fn resolve_window_buffered_deletes(&self, core: &CoreIndex, window: rustic_core::admission::Allocation<registry::Deletes>) {
        let (window, _memory) = window.into_parts();
        for (id, seq_no) in window {
            self.delete_older_or_fail(core, &id, seq_no);
            let mut unresolved = self.unresolved.lock();
            if unresolved.get(id.as_ref()) == Some(&seq_no) { unresolved.remove(id.as_ref()); }
        }
    }

    /// [`Shard::delete_older`] of a buffered delete. Its op was acknowledged
    /// already: an older copy that cannot be deleted would stay live beside
    /// the new one, so a failure fails the engine (ES fails its engine on an
    /// IndexWriter failure), and the copy is recovered from the translog or
    /// the primary.
    fn delete_older_or_fail(&self, core: &CoreIndex, id: &str, seq_no: i64) {
        let injected = self.fail_older_deletes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).is_ok();
        let result = if injected { Err(EsError::internal("injected delete failure")) } else { Self::delete_older(core, id, seq_no) };
        if let Err(error) = result {
            self.mark_tragic(&format!("could not delete the previous version of [{id}]: {error}"));
        }
    }

    /// Test seam: the next `count` buffered-delete resolutions fail.
    #[doc(hidden)]
    pub fn fail_next_older_deletes(&self, count: usize) {
        self.fail_older_deletes.store(count, Ordering::Release);
    }

    /// Resolve `id`'s buffered delete now (callers replacing its copy in
    /// place under the core lock).
    fn resolve_buffered_delete(&self, core: &CoreIndex, id: &str) {
        let seq_no = self.unresolved.lock().remove(id);
        if let Some(seq_no) = seq_no {
            self.delete_older_or_fail(core, id, seq_no);
        }
    }

    /// Check a pooled memtable into the index (caller holds the write state
    /// and the core): in its documents' order, a document whose id has a
    /// later-sequenced op is installed deleted; otherwise every older live
    /// copy of its id is deleted first.
    fn checkin_pool_writer(&self, ws: &mut WriteState, core: &mut CoreIndex, pooled: PoolWriter) {
        // The index's writers count in the indexing buffer from here: the
        // pooled charge is released with `_charge` at the end of check-in.
        let PoolWriter { queue_memory: _queue_memory, docs_memory: _docs_memory, registered_memory: _registered_memory, writer, docs, registered, checked_in, charge: _charge, accounted_docs: _ } = pooled;
        // Under the write state: a completing request then skips pending
        // entries for documents already checked in here.
        checked_in.store(true, Ordering::Release);
        let mut released = Vec::new();
        let t_docs = std::time::Instant::now();
        let older_count = docs.iter().filter(|d| d.older).count();
        for doc in &docs {
            // Auto-id documents included: a delete or overwrite of the
            // returned id may have been sequenced while this one was pooled.
            let stale = ws.versions.get(&doc.id).is_some_and(|e| e.seq_no > doc.seq_no);
            if stale {
                writer.delete_doc(doc.local);
                continue;
            }
            if doc.older {
                self.buffer_delete(doc.id.clone(), doc.seq_no);
            }
            // One lookup: a newer pending copy (rare) goes back.
            if let Some((key, p)) = ws.pending_docs.remove_entry(doc.id.as_ref()) {
                if p.seq_no <= doc.seq_no { released.push(p); } else { ws.pending_docs.insert(key, p); }
            }
        }
        let t_core = std::time::Instant::now();
        if writer.num_docs() > 0 {
            core.checkin_writer(writer);
        }
        note_refresh_phase(|p| { p.checkin_docs_loop += t_core - t_docs; p.checkin_core += t_core.elapsed(); p.checkin_older += older_count as u32; });
        // A refresh's check-ins free the pooled copies together, off this
        // thread when large ([`CHECKIN_DROPS`]): a refresh checks in every
        // document written since the last one, and freeing their parsed
        // documents here (under the write state and the core, on the thread
        // a search-idle search waits on) took about 130 ms of a 340 ms
        // refresh (28k TSDB docs).
        if DEFER_CHECKIN_DROPS.with(|d| d.get()) {
            let n = docs.len();
            CHECKIN_DROPS.with(|c| { let mut c = c.borrow_mut(); c.0.push(Box::new((docs, released))); c.1 += n; });
        }
        for claim in registered {
            // The last reference (no other memtable of the request and not
            // the request itself holds one) releases the claim.
            let last = Arc::strong_count(&claim) == 1;
            let first = claim.0;
            drop(claim);
            if last { ws.release_claim(first); }
        }
    }

    /// After check-ins: seal when the memtables pile up (as `apply`).
    fn after_checkin(&self, ws: &mut WriteState, core: &mut CoreIndex) {
        if self.buffer_full(core.writing_bytes() as u64) || core.num_writing_docs() >= self.seal_threshold() || core.num_open_memtables() > MAX_OPEN_MEMTABLES {
            if self.bg.served.load(Ordering::Acquire) {
                self.bg.want_seal.store(true, Ordering::Release);
            } else if let Err(e) = core.seal() {
                tracing::error!("seal failed: {e}");
            } else {
                self.note_buffered(core.writing_bytes() as u64);
            }
            ws.prune_versions(&self.unresolved.lock());
        }
    }

    /// Check the idle pooled memtables into the index (refresh, flush,
    /// realtime GET of a pending id). `wait`: `None` blocks for the write
    /// state and the core; `Some(d)` gives up after `d` (returns `false`,
    /// nothing done). Requests holding a pooled memtable are not waited
    /// for: they return it before acknowledging, so a refresh that starts
    /// after an acknowledgement checks it in.
    pub fn checkin_pool(&self, wait: Option<std::time::Duration>) -> bool {
        let checkin = match wait {
            None => self.checkin_lock.lock(),
            Some(d) => match self.checkin_lock.try_lock_for(d) { Some(guard) => guard, None => return false },
        };
        let mut writers = std::mem::take(&mut *self.frozen.lock());
        writers.extend(std::mem::take(&mut *self.pool.lock()));
        let done = self.checkin_taken(writers, wait);
        drop(checkin);
        self.resolve_buffered_deletes();
        done
    }

    /// Check `writers` (taken out of the pool / frozen list) in. Taking them
    /// out first matters: while this waits for the write state, a request
    /// must not take a memtable holding writes completed before the
    /// check-in. Writers it cannot check in (locks busy within `wait`) go
    /// to the frozen list, for the next check-in.
    fn checkin_taken(&self, writers: std::collections::LinkedList<PoolWriter>, wait: Option<std::time::Duration>) -> bool {
        if writers.is_empty() {
            return true;
        }
        let t = std::time::Instant::now();
        let ws = match wait {
            None => Some(self.write.lock()),
            Some(d) => self.write.try_lock_for(d),
        };
        note_refresh_phase(|p| p.write_lock += t.elapsed());
        let Some(mut ws) = ws else {
            self.frozen.lock().extend(writers);
            return false;
        };
        let t = std::time::Instant::now();
        let core = match wait {
            None => Some(self.core.write()),
            Some(d) => self.core.try_write_for(d),
        };
        note_refresh_phase(|p| p.core_lock += t.elapsed());
        let Some(mut core) = core else {
            self.frozen.lock().extend(writers);
            return false;
        };
        // Handed off meanwhile: these too.
        let mut writers = writers;
        writers.extend(std::mem::take(&mut *self.frozen.lock()));
        let t = std::time::Instant::now();
        let (memtables, docs) = (writers.len() as u32, writers.iter().map(|w| w.docs.len() as u32).sum::<u32>());
        for pooled in writers {
            self.checkin_pool_writer(&mut ws, &mut core, pooled);
        }
        let t2 = std::time::Instant::now();
        self.after_checkin(&mut ws, &mut core);
        // A refresh's check-in also prunes the version map (ES drops the
        // refreshed entries of its live version map at refresh): it stays
        // the size of the writes since the last refresh instead of growing
        // until a seal (250k entries under touch writes, each check-in's
        // lookups missing the cache). The entries are freed with the
        // refresh's other drops, after the locks ([`CHECKIN_DROPS`]).
        if DEFER_CHECKIN_DROPS.with(|d| d.get()) {
            let pruned = ws.prune_versions_deferred(&self.unresolved.lock());
            let n = pruned.len();
            if n > 0 { CHECKIN_DROPS.with(|c| { let mut c = c.borrow_mut(); c.0.push(Box::new(pruned)); c.1 += n; }); }
        }
        note_refresh_phase(|p| { p.checkin_writers += t2 - t; p.after_checkin += t2.elapsed(); p.checkin_memtables += memtables; p.checkin_docs += docs; });
        true
    }

    /// What a refresh makes visible: every write completed before now. The
    /// idle pooled memtables are taken first (so no request takes one of
    /// them meanwhile); then the freeze makes every holder hand its
    /// memtable off at its next document boundary (those memtables can hold
    /// writes completed before now too); then all of them are checked in.
    /// `wait`: `None` blocks; `Some(d)` gives up (returns `false`) if a
    /// holder or the locks are not free within `d`.
    pub fn checkin_completed_writes(&self, wait: Option<std::time::Duration>) -> bool {
        let t = std::time::Instant::now();
        // One check-in at a time takes the pooled memtables (see
        // `checkin_lock`): a blocking one waits for one in progress, so it
        // never misses memtables that check-in still holds.
        let checkin = match wait {
            None => self.checkin_lock.lock(),
            Some(d) => match self.checkin_lock.try_lock_for(d) { Some(guard) => guard, None => return false },
        };
        note_refresh_phase(|p| p.checkin_lock += t.elapsed());
        let t = std::time::Instant::now();
        let (target, mut writers) = self.begin_freeze();
        note_refresh_phase(|p| p.begin_freeze += t.elapsed());
        let t = std::time::Instant::now();
        let frozen = self.wait_freeze(target, wait.unwrap_or(FREEZE_WAIT));
        note_refresh_phase(|p| p.handoff += t.elapsed());
        writers.extend(std::mem::take(&mut *self.frozen.lock()));
        let done = self.checkin_taken(writers, wait);
        drop(checkin);
        // Older copies of what was checked in: outside the write state. A
        // non-blocking check-in (a search-idle search refreshing, the
        // scheduler) never waits on the core for it: its publication
        // resolves what is left under the core lock it already holds.
        if wait.is_none() {
            self.resolve_buffered_deletes();
        }
        done && frozen
    }

    /// Forwarded auto-ID requests intentionally become explicit creates for
    /// retry safety. Preserve their version/duplicate checks, but prepare fresh
    /// vectors concurrently while the ordinary write/core fences retain absence.
    #[cfg(feature = "mutable-hnsw")]
    #[cfg_attr(not(test), allow(dead_code))]
    fn prepare_fresh_mutable_batch(&self, ws: &WriteState, core: &mut CoreIndex, ops: &[WriteOp])
        -> Option<Vec<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>>
    {
        if !Self::fresh_mutable_batch_candidate(ops) || !Self::fresh_ids(ws, core, ops) { return None; }
        core.enable_new_mutable_vector_fields().ok()?;
        let (schema, store) = (core.schema.clone(), core.vector_store().clone());
        Self::prepare_mutable_tokens_with(ops, &schema, &store)
    }

    /// A batch whose vector rows may be prepared ahead of the seq_no section:
    /// several plain index ops (no conditions, retries or versions), some
    /// holding vectors.
    #[cfg(feature = "mutable-hnsw")]
    fn fresh_mutable_batch_candidate(ops: &[WriteOp]) -> bool {
        if ops.len() < 2 || !ops.iter().any(|op| matches!(op, WriteOp::Index { doc, .. } if doc.has_vectors())) { return false; }
        ops.iter().all(|op| match op {
            WriteOp::Index { cond, .. } => !(cond.is_retry || cond.coordinator_id.is_some() || cond.if_seq_no.is_some()
                || cond.if_primary_term.is_some() || cond.version.is_some()),
            _ => false,
        })
    }

    /// Every id of `ops` is distinct and new (no live document, no version
    /// entry). Call with the write state and the core held.
    #[cfg(feature = "mutable-hnsw")]
    fn fresh_ids(ws: &WriteState, core: &CoreIndex, ops: &[WriteOp]) -> bool {
        let mut ids = std::collections::HashSet::with_capacity(ops.len());
        let leaves = core.all_leaves();
        for op in ops {
            let WriteOp::Index { id, .. } = op else { return false; };
            if !ids.insert(id) || ws.versions.contains_key(id) || Self::lookup_in_impl(&leaves, id, true, true).is_some() { return false; }
        }
        true
    }

    /// Prepare the vector rows of `ops` holding no shard lock (the core is
    /// taken briefly to activate new mutable fields and read the schema).
    #[cfg(feature = "mutable-hnsw")]
    fn prepare_mutable_tokens(&self, ops: &[WriteOp])
        -> Option<Vec<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>>
    {
        let (schema, store) = {
            let core = self.core.write();
            core.enable_new_mutable_vector_fields().ok()?;
            (core.schema.clone(), core.vector_store().clone())
        };
        Self::prepare_mutable_tokens_with(ops, &schema, &store)
    }

    #[cfg(feature = "mutable-hnsw")]
    fn prepare_mutable_tokens_with(ops: &[WriteOp], schema: &[CoreField], store: &Arc<rustic_core::vector::store::VectorStore>)
        -> Option<Vec<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>>
    {
        let admission_context=rustic_core::admission::Context::capture();
        use rayon::prelude::*;
        let _maintenance = store.mutable_ingestion_batch().ok()?;
        #[cfg(feature = "gpu")]
        {
            let docs: Vec<_> = ops.iter().map(|op| match op {
                WriteOp::Index { doc, .. } => doc, _ => unreachable!(),
            }).collect();
            match rustic_core::segment::SegmentWriter::prepare_gpu_documents(&docs, schema, store) {
                Ok(Some(tokens)) => return Some(tokens.into_iter().map(Ok).collect()),
                Ok(None) => {},
                // GPU preparation owns cancellation. A failure after reserving
                // rows must not admit a partially prepared document batch.
                Err(error) => return Some(ops.iter().map(|_| Err(rustic_core::error::RusticError::Segment(error.to_string()))).collect()),
            }
        }
        let pool = vector_prepare_pool()?;
        // A single attempt: callers hold `batches.read()`, and nothing may
        // wait while holding a shard lock (readers no longer cap writers'
        // graph budgets, see `MutableHnswLimits::default`).
        let _ = pool;
        Some(prepare_bulk_vectors(ops, schema, store, &admission_context))
    }

    /// Private mutable bulk writers prepare concurrently; graph completion is
    /// serialized by each field owner. Only successful admissions enter WAL,
    /// and no writer becomes refresh-visible before the WAL boundary succeeds.
    #[cfg(feature = "mutable-hnsw")]
    fn apply_mutable_append_only(&self, ops: Vec<WriteOp>, sync: bool) -> Vec<Result<WriteResult>> {
        let _shard_owner=self.admitted_owner.enter();
        let admission_context=rustic_core::admission::Context::capture();
        use rayon::prelude::*;
        let n = ops.len();
        let claim_memory=match Self::prepare_claim_node() {
            Ok(memory)=>memory,Err(error)=>return (0..n).map(|_|Err(error.clone())).collect(),
        };
        let _current = CurrentGuard::new(vec![(&self.counters.index_current, n as u64)]);
        let started = std::time::Instant::now();
        let _batch = self.batches.read();
        let (schema, store) = {
            let core = self.core.write();
            if let Err(error) = core.enable_new_mutable_vector_fields() {
                return (0..n).map(|_| Err(core_write_error(&error))).collect();
            }
            (core.schema.clone(), core.vector_store().clone())
        };
        let maintenance_batch = match store.mutable_ingestion_batch() {
            Ok(batch) => batch,
            Err(error) => return (0..n).map(|_| Err(core_write_error(&error))).collect(),
        };
        // Schedule graph preparation per document, independently of text-writer
        // chunking. A typical 256-row bulk splits to ~128 rows per shard: fixed
        // 64-row tasks limited that shard to just two preparation workers.
        #[cfg(feature = "gpu")]
        let gpu_tokens = {
            let docs: Vec<_> = ops.iter().map(|op| match op { WriteOp::Index { doc, .. } => doc, _ => unreachable!() }).collect();
            match rustic_core::segment::SegmentWriter::prepare_gpu_documents(&docs, &schema, &store) {
                Ok(tokens) => tokens,
                Err(error) => return (0..n).map(|_| Err(core_write_error(&error))).collect(),
            }
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_tokens: Option<Vec<rustic_core::segment::PreparedDocumentVectors>> = None;
        let prepared: Vec<_> = if let Some(tokens) = gpu_tokens {
            ops.into_iter().zip(tokens.into_iter().map(Ok)).collect()
        } else {
            // On the dedicated pool, never stealing while it waits: this
            // thread holds `batches.read()`, and stealing another vector bulk
            // of this shard on a shared pool would re-take it behind a queued
            // writer (flush, schema change) and deadlock.
            // One attempt, never a wait: this thread holds `batches.read()`.
            let tokens: Vec<_> = prepare_bulk_vectors(&ops, &schema, &store, &admission_context);
            ops.into_iter().zip(tokens).collect()
        };
        drop(maintenance_batch);
        // Private chunk writers are prepared before sequence allocation/WAL.
        // No graph token or already published writer is changed on refusal.
        let writer_count = prepared.len().div_ceil(MUTABLE_TEXT_CHUNK_DOCS);
        let writer_bytes = writer_count.checked_mul(std::mem::size_of::<rustic_core::segment::SegmentWriter>());
        let _writer_memory = match writer_bytes.ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "writer batch capacity overflow"))
            .and_then(|bytes| rustic_core::admission::shared().reserve(bytes as u64, "private bitmap writer batch").map_err(EsError::from)) {
            Ok(memory) => memory, Err(error) => return (0..n).map(|_| Err(error.clone())).collect(),
        };
        let mut bitmap_writers = Vec::with_capacity(writer_count);
        let stored_compression = self.stored_compression();
        for chunk in prepared.chunks(MUTABLE_TEXT_CHUNK_DOCS) {
            let mut writer = rustic_core::segment::SegmentWriter::new(0);
            writer.set_stored_compression(stored_compression);
            let admission = chunk.iter().try_fold(0usize, |count, (op, token)| {
                if token.is_err() { return Ok(count); }
                let WriteOp::Index { doc, .. } = op else { unreachable!() };
                count.checked_add(rustic_core::segment::SegmentWriter::document_slots(doc)?)
                    .ok_or_else(|| rustic_core::error::RusticError::Memory("batch document count overflow".into()))
            }).and_then(|count| writer.prepare_bitmap_capacity(count));
            if let Err(error) = admission {
                let error = EsError::from(error);
                return (0..n).map(|_| Err(error.clone())).collect();
            }
            bitmap_writers.push(writer);
        }
        let admitted = prepared.iter().filter(|(_,token)| token.is_ok()).count();
        // The replication tailer consumes sequence-ordered WAL. Keep allocation
        // and append under one fence; graph preparation above remains parallel.
        // Admitted vector tokens already own rollback. Text assembly below is
        // infallible and stays private until this durability boundary succeeds.
        let mut ws = self.write.lock();
        let first = ws.seq_no + 1;
        ws.seq_no += admitted as i64;
        let primary_term = self.primary_term();
        let mut next_seq_no = first;
        let ordered: Vec<_> = prepared.into_iter().map(|(op, token)| {
            let seq_no = next_seq_no;
            if token.is_ok() { next_seq_no += 1; }
            (seq_no, op, token)
        }).collect();
        let log: Vec<_> = ordered.iter().filter_map(|(seq_no, op, token)| {
            if token.is_err() { return None; }
            let WriteOp::Index { id, source, routing, .. } = op else { unreachable!() };
            Some(TranslogOp::Index { id: id.clone(), source_json: source.clone(),
                seq_no: *seq_no as u64, version: 1, routing: routing.clone() })
        }).collect();
        let mut claimed = false;
        let durability = (|| -> Result<()> {
            if log.is_empty() { return Ok(()); }
            ws.translog.append_many(&log)?;
            let sp = ws.translog.sync_point();
            *self.tail.lock() = Some(sp.clone());
            // Processed (local checkpoint) once installed, below; until then
            // a capture does not contain these ops.
            ws.claim(first, claim_memory);
            claimed = true;
            drop(ws);
            if sync { self.group_sync(&sp.0, sp.1)?; }
            Ok(())
        })();
        if let Err(error) = durability {
            {
                let mut ws = self.write.lock();
                let seq_nos: Vec<i64> = log.iter().map(|op| op.seq_no() as i64).collect();
                if claimed {
                    // Appended (the sync failed): processed, no hole.
                    for &seq_no in &seq_nos { ws.hist.mark(seq_no); }
                    ws.release_claim(first);
                } else {
                    // Never appended: no-ops, and the engine fails itself
                    // (ES translog tragic event; a failed fsync is marked
                    // by `group_sync`).
                    self.mark_tragic(&format!("translog append failed: {error}"));
                    let _ = ws.fill_noops(&seq_nos, primary_term, "translog append failed", false);
                }
            }
            let error = error.to_string();
            return ordered.into_iter().map(|(_, _, token)| match token {
                Err(error) => Err(core_write_error(&error)),
                Ok(_) => Err(EsError::internal(format!("translog durability failure: {error}"))),
            }).collect();
        }
        let batches: Vec<_> = on_vector_pool(|| ordered.into_par_iter().chunks(MUTABLE_TEXT_CHUNK_DOCS).zip(bitmap_writers.into_par_iter()).map(|(chunk, mut writer)| {
            let _admission=admission_context.enter();
            let mut results = Vec::with_capacity(chunk.len());
            for (seq_no, op, token) in chunk {
                let token = match token {
                    Ok(token) => token,
                    Err(error) => { results.push(Err(core_write_error(&error))); continue; }
                };
                let WriteOp::Index { id, source, mut doc, ignored, routing, .. } = op else { unreachable!() };
                doc.fields.push((ID_FIELD, FieldValue::Keyword(id.clone())));
                if !doc.fields.iter().any(|(field,_)| *field == SOURCE_FIELD) {
                    doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
                }
                doc.fields.push((VERSION_FIELD, FieldValue::I64(1)));
                doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
                doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(primary_term as i64)));
                if let Some(value) = &routing { doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(value.clone()))); }
                for field in ignored { doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(field))); }
                writer.append_prepared_document(&doc, &schema, token);
                results.push(Ok(WriteResult { id, version: 1, seq_no, primary_term, result: "created" }));
            }
            (writer, results)
        }).collect());
        let mut results = Vec::with_capacity(n);
        {
            let mut core = self.core.write();
            for (writer, batch) in batches {
                if writer.num_docs() > 0 { core.checkin_writer(writer); }
                results.extend(batch);
            }
            if self.buffer_full(core.writing_bytes() as u64) || core.num_writing_docs() >= self.seal_threshold() {
                if self.bg.served.load(Ordering::Acquire) { self.bg.want_seal.store(true, Ordering::Release); }
                else if let Err(error) = core.seal() { tracing::error!("seal failed: {error}"); }
            }
        }
        if claimed {
            let mut ws = self.write.lock();
            for op in &log { ws.hist.mark(op.seq_no() as i64); }
            ws.release_claim(first);
        }
        let successes = results.iter().filter(|result| result.is_ok()).count() as u64;
        self.counters.index_total.fetch_add(successes, Ordering::Relaxed);
        self.counters.index_failed.fetch_add(n as u64 - successes, Ordering::Relaxed);
        Counters::add_time(&self.counters.index_time_nanos, started.elapsed());
        if successes > 0 { self.sched.note_write(); }
        self.files_changed();
        results
    }

    /// Record the translog's tragic error (first one wins): from now on the
    /// engine refuses ops, and the copy is reported failed.
    fn mark_tragic(&self, error: &str) {
        let mut tragic = self.tragic.lock();
        if tragic.is_none() {
            tracing::error!(path = %self.path.display(), "tragic event, failing the engine: {error}");
            *tragic = Some(error.to_string());
        }
    }

    /// Retire this copy's handle on its files (it is closed or failed, and
    /// another copy may open the same directory): detached seal and merge
    /// jobs still encoding cannot install, rename over or delete anything
    /// there any more (their writes fail), and a rename in progress is
    /// waited for. Reads stay possible.
    ///
    /// A flush in flight is waited for (`flush_lock`), and none starts
    /// after: the shard state file and the translog trim go through the
    /// flush, so a retired copy never rewrites `shard_state.json` or trims
    /// the translog under the copy opened next; the translog is closed for
    /// appends too.
    pub fn close_files(&self) {
        let _flush = self.bg.flush_lock.lock();
        self.bg.files_closed.store(true, Ordering::Release);
        rustic_core::codec::directory::Directory::close(&**self.core.read().directory());
        self.write.lock().translog.fail("the shard copy is closed");
    }

    /// The tragic error that failed this engine, if any.
    pub fn tragic_error(&self) -> Option<String> {
        self.tragic.lock().clone()
    }

    /// The tragic error, once (the shard copy is to be failed and reported
    /// to the master; `None` afterwards and while the engine is healthy).
    pub fn take_tragic_report(&self) -> Option<String> {
        let error = self.tragic_error()?;
        (!self.tragic_reported.swap(true, Ordering::AcqRel)).then_some(error)
    }

    fn tragic_exception(error: &str) -> EsError {
        // A translog failure is a `translog_exception`; any other tragedy
        // (a failed delete of an older copy, a panic mid-request, a failed
        // history side log) an `engine_exception`.
        let kind = if error.starts_with("translog ") { "translog_exception" } else { "engine_exception" };
        EsError::new(500, kind, format!("engine failed after a tragic event: {error}"))
    }

    /// The tragic failure as the error health and APIs report it.
    pub fn tragic_failure(&self) -> Option<EsError> {
        self.tragic_error().map(|error| Self::tragic_exception(&error))
    }

    /// Group commit: make translog appends up to `mark` durable. Called
    /// without the write lock; one fsync covers every append made before it
    /// started, so concurrent writers waiting here mostly find their mark
    /// already synced. `file` is the generation current at `mark` (a later
    /// roll fsyncs it first, so syncing it again is only redundant).
    ///
    /// A failed fsync is final (fsyncgate: the kernel may clear the error
    /// and drop the dirty pages, so a retry that succeeds proves nothing):
    /// it fails the engine and closes the translog, and every waiter, the
    /// ones queued behind it included, fails from then on.
    fn group_sync(&self, file: &std::fs::File, mark: u64) -> Result<()> {
        // The indexing is done: the shard throttle does not cover the fsync
        // (ES takes its indexing throttle per operation, and syncs the
        // translog after), so throttled requests still share group commits.
        HeldThrottle::release();
        if let Some(error) = self.tragic_error() {
            return Err(Self::tragic_exception(&error));
        }
        let mut synced = self.synced.lock();
        // Checked again under `synced`: a failed sync marks the engine
        // tragic before it releases `synced`.
        if let Some(error) = self.tragic_error() {
            return Err(Self::tragic_exception(&error));
        }
        if *synced >= mark {
            return Ok(());
        }
        // A translog closed by a failure outside a group commit (a flush's
        // roll fsync): its unsynced appends are lost for good, a retried
        // fsync proves nothing. Lock order: `synced` → write state (no
        // caller holds the write state here).
        if let Some(reason) = self.write.lock().translog.failed() {
            self.mark_tragic(&reason);
            return Err(Self::tragic_exception(&reason));
        }
        // Everything appended so far is covered too: take the newest sync
        // point (published after every append) before syncing.
        let tail = self.tail.lock().clone();
        #[cfg(test)]
        let injected = self.fail_syncs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).is_ok();
        #[cfg(not(test))]
        let injected = false;
        let r = if injected { Err(std::io::Error::other("injected translog fsync failure")) } else { match &tail {
            Some((cur, _)) if std::ptr::eq(&**cur, file) => file.sync_data(),
            Some((cur, _)) => cur.sync_data().and_then(|_| file.sync_data()),
            None => file.sync_data(),
        } };
        if let Err(error) = r {
            let reason = format!("translog fsync failed: {error}");
            self.mark_tragic(&reason);
            // Callers hold no write lock here; `synced` is released first
            // (nothing takes it under the write lock).
            drop(synced);
            self.write.lock().translog.fail(&reason);
            return Err(error.into());
        }
        *synced = tail.map(|t| t.1).unwrap_or(0).max(mark);
        Ok(())
    }

    fn current(&self, ws: &WriteState, core: &CoreIndex, id: &str) -> (Option<DocAddress>, Option<VersionEntry>) {
        // The copy found is replaced in place: no older one may stay live.
        self.resolve_buffered_delete(core, id);
        let found = Self::lookup(core, id);
        let entry = match ws.versions.get(id) {
            Some(e) => Some(*e),
            None => found.map(|(_, v, s)| VersionEntry { version: v, seq_no: s, deleted: false, primary_term: 0 }),
        };
        (found.map(|f| f.0), entry)
    }

    fn check(cond: &WriteCond, index: &str, uuid: &str, id: &str, cur: Option<VersionEntry>, cur_term: u64) -> Result<u64> {
        let live = cur.filter(|e| !e.deleted);
        if cond.create {
            if let Some(e) = live {
                return Err(EsError::version_conflict(index, uuid, id, format!("version conflict, document already exists (current version [{}])", e.version)));
            }
        }
        if let (Some(s), Some(t)) = (cond.if_seq_no, cond.if_primary_term) {
            match live {
                Some(e) if e.seq_no == s && t == cur_term => {}
                Some(e) => {
                    return Err(EsError::version_conflict(
                        index,
                        uuid,
                        id,
                        format!("version conflict, required seqNo [{s}], primary term [{t}]. current document has seqNo [{}] and primary term [{}]", e.seq_no, cur_term),
                    ))
                }
                None => {
                    return Err(EsError::version_conflict(index, uuid, id, format!("version conflict, required seqNo [{s}], primary term [{t}]. but no document was found")));
                }
            }
        }
        match (cond.version_type.unwrap_or(VersionType::Internal), cond.version) {
            (VersionType::External, Some(v)) | (VersionType::ExternalGte, Some(v)) => {
                let gte = cond.version_type == Some(VersionType::ExternalGte);
                if let Some(e) = cur {
                    let ok = if gte { v >= e.version } else { v > e.version };
                    if !ok {
                        let rel = if gte { "is higher than" } else { "is higher or equal to" };
                        return Err(EsError::version_conflict(index, uuid, id, format!("version conflict, current version [{}] {rel} the one provided [{v}]", e.version)));
                    }
                }
                Ok(v)
            }
            _ => Ok(live.map(|e| e.version + 1).unwrap_or_else(|| cur.map(|e| e.version + 1).unwrap_or(1))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_index(
        &self,
        ws: &mut WriteState,
        core: &mut CoreIndex,
        index: &str,
        uuid: &str,
        id: String,
        source: String,
        mut doc: Document,
        ignored: Vec<String>,
        routing: Option<String>,
        cond: WriteCond,
        auto_id: bool,
        log: &mut Vec<TranslogOp>,
        #[cfg(feature = "mutable-hnsw")]
        prepared: Option<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>,
    ) -> Result<WriteResult> {
        let (addr, cur) = if auto_id { (None, None) } else { self.current(ws, core, &id) };
        if let Some(orig) = self.retry_hit(&cond, &id, cur) {
            return Ok(orig);
        }
        // `if_primary_term` is compared with the document's own term.
        let cur_term = if cond.if_primary_term.is_some() { addr.as_ref().map(|a| hooks_ext::doc_term(core, a)).unwrap_or(1) } else { 1 };
        let version = Self::check(&cond, index, uuid, &id, cur, cur_term)?;
        let existed = cur.map(|e| !e.deleted).unwrap_or(false);
        core.prepare_document_bitmap(&doc)?;
        let version_key = ws.prepare_version_key(&id)?;
        ws.seq_no += 1;
        let seq_no = ws.seq_no;
        if cur.is_some() { self.note_update(seq_no); }
        doc.fields.push((ID_FIELD, FieldValue::Keyword(id.clone())));
        // The parse may have set the stored form (`vector_source::stored_form`).
        if !doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD) {
            doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source.clone())));
        }
        doc.fields.push((VERSION_FIELD, FieldValue::I64(version as i64)));
        doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
        let primary_term = self.primary_term();
        doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(primary_term as i64)));
        if let Some(r) = &routing {
            doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r.clone())));
        }
        for f in ignored {
            doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
        }
        #[cfg(feature = "mutable-hnsw")]
        if let Some(prepared) = prepared {
            debug_assert!(addr.is_none());
            core.add_prepared_document(&doc, prepared?);
        } else {
            core.replace_document(addr.as_ref(), &doc)?;
        }
        #[cfg(not(feature = "mutable-hnsw"))]
        core.replace_document(addr.as_ref(), &doc)?;
        if !auto_id {
            ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: false, primary_term });
        }
        log.push(TranslogOp::Index { id: id.clone(), source_json: source, seq_no: seq_no as u64, version, routing });
        let res = WriteResult { id, version, seq_no, primary_term, result: if existed { "updated" } else { "created" } };
        if !auto_id {
            self.retry_note(&cond, &res, false);
        }
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_delete(&self, ws: &mut WriteState, core: &mut CoreIndex, index: &str, uuid: &str, id: String, cond: WriteCond, log: &mut Vec<TranslogOp>) -> Result<WriteResult> {
        let (addr, cur) = self.current(ws, core, &id);
        if let Some(orig) = self.retry_hit(&cond, &id, cur) {
            return Ok(orig);
        }
        // `if_primary_term` is compared with the document's own term.
        let cur_term = if cond.if_primary_term.is_some() { addr.as_ref().map(|a| hooks_ext::doc_term(core, a)).unwrap_or(1) } else { 1 };
        let version = Self::check(&cond, index, uuid, &id, cur, cur_term)?;
        let existed = addr.is_some();
        let version_key = ws.prepare_version_key(&id)?;
        ws.seq_no += 1;
        let seq_no = ws.seq_no;
        self.note_update(seq_no);
        if let Some(a) = addr {
            core.delete_doc(&a)?;
        }
        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: true, primary_term: self.primary_term() });
        log.push(TranslogOp::Delete { id: id.clone(), seq_no: seq_no as u64, version });
        let res = WriteResult { id, version, seq_no, primary_term: self.primary_term(), result: if existed { "deleted" } else { "not_found" } };
        self.retry_note(&cond, &res, true);
        Ok(res)
    }

    /// Try a bounded batch of wholly new, ordinary documents. A false return
    /// leaves both the shard and prepared documents untouched for ordered replay.
    pub(crate) fn try_replay_new_batch(&self, ops: &[ReplayIndex<'_>], prepared: &mut Vec<crate::document::ParsedDoc>) -> Result<bool> {
        let _shard_owner=self.admitted_owner.enter();
        self.note_replay_write();
        let admission_context=rustic_core::admission::Context::capture();
        use rayon::prelude::*;
        let has_vectors = prepared.iter().any(|p| p.doc.has_vectors());
        #[cfg(not(feature = "mutable-hnsw"))]
        if has_vectors { return Ok(false); }
        if ops.len() < (if has_vectors { 2 } else { 64 }) || ops.len() != prepared.len()
            || prepared.iter().any(|p| !p.doc.nested.is_empty()) {
            return Ok(false);
        }
        let _throttle = self.merge_throttle();
        let mut ids = std::collections::HashSet::with_capacity(ops.len());
        if ops.iter().any(|o| !ids.insert(o.id)) { return Ok(false); }
        let _version_slots = VersionMapSlots::new(self, ops.len())?;
        let version_keys = registry::Keys::prepare(ops.iter().map(|op| op.id))?;
        static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
        static VECTOR_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
        // Replica construction participates in the acknowledgement latency.
        // Preserve the text pool's bound, but let vector preparation use the
        // CPUs assigned to this node just as primary preparation does.
        let selected = if has_vectors { &VECTOR_POOL } else { &POOL };
        let workers = if has_vectors { std::thread::available_parallelism().map_or(4, usize::from) } else { 4 };
        let Some(pool) = selected.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(workers)
            .thread_name(|i| format!("replica-write-{i}")).build().ok()) else { return Ok(false); };
        // Prepare every vector before taking the write state and the core
        // (graph work must not run under them). Invalid chunks keep the
        // ordered fallback; dropping successful tokens (also when the absence
        // check below fails) releases their unpublished reservations. Recovery
        // has already persisted its replica WAL before entering this method.
        // The upgradable read keeps commits out from preparation through
        // installation (a flush capturing the mutable graph must see no
        // ingestion in flight) while append-only batches keep running; the
        // upgrade below then excludes them too, with no window in between.
        let batch = self.batches.upgradable_read();
        #[cfg(feature = "mutable-hnsw")]
        let vector_tokens = if has_vectors {
            let (schema, store) = {
                let core = self.core.read();
                (core.schema.clone(), core.vector_store().clone())
            };
            let _maintenance = store.mutable_ingestion_batch()?;
            // Staged together (`prepare_documents_vectors`); a document
            // without vectors gets an empty token, as one by one.
            let docs: Vec<&rustic_core::types::Document> = prepared.iter().map(|pd| &pd.doc).collect();
            let tokens: Vec<_> = install_without_stealing(pool, || prepare_documents_vectors(&docs, &schema, &store, &admission_context));
            let Ok(tokens) = tokens.into_iter().collect::<rustic_core::error::Result<Vec<_>>>() else { return Ok(false); };
            Some(tokens)
        } else { None };
        // Exclude in-flight append-only batches and commits, then all ordinary
        // writes/replays. The absence check remains valid until installation.
        let _batch = parking_lot::RwLockUpgradableReadGuard::upgrade(batch);
        let mut ws = self.write.lock();
        let mut core = self.core.write();
        // The exclusive core lock keeps this view stable for the whole
        // absence check. Avoid rebuilding the same leaf vector per ID.
        {
            let leaves = core.all_leaves();
            if ops.iter().any(|o| ws.versions.contains_key(o.id) || Self::lookup_in_impl(&leaves, o.id, true, true).is_some()) {
                return Ok(false);
            }
        }
        let terms: Vec<_> = ops.iter().map(|o| self.term_of_locked(&ws, o.seq_no)).collect();
        // Keep the core exclusive for visibility, and reuse its writers and
        // schema caches instead of checking out/resetting them per chunk.
        let docs: Vec<_> = install_without_stealing(pool, || prepared.par_iter_mut().enumerate().map(|(i, pd)| {
            let _admission=admission_context.enter();
            let op = &ops[i];
            let mut doc = std::mem::take(&mut pd.doc);
            doc.fields.push((ID_FIELD, FieldValue::Keyword(op.id.to_string())));
            if !doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD) {
                doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(pd.source.take().unwrap_or_else(|| op.source.to_string()))));
            }
            doc.fields.push((VERSION_FIELD, FieldValue::I64(op.version as i64)));
            doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(op.seq_no)));
            doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(terms[i] as i64)));
            if let Some(r) = op.routing { doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r.to_string()))); }
            for f in std::mem::take(&mut pd.ignored) { doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f))); }
            doc
        }).collect());
        #[cfg(feature = "mutable-hnsw")]
        if let Some(tokens) = vector_tokens {
            let schema = &core.schema;
            let stored_compression = core.stored_compression();
            let writers: Vec<_> = install_without_stealing(pool, || tokens.into_par_iter().chunks(MUTABLE_TEXT_CHUNK_DOCS).enumerate().map(|(chunk, tokens)| {
            let _admission=admission_context.enter();
                let mut writer = rustic_core::segment::SegmentWriter::new(0);
                writer.set_stored_compression(stored_compression);
                let chunk_docs = &docs[chunk * MUTABLE_TEXT_CHUNK_DOCS..][..tokens.len()];
                let slots = chunk_docs.iter().try_fold(0usize, |count, doc| count.checked_add(rustic_core::segment::SegmentWriter::document_slots(doc)?)
                    .ok_or_else(|| rustic_core::error::RusticError::Memory("replay bitmap count overflow".into())))?;
                writer.prepare_bitmap_capacity(slots)?;
                for (doc, token) in chunk_docs.iter().zip(tokens) {
                    writer.append_prepared_document(doc, schema, token);
                }
                Ok::<_, rustic_core::error::RusticError>(writer)
            }).collect::<rustic_core::error::Result<Vec<_>>>())?;
            for writer in writers { core.checkin_writer(writer); }
        } else {
            let core_ref = &mut *core;
            install_without_stealing(pool, || core_ref.add_documents_parallel(&docs, 4))?;
        }
        #[cfg(not(feature = "mutable-hnsw"))]
        {
            let core_ref = &mut *core;
            install_without_stealing(pool, || core_ref.add_documents_parallel(&docs, 4))?;
        }
        self.request_replay_seal(&core);
        drop(core);
        for (op, version_key) in ops.iter().zip(version_keys) {
            ws.seq_no = ws.seq_no.max(op.seq_no);
            ws.hist.tombstones.remove(op.id);
            ws.versions.insert(version_key, VersionEntry { version: op.version, seq_no: op.seq_no, deleted: false, primary_term: 0 });
            ws.hist.mark(op.seq_no);
        }
        prepared.clear();
        self.sched.note_write();
        Ok(true)
    }

    /// Local translog replay at open (ES `IndexingMemoryController` covers
    /// recovering shards too): no scheduler serves the copy yet, so nothing
    /// else writes its buffer out, and every replayed op would stay in
    /// memtables until the final publish, whatever the size of the unflushed
    /// translog. After each replayed chunk, seal here (visibility unchanged)
    /// once the memtables hold this copy's share of the indexing buffer, the
    /// seal threshold of docs, or too many open memtables, as an unserved
    /// copy's own writes do. Runs under the replay, so the seal's memory is
    /// forced, never refused. A served copy hands it to its seal workers.
    pub fn seal_replay_if_buffer_full(&self) {
        let mut ws = self.write.lock();
        let mut core = self.core.write();
        if !(self.buffer_full(core.writing_bytes() as u64) || core.num_writing_docs() >= self.seal_threshold() || core.num_open_memtables() > MAX_OPEN_MEMTABLES) {
            return;
        }
        if self.bg.served.load(Ordering::Acquire) {
            self.bg.want_seal.store(true, Ordering::Release);
            return;
        }
        self.files_changed();
        match core.seal() {
            Ok(()) => {
                self.note_buffered(core.writing_bytes() as u64);
                // As after any seal: the version map stays bounded by the
                // memtables, not by every id replayed since the last flush.
                ws.prune_versions(&self.unresolved.lock());
            }
            // The memtables stay as they are; the next chunk or the final
            // publish tries again.
            Err(e) => tracing::error!("translog replay seal failed: {e}"),
        }
    }

    /// Replica/recovery writes need the same bounded writer lifecycle as
    /// primaries, even when refresh is disabled or searches are idle.
    fn request_replay_seal(&self, core: &CoreIndex) {
        if self.bg.served.load(Ordering::Acquire) && core.num_writing_docs() >= self.seal_threshold() {
            self.bg.want_seal.store(true, Ordering::Release);
        }
    }

    /// The newest seq_no this copy knows for `id` (live copy, version map,
    /// delete tombstone).
    pub fn known_seq_no(&self, id: &str) -> Option<i64> {
        let live = { let core = self.core.read(); Self::lookup(&core, id).map(|f| f.2) };
        let ws = self.write.lock();
        hooks_ext::known_seq(&ws, live, id)
    }

    /// Replicated ops applied concurrently, DWPT-style ([`Shard::apply_staged`]
    /// for replicas): staleness is decided per op by seq_no under the write
    /// state (the version map and delete tombstones: an older seq_no for an
    /// id never overwrites a newer one, whatever order batches arrive in);
    /// documents invert into a pooled memtable off every lock; check-in
    /// resolves copies in seq_no order. Returns, per op, whether it applied
    /// (`false`: stale or a duplicate; it is still marked processed). The
    /// caller (replication) owns durability and fencing.
    pub fn replay_staged(&self, mut ops: Vec<StagedReplay>) -> Result<crate::translog::AdmittedBatch<bool>> {
        let _shard_owner=self.admitted_owner.enter();
        self.note_replay_write();
        let claim_memory=Self::prepare_claim_node()?;
        let admission_context=rustic_core::admission::Context::capture();
        if let Some(error) = self.tragic_error() {
            return Err(Self::tragic_exception(&error));
        }
        enum Planned {
            Index { delete_key: registry::DeleteKey, id: String, seq_no: i64, older: bool, doc: Document, version: u64, term: u64, source: Option<String>,
                routing: Option<String>, ignored: Vec<String>,
                #[cfg(feature = "mutable-hnsw")]
                token: Option<rustic_core::segment::PreparedDocumentVectors> },
            Delete { id: String, seq_no: i64 },
        }
        let _throttle = self.merge_throttle();
        self.files_changed();
        let n = ops.len();
        let mut per = std::mem::size_of::<Planned>() + 2 * std::mem::size_of::<Option<i64>>() + 2 * std::mem::size_of::<i64>()
            + std::mem::size_of::<(String, Arc<PendingDoc>, Arc<PoolCheckin>)>() + std::mem::size_of::<(String, i64)>();
        #[cfg(feature = "mutable-hnsw")]
        { per += std::mem::size_of::<Option<rustic_core::segment::PreparedDocumentVectors>>()
            + std::mem::size_of::<Option<rustic_core::error::Result<rustic_core::segment::PreparedDocumentVectors>>>(); }
        let bytes = n.checked_mul(per).ok_or_else(|| EsError::new(429,"circuit_breaking_exception","replay workspace overflow"))?;
        // Forced, never refused: callers run this under `memory::Replay`
        // (a live one for a replicated batch, already in the replica WAL by
        // now and admitted before its append; a recovery one for replica
        // WAL replay and peer recovery). Refusing here would leave durable
        // ops unmarked.
        let _workspace = crate::memory::breakers().reserve_request(bytes as u64,"replica replay concrete containers")?;
        let result_memory = crate::memory::breakers().reserve_request(n as u64,"replica replay results")?;
        let _batch = self.batches.read();
        for op in &mut ops {
            if let StagedReplay::Index { doc, id, source, routing, ignored, .. } = op {
                PendingDoc::prepare_document(doc, id, source, routing.as_ref(), ignored)?;
            }
        }
        let _pending_slots = PendingMapSlots::new(self, ops.iter().filter(|op| matches!(op, StagedReplay::Index { .. })).count())?;
        let _version_slots = VersionMapSlots::new(self, n)?;
        let version_keys = registry::Keys::prepare(ops.iter().map(|op| match op {
            StagedReplay::Index { id, .. } | StagedReplay::Delete { id, .. } => id.as_str(),
            StagedReplay::Skip { .. } => "",
        }))?;
        let delete_keys = registry::DeleteKeys::prepare(ops.iter().map(|op| match op { StagedReplay::Index { id, .. } | StagedReplay::Delete { id, .. } => id.as_str(), StagedReplay::Skip { .. } => "" }))?;
        let slots = ops.iter().try_fold(0usize, |count, op| {
            let slots = match op { StagedReplay::Index { doc, .. } => rustic_core::segment::SegmentWriter::document_slots(doc)?, _ => 0 };
            count.checked_add(slots).ok_or_else(|| rustic_core::error::RusticError::Memory("replay bitmap count overflow".into()))
        })?;
        let prepared_pool = PreparedPool::new(self, slots)?;
        let schema = self.core_schema();
        #[cfg(feature = "mutable-hnsw")]
        let mut tokens: Vec<Option<rustic_core::segment::PreparedDocumentVectors>> = {
            let vectors = ops.iter().any(|op| matches!(op, StagedReplay::Index { doc, .. } if doc.has_vectors()));
            if vectors {
                let store = {
                    let core = self.core.write();
                    core.enable_new_mutable_vector_fields()?;
                    core.vector_store().clone()
                };
                let (schema, store) = (&schema, &store);
                // Staged together (`prepare_documents_vectors`).
                let (positions, docs): (Vec<usize>, Vec<&rustic_core::types::Document>) = ops.iter().enumerate()
                    .filter_map(|(i, op)| match op { StagedReplay::Index { doc, .. } if doc.has_vectors() => Some((i, doc)), _ => None })
                    .unzip();
                let prepared = prepare_documents_vectors(&docs, schema, store, &admission_context);
                let mut out: Vec<Option<rustic_core::segment::PreparedDocumentVectors>> = (0..n).map(|_| None).collect();
                for (i, token) in positions.into_iter().zip(prepared) { out[i] = Some(token?); }
                out
            } else {
                (0..n).map(|_| None).collect()
            }
        };
        // MSU (ES): with the primary's max seq_no of updates and deletes
        // known, an index op above it is the first copy of its id: no live
        // copy to look up, no older copy to delete.
        let fresh_above = self.msu_from_primary.load(Ordering::Acquire).then(|| self.msu.load(Ordering::Acquire));
        let fresh = |op: &StagedReplay| matches!((op, fresh_above), (StagedReplay::Index { seq_no, .. }, Some(msu)) if *seq_no > msu);
        let lookup_all = |core: &CoreIndex, ops: &[StagedReplay]| -> Vec<Option<i64>> {
            let leaves = core.all_leaves();
            ops.iter().map(|op| match op {
                _ if fresh(op) => None,
                StagedReplay::Index { id, .. } | StagedReplay::Delete { id, .. } => Self::lookup_in(&leaves, id, true).map(|f| f.2),
                StagedReplay::Skip { .. } => None,
            }).collect()
        };
        let mut prunes = self.write.lock().prunes;
        let mut found = { let core = self.core.read(); lookup_all(&core, &ops) };
        // Pruned meanwhile: again off the write state (see `apply_staged`).
        for _ in 0..3 {
            let now = self.write.lock().prunes;
            if now == prunes {
                break;
            }
            prunes = now;
            found = lookup_all(&self.core.read(), &ops);
        }
        let mut applied = Vec::with_capacity(n);
        let mut planned = Vec::with_capacity(n);
        let mut seqs = Vec::with_capacity(n);
        let claim = {
            let mut ws = self.write.lock();
            if ws.prunes != prunes {
                found = lookup_all(&self.core.read(), &ops);
            }
            for (i, (((op, live), version_key), delete_key)) in ops.into_iter().zip(found).zip(version_keys).zip(delete_keys).enumerate() {
                #[cfg(feature = "mutable-hnsw")]
                let token = tokens[i].take();
                #[cfg(not(feature = "mutable-hnsw"))]
                let _ = i;
                let (id, seq_no) = match &op {
                    StagedReplay::Index { id, seq_no, .. } | StagedReplay::Delete { id, seq_no, .. } => (id.clone(), *seq_no),
                    StagedReplay::Skip { seq_no } => {
                        seqs.push(*seq_no);
                        ws.seq_no = ws.seq_no.max(*seq_no);
                        applied.push(false);
                        continue;
                    }
                };
                seqs.push(seq_no);
                ws.seq_no = ws.seq_no.max(seq_no);
                if hooks_ext::known_seq(&ws, live, &id).is_some_and(|c| c >= seq_no) {
                    applied.push(false);
                    continue;
                }
                applied.push(true);
                let first_copy = fresh(&op);
                match op {
                    StagedReplay::Index { id, source, doc, ignored, routing, seq_no, version } => {
                        let term = self.term_of_locked(&ws, seq_no);
                        let source = (!doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD)).then_some(source);
                        // The append path needs no version-map entry for the
                        // id either (a reordered op of it may have landed).
                        let older = !(first_copy && !ws.versions.contains_key(&id) && !ws.hist.tombstones.contains_key(&id));
                        if older && (live.is_some() || ws.versions.contains_key(&id)) { self.note_update(seq_no); }
                        ws.hist.tombstones.remove(&id);
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: false, primary_term: term });
                        planned.push(Planned::Index { delete_key, id, seq_no, older, doc, version, term, source, routing, ignored,
                            #[cfg(feature = "mutable-hnsw")]
                            token });
                    }
                    StagedReplay::Delete { id, seq_no, version } => {
                        self.note_update(seq_no);
                        let term = self.term_of_locked(&ws, seq_no);
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: true, primary_term: term });
                        ws.hist.tombstones.insert(id.clone(), (seq_no, version));
                        planned.push(Planned::Delete { id, seq_no });
                    }
                    StagedReplay::Skip { .. } => unreachable!("skipped above"),
                }
            }
            // Version-map claim of the planned ops until their check-in.
            let claim = planned.iter().map(|p| match p { Planned::Index { seq_no, .. } | Planned::Delete { seq_no, .. } => *seq_no }).min();
            if let Some(first) = claim {
                ws.claim(first, claim_memory);
            }
            claim
        };
        let (mut inverting, mut bitmap_credits) = prepared_pool.into_inverting(claim, seqs.clone());
        let mut pending = Vec::with_capacity(n);
        let mut deletes = Vec::with_capacity(n);
        let stored_compression = self.stored_compression();
        for item in planned {
            let (holding, pooled) = inverting.parts();
            self.observe_freeze(holding, pooled);
            self.maybe_injected_panic();
            match item {
                Planned::Index { delete_key, id, seq_no, older, mut doc, version, term, source, routing, ignored, #[cfg(feature = "mutable-hnsw")] token } => {
                    // Metadata fields as `replay_index` writes them.
                    doc.fields.push((ID_FIELD, FieldValue::Keyword(id.clone())));
                    if let Some(source) = source {
                        doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
                    }
                    doc.fields.push((VERSION_FIELD, FieldValue::I64(version as i64)));
                    doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
                    doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(term as i64)));
                    if let Some(r) = routing {
                        doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r)));
                    }
                    for f in ignored {
                        doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
                    }
                    pooled.writer.set_stored_compression(stored_compression);
                    pooled.writer.ensure_document_bitmap(&doc, &mut bitmap_credits);
                    #[cfg(feature = "mutable-hnsw")]
                    match token {
                        Some(token) => pooled.writer.append_prepared_document(&doc, &schema, token),
                        None => pooled.writer.add_document(&doc, &schema),
                    }
                    #[cfg(not(feature = "mutable-hnsw"))]
                    pooled.writer.add_document(&doc, &schema);
                    pooled.admit_next_doc(holding);
                    pooled.docs.push(PoolDoc { id: delete_key, seq_no, local: pooled.writer.num_docs() - 1, older });
                    pending.push((id, Arc::new(PendingDoc { seq_no, version, primary_term: term, doc }), pooled.checked_in.clone()));
                }
                Planned::Delete { id, seq_no } => deletes.push((id, seq_no)),
            }
        }
        let (mut holding, mut pooled) = inverting.finish();
        self.observe_freeze(&mut holding, &mut pooled);
        self.complete_pooled(holding, pooled, pending, &deletes, &seqs, true)?;
        if applied.iter().any(|a| *a) { self.sched.note_write(); }
        self.files_changed();
        Ok(crate::translog::AdmittedBatch::from_reserved(applied, result_memory))
    }

    /// Replay one translog op during recovery (no concurrency checks).
    pub fn replay_index(&self, id: String, source: String, mut doc: Document, ignored: Vec<String>, routing: Option<String>, seq_no: i64, version: u64) -> Result<()> {
        let _shard_owner=self.admitted_owner.enter();
        self.note_replay_write();
        let _throttle = self.merge_throttle();
        let mut ws = self.write.lock();
        let mut core = self.core.write();
        let version_key = ws.prepare_version_key(&id)?;
        self.resolve_buffered_delete(&core, &id);
        let found = Self::lookup(&core, &id);
        // Not newer than what this copy holds for the id (out-of-order
        // replica op, or already applied): only mark it processed.
        if hooks_ext::known_seq(&ws, found.as_ref().map(|f| f.2), &id).is_some_and(|c| c >= seq_no) {
            ws.seq_no = ws.seq_no.max(seq_no);
            ws.hist.mark(seq_no);
            return Ok(());
        }
        let term = self.term_of_locked(&ws, seq_no);
        doc.fields.push((ID_FIELD, FieldValue::Keyword(id.clone())));
        // The parse may have set the stored form (`vector_source::stored_form`).
        if !doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD) {
            doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
        }
        doc.fields.push((VERSION_FIELD, FieldValue::I64(version as i64)));
        doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
        doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(term as i64)));
        if let Some(r) = routing {
            doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r)));
        }
        for f in ignored {
            doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
        }
        core.replace_document(found.as_ref().map(|found| &found.0), &doc)?;
        self.request_replay_seal(&core);
        self.sched.note_write();
        ws.seq_no = ws.seq_no.max(seq_no);
        ws.hist.tombstones.remove(&id);
        let primary_term = self.term_of_locked(&ws, seq_no);
        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: false, primary_term });
        ws.hist.mark(seq_no);
        Ok(())
    }

    pub fn replay_delete(&self, id: &str, seq_no: i64, version: u64) -> Result<()> {
        let _shard_owner=self.admitted_owner.enter();
        self.note_replay_write();
        let mut ws = self.write.lock();
        let mut core = self.core.write();
        let version_key = ws.prepare_version_key(&id)?;
        self.resolve_buffered_delete(&core, id);
        let found = Self::lookup(&core, id);
        if hooks_ext::known_seq(&ws, found.as_ref().map(|f| f.2), id).is_some_and(|c| c >= seq_no) {
            ws.seq_no = ws.seq_no.max(seq_no);
            ws.hist.mark(seq_no);
            return Ok(());
        }
        if let Some((a, _, _)) = found {
            core.delete_doc(&a)?;
            self.sched.note_write();
        }
        ws.seq_no = ws.seq_no.max(seq_no);
        let primary_term = self.term_of_locked(&ws, seq_no);
        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: true, primary_term });
        ws.hist.tombstones.insert(id.to_string(), (seq_no, version));
        ws.hist.mark(seq_no);
        ws.hist.prune_tombstones();
        Ok(())
    }

    /// Record `seq_nos` as no-ops of `term` that never reached this copy's
    /// engine (a promoted replica's held no-ops and gap fills, ES
    /// `fillSeqNoGaps`): marked processed in the history, and the sequence
    /// generation continues above them.
    /// Fill every seq_no not processed up to the max seq_no with a no-op
    /// (ES fills a primary's gaps at the end of store recovery): batches
    /// append to the translog out of seq_no order, so a crash can leave a
    /// later batch durable and an earlier one never written; without the
    /// no-ops the local checkpoint would never pass the hole. Returns how
    /// many were filled.
    /// The no-ops are durable (fsynced) before the seq_nos count as
    /// processed; a failure to record them fails the engine.
    pub fn fill_gaps_to_max_seq_no(&self, term: u64) -> Result<usize> {
        let mut ws = self.write.lock();
        let max = ws.seq_no;
        let gaps: Vec<i64> = (ws.hist.checkpoint + 1..=max).filter(|seq_no| !ws.hist.is_processed(*seq_no)).collect();
        if let Err(error) = ws.fill_noops(&gaps, term, "filling seq_no gap at the end of local recovery", true) {
            drop(ws);
            self.mark_tragic(&format!("history side log append failed: {error}"));
            return Err(error);
        }
        Ok(gaps.len())
    }

    /// [`Shard::fill_gaps_to_max_seq_no`] for given seq_nos (a promoted
    /// replica's no-ops): durable before processed; a failure fails the
    /// engine.
    pub fn fill_seq_no_gaps(&self, seq_nos: &[i64], term: u64, reason: &str) -> Result<()> {
        let _shard_owner=self.admitted_owner.enter();
        let Some(&max) = seq_nos.iter().max() else { return Ok(()) };
        let mut ws = self.write.lock();
        if let Err(error) = ws.fill_noops(seq_nos, term, reason, true) {
            drop(ws);
            self.mark_tragic(&format!("history side log append failed: {error}"));
            return Err(error);
        }
        ws.seq_no = ws.seq_no.max(max);
        Ok(())
    }

    /// Realtime get.
    pub fn get(&self, id: &str) -> Result<Option<GetResult>> {
        self.get_mode(id, true)
    }

    /// Get, realtime or (`realtime=false`) as searches see the doc.
    pub fn get_mode(&self, id: &str, realtime: bool) -> Result<Option<GetResult>> {
        self.counted_get(|| if realtime { self.get_inner(id) } else { self.get_visible(id) })
    }

    fn counted_get(&self, load: impl FnOnce() -> Result<Option<GetResult>>) -> Result<Option<GetResult>> {
        let t0 = std::time::Instant::now();
        let r = {
            let _current = CurrentGuard::new(vec![(&self.counters.get_current, 1)]);
            load()
        };
        let c = &self.counters;
        let el = t0.elapsed();
        c.get_total.fetch_add(1, Ordering::Relaxed);
        Counters::add_time(&c.get_time_nanos, el);
        if matches!(&r, Ok(Some(_))) {
            c.get_exists_total.fetch_add(1, Ordering::Relaxed);
            Counters::add_time(&c.get_exists_time_nanos, el);
        } else if matches!(&r, Ok(None)) {
            c.get_missing_total.fetch_add(1, Ordering::Relaxed);
            Counters::add_time(&c.get_missing_time_nanos, el);
        }
        r
    }

    /// Realtime lookup that does not count towards the `get` stats (internal
    /// pre-checks of write requests, which ES does not count as gets).
    pub fn peek(&self, id: &str) -> Result<Option<GetResult>> {
        self.get_inner(id)
    }

    /// Set the [`SourcePatch`] of loaded sources.
    pub fn set_source_patch(&self, p: Option<SourcePatch>) {
        *self.patch.write() = p;
    }

    /// The `_source` of the document at `addr` as loaded (see [`SourcePatch`]).
    fn loaded_source(&self, core: &CoreIndex, addr: &rustic_core::types::DocAddress, doc: &Document) -> Option<String> {
        let raw = doc.get(SOURCE_FIELD).and_then(|v| v.as_text())?;
        let patch = self.patch.read().clone();
        if let Some(p) = patch {
            let leaves = core.all_leaves();
            if let Some(leaf) = leaves.get(addr.segment_id as usize) {
                if let Some(s) = p(raw, &|fid| leaf.vector(fid, addr.doc_id), &crate::synthetic::LeafColumns { leaf, doc: addr.doc_id }) {
                    return Some(s);
                }
            }
        }
        Some(raw.to_string())
    }

    /// The pending document of `id` while it is the id's latest op (its
    /// pooled memtable is not checked in yet).
    pub(crate) fn pending_doc(&self, id: &str) -> Option<Arc<PendingDoc>> {
        let ws = self.write.lock();
        ws.versions.get(id).and_then(|e| ws.pending_docs.get(id).filter(|p| p.seq_no == e.seq_no && !e.deleted).cloned())
    }

    /// Realtime view of `id` from the version map: `Some(pending)` for an
    /// op not checked in yet, `Err(())` when its latest op is a delete
    /// (sequenced, possibly not applied to the core yet), `Ok(None)`: the
    /// core holds the answer.
    fn realtime_entry(&self, id: &str) -> std::result::Result<Option<Arc<PendingDoc>>, ()> {
        let ws = self.write.lock();
        match ws.versions.get(id) {
            Some(e) if e.deleted => Err(()),
            // Sequencing reserves the next version before its off-lock WAL
            // append. Until that append publishes its pending document, the
            // preceding acknowledged pooled document is still the realtime
            // answer. Requiring equality exposes an older core copy during
            // every such append window. Check-in removes pending entries only
            // after their document is installed under these same locks.
            Some(e) => Ok(ws.pending_docs.get(id).filter(|p| p.seq_no <= e.seq_no).cloned()),
            None => Ok(None),
        }
    }

    fn get_inner(&self, id: &str) -> Result<Option<GetResult>> {
        // The latest op of `id` may be in a pooled memtable not checked in
        // yet: serve it from the version map (ES realtime GET reads the
        // version map / translog the same way). Pending docs are complete
        // documents with their metadata fields.
        let Ok(pending) = self.realtime_entry(id) else { return Ok(None) };
        if let Some(p) = pending {
            let raw = p.doc.get(SOURCE_FIELD).and_then(|v| v.as_text()).map(str::to_string);
            let patch = self.patch.read().clone();
            let source = match (raw, patch) {
                (Some(raw), Some(patch)) => Some(patch(&raw, &|field| p.doc.fields.iter()
                    .find_map(|(f, v)| match v { FieldValue::Vector(row) if *f == field => Some(row.clone()), _ => None }), &p.doc)
                    .unwrap_or(raw)),
                (raw, _) => raw,
            };
            let routing = p.doc.get(ROUTING_FIELD).and_then(|v| v.as_text()).map(str::to_string);
            return Ok(Some(GetResult { source, version: p.version, seq_no: p.seq_no, primary_term: p.primary_term, routing, doc: p.doc.try_clone()? }));
        }
        let core = self.core.read();
        let Some((addr, version, seq_no)) = Self::lookup(&core, id) else { return Ok(None) };
        let primary_term = hooks_ext::doc_term(&core, &addr);
        let doc = core.get_doc(&addr)?;
        let source = self.loaded_source(&core, &addr, &doc);
        let routing = doc.get(ROUTING_FIELD).and_then(|v| v.as_text()).map(str::to_string);
        Ok(Some(GetResult { source, version, seq_no, primary_term, routing, doc }))
    }

    /// Non-realtime get (`realtime=false`): the doc as searches see it
    /// (ES reads it through the external searcher).
    fn get_visible(&self, id: &str) -> Result<Option<GetResult>> {
        self.before_search();
        let core = self.core.read();
        let leaves = core.leaves();
        let Some((addr, version, seq_no)) = Self::lookup_in(&leaves, id, false) else { return Ok(None) };
        let primary_term = hooks_ext::doc_term(&core, &addr);
        let doc = core.get_doc(&addr)?;
        let source = self.loaded_source(&core, &addr, &doc);
        let routing = doc.get(ROUTING_FIELD).and_then(|v| v.as_text()).map(str::to_string);
        Ok(Some(GetResult { source, version, seq_no, primary_term, routing, doc }))
    }

    /// ES refresh: make every write so far visible to searches.
    pub fn refresh(&self) -> Result<()> {
        let _shard_owner=self.admitted_owner.enter();
        let _refreshing = self.refreshing();
        let t0 = std::time::Instant::now();
        let _phases = RefreshPhases::begin();
        let published = self.publish();
        let elapsed = t0.elapsed();
        if elapsed >= SLOW_REFRESH {
            let p = RefreshPhases::current();
            let ms = |d: std::time::Duration| d.as_millis();
            tracing::warn!(path = %self.path.display(), total_ms = ms(elapsed), checkin_lock_wait_ms = ms(p.checkin_lock),
                handoff_wait_ms = ms(p.handoff), write_lock_wait_ms = ms(p.write_lock), core_lock_wait_ms = ms(p.core_lock),
                seal_first = p.seal_first, seal_lock_wait_ms = ms(p.seal_lock), buffer_encode_ms = ms(p.encode),
                encoded_bytes = p.encoded_bytes, in_flight_seal_wait_ms = ms(p.in_flight_seal),
                publish_core_wait_ms = ms(p.publish_core_lock), publish_ms = ms(p.publish), sorted_fence_wait_ms = ms(p.sorted_fence), "slow engine refresh phases");
        }
        published?;
        self.counters.refresh_total.fetch_add(1, Ordering::Relaxed);
        Counters::add_time(&self.counters.refresh_time_nanos, elapsed);
        Ok(())
    }

    /// Whether a refresh must seal what it would publish first (more than
    /// [`LARGE_PUBLISH_BYTES`] of memtables; index-sorted cores seal at
    /// every refresh anyway).
    fn publish_needs_seal(&self, core: &CoreIndex) -> bool {
        let limit = match self.bg.large_publish_override.load(Ordering::Acquire) { 0 => LARGE_PUBLISH_BYTES, n => n };
        core.index_sort().is_none() && core.num_unpublished_bytes() > limit
    }

    /// Seal what a large publish would make searchable before it does
    /// (visibility unchanged; the publish that follows makes the segments
    /// searchable). A copy the scheduler serves seals it on this thread
    /// ([`Shard::seal_unpublished_served`]), as a Lucene refresh writes its
    /// own segments. A copy no scheduler serves claims coalesced jobs,
    /// encodes them here on the encode pool and installs them.
    fn seal_unpublished(&self) -> Result<()> {
        self.files_changed();
        if self.bg.served.load(Ordering::Acquire) {
            self.seal_unpublished_served();
            return Ok(());
        }
        let mut first_err = None;
        let mut close = true;
        loop {
            let t = std::time::Instant::now();
            let permit = self.bg.seal_running.acquire();
            // Claimed under the seal lock, as a background seal claims.
            let job = {
                let _pin = self.bg.seal_lock.lock();
                note_refresh_phase(|p| p.seal_lock += t.elapsed());
                let mut core = self.core.write();
                let job = core.begin_seal_bytes(close, CoreIndex::COALESCE_MAX_BYTES)?;
                self.note_buffered(core.writing_bytes() as u64);
                job
            };
            close = false;
            let Some(job) = job else { break };
            let t = std::time::Instant::now();
            let sealed = job.run();
            note_refresh_phase(|p| p.encode += t.elapsed());
            let _pin = self.bg.seal_lock.lock();
            if let Err(e) = self.core.write().finish_seal(sealed) {
                first_err.get_or_insert(e);
            }
            drop(permit);
        }
        self.bg.want_merge.store(true, Ordering::Release);
        match first_err {
            Some(e) => Err(e.into()),
            None => Ok(()),
        }
    }

    /// [`Shard::seal_unpublished`] of a copy the scheduler serves. The
    /// refresh claims, at one boundary, every memtable it is about to
    /// publish that no background seal claimed (the memtables being written,
    /// closed now, and the unclaimed parked ones: never the published ones,
    /// which background seals write out), encodes the chunks on this thread
    /// (the encode pool, at most `MAX_PARALLEL_SEALS` at once, one seal-job
    /// permit) and installs each as it is encoded (the seal lock only to
    /// claim and to install, as a background seal). Memtables background
    /// seals had already claimed are encoding on other workers: the refresh
    /// waits for them at most [`IN_FLIGHT_SEAL_WAIT`]. It never waits for the
    /// background seal queue itself, so a copy whose background seals fall
    /// behind still refreshes at the pace of its own encoding, and its
    /// refresh is work done on the backlog rather than a timeout.
    ///
    /// Memory: the claim and the encoding take ordinary admission (forced
    /// while a recovery replay's seals are, as background seals). When a
    /// permit, the seal lock or the admission is not to be had in time, the
    /// rest is published as memtables (the same bytes, already charged:
    /// publishing never adds memory) and the whole buffer is handed to the
    /// background seals. Unsealed bytes stay bounded by the write path
    /// (buffer-driven write-outs, the shard throttle and the write stall,
    /// [`Shard::merge_throttle`]), not by a refresh wait.
    fn seal_unpublished_served(&self) {
        let deadline = std::time::Instant::now() + PUBLISH_SEAL_WAIT;
        let seal = || {
            let t = std::time::Instant::now();
            let Some(permit) = self.bg.seal_running.acquire_until(deadline) else {
                note_refresh_phase(|p| p.seal_lock += t.elapsed());
                self.write_indexing_buffer();
                return;
            };
            let claimed = {
                let pin = self.bg.seal_lock.lock_until(deadline);
                note_refresh_phase(|p| p.seal_lock += t.elapsed());
                let Some(_pin) = pin else {
                    self.write_indexing_buffer();
                    return;
                };
                let mut core = self.core.write();
                let jobs = core.begin_seal_unpublished_chunks(CoreIndex::COALESCE_MAX_BYTES);
                self.note_buffered(core.writing_bytes() as u64);
                // Every parked memtable is claimed now: by this refresh, or
                // by a background seal in flight.
                jobs.map(|jobs| (jobs, core.parked_memtable_ids()))
            };
            let (jobs, parked) = match claimed {
                Ok(claimed) => claimed,
                Err(error) => {
                    tracing::warn!("refresh seal admission for [{}] failed (published as memtables, written out in the background): {error}", self.path.display());
                    self.write_indexing_buffer();
                    return;
                }
            };
            let bytes = jobs.iter().map(rustic_core::index::SealJob::memtable_bytes).fold(0, u64::saturating_add);
            let t0 = std::time::Instant::now();
            let finished = self.finish_seal_chunks_pinned(jobs);
            note_refresh_phase(|p| { p.encode += t0.elapsed(); p.encoded_bytes += bytes; });
            match finished {
                // Its chunks encode concurrently: the rate is the whole's.
                Ok(()) => self.note_seal_rate(bytes, t0.elapsed()),
                Err(error) => {
                    tracing::error!("refresh seal of [{}] failed (its memtables stay for a later seal): {error}", self.path.display());
                    self.write_indexing_buffer();
                }
            }
            drop(permit);
            // The background seals in flight are encoding on their own
            // workers: wait for them (bounded), so their memtables become
            // segments before they become searchable.
            let waited = std::time::Instant::now();
            let in_flight = waited + IN_FLIGHT_SEAL_WAIT;
            while self.core.read().holds_parked(&parked) && std::time::Instant::now() < in_flight {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            note_refresh_phase(|p| p.in_flight_seal += waited.elapsed());
        };
        if self.replay_seals_forced() { rustic_core::admission::forced(seal) } else { seal() }
        self.bg.want_merge.store(true, Ordering::Release);
    }

    /// [`Shard::finish_seal_chunks`] without the caller holding the seal
    /// lock: each install takes it, as a background seal's does.
    fn finish_seal_chunks_pinned(&self, jobs: Vec<rustic_core::index::SealJob>) -> Result<()> {
        let mut first_error = None;
        let mut jobs = PendingSealChunks::new(self, jobs);
        jobs.run(|sealed| {
            if let Err(error) = self.install_seal_retrying(sealed, |core, sealed| core.try_finish_seal_captured(sealed).map(|r| r.map(|_| ())), true) {
                first_error.get_or_insert(error);
            }
        });
        first_error.map_or(Ok(()), |error| Err(error.into()))
    }

    /// Publish pending writes (no stats); whether anything changed.
    pub(crate) fn publish(&self) -> Result<bool> {
        let _defer = DeferCheckinDrops::enter();
        // The write generation this refresh covers: taken before the
        // completed writes are, so a write landing meanwhile keeps the copy
        // dirty (cleared only for what was published).
        let generation = self.sched.note_refresh();
        // Every op processed by now is published by this refresh if every
        // writer hands off (the refreshed checkpoint).
        let processed = self.processed_checkpoint();
        // The listeners' ticket before the pooled memtables are taken: a
        // write acknowledged before it returned its memtable before.
        let ticket = self.listeners.begin();
        // Every write completed before now joins (pooled memtables frozen
        // at a document boundary: no whole request is waited for). The
        // ticket completes only if every holder handed off.
        let ticket = self.checkin_completed_writes(None).then_some(ticket);
        // A large buffer becomes segments before it becomes searchable.
        let seal_first = self.publish_needs_seal(&self.core.read());
        note_refresh_phase(|p| p.seal_first = seal_first);
        if seal_first {
            self.seal_unpublished()?;
        }
        // Unsorted refresh and sorted metadata-only refresh never encode. In
        // particular they may publish visibility during detached commit I/O.
        let t = std::time::Instant::now();
        let sorted = {
            let core = self.core.write();
            note_refresh_phase(|p| p.publish_core_lock += t.elapsed());
            let t = std::time::Instant::now();
            if !Self::needs_sorted_seal(&core) {
                let published = self.publish_locked_with(core, false, ticket);
                note_refresh_phase(|p| p.publish += t.elapsed());
                Err(published)
            } else { Ok(()) }
        };
        let published = match sorted {
            Err(published) => published,
            Ok(()) => {
                let t = std::time::Instant::now();
                let published = self.publish_sorted(ticket);
                note_refresh_phase(|p| p.publish += t.elapsed());
                published
            }
        };
        self.note_published(generation, ticket.is_some(), &published);
        if ticket.is_some() && published.is_ok() {
            self.note_refreshed_checkpoint(processed);
        }
        published
    }

    /// A refresh started at `generation` finished: the copy is clean (for
    /// the scheduler) only if it published and every holder had handed its
    /// writes off to it.
    fn note_published(&self, generation: u64, complete: bool, published: &Result<bool>) {
        if complete && published.is_ok() {
            self.sched.refreshed(generation);
        }
    }

    /// Whether a refresh must first seal the writers of an index-sorted
    /// core (sorted writers become sorted segments before they become
    /// searchable).
    fn needs_sorted_seal(core: &CoreIndex) -> bool {
        core.index_sort().is_some() && !core.can_refresh_during_commit()
    }

    /// The sorted refresh: fence writes, but release the core during
    /// encoding (detached seal) so current searchers, GET and statistics
    /// remain available; then the short publication.
    fn publish_sorted(&self, ticket: Option<u64>) -> Result<bool> {
        let t = std::time::Instant::now();
        let _seal = self.seal_fences();
        note_refresh_phase(|p| p.sorted_fence += t.elapsed());
        self.seal_and_publish_sorted(ticket)
    }

    /// The sorted refresh with `seal_lock` held by the caller: close the
    /// memtables being written into a detached seal (short core lock), sort
    /// and encode them off every lock while writes continue into fresh
    /// memtables, install (deletes taken meanwhile are remapped), then
    /// publish only sealed segments and pending deletes (short swap).
    fn seal_and_publish_sorted(&self, ticket: Option<u64>) -> Result<bool> {
        // `ticket` was taken before the pooled memtables were checked in:
        // what this refresh makes visible is what `begin_seal` closes now.
        let started = std::time::Instant::now();
        let job = {
            let mut core = self.core.write();
            // What this seal holds: every copy checked in so far, with its
            // buffered delete (check-ins take the core write lock).
            // Never both locks at once in this order (resolvers take window,
            // then unresolved); no check-in can add to `unresolved` while the
            // core write lock is held.
            let window = registry::delete_window(&self.unresolved.lock())?;
            *self.sorted_window.lock() = Some(window);
            match core.begin_seal_chunks(true, CoreIndex::COALESCE_MAX_BYTES) {
                Ok(jobs)=>jobs,
                Err(error)=>{*self.sorted_window.lock()=None;return Err(error.into());}
            }
        };
        let (claimed, jobs) = (started.elapsed(), job.len());
        if let Err(error) = self.finish_seal_chunks(job) {
            // Nothing published: the window's deletes stay buffered.
            *self.sorted_window.lock() = None;
            return Err(error);
        }
        let sealed = started.elapsed();
        let core = self.core.write();
        let locked = started.elapsed();
        let published = self.publish_locked_with(core, true, None);
        tracing::debug!(target: "seal_profile", jobs, claim_ms = claimed.as_millis() as u64, sealed_ms = sealed.as_millis() as u64,
            publish_lock_ms = (locked - sealed).as_millis() as u64, total_ms = started.elapsed().as_millis() as u64, "sorted refresh");
        if let (true, Some(ticket)) = (published.is_ok(), ticket) {
            self.listeners.complete(ticket);
        }
        published
    }

    /// The refresh critical section: under the core write lock only the
    /// writer-side swap (memtables being written close into published
    /// ones, pending deletes become visible, the new reader view is built
    /// from `Arc`s and memtable delete copies and installed). Its hold time
    /// is recorded ([`refresh_lock_hold`]). Index-sorted cores seal their
    /// writers before this, off the core lock ([`Shard::publish_sorted`]).
    fn publish_locked(&self, core: parking_lot::RwLockWriteGuard<'_, CoreIndex>, ticket: Option<u64>) -> Result<bool> {
        self.publish_locked_with(core, false, ticket)
    }

    /// [`Shard::publish_locked`]; `sealed_only`: the sorted refresh's
    /// publication of sealed segments and deletes only
    /// ([`rustic_core::Index::publish_sealed`]).
    /// `ticket`: the refresh listeners' ticket this publication completes
    /// (`None`: the sorted refresh completes its own).
    fn publish_locked_with(&self, mut core: parking_lot::RwLockWriteGuard<'_, CoreIndex>, sealed_only: bool, ticket: Option<u64>) -> Result<bool> {
        let held = std::time::Instant::now();
        // Mutable-graph delete repair is not part of the critical section:
        // background workers run it in bounded batches (`repair_vectors`).
        // A copy and the deletion of the copy it replaces become visible
        // together (the bulk of the deletes was resolved before, outside
        // this lock).
        if sealed_only {
            if let Some(window) = self.sorted_window.lock().take() {
                self.resolve_window_buffered_deletes(&core, window);
            }
        } else {
            self.resolve_all_buffered_deletes(&core);
        }
        let changed = if sealed_only { core.publish_sealed() }
            else if core.index_sort().is_some() && MEMTABLES_FIRST.with(|m| m.get()) { core.publish_memtables() }
            else { core.publish_visible() };
        // The memtables just published closed (rustic-core charges them
        // from now on): the indexing buffer counts only those still written.
        self.note_buffered(core.writing_bytes() as u64);
        let seal_due = matches!(changed, Ok(true)) && self.bg.served.load(Ordering::Acquire) && core.frozen_over_limit();
        let repair_due = matches!(changed, Ok(true)) && core.vector_store().has_mutable_fields();
        drop(core);
        note_refresh_hold(held.elapsed());
        if let (Some(ticket), Ok(_)) = (ticket, &changed) {
            self.listeners.complete(ticket);
        }
        if seal_due {
            self.bg.want_seal.store(true, Ordering::Release);
        }
        if repair_due {
            self.bg.want_repair.store(true, Ordering::Release);
            if !self.bg.served.load(Ordering::Acquire) {
                // No background worker serves this copy: repair now, still
                // outside the core lock.
                self.bg.want_repair.store(false, Ordering::Release);
                self.repair_vectors();
            }
        }
        Ok(changed?)
    }

    /// Physical mutable-graph delete repair, one bounded batch per field per
    /// background pass, outside every shard lock (the core is only
    /// try-read to reach its vector store); re-armed while deletes remain
    /// or the core is busy. Searches never depend on it: their reader views
    /// mask deleted documents.
    pub(crate) fn repair_vectors(&self) {
        let Some(store) = self.core.try_read().map(|core| core.vector_store().clone()) else {
            self.bg.want_repair.store(true, Ordering::Release);
            return;
        };
        match store.drain_mutable_deletes_batch(VECTOR_REPAIR_BATCH) {
            Ok(true) => self.bg.want_repair.store(true, Ordering::Release),
            Ok(false) => {}
            Err(e) => {
                tracing::warn!("mutable vector delete repair of [{}] failed (retried): {e}", self.path.display());
                self.bg.want_repair.store(true, Ordering::Release);
            }
        }
    }

    /// ES `refresh=wait_for`: return once a refresh made the writes this
    /// thread completed visible, without forcing one (the next scheduled
    /// refresh, or any other, satisfies it). Forces a refresh (returns
    /// `true`, ES `forced_refresh`) when `max_listeners` are already waiting
    /// (`index.max_refresh_listeners`) or after `max_wait`.
    pub fn wait_for_refresh(&self, max_listeners: usize, max_wait: std::time::Duration) -> Result<bool> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.add_refresh_listener(max_listeners, max_wait, Box::new(move |result| { let _ = tx.send(result); }));
        match rx.recv_timeout(max_wait) {
            Ok(result) => result,
            // No scheduler forced it (a copy no worker serves): refresh here.
            Err(_) => {
                self.refresh()?;
                rx.recv().unwrap_or(Ok(true)).map(|_| true)
            }
        }
    }

    /// ES `RefreshListeners.addOrNotify`: call `callback` once a refresh
    /// made the writes this thread completed visible, without waiting (the
    /// write thread returns; the refresh that completes the listener sends
    /// the response). Past `max_listeners` waiting listeners a refresh is
    /// forced here (`Ok(true)`). A listener still waiting after `max_wait`
    /// gets a forced refresh from the scheduler (`Ok(true)`).
    pub fn add_refresh_listener(&self, max_listeners: usize, max_wait: std::time::Duration, callback: RefreshCallback) {
        if self.listeners.waiting.load(Ordering::Acquire) >= max_listeners {
            let result = self.refresh().map(|_| true);
            callback(result);
            return;
        }
        let target = self.listeners.started.load(Ordering::Acquire) + 1;
        {
            // Under `completed`: a refresh completing meanwhile either sees
            // this listener or has already passed its ticket (and a close
            // either sees it or has already marked the copy closed).
            let completed = self.listeners.completed.lock();
            if *completed < target && !self.listeners.closed.load(Ordering::Acquire) {
                self.listeners.waiting.fetch_add(1, Ordering::AcqRel);
                self.listeners.parked.lock().push(ParkedListener {
                    target, deadline: std::time::Instant::now() + max_wait, forced: false, callback,
                });
                drop(completed);
                // The write marked the copy dirty: a scheduled refresh is due.
                self.sched.note_write();
                return;
            }
        }
        callback(Ok(false));
    }

    /// Listeners waited past their deadline: a refresh must be forced for
    /// them (the scheduler hands the copy to a worker).
    pub(crate) fn refresh_listeners_expired(&self) -> bool {
        self.listeners.expire(std::time::Instant::now())
    }

    /// Reader-pressure gauges for `_nodes/stats` (never waits: a busy core
    /// reports no vector figures this time).
    pub fn reader_pressure(&self) -> ReaderPressure {
        let vectors = self.core.try_read().map(|core| core.vector_store().mutable_pressure()).unwrap_or_default();
        ReaderPressure { oldest_live_view_age: self.views.oldest_live_view_age(), vectors }
    }

    /// A scheduled refresh the scheduler handed off because the core was
    /// busy ([`crate::refresh`]), run on a background worker. It waits for
    /// the core at most [`DEFERRED_REFRESH_WAIT`] (enough to queue behind
    /// short write batches on the fair lock), never for a long holder such
    /// as a flush: a background worker parked there would stall seals and
    /// merges of every copy. When still busy the refresh is re-armed and the
    /// next scheduler tick retries it.
    pub(crate) fn refresh_deferred(&self) {
        let _refreshing = self.refreshing();
        let t0 = std::time::Instant::now();
        match self.try_publish_for(DEFERRED_REFRESH_WAIT) {
            Some(Ok(true)) => {
                self.counters.refresh_total.fetch_add(1, Ordering::Relaxed);
                Counters::add_time(&self.counters.refresh_time_nanos, t0.elapsed());
            }
            Some(Ok(false)) => {}
            Some(Err(e)) => tracing::warn!("deferred refresh of [{}] failed: {e}", self.path.display()),
            None => self.bg.want_refresh.store(true, Ordering::Release),
        }
    }

    /// A refresh not run by a search is under way until the guard drops.
    fn refreshing(&self) -> impl Drop + '_ {
        struct Refreshing<'a>(&'a std::sync::atomic::AtomicUsize);
        impl Drop for Refreshing<'_> { fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); } }
        self.bg.refreshing.fetch_add(1, Ordering::AcqRel);
        Refreshing(&self.bg.refreshing)
    }

    /// [`Shard::try_publish`], waiting at most `wait` for the core.
    fn try_publish_for(&self, wait: std::time::Duration) -> Option<Result<bool>> {
        let _defer = DeferCheckinDrops::enter();
        let generation = self.sched.note_refresh();
        // As [`Shard::publish`]: what a complete publication makes visible.
        let processed = self.processed_checkpoint();
        let ticket = self.listeners.begin();
        let t0 = std::time::Instant::now();
        let _phases = RefreshPhases::begin();
        if !self.checkin_completed_writes(Some(wait)) {
            return None;
        }
        let phases = RefreshPhases::current();
        let checked_in = t0.elapsed();
        let ticket = Some(ticket);
        let core = self.core.try_write_for(wait)?;
        let core_locked = t0.elapsed();
        // A large buffer: sealed first (this is a background worker).
        let core = if self.publish_needs_seal(&core) {
            drop(core);
            if let Err(e) = self.seal_unpublished() {
                return Some(Err(e));
            }
            self.core.try_write_for(wait)?
        } else { core };
        // An index-sorted core publishes its memtables as they are (ES
        // makes its in-memory buffer searchable at refresh) and seals them
        // into sorted segments in the background, off the search's path;
        // while a detached seal is in flight, the sorted refresh as before.
        let memtables_first = Self::needs_sorted_seal(&core) && publish_sorted_memtables() && !core.seal_in_flight();
        let published = if Self::needs_sorted_seal(&core) && !memtables_first {
            drop(core);
            self.try_publish_sorted(ticket)?
        } else {
            MEMTABLES_FIRST.with(|m| m.set(memtables_first));
            let published = self.publish_locked(core, ticket);
            MEMTABLES_FIRST.with(|m| m.set(false));
            if memtables_first && published.is_ok() {
                // Sealed (sorted) by the next background seal, whatever
                // their size.
                self.bg.draining.store(true, Ordering::Release);
                self.bg.want_seal.store(true, Ordering::Release);
            }
            published
        };
        self.note_published(generation, true, &published);
        if published.is_ok() {
            self.note_refreshed_checkpoint(processed);
        }
        let ms = |d: std::time::Duration| d.as_millis() as u64;
        tracing::debug!(target: "seal_profile", memtables_first, checkin_ms = ms(checked_in), checkin_lock_ms = ms(phases.checkin_lock), handoff_ms = ms(phases.handoff),
            write_lock_ms = ms(phases.write_lock), checkin_core_lock_ms = ms(phases.core_lock), core_lock_ms = ms(core_locked - checked_in),
            begin_freeze_ms = ms(phases.begin_freeze), checkin_writers_ms = ms(phases.checkin_writers), after_checkin_ms = ms(phases.after_checkin),
            memtables = phases.checkin_memtables, docs = phases.checkin_docs, docs_loop_ms = ms(phases.checkin_docs_loop), core_checkin_ms = ms(phases.checkin_core), older = phases.checkin_older,
            total_ms = ms(t0.elapsed()), "search-idle refresh");
        Some(published)
    }

    /// [`Shard::publish_sorted`] for a background worker: only when its
    /// fences are free right now (`None`: busy, retried by the scheduler).
    /// The detached seal holds the fences, never the core, while encoding.
    fn try_publish_sorted(&self, ticket: Option<u64>) -> Option<Result<bool>> {
        let _seal = self.try_seal_fences()?;
        Some(self.seal_and_publish_sorted(ticket))
    }

    /// [`Shard::publish`] if the core write lock is free now (`None`: busy,
    /// nothing done; the copy stays due for a refresh).
    pub(crate) fn try_publish(&self) -> Option<Result<bool>> {
        let _defer = DeferCheckinDrops::enter();
        let generation = self.sched.note_refresh();
        // As [`Shard::publish`]: what a complete publication makes visible
        // (a fleet search's checkpoint wait sees scheduled refreshes).
        let processed = self.processed_checkpoint();
        let ticket = self.listeners.begin();
        if !self.checkin_completed_writes(Some(std::time::Duration::ZERO)) {
            // A holder has not handed off yet: a background worker refreshes
            // with a bounded wait (`Shard::refresh_deferred`).
            return None;
        }
        let ticket = Some(ticket);
        let core = self.core.try_write()?;
        if Self::needs_sorted_seal(&core) || self.publish_needs_seal(&core) {
            // Encoding first: a background worker runs it (see
            // `Shard::refresh_deferred`).
            return None;
        }
        let published = self.publish_locked(core, ticket);
        self.note_published(generation, true, &published);
        if published.is_ok() {
            self.note_refreshed_checkpoint(processed);
        }
        Some(published)
    }

    /// Install captured chunks incrementally, releasing completed memtables
    /// before encoding the next chunk. Finish ALL claims even on an error;
    /// failed writers stay in the core and become eligible for a later retry.
    fn finish_seal_chunks(&self, jobs: Vec<rustic_core::index::SealJob>) -> Result<()> {
        let mut first_error = None;
        let mut jobs = PendingSealChunks::new(self, jobs);
        let mut installing = std::time::Duration::ZERO;
        let started = std::time::Instant::now();
        jobs.run(|sealed| {
            let t = std::time::Instant::now();
            if let Err(error) = self.install_seal_retrying(sealed, |core, sealed| core.try_finish_seal_captured(sealed).map(|r| r.map(|_| ())), false) {
                first_error.get_or_insert(error);
            }
            installing += t.elapsed();
        });
        tracing::debug!(target: "seal_profile", install_ms = installing.as_millis() as u64, total_ms = started.elapsed().as_millis() as u64, "seal chunks");
        first_error.map_or(Ok(()), |error| Err(error.into()))
    }

    /// Seal every memtable, published or not, into segments (visibility
    /// unchanged; the segment half of a flush), detached: the core lock is
    /// held only to begin and to install. Callers needing "every write so
    /// far" fence writes themselves (the cluster's recovery source holds its
    /// replication gate).
    pub fn seal_all(&self) -> Result<()> {
        let _seal = self.seal_fences();
        // "Every memtable": pooled ones too.
        self.checkin_completed_writes(None);
        self.files_changed();
        let jobs = { self.core.write().begin_seal_chunks(true, CoreIndex::COALESCE_MAX_BYTES)? };
        self.finish_seal_chunks(jobs)?;
        self.bg.want_merge.store(true, Ordering::Release);
        Ok(())
    }

    /// Best-effort [`Shard::seal_all`] that never waits to start: skipped
    /// (`Ok(false)`) when the seal lock, the write state or the core is held
    /// right now. For an optional pre-seal ahead of a fenced seal (recovery
    /// source), which must not delay the fence behind a long holder.
    pub fn try_seal_all(&self) -> Result<bool> {
        let Some(_seal) = self.try_seal_fences() else { return Ok(false) };
        if !self.checkin_completed_writes(Some(std::time::Duration::ZERO)) {
            return Ok(false);
        }
        let Some(mut core) = self.core.try_write() else { return Ok(false) };
        self.files_changed();
        let jobs = core.begin_seal_chunks(true, CoreIndex::COALESCE_MAX_BYTES)?;
        drop(core);
        self.finish_seal_chunks(jobs)?;
        self.bg.want_merge.store(true, Ordering::Release);
        Ok(true)
    }

    /// The files the core's last commit point references, by name in the
    /// core directory, `meta.json` last: text segments and their deletes,
    /// vector payload segments, mutable graph checkpoints and their code
    /// log chunks, DiskBBQ files. Exactly what a copy of the commit needs
    /// (file recovery ships the same set, see `capture_files`); anything
    /// else in the directory (a merge or seal still encoding, files a
    /// commit retired, a crash's leftovers) belongs to no commit. Hold
    /// [`Shard::files_pinned`] while using the names: no commit replaces
    /// them meanwhile.
    pub fn committed_core_files(&self) -> Result<Vec<String>> {
        let pinned = self.core.read().pin_commit()?;
        Ok(pinned.iter().map(|f| f.name.clone()).collect())
    }

    /// Pin the shard's files: no seal or merge installs and no commit (which
    /// deletes retired files) until the guard drops. Reads, refreshes and
    /// writes continue (they create no file; an index-sorted refresh, which
    /// seals, waits). For copying the shard directory.
    pub fn files_pinned(&self) -> FilesPinned {
        // No seal or merge installs (nothing retires a segment's files) and
        // no commit (it is the flush, which takes `merge_lock`) meanwhile.
        // Seals and merges still encoding are not waited for: the files they
        // are writing belong to no commit, so a copy of the directory holds
        // them only as unreferenced files (a throttled merge can run for
        // hours).
        let seal = self.bg.seal_lock.lock();
        let merge = self.bg.merge_lock.lock();
        // A flush's commit writes and deletes files with `merge_lock`
        // released (see `commit_lock`): this pin waits it out too.
        let commit = self.bg.commit_lock.lock();
        FilesPinned { _commit: commit, _merge: merge, _seal: seal }
    }

    /// Seal the published memtables into segments (visibility unchanged),
    /// detached from the core lock where possible. Background use only
    /// ([`Shard::seal_published_if_idle`]); request paths (`_segments`,
    /// `_cat/segments`, stats, disk usage) read [`Shard::reader_view`]
    /// instead, where published memtables are leaves of their own.
    pub fn seal_published(&self) {
        // Completed writes join the index as memtables being written
        // (visibility unchanged).
        self.checkin_completed_writes(None);
        let needs = self.core.read().has_frozen();
        if needs {
            self.files_changed();
            let _seal = self.seal_fences();
            // Published leaves remain readable while their segment files
            // are encoded. Do not close or publish the active writers here.
            let jobs = { self.core.write().begin_seal_chunks(false, CoreIndex::COALESCE_MAX_BYTES) };
            let result = match jobs {
                Err(error)=>Err(error.into()),
                Ok(jobs) if jobs.is_empty()=>self.core.write().seal_frozen().map_err(Into::into),
                Ok(jobs)=>self.finish_seal_chunks(jobs),
            };
            if let Err(e) = result {
                tracing::error!("seal failed: {e}");
            }
            self.bg.want_merge.store(true, Ordering::Release);
        }
    }

    /// Seal the published memtables of a copy that saw no write for
    /// `idle`, so its vectors reach the vector store (and get an HNSW graph
    /// from the optimizer) instead of being brute-forced in memory.
    pub fn seal_published_if_idle(&self, idle: std::time::Duration) {
        if self.sched.write_idle() >= idle {
            self.seal_published();
        }
    }

    /// The reader searches use is stale as far as the core can tell without
    /// waiting: a publish would show something (memtables being written or
    /// parked, segments sealed in the background since the last refresh,
    /// pending deletes, a view not rebuilt), whether or not a write was
    /// noted since ([`crate::refresh`]'s scheduled refresh and a search-idle
    /// copy's first search use it beside the dirty flag). `false` when the
    /// core is busy: the next scheduler tick looks again.
    pub(crate) fn reader_stale(&self) -> bool {
        self.core.try_read().is_some_and(|core| core.has_unpublished())
    }

    /// Some write is not yet visible to searches.
    pub fn has_unpublished(&self) -> bool {
        self.pool.lock().iter().any(|pooled| !pooled.docs.is_empty()) || self.core.read().has_unpublished()
    }

    /// Call before a search reads this shard: marks it searched and, when a
    /// scheduled refresh was skipped because the shard was search-idle,
    /// refreshes first (ES `awaitShardSearchActive`).
    ///
    /// The refresh only try-locks the core: when a flush, merge install or
    /// write batch holds it, the search reads the current view, and the
    /// copy (search-active again, still dirty) is refreshed by the next
    /// scheduler tick.
    pub fn before_search(&self) {
        // Background encode parallelism follows search load.
        rustic_core::note_search();
        // A search-idle copy holding writes a search cannot see yet (or a
        // stale reader: data sealed after its last refresh): this search
        // refreshes it first, whether or not a scheduler tick already
        // skipped a refresh (a write just replicated to an idle replica).
        if self.sched.idle.load(Ordering::SeqCst) && (self.sched.is_dirty() || self.reader_stale()) {
            self.sched.skipped.store(true, Ordering::SeqCst);
        }
        self.sched.note_search();
        // A refresh API call or a scheduled refresh is under way (the first
        // refresh after a refresh-disabled load, say): this search reads
        // the last published reader, as every ES search does, instead of
        // waiting for it or starting a second one beside it. A skipped
        // refresh stays due for the next search or tick.
        if self.bg.refreshing.load(Ordering::SeqCst) > 0 && !self.bg.idle_refreshing.load(Ordering::SeqCst) {
            return;
        }
        // `idle_refreshing` is raised before `skipped` is taken (both
        // SeqCst), so a search arriving at any point of the refresh sees
        // one of them and waits.
        if self.sched.skipped.load(Ordering::SeqCst) || self.bg.idle_refreshing.load(Ordering::SeqCst) {
            // ES: every search on a search-idle copy waits for the refresh
            // it triggers. The first one refreshes; the others arriving
            // meanwhile wait here for it, then see its view.
            // A copy the scheduler serves: a search waits for another
            // search's refresh a bounded time only
            // ([`IDLE_SEARCH_OTHER_REFRESH_WAIT`]), then reads the last
            // published view, as a search on a search-active copy does.
            // Waiting without bound queued every search behind whatever that
            // refresh waited on (the core lock under a flush capture or merge
            // install, completed writes' hand-off): after a 5M-document
            // preload (wiki mixed load, c28j) every search of the 60 s idle
            // phase took about 41 s. A copy no scheduler serves (tests,
            // tools) has nobody else to refresh it and still waits.
            let served = self.bg.served.load(Ordering::Acquire);
            let waiting = if served { self.bg.idle_refresh.try_lock_for(IDLE_SEARCH_OTHER_REFRESH_WAIT) } else { Some(self.bg.idle_refresh.lock()) };
            let Some(_waiting) = waiting else { return };
            if !self.sched.skipped.load(Ordering::SeqCst) {
                return;
            }
            self.bg.idle_refreshing.store(true, Ordering::SeqCst);
            struct Lower<'a>(&'a std::sync::atomic::AtomicBool);
            impl Drop for Lower<'_> { fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); } }
            let _lower = Lower(&self.bg.idle_refreshing);
            if !self.sched.take_skipped() {
                return;
            }
            let t0 = std::time::Instant::now();
            // A large backlog is sealed first (by this refresh, waiting at
            // most `IN_FLIGHT_SEAL_WAIT` for background seals in flight), then
            // published (ES: a search-idle shard's refresh writes its own
            // segments, and the search waits for it). A served copy's refresh
            // waits for the core and for completed writes' hand-off at most
            // [`IDLE_SEARCH_REFRESH_WAIT`]: a flush capture, a merge install
            // or a write batch holding them hands the refresh to the
            // background workers (`refresh_deferred`), and the search reads
            // the last published view; the copy is search-active and dirty,
            // so the scheduler keeps refreshing it.
            let unpublished = self.core.try_read().map(|core| (core.num_unpublished_bytes(), self.publish_needs_seal(&core)));
            let published = if served { self.try_publish_for(IDLE_SEARCH_REFRESH_WAIT) } else { Some(self.publish()) };
            if published.is_none() {
                self.bg.want_refresh.store(true, Ordering::Release);
            }
            let took = t0.elapsed();
            tracing::debug!(target: "seal_profile", took_ms = took.as_millis() as u64, "search waited for its idle refresh");
            if took >= SLOW_INSTALL {
                let (bytes, sealed) = unpublished.map_or((None, None), |(b, s)| (Some(b), Some(s)));
                tracing::warn!(path = %self.path.display(), took_ms = took.as_millis() as u64, served, unpublished_bytes = ?bytes, sealed_backlog = ?sealed,
                    outcome = match &published { Some(Ok(true)) => "published", Some(Ok(false)) => "unchanged", Some(Err(_)) => "failed", None => "handed off" },
                    "slow search-idle refresh (searches on this copy waited for it)");
            }
            match published {
                Some(Ok(true)) => {
                    self.counters.refresh_total.fetch_add(1, Ordering::Relaxed);
                    Counters::add_time(&self.counters.refresh_time_nanos, t0.elapsed());
                }
                Some(Ok(false)) => {}
                Some(Err(e)) => tracing::warn!("search-idle refresh of [{}] failed: {e}", self.path.display()),
                None => tracing::debug!("search-idle refresh of [{}] deferred: core busy", self.path.display()),
            }
        }
    }

    /// Age of the last successful flush start; request fsync is independent.
    pub fn elapsed_since_flush(&self) -> std::time::Duration { self.last_flush_started.lock().elapsed() }

    /// The copy closes: `refresh=wait_for` listeners fire now (ES
    /// `RefreshListeners.close`).
    pub fn close_refresh_listeners(&self) {
        self.listeners.close();
    }

    /// Seal + commit + persist the commit point + trim the translog.
    pub fn flush(&self) -> Result<()> {
        self.flush_then(|| {})
    }

    /// [`Shard::flush`], calling `captured` as soon as the commit point is
    /// captured (writers closed, translog rolled, seq_no fixed), before the
    /// expensive part: a caller's write fence (the cluster's replication
    /// gate) needs to be held only until then.
    ///
    /// The flush holds no lock across encoding or commit I/O that writes or
    /// searches need:
    /// 1. capture (short): wait out append-only batches (`batches`), then
    ///    under the write state and a short core lock close the memtables
    ///    being written into the seal job (new writes go to fresh ones), roll
    ///    the translog (ops <= the commit seq_no are in older generations)
    ///    and fix the commit seq_no; then release both;
    /// 2. seal the closed memtables off the core lock (detached seal);
    /// 3. capture the commit inputs under a short core lock and write
    ///    deletion files, the vector checkpoint and `meta.json` off-lock;
    ///    fsync the core directory (the commit is durable);
    /// 4. persist the shard state and only then trim the translog below the
    ///    rolled generation, keeping what retention needs
    ///    (`trim_translog_retained`).
    ///
    /// Writes after the capture land in fresh memtables and the new
    /// translog generation; their deletes may reach the committed `.del`
    /// files, which replay of the retained ops (above the commit seq_no)
    /// makes idempotent. `seal_lock` and `merge_lock` pin the segment set
    /// for the duration (background seals / merge installs wait).
    ///
    /// Cores with mutable vector fields take the graphs' capture point in
    /// step 1 too (one short graph-writer hold per field, no ingestion in
    /// flight under the fence): the checkpoint is the graph exactly as of
    /// the commit seq_no. The graphs are copied in step 3 (chunked, graph
    /// writers interleaved) and streamed with the commit, while writes made
    /// after the capture land above it and replay from the translog.
    pub fn flush_then(&self, captured: impl FnOnce()) -> Result<()> {
        self.flush_inner(captured)
    }

    /// Install (or clear) the replication retention of a cluster primary:
    /// `floor()` is the lowest seq_no some copy, or the primary's own
    /// replication tailer, may still need from the translog (`None`: no
    /// constraint, e.g. its tracker is gone). Every flush then keeps the
    /// generations holding ops at or above it (ES peer-recovery retention
    /// leases), so a primary flushes without fencing its writes: in-flight
    /// writes stay above the commit point (their claims) and above the floor
    /// (no copy has them yet). Called at trim time under the write state;
    /// `floor` must take only leaf locks.
    pub fn set_replication_retention(&self, floor: Option<Arc<dyn Fn() -> Option<i64> + Send + Sync>>) {
        *self.retention.lock() = floor;
    }

    /// The oldest translog generation (below `next_gen`) that may hold an op
    /// `>= seq_no` (a generation of unknown content counts as one), or
    /// `next_gen`: trimming below it drops no such op.
    /// The lowest seq_no of any document in a memtable (published, parked
    /// or being written; deleted ones included): `None` when no memtable
    /// holds one.
    fn min_memtable_seq_no(core: &CoreIndex, include: impl Fn(u32) -> bool) -> Option<i64> {
        let mut min: Option<i64> = None;
        for leaf in core.all_leaves() {
            let rustic_core::query::LeafReader::Memtable(writer, ..) = &leaf else { continue };
            if !include(writer.id) { continue; }
            for doc in 0..leaf.num_docs() {
                if let Some(seq_no) = leaf.i64_value(SEQ_NO_FIELD, doc) {
                    min = Some(min.map_or(seq_no, |m| m.min(seq_no)));
                }
            }
        }
        min
    }

    fn generation_holding(&self, ws: &WriteState, seq_no: i64, next_gen: u64) -> Result<u64> {
        for (generation, _) in crate::translog::list_generations(&self.path.join(TRANSLOG_DIR))? {
            if generation >= next_gen { break; }
            match ws.hist.gen_max_seq.get(&generation) {
                Some(&max) if max < seq_no => continue,
                _ => return Ok(generation),
            }
        }
        Ok(next_gen)
    }

    fn flush_inner(&self, captured: impl FnOnce()) -> Result<()> {
        let _shard_owner=self.admitted_owner.enter();
        let t0 = std::time::Instant::now();
        let _flush = self.bg.flush_lock.lock();
        if self.bg.files_closed.load(Ordering::Acquire) {
            return Err(EsError::new(500, "already_closed_exception", "the shard copy's files are closed"));
        }
        // Background seals neither claim nor install meanwhile: the segment
        // set stays pinned (released while this flush's own seal encodes,
        // see step 2).
        let mut seal_pin = Some(self.bg.seal_lock.lock());
        let seal_lock_acquired = t0.elapsed();
        // Merge installs are excluded from the capture until the commit's
        // segment set is fixed (step 3), not across the commit's writes.
        let mut merge_pin = Some(self.bg.merge_lock.lock());
        self.files_changed();
        // 1. Capture. In-flight requests are never waited for: their pooled
        // memtables freeze at a document boundary (ES DWPT flush-pending)
        // and join the commit with every completed write; what is still in
        // flight stays out of it, above the commit's local checkpoint (the
        // lowest seq_no not in the capture, minus one), and is replayed from
        // the translog, which is trimmed only above it.
        // Idle pooled memtables first (no request may take one meanwhile),
        // then the freeze; all of them join under the fence below.
        // Held until they are checked in below (see `checkin_lock`).
        let mut checkin_pin = Some(self.checkin_lock.lock());
        let (target, mut taken) = self.begin_freeze();
        self.wait_freeze(target, FLUSH_FREEZE_WAIT);
        taken.extend(std::mem::take(&mut *self.frozen.lock()));
        // Graph captures never quiesce graph ingestion: what is in flight
        // at the capture point is left out of the captured graph and
        // reconciled at open (see `MutableIngestField::begin_capture_checkpoint`).
        let batches_acquired = t0.elapsed();
        let mut fence = Some(self.write.lock());
        let write_acquired = t0.elapsed();
        let (mut seal, commit_seq, next_gen, keep_gen, seqno, core_generation, mut vector_capture, sealed_before, mut deletions_at) = {
            let ws = fence.as_mut().expect("write state held");
            self.publish_write_stats(ws);
            // Graph saves are scheduled (see `GraphSaves`): most commits
            // keep the previous graph checkpoint.
            let save_graphs = self.graph_saves.lock().due(ws.seq_no);
            let (seal, capture, unsealed, sealed_before, deletions_at) = {
                let mut core = self.core.write();
                let mut pooled = std::mem::take(&mut taken);
                pooled.extend(std::mem::take(&mut *self.frozen.lock()));
                pooled.extend(std::mem::take(&mut *self.pool.lock()));
                for writer in pooled {
                    self.checkin_pool_writer(&mut **ws, &mut core, writer);
                }
                drop(checkin_pin.take());
                // The commit is a point in time: the older copies replaced
                // by what it holds are deleted in it, and it records the
                // deletions as of now (later deletes are above its point and
                // replay from the translog; a commit read without the
                // translog, e.g. a snapshot, is self-consistent).
                self.resolve_all_buffered_deletes(&core);
                let deletions_at = core.deletions_snapshot()?;
                // The graphs' capture point (fallible, no side effects). A
                // graph save needs no graph ingestion in flight; when some
                // is, this commit keeps the previous graph checkpoints.
                let capture = if core.vector_store().has_mutable_fields() {
                    match core.begin_vector_capture(save_graphs) {
                        Ok(capture) => Some((capture, save_graphs)),
                        Err(error) if save_graphs && ingestion_in_flight(&error) => {
                            self.graph_saves.lock().skipped += 1;
                            Some((core.begin_vector_capture(false)?, false))
                        }
                        Err(error) => return Err(error.into()),
                    }
                } else { None };
                let seal = core.begin_flush_seal_chunks(true, CoreIndex::COALESCE_MAX_BYTES)?;
                // Background seals encoding other memtables (not this
                // flush's to seal, and a flush never waits for them): the
                // commit holds only sealed segments, so its point stays
                // below every document still in one of their memtables.
                let ours: Vec<u32> = seal.iter().flat_map(|job| job.memtable_ids()).collect();
                let unsealed = Self::min_memtable_seq_no(&core, |id| core.is_sealing(id) && !ours.contains(&id));
                // This flush's seal encodes with the segment set unpinned:
                // background seals keep claiming and installing (writes are
                // not held behind a long encode), and the commit leaves out
                // the text segments they install after this point (their
                // documents are above the commit point and replay from the
                // translog). Vector points are owned by text segments: a
                // reopen attaches only the committed ones, so the points of
                // a left-out segment stay dead there (the vector segments
                // holding them may be listed; their rows are re-added by the
                // replay), and mutable graphs were captured at this point.
                // Index-sorted cores stay pinned: their refresh seals and
                // publishes only sealed segments behind `seal_lock`, and a
                // flush's commit must not resolve deletes a sorted refresh
                // in flight would publish without their new copies.
                let sealed_before = (!seal.is_empty() && core.index_sort().is_none()).then(|| core.sealed_segment_ids());
                (seal, capture, unsealed, sealed_before, deletions_at)
            };
            // Core lock is released before this guard is constructed. Every
            // subsequent early return (roll/listing/capture callback failure)
            // cancels the remaining claims without discarding their writers.
            let seal = PendingSealChunks::new(self, seal);
            // A failed roll (its fsync of the generation it leaves) is a
            // translog tragedy: the engine fails, and no writer waiting for
            // that generation's fsync retries it.
            let next_gen = match ws.translog.roll() {
                Ok(next_gen) => next_gen,
                Err(error) => {
                    self.mark_tragic(&format!("translog roll failed: {error}"));
                    return Err(error);
                }
            };
            // `gen_max_seq` records what the rolled translog generation
            // contains (up to max_seq_no); the commit point below can be
            // lower (ops of in-flight batches are in that generation but not
            // in this commit): the two are no longer equal.
            let mut seqno = self.commit_seqno_state(ws, next_gen);
            let (commit_seq, oldest_claimed) = ws.capture_floor();
            let commit_seq = match unsealed { Some(min) => commit_seq.min(min - 1), None => commit_seq };
            // The persisted seq_no state describes this capture: an op above
            // the commit point may have completed with its document in a
            // memtable the capture does not hold (taken by an in-flight
            // request), so replay must not skip it as processed, nor treat
            // a delete tombstone above it as applied.
            if let Some(state) = seqno.as_mut() {
                state.local_checkpoint = state.local_checkpoint.min(commit_seq);
                state.processed_above.retain(|seq_no| *seq_no <= commit_seq);
                state.tombstones.retain(|(_, seq_no, _)| *seq_no <= commit_seq);
            }
            let mut keep_gen = oldest_claimed.map_or(next_gen, |generation| generation.min(next_gen));
            if let Some(min) = unsealed {
                // The unsealed documents replay from the translog.
                keep_gen = keep_gen.min(self.generation_holding(ws, min, next_gen)?);
            }
            (seal, commit_seq, next_gen, keep_gen, seqno, ws.core_generation, capture, sealed_before, deletions_at)
        };
        // What this flush seals itself (reported by a slow flush).
        let SealCaptureStats { jobs: capture_jobs, memtables: capture_memtables, docs: capture_docs, bytes: capture_bytes, coalesced: capture_coalesced } = seal.stats;
        // Writes continue (into fresh memtables) from here on.
        fence = None;
        captured();
        run_capture_releases(&self.path);
        let core_acquired = t0.elapsed();
        // Unpinned from here (see `sealed_before`); never taken again.
        let mut committed_ids = sealed_before;
        if committed_ids.is_some() {
            seal_pin = None;
        }
        let pause = self.flush_pause.lock().take();
        if let Some(hook) = pause { hook(); }
        // 2. Seal what the capture closed, off the core lock (and, see
        // `sealed_before`, with the segment set unpinned meanwhile).
        // `seal_lock` is not taken again (lock order: it comes before
        // `merge_lock`, held here): `merge_lock` alone keeps merges from
        // installing, and the commit below is prepared under the same core
        // lock that decides which segments are this commit's.
        // The chunks encode concurrently (as one job over the capture did)
        // and install one by one as they finish.
        let mut seal_error = None;
        seal.run(|sealed| {
            let installed = self.install_seal_retrying(sealed, |core, sealed| {
                let before_ours = committed_ids.as_ref().map(|_| core.sealed_segment_ids());
                // Its segments commit their memtables' deletions as of the capture.
                let result = core.try_finish_seal_captured(sealed)?;
                if let (Some(ids), Some(before_ours)) = (committed_ids.as_mut(), before_ours) {
                    // This flush's own segments join its commit.
                    ids.extend(core.sealed_segment_ids().difference(&before_ours).copied());
                }
                Ok(result)
            }, false);
            match installed {
                Ok(captured) => deletions_at.extend(captured),
                Err(error) => { seal_error.get_or_insert(error); }
            }
        });
        if let Some(error) = seal_error { return Err(error.into()); }
        // Still held (to the end, as before) when the segment set stayed pinned.
        let _seal_pin = seal_pin;
        let sealed = t0.elapsed();
        self.bg.want_merge.store(true, Ordering::Release);
        // 3. Commit. Mutable graphs: copy them as of the capture point
        // first, off every lock. Detached for every core, index-sorted ones
        // included: their refresh publishes only sealed segments and seals
        // behind `seal_lock` (held here), and the job holds its segments,
        // whose files outlive any merge.
        let graph_saved = vector_capture.as_ref().is_some_and(|(_, saved)| *saved);
        // The older copies of what the commit holds were deleted at the
        // capture (`deletions_at`); deletes buffered since are left for the
        // next refresh, which publishes them with their new copies.
        {
            let core = self.core.write();
            self.note_buffered(core.writing_bytes() as u64);
            // The graphs are copied as of the capture point by the job
            // itself (O(graph): never under `merge_lock`, see below).
            let mut commit = match vector_capture.take() {
                Some((capture, _)) => core.prepare_commit_deferred(capture)?,
                None => core.prepare_commit()?,
            };
            commit.with_deletions(std::mem::take(&mut deletions_at));
            if let Some(ids) = &committed_ids {
                // Segments background seals installed after the capture.
                let later: std::collections::HashSet<u32> = core.sealed_segment_ids().difference(ids).copied().collect();
                commit.exclude_segments(&later);
            }
            drop(core);
            // The commit's segment set is fixed and its job holds those
            // segments (their files outlive any merge that replaces them):
            // merges may install from here on. Graph copies and checkpoint
            // writes take minutes on a large graph; with `merge_lock` held
            // across them every merge install waited them out.
            let _commit_pin = self.bg.commit_lock.lock();
            drop(merge_pin.take());
            commit.run()?;
        }
        // The new segment files, deletion files and `meta.json` entries are
        // durable only once the directory is.
        if let Ok(dir) = std::fs::File::open(core_dir(&self.path, core_generation)) {
            dir.sync_all()?;
        }
        let committed = t0.elapsed();
        // 4. State, then trim.
        let mut ws = match fence.take() {
            Some(ws) => ws,
            None => self.write.lock(),
        };
        let graph_checkpoint_seq_no = {
            let mut saves = self.graph_saves.lock();
            if graph_saved {
                saves.seq_no = Some(commit_seq);
                saves.saved_at = std::time::Instant::now();
                saves.skipped = 0;
                saves.forced = false;
            }
            saves.seq_no
        };
        let st = ShardState { committed_seq_no: commit_seq, translog_generation: next_gen, core_generation, seqno, graph_checkpoint_seq_no };
        write_state(&self.path, &st)?;
        // Keeps generations still needed by a retention lock / lease, and
        // those holding ops above the commit's local checkpoint (and, for a
        // cluster primary, ops some copy still needs).
        let retention = self.retention.lock().clone();
        let keep_gen = match retention.and_then(|floor| floor()) {
            Some(seq_no) => keep_gen.min(self.generation_holding(&ws, seq_no, next_gen)?),
            None => keep_gen,
        };
        self.trim_translog_retained(&mut ws, keep_gen)?;
        let persisted = t0.elapsed();
        ws.committed_seq_no = commit_seq;
        *self.last_flush_started.lock() = t0;
        // Keep delete tombstones (ES `index.gc_deletes`) so a re-index continues the version.
        ws.prune_versions(&self.unresolved.lock());
        drop(ws);
        self.counters.flush_total.fetch_add(1, Ordering::Relaxed);
        Counters::add_time(&self.counters.flush_time_nanos, t0.elapsed());
        self.files_changed();
        if t0.elapsed() >= std::time::Duration::from_secs(5) {
            tracing::warn!(path=%self.path.display(),
                seal_lock_ms=seal_lock_acquired.as_millis() as u64,
                batches_wait_ms=(batches_acquired-seal_lock_acquired).as_millis() as u64,
                write_wait_ms=(write_acquired-batches_acquired).as_millis() as u64,
                capture_ms=(core_acquired-write_acquired).as_millis() as u64,
                seal_ms=(sealed-core_acquired).as_millis() as u64,
                commit_ms=(committed-sealed).as_millis() as u64,
                state_translog_ms=(persisted-committed).as_millis() as u64,
                finish_ms=(t0.elapsed()-persisted).as_millis() as u64,
                seal_jobs=capture_jobs, seal_memtables=capture_memtables, seal_docs=capture_docs,
                seal_memtable_bytes=capture_bytes, seal_coalesced_jobs=capture_coalesced,
                "slow engine flush phases");
        }
        Ok(())
    }

    pub fn translog_size(&self) -> u64 {
        self.write.lock().translog.size_bytes()
    }

    /// Translog and seq_no figures for stats APIs, without waiting behind a
    /// long holder of the write state. A force merge or flush holds it for
    /// as long as the merge or commit takes (minutes on large shards), and
    /// no write can change these figures meanwhile: the holder publishes them
    /// when it takes the lock, so the published copy stays exact until its
    /// own commit. Otherwise the lock is taken as usual, waiting briefly.
    /// Decisions (flush thresholds, snapshots) use the exact getters.
    pub fn write_stats(&self) -> WriteStats {
        match self.write.try_lock_for(std::time::Duration::from_millis(50)) {
            Some(ws) => self.publish_write_stats(&ws),
            None => *self.write_stats.lock(),
        }
    }

    fn publish_write_stats(&self, ws: &WriteState) -> WriteStats {
        let st = WriteStats {
            translog_bytes: ws.translog.size_bytes(),
            uncommitted_ops: (ws.seq_no - ws.committed_seq_no).max(0),
            max_seq_no: ws.seq_no,
            local_checkpoint: ws.hist.checkpoint,
        };
        *self.write_stats.lock() = st;
        st
    }

    pub fn uncommitted_ops(&self) -> i64 {
        let ws = self.write.lock();
        ws.seq_no - ws.committed_seq_no
    }

    /// Force merge (ES `_forcemerge`): merge the segments present when the
    /// request starts down to `max_segments` (or expunge their deletes), then
    /// flush. As with Lucene's `IndexWriter.forceMerge`, indexing, searches and
    /// statistics continue meanwhile: writes land in new memtables / segments
    /// (so the shard may hold more than `max_segments` segments when it
    /// returns), and deletes of merged docs are carried into the result. Only
    /// the inputs' selection and the install take the core write lock. Lock
    /// order is kept by never holding two of seal / merge / write at once.
    ///
    /// Sorted inputs and single-segment delete expunges use the same detached
    /// merge protocol; no encoding fallback may hold the core write lock.
    pub fn force_merge(&self, max_segments: usize, only_expunge_deletes: bool) -> Result<()> {
        self.force_merge_with(max_segments, only_expunge_deletes, true)
    }

    /// [`Shard::force_merge`]; `flush: false` (ES `flush=false`) leaves the
    /// result uncommitted, for callers that commit under their own
    /// protocol (a cluster primary trims its translog only once every copy
    /// acknowledged).
    ///
    /// `max_segments == 0` (ES `max_num_segments` unset, -1 or any value
    /// <= 0) is not a full merge: as ES's engine calls
    /// `IndexWriter#maybeMerge`, it only requests the merges the merge
    /// policy wants (run by the background merge scheduler, not awaited),
    /// then flushes and refreshes when `flush`.
    pub fn force_merge_with(&self, max_segments: usize, only_expunge_deletes: bool, flush: bool) -> Result<()> {
        if max_segments == 0 && !only_expunge_deletes {
            self.checkin_completed_writes(None);
            // The tail becomes segments first, so the policy sees every doc
            // and the commit after the merges seals nothing new (whose small
            // segments would make more merges due after it).
            if flush {
                self.flush()?;
            } else {
                self.seal_all()?;
            }
            self.bg.want_merge.store(true, Ordering::Release);
            // Run (and wait out) the merges the policy wants, then commit and
            // publish them: their inputs' files go at that commit, and stats
            // and searches see the merged segments. Merges started here run
            // alongside background workers'; the wait holds no lock (each
            // merge takes its own locks only to claim and to install).
            self.run_policy_merges();
            if flush {
                // Publish first (its seals' merges run too), then commit.
                self.refresh()?;
                self.run_policy_merges();
                self.flush()?;
            }
            // Writes meanwhile may have made new merges due.
            self.bg.want_merge.store(true, Ordering::Release);
            return Ok(());
        }
        // Every completed write takes part (and counts in the gauges).
        self.checkin_completed_writes(None);
        self.files_changed();
        // Explicit merges are visible to monitoring through commit; the gauges
        // are released on every return/error path. The completed totals below
        // still count only successful, non-no-op merges.
        let current_docs = self.core.read().all_leaves().iter().map(|leaf| leaf.num_live_docs_latest() as u64).sum();
        let _current = CurrentGuard::new(vec![
            (&self.counters.merge_current, 1),
            (&self.counters.merge_current_docs, current_docs),
        ]);
        // 1. Seal the unsealed tail so every doc indexed before the request is
        //    in a segment (encoded off the core lock, NRT seal protocol).
        {
            // No `batches` fence: append-only batches still in flight are
            // not acknowledged before this request, and must not stall
            // behind its encoding.
            let _seal = self.seal_fences();
            let generation = self.sched.note_refresh();
            let ticket = self.listeners.begin();
            let ticket = self.checkin_completed_writes(None).then_some(ticket);
            let jobs = { self.core.write().begin_seal_chunks(true, CoreIndex::COALESCE_MAX_BYTES)? };
            self.finish_seal_chunks(jobs)?;
            // Make it searchable with the short refresh publication, then
            // seal what that published (documents written meanwhile), again
            // detached: nothing here encodes under the core lock.
            {
                let core = self.core.write();
                let complete = ticket.is_some();
                let published = if Self::needs_sorted_seal(&core) {
                    drop(core);
                    self.seal_and_publish_sorted(ticket)
                } else {
                    self.publish_locked(core, ticket)
                };
                self.note_published(generation, complete, &published);
                published?;
            }
            let jobs = { self.core.write().begin_seal_chunks(false, CoreIndex::COALESCE_MAX_BYTES)? };
            self.finish_seal_chunks(jobs)?;
        }
        // 2. Merge. Vectors first, without the merge fences; then
        //    `merge_running` waits out a running background merge; `merge_lock` is
        //    held only to claim the inputs; background merges of segments
        //    written meanwhile then run alongside the encoding.
        let t0 = std::time::Instant::now();
        let _force = self.bg.force_lock.lock();
        // The vector side first, outside the merge fences: its IVF / graph
        // rebuilds and compaction can take long, and a flush (translog trim)
        // needs `merge_lock`. The vector store swaps its segments under its
        // own synchronization and defers replaced files to the next commit
        // (the background vector optimizer runs the same work without any
        // shard fence), and `force_lock` keeps force merges from overlapping.
        // Pin store/directory ownership, then release the core read guard
        // before graph work. A queued writer behind a long read guard can
        // also stall subsequent readers on the fair core RwLock.
        let optimizer = { self.core.read().vector_optimizer() };
        let vectors = optimizer.force_merge(max_segments, only_expunge_deletes)?;
        let vectors_changed = vectors != rustic_core::vector::OptimizeReport::default();
        let (job, in_place, merged_docs, merged_bytes, noop) = {
            // Waits out a running background merge (its inputs are claimed).
            let _running = self.bg.merge_running.acquire_all();
            let _merge = self.bg.merge_lock.lock();
            let (needed, merged_docs) = {
                let core = self.core.read();
                let segs = core.num_segments();
                let deleted = core.num_docs() - core.num_live_docs();
                let needed = if only_expunge_deletes { deleted > 0 } else { segs > max_segments.max(1) || (deleted > 0 && segs > 0) };
                (needed, core.num_live_docs() as u64)
            };
            if !needed && !vectors_changed {
                // Nothing to merge: still committed below, as ES flushes
                // after a force merge whatever it merged (outside the merge
                // fences: a flush takes `merge_lock`).
                (None, false, merged_docs, 0, true)
            } else {
                let merged_bytes = dir_size(&self.path);
                let job = if needed { self.core.write().begin_force_merge()? } else { None };
                let in_place = needed && job.is_none();
                (job, in_place, merged_docs, merged_bytes, false)
            }
        };
        match job {
            Some(job) => {
                // Encoded on the force-merge pool, so seals and background
                // merges keep their own workers.
                let merged = match job.run() {
                    Ok(m) => m,
                    Err(e) => {
                        self.core.write().release_merge(&job);
                        return Err(e.into());
                    }
                };
                // The inputs are claimed, so no other merge removed them;
                // writes meanwhile only add segments and deletes, which the
                // install carries over.
                {
                    // A detached commit pins the current segment set with this
                    // permit. Encoding may overlap it, installation may not.
                    let _merge = self.bg.merge_lock.lock();
                    if !self.core.write().finish_merge(job, merged)? {
                        return Err(EsError::internal("force-merge inputs changed while claimed"));
                    }
                }
                // Segments written during the merge: compact them as usual.
                self.bg.want_merge.store(true, Ordering::Release);
            }
            None if in_place => {
                // Required work without an available input claim indicates a
                // concurrent maintenance conflict. Never turn that condition
                // into an unbounded foreground encoding under the core lock.
                return Err(EsError::new(503, "unavailable_shards_exception", "force merge could not claim its inputs; retry after maintenance"));
            }
            None => {}
        }
        // 3. Commit (ES force merge flushes by default): seals what was
        //    written meanwhile, commits, rolls and trims the translog.
        if flush {
            self.flush()?;
        }
        if noop {
            return Ok(());
        }
        self.counters.merge_total.fetch_add(1, Ordering::Relaxed);
        self.counters.merge_docs.fetch_add(merged_docs, Ordering::Relaxed);
        self.counters.merge_size_bytes.fetch_add(merged_bytes, Ordering::Relaxed);
        Counters::add_time(&self.counters.merge_time_nanos, t0.elapsed());
        self.files_changed();
        Ok(())
    }

    /// Store size on disk (segments + translog).
    /// The average `_source` bytes of a few documents of the current reader
    /// view (docs spread over its largest leaf; `None` when none has a
    /// `_source`), sampled once per view generation: a coordinator sizes
    /// every multi-shard search's hits with it (large hits fetch only the
    /// winners), so the per-request cost is a lock and a compare, not three
    /// stored-document reads per shard.
    pub fn sampled_source_bytes(&self) -> Option<u64> {
        const SAMPLES: u64 = 3;
        let view = self.reader_view();
        // The view's generation, and its address for views no slot numbered.
        let generation = (view.generation(), Arc::as_ptr(&view) as usize);
        if let Some((g, bytes)) = *self.source_sample.lock() {
            if g == generation {
                return bytes;
            }
        }
        let leaves = view.leaves();
        let sampled = leaves.iter().max_by_key(|l| l.num_docs()).and_then(|leaf| {
            let n = leaf.num_docs();
            let (mut bytes, mut seen) = (0u64, 0u64);
            for i in 1..=SAMPLES {
                let d = (n as u64 * i / (SAMPLES + 1)) as u32;
                if d >= n || leaf.is_deleted(d) {
                    continue;
                }
                if let Ok(Some(doc)) = leaf.get_doc(d) {
                    if let Some(src) = doc.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()) {
                        bytes += src.len() as u64;
                        seen += 1;
                    }
                }
            }
            (seen > 0).then(|| bytes / seen)
        });
        *self.source_sample.lock() = Some((generation, sampled));
        sampled
    }

    pub fn store_size(&self) -> u64 {
        let fg = self.files_gen.load(Ordering::Acquire);
        if let Some((g, t, size)) = *self.store_cache.lock() {
            if g == fg && t.elapsed() < STORE_STATS_MAX_AGE {
                return size;
            }
        }
        // ES `StoreStats`: the index files only; the translog has its own
        // stats (`translog.size_in_bytes`) and is not store (a refresh-heavy
        // ingest between flushes otherwise reported its translog as store).
        let size = store_dir_size(&self.path);
        *self.store_cache.lock() = Some((fg, std::time::Instant::now(), size));
        size
    }

    /// The shard's files may have changed (see `files_gen`).
    pub(crate) fn files_changed(&self) {
        self.files_gen.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether translog appends made so far still wait for an fsync
    /// (`durability: async`): a crash now could lose them.
    pub fn translog_has_unsynced(&self) -> bool {
        let tail = self.tail.lock().clone();
        tail.is_some_and(|(_, mark)| *self.synced.lock() < mark)
    }

    /// Make every translog append so far durable (group commit, without
    /// the write lock: concurrent requests share one fsync).
    pub fn sync_translog(&self) -> Result<()> {
        let tail = self.tail.lock().clone();
        if let Some((file, mark)) = tail {
            self.group_sync(&file, mark)?;
        }
        // The history side log (no-ops, terms) appended without an fsync
        // (async durability) is made durable with the translog.
        let mut ws = self.write.lock();
        if ws.hist.side.has_unsynced() {
            if let Err(error) = ws.hist.side.sync() {
                drop(ws);
                self.mark_tragic(&format!("history side log fsync failed: {error}"));
                return Err(error);
            }
        }
        Ok(())
    }
}

fn write_state(path: &Path, st: &ShardState) -> Result<()> {
    let tmp = path.join(format!("{STATE_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec(st)?)?;
    rustic_core::platform::sync_file(&tmp)?;
    std::fs::rename(&tmp, path.join(STATE_FILE))?;
    rustic_core::platform::sync_dir(path)?;
    Ok(())
}

/// [`dir_size`] of a shard directory without its translog.
fn store_dir_size(p: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            if e.file_name() == TRANSLOG_DIR {
                continue;
            }
            let Ok(md) = e.metadata() else { continue };
            total += if md.is_dir() { dir_size(&e.path()) } else { md.len() };
        }
    }
    total
}

pub fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                total += dir_size(&e.path());
            } else {
                total += md.len();
            }
        }
    }
    total
}

pub type SharedShard = Arc<Shard>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::reserved_core_fields;

    #[test]
    fn pool_queue_nodes_and_escaped_checkin_flag_keep_owners() {
        let owner=rustic_core::admission::Owner::new();let _scope=owner.enter();
        let memory=rustic_core::admission::shared().reserve(PoolWriter::queue_bytes() as u64,"queue test").unwrap();
        let writer=PoolWriter::fresh_with_credit(Some(memory));
        let flag=writer.checked_in.clone();
        let mut pool=std::collections::LinkedList::new();pool.push_back(writer);
        let mut frozen=std::collections::LinkedList::new();frozen.append(&mut pool);
        assert_eq!(owner.bytes().current,PoolWriter::queue_bytes() as u64);
        drop(frozen);
        assert_eq!(owner.bytes().current,PoolWriter::flag_bytes() as u64,"pending completion still holds flag");
        drop(flag);assert_eq!(owner.bytes().current,0);
    }

    #[test]
    fn staged_claim_keeps_admission_until_last_reference() {
        let tmp=tempfile::tempdir().unwrap();let shard=Shard::create(tmp.path(),"i",schema()).unwrap();
        let owner=rustic_core::admission::Owner::new();let _scope=owner.enter();
        let mut ws=shard.write.lock();
        ws.claim(1,Shard::prepare_claim_node().unwrap());
        assert_eq!(owner.bytes().current,2048);
        ws.claim(1,Shard::prepare_claim_node().unwrap());
        assert_eq!(owner.bytes().current,2048,"duplicate retains original node ownership");
        ws.release_claim(1);assert_eq!(owner.bytes().current,2048);
        ws.release_claim(1);assert_eq!(owner.bytes().current,0);
    }

    #[test]
    fn pooled_registry_preflight_and_handoff_keep_live_buffer_owners() {
        let owner=rustic_core::admission::Owner::new();let _scope=owner.enter();
        let mut pooled=PoolWriter::fresh();
        pooled.prepare_registry(3).unwrap();
        let retained=owner.bytes().current;
        assert!(retained>0);
        assert!(pooled.prepare_registry(usize::MAX).is_err());
        assert_eq!(owner.bytes().current,retained);
        let bytes=4*(std::mem::size_of::<PoolDoc>()+std::mem::size_of::<Arc<ClaimRef>>())+PoolWriter::queue_bytes();
        let credits=rustic_core::admission::shared().reserve(bytes as u64,"handoff test").unwrap();
        let mut holding=Holding{token:0,seen:0,claim:None,registry_credits:Some(credits)};
        let before=owner.bytes().current;
        let mut next=PoolWriter::fresh();next.admit_claim_slot(&mut holding);next.admit_next_doc(&mut holding);
        assert_eq!(owner.bytes().current,before,"handoff only transfers pre-WAL credits");
        drop(holding);
        assert_eq!(owner.bytes().current,retained+(std::mem::size_of::<PoolDoc>()+std::mem::size_of::<Arc<ClaimRef>>()) as u64);
        drop(next);drop(pooled);assert_eq!(owner.bytes().current,0);
    }

    /// A refused seal admission's backoff: no seal starts (and the core is
    /// not taken) before its deadline, and the seal stays due.
    #[test]
    fn refused_seal_backoff_defers_but_keeps_the_seal_due() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for row in 0..3 {
            s.apply("i", "u", vec![idx(&row.to_string(), "x", WriteCond::default())], false).remove(0).unwrap();
            s.refresh().unwrap();
        }
        s.bg.idle_seal.store(true, Ordering::Release);
        s.bg.seal_backoff_until.store(seal_clock_nanos() + 60_000_000_000, Ordering::Release);
        let core = s.core.write();
        assert!(s.start_seal().is_none(), "backing off: returns without the core lock");
        drop(core);
        assert!(s.bg.want_seal.load(Ordering::Acquire), "still due");
        s.bg.seal_backoff_until.store(0, Ordering::Release);
        let task = s.start_seal().expect("a seal after the backoff");
        assert_eq!(s.bg.seal_refusals.load(Ordering::Acquire), 0);
        s.finish_seal(task.run());
        assert!(!s.core.read().seal_in_flight());
    }

    /// Review PR3 H1 residual: a served copy's background seals run on the
    /// refresh workers, off the replay's thread. While a replay of
    /// acknowledged ops writes to the copy they are forced (job admission,
    /// encode and install charged, never refused), or the forced replay
    /// would keep growing memtables that a refused seal cannot write out.
    /// Ordinary writes keep ordinary seals.
    #[test]
    fn background_seals_of_a_copy_a_replay_writes_to_are_forced() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        // As the refresh worker: another thread, outside any replay.
        let seal_on_worker = |s: &Shard| std::thread::scope(|scope| scope.spawn(|| {
            s.bg.idle_seal.store(true, Ordering::Release);
            let task = s.start_seal().expect("a background seal");
            let forced = (task.forced, task.job.is_forced());
            s.finish_seal(task.run());
            assert!(!s.core.read().seal_in_flight());
            forced
        }).join().unwrap());
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        s.refresh().unwrap();
        assert!(!s.replay_seals_forced());
        assert_eq!(seal_on_worker(&s), (false, false), "an ordinary write's seal is not forced");
        crate::memory::replaying("test replay", || s.replay_index("r".into(), r#"{"title":"r"}"#.into(), doc("r"), vec![], None, 1, 1)).unwrap();
        assert!(s.replay_seals_forced(), "the replay's write is noted");
        assert_eq!(seal_on_worker(&s), (true, true), "the replay's memtables are sealed forced");
        assert!(s.get("r").unwrap().is_some());
    }

    /// Tom's replay policy: only recovery replays (WAL at open, peer
    /// recovery, promotion) force the copy's background seals. A live
    /// replicated batch's completion past its WAL append charges without
    /// refusal (a live `memory::Replay`) but forces no seal, so a replica
    /// under a bulk load keeps ordinary seals; nested in a peer recovery,
    /// a batch is part of the recovery.
    #[test]
    fn live_replicated_writes_keep_ordinary_seals_and_recovery_replays_force_them() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        let seal_on_worker = |s: &Shard| std::thread::scope(|scope| scope.spawn(|| {
            s.bg.idle_seal.store(true, Ordering::Release);
            let task = s.start_seal().expect("a background seal");
            let forced = (task.forced, task.job.is_forced());
            s.finish_seal(task.run());
            forced
        }).join().unwrap());
        crate::memory::replaying_live("test live batch", || {
            assert!(crate::memory::is_replaying() && !crate::memory::is_recovery_replay());
            s.replay_index("l".into(), r#"{"title":"l"}"#.into(), doc("l"), vec![], None, 0, 1)
        }).unwrap();
        assert!(!s.replay_seals_forced(), "a live replicated batch forced the copy's seals");
        assert_eq!(seal_on_worker(&s), (false, false), "a live batch's memtables are sealed with ordinary admission");
        crate::memory::replaying("test peer recovery", || crate::memory::replaying_live("recovered batch", || {
            s.replay_index("p".into(), r#"{"title":"p"}"#.into(), doc("p"), vec![], None, 1, 1)
        })).unwrap();
        assert!(s.replay_seals_forced(), "a peer recovery's batch did not force the copy's seals");
        assert_eq!(seal_on_worker(&s), (true, true));
        assert!(s.get("l").unwrap().is_some() && s.get("p").unwrap().is_some());
    }

    /// Pre-WAL handoff credits that fall short (here: none at all) at every
    /// post-WAL boundary (claim slot, document slots doubling, a freeze
    /// handoff mid-batch) neither panic nor fail the request, and the shard
    /// keeps indexing, refreshing and flushing.
    #[test]
    fn short_handoff_credits_never_panic_or_fail_the_shard() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let empty = rustic_core::admission::shared().reserve(0, "short handoff test").unwrap();
        let mut holding = Holding { token: 0, seen: 0, claim: None, registry_credits: Some(empty) };
        let mut pooled = PoolWriter::fresh();
        for i in 0..1000u32 {
            pooled.admit_claim_slot(&mut holding);
            pooled.admit_next_doc(&mut holding);
            pooled.docs.push(PoolDoc { id: registry::DeleteKey::prepare(&format!("d{i}")).unwrap(), seq_no: i as i64, local: i, older: false });
        }
        assert!(pooled.docs.capacity() >= 1000);
        // A freeze handoff with no credit left.
        let (mut held, mut writer) = s.hold_pool_writer(None).unwrap();
        held.registry_credits = Some(rustic_core::admission::shared().reserve(0, "short handoff test").unwrap());
        writer.docs.push(PoolDoc { id: registry::DeleteKey::prepare("x").unwrap(), seq_no: 0, local: 0, older: false });
        s.freeze_gen.fetch_add(1, Ordering::AcqRel);
        s.observe_freeze(&mut held, &mut writer);
        assert_eq!(s.frozen.lock().len(), 1, "the memtable was handed off");
        writer.admit_next_doc(&mut held);
        s.release_holding(held);
        s.frozen.lock().clear();
        drop((pooled, writer, holding));
        // Credits short and the fallback admission refused too (a full
        // budget): every boundary still proceeds, unadmitted, and the
        // shortfall is recorded as unplanned instead of failing.
        let full = rustic_core::admission::Budget::new(0);
        TEST_HANDOFF_BUDGET.with(|b| *b.borrow_mut() = Some(full.clone()));
        let owner = rustic_core::admission::Owner::new();
        {
            let _owner = owner.enter();
            let mut holding = Holding { token: 0, seen: 0, claim: None, registry_credits: Some(full.reserve(0, "short handoff test").unwrap()) };
            let mut pooled = PoolWriter::fresh();
            for i in 0..1000u32 {
                pooled.admit_claim_slot(&mut holding);
                pooled.admit_next_doc(&mut holding);
                pooled.docs.push(PoolDoc { id: registry::DeleteKey::prepare(&format!("r{i}")).unwrap(), seq_no: i as i64, local: i, older: false });
            }
            assert!(pooled.docs.capacity() >= 1000 && pooled.registered.capacity() >= 1, "growth proceeded without admission");
            assert!(pooled.docs_memory.is_none(), "nothing was admitted from the full budget");
            let (mut held, mut writer) = s.hold_pool_writer(None).unwrap();
            held.registry_credits = Some(full.reserve(0, "short handoff test").unwrap());
            writer.docs.push(PoolDoc { id: registry::DeleteKey::prepare("y").unwrap(), seq_no: 0, local: 0, older: false });
            s.freeze_gen.fetch_add(1, Ordering::AcqRel);
            s.observe_freeze(&mut held, &mut writer);
            assert_eq!(s.frozen.lock().len(), 1, "the memtable was handed off without credit or admission");
            s.release_holding(held);
            s.frozen.lock().clear();
            drop((pooled, writer, holding));
        }
        TEST_HANDOFF_BUDGET.with(|b| *b.borrow_mut() = None);
        assert!(owner.has_unplanned_allocations(), "the refused handoffs were recorded as unplanned");
        assert_eq!(full.used(), 0);
        // The shard is healthy: it indexes, refreshes and flushes.
        assert!(s.apply("i", "u", vec![idx("1", "after short credits", WriteCond::default())], false)[0].is_ok());
        s.refresh().unwrap();
        s.flush().unwrap();
        assert_eq!(s.core.read().num_live_docs(), 1);
        assert!(s.tragic_error().is_none());
    }


    fn schema() -> Vec<CoreField> {
        let mut s = reserved_core_fields();
        s.push(CoreField::new(8, "title", rustic_core::types::FieldType::Text));
        s
    }

    fn doc(title: &str) -> Document {
        let mut d = Document::new();
        d.add_text(8, title);
        d
    }

    fn idx(id: &str, title: &str, cond: WriteCond) -> WriteOp {
        WriteOp::Index { id: id.into(), source: format!("{{\"title\":\"{title}\"}}"), doc: doc(title), ignored: vec![], routing: None, cond, auto_id: false }
    }

    /// Large append-only batches written from a rayon pool while the
    /// indexing throttle is engaged all complete: a batch inverting its
    /// parts must not, while holding the throttle and `batches`, pick up
    /// another batch on its thread (which then waits for a throttle this
    /// thread, or a thread waiting for this one, holds, forever). Two
    /// shards, so threads holding one shard's throttle are free to steal.
    #[test]
    fn throttled_append_only_batches_from_a_pool_never_wait_on_themselves() {
        let tmp = tempfile::tempdir().unwrap();
        let shards: Vec<std::sync::Arc<Shard>> = (0..2).map(|i| std::sync::Arc::new(Shard::create(&tmp.path().join(i.to_string()), "thr", schema()).unwrap())).collect();
        for s in &shards {
            s.bg.stall_limit_override.store(u64::MAX / 2, Ordering::Release);
            s.set_memory_throttled(true);
        }
        let title = "w ".repeat(1500);
        let batch = move |b: usize| -> Vec<WriteOp> {
            (0..1000).map(|i| WriteOp::Index { id: format!("{b}-{i}"), source: format!("{{\"title\":\"{title}\"}}"), doc: doc(&title), ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true }).collect()
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let pool_shards = shards.clone();
        std::thread::spawn(move || {
            use rayon::prelude::*;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
            let ok = pool.install(|| (0..48usize).into_par_iter().map(|b| pool_shards[b % 2].apply("thr", "u", batch(b), false).iter().all(|r| r.is_ok())).collect::<Vec<_>>());
            let _ = tx.send(ok);
        });
        let ok = rx.recv_timeout(std::time::Duration::from_secs(120)).expect("throttled batches deadlocked");
        assert!(ok.iter().all(|o| *o));
        for s in &shards {
            s.set_memory_throttled(false);
            s.checkin_completed_writes(None);
            let live: u64 = s.core.read().all_leaves().iter().map(|l| l.num_live_docs_latest() as u64).sum();
            assert_eq!(live, 24_000);
        }
    }

    #[test]
    fn flush_and_reopen_schedule_existing_tiny_segments_for_compaction() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "tiny", schema()).unwrap();
        for i in 0..30 {
            assert!(s.apply("tiny", "u", vec![idx(&i.to_string(), "small", WriteCond::default())], true)[0].is_ok());
            s.refresh().unwrap();
            s.flush().unwrap();
            assert!(s.bg.want_merge.load(Ordering::Acquire));
        }
        assert!(s.core.read().num_segments() >= 30);
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.is_empty());
        assert!(s.background_due(), "reopen must schedule work without waiting for writes");
        s.background_until(|s| s.core.read().num_segments() <= merge_policy().max_segments_per_tier);
        assert!(s.core.read().num_segments() <= merge_policy().max_segments_per_tier);
        for i in 0..30 { assert!(s.get(&i.to_string()).unwrap().is_some()); }
    }

    #[test]
    fn replica_writes_request_seals_without_refresh() {
        for batched in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "replica-seal", schema()).unwrap();
            // Drive the background pass explicitly, so this verifies replay
            // itself requests work rather than relying on refresh ticks.
            s.bg.served.store(true, Ordering::Release);
            s.bg.threshold.store(64, Ordering::Relaxed);
            for round in 0..2 {
                let ids: Vec<_> = (round*64..(round+1)*64).map(|i| i.to_string()).collect();
                if batched {
                    let ops: Vec<_> = ids.iter().enumerate().map(|(i,id)| ReplayIndex {
                        id, source: "{}", routing: None, seq_no: (round*64+i) as i64, version: 1,
                    }).collect();
                    let mut prepared = ids.iter().map(|_| crate::document::ParsedDoc {
                        doc: doc("replica"), ignored: vec![], source: None,
                    }).collect();
                    assert!(s.try_replay_new_batch(&ops, &mut prepared).unwrap());
                } else {
                    for (i,id) in ids.iter().enumerate() {
                        s.replay_index(id.clone(), "{}".into(), doc("replica"), vec![], None, (round*64+i) as i64, 1).unwrap();
                    }
                }
                assert!(s.bg.want_seal.load(Ordering::Acquire), "batched={batched}");
                s.run_background();
                assert_eq!(s.core.read().num_writing_docs(), 0, "batched={batched}");
                for id in &ids { assert!(s.get(id).unwrap().is_some(), "missing {id}"); }
            }
            assert_eq!(s.max_seq_no(), 127);
            s.flush().unwrap();
            drop(s);
            let (reopened, _) = Shard::open(tmp.path(), schema()).unwrap();
            for i in 0..128 { assert!(reopened.get(&i.to_string()).unwrap().is_some()); }
        }
    }

    #[test]
    fn primary_translog_format_gate_reads_legacy_and_rejects_old_readers() {
        let legacy = serde_json::json!({"committed_seq_no":-1,"translog_generation":1,"core_generation":0});
        let state: ShardState = serde_json::from_value(legacy).unwrap();
        let current = serde_json::to_value(&state).unwrap();
        assert_eq!(current["translog_generation"], serde_json::json!({"format":2,"generation":1}));
        #[derive(Deserialize)]
        struct Old { #[allow(dead_code)] translog_generation: u64 }
        assert!(serde_json::from_value::<Old>(current.clone()).is_err());
        let mut future = current;
        future["translog_generation"]["format"] = serde_json::json!(3);
        assert!(serde_json::from_value::<ShardState>(future).is_err());
    }

    #[test]
    fn get_failure_is_not_a_missing_document_or_a_missing_stat() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let error = s.counted_get(|| Err(rustic_core::RusticError::Memory("decode denied".into()).into())).unwrap_err();
        assert_eq!(error.status, 429);
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert_eq!(s.counters.get_total.load(Ordering::Relaxed), 1);
        assert_eq!(s.counters.get_missing_total.load(Ordering::Relaxed), 0);
        assert_eq!(s.counters.get_exists_total.load(Ordering::Relaxed), 0);
        assert_eq!(s.counters.get_current.load(Ordering::Relaxed), 0);
        assert!(s.get("absent").unwrap().is_none());
        assert_eq!(s.counters.get_missing_total.load(Ordering::Relaxed), 1);
        assert!(s.peek("absent").unwrap().is_none());
        assert_eq!(s.counters.get_total.load(Ordering::Relaxed), 2, "write prechecks do not count as GETs");
    }

    #[test]
    fn versions_seq_nos_and_conflicts() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let r = s.apply("i", "u", vec![idx("1", "a", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!((r.version, r.seq_no, r.result), (1, 0, "created"));
        let r = s.apply("i", "u", vec![idx("1", "b", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!((r.version, r.seq_no, r.result), (2, 1, "updated"));
        // create on existing
        let e = s.apply("i", "u", vec![idx("1", "c", WriteCond { create: true, ..Default::default() })], true).remove(0).unwrap_err();
        assert_eq!(e.status, 409);
        assert_eq!(e.reason, "[1]: version conflict, document already exists (current version [2])");
        // if_seq_no mismatch
        let e = s.apply("i", "u", vec![idx("1", "c", WriteCond { if_seq_no: Some(0), if_primary_term: Some(1), ..Default::default() })], true).remove(0).unwrap_err();
        assert_eq!(e.reason, "[1]: version conflict, required seqNo [0], primary term [1]. current document has seqNo [1] and primary term [1]");
        let r = s.apply("i", "u", vec![idx("1", "c", WriteCond { if_seq_no: Some(1), if_primary_term: Some(1), ..Default::default() })], true).remove(0).unwrap();
        assert_eq!(r.version, 3);
        // external versions
        let r = s.apply("i", "u", vec![idx("2", "x", WriteCond { version: Some(10), version_type: Some(VersionType::External), ..Default::default() })], true).remove(0).unwrap();
        assert_eq!(r.version, 10);
        let e = s.apply("i", "u", vec![idx("2", "x", WriteCond { version: Some(10), version_type: Some(VersionType::External), ..Default::default() })], true).remove(0).unwrap_err();
        assert_eq!(e.reason, "[2]: version conflict, current version [10] is higher or equal to the one provided [10]");
        // get + delete
        let g = s.get("1").unwrap().unwrap();
        assert_eq!(g.source.as_deref(), Some("{\"title\":\"c\"}"));
        assert_eq!(g.version, 3);
        let r = s.apply("i", "u", vec![WriteOp::Delete { id: "1".into(), cond: WriteCond::default() }], true).remove(0).unwrap();
        assert_eq!((r.result, r.version), ("deleted", 4));
        assert!(s.get("1").unwrap().is_none());
        let r = s.apply("i", "u", vec![WriteOp::Delete { id: "1".into(), cond: WriteCond::default() }], true).remove(0).unwrap();
        assert_eq!(r.result, "not_found");
        // re-create after delete continues the version
        let r = s.apply("i", "u", vec![idx("1", "d", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!((r.version, r.result), (6, "created"));
    }

    #[test]
    fn write_and_get_carry_the_documents_primary_term() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let r = s.apply("i", "u", vec![idx("1", "a", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!(r.primary_term, s.primary_term());
        assert_eq!(r.primary_term, 1);
        s.set_primary_term(2).unwrap();
        let r = s.apply("i", "u", vec![idx("2", "b", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!(r.primary_term, 2);
        // GET: each document's own term.
        assert_eq!(s.get("1").unwrap().unwrap().primary_term, 1);
        assert_eq!(s.get("2").unwrap().unwrap().primary_term, 2);
        // Delete: the shard's current term.
        let r = s.apply("i", "u", vec![WriteOp::Delete { id: "1".into(), cond: WriteCond::default() }], true).remove(0).unwrap();
        assert_eq!((r.result, r.primary_term), ("deleted", 2));
    }

    #[test]
    fn validation() {
        assert!(WriteCond { version: Some(1), ..Default::default() }.validate(false).is_err());
        assert!(WriteCond { if_seq_no: Some(1), ..Default::default() }.validate(false).is_err());
        assert!(WriteCond { version: Some(1), version_type: Some(VersionType::External), ..Default::default() }.validate(false).is_ok());
    }

    #[test]
    fn flush_reopen_and_translog_replay() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            s.apply("i", "u", vec![idx("1", "a", WriteCond::default()), idx("2", "b", WriteCond::default())], true);
            s.flush().unwrap();
            s.apply("i", "u", vec![idx("3", "c", WriteCond::default()), WriteOp::Delete { id: "1".into(), cond: WriteCond::default() }], true);
        }
        let (s, ops) = Shard::open(tmp.path(), schema()).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(s.max_seq_no(), 1);
        for op in ops.iter().cloned() {
            match op {
                TranslogOp::Index { id, source_json, seq_no, version, routing } => s.replay_index(id, source_json, doc("c"), vec![], routing, seq_no as i64, version).unwrap(),
                TranslogOp::Delete { id, seq_no, version } => s.replay_delete(&id, seq_no as i64, version).unwrap(),
            }
        }
        assert!(s.get("1").unwrap().is_none());
        assert!(s.get("2").unwrap().is_some());
        assert_eq!(s.get("3").unwrap().unwrap().seq_no, 2);
        assert_eq!(s.max_seq_no(), 3);
        // next op continues the sequence
        let r = s.apply("i", "u", vec![idx("4", "d", WriteCond::default())], true).remove(0).unwrap();
        assert_eq!(r.seq_no, 4);
    }

    /// Review PR3 H3: a local translog replay seals as its memtables fill,
    /// though no scheduler serves the copy yet, instead of holding every
    /// replayed op in memtables until the final publish.
    #[test]
    fn translog_replay_seals_as_the_buffer_fills_before_the_copy_is_served() {
        let tmp = tempfile::tempdir().unwrap();
        let total = 1000;
        {
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            s.bg.threshold.store(1_000_000, Ordering::Relaxed);
            for chunk in (0..total).collect::<Vec<_>>().chunks(100) {
                let ops = chunk.iter().map(|i| idx(&i.to_string(), "t", WriteCond::default())).collect();
                assert!(s.apply("i", "u", ops, true).iter().all(Result::is_ok));
            }
        }
        let (s, ops) = Shard::open(tmp.path(), schema()).unwrap();
        assert_eq!(ops.len(), total);
        assert!(!s.bg.served.load(Ordering::Acquire));
        let threshold = 100;
        s.bg.threshold.store(threshold, Ordering::Relaxed);
        let mut peak = 0;
        crate::memory::replaying("test translog replay", || {
            for chunk in ops.chunks(64) {
                for op in chunk.iter().cloned() {
                    let TranslogOp::Index { id, source_json, seq_no, version, routing } = op else { unreachable!() };
                    s.replay_index(id, source_json, doc("t"), vec![], routing, seq_no as i64, version).unwrap();
                }
                peak = peak.max(s.core.read().num_writing_docs());
                s.seal_replay_if_buffer_full();
                assert!(s.core.read().num_writing_docs() < threshold, "the replay sealed its memtables once they reached the threshold");
            }
        });
        assert!(peak < threshold + 64, "memtables stayed bounded by the threshold plus a chunk: {peak}");
        s.publish().unwrap();
        for i in [0, total / 2, total - 1] {
            assert!(s.get(&i.to_string()).unwrap().is_some(), "{i}");
        }
        assert_eq!(s.max_seq_no(), total as i64 - 1);
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn mutable_recovery_batch_prepares_before_publication_and_rejects_existing_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        let ids: Vec<_> = (0..63).map(|i| i.to_string()).collect();
        let ops: Vec<_> = ids.iter().enumerate().map(|(i,id)| ReplayIndex {
            id, source: "{}", routing: None, seq_no: i as i64, version: 1 }).collect();
        let make_docs = |invalid: bool| (0..63).map(|i| {
            let mut doc = doc("recovery");
            doc.add_vector(9, vec![if invalid && i == 31 { f32::NAN } else { i as f32 }, 1.]);
            crate::document::ParsedDoc { doc, ignored: vec![], source: None }
        }).collect::<Vec<_>>();
        let mut invalid = make_docs(true);
        assert!(!s.try_replay_new_batch(&ops, &mut invalid).unwrap());
        assert_eq!(invalid.len(), 63);
        assert!(ids.iter().all(|id| s.get(id).unwrap().is_none()));
        assert_eq!(s.max_seq_no(), -1);
        let mut valid = make_docs(false);
        assert!(s.try_replay_new_batch(&ops, &mut valid).unwrap());
        assert!(valid.is_empty());
        assert!(ids.iter().all(|id| s.get(id).unwrap().is_some()));
        let mut duplicate = make_docs(false);
        assert!(!s.try_replay_new_batch(&ops, &mut duplicate).unwrap());
        assert_eq!(duplicate.len(), 63);
        s.flush().unwrap(); drop(s);
        let (s, _) = Shard::open(tmp.path(), fields).unwrap();
        assert!(ids.iter().all(|id| s.get(id).unwrap().is_some()));
        assert_eq!(s.max_seq_no(), 62);
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn mutable_backend_is_selected_by_create_schema_and_reopen_without_environment() {
        let vector = CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw()));
        for dynamic in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let mut fields = schema();
            if !dynamic { fields.push(vector.clone()); }
            let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
            if dynamic { fields.push(vector.clone()); s.sync_schema(&fields).unwrap(); }
            assert_eq!(s.core.read().vector_store().mutable_ingestion_batch().unwrap().len(), 1);
            let mut document = doc("vector"); document.add_vector(9, vec![1.,2.]);
            let result = s.apply("i", "u", vec![WriteOp::Index { id: "1".into(), source: "{}".into(),
                doc: document, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }], true);
            assert!(result[0].is_ok());
            s.flush().unwrap(); drop(s);
            let (s, replay) = Shard::open(tmp.path(), fields).unwrap();
            assert!(replay.is_empty());
            assert_eq!(s.core.read().vector_store().mutable_ingestion_batch().unwrap().len(), 1);
            assert!(s.get("1").unwrap().is_some());
        }
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn mutable_bulk_admission_rejects_before_sequence_and_replays_only_successes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        let ops = (0..130).map(|i| {
            let mut doc = doc("bulk");
            doc.add_vector(9, vec![if i == 65 { f32::NAN } else { i as f32 },1.]);
            WriteOp::Index { id: i.to_string(), source: "{}".into(), doc, ignored: vec![],
                routing: None, cond: WriteCond::default(), auto_id: true }
        }).collect();
        let results = s.apply_mutable_append_only(ops, true);
        assert_eq!(results.len(), 130);
        s.refresh().unwrap();
        s.seal_published();
        assert_eq!(s.core.read().num_segments(), 1,
            "64-document preparation tasks must not become separate text segments");
        assert!(results[65].is_err());
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 129);
        assert_eq!(s.max_seq_no(), 128);
        assert!(s.get("65").unwrap().is_none());
        for (seq, result) in results.iter().filter_map(|result| result.as_ref().ok()).enumerate() {
            assert_eq!(result.seq_no, seq as i64);
        }
        drop(s);
        let (_, replay) = Shard::open(tmp.path(), fields).unwrap();
        assert_eq!(replay.len(), 129);
        assert!(replay.iter().all(|op| !matches!(op, TranslogOp::Index { id, .. } if id == "65")));
    }

    /// An explicit-id bulk stages its vector documents together
    /// (`prepare_documents_vectors`); one invalid row fails only its own
    /// document (per-document fallback), a document without vectors is
    /// indexed, and a later bulk overwriting ids replaces their rows.
    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn explicit_id_vector_bulk_stages_together_and_reports_each_document() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        let make = |i: usize, x: f32| {
            let mut d = doc("explicit");
            if i != 7 { d.add_vector(9, vec![x, 1.]); }
            WriteOp::Index { id: i.to_string(), source: "{}".into(), doc: d, ignored: vec![],
                routing: None, cond: WriteCond::default(), auto_id: false }
        };
        let results = s.apply("i", "u", (0..130).map(|i| make(i, if i == 65 { f32::NAN } else { i as f32 })).collect(), true);
        assert_eq!(results.len(), 130);
        assert!(results[65].is_err());
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 129);
        let overwrite = s.apply("i", "u", (0..4).map(|i| make(i, 1000. + i as f32)).collect(), true);
        assert!(overwrite.iter().all(Result::is_ok));
        s.refresh().unwrap();
        for id in (0..130).filter(|&i| i != 65) { assert!(s.get(&id.to_string()).unwrap().is_some(), "{id}"); }
        assert!(s.get("65").unwrap().is_none());
        drop(s);
        let (_, replay) = Shard::open(tmp.path(), fields).unwrap();
        assert_eq!(replay.len(), 133);
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn forwarded_mutable_creates_prepare_in_parallel_without_bypassing_conflicts() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        let make = |id: usize| {
            let mut d = doc("forwarded"); d.add_vector(9, vec![id as f32, 1.]);
            WriteOp::Index { id: id.to_string(), source: "{}".into(), doc: d, ignored: vec![],
                routing: None, cond: WriteCond { create: true, ..Default::default() }, auto_id: false }
        };
        // Assert actual admission into the parallel path, not merely HTTP success.
        {
            let ws = s.write.lock(); let mut core = s.core.write();
            let tokens = s.prepare_fresh_mutable_batch(&ws, &mut core, &[make(0), make(1)]).unwrap();
            assert!(tokens.iter().all(|token| token.is_ok()));
            drop(tokens); // cancellation must not become a visible document
        }
        // Exercise the production caller: a Rayon worker holding shard locks
        // must not steal the competing create while waiting for preparation.
        let outer = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let (first, second) = outer.install(|| rayon::join(
            || s.apply("i", "uuid", (0..128).map(make).collect(), true),
            || s.apply("i", "uuid", (0..128).map(make).collect(), true)));
        assert_eq!(first.iter().chain(&second).filter(|r| r.is_ok()).count(), 128);
        assert_eq!(first.iter().chain(&second).filter(|r| r.is_err()).count(), 128);
        let before = s.max_seq_no();
        let retry = s.apply("i", "uuid", vec![make(0), make(1)], true);
        assert!(retry.iter().all(Result::is_err));
        assert_eq!(s.max_seq_no(), before);
        let duplicate = s.apply("i", "uuid", vec![make(128), make(128)], true);
        assert!(duplicate[0].is_ok()); assert!(duplicate[1].is_err());
        s.refresh().unwrap();
        for id in 0..129 { assert!(s.get(&id.to_string()).unwrap().is_some()); }
        drop(s);
        let (_, replay) = Shard::open(tmp.path(), fields).unwrap();
        assert_eq!(replay.len(), 129);
        for (seq, op) in replay.iter().enumerate() { assert_eq!(op.seq_no(), seq as u64); }
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn concurrent_mutable_batches_keep_wal_sequence_order() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for batch in 0..4 {
                let s = &s; let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for round in 0..4 {
                        let ops = (0..128).map(|i| {
                            let id = batch * 512 + round * 128 + i;
                            let mut doc = doc("ordered"); doc.add_vector(9, vec![id as f32, 1.]);
                            WriteOp::Index { id: id.to_string(), source: "{}".into(), doc,
                                ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true }
                        }).collect();
                        assert!(s.apply_mutable_append_only(ops, true).iter().all(Result::is_ok));
                    }
                });
            }
        });
        drop(s);
        let (_, replay) = Shard::open(tmp.path(), fields).unwrap();
        assert_eq!(replay.len(), 2048);
        for (expected, operation) in replay.iter().enumerate() {
            assert_eq!(operation.seq_no(), expected as u64, "tailer requires monotonically ordered WAL");
        }
    }

    #[test]
    fn force_merge_keeps_accepting_writes() {
        // ES / Lucene: indexing continues during a force merge. New docs,
        // updates and deletes of docs being merged, issued while the merge's
        // output I/O is stalled, complete at once and survive its install and
        // commit.
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
            thread: parking_lot::Mutex<Option<String>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    *self.thread.lock() = std::thread::current().name().map(str::to_string);
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(), pause: std::sync::atomic::AtomicBool::new(false), entered, resume: parking_lot::Mutex::new(release), thread: Default::default() });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
        core.set_nrt(true);
        core.adopt_view_slot(s.views.clone()).unwrap();
        *s.core.write() = core;
        for row in 0..4 {
            assert!(s.apply("i", "u", vec![idx(&row.to_string(), "old", WriteCond::default())], false)[0].is_ok());
            s.checkin_pool(None); s.core.write().refresh().unwrap();
        }
        dir.pause.store(true, Ordering::Release);
        let merging = s.clone();
        let merge = std::thread::spawn(move || merging.force_merge(1, false));
        wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let writer = s.clone();
        let (done, written) = std::sync::mpsc::channel();
        let writes = std::thread::spawn(move || {
            let r = writer.apply("i", "u", vec![
                idx("new", "fresh", WriteCond::default()),
                idx("1", "updated", WriteCond::default()),
                WriteOp::Delete { id: "2".into(), cond: WriteCond::default() },
            ], true);
            done.send(()).unwrap();
            r
        });
        let completed = written.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        // More writes, sealed into segments while the merge is stalled: a
        // background merge compacts them alongside it.
        for round in 0..4 {
            assert!(s.apply("i", "u", vec![idx(&format!("bg-{round}"), "bg", WriteCond::default())], false)[0].is_ok());
            s.checkin_pool(None); s.core.write().refresh().unwrap();
            let seal = s.core.write().begin_seal(true).unwrap();
            if let Some(job) = seal { let done = job.run(); s.core.write().finish_seal(done).unwrap(); }
        }
        let before_bg = s.core.read().num_segments();
        let policy = rustic_core::merge::MergePolicy { max_segments_per_tier: 2, floor_segment_docs: 1, ..Default::default() };
        let bg = s.core.write().begin_merge(&policy);
        let bg_ran = bg.map(|job| { let m = job.run().unwrap(); s.core.write().finish_merge(job, m).unwrap() }).unwrap_or(false);
        let after_bg = s.core.read().num_segments();
        // Always release the merge, including when the assertion fails.
        resume.send(()).unwrap();
        merge.join().unwrap().unwrap();
        let results = writes.join().unwrap();
        assert!(completed, "writes waited behind the force merge");
        assert!(bg_ran && after_bg < before_bg, "background merge during the force merge: {before_bg} -> {after_bg}");
        assert!(dir.thread.lock().as_deref().is_some_and(|n| n.starts_with("rustic-force-merge")), "force merge encoded on {:?}", dir.thread.lock());
        assert!(results.iter().all(|r| r.is_ok()), "{results:?}");
        s.refresh().unwrap();
        assert!(s.get("new").unwrap().is_some());
        assert!(s.get("2").unwrap().is_none(), "a delete of a merged doc during the merge was lost");
        let one = s.get("1").unwrap().unwrap();
        assert_eq!(one.version, 2, "the update during the merge was lost or doubled");
        assert!(s.get("0").unwrap().is_some() && s.get("3").unwrap().is_some());
        assert_eq!(s.core.read().num_live_docs(), 8);
        // The merge's flush committed the writes made meanwhile.
        assert_eq!(s.committed_seq_no(), s.max_seq_no());
        assert_eq!(Counters::get(&s.counters.merge_total), 1);
    }

    /// Live copies of `id` in the core (realtime deletions).
    fn live_copies(s: &Shard, id: &str) -> usize {
        let core = s.core.read();
        let leaves = core.all_leaves();
        let term = Term::new_text(ID_FIELD, id);
        leaves.iter().map(|leaf| {
            let mut n = 0;
            if let Some(mut postings) = leaf.postings(&term) {
                while let Some(d) = postings.next() {
                    if !leaf.is_deleted_latest(d) { n += 1; }
                }
            }
            n
        }).sum()
    }

    #[test]
    fn concurrent_writers_on_overlapping_ids_keep_es_semantics() {
        // W2: many threads, overlapping ids, batches repeating ids, creates,
        // deletes, conditional (if_seq_no) read-modify-writes, with
        // concurrent refreshes and flushes. Afterwards, per id in seq_no
        // order, every successful op is what ES allows (create only on a
        // missing doc, if_seq_no only on that exact op, versions +1), the
        // index holds exactly one live copy of the last op's doc (none
        // after a delete), and realtime GET agrees.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self, n: u64) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 % n }
        }
        #[derive(Clone, Debug)]
        struct Done { id: String, seq_no: i64, version: u64, delete: bool, create: bool, if_seq_no: Option<i64> }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = Arc::new(Mutex::new(Vec::<Done>::new()));
        let maintenance = {
            let (s, stop) = (s.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut round = 0u64;
                while !stop.load(Ordering::Acquire) {
                    s.refresh().unwrap();
                    if round % 7 == 3 { s.flush().unwrap(); }
                    if round % 11 == 5 { s.seal_published(); }
                    round += 1;
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            })
        };
        let autos = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let writers: Vec<_> = (0..8u64).map(|thread| {
            let (s, done, autos) = (s.clone(), done.clone(), autos.clone());
            std::thread::spawn(move || {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (thread + 1));
                for round in 0..250u64 {
                    // Auto-id bulks (the append-only path) interleaved.
                    if round % 9 == 4 {
                        let ops = (0..3).map(|i| WriteOp::Index { id: format!("auto-{thread}-{round}-{i}"), source: "{}".into(), doc: doc("auto"),
                            ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true }).collect();
                        let ok = s.apply("i", "u", ops, false).iter().filter(|r| r.is_ok()).count();
                        autos.fetch_add(ok as u64, Ordering::Relaxed);
                        continue;
                    }
                    let size = 1 + rng.next(4) as usize;
                    let mut ops = Vec::with_capacity(size);
                    let mut meta = Vec::with_capacity(size);
                    for _ in 0..size {
                        let id = format!("d{}", rng.next(12));
                        match rng.next(5) {
                            0 | 1 => { ops.push(idx(&id, "v", WriteCond::default())); meta.push((id, false, false, None)); }
                            2 => { ops.push(idx(&id, "v", WriteCond { create: true, ..Default::default() })); meta.push((id, false, true, None)); }
                            3 => { ops.push(WriteOp::Delete { id: id.clone(), cond: WriteCond::default() }); meta.push((id, true, false, None)); }
                            _ => {
                                // Read-modify-write (the update API's shape).
                                let Some(current) = s.get(&id).unwrap() else { continue };
                                let cond = WriteCond { if_seq_no: Some(current.seq_no), if_primary_term: Some(current.primary_term), ..Default::default() };
                                ops.push(idx(&id, "u", cond));
                                meta.push((id, false, false, Some(current.seq_no)));
                            }
                        }
                    }
                    if ops.is_empty() { continue; }
                    let results = s.apply("i", "u", ops, false);
                    let mut out = done.lock();
                    for (result, (id, delete, create, if_seq_no)) in results.into_iter().zip(meta) {
                        if let Ok(r) = result {
                            out.push(Done { id, seq_no: r.seq_no, version: r.version, delete, create, if_seq_no });
                        }
                    }
                }
            })
        }).collect();
        for w in writers { w.join().unwrap(); }
        stop.store(true, Ordering::Release);
        maintenance.join().unwrap();
        s.refresh().unwrap();
        let mut done = done.lock().clone();
        done.sort_by_key(|d| d.seq_no);
        let seqs: std::collections::HashSet<i64> = done.iter().map(|d| d.seq_no).collect();
        assert_eq!(seqs.len(), done.len(), "a seq_no was assigned twice");
        let mut state: HashMap<String, (u64, bool, i64)> = HashMap::new();
        for d in &done {
            let prior = state.get(&d.id).copied();
            let live = prior.filter(|p| !p.1);
            if d.create { assert!(live.is_none(), "create succeeded over a live doc: {d:?} after {prior:?}"); }
            if let Some(expected) = d.if_seq_no {
                assert_eq!(live.map(|p| p.2), Some(expected), "if_seq_no succeeded against another op: {d:?}");
            }
            assert_eq!(d.version, prior.map_or(1, |p| p.0 + 1), "version of {d:?} after {prior:?}");
            state.insert(d.id.clone(), (d.version, d.delete, d.seq_no));
        }
        for id in (0..12).map(|n| format!("d{n}")) {
            let expect = state.get(&id).copied().filter(|p| !p.1);
            assert_eq!(live_copies(&s, &id), usize::from(expect.is_some()), "live copies of {id} (last op {:?})", state.get(&id));
            assert_eq!(s.get(&id).unwrap().map(|g| (g.version, g.seq_no)), expect.map(|p| (p.0, p.2)), "realtime GET of {id}");
        }
        // Search agrees after the refresh: the explicit ids' live copies
        // plus every acked auto-id document.
        let visible: u64 = s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs() as u64).sum();
        let live_explicit = state.values().filter(|p| !p.1).count() as u64;
        assert_eq!(visible, live_explicit + autos.load(Ordering::Relaxed));
        let autos = autos.load(Ordering::Relaxed) as i64;
        assert_eq!(s.max_seq_no() + 1, done.len() as i64 + autos, "every assigned seq_no belongs to a successful op");
        assert_eq!(s.local_checkpoint(), s.max_seq_no(), "every op completed");
    }

    #[test]
    fn concurrent_replica_batches_apply_newest_seq_no_per_id() {
        // W4: replicated batches apply concurrently and in any order; per
        // id the newest seq_no wins (deletes included), older ones are
        // stale whenever they arrive.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let index = |id: &str, title: &str, seq_no: i64| StagedReplay::Index { id: id.into(), source: "{}".into(), doc: doc(title),
            ignored: vec![], routing: None, seq_no, version: seq_no as u64 + 1 };
        for round in 0..20i64 {
            let base = round * 10;
            let newer = vec![index("x", "new", base + 5), StagedReplay::Delete { id: "y".into(), seq_no: base + 6, version: base as u64 + 7 }];
            let older = vec![index("x", "old", base + 3), index("y", "old", base + 2), index(&format!("z{round}"), "z", base + 4)];
            let (a, b) = (s.clone(), s.clone());
            let first = std::thread::spawn(move || a.replay_staged(newer).unwrap());
            let second = std::thread::spawn(move || b.replay_staged(older).unwrap());
            let (newer, older) = (first.join().unwrap(), second.join().unwrap());
            assert!(newer.iter().all(|applied| *applied), "the newest ops always apply");
            assert!(older[2], "an unrelated id applies");
            let x = s.get("x").unwrap().unwrap();
            assert_eq!(x.seq_no, base + 5, "x holds its newest op");
            assert!(s.get("y").unwrap().is_none(), "the newest op on y is a delete");
            if round % 3 == 0 { s.refresh().unwrap(); }
        }
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "x"), 1);
        assert_eq!(live_copies(&s, "y"), 0);
        for round in 0..20 { assert_eq!(live_copies(&s, &format!("z{round}")), 1); }
    }

    #[test]
    fn mapping_change_during_sustained_bulks_does_not_block_writes() {
        // M1: a mapping change takes only the core briefly; bulks keep
        // flowing and documents with the new field index right after.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bulk = {
            let (s, stop) = (s.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut round = 0usize;
                while !stop.load(Ordering::Acquire) {
                    let ops = (0..2000).map(|i| idx(&format!("m{round}-{i}"), "text", WriteCond::default())).collect();
                    assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
                    round += 1;
                }
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut grown = schema();
        grown.push(CoreField::new(50, "added", rustic_core::types::FieldType::Keyword));
        let t0 = std::time::Instant::now();
        s.sync_schema(&grown).unwrap();
        let took = t0.elapsed();
        stop.store(true, Ordering::Release);
        bulk.join().unwrap();
        assert!(took < std::time::Duration::from_millis(500), "mapping change waited {took:?} behind bulks");
        let mut with_field = doc("new");
        with_field.fields.push((50, FieldValue::Keyword("x".into())));
        let op = WriteOp::Index { id: "added".into(), source: "{}".into(), doc: with_field, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false };
        assert!(s.apply("i", "u", vec![op], false)[0].is_ok());
        s.refresh().unwrap();
        assert!(s.get("added").unwrap().is_some());
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn vector_flush_does_not_block_writes_beyond_its_capture() {
        // Item 2: a flush of mutable vectors fences writes only at the graph
        // capture point; writes issued while its seal I/O is stalled
        // complete at once.
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let make = |id: String, x: f32| {
            let mut d = doc("vec"); d.add_vector(9, vec![x, 1.]);
            WriteOp::Index { id, source: "{}".into(), doc: d, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }
        };
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", fields.clone()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
            pause: std::sync::atomic::AtomicBool::new(false), entered, resume: parking_lot::Mutex::new(release) });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", fields, Default::default()).unwrap();
        core.set_nrt(true);
        core.set_auto_seal(false);
        core.adopt_view_slot(s.views.clone()).unwrap();
        *s.core.write() = core;
        assert!(s.apply("i", "u", (0..64).map(|i| make(format!("a{i}"), i as f32)).collect(), false).iter().all(Result::is_ok));
        dir.pause.store(true, Ordering::Release);
        let flushing = s.clone();
        let flush = std::thread::spawn(move || flushing.flush());
        wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let writer = s.clone();
        let (done, written) = std::sync::mpsc::channel();
        let writes = std::thread::spawn(move || {
            let r = writer.apply("i", "u", (0..64).map(|i| make(format!("b{i}"), 100. + i as f32)).collect(), false);
            done.send(()).unwrap();
            r
        });
        let completed = written.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        resume.send(()).unwrap();
        flush.join().unwrap().unwrap();
        let results = writes.join().unwrap();
        assert!(completed, "vector writes waited behind the flush's seal I/O");
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        s.refresh().unwrap();
        assert!(s.get("a0").unwrap().is_some() && s.get("b63").unwrap().is_some());
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn concurrent_vector_replay_and_flushes_never_fail_the_flush() {
        // B2(d): replicated vector batches applying while flushes run: a
        // flush never captures the graph with ingestion in flight.
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Arc::new(Shard::create(tmp.path(), "i", fields).unwrap());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let replicas: Vec<_> = (0..3i64).map(|thread| {
            let (s, stop) = (s.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut round = 0i64;
                while !stop.load(Ordering::Acquire) && round < 200 {
                    let ops = (0..16i64).map(|i| {
                        let seq_no = (round * 3 + thread) * 16 + i;
                        let mut d = doc("r"); d.add_vector(9, vec![seq_no as f32, 1.]);
                        StagedReplay::Index { id: format!("r{seq_no}"), source: "{}".into(), doc: d, ignored: vec![], routing: None, seq_no, version: 1 }
                    }).collect();
                    s.replay_staged(ops).unwrap();
                    round += 1;
                }
            })
        }).collect();
        for _ in 0..10 {
            s.flush().expect("flush failed beside concurrent vector replay");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        stop.store(true, Ordering::Release);
        for r in replicas { r.join().unwrap(); }
        s.flush().unwrap();
        s.refresh().unwrap();
        assert!(s.get("r0").unwrap().is_some());
    }

    #[test]
    fn single_document_writes_reuse_pooled_memtables() {
        // W1/U1: every write is staged, yet sequential single-document
        // writes share one pooled memtable (no memtable per write); realtime
        // GET sees each before the refresh checks it in.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for i in 0..200 {
            assert!(s.apply("i", "u", vec![idx(&format!("s{i}"), "t", WriteCond::default())], false)[0].is_ok());
            assert!(s.get(&format!("s{i}")).unwrap().is_some(), "realtime GET of a pending doc");
        }
        let visible = |s: &Shard| s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs() as u64).sum::<u64>();
        assert_eq!(visible(&s), 0, "pending until a refresh checks them in");
        s.refresh().unwrap();
        assert_eq!(visible(&s), 200);
        let leaves = s.core.read().leaves().len();
        assert!(leaves <= 2, "{leaves} leaves for 200 single-doc writes");
    }

    #[test]
    fn flushes_beside_in_flight_bulks_lose_nothing_across_a_crash() {
        // R1: a flush never waits for in-flight bulks; its commit point is
        // the capture's local checkpoint and the translog keeps every op
        // above it. Crash (no final flush) and reopen: every acked op is
        // present exactly once after the replay.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let acked = Arc::new(Mutex::new(Vec::<String>::new()));
        let writers: Vec<_> = (0..4).map(|thread| {
            let (s, stop, acked) = (s.clone(), stop.clone(), acked.clone());
            std::thread::spawn(move || {
                let mut round = 0;
                while !stop.load(Ordering::Acquire) {
                    let ids: Vec<String> = (0..500).map(|i| format!("w{thread}-{round}-{i}")).collect();
                    let ops = ids.iter().map(|id| idx(id, "a longer body to analyze so batches take a while", WriteCond::default())).collect();
                    let results = s.apply("i", "u", ops, false);
                    let mut acked = acked.lock();
                    for (id, r) in ids.into_iter().zip(results) { if r.is_ok() { acked.push(id); } }
                    round += 1;
                }
            })
        }).collect();
        let mut slowest = std::time::Duration::ZERO;
        for _ in 0..8 {
            std::thread::sleep(std::time::Duration::from_millis(30));
            let t0 = std::time::Instant::now();
            s.flush().unwrap();
            slowest = slowest.max(t0.elapsed());
        }
        stop.store(true, Ordering::Release);
        for w in writers { w.join().unwrap(); }
        let acked = acked.lock().clone();
        assert!(!acked.is_empty());
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        let committed = s.committed_seq_no();
        assert!(replay.iter().all(|op| op.seq_no() as i64 > committed), "replay starts above the commit point");
        for op in replay.iter().cloned() {
            if let TranslogOp::Index { id, source_json, seq_no, version, routing } = op {
                s.replay_index(id, source_json, doc("replayed"), vec![], routing, seq_no as i64, version).unwrap();
            }
        }
        s.refresh().unwrap();
        for id in &acked {
            assert_eq!(live_copies(&s, id), 1, "acked {id} lost or duplicated");
        }
        assert!(slowest < std::time::Duration::from_secs(5), "a flush took {slowest:?}");
    }

    #[test]
    fn refresh_shows_an_acked_write_whose_memtable_another_bulk_holds() {
        // R3: request A is acked with its document in a pooled memtable;
        // bulk B then takes that memtable. A refresh meanwhile freezes it at
        // B's next document boundary: A is visible right after, and the
        // refresh never waits for the rest of B.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("a", "acked first", WriteCond::default())], false)[0].is_ok());
        let bulk = {
            let s = s.clone();
            std::thread::spawn(move || {
                let t0 = std::time::Instant::now();
                let ops = (0..40_000).map(|i| idx(&format!("b{i}"), "a long body for the analyzer to chew on", WriteCond::default())).collect();
                assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
                t0.elapsed()
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(30));
        let t0 = std::time::Instant::now();
        s.refresh().unwrap();
        let refresh = t0.elapsed();
        assert!(s.get_mode("a", false).unwrap().is_some(), "the acked write is not visible after the refresh");
        let bulk = bulk.join().unwrap();
        assert!(refresh < bulk, "refresh ({refresh:?}) waited for the whole bulk ({bulk:?})");
    }

    /// Arm a pause of the next staged write between its sequencing and its
    /// inversion: returns (sequenced, release).
    fn hold_next_staged_write(s: &Shard) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (sequenced_tx, sequenced) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        s.pause_next_staged_write(move || {
            let _ = sequenced_tx.send(());
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
        });
        (sequenced, release)
    }

    #[test]
    fn flush_is_prompt_while_bulks_are_held_mid_flight() {
        // A flush never waits for in-flight bulks or a running merge: one
        // bulk is held between sequencing and inversion, a second has
        // inverted and waits for the write state (held elsewhere, as by a
        // long check-in), and a background merge is encoding.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("before", "b", WriteCond::default())], false)[0].is_ok());
        let prompt = std::time::Duration::from_secs(2);
        // 1. Held between sequencing and inversion (its claim holds the commit point).
        let (sequenced, release_held) = hold_next_staged_write(&s);
        let held = {
            let s = s.clone();
            std::thread::spawn(move || s.apply("i", "u", vec![idx("held", "h", WriteCond::default())], false))
        };
        sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        // 2. Sequenced, then (once the write state is taken by another
        // thread) inverted, blocked completing.
        let (sequenced2, release_blocked) = hold_next_staged_write(&s);
        let blocked = {
            let s = s.clone();
            std::thread::spawn(move || s.apply("i", "u", vec![idx("blocked", "k", WriteCond::default())], false))
        };
        sequenced2.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let (taken_tx, taken) = std::sync::mpsc::channel();
        let (free_tx, free) = std::sync::mpsc::channel::<()>();
        let holder = {
            let s = s.clone();
            std::thread::spawn(move || {
                let ws = s.write.lock();
                taken_tx.send(()).unwrap();
                let _ = free.recv_timeout(std::time::Duration::from_secs(30));
                drop(ws);
            })
        };
        taken.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        release_blocked.send(()).unwrap();
        let t0 = std::time::Instant::now();
        while s.holders.lock().is_empty() && t0.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!s.holders.lock().is_empty(), "the second bulk never took a pooled memtable");
        std::thread::sleep(std::time::Duration::from_millis(100));
        // A freeze is handed the blocked bulk's memtable at once.
        let t0 = std::time::Instant::now();
        assert!(s.freeze_pool(FREEZE_WAIT), "a holder waiting for the write state did not hand off");
        assert!(t0.elapsed() < prompt, "freeze waited {:?} for a completing bulk", t0.elapsed());
        assert!(s.frozen.lock().iter().any(|p| !p.docs.is_empty()), "the handed-off memtable holds the blocked bulk's document");
        free_tx.send(()).unwrap();
        holder.join().unwrap();
        assert!(blocked.join().unwrap()[0].is_ok());
        // 3. A background merge encoding (holding its admission) does not delay the flush.
        let merging = s.bg.merge_running.acquire();
        let t0 = std::time::Instant::now();
        s.flush().unwrap();
        let flush = t0.elapsed();
        drop(merging);
        assert!(flush < prompt, "flush took {flush:?} beside a held bulk and a running merge");
        assert!(s.committed_seq_no() >= 0);
        release_held.send(()).unwrap();
        assert!(held.join().unwrap()[0].is_ok());
        s.refresh().unwrap();
        for id in ["before", "held", "blocked"] {
            assert!(s.get_mode(id, false).unwrap().is_some(), "[{id}] lost");
        }
    }

    #[test]
    fn bulks_complete_promptly_during_a_slow_flush() {
        // Nothing a bulk needs is held while a flush seals and commits: a
        // flush paused after its capture point leaves bulks (new ids and
        // updates), refresh and realtime GET running.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        for i in 0..200 { assert!(s.apply("i", "u", vec![idx(&format!("d{i}"), "v1", WriteCond::default())], false)[0].is_ok()); }
        s.refresh().unwrap();
        let (paused_tx, paused) = std::sync::mpsc::channel();
        let (resume, resume_rx) = std::sync::mpsc::channel::<()>();
        s.pause_next_flush_commit(move || {
            let _ = paused_tx.send(());
            let _ = resume_rx.recv_timeout(std::time::Duration::from_secs(30));
        });
        let flush = { let s = s.clone(); std::thread::spawn(move || s.flush()) };
        paused.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let t0 = std::time::Instant::now();
        let ops = (0..400).map(|i| idx(&format!("d{i}"), "v2", WriteCond::default())).collect();
        assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
        s.refresh().unwrap();
        assert!(s.get_mode("d399", true).unwrap().is_some());
        let elapsed = t0.elapsed();
        resume.send(()).unwrap();
        flush.join().unwrap().unwrap();
        assert!(elapsed < std::time::Duration::from_secs(2), "bulk + refresh took {elapsed:?} behind a paused flush");
        s.refresh().unwrap();
        for i in 0..400 { assert_eq!(live_copies(&s, &format!("d{i}")), 1, "d{i}"); }
    }

    /// Live copies of `id` as searches see them (the published view).
    fn visible_copies(s: &Shard, id: &str) -> usize {
        let core = s.core.read();
        let leaves = core.leaves();
        let term = Term::new_text(ID_FIELD, id);
        leaves.iter().map(|leaf| {
            let mut n = 0;
            if let Some(mut postings) = leaf.postings(&term) {
                while let Some(d) = postings.next() {
                    if !leaf.is_deleted(d) { n += 1; }
                }
            }
            n
        }).sum()
    }

    #[test]
    fn concurrent_updates_keep_exactly_one_visible_copy_at_every_refresh() {
        // Older-copy deletes are buffered at check-in and resolved outside
        // the write state; a refresh publishes each new copy together with
        // its predecessor's delete, so no refresh shows two copies or none.
        const IDS: usize = 300;
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let ops = (0..IDS).map(|i| idx(&format!("u{i}"), "v0", WriteCond::default())).collect();
        assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
        s.refresh().unwrap();
        s.flush().unwrap();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writers: Vec<_> = (0..4).map(|w| {
            let (s, stop) = (s.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut round = 0usize;
                while !stop.load(Ordering::Acquire) {
                    let ops = (0..40).map(|k| idx(&format!("u{}", (round * 37 + k * 7 + w * 11) % IDS), "upd", WriteCond::default())).collect();
                    for r in s.apply("i", "u", ops, false) { r.unwrap(); }
                    round += 1;
                }
            })
        }).collect();
        for _ in 0..30 {
            s.refresh().unwrap();
            for i in 0..IDS {
                assert_eq!(visible_copies(&s, &format!("u{i}")), 1, "u{i} after a refresh");
            }
            assert_eq!(s.reader_view().num_live_docs(), IDS as u64);
            // Realtime reads see one (the newest) copy meanwhile.
            assert!(s.get_mode("u7", true).unwrap().is_some());
        }
        stop.store(true, Ordering::Release);
        for w in writers { w.join().unwrap(); }
        s.refresh().unwrap();
        for i in 0..IDS { assert_eq!(visible_copies(&s, &format!("u{i}")), 1); }
        s.flush().unwrap();
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        for op in replay.iter().cloned() {
            match op {
                TranslogOp::Index { id, source_json, seq_no, version, routing } =>
                    s.replay_index(id, source_json, doc("replayed"), vec![], routing, seq_no as i64, version).unwrap(),
                TranslogOp::Delete { id, seq_no, version } => s.replay_delete(&id, seq_no as i64, version).unwrap(),
            }
        }
        s.refresh().unwrap();
        for i in 0..IDS { assert_eq!(live_copies(&s, &format!("u{i}")), 1, "u{i} after reopen"); }
    }

    #[test]
    fn checkin_does_not_look_up_older_copies() {
        // The check-in of updates records their older-copy deletes and
        // looks nothing up: its cost does not grow with the index.
        let checkin_time = |existing: usize| {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            for chunk in 0..(existing / 1000).max(1) {
                let ops = (chunk * 1000..((chunk + 1) * 1000).min(existing.max(1000))).map(|i| idx(&format!("e{i}"), "old", WriteCond::default())).collect();
                assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
                s.refresh().unwrap();
                s.flush().unwrap();
            }
            let ops = (0..1000).map(|i| idx(&format!("e{i}"), "new", WriteCond::default())).collect();
            assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
            let mut writers = std::mem::take(&mut *s.pool.lock());
            writers.extend(std::mem::take(&mut *s.frozen.lock()));
            let t0 = std::time::Instant::now();
            assert!(s.checkin_taken(writers, None));
            let took = t0.elapsed();
            assert_eq!(s.unresolved.lock().len(), 1000, "every update's older copy is buffered");
            assert_eq!(live_copies(&s, "e5"), 2, "nothing resolved at check-in");
            s.refresh().unwrap();
            assert_eq!(live_copies(&s, "e5"), 1);
            assert_eq!(visible_copies(&s, "e5"), 1);
            took
        };
        let small = checkin_time(1000);
        let large = checkin_time(20_000);
        assert!(large < small * 5 + std::time::Duration::from_millis(50), "check-in grew with the index: {small:?} -> {large:?}");
    }

    #[test]
    fn pooled_auto_id_documents_can_be_read_deleted_and_overwritten_before_refresh() {
        // B1: an auto-id document still in a pooled memtable is found by a
        // realtime GET, a delete of its id wins, and an overwrite leaves one copy.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let auto = |id: &str| WriteOp::Index { id: id.into(), source: "{\"title\":\"a\"}".into(), doc: doc("a"),
            ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true };
        assert!(s.apply("i", "u", vec![auto("a1")], false)[0].is_ok());
        assert!(s.apply("i", "u", vec![auto("a2")], false)[0].is_ok());
        assert!(s.get_mode("a1", true).unwrap().is_some(), "realtime GET of a pooled auto-id document");
        let deleted = s.apply("i", "u", vec![WriteOp::Delete { id: "a1".into(), cond: WriteCond::default() }], false).remove(0).unwrap();
        assert_eq!(deleted.result, "deleted");
        let overwritten = s.apply("i", "u", vec![idx("a2", "b", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!(overwritten.version, 2);
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "a1"), 0, "the acked delete lost to the pooled auto-id document");
        assert!(s.get_mode("a1", true).unwrap().is_none());
        assert_eq!(live_copies(&s, "a2"), 1, "overwrite left two copies");
        assert_eq!(visible_copies(&s, "a2"), 1);
    }

    #[test]
    fn a_request_claim_covers_every_memtable_it_wrote_into() {
        // B2: a request hands its first memtable off to a freeze mid-way and
        // completes in a second one; checking only the second in must not
        // release the claim (version map pruning, commit point) while the
        // first one's documents are not in the index.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        { let mut ws = s.write.lock(); ws.seq_no = 6; ws.claim(5, Shard::prepare_claim_node().unwrap()); }
        let (mut holding, mut pooled) = s.hold_pool_writer(Some(5)).unwrap();
        pooled.docs.push(PoolDoc { id: registry::DeleteKey::prepare("x").unwrap(), seq_no: 5, local: 0, older: false });
        s.freeze_gen.fetch_add(1, Ordering::AcqRel);
        s.observe_freeze(&mut holding, &mut pooled);
        assert_eq!(s.frozen.lock().len(), 1, "the first memtable was handed off");
        pooled.docs.push(PoolDoc { id: registry::DeleteKey::prepare("y").unwrap(), seq_no: 6, local: 0, older: false });
        pooled.registered.push(holding.claim.take().unwrap());
        s.release_holding(holding);
        {
            let mut ws = s.write.lock();
            let mut core = s.core.write();
            s.checkin_pool_writer(&mut ws, &mut core, pooled);
        }
        assert_eq!(s.write.lock().capture_floor().0, 4, "the claim was released with a memtable of the request still out");
        let frozen = std::mem::take(&mut *s.frozen.lock());
        {
            let mut ws = s.write.lock();
            let mut core = s.core.write();
            for writer in frozen { s.checkin_pool_writer(&mut ws, &mut core, writer); }
        }
        assert_eq!(s.write.lock().capture_floor(), (6, None), "released once every memtable is in");
    }

    #[test]
    fn a_failed_translog_append_installs_nothing_and_a_retry_is_sequenced_afresh() {
        // B3: ops whose translog append failed are not indexed, not
        // realtime-visible and leave the version map as it was; the retry
        // is a fresh op (not acknowledged from the failed one).
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        assert!(s.apply("i", "u", vec![idx("y", "v1", WriteCond::default())], false)[0].is_ok());
        s.fail_next_translog_appends(1);
        let failed = s.apply("i", "u", vec![idx("x", "v1", WriteCond::default()), idx("y", "v2", WriteCond::default())], false);
        assert!(failed.iter().all(Result::is_err), "{failed:?}");
        assert!(s.get_mode("x", true).unwrap().is_none(), "an op that never reached the translog is readable");
        assert_eq!(s.get_mode("y", true).unwrap().unwrap().version, 1);
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "x"), 0, "an op that never reached the translog was indexed");
        // The engine failed itself (translog tragic event): a retry fails
        // fast and is never acknowledged; the copy is reported failed once.
        let retry = s.apply("i", "u", vec![idx("x", "v1", WriteCond::default())], false).remove(0);
        assert!(retry.as_ref().is_err_and(|e| e.to_string().contains("translog")), "{retry:?}");
        assert!(s.take_tragic_report().is_some());
        assert!(s.take_tragic_report().is_none(), "reported once");
        assert_eq!(s.max_seq_no(), 2, "a refused op takes no seq_no");
        s.flush().unwrap();
        drop(s);
        // Reallocated: the reopened copy sequences the retry afresh.
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.iter().all(|op| !matches!(op, TranslogOp::Index { id, .. } if id == "x")), "the failed op is not in the translog");
        let x = s.apply("i", "u", vec![idx("x", "v1", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!((x.version, x.result), (1, "created"));
        assert!(x.seq_no > 2, "the failed ops' seq_nos are not reused");
        let y = s.apply("i", "u", vec![idx("y", "v2", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!((y.version, y.result), (2, "updated"));
    }

    #[test]
    fn a_memtable_returned_during_a_refresh_joins_that_refresh() {
        // B4: request A is acknowledged with its document in a pooled
        // memtable that request H then takes; H completes after a refresh
        // started (past its last document boundary): the memtable must join
        // that refresh, not go back to the pool.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("a", "acked", WriteCond::default())], false)[0].is_ok());
        let (holding, pooled) = s.hold_pool_writer(None).unwrap();
        assert_eq!(pooled.docs.len(), 1, "H holds the memtable with A's document");
        let seen = holding.seen;
        let refresh = { let s = s.clone(); std::thread::spawn(move || s.refresh()) };
        let t0 = std::time::Instant::now();
        while s.freeze_gen.load(Ordering::Acquire) == seen && t0.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        s.complete_pooled(holding, pooled, Vec::new(), &[], &[], false).unwrap();
        refresh.join().unwrap().unwrap();
        assert_eq!(visible_copies(&s, "a"), 1, "the refresh returned without an acknowledged write");
    }

    #[test]
    fn a_sorted_refresh_publishes_an_update_with_its_new_copy() {
        // B5: an update checked in (and its older copy's delete resolved)
        // while a sorted refresh encodes is not in that refresh: the refresh
        // must not publish the older copy's delete without the new copy.
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
            pause: std::sync::atomic::AtomicBool::new(false), entered, resume: parking_lot::Mutex::new(release) });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
        core.set_nrt(true);
        core.set_auto_seal(false);
        core.set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
            field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: true, max_mode: false, missing_first: false,
        }])));
        core.adopt_view_slot(s.views.clone()).unwrap();
        *s.core.write() = core;
        assert!(s.apply("i", "u", vec![idx("u", "v1", WriteCond::default())], false)[0].is_ok());
        s.refresh().unwrap();
        assert_eq!(visible_copies(&s, "u"), 1);
        assert!(s.apply("i", "u", vec![idx("other", "o", WriteCond::default())], false)[0].is_ok());
        dir.pause.store(true, Ordering::Release);
        let refresh = { let s = s.clone(); std::thread::spawn(move || s.refresh()) };
        wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        // While the sorted refresh encodes: update `u`, check it in, resolve.
        let updated = s.apply("i", "u", vec![idx("u", "v2", WriteCond::default())], false);
        s.checkin_pool(None);
        resume.send(()).unwrap();
        refresh.join().unwrap().unwrap();
        assert!(updated[0].is_ok());
        assert_eq!(visible_copies(&s, "u"), 1, "the update vanished from search: its older copy's delete was published alone");
        s.refresh().unwrap();
        assert_eq!(visible_copies(&s, "u"), 1);
        assert_eq!(s.get_mode("u", false).unwrap().unwrap().version, 2);
    }

    #[test]
    fn flush_does_not_wait_for_a_background_seal_encoding() {
        // A background seal paused mid-encode: a flush commits what is
        // sealed, with its commit point below the memtables still encoding
        // (they replay from the translog), and returns promptly; the next
        // flush covers everything.
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
            pause: std::sync::atomic::AtomicBool::new(false), entered, resume: parking_lot::Mutex::new(release) });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
        core.set_nrt(true);
        core.set_auto_seal(false);
        core.adopt_view_slot(s.views.clone()).unwrap();
        *s.core.write() = core;
        let a = s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        s.checkin_pool(None);
        // A background seal of the memtable holding `a`, paused in its encoding.
        let task = {
            let running = s.bg.seal_running.acquire();
            let job = s.core.write().begin_seal(true).unwrap().expect("a memtable to seal");
            SealTask { _guard: running, job, forced: false, concurrent: 1 }
        };
        dir.pause.store(true, Ordering::Release);
        let encoding = std::thread::spawn(move || task.run());
        wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let b = s.apply("i", "u", vec![idx("b", "y", WriteCond::default())], false).remove(0).unwrap();
        let t0 = std::time::Instant::now();
        let flushed = s.flush();
        let took = t0.elapsed();
        let committed = s.committed_seq_no();
        resume.send(()).unwrap();
        let done = encoding.join().unwrap();
        flushed.unwrap();
        assert!(took < std::time::Duration::from_secs(2), "flush waited {took:?} for a background seal's encoding");
        assert!(committed < a.seq_no, "the commit point ({committed}) is above a document still encoding ({})", a.seq_no);
        s.finish_seal(done);
        s.flush().unwrap();
        assert_eq!(s.committed_seq_no(), b.seq_no);
        s.refresh().unwrap();
        assert!(s.get_mode("a", false).unwrap().is_some() && s.get_mode("b", false).unwrap().is_some());
    }

    #[test]
    fn realtime_get_keeps_acked_pending_document_during_next_version_reservation() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let first = s.apply("i", "u", vec![idx("x", "core-old", WriteCond::default())], false).remove(0).unwrap();
        s.refresh().unwrap();
        let acked = s.apply("i", "u", vec![idx("x", "acked-pending", WriteCond::default())], false).remove(0).unwrap();
        // Recreate the precise stage-2 -> stage-2d append window without
        // sleeping or relying on a particular disk latency: version metadata
        // is reserved, but the new WAL entry/pending document is not published.
        let previous = {
            let mut ws = s.write.lock();
            assert_eq!(ws.pending_docs["x"].seq_no, acked.seq_no);
            let previous = ws.versions["x"];
            ws.versions.replace_existing("x", VersionEntry {
                seq_no: previous.seq_no + 1, version: previous.version + 1, ..previous
            });
            previous
        };
        assert_eq!(Shard::lookup(&s.core.read(), "x").unwrap().2, first.seq_no,
            "fixture must retain the obsolete core document");
        let got = s.get("x").unwrap().unwrap();
        assert_eq!((got.seq_no, got.version), (acked.seq_no, acked.version));
        assert!(got.source.as_deref().unwrap().contains("acked-pending"));
        // An append failure restores the reservation, leaving the same
        // acknowledged source visible. A later real append publishes normally.
        s.write.lock().versions.replace_existing("x", previous);
        assert_eq!(s.get("x").unwrap().unwrap().seq_no, acked.seq_no);
        let next = s.apply("i", "u", vec![idx("x", "next-durable", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!(s.get("x").unwrap().unwrap().seq_no, next.seq_no);
        s.refresh().unwrap();
        assert_eq!(s.get("x").unwrap().unwrap().seq_no, next.seq_no);
    }

    #[test]
    fn realtime_get_reads_an_op_in_flight_from_the_version_map() {
        // R2: between sequencing and inversion an op is served by realtime
        // GET (ES LiveVersionMap), so a read-modify-write sees the newest
        // version and its if_seq_no does not conflict spuriously.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("x", "v1", WriteCond::default()), idx("gone", "v1", WriteCond::default())], false).iter().all(Result::is_ok));
        s.refresh().unwrap();
        let (sequenced, release) = hold_next_staged_write(&s);
        let held = {
            let s = s.clone();
            std::thread::spawn(move || s.apply("i", "u", vec![idx("x", "v2", WriteCond::default()),
                WriteOp::Delete { id: "gone".into(), cond: WriteCond::default() }], false))
        };
        sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let got = s.get_mode("x", true).unwrap().expect("x");
        let gone = s.get_mode("gone", true).unwrap();
        release.send(()).unwrap();
        let held = held.join().unwrap();
        let v2 = held[0].as_ref().unwrap();
        assert_eq!((got.version, got.seq_no), (v2.version, v2.seq_no), "realtime GET returned the copy the in-flight op replaces");
        assert!(gone.is_none(), "realtime GET returned a document whose delete was in flight");
        let cas = s.apply("i", "u", vec![idx("x", "v3", WriteCond { if_seq_no: Some(got.seq_no), if_primary_term: Some(got.primary_term), ..Default::default() })], false).remove(0);
        assert!(cas.is_ok(), "{cas:?}");
    }

    #[test]
    fn pending_document_metadata_uses_presequenced_field_capacity() {
        let mut document = doc("body");
        let source = "{\"title\":\"body\"}".to_owned();
        let routing = "route".to_owned();
        let ignored = vec!["ignored_a".to_owned(), "ignored_b".to_owned()];
        PendingDoc::prepare_document(&mut document, &"id".to_owned(), &source, Some(&routing), &ignored).unwrap();
        let capacity = document.fields.capacity();
        document.fields.push((ID_FIELD, FieldValue::Keyword("id".into())));
        document.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
        document.fields.push((VERSION_FIELD, FieldValue::I64(1)));
        document.fields.push((SEQ_NO_FIELD, FieldValue::I64(0)));
        document.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(1)));
        document.fields.push((ROUTING_FIELD, FieldValue::Keyword(routing)));
        for name in ignored { document.fields.push((IGNORED_FIELD, FieldValue::Keyword(name))); }
        assert_eq!(document.fields.capacity(), capacity, "post-sequencing metadata must not grow the field array");
        assert_eq!(document.fields.len(), 9);
    }

    #[test]
    fn pending_replica_identifier_retains_input_capacity_admission() {
        let owner = rustic_core::admission::Owner::new();
        let _scope = owner.enter();
        let mut identifier = String::with_capacity(65_536);
        identifier.push_str("id");
        let mut document = doc("body");
        PendingDoc::prepare_document(&mut document, &identifier, &String::new(), None, &[]).unwrap();
        assert!(owner.bytes().current >= identifier.capacity() as u64, "moved replica key must retain its whole allocation");
        drop(document);
        assert_eq!(owner.bytes().current, 0);
    }

    #[test]
    fn pending_map_claims_cover_concurrent_uninstalled_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "pending-map", schema()).unwrap();
        let first = PendingMapSlots::new(&s, 17).unwrap();
        let second = PendingMapSlots::new(&s, 31).unwrap();
        {
            let ws = s.write.lock();
            assert_eq!(ws.pending_inflight, 48);
            assert!(ws.pending_docs.capacity() >= 48);
            assert!(ws.pending_memory.as_ref().unwrap().bytes() > 0);
        }
        assert!(PendingMapSlots::new(&s, usize::MAX).is_err());
        assert_eq!(s.write.lock().pending_inflight, 48, "refusal must not install a claim");
        drop(first);
        assert_eq!(s.write.lock().pending_inflight, 31);
        drop(second);
        let ws = s.write.lock();
        assert_eq!(ws.pending_inflight, 0);
        assert!(ws.pending_memory.as_ref().unwrap().bytes() > 0, "empty retained table still owns its allocation");
    }

    #[test]
    fn prepared_pool_keeps_acknowledged_writer_freezable_before_inversion() {
        let tmp=tempfile::tempdir().unwrap();
        let s=Shard::create(tmp.path(),"prepared-freeze",schema()).unwrap();
        assert!(s.apply("prepared-freeze","u",vec![idx("acked","before",WriteCond::default())],false)[0].is_ok());
        let prepared=PreparedPool::new(&s,1).unwrap();
        assert!(s.holders.lock().is_empty(),"pre-WAL preparation is not active inversion");
        assert!(s.pool.lock().iter().any(|p|!p.docs.is_empty()),"older acknowledged writer stays visible to freeze");
        s.refresh().unwrap();
        assert_eq!(visible_copies(&s,"acked"),1);
        // The prepared idle writer was consumed by refresh. Activation must
        // use its pre-admitted empty fallback, without losing a holder.
        let (inverting,_credits)=prepared.into_inverting(None,Vec::new());
        assert_eq!(s.holders.lock().len(),1);
        let (holding,writer)=inverting.finish();
        assert!(writer.docs.is_empty());
        drop(writer);s.release_holding(holding);
        assert!(s.holders.lock().is_empty());
    }

    #[test]
    fn prepared_pool_reuses_still_eligible_idle_writer() {
        let tmp=tempfile::tempdir().unwrap();
        let s=Shard::create(tmp.path(),"prepared-reuse",schema()).unwrap();
        assert!(s.apply("prepared-reuse","u",vec![idx("acked","before",WriteCond::default())],false)[0].is_ok());
        let prepared=PreparedPool::new(&s,1).unwrap();
        let (inverting,_credits)=prepared.into_inverting(None,Vec::new());
        let (holding,writer)=inverting.finish();
        assert!(!writer.docs.is_empty(),"eligible pooled documents must be reused");
        s.pool.lock().push_back(writer);s.release_holding(holding);
        s.refresh().unwrap();assert_eq!(visible_copies(&s,"acked"),1);
    }

    /// Concurrent requests: each prepares the newest pooled memtables, so
    /// the second to invert reuses the next one instead of a fresh memtable
    /// (which would join the pool: dozens of small memtables per refresh).
    #[test]
    fn concurrent_prepared_requests_reuse_older_pooled_writers() {
        let tmp=tempfile::tempdir().unwrap();
        let s=Shard::create(tmp.path(),"prepared-concurrent",schema()).unwrap();
        assert!(s.apply("prepared-concurrent","u",vec![idx("a","first",WriteCond::default())],false)[0].is_ok());
        // Two pooled memtables: the acknowledged one, and a fresh one a
        // racing request inverted into.
        let (p1,p2)=(PreparedPool::new(&s,1).unwrap(),PreparedPool::new(&s,1).unwrap());
        let (h1,w1)=p1.into_inverting(None,Vec::new()).0.finish();
        let (h2,w2)=p2.into_inverting(None,Vec::new()).0.finish();
        assert!(!w1.docs.is_empty() && w2.docs.is_empty());
        s.pool.lock().push_back(w1);s.pool.lock().push_back(w2);
        s.release_holding(h1);s.release_holding(h2);
        // Two more racing requests: both prepared before either inverts.
        let (p3,p4)=(PreparedPool::new(&s,1).unwrap(),PreparedPool::new(&s,1).unwrap());
        let (h3,w3)=p3.into_inverting(None,Vec::new()).0.finish();
        let (h4,w4)=p4.into_inverting(None,Vec::new()).0.finish();
        assert!(w3.docs.is_empty(),"the newest pooled memtable first");
        assert!(!w4.docs.is_empty(),"the older pooled memtable, prepared too, not a fresh one");
        assert!(s.pool.lock().is_empty());
        s.pool.lock().push_back(w3);s.pool.lock().push_back(w4);
        s.release_holding(h3);s.release_holding(h4);
        s.refresh().unwrap();assert_eq!(visible_copies(&s,"a"),1);
    }

    #[test]
    fn bitmap_preflight_refusal_releases_holder_without_losing_acknowledged_pool() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "bitmap-refusal", schema()).unwrap();
        let ack = s.apply("bitmap-refusal", "u", vec![idx("a", "acked", WriteCond::default())], false)
            .remove(0).unwrap();
        assert!(s.pool.lock().iter().any(|p| !p.docs.is_empty()), "fixture must retain an acknowledged pooled document");
        let before = (s.max_seq_no(), s.local_checkpoint());
        // A real capacity-overflow refusal during preparation, without changing
        // the process-global admission limit used by other tests.
        let error = PreparedPool::new(&s, usize::MAX).err().expect("bitmap admission must refuse overflow");
        assert_eq!(error.status, 429);
        assert!(s.holders.lock().is_empty(), "refused request must release its refresh holder");
        assert_eq!((s.max_seq_no(), s.local_checkpoint()), before);
        assert!(s.tragic_error().is_none(), "pre-WAL refusal must not fail the engine");
        assert_eq!(s.get("a").unwrap().unwrap().seq_no, ack.seq_no);
        s.refresh().unwrap();
        assert_eq!(visible_copies(&s, "a"), 1, "checkout refusal lost an acknowledged writer");
        let next = s.apply("bitmap-refusal", "u", vec![idx("b", "after", WriteCond::default())], false)
            .remove(0).unwrap();
        assert_eq!(next.seq_no, ack.seq_no + 1);
    }

    #[test]
    fn a_panic_mid_inversion_strands_no_memtable() {
        // R3: a request panics while inverting into a pooled memtable that
        // holds another request's acknowledged document: the memtable is
        // checked in by the next refresh (which does not wait for the dead
        // holder), no seq_no stays a hole, and the engine fails itself.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("a", "acked", WriteCond::default())], false)[0].is_ok());
        s.panic_next_inversion();
        let panicked = { let s = s.clone(); std::thread::spawn(move || s.apply("i", "u", vec![idx("b", "x", WriteCond::default())], false)) };
        assert!(panicked.join().is_err(), "the injected panic did not fire");
        assert!(s.holders.lock().is_empty(), "the dead request is still a holder");
        let t0 = std::time::Instant::now();
        s.refresh().unwrap();
        assert!(t0.elapsed() < std::time::Duration::from_secs(2), "refresh waited {:?} for a dead holder", t0.elapsed());
        assert_eq!(visible_copies(&s, "a"), 1, "another request's acknowledged document was lost");
        assert_eq!(s.local_checkpoint(), s.max_seq_no(), "the panicked op left a checkpoint hole");
        assert!(s.tragic_error().is_some(), "the engine did not fail itself");
    }

    #[test]
    fn writes_stall_on_published_memtables_waiting_for_a_seal() {
        // Refreshes move the memtables being written to the published
        // ones: the write stall must count those (and pooled ones), or a
        // slow seal lets them pile up without bound.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.threshold.store(100, Ordering::Relaxed);
        // Batches below the seal threshold, each published by a refresh.
        for round in 0..8 {
            let ops = (0..60).map(|i| idx(&format!("d{round}-{i}"), "x", WriteCond::default())).collect();
            assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
            s.refresh().unwrap();
        }
        let core = s.core.read();
        assert_eq!(core.num_writing_docs(), 0, "refreshes left nothing being written");
        assert!(core.writing_bytes() == 0 && core.num_unsealed_bytes() > 0, "published memtables count toward the stall (bytes)");
        drop(core);
        assert!(s.unsealed_bytes() >= s.core.read().num_unsealed_bytes());
        assert!(s.stall_limit_bytes() >= 32 << 20, "twice the shard's buffer share, at least 16mb");
    }

    #[test]
    fn sealing_a_long_backlog_of_published_memtables_needs_little_stack() {
        // Hundreds of published memtables sealed at once, from a thread
        // with a small stack: the seal must neither recurse nor nest one
        // encode job per memtable.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.bg.threshold.store(1_000_000, Ordering::Relaxed);
        for round in 0..300 {
            assert!(s.apply("i", "u", vec![idx(&format!("d{round}"), "x", WriteCond::default())], false)[0].is_ok());
            s.refresh().unwrap();
        }
        assert!(s.core.read().num_closed_memtables() >= 100, "fixture: a backlog of published memtables");
        let sealer = { let s = s.clone(); std::thread::Builder::new().stack_size(256 << 10).spawn(move || s.seal_all()).unwrap() };
        sealer.join().expect("the seal overflowed a small stack").unwrap();
        assert_eq!(s.core.read().num_closed_memtables(), 0);
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 300);
    }

    #[test]
    fn refresh_listeners_complete_from_the_refresh_without_a_waiting_thread() {
        // R6: a wait_for listener returns at once and is completed by the
        // refresh that makes the write visible; past the listener limit a
        // refresh is forced; an expired listener gets a forced refresh.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        assert!(s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false)[0].is_ok());
        let (tx, rx) = std::sync::mpsc::channel();
        let first = tx.clone();
        let t0 = std::time::Instant::now();
        s.add_refresh_listener(2, std::time::Duration::from_secs(30), Box::new(move |r| { first.send(("first", r.unwrap())).unwrap(); }));
        assert!(t0.elapsed() < std::time::Duration::from_millis(500), "registering a listener waited");
        assert!(rx.try_recv().is_err(), "completed before any refresh");
        assert!(s.get_mode("a", false).unwrap().is_none());
        s.refresh().unwrap();
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap(), ("first", false));
        assert!(s.get_mode("a", false).unwrap().is_some());
        // The limit: one waiting, a second registration forces a refresh.
        assert!(s.apply("i", "u", vec![idx("b", "x", WriteCond::default())], false)[0].is_ok());
        let waiting = tx.clone();
        s.add_refresh_listener(1, std::time::Duration::from_secs(30), Box::new(move |r| { waiting.send(("waiting", r.unwrap())).unwrap(); }));
        let over = tx.clone();
        s.add_refresh_listener(1, std::time::Duration::from_secs(30), Box::new(move |r| { over.send(("over", r.unwrap())).unwrap(); }));
        let mut got: Vec<_> = (0..2).map(|_| rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap()).collect();
        got.sort();
        assert_eq!(got, vec![("over", true), ("waiting", false)]);
        // Expired: the scheduler sees it and the forced refresh completes it.
        assert!(s.apply("i", "u", vec![idx("c", "x", WriteCond::default())], false)[0].is_ok());
        let expired = tx.clone();
        s.add_refresh_listener(1000, std::time::Duration::ZERO, Box::new(move |r| { expired.send(("expired", r.unwrap())).unwrap(); }));
        assert!(s.refresh_listeners_expired());
        s.refresh().unwrap();
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap(), ("expired", true));
    }

    /// The flush's own seal encodes with `seal_lock` free: a seal of
    /// writes made after its capture installs meanwhile, and the commit
    /// leaves that segment out (its documents, vector rows included, replay
    /// from the translog).
    fn flush_commits_without_segments_sealed_after_its_capture(fields: Vec<CoreField>, op: fn(&str) -> WriteOp) {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", fields.clone()).unwrap());
        for round in 0..3 {
            s.apply("i", "u", vec![op(&format!("d{round}"))], false).remove(0).unwrap();
            s.refresh().unwrap();
        }
        let during = s.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        s.pause_next_flush_commit(move || {
            assert!(during.bg.seal_lock.try_lock().is_some(), "the flush pins the segment set while it encodes");
            let late = during.apply("i", "u", vec![op("late")], false).remove(0).unwrap();
            during.refresh().unwrap();
            let job = during.core.write().begin_seal_part(false, usize::MAX).unwrap().expect("the late memtable is unclaimed");
            let sealed = job.run();
            during.core.write().finish_seal(sealed).unwrap();
            tx.send(late.seq_no).unwrap();
        });
        s.flush().unwrap();
        let late = rx.recv().unwrap();
        assert!(s.committed_seq_no() < late);
        assert_eq!(s.reader_view().num_live_docs(), 4);
        drop(s);
        let (reopened, replay) = Shard::open(tmp.path(), fields).unwrap();
        for round in 0..3 { assert!(reopened.get(&format!("d{round}")).unwrap().is_some()); }
        assert!(reopened.get("late").unwrap().is_none(), "a segment sealed after the capture is not in the commit");
        assert_eq!(replay.iter().map(|op| op.seq_no()).collect::<Vec<_>>(), vec![late as u64]);
        assert_eq!(reopened.reader_view().num_live_docs(), 3);
    }

    #[test]
    fn an_idle_seal_writes_the_buffer_out_without_publishing_it() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for i in 0..10 {
            s.apply("i", "u", vec![idx(&i.to_string(), "x", WriteCond::default())], false).remove(0).unwrap();
        }
        assert_eq!(s.reader_view().num_live_docs(), 0);
        s.bg.idle_seal.store(true, Ordering::Release);
        s.bg.want_seal.store(true, Ordering::Release);
        s.run_background();
        assert_eq!(s.core.read().num_writing_docs(), 0, "the buffer was written out");
        assert!(!s.core.read().hidden_segments().is_empty());
        assert_eq!(s.reader_view().num_live_docs(), 0, "visibility unchanged");
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 10);
        for i in 0..10 { assert!(s.get(&i.to_string()).unwrap().is_some()); }
    }

    #[test]
    fn a_flush_encodes_unpinned_and_commits_without_segments_sealed_after_its_capture() {
        flush_commits_without_segments_sealed_after_its_capture(schema(), |id| idx(id, "x", WriteCond::default()));
    }

    #[test]
    fn a_vector_flush_encodes_unpinned_and_commits_without_segments_sealed_after_its_capture() {
        let mut fields = schema();
        // An explicit segmented (flat) mapping: a field without one is
        // taken as a graph field, which the mutable backend accepts only
        // with explicit options.
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 }).with_similarity("l2_norm")
            .with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw().flat())));
        flush_commits_without_segments_sealed_after_its_capture(fields, |id| {
            let mut d = doc("vec");
            d.add_vector(9, vec![id.len() as f32, 1.]);
            WriteOp::Index { id: id.into(), source: "{}".into(), doc: d, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }
        });
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn a_mutable_vector_flush_encodes_unpinned_and_commits_without_segments_sealed_after_its_capture() {
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        flush_commits_without_segments_sealed_after_its_capture(fields, |id| {
            let mut d = doc("vec");
            d.add_vector(9, vec![id.len() as f32, 1.]);
            WriteOp::Index { id: id.into(), source: "{}".into(), doc: d, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }
        });
    }

    #[test]
    fn a_flush_beside_two_in_flight_seals_commits_below_their_documents() {
        // G1: two background seal jobs are in flight (claimed, not
        // installed); a flush seals the rest and commits with its point below
        // their documents; once they install, the next flush covers all.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let mut seqs = Vec::new();
        for round in 0..4 {
            let r = s.apply("i", "u", vec![idx(&format!("d{round}"), "x", WriteCond::default())], false).remove(0).unwrap();
            seqs.push(r.seq_no);
            s.refresh().unwrap();
        }
        let (first, second) = {
            let mut core = s.core.write();
            (core.begin_seal_part(false, 1).unwrap().unwrap(), core.begin_seal_part(false, 1).unwrap().unwrap())
        };
        let late = s.apply("i", "u", vec![idx("late", "x", WriteCond::default())], false).remove(0).unwrap();
        s.flush().unwrap();
        let lowest_in_flight = seqs[..2].iter().min().copied().unwrap();
        assert!(s.committed_seq_no() < lowest_in_flight, "commit point {} above a document still encoding ({lowest_in_flight})", s.committed_seq_no());
        let (first, second) = (first.run(), second.run());
        s.core.write().finish_seal(first).unwrap();
        s.core.write().finish_seal(second).unwrap();
        s.flush().unwrap();
        assert_eq!(s.committed_seq_no(), late.seq_no);
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 5);
    }

    #[test]
    fn seal_semaphore_admits_jobs_up_to_its_limit_and_one_exclusive_holder() {
        let sem = BgSemaphore::new(2);
        let a = sem.try_acquire().unwrap();
        let b = sem.try_acquire().unwrap();
        assert!(sem.try_acquire().is_none());
        assert!(sem.try_acquire_all().is_none());
        drop(a);
        drop(b);
        let all = sem.try_acquire_all().unwrap();
        assert!(sem.try_acquire().is_none(), "an exclusive holder keeps permits off");
        drop(all);
        assert!(sem.try_acquire().is_some());
    }

    #[test]
    fn writes_to_other_ids_proceed_while_one_id_is_locked() {
        // G2: an id whose per-id lock is held blocks only writes of that id
        // (same-id ops serialize); other ids sequence and complete.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let others: Vec<String> = (0..50).map(|i| format!("o{i}")).collect();
        let held = s.id_locks.lock_all(std::iter::once("x"));
        let blocked = { let s = s.clone(); std::thread::spawn(move || s.apply("i", "u", vec![idx("x", "v", WriteCond::default())], false)) };
        let t0 = std::time::Instant::now();
        let free: Vec<_> = others.chunks(10).map(|ids| {
            let (s, ids) = (s.clone(), ids.to_vec());
            std::thread::spawn(move || s.apply("i", "u", ids.iter().map(|id| idx(id, "v", WriteCond::default())).collect(), false))
        }).collect();
        for t in free { assert!(t.join().unwrap().iter().all(Result::is_ok)); }
        assert!(t0.elapsed() < std::time::Duration::from_secs(2), "other ids waited for the locked one");
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!blocked.is_finished(), "a write of the locked id did not wait for its lock");
        drop(held);
        assert!(blocked.join().unwrap()[0].is_ok());
    }

    /// M1: two bulks of 1,000 different ids each are inside their
    /// sequencing at the same time (striped locks made such bulks take
    /// nearly every stripe and run one after the other), and the per-id
    /// table empties once they are done.
    #[test]
    fn large_bulks_of_different_ids_sequence_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        // Each batch waits in its sequencing (ids locked) until both are in.
        let inside = Arc::new((Mutex::new(0usize), parking_lot::Condvar::new()));
        let overlapped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let probe: Arc<dyn Fn() + Send + Sync> = {
            let (inside, overlapped) = (inside.clone(), overlapped.clone());
            Arc::new(move || {
                let (count, cv) = &*inside;
                let mut n = count.lock();
                *n += 1;
                cv.notify_all();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while *n < 2 && std::time::Instant::now() < deadline {
                    cv.wait_until(&mut n, deadline);
                }
                if *n >= 2 { overlapped.store(true, Ordering::SeqCst); }
            })
        };
        s.probe_next_sequencing(Some(probe));
        let bulks: Vec<_> = ["a", "b"].into_iter().map(|prefix| {
            let s = s.clone();
            std::thread::spawn(move || s.apply("i", "u", (0..1000).map(|i| idx(&format!("{prefix}{i}"), "v", WriteCond::default())).collect(), false))
        }).collect();
        for b in bulks { assert!(b.join().unwrap().iter().all(Result::is_ok)); }
        s.probe_next_sequencing(None);
        assert!(overlapped.load(Ordering::SeqCst), "the two bulks never sequenced at the same time");
        assert_eq!(s.id_locks.len(), 0, "unlocked ids leave the table");
        // The same id still serializes.
        let held = s.id_locks.lock_all(std::iter::once("a1"));
        let blocked = { let s = s.clone(); std::thread::spawn(move || s.apply("i", "u", vec![idx("a1", "w", WriteCond::default())], false)) };
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!blocked.is_finished());
        drop(held);
        assert!(blocked.join().unwrap()[0].is_ok());
    }

    /// M2: `if_primary_term` is compared with the term of the id's latest
    /// op, also when the copy a batch's lookup found is an older one (the
    /// newest, of a later term, was checked in between that lookup and the
    /// version checks): no false conflict.
    #[test]
    fn cas_compares_with_the_latest_ops_term_not_an_older_copys() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.apply("i", "u", vec![idx("x", "v1", WriteCond::default())], false).remove(0).unwrap();
        s.refresh().unwrap();
        s.set_primary_term(2).unwrap();
        // The latest copy, of term 2, waits in a pooled memtable.
        let latest = s.apply("i", "u", vec![idx("x", "v2", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!(latest.primary_term, 2);
        // The CAS batch's lookup sees the term-1 copy; the term-2 one is
        // checked in right after it.
        let during = s.clone();
        s.probe_next_sequencing(Some(Arc::new(move || { during.refresh().unwrap(); })));
        let cas = s.apply("i", "u", vec![idx("x", "v3", WriteCond { if_seq_no: Some(latest.seq_no), if_primary_term: Some(2), ..Default::default() })], false).remove(0);
        s.probe_next_sequencing(None);
        assert!(cas.is_ok(), "{cas:?}");
    }

    /// M3: an acknowledged update whose older copy cannot be deleted at
    /// resolution fails the engine (no two live copies served, nothing more
    /// acknowledged), instead of being logged and forgotten.
    #[test]
    fn a_failed_older_copy_delete_fails_the_engine() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", vec![idx("x", "v1", WriteCond::default())], false).remove(0).unwrap();
        s.refresh().unwrap();
        s.apply("i", "u", vec![idx("x", "v2", WriteCond::default())], false).remove(0).unwrap();
        s.fail_next_older_deletes(1);
        let _ = s.refresh();
        assert!(s.tragic_error().is_some(), "the engine kept going with the older copy live");
        assert!(s.apply("i", "u", vec![idx("y", "v", WriteCond::default())], false).remove(0).is_err());
    }

    /// M5: `refresh=wait_for` listeners of a copy that closes fire at the
    /// close (ES `RefreshListeners.close`), and ones registered afterwards
    /// fire at once; dropping a copy fires them too.
    #[test]
    fn refresh_listeners_fire_when_the_copy_closes() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let waiting = tx.clone();
        s.add_refresh_listener(1000, std::time::Duration::from_secs(3600), Box::new(move |r| { waiting.send(r.unwrap()).unwrap(); }));
        assert!(rx.try_recv().is_err(), "the listener waits for a refresh");
        s.close_refresh_listeners();
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap(), false);
        let late = tx.clone();
        s.add_refresh_listener(1000, std::time::Duration::from_secs(3600), Box::new(move |r| { late.send(r.unwrap()).unwrap(); }));
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap(), false, "a listener on a closed copy fires at once");
        let tmp2 = tempfile::tempdir().unwrap();
        let dropped = Shard::create(tmp2.path(), "i", schema()).unwrap();
        dropped.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        dropped.add_refresh_listener(1000, std::time::Duration::from_secs(3600), Box::new(move |r| { tx.send(r.unwrap()).unwrap(); }));
        drop(dropped);
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap(), false);
    }

    /// M4: a commit is a point in time. A document updated while a flush
    /// is between its capture and its commit (the new copy is above the
    /// commit point) keeps its old copy live in that commit, read without
    /// the translog (snapshot, file copy); the translog holds the update.
    #[test]
    fn an_update_during_a_flush_window_keeps_the_old_copy_live_in_the_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let old = s.apply("i", "u", vec![idx("x", "v1", WriteCond::default())], false).remove(0).unwrap();
        s.refresh().unwrap();
        s.flush().unwrap();
        s.apply("i", "u", vec![idx("y", "v1", WriteCond::default())], false).remove(0).unwrap();
        let during = s.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        s.pause_next_flush_commit(move || {
            // Update x after the capture and make it visible: the refresh
            // deletes the committed old copy.
            let new = during.apply("i", "u", vec![idx("x", "v2", WriteCond::default())], false).remove(0).unwrap();
            during.refresh().unwrap();
            tx.send(new.seq_no).unwrap();
        });
        s.flush().unwrap();
        let new_seq_no = rx.recv().unwrap();
        assert!(s.committed_seq_no() < new_seq_no);
        drop(s);
        // The commit alone: the old copy of x is live, y is there.
        let committed = rustic_core::Index::open(&core_dir(tmp.path(), 0)).unwrap();
        let leaves = committed.all_leaves();
        let found = Shard::lookup_in(&leaves, "x", true).expect("the old copy of x is live in the commit");
        assert_eq!(found.2, old.seq_no);
        assert!(Shard::lookup_in(&leaves, "y", true).is_some());
        drop(leaves);
        drop(committed);
        // With the translog, the update replays.
        let (_reopened, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.iter().any(|op| op.seq_no() as i64 == new_seq_no));
    }

    /// M4 on an index-sorted core: the commit's capture-time deletions of a
    /// memtable the flush seals follow the sort's reordering of its
    /// documents (a delete before the capture stays on its document; an
    /// update after it leaves the old copy live in the commit).
    #[test]
    fn a_sorted_flush_commits_capture_deletions_on_the_sorted_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.core.write().set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
            field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: true, max_mode: false, missing_first: false,
        }])));
        // Written a, m, x: sorted x, m, a (reverse id order).
        for id in ["a", "m", "x"] {
            s.apply("i", "u", vec![idx(id, "v1", WriteCond::default())], false).remove(0).unwrap();
        }
        let old_x = s.get("x").unwrap().unwrap();
        s.apply("i", "u", vec![WriteOp::Delete { id: "a".into(), cond: WriteCond::default() }], false).remove(0).unwrap();
        let during = s.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        s.pause_next_flush_commit(move || {
            // After the capture (no refresh: a sorted refresh waits for the
            // flush).
            let new = during.apply("i", "u", vec![idx("x", "v2", WriteCond::default())], false).remove(0).unwrap();
            tx.send(new.seq_no).unwrap();
        });
        s.flush().unwrap();
        let new_seq_no = rx.recv().unwrap();
        assert!(s.committed_seq_no() < new_seq_no);
        drop(s);
        let committed = rustic_core::Index::open(&core_dir(tmp.path(), 0)).unwrap();
        let leaves = committed.all_leaves();
        assert!(Shard::lookup_in(&leaves, "a", true).is_none(), "the delete before the capture landed on another document");
        assert!(Shard::lookup_in(&leaves, "m", true).is_some());
        let x = Shard::lookup_in(&leaves, "x", true).expect("the old copy of x is live in the commit");
        assert_eq!(x.2, old_x.seq_no);
    }

    /// M8: pinning the files (snapshot, clone/split copies) does not wait
    /// for a merge or a seal that is still encoding (a throttled merge can
    /// run for hours): only installs and commits are held off.
    #[test]
    fn files_pinned_does_not_wait_for_encoding_merges_or_seals() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        // A merge and a seal in their encoding (their permits held).
        let merging = s.bg.merge_running.try_acquire().expect("a merge permit");
        let sealing = s.bg.seal_running.try_acquire().expect("a seal permit");
        let pinning = { let s = s.clone(); std::thread::spawn(move || { drop(s.files_pinned()); }) };
        let t0 = std::time::Instant::now();
        while !pinning.is_finished() && t0.elapsed() < std::time::Duration::from_secs(2) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let finished = pinning.is_finished();
        drop((merging, sealing));
        pinning.join().unwrap();
        assert!(finished, "files_pinned waited for an encoding merge or seal");
    }

    /// M15: files no commit refers to (a segment sealed after the last
    /// commit before a crash, an interrupted write's `.tmp`) are removed when
    /// the copy opens; committed files stay.
    #[test]
    fn opening_a_copy_sweeps_files_no_commit_refers_to() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
            s.flush().unwrap();
        }
        let core = core_dir(tmp.path(), 0);
        let committed: Vec<String> = std::fs::read_dir(&core).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        for orphan in ["seg_00009999.rstc", "seg_00009999.del", ".merge-store-1-2-3.tmp"] {
            std::fs::write(core.join(orphan), b"leftover").unwrap();
        }
        let (s, _) = Shard::open(tmp.path(), schema()).unwrap();
        for orphan in ["seg_00009999.rstc", "seg_00009999.del", ".merge-store-1-2-3.tmp"] {
            assert!(!core.join(orphan).exists(), "{orphan} was kept");
        }
        for name in committed {
            assert!(core.join(&name).exists(), "committed file {name} was removed");
        }
        assert!(s.get("a").unwrap().is_some());
    }

    /// R4: replicated ops never stall per engine call (a batch on the
    /// ordered path calls the engine once per op, holding the replica's
    /// state): the byte stall is one bounded wait per batch, taken before
    /// the replica locks anything, and a replica never rejects a batch for
    /// its buffer (a rejection fails the copy: churn).
    #[test]
    fn replicated_batches_wait_once_and_are_never_rejected_for_the_buffer() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        s.replay_index("a".into(), "{}".into(), doc("replica"), vec![], None, 0, 1).unwrap();
        // Every unsealed byte is over the limit now, and nothing seals.
        s.bg.stall_limit_override.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        for seq in 1..21 {
            s.replay_index(format!("r{seq}"), "{}".into(), doc("replica"), vec![], None, seq, 1).unwrap();
        }
        assert!(t0.elapsed() < REPLICA_STALL, "20 replicated ops stalled {:?}", t0.elapsed());
        let t0 = std::time::Instant::now();
        s.admit_replicated();
        assert!(t0.elapsed() < REPLICA_STALL + std::time::Duration::from_millis(500), "a batch stalled {:?}", t0.elapsed());
        assert!(s.bg.idle_seal.load(Ordering::Acquire) && s.bg.want_seal.load(Ordering::Acquire), "the stall asked for the buffer to be written out");
        // Far over the limit after its wait: still admitted.
        let before = crate::indexing_pressure::global().stats().replica_rejections;
        s.admit_replicated();
        assert_eq!(crate::indexing_pressure::global().stats().replica_rejections, before);
    }

    #[test]
    fn sustained_indexing_keeps_the_node_buffer_and_segment_counts_bounded() {
        let dirs: Vec<_> = (0..4).map(|_| tempfile::tempdir().unwrap()).collect();
        let copies: Vec<Shard> = dirs.iter().map(|d| {
            let s = Shard::create(d.path(), "i", schema()).unwrap();
            s.bg.served.store(true, Ordering::Release);
            s
        }).collect();
        let buffer: u64 = 1 << 20;
        let settle = |s: &Shard| { s.settle_background(); };
        let text = "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(4);
        for round in 0..120 {
            for (c, s) in copies.iter().enumerate() {
                // Copy 0 takes the most writes.
                let requests = if c == 0 { 3 } else { 1 + round % 2 };
                for request in 0..requests {
                    let ops = (0..50).map(|k| idx(&format!("{c}-{round}-{request}-{k}"), &format!("{text} {round} {k}"), WriteCond::default())).collect();
                    assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
                }
                s.refresh().unwrap();
            }
            let sizes: Vec<(u64, u64)> = copies.iter().map(|s| s.indexing_buffer_bytes().unwrap()).collect();
            for i in crate::refresh::buffers_to_write(&sizes, buffer).0 {
                copies[i].write_indexing_buffer();
            }
            for s in &copies { settle(s); }
            let total: u64 = copies.iter().map(|s| s.indexing_buffer_bytes().unwrap().0).sum();
            assert!(total <= buffer + CoreIndex::COALESCE_MAX_BYTES, "round {round}: the node holds {total} unsealed bytes");
        }
        for (c, s) in copies.iter().enumerate() {
            s.bg.want_merge.store(true, Ordering::Release);
            settle(s);
            let segments = s.core.read().num_segments();
            assert!(segments < 50, "copy {c} has {segments} segments");
            s.refresh().unwrap();
        }
        assert_eq!(copies[0].reader_view().num_live_docs(), 120 * 3 * 50);
    }

    /// An index-sorted copy's search-idle refresh makes its memtables
    /// searchable as they are (no seal on the search's path); the
    /// background seal then turns them into sorted segments. Every write is
    /// visible to the search that refreshed, realtime GET and deletes work
    /// throughout, and the sealed segments hold their docs in sort order.
    #[test]
    fn a_sorted_search_idle_refresh_publishes_memtables_and_seals_them_sorted_later() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.core.write().set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
            field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: false, max_mode: false, missing_first: false,
        }])));
        s.bg.served.store(true, Ordering::Release);
        let mut written = 0u64;
        let mut saw_memtables = false;
        for round in 0..3u64 {
            // Ids written out of sort order.
            for k in 0..40u64 {
                let id = format!("{:03}", (k * 37 + round * 11) % 997);
                s.apply("i", "u", vec![idx(&format!("{round}-{id}"), "v", WriteCond::default())], false).remove(0).unwrap();
                written += 1;
            }
            s.sched.idle.store(true, Ordering::SeqCst);
            s.sched.skipped.store(true, Ordering::SeqCst);
            s.before_search();
            assert_eq!(s.reader_view().num_live_docs(), written, "round {round}: the refreshing search sees every write");
            saw_memtables |= s.core.read().num_published_memtables() > 0;
            assert!(s.get(&format!("{round}-{:03}", (round * 11) % 997)).unwrap().is_some());
        }
        assert!(saw_memtables, "the refresh published memtables, unsealed");
        // A delete, refreshed the same way.
        s.apply("i", "u", vec![WriteOp::Delete { id: "1-011".into(), cond: WriteCond::default() }], false).remove(0).unwrap();
        s.sched.idle.store(true, Ordering::SeqCst);
        s.sched.skipped.store(true, Ordering::SeqCst);
        s.before_search();
        assert_eq!(s.reader_view().num_live_docs(), written - 1);
        // The background seal: sorted segments only, the same docs.
        s.bg.draining.store(true, Ordering::Release);
        s.bg.want_seal.store(true, Ordering::Release);
        s.settle_background();
        s.refresh().unwrap();
        let core = s.core.read();
        assert_eq!(core.num_published_memtables(), 0, "sealed in the background");
        for seg in core.segments() {
            let leaf = rustic_core::query::LeafReader::Segment(seg);
            let ids: Vec<String> = (0..seg.num_docs).filter_map(|d| leaf.keyword_value(ID_FIELD, d)).collect();
            assert!(ids.windows(2).all(|w| w[0] <= w[1]), "segment {} in sort order: {ids:?}", seg.id);
        }
        drop(core);
        assert_eq!(s.reader_view().num_live_docs(), written - 1);
        assert!(s.get("1-011").unwrap().is_none());
    }

    /// A refresh that would make a large buffer searchable (a search-idle
    /// copy's replicated backlog) seals it into coalesced segments first,
    /// as a Lucene refresh flushes its buffers: searches then read segments,
    /// never the memtables. The first search on the search-idle copy waits
    /// for that refresh and sees every document.
    #[test]
    fn a_large_publish_seals_the_backlog_into_segments_first() {
        let tmp = tempfile::tempdir().unwrap();
        // No scheduler serves this copy: the publish seals in place.
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for seq in 0..200i64 {
            s.replay_index(format!("r{seq}"), "{}".into(), doc("replicated backlog"), vec![], None, seq, 1).unwrap();
        }
        s.bg.large_publish_override.store(1, Ordering::Release);
        // Search-idle: its scheduled refreshes were skipped.
        s.sched.skipped.store(true, Ordering::Release);
        s.before_search();
        let core = s.core.read();
        assert_eq!(core.num_published_memtables(), 0, "the backlog was published as memtables");
        assert_eq!(core.num_unpublished_bytes(), 0);
        assert_eq!(core.num_segments(), 1, "one coalesced segment for the backlog");
        drop(core);
        assert_eq!(s.reader_view().num_live_docs(), 200);
    }

    /// ES IMC per copy on the live write path: past its share of the node's
    /// buffer a served copy's whole buffer is written out at once (not at
    /// the next node-wide check); past 1.5x its share its writes are
    /// throttled to one at a time until the seals catch up; a recovery
    /// replay is never throttled.
    #[test]
    fn a_copy_past_its_buffer_share_is_written_out_and_throttled_until_sealed() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        for row in 0..200 {
            s.apply("i", "u", vec![idx(&row.to_string(), "buffered text", WriteCond::default())], false).remove(0).unwrap();
        }
        let unsealed = s.unsealed_bytes();
        assert!(unsealed > 0);
        let reset = || { s.bg.idle_seal.store(false, Ordering::Release); s.bg.want_seal.store(false, Ordering::Release); };
        // Under its share (the stall limit is twice the share): nothing.
        reset();
        s.bg.stall_limit_override.store(unsealed * 4, Ordering::Release);
        assert!(s.merge_throttle().is_none());
        assert!(!s.bg.idle_seal.load(Ordering::Acquire), "written out under its share");
        // Past its share (0.8x the buffer), under 1.5x: written out, not throttled.
        reset();
        s.bg.stall_limit_override.store(unsealed * 8 / 5, Ordering::Release);
        assert!(s.merge_throttle().is_none(), "throttled under 1.5x its share");
        assert!(s.bg.idle_seal.load(Ordering::Acquire) && s.bg.want_seal.load(Ordering::Acquire), "not written out past its share");
        // Past 1.5x its share (0.5x the buffer): throttled, and written out.
        reset();
        s.bg.stall_limit_override.store(unsealed, Ordering::Release);
        assert!(s.merge_throttle().is_some(), "not throttled past 1.5x its share");
        assert!(s.bg.idle_seal.load(Ordering::Acquire));
        assert!(crate::memory::replaying("test recovery", || s.merge_throttle().is_none()), "a recovery replay was throttled");
        // The seals catch up: the throttle lifts.
        s.run_background();
        assert_eq!(s.unsealed_bytes(), 0);
        assert!(s.merge_throttle().is_none(), "still throttled once sealed");
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 200);
    }

    /// Sustained writes to a served copy: the buffer-driven write-outs keep
    /// its unsealed bytes near its share (they used to wait for the node's
    /// check, the seal thresholds or the write stall at twice the share).
    #[test]
    fn buffer_driven_write_outs_cap_a_served_copys_unsealed_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        let batch = |round: usize| (0..50).map(|k| idx(&format!("{round}-{k}"), "lorem ipsum dolor sit amet consectetur", WriteCond::default())).collect::<Vec<_>>();
        assert!(s.apply("i", "u", batch(0), false).iter().all(Result::is_ok));
        let b = s.unsealed_bytes();
        let share = 4 * b;
        s.bg.stall_limit_override.store(2 * share, Ordering::Release);
        let mut max = 0;
        for round in 1..60 {
            let t0 = std::time::Instant::now();
            assert!(s.apply("i", "u", batch(round), false).iter().all(Result::is_ok));
            assert!(t0.elapsed() < std::time::Duration::from_secs(10), "round {round}: the write stalled {:?}", t0.elapsed());
            // A background worker's pass (no scheduler serves this copy here).
            if s.background_due() { s.run_background(); }
            max = max.max(s.unsealed_bytes());
        }
        assert!(max <= share + 2 * b, "unsealed bytes reached {max} with a share of {share} (batches of {b})");
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 60 * 50);
    }

    /// The shard throttle serializes a throttled write's indexing, not its
    /// translog fsync: the group commit releases it, so throttled requests
    /// still share fsyncs; and a throttled synced write leaves it free.
    #[test]
    fn a_throttled_write_releases_the_shard_throttle_before_its_translog_fsync() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        let held = HeldThrottle::hold(Some(s.bg.indexing_throttle.lock()));
        assert!(s.bg.indexing_throttle.try_lock().is_none());
        let (file, mark) = s.write.lock().translog.sync_point();
        s.group_sync(&file, mark).unwrap();
        assert!(s.bg.indexing_throttle.try_lock().is_some(), "the fsync kept the shard throttle");
        drop(held);
        // A throttled copy (the node's buffer check): synced writes complete
        // and leave the throttle free.
        s.bg.stall_limit_override.store(u64::MAX / 4, Ordering::Release);
        s.bg.memory_throttled.store(true, Ordering::Release);
        assert!(s.merge_throttle().is_some());
        for row in 0..5 {
            s.apply("i", "u", vec![idx(&format!("t{row}"), "throttled", WriteCond::default())], true).remove(0).unwrap();
        }
        assert!(s.bg.indexing_throttle.try_lock().is_some());
        assert!(HELD_THROTTLE.with(|slot| slot.borrow().is_none()), "a throttle hold leaked past its write");
    }

    /// Wiki mixed load (c28j): right after a 5M-document preload every
    /// search of the idle phase took about 41 s, queued behind the
    /// search-idle refresh, which waited without bound. On a copy the
    /// scheduler serves, a search whose refresh finds the core held (a
    /// flush capture, a merge install) reads the last published view after
    /// a bounded wait, and hands the refresh to the background workers.
    #[test]
    fn a_search_on_a_served_search_idle_copy_never_waits_on_a_held_core() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.bg.served.store(true, Ordering::Release);
        for row in 0..20 {
            s.apply("i", "u", vec![idx(&format!("a{row}"), "published", WriteCond::default())], false).remove(0).unwrap();
        }
        s.refresh().unwrap();
        for row in 0..20 {
            s.apply("i", "u", vec![idx(&format!("b{row}"), "unpublished", WriteCond::default())], false).remove(0).unwrap();
        }
        s.sched.idle.store(true, Ordering::SeqCst);
        s.sched.skipped.store(true, Ordering::SeqCst);
        // Something holds the core for far longer than a search may wait.
        let held = s.core.write();
        let searcher = { let s = s.clone(); std::thread::spawn(move || { let t0 = std::time::Instant::now(); s.before_search(); t0.elapsed() }) };
        // A second search arriving meanwhile waits for the first's refresh
        // a bounded time too.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let second = { let s = s.clone(); std::thread::spawn(move || { let t0 = std::time::Instant::now(); s.before_search(); t0.elapsed() }) };
        let (first, second) = (searcher.join().unwrap(), second.join().unwrap());
        drop(held);
        assert!(first < IDLE_SEARCH_REFRESH_WAIT * 4 + std::time::Duration::from_secs(5), "the search waited {first:?} on the held core");
        assert!(second < IDLE_SEARCH_OTHER_REFRESH_WAIT + std::time::Duration::from_secs(5), "the second search waited {second:?}");
        assert!(s.bg.want_refresh.load(Ordering::Acquire), "the refresh was handed to the background workers");
        assert_eq!(s.reader_view().num_live_docs(), 20, "the searches read the last published view");
        // The background refresh publishes the rest.
        s.refresh_deferred();
        assert_eq!(s.reader_view().num_live_docs(), 40);
    }

    /// A search never waits on a refresh it did not start (the refresh API,
    /// a scheduled refresh): it reads the last published reader, even on a
    /// search-idle copy with writes it cannot see yet, and the skipped
    /// refresh stays due for the next search.
    #[test]
    fn a_search_reads_the_last_reader_while_another_refresh_is_under_way() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for row in 0..50 {
            s.apply("i", "u", vec![idx(&format!("a{row}"), "published", WriteCond::default())], false).remove(0).unwrap();
        }
        s.refresh().unwrap();
        for row in 0..50 {
            s.apply("i", "u", vec![idx(&format!("b{row}"), "unpublished", WriteCond::default())], false).remove(0).unwrap();
        }
        // Search-idle with unpublished writes: a search would refresh first.
        s.sched.idle.store(true, Ordering::SeqCst);
        let api_refresh = s.refreshing();
        let t0 = std::time::Instant::now();
        s.before_search();
        assert!(t0.elapsed() < std::time::Duration::from_secs(1), "the search waited {:?}", t0.elapsed());
        assert_eq!(s.reader_view().num_live_docs(), 50, "the search ran a refresh beside the one under way");
        drop(api_refresh);
        s.sched.idle.store(true, Ordering::SeqCst);
        s.before_search();
        assert_eq!(s.reader_view().num_live_docs(), 100, "the skipped refresh was not kept due");
    }

    /// The backlog cap: the memory bound until a seal rate is measured, then
    /// `REFRESH_SEAL_TARGET` of seal throughput, never below the floor nor
    /// above the memory bound.
    #[test]
    fn the_backlog_is_capped_at_seconds_of_measured_seal_work() {
        let gb = 1u64 << 30;
        let mb = 1u64 << 20;
        let secs = REFRESH_SEAL_TARGET.as_secs();
        assert_eq!(backlog_limit(6 * gb, 0), 6 * gb, "nothing measured: the memory bound");
        assert_eq!(backlog_limit(6 * gb, 100 * mb), 100 * mb * secs, "seconds of seal work");
        assert_eq!(backlog_limit(6 * gb, mb), BACKLOG_FLOOR_BYTES, "never below the floor");
        assert_eq!(backlog_limit(128 * mb, 100 * mb), 128 * mb, "never above the memory bound");
    }

    /// Seals measure the copy's throughput, and the stall limit follows it.
    #[test]
    fn a_background_seal_records_the_copys_seal_throughput() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        let text = "github archive event payload text ".repeat(12);
        for round in 0..40 {
            let ops = (0..100).map(|k| idx(&format!("{round}-{k}"), &format!("{text} {round} {k}"), WriteCond::default())).collect();
            assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
        }
        assert!(s.unsealed_bytes() >= 256 << 10, "too little buffered to measure: {}", s.unsealed_bytes());
        assert_eq!(s.bg.seal_rate.load(Ordering::Acquire), 0);
        s.write_indexing_buffer();
        let task = s.start_seal().expect("a background seal");
        s.finish_seal(task.run());
        let rate = s.bg.seal_rate.load(Ordering::Acquire);
        assert!(rate > 0, "no seal throughput recorded");
        // (The memory bound depends on the node-wide active copy count,
        // which parallel tests change; the cap itself is tested above.)
        assert!(s.stall_limit_bytes() >= BACKLOG_FLOOR_BYTES.min(16 << 21));
    }

    /// A served copy's large refresh writes its own segments: it never waits
    /// for the background seal workers (none run here; the refresh used to
    /// wait 30 s for them and then publish the backlog as memtables).
    #[test]
    fn a_served_large_refresh_seals_inline_without_waiting_for_background_seals() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        for row in 0..200 {
            s.apply("i", "u", vec![idx(&row.to_string(), "served backlog", WriteCond::default())], false).remove(0).unwrap();
        }
        s.bg.large_publish_override.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        s.refresh().unwrap();
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the refresh waited {:?}", t0.elapsed());
        let core = s.core.read();
        assert_eq!(core.num_published_memtables(), 0, "the backlog was published as memtables");
        assert_eq!(core.num_unpublished_bytes(), 0);
        assert!(core.num_segments() >= 1);
        assert!(!core.seal_in_flight());
        drop(core);
        assert_eq!(s.reader_view().num_live_docs(), 200);
    }

    /// Memtables a background seal already claimed are waited for at most
    /// `IN_FLIGHT_SEAL_WAIT`, then published as they are; the refresh seals the
    /// rest itself, and the background seal installs its share afterwards.
    #[test]
    fn a_served_refresh_waits_for_an_in_flight_background_seal_only_briefly() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.bg.served.store(true, Ordering::Release);
        for row in 0..100 {
            s.apply("i", "u", vec![idx(&format!("a{row}"), "claimed", WriteCond::default())], false).remove(0).unwrap();
        }
        // A background write-out claims the first 100, and never finishes
        // during the refresh.
        s.write_indexing_buffer();
        let task = s.start_seal().expect("a background seal");
        s.bg.seal_running.set_max(2);
        for row in 0..100 {
            s.apply("i", "u", vec![idx(&format!("b{row}"), "unclaimed", WriteCond::default())], false).remove(0).unwrap();
        }
        s.bg.large_publish_override.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        s.refresh().unwrap();
        assert!(t0.elapsed() < IN_FLIGHT_SEAL_WAIT + std::time::Duration::from_secs(5), "the refresh waited {:?}", t0.elapsed());
        assert_eq!(s.reader_view().num_live_docs(), 200, "the refresh published everything");
        s.finish_seal(task.run());
        let core = s.core.read();
        assert!(!core.seal_in_flight());
        assert_eq!(core.num_published_memtables(), 0, "both halves end up in segments");
        drop(core);
        assert_eq!(s.reader_view().num_live_docs(), 200);
    }

    /// ES flushes after a force merge whatever it merged: a copy already at
    /// one segment (a coalesced seal) is still committed.
    #[test]
    fn a_force_merge_with_nothing_to_merge_still_commits() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for id in ["a", "b", "c"] {
            s.apply("i", "u", vec![idx(id, "x", WriteCond::default())], false).remove(0).unwrap();
        }
        s.refresh().unwrap();
        s.seal_all().unwrap();
        assert_eq!(s.core.read().num_segments(), 1);
        assert!(s.committed_seq_no() < 2);
        s.force_merge(1, false).unwrap();
        assert_eq!(s.committed_seq_no(), 2, "the force merge did not commit");
    }

    /// Gap-fill no-ops are durable history: a failure to record them fails
    /// the engine and leaves the seq_nos unprocessed (the checkpoint never
    /// passes a hole nothing recorded); recorded ones are fsynced. No-ops
    /// recorded without an fsync (async durability) become durable with
    /// the translog's sync.
    #[test]
    fn gap_fill_noops_are_durable_or_fail_the_engine() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.replay_index("a".into(), "{}".into(), doc("x"), vec![], None, 0, 1).unwrap();
        s.replay_index("c".into(), "{}".into(), doc("x"), vec![], None, 2, 1).unwrap();
        assert_eq!(s.local_checkpoint(), 0);
        s.write.lock().hist.side.fail_next_appends(1);
        assert!(s.fill_gaps_to_max_seq_no(1).is_err());
        assert!(s.tragic_error().is_some(), "a lost gap-fill record left the engine healthy");
        assert_eq!(s.local_checkpoint(), 0, "a hole nothing recorded counts as processed");
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.replay_index("a".into(), "{}".into(), doc("x"), vec![], None, 0, 1).unwrap();
        s.replay_index("c".into(), "{}".into(), doc("x"), vec![], None, 2, 1).unwrap();
        assert_eq!(s.fill_gaps_to_max_seq_no(1).unwrap(), 1);
        assert_eq!(s.local_checkpoint(), 2);
        assert!(!s.write.lock().hist.side.has_unsynced(), "a gap fill was not fsynced");
        let _ = s.write.lock().fill_noops(&[3], 1, "async", false);
        assert!(s.write.lock().hist.side.has_unsynced());
        s.sync_translog().unwrap();
        assert!(!s.write.lock().hist.side.has_unsynced(), "the side log was left unsynced by the translog sync");
    }

    /// M9: a merge requested while every merge slot is busy is not lost:
    /// once a slot frees, the next background pass runs it.
    #[test]
    fn a_merge_request_survives_busy_merge_slots() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for i in 0..30 {
            s.apply("i", "u", vec![idx(&i.to_string(), "small", WriteCond::default())], false).remove(0).unwrap();
            s.refresh().unwrap();
            s.seal_all().unwrap();
        }
        let before = s.core.read().num_segments();
        assert!(before > merge_policy().max_segments_per_tier, "{before} segments need merging");
        let slots: Vec<_> = std::iter::from_fn(|| s.bg.merge_running.try_acquire()).collect();
        s.bg.want_merge.store(true, Ordering::Release);
        s.run_background();
        assert_eq!(s.core.read().num_segments(), before, "merged with every slot held");
        assert!(s.bg.want_merge.load(Ordering::Acquire), "the merge request was dropped while the slots were busy");
        drop(slots);
        // A merge also needs one of the node-wide merge permits, which every
        // copy in the process shares (other tests' merges, halved while any
        // search runs): while those are busy the request stays set, as it
        // does in a node, where the scheduler's tick dispatches it again.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while s.core.read().num_segments() >= before && std::time::Instant::now() < deadline {
            assert!(s.bg.want_merge.load(Ordering::Acquire) || s.core.read().num_segments() < before, "the merge request was dropped while the node's merge permits were busy");
            s.run_background();
            if s.core.read().num_segments() >= before { std::thread::sleep(std::time::Duration::from_millis(10)); }
        }
        assert!(s.core.read().num_segments() < before, "the merge request was dropped while the slots were busy");
    }

    #[test]
    fn sustained_sorted_indexing_keeps_segments_bounded_and_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.core.write().set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
            field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: false, max_mode: false, missing_first: false,
        }])));
        s.bg.served.store(true, Ordering::Release);
        let mut live = std::collections::BTreeSet::new();
        for round in 0..60u32 {
            for k in 0..3u32 {
                let id = format!("{:05}", (round * 7919 + k * 104_729) % 100_000);
                s.apply("i", "u", vec![idx(&id, "v", WriteCond::default())], false).remove(0).unwrap();
                live.insert(id);
            }
            if round % 5 == 4 {
                let gone = live.iter().next().cloned().unwrap();
                s.apply("i", "u", vec![WriteOp::Delete { id: gone.clone(), cond: WriteCond::default() }], false).remove(0).unwrap();
                live.remove(&gone);
            }
            s.refresh().unwrap();
            s.settle_background();
        }
        s.bg.want_merge.store(true, Ordering::Release);
        s.settle_background();
        let core = s.core.read();
        assert!(core.num_segments() < 50, "{} segments", core.num_segments());
        let leaves = core.leaves();
        let mut seen = std::collections::BTreeSet::new();
        for leaf in &leaves {
            let mut previous: Option<String> = None;
            for d in 0..leaf.num_docs() {
                let Some(id) = leaf.keyword_value(ID_FIELD, d) else { continue };
                if let Some(p) = &previous { assert!(p.as_str() <= id.as_str(), "segment out of sort order: {p} > {id}"); }
                previous = Some(id.clone());
                if !leaf.is_deleted(d) { seen.insert(id); }
            }
        }
        assert_eq!(seen, live, "the live documents differ");
    }

    /// The merge backlog is measured while every merge slot is busy, so the
    /// indexing throttle (`max_merge_count`) engages as segments pile up
    /// faster than merges run (it never did: the backlog was only measured
    /// when a merge started).
    #[test]
    fn the_indexing_throttle_engages_while_merge_slots_are_busy() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        { let mut c = s.bg.merge_config.write(); c.max_merge_count = 0; c.auto_throttle = false; }
        let slots: Vec<_> = std::iter::from_fn(|| s.bg.merge_running.try_acquire()).collect();
        for i in 0..30 {
            s.apply("i", "u", vec![idx(&i.to_string(), "small", WriteCond::default())], false).remove(0).unwrap();
            s.refresh().unwrap();
            s.seal_all().unwrap();
        }
        assert!(s.start_merge().is_none());
        assert!(s.bg.merge_backlog.load(Ordering::Acquire) > 0, "the backlog was not measured with the slots busy");
        assert!(s.merge_throttle().is_some(), "writes are not throttled behind the merge backlog");
        drop(slots);
    }

    /// ES's indexing throttle condition: merges in flight past
    /// `max_merge_count`, and with the auto IO throttle on only once the
    /// merge IO rate is at its maximum.
    #[test]
    fn the_indexing_throttle_waits_for_unthrottled_merge_io() {
        assert!(!merge_throttle_due(9, 9, false, false));
        assert!(merge_throttle_due(10, 9, false, false));
        assert!(!merge_throttle_due(10, 9, true, false), "merges still IO-throttled: indexing is not");
        assert!(merge_throttle_due(10, 9, true, true));
        assert!(!merge_throttle_due(9, 9, true, true));
    }

    /// A retired copy (its files closed for the copy opened next) never
    /// flushes again: no shard state rewrite, no translog trim, no append.
    #[test]
    fn a_copy_with_closed_files_never_flushes_or_appends() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        let state = std::fs::read(tmp.path().join(STATE_FILE)).unwrap();
        s.close_files();
        assert!(s.flush().is_err(), "a retired copy flushed");
        assert_eq!(std::fs::read(tmp.path().join(STATE_FILE)).unwrap(), state, "the shard state was rewritten");
        assert!(s.apply("i", "u", vec![idx("b", "x", WriteCond::default())], false).remove(0).is_err(), "a retired copy appended");
    }

    #[test]
    fn concurrent_writers_get_unique_gap_free_seq_nos() {
        // G2: seq_nos stay unique and gap-free with sequencing split from
        // the translog append; same-id ops apply in order (one live copy).
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let writers: Vec<_> = (0..8).map(|w| {
            let s = s.clone();
            std::thread::spawn(move || {
                let mut seqs = Vec::new();
                for round in 0..50 {
                    let ops = (0..5).map(|k| idx(&format!("id{}", (w * 7 + round + k) % 40), "v", WriteCond::default())).collect();
                    for r in s.apply("i", "u", ops, false) { seqs.push(r.unwrap().seq_no); }
                }
                seqs
            })
        }).collect();
        let mut all: Vec<i64> = writers.into_iter().flat_map(|w| w.join().unwrap()).collect();
        all.sort_unstable();
        let n = all.len() as i64;
        all.dedup();
        assert_eq!(all.len() as i64, n, "a seq_no was assigned twice");
        assert_eq!(all, (0..n).collect::<Vec<_>>(), "seq_nos have a gap");
        assert_eq!(s.local_checkpoint(), n - 1);
        s.refresh().unwrap();
        for i in 0..40 { assert_eq!(live_copies(&s, &format!("id{i}")), 1, "id{i}"); }
        s.flush().unwrap();
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.is_empty() || replay.iter().all(|op| op.seq_no() as i64 > s.committed_seq_no()));
    }

    #[test]
    fn replica_replay_appends_first_copies_above_the_primarys_msu() {
        // MSU (ES max_seq_no_of_updates_or_deletes): with the primary's MSU
        // known, replayed ops above it are first copies (no buffered delete,
        // no lookup); ops at or below it take the update path; the copy stays
        // exact, also once it sequences as a primary.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        let index = |id: &str, title: &str, seq_no: i64| StagedReplay::Index { id: id.into(), source: format!("{{\"title\":\"{title}\"}}"),
            doc: doc(title), ignored: vec![], routing: None, seq_no, version: 1 };
        s.advance_max_seq_no_of_updates_from_primary(-1);
        assert!(s.replay_staged((0..10).map(|i| index(&format!("d{i}"), "v1", i)).collect()).unwrap().iter().all(|a| *a));
        let mut writers = std::mem::take(&mut *s.pool.lock());
        writers.extend(std::mem::take(&mut *s.frozen.lock()));
        assert!(s.checkin_taken(writers, None));
        assert!(s.unresolved.lock().is_empty(), "first copies buffered older-copy deletes");
        s.refresh().unwrap();
        // The primary updated d3 (seq 10) and deleted d4 (seq 11): MSU 11.
        s.advance_max_seq_no_of_updates_from_primary(11);
        let update = StagedReplay::Index { id: "d3".into(), source: "{\"title\":\"v2\"}".into(), doc: doc("v2"), ignored: vec![], routing: None, seq_no: 10, version: 2 };
        assert!(s.replay_staged(vec![update, StagedReplay::Delete { id: "d4".into(), seq_no: 11, version: 2 }]).unwrap().iter().all(|a| *a));
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "d3"), 1, "the update left two copies");
        assert_eq!(live_copies(&s, "d4"), 0);
        assert_eq!(s.get_mode("d3", false).unwrap().unwrap().version, 2);
        assert!(s.max_seq_no_of_updates() >= 11);
        // Promoted: its own writes keep one copy per id.
        assert!(s.apply("i", "u", vec![idx("d5", "v2", WriteCond::default())], false)[0].is_ok());
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "d5"), 1);
        assert!(s.max_seq_no_of_updates() > 11, "the overwrite raised the MSU");
    }

    #[test]
    fn background_merges_run_concurrently_on_disjoint_segments() {
        // G3: up to merge_threads() background merges of one shard run at
        // once, each claiming disjoint segments; the backlog is tracked for
        // the write stall and shrinks as they install.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for round in 0..40 {
            assert!(s.apply("i", "u", vec![idx(&format!("d{round}"), "x", WriteCond::default())], false)[0].is_ok());
            s.refresh().unwrap();
            s.seal_all().unwrap();
        }
        let before = s.core.read().num_segments();
        assert!(before >= 30, "fixture: many small segments ({before})");
        let first = s.start_merge_waiting().expect("a merge");
        let backlog = s.bg.merge_backlog.load(Ordering::Acquire);
        // The policy itself must offer a second merge of disjoint segments
        // while the first is claimed (it does not serialise merges).
        {
            let mut core = s.core.write();
            let policy = s.bg.merge_config.read().policy(core.avg_doc_bytes());
            let other = core.begin_merge(&policy).expect("the policy proposes a second merge while the first runs");
            let (a, b) = (first.job.segment_ids(), other.segment_ids());
            assert!(a.iter().all(|id| !b.contains(id)), "the merges share a segment: {a:?} {b:?}");
            core.release_merge(&other);
        }
        // A second slot: node-wide merge permits are shared with every other
        // shard of the process (other tests merge and search concurrently),
        // so wait for one briefly instead of asserting it is free at once.
        let second = if merge_threads() >= 2 {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
            loop {
                if let Some(t) = s.start_merge() { break Some(t); }
                if std::time::Instant::now() >= until { break None; }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        } else { None };
        if let Some(second) = second {
            assert!(backlog > 0, "40 small segments leave more than one merge to do");
            let (a, b) = (first.job.segment_ids(), second.job.segment_ids());
            assert!(a.iter().all(|id| !b.contains(id)), "the merges share a segment: {a:?} {b:?}");
            let (da, db) = std::thread::scope(|scope| {
                let x = scope.spawn(move || first.run());
                let y = scope.spawn(move || second.run());
                (x.join().unwrap(), y.join().unwrap())
            });
            assert!(s.finish_merge(db));
            assert!(s.finish_merge(da));
        } else {
            assert!(s.finish_merge(first.run()));
        }
        assert!(s.core.read().num_segments() < before, "the merges installed");
        assert!(s.bg.merge_backlog.load(Ordering::Acquire) <= backlog);
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 40);
    }

    /// ES `MergePolicyConfig`: `index.merge.policy.type` `unset` (the
    /// default) picks the time-based `LogByteSizeMergePolicy` when the copy's
    /// mapping had a `@timestamp` date field at open, else tiered; `tiered`
    /// and `time_based` override it. The time-based policy merges
    /// `merge_factor` (32) segments from `floor_segment` up to
    /// `max_merged_segment`, else 100gb; tiered keeps its 5gb default.
    #[test]
    fn merge_policy_type_selects_time_based_like_es() {
        let settings = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            MergeConfig::from_settings(move |k| map.get(k).cloned())
        };
        let with_ts = |mut c: MergeConfig, ts: bool| { c.timestamp_field = ts; c };
        let d = settings(&[]).unwrap();
        assert_eq!((d.policy_type, d.merge_factor, d.time_based_max_merged_segment_bytes, d.max_merged_segment_bytes), (MergePolicyType::Unset, 32, 100 << 30, 5 << 30));
        assert!(!with_ts(d.clone(), false).time_based());
        assert!(with_ts(d.clone(), true).time_based());
        let p = with_ts(d.clone(), true).policy(None);
        assert_eq!(p.log_bytes, Some(rustic_core::merge::LogBytes { merge_factor: 32, min_merge_bytes: 16 << 20, max_merge_bytes: 100 << 30 }));
        assert_eq!(with_ts(d.clone(), false).policy(None).log_bytes, None);
        // Overrides, case-insensitive (ES upper-cases enum settings).
        let tiered = settings(&[("index.merge.policy.type", "tiered")]).unwrap();
        assert!(!with_ts(tiered, true).time_based());
        let time = settings(&[("index.merge.policy.type", "TIME_BASED")]).unwrap();
        assert!(with_ts(time, false).time_based());
        let err = settings(&[("index.merge.policy.type", "log")]).unwrap_err().to_string();
        assert!(err.contains("No enum constant org.elasticsearch.index.MergePolicyConfig.Type.LOG"), "{err}");
        // An explicit max_merged_segment is both policies'; 0b is unset.
        let c = settings(&[("index.merge.policy.max_merged_segment", "20gb"), ("index.merge.policy.merge_factor", "10")]).unwrap();
        assert_eq!((c.max_merged_segment_bytes, c.time_based_max_merged_segment_bytes, c.merge_factor), (20 << 30, 20 << 30, 10));
        let c = settings(&[("index.merge.policy.max_merged_segment", "0b")]).unwrap();
        assert_eq!((c.max_merged_segment_bytes, c.time_based_max_merged_segment_bytes), (5 << 30, 100 << 30));
        let err = settings(&[("index.merge.policy.merge_factor", "1")]).unwrap_err().to_string();
        assert!(err.contains("must be >= 2"), "{err}");
    }

    #[test]
    fn merge_settings_parse_validate_and_apply_dynamically() {
        let settings = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            MergeConfig::from_settings(move |k| map.get(k).cloned())
        };
        let c = settings(&[("index.merge.scheduler.max_thread_count", "2"), ("index.merge.policy.segments_per_tier", "5.0"),
            ("index.merge.policy.floor_segment", "4mb"), ("index.merge.policy.max_merged_segment", "1gb"),
            ("index.merge.scheduler.auto_throttle", "false")]).unwrap();
        if std::env::var("RUSTIC_MERGE_THREADS").is_err() { assert_eq!(c.max_thread_count, 2); }
        assert_eq!(c.max_merge_count, c.max_thread_count + 5, "default max_merge_count is max_thread_count + 5");
        assert_eq!((c.segments_per_tier, c.floor_segment_bytes, c.max_merged_segment_bytes, c.auto_throttle), (5.0, 4 << 20, 1 << 30, false));
        let d = MergeConfig::default();
        assert_eq!((d.segments_per_tier, d.max_merge_at_once, d.floor_segment_bytes, d.max_merged_segment_bytes, d.deletes_pct_allowed, d.expunge_deletes_allowed),
            (8.0, 16, 16 << 20, 5 << 30, 20.0, 10.0));
        assert!(settings(&[("index.merge.policy.deletes_pct_allowed", "60")]).is_err());
        assert!(settings(&[("index.merge.policy.segments_per_tier", "1")]).is_err());
        assert!(settings(&[("index.merge.scheduler.auto_throttle", "maybe")]).is_err());
        assert!(settings(&[("index.merge.policy.floor_segment", "lots")]).is_err());
        // An inverted pair in one update is rejected; stored inverted, clamped.
        let mut flat = crate::settings::FlatSettings::new();
        flat.insert("index.merge.scheduler.max_thread_count".into(), serde_json::json!("4"));
        flat.insert("index.merge.scheduler.max_merge_count".into(), serde_json::json!("2"));
        let err = crate::settings::validate(&flat).unwrap_err();
        assert!(err.to_string().contains("maxThreadCount (= 4) should be <= maxMergeCount (= 2)"), "{err}");
        if std::env::var("RUSTIC_MERGE_THREADS").is_err() {
            let clamped = settings(&[("index.merge.scheduler.max_thread_count", "4"), ("index.merge.scheduler.max_merge_count", "2")]).unwrap();
            assert_eq!(clamped.max_merge_count, 4);
        }
        // Dynamic: the shard's merge slots follow max_thread_count.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.set_merge_config(MergeConfig { max_thread_count: 1, ..MergeConfig::default() });
        let one = s.bg.merge_running.try_acquire().unwrap();
        assert!(s.bg.merge_running.try_acquire().is_none());
        s.set_merge_config(MergeConfig { max_thread_count: 3, ..MergeConfig::default() });
        let two = s.bg.merge_running.try_acquire().unwrap();
        let three = s.bg.merge_running.try_acquire().unwrap();
        assert!(s.bg.merge_running.try_acquire().is_none());
        drop((one, two, three));
    }

    #[test]
    fn many_tiny_segments_converge_below_the_tier_budget() {
        // Tiny seals are merge-eligible (below floor_segment): background
        // merges bring the count under segments_per_tier.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.set_merge_config(MergeConfig { segments_per_tier: 8.0, max_merge_at_once: 8, ..MergeConfig::default() });
        for round in 0..80 {
            assert!(s.apply("i", "u", vec![idx(&format!("d{round}"), "x", WriteCond::default())], false)[0].is_ok());
            s.refresh().unwrap();
            s.seal_all().unwrap();
        }
        assert!(s.core.read().num_segments() >= 60);
        let mut rounds = 0;
        while let Some(task) = s.start_merge() {
            assert!(s.finish_merge(task.run()));
            rounds += 1;
            assert!(rounds < 100, "merging does not converge");
        }
        let left = s.core.read().num_segments();
        assert!(left <= 8, "{left} segments left (segments_per_tier 8)");
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs(), 80);
    }

    #[test]
    fn replay_above_the_floor_keeps_every_doc_once_and_deletes_win() {
        // D1/D2: an explicit-id batch held between sequencing and inversion
        // holds the commit point below completed auto-id bulks and a
        // completed delete; the flush commits those above it, a crash
        // follows, and the replay of every op above the commit point must
        // neither duplicate an auto-id document nor resurrect the deleted id
        // (its older index op is in the held batch).
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("x", "first", WriteCond::default())], false)[0].is_ok());
        let (sequenced, release) = hold_next_staged_write(&s);
        let held = {
            let s = s.clone();
            std::thread::spawn(move || s.apply("i", "u", vec![idx("x", "held", WriteCond::default()), idx("held1", "h", WriteCond::default())], false))
        };
        sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let mut auto_ids = Vec::new();
        for round in 0..20 {
            let ops = (0..3).map(|i| WriteOp::Index { id: format!("auto-{round}-{i}"), source: "{}".into(), doc: doc("auto"),
                ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true }).collect();
            for r in s.apply("i", "u", ops, false) { auto_ids.push(r.unwrap().id); }
        }
        assert!(s.apply("i", "u", vec![WriteOp::Delete { id: "x".into(), cond: WriteCond::default() }], false)[0].is_ok());
        s.flush().unwrap();
        let committed = s.committed_seq_no();
        release.send(()).unwrap();
        let held = held.join().unwrap();
        let x_held = held[0].as_ref().unwrap().seq_no;
        assert!(committed < x_held, "the held batch holds the commit point ({committed} vs {x_held})");
        assert!(s.max_seq_no() > x_held + 60, "the auto-id bulks and the delete are above it");
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.iter().any(|op| matches!(op, TranslogOp::Index { id, .. } if id.starts_with("auto-"))),
            "committed auto-id ops above the commit point are replayed");
        for op in replay.iter().cloned() {
            match op {
                TranslogOp::Index { id, source_json, seq_no, version, routing } =>
                    s.replay_index(id, source_json, doc("replayed"), vec![], routing, seq_no as i64, version).unwrap(),
                TranslogOp::Delete { id, seq_no, version } => s.replay_delete(&id, seq_no as i64, version).unwrap(),
            }
        }
        s.refresh().unwrap();
        for id in &auto_ids { assert_eq!(live_copies(&s, id), 1, "auto-id {id} duplicated or lost by the replay"); }
        assert_eq!(live_copies(&s, "x"), 0, "the delete lost to an older replayed index op");
        assert_eq!(live_copies(&s, "held1"), 1);
    }

    #[test]
    fn failed_translog_appends_leave_no_checkpoint_hole() {
        // D3: an append that fails leaves its seq_nos as no-ops: the local
        // checkpoint passes them, the next commit covers them and retention
        // is not stalled.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        assert!(s.apply("i", "u", vec![idx("a", "t", WriteCond::default())], false)[0].is_ok());
        s.fail_next_translog_appends(1);
        let failed = s.apply("i", "u", vec![idx("b", "t", WriteCond::default()), idx("c", "t", WriteCond::default())], false);
        assert!(failed.iter().all(Result::is_err), "the failed append fails its ops: {failed:?}");
        assert_eq!(s.max_seq_no(), 2);
        assert_eq!(s.local_checkpoint(), 2, "no hole at the failed seq_nos");
        let noops: Vec<i64> = s.changes_since(0).unwrap()
            .filter(|op| matches!(&**op, crate::hooks::changes::ChangeOp::NoOp { .. })).map(|op| op.seq_no()).collect();
        assert_eq!(noops, vec![1, 2], "the failed seq_nos are no-ops in the history");
        s.flush().unwrap();
        assert_eq!(s.committed_seq_no(), 2, "the commit covers them");
        drop(s);
        // The failed engine is replaced (a reallocated copy reopens it).
        let (s, _) = Shard::open(tmp.path(), schema()).unwrap();
        assert_eq!(s.local_checkpoint(), 2, "no hole survives a reopen");
        let d = s.apply("i", "u", vec![idx("d", "t", WriteCond::default())], false).remove(0).unwrap();
        assert_eq!(d.seq_no, 3);
    }

    /// R4: the auto-id append-only path never acknowledges an op that is not
    /// durable here. A failed append indexes nothing and leaves no-ops; a
    /// failed fsync fails the ops; either fails the engine (translog tragic
    /// event), which refuses every later op and is reported once.
    #[test]
    fn append_only_writes_are_never_acknowledged_after_a_translog_failure() {
        let auto = |id: &str| WriteOp::Index { id: id.into(), source: "{\"title\":\"a\"}".into(), doc: doc("a"),
            ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true };
        for fsync in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            assert!(s.apply("i", "u", vec![auto("a0"), auto("a1")], true).iter().all(Result::is_ok));
            if fsync { s.fail_next_translog_syncs(1); } else { s.fail_next_translog_appends(1); }
            let failed = s.apply("i", "u", vec![auto("b0"), auto("b1"), auto("b2")], true);
            assert!(failed.iter().all(Result::is_err), "fsync {fsync}: acknowledged a non-durable op: {failed:?}");
            assert!(s.tragic_error().is_some(), "fsync {fsync}: the engine was not failed");
            assert_eq!(s.max_seq_no(), 4);
            assert_eq!(s.local_checkpoint(), 4, "fsync {fsync}: a hole at the failed seq_nos");
            if !fsync {
                s.refresh().unwrap();
                assert_eq!(live_copies(&s, "b0"), 0, "an op that never reached the translog was indexed");
            }
            let retry = s.apply("i", "u", vec![auto("c0"), auto("c1")], true);
            assert!(retry.iter().all(Result::is_err), "fsync {fsync}: the failed engine took a write: {retry:?}");
            assert_eq!(s.max_seq_no(), 4, "a refused op takes no seq_no");
            assert!(s.take_tragic_report().is_some());
            assert!(s.take_tragic_report().is_none(), "reported once");
        }
    }

    /// R1 (fsyncgate): a failed group-commit fsync is final. Two waiters
    /// on the same unsynced appends: the first one's fsync fails; the
    /// second one must not retry the fsync (the kernel may have cleared the
    /// error and dropped the pages) and acknowledge. Later flushes refuse
    /// too (the translog is closed).
    #[test]
    fn a_waiter_behind_a_failed_group_commit_fsync_never_acknowledges() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        s.fail_next_translog_syncs(1);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let waiters: Vec<_> = (0..2).map(|_| {
            let (s, barrier) = (s.clone(), barrier.clone());
            std::thread::spawn(move || { barrier.wait(); s.sync_translog() })
        }).collect();
        let results: Vec<_> = waiters.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(results.iter().all(Result::is_err), "a waiter acknowledged after a failed fsync: {results:?}");
        assert!(s.sync_translog().is_err(), "a later sync retried the failed fsync");
        assert!(s.tragic_error().is_some());
        assert!(s.flush().is_err(), "a flush rolled the closed translog");
    }

    /// R1 (flush roll): a failed fsync in the flush's translog roll fails
    /// the engine, and a writer that appended before the roll and waits
    /// for its group commit afterwards never acknowledges (its retried
    /// fsync of the same file would succeed once the kernel dropped the
    /// error).
    #[test]
    fn a_failed_roll_fsync_fails_the_engine_and_every_waiting_writer() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", vec![idx("a", "x", WriteCond::default())], false).remove(0).unwrap();
        s.write.lock().translog.fail_next_fsync();
        assert!(s.flush().is_err(), "the roll's fsync failed");
        assert!(s.tragic_error().is_some(), "a failed roll left the engine healthy");
        assert!(!s.core.read().seal_in_flight(), "failed pre-encode roll stranded captured seal claims");
        assert!(s.get("a").unwrap().is_some(), "cancelling unrun claims must not discard their documents");
        assert!(s.sync_translog().is_err(), "a waiting writer acknowledged after the failed roll fsync");
    }

    #[test]
    fn failed_generation_lookup_releases_all_unrun_capture_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        // Use the shard publication boundary, which also retires the pool's
        // current writer. Core-only refresh can leave all four appends in one
        // reusable pool writer and therefore cannot establish this fixture.
        s.bg.threshold.store(1_000_000, Ordering::Relaxed);
        for row in 0..4 {
            s.apply("i", "u", vec![idx(&row.to_string(), "x", WriteCond::default())], false).remove(0).unwrap();
            s.refresh().unwrap();
        }
        assert!(s.core.read().num_closed_memtables() >= 2, "fixture must retain multiple published writers");
        let jobs = s.core.write().begin_seal_chunks(true, 1).unwrap();
        assert!(jobs.len() >= 2, "cleanup must cover multiple independently claimed chunks");
        let translog = s.path.join(TRANSLOG_DIR);
        let saved = s.path.join("translog-retained-for-lookup-failure");
        std::fs::rename(&translog, &saved).unwrap();
        std::fs::write(&translog, b"not a directory").unwrap();
        let result: Result<()> = (|| {
            // This is the same fallible pre-encode lookup and ownership
            // scope as flush_inner after claiming its capture.
            let _pending = PendingSealChunks::new(&s, jobs);
            let ws = s.write.lock();
            s.generation_holding(&ws, 0, u64::MAX)?;
            Ok(())
        })();
        // Restore retained files before assertions; no corpus or log is lost.
        std::fs::remove_file(&translog).unwrap();
        std::fs::rename(&saved, &translog).unwrap();
        assert!(result.is_err());
        assert!(!s.core.read().seal_in_flight());
        assert!(s.tragic_error().is_none(), "a failed listing alone does not corrupt the translog");
        for row in 0..4 { assert!(s.get(&row.to_string()).unwrap().is_some()); }
        s.seal_all().unwrap();
        s.flush().unwrap();
        assert!(!s.core.read().seal_in_flight());
    }

    /// R4: the mutable (vector) append-only path fails the engine on a
    /// failed append or fsync and acknowledges none of the batch.
    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn mutable_append_only_writes_fail_the_engine_on_a_translog_failure() {
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let batch = |base: usize| -> Vec<WriteOp> { (0..4).map(|i| {
            let mut doc = doc("v"); doc.add_vector(9, vec![(base + i) as f32, 1.]);
            WriteOp::Index { id: format!("{base}-{i}"), source: "{}".into(), doc,
                ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: true }
        }).collect() };
        for fsync in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
            assert!(s.apply("i", "u", batch(0), true).iter().all(Result::is_ok));
            if fsync { s.fail_next_translog_syncs(1); } else { s.fail_next_translog_appends(1); }
            let failed = s.apply("i", "u", batch(10), true);
            assert!(failed.iter().all(Result::is_err), "fsync {fsync}: {failed:?}");
            assert!(s.tragic_error().is_some(), "fsync {fsync}: the engine was not failed");
            assert!(s.apply("i", "u", batch(20), true).iter().all(Result::is_err), "fsync {fsync}: the failed engine took a write");
        }
    }

    #[test]
    fn parked_frozen_memtables_stay_visible_to_realtime_reads_and_versioning() {
        // D4: a check-in that cannot take the core parks the memtables it
        // took in the frozen list; until a later check-in their documents
        // stay addressable (realtime GET, version checks).
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let a = s.apply("i", "u", vec![idx("a", "parked", WriteCond::default())], false).remove(0).unwrap();
        {
            let _core = s.core.write();
            let other = s.clone();
            let parked = std::thread::spawn(move || other.checkin_completed_writes(Some(std::time::Duration::ZERO)));
            assert!(!parked.join().unwrap(), "the check-in could not take the core");
        }
        assert!(!s.frozen.lock().is_empty(), "the memtable is parked");
        let got = s.get("a").unwrap().expect("a parked document is addressable");
        assert_eq!(got.seq_no, a.seq_no);
        let conflict = s.apply("i", "u", vec![idx("a", "x", WriteCond { create: true, ..Default::default() })], false).remove(0);
        assert!(conflict.is_err(), "a create over a parked document conflicts");
        let cas = WriteCond { if_seq_no: Some(a.seq_no), if_primary_term: Some(a.primary_term), ..Default::default() };
        assert!(s.apply("i", "u", vec![idx("a", "updated", cas)], false)[0].is_ok(), "CAS against the parked version");
        s.refresh().unwrap();
        assert_eq!(live_copies(&s, "a"), 1);
        assert_eq!(s.get("a").unwrap().unwrap().version, 2);
    }

    #[test]
    fn refresh_sees_writes_another_checkin_holds() {
        // A check-in that took the idle pooled memtables and is still
        // waiting (here for a holder to hand off, longer than a refresh's
        // own freeze wait) holds those writes alone; a refresh starting
        // meanwhile must not return without them.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        // A request holding a (still empty) pooled memtable that never
        // reaches a document boundary while the check-ins run.
        let (holding, pooled) = s.hold_pool_writer(None).unwrap();
        for id in ["a", "b", "c"] {
            assert!(s.apply("i", "u", vec![idx(id, "acked", WriteCond::default())], false)[0].is_ok());
        }
        let deferred = { let s = s.clone(); std::thread::spawn(move || s.checkin_completed_writes(Some(std::time::Duration::from_secs(60)))) };
        let t = std::time::Instant::now();
        while !s.pool.lock().is_empty() {
            assert!(t.elapsed() < std::time::Duration::from_secs(10), "the deferred check-in never took the pool");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let refresh = { let s = s.clone(); std::thread::spawn(move || { s.refresh().unwrap(); s.reader_view().num_live_docs() as u64 }) };
        // Past the refresh's own freeze wait.
        std::thread::sleep(FREEZE_WAIT + std::time::Duration::from_secs(1));
        s.release_holding(holding);
        drop(pooled);
        let _ = deferred.join().unwrap();
        assert_eq!(refresh.join().unwrap(), 3, "the refresh returned without writes acknowledged before it");
    }

    #[test]
    fn concurrent_bulk_refresh_flush_search_keep_refresh_visibility() {
        // Bulks, explicit refreshes, flushes, scheduled / deferred check-ins
        // and searches at once: every refresh shows at least what was
        // acknowledged before it started, and nothing stalls.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let acked = Arc::new(AtomicU64::new(0));
        let mut threads = Vec::new();
        for w in 0..4 {
            let (s, stop, acked) = (s.clone(), stop.clone(), acked.clone());
            threads.push(std::thread::spawn(move || {
                let mut round = 0usize;
                while !stop.load(Ordering::Acquire) {
                    let ops = (0..50).map(|i| idx(&format!("w{w}-{round}-{i}"), "stress text", WriteCond::default())).collect();
                    let results = s.apply("i", "u", ops, false);
                    assert!(results.iter().all(Result::is_ok));
                    acked.fetch_add(results.len() as u64, Ordering::AcqRel);
                    round += 1;
                }
                "bulk"
            }));
        }
        let max_refresh = Arc::new(AtomicU64::new(0));
        for _ in 0..2 {
            let (s, stop, acked, max_refresh) = (s.clone(), stop.clone(), acked.clone(), max_refresh.clone());
            threads.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let before = acked.load(Ordering::Acquire);
                    let t = std::time::Instant::now();
                    s.refresh().unwrap();
                    max_refresh.fetch_max(t.elapsed().as_millis() as u64, Ordering::AcqRel);
                    let visible = s.reader_view().num_live_docs() as u64;
                    assert!(visible >= before, "refresh shows {visible} docs, {before} were acknowledged before it");
                }
                "refresh"
            }));
        }
        {
            let (s, stop) = (s.clone(), stop.clone());
            threads.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    s.flush().unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                "flush"
            }));
        }
        {
            let (s, stop) = (s.clone(), stop.clone());
            threads.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let _ = s.try_publish();
                    let _ = s.try_publish_for(std::time::Duration::from_millis(5));
                    s.checkin_completed_writes(Some(std::time::Duration::from_millis(2)));
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                "scheduled"
            }));
        }
        {
            let (s, stop) = (s.clone(), stop.clone());
            threads.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let _ = s.reader_view().num_live_docs();
                    let _ = s.get("w0-0-0").unwrap();
                }
                "search"
            }));
        }
        std::thread::sleep(std::time::Duration::from_secs(4));
        stop.store(true, Ordering::Release);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let total = threads.len();
        std::thread::spawn(move || { for t in threads { done_tx.send(t.join()).unwrap(); } });
        for _ in 0..total {
            let joined = done_rx.recv_timeout(std::time::Duration::from_secs(60)).expect("a thread stalled");
            joined.unwrap();
        }
        s.refresh().unwrap();
        assert_eq!(s.reader_view().num_live_docs() as u64, acked.load(Ordering::Acquire));
        assert!(max_refresh.load(Ordering::Acquire) < 10_000, "a refresh took {} ms", max_refresh.load(Ordering::Acquire));
    }

    #[test]
    fn bulk_in_flight_never_blocks_refresh_or_realtime_get() {
        // W1: a long bulk inverts into its pooled memtable outside every
        // shard lock; refreshes and realtime GETs meanwhile take only the
        // short sections.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("probe", "t", WriteCond::default())], false)[0].is_ok());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bulk = {
            let (s, stop) = (s.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut round = 0usize;
                while !stop.load(Ordering::Acquire) {
                    let ops = (0..4000).map(|i| idx(&format!("b{round}-{i}"), "some longer text to analyze for the bulk", WriteCond::default())).collect();
                    assert!(s.apply("i", "u", ops, false).iter().all(Result::is_ok));
                    round += 1;
                }
                round
            })
        };
        let mut slowest = std::time::Duration::ZERO;
        let started = std::time::Instant::now();
        while started.elapsed() < std::time::Duration::from_secs(3) {
            let t0 = std::time::Instant::now();
            s.refresh().unwrap();
            assert!(s.get("probe").unwrap().is_some());
            slowest = slowest.max(t0.elapsed());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        stop.store(true, Ordering::Release);
        let rounds = bulk.join().unwrap();
        assert!(rounds >= 1);
        // A refresh checks in the idle pooled memtables (up to one bulk's
        // documents); it never waits for a bulk being inverted.
        assert!(slowest < std::time::Duration::from_millis(750), "refresh + GET took {slowest:?} during bulks");
    }

    #[test]
    fn scheduled_and_deferred_refreshes_record_the_refreshed_checkpoint() {
        // A fleet search waits for `refreshed_checkpoint`: the scheduler's
        // refreshes (`try_publish`, and `refresh_deferred` when the core was
        // busy) advance it as an explicit refresh does.
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        assert!(s.apply("i", "u", vec![idx("1", "t", WriteCond::default())], false)[0].is_ok());
        let first = s.processed_checkpoint();
        assert!(first >= 0);
        assert!(s.refreshed_checkpoint() < first);
        // A writer not yet handed off hands the refresh to the deferred path.
        match s.try_publish() {
            Some(published) => assert!(published.unwrap()),
            None => s.refresh_deferred(),
        }
        assert_eq!(s.refreshed_checkpoint(), first);
        assert!(s.apply("i", "u", vec![idx("2", "t", WriteCond::default())], false)[0].is_ok());
        let second = s.processed_checkpoint();
        s.refresh_deferred();
        assert_eq!(s.refreshed_checkpoint(), second);
    }

    #[test]
    fn refresh_wait_for_waits_for_the_next_refresh_without_forcing_one() {
        // ES: `refresh=wait_for` parks until a refresh makes the write
        // visible; past `index.max_refresh_listeners` it forces one.
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        assert!(s.apply("i", "u", vec![idx("1", "t", WriteCond::default())], false)[0].is_ok());
        let waiter = s.clone();
        let (done, waited) = std::sync::mpsc::channel();
        let listener = std::thread::spawn(move || {
            let forced = waiter.wait_for_refresh(1000, std::time::Duration::from_secs(30)).unwrap();
            done.send(()).unwrap();
            forced
        });
        assert!(waited.recv_timeout(std::time::Duration::from_millis(300)).is_err(), "wait_for returned before any refresh");
        assert!(s.core.read().leaves().iter().all(|leaf| leaf.num_live_docs_latest() == 0), "wait_for forced a refresh");
        s.refresh().unwrap();
        waited.recv_timeout(std::time::Duration::from_secs(10)).expect("the refresh released the listener");
        assert!(!listener.join().unwrap(), "satisfied by the refresh, not forced");
        // Listener limit reached: forced at once.
        assert!(s.apply("i", "u", vec![idx("2", "t", WriteCond::default())], false)[0].is_ok());
        assert!(s.wait_for_refresh(0, std::time::Duration::from_secs(30)).unwrap());
        assert!(s.get_mode("2", false).unwrap().is_some());
    }

    #[test]
    fn sorted_refresh_does_not_block_writes_while_its_seal_encodes() {
        // The sorted refresh fences nothing while it sorts and writes its
        // segment: the core is held only to begin the seal and, briefly, to
        // install and publish it. Writes issued while its I/O is stalled
        // complete at once and become visible at the next refresh.
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
            pause: std::sync::atomic::AtomicBool::new(false), entered, resume: parking_lot::Mutex::new(release) });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
        core.set_nrt(true);
        core.set_auto_seal(false);
        core.set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
            field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: true, max_mode: false, missing_first: false,
        }])));
        *s.core.write() = core;
        for row in 0..8 {
            assert!(s.apply("i", "u", vec![idx(&format!("before-{row}"), "t", WriteCond::default())], false)[0].is_ok());
        }
        dir.pause.store(true, Ordering::Release);
        let refreshing = s.clone();
        let refresh = std::thread::spawn(move || refreshing.refresh());
        wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        let writer = s.clone();
        let (done, written) = std::sync::mpsc::channel();
        let writes = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let results: Vec<_> = (0..16).map(|row| writer.apply("i", "u",
                vec![idx(&format!("during-{row}"), "t", WriteCond::default())], false)).collect();
            done.send(started.elapsed()).unwrap();
            results
        });
        let took = written.recv_timeout(std::time::Duration::from_secs(5));
        let readable = s.core.try_read().is_some();
        // Always release the refresh, including when an assertion fails.
        resume.send(()).unwrap();
        refresh.join().unwrap().unwrap();
        let results = writes.join().unwrap();
        assert!(took.is_ok(), "writes waited behind the sorted refresh's seal I/O");
        assert!(readable, "the sorted refresh held the core across its seal I/O");
        assert!(results.iter().all(|r| r.iter().all(Result::is_ok)), "{results:?}");
        assert_eq!(s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 8,
            "the refresh publishes what it sealed, not the writes made meanwhile");
        s.refresh().unwrap();
        assert_eq!(s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 24);
        assert!(s.get("during-0").unwrap().is_some() && s.get("before-0").unwrap().is_some());
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn flushes_skip_graph_saves_and_reopen_catches_the_graph_up() {
        // Vespa-style scheduling: only the first flush saves the graph; the
        // later ones commit text only. Reopening after them (a crash before
        // the next graph save) catches the graph up from the commit.
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let make = |id: usize| {
            let mut d = doc("graph"); d.add_vector(9, vec![id as f32, 1.]);
            WriteOp::Index { id: id.to_string(), source: "{}".into(), doc: d, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }
        };
        let s = Shard::create(tmp.path(), "i", fields.clone()).unwrap();
        {
            let mut saves = s.graph_saves.lock();
            saves.max_ops = i64::MAX;
            saves.max_age = std::time::Duration::from_secs(3600);
        }
        let rmvc = |s: &Shard| -> Vec<String> {
            let generation = s.write.lock().core_generation;
            let mut names: Vec<String> = std::fs::read_dir(core_dir(tmp.path(), generation)).unwrap().flatten()
                .map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".rmvc")).collect();
            names.sort();
            names
        };
        assert!(s.apply("i", "u", (0..40).map(make).collect(), false).iter().all(Result::is_ok));
        s.flush().unwrap();
        let first = rmvc(&s);
        assert_eq!(first.len(), 1);
        let saved_at = s.graph_saves.lock().seq_no;
        assert_eq!(saved_at, Some(s.committed_seq_no()));
        for round in 1..3usize {
            assert!(s.apply("i", "u", (round * 40..round * 40 + 40).map(make).collect(), false).iter().all(Result::is_ok));
            s.flush().unwrap();
            assert_eq!(rmvc(&s), first, "a flush without a due graph save keeps the graph checkpoint");
            assert_eq!(s.graph_saves.lock().seq_no, saved_at);
        }
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), fields).unwrap();
        assert!(replay.is_empty(), "everything is in the text commit");
        assert_eq!(s.core.read().vector_store().mutable_active_nodes(9), Some(120));
        for id in [0usize, 60, 119] { assert!(s.get(&id.to_string()).unwrap().is_some()); }
    }

    /// H2: under sustained graph ingestion every flush found some in
    /// flight. A due graph save neither skips nor quiesces ingestion: the
    /// capture leaves the in-flight batch out of the saved graph and the
    /// flush completes while that batch is still held; reopening
    /// reconciles the points the saved graph lacks.
    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn a_due_graph_save_captures_without_pausing_in_flight_ingestion() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let make = |id: usize| {
            let mut d = doc("graph"); d.add_vector(9, vec![id as f32, 1.]);
            WriteOp::Index { id: id.to_string(), source: "{}".into(), doc: d, ignored: vec![], routing: None, cond: WriteCond::default(), auto_id: false }
        };
        {
            let s = Arc::new(Shard::create(tmp.path(), "i", fields.clone()).unwrap());
            { let mut saves = s.graph_saves.lock(); saves.max_ops = 0; saves.growth = 0.0; } // every flush has a save due
            assert!(s.apply("i", "u", (0..40).map(make).collect(), false).iter().all(Result::is_ok));
            s.flush().unwrap();
            let saved_at = s.graph_saves.lock().seq_no;
            assert!(s.apply("i", "u", (80..120).map(make).collect(), false).iter().all(Result::is_ok));
            // A vector bulk held with its graph ingestion in flight (rows
            // reserved, graph preparation not done), under the ingestion
            // fence (`batches.read`) as every vector bulk is.
            let (sequenced_tx, sequenced) = std::sync::mpsc::channel();
            let (release, release_rx) = std::sync::mpsc::channel::<()>();
            s.core.read().vector_store().pause_next_mutable_stage(move || {
                let _ = sequenced_tx.send(());
                let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
            });
            let held = {
                let s = s.clone();
                std::thread::spawn(move || s.apply("i", "u", (40..80).map(make).collect(), false))
            };
            sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
            // Completes with the bulk still held: no pause, no skip.
            s.flush().unwrap();
            assert!(s.graph_saves.lock().seq_no > saved_at, "the due graph save ran with ingestion in flight");
            assert_eq!(s.graph_saves.lock().skipped, 0);
            assert!(!s.graph_save_overdue());
            assert!(!held.is_finished(), "the in-flight bulk was never waited for");
            release.send(()).unwrap();
            assert!(held.join().unwrap().iter().all(Result::is_ok));
            s.flush().unwrap();
        }
        let (s, _replay) = Shard::open(tmp.path(), fields).unwrap();
        assert_eq!(s.core.read().vector_store().mutable_active_nodes(9), Some(120), "every point is in the reopened graph");
        for id in [0usize, 40, 79, 119] { assert!(s.get(&id.to_string()).unwrap().is_some()); }
    }

    /// The save schedule is geometric: after a save at seq_no `n`, the next
    /// is due only once `max(max_ops, growth * n)` more ops are committed.
    #[test]
    fn graph_saves_are_scheduled_geometrically() {
        let mut saves = GraphSaves::new(Some(1_000_000));
        saves.max_ops = 200_000;
        saves.growth = 0.5;
        assert!(!saves.due(1_200_000), "max_ops alone is not enough for a large graph");
        assert!(!saves.due(1_499_999));
        assert!(saves.due(1_500_000));
        let mut small = GraphSaves::new(Some(100));
        small.max_ops = 200_000;
        assert!(!small.due(150_000));
        assert!(small.due(200_100));
        // Nothing new: an old checkpoint is still exact.
        let mut idle = GraphSaves::new(Some(10));
        idle.max_age = std::time::Duration::ZERO;
        assert!(!idle.due(10));
        assert!(idle.due(11));
        assert!(GraphSaves::new(None).due(0));
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn long_pit_never_stalls_vector_writes_or_flush() {
        // Readers never make writers wait: a retained view pinning the old
        // graph while every vector is replaced, repaired and flushed several
        // times over leaves writes and flushes flowing, and the view keeps
        // its own documents.
        fn make(id: usize, x: f32) -> WriteOp {
            let mut d = doc("pit"); d.add_vector(9, vec![x, 1.]);
            WriteOp::Index { id: id.to_string(), source: "{}".into(), doc: d, ignored: vec![],
                routing: None, cond: WriteCond::default(), auto_id: false }
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut fields = schema();
        fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
        let s = Arc::new(Shard::create(tmp.path(), "i", fields).unwrap());
        assert!(s.apply("i", "u", (0..256).map(|i| make(i, i as f32)).collect(), false).iter().all(Result::is_ok));
        s.refresh().unwrap();
        let pit = s.retain_view().unwrap();
        assert_eq!(pit.num_live_docs(), 256);
        let writer = s.clone();
        let (done, finished) = std::sync::mpsc::channel();
        let writes = std::thread::spawn(move || {
            let mut ok = true;
            for round in 1..=8usize {
                let results = writer.apply("i", "u", (0..256).map(|i| make(i, (round * 1000 + i) as f32)).collect(), false);
                ok &= results.iter().all(Result::is_ok);
                writer.refresh().unwrap();
                writer.repair_vectors();
            }
            done.send(()).unwrap();
            ok
        });
        let flusher = s.clone();
        let flushes = std::thread::spawn(move || (0..4).map(|_| flusher.flush()).collect::<Vec<_>>());
        let flowed = finished.recv_timeout(std::time::Duration::from_secs(60)).is_ok();
        assert!(flowed, "vector writes stalled while a retained view pinned the graph");
        assert!(writes.join().unwrap(), "a vector write failed because a reader pinned the graph");
        assert!(flushes.join().unwrap().iter().all(Result::is_ok));
        assert_eq!(pit.num_live_docs(), 256, "the retained view lost its documents");
        drop(pit);
        s.refresh().unwrap();
        s.repair_vectors();
        // What searches see: segments and published memtables. The core's
        // own live count covers sealed segments only, and whether the last
        // round was sealed depends on whether a flush ran after it (no
        // scheduler seals this copy), so it is checked after a flush.
        assert_eq!(s.reader_view().num_live_docs(), 256, "the refreshed view lost or duplicated documents");
        s.flush().unwrap();
        assert_eq!(s.core.read().num_live_docs(), 256, "the flushed segments lost or duplicated documents");
        assert!(s.get("0").unwrap().is_some_and(|d| d.version == 9));
    }

    /// A shard with a registered dense_vector field keeps the kNN fetch's
    /// small stored chunks (82b572f6): its ~1.5 KB docs close their chunks
    /// at 8 KB through seal, flush and merge, where a shard without vectors
    /// fills 64 KB chunks (851ae318). Rally's recall-knn reads each hit's
    /// stored doc for `fields`.
    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn vector_shard_stores_large_docs_in_small_chunks() {
        fn make(i: usize, vector: bool) -> WriteOp {
            let text = format!("doc {i} {}", "x".repeat(1500));
            let mut d = doc(&text);
            if vector { d.add_vector(9, vec![i as f32, 1.]); }
            WriteOp::Index { id: i.to_string(), source: format!("{{\"title\":\"{text}\"}}"), doc: d, ignored: vec![],
                routing: None, cond: WriteCond::default(), auto_id: false }
        }
        let chunk_docs = |vector: bool| -> Vec<u32> {
            let tmp = tempfile::tempdir().unwrap();
            let mut fields = schema();
            if vector {
                fields.push(CoreField::new(9, "v", rustic_core::types::FieldType::Vector { dim: 2 })
                    .with_similarity("l2_norm").with_vector_options(Some(rustic_core::vector::store::VectorFieldOptions::f32_hnsw())));
            }
            let s = Shard::create(tmp.path(), "i", fields).unwrap();
            for round in 0..3 {
                assert!(s.apply("i", "u", (round * 200..round * 200 + 200).map(|i| make(i, vector)).collect(), false).iter().all(Result::is_ok));
                s.refresh().unwrap();
                s.flush().unwrap();
            }
            s.force_merge(1, false).unwrap();
            let core = s.core.read();
            assert!(!core.segments().is_empty());
            core.segments().iter().flat_map(|seg| { let r = seg.store().unwrap(); (0..r.num_chunks()).map(|c| r.chunk_docs(c).1).collect::<Vec<_>>() }).collect()
        };
        let small = chunk_docs(true);
        assert!(!small.is_empty() && small.iter().all(|&n| n <= 8), "vector shard chunks: {small:?}");
        let large = chunk_docs(false);
        assert!(large.iter().any(|&n| n >= 16), "plain shard chunks: {large:?}");
    }

    #[test]
    fn flush_and_idle_seal_allow_statistics_while_segment_io_is_paused() {
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            fail: bool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                    if self.fail { return Err(std::io::Error::other("injected merge write failure").into()); }
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        for (action, fail, sorted) in [(0, false, false), (0, true, false), (1, false, false), (1, true, false), (0, false, true), (0, true, true), (2, false, true), (2, true, true)] {
            let flush = action == 0;
            let tmp = tempfile::tempdir().unwrap();
            let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
            let (entered, wait) = std::sync::mpsc::channel();
            let (resume, release) = std::sync::mpsc::channel();
            let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
                pause: std::sync::atomic::AtomicBool::new(false), fail, entered,
                resume: parking_lot::Mutex::new(release) });
            let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
            core.set_nrt(true);
            core.set_auto_seal(false);
            if sorted { core.set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
                field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: true, max_mode: false, missing_first: false,
            }]))); }
            *s.core.write() = core;
            assert!(s.apply("i", "u", vec![idx("published", "t", WriteCond::default())], false)[0].is_ok());
            s.publish().unwrap();
            assert!(s.apply("i", "u", vec![idx("hidden", "t", WriteCond::default())], false)[0].is_ok());
            let before = s.write_stats();
            dir.pause.store(true, Ordering::Release);
            let flushing = s.clone();
            let worker = std::thread::spawn(move || {
                if flush { flushing.flush() } else if action == 2 { flushing.refresh() } else { flushing.seal_published(); Ok(()) }
            });
            wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
            // The paused write is one chunk's; the other chunks (a flush and
            // a sorted refresh seal the published and the hidden memtable
            // as separate chunks, encoding together) may finish meanwhile
            // and install, which takes the core write lock for microseconds
            // (a try-lock at that instant fails). What must not happen is a
            // core lock held across the paused I/O: a bounded wait tells the
            // two apart (the pause lasts until `resume`, up to 30 s).
            let hold = std::time::Duration::from_secs(5);
            let readable = s.core.try_read_for(hold).map(|core| core.all_leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>());
            let visible = s.core.try_read_for(hold).map(|core| core.leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>());
            let during = s.write_stats();
            resume.send(()).unwrap();
            let result = worker.join().unwrap();
            assert_eq!(readable, Some(2), "statistics blocked behind segment encoding");
            assert_eq!(during, before);
            assert_eq!(visible, Some(1), "maintenance changed refresh visibility");
            assert_eq!(result.is_err(), action != 1 && fail, "{result:?}");
            assert_eq!(s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), if action == 2 && !fail { 2 } else { 1 });
            assert_eq!(s.core.read().all_leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 2);
            if action == 2 && fail {
                s.refresh().unwrap();
                assert_eq!(s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 2, "failed sorted seal must remain retryable");
            }
            if flush && !fail { assert_eq!(s.committed_seq_no(), s.max_seq_no()); }
        }
    }

    #[test]
    fn flush_commit_allows_reads_and_refresh_while_manifest_io_is_paused() {
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            fail: bool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if name == "meta.json" && self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                    if self.fail { return Err(std::io::Error::other("injected merge write failure").into()); }
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        for (flush, fail, sorted) in [(true, false, false), (true, true, false), (true, false, true), (true, true, true)] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
            let (entered, wait) = std::sync::mpsc::channel();
            let (resume, release) = std::sync::mpsc::channel();
            let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
                pause: std::sync::atomic::AtomicBool::new(false), fail, entered,
                resume: parking_lot::Mutex::new(release) });
            let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
            core.set_nrt(true);
            core.set_auto_seal(false);
            if sorted { core.set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![rustic_core::index_sort::IndexSortField {
                field: ID_FIELD, kind: rustic_core::index_sort::IndexSortKind::Keyword, reverse: true, max_mode: false, missing_first: false,
            }]))); }
            *s.core.write() = core;
            assert!(s.apply("i", "u", vec![idx("published", "t", WriteCond::default())], false)[0].is_ok());
            s.publish().unwrap();
            assert!(s.apply("i", "u", vec![idx("hidden", "t", WriteCond::default())], false)[0].is_ok());
            let before = s.write_stats();
            let committed_before = s.committed_seq_no();
            dir.pause.store(true, Ordering::Release);
            let flushing = s.clone();
            let worker = std::thread::spawn(move || {
                if flush { flushing.flush() } else { flushing.seal_published(); Ok(()) }
            });
            wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
            let readable = s.core.try_read().map(|core| core.all_leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>());
            let visible = s.core.try_read().map(|core| core.leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>());
            let during = s.write_stats();
            // Test refresh without risking a blocked test thread: the guard
            // proves the operation can acquire its lock while I/O is paused.
            let refreshed = s.core.try_write().map(|mut core| core.publish().unwrap());
            resume.send(()).unwrap();
            let result = worker.join().unwrap();
            assert_eq!(readable, Some(2), "statistics blocked behind manifest I/O");
            assert_eq!(refreshed, Some(true), "refresh blocked behind manifest I/O");
            assert_eq!(during, before);
            assert_eq!(visible, Some(1), "maintenance changed refresh visibility");
            assert_eq!(result.is_err(), flush && fail, "{result:?}");
            assert_eq!(s.core.read().leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 2);
            assert_eq!(s.core.read().all_leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>(), 2);
            if !fail {
                assert_eq!(s.committed_seq_no(), s.max_seq_no());
                let reopened = rustic_core::Index::open_in(dir.clone()).unwrap();
                assert_eq!(reopened.all_leaves().iter().map(|l| l.num_live_docs_latest()).sum::<u32>(), 2);
            } else {
                assert_eq!(s.committed_seq_no(), committed_before);
            }
        }
    }

    #[test]
    fn force_merge_allows_statistics_while_segment_io_is_paused() {
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            fail: bool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(30)).unwrap();
                    if self.fail { return Err(std::io::Error::other("injected merge write failure").into()); }
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        // Pause first the merge output, then the pre-merge flush output.
        // Both used to keep the core write lock for the entire I/O phase.
        for (unsealed_tail, fail) in [(false, false), (true, false), (false, true), (true, true)] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
            let (entered, wait) = std::sync::mpsc::channel();
            let (resume, release) = std::sync::mpsc::channel();
            let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(),
                pause: std::sync::atomic::AtomicBool::new(false), fail, entered,
                resume: parking_lot::Mutex::new(release) });
            let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
            core.set_nrt(true);
            core.adopt_view_slot(s.views.clone()).unwrap();
            *s.core.write() = core;
            for row in 0..2 {
                assert!(s.apply("i", "u", vec![idx(&row.to_string(), "t", WriteCond::default())], false)[0].is_ok());
                s.checkin_pool(None); s.core.write().refresh().unwrap();
            }
            if unsealed_tail {
                assert!(s.apply("i", "u", vec![idx("tail", "t", WriteCond::default())], false)[0].is_ok());
            }
            let before = s.write_stats();
            dir.pause.store(true, Ordering::Release);
            let merging = s.clone();
            let thread = std::thread::spawn(move || merging.force_merge(1, false));
            wait.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
            let readable = s.core.try_read().map(|core| core.all_leaves().iter().map(|leaf| leaf.num_live_docs_latest()).sum::<u32>());
            let current = Counters::get(&s.counters.merge_current);
            let current_docs = Counters::get(&s.counters.merge_current_docs);
            let completed_while_paused = Counters::get(&s.counters.merge_total);
            // Translog / seq_no stats: the merge holds the write state, and
            // publishes the (unchanging) figures instead of blocking readers.
            let asked = std::time::Instant::now();
            let during = s.write_stats();
            let stats_wait = asked.elapsed();
            // Always release the worker, including when the assertion fails.
            resume.send(()).unwrap();
            let result = thread.join().unwrap();
            assert_eq!(result.is_err(), fail, "{result:?}");
            assert!(stats_wait < std::time::Duration::from_secs(1), "write stats waited {stats_wait:?} behind the merge");
            assert_eq!(during, before);
            assert_eq!(current, 1, "force merge was invisible during I/O");
            assert_eq!(current_docs, if unsealed_tail { 3 } else { 2 });
            assert_eq!(completed_while_paused, 0);
            assert_eq!(Counters::get(&s.counters.merge_current), 0);
            assert_eq!(Counters::get(&s.counters.merge_current_docs), 0);
            assert_eq!(Counters::get(&s.counters.merge_total), if fail { 0 } else { 1 });
            if !fail {
                // An already merged shard is a no-op, not another completed merge.
                s.force_merge(1, false).unwrap();
                assert_eq!(Counters::get(&s.counters.merge_total), 1);
                assert_eq!(Counters::get(&s.counters.merge_current), 0);
                assert_eq!(Counters::get(&s.counters.merge_current_docs), 0);
            }
            assert_eq!(readable, Some(if unsealed_tail { 3 } else { 2 }), "core locked during segment I/O (tail={unsealed_tail})");
        }
    }

    #[test]
    fn merge_encoder_panic_releases_claims_and_counters() {
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PanicDirectory { inner: RamDirectory, fail: std::sync::atomic::AtomicBool }
        impl Directory for PanicDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, bytes: &[u8]) -> rustic_core::error::Result<()> {
                assert!(!self.fail.swap(false, Ordering::AcqRel), "injected encoder panic");
                self.inner.write(name, bytes)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        // Seals use the same failure discipline, including the sorted refresh
        // path. A failed encode must retain the writer and release its claim.
        for sorted in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            let dir = Arc::new(PanicDirectory { inner: RamDirectory::new(), fail: false.into() });
            let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
            core.set_nrt(true);
            core.set_auto_seal(false);
            if sorted {
                core.set_index_sort(Some(rustic_core::index_sort::IndexSort::new(vec![
                    rustic_core::index_sort::IndexSortField { field: ID_FIELD,
                        kind: rustic_core::index_sort::IndexSortKind::Keyword,
                        reverse: true, max_mode: false, missing_first: false },
                ])));
            }
            *s.core.write() = core;
            for row in 0..4 {
                assert!(s.apply("i", "u", vec![idx(&row.to_string(), "text", WriteCond::default())], false)[0].is_ok());
            }
            dir.fail.store(true, Ordering::Release);
            s.checkin_pool(None);
            let job = s.core.write().begin_seal(true).unwrap().unwrap();
            let done = job.run();
            let err = s.core.write().finish_seal(done).unwrap_err();
            assert!(err.to_string().contains("seal encoder panicked"), "{err}");
            assert!(!s.core.read().seal_in_flight());
            let retry = s.core.write().begin_seal(true).unwrap().expect("panic must release seal claim");
            let done = retry.run();
            s.core.write().finish_seal(done).unwrap();
            s.refresh().unwrap();
            assert_eq!(s.core.read().num_live_docs(), 4);
        }
        for force in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            let dir = Arc::new(PanicDirectory { inner: RamDirectory::new(), fail: false.into() });
            let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
            core.set_nrt(true);
            core.adopt_view_slot(s.views.clone()).unwrap();
            *s.core.write() = core;
            for row in 0..24 {
                assert!(s.apply("i", "u", vec![idx(&row.to_string(), "text", WriteCond::default())], false)[0].is_ok());
                s.checkin_pool(None); s.core.write().refresh().unwrap();
                s.core.write().seal().unwrap();
            }
            let before = s.core.read().num_segments();
            assert!(before > 1);
            dir.fail.store(true, Ordering::Release);
            if force {
                let err = s.force_merge_with(1, false, false).unwrap_err();
                assert!(err.to_string().contains("injected encoder panic"), "{err}");
            } else {
                let task = s.start_merge_waiting().expect("background merge candidate");
                let done = task.run();
                assert!(done.merged.as_ref().err().unwrap().to_string().contains("injected encoder panic"));
                assert!(!s.finish_merge(done));
                assert_eq!(Counters::get(&s.counters.merge_current), 0);
                assert_eq!(Counters::get(&s.counters.merge_current_docs), 0);
                let retry = s.start_merge_waiting().expect("failed inputs must be eligible again");
                assert!(s.finish_merge(retry.run()));
                assert_eq!(Counters::get(&s.counters.merge_current), 0);
                assert!(s.core.read().num_segments() < before);
            }
            assert_eq!(s.core.read().num_live_docs(), 24);
            if force {
                assert_eq!(Counters::get(&s.counters.merge_current), 0);
                assert_eq!(Counters::get(&s.counters.merge_current_docs), 0);
                s.force_merge_with(1, false, false).unwrap();
                assert_eq!(s.core.read().num_segments(), 1);
            }
        }
    }

    /// `max_num_segments` unset / -1 (`max_segments` 0) runs the merges the
    /// policy wants (ES `maybeMerge`) and returns once they are done (and
    /// committed): 15 small segments end as the policy's result, not one
    /// segment. A positive count still merges down to it.
    #[test]
    fn force_merge_without_a_segment_count_only_runs_policy_merges() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        // One sealed, published segment per round (as the busy-slots test).
        for round in 0..15 {
            s.apply("i", "u", vec![idx(&format!("d{round}"), "t", WriteCond::default())], false).remove(0).unwrap();
            s.refresh().unwrap();
            s.seal_all().unwrap();
        }
        let before = s.core.read().num_segments();
        assert!(before >= 15, "{before} segments");
        assert_eq!(s.core.read().num_live_docs(), 15);
        s.force_merge_with(0, false, true).unwrap();
        let after = s.core.read().num_segments();
        assert!(after > 1 && after < before, "{after} segments: the policy's merges, not a full merge");
        assert_eq!(s.core.read().num_live_docs(), 15);
        s.force_merge_with(1, false, true).unwrap();
        assert_eq!(s.core.read().num_segments(), 1);
        assert_eq!(s.core.read().num_live_docs(), 15);
        for round in 0..15 {
            assert!(s.get(&format!("d{round}")).unwrap().is_some(), "d{round}");
        }
    }

    #[test]
    fn force_merge_preserves_unpublished_tail_and_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for round in 0..2 {
            s.apply("i", "u", vec![idx(&format!("sealed-{round}"), "t", WriteCond::default())], false);
            s.checkin_pool(None); s.core.write().refresh().unwrap();
        }
        s.apply("i", "u", vec![idx("unpublished", "t", WriteCond::default())], false);
        s.force_merge(1, false).unwrap();
        assert_eq!(s.core.read().num_segments(), 1);
        assert_eq!(s.core.read().num_live_docs(), 3);
        assert!(s.get("unpublished").unwrap().is_some());
        drop(s);
        let (s, replay) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(replay.is_empty());
        assert_eq!(s.core.read().num_live_docs(), 3);
        assert!(s.get("unpublished").unwrap().is_some());
    }

    #[test]
    fn force_merge_rebuilds() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        for round in 0..3 {
            let ops = (0..10).map(|i| idx(&format!("{round}-{i}"), "t", WriteCond::default())).collect();
            s.apply("i", "u", ops, false);
            s.checkin_pool(None); s.core.write().refresh().unwrap();
        }
        s.apply("i", "u", vec![WriteOp::Delete { id: "0-0".into(), cond: WriteCond::default() }], false);
        assert_eq!(s.core.read().num_segments(), 3);
        s.force_merge(1, false).unwrap();
        let core = s.core.read();
        assert_eq!(core.num_segments(), 1);
        assert_eq!(core.num_live_docs(), 29);
        drop(core);
        assert!(s.get("2-9").unwrap().is_some());
        drop(s);
        let (s, ops) = Shard::open(tmp.path(), schema()).unwrap();
        assert!(ops.is_empty());
        assert_eq!(s.core.read().num_live_docs(), 29);
    }

    /// Stage 3: writes, refreshes and searches proceed while a flush is in
    /// its commit I/O (here paused at the `meta.json` write); the commit
    /// covers exactly the ops acknowledged before its capture, and the
    /// writes made during it stay above the commit point (still in the
    /// retained translog).
    #[test]
    fn writes_refresh_and_search_proceed_during_flush_commit() {
        use rustic_core::codec::directory::{Directory, FileSlice, RamDirectory};
        #[derive(Debug)]
        struct PausedDirectory {
            inner: RamDirectory,
            pause: std::sync::atomic::AtomicBool,
            entered: std::sync::mpsc::Sender<()>,
            resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Directory for PausedDirectory {
            fn open_read(&self, name: &str) -> rustic_core::error::Result<FileSlice> { self.inner.open_read(name) }
            fn write(&self, name: &str, data: &[u8]) -> rustic_core::error::Result<()> {
                if name == "meta.json" && self.pause.swap(false, Ordering::AcqRel) {
                    self.entered.send(()).unwrap();
                    self.resume.lock().recv_timeout(std::time::Duration::from_secs(60)).unwrap();
                }
                self.inner.write(name, data)
            }
            fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
            fn delete(&self, name: &str) -> rustic_core::error::Result<()> { self.inner.delete(name) }
            fn list(&self) -> rustic_core::error::Result<Vec<String>> { self.inner.list() }
            fn list_admitted(&self, temporary: bool) -> rustic_core::error::Result<rustic_core::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        }
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(Shard::create(tmp.path(), "i", schema()).unwrap());
        let (entered, wait) = std::sync::mpsc::channel();
        let (resume, release) = std::sync::mpsc::channel();
        let dir = Arc::new(PausedDirectory { inner: RamDirectory::new(), pause: std::sync::atomic::AtomicBool::new(false),
            entered, resume: parking_lot::Mutex::new(release) });
        let mut core = rustic_core::Index::create_in(dir.clone(), "i", schema(), Default::default()).unwrap();
        core.set_nrt(true);
        core.set_auto_seal(false);
        core.adopt_view_slot(s.views.clone()).unwrap();
        *s.core.write() = core;
        for row in 0..10 {
            assert!(s.apply("i", "u", vec![idx(&format!("before{row}"), "t", WriteCond::default())], false)[0].is_ok());
        }
        s.publish().unwrap();
        let captured_max = s.max_seq_no();
        dir.pause.store(true, Ordering::Release);
        let flushing = s.clone();
        let worker = std::thread::spawn(move || flushing.flush());
        wait.recv_timeout(std::time::Duration::from_secs(30)).expect("flush never reached its commit");
        // While the commit I/O is paused: write, refresh, search.
        let (tx, rx) = std::sync::mpsc::channel();
        let during = s.clone();
        std::thread::spawn(move || {
            let w = during.apply("i", "u", vec![idx("during", "t", WriteCond::default())], false);
            let refreshed = during.publish();
            let live = during.reader_view().num_live_docs();
            let _ = tx.send((w[0].is_ok(), refreshed.is_ok(), live));
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(20));
        resume.send(()).unwrap();
        worker.join().unwrap().unwrap();
        assert_eq!(got.expect("a write, refresh or search waited for the flush commit"), (true, true, 11));
        // The commit point is the capture: the write made during the commit
        // is above it.
        assert_eq!(s.committed_seq_no(), captured_max);
        assert!(s.max_seq_no() > captured_max);
        s.flush().unwrap();
        assert_eq!(s.committed_seq_no(), s.max_seq_no());
    }
}

#[cfg(test)]
mod core_write_error_tests {
    use super::*;

    /// A write item refused by the core's memory admission fails with 429
    /// `circuit_breaking_exception` and ES's fields (the kNN agent's bulk
    /// past box memory got 500 "Memory allocation failed: [core] [mutable
    /// row plane] ..."); any other core failure stays a 500.
    #[test]
    fn admission_refusals_are_429_and_other_core_errors_500() {
        let refused = rustic_core::RusticError::Memory("[core] [mutable row plane] needs 1048576 bytes, retained 30000000000, limit 30064771072".into());
        let e = core_write_error(&refused);
        assert_eq!((e.status, e.kind.as_str()), (429, "circuit_breaking_exception"));
        let v = e.to_json();
        assert_eq!(v["error"]["bytes_wanted"], 1048576);
        assert_eq!(v["error"]["bytes_limit"], 30064771072u64);
        assert_eq!(v["error"]["durability"], "TRANSIENT");
        assert!(v["error"]["reason"].as_str().unwrap().contains("[mutable row plane]"));
        let overflow = core_write_error(&rustic_core::RusticError::Memory("batch document count overflow".into()));
        assert_eq!((overflow.status, overflow.to_json()["error"]["durability"].clone()), (429, serde_json::json!("PERMANENT")));
        let corrupt = core_write_error(&rustic_core::RusticError::Corrupted("bad chunk".into()));
        assert_eq!((corrupt.status, corrupt.kind.as_str()), (500, "exception"));
        assert_eq!(corrupt.reason, "Corrupted index: bad chunk");
    }
}
