//! The replica write-ahead log (`<shard path>/cluster-replica-wal/`).
//!
//! Needed because the engine's replay path (`Shard::replay_index` /
//! `replay_delete`) does not write the engine translog: without this log a
//! replica would lose every op applied since its last flush on a crash.
//!
//! Files `wal-<gen:08>.log`, records `[len u32 LE][crc32 u32 LE][bincode
//! WalRecord]`; one `write(2)` + `fsync` per appended batch. A torn tail on
//! the newest generation is truncated on open. `state.json` holds the
//! durable checkpoint floor (everything `<=` it is committed in the engine)
//! and the highest primary term seen.
//!
//! Replay streams: one record (one replicated batch, admitted when it was
//! written) is decoded at a time and handed to the caller, which applies or
//! filters its ops before the next is read. Nothing holds the whole log, so
//! a large unflushed WAL never needs a request-breaker reservation of its
//! size (ES streams its translog too). The records were acknowledged: a
//! replay runs as an `elasticsearch_engine::memory::Replay`, whose breaker
//! reservations are accounted but never refused or waited for (ES applies
//! translog recovery with no circuit breaker), so a record larger than the
//! breaker still replays. Only real failures (I/O, corruption) fail it.

use super::frame::{self, Decoded};
use super::ops::ReplicaOp;
use super::{write_atomic, ReplicationError, Result};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeSeed;
use elasticsearch_engine::translog::AdmittedBatch;
use elasticsearch_engine::memory::breakers;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const WAL_DIR: &str = "cluster-replica-wal";
const STATE_FILE: &str = "state.json";
const COMPRESSED_MAGIC: &[u8; 4] = b"RWZ\x01";
const MAX_COMPRESSED_BODY: usize = 256 * 1024 * 1024;

fn corrupt(detail: impl Into<String>) -> ReplicationError {
    ReplicationError::Corrupt { what: "replica wal".into(), detail: detail.into() }
}

/// An envelope makes old binaries reject state before reading (and potentially
/// truncating) an unfamiliar compressed log. Legacy bare state remains readable.
fn decode_state(bytes: &[u8]) -> Result<WalState> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    if let Some(version) = value.get("wal_format") {
        if version.as_u64() != Some(2) { return Err(corrupt("unsupported WAL state format")); }
        Ok(serde_json::from_value(value.get("state").cloned().ok_or_else(|| corrupt("missing WAL state"))?)?)
    } else { Ok(serde_json::from_value(value)?) }
}

fn persist_state(dir: &Path, state: &WalState) -> Result<()> {
    let bytes = serde_json::to_vec(&serde_json::json!({"wal_format": 2, "state": state}))?;
    write_atomic(dir, STATE_FILE, &bytes)?;
    // The format gate must survive a crash before compressed records are written.
    elasticsearch_engine::rustic_core::platform::sync_dir(dir)?;
    Ok(())
}

struct EncodedRecord {
    bytes: Vec<u8>,
    _memory: elasticsearch_engine::memory::Reservation,
}
impl std::ops::Deref for EncodedRecord {
    type Target = [u8];
    fn deref(&self) -> &[u8] { &self.bytes }
}
fn buffer(size: usize, label: &str) -> Result<EncodedRecord> {
    let memory = breakers().reserve_request(size as u64, label)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("WAL allocation: {e}")))?;
    Ok(EncodedRecord { bytes, _memory: memory })
}
fn encode_record(rec: &WalRecord) -> Result<EncodedRecord> {
    let size = usize::try_from(bincode::serialized_size(rec)?).map_err(|_| corrupt("WAL encoding size overflow"))?;
    let mut body = buffer(size, "replica WAL serialized record")?;
    struct Fixed<'a>(&'a mut Vec<u8>);
    impl Write for Fixed<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.0.capacity() - self.0.len() { return Err(std::io::Error::other("WAL serialization exceeded admitted size")); }
            self.0.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    bincode::serialize_into(Fixed(&mut body.bytes), rec)?;
    if (4096..=MAX_COMPRESSED_BODY).contains(&body.len()) {
        let bound = lz4_flex::block::get_maximum_output_size(body.len()).checked_add(8).ok_or_else(|| corrupt("WAL compression size overflow"))?;
        let mut encoded = buffer(bound, "replica WAL compression output")?;
        encoded.bytes.resize(bound, 0);
        encoded.bytes[..4].copy_from_slice(COMPRESSED_MAGIC);
        encoded.bytes[4..8].copy_from_slice(&(body.len() as u32).to_le_bytes());
        let written = lz4_flex::block::compress_into(&body, &mut encoded.bytes[8..]).map_err(|e| corrupt(format!("WAL compression: {e}")))?;
        encoded.bytes.truncate(8 + written);
        if encoded.len() < body.len() { return Ok(encoded); }
    }
    Ok(body)
}

fn decode_record(body: &[u8]) -> Result<WalRecord> {
    if body.starts_with(b"RWZ") {
        if body.get(..4) != Some(COMPRESSED_MAGIC.as_slice()) || body.len() < 8 {
            return Err(corrupt("unsupported compressed WAL record"));
        }
        let size = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        if size > MAX_COMPRESSED_BODY { return Err(corrupt("compressed WAL size exceeds limit")); }
        let _memory = breakers().reserve_request(size as u64, "replica WAL decompression")?;
        let mut decoded = Vec::new();
        decoded.try_reserve_exact(size).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("WAL decompression allocation: {e}")))?;
        decoded.resize(size, 0);
        let actual = lz4_flex::block::decompress_into(&body[8..], &mut decoded).map_err(|e| corrupt(format!("compressed WAL: {e}")))?;
        if actual != size { return Err(corrupt("compressed WAL size mismatch")); }
        return bincode::deserialize(&decoded).map_err(decoded_error);
    }
    bincode::deserialize(body).map_err(decoded_error)
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalRecord {
    /// A batch of ops, with the global checkpoint known when it was
    /// accepted (persisted in the same fsync as the ops).
    Ops { #[serde(deserialize_with="decode_ops")] ops: AdmittedBatch<ReplicaOp>, global_checkpoint: i64 },
    /// Every seq_no `<=` checkpoint is reflected in the engine (snapshot
    /// recovery completed).
    FastForward(i64),
    /// The global checkpoint advanced without ops.
    GlobalCheckpoint(i64),
}

pub(super) fn decode_ops<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<AdmittedBatch<ReplicaOp>, D::Error> {
    super::decode::Operations(breakers()).deserialize(d)
}
fn decoded_error(error: bincode::Error) -> ReplicationError {
    if let bincode::ErrorKind::Custom(message) = &*error {
        if let Some(reason) = message.strip_prefix(&elasticsearch_transport::decode_admission_error("")) {
            return elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", reason).into();
        }
    }
    corrupt(format!("WAL payload: {error}"))
}
impl WalRecord {
    pub fn try_clone(&self) -> elasticsearch_engine::error::Result<Self> {
        Ok(match self {
            Self::Ops { ops, global_checkpoint } => Self::Ops { ops: ReplicaOp::try_clone_batch(ops)?, global_checkpoint: *global_checkpoint },
            Self::FastForward(value) => Self::FastForward(*value),
            Self::GlobalCheckpoint(value) => Self::GlobalCheckpoint(*value),
        })
    }

    fn max_seq_no(&self) -> i64 {
        match self {
            WalRecord::Ops { ops, .. } => ops.iter().map(ReplicaOp::seq_no).max().unwrap_or(-1),
            WalRecord::FastForward(s) => *s,
            WalRecord::GlobalCheckpoint(_) => -1,
        }
    }
}

/// Durable per-copy state (`state.json`, written atomically + fsynced).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalState {
    /// Every op `<=` this is committed in the engine (no need to replay).
    pub checkpoint_floor: i64,
    /// Highest primary term accepted (fencing).
    pub primary_term: u64,
    /// Global checkpoint as of the last trim (records carry newer ones).
    #[serde(default = "unassigned")]
    pub global_checkpoint: i64,
    /// A higher term was seen while this copy held ops above this global
    /// checkpoint: those ops must be verified by a resync.
    #[serde(default)]
    pub resync_pending_gcp: Option<i64>,
    /// The copy diverged from the primary; it must be wiped and recovered.
    #[serde(default)]
    pub needs_full_recovery: Option<String>,
    /// A promotion to this term was adopted (written with the term itself)
    /// but has not finished: gap fill, resync collection and the WAL trim
    /// are still owed. Promotion runs again under the same term until it
    /// clears this (also after a crash).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion_pending: Option<u64>,
}

fn unassigned() -> i64 {
    super::checkpoint::UNASSIGNED_SEQ_NO
}

impl Default for WalState {
    fn default() -> Self {
        WalState { checkpoint_floor: -1, primary_term: 0, global_checkpoint: unassigned(), resync_pending_gcp: None, needs_full_recovery: None, promotion_pending: None }
    }
}

pub struct ReplicaWal {
    dir: PathBuf,
    generation: u64,
    file: WalFile,
    /// Records appended so far (all generations): group-commit marks.
    appended: u64,
    /// Highest mark known durable ([`WalSync::wait`]).
    synced: std::sync::Arc<parking_lot::Mutex<u64>>,
    /// Sticky fsync failure (see [`SyncFailure`]).
    failure: SyncFailure,
    /// generation -> highest seq_no recorded in it (-1 = none).
    gens: BTreeMap<u64, i64>,
    state: WalState,
    /// Replayed since open: a torn tail is cut off and the generations'
    /// seq_no ranges are known. Appends wait for it.
    replayed: bool,
}

fn file_name(g: u64) -> String {
    format!("wal-{g:08}.log")
}

/// The first fsync error of a WAL, sticky ("fsyncgate"): after a failed
/// fsync the kernel may have dropped the dirty pages and cleared the error,
/// so a later fsync can succeed without the records being on disk. Once
/// one fails, every later wait and append fails too (the copy is failed and
/// recovered, as ES fails the engine on a translog tragic event).
#[derive(Clone, Default)]
struct SyncFailure {
    error: std::sync::Arc<parking_lot::Mutex<Option<String>>>,
    /// Test seam ([`ReplicaWal::fail_next_syncs`]): fsyncs still to fail.
    #[cfg(test)]
    inject: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl SyncFailure {
    fn check(&self) -> Result<()> {
        match &*self.error.lock() {
            Some(error) => Err(ReplicationError::IllegalState(format!("replica WAL failed: an earlier fsync failed: {error}"))),
            None => Ok(()),
        }
    }

    /// `file.sync_data()`, recording a failure for good.
    fn sync(&self, file: &File) -> Result<()> {
        self.check()?;
        #[cfg(test)]
        let injected = self.inject.fetch_update(std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire, |n| n.checked_sub(1)).is_ok();
        #[cfg(not(test))]
        let injected = false;
        let result = if injected { Err(std::io::Error::other("injected replica WAL fsync failure")) } else { file.sync_data() };
        if let Err(error) = result {
            let mut sticky = self.error.lock();
            if sticky.is_none() {
                tracing::error!("replica WAL fsync failed, failing the copy: {error}");
                *sticky = Some(error.to_string());
            }
            return Err(error.into());
        }
        Ok(())
    }
}

/// The current generation's file and the mark of its last append.
#[derive(Clone)]
struct WalFile {
    file: std::sync::Arc<File>,
    last: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl WalFile {
    fn open(path: &Path) -> Result<WalFile> {
        Ok(WalFile { file: std::sync::Arc::new(OpenOptions::new().create(true).append(true).open(path)?),
            last: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)) })
    }
}

/// Durability of an append made with [`ReplicaWal::append_deferred`]:
/// [`WalSync::wait`] makes it durable, sharing one fsync with every
/// concurrent append to the same file (group commit).
pub struct WalSync {
    file: WalFile,
    mark: u64,
    synced: std::sync::Arc<parking_lot::Mutex<u64>>,
    failure: SyncFailure,
}

impl WalSync {
    /// Durable once this returns. Call without the replica's apply lock.
    pub fn wait(&self) -> Result<()> {
        let mut synced = self.synced.lock();
        // A failed fsync anywhere fails every waiter, also one whose mark an
        // earlier (successful) fsync covered: it is not reported durable
        // while the copy is failing.
        self.failure.check()?;
        if *synced >= self.mark {
            return Ok(());
        }
        // Everything appended to this file so far is covered by its fsync.
        let target = self.file.last.load(std::sync::atomic::Ordering::Acquire);
        self.failure.sync(&self.file.file)?;
        *synced = (*synced).max(target);
        Ok(())
    }
}

fn parse_generation(name: &str) -> Option<u64> {
    name.strip_prefix("wal-")?.strip_suffix(".log")?.parse().ok()
}

/// The record framed at `pos`, and where the next one starts. `None` at the
/// end of the valid prefix (a torn or corrupt tail starts there).
fn read_frame(file: &mut File, pos: usize, total: usize) -> Result<Option<(WalRecord, usize)>> {
    if total.saturating_sub(pos) < frame::HEADER_LEN { return Ok(None); }
    file.seek(SeekFrom::Start(pos as u64))?;
    let mut header = [0u8; frame::HEADER_LEN];
    file.read_exact(&mut header)?;
    let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let Some(end) = pos.checked_add(frame::HEADER_LEN).and_then(|v| v.checked_add(len)) else { return Ok(None) };
    if len == 0 || end > total { return Ok(None); }
    let _memory = breakers().reserve_request((frame::HEADER_LEN + len) as u64, "replica WAL encoded frame")?;
    let mut data = Vec::new();
    data.try_reserve_exact(frame::HEADER_LEN + len).map_err(|e| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("WAL frame allocation: {e}")))?;
    data.extend_from_slice(&header); data.resize(frame::HEADER_LEN + len, 0);
    file.read_exact(&mut data[frame::HEADER_LEN..])?;
    let Decoded::Record(body, _) = frame::decode(&data) else { return Ok(None) };
    // The record owns its decoder reservations; the frame buffer goes.
    Ok(Some((decode_record(body)?, end)))
}

/// Hand every record of `path` to `f`, one at a time, in append order.
/// Returns (end of the valid prefix, file length).
fn scan_records(path: &Path, f: &mut dyn FnMut(WalRecord) -> Result<()>) -> Result<(usize, usize)> {
    let mut file = File::open(path)?;
    let total = usize::try_from(file.metadata()?.len()).map_err(|_| corrupt("WAL file length overflow"))?;
    let mut pos = 0usize;
    while let Some((record, end)) = read_frame(&mut file, pos, total)? {
        f(record)?;
        pos = end;
    }
    Ok((pos, total))
}

/// The generations a replay reads ([`ReplicaWal::replay_plan`]), taken out
/// of the WAL so the caller can apply records to state that owns it.
pub struct ReplayPlan {
    generations: Vec<(u64, PathBuf)>,
}

/// What a replay learned ([`ReplicaWal::replayed`]).
pub struct ReplayOutcome {
    maxes: Vec<(u64, i64)>,
    pub records: u64,
}

impl ReplayPlan {
    /// Stream every record to `f`; a torn tail on the newest generation is
    /// cut off (a crash mid-append), a corrupt older one fails the replay.
    /// Reading and `f` run as a replay of acknowledged ops: no circuit
    /// breaker refuses or delays them (module docs).
    pub fn run(&self, f: impl FnMut(WalRecord) -> Result<()>) -> Result<ReplayOutcome> {
        elasticsearch_engine::memory::replaying("replica WAL replay", || self.run_replay(f))
    }

    fn run_replay(&self, mut f: impl FnMut(WalRecord) -> Result<()>) -> Result<ReplayOutcome> {
        let last = self.generations.last().map(|(g, _)| *g);
        let mut outcome = ReplayOutcome { maxes: Vec::new(), records: 0 };
        for (g, path) in &self.generations {
            let mut max = -1;
            let (good, total) = scan_records(path, &mut |record| {
                max = max.max(record.max_seq_no());
                outcome.records += 1;
                f(record)
            })?;
            if good != total {
                if Some(*g) == last {
                    let file = OpenOptions::new().write(true).open(path)?;
                    file.set_len(good as u64)?;
                    file.sync_all()?;
                } else {
                    return Err(ReplicationError::Corrupt { what: "replica wal".into(), detail: format!("generation {g} is corrupt at offset {good}") });
                }
            }
            outcome.maxes.push((*g, max));
        }
        Ok(outcome)
    }
}

/// Delete the WAL under `shard_path` (state included), durably.
pub fn remove(shard_path: &Path) -> Result<()> {
    match std::fs::remove_dir_all(shard_path.join(WAL_DIR)) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    super::sync_dir(shard_path);
    Ok(())
}

impl ReplicaWal {
    /// Open (or create) the WAL under `shard_path`. Nothing is read beyond
    /// its state: replay it ([`replay_plan`](Self::replay_plan) +
    /// [`replayed`](Self::replayed), or [`replay`](Self::replay)) before
    /// appending, which also cuts off a torn tail.
    pub fn open(shard_path: &Path) -> Result<ReplicaWal> {
        let dir = shard_path.join(WAL_DIR);
        std::fs::create_dir_all(&dir)?;
        let state: WalState = match std::fs::read(dir.join(STATE_FILE)) {
            Ok(b) => decode_state(&b)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => WalState::default(),
            Err(e) => return Err(e.into()),
        };
        let mut gens = BTreeMap::new();
        for e in std::fs::read_dir(&dir)? {
            let e = e?;
            if let Some(g) = e.file_name().to_str().and_then(parse_generation) {
                // Unknown until replayed: never trimmed before then.
                gens.insert(g, i64::MAX);
            }
        }
        persist_state(&dir, &state)?;
        let generation = gens.keys().next_back().copied().unwrap_or(1);
        gens.entry(generation).or_insert(-1);
        let file = WalFile::open(&dir.join(file_name(generation)))?;
        super::sync_dir(&dir);
        Ok(ReplicaWal { dir, generation, file, appended: 0, synced: std::sync::Arc::new(parking_lot::Mutex::new(0)), failure: SyncFailure::default(), gens, state, replayed: false })
    }

    /// The generations to replay, oldest first.
    pub fn replay_plan(&self) -> ReplayPlan {
        ReplayPlan { generations: self.gens.keys().map(|g| (*g, self.dir.join(file_name(*g)))).filter(|(_, path)| path.exists()).collect() }
    }

    /// Record what a replay of [`replay_plan`](Self::replay_plan) found;
    /// appends are allowed from here on.
    pub fn replayed(&mut self, outcome: &ReplayOutcome) {
        for (g, max) in &outcome.maxes {
            if let Some(known) = self.gens.get_mut(g) { *known = *max; }
        }
        for known in self.gens.values_mut() {
            if *known == i64::MAX { *known = -1; }
        }
        self.replayed = true;
    }

    /// Replay every record to `f` (see [`ReplayPlan::run`]).
    pub fn replay(&mut self, f: impl FnMut(WalRecord) -> Result<()>) -> Result<u64> {
        let outcome = self.replay_plan().run(f)?;
        self.replayed(&outcome);
        Ok(outcome.records)
    }

    pub fn state(&self) -> &WalState {
        &self.state
    }

    /// Durably append one record (write + fsync).
    pub fn append(&mut self, rec: &WalRecord) -> Result<()> {
        self.append_deferred(rec)?.wait()
    }

    /// Append one record (one `write(2)`, no fsync): durable once the
    /// returned [`WalSync`] is waited for (outside the caller's lock, so
    /// concurrent batches share fsyncs).
    pub fn append_deferred(&mut self, rec: &WalRecord) -> Result<WalSync> {
        self.failure.check()?;
        if !self.replayed {
            return Err(ReplicationError::IllegalState("replica WAL appended before it was replayed".into()));
        }
        let body = encode_record(rec)?;
        let mut buf = buffer(body.len().checked_add(frame::HEADER_LEN).ok_or_else(|| corrupt("WAL frame size overflow"))?, "replica WAL encoded output")?;
        frame::encode(&body, &mut buf.bytes);
        (&*self.file.file).write_all(&buf)?;
        self.appended += 1;
        self.file.last.store(self.appended, std::sync::atomic::Ordering::Release);
        let m = self.gens.entry(self.generation).or_insert(-1);
        *m = (*m).max(rec.max_seq_no());
        Ok(WalSync { file: self.file.clone(), mark: self.appended, synced: self.synced.clone(), failure: self.failure.clone() })
    }

    /// Test seam: the next `count` WAL fsyncs fail.
    #[cfg(test)]
    pub(crate) fn fail_next_syncs(&self, count: usize) {
        self.failure.inject.store(count, std::sync::atomic::Ordering::Release);
    }

    /// Change the durable state; persisted (fsynced) before returning.
    pub fn update_state(&mut self, f: impl FnOnce(&mut WalState)) -> Result<()> {
        let mut next = self.state.clone();
        f(&mut next);
        if next != self.state {
            persist_state(&self.dir, &next)?;
            self.state = next;
        }
        Ok(())
    }

    /// Hand every record still on disk to `f`, one at a time, in append
    /// order (re-read from the files, which hold exactly what was fsynced).
    /// The records were acknowledged: reading them (and `f`) is never
    /// refused or delayed by a circuit breaker (module docs).
    pub fn for_each_record(&self, mut f: impl FnMut(WalRecord) -> Result<()>) -> Result<()> {
        elasticsearch_engine::memory::replaying("replica WAL read", || {
            for g in self.gens.keys() {
                let path = self.dir.join(file_name(*g));
                if path.exists() {
                    scan_records(&path, &mut f)?;
                }
            }
            Ok(())
        })
    }

    /// Every record still on disk, in append order.
    #[cfg(test)]
    pub fn read_all(&self) -> Result<Vec<WalRecord>> {
        let mut out = Vec::new();
        self.for_each_record(|record| { out.push(record); Ok(()) })?;
        Ok(out)
    }

    /// Record that every op `< below_seq_no` is committed in the engine
    /// (and the global checkpoint `global_checkpoint`), roll to a new
    /// generation and delete the generations that only hold ops below it.
    pub fn trim_below(&mut self, below_seq_no: i64, global_checkpoint: i64) -> Result<()> {
        let floor = below_seq_no - 1;
        self.update_state(|s| {
            s.checkpoint_floor = s.checkpoint_floor.max(floor);
            s.global_checkpoint = s.global_checkpoint.max(global_checkpoint);
        })?;
        if self.gens.get(&self.generation).copied().unwrap_or(-1) >= 0 {
            let next = self.generation + 1;
            // Appends to the generation being left are durable from here on
            // (their pending waits find the mark synced).
            self.failure.sync(&self.file.file)?;
            {
                let mut synced = self.synced.lock();
                *synced = (*synced).max(self.appended);
            }
            self.file = WalFile::open(&self.dir.join(file_name(next)))?;
            super::sync_dir(&self.dir);
            self.generation = next;
            self.gens.insert(next, -1);
        }
        let doomed: Vec<u64> = self.gens.iter().filter(|(g, max)| **g != self.generation && **max < below_seq_no).map(|(g, _)| *g).collect();
        for g in doomed {
            match std::fs::remove_file(self.dir.join(file_name(g))) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            self.gens.remove(&g);
        }
        Ok(())
    }

    pub fn num_generations(&self) -> usize {
        self.gens.len()
    }

    pub fn size_bytes(&self) -> u64 {
        self.gens.keys().filter_map(|g| std::fs::metadata(self.dir.join(file_name(*g))).ok()).map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Open and replay, collecting every record.
    fn open_all(path: &Path) -> Result<(ReplicaWal, Vec<WalRecord>)> {
        let mut wal = ReplicaWal::open(path)?;
        let mut records = Vec::new();
        wal.replay(|record| { records.push(record); Ok(()) })?;
        Ok((wal, records))
    }

    fn idx(seq: i64) -> ReplicaOp {
        ReplicaOp::Index { id: format!("d{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: 1 }
    }

    /// B8c ("fsyncgate"): once a group-commit fsync fails, a second waiter
    /// whose retried fsync succeeds (the kernel cleared the error) must not
    /// report its record durable, and nothing more is appended.
    #[test]
    fn a_failed_wal_fsync_fails_every_later_wait_and_append() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut wal, _) = open_all(tmp.path()).unwrap();
        let a = wal.append_deferred(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(0)]), global_checkpoint: -1 }).unwrap();
        let b = wal.append_deferred(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(1)]), global_checkpoint: -1 }).unwrap();
        wal.fail_next_syncs(1);
        assert!(a.wait().is_err());
        assert!(b.wait().is_err(), "a retried fsync acknowledged a record after an fsync error");
        assert!(wal.append(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(2)]), global_checkpoint: -1 }).is_err());
        assert!(wal.append_deferred(&WalRecord::GlobalCheckpoint(0)).is_err());
    }

    /// Replay never waits on or is refused by the request breaker (ES
    /// applies translog recovery with no breaker): a record larger than the
    /// whole breaker, while other work holds all of it, replays at once.
    /// Before, the replay waited a minute for memory and then failed the
    /// copy, which recovered from its primary.
    #[test]
    fn a_record_larger_than_the_breaker_replays_without_waiting() {
        use elasticsearch_engine::memory::{scoped, Breakers, RequestScope};
        let tmp = tempfile::tempdir().unwrap();
        let big = ReplicaOp::Index { id: "big".into(), source_json: format!("{{\"p\":\"{}\"}}", "x".repeat(256 * 1024)), routing: None, seq_no: 0, version: 1, primary_term: 1 };
        let record = WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![big, idx(1)]), global_checkpoint: -1 };
        {
            let (mut wal, _) = open_all(tmp.path()).unwrap();
            wal.append(&record).unwrap();
        }
        let tiny = Breakers::leak_for_tests(1 << 30);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        tiny.request.set_limit(64 * 1024);
        let competing = tiny.reserve_request(64 * 1024, "competing request").unwrap();
        let tripped = tiny.request.tripped();
        let started = std::time::Instant::now();
        // Compared inside the scope: what the replay decoded is charged to it.
        let expected = [vec![record.try_clone().unwrap()], vec![record]];
        scoped(Some(RequestScope::new("WAL replay past the breaker", tiny)), || {
            let (wal, records) = open_all(tmp.path()).expect("an acknowledged record replays past the breaker");
            assert_eq!(records, expected[0]);
            drop(records);
            let again = wal.read_all().expect("a WAL read is not refused either");
            assert_eq!(again, expected[1]);
        });
        assert!(started.elapsed() < std::time::Duration::from_secs(30), "the replay waited for memory");
        assert_eq!(tiny.request.tripped(), tripped, "nothing was refused");
        assert_eq!(tiny.request.reserved(), 64 * 1024, "only the competing reservation is left");
        drop(competing);
    }

    #[test]
    fn compressed_wal_migrates_legacy_and_recovers_mixed_records() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(WAL_DIR);
        fs::create_dir_all(&dir).unwrap();
        let state = WalState { primary_term: 3, ..Default::default() };
        fs::write(dir.join(STATE_FILE), serde_json::to_vec(&state).unwrap()).unwrap();
        let legacy = WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(0)]), global_checkpoint: -1 };
        let mut bytes = Vec::new();
        frame::encode(&bincode::serialize(&legacy).unwrap(), &mut bytes);
        fs::write(dir.join(file_name(1)), bytes).unwrap();
        let big = WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test((1..400).map(idx).collect()), global_checkpoint: 0 };
        assert!(encode_record(&big).unwrap().starts_with(COMPRESSED_MAGIC));
        {
            let (mut wal, records) = open_all(tmp.path()).unwrap();
            assert_eq!(records, vec![legacy.try_clone().unwrap()]);
            assert_eq!(wal.state(), &state);
            // The previous binary's bare-state decoder refuses the format gate.
            assert!(serde_json::from_slice::<WalState>(&fs::read(dir.join(STATE_FILE)).unwrap()).is_err());
            wal.append(&big).unwrap();
            wal.append(&WalRecord::GlobalCheckpoint(399)).unwrap();
        }
        let path = dir.join(file_name(1));
        let good_len = fs::metadata(&path).unwrap().len();
        OpenOptions::new().append(true).open(&path).unwrap().write_all(&[10, 0, 0]).unwrap();
        let (mut wal, records) = open_all(tmp.path()).unwrap();
        assert_eq!(records, vec![legacy, big, WalRecord::GlobalCheckpoint(399)]);
        assert_eq!(fs::metadata(&path).unwrap().len(), good_len);
        wal.trim_below(400, 399).unwrap();
        drop(wal);
        assert!(open_all(tmp.path()).unwrap().1.is_empty());
    }

    #[test]
    fn malformed_compression_and_future_formats_fail_without_truncation() {
        let tmp = tempfile::tempdir().unwrap();
        let (wal, _) = open_all(tmp.path()).unwrap();
        drop(wal);
        let dir = tmp.path().join(WAL_DIR);
        let path = dir.join(file_name(1));
        for body in [b"RWZ\x01".to_vec(), [b"RWZ\x01".as_slice(), &u32::MAX.to_le_bytes()].concat(), b"RWZ\x02unsupported".to_vec()] {
            let mut bytes = Vec::new();
            frame::encode(&body, &mut bytes);
            fs::write(&path, &bytes).unwrap();
            assert!(open_all(tmp.path()).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
        fs::write(dir.join(STATE_FILE), br#"{"wal_format":99,"state":{}}"#).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(open_all(tmp.path()).is_err());
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn append_reopen_trim_and_torn_tail() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let (mut w, recs) = open_all(tmp.path()).unwrap();
            assert!(recs.is_empty());
            w.append(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(0), idx(1)]), global_checkpoint: -1 }).unwrap();
            w.append(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(5)]), global_checkpoint: 1 }).unwrap();
            w.trim_below(2, 1).unwrap(); // gen 1 holds seq 5: kept
            assert_eq!(w.num_generations(), 2);
            w.append(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(2)]), global_checkpoint: 1 }).unwrap();
            w.append(&WalRecord::FastForward(4)).unwrap();
            w.update_state(|s| s.primary_term = 3).unwrap();
        }
        // Torn tail on the newest generation.
        let path = tmp.path().join(WAL_DIR).join(file_name(2));
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[9, 0, 0, 0, 1, 2]).unwrap();
        drop(f);
        {
            let (mut w, recs) = open_all(tmp.path()).unwrap();
            assert_eq!(recs.len(), 4);
            assert_eq!(recs[3], WalRecord::FastForward(4));
            assert_eq!(w.state(), &WalState { checkpoint_floor: 1, primary_term: 3, global_checkpoint: 1, ..Default::default() });
            assert_eq!(w.read_all().unwrap().len(), 4);
            w.append(&WalRecord::Ops { ops: super::super::ops::admitted_ops_for_test(vec![idx(6)]), global_checkpoint: 6 }).unwrap();
            w.trim_below(7, 6).unwrap();
            assert_eq!(w.num_generations(), 1);
        }
        let (_, recs) = open_all(tmp.path()).unwrap();
        assert!(recs.is_empty());
    }
}
