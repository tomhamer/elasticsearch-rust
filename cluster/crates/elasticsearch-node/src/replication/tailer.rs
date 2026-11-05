//! Incremental reader of a primary shard's engine translog.
//!
//! Format (engine `translog.rs`): files `translog-<gen:08>.tlog` under
//! `<shard path>/translog`, records `[len u32 LE][crc32 u32 LE][bincode
//! TranslogOp]`. The engine appends a whole batch with one `write(2)` under
//! its write lock, so on the primary the log is in seq_no order. A reader
//! can still observe a partially written tail: it is left unconsumed and
//! re-read on the next poll.
//!
//! Generations roll on flush / force-merge (`roll` = fsync + create
//! `gen+1` under the engine write lock) and every older generation is then
//! deleted. The tailer keeps the current generation's file open, so on
//! Unix it can finish reading a generation even after it was unlinked; it
//! moves to `gen+1` only once that file exists, re-reading the old one to
//! its end first (existence of `gen+1` proves `gen` is final). If the
//! generation it needs next is gone, or a seek asks for ops below the
//! oldest one still on disk, it reports
//! [`ReplicationError::OpsTrimmed`](super::ReplicationError::OpsTrimmed)
//! so the caller can fall back to a full recovery. Callers prevent that in
//! steady state by flushing only through
//! [`safe_flush`](super::flush_gate::safe_flush), which drains the tailer
//! first.

use super::frame::{self, Decoded};
use super::{ReplicationError, Result};
use elasticsearch_engine::translog::TranslogOp;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const TRANSLOG_DIR: &str = "translog";
const SHARD_STATE: &str = "shard_state.json";
const READ_BUFFER_BYTES: usize = 64 * 1024;

#[cfg(not(test))]
type TailerFile = File;

/// Count actual file reads in regression tests (including buffered read-ahead).
#[cfg(test)]
struct TailerFile {
    file: File,
    bytes_read: std::sync::Arc<std::sync::atomic::AtomicU64>,
}
#[cfg(test)]
impl Read for TailerFile {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(out)?;
        self.bytes_read.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}
#[cfg(test)]
impl Seek for TailerFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> { self.file.seek(pos) }
}


/// Where a tailer stands (persistable; for stats and debugging).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailerPosition {
    pub generation: u64,
    /// Byte offset of the next unread record in `generation`.
    pub offset: u64,
    /// seq_no of the last op returned (-1 = none yet).
    pub last_seq_no: i64,
    /// Ops below this seq_no are skipped (set by `seek_to_seq_no`).
    pub from_seq_no: i64,
}

pub struct TranslogTailer {
    shard_path: PathBuf,
    dir: PathBuf,
    generation: u64,
    offset: u64,
    file: Option<BufReader<TailerFile>>,
    last_seq_no: i64,
    from_seq_no: i64,
    /// Ops of different ids reach the primary's translog out of seq_no
    /// order (ES appends under the translog's own lock; the engine's per-id
    /// locks keep only each id's ops in order): an op is a duplicate only if
    /// it was returned, so returned seq_nos are tracked as a contiguous
    /// prefix (`returned_through`) plus the ones above it.
    returned_through: i64,
    returned_above: std::collections::BTreeSet<i64>,
    /// Seq_no ranges (start → end, inclusive) the prefix jumped over
    /// without returning them (no-op gaps, which never reach the translog,
    /// far behind the newest op): an op that later shows up in one is an
    /// error, never dropped.
    jumped: std::collections::BTreeMap<i64, i64>,
    /// [`TranslogTailer::REORDER_WINDOW`] (smaller in tests).
    reorder_window: i64,
    /// Set when we moved past a generation we could not read in full; the
    /// next op seen must be contiguous with what we returned.
    verify_contiguous: bool,
    /// The last poll stopped at a clean end of the newest generation:
    /// every op appended before it started has been read.
    caught_up: bool,
    buf: FrameBuffer,
    /// A compressed physical frame can contain more ops than one poll.
    /// Decode it once, retaining only that frame's remaining operations.
    pending: Option<elasticsearch_engine::translog::AdmittedRecord>,
    pending_end: u64,
    #[cfg(test)]
    bytes_read: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

fn file_name(generation: u64) -> String {
    format!("translog-{generation:08}.tlog")
}

fn parse_generation(name: &str) -> Option<u64> {
    name.strip_prefix("translog-")?.strip_suffix(".tlog")?.parse().ok()
}

fn list_generations(dir: &Path) -> Result<Vec<u64>> {
    let mut v = Vec::new();
    match std::fs::read_dir(dir) {
        Ok(rd) => {
            for e in rd {
                if let Some(g) = e?.file_name().to_str().and_then(parse_generation) {
                    v.push(g);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    v.sort_unstable();
    Ok(v)
}

/// The engine's committed seq_no (`shard_state.json`), -1 if unknown.
pub(crate) fn committed_seq_no(shard_path: &Path) -> i64 {
    std::fs::read(shard_path.join(SHARD_STATE))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["committed_seq_no"].as_i64())
        .unwrap_or(-1)
}

/// Read at most one physical frame. A corrupt length must not reserve an
/// arbitrary allocation: `take` grows the buffer only for bytes that exist.
/// Callers rewind an incomplete/corrupt tail so an in-progress append can
/// be retried. Buffered read-ahead is bounded independently of WAL size.
struct FrameBuffer {
    bytes: Vec<u8>,
    reservation: Option<elasticsearch_engine::memory::Reservation>,
    breakers: &'static elasticsearch_engine::memory::Breakers,
}
impl std::ops::Deref for FrameBuffer {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target { &self.bytes }
}
impl FrameBuffer {
    fn new() -> Self {
        Self { bytes: Vec::new(), reservation: None, breakers: elasticsearch_engine::memory::breakers() }
    }
    fn release(&mut self) {
        // Free storage before its charge. Vec::clear would retain both capacity
        // and the request reservation for the lifetime of an idle tailer.
        self.bytes = Vec::new();
        self.reservation = None;
    }
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let required = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| std::io::Error::other("translog frame size overflow"))?;
        if required > self.bytes.capacity() {
            let capacity = required.max(self.bytes.capacity().saturating_mul(2));
            // A separate replacement allocation makes the old+new overlap
            // explicit; the prior owner remains charged until its Vec is gone.
            let reservation = self.breakers.reserve_request(capacity as u64, "translog encoded frame")?;
            let mut replacement = Vec::new();
            replacement.try_reserve_exact(capacity).map_err(|error| elasticsearch_engine::error::EsError::new(429, "circuit_breaking_exception", format!("translog frame allocation refused: {error}")))?;
            replacement.extend_from_slice(&self.bytes);
            self.bytes = replacement;
            self.reservation = Some(reservation);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
}

fn read_frame(reader: &mut impl Read, buf: &mut FrameBuffer) -> Result<()> {
    buf.bytes.clear();
    // Read into bounded stack storage first: a corrupt advertised length does
    // not preallocate bytes absent from the file. Every retained growth is admitted.
    let mut chunk = [0u8; 8192];
    let mut remaining = frame::HEADER_LEN;
    while remaining != 0 {
        let n = reader.read(&mut chunk[..remaining])?;
        if n == 0 { return Ok(()); }
        buf.append(&chunk[..n])?;
        remaining -= n;
    }
    remaining = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
    while remaining != 0 {
        let n = reader.read(&mut chunk[..remaining.min(8192)])?;
        if n == 0 { break; }
        buf.append(&chunk[..n])?;
        remaining -= n;
    }
    Ok(())
}

impl TranslogTailer {
    /// A tailer over the shard at `shard_path` (the engine's `Shard::path()`,
    /// i.e. `<index dir>/shard-0`), positioned at the start of the oldest
    /// generation on disk: the first poll returns every op still in the log.
    pub fn open(shard_path: &Path) -> Result<Self> {
        let dir = shard_path.join(TRANSLOG_DIR);
        let generation = list_generations(&dir)?.first().copied().unwrap_or(1);
        let mut t = Self {
            shard_path: shard_path.to_path_buf(),
            dir,
            generation,
            offset: 0,
            file: None,
            last_seq_no: -1,
            from_seq_no: 0,
            returned_through: -1,
            returned_above: std::collections::BTreeSet::new(),
            jumped: std::collections::BTreeMap::new(),
            reorder_window: Self::REORDER_WINDOW,
            verify_contiguous: false,
            caught_up: false,
            buf: FrameBuffer::new(),
            pending: None,
            pending_end: 0,
            #[cfg(test)]
            bytes_read: Default::default(),
        };
        t.file = t.open_gen(generation)?;
        Ok(t)
    }

    /// A tailer positioned so the first poll returns ops `>= from_seq_no`.
    pub fn open_at(shard_path: &Path, from_seq_no: i64) -> Result<Self> {
        let mut t = Self::open(shard_path)?;
        t.seek_to_seq_no(from_seq_no)?;
        Ok(t)
    }

    pub fn position(&self) -> TailerPosition {
        TailerPosition { generation: self.generation, offset: self.offset, last_seq_no: self.last_seq_no, from_seq_no: self.from_seq_no }
    }

    pub fn last_seq_no(&self) -> i64 {
        self.last_seq_no
    }

    fn open_gen(&self, generation: u64) -> Result<Option<BufReader<TailerFile>>> {
        match File::open(self.dir.join(file_name(generation))) {
            Ok(file) => {
                #[cfg(test)]
                let file = TailerFile { file, bytes_read: self.bytes_read.clone() };
                Ok(Some(BufReader::with_capacity(READ_BUFFER_BYTES, file)))
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Reposition so the next poll returns ops with `seq_no >= from` (the
    /// translog is seq_no-ordered on a primary). Fails with `OpsTrimmed` if
    /// ops in `[from, oldest available)` were already trimmed.
    pub fn seek_to_seq_no(&mut self, from: i64) -> Result<()> {
        let from = from.max(0);
        let gens = list_generations(&self.dir)?;
        let mut oldest_available: Option<i64> = None;
        let mut target = None;
        let mut last_gen_end = None;
        let mut buf = FrameBuffer::new();
        'generations: for &g in &gens {
            let Some(mut f) = self.open_gen(g)? else { continue };
            let mut offset = 0u64;
            loop {
                read_frame(&mut f, &mut buf)?;
                match frame::decode(&buf) {
                    Decoded::Record(body, len) => {
                        let mut found = false;
                        elasticsearch_engine::translog::visit_record(body, |op| {
                            let seq = op.seq_no() as i64;
                            oldest_available = Some(oldest_available.map_or(seq, |old| old.min(seq)));
                            found |= seq >= from;
                            Ok(())
                        })?;
                        if found {
                            // Translog operations are seq_no ordered. The
                            // first matching frame is enough; later frames
                            // are decoded lazily by bounded polls.
                            target = Some((g, offset, f));
                            break 'generations;
                        }
                        offset += len as u64;
                    }
                    Decoded::Incomplete | Decoded::Corrupt => {
                        last_gen_end = Some((g, offset, f));
                        break;
                    }
                }
            }
        }
        // Appends are not strictly seq_no-ordered (concurrent batches are
        // sequenced before they reach the translog, within a bounded
        // window): a frame holding `from + 1` can precede the one holding
        // `from`. The frames scanned so far only bound the oldest op from
        // above; before reporting a trim, look on through the retained
        // generations for anything at or below `from` (the common, ordered
        // case never gets here, so a seek stays bounded).
        // Nothing above the commit's checkpoint is ever trimmed (the translog
        // keeps every generation holding an op above it): such an op is
        // retained or not appended yet (sequenced by a batch still on its
        // way to the translog, after a flush rolled to an empty generation).
        // Only a seek at or below the commit point can need trimmed ops.
        let above_commit = committed_seq_no(&self.shard_path) + 1;
        if let (Some((g, _, _)), Some(oldest)) = (&target, oldest_available) {
            if oldest > from && from < above_commit {
                oldest_available = Some(self.oldest_seq_no_from(&gens, *g, from, oldest)?);
            }
        }
        let available_from = oldest_available.map_or(above_commit, |oldest| oldest.min(above_commit));
        if from < available_from {
            return Err(ReplicationError::OpsTrimmed { needed_from: from, available_from });
        }
        let (g, off, mut f) = match (target, last_gen_end) {
            (Some(t), _) => t,
            (None, Some(e)) => e,
            (None, None) => {
                let g = gens.last().copied().unwrap_or(self.generation);
                (g, 0, match self.open_gen(g)? {
                    Some(f) => f,
                    None => {
                        self.generation = g;
                        self.offset = 0;
                        self.file = None;
                        self.from_seq_no = from;
                        self.last_seq_no = from - 1;
                        self.reset_returned(from);
                        self.verify_contiguous = false;
                        self.pending = None;
                        self.pending_end = 0;
                        return Ok(());
                    }
                })
            }
        };
        f.seek(SeekFrom::Start(off))?;
        self.generation = g;
        self.offset = off;
        self.file = Some(f);
        self.from_seq_no = from;
        self.last_seq_no = from - 1;
        self.reset_returned(from);
        self.verify_contiguous = false;
        self.pending = None;
        self.pending_end = off;
        Ok(())
    }

    /// The lowest seq_no in the retained generations from `first` on,
    /// starting from `oldest` and stopping as soon as it is at or below
    /// `from`.
    fn oldest_seq_no_from(&self, gens: &[u64], first: u64, from: i64, mut oldest: i64) -> Result<i64> {
        let mut buf = FrameBuffer::new();
        for &g in gens.iter().filter(|g| **g >= first) {
            let Some(mut f) = self.open_gen(g)? else { continue };
            loop {
                read_frame(&mut f, &mut buf)?;
                match frame::decode(&buf) {
                    Decoded::Record(body, _) => {
                        elasticsearch_engine::translog::visit_record(body, |op| {
                            oldest = oldest.min(op.seq_no() as i64);
                            Ok(())
                        })?;
                        if oldest <= from { return Ok(oldest); }
                    }
                    Decoded::Incomplete | Decoded::Corrupt => break,
                }
            }
        }
        Ok(oldest)
    }

    /// Nothing below `from` is returned again.
    fn reset_returned(&mut self, from: i64) {
        self.returned_through = from - 1;
        self.returned_above.clear();
        self.jumped.clear();
    }

    /// Most jumped-over seq_no ranges tracked: past it the tailer fails
    /// (the protection never lapses silently).
    const MAX_JUMPED: usize = 1 << 16;

    fn jumped_over(&self, seq: i64) -> bool {
        self.jumped.range(..=seq).next_back().is_some_and(|(_, &end)| seq <= end)
    }

    /// The watermark: `processed` is the primary's processed local
    /// checkpoint sampled before the last poll (a seq_no is processed only
    /// after its append, or as a no-op that never reaches the translog).
    /// When that poll read to a clean end of the newest generation, every
    /// op at or below it has been read and none can appear later, so the
    /// jumped ranges there need no tracking: they are dropped (the loud
    /// cap then counts only ranges above the watermark). Returns whether
    /// it applied.
    pub fn prune_jumped_through(&mut self, processed: i64) -> bool {
        if !self.caught_up {
            return false;
        }
        let below: Vec<i64> = self.jumped.range(..=processed).map(|(start, _)| *start).collect();
        for start in below {
            let end = self.jumped.remove(&start).expect("listed range");
            if end > processed {
                self.jumped.insert(processed + 1, end);
            }
        }
        true
    }

    fn add_jumped(&mut self, start: i64, end: i64) -> Result<()> {
        // Extends the range just below it (a gap the prefix passes in
        // several steps stays one range).
        if let Some((_, prev_end)) = self.jumped.range_mut(..start).next_back() {
            if *prev_end == start - 1 {
                *prev_end = end;
                return Ok(());
            }
        }
        if self.jumped.len() >= Self::MAX_JUMPED {
            return Err(ReplicationError::Corrupt {
                what: "translog".into(),
                detail: format!("more than {} seq_no gaps far behind the newest op (at [{start}..={end}]): the history cannot be tracked", Self::MAX_JUMPED),
            });
        }
        self.jumped.insert(start, end);
        Ok(())
    }

    /// Ops this many seq_nos below the newest returned are never new: no
    /// in-flight batch lags that far (bounds the tracking across no-op
    /// gaps, which never reach the translog).
    const REORDER_WINDOW: i64 = 1 << 20;

    /// Test seam: a smaller reorder window.
    #[cfg(test)]
    fn set_reorder_window(&mut self, window: i64) {
        self.reorder_window = window;
    }

    fn already_returned(&self, seq: i64) -> bool {
        seq <= self.returned_through || seq < self.last_seq_no - self.reorder_window || self.returned_above.contains(&seq)
    }

    fn mark_returned(&mut self, seq: i64) -> Result<()> {
        self.returned_above.insert(seq);
        while self.returned_above.remove(&(self.returned_through + 1)) {
            self.returned_through += 1;
        }
        // A gap (a no-op seq_no) far behind: move the prefix past it,
        // recording the skipped ranges (walking only the returned seq_nos
        // in between, never the skipped range itself).
        let floor = self.last_seq_no.max(seq) - self.reorder_window;
        if self.returned_through < floor {
            let mut start = self.returned_through + 1;
            let returned: Vec<i64> = self.returned_above.range(start..=floor).copied().collect();
            for r in returned {
                if r > start {
                    self.add_jumped(start, r - 1)?;
                }
                start = r + 1;
            }
            if start <= floor {
                self.add_jumped(start, floor)?;
            }
            self.returned_through = floor;
            self.returned_above = self.returned_above.split_off(&(floor + 1));
            while self.returned_above.remove(&(self.returned_through + 1)) {
                self.returned_through += 1;
            }
        }
        Ok(())
    }

    /// Every complete op appended since the last poll, in log order.
    pub fn poll(&mut self) -> Result<elasticsearch_engine::translog::AdmittedBatch<TranslogOp>> {
        self.poll_limited(usize::MAX)
    }

    /// Like [`poll`](Self::poll) but returns at most `max_ops` ops (the rest
    /// stays for the next call).
    pub fn poll_limited(&mut self, max_ops: usize) -> Result<elasticsearch_engine::translog::AdmittedBatch<TranslogOp>> {
        self.poll_mapped(max_ops, |op| op)
    }

    /// Map by moving payloads directly into preadmitted output slots. A
    /// refused destination allocation leaves the pending operation unmarked.
    pub fn poll_mapped<T>(&mut self, max_ops: usize, mut map: impl FnMut(TranslogOp) -> T)
        -> Result<elasticsearch_engine::translog::AdmittedBatch<T>> {
        let result = self.poll_limited_inner(max_ops, &mut map);
        // Includes EOF, torn/corrupt records, read/decoder errors and breaker
        // refusal. Pending operations own their data independently of raw bytes.
        self.buf.release();
        result
    }

    fn poll_limited_inner<T>(&mut self, max_ops: usize, map: &mut impl FnMut(TranslogOp) -> T)
        -> Result<elasticsearch_engine::translog::AdmittedBatch<T>> {
        let mut out = elasticsearch_engine::translog::AdmittedBatch::new(self.buf.breakers)?;
        self.caught_up = false;
        loop {
            if out.len() >= max_ops {
                return Ok(out);
            }
            if self.file.is_none() {
                self.file = self.open_gen(self.generation)?;
            }
            let (clean_end, corrupt) = match self.read_current(&mut out, max_ops, map) {
                Ok(result) => result,
                // Earlier operations are already marked returned. Deliver that
                // admitted prefix instead of discarding it on a later frame's
                // memory refusal. The refused frame was rewound and is retried
                // on the next poll, which reports refusal if no prefix exists.
                Err(ReplicationError::Engine(ref error)) if error.kind == "circuit_breaking_exception" && !out.is_empty() => return Ok(out),
                Err(error) => return Err(error),
            };
            if out.len() >= max_ops {
                return Ok(out);
            }
            // Is there a newer generation?
            let gens = list_generations(&self.dir)?;
            let Some(&next) = gens.iter().find(|&&g| g > self.generation) else {
                if corrupt {
                    // The active generation may be mid-write; retry later.
                    tracing::debug!("translog tail of gen {} not yet complete", self.generation);
                }
                if self.verify_contiguous {
                    self.check_nothing_trimmed()?;
                }
                self.caught_up = clean_end && !corrupt && self.pending.as_ref().is_none_or(|p| p.len() == 0);
                return Ok(out);
            };
            // `next` exists, so the current generation is final: read the
            // rest of it once more before switching.
            let fully_read = if self.file.is_some() {
                let _ = (clean_end, corrupt);
                let (clean_end2, corrupt2) = match self.read_current(&mut out, max_ops, map) {
                    Ok(result) => result,
                    Err(ReplicationError::Engine(ref error)) if error.kind == "circuit_breaking_exception" && !out.is_empty() => return Ok(out),
                    Err(error) => return Err(error),
                };
                if out.len() >= max_ops {
                    return Ok(out);
                }
                if corrupt2 || !clean_end2 {
                    return Err(ReplicationError::Corrupt {
                        what: "translog".into(),
                        detail: format!("generation {} has a torn or corrupt record at offset {}", self.generation, self.offset),
                    });
                }
                true
            } else {
                false
            };
            if !fully_read || next != self.generation + 1 {
                // We never saw (all of) the generations in between.
                self.verify_contiguous = true;
            }
            self.generation = next;
            self.offset = 0;
            self.pending_end = 0;
            self.file = self.open_gen(next)?;
        }
    }

    /// Read records from the open file starting at `offset`. Returns
    /// (stopped at a clean end with no partial bytes, hit a bad checksum).
    fn read_current<T>(&mut self, out: &mut elasticsearch_engine::translog::AdmittedBatch<T>, max_ops: usize,
        map: &mut impl FnMut(TranslogOp) -> T) -> Result<(bool, bool)> {
        while out.len() < max_ops {
            while let Some(op) = self.pending.as_ref().and_then(|p| p.first()) {
                let seq = op.seq_no() as i64;
                if self.jumped_over(seq) || (seq < self.last_seq_no - self.reorder_window && seq > self.returned_through && !self.returned_above.contains(&seq)) {
                    // Appended after ops more than the reorder window above
                    // it were returned: never silently dropped.
                    return Err(ReplicationError::Corrupt {
                        what: "translog".into(),
                        detail: format!("op [{seq}] arrived more than {} seq_nos behind the newest returned op [{}]", self.reorder_window, self.last_seq_no),
                    });
                }
                if seq < self.from_seq_no || self.already_returned(seq) {
                    self.pending.as_mut().expect("pending record").discard_next();
                    continue;
                }
                if self.verify_contiguous {
                    let expected = self.last_seq_no + 1;
                    // Only ops at or below the commit point can have been
                    // trimmed (every generation holding an op above it is
                    // retained, as for `seek_to_seq_no`). A first op above
                    // `expected` when `expected` is above the commit is a
                    // batch that reached the translog ahead of an earlier
                    // one (seq_nos are assigned before the append): the
                    // earlier batch's ops follow, and are returned then.
                    // A generation skipped here is one with no ops at all,
                    // trimmed before the tailer reached it (two flushes of
                    // a quiet shard).
                    if seq > expected && expected <= committed_seq_no(&self.shard_path) {
                        return Err(ReplicationError::OpsTrimmed { needed_from: expected, available_from: seq });
                    }
                    self.verify_contiguous = false;
                }
                out.reserve_one()?;
                self.mark_returned(seq)?;
                self.last_seq_no = self.last_seq_no.max(seq);
                self.pending.as_mut().expect("pending record").move_next(out, &mut *map);
                if out.len() >= max_ops { break; }
            }
            if self.pending.as_ref().is_none_or(|p| p.len() == 0) {
                // Keep the public position on the physical frame until all
                // its operations have been consumed (compressed batches).
                self.pending = None;
                self.offset = self.pending_end;
            }
            if out.len() >= max_ops { return Ok((true, false)); }
            let Some(f) = self.file.as_mut() else { return Ok((true, false)) };
            if let Err(e) = read_frame(f, &mut self.buf) {
                let _ = f.seek(SeekFrom::Start(self.offset));
                return Err(e.into());
            }
            match frame::decode(&self.buf) {
                Decoded::Record(body, len) => {
                    let decoded = match elasticsearch_engine::translog::decode_admitted_record(body, self.buf.breakers) {
                        Ok(decoded) => decoded,
                        Err(e) => {
                            let _ = f.seek(SeekFrom::Start(self.offset));
                            return Err(e.into());
                        }
                    };
                    self.pending = Some(decoded);
                    self.pending_end = self.offset + len as u64;
                    // The decoder has produced owned operations; no borrowed
                    // slice of this encoded frame survives in `pending`. The
                    // buffer and its reservation serve the next frame of this
                    // poll (`read_frame` clears it); the poll releases it.
                }
                Decoded::Incomplete => {
                    f.seek(SeekFrom::Start(self.offset))?;
                    return Ok((self.buf.is_empty(), false));
                }
                Decoded::Corrupt => {
                    f.seek(SeekFrom::Start(self.offset))?;
                    return Ok((false, true));
                }
            }
        }
        Ok((true, false))
    }

    /// After skipping unreadable generations with no op seen since: the
    /// ops we still need must not be committed-and-trimmed.
    fn check_nothing_trimmed(&self) -> Result<()> {
        let committed = committed_seq_no(&self.shard_path);
        let expected = self.last_seq_no + 1;
        if committed >= expected {
            return Err(ReplicationError::OpsTrimmed { needed_from: expected, available_from: committed + 1 });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::index::{IndexService, WriteReq};
    use elasticsearch_engine::settings;
    use elasticsearch_engine::shard::WriteCond;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::io::Write;

    fn svc(dir: &Path) -> IndexService {
        IndexService::create(dir, "t", settings::normalize_index_settings(&json!({})), &json!({}), BTreeMap::new()).unwrap()
    }

    fn put(s: &IndexService, id: &str, n: i64) {
        let r = s.write(vec![WriteReq::Index { id: Some(id.into()), source: format!("{{\"n\":{n}}}"), routing: None, cond: WriteCond::default() }], false);
        r[0].as_ref().unwrap();
    }

    fn seqs(ops: &[TranslogOp]) -> Vec<u64> {
        ops.iter().map(|o| o.seq_no()).collect()
    }

    #[test]
    fn tails_across_flush_and_roll() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        let mut t = TranslogTailer::open(crate::shards::copy_of(&s, 0).shard.path()).unwrap();
        assert!(t.poll().unwrap().is_empty());
        put(&s, "a", 1);
        put(&s, "b", 2);
        assert_eq!(seqs(&t.poll().unwrap()), vec![0, 1]);
        assert!(t.poll().unwrap().is_empty());
        put(&s, "c", 3);
        // Flush rolls to gen 2 and deletes gen 1; the tailer still holds gen 1
        // open and must not lose the last record written before the roll.
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        put(&s, "d", 4);
        s.write(vec![WriteReq::Delete { id: "a".into(), routing: None, cond: WriteCond::default() }], false);
        let ops = t.poll().unwrap();
        assert_eq!(seqs(&ops), vec![2, 3, 4]);
        assert!(matches!(ops[2], TranslogOp::Delete { .. }));
        assert_eq!(t.position().generation, 2);
        // Several flushes with nothing in between.
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        put(&s, "e", 5);
        assert_eq!(seqs(&t.poll().unwrap()), vec![5]);
        // poll_limited keeps the rest for later.
        for i in 0..10 {
            put(&s, &format!("x{i}"), i);
        }
        assert_eq!(t.poll_limited(4).unwrap().len(), 4);
        assert_eq!(seqs(&t.poll().unwrap()), (10..16).collect::<Vec<_>>());
    }

    #[test]
    fn torn_tail_is_retried() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        put(&s, "a", 1);
        let mut t = TranslogTailer::open(crate::shards::copy_of(&s, 0).shard.path()).unwrap();
        assert_eq!(t.poll().unwrap().len(), 1);
        // Simulate a half-visible append: write a header + part of a body.
        let path = crate::shards::copy_of(&s, 0).shard.path().join("translog").join(file_name(1));
        let op = TranslogOp::Index { id: "z".into(), source_json: "{}".into(), seq_no: 1, version: 1, routing: None };
        let mut rec = Vec::new();
        frame::encode(&bincode::serialize(&op).unwrap(), &mut rec);
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&rec[..rec.len() - 3]).unwrap();
        assert!(t.poll().unwrap().is_empty());
        f.write_all(&rec[rec.len() - 3..]).unwrap();
        assert_eq!(seqs(&t.poll().unwrap()), vec![1]);
    }

    #[test]
    fn seek_and_trimmed_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        for i in 0..5 {
            put(&s, &format!("d{i}"), i);
        }
        let mut t = TranslogTailer::open_at(crate::shards::copy_of(&s, 0).shard.path(), 3).unwrap();
        assert_eq!(seqs(&t.poll().unwrap()), vec![3, 4]);
        t.seek_to_seq_no(0).unwrap();
        assert_eq!(seqs(&t.poll().unwrap()), vec![0, 1, 2, 3, 4]);
        // Seek past the end: positioned at the end, gets future ops.
        t.seek_to_seq_no(5).unwrap();
        assert!(t.poll().unwrap().is_empty());
        put(&s, "d5", 5);
        assert_eq!(seqs(&t.poll().unwrap()), vec![5]);

        // A lagging tailer (opened, never polled, file handle on gen 1).
        let mut lagging = TranslogTailer::open_at(crate::shards::copy_of(&s, 0).shard.path(), 4).unwrap();
        crate::shards::copy_of(&s, 0).shard.flush().unwrap(); // committed 5, gen 1 deleted
        put(&s, "d6", 6);
        // It still holds gen 1 open, so it can finish it.
        assert_eq!(seqs(&lagging.poll().unwrap()), vec![4, 5, 6]);

        // Ops below the commit point are gone from the translog.
        let e = TranslogTailer::open_at(crate::shards::copy_of(&s, 0).shard.path(), 2).err().unwrap();
        assert!(matches!(e, ReplicationError::OpsTrimmed { needed_from: 2, available_from: 6 }), "{e}");
        // Empty translog after a flush: availability comes from the commit.
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        let e = TranslogTailer::open_at(crate::shards::copy_of(&s, 0).shard.path(), 6).err().unwrap();
        assert!(matches!(e, ReplicationError::OpsTrimmed { needed_from: 6, available_from: 7 }), "{e}");
        let mut t = TranslogTailer::open_at(crate::shards::copy_of(&s, 0).shard.path(), 7).unwrap();
        put(&s, "d7", 7);
        assert_eq!(seqs(&t.poll().unwrap()), vec![7]);
    }

    /// Two flushes of a quiet shard roll through a generation that never
    /// holds an op, and the second trims it before the tailer reaches it.
    /// When writes resume, a batch can reach the translog ahead of an
    /// earlier one (seq_nos are assigned before the append). Those ops are
    /// above the commit point, never trimmed: the skipped generation is no
    /// gap. (The contiguity check reported them trimmed, and kept failing
    /// on the same op forever.)
    #[test]
    fn a_quiet_generation_trimmed_before_the_tailer_reached_it_is_no_gap() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        put(&s, "a", 0);
        let path = crate::shards::copy_of(&s, 0).shard.path().to_path_buf();
        let mut t = TranslogTailer::open(&path).unwrap();
        assert_eq!(seqs(&t.poll().unwrap()), vec![0]);
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        let gens = list_generations(&path.join(TRANSLOG_DIR)).unwrap();
        assert_eq!(gens.len(), 1, "{gens:?}");
        assert!(gens[0] >= t.generation + 2, "precondition: the tailer will skip a generation: {gens:?} from {}", t.generation);
        // Seq_nos 1 and 2 reach the new generation as 2 first, then 1.
        let mut log = elasticsearch_engine::translog::Translog::open(&path.join(TRANSLOG_DIR), gens[0]).unwrap();
        for seq_no in [2u64, 1] {
            log.append_many(&[TranslogOp::Index { id: format!("x{seq_no}"), source_json: "{}".into(), seq_no, version: 1, routing: None }]).unwrap();
        }
        log.fsync().unwrap();
        assert_eq!(seqs(&t.poll().expect("ops above the commit point reported trimmed")), vec![2, 1]);
    }

    #[test]
    fn skipped_generations_report_trimmed() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        put(&s, "a", 0);
        let mut t = TranslogTailer::open(crate::shards::copy_of(&s, 0).shard.path()).unwrap();
        assert_eq!(t.poll().unwrap().len(), 1);
        // Drop the open handle to simulate a reader that lost its file
        // (e.g. restarted with a persisted position).
        t.file = None;
        put(&s, "b", 1); // gen 1, seq 1 — the tailer will never see it
        crate::shards::copy_of(&s, 0).shard.flush().unwrap(); // gen 2; gen 1 deleted
        crate::shards::copy_of(&s, 0).shard.flush().unwrap(); // gen 3; gen 2 deleted
        put(&s, "c", 2);
        let e = t.poll().err().unwrap();
        assert!(matches!(e, ReplicationError::OpsTrimmed { needed_from: 1, available_from: 2 }), "{e}");

        // Same without any later op: detected through the commit point.
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        put(&s, "a", 0);
        let mut t = TranslogTailer::open(crate::shards::copy_of(&s, 0).shard.path()).unwrap();
        assert_eq!(t.poll().unwrap().len(), 1);
        t.file = None;
        put(&s, "b", 1);
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        crate::shards::copy_of(&s, 0).shard.flush().unwrap();
        let e = t.poll().err().unwrap();
        assert!(matches!(e, ReplicationError::OpsTrimmed { needed_from: 1, available_from: 2 }), "{e}");
    }
}


#[cfg(test)]
mod compressed_batch_tests {
    use super::*;
    use elasticsearch_engine::translog::Translog;
    use std::io::Write;
    fn ops(from: u64, to: u64) -> Vec<TranslogOp> {
        (from..to).map(|seq_no| TranslogOp::Index {
            id: format!("d{seq_no}"), source_json: "{\"value\":123}".into(), seq_no, version:1, routing:None
        }).collect()
    }
    #[test]
    fn limited_polls_seek_and_roll_keep_compressed_batch_positions() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR),1).unwrap();
        log.append_many_with(&ops(0,200), true).unwrap();
        log.fsync().unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        // Retain each page's real owner while checking the combined sequence;
        // do not unwrap admitted payloads or deep-copy them into an unowned Vec.
        let mut pages = Vec::new();
        let mut count = 0;
        while count < 200 {
            let page = tail.poll_limited(7).unwrap();
            assert!(!page.is_empty() && page.len() <= 7);
            count += page.len();
            pages.push(page);
        }
        let got: Vec<_> = pages.iter().flat_map(|page| page.iter()).collect();
        assert_eq!(got, ops(0,200).iter().collect::<Vec<_>>());
        tail.seek_to_seq_no(57).unwrap();
        assert_eq!(tail.poll_limited(3).unwrap(), ops(57,60));
        tail.seek_to_seq_no(197).unwrap();
        log.roll().unwrap();
        log.append_many_with(&ops(200,400), true).unwrap();
        log.fsync().unwrap();
        log.trim_to(2).unwrap();
        assert_eq!(tail.poll_limited(9).unwrap(), ops(197,206));
        assert_eq!(tail.poll().unwrap(), ops(206,400));
    }
    /// B5: a seek to `from` after a trim, where the frame holding `from + 1`
    /// was appended before the one holding `from`, finds `from` instead of
    /// reporting it trimmed.
    #[test]
    fn seek_after_trim_finds_an_op_appended_after_a_higher_one() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR),1).unwrap();
        log.append_many(&ops(0,10)).unwrap();
        log.roll().unwrap();
        log.trim_to(2).unwrap();
        log.append_many(&ops(11,12)).unwrap();
        log.append_many(&ops(10,11)).unwrap();
        log.fsync().unwrap();
        // The commit covers 0..=9 (what the trim dropped).
        std::fs::write(tmp.path().join(SHARD_STATE), br#"{"committed_seq_no": 9}"#).unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        tail.seek_to_seq_no(10).unwrap();
        let mut got: Vec<u64> = tail.poll().unwrap().iter().map(TranslogOp::seq_no).collect();
        got.sort_unstable();
        assert_eq!(got, vec![10, 11]);
        // Truly trimmed ops are still reported.
        assert!(matches!(tail.seek_to_seq_no(5), Err(ReplicationError::OpsTrimmed { needed_from: 5, available_from: 10 })));
    }
    /// B5: a flush rolled to an empty generation while a batch had seq_no
    /// 10 sequenced but not appended, and 11 was appended: a seek to 10 (the
    /// op above the commit point is on its way) is not "trimmed".
    #[test]
    fn seek_to_an_op_above_the_commit_not_appended_yet_is_not_trimmed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR),1).unwrap();
        log.append_many(&ops(0,10)).unwrap();
        log.roll().unwrap();
        log.trim_to(2).unwrap();
        std::fs::write(tmp.path().join(SHARD_STATE), br#"{"committed_seq_no": 9}"#).unwrap();
        log.append_many(&ops(11,12)).unwrap();
        log.fsync().unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        tail.seek_to_seq_no(10).expect("an op above the commit point was reported trimmed");
        assert_eq!(tail.poll().unwrap(), ops(11,12));
        // The batch's append lands: the tailer returns it too.
        log.append_many(&ops(10,11)).unwrap();
        log.fsync().unwrap();
        assert_eq!(tail.poll().unwrap(), ops(10,11));
    }
    #[test]
    fn partially_written_compressed_frame_waits_for_complete_bytes() {
        let source = tempfile::tempdir().unwrap();
        let mut log = Translog::open(source.path(),1).unwrap();
        log.append_many_with(&ops(0,200), true).unwrap();
        log.fsync().unwrap();
        let bytes = std::fs::read(source.path().join(file_name(1))).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(TRANSLOG_DIR)).unwrap();
        let path = tmp.path().join(TRANSLOG_DIR).join(file_name(1));
        let split = bytes.len()/2;
        std::fs::write(&path,&bytes[..split]).unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert!(tail.poll().unwrap().is_empty());
        std::fs::OpenOptions::new().append(true).open(path).unwrap().write_all(&bytes[split..]).unwrap();
        assert_eq!(tail.poll().unwrap(),ops(0,200));
    }
}


#[cfg(test)]
mod bounded_frame_tests {
    use super::*;
    use elasticsearch_engine::translog::Translog;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    fn op(seq_no: u64) -> TranslogOp {
        TranslogOp::Index { id: format!("doc-{seq_no}"), source_json: "x".repeat(1024), seq_no, version: 1, routing: None }
    }

    fn encoded(op: &TranslogOp) -> Vec<u8> {
        let mut bytes = Vec::new();
        frame::encode(&bincode::serialize(op).unwrap(), &mut bytes);
        bytes
    }

    fn install_first(shard: &Path, bytes: &[u8]) -> PathBuf {
        let dir = shard.join(TRANSLOG_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(file_name(1));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn encoded_frame_refusal_preserves_returned_prefix_and_rewinds_next_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let small = TranslogOp::Delete { id: "small".into(), seq_no: 0, version: 1 };
        let large = op(1);
        let mut bytes = encoded(&small);
        bytes.extend(encoded(&large));
        install_first(tmp.path(), &bytes);
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        // Room for the small frame's buffer (kept for the next frame of the
        // poll) with its output, not for the large frame.
        let budget = elasticsearch_engine::memory::Breakers::leak_for_tests(1500);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        tail.buf.breakers = budget;
        assert_eq!(tail.poll().unwrap(), vec![small]);
        let before = tail.position();
        assert!(matches!(tail.poll(), Err(ReplicationError::Engine(error)) if error.kind == "circuit_breaking_exception"));
        assert_eq!(tail.position(), before, "refused frame must remain unread");
        assert_eq!(tail.buf.capacity(), 0);
        assert_eq!(budget.stats().into_iter().find(|b| b.name == "request").unwrap().estimated, 0);
        // Change only the isolated budget, not storage ownership, to allow the
        // unchanged refused frame on the next poll.
        tail.buf.breakers = elasticsearch_engine::memory::breakers();
        assert_eq!(tail.poll().unwrap(), vec![large]);
        assert!(tail.poll().unwrap().is_empty());
    }

    #[test]
    fn encoded_storage_releases_while_decoded_pending_and_output_remain_admitted() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR), 1).unwrap();
        let expected: Vec<_> = (0..200).map(op).collect();
        log.append_many_with(&expected, true).unwrap();
        log.fsync().unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        let budget = elasticsearch_engine::memory::Breakers::leak_for_tests(1024 * 1024);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        tail.buf.breakers = budget;
        for operation in expected {
            let output = tail.poll_limited(1).unwrap();
            assert_eq!(output, vec![operation]);
            assert_eq!(tail.buf.capacity(), 0);
            assert!(budget.stats().iter().find(|b| b.name == "request").unwrap().estimated > 0, "decoded output remains owned");
            drop(output);
            let held = budget.stats().iter().find(|b| b.name == "request").unwrap().estimated;
            assert_eq!(held > 0, tail.pending.as_ref().is_some_and(|p| p.len() > 0));
        }
        assert!(tail.poll().unwrap().is_empty());
        assert_eq!(tail.buf.capacity(), 0);
        assert_eq!(budget.stats().into_iter().find(|b| b.name == "request").unwrap().estimated, 0);
    }

    #[test]
    fn mapped_slot_refusal_does_not_ack_or_consume_pending_operation() {
        let tmp = tempfile::tempdir().unwrap();
        let op = TranslogOp::Delete { id: "a".into(), seq_no: 0, version: 1 };
        install_first(tmp.path(), &encoded(&op));
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        let budget = elasticsearch_engine::memory::Breakers::leak_for_tests(2048);
        budget.apply_settings(&|k| (k == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        tail.buf.breakers = budget;
        let before = tail.position();
        let result = tail.poll_mapped(1, |_| [0u8; 4096]);
        assert!(matches!(result, Err(ReplicationError::Engine(e)) if e.kind == "circuit_breaking_exception"));
        assert_eq!(tail.position(), before);
        assert_eq!(tail.pending.as_ref().unwrap().first(), Some(&op));
        tail.buf.breakers = elasticsearch_engine::memory::breakers();
        assert_eq!(tail.poll().unwrap(), vec![op]);
        assert!(tail.poll().unwrap().is_empty());
    }

    #[test]
    fn limited_polls_read_each_physical_frame_once_not_the_remaining_wal() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR), 1).unwrap();
        // Individual frames deliberately avoid compression. A small poll
        // must not read this entire >10 MiB suffix or retain its allocation.
        for seq in 0..10_000 { log.append(&op(seq)).unwrap(); }
        log.fsync().unwrap();
        let size = std::fs::metadata(tmp.path().join(TRANSLOG_DIR).join(file_name(1))).unwrap().len();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert!(tail.poll_limited(0).unwrap().is_empty());
        assert_eq!(tail.bytes_read.load(Ordering::Relaxed), 0);
        assert_eq!(tail.poll_limited(7).unwrap(), (0..7).map(op).collect::<Vec<_>>());
        assert!(tail.bytes_read.load(Ordering::Relaxed) <= READ_BUFFER_BYTES as u64);
        assert!(tail.buf.capacity() < READ_BUFFER_BYTES);
        let mut next = 7;
        loop {
            let page = tail.poll_limited(31).unwrap();
            if page.is_empty() { break; }
            for got in &page { assert_eq!(*got, op(next)); next += 1; }
        }
        assert_eq!(next, 10_000);
        assert_eq!(tail.bytes_read.load(Ordering::Relaxed), size, "bounded polls must not repeatedly read the remaining suffix");
    }

    #[test]
    fn seek_near_start_does_not_materialize_or_read_the_rest_of_generation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR), 1).unwrap();
        for seq in 0..2_000 { log.append(&op(seq)).unwrap(); }
        log.fsync().unwrap();
        let mut tail = TranslogTailer::open_at(tmp.path(), 3).unwrap();
        assert!(tail.bytes_read.load(Ordering::Relaxed) <= READ_BUFFER_BYTES as u64);
        assert_eq!(tail.poll_limited(4).unwrap(), (3..7).map(op).collect::<Vec<_>>());
        assert!(tail.bytes_read.load(Ordering::Relaxed) <= 2 * READ_BUFFER_BYTES as u64);
        tail.seek_to_seq_no(1_998).unwrap();
        assert_eq!(tail.poll().unwrap(), vec![op(1_998), op(1_999)]);
        log.append(&op(2_000)).unwrap();
        assert_eq!(tail.poll().unwrap(), vec![op(2_000)]);
    }

    #[test]
    fn compressed_frame_remainder_is_retained_across_limited_polls_and_roll() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(&tmp.path().join(TRANSLOG_DIR), 1).unwrap();
        log.append_many_with(&(0..200).map(op).collect::<Vec<_>>(), true).unwrap();
        log.fsync().unwrap();
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert_eq!(tail.poll_limited(3).unwrap(), (0..3).map(op).collect::<Vec<_>>());
        assert_eq!(tail.position().offset, 0, "partly delivered compressed frame remains the persisted position");
        let first_read = tail.bytes_read.load(Ordering::Relaxed);
        assert_eq!(tail.poll_limited(3).unwrap(), (3..6).map(op).collect::<Vec<_>>());
        assert_eq!(tail.bytes_read.load(Ordering::Relaxed), first_read, "pending frame must not be read again");
        log.roll().unwrap();
        log.append_many(&(200..210).map(op).collect::<Vec<_>>()).unwrap();
        log.fsync().unwrap();
        log.trim_to(2).unwrap();
        let mut next = 6;
        loop {
            let page = tail.poll_limited(7).unwrap();
            if page.is_empty() { break; }
            for got in &page { assert_eq!(*got, op(next)); next += 1; }
        }
        assert_eq!(next, 210);
        assert_eq!(tail.position().generation, 2);
    }

    #[test]
    fn incomplete_headers_bodies_and_oversized_lengths_remain_retryable() {
        let bytes = encoded(&op(0));
        for split in [0, 1, 4, 7, 8, bytes.len() - 1] {
            let tmp = tempfile::tempdir().unwrap();
            let path = install_first(tmp.path(), &bytes[..split]);
            let mut tail = TranslogTailer::open(tmp.path()).unwrap();
            assert!(tail.poll().unwrap().is_empty(), "split {split}");
            assert_eq!(tail.position().offset, 0);
            std::fs::OpenOptions::new().append(true).open(path).unwrap().write_all(&bytes[split..]).unwrap();
            assert_eq!(tail.poll().unwrap(), vec![op(0)], "split {split}");
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut oversized = vec![0; frame::HEADER_LEN];
        oversized[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let path = install_first(tmp.path(), &oversized);
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert!(tail.poll().unwrap().is_empty());
        assert!(tail.buf.capacity() < READ_BUFFER_BYTES, "announced length must not preallocate the frame");
        std::fs::write(path, &bytes).unwrap();
        assert_eq!(tail.poll().unwrap(), vec![op(0)]);
    }

    #[test]
    fn corrupt_active_frame_retries_but_closed_corrupt_or_torn_generation_errors() {
        let good = encoded(&op(0));
        let mut bad = good.clone();
        bad[4] ^= 0x80; // Outer checksum mismatch.
        let tmp = tempfile::tempdir().unwrap();
        let path = install_first(tmp.path(), &bad);
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert!(tail.poll().unwrap().is_empty());
        assert_eq!(tail.position().offset, 0);
        std::fs::write(&path, &good).unwrap();
        assert_eq!(tail.poll().unwrap(), vec![op(0)]);

        for bytes in [&bad[..], &good[..good.len() - 1], &good[..4]] {
            let tmp = tempfile::tempdir().unwrap();
            install_first(tmp.path(), bytes);
            std::fs::write(tmp.path().join(TRANSLOG_DIR).join(file_name(2)), []).unwrap();
            let mut tail = TranslogTailer::open(tmp.path()).unwrap();
            assert!(matches!(tail.poll(), Err(ReplicationError::Corrupt { .. })));
            assert_eq!(tail.position().generation, 1);
            assert_eq!(tail.position().offset, 0);
        }
    }

    #[test]
    fn checksum_valid_malformed_body_is_an_error_even_on_active_generation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bytes = Vec::new();
        frame::encode(b"TLZ\x7funsupported", &mut bytes);
        install_first(tmp.path(), &bytes);
        let mut tail = TranslogTailer::open(tmp.path()).unwrap();
        assert!(tail.poll().unwrap_err().to_string().contains("unsupported compressed batch"));
        assert_eq!(tail.position().offset, 0);
        assert!(tail.poll().is_err(), "retry must not silently skip malformed frame");
    }

    /// Run in separate EC2 processes for preparation and measurement so
    /// fixture construction does not inflate the reader's peak RSS.
    #[test]
    #[ignore = "standalone EC2 WAL read-amplification measurement"]
    fn tailer_read_amplification_poc() {
        let dir = PathBuf::from(std::env::var("TAILER_POC_DIR").expect("set a dedicated TAILER_POC_DIR"));
        let count: u64 = std::env::var("TAILER_POC_OPS").unwrap_or_else(|_| "500000".into()).parse().unwrap();
        let batch: usize = std::env::var("TAILER_POC_BATCH").unwrap_or_else(|_| "1000".into()).parse().unwrap();
        let from: u64 = std::env::var("TAILER_POC_FROM").unwrap_or_else(|_| "0".into()).parse().unwrap();
        let compressed = std::env::var("TAILER_POC_COMPRESSED").is_ok_and(|v| v == "1");
        assert!(batch > 0 && from <= count);
        if std::env::var("TAILER_POC_PREPARE").is_ok_and(|v| v == "1") {
            // Refuse to overwrite a prior fixture or any unrelated data.
            std::fs::create_dir(&dir).unwrap();
            let mut log = Translog::open(&dir.join(TRANSLOG_DIR), 1).unwrap();
            if compressed {
                for start in (0..count).step_by(5000) {
                    log.append_many_with(&(start..(start + 5000).min(count)).map(op).collect::<Vec<_>>(), true).unwrap();
                }
            } else {
                for seq in 0..count { log.append(&op(seq)).unwrap(); }
            }
            log.fsync().unwrap();
            println!("prepared ops={count} compressed={compressed} bytes={}",
                std::fs::metadata(dir.join(TRANSLOG_DIR).join(file_name(1))).unwrap().len());
            return;
        }
        let start = std::time::Instant::now();
        let mut tail = TranslogTailer::open_at(&dir, from as i64).unwrap();
        let seek_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut expected = from;
        let mut polls = 0u64;
        loop {
            let page = tail.poll_limited(batch).unwrap();
            polls += 1;
            if page.is_empty() { break; }
            for got in &page {
                assert_eq!(got.seq_no(), expected);
                expected += 1;
            }
        }
        assert_eq!(expected, count);
        println!("ops={} batch={batch} polls={polls} seek_ms={seek_ms:.3} total_ms={:.3} file_bytes={} actual_read_bytes={}",
            count - from, start.elapsed().as_secs_f64() * 1000.0,
            std::fs::metadata(dir.join(TRANSLOG_DIR).join(file_name(1))).unwrap().len(),
            tail.bytes_read.load(Ordering::Relaxed));
    }

}

#[cfg(test)]
mod reorder_tests {
    use super::*;

    /// G2 appends ops of different ids out of seq_no order: the tailer
    /// returns each op once, including one appended after a higher seq_no.
    #[test]
    fn out_of_order_appends_are_returned_once() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        let mut translog = elasticsearch_engine::translog::Translog::open(&shard.join(TRANSLOG_DIR), 1).unwrap();
        let op = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: "{}".into(), seq_no: seq, version: 1, routing: None };
        translog.append_many(&[op(0), op(2)]).unwrap();
        translog.fsync().unwrap();
        let mut tailer = TranslogTailer::open(shard).unwrap();
        let first: Vec<u64> = tailer.poll().unwrap().iter().map(|o| o.seq_no()).collect();
        assert_eq!(first, vec![0, 2]);
        translog.append_many(&[op(1), op(3)]).unwrap();
        translog.fsync().unwrap();
        let second: Vec<u64> = tailer.poll().unwrap().iter().map(|o| o.seq_no()).collect();
        assert_eq!(second, vec![1, 3], "an op appended after a higher seq_no was dropped");
        assert!(tailer.poll().unwrap().is_empty());
        assert_eq!(tailer.last_seq_no(), 3);
    }

    /// M6: a seq_no gap far behind the newest op (a no-op, never in the
    /// translog) moves the prefix on; an op that does show up there later
    /// is an error, never silently dropped as already returned.
    #[test]
    fn an_op_beyond_the_reorder_window_is_an_error_not_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        let mut translog = elasticsearch_engine::translog::Translog::open(&shard.join(TRANSLOG_DIR), 1).unwrap();
        let op = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: "{}".into(), seq_no: seq, version: 1, routing: None };
        let mut tailer = TranslogTailer::open(shard).unwrap();
        tailer.set_reorder_window(8);
        // 5 is missing (a gap); ops run far past it.
        translog.append_many(&(0..30u64).filter(|s| *s != 5).map(op).collect::<Vec<_>>()).unwrap();
        translog.fsync().unwrap();
        let first = tailer.poll().unwrap();
        assert_eq!(first.len(), 29);
        assert!(tailer.poll().unwrap().is_empty(), "a gap that never arrives is no error");
        translog.append_many(&[op(5)]).unwrap();
        translog.fsync().unwrap();
        assert!(matches!(tailer.poll(), Err(ReplicationError::Corrupt { .. })), "a late op beyond the window was dropped");
    }

    /// M6 watermark: after a poll to a clean end of the newest generation,
    /// the jumped ranges at or below the primary's processed checkpoint
    /// sampled before it are dropped (none of those ops can still arrive);
    /// ranges above stay tracked, and nothing is pruned after a poll that
    /// stopped short.
    #[test]
    fn jumped_ranges_below_the_processed_watermark_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        let mut translog = elasticsearch_engine::translog::Translog::open(&shard.join(TRANSLOG_DIR), 1).unwrap();
        let op = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: "{}".into(), seq_no: seq, version: 1, routing: None };
        let mut tailer = TranslogTailer::open(shard).unwrap();
        tailer.set_reorder_window(8);
        // Gaps at 5 and 40 (no-ops), ops far past both.
        translog.append_many(&(0..100u64).filter(|s| *s != 5 && *s != 40).map(op).collect::<Vec<_>>()).unwrap();
        translog.fsync().unwrap();
        // A poll that stops short prunes nothing.
        assert_eq!(tailer.poll_limited(10).unwrap().len(), 10);
        assert!(!tailer.prune_jumped_through(99));
        assert_eq!(tailer.poll().unwrap().len(), 88);
        assert_eq!(tailer.jumped.len(), 2, "{:?}", tailer.jumped);
        // Processed through 20 when the caught-up poll started: 5 goes, 40 stays.
        assert!(tailer.prune_jumped_through(20));
        assert_eq!(tailer.jumped.keys().copied().collect::<Vec<_>>(), vec![40]);
        // A late op in the remaining range is still an error.
        translog.append_many(&[op(40)]).unwrap();
        translog.fsync().unwrap();
        assert!(matches!(tailer.poll(), Err(ReplicationError::Corrupt { .. })));
    }

    /// M6: a huge gap is recorded as one range (never walked seq_no by
    /// seq_no), a late op anywhere in it is still an error, and past the
    /// tracking cap the tailer fails instead of silently dropping the
    /// protection.
    #[test]
    fn jumped_gaps_are_ranges_and_the_cap_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        let mut translog = elasticsearch_engine::translog::Translog::open(&shard.join(TRANSLOG_DIR), 1).unwrap();
        let op = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: "{}".into(), seq_no: seq, version: 1, routing: None };
        let mut tailer = TranslogTailer::open(shard).unwrap();
        tailer.set_reorder_window(8);
        // A gap of 10 million seq_nos (no-ops that never reach the translog).
        translog.append_many(&[op(0), op(1)]).unwrap();
        translog.append_many(&(10_000_002..10_000_030u64).map(op).collect::<Vec<_>>()).unwrap();
        translog.fsync().unwrap();
        let t0 = std::time::Instant::now();
        assert_eq!(tailer.poll().unwrap().len(), 30);
        assert!(t0.elapsed() < std::time::Duration::from_secs(1), "the gap was walked seq_no by seq_no: {:?}", t0.elapsed());
        assert_eq!(tailer.jumped.len(), 1, "one range for one gap: {:?}", tailer.jumped);
        translog.append_many(&[op(5_000_000)]).unwrap();
        translog.fsync().unwrap();
        assert!(matches!(tailer.poll(), Err(ReplicationError::Corrupt { .. })), "a late op inside the gap was dropped");
        // Past the cap: an error, not a lapse.
        let mut capped = TranslogTailer::open(shard).unwrap();
        capped.set_reorder_window(8);
        for i in 0..TranslogTailer::MAX_JUMPED as i64 {
            capped.jumped.insert(-10 * (i + 2), -10 * (i + 2));
        }
        capped.last_seq_no = 100;
        capped.returned_through = 50;
        assert!(matches!(capped.mark_returned(101), Err(ReplicationError::Corrupt { .. })), "the cap lapsed silently");
    }
}
