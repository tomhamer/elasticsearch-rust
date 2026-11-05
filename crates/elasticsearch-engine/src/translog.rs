//! Append-only write-ahead log (the ES translog).
//!
//! One file per generation (`translog-<gen>.tlog`). Each frame is
//! `u32 len | u32 crc32 | body`. The body is a legacy `bincode(op)` or
//! `TLZ\x01 | LZ4-size-prefixed(original frames)` for a large batch.
//! Replay walks every generation in order. A torn frame or failed outer
//! checksum ends the recoverable prefix; [`Translog::open`] truncates that
//! tail of the newest generation (the only one a crash can tear) and fails
//! on one in an older generation (corruption). A complete, checksum-valid
//! body with an unsupported format or invalid contents fails without truncation.
//!
//! Durability: `append` issues a `write(2)` per record so a process exit
//! (not a power loss) loses nothing; [`Translog::fsync`] gives the stronger
//! guarantee and is what `flush` calls. Request-durability writes (bulk and
//! single-doc APIs) fsync through [`Translog::sync_point`] outside the
//! shard's write lock, so concurrent writers share one fsync (group commit).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{EsError, Result};

mod admitted;
pub use admitted::AdmittedBatch;

const HEADER_LEN: usize = 8;

#[cfg(test)]
static CRC_TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();

/// CRC-32 (IEEE).
pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Table-driven CRC-32 (IEEE): reference for [`crc32`].
#[cfg(test)]
fn crc32_reference(data: &[u8]) -> u32 {
    let t = CRC_TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = t[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// One logged operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranslogOp {
    /// Index (create or replace) a document.
    Index {
        id: String,
        /// The raw `_source` as a JSON string.
        source_json: String,
        seq_no: u64,
        version: u64,
        routing: Option<String>,
    },
    /// Delete a document by id.
    Delete { id: String, seq_no: u64, version: u64 },
}

impl TranslogOp {
    /// Sequence number of the op.
    pub fn seq_no(&self) -> u64 {
        match self {
            TranslogOp::Index { seq_no, .. } | TranslogOp::Delete { seq_no, .. } => *seq_no,
        }
    }
}

const BATCH_MAGIC: &[u8; 4] = b"TLZ\x01";

/// Whether appends LZ4-pack large batches: `RUSTIC_TRANSLOG_COMPRESS=true`.
/// Off by default, as ES writes its translog uncompressed: compressing
/// every bulk on the write path cost ~4% of indexing CPU; readers decode
/// either form.
/// How large batches are packed: `RUSTIC_TRANSLOG_COMPRESS=true` always
/// LZ4-packs them, `false` never; unset, a batch is packed when a sample of
/// it compresses to under half ([`Compress::Adaptive`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compress {
    Never,
    Always,
    /// Pack when the batch's first [`SAMPLE_BYTES`] compress to under half:
    /// metrics bulks (a 1,000-doc k8s bulk is ~10 MB of JSON that shrinks
    /// ~10x) stop writing their whole source to disk per bulk, while text
    /// bulks that barely compress (wikipedia), where packing was ~4% of
    /// indexing CPU, stay plain as ES's translog is.
    Adaptive,
}

/// Bytes of a batch [`Compress::Adaptive`] samples.
const SAMPLE_BYTES: usize = 64 << 10;

fn compress_batches() -> Compress {
    static ON: std::sync::OnceLock<Compress> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("RUSTIC_TRANSLOG_COMPRESS").ok().map(|v| v.to_ascii_lowercase()).as_deref() {
        Some("1" | "true") => Compress::Always,
        Some("0" | "false") => Compress::Never,
        _ => Compress::Adaptive,
    })
}
const MAX_BATCH_BYTES: usize = 256 * 1024 * 1024;

/// A physical frame can contain one legacy operation or a compressed batch.
#[cfg(test)]
pub enum TranslogRecord {
    Single(TranslogOp),
    Batch(Vec<TranslogOp>),
}
#[cfg(test)]
impl TranslogRecord {
    pub fn len(&self) -> usize {
        match self { Self::Single(_) => 1, Self::Batch(v) => v.len() }
    }
    pub fn into_ops(self) -> impl Iterator<Item = TranslogOp> {
        let (single, batch) = match self {
            Self::Single(op) => (Some(op), Vec::new()),
            Self::Batch(ops) => (None, ops),
        };
        single.into_iter().chain(batch)
    }
}

/// Decode a checksum-verified outer frame. Inner records retain their original
/// CRCs. Complete malformed compressed frames fail, never become a torn tail.
#[cfg(test)]
pub fn decode_record(body: &[u8]) -> Result<TranslogRecord> {
    decode_record_with_breakers(body, crate::memory::breakers())
}

/// Borrowed wire view: validating or inspecting sequence numbers must not
/// allocate every source string in the log. Variant/field order matches the
/// legacy bincode schema exactly; this type is never serialized on the wire.
#[derive(Debug, Deserialize)]
pub enum TranslogOpRef<'a> {
    Index { id: &'a str, source_json: &'a str, seq_no: u64, version: u64,
        #[serde(borrow)] routing: Option<&'a str> },
    Delete { id: &'a str, seq_no: u64, version: u64 },
}
impl TranslogOpRef<'_> {
    pub fn seq_no(&self) -> u64 {
        match self { Self::Index { seq_no, .. } | Self::Delete { seq_no, .. } => *seq_no }
    }
}

/// Inspect validated borrowed operations while their encoded storage is alive.
/// The callback cannot retain a view outside this call. Complete malformed
/// records still error; outer torn-tail handling belongs to the file reader.
pub fn visit_record<F>(body: &[u8], mut visit: F) -> Result<usize>
where F: for<'a> FnMut(TranslogOpRef<'a>) -> Result<()> {
    with_record_frames(body, crate::memory::breakers(), |encoded| {
        let op = bincode::deserialize(encoded).map_err(|e| EsError::internal(format!("corrupt translog: {e}")))?;
        visit(op)
    })
}

#[cfg(test)]
fn decode_record_with_breakers(body: &[u8], breakers: &'static crate::memory::Breakers) -> Result<TranslogRecord> {
    let compressed = body.starts_with(b"TLZ");
    let mut single = None;
    let mut ops = Vec::new();
    with_record_frames(body, breakers, |encoded| {
        let op = bincode::deserialize(encoded).map_err(|e| EsError::internal(format!("corrupt translog: {e}")))?;
        if compressed { ops.push(op); } else { single = Some(op); }
        Ok(())
    })?;
    Ok(if compressed { TranslogRecord::Batch(ops) } else { TranslogRecord::Single(single.expect("one legacy frame")) })
}

fn with_record_frames(body: &[u8], breakers: &'static crate::memory::Breakers, mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<usize> {
    let bad = |message: String| EsError::internal(format!("corrupt translog: {message}"));
    if !body.starts_with(b"TLZ") { visit(body)?; return Ok(1); }
    if body.get(..4) != Some(BATCH_MAGIC.as_slice()) || body.len() < 8 {
        return Err(bad("unsupported compressed batch".into()));
    }
    let size = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    if size > MAX_BATCH_BYTES { return Err(bad("batch exceeds size limit".into())); }
    // This owner covers the transient raw workspace, not callback-owned data.
    let _workspace = breakers.reserve_request(size as u64, "translog decompression workspace")?;
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(|error| EsError::new(429, "circuit_breaking_exception", format!("translog decompression allocation refused: {error}")))?;
    data.resize(size, 0);
    let decoded = lz4_flex::block::decompress_into(&body[8..], &mut data).map_err(|e| bad(e.to_string()))?;
    if decoded != size { return Err(bad("decompressed batch size mismatch".into())); }
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos < data.len() {
        if data.len() - pos < HEADER_LEN { return Err(bad("short inner header".into())); }
        let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap());
        let start = pos + HEADER_LEN;
        let end = start.checked_add(len).filter(|&e| e <= data.len()).ok_or_else(|| bad("short inner record".into()))?;
        if crc32(&data[start..end]) != crc { return Err(bad("inner checksum mismatch".into())); }
        visit(&data[start..end])?;
        count += 1;
        pos = end;
    }
    if count == 0 { return Err(bad("empty compressed batch".into())); }
    Ok(count)
}

/// Owned decode storage. Fields stay private so a consumer cannot discard the
/// reservation while keeping the decoded strings. The iterator carries the
/// owner until both its remaining allocation and every yielded operation die.
pub struct AdmittedRecord {
    ops: std::vec::IntoIter<TranslogOp>,
    owner: Arc<crate::memory::Reservation>,
}
/// Conversion storage owns both original string admission and new slots.
/// The mapping closure is move-only by contract (the replication conversion
/// transfers strings); a deep-copy conversion must separately reserve copies.
struct MappedAdmittedRecord<T> {
    ops: Vec<T>,
    _original: Arc<crate::memory::Reservation>,
    _slots: crate::memory::Reservation,
}
impl AdmittedRecord {
    pub fn first(&self) -> Option<&TranslogOp> { self.ops.as_slice().first() }
    /// Transfer one decoded operation after the destination has reserved its
    /// value/owner slots. The callback may move, but not deep-copy, payloads.
    pub fn move_next<T>(&mut self, out: &mut AdmittedBatch<T>, map: impl FnOnce(TranslogOp) -> T) {
        let op = self.ops.next().expect("pending admitted operation");
        out.push_shared_reserved(map(op), self.owner.clone());
    }
    pub fn discard_next(&mut self) { self.ops.next(); }
    fn try_map<T>(self, breakers: &'static crate::memory::Breakers, mut map: impl FnMut(TranslogOp) -> T)
        -> std::result::Result<MappedAdmittedRecord<T>, (EsError, Self)>
    {
        let Some(bytes) = self.len().checked_mul(std::mem::size_of::<T>()) else {
            return Err((allocation_error("mapped operation slots overflow"), self));
        };
        let slots = match breakers.reserve_request(bytes as u64, "translog mapped operation slots") {
            Ok(slots) => slots, Err(error) => return Err((error, self)),
        };
        let mut ops = Vec::new();
        if let Err(error) = ops.try_reserve_exact(self.len()) {
            return Err((allocation_error(error), self));
        }
        // No fallible handoff remains after consuming the input. Admission or
        // allocation refusal returns it intact for pending/requeue ownership.
        let mut output = MappedAdmittedRecord { ops, _original: self.owner.clone(), _slots: slots };
        for op in self.ops { output.ops.push(map(op)); }
        Ok(output)
    }
}

pub struct AdmittedOperation {
    op: TranslogOp,
    _owner: Arc<crate::memory::Reservation>,
}
impl std::ops::Deref for AdmittedOperation {
    type Target = TranslogOp;
    fn deref(&self) -> &TranslogOp { &self.op }
}
impl AdmittedOperation {
    /// Move payloads under their existing frame owner into destination slots.
    pub fn move_into<T>(self, out: &mut AdmittedBatch<T>, map: impl FnOnce(TranslogOp) -> T)
        -> std::result::Result<(), (EsError, Self)> {
        if let Err(error) = out.reserve_one() { return Err((error, self)); }
        out.push_shared_reserved(map(self.op), self._owner);
        Ok(())
    }
    /// Deep copies require their own reservation; sharing the original owner
    /// is valid only for moving the original strings, never for Clone.
    fn try_clone(&self, breakers: &'static crate::memory::Breakers) -> Result<Self> {
        let view = match &self.op {
            TranslogOp::Index { id, source_json, seq_no, version, routing } => TranslogOpRef::Index {
                id, source_json, seq_no: *seq_no, version: *version, routing: routing.as_deref(),
            },
            TranslogOp::Delete { id, seq_no, version } => TranslogOpRef::Delete { id, seq_no: *seq_no, version: *version },
        };
        let bytes = view.owned_string_bytes()?.checked_add(std::mem::size_of::<crate::memory::Reservation>() + 2 * std::mem::size_of::<usize>())
            .ok_or_else(|| allocation_error("copy size overflow"))?;
        let reservation = breakers.reserve_request(bytes as u64, "translog operation copy")?;
        let owner = Arc::new(reservation);
        let op = view.materialize()?;
        Ok(Self { op, _owner: owner })
    }
}
impl Iterator for AdmittedRecord {
    type Item = AdmittedOperation;
    fn next(&mut self) -> Option<Self::Item> {
        self.ops.next().map(|op| AdmittedOperation { op, _owner: self.owner.clone() })
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.ops.size_hint() }
}
impl ExactSizeIterator for AdmittedRecord {}

fn allocation_error(message: impl std::fmt::Display) -> EsError {
    EsError::new(429, "circuit_breaking_exception", format!("translog owned decode allocation refused: {message}"))
}
fn copy_admitted_string(value: &str) -> Result<String> {
    let mut out = String::new();
    out.try_reserve_exact(value.len()).map_err(allocation_error)?;
    out.push_str(value);
    Ok(out)
}
impl TranslogOpRef<'_> {
    fn owned_string_bytes(&self) -> Result<usize> {
        let (id, source, routing) = match self {
            Self::Index { id, source_json, routing, .. } => (id.len(), source_json.len(), routing.map_or(0, str::len)),
            Self::Delete { id, .. } => (id.len(), 0, 0),
        };
        id.checked_add(source).and_then(|n| n.checked_add(routing)).ok_or_else(|| allocation_error("size overflow"))
    }
    fn materialize(self) -> Result<TranslogOp> {
        Ok(match self {
            Self::Index { id, source_json, seq_no, version, routing } => TranslogOp::Index {
                id: copy_admitted_string(id)?, source_json: copy_admitted_string(source_json)?, seq_no, version,
                routing: routing.map(copy_admitted_string).transpose()?,
            },
            Self::Delete { id, seq_no, version } => TranslogOp::Delete { id: copy_admitted_string(id)?, seq_no, version },
        })
    }
}

/// One pass over the record: each operation is parsed once (one bincode
/// pass and UTF-8 check) and its owned size, with any growth of the output
/// vector, is admitted before its strings are copied. Kept private until all
/// retained consumers can accept the owner-carrying API.
pub fn decode_admitted_record(body: &[u8], breakers: &'static crate::memory::Breakers) -> Result<AdmittedRecord> {
    let mut reservation = breakers.reserve_request((std::mem::size_of::<crate::memory::Reservation>() + 2 * std::mem::size_of::<usize>()) as u64, "translog owned operations")?;
    let mut ops: Vec<TranslogOp> = Vec::new();
    // For compressed input the raw workspace is admitted separately while
    // the decoded owner grows.
    with_record_frames(body, breakers, |encoded| {
        let view: TranslogOpRef<'_> = bincode::deserialize(encoded).map_err(|e| EsError::internal(format!("corrupt translog: {e}")))?;
        if ops.len() == ops.capacity() {
            let more = ops.capacity().max(1);
            let bytes = more.checked_mul(std::mem::size_of::<TranslogOp>()).ok_or_else(|| allocation_error("operation count overflow"))?;
            reservation.grow(bytes as u64, "translog owned operations")?;
            ops.try_reserve_exact(more).map_err(allocation_error)?;
        }
        reservation.grow(view.owned_string_bytes()? as u64, "translog owned operations")?;
        ops.push(view.materialize()?);
        Ok(())
    })?;
    Ok(AdmittedRecord { ops: ops.into_iter(), owner: Arc::new(reservation) })
}

/// The write-ahead log for one shard.
pub struct Translog {
    dir: PathBuf,
    generation: u64,
    file: Arc<File>,
    /// Appends so far (all generations): the mark [`Translog::sync_point`]
    /// hands out.
    appended: u64,
    /// Test seam ([`Translog::fail_next_appends`]): appends still to fail.
    fail_appends: usize,
    /// Test seam ([`Translog::tear_next_append`]): the next append writes
    /// half its bytes, then fails.
    tear_next: bool,
    /// Test seam ([`Translog::fail_next_fsync`]): the next fsync fails.
    fail_fsync: bool,
    /// Bytes of the current generation written by successful appends.
    len: u64,
    /// The first failed write or sync (ES translog tragedy): every later
    /// append, sync and roll fails with it, so nothing is ever appended
    /// after a partial frame and acknowledged.
    failed: Option<String>,
    /// Bytes of the older generation files, keyed by the directory's
    /// modification time and the current generation: stats read the size on
    /// every call (`_stats`, `_nodes/stats`), and listing and stat-ing every
    /// file each time cost more than the rest of the shard's stats. Any file
    /// created, renamed or removed in the directory changes its mtime.
    older_bytes: std::sync::Mutex<Option<(std::time::SystemTime, u64, u64)>>,
}

pub(crate) fn file_name(generation: u64) -> String {
    format!("translog-{generation:08}.tlog")
}

/// Parse `translog-<gen>.tlog` into its generation.
fn parse_generation(name: &str) -> Option<u64> {
    name.strip_prefix("translog-")?
        .strip_suffix(".tlog")?
        .parse()
        .ok()
}

impl Translog {
    /// Open the log under `dir` whose oldest needed generation is
    /// `generation` (the commit's). A crash can tear (or zero-fill) only
    /// the newest generation on disk (a roll fsyncs the previous one
    /// before creating the next, and an open truncates and syncs before
    /// creating its own): that one keeps its readable prefix. A bad frame
    /// in any older generation from `generation` on is media corruption,
    /// not a tear: the open fails (ES `TranslogCorruptedException`)
    /// instead of dropping acknowledged ops behind it. Appends then go to
    /// a fresh generation above every one on disk, never after another
    /// file's tail.
    pub fn open(dir: &Path, generation: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut highest: Option<u64> = None;
        let generations: Vec<(u64, PathBuf)> = list_generations(dir)?.into_iter().filter(|(g, _)| *g >= generation).collect();
        let newest = generations.last().map(|(g, _)| *g);
        for (g, path) in generations {
            let good_len = validate_file(&path)?;
            let len = std::fs::metadata(&path)?.len();
            if len != good_len as u64 && Some(g) != newest {
                return Err(EsError::new(500, "translog_corrupted_exception", format!(
                    "translog generation [{g}] at {} is corrupted at byte [{good_len}] of [{len}] and is not the newest generation",
                    path.display())));
            }
            if len != good_len as u64 {
                let f = OpenOptions::new().write(true).open(&path)?;
                f.set_len(good_len as u64)?;
                f.sync_all()?;
            }
            highest = Some(highest.map_or(g, |h| h.max(g)));
        }
        let next = highest.map_or(generation, |h| h + 1);
        let file = Self::create_generation(dir, next)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            generation: next,
            file: Arc::new(file),
            appended: 0,
            fail_appends: 0,
            tear_next: false,
            fail_fsync: false,
            len: 0,
            failed: None,
            older_bytes: std::sync::Mutex::new(None),
        })
    }

    /// Create generation `next` (it must not exist: appends never follow
    /// another file's bytes) and make its directory entry durable.
    fn create_generation(dir: &Path, next: u64) -> Result<File> {
        let file = OpenOptions::new().create_new(true).append(true).read(true).open(dir.join(file_name(next)))?;
        rustic_core::platform::sync_dir(dir)?;
        Ok(file)
    }

    /// Current generation number.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Test seam: the next `count` appends fail without writing anything.
    #[doc(hidden)]
    pub fn fail_next_appends(&mut self, count: usize) {
        self.fail_appends = count;
    }

    /// Test seam: the next append writes half of its bytes and fails (a
    /// partial `write(2)`: ENOSPC / EIO mid-frame).
    #[doc(hidden)]
    pub fn tear_next_append(&mut self) {
        self.tear_next = true;
    }

    /// Test seam: the next fsync (of a flush's roll, or a direct one) fails
    /// as `fsync(2)` returning EIO would.
    #[doc(hidden)]
    pub fn fail_next_fsync(&mut self) {
        self.fail_fsync = true;
    }

    /// The failure that closed the translog, if any.
    pub fn failed(&self) -> Option<&str> {
        self.failed.as_deref()
    }

    /// Close the translog after a failure seen outside it (a group-commit
    /// fsync of its file): every later append, sync and roll fails. The
    /// first reason wins.
    pub fn fail(&mut self, reason: &str) {
        if self.failed.is_none() {
            self.failed = Some(reason.to_string());
        }
    }

    fn check_failed(&self) -> Result<()> {
        match &self.failed {
            Some(reason) => Err(EsError::internal(format!("translog is closed after a failed write: {reason}"))),
            None => Ok(()),
        }
    }

    /// Write `bytes` at the end of the current generation. A failed write
    /// is cut back to the last whole frame (best effort) and closes the
    /// translog for good.
    fn write_frames(&mut self, bytes: &[u8]) -> Result<()> {
        self.check_failed()?;
        let written = if std::mem::take(&mut self.tear_next) {
            (&*self.file).write_all(&bytes[..bytes.len() / 2])
                .and_then(|_| Err(std::io::Error::other("injected partial translog write")))
        } else {
            (&*self.file).write_all(bytes)
        };
        match written {
            Ok(()) => {
                self.len += bytes.len() as u64;
                self.appended += 1;
                Ok(())
            }
            Err(e) => {
                let _ = self.file.set_len(self.len);
                self.failed = Some(e.to_string());
                Err(e.into())
            }
        }
    }

    fn injected_failure(&mut self) -> Result<()> {
        if self.fail_appends > 0 {
            self.fail_appends -= 1;
            return Err(EsError::internal("injected translog append failure"));
        }
        Ok(())
    }

    /// Append one op (one `write(2)`, no fsync).
    pub fn append(&mut self, op: &TranslogOp) -> Result<()> {
        self.injected_failure()?;
        let body = bincode::serialize(op).map_err(|e| EsError::internal(format!("translog: {e}")))?;
        let mut rec = Vec::with_capacity(HEADER_LEN + body.len());
        rec.extend_from_slice(&(body.len() as u32).to_le_bytes());
        rec.extend_from_slice(&crc32(&body).to_le_bytes());
        rec.extend_from_slice(&body);
        self.write_frames(&rec)
    }

    /// Append several ops with a single `write(2)`.
    pub fn append_many(&mut self, ops: &[TranslogOp]) -> Result<()> {
        let encoded = Self::encode_batch(ops)?;
        self.append_encoded(&encoded)
    }

    /// [`Translog::append_many`] with batch compression on or off
    /// whatever the process setting (tests of compressed frames).
    #[doc(hidden)]
    pub fn append_many_with(&mut self, ops: &[TranslogOp], compress: bool) -> Result<()> {
        let encoded = Self::encode_batch_mode(ops, if compress { Compress::Always } else { Compress::Never })?;
        self.append_encoded(&encoded)
    }

    /// The bytes [`Translog::append_many`] writes for `ops` (records, LZ4-
    /// packed as one batch frame when large): pure, so writers serialize and
    /// compress before taking the translog lock and only the `write(2)` is
    /// serialized.
    pub fn encode_batch(ops: &[TranslogOp]) -> Result<Vec<u8>> {
        Self::encode_batch_mode(ops, compress_batches())
    }

    /// [`Translog::encode_batch`]; `compress`: pack large batches as one
    /// LZ4 frame (readers take both forms).
    pub fn encode_batch_with(ops: &[TranslogOp], compress: bool) -> Result<Vec<u8>> {
        Self::encode_batch_mode(ops, if compress { Compress::Always } else { Compress::Never })
    }

    /// [`Translog::encode_batch`] with packing `mode`.
    pub fn encode_batch_mode(ops: &[TranslogOp], mode: Compress) -> Result<Vec<u8>> {
        let size: u64 = ops.iter().map(|op| bincode::serialized_size(op).unwrap_or(0) + HEADER_LEN as u64).sum();
        let mut buf = Vec::with_capacity(size as usize);
        for op in ops {
            // Header placeholder, then the body serialized in place.
            let at = buf.len();
            buf.extend_from_slice(&[0u8; HEADER_LEN]);
            bincode::serialize_into(&mut buf, op).map_err(|e| EsError::internal(format!("translog: {e}")))?;
            let body_len = buf.len() - at - HEADER_LEN;
            let crc = crc32(&buf[at + HEADER_LEN..]);
            buf[at..at + 4].copy_from_slice(&(body_len as u32).to_le_bytes());
            buf[at + 4..at + 8].copy_from_slice(&crc.to_le_bytes());
        }
        let eligible = ops.len() >= 64 && (4096..=MAX_BATCH_BYTES).contains(&buf.len());
        let compress = eligible
            && match mode {
                Compress::Never => false,
                Compress::Always => true,
                Compress::Adaptive => {
                    let sample = &buf[..buf.len().min(SAMPLE_BYTES)];
                    lz4_flex::compress(sample).len() * 2 < sample.len()
                }
            };
        if compress {
            let packed = lz4_flex::compress_prepend_size(&buf);
            if HEADER_LEN + BATCH_MAGIC.len() + packed.len() < buf.len() {
                let mut body = Vec::with_capacity(BATCH_MAGIC.len() + packed.len());
                body.extend_from_slice(BATCH_MAGIC);
                body.extend_from_slice(&packed);
                buf.clear();
                buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
                buf.extend_from_slice(&crc32(&body).to_le_bytes());
                buf.extend_from_slice(&body);
            }
        }
        Ok(buf)
    }

    /// Write bytes from [`Translog::encode_batch`] (one `write(2)`, no fsync).
    pub fn append_encoded(&mut self, encoded: &[u8]) -> Result<()> {
        self.check_failed()?;
        self.injected_failure()?;
        self.write_frames(encoded)
    }

    /// The current generation's file and the append mark: once that file
    /// is `sync_data`'d, every append up to the mark is durable (a roll
    /// fsyncs the generation it leaves).
    pub fn sync_point(&self) -> (Arc<File>, u64) {
        (self.file.clone(), self.appended)
    }

    /// Flush the current generation to stable storage.
    pub fn fsync(&mut self) -> Result<()> {
        self.check_failed()?;
        let synced = if std::mem::take(&mut self.fail_fsync) { Err(std::io::Error::other("injected translog fsync failure")) } else { self.file.sync_data() };
        if let Err(e) = synced {
            self.failed = Some(e.to_string());
            return Err(e.into());
        }
        Ok(())
    }

    /// Start a new generation; subsequent appends go to the new file. The
    /// old generation stays on disk until [`Translog::trim_to`].
    pub fn roll(&mut self) -> Result<u64> {
        self.fsync()?;
        let next = self.generation + 1;
        self.file = Arc::new(Self::create_generation(&self.dir, next)?);
        self.generation = next;
        self.len = 0;
        Ok(next)
    }

    /// Delete every generation file older than `generation`.
    pub fn trim_to(&mut self, generation: u64) -> Result<()> {
        for (g, path) in list_generations(&self.dir)? {
            if g < generation {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    /// All ops with `seq_no >= from_seq_no`, across every generation file
    /// present, in file then record order. Stops silently at a corrupt tail.
    /// Every op is held at once under the request breaker: store recovery
    /// streams them with [`Translog::replay_stream`] instead.
    pub fn replay(&self, from_seq_no: u64) -> Result<AdmittedBatch<TranslogOp>> {
        let mut out = AdmittedBatch::new(crate::memory::breakers())?;
        for (_, path) in list_generations(&self.dir)? {
            scan_from_admitted(&path, 0, |op, _| {
                if op.seq_no() >= from_seq_no { op.move_into(&mut out, |op| op).map_err(|(error, _)| error)?; }
                Ok(true)
            })?;
        }
        Ok(out)
    }

    /// The ops [`Translog::replay`] returns, less those whose seq_no is in
    /// `skip`, read lazily a bounded chunk at a time
    /// ([`TranslogReplay::for_each_chunk`]). The generation files present
    /// now are opened and their lengths taken: a flush or trim while the
    /// replay runs cannot remove what it has still to read, and appends
    /// after this point are not replayed.
    pub fn replay_stream(&self, from_seq_no: u64, skip: std::collections::HashSet<i64>) -> Result<TranslogReplay> {
        let mut files = Vec::new();
        for (_, path) in list_generations(&self.dir)? {
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let len = file.metadata()?.len();
            files.push((file, len));
        }
        Ok(TranslogReplay { files, from_seq_no, skip })
    }

    /// Total bytes across all generation files.
    pub fn size_bytes(&self) -> u64 {
        let mtime = std::fs::metadata(&self.dir).and_then(|m| m.modified()).ok();
        if let Some(t) = mtime {
            let cached = *self.older_bytes.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((at, generation, older)) = cached {
                if at == t && generation == self.generation {
                    // The current generation only grows by this writer's appends.
                    return older + self.current_len();
                }
            }
        }
        let Ok(files) = list_generations(&self.dir) else { return 0 };
        let mut older = 0u64;
        let mut current = 0u64;
        for (g, p) in &files {
            let len = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            if *g == self.generation { current = len; } else { older += len; }
        }
        if let Some(t) = mtime {
            *self.older_bytes.lock().unwrap_or_else(|e| e.into_inner()) = Some((t, self.generation, older));
        }
        older + current
    }

    /// Bytes of the current generation file (its successful appends; a torn
    /// write after a failure is counted once the size is read from disk).
    fn current_len(&self) -> u64 {
        if self.failed.is_some() {
            return self.file.metadata().map(|m| m.len()).unwrap_or(self.len);
        }
        self.len
    }
}

/// Ops a store recovery replays per chunk ([`TranslogReplay::for_each_chunk`]).
pub const REPLAY_CHUNK_OPS: usize = 256;

/// The translog ops a store recovery replays, read from the generation
/// files a chunk at a time ([`Translog::replay_stream`]): only one chunk of
/// operations (and the frame being decoded) is held at once, however much
/// the translog holds (ES `runTranslogRecovery` reads a `Translog.Snapshot`
/// one op at a time).
pub struct TranslogReplay {
    /// Open generation files, oldest first, with their lengths at creation.
    files: Vec<(File, u64)>,
    from_seq_no: u64,
    /// Seq_nos above the checkpoint the commit already processed.
    skip: std::collections::HashSet<i64>,
}

/// Where the next chunk of a [`TranslogReplay`] starts: a frame, and how
/// many of its ops earlier chunks consumed (a compressed frame holds many).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct ReplayCursor {
    file: usize,
    pos: u64,
    consumed: usize,
}

impl TranslogReplay {
    /// Nothing to replay (a new, empty store).
    pub fn empty() -> TranslogReplay {
        TranslogReplay { files: Vec::new(), from_seq_no: 0, skip: Default::default() }
    }

    /// Hand every op to `apply` in order, at most `max_ops` per chunk.
    /// The ops were acknowledged: reading and applying them is a
    /// [`crate::memory::Replay`] (ES `recoverFromTranslog` applies them with
    /// no circuit breaker), whose reservations are accounted but never
    /// refused, so a replay never waits for memory and never fails for it.
    /// Memory stays bounded by the chunk (and the frame being decoded).
    /// Errors (I/O, corruption, an allocation the system refuses) are
    /// returned.
    pub fn for_each_chunk(&self, max_ops: usize, mut apply: impl FnMut(AdmittedBatch<TranslogOp>) -> Result<()>) -> Result<()> {
        let max_ops = max_ops.max(1);
        crate::memory::replaying("local translog recovery", || {
            let mut cursor = ReplayCursor::default();
            loop {
                let chunk = self.read_chunk(&mut cursor, max_ops)?;
                if chunk.is_empty() { return Ok(()); }
                apply(chunk)?;
            }
        })
    }

    /// Every op at once (tests and small logs; recovery uses
    /// [`TranslogReplay::for_each_chunk`]).
    pub fn collect(&self) -> Result<AdmittedBatch<TranslogOp>> {
        let mut cursor = ReplayCursor::default();
        self.read_chunk(&mut cursor, usize::MAX)
    }

    /// Up to `max_ops` ops to replay from `cursor`, which moves past them
    /// (and past the ops filtered out) only when the whole chunk was read.
    /// Fewer than `max_ops` only at the end. Each file stops silently at a
    /// torn or corrupt tail, as [`Translog::replay`].
    fn read_chunk(&self, cursor: &mut ReplayCursor, max_ops: usize) -> Result<AdmittedBatch<TranslogOp>> {
        let mut c = *cursor;
        let mut out = AdmittedBatch::new(crate::memory::breakers())?;
        let mut header = [0u8; HEADER_LEN];
        'files: while c.file < self.files.len() && out.len() < max_ops {
            let (mut file, file_len): (&File, u64) = (&self.files[c.file].0, self.files[c.file].1);
            let next_file = |c: &mut ReplayCursor| *c = ReplayCursor { file: c.file + 1, pos: 0, consumed: 0 };
            if file_len.saturating_sub(c.pos) < HEADER_LEN as u64 { next_file(&mut c); continue; }
            file.seek(SeekFrom::Start(c.pos))?;
            file.read_exact(&mut header)?;
            let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(header[4..].try_into().unwrap());
            let Some(end) = c.pos.checked_add(HEADER_LEN as u64).and_then(|n| n.checked_add(len as u64)) else { next_file(&mut c); continue };
            if len == 0 || end > file_len { next_file(&mut c); continue; }
            // Forced under the replay ([`TranslogReplay::for_each_chunk`]):
            // one acknowledged op may be larger than the breaker allows.
            let _encoded = crate::memory::breakers().reserve_request(len as u64, "translog replay frame")?;
            let mut body = Vec::new();
            body.try_reserve_exact(len).map_err(allocation_error)?;
            body.resize(len, 0);
            file.read_exact(&mut body)?;
            if crc32(&body) != crc { next_file(&mut c); continue; }
            let decoded = decode_admitted_record(&body, crate::memory::breakers())?;
            drop(body);
            let count = decoded.len();
            for (i, op) in decoded.enumerate() {
                if i < c.consumed { continue; }
                if out.len() >= max_ops { break 'files; }
                c.consumed = i + 1;
                let seq_no = op.seq_no();
                if seq_no >= self.from_seq_no && !self.skip.contains(&(seq_no as i64)) {
                    op.move_into(&mut out, |op| op).map_err(|(error, _)| error)?;
                }
            }
            if c.consumed >= count { c.pos = end; c.consumed = 0; }
        }
        *cursor = c;
        Ok(out)
    }
}

/// Generation files under `dir`, sorted by generation.
pub(crate) fn list_generations(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut v = Vec::new();
    if !dir.exists() {
        return Ok(v);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Some(g) = entry.file_name().to_str().and_then(parse_generation) {
            v.push((g, entry.path()));
        }
    }
    v.sort();
    Ok(v)
}

/// Validate one physical frame at a time without materializing owned document
/// strings. Admission failures and checksum-valid malformed records propagate:
/// neither may be mistaken for a crash tail eligible for truncation.
fn validate_file(path: &Path) -> Result<u64> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut pos = 0u64;
    while file_len.saturating_sub(pos) >= HEADER_LEN as u64 {
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)?;
        let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(header[4..].try_into().unwrap());
        let end = pos + HEADER_LEN as u64 + len as u64;
        if len == 0 || end > file_len { break; }
        // The body is destroyed before its owner. There is no retained buffer
        // or old/new allocation overlap between frames.
        let _frame = crate::memory::breakers().reserve_request(len as u64, "translog validation frame")?;
        let mut body = Vec::new();
        body.try_reserve_exact(len).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("translog validation allocation refused: {e}")))?;
        body.resize(len, 0);
        file.read_exact(&mut body)?;
        if crc32(&body) != crc { break; }
        visit_record(&body, |_| Ok(()))?;
        pos = end;
    }
    Ok(pos)
}

/// Framed admitted scan. Encoded storage is bounded to the actual file frame;
/// each yielded operation retains the decoded allocation until moved or dropped.
/// `end` stays at the physical frame start until its last inner operation.
/// Callback errors stop before any caller-owned cursor can be advanced.
pub(crate) fn scan_from_admitted(path: &Path, offset: u64,
    mut f: impl FnMut(AdmittedOperation, u64) -> Result<bool>) -> Result<()> {
    scan_admitted_prefix(path, offset, &mut f).map(|_| ())
}
fn scan_admitted_prefix(path: &Path, offset: u64,
    f: &mut impl FnMut(AdmittedOperation, u64) -> Result<bool>) -> Result<u64> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };
    let file_len = file.metadata()?.len();
    file.seek(SeekFrom::Start(offset))?;
    let mut pos = offset;
    let mut header = [0u8; HEADER_LEN];
    while file_len.saturating_sub(pos) >= HEADER_LEN as u64 {
        file.read_exact(&mut header)?;
        let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(header[4..].try_into().unwrap());
        let Some(end) = pos.checked_add(HEADER_LEN as u64).and_then(|n| n.checked_add(len as u64)) else { break };
        if len == 0 || end > file_len { break; }
        let _encoded = crate::memory::breakers().reserve_request(len as u64, "translog scan frame")?;
        let mut body = Vec::new();
        body.try_reserve_exact(len).map_err(allocation_error)?;
        body.resize(len, 0);
        file.read_exact(&mut body)?;
        if crc32(&body) != crc { break; }
        let decoded = decode_admitted_record(&body, crate::memory::breakers())?;
        let count = decoded.len();
        for (i, operation) in decoded.enumerate() {
            let resume = if i + 1 == count { end } else { pos };
            if !f(operation, resume)? { return Ok(resume); }
        }
        pos = end;
    }
    Ok(pos)
}
pub(crate) fn read_file_admitted(path: &Path) -> Result<(AdmittedBatch<TranslogOp>, usize)> {
    let mut out = AdmittedBatch::new(crate::memory::breakers())?;
    let end = scan_admitted_prefix(path, 0, &mut |op, _| {
        op.move_into(&mut out, |op| op).map_err(|(error, _)| error)?;
        Ok(true)
    })?;
    Ok((out, usize::try_from(end).map_err(|_| allocation_error("file offset overflow"))?))
}

/// Recovery-source listing whose paths and backing array retain their
/// before-growth admission until the caller has copied them into its own
/// captured-source owner. OS directory iterator buffers remain OS inputs.
pub(crate) fn list_generations_admitted(dir: &Path) -> Result<rustic_core::admission::Allocation<Vec<(u64, PathBuf)>>> {
    if !dir.exists() {
        return Ok(rustic_core::admission::Allocation::new(Vec::new(), rustic_core::admission::shared().reserve(0, "translog generation listing")?));
    }
    list_named_generations_admitted(dir, parse_generation, false)
}

/// `ignore_entry_errors` preserves the historical side-log iterator policy;
/// admission errors are never treated as missing entries.
pub(crate) fn list_named_generations_admitted(dir: &Path, parse: fn(&str) -> Option<u64>, ignore_entry_errors: bool)
    -> Result<rustic_core::admission::Allocation<Vec<(u64, PathBuf)>>> {
    use rustic_core::admission::{self, Allocation};
    let mut memory=admission::shared().reserve(0,"history generation listing")?;
    let mut files:Vec<(u64,PathBuf)>=Vec::new();
    let entries=match std::fs::read_dir(dir) {
        Ok(entries)=>entries,
        Err(error) if ignore_entry_errors && error.kind()==std::io::ErrorKind::NotFound=>return Ok(Allocation::new(files,memory)),
        Err(error)=>return Err(error.into()),
    };
    for entry in entries {
        let entry=match entry {Ok(entry)=>entry,Err(_) if ignore_entry_errors=>continue,Err(error)=>return Err(error.into())};
        let name=entry.file_name();
        let Some(generation)=name.to_str().and_then(parse) else {continue};
        if files.len()==files.capacity() {
            let old=files.capacity();
            let capacity=old.max(4).checked_mul(2).ok_or_else(||EsError::new(429,"circuit_breaking_exception","generation listing capacity overflow"))?;
            let bytes=capacity.checked_mul(std::mem::size_of::<(u64,PathBuf)>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","generation listing capacity overflow"))?;
            // Old backing remains live while the replacement is allocated.
            memory.grow(bytes as u64,"generation listing replacement")?;
            files.try_reserve_exact(capacity-files.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",format!("generation listing allocation refused: {e}")))?;
            memory.shrink((old*std::mem::size_of::<(u64,PathBuf)>()) as u64);
        }
        let bytes=dir.as_os_str().as_encoded_bytes().len().checked_add(name.as_encoded_bytes().len()).and_then(|n|n.checked_add(1))
            .ok_or_else(||EsError::new(429,"circuit_breaking_exception","generation path capacity overflow"))?;
        memory.grow(bytes as u64,"generation listing path")?;
        let mut path=PathBuf::new();
        path.try_reserve_exact(bytes).map_err(|e|EsError::new(429,"circuit_breaking_exception",format!("generation path allocation refused: {e}")))?;
        path.push(dir);path.push(name);
        files.push((generation,path));
    }
    // Same total (generation,path) order, without stable-sort workspace.
    files.sort_unstable();
    Ok(Allocation::new(files,memory))
}

/// Decode a generation file. Returns the good ops and the byte length of
/// the valid prefix (everything after it is torn or corrupt).
#[cfg(test)]
pub(crate) fn read_file(path: &Path) -> Result<(Vec<TranslogOp>, usize)> {
    let mut data = Vec::new();
    match File::open(path) {
        Ok(mut f) => {
            f.read_to_end(&mut data)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e.into()),
    }
    let mut ops = Vec::new();
    let mut pos = 0usize;
    while pos + HEADER_LEN <= data.len() {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        let start = pos + HEADER_LEN;
        let Some(end) = start.checked_add(len) else { break };
        // A zero-length record is a zero-filled tail (no record is empty).
        if len == 0 || end > data.len() {
            break; // torn tail
        }
        let body = &data[start..end];
        if crc32(body) != crc {
            break; // corrupt record
        }
        ops.extend(decode_record(body)?.into_ops());
        pos = end;
    }
    Ok((ops, pos))
}

/// Decode the records of a generation file from byte `offset` on, one at
/// a time, calling `f(op, end)` (`end` = the offset just past the record)
/// until it returns `false`. Stops at the end of the file or at a torn or
/// corrupt record. Only what is consumed is read (change feeds resume
/// where they stopped instead of re-reading the generation).
#[cfg(test)]
pub(crate) fn scan_from(path: &Path, offset: u64, mut f: impl FnMut(TranslogOp, u64) -> bool) -> Result<()> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let file_len = file.metadata()?.len();
    let mut r = std::io::BufReader::with_capacity(1 << 16, file);
    r.seek(SeekFrom::Start(offset))?;
    let mut pos = offset;
    let mut hdr = [0u8; HEADER_LEN];
    let mut body = Vec::new();
    while pos + HEADER_LEN as u64 <= file_len {
        r.read_exact(&mut hdr)?;
        let len = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as u64;
        let crc = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let end = pos + HEADER_LEN as u64 + len;
        if len == 0 || end > file_len {
            break; // torn (or zero-filled) tail
        }
        body.resize(len as usize, 0);
        r.read_exact(&mut body)?;
        if crc32(&body) != crc {
            break;
        }
        let decoded = decode_record(&body)?;
        let count = decoded.len();
        for (i, op) in decoded.into_ops().enumerate() {
            // A change-feed cursor also stores next_seq_no. Keep the physical
            // frame start until its final op so a partial page cannot lose ops.
            let resume = if i + 1 == count { end } else { pos };
            if !f(op, resume) { return Ok(()); }
        }
        pos = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn admitted_decode_owner_survives_partial_consumption_and_refuses_before_copy() {
        let expected = index("retained", 17);
        let body = bincode::serialize(&expected).unwrap();
        let budget = crate::memory::Breakers::leak_for_tests(1024 * 1024);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let used = || budget.stats().into_iter().find(|s| s.name == "request").unwrap().estimated;
        let mut decoded = decode_admitted_record(&body, budget).unwrap();
        assert!(used() > 0);
        let held = decoded.next().unwrap();
        assert_eq!(held.op, expected);
        drop(decoded);
        assert!(used() > 0, "yielded operation must retain decoded storage admission");
        let before_copy = used();
        let copied = held.try_clone(budget).unwrap();
        assert!(used() > before_copy, "deep copy needs an independent reservation");
        assert_eq!(copied.op, held.op);
        drop(held);
        assert!(used() > 0, "copy retains its own admission after source drop");
        drop(copied);
        assert_eq!(used(), 0);
        let tiny = crate::memory::Breakers::leak_for_tests(1);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let error = decode_admitted_record(&body, tiny).err().unwrap();
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert_eq!(tiny.stats().into_iter().find(|s| s.name == "request").unwrap().estimated, 0);
    }

    #[test]
    fn admitted_conversion_refusal_returns_unconsumed_operations() {
        let expected = index("handoff", 23);
        let body = bincode::serialize(&expected).unwrap();
        let budget = crate::memory::Breakers::leak_for_tests(1024 * 1024);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let tiny = crate::memory::Breakers::leak_for_tests(1);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let decoded = decode_admitted_record(&body, budget).unwrap();
        let mut called = false;
        let (error, intact) = decoded.try_map(tiny, |op| { called = true; op }).err().unwrap();
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert!(!called, "conversion must not consume an acknowledged input on refusal");
        assert_eq!(intact.len(), 1);
        let mapped = match intact.try_map(budget, |op| op) { Ok(v) => v, Err((error, _)) => panic!("{error}") };
        assert_eq!(mapped.ops, vec![expected]);
        assert!(budget.stats().into_iter().find(|s| s.name == "request").unwrap().estimated > 0);
        drop(mapped);
        assert_eq!(budget.stats().into_iter().find(|s| s.name == "request").unwrap().estimated, 0);
    }

    /// Adaptive packing: a batch of repetitive metrics-like sources is
    /// LZ4-packed, one of incompressible sources stays plain; both decode.
    #[test]
    fn adaptive_batches_pack_only_compressible_sources() {
        let metrics: Vec<TranslogOp> = (0..200).map(|i| TranslogOp::Index {
            id: format!("id-{i}"), seq_no: i, version: 1, routing: None,
            source_json: format!("{{\"@timestamp\":{i},\"host\":{{\"ip\":[{}]}}}}", (0..150).map(|k| format!("\"10.0.{}.1\"", k % 251)).collect::<Vec<_>>().join(",")),
        }).collect();
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let noise: Vec<TranslogOp> = (0..200).map(|i| TranslogOp::Index {
            id: format!("id-{i}"), seq_no: i, version: 1, routing: None,
            source_json: (0..2000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; char::from(b'!' + (x % 90) as u8) }).collect(),
        }).collect();
        let packed = Translog::encode_batch_mode(&metrics, Compress::Adaptive).unwrap();
        assert!(packed[HEADER_LEN..].starts_with(BATCH_MAGIC));
        assert!(packed.len() * 4 < Translog::encode_batch_mode(&metrics, Compress::Never).unwrap().len());
        let plain = Translog::encode_batch_mode(&noise, Compress::Adaptive).unwrap();
        assert_eq!(plain, Translog::encode_batch_mode(&noise, Compress::Never).unwrap());
        for (bytes, ops) in [(&packed, &metrics), (&plain, &noise)] {
            let mut pos = 0;
            let mut got = Vec::new();
            while pos < bytes.len() {
                let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                got.extend(decode_record(&bytes[pos + HEADER_LEN..pos + HEADER_LEN + len]).unwrap().into_ops());
                pos += HEADER_LEN + len;
            }
            assert_eq!(&got, ops);
        }
    }

    #[test]
    fn borrowed_record_schema_matches_owned_validation() {
        let ops = [index("utf8-\u{03bb}", 42), TranslogOp::Index {
            id: "routed".into(), source_json: "{}".into(), seq_no: 43,
            version: 7, routing: Some("route".into()),
        }, TranslogOp::Delete { id: "deleted".into(), seq_no: 44, version: 9 }];
        for op in &ops {
            let mut bytes = bincode::serialize(op).unwrap();
            // Legacy deserialize accepts trailing bytes; borrowed validation
            // must retain this existing format behavior.
            for trailing in [false, true] {
                if trailing { bytes.extend_from_slice(&[0xde, 0xad]); }
                let mut seq = None;
                assert_eq!(visit_record(&bytes, |view| { seq = Some(view.seq_no()); Ok(()) }).unwrap(), 1);
                assert_eq!(seq, Some(op.seq_no()));
                assert_eq!(decode_record(&bytes).unwrap().into_ops().next(), Some(op.clone()));
            }
        }
        let legacy = bincode::serialize(&ops[0]).unwrap();
        let mut invalid_tag = legacy.clone();
        invalid_tag[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut invalid_utf8 = legacy.clone();
        invalid_utf8[12] = 0xff; // fixed-int enum tag + string length
        for bad in [invalid_tag, invalid_utf8, legacy[..legacy.len()-1].to_vec()] {
            assert!(decode_record(&bad).is_err());
            assert!(visit_record(&bad, |_| Ok(())).is_err());
        }
        let repeated: Vec<_> = ops.iter().cycle().take(600).cloned().collect();
        let encoded = Translog::encode_batch_with(&repeated, true).unwrap();
        assert!(encoded[HEADER_LEN..].starts_with(BATCH_MAGIC));
        let mut seqs = Vec::new();
        let count = visit_record(&encoded[HEADER_LEN..], |view| { seqs.push(view.seq_no()); Ok(()) }).unwrap();
        assert_eq!(count, repeated.len());
        assert_eq!(seqs, repeated.iter().map(TranslogOp::seq_no).collect::<Vec<_>>());
    }

    #[test]
    fn borrowed_compressed_validation_rejects_inner_crc_and_preserves_file_on_error() {
        let raw = bincode::serialize(&index("crc", 9)).unwrap();
        let mut inner = Vec::new();
        inner.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        inner.extend_from_slice(&(crc32(&raw) ^ 1).to_le_bytes());
        inner.extend_from_slice(&raw);
        let mut body = BATCH_MAGIC.to_vec();
        body.extend_from_slice(&lz4_flex::compress_prepend_size(&inner));
        assert!(visit_record(&body, |_| Ok(())).is_err());
        assert!(decode_record(&body).is_err());
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(file_name(1));
        let mut frame = Vec::new();
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc32(&body).to_le_bytes());
        frame.extend_from_slice(&body);
        std::fs::write(&path, &frame).unwrap();
        assert!(Translog::open(tmp.path(), 1).is_err());
        assert_eq!(std::fs::read(path).unwrap(), frame, "valid outer CRC and malformed contents must never truncate");
    }

    #[test]
    fn compressed_batch_replay_and_paged_scan_preserve_every_op() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        let ops: Vec<_> = (0..200).map(|n| index(&format!("d{n}"), n)).collect();
        log.append_encoded(&Translog::encode_batch_with(&ops, true).unwrap()).unwrap();
        log.fsync().unwrap();
        let path = tmp.path().join(file_name(1));
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[8..12], BATCH_MAGIC);
        assert_eq!(log.replay(0).unwrap(), ops);
        assert_eq!(log.replay(57).unwrap(), ops[57..]);
        let (mut offset, mut next) = (0u64, 0u64);
        let mut all = Vec::new();
        while all.len() < ops.len() {
            let mut page = Vec::new();
            scan_from(&path, offset, |op, resume| {
                if op.seq_no() < next { offset = resume; return true; }
                if page.len() == 7 { return false; }
                next = op.seq_no() + 1;
                offset = resume;
                page.push(op);
                true
            }).unwrap();
            assert!(!page.is_empty());
            all.extend(page);
        }
        assert_eq!(all, ops);
        assert_eq!(offset, bytes.len() as u64);
        drop(log);
        OpenOptions::new().append(true).open(&path).unwrap().write_all(&[10,0,0]).unwrap();
        let log = Translog::open(tmp.path(), 1).unwrap();
        assert_eq!(log.replay(0).unwrap(), ops);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes.len() as u64);
    }

    #[test]
    fn compressed_workspace_is_admitted_before_decode_and_released_on_all_exits() {
        let budget = crate::memory::Breakers::leak_for_tests(64);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        // Malformed compressed bytes with a large otherwise permitted size:
        // admission must fail before decompression examines/allocates the body.
        let refused = [BATCH_MAGIC.as_slice(), &4096u32.to_le_bytes(), &[0xff]].concat();
        let error = decode_record_with_breakers(&refused, budget).err().unwrap();
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert_eq!(error.status, 429);
        assert_eq!(budget.stats().into_iter().find(|b| b.name == "request").unwrap().estimated, 0);

        let budget = crate::memory::Breakers::leak_for_tests(1024 * 1024);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let expected: Vec<_> = (0..500).map(|seq| index("same-id", seq)).collect();
        let encoded = Translog::encode_batch_with(&expected, true).unwrap();
        let body = &encoded[HEADER_LEN..];
        assert!(body.starts_with(BATCH_MAGIC), "fixture must exercise compressed decoding");
        let decoded = decode_record_with_breakers(body, budget).unwrap();
        assert_eq!(decoded.into_ops().collect::<Vec<_>>(), expected);
        assert_eq!(budget.stats().into_iter().find(|b| b.name == "request").unwrap().estimated, 0);
        let malformed = [BATCH_MAGIC.as_slice(), &32u32.to_le_bytes(), &[0xff]].concat();
        assert!(decode_record_with_breakers(&malformed, budget).is_err());
        assert_eq!(budget.stats().into_iter().find(|b| b.name == "request").unwrap().estimated, 0);
    }

    #[test]
    fn complete_bad_compressed_translog_is_not_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(file_name(1));
        for body in [b"TLZ\x02future".to_vec(), [b"TLZ\x01".as_slice(), &u32::MAX.to_le_bytes()].concat(), b"TLZ\x01".to_vec()] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&crc32(&body).to_le_bytes());
            bytes.extend_from_slice(&body);
            std::fs::write(&path, &bytes).unwrap();
            assert!(Translog::open(tmp.path(), 1).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }

    /// Store recovery streams a translog far bigger than the request
    /// breaker allows at once: chunks stay bounded, filtered seq_nos are
    /// skipped, compressed frames split across chunks lose nothing, and the
    /// breaker never refuses or delays the replay (its reservations are
    /// forced, ES `recoverFromTranslog`): a chunk wider than the limit
    /// replays, is counted on the breaker and noted once on the replay,
    /// with no trip. A refusal `apply` returns is returned, not retried.
    #[test]
    fn replay_stream_bounds_memory_and_is_never_refused_by_the_breaker() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        let padding = "x".repeat(400);
        let big = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: format!("{{\"n\":{seq},\"p\":\"{padding}\"}}"), seq_no: seq, version: 1, routing: None };
        let mut ops: Vec<TranslogOp> = Vec::new();
        for seq in 0..3000 {
            ops.push(big(seq));
            log.append(ops.last().unwrap()).unwrap();
        }
        log.roll().unwrap();
        // A compressed frame in the second generation, wider than a chunk.
        let batch: Vec<TranslogOp> = (3000..3600).map(big).collect();
        log.append_many_with(&batch, true).unwrap();
        ops.extend(batch);
        log.fsync().unwrap();

        let budget = crate::memory::Breakers::leak_for_tests(1 << 30);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        // Far below the ~2mb the ops need at once, and below one chunk of
        // them (~110kb) too.
        budget.request.set_limit(64 * 1024);
        let scope = crate::memory::RequestScope::new("translog replay test", budget);
        crate::memory::scoped(Some(scope), || {
            let error = log.replay(0).err().expect("the whole translog, outside a replay, exceeds the breaker");
            assert_eq!(error.status, 429);
            let tripped = budget.request.tripped();

            let skip: std::collections::HashSet<i64> = [5, 3100].into_iter().collect();
            let replay = log.replay_stream(0, skip.clone()).unwrap();
            let watched = crate::memory::Replay::new("translog replay test");
            let (mut seen, mut chunks, mut peak) = (Vec::new(), 0, 0u64);
            watched.run(|| replay.for_each_chunk(REPLAY_CHUNK_OPS, |chunk| {
                assert!(chunk.len() <= REPLAY_CHUNK_OPS);
                chunks += 1;
                peak = peak.max(budget.request.reserved());
                chunk.try_for_each(|op| { seen.push(op.seq_no()); Ok(()) })
            })).unwrap();
            let expected: Vec<u64> = ops.iter().map(TranslogOp::seq_no).filter(|s| !skip.contains(&(*s as i64))).collect();
            assert_eq!(seen, expected, "every op once, in order");
            assert!(chunks > ops.len() / REPLAY_CHUNK_OPS);
            assert!(peak > 64 * 1024, "a chunk is charged past the limit: {peak}");
            assert!(peak < 1024 * 1024, "only a chunk (and the frame being decoded) is held at once: {peak}");
            assert!(watched.over_limit() > 0, "going past the limit is noted on the replay");
            assert_eq!(budget.request.tripped(), tripped, "nothing was refused");

            // Starting above a seq_no; an error from `apply` (a 429 too) is
            // returned at once, not retried.
            let replay = log.replay_stream(3550, Default::default()).unwrap();
            let mut tail = Vec::new();
            replay.for_each_chunk(REPLAY_CHUNK_OPS, |chunk| chunk.try_for_each(|op| { tail.push(op.seq_no()); Ok(()) })).unwrap();
            assert_eq!(tail, (3550..3600).collect::<Vec<_>>());
            let error = replay.for_each_chunk(REPLAY_CHUNK_OPS, |_| Err(EsError::internal("boom"))).err().unwrap();
            assert_eq!(error.status, 500);
            let mut calls = 0;
            let error = replay.for_each_chunk(REPLAY_CHUNK_OPS, |_| { calls += 1; Err(EsError::new(429, "circuit_breaking_exception", "allocation refused")) }).err().unwrap();
            assert_eq!((error.status, calls), (429, 1));
            assert!(!crate::memory::is_replaying(), "the replay scope ends with the replay");
        });
        assert_eq!(budget.request.reserved(), 0, "nothing left reserved");
        // Appends after the stream was created are not part of it; a trim
        // while it runs does not lose what it still has to read.
        let replay = log.replay_stream(0, Default::default()).unwrap();
        log.append(&big(9999)).unwrap();
        log.roll().unwrap();
        log.trim_to(log.generation()).unwrap();
        let mut count = 0;
        replay.for_each_chunk(REPLAY_CHUNK_OPS, |chunk| { count += chunk.len(); Ok(()) }).unwrap();
        assert_eq!(count, ops.len());
        assert!(TranslogReplay::empty().collect().unwrap().is_empty());
    }

    /// Regression: replay never waits on the breaker. With other work
    /// holding the whole request breaker (which, before, made the replay
    /// retry forever), and a single op larger than the limit, the replay
    /// completes at once.
    #[test]
    fn replay_is_not_delayed_by_a_held_competing_reservation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        let huge = TranslogOp::Index { id: "huge".into(), source_json: format!("{{\"p\":\"{}\"}}", "x".repeat(256 * 1024)), seq_no: 0, version: 1, routing: None };
        log.append(&huge).unwrap();
        log.append(&index("small", 1)).unwrap();
        log.fsync().unwrap();
        let budget = crate::memory::Breakers::leak_for_tests(1 << 30);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        budget.request.set_limit(64 * 1024);
        let competing = budget.reserve_request(64 * 1024, "competing request").unwrap();
        let replay = log.replay_stream(0, Default::default()).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let scope = crate::memory::RequestScope::new("held reservation replay test", budget);
            let result = crate::memory::scoped(Some(scope), || {
                let mut seen = Vec::new();
                replay.for_each_chunk(REPLAY_CHUNK_OPS, |chunk| chunk.try_for_each(|op| { seen.push(op.seq_no()); Ok(()) })).map(|()| seen)
            });
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        let seen = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("the replay waited on the breaker").unwrap();
        assert_eq!(seen, vec![0, 1]);
        assert_eq!(budget.request.reserved(), 64 * 1024, "only the competing reservation is left");
        drop(competing);
    }

    #[test]
    fn crc32_matches_reference_table() {
        for data in [&b""[..], b"a", b"123456789", &[0xffu8; 1000][..]] {
            assert_eq!(crc32(data), crc32_reference(data));
        }
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
    use super::*;

    fn index(id: &str, seq: u64) -> TranslogOp {
        TranslogOp::Index {
            id: id.into(),
            source_json: format!("{{\"n\":{seq}}}"),
            seq_no: seq,
            version: 1,
            routing: None,
        }
    }

    #[test]
    fn crc_matches_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn append_and_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        log.append(&index("a", 1)).unwrap();
        log.append(&TranslogOp::Delete { id: "a".into(), seq_no: 2, version: 2 }).unwrap();
        log.append(&index("b", 3)).unwrap();
        log.fsync().unwrap();
        drop(log);

        let log = Translog::open(tmp.path(), 1).unwrap();
        let ops = log.replay(0).unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[1], TranslogOp::Delete { id: "a".into(), seq_no: 2, version: 2 });
        assert_eq!(log.replay(3).unwrap(), vec![index("b", 3)]);
        assert!(log.replay(4).unwrap().is_empty());
    }

    #[test]
    fn scan_from_resumes_at_record_boundaries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        for i in 0..5 {
            log.append(&index("a", i)).unwrap();
        }
        let path = tmp.path().join(file_name(1));
        let mut ends = Vec::new();
        scan_from(&path, 0, |op, end| {
            ends.push((op.seq_no(), end));
            op.seq_no() < 2
        })
        .unwrap();
        assert_eq!(ends.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        let mut rest = Vec::new();
        scan_from(&path, ends[2].1, |op, _| {
            rest.push(op.seq_no());
            true
        })
        .unwrap();
        assert_eq!(rest, vec![3, 4]);
        // A torn tail is not decoded.
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(&[9, 0, 0, 0, 1]).unwrap();
        let mut n = 0;
        scan_from(&path, 0, |_, _| {
            n += 1;
            true
        })
        .unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn roll_and_trim() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        log.append(&index("a", 1)).unwrap();
        assert_eq!(log.roll().unwrap(), 2);
        log.append(&index("b", 2)).unwrap();
        assert_eq!(log.replay(0).unwrap().len(), 2);
        log.trim_to(2).unwrap();
        assert!(!tmp.path().join(file_name(1)).exists());
        assert_eq!(log.replay(0).unwrap(), vec![index("b", 2)]);
        // Reopen continues on generation 2.
        drop(log);
        let mut log = Translog::open(tmp.path(), 2).unwrap();
        log.append(&index("c", 3)).unwrap();
        assert_eq!(log.replay(0).unwrap().len(), 2);
    }

    #[test]
    fn torn_tail_is_ignored_and_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(file_name(1));
        {
            let mut log = Translog::open(tmp.path(), 1).unwrap();
            log.append(&index("a", 1)).unwrap();
            log.append(&index("b", 2)).unwrap();
        }
        let good_len = std::fs::metadata(&path).unwrap().len();
        // Append a partial record: header claiming 100 bytes, only 3 present.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(b"xyz").unwrap();
        }
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        assert_eq!(log.replay(0).unwrap().len(), 2);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len);
        // Appends after the truncation are readable.
        log.append(&index("c", 3)).unwrap();
        assert_eq!(log.replay(0).unwrap().len(), 3);
    }

    /// B1: a crash in a flush's window (the state still names generation
    /// 1, generation 2 holds acked ops and ends torn mid-frame or
    /// zero-filled): no later append lands after that tail, and every acked
    /// op survives two more restarts.
    #[test]
    fn reopen_never_appends_after_a_torn_later_generation() {
        for zero_filled in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let mut log = Translog::open(tmp.path(), 1).unwrap();
            log.append(&index("a", 1)).unwrap();
            log.fsync().unwrap();
            assert_eq!(log.roll().unwrap(), 2);
            log.append(&index("b", 2)).unwrap();
            log.append(&index("c", 3)).unwrap();
            log.fsync().unwrap();
            drop(log);
            let second = tmp.path().join(file_name(2));
            let mut expected = vec![index("a", 1), index("b", 2)];
            if zero_filled {
                expected.push(index("c", 3));
                OpenOptions::new().append(true).open(&second).unwrap().write_all(&[0u8; 4096]).unwrap();
            } else {
                // `c`'s frame cut in the middle.
                let len = std::fs::metadata(&second).unwrap().len();
                OpenOptions::new().write(true).open(&second).unwrap().set_len(len - 5).unwrap();
            }
            // First restart, from the state's generation 1.
            let mut log = Translog::open(tmp.path(), 1).unwrap();
            assert_eq!(log.replay(0).unwrap(), expected, "zero_filled={zero_filled}");
            log.append(&index("d", 4)).unwrap();
            log.roll().unwrap();
            log.append(&index("e", 5)).unwrap();
            log.fsync().unwrap();
            drop(log);
            expected.extend([index("d", 4), index("e", 5)]);
            for restart in 0..2 {
                let log = Translog::open(tmp.path(), 1).unwrap();
                assert_eq!(log.replay(0).unwrap(), expected, "zero_filled={zero_filled} restart={restart}");
            }
        }
    }

    /// R3: only the newest generation can be torn by a crash. A bad frame
    /// in an older one (media corruption) fails the open and keeps the
    /// file as it is, instead of dropping the acknowledged ops after it; a
    /// torn newest generation is still cut back to its readable prefix.
    #[test]
    fn a_bad_frame_in_an_older_generation_fails_the_open() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        log.append(&index("a", 1)).unwrap();
        log.append(&index("b", 2)).unwrap();
        log.roll().unwrap();
        log.append(&index("c", 3)).unwrap();
        log.append(&index("d", 4)).unwrap();
        log.fsync().unwrap();
        drop(log);
        // Flip a byte in the first record's body of generation 1.
        let first = tmp.path().join(file_name(1));
        let mut data = std::fs::read(&first).unwrap();
        data[HEADER_LEN + 2] ^= 0xFF;
        std::fs::write(&first, &data).unwrap();
        let error = Translog::open(tmp.path(), 1).err().expect("a corrupted older generation was truncated");
        assert!(error.to_string().contains("corrupted"), "{error}");
        assert_eq!(std::fs::read(&first).unwrap(), data, "the corrupted generation was changed");
        // Repaired, and the newest generation torn instead: cut back.
        data[HEADER_LEN + 2] ^= 0xFF;
        std::fs::write(&first, &data).unwrap();
        let second = tmp.path().join(file_name(2));
        let len = std::fs::metadata(&second).unwrap().len();
        OpenOptions::new().write(true).open(&second).unwrap().set_len(len - 5).unwrap();
        let log = Translog::open(tmp.path(), 1).unwrap();
        assert_eq!(log.replay(0).unwrap(), vec![index("a", 1), index("b", 2), index("c", 3)]);
    }

    /// B2: a write that fails partway (ENOSPC / EIO mid-frame) closes the
    /// translog: no later append or sync succeeds (nothing is acked after
    /// the partial frame), the frame is cut back, and the acked ops survive
    /// two restarts.
    #[test]
    fn a_partial_write_closes_the_translog() {
        let tmp = tempfile::tempdir().unwrap();
        let mut log = Translog::open(tmp.path(), 1).unwrap();
        log.append(&index("a", 1)).unwrap();
        log.fsync().unwrap();
        log.tear_next_append();
        assert!(log.append_many(&[index("b", 2), index("c", 3)]).is_err());
        assert!(log.append(&index("d", 4)).is_err(), "an append after a partial write");
        assert!(log.append_many(&[index("e", 5)]).is_err());
        assert!(log.fsync().is_err());
        assert!(log.roll().is_err());
        assert_eq!(std::fs::metadata(tmp.path().join(file_name(1))).unwrap().len(),
            std::fs::read(tmp.path().join(file_name(1))).map(|b| read_file_len(&b)).unwrap(), "the partial frame was cut back");
        drop(log);
        for _ in 0..2 {
            let log = Translog::open(tmp.path(), 1).unwrap();
            assert_eq!(log.replay(0).unwrap(), vec![index("a", 1)]);
        }
    }

    /// Length of the readable prefix of a generation's bytes.
    fn read_file_len(bytes: &[u8]) -> u64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.tlog");
        std::fs::write(&path, bytes).unwrap();
        read_file(&path).unwrap().1 as u64
    }

    #[test]
    fn corrupt_record_stops_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(file_name(1));
        {
            let mut log = Translog::open(tmp.path(), 1).unwrap();
            log.append(&index("a", 1)).unwrap();
            log.append(&index("b", 2)).unwrap();
            log.append(&index("c", 3)).unwrap();
        }
        // Flip a byte inside the second record's body.
        let mut data = std::fs::read(&path).unwrap();
        let first_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let second_body = HEADER_LEN + first_len + HEADER_LEN + 2;
        data[second_body] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();
        let log = Translog::open(tmp.path(), 1).unwrap();
        assert_eq!(log.replay(0).unwrap(), vec![index("a", 1)]);
    }
}

#[cfg(test)]
mod size_cache_tests {
    use super::*;

    fn listed(dir: &Path) -> u64 {
        list_generations(dir).unwrap().iter().map(|(_, p)| std::fs::metadata(p).unwrap().len()).sum()
    }

    /// The cached size of the older generations always equals a fresh
    /// listing: across appends, rolls, trims and files removed from outside.
    #[test]
    fn size_bytes_matches_the_files_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let mut t = Translog::open(tmp.path(), 1).unwrap();
        let op = |seq: u64| TranslogOp::Index { id: format!("d{seq}"), source_json: format!("{{\"n\":{seq}}}"), seq_no: seq, version: 1, routing: None };
        let mut seq = 0;
        for round in 0..6 {
            for _ in 0..20 { t.append(&op(seq)).unwrap(); seq += 1; }
            assert_eq!(t.size_bytes(), listed(tmp.path()), "round {round}");
            assert_eq!(t.size_bytes(), listed(tmp.path()), "cached, round {round}");
            t.roll().unwrap();
            assert_eq!(t.size_bytes(), listed(tmp.path()), "after roll {round}");
        }
        t.trim_to(t.generation() - 2).unwrap();
        assert_eq!(t.size_bytes(), listed(tmp.path()), "after trim");
        let oldest = list_generations(tmp.path()).unwrap()[0].1.clone();
        std::fs::remove_file(oldest).unwrap();
        assert_eq!(t.size_bytes(), listed(tmp.path()), "after an outside removal");
    }
}
