//! Mutable tensor ownership using a current atomic reference directory and the
//! graph's generation domain. Text storage locations are deliberately absent:
//! a text flush/merge must not change graph identity or vector availability.
use super::*;
use super::array_store::{ArrayStore, ArrayStoreConfig};
use std::sync::atomic::{AtomicPtr, AtomicUsize};
mod quantized;
pub(crate) use quantized::SharedQuantized;
use quantized::{QuantizedOwner, EncodedRow};
mod code_log;
use code_log::{CodeLog, CodeLogs, ChunkRef};
mod code_plane;
use code_plane::CodePlane;
mod row_plane;
use row_plane::{RowPlane, PLANE_REF};
pub use row_plane::RowPlaneRef;
pub use code_plane::CodePlaneRef;

/// A source directory slot with this bit set addresses a code-only row in
/// one of the owner's [`CodeLogs`] (its index in the low 30 bits, the log
/// slot in [`LOG_SLOT`]) instead of an [`ArrayStore`] row. ArrayStore
/// references stay below it: the source store is configured with at most
/// 2^9 buffers (offset bits 22).
const LOG_TAG: u32 = 1 << 31;
/// Which of the two [`CodeLogs`] slots a tagged row lives in.
const LOG_SLOT: u32 = 1 << 30;

#[inline]
fn log_index(raw: u32) -> Option<u64> {
    (raw & LOG_TAG != 0).then(|| u64::from(raw & (LOG_SLOT - 1)))
}
#[inline]
fn log_slot(raw: u32) -> usize { usize::from(raw & LOG_SLOT != 0) }
/// The directory tag of row `index` of the log in `slot`.
fn log_tag(slot: usize, index: u64) -> Result<u32> {
    let index = u32::try_from(index).ok().filter(|i| *i < LOG_SLOT).ok_or_else(|| failure("code log row index exhausted"))?;
    Ok(LOG_TAG | if slot == 1 { LOG_SLOT } else { 0 } | index)
}
/// Below this many rows a code log is never compacted (a tiny log's dead
/// rows cost less than copying it).
const CODE_LOG_COMPACT_MIN_ROWS: u64 = 1 << 16;
/// Live rows one compaction call copies (in batches of
/// [`CODE_LOG_COMPACT_BATCH`] slots, the graph writer released between
/// them and full target chunks sealed to files, so neither the writer hold
/// nor the anonymous memory grows with the log).
const CODE_LOG_COMPACT_ROWS_PER_CALL: usize = 1 << 20;
const CODE_LOG_COMPACT_BATCH: usize = 1 << 14;

/// Snapshot identity workspace is admitted before copying and drops before
/// the returned owner escapes; graph/source resident storage has its own guards.
fn restore_identities(graph:&MutableHnsw)->Result<crate::admission::Allocation<Vec<(u64,NodeHandle)>>> {
    let count=usize::try_from(graph.active_nodes()).map_err(|_|failure("checkpoint identity count"))?;
    let bytes=count.checked_mul(std::mem::size_of::<(u64,NodeHandle)>()).ok_or_else(||failure("checkpoint identity size"))?;
    let memory=crate::admission::shared().reserve(bytes as u64,"checkpoint restore identities")?;
    let mut identities=Vec::new();
    identities.try_reserve_exact(count).map_err(|_|failure("checkpoint identities allocation"))?;
    let snapshot=graph.try_snapshot()?;
    for identity in snapshot.visible_identities() {
        if identities.len()==count{return Err(failure("checkpoint identity count changed"));}
        identities.push(identity);
    }
    Ok(crate::admission::Allocation::new(identities,memory))
}
fn restore_words(count:usize,label:&str)->Result<crate::admission::Reservation> {
    let bytes=count.checked_mul(4).ok_or_else(||failure("checkpoint row workspace size"))?;
    crate::admission::shared().reserve(bytes as u64,label)
}

/// A released plane row can still be borrowed by a reader pinned before
/// its release (it looked the row up by handle through the directory), and
/// slot reuse does not wait for such readers. A reused slot therefore gets
/// its row in the plane only once every guard older than the release has
/// drained; until then the row lives in the row store and is moved into the
/// plane by a later commit.
#[derive(Default)]
struct PlaneQuarantine {
    /// Slot -> domain generation current at its plane row's release.
    released: std::collections::HashMap<u32, u64>,
    /// Rows installed in the row store over a released slot.
    installed: Vec<(NodeHandle, u64)>,
}

struct SourceDirectory {
    slots:Box<[AtomicU32]>,
    _memory:crate::admission::Reservation,
}
impl std::ops::Deref for SourceDirectory {type Target=[AtomicU32];fn deref(&self)->&Self::Target{&self.slots}}
impl std::ops::DerefMut for SourceDirectory {fn deref_mut(&mut self)->&mut Self::Target{&mut self.slots}}
impl SourceDirectory {
    fn new(capacity:usize,previous:Option<&Self>)->Result<Self>{
        let bytes=capacity.checked_mul(std::mem::size_of::<AtomicU32>()).ok_or_else(||failure("source directory size"))?;
        let memory=crate::admission::shared().reserve(bytes as u64,"mutable source directory")?;
        let mut slots=Vec::new();slots.try_reserve_exact(capacity).map_err(|_|failure("source directory allocation"))?;
        slots.extend((0..capacity).map(|id|AtomicU32::new(previous.and_then(|p|p.get(id)).map_or(0,|v|v.load(Relaxed)))));
        Ok(Self{slots:slots.into_boxed_slice(),_memory:memory})
    }
}
struct Directory {
    current: SourceDirectory,
    retired: Vec<(u64, SourceDirectory)>,
    retired_memory:Option<crate::admission::Reservation>,
}
impl Directory {
    fn reserve_retirement(&mut self)->Result<()> {
        if self.retired.len()<self.retired.capacity(){return Ok(())}
        let capacity=self.retired.len().checked_add(1).and_then(|v|v.checked_mul(2)).ok_or_else(||failure("source retirement size"))?;
        let bytes=capacity.checked_mul(std::mem::size_of::<(u64,SourceDirectory)>()).ok_or_else(||failure("source retirement bytes"))?;
        let memory=crate::admission::shared().reserve(bytes as u64,"source retired directory entries")?;
        // Separate allocation keeps the old capacity charged through the move.
        let mut replacement=Vec::new();replacement.try_reserve_exact(capacity).map_err(|_|failure("source retirement allocation"))?;
        replacement.append(&mut self.retired);
        self.retired=replacement;
        self.retired_memory=Some(memory);Ok(())
    }
}

/// Durable f32 rows by source key (the vector store's mapped `.rvec`
/// segments). A quantized owner keeps only codes and corrections in memory
/// for rows whose f32 is durable here (Vespa: one in-memory representation
/// per field); graph construction, exact scoring and checkpoints read the
/// f32 through it.
pub trait RawRows: Send + Sync {
    /// A read view; `None` when the rows are gone (store dropped).
    fn snapshot(&self) -> Result<Option<Box<dyn RawRowsSnapshot>>>;
}

/// A [`RawRows`] read view: rows it returns stay valid while it lives.
pub trait RawRowsSnapshot: Send + Sync {
    fn row(&self, key: u64) -> Option<&[f32]>;
    /// A failed pin must not be reported as an absent durable row.
    fn execution_result(&self) -> Result<()> { Ok(()) }
}

/// Rows [`MutableVectorOwner::shed_some`] examines at least per call.
const SHED_SCAN_MIN: usize = 256;

/// A lazily taken [`RawRowsSnapshot`] for one source read.
#[derive(Default)]
struct LazyRaw(std::sync::OnceLock<Result<Option<Box<dyn RawRowsSnapshot>>>>);

impl LazyRaw {
    fn row(&self, owner: &MutableVectorOwner, key: u64) -> Option<&[f32]> {
        self.0.get_or_init(|| match owner.raw.get() {Some(raw)=>raw.snapshot(),None=>Ok(None)})
            .as_ref().ok()?.as_ref()?.row(key)
    }
    fn execution_result(&self)->Result<()> {
        match self.0.get() {
            Some(Err(RusticError::Memory(error)))=>Err(RusticError::Memory(error.clone())),
            Some(Err(error))=>Err(failure(&error.to_string())),
            Some(Ok(Some(raw)))=>raw.execution_result(),
            _=>Ok(()),
        }
    }
    fn row_result(&self,owner:&MutableVectorOwner,key:u64)->Result<Option<&[f32]>> {
        let row=self.row(owner,key);self.execution_result()?;Ok(row)
    }
}

/// Graph-writer serialized queue. Requeueing popped rows reuses its capacity.
#[derive(Default)]
struct UnshedQueue {
    rows: std::collections::VecDeque<(NodeHandle,u64)>,
    memory: Option<crate::admission::Reservation>,
}
impl std::ops::Deref for UnshedQueue {type Target=std::collections::VecDeque<(NodeHandle,u64)>;fn deref(&self)->&Self::Target{&self.rows}}
impl std::ops::DerefMut for UnshedQueue {fn deref_mut(&mut self)->&mut Self::Target{&mut self.rows}}
impl UnshedQueue {
    fn admit_push(&mut self)->Result<()> {
        if self.rows.len()<self.rows.capacity(){return Ok(())}
        let capacity=self.rows.capacity().checked_mul(2).unwrap_or(usize::MAX).max(8);
        let bytes=capacity.checked_mul(std::mem::size_of::<(NodeHandle,u64)>()).ok_or_else(||failure("source queue capacity overflow"))?;
        let memory=crate::admission::shared().reserve(bytes as u64,"source shedding queue")?;
        let mut next=std::collections::VecDeque::new();
        next.try_reserve_exact(capacity).map_err(|_|failure("source shedding queue allocation"))?;
        next.extend(self.rows.drain(..));
        self.rows=next;self.memory=Some(memory);Ok(())
    }
}

/// One owner per graph/field/shard. Installation precedes graph preparation;
/// source removal follows the graph's reclamation acknowledgment. This is a
/// current source view, not a historical PIT capture or a durability manifest.
pub struct MutableVectorOwner {
    graph: Arc<MutableHnsw>,
    quantized: Option<QuantizedOwner>,
    /// Code-only rows of a quantized field (file-backed once their chunk is
    /// full; see [`CodeLog`]), compacted into a fresh log when mostly dead
    /// ([`CodeLogs`]). `None` for f32 fields.
    log: Option<CodeLogs>,
    /// [`CODE_LOG_COMPACT_MIN_ROWS`], lowered by tests.
    compact_min_rows: std::sync::atomic::AtomicU64,
    /// Where durable f32 rows are read for code-only rows ([`RawRows`]).
    raw: std::sync::OnceLock<Arc<dyn RawRows>>,
    /// Quantized rows installed with their f32, oldest first: once the f32
    /// is durable ([`RawRows`]) they shed it ([`MutableVectorOwner::shed_some`]).
    unshed: parking_lot::Mutex<UnshedQueue>,
    rows: ArrayStore<f32, 22>,
    directory: parking_lot::Mutex<Directory>,
    pending_reclamation: parking_lot::Mutex<Vec<ReclaimedNode>>,
    pointer: AtomicPtr<AtomicU32>,
    len: AtomicUsize,
    /// A quantized field's codes by node id ([`CodePlane`]): what plain
    /// traversal scores. Written beside every row install that carries codes.
    /// `None` for an f32 field (see [`Self::attach_code_plane`]).
    plane: Option<Box<CodePlane>>,
    /// An unquantized field's rows by node id ([`RowPlane`]): its only
    /// in-memory copy of each row (the directory slot holds [`PLANE_REF`]).
    /// `None` for a quantized field, or when f32 rows shed to the durable
    /// store (`RUSTIC_F32_GRAPH_SHED`).
    row_plane: Option<Box<RowPlane>>,
    /// Plane slots released while a reader may still hold their old row
    /// (borrowed through the directory by handle), and rows installed in the
    /// row store instead because their slot was such a one (see
    /// [`PlaneQuarantine`]).
    quarantine: parking_lot::Mutex<PlaneQuarantine>,
    /// `quarantine.installed.len()`: while nonzero the plane is not every
    /// row's copy, so traversal does not read rows by id (`dense_layout`).
    quarantined: AtomicUsize,
    /// Graph maintenance compares exact rows, not decoded codes
    /// (`get_node_approx`): 1-bit codes decode too coarsely for diversity
    /// selection and pruning (nodes lost all their links).
    exact_mutation_rows: bool,
}

/// Holds the same generation domain as the graph. Every lookup acquires the
/// current directory pointer; pinning does not clone a directory or row page.
pub struct MutableVectorRead<'a> {
    owner: &'a MutableVectorOwner,
    guard: GenerationGuard,
    raw: LazyRaw,
    decoded: DecodedRows,
}

/// Rows decoded from codes for one read ([`MutableVectorSource::get_node_approx`]),
/// each once: append-only, so a returned row stays valid while the read lives.
/// A retained capture (PIT, scroll) can decode a large share of the field
/// over its lifetime, so the rows and their index are charged to the shared
/// memory budget in blocks; a denied charge leaves the row unresolved and
/// fails only the operation that asked (on its thread, through
/// [`DecodedRows::execution_result`]): later operations on the same read
/// retry the charge, so a PIT is never broken for good. A retained capture's
/// cache also counts as retained-capture memory (node accounting and the
/// retained budget, which can release the capture).
#[derive(Default)]
struct DecodedRows(parking_lot::Mutex<DecodedState>);

thread_local! {
    /// A decoded-row charge denied during the current operation on this
    /// thread, reported (and cleared) by the next execution result check.
    static DECODE_DENIED: std::cell::RefCell<Option<RusticError>> = const { std::cell::RefCell::new(None) };
}

#[derive(Default)]
struct DecodedState {
    rows: Vec<Box<[f32]>>,
    index: std::collections::HashMap<u64, usize, std::hash::BuildHasherDefault<HandleHasher>>,
    /// Bytes charged so far (at least the bytes held).
    memory: Option<crate::admission::Reservation>,
    held: u64,
    /// Retained captures: the held bytes on node-wide retained accounting.
    retained: Option<crate::accounting::Charge>,
    /// The calibration epochs this read decoded rows of (one, almost
    /// always): a decode resolves its epoch here, not in the owner's shared
    /// epoch map, whose lock word and `Arc` counts every concurrent insert
    /// would otherwise bounce. Holding them keeps them from collection, as
    /// any other holder does.
    epochs: Vec<(u64, Arc<quantized::Epoch>)>,
    /// Test seam: deny the next charge.
    #[cfg(test)]
    deny_next: bool,
}

/// Smallest budget increment of a decoded-row cache.
const DECODED_CHARGE_BLOCK: u64 = 64 << 10;

/// Node handles (slot and incarnation) are already well distributed.
#[derive(Default)]
struct HandleHasher(u64);
impl std::hash::Hasher for HandleHasher {
    fn finish(&self) -> u64 { self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) }
    fn write(&mut self, bytes: &[u8]) { for &b in bytes { self.0 = (self.0 << 8) | u64::from(b); } }
    fn write_u64(&mut self, n: u64) { self.0 = n; }
}
impl DecodedRows {
    /// The cache of a retained capture (see the type docs).
    fn retained() -> Self {
        Self(parking_lot::Mutex::new(DecodedState {
            retained: Some(crate::accounting::Charge::new(crate::accounting::Category::RetainedCaptures, 0)),
            ..Default::default()
        }))
    }

    /// Bytes the cache holds.
    fn held_bytes(&self) -> u64 { self.0.lock().held }

    #[cfg(test)]
    fn get_or_decode(&self, id: u64, decode: impl FnOnce(&mut Vec<f32>) -> bool) -> Option<&[f32]> {
        self.get_or_decode_with(id, |_, out| decode(out))
    }

    /// Row `id`, decoded once by `decode` (given this read's epoch cache).
    fn get_or_decode_with(&self, id: u64, decode: impl FnOnce(&mut Vec<(u64, Arc<quantized::Epoch>)>, &mut Vec<f32>) -> bool) -> Option<&[f32]> {
        let mut guard = self.0.lock();
        let state = &mut *guard;
        if let Some(&index) = state.index.get(&id) {
            let row: *const [f32] = &*state.rows[index];
            // SAFETY: boxed rows are never removed or moved while `self` lives.
            return Some(unsafe { &*row });
        }
        let mut out = Vec::new();
        if !decode(&mut state.epochs, &mut out) { return None; }
        // The row, its box pointer and index entry, and the vectors'
        // amortised growth (charged as one more entry's worth).
        let entry = (std::mem::size_of::<Box<[f32]>>() + 2 * std::mem::size_of::<(u64, usize)>()) as u64;
        let held = state.held.saturating_add((out.len() * 4) as u64).saturating_add(2 * entry);
        let charged = state.memory.as_ref().map_or(0, crate::admission::Reservation::bytes);
        if held > charged {
            let more = (held - charged).max(DECODED_CHARGE_BLOCK).max(charged / 4);
            #[cfg(test)]
            if std::mem::take(&mut state.deny_next) {
                DECODE_DENIED.with(|denied| *denied.borrow_mut() = Some(RusticError::Memory("test denial".into())));
                return None;
            }
            let result = match state.memory.as_mut() {
                Some(memory) => memory.grow(more, "decoded quantized rows"),
                None => crate::admission::shared().reserve(more, "decoded quantized rows").map(|memory| { state.memory = Some(memory); }),
            };
            if let Err(error) = result {
                DECODE_DENIED.with(|denied| *denied.borrow_mut() = Some(error));
                return None;
            }
        }
        state.held = held;
        if let Some(charge) = state.retained.as_mut() { charge.resize(held); }
        let boxed = out.into_boxed_slice();
        let row: *const [f32] = &*boxed;
        state.rows.push(boxed);
        let index = state.rows.len() - 1;
        state.index.insert(id, index);
        // SAFETY: as above.
        Some(unsafe { &*row })
    }

    /// A charge denied during this thread's current operation, as that
    /// operation's failure (reported once).
    fn execution_result(&self) -> Result<()> {
        match DECODE_DENIED.with(|denied| denied.borrow_mut().take()) {
            Some(RusticError::Memory(error)) => Err(RusticError::Memory(error)),
            Some(error) => Err(failure(&error.to_string())),
            None => Ok(()),
        }
    }

    #[cfg(test)]
    fn charged_bytes(&self) -> u64 {
        self.0.lock().memory.as_ref().map_or(0, crate::admission::Reservation::bytes)
    }
}

/// [`MutableVectorSource::code_scorer`]: a symmetric estimate of two
/// stored 7-bit OSQ code rows of `epoch` (one row's codes as the query's,
/// which the estimate supports when doc and query codes share a width),
/// as a distance (the negated score: lower is closer). `None` for any
/// other quantizer.
fn code_pair_scorer(quantized: &QuantizedOwner, epoch: u64, dim: usize) -> Option<Box<dyn Fn(&[f32], &[f32]) -> f64 + 'static>> {
    let epoch = quantized.get(epoch)?;
    let crate::vector::es_quant::EsQuantizer::Osq(osq) = &epoch.q else { return None };
    if osq.doc_bits != 7 || osq.query_bits != osq.doc_bits || osq.centroid.len() != dim { return None; }
    let osq = osq.clone();
    /// A code row's corrections and packed codes (`dim` bytes).
    #[cfg(target_endian = "little")]
    fn split(w: &[f32], dim: usize) -> Option<(crate::vector::osq::OsqCorrections, &[u8])> {
        if w.len() != 4 + dim.div_ceil(4) { return None; }
        let corrections = crate::vector::osq::OsqCorrections { lower: w[0], upper: w[1], additional: w[2], component_sum: w[3].to_bits() as i32 };
        // SAFETY: initialized word payloads, read as their bytes (the packed
        // codes, as `EsQuantScorer::score_words` reads them).
        let codes = unsafe { std::slice::from_raw_parts(w[4..].as_ptr().cast::<u8>(), dim) };
        Some((corrections, codes))
    }
    #[cfg(target_endian = "little")]
    return Some(Box::new(move |a: &[f32], b: &[f32]| match (split(a, dim), split(b, dim)) {
        (Some((ca, xa)), Some((cb, xb))) => -f64::from(osq.score(xa, &ca, xb, &cb)),
        _ => f64::NAN,
    }));
    #[cfg(not(target_endian = "little"))]
    { let _ = osq; None }
}

/// [`MutableVectorSource::get_node_approx`] of a quantized owner's row.
/// The epoch is resolved only for a row not decoded yet, and then from the
/// read's own epoch cache.
fn approx_row<'s>(quantized: &QuantizedOwner, decoded: &'s DecodedRows, row: &[f32], dim: usize, handle: NodeHandle) -> Option<&'s [f32]> {
    let id = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
    let codes = row.get(codes_start(row, dim)..)?;
    decoded.get_or_decode_with(handle.as_u64(), |epochs, out| {
        let at = match epochs.iter().position(|(epoch, _)| *epoch == id) {
            Some(at) => at,
            None => {
                let Some(epoch) = quantized.get(id) else { return false };
                epochs.push((id, epoch));
                epochs.len() - 1
            }
        };
        epochs[at].1.q.decode_words(codes, dim, out)
    })
}

/// Owned, saver-style source capture. Reference metadata is copied once;
/// vector payloads remain shared under a generation guard. This does not by
/// itself pin ES document masks or fetch ownership.
pub struct CapturedMutableVectorRead {
    owner: Arc<MutableVectorOwner>,
    guard: GenerationGuard,
    references: crate::admission::Allocation<Vec<LevelsRef>>,
    raw: LazyRaw,
    decoded: DecodedRows,
    /// The score quantizer as of the capture: a retained view (PIT,
    /// scroll, `search_after`) estimates its scores with it, not with the
    /// live field's later centroid.
    score: Option<crate::vector::es_quant::EsQuantizer>,
}

/// Saver-style capture. Graph/source references and allocator metadata are copied;
/// graph serialization and vector streaming happen after the writer fence releases. Publish graph and row files
/// together with ES document ownership and the WAL sequence in one manifest.
pub struct MutableVectorCheckpoint {
    graph: std::sync::OnceLock<crate::admission::Allocation<Vec<u8>>>,
    graph_owner: Arc<MutableHnsw>,
    graph_view: MutableHnswSnapshot,
    incarnations: crate::admission::Allocation<Vec<u32>>,
    watermark: Option<u64>,
    retirements: crate::admission::Allocation<std::collections::HashMap<NodeHandle, u64>>,
    source: CapturedMutableVectorRead,
    identities: crate::admission::Allocation<Vec<(u64, NodeHandle)>>,
    dim: u32,
    quantization: Option<crate::admission::Allocation<Vec<u8>>>,
    mutation_generation: u64,
    /// Code log rows appended when the source references were captured:
    /// every tagged reference of `source` naming the log in `log_slot` is
    /// below it. During a code log compaction, references naming the other
    /// slot are written with their codes inline instead.
    log_rows: Option<u64>,
    log_slot: usize,
}

/// A mutable checkpoint at its capture point, not copied yet
/// ([`MutableVectorOwner::begin_checkpoint`]).
pub struct PendingMutableCheckpoint {
    owner: Arc<MutableVectorOwner>,
    pending: PendingCapture,
    watermark: Option<u64>,
    retirements: crate::admission::Allocation<std::collections::HashMap<NodeHandle, u64>>,
    nodes: u32,
    quantization: Option<crate::admission::Allocation<Vec<u8>>>,
    mutation_generation: u64,
}

impl PendingMutableCheckpoint {
    /// The graph's [`MutableHnsw::mutation_generation`] at the capture point.
    pub fn mutation_generation(&self) -> u64 { self.mutation_generation }

    /// Copy the graph as of the capture point and capture the source rows
    /// its nodes reference.
    pub fn finish(self) -> Result<MutableVectorCheckpoint> {
        let Self { owner, pending, watermark, retirements, nodes, quantization, mutation_generation } = self;
        let view = owner.graph.finish_capture(pending)?;
        let count=view.visible_identities().count();
        let bytes=count.checked_mul(std::mem::size_of::<(u64,NodeHandle)>()).ok_or_else(||failure("checkpoint identity bytes"))?;
        let identities_memory=crate::admission::shared().reserve(bytes as u64,"checkpoint captured identities")?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(count).map_err(|_| failure("checkpoint identity allocation"))?;
        identities.extend(view.visible_identities());
        let identities=crate::admission::Allocation::new(identities,identities_memory);
        let (source, log_rows, log_slot) = {
            let _writer = owner.graph.control.writer.lock();
            // The log's single appender holds this writer: the row count
            // matches the tags these references carry.
            let log_slot = owner.log.as_ref().map_or(0, CodeLogs::active_slot);
            (owner.capture_references(&view)?, owner.log.as_ref().map(|logs| logs.active().rows()), log_slot)
        };
        let incarnations_memory=restore_words(nodes as usize,"checkpoint captured incarnations")?;
        let mut incarnations = Vec::new();
        incarnations.try_reserve_exact(nodes as usize)
            .map_err(|_| failure("checkpoint incarnation allocation"))?;
        let table = owner.graph.control.nodes.read(&view.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        for id in 1..=nodes {
            incarnations.push(table.get(id).ok_or_else(|| failure("checkpoint node absent"))?.metadata().incarnation);
        }
        drop(table);
        Ok(MutableVectorCheckpoint { graph: std::sync::OnceLock::new(), graph_owner: owner.graph.clone(),
            graph_view: view, incarnations:crate::admission::Allocation::new(incarnations,incarnations_memory), watermark,
            retirements, source, identities, dim: owner.graph.dim,
            quantization, mutation_generation, log_rows, log_slot })
    }
}

/// Piece size of a streamed graph checkpoint ([`MutableVectorCheckpoint::write_graph`]).
/// Small under test, so the unit tests' graphs stream in several pieces.
const GRAPH_STREAM_CHUNK: usize = if cfg!(test) { 96 } else { 4 << 20 };

impl MutableVectorCheckpoint {
    /// The graph's [`MutableHnsw::mutation_generation`] this checkpoint is.
    pub fn mutation_generation(&self) -> u64 { self.mutation_generation }

    /// The graph section of the checkpoint file: its length, then the
    /// graph, streamed from the captured view in pieces of
    /// [`GRAPH_STREAM_CHUNK`] (the bytes [`Self::graph_bytes`] would hold,
    /// never all in memory: at tens of millions of nodes they are many GB).
    fn write_graph(&self, output: &mut dyn crate::codec::directory::OutputFile) -> Result<()> {
        if let Some(graph) = self.graph.get() {
            output.write_all(&(graph.len() as u64).to_le_bytes())?;
            return output.write_all(graph);
        }
        let allocated = self.incarnations.len() as u32;
        let length = self.graph_owner.retained_len(&self.graph_view, allocated)?;
        output.write_all(&(length as u64).to_le_bytes())?;
        let written = self.graph_owner.stream_retained(&self.graph_view, allocated, self.watermark, &self.retirements,
            Some(&self.incarnations), GRAPH_STREAM_CHUNK, &mut |bytes: &[u8]| output.write_all(bytes))?;
        if written != length { return Err(failure("graph checkpoint length changed while streaming")); }
        Ok(())
    }

    pub fn graph_bytes(&self) -> Result<&[u8]> {
        if self.graph.get().is_none() {
            let bytes=self.graph_owner.encode_retained_admitted(&self.graph_view,
                self.incarnations.len() as u32,self.watermark,&self.retirements,Some(&self.incarnations))?;
            // A racing initializer keeps its own guard until the losing bytes drop.
            let _=self.graph.set(bytes);
        }
        Ok(self.graph.get().expect("initialized graph checkpoint").as_slice())
    }

    /// Atomically publish one graph/source pair through the index storage API.
    /// The caller must reference this immutable file from the matching document
    /// commit/WAL boundary. Finishing this file alone does not commit documents.
    ///
    /// Format 2 (`RMVC\x02`, rows `RMVR\x04`): rows in the code log are
    /// referenced by index, never copied; the log files covering them are
    /// made durable first (only chunk bytes not yet in a file are written)
    /// and returned, for the commit manifest to reference. f32 durable in
    /// the vector store is referenced too. What the file itself holds is
    /// the graph and per-node identities: O(graph), not O(vectors).
    ///
    /// Format 2 is written only under write version 13 (a cluster whose
    /// format floor is 13: every member reads it, see
    /// [`crate::codec::write_version`]), read once per file: every
    /// rustic build reports the same ES version, so a replica may be
    /// allocated from a new primary to an old node mid-upgrade. Below it the
    /// file is format 1 with the rows older builds read (RMVR v1/v2, or v3
    /// under v12), and no code files are referenced.
    pub fn write_file(&self, directory: &dyn crate::codec::directory::Directory, name: &str) -> Result<Vec<String>> {
        self.write_file_versioned(directory, name, crate::codec::write_version())
    }

    /// [`Self::write_file`] as format `version`, read once by the caller: the
    /// file's header and rows follow it even if the cluster's format floor
    /// rises while the file is written.
    pub fn write_file_versioned(&self, directory: &dyn crate::codec::directory::Directory, name: &str, version: u32) -> Result<Vec<String>> {
        if version < crate::codec::RMVC2_SINCE {
            return self.write_file_compatible(directory, name, version).map(|()| Vec::new());
        }
        // The capture's guard keeps this log alive through a later switch.
        let (chunks, code_files) = match (self.source.owner.log.as_ref().and_then(|logs| logs.get(self.log_slot)), self.log_rows) {
            (Some(log), Some(rows)) if rows > 0 => {
                let chunks = log.persist(directory, rows)?;
                let files = log.file_names(&chunks);
                (chunks, files)
            }
            _ => (Vec::new(), Vec::new()),
        };
        let mut output = directory.create_output(name)?;
        output.cold_checkpoint();
        output.write_all(b"RMVC\x02\0\0\0")?;
        self.write_graph(output.as_mut())?;
        // Adapt without buffering the vector payload. Directory::finish remains
        // the only publication point and is never called after a stream error.
        struct Sink<'a>(&'a mut dyn crate::codec::directory::OutputFile);
        impl std::io::Write for Sink<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.write_all(bytes).map_err(std::io::Error::other)?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        self.write_rows_logged(&mut Sink(output.as_mut()), &chunks)?;
        output.finish()?;
        Ok(code_files)
    }

    /// Format 1 with the rows the write version selects ([`Self::write_rows`]:
    /// RMVR v1/v2, or v3 under v12): what builds before v13 read.
    fn write_file_compatible(&self, directory: &dyn crate::codec::directory::Directory, name: &str, version: u32) -> Result<()> {
        let mut output = directory.create_output(name)?;
        output.cold_checkpoint();
        output.write_all(b"RMVC\x01\0\0\0")?;
        self.write_graph(output.as_mut())?;
        struct Sink<'a>(&'a mut dyn crate::codec::directory::OutputFile);
        impl std::io::Write for Sink<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.write_all(bytes).map_err(std::io::Error::other)?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        self.write_rows_versioned(&mut Sink(output.as_mut()), version)?;
        output.finish()
    }

    /// RMVR v4 ([`Self::write_file`]): as v3, plus rows that live in the
    /// code log, written as their log index (kind 2) with the log's chunk
    /// list in the header.
    fn write_rows_logged(&self, output: &mut impl std::io::Write, chunks: &[ChunkRef]) -> Result<()> {
        use crate::codec::{crc32, crc32_update};
        let mut checksum = 0xffff_ffff;
        let mut write = |bytes: &[u8]| -> Result<()> {
            output.write_all(bytes)?;
            checksum = crc32_update(checksum, bytes);
            Ok(())
        };
        let dim = self.dim as usize;
        write(b"RMVR\x04\0\0\0")?;
        write(&self.dim.to_le_bytes())?;
        write(&(self.identities.len() as u64).to_le_bytes())?;
        write(&crc32(self.graph_bytes()?).to_le_bytes())?;
        write(&u32::from(self.quantization.is_some()).to_le_bytes())?;
        if let Some(quantization) = &self.quantization {
            write(&(quantization.len() as u64).to_le_bytes())?;
            write(quantization)?;
        }
        let log = self.source.owner.log.as_ref().and_then(|logs| logs.get(self.log_slot)).filter(|_| !chunks.is_empty());
        let log_rows = self.log_rows.unwrap_or(0);
        write(&u32::from(log.is_some()).to_le_bytes())?;
        if let Some(log) = log {
            write(&log.id().to_le_bytes())?;
            write(&u32::try_from(log.row_words()).map_err(|_| failure("code log row width"))?.to_le_bytes())?;
            write(&u32::try_from(log.rows_per_chunk()).map_err(|_| failure("code log chunk rows"))?.to_le_bytes())?;
            write(&u32::try_from(chunks.len()).map_err(|_| failure("code log chunk count"))?.to_le_bytes())?;
            for chunk in chunks { write(&chunk.rows.to_le_bytes())?; }
        }
        let mut row_memory=None;
        let mut row = Vec::new();
        for &(key, handle) in self.identities.iter() {
            let reference = self.source.references.get(internal_id(handle) as usize).copied().unwrap_or(LevelsRef::INVALID);
            // A row still in the log being compacted away falls through and
            // is written inline (code-only, f32 durable), like an ArrayStore
            // code-only row: the file references one log only.
            if let Some(index) = log_index(reference.raw()).filter(|_| log_slot(reference.raw()) == self.log_slot) {
                // Validates the identity the row carries, as for any row.
                self.source.encoded_node(handle, key).ok_or_else(|| failure("captured checkpoint source missing"))?;
                if log.is_none() || index >= log_rows { return Err(failure("code log row beyond the captured log")); }
                reserve_output_row(&mut row,&mut row_memory,28)?;
                row.clear();
                put64(&mut row, key);
                put64(&mut row, handle.as_u64());
                put32(&mut row, 2);
                put64(&mut row, index);
                write(&row)?;
                continue;
            }
            let encoded = self.source.encoded_node(handle, key).ok_or_else(|| failure("captured checkpoint source missing"))?;
            let durable = self.source.raw.row_result(&self.source.owner, key)?.filter(|raw| raw.len() == dim);
            let embedded: Option<&[f32]> = if is_code_only(encoded) {
                if durable.is_none() { return Err(failure("code-only checkpoint row has no durable f32")); }
                None
            } else {
                let memory = &encoded[8..8 + dim];
                match durable {
                    Some(raw) if raw.iter().zip(memory).all(|(a, b)| a.to_bits() == b.to_bits()) => None,
                    _ => Some(memory),
                }
            };
            let raw_words=embedded.map_or(0,|values|values.len());
            let codes=if self.quantization.is_some(){&encoded[codes_start(encoded,dim)..]}else{&[]};
            let length=raw_words.checked_add(codes.len()).and_then(|v|v.checked_mul(4))
                .and_then(|v|v.checked_add(if self.quantization.is_some(){32}else{20})).ok_or_else(||failure("checkpoint output row bytes"))?;
            reserve_output_row(&mut row,&mut row_memory,length)?;
            row.clear();
            put64(&mut row, key);
            put64(&mut row, handle.as_u64());
            put32(&mut row, u32::from(embedded.is_some()));
            if let Some(values) = embedded {
                for value in values { put32(&mut row, value.to_bits()); }
            }
            if self.quantization.is_some() {
                put32(&mut row, encoded[3].to_bits());
                put32(&mut row, encoded[4].to_bits());
                put32(&mut row, u32::try_from(codes.len()).map_err(|_| failure("encoded checkpoint row length"))?);
                for value in codes { put32(&mut row, value.to_bits()); }
            }
            write(&row)?;
        }
        output.write_all(&(!checksum).to_le_bytes())?;
        Ok(())
    }

    /// Stream the rows with stable graph identities and a pair checksum.
    /// A row's f32 is referenced, not embedded, when it is durable in the
    /// vector store ([`RawRows`]: the `.rvec` the commit references), so raw
    /// vectors are not stored twice on disk; otherwise (a node of a document
    /// not in this commit, or no store) it is embedded. Quantized rows carry
    /// their codes and corrections. An I/O failure must leave the caller's
    /// checkpoint unpublished.
    pub fn write_rows(&self, output: &mut impl std::io::Write) -> Result<()> {
        self.write_rows_versioned(output, crate::codec::write_version())
    }

    /// [`Self::write_rows`] as format `version` (the enclosing file's).
    pub fn write_rows_versioned(&self, output: &mut impl std::io::Write, version: u32) -> Result<()> {
        // RMVR v3 (referenced f32, code-only rows) only when the file is
        // written as the next format generation; below it, what the previous
        // release reads (v1 f32 rows, or v2 full rows for a quantized
        // field), so a rollback keeps its graph checkpoints.
        if version >= crate::codec::RMVR3_SINCE {
            self.write_rows_v3(output)
        } else {
            self.write_rows_compatible(output)
        }
    }

    /// RMVR v1 (unquantized: every f32 embedded) or v2 (quantized: the full
    /// row, f32 and codes). A code-only row (its f32 shed from memory once
    /// durable) is written with the f32 read back from the vector store.
    fn write_rows_compatible(&self, output: &mut impl std::io::Write) -> Result<()> {
        use crate::codec::{crc32, crc32_update};
        let mut checksum = 0xffff_ffff;
        let mut write = |bytes: &[u8]| -> Result<()> {
            output.write_all(bytes)?;
            checksum = crc32_update(checksum, bytes);
            Ok(())
        };
        let dim = self.dim as usize;
        write(if self.quantization.is_some() { b"RMVR\x02\0\0\0" } else { b"RMVR\x01\0\0\0" })?;
        write(&self.dim.to_le_bytes())?;
        write(&(self.identities.len() as u64).to_le_bytes())?;
        write(&crc32(self.graph_bytes()?).to_le_bytes())?;
        if let Some(quantization) = &self.quantization {
            write(&(quantization.len() as u64).to_le_bytes())?;
            write(quantization)?;
        }
        let mut row_memory=None;
        let mut row = Vec::new();
        for &(key, handle) in self.identities.iter() {
            let encoded = self.source.encoded_node(handle, key).ok_or_else(|| failure("captured checkpoint source missing"))?;
            let raw=if is_code_only(encoded) {
                Some(self.source.raw.row_result(&self.source.owner,key)?.filter(|raw|raw.len()==dim)
                    .ok_or_else(||failure("code-only checkpoint row has no durable f32"))?)
            } else {None};
            let padded=dim.div_ceil(8).checked_mul(8).and_then(|v|v.checked_add(8)).ok_or_else(||failure("checkpoint row size"))?;
            let words=if raw.is_some(){padded.checked_add(encoded.len()-8).ok_or_else(||failure("checkpoint row size"))?}else{encoded.len()};
            let values=if self.quantization.is_some(){words}else{dim};
            let length=values.checked_mul(4).and_then(|v|v.checked_add(if self.quantization.is_some(){20}else{16})).ok_or_else(||failure("checkpoint output row bytes"))?;
            reserve_output_row(&mut row,&mut row_memory,length)?;
            row.clear();put64(&mut row,key);put64(&mut row,handle.as_u64());
            if self.quantization.is_some() {
                put32(&mut row,u32::try_from(words).map_err(|_|failure("encoded checkpoint row length"))?);
                if let Some(raw)=raw {
                    // Compatible full-row layout, streamed without temporary
                    // rebuilt u32 and f32 copies of the same vector.
                    for word in [key as u32,(key>>32) as u32,handle.generation(),encoded[3].to_bits(),encoded[4].to_bits(),0,0,0] {put32(&mut row,word);}
                    for value in raw {put32(&mut row,value.to_bits());}
                    for _ in dim..padded-8 {put32(&mut row,0);}
                    for value in &encoded[8..] {put32(&mut row,value.to_bits());}
                } else {for value in encoded {put32(&mut row,value.to_bits());}}
            } else {
                for value in raw.unwrap_or_else(||&encoded[8..8+dim]) {put32(&mut row,value.to_bits());}
            }
            debug_assert_eq!(row.len(),length);write(&row)?;
        }
        output.write_all(&(!checksum).to_le_bytes())?;
        Ok(())
    }

    /// RMVR v3: f32 referenced when durable in the vector store, codes for
    /// quantized rows (see [`Self::write_rows`]).
    fn write_rows_v3(&self, output: &mut impl std::io::Write) -> Result<()> {
        use crate::codec::{crc32, crc32_update};
        let mut checksum = 0xffff_ffff;
        let mut write = |bytes: &[u8]| -> Result<()> {
            output.write_all(bytes)?;
            checksum = crc32_update(checksum, bytes);
            Ok(())
        };
        let dim = self.dim as usize;
        write(b"RMVR\x03\0\0\0")?;
        write(&self.dim.to_le_bytes())?;
        write(&(self.identities.len() as u64).to_le_bytes())?;
        write(&crc32(self.graph_bytes()?).to_le_bytes())?;
        write(&u32::from(self.quantization.is_some()).to_le_bytes())?;
        if let Some(quantization) = &self.quantization {
            write(&(quantization.len() as u64).to_le_bytes())?;
            write(quantization)?;
        }
        let mut row_memory=None;
        let mut row = Vec::new();
        for &(key, handle) in self.identities.iter() {
            let encoded = self.source.encoded_node(handle, key).ok_or_else(|| failure("captured checkpoint source missing"))?;
            let durable = self.source.raw.row_result(&self.source.owner, key)?.filter(|raw| raw.len() == dim);
            let embedded: Option<&[f32]> = if is_code_only(encoded) {
                if durable.is_none() { return Err(failure("code-only checkpoint row has no durable f32")); }
                None
            } else {
                let memory = &encoded[8..8 + dim];
                // Referenced only when the durable row is exactly this one.
                match durable {
                    Some(raw) if raw.iter().zip(memory).all(|(a, b)| a.to_bits() == b.to_bits()) => None,
                    _ => Some(memory),
                }
            };
            let raw_words=embedded.map_or(0,|values|values.len());
            let codes=if self.quantization.is_some(){&encoded[codes_start(encoded,dim)..]}else{&[]};
            let length=raw_words.checked_add(codes.len()).and_then(|v|v.checked_mul(4))
                .and_then(|v|v.checked_add(if self.quantization.is_some(){32}else{20})).ok_or_else(||failure("checkpoint output row bytes"))?;
            reserve_output_row(&mut row,&mut row_memory,length)?;
            row.clear();
            put64(&mut row, key);
            put64(&mut row, handle.as_u64());
            put32(&mut row, u32::from(embedded.is_some()));
            if let Some(values) = embedded {
                for value in values { put32(&mut row, value.to_bits()); }
            }
            if self.quantization.is_some() {
                let codes = &encoded[codes_start(encoded, dim)..];
                put32(&mut row, encoded[3].to_bits());
                put32(&mut row, encoded[4].to_bits());
                put32(&mut row, u32::try_from(codes.len()).map_err(|_| failure("encoded checkpoint row length"))?);
                for value in codes { put32(&mut row, value.to_bits()); }
            }
            write(&row)?;
        }
        output.write_all(&(!checksum).to_le_bytes())?;
        Ok(())
    }
}

impl MutableVectorOwner {
    /// Capture graph first so reclamation cannot remove any of its source rows
    /// between the two captures. ES must hold its shard publication fence around
    /// this operation and the corresponding document/fetch capture.
    pub fn capture(self: &Arc<Self>) -> Result<(MutableHnswSnapshot, CapturedMutableVectorRead)> {
        let graph = self.graph.capture_snapshot()?;
        let _writer = self.graph.control.writer.lock();
        let source = self.capture_references(&graph)?;
        Ok((graph, source))
    }

    // The caller holds the graph writer and an earlier graph generation guard.
    fn capture_references(self: &Arc<Self>, graph: &MutableHnswSnapshot) -> Result<CapturedMutableVectorRead> {
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let directory = self.directory.lock();
        let bytes=directory.current.len().checked_mul(std::mem::size_of::<LevelsRef>()).ok_or_else(||failure("captured source bytes"))?;
        let memory=crate::admission::shared().reserve(bytes as u64,"captured source references")?;
        let mut references = Vec::new();
        references.try_reserve_exact(directory.current.len()).map_err(|_| failure("capture source allocation"))?;
        references.extend(directory.current.iter().enumerate().map(|(id, r)| {
            // Retain only identities represented by this captured graph. A
            // previously retired token may already have been handed to source
            // reclamation before our guard; such unrelated slots are excluded.
            if u32::try_from(id).ok().and_then(|id| graph.current_node(id)).is_some() {
                LevelsRef::from_raw(r.load(Acquire))
            } else { LevelsRef::from_raw(0) }
        }));
        self.rows.view(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        Ok(CapturedMutableVectorRead { owner: self.clone(), guard, references:crate::admission::Allocation::new(references,memory), raw: LazyRaw::default(), decoded: DecodedRows::retained(), score: self.score_quantizer() })
    }
    /// Capture a coherent logical graph and its exact owned vectors. Streaming
    /// row output happens after releasing the writer; retained generations keep
    /// the captured payloads alive. The ES caller must additionally fence its
    /// document/sequence publication and atomically publish the complete manifest.
    pub fn capture_checkpoint(self: &Arc<Self>) -> Result<MutableVectorCheckpoint> {
        self.begin_checkpoint()?.finish()
    }

    /// The capture point of [`Self::capture_checkpoint`]: one short
    /// topology-writer hold records the graph generation and the writer
    /// state that must match it; [`PendingMutableCheckpoint::finish`] copies
    /// the graph (chunked, graph writers interleaved) as of that point.
    pub fn begin_checkpoint(self: &Arc<Self>) -> Result<PendingMutableCheckpoint> {
        let (pending, (watermark, retirements, nodes, quantization, mutation_generation)) = self.graph.begin_capture(|writer, _| {
            self.graph.control.check_readable()?;
            // Pending reservations (graph ingestion in flight) are not part
            // of the captured graph: their nodes are invisible to it, their
            // rows are not captured (`capture_references` keeps only captured
            // nodes), and their slots restore free. Reopening inserts their
            // committed points (`reconcile_committed_points`); uncommitted
            // ones replay from the translog. No ingestion pause is needed.
            let count=writer.visibility_retained.len();
            let buckets=count.max(1).checked_next_power_of_two().and_then(|v|v.checked_mul(2)).ok_or_else(||failure("capture retirement capacity"))?;
            let bytes=buckets.checked_mul(std::mem::size_of::<(NodeHandle,u64)>()+1).and_then(|v|v.checked_add(32)).ok_or_else(||failure("capture retirement bytes"))?;
            let memory=crate::admission::shared().reserve(bytes as u64,"checkpoint retirement membership")?;
            let mut retirements=std::collections::HashMap::new();retirements.try_reserve(count).map_err(|_|failure("checkpoint retirement allocation"))?;
            retirements.extend(writer.visibility_retained.iter().map(|(&id,&generation)|(id,generation)));
            let retirements=crate::admission::Allocation::new(retirements,memory);
            Ok((writer.watermark, retirements, self.graph.num_nodes(),
                self.quantized.as_ref().map(QuantizedOwner::checkpoint).transpose()?,
                self.graph.mutation_generation()))
        })?;
        Ok(PendingMutableCheckpoint { owner: self.clone(), pending, watermark, retirements, nodes, quantization, mutation_generation })
    }

    /// Open an atomically published graph/source pair, retaining mapped bytes
    /// only during decoding. Document ownership and WAL replay are the caller's
    /// responsibility; this never reconstructs adjacency from the vector rows.
    pub fn open_file(directory: &dyn crate::codec::directory::Directory, name: &str) -> Result<(Arc<MutableHnsw>, Self)> {
        Self::open_file_with(directory, name, None)
    }

    /// [`Self::open_file`] reading referenced f32 rows from `raw` (the
    /// vector store the checkpoint's commit references).
    pub fn open_file_with(directory: &dyn crate::codec::directory::Directory, name: &str, raw: Option<&dyn RawRowsSnapshot>) -> Result<(Arc<MutableHnsw>, Self)> {
        let file = directory.open_read(name)?;
        let mut input = Decoder { bytes: file.as_slice(), offset: 0 };
        let header = input.take(8)?;
        let logged = header == b"RMVC\x02\0\0\0";
        if !logged && header != b"RMVC\x01\0\0\0" { return Err(corrupt("owned checkpoint file version")); }
        let graph_len = usize::try_from(input.u64()?).map_err(|_| corrupt("owned checkpoint graph length"))?;
        let graph = input.take(graph_len)?;
        let rows = input.take(input.remaining())?;
        if logged { return Self::restore_logged(directory, graph, rows, raw); }
        Self::restore_checkpoint_with(graph, rows, raw)
    }

    /// Only the graph of an owned checkpoint file (its dimensions,
    /// similarity and parameters), for rebuilding a field whose rows cannot
    /// be restored.
    pub fn open_graph_file(directory: &dyn crate::codec::directory::Directory, name: &str) -> Result<Arc<MutableHnsw>> {
        let file = directory.open_read(name)?;
        let mut input = Decoder { bytes: file.as_slice(), offset: 0 };
        let header = input.take(8)?;
        if header != b"RMVC\x01\0\0\0" && header != b"RMVC\x02\0\0\0" { return Err(corrupt("owned checkpoint file version")); }
        let graph_len = usize::try_from(input.u64()?).map_err(|_| corrupt("owned checkpoint graph length"))?;
        Ok(Arc::new(MutableHnsw::open(input.take(graph_len)?)?))
    }

    /// Restore a matching graph/row pair without rebuilding HNSW or reading an
    /// external fixture. Checksums are integrity checks, not authentication.
    pub fn restore_checkpoint(graph_bytes: &[u8], row_bytes: &[u8]) -> Result<(Arc<MutableHnsw>, Self)> {
        Self::restore_checkpoint_with(graph_bytes, row_bytes, None)
    }

    /// [`Self::restore_checkpoint`] reading referenced f32 rows from `raw`.
    pub fn restore_checkpoint_with(graph_bytes: &[u8], row_bytes: &[u8], raw: Option<&dyn RawRowsSnapshot>) -> Result<(Arc<MutableHnsw>, Self)> {
        use crate::codec::crc32;
        if row_bytes.len() < 28 { return Err(corrupt("owned source checkpoint truncated")); }
        let (body, checksum) = row_bytes.split_at(row_bytes.len() - 4);
        if crc32(body) != u32::from_le_bytes(checksum.try_into().unwrap()) {
            return Err(corrupt("owned source checksum"));
        }
        if body.starts_with(b"RMVR\x04\0\0\0") { return Err(corrupt("code log checkpoint rows need their directory")); }
        if body.starts_with(b"RMVR\x03\0\0\0") { return Self::restore_referencing(graph_bytes, body, raw); }
        if body.starts_with(b"RMVR\x02\0\0\0") { return Self::restore_quantized(graph_bytes, body); }
        let mut input = Decoder { bytes: body, offset: 0 };
        if input.take(8)? != b"RMVR\x01\0\0\0" { return Err(corrupt("owned source version")); }
        let dim = input.u32()?;
        let count = usize::try_from(input.u64()?).map_err(|_| corrupt("owned source row count"))?;
        if input.u32()? != crc32(graph_bytes) { return Err(corrupt("owned source graph mismatch")); }
        let row_size = (dim as usize).checked_mul(4).and_then(|n| n.checked_add(16))
            .ok_or_else(|| corrupt("owned source row size"))?;
        if count.checked_mul(row_size) != Some(input.remaining()) {
            return Err(corrupt("owned source rows/trailing data"));
        }
        let graph = Arc::new(MutableHnsw::open(graph_bytes)?);
        if graph.dim != dim || graph.active_nodes() as usize != count {
            return Err(corrupt("owned source graph shape"));
        }
        let owner = Self::new(graph.clone())?;
        {
            let mut writer = graph.control.writer.lock();
            let identities = restore_identities(&graph)?;
            // The short-lived snapshot above is dropped before source growth.
            let _values_memory=restore_words(dim as usize,"checkpoint restored float row")?;
            let mut values = Vec::new();
            values.try_reserve_exact(dim as usize).map_err(|_| failure("checkpoint row allocation"))?;
            for (i, (key, handle)) in identities.iter().copied().enumerate() {
                if input.u64()? != key || input.u64()? != handle.as_u64() {
                    return Err(corrupt("owned source graph identity"));
                }
                values.clear();
                for _ in 0..dim { values.push(f32::from_bits(input.u32()?)); }
                owner.install_row(handle, key, &values)?;
                if i % 256 == 255 { owner.commit_locked(&mut writer)?; }
            }
        }
        owner.commit()?;
        Ok((graph, owner))
    }

    /// Version 3 rows: f32 embedded or referenced (durable in `raw`), codes
    /// for quantized fields. A quantized row with a referenced f32 is
    /// restored code-only (its f32 stays on disk).
    fn restore_referencing(graph_bytes: &[u8], body: &[u8], raw: Option<&dyn RawRowsSnapshot>) -> Result<(Arc<MutableHnsw>, Self)> {
        let mut input = Decoder { bytes: body, offset: 8 };
        let dim = input.u32()?;
        let count = usize::try_from(input.u64()?).map_err(|_| corrupt("source row count"))?;
        if input.u32()? != crate::codec::crc32(graph_bytes) { return Err(corrupt("source graph checksum")); }
        let quantized = match input.u32()? { 0 => false, 1 => true, _ => return Err(corrupt("source row flags")) };
        let graph = Arc::new(MutableHnsw::open(graph_bytes)?);
        if graph.dim != dim || graph.active_nodes() as usize != count { return Err(corrupt("source graph shape")); }
        let mut owner = Self::new(graph.clone())?;
        if quantized {
            let length = usize::try_from(input.u64()?).map_err(|_| corrupt("quantized calibration length"))?;
            owner.quantized = Some(QuantizedOwner::restore(input.take(length)?, dim as usize, graph.similarity)?);
            owner.log = Some(CodeLogs::new(CodeLog::new()?));
            owner.attach_code_plane()?;
        }
        let retained_epochs = owner.quantized.as_ref().map(QuantizedOwner::retain_epochs).transpose()?;
        {
            let mut writer = graph.control.writer.lock();
            let identities = restore_identities(&graph)?;
            let _values_memory=restore_words(dim as usize,"checkpoint restored float row")?;
            let mut values = Vec::new();
            values.try_reserve_exact(dim as usize).map_err(|_| failure("checkpoint row allocation"))?;
            for (index, (key, handle)) in identities.iter().copied().enumerate() {
                if input.u64()? != key || input.u64()? != handle.as_u64() { return Err(corrupt("source row identity")); }
                let embedded = match input.u32()? { 0 => false, 1 => true, _ => return Err(corrupt("source row kind")) };
                values.clear();
                if embedded {
                    for _ in 0..dim { values.push(f32::from_bits(input.u32()?)); }
                }
                if quantized {
                    let epoch_id = u64::from(input.u32()?) | (u64::from(input.u32()?) << 32);
                    let words = input.u32()? as usize;
                    if words == 0 || words > 4 + (dim as usize).div_ceil(4) + 8 || words > input.remaining() / 4 {
                        return Err(corrupt("quantized row length"));
                    }
                    let _codes_memory=restore_words(words,"checkpoint restored codes")?;
                    let mut codes=Vec::new();codes.try_reserve_exact(words).map_err(|_|failure("checkpoint codes allocation"))?;
                    for _ in 0..words {codes.push(input.u32()?);}
                    let epoch = owner.quantized.as_ref().unwrap().get(epoch_id).ok_or_else(|| corrupt("quantized row calibration missing"))?;
                    if embedded {
                        owner.install_row_encoded(handle, key, &values, Some(&EncodedRow { epoch, words: codes, _memory:None }))?;
                    } else {
                        owner.install_code_only(handle, key, &epoch, &codes)?;
                    }
                } else if embedded {
                    owner.install_row(handle, key, &values)?;
                } else {
                    let row = raw.and_then(|raw| raw.row(key));
                    if let Some(raw)=raw { raw.execution_result()?; }
                    let row = row.filter(|row| row.len() == dim as usize)
                        .ok_or_else(|| corrupt("referenced source row missing from the vector store"))?;
                    owner.install_row(handle, key, row)?;
                }
                if index % 256 == 255 { owner.commit_locked(&mut writer)?; }
            }
        }
        if input.remaining() != 0 { return Err(corrupt("source trailing bytes")); }
        drop(retained_epochs);
        owner.commit()?;
        Ok((graph, owner))
    }

    /// Format 2 rows (`RMVR` v4, see [`MutableVectorCheckpoint::write_file`]):
    /// v3 rows plus code log references, the log reopened from the chunk
    /// files the header lists (mapped; nothing is copied for them).
    fn restore_logged(directory: &dyn crate::codec::directory::Directory, graph_bytes: &[u8], row_bytes: &[u8], raw: Option<&dyn RawRowsSnapshot>) -> Result<(Arc<MutableHnsw>, Self)> {
        use crate::codec::crc32;
        if row_bytes.len() < 32 { return Err(corrupt("owned source checkpoint truncated")); }
        let (body, checksum) = row_bytes.split_at(row_bytes.len() - 4);
        if crc32(body) != u32::from_le_bytes(checksum.try_into().unwrap()) {
            return Err(corrupt("owned source checksum"));
        }
        let mut input = Decoder { bytes: body, offset: 0 };
        if input.take(8)? != b"RMVR\x04\0\0\0" { return Err(corrupt("owned source version")); }
        let dim = input.u32()?;
        let count = usize::try_from(input.u64()?).map_err(|_| corrupt("source row count"))?;
        if input.u32()? != crc32(graph_bytes) { return Err(corrupt("source graph checksum")); }
        let quantized = match input.u32()? { 0 => false, 1 => true, _ => return Err(corrupt("source row flags")) };
        let graph = Arc::new(MutableHnsw::open(graph_bytes)?);
        if graph.dim != dim || graph.active_nodes() as usize != count { return Err(corrupt("source graph shape")); }
        let mut owner = Self::new(graph.clone())?;
        if quantized {
            let length = usize::try_from(input.u64()?).map_err(|_| corrupt("quantized calibration length"))?;
            owner.quantized = Some(QuantizedOwner::restore(input.take(length)?, dim as usize, graph.similarity)?);
            owner.attach_code_plane()?;
        }
        let log = match input.u32()? {
            0 => None,
            1 => {
                let id = input.u64()?;
                let row_words = input.u32()? as usize;
                let rows_per_chunk = input.u32()? as usize;
                let chunks = input.u32()? as usize;
                if row_words == 0 || rows_per_chunk == 0 || chunks > input.remaining() / 4 { return Err(corrupt("code log header")); }
                let mut refs = Vec::with_capacity(chunks);
                for chunk in 0..chunks { refs.push(ChunkRef { chunk: chunk as u32, rows: input.u32()? }); }
                Some(CodeLog::restore(directory, id, row_words, rows_per_chunk, &refs)?)
            }
            _ => return Err(corrupt("code log flag")),
        };
        if log.is_some() && !quantized { return Err(corrupt("code log of an unquantized field")); }
        owner.log = if quantized { Some(CodeLogs::new(match log { Some(log) => log, None => CodeLog::new()? })) } else { None };
        let retained_epochs = owner.quantized.as_ref().map(QuantizedOwner::retain_epochs).transpose()?;
        {
            let mut writer = graph.control.writer.lock();
            let identities = restore_identities(&graph)?;
            let _values_memory=restore_words(dim as usize,"checkpoint restored float row")?;
            let mut values = Vec::new();
            values.try_reserve_exact(dim as usize).map_err(|_| failure("checkpoint row allocation"))?;
            for (index, (key, handle)) in identities.iter().copied().enumerate() {
                if input.u64()? != key || input.u64()? != handle.as_u64() { return Err(corrupt("source row identity")); }
                match input.u32()? {
                    2 => owner.install_log_ref(handle, key, input.u64()?)?,
                    kind @ (0 | 1) => owner.restore_v3_row(&mut input, &mut values, handle, key, kind == 1, quantized, raw)?,
                    _ => return Err(corrupt("source row kind")),
                }
                if index % 256 == 255 { owner.commit_locked(&mut writer)?; }
                // Rows restored into the log: write its full chunks to files
                // as they fill, so they are page cache, not anonymous memory.
                if index % 65536 == 65535 {
                    if let Some(logs) = &owner.log {
                        let sealed = logs.active().seal_full(directory)?;
                        if !sealed.is_empty() { logs.active().install_sealed(sealed, graph.control.domain.current_generation()); }
                    }
                }
            }
        }
        if input.remaining() != 0 { return Err(corrupt("source trailing bytes")); }
        if let Some(logs) = &owner.log {
            let sealed = logs.active().seal_full(directory)?;
            if !sealed.is_empty() {
                let mut writer = graph.control.writer.lock();
                logs.active().install_sealed(sealed, graph.control.domain.current_generation());
                owner.commit_locked(&mut writer)?;
            }
        }
        drop(retained_epochs);
        owner.commit()?;
        Ok((graph, owner))
    }

    /// One v3-layout row (after its identity and kind): embedded f32 when
    /// `embedded`, then the codes of a quantized field. Caller holds the
    /// graph writer boundary (restore).
    fn restore_v3_row(&self, input: &mut Decoder<'_>, values: &mut Vec<f32>, handle: NodeHandle, key: u64, embedded: bool, quantized: bool, raw: Option<&dyn RawRowsSnapshot>) -> Result<()> {
        let dim = self.graph.dim as usize;
        values.clear();
        if embedded {
            for _ in 0..dim { values.push(f32::from_bits(input.u32()?)); }
        }
        if quantized {
            let epoch_id = u64::from(input.u32()?) | (u64::from(input.u32()?) << 32);
            let words = input.u32()? as usize;
            if words == 0 || words > 4 + dim.div_ceil(4) + 8 || words > input.remaining() / 4 {
                return Err(corrupt("quantized row length"));
            }
            let _codes_memory=restore_words(words,"checkpoint restored codes")?;
            let mut codes=Vec::new();codes.try_reserve_exact(words).map_err(|_|failure("checkpoint codes allocation"))?;
            for _ in 0..words {codes.push(input.u32()?);}
            let epoch = self.quantized.as_ref().ok_or_else(|| corrupt("quantized row of an unquantized owner"))?
                .get(epoch_id).ok_or_else(|| corrupt("quantized row calibration missing"))?;
            // A row whose f32 the vector store holds durably (bit for bit)
            // needs no f32 here: restore it code-only, like a shed row, so a
            // checkpoint captured before shedding caught up does not restore
            // every unshed row's f32 into anonymous memory.
            let durable = embedded && raw.and_then(|raw| raw.row(key))
                .is_some_and(|row| row.len() == dim && row.iter().zip(values.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
            if let Some(raw) = raw.filter(|_| embedded) { raw.execution_result()?; }
            if embedded && !durable {
                self.install_row_encoded(handle, key, values, Some(&EncodedRow { epoch, words: codes, _memory:None }))
            } else if self.log.as_ref().is_some_and(CodeLogs::shed_room) && crate::codec::write_version() >= crate::codec::RMVC2_SINCE {
                // Only a format-2 checkpoint can reference log rows (as shedding).
                self.install_code_only_logged(handle, key, &epoch, &codes)
            } else {
                self.install_code_only(handle, key, &epoch, &codes)
            }
        } else if embedded {
            self.install_row(handle, key, values)
        } else {
            let row = raw.and_then(|raw| raw.row(key));
            if let Some(raw)=raw { raw.execution_result()?; }
            let row = row.filter(|row| row.len() == dim)
                .ok_or_else(|| corrupt("referenced source row missing from the vector store"))?;
            // Durable: restore it shed (raw-only) instead of copying its f32.
            if shed_f32_rows() { self.install_raw_only(handle, key) } else { self.install_row(handle, key, row) }
        }
    }

    /// Install a restored f32 node as a [`RAW_ONLY`] row (its f32 durable
    /// in [`RawRows`]). Caller holds the graph writer boundary (restore).
    fn install_raw_only(&self, handle: NodeHandle, key: u64) -> Result<()> {
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !self.identity_matches(&guard, handle, key) {
            return Err(failure("source installation identity mismatch"));
        }
        let mut directory = self.directory.lock();
        let slot = internal_id(handle) as usize;
        self.grow(&mut directory, slot + 1)?;
        let owner = &directory.current[slot];
        if owner.load(Relaxed) != 0 { return Err(failure("source slot still owned")); }
        let values = raw_only_row(key, handle)?;
        let mut batch = self.rows.writer().stage_batch(1, &[]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let staged = batch.stage(&values).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let reference = batch.resolve(staged).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        owner.store(reference.raw(), Release);
        unsafe { batch.finish_after_publish() };
        Ok(())
    }

    fn restore_quantized(graph_bytes: &[u8], body: &[u8]) -> Result<(Arc<MutableHnsw>, Self)> {
        let mut input = Decoder { bytes: body, offset: 8 };
        let dim = input.u32()?;
        let count = usize::try_from(input.u64()?).map_err(|_| corrupt("quantized row count"))?;
        if input.u32()? != crate::codec::crc32(graph_bytes) { return Err(corrupt("quantized graph checksum")); }
        let length = usize::try_from(input.u64()?).map_err(|_| corrupt("quantized calibration length"))?;
        let graph = Arc::new(MutableHnsw::open(graph_bytes)?);
        if graph.dim != dim || graph.active_nodes() as usize != count { return Err(corrupt("quantized graph shape")); }
        let quantized = QuantizedOwner::restore(input.take(length)?, dim as usize, graph.similarity)?;
        let mut owner = Self::new(graph.clone())?;
        owner.quantized = Some(quantized);
        owner.log = Some(CodeLogs::new(CodeLog::new()?));
        owner.attach_code_plane()?;
        let retained_epochs = owner.quantized.as_ref().unwrap().retain_epochs()?;
        {
            let mut writer = graph.control.writer.lock();
            let identities = restore_identities(&graph)?;
            for (index, (key, handle)) in identities.iter().copied().enumerate() {
                if input.u64()? != key || input.u64()? != handle.as_u64() { return Err(corrupt("quantized row identity")); }
                let words = input.u32()? as usize;
                let base = (dim as usize).div_ceil(8) * 8 + 8;
                // At most byte-aligned 7-bit codes plus four correction words.
                if words <= base || words > base + 4 + (dim as usize).div_ceil(4) || words > input.remaining() / 4 {
                    return Err(corrupt("quantized row length"));
                }
                let _values_memory=restore_words(words,"checkpoint quantized source row")?;
                let mut values=Vec::new();values.try_reserve_exact(words).map_err(|_|failure("checkpoint quantized row allocation"))?;
                for _ in 0..words {values.push(f32::from_bits(input.u32()?));}
                let raw = decode_row(&values, dim as usize, handle, key).ok_or_else(|| corrupt("quantized row owner"))?;
                let epoch_id = u64::from(values[3].to_bits()) | (u64::from(values[4].to_bits()) << 32);
                let epoch = owner.quantized.as_ref().unwrap().get(epoch_id).ok_or_else(|| corrupt("quantized row calibration missing"))?;
                let _encoded_memory=restore_words(words-base,"checkpoint encoded row copy")?;
                let mut codes=Vec::new();codes.try_reserve_exact(words-base).map_err(|_|failure("checkpoint encoded row allocation"))?;
                codes.extend(values[base..].iter().map(|v|v.to_bits()));
                let encoded = EncodedRow { epoch, words:codes, _memory:None };
                // Check row shape and finite score without rebuilding calibration.
                // Quantized scorer construction: optional normalized f32 row,
                // byte codes + byte reuse buffer, OSQ centered f32 + i32 codes.
                // The quantizer itself is shared, not cloned calibration rows.
                let scorer_bytes=(dim as usize).checked_mul(4+1+1+4+4).ok_or_else(||failure("checkpoint scorer workspace"))?;
                let _scorer_memory=crate::admission::shared().reserve(scorer_bytes as u64,"checkpoint validation scorer")?;
                let scorer = crate::vector::es_quant::EsQuantScorer::new(encoded.epoch.q.clone(), graph.similarity, raw);
                if scorer.score_words(&values[base..], dim as usize).is_none_or(|s| !s.is_finite()) {
                    return Err(corrupt("quantized row codes/corrections"));
                }
                owner.install_row_encoded(handle, key, raw, Some(&encoded))?;
                if index % 256 == 255 { owner.commit_locked(&mut writer)?; }
            }
        }
        if input.remaining() != 0 { return Err(corrupt("quantized trailing bytes")); }
        drop(retained_epochs);
        owner.commit()?;
        Ok((graph, owner))
    }

    pub fn new(graph: Arc<MutableHnsw>) -> Result<Self> {
        let width = (graph.dim as usize).div_ceil(8) * 8 + 8;
        if graph.dim == 0 { return Err(failure("zero source dimensions")); }
        // Larger rows use ArrayStore's large-array representation instead of
        // allocating a size-class descriptor for every possible dimension.
        let mut config = ArrayStoreConfig::hnsw::<22>(width.min(4096));
        // Keep every ArrayStore reference below LOG_TAG, and buffer 511
        // (whose last offset is PLANE_REF) unallocated.
        config.max_buffers = config.max_buffers.min((1 << 9) - 1);
        let rows = ArrayStore::new(graph.control.domain.clone(), config)
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut current=SourceDirectory::new(16,None)?;
        let pointer = AtomicPtr::new(current.as_mut_ptr());
        // Rows by node id, and the graph's dense visibility words the dense
        // traversal checks (`MutableHnswSnapshot::dense_beam`).
        let row_plane = (!shed_f32_rows()).then(|| Box::new(RowPlane::new(graph.dim as usize)));
        graph.control.enable_dense_publications()?;
        Ok(Self { graph, quantized: None, log: None, compact_min_rows: std::sync::atomic::AtomicU64::new(CODE_LOG_COMPACT_MIN_ROWS), raw: std::sync::OnceLock::new(), unshed: parking_lot::Mutex::new(UnshedQueue::default()),
            rows, directory: parking_lot::Mutex::new(Directory { current, retired: Vec::new(), retired_memory:None }),
            pending_reclamation: parking_lot::Mutex::new(Vec::new()),
            pointer, len: AtomicUsize::new(16), plane: None, row_plane,
            quarantine: parking_lot::Mutex::new(PlaneQuarantine::default()), quarantined: AtomicUsize::new(0),
            exact_mutation_rows: false })
    }

    pub(crate) fn new_quantized(graph: Arc<MutableHnsw>, spec: Option<crate::vector::es_quant::EsQuant>) -> Result<Self> {
        let mut owner = Self::new(graph)?;
        owner.quantized = spec.map(|spec| QuantizedOwner::new(spec, owner.graph.dim as usize, owner.graph.similarity)).transpose()?;
        if owner.quantized.is_some() {
            // Quantized rows keep their f32 (until shed) in the row store.
            owner.row_plane = None;
            owner.log = Some(CodeLogs::new(CodeLog::new()?));
            owner.attach_code_plane()?;
        }
        Ok(owner)
    }
    /// A quantized owner's code plane, and its graph's dense publication
    /// copy (what the plane traversal checks visibility in). Called once the
    /// owner is quantized, before any row is installed; an f32 field has
    /// neither, so its search and insert paths carry none of their cost.
    fn attach_code_plane(&mut self) -> Result<()> {
        self.graph.control.enable_dense_publications()?;
        self.plane = Some(Box::default());
        self.exact_mutation_rows = self.quantized.as_ref().is_some_and(|q| q.spec().bits < 4);
        Ok(())
    }
    pub(crate) fn quantization(&self) -> Option<crate::vector::es_quant::EsQuant> {
        self.quantized.as_ref().map(QuantizedOwner::spec)
    }
    /// The quantizer unrescored result scores are estimated with
    /// ([`QuantizedOwner::score_quantizer`]); `None` for f32 fields.
    pub(crate) fn score_quantizer(&self) -> Option<crate::vector::es_quant::EsQuantizer> {
        self.quantized.as_ref()?.score_quantizer()
    }
    pub(crate) fn prepare_codes(&self, rows: &[&[f32]]) -> Result<Vec<Option<EncodedRow>>> {
        match &self.quantized {
            Some(owner) => Ok(owner.prepare(rows)?.into_iter().map(Some).collect()),
            None => Ok((0..rows.len()).map(|_| None).collect()),
        }
    }

    /// Restore owned rows for an already decoded logical graph, without
    /// reinserting nodes or changing adjacency. The caller supplies the source
    /// generation selected by the same durable manifest as the graph.
    /// This adapter validates identity/shape; it is not itself a WAL protocol.
    pub fn restore(graph: Arc<MutableHnsw>, source: &(impl MutableVectorSource + ?Sized)) -> Result<Self> {
        let owner = Self::new(graph.clone())?;
        {
            let mut writer = graph.control.writer.lock();
            // Copy logical identities, then drop the graph read guard. Holding
            // it for the entire import would retain every superseded source
            // buffer and can exhaust retirement budgets on large checkpoints.
            // The writer fence keeps the private decoded topology unchanged.
            let identities=restore_identities(&graph)?;
            for (index, (key, handle)) in identities.iter().copied().enumerate() {
                let row = source.get_node(handle, key).ok_or_else(|| failure("missing restored source row"))?;
                owner.install_row(handle, key, row)?;
                if index % 256 == 255 { owner.commit_locked(&mut writer)?; }
            }
        }
        owner.commit()?;
        Ok(owner)
    }

    pub fn pin(&self) -> Result<MutableVectorRead<'_>> {
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.graph.control.check_readable()?;
        self.rows.view(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        Ok(MutableVectorRead { owner: self, guard, raw: LazyRaw::default(), decoded: DecodedRows::default() })
    }

    /// Capture graph visibility before the source guard. Lookup still uses the
    /// current source directory, so concurrent directory growth cannot hide
    /// rows acquired later by preparation or graph traversal.
    pub fn read(&self) -> Result<(MutableHnswSnapshot, MutableVectorRead<'_>)> {
        let graph = self.graph.try_snapshot()?;
        Ok((graph, self.pin()?))
    }

    /// A document-only update can keep its vector/graph identity when every
    /// cell is unchanged. The ES owner still has to publish document visibility.
    pub fn unchanged(&self, handle: NodeHandle, key: u64, values: &[f32]) -> Result<bool> {
        let read = self.pin()?;
        Ok(read.get_node(handle, key).is_some_and(|row| row.len() == values.len()
            && row.iter().zip(values).all(|(a, b)| a.to_bits() == b.to_bits())))
    }

    /// Relocate an unchanged row under the same graph handle. Old row borrows
    /// remain valid through their generation guards. Reject changed contents:
    /// changing vector geometry requires graph remove/insert, not relocation.
    pub fn relocate(&self, handle: NodeHandle, key: u64, values: &[f32]) -> Result<()> {
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        let read = self.pin()?;
        let previous = read.get_node(handle, key).ok_or_else(|| failure("source relocation owner absent"))?;
        if previous.len() != values.len() || previous.iter().zip(values).any(|(a,b)| a.to_bits() != b.to_bits()) {
            return Err(failure("source relocation changes vector geometry"));
        }
        let directory = self.directory.lock();
        let owner = &directory.current[internal_id(handle) as usize];
        let old = LevelsRef::from_raw(owner.load(Acquire));
        // A code log or row plane row is immutable and never moves.
        if log_index(old.raw()).is_some() || old.raw() == PLANE_REF { return Ok(()); }
        let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let token = unsafe { view.retirement_token(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
        let encoded: Vec<u32> = unsafe { view.get(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?
            .iter().map(|value| value.to_bits()).collect();
        let mut batch = self.rows.writer().stage_batch(1, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let index = batch.stage(&encoded).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let new = batch.resolve(index).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        owner.store(new.raw(), Release);
        unsafe { batch.finish_after_publish() };
        Ok(())
    }

    /// Bytes of the live source rows (identity headers, f32 cells of rows
    /// that keep them, codes and corrections).
    pub fn row_bytes_used(&self) -> usize {
        self.rows.stats().used_bytes + self.row_plane.as_ref().map_or(0, |plane| plane.bytes())
    }

    /// Bytes of the field's code log (quantized codes and corrections),
    /// in memory and in its mapped files: `0` without one.
    pub fn code_log_bytes(&self) -> usize {
        self.log.as_ref().map_or(0, |logs| { let (anon, mapped) = logs.memory_bytes(); anon + mapped })
    }

    /// Source storage and directory accounting, including generation-held
    /// copies. Payload accounting is separate from process/allocator RSS.
    pub fn memory_bytes(&self) -> (usize, usize) {
        let rows = self.rows.stats();
        let directory = self.directory.lock();
        let held = directory.retired.iter().map(|(_, array)| array.len() * std::mem::size_of::<AtomicU32>()).sum::<usize>();
        let quantized = self.quantized.as_ref().map_or(0, QuantizedOwner::memory_bytes);
        let log = self.log.as_ref().map_or(0, |logs| logs.memory_bytes().0);
        (rows.allocated_bytes + held + directory.current.len() * std::mem::size_of::<AtomicU32>() + quantized + log + self.plane.as_ref().map_or(0, |plane| plane.bytes()) + self.row_plane.as_ref().map_or(0, |plane| plane.bytes()), rows.held_bytes + held)
    }

    /// Compact tensor storage by relocating references, independently of text
    /// files and graph geometry. Uses the same bounded default buffer selection
    /// as graph array compaction, including partial-failure reactivation.
    pub fn compact_storage(&self) -> Result<usize> {
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        let selected = self.rows.writer().default_compaction_buffers();
        if selected.is_empty() { return Ok(0); }
        self.rows.writer().start_compaction(&selected).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut finished = 0;
        let result = (|| -> Result<usize> {
            let read = self.pin()?;
            let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
            let directory = self.directory.lock();
            for owner in directory.current.iter().take(self.graph.num_nodes() as usize + 1) {
                let old = LevelsRef::from_raw(owner.load(Acquire));
                if !old.is_valid() || log_index(old.raw()).is_some() || old.raw() == PLANE_REF || !selected.contains(&old.buffer_id()) { continue; }
                let values = unsafe { view.get(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
                let mut encoded = Vec::new();
                encoded.try_reserve_exact(values.len()).map_err(|_| failure("source compaction staging allocation"))?;
                encoded.extend(values.iter().map(|v| v.to_bits()));
                let token = unsafe { view.retirement_token(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
                let mut batch = self.rows.writer().stage_batch(1, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
                let index = batch.stage(&encoded).map_err(crate::vector::hnsw::mutable::failure_error)?;
                let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
                let new = batch.resolve(index).map_err(crate::vector::hnsw::mutable::failure_error)?;
                let batch = batch.begin_publish();
                owner.store(new.raw(), Release);
                unsafe { batch.finish_after_publish() };
            }
            for &id in &selected {
                self.rows.writer().finish_compaction(id).map_err(crate::vector::hnsw::mutable::failure_error)?;
                finished += 1;
            }
            Ok(finished)
        })();
        if result.is_err() {
            for &id in &selected[finished..] {
                if self.rows.writer().abort_compaction(id).is_err() { self.graph.control.poison(); }
            }
        }
        // Even partial moves may retire storage. Assign it before the common
        // generation advances; never replay already published owner changes.
        self.rows.writer().assign_generation().map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.graph.control.finish_mutation(&mut writer);
        result
    }

    fn grow(&self, directory: &mut Directory, required: usize) -> Result<()> {
        if required <= directory.current.len() { return Ok(()); }
        let capacity = required.checked_mul(2).ok_or_else(|| failure("source directory overflow"))?;
        let mut replacement=SourceDirectory::new(capacity,Some(&directory.current))?;
        directory.reserve_retirement()?;
        let pointer = replacement.as_mut_ptr();
        let old = std::mem::replace(&mut directory.current, replacement);
        // The outer graph writer prevents generation advancement between
        // recording this retirement and publishing the new directory.
        directory.retired.push((self.graph.control.domain.current_generation(), old));
        self.pointer.store(pointer, Release);
        self.len.store(capacity, Release);
        Ok(())
    }

    /// Install an immutable row for an outstanding graph reservation. Float
    /// cells begin on a 32-byte boundary after an immutable identity header.
    /// Source identity never depends on a subsequently reused node-table slot.
    pub fn install(&self, reservation: &InsertReservation, vector: &[f32]) -> Result<()> {
        let encoded = self.prepare_codes(&[vector])?.pop().flatten();
        self.install_prepared(reservation, vector, encoded.as_ref())
    }
    pub(crate) fn install_prepared(&self, reservation: &InsertReservation, vector: &[f32], encoded: Option<&EncodedRow>) -> Result<()> {
        if !Arc::ptr_eq(&reservation.control, &self.graph.control) {
            return Err(failure("source reservation belongs to another graph"));
        }
        if vector.len() != self.graph.dim as usize || !self.graph.similarity.valid_row(vector) {
            return Err(failure("invalid source vector"));
        }
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        if writer.pending.get(&reservation.key()) != Some(&reservation.handle()) {
            return Err(failure("source reservation is no longer pending"));
        }
        self.install_row_encoded(reservation.handle(), reservation.key(), vector, encoded)
    }

    /// Reserve graph slots for `keys` (increasing) and install their rows
    /// (as [`MutableHnsw::reserve_insert`] then [`Self::install_prepared`]
    /// for each) under ONE topology-writer hold, where a writer hold per
    /// call made a staging slice queue twice per row behind the
    /// completions of every other bulk of the shard. Each reservation is
    /// pushed to `out` as soon as it exists: on an error the caller drops
    /// them after this returns (a reservation's drop takes the writer).
    pub(crate) fn reserve_install_batch(&self, keys: &[u64], rows: &[&[f32]], encoded: &[Option<EncodedRow>],
        out: &mut Vec<InsertReservation>) -> Result<()> {
        if keys.len() != rows.len() || keys.len() != encoded.len() {
            return Err(failure("source batch shape mismatch"));
        }
        if rows.iter().any(|row| row.len() != self.graph.dim as usize || !self.graph.similarity.valid_row(row)) {
            return Err(failure("invalid source vector"));
        }
        let mut writer = self.graph.control.writer.lock();
        for ((&key, &row), encoded) in keys.iter().zip(rows).zip(encoded) {
            let reservation = self.graph.control.reserve_insert_locked(&mut writer, key, false)?;
            let (handle, key) = (reservation.handle(), reservation.key());
            out.push(reservation);
            self.install_row_encoded(handle, key, row, encoded.as_ref())?;
        }
        Ok(())
    }

    // Caller holds the graph writer boundary. No reference is published before
    // validating the complete row, so restoration failure drops only this owner.
    fn install_row(&self, handle: NodeHandle, key: u64, vector: &[f32]) -> Result<()> {
        self.install_row_encoded(handle, key, vector, None)
    }
    fn install_row_encoded(&self, handle: NodeHandle, key: u64, vector: &[f32], encoded: Option<&EncodedRow>) -> Result<()> {
        if self.quantized.is_some() != encoded.is_some() { return Err(failure("missing/unexpected encoded source row")); }
        if let (Some(quantized), Some(encoded)) = (&self.quantized, encoded) {
            if quantized.get(encoded.epoch.id).is_none_or(|epoch| !Arc::ptr_eq(&epoch, &encoded.epoch)) {
                return Err(failure("encoded row belongs to a foreign calibration owner"));
            }
        }
        if vector.len() != self.graph.dim as usize || !self.graph.similarity.valid_row(vector) {
            return Err(failure("invalid restored source vector"));
        }
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !self.identity_matches(&guard, handle, key) {
            return Err(failure("source installation identity mismatch"));
        }
        let mut directory = self.directory.lock();
        let slot = internal_id(handle) as usize;
        self.grow(&mut directory, slot + 1)?;
        let owner = &directory.current[slot];
        if owner.load(Relaxed) != 0 { return Err(failure("source slot still owned")); }
        let width = vector.len().div_ceil(8).checked_mul(8).and_then(|v|v.checked_add(8))
            .and_then(|v|v.checked_add(encoded.map_or(0,|row|row.words.len())))
            .ok_or_else(||failure("source row size overflow"))?;
        let _row_memory=restore_words(width,"source installed row workspace")?;
        let mut values = Vec::new();
        values.try_reserve_exact(width).map_err(|_|failure("source row allocation"))?;
        encode_row(&mut values, vector, key, handle)?;
        if let Some(encoded) = encoded {
            values[3] = encoded.epoch.id as u32;
            values[4] = (encoded.epoch.id >> 32) as u32;
            values.extend_from_slice(&encoded.words);
            // Sheds its f32 once that is durable.
            let mut queue=self.unshed.lock();queue.admit_push()?;queue.push_back((handle,key));
        } else if self.quantized.is_none() && shed_f32_rows() {
            // An f32 field's row sheds its f32 too, once durable: searches
            // then read it from the mapped vector segment (page cache).
            let mut queue=self.unshed.lock();queue.admit_push()?;queue.push_back((handle,key));
        }
        let mut quarantined = false;
        if let (Some(plane), None) = (&self.row_plane, encoded) {
            let id = u32::try_from(slot).map_err(|_| failure("row plane slot"))?;
            let mut quarantine = self.quarantine.lock();
            let reusable = quarantine.released.get(&id)
                .is_none_or(|&released| released < self.graph.control.domain.update_oldest_used_generation());
            if reusable {
                quarantine.released.remove(&id);
                // The row's only copy: complete in the plane before the slot
                // (and later the node) can be followed to it.
                plane.write(id, &values)?;
                owner.store(PLANE_REF, Release);
                return Ok(());
            }
            quarantine.installed.try_reserve(1).map_err(|_| failure("row plane quarantine"))?;
            quarantine.installed.push((handle, key));
            self.quarantined.store(quarantine.installed.len(), Release);
            quarantined = true;
        }
        let _ = quarantined;
        let mut batch = self.rows.writer().stage_batch(1, &[]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let staged = batch.stage(&values).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let reference = batch.resolve(staged).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        if let (Some(quantized), Some(encoded)) = (&self.quantized, encoded) {
            quantized.installed(&encoded.epoch);
            self.write_plane(slot, encoded.epoch.id, &encoded.words);
        }
        owner.store(reference.raw(), Release);
        unsafe { batch.finish_after_publish() };
        Ok(())
    }

    /// Move rows installed over a quarantined slot into the plane once no
    /// reader older than the slot's release remains (`PlaneQuarantine`),
    /// retiring their row store copies. Caller holds the graph writer.
    fn move_quarantined_rows(&self) -> Result<()> {
        let Some(plane) = &self.row_plane else { return Ok(()) };
        let oldest = self.graph.control.domain.update_oldest_used_generation();
        let mut quarantine = self.quarantine.lock();
        let read = self.pin()?;
        let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let directory = self.directory.lock();
        let mut i = 0;
        while i < quarantine.installed.len() {
            let (handle, key) = quarantine.installed[i];
            let id = internal_id(handle);
            if quarantine.released.get(&id).is_some_and(|&released| released >= oldest) { i += 1; continue; }
            quarantine.installed.swap_remove(i);
            quarantine.released.remove(&id);
            let Some(owner) = directory.current.get(id as usize) else { continue };
            let old = LevelsRef::from_raw(owner.load(Acquire));
            // Released (or replaced) since: nothing to move.
            if !old.is_valid() || old.raw() == PLANE_REF || log_index(old.raw()).is_some() { continue; }
            let Some(values) = read.encoded_node(handle, key) else { continue };
            let encoded: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let token = unsafe { view.retirement_token(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
            let batch = self.rows.writer().stage_batch(0, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
            let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
            plane.write(id, &encoded)?;
            let batch = batch.begin_publish();
            owner.store(PLANE_REF, Release);
            unsafe { batch.finish_after_publish() };
        }
        self.quarantined.store(quarantine.installed.len(), Release);
        Ok(())
    }

    /// Node `slot`'s [`CodePlane`] entry, best effort: an entry that cannot
    /// be written (memory admission) is left empty, and searches score that
    /// node from its row. Caller holds the graph writer, before publication.
    fn write_plane(&self, slot: usize, epoch: u64, words: &[u32]) {
        let Ok(id) = u32::try_from(slot) else { return };
        let Some(plane) = &self.plane else { return };
        if plane.write(id, epoch, words).is_err() { plane.clear(id); }
    }

    /// Install a code-only quantized row (its f32 durable in [`RawRows`]).
    /// Caller holds the graph writer boundary (restore).
    fn install_code_only(&self, handle: NodeHandle, key: u64, epoch: &Arc<quantized::Epoch>, words: &[u32]) -> Result<()> {
        let quantized = self.quantized.as_ref().ok_or_else(|| failure("code-only row on an unquantized owner"))?;
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !self.identity_matches(&guard, handle, key) {
            return Err(failure("source installation identity mismatch"));
        }
        let mut directory = self.directory.lock();
        let slot = internal_id(handle) as usize;
        self.grow(&mut directory, slot + 1)?;
        let owner = &directory.current[slot];
        if owner.load(Relaxed) != 0 { return Err(failure("source slot still owned")); }
        let values = code_only_row(key, handle, epoch.id, words)?;
        let mut batch = self.rows.writer().stage_batch(1, &[]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let staged = batch.stage(&values).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let reference = batch.resolve(staged).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        quantized.installed(epoch);
        self.write_plane(slot, epoch.id, words);
        owner.store(reference.raw(), Release);
        unsafe { batch.finish_after_publish() };
        Ok(())
    }

    /// Where durable f32 rows are read for code-only rows (set once, by the
    /// owning vector field).
    pub fn set_raw_rows(&self, raw: Arc<dyn RawRows>) {
        let _ = self.raw.set(raw);
    }

    /// Replace the row of `(handle, key)` with `values` (same identity),
    /// retiring the old one through the generation domain.
    fn replace_row(&self, handle: NodeHandle, key: u64, read: &MutableVectorRead<'_>, values: &[u32]) -> Result<()> {
        let directory = self.directory.lock();
        let owner = directory.current.get(internal_id(handle) as usize).ok_or_else(|| failure("source row owner absent"))?;
        let old = LevelsRef::from_raw(owner.load(Acquire));
        if !old.is_valid() || read.encoded_node(handle, key).is_none() { return Err(failure("source row owner absent")); }
        if old.raw() == PLANE_REF { return Err(failure("a row plane row is never replaced")); }
        // A code log row is not an ArrayStore allocation: nothing to retire
        // (the row turns dead once the slot stops tagging it, below).
        let old_log = log_index(old.raw()).and_then(|_| self.log.as_ref()?.get(log_slot(old.raw())));
        let tokens = if log_index(old.raw()).is_some() { Vec::new() } else {
            let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
            vec![unsafe { view.retirement_token(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?]
        };
        let mut batch = self.rows.writer().stage_batch(1, &tokens).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let index = batch.stage(values).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let new = batch.resolve(index).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        owner.store(new.raw(), Release);
        unsafe { batch.finish_after_publish() };
        if let Some(log) = old_log { log.untagged(); }
        Ok(())
    }

    /// Move a quantized row into the code log: `values` is the code-only
    /// row of the same identity. Its ArrayStore row is retired through the
    /// generation domain and the slot tagged with the log index. The caller
    /// holds the graph writer (the log's single appender).
    fn move_row_to_log(&self, logs: &CodeLogs, handle: NodeHandle, key: u64, read: &MutableVectorRead<'_>, values: &[u32]) -> Result<()> {
        let (slot, log) = (logs.active_slot(), logs.active());
        let directory = self.directory.lock();
        let owner = directory.current.get(internal_id(handle) as usize).ok_or_else(|| failure("source row owner absent"))?;
        let old = LevelsRef::from_raw(owner.load(Acquire));
        if !old.is_valid() || log_index(old.raw()).is_some() || old.raw() == PLANE_REF || read.encoded_node(handle, key).is_none() {
            return Err(failure("source row owner absent"));
        }
        let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let token = unsafe { view.retirement_token(old) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
        // Complete in the log before any reader can follow the tag to it.
        let index = log.append(values)?;
        let tagged = log_tag(slot, index)?;
        let batch = self.rows.writer().stage_batch(0, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.begin_publish();
        owner.store(tagged, Release);
        unsafe { batch.finish_after_publish() };
        log.tagged();
        Ok(())
    }

    /// Install a restored code-only node by appending its row to the code
    /// log (sealed to files as chunks fill, see `restore_logged`), not to
    /// the anonymous row store. Caller holds the graph writer boundary.
    fn install_code_only_logged(&self, handle: NodeHandle, key: u64, epoch: &Arc<quantized::Epoch>, words: &[u32]) -> Result<()> {
        let quantized = self.quantized.as_ref().ok_or_else(|| failure("code-only row on an unquantized owner"))?;
        let log = self.log.as_ref().ok_or_else(|| failure("code-only log row without a code log"))?;
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !self.identity_matches(&guard, handle, key) {
            return Err(failure("source installation identity mismatch"));
        }
        let mut directory = self.directory.lock();
        let slot = internal_id(handle) as usize;
        self.grow(&mut directory, slot + 1)?;
        let owner = &directory.current[slot];
        if owner.load(Relaxed) != 0 { return Err(failure("source slot still owned")); }
        let values = code_only_row(key, handle, epoch.id, words)?;
        let (log_slot, active) = (log.active_slot(), log.active());
        let index = active.append(&values)?;
        quantized.installed(epoch);
        self.write_plane(slot, epoch.id, words);
        owner.store(log_tag(log_slot, index)?, Release);
        active.tagged();
        drop(guard);
        Ok(())
    }

    /// Install a restored node whose code-only row is log row `index`.
    /// Caller holds the graph writer boundary (restore).
    fn install_log_ref(&self, handle: NodeHandle, key: u64, index: u64) -> Result<()> {
        let quantized = self.quantized.as_ref().ok_or_else(|| corrupt("code log row on an unquantized owner"))?;
        let logs = self.log.as_ref().ok_or_else(|| corrupt("code log row without a code log"))?;
        let (log_slot, log) = (logs.active_slot(), logs.active());
        if index > code_log::MAX_ROWS || index >= log.rows() { return Err(corrupt("code log row index")); }
        let guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !self.identity_matches(&guard, handle, key) {
            return Err(failure("source installation identity mismatch"));
        }
        let row = log.row(index).ok_or_else(|| corrupt("code log row absent"))?;
        if !is_code_only(row) || decode_row(row, self.graph.dim as usize, handle, key).is_none() {
            return Err(corrupt("code log row identity"));
        }
        let epoch_id = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
        let epoch = quantized.get(epoch_id).ok_or_else(|| corrupt("quantized row calibration missing"))?;
        let mut directory = self.directory.lock();
        let slot = internal_id(handle) as usize;
        self.grow(&mut directory, slot + 1)?;
        let owner = &directory.current[slot];
        if owner.load(Relaxed) != 0 { return Err(failure("source slot still owned")); }
        quantized.installed(&epoch);
        let words: Vec<u32> = row[8..].iter().map(|w| w.to_bits()).collect();
        self.write_plane(slot, epoch_id, &words);
        // The log's slot, not the node's (`slot` above is its row slot).
        owner.store(log_tag(log_slot, index)?, Release);
        log.tagged();
        drop(guard);
        Ok(())
    }

    /// Write full code log chunks to their files (off the graph writer) and
    /// serve them from the mapping from now on: their pages are page cache,
    /// not anonymous memory. A no-op for f32 fields.
    /// Also runs a bounded step of code log compaction
    /// ([`Self::compact_code_log`]).
    pub fn seal_code_log(&self, directory: &dyn crate::codec::directory::Directory) -> Result<()> {
        if self.log.is_none() { return Ok(()); }
        // Compaction is housekeeping: a failure (say, memory admission for
        // the new log) retries at the next commit instead of failing this one.
        if let Err(error) = self.compact_code_log(directory) {
            tracing::warn!("code log compaction deferred: {error}");
        }
        self.seal_active_log(directory)
    }

    fn seal_active_log(&self, directory: &dyn crate::codec::directory::Directory) -> Result<()> {
        let Some(logs) = &self.log else { return Ok(()) };
        // Keeps `log` allocated through a concurrent compaction switch and
        // reclamation (it would retire at a generation >= this guard's).
        let _guard = self.graph.control.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let slot = logs.active_slot();
        let log = logs.active();
        let sealed = log.seal_full(directory)?;
        if sealed.is_empty() { return Ok(()); }
        let mut writer = self.graph.control.writer.lock();
        // A compaction may have switched logs meanwhile: the sealed chunks
        // still belong to `log`, which stays allocated until reclaimed
        // under this writer (it retires at a later generation).
        if let Some(log) = logs.get(slot) { log.install_sealed(sealed, self.graph.control.domain.current_generation()); }
        self.commit_locked(&mut writer)
    }

    /// Bounded code log compaction (a no-op for f32 fields). When the active
    /// log is mostly dead rows ([`CodeLog::wants_compaction`]) or nearly
    /// full, a fresh log (new id) becomes active and up to
    /// [`CODE_LOG_COMPACT_ROWS_PER_CALL`] live rows are copied into it,
    /// retagging their source slots; later calls continue where this one
    /// stopped. Once no slot names the old log it is retired, freed when no
    /// older reader remains, and its files go once no commit references
    /// them. The graph is marked changed so the next commit saves a
    /// checkpoint referencing only the new log. Returns the rows copied.
    pub fn compact_code_log(&self, directory: &dyn crate::codec::directory::Directory) -> Result<usize> {
        let Some(logs) = &self.log else { return Ok(0) };
        {
            let mut writer = self.graph.control.writer.lock();
            self.graph.control.admit_write(&mut writer)?;
            // Free a previous switch's log if no older reader remains.
            logs.reclaim(self.graph.control.domain.update_oldest_used_generation());
            if logs.compacting().is_none() {
                if !logs.idle() || !logs.active().wants_compaction(self.compact_min_rows.load(Relaxed)) { return Ok(0); }
                logs.begin_compaction(CodeLog::new()?)?;
            }
        }
        let mut copied = 0;
        loop {
            let (moved, done) = {
                let mut writer = self.graph.control.writer.lock();
                self.graph.control.admit_write(&mut writer)?;
                let result = self.migrate_log_batch(logs);
                let committed = self.commit_locked(&mut writer);
                let (moved, done) = result?;
                committed?;
                (moved, done)
            };
            copied += moved;
            // Keep the copy's anonymous chunks bounded: full ones go to files.
            self.seal_active_log(directory)?;
            // A full target stalls the copy (backpressure): resume later.
            if done || copied >= CODE_LOG_COMPACT_ROWS_PER_CALL || logs.active().is_full() { return Ok(copied); }
        }
    }

    /// Copy the next batch of rows still in the log being compacted into
    /// the active log and retag their slots. Caller holds the graph writer
    /// (every tag writer does: shedding, release, unshed, restore), so no
    /// slot can gain a tag naming the old log meanwhile. Returns the rows
    /// copied and whether the migration finished.
    fn migrate_log_batch(&self, logs: &CodeLogs) -> Result<(usize, bool)> {
        let Some((old, cursor)) = logs.compacting() else { return Ok((0, true)) };
        let (slot, target) = (logs.active_slot(), logs.active());
        let old_slot = 1 - slot;
        let directory = self.directory.lock();
        let end = cursor.saturating_add(CODE_LOG_COMPACT_BATCH).min(directory.current.len());
        let mut words = Vec::new();
        let mut moved = 0;
        for position in cursor..end {
            let owner = &directory.current[position];
            let raw = owner.load(Acquire);
            let Some(index) = log_index(raw).filter(|_| log_slot(raw) == old_slot) else { continue };
            if target.is_full() {
                // Unreachable while shedding keeps the copy's room reserved
                // (`CodeLogs::shed_room`); report rather than stall silently.
                logs.advance(position);
                return Err(failure("code log compaction target full"));
            }
            let row = old.row(index).ok_or_else(|| failure("code log row absent during compaction"))?;
            words.clear();
            words.try_reserve(row.len()).map_err(|_| failure("code log compaction row allocation"))?;
            words.extend(row.iter().map(|v| v.to_bits()));
            // Complete in the new log before any reader can follow the tag.
            let new = target.append(&words)?;
            owner.store(log_tag(slot, new)?, Release);
            target.tagged();
            old.untagged();
            moved += 1;
        }
        if end >= directory.current.len() {
            drop(directory);
            logs.finish_compaction(self.graph.control.domain.current_generation());
            // The next commit saves a checkpoint naming only the new log.
            self.graph.control.note_mutation();
            return Ok((moved, true));
        }
        logs.advance(end);
        Ok((moved, false))
    }

    /// Test seam: compact code logs from this many rows up.
    #[cfg(test)]
    pub(crate) fn set_code_log_compact_min_rows(&self, rows: u64) {
        self.compact_min_rows.store(rows, Relaxed);
    }

    /// Test seam: the active code log holds at most `rows` rows.
    #[cfg(test)]
    pub(crate) fn set_code_log_capacity(&self, rows: u64) {
        if let Some(logs) = &self.log { logs.active().set_test_capacity(rows); }
    }

    /// Test seam: (id, rows, live rows) of the active code log and whether
    /// a compaction is still copying rows.
    #[cfg(test)]
    pub(crate) fn code_log_state(&self) -> Option<(u64, u64, u64, bool)> {
        let logs = self.log.as_ref()?;
        let log = logs.active();
        Some((log.id(), log.rows(), log.live_rows(), logs.compacting().is_some()))
    }

    /// Code log files this owner wrote or restored (each may be referenced
    /// by a commit, or about to be).
    pub fn code_log_files(&self) -> Vec<String> {
        self.log.as_ref().map_or_else(Vec::new, CodeLogs::persisted_files)
    }

    /// Diagnostic breakdown of this owner's resident storage (row store,
    /// shedding backlog, directory, planes, code log), for
    /// `RUSTIC_VECTOR_MEMORY_DIAGNOSTIC` (see `MutableIngestField::drain_some`).
    pub(crate) fn memory_diagnostic(&self) -> String {
        let rows = self.rows.stats();
        let unshed = self.unshed.lock().len();
        let (slots, retired) = { let directory = self.directory.lock(); (directory.current.len(), directory.retired.len()) };
        let (log_anon, log_mapped) = self.code_log_memory();
        format!("rows={rows:?} unshed={unshed} directory_slots={slots} retired_directories={retired} plane_bytes={} row_plane_bytes={} \
            log_anon={log_anon} log_mapped={log_mapped} log_state={:?} quantizer_bytes={}",
            self.plane.as_ref().map_or(0, |plane| plane.bytes()), self.row_plane.as_ref().map_or(0, |plane| plane.bytes()),
            self.log.as_ref().map(|logs| { let log = logs.active(); (log.id(), log.rows(), log.live_rows(), logs.compacting().is_some()) }),
            self.quantized.as_ref().map_or(0, QuantizedOwner::memory_bytes))
    }

    /// Code log (anonymous bytes, mapped file bytes).
    pub fn code_log_memory(&self) -> (usize, usize) {
        self.log.as_ref().map_or((0, 0), CodeLogs::memory_bytes)
    }

    /// Shed the in-memory f32 of up to `max` quantized rows whose f32 is
    /// durable ([`RawRows`]), oldest first: they keep codes and corrections
    /// only. Rows whose f32 is not durable yet go to the back of the queue,
    /// so one row that stays unsealed (or never becomes durable) does not
    /// stall shedding of every row queued behind it; a call examines a
    /// bounded number of rows. Returns how many rows shed.
    pub fn shed_some(&self, max: usize) -> Result<usize> {
        if self.quantized.is_none() && !shed_f32_rows() { return Ok(0); }
        let mut shed = 0;
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        let read = self.pin()?;
        let dim = self.graph.dim as usize;
        let mut budget = max.saturating_mul(2).max(SHED_SCAN_MIN).min(self.unshed.lock().len());
        let deferred_bytes=budget.checked_mul(std::mem::size_of::<(NodeHandle,u64)>()).ok_or_else(||failure("source deferred capacity overflow"))?;
        let _deferred_memory=crate::admission::shared().reserve(deferred_bytes as u64,"source shedding deferred rows")?;
        let mut deferred = Vec::new();
        deferred.try_reserve_exact(budget).map_err(|_|failure("source deferred rows allocation"))?;
        // Below write version 13 no checkpoint can reference code log rows:
        // every graph save is format 1 and writes each shed row in full
        // (its f32 read back from the store), so a log row would only add
        // chunk files no commit lists. Shed rows stay code-only in the row
        // store until every member reads format 2.
        let into_log = crate::codec::write_version() >= crate::codec::RMVC2_SINCE;
        while shed < max && budget > 0 {
            // A full code log is backpressure, not a failure: rows keep their
            // f32 until compaction (at the next seal) frees room.
            // (While compacting, room for the rows still to copy stays
            // reserved in the target: see `CodeLogs::shed_room`.)
            if into_log && self.log.as_ref().is_some_and(|logs| !logs.shed_room()) { break; }
            budget -= 1;
            let Some((handle, key)) = self.unshed.lock().pop_front() else { break };
            let Some(row) = read.encoded_node(handle, key) else { continue }; // released
            if is_code_only(row) { continue; }
            let durable=match read.raw.row_result(self,key) {
                Ok(row)=>row,
                Err(error)=>{let mut queue=self.unshed.lock();queue.push_front((handle,key));queue.extend(deferred);return Err(error);}
            };
            let Some(raw) = durable else {
                // Not durable yet (unsealed): retry on a later round.
                deferred.push((handle, key));
                continue;
            };
            if raw.len() != dim || raw.iter().zip(&row[8..8 + dim]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                continue; // a different durable row: keep the in-memory f32
            }
            let epoch = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
            let result=(||->Result<()> {
                if self.quantized.is_none() {
                    let values=raw_only_row(key,handle)?;
                    return self.replace_row(handle,key,&read,&values);
                }
                let codes=&row[codes_start(row,dim)..];
                let _codes_memory=restore_words(codes.len(),"source shedding code copy")?;
                let mut words=Vec::new();words.try_reserve_exact(codes.len()).map_err(|_|failure("source shedding code allocation"))?;
                words.extend(codes.iter().map(|v|v.to_bits()));
                let values=code_only_row(key,handle,epoch,&words)?;
                match self.log.as_ref().filter(|_| into_log) {
                    Some(logs) => self.move_row_to_log(logs,handle,key,&read,&values),
                    None => self.replace_row(handle,key,&read,&values),
                }
            })();
            if let Err(error)=result {
                let mut queue=self.unshed.lock();
                queue.push_front((handle,key));queue.extend(deferred);
                return Err(error);
            }
            shed += 1;
        }
        self.unshed.lock().extend(deferred);
        drop(read);
        if shed > 0 { self.commit_locked(&mut writer)?; }
        Ok(shed)
    }

    /// Test seam: move the newest queued row to the front of the shed
    /// queue (an unsealed row ahead of durable ones).
    #[cfg(test)]
    pub(crate) fn unshed_newest_first(&self) {
        let mut queue = self.unshed.lock();
        if let Some(newest) = queue.pop_back() { queue.push_front(newest); }
    }

    /// Bring back the in-memory f32 of a code-only row whose durable f32 is
    /// about to go (its document deleted and merged away while the graph
    /// still holds the node, pending repair).
    pub fn unshed(&self, handle: NodeHandle, key: u64, raw: &[f32]) -> Result<()> {
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        let read = self.pin()?;
        let Some(row) = read.encoded_node(handle, key) else { return Ok(()) };
        if !is_code_only(row) { return Ok(()); }
        if raw.len() != self.graph.dim as usize { return Err(failure("unshed row dimension")); }
        let width=raw.len().div_ceil(8).checked_mul(8).and_then(|v|v.checked_add(8))
            .and_then(|v|v.checked_add(row.len().saturating_sub(8)))
            .ok_or_else(||failure("unshed row size overflow"))?;
        let _memory=restore_words(width,"unshed row reconstruction workspace")?;
        let mut values = Vec::new();
        values.try_reserve_exact(width).map_err(|_|failure("unshed row allocation"))?;
        encode_row(&mut values, raw, key, handle)?;
        values[3] = row[3].to_bits();
        values[4] = row[4].to_bits();
        values.extend(row[8..].iter().map(|v| v.to_bits()));
        self.replace_row(handle, key, &read, &values)?;
        drop(read);
        self.commit_locked(&mut writer)
    }

    /// A source directory acquire observes reservation identity initialization.
    /// That identity is immutable until both graph generations and the source
    /// reclamation acknowledgment permit reuse. Growth copies it unchanged.
    /// This also covers Reserved and Retired nodes: source preparation precedes
    /// graph publication, and retained readers may outlive logical removal.
    #[inline]
    fn identity_matches(&self, guard: &GenerationGuard, handle: NodeHandle, key: u64) -> bool {
        let Ok(table) = self.graph.control.nodes.read(guard) else { return false; };
        let Some(record) = table.get(internal_id(handle)) else { return false; };
        record.incarnation_after_acquire() == handle.generation()
            && record.source_key_after_acquire() == key
    }

    /// Collect graph acknowledgments and retain them across cleanup failures.
    /// Retrying this method finishes pending cleanup before permitting slot
    /// reuse. In particular, returning an allocation error must not drop the
    /// token while its source directory entry is still installed.
    pub fn reclaim(&self) -> Result<usize> {
        self.reclaim_some(usize::MAX)
    }

    /// [`Self::reclaim`], releasing at most `max` rows; the rest stay
    /// queued (owned) for a later call. Bounds the work a write path pays.
    pub fn reclaim_some(&self, max: usize) -> Result<usize> {
        let mut pending = self.pending_reclamation.lock();
        pending.extend(self.graph.collect_reclaimed());
        let mut released = 0;
        while released < max {
            let Some(token) = pending.last() else { break };
            self.release(token)?;
            drop(pending.pop()); // acknowledgment only after source publication
            released += 1;
        }
        Ok(released)
    }

    /// Acknowledgments collected but not released yet.
    #[cfg(test)]
    pub(crate) fn pending_reclamations(&self) -> usize {
        self.pending_reclamation.lock().len()
    }

    /// Remove only after graph readers/preparations have drained. The caller
    /// keeps this token on failure and drops it AFTER success returns, allowing
    /// graph slot reuse only after the source owner has been cleared.
    pub fn release(&self, reclaimed: &ReclaimedNode) -> Result<()> {
        if !self.graph.owns_reclaimed(reclaimed) { return Err(failure("foreign source reclamation")); }
        let mut writer = self.graph.control.writer.lock();
        self.graph.control.admit_write(&mut writer)?;
        let directory = self.directory.lock();
        let slot = internal_id(reclaimed.handle()) as usize;
        let Some(owner) = directory.current.get(slot) else { return Ok(()) }; // cancelled before installation
        let reference = LevelsRef::from_raw(owner.load(Acquire));
        if !reference.is_valid() { return Ok(()); }
        let read = self.pin()?;
        if read.encoded_node(reclaimed.handle(), reclaimed.key()).is_none() {
            return Err(failure("source reclamation identity mismatch"));
        }
        if reference.raw() == PLANE_REF {
            // No graph reader reaches this slot's node any more, but a reader
            // pinned before now may hold its row: the entry is rewritten only
            // after such readers drain (`PlaneQuarantine`).
            let mut quarantine = self.quarantine.lock();
            quarantine.released.try_reserve(1).map_err(|_| failure("row plane quarantine"))?;
            quarantine.released.insert(slot as u32, self.graph.control.domain.current_generation());
            owner.store(0, Release);
            return Ok(());
        }
        if log_index(reference.raw()).is_some() {
            // The log row stays (append-only) and turns dead; only the slot
            // is released. Compaction later drops dead rows.
            if let Some(quantized) = &self.quantized {
                let row = read.encoded_node(reclaimed.handle(), reclaimed.key()).ok_or_else(|| failure("reclaimed encoded source absent"))?;
                quantized.released(u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32))?;
            }
            // Reclaimed: no reader reaches this slot's node any more.
            if let (Some(plane), Ok(id)) = (&self.plane, u32::try_from(slot)) { plane.clear(id); }
            owner.store(0, Release);
            if let Some(log) = self.log.as_ref().and_then(|logs| logs.get(log_slot(reference.raw()))) { log.untagged(); }
            return Ok(());
        }
        let view = self.rows.view(&read.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let token = unsafe { view.retirement_token(reference) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = self.rows.writer().stage_batch(0, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        if let Some(quantized) = &self.quantized {
            let row = read.encoded_node(reclaimed.handle(), reclaimed.key()).ok_or_else(|| failure("reclaimed encoded source absent"))?;
            quantized.released(u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32))?;
        }
        let batch = batch.begin_publish();
        if let (Some(plane), Ok(id)) = (&self.plane, u32::try_from(slot)) { plane.clear(id); }
        owner.store(0, Release);
        unsafe { batch.finish_after_publish() };
        Ok(())
    }

    /// Assign source holds before advancing the common generation. This is a
    /// commit-boundary operation, not a lookup or a per-vector distance cost.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.graph.control.writer.lock();
        self.commit_locked(&mut writer)
    }

    /// Test seam: node `handle`'s code words (corrections and codes, as
    /// scoring reads them) and their epoch.
    #[cfg(test)]
    pub(crate) fn code_words_for_test(&self, handle: NodeHandle, key: u64) -> Option<(u64, Vec<f32>)> {
        let read = self.pin().ok()?;
        let row = read.encoded_node(handle, key)?;
        let epoch = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
        Some((epoch, row[codes_start(row, self.graph.dim as usize)..].to_vec()))
    }

    /// Test seam: a scorer of `query` for codes of `epoch`.
    #[cfg(test)]
    pub(crate) fn code_scorer_for_test(&self, epoch: u64, query: &[f32]) -> Option<crate::vector::es_quant::EsQuantScorer> {
        self.quantized.as_ref()?.scorer_for_test(epoch, query)
    }

    fn commit_locked(&self, writer: &mut Writer) -> Result<()> {
        if self.quarantined.load(Relaxed) != 0 { self.move_quarantined_rows()?; }
        self.rows.writer().assign_generation().map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.graph.control.finish_mutation(writer);
        let oldest = self.graph.control.domain.update_oldest_used_generation();
        self.rows.writer().reclaim();
        self.directory.lock().retired.retain(|(generation, _)| *generation >= oldest);
        if let Some(logs) = &self.log { logs.reclaim(oldest); }
        if let Some(quantized) = &self.quantized { quantized.collect(); }
        Ok(())
    }
}

impl MutableVectorSource for MutableVectorRead<'_> {
    fn execution_result(&self)->Result<()> {self.raw.execution_result()?;self.decoded.execution_result()}
    fn get(&self, _: u64) -> Option<&[f32]> { None }
    fn code_row(&self, handle: NodeHandle, key: u64) -> Option<(u64, &[f32])> {
        if self.owner.quantized.is_none() || self.owner.exact_mutation_rows { return None; }
        let row = self.encoded_node(handle, key)?;
        let epoch = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
        Some((epoch, row.get(codes_start(row, self.owner.graph.dim as usize)..)?))
    }
    fn code_scorer(&self, epoch: u64) -> Option<Box<dyn Fn(&[f32], &[f32]) -> f64 + '_>> {
        code_pair_scorer(self.owner.quantized.as_ref()?, epoch, self.owner.graph.dim as usize)
    }
    fn get_node_approx(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let Some(quantized) = self.owner.quantized.as_ref().filter(|_| !self.owner.exact_mutation_rows) else { return self.get_node(handle, key) };
        let row = self.encoded_node(handle, key)?;
        approx_row(quantized, &self.decoded, row, self.owner.graph.dim as usize, handle)
    }
    fn get_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let row = self.encoded_node(handle, key)?;
        if is_code_only(row) {
            return self.raw.row(self.owner, key).filter(|raw| raw.len() == self.owner.graph.dim as usize);
        }
        Some(&row[8..8 + self.owner.graph.dim as usize])
    }
    fn bind_query(&self, query: &[f32]) -> Option<Box<dyn MutableQueryScorer + '_>> {
        let owner = self.owner.quantized.as_ref()?;
        let inner = quantized::BoundCodes { owner, query: query.to_vec(),
            row: move |handle, key| self.encoded_node(handle, key), cache: Default::default(), shared: None };
        // The live graph's traversal reads codes by node id (the plane is
        // written for every node this read can acquire).
        match self.owner.plane.as_deref().map(|plane| (plane, plane.stride())) {
            Some((plane, stride)) if stride != 0 => Some(Box::new(quantized::PlaneCodes { inner, plane, stride })),
            _ => Some(Box::new(inner)),
        }
    }
    fn bind_query_shared<'s>(&'s self, query: &[f32], shared: &'s SharedQuantized) -> Option<Box<dyn MutableQueryScorer + 's>> {
        let owner = self.owner.quantized.as_ref()?;
        let inner = quantized::BoundCodes { owner, query: query.to_vec(),
            row: move |handle, key| self.encoded_node(handle, key), cache: Default::default(), shared: Some(shared) };
        match self.owner.plane.as_deref().map(|plane| (plane, plane.stride())) {
            Some((plane, stride)) if stride != 0 => Some(Box::new(quantized::PlaneCodes { inner, plane, stride })),
            _ => Some(Box::new(inner)),
        }
    }

    fn dense_layout(&self) -> Option<DenseLayout<'_>> {
        if let Some(plane) = self.owner.plane.as_deref().filter(|plane| plane.stride() != 0) {
            return self.owner.quantized.is_some().then(|| DenseLayout::Codes(plane.view()));
        }
        // Rows by id only while the plane holds every row (no quarantine).
        if self.owner.quarantined.load(Acquire) != 0 { return None; }
        self.owner.row_plane.as_deref().map(|plane| DenseLayout::Rows(plane.view()))
    }

    #[inline]
    fn prefetch_node(&self, handle: NodeHandle, _: u64) {
        let slot = internal_id(handle) as usize;
        if slot >= self.owner.len.load(Acquire) { return; }
        let pointer = self.owner.pointer.load(Acquire);
        // The pinned generation retains this directory through growth.
        prefetch(unsafe { pointer.add(slot) }.cast::<u8>(), 4);
    }

    #[inline]
    fn prefetch_slot(&self, id: u32) {
        let slot = id as usize;
        if slot >= self.owner.len.load(Acquire) { return; }
        let pointer = self.owner.pointer.load(Acquire);
        // The directory entry; the row needs it loaded first.
        prefetch(unsafe { pointer.add(slot) }.cast::<u8>(), 4);
    }

    #[inline]
    fn prefetch_row(&self, handle: NodeHandle, _: u64) {
        let slot = internal_id(handle) as usize;
        if slot >= self.owner.len.load(Acquire) { return; }
        let pointer = self.owner.pointer.load(Acquire);
        // Same guard/reference contract as encoded_node; only the address is
        // used, the row is not read here.
        let reference = LevelsRef::from_raw(unsafe { &*pointer.add(slot) }.load(Acquire));
        if !reference.is_valid() { return; }
        if reference.raw() == PLANE_REF {
            if let Some(plane) = &self.owner.row_plane { plane.view().prefetch(slot as u32, self.owner.graph.dim as usize * 4); }
            return;
        }
        let values = if let Some(index) = log_index(reference.raw()) {
            // Code-log rows: `row` only computes the address.
            self.owner.log.as_ref().and_then(|logs| logs.get(log_slot(reference.raw()))?.row(index))
        } else {
            let view = unsafe { self.owner.rows.view_prevalidated(&self.guard) };
            unsafe { view.get(reference) }.ok()
        };
        if let Some(values) = values {
            prefetch(values.as_ptr().cast::<u8>(), (values.len() * 4).min(512));
        }
    }

}
impl MutableVectorRead<'_> {
    #[inline]
    fn encoded_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let slot = internal_id(handle) as usize;
        if slot >= self.owner.len.load(Acquire) { return None; }
        let pointer = self.owner.pointer.load(Acquire);
        // Length follows pointer publication; this guard retains displaced
        // directories and acquired row allocations through relocation/reuse.
        let reference = LevelsRef::from_raw(unsafe { &*pointer.add(slot) }.load(Acquire));
        if !reference.is_valid() { return None; }
        if reference.raw() == PLANE_REF {
            let values = self.owner.row_plane.as_ref()?.entry(slot as u32)?;
            decode_row(values, self.owner.graph.dim as usize, handle, key)?;
            return Some(values);
        }
        if let Some(index) = log_index(reference.raw()) {
            // The pinned guard keeps the log row's buffer/mapping alive (and
            // the log itself through a compaction switch).
            let values = self.owner.log.as_ref()?.get(log_slot(reference.raw()))?.row(index)?;
            decode_row(values, self.owner.graph.dim as usize, handle, key)?;
            return Some(values);
        }
        // SAFETY: pin/capture validates this exact owner/guard pair once;
        // private fields prevent constructing a source with a foreign guard.
        let view = unsafe { self.owner.rows.view_prevalidated(&self.guard) };
        let values = unsafe { view.get(reference) }.ok()?;
        decode_row(values, self.owner.graph.dim as usize, handle, key)?;
        Some(values)
    }

}

impl MutableVectorSource for CapturedMutableVectorRead {
    fn execution_result(&self)->Result<()> {self.raw.execution_result()?;self.decoded.execution_result()}
    fn get(&self, _: u64) -> Option<&[f32]> { None }
    fn get_node_approx(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let Some(quantized) = self.owner.quantized.as_ref().filter(|_| !self.owner.exact_mutation_rows) else { return self.get_node(handle, key) };
        let row = self.encoded_node(handle, key)?;
        approx_row(quantized, &self.decoded, row, self.owner.graph.dim as usize, handle)
    }
    fn get_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let row = self.encoded_node(handle, key)?;
        if is_code_only(row) {
            return self.raw.row(&self.owner, key).filter(|raw| raw.len() == self.owner.graph.dim as usize);
        }
        Some(&row[8..8 + self.owner.graph.dim as usize])
    }
    fn bind_query(&self, query: &[f32]) -> Option<Box<dyn MutableQueryScorer + '_>> {
        let owner = self.owner.quantized.as_ref()?;
        Some(Box::new(quantized::BoundCodes { owner, query: query.to_vec(),
            row: move |handle, key| self.encoded_node(handle, key), cache: Default::default(), shared: None }))
    }

    #[inline]
    fn prefetch_node(&self, handle: NodeHandle, _: u64) {
        if let Some(reference) = self.references.get(internal_id(handle) as usize) {
            prefetch((reference as *const LevelsRef).cast::<u8>(), 4);
        }
    }

    #[inline]
    fn prefetch_row(&self, handle: NodeHandle, _: u64) {
        let Some(&reference) = self.references.get(internal_id(handle) as usize) else { return };
        if !reference.is_valid() { return; }
        if reference.raw() == PLANE_REF {
            if let Some(plane) = &self.owner.row_plane { plane.view().prefetch(internal_id(handle), self.owner.graph.dim as usize * 4); }
            return;
        }
        // Same guard/reference contract as encoded_node; address only.
        let values = if let Some(index) = log_index(reference.raw()) {
            self.owner.log.as_ref().and_then(|logs| logs.get(log_slot(reference.raw()))?.row(index))
        } else {
            let view = unsafe { self.owner.rows.view_prevalidated(&self.guard) };
            unsafe { view.get(reference) }.ok()
        };
        if let Some(values) = values {
            prefetch(values.as_ptr().cast::<u8>(), (values.len() * 4).min(512));
        }
    }
}
impl CapturedMutableVectorRead {
    /// Heap bytes of the capture's own copy of the row references.
    pub fn captured_bytes(&self) -> usize {
        self.references.capacity() * std::mem::size_of::<LevelsRef>()
    }

    /// Bytes of rows this capture decoded from codes and keeps for its
    /// lifetime (released with the capture).
    pub fn decoded_bytes(&self) -> usize {
        usize::try_from(self.decoded.held_bytes()).unwrap_or(usize::MAX)
    }

    /// The owner's [`MutableVectorOwner::score_quantizer`] as of the capture.
    pub(crate) fn score_quantizer(&self) -> Option<crate::vector::es_quant::EsQuantizer> {
        self.score.clone()
    }

    /// Read the f32 of code-only rows from `raw` instead of the owner's
    /// (live) rows: a retained view's own store. Effective before the
    /// first such read only.
    pub fn bind_raw(&self, raw: &dyn RawRows) {
        let _ = self.raw.0.set(raw.snapshot());
    }

    #[inline]
    fn encoded_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let reference = *self.references.get(internal_id(handle) as usize)?;
        if !reference.is_valid() { return None; }
        if let Some(index) = log_index(reference.raw()) {
            let values = self.owner.log.as_ref()?.get(log_slot(reference.raw()))?.row(index)?;
            decode_row(values, self.owner.graph.dim as usize, handle, key)?;
            return Some(values);
        }
        if reference.raw() == PLANE_REF {
            // The capture's guard keeps the slot from reuse: its entry stays.
            let values = self.owner.row_plane.as_ref()?.entry(internal_id(handle))?;
            decode_row(values, self.owner.graph.dim as usize, handle, key)?;
            return Some(values);
        }
        // SAFETY: pin/capture validates this exact owner/guard pair once;
        // private fields prevent constructing a source with a foreign guard.
        let view = unsafe { self.owner.rows.view_prevalidated(&self.guard) };
        let values = unsafe { view.get(reference) }.ok()?;
        decode_row(values, self.owner.graph.dim as usize, handle, key)?;
        Some(values)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn atomic_checkpoint_file_roundtrip_and_failed_output_preserve_previous_commit() {
        use crate::codec::directory::{Directory, FileSlice, OutputFile, RamDirectory};
        #[derive(Debug)]
        struct FailingDirectory<'a>(&'a RamDirectory, usize);
        struct FailingOutput(usize);
        impl OutputFile for FailingOutput {
            fn write_all(&mut self, _: &[u8]) -> Result<()> {
                if self.0 == 0 { return Err(failure("injected checkpoint write failure")); }
                self.0 -= 1;
                Ok(())
            }
            fn finish(self: Box<Self>) -> Result<()> { panic!("failed output must not publish") }
        }
        impl Directory for FailingDirectory<'_> {
            fn open_read(&self, n: &str) -> Result<FileSlice> { self.0.open_read(n) }
            fn write(&self, _: &str, _: &[u8]) -> Result<()> { panic!("must stream") }
            fn exists(&self, n: &str) -> bool { self.0.exists(n) }
            fn delete(&self, n: &str) -> Result<()> { self.0.delete(n) }
            fn list(&self) -> Result<Vec<String>> { self.0.list() }
            fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.0.list_admitted(temporary) }
            fn create_output<'a>(&'a self, _: &str) -> Result<Box<dyn OutputFile + 'a>> { Ok(Box::new(FailingOutput(self.1))) }
        }
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let reservation = graph.reserve_insert(7).unwrap();
        owner.install(&reservation, &[3., 4.]).unwrap();
        let read = owner.pin().unwrap();
        let handle = graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
        drop(read);
        let checkpoint = owner.capture_checkpoint().unwrap();
        let directory = RamDirectory::new();
        checkpoint.write_file(&directory, "vector-checkpoint").unwrap();
        let original = directory.open_read("vector-checkpoint").unwrap();
        let (restored, rows) = MutableVectorOwner::open_file(&directory, "vector-checkpoint").unwrap();
        assert_eq!(restored.encode(), checkpoint.graph_bytes().unwrap());
        assert_eq!(rows.pin().unwrap().get_node(handle, 7), Some([3., 4.].as_slice()));
        for fail_after in 0..6 {
            assert!(checkpoint.write_file(&FailingDirectory(&directory, fail_after), "vector-checkpoint").is_err());
            assert_eq!(directory.open_read("vector-checkpoint").unwrap().as_slice(), original.as_slice());
        }
        for n in [0, 7, 15, original.len() - 1] {
            directory.write("bad", &original.as_slice()[..n]).unwrap();
            assert!(MutableVectorOwner::open_file(&directory, "bad").is_err());
        }
        let mut damaged = original.as_slice().to_vec();
        damaged[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        directory.write("bad", &damaged).unwrap();
        assert!(MutableVectorOwner::open_file(&directory, "bad").is_err());
        let mut damaged = original.as_slice().to_vec();
        damaged.push(0);
        directory.write("bad", &damaged).unwrap();
        assert!(MutableVectorOwner::open_file(&directory, "bad").is_err());
    }

    /// The graph section is streamed in pieces (`GRAPH_STREAM_CHUNK` is
    /// 96 bytes under test): the file holds exactly the bytes the in-memory
    /// encoding does, and reopens to the same graph and rows.
    #[test]
    fn streamed_graph_checkpoint_matches_the_in_memory_encoding() {
        let graph = Arc::new(MutableHnsw::new(4, Similarity::L2, HnswParams { m: 8, ef_construction: 32, ..HnswParams::default() }));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let row = |k: u64| [k as f32, (k % 7) as f32, (k % 13) as f32 * 0.5, 1.0];
        let mut handles = Vec::new();
        for k in 0..400u64 {
            let reservation = graph.reserve_insert(k).unwrap();
            owner.install(&reservation, &row(k)).unwrap();
            let read = owner.pin().unwrap();
            handles.push(graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap());
        }
        let checkpoint = owner.capture_checkpoint().unwrap();
        use crate::codec::directory::Directory as _;
        let directory = crate::codec::directory::RamDirectory::new();
        checkpoint.write_file(&directory, "streamed").unwrap();
        let file = directory.open_read("streamed").unwrap();
        let file = file.as_slice();
        let graph_bytes = checkpoint.graph_bytes().unwrap();
        assert!(graph_bytes.len() > 10 * GRAPH_STREAM_CHUNK, "the graph spans many pieces");
        assert_eq!(&file[8..16], (graph_bytes.len() as u64).to_le_bytes().as_slice());
        assert_eq!(&file[16..16 + graph_bytes.len()], graph_bytes);
        // Written again with the encoding cached: the same file.
        checkpoint.write_file(&directory, "cached").unwrap();
        assert_eq!(directory.open_read("cached").unwrap().as_slice(), file);
        let (restored, rows) = MutableVectorOwner::open_file(&directory, "streamed").unwrap();
        assert_eq!(restored.encode(), graph_bytes);
        let read = rows.pin().unwrap();
        for k in [0u64, 199, 399] { assert_eq!(read.get_node(handles[k as usize], k), Some(row(k).as_slice()), "row {k}"); }
    }

    /// A checkpoint's header and rows follow the version its writer
    /// captured, not the live write version (the cluster floor rising
    /// mid-write must not put v3/v4 rows under an older header).
    #[test]
    fn checkpoint_file_follows_the_captured_write_version() {
        use crate::codec::directory::{Directory, RamDirectory};
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let reservation = graph.reserve_insert(7).unwrap();
        owner.install(&reservation, &[3., 4.]).unwrap();
        let read = owner.pin().unwrap();
        let handle = graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
        drop(read);
        let checkpoint = owner.capture_checkpoint().unwrap();
        let rows_header = |bytes: &[u8]| {
            let graph_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            bytes[16 + graph_len..16 + graph_len + 8].to_vec()
        };
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
        let directory = RamDirectory::new();
        assert!(checkpoint.write_file_versioned(&directory, "old", crate::codec::MIN_WRITE_VERSION).unwrap().is_empty());
        let old = directory.open_read("old").unwrap();
        assert_eq!(&old.as_slice()[..8], b"RMVC\x01\0\0\0");
        assert_eq!(rows_header(old.as_slice()), b"RMVR\x01\0\0\0", "v11 rows under a v11 header");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::MIN_WRITE_VERSION));
        checkpoint.write_file_versioned(&directory, "v12", crate::codec::RMVR3_SINCE).unwrap();
        let v12 = directory.open_read("v12").unwrap();
        assert_eq!(&v12.as_slice()[..8], b"RMVC\x01\0\0\0");
        assert_eq!(rows_header(v12.as_slice()), b"RMVR\x03\0\0\0");
        checkpoint.write_file_versioned(&directory, "v13", crate::codec::RMVC2_SINCE).unwrap();
        assert_eq!(&directory.open_read("v13").unwrap().as_slice()[..8], b"RMVC\x02\0\0\0");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        let (_, rows) = MutableVectorOwner::open_file(&directory, "old").unwrap();
        assert_eq!(rows.pin().unwrap().get_node(handle, 7), Some([3., 4.].as_slice()));
    }

    /// A 1-bit (bbq) field's graph stays connected: diversity selection
    /// weighs candidates on one distance scale (the regression: almost every
    /// candidate after the first was rejected, nodes kept one or two links
    /// and, after reverse-link pruning, some none).
    #[test]
    fn one_bit_graph_stays_connected() {
        use crate::vector::es_quant::EsQuant;
        const DIMS: usize = 64;
        let graph = Arc::new(MutableHnsw::new(DIMS as u32, Similarity::DotProduct, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new_quantized(graph.clone(), Some(EsQuant { bits: 1, osq: true, confidence_interval: None })).unwrap());
        let vectors: Vec<Vec<f32>> = (0..1200usize).map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let c = (i % 17) as f32;
            let v: Vec<f32> = (0..DIMS).map(|d| {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                (d as f32 * 0.31 + c).sin() + 0.3 * ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            }).collect();
            let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.into_iter().map(|a| a / n).collect()
        }).collect();
        let refs: Vec<_> = vectors.iter().map(Vec::as_slice).collect();
        let codes = owner.prepare_codes(&refs).unwrap();
        for (key, (raw, code)) in vectors.iter().zip(&codes).enumerate() {
            let reservation = graph.reserve_insert(key as u64).unwrap();
            owner.install_prepared(&reservation, raw, code.as_ref()).unwrap();
            let read = owner.pin().unwrap();
            graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
        }
        let view = graph.snapshot();
        let (mut nodes, mut links) = (0usize, 0usize);
        for id in 1..view.len {
            if let Some(n) = view.current_node(id) { nodes += 1; links += n.links(0).map_or(0, <[u32]>::len); }
        }
        let mut seen = vec![false; view.len as usize];
        let mut stack = vec![view.entry as u32];
        let mut reached = 0;
        while let Some(id) = stack.pop() {
            if id == 0 || id >= view.len || std::mem::replace(&mut seen[id as usize], true) { continue; }
            let Some(n) = view.current_node(id) else { continue };
            reached += 1;
            stack.extend_from_slice(n.links(0).unwrap_or(&[]));
        }
        eprintln!("one-bit graph: {nodes} nodes, mean level-0 degree {:.1}, {reached} reachable", links as f64 / nodes as f64);
        assert_eq!(nodes, 1200);
        assert!(links >= 10 * nodes, "mean level-0 degree {:.1}", links as f64 / nodes as f64);
        assert!(reached + 2 >= nodes, "{reached} of {nodes} nodes reachable at level 0");
    }

    #[test]
    fn quantized_owner_preserves_codes_through_capture_relocation_cancel_and_checkpoint() {
        use crate::vector::es_quant::EsQuant;
        for (bits, osq) in [(7, false), (4, false), (7, true), (4, true), (1, true)] {
            for similarity in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct, Similarity::MaxInnerProduct] {
                let graph = Arc::new(MutableHnsw::new(13, similarity, HnswParams::default()));
                let owner = Arc::new(MutableVectorOwner::new_quantized(graph.clone(), Some(EsQuant {
                    bits, osq, confidence_interval: None,
                })).unwrap());
                let vectors: Vec<Vec<f32>> = (0..48).map(|i| (0..13).map(|d|
                    ((i * 17 + d * 7) % 71) as f32 / 37.0 - 0.8).collect()).collect();
                let refs: Vec<_> = vectors.iter().map(Vec::as_slice).collect();
                let codes = owner.prepare_codes(&refs).unwrap();
                let mut handles = Vec::new();
                for (key, (raw, code)) in vectors.iter().zip(&codes).enumerate() {
                    let reservation = graph.reserve_insert(key as u64).unwrap();
                    owner.install_prepared(&reservation, raw, code.as_ref()).unwrap();
                    let read = owner.pin().unwrap();
                    handles.push(graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap());
                }
                let query = &vectors[7];
                let (captured_graph, captured) = owner.capture().unwrap();
                let read = owner.pin().unwrap();
                let scorer = read.bind_query(query).expect("encoded traversal must be bound");
                let expected: Vec<_> = handles.iter().enumerate().map(|(key, &handle)|
                    scorer.score(handle, key as u64).unwrap().to_bits()).collect();
                assert!(scorer.score(handles[0], 999).is_none(), "identity cannot cross point keys");
                let before = captured_graph.search(query, 10, 48, &captured, |key| key % 2 == 0).hits;
                drop(scorer); drop(read);
                owner.relocate(handles[7], 7, &vectors[7]).unwrap();
                {
                    let read = owner.pin().unwrap();
                    graph.remove(handles[0], &read).unwrap();
                    let reservation = graph.reserve_insert(1000).unwrap();
                    owner.install(&reservation, &vectors[2]).unwrap();
                    drop(reservation); // cancel after code publication
                }
                owner.commit().unwrap();
                owner.reclaim().unwrap();
                let captured_scorer = captured.bind_query(query).unwrap();
                for (key, &handle) in handles.iter().enumerate() {
                    assert_eq!(captured_scorer.score(handle, key as u64).unwrap().to_bits(), expected[key]);
                }
                assert_eq!(captured_graph.search(query, 10, 48, &captured, |key| key % 2 == 0).hits, before);
                drop(captured_scorer); drop(captured); drop(captured_graph);
                owner.commit().unwrap(); owner.reclaim().unwrap();
                let checkpoint = owner.capture_checkpoint().unwrap();
                let mut rows = Vec::new(); checkpoint.write_rows(&mut rows).unwrap();
                let (restored, restored_owner) = MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &rows).unwrap();
                let restored_read = restored_owner.pin().unwrap();
                let live = owner.pin().unwrap();
                for filter in [false, true] {
                    assert_eq!(graph.search(query, 10, 48, &live, |key| !filter || key % 2 == 0).hits,
                        restored.search(query, 10, 48, &restored_read, |key| !filter || key % 2 == 0).hits);
                }
                let scorer = restored_read.bind_query(query).unwrap();
                for key in 1..handles.len() { assert_eq!(scorer.score(handles[key], key as u64).unwrap().to_bits(), expected[key]); }
                let mut broken = rows.clone(); broken[32] ^= 1;
                assert!(MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &broken).is_err());
                for len in [0, 8, 27, rows.len()-1] {
                    assert!(MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &rows[..len]).is_err());
                }
            }
        }
    }

    #[test]
    fn owned_checkpoint_streams_stable_rows_after_live_graph_mutation() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let mut handles = Vec::new();
        for key in [3, 19, 100] {
            let reservation = graph.reserve_insert(key).unwrap();
            owner.install(&reservation, &[key as f32, -0.0]).unwrap();
            let read = owner.pin().unwrap();
            handles.push(graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap());
        }
        let read = owner.pin().unwrap();
        let expected = graph.search(&[3., 0.], 3, 10, &read, |_| true).hits;
        drop(read);
        let original_graph = graph.encode();
        let checkpoint = owner.capture_checkpoint().unwrap();
        {
            let read = owner.pin().unwrap();
            graph.remove(handles[0], &read).unwrap();
        }
        owner.commit().unwrap();
        assert_eq!(owner.reclaim().unwrap(), 0, "capture retains old row");
        let mut rows = Vec::new();
        checkpoint.write_rows(&mut rows).unwrap();
        assert_eq!(checkpoint.graph_bytes().unwrap(), original_graph);
        let (restored, restored_owner) = MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &rows).unwrap();
        let read = restored_owner.pin().unwrap();
        assert_eq!(restored.search(&[3., 0.], 3, 10, &read, |_| true).hits, expected);
        assert_eq!(read.get_node(handles[0], 3).unwrap()[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(restored.encode(), checkpoint.graph_bytes().unwrap());
        assert!(MutableVectorOwner::restore_checkpoint(&graph.encode(), &rows).is_err());
        let mut damaged = rows.clone();
        damaged[24] ^= 1;
        assert!(MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &damaged).is_err());
        // A recomputed checksum cannot hide a row/graph identity mismatch.
        let n = damaged.len() - 4;
        let checksum = crate::codec::crc32(&damaged[..n]);
        damaged[n..].copy_from_slice(&checksum.to_le_bytes());
        assert!(MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &damaged).is_err());
        for n in [0, 8, 27, rows.len() - 1] {
            assert!(MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &rows[..n]).is_err());
        }
        drop(checkpoint);
        assert_eq!(owner.reclaim().unwrap(), 1);
    }

    #[test]
    fn owned_checkpoint_leaves_pending_work_out_and_propagates_output_failure() {
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::Error::other("injected output failure")) }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let reservation = graph.reserve_insert(1).unwrap();
        // Graph ingestion in flight never blocks a capture: the pending node
        // is not in the captured graph, and its slot restores free.
        let pending = owner.capture_checkpoint().unwrap();
        let mut rows = Vec::new();
        pending.write_rows(&mut rows).unwrap();
        let (restored, _) = MutableVectorOwner::restore_checkpoint(pending.graph_bytes().unwrap(), &rows).unwrap();
        assert_eq!(restored.active_nodes(), 0);
        drop(pending);
        drop(reservation);
        let original_graph = graph.encode();
        let checkpoint = owner.capture_checkpoint().unwrap();
        let reservation = graph.reserve_insert(2).unwrap();
        owner.install(&reservation, &[2., 1.]).unwrap();
        let read = owner.pin().unwrap();
        graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
        drop(read);
        assert!(checkpoint.write_rows(&mut Broken).is_err());
        let mut rows = Vec::new();
        checkpoint.write_rows(&mut rows).unwrap();
        let (restored, _) = MutableVectorOwner::restore_checkpoint(checkpoint.graph_bytes().unwrap(), &rows).unwrap();
        assert_eq!(restored.active_nodes(), 0);
        assert_eq!(checkpoint.graph_bytes().unwrap(), original_graph, "later allocation cannot change captured empty slots");
    }

    use super::*;

    #[test]
    fn concurrent_relocation_never_hides_a_live_source() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let permit = graph.reserve_insert(1).unwrap();
        owner.install(&permit, &[3., 4.]).unwrap();
        let read = owner.pin().unwrap();
        let handle = graph.complete(graph.prepare(permit, &read).unwrap(), &read).unwrap();
        drop(read);
        let gate = std::sync::Barrier::new(5);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let owner = &owner;
                let gate = &gate;
                scope.spawn(move || {
                    let read = owner.pin().unwrap();
                    gate.wait();
                    for _ in 0..4000 {
                        assert_eq!(read.get_node(handle, 1), Some([3., 4.].as_slice()));
                    }
                });
            }
            gate.wait();
            for _ in 0..500 { owner.relocate(handle, 1, &[3., 4.]).unwrap(); }
        });
    }

    #[test]
    fn source_view_after_retirement_ack_does_not_rebind_captured_rows() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let permit = graph.reserve_insert(1).unwrap();
        owner.install(&permit, &[1., 2.]).unwrap();
        let read = owner.pin().unwrap();
        let old = graph.complete(graph.prepare(permit, &read).unwrap(), &read).unwrap();
        graph.remove(old, &read).unwrap();
        drop(read);
        let mut tokens = graph.collect_reclaimed();
        assert_eq!(tokens.len(), 1);
        // This reader did not prevent issuance of the reclamation token.
        let late = owner.pin().unwrap();
        let borrowed = late.get_node(old, 1).unwrap();
        let (_, captured) = owner.capture().unwrap();
        assert!(captured.get_node(old, 1).is_none()); // not in captured graph
        owner.release(&tokens[0]).unwrap();
        drop(tokens.pop());
        let next = graph.reserve_insert(2).unwrap();
        assert_eq!(next.handle().slot(), old.slot());
        owner.install(&next, &[9., 8.]).unwrap();
        assert_eq!(borrowed, &[1., 2.]);
        assert!(late.get_node(old, 1).is_none());
        assert_eq!(late.get_node(next.handle(), 2), Some([9., 8.].as_slice()));
        assert!(captured.get_node(next.handle(), 2).is_none());
    }

    /// A row installed over a plane slot released while an older reader may
    /// hold its row waits in the row store (rows by id stay off), and moves
    /// into the plane at a commit once that reader is gone.
    #[test]
    fn reused_plane_slot_waits_for_older_readers_then_moves_into_the_plane() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let permit = graph.reserve_insert(1).unwrap();
        owner.install(&permit, &[1., 2.]).unwrap();
        let read = owner.pin().unwrap();
        let old = graph.complete(graph.prepare(permit, &read).unwrap(), &read).unwrap();
        assert!(read.dense_layout().is_some());
        graph.remove(old, &read).unwrap();
        drop(read);
        let mut tokens = graph.collect_reclaimed();
        let late = owner.pin().unwrap();
        let borrowed = late.get_node(old, 1).unwrap();
        owner.release(&tokens[0]).unwrap();
        drop(tokens.pop());
        let next = graph.reserve_insert(2).unwrap();
        assert_eq!(next.handle().slot(), old.slot());
        owner.install(&next, &[9., 8.]).unwrap();
        assert_eq!(owner.quarantined.load(Relaxed), 1);
        assert!(owner.pin().unwrap().dense_layout().is_none(), "rows by id off while quarantined");
        owner.commit().unwrap();
        assert_eq!(owner.quarantined.load(Relaxed), 1, "the older reader still holds the old row");
        assert_eq!(borrowed, &[1., 2.]);
        drop(late);
        owner.commit().unwrap();
        owner.commit().unwrap();
        assert_eq!(owner.quarantined.load(Relaxed), 0);
        let read = owner.pin().unwrap();
        assert!(read.dense_layout().is_some());
        assert_eq!(read.get_node(next.handle(), 2), Some([9., 8.].as_slice()));
        let directory = owner.directory.lock();
        assert_eq!(directory.current[internal_id(next.handle()) as usize].load(Relaxed), PLANE_REF);
    }

    #[test]
    fn dense_source_checks_identity_before_publication_and_after_capture() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let reservation = graph.reserve_insert(42).unwrap();
        let handle = reservation.handle();
        owner.install(&reservation, &[3., 4.]).unwrap();
        let read = owner.pin().unwrap();
        assert_eq!(read.get_node(handle, 42), Some([3., 4.].as_slice()));
        assert!(read.get_node(handle, 43).is_none());
        assert!(read.get_node(NodeHandle::new(handle.slot(), handle.generation() + 1), 42).is_none());
        let handle = graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
        let (_, captured) = owner.capture().unwrap();
        assert!(captured.get_node(handle, 43).is_none());
        graph.remove(handle, &read).unwrap();
        owner.commit().unwrap();
        assert_eq!(captured.get_node(handle, 42), Some([3., 4.].as_slice()));
        assert_eq!(read.get_node(handle, 42), Some([3., 4.].as_slice()));
        assert_eq!(owner.reclaim().unwrap(), 0);
        drop(captured);
        drop(read);
        assert_eq!(owner.reclaim().unwrap(), 1);
    }

    #[test]
    fn owned_payload_alignment_survives_relocation_and_capture() {
        for dim in [2, 96, 4097] {
            let graph = Arc::new(MutableHnsw::new(dim, Similarity::L2, HnswParams::default()));
            let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
            let values = vec![0.25; dim as usize];
            let reservation = graph.reserve_insert(42).unwrap();
            owner.install(&reservation, &values).unwrap();
            let read = owner.pin().unwrap();
            let handle = graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap();
            let row = read.get_node(handle, 42).unwrap();
            assert_eq!(row.as_ptr() as usize % 32, 0);
            assert_eq!(row, values);
            let (_, captured) = owner.capture().unwrap();
            owner.relocate(handle, 42, &values).unwrap();
            let current = owner.pin().unwrap();
            for row in [current.get_node(handle, 42).unwrap(), captured.get_node(handle, 42).unwrap()] {
                assert_eq!(row.as_ptr() as usize % 32, 0);
                assert_eq!(row, values);
            }
        }
    }

    #[test]
    fn bounded_reclaim_releases_at_most_its_share_and_keeps_the_rest_queued() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let mut handles = Vec::new();
        for key in 0..40 {
            let reservation = graph.reserve_insert(key).unwrap();
            owner.install(&reservation, &[key as f32, 1.]).unwrap();
            let read = owner.pin().unwrap();
            handles.push(graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap());
        }
        for handle in &handles[..30] {
            let read = owner.pin().unwrap();
            graph.remove(*handle, &read).unwrap();
        }
        owner.commit().unwrap();
        assert_eq!(owner.reclaim_some(8).unwrap(), 8, "a write path pays for 8 rows only");
        assert_eq!(owner.pending_reclamations(), 22, "the rest stay queued and owned");
        assert_eq!(owner.reclaim_some(8).unwrap(), 8);
        assert_eq!(owner.reclaim().unwrap(), 14, "maintenance releases the backlog");
        assert_eq!(owner.pending_reclamations(), 0);
        let read = owner.pin().unwrap();
        for (key, handle) in handles.iter().enumerate() {
            assert_eq!(read.get_node(*handle, key as u64).is_some(), key >= 30, "key {key}");
        }
    }

    #[test]
    fn captured_graph_and_source_survive_removal_relocation_and_compaction() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = Arc::new(MutableVectorOwner::new(graph.clone()).unwrap());
        let mut handles = Vec::new();
        for key in 0..80 {
            let reservation = graph.reserve_insert(key).unwrap();
            owner.install(&reservation, &[key as f32, 1.]).unwrap();
            let read = owner.pin().unwrap();
            handles.push(graph.complete(graph.prepare(reservation, &read).unwrap(), &read).unwrap());
        }
        let (captured, source) = owner.capture().unwrap();
        let restored_graph = Arc::new(MutableHnsw::open(&graph.encode()).unwrap());
        let restored_owner = MutableVectorOwner::restore(restored_graph.clone(), &source).unwrap();
        let restored_source = restored_owner.pin().unwrap();
        assert_eq!(restored_graph.encode(), graph.encode());
        assert_eq!(restored_graph.search(&[37., 1.], 10, 80, &restored_source, |_| true).hits,
            captured.search(&[37., 1.], 10, 80, &source, |_| true).hits);
        let before = captured.search(&[37., 1.], 10, 80, &source, |_| true).hits;
        for key in 0..80 {
            let read = owner.pin().unwrap();
            graph.remove(handles[key], &read).unwrap();
        }
        owner.commit().unwrap();
        graph.compact_storage().unwrap();
        owner.compact_storage().unwrap();
        assert_eq!(owner.reclaim().unwrap(), 0);
        assert_eq!(captured.search(&[37., 1.], 10, 80, &source, |_| true).hits, before);
        assert_eq!(source.get_node(handles[37], 37), Some([37., 1.].as_slice()));
        drop(captured);
        drop(source);
        assert_eq!(owner.reclaim().unwrap(), 80);
    }

    #[test]
    fn pinned_source_follows_directory_growth_and_retains_borrowed_rows() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = MutableVectorOwner::new(graph.clone()).unwrap();
        let read = owner.pin().unwrap();
        let mut handles = Vec::new();
        for key in 0..80 {
            let reservation = graph.reserve_insert(key).unwrap();
            owner.install(&reservation, &[key as f32, 1.]).unwrap();
            let prepared = graph.prepare(reservation, &read).unwrap();
            handles.push(graph.complete(prepared, &read).unwrap());
        }
        let borrowed = read.get_node(handles[0], 0).unwrap();
        assert_eq!(borrowed, &[0., 1.]);
        let topology = graph.encode();
        assert!(owner.unchanged(handles[0], 0, &[0., 1.]).unwrap());
        assert!(!owner.unchanged(handles[0], 0, &[2., 1.]).unwrap());
        assert!(owner.relocate(handles[0], 0, &[2., 1.]).is_err());
        owner.relocate(handles[0], 0, &[0., 1.]).unwrap();
        owner.commit().unwrap();
        assert_eq!(graph.encode(), topology);
        assert_eq!(borrowed, &[0., 1.]);
        assert!(owner.memory_bytes().1 > 0);
        assert_eq!(read.get_node(handles[79], 79), Some([79., 1.].as_slice()));
        assert!(read.get_node(handles[79], 78).is_none());
        let result = graph.search(&[79., 1.], 1, 80, &read, |_| true);
        assert_eq!(result.hits[0].0, 79);
        graph.remove(handles[0], &read).unwrap();
        owner.commit().unwrap();
        assert!(graph.collect_reclaimed().is_empty());
        assert_eq!(borrowed, &[0., 1.]);
        drop(read);
        assert_eq!(owner.reclaim().unwrap(), 1);
        assert_eq!(owner.reclaim().unwrap(), 0);
        owner.commit().unwrap();
        let reservation = graph.reserve_insert(80).unwrap();
        assert_eq!(reservation.handle().slot(), handles[0].slot());
        let new_handle = reservation.handle();
        owner.install(&reservation, &[80., 1.]).unwrap();
        let current = owner.pin().unwrap();
        assert!(current.get_node(handles[0], 0).is_none());
        assert_eq!(current.get_node(new_handle, 80), Some([80., 1.].as_slice()));
        graph.complete(graph.prepare(reservation, &current).unwrap(), &current).unwrap();
    }

    #[test]
    fn cancelled_source_installation_releases_before_slot_reuse() {
        let graph = Arc::new(MutableHnsw::new(2, Similarity::L2, HnswParams::default()));
        let owner = MutableVectorOwner::new(graph.clone()).unwrap();
        let reservation = graph.reserve_insert(0).unwrap();
        let handle = reservation.handle();
        assert!(owner.install(&reservation, &[f32::NAN, 1.]).is_err());
        owner.install(&reservation, &[0., 1.]).unwrap();
        assert!(owner.install(&reservation, &[2., 1.]).is_err());
        drop(reservation);
        assert_eq!(owner.reclaim().unwrap(), 1);
        assert_eq!(owner.reclaim().unwrap(), 0);
        owner.commit().unwrap();
        let next = graph.reserve_insert(1).unwrap();
        assert_eq!(next.handle().slot(), handle.slot());
        owner.install(&next, &[1., 1.]).unwrap();
    }
}

// Immutable identity shares the beginning of the allocation with vector data,
// avoiding a dependent tail-line fetch. Keep a 32-byte prefix so vectors remain
// aligned. Unlike current node metadata, this identity cannot change while a
// late source reader still holds a previously acquired allocation.
fn reserve_output_row(row:&mut Vec<u8>,memory:&mut Option<crate::admission::Reservation>,needed:usize)->Result<()> {
    if row.capacity()>=needed{return Ok(())}
    let capacity=needed.max(row.capacity().saturating_mul(2));
    let next_memory=crate::admission::shared().reserve(capacity as u64,"checkpoint streaming row")?;
    let mut next=Vec::new();next.try_reserve_exact(capacity).map_err(|_|failure("checkpoint streaming row allocation"))?;
    // Every caller is beginning a new row; no previous bytes need copying.
    *row=next;*memory=Some(next_memory);Ok(())
}

fn encode_row(out: &mut Vec<u32>, vector: &[f32], key: u64, handle: NodeHandle) -> Result<()> {
    let width = vector.len().div_ceil(8) * 8 + 8;
    out.try_reserve_exact(width).map_err(|_| failure("source row allocation"))?;
    out.resize(8, 0);
    out[0] = key as u32;
    out[1] = (key >> 32) as u32;
    out[2] = handle.generation();
    out.extend(vector.iter().map(|v| v.to_bits()));
    out.resize(width, 0);
    Ok(())
}
/// Header word 5 of a code-only quantized row (identity header, then codes
/// and corrections; the f32 is durable in [`RawRows`]).
const CODE_ONLY: u32 = 0x434f_4445;

/// Header word 5 of a shed f32 row of an unquantized field: the identity
/// header only (no codes); the f32 is durable in [`RawRows`] and read from
/// there (the mapped `.rvec`), as Lucene reads a float HNSW's vectors.
const RAW_ONLY: u32 = 0x5241_5752;

/// A row without in-memory f32 (code-only, or [`RAW_ONLY`]).
#[inline]
fn is_code_only(values: &[f32]) -> bool {
    values.get(5).is_some_and(|v| v.to_bits() == CODE_ONLY || v.to_bits() == RAW_ONLY)
}

fn raw_only_row(key: u64, handle: NodeHandle) -> Result<crate::admission::Allocation<Vec<u32>>> {
    let memory=restore_words(8,"source raw-only row workspace")?;
    let mut out = Vec::new();
    out.try_reserve_exact(8).map_err(|_| failure("source row allocation"))?;
    out.extend_from_slice(&[key as u32, (key >> 32) as u32, handle.generation(), 0, 0, RAW_ONLY, 0, 0]);
    Ok(crate::admission::Allocation::new(out,memory))
}

/// Whether unquantized (f32) fields shed their in-memory f32 once durable
/// (`RUSTIC_F32_GRAPH_SHED=1`; off by default). On 100k 1536-d vectors it
/// halves a node's anonymous memory (1.43 -> 0.74 GB) but costs 16%
/// indexing (1,452 -> 1,224 docs/s) and 12% search latency (p50 2.61 ->
/// 2.93 ms): every traversal step resolves the durable row through the
/// vector store's point table instead of reading it in place.
fn shed_f32_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTIC_F32_GRAPH_SHED").is_ok_and(|v| v == "1"))
}

/// Where a row's codes start: after the identity header (code-only) or
/// after the padded f32 cells.
#[inline]
pub(super) fn codes_start(values: &[f32], dim: usize) -> usize {
    if is_code_only(values) { 8 } else { dim.div_ceil(8) * 8 + 8 }
}

fn code_only_row(key: u64, handle: NodeHandle, epoch: u64, words: &[u32]) -> Result<crate::admission::Allocation<Vec<u32>>> {
    if epoch == 0 || words.is_empty() { return Err(failure("code-only row without codes")); }
    let width=words.len().checked_add(8).ok_or_else(||failure("source row size overflow"))?;
    let memory=restore_words(width,"source code-only row workspace")?;
    let mut out = Vec::new();
    out.try_reserve_exact(width).map_err(|_| failure("source row allocation"))?;
    out.extend_from_slice(&[key as u32, (key >> 32) as u32, handle.generation(), epoch as u32, (epoch >> 32) as u32, CODE_ONLY, 0, 0]);
    out.extend_from_slice(words);
    Ok(crate::admission::Allocation::new(out,memory))
}

#[inline]
fn decode_row(values: &[f32], dim: usize, handle: NodeHandle, key: u64) -> Option<&[f32]> {
    if values.len() < 8 || values[0].to_bits() != key as u32
        || values[1].to_bits() != (key >> 32) as u32
        || values[2].to_bits() != handle.generation() { return None; }
    if values[5].to_bits() == RAW_ONLY {
        // Neither f32 nor codes in memory: the f32 is read from the durable rows.
        let encoded = values[3].to_bits() != 0 || values[4].to_bits() != 0;
        return (!encoded && values.len() == 8).then(|| &values[8..8]);
    }
    if is_code_only(values) {
        // Codes only: the f32 is read from the durable rows.
        let encoded = values[3].to_bits() != 0 || values[4].to_bits() != 0;
        return (encoded && values.len() > 8).then(|| &values[8..8]);
    }
    let width = dim.div_ceil(8) * 8 + 8;
    if values.len() < width { return None; }
    let encoded = values[3].to_bits() != 0 || values[4].to_bits() != 0;
    if encoded != (values.len() > width) { return None; }
    Some(&values[8..8 + dim])
}

#[cfg(test)]
mod source_directory_admission_tests {
    use super::*;
    #[test]
    fn decoded_row_cache_is_charged_to_the_memory_budget() {
        let decoded = DecodedRows::default();
        assert_eq!(decoded.charged_bytes(), 0);
        let dim = 1000;
        for id in 0..10u64 {
            let row = decoded.get_or_decode(id, |out| { out.extend((0..dim).map(|d| d as f32)); true }).unwrap();
            assert_eq!(row.len(), dim);
        }
        let charged = decoded.charged_bytes();
        assert!(charged >= 10 * dim as u64 * 4, "{charged} bytes charged for 10 decoded rows");
        // A cached row is not decoded or charged again.
        let again = decoded.get_or_decode(3, |_| panic!("decoded twice")).unwrap();
        assert_eq!(again[999], 999.0);
        assert_eq!(decoded.charged_bytes(), charged);
        assert!(decoded.execution_result().is_ok());
        // A failed decode charges nothing.
        assert!(decoded.get_or_decode(99, |_| false).is_none());
        assert_eq!(decoded.charged_bytes(), charged);
    }

    #[test]
    fn a_denied_decode_fails_only_the_current_operation() {
        let decoded = DecodedRows::retained();
        let fill = |out: &mut Vec<f32>| { out.extend((0..(DECODED_CHARGE_BLOCK as usize)).map(|d| d as f32)); true };
        assert!(decoded.get_or_decode(1, fill).is_some());
        assert!(decoded.execution_result().is_ok());
        let held = decoded.held_bytes();
        assert!(held >= DECODED_CHARGE_BLOCK * 4);
        decoded.0.lock().deny_next = true;
        assert!(decoded.get_or_decode(2, fill).is_none());
        assert!(matches!(decoded.execution_result(), Err(RusticError::Memory(_))), "the operation that hit the denial fails");
        assert!(decoded.execution_result().is_ok(), "reported once: the next operation is unaffected");
        // The read (a PIT's capture) keeps working: the charge is retried.
        assert!(decoded.get_or_decode(2, fill).is_some());
        assert!(decoded.execution_result().is_ok());
        assert!(decoded.get_or_decode(1, |_| panic!("cached")).is_some());
        assert!(decoded.held_bytes() > held, "a retained cache reports what it holds");
    }
    #[test]
    fn shedding_queue_retains_capacity_and_reuses_popped_slots() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let mut queue=UnshedQueue::default();
        for key in 0..17 {queue.admit_push().unwrap();queue.push_back((NodeHandle::new(key as u32,1),key));}
        let held=owner.bytes().current;
        assert!(held>=17*std::mem::size_of::<(NodeHandle,u64)>() as u64);
        let first=queue.pop_front().unwrap();
        let second=queue.pop_front().unwrap();
        queue.push_front(first);queue.push_back(second);
        assert_eq!(queue.len(),17);assert_eq!(owner.bytes().current,held);
        drop(queue);assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn code_only_assembly_retains_workspace_until_row_drops() {
        let owner=crate::admission::Owner::new();
        let scope=owner.enter();
        let row=code_only_row(17,NodeHandle::new(3,2),5,&[7,9,11]).unwrap();
        assert_eq!(row.len(),11);
        assert_eq!(&row[8..],&[7,9,11]);
        assert_eq!(owner.bytes().current,11*4);
        drop(scope);
        assert_eq!(owner.bytes().current,11*4);
        drop(row);
        assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn source_directory_retirement_keeps_each_generation_charged() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let mut old=SourceDirectory::new(16,None).unwrap();old[3].store(91,Relaxed);
        let mut directory=Directory{current:old,retired:Vec::new(),retired_memory:None};
        let before=owner.bytes().current;
        assert!(SourceDirectory::new(usize::MAX,Some(&directory.current)).is_err());
        assert_eq!(owner.bytes().current,before);assert_eq!(directory.current[3].load(Relaxed),91);
        let replacement=SourceDirectory::new(64,Some(&directory.current)).unwrap();
        directory.reserve_retirement().unwrap();
        let old=std::mem::replace(&mut directory.current,replacement);
        directory.retired.push((7,old));
        let retained=owner.bytes().current;
        assert!(retained>=80*4);
        assert_eq!(directory.retired[0].1[3].load(Relaxed),91);
        directory.current[3].store(12,Relaxed);
        assert_eq!(directory.retired[0].1[3].load(Relaxed),91);
        directory.retired.clear();assert_eq!(owner.bytes().current,retained-16*4);
        drop(directory);assert_eq!(owner.bytes().current,0);
    }
}
