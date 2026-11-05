//! Compact hot references and side metadata with Vespa TypeStableVector's publication contract.
//!
//! Pinned source: vespa-engine/vespa 0521b00640ef934b359d2fc356f97d84499dd17d,
//! vespalib/util/type_stable_vector.{h,hpp}. Growth copies the current array,
//! release-publishes its new pointer, and generation-holds the old allocation.
//! Every current-node lookup acquires the CURRENT pointer. A previously borrowed
//! slice is safe, but does not receive mutations performed after a copy/growth.
//!
//! This module owns array allocations, NOT source rows, LevelsRef allocations,
//! node-ID reuse, or a historical topology. The graph adapter must coordinate
//! those retirements in the same generation domain. Scalar side metadata is a
//! POC adaptation; it is dense, not attached to each edge/candidate. One external
//! graph writer must serialize publication/assignment/advance across all stores.
//! The local mutex serializes table mutations only. Guard acquisition and table
//! reads do not lock. A guard owns hold cells, so reads also borrow this table.

use super::array_store::LevelsRef;
use super::generation::{GenerationGuard, GenerationHandler};
use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum NodeState {
    Vacant = 0,
    Reserved = 1,
    Published = 2,
    Retired = 3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NodeMetadata {
    pub source_key: u64,
    pub incarnation: u32,
    pub publication: u64,
    pub norm: f32,
    pub state: NodeState,
}

/// Source-side scalars only. The graph currently stores its cached inverse
/// norm in `norm`; no readiness/state load is needed after LevelsRef acquire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NodeIdentity {
    pub source_key: u64,
    pub incarnation: u32,
    pub publication: u64,
    pub norm: f32,
}

/// Side metadata has no per-node allocation. The separate LevelsRef plane is the release/acquire
/// publication boundary for immutable source identity/norm/publication fields.
/// Metadata must not be reused until all relevant readers/prepares have drained.
#[repr(C)]
pub(super) struct NodeRecord {
    incarnation: AtomicU32,
    source_key: AtomicU64,
    publication: AtomicU64,
    norm: AtomicU32,
    state: AtomicU32,
}

impl NodeRecord {
    /// The vacant value (an allocation starts zeroed: see `allocate`).
    #[cfg_attr(not(test), allow(dead_code))]
    fn vacant() -> Self {
        Self {
            incarnation: AtomicU32::new(0),
            source_key: AtomicU64::new(0),
            publication: AtomicU64::new(0),
            norm: AtomicU32::new(0),
            state: AtomicU32::new(NodeState::Vacant as u32),
        }
    }

    /// Only called by the serialized writer, which cannot mutate this record
    /// concurrently. Readers only load these atomics.
    fn copy_for_growth(&self) -> Self {
        Self {
            incarnation: AtomicU32::new(self.incarnation.load(Ordering::Relaxed)),
            source_key: AtomicU64::new(self.source_key.load(Ordering::Relaxed)),
            publication: AtomicU64::new(self.publication.load(Ordering::Relaxed)),
            norm: AtomicU32::new(self.norm.load(Ordering::Relaxed)),
            state: AtomicU32::new(self.state.load(Ordering::Relaxed)),
        }
    }

    fn scalar_identity(&self) -> NodeIdentity {
        NodeIdentity {
            source_key: self.source_key.load(Ordering::Relaxed),
            incarnation: self.incarnation.load(Ordering::Relaxed),
            publication: self.publication.load(Ordering::Relaxed),
            norm: f32::from_bits(self.norm.load(Ordering::Relaxed)),
        }
    }

    /// Read source identity/norm/publication after acquiring a VALID LevelsRef
    /// from this record (or an already-guarded reference for the same immutable
    /// incarnation). Keep the matching storage-domain guard continuously held
    /// from reference acquisition through this read. The graph adapter's ID
    /// retirement/source-acknowledgment protocol must prevent incarnation reuse
    /// while that guard or a prepared reference remains live.
    ///
    /// The acquire synchronizes these Relaxed loads with publication. Removal
    /// can invalidate the current LevelsRef without changing these scalars;
    /// a captured traversal reference can therefore still read its source row.
    /// There is deliberately no diagnostic state/READY acquire on this path.
    ///
    /// These are safe atomic scalar reads even without that semantic contract,
    /// but then they do not establish a valid source identity or published node.
    /// A fresh guard cannot resurrect an old arbitrary LevelsRef. A stale copy
    /// borrowed before growth must not be paired with a different incarnation
    /// acquired from the new current table.
    pub(super) fn identity_after_acquire(&self) -> NodeIdentity {
        self.scalar_identity()
    }

    // The same valid-ref/continuous-guard contract as identity_after_acquire
    // applies to each lazy accessor. In particular, a borrowed old allocation
    // is usable only after acquiring its own valid ref, never by combining its
    // Reserved scalars with a newly published ref from a different table copy.
    #[inline]
    pub(super) fn publication_after_acquire(&self) -> u64 {
        self.publication.load(Ordering::Relaxed)
    }
    #[inline]
    pub(super) fn incarnation_after_acquire(&self) -> u32 {
        self.incarnation.load(Ordering::Relaxed)
    }
    #[inline]
    pub(super) fn source_key_after_acquire(&self) -> u64 {
        self.source_key.load(Ordering::Relaxed)
    }
    #[inline]
    pub(super) fn norm_after_acquire(&self) -> f32 {
        f32::from_bits(self.norm.load(Ordering::Relaxed))
    }

    /// Scalar metadata, not a coherent historical snapshot. For a published
    /// node, acquire levels_ref FIRST; the adapter's no-reuse-under-guard rule
    /// then makes identity/norm/publication immutable for that incarnation.
    /// State may change to Retired while an old LevelsRef remains readable.
    pub(super) fn metadata(&self) -> NodeMetadata {
        // Diagnostic path also supports Reserved/Vacant records. Acquiring state
        // first observes reservation initialization, but is not a historical
        // snapshot and may race later state/norm changes while Reserved.
        let state = self.state();
        let identity = self.scalar_identity();
        NodeMetadata {
            source_key: identity.source_key,
            incarnation: identity.incarnation,
            publication: identity.publication,
            norm: identity.norm,
            state,
        }
    }

    fn state(&self) -> NodeState {
        match self.state.load(Ordering::Acquire) {
            0 => NodeState::Vacant,
            1 => NodeState::Reserved,
            2 => NodeState::Published,
            3 => NodeState::Retired,
            _ => unreachable!("node state is only written by this module"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeTableLimits {
    /// Includes reserved node zero. Published length is a node-ID bound, not a
    /// live-node count. Graph ID reuse prevents churn growing this bound forever.
    pub max_slots: u32,
    pub initial_capacity: u32,
    pub grow_percent: u32,
    pub grow_delta: u32,
    /// Old allocation payload budget, floored at one current allocation so
    /// that growing a large table cannot permanently block on its own size.
    /// Additional pinned copies still apply backpressure; max_payload_bytes
    /// remains the strict aggregate allocation bound.
    pub max_held_bytes: usize,
    pub max_held_allocations: usize,
    /// Current + held + new replacement payload during copy. Allocation/list
    /// metadata is accounted separately; this is explicitly a payload budget.
    pub max_payload_bytes: usize,
}

impl Default for NodeTableLimits {
    fn default() -> Self {
        Self {
            max_slots: u32::MAX,
            initial_capacity: 16,
            grow_percent: 100,
            grow_delta: 0,
            max_held_bytes: 256 * 1024 * 1024,
            max_held_allocations: 64,
            max_payload_bytes: usize::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NodeTableError {
    InvalidLimits,
    WrongDomain,
    NodeZero,
    OutOfBounds,
    InvalidMetadata,
    InvalidLevelsRef,
    StaleLevelsRef,
    UnchangedLevelsRef,
    WrongState,
    CapacityLimit,
    HeldLimit,
    PayloadLimit,
    AllocationFailed,
    MemoryLimit,
}

impl fmt::Display for NodeTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node table: {:?}", self)
    }
}
impl std::error::Error for NodeTableError {}

/// Compact hot references are separate from cold identity/publication metadata.
/// One allocation owner publishes both arrays, avoiding mixed growth snapshots.
struct NodeAllocation {
    levels: Vec<AtomicU32>,
    /// Each slot's visibility word, dense by id, for a graph whose source
    /// keeps its rows or codes dense by id (empty otherwise, see
    /// [`NodeTable::enable_dense`]): the record's publication while the slot
    /// holds a published node (valid levels), `u64::MAX` otherwise (vacant,
    /// reserved, removed). A traversal's visibility check of a neighbour is
    /// one load of 8 bytes here (`word <= view publication`) instead of the
    /// levels word and the neighbour's 32-byte record (its key and
    /// incarnation are only read for the nodes a search admits).
    /// Publication stores it (Release) after the levels; removal resets it.
    publications: Vec<HotSlot>,
    records: Vec<NodeRecord>,
    removed: Vec<RemovedLevels>,
    _memory: crate::admission::Reservation,
}

/// A slot's dense traversal words ([`NodeAllocation::publications`]): its
/// visibility word, and its current level-0 links reference (0: unknown or
/// empty, read the levels; reset by removal), kept by the graph writer
/// with every level-0 link publication, so an expansion reads its links in
/// two dependent loads (this word, the links) instead of four.
#[repr(C, align(16))]
pub(super) struct HotSlot {
    pub(super) vis: AtomicU64,
    pub(super) links0: AtomicU32,
    _spare: AtomicU32,
}
impl HotSlot {
    /// The vacant value (an allocation starts zeroed: see `allocate`).
    #[cfg_attr(not(test), allow(dead_code))]
    fn vacant() -> Self {
        Self { vis: AtomicU64::new(u64::MAX), links0: AtomicU32::new(0), _spare: AtomicU32::new(0) }
    }
}

/// The level reference a slot owned when it was last removed, and the domain
/// generation current at that removal. Removal rewrites the neighbours' links
/// and then invalidates the slot in one publication, but a live (uncaptured)
/// search may already hold a neighbour's OLD link list naming the removed slot.
/// Such a reader admitted at or before `generation` still pins the retired
/// levels/links (they are assigned a generation >= this one and reclaimed only
/// strictly below the oldest guard), so it may keep traversing through the
/// removed node without returning it. Readers admitted later reject the entry.
struct RemovedLevels {
    levels: AtomicU32,
    generation: AtomicU64,
}
impl RemovedLevels {
    /// The vacant value (an allocation starts zeroed: see `allocate`).
    #[cfg_attr(not(test), allow(dead_code))]
    fn empty() -> Self {
        Self { levels: AtomicU32::new(0), generation: AtomicU64::new(0) }
    }
    fn copy_for_growth(&self) -> Self {
        Self {
            levels: AtomicU32::new(self.levels.load(Ordering::Relaxed)),
            generation: AtomicU64::new(self.generation.load(Ordering::Relaxed)),
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct NodeRecordRef<'a> {
    levels: &'a AtomicU32,
    record: &'a NodeRecord,
    removed: &'a RemovedLevels,
}
impl std::ops::Deref for NodeRecordRef<'_> {
    type Target = NodeRecord;
    #[inline]
    fn deref(&self) -> &NodeRecord { self.record }
}
impl<'a> NodeRecordRef<'a> {
    #[inline]
    pub(super) fn levels_ref(&self) -> LevelsRef { LevelsRef::from_raw(self.levels.load(Ordering::Acquire)) }
    /// Keep only the identity borrow after acquiring/capturing the level ref.
    /// Its lifetime still belongs to the same continuously pinned table owner.
    #[inline]
    pub(super) fn identity_record(self) -> &'a NodeRecord { self.record }

    /// The levels this slot owned at its most recent removal, only when a
    /// reader whose guard generation is `guard_generation` is guaranteed to
    /// still pin them (see `RemovedLevels`). Call only after the current
    /// levels were observed invalid. Any value that passes the check belongs
    /// to a removal at or after the reader's admission: slot reuse needs every
    /// such reader to drain first, so the pair cannot be torn under it.
    #[inline]
    pub(super) fn removed_levels_for(&self, guard_generation: u64) -> Option<LevelsRef> {
        // Generation first (Acquire pairs with the writer's Release store
        // after the levels store), then the levels it publishes.
        if guard_generation > self.removed.generation.load(Ordering::Acquire) {
            return None;
        }
        let levels = LevelsRef::from_raw(self.removed.levels.load(Ordering::Acquire));
        levels.is_valid().then_some(levels)
    }
}
impl NodeAllocation {
    fn len(&self) -> usize { self.records.len() }
    #[inline]
    fn get(&self, id: usize) -> Option<NodeRecordRef<'_>> {
        Some(NodeRecordRef { levels: self.levels.get(id)?, record: self.records.get(id)?, removed: self.removed.get(id)? })
    }
}
const NODE_BYTES: usize = size_of::<NodeRecord>() + size_of::<AtomicU32>() + size_of::<RemovedLevels>();
/// `capacity` zeroed elements (one zeroed allocation, not written element
/// by element). Only for this table's element types, whose all-zero bytes
/// are a valid value (atomics).
fn zeroed_vec<T>(capacity: usize) -> Result<Vec<T>, NodeTableError> {
    if capacity == 0 || size_of::<T>() == 0 { return Ok(Vec::new()); }
    let layout = std::alloc::Layout::array::<T>(capacity).map_err(|_| NodeTableError::CapacityLimit)?;
    // SAFETY: a nonzero layout; the elements are atomics, valid when zero.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<T>();
    if ptr.is_null() { return Err(NodeTableError::AllocationFailed); }
    // SAFETY: allocated by the global allocator with `T`'s alignment and
    // `capacity` elements, all initialized (zero).
    Ok(unsafe { Vec::from_raw_parts(ptr, capacity, capacity) })
}

/// Per node with the dense publication copy.
const DENSE_NODE_BYTES: usize = NODE_BYTES + size_of::<HotSlot>();

struct HeldAllocation {
    rows: Box<NodeAllocation>,
    // Pending retirement is not reclaimable until assigned by the graph writer.
    generation: Option<u64>,
}

struct Writer {
    current: Box<NodeAllocation>,
    held: Vec<HeldAllocation>,
    held_bytes: usize,
    #[cfg(test)]
    fail_next_allocation: bool,
}

pub(super) struct NodeTable {
    domain: GenerationHandler,
    limits: NodeTableLimits,
    current: AtomicPtr<NodeAllocation>,
    /// Monotonic published bound. Reader loads it BEFORE the current pointer.
    /// Observing an expanded bound orders the subsequent pointer load after the
    /// release publication of the allocation that can hold that bound.
    len: AtomicU32,
    /// Allocations carry the dense publication copy (a quantized field's
    /// graph). Set once, under the writer lock; never cleared.
    dense: std::sync::atomic::AtomicBool,
    writer: Mutex<Writer>,
    budget: std::sync::Arc<crate::admission::Budget>,
    _metadata: crate::admission::Reservation,
}

/// See [`NodeTableRead::hot`].
#[derive(Clone, Copy)]
pub(super) struct HotNodes<'a> {
    pub(super) levels: &'a [AtomicU32],
    pub(super) slots: &'a [HotSlot],
}

pub(super) struct NodeTableRead<'a> {
    table: &'a NodeTable,
    _guard: &'a GenerationGuard,
}

#[derive(Clone, Copy)]
enum NodeChange {
    Publish { levels: LevelsRef, publication: u64, links0: u32 },
    Replace { levels: LevelsRef },
    Invalidate,
}

/// One prevalidated node-owner transition. The retained table writer lock
/// prevents growth or competing changes between validation and commit. This is
/// not a general multi-node transaction: the graph's outer writer coordinates
/// other stores, publication order and pre-reserved retirement/source credits.
/// Dropping without commit changes nothing and simply releases this lock.
/// Do not call another table method that takes its writer mutex until commit or
/// abort (including preflight, memory_usage, growth and reclamation); ordinary
/// guarded reads remain lock-free. Collecting several transitions from this
/// table would deadlock. Multi-node compaction must commit per owner, or later
/// acquire a distinct writer-batch API rather than collecting these tokens.
#[must_use = "commit the transition to publish it, or drop it to abort without changes"]
pub(super) struct NodeTransition<'a> {
    writer: MutexGuard<'a, Writer>,
    id: u32,
    old: LevelsRef,
    change: NodeChange,
    /// Domain generation at preflight; never above the generation the graph
    /// writer later assigns to this transition's retired owners.
    generation: u64,
}

impl NodeTransition<'_> {
    /// The current owner observed during preflight. For publication this is
    /// invalid; replacement/invalidation return the owner to retire afterwards.
    pub(super) fn old_levels_ref(&self) -> LevelsRef {
        self.old
    }

    /// Only atomic stores, followed by releasing the retained writer lock.
    /// No allocation, validation, callbacks, generation changes, or Result.
    /// The caller already prepared array/source ownership and retirement credit.
    pub(super) fn commit(self) -> LevelsRef {
        // SAFETY: preflight validated a nonzero ID inside the current array.
        // The retained mutex excludes resize/growth or competing row mutation.
        let row = self.writer.current.get(self.id as usize).expect("prevalidated node");
        match self.change {
            NodeChange::Publish {
                levels,
                publication,
                links0,
            } => {
                row.publication.store(publication, Ordering::Relaxed);
                row.state
                    .store(NodeState::Published as u32, Ordering::Relaxed);
                row.levels.store(levels.raw(), Ordering::Release);
                // After the levels: a reader that acquires the word sees them
                // and everything installed before them (rows, codes).
                if let Some(dense) = self.writer.current.publications.get(self.id as usize) {
                    dense.links0.store(links0, Ordering::Relaxed);
                    dense.vis.store(publication, Ordering::Release);
                }
            }
            NodeChange::Replace { levels } => {
                row.levels.store(levels.raw(), Ordering::Release);
            }
            NodeChange::Invalidate => {
                if let Some(dense) = self.writer.current.publications.get(self.id as usize) {
                    dense.vis.store(u64::MAX, Ordering::Release);
                    dense.links0.store(0, Ordering::Release);
                }
                if self.old.is_valid() {
                    // Retain the removed owner for readers admitted at or
                    // before this generation (RemovedLevels). Levels first,
                    // then the generation that publishes them.
                    row.removed.levels.store(self.old.raw(), Ordering::Relaxed);
                    row.removed.generation.store(self.generation, Ordering::Release);
                }
                row.levels.store(0, Ordering::Release);
                row.state
                    .store(NodeState::Retired as u32, Ordering::Release);
            }
        }
        self.old
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NodeTableMemory {
    pub node_record_bytes: usize,
    pub len: u32,
    pub capacity: usize,
    pub current_payload_bytes: usize,
    /// Includes reserved zero and vacant entries inside the current bound.
    pub used_payload_bytes: usize,
    pub reserved_zero_bytes: usize,
    pub vacant_reusable_bytes: usize,
    pub retired_node_bytes: usize,
    pub unused_capacity_bytes: usize,
    pub held_payload_bytes: usize,
    pub pending_allocations: usize,
    pub assigned_allocations: usize,
    pub hold_metadata_capacity_bytes: usize,
    pub table_struct_bytes: usize,
    /// Exact owned Rust payload/header capacities; excludes allocator-private
    /// overhead and the shared generation domain (reported by that domain).
    pub accounted_bytes: usize,
}

impl NodeTable {
    pub(super) fn new(
        domain: GenerationHandler,
        limits: NodeTableLimits,
    ) -> Result<Self, NodeTableError> {
        Self::with_budget(domain, limits, crate::admission::shared())
    }

    fn with_budget(domain: GenerationHandler, limits: NodeTableLimits, budget: std::sync::Arc<crate::admission::Budget>) -> Result<Self, NodeTableError> {
        if limits.max_slots == 0
            || limits.initial_capacity == 0
            || limits.initial_capacity > limits.max_slots
            || (limits.grow_percent == 0 && limits.grow_delta == 0)
        {
            return Err(NodeTableError::InvalidLimits);
        }
        if (limits.initial_capacity as usize)
            .checked_mul(NODE_BYTES)
            .is_none_or(|bytes| bytes > limits.max_payload_bytes)
        {
            return Err(NodeTableError::PayloadLimit);
        }
        // Bounded hold-list capacity including old/new Vec overlap. The
        // separate payload owners remain charged until generation reclamation.
        let metadata = budget.reserve((limits.max_held_allocations as u64).saturating_mul((3 * size_of::<HeldAllocation>()) as u64).saturating_add(size_of::<Self>() as u64).saturating_add(256), "mutable node hold metadata").map_err(|_| NodeTableError::MemoryLimit)?;
        let mut rows = Self::allocate(limits.initial_capacity as usize, false, &budget)?;
        if Self::bytes(&rows) > limits.max_payload_bytes {
            return Err(NodeTableError::PayloadLimit);
        }
        let current = AtomicPtr::new(&mut *rows);
        Ok(Self {
            domain,
            limits,
            budget,
            _metadata: metadata,
            current,
            len: AtomicU32::new(1),
            dense: std::sync::atomic::AtomicBool::new(false),
            writer: Mutex::new(Writer {
                current: rows,
                held: Vec::new(),
                held_bytes: 0,
                #[cfg(test)]
                fail_next_allocation: false,
            }),
        })
    }

    fn allocate(capacity: usize, dense: bool, budget: &std::sync::Arc<crate::admission::Budget>) -> Result<Box<NodeAllocation>, NodeTableError> {
        let node_bytes = if dense { DENSE_NODE_BYTES } else { NODE_BYTES };
        capacity.checked_mul(node_bytes).filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(NodeTableError::CapacityLimit)?;
        let memory = budget.reserve((capacity as u64).saturating_mul(node_bytes as u64).saturating_add(size_of::<NodeAllocation>() as u64), "mutable node arrays").map_err(|_| NodeTableError::MemoryLimit)?;
        // Zeroed, not written: an all-zero record, levels word and removal
        // record are their vacant values, and fresh pages stay unbacked
        // until a slot is used, so the capacity growth reserves ahead (up
        // to twice the nodes) costs address space, not resident memory.
        // A dense word's vacant value is not zero: `ensure_len` writes it
        // into each slot before the slot is exposed (`len`), and growth
        // copies the exposed ones.
        let records = zeroed_vec::<NodeRecord>(capacity)?;
        let levels = zeroed_vec::<AtomicU32>(capacity)?;
        let publications = if dense { zeroed_vec::<HotSlot>(capacity)? } else { Vec::new() };
        let removed = zeroed_vec::<RemovedLevels>(capacity)?;
        Ok(Box::new(NodeAllocation { levels, publications, records, removed, _memory: memory }))
    }
    fn bytes(rows: &NodeAllocation) -> usize {
        rows.records.capacity() * size_of::<NodeRecord>() + rows.levels.capacity() * size_of::<AtomicU32>()
            + rows.publications.capacity() * size_of::<HotSlot>()
            + rows.removed.capacity() * size_of::<RemovedLevels>()
    }

    #[inline]
    pub(super) fn read<'a>(
        &'a self,
        guard: &'a GenerationGuard,
    ) -> Result<NodeTableRead<'a>, NodeTableError> {
        if !self.domain.owns_guard(guard) {
            return Err(NodeTableError::WrongDomain);
        }
        Ok(NodeTableRead {
            table: self,
            _guard: guard,
        })
    }

    /// Reuse admission-time ownership validation for this exact table/guard.
    /// # Safety
    /// `read(guard)` must have succeeded and both borrows must remain unchanged.
    #[inline]
    pub(super) unsafe fn read_prevalidated<'a>(&'a self, guard: &'a GenerationGuard) -> NodeTableRead<'a> {
        NodeTableRead { table: self, _guard: guard }
    }

    /// Ensure an addressable node-ID bound. Like TypeStableVector::ensure_size /
    /// reserve, requested growth uses grow(required), not grow(old capacity).
    /// No shrink operation: the graph's separate active bound may shrink.
    pub(super) fn ensure_len(&self, required: u32) -> Result<(), NodeTableError> {
        if required > self.limits.max_slots {
            return Err(NodeTableError::CapacityLimit);
        }
        let mut writer = self.writer.lock();
        if required <= self.len.load(Ordering::Relaxed) {
            return Ok(());
        }
        if required as usize > writer.current.len() {
            let grown = u64::from(required)
                + u64::from(required) * u64::from(self.limits.grow_percent) / 100
                + u64::from(self.limits.grow_delta);
            let capacity = grown.min(u64::from(self.limits.max_slots)) as usize;
            self.replace_allocation(&mut writer, capacity)?;
        }
        // The slots this exposes get their vacant dense word (allocations
        // start zeroed, see `allocate`) before readers can reach them.
        let exposed = self.len.load(Ordering::Relaxed) as usize;
        for slot in writer.current.publications.iter().take(required as usize).skip(exposed) {
            slot.vis.store(u64::MAX, Ordering::Relaxed);
        }
        self.len.store(required, Ordering::Release);
        Ok(())
    }

    /// Give this table the dense publication copy (a quantized field's
    /// graph, whose code plane traversal reads it): the current allocation is
    /// replaced by one carrying it (filled from the records), as growth
    /// replaces it, and every later allocation carries it. The old allocation
    /// is held like a grown one; the caller is the graph writer, which assigns
    /// its generation (`maintenance_pending`). No-op once enabled.
    pub(super) fn enable_dense(&self) -> Result<(), NodeTableError> {
        let mut writer = self.writer.lock();
        if self.dense.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.dense.store(true, Ordering::Relaxed);
        let capacity = writer.current.len();
        if let Err(error) = self.replace_allocation(&mut writer, capacity) {
            self.dense.store(false, Ordering::Relaxed);
            return Err(error);
        }
        Ok(())
    }

    /// Replace the current allocation by one of `capacity` (at least the
    /// current length) with the same contents, holding the old one for its
    /// readers. Caller holds the table writer lock.
    fn replace_allocation(&self, writer: &mut Writer, capacity: usize) -> Result<(), NodeTableError> {
        let dense = self.dense.load(Ordering::Relaxed);
        let old_bytes = Self::bytes(&writer.current);
        let held_after = writer
            .held_bytes
            .checked_add(old_bytes)
            .ok_or(NodeTableError::HeldLimit)?;
        if held_after > self.limits.max_held_bytes.max(old_bytes)
            || writer.held.len() >= self.limits.max_held_allocations
        {
            return Err(NodeTableError::HeldLimit);
        }
        let wanted_bytes = capacity
            .checked_mul(if dense { DENSE_NODE_BYTES } else { NODE_BYTES })
            .ok_or(NodeTableError::CapacityLimit)?;
        if held_after
            .checked_add(wanted_bytes)
            .is_none_or(|bytes| bytes > self.limits.max_payload_bytes)
        {
            return Err(NodeTableError::PayloadLimit);
        }
        // Reserve bookkeeping before any publication. Failure leaves both
        // the old pointer and published bound unchanged.
        writer
            .held
            .try_reserve(1)
            .map_err(|_| NodeTableError::AllocationFailed)?;
        #[cfg(test)]
        if std::mem::take(&mut writer.fail_next_allocation) {
            return Err(NodeTableError::AllocationFailed);
        }
        let mut replacement = Self::allocate(capacity, dense, &self.budget)?;
        if held_after
            .checked_add(Self::bytes(&replacement))
            .is_none_or(|bytes| bytes > self.limits.max_payload_bytes)
        {
            return Err(NodeTableError::PayloadLimit);
        }
        // Only the exposed slots (`len`) are copied: the others are vacant
        // in both allocations (zeroed; see `allocate`), and writing them
        // would back the new allocation's whole capacity.
        let exposed = (self.len.load(Ordering::Relaxed) as usize).min(writer.current.records.len());
        for (to, from) in replacement.records.iter_mut().zip(&writer.current.records).take(exposed) {
            *to = from.copy_for_growth();
        }
        for (to, from) in replacement.levels.iter_mut().zip(&writer.current.levels).take(exposed) {
            *to = AtomicU32::new(from.load(Ordering::Relaxed));
        }
        // The visibility words: each published record's publication (the
        // records are their source when the old allocation has none).
        for (id, ((to, from), levels)) in replacement.publications.iter_mut().zip(&writer.current.records).zip(&writer.current.levels).take(exposed).enumerate() {
            let published = levels.load(Ordering::Relaxed) != 0;
            to.vis = AtomicU64::new(if published { from.publication.load(Ordering::Relaxed) } else { u64::MAX });
            // Level-0 links: carried along (filled by the graph storage when
            // the words are first enabled on a populated graph).
            let links0 = writer.current.publications.get(id).map_or(0, |old| old.links0.load(Ordering::Relaxed));
            to.links0 = AtomicU32::new(if published { links0 } else { 0 });
        }
        for (to, from) in replacement.removed.iter_mut().zip(&writer.current.removed).take(exposed) {
            *to = from.copy_for_growth();
        }
        let pointer = &mut *replacement as *mut NodeAllocation;
        let old = std::mem::replace(&mut writer.current, replacement);
        // New pointer first, then any expanded bound. Keep the old owner
        // queued even if later generation advancement fails.
        self.current.store(pointer, Ordering::Release);
        writer.held.push(HeldAllocation {
            rows: old,
            generation: None,
        });
        writer.held_bytes = held_after;
        Ok(())
    }

    /// A reservation can precede source norm calculation. Update only an
    /// unpublished Reserved node, using the CURRENT table after any growth.
    /// The later publish's LevelsRef release orders this store for readers.
    /// This cannot alter a published/retired/vacant identity or norm.
    pub(super) fn set_reserved_norm(&self, id: u32, norm: f32) -> Result<(), NodeTableError> {
        if !norm.is_finite() || norm < 0.0 {
            return Err(NodeTableError::InvalidMetadata);
        }
        let writer = self.writer.lock();
        let row = self.row(&writer, id)?;
        if row.state() != NodeState::Reserved || row.levels_ref().is_valid() {
            return Err(NodeTableError::WrongState);
        }
        row.norm.store(norm.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    fn row<'a>(&self, writer: &'a Writer, id: u32) -> Result<NodeRecordRef<'a>, NodeTableError> {
        if id == 0 {
            return Err(NodeTableError::NodeZero);
        }
        if id >= self.len.load(Ordering::Relaxed) {
            return Err(NodeTableError::OutOfBounds);
        }
        Ok(writer.current.get(id as usize).expect("validated node bound"))
    }

    /// Copy current metadata under the table writer lock. This is the control
    /// path for reservation cancellation and source acknowledgment; unlike a
    /// query view, it needs no fallible reader-guard admission. The graph writer
    /// must serialize the identity check with the subsequent transition. Do not
    /// call while a NodeTransition already holds this table's writer lock.
    pub(super) fn writer_metadata(&self, id: u32) -> Result<NodeMetadata, NodeTableError> {
        let writer = self.writer.lock();
        Ok(self.row(&writer, id)?.metadata())
    }

    /// Initialize an unpublished reservation. Call ensure_len first.
    ///
    /// # Safety
    /// The graph writer owns this ID. If reused, every old graph/prepared reader
    /// that could read its metadata has drained, and source retirement has been
    /// acknowledged. Source identity stays immutable until that condition holds
    /// again; norm may change only while Reserved, and becomes immutable at
    /// publication. The source row must cover preparation through completion.
    pub(super) unsafe fn initialize(
        &self,
        id: u32,
        metadata: NodeMetadata,
    ) -> Result<(), NodeTableError> {
        if metadata.incarnation == 0
            || !metadata.norm.is_finite()
            || metadata.norm < 0.0
            || metadata.state != NodeState::Reserved
        {
            return Err(NodeTableError::InvalidMetadata);
        }
        let writer = self.writer.lock();
        let row = self.row(&writer, id)?;
        if row.state() != NodeState::Vacant || row.levels_ref().is_valid() {
            return Err(NodeTableError::WrongState);
        }
        row.source_key.store(metadata.source_key, Ordering::Relaxed);
        row.incarnation
            .store(metadata.incarnation, Ordering::Relaxed);
        row.publication
            .store(metadata.publication, Ordering::Relaxed);
        row.norm.store(metadata.norm.to_bits(), Ordering::Relaxed);
        row.state
            .store(NodeState::Reserved as u32, Ordering::Release);
        Ok(())
    }

    /// # Safety
    /// `levels` is a fully initialized live allocation in this graph's shared
    /// domain. Its transitive link/source owners are installed before this call.
    /// Their retirement is coordinated by the graph adapter, not this table.
    pub(super) unsafe fn publish(
        &self,
        id: u32,
        levels: LevelsRef,
        publication: u64,
    ) -> Result<(), NodeTableError> {
        // SAFETY: this method forwards its unchanged adoption obligation.
        unsafe { self.preflight_publish(id, levels, publication)? }.commit();
        Ok(())
    }

    /// Validate publication and retain the CURRENT table writer/root until
    /// commit/abort. The norm must already be set while Reserved.
    ///
    /// # Safety
    /// Same graph/domain/source ownership requirement as publish. Keep the new
    /// LevelsRef and all transitive owners alive through commit or abort. After
    /// commit, the graph adapter owns their guarded retirement obligation.
    pub(super) unsafe fn preflight_publish(
        &self,
        id: u32,
        levels: LevelsRef,
        publication: u64,
    ) -> Result<NodeTransition<'_>, NodeTableError> {
        // SAFETY: forwarded.
        unsafe { self.preflight_publish_with_links0(id, levels, publication, 0) }
    }

    /// [`Self::preflight_publish`], recording the node's level-0 links
    /// reference in its dense words (when the table keeps them).
    ///
    /// # Safety
    /// As `preflight_publish`; `links0` is the published level-0 links
    /// reference of `levels` (or 0).
    pub(super) unsafe fn preflight_publish_with_links0(
        &self,
        id: u32,
        levels: LevelsRef,
        publication: u64,
        links0: u32,
    ) -> Result<NodeTransition<'_>, NodeTableError> {
        self.preflight(
            id,
            Some(LevelsRef::INVALID),
            NodeChange::Publish {
                levels,
                publication,
                links0,
            },
        )
    }

    /// Record new level-0 links references (`(id, raw)`) of published
    /// nodes in their dense words, after the graph writer published them.
    /// No allocation or failure (a no-op without the dense words).
    pub(super) fn set_links0(&self, updates: &[(u32, u32)]) {
        if updates.is_empty() || !self.dense.load(Ordering::Relaxed) { return; }
        let writer = self.writer.lock();
        for &(id, raw) in updates {
            if let Some(slot) = writer.current.publications.get(id as usize) {
                slot.links0.store(raw, Ordering::Release);
            }
        }
    }

    /// Fill every published node's level-0 links word from its levels
    /// (`links0` of a levels reference), when the dense words are enabled
    /// on a populated graph. Caller holds the graph writer.
    pub(super) fn fill_links0(&self, links0: impl Fn(LevelsRef) -> u32) {
        let writer = self.writer.lock();
        let len = (self.len.load(Ordering::Relaxed) as usize).min(writer.current.levels.len());
        for id in 1..len {
            let levels = LevelsRef::from_raw(writer.current.levels[id].load(Ordering::Relaxed));
            if let Some(slot) = writer.current.publications.get(id).filter(|_| levels.is_valid()) {
                slot.links0.store(links0(levels), Ordering::Release);
            }
        }
    }

    /// # Safety
    /// Same allocation/publication contract as publish. Retire the returned
    /// old LevelsRef after publication using the shared generation domain.
    pub(super) unsafe fn replace_levels(
        &self,
        id: u32,
        levels: LevelsRef,
    ) -> Result<LevelsRef, NodeTableError> {
        Ok(self
            .preflight(id, None, NodeChange::Replace { levels })?
            .commit())
    }

    /// Validate the expected current owner before a graph adapter begins
    /// publishing other array owners. No ownership stores happen in preflight.
    ///
    /// # Safety
    /// Same adoption/retirement contract as replace_levels. The new reference
    /// and transitive owners must remain live through commit or abort.
    pub(super) unsafe fn preflight_replace_levels(
        &self,
        id: u32,
        expected: LevelsRef,
        levels: LevelsRef,
    ) -> Result<NodeTransition<'_>, NodeTableError> {
        self.preflight(id, Some(expected), NodeChange::Replace { levels })
    }

    /// Invalidates current ownership without clearing old scalar metadata or
    /// held copies. The caller must retire returned levels/links and source/ID.
    /// Reserved nodes can also be cancelled through this path.
    pub(super) fn invalidate(&self, id: u32) -> Result<LevelsRef, NodeTableError> {
        Ok(self.preflight(id, None, NodeChange::Invalidate)?.commit())
    }

    /// Expected is invalid when cancelling a Reserved node, or the valid
    /// current owner when deleting a Published node. A stale owner is rejected.
    /// Retirement/source-ID queue capacity must be admitted before graph commit.
    /// The graph's outer writer must first validate its full handle/incarnation;
    /// INVALID alone cannot distinguish two successive reservations of one ID.
    pub(super) fn preflight_invalidate(
        &self,
        id: u32,
        expected: LevelsRef,
    ) -> Result<NodeTransition<'_>, NodeTableError> {
        self.preflight(id, Some(expected), NodeChange::Invalidate)
    }

    fn preflight(
        &self,
        id: u32,
        expected: Option<LevelsRef>,
        change: NodeChange,
    ) -> Result<NodeTransition<'_>, NodeTableError> {
        match change {
            NodeChange::Publish { levels, .. } | NodeChange::Replace { levels }
                if !levels.is_valid() =>
            {
                return Err(NodeTableError::InvalidLevelsRef);
            }
            _ => {}
        }
        let writer = self.writer.lock();
        let row = self.row(&writer, id)?;
        let state = row.state();
        let old = row.levels_ref();
        let valid_state = match change {
            NodeChange::Publish { .. } => state == NodeState::Reserved && !old.is_valid(),
            NodeChange::Replace { .. } => state == NodeState::Published && old.is_valid(),
            NodeChange::Invalidate => {
                (state == NodeState::Reserved && !old.is_valid())
                    || (state == NodeState::Published && old.is_valid())
            }
        };
        if !valid_state {
            return Err(NodeTableError::WrongState);
        }
        if expected.is_some_and(|expected| expected != old) {
            return Err(NodeTableError::StaleLevelsRef);
        }
        // Returning the still-current allocation as a retirement candidate
        // would invite retiring a live owner. A caller can skip a no-op swap.
        if matches!(change, NodeChange::Replace { levels } if levels == old) {
            return Err(NodeTableError::UnchangedLevelsRef);
        }
        Ok(NodeTransition {
            writer,
            id,
            old,
            change,
            generation: self.domain.current_generation(),
        })
    }

    /// # Safety
    /// The external graph/source-ID retirement protocol proves no old reader or
    /// prepared operation may acquire/read this incarnation's metadata, and
    /// source reclamation is acknowledged. This table's allocation holds alone
    /// do NOT prove that an ID is reusable.
    pub(super) unsafe fn mark_reclaimed(&self, id: u32) -> Result<(), NodeTableError> {
        let writer = self.writer.lock();
        let row = self.row(&writer, id)?;
        if row.state() != NodeState::Retired || row.levels_ref().is_valid() {
            return Err(NodeTableError::WrongState);
        }
        row.state.store(NodeState::Vacant as u32, Ordering::Release);
        Ok(())
    }

    /// Graph-writer protocol: publish all stores, assign their pending holds,
    /// then advance the shared handler. Do not pass arbitrary generation numbers.
    pub(super) fn assign_generation(&self) {
        let mut writer = self.writer.lock();
        let generation = self.domain.current_generation();
        for held in &mut writer.held {
            if held.generation.is_none() {
                held.generation = Some(generation);
            }
        }
    }

    /// Pending and equal-generation allocations remain held. An advance failure
    /// therefore retains old roots; it cannot turn into premature reclamation.
    pub(super) fn reclaim(&self) -> usize {
        let mut writer = self.writer.lock();
        let oldest = self.domain.update_oldest_used_generation();
        let before = writer.held_bytes;
        writer.held.retain(|held| {
            held.generation
                .is_none_or(|generation| generation >= oldest)
        });
        writer.held_bytes = writer.held.iter().map(|held| Self::bytes(&held.rows)).sum();
        before - writer.held_bytes
    }

    pub(super) fn memory_usage(&self) -> NodeTableMemory {
        let writer = self.writer.lock();
        let len = self.len.load(Ordering::Relaxed);
        let record_bytes = if self.dense.load(Ordering::Relaxed) { DENSE_NODE_BYTES } else { NODE_BYTES };
        let current_bytes = Self::bytes(&writer.current);
        let used_bytes = len as usize * record_bytes;
        let mut vacant = 0;
        let mut retired = 0;
        for row in &writer.current.records[1..len as usize] {
            match row.state() {
                NodeState::Vacant => vacant += record_bytes,
                NodeState::Retired => retired += record_bytes,
                _ => {}
            }
        }
        let pending = writer
            .held
            .iter()
            .filter(|held| held.generation.is_none())
            .count();
        let hold_metadata = writer.held.capacity() * size_of::<HeldAllocation>();
        let table_bytes = size_of::<Self>();
        NodeTableMemory {
            node_record_bytes: record_bytes,
            len,
            capacity: writer.current.len(),
            current_payload_bytes: current_bytes,
            used_payload_bytes: used_bytes,
            reserved_zero_bytes: record_bytes,
            vacant_reusable_bytes: vacant,
            retired_node_bytes: retired,
            unused_capacity_bytes: current_bytes - used_bytes,
            held_payload_bytes: writer.held_bytes,
            pending_allocations: pending,
            assigned_allocations: writer.held.len() - pending,
            hold_metadata_capacity_bytes: hold_metadata,
            table_struct_bytes: table_bytes,
            accounted_bytes: current_bytes + writer.held_bytes + hold_metadata + table_bytes,
        }
    }

    /// Constant-time reclamation predicate support. Diagnostic memory_usage
    /// scans current rows and must not be used on a frequent drain hot path.
    pub(super) fn held_payload_bytes(&self) -> usize {
        self.writer.lock().held_bytes
    }
}

impl<'a> NodeTableRead<'a> {
    // References borrow the table and guard, not this short-lived wrapper. This
    // lets an owning graph snapshot create a temporary view for each lookup.
    #[inline]
    pub(super) fn get(&self, id: u32) -> Option<NodeRecordRef<'a>> {
        // Bound BEFORE pointer: a newer bound cannot authorize indexing into an
        // older, smaller allocation. Both loads remain inside this guard.
        let len = self.table.len.load(Ordering::Acquire);
        if id == 0 || id >= len {
            return None;
        }
        let current = self.table.current.load(Ordering::Acquire);
        // SAFETY: same-domain guard pinned before acquiring current; the table
        // borrow keeps all owners alive; growth holds the old allocation until
        // this guard drains. Every addressable record was initialized.
        unsafe { &*current }.get(id as usize)
    }

    /// [`get`](Self::get) with the record's publication from the dense
    /// copy (same value and contract as `NodeRecord::publication_after_acquire`):
    /// for a traversal that otherwise reads nothing of the record.
    #[inline]
    pub(super) fn get_with_publication(&self, id: u32) -> Option<(NodeRecordRef<'a>, &'a AtomicU64)> {
        let len = self.table.len.load(Ordering::Acquire);
        if id == 0 || id >= len {
            return None;
        }
        let current = self.table.current.load(Ordering::Acquire);
        // SAFETY: as `get`.
        let allocation = unsafe { &*current };
        let row = allocation.get(id as usize)?;
        // Without the dense copy (before `enable_dense`), the record's own.
        let record: &'a NodeRecord = row.record;
        let publication = allocation.publications.get(id as usize).map_or(&record.publication, |slot| &slot.vis);
        Some((row, publication))
    }

    /// Warm `id`'s levels word and dense publication (the record is left for
    /// the nodes a search admits), without reading them.
    #[inline]
    pub(super) fn prefetch_hot(&self, id: u32) {
        let len = self.table.len.load(Ordering::Acquire);
        if id == 0 || id >= len { return; }
        let current = self.table.current.load(Ordering::Acquire);
        // SAFETY: as `get`; addresses only, nothing is read.
        let allocation = unsafe { &*current };
        let Some(levels) = allocation.levels.get(id as usize) else { return };
        crate::vector::distance::prefetch((levels as *const AtomicU32).cast::<u8>(), 4);
        match allocation.publications.get(id as usize) {
            Some(slot) => crate::vector::distance::prefetch((slot as *const HotSlot).cast::<u8>(), 8),
            None => if let Some(record) = allocation.records.get(id as usize) {
                crate::vector::distance::prefetch((record as *const NodeRecord).cast::<u8>(), size_of::<NodeRecord>());
            },
        }
    }

    /// Warm `id`'s levels word and identity record (separate arrays, so two
    /// lines) without reading them. Traversal issues this for a whole
    /// neighbour list before resolving any of it.
    #[inline]
    pub(super) fn prefetch(&self, id: u32) {
        if let Some(row) = self.get(id) {
            crate::vector::distance::prefetch((row.levels as *const AtomicU32).cast::<u8>(), 4);
            crate::vector::distance::prefetch((row.record as *const NodeRecord).cast::<u8>(), size_of::<NodeRecord>());
        }
    }

    /// The current allocation's levels words and visibility words (see
    /// [`NodeAllocation::publications`]), `None` without the dense words.
    /// Bounded by the published length, read before the allocation (as
    /// `get`). A borrowed allocation stays valid for the guard's lifetime
    /// but misses later growth's mutations: a traversal re-borrows it per
    /// expansion and checks a word's node again on the current table before
    /// it relies on a change (removal) it could have missed.
    #[inline]
    pub(super) fn hot(&self) -> Option<HotNodes<'a>> {
        let len = self.table.len.load(Ordering::Acquire) as usize;
        let current = self.table.current.load(Ordering::Acquire);
        // SAFETY: as `get`.
        let allocation: &'a NodeAllocation = unsafe { &*current };
        let len = len.min(allocation.levels.len());
        if allocation.publications.len() < len { return None; }
        Some(HotNodes { levels: &allocation.levels[..len], slots: &allocation.publications[..len] })
    }

    /// An explicitly borrowed slice may become stale after growth. Use get for
    /// CURRENT lookup. Row zero is included for table diagnostics, always vacant.
    pub(super) fn slice(&self, count: u32) -> Option<Vec<NodeRecordRef<'a>>> {
        let len = self.table.len.load(Ordering::Acquire);
        if count > len {
            return None;
        }
        let current = self.table.current.load(Ordering::Acquire);
        // SAFETY: same bound/order/lifetime proof as get; nonnull even count 0.
        Some((0..count as usize).map(|id| unsafe { &*current }.get(id).expect("validated bound")).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::generation::{GenerationError, GenerationLimits};
    use super::*;
    use std::sync::Arc;

    fn domain() -> GenerationHandler {
        GenerationHandler::new(GenerationLimits::default()).unwrap()
    }

    fn limits() -> NodeTableLimits {
        NodeTableLimits {
            max_slots: 128,
            initial_capacity: 2,
            max_held_bytes: 32 * 1024,
            max_payload_bytes: 64 * 1024,
            ..NodeTableLimits::default()
        }
    }

    #[test]
    fn shared_admission_refusal_keeps_published_node_bound_and_pointer() {
        let budget = crate::admission::Budget::new(1 << 20);
        let table = NodeTable::with_budget(domain(), limits(), budget.clone()).unwrap();
        let pointer = table.current.load(Ordering::Acquire);
        let len = table.len.load(Ordering::Acquire);
        let used = budget.used();
        budget.set_limit(used);
        assert_eq!(table.ensure_len(8), Err(NodeTableError::MemoryLimit));
        assert_eq!(table.current.load(Ordering::Acquire), pointer);
        assert_eq!(table.len.load(Ordering::Acquire), len);
        assert_eq!(budget.used(), used);
        assert!(matches!(super::super::failure_error(NodeTableError::MemoryLimit), crate::RusticError::Memory(_)));
        drop(table);assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shared_admission_retains_old_node_arrays_until_guards_drain() {
        let d = domain();
        let budget = crate::admission::Budget::new(1 << 20);
        let table = NodeTable::with_budget(d.clone(), limits(), budget.clone()).unwrap();
        table.ensure_len(2).unwrap();
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let old = view.get(1).unwrap();
        table.ensure_len(8).unwrap();
        table.assign_generation();d.inc_generation().unwrap();
        let used = budget.used();
        assert_eq!(table.reclaim(), 0);
        assert_eq!(budget.used(), used);
        assert_eq!(old.metadata().state, NodeState::Vacant);
        drop(view);drop(guard);
        assert!(table.reclaim() > 0);
        assert!(budget.used() < used);
        drop(table);assert_eq!(budget.used(), 0);
    }

    fn metadata(key: u64, incarnation: u32) -> NodeMetadata {
        NodeMetadata {
            source_key: key,
            incarnation,
            publication: 0,
            norm: 2.5,
            state: NodeState::Reserved,
        }
    }

    // This isolated fixture uses numeric LevelsRefs without dereferencing them;
    // table tests have no live source/array store whose ownership could be lost.
    fn add(table: &NodeTable, id: u32, key: u64, incarnation: u32) {
        table.ensure_len(id + 1).unwrap();
        unsafe {
            table.initialize(id, metadata(key, incarnation)).unwrap();
            table.publish(id, LevelsRef::from_raw(id + 100), 7).unwrap();
        }
    }

    #[test]
    fn dense_layout_reserved_zero_and_foreign_guard() {
        assert_eq!(size_of::<AtomicU32>(), 4);
        assert_eq!(size_of::<NodeRecord>(), 32);
        let allocation = NodeTable::allocate(4, false, &crate::admission::Budget::new(1 << 20)).unwrap();
        assert_eq!(unsafe { allocation.levels.as_ptr().add(1) } as usize - allocation.levels.as_ptr() as usize, 4);
        assert_eq!(size_of::<LevelsRef>(), 4);
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        assert!(view.get(0).is_none());
        assert!(view.get(1).is_none());
        assert!(view.slice(2).is_none());
        assert_eq!(view.slice(1).unwrap()[0].state(), NodeState::Vacant);
        let foreign = domain().take_guard().unwrap();
        assert!(matches!(
            table.read(&foreign),
            Err(NodeTableError::WrongDomain)
        ));
        assert_eq!(
            unsafe { table.initialize(0, metadata(0, 1)) },
            Err(NodeTableError::NodeZero)
        );
        assert_eq!(table.invalidate(0), Err(NodeTableError::NodeZero));
        assert_eq!(table.memory_usage().reserved_zero_bytes, NODE_BYTES);
    }

    #[test]
    fn old_slice_survives_growth_current_lookup_sees_later_mutation() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        add(&table, 1, 11, 1);
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let old = view.slice(2).unwrap();
        // The temporary wrapper can drop; table+guard are the actual owners.
        let from_temporary_view = table.read(&guard).unwrap().get(1).unwrap();
        let old_ref = old[1].levels_ref();
        add(&table, 2, 22, 1); // forces current-root copy
        assert_eq!(old[1].metadata().source_key, 11);
        unsafe {
            table.replace_levels(1, LevelsRef::from_raw(999)).unwrap();
        }
        assert_eq!(old[1].levels_ref(), old_ref);
        assert_eq!(from_temporary_view.levels_ref(), old_ref);
        assert_eq!(view.get(1).unwrap().levels_ref().raw(), 999);
        assert_eq!(view.get(2).unwrap().metadata().source_key, 22);
        table.assign_generation();
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 0);
        assert_eq!(old[1].levels_ref(), old_ref);
        drop(guard);
        assert_eq!(table.reclaim(), 2 * NODE_BYTES);
        assert_eq!(table.memory_usage().held_payload_bytes, 0);
        assert_eq!(table.reclaim(), 0); // exact once
    }

    /// `allocate` zeroes instead of writing the vacant values: those of the
    /// record, levels word and removal record must be all-zero bytes.
    #[test]
    fn vacant_values_are_zero_bytes() {
        fn bytes<T>(value: &T) -> Vec<u8> {
            // SAFETY: reads the value's own bytes (atomics, no padding read
            // through a reference to uninitialized memory: compared below).
            unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }.to_vec()
        }
        let zeroed = zeroed_vec::<NodeRecord>(3).unwrap();
        assert!(zeroed.iter().all(|r| r.state.load(Ordering::Relaxed) == NodeState::Vacant as u32
            && r.incarnation.load(Ordering::Relaxed) == 0 && r.source_key.load(Ordering::Relaxed) == 0
            && r.publication.load(Ordering::Relaxed) == 0 && r.norm.load(Ordering::Relaxed) == 0));
        let v = NodeRecord::vacant();
        assert_eq!((v.state.load(Ordering::Relaxed), v.incarnation.load(Ordering::Relaxed), v.source_key.load(Ordering::Relaxed),
            v.publication.load(Ordering::Relaxed), v.norm.load(Ordering::Relaxed)), (0, 0, 0, 0, 0));
        let r = RemovedLevels::empty();
        assert_eq!(bytes(&r.levels), vec![0; 4]);
        assert_eq!(bytes(&r.generation), vec![0; 8]);
        assert_eq!(HotSlot::vacant().vis.load(Ordering::Relaxed), u64::MAX, "the dense word is written at exposure");
    }

    #[test]
    fn dense_publications_are_filled_from_the_records_and_kept_on_growth() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(3).unwrap();
        unsafe {
            table.initialize(1, metadata(11, 1)).unwrap();
            table.publish(1, LevelsRef::from_raw(101), 7).unwrap();
        }
        {
            // An f32 graph: no dense copy, the record's own publication.
            let guard = d.take_guard().unwrap();
            let view = table.read(&guard).unwrap();
            let (row, publication) = view.get_with_publication(1).unwrap();
            assert!(std::ptr::eq(publication, &row.record.publication));
            assert_eq!(publication.load(Ordering::Relaxed), 7);
            assert_eq!(table.memory_usage().node_record_bytes, NODE_BYTES);
        }
        table.enable_dense().unwrap();
        table.enable_dense().unwrap(); // once
        assert_eq!(table.memory_usage().node_record_bytes, DENSE_NODE_BYTES);
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let (row, publication) = view.get_with_publication(1).unwrap();
        assert!(!std::ptr::eq(publication, &row.record.publication));
        assert_eq!(publication.load(Ordering::Relaxed), 7);
        unsafe {
            table.initialize(2, metadata(12, 1)).unwrap();
            table.publish(2, LevelsRef::from_raw(102), 9).unwrap();
        }
        assert_eq!(view.get_with_publication(2).unwrap().1.load(Ordering::Relaxed), 9);
        // Growth copies the dense copy along.
        table.ensure_len(40).unwrap();
        let (row, publication) = view.get_with_publication(2).unwrap();
        assert!(!std::ptr::eq(publication, &row.record.publication));
        assert_eq!(publication.load(Ordering::Relaxed), 9);
        assert_eq!(view.get_with_publication(1).unwrap().1.load(Ordering::Relaxed), 7);
        // Not published (vacant, reserved, removed): the word is MAX, before
        // and after growth.
        let word = |id: u32| table.read(&guard).unwrap().hot().unwrap().slots[id as usize].vis.load(Ordering::Relaxed);
        let links0 = |id: u32| table.read(&guard).unwrap().hot().unwrap().slots[id as usize].links0.load(Ordering::Relaxed);
        assert_eq!(word(3), u64::MAX);
        unsafe { table.initialize(3, metadata(13, 1)).unwrap(); }
        assert_eq!(word(3), u64::MAX, "reserved");
        table.invalidate(1).unwrap();
        assert_eq!(word(1), u64::MAX, "removed");
        assert_eq!(word(2), 9);
        table.ensure_len(100).unwrap();
        assert_eq!((word(1), word(2), word(3), word(90)), (u64::MAX, 9, u64::MAX, u64::MAX));
        let hot = table.read(&guard).unwrap().hot().unwrap();
        assert_eq!(hot.levels.len(), hot.slots.len());
        assert_eq!(hot.levels[2].load(Ordering::Relaxed), 102);
        // Level-0 links words: set at publication, updated, carried on
        // growth, filled, reset by removal.
        assert_eq!(links0(2), 0);
        unsafe { table.preflight_publish_with_links0(3, LevelsRef::from_raw(103), 11, 77) }.unwrap().commit();
        assert_eq!((word(3), links0(3)), (11, 77));
        table.set_links0(&[(3, 78), (2, 52)]);
        assert_eq!((links0(2), links0(3)), (52, 78));
        table.ensure_len(128).unwrap();
        assert_eq!((links0(2), links0(3)), (52, 78));
        table.fill_links0(|levels| levels.raw() + 1000);
        assert_eq!((links0(1), links0(2), links0(3)), (0, 1102, 1103), "only published nodes");
        table.invalidate(3).unwrap();
        assert_eq!((word(3), links0(3)), (u64::MAX, 0));
    }

    #[test]
    fn reserved_norm_is_validated_and_immutable_after_publication() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        assert_eq!(
            table.set_reserved_norm(0, 1.0),
            Err(NodeTableError::NodeZero)
        );
        assert_eq!(
            table.set_reserved_norm(2, 1.0),
            Err(NodeTableError::OutOfBounds)
        );
        assert_eq!(
            table.set_reserved_norm(1, 1.0),
            Err(NodeTableError::WrongState)
        );
        unsafe {
            table
                .initialize(
                    1,
                    NodeMetadata {
                        norm: 0.0,
                        ..metadata(11, 1)
                    },
                )
                .unwrap();
        }
        table.set_reserved_norm(1, 0.125).unwrap();
        table.set_reserved_norm(1, 0.25).unwrap();
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        for norm in [f32::NAN, f32::INFINITY, -1.0] {
            assert_eq!(
                table.set_reserved_norm(1, norm),
                Err(NodeTableError::InvalidMetadata)
            );
            assert_eq!(view.get(1).unwrap().metadata().norm, 0.25);
        }
        unsafe {
            table.publish(1, LevelsRef::from_raw(101), 7).unwrap();
        }
        let record = view.get(1).unwrap();
        assert!(record.levels_ref().is_valid()); // publication acquire first
        let identity = record.identity_after_acquire();
        assert_eq!(
            identity,
            NodeIdentity {
                source_key: 11,
                incarnation: 1,
                publication: 7,
                norm: 0.25
            }
        );
        assert_eq!(
            table.set_reserved_norm(1, 0.5),
            Err(NodeTableError::WrongState)
        );
        assert_eq!(record.identity_after_acquire(), identity);
        table.invalidate(1).unwrap();
        assert_eq!(
            table.set_reserved_norm(1, 0.5),
            Err(NodeTableError::WrongState)
        );
        // Existing acquired ownership retains immutable scalar identity even
        // though the diagnostic state has changed and current ref is invalid.
        assert_eq!(record.metadata().state, NodeState::Retired);
        assert_eq!(record.identity_after_acquire(), identity);
        drop(guard);
        // Fixture owns no external row/preparation; all its reads have drained.
        unsafe {
            table.mark_reclaimed(1).unwrap();
        }
        assert_eq!(
            table.set_reserved_norm(1, 0.5),
            Err(NodeTableError::WrongState)
        );
    }

    #[test]
    fn reserved_norm_updates_current_root_and_guarded_published_identity_survives_growth() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        unsafe {
            table
                .initialize(
                    1,
                    NodeMetadata {
                        norm: 0.0,
                        ..metadata(11, 1)
                    },
                )
                .unwrap();
        }
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let old_reserved = view.get(1).unwrap();
        assert!(!old_reserved.levels_ref().is_valid());
        table.ensure_len(3).unwrap(); // copies the reserved record
        table.set_reserved_norm(1, 0.125).unwrap();
        assert_eq!(old_reserved.metadata().norm, 0.0); // safe old diagnostic copy
        assert_eq!(view.get(1).unwrap().metadata().norm, 0.125);
        unsafe {
            table.publish(1, LevelsRef::from_raw(101), 7).unwrap();
        }
        assert!(!old_reserved.levels_ref().is_valid()); // never falsely published
        let acquired = view.get(1).unwrap();
        assert!(acquired.levels_ref().is_valid());
        let identity = acquired.identity_after_acquire();
        table.ensure_len(7).unwrap(); // retains the now-published old copy
        table.invalidate(1).unwrap();
        assert!(!view.get(1).unwrap().levels_ref().is_valid());
        assert!(acquired.levels_ref().is_valid());
        assert_eq!(acquired.identity_after_acquire(), identity);
        assert_eq!(identity.norm, 0.125);
        table.assign_generation();
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 0);
        assert_eq!(acquired.identity_after_acquire(), identity);
        drop(guard);
        assert!(table.reclaim() > 0);
        assert_eq!(table.memory_usage().held_payload_bytes, 0);
    }

    #[test]
    fn transition_preflight_rejects_bad_state_bounds_and_stale_owner_without_changes() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        assert!(matches!(
            unsafe { table.preflight_publish(0, LevelsRef::from_raw(101), 7) },
            Err(NodeTableError::NodeZero)
        ));
        assert!(matches!(
            unsafe { table.preflight_publish(2, LevelsRef::from_raw(101), 7) },
            Err(NodeTableError::OutOfBounds)
        ));
        assert!(matches!(
            unsafe { table.preflight_publish(1, LevelsRef::from_raw(101), 7) },
            Err(NodeTableError::WrongState)
        ));
        unsafe {
            table.initialize(1, metadata(11, 1)).unwrap();
        }
        assert!(matches!(
            unsafe { table.preflight_publish(1, LevelsRef::INVALID, 7) },
            Err(NodeTableError::InvalidLevelsRef)
        ));
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        assert!(!view.get(1).unwrap().levels_ref().is_valid());
        assert_eq!(view.get(1).unwrap().metadata().state, NodeState::Reserved);
        let publish = unsafe {
            table
                .preflight_publish(1, LevelsRef::from_raw(101), 7)
                .unwrap()
        };
        assert_eq!(publish.old_levels_ref(), LevelsRef::INVALID);
        assert_eq!(publish.commit(), LevelsRef::INVALID);
        let before = view.get(1).unwrap().metadata();
        assert!(matches!(
            unsafe { table.preflight_publish(1, LevelsRef::from_raw(201), 8) },
            Err(NodeTableError::WrongState)
        ));
        assert!(matches!(
            unsafe {
                table.preflight_replace_levels(
                    1,
                    LevelsRef::from_raw(999),
                    LevelsRef::from_raw(201),
                )
            },
            Err(NodeTableError::StaleLevelsRef)
        ));
        assert!(matches!(
            table.preflight_invalidate(1, LevelsRef::INVALID),
            Err(NodeTableError::StaleLevelsRef)
        ));
        assert!(matches!(
            unsafe {
                table.preflight_replace_levels(
                    1,
                    LevelsRef::from_raw(101),
                    LevelsRef::from_raw(101),
                )
            },
            Err(NodeTableError::UnchangedLevelsRef)
        ));
        assert_eq!(view.get(1).unwrap().levels_ref().raw(), 101);
        assert_eq!(view.get(1).unwrap().metadata(), before);
    }

    #[test]
    fn dropping_each_preflight_transition_is_a_noop() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        unsafe {
            table.initialize(1, metadata(11, 1)).unwrap();
        }
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let reserved = view.get(1).unwrap().metadata();
        drop(unsafe {
            table
                .preflight_publish(1, LevelsRef::from_raw(101), 7)
                .unwrap()
        });
        assert_eq!(view.get(1).unwrap().metadata(), reserved);
        assert_eq!(view.get(1).unwrap().levels_ref(), LevelsRef::INVALID);
        // Cancellation can also be preflighted/aborted before any array exists.
        drop(table.preflight_invalidate(1, LevelsRef::INVALID).unwrap());
        assert_eq!(view.get(1).unwrap().metadata(), reserved);
        unsafe {
            table
                .preflight_publish(1, LevelsRef::from_raw(101), 7)
                .unwrap()
        }
        .commit();
        let published = view.get(1).unwrap().metadata();
        let replacement = unsafe {
            table
                .preflight_replace_levels(1, LevelsRef::from_raw(101), LevelsRef::from_raw(201))
                .unwrap()
        };
        assert_eq!(replacement.old_levels_ref().raw(), 101);
        drop(replacement);
        drop(
            table
                .preflight_invalidate(1, LevelsRef::from_raw(101))
                .unwrap(),
        );
        assert_eq!(view.get(1).unwrap().levels_ref().raw(), 101);
        assert_eq!(view.get(1).unwrap().metadata(), published);
    }

    #[test]
    fn transitions_target_current_root_after_growth_and_preserve_guarded_old_identity() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        add(&table, 1, 11, 1);
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let old_record = view.get(1).unwrap();
        let old_ref = old_record.levels_ref();
        let identity = old_record.identity_after_acquire();
        table.ensure_len(3).unwrap();
        let replace = unsafe {
            table
                .preflight_replace_levels(1, old_ref, LevelsRef::from_raw(201))
                .unwrap()
        };
        // Reads are lock-free while the transition retains writer exclusivity.
        assert_eq!(view.get(1).unwrap().levels_ref(), old_ref);
        assert_eq!(replace.old_levels_ref(), old_ref);
        assert_eq!(replace.commit(), old_ref);
        assert_eq!(old_record.levels_ref(), old_ref);
        assert_eq!(view.get(1).unwrap().levels_ref().raw(), 201);
        let invalidate = table
            .preflight_invalidate(1, LevelsRef::from_raw(201))
            .unwrap();
        assert_eq!(invalidate.commit().raw(), 201);
        assert_eq!(view.get(1).unwrap().levels_ref(), LevelsRef::INVALID);
        assert_eq!(view.get(1).unwrap().metadata().state, NodeState::Retired);
        assert_eq!(old_record.levels_ref(), old_ref);
        assert_eq!(old_record.identity_after_acquire(), identity);
        table.assign_generation();
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 0);
        assert_eq!(old_record.identity_after_acquire(), identity);
        drop(guard);
        assert!(table.reclaim() > 0);
    }

    #[test]
    fn transition_commits_need_no_table_allocation_or_fallible_growth_work() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        unsafe {
            table.initialize(1, metadata(11, 1)).unwrap();
        }
        table.writer.lock().fail_next_allocation = true;
        let before = table.memory_usage();
        unsafe {
            table
                .preflight_publish(1, LevelsRef::from_raw(101), 7)
                .unwrap()
        }
        .commit();
        unsafe {
            table
                .preflight_replace_levels(1, LevelsRef::from_raw(101), LevelsRef::from_raw(201))
                .unwrap()
        }
        .commit();
        table
            .preflight_invalidate(1, LevelsRef::from_raw(201))
            .unwrap()
            .commit();
        let after = table.memory_usage();
        assert_eq!(after.current_payload_bytes, before.current_payload_bytes);
        assert_eq!(
            after.hold_metadata_capacity_bytes,
            before.hold_metadata_capacity_bytes
        );
        assert_eq!(after.held_payload_bytes, before.held_payload_bytes);
        assert_eq!(after.retired_node_bytes, NODE_BYTES);
        // Failure injection was not consumed by any preflight or commit; only
        // the subsequent actual growth attempts an allocation and can fail.
        assert_eq!(table.ensure_len(3), Err(NodeTableError::AllocationFailed));
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        assert_eq!(view.get(1).unwrap().levels_ref(), LevelsRef::INVALID);
        assert_eq!(view.get(1).unwrap().metadata().publication, 7);
    }

    #[test]
    fn pending_and_equal_generation_holds_cannot_be_reclaimed() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(3).unwrap();
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 0); // still unassigned
        table.assign_generation();
        assert_eq!(table.reclaim(), 0); // retirement == oldest
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 2 * NODE_BYTES);
    }

    #[test]
    fn one_large_retirement_can_grow_but_pinned_copies_remain_bounded() {
        let d = domain();
        let mut configured = limits();
        configured.max_held_bytes = 1;
        let table = NodeTable::new(d.clone(), configured).unwrap();
        let guard = d.take_guard().unwrap();
        table.ensure_len(3).unwrap();
        table.assign_generation();
        d.inc_generation().unwrap();
        assert_eq!(table.reclaim(), 0);
        assert_eq!(table.ensure_len(7), Err(NodeTableError::HeldLimit));
        drop(guard);
        assert!(table.reclaim() > 0);
        table.ensure_len(7).unwrap();
        assert_eq!(table.memory_usage().len, 7);
    }

    #[test]
    fn bounded_growth_failure_preserves_pointer_bound_and_records() {
        let d = domain();
        let mut configured = limits();
        configured.max_held_allocations = 1;
        let table = NodeTable::new(d.clone(), configured).unwrap();
        add(&table, 1, 11, 1);
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        let first_pointer = view.get(1).unwrap().record as *const NodeRecord;
        table.writer.lock().fail_next_allocation = true;
        assert_eq!(table.ensure_len(3), Err(NodeTableError::AllocationFailed));
        assert_eq!(view.get(1).unwrap().record as *const NodeRecord, first_pointer);
        assert!(view.get(2).is_none());
        assert_eq!(view.get(1).unwrap().metadata().source_key, 11);
        table.ensure_len(3).unwrap();
        assert_eq!(table.ensure_len(7), Err(NodeTableError::HeldLimit));
        assert_eq!(table.memory_usage().len, 3);
        assert_eq!(table.ensure_len(129), Err(NodeTableError::CapacityLimit));
        table.assign_generation();
        d.inc_generation().unwrap();
        drop(guard);
        table.reclaim();
        table.ensure_len(7).unwrap();
        assert_eq!(table.memory_usage().len, 7);
    }

    #[test]
    fn payload_budget_and_generation_exhaustion_fail_safely() {
        let d = domain();
        let mut configured = limits();
        configured.max_payload_bytes = 2 * NODE_BYTES;
        let table = NodeTable::new(d, configured).unwrap();
        assert_eq!(table.ensure_len(3), Err(NodeTableError::PayloadLimit));
        assert_eq!(table.memory_usage().len, 1);
        let d = GenerationHandler::new(GenerationLimits {
            max_generation: 0,
            ..GenerationLimits::default()
        })
        .unwrap();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(3).unwrap();
        table.assign_generation();
        assert_eq!(d.inc_generation(), Err(GenerationError::Exhausted));
        assert_eq!(table.reclaim(), 0);
        assert_eq!(table.memory_usage().held_payload_bytes, 2 * NODE_BYTES);
    }

    #[test]
    fn fixed_id_churn_stays_dense_after_external_retirement() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        for incarnation in 1..=100 {
            add(&table, 1, u64::from(incarnation), incarnation);
            let guard = d.take_guard().unwrap();
            let view = table.read(&guard).unwrap();
            assert_eq!(view.get(1).unwrap().metadata().incarnation, incarnation);
            assert!(table.invalidate(1).unwrap().is_valid());
            assert!(!view.get(1).unwrap().levels_ref().is_valid());
            assert_eq!(
                view.get(1).unwrap().metadata().source_key,
                u64::from(incarnation)
            );
            assert_eq!(table.memory_usage().retired_node_bytes, NODE_BYTES);
            // Fixture has no source owner or other prepare/query; guard drain
            // provides its external ID-retirement proof before reuse.
            drop(guard);
            unsafe {
                table.mark_reclaimed(1).unwrap();
            }
            assert_eq!(table.memory_usage().vacant_reusable_bytes, NODE_BYTES);
        }
        let usage = table.memory_usage();
        assert_eq!(usage.current_payload_bytes, 2 * NODE_BYTES);
        assert_eq!(usage.held_payload_bytes, 0);
        assert_eq!(
            usage.accounted_bytes,
            usage.current_payload_bytes
                + usage.table_struct_bytes
                + usage.hold_metadata_capacity_bytes
        );
    }

    #[test]
    fn state_and_metadata_failures_do_not_publish_partial_nodes() {
        let d = domain();
        let table = NodeTable::new(d.clone(), limits()).unwrap();
        table.ensure_len(2).unwrap();
        let bad = NodeMetadata {
            norm: f32::NAN,
            ..metadata(1, 1)
        };
        assert_eq!(
            unsafe { table.initialize(1, bad) },
            Err(NodeTableError::InvalidMetadata)
        );
        unsafe {
            table.initialize(1, metadata(1, 1)).unwrap();
        }
        assert_eq!(
            unsafe { table.initialize(1, metadata(2, 2)) },
            Err(NodeTableError::WrongState)
        );
        assert_eq!(
            unsafe { table.publish(1, LevelsRef::INVALID, 7) },
            Err(NodeTableError::InvalidLevelsRef)
        );
        let guard = d.take_guard().unwrap();
        let view = table.read(&guard).unwrap();
        assert_eq!(view.get(1).unwrap().metadata().state, NodeState::Reserved);
        assert!(!view.get(1).unwrap().levels_ref().is_valid());
        // Cancelling an unpublished reservation returns no array to retire.
        assert_eq!(table.invalidate(1).unwrap(), LevelsRef::INVALID);
        assert_eq!(table.invalidate(1), Err(NodeTableError::WrongState));
    }

    #[test]
    fn concurrent_reader_sees_initialized_rows_across_many_growths() {
        let d = domain();
        let table = Arc::new(NodeTable::new(d.clone(), limits()).unwrap());
        let reader_table = table.clone();
        let reader_domain = d.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let reader = std::thread::spawn(move || {
            let guard = reader_domain.take_guard().unwrap();
            let view = reader_table.read(&guard).unwrap();
            ready_tx.send(()).unwrap();
            for _ in 0..5000 {
                for id in 1..64 {
                    if let Some(row) = view.get(id) {
                        if row.levels_ref().is_valid() {
                            let meta = row.metadata();
                            assert_eq!(meta.source_key, u64::from(id) * 10);
                            assert_eq!(meta.incarnation, 1);
                            assert_eq!(meta.publication, 7);
                            assert_eq!(meta.norm, 2.5);
                        }
                    }
                }
            }
        });
        // Writer cannot finish before the reader has pinned and acquired its
        // view. Bounded wait and a finite reader loop avoid stranded workers if
        // either side fails; old_slice_survives_growth supplies the deterministic
        // retained-slice assertion independently of thread scheduling.
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        for id in 1..64 {
            add(&table, id, u64::from(id) * 10, 1);
            table.assign_generation();
            d.inc_generation().unwrap();
            table.reclaim();
        }
        reader.join().unwrap();
        table.assign_generation();
        d.inc_generation().unwrap();
        table.reclaim();
        assert_eq!(table.memory_usage().held_payload_bytes, 0);
    }
}
