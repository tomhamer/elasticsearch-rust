//! Storage and source-ID lifetime coordinator for the mutable graph.
//!
//! Queries and prepared inserts pin the SAME generation domain as node-table
//! growth, levels, links and source-ID retirements. A guard only retains the
//! domain's hold cells: snapshots/prepared operations also own this storage Arc.
//! This is current-topology memory safety, not a historical ES/PIT read view.
//!
//! One outer writer serializes graph publication and all maintenance. Reservation
//! admission reserves physical cancellation-queue capacity before issuing a key
//! or ID. Removing a queue entry into a source-acknowledgment token does not free
//! its logical retirement credit. Only token drop returns the ID to the free
//! list, after the caller has removed the matching source descriptor. Older
//! independently pinned source roots must continue owning their immutable rows.

use super::array_store::{ArrayStore, ArrayStoreConfig, ArrayStoreStats, LevelsRef, StoreError};
use super::generation::{GenerationError, GenerationHandler, GenerationLimits};
use super::node_table::{NodeMetadata, NodeState, NodeTable, NodeTableLimits};
use super::{MutableHnswLimits, MutableHnswStats, NodeHandle, failure, internal_id};
use crate::error::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

#[derive(Clone, Copy)]
struct RetiredNode {
    handle: NodeHandle,
    key: u64,
    // None remains held until assigned by the common maintenance coordinator.
    generation: Option<u64>,
}

#[derive(Clone, Copy)]
enum MaintenanceFailure {
    Store(StoreError),
    Generation(GenerationError),
}

/// Diagnostic storage payload/capacities, excluding allocator-private overhead.
/// `accounted_bytes` includes metadata; payload and metadata are disjoint.
/// Array-store payload includes per-buffer Allocation headers and large-array
/// descriptor slabs, not just element values. Used/dead/held are subcategories.
#[derive(Clone, Copy, Debug, Default)]
pub struct MutableArrayMemory {
    pub allocated_payload_bytes: usize,
    pub used_payload_bytes: usize,
    pub dead_payload_bytes: usize,
    pub held_payload_bytes: usize,
    pub metadata_bytes: usize,
    pub accounted_bytes: usize,
}
impl From<ArrayStoreStats> for MutableArrayMemory {
    fn from(stats: ArrayStoreStats) -> Self {
        Self {
            allocated_payload_bytes: stats.allocated_bytes.saturating_sub(stats.metadata_bytes),
            used_payload_bytes: stats.used_bytes,
            dead_payload_bytes: stats.dead_bytes,
            held_payload_bytes: stats.held_bytes,
            metadata_bytes: stats.metadata_bytes,
            accounted_bytes: stats.allocated_bytes,
        }
    }
}

/// Current-topology representation accounting, NOT process RSS. Arrays include
/// growth/retirement holds; node-table bytes include current and old allocations.
/// The total counts embedded headers once and includes known HashMap capacity
/// payload as a LOWER BOUND (std does not expose bucket/control allocation).
/// Allocator-private overhead, Arc control blocks, source rows, caller-owned
/// snapshots, prepared proposals and query scratch are explicitly excluded.
/// Outstanding ACK token structs are counted, but their caller-owned container's
/// spare capacity is not exposed and therefore is also excluded.
#[derive(Clone, Copy, Debug, Default)]
pub struct MutableStorageMemory {
    pub levels: MutableArrayMemory,
    pub links: MutableArrayMemory,
    pub node_current_payload_bytes: usize,
    pub node_held_payload_bytes: usize,
    pub node_accounted_bytes: usize,
    pub generation_accounted_bytes: usize,
    pub generation_cells: usize,
    pub guard_references: u64,
    pub free_slot_capacity: usize,
    pub retirement_capacity: usize,
    pub pending_reservation_capacity: usize,
    pub visibility_retirement_capacity: usize,
    pub free_slot_payload_bytes: usize,
    pub retirement_payload_bytes: usize,
    pub hash_capacity_payload_bytes_lower_bound: usize,
    pub outstanding_ack_tokens: usize,
    pub outstanding_ack_payload_bytes: usize,
    pub graph_struct_bytes: usize,
    pub accounted_bytes_lower_bound: usize,
}

pub(super) struct Writer {
    pub(super) watermark: Option<u64>,
    pub(super) pending: HashMap<u64, NodeHandle>,
    pub(super) free: Vec<u32>,
    pub(super) visibility_retained: HashMap<NodeHandle, u64>,
    pub(super) exhausted: usize,
    // Never mem::take this queue: spare capacity belongs to outstanding permits.
    retired: Vec<RetiredNode>,
    // Entries before this cursor are already assigned. Avoid rescanning a long
    // held-reader queue on every otherwise unrelated graph insertion.
    unassigned_from: usize,
    outstanding_acks: usize,
    maintenance_pending: bool,
    maintenance_batches: usize,
    blocked: Option<MaintenanceFailure>,
    /// A chunked graph capture in progress (`MutableHnsw::capture_snapshot`).
    pub(super) capture_log: Option<CaptureLog>,
    pending_memory:Option<crate::admission::Reservation>,
    free_memory:Option<crate::admission::Reservation>,
    visibility_memory:Option<crate::admission::Reservation>,
    retired_memory:Option<crate::admission::Reservation>,
}

/// Pre-images for a chunked capture: internal ids at or above `cursor` (and
/// below the captured `len`) not copied yet; the first topology change of
/// such a node records its levels/links as of the capture point.
pub(super) struct CaptureLog {
    pub(super) cursor: u32,
    pub(super) len: u32,
    pub(super) images: HashMap<u32, (LevelsRef, crate::admission::Allocation<Vec<super::array_store::LinksRef>>)>,
    pub(super) memory:Option<crate::admission::Reservation>,
}
impl CaptureLog {
    pub(super) fn reserve_image(&mut self)->Result<()> {
        let needed=self.images.len().checked_add(1).ok_or_else(||failure("capture image capacity"))?;
        reserve_map(&mut self.images,&mut self.memory,needed,"graph capture pre-image map")
    }
}

fn reserve_vec<T>(values:&mut Vec<T>,memory:&mut Option<crate::admission::Reservation>,needed:usize,label:&str)->Result<()> {
    if needed<=values.capacity(){return Ok(())}
    let capacity=needed.max(values.capacity().saturating_mul(2));
    let bytes=capacity.checked_mul(std::mem::size_of::<T>()).ok_or_else(||failure("graph registry size overflow"))?;
    let next_memory=crate::admission::shared().reserve(bytes as u64,label)?;
    let mut next=Vec::new();next.try_reserve_exact(capacity).map_err(|_|failure("graph registry allocation"))?;
    next.append(values);*values=next;*memory=Some(next_memory);Ok(())
}
fn reserve_map<K:Eq+std::hash::Hash,V>(values:&mut HashMap<K,V>,memory:&mut Option<crate::admission::Reservation>,needed:usize,label:&str)->Result<()> {
    if needed<=values.capacity(){return Ok(())}
    let capacity=needed.max(values.capacity().saturating_mul(2));
    let buckets=capacity.max(1).checked_next_power_of_two().and_then(|v|v.checked_mul(2)).ok_or_else(||failure("graph map capacity overflow"))?;
    let bytes=buckets.checked_mul(std::mem::size_of::<(K,V)>()+1).and_then(|v|v.checked_add(32)).ok_or_else(||failure("graph map bytes overflow"))?;
    let next_memory=crate::admission::shared().reserve(bytes as u64,label)?;
    let mut next=HashMap::new();next.try_reserve(capacity).map_err(|_|failure("graph map allocation"))?;
    next.extend(values.drain());*values=next;*memory=Some(next_memory);Ok(())
}
impl Writer {
    pub(super) fn reserve_slots(&mut self,needed:usize)->Result<()> {
        reserve_vec(&mut self.free,&mut self.free_memory,needed,"graph free slot registry")?;
        reserve_map(&mut self.visibility_retained,&mut self.visibility_memory,needed,"graph visibility retirement registry")
    }
    fn retirement_obligations(&self) -> usize {
        self.retired.len() + self.outstanding_acks + self.pending.len()
    }
}

pub(super) struct GraphStorage {
    pub(super) domain: GenerationHandler,
    pub(super) nodes: NodeTable,
    pub(super) levels: ArrayStore<AtomicU32, 22>,
    pub(super) links: ArrayStore<u32, 20>,
    pub(super) writer: Mutex<Writer>,
    pub(super) entry: AtomicU64,
    pub(super) publication: AtomicU64,
    pub(super) active: AtomicU32,
    /// Published/searchable node-ID bound, including zero; reservations do not
    /// grow it. The node table and allocated count separately cover reservations.
    pub(super) live_bound: AtomicU32,
    /// Public slot high-water count; never shrinks and is preserved by the codec.
    pub(super) allocated: AtomicU32,
    pub(super) limits: MutableHnswLimits,
    poisoned: AtomicBool,
    /// Bumped by every published topology change (the two publication
    /// choke points: `publish_overlay` for insert / remove / links /
    /// relocation, `relocate_levels` for compaction): an unchanged value
    /// means an unchanged graph (see `MutableHnsw::mutation_generation`).
    mutations: AtomicU64,
    /// Odd while a topology publication (`publish_overlay`) is storing:
    /// a reader that sees the same even value before and after reading
    /// links read them as of no publication in progress (a seqlock over
    /// visibility, see `MutableHnswSnapshot::quiet`).
    topology_seq: AtomicU64,
}

/// Generation advances deferred because every hold cell was pinned (see
/// `GraphStorage::maintain`); warned about at most once a minute.
fn note_deferred_generation() {
    static DEFERRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static LAST_WARN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = DEFERRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let last = LAST_WARN.load(std::sync::atomic::Ordering::Relaxed);
    if now >= last + 60 && LAST_WARN.compare_exchange(last, now, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
        tracing::warn!(deferred_advances = count, "mutable hnsw: generation hold cells all pinned by an old reader; reclamation deferred, writes continue");
    }
}

impl GraphStorage {
    pub(super) fn new(limits: MutableHnswLimits) -> Result<Self> {
        Self::new_with_generation_limits(limits, GenerationLimits::default())
    }

    fn new_with_generation_limits(
        limits: MutableHnswLimits,
        generation_limits: GenerationLimits,
    ) -> Result<Self> {
        if limits.max_slots == 0
            || limits.max_slots == u32::MAX
            || limits.max_pending == 0
            || limits.max_retired == 0
            || limits.max_retired_adjacency_bytes == 0
        {
            return Err(failure("invalid mutable storage limits"));
        }
        let domain =
            GenerationHandler::new(generation_limits).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let nodes = NodeTable::new(
            domain.clone(),
            NodeTableLimits {
                max_slots: limits.max_slots + 1,
                initial_capacity: (limits.max_slots + 1).min(16),
                ..NodeTableLimits::default()
            },
        )
        .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let levels = ArrayStore::new(domain.clone(), ArrayStoreConfig::hnsw::<22>(16))
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let links = ArrayStore::new(domain.clone(), ArrayStoreConfig::hnsw::<20>(193))
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        Ok(Self {
            domain,
            nodes,
            levels,
            links,
            writer: Mutex::new(Writer {
                watermark: None,
                pending: HashMap::new(),
                free: Vec::new(),
                visibility_retained: HashMap::new(),
                exhausted: 0,
                retired: Vec::new(),
                unassigned_from: 0,
                outstanding_acks: 0,
                maintenance_pending: false,
                maintenance_batches: 0,
                blocked: None,
                capture_log: None,
                pending_memory:None,free_memory:None,visibility_memory:None,retired_memory:None,
            }),
            entry: AtomicU64::new(0),
            publication: AtomicU64::new(0),
            active: AtomicU32::new(0),
            live_bound: AtomicU32::new(1),
            allocated: AtomicU32::new(0),
            limits,
            poisoned: AtomicBool::new(false),
            mutations: AtomicU64::new(0),
            topology_seq: AtomicU64::new(0),
        })
    }

    /// Safe from a publication-unwind guard while the outer writer is held.
    pub(super) fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// Record a published topology change (call under the writer).
    /// Start storing a topology publication (graph writer only).
    pub(super) fn begin_topology_change(&self) {
        let seq = self.topology_seq.load(Ordering::Relaxed);
        self.topology_seq.store(seq | 1, Ordering::Relaxed);
        // Readers that see any later store see the odd value after it.
        std::sync::atomic::fence(Ordering::Release);
    }
    /// Finish it: the next even value.
    pub(super) fn end_topology_change(&self) {
        let seq = self.topology_seq.load(Ordering::Relaxed);
        self.topology_seq.store((seq | 1) + 1, Ordering::Release);
    }
    #[inline]
    pub(super) fn topology_seq(&self) -> u64 {
        self.topology_seq.load(Ordering::Acquire)
    }
    #[inline]
    pub(super) fn topology_seq_after_fence(&self) -> u64 {
        std::sync::atomic::fence(Ordering::Acquire);
        self.topology_seq.load(Ordering::Relaxed)
    }

    pub(super) fn note_mutation(&self) {
        self.mutations.fetch_add(1, Ordering::Release);
    }

    pub(super) fn mutation_generation(&self) -> u64 {
        self.mutations.load(Ordering::Acquire)
    }

    pub(super) fn check_readable(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(failure("mutable graph poisoned by incomplete publication"))
        } else {
            Ok(())
        }
    }

    /// Retry pending maintenance BEFORE a new mutation. A previous committed
    /// mutation remains committed even if generation assignment/advance failed.
    pub(super) fn admit_write(&self, writer: &mut Writer) -> Result<()> {
        self.check_readable()?;
        if writer.maintenance_batches == 0 || writer.blocked.is_some()
            || self.check_retired_adjacency_budget().is_err()
        {
            self.maintain(writer);
        }
        if let Some(error) = writer.blocked {
            return Err(match error {
                MaintenanceFailure::Store(e) => super::failure_error(e),
                MaintenanceFailure::Generation(e) => {
                    failure(&format!("generation maintenance: {e}"))
                }
            });
        }
        self.check_retired_adjacency_budget()
    }

    pub(super) fn check_retired_adjacency_budget(&self) -> Result<()> {
        let held = self
            .levels
            .held_bytes_upper_bound()
            .saturating_add(self.links.held_bytes_upper_bound());
        if held >= self.limits.max_retired_adjacency_bytes {
            Err(failure("mutable retired adjacency budget exhausted"))
        } else {
            Ok(())
        }
    }

    /// Reserve one possible retirement while retaining all cancellation credits
    /// already promised to pending reservations. This consumes no logical credit
    /// until commit, so an overlay/preflight error needs no refund operation.
    pub(super) fn reserve_retirement(&self, writer: &mut Writer) -> Result<()> {
        if writer.retirement_obligations() >= self.limits.max_retired {
            return Err(failure("mutable source retirement budget exhausted"));
        }
        let additional = writer
            .pending
            .len()
            .checked_add(1)
            .ok_or_else(|| failure("mutable cancellation capacity overflow"))?;
        let needed=writer.retired.len().checked_add(additional).ok_or_else(||failure("graph retirement capacity overflow"))?;
        reserve_vec(&mut writer.retired,&mut writer.retired_memory,needed,"graph cancellation registry")?;
        Ok(())
    }

    pub(super) fn reserve_insert(self: &Arc<Self>, key: u64) -> Result<InsertReservation> {
        self.reserve_insert_with(key, false)
    }

    /// [`Self::reserve_insert`] of a key at or below the watermark that the
    /// graph does not hold (the caller checks its identities): an open-time
    /// catch-up of a committed point whose admission was in flight when the
    /// checkpoint was captured. Such admissions are left out of the captured
    /// graph, but their keys can lie below those of later admissions that
    /// completed first, and the checkpoint's watermark covers them all.
    /// The watermark stays the highest key issued.
    pub(super) fn reserve_insert_backfill(self: &Arc<Self>, key: u64) -> Result<InsertReservation> {
        self.reserve_insert_with(key, true)
    }

    fn reserve_insert_with(self: &Arc<Self>, key: u64, backfill: bool) -> Result<InsertReservation> {
        let mut writer = self.writer.lock();
        self.reserve_insert_locked(&mut writer, key, backfill)
    }

    /// [`Self::reserve_insert`] under a writer the caller holds (a batch of
    /// reservations under one hold). A reservation dropped while that
    /// writer is held would deadlock (its drop takes the writer): the
    /// caller drops them only after releasing it.
    pub(super) fn reserve_insert_locked(self: &Arc<Self>, writer: &mut Writer, key: u64, backfill: bool) -> Result<InsertReservation> {
        self.admit_write(writer)?;
        if writer.watermark.is_some_and(|watermark| key <= watermark) {
            if !backfill {
                return Err(failure("mutable keys must increase monotonically"));
            }
            if writer.pending.contains_key(&key) {
                return Err(failure("mutable backfill key is already reserved"));
            }
        }
        if writer.pending.len() >= self.limits.max_pending {
            return Err(failure("mutable pending reservation budget exhausted"));
        }
        self.reserve_retirement(writer)?;
        let pending_needed=writer.pending.len().checked_add(1).ok_or_else(||failure("graph pending capacity overflow"))?;
        {let writer=&mut *writer;reserve_map(&mut writer.pending,&mut writer.pending_memory,pending_needed,"graph pending registry")?;};
        let allocated = self.allocated.load(Ordering::Relaxed);
        let (slot, incarnation, new_slot) = if let Some(&slot) = writer.free.last() {
            let metadata = self
                .nodes
                .writer_metadata(slot + 1)
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
            if metadata.state != NodeState::Vacant {
                self.poison();
                return Err(failure("mutable free slot is not vacant"));
            }
            let incarnation = metadata
                .incarnation
                .checked_add(1)
                .ok_or_else(|| failure("mutable slot incarnation exhausted"))?;
            (slot, incarnation, false)
        } else {
            if allocated >= self.limits.max_slots {
                return Err(failure("mutable slot limit reached"));
            }
            // Every admitted slot can eventually return via a token. Reserving
            // against peak allocated slots covers tokens not currently in free.
            let needed = allocated as usize + 1;
            writer.reserve_slots(needed)?;
            self.nodes
                .ensure_len(allocated + 2)
                .map_err(crate::vector::hnsw::mutable::failure_error)?;
            writer.maintenance_pending = true;
            (allocated, 1, true)
        };
        let handle = NodeHandle::new(slot, incarnation);
        // SAFETY: new slot, or a vacant slot returned only after graph readers
        // drained and the external source descriptor was acknowledged.
        unsafe {
            self.nodes.initialize(
                slot + 1,
                NodeMetadata {
                    source_key: key,
                    incarnation,
                    publication: 0,
                    norm: 0.0,
                    state: NodeState::Reserved,
                },
            )
        }
        .map_err(crate::vector::hnsw::mutable::failure_error)?;
        // All allocations and validation precede issuing the key or handle.
        if new_slot {
            self.allocated.store(allocated + 1, Ordering::Release);
        } else {
            writer.free.pop();
        }
        writer.pending.insert(key, handle);
        writer.watermark = Some(writer.watermark.map_or(key, |watermark| watermark.max(key)));
        Ok(InsertReservation {
            control: self.clone(),
            key,
            handle,
            pending: true,
        })
    }

    /// Caller already invalidated the node and admitted capacity under this same
    /// writer. Publication is complete; this path must not fail or allocate.
    pub(super) fn retire_committed(&self, writer: &mut Writer, handle: NodeHandle, key: u64) {
        if writer.retired.len() == writer.retired.capacity() {
            // Broken admission credit must not allocate after publication or
            // allow reuse of an unqueued source identity. Fail closed.
            self.poison();
            return;
        }
        writer.retired.push(RetiredNode {
            handle,
            key,
            generation: None,
        });
        writer.maintenance_pending = true;
        self.shrink_live_bound();
    }

    fn shrink_live_bound(&self) {
        let mut bound = self.live_bound.load(Ordering::Relaxed);
        while bound > 1 {
            match self.nodes.writer_metadata(bound - 1) {
                Ok(metadata) if metadata.state != NodeState::Published => bound -= 1,
                Ok(_) => break,
                Err(_) => {
                    self.poison();
                    return;
                }
            }
        }
        self.live_bound.store(bound, Ordering::Release);
    }

    /// Give the node table its dense publication copy, which a quantized
    /// field's code plane traversal reads (`NodeTable::enable_dense`); the
    /// replaced allocation is retired like a grown one.
    pub(super) fn enable_dense_publications(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        self.nodes.enable_dense().map_err(crate::vector::hnsw::mutable::failure_error)?;
        // A populated graph (restored before its owner): every published
        // node's level-0 links word, from its levels.
        {
            let guard = self.domain.take_guard().map_err(crate::vector::hnsw::mutable::failure_error)?;
            let levels = self.levels.view(&guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
            // SAFETY: current owners read under the graph writer and this guard.
            self.nodes.fill_links0(|reference| unsafe { levels.get(reference) }.ok()
                .and_then(|refs| refs.first()).map_or(0, |links| links.load(Ordering::Acquire)));
        }
        self.finish_mutation(&mut writer);
        Ok(())
    }

    /// After a successful graph mutation, record maintenance failure as write
    /// backpressure. Never turn a published insertion/deletion into an error
    /// which a caller might retry as if the mutation had not happened.
    pub(super) fn finish_mutation(&self, writer: &mut Writer) {
        writer.maintenance_pending = true;
        if writer.maintenance_batches == 0 {
            self.maintain(writer);
        }
    }

    pub(super) fn begin_maintenance_batch(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        self.admit_write(&mut writer)?;
        writer.maintenance_batches = writer.maintenance_batches.checked_add(1)
            .ok_or_else(|| failure("maintenance batch count exhausted"))?;
        Ok(())
    }

    pub(super) fn end_maintenance_batch(&self) {
        let mut writer = self.writer.lock();
        debug_assert!(writer.maintenance_batches > 0);
        writer.maintenance_batches -= 1;
        if writer.maintenance_batches == 0 {
            // Publication already happened. As for individual mutation
            // maintenance, failure applies backpressure to the NEXT write;
            // it must not invite replay of a committed operation.
            self.maintain(&mut writer);
        }
    }

    /// Call before staging arrays, including a stage which may abort. Aborted
    /// unpublished arrays still enter generation holds and need maintenance.
    pub(super) fn mark_maintenance_pending(&self, writer: &mut Writer) {
        writer.maintenance_pending = true;
    }

    fn maintain(&self, writer: &mut Writer) {
        if writer.maintenance_pending {
            // Partial assignment is harmless: all stores use this same current
            // generation and advancement happens only after every assignment.
            if let Err(error) = self.levels.writer().assign_generation() {
                writer.blocked = Some(MaintenanceFailure::Store(error));
                return;
            }
            if let Err(error) = self.links.writer().assign_generation() {
                writer.blocked = Some(MaintenanceFailure::Store(error));
                return;
            }
            self.nodes.assign_generation();
            let generation = self.domain.current_generation();
            let unassigned_from = writer.unassigned_from;
            for retired in &mut writer.retired[unassigned_from..] {
                if retired.generation.is_none() {
                    retired.generation = Some(generation);
                }
            }
            writer.unassigned_from = writer.retired.len();
            match self.domain.inc_generation() {
                Ok(_) => {
                    writer.maintenance_pending = false;
                    writer.blocked = None;
                }
                // Every hold cell is pinned behind an old reader (a flush's
                // graph checkpoint streaming for a minute, a PIT): the
                // advance is deferred, not failed. What was retired keeps
                // the current generation and stays held (each store's queue
                // and the node table keep equal-generation entries, see
                // `NodeTable::reclaim`), reclaimed once a later maintenance
                // advances past it; writes go on. Blocking them here failed
                // every write until the reader let go (14.9% of a 41.5M-doc
                // int8 Rally ingest, with no error in the server log).
                Err(GenerationError::CellLimit) => {
                    writer.blocked = None;
                    note_deferred_generation();
                }
                Err(error) => {
                    writer.blocked = Some(MaintenanceFailure::Generation(error));
                    return;
                }
            }
        }
        // Each store derives its strict oldest-used bound from this domain.
        self.domain.update_oldest_used_generation();
        self.levels.writer().reclaim();
        self.links.writer().reclaim();
        self.nodes.reclaim();
    }

    pub(super) fn collect_reclaimed(self: &Arc<Self>) -> Vec<ReclaimedNode> {
        let mut writer = self.writer.lock();
        if self.check_readable().is_err() {
            return Vec::new();
        }
        self.maintain(&mut writer);
        let oldest = self.domain.update_oldest_used_generation();
        let count = writer
            .retired
            .iter()
            .filter(|retired| {
                retired
                    .generation
                    .is_some_and(|generation| generation < oldest)
            })
            .count();
        let mut reclaimed = Vec::new();
        // Allocation failure leaves every retirement queued and owned.
        if reclaimed.try_reserve_exact(count).is_err() {
            return reclaimed;
        }
        let mut index = 0;
        while index < writer.retired.len() {
            if writer.retired[index]
                .generation
                .is_some_and(|generation| generation < oldest)
            {
                let retired = writer.retired.swap_remove(index);
                // swap_remove may move an unassigned tail into this position
                // after a failed maintenance attempt. Conservatively rescan it.
                writer.unassigned_from = writer.unassigned_from.min(index);
                writer.outstanding_acks += 1;
                reclaimed.push(ReclaimedNode {
                    control: self.clone(),
                    retired,
                });
            } else {
                index += 1;
            }
        }
        reclaimed
    }

    pub(super) fn owns_reclaimed(self: &Arc<Self>, token: &ReclaimedNode) -> bool {
        Arc::ptr_eq(self, &token.control)
    }

    /// Cheap drain predicate; no array/row scans and no generation advancement.
    /// A false result concerns retired ownership only, not pending insertions,
    /// total packed-store capacity or independently retained source roots.
    /// Poison or blocked maintenance fails closed rather than claiming a drain.
    pub(super) fn reclamation_pending(&self) -> bool {
        let writer = self.writer.lock();
        self.poisoned.load(Ordering::Acquire)
            || writer.blocked.is_some()
            || !writer.retired.is_empty()
            || writer.outstanding_acks != 0
            || self.levels.held_bytes_upper_bound() != 0
            || self.links.held_bytes_upper_bound() != 0
            || self.nodes.held_payload_bytes() != 0
    }

    pub(super) fn stats(&self) -> MutableHnswStats {
        let writer = self.writer.lock();
        let links = self.links.stats();
        MutableHnswStats {
            allocated_slots: self.allocated.load(Ordering::Acquire),
            active_nodes: self.active.load(Ordering::Acquire),
            pending_reservations: writer.pending.len(),
            reusable_slots: writer.free.len(),
            retired_slots: writer.retired.len() + writer.outstanding_acks,
            exhausted_slots: writer.exhausted,
            visibility_retained: writer.visibility_retained.len(),
            issued_key_watermark: writer.watermark,
            // Preserve the legacy neighbor-only scope: an isolated published
            // node has zero live adjacency even though it owns a levels row.
            // Held bytes now include link-slab growth/capacity and headers, so
            // they are not byte-identical to the old per-Vec payload counter.
            // memory_usage reports levels and the whole representation.
            live_adjacency_bytes: links.used_bytes,
            retired_adjacency_bytes: links.held_bytes,
            neighbor_size: std::mem::size_of::<u32>(),
        }
    }

    /// Diagnostic O(number of stored arrays) scan; never a write-admission path.
    pub(super) fn memory_usage(&self) -> MutableStorageMemory {
        let writer = self.writer.lock();
        let levels = MutableArrayMemory::from(self.levels.stats());
        let links = MutableArrayMemory::from(self.links.stats());
        let nodes = self.nodes.memory_usage();
        let generation = self.domain.stats();
        let free_bytes = writer.free.capacity() * size_of::<u32>();
        let retired_bytes = writer.retired.capacity() * size_of::<RetiredNode>();
        let hash_bytes = writer.pending.capacity() * size_of::<(u64, NodeHandle)>()
            + writer.visibility_retained.capacity() * size_of::<(NodeHandle, u64)>();
        let ack_bytes = writer.outstanding_acks * size_of::<ReclaimedNode>();
        // GraphStorage already embeds the array/node objects and Writer's
        // collection headers. Domain core is separately Arc-owned. Tokens are
        // caller-owned but part of the still-outstanding retirement obligation.
        let graph_bytes = size_of::<Self>();
        let total = graph_bytes
            + levels
                .accounted_bytes
                .saturating_sub(size_of::<ArrayStore<AtomicU32, 22>>())
            + links
                .accounted_bytes
                .saturating_sub(size_of::<ArrayStore<u32, 20>>())
            + nodes.accounted_bytes.saturating_sub(size_of::<NodeTable>())
            + generation.accounted_bytes
            + free_bytes
            + retired_bytes
            + hash_bytes
            + ack_bytes;
        MutableStorageMemory {
            levels,
            links,
            node_current_payload_bytes: nodes.current_payload_bytes,
            node_held_payload_bytes: nodes.held_payload_bytes,
            node_accounted_bytes: nodes.accounted_bytes,
            generation_accounted_bytes: generation.accounted_bytes,
            generation_cells: generation.allocated_cells,
            guard_references: generation.guard_refs,
            free_slot_capacity: writer.free.capacity(),
            retirement_capacity: writer.retired.capacity(),
            pending_reservation_capacity: writer.pending.capacity(),
            visibility_retirement_capacity: writer.visibility_retained.capacity(),
            free_slot_payload_bytes: free_bytes,
            retirement_payload_bytes: retired_bytes,
            hash_capacity_payload_bytes_lower_bound: hash_bytes,
            outstanding_ack_tokens: writer.outstanding_acks,
            outstanding_ack_payload_bytes: ack_bytes,
            graph_struct_bytes: graph_bytes,
            accounted_bytes_lower_bound: total,
        }
    }
}

/// A source descriptor must be installed before prepare/complete. Dropping an
/// unconsumed reservation cancels it but still requires source acknowledgment.
pub struct InsertReservation {
    pub(super) control: Arc<GraphStorage>,
    pub(super) key: u64,
    pub(super) handle: NodeHandle,
    pub(super) pending: bool,
}
impl InsertReservation {
    pub fn key(&self) -> u64 {
        self.key
    }
    pub fn handle(&self) -> NodeHandle {
        self.handle
    }
}
impl Drop for InsertReservation {
    fn drop(&mut self) {
        if !self.pending {
            return;
        }
        let mut writer = self.control.writer.lock();
        // Never call source code, acquire a reader guard or advance generations
        // in Drop. The permit owns cancellation capacity from admission.
        if writer.pending.get(&self.key) != Some(&self.handle) {
            self.control.poison();
            return;
        }
        let id = internal_id(self.handle);
        let valid = self
            .control
            .nodes
            .writer_metadata(id)
            .is_ok_and(|metadata| {
                metadata.state == NodeState::Reserved
                    && metadata.incarnation == self.handle.generation()
                    && metadata.source_key == self.key
            });
        if !valid {
            self.control.poison();
            return;
        }
        let Ok(transition) = self
            .control
            .nodes
            .preflight_invalidate(id, LevelsRef::INVALID)
        else {
            self.control.poison();
            return;
        };
        transition.commit();
        writer.pending.remove(&self.key);
        self.control
            .retire_committed(&mut writer, self.handle, self.key);
        self.pending = false;
    }
}

/// Graph/prepared readers have drained, but the slot is NOT reusable yet. The
/// caller removes the matching source descriptor before dropping this token.
/// Arbitrary independently pinned source roots must own their older rows.
pub struct ReclaimedNode {
    pub(super) control: Arc<GraphStorage>,
    retired: RetiredNode,
}
impl ReclaimedNode {
    pub fn handle(&self) -> NodeHandle {
        self.retired.handle
    }
    pub fn key(&self) -> u64 {
        self.retired.key
    }
}
impl Drop for ReclaimedNode {
    fn drop(&mut self) {
        let mut writer = self.control.writer.lock();
        let handle = self.retired.handle;
        let id = internal_id(handle);
        let valid = self
            .control
            .nodes
            .writer_metadata(id)
            .is_ok_and(|metadata| {
                metadata.state == NodeState::Retired
                    && metadata.incarnation == handle.generation()
                    && metadata.source_key == self.retired.key
            });
        if !valid {
            self.control.poison();
            return;
        }
        if writer.outstanding_acks == 0
            || (handle.generation() != u32::MAX && writer.free.len() == writer.free.capacity())
        {
            self.control.poison();
            return;
        }
        // SAFETY: only strict-oldest retirements become tokens. The token owns
        // this slot until Drop, which acknowledges external source cleanup.
        if unsafe { self.control.nodes.mark_reclaimed(id) }.is_err() {
            self.control.poison();
            return;
        }
        if handle.generation() == u32::MAX {
            writer.exhausted += 1;
        } else {
            writer.free.push(handle.slot());
        }
        writer.outstanding_acks -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn storage(max_slots: u32, max_retired: usize) -> Arc<GraphStorage> {
        Arc::new(
            GraphStorage::new(MutableHnswLimits {
                max_slots,
                max_pending: max_retired,
                max_retired,
                ..MutableHnswLimits::default()
            })
            .unwrap(),
        )
    }

    #[test]
    fn cancellation_waits_for_shared_guard_then_source_ack_before_reuse() {
        let storage = storage(1, 2);
        let permit = storage.reserve_insert(7).unwrap();
        let first = permit.handle();
        let held = storage.domain.take_guard().unwrap();
        let read = storage.nodes.read(&held).unwrap();
        let old = read.get(internal_id(first)).unwrap();
        drop(permit);
        assert!(storage.reclamation_pending());
        assert!(storage.collect_reclaimed().is_empty());
        assert_eq!(old.metadata().source_key, 7);
        assert!(storage.reserve_insert(8).is_err());
        drop(held);
        let mut tokens = storage.collect_reclaimed();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].key(), 7);
        assert_eq!(tokens[0].handle(), first);
        assert!(
            storage.reserve_insert(8).is_err(),
            "source ACK still owns slot"
        );
        assert!(
            storage.reclamation_pending(),
            "outstanding source ACK is held ownership"
        );
        drop(tokens.pop());
        assert!(!storage.reclamation_pending());
        let second = storage.reserve_insert(8).unwrap();
        assert_eq!(second.handle().slot(), first.slot());
        assert_eq!(second.handle().generation(), first.generation() + 1);
        assert_eq!(storage.stats().allocated_slots, 1);
    }

    #[test]
    fn outstanding_ack_holds_budget_and_queue_capacity_survives_collection() {
        let storage = storage(8, 8);
        let mut permits: Vec<_> = (0..8)
            .map(|key| Some(storage.reserve_insert(key).unwrap()))
            .collect();
        let capacity = storage.writer.lock().retired.capacity();
        assert!(capacity >= 8);
        for item in permits.iter_mut().take(4) {
            drop(item.take());
        }
        let first = storage.collect_reclaimed();
        assert_eq!(first.len(), 4);
        assert_eq!(storage.writer.lock().retired.capacity(), capacity);
        for item in &mut permits {
            drop(item.take());
        }
        assert_eq!(storage.writer.lock().retired.capacity(), capacity);
        let second = storage.collect_reclaimed();
        assert_eq!(second.len(), 4);
        assert_eq!(storage.stats().retired_slots, 8);
        assert!(storage.reserve_insert(8).is_err());
        assert_eq!(storage.stats().issued_key_watermark, Some(7));
        drop(first);
        drop(second);
        assert_eq!(storage.stats().retired_slots, 0);
        assert_eq!(storage.stats().reusable_slots, 8);
        assert!(storage.reserve_insert(8).is_ok());
    }

    #[test]
    fn peak_free_credit_covers_all_permits_and_ack_tokens() {
        let storage = storage(32, 32);
        let permits: Vec<_> = (0..32)
            .map(|key| storage.reserve_insert(key).unwrap())
            .collect();
        assert!(storage.writer.lock().free.capacity() >= 32);
        assert_eq!(
            storage.live_bound.load(Ordering::Acquire),
            1,
            "unpublished reservations are not searchable"
        );
        drop(permits);
        assert_eq!(storage.live_bound.load(Ordering::Acquire), 1);
        assert_eq!(storage.allocated.load(Ordering::Acquire), 32);
        let free_capacity = storage.writer.lock().free.capacity();
        let tokens = storage.collect_reclaimed();
        assert_eq!(tokens.len(), 32);
        drop(tokens);
        assert_eq!(storage.writer.lock().free.capacity(), free_capacity);
        assert_eq!(storage.writer.lock().free.len(), 32);
    }

    #[test]
    fn pinned_hold_cells_defer_the_advance_without_blocking_writes() {
        let storage = Arc::new(
            GraphStorage::new_with_generation_limits(
                MutableHnswLimits {
                    max_slots: 2,
                    ..MutableHnswLimits::default()
                },
                GenerationLimits {
                    max_cells: 1,
                    ..GenerationLimits::default()
                },
            )
            .unwrap(),
        );
        let mut permit = storage.reserve_insert(0).unwrap();
        let guard = storage.domain.take_guard().unwrap();
        let levels = storage.levels.writer().add(&[0]).unwrap();
        let mut writer = storage.writer.lock();
        // SAFETY: isolated real live levels allocation, no links/source reads.
        unsafe {
            storage
                .nodes
                .preflight_publish(internal_id(permit.handle), levels, 1)
        }
        .unwrap()
        .commit();
        writer.pending.remove(&permit.key);
        permit.pending = false;
        storage.publication.store(1, Ordering::Release);
        storage.active.store(1, Ordering::Release);
        let before = storage.domain.current_generation();
        storage.finish_mutation(&mut writer);
        // Every cell is pinned (one cell, held by `guard`): the advance is
        // deferred, writes are not blocked, nothing is republished.
        assert!(writer.blocked.is_none());
        assert!(writer.maintenance_pending, "the advance is retried later");
        assert_eq!(storage.domain.current_generation(), before);
        assert!(storage.check_readable().is_ok());
        storage.admit_write(&mut writer).unwrap();
        assert_eq!(
            storage.nodes.writer_metadata(1).unwrap().state,
            NodeState::Published
        );
        assert_eq!(storage.publication.load(Ordering::Acquire), 1);
        // The reader lets go: the next maintenance advances.
        drop(guard);
        storage.admit_write(&mut writer).unwrap();
        assert!(writer.blocked.is_none());
        assert!(!writer.maintenance_pending);
        assert!(storage.domain.current_generation() > before);
        assert_eq!(storage.publication.load(Ordering::Acquire), 1);
        assert_eq!(storage.active.load(Ordering::Acquire), 1);
    }

    #[test]
    fn cancelled_retirement_survives_failed_assignment_boundary() {
        let storage = Arc::new(
            GraphStorage::new_with_generation_limits(
                MutableHnswLimits {
                    max_slots: 1,
                    ..MutableHnswLimits::default()
                },
                GenerationLimits {
                    max_cells: 1,
                    ..GenerationLimits::default()
                },
            )
            .unwrap(),
        );
        let permit = storage.reserve_insert(0).unwrap();
        let guard = storage.domain.take_guard().unwrap();
        drop(permit);
        let current = storage.domain.current_generation();
        assert!(storage.collect_reclaimed().is_empty());
        {
            let writer = storage.writer.lock();
            assert_eq!(writer.retired.len(), 1);
            assert_eq!(writer.retired[0].generation, Some(current));
            assert_eq!(writer.outstanding_acks, 0);
        }
        drop(guard);
        let tokens = storage.collect_reclaimed();
        assert_eq!(tokens.len(), 1);
        drop(tokens);
        assert!(storage.reserve_insert(1).is_ok());
    }

    #[test]
    fn failed_admission_does_not_issue_watermark_and_wrong_graph_cannot_ack() {
        let storage = storage(1, 1);
        let other = self::storage(1, 1);
        let permit = storage.reserve_insert(5).unwrap();
        assert!(storage.reserve_insert(6).is_err());
        assert_eq!(storage.stats().issued_key_watermark, Some(5));
        drop(permit);
        let tokens = storage.collect_reclaimed();
        assert_eq!(tokens.len(), 1);
        assert!(storage.owns_reclaimed(&tokens[0]));
        assert!(!other.owns_reclaimed(&tokens[0]));
    }

    #[test]
    fn poisoned_publication_stops_reads_writes_and_new_source_acknowledgments() {
        let storage = storage(2, 2);
        let permit = storage.reserve_insert(0).unwrap();
        drop(permit);
        storage.poison();
        assert!(storage.check_readable().is_err());
        assert!(storage.reserve_insert(1).is_err());
        assert!(storage.collect_reclaimed().is_empty());
        assert_eq!(storage.stats().retired_slots, 1);
    }

    #[test]
    fn memory_report_includes_dense_growth_guards_and_outstanding_ack_owners() {
        let storage = storage(32, 32);
        let held = storage.domain.take_guard().unwrap();
        let permits: Vec<_> = (0..32)
            .map(|key| storage.reserve_insert(key).unwrap())
            .collect();
        let retained = storage.memory_usage();
        assert!(retained.node_held_payload_bytes > 0);
        assert!(
            storage.reclamation_pending(),
            "node-table growth alone is a hold"
        );
        assert!(
            retained.node_current_payload_bytes
                >= 33 * size_of::<super::super::node_table::NodeRecord>()
        );
        assert!(retained.guard_references >= 1);
        assert!(retained.pending_reservation_capacity >= 32);
        assert!(retained.retirement_capacity >= 32);
        assert!(retained.free_slot_capacity >= 32);
        assert!(
            retained.accounted_bytes_lower_bound
                >= retained.node_current_payload_bytes + retained.node_held_payload_bytes
        );
        drop(permits);
        assert!(storage.collect_reclaimed().is_empty());
        drop(held);
        let tokens = storage.collect_reclaimed();
        let awaiting_ack = storage.memory_usage();
        assert_eq!(awaiting_ack.node_held_payload_bytes, 0);
        assert_eq!(awaiting_ack.outstanding_ack_tokens, 32);
        assert_eq!(
            awaiting_ack.outstanding_ack_payload_bytes,
            32 * size_of::<ReclaimedNode>()
        );
        drop(tokens);
        assert_eq!(storage.memory_usage().outstanding_ack_payload_bytes, 0);
    }
}

#[cfg(test)]
mod registry_admission_tests {
    use super::*;
    #[test]
    fn graph_registry_growth_retains_old_contents_and_refusal_is_non_mutating() {
        let owner=crate::admission::Owner::new();let _scope=owner.enter();
        let mut values=Vec::<u32>::new();let mut memory=None;
        reserve_vec(&mut values,&mut memory,4,"test free registry").unwrap();
        values.extend([3,7]);let before=owner.bytes().current;
        assert!(reserve_vec(&mut values,&mut memory,usize::MAX,"overflow").is_err());
        assert_eq!(values,[3,7]);assert_eq!(owner.bytes().current,before);
        reserve_vec(&mut values,&mut memory,17,"test growth").unwrap();
        assert_eq!(values,[3,7]);assert!(owner.bytes().current>=17*4);
        let mut map=HashMap::<u64,u64>::new();let mut map_memory=None;
        reserve_map(&mut map,&mut map_memory,4,"test pending registry").unwrap();map.insert(3,7);
        let before=owner.bytes().current;
        assert!(reserve_map(&mut map,&mut map_memory,usize::MAX,"overflow").is_err());
        assert_eq!(map.get(&3),Some(&7));assert_eq!(owner.bytes().current,before);
        reserve_map(&mut map,&mut map_memory,64,"test pending growth").unwrap();
        assert_eq!(map.get(&3),Some(&7));
        drop(map);drop(map_memory);drop(values);drop(memory);assert_eq!(owner.bytes().current,0);
    }
}
