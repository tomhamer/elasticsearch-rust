//! Isolated port of Vespa's GenerationHold / GenerationGuard / GenerationHandler.
//!
//! Reference: vespa-engine/vespa revision
//! `0521b00640ef934b359d2fc356f97d84499dd17d`,
//! `vespalib/src/vespa/vespalib/util/generation_{hold,guard}.{h,cpp}` and
//! `generationhandler.{h,cpp}`. No graph or storage integration lives here.
//!
//! Readers acquire the current hold *before* loading protected references. The
//! low reference-count bit means invalid; every guard adds/subtracts two.
//! Invalidation succeeds only from zero, and failed acquisitions retry current.
//! Cells are type-stable until the entire Arc-owned domain is destroyed. This
//! permits a delayed reader to encounter an invalid or already reused cell.
//! Such a reader has not loaded protected data yet, so acquiring the reused
//! cell's newer generation is safe. Existing guards prevent invalidation.
//! An unused current cell may also be morphed in place. A reader acquiring
//! after the writer's zero-reference observation can see its guard's generation
//! change to the new value; the AcqRel reference RMW orders that reader after
//! prior writer publication. A guard's generation is not an immutable PIT ID.
//!
//! Writer protocol, matching GenerationHoldList: replace published references,
//! assign their old storage the handler's current generation, then increment
//! the generation. Reclaim assigned generations STRICTLY LESS than the updated
//! oldest used generation. The cached oldest value may lag and is conservative.
//! Assigning a later retirement an older generation, loading protected storage
//! before acquiring a guard, or reclaiming at equality is not safe. This module
//! deliberately does not own payload hold lists or provide a reclaim-all API.
//! One external topology writer must serialize this entire publish/assign/
//! advance sequence across every store sharing the domain. The internal mutex
//! only protects handler bookkeeping, not external retirement queues.
//!
//! Rust adaptations: guards retain Arc<Core>, so they may outlive handler
//! handles and move between threads without external destruction coordination.
//! A mutex serializes writer/list operations only; reader acquisition, clone,
//! and drop use atomics. Bounded CAS increments replace unchecked fetch_add so
//! counters cannot wrap. Generation exhaustion and cell admission fail before
//! publishing a new generation. Cells remain allocated after reuse, as in Vespa.
//! Cell admission reserves all index-list capacity, so invalidation/reuse never
//! needs another container allocation. Container reservation errors are returned;
//! Box/Arc allocation still follows Rust's allocator-wide OOM policy.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

const INVALID: u32 = 1;
const MAX_REFS: u32 = (u32::MAX - 1) / 2;
const MAX_CELLS: u32 = 1 << 20;

#[derive(Clone, Copy, Debug)]
pub(super) struct GenerationLimits {
    pub max_cells: u32,
    pub max_guards_per_cell: u32,
    pub max_generation: u64,
}

impl Default for GenerationLimits {
    fn default() -> Self {
        Self {
            max_cells: 65_536,
            max_guards_per_cell: MAX_REFS,
            max_generation: u64::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenerationError {
    InvalidLimits,
    CellLimit,
    GuardLimit,
    Exhausted,
    AllocationFailed,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid generation-domain limits",
            Self::CellLimit => "generation hold-cell limit reached",
            Self::GuardLimit => "generation guard reference limit reached",
            Self::Exhausted => "generation counter exhausted",
            Self::AllocationFailed => "generation bookkeeping allocation failed",
        })
    }
}
impl std::error::Error for GenerationError {}

struct HoldCell {
    refs: AtomicU32,
    generation: AtomicU64,
}

impl HoldCell {
    fn invalid() -> Self {
        Self {
            refs: AtomicU32::new(INVALID),
            generation: AtomicU64::new(0),
        }
    }

    /// Returns whether the *previous* state was valid. Even a failed acquire
    /// temporarily contributes two, just like Vespa; its caller must release.
    fn add_ref(&self, limit: u32, order: Ordering) -> Result<bool, GenerationError> {
        let mut old = self.refs.load(Ordering::Relaxed);
        loop {
            if old / 2 >= limit {
                return Err(GenerationError::GuardLimit);
            }
            // limit <= MAX_REFS proves this cannot overflow, even when invalid.
            match self
                .refs
                .compare_exchange_weak(old, old + 2, order, Ordering::Relaxed)
            {
                Ok(_) => return Ok(old & INVALID == 0),
                Err(now) => old = now,
            }
        }
    }

    fn release(&self) {
        let old = self.refs.fetch_sub(2, Ordering::Release);
        assert!(old / 2 > 0, "generation reference underflow");
    }

    fn invalidate(&self) -> bool {
        match self
            .refs
            .compare_exchange(0, INVALID, Ordering::AcqRel, Ordering::Relaxed)
        {
            Ok(_) => true,
            Err(refs) => {
                assert_eq!(refs & INVALID, 0, "active hold already invalid");
                false
            }
        }
    }

    fn validate(&self) {
        // Failed acquisitions may still own temporary references, hence a
        // reused invalid cell need not have refs == 1 here.
        let old = self.refs.fetch_sub(INVALID, Ordering::Release);
        assert_eq!(old & INVALID, INVALID, "free hold already valid");
    }

    fn refs(&self) -> u32 {
        self.refs.load(Ordering::Relaxed) / 2
    }
}

struct Writer {
    // Boxes must never be removed/replaced. Vec movement does not move cells.
    cells: Vec<Box<HoldCell>>,
    active: VecDeque<usize>,
    free: Vec<usize>,
}

impl Writer {
    /// Every allocated cell can eventually enter either index list. Reserve
    /// both lists before admitting a cell, including when their current len is
    /// zero; reserving just one more than len would not cover later draining.
    fn reserve_cell_capacity(&mut self, count: usize) -> Result<(), GenerationError> {
        if self.cells.capacity() < count {
            self.cells
                .try_reserve(count - self.cells.len())
                .map_err(|_| GenerationError::AllocationFailed)?;
        }
        if self.active.capacity() < count {
            self.active
                .try_reserve(count - self.active.len())
                .map_err(|_| GenerationError::AllocationFailed)?;
        }
        if self.free.capacity() < count {
            self.free
                .try_reserve(count - self.free.len())
                .map_err(|_| GenerationError::AllocationFailed)?;
        }
        Ok(())
    }
}

struct Core {
    current: AtomicPtr<HoldCell>,
    generation: AtomicU64,
    oldest: AtomicU64,
    limits: GenerationLimits,
    writer: Mutex<Writer>,
}

impl Core {
    fn update_oldest(&self, writer: &mut Writer) -> u64 {
        debug_assert!(writer.free.capacity() >= writer.cells.len());
        while writer.active.len() > 1 {
            let index = writer.active[0];
            if !writer.cells[index].invalidate() {
                break;
            }
            writer.active.pop_front();
            writer.free.push(index);
        }
        let oldest = writer.cells[writer.active[0]]
            .generation
            .load(Ordering::Relaxed);
        self.oldest.store(oldest, Ordering::Relaxed);
        oldest
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Every guard and in-progress acquisition retains this Core. Therefore
        // its final Arc cannot disappear while any cell reference is alive.
        debug_assert!(
            self.writer
                .get_mut()
                .cells
                .iter()
                .all(|cell| cell.refs() == 0)
        );
    }
}

#[derive(Clone)]
pub(super) struct GenerationHandler {
    core: Arc<Core>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GenerationStats {
    pub current: u64,
    pub oldest: u64,
    pub allocated_cells: usize,
    pub active_cells: usize,
    pub free_cells: usize,
    /// Approximate while readers acquire/drop concurrently.
    pub guard_refs: u64,
    /// Owned cell and container payload/capacity only. Allocator overhead and
    /// Arc's private control header are not exposed and are excluded.
    pub hold_cell_bytes: usize,
    pub cell_pointer_capacity_bytes: usize,
    pub active_index_capacity_bytes: usize,
    pub free_index_capacity_bytes: usize,
    /// Includes embedded mutex/list headers, excludes the Arc control header.
    pub core_struct_bytes: usize,
    pub accounted_bytes: usize,
}

impl GenerationHandler {
    pub(super) fn new(limits: GenerationLimits) -> Result<Self, GenerationError> {
        if limits.max_cells == 0
            || limits.max_cells > MAX_CELLS
            || limits.max_guards_per_cell == 0
            || limits.max_guards_per_cell > MAX_REFS
        {
            return Err(GenerationError::InvalidLimits);
        }
        let mut writer = Writer {
            cells: Vec::new(),
            active: VecDeque::new(),
            free: Vec::new(),
        };
        writer.reserve_cell_capacity(1)?;
        let mut first = Box::new(HoldCell::invalid());
        first.validate();
        let current = std::ptr::from_mut(first.as_mut());
        writer.cells.push(first);
        writer.active.push_back(0);
        Ok(Self {
            core: Arc::new(Core {
                current: AtomicPtr::new(current),
                generation: AtomicU64::new(0),
                oldest: AtomicU64::new(0),
                limits,
                writer: Mutex::new(writer),
            }),
        })
    }

    /// Pin before loading any graph/source references protected by this domain.
    pub(super) fn take_guard(&self) -> Result<GenerationGuard, GenerationError> {
        loop {
            let cell = self.core.current.load(Ordering::Acquire);
            // SAFETY: cell was published by this Core and its boxes are stable
            // for the lifetime retained by self, including after invalidation.
            if let Some(guard) = unsafe { self.acquire_from(cell) }? {
                return Ok(guard);
            }
            std::hint::spin_loop();
        }
    }

    /// # Safety
    /// `cell` must be a non-null pointer obtained from this Core.current while
    /// holding this domain alive. It may now be invalid or reused, never foreign.
    unsafe fn acquire_from(
        &self,
        cell: *mut HoldCell,
    ) -> Result<Option<GenerationGuard>, GenerationError> {
        // SAFETY: current points into one of Core.writer.cells' stable Boxes.
        // Cells are never removed, even after invalidation/reuse. self retains
        // Core while dereferencing; a successful guard then retains the Core.
        let hold = unsafe { &*cell };
        if hold.add_ref(self.core.limits.max_guards_per_cell, Ordering::AcqRel)? {
            Ok(Some(GenerationGuard {
                core: Arc::clone(&self.core),
                cell: NonNull::new(cell).expect("current generation is non-null"),
            }))
        } else {
            hold.release();
            Ok(None)
        }
    }

    /// Writer-side generation advance. The caller must first assign all newly
    /// retired storage the previous current generation. On error no new
    /// generation is published, and that storage must remain on hold.
    /// This does not roll back already published graph/storage references:
    /// callers must admit before publication or treat failure as reclamation
    /// backpressure and retain pending/assigned retired objects until retry.
    pub(super) fn inc_generation(&self) -> Result<u64, GenerationError> {
        let mut writer = self.core.writer.lock();
        let current = self.core.generation.load(Ordering::Relaxed);
        if current >= self.core.limits.max_generation {
            return Err(GenerationError::Exhausted);
        }
        let next = current + 1;
        let last = *writer.active.back().expect("one current cell");
        // Vespa's zero-sized AcqRel RMW is deliberate: it orders a concurrent
        // reader's acquire after prior writer publication even when this cell
        // is morphed in place. A reader arriving after this zero observation
        // may observe the newer generation; it has not loaded protected data.
        if writer.cells[last].refs.fetch_add(0, Ordering::AcqRel) / 2 == 0 {
            self.core.generation.store(next, Ordering::Relaxed);
            writer.cells[last].generation.store(next, Ordering::Relaxed);
            self.core.update_oldest(&mut writer);
            return Ok(next);
        }
        // Bounded adaptation: reclaim unused cells before rejecting admission.
        // This only advances the conservative oldest cache, not current.
        self.core.update_oldest(&mut writer);
        let index = if let Some(index) = writer.free.pop() {
            index
        } else {
            if writer.cells.len() >= self.core.limits.max_cells as usize {
                return Err(GenerationError::CellLimit);
            }
            let index = writer.cells.len();
            // No new cell or current generation is published until all three
            // bookkeeping containers can represent the admitted cell. Partial
            // capacity increases on error are harmless and remain accounted.
            writer.reserve_cell_capacity(index + 1)?;
            writer.cells.push(Box::new(HoldCell::invalid()));
            index
        };
        let cell = &writer.cells[index];
        cell.generation.store(next, Ordering::Relaxed);
        cell.validate();
        let pointer = std::ptr::from_ref(cell.as_ref()).cast_mut();
        debug_assert!(writer.active.capacity() >= writer.cells.len());
        writer.active.push_back(index);
        self.core.generation.store(next, Ordering::Relaxed);
        self.core.current.store(pointer, Ordering::Release);
        self.core.update_oldest(&mut writer);
        Ok(next)
    }

    pub(super) fn update_oldest_used_generation(&self) -> u64 {
        self.core.update_oldest(&mut self.core.writer.lock())
    }

    pub(super) fn current_generation(&self) -> u64 {
        self.core.generation.load(Ordering::Relaxed)
    }

    pub(super) fn oldest_used_generation(&self) -> u64 {
        self.core.oldest.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn owns_guard(&self, guard: &GenerationGuard) -> bool {
        Arc::ptr_eq(&self.core, &guard.core)
    }

    pub(super) fn stats(&self) -> GenerationStats {
        let writer = self.core.writer.lock();
        let hold_cell_bytes = writer.cells.len() * std::mem::size_of::<HoldCell>();
        let cell_pointer_capacity_bytes =
            writer.cells.capacity() * std::mem::size_of::<Box<HoldCell>>();
        let active_index_capacity_bytes = writer.active.capacity() * std::mem::size_of::<usize>();
        let free_index_capacity_bytes = writer.free.capacity() * std::mem::size_of::<usize>();
        let core_struct_bytes = std::mem::size_of::<Core>();
        GenerationStats {
            current: self.current_generation(),
            oldest: self.oldest_used_generation(),
            allocated_cells: writer.cells.len(),
            active_cells: writer.active.len(),
            free_cells: writer.free.len(),
            guard_refs: writer.cells.iter().map(|cell| u64::from(cell.refs())).sum(),
            hold_cell_bytes,
            cell_pointer_capacity_bytes,
            active_index_capacity_bytes,
            free_index_capacity_bytes,
            core_struct_bytes,
            accounted_bytes: hold_cell_bytes
                + cell_pointer_capacity_bytes
                + active_index_capacity_bytes
                + free_index_capacity_bytes
                + core_struct_bytes,
        }
    }
}

pub(super) struct GenerationGuard {
    core: Arc<Core>,
    cell: NonNull<HoldCell>,
}

// SAFETY: a guard owns a ref on its stable atomic-only HoldCell and keeps the
// containing Core alive. No thread-affine epoch guard or non-atomic cell data
// is exposed. Shared access can only read generation or atomically add refs.
unsafe impl Send for GenerationGuard {}
unsafe impl Sync for GenerationGuard {}

impl GenerationGuard {
    fn hold(&self) -> &HoldCell {
        // SAFETY: this guard's Arc keeps the cell allocation alive; its counted
        // reference prevents invalidation/reuse until this guard is dropped.
        unsafe { self.cell.as_ref() }
    }

    pub(super) fn generation(&self) -> u64 {
        self.hold().generation.load(Ordering::Relaxed)
    }

    pub(super) fn try_clone(&self) -> Result<Self, GenerationError> {
        let valid = self
            .hold()
            .add_ref(self.core.limits.max_guards_per_cell, Ordering::Relaxed)?;
        assert!(valid, "a guarded generation cannot be invalidated");
        Ok(Self {
            core: Arc::clone(&self.core),
            cell: self.cell,
        })
    }
}

impl Clone for GenerationGuard {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("generation guard clone limit reached")
    }
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.hold().release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    fn handler() -> GenerationHandler {
        GenerationHandler::new(GenerationLimits::default()).unwrap()
    }

    #[test]
    fn guard_clone_send_drop_and_domain_lifetime() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<GenerationGuard>();
        let domain = handler();
        let guard = domain.take_guard().unwrap();
        let copy = guard.clone();
        assert!(domain.owns_guard(&copy));
        assert!(!handler().owns_guard(&copy));
        assert_eq!(domain.stats().guard_refs, 2);
        drop(domain);
        let returned = std::thread::spawn(move || {
            assert_eq!(copy.generation(), 0);
            copy.try_clone().unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(returned.generation(), 0);
        drop(guard);
        drop(returned); // The last guard may destroy the whole domain.
    }

    #[test]
    fn oldest_generation_is_conservative_and_reclamation_is_strict() {
        let domain = handler();
        let zero = domain.take_guard().unwrap();
        assert_eq!(domain.inc_generation(), Ok(1));
        let one = domain.take_guard().unwrap();
        assert_eq!(domain.inc_generation(), Ok(2));
        assert_eq!(domain.update_oldest_used_generation(), 0);
        let retired = [0, 1, 2];
        assert!(
            retired
                .iter()
                .all(|g| *g >= domain.oldest_used_generation())
        );
        drop(zero);
        assert_eq!(domain.oldest_used_generation(), 0); // Cached, no reader write.
        assert_eq!(domain.update_oldest_used_generation(), 1);
        assert_eq!(
            retired.into_iter().filter(|g| *g < 1).collect::<Vec<_>>(),
            [0]
        );
        drop(one);
        assert_eq!(domain.update_oldest_used_generation(), 2);
        assert_eq!(domain.stats().free_cells, 2);
    }

    #[test]
    fn unused_current_morphs_and_hold_cells_plateau_under_reuse() {
        let domain = handler();
        for expected in 1..100 {
            assert_eq!(domain.inc_generation(), Ok(expected));
        }
        assert_eq!(domain.stats().allocated_cells, 1);
        for _ in 0..100 {
            let guard = domain.take_guard().unwrap();
            let guarded = guard.generation();
            domain.inc_generation().unwrap();
            assert_eq!(guard.generation(), guarded);
            assert_eq!(domain.update_oldest_used_generation(), guarded);
            drop(guard);
            assert_eq!(domain.update_oldest_used_generation(), guarded + 1);
        }
        let stats = domain.stats();
        assert_eq!(stats.allocated_cells, 2);
        assert_eq!(stats.hold_cell_bytes, 2 * std::mem::size_of::<HoldCell>());
        assert!(stats.accounted_bytes > stats.hold_cell_bytes);
        let peak = stats.accounted_bytes;
        for _ in 0..100 {
            let guard = domain.take_guard().unwrap();
            domain.inc_generation().unwrap();
            drop(guard);
            domain.update_oldest_used_generation();
        }
        assert_eq!(domain.stats().accounted_bytes, peak);
    }

    #[test]
    fn draining_and_reusing_admitted_cells_never_grows_index_lists() {
        let domain = handler();
        let mut guards = Vec::new();
        // Keep every older generation live while the free list is empty. Its
        // admission capacity must cover the subsequent all-at-once drain.
        for _ in 0..257 {
            guards.push(domain.take_guard().unwrap());
            domain.inc_generation().unwrap();
        }
        let peak = domain.stats();
        assert_eq!(peak.allocated_cells, 258);
        assert_eq!(peak.free_cells, 0);
        let (cells_pointer, free_pointer) = {
            let writer = domain.core.writer.lock();
            assert!(writer.active.capacity() >= writer.cells.len());
            assert!(writer.free.capacity() >= writer.cells.len());
            (
                writer.cells.as_ptr() as usize,
                writer.free.as_ptr() as usize,
            )
        };
        drop(guards);
        assert_eq!(domain.update_oldest_used_generation(), 257);
        assert_eq!(domain.stats().free_cells, 257);
        for _ in 0..1_024 {
            let guard = domain.take_guard().unwrap();
            domain.inc_generation().unwrap();
            drop(guard);
            domain.update_oldest_used_generation();
        }
        let after = domain.stats();
        assert_eq!(after.allocated_cells, peak.allocated_cells);
        assert_eq!(after.accounted_bytes, peak.accounted_bytes);
        assert_eq!(
            after.active_index_capacity_bytes,
            peak.active_index_capacity_bytes
        );
        assert_eq!(
            after.free_index_capacity_bytes,
            peak.free_index_capacity_bytes
        );
        let writer = domain.core.writer.lock();
        assert_eq!(writer.cells.as_ptr() as usize, cells_pointer);
        assert_eq!(writer.free.as_ptr() as usize, free_pointer);
    }

    #[test]
    fn bookkeeping_capacity_error_keeps_current_and_existing_guards_valid() {
        let domain = handler();
        let guard = domain.take_guard().unwrap();
        let before = domain.stats();
        let current = domain.core.current.load(Ordering::Acquire);
        {
            let mut writer = domain.core.writer.lock();
            // Deterministic try_reserve capacity overflow, not an attempt to
            // exhaust host memory or an allocator-wide OOM recovery claim.
            assert_eq!(
                writer.reserve_cell_capacity(usize::MAX),
                Err(GenerationError::AllocationFailed)
            );
            assert_eq!(writer.cells.len(), before.allocated_cells);
            assert_eq!(writer.active.len(), before.active_cells);
            assert_eq!(writer.free.len(), before.free_cells);
        }
        assert_eq!(domain.core.current.load(Ordering::Acquire), current);
        assert_eq!(domain.current_generation(), before.current);
        assert_eq!(guard.generation(), before.current);
        assert_eq!(domain.inc_generation(), Ok(1));
        assert_eq!(guard.generation(), 0);
        drop(guard);
        assert_eq!(domain.update_oldest_used_generation(), 1);
    }

    #[test]
    fn delayed_acquirer_retries_invalid_cell_or_acquires_its_new_generation() {
        for reuse in [false, true] {
            let domain = handler();
            let zero = domain.take_guard().unwrap();
            let reader = domain.clone();
            let (loaded_tx, loaded_rx) = mpsc::channel();
            let (go_tx, go_rx) = mpsc::channel();
            let task = std::thread::spawn(move || {
                let loaded = reader.core.current.load(Ordering::Acquire);
                loaded_tx.send(loaded as usize).unwrap();
                go_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                // SAFETY: loaded came from this same retained domain before
                // the pause. Its stable cell survives invalidation and reuse.
                let attempt = unsafe { reader.acquire_from(loaded) }.unwrap();
                if reuse {
                    let guard = attempt.expect("reused valid cell can be acquired");
                    assert_eq!(guard.generation(), 2);
                } else {
                    assert!(attempt.is_none(), "invalid old cell must be rejected");
                    assert_eq!(reader.take_guard().unwrap().generation(), 1);
                }
            });
            let loaded = loaded_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            domain.inc_generation().unwrap();
            let one = domain.take_guard().unwrap();
            drop(zero);
            domain.update_oldest_used_generation();
            if reuse {
                domain.inc_generation().unwrap();
                assert_eq!(domain.core.current.load(Ordering::Acquire) as usize, loaded);
            }
            go_tx.send(()).unwrap();
            task.join().unwrap();
            drop(one);
        }
    }

    #[test]
    fn failed_acquisition_can_overlap_cell_revalidation_without_a_guard() {
        let cell = HoldCell::invalid();
        assert!(!cell.add_ref(MAX_REFS, Ordering::AcqRel).unwrap());
        assert_eq!(cell.refs.load(Ordering::Relaxed), 3);
        cell.generation.store(17, Ordering::Relaxed);
        cell.validate(); // The failed acquirer's temporary +2 still exists.
        assert_eq!(cell.refs(), 1);
        cell.release(); // It retries current instead of returning this as a guard.
        assert_eq!(cell.refs(), 0);
        assert!(cell.invalidate());
    }

    #[test]
    fn zero_ref_morph_can_relabel_a_reader_acquired_after_writer_observation() {
        let domain = handler();
        let mut writer = domain.core.writer.lock();
        assert_eq!(writer.cells[0].refs.fetch_add(0, Ordering::AcqRel), 0);
        // Pause the writer after Vespa's zero observation. Reader acquisition
        // does not need the writer mutex and may finish before the morph store.
        let reader = domain.clone();
        let guard = std::thread::spawn(move || reader.take_guard().unwrap())
            .join()
            .unwrap();
        assert_eq!(guard.generation(), 0);
        domain.core.generation.store(1, Ordering::Relaxed);
        writer.cells[0].generation.store(1, Ordering::Relaxed);
        assert_eq!(domain.core.update_oldest(&mut writer), 1);
        assert_eq!(guard.generation(), 1);
        // This is safe only because protected data must be loaded AFTER the
        // guard's acquire RMW, ordered after writer publication by the zero RMW.
        // A prepared operation must not use this value as a fixed historical ID.
    }

    #[test]
    fn bounds_fail_without_wrapping_or_releasing_a_live_generation() {
        for limits in [
            GenerationLimits {
                max_cells: 0,
                ..Default::default()
            },
            GenerationLimits {
                max_cells: MAX_CELLS + 1,
                ..Default::default()
            },
            GenerationLimits {
                max_guards_per_cell: 0,
                ..Default::default()
            },
            GenerationLimits {
                max_guards_per_cell: MAX_REFS + 1,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                GenerationHandler::new(limits),
                Err(GenerationError::InvalidLimits)
            ));
        }
        let domain = GenerationHandler::new(GenerationLimits {
            max_cells: 2,
            max_guards_per_cell: 2,
            max_generation: 2,
        })
        .unwrap();
        let zero = domain.take_guard().unwrap();
        let copy = zero.try_clone().unwrap();
        assert!(matches!(zero.try_clone(), Err(GenerationError::GuardLimit)));
        assert!(matches!(
            domain.take_guard(),
            Err(GenerationError::GuardLimit)
        ));
        assert_eq!(domain.stats().guard_refs, 2);
        domain.inc_generation().unwrap();
        let one = domain.take_guard().unwrap();
        assert_eq!(domain.inc_generation(), Err(GenerationError::CellLimit));
        assert_eq!(domain.current_generation(), 1);
        assert_eq!(domain.oldest_used_generation(), 0);
        drop(zero);
        drop(copy);
        assert_eq!(domain.inc_generation(), Ok(2));
        assert_eq!(one.generation(), 1);
        assert_eq!(domain.inc_generation(), Err(GenerationError::Exhausted));
        assert_eq!(domain.current_generation(), 2);
        let full = HoldCell {
            refs: AtomicU32::new(MAX_REFS * 2),
            generation: AtomicU64::new(0),
        };
        assert_eq!(
            full.add_ref(MAX_REFS, Ordering::AcqRel),
            Err(GenerationError::GuardLimit)
        );
        assert_eq!(full.refs.load(Ordering::Relaxed), u32::MAX - 1);
        full.refs.store(u32::MAX, Ordering::Relaxed);
        assert_eq!(
            full.add_ref(MAX_REFS, Ordering::AcqRel),
            Err(GenerationError::GuardLimit)
        );
        assert_eq!(full.refs.load(Ordering::Relaxed), u32::MAX);
    }

    #[test]
    fn concurrent_acquire_clone_and_writer_advance_leave_no_references() {
        let domain = handler();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let domain = &domain;
                scope.spawn(move || {
                    for _ in 0..2_000 {
                        let guard = domain.take_guard().unwrap();
                        let copy = guard.try_clone().unwrap();
                        // These are separate atomics, not one coherent snapshot.
                        // Do not require generation() <= a simultaneously read
                        // current_generation() while publication is in flight.
                        assert!(guard.generation() <= 2_000);
                        drop(copy);
                        drop(guard);
                    }
                });
            }
            for _ in 0..2_000 {
                domain.inc_generation().unwrap();
            }
        });
        assert_eq!(domain.stats().guard_refs, 0);
        assert_eq!(
            domain.update_oldest_used_generation(),
            domain.current_generation()
        );
    }
}
