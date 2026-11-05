//! Packed ArrayStore port used by the mutable graph.
//!
//! Pinned reference: vespa-engine/vespa 0521b00640ef934b359d2fc356f97d84499dd17d,
//! datastore/{array_store.hpp,datastorebase.cpp,bufferstate.cpp,entryref.h}.
//! Small arrays occupy fixed-size slab entries; large arrays use type-zero
//! descriptors. Growth changes a physical allocation, preserving logical refs.
//! Compaction changes refs and requires the caller to publish each replacement
//! before retiring the old entry. Old entries/allocations/buffers are held in
//! two phases and reclaimed only below the shared domain's oldest generation.
//!
//! Rust adaptations: sealed element types, a writer-only mutex, fallible bounds,
//! and a borrowed read view. The rare large-array path owns boxed payloads and
//! deep-copies them on slab growth, matching BufferType::fallback_copy plus
//! vespalib::Array copy construction. Ordinary arrays have no per-array owner.
//!
//! A generation guard alone does not own this store's allocations. A read view
//! borrows BOTH store and guard. Its raw get is unsafe: a fresh guard cannot
//! resurrect a stale reference that was acquired before that guard. The graph
//! adapter must load refs from published owners after pinning the shared domain.
//! No arbitrary safe Ref+Guard -> slice API is provided.

use super::generation::{GenerationGuard, GenerationHandler};
use parking_lot::{Mutex, MutexGuard};
use std::cell::UnsafeCell;
use std::collections::VecDeque;
#[cfg(test)]
use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(super) struct EntryRef<const OFFSET_BITS: u32>(u32);

pub(super) type LevelsRef = EntryRef<22>;
pub(super) type LinksRef = EntryRef<20>;

impl<const B: u32> EntryRef<B> {
    pub(super) const INVALID: Self = Self(0);
    /// Decodes bits only. This confers neither allocation liveness nor domain.
    pub(super) const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
    pub(super) const fn raw(self) -> u32 {
        self.0
    }
    pub(super) const fn is_valid(self) -> bool {
        self.0 != 0
    }
    pub(super) fn buffer_id(self) -> u32 {
        self.0 >> B
    }
    pub(super) fn offset(self) -> u32 {
        self.0 & ((1u32 << B) - 1)
    }
    fn new(buffer: u32, offset: u32) -> Self {
        Self((buffer << B) | offset)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
    impl Sealed for std::sync::atomic::AtomicU32 {}
}

/// Immutable link/vector scalars and atomically replaceable references.
pub(super) trait ArrayElement: sealed::Sealed + Send + Sync + 'static {
    /// Where this element type's slabs are accounted: f32 arrays hold
    /// source rows, the others graph topology.
    const ACCOUNT: crate::accounting::Category;
    fn from_value(value: u32) -> Self;
    fn value(&self) -> u32;
}
impl ArrayElement for u32 {
    const ACCOUNT: crate::accounting::Category = crate::accounting::Category::VectorGraphs;
    fn from_value(value: u32) -> Self {
        value
    }
    fn value(&self) -> u32 {
        *self
    }
}
impl ArrayElement for f32 {
    const ACCOUNT: crate::accounting::Category = crate::accounting::Category::VectorRows;
    fn from_value(value: u32) -> Self { Self::from_bits(value) }
    fn value(&self) -> u32 { self.to_bits() }
}
impl ArrayElement for AtomicU32 {
    const ACCOUNT: crate::accounting::Category = crate::accounting::Category::VectorGraphs;
    fn from_value(value: u32) -> Self {
        Self::new(value)
    }
    fn value(&self) -> u32 {
        self.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AllocationSpec {
    pub min_entries: usize,
    pub max_entries: usize,
    pub entries_for_new_buffer: usize,
    pub grow_factor: f32,
}

#[derive(Clone, Debug)]
pub(super) struct ArrayStoreConfig {
    pub max_small_array_len: usize,
    pub max_buffers: usize,
    /// Class zero is the large-array descriptor; class n stores n elements.
    pub specs: Vec<AllocationSpec>,
}

impl ArrayStoreConfig {
    /// Pinned optimizeForHugePage class thresholds. The 512Ki value is a
    /// switch threshold, NOT an eagerly allocated minimum per class. Allocator
    /// byte rounding is deliberately separate from this configuration.
    pub(super) fn hnsw<const B: u32>(max_small_array_len: usize) -> Self {
        let max_offset = 1usize << B;
        let max_bytes = 256 * 1024 * 1024usize;
        let huge_page = 2 * 1024 * 1024usize;
        let min_switch = 512 * 1024usize;
        let spec = |class: usize| {
            // Vespa also derives the large spec cap from the largest small
            // entry class; the actual descriptor allocation is accounted below.
            let entry_size = class.max(1) * size_of::<u32>();
            let max_entries = max_offset.min(max_bytes.div_ceil(entry_size));
            let lower = min_switch.min(max_entries);
            let threshold = (huge_page / entry_size).clamp(lower, max_entries);
            AllocationSpec {
                min_entries: 0,
                max_entries,
                entries_for_new_buffer: ((threshold - lower) / 4096) * 4096 + lower,
                grow_factor: 0.3,
            }
        };
        let mut specs = Vec::with_capacity(max_small_array_len + 1);
        let mut large = spec(max_small_array_len);
        large.entries_for_new_buffer = min_switch.min(large.max_entries);
        specs.push(large);
        specs.extend((1..=max_small_array_len).map(spec));
        Self {
            max_small_array_len,
            max_buffers: 1usize << (32 - B),
            specs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StoreError {
    InvalidConfig,
    ForeignGuard,
    InvalidRef,
    NotLive,
    NotCompacting,
    LiveOwnersRemain,
    AddressSpaceExhausted,
    AllocationFailed,
    MemoryLimit,
    IndexOutOfBounds,
    ForeignStore,
    DuplicateRetirement,
    InvalidBatch,
    IncompleteBatch,
    OwnerChanged,
}
impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "packed array store: {self:?}")
    }
}
impl std::error::Error for StoreError {}

// All sealed ArrayElement types are four bytes. Blocks provide a 32-byte
// base without changing the logical contiguous element layout.
#[repr(C, align(32))]
struct AlignedBlock<E: ArrayElement>([UnsafeCell<E>; 8]);
/// Charged to node-wide accounting while alive (released when freed, after
/// generation reclamation for retired arrays). No field for it: the
/// descriptor's size feeds the allocator-rounded buffer sizes.
struct AlignedElements<E: ArrayElement> {
    blocks: Box<[AlignedBlock<E>]>,
    len: usize,
    _memory: crate::admission::Reservation,
}
impl<E: ArrayElement> Drop for AlignedElements<E> {
    fn drop(&mut self) {
        crate::accounting::sub(E::ACCOUNT, self.allocated_bytes() as u64);
    }
}
impl<E: ArrayElement> AlignedElements<E> {
    fn zeroed(len: usize, budget: &std::sync::Arc<crate::admission::Budget>) -> Result<Self, StoreError> {
        assert_eq!(size_of::<E>(), 4);
        assert_eq!(size_of::<AlignedBlock<E>>(), 32);
        let count = len.div_ceil(8);
        let bytes = count.checked_mul(32).ok_or(StoreError::AllocationFailed)?;
        let memory = budget.reserve(bytes as u64, "mutable vector slab").map_err(|_| StoreError::MemoryLimit)?;
        let mut blocks = Vec::new();
        blocks.try_reserve_exact(count).map_err(|_| StoreError::AllocationFailed)?;
        blocks.resize_with(count, || AlignedBlock(std::array::from_fn(|_| UnsafeCell::new(E::from_value(0)))));
        crate::accounting::add(E::ACCOUNT, (count * 32) as u64);
        Ok(Self { blocks: blocks.into_boxed_slice(), len, _memory: memory })
    }
    fn allocated_bytes(&self) -> usize { self.blocks.len() * 32 }
}
impl<E: ArrayElement> std::ops::Deref for AlignedElements<E> {
    type Target = [UnsafeCell<E>];
    fn deref(&self) -> &Self::Target {
        // Blocks have exactly eight contiguous elements and no trailing padding.
        // The boxed owner retains initialized storage for the returned borrow.
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr().cast(), self.len) }
    }
}

enum Payload<E: ArrayElement> {
    Small(AlignedElements<E>),
    Large(Box<[UnsafeCell<Option<AlignedElements<E>>>]>),
}

struct Allocation<E: ArrayElement> {
    class: usize,
    capacity: usize,
    payload: Payload<E>,
    // Updated only by the writer; avoids scanning large descriptors on retirement.
    large_payload_bytes: AtomicUsize,
    budget: std::sync::Arc<crate::admission::Budget>,
    _memory: crate::admission::Reservation,
}

// SAFETY: only the serial writer initializes/reuses entries, and only after
// generation reclamation proves no borrowed reader can access those entries.
// Published u32 arrays are immutable. AtomicU32 replacements are atomic. Header
// and slab addresses never change; growth publishes a different Allocation.
unsafe impl<E: ArrayElement> Send for Allocation<E> {}
unsafe impl<E: ArrayElement> Sync for Allocation<E> {}

impl<E: ArrayElement> Allocation<E> {
    fn new(class: usize, capacity: usize, budget: &std::sync::Arc<crate::admission::Budget>) -> Result<Box<Self>, StoreError> {
        // Descriptor/header plus writer entry/free-list metadata. Slab elements
        // have their own guard, retained through generation reclamation.
        let per_entry = if class == 0 { size_of::<Option<AlignedElements<E>>>() } else { 0 };
        let bytes = capacity.checked_mul(per_entry.saturating_add(16)).and_then(|n| n.checked_add(size_of::<Self>() + 256)).ok_or(StoreError::AllocationFailed)?;
        let memory = budget.reserve(bytes as u64, "mutable vector descriptors").map_err(|_| StoreError::MemoryLimit)?;
        let payload = if class == 0 {
            let mut v = Vec::new();
            v.try_reserve_exact(capacity)
                .map_err(|_| StoreError::AllocationFailed)?;
            v.resize_with(capacity, || UnsafeCell::new(None));
            Payload::Large(v.into_boxed_slice())
        } else {
            let count = capacity
                .checked_mul(class)
                .ok_or(StoreError::AllocationFailed)?;
            Payload::Small(AlignedElements::zeroed(count, budget)?)
        };
        Ok(Box::new(Self {
            class,
            capacity,
            payload,
            large_payload_bytes: AtomicUsize::new(0),
            budget: budget.clone(),
            _memory: memory,
        }))
    }

    fn physical_bytes(&self) -> usize {
        size_of::<Self>()
            + match &self.payload {
                Payload::Small(v) => v.allocated_bytes(),
                Payload::Large(v) => v.len() * size_of::<Option<AlignedElements<E>>>(),
            }
    }

    fn owned_bytes(&self) -> usize {
        self.physical_bytes()
            .saturating_add(self.large_payload_bytes.load(Ordering::Relaxed))
    }

    /// Caller either holds the matching generation for an acquired allocation
    /// identity, or is the serial writer before reuse. No contents may be
    /// overwritten while the returned borrow is used.
    unsafe fn get(&self, offset: usize) -> Result<&[E], StoreError> {
        if offset >= self.capacity {
            return Err(StoreError::InvalidRef);
        }
        match &self.payload {
            Payload::Small(v) => {
                // SAFETY: UnsafeCell<E> has E's layout; allocation is fully
                // initialized and offset/class bounds fit the boxed slab.
                Ok(unsafe {
                    std::slice::from_raw_parts(
                        v.as_ptr().add(offset * self.class).cast::<E>(),
                        self.class,
                    )
                })
            }
            Payload::Large(v) => {
                // SAFETY: caller's lifetime contract prevents replacing this
                // descriptor. The descriptor owns its payload for that lifetime.
                let payload = unsafe { &*v[offset].get() }.as_ref().ok_or(StoreError::InvalidRef)?;
                // Same publication/reclamation contract as the inline slab.
                Ok(unsafe { std::slice::from_raw_parts(payload.as_ptr().cast::<E>(), payload.len()) })
            }
        }
    }

    /// The slot is unpublished or reclaimed; only the single writer calls this.
    unsafe fn write(&self, offset: usize, values: &[u32]) -> Result<(), StoreError> {
        match &self.payload {
            Payload::Small(v) => {
                debug_assert_eq!(values.len(), self.class);
                for (dst, value) in v[offset * self.class..(offset + 1) * self.class]
                    .iter()
                    .zip(values)
                {
                    // SAFETY: no reader can access this unpublished/reclaimed
                    // slot; these sealed element types have no destructor.
                    unsafe { dst.get().write(E::from_value(*value)) };
                }
            }
            Payload::Large(v) => {
                let payload = AlignedElements::<E>::zeroed(values.len(), &self.budget)?;
                for (slot, value) in payload.iter().zip(values) {
                    unsafe { slot.get().write(E::from_value(*value)) };
                }
                let bytes = payload.allocated_bytes();
                // SAFETY: same exclusive slot ownership. Only the rare large
                // class has a per-array Box; ordinary entries are inline.
                unsafe { *v[offset].get() = Some(payload) };
                self.large_payload_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Only fully reclaimed entries may be cleared.
    unsafe fn clear(&self, offset: usize) {
        if let Payload::Large(v) = &self.payload {
            // SAFETY: every potentially borrowing guard has drained.
            let old = unsafe { (&mut *v[offset].get()).take() };
            if let Some(payload) = old {
                self.large_payload_bytes
                    .fetch_sub(payload.allocated_bytes(), Ordering::Relaxed);
            }
        }
    }

    /// Writer-owned copy; atomic level references are acquire-loaded. The old
    /// slab remains alive and unmodified after pointer publication.
    fn grown(&self, capacity: usize) -> Result<Box<Self>, StoreError> {
        let new = Self::new(self.class, capacity, &self.budget)?;
        match (&self.payload, &new.payload) {
            (Payload::Small(old), Payload::Small(dst)) => {
                for (src, dst) in old.iter().zip(dst.iter()) {
                    // SAFETY: writer lock excludes non-atomic initialization;
                    // concurrent readers only read, atomic cells use .value().
                    let value = unsafe { &*src.get() }.value();
                    unsafe { dst.get().write(E::from_value(value)) };
                }
            }
            (Payload::Large(old), Payload::Large(dst)) => {
                for (src, dst) in old.iter().zip(dst.iter()) {
                    // SAFETY: descriptors are mutated only by this writer.
                    if let Some(payload) = unsafe { &*src.get() } {
                        let copied = AlignedElements::<E>::zeroed(payload.len(), &self.budget)?;
                        for (from, to) in payload.iter().zip(copied.iter()) {
                            let value = unsafe { &*from.get() }.value();
                            unsafe { to.get().write(E::from_value(value)) };
                        }
                        unsafe { *dst.get() = Some(copied) };
                    }
                }
            }
            _ => unreachable!(),
        }
        new.large_payload_bytes.store(
            self.large_payload_bytes.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Ok(new)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryState {
    Reserved,
    Live,
    Retired,
    Free,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferMode {
    Active,
    Compacting,
    Held,
}
struct BufferState {
    class: usize,
    mode: BufferMode,
    entries: Vec<EntryState>,
    free: Vec<u32>,
    free_buffer_position: Option<usize>,
    // Includes reserved ref zero and fully reclaimed entries, but not held
    // retirements, matching BufferTypeBase's used-minus-dead growth input.
    dead_entries: usize,
}
enum Retired<E: ArrayElement, const B: u32> {
    Entry(EntryRef<B>, usize),
    Allocation(Box<Allocation<E>>),
    Buffer(u32, usize),
}
struct WriterState<E: ArrayElement, const B: u32> {
    buffers: Vec<Option<BufferState>>,
    primary: Vec<Option<u32>>,
    // Per-type nonempty free-buffer lists, as in Vespa FreeList. A buffer's
    // reverse position makes disable/remove O(1); add never scans descriptors.
    free_buffers: Vec<Vec<u32>>,
    pending: Vec<Retired<E, B>>,
    // One prepaid whole-buffer retirement per Compacting buffer. Ordinary
    // writes must preserve this credit even between compaction owner moves.
    compaction_credits: usize,
    held: VecDeque<(u64, Retired<E, B>)>,
    growths: u64,
    compactions: u64,
    next_batch: u64,
    // Conservative held-payload charge: whole buffers may overlap entry holds.
    held_charge: usize,
    #[cfg(test)]
    allocation_failure: Option<usize>,
    #[cfg(test)]
    allocation_attempts: usize,
}

impl<E: ArrayElement, const B: u32> WriterState<E, B> {
    fn charge_held(&mut self, bytes: usize) {
        self.held_charge = self.held_charge.saturating_add(bytes);
    }
    fn release_held(&mut self, bytes: usize) {
        // A saturated charge stays blocked; subtracting could undercount.
        if self.held_charge != usize::MAX {
            self.held_charge = self
                .held_charge
                .checked_sub(bytes)
                .expect("held charge invariant");
        }
    }
}

/// Stable, directly indexed read directory. Small-array reads do not load the
/// separately allocated writer descriptor or branch on its payload enum.
/// A buffer's class is immutable until generation-safe whole-buffer reuse.
struct BufferDirectoryEntry<E: ArrayElement> {
    allocation: AtomicPtr<Allocation<E>>,
    small_data: AtomicPtr<E>,
    class: AtomicUsize,
    capacity: AtomicUsize,
}
impl<E: ArrayElement> BufferDirectoryEntry<E> {
    fn empty() -> Self {
        Self { allocation: AtomicPtr::new(ptr::null_mut()), small_data: AtomicPtr::new(ptr::null_mut()),
            class: AtomicUsize::new(0), capacity: AtomicUsize::new(0) }
    }
    fn load(&self, order: Ordering) -> *mut Allocation<E> { self.allocation.load(order) }
    fn publish_read_directory(&self, allocation: *mut Allocation<E>) {
        if allocation.is_null() {
            // Whole-buffer reclamation has already excluded every old reader.
            self.small_data.store(ptr::null_mut(), Ordering::Release);
            self.capacity.store(0, Ordering::Relaxed);
            self.class.store(0, Ordering::Relaxed);
        } else {
            // SAFETY: the single writer owns this allocation and publishes the
            // directory before exposing any new reference into its payload.
            let allocation = unsafe { &*allocation };
            let data = match &allocation.payload {
                Payload::Small(values) => values.as_ptr().cast::<E>().cast_mut(),
                Payload::Large(_) => ptr::null_mut(),
            };
            self.class.store(allocation.class, Ordering::Relaxed);
            self.capacity.store(allocation.capacity, Ordering::Relaxed);
            self.small_data.store(data, Ordering::Release);
        }
    }
    fn store(&self, allocation: *mut Allocation<E>, order: Ordering) {
        self.publish_read_directory(allocation);
        self.allocation.store(allocation, order);
    }
    fn swap(&self, allocation: *mut Allocation<E>, order: Ordering) -> *mut Allocation<E> {
        self.publish_read_directory(allocation);
        self.allocation.swap(allocation, order)
    }
    fn get_mut(&mut self) -> &mut *mut Allocation<E> { self.allocation.get_mut() }
}

pub(super) struct ArrayStore<E: ArrayElement, const B: u32> {
    domain: GenerationHandler,
    config: ArrayStoreConfig,
    // Stable descriptor table. A slot is reset/reused only after the same
    // generation domain has released all old references to that buffer.
    descriptors: Box<[BufferDirectoryEntry<E>]>,
    writer: Mutex<WriterState<E, B>>,
    budget: std::sync::Arc<crate::admission::Budget>,
    _memory: crate::admission::Reservation,
}

impl<E: ArrayElement, const B: u32> ArrayStore<E, B> {
    pub(super) fn new(
        domain: GenerationHandler,
        config: ArrayStoreConfig,
    ) -> Result<Self, StoreError> {
        Self::with_budget(domain, config, crate::admission::shared())
    }

    fn with_budget(domain: GenerationHandler, config: ArrayStoreConfig, budget: std::sync::Arc<crate::admission::Budget>) -> Result<Self, StoreError> {
        if !(1..32).contains(&B)
            || config.max_buffers == 0
            || config.max_buffers > (1usize << (32 - B))
            || config.max_small_array_len == 0
            || config.specs.len() != config.max_small_array_len + 1
            || config.specs.iter().any(|spec| {
                spec.max_entries < 2
                    || spec.max_entries > (1usize << B)
                    || spec.min_entries > spec.max_entries
                    || spec.entries_for_new_buffer > spec.max_entries
                    || !spec.grow_factor.is_finite()
                    || spec.grow_factor < 0.0
                    || spec.grow_factor > 1.0
            })
        {
            return Err(StoreError::InvalidConfig);
        }
        let bytes = (config.max_buffers as u64).saturating_mul((size_of::<BufferDirectoryEntry<E>>() + size_of::<Option<BufferState>>()) as u64)
            .saturating_add((config.specs.len() as u64).saturating_mul(64)).saturating_add(4096);
        let memory = budget.reserve(bytes, "mutable vector directory").map_err(|_| StoreError::MemoryLimit)?;
        let descriptors = (0..config.max_buffers)
            .map(|_| BufferDirectoryEntry::empty())
            .collect();
        let state = WriterState {
            buffers: (0..config.max_buffers).map(|_| None).collect(),
            primary: vec![None; config.max_small_array_len + 1],
            free_buffers: vec![Vec::new(); config.max_small_array_len + 1],
            pending: Vec::new(),
            compaction_credits: 0,
            held: VecDeque::new(),
            growths: 0,
            compactions: 0,
            next_batch: 0,
            held_charge: 0,
            #[cfg(test)]
            allocation_failure: None,
            #[cfg(test)]
            allocation_attempts: 0,
        };
        Ok(Self {
            domain,
            config,
            descriptors,
            writer: Mutex::new(state),
            budget,
            _memory: memory,
        })
    }

    pub(super) fn writer(&self) -> ArrayWriter<'_, E, B> {
        ArrayWriter {
            store: self,
            state: self.writer.lock(),
        }
    }

    #[inline]
    pub(super) fn view<'a>(
        &'a self,
        guard: &'a GenerationGuard,
    ) -> Result<ArrayReadView<'a, E, B>, StoreError> {
        if !self.domain.owns_guard(guard) {
            return Err(StoreError::ForeignGuard);
        }
        Ok(ArrayReadView {
            store: self,
            _guard: guard,
        })
    }

    /// Construct access after query admission has validated this same store and
    /// guard. This removes repeated domain identity checks on every graph edge.
    /// # Safety
    /// `self.view(guard)` must have succeeded for this exact store/guard pair;
    /// neither may be substituted, and their borrows must outlive all slices.
    #[inline]
    pub(super) unsafe fn view_prevalidated<'a>(&'a self, guard: &'a GenerationGuard) -> ArrayReadView<'a, E, B> {
        ArrayReadView { store: self, _guard: guard }
    }

    /// O(1) admission charge for pending and assigned retired payloads. Whole
    /// buffer holds may conservatively also charge their individual entries.
    /// Metadata capacity is reported separately by stats(). Overflow fails closed.
    pub(super) fn held_bytes_upper_bound(&self) -> usize {
        self.writer.lock().held_charge
    }

    pub(super) fn stats(&self) -> ArrayStoreStats {
        self.writer().stats()
    }
    #[cfg(test)]
    pub(super) fn test_allocation_failure(&self, after: Option<usize>) {
        self.writer.lock().allocation_failure = after;
    }
}

impl<E: ArrayElement, const B: u32> Drop for ArrayStore<E, B> {
    fn drop(&mut self) {
        // Read views borrow Self, so no borrowed payload can outlive this drop.
        // Guards alone can outlive Self; they only own generation hold cells.
        for descriptor in &mut self.descriptors {
            let ptr = *descriptor.get_mut();
            if !ptr.is_null() {
                // SAFETY: every current descriptor owns exactly one Box.
                unsafe { drop(Box::from_raw(ptr)) };
            }
        }
        // Old physical allocations are owned by retirement queue Boxes; entry
        // and buffer retirements carry no second ownership of descriptors.
    }
}

pub(super) struct ArrayReadView<'a, E: ArrayElement, const B: u32> {
    store: &'a ArrayStore<E, B>,
    _guard: &'a GenerationGuard,
}
impl<'a, E: ArrayElement, const B: u32> ArrayReadView<'a, E, B> {
    /// Capture an existing owner for a later staged replacement while this
    /// same store/domain guard remains held. Tokens never enter graph edges.
    ///
    /// # Safety
    /// The same acquired-reference contract as get applies. In particular this
    /// cannot turn a stale raw ref plus a fresh guard into a valid retirement.
    pub(super) unsafe fn retirement_token(
        &self,
        reference: EntryRef<B>,
    ) -> Result<RetirementToken<'a, B>, StoreError> {
        if !reference.is_valid() {
            return Err(StoreError::InvalidRef);
        }
        // SAFETY: forwarded acquired-reference obligation; verifies bounds.
        unsafe {
            self.get(reference)?;
        }
        Ok(RetirementToken {
            store: self.store as *const ArrayStore<E, B> as usize,
            reference,
            _guard_and_store: PhantomData,
        })
    }
    /// Borrow an acquired, generation-protected reference.
    ///
    /// # Safety
    /// A nonzero reference must have been loaded from a current published owner
    /// AFTER this view's guard was acquired, or reached through another such
    /// continuously protected reference in this same store/domain. The owner
    /// publication must precede retirement and generation assignment. A stale
    /// integer plus a new guard is insufficient, even if its bits match a live
    /// allocation. The same guard must remain held for the returned borrow.
    /// Empty refs are always valid. Checks catch malformed bounds but do not
    /// establish allocation identity or make stale refs safe.
    #[inline]
    pub(super) unsafe fn get(&self, reference: EntryRef<B>) -> Result<&'a [E], StoreError> {
        if !reference.is_valid() {
            return Ok(&[]);
        }
        let descriptor = self
            .store
            .descriptors
            .get(reference.buffer_id() as usize)
            .ok_or(StoreError::InvalidRef)?;
        let data = descriptor.small_data.load(Ordering::Acquire);
        if !data.is_null() {
            let offset = reference.offset() as usize;
            let class = descriptor.class.load(Ordering::Relaxed);
            if offset >= descriptor.capacity.load(Ordering::Relaxed) {
                return Err(StoreError::InvalidRef);
            }
            // SAFETY: class cannot change while this generation-protected ref
            // exists. Growth may pair an old payload with a newer capacity, but
            // an old acquired ref fits both. A newly published ref happens after
            // publication of the grown payload, so its acquire cannot read the
            // earlier pointer. Guarded old allocations remain held as before.
            return Ok(unsafe { std::slice::from_raw_parts(data.add(offset * class), class) });
        }
        // Rare large-array path retains descriptor-owned variable-sized rows.
        let pointer = descriptor.load(Ordering::Acquire);
        if pointer.is_null() { return Err(StoreError::InvalidRef); }
        unsafe { (&*pointer).get(reference.offset() as usize) }
    }
}

/// Same-store identity and lifetime proof, not a new persistent reference type.
/// The borrow keeps the source read guard AND store alive through the batch.
#[derive(Clone, Copy, Debug)]
pub(super) struct RetirementToken<'a, const B: u32> {
    store: usize,
    reference: EntryRef<B>,
    _guard_and_store: PhantomData<&'a GenerationGuard>,
}

/// A staged array has no exposed raw ref until all staging has succeeded.
#[derive(Clone, Copy, Debug)]
pub(super) struct StagedIndex<'a> {
    store: usize,
    batch: u64,
    index: usize,
    _store: PhantomData<&'a ()>,
}

pub(super) struct StagedArrayBatch<'a, E: ArrayElement, const B: u32> {
    writer: ArrayWriter<'a, E, B>,
    id: u64,
    limit: usize,
    staged: Vec<EntryRef<B>>,
    old: Vec<RetirementToken<'a, B>>,
    failed: bool,
    rollback: bool,
}

impl<'a, E: ArrayElement, const B: u32> StagedArrayBatch<'a, E, B> {
    pub(super) fn stage(&mut self, values: &[u32]) -> Result<StagedIndex<'a>, StoreError> {
        if self.failed || self.staged.len() >= self.limit {
            self.failed = true;
            return Err(StoreError::InvalidBatch);
        }
        let reference = match self.writer.add(values) {
            Ok(reference) => reference,
            Err(error) => {
                // A failed add can already have held a relocated physical
                // slab. Disallow retries with new classes so the pre-reserved
                // growth-credit bound remains one per planned stage attempt.
                self.failed = true;
                return Err(error);
            }
        };
        let index = self.staged.len();
        self.staged.push(reference); // exact stage count reserved up front
        Ok(StagedIndex {
            store: self.writer.store as *const _ as usize,
            batch: self.id,
            index,
            _store: PhantomData,
        })
    }

    pub(super) fn finish_staging(self) -> Result<PreparedArrayBatch<'a, E, B>, StoreError> {
        if self.failed || self.staged.len() != self.limit {
            return Err(StoreError::IncompleteBatch);
        }
        Ok(PreparedArrayBatch { batch: self })
    }
}

impl<E: ArrayElement, const B: u32> Drop for StagedArrayBatch<'_, E, B> {
    fn drop(&mut self) {
        if self.rollback {
            // Staged arrays were never exposed to owner publication. Retire
            // them conservatively using reserved credits, without allocation
            // or a raw/immediate-reuse discard API.
            for reference in &self.staged {
                self.writer.retire_prevalidated(*reference);
            }
        }
    }
}

/// All allocations succeeded. Resolve refs for cross-store staging, preflight
/// owner slots after BOTH stores have finished growing, then begin publication.
pub(super) struct PreparedArrayBatch<'a, E: ArrayElement, const B: u32> {
    batch: StagedArrayBatch<'a, E, B>,
}
impl<'a, E: ArrayElement, const B: u32> PreparedArrayBatch<'a, E, B> {
    /// The resolved ref may be used to stage another store's arrays. It must
    /// not be published to a current owner until EVERY participating batch has
    /// called begin_publish, disarming all abort paths before the first swap.
    pub(super) fn resolve(&self, index: StagedIndex<'_>) -> Result<EntryRef<B>, StoreError> {
        if index.store != self.batch.writer.store as *const _ as usize
            || index.batch != self.batch.id
        {
            return Err(StoreError::InvalidBatch);
        }
        self.batch
            .staged
            .get(index.index)
            .copied()
            .ok_or(StoreError::InvalidBatch)
    }

    /// Disarm abort BEFORE the first owner publication. An unexpected unwind
    /// afterward conservatively retains both old/new arrays, never retires a
    /// possibly published ref. The graph coordinator must then fail closed;
    /// this primitive does not claim multi-owner visibility is atomic.
    pub(super) fn begin_publish(mut self) -> PublishingArrayBatch<'a, E, B> {
        self.batch.rollback = false;
        PublishingArrayBatch { batch: self.batch }
    }
}

pub(super) struct PublishingArrayBatch<'a, E: ArrayElement, const B: u32> {
    batch: StagedArrayBatch<'a, E, B>,
}
impl<E: ArrayElement, const B: u32> PublishingArrayBatch<'_, E, B> {
    /// No allocation, user callback, or recoverable error remains here.
    ///
    /// # Safety
    /// Every old ref's current owner was already replaced/invalidated, and all
    /// staged refs were adopted by the intended owners. The topology writer
    /// must remain exclusive across preflight, owner publication and this call.
    pub(super) unsafe fn finish_after_publish(mut self) {
        for token in &self.batch.old {
            self.batch.writer.retire_prevalidated(token.reference);
        }
    }
}

/// Opaque current-allocation pointer resolved only after staging has ended.
/// It is usable exclusively with its originating PublishingArrayBatch.
pub(super) struct AtomicReplacement<'a> {
    store: usize,
    batch: u64,
    pointer: *const AtomicU32,
    value: u32,
    _owner: PhantomData<&'a AtomicU32>,
}

impl<'a, const B: u32> PreparedArrayBatch<'a, AtomicU32, B> {
    pub(super) fn preflight_replace(
        &self,
        owner: RetirementToken<'a, B>,
        index: usize,
        expected: u32,
        value: u32,
    ) -> Result<AtomicReplacement<'a>, StoreError> {
        let writer = &self.batch.writer;
        if owner.store != writer.store as *const _ as usize {
            return Err(StoreError::ForeignStore);
        }
        writer.checked_state(owner.reference)?;
        // SAFETY: retained writer owns current liveness; staging/growth ended.
        let slice = unsafe {
            writer
                .allocation(owner.reference.buffer_id())
                .get(owner.reference.offset() as usize)?
        };
        let entry = slice.get(index).ok_or(StoreError::IndexOutOfBounds)?;
        if entry.load(Ordering::Acquire) != expected {
            return Err(StoreError::OwnerChanged);
        }
        Ok(AtomicReplacement {
            store: owner.store,
            batch: self.batch.id,
            pointer: entry,
            value,
            _owner: PhantomData,
        })
    }
}
impl<const B: u32> PublishingArrayBatch<'_, AtomicU32, B> {
    /// # Safety
    /// `replacement` was preflighted by THIS batch, and the topology writer
    /// retained exclusivity. No further array staging/growth occurs after that
    /// preflight. This method intentionally performs no fallible revalidation.
    pub(super) unsafe fn replace_prevalidated(&mut self, replacement: AtomicReplacement<'_>) {
        debug_assert_eq!(
            replacement.store,
            self.batch.writer.store as *const _ as usize
        );
        debug_assert_eq!(replacement.batch, self.batch.id);
        // SAFETY: caller guarantees same live writer/current allocation.
        unsafe { (&*replacement.pointer).store(replacement.value, Ordering::Release) };
    }
}

pub(super) struct ArrayWriter<'a, E: ArrayElement, const B: u32> {
    store: &'a ArrayStore<E, B>,
    state: MutexGuard<'a, WriterState<E, B>>,
}

impl<'a, E: ArrayElement, const B: u32> ArrayWriter<'a, E, B> {
    /// Build only the final arrays selected by the graph overlay. Old refs are
    /// unique preexisting owners, not intermediate arrays from the same batch.
    pub(super) fn stage_batch(
        mut self,
        count: usize,
        old: &[RetirementToken<'a, B>],
    ) -> Result<StagedArrayBatch<'a, E, B>, StoreError> {
        let credits = count
            .checked_mul(2)
            .and_then(|n| n.checked_add(old.len()))
            .ok_or(StoreError::AllocationFailed)?;
        let id = self
            .state
            .next_batch
            .checked_add(1)
            .ok_or(StoreError::InvalidBatch)?;
        self.allocation_checkpoint()?;
        // Every link update retires its old lists here: a cheap hasher, not
        // SipHash, for this duplicate check on the graph's write path.
        let mut unique: ahash::AHashSet<_> = ahash::AHashSet::new();
        unique
            .try_reserve(old.len())
            .map_err(|_| StoreError::AllocationFailed)?;
        for token in old {
            if token.store != self.store as *const _ as usize {
                return Err(StoreError::ForeignStore);
            }
            if !unique.insert(token.reference) {
                return Err(StoreError::DuplicateRetirement);
            }
            let state = self.checked_state(token.reference)?;
            let len = state.entries.len();
            self.reserve_reuse_capacity(token.reference.buffer_id(), len)?;
        }
        self.allocation_checkpoint()?;
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(count)
            .map_err(|_| StoreError::AllocationFailed)?;
        let mut old_refs = Vec::new();
        old_refs
            .try_reserve_exact(old.len())
            .map_err(|_| StoreError::AllocationFailed)?;
        old_refs.extend_from_slice(old);
        self.allocation_checkpoint()?;
        // At most Nnew growth allocations, Nnew abort retirements, and Nold
        // commit retirements. The writer stays owned, so nobody spends credit.
        self.reserve_pending(credits)?;
        self.state.next_batch = id;
        Ok(StagedArrayBatch {
            writer: self,
            id,
            limit: count,
            staged,
            old: old_refs,
            failed: false,
            rollback: true,
        })
    }

    fn allocation_checkpoint(&mut self) -> Result<(), StoreError> {
        #[cfg(test)]
        {
            self.state.allocation_attempts += 1;
            if let Some(remaining) = self.state.allocation_failure.as_mut() {
                if *remaining == 0 {
                    return Err(StoreError::AllocationFailed);
                }
                *remaining -= 1;
            }
        }
        Ok(())
    }

    fn reserve_pending(&mut self, additional: usize) -> Result<(), StoreError> {
        let reserved = additional
            .checked_add(self.state.compaction_credits)
            .ok_or(StoreError::AllocationFailed)?;
        self.state
            .pending
            .try_reserve(reserved)
            .map_err(|_| StoreError::AllocationFailed)
    }

    fn reserve_reuse_capacity(&mut self, id: u32, len: usize) -> Result<(), StoreError> {
        let free = &mut self.state.buffers[id as usize].as_mut().unwrap().free;
        free.try_reserve(len.saturating_sub(free.len()))
            .map_err(|_| StoreError::AllocationFailed)
    }

    fn allocation(&self, id: u32) -> &Allocation<E> {
        let pointer = self.store.descriptors[id as usize].load(Ordering::Relaxed);
        // SAFETY: writer serializes descriptor ownership; caller validated ID.
        unsafe { &*pointer }
    }

    fn checked_state(&self, reference: EntryRef<B>) -> Result<&BufferState, StoreError> {
        if !reference.is_valid() {
            return Err(StoreError::InvalidRef);
        }
        let state = self
            .state
            .buffers
            .get(reference.buffer_id() as usize)
            .and_then(Option::as_ref)
            .ok_or(StoreError::InvalidRef)?;
        if state.entries.get(reference.offset() as usize) != Some(&EntryState::Live) {
            return Err(StoreError::NotLive);
        }
        Ok(state)
    }

    fn class_live_entries(&self, class: usize) -> usize {
        self.state
            .buffers
            .iter()
            .filter_map(Option::as_ref)
            .filter(|state| state.class == class && state.mode != BufferMode::Held)
            .map(|state| state.entries.len() - state.dead_entries)
            .sum()
    }

    fn active_class_buffers(&self, class: usize) -> usize {
        self.state
            .buffers
            .iter()
            .filter_map(Option::as_ref)
            .filter(|state| state.class == class && state.mode != BufferMode::Held)
            .count()
    }

    fn scaled_switch_threshold(&self, class: usize) -> usize {
        let spec = self.store.config.specs[class];
        let count = self.active_class_buffers(class);
        if count <= 1 || spec.entries_for_new_buffer == 0 {
            return spec.entries_for_new_buffer;
        }
        // BufferTypeBase::get_scaled_num_entries_for_new_buffer uses double
        // pow after promoting its float grow factor, then truncates the result.
        let scale = (1.0 + f64::from(spec.grow_factor)).powf((count - 1) as f64);
        ((spec.entries_for_new_buffer as f64 * scale) as usize).min(spec.max_entries)
    }

    fn capacity(&self, class: usize, needed: usize, used: usize) -> Result<usize, StoreError> {
        let spec = self.store.config.specs[class];
        // Pinned buffer_type.cpp106-142: class-wide used-minus-dead, float
        // multiplication followed by truncation. A held entry still contributes
        // until reclaimed; a whole held buffer no longer belongs to the class.
        let growth = (self.class_live_entries(class) as f32 * spec.grow_factor) as usize;
        let wanted = needed
            .max(spec.min_entries)
            .max(used.saturating_add(growth))
            .min(spec.max_entries);
        if wanted < needed {
            return Err(StoreError::AddressSpaceExhausted);
        }
        let entry_bytes = if class == 0 {
            size_of::<Option<AlignedElements<E>>>()
        } else {
            class * size_of::<E>()
        };
        let bytes = wanted
            .checked_mul(entry_bytes)
            .ok_or(StoreError::AllocationFailed)?;
        // Pinned bufferstate.cpp roundUpToMatchAllocator: powers of two below
        // the huge-page size, then huge-page multiples, capped by max entries.
        // Rust still uses its configured allocator; this ports requested sizes,
        // not a claim to install Vespa's vespamalloc or mmap allocator.
        const HUGE_PAGE: usize = 2 * 1024 * 1024;
        let rounded = if bytes < HUGE_PAGE {
            bytes
                .checked_next_power_of_two()
                .ok_or(StoreError::AllocationFailed)?
        } else {
            bytes
                .checked_add(HUGE_PAGE - 1)
                .ok_or(StoreError::AllocationFailed)?
                / HUGE_PAGE
                * HUGE_PAGE
        };
        let maximum_bytes = spec
            .max_entries
            .checked_mul(entry_bytes)
            .ok_or(StoreError::AllocationFailed)?;
        Ok(rounded.min(maximum_bytes) / entry_bytes)
    }

    fn activate(&mut self, class: usize) -> Result<u32, StoreError> {
        self.allocation_checkpoint()?;
        let id = self
            .state
            .buffers
            .iter()
            .position(Option::is_none)
            .ok_or(StoreError::AddressSpaceExhausted)?;
        // Ref zero is reserved only in physical buffer zero.
        let reserved = usize::from(id == 0);
        let capacity = self.capacity(class, reserved + 1, 0)?;
        let allocation = Allocation::<E>::new(class, capacity, &self.store.budget)?;
        let mut entries = Vec::new();
        entries
            .try_reserve(reserved + 1)
            .map_err(|_| StoreError::AllocationFailed)?;
        entries.resize(reserved, EntryState::Reserved);
        let mut free = Vec::new();
        free.try_reserve(reserved + 1)
            .map_err(|_| StoreError::AllocationFailed)?;
        let count = self.active_class_buffers(class) + 1;
        let list = &mut self.state.free_buffers[class];
        list.try_reserve(count.saturating_sub(list.len()))
            .map_err(|_| StoreError::AllocationFailed)?;
        self.state.buffers[id] = Some(BufferState {
            class,
            mode: BufferMode::Active,
            entries,
            free,
            free_buffer_position: None,
            dead_entries: reserved,
        });
        self.store.descriptors[id].store(Box::into_raw(allocation), Ordering::Release);
        self.state.primary[class] = Some(id as u32);
        Ok(id as u32)
    }

    fn grow(&mut self, id: u32) -> Result<(), StoreError> {
        self.allocation_checkpoint()?;
        let state = self.state.buffers[id as usize].as_ref().unwrap();
        let old = self.allocation(id);
        let capacity = self.capacity(state.class, state.entries.len() + 1, state.entries.len())?;
        if capacity <= old.capacity {
            return Err(StoreError::AddressSpaceExhausted);
        }
        let grown = old.grown(capacity)?;
        self.reserve_pending(1)?;
        let old = self.store.descriptors[id as usize].swap(Box::into_raw(grown), Ordering::AcqRel);
        // SAFETY: swap transferred unique old descriptor ownership into hold.
        let old = unsafe { Box::from_raw(old) };
        self.state.charge_held(old.owned_bytes());
        self.state.pending.push(Retired::Allocation(old));
        self.state.growths += 1;
        Ok(())
    }

    fn insertion_buffer(&mut self, class: usize) -> Result<u32, StoreError> {
        let Some(mut id) = self.state.primary[class] else {
            return self.activate(class);
        };
        let state = self.state.buffers[id as usize].as_ref().unwrap();
        if state.entries.len() < self.allocation(id).capacity {
            return Ok(id);
        }
        let spec = self.store.config.specs[class];
        if state.entries.len() + 1 >= self.scaled_switch_threshold(class) {
            // Pinned consider_grow_active_buffer: only resume/grow an existing
            // class buffer once there are at least four noncompacting buffers.
            let mut active = 0;
            let mut least = id;
            let mut used = state.entries.len();
            for (candidate, state) in self.state.buffers.iter().enumerate() {
                let Some(state) = state else { continue };
                if state.class == class && state.mode == BufferMode::Active {
                    active += 1;
                    // Pinned selection retains current primary on equal size.
                    if state.entries.len() < used {
                        least = candidate as u32;
                        used = state.entries.len();
                    }
                }
            }
            if spec.entries_for_new_buffer != 0 && active >= 4 {
                if used < spec.max_entries {
                    id = least;
                    self.state.primary[class] = Some(id);
                    if used == self.allocation(id).capacity {
                        self.grow(id)?;
                    }
                    return Ok(id);
                }
            }
            return self.activate(class);
        }
        self.grow(id)?;
        Ok(id)
    }

    pub(super) fn add(&mut self, values: &[u32]) -> Result<EntryRef<B>, StoreError> {
        if values.is_empty() {
            return Ok(EntryRef::INVALID);
        }
        let class = if values.len() <= self.store.config.max_small_array_len {
            values.len()
        } else {
            0
        };
        let free = self.state.free_buffers[class].last().map(|&id| {
            let state = self.state.buffers[id as usize].as_ref().unwrap();
            debug_assert_eq!(state.mode, BufferMode::Active);
            (
                id,
                *state.free.last().expect("listed free buffer is nonempty"),
            )
        });
        let (id, offset, recycled) = match free {
            Some((id, offset)) => (id, offset, true),
            None => {
                let id = self.insertion_buffer(class)?;
                self.allocation_checkpoint()?;
                let state = self.state.buffers[id as usize].as_mut().unwrap();
                state
                    .entries
                    .try_reserve(1)
                    .map_err(|_| StoreError::AllocationFailed)?;
                let offset = state.entries.len() as u32;
                self.reserve_reuse_capacity(id, offset as usize + 1)?;
                (id, offset, false)
            }
        };
        if class == 0 {
            self.allocation_checkpoint()?;
        }
        // SAFETY: chosen slot is unpublished or fully reclaimed; no view can
        // borrow it under the required acquired-reference protocol.
        unsafe { self.allocation(id).write(offset as usize, values)? };
        let state = self.state.buffers[id as usize].as_mut().unwrap();
        if recycled {
            // Mutation must execute in optimized builds too; debug assertions
            // are only diagnostics and must never own allocator side effects.
            let popped = state.free.pop();
            debug_assert_eq!(popped, Some(offset));
            state.entries[offset as usize] = EntryState::Live;
            state.dead_entries -= 1;
        } else {
            state.entries.push(EntryState::Live);
        }
        if recycled && state.free.is_empty() {
            self.unlist_free_buffer(id);
        }
        Ok(EntryRef::new(id, offset))
    }

    fn list_free_buffer(&mut self, id: u32) {
        let state = self.state.buffers[id as usize].as_ref().unwrap();
        if state.mode != BufferMode::Active
            || state.free.is_empty()
            || state.free_buffer_position.is_some()
        {
            return;
        }
        let class = state.class;
        let position = self.state.free_buffers[class].len();
        self.state.free_buffers[class].push(id);
        self.state.buffers[id as usize]
            .as_mut()
            .unwrap()
            .free_buffer_position = Some(position);
    }

    fn unlist_free_buffer(&mut self, id: u32) {
        let state = self.state.buffers[id as usize].as_ref().unwrap();
        let Some(position) = state.free_buffer_position else {
            return;
        };
        let class = state.class;
        self.state.free_buffers[class].swap_remove(position);
        if let Some(&moved) = self.state.free_buffers[class].get(position) {
            self.state.buffers[moved as usize]
                .as_mut()
                .unwrap()
                .free_buffer_position = Some(position);
        }
        self.state.buffers[id as usize]
            .as_mut()
            .unwrap()
            .free_buffer_position = None;
    }

    pub(super) fn read_current(&self, reference: EntryRef<B>) -> Result<Vec<u32>, StoreError> {
        if !reference.is_valid() {
            return Ok(Vec::new());
        }
        self.checked_state(reference)?;
        // SAFETY: writer lock owns liveness and prevents concurrent reuse.
        let slice = unsafe {
            self.allocation(reference.buffer_id())
                .get(reference.offset() as usize)?
        };
        Ok(slice.iter().map(ArrayElement::value).collect())
    }

    /// Caller must first replace/invalidate every current owner of this ref.
    /// Existing acquired refs remain readable until their guards drain.
    pub(super) fn remove(&mut self, reference: EntryRef<B>) -> Result<(), StoreError> {
        if !reference.is_valid() {
            return Ok(());
        }
        self.checked_state(reference)?;
        self.reserve_pending(1)?;
        self.retire_prevalidated(reference);
        Ok(())
    }

    /// Caller holds writer, validated Live once, and reserved a pending slot.
    /// No allocation or user callback belongs in the publication/abort phase.
    fn retire_prevalidated(&mut self, reference: EntryRef<B>) {
        if !reference.is_valid() {
            return;
        }
        debug_assert!(
            self.state.pending.len() + self.state.compaction_credits
                < self.state.pending.capacity()
        );
        self.state.buffers[reference.buffer_id() as usize]
            .as_mut()
            .unwrap()
            .entries[reference.offset() as usize] = EntryState::Retired;
        let allocation = self.allocation(reference.buffer_id());
        let bytes = if allocation.class == 0 {
            // SAFETY: validated live entry under this writer.
            size_of::<Option<AlignedElements<E>>>()
                + unsafe {
                    allocation
                        .get(reference.offset() as usize)
                        .expect("live large entry")
                }
                .len().div_ceil(8) * 32
        } else {
            allocation.class * size_of::<E>()
        };
        self.state.charge_held(bytes);
        self.state.pending.push(Retired::Entry(reference, bytes));
    }

    /// Assign all pending holds CURRENT generation before the graph coordinator
    /// advances the shared domain. On failure, pending holds remain retained.
    pub(super) fn assign_generation(&mut self) -> Result<(), StoreError> {
        let count = self.state.pending.len();
        self.state
            .held
            .try_reserve(count)
            .map_err(|_| StoreError::AllocationFailed)?;
        let generation = self.store.domain.current_generation();
        let state = &mut *self.state;
        state
            .held
            .extend(state.pending.drain(..).map(|item| (generation, item)));
        Ok(())
    }

    pub(super) fn reclaim(&mut self) -> usize {
        let oldest = self.store.domain.update_oldest_used_generation();
        let count = self
            .state
            .held
            .iter()
            .take_while(|(g, _)| *g < oldest)
            .count();
        // Match DataStoreBase: entry refs are reclaimed before whole buffers.
        // Two passes over the existing queue avoid a cleanup-time allocation.
        for index in 0..count {
            if let Retired::Entry(reference, _) = self.state.held[index].1 {
                let id = reference.buffer_id();
                let offset = reference.offset();
                // SAFETY: strict generation boundary proves readers drained.
                unsafe { self.allocation(id).clear(offset as usize) };
                let state = self.state.buffers[id as usize].as_mut().unwrap();
                debug_assert_eq!(state.entries[offset as usize], EntryState::Retired);
                state.entries[offset as usize] = EntryState::Free;
                state.dead_entries += 1;
                if state.mode == BufferMode::Active {
                    state.free.push(offset);
                }
                self.list_free_buffer(id);
            }
        }
        for _ in 0..count {
            let item = self.state.held.pop_front().unwrap().1;
            let bytes = match &item {
                Retired::Entry(_, bytes) | Retired::Buffer(_, bytes) => *bytes,
                Retired::Allocation(allocation) => allocation.owned_bytes(),
            };
            self.state.release_held(bytes);
            if let Retired::Buffer(id, _) = item {
                debug_assert_eq!(
                    self.state.buffers[id as usize].as_ref().unwrap().mode,
                    BufferMode::Held
                );
                let pointer =
                    self.store.descriptors[id as usize].swap(ptr::null_mut(), Ordering::AcqRel);
                // SAFETY: no current owners (finish precondition), and all old
                // owners/entry holds have drained before buffer ID reuse.
                unsafe { drop(Box::from_raw(pointer)) };
                self.state.buffers[id as usize] = None;
            }
            // Physical growth allocations are uniquely owned by this item and
            // drop here. Large payload copies have independent ownership.
        }
        count
    }

    /// Pinned default CompactionStrategy: 64Ki slack, 5% dead memory,
    /// 20% dead address space, at most one worst buffer per reason. Candidate
    /// ordering is descending dead amount, not dead/used ratio.
    pub(super) fn default_compaction_buffers(&self) -> Vec<u32> {
        let mut used_bytes = 0usize;
        let mut dead_bytes = 0usize;
        let mut used_entries = 0usize;
        let mut dead_entries = 0usize;
        let mut worst_memory = (0usize, 0u32);
        let mut worst_address = (0usize, 0u32);
        for (id, state) in self.state.buffers.iter().enumerate() {
            let Some(state) = state else { continue };
            if state.mode != BufferMode::Active { continue; }
            let unit = if state.class == 0 { size_of::<Option<AlignedElements<E>>>() } else { state.class * size_of::<E>() };
            let mut used = state.entries.len() * unit;
            let dead = state.dead_entries * unit;
            if let Payload::Large(values) = &self.allocation(id as u32).payload {
                for value in values.iter().take(state.entries.len()) {
                    // Writer lock protects these descriptors; a free large
                    // entry has already released its separately owned payload.
                    if let Some(row) = unsafe { &*value.get() } { used += row.allocated_bytes(); }
                }
            }
            used_bytes += used;
            dead_bytes += dead;
            used_entries += state.entries.len();
            dead_entries += state.dead_entries;
            if dead > worst_memory.0 { worst_memory = (dead, id as u32); }
            if state.dead_entries > worst_address.0 { worst_address = (state.dead_entries, id as u32); }
        }
        let mut selected = Vec::with_capacity(2);
        if dead_bytes >= 0x10000 && dead_bytes as f64 > used_bytes as f64 * f64::from(0.05f32) {
            selected.push(worst_memory.1);
        }
        if dead_entries >= 0x10000 && dead_entries as f64 > used_entries as f64 * f64::from(0.2f32)
            && !selected.contains(&worst_address.1)
        {
            selected.push(worst_address.1);
        }
        selected
    }

    pub(super) fn start_compaction(&mut self, ids: &[u32]) -> Result<(), StoreError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut unique: ahash::AHashSet<u32> = ahash::AHashSet::new();
        unique
            .try_reserve(ids.len())
            .map_err(|_| StoreError::AllocationFailed)?;
        for &id in ids {
            let state = self
                .state
                .buffers
                .get(id as usize)
                .and_then(Option::as_ref)
                .ok_or(StoreError::InvalidRef)?;
            if !unique.insert(id) || state.mode != BufferMode::Active {
                return Err(StoreError::NotCompacting);
            }
        }
        // Everything that can allocate for finish/abort is admitted before
        // changing a buffer mode or allowing the caller to publish an owner.
        self.allocation_checkpoint()?;
        self.reserve_pending(ids.len())?;
        for &id in ids {
            let state = self.state.buffers[id as usize].as_ref().unwrap();
            let len = state.entries.len();
            let class = state.class;
            self.reserve_reuse_capacity(id, len)?;
            let count = self.active_class_buffers(class);
            let list = &mut self.state.free_buffers[class];
            list.try_reserve(count.saturating_sub(list.len()))
                .map_err(|_| StoreError::AllocationFailed)?;
        }
        self.state.compaction_credits += ids.len(); // checked by reserve_pending
        for &id in ids {
            self.unlist_free_buffer(id);
            let state = self.state.buffers[id as usize].as_mut().unwrap();
            state.mode = BufferMode::Compacting;
            let class = state.class;
            state.free.clear();
            if self.state.primary[class] == Some(id) {
                self.state.primary[class] = None;
            }
        }
        Ok(())
    }

    /// Copy only; caller release-publishes the new ref, then remove(old). If
    /// publication is abandoned, remove(new). Never finish a partial buffer.
    pub(super) fn move_on_compact(&mut self, old: EntryRef<B>) -> Result<EntryRef<B>, StoreError> {
        if self.checked_state(old)?.mode != BufferMode::Compacting {
            return Err(StoreError::NotCompacting);
        }
        let values = self.read_current(old)?;
        self.add(&values)
    }

    pub(super) fn finish_compaction(&mut self, id: u32) -> Result<(), StoreError> {
        let state = self
            .state
            .buffers
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or(StoreError::InvalidRef)?;
        if state.mode != BufferMode::Compacting {
            return Err(StoreError::NotCompacting);
        }
        if state.entries.contains(&EntryState::Live) {
            return Err(StoreError::LiveOwnersRemain);
        }
        // The selected buffer owns this credit across writer lock releases,
        // intervening writes, generation assignment and partial sweep failure.
        debug_assert!(self.state.compaction_credits > 0);
        debug_assert!(
            self.state.pending.len() + self.state.compaction_credits
                <= self.state.pending.capacity()
        );
        self.state.compaction_credits -= 1;
        self.state.buffers[id as usize].as_mut().unwrap().mode = BufferMode::Held;
        let bytes = self.allocation(id).owned_bytes();
        self.state.charge_held(bytes);
        self.state.pending.push(Retired::Buffer(id, bytes));
        self.state.compactions += 1;
        Ok(())
    }

    /// Failed/partial compaction retains old refs. Reactivate the remaining
    /// source buffer; entries already published elsewhere remain retired.
    pub(super) fn abort_compaction(&mut self, id: u32) -> Result<(), StoreError> {
        let state = self
            .state
            .buffers
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .ok_or(StoreError::InvalidRef)?;
        if state.mode != BufferMode::Compacting {
            return Err(StoreError::NotCompacting);
        }
        state.mode = BufferMode::Active;
        state.free.clear();
        for (i, entry) in state.entries.iter().enumerate() {
            if *entry == EntryState::Free {
                state.free.push(i as u32);
            }
        }
        let class = state.class;
        debug_assert!(self.state.compaction_credits > 0);
        self.state.compaction_credits -= 1;
        if self.state.primary[class].is_none() {
            self.state.primary[class] = Some(id);
        }
        self.list_free_buffer(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::generation::GenerationLimits;
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    fn domain() -> GenerationHandler {
        GenerationHandler::new(GenerationLimits::default()).unwrap()
    }
    fn config(
        max_small: usize,
        buffers: usize,
        min: usize,
        max: usize,
        switch: usize,
    ) -> ArrayStoreConfig {
        ArrayStoreConfig {
            max_small_array_len: max_small,
            max_buffers: buffers,
            specs: vec![
                AllocationSpec {
                    min_entries: min,
                    max_entries: max,
                    entries_for_new_buffer: switch,
                    grow_factor: 0.3,
                };
                max_small + 1
            ],
        }
    }
    fn commit<E: ArrayElement, const B: u32>(store: &ArrayStore<E, B>, domain: &GenerationHandler) {
        store.writer().assign_generation().unwrap();
        domain.inc_generation().unwrap();
    }
    fn assert_accounting(stats: ArrayStoreStats) {
        assert!(
            stats.allocated_bytes
                >= stats.used_bytes + stats.held_bytes + stats.dead_bytes + stats.metadata_bytes,
            "{stats:?}"
        );
    }

    #[test]
    fn resident_admission_refuses_growth_without_changing_live_rows() {
        let d = domain();
        let budget = crate::admission::Budget::new(1 << 20);
        let store = ArrayStore::<u32, 12>::with_budget(d.clone(), config(2, 8, 2, 4096, 4096), budget.clone()).unwrap();
        let first = store.writer().add(&[7, 8]).unwrap();
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        let old = unsafe { view.get(first) }.unwrap();
        let held = budget.used();
        budget.set_limit(held);
        let mut refused = false;
        for i in 0..4096 {
            match store.writer().add(&[i, i]) {
                Ok(_) => (),
                Err(StoreError::MemoryLimit) => { refused = true; break; },
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
        assert!(refused);
        assert_eq!(old, &[7, 8]);
        assert_eq!(budget.used(), held);
        drop(view);drop(guard);drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn retired_slab_remains_admitted_until_generation_reclamation() {
        let d = domain();
        let budget = crate::admission::Budget::new(1 << 20);
        let store = ArrayStore::<u32, 12>::with_budget(d.clone(), config(2, 8, 2, 4096, 4096), budget.clone()).unwrap();
        let first = store.writer().add(&[7, 8]).unwrap();
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        let old = unsafe { view.get(first) }.unwrap();
        for i in 0..1024 { store.writer().add(&[i, i]).unwrap(); }
        assert!(store.stats().growths > 0);
        commit(&store, &d);
        let held = budget.used();
        store.writer().reclaim();
        assert_eq!(budget.used(), held);
        assert_eq!(old, &[7, 8]);
        drop(view);drop(guard);
        store.writer().reclaim();
        assert!(budget.used() < held);
        drop(store);assert_eq!(budget.used(), 0);
    }

    #[test]
    fn resident_limit_preserves_typed_core_memory_error() {
        assert!(matches!(super::super::failure_error(StoreError::MemoryLimit), crate::RusticError::Memory(_)));
    }

    #[test]
    fn compaction_admission_failure_does_not_disable_current_buffers() {
        let store = ArrayStore::<u32, 8>::new(domain(), config(2, 8, 2, 128, 128)).unwrap();
        let old = store.writer().add(&[7]).unwrap();
        let id = old.buffer_id();
        let primary = store.writer.lock().primary.clone();
        store.writer.lock().allocation_failure = Some(0);
        assert_eq!(
            store.writer().start_compaction(&[id]),
            Err(StoreError::AllocationFailed)
        );
        let writer = store.writer();
        assert_eq!(writer.state.compaction_credits, 0);
        assert_eq!(writer.state.primary, primary);
        assert_eq!(
            writer.state.buffers[id as usize].as_ref().unwrap().mode,
            BufferMode::Active
        );
        assert_eq!(writer.read_current(old).unwrap(), [7]);
    }

    #[test]
    fn compaction_finish_credit_survives_writes_and_generation_assignment() {
        let d = domain();
        let store = ArrayStore::<u32, 8>::new(d.clone(), config(2, 8, 2, 128, 128)).unwrap();
        let guard = d.take_guard().unwrap();
        let old = store.writer().add(&[7]).unwrap();
        let id = old.buffer_id();
        store.writer().start_compaction(&[id]).unwrap();
        let moved = store.writer().move_on_compact(old).unwrap();
        store.writer().remove(old).unwrap();
        // Distinct class forces ordinary allocation growth while the pending
        // whole-buffer retirement credit must stay unavailable to those writes.
        for n in 0..64 {
            let noise = store.writer().add(&[n, n]).unwrap();
            store.writer().remove(noise).unwrap();
            if n % 16 == 0 {
                commit(&store, &d);
            }
            let state = store.writer.lock();
            assert_eq!(state.compaction_credits, 1);
            assert!(state.pending.capacity() >= state.pending.len() + 1);
        }
        let (capacity, attempts) = {
            let mut state = store.writer.lock();
            state.allocation_failure = Some(0);
            (state.pending.capacity(), state.allocation_attempts)
        };
        store.writer().finish_compaction(id).unwrap();
        {
            let state = store.writer.lock();
            assert_eq!(state.pending.capacity(), capacity);
            assert_eq!(state.allocation_attempts, attempts);
            assert_eq!(state.compaction_credits, 0);
            assert_eq!(
                state.buffers[id as usize].as_ref().unwrap().mode,
                BufferMode::Held
            );
        }
        assert_eq!(store.writer().read_current(moved).unwrap(), [7]);
        assert_eq!(
            unsafe { store.view(&guard).unwrap().get(old) }.unwrap(),
            [7]
        );
        commit(&store, &d);
        assert_eq!(store.writer().reclaim(), 0);
        drop(guard);
        assert!(store.writer().reclaim() > 0);
        assert!(store.writer.lock().buffers[id as usize].is_none());
    }

    #[test]
    fn partial_compaction_abort_reuses_prepaid_free_list_capacity() {
        let d = domain();
        let store = ArrayStore::<u32, 8>::new(d.clone(), config(1, 8, 8, 128, 128)).unwrap();
        let a = store.writer().add(&[7]).unwrap();
        let b = store.writer().add(&[8]).unwrap();
        let id = a.buffer_id();
        assert_eq!(b.buffer_id(), id);
        store.writer().start_compaction(&[id]).unwrap();
        let moved = store.writer().move_on_compact(a).unwrap();
        store.writer().remove(a).unwrap();
        commit(&store, &d);
        store.writer().reclaim();
        let (free_capacity, list_capacity, attempts) = {
            let mut state = store.writer.lock();
            state.allocation_failure = Some(0);
            (
                state.buffers[id as usize].as_ref().unwrap().free.capacity(),
                state.free_buffers[1].capacity(),
                state.allocation_attempts,
            )
        };
        store.writer().abort_compaction(id).unwrap();
        let writer = store.writer();
        assert_eq!(writer.state.compaction_credits, 0);
        assert_eq!(writer.state.allocation_attempts, attempts);
        assert_eq!(
            writer.state.buffers[id as usize]
                .as_ref()
                .unwrap()
                .free
                .capacity(),
            free_capacity
        );
        assert_eq!(writer.state.free_buffers[1].capacity(), list_capacity);
        assert!(writer.state.buffers[id as usize]
            .as_ref()
            .unwrap()
            .free
            .contains(&a.offset()));
        assert_eq!(writer.read_current(b).unwrap(), [8]);
        assert_eq!(writer.read_current(moved).unwrap(), [7]);
    }

    #[test]
    fn constant_time_held_charge_covers_entries_growth_and_compacted_buffers() {
        for values in [vec![10, 20], vec![10, 20, 30, 40, 50]] {
            let d = domain();
            let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 8, 2, 16, 16)).unwrap();
            let guard = d.take_guard().unwrap();
            let mut refs = Vec::new();
            for _ in 0..7 {
                refs.push(store.writer().add(&values).unwrap());
                assert!(store.held_bytes_upper_bound() >= store.stats().held_bytes);
            }
            assert!(store.stats().growths > 0);
            let id = refs[0].buffer_id();
            store.writer().start_compaction(&[id]).unwrap();
            for old in refs {
                assert_eq!(old.buffer_id(), id);
                let new = store.writer().move_on_compact(old).unwrap();
                assert_eq!(store.writer().read_current(new).unwrap(), values);
                store.writer().remove(old).unwrap();
                assert!(store.held_bytes_upper_bound() >= store.stats().held_bytes);
            }
            store.writer().finish_compaction(id).unwrap();
            let pending_capacity = store.writer.lock().pending.capacity();
            let before = store.held_bytes_upper_bound();
            assert!(before > 0);
            commit(&store, &d);
            assert_eq!(store.writer.lock().pending.capacity(), pending_capacity);
            assert_eq!(store.held_bytes_upper_bound(), before);
            assert_eq!(store.writer().reclaim(), 0);
            assert!(store.held_bytes_upper_bound() >= store.stats().held_bytes);
            drop(guard);
            assert!(store.writer().reclaim() > 0);
            assert_eq!(store.held_bytes_upper_bound(), 0);
            assert_eq!(store.stats().held_bytes, 0);
            assert_accounting(store.stats());
        }
    }

    #[test]
    fn held_charge_overflow_remains_fail_closed_after_reclaim() {
        let store = ArrayStore::<u32, 4>::new(domain(), config(2, 4, 2, 16, 16)).unwrap();
        let mut state = store.writer.lock();
        state.held_charge = usize::MAX - 1;
        state.charge_held(2);
        state.release_held(2);
        assert_eq!(state.held_charge, usize::MAX);
    }

    #[test]
    fn packed_reference_layout_empty_arrays_bounds_and_domain() {
        assert_eq!(size_of::<LevelsRef>(), 4);
        assert_eq!(size_of::<LinksRef>(), 4);
        let last = LevelsRef::new(1023, (1 << 22) - 1);
        assert_eq!(last.raw(), u32::MAX);
        assert_eq!(last.buffer_id(), 1023);
        assert_eq!(last.offset(), (1 << 22) - 1);
        assert_eq!(LinksRef::new(4095, (1 << 20) - 1).raw(), u32::MAX);
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(3, 4, 2, 16, 16)).unwrap();
        assert_eq!(store.writer().add(&[]).unwrap(), EntryRef::INVALID);
        let first = store.writer().add(&[1, 2]).unwrap();
        assert!(first.is_valid());
        assert_eq!(first.buffer_id(), 0);
        assert_eq!(first.offset(), 1);
        assert_eq!(
            store.writer().read_current(EntryRef::from_raw(u32::MAX)),
            Err(StoreError::InvalidRef)
        );
        assert_eq!(
            store.writer().read_current(EntryRef::new(0, 15)),
            Err(StoreError::NotLive)
        );
        let foreign = domain().take_guard().unwrap();
        assert!(matches!(
            store.view(&foreign),
            Err(StoreError::ForeignGuard)
        ));
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: first is still the writer's current live owner, acquired
        // after guard pin; INVALID is always a valid empty slice.
        assert_eq!(unsafe { view.get(first) }.unwrap(), [1, 2]);
        assert!(unsafe { view.get(EntryRef::INVALID) }.unwrap().is_empty());
        assert_accounting(store.stats());
    }

    #[test]
    fn inline_directory_readers_follow_growth_without_descriptor_payload_access() {
        let d = domain();
        let store = ArrayStore::<u32, 12>::new(d.clone(), config(2, 4, 2, 4096, 4096)).unwrap();
        let first = store.writer().add(&[10, 20]).unwrap();
        let guard = d.take_guard().unwrap();
        store.view(&guard).unwrap();
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for n in 0..1000 { store.writer().add(&[n, n]).unwrap(); }
            });
            let view = unsafe { store.view_prevalidated(&guard) };
            for _ in 0..10000 {
                // Guard predates growth and the first reference is still owned.
                assert_eq!(unsafe { view.get(first) }.unwrap(), &[10, 20]);
            }
            writer.join().unwrap();
        });
        assert!(store.stats().growths > 1);
        assert_eq!(unsafe { store.view(&guard).unwrap().get(first) }.unwrap(), &[10, 20]);
    }

    #[test]
    fn growth_preserves_refs_and_old_slices_but_current_atomic_writes_follow_new_slab() {
        let d = domain();
        let store = ArrayStore::<AtomicU32, 4>::new(d.clone(), config(2, 4, 2, 16, 16)).unwrap();
        let first = store.writer().add(&[10, 20]).unwrap();
        {
            let guard = d.take_guard().unwrap();
            let view = store.view(&guard).unwrap();
            // SAFETY: first remains currently owned; guard precedes load.
            let old = unsafe { view.get(first) }.unwrap();
            let old_address = old.as_ptr();
            for n in 0..7 {
                store.writer().add(&[n, n]).unwrap();
            }
            assert!(store.stats().growths > 1);
            assert_eq!(store.writer().replace(first, 0, 99), Ok(10));
            assert_eq!(
                store.writer().replace(first, 2, 99),
                Err(StoreError::IndexOutOfBounds)
            );
            // SAFETY: same continuously held guard, unchanged current owner.
            let current = unsafe { view.get(first) }.unwrap();
            assert_ne!(old_address, current.as_ptr());
            assert_eq!(old[0].load(Ordering::Acquire), 10);
            assert_eq!(current[0].load(Ordering::Acquire), 99);
            commit(&store, &d);
            assert_eq!(store.writer().reclaim(), 0);
            assert!(store.stats().held_bytes > 0);
            assert_accounting(store.stats());
        }
        assert!(store.writer().reclaim() > 0);
        assert_eq!(store.stats().held_bytes, 0);
        assert_eq!(store.writer().read_current(first).unwrap(), [99, 20]);
        assert_accounting(store.stats());
    }

    #[test]
    fn large_fallback_growth_counts_copied_payloads_and_preserves_old_borrows() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 4, 2, 16, 16)).unwrap();
        let first = store.writer().add(&[1, 2, 3, 4, 5]).unwrap();
        {
            let guard = d.take_guard().unwrap();
            let view = store.view(&guard).unwrap();
            // SAFETY: first current owner is loaded under guard.
            let old = unsafe { view.get(first) }.unwrap();
            for n in 0..6 {
                store.writer().add(&[n, n, n, n, n]).unwrap();
            }
            // The conceptual current owner of first is invalidated here.
            store.writer().remove(first).unwrap();
            commit(&store, &d);
            assert_eq!(store.writer().reclaim(), 0);
            assert_eq!(old, [1, 2, 3, 4, 5]);
            assert_eq!(store.stats().retired_arrays, 1);
            assert_accounting(store.stats());
        }
        store.writer().reclaim();
        assert_eq!(store.stats().retired_arrays, 0);
        assert_eq!(store.stats().held_bytes, 0);
        let reused = store.writer().add(&[9, 8, 7, 6, 5]).unwrap();
        assert_eq!(reused, first);
        assert_eq!(
            store.writer().read_current(reused).unwrap(),
            [9, 8, 7, 6, 5]
        );
        assert_accounting(store.stats());
    }

    #[test]
    fn large_atomic_arrays_are_deep_copied_on_growth_like_vespa_array_copy() {
        let d = domain();
        let store = ArrayStore::<AtomicU32, 4>::new(d.clone(), config(1, 4, 2, 16, 16)).unwrap();
        let first = store.writer().add(&[10, 20, 30]).unwrap();
        {
            let guard = d.take_guard().unwrap();
            let view = store.view(&guard).unwrap();
            // SAFETY: first is a current owner acquired after guard pinning.
            let old = unsafe { view.get(first) }.unwrap();
            let old_address = old.as_ptr();
            // Descriptor size includes ownership metadata. Allocator rounding
            // can leave more than one free slot; force an observed growth,
            // rather than assuming the second insertion must grow the slab.
            let growths = store.stats().growths;
            for _ in 0..16 {
                store.writer().add(&[40, 50, 60]).unwrap();
                if store.stats().growths > growths { break; }
            }
            assert!(store.stats().growths > growths, "fixture must exercise descriptor growth");
            store.writer().replace(first, 0, 77).unwrap();
            // SAFETY: same live ref and continuously held guard.
            let current = unsafe { view.get(first) }.unwrap();
            assert_ne!(
                old_address,
                current.as_ptr(),
                "large arrays deep-copy, never alias current writes"
            );
            assert_eq!(old[0].load(Ordering::Acquire), 10);
            assert_eq!(current[0].load(Ordering::Acquire), 77);
            commit(&store, &d);
            assert_eq!(store.writer().reclaim(), 0);
            assert_accounting(store.stats());
        }
        store.writer().reclaim();
        assert_eq!(store.stats().held_bytes, 0);
        assert_eq!(store.writer().read_current(first).unwrap(), [77, 20, 30]);
    }

    #[test]
    fn per_class_free_buffer_lists_disable_and_restore_without_cross_class_reuse() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 8, 4, 4, 4)).unwrap();
        let mut one = Vec::new();
        for n in 0..6 {
            one.push(store.writer().add(&[n]).unwrap());
        }
        let two = store.writer().add(&[1, 2]).unwrap();
        assert_ne!(one[0].buffer_id(), one[3].buffer_id());
        for reference in [one[0], one[3], two] {
            store.writer().remove(reference).unwrap();
        }
        commit(&store, &d);
        store.writer().reclaim();
        {
            let writer = store.writer();
            assert_eq!(writer.state.free_buffers[1].len(), 2);
            assert_eq!(writer.state.free_buffers[2], [two.buffer_id()]);
        }
        store
            .writer()
            .start_compaction(&[one[0].buffer_id()])
            .unwrap();
        assert_eq!(store.writer().state.free_buffers[1], [one[3].buffer_id()]);
        assert_eq!(store.writer().add(&[99]).unwrap(), one[3]);
        assert!(store.writer().state.free_buffers[1].is_empty());
        // Aborting restores the now-reusable source entry to its class list.
        store.writer().abort_compaction(one[0].buffer_id()).unwrap();
        assert_eq!(store.writer().add(&[88]).unwrap(), one[0]);
        assert!(store.writer().state.free_buffers[1].is_empty());
        assert_eq!(store.writer().add(&[7, 8]).unwrap(), two);
        assert!(store.writer().state.free_buffers[2].is_empty());
        assert_accounting(store.stats());
    }

    #[test]
    fn growth_uses_class_live_counts_scaled_switch_and_portable_byte_rounding() {
        let d = domain();
        let store = ArrayStore::<u32, 5>::new(d.clone(), config(1, 8, 4, 32, 4)).unwrap();
        let mut refs = Vec::new();
        // Three active buffers scale the switch threshold from 4 to floor(6.76).
        // Thus the third grows from 4 to 8 slots instead of switching at 5.
        for n in 0..12 {
            refs.push(store.writer().add(&[n]).unwrap());
        }
        {
            let writer = store.writer();
            assert_eq!(writer.active_class_buffers(1), 3);
            assert_eq!(writer.scaled_switch_threshold(1), 6);
            assert_eq!(writer.allocation(refs[11].buffer_id()).capacity, 8);
            assert_eq!(refs[7].buffer_id(), refs[11].buffer_id());
        }
        // Fourth buffer's first resize uses all 19 live arrays, not just its
        // previous four-slot capacity: wanted=4+floor(19*.3)=9, rounded to16.
        for n in 12..20 {
            refs.push(store.writer().add(&[n]).unwrap());
        }
        let writer = store.writer();
        assert_eq!(writer.active_class_buffers(1), 4);
        assert_eq!(writer.allocation(refs[19].buffer_id()).capacity, 16);
        assert_eq!(writer.class_live_entries(1), 20);
        drop(writer);
        store.writer().remove(refs[0]).unwrap();
        assert_eq!(store.writer().class_live_entries(1), 20, "held is not dead");
        commit(&store, &d);
        store.writer().reclaim();
        assert_eq!(store.writer().class_live_entries(1), 19);
        assert_accounting(store.stats());
    }

    #[test]
    fn entry_reuse_requires_assignment_advance_and_strict_oldest_boundary() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(1, 4, 8, 16, 16)).unwrap();
        let first = store.writer().add(&[42]).unwrap();
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: loaded from current owner after pinning.
        let old = unsafe { view.get(first) }.unwrap();
        store.writer().remove(first).unwrap();
        assert_eq!(store.writer().remove(first), Err(StoreError::NotLive));
        assert_eq!(store.writer().reclaim(), 0); // Pending, not assigned.
        store.writer().assign_generation().unwrap();
        assert_eq!(store.writer().reclaim(), 0); // Equality cannot reclaim.
        d.inc_generation().unwrap();
        assert_eq!(store.writer().reclaim(), 0); // Old reader still alive.
        let second = store.writer().add(&[7]).unwrap();
        assert_ne!(second, first);
        assert_eq!(old, [42]);
        drop(guard);
        assert_eq!(store.writer().reclaim(), 1);
        assert_eq!(store.writer().add(&[8]).unwrap(), first);
        assert_accounting(store.stats());
    }

    #[test]
    fn transferred_guard_keeps_arrays_alive_for_prepare_completion() {
        let d = domain();
        let store =
            Arc::new(ArrayStore::<u32, 4>::new(d.clone(), config(1, 4, 8, 16, 16)).unwrap());
        let first = store.writer().add(&[12]).unwrap();
        let guard = d.take_guard().unwrap();
        // A proposal acquired first while guarded, then transfers BOTH the
        // payload owner and guard; guard alone would not own the ArrayStore.
        let other = Arc::clone(&store);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel();
        let task = std::thread::spawn(move || {
            let view = other.view(&guard).unwrap();
            ready_tx.send(()).unwrap();
            go_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            // SAFETY: first was acquired before transfer, same guard held.
            unsafe { view.get(first) }.unwrap().to_vec()
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        store.writer().remove(first).unwrap();
        commit(&store, &d);
        let reclaimed_while_held = store.writer().reclaim();
        let second = store.writer().add(&[13]);
        // Release and join before assertions so an assertion cannot strand
        // the worker; disconnected channels also terminate its bounded wait.
        go_tx.send(()).unwrap();
        let observed = task.join().unwrap();
        assert_eq!(observed, [12]);
        assert_eq!(reclaimed_while_held, 0);
        assert_ne!(second.unwrap(), first);
        assert!(store.writer().reclaim() > 0);
        assert_eq!(store.writer().add(&[14]).unwrap(), first);
    }

    #[test]
    fn compaction_changes_references_and_holds_buffer_ids_until_readers_drain() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(3, 4, 4, 16, 16)).unwrap();
        let a = store.writer().add(&[1, 2]).unwrap();
        let b = store.writer().add(&[3, 4]).unwrap();
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: current owners loaded after guard acquisition.
        let old_a = unsafe { view.get(a) }.unwrap();
        let old_b = unsafe { view.get(b) }.unwrap();
        let id = a.buffer_id();
        store.writer().start_compaction(&[id]).unwrap();
        let new_a = store.writer().move_on_compact(a).unwrap();
        assert_ne!(new_a, a);
        // Caller publishes new_a before removing old a.
        store.writer().remove(a).unwrap();
        assert_eq!(
            store.writer().finish_compaction(id),
            Err(StoreError::LiveOwnersRemain)
        );
        let new_b = store.writer().move_on_compact(b).unwrap();
        store.writer().remove(b).unwrap();
        store.writer().finish_compaction(id).unwrap();
        commit(&store, &d);
        assert_eq!(store.writer().reclaim(), 0);
        assert_eq!(store.stats().held_buffers, 1);
        assert_eq!(old_a, [1, 2]);
        assert_eq!(old_b, [3, 4]);
        assert_eq!(store.writer().read_current(new_a).unwrap(), [1, 2]);
        assert_eq!(store.writer().read_current(new_b).unwrap(), [3, 4]);
        assert_accounting(store.stats());
        drop(guard);
        store.writer().reclaim();
        assert_eq!(store.stats().held_buffers, 0);
        assert_eq!(store.stats().held_bytes, 0);
        let reused_id = store.writer().add(&[5, 6, 7]).unwrap();
        assert_eq!(reused_id.buffer_id(), id);
        // Never read old a/b with a new guard after reuse: that is forbidden,
        // even if their integer bits coincide with a different live entry.
    }

    #[test]
    fn old_level_array_protects_later_retired_links_in_the_same_generation_domain() {
        let d = domain();
        let links = ArrayStore::<u32, 4>::new(d.clone(), config(2, 4, 4, 16, 16)).unwrap();
        let levels = ArrayStore::<AtomicU32, 4>::new(d.clone(), config(2, 4, 4, 16, 16)).unwrap();
        let old_link = links.writer().add(&[11, 22]).unwrap();
        let old_levels = levels.writer().add(&[old_link.raw()]).unwrap();
        let current_levels = AtomicU32::new(old_levels.raw());
        {
            let guard = d.take_guard().unwrap();
            let lv = levels.view(&guard).unwrap();
            let lk = links.view(&guard).unwrap();
            let acquired = EntryRef::from_raw(current_levels.load(Ordering::Acquire));
            // SAFETY: acquired from current published owner after pinning.
            let old_level_slice = unsafe { lv.get(acquired) }.unwrap();
            levels
                .writer()
                .start_compaction(&[old_levels.buffer_id()])
                .unwrap();
            let new_levels = levels.writer().move_on_compact(old_levels).unwrap();
            current_levels.store(new_levels.raw(), Ordering::Release);
            levels.writer().remove(old_levels).unwrap();
            levels
                .writer()
                .finish_compaction(old_levels.buffer_id())
                .unwrap();
            commit(&levels, &d);

            // Retire the transitive link in a LATER generation. The old levels
            // slice is still allowed to discover it under the original guard.
            links
                .writer()
                .start_compaction(&[old_link.buffer_id()])
                .unwrap();
            let new_link = links.writer().move_on_compact(old_link).unwrap();
            levels
                .writer()
                .replace(new_levels, 0, new_link.raw())
                .unwrap();
            links.writer().remove(old_link).unwrap();
            links
                .writer()
                .finish_compaction(old_link.buffer_id())
                .unwrap();
            commit(&links, &d);
            assert_eq!(levels.writer().reclaim(), 0);
            assert_eq!(links.writer().reclaim(), 0);
            let acquired_old_link = EntryRef::from_raw(old_level_slice[0].load(Ordering::Acquire));
            assert_eq!(acquired_old_link, old_link);
            // SAFETY: reference reached transitively through guarded old levels.
            assert_eq!(unsafe { lk.get(acquired_old_link) }.unwrap(), [11, 22]);
            let fresh = d.take_guard().unwrap();
            let fresh_levels = levels.view(&fresh).unwrap();
            let fresh_links = links.view(&fresh).unwrap();
            let current = EntryRef::from_raw(current_levels.load(Ordering::Acquire));
            // SAFETY: current owner acquired after fresh guard; transitive ref
            // comes from that guarded current levels slice.
            let current_slice = unsafe { fresh_levels.get(current) }.unwrap();
            let current_link = EntryRef::from_raw(current_slice[0].load(Ordering::Acquire));
            assert_eq!(current_link, new_link);
            assert_eq!(unsafe { fresh_links.get(current_link) }.unwrap(), [11, 22]);
        }
        levels.writer().reclaim();
        links.writer().reclaim();
        assert_eq!(levels.stats().held_bytes, 0);
        assert_eq!(links.stats().held_bytes, 0);
    }

    #[test]
    fn partial_compaction_exhaustion_keeps_unmoved_owners_valid_and_can_abort() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(1, 2, 4, 4, 4)).unwrap();
        let mut owners = Vec::new();
        for n in 0..6 {
            owners.push(store.writer().add(&[n]).unwrap());
        }
        let source = owners[0].buffer_id();
        store.writer().start_compaction(&[source]).unwrap();
        let moved = store.writer().move_on_compact(owners[0]).unwrap();
        store.writer().remove(owners[0]).unwrap();
        owners[0] = moved; // publication represented by this owner slot
        assert_eq!(
            store.writer().move_on_compact(owners[1]),
            Err(StoreError::AddressSpaceExhausted)
        );
        assert_eq!(
            store.writer().finish_compaction(source),
            Err(StoreError::LiveOwnersRemain)
        );
        store.writer().abort_compaction(source).unwrap();
        for (n, owner) in owners.iter().enumerate() {
            assert_eq!(store.writer().read_current(*owner).unwrap(), [n as u32]);
        }
        commit(&store, &d);
        store.writer().reclaim();
        assert_eq!(store.stats().held_buffers, 0);
        assert_accounting(store.stats());
    }

    #[test]
    fn fixed_cardinality_churn_plateaus_after_drain_without_per_array_owners() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 8, 4, 16, 8)).unwrap();
        let mut owners = Vec::new();
        for n in 0..12 {
            owners.push(store.writer().add(&[n, n]).unwrap());
        }
        commit(&store, &d);
        store.writer().reclaim();
        let mut plateau = None;
        for round in 0..100 {
            for owner in owners.drain(..) {
                store.writer().remove(owner).unwrap();
            }
            commit(&store, &d);
            store.writer().reclaim();
            for n in 0..12 {
                owners.push(store.writer().add(&[round, n]).unwrap());
            }
            let stats = store.stats();
            assert_eq!(stats.live_arrays, 12);
            assert_eq!(stats.retired_arrays, 0);
            assert_eq!(stats.held_bytes, 0);
            assert_accounting(stats);
            if round == 5 {
                plateau = Some(stats.allocated_bytes);
            }
            if round > 5 {
                assert_eq!(Some(stats.allocated_bytes), plateau);
            }
        }
    }

    #[test]
    fn reclaimed_entries_are_consumed_once_and_keep_distinct_payloads_in_release() {
        // Run in the fast/release profile as well as debug. Previously pop()
        // lived inside debug_assert_eq!, so release repeatedly overwrote the
        // same reclaimed slot and silently returned duplicate live refs.
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 4, 8, 16, 16)).unwrap();
        let original: Vec<_> = (0..4)
            .map(|n| store.writer().add(&[n, n + 10]).unwrap())
            .collect();
        for reference in &original {
            store.writer().remove(*reference).unwrap();
        }
        commit(&store, &d);
        store.writer().reclaim();
        assert_eq!(store.stats().reusable_arrays, 4);

        let mut reused = Vec::new();
        for n in 0..4 {
            let reference = store.writer().add(&[100 + n, 200 + n]).unwrap();
            assert!(
                !reused.contains(&reference),
                "reclaimed slot returned twice"
            );
            reused.push(reference);
            let stats = store.stats();
            assert_eq!(stats.live_arrays, reused.len());
            assert_eq!(stats.reusable_arrays, 4 - reused.len());
            // Check all previous payloads after every new insertion, so this
            // detects overwrites as well as free-list accounting failures.
            for (i, existing) in reused.iter().enumerate() {
                assert_eq!(
                    store.writer().read_current(*existing).unwrap(),
                    [100 + i as u32, 200 + i as u32]
                );
            }
        }
        assert_eq!(
            reused.iter().copied().collect::<HashSet<_>>(),
            original.into_iter().collect::<HashSet<_>>()
        );
        assert!(store.writer().state.free_buffers[2].is_empty());
        let appended = store.writer().add(&[300, 400]).unwrap();
        assert!(!reused.contains(&appended));
        assert_accounting(store.stats());
    }

    #[test]
    fn staged_batch_rejects_foreign_duplicate_and_retired_old_refs() {
        let d = domain();
        let a = ArrayStore::<u32, 4>::new(d.clone(), config(2, 4, 2, 16, 16)).unwrap();
        let b = ArrayStore::<u32, 4>::new(d.clone(), config(2, 4, 2, 16, 16)).unwrap();
        let ar = a.writer().add(&[1, 2]).unwrap();
        let br = b.writer().add(&[3, 4]).unwrap();
        assert_eq!(ar, br, "foreign allocation has coincident compact bits");
        let guard = d.take_guard().unwrap();
        let av = a.view(&guard).unwrap();
        let bv = b.view(&guard).unwrap();
        // SAFETY: each ref is its own store's current live owner after pinning.
        let at = unsafe { av.retirement_token(ar) }.unwrap();
        let bt = unsafe { bv.retirement_token(br) }.unwrap();
        assert!(matches!(
            a.writer().stage_batch(1, &[bt]),
            Err(StoreError::ForeignStore)
        ));
        assert!(matches!(
            a.writer().stage_batch(1, &[at, at]),
            Err(StoreError::DuplicateRetirement)
        ));
        assert!(matches!(
            a.writer().stage_batch(usize::MAX, &[]),
            Err(StoreError::AllocationFailed)
        ));
        assert_eq!(a.writer().read_current(ar).unwrap(), [1, 2]);
        a.writer().remove(ar).unwrap(); // conceptual owner invalidated first
        commit(&a, &d);
        assert!(matches!(
            a.writer().stage_batch(1, &[at]),
            Err(StoreError::NotLive)
        ));
        assert_eq!(a.writer().reclaim(), 0, "token's reader prevents ABA reuse");
    }

    #[test]
    fn each_admitted_allocation_checkpoint_preserves_old_owners_and_cleanup_capacity() {
        fn exercise(fail_after: Option<usize>) -> usize {
            let d = domain();
            let store = ArrayStore::<u32, 4>::new(d.clone(), config(2, 8, 2, 16, 16)).unwrap();
            let old = store.writer().add(&[7, 8]).unwrap();
            let owner = AtomicU32::new(old.raw());
            let guard = d.take_guard().unwrap();
            let view = store.view(&guard).unwrap();
            let acquired = EntryRef::from_raw(owner.load(Ordering::Acquire));
            // SAFETY: loaded from current owner after pinning this store.
            let token = unsafe { view.retirement_token(acquired) }.unwrap();
            {
                let mut writer = store.writer();
                writer.state.allocation_failure = fail_after;
                writer.state.allocation_attempts = 0;
            }
            let result = (|| {
                let mut batch = store.writer().stage_batch(3, &[token])?;
                batch.stage(&[10, 11])?; // forces same-ref physical growth
                batch.stage(&[12])?; // new small-array class
                batch.stage(&[13, 14, 15, 16])?; // large fallback allocation
                batch.finish_staging()
            })();
            if fail_after.is_some() {
                assert!(matches!(&result, Err(StoreError::AllocationFailed)));
            }
            drop(result); // successful preflight is deliberately aborted too
            let attempts = store.writer().state.allocation_attempts;
            assert_eq!(owner.load(Ordering::Acquire), old.raw());
            assert_eq!(store.writer().read_current(old).unwrap(), [7, 8]);
            assert_eq!(store.stats().live_arrays, 1);
            {
                let mut writer = store.writer();
                // Abort/reclaim must not enter any allocation checkpoint even
                // when the next attempted allocation is forced to fail.
                writer.state.allocation_failure = Some(0);
                for state in writer.state.buffers.iter().flatten() {
                    assert!(state.free.capacity() >= state.entries.len());
                }
                for class in 0..writer.state.free_buffers.len() {
                    assert!(
                        writer.state.free_buffers[class].capacity()
                            >= writer.active_class_buffers(class)
                    );
                }
                writer.assign_generation().unwrap();
            }
            d.inc_generation().unwrap();
            assert_eq!(store.writer().reclaim(), 0);
            // SAFETY: same guard continues to protect the unchanged old owner.
            assert_eq!(unsafe { view.get(old) }.unwrap(), [7, 8]);
            drop(guard);
            store.writer().reclaim();
            assert_eq!(store.writer().state.allocation_attempts, attempts);
            assert_eq!(store.stats().retired_arrays, 0);
            assert_eq!(store.stats().held_bytes, 0);
            assert_eq!(store.stats().live_arrays, 1);
            assert_accounting(store.stats());
            attempts
        }
        let attempts = exercise(None);
        assert!(
            attempts >= 9,
            "exercise batch vectors, growth, new classes and large payload"
        );
        for point in 0..attempts {
            exercise(Some(point));
        }
    }

    #[test]
    fn aborted_stages_are_retired_and_opaque_indexes_cannot_cross_batches() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(1, 4, 8, 16, 16)).unwrap();
        let guard = d.take_guard().unwrap();
        let mut batch = store.writer().stage_batch(1, &[]).unwrap();
        let index = batch.stage(&[10]).unwrap();
        let ready = batch.finish_staging().unwrap();
        let aborted = ready.resolve(index).unwrap();
        drop(ready);
        assert_eq!(store.stats().retired_arrays, 1);
        let mut next = store.writer().stage_batch(1, &[]).unwrap();
        let next_index = next.stage(&[20]).unwrap();
        let next = next.finish_staging().unwrap();
        assert_eq!(next.resolve(index), Err(StoreError::InvalidBatch));
        assert_ne!(next.resolve(next_index).unwrap(), aborted);
        drop(next);
        commit(&store, &d);
        assert_eq!(store.writer().reclaim(), 0);
        drop(guard);
        assert_eq!(store.writer().reclaim(), 2);
        assert_eq!(store.stats().reusable_arrays, 2);
    }

    #[test]
    fn post_growth_atomic_preflight_and_publication_have_no_allocation_failure() {
        let d = domain();
        let store = ArrayStore::<AtomicU32, 4>::new(d.clone(), config(1, 4, 2, 16, 16)).unwrap();
        let owner = store.writer().add(&[123]).unwrap();
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: current owner acquired after guard, before staging.
        let token = unsafe { view.retirement_token(owner) }.unwrap();
        let old_slice = unsafe { view.get(owner) }.unwrap();
        let mut batch = store.writer().stage_batch(1, &[]).unwrap();
        let index = batch.stage(&[456]).unwrap(); // grow/copy owner's physical slab
        let mut ready = batch.finish_staging().unwrap();
        let staged = ready.resolve(index).unwrap();
        assert!(matches!(
            ready.preflight_replace(token, 0, 999, staged.raw()),
            Err(StoreError::OwnerChanged)
        ));
        let replacement = ready
            .preflight_replace(token, 0, 123, staged.raw())
            .unwrap();
        ready.batch.writer.state.allocation_failure = Some(0);
        let attempts = ready.batch.writer.state.allocation_attempts;
        let mut publishing = ready.begin_publish();
        // SAFETY: token was preflighted in this batch after all growth. The
        // existing owner now adopts the sole new array; nothing is retired.
        unsafe {
            publishing.replace_prevalidated(replacement);
            publishing.finish_after_publish();
        }
        assert_eq!(store.writer().state.allocation_attempts, attempts);
        assert_eq!(old_slice[0].load(Ordering::Acquire), 123);
        assert_eq!(store.writer().read_current(owner).unwrap(), [staged.raw()]);
        assert_eq!(store.writer().read_current(staged).unwrap(), [456]);
        assert_eq!(store.stats().live_arrays, 2);
    }

    #[test]
    fn committed_replacements_retire_only_old_refs_and_keep_guarded_old_payloads() {
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(1, 4, 8, 16, 16)).unwrap();
        let old = store.writer().add(&[5]).unwrap();
        let owner = AtomicU32::new(old.raw());
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: current owner acquired after pinning.
        let token =
            unsafe { view.retirement_token(EntryRef::from_raw(owner.load(Ordering::Acquire))) }
                .unwrap();
        let mut batch = store.writer().stage_batch(1, &[token]).unwrap();
        let index = batch.stage(&[6]).unwrap();
        let mut ready = batch.finish_staging().unwrap();
        let new = ready.resolve(index).unwrap();
        ready.batch.writer.state.allocation_failure = Some(0);
        let publishing = ready.begin_publish();
        owner.store(new.raw(), Ordering::Release);
        // SAFETY: old owner has been replaced by the sole staged array.
        unsafe { publishing.finish_after_publish() };
        assert_eq!(store.stats().live_arrays, 1);
        assert_eq!(store.stats().retired_arrays, 1);
        commit(&store, &d);
        assert_eq!(store.writer().reclaim(), 0);
        // SAFETY: the old acquired ref remains continuously guarded.
        assert_eq!(unsafe { view.get(old) }.unwrap(), [5]);
        assert_eq!(store.writer().read_current(new).unwrap(), [6]);
        drop(guard);
        store.writer().reclaim();
        assert_eq!(store.stats().retired_arrays, 0);
    }

    #[test]
    fn unwinding_after_publication_begins_never_retires_possibly_published_arrays() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let d = domain();
        let store = ArrayStore::<u32, 4>::new(d.clone(), config(1, 4, 8, 16, 16)).unwrap();
        let old = store.writer().add(&[1]).unwrap();
        let owner = AtomicU32::new(old.raw());
        let guard = d.take_guard().unwrap();
        let view = store.view(&guard).unwrap();
        // SAFETY: current owner loaded under guard.
        let token = unsafe { view.retirement_token(old) }.unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut batch = store.writer().stage_batch(1, &[token]).unwrap();
            let index = batch.stage(&[2]).unwrap();
            let ready = batch.finish_staging().unwrap();
            let new = ready.resolve(index).unwrap();
            let _publishing = ready.begin_publish();
            owner.store(new.raw(), Ordering::Release);
            panic!("injected unexpected publication unwind");
        }));
        assert!(result.is_err());
        let new = EntryRef::from_raw(owner.load(Ordering::Acquire));
        assert_ne!(new, old);
        assert_eq!(
            store.stats().live_arrays,
            2,
            "both sides conservatively retained"
        );
        assert_eq!(store.stats().retired_arrays, 0);
        commit(&store, &d);
        assert_eq!(store.writer().reclaim(), 0);
        assert_eq!(store.writer().read_current(old).unwrap(), [1]);
        assert_eq!(store.writer().read_current(new).unwrap(), [2]);
        // A real graph coordinator must poison/fail closed after this unwind;
        // this test only establishes safe allocator ownership, not atomicity.
    }

    #[test]
    fn hnsw_configuration_does_not_eagerly_allocate_switch_thresholds() {
        let config = ArrayStoreConfig::hnsw::<20>(193);
        assert!(config.specs.iter().all(|spec| spec.min_entries == 0));
        assert_eq!(config.specs[1].entries_for_new_buffer, 512 * 1024);
        let d = domain();
        let store = ArrayStore::<u32, 20>::new(d, config).unwrap();
        let before = store.stats();
        assert_eq!(before.active_buffers, 0);
        assert_eq!(before.allocated_bytes, before.metadata_bytes);
        store.writer().add(&[1, 2]).unwrap();
        let after = store.stats();
        assert_eq!(after.active_buffers, 1);
        assert!(after.allocated_bytes - before.allocated_bytes < 4096);
    }
}

impl<const B: u32> ArrayWriter<'_, AtomicU32, B> {
    /// Atomically publish one current level entry. Growth/compaction always
    /// resolve the current allocation, never mutate an old copied slab.
    pub(super) fn replace(
        &mut self,
        reference: EntryRef<B>,
        index: usize,
        value: u32,
    ) -> Result<u32, StoreError> {
        self.checked_state(reference)?;
        // SAFETY: writer liveness ownership, atomic mutation only.
        let slice = unsafe {
            self.allocation(reference.buffer_id())
                .get(reference.offset() as usize)?
        };
        let entry = slice.get(index).ok_or(StoreError::IndexOutOfBounds)?;
        Ok(entry.swap(value, Ordering::AcqRel))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ArrayStoreStats {
    pub active_buffers: usize,
    pub compacting_buffers: usize,
    pub held_buffers: usize,
    pub live_arrays: usize,
    pub retired_arrays: usize,
    pub reusable_arrays: usize,
    pub allocated_bytes: usize,
    pub used_bytes: usize,
    pub held_bytes: usize,
    pub dead_bytes: usize,
    pub metadata_bytes: usize,
    pub growths: u64,
    pub compactions: u64,
}

impl<E: ArrayElement, const B: u32> ArrayWriter<'_, E, B> {
    pub(super) fn stats(&self) -> ArrayStoreStats {
        let mut stats = ArrayStoreStats {
            metadata_bytes: size_of::<ArrayStore<E, B>>()
                + self.store.descriptors.len() * size_of::<BufferDirectoryEntry<E>>()
                + self.state.buffers.capacity() * size_of::<Option<BufferState>>()
                + self.state.primary.capacity() * size_of::<Option<u32>>()
                + self.state.free_buffers.capacity() * size_of::<Vec<u32>>()
                + self
                    .state
                    .free_buffers
                    .iter()
                    .map(|list| list.capacity() * size_of::<u32>())
                    .sum::<usize>()
                + self.state.pending.capacity() * size_of::<Retired<E, B>>()
                + self.state.held.capacity() * size_of::<(u64, Retired<E, B>)>()
                + self.store.config.specs.capacity() * size_of::<AllocationSpec>(),
            growths: self.state.growths,
            compactions: self.state.compactions,
            ..ArrayStoreStats::default()
        };
        // Large payloads are deep-copied on growth, matching pinned Array copy
        // construction. Every descriptor owns a distinct payload allocation.
        for (id, state) in self.state.buffers.iter().enumerate() {
            let Some(state) = state else { continue };
            match state.mode {
                BufferMode::Active => stats.active_buffers += 1,
                BufferMode::Compacting => stats.compacting_buffers += 1,
                BufferMode::Held => stats.held_buffers += 1,
            }
            stats.metadata_bytes += state.entries.capacity() * size_of::<EntryState>()
                + state.free.capacity() * size_of::<u32>();
            let allocation = self.allocation(id as u32);
            stats.allocated_bytes += allocation.physical_bytes();
            if state.mode == BufferMode::Held {
                stats.held_bytes += allocation.physical_bytes();
            }
            for (offset, entry) in state.entries.iter().enumerate() {
                let descriptor_bytes = if state.class == 0 {
                    size_of::<Option<AlignedElements<E>>>()
                } else {
                    state.class * size_of::<E>()
                };
                let mut bytes = descriptor_bytes;
                if let Payload::Large(v) = &allocation.payload {
                    // SAFETY: writer mutex owns descriptor mutations.
                    if let Some(payload) = unsafe { &*v[offset].get() } {
                        let payload_bytes = payload.allocated_bytes();
                        stats.allocated_bytes += payload_bytes;
                        bytes += payload_bytes;
                        if state.mode == BufferMode::Held {
                            stats.held_bytes += payload_bytes;
                        }
                    }
                }
                match entry {
                    EntryState::Live => {
                        stats.live_arrays += 1;
                        stats.used_bytes += bytes;
                    }
                    EntryState::Retired => {
                        stats.retired_arrays += 1;
                        if state.mode != BufferMode::Held {
                            stats.held_bytes += bytes;
                        }
                    }
                    EntryState::Free => {
                        stats.reusable_arrays += 1;
                        if state.mode != BufferMode::Held {
                            stats.dead_bytes += bytes;
                        }
                    }
                    EntryState::Reserved => {
                        if state.mode != BufferMode::Held {
                            stats.dead_bytes += bytes;
                        }
                    }
                }
            }
        }
        let retired = self
            .state
            .pending
            .iter()
            .chain(self.state.held.iter().map(|(_, item)| item));
        for item in retired {
            if let Retired::Allocation(allocation) = item {
                stats.allocated_bytes += allocation.physical_bytes();
                stats.held_bytes += allocation.physical_bytes();
                if let Payload::Large(v) = &allocation.payload {
                    for slot in v.iter() {
                        // SAFETY: old allocations are immutable after growth.
                        if let Some(payload) = unsafe { &*slot.get() } {
                            let bytes = payload.allocated_bytes();
                            stats.allocated_bytes += bytes;
                            stats.held_bytes += bytes;
                        }
                    }
                }
            }
        }
        stats.allocated_bytes += stats.metadata_bytes;
        stats
    }
}
