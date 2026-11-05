//! Vespa's per-search-layer visited policy, isolated for the mutable HNSW POC.
//!
//! Reference revision: `0521b00640ef934b359d2fc356f97d84499dd17d`.
//! - `searchlib/src/vespa/searchlib/tensor/hnsw_index.cpp:319-338,621-656`
//!   estimates visits and selects bitmap iff estimate >= nodeid_limit / 128.
//! - `bitvector_visited_tracker.{h,cpp}` and `hash_set_visited_tracker.{h,cpp}`
//!   create fresh trackers inside each search-layer helper. There is no retained
//!   thread-local stamp array or generation-reset policy in those trackers.
//!
//! Representation differences: the bitmap is a Rust boxed u64 slice, without
//! Vespa AllocatedBitVector's allocator/alignment details. The hash variant uses
//! the repository's existing ahash::AHashSet, not vespalib::hash_set; its bucket
//! layout, hash function and allocation size differ. Selection, estimate, and
//! mark semantics for valid node IDs follow the reference. Out-of-range IDs are
//! rejected defensively here; Vespa's traversal checks its limit before marking.
//! Allocation counters describe owned storage, not allocator arenas or RSS.

use ahash::AHashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Policy {
    Bitmap,
    Hash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FilterEstimate {
    /// GlobalFilter::size(): document-ID universe, not graph-node count.
    pub size: u32,
    /// GlobalFilter::count(): accepted documents, not observed ANN hits.
    pub count: u32,
}

/// Port of HnswIndex::estimate_visited_nodes. The caller supplies the actual
/// max_links_for_level (2*m for level zero, m for upper levels).
/// Filter statistics must describe the same GlobalFilter used for traversal.
pub(super) fn estimate_visited_nodes(
    max_links_for_level: u32,
    nodeid_limit: u32,
    neighbors_to_find: u32,
    filter: Option<FilterEstimate>,
) -> u32 {
    let base = u64::from(max_links_for_level) * u64::from(neighbors_to_find) + 100;
    if base >= u64::from(nodeid_limit) {
        return nodeid_limit;
    }
    let Some(filter) = filter else {
        return base as u32;
    };
    assert!(
        filter.count <= filter.size,
        "filter count exceeds its universe"
    );
    if filter.count == 0 {
        return nodeid_limit;
    }
    let scaled = (f64::from(filter.size) / f64::from(filter.count)) * base as f64;
    if scaled >= f64::from(nodeid_limit) {
        nodeid_limit
    } else {
        scaled as u32 // C++ return conversion truncates toward zero.
    }
}

pub(super) fn choose_policy(nodeid_limit: u32, estimated_visited_nodes: u32) -> Policy {
    if estimated_visited_nodes >= nodeid_limit / 128 {
        Policy::Bitmap
    } else {
        Policy::Hash
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StorageUsage {
    /// Exact owned bitmap payload; excludes the struct/allocator overhead.
    pub bitmap_bytes: usize,
    /// Hash capacity in entries. AHashSet does not expose allocated byte size;
    /// multiplying this by four would omit control bytes and empty buckets.
    pub hash_capacity: usize,
    pub hash_entries: usize,
}

enum Storage {
    Bitmap(Box<[u64]>),
    Hash(AHashSet<u32>),
}

pub(super) struct VisitedTracker {
    limit: u32,
    storage: Storage,
}

impl Default for VisitedTracker {
    fn default() -> Self {
        Self {
            limit: 0,
            storage: Storage::Bitmap(Box::default()),
        }
    }
}

impl VisitedTracker {
    pub(super) fn new(nodeid_limit: u32, estimated_visited_nodes: u32) -> Self {
        let storage = match choose_policy(nodeid_limit, estimated_visited_nodes) {
            Policy::Bitmap => {
                Storage::Bitmap(vec![0; (nodeid_limit as usize).div_ceil(64)].into_boxed_slice())
            }
            Policy::Hash => {
                Storage::Hash(AHashSet::with_capacity(estimated_visited_nodes as usize))
            }
        };
        Self {
            limit: nodeid_limit,
            storage,
        }
    }

    /// Fresh layer state, matching a newly constructed Vespa tracker. Release
    /// old capacity before allocating the replacement, including large->small
    /// and bitmap<->hash transitions; never retain the maximum request size.
    pub(super) fn reset(&mut self, nodeid_limit: u32, estimated_visited_nodes: u32) {
        *self = Self::default();
        *self = Self::new(nodeid_limit, estimated_visited_nodes);
    }

    pub(super) fn policy(&self) -> Policy {
        match self.storage {
            Storage::Bitmap(_) => Policy::Bitmap,
            Storage::Hash(_) => Policy::Hash,
        }
    }

    /// 0 = not visited, 1 = visited or outside the captured node universe.
    /// This compatibility method has no rejected-node stamp state.
    #[inline]
    pub(super) fn state(&self, nodeid: u32) -> u8 {
        if nodeid >= self.limit {
            return 1;
        }
        u8::from(match &self.storage {
            Storage::Bitmap(words) => words[nodeid as usize / 64] & (1u64 << (nodeid % 64)) != 0,
            Storage::Hash(nodes) => nodes.contains(&nodeid),
        })
    }

    #[inline]
    pub(super) fn mark(&mut self, nodeid: u32) {
        self.try_mark(nodeid);
    }

    #[inline]
    pub(super) fn insert(&mut self, nodeid: u32) -> bool {
        self.try_mark(nodeid)
    }

    #[inline]
    pub(super) fn try_mark(&mut self, nodeid: u32) -> bool {
        if nodeid >= self.limit {
            return false;
        }
        match &mut self.storage {
            Storage::Bitmap(words) => {
                let word = &mut words[nodeid as usize / 64];
                let mask = 1u64 << (nodeid % 64);
                let unseen = *word & mask == 0;
                *word |= mask;
                unseen
            }
            Storage::Hash(nodes) => nodes.insert(nodeid),
        }
    }

    pub(super) fn storage_usage(&self) -> StorageUsage {
        match &self.storage {
            Storage::Bitmap(words) => StorageUsage {
                bitmap_bytes: std::mem::size_of_val(words.as_ref()),
                ..Default::default()
            },
            Storage::Hash(nodes) => StorageUsage {
                hash_capacity: nodes.capacity(),
                hash_entries: nodes.len(),
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_uses_vespa_integer_boundary_including_small_universes() {
        assert_eq!(choose_policy(1_000_000, 7_811), Policy::Hash);
        assert_eq!(choose_policy(1_000_000, 7_812), Policy::Bitmap);
        assert_eq!(choose_policy(128, 0), Policy::Hash);
        assert_eq!(choose_policy(128, 1), Policy::Bitmap);
        assert_eq!(choose_policy(127, 0), Policy::Bitmap);
        assert_eq!(choose_policy(0, 0), Policy::Bitmap);
        assert_eq!(choose_policy(u32::MAX, u32::MAX / 128 - 1), Policy::Hash);
        assert_eq!(choose_policy(u32::MAX, u32::MAX / 128), Policy::Bitmap);
    }

    #[test]
    fn estimates_match_level_capacity_filter_scaling_clamp_and_truncation() {
        assert_eq!(estimate_visited_nodes(32, 1_000_000, 100, None), 3_300);
        assert_eq!(estimate_visited_nodes(16, 1_000_000, 100, None), 1_700);
        assert_eq!(estimate_visited_nodes(32, 3_000, 100, None), 3_000);
        assert_eq!(estimate_visited_nodes(32, 0, 100, None), 0);
        let ten_percent = Some(FilterEstimate {
            size: 1_000_000,
            count: 100_000,
        });
        assert_eq!(
            estimate_visited_nodes(32, 1_000_000, 100, ten_percent),
            33_000
        );
        assert_eq!(estimate_visited_nodes(32, 5_000, 100, ten_percent), 5_000);
        assert_eq!(
            estimate_visited_nodes(1, 100_000, 1, Some(FilterEstimate { size: 10, count: 3 })),
            336
        );
        assert_eq!(
            estimate_visited_nodes(1, 10_000, 1, Some(FilterEstimate { size: 0, count: 0 })),
            10_000
        );
        assert_eq!(
            estimate_visited_nodes(u32::MAX, u32::MAX, u32::MAX, None),
            u32::MAX
        );
    }

    #[test]
    fn bitmap_and_hash_agree_on_mark_duplicates_and_bounds() {
        for estimate in [0, 8] {
            let mut tracker = VisitedTracker::new(1_024, estimate);
            assert_eq!(
                tracker.policy(),
                if estimate == 0 {
                    Policy::Hash
                } else {
                    Policy::Bitmap
                }
            );
            for id in [0, 1, 63, 64, 65, 1_023] {
                assert_eq!(tracker.state(id), 0);
                assert!(tracker.try_mark(id));
                assert_eq!(tracker.state(id), 1);
                assert!(!tracker.insert(id));
                tracker.mark(id);
                assert_eq!(tracker.state(id), 1);
            }
            for id in [1_024, 1_025, u32::MAX] {
                assert_eq!(tracker.state(id), 1);
                assert!(!tracker.try_mark(id));
                tracker.mark(id);
            }
        }
        let mut empty = VisitedTracker::new(0, 0);
        assert!(!empty.try_mark(0));
        assert_eq!(empty.state(0), 1);
        assert_eq!(empty.storage_usage(), StorageUsage::default());
    }

    #[test]
    fn reset_has_no_stale_marks_across_same_size_and_representation_changes() {
        let mut tracker = VisitedTracker::new(1_024, 8);
        for estimate in [8, 0, 0, 8] {
            tracker.mark(17);
            tracker.mark(63);
            tracker.reset(1_024, estimate);
            assert_eq!(tracker.state(17), 0);
            assert_eq!(tracker.state(63), 0);
            assert!(tracker.try_mark(17));
            assert!(!tracker.try_mark(17));
        }
        tracker.reset(3, 0);
        assert_eq!(tracker.state(2), 0);
        assert_eq!(tracker.state(17), 1);
        tracker.reset(1_024, 8);
        assert_eq!(tracker.state(17), 0);
    }

    #[test]
    fn large_then_small_workloads_release_owned_peak_capacity() {
        let mut tracker = VisitedTracker::new(1_000_000, 1_000_000);
        assert_eq!(tracker.storage_usage().bitmap_bytes, 125_000);
        tracker.mark(999_999);
        tracker.reset(65, 0);
        assert_eq!(tracker.storage_usage().bitmap_bytes, 16);
        assert_eq!(tracker.state(64), 0);
        tracker.reset(1_000_000, 4_000);
        assert_eq!(tracker.policy(), Policy::Hash);
        for id in 0..8_000 {
            assert!(tracker.insert(id));
        }
        let peak = tracker.storage_usage();
        assert_eq!(peak.hash_entries, 8_000);
        assert!(peak.hash_capacity >= 8_000);
        tracker.reset(1_000_000, 1);
        let small = tracker.storage_usage();
        assert_eq!(small.bitmap_bytes, 0);
        assert_eq!(small.hash_entries, 0);
        assert!(small.hash_capacity < peak.hash_capacity);
        assert_eq!(tracker.state(7_999), 0);
        tracker.reset(1_024, 8);
        assert_eq!(tracker.storage_usage().hash_capacity, 0);
        assert_eq!(tracker.storage_usage().bitmap_bytes, 128);
    }
}

// The traversal selects one concrete tracker once per layer, as Vespa's
// templated search helpers do. Keep the enum adapter for legacy callers/tests.
pub(super) trait LayerVisited: Default {
    fn reset(&mut self, limit: u32, estimate: u32);
    fn state(&self, id: u32) -> u8;
    fn insert(&mut self, id: u32) -> bool;
    /// The search is done: drop what a fresh tracker would not hold.
    fn release(&mut self) { *self = Self::default(); }
}

/// A visited bitmap over the node universe, retained per thread across
/// layers and queries ([`with_layer_visited`]): a reset clears only the
/// words the previous layer touched (or the whole map when that is
/// cheaper), so a layer neither allocates, zeroes `limit / 8` bytes, nor
/// hashes (Vespa's `BitVectorVisitedTracker` / `HashSetVisitedTracker`
/// allocate per layer; their marks and bounds are the same).
#[derive(Default)]
pub(super) struct SparseVisited { limit: u32, words: Vec<u64>, touched: Vec<u32> }
impl LayerVisited for SparseVisited {
    fn reset(&mut self, limit: u32, _: u32) {
        let needed = (limit as usize).div_ceil(64);
        if self.touched.len() >= self.words.len() / 8 {
            self.words.fill(0);
        } else {
            for &word in &self.touched { self.words[word as usize] = 0; }
        }
        self.touched.clear();
        // A much smaller graph than the last one: do not keep the large map.
        if self.words.len() > (1 << 14) && self.words.len() > needed.saturating_mul(4) {
            self.words = Vec::new();
            self.touched = Vec::new();
        }
        if self.words.len() < needed { self.words.resize(needed, 0); }
        self.limit = limit;
    }
    #[inline] fn state(&self, id: u32) -> u8 {
        u8::from(id >= self.limit || self.words[id as usize / 64] & (1u64 << (id % 64)) != 0)
    }
    #[inline] fn insert(&mut self, id: u32) -> bool {
        if id >= self.limit { return false; }
        let index = id as usize / 64;
        let word = &mut self.words[index];
        let mask = 1u64 << (id % 64);
        if *word & mask != 0 { return false; }
        if *word == 0 { self.touched.push(index as u32); }
        *word |= mask;
        true
    }
    /// Kept for the next layer (its reset clears the touched words).
    fn release(&mut self) {}
}

thread_local! {
    static LAYER_VISITED: std::cell::RefCell<SparseVisited> = std::cell::RefCell::new(SparseVisited::default());
}

/// Run `f` with this thread's retained [`SparseVisited`], reset for
/// `limit` nodes. A nested call (none today) gets a fresh tracker.
pub(super) fn with_layer_visited<R>(limit: u32, f: impl FnOnce(&mut SparseVisited) -> R) -> R {
    LAYER_VISITED.with(|cell| {
        let mut visited = cell.take();
        visited.reset(limit, 0);
        let out = f(&mut visited);
        cell.replace(visited);
        out
    })
}
impl LayerVisited for VisitedTracker {
    fn reset(&mut self, limit: u32, estimate: u32) { VisitedTracker::reset(self, limit, estimate); }
    fn state(&self, id: u32) -> u8 { VisitedTracker::state(self, id) }
    fn insert(&mut self, id: u32) -> bool { VisitedTracker::insert(self, id) }
}
// Vespa's per-layer trackers, kept as the reference the retained
// `SparseVisited` is tested against.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Default)]
pub(super) struct BitmapVisited { limit: u32, words: Box<[u64]> }
impl LayerVisited for BitmapVisited {
    fn reset(&mut self, limit: u32, _: u32) {
        *self = Self::default();
        self.limit = limit;
        self.words = vec![0; (limit as usize).div_ceil(64)].into_boxed_slice();
    }
    #[inline] fn state(&self, id: u32) -> u8 {
        u8::from(id >= self.limit || self.words[id as usize / 64] & (1u64 << (id % 64)) != 0)
    }
    #[inline] fn insert(&mut self, id: u32) -> bool {
        if id >= self.limit { return false; }
        let word = &mut self.words[id as usize / 64];
        let mask = 1u64 << (id % 64);
        let unseen = *word & mask == 0;
        *word |= mask;
        unseen
    }
}
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Default)]
pub(super) struct HashVisited { limit: u32, nodes: AHashSet<u32> }
impl LayerVisited for HashVisited {
    fn reset(&mut self, limit: u32, estimate: u32) {
        *self = Self::default();
        self.limit = limit;
        self.nodes = AHashSet::with_capacity(estimate as usize);
    }
    #[inline] fn state(&self, id: u32) -> u8 { u8::from(id >= self.limit || self.nodes.contains(&id)) }
    #[inline] fn insert(&mut self, id: u32) -> bool { id < self.limit && self.nodes.insert(id) }
}

#[cfg(test)]
mod specialized_tests {
    use super::*;
    #[test]
    fn concrete_trackers_match_reference_across_bounds_and_resets() {
        for limit in [0, 1, 63, 64, 65, 10000] {
            let mut bitmap = BitmapVisited::default();
            let mut hash = HashVisited::default();
            let mut sparse = SparseVisited::default();
            for _ in 0..2 {
                bitmap.reset(limit, limit);
                hash.reset(limit, limit);
                sparse.reset(limit, limit);
                for id in (0..limit + 2).rev().chain(0..limit + 2) {
                    assert_eq!(bitmap.state(id), hash.state(id));
                    assert_eq!(sparse.state(id), hash.state(id));
                    let inserted = bitmap.insert(id);
                    assert_eq!(inserted, hash.insert(id));
                    assert_eq!(inserted, sparse.insert(id));
                    assert_eq!(bitmap.state(id), hash.state(id));
                    assert_eq!(sparse.state(id), hash.state(id));
                }
            }
        }
    }

    /// A retained tracker forgets every mark on reset, whether it clears the
    /// touched words or the whole map, across growing and shrinking graphs.
    #[test]
    fn retained_sparse_tracker_resets_cleanly_across_limits() {
        let mut sparse = SparseVisited::default();
        for (limit, marks) in [(100_000u32, 7u32), (100_000, 90_000), (1_000, 300), (5_000_000, 11), (64, 64), (100_000, 3)] {
            sparse.reset(limit, 0);
            for id in 0..limit { assert_eq!(sparse.state(id), 0, "limit {limit}: id {id} still marked"); }
            assert_eq!(sparse.state(limit), 1);
            for i in 0..marks { let id = (i.wrapping_mul(2_654_435_761)) % limit.max(1); sparse.insert(id); }
        }
        with_layer_visited(10, |v| { assert!(v.insert(3)); assert!(!v.insert(3)); });
        with_layer_visited(10, |v| assert_eq!(v.state(3), 0, "the retained tracker is reset per layer"));
    }
}
