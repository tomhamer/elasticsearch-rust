//! Published reader views (Lucene's `DirectoryReader` as handed out by
//! `SearcherManager`).
//!
//! Every change to what searches see (a publish / refresh, a seal or merge
//! installed, a delete made visible) builds a new immutable [`ReaderView`]
//! under the index's own lock and swaps it into the index's [`ViewSlot`].
//! A search clones the current view's `Arc` (the slot's mutex is held only
//! for that clone) and then runs holding no index lock at all: writers,
//! refreshes, seals and merges proceed meanwhile and the view keeps reading
//! the leaves it was built over.
//!
//! What a view pins, all exact as of its publication:
//! - sealed segments as [`Segment::reader_snapshot`] copies: the shared file
//!   data and caches of the live segment with the deletions searches saw at
//!   publication, frozen (a later publish never changes them);
//! - published memtables (`Arc<SegmentWriter>`, whose documents never change
//!   once published) each with a frozen copy of the deletions searches saw
//!   at publication ([`LeafReader::Memtable`]'s second field). A later
//!   publish changes the live memtable's deletions, never the view's copy,
//!   so a view (ordinary or retained) never sees a delete made visible after
//!   it was taken. Memtable bitsets are small (`num_docs / 8` bytes);
//! - files: a merged-away segment's files are deleted only once no
//!   `Segment` (live, view copy or PIT capture) shares its data any more
//!   (see `Index::commit`).
//!
//! Mutable HNSW: a view does not pin a graph generation; ordinary searches
//! pin one only for the duration of each graph traversal, as before. A
//! retained view captures the vector store ([`Segment::capture_read_view`]),
//! which pins a generation until the view is dropped. Long-lived retained
//! views therefore hold retired graph memory back until they close; by
//! default that memory is not capped (`MutableHnswLimits::default`), so
//! readers never make writers wait or fail.
//!
//! Mutable-graph deletes are removed physically only once no published
//! view older than the delete is alive ([`ViewSlot::min_live_generation`],
//! stamped through `VectorStore::delete_epoch`), so an in-flight view never
//! loses a node it counts live, and a retained view captures (at
//! [`ViewSlot::retain`]) a graph still holding every node its publication
//! counts live; deletes made visible later are masked by the view's frozen
//! deletions. If an embedder configures an explicit retirement cap and long
//! retained views exhaust it, writes fail fast (never wait under a shard
//! lock) and background repair is deferred instead of failing.
//!
//! Views exclude the memtables being written (NRT mode makes them visible
//! only at publish). Outside NRT mode [`crate::Index::leaves`] also lists the
//! open memtables and deletes are visible at once; views do not follow those
//! (they are republished only by publish / seal / merge), so only NRT
//! indices should search views.

use crate::error::{Result, RusticError};
use crate::postings::BitSet;
use crate::query::reader::LeafReader;
use crate::segment::{Segment, SegmentWriter};
use crate::types::{DocAddress, Document};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static MOVED_OWNER_REBUILDS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of mutable kNN queries that met a point whose owner a
/// merge installed after their view was taken, and so rebuilt the view's
/// point map (O(docs of the view's segments), once per such query). For
/// benchmarks: this should stay rare (only queries racing a merge install).
pub fn moved_owner_rebuilds() -> u64 {
    MOVED_OWNER_REBUILDS.load(Ordering::Relaxed)
}

#[cfg_attr(not(feature = "mutable-hnsw-poc"), allow(dead_code))]
pub(crate) fn note_moved_owner_rebuild(points: usize) {
    let n = MOVED_OWNER_REBUILDS.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::debug!(points, total = n, "mutable kNN rebuilt a reader view's point map after a merge moved owners");
}

/// An immutable set of searchable leaves: sealed segments (in leaf order)
/// followed by published memtables, as of one publication.
pub struct ReaderView {
    segments: Vec<Arc<Segment>>,
    /// Published memtables with their frozen visible deletions.
    frozen: Vec<(Arc<SegmentWriter>, Arc<BitSet>)>,
    generation: u64,
    /// Writer-side counts as of the publication, for stats only: sealed
    /// segments not yet searchable, and closed memtables not yet published.
    /// Sealed segments not yet searchable: `(id, docs, bytes)`.
    non_searchable_segment_docs: Vec<(u32, u64, u64)>,
    /// Published views count as live readers of the mutable graph while
    /// they exist (see [`ViewSlot::min_live_generation`]); released on drop.
    registry: Option<Weak<ViewSlot>>,
    committed: Arc<parking_lot::RwLock<CommittedSegments>>,
    /// Node-wide accounting of the frozen memtable deletions this view
    /// holds (a retained copy counts the ones it shares too).
    _charge: crate::accounting::Charge,
    _memory: Option<Arc<crate::admission::Reservation>>,
}

/// Accounting of a view's frozen memtable deletions.
fn frozen_charge(frozen: &[(Arc<SegmentWriter>, Arc<BitSet>)]) -> crate::accounting::Charge {
    let bytes: usize = frozen.iter().map(|(_, dels)| dels.words().len() * 8 + 64).sum();
    crate::accounting::Charge::new(crate::accounting::Category::ReaderViews, bytes as u64)
}

impl Drop for ReaderView {
    fn drop(&mut self) {
        if let Some(slot) = self.registry.take().and_then(|slot| slot.upgrade()) {
            slot.release(self.generation);
        }
    }
}

impl ReaderView {
    /// A view with no leaves.
    pub fn empty() -> Self {
        Self { segments: Vec::new(), frozen: Vec::new(), generation: 0, non_searchable_segment_docs: Vec::new(), registry: None, committed: Default::default(), _charge: Default::default(), _memory: None }
    }

    /// A view over exactly `segments` (no memtables), e.g. a set of segments
    /// captured elsewhere.
    pub fn of_segments(segments: Vec<Arc<Segment>>) -> Self {
        Self { segments, frozen: Vec::new(), generation: 0, non_searchable_segment_docs: Vec::new(), registry: None, committed: Default::default(), _charge: Default::default(), _memory: None }
    }

    /// The searchable leaves: segments, then published memtables. Leaf `i`
    /// is `DocAddress::segment_id == i`, as for [`crate::Index::leaves`].
    pub fn leaves(&self) -> Vec<LeafReader<'_>> {
        let mut v: Vec<LeafReader<'_>> = Vec::with_capacity(self.segments.len() + self.frozen.len());
        v.extend(self.segments.iter().map(|s| LeafReader::Segment(s)));
        v.extend(self.frozen.iter().map(|(w, dels)| LeafReader::Memtable(w, Some(dels))));
        v
    }

    /// The sealed segments, in leaf order.
    pub fn segments(&self) -> &[Arc<Segment>] {
        &self.segments
    }

    /// Number of published memtables (leaves after the segments).
    pub fn num_memtables(&self) -> usize {
        self.frozen.len()
    }

    /// The segment count stats report for this copy (no lock): searchable
    /// segments and published memtables, plus segments sealed but not yet
    /// searchable and closed memtables not yet published (each closed
    /// memtable counts as the segment it seals into), as of the view's
    /// publication, plus retired segments still referenced by the last commit.
    pub fn segments_for_stats(&self) -> usize {
        let committed = self.committed.read();
        self.num_leaves() + self.non_searchable_segment_docs.len() + committed.iter().filter(|(id, _)|
            !self.segments.iter().any(|s| s.id == *id)
                && !self.frozen.iter().any(|(w, _)| w.id == *id)
                && !self.non_searchable_segment_docs.iter().any(|(pending, _, _)| pending == id)).count()
    }

    /// Document counts for the same storage units counted by `_stats`.
    /// Searchable leaves come first, followed by sealed/unpublished units.
    /// Captured at publication; enumeration never seals or takes an index lock.
    pub fn segment_doc_counts_for_stats(&self) -> Vec<u64> {
        self.segment_stats().iter().map(|s| s.docs).collect()
    }

    /// Searchable and pending units, plus last-commit segments retired from
    /// search by a merge. A sealed segment is not committed until meta.json
    /// has been replaced successfully. Memtables are never committed.
    pub fn segment_stats(&self) -> Vec<SegmentStats> {
        let mut out=Vec::new();
        self.try_visit_segment_stats::<std::convert::Infallible>(&mut |s|{out.push(s);Ok(())}).unwrap();
        out
    }

    /// Stream scalar observability records into an admitted caller-owned sink.
    pub fn try_visit_segment_stats<E>(&self, emit:&mut dyn FnMut(SegmentStats)->std::result::Result<(),E>)->std::result::Result<(),E> {
        let committed = self.committed.read();
        for s in &self.segments {
            emit(SegmentStats { id: s.id, docs: u64::from(s.num_docs), search: true,
                committed: committed.iter().any(|(id, _)| *id == s.id),
                bytes: s.segment_bytes(), deleted: u64::from(s.num_docs - s.num_live_docs().min(s.num_docs)) })?;
        }
        for (w, d) in &self.frozen {
            let n = w.num_docs();
            let deleted = u64::from(d.count().min(n));
            emit(SegmentStats { id: w.id, docs: u64::from(n), search: true, committed: false, bytes: 0, deleted })?;
        }
        for &(id, docs, bytes) in &self.non_searchable_segment_docs {
            // A sealed segment's files exist before a refresh publishes it
            // (ES lists such a segment with its size too).
            emit(SegmentStats { id, docs, search: false, committed: committed.iter().any(|(c, _)| *c == id), bytes, deleted: 0 })?;
        }
        for &(id, docs) in committed.iter() {
            if !self.segments.iter().any(|s|s.id==id) && !self.frozen.iter().any(|(w,_)|w.id==id) && !self.non_searchable_segment_docs.iter().any(|(s,_,_)|*s==id) {
                emit(SegmentStats { id, docs, search: false, committed: true, bytes: 0, deleted: 0 })?;
            }
        }
        Ok(())
    }

    /// Number of leaves.
    pub fn num_leaves(&self) -> usize {
        self.segments.len() + self.frozen.len()
    }

    /// Publication counter of the slot that built this view (`0` for views
    /// not built by a slot). Increases with every publication.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Documents (deleted ones included) over every leaf.
    pub fn num_docs(&self) -> u64 {
        self.leaves().iter().map(|l| l.num_docs() as u64).sum()
    }

    /// Live documents as searches of this view see them.
    pub fn num_live_docs(&self) -> u64 {
        self.leaves().iter().map(|l| l.num_live_docs() as u64).sum()
    }

    /// Whether [`ViewSlot::retain`] cannot capture this view's mutable
    /// vectors: its published memtables hold mutable-graph points but no
    /// sealed segment carries the vector store, so a retained copy would
    /// search the live graph. The owner then seals the published memtables
    /// before retaining.
    pub fn needs_sealed_capture(&self) -> bool {
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            !self.segments.iter().any(|s| s.vector_store().is_some())
                && self.frozen.iter().any(|(w, _)| w.mutable_vector_store().is_some())
        }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        {
            false
        }
    }

    /// The stored document at `addr` (leaf index in [`ReaderView::leaves`]).
    pub fn get_doc(&self, addr: &DocAddress) -> Result<Document> {
        let leaves = self.leaves();
        let leaf = leaves
            .get(addr.segment_id as usize)
            .ok_or_else(|| RusticError::IndexNotFound(format!("segment {}", addr.segment_id)))?;
        leaf.get_doc(addr.doc_id)?.ok_or_else(|| RusticError::IndexNotFound(format!("{}", addr)))
    }
}

/// One storage unit reported by segment observability APIs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentStats {
    pub id: u32,
    /// Documents, deleted ones included (Lucene `maxDoc`).
    pub docs: u64,
    pub search: bool,
    pub committed: bool,
    /// Segment file bytes (`0` for memtables and units this view does not
    /// hold). Not on the wire: the positional layout older peers read stays.
    #[serde(skip)]
    pub bytes: u64,
    /// Deleted documents (searchable units only; not on the wire).
    #[serde(skip)]
    pub deleted: u64,
}

#[derive(Default)]
struct CommittedSegments {
    segments: Vec<(u32, u64)>,
    _memory: Option<crate::admission::Reservation>,
}
impl std::ops::Deref for CommittedSegments {
    type Target = Vec<(u32, u64)>;
    fn deref(&self) -> &Self::Target { &self.segments }
}

/// Where an index publishes its current [`ReaderView`]. Shared (`Arc`) so a
/// shard can read the current view without taking the index lock.
pub struct ViewSlot {
    committed: Arc<parking_lot::RwLock<CommittedSegments>>,
    /// The current view and the generation the next one gets (assigned
    /// under this mutex, so views install in generation order).
    current: parking_lot::Mutex<(Arc<ReaderView>, u64)>,
    /// Published views alive, by generation (count, when published).
    live: parking_lot::Mutex<std::collections::BTreeMap<u64, (usize, std::time::Instant)>>,
    me: Weak<ViewSlot>,
}

impl ViewSlot {
    pub fn new() -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            current: parking_lot::Mutex::new((Arc::new(ReaderView::empty()), 1)),
            live: parking_lot::Mutex::new(std::collections::BTreeMap::new()),
            me: me.clone(),
            committed: Default::default(),
        })
    }

    /// Called only after a successful manifest replacement (or reopening it).
    pub(crate) fn prepare_committed(segments: impl ExactSizeIterator<Item = (u32, u64)>) -> crate::error::Result<crate::admission::Allocation<Vec<(u32, u64)>>> {
        let memory = crate::admission::shared().reserve((segments.len() as u64).saturating_mul(std::mem::size_of::<(u32, u64)>() as u64), "committed segment registry")?;
        Ok(crate::admission::Allocation::new(segments.collect(), memory))
    }

    /// Whether the last successful commit lists segment `id`.
    pub(crate) fn is_committed(&self, id: u32) -> bool {
        self.committed.read().iter().any(|(c, _)| *c == id)
    }

    pub(crate) fn set_committed(&self, segments: crate::admission::Allocation<Vec<(u32, u64)>>) {
        let (segments, memory) = segments.into_parts();
        *self.committed.write() = CommittedSegments { segments, _memory: Some(memory) };
    }

    /// The generation the next published view gets: a delete made visible
    /// now is first excluded by that view, and counted live by every older
    /// one.
    pub fn next_generation(&self) -> u64 {
        self.current.lock().1
    }

    /// The oldest published view still alive (`None`: none). Views searching
    /// the live mutable graph never lose a node they count live: a node
    /// deleted as of generation `g` is physically removed only once this is
    /// `>= g`. Retained (PIT) views search their own captured graph and do
    /// not count here.
    pub fn min_live_generation(&self) -> Option<u64> {
        self.live.lock().keys().next().copied()
    }

    /// How long the oldest superseded published view still alive has been
    /// around (zero when only the current view is): a search or capture
    /// holding an old view that long delays mutable-graph delete repair.
    pub fn oldest_live_view_age(&self) -> std::time::Duration {
        let current = self.generation();
        match self.live.lock().iter().next() {
            Some((&generation, &(_, published))) if generation < current => published.elapsed(),
            _ => std::time::Duration::ZERO,
        }
    }

    fn release(&self, generation: u64) {
        let mut live = self.live.lock();
        if let Some((count, _)) = live.get_mut(&generation) {
            *count -= 1;
            if *count == 0 {
                live.remove(&generation);
            }
        }
    }

    /// The current view. The slot's mutex is held only to clone the `Arc`.
    pub fn load(&self) -> Arc<ReaderView> {
        self.current.lock().0.clone()
    }

    /// Generation of the current view.
    pub fn generation(&self) -> u64 {
        self.current.lock().0.generation
    }

    /// Install a view over `segments` (already reader snapshots) and
    /// `frozen` (memtables with their frozen deletions). The previous view is
    /// dropped outside the slot's mutex (it may hold the last reference to
    /// large memtables).
    pub(crate) fn publish_admitted(&self, segments: Vec<Arc<Segment>>, frozen: Vec<(Arc<SegmentWriter>, Arc<BitSet>)>, non_searchable_segment_docs: Vec<(u32, u64, u64)>, memory: Arc<crate::admission::Reservation>) {
        let old = {
            let mut current = self.current.lock();
            let generation = current.1;
            current.1 += 1;
            self.live.lock().entry(generation).or_insert((0, std::time::Instant::now())).0 += 1;
            let charge = frozen_charge(&frozen);
            let view = Arc::new(ReaderView { segments, frozen, generation, non_searchable_segment_docs, registry: Some(self.me.clone()), committed: self.committed.clone(), _charge: charge, _memory: Some(memory) });
            std::mem::replace(&mut current.0, view)
        };
        drop(old);
    }

    pub(crate) fn view_container_bytes(segments: usize, frozen: usize, pending: usize) -> u64 {
        (segments as u64).saturating_mul((std::mem::size_of::<Segment>() + 64) as u64)
            .saturating_add((frozen as u64).saturating_mul(96))
            .saturating_add((pending as u64).saturating_mul(24))
            .saturating_add(if segments + frozen + pending == 0 { 0 } else { 256 })
    }

    #[cfg(test)]
    fn publish(&self, segments: Vec<Arc<Segment>>, frozen: Vec<(Arc<SegmentWriter>, Arc<BitSet>)>, pending: Vec<(u32,u64,u64)>) {
        let memory = crate::admission::shared().reserve(Self::view_container_bytes(segments.len(), frozen.len(), pending.len()), "test view containers").unwrap();
        self.publish_admitted(segments, frozen, pending, Arc::new(memory));
    }

    /// A retained copy of the current view for a point in time / scroll /
    /// ES|QL capture: the same leaves and frozen deletions, with the
    /// segments carrying a captured vector store
    /// ([`Segment::capture_read_view`]: point owners, liveness and mutable
    /// graphs as of this call; see the module docs for the gap this leaves).
    /// No index lock is taken.
    pub fn retain(&self) -> Result<ReaderView> {
        let view = self.load();
        let memory = crate::admission::shared().reserve(Self::view_container_bytes(view.segments.len(), view.frozen.len(), view.non_searchable_segment_docs.len()), "retained reader containers")?;
        let segments = Segment::capture_read_view(&view.segments)?;
        Ok(ReaderView {
            segments,
            frozen: view.frozen.clone(),
            generation: view.generation,
            non_searchable_segment_docs: view.non_searchable_segment_docs.clone(),
            registry: None,
            committed: self.committed.clone(),
            _charge: frozen_charge(&view.frozen),
            _memory: Some(Arc::new(memory)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_stats_enumeration_includes_each_pending_unit() {
        let mut view = super::ReaderView::empty();
        view.non_searchable_segment_docs = vec![(0, 3, 30), (1, 7, 70), (2, 11, 110)];
        assert_eq!(view.segment_stats().iter().map(|s| s.bytes).collect::<Vec<_>>(), vec![30, 70, 110], "a sealed, unpublished segment reports its bytes");
        assert_eq!(view.segment_doc_counts_for_stats(), vec![3, 7, 11]);
        assert_eq!(view.segments_for_stats(), 3);
        assert_eq!(view.num_leaves(), 0);
    }

    #[test]
    fn segment_stats_memtables_uncommitted_and_commit_only_units_visible() {
        let slot = ViewSlot::new();
        slot.publish(Vec::new(), vec![(Arc::new(SegmentWriter::new(1)), Arc::new(BitSet::new(0)))], vec![(2, 9, 0)]);
        slot.set_committed(ViewSlot::prepare_committed(vec![(3, 7)].into_iter()).unwrap());
        let view = slot.load();
        let entries = view.segment_stats();
        assert_eq!(entries.iter().map(|s| (s.id, s.search, s.committed)).collect::<Vec<_>>(),
            vec![(1, true, false), (2, false, false), (3, false, true)]);
        assert_eq!(view.segments_for_stats(), 3);
        slot.set_committed(ViewSlot::prepare_committed(vec![(2, 9)].into_iter()).unwrap());
        assert!(view.segment_stats().iter().any(|s| s.id == 2 && s.committed && !s.search));
        assert!(!view.segment_stats().iter().any(|s| s.id == 3));
    }

    #[test]
    fn views_charge_their_frozen_deletions_until_dropped() {
        let slot = ViewSlot::new();
        let w = Arc::new(SegmentWriter::new(0));
        let dels = Arc::new(BitSet::new(1000));
        slot.publish(Vec::new(), vec![(w, dels)], Vec::new());
        let view = slot.load();
        assert_eq!(view._charge.bytes(), (1000u64.div_ceil(64)) * 8 + 64);
        let retained = slot.retain().unwrap();
        assert_eq!(retained._charge.bytes(), view._charge.bytes());
        assert_eq!(ReaderView::empty()._charge.bytes(), 0);
    }

    #[test]
    fn oldest_live_view_age_tracks_superseded_views_only() {
        let slot = ViewSlot::new();
        slot.publish(Vec::new(), Vec::new(), Vec::new());
        assert_eq!(slot.oldest_live_view_age(), std::time::Duration::ZERO, "current view only");
        let held = slot.load();
        slot.publish(Vec::new(), Vec::new(), Vec::new());
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(slot.oldest_live_view_age() >= std::time::Duration::from_millis(5));
        assert_eq!(slot.min_live_generation(), Some(held.generation));
        drop(held);
        assert_eq!(slot.oldest_live_view_age(), std::time::Duration::ZERO);
        assert_eq!(slot.min_live_generation(), Some(slot.generation()));
    }
}
