//! The Index: top-level API for creating, indexing, and searching.
//!
//! An Index is a collection of segments. Documents are added to
//! per-thread memtables; `publish` (ES refresh) makes them searchable
//! without encoding, and they are later sealed into segments, either
//! inline (`auto_seal`) or by background seal / tiered-merge jobs
//! (`begin_seal` / `begin_merge`, which the ES layer drives).
//!
//! Persistence: an index lives in a [`Directory`]. `Index::create` and
//! `Index::open` manage a `meta.json` commit point; `commit()` writes
//! segment files, deletion bitmaps and the meta file atomically.
//!
//! Dense vectors live in the index's [`VectorStore`] (see
//! [`crate::vector::store`]): text segments only map `doc -> PointId`, so
//! text merges never touch vectors or HNSW graphs; graphs are built only
//! by [`Index::optimize_vectors`]. The store's segment list and id
//! counters are part of the `meta.json` commit point.

use crate::codec::directory::{Directory, RamDirectory};
use crate::query::bm25::Bm25Params;
use crate::query::collector::collect_top_k;
use crate::query::reader::{field_stats, term_stats, LeafReader};
use crate::error::{Result, RusticError};
/// Sealed segments at least this large get their mutable node maps built
/// at publication (`VectorStore::warm_mutable_projections`); a smaller
/// segment's map costs its first query well under a millisecond.
#[cfg(feature = "mutable-hnsw-poc")]
const WARM_PROJECTION_DOCS: u32 = 65_536;
use crate::segment::{Segment, SegmentWriter};
use crate::types::*;
use crate::vector::store::{OptimizeReport, OptimizerBacklog, OptimizerPolicy, PlannerConfig, SearchStats, VectorStore};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const META_FILE: &str = "meta.json";

/// Serialize twice: count without buffering, then reserve the exact encoded
/// capacity before producing it. The guard outlives the write and its buffer.
fn write_admitted_json<T: Serialize>(directory: &dyn Directory, name: &str, value: &T) -> Result<()> {
    struct Count(u64);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len() as u64).ok_or_else(|| std::io::Error::other("JSON length overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut count = Count(0);
    serde_json::to_writer_pretty(&mut count, value).map_err(|e| RusticError::Serialization(e.to_string()))?;
    let memory = crate::admission::shared().reserve(count.0.saturating_mul(2), "commit JSON encoding and write")?;
    let capacity = usize::try_from(count.0).map_err(|_| RusticError::Memory("commit JSON length overflow".into()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|e| RusticError::Memory(format!("commit JSON allocation refused: {e}")))?;
    bytes.resize(capacity, 0);
    // Never allow the second serialization pass to grow beyond admission.
    // A stateful serializer must fail before the durable manifest changes.
    let mut output = std::io::Cursor::new(bytes.as_mut_slice());
    serde_json::to_writer_pretty(&mut output, value).map_err(|e| RusticError::Serialization(e.to_string()))?;
    if output.position() != count.0 {
        return Err(RusticError::Serialization("commit JSON size changed between passes".into()));
    }
    directory.write(name, &bytes)?;
    drop(bytes);
    drop(memory);
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct MetaSegment {
    id: u32,
    num_docs: u32,
}

#[derive(Serialize, Deserialize, Default)]
struct MetaVectorStore {
    next_point_id: u64,
    next_segment_id: u32,
    /// Live vector segment ids (`vec_XXXXXXXX.rvec`).
    segments: Vec<u32>,
    #[serde(default)]
    mutable_fields: Vec<crate::vector::store::MutableFieldManifest>,
    #[serde(default)]
    next_mutable_generation: u64,
    /// The DiskBBQ indexes' files (`ivf_*.rivc` / `.rivp`).
    #[serde(default)]
    ivf_fields: Vec<crate::vector::ivf_field::IvfFieldManifest>,
}

#[derive(Serialize, Deserialize)]
struct Meta {
    name: String,
    schema: Vec<Field>,
    settings: IndexSettings,
    segments: Vec<MetaSegment>,
    next_segment_id: u32,
    #[serde(default)]
    vector_store: MetaVectorStore,
}

/// An index-level k-NN request (ES `knn` section).
#[derive(Clone, Debug)]
pub struct KnnRequest {
    pub field: u16,
    pub vector: Vec<f32>,
    pub k: usize,
    /// HNSW beam width (ES `num_candidates`); at least `k` is used.
    pub num_candidates: usize,
    /// Pre-filter: only docs matching this query are candidates.
    pub filter: Option<crate::query::query::Query>,
    /// Drop hits scoring below this.
    pub similarity_threshold: Option<f32>,
    /// Multiplies the scores.
    pub boost: f32,
}

impl KnnRequest {
    pub fn new(field: u16, vector: Vec<f32>, k: usize) -> Self {
        Self { field, vector, k, num_candidates: k.saturating_mul(10).max(100), filter: None, similarity_threshold: None, boost: 1.0 }
    }
}

/// A background flush detached from its [`Index`] (see
/// [`Index::begin_seal`]): closed memtables, encoded into segments by
/// [`SealJob::run`] without any index lock (they stay readable and take
/// deletes meanwhile), installed by [`Index::finish_seal`].
pub struct SealJob {
    admission: crate::admission::Context,
    /// Each memtable, and whether it was published when the job began.
    mems: Vec<(Arc<SegmentWriter>, bool)>,
    directory: Arc<dyn Directory>,
    vectors: Arc<VectorStore>,
    sort: Option<crate::index_sort::IndexSort>,
    sorted_ids: Vec<u32>,
    /// Several published memtables (no vectors, no index sort) are written
    /// as one segment with this id (Lucene flushes a DWPT's RAM buffer as one
    /// segment; refresh-sized memtables would otherwise each become a tiny
    /// segment).
    coalesce_id: Option<u32>,
    /// Each memtable's deletions when the job began (its capture point:
    /// a flush commits these, see [`Index::finish_seal_captured`]).
    captured: Vec<crate::postings::BitSet>,
    /// A large coalesced job's place among the node's
    /// [`MAX_LARGE_SEALS`] (released when the job is dropped or run).
    _large: Option<LargeSeal>,
    _memory: crate::admission::Reservation,
}

/// Coalesced seal jobs above [`Index::COALESCE_MAX_BYTES`] running at once
/// on the node: each holds its memtables until it installs and encodes
/// unthrottled, so more of them only lengthen stalls and refresh waits.
/// Past the bound a job takes the ordinary 128 MiB.
const MAX_LARGE_SEALS: usize = 2;
static LARGE_SEALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A place among [`LARGE_SEALS`].
struct LargeSeal;

impl LargeSeal {
    fn acquire() -> Option<LargeSeal> {
        use std::sync::atomic::Ordering::AcqRel;
        if LARGE_SEALS.fetch_add(1, AcqRel) < MAX_LARGE_SEALS {
            Some(LargeSeal)
        } else {
            LARGE_SEALS.fetch_sub(1, AcqRel);
            None
        }
    }
}

impl Drop for LargeSeal {
    fn drop(&mut self) {
        LARGE_SEALS.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// The outcome of a [`SealJob`]: each memtable with its segment, or (when
/// coalesced) every memtable with the one segment holding all of them, in
/// order, or (index-sorted) in sort order with each memtable's doc map
/// (old doc → doc in the segment).
/// What a coalescing seal job encoded ([`SealJob::run`]).
enum Coalesced {
    /// One segment (and, index-sorted, each memtable's doc map into it).
    One(Segment, Option<crate::admission::Allocation<Vec<Vec<DocId>>>>),
    /// The memtables' own segments, in job order: coalescing was refused.
    Parts(Vec<Segment>),
}

thread_local! {
    /// Set by [`collect_retired_writers`]: seal installs on this thread keep
    /// the memtables they replace here instead of freeing them in place.
    static RETIRED_WRITERS: std::cell::RefCell<Option<Vec<Arc<SegmentWriter>>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f`, returning with its result the memtables the seal installs it
/// made replaced (their last references, when nothing else holds them), for
/// the caller to free after releasing the core lock, or off its thread. A
/// sorted refresh's coalesced seal replaces dozens of parked memtables;
/// freeing them inside the install cost about 9 ms of a 45 ms search-idle
/// refresh (10k TSDB docs) under the core write lock.
pub fn collect_retired_writers<R>(f: impl FnOnce() -> R) -> (R, Vec<Arc<SegmentWriter>>) {
    let outer = RETIRED_WRITERS.with(|r| r.borrow_mut().replace(Vec::new()));
    let result = f();
    let retired = RETIRED_WRITERS.with(|r| std::mem::replace(&mut *r.borrow_mut(), outer)).unwrap_or_default();
    (result, retired)
}

fn retire_writers(writers: Vec<Arc<SegmentWriter>>) {
    let writers = RETIRED_WRITERS.with(|r| match r.borrow_mut().as_mut() { Some(retired) => { retired.extend(writers); None } None => Some(writers) });
    drop(writers);
}

pub struct SealedMemtables {
    done: Vec<(Arc<SegmentWriter>, Result<(Segment, Option<crate::admission::Allocation<Vec<DocId>>>)>)>,
    coalesced: Option<(Vec<Arc<SegmentWriter>>, Result<(Segment, Option<crate::admission::Allocation<Vec<Vec<DocId>>>>)>)>,
    captured: Vec<crate::postings::BitSet>,
    _memory: crate::admission::Reservation,
}

/// A seal install refused before it changed anything
/// ([`Index::try_finish_seal_captured`]): the job, claims and encoded
/// segments intact, to retry or abandon.
pub struct SealInstallRefused {
    pub error: RusticError,
    pub sealed: SealedMemtables,
}

#[cfg(test)]
thread_local! { static REFUSE_SEAL_INSTALL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
thread_local! { static REFUSE_SEAL_COALESCE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Deletion bitmaps by segment id: what a commit writes for those
/// segments instead of their current deletions ([`CommitJob::with_deletions`]).
pub type DeletionsAt = HashMap<u32, Arc<crate::postings::BitSet>>;

impl SealJob {
    /// Memtables this job seals.
    pub fn num_memtables(&self) -> usize {
        self.mems.len()
    }

    /// Whether the job coalesces its memtables into one segment.
    pub fn coalesces(&self) -> bool {
        self.coalesce_id.is_some()
    }

    /// Documents being sealed.
    pub fn num_docs(&self) -> u64 {
        self.mems.iter().map(|(w, _)| w.num_docs() as u64).sum()
    }

    /// Estimated heap bytes of the memtables being sealed.
    pub fn estimated_bytes(&self) -> u64 {
        self.mems.iter().map(|(w, _)| w.estimated_bytes() as u64).sum()
    }

    /// Encode every memtable (concurrently when several). No index lock
    /// is needed.
    /// Estimated heap bytes of the memtables this job seals (its share of
    /// the backlog; the owner measures seal throughput with it).
    pub fn memtable_bytes(&self) -> u64 {
        self.mems.iter().map(|(w, _)| w.estimated_bytes() as u64).fold(0, u64::saturating_add)
    }

    /// Encode threads [`SealJob::run`] uses at once: one for a single
    /// memtable or with vectors, else up to [`Index::MAX_PARALLEL_SEALS`].
    pub fn encode_parallelism(&self) -> usize {
        // A coalesced job encodes its memtables as one (one encode thread,
        // see `union_one_pass` / `sorted_one_pass`): reporting one thread
        // per memtable left a refresh's coalesced chunks running one at a
        // time.
        if self.mems.len() == 1 || self.coalesce_id.is_some() || self.mems.iter().any(|(w, _)| w.has_vectors()) { 1 } else { self.mems.len().min(Index::MAX_PARALLEL_SEALS) }
    }

    /// Whether the job encodes under a [`crate::admission::forced`] context
    /// (it seals a replay's memtables: charged, never refused).
    pub fn is_forced(&self) -> bool {
        self.admission.is_forced()
    }

    /// The memtables this job seals.
    pub fn memtable_ids(&self) -> Vec<u32> {
        self.mems.iter().map(|(w, _)| w.id).collect()
    }

    pub fn run(mut self) -> SealedMemtables {
        let _admission = self.admission.enter();
        let profile = tracing::enabled!(target: "seal_profile", tracing::Level::DEBUG);
        let started = std::time::Instant::now();
        let captured = std::mem::take(&mut self.captured);
        let dir = &*self.directory;
        let store = &self.vectors;
        // An unsorted coalesced job: its memtables as one memtable in their
        // order, encoded once, as a Lucene DWPT that took many requests'
        // documents is flushed as one segment (else each memtable is written
        // and the segments merged: everything written twice, read once).
        if let (None, Some(id)) = (self.sort.as_ref(), self.coalesce_id) {
            let started = std::time::Instant::now();
            if let Some(merged) = union_one_pass(&self.mems, id, dir, store) {
                tracing::debug!(target: "seal_profile", memtables = self.mems.len(), docs = self.num_docs(), ok = merged.is_ok(),
                    total_ms = started.elapsed().as_millis() as u64, "coalesced seal (one pass)");
                let writers = self.mems.into_iter().map(|(w, _)| w).collect();
                return SealedMemtables { done: Vec::new(), coalesced: Some((writers, merged)), captured, _memory: self._memory };
            }
        }
        // An index-sorted coalesced job: its memtables as one memtable in
        // sort order, encoded once (else each memtable is written and the
        // segments are sort-merged: everything encoded twice, decoded once).
        if let (Some(sort), Some(id)) = (self.sort.as_ref(), self.coalesce_id) {
            let started = std::time::Instant::now();
            if let Some(merged) = sorted_one_pass(&self.mems, sort, id, dir, store) {
                tracing::debug!(target: "seal_profile", memtables = self.mems.len(), docs = self.num_docs(), ok = merged.is_ok(),
                    total_ms = started.elapsed().as_millis() as u64, "sorted seal (one pass)");
                let writers = self.mems.into_iter().map(|(w, _)| w).collect();
                return SealedMemtables { done: Vec::new(), coalesced: Some((writers, merged)), captured, _memory: self._memory };
            }
        }
        let encode = |ordinal: usize, w: &Arc<SegmentWriter>, published: bool| -> Result<(Segment, Option<crate::admission::Allocation<Vec<DocId>>>)> {
            let segment = w.write_to_store_ref(dir, Some(store), !published)?;
            // A coalesced job sorts once, over all its memtables (below).
            let Some(sort) = self.sort.as_ref().filter(|_| self.coalesce_id.is_none()) else { return Ok((segment, None)); };
            let input = Arc::new(segment);
            let mut memory = sort.reserve_workspace(std::slice::from_ref(&input))?;
            let leaf = LeafReader::Segment(&input);
            // Preserve EVERY row. Deletion visibility can change while the job
            // runs, and installation remaps the writer's current masks.
            let roots = (0..input.num_docs).filter(|&doc| leaf.is_root(doc)).map(|doc| (0, doc)).collect();
            let mut order = Vec::with_capacity(input.num_docs as usize);
            for (_, root) in sort.sort_docs(&[leaf], roots) {
                let begin = input.nested_info().map_or(root, |n| n.block_start(root));
                order.extend((begin..=root).map(|doc| (0, doc)));
            }
            let mut remap = vec![0; input.num_docs as usize];
            for (new, &(_, old)) in order.iter().enumerate() { remap[old as usize] = new as DocId; }
            let visible = [Arc::new(crate::postings::BitSet::new(input.num_docs))];
            // A seal: never throttled like a merge.
            let sorted = crate::merge::merge_segments_snapshot_detached(crate::encode_pool(), dir,
                std::slice::from_ref(&input), self.sorted_ids[ordinal], &visible, Some(order), false)?;
            // The temporary unsorted segment was never an installed leaf. Its
            // vector point owners are reconciled only at finish_seal.
            dir.delete(&crate::codec::segment_file_name(input.id))?;
            drop(visible);
            memory.shrink(memory.bytes().saturating_sub((remap.capacity()*std::mem::size_of::<DocId>()) as u64));
            Ok((sorted, Some(crate::admission::Allocation::new(remap,memory))))
        };
        // Always return each writer to finish_seal, including after an encoder
        // unwind. Otherwise its explicit claim would permanently block refresh
        // and all subsequent seals. Process aborts cannot be recovered here.
        let admission = crate::admission::Context::capture();
        let encode = |ordinal, w: &Arc<SegmentWriter>, published| {
            let _admission = admission.enter();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(ordinal, w, published)))
                .unwrap_or_else(|payload| {
                    let detail = payload.downcast_ref::<String>().map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("non-string panic payload");
                    Err(RusticError::Segment(format!("seal encoder panicked: {detail}")))
                })
        };
        let done = if self.mems.len() == 1 || self.mems.iter().any(|(w, _)| w.has_vectors()) {
            self.mems.iter().enumerate().map(|(i, (w, published))| encode(i, w, *published)).collect()
        } else {
            // A few memtables at a time (see `Index::MAX_PARALLEL_SEALS`): a
            // backlog sealed at once nested its encode jobs on the encode
            // workers' stacks until one overflowed.
            let encode = &encode;
            let mut done = Vec::with_capacity(self.mems.len());
            let indexed: Vec<(usize, &(Arc<SegmentWriter>, bool))> = self.mems.iter().enumerate().collect();
            // A foreground seal's memtables encode on the foreground pool
            // from these threads too.
            let foreground = crate::foreground_encode_active();
            for group in indexed.chunks(Index::MAX_PARALLEL_SEALS) {
                done.extend(std::thread::scope(|scope| {
                    let handles: Vec<_> = group.iter().map(|&(i, (w, published))|
                        scope.spawn(move || if foreground { crate::foreground_encode(|| encode(i, w, *published)) } else { encode(i, w, *published) })).collect();
                    handles.into_iter().map(|h| h.join().expect("flush thread panicked")).collect::<Vec<_>>()
                }));
            }
            done
        };
        if profile {
            tracing::debug!(target: "seal_profile", memtables = self.mems.len(), docs = self.num_docs(), bytes = self.estimated_bytes(),
                sorted = self.sort.is_some(), coalesced = self.coalesce_id.is_some(), parts_ms = started.elapsed().as_millis() as u64, "seal parts encoded");
        }
        if let Some(id) = self.coalesce_id {
            // One segment for the job: the memtables' segments merged in
            // order, or (index-sorted) merge-sorted into one sorted segment
            // (nothing dropped: deletes are applied at installation through
            // each memtable's doc map). Vector rows come along in the merged
            // point map; their owners move to the segment at installation.
            let sort = self.sort.clone();
            let merged = (|| -> Result<Coalesced> {
                let mut parts = Vec::with_capacity(done.len());
                for part in done { parts.push(Arc::new(part?.0)); }
                let keep: Vec<Arc<crate::postings::BitSet>> = parts.iter().map(|s| Arc::new(crate::postings::BitSet::new(s.num_docs))).collect();
                let mut sort_memory = sort.as_ref().map(|sort| sort.reserve_workspace(&parts)).transpose()?;
                let (order, map) = match &sort {
                    None => (None, None),
                    Some(sort) => {
                        let leaves: Vec<LeafReader<'_>> = parts.iter().map(|s| LeafReader::Segment(s)).collect();
                        // EVERY row (deletion visibility can change while the
                        // job runs); a nested block moves with its root.
                        let roots: Vec<(usize, DocId)> = parts.iter().enumerate().flat_map(|(i, s)| {
                            let leaf = LeafReader::Segment(s);
                            (0..s.num_docs).filter(move |&d| leaf.is_root(d)).map(move |d| (i, d))
                        }).collect();
                        let mut order = Vec::with_capacity(parts.iter().map(|s| s.num_docs as usize).sum());
                        for (i, root) in sort.sort_docs(&leaves, roots) {
                            let begin = parts[i].nested_info().map_or(root, |n| n.block_start(root));
                            order.extend((begin..=root).map(|d| (i, d)));
                        }
                        let mut map: Vec<Vec<DocId>> = parts.iter().map(|s| vec![0; s.num_docs as usize]).collect();
                        for (new, &(i, old)) in order.iter().enumerate() { map[i][old as usize] = new as DocId; }
                        (Some(order), Some(map))
                    }
                };
                let merged = crate::merge::merge_segments_snapshot_detached(crate::encode_pool(), dir, &parts, id, &keep, order, false);
                #[cfg(test)]
                let merged = if REFUSE_SEAL_COALESCE.with(|n| { let left = n.get(); if left > 0 { n.set(left - 1); } left > 0 }) {
                    if let Ok(m) = merged { let _ = dir.delete(&crate::codec::segment_file_name(m.id)); }
                    Err(RusticError::Memory("[core] [merge term dictionary growth] needs 129 bytes (test)".into()))
                } else { merged };
                // Coalescing is optional: refused memory (a flush beside
                // in-flight bulks, whose memtables hold most of the core
                // budget) leaves the memtables' own segments, each installed
                // as an uncoalesced job would have (unsorted indices; a
                // sorted one's parts are unsorted and must be merged).
                if let (Err(RusticError::Memory(reason)), None) = (&merged, &sort) {
                    tracing::debug!("seal coalescing of {} memtables refused, installing them as they are: {reason}", parts.len());
                    drop(keep);
                    let parts: Option<Vec<Segment>> = parts.into_iter().map(|p| Arc::try_unwrap(p).ok()).collect();
                    if let Some(parts) = parts {
                        return Ok(Coalesced::Parts(parts));
                    }
                    return merged.map(|_| unreachable!("merged is an error here"));
                }
                // The parts were never installed leaves.
                for part in &parts { let _ = dir.delete(&crate::codec::segment_file_name(part.id)); }
                merged.map(|m| {
                    let map = map.map(|map| {
                        let mut memory = sort_memory.take().expect("sorted remap admission");
                        let bytes = map.capacity()*std::mem::size_of::<Vec<DocId>>() + map.iter().map(|v|v.capacity()*std::mem::size_of::<DocId>()).sum::<usize>();
                        memory.shrink(memory.bytes().saturating_sub(bytes as u64));
                        crate::admission::Allocation::new(map,memory)
                    });
                    Coalesced::One(m,map)
                })
            })();
            let merged = match merged {
                Ok(Coalesced::Parts(parts)) => {
                    let done = self.mems.into_iter().map(|(w, _)| w).zip(parts.into_iter().map(|p| Ok((p, None)))).collect();
                    return SealedMemtables { done, coalesced: None, captured, _memory: self._memory };
                }
                Ok(Coalesced::One(m, map)) => Ok((m, map)),
                Err(e) => Err(e),
            };
            if profile {
                tracing::debug!(target: "seal_profile", memtables = self.mems.len(), total_ms = started.elapsed().as_millis() as u64, "seal coalesced");
            }
            let writers = self.mems.into_iter().map(|(w, _)| w).collect();
            return SealedMemtables { done: Vec::new(), coalesced: Some((writers, merged)), captured, _memory:self._memory };
        }
        SealedMemtables { done: self.mems.into_iter().map(|(w, _)| w).zip(done).collect(), coalesced: None, captured, _memory:self._memory }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: seal index-sorted coalesced jobs the two-pass way (reference).
    static NO_ONE_PASS_SORTED_SEAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`SealJob::run`]'s one-pass sorted seal of `mems` (closed memtables of
/// an index-sorted core) as segment `id`: the memtables' docs sorted, the
/// memtables written as one ([`SegmentWriter::sorted_union`]) and encoded
/// once, with each memtable's `old -> new` doc map for installation (as the
/// sort-merge of the memtables' segments produces). Every row is kept
/// (deletion visibility can change while the job runs: installation
/// remaps each writer's current masks). `None`: this path does not apply
/// (the caller writes and sort-merges).
#[allow(clippy::type_complexity)]
fn sorted_one_pass(mems: &[(Arc<SegmentWriter>, bool)], sort: &crate::index_sort::IndexSort, id: u32, dir: &dyn Directory, store: &Arc<VectorStore>)
    -> Option<Result<(Segment, Option<crate::admission::Allocation<Vec<Vec<DocId>>>>)>> {
    #[cfg(test)]
    if NO_ONE_PASS_SORTED_SEAL.with(|n| n.get()) {
        return None;
    }
    let writers: Vec<&SegmentWriter> = mems.iter().map(|(w, _)| &**w).collect();
    if writers.len() < 2 || !SegmentWriter::sorted_union_supported(&writers) {
        return None;
    }
    let docs: u64 = writers.iter().map(|w| w.num_docs() as u64).sum();
    // The order and the maps, the sort's keys, and the union memtable
    // (about the inputs' size) until it is encoded.
    let bytes = docs.saturating_mul(256).saturating_add(writers.iter().map(|w| w.indexing_bytes() as u64).sum::<u64>());
    let mut memory = match crate::admission::shared().reserve(bytes, "one-pass sorted seal workspace") {
        Ok(memory) => memory,
        Err(_) => return None,
    };
    let leaves: Vec<LeafReader<'_>> = writers.iter().map(|w| LeafReader::Memtable(w, None)).collect();
    let roots: Vec<(usize, DocId)> = writers.iter().enumerate().flat_map(|(i, w)| (0..w.num_docs()).map(move |d| (i, d))).collect();
    let t0 = std::time::Instant::now();
    let order = sort.sort_docs(&leaves, roots);
    drop(leaves);
    let t1 = std::time::Instant::now();
    // On the encode pool (as every seal's encoding): `sorted_union`'s
    // joins on the global pool queued behind the parallel bulk writes
    // running there, which wait in `wait_for_seals` for this very seal.
    let union = crate::on_pool(crate::encode_pool(), || SegmentWriter::sorted_union(&writers, &order, id))?;
    let t2 = std::time::Instant::now();
    let mut map: Vec<Vec<DocId>> = writers.iter().map(|w| vec![0; w.num_docs() as usize]).collect();
    for (new, &(i, old)) in order.iter().enumerate() { map[i][old as usize] = new as DocId; }
    drop(order);
    let segment = union.write_to_store_ref(dir, Some(store), false);
    // Freed off the refresh's path (a search-idle refresh waits for this
    // seal; the union holds every sealed doc).
    // On the encode pool, not the global one (see `sorted_union` above):
    // with every global worker waiting for seals the union stayed alive.
    crate::encode_pool().spawn(move || drop(union));
    tracing::debug!(target: "seal_profile", sort_ms = (t1 - t0).as_millis() as u64, union_ms = (t2 - t1).as_millis() as u64,
        write_ms = t2.elapsed().as_millis() as u64, "one-pass stages");
    let kept = map.iter().map(|m| m.capacity() * std::mem::size_of::<DocId>() + std::mem::size_of::<Vec<DocId>>()).sum::<usize>() as u64;
    memory.shrink(memory.bytes().saturating_sub(kept));
    Some(segment.map(|segment| (segment, Some(crate::admission::Allocation::new(map, memory)))))
}

/// [`SealJob::run`]'s one-pass seal of `mems` (closed memtables of an
/// unsorted core) as segment `id`: the memtables written as one
/// ([`SegmentWriter::sorted_union`] in their own order, docs concatenated
/// as the merge of their segments numbers them, so installation needs no
/// doc map) and encoded once. `None`: this path does not apply (vectors,
/// nested docs, geo, or its workspace refused: the caller writes and merges).
#[allow(clippy::type_complexity)]
fn union_one_pass(mems: &[(Arc<SegmentWriter>, bool)], id: u32, dir: &dyn Directory, store: &Arc<VectorStore>)
    -> Option<Result<(Segment, Option<crate::admission::Allocation<Vec<Vec<DocId>>>>)>> {
    #[cfg(test)]
    if NO_ONE_PASS_SORTED_SEAL.with(|n| n.get()) {
        return None;
    }
    let writers: Vec<&SegmentWriter> = mems.iter().map(|(w, _)| &**w).collect();
    if writers.len() < 2 || !SegmentWriter::sorted_union_supported(&writers) {
        return None;
    }
    let docs: u64 = writers.iter().map(|w| w.num_docs() as u64).sum();
    // The order, and the union memtable (about the inputs' size) until it
    // is encoded.
    let bytes = docs.saturating_mul(16).saturating_add(writers.iter().map(|w| w.indexing_bytes() as u64).sum::<u64>());
    let memory = crate::admission::shared().reserve(bytes, "one-pass coalesced seal workspace").ok()?;
    let order: Vec<(usize, DocId)> = writers.iter().enumerate().flat_map(|(i, w)| (0..w.num_docs()).map(move |d| (i, d))).collect();
    // On the encode pool: see `sorted_one_pass` (writes on the global pool
    // wait for this seal; joins queued behind them stalled bulks for
    // minutes on the joins track).
    let union = crate::on_pool(crate::encode_pool(), || SegmentWriter::sorted_union(&writers, &order, id))?;
    drop(order);
    let segment = union.write_to_store_ref(dir, Some(store), false);
    drop(union);
    drop(memory);
    Some(segment.map(|segment| (segment, None)))
}

/// A background merge detached from its [`Index`] (see
/// [`Index::begin_merge`]): sealed segments merged by [`MergeJob::run`]
/// without any index lock, installed by [`Index::finish_merge`].
pub struct MergeJob {
    admission: crate::admission::Context,
    /// A force merge: encoded on [`crate::force_merge_pool`].
    force: bool,
    /// Rate-limited by the merge throttle (ES `index.merge.scheduler.auto_throttle`).
    throttled: bool,
    chosen: Vec<Arc<Segment>>,
    /// The deletions (as searches saw them) each input was merged with.
    visible: Vec<Arc<crate::postings::BitSet>>,
    new_id: u32,
    directory: Arc<dyn Directory>,
    sort: Option<crate::index_sort::IndexSort>,
    order: std::sync::OnceLock<Option<crate::admission::Allocation<Vec<(usize, DocId)>>>>,
    install_map: std::sync::OnceLock<crate::merge::DocMap>,
    /// Per input, the deletes taken while it was being merged, as of the end
    /// of encoding: `(latest, visible, (merged doc, visible?))`, computed off
    /// the index lock; installation reuses them for inputs that took no
    /// delete since.
    carried: std::sync::OnceLock<crate::admission::Allocation<Vec<CarriedDeletes>>>,
    /// The inputs' retired-file registry entries, admitted before encoding
    /// so installing never refuses after the work is done.
    retired: std::sync::OnceLock<Vec<(u32, RetiredSegment)>>,
    _planning_memory: crate::admission::Reservation,
}

/// See `MergeJob::carried`.
type CarriedDeletes = (Arc<crate::postings::BitSet>, Arc<crate::postings::BitSet>, crate::admission::Allocation<Vec<(DocId, bool)>>);

/// The deletes of input `input` (`latest`, searches seeing `visible`) that
/// the merge did not drop (absent from `before`, the snapshot it encoded
/// with), as `(merged doc, visible to searches)`.
fn carried_deletes(map: &crate::merge::DocMap, input: usize, latest: &crate::postings::BitSet,
    visible: &crate::postings::BitSet, before: &crate::postings::BitSet, budget: &Arc<crate::admission::Budget>) -> Result<crate::admission::Allocation<Vec<(DocId, bool)>>> {
    // Counted first: the list is allocated once at its exact size (most
    // deletes predate the merge, so `latest.count()` would overstate it).
    let carried = || latest.iter_set_bits().filter(|&d| !before.get(d)).filter_map(|d| map.new_id(input, d).map(|n| (n, visible.get(d))));
    let count = carried().count();
    let memory=budget.reserve((count*std::mem::size_of::<(DocId,bool)>()) as u64,"merge carried deletions")?;
    let mut out = Vec::with_capacity(count);
    out.extend(carried());
    Ok(crate::admission::Allocation::new(out,memory))
}

impl MergeJob {
    /// Whether the merge throttle applies (default `true`).
    pub fn set_throttled(&mut self, throttled: bool) {
        self.throttled = throttled;
    }

    /// The segments this merge claimed.
    pub fn segment_ids(&self) -> Vec<u32> {
        self.chosen.iter().map(|s| s.id).collect()
    }

    /// Live documents merged.
    pub fn num_docs(&self) -> u64 {
        self.visible.iter().zip(&self.chosen).map(|(d, s)| (s.num_docs - d.count().min(s.num_docs)) as u64).sum()
    }

    // Compute sorting outside the index lock, against the same deletion
    // snapshot used by encoding and the later concurrent-delete remapping.
    fn order(&self) -> Result<&Option<crate::admission::Allocation<Vec<(usize, DocId)>>>> {
        if let Some(order)=self.order.get(){return Ok(order);}
        let order=if let Some(sort)=&self.sort {
            let mut memory=sort.reserve_workspace(&self.chosen)?;
            let leaves:Vec<_>=self.chosen.iter().map(|s|LeafReader::Segment(s)).collect();
            let roots=self.chosen.iter().enumerate().flat_map(|(i,s)|{
                let dels=&self.visible[i];let leaf=LeafReader::Segment(s);
                (0..s.num_docs).filter(move|&d|!dels.get(d)&&leaf.is_root(d)).map(move|d|(i,d))
            }).collect();
            let mut order=Vec::new();
            for(i,root) in sort.sort_docs(&leaves,roots){let start=self.chosen[i].nested_info().map_or(root,|n|n.block_start(root));order.extend((start..=root).map(|doc|(i,doc)));}
            drop(leaves);
            memory.shrink(memory.bytes().saturating_sub((order.capacity()*std::mem::size_of::<(usize,DocId)>()) as u64));
            Some(crate::admission::Allocation::new(order,memory))
        }else{None};
        let _=self.order.set(order);
        Ok(self.order.get().unwrap())
    }

    fn carried_now(&self) -> Result<&crate::admission::Allocation<Vec<CarriedDeletes>>> {
        if let Some(carried)=self.carried.get(){return Ok(carried);}
        let metadata=crate::admission::shared().reserve((self.chosen.len() as u64).saturating_mul(256),"merge carried delete metadata")?;
        let map=self.install_map()?;
        let mut carried=Vec::with_capacity(self.chosen.len());
        for(i,seg) in self.chosen.iter().enumerate(){
            let(latest,visible)=(seg.latest_deletions()?,seg.deletions());
            let list=carried_deletes(map,i,&latest,&visible,&self.visible[i],&crate::admission::shared())?;
            carried.push((latest,visible,list));
        }
        let _=self.carried.set(crate::admission::Allocation::new(carried,metadata));
        Ok(self.carried.get().unwrap())
    }

    fn install_map(&self) -> Result<&crate::merge::DocMap> {
        if let Some(map)=self.install_map.get(){return Ok(map);}
        let order=self.order()?;
        let _copy=crate::admission::shared().reserve(order.as_ref().map_or(0,|o|(o.len()*std::mem::size_of::<(usize,DocId)>()) as u64),"merge install order copy")?;
        let map=crate::merge::admitted_doc_map(&self.chosen,Some(&self.visible),order.as_ref().map(|o|o.to_vec()))?;
        let _=self.install_map.set(map);
        Ok(self.install_map.get().unwrap())
    }

    /// Merge the inputs. No index lock is needed.
    pub fn run(&self) -> Result<Segment> {
        let _admission = self.admission.enter();
        // Background merges on the merge pool: their throttle sleeps never
        // hold an encode thread a seal needs.
        let pool = if self.force { crate::force_merge_pool() } else { crate::merge_pool() };
        // Inputs and deletion snapshots are immutable. A failed encoder must
        // return through the caller's release path so claimed segments and
        // active-merge counters do not remain stuck after a worker unwind.
        // This does not catch aborts (including allocation failure).
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let order=self.order()?;
            // What installation needs and can be admitted up front is, before
            // any encoding: the potentially document-sized delete remapping
            // (built off the core lock, not while foreground reads wait) and
            // the retired-file registry entries.
            self.install_map()?;
            if self.retired.get().is_none() { let _ = self.retired.set(Index::prepare_retired(&self.chosen)?); }
            let _copy=crate::admission::shared().reserve(order.as_ref().map_or(0,|o|(o.len()*std::mem::size_of::<(usize,DocId)>()) as u64),"merge encoder order copy")?;
            let merged = crate::merge::merge_segments_snapshot_detached(pool, &*self.directory, &self.chosen, self.new_id, &self.visible, order.as_ref().map(|o|o.to_vec()), self.throttled)
                .inspect_err(|_| { let _ = self.directory.delete(&crate::codec::segment_file_name(self.new_id)); })?;
            // The deletes taken meanwhile, as of now. A refusal must not leave
            // the encoded output behind until the next restart's sweep.
            if let Err(e) = self.carried_now() {
                drop(merged);
                let _ = self.directory.delete(&crate::codec::segment_file_name(self.new_id));
                return Err(e);
            }
            Ok(merged)
        })).unwrap_or_else(|payload| {
            let detail = payload.downcast_ref::<String>().map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            Err(RusticError::Segment(format!("merge encoder panicked: {detail}")))
        })
    }
}

/// A vector optimizer pass detached from its [`Index`]
/// (see [`Index::vector_optimizer`]).
pub struct VectorOptimizer {
    admission: crate::admission::Context,
    store: Arc<VectorStore>,
    directory: Arc<dyn Directory>,
    policy: OptimizerPolicy,
}

impl VectorOptimizer {
    /// Explicit vector maintenance using owned store/directory handles.
    /// Callers must release the index guard before running this work.
    pub fn force_merge(&self, max_segments: usize, only_expunge_deletes: bool) -> Result<OptimizeReport> {
        let _admission = self.admission.enter();
        if only_expunge_deletes {
            self.store.expunge_deletes(&*self.directory, &self.policy)
        } else {
            self.store.force_merge(&*self.directory, &self.policy, max_segments)
        }
    }

    /// Warm segments opened from disk (so the first query after a restart
    /// does not pay for it), then run the indexing / vacuum / merge
    /// optimizers once. Replaced files go at the next commit.
    pub fn run(&self) -> Result<OptimizeReport> {
        let _admission = self.admission.enter();
        self.store.warm();
        #[cfg(feature = "mutable-hnsw-poc")]
        self.store.run_migrations()?;
        self.store.optimize(&*self.directory, &self.policy)
    }

    /// Index fresh chunks before unrelated maintenance. Existing segments
    /// are not warmed here: a restart's warming must not precede fresh graph
    /// construction for every other shard in the same background round.
    pub fn index_pending(&self) -> Result<OptimizeReport> {
        let _admission = self.admission.enter();
        self.store.index_pending(&*self.directory, &self.policy)
    }

    /// Warm retained segments, then do at most one vacuum or merge rebuild.
    /// Call after every shard has received its indexing opportunity.
    /// Warming uses per-segment caches (O(segments) after the first pass),
    /// but cold restart warming is not bounded by the rebuild allowance.
    pub fn maintain_once(&self) -> Result<OptimizeReport> {
        let _admission = self.admission.enter();
        self.store.warm();
        // Scheduled `index_options` migrations build their graphs here, in
        // the background maintenance pass, never under a shard lock.
        #[cfg(feature = "mutable-hnsw-poc")]
        self.store.run_migrations()?;
        self.store.maintain_once(&*self.directory, &self.policy)
    }

    /// Test seam: rewrite vector segment `id` ([`VectorStore`] vacuum).
    #[cfg(test)]
    pub(crate) fn rewrite_segment(&self, id: u32) -> Result<usize> {
        self.store.rewrite_segment_for_test(&*self.directory, &self.policy, id)
    }

    /// Test seam: merge vector segments `ids` ([`VectorStore`] merge), with
    /// no index step after it.
    #[cfg(test)]
    pub(crate) fn merge_segments(&self, ids: &[u32]) -> Result<usize> {
        self.store.merge_segments_for_test(&*self.directory, &self.policy, ids)
    }

    /// Vector-store backlog only (document memtables are not counted).
    pub fn backlog(&self) -> OptimizerBacklog {
        self.store.optimizer_backlog()
    }
}

/// One hit of an index-level search: leaf index (as in [`Index::leaves`]),
/// doc within the leaf, score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KnnHit {
    pub leaf: usize,
    pub doc: DocId,
    pub score: f32,
}

/// Owned durable checkpoint inputs. The caller must prevent writes, other
/// commits, and segment installation until `run` completes. Searches may continue.
/// Refresh may continue only when it cannot seal or merge inline (see
/// [`Index::can_refresh_during_commit`]).
/// Capturing the job alone is not a concurrency barrier.
pub struct CommitJob {
    admission: crate::admission::Context,
    view_slot: Arc<crate::reader_view::ViewSlot>,
    name: String,
    schema: Vec<Field>,
    settings: IndexSettings,
    segments: Vec<Arc<Segment>>,
    hidden: Vec<Arc<Segment>>,
    next_segment_id: u32,
    directory: Arc<dyn Directory>,
    vectors: Arc<VectorStore>,
    pending_deletes: Arc<parking_lot::Mutex<Option<crate::admission::Allocation<Vec<crate::vector::store::RetiredIvfFiles>>>>>,
    retired: Arc<parking_lot::Mutex<std::collections::BTreeMap<u32, RetiredSegment>>>,
    /// Deletions to commit instead of the segments' current ones
    /// ([`CommitJob::with_deletions`]).
    deletions_at: Option<DeletionsAt>,
    /// Mutable graph checkpoints captured at prepare time
    /// ([`Index::prepare_commit_captured`]); `None`: capture at `run`.
    #[cfg(feature = "mutable-hnsw-poc")]
    captured: Option<Vec<crate::vector::store::CapturedFieldCheckpoint>>,
    _memory: crate::admission::Reservation,
    capture_memory: Option<crate::admission::Reservation>,
    /// A capture point whose graphs [`CommitJob::run`] copies first
    /// ([`Index::prepare_commit_deferred`]).
    deferred_capture: Option<VectorCapture>,
}

/// Mutable-graph checkpoints at their capture point
/// ([`Index::begin_vector_capture`]).
pub struct VectorCapture {
    admission: crate::admission::Context,
    #[cfg(feature = "mutable-hnsw-poc")]
    pending: Vec<crate::vector::store::PendingFieldCheckpoint>,
    #[cfg(feature = "mutable-hnsw-poc")]
    done: Option<Vec<crate::vector::store::CapturedFieldCheckpoint>>,
    _memory: Option<crate::admission::Reservation>,
}

impl VectorCapture {
    /// Copy the captured graphs (chunked; graph writers interleave). Call
    /// without any index or shard lock.
    pub fn finish(&mut self) -> Result<()> {
        let _admission = self.admission.enter();
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let pending = std::mem::take(&mut self.pending);
            self.done = Some(crate::vector::store::finish_mutable_checkpoint(pending)?);
        }
        Ok(())
    }
}

/// Files of a merged-away segment and a weak handle on its shared data
/// (see `Index::retired`).
struct RetiredSegment {
    names: [String; 2],
    /// Bytes of its segment file (and term data) when it was retired.
    bytes: u64,
    data: std::sync::Weak<crate::segment::SegmentData>,
    _memory: Arc<crate::admission::Reservation>,
}

impl CommitJob {
    /// Write `deletions` for the segments they name instead of their
    /// current deletions: the commit is a point in time (deletes after its
    /// capture point are above it and replay from the translog).
    pub fn with_deletions(&mut self, deletions: DeletionsAt) {
        self.deletions_at = Some(deletions);
    }

    /// Leave the segments `ids` out of the commit (sealed after its capture
    /// point: their documents are above it and replay from the translog;
    /// their files stay for the next commit).
    pub fn exclude_segments(&mut self, ids: &std::collections::HashSet<u32>) {
        self.segments.retain(|s| !ids.contains(&s.id));
        self.hidden.retain(|s| !ids.contains(&s.id));
    }

    pub fn run(mut self) -> Result<()> {
        let _admission = self.admission.enter();
        // A deferred capture (`Index::prepare_commit_deferred`): copy its
        // graphs first, off every index lock.
        if let Some(mut capture) = self.deferred_capture.take() {
            capture.finish()?;
            #[cfg(feature = "mutable-hnsw-poc")]
            {
                self.captured = Some(capture.done.take().ok_or_else(|| RusticError::Segment("vector capture not finished".into()))?);
                self.capture_memory = capture._memory.take();
            }
        }
        // Segment files sealed or merged since the last commit were written
        // unsynced (`Directory::write_parts_unsynced`): durable before the
        // manifest that may reference them is.
        self.directory.sync_unsynced()?;
        #[cfg(feature = "mutable-hnsw-poc")]
        let mut mutable_commit = self.vectors.mutable_commit.lock();
        #[cfg(feature = "mutable-hnsw-poc")]
        let _registry_memory = self.vectors.checkpoint_registry_memory()?;
        for segment in self.segments.iter().chain(&self.hidden) {
            match self.deletions_at.as_ref().and_then(|at| at.get(&segment.id)) {
                Some(bits) => segment.write_deletions_of(&*self.directory, bits)?,
                None => segment.write_deletions(&*self.directory)?,
            }
        }
        #[cfg(feature = "mutable-hnsw-poc")]
        let (written_fields, next_mutable_generation) = {
            let generation = mutable_commit.next_generation;
            let next = generation.checked_add(1).ok_or_else(|| RusticError::Segment("mutable checkpoint generation exhausted".into()))?;
            // Consume before streaming: a failed attempt is never a live manifest.
            mutable_commit.next_generation = next;
            let fields = match &self.captured {
                // Fields unchanged since the last commit keep its file.
                Some(captured) => self.vectors.write_captured_mutable_checkpoint(&*self.directory, generation, captured)?,
                None => self.vectors.write_mutable_checkpoint(&*self.directory, generation)?.into_iter()
                    .map(|entry| crate::vector::store::WrittenField { entry, mutation_generation: None, fresh: true }).collect(),
            };
            (fields, next)
        };
        #[cfg(feature = "mutable-hnsw-poc")]
        let mutable_count = written_fields.len().saturating_add(mutable_commit.files.len()).saturating_add(mutable_commit.rebuilds.len());
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let mutable_count = 0usize;
        let _workspace = match crate::admission::shared().reserve((mutable_count as u64).saturating_mul(1024)
            .saturating_add(((self.segments.len() + self.hidden.len()) as u64).saturating_mul(128)).saturating_add(4096), "commit publication workspace") {
            Ok(memory) => memory,
            Err(error) => {
                #[cfg(feature = "mutable-hnsw-poc")]
                for written in written_fields.iter().filter(|written| written.fresh) { let _ = self.directory.delete(&written.entry.filename()); }
                return Err(error);
            }
        };
        #[cfg(feature = "mutable-hnsw-poc")]
        let mutable_fields: Vec<_> = written_fields.iter().map(|written| written.entry.clone()).collect();
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let (mutable_fields, next_mutable_generation) = (Vec::new(), 0);
        let vp = match self.vectors.commit_point() {
            Ok(vp) => vp,
            Err(error) => {
                #[cfg(feature = "mutable-hnsw-poc")]
                for written in written_fields.iter().filter(|written| written.fresh) { let _ = self.directory.delete(&written.entry.filename()); }
                return Err(error);
            }
        };
        let meta = Meta {
            name: self.name, schema: self.schema, settings: self.settings,
            segments: self.segments.iter().chain(&self.hidden).map(|s| MetaSegment { id: s.id, num_docs: s.num_docs }).collect(),
            next_segment_id: self.next_segment_id,
            vector_store: MetaVectorStore { next_point_id: vp.next_point_id, next_segment_id: vp.next_segment_id,
                segments: vp.segments, mutable_fields, next_mutable_generation, ivf_fields: vp.ivf },
        };
        // Everything needed for publication and cleanup is admitted before
        // durable replacement. Post-rename cleanup failure cannot roll back
        // or delay the in-memory durable checkpoint bookkeeping.
        let prepared = (|| -> Result<_> {
            let committed = crate::reader_view::ViewSlot::prepare_committed(meta.segments.iter().map(|s| (s.id, u64::from(s.num_docs))))?;
            let _cleanup_workspace = crate::admission::shared().reserve((vp.obsolete.len() as u64).saturating_mul(128).saturating_add(4096), "commit obsolete file producer")?;
            let mut obsolete: Vec<String> = vp.obsolete.iter().copied().map(crate::codec::vector_segment_file_name).collect();
            #[cfg(feature = "mutable-hnsw-poc")]
            let mutable = {
                let memory = crate::admission::shared().reserve((written_fields.len() as u64).saturating_add(mutable_commit.rebuilds.len() as u64)
                    .saturating_mul(512).saturating_add(4096), "durable mutable checkpoint registry")?;
                let current: Vec<String> = meta.vector_store.mutable_fields.iter().flat_map(|entry| entry.files()).collect();
                obsolete.extend(mutable_commit.files.iter().filter(|name| !current.contains(name)).cloned());
                let checkpointed = written_fields.iter().filter_map(|written| written.mutation_generation
                    .map(|generation| (written.entry.field, (written.entry.clone(), generation)))).collect();
                (current, checkpointed, memory)
            };
            #[cfg(not(feature = "mutable-hnsw-poc"))]
            let mutable = ();
            let obsolete = crate::vector::store::retire_ivf_files(obsolete)?;
            let pending = self.pending_deletes.lock();
            let count = pending.as_ref().map_or(0, |files| files.len()).saturating_add(vp.obsolete_files.len()).saturating_add(1);
            let memory = crate::admission::shared().reserve((count as u64).saturating_mul(std::mem::size_of::<crate::vector::store::RetiredIvfFiles>() as u64), "pending delete registry")?;
            let mut files = Vec::with_capacity(count);
            if let Some(pending) = pending.as_ref() { files.extend(pending.iter().cloned()); }
            files.push(obsolete);
            files.extend(vp.obsolete_files.iter().cloned());
            let files = crate::admission::Allocation::new(files, memory);
            write_admitted_json(&*self.directory, META_FILE, &meta)?;
            Ok((committed, mutable, files))
        })();
        let (committed, mutable, mut files) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                #[cfg(feature = "mutable-hnsw-poc")]
                for written in written_fields.iter().filter(|written| written.fresh) { let _ = self.directory.delete(&written.entry.filename()); }
                self.vectors.restore_obsolete(vp.obsolete);
                self.vectors.restore_obsolete_files(vp.obsolete_files);
                return Err(error);
            }
        };
        self.view_slot.set_committed(committed);
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let (current, checkpointed, memory) = mutable;
            mutable_commit.files = current;
            mutable_commit.checkpointed = checkpointed;
            mutable_commit.metadata_memory = Some(memory);
            for (field, (_, generation)) in &mutable_commit.checkpointed {
                if let Some(field) = self.vectors.registered_mutable_field(*field) { field.checkpoint_durable(*generation); }
            }
        }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let _ = mutable;
        // Transfer admitted names, not new unowned String copies, to retry.
        // The unreadable entries leave the registry under its lock; their
        // files are deleted outside it (a merge install takes the lock under
        // the core write lock, and deletes may be slow, e.g. object stores).
        // Entries whose delete fails go back for the next commit.
        let (_, mut cleanup_error) = delete_dead_retired(&self.retired, &*self.directory, |_| false);
        files.retain(|group| {
            for file in group.iter() {
                if self.directory.exists(file) {
                    if let Err(error) = self.directory.delete(file) {
                        if cleanup_error.is_none() { cleanup_error = Some(error); }
                        return true;
                    }
                }
            }
            false
        });
        *self.pending_deletes.lock() = Some(files);
        if let Some(error) = cleanup_error { return Err(error); }
        Ok(())
    }
}

/// Delete the files of retired segments no reader holds any more, except
/// those `keep` names; entries whose delete fails stay for a later pass.
/// Returns how many segments went and the first delete error.
fn delete_dead_retired(retired: &parking_lot::Mutex<BTreeMap<u32, RetiredSegment>>, directory: &dyn Directory, keep: impl Fn(u32) -> bool) -> (usize, Option<RusticError>) {
    let mut cleanup_error = None;
    let dead: Vec<(u32, RetiredSegment)> = {
        let mut retired = retired.lock();
        let ids: Vec<u32> = retired.iter().filter(|(id, r)| r.data.strong_count() == 0 && !keep(**id)).map(|(id, _)| *id).collect();
        ids.into_iter().filter_map(|id| retired.remove(&id).map(|r| (id, r))).collect()
    };
    let mut failed = Vec::new();
    let mut deleted_segments = 0;
    for (id, entry) in dead {
        let deleted = entry.names.iter().all(|name| {
            if !directory.exists(name) { return true; }
            match directory.delete(name) {
                Ok(()) => true,
                Err(error) => {
                    if cleanup_error.is_none() { cleanup_error = Some(error); }
                    false
                }
            }
        });
        if deleted { deleted_segments += 1; } else { failed.push((id, entry)); }
    }
    if !failed.is_empty() { retired.lock().extend(failed); }
    (deleted_segments, cleanup_error)
}

/// [`Index`]'s merge backlog before planning ever ran (see
/// [`Index::num_pending_merges`]).
const BACKLOG_UNKNOWN: usize = usize::MAX;

/// The tiered merge backlog is counted up to this many pending merges per
/// visibility class: its readers only ask whether merges are pending at all
/// and whether they pass `index.merge.scheduler.max_merge_count` (the
/// thread count plus five), and counting every proposal of a large backlog
/// is quadratic in its segments.
const PENDING_MERGES_COUNTED: usize = 64;

/// The main index structure.
pub struct Index {
    /// Index name.
    pub name: String,
    /// Schema (field definitions).
    pub schema: Vec<Field>,
    /// Where segments live.
    directory: Arc<dyn Directory>,
    /// Sealed, searchable segments.
    segments: Vec<Arc<Segment>>,
    /// In-memory segments being written to (memtables). `add_document`
    /// uses the first; `add_documents_parallel` fills one per thread
    /// (Lucene's per-thread `DocumentsWriterPerThread`). Each is a
    /// searchable leaf once it holds documents, unless the index is in
    /// near-real-time mode ([`Index::set_nrt`]), where they become visible
    /// only at the next [`Index::publish`].
    writers: Vec<SegmentWriter>,
    /// NRT mode: memtables published by [`Index::publish`] (searchable,
    /// no longer written to), until sealed into segments.
    frozen: Vec<Arc<SegmentWriter>>,
    /// NRT mode: memtables sealed before any publish made them visible
    /// (a flush or the size threshold): written, not yet searchable.
    hidden: Vec<Arc<Segment>>,
    /// NRT mode: memtables detached for a background seal
    /// ([`Index::begin_seal`]) before any publish made them visible:
    /// realtime leaves (after `hidden`), not searchable, no longer written
    /// to. A publish moves them to `frozen`.
    parked: Vec<Arc<SegmentWriter>>,
    sealing: std::collections::BTreeSet<u32>,
    /// [`Index::publish`] seals published memtables and merges small
    /// segments itself (default). Off, the owner runs those as background
    /// jobs ([`Index::begin_seal`], [`Index::begin_merge`]).
    auto_seal: bool,
    /// Near-real-time visibility (ES refresh semantics), see
    /// [`Index::set_nrt`].
    nrt: bool,
    /// NRT mode: some delete is not yet visible to searches.
    /// Some delete is pending on a searchable leaf (set by shared-access
    /// deletes too: [`Index::delete_doc_shared`]).
    pending_dels: std::sync::atomic::AtomicBool,
    /// Index sort: segments written by flushes and merges hold their docs
    /// in this order (see [`Index::set_index_sort`]).
    index_sort: Option<crate::index_sort::IndexSort>,
    preserve_published_leaves: bool,
    /// Stored-field chunk framing and compression of memtables written from
    /// now on (ES `index.codec`; see [`Index::set_stored_compression`]).
    stored_compression: crate::codec::store::StoredCompression,
    /// Next segment ID.
    next_segment_id: u32,
    /// Index settings.
    pub settings: IndexSettings,
    /// Dense vectors of every flushed segment.
    vectors: Arc<VectorStore>,
    /// Vector optimizer policy (used by [`Index::optimize_vectors`]).
    pub optimizer_policy: OptimizerPolicy,
    /// Files no commit point references any more, deleted at the next
    /// commit.
    pending_deletes: Arc<parking_lot::Mutex<Option<crate::admission::Allocation<Vec<crate::vector::store::RetiredIvfFiles>>>>>,
    /// Files of merged-away segments that a reader may still read: each
    /// with a weak handle on the segment's shared data. A commit deletes
    /// them only once the handle is dead, i.e. no reader view, PIT capture
    /// or in-flight job holds any copy of the segment (Lucene's
    /// `IndexFileDeleter` reference counts).
    retired: Arc<parking_lot::Mutex<std::collections::BTreeMap<u32, RetiredSegment>>>,
    /// Where the current [`ReaderView`](crate::reader_view::ReaderView) is
    /// published (shared with the owner, which reads it without this
    /// index's lock).
    view_slot: Arc<crate::reader_view::ViewSlot>,
    view_dirty: std::sync::atomic::AtomicBool,
    /// The merge backlog last measured ([`Index::num_pending_merges`]),
    /// reported again while its planning admission is refused.
    merge_backlog: std::sync::atomic::AtomicUsize,
    /// Segments claimed by in-flight [`MergeJob`]s (released by
    /// [`Index::finish_merge`] / [`Index::release_merge`]): concurrent jobs and
    /// foreground small-segment merges never pick them.
    merging: std::collections::BTreeSet<u32>,
    _restored_metadata_memory: Option<Arc<crate::admission::Reservation>>,
}

impl Index {
    /// Create an in-memory index (no persistence).
    pub fn new(name: &str, schema: Vec<Field>, settings: IndexSettings) -> Self {
        let index = Self {
            name: name.to_string(),
            schema,
            directory: Arc::new(RamDirectory::new()),
            segments: Vec::new(),
            writers: Vec::new(),
            frozen: Vec::new(),
            hidden: Vec::new(),
            parked: Vec::new(),
            sealing: Default::default(),
            auto_seal: true,
            nrt: false,
            pending_dels: std::sync::atomic::AtomicBool::new(false),
            index_sort: None,
            stored_compression: crate::codec::store::StoredCompression::Fast,
            preserve_published_leaves: false,
            next_segment_id: 0,
            settings,
            vectors: VectorStore::new(),
            optimizer_policy: OptimizerPolicy::default(),
            pending_deletes: Arc::new(parking_lot::Mutex::new(None)),
            retired: Arc::new(parking_lot::Mutex::new(std::collections::BTreeMap::new())),
            view_slot: crate::reader_view::ViewSlot::new(),
            view_dirty: std::sync::atomic::AtomicBool::new(false),
            merge_backlog: std::sync::atomic::AtomicUsize::new(BACKLOG_UNKNOWN),
            merging: Default::default(),
            _restored_metadata_memory: None,
        };
        index.sync_vector_field_options();
        index.publish_view().expect("empty initial reader view");
        index
    }

    /// Create a new on-disk index at `path` and write its commit point.
    pub fn create(
        path: &std::path::Path,
        name: &str,
        schema: Vec<Field>,
        settings: IndexSettings,
    ) -> Result<Self> {
        let directory: Arc<dyn Directory> = Arc::new(crate::codec::directory::MmapDirectory::open(path)?);
        Self::create_in(directory, name, schema, settings)
    }

    /// [`Index::create`] in any [`Directory`] (e.g. an object store).
    pub fn create_in(directory: Arc<dyn Directory>, name: &str, schema: Vec<Field>, settings: IndexSettings) -> Result<Self> {
        let index = Self {
            name: name.to_string(),
            schema,
            directory,
            segments: Vec::new(),
            writers: Vec::new(),
            frozen: Vec::new(),
            hidden: Vec::new(),
            parked: Vec::new(),
            sealing: Default::default(),
            auto_seal: true,
            nrt: false,
            pending_dels: std::sync::atomic::AtomicBool::new(false),
            index_sort: None,
            stored_compression: crate::codec::store::StoredCompression::Fast,
            preserve_published_leaves: false,
            next_segment_id: 0,
            settings,
            vectors: VectorStore::new(),
            optimizer_policy: OptimizerPolicy::default(),
            pending_deletes: Arc::new(parking_lot::Mutex::new(None)),
            retired: Arc::new(parking_lot::Mutex::new(std::collections::BTreeMap::new())),
            view_slot: crate::reader_view::ViewSlot::new(),
            view_dirty: std::sync::atomic::AtomicBool::new(false),
            merge_backlog: std::sync::atomic::AtomicUsize::new(BACKLOG_UNKNOWN),
            merging: Default::default(),
            _restored_metadata_memory: None,
        };
        index.sync_vector_field_options();
        index.publish_view()?;
        index.commit()?;
        Ok(index)
    }

    /// Open an existing on-disk index from its commit point.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let directory: Arc<dyn Directory> = Arc::new(crate::codec::directory::MmapDirectory::open(path)?);
        if !directory.exists(META_FILE) {
            return Err(RusticError::IndexNotFound(format!("no commit point at {}", path.display())));
        }
        Self::open_in(directory)
    }

    /// [`Index::open`] from any [`Directory`] holding a commit point (e.g.
    /// an object-store directory).
    pub fn open_in(directory: Arc<dyn Directory>) -> Result<Self> {
        let meta_bytes = directory
            .open_read(META_FILE)
            .map_err(|_| RusticError::IndexNotFound(format!("no commit point in {directory:?}")))?;
        let metadata_memory = Arc::new(crate::admission::shared().reserve((meta_bytes.len() as u64).saturating_mul(32).saturating_add(8192), "restored index metadata")?);
        let meta: Meta = serde_json::from_slice(meta_bytes.as_slice())
            .map_err(|e| RusticError::Corrupted(format!("meta.json: {e}")))?;
        let vs = &meta.vector_store;
        let vectors = VectorStore::open(&*directory, &vs.segments, vs.next_point_id, vs.next_segment_id)?;
        #[cfg(feature = "mutable-hnsw-poc")]
        vectors.restore_mutable_checkpoint(&*directory, &vs.mutable_fields, vs.next_mutable_generation)?;
        // DiskBBQ: the committed indexes; rows past them (written after the
        // checkpoint) are searched flat until maintenance indexes them.
        vectors.open_ivf(&*directory, &vs.ivf_fields)?;
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        if !vs.mutable_fields.is_empty() {
            return Err(RusticError::Unsupported("index requires mutable HNSW backend".into()));
        }
        let mut index = Self {
            name: meta.name,
            schema: meta.schema,
            directory,
            segments: Vec::with_capacity(meta.segments.len()),
            writers: Vec::new(),
            frozen: Vec::new(),
            hidden: Vec::new(),
            parked: Vec::new(),
            sealing: Default::default(),
            auto_seal: true,
            nrt: false,
            pending_dels: std::sync::atomic::AtomicBool::new(false),
            index_sort: None,
            stored_compression: crate::codec::store::StoredCompression::Fast,
            preserve_published_leaves: false,
            next_segment_id: meta.next_segment_id,
            settings: meta.settings,
            vectors,
            optimizer_policy: OptimizerPolicy::default(),
            pending_deletes: Arc::new(parking_lot::Mutex::new(None)),
            retired: Arc::new(parking_lot::Mutex::new(std::collections::BTreeMap::new())),
            view_slot: crate::reader_view::ViewSlot::new(),
            view_dirty: std::sync::atomic::AtomicBool::new(false),
            merge_backlog: std::sync::atomic::AtomicUsize::new(BACKLOG_UNKNOWN),
            merging: Default::default(),
            _restored_metadata_memory: Some(metadata_memory.clone()),
        };
        index.sync_vector_field_options();
        index.view_slot.set_committed(crate::reader_view::ViewSlot::prepare_committed(meta.segments.iter().map(|s| (s.id, u64::from(s.num_docs))))?);
        for seg in &meta.segments {
            index.segments.push(Arc::new(Segment::open_with_store(&*index.directory, seg.id, &index.vectors)?));
        }
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let migrate = index.vectors.validate_mutable_schema(&index.schema)?;
            index.vectors.reconcile_mutable_checkpoint()?;
            index.request_vector_migrations(&migrate)?;
        }
        index.publish_view()?;
        Ok(index)
    }

    /// Delete the files of this index's directory that its commit does not
    /// refer to: segment and deletion files of segments not in it (sealed or
    /// merged after the last commit, then a crash) and `*.tmp` leftovers of
    /// interrupted writes. Vector files are left to the vector store (rows a
    /// graph checkpoint references may live outside its segment list). Only
    /// for the directory's owner right after
    /// [`Index::open`] (no other index or reader uses the directory).
    /// Returns how many files it deleted.
    pub fn sweep_orphan_files(&self) -> Result<usize> {
        let _live_memory=crate::admission::shared().reserve((self.segments.len() as u64).saturating_add(self.hidden.len() as u64).saturating_mul(32).saturating_add(256), "orphan sweep segment lookup")?;
        let live: std::collections::HashSet<u32> = self.segments.iter().chain(&self.hidden).map(|s|s.id).collect();
        let segment_id = |name: &str| -> Option<u32> {
            let stem = name.strip_prefix("seg_")?;
            let stem = stem.strip_suffix(".rstc").or_else(|| stem.strip_suffix(".del"))?;
            stem.parse().ok()
        };
        // DiskBBQ files neither a current index nor the commit point on disk
        // references (a step that wrote them before a crash, or replaced
        // ones). Files of an index that failed to open stay until a commit
        // no longer lists them: pinning and recovering that commit need them.
        let mut _committed_ivf_memory = None;
        let mut ivf_live = self.vectors.ivf_files()?;
        if let Ok(bytes) = self.directory.open_read(META_FILE) {
            // The decode and its derived file-name set coexist until sweep
            // finishes; a denied reservation must not become an empty set.
            _committed_ivf_memory = Some(crate::admission::shared().reserve((bytes.len() as u64).saturating_mul(16).saturating_add(4096), "IVF committed file sweep")?);
            if let Ok(meta) = serde_json::from_slice::<Meta>(bytes.as_slice()) {
                ivf_live.extend(meta.vector_store.ivf_fields.iter().flat_map(|m| m.files()));
                // Code log chunks of the committed graph checkpoints.
                ivf_live.extend(meta.vector_store.mutable_fields.iter().flat_map(|m| m.code_files.iter().cloned()));
            }
        }
        // Code log chunks the open graphs reference or wrote since (a full
        // chunk is written before any commit references it).
        #[cfg(feature = "mutable-hnsw-poc")]
        ivf_live.extend(self.vectors.code_log_files());
        let mut swept = 0;
        // `list` hides unfinished writes on disk; ask for them too (merge
        // scratch files, interrupted streaming outputs and atomic writes).
        let names = self.directory.list_admitted(true)?;
        for name in names.iter() {
            let orphan = if name.ends_with(".tmp") {
                true
            } else if let Some(id) = segment_id(&name) {
                !live.contains(&id)
            } else if crate::vector::ivf_field::is_ivf_file(&name) || crate::vector::store::is_code_log_file(&name) {
                !ivf_live.contains(name)
            } else {
                false
            };
            if orphan {
                self.directory.delete(&name)?;
                swept += 1;
            }
        }
        Ok(swept)
    }

    /// The slot this index publishes its reader views to.
    pub fn view_slot(&self) -> &Arc<crate::reader_view::ViewSlot> {
        &self.view_slot
    }

    /// The current reader view (see [`crate::reader_view`]).
    pub fn reader_view(&self) -> Arc<crate::reader_view::ReaderView> {
        self.view_slot.load()
    }

    /// Publish this index's views to `slot` from now on (an owner that
    /// replaces its index keeps one slot), and publish the current one.
    pub fn adopt_view_slot(&mut self, slot: Arc<crate::reader_view::ViewSlot>) -> Result<()> {
        self.view_slot = slot;
        if self.nrt {
            self.vectors.set_view_clock(Some(&self.view_slot));
        }
        self.publish_view()
    }

    /// Build the reader view of what searches see now (searchable segments
    /// as frozen-deletion snapshots, then published memtables) and install
    /// it. Called by every operation that changes the searchable state.
    fn publish_view(&self) -> Result<()> {
        self.publish_view_with_budget(&crate::admission::shared())
    }

    fn publish_view_with_budget(&self, budget: &Arc<crate::admission::Budget>) -> Result<()> {
        self.view_dirty.store(true, std::sync::atomic::Ordering::Release);
        let memory = budget.reserve(self.view_bytes(), "published reader view")?;
        self.publish_view_admitted(memory);
        Ok(())
    }

    /// Admission a reader view of the current state retains. Never grows
    /// when a merge replaces segments, so a merge admits it before its swap.
    fn view_bytes(&self) -> u64 {
        crate::reader_view::ViewSlot::view_container_bytes(self.segments.len(), self.frozen.len(), self.hidden.len() + self.parked.len())
            .saturating_add(self.frozen.iter().map(|w| crate::postings::BitSet::storage_bytes(w.num_docs())).fold(0, u64::saturating_add))
    }

    /// Publish the view within `memory`, admitted for at least
    /// [`Self::view_bytes`]; returns any excess.
    fn publish_view_admitted(&self, mut memory: crate::admission::Reservation) {
        self.view_dirty.store(true, std::sync::atomic::Ordering::Release);
        let bytes = self.view_bytes();
        debug_assert!(memory.bytes() >= bytes, "reader view admitted for a smaller state");
        memory.shrink(memory.bytes().saturating_sub(bytes));
        let memory = Arc::new(memory);
        let segments = self.segments.iter().map(|s| Arc::new(s.reader_snapshot())).collect();
        let frozen = self.frozen.iter().map(|w| (w.clone(), Arc::new(w.deletions().with_memory(memory.clone())))).collect();
        let pending = self.hidden.iter().map(|s| (s.id, u64::from(s.num_docs), s.segment_bytes()))
            .chain(self.parked.iter().map(|w| (w.id, u64::from(w.num_docs()), 0))).collect();
        self.view_slot.publish_admitted(segments, frozen, pending, memory);
        self.view_dirty.store(false, std::sync::atomic::Ordering::Release);
        // Large newly searchable segments (a merge's output): their mutable
        // node maps are built in the background, not by their first query.
        #[cfg(feature = "mutable-hnsw-poc")]
        self.vectors.warm_mutable_projections(self.segments.iter().filter(|s| s.num_docs >= WARM_PROJECTION_DOCS).cloned().collect());
    }

    /// Hand a merged-away segment's files to the deleter: they go at the
    /// first commit after no reader holds the segment any more.
    fn prepare_retired(segments: &[Arc<Segment>]) -> Result<Vec<(u32, RetiredSegment)>> {
        Self::prepare_retired_with_budget(segments, &crate::admission::shared())
    }

    fn prepare_retired_with_budget(segments: &[Arc<Segment>], budget: &Arc<crate::admission::Budget>) -> Result<Vec<(u32, RetiredSegment)>> {
        // Includes exact two filename allocations, staging tuple and a
        // conservative BTree node share; no retained Vec capacity after removal.
        let memory = Arc::new(budget.reserve((segments.len() as u64).saturating_mul(512), "retired reader file registry")?);
        Ok(segments.iter().map(|segment| (segment.id, RetiredSegment {
            names: [crate::codec::segment_file_name(segment.id), crate::codec::deletes_file_name(segment.id)],
            bytes: segment.segment_bytes(),
            data: segment.data_handle(), _memory: memory.clone(),
        })).collect())
    }

    fn retire_prepared(&self, segments: Vec<(u32, RetiredSegment)>) {
        self.retired.lock().extend(segments);
    }

    /// Delete the files of merged-away segments that no reader holds and
    /// the last commit does not list, without waiting for the next commit
    /// (Lucene's `IndexFileDeleter` drops a file once neither a commit nor
    /// a reader references it). Files the last commit lists stay until the
    /// next commit. Safe at any time: an in-flight commit holds the
    /// segments it will list. Returns how many segments' files went.
    ///
    /// Returned as a job to run without the index lock (deletes may be slow).
    pub fn retired_sweep(&self) -> impl FnOnce() -> Result<usize> + Send + 'static {
        let (retired, directory, slot) = (self.retired.clone(), self.directory.clone(), self.view_slot.clone());
        move || {
            let (n, error) = delete_dead_retired(&retired, &*directory, |id| slot.is_committed(id));
            match error { Some(e) => Err(e), None => Ok(n) }
        }
    }

    /// Bytes of segments the last commit lists that merges have replaced
    /// since: they stay on disk until the next commit (ES flushes after
    /// merges for this, `maybeFlushAfterMerge`).
    pub fn merged_away_committed_bytes(&self) -> u64 {
        let slot = &self.view_slot;
        self.retired.lock().iter().filter(|(id, _)| slot.is_committed(**id)).map(|(_, r)| r.bytes).sum()
    }

    /// The directory this index lives in.
    pub fn directory(&self) -> &Arc<dyn Directory> {
        &self.directory
    }

    /// Write segment files, deletion bitmaps and the `meta.json` commit
    /// point. Safe to call after any mutation; the previous commit stays
    /// valid until the rename swaps in the new one.
    pub fn commit(&self) -> Result<()> {
        self.prepare_commit()?.run()
    }

    /// Whether publication only changes visibility, without encoding or
    /// replacing the segments pinned by a detached commit.
    pub fn can_refresh_during_commit(&self) -> bool {
        !self.auto_seal && (self.index_sort.is_none() ||
            (self.frozen.is_empty() && self.parked.is_empty() && self.writers.iter().all(|w| w.num_docs() == 0)))
    }

    /// Capture a durable checkpoint for I/O outside the owner's core lock.
    /// See [`CommitJob`] for the required external serialization contract.
    pub fn prepare_commit(&self) -> Result<CommitJob> {
        let bytes = self.schema.iter().fold(self.name.len() as u64, |n, field| n
            .saturating_add(std::mem::size_of::<Field>() as u64).saturating_add(field.name.len() as u64)
            .saturating_add(field.analyzer.len() as u64).saturating_add(field.similarity.len() as u64))
            .saturating_add(((self.segments.len() + self.hidden.len()) as u64).saturating_mul(8)).saturating_add(1024);
        let memory = crate::admission::shared().reserve(bytes, "captured commit metadata")?;
        Ok(CommitJob {
            admission: crate::admission::Context::capture(),
            view_slot: self.view_slot.clone(),
            name: self.name.clone(), schema: self.schema.clone(), settings: self.settings.clone(),
            segments: self.segments.clone(), hidden: self.hidden.clone(),
            next_segment_id: self.next_segment_id, directory: self.directory.clone(),
            vectors: self.vectors.clone(), pending_deletes: self.pending_deletes.clone(),
            retired: self.retired.clone(), deletions_at: None,
            #[cfg(feature = "mutable-hnsw-poc")]
            captured: None,
            _memory: memory,
            capture_memory: None,
            deferred_capture: None,
        })
    }

    /// [`Index::prepare_commit`] that also captures the mutable graph
    /// checkpoints now (graph and rows owned by the job), so the caller's
    /// write fence is needed only up to here: [`CommitJob::run`] then
    /// streams them, with the deletion files and `meta.json`, without it.
    pub fn prepare_commit_captured(&self) -> Result<CommitJob> {
        #[allow(unused_mut)]
        let mut job = self.prepare_commit()?;
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let _commit = self.vectors.mutable_commit.lock();
            job.capture_memory = Some(self.vectors.checkpoint_registry_memory()?);
            job.captured = Some(self.vectors.capture_mutable_checkpoint()?);
        }
        Ok(job)
    }

    /// The capture point of the mutable-graph checkpoints for a commit,
    /// taken under the owner's write fence with no ingestion in flight
    /// (cheap: one short graph-writer hold per field). The graphs as of
    /// this point are copied later by [`VectorCapture::finish`] off every
    /// lock, while writes continue (they land after the capture point and
    /// are replayed from the translog on recovery); then
    /// [`Index::prepare_commit_with`]. The capture must be taken with the
    /// memtables of the commit closed (`begin_seal`), so the checkpoint
    /// covers exactly the commit's documents.
    ///
    /// `save_graphs == false` (graph saves scheduled apart from text
    /// commits): fields keep the graph checkpoint of the previous commit
    /// (only a field without one is captured); reopening catches the graph
    /// up with the documents committed since
    /// (`MutableIngestField::reconcile_committed_points`).
    pub fn begin_vector_capture(&self, save_graphs: bool) -> Result<VectorCapture> {
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let commit = self.vectors.mutable_commit.lock();
            let memory = self.vectors.checkpoint_registry_memory()?;
            Ok(VectorCapture { admission: crate::admission::Context::capture(), pending: self.vectors.begin_mutable_checkpoint(&commit.checkpointed, save_graphs)?, done: None, _memory: Some(memory) })
        }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        {
            let _ = save_graphs;
            Ok(VectorCapture { admission: crate::admission::Context::capture(), _memory: None })
        }
    }

    /// Some mutable vector field has no graph checkpoint in any commit yet:
    /// its first [`Index::begin_vector_capture`] needs graph ingestion
    /// quiesced (no reservation in flight).
    pub fn has_uncheckpointed_mutable_fields(&self) -> bool {
        #[cfg(feature = "mutable-hnsw-poc")]
        { self.vectors.has_uncheckpointed_mutable_fields() }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        { false }
    }

    /// [`Index::prepare_commit`] with checkpoints from a finished
    /// [`Index::begin_vector_capture`].
    pub fn prepare_commit_with(&self, capture: VectorCapture) -> Result<CommitJob> {
        #[allow(unused_mut)]
        let mut job = self.prepare_commit()?;
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let done = capture.done.ok_or_else(|| RusticError::Segment("vector capture not finished".into()))?;
            job.captured = Some(done);
            job.capture_memory = capture._memory;
        }
        #[cfg(not(feature = "mutable-hnsw-poc"))]
        let _ = capture;
        Ok(job)
    }

    /// [`Index::prepare_commit_with`] from a capture not yet finished: the
    /// job copies the graphs as of the capture point at the start of
    /// [`CommitJob::run`]. The commit's segment set is fixed here, so the
    /// caller can let merges install while the job copies and writes the
    /// graphs (O(graph) work and I/O) instead of excluding them throughout.
    pub fn prepare_commit_deferred(&self, capture: VectorCapture) -> Result<CommitJob> {
        let mut job = self.prepare_commit()?;
        job.deferred_capture = Some(capture);
        Ok(job)
    }

    /// Enable one mutable graph for each explicitly mapped f32 HNSW field.
    /// Call under the shard write fence. Existing segmented data requires an
    /// explicit migration; never silently switch it to an empty graph.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn enable_mutable_vectors(&self) -> Result<()> {
        self.activate_mutable_vectors()
    }

    /// Activate empty mutable fields. Existing legacy graph rows require
    /// explicit reindexing; they cannot silently select a segmented fallback.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn enable_new_mutable_vector_fields(&self) -> Result<()> {
        self.activate_mutable_vectors()
    }

    /// Whether every graph vector field already has its mutable graph (or a
    /// scheduled migration that will build one): then
    /// [`Self::enable_new_mutable_vector_fields`] has nothing to do, and a
    /// write need not take the index exclusively to call it.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn mutable_vector_fields_ready(&self) -> bool {
        self.schema.iter().all(|field| {
            !matches!(field.field_type, FieldType::Vector { .. })
                || !field.vector_options.unwrap_or_default().graph
                || self.vectors.registered_mutable_field(field.id).is_some()
                || self.vectors.migration_pending(field.id)
        })
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn validate_vector_backend_options(schema: &[Field]) -> Result<()> {
        for field in schema {
            if !matches!(field.field_type, FieldType::Vector { .. }) { continue; }
            let options = field.vector_options.unwrap_or_default();
            // Packed (`bit` / `byte`) rows are only read by their own kernels.
            if crate::codec::vectors::similarity_by_name(&field.similarity).is_some_and(|s| s.is_packed())
                && (options.quantization != crate::vector::Quantization::None || options.ivf.is_some() || options.gpu) {
                return Err(RusticError::InvalidQuery(format!("packed vector field [{}] must be unquantized, without IVF or GPU builds", field.name)));
            }
            if options.graph { crate::vector::es_quant::EsQuant::for_mutable(&options)?; }
        }
        Ok(())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn activate_mutable_vectors(&self) -> Result<()> {
        Self::validate_vector_backend_options(&self.schema)?;
        // A new field's graph is built with its mapped options (the owner
        // reads them from the store): push them first.
        self.sync_vector_field_options();
        for field in &self.schema {
            let FieldType::Vector { dim } = field.field_type else { continue };
            let options = field.vector_options.unwrap_or_default();
            if !options.graph { continue; }
            if self.vectors.registered_mutable_field(field.id).is_some() { continue; }
            let (similarity, params) = Self::graph_config(field, options)?;
            if !self.vectors.segments(field.id).is_empty() || self.memtables_hold_vectors(field.id) {
                // Rows stored before the field had a graph (`flat` mapped
                // to `hnsw`, or a store from before mutable graphs): the
                // graph is built from them in the background, as ES indexes
                // the existing data in the new format; exact search serves
                // the field until then.
                if !self.vectors.migration_pending(field.id) {
                    self.vectors.request_migration(field.id, u32::from(dim), similarity, params, true);
                }
                continue;
            }
            self.vectors.mutable_field(field.id, u32::from(dim), similarity, params)?;
        }
        Ok(())
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn graph_config(field: &Field, options: crate::vector::store::VectorFieldOptions) -> Result<(crate::vector::Similarity, crate::vector::HnswParams)> {
        use crate::vector::HnswParams;
        let similarity = crate::codec::vectors::similarity_by_name(&field.similarity)
            .ok_or_else(|| RusticError::InvalidQuery("unknown mutable vector similarity".into()))?;
        let defaults = HnswParams::default();
        Ok((similarity, HnswParams { m: options.m.unwrap_or(defaults.m), ef_construction: options.ef_construction.unwrap_or(defaults.ef_construction), threads: 0, ..defaults }))
    }

    #[cfg(feature = "mutable-hnsw-poc")]
    fn memtables_hold_vectors(&self, field: u16) -> bool {
        self.all_leaves().iter().any(|leaf| matches!(leaf, LeafReader::Memtable(writer, ..) if writer.has_vector_field(field)))
    }

    /// Validate an ES mapping change (vector backend options) without
    /// changing refresh visibility. Option changes of fields holding data
    /// are scheduled by [`Index::request_vector_migrations`].
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn prepare_vector_schema_change(&mut self, schema: &[Field]) -> Result<()> {
        Self::validate_vector_backend_options(schema)
    }

    /// Vector fields whose graph must be rebuilt (or dropped) for
    /// `schema`: a registered graph whose options, quantization or backend
    /// change (ES `dense_vector` `index_options` updates: `int8_hnsw` →
    /// `int4_hnsw` → `bbq_hnsw`, new `m` / `ef_construction`, `int8_hnsw` →
    /// `bbq_flat`), or a graph field whose rows predate its graph (`flat` →
    /// `hnsw`). Call with the schema about to be applied.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn vector_fields_to_migrate(&self, schema: &[Field]) -> Vec<u16> {
        let mut out = self.vectors.incompatible_mutable_fields(schema);
        // A DiskBBQ index whose field changes type or code parameters.
        for f in schema {
            let FieldType::Vector { dim } = f.field_type else { continue };
            let Some(sim) = crate::codec::vectors::similarity_by_name(&f.similarity) else { continue };
            if !out.contains(&f.id) && self.vectors.ivf_field_stale(f.id, sim, u32::from(dim), &f.vector_options.unwrap_or_default()) {
                out.push(f.id);
            }
        }
        for f in schema {
            if !matches!(f.field_type, FieldType::Vector { .. }) || !f.vector_options.is_some_and(|o| o.graph) { continue; }
            if out.contains(&f.id) || self.vectors.registered_mutable_field(f.id).is_some() { continue; }
            if !self.vectors.segments(f.id).is_empty() || self.memtables_hold_vectors(f.id) { out.push(f.id); }
        }
        out
    }

    /// Schedule `fields`' graphs to be rebuilt for the current schema
    /// (ES applies a `dense_vector` `index_options` update to the data a
    /// field already holds). Nothing is built here: the background vector
    /// maintenance builds each new graph off every lock and swaps it in
    /// ([`VectorStore::run_migrations`]); until then the current graph keeps
    /// serving searches and admitting rows. Call after updating
    /// [`Index::schema`].
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn request_vector_migrations(&self, fields: &[u16]) -> Result<()> {
        for &id in fields {
            let Some(field) = self.schema.iter().find(|f| f.id == id) else { continue };
            let FieldType::Vector { dim } = field.field_type else { continue };
            let options = field.vector_options.unwrap_or_default();
            let (similarity, params) = Self::graph_config(field, options)?;
            self.vectors.request_migration(id, u32::from(dim), similarity, params, options.graph);
        }
        Ok(())
    }

    /// Run the scheduled migrations now (what the background maintenance
    /// does); returns how many completed.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn complete_vector_migrations(&self) -> Result<usize> {
        self.vectors.run_migrations()
    }

    /// The index's vector store.
    pub fn vector_store(&self) -> &Arc<VectorStore> {
        &self.vectors
    }

    /// Run the vector optimizers (indexing / vacuum / merge) once with
    /// [`Index::optimizer_policy`]. The only place HNSW graphs are built;
    /// safe to call from a background thread between commits (`&self`).
    /// Replaced vector segment files are deleted by the next `commit()`.
    pub fn optimize_vectors(&self) -> Result<OptimizeReport> {
        self.vector_optimizer().run()
    }

    /// An owned [`Index::optimize_vectors`] job, so a caller holding the
    /// index behind a lock can release it before the (possibly long) graph
    /// builds: the store swaps segments under its own locks, searches keep
    /// running on the old segments, and [`Index::commit`] reads the vector
    /// commit point atomically against the swap.
    pub fn vector_optimizer(&self) -> VectorOptimizer {
        self.sync_vector_field_options();
        VectorOptimizer { admission: crate::admission::Context::capture(), store: self.vectors.clone(), directory: self.directory.clone(), policy: self.optimizer_policy.clone() }
    }

    /// Vector side of ES `_forcemerge`: seal every appendable vector chunk
    /// and merge each vector field down to at most `max_segments` sealed
    /// segments ([`VectorStore::force_merge`]); with `only_expunge_deletes`,
    /// rebuild the sealed segments that hold deleted points instead
    /// ([`VectorStore::expunge_deletes`]). Text segments are untouched (see
    /// [`Index::force_merge`]). `&self`: searches keep running on the old
    /// segments until each swap. Replaced files go at the next `commit()`.
    pub fn force_merge_vectors(&self, max_segments: usize, only_expunge_deletes: bool) -> Result<OptimizeReport> {
        self.vector_optimizer().force_merge(max_segments, only_expunge_deletes)
    }

    /// Push every vector field's [`Field::vector_options`] into the vector
    /// store (per-field quantization / graph / rescoring policy). Called
    /// at open and before each optimizer run; call it after changing a
    /// field's options in [`Index::schema`] so searches see them at once.
    pub fn sync_vector_field_options(&self) {
        for f in &self.schema {
            if let (FieldType::Vector { .. }, Some(o)) = (&f.field_type, f.vector_options) {
                if self.vectors.field_options(f.id) != o {
                    self.vectors.set_field_options(f.id, o);
                }
            }
        }
    }

    /// [`Index::sync_vector_field_options`], also for the vectors still in
    /// memtables (searched with the field's current policy).
    pub fn sync_all_vector_field_options(&mut self) {
        self.sync_vector_field_options();
        let opts: Vec<(u16, Option<crate::vector::store::VectorFieldOptions>)> =
            self.schema.iter().filter(|f| matches!(f.field_type, FieldType::Vector { .. })).map(|f| (f.id, f.vector_options)).collect();
        for w in self.writers.iter_mut() {
            for (id, o) in &opts {
                w.set_vector_options(*id, *o);
            }
        }
        // Closed memtables shared with a flush keep their options (only a
        // memtable with vectors is ever sealed in the foreground, and those
        // are not shared).
        for w in self.frozen.iter_mut().chain(self.parked.iter_mut()) {
            if let Some(w) = Arc::get_mut(w) {
                for (id, o) in &opts {
                    w.set_vector_options(*id, *o);
                }
            }
        }
    }

    /// The vector store's registered policy of `field`.
    pub fn vector_field_options(&self, field: u16) -> crate::vector::store::VectorFieldOptions {
        self.vectors.field_options(field)
    }

    /// Merge the sealed segments with the given ids into one (dropping
    /// deleted docs). Vector-wise this only rewrites the merged segment's
    /// `doc -> PointId` column and moves point ownership: no vector is
    /// copied and no HNSW graph is touched. Returns the new segment id, or
    /// `None` if no listed segment exists. Old files go at the next commit.
    pub fn merge_segments(&mut self, ids: &[u32]) -> Result<Option<u32>> {
        let chosen: Vec<Arc<Segment>> = self.segments.iter().filter(|s| ids.contains(&s.id)).cloned().collect();
        let Some(pos) = self.segments.iter().position(|s| ids.contains(&s.id)) else {
            return Ok(None);
        };
        let new_id = self.next_segment_id;
        self.next_segment_id += 1;
        self.renumber_writers();
        let order = self.sorted_order(&chosen)?;
        let (order,_memory)=match order{Some(order)=>{let (v,m)=order.into_parts();(Some(v),Some(m))},None=>(None,None)};
        let merged = match self.merge_into(&chosen, new_id, order, false) {
            Ok(merged) => merged,
            Err(e) => {
                // Nothing references the output yet (an open refused for
                // memory, a failed write): don't leave it on disk.
                let _ = self.directory.delete(&crate::codec::segment_file_name(new_id));
                return Err(e);
            }
        };
        let retired = Self::prepare_retired(&chosen)?;
        self.segments.retain(|s| !ids.contains(&s.id));
        self.segments.insert(pos, Arc::new(merged));
        self.retire_prepared(retired);
        self.publish_view()?;
        Ok(Some(new_id))
    }

    /// Merge every sealed segment into one.
    pub fn force_merge(&mut self) -> Result<()> {
        if self.segments.len() > 1 {
            let ids: Vec<u32> = self.segments.iter().map(|s| s.id).collect();
            self.merge_segments(&ids)?;
        }
        Ok(())
    }

    /// Global k-NN over every leaf: the vector store (all flushed points,
    /// fanned out over vector segments in parallel) plus the memtable
    /// (exact). With `req.filter`, the filter query is evaluated per leaf
    /// into a doc bitset; broad filters go to the store as those bitsets
    /// (a point's owner doc is checked only when the graph visits it),
    /// selective ones are mapped to an explicit point set.
    /// Hits are score-descending; `leaf` indexes [`Index::leaves`].
    pub fn knn_search(&self, req: &KnnRequest) -> Result<Vec<KnnHit>> {
        Ok(self.knn_search_with(req, &PlannerConfig::default())?.0)
    }

    /// [`Index::knn_search`] with an explicit planner config, also returning
    /// how the planner executed.
    pub fn knn_search_with(&self, req: &KnnRequest, cfg: &PlannerConfig) -> Result<(Vec<KnnHit>, SearchStats)> {
        self.knn_search_in(req, cfg, None)
    }

    /// [`Index::knn_search_with`] evaluating a numeric-range `req.filter`
    /// through `filter_ctx` (a context built over the same leaves for
    /// another query of the request), so its per-leaf bitset is shared
    /// with that query instead of being built twice.
    fn knn_search_in(
        &self,
        req: &KnnRequest,
        cfg: &PlannerConfig,
        filter_ctx: Option<&crate::query::query::SearchContext>,
    ) -> Result<(Vec<KnnHit>, SearchStats)> {
        use crate::vector::store::{DocFilter, OrdFilter, PointFilter};
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static T_BITS: AtomicU64 = AtomicU64::new(0);
        static T_ORD: AtomicU64 = AtomicU64::new(0);
        static T_SEARCH: AtomicU64 = AtomicU64::new(0);
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let trace = std::env::var_os("RUSTIC_KNN_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let leaves = self.leaves();
        #[cfg(feature = "mutable-hnsw-poc")]
        {
            let context = crate::query::query::SearchContext::for_query(
                req.filter.as_ref().unwrap_or(&crate::query::query::Query::MatchAll), &leaves, Bm25Params::default());
            if let Some(error) = context.execution_error.lock().clone() { return Err(RusticError::InvalidQuery(error)); }
            if let Some((hits, stats)) = crate::query::mutable_knn::search(&leaves, req.field, &req.vector,
                req.k, req.num_candidates.max(req.k), req.filter.as_ref(), &context,
                cfg.full_scan_threshold == usize::MAX, cfg.oversample)? {
                let hits = hits.into_iter().filter(|(score,_,_)| req.similarity_threshold.is_none_or(|limit| *score >= limit))
                    .map(|(score,leaf,doc)| KnnHit { leaf, doc, score: score * req.boost }).collect();
                return Ok((hits, stats));
            }
        }
        let k = req.k;
        let mut doc_filter: Option<DocFilter> = None;
        let mut filter_memory=Vec::new();
        // Per-leaf filter bits for memtable leaves (indexed by leaf).
        let mut mem_filters: Vec<Option<Vec<u64>>> = vec![None; leaves.len()];
        if let Some(f) = &req.filter {
            let own_ctx;
            let ctx = match filter_ctx {
                Some(c) if f.is_numeric_range() => c,
                _ => {
                    own_ctx = crate::query::query::SearchContext::for_query(f, &leaves, Bm25Params::default());
                    &own_ctx
                }
            };
            let mut df = DocFilter::new();
            for (li, leaf) in leaves.iter().enumerate() {
                // Numeric ranges come straight from the points index as a
                // bitset (no doc-by-doc scorer walk).
                let (mut words,memory)=f.filter_bits(leaf,ctx)?.into_parts();
                filter_memory.push(memory);
                if let Some(dels) = leaf.deleted_bits() {
                    for (w, d) in words.iter_mut().zip(dels.words()) {
                        *w &= !d;
                    }
                }
                match leaf {
                    LeafReader::Segment(seg) => df.add_segment(seg.id, words),
                    LeafReader::Memtable(_, ..) => mem_filters[li] = Some(words),
                }
            }
            doc_filter = Some(df);
        }
        let t1 = std::time::Instant::now();
        let ef = req.num_candidates.max(k);
        // Selective filter (the planner searches exactly, or traverses the
        // graph two-hop, both checking the filter per candidate): resolve
        // the matching docs to per-segment point ordinals once, so each
        // check is one bit test and exact search enumerates just those.
        // Broad filters stay doc bitsets (checked only for visited nodes).
        let selective = doc_filter.as_ref().is_some_and(|df| {
            let total: usize = self.vectors.segments(req.field).iter().map(|s| s.len() as usize).sum();
            df.len() <= cfg.full_scan_threshold || (df.len() as f32) < cfg.two_hop_fraction * total as f32
                || ((df.len() as f32) < 0.6 * total as f32 && self.segments.iter().all(|s| s.point_identity_base(req.field).is_some()))
        });
        let ord_filter: Option<OrdFilter> = if selective {
            doc_filter.take().map(|df| {
                self.vectors.ord_filter(req.field, df, |sid| {
                    let seg = self.segments.iter().find(|s| s.id == sid)?;
                    Some((seg.point_map().column(req.field)?, seg.point_identity_base(req.field)))
                })
            })
        } else {
            None
        };
        let pf = match (&ord_filter, &doc_filter) {
            (Some(o), _) => Some(PointFilter::Ords(o)),
            (None, Some(d)) => Some(PointFilter::Docs(d)),
            (None, None) => None,
        };
        let t2 = std::time::Instant::now();
        let (hits, stats) = self.vectors.search_filtered(req.field, &req.vector, k, ef, pf, cfg)?;
        if trace {
            let t3 = std::time::Instant::now();
            T_BITS.fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
            T_ORD.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
            T_SEARCH.fetch_add((t3 - t2).as_nanos() as u64, Relaxed);
            let c = CALLS.fetch_add(1, Relaxed) + 1;
            let every = std::env::var("RUSTIC_KNN_TRACE_EVERY").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&v| v > 0).unwrap_or(500);
            if c % every == 0 {
                eprintln!("[knn-time] calls={c} filtered={} bits={}us ord={}us search={}us ords={} stats={stats:?}", req.filter.is_some(), T_BITS.swap(0, Relaxed) / every / 1000, T_ORD.swap(0, Relaxed) / every / 1000, T_SEARCH.swap(0, Relaxed) / every / 1000, ord_filter.as_ref().map(|o| o.len()).unwrap_or(0));
            }
        }
        let leaf_of: HashMap<u32, usize> = self.segments.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
        let mut out: Vec<KnnHit> = hits
            .into_iter()
            .filter_map(|h| leaf_of.get(&h.segment).map(|&leaf| KnnHit { leaf, doc: h.doc, score: h.score }))
            .collect();
        for (leaf, l) in leaves.iter().enumerate() {
            let LeafReader::Memtable(w, ..) = l else { continue };
            let mem_filter = mem_filters.get(leaf).and_then(|f| f.as_ref());
            let accept = |d: DocId| {
                !w.is_deleted(d) && mem_filter.is_none_or(|b| b.get(d as usize / 64).is_some_and(|x| x >> (d % 64) & 1 == 1))
            };
            if doc_filter.is_none() || mem_filter.is_some() {
                out.extend(w.knn_with(req.field, &req.vector, k, accept, cfg).into_iter().map(|(doc, score)| KnnHit { leaf, doc, score }));
            }
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)));
        out.truncate(k);
        out.retain(|h| req.similarity_threshold.is_none_or(|t| h.score >= t));
        for h in &mut out {
            h.score *= req.boost;
        }
        Ok((out, stats))
    }

    /// Hybrid search with ES `bool.should` semantics: a doc's score is its
    /// `text` score (if it matches) plus the score of every `knn` request
    /// whose global top-k contains it. Returns the best `top_k`.
    pub fn hybrid_search(&self, text: Option<&crate::query::query::Query>, knn: &[KnnRequest], top_k: usize) -> Result<Vec<KnnHit>> {
        let opts = crate::query::searcher::SearchOptions { top_k, ..Default::default() };
        self.hybrid_search_with(text, knn, &opts)
    }

    /// [`Index::hybrid_search`] with explicit search options for the text
    /// leg (`top_k` = size, `from`, `track_total_hits`, similarity,
    /// `parallel`, ...). Returns hits `from..from + top_k` of the fused
    /// ranking.
    ///
    /// The text query's [`SearchContext`](crate::query::query::SearchContext)
    /// is built once and reused to score the k-NN docs that are not in the
    /// text top-k: one scorer per leaf, advanced through those docs in
    /// ascending order (Lucene scores `bool.should(text, knn)` in one pass;
    /// this is the same work restricted to the k-NN docs).
    pub fn hybrid_search_with(
        &self,
        text: Option<&crate::query::query::Query>,
        knn: &[KnnRequest],
        opts: &crate::query::searcher::SearchOptions,
    ) -> Result<Vec<KnnHit>> {
        use crate::query::docset::TERMINATED;
        use crate::query::searcher::{SearchOptions, Searcher};
        let need = opts.from.saturating_add(opts.top_k);
        let leaves = self.leaves();
        // (leaf, doc) -> (text score, k-NN score sum).
        let mut acc: HashMap<(usize, DocId), (f32, f32)> = HashMap::new();
        let searcher = Searcher::new(leaves.clone());
        let ctx = text.map(|q| {
            let text_opts = SearchOptions { top_k: need, from: 0, ..opts.clone() };
            let ctx = searcher.context(q, &text_opts);
            for h in searcher.search_with_context(q, &text_opts, &ctx).hits {
                acc.insert((h.leaf, h.doc), (h.score, 0.0));
            }
            ctx
        });
        if let Some(error) = ctx.as_ref().and_then(|ctx| ctx.execution_error.lock().clone()) {
            return Err(RusticError::InvalidQuery(error));
        }
        // k-NN docs not in the text top-k still get their text score.
        let mut missing: Vec<(usize, DocId)> = Vec::new();
        let planner = PlannerConfig::default();
        for req in knn {
            for h in self.knn_search_in(req, &planner, ctx.as_ref())?.0 {
                let key = (h.leaf, h.doc);
                match acc.get_mut(&key) {
                    Some(e) => e.1 += h.score,
                    None => {
                        acc.insert(key, (0.0, h.score));
                        missing.push(key);
                    }
                }
            }
        }
        if let (Some(q), Some(ctx)) = (text, &ctx) {
            missing.sort_unstable();
            let mut i = 0;
            while i < missing.len() {
                let leaf = missing[i].0;
                let j = i + missing[i..].partition_point(|&(l, _)| l == leaf);
                if let Some(mut s) = q.scorer(&leaves[leaf], ctx) {
                    for &(_, doc) in &missing[i..j] {
                        let cur = s.doc();
                        let at = if cur != TERMINATED && cur >= doc { cur } else { s.seek(doc) };
                        if at == TERMINATED {
                            break;
                        }
                        if at == doc {
                            if let Some(e) = acc.get_mut(&(leaf, doc)) {
                                e.0 = s.score();
                            }
                        }
                    }
                }
                i = j;
            }
        }
        let mut out: Vec<KnnHit> = acc.into_iter().map(|((leaf, doc), (t, k))| KnnHit { leaf, doc, score: t + k }).collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)));
        out.truncate(need);
        out.drain(..opts.from.min(out.len()));
        Ok(out)
    }

    /// Get a field by name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.schema.iter().find(|f| f.name == name)
    }

    /// Get a field ID by name.
    pub fn field_id(&self, name: &str) -> Option<u16> {
        self.schema.iter().find(|f| f.name == name).map(|f| f.id)
    }

    /// Give memtable `i` the segment id it will be flushed as.
    fn renumber_writers(&mut self) {
        for (i, w) in self.writers.iter_mut().enumerate() {
            w.id = self.next_segment_id + i as u32;
        }
    }

    /// Ensure at least `n` memtables exist.
    fn ensure_writers(&mut self, n: usize) {
        while self.writers.len() < n {
            let id = self.next_segment_id + self.writers.len() as u32;
            let mut w = SegmentWriter::new(id);
            w.set_stored_compression(self.stored_compression);
            self.writers.push(w);
        }
    }

    /// Leaf index (in [`Index::all_leaves`] order) of the first non-empty
    /// memtable being written to.
    fn writers_base(&self) -> u32 {
        (self.segments.len() + self.frozen.len() + self.hidden.len() + self.parked.len()) as u32
    }

    /// Add a document to the index. A document with nested children
    /// (`doc.nested`) is written as one contiguous block, children first;
    /// the returned address is the parent's (leaf index as in
    /// [`Index::all_leaves`]).
    pub fn add_document(&mut self, doc: &Document) -> Result<DocAddress> {
        self.replace_document(None, doc)
    }

    /// Called before sequence/WAL mutation by shard paths that later install
    /// an already prepared vector document under the same core write fence.
    pub fn prepare_document_bitmap(&mut self, doc: &Document) -> Result<()> {
        self.ensure_writers(1);
        self.writers[0].prepare_bitmap_capacity(SegmentWriter::document_slots(doc)?)
    }

    /// Admit replacement graph rows before deleting the previous document.
    /// The caller retains the shard write/WAL fence across this operation.
    pub fn replace_document(&mut self, previous: Option<&DocAddress>, doc: &Document) -> Result<DocAddress> {
        self.prepare_document_bitmap(doc)?;
        #[cfg(feature = "mutable-hnsw-poc")]
        let prepared = {
            let mut points = BTreeMap::new();
            if let Some(previous) = previous {
                let leaves = self.all_leaves();
                if let Some(leaf) = leaves.get(previous.segment_id as usize) {
                    let start = leaf.nested_info().map_or(previous.doc_id, |nested| nested.block_start(previous.doc_id));
                    if previous.doc_id - start + 1 == doc.block_len() as u32 {
                        for field in self.schema.iter().filter(|field| matches!(field.field_type, FieldType::Vector { .. })) {
                            for offset in 0..doc.block_len() as u32 {
                                let point = match leaf {
                                    LeafReader::Segment(segment) => segment.point(field.id, start + offset),
                                    LeafReader::Memtable(writer, ..) => writer.reserved_vector_point(field.id, start + offset),
                                };
                                if let Some(point) = point { points.insert((field.id, offset), point); }
                            }
                        }
                    }
                }
            }
            self.writers[0].prepare_document_vectors_reusing(doc, &self.schema, &self.vectors, &points)?
        };
        if let Some(previous) = previous { self.delete_doc(previous)?; }
        let base = self.writers_base();
        let writer = &mut self.writers[0];
        writer.add_document(doc, &self.schema);
        #[cfg(feature = "mutable-hnsw-poc")]
        writer.accept_document_vectors(prepared);
        let doc_id = writer.num_docs() as DocId - 1;
        Ok(DocAddress::new(base, doc_id))
    }

    /// Install an already admitted fresh document under the caller's write
    /// fence. Preparation may run concurrently; publication stays ordered.
    #[cfg(feature = "mutable-hnsw-poc")]
    pub fn add_prepared_document(&mut self, doc: &Document, prepared: crate::segment::PreparedDocumentVectors) {
        self.ensure_writers(1);
        self.writers[0].append_prepared_document(doc, &self.schema, prepared);
    }

    /// Take a memtable out for lock-free indexing (Lucene's per-thread
    /// `DocumentsWriterPerThread`): the caller fills it without holding the
    /// index lock and gives it back with [`Index::checkin_writer`]. Until
    /// then its documents are invisible, realtime reads included, so it
    /// must only take documents nothing can reference yet (new,
    /// auto-generated ids). Returns the memtable and the schema to index
    /// with (fields added meanwhile reach the memtable at check-in).
    ///
    /// The memtable handed out never holds a document: an empty one, else a
    /// fresh one. A memtable that holds documents (checked in before, or
    /// written through the index) never leaves the index, so every document
    /// already indexed (explicit or auto-generated id) stays visible to the
    /// realtime lookups that updates and deletes by id rely on. Concurrent
    /// batches therefore each fill their own memtable; the owner seals them
    /// when they pile up ([`Index::num_open_memtables`]).
    pub fn checkout_writer(&mut self) -> (SegmentWriter, Vec<Field>) {
        let mut w = match self.writers.iter().position(|w| w.num_docs() == 0) {
            Some(i) => self.writers.swap_remove(i),
            None => SegmentWriter::new(self.next_segment_id + self.writers.len() as u32),
        };
        w.reset_schema_cache();
        w.set_stored_compression(self.stored_compression);
        (w, self.schema.clone())
    }

    /// Return a memtable taken with [`Index::checkout_writer`]: its
    /// documents become part of the index (realtime at once; searchable at
    /// the next [`Index::publish`] in NRT mode, else at once).
    pub fn checkin_writer(&mut self, mut w: SegmentWriter) {
        w.reset_schema_cache();
        self.writers.push(w);
        self.renumber_writers();
    }

    /// Lucene `addDocuments` block: `children` (nested docs of `path`)
    /// followed by `parent`, kept contiguous through flushes and merges and
    /// deleted together. Returns the parent's address.
    pub fn add_block(&mut self, path: &str, children: &[Document], parent: &Document) -> Result<DocAddress> {
        let mut p = parent.clone();
        for c in children {
            p.add_nested(path, c.clone());
        }
        self.add_document(&p)
    }

    /// Add a batch of documents with `threads` indexing threads, each
    /// appending a contiguous slice of `docs` to its own memtable (Lucene's
    /// `DocumentsWriterPerThread`; no shared state on the hot path). Every
    /// memtable is a searchable leaf and is flushed as its own segment by
    /// [`Index::refresh`]. Returns each doc's address (leaf index as in
    /// [`Index::all_leaves`], doc id within the leaf).
    pub fn add_documents_parallel(&mut self, docs: &[Document], threads: usize) -> Result<Vec<DocAddress>> {
        let threads = threads.max(1).min(docs.len().max(1));
        let base = self.writers_base();
        if threads == 1 {
            self.ensure_writers(1);
            let w = &mut self.writers[0];
            let slots = docs.iter().try_fold(0usize, |count, doc| count.checked_add(SegmentWriter::document_slots(doc)?)
                .ok_or_else(|| RusticError::Memory("parallel bitmap count overflow".into())))?;
            w.prepare_bitmap_capacity(slots)?;
            let mut out = Vec::with_capacity(docs.len());
            for d in docs {
                #[cfg(feature = "mutable-hnsw-poc")]
                let prepared = w.prepare_document_vectors(d, &self.schema, &self.vectors)?;
                w.add_document(d, &self.schema);
                #[cfg(feature = "mutable-hnsw-poc")]
                w.accept_document_vectors(prepared);
                out.push(DocAddress::new(base, w.num_docs() - 1));
            }
            return Ok(out);
        }
        self.ensure_writers(threads);
        let per = docs.len().div_ceil(threads);
        for (writer, chunk) in self.writers.iter_mut().zip(docs.chunks(per)) {
            let slots = chunk.iter().try_fold(0usize, |count, doc| count.checked_add(SegmentWriter::document_slots(doc)?)
                .ok_or_else(|| RusticError::Memory("parallel bitmap count overflow".into())))?;
            writer.prepare_bitmap_capacity(slots)?;
        }
        let starts: Vec<u32> = self.writers.iter().map(|w| w.num_docs()).collect();
        let schema = &self.schema;
        // Reuse the caller's Rayon pool rather than spawning OS threads for
        // each batch. Mutable writers are disjoint and share one schema.
        use rayon::prelude::*;
        let admission = crate::admission::Context::capture();
        self.writers.par_iter_mut().zip(docs.par_chunks(per)).try_for_each(|(w, chunk)| -> Result<()> {
            let _admission = admission.enter();
            for d in chunk {
                #[cfg(feature = "mutable-hnsw-poc")]
                let prepared = w.prepare_document_vectors(d, schema, &self.vectors)?;
                w.add_document(d, schema);
                #[cfg(feature = "mutable-hnsw-poc")]
                w.accept_document_vectors(prepared);
            }
            Ok(())
        })?;
        // Leaf index of each memtable: sealed segments, then non-empty
        // memtables in order.
        let mut leaf_of = Vec::with_capacity(self.writers.len());
        let mut next = base;
        for w in &self.writers {
            leaf_of.push(next);
            if w.num_docs() > 0 {
                next += 1;
            }
        }
        // A document with nested children occupies a block of docs (on
        // its thread's memtable, so blocks stay contiguous); its address
        // is the block's last doc.
        let mut out = Vec::with_capacity(docs.len());
        for (i, chunk) in docs.chunks(per).enumerate() {
            let mut next = starts[i];
            for d in chunk {
                next += d.block_len() as u32;
                out.push(DocAddress::new(leaf_of[i], next - 1));
            }
        }
        Ok(out)
    }

    /// Add multiple documents (bulk indexing).
    pub fn add_documents(&mut self, docs: &[Document]) -> Result<Vec<DocAddress>> {
        let mut addresses = Vec::with_capacity(docs.len());
        for doc in docs {
            addresses.push(self.add_document(doc)?);
        }
        Ok(addresses)
    }

    /// Near-real-time mode (ES refresh semantics). Off (the default), every
    /// added document and every delete is visible to searches at once (the
    /// memtable is a leaf). On, searches see the index as of the last
    /// [`Index::publish`]: new documents (in memtables, or in segments
    /// [`Index::seal`]ed meanwhile) and deletes stay invisible until then,
    /// while [`Index::all_leaves`] and the `*_latest` deletion views show
    /// every write (realtime get, version checks). Turning it off publishes.
    pub fn set_nrt(&mut self, on: bool) {
        if !on && self.nrt {
            // Seals only: a failure leaves the data in memtables.
            let _ = self.publish();
        }
        self.nrt = on;
        // NRT readers search published views: mutable-graph deletes wait
        // for the views that still count them live (outside NRT mode views
        // do not follow deletes, so nothing waits).
        self.vectors.set_view_clock(on.then_some(&self.view_slot));
    }

    /// Sort the docs of every segment written from now on (flushes and
    /// merges) by `sort` (ES `index.sort.*`; not persisted, set it after
    /// opening). In NRT mode published memtables are sealed at once, so
    /// searches only ever see sorted segments.
    pub fn set_index_sort(&mut self, sort: Option<crate::index_sort::IndexSort>) {
        self.index_sort = sort.filter(|s| !s.fields.is_empty());
    }

    /// Frame and compress the stored fields of memtables as `mode` (ES
    /// `index.codec`: `best_compression` is large zstd chunks; not
    /// persisted, set it after opening). Memtables already holding docs
    /// frame their next chunks this way and compress all of theirs so at
    /// seal; merges keep each input chunk's compression (a sorting merge
    /// re-chunks zstd when an input is zstd).
    pub fn set_stored_compression(&mut self, mode: crate::codec::store::StoredCompression) {
        self.stored_compression = mode;
        for w in self.writers.iter_mut() {
            w.set_stored_compression(mode);
        }
    }

    /// See [`Index::set_stored_compression`].
    pub fn stored_compression(&self) -> crate::codec::store::StoredCompression {
        self.stored_compression
    }

    /// Keep refresh-visible leaves separate when sealing. Their boundaries
    /// are observable by a leaf sorter; explicit/background merges remain allowed.
    pub fn set_preserve_published_leaves(&mut self, preserve: bool) {
        self.preserve_published_leaves = preserve;
    }

    pub fn preserves_published_leaves(&self) -> bool {
        self.preserve_published_leaves
    }

    /// The index sort, if any.
    pub fn index_sort(&self) -> Option<&crate::index_sort::IndexSort> {
        self.index_sort.as_ref()
    }

    /// Whether NRT mode is on (see [`Index::set_nrt`]).
    pub fn is_nrt(&self) -> bool {
        self.nrt
    }

    /// NRT mode: some write (document or delete) is not yet visible to
    /// searches.
    pub fn has_unpublished(&self) -> bool {
        self.nrt && (self.view_dirty.load(std::sync::atomic::Ordering::Acquire) || self.pending_dels.load(std::sync::atomic::Ordering::Acquire) || !self.hidden.is_empty() || !self.parked.is_empty() || self.writers.iter().any(|w| w.num_docs() > 0))
    }

    /// ES refresh: make every write so far visible to searches (NRT mode;
    /// a no-op otherwise). Cheap: memtables are published as they are
    /// (frozen, still in memory), pending deletes become visible, segments
    /// sealed meanwhile join the searchable set; then
    /// [`Index::maybe_seal_published`]. Returns whether anything changed.
    pub fn publish(&mut self) -> Result<bool> {
        let changed = self.publish_visible()?;
        #[cfg(feature = "mutable-hnsw-poc")]
        self.vectors.drain_mutable_deletes()?;
        Ok(changed)
    }

    /// [`Index::publish`] without the mutable-graph delete repair: the
    /// owner repairs in bounded background batches
    /// ([`VectorStore::drain_mutable_deletes_batch`]) outside its locks.
    /// Visibility is unchanged by the deferral: searches mask deleted
    /// documents through their reader view's deletions, so a deleted
    /// document still in the graph is never returned.
    pub fn publish_visible(&mut self) -> Result<bool> {
        if self.index_sort.is_some() && self.seal_in_flight() {
            return Err(RusticError::Segment("sorted refresh must wait for its detached seal installation".into()));
        }
        let changed = self.publish_writes_and_view()?;
        if !changed { return Ok(false); }
        self.maybe_seal_published()?;
        Ok(true)
    }

    /// [`Index::publish_visible`] of an index-sorted core without sealing:
    /// its memtables become searchable as they are (as ES makes its
    /// in-memory buffer searchable at refresh) and the owner seals them into
    /// sorted segments in the background ([`Index::begin_seal_chunks`]).
    pub fn publish_memtables(&mut self) -> Result<bool> {
        if self.seal_in_flight() {
            return Err(RusticError::Segment("a refresh publishing memtables must wait for the detached seal installation".into()));
        }
        self.publish_writes_and_view()
    }

    /// Refresh of an index-sorted core without encoding: make visible what
    /// is sealed (hidden segments, written in sort order) and every pending
    /// delete, never a memtable (searches only see sorted segments). The
    /// owner seals the memtables being written first, detached
    /// ([`Index::begin_seal`] / [`Index::finish_seal`]; deletes taken during
    /// the seal are remapped at installation); documents written since stay
    /// unpublished until the next refresh. Returns whether anything changed.
    pub fn publish_sealed(&mut self) -> Result<bool> {
        if self.seal_in_flight() {
            return Err(RusticError::Segment("sorted refresh must wait for its detached seal installation".into()));
        }
        if !self.nrt || !(self.view_dirty.load(std::sync::atomic::Ordering::Acquire) || self.pending_dels.load(std::sync::atomic::Ordering::Acquire) || !self.hidden.is_empty()) {
            return Ok(false);
        }
        for s in &self.segments {
            s.publish_deletions()?;
        }
        for w in &self.frozen {
            w.publish_deletions();
        }
        // Complete all fallible publication before moving hidden ownership.
        for s in &self.hidden { s.publish_deletions()?; }
        for s in std::mem::take(&mut self.hidden) {
            if let Some(store) = s.vector_store() {
                store.reveal(&s);
            }
            self.segments.push(s);
        }
        // Deletes are pending on searchable leaves only (memtables being
        // written and parked ones delete at once): all published now.
        self.pending_dels.store(false, std::sync::atomic::Ordering::Release);
        self.publish_view()?;
        Ok(true)
    }

    /// [`Index::publish_writes`] and, when anything changed, the new reader
    /// view (views taken before keep their frozen deletions).
    fn publish_writes_and_view(&mut self) -> Result<bool> {
        let changed = self.publish_writes()? || self.view_dirty.load(std::sync::atomic::Ordering::Acquire);
        if changed {
            self.publish_view()?;
        }
        Ok(changed)
    }

    fn publish_writes(&mut self) -> Result<bool> {
        if !self.has_unpublished() {
            return Ok(false);
        }
        for s in &self.segments {
            s.publish_deletions()?;
        }
        for w in &self.frozen {
            w.publish_deletions();
        }
        // Complete all fallible publication before moving hidden ownership.
        for s in &self.hidden { s.publish_deletions()?; }
        for s in std::mem::take(&mut self.hidden) {
            if let Some(store) = s.vector_store() {
                store.reveal(&s);
            }
            self.segments.push(s);
        }
        for w in std::mem::take(&mut self.parked) {
            w.publish_deletions();
            self.frozen.push(w);
        }
        for mut w in std::mem::take(&mut self.writers) {
            if w.num_docs() > 0 {
                w.publish_deletions();
                // A closed memtable keeps the segment id it was given.
                w.id = self.next_segment_id;
                self.next_segment_id += 1;
                w.account_closed();
                self.frozen.push(Arc::new(w));
            }
        }
        self.renumber_writers();
        self.pending_dels.store(false, std::sync::atomic::Ordering::Release);
        Ok(true)
    }

    /// Published memtables kept in memory before they are sealed into a
    /// segment: at most this many ...
    const MAX_FROZEN: usize = 8;
    /// ... holding at most this many docs in total.
    const MAX_FROZEN_DOCS: u32 = 10_000;

    /// NRT housekeeping of [`Index::publish`]: seal the published
    /// memtables once there are many (searching a memtable re-encodes its
    /// postings per query) and merge the small segments that accumulate
    /// from frequent refreshes.
    fn maybe_seal_published(&mut self) -> Result<()> {
        let sorted = self.index_sort.is_some() && !self.frozen.is_empty();
        if sorted || (self.auto_seal && self.frozen_over_limit()) {
            self.seal_frozen()?;
            self.merge_small_segments()?;
            self.publish_view()?;
        }
        Ok(())
    }

    /// Published memtables are many or large enough to be sealed.
    pub fn frozen_over_limit(&self) -> bool {
        let docs: u32 = self.frozen.iter().map(|w| w.num_docs()).sum();
        self.frozen.len() > Self::MAX_FROZEN || docs > Self::MAX_FROZEN_DOCS
    }

    /// Whether [`Index::publish`] seals and merges itself (see
    /// [`Index::set_auto_seal`]).
    pub fn auto_seal(&self) -> bool {
        self.auto_seal
    }

    /// Off: [`Index::publish`] leaves sealing published memtables and
    /// merging small segments to the owner's background jobs
    /// ([`Index::begin_seal`] / [`Index::begin_merge`]). Index-sorted
    /// indices still seal at publish (searches only see sorted segments).
    pub fn set_auto_seal(&mut self, on: bool) {
        self.auto_seal = on;
    }

    /// Small visible segments are merged once ten of a size class (powers
    /// of ten of their doc count, below [`Index::SMALL_SEGMENT_DOCS`])
    /// accumulate, so each doc is rewritten at most once per class.
    const SMALL_SEGMENT_DOCS: u32 = 100_000;

    fn merge_small_segments(&mut self) -> Result<()> {
        loop {
            let mut classes: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
            for s in &self.segments {
                if s.num_docs < Self::SMALL_SEGMENT_DOCS && !s.has_pending_deletions() && !self.merging.contains(&s.id) {
                    classes.entry(s.num_docs.max(1).ilog10()).or_default().push(s.id);
                }
            }
            let Some(ids) = classes.into_values().find(|v| v.len() >= 10) else { return Ok(()) };
            self.merge_segments(&ids)?;
        }
    }

    /// Seal `writers` (non-empty) into segments, concurrently when several
    /// (each flush also encodes its sections in parallel). `hidden`: see
    /// [`SegmentWriter::write_to_store_as`].
    /// Memtables a seal encodes concurrently ([`Index::seal_writers`], [`SealJob::run`]).
    pub const MAX_PARALLEL_SEALS: usize = 4;

    fn seal_writers(&mut self, writers: Vec<Arc<SegmentWriter>>, hidden: bool) -> Result<Vec<Arc<Segment>>> {
        if writers.is_empty() {
            return Ok(Vec::new());
        }
        let dir = &*self.directory;
        let store = &self.vectors;
        let admission = crate::admission::Context::capture();
        let admission = &admission;
        let segments: Vec<Result<Segment>> = if writers.len() == 1 || writers.iter().any(|w| w.has_vectors()) {
            writers.iter().map(|w| w.write_to_store_ref(dir, Some(store), hidden)).collect()
        } else {
            // A few at a time: every memtable's encode is a top-level job
            // on the encode pool, and a pool worker waiting inside one job
            // runs other queued top-level jobs on its own stack. Hundreds of
            // memtables sealed at once (a backlog) nested that deep enough
            // to overflow an encode thread's stack.
            let mut out = Vec::with_capacity(writers.len());
            for group in writers.chunks(Self::MAX_PARALLEL_SEALS) {
                out.extend(std::thread::scope(|s| {
                    let handles: Vec<_> = group
                        .iter()
                        .map(|w| s.spawn(move || admission.run(|| w.write_to_store_ref(dir, Some(store), hidden))))
                        .collect();
                    handles.into_iter().map(|h| h.join().expect("flush thread panicked")).collect::<Vec<_>>()
                }));
            }
            out
        };
        let segments: Vec<Arc<Segment>> = segments.into_iter().map(|s| s.map(Arc::new)).collect::<Result<_>>()?;
        if self.index_sort.is_none() {
            return Ok(segments);
        }
        segments.into_iter().map(|s| self.sort_segment(s, hidden)).collect()
    }

    /// The live docs (as searches see them) of `segments` in index-sort
    /// order; `None` when they already are in that order (or no sort).
    fn sorted_order(&self, segments: &[Arc<Segment>]) -> Result<Option<crate::admission::Allocation<Vec<(usize, DocId)>>>> {
        let Some(sort) = self.index_sort.as_ref() else{return Ok(None)};
        let mut memory=sort.reserve_workspace(segments)?;
        let leaves: Vec<LeafReader<'_>> = segments.iter().map(|s| LeafReader::Segment(s)).collect();
        // Sort root docs; a nested block (children, then its root) moves
        // as a unit (Lucene's parent field).
        let roots: Vec<(usize, DocId)> = segments
            .iter()
            .enumerate()
            .flat_map(|(i, s)| {
                let dels = s.deletions();
                let leaf = LeafReader::Segment(s);
                (0..s.num_docs).filter(move |&d| !dels.get(d) && leaf.is_root(d)).map(move |d| (i, d))
            })
            .collect();
        let sorted = sort.sort_docs(&leaves, roots.clone());
        if sorted == roots && segments.len() == 1 {
            return Ok(None);
        }
        let mut order = Vec::with_capacity(sorted.len());
        for (i, root) in sorted {
            let start = segments[i].nested_info().map_or(root, |n| n.block_start(root));
            order.extend((start..=root).map(|d| (i, d)));
        }
        drop(roots);drop(leaves);
        memory.shrink(memory.bytes().saturating_sub((order.capacity()*std::mem::size_of::<(usize,DocId)>()) as u64));
        Ok(Some(crate::admission::Allocation::new(order,memory)))
    }

    /// Rewrite a freshly flushed segment in index-sort order (Lucene sorts
    /// a segment at flush). The unsorted file is dropped at once: nothing
    /// ever saw it.
    fn sort_segment(&mut self, seg: Arc<Segment>, hidden: bool) -> Result<Arc<Segment>> {
        let Some(order) = self.sorted_order(std::slice::from_ref(&seg))? else { return Ok(seg) };
        let (order,_memory)=order.into_parts();
        let new_id = self.next_segment_id;
        self.next_segment_id += 1;
        self.renumber_writers();
        let chosen = [seg];
        let merged = self.merge_into(&chosen, new_id, Some(order), true)?;
        if hidden {
            if let Some(store) = merged.vector_store() {
                store.attach_hidden(&merged);
            }
        }
        let old = &chosen[0];
        for f in [crate::codec::segment_file_name(old.id), crate::codec::deletes_file_name(old.id)] {
            if self.directory.exists(&f) {
                let _ = self.directory.delete(&f);
            }
        }
        Ok(Arc::new(merged))
    }

    /// Start a background flush (NRT mode, no index sort): with
    /// `close_writers` the memtables being written are closed (parked,
    /// still realtime leaves; new writes go to fresh memtables); they and
    /// the published memtables are handed to the returned job. `None` when there is nothing to seal, a previous
    /// job is still parked. Vector chunks are encoded by the detached job,
    /// with their final visibility reconciled at installation. One job at a time: the caller
    /// installs it with [`Index::finish_seal`] before beginning another,
    /// and runs no foreground seal / merge / commit meanwhile.
    pub fn begin_seal(&mut self, close_writers: bool) -> Result<Option<SealJob>> {
        self.begin_seal_part(close_writers, usize::MAX)
    }

    /// [`Index::begin_seal`] claiming at most `max_memtables` of the
    /// memtables no other in-flight seal job claimed: several jobs seal
    /// disjoint sets concurrently (Lucene flushes DWPTs concurrently), each
    /// installed independently by [`Index::finish_seal`]. Sealing never
    /// changes what searches see (published memtables become searchable
    /// segments with the same deletions, parked ones hidden segments), so
    /// install order does not matter. `None`: nothing unclaimed to seal.
    /// Largest memtable total (estimated heap bytes) one seal job writes as
    /// one coalesced segment.
    pub const COALESCE_MAX_BYTES: u64 = 128 << 20;
    /// Largest memtable total one seal job coalesces when the unclaimed
    /// backlog is large ([`Index::begin_seal_bytes`]).
    pub const COALESCE_BACKLOG_MAX_BYTES: u64 = 512 << 20;

    pub fn begin_seal_part(&mut self, close_writers: bool, max_memtables: usize) -> Result<Option<SealJob>> {
        if !self.nrt {
            return Ok(None);
        }
        if close_writers {
            let closed = self.close_writers();
            if !closed.is_empty() {
                self.parked.extend(closed);
                // Parked memtables count in the view's stats.
                self.publish_view()?;
            }
        }
        let _selection=crate::admission::shared().reserve((self.frozen.len().saturating_add(self.parked.len()) as u64).saturating_mul(256),"seal candidate metadata")?;
        let mut mems: Vec<(Arc<SegmentWriter>, bool)> = self.frozen.iter().filter(|w| !self.sealing.contains(&w.id)).map(|w| (w.clone(), true)).collect();
        mems.extend(self.parked.iter().filter(|w| !self.sealing.contains(&w.id)).map(|w| (w.clone(), false)));
        mems.truncate(max_memtables.max(1));
        self.claim_seal(mems, Self::COALESCE_MAX_BYTES)
    }

    /// [`Index::begin_seal_part`] writing one segment where it can: the
    /// unclaimed memtables of one visibility class (published first, else
    /// parked), without vectors unless the first has them, together at
    /// most `max_bytes` (at least one memtable), so the job coalesces them
    /// into a single segment however many small memtables (one per
    /// concurrent writer) the buffer holds. Call again while
    /// [`Index::num_unclaimed_memtables`] is above zero for the rest.
    pub fn begin_seal_bytes(&mut self, close_writers: bool, max_bytes: u64) -> Result<Option<SealJob>> {
        self.begin_seal_bytes_inner(close_writers, max_bytes, true, true)
    }

    /// Capture all currently unclaimed memtables at ONE caller-held boundary,
    /// partitioned into independently installable byte-bounded jobs. New writes
    /// after this call are not claimed. Each job retains its deletion capture;
    /// callers must install every job, including after an earlier job fails, so
    /// no remaining claim is stranded. A single oversized memtable is indivisible.
    ///
    /// Chunks coalesce their memtables only when the whole capture is within
    /// `max_bytes` (at most one chunk per visibility class: a merge pass over
    /// at most `max_bytes` in all, as one job over the capture made): a
    /// larger capture's chunks are written memtable by memtable, without an
    /// extra merge pass per chunk.
    pub fn begin_seal_chunks(&mut self, close_writers: bool, max_bytes: u64) -> Result<Vec<SealJob>> {
        self.begin_seal_chunks_with(close_writers, max_bytes, true)
    }

    /// [`Index::begin_seal_chunks`] for a flush: a capture within
    /// `max_bytes` is not coalesced either; every memtable becomes its own
    /// segment, encoded in parallel (a Lucene flush writes a segment per
    /// DWPT and leaves combining them to the merge policy). Only a pile of
    /// tiny memtables (see [`MANY_MEMTABLES`]) is still coalesced per chunk.
    ///
    /// The coalescing merge ran serially inside the flush, over up to
    /// `max_bytes` of memtables: beside four concurrent bulk writers
    /// (shard::tests::flushes_beside_in_flight_bulks_lose_nothing_across_a_crash)
    /// the slowest flush took about 5 s on 16 cores, and on 32 cores its
    /// term dictionary was refused memory the bulks' memtables held.
    pub fn begin_flush_seal_chunks(&mut self, close_writers: bool, max_bytes: u64) -> Result<Vec<SealJob>> {
        self.begin_seal_chunks_with(close_writers, max_bytes, false)
    }

    fn begin_seal_chunks_with(&mut self, close_writers: bool, max_bytes: u64, coalesce_small: bool) -> Result<Vec<SealJob>> {
        if !self.nrt {
            return Ok(Vec::new());
        }
        if close_writers {
            let closed = self.close_writers();
            if !closed.is_empty() {
                self.parked.extend(closed);
                self.publish_view()?;
            }
        }
        let max_bytes = parallel_chunk_bytes(self.frozen.iter().chain(self.parked.iter()).filter(|w| !self.sealing.contains(&w.id))
            .map(|w| w.estimated_bytes() as u64).fold(0u64, u64::saturating_add), max_bytes);
        // Every chunk is one segment, flushes included: a memtable is one
        // bulk request's share of the shard (eql: ~1,000 docs), and writing
        // each as its own segment left a flush or a large refresh with dozens
        // of tiny segments for the merge policy to rewrite (an eql load
        // merged ~30x as often as ES). Lucene flushes a DWPT, which holds
        // many requests' documents, as one segment. Coalescing is a single
        // encode of the chunk ([`union_one_pass`]); chunks encode in parallel.
        let _ = coalesce_small;
        let coalesce = true;
        let mut jobs = Vec::new();
        loop {
            match self.begin_seal_bytes_inner(false, max_bytes, false, coalesce) {
                Ok(Some(job)) => jobs.push(job),
                Ok(None) => return Ok(jobs),
                Err(error) => {
                    // Refusal never loses already claimed, required work.
                    // No encoding ran: release every claim so a retry sees
                    // the same memtables and captures a fresh boundary.
                    for job in jobs { self.cancel_unrun_seal(job); }
                    return Err(error);
                }
            }
        }
    }

    /// [`Index::begin_seal_chunks`] of what the next publish would make
    /// searchable only: the memtables being written (closed now, at one
    /// boundary) and the parked ones no in-flight job claimed. Published
    /// memtables are left to the owner's background seals. A refresh that
    /// writes its own segments (a Lucene refresh flushes its DWPTs) claims
    /// these, so it never encodes more than it is about to publish. Chunks
    /// are at most `max_bytes` (at least one memtable), split by vector
    /// content; they coalesce into one segment each only when the whole
    /// capture is within `max_bytes`. On a refusal no claim is kept.
    pub fn begin_seal_unpublished_chunks(&mut self, max_bytes: u64) -> Result<Vec<SealJob>> {
        if !self.nrt {
            return Ok(Vec::new());
        }
        let closed = self.close_writers();
        if !closed.is_empty() {
            self.parked.extend(closed);
            self.publish_view()?;
        }
        let max_bytes = max_bytes.max(1);
        let _selection = crate::admission::shared().reserve((self.parked.len() as u64).saturating_mul(256), "seal candidate metadata")?;
        let unclaimed: Vec<Arc<SegmentWriter>> = self.parked.iter().filter(|w| !self.sealing.contains(&w.id)).cloned().collect();
        let max_bytes = parallel_chunk_bytes(unclaimed.iter().map(|w| w.estimated_bytes() as u64).fold(0u64, u64::saturating_add), max_bytes);
        // Each chunk one segment (see `begin_seal_chunks_with`).
        let coalesce = true;
        let mut jobs = Vec::new();
        for vectors in [false, true] {
            let mut chunk: Vec<(Arc<SegmentWriter>, bool)> = Vec::new();
            let mut bytes = 0u64;
            let group = unclaimed.iter().filter(|w| w.has_vectors() == vectors);
            for w in group.cloned().map(Some).chain(std::iter::once(None)) {
                let b = w.as_ref().map_or(0, |w| w.estimated_bytes() as u64);
                if !chunk.is_empty() && (w.is_none() || bytes.saturating_add(b) > max_bytes) {
                    let bound = chunk_coalesce_bound(coalesce, &chunk, max_bytes);
                    match self.claim_seal(std::mem::take(&mut chunk), bound) {
                        Ok(Some(job)) => jobs.push(job),
                        Ok(None) => {}
                        Err(error) => {
                            for job in jobs { self.cancel_unrun_seal(job); }
                            return Err(error);
                        }
                    }
                    bytes = 0;
                }
                if let Some(w) = w {
                    bytes = bytes.saturating_add(b);
                    chunk.push((w, false));
                }
            }
        }
        Ok(jobs)
    }

    /// Ids of the parked memtables (closed, not published, not sealed yet).
    pub fn parked_memtable_ids(&self) -> Vec<u32> {
        self.parked.iter().map(|w| w.id).collect()
    }

    /// Whether any of `ids` is still a parked memtable here.
    pub fn holds_parked(&self, ids: &[u32]) -> bool {
        self.parked.iter().any(|w| ids.contains(&w.id))
    }

    fn begin_seal_bytes_inner(&mut self, close_writers: bool, max_bytes: u64, adaptive: bool, coalesce: bool) -> Result<Option<SealJob>> {
        if !self.nrt {
            return Ok(None);
        }
        if close_writers {
            let closed = self.close_writers();
            if !closed.is_empty() {
                self.parked.extend(closed);
                self.publish_view()?;
            }
        }
        let _selection=crate::admission::shared().reserve((self.frozen.len().saturating_add(self.parked.len()) as u64).saturating_mul(256),"seal candidate metadata")?;
        let published: Vec<Arc<SegmentWriter>> = self.frozen.iter().filter(|w| !self.sealing.contains(&w.id)).cloned().collect();
        let (candidates, class) = if published.is_empty() {
            (self.parked.iter().filter(|w| !self.sealing.contains(&w.id)).cloned().collect::<Vec<_>>(), false)
        } else {
            (published, true)
        };
        let Some(first)=candidates.first() else{return Ok(None)};
        let vectors = first.has_vectors();
        // A large backlog (the tail of a bulk load, a refresh after one) is
        // written as a few large segments, not one per `max_bytes` (Lucene
        // flushes its whole RAM buffer): each job takes a quarter of it, up
        // to [`Self::COALESCE_BACKLOG_MAX_BYTES`].
        let backlog: u64 = candidates.iter().filter(|w| w.has_vectors() == vectors).map(|w| w.estimated_bytes() as u64).sum();
        let target = if adaptive { coalesce_target(max_bytes, backlog) } else { max_bytes };
        let large = (target > max_bytes).then(LargeSeal::acquire).flatten();
        let max_bytes = if large.is_some() { target } else { max_bytes };
        let mut mems = Vec::new();
        let mut bytes = 0u64;
        for w in candidates.into_iter().filter(|w| w.has_vectors() == vectors) {
            let b = w.estimated_bytes() as u64;
            if !mems.is_empty() && bytes + b > max_bytes {
                continue;
            }
            bytes += b;
            mems.push((w, class));
        }
        // A zero coalescing bound writes each memtable as its own segment.
        let bound = chunk_coalesce_bound(coalesce, &mems, max_bytes);
        let Some(mut job) = self.claim_seal(mems, bound)? else{return Ok(None)};
        job._large = large;
        Ok(Some(job))
    }

    /// Claim `mems` for one seal job (coalesced into one segment when they
    /// share a visibility class and are small enough; index-sorted ones
    /// merge-sorted, vector ones (mutable graph fields too) with their
    /// rows, whose owners move to the segment at installation).
    fn claim_seal(&mut self, mems: Vec<(Arc<SegmentWriter>, bool)>, max_bytes: u64) -> Result<Option<SealJob>> {
        self.claim_seal_with_budget(mems,max_bytes,&crate::admission::shared())
    }

    fn claim_seal_with_budget(&mut self, mems: Vec<(Arc<SegmentWriter>, bool)>, max_bytes: u64, budget:&Arc<crate::admission::Budget>) -> Result<Option<SealJob>> {
        if mems.is_empty() {
            return Ok(None);
        }
        let bytes=mems.iter().fold((mems.len() as u64).saturating_mul(2048),|sum,(w,_)|sum.saturating_add((w.num_docs() as u64).saturating_mul(4)));
        let memory=budget.reserve(bytes,"seal descriptors and deletion snapshots")?;
        self.sealing.extend(mems.iter().map(|(w, _)| w.id));
        let sorted_ids = if self.index_sort.is_some() {
            let ids = (0..mems.len()).map(|_| { let id = self.next_segment_id; self.next_segment_id += 1; id }).collect();
            self.renumber_writers();
            ids
        } else { Vec::new() };
        // Memtables of one visibility class (all published, or all parked:
        // a copy no refresh publishes) and together small enough that one
        // more merge pass costs little.
        let one_class = mems.iter().all(|(_, published)| *published) || mems.iter().all(|(_, published)| !*published);
        let small = mems.iter().map(|(w, _)| w.estimated_bytes() as u64).sum::<u64>() <= max_bytes;
        let preserve = self.preserve_published_leaves && mems.iter().any(|(_, published)| *published);
        let coalesce_id = (mems.len() >= 2 && one_class && small && !preserve).then(|| {
            let id = self.next_segment_id;
            self.next_segment_id += 1;
            self.renumber_writers();
            id
        });
        let captured = mems.iter().map(|(w, _)| w.latest_deletions()).collect();
        Ok(Some(SealJob { admission: crate::admission::Context::capture(), mems, directory: self.directory.clone(), vectors: self.vectors.clone(),
            sort: self.index_sort.clone(), sorted_ids, coalesce_id, captured, _large: None, _memory: memory }))
    }

    /// Abandon a job BEFORE encoding begins. Its writers stay in the index
    /// with all visibility/deletions intact and may be claimed by a retry.
    /// Consuming the unrun job prevents cancelling an already installed job
    /// or removing the claim of a subsequent retry using the same writer IDs.
    pub fn cancel_unrun_seal(&mut self, job: SealJob) {
        for (writer, _) in &job.mems { self.sealing.remove(&writer.id); }
    }

    /// Install a finished [`SealJob`]: each memtable is replaced by its
    /// segment where it stands now (published: a searchable segment; still
    /// parked: a hidden one), with the deletes it took meanwhile. A
    /// memtable whose encoding failed stays a memtable (a later seal
    /// retries it); the first error is returned.
    pub fn finish_seal(&mut self, sealed: SealedMemtables) -> Result<()> {
        self.finish_seal_captured(sealed).map(|_| ())
    }

    /// [`Index::finish_seal`], also returning, for each segment it
    /// installed, the deletions its memtables had when the job began (in the
    /// segment's doc order): a flush commits those, not deletes taken after
    /// its capture point.
    /// A refused install abandons the encoded work ([`Index::abandon_seal`]);
    /// [`Index::try_finish_seal_captured`] lets the caller retry it instead.
    pub fn finish_seal_captured(&mut self, sealed: SealedMemtables) -> Result<DeletionsAt> {
        match self.try_finish_seal_captured(sealed) {
            Ok(result) => result,
            Err(SealInstallRefused { error, sealed }) => {
                self.abandon_seal(sealed);
                Err(error)
            }
        }
    }

    /// [`Index::finish_seal_captured`] unless the install's admission is
    /// refused up front: then nothing changed, the job keeps its claims and
    /// its encoded segments, and the caller retries the returned job later
    /// (briefly, off the core lock: claims held block sorted refreshes and
    /// flush trimming) or gives it up with [`Index::abandon_seal`].
    pub fn try_finish_seal_captured(&mut self, sealed: SealedMemtables) -> std::result::Result<Result<DeletionsAt>, SealInstallRefused> {
        let bitmap_bytes = sealed.done.iter().map(|(w, _)| crate::postings::BitSet::storage_bytes(w.num_docs()))
            .chain(sealed.coalesced.iter().flat_map(|(writers, _)| writers.iter().map(|w| crate::postings::BitSet::storage_bytes(w.num_docs()))))
            .fold(0u64, u64::saturating_add).saturating_mul(8);
        #[cfg(test)]
        if REFUSE_SEAL_INSTALL.with(|n| { let left = n.get(); if left > 0 { n.set(left - 1); } left > 0 }) {
            return Err(SealInstallRefused { error: RusticError::Memory("injected seal install admission refusal".into()), sealed });
        }
        let bitmap_memory = match crate::admission::shared().reserve(bitmap_bytes, "seal deletion replacement snapshots") {
            Ok(memory) => Arc::new(memory),
            Err(error) => return Err(SealInstallRefused { error, sealed }),
        };
        Ok(self.install_sealed(sealed, bitmap_memory))
    }

    /// Give up an encoded job that was not installed: its claims are
    /// released (its memtables stay, for a later seal) and its encoded
    /// segment files deleted.
    pub fn abandon_seal(&mut self, sealed: SealedMemtables) {
        for (writer, seg) in &sealed.done {
            self.sealing.remove(&writer.id);
            if let Ok((seg, _)) = seg { let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id)); }
        }
        if let Some((writers, merged)) = &sealed.coalesced {
            for writer in writers { self.sealing.remove(&writer.id); }
            if let Ok((seg, _)) = merged { let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id)); }
        }
    }

    fn install_sealed(&mut self, sealed: SealedMemtables, bitmap_memory: Arc<crate::admission::Reservation>) -> Result<DeletionsAt> {
        // Every claim of the job is released before any fallible step: an
        // output whose install is refused stays a memtable for a later seal,
        // and the remaining outputs still install. A claim left behind would
        // keep `seal_in_flight` true for good (sorted refreshes, flush
        // trimming and writes would wait on a seal that never finishes).
        for (writer, _) in &sealed.done { self.sealing.remove(&writer.id); }
        if let Some((writers, _)) = &sealed.coalesced { for writer in writers { self.sealing.remove(&writer.id); } }
        let mut first_err = None;
        let mut at: DeletionsAt = HashMap::new();
        let captured = sealed.captured;
        if let Some((writers, merged)) = sealed.coalesced {
            // Each memtable's doc `d` in the segment: its doc map (sorted),
            // else its offset in the concatenation plus `d`.
            let place = |map: &Option<crate::admission::Allocation<Vec<Vec<DocId>>>>, num_docs: u32, mask: &dyn Fn(usize) -> crate::postings::BitSet| {
                let mut bits = crate::postings::BitSet::new(num_docs);
                let mut offset = 0u32;
                for (i, w) in writers.iter().enumerate() {
                    for d in mask(i).iter_set_bits() {
                        bits.set(match map { Some(m) => m[i][d as usize], None => offset + d });
                    }
                    offset += w.num_docs();
                }
                bits.with_memory(bitmap_memory.clone())
            };
            let empty = |i: usize| crate::postings::BitSet::new(writers[i].num_docs());
            let captured_mask = |i: usize| captured.get(i).cloned().unwrap_or_else(|| empty(i));
            let vectors = writers.iter().any(|w| w.has_vectors());
            let writer_ids: Vec<u32> = writers.iter().map(|w| w.id).collect();
            let published = writers.iter().all(|w| self.frozen.iter().any(|f| Arc::ptr_eq(f, w)));
            let parked = !published && writers.iter().all(|w| self.parked.iter().any(|f| Arc::ptr_eq(f, w)));
            match (merged, published, parked) {
                (Ok((seg, map)), false, true) => {
                    // Still parked: one hidden segment, with each memtable's
                    // deletions placed through its doc map.
                    if let Err(e) = seg.set_deletions_prepaid(place(&map, seg.num_docs, &|i| writers[i].latest_deletions()), None, &bitmap_memory) {
                        // Refused: the memtables stay parked for a later seal.
                        let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
                        first_err.get_or_insert(e);
                    } else {
                    at.insert(seg.id, Arc::new(place(&map, seg.num_docs, &captured_mask)));
                    if vectors { self.vectors.attach_coalesced(&seg, &writer_ids, true); }
                    self.parked.retain(|f| !writers.iter().any(|w| Arc::ptr_eq(f, w)));
                    self.hidden.push(Arc::new(seg));
                    }
                }
                (Ok((seg, map)), true, _) => {
                    let pending = writers.iter().any(|w| w.has_pending_deletions());
                    let latest = place(&map, seg.num_docs, &|i| writers[i].latest_deletions());
                    let visible = pending.then(|| place(&map, seg.num_docs, &|i| writers[i].deletions()));
                    if let Err(e) = seg.set_deletions_prepaid(latest, visible, &bitmap_memory) {
                        // Refused: the memtables stay published for a later seal.
                        let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
                        first_err.get_or_insert(e);
                    } else {
                    at.insert(seg.id, Arc::new(place(&map, seg.num_docs, &captured_mask)));
                    // Owners move from the memtables' rows to the segment
                    // (the merge path's reconciliation).
                    if vectors { self.vectors.attach_coalesced(&seg, &writer_ids, false); }
                    self.frozen.retain(|f| !writers.iter().any(|w| Arc::ptr_eq(f, w)));
                    self.segments.push(Arc::new(seg));
                    }
                }
                (Ok((seg, _)), false, false) => {
                    // The memtables no longer share one class (not expected:
                    // a refresh publishes every parked memtable at once):
                    // drop the copy; the memtables stay for a later seal,
                    // and the error tells a flush its seal did not install.
                    let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
                    first_err.get_or_insert(RusticError::Segment("coalesced seal: its memtables left their visibility class".into()));
                }
                (Err(e), _, _) => { first_err.get_or_insert(e); }
            }
            retire_writers(writers);
        }
        for (i, (w, seg)) in sealed.done.into_iter().enumerate() {
            let (seg, remap) = match seg {
                Ok(s) => s,
                Err(e) => {
                    first_err.get_or_insert(e);
                    continue;
                }
            };
            let remap_mask = |mask: crate::postings::BitSet| {
                let Some(map) = &remap else { return mask.with_memory(bitmap_memory.clone()); };
                let mut reordered = crate::postings::BitSet::new(seg.num_docs);
                for old in mask.iter_set_bits() { reordered.set(map[old as usize]); }
                reordered.with_memory(bitmap_memory.clone())
            };
            if let Some(c) = captured.get(i) {
                at.insert(seg.id, Arc::new(remap_mask(c.clone())));
            }
            if let Some(pos) = self.frozen.iter().position(|f| Arc::ptr_eq(f, &w)) {
                let visible = w.has_pending_deletions().then(|| remap_mask(w.deletions()));
                if let Err(e) = seg.set_deletions_prepaid(remap_mask(w.latest_deletions()), visible, &bitmap_memory) {
                    // Refused: this memtable stays published for a later
                    // seal; the job's other outputs still install.
                    at.remove(&seg.id);
                    let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
                    first_err.get_or_insert(e);
                    continue;
                }
                // A refresh may have published the writer after encoding
                // captured its hidden state. Reconcile owner visibility and
                // concurrent deletes before replacing the readable memtable.
                if w.has_vectors() { self.vectors.attach(&seg); }
                self.frozen.remove(pos);
                self.segments.push(Arc::new(seg));
            } else if let Some(pos) = self.parked.iter().position(|f| Arc::ptr_eq(f, &w)) {
                if let Err(e) = seg.set_deletions_prepaid(remap_mask(w.latest_deletions()), None, &bitmap_memory) {
                    // Refused: this memtable stays parked for a later seal.
                    at.remove(&seg.id);
                    let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
                    first_err.get_or_insert(e);
                    continue;
                }
                if w.has_vectors() { self.vectors.attach_hidden(&seg); }
                self.parked.remove(pos);
                self.hidden.push(Arc::new(seg));
            } else {
                // Sealed in the foreground meanwhile (not expected): drop the copy.
                at.remove(&seg.id);
                let _ = self.directory.delete(&crate::codec::segment_file_name(seg.id));
            }
        }
        self.publish_view()?;
        match first_err {
            Some(e) => Err(e),
            None => Ok(at),
        }
    }

    /// Every sealed segment's deletions now (searchable and hidden; cheap
    /// `Arc` snapshots): a flush's capture point ([`DeletionsAt`]).
    pub fn deletions_snapshot(&self) -> Result<DeletionsAt> {
        self.segments.iter().chain(&self.hidden).map(|s| Ok((s.id, s.latest_deletions()?))).collect()
    }

    /// Start a background merge (NRT mode) of the first
    /// candidate `policy` finds among searchable or hidden segments, skipping
    /// segments another job claimed. Installed with [`Index::finish_merge`], or
    /// abandoned with [`Index::release_merge`].
    /// Average bytes per document of the segments (searchable and hidden;
    /// `None` without any): converts ES's byte-sized merge policy settings to
    /// this policy's document counts.
    pub fn avg_doc_bytes(&self) -> Option<f64> {
        let (bytes, docs) = self.segments.iter().chain(self.hidden.iter())
            .fold((0u64, 0u64), |(b, d), s| (b + s.file_bytes() as u64, d + s.num_docs as u64));
        (docs > 0).then(|| bytes as f64 / docs as f64)
    }

    /// Merges `policy` would start now over the segments no job claimed
    /// (searchable and hidden classes): the merge backlog (ES
    /// `ConcurrentMergeScheduler` pending merges).
    pub fn num_pending_merges(&self, policy: &crate::merge::MergePolicy) -> usize {
        self.num_pending_merges_on(policy, &crate::admission::shared())
    }

    fn num_pending_merges_on(&self, policy: &crate::merge::MergePolicy, budget: &Arc<crate::admission::Budget>) -> usize {
        if !self.nrt { return 0; }
        let count=self.segments.len().saturating_add(self.hidden.len());
        // Telemetry must remain allocation-free on admission denial. The
        // last measured backlog stands meanwhile: reporting every segment as
        // pending would throttle indexing to one request per shard while
        // `begin_merge`, denied the same way, starts nothing. The scheduler
        // retries planning while that backlog is nonzero, so a backlog never
        // measured (planning refused since open) reports one pending merge
        // while there are segments: a write-idle index still re-arms its
        // merge request instead of reading as fully merged.
        let Ok(_memory)=budget.reserve((count as u64).saturating_mul(1024),"merge backlog planning") else {
            return match self.merge_backlog.load(std::sync::atomic::Ordering::Acquire) {
                BACKLOG_UNKNOWN => usize::from(count > 0),
                last => last,
            };
        };
        let backlog = if policy.log_bytes.is_some() {
            self.log_proposals(policy).len().min(PENDING_MERGES_COUNTED)
        } else if policy.tiered.is_some() {
            self.tiered_proposals(policy, PENDING_MERGES_COUNTED).len()
        } else {
            [&self.segments, &self.hidden].into_iter().map(|segments| self.policy_proposals(policy, segments, PENDING_MERGES_COUNTED).len()).sum()
        };
        self.merge_backlog.store(backlog, std::sync::atomic::Ordering::Release);
        backlog
    }

    pub fn begin_merge(&mut self, policy: &crate::merge::MergePolicy) -> Option<MergeJob> {
        if !self.nrt {
            return None;
        }
        // Never mix visibility classes: sealing must not make newly indexed
        // documents searchable before refresh. Hidden segments still need
        // compaction when refresh is disabled or suppressed by search-idle.
        // The cheapest merge of either class first (delete-weighted size, as
        // [`crate::merge::merge_weight`]): tiny segments a burst of seals
        // left, hidden ones included, before a large rewrite. Every fourth
        // pick takes the proposal over the oldest segments instead, so a
        // large (or delete-reclaiming) merge ages into its turn under a
        // steady stream of small ones.
        static PICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let oldest_turn = PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 4 == 3;
        // Background work is optional: leave inputs unclaimed for the next
        // scheduler retry when planning admission is unavailable.
        let memory=crate::admission::shared().reserve((self.segments.len().saturating_add(self.hidden.len()) as u64).saturating_mul(1024),"merge candidate planning").ok()?;
        let mut proposals: Vec<Vec<Arc<Segment>>> = Vec::new();
        if policy.log_bytes.is_some() {
            // Every merge the time-based policy proposes (adjacent windows,
            // the merging ones skipped); the cheapest is started below.
            for cand in self.log_proposals(policy) {
                let chosen: Vec<_> = self.segments.iter().chain(&self.hidden).filter(|s| cand.segment_ids.contains(&s.id)).cloned().collect();
                if !chosen.is_empty() { proposals.push(chosen); }
            }
        } else if policy.tiered.is_some() {
            // One allowance over both classes (Lucene counts every flushed
            // segment, refreshed or not): the policy's best merge.
            if let Some(cand) = self.tiered_proposals(policy, 1).into_iter().next() {
                let chosen: Vec<_> = self.segments.iter().chain(&self.hidden).filter(|s| cand.segment_ids.contains(&s.id)).cloned().collect();
                if !chosen.is_empty() { proposals.push(chosen); }
            }
        }
        for segments in [&self.segments, &self.hidden].into_iter().filter(|_| policy.tiered.is_none() && policy.log_bytes.is_none()) {
            for (k, cand) in self.policy_proposals(policy, segments, 1).into_iter().enumerate() {
                // The tiered policy ranks its proposals: each class's best.
                if policy.tiered.is_some() && k > 0 {
                    break;
                }
                let chosen: Vec<_> = segments.iter().filter(|s| cand.segment_ids.contains(&s.id)).cloned().collect();
                if !chosen.is_empty() {
                    proposals.push(chosen);
                }
            }
        }
        let weight = |c: &Vec<Arc<Segment>>| crate::merge::merge_weight(c.iter().map(|s| s.num_live_docs() as u64).sum(), c.iter().map(|s| s.num_docs as u64).sum());
        let chosen = if oldest_turn && policy.tiered.is_none() && policy.log_bytes.is_none() {
            proposals.into_iter().min_by_key(|c| c.iter().map(|s| s.id).min().unwrap_or(u32::MAX))
        } else {
            proposals.into_iter().min_by_key(weight)
        }?;
        Some(self.claim(chosen, false, memory))
    }

    /// `policy`'s merges over one visibility class: the byte-sized tiered
    /// policy sees the claimed segments too (they count toward the index
    /// size), the doc-count policy only unclaimed ones.
    /// `tiered_limit`: at most this many tiered proposals (the best ones;
    /// see [`crate::merge::MergePolicy::find_merges_tiered_up_to`]).
    fn policy_proposals(&self, policy: &crate::merge::MergePolicy, segments: &[Arc<Segment>], tiered_limit: usize) -> Vec<crate::merge::MergeCandidate> {
        if policy.tiered.is_some() {
            let sizes: Vec<_> = segments.iter().map(|s| self.segment_size(s, 0)).collect();
            return policy.find_merges_tiered_up_to(&sizes, tiered_limit);
        }
        let stats: Vec<_> = segments.iter().filter(|s| !self.merging.contains(&s.id))
            .map(|s| (s.id, s.num_docs, s.num_live_docs())).collect();
        policy.find_merges(&stats)
    }

    fn segment_size(&self, s: &Segment, class: u8) -> crate::merge::SegmentSize {
        crate::merge::SegmentSize { id: s.id, docs: s.num_docs, live: s.num_live_docs(), bytes: s.segment_bytes(), merging: self.merging.contains(&s.id), class }
    }

    /// The time-based (log byte size) policy's merges: searchable segments
    /// and hidden ones each in their order, a merge within one class (the
    /// classes never mix; Lucene's single segment list is ordered alike).
    fn log_proposals(&self, policy: &crate::merge::MergePolicy) -> Vec<crate::merge::MergeCandidate> {
        let mut out = Vec::new();
        for (class, segments) in [&self.segments, &self.hidden].into_iter().enumerate() {
            let sizes: Vec<_> = segments.iter().map(|s| self.segment_size(s, class as u8)).collect();
            out.extend(policy.find_merges_log_bytes(&sizes));
        }
        out
    }

    /// The tiered policy's merges with searchable (class 0) and hidden
    /// (class 1) segments counted against one segment allowance; each
    /// proposal stays within one class.
    fn tiered_proposals(&self, policy: &crate::merge::MergePolicy, limit: usize) -> Vec<crate::merge::MergeCandidate> {
        let sizes: Vec<_> = self.segments.iter().map(|s| self.segment_size(s, 0))
            .chain(self.hidden.iter().map(|s| self.segment_size(s, 1))).collect();
        policy.find_merges_tiered_up_to(&sizes, limit)
    }

    /// A job over `chosen`, claimed until finished or released.
    fn claim(&mut self, chosen: Vec<Arc<Segment>>, force: bool, memory: crate::admission::Reservation) -> MergeJob {
        self.merging.extend(chosen.iter().map(|s| s.id));
        let visible = chosen.iter().map(|s| s.deletions()).collect();
        let new_id = self.next_segment_id;
        self.next_segment_id += 1;
        self.renumber_writers();
        MergeJob { admission: crate::admission::Context::capture(), force, throttled: true, chosen, visible, new_id, directory: self.directory.clone(), sort: self.index_sort.clone(), order: Default::default(), install_map: Default::default(), carried: Default::default(), retired: Default::default(), _planning_memory: memory }
    }

    /// Release a job's segments without installing it (its encoding failed).
    pub fn release_merge(&mut self, job: &MergeJob) {
        for s in &job.chosen {
            self.merging.remove(&s.id);
        }
    }

    /// Prepare a text force merge without doing its encoding
    /// under the caller's index write lock. Vector point ownership remains
    /// with the inputs until installation, including segmented-vector fields.
    /// Merges the searchable segments no other job claimed; writes, seals
    /// and background merges of other segments may run meanwhile.
    pub fn begin_force_merge(&mut self) -> Result<Option<MergeJob>> {
        let memory=crate::admission::shared().reserve((self.segments.len() as u64).saturating_mul(1024),"force merge candidate planning")?;
        let chosen: Vec<_> = self.segments.iter().filter(|s| !self.merging.contains(&s.id)).cloned().collect();
        // Expunging even one segment can encode gigabytes. It must use the
        // detached protocol too, including concurrent deletion reconciliation.
        if chosen.is_empty() || (chosen.len() == 1 && chosen[0].num_live_docs() == chosen[0].num_docs) {
            return Ok(None);
        }
        Ok(Some(self.claim(chosen, true, memory)))
    }

    /// Install a finished [`MergeJob`]: its inputs are replaced by the
    /// merged segment (at the first input's position), with the deletes
    /// the inputs took meanwhile carried over (pending ones stay pending).
    /// Returns whether it was installed.
    pub fn finish_merge(&mut self, job: MergeJob, merged: Segment) -> Result<bool> {
        self.finish_merge_with_budget(job, merged, &crate::admission::shared())
    }

    fn finish_merge_with_budget(&mut self, mut job: MergeJob, merged: Segment, budget: &Arc<crate::admission::Budget>) -> Result<bool> {
        self.release_merge(&job);
        let output = crate::codec::segment_file_name(merged.id);
        let staged_retired = job.retired.take();
        let ids: Vec<u32> = job.chosen.iter().map(|s| s.id).collect();
        // A refresh can move all hidden inputs to the searchable set while
        // encoding runs. Install into their current set, never their old one.
        let hidden = job.chosen.iter().all(|c| self.hidden.iter().any(|s| Arc::ptr_eq(s, c)));
        let inputs = if hidden { &self.hidden } else { &self.segments };
        let present = job.chosen.iter().all(|c| inputs.iter().any(|s| Arc::ptr_eq(s, c)));
        let Some(pos) = inputs.iter().position(|s| ids.contains(&s.id)).filter(|_| present) else {
            let _ = self.directory.delete(&output);
            return Ok(false);
        };
        // Everything that can be refused happens before the swap. A refused
        // install deletes its output (rather than leaving it to the next
        // restart's orphan sweep), and a done swap always publishes its view.
        let staged = (|| -> Result<_> {
            let map = job.install_map()?;
            let mut pending = false;
            let precomputed = job.carried.get();
            for (i, seg) in job.chosen.iter().enumerate() {
                let (latest, visible) = (seg.latest_deletions()?, seg.deletions());
                // Off-lock result, unless this input took a delete (or a publish)
                // since encoding finished: then only the scan is redone here.
                let owned;
                let carried: &[(DocId, bool)] = match precomputed.and_then(|c| c.get(i)) {
                    Some((l, v, list)) if Arc::ptr_eq(l, &latest) && Arc::ptr_eq(v, &visible) => list,
                    _ => {
                        owned = carried_deletes(map, i, &latest, &visible, &job.visible[i], budget)?;
                        &owned
                    }
                };
                for &(n, now_visible) in carried {
                    if now_visible {
                        merged.delete_doc(n)?;
                    } else {
                        merged.delete_doc_pending(n)?;
                        pending = true;
                    }
                }
            }
            let retired = match staged_retired {
                Some(retired) => retired,
                None => Self::prepare_retired_with_budget(&job.chosen, budget)?,
            };
            // The view after the swap holds fewer segments than now.
            let view = budget.reserve(self.view_bytes(), "published reader view")?;
            Ok((pending, retired, view))
        })();
        let (pending, retired, view) = match staged {
            Ok(staged) => staged,
            Err(e) => {
                drop(merged);
                let _ = self.directory.delete(&output);
                return Err(e);
            }
        };
        if pending {
            self.pending_dels.store(true, std::sync::atomic::Ordering::Release);
        }
        // Encoding must not move owners while old leaves still serve queries.
        // Publish only after validating inputs and carrying over concurrent deletes.
        if hidden { self.vectors.attach_hidden(&merged); }
        else { self.vectors.attach(&merged); }
        let outputs = if hidden { &mut self.hidden } else { &mut self.segments };
        outputs.retain(|s| !ids.contains(&s.id));
        outputs.insert(pos.min(outputs.len()), Arc::new(merged));
        self.retire_prepared(retired);
        // A hidden install changes only the view's stats counts.
        self.publish_view_admitted(view);
        Ok(true)
    }

    /// Merge `chosen` into segment `new_id` (in `order`, else input order),
    /// keeping deletes not yet published pending on the result.
    /// `seal`: a sorted seal's rewrite (never throttled, see
    /// [`crate::merge::seal_segments_with`]).
    fn merge_into(&self, chosen: &[Arc<Segment>], new_id: u32, order: Option<Vec<(usize, DocId)>>, seal: bool) -> Result<Segment> {
        let _map_copy = crate::admission::shared().reserve(order.as_ref().map_or(0, |o| (o.len()*std::mem::size_of::<(usize,DocId)>()) as u64), "synchronous merge install order copy")?;
        let map_order = order.clone();
        let merged = if seal {
            crate::merge::seal_segments_with(&*self.directory, chosen, new_id, order)?
        } else {
            crate::merge::merge_segments_with(&*self.directory, chosen, new_id, order)?
        };
        if chosen.iter().any(|s| s.has_pending_deletions()) {
            let map = crate::merge::admitted_doc_map(chosen, None, map_order)?;
            for (i, seg) in chosen.iter().enumerate() {
                if !seg.has_pending_deletions() {
                    continue;
                }
                let (latest, visible) = (seg.latest_deletions()?, seg.deletions());
                for d in latest.iter_set_bits() {
                    if !visible.get(d) {
                        if let Some(n) = map.new_id(i, d) {
                            merged.delete_doc_pending(n)?;
                        }
                    }
                }
            }
            merged.write_deletions(&*self.directory)?;
        }
        Ok(merged)
    }

    /// Whether published memtables are held in memory (not yet sealed).
    pub fn has_frozen(&self) -> bool {
        !self.frozen.is_empty()
    }

    /// Seal the published memtables into (searchable) segments; visibility
    /// does not change.
    pub fn seal_frozen(&mut self) -> Result<()> {
        // Memtables a background seal is encoding (shared with its job)
        // stay: the job installs them.
        let (busy, frozen): (Vec<_>, Vec<_>) = std::mem::take(&mut self.frozen).into_iter().partition(|w| self.sealing.contains(&w.id));
        self.frozen = busy;
        if frozen.is_empty() {
            return Ok(());
        }
        let segs = self.seal_writers(frozen, false)?;
        self.segments.extend(segs);
        self.publish_view()?;
        Ok(())
    }

    /// Closed memtables (published, or detached for a background seal):
    /// what Lucene would already hold as flushed segments.
    pub fn num_closed_memtables(&self) -> usize {
        self.frozen.len() + self.parked.len()
    }

    /// Memtables being written that hold documents (each becomes a leaf at
    /// publish and a segment at seal).
    pub fn num_open_memtables(&self) -> usize {
        self.writers.iter().filter(|w| w.num_docs() > 0).count()
    }

    /// Documents in the memtables being written (not yet closed).
    pub fn num_writing_docs(&self) -> u32 {
        self.writers.iter().map(|w| w.num_docs()).sum()
    }

    /// Documents in memtables of any kind, not in a segment yet: being
    /// written, published (frozen) and closed for a seal (parked). What the
    /// background seal is behind on (Lucene's flush-pending RAM).
    pub fn num_unsealed_bytes(&self) -> u64 {
        self.writers.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
            + self.frozen.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
            + self.parked.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
    }

    /// Bytes the next publish would make searchable as memtables: the ones
    /// being written and the parked ones.
    pub fn num_unpublished_bytes(&self) -> u64 {
        self.writers.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
            + self.parked.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
    }

    /// Ids of the memtables the next publish would make searchable (being
    /// written, parked).
    pub fn unpublished_memtable_ids(&self) -> Vec<u32> {
        self.writers.iter().map(|w| w.id).chain(self.parked.iter().map(|w| w.id)).collect()
    }

    /// Whether any of `ids` is still a memtable here (not sealed yet).
    pub fn holds_memtables(&self, ids: &[u32]) -> bool {
        self.writers.iter().any(|w| ids.contains(&w.id))
            || self.parked.iter().chain(self.frozen.iter()).any(|w| ids.contains(&w.id))
    }

    /// Published memtables (searchable, not sealed yet).
    pub fn num_published_memtables(&self) -> usize {
        self.frozen.len()
    }

    /// [`Index::num_unsealed_bytes`] of the memtables no seal job claimed
    /// (being written, or published / parked and unclaimed).
    pub fn num_unclaimed_bytes(&self) -> u64 {
        self.writers.iter().map(|w| w.estimated_bytes() as u64).sum::<u64>()
            + self.frozen.iter().chain(self.parked.iter()).filter(|w| !self.sealing.contains(&w.id)).map(|w| w.estimated_bytes() as u64).sum::<u64>()
    }

    /// [`Index::num_unsealed_bytes`] in documents.
    pub fn num_unsealed_docs(&self) -> u64 {
        self.writers.iter().map(|w| w.num_docs() as u64).sum::<u64>()
            + self.frozen.iter().map(|w| w.num_docs() as u64).sum::<u64>()
            + self.parked.iter().map(|w| w.num_docs() as u64).sum::<u64>()
    }

    /// Estimated heap bytes of the memtables being written: their indexing
    /// structures, without the read caches searches build over them (those
    /// carry their own accounting charge, and ES's indexing buffer counts
    /// only indexing memory).
    pub fn writing_bytes(&self) -> usize {
        self.writers.iter().map(|w| w.indexing_bytes()).sum()
    }

    /// A background seal job is encoding some memtable.
    pub fn seal_in_flight(&self) -> bool {
        !self.sealing.is_empty()
    }

    /// Whether memtable `id` is claimed by an in-flight seal job.
    pub fn is_sealing(&self, id: u32) -> bool {
        self.sealing.contains(&id)
    }

    /// Memtables (published or parked) no in-flight seal job claimed.
    pub fn num_unclaimed_memtables(&self) -> usize {
        self.frozen.iter().chain(self.parked.iter()).filter(|w| !self.sealing.contains(&w.id)).count()
    }

    /// Close the non-empty memtables being written (they keep their docs,
    /// as closed memtables, under the segment ids they get now): `frozen`
    /// outside NRT mode, else `parked`.
    fn close_writers(&mut self) -> Vec<Arc<SegmentWriter>> {
        let mut out = Vec::new();
        for mut w in std::mem::take(&mut self.writers) {
            if w.num_docs() > 0 {
                w.id = self.next_segment_id;
                self.next_segment_id += 1;
                w.account_closed();
                out.push(Arc::new(w));
            }
        }
        self.renumber_writers();
        out
    }

    /// Write every memtable to a segment without changing what searches
    /// see (the segment half of an ES flush): published memtables become
    /// searchable segments, unpublished ones (NRT mode) segments that
    /// stay invisible until the next [`Index::publish`].
    pub fn seal(&mut self) -> Result<()> {
        if !self.nrt {
            return self.refresh();
        }
        self.seal_frozen()?;
        let (busy, parked): (Vec<_>, Vec<_>) = std::mem::take(&mut self.parked).into_iter().partition(|w| self.sealing.contains(&w.id));
        self.parked = busy;
        let segs = self.seal_writers(parked, true)?;
        self.hidden.extend(segs);
        let writers = self.close_writers();
        let segs = self.seal_writers(writers, true)?;
        self.hidden.extend(segs);
        // Searchable leaves unchanged; the stats counts of the view moved.
        self.publish_view()?;
        Ok(())
    }

    /// Publish everything (see [`Index::publish`]) and write every memtable
    /// to a searchable segment.
    ///
    /// Every non-empty memtable becomes one segment; several memtables are
    /// flushed concurrently (each flush also encodes its sections in
    /// parallel).
    pub fn refresh(&mut self) -> Result<()> {
        self.publish_writes_and_view()?;
        #[cfg(feature = "mutable-hnsw-poc")]
        self.vectors.drain_mutable_deletes()?;
        self.seal_frozen()?;
        let writers = self.close_writers();
        let segs = self.seal_writers(writers, false)?;
        self.segments.extend(segs);
        self.publish_view()?;
        Ok(())
    }

    /// Delete a document (Lucene live docs). Persists on `commit()`. `addr`
    /// is a leaf index in [`Index::all_leaves`] order (memtables
    /// included). In NRT mode searches keep seeing the doc until the next
    /// [`Index::publish`].
    pub fn delete_doc(&mut self, addr: &DocAddress) -> Result<()> {
        self.delete_doc_shared(addr)
    }

    /// [`Index::delete_doc`] through a shared reference: every leaf keeps
    /// its deletions behind its own lock, so deletes of existing documents
    /// can be applied under a read guard of the index (the leaf set cannot
    /// change meanwhile) while searches and other deletes proceed.
    pub fn delete_doc_shared(&self, addr: &DocAddress) -> Result<()> {
        let mut idx = addr.segment_id as usize;
        if idx < self.segments.len() {
            if self.nrt {
                self.segments[idx].delete_doc_pending(addr.doc_id)?;
                self.pending_dels.store(true, std::sync::atomic::Ordering::Release);
            } else {
                // Visible at once to `leaves()`; reader views follow at the
                // next publish / seal / merge (views are for NRT indices).
                self.segments[idx].delete_doc(addr.doc_id)?;
            }
            return Ok(());
        }
        idx -= self.segments.len();
        if idx < self.frozen.len() {
            self.frozen[idx].delete_doc_pending(addr.doc_id)?;
            self.pending_dels.store(true, std::sync::atomic::Ordering::Release);
            return Ok(());
        }
        idx -= self.frozen.len();
        if idx < self.hidden.len() {
            // Hidden leaves are not searchable; keep the prepared pending
            // delete path so acknowledged updates never need fresh COW
            // admission here. Publication makes its vector deletes visible.
            self.hidden[idx].delete_doc_pending(addr.doc_id)?;
            return Ok(());
        }
        idx -= self.hidden.len();
        if idx < self.parked.len() {
            self.parked[idx].delete_doc(addr.doc_id);
            return Ok(());
        }
        idx -= self.parked.len();
        if let Some(w) = self.writers.iter().filter(|w| w.num_docs() > 0).nth(idx) {
            w.delete_doc(addr.doc_id);
            return Ok(());
        }
        Err(RusticError::IndexNotFound(format!("segment {}", addr.segment_id)))
    }

    /// The searchable segments, in leaf order.
    pub fn segments(&self) -> &[Arc<Segment>] {
        &self.segments
    }

    /// NRT mode: segments written but not yet searchable.
    pub fn hidden_segments(&self) -> &[Arc<Segment>] {
        &self.hidden
    }

    /// Ids of every sealed segment, searchable and hidden.
    pub fn sealed_segment_ids(&self) -> std::collections::HashSet<u32> {
        self.segments.iter().chain(&self.hidden).map(|s| s.id).collect()
    }

    /// The schema has a vector field.
    pub fn has_vector_fields(&self) -> bool {
        self.schema.iter().any(|f| matches!(f.field_type, FieldType::Vector { .. }))
    }

    /// Every searchable leaf: sealed segments followed by the memtables
    /// that hold searchable documents. Leaf index `i` corresponds to
    /// `DocAddress::segment_id == i`; these leaves are a prefix of
    /// [`Index::all_leaves`], so addresses agree between the two.
    pub fn leaves(&self) -> Vec<LeafReader<'_>> {
        let mut v: Vec<LeafReader<'_>> = self.segments.iter().map(|s| LeafReader::Segment(s)).collect();
        v.extend(self.frozen.iter().map(|w| LeafReader::Memtable(w, None)));
        if !self.nrt {
            for w in &self.writers {
                if w.num_docs() > 0 {
                    v.push(LeafReader::Memtable(w, None));
                }
            }
        }
        v
    }

    /// Every leaf holding documents, searchable or not (realtime reads):
    /// [`Index::leaves`], then (NRT mode) the hidden segments and the
    /// memtables being written. Deletes not yet published are visible
    /// through the `*_latest` views only.
    pub fn all_leaves(&self) -> Vec<LeafReader<'_>> {
        let mut v: Vec<LeafReader<'_>> = self.segments.iter().map(|s| LeafReader::Segment(s)).collect();
        v.extend(self.frozen.iter().map(|w| LeafReader::Memtable(w, None)));
        v.extend(self.hidden.iter().map(|s| LeafReader::Segment(s)));
        v.extend(self.parked.iter().map(|w| LeafReader::Memtable(w, None)));
        for w in &self.writers {
            if w.num_docs() > 0 {
                v.push(LeafReader::Memtable(w, None));
            }
        }
        v
    }

    /// Run `f` over every leaf in parallel (one rayon task per leaf, which is
    /// Lucene's concurrent segment search) and return the per-leaf results
    /// in leaf order.
    pub fn map_leaves_parallel<R: Send>(&self, f: impl Fn(usize, &LeafReader<'_>) -> R + Sync) -> Vec<R> {
        use rayon::prelude::*;
        let leaves = self.leaves();
        if leaves.len() <= 1 {
            return leaves.iter().enumerate().map(|(i, l)| f(i, l)).collect();
        }
        let admission = crate::admission::Context::capture();
        leaves.par_iter().enumerate().map(|(i, l)| admission.run(|| f(i, l))).collect()
    }

    /// Parallel variant of [`Index::search_term`].
    pub fn search_term_parallel(&self, term: &Term, top_k: usize) -> Result<SearchResults> {
        let start = std::time::Instant::now();
        let leaves = self.leaves();
        let stats = term_stats(&leaves, term);
        if stats.doc_freq == 0 {
            return Ok(SearchResults { took_ms: start.elapsed().as_millis() as u64, ..Default::default() });
        }
        let fs = field_stats(&leaves, term.field);
        let per_leaf = self.map_leaves_parallel(|i, leaf| {
            let Some(mut scorer) = leaf.term_scorer(term, Bm25Params::default(), stats, fs) else {
                return (Vec::new(), 0u64);
            };
            let (hits, n) = collect_top_k(&mut *scorer, top_k, leaf.live_docs());
            (hits.into_iter().map(|h| Hit::new(DocAddress::new(i as u32, h.doc), h.score)).collect::<Vec<_>>(), n)
        });
        let mut all_hits = Vec::new();
        let mut total = 0;
        for (hits, n) in per_leaf {
            total += n;
            all_hits.extend(hits);
        }
        all_hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_address.segment_id.cmp(&b.doc_address.segment_id)).then(a.doc_address.doc_id.cmp(&b.doc_address.doc_id)));
        all_hits.truncate(top_k);
        Ok(SearchResults {
            total_hits: total,
            max_score: all_hits.first().map(|h| h.score).unwrap_or(0.0),
            hits: all_hits,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Search for a phrase (consecutive terms in one field, with `slop`).
    pub fn search_phrase(&self, terms: &[Term], slop: u32, top_k: usize) -> Result<SearchResults> {
        let start = std::time::Instant::now();
        let leaves = self.leaves();
        let Some(first) = terms.first() else {
            return Ok(SearchResults::default());
        };
        let stats: Vec<_> = terms.iter().map(|t| term_stats(&leaves, t)).collect();
        if stats.iter().any(|s| s.doc_freq == 0) {
            return Ok(SearchResults { took_ms: start.elapsed().as_millis() as u64, ..Default::default() });
        }
        let fs = field_stats(&leaves, first.field);
        let mut all_hits = Vec::new();
        let mut total = 0u64;
        for (i, leaf) in leaves.iter().enumerate() {
            let Some(mut s) = leaf.phrase_scorer(terms, slop, Bm25Params::default(), &stats, fs) else {
                continue;
            };
            let (hits, n) = collect_top_k(&mut *s, top_k, leaf.live_docs());
            total += n;
            all_hits.extend(hits.into_iter().map(|h| Hit::new(DocAddress::new(i as u32, h.doc), h.score)));
        }
        all_hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        all_hits.truncate(top_k);
        Ok(SearchResults {
            total_hits: total,
            max_score: all_hits.first().map(|h| h.score).unwrap_or(0.0),
            hits: all_hits,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Get the number of searchable segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get the total number of documents (across all segments).
    pub fn num_docs(&self) -> u32 {
        self.segments.iter().map(|s| s.num_docs).sum()
    }

    /// Get the total number of live documents.
    pub fn num_live_docs(&self) -> u32 {
        self.segments.iter().map(|s| s.num_live_docs()).sum()
    }

    /// Documents buffered in memtables (not yet flushed).
    pub fn num_pending_docs(&self) -> u32 {
        self.writers.iter().map(|w| w.num_docs()).chain(self.frozen.iter().chain(&self.parked).map(|w| w.num_docs())).sum()
    }

    /// Search the index for a term, returning the top-k hits by BM25 score.
    ///
    /// Scores use index-wide statistics (same term, same weight in every
    /// leaf) and include unflushed documents in the memtable.
    pub fn search_term(&self, term: &Term, top_k: usize) -> Result<SearchResults> {
        let start = std::time::Instant::now();
        let leaves = self.leaves();
        let stats = term_stats(&leaves, term);
        if stats.doc_freq == 0 {
            return Ok(SearchResults {
                took_ms: start.elapsed().as_millis() as u64,
                ..SearchResults::default()
            });
        }
        let fs = field_stats(&leaves, term.field);
        let mut all_hits: Vec<Hit> = Vec::new();
        let mut total = 0u64;
        for (i, leaf) in leaves.iter().enumerate() {
            let Some(mut scorer) = leaf.term_scorer(term, Bm25Params::default(), stats, fs) else {
                continue;
            };
            let (hits, n) = collect_top_k(&mut *scorer, top_k, leaf.live_docs());
            total += n;
            all_hits.extend(hits.into_iter().map(|h| Hit::new(DocAddress::new(i as u32, h.doc), h.score)));
        }
        all_hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_address.segment_id.cmp(&b.doc_address.segment_id)).then(a.doc_address.doc_id.cmp(&b.doc_address.doc_id)));
        all_hits.truncate(top_k);
        Ok(SearchResults {
            total_hits: total,
            max_score: all_hits.first().map(|h| h.score).unwrap_or(0.0),
            hits: all_hits,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Search with a range query on a fast field (constant score), including
    /// the memtable.
    pub fn search_range(&self, query: &RangeQuery, top_k: usize) -> Result<SearchResults> {
        let start = std::time::Instant::now();
        let mut all_hits: Vec<Hit> = Vec::new();
        let mut total = 0u64;
        for (i, leaf) in self.leaves().iter().enumerate() {
            let matching: Vec<DocId> = match (&query.lower, &query.upper) {
                (Some(RangeBound::I64(lo)), up) => {
                    let hi = match up { Some(RangeBound::I64(u)) => *u, _ => i64::MAX };
                    leaf.range_i64(query.field, *lo, hi)
                }
                (None, Some(RangeBound::I64(u))) => leaf.range_i64(query.field, i64::MIN, *u),
                (Some(RangeBound::F64(lo)), up) => {
                    let hi = match up { Some(RangeBound::F64(u)) => *u, _ => f64::MAX };
                    leaf.range_f64(query.field, *lo, hi)
                }
                (None, Some(RangeBound::F64(u))) => leaf.range_f64(query.field, f64::MIN, *u),
                (Some(RangeBound::U64(lo)), up) => {
                    let hi = match up { Some(RangeBound::U64(u)) => *u, _ => u64::MAX };
                    leaf.range_u64(query.field, *lo, hi)
                }
                (None, Some(RangeBound::U64(u))) => leaf.range_u64(query.field, 0, *u),
                _ => continue,
            };
            let live = leaf.live_docs();
            for d in matching {
                if live(d) {
                    total += 1;
                    if all_hits.len() < top_k {
                        all_hits.push(Hit::new(DocAddress::new(i as u32, d), 1.0));
                    }
                }
            }
        }
        Ok(SearchResults {
            total_hits: total,
            hits: all_hits,
            max_score: 1.0,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Get a document by its address (leaf index in [`Index::all_leaves`]
    /// order, which agrees with [`Index::leaves`] + doc id).
    pub fn get_doc(&self, addr: &DocAddress) -> Result<Document> {
        let leaves = self.all_leaves();
        let leaf = leaves
            .get(addr.segment_id as usize)
            .ok_or_else(|| RusticError::IndexNotFound(format!("segment {}", addr.segment_id)))?;
        leaf.get_doc(addr.doc_id)?
            .ok_or_else(|| RusticError::IndexNotFound(format!("{}", addr)))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn commit_json_second_pass_cannot_grow_or_truncate_durable_manifest() {
        struct Changing { calls: std::cell::Cell<usize>, grow: bool }
        impl serde::Serialize for Changing {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
                let first = self.calls.replace(self.calls.get() + 1) == 0;
                serializer.serialize_str(if first == self.grow { "x" } else { "longer value" })
            }
        }
        let dir = RamDirectory::new();
        dir.write(META_FILE, b"original durable bytes").unwrap();
        for grow in [true, false] {
            let value = Changing { calls: std::cell::Cell::new(0), grow };
            assert!(matches!(write_admitted_json(&dir, META_FILE, &value), Err(RusticError::Serialization(_))));
            assert_eq!(dir.open_read(META_FILE).unwrap().as_slice(), b"original durable bytes");
        }
    }
    #[test]
    fn retired_registry_admits_before_names_and_holds_until_entry_removal() {
        let dir = crate::codec::directory::RamDirectory::new();
        let mut writer = SegmentWriter::new(7);
        writer.add_document(&Document::new(), &[]);
        let segment = Arc::new(writer.write_to(&dir).unwrap());
        let budget = crate::admission::Budget::new(511);
        assert!(matches!(Index::prepare_retired_with_budget(&[segment.clone()], &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), 0);
        budget.set_limit(512);
        let entries = Index::prepare_retired_with_budget(&[segment.clone()], &budget).unwrap();
        let mut registry: std::collections::BTreeMap<_, _> = entries.into_iter().collect();
        assert_eq!(budget.used(), 512);
        assert_eq!(registry[&7].data.strong_count(), 1);
        drop(segment);
        assert_eq!(registry[&7].data.strong_count(), 0);
        registry.remove(&7);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn refused_reader_publication_keeps_previous_generation_and_can_retry() {
        let mut index = Index::new("views", Vec::new(), Default::default());
        index.set_nrt(true);
        index.add_document(&Document::new()).unwrap();
        index.publish().unwrap();
        let budget = crate::admission::Budget::new(u64::MAX);
        index.publish_view_with_budget(&budget).unwrap();
        let retained = index.reader_view();
        let generation = retained.generation();
        let bytes = budget.used();
        assert!(bytes > 0);
        budget.set_limit(bytes);
        assert!(matches!(index.publish_view_with_budget(&budget), Err(RusticError::Memory(_))));
        assert_eq!(index.reader_view().generation(), generation);
        assert!(index.has_unpublished(), "publication denial must remain retryable");
        budget.set_limit(bytes * 2);
        index.publish_view_with_budget(&budget).unwrap();
        assert!(index.reader_view().generation() > generation);
        assert_eq!(budget.used(), bytes * 2);
        drop(retained);
        assert_eq!(budget.used(), bytes);
        drop(index);
        assert_eq!(budget.used(), 0);
    }

    use super::*;
    use crate::query::Query;

    /// v12 end to end: segments flushed and merged under write version 12
    /// have v12 headers and impact staircases, and pruned top-k searches
    /// over them (before and after the merge) match exhaustive scoring.
    #[test]
    fn v12_segments_flush_merge_and_search_like_exhaustive() {
        use crate::query::searcher::{SearchOptions, Searcher, TrackTotalHits};
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
        let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "body", FieldType::Text)];
        let dir: Arc<dyn Directory> = Arc::new(crate::codec::directory::RamDirectory::default());
        let mut index = Index::create_in(dir, "v12", schema, IndexSettings::default()).unwrap();
        let words = ["the", "of", "series", "film", "idol", "tv", "rare"];
        let mut seed = 5u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed >> 33
        };
        for _ in 0..3 {
            for _ in 0..3000 {
                let mut d = Document::new();
                d.add_text(0, &(0..1 + next() % 3).map(|_| words[next() as usize % 6]).collect::<Vec<_>>().join(" "));
                let n = 1 + next() % 60;
                d.add_text(1, &(0..n).map(|_| words[(next() % 100) as usize % if next() % 50 == 0 { 7 } else { 6 }]).collect::<Vec<_>>().join(" "));
                index.add_document(&d).unwrap();
            }
            index.refresh().unwrap();
        }
        let dm = |w: &str| Query::DisMax { queries: vec![Query::Term(Term::new_text(0, w)), Query::Term(Term::new_text(1, w))], tie_breaker: 0.0 };
        let check = |index: &Index, what: &str| {
            assert!(index.segments().iter().all(|s| s.format_version() == crate::codec::VERSION), "{what}: headers");
            // Every segment's lists carry staircases (some block of a common
            // word has more than one competitive step).
            for s in index.segments() {
                let r = s.postings(&Term::new_text(1, "the")).unwrap();
                assert!(r.num_full_blocks() > 0 && (0..r.num_full_blocks()).any(|b| r.skip(b).impacts[0] != 0), "{what}: segment {} carries no staircase", s.id);
            }
            let view = index.reader_view();
            let searcher = Searcher::new(view.leaves());
            for q in [
                Query::bool_query().should(dm("the")).should(dm("idol")).should(dm("series")).should(dm("rare")).build(),
                Query::bool_query().should(dm("of")).should(dm("film")).build(),
                dm("tv"),
            ] {
                for k in [1usize, 10] {
                    let pruned = searcher.search(&q, &SearchOptions { top_k: k, track_total_hits: TrackTotalHits::UpTo(100), ..Default::default() });
                    let exact = searcher.search(&q, &SearchOptions { top_k: k, track_total_hits: TrackTotalHits::True, ..Default::default() });
                    let hits = |t: &crate::query::searcher::TopDocs| t.hits.iter().map(|h| (h.leaf, h.doc, h.score.to_bits())).collect::<Vec<_>>();
                    assert_eq!(hits(&pruned), hits(&exact), "{what}: {q:?} k {k}");
                    assert_eq!(pruned.hits.len(), k);
                }
            }
        };
        check(&index, "flushed");
        index.force_merge().unwrap();
        assert_eq!(index.segments().len(), 1);
        check(&index, "merged");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    fn test_index() -> Index {
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "body", FieldType::Text),
            Field::new(2, "price", FieldType::F64),
            Field::new(3, "in_stock", FieldType::Bool),
            Field::new(4, "category", FieldType::Keyword),
        ];
        Index::new("test", schema, IndexSettings::default())
    }

    #[test]
    fn test_index_and_search() {
        let mut index = test_index();

        // Add documents
        let mut doc1 = Document::new();
        doc1.add_text(0, "Rust Programming Language");
        doc1.add_text(1, "Rust is a systems programming language");
        doc1.add_f64(2, 49.99);
        doc1.add_bool(3, true);
        doc1.add_keyword(4, "books");
        index.add_document(&doc1).unwrap();

        let mut doc2 = Document::new();
        doc2.add_text(0, "Learning Python");
        doc2.add_text(1, "Python is easy to learn");
        doc2.add_f64(2, 29.99);
        doc2.add_bool(3, false);
        doc2.add_keyword(4, "books");
        index.add_document(&doc2).unwrap();

        let mut doc3 = Document::new();
        doc3.add_text(0, "Advanced Rust");
        doc3.add_text(1, "Rust ownership and borrowing");
        doc3.add_f64(2, 59.99);
        doc3.add_bool(3, true);
        doc3.add_keyword(4, "books");
        index.add_document(&doc3).unwrap();

        // Refresh to make docs searchable
        index.refresh().unwrap();

        // Search for "rust"
        let term = Term::new_text(0, "rust");
        let results = index.search_term(&term, 10).unwrap();
        assert!(results.total_hits >= 2); // doc1 and doc3 have "rust" in title

        // Search for "python"
        let term = Term::new_text(0, "python");
        let results = index.search_term(&term, 10).unwrap();
        assert_eq!(results.total_hits, 1);
    }

    #[test]
    fn test_range_query() {
        let mut index = test_index();

        for i in 0..100 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc {}", i));
            doc.add_f64(2, i as f64);
            index.add_document(&doc).unwrap();
        }
        index.refresh().unwrap();

        // Search for price in [10, 20]
        let query = RangeQuery::range_f64(2, Some(10.0), Some(20.0));
        let results = index.search_range(&query, 100).unwrap();
        assert_eq!(results.total_hits, 11); // 10, 11, ..., 20
    }

    #[test]
    fn test_get_doc() {
        let mut index = test_index();

        let mut doc = Document::new();
        doc.add_text(0, "Hello World");
        doc.add_f64(2, 9.99);
        let addr = index.add_document(&doc).unwrap();
        index.refresh().unwrap();

        let retrieved = index.get_doc(&addr).unwrap();
        assert_eq!(retrieved.get(0).unwrap().as_text(), Some("Hello World"));
    }

    #[test]
    fn test_num_docs() {
        let mut index = test_index();

        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc {}", i));
            index.add_document(&doc).unwrap();
        }
        index.refresh().unwrap();

        assert_eq!(index.num_docs(), 10);
        assert_eq!(index.num_live_docs(), 10);
        assert_eq!(index.num_segments(), 1);
    }

    #[test]
    fn test_search_sees_memtable_before_refresh() {
        let mut index = test_index();
        for i in 0..20 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("pending doc {i}"));
            doc.add_f64(2, i as f64);
            index.add_document(&doc).unwrap();
        }
        // Nothing flushed yet.
        assert_eq!(index.num_segments(), 0);
        assert_eq!(index.num_pending_docs(), 20);
        let r = index.search_term(&Term::new_text(0, "pending"), 5).unwrap();
        assert_eq!(r.total_hits, 20);
        assert_eq!(r.hits.len(), 5);
        assert_eq!(r.hits[0].doc_address.segment_id, 0); // memtable is leaf 0
        let doc = index.get_doc(&r.hits[0].doc_address).unwrap();
        assert!(doc.get(0).unwrap().as_text().unwrap().starts_with("pending"));
        assert_eq!(index.search_range(&RangeQuery::range_f64(2, Some(5.0), Some(9.0)), 10).unwrap().total_hits, 5);

        index.refresh().unwrap();
        for i in 20..30 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("pending doc {i}"));
            index.add_document(&doc).unwrap();
        }
        let r = index.search_term(&Term::new_text(0, "pending"), 50).unwrap();
        assert_eq!(r.total_hits, 30);
        assert!(r.hits.iter().any(|h| h.doc_address.segment_id == 1), "memtable leaf after the segment");
        index.delete_doc(&DocAddress::new(1, 0)).unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "pending"), 50).unwrap().total_hits, 29);
        let phrase = index.search_phrase(&[Term::new_text(0, "pending"), Term::new_text(0, "doc")], 0, 50).unwrap();
        assert_eq!(phrase.total_hits, 29);
    }

    #[test]
    fn test_parallel_search_matches_sequential() {
        let mut index = test_index();
        for seg in 0..4 {
            for i in 0..50 {
                let mut doc = Document::new();
                doc.add_text(0, &format!("common term{} seg{seg}", i % 5));
                index.add_document(&doc).unwrap();
            }
            index.refresh().unwrap();
        }
        let mut doc = Document::new();
        doc.add_text(0, "common pending");
        index.add_document(&doc).unwrap();
        let a = index.search_term(&Term::new_text(0, "common"), 7).unwrap();
        let b = index.search_term_parallel(&Term::new_text(0, "common"), 7).unwrap();
        assert_eq!(a.total_hits, 201);
        assert_eq!(a.total_hits, b.total_hits);
        let ka: Vec<_> = a.hits.iter().map(|h| (h.doc_address, h.score)).collect();
        let kb: Vec<_> = b.hits.iter().map(|h| (h.doc_address, h.score)).collect();
        assert_eq!(ka, kb);
    }

    #[test]
    fn test_delete_doc() {
        let mut index = test_index();
        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(0, &format!("doc common {i}"));
            index.add_document(&doc).unwrap();
        }
        index.refresh().unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "common"), 100).unwrap().total_hits, 10);

        index.delete_doc(&DocAddress::new(0, 3)).unwrap();
        index.delete_doc(&DocAddress::new(0, 7)).unwrap();
        assert_eq!(index.num_live_docs(), 8);
        assert_eq!(index.search_term(&Term::new_text(0, "common"), 100).unwrap().total_hits, 8);
    }

    #[test]
    fn segment_stats_tracks_only_successful_commit_and_retired_segments() {
        let mut index = Index::new("stats", vec![Field::new(0, "n", FieldType::Text)], IndexSettings::default());
        for _ in 0..2 {
            let mut doc = Document::new(); doc.add_text(0, "one");
            index.add_document(&doc).unwrap(); index.refresh().unwrap();
        }
        assert!(index.reader_view().segment_stats().iter().all(|s| !s.committed));
        index.commit().unwrap();
        let committed = index.reader_view().segment_stats();
        assert_eq!(committed.len(), 2);
        assert!(committed.iter().all(|s| s.committed));
        index.force_merge().unwrap();
        let merged = index.reader_view().segment_stats();
        assert_eq!(merged.iter().filter(|s| s.committed && !s.search).count(), 2);
        assert_eq!(merged.iter().filter(|s| s.search && !s.committed).count(), 1);
        assert_eq!(merged.iter().filter(|s| s.committed).count(), committed.len());
        index.commit().unwrap();
        assert!(index.reader_view().segment_stats().iter().all(|s| s.committed && s.search));
        let reopened = Index::open_in(index.directory.clone()).unwrap();
        assert_eq!(reopened.reader_view().segment_stats().iter().filter(|s| s.committed).count(), index.reader_view().num_leaves());
    }

    #[test]
    fn merged_away_files_go_without_a_commit_unless_the_commit_lists_them() {
        let mut index = Index::new("sweep", vec![Field::new(0, "n", FieldType::Text)], IndexSettings::default());
        let add = |index: &mut Index, n: usize| for _ in 0..n {
            let mut doc = Document::new(); doc.add_text(0, "one");
            index.add_document(&doc).unwrap(); index.refresh().unwrap();
        };
        let ids = |index: &Index| index.reader_view().segment_stats().iter().filter(|s| s.search).map(|s| s.id).collect::<Vec<u32>>();
        let on_disk = |index: &Index, id: u32| index.directory.exists(&crate::codec::segment_file_name(id));
        add(&mut index, 2);
        index.commit().unwrap();
        let committed = ids(&index);
        assert_eq!(index.merged_away_committed_bytes(), 0);
        index.force_merge().unwrap();
        // Committed inputs stay (the commit on disk lists them) and are
        // reported, until the next commit.
        assert!(index.merged_away_committed_bytes() > 0);
        assert_eq!(index.retired_sweep()().unwrap(), 0);
        assert!(committed.iter().all(|&id| on_disk(&index, id)));
        // Segments written and merged away since the commit go at once.
        add(&mut index, 3);
        let fresh: Vec<u32> = ids(&index).into_iter().filter(|id| !committed.contains(id)).collect();
        assert!(fresh.len() >= 3 && fresh.iter().all(|&id| on_disk(&index, id)));
        index.force_merge().unwrap();
        let live = ids(&index);
        assert_eq!(live.len(), 1);
        let gone: Vec<u32> = fresh.iter().copied().filter(|id| !live.contains(id)).collect();
        assert!(!gone.is_empty());
        assert!(index.retired_sweep()().unwrap() >= gone.len());
        assert!(gone.iter().all(|&id| !on_disk(&index, id)), "{gone:?}");
        assert!(committed.iter().all(|&id| on_disk(&index, id)));
        index.commit().unwrap();
        assert_eq!(index.merged_away_committed_bytes(), 0);
        assert!(committed.iter().all(|&id| !on_disk(&index, id)));
    }

    #[test]
    fn readers_pin_merged_away_files_until_dropped() {
        let mut index = Index::new("pin", vec![Field::new(0, "n", FieldType::Text)], IndexSettings::default());
        for _ in 0..2 {
            let mut doc = Document::new(); doc.add_text(0, "one");
            index.add_document(&doc).unwrap(); index.refresh().unwrap();
        }
        let held = index.reader_view();
        let inputs: Vec<u32> = held.segment_stats().iter().filter(|s| s.search).map(|s| s.id).collect();
        index.force_merge().unwrap();
        assert_eq!(index.retired_sweep()().unwrap(), 0);
        assert!(inputs.iter().all(|&id| index.directory.exists(&crate::codec::segment_file_name(id))));
        drop(held);
        assert_eq!(index.retired_sweep()().unwrap(), inputs.len());
        assert!(inputs.iter().all(|&id| !index.directory.exists(&crate::codec::segment_file_name(id))));
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "price", FieldType::F64),
            Field::new(2, "category", FieldType::Keyword),
        ];

        {
            let mut index = Index::create(dir.path(), "persist", schema.clone(), IndexSettings::default()).unwrap();
            for i in 0..50 {
                let mut doc = Document::new();
                doc.add_text(0, &format!("item {i} widgets"));
                doc.add_f64(1, i as f64 * 1.5);
                doc.add_keyword(2, if i % 2 == 0 { "even" } else { "odd" });
                index.add_document(&doc).unwrap();
            }
            index.refresh().unwrap();
            index.delete_doc(&DocAddress::new(0, 4)).unwrap();
            index.commit().unwrap();
        }

        let index = Index::open(dir.path()).unwrap();
        assert_eq!(index.name, "persist");
        assert_eq!(index.num_docs(), 50);
        assert_eq!(index.num_live_docs(), 49);
        assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 100).unwrap().total_hits, 49);
        let doc = index.get_doc(&DocAddress::new(0, 7)).unwrap();
        assert_eq!(doc.get(0).unwrap().as_text(), Some("item 7 widgets"));
        assert_eq!(doc.get(1).unwrap().as_f64(), Some(10.5));
    }

    /// `node.store.allow_mmap: false`: indexing, refresh, deletes, a
    /// merge, search and reopening from the commit point all work on a
    /// directory that reads files into memory, and no store file is mapped.
    #[test]
    fn unmapped_directory_index_roundtrip() {
        use crate::codec::directory::MmapDirectory;
        let dir = tempfile::tempdir().unwrap();
        let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "price", FieldType::F64)];
        {
            let directory: Arc<dyn Directory> = Arc::new(MmapDirectory::open_with(dir.path(), false).unwrap());
            let mut index = Index::create_in(directory, "unmapped", schema.clone(), IndexSettings::default()).unwrap();
            for batch in 0..3 {
                for i in 0..200 {
                    let mut doc = Document::new();
                    doc.add_text(0, &format!("item {} widgets b{batch}", batch * 200 + i));
                    doc.add_f64(1, (batch * 200 + i) as f64);
                    index.add_document(&doc).unwrap();
                }
                index.refresh().unwrap();
            }
            index.delete_doc(&DocAddress::new(0, 4)).unwrap();
            index.commit().unwrap();
            index.force_merge().unwrap();
            index.commit().unwrap();
            assert_eq!(index.num_live_docs(), 599);
            assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 1000).unwrap().total_hits, 599);
            assert_eq!(index.search_term(&Term::new_text(0, "b2"), 1000).unwrap().total_hits, 200);
            #[cfg(target_os = "linux")]
            assert_eq!(crate::codec::directory::mapped_paths_under(dir.path()), Vec::<String>::new());
        }
        // Recovery: reopen from the commit point, still unmapped.
        let directory: Arc<dyn Directory> = Arc::new(MmapDirectory::open_with(dir.path(), false).unwrap());
        let index = Index::open_in(directory).unwrap();
        assert_eq!(index.num_live_docs(), 599);
        assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 1000).unwrap().total_hits, 599);
        let hit = index.search_term(&Term::new_text(0, "b1"), 1000).unwrap();
        assert_eq!(hit.total_hits, 200);
        #[cfg(target_os = "linux")]
        assert_eq!(crate::codec::directory::mapped_paths_under(dir.path()), Vec::<String>::new());
        // The same files through a mapping directory give the same answers.
        let mapped = Index::open_in(Arc::new(MmapDirectory::open_with(dir.path(), true).unwrap())).unwrap();
        assert_eq!(mapped.search_term(&Term::new_text(0, "widgets"), 1000).unwrap().total_hits, 599);
        #[cfg(target_os = "linux")]
        assert!(!crate::codec::directory::mapped_paths_under(dir.path()).is_empty());
    }

    /// `node.store.allow_mmap: false` with a merge whose output would go
    /// over the memory limit: the merge fails with the store file refusal,
    /// the index keeps serving its inputs, a reopen recovers it, and the
    /// same merge succeeds once the limit allows it.
    #[test]
    fn unmapped_merge_over_the_limit_fails_cleanly() {
        use crate::codec::directory::{MmapDirectory, StoreFileRefusal};
        static BUDGET: std::sync::OnceLock<Arc<crate::admission::Budget>> = std::sync::OnceLock::new();
        fn budget() -> Arc<crate::admission::Budget> { BUDGET.get_or_init(|| crate::admission::Budget::new(1 << 30)).clone() }
        fn admit(bytes: u64, file: &str) -> std::result::Result<Box<dyn Send>, StoreFileRefusal> {
            let b = budget();
            b.reserve(bytes, file).map(|r| Box::new(r) as Box<dyn Send>).map_err(|e| StoreFileRefusal { limit: b.limit(), reason: e.to_string() })
        }
        let dir = tempfile::tempdir().unwrap();
        let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "price", FieldType::F64)];
        let open = || -> Arc<dyn Directory> { Arc::new(MmapDirectory::open_in_memory(dir.path(), admit).unwrap()) };
        let mut index = Index::create_in(open(), "unmapped-merge", schema, IndexSettings::default()).unwrap();
        for batch in 0..4 {
            for i in 0..500 {
                let mut doc = Document::new();
                doc.add_text(0, &format!("item {} widgets b{batch} {:x}", batch * 500 + i, (batch * 500 + i) as u64 * 0x9E37_79B9_7F4A_7C15));
                doc.add_f64(1, (batch * 500 + i) as f64);
                index.add_document(&doc).unwrap();
            }
            index.refresh().unwrap();
        }
        index.commit().unwrap();
        assert!(index.sealed_segment_ids().len() >= 2, "a merge needs several segments");
        // Room for what is held now, not for a merged copy of it.
        let held = budget().used();
        assert!(held > 0, "the open segments are charged");
        budget().set_limit(held + held / 2);
        let err = index.force_merge().unwrap_err();
        assert!(err.to_string().contains("[node.store.allow_mmap] is false"), "{err}");
        assert!(err.to_string().contains("[vm.max_map_count]"), "{err}");
        assert!(budget().used() <= held + held / 2);
        let on_disk = |id: u32| dir.path().join(crate::codec::segment_file_name(id)).exists();
        assert!(!on_disk(index.next_segment_id - 1), "the refused merge output is removed");
        // Still serving the inputs.
        assert_eq!(index.num_live_docs(), 2000);
        assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 5000).unwrap().total_hits, 2000);
        assert_eq!(index.search_term(&Term::new_text(0, "b3"), 5000).unwrap().total_hits, 500);
        index.commit().unwrap();
        drop(index);
        // Recovery from the commit point (the failed merge left none of its
        // output referenced), still within the lowered limit.
        let mut index = Index::open_in(open()).unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 5000).unwrap().total_hits, 2000);
        // With room, the merge goes through.
        budget().set_limit(1 << 30);
        index.force_merge().unwrap();
        index.commit().unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "widgets"), 5000).unwrap().total_hits, 2000);
        assert_eq!(index.sealed_segment_ids().len(), 1);
    }

    /// Numeric range filters (points index -> bitset, `FilteredScorer`
    /// over the bulk paths) against brute force, on sealed segments and
    /// the memtable, with inclusive / exclusive / unbounded bounds.
    #[test]
    fn range_filters_match_brute_force_on_every_path() {
        use crate::query::query::{Bound, Query};
        use crate::query::searcher::{SearchOptions, Searcher};
        let mut index = test_index();
        let mut prices = Vec::new();
        let mut x = 12345u64;
        for seg in 0..3 {
            for i in 0..700 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let p = match i % 50 {
                    0 => -0.0,
                    1 => f64::INFINITY,
                    _ => (x % 2000) as f64 / 10.0,
                };
                let mut d = Document::new();
                let words = ["alpha", "beta", "gamma", "delta"];
                d.add_text(1, &format!("{} {} filler{}", words[i % 4], words[(i / 4) % 4], i % 7));
                d.add_f64(2, p);
                index.add_document(&d).unwrap();
                prices.push((seg, p, i));
            }
            if seg < 2 {
                index.refresh().unwrap(); // the last batch stays in the memtable
            }
        }
        index.delete_doc(&DocAddress::new(0, 3)).unwrap();
        let leaves = index.leaves();
        let searcher = Searcher::new(leaves.clone());
        let text = Query::bool_query().should(Query::term(1, "alpha")).should(Query::term(1, "gamma")).build();
        let in_range = |p: f64, lo: Bound<f64>, hi: Bound<f64>| {
            (match lo {
                Bound::Included(l) => p >= l,
                Bound::Excluded(l) => p > l,
                Bound::Unbounded => true,
            }) && (match hi {
                Bound::Included(h) => p <= h,
                Bound::Excluded(h) => p < h,
                Bound::Unbounded => true,
            })
        };
        let ranges = [
            (Bound::Unbounded, Bound::Included(20.0)),
            (Bound::Excluded(0.0), Bound::Excluded(150.0)),
            (Bound::Included(-0.0), Bound::Included(0.0)),
            (Bound::Included(50.0), Bound::Unbounded),
            (Bound::Excluded(f64::INFINITY), Bound::Unbounded),
            (Bound::Unbounded, Bound::Unbounded),
            (Bound::Included(10.0), Bound::Included(5.0)),
        ];
        for (lo, hi) in ranges {
            let range = Query::RangeF64 { field: 2, lower: lo, upper: hi };
            let filtered = Query::bool_query().must(text.clone()).filter(range.clone()).build();
            // Brute force over (leaf, doc).
            let mut want_range = Vec::new();
            let mut want_text = Vec::new();
            for (li, leaf) in leaves.iter().enumerate() {
                for d in 0..leaf.num_docs() {
                    if !(leaf.live_docs())(d) {
                        continue;
                    }
                    let (_, p, i) = prices[li * 700 + d as usize];
                    if in_range(p, lo, hi) {
                        want_range.push((li, d));
                        // "alpha" / "gamma" as the first or second word.
                        if i % 2 == 0 || (i / 4) % 2 == 0 {
                            want_text.push((li, d));
                        }
                    }
                }
            }
            for (q, want) in [(&range, &want_range), (&filtered, &want_text)] {
                let all = SearchOptions { top_k: 5000, ..Default::default() };
                let td = searcher.search(q, &all);
                let mut got: Vec<(usize, u32)> = td.hits.iter().map(|h| (h.leaf, h.doc)).collect();
                got.sort_unstable();
                assert_eq!(&got, want, "{lo:?}..{hi:?}");
                assert_eq!(td.total_hits as usize, want.len());
                // Bulk top-k (filter folded into the deleted docs) vs the
                // per-doc path (min_score disables bulk): same hits, same count.
                let bulk = searcher.search(q, &SearchOptions { top_k: 10, ..Default::default() });
                let per_doc = searcher.search(q, &SearchOptions { top_k: 10, min_score: Some(f32::MIN), ..Default::default() });
                let key = |t: &crate::query::searcher::TopDocs| t.hits.iter().map(|h| (h.leaf, h.doc, h.score.to_bits())).collect::<Vec<_>>();
                assert_eq!(key(&bulk), key(&per_doc), "{lo:?}..{hi:?}");
                assert_eq!(bulk.total_hits, per_doc.total_hits);
                // k-NN-style pre-filter bitsets agree too.
                if std::ptr::eq(q, &range) {
                    let ctx = crate::query::query::SearchContext::for_query(q, &leaves, Bm25Params::default());
                    let mut n = 0;
                    for (li, leaf) in leaves.iter().enumerate() {
                        let bits = q.filter_bits(leaf, &ctx).unwrap();
                        for d in 0..leaf.num_docs() {
                            if (bits[d as usize / 64] >> (d % 64)) & 1 == 1 && (leaf.live_docs())(d) {
                                assert!(want.contains(&(li, d)));
                                n += 1;
                            }
                        }
                    }
                    assert_eq!(n, want.len());
                }
            }
        }
        // Integer bounds at the extremes.
        let none = Query::RangeI64 { field: 2, lower: Bound::Excluded(i64::MAX), upper: Bound::Unbounded };
        assert_eq!(searcher.search(&none, &SearchOptions::default()).total_hits, 0);
    }

    /// `_score` is Lucene's BM25 (no `(k1 + 1)` factor) with Lucene's field
    /// `docCount` (docs that have the field) as `N` and in `avgdl`.
    #[test]
    fn bm25_score_is_lucene_with_field_doc_count() {
        use crate::query::query::Query;
        use crate::query::searcher::{SearchOptions, Searcher};
        let mut index = test_index();
        let bodies = ["red fox", "red red fox jumps", "blue whale", "red"];
        for (i, b) in bodies.iter().enumerate() {
            let mut d = Document::new();
            d.add_text(0, &format!("title {i}"));
            d.add_text(1, b);
            index.add_document(&d).unwrap();
        }
        // Docs without `body`: they must not count in body's docCount/avgdl.
        for i in 0..6 {
            let mut d = Document::new();
            d.add_text(0, &format!("only a title {i}"));
            index.add_document(&d).unwrap();
        }
        for refresh in [false, true] {
            if refresh {
                index.refresh().unwrap();
            }
            let td = Searcher::new(index.leaves()).search(&Query::term(1, "red"), &SearchOptions::default());
            let (doc_count, df, sum_len) = (4.0f64, 3.0f64, 2.0 + 4.0 + 2.0 + 1.0);
            let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln() as f32;
            let avgdl = (sum_len / doc_count) as f32;
            let lucene = |freq: f32, dl: f32| {
                let inv = 1.0f32 / (1.2 * ((1.0 - 0.75) + 0.75 * dl / avgdl));
                idf - idf / (1.0 + freq * inv)
            };
            let want = [(1u32, lucene(2.0, 4.0)), (0, lucene(1.0, 2.0)), (3, lucene(1.0, 1.0))];
            let mut got: Vec<(u32, f32)> = td.hits.iter().map(|h| (h.doc, h.score)).collect();
            got.sort_by_key(|(d, _)| *d);
            let mut want = want.to_vec();
            want.sort_by_key(|(d, _)| *d);
            assert_eq!(got.len(), 3);
            for ((gd, gs), (wd, ws)) in got.iter().zip(&want) {
                assert_eq!(gd, wd);
                assert_eq!(gs.to_bits(), ws.to_bits(), "doc {gd}: {gs} vs Lucene {ws} (refresh={refresh})");
            }
        }
    }

    /// A segment none of whose docs has `body` (sealed before any doc had
    /// it, like the deleted older versions of re-indexed documents) adds
    /// nothing to body's `docCount`, as a Lucene segment without the field:
    /// `N` must not depend on how many such segments are still unmerged.
    #[test]
    fn bm25_doc_count_ignores_segments_without_the_field() {
        use crate::query::query::Query;
        use crate::query::searcher::{SearchOptions, Searcher};
        let mut index = test_index();
        for i in 0..6 {
            let mut d = Document::new();
            d.add_text(0, &format!("only a title {i}"));
            index.add_document(&d).unwrap();
        }
        index.refresh().unwrap();
        for b in ["red fox", "red red fox jumps", "blue whale", "red"] {
            let mut d = Document::new();
            d.add_text(1, b);
            index.add_document(&d).unwrap();
        }
        for refresh in [false, true] {
            if refresh {
                index.refresh().unwrap();
            }
            assert!(index.leaves().len() >= 2, "the title-only docs are a leaf of their own");
            let td = Searcher::new(index.leaves()).search(&Query::term(1, "red"), &SearchOptions::default());
            let (doc_count, df, sum_len) = (4.0f64, 3.0f64, 2.0 + 4.0 + 2.0 + 1.0);
            let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln() as f32;
            let avgdl = (sum_len / doc_count) as f32;
            let lucene = |freq: f32, dl: f32| {
                let inv = 1.0f32 / (1.2 * ((1.0 - 0.75) + 0.75 * dl / avgdl));
                idf - idf / (1.0 + freq * inv)
            };
            let mut want = vec![lucene(2.0, 4.0).to_bits(), lucene(1.0, 2.0).to_bits(), lucene(1.0, 1.0).to_bits()];
            let mut got: Vec<u32> = td.hits.iter().map(|h| h.score.to_bits()).collect();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want, "scores with body's docCount 4 (refresh={refresh})");
        }
    }

    #[test]
    fn test_parallel_writers_searchable_and_flush() {
        let mut index = test_index();
        let docs: Vec<Document> = (0..1000)
            .map(|i| {
                let mut d = Document::new();
                d.add_text(0, &format!("shared word{} n{i}", i % 10));
                d.add_f64(2, i as f64);
                d.add_keyword(4, if i % 2 == 0 { "even" } else { "odd" });
                d
            })
            .collect();
        let addrs = index.add_documents_parallel(&docs, 4).unwrap();
        assert_eq!(addrs.len(), 1000);
        assert_eq!(index.leaves().len(), 4);
        assert_eq!(index.num_pending_docs(), 1000);
        // Every memtable leaf is searchable and addresses resolve.
        assert_eq!(index.search_term(&Term::new_text(0, "shared"), 10).unwrap().total_hits, 1000);
        assert_eq!(index.search_term(&Term::new_text(4, "even"), 10).unwrap().total_hits, 500);
        for (i, a) in addrs.iter().enumerate().step_by(97) {
            let d = index.get_doc(a).unwrap();
            assert_eq!(d.get(2).unwrap().as_f64(), Some(i as f64));
        }
        assert_eq!(index.search_range(&RangeQuery::range_f64(2, Some(100.0), Some(199.0)), 1000).unwrap().total_hits, 100);
        index.delete_doc(&addrs[999]).unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "shared"), 10).unwrap().total_hits, 999);
        // A second batch reuses the same memtables.
        index.add_documents_parallel(&docs[..100], 4).unwrap();
        assert_eq!(index.search_term(&Term::new_text(0, "shared"), 10).unwrap().total_hits, 1099);
        index.refresh().unwrap();
        assert_eq!(index.num_segments(), 4);
        assert_eq!(index.num_docs(), 1100);
        assert_eq!(index.num_live_docs(), 1099);
        assert_eq!(index.search_term(&Term::new_text(0, "word3"), 10).unwrap().total_hits, 110);
        let ids: Vec<u32> = index.segments().iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
        index.add_document(&docs[0]).unwrap();
        index.refresh().unwrap();
        assert_eq!(index.segments().last().unwrap().id, 4);
        index.force_merge().unwrap();
        assert_eq!(index.num_docs(), 1100);
        assert_eq!(index.search_term(&Term::new_text(0, "shared"), 10).unwrap().total_hits, 1100);
    }

    fn nrt_index() -> Index {
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "rank", FieldType::I64),
            Field::new(2, "emb", FieldType::Vector { dim: 2 }).with_similarity("l2_norm"),
            Field::new(3, "name", FieldType::Keyword),
        ];
        let mut index = Index::new("nrt", schema, IndexSettings::default());
        index.set_nrt(true);
        index
    }

    fn nrt_doc(word: &str, rank: i64) -> Document {
        let mut d = Document::new();
        d.add_text(0, word);
        d.add_i64(1, rank);
        d.add_vector(2, vec![rank as f32, 0.0]);
        d
    }

    fn hits(index: &Index, word: &str) -> u64 {
        index.search_term(&Term::new_text(0, word), 100).unwrap().total_hits
    }

    fn live_latest(index: &Index) -> u32 {
        index.all_leaves().iter().map(|l| l.num_live_docs_latest()).sum()
    }

    #[test]
    fn segment_stats_match_enumeration_across_refresh_and_seal() {
        let mut index = nrt_index();
        for n in 0..3 {
            index.add_document(&nrt_doc("foo", n)).unwrap();
            index.publish().unwrap();
        }
        let before = index.reader_view();
        assert_eq!(before.segment_doc_counts_for_stats(), vec![1, 1, 1]);
        assert_eq!(before.segments_for_stats(), 3);
        index.add_document(&nrt_doc("foo", 4)).unwrap();
        index.seal().unwrap();
        let after = index.reader_view();
        assert_eq!(after.segment_doc_counts_for_stats().len(), after.segments_for_stats());
        assert_eq!(after.segment_doc_counts_for_stats().iter().sum::<u64>(), 4);
        assert_eq!(after.segments_for_stats(), after.num_leaves() + 1);
        assert_eq!(before.segment_doc_counts_for_stats(), vec![1, 1, 1], "retained snapshot stays unchanged");
    }

    #[test]
    fn nrt_writes_are_invisible_until_publish() {
        let mut index = nrt_index();
        let a = index.add_document(&nrt_doc("foo", 1)).unwrap();
        assert_eq!(hits(&index, "foo"), 0);
        assert!(index.leaves().is_empty());
        assert_eq!(index.all_leaves().len(), 1);
        assert_eq!(index.get_doc(&a).unwrap().get(1).and_then(|v| v.as_i64()), Some(1));
        assert!(index.has_unpublished());
        assert!(index.publish().unwrap());
        assert!(!index.has_unpublished());
        assert_eq!(hits(&index, "foo"), 1);
        // A pending delete hides the doc from realtime reads only.
        index.delete_doc(&a).unwrap();
        assert_eq!(hits(&index, "foo"), 1);
        assert_eq!(live_latest(&index), 0);
        assert!(index.all_leaves()[0].is_deleted_latest(a.doc_id));
        assert!(!index.leaves()[0].is_deleted(a.doc_id));
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 0);
    }

    #[test]
    fn nrt_seal_keeps_visibility_and_vectors_hidden() {
        let mut index = nrt_index();
        index.add_document(&nrt_doc("foo", 1)).unwrap();
        index.publish().unwrap();
        let b = index.add_document(&nrt_doc("foo", 2)).unwrap();
        // Flush: the published memtable becomes a searchable segment, the
        // unpublished one a hidden segment.
        index.seal().unwrap();
        assert_eq!(index.segments().len(), 1);
        assert_eq!(index.hidden_segments().len(), 1);
        assert_eq!(hits(&index, "foo"), 1);
        let knn = |index: &Index| index.knn_search(&KnnRequest::new(2, vec![2.0, 0.0], 10)).unwrap().len();
        assert_eq!(knn(&index), 1);
        // The hidden segment's docs are still found by realtime reads.
        assert_eq!(live_latest(&index), 2);
        assert_eq!(b, DocAddress::new(1, 0));
        assert_eq!(index.get_doc(&DocAddress::new(1, 0)).unwrap().get(1).and_then(|v| v.as_i64()), Some(2));
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 2);
        assert_eq!(knn(&index), 2);
        // Pending deletes keep vectors searchable until published.
        index.delete_doc(&DocAddress::new(0, 0)).unwrap();
        assert_eq!(knn(&index), 2);
        index.publish().unwrap();
        assert_eq!(knn(&index), 1);
        assert_eq!(hits(&index, "foo"), 1);
    }

    fn text_doc(word: &str, rank: i64) -> Document {
        let mut d = Document::new();
        d.add_text(0, word);
        d.add_i64(1, rank);
        d
    }

    /// Address (in `all_leaves` order) of the live doc with `rank`.
    fn find_rank(index: &Index, rank: i64) -> Option<DocAddress> {
        for (i, l) in index.all_leaves().iter().enumerate() {
            for d in 0..l.num_docs() {
                if !l.is_deleted_latest(d) && l.i64_value(1, d) == Some(rank) {
                    return Some(DocAddress::new(i as u32, d));
                }
            }
        }
        None
    }

    /// Several published memtables sealed by one job become one segment,
    /// with each memtable's deletes (including ones taken while it encoded)
    /// in place.
    #[test]
    fn refused_seal_claim_keeps_memtables_unclaimed_and_retryable() {
        let mut index=nrt_index();
        index.add_document(&text_doc("foo",1)).unwrap();index.publish().unwrap();
        let budget=crate::admission::Budget::new(1);
        let mems=index.frozen.iter().map(|w|(w.clone(),true)).collect();
        assert!(matches!(index.claim_seal_with_budget(mems,Index::COALESCE_MAX_BYTES,&budget),Err(RusticError::Memory(_))));
        assert!(!index.seal_in_flight());assert_eq!(budget.used(),0);
        budget.set_limit(1<<20);
        let mems=index.frozen.iter().map(|w|(w.clone(),true)).collect();
        let job=index.claim_seal_with_budget(mems,Index::COALESCE_MAX_BYTES,&budget).unwrap().unwrap();
        assert!(budget.used()>0);
        let output=job.run();assert!(budget.used()>0);
        index.finish_seal(output).unwrap();assert_eq!(budget.used(),0);
        assert_eq!(index.num_live_docs(),1);
    }

    /// A refused install of one output of a multi-output (non-coalesced)
    /// seal job releases every claim of the job: the refused memtable stays
    /// a memtable, the other outputs install, and later seals, sorted-free
    /// refreshes and commits proceed.
    #[test]
    fn refused_install_of_one_seal_output_strands_no_claims() {
        let mut index = nrt_index();
        index.set_preserve_published_leaves(true);
        for round in 0..3 {
            for i in 0..4 { index.add_document(&text_doc("foo", round * 4 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let gone = find_rank(&index, 5).unwrap();
        index.delete_doc(&gone).unwrap();
        let job = index.begin_seal_part(false, 8).unwrap().expect("a job");
        assert_eq!(job.memtable_ids().len(), 3);
        let sealed = job.run();
        assert!(sealed.coalesced.is_none() && sealed.done.len() == 3, "preserved leaves are not coalesced");
        crate::segment::Segment::refuse_next_seal_installs(1);
        assert!(matches!(index.finish_seal(sealed), Err(RusticError::Memory(_))));
        crate::segment::Segment::refuse_next_seal_installs(0);
        assert!(!index.seal_in_flight(), "no claim of the job may remain");
        assert!(index.sealing.is_empty());
        assert_eq!(index.segments().len(), 2, "the other outputs installed");
        assert_eq!(index.frozen.len(), 1, "the refused memtable stays a memtable");
        assert_eq!(live_latest(&index), 11);
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 11);
        // A later seal takes the refused memtable; commits work.
        let job = index.begin_seal_part(false, 8).unwrap().expect("the refused memtable is unclaimed");
        index.finish_seal(job.run()).unwrap();
        assert!(!index.seal_in_flight());
        assert!(index.frozen.is_empty());
        assert_eq!(index.segments().len(), 3);
        index.commit().unwrap();
        let reopened = Index::open_in(index.directory.clone()).unwrap();
        assert_eq!(hits(&reopened, "foo"), 11);
    }

    /// A seal install refused before it changed anything keeps the job's
    /// claims and encoded segments for a retry; abandoning it releases the
    /// claims and deletes the encoded files.
    #[test]
    fn refused_seal_install_keeps_encoded_work_for_retry_or_abandon() {
        let mut index = nrt_index();
        index.set_preserve_published_leaves(true);
        for round in 0..2 {
            for i in 0..3 { index.add_document(&text_doc("foo", round * 3 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let sealed = index.begin_seal_part(false, 8).unwrap().expect("a job").run();
        REFUSE_SEAL_INSTALL.with(|n| n.set(1));
        let refused = index.try_finish_seal_captured(sealed).err().expect("refused");
        assert!(matches!(refused.error, RusticError::Memory(_)));
        assert!(index.seal_in_flight(), "claims kept while the caller may retry");
        assert_eq!(index.frozen.len(), 2);
        index.try_finish_seal_captured(refused.sealed).ok().expect("admitted").unwrap();
        assert!(!index.seal_in_flight());
        assert_eq!(index.segments().len(), 2);
        assert_eq!(hits(&index, "foo"), 6);
        // Abandoned: claims released, encoded files gone, memtables stay.
        for i in 0..3 { index.add_document(&text_doc("bar", 10 + i)).unwrap(); }
        index.publish().unwrap();
        let sealed = index.begin_seal_part(false, 8).unwrap().expect("a job").run();
        let ids: Vec<u32> = sealed.done.iter().filter_map(|(_, seg)| seg.as_ref().ok().map(|(s, _)| s.id)).collect();
        assert!(!ids.is_empty());
        REFUSE_SEAL_INSTALL.with(|n| n.set(1));
        assert!(matches!(index.finish_seal(sealed), Err(RusticError::Memory(_))));
        assert!(!index.seal_in_flight());
        for id in ids { assert!(!index.directory.exists(&crate::codec::segment_file_name(id))); }
        assert_eq!(index.frozen.len(), 1);
        assert_eq!(hits(&index, "bar"), 3);
        let job = index.begin_seal_part(false, 8).unwrap().expect("retryable");
        index.finish_seal(job.run()).unwrap();
        assert!(index.frozen.is_empty());
        assert_eq!(hits(&index, "bar"), 3);
    }

    #[test]
    fn a_seal_job_writes_its_published_memtables_as_one_segment() {
        let mut index = nrt_index();
        for round in 0..3 {
            for i in 0..5 {
                index.add_document(&text_doc("foo", round * 5 + i)).unwrap();
            }
            index.publish().unwrap();
        }
        let gone = find_rank(&index, 2).unwrap();
        index.delete_doc(&gone).unwrap();
        index.publish().unwrap();
        let job = index.begin_seal_part(false, 8).unwrap().expect("a job");
        let late = find_rank(&index, 12).unwrap();
        index.delete_doc(&late).unwrap(); // taken while the job encodes
        let sealed = job.run();
        index.finish_seal(sealed).unwrap();
        assert_eq!(index.segments().len(), 1, "one segment for the job");
        assert_eq!(live_latest(&index), 13);
        assert_eq!(hits(&index, "foo"), 14, "the late delete is still pending for searches");
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 13);
    }

    /// A memtable's id filter never denies an id it holds, also for ids
    /// added after it grew, denies most others, and filters nothing else.
    #[test]
    fn memtable_unique_filter_has_no_false_negatives() {
        let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(5, "_id", FieldType::Keyword), Field::new(3, "name", FieldType::Keyword)];
        let mut index = Index::new("ids", schema, IndexSettings::default());
        index.set_nrt(true);
        let add = |index: &mut Index, i: u32| {
            let mut d = Document::new();
            d.add_keyword(5, &format!("id-{i}"));
            d.add_keyword(3, &format!("name-{i}"));
            index.add_document(&d).unwrap();
        };
        let check = |index: &Index, upto: u32| {
            let leaves = index.all_leaves();
            let writer = leaves.iter().find_map(|l| match l { LeafReader::Memtable(w, ..) => Some(*w), _ => None }).expect("a memtable");
            for i in 0..upto {
                assert!(writer.may_contain_unique_term(&Term::new_text(5, &format!("id-{i}"))), "id-{i} denied");
            }
            assert!(writer.may_contain_unique_term(&Term::new_text(3, "name-absent")), "other fields are never filtered");
            (upto..upto + 2000).filter(|i| writer.may_contain_unique_term(&Term::new_text(5, &format!("id-{i}")))).count()
        };
        for i in 0..100 { add(&mut index, i); }
        assert!(check(&index, 100) < 200, "the filter denies most absent ids");
        // Past several regrowths.
        for i in 100..20_000 { add(&mut index, i); }
        assert!(check(&index, 20_000) < 400);
    }

    /// Parked memtables (a copy no refresh publishes) sealed by one job
    /// become one hidden segment, deletes in place, invisible until a
    /// publish.
    #[test]
    fn a_seal_job_writes_its_parked_memtables_as_one_hidden_segment() {
        let mut index = nrt_index();
        for round in 0..3 {
            for i in 0..5 {
                index.add_document(&text_doc("foo", round * 5 + i)).unwrap();
            }
            let closed = index.close_writers();
            index.parked.extend(closed);
        }
        let gone = find_rank(&index, 7).unwrap();
        index.delete_doc(&gone).unwrap();
        let job = index.begin_seal_part(false, 8).unwrap().expect("a job");
        let sealed = job.run();
        index.finish_seal(sealed).unwrap();
        assert_eq!(index.hidden_segments().len(), 1, "one hidden segment for the job");
        assert!(index.parked.is_empty());
        assert_eq!(live_latest(&index), 14);
        assert_eq!(hits(&index, "foo"), 0, "parked documents stay invisible");
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 14);
    }

    /// A buffer write-out of many small memtables in both visibility
    /// classes (published by refreshes, parked by the write-out) seals one
    /// class per job, each job as one segment, instead of one segment per
    /// memtable.
    #[test]
    fn byte_bounded_seal_jobs_write_one_segment_per_visibility_class() {
        let mut index = nrt_index();
        // The owner's background jobs seal (a publish does not seal itself).
        index.set_auto_seal(false);
        for round in 0..20 {
            for i in 0..5 {
                index.add_document(&text_doc("foo", round * 5 + i)).unwrap();
            }
            index.publish().unwrap();
        }
        for round in 20..23 {
            for i in 0..5 {
                index.add_document(&text_doc("foo", round * 5 + i)).unwrap();
            }
            let closed = index.close_writers();
            index.parked.extend(closed);
        }
        let published = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("the published memtables");
        assert_eq!(published.memtable_ids().len(), 20);
        let parked = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("the parked memtables");
        assert_eq!(parked.memtable_ids().len(), 3);
        assert!(index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().is_none(), "everything is claimed");
        index.finish_seal(published.run()).unwrap();
        index.finish_seal(parked.run()).unwrap();
        assert_eq!(index.segments().len(), 1, "one segment for the published memtables");
        assert_eq!(index.hidden_segments().len(), 1, "one hidden segment for the parked ones");
        assert_eq!(hits(&index, "foo"), 100);
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 115);
        // A byte bound smaller than one memtable still takes one per job.
        for i in 0..5 { index.add_document(&text_doc("bar", 200 + i)).unwrap(); }
        index.publish().unwrap();
        for i in 0..5 { index.add_document(&text_doc("bar", 300 + i)).unwrap(); }
        index.publish().unwrap();
        assert_eq!(index.begin_seal_bytes(false, 1).unwrap().expect("one").memtable_ids().len(), 1);
    }

    /// G1: seal jobs claim disjoint memtable sets, encode concurrently and    /// G1: seal jobs claim disjoint memtable sets, encode concurrently and
    /// install independently, in either order.
    #[test]
    fn concurrent_seal_jobs_claim_disjoint_memtables_and_install_independently() {
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 {
                index.add_document(&text_doc("foo", round * 5 + i)).unwrap();
            }
            index.publish().unwrap();
        }
        let first = index.begin_seal_part(false, 2).unwrap().expect("first job");
        let second = index.begin_seal_part(false, 2).unwrap().expect("second job");
        let first_refs: Vec<_> = first.mems.iter().map(|(w, _)| Arc::downgrade(w)).collect();
        let second_refs: Vec<_> = second.mems.iter().map(|(w, _)| Arc::downgrade(w)).collect();
        let mut a = first.memtable_ids();
        let b = second.memtable_ids();
        assert_eq!((a.len(), b.len()), (2, 2));
        assert!(a.iter().all(|id| !b.contains(id)), "the jobs share a memtable: {a:?} {b:?}");
        assert!(index.begin_seal_part(false, 2).unwrap().is_none(), "everything is claimed");
        let (first, second) = std::thread::scope(|s| {
            let x = s.spawn(move || first.run());
            let y = s.spawn(move || second.run());
            (x.join().unwrap(), y.join().unwrap())
        });
        index.finish_seal(second).unwrap();
        assert!(index.seal_in_flight(), "the first job is still in flight");
        assert!(second_refs.iter().all(|w| w.upgrade().is_none()),
            "an installed chunk releases its memtables without waiting for other chunks");
        assert!(first_refs.iter().all(|w| w.upgrade().is_some()),
            "unfinished chunks retain writers for delete/visibility reconciliation");
        index.finish_seal(first).unwrap();
        assert!(!index.seal_in_flight());
        // Each job writes its two published memtables as one segment.
        assert_eq!(index.segments().len(), 2);
        assert_eq!(hits(&index, "foo"), 20);
        a.extend(b);
        assert_eq!(a.len(), 4);
    }

    /// A capture larger than the chunk bound is written memtable by
    /// memtable (no merge pass per chunk, encoded in parallel); a small one
    /// still coalesces.
    #[test]
    fn large_foreground_captures_coalesce_each_chunk() {
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 { index.add_document(&text_doc("foo", round * 5 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let one = index.frozen[0].estimated_bytes() as u64;
        let total: u64 = index.frozen.iter().map(|w| w.estimated_bytes() as u64).sum();
        // Two memtables per chunk, four in all: over the bound. Each chunk
        // is one segment (a single encode of the chunk, see
        // `union_one_pass`), as a Lucene DWPT of many requests is.
        let jobs = index.begin_seal_chunks(false, one * 2 + one / 2).unwrap();
        let chunks = jobs.len();
        assert!(chunks >= 2);
        assert!(jobs.iter().all(|job| job.coalesce_id.is_some()), "each chunk one segment");
        for job in jobs { index.finish_seal(job.run()).unwrap(); }
        assert_eq!(index.segments().len(), chunks);
        for i in 0..5 { index.add_document(&text_doc("bar", i)).unwrap(); index.publish().unwrap(); }
        let jobs = index.begin_seal_chunks(false, total * 4).unwrap();
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].coalesce_id.is_some(), "a small capture still coalesces");
        for job in jobs { index.finish_seal(job.run()).unwrap(); }
        assert_eq!(index.segments().len(), chunks + 1);
        assert_eq!(hits(&index, "foo") + hits(&index, "bar"), 25);
    }

    /// chaos seed 3: a large capture of thousands of
    /// memtables of a few documents each was written memtable by memtable
    /// (11,241 segments of 1-10 documents). They come from concurrent
    /// writers, each holding its own memtable (`checkout_writer`), closed
    /// together; a chunk that is such a pile is coalesced into one segment,
    /// by a flush too.
    #[test]
    fn a_pile_of_tiny_memtables_coalesces_per_chunk_in_a_large_capture() {
        let piles = MANY_MEMTABLES * 2;
        for flush in [false, true] {
            let mut index = nrt_index();
            // `piles` concurrent writers of two documents each.
            let mut writers = Vec::new();
            for round in 0..piles {
                let (mut w, schema) = index.checkout_writer();
                for i in 0..2 { w.add_document(&text_doc("foo", (round * 2 + i) as i64), &schema); }
                writers.push(w);
            }
            let bytes: Vec<u64> = writers.iter().map(|w| w.estimated_bytes() as u64).collect();
            for w in writers { index.checkin_writer(w); }
            let total: u64 = bytes.iter().sum();
            // Half the capture per chunk: over the bound.
            let bound = bytes.iter().take(MANY_MEMTABLES).sum::<u64>();
            assert!(total > bound);
            let jobs = if flush { index.begin_flush_seal_chunks(true, bound) } else { index.begin_seal_chunks(true, bound) }.unwrap();
            assert_eq!(jobs.iter().map(|j| j.mems.len()).sum::<usize>(), piles, "every writer is closed into the capture");
            let piled: Vec<_> = jobs.iter().filter(|job| job.mems.len() >= MANY_MEMTABLES).collect();
            assert!(!piled.is_empty(), "{:?}", jobs.iter().map(|j| j.mems.len()).collect::<Vec<_>>());
            assert!(piled.iter().all(|job| job.coalesce_id.is_some()), "a pile of tiny memtables coalesces");
            for job in jobs { index.finish_seal(job.run()).unwrap(); }
            index.publish().unwrap();
            assert!(index.segments().len() <= piles - MANY_MEMTABLES + 1, "{} segments", index.segments().len());
            assert_eq!(index.num_live_docs() as usize, piles * 2, "every document is in a segment");
        }
        // Ordinary bulk memtables are not a pile.
        assert!(pile_of_tiny_memtables(64 * 10, 64));
        assert!(!pile_of_tiny_memtables(64 * 500, 64));
        assert!(!pile_of_tiny_memtables(10, 10));
    }

    /// A refresh or flush cuts a large capture into as many coalesced
    /// chunks as encode at once (each one segment, one encode thread), so
    /// it waits for a quarter of the capture's encode rather than all of it
    /// (msmarco refresh-after-index: one 164 MB chunk took 2.3 s).
    #[test]
    fn foreground_captures_split_into_parallel_coalesced_chunks() {
        const MB: u64 = 1 << 20;
        assert_eq!(parallel_chunk_bytes(164 * MB, 128 * MB), 41 * MB, "four chunks encode at once");
        assert_eq!(parallel_chunk_bytes(10 * MB, 128 * MB), 16 * MB, "a small capture stays one chunk");
        assert_eq!(parallel_chunk_bytes(1024 * MB, 128 * MB), 128 * MB, "never past the bound");
        assert_eq!(parallel_chunk_bytes(10 * MB, 4 * MB), 4 * MB);
        // A coalesced job is one encode thread for the chunk scheduler.
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 { index.add_document(&text_doc("foo", round * 5 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let jobs = index.begin_seal_chunks(false, 1 << 30).unwrap();
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].coalesce_id.is_some());
        assert_eq!(jobs[0].encode_parallelism(), 1);
        for job in jobs { index.finish_seal(job.run()).unwrap(); }
        assert_eq!(hits(&index, "foo"), 20);
    }

    /// A flush writes a small capture's memtables as one segment in one
    /// encode (no merge pass inside the flush).
    #[test]
    fn a_flush_coalesces_a_small_capture_into_one_segment() {
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 { index.add_document(&text_doc("foo", round * 5 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let total: u64 = index.frozen.iter().map(|w| w.estimated_bytes() as u64).sum();
        let memtables = index.frozen.len();
        assert!(memtables >= 2);
        // A flush writes its memtables as one segment in one encode (no
        // merge pass), as Lucene flushes a DWPT: one segment per memtable
        // left dozens of tiny segments per flush under a bulk load.
        let jobs = index.begin_flush_seal_chunks(false, total * 4).unwrap();
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].coalesce_id.is_some(), "one segment for the flush");
        for job in jobs { index.finish_seal(job.run()).unwrap(); }
        assert_eq!(index.segments().len(), 1);
        assert!(memtables >= 2);
        assert_eq!(hits(&index, "foo"), 20);
    }

    /// c28f: a flush beside in-flight bulks failed with a 429 when its
    /// coalescing seal's merge was refused memory ("merge term dictionary
    /// growth"). Coalescing is optional: the memtables' own segments are
    /// installed instead, and nothing is lost.
    #[test]
    fn a_coalescing_seal_refused_memory_installs_each_memtable() {
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 { index.add_document(&text_doc("foo", round * 5 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let total: u64 = index.frozen.iter().map(|w| w.estimated_bytes() as u64).sum();
        let jobs = index.begin_seal_chunks(false, total * 4).unwrap();
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].coalesce_id.is_some(), "a small capture coalesces");
        // The write-and-merge path (the one-pass encode's fallback).
        NO_ONE_PASS_SORTED_SEAL.with(|n| n.set(true));
        REFUSE_SEAL_COALESCE.with(|n| n.set(1));
        for job in jobs { index.finish_seal(job.run()).unwrap(); }
        NO_ONE_PASS_SORTED_SEAL.with(|n| n.set(false));
        assert_eq!(REFUSE_SEAL_COALESCE.with(|n| n.get()), 0, "the coalescing merge was refused");
        assert_eq!(index.segments().len(), 4, "one segment per memtable");
        assert!(!index.seal_in_flight());
        assert_eq!(hits(&index, "foo"), 20);
        index.commit().unwrap();
    }

    #[test]
    fn foreground_chunks_share_one_capture_and_release_as_they_install() {
        let mut index = nrt_index();
        for round in 0..4 {
            for i in 0..5 { index.add_document(&text_doc("foo", round * 5 + i)).unwrap(); }
            index.publish().unwrap();
        }
        // A deliberately tiny bound selects one indivisible memtable per job.
        let jobs = index.begin_seal_chunks(true, 1).unwrap();
        assert_eq!(jobs.len(), 4);
        assert_eq!(jobs.iter().map(SealJob::num_docs).sum::<u64>(), 20);
        let refs: Vec<_> = jobs.iter().map(|job| Arc::downgrade(&job.mems[0].0)).collect();
        // These changes occur after the common capture, before any encoding.
        index.add_document(&text_doc("foo", 20)).unwrap();
        for id in [3, 18] { index.delete_doc(&find_rank(&index, id).unwrap()).unwrap(); }
        let mut captures = HashMap::new();
        for (i, job) in jobs.into_iter().enumerate() {
            captures.extend(index.finish_seal_captured(job.run()).unwrap());
            assert!(refs[i].upgrade().is_none(), "installed chunk must release without waiting for the tail");
            assert!(refs.iter().skip(i + 1).all(|w| w.upgrade().is_some()));
        }
        assert!(!index.seal_in_flight());
        assert_eq!(captures.len(), 4);
        assert!(captures.values().all(|mask| mask.count() == 0), "later deletes must not move the capture boundary");
        assert_eq!(index.segments().iter().map(|segment| segment.num_docs).sum::<u32>(), 20,
            "new writes must remain outside the captured jobs");
        assert_eq!(hits(&index, "foo"), 20, "pending deletes remain invisible until refresh");
        assert_eq!(live_latest(&index), 19);
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 19);
    }

    #[test]
    fn background_seal_keeps_deletes_and_visibility_taken_meanwhile() {
        let mut index = nrt_index();
        for i in 0..10 {
            index.add_document(&text_doc("foo", i)).unwrap();
        }
        index.publish().unwrap();
        for i in 10..20 {
            index.add_document(&text_doc("foo", i)).unwrap();
        }
        let job = index.begin_seal(true).unwrap().unwrap();
        assert_eq!(job.num_docs(), 20);
        assert!(index.seal_in_flight());
        assert!(index.begin_seal(true).unwrap().is_none(), "one seal job at a time");
        // Writes go to a fresh memtable meanwhile; the closed ones are
        // still realtime leaves that take deletes.
        index.add_document(&text_doc("foo", 20)).unwrap();
        assert_eq!(live_latest(&index), 21);
        let a = find_rank(&index, 3).unwrap();
        index.delete_doc(&a).unwrap(); // published memtable: pending delete
        let b = find_rank(&index, 15).unwrap();
        index.delete_doc(&b).unwrap(); // parked memtable
        assert_eq!(hits(&index, "foo"), 10);
        // A foreground seal leaves the job's memtables alone.
        index.seal_frozen().unwrap();
        assert!(index.segments().is_empty());
        let sealed = job.run();
        index.finish_seal(sealed).unwrap();
        assert!(!index.seal_in_flight());
        assert_eq!(index.segments().len(), 1);
        assert_eq!(index.hidden_segments().len(), 1);
        // Searches still see doc 3 (its delete is pending), realtime reads do not.
        assert_eq!(hits(&index, "foo"), 10);
        assert_eq!(live_latest(&index), 19);
        assert!(find_rank(&index, 3).is_none() && find_rank(&index, 15).is_none());
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 19);
        // Durable: a commit + reopen sees the sealed segments with their deletes.
        index.seal().unwrap();
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 19);
    }

    #[test]
    fn publish_during_background_seal_makes_parked_memtables_searchable() {
        let mut index = nrt_index();
        for i in 0..5 {
            index.add_document(&text_doc("foo", i)).unwrap();
        }
        let job = index.begin_seal(true).unwrap().unwrap();
        assert_eq!(hits(&index, "foo"), 0);
        index.add_document(&text_doc("foo", 5)).unwrap();
        index.publish().unwrap();
        // The parked memtable was published with the newer one.
        assert_eq!(hits(&index, "foo"), 6);
        let d = find_rank(&index, 2).unwrap();
        index.delete_doc(&d).unwrap();
        index.finish_seal(job.run()).unwrap();
        // Sealed as a searchable segment (it was published meanwhile).
        assert_eq!(index.segments().len(), 1);
        assert!(index.hidden_segments().is_empty());
        assert_eq!(hits(&index, "foo"), 6);
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 5);
        assert_eq!(index.segments()[0].num_live_docs(), 4);
    }

    #[test]
    fn detached_force_merge_keeps_old_readers_live_until_publication() {
        let mut index = nrt_index();
        for batch in 0..3 {
            for row in 0..10 { index.add_document(&text_doc("foo", batch * 10 + row)).unwrap(); }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        let deleted = find_rank(&index, 1).unwrap();
        index.delete_doc(&deleted).unwrap();
        index.publish().unwrap();
        let index = std::sync::Arc::new(parking_lot::RwLock::new(index));
        let job = index.write().begin_force_merge().unwrap().unwrap();
        // Pause a real merge before publication and prove readers can still
        // inspect the old segments; no latency threshold or huge fixture.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let merging = index.clone();
        let worker = std::thread::spawn(move || {
            let merged = job.run().unwrap();
            done_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            assert!(merging.write().finish_merge(job, merged).unwrap());
        });
        done_rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        {
            let old = index.try_read().expect("merge must not hold the index lock");
            assert_eq!(old.num_segments(), 3);
            assert_eq!(hits(&old, "foo"), 29);
        }
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        let merged = index.read();
        assert_eq!(merged.num_segments(), 1);
        assert_eq!(hits(&merged, "foo"), 29);
        assert!(find_rank(&merged, 1).is_none());
        for row in (0..30).filter(|row| *row != 1) { assert!(find_rank(&merged, row).is_some()); }
    }

    #[test]
    fn reader_views_stay_exact_and_pin_merged_away_files() {
        fn live(view: &crate::reader_view::ReaderView) -> u32 {
            view.leaves().iter().map(|l| l.num_live_docs()).sum()
        }
        let mut index = nrt_index();
        for batch in 0..3 {
            for row in 0..10 { index.add_document(&text_doc("foo", batch * 10 + row)).unwrap(); }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        // A published memtable as well as sealed segments.
        for row in 30..35 { index.add_document(&text_doc("foo", row)).unwrap(); }
        index.publish().unwrap();
        let view = index.reader_view();
        let retained = index.view_slot().retain().unwrap();
        assert_eq!((view.segments().len(), view.num_memtables()), (3, 1));
        assert_eq!((live(&view), live(&retained)), (35, 35));
        let old: Vec<u32> = view.segments().iter().map(|s| s.id).collect();
        // Deletes in a segment and in the published memtable, made visible,
        // then everything merged away and committed.
        for rank in [1, 31] {
            let addr = find_rank(&index, rank).unwrap();
            index.delete_doc(&addr).unwrap();
        }
        assert_eq!(live(&view), 35, "pending deletes are invisible");
        index.publish().unwrap();
        index.seal().unwrap();
        index.force_merge().unwrap();
        index.commit().unwrap();
        assert_eq!(live(&index.reader_view()), 33);
        assert_eq!(index.reader_view().segments().len(), 1);
        // Views taken before keep their deletions (segment and memtable
        // copies are frozen at publication) and their files.
        assert_eq!((live(&view), live(&retained)), (35, 35));
        assert_eq!(hits(&index, "foo"), 33);
        for id in &old {
            assert!(index.directory().exists(&crate::codec::segment_file_name(*id)), "segment {id} deleted under a live view");
        }
        for (leaf, l) in retained.leaves().iter().enumerate() {
            for d in (0..l.num_docs()).filter(|d| !l.is_deleted(*d)) {
                assert!(retained.get_doc(&DocAddress::new(leaf as u32, d)).is_ok());
            }
        }
        drop(view);
        drop(retained);
        index.commit().unwrap();
        for id in &old {
            assert!(!index.directory().exists(&crate::codec::segment_file_name(*id)), "segment {id} kept after its readers closed");
        }
    }

    #[test]
    fn background_merges_run_beside_a_force_merge_on_unclaimed_segments() {
        let mut index = nrt_index();
        let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
        let add_segment = |index: &mut Index, seg: i64| {
            for i in 0..10 {
                index.add_document(&text_doc("foo", seg * 10 + i)).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        };
        for seg in 0..3 {
            add_segment(&mut index, seg);
        }
        let force = index.begin_force_merge().unwrap().unwrap();
        let claimed: Vec<u32> = index.segments().iter().map(|s| s.id).collect();
        assert!(index.begin_merge(&policy).is_none(), "every segment is claimed by the force merge");
        assert!(index.begin_force_merge().unwrap().is_none(), "no second force merge over claimed segments");
        // Writes during the force merge: new segments, merged in the background.
        for seg in 3..7 {
            add_segment(&mut index, seg);
        }
        let background = index.begin_merge(&policy).unwrap();
        let picked: Vec<u32> = background.chosen.iter().map(|s| s.id).collect();
        assert!(picked.iter().all(|id| !claimed.contains(id)), "{picked:?} overlaps {claimed:?}");
        // A failed encoding releases its claim (and nothing is installed).
        index.release_merge(&background);
        let background = index.begin_merge(&policy).unwrap();
        let merged = background.run().unwrap();
        assert!(index.finish_merge(background, merged).unwrap());
        let merged = force.run().unwrap();
        assert!(index.finish_merge(force, merged).unwrap(), "the force merge's claimed inputs stayed in place");
        assert_eq!(hits(&index, "foo"), 70);
        assert!(index.segments().len() < 7);
        assert!(index.merging.is_empty(), "every claim released");
    }

    #[test]
    fn detached_sorted_seal_remaps_deletes_and_retries_with_retained_readers() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        for reverse in [false, true] {
            let mut index = nrt_index();
            index.set_index_sort(Some(IndexSort::new(vec![IndexSortField {
                field: 1, kind: IndexSortKind::Long, reverse, max_mode: false, missing_first: false,
            }])));
            for rank in [9, 2, 7, 1, 5] { index.add_document(&text_doc("foo", rank)).unwrap(); }
            let job = index.begin_seal(true).unwrap().expect("sorted seal detaches");
            assert!(index.publish().is_err(), "must not expose unsorted in-flight writers");
            index.delete_doc(&find_rank(&index, 7).unwrap()).unwrap();
            let done = job.run();
            // Deletion after encoding still needs the same ordinal map.
            index.delete_doc(&find_rank(&index, 2).unwrap()).unwrap();
            index.finish_seal(done).unwrap();
            assert!(!index.seal_in_flight());
            assert_eq!(hits(&index, "foo"), 0);
            index.publish().unwrap();
            assert_eq!(hits(&index, "foo"), 3);
            assert!(find_rank(&index, 2).is_none() && find_rank(&index, 7).is_none());
            index.commit().unwrap();
            let reopened = Index::open_in(index.directory.clone()).unwrap();
            assert_eq!(hits(&reopened, "foo"), 3);
        }
        // A captured reader is not a maintenance job. It cannot prevent seals.
        let mut index = nrt_index();
        index.add_document(&text_doc("foo", 1)).unwrap();
        index.publish().unwrap();
        let retained = index.frozen[0].clone();
        let job = index.begin_seal(false).unwrap().expect("reader Arc does not claim a seal");
        index.finish_seal(job.run()).unwrap();
        assert_eq!(retained.num_docs(), 1);
        assert_eq!(index.segments.len(), 1);
    }

    #[test]
    fn single_segment_expunge_is_detached_and_preserves_concurrent_changes() {
        let mut index = nrt_index();
        for i in 0..10 { index.add_document(&text_doc("foo", i)).unwrap(); }
        index.publish().unwrap();
        index.seal().unwrap();
        assert!(index.begin_force_merge().unwrap().is_none(), "clean single segment is a no-op");
        index.delete_doc(&find_rank(&index, 1).unwrap()).unwrap();
        index.publish().unwrap();
        let job = index.begin_force_merge().unwrap().expect("single-segment expunge must detach");
        assert!(index.begin_force_merge().unwrap().is_none(), "claimed input cannot be selected twice");
        index.delete_doc(&find_rank(&index, 2).unwrap()).unwrap();
        index.add_document(&text_doc("foo", 10)).unwrap();
        let merged = job.run().unwrap();
        assert_eq!(merged.num_docs, 9);
        assert!(index.finish_merge(job, merged).unwrap());
        assert_eq!(hits(&index, "foo"), 9, "pending delete and new writer remain unrefreshed");
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 9);
        assert!(find_rank(&index, 1).is_none() && find_rank(&index, 2).is_none());
        assert!(find_rank(&index, 10).is_some());
    }

    #[test]
    fn refused_merge_install_deletes_its_output_and_keeps_the_inputs_served() {
        let mut index = nrt_index();
        let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
        for seg in 0..4 {
            for i in 0..10 {
                index.add_document(&text_doc("foo", seg * 10 + i)).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        let before: Vec<u32> = index.segments().iter().map(|s| s.id).collect();
        let job = index.begin_merge(&policy).unwrap();
        let output = crate::codec::segment_file_name(job.new_id);
        let merged = job.run().unwrap();
        assert!(index.directory().exists(&output));
        // The install's view admission is refused: nothing is swapped, and
        // the encoded output does not linger until a restart's sweep.
        assert!(matches!(index.finish_merge_with_budget(job, merged, &crate::admission::Budget::new(0)), Err(RusticError::Memory(_))));
        assert!(!index.directory().exists(&output));
        assert_eq!(index.segments().iter().map(|s| s.id).collect::<Vec<_>>(), before);
        assert_eq!(hits(&index, "foo"), 40);
        // The released inputs merge again once admission allows it.
        let job = index.begin_merge(&policy).unwrap();
        let merged = job.run().unwrap();
        assert!(index.finish_merge(job, merged).unwrap());
        assert!(index.segments().len() < before.len());
        assert_eq!(hits(&index, "foo"), 40);
    }

    #[test]
    fn denied_backlog_planning_reports_the_last_measured_backlog() {
        let mut index = nrt_index();
        let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
        for seg in 0..4 {
            for i in 0..10 {
                index.add_document(&text_doc("foo", seg * 10 + i)).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        let measured = index.num_pending_merges_on(&policy, &crate::admission::Budget::new(1 << 20));
        assert!(measured > 0 && measured < index.segments().len(), "one merge of the four segments is due");
        // Denied planning neither inflates the backlog to every segment
        // (which throttles indexing) nor hides it.
        assert_eq!(index.num_pending_merges_on(&policy, &crate::admission::Budget::new(0)), measured);
    }

    #[test]
    fn backlog_denied_since_open_is_not_reported_as_merged() {
        let mut index = nrt_index();
        let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
        let denied = crate::admission::Budget::new(0);
        for seg in 0..4 {
            for i in 0..10 {
                index.add_document(&text_doc("foo", seg * 10 + i)).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        // Never measured: one pending merge keeps the scheduler retrying,
        // without counting every segment (which would throttle indexing).
        assert_eq!(index.num_pending_merges_on(&policy, &denied), 1);
        let measured = index.num_pending_merges_on(&policy, &crate::admission::Budget::new(1 << 20));
        assert_eq!(index.num_pending_merges_on(&policy, &denied), measured);
    }

    #[test]
    fn background_merge_carries_deletes_taken_meanwhile() {
        let mut index = nrt_index();
        let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
        for seg in 0..4 {
            for i in 0..10 {
                index.add_document(&text_doc("foo", seg * 10 + i)).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        assert_eq!(index.segments().len(), 4);
        // Deleted before the merge (visible): dropped by it.
        let d = find_rank(&index, 1).unwrap();
        index.delete_doc(&d).unwrap();
        index.publish().unwrap();
        let job = index.begin_merge(&policy).unwrap();
        // Deleted while merging: one published, one still pending.
        let a = find_rank(&index, 2).unwrap();
        index.delete_doc(&a).unwrap();
        index.publish().unwrap();
        let b = find_rank(&index, 3).unwrap();
        index.delete_doc(&b).unwrap();
        let merged = job.run().unwrap();
        assert!(index.finish_merge(job, merged).unwrap());
        assert_eq!(index.segments().len(), 2);
        assert_eq!(hits(&index, "foo"), 38, "pending delete still visible");
        assert!(find_rank(&index, 1).is_none() && find_rank(&index, 2).is_none() && find_rank(&index, 3).is_none());
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 37);
        // Every other doc survived, exactly once.
        for r in (0..40).filter(|r| ![1, 2, 3].contains(r)) {
            assert!(find_rank(&index, r).is_some(), "rank {r} lost");
        }
        assert_eq!(live_latest(&index), 37);
    }

    #[test]
    fn background_hidden_merge_preserves_refresh_and_delete_visibility() {
        for publish_during_merge in [false, true] {
            let mut index = nrt_index();
            let policy = crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() };
            // One visible segment must not be mixed into hidden merge inputs.
            index.add_document(&text_doc("visible", 100)).unwrap();
            index.publish().unwrap();
            index.seal().unwrap();
            for segment in 0..4 {
                for i in 0..10 { index.add_document(&text_doc("foo", segment * 10 + i)).unwrap(); }
                let seal = index.begin_seal(true).unwrap().unwrap();
                index.finish_seal(seal.run()).unwrap();
            }
            assert_eq!(index.hidden.len(), 4);
            assert_eq!(hits(&index, "foo"), 0);
            let job = index.begin_merge(&policy).expect("hidden segments need background compaction");
            assert!(job.chosen.iter().all(|s| index.hidden.iter().any(|h| Arc::ptr_eq(s, h))));
            let a = find_rank(&index, 2).unwrap();
            index.delete_doc(&a).unwrap();
            if publish_during_merge { index.publish().unwrap(); }
            let b = find_rank(&index, 3).unwrap();
            index.delete_doc(&b).unwrap();
            let merged = job.run().unwrap();
            assert!(index.finish_merge(job, merged).unwrap());
            assert_eq!(live_latest(&index), 39);
            assert_eq!(hits(&index, "visible"), 1);
            assert_eq!(hits(&index, "foo"), if publish_during_merge { 39 } else { 0 });
            assert_eq!(index.hidden.len(), if publish_during_merge { 0 } else { 2 });
            index.publish().unwrap();
            assert_eq!(hits(&index, "foo"), 38);
            for rank in 0..40 {
                assert_eq!(find_rank(&index, rank).is_some(), ![2, 3].contains(&rank), "rank {rank}");
            }
            assert_eq!(live_latest(&index), 39);
        }
    }

    #[test]
    fn nrt_pending_deletes_survive_seal_and_merge() {
        let mut index = nrt_index();
        let mut addrs = Vec::new();
        for i in 0..20 {
            addrs.push(index.add_document(&nrt_doc("foo", i)).unwrap());
        }
        index.publish().unwrap();
        index.delete_doc(&addrs[3]).unwrap();
        index.seal().unwrap();
        assert_eq!(hits(&index, "foo"), 20);
        index.add_document(&nrt_doc("foo", 99)).unwrap();
        index.publish().unwrap();
        index.seal().unwrap();
        index.delete_doc(&DocAddress::new(0, 5)).unwrap();
        let ids: Vec<u32> = index.segments().iter().map(|s| s.id).collect();
        index.merge_segments(&ids).unwrap();
        assert_eq!(index.segments().len(), 1);
        // doc 5 is pending: still visible after the merge.
        assert_eq!(hits(&index, "foo"), 20);
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 19);
    }

    #[test]
    fn detached_sorted_merge_preserves_concurrent_deletes_and_new_segments() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        for reverse in [false, true] {
            let mut index = nrt_index();
            index.set_index_sort(Some(IndexSort::new(vec![IndexSortField {
                field: 1, kind: IndexSortKind::Long, reverse, max_mode: false, missing_first: false,
            }])));
            for values in [[4, 1], [3, 2]] {
                for rank in values {
                    let mut doc = Document::new(); doc.add_text(0, "foo"); doc.add_i64(1, rank);
                    index.add_document(&doc).unwrap();
                }
                index.publish().unwrap();
            }
            let job = index.begin_force_merge().unwrap().expect("sorted merge must detach");
            let output = job.run().unwrap();
            let old_doc = (0..index.segments[1].num_docs).find(|&d| {
                LeafReader::Segment(&index.segments[1]).i64_value(1, d) == Some(3)
            }).unwrap();
            index.delete_doc(&DocAddress::new(1, old_doc)).unwrap();
            let mut doc = Document::new(); doc.add_text(0, "foo"); doc.add_i64(1, 5);
            index.add_document(&doc).unwrap();
            index.publish().unwrap();
            assert!(index.finish_merge(job, output).unwrap());
            let merged = LeafReader::Segment(&index.segments[0]);
            let ranks: Vec<_> = (0..merged.num_docs()).filter(|&d| !merged.is_deleted(d))
                .map(|d| merged.i64_value(1, d).unwrap()).collect();
            assert_eq!(ranks, if reverse { vec![4, 2, 1] } else { vec![1, 2, 4] });
            assert_eq!(hits(&index, "foo"), 4);
            index.commit().unwrap();
            assert_eq!(hits(&Index::open_in(index.directory.clone()).unwrap(), "foo"), 4);
        }
    }

    /// A coalesced seal (unsorted, and index-sorted with enough docs for the
    /// parallel sort) completes while every global rayon worker is blocked,
    /// as the ES layer's parallel bulk writes block in `wait_for_seals` for
    /// this very seal (joins track: bulks stalled for minutes while
    /// `sorted_union`'s joins and the sort queued on the global pool behind
    /// them). Its union, sort and encoding run on the encode pool.
    #[test]
    fn coalesced_seals_do_not_need_the_global_rayon_pool() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let _test_gate = crate::global_rayon_test_gate();
        for sorted in [false, true] {
            let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "rank", FieldType::I64)];
            let mut index = Index::new("pool", schema, IndexSettings::default());
            index.set_nrt(true);
            index.set_auto_seal(false);
            if sorted {
                index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: false, max_mode: false, missing_first: false }])));
            }
            for batch in 0..3i64 {
                for i in 0..2_000i64 { index.add_document(&text_doc("foo", (i * 7 + batch) % 6_000)).unwrap(); }
                let closed = index.close_writers();
                index.parked.extend(closed);
            }
            let job = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("a job");
            assert_eq!(job.memtable_ids().len(), 3);
            // Every global worker blocked on a lock this test holds.
            let lock = Arc::new(parking_lot::Mutex::new(()));
            let guard = lock.lock();
            let busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let n = rayon::current_num_threads();
            for _ in 0..n {
                let (lock, busy) = (lock.clone(), busy.clone());
                rayon::spawn(move || {
                    busy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    drop(lock.lock());
                });
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while busy.load(std::sync::atomic::Ordering::SeqCst) < n {
                if std::time::Instant::now() >= deadline {
                    drop(guard);
                    panic!("global Rayon blockers did not start");
                }
                std::thread::yield_now();
            }
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(job.run()); });
            let got = rx.recv_timeout(std::time::Duration::from_secs(20));
            drop(guard);
            let sealed = got.unwrap_or_else(|_| panic!("the {} coalesced seal waited on the global rayon pool", if sorted { "sorted" } else { "unsorted" }));
            assert!(sealed.coalesced.as_ref().is_some_and(|(_, r)| r.is_ok()), "one-pass coalesced seal (sorted: {sorted})");
            index.finish_seal(sealed).unwrap();
            index.publish().unwrap();
            assert_eq!(hits(&index, "foo"), 6_000);
        }
    }

    /// Index-sorted memtables coalesce: one job's parked memtables (ranks
    /// interleaved across them) merge-sort into one sorted segment, a
    /// delete taken before the job and one taken while it encodes land on
    /// the right documents (live deletions through the merge's doc map,
    /// capture-time ones too), and the vector rows follow their documents
    /// (kNN exact).
    #[test]
    fn coalesced_sorted_seals_merge_sort_their_memtables() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let mut index = nrt_index();
        index.set_auto_seal(false);
        index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: false, max_mode: false, missing_first: false }])));
        for batch in [[9, 3, 6], [1, 8, 4], [7, 2, 5]] {
            for r in batch { index.add_document(&nrt_doc("foo", r)).unwrap(); }
            let closed = index.close_writers();
            index.parked.extend(closed);
        }
        let before = find_rank(&index, 8).unwrap();
        index.delete_doc(&before).unwrap();
        let job = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("a job");
        assert_eq!(job.memtable_ids().len(), 3);
        let during = find_rank(&index, 3).unwrap();
        index.delete_doc(&during).unwrap();
        let at = index.finish_seal_captured(job.run()).unwrap();
        assert_eq!(index.hidden_segments().len(), 1, "one sorted segment for the three memtables");
        let seg = index.hidden_segments()[0].clone();
        let ranks: Vec<i64> = { let leaf = LeafReader::Segment(&seg); (0..seg.num_docs).map(|d| leaf.i64_value(1, d).unwrap()).collect() };
        assert_eq!(ranks, (1..=9).collect::<Vec<i64>>(), "the coalesced segment is not in sort order");
        let latest = seg.latest_deletions().unwrap();
        let deleted: Vec<i64> = (0..seg.num_docs).filter(|&d| latest.get(d)).map(|d| ranks[d as usize]).collect();
        assert_eq!(deleted, vec![3, 8], "a delete landed on another document");
        let captured = &at[&seg.id];
        let at_capture: Vec<i64> = (0..seg.num_docs).filter(|&d| captured.get(d)).map(|d| ranks[d as usize]).collect();
        assert_eq!(at_capture, vec![8], "capture-time deletions were not remapped through the sort");
        index.publish().unwrap();
        assert_eq!(hits(&index, "foo"), 7);
        let leaf = &index.leaves()[0];
        for d in 0..leaf.num_docs() {
            let r = leaf.i64_value(1, d).unwrap();
            assert_eq!(leaf.vector(2, d), Some(vec![r as f32, 0.0]), "rank {r}'s vector moved");
        }
        for (query, want) in [(6.0, 6), (3.2, 4), (8.1, 9)] {
            let top = index.knn_search(&KnnRequest::new(2, vec![query, 0.0], 1)).unwrap();
            assert_eq!(index.leaves()[top[0].leaf].i64_value(1, top[0].doc), Some(want), "query {query}");
        }
    }

    /// Coalescing (a): an index-sorted coalesced seal against a reference
    /// that seals each memtable on its own, with the same operations: a
    /// delete before the job, then a delete and an update (delete + new
    /// copy) of rows from different memtables while the job encodes. Doc
    /// order inside every segment, the live documents and the search
    /// results are identical.
    #[test]
    fn one_pass_sorted_seal_equals_the_two_pass_seal() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let feature = |token: &str, weight: f32| format!("{token}\u{0}{:08x}", weight.to_bits() & !0x7fff);
        let build = |one_pass: bool| -> Index {
            let schema = vec![
                Field::new(0, "title", FieldType::Text),
                Field::new(1, "rank", FieldType::I64),
                Field::new(3, "name", FieldType::Keyword),
                Field::new(4, "tags", FieldType::Keyword),
                Field::new(5, "score", FieldType::F64),
                Field::new(6, "flag", FieldType::Bool),
                Field::new(7, "feats", FieldType::Keyword),
            ];
            let mut index = Index::new("one-pass", schema, IndexSettings::default());
            index.set_nrt(true);
            index.set_auto_seal(false);
            index.set_index_sort(Some(IndexSort::new(vec![
                IndexSortField { field: 3, kind: IndexSortKind::Keyword, reverse: false, max_mode: false, missing_first: false },
                IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: true, max_mode: false, missing_first: false },
            ])));
            let words = ["quick brown fox", "lazy dog", "brown dog jumps", "the quick dog", "fox"];
            let mut n = 0i64;
            for batch in 0..4 {
                for i in 0..7 {
                    n += 1;
                    let mut d = Document::new();
                    d.add_text(0, words[(n as usize * 7 + batch) % words.len()]);
                    if n % 5 != 0 { d.add_text(0, "second value"); }
                    d.add_i64(1, (n * 37) % 11);
                    if n % 3 != 0 { d.add_keyword(3, ["b", "a", "c"][(n % 3) as usize]); }
                    // Multi-valued in some memtables only.
                    d.add_keyword(4, &format!("t{}", n % 4));
                    if batch % 2 == 0 && i % 2 == 0 { d.add_keyword(4, &format!("t{}", (n + 1) % 4)); }
                    if i != 3 { d.add_f64(5, n as f64 * 0.5); }
                    if n % 2 == 0 { d.add_bool(6, true); }
                    if i % 3 != 2 {
                        d.add_keyword(7, &feature("alpha", 1.5 + n as f32));
                        d.add_keyword(7, &feature("beta", 0.25));
                    }
                    index.add_document(&d).unwrap();
                }
                let closed = index.close_writers();
                index.parked.extend(closed);
            }
            let victim = find_rank(&index, 4).unwrap();
            index.delete_doc(&victim).unwrap();
            let job = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().unwrap();
            assert!(job.coalesces() && job.num_memtables() == 4);
            NO_ONE_PASS_SORTED_SEAL.with(|f| f.set(!one_pass));
            let sealed = job.run();
            NO_ONE_PASS_SORTED_SEAL.with(|f| f.set(false));
            index.finish_seal(sealed).unwrap();
            index.publish().unwrap();
            assert_eq!(index.segments().len(), 1);
            index
        };
        type Row = (bool, Vec<i64>, Vec<String>, Vec<String>, Option<f64>, Option<bool>, Vec<String>, Option<Vec<u8>>);
        let view = |index: &Index| -> (Vec<Row>, Vec<u64>, Vec<(u64, u64)>) {
            let leaf = &index.leaves()[0];
            let rows = (0..leaf.num_docs()).map(|d| (leaf.is_deleted(d), leaf.i64_values(1, d), leaf.keyword_values(3, d), leaf.keyword_values(4, d),
                leaf.f64_value(5, d), leaf.bool_value(6, d), leaf.keyword_values(7, d), leaf.get_doc(d).unwrap().map(|doc| bincode::serialize(&doc).unwrap()))).collect();
            let terms = ["quick", "dog", "fox", "second", "value"].iter().map(|w| hits(index, w)).collect::<Vec<_>>();
            let phrase = vec![index.search_phrase(&[Term::new_text(0, "brown"), Term::new_text(0, "dog")], 0, 100).unwrap().total_hits];
            let stats = [0u16, 3, 4, 7].iter().map(|&f| { let s = leaf.field_stats(f); (s.num_docs, s.sum_field_len) }).collect();
            (rows, [terms, phrase].concat(), stats)
        };
        let one = build(true);
        let two = build(false);
        let (rows, hits_one, stats) = view(&one);
        let (rows_two, hits_two, stats_two) = view(&two);
        assert_eq!((&rows, &hits_one), (&rows_two, &hits_two), "one pass differs from two");
        // Field statistics: the same totals. Keyword docCount counts the
        // deleted row (as an unsorted seal and Lucene do until a merge
        // drops it); the two-pass seal recounted live docs of its
        // intermediate segments, which already carried the delete.
        for (i, (one, two)) in stats.iter().zip(&stats_two).enumerate() {
            assert_eq!(one.1, two.1, "sumTotalTermFreq of stats field #{i}");
            assert!(one.0 == two.0 || one.0 == two.0 + 1, "docCount of stats field #{i}: {} vs {}", one.0, two.0);
        }
        assert_eq!(stats[0], stats_two[0]);
        assert_eq!(rows.iter().filter(|r| !r.0).count(), 27);
        assert!(rows.iter().any(|r| r.3.len() == 2) && rows.iter().any(|r| r.6.len() == 2));
    }

    #[test]
    fn a_coalesced_sorted_seal_equals_per_memtable_seals_under_concurrent_updates() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let build = |coalesce: bool| -> Index {
            let mut index = nrt_index();
            index.set_auto_seal(false);
            index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: false, max_mode: false, missing_first: false }])));
            for batch in [[9, 3, 6], [1, 8, 4], [7, 2, 5]] {
                for r in batch { index.add_document(&nrt_doc("foo", r)).unwrap(); }
                let closed = index.close_writers();
                index.parked.extend(closed);
            }
            let before = find_rank(&index, 8).unwrap();
            index.delete_doc(&before).unwrap();
            let jobs: Vec<SealJob> = if coalesce {
                vec![index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().unwrap()]
            } else {
                std::iter::from_fn(|| index.begin_seal_part(false, 1).unwrap()).collect()
            };
            assert_eq!(jobs.len(), if coalesce { 1 } else { 3 });
            // While the jobs encode: delete rank 3 (memtable 1), update rank 6
            // (memtable 1) and rank 4 (memtable 2).
            for r in [3, 6, 4] {
                let at = find_rank(&index, r).unwrap();
                index.delete_doc(&at).unwrap();
            }
            for r in [6, 4] { index.add_document(&nrt_doc("bar", r)).unwrap(); }
            for job in jobs { index.finish_seal(job.run()).unwrap(); }
            index.publish().unwrap();
            index
        };
        let view = |index: &Index| -> (Vec<(i64, String)>, u64, u64) {
            let mut live = Vec::new();
            for leaf in index.leaves() {
                let ranks: Vec<i64> = (0..leaf.num_docs()).map(|d| leaf.i64_value(1, d).unwrap()).collect();
                assert!(ranks.windows(2).all(|w| w[0] <= w[1]), "a segment out of sort order: {ranks:?}");
                for d in 0..leaf.num_docs() {
                    if leaf.is_deleted(d) { continue; }
                    let doc = leaf.get_doc(d).unwrap().unwrap();
                    live.push((ranks[d as usize], doc.get(0).and_then(|v| v.as_text()).unwrap_or_default().to_string()));
                }
            }
            live.sort();
            (live, hits(index, "foo"), hits(index, "bar"))
        };
        let coalesced = build(true);
        let reference = build(false);
        let (live, foo, bar) = view(&coalesced);
        assert_eq!((live.clone(), foo, bar), view(&reference), "the coalesced seal differs from per-memtable seals");
        assert_eq!(live, vec![(1, "foo".into()), (2, "foo".into()), (4, "bar".into()), (5, "foo".into()), (6, "bar".into()), (7, "foo".into()), (9, "foo".into())]);
        // kNN over the sorted, coalesced vectors.
        for (query, want) in [(4.2, 4), (8.6, 9), (2.9, 2)] {
            let top = coalesced.knn_search(&KnnRequest::new(2, vec![query, 0.0], 1)).unwrap();
            assert_eq!(coalesced.leaves()[top[0].leaf].i64_value(1, top[0].doc), Some(want), "query {query}");
        }
    }

    /// Safety gate before changing within-job vector scheduling: independent
    /// jobs already run concurrently in the background scheduler. Exercise
    /// provisional chunk registration, visibility changes and reverse install.
    #[test]
    fn concurrent_vector_seal_chunks_preserve_owners_deletes_and_reopen() {
        for published in [false, true] {
            let dir: Arc<dyn Directory> = Arc::new(RamDirectory::new());
            let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "rank", FieldType::I64),
                Field::new(2, "emb", FieldType::Vector { dim: 2 }).with_similarity("l2_norm"),
                Field::new(3, "name", FieldType::Keyword)];
            let mut index = Index::create_in(dir.clone(), "v", schema, IndexSettings::default()).unwrap();
            index.set_nrt(true);
            index.set_auto_seal(false);
            for round in 0..4 {
                for row in 0..5 { index.add_document(&nrt_doc("foo", round * 5 + row)).unwrap(); }
                if published { index.publish().unwrap(); }
                else { let closed = index.close_writers(); index.parked.extend(closed); }
            }
            let jobs = index.begin_seal_chunks(true, 1).unwrap();
            assert_eq!(jobs.len(), 4);
            let mut completed = std::thread::scope(|scope| {
                let workers: Vec<_> = jobs.into_iter().map(|job| scope.spawn(move || job.run())).collect();
                workers.into_iter().map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
            });
            // Encoder snapshots are stale when installed: a refresh changes
            // parked owners, and deletes change two different input chunks.
            index.publish().unwrap();
            for rank in [3, 18] { index.delete_doc(&find_rank(&index, rank).unwrap()).unwrap(); }
            index.add_document(&nrt_doc("bar", 20)).unwrap();
            while let Some(done) = completed.pop() { index.finish_seal(done).unwrap(); }
            index.publish().unwrap();
            let check = |index: &Index| {
                let all = index.knn_search(&KnnRequest::new(2, vec![0.0, 0.0], 30)).unwrap();
                let mut ranks: Vec<_> = all.iter().map(|hit| index.leaves()[hit.leaf].i64_value(1, hit.doc).unwrap()).collect();
                ranks.sort();
                assert_eq!(ranks, (0..21).filter(|rank| ![3, 18].contains(rank)).collect::<Vec<i64>>());
                assert_eq!(index.vectors.num_alive(2), 19);
            };
            // Include the fresh tail before the durable reopen check.
            for job in index.begin_seal_chunks(true, 1).unwrap() { index.finish_seal(job.run()).unwrap(); }
            check(&index);
            index.commit().unwrap();
            drop(index);
            check(&Index::open_in(dir).unwrap());
        }
    }

    /// Vector memtables coalesce: published memtables with vector rows seal
    /// into one segment whose points the store re-owns at installation;
    /// kNN stays exact (against a flat scan) after the seal and a reopen
    /// from the commit.
    #[test]
    fn coalesced_vector_seals_keep_knn_exact_across_a_reopen() {
        let dir: Arc<dyn Directory> = Arc::new(RamDirectory::new());
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "rank", FieldType::I64),
            Field::new(2, "emb", FieldType::Vector { dim: 2 }).with_similarity("l2_norm"),
            Field::new(3, "name", FieldType::Keyword),
        ];
        let mut index = Index::create_in(dir.clone(), "v", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.set_auto_seal(false);
        for round in 0..5i64 {
            for i in 0..4 { index.add_document(&nrt_doc("foo", round * 4 + i)).unwrap(); }
            index.publish().unwrap();
        }
        let gone = find_rank(&index, 10).unwrap();
        index.delete_doc(&gone).unwrap();
        index.publish().unwrap();
        let job = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("a job");
        assert_eq!(job.memtable_ids().len(), 5);
        index.finish_seal(job.run()).unwrap();
        assert_eq!(index.segments().len(), 1, "one segment for the five vector memtables");
        // Flat: the live rank nearest to q.
        let exact = |q: f32| -> i64 {
            (0..20i64).filter(|r| *r != 10).min_by(|a, b| (*a as f32 - q).abs().partial_cmp(&(*b as f32 - q).abs()).unwrap()).unwrap()
        };
        let check = |index: &Index| {
            for q in [0.2f32, 9.6, 10.4, 13.0, 19.0] {
                let top = index.knn_search(&KnnRequest::new(2, vec![q, 0.0], 1)).unwrap();
                assert_eq!(index.leaves()[top[0].leaf].i64_value(1, top[0].doc), Some(exact(q)), "query {q}");
            }
            // Every live doc once, no dead one: all 19 for k = 30.
            let all = index.knn_search(&KnnRequest::new(2, vec![0.0, 0.0], 30)).unwrap();
            let mut ranks: Vec<i64> = all.iter().map(|h| index.leaves()[h.leaf].i64_value(1, h.doc).unwrap()).collect();
            ranks.sort();
            assert_eq!(ranks, (0..20).filter(|r| *r != 10).collect::<Vec<i64>>(), "a dead or duplicate point was found");
            assert_eq!(index.vectors.num_alive(2), 19, "alive points differ from the live docs");
        };
        check(&index);
        index.commit().unwrap();
        drop(index);
        let reopened = Index::open_in(dir).unwrap();
        assert_eq!(reopened.segments().len(), 1);
        check(&reopened);
    }

    #[test]
    fn index_sort_orders_flushed_and_merged_segments() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let mut index = nrt_index();
        index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: false, max_mode: false, missing_first: false }])));
        for r in [4, 1, 3, 2] {
            index.add_document(&nrt_doc("foo", r)).unwrap();
        }
        let mut no_rank = Document::new();
        no_rank.add_text(0, "foo");
        index.add_document(&no_rank).unwrap();
        index.publish().unwrap();
        // Published memtables of a sorted index are sealed (sorted) at once.
        assert_eq!(index.segments().len(), 1);
        let ranks = |index: &Index| -> Vec<Option<i64>> {
            index.leaves().iter().flat_map(|l| (0..l.num_docs()).filter(|&d| !l.is_deleted(d)).map(|d| l.i64_value(1, d)).collect::<Vec<_>>()).collect()
        };
        assert_eq!(ranks(&index), vec![Some(1), Some(2), Some(3), Some(4), None]);
        assert_eq!(hits(&index, "foo"), 5);
        for r in [8, 6, 5, 7] {
            index.add_document(&nrt_doc("foo", r)).unwrap();
        }
        index.publish().unwrap();
        assert_eq!(ranks(&index), vec![Some(1), Some(2), Some(3), Some(4), None, Some(5), Some(6), Some(7), Some(8)]);
        index.delete_doc(&DocAddress::new(0, 1)).unwrap(); // rank 2
        index.refresh().unwrap();
        index.force_merge().unwrap();
        assert_eq!(ranks(&index), vec![Some(1), Some(3), Some(4), Some(5), Some(6), Some(7), Some(8), None]);
        // Postings, stored fields and vectors follow the new order.
        assert_eq!(hits(&index, "foo"), 8);
        let leaf = &index.leaves()[0];
        for d in 0..leaf.num_docs() {
            let doc = leaf.get_doc(d).unwrap().unwrap();
            assert_eq!(doc.get(1).and_then(|v| v.as_i64()), leaf.i64_value(1, d));
            if let Some(r) = leaf.i64_value(1, d) {
                assert_eq!(leaf.vector(2, d), Some(vec![r as f32, 0.0]));
            }
        }
        let top = index.knn_search(&KnnRequest::new(2, vec![6.0, 0.0], 1)).unwrap();
        assert_eq!(index.leaves()[top[0].leaf].i64_value(1, top[0].doc), Some(6));
    }

    #[test]
    fn sorted_seal_and_merge_remap_keyword_and_numeric_doc_values() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "rank", FieldType::I64),
            Field::new(3, "name", FieldType::Keyword),
            Field::new(4, "tags", FieldType::Keyword),
        ];
        let mut index = Index::new("sorted-dv", schema, IndexSettings::default());
        index.set_nrt(true);
        index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 1, kind: IndexSortKind::Long, reverse: true, max_mode: false, missing_first: false }])));
        // (rank, name, tags): the first segment's tags are multi-valued, the
        // second's single-valued (merged into the multi-valued column).
        let groups: [&[(i64, Option<&str>, &[&str])]; 2] = [
            &[(5, Some("e"), &["x"]), (1, Some("a"), &["z", "a"]), (3, None, &[]), (2, Some("b"), &["b", "c"])],
            &[(4, Some("d"), &["d"]), (6, Some("a"), &[]), (7, None, &["q"])],
        ];
        for group in groups {
            for &(rank, name, tags) in group {
                let mut d = Document::new();
                d.add_text(0, "foo");
                d.add_i64(1, rank);
                if let Some(name) = name { d.add_keyword(3, name); }
                for t in tags { d.add_keyword(4, t); }
                index.add_document(&d).unwrap();
            }
            index.publish().unwrap();
        }
        let rows = |index: &Index| -> Vec<(Option<i64>, String, String)> {
            index.leaves().iter().flat_map(|l| (0..l.num_docs()).filter(|&d| !l.is_deleted(d))
                .map(|d| (l.i64_value(1, d), l.keyword_values(3, d).join(","), l.keyword_values(4, d).join(","))).collect::<Vec<_>>()).collect()
        };
        let row = |r: i64, n: &str, t: &str| (Some(r), n.to_string(), t.to_string());
        // Each sealed segment is sorted (rank descending) with its values.
        assert_eq!(rows(&index), vec![row(5, "e", "x"), row(3, "", ""), row(2, "b", "b,c"), row(1, "a", "a,z"), row(7, "", "q"), row(6, "a", ""), row(4, "d", "d")]);
        // Deleting rank 2 and merging: one sorted segment over both.
        // (A doc address names the segment by its position.)
        let target = index.segments().iter().enumerate().find_map(|(i, seg)| {
            let leaf = LeafReader::Segment(seg);
            (0..seg.num_docs).find(|&d| leaf.i64_value(1, d) == Some(2)).map(|d| DocAddress::new(i as u32, d))
        }).expect("rank 2 is sealed");
        index.delete_doc(&target).unwrap();
        index.refresh().unwrap();
        index.force_merge().unwrap();
        assert_eq!(index.segments().len(), 1);
        let want = vec![row(7, "", "q"), row(6, "a", ""), row(5, "e", "x"), row(4, "d", "d"), row(3, "", ""), row(1, "a", "a,z")];
        assert_eq!(rows(&index), want);
        index.commit().unwrap();
        let reopened = Index::open_in(index.directory.clone()).unwrap();
        assert_eq!(rows(&reopened), want);
    }

    #[test]
    fn detached_sorted_merge_keeps_nested_blocks_together() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let mut index = nrt_index();
        index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 3, kind: IndexSortKind::Keyword, reverse: true, max_mode: true, missing_first: false }])));
        for (name, kids) in [("b", 2), ("c", 1), ("a", 3)] {
            let mut doc = Document::new();
            doc.add_keyword(3, name);
            for child in 0..kids {
                let mut nested = Document::new();
                nested.add_i64(1, child);
                doc.add_nested("kids", nested);
            }
            index.add_document(&doc).unwrap();
            index.publish().unwrap();
        }
        assert_eq!(index.segments().len(), 3);
        let job = index.begin_force_merge().unwrap().unwrap();
        let merged = job.run().unwrap();
        assert!(job.install_map.get().is_some(), "installation map must be prepared outside the core lock");
        assert!(index.finish_merge(job, merged).unwrap());
        let leaves = index.leaves();
        let leaf = &leaves[0];
        let roots: Vec<_> = (0..leaf.num_docs()).filter(|&d| leaf.is_root(d)).map(|d| {
            let start = leaf.nested_info().unwrap().block_start(d);
            let children: Vec<_> = (start..d).map(|c| leaf.i64_value(1, c).unwrap()).collect();
            (leaf.keyword_values(3, d).join(","), children)
        }).collect();
        assert_eq!(roots, vec![("c".into(), vec![0]), ("b".into(), vec![0, 1]), ("a".into(), vec![0, 1, 2])]);
        index.commit().unwrap();
        let reopened = Index::open_in(index.directory.clone()).unwrap();
        assert_eq!(reopened.segments()[0].num_docs, 9);
    }

    #[test]
    fn index_sort_moves_nested_blocks_as_units() {
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        let mut index = nrt_index();
        index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 3, kind: IndexSortKind::Keyword, reverse: true, max_mode: true, missing_first: false }])));
        for (name, kids) in [("b", 2), ("c", 1), ("a", 3)] {
            let mut d = Document::new();
            d.add_keyword(3, name);
            for k in 0..kids {
                let mut c = Document::new();
                c.add_i64(1, k);
                d.add_nested("kids", c);
            }
            index.add_document(&d).unwrap();
        }
        index.publish().unwrap();
        let leaf = &index.leaves()[0];
        let roots: Vec<(String, u32)> = (0..leaf.num_docs())
            .filter(|&d| leaf.is_root(d))
            .map(|d| (leaf.keyword_values(3, d).join(","), d - leaf.nested_info().map_or(d, |n| n.block_start(d))))
            .collect();
        assert_eq!(roots, vec![("c".to_string(), 1), ("b".to_string(), 2), ("a".to_string(), 3)]);
    }
}


#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod mutable_commit_tests {
    use super::*;
    use crate::codec::directory::FileSlice;
    use crate::vector::{HnswParams, Similarity, store::VectorFieldOptions};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Default)]
    struct MetaFailureDirectory { inner: RamDirectory, fail_meta: AtomicBool }
    impl Directory for MetaFailureDirectory {
        fn open_read(&self, name: &str) -> Result<FileSlice> { self.inner.open_read(name) }
        fn write(&self, name: &str, bytes: &[u8]) -> Result<()> {
            if name == META_FILE && self.fail_meta.load(Ordering::Acquire) {
                return Err(RusticError::Io(std::io::Error::other("injected manifest failure")));
            }
            self.inner.write(name, bytes)
        }
        fn list(&self) -> Result<Vec<String>> { self.inner.list() }
        fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
        fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
        fn delete(&self, name: &str) -> Result<()> { self.inner.delete(name) }
    }
    fn schema() -> Vec<Field> {
        vec![Field::new(1, "vector", FieldType::Vector { dim: 2 }).with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions::f32_hnsw()))]
    }
    fn append(index: &mut Index, field: &Arc<crate::vector::mutable_ingest::MutableIngestField>, value: f32) {
        let id = index.next_segment_id;
        index.next_segment_id += 1;
        let mut writer = SegmentWriter::new(id);
        let mut document = Document::new();
        document.add_vector(1, vec![value, 1.]);
        writer.add_document(&document, &index.schema);
        writer.stage_mutable_field(1, field).unwrap();
        let segment = writer.write_to_store_ref(&*index.directory, Some(&index.vectors), false).unwrap();
        index.segments.push(Arc::new(segment));
    }
    fn hits(index: &Index) -> Vec<(u64, f32)> {
        use crate::vector::vespa_policy::*;
        let count = index.vectors.next_point_id() as u32;
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 10, explore_additional_hits: 90 }, count, PreparedGlobalFilter::Inactive).unwrap();
        index.vectors.search_mutable_ann(1, &[0.,1.], &plan,
            |key| index.vectors.owner(key).is_some(), None).unwrap().unwrap().hits
    }
    #[test]
    fn mutable_graph_commit_reopens_and_manifest_failure_preserves_previous_graph() {
        let dir = Arc::new(MetaFailureDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let params = HnswParams { threads: 4, ..Default::default() };
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, params).unwrap();
        append(&mut index, &field, 0.);
        index.commit().unwrap();
        index.publish_view().unwrap();
        assert_eq!(index.reader_view().segment_stats().iter().filter(|s| s.committed).count(), 1);
        let first = hits(&index);
        let reopened = Index::open_in(dir.clone()).unwrap();
        assert_eq!(hits(&reopened), first);
        assert!(reopened.vectors.mutable_field(1, 2, Similarity::L2, params).is_ok());
        let old_manifest = dir.open_read(META_FILE).unwrap().as_slice().to_vec();
        append(&mut index, &field, 10.);
        dir.fail_meta.store(true, Ordering::Release);
        assert!(index.commit().is_err());
        index.publish_view().unwrap();
        assert_eq!(index.reader_view().segment_stats().iter().filter(|s| s.committed).count(), 1);
        assert_eq!(dir.open_read(META_FILE).unwrap().as_slice(), old_manifest);
        assert_eq!(hits(&Index::open_in(dir.clone()).unwrap()), first);
        dir.fail_meta.store(false, Ordering::Release);
        index.commit().unwrap();
        assert_eq!(hits(&Index::open_in(dir.clone()).unwrap()), hits(&index));
        assert_eq!(dir.list().unwrap().iter().filter(|name| name.ends_with(".rmvc")).count(), 1);
    }
    /// U5: a quantized field keeps only codes in memory once its f32 is
    /// durable in the store's segments; search, exact scoring and
    /// rescoring are unchanged; the checkpoint references the durable f32
    /// and reopens code-only; inserts after reopen build the graph from the
    /// on-disk rows.
    #[test]
    fn quantized_rows_keep_codes_only_once_their_f32_is_durable() {
        // Code-only checkpoint rows are RMVR v3 (the next write version).
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::RMVR3_SINCE));
        use crate::vector::store::VectorFieldOptions;
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) }))];
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "quantized", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let row = |i: usize| -> Vec<f32> { (0..16).map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0).collect() };
        let add = |index: &mut Index, rows: std::ops::Range<usize>| {
            for i in rows {
                let mut document = Document::new();
                document.add_vector(1, row(i));
                index.add_document(&document).unwrap();
            }
        };
        add(&mut index, 0..300);
        index.publish().unwrap();
        index.seal().unwrap();
        let search = |index: &Index, query: &[f32], exact: bool| -> Vec<(u32, usize, DocId)> {
            let view = index.reader_view();
            let leaves = view.leaves();
            let ctx = crate::query::query::SearchContext::for_query(&crate::query::query::Query::MatchAll, &leaves, Bm25Params::default());
            crate::query::mutable_knn::search(&leaves, 1, query, 10, 100, None, &ctx, exact, None)
                .unwrap().unwrap().0.into_iter().map(|(score, leaf, doc)| (score.to_bits(), leaf, doc)).collect()
        };
        let before = (search(&index, &row(7), false), search(&index, &row(7), true));
        let field = index.vectors.registered_mutable_field(1).unwrap();
        let full = field.source_row_bytes();
        assert_eq!(field.shed_all().unwrap(), 300, "every sealed row sheds its f32");
        let codes = field.source_row_bytes();
        assert!(codes * 4 < full * 3, "{full} -> {codes}");
        assert!(codes / 300 < 32 + 16 * 4, "{} bytes per code-only row (f32 alone is 64)", codes / 300);
        assert_eq!((search(&index, &row(7), false), search(&index, &row(7), true)), before, "results changed once the f32 left memory");
        // The checkpoint references the durable f32; reopen restores code-only rows.
        let mut capture = index.begin_vector_capture(true).unwrap();
        capture.finish().unwrap();
        index.prepare_commit_with(capture).unwrap().run().unwrap();
        let mut reopened = Index::open_in(dir.clone()).unwrap();
        reopened.set_nrt(true);
        assert_eq!((search(&reopened, &row(7), false), search(&reopened, &row(7), true)), before, "reopen changed the results");
        let restored = reopened.vectors.registered_mutable_field(1).unwrap();
        assert!(restored.source_row_bytes() / 300 < 32 + 16 * 4, "reopened rows are code-only");
        // Graph construction after reopen reads the on-disk rows.
        add(&mut reopened, 300..320);
        reopened.publish().unwrap();
        let hits = search(&reopened, &row(311), true);
        let view = reopened.reader_view();
        let leaves = view.leaves();
        let (_, leaf, doc) = hits[0];
        assert_eq!(leaves[leaf].with_vector(1, doc, <[f32]>::to_vec), Some(row(311)), "the new document is its own nearest neighbour");
    }

    fn quantized_index(name: &str, rows: std::ops::Range<usize>) -> Index {
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) }))];
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), name, schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        add_quantized(&mut index, rows);
        index
    }
    fn add_quantized(index: &mut Index, rows: std::ops::Range<usize>) {
        for i in rows {
            let mut document = Document::new();
            document.add_vector(1, (0..16).map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0).collect());
            index.add_document(&document).unwrap();
        }
    }

    fn quantized_hits(index: &Index, query: &[f32]) -> Vec<(u64, f32)> {
        use crate::vector::vespa_policy::*;
        let count = index.vectors.next_point_id() as u32;
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 10, explore_additional_hits: 90 }, count, PreparedGlobalFilter::Inactive).unwrap();
        index.vectors.search_mutable_ann(1, query, &plan,
            |key| index.vectors.owner(key).is_some(), None).unwrap().unwrap().hits
    }

    /// File-backed codes: shed rows move to the code log (out of anonymous
    /// row storage) and are served identically; a graph checkpoint
    /// references them through code log files the manifest lists (writing
    /// only rows no earlier file holds), a later commit drops a superseded
    /// prefix file, a commit without a graph save keeps them, the index
    /// reopens from them, and stray chunk files are swept.
    #[test]
    fn shed_rows_live_in_the_code_log_and_checkpoints_reference_them() {
        // Code log references are a format-2 (write version 13) feature.
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::RMVC2_SINCE));
        let options = VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) };
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm").with_vector_options(Some(options))];
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "code-log", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        add_quantized(&mut index, 0..300);
        index.publish().unwrap();
        index.seal().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        let query: Vec<f32> = (0..16).map(|d| ((d * 5) % 97) as f32 / 97.0).collect();
        let before = quantized_hits(&index, &query);
        let full = field.source_row_bytes();
        assert_eq!(field.shed_all().unwrap(), 300);
        field.commit_owner().unwrap();
        assert!(field.source_row_bytes() * 10 < full, "shed rows left the anonymous row store ({} of {full} bytes)", field.source_row_bytes());
        for point in 0..300 { assert!(field.row_readable(point)); }
        assert_eq!(quantized_hits(&index, &query), before, "the code log serves identical codes");
        let commit = |index: &Index, save: bool| -> crate::vector::store::MutableFieldManifest {
            let mut capture = index.begin_vector_capture(save).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let meta: Meta = serde_json::from_slice(dir.open_read(META_FILE).unwrap().as_slice()).unwrap();
            meta.vector_store.mutable_fields[0].clone()
        };
        let first = commit(&index, true);
        assert_eq!(first.code_files.len(), 1, "{:?}", first.code_files);
        assert!(dir.exists(&first.code_files[0]));
        add_quantized(&mut index, 300..600);
        index.publish().unwrap();
        index.seal().unwrap();
        assert_eq!(field.shed_all().unwrap(), 300);
        let second = commit(&index, true);
        assert_eq!(second.code_files.len(), 1);
        assert_ne!(second.code_files, first.code_files, "a longer prefix is a new immutable file");
        assert!(!dir.exists(&first.code_files[0]), "the superseded prefix goes with the commit");
        // More rows committed without a graph save: reopening catches up.
        add_quantized(&mut index, 600..650);
        index.publish().unwrap();
        index.seal().unwrap();
        field.shed_all().unwrap();
        let third = commit(&index, false);
        assert_eq!(third.code_files, second.code_files, "a commit keeping the graph checkpoint keeps its files");
        let live = quantized_hits(&index, &query);
        let reopened = Index::open_in(dir.clone()).unwrap();
        assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(650));
        let restored = quantized_hits(&reopened, &query);
        let overlap = restored.iter().filter(|(key, _)| live.iter().any(|(k, _)| k == key)).count();
        assert!(overlap + 1 >= live.len().min(10), "reopened results {restored:?} vs {live:?}");
        // A stray chunk file (written before a crash) is swept; referenced ones stay.
        let stray = "mvq_00000000000000aa_00000000_00000001.rmvq";
        dir.write(stray, &[0; 4]).unwrap();
        assert!(reopened.sweep_orphan_files().unwrap() >= 1);
        assert!(!dir.exists(stray));
        assert!(second.code_files.iter().all(|name| dir.exists(name)));
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    fn code_log_index(name: &str, rows: std::ops::Range<usize>) -> (Arc<RamDirectory>, Index) {
        let options = VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) };
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm").with_vector_options(Some(options))];
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), name, schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        add_quantized(&mut index, rows);
        index.publish().unwrap();
        index.seal().unwrap();
        (dir, index)
    }

    /// Delete the documents of `points` and let graph repair release their
    /// source slots, until the code log has `live` live rows.
    fn delete_points_until_live(index: &mut Index, points: impl Iterator<Item = u64> + Clone, live: u64) {
        for (leaf, segment) in index.segments.iter().enumerate() {
            for point in points.clone() {
                if let Some(doc) = segment.doc_of_point(1, point) {
                    index.delete_doc_shared(&DocAddress::new(leaf as u32, doc)).unwrap();
                }
            }
        }
        index.publish().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        for _ in 0..200 {
            index.vectors.drain_mutable_deletes().unwrap();
            field.maintain().unwrap();
            if field.code_log_state().unwrap().2 == live { return; }
            std::thread::yield_now();
        }
        panic!("code log live rows {:?}, expected {live}", field.code_log_state());
    }

    /// H6: deleted (or updated) shed rows stay in the append-only code log
    /// as dead rows. Once most are dead, the next commit copies the live
    /// ones into a fresh log; results are unchanged, the checkpoint after
    /// the switch references only the new log, the old log's files are
    /// swept, and the index reopens from the new log.
    #[test]
    fn code_log_compacts_dead_rows_into_a_fresh_log_at_commit() {
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::RMVC2_SINCE));
        let (dir, mut index) = code_log_index("code-log-compaction", 0..600);
        let field = index.vectors.registered_mutable_field(1).unwrap();
        assert_eq!(field.shed_all().unwrap(), 600);
        field.commit_owner().unwrap();
        field.set_code_log_compact_min_rows(1);
        let commit = |index: &Index, save: bool| -> crate::vector::store::MutableFieldManifest {
            let mut capture = index.begin_vector_capture(save).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let meta: Meta = serde_json::from_slice(dir.open_read(META_FILE).unwrap().as_slice()).unwrap();
            meta.vector_store.mutable_fields[0].clone()
        };
        let first = commit(&index, true);
        let (old_id, rows, live, _) = field.code_log_state().unwrap();
        assert_eq!((rows, live), (600, 600), "no dead rows yet: no compaction");
        assert!(first.code_files.iter().all(|name| name.contains(&format!("{old_id:016x}"))));
        delete_points_until_live(&mut index, (0..600u64).filter(|p| p % 3 != 0), 200);
        let (id, rows, _, _) = field.code_log_state().unwrap();
        assert_eq!((id, rows), (old_id, 600), "deleted rows stay in the log, dead");
        let query: Vec<f32> = (0..16).map(|d| ((d * 5) % 97) as f32 / 97.0).collect();
        let before = quantized_hits(&index, &query);
        // The commit's seal step compacts: two thirds of the rows are dead.
        commit(&index, true);
        let (new_id, rows, live, compacting) = field.code_log_state().unwrap();
        assert_ne!(new_id, old_id, "a fresh log id");
        assert_eq!((rows, live, compacting), (200, 200, false), "only live rows were copied");
        for point in (0..600u64).filter(|p| p % 3 == 0) { assert!(field.row_readable(point), "point {point}"); }
        assert_eq!(quantized_hits(&index, &query), before, "compaction changed the results");
        // The graph is marked changed: the next commit saves a checkpoint
        // naming only the new log, and the old log's files go.
        let after = commit(&index, true);
        assert!(!after.code_files.is_empty());
        assert!(after.code_files.iter().all(|name| name.contains(&format!("{new_id:016x}"))), "{:?}", after.code_files);
        field.commit_owner().unwrap();
        index.sweep_orphan_files().unwrap();
        let old_prefix = format!("mvq_{old_id:016x}_");
        assert!(dir.list().unwrap().iter().all(|name| !name.starts_with(&old_prefix)), "old log files swept");
        assert!(after.code_files.iter().all(|name| dir.exists(name)));
        let reopened = Index::open_in(dir.clone()).unwrap();
        assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(200));
        let restored = quantized_hits(&reopened, &query);
        let overlap = restored.iter().filter(|(key, _)| before.iter().any(|(k, _)| k == key)).count();
        assert!(overlap + 1 >= before.len().min(10), "reopened results {restored:?} vs {before:?}");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// H6: a full code log is backpressure, not an error: shedding stops
    /// (rows keep their f32), delete repair and maintenance keep running,
    /// and once compaction frees room shedding resumes.
    #[test]
    fn full_code_log_is_backpressure_and_compaction_frees_room() {
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::RMVC2_SINCE));
        let (_dir, mut index) = code_log_index("code-log-full", 0..300);
        let field = index.vectors.registered_mutable_field(1).unwrap();
        field.set_code_log_capacity(100);
        field.set_code_log_compact_min_rows(1);
        assert_eq!(field.shed_all().unwrap(), 100, "shedding stops at the full log");
        assert_eq!(field.shed_all().unwrap(), 0, "and stays stopped without error");
        field.maintain().unwrap();
        // Delete repair's shedding step no longer fails on the full log.
        field.drain_some(64).unwrap();
        for point in 0..300 { assert!(field.row_readable(point), "point {point}"); }
        // Delete most of the shed rows: compaction at the next commit makes room.
        let (old_id, _, _, _) = field.code_log_state().unwrap();
        delete_points_until_live(&mut index, 0..80u64, 20);
        let mut capture = index.begin_vector_capture(true).unwrap();
        capture.finish().unwrap();
        index.prepare_commit_with(capture).unwrap().run().unwrap();
        let (new_id, rows, live, _) = field.code_log_state().unwrap();
        assert_ne!(new_id, old_id);
        assert_eq!((rows, live), (20, 20));
        assert_eq!(field.shed_all().unwrap(), 200, "shedding resumes into the fresh log");
        for point in 80..300 { assert!(field.row_readable(point), "point {point}"); }
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// K7: below write version 13 no checkpoint can reference code log
    /// rows, so shed rows stay code-only in the row store: nothing enters
    /// the log and no chunk file is written that no commit lists.
    #[test]
    fn rows_shed_below_format_two_stay_out_of_the_code_log() {
        for version in [crate::codec::DEFAULT_WRITE_VERSION, crate::codec::RMVR3_SINCE, crate::codec::RMVC2_SINCE] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let (dir, index) = code_log_index("shed-below-floor", 0..100);
            let field = index.vectors.registered_mutable_field(1).unwrap();
            assert_eq!(field.shed_all().unwrap(), 100, "version {version}");
            let mut capture = index.begin_vector_capture(true).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let (_, log_rows, _, _) = field.code_log_state().unwrap();
            let chunk_files = dir.list().unwrap().into_iter().filter(|name| crate::vector::store::is_code_log_file(name)).count();
            if version >= crate::codec::RMVC2_SINCE {
                assert_eq!(log_rows, 100);
                assert!(chunk_files > 0);
            } else {
                assert_eq!((log_rows, chunk_files), (0, 0), "version {version}");
            }
            for point in 0..100 { assert!(field.row_readable(point)); }
        }
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// M11: a sealed payload segment whose dead rows the graph still holds
    /// (repair deferred by an older live view) is not rewritten on every
    /// optimizer pass: the rewrite would keep every one of them. It is
    /// vacuumed once the repair frees them.
    #[test]
    fn vacuum_skips_segments_whose_dead_rows_the_graph_still_holds() {
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "held-vacuum", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        for i in 0..100 {
            let mut document = Document::new();
            document.add_vector(1, vec![i as f32, 1.]);
            index.add_document(&document).unwrap();
        }
        index.publish().unwrap();
        index.seal().unwrap();
        let job = index.vector_optimizer();
        job.force_merge(1, false).unwrap();
        assert!(index.vectors.segments(1).iter().all(|s| s.is_sealed()));
        let segment = index.segments[0].id;
        // An older view counts the documents live: their repair waits.
        let old_view = index.reader_view();
        for doc in 0..50 {
            index.delete_doc(&DocAddress::new(segment, doc)).unwrap();
        }
        index.publish().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        assert!(field.holds_point(0), "repair deferred by the older view");
        for pass in 0..2 {
            let report = job.run().unwrap();
            assert_eq!(report.vacuumed, 0, "pass {pass}: rewrote a segment whose dead rows are all held");
        }
        drop(old_view);
        index.publish().unwrap();
        assert!(!field.holds_point(0));
        assert_eq!(job.run().unwrap().vacuumed, 1, "vacuumed once the repair freed the rows");
    }

    /// H3: reopening on an older graph checkpoint catches the graph up in
    /// bounded batches (it buffered every missing row at once, serially).
    #[test]
    fn open_catch_up_streams_the_missing_points_in_batches() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "catch-up", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let add = |index: &mut Index, rows: std::ops::Range<i32>| {
            for i in rows {
                let mut document = Document::new();
                document.add_vector(1, vec![i as f32, 1.]);
                index.add_document(&document).unwrap();
            }
        };
        let commit = |index: &Index, save: bool| {
            let mut capture = index.begin_vector_capture(save).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
        };
        add(&mut index, 0..10);
        index.publish().unwrap();
        index.seal().unwrap();
        commit(&index, true);
        add(&mut index, 10..310);
        index.publish().unwrap();
        index.seal().unwrap();
        commit(&index, false); // keeps the 10-node graph checkpoint
        crate::vector::mutable_ingest::CATCH_UP_MAX_BUFFERED.with(|max| max.set(0));
        let reopened = Index::open_in(dir).unwrap();
        assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(310));
        let buffered = crate::vector::mutable_ingest::CATCH_UP_MAX_BUFFERED.with(|max| max.get());
        assert!(buffered > 0 && buffered <= 64, "the catch-up buffered {buffered} rows at once");
        let hits = hits(&reopened);
        assert_eq!(hits.len(), 10);
    }

    /// H1: unfiltered mutable kNN during indexing meets graph nodes of
    /// documents the reader view does not hold (owner not installed yet).
    /// They resolve to "not in this view" without a per-query map of every
    /// view point (it was rebuilt on every such query).
    #[test]
    fn knn_during_indexing_does_not_rebuild_a_point_map_per_query() {
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "no-rebuild", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let add = |index: &mut Index, rows: std::ops::Range<i32>| {
            for i in rows {
                let mut document = Document::new();
                document.add_vector(1, vec![i as f32, 1.]);
                index.add_document(&document).unwrap();
            }
        };
        add(&mut index, 0..50);
        index.publish().unwrap();
        index.seal().unwrap();
        add(&mut index, 50..60);
        index.publish().unwrap();
        let view = index.reader_view();
        // Indexed after the view: in the graph, not in the view.
        add(&mut index, 60..120);
        let leaves = view.leaves();
        let ctx = crate::query::query::SearchContext::for_query(&crate::query::query::Query::MatchAll, &leaves, Bm25Params::default());
        let before = crate::reader_view::moved_owner_rebuilds();
        for i in 0..20 {
            // Near the documents the view does not hold: the hits are the
            // view's own nearest ones (50..60 in its memtable, then the
            // segment's highest), never a document indexed after the view.
            let query = (60 + i) as f32;
            let hits = crate::query::mutable_knn::search(&leaves, 1, &[query, 1.], 10, 100, None, &ctx, false, None).unwrap().unwrap().0;
            let mut values: Vec<f32> = hits.iter().map(|(_, leaf, doc)| leaves[*leaf].with_vector(1, *doc, |row| row[0]).unwrap()).collect();
            values.sort_by(f32::total_cmp);
            assert_eq!(values, (50..60).map(|v| v as f32).collect::<Vec<_>>(), "query {query}");
        }
        let rebuilt = crate::reader_view::moved_owner_rebuilds() - before;
        assert!(rebuilt < 5, "{rebuilt} point maps built for 20 queries");
    }

    /// A commit prepared from an unfinished capture
    /// (`Index::prepare_commit_deferred`, so a flush can let merges install
    /// while the job copies and writes the graphs) commits the graph as of
    /// its capture point: documents added after the prepare are in neither
    /// the commit nor its graph checkpoint, and the reopened graph serves the
    /// committed documents.
    #[test]
    fn deferred_capture_commits_the_graph_as_of_its_capture_point() {
        let mut index = quantized_index("deferred-capture", 0..200);
        index.publish().unwrap();
        index.seal().unwrap();
        let capture = index.begin_vector_capture(true).unwrap();
        let job = index.prepare_commit_deferred(capture).unwrap();
        add_quantized(&mut index, 200..260);
        index.publish().unwrap();
        index.seal().unwrap();
        job.run().unwrap();
        let reopened = Index::open_in(index.directory().clone()).unwrap();
        assert_eq!(reopened.num_docs(), 200, "the commit holds the documents as of its prepare");
        assert!(reopened.vectors.mutable_commit.lock().checkpointed.contains_key(&1), "the graph checkpoint is in the commit");
        assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(200), "the reopened graph holds the committed documents");
    }

    /// R8b: a merge that drops the durable rows of points the graph still
    /// holds (repair pending) brings their f32 back into memory before the
    /// swap, so the rows stay readable across it.
    #[test]
    fn rows_the_graph_holds_stay_readable_across_a_merge_that_drops_them() {
        let mut index = quantized_index("handoff", 0..64);
        index.publish().unwrap();
        index.seal().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        assert_eq!(field.shed_all().unwrap(), 64, "code-only rows");
        let segment = index.segments[0].id;
        for doc in 0..64 {
            index.delete_doc(&DocAddress::new(segment, doc)).unwrap();
        }
        // Deletes visible, graph repair deferred: the graph still holds the nodes.
        index.publish_visible().unwrap();
        assert!(field.holds_point(0));
        let observed = Arc::new(parking_lot::Mutex::new(None));
        let (probe, seen) = (field.clone(), observed.clone());
        crate::vector::store::AFTER_SWAP.with(|hook| *hook.borrow_mut() = Some(Box::new(move || {
            *seen.lock() = Some(probe.row_readable(0));
        })));
        // The indexing pass drops the fully dead chunk without a repair.
        index.vector_optimizer().index_pending().unwrap();
        assert_eq!(*observed.lock(), Some(true), "a row the graph holds was unreadable right after the swap (None: no swap ran)");
        assert!(field.row_readable(0));
    }

    #[test]
    fn periodic_maintenance_sheds_rows_sealed_after_refresh_repair() {
        let mut index = quantized_index("idle-shed-progress", 0..300);
        index.publish().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        let full = field.source_row_bytes();
        // A refresh-triggered repair exhausts eligible work before seal.
        assert!(!index.vectors.drain_mutable_deletes_batch(4096).unwrap());
        index.vector_optimizer().maintain_once().unwrap();
        assert_eq!(field.source_row_bytes(), full, "unsealed raw rows must survive");
        index.seal().unwrap();
        // No new write or refresh re-arms repair. Periodic maintenance alone
        // must release redundant f32 while keeping every row readable.
        index.vector_optimizer().maintain_once().unwrap();
        assert!(field.source_row_bytes() * 4 < full * 3);
        for point in 0..300 { assert!(field.row_readable(point)); }
        assert_eq!(field.shed_all().unwrap(), 0, "maintenance already shed durable rows");
    }

    /// R8c: one row whose f32 is not durable yet (unsealed) at the head of
    /// the shed queue does not stall shedding of the durable rows behind it.
    #[test]
    fn an_unsealed_row_does_not_stall_shedding_of_durable_rows() {
        let mut index = quantized_index("shed-progress", 0..300);
        index.publish().unwrap();
        index.seal().unwrap();
        add_quantized(&mut index, 300..301);
        index.publish().unwrap();
        let field = index.vectors.registered_mutable_field(1).unwrap();
        field.unshed_newest_first();
        assert_eq!(field.shed_all().unwrap(), 300, "shedding stalled behind the unsealed row");
        index.seal().unwrap();
        assert_eq!(field.shed_all().unwrap(), 1, "the deferred row sheds once durable");
    }

    /// Exact (and graph) mutable kNN reads code-only rows through the
    /// store, which takes `state` then `table`. The query must not hold the
    /// point table meanwhile: a writer holding `state` (segment install,
    /// merge swap) waits for `table`, and the fair table lock queues the
    /// query's nested read behind any waiting writer, so the query, the
    /// shard's writers and its merges hang forever.
    #[test]
    fn mutable_knn_never_holds_the_point_table_while_reading_rows() {
        use crate::vector::store::VectorFieldOptions;
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) }))];
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "lock-order", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let row = |i: usize| -> Vec<f32> { (0..16).map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0).collect() };
        for i in 0..300 {
            let mut document = Document::new();
            document.add_vector(1, row(i));
            index.add_document(&document).unwrap();
        }
        index.publish().unwrap();
        index.seal().unwrap();
        // Code-only rows: exact scoring and rescoring read the f32 from the store.
        assert_eq!(index.vectors.registered_mutable_field(1).unwrap().shed_all().unwrap(), 300);
        let search = |index: &Index, query: &[f32], exact: bool| -> usize {
            let view = index.reader_view();
            let leaves = view.leaves();
            let ctx = crate::query::query::SearchContext::for_query(&crate::query::query::Query::MatchAll, &leaves, Bm25Params::default());
            crate::query::mutable_knn::search(&leaves, 1, query, 10, 100, None, &ctx, exact, None).unwrap().unwrap().0.len()
        };
        // Deterministic lock-order assertion: every row read checks that the
        // point table is free.
        index.vectors.row_lock_probe.store(true, Ordering::Release);
        for exact in [true, false] {
            assert_eq!(search(&index, &row(7), exact), 10);
        }
        index.vectors.row_lock_probe.store(false, Ordering::Release);
        assert!(index.vectors.row_lock_probed.load(Ordering::Acquire) > 0, "the searches read code-only rows");
        assert_eq!(index.vectors.row_lock_violations.load(Ordering::Acquire), 0, "a row was read with the point table locked");
        // Exact kNN against concurrent state -> table writers (the segment
        // install a seal or merge swap does), bounded by a timeout.
        let index = Arc::new(index);
        let segment = index.segments.first().cloned().expect("sealed segment");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer = {
            let (index, stop) = (index.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    index.vectors.attach_prepared(&segment, &mut [], false);
                    index.vectors.reveal(&segment);
                }
            })
        };
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let query = index.clone();
        std::thread::spawn(move || {
            for i in 0..200 {
                assert_eq!(search(&query, &row(i % 300), true), 10);
            }
            done_tx.send(()).unwrap();
        });
        let finished = done_rx.recv_timeout(std::time::Duration::from_secs(120));
        stop.store(true, Ordering::Release);
        assert!(finished.is_ok(), "exact kNN hung against concurrent store writers");
        writer.join().unwrap();
    }

    /// Rollback safety: with the default (compatible) write version a graph
    /// checkpoint is format 1 with RMVR v1 (unquantized) or v2 (quantized
    /// full rows) rows, what the previous release reads; v12 adds RMVR v3
    /// rows. Under write version 13 it is format 2 (RMVR v4): the graph plus references
    /// (durable f32 in the vector store, codes in the code log files the
    /// manifest lists). Every case reopens.
    #[test]
    fn checkpoint_rows_are_compatible_unless_the_next_write_version_is_set() {
        for (quantized, version) in [(false, crate::codec::DEFAULT_WRITE_VERSION), (true, crate::codec::DEFAULT_WRITE_VERSION), (false, crate::codec::RMVR3_SINCE), (true, crate::codec::RMVR3_SINCE), (false, crate::codec::RMVC2_SINCE), (true, crate::codec::RMVC2_SINCE)] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let options = if quantized {
                VectorFieldOptions { oversample: 2.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) }
            } else { VectorFieldOptions::f32_hnsw() };
            let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 16 }).with_similarity("l2_norm").with_vector_options(Some(options))];
            let dir = Arc::new(RamDirectory::default());
            let mut index = Index::create_in(dir.clone(), "rows-format", schema, IndexSettings::default()).unwrap();
            index.set_nrt(true);
            index.enable_mutable_vectors().unwrap();
            for i in 0..50usize {
                let mut document = Document::new();
                document.add_vector(1, (0..16).map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0).collect());
                index.add_document(&document).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
            if quantized {
                assert_eq!(index.vectors.registered_mutable_field(1).unwrap().shed_all().unwrap(), 50, "code-only rows");
            }
            let mut capture = index.begin_vector_capture(true).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let meta: Meta = serde_json::from_slice(dir.open_read(META_FILE).unwrap().as_slice()).unwrap();
            let entry = &meta.vector_store.mutable_fields[0];
            let file = dir.open_read(&entry.filename()).unwrap();
            let bytes = file.as_slice();
            let graph_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            let (graph, rows) = (&bytes[16..16 + graph_len], &bytes[16 + graph_len..]);
            if version >= crate::codec::RMVC2_SINCE {
                // Every node reads v13: format 2 (graph + references), RMVR v4.
                assert_eq!(&bytes[..8], b"RMVC\x02\0\0\0");
                assert_eq!(&rows[..8], b"RMVR\x04\0\0\0", "quantized {quantized}, write version {version}");
                if quantized {
                    // Shed rows live in the code log: referenced, never copied.
                    assert!(!entry.code_files.is_empty() && entry.code_files.iter().all(|name| dir.exists(name)));
                } else {
                    assert!(entry.code_files.is_empty());
                }
            } else {
                // Rollback safety: what the previous release reads (v1
                // unquantized, v2 quantized full rows; code-only rows get
                // their f32 back from the store), no code files.
                assert_eq!(&bytes[..8], b"RMVC\x01\0\0\0");
                let expected: &[u8; 8] = match (quantized, version >= crate::codec::RMVR3_SINCE) {
                    (_, true) => b"RMVR\x03\0\0\0",
                    (true, false) => b"RMVR\x02\0\0\0",
                    (false, false) => b"RMVR\x01\0\0\0",
                };
                assert_eq!(&rows[..8], expected, "quantized {quantized}, write version {version}");
                assert!(entry.code_files.is_empty());
                if version == crate::codec::DEFAULT_WRITE_VERSION {
                    // The previous release's decode: embedded rows, no vector store.
                    let (restored, _) = crate::vector::hnsw::MutableVectorOwner::restore_checkpoint(graph, rows).unwrap();
                    assert_eq!(restored.active_nodes(), 50);
                }
            }
            let reopened = Index::open_in(dir.clone()).unwrap();
            assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(50), "quantized {quantized}, write version {version}");
        }
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// Graph saves scheduled apart from text commits: a commit that keeps
    /// the previous graph checkpoint reopens with the graph caught up from
    /// the committed vectors.
    #[test]
    fn commit_without_graph_save_reopens_with_the_graph_caught_up() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let commit = |index: &Index, save: bool| -> u64 {
            let mut capture = index.begin_vector_capture(save).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let meta: Meta = serde_json::from_slice(dir.open_read(META_FILE).unwrap().as_slice()).unwrap();
            meta.vector_store.mutable_fields[0].generation
        };
        append(&mut index, &field, 0.);
        append(&mut index, &field, 3.);
        let saved = commit(&index, true);
        append(&mut index, &field, 7.);
        append(&mut index, &field, 9.);
        assert_eq!(commit(&index, false), saved, "the text commit keeps the graph checkpoint");
        // "Crash" here: reopen from the text commit and the older graph.
        let reopened = Index::open_in(dir.clone()).unwrap();
        assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(4));
        assert_eq!(hits(&reopened).len(), 4);
        assert_eq!(hits(&reopened), hits(&index));
    }

    /// A commit that reuses an older graph checkpoint keeps the vector rows
    /// that checkpoint references through vector merges (pinned until a
    /// newer checkpoint is durable); a checkpoint whose rows are gone anyway
    /// reopens by rebuilding the graph from the committed rows instead of
    /// failing the shard ("referenced source row missing").
    #[test]
    fn reused_graph_checkpoint_keeps_its_rows_through_vector_merges() {
        // Rows reference the vector store only in RMVR v3 (the next write
        // version); the compatible formats embed every f32.
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::RMVR3_SINCE));
        let commit = |index: &Index, save: bool| {
            let mut capture = index.begin_vector_capture(save).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
        };
        for unpinned in [false, true] {
            let dir = Arc::new(RamDirectory::default());
            let mut index = Index::create_in(dir.clone(), "pinned-rows", schema(), IndexSettings::default()).unwrap();
            index.set_nrt(true);
            index.enable_mutable_vectors().unwrap();
            for i in 0..40 {
                let mut document = Document::new();
                document.add_vector(1, vec![i as f32, 1.]);
                index.add_document(&document).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
            // The graph checkpoint references the sealed (durable) rows.
            commit(&index, true);
            let segment = index.segments[0].id;
            for doc in 0..20 {
                index.delete_doc(&DocAddress::new(segment, doc)).unwrap();
            }
            index.publish().unwrap();
            assert_eq!(index.vectors.mutable_active_nodes(1), Some(20), "the graph removed the deleted nodes");
            if unpinned {
                // Model rows lost regardless (e.g. a pin missed): the open
                // path must still recover.
                index.vectors.registered_mutable_field(1).unwrap().checkpoint_durable(u64::MAX);
            }
            // Combine the payloads: dead rows no pin keeps are dropped.
            index.vector_optimizer().force_merge(1, false).unwrap();
            let stored: usize = index.vectors.segments(1).iter().map(|s| s.len() as usize).sum();
            assert_eq!(stored, if unpinned { 20 } else { 40 }, "the merge keeps exactly the live and pinned rows");
            // A text commit reusing the checkpoint that still holds the 20 nodes.
            commit(&index, false);
            let reopened = Index::open_in(dir.clone()).unwrap();
            assert_eq!(reopened.vectors.mutable_active_nodes(1), Some(20), "unpinned: {unpinned}");
            assert_eq!(hits(&reopened).len(), 10, "unpinned: {unpinned}");
            // kNN through the reopened (or rebuilt) graph and the view's
            // vectors: only the 20 live documents, exact and approximate.
            let view = reopened.reader_view();
            let leaves = view.leaves();
            let ctx = crate::query::query::SearchContext::for_query(&crate::query::query::Query::MatchAll, &leaves, Bm25Params::default());
            for exact in [true, false] {
                let found = crate::query::mutable_knn::search(&leaves, 1, &[0., 1.], 10, 100, None, &ctx, exact, None).unwrap().unwrap().0;
                assert_eq!(found.len(), 10, "unpinned: {unpinned}, exact: {exact}");
                for (_, leaf, doc) in found {
                    let value = leaves[leaf].with_vector(1, doc, |row| row[0]).expect("a hit's vector is readable");
                    assert!(value >= 20., "unpinned: {unpinned}, exact: {exact}: deleted document {value} returned");
                }
            }
            // A rebuilt graph has no reusable checkpoint: the next commit captures it.
            assert_eq!(reopened.vectors.mutable_commit.lock().checkpointed.contains_key(&1), !unpinned);
        }
    }

    /// B1: a retained view whose graph pin is released (memory budget)
    /// serves kNN by exact search over its own vectors: the same hits,
    /// while writes continue.
    #[test]
    fn released_retained_graph_serves_exact_knn_with_the_same_hits() {
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "mutable", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let add = |index: &mut Index, rows: std::ops::Range<i32>| {
            for i in rows {
                let mut document = Document::new();
                document.add_vector(1, vec![i as f32, 1.]);
                index.add_document(&document).unwrap();
            }
        };
        add(&mut index, 0..40);
        index.publish().unwrap();
        index.seal().unwrap();
        let retained = index.view_slot().retain().unwrap();
        let search = |view: &crate::reader_view::ReaderView| -> Vec<(usize, DocId)> {
            let leaves = view.leaves();
            let ctx = crate::query::query::SearchContext::for_query(&crate::query::query::Query::MatchAll, &leaves, Bm25Params::default());
            crate::query::mutable_knn::search(&leaves, 1, &[3.2, 1.], 5, 100, None, &ctx, false, None)
                .unwrap().unwrap().0.into_iter().map(|(_, leaf, doc)| (leaf, doc)).collect()
        };
        let before = search(&retained);
        assert_eq!(before.len(), 5);
        let captured = retained.segments().iter().find_map(|segment| segment.vector_store()
            .and_then(|store| store.captured_mutable_field(1))).expect("retained capture");
        assert!(captured.release());
        assert!(captured.is_released());
        // Writers are unaffected; the released view keeps its own hits.
        add(&mut index, 40..60);
        index.publish().unwrap();
        assert_eq!(search(&retained), before);
    }

    /// Skip unchanged fields: a commit with no graph change reuses the last
    /// checkpoint file; an insert and a delete repair each make the next
    /// commit write the field again (compaction is covered by the graph's
    /// own relocation test); reopen restores from the reused file.
    #[test]
    fn unchanged_mutable_fields_reuse_their_checkpoint_and_mutations_rewrite_it() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let commit = |index: &Index| -> u64 {
            let mut capture = index.begin_vector_capture(true).unwrap();
            capture.finish().unwrap();
            index.prepare_commit_with(capture).unwrap().run().unwrap();
            let meta: Meta = serde_json::from_slice(dir.open_read(META_FILE).unwrap().as_slice()).unwrap();
            assert_eq!(meta.vector_store.mutable_fields.len(), 1);
            assert_eq!(dir.list().unwrap().iter().filter(|name| name.ends_with(".rmvc")).count(), 1,
                "exactly the referenced checkpoint file remains");
            meta.vector_store.mutable_fields[0].generation
        };
        append(&mut index, &field, 0.);
        append(&mut index, &field, 3.);
        let first = commit(&index);
        assert_eq!(commit(&index), first, "an unchanged graph reuses its checkpoint");
        append(&mut index, &field, 7.);
        let inserted = commit(&index);
        assert_ne!(inserted, first, "an insert rewrites the checkpoint");
        assert_eq!(commit(&index), inserted);
        // A document delete queues its point; the repair removes the node.
        index.segments[0].delete_doc(0).unwrap();
        field.drain_abandoned().unwrap();
        let repaired = commit(&index);
        assert_ne!(repaired, inserted, "a delete repair rewrites the checkpoint");
        assert_eq!(commit(&index), repaired);
        let reopened = Index::open_in(dir.clone()).unwrap();
        assert_eq!(hits(&reopened).len(), hits(&index).len());
    }

    #[test]
    fn mutable_commit_rejects_unstaged_points_and_reopen_rejects_schema_mismatch() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        append(&mut index, &field, 1.);
        index.commit().unwrap();
        let valid = dir.open_read(META_FILE).unwrap().as_slice().to_vec();
        let mut meta: Meta = serde_json::from_slice(&valid).unwrap();
        meta.schema[0].field_type = FieldType::Vector { dim: 3 };
        dir.write(META_FILE, &serde_json::to_vec(&meta).unwrap()).unwrap();
        assert!(Index::open_in(dir.clone()).is_err());
        dir.write(META_FILE, &valid).unwrap();
        let mut writer = SegmentWriter::new(99);
        let mut document = Document::new(); document.add_vector(1, vec![2.,1.]);
        writer.add_document(&document, &index.schema);
        // An explicit negative test of the old, unstaged write path.
        let segment = writer.write_to_store_ref(&*dir, Some(&index.vectors), false).unwrap();
        index.segments.push(Arc::new(segment));
        assert!(index.commit().is_err());
        assert_eq!(dir.open_read(META_FILE).unwrap().as_slice(), valid);
        assert_eq!(hits(&Index::open_in(dir).unwrap()).len(), 1);
    }

    #[test]
    fn mutable_document_admission_precedes_replacement_and_preserves_nested_order() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let mut first = Document::new(); first.add_vector(1, vec![1.,1.]);
        let old = index.add_document(&first).unwrap();
        let mut invalid = Document::new(); invalid.add_vector(1, vec![f32::NAN,1.]);
        assert!(index.replace_document(Some(&old), &invalid).is_err());
        assert_eq!(index.writers[0].num_docs(), 1);
        assert!(!index.writers[0].is_deleted(old.doc_id));
        let mut child = Document::new(); child.add_vector(1, vec![2.,1.]);
        let mut replacement = Document::new(); replacement.add_vector(1, vec![3.,1.]);
        replacement.add_vector(1, vec![99.,1.]); // first value wins, like the text writer
        replacement.add_nested("child", child);
        index.replace_document(Some(&old), &replacement).unwrap();
        assert!(index.writers[0].is_deleted(old.doc_id));
        let points = index.writers[0].reserved_vector_points(1);
        assert_eq!(points.len(), 3);
        assert_eq!(points.iter().map(|row| row.0).collect::<Vec<_>>(), vec![0,1,2]);
        field.drain_abandoned().unwrap();
        index.refresh().unwrap();
        index.commit().unwrap();
        assert_eq!(hits(&Index::open_in(dir).unwrap()).len(), 2);
    }

    #[test]
    fn mutable_query_dispatch_covers_published_and_sealed_documents() {
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        for n in 0..8 {
            let mut doc = Document::new(); doc.add_vector(1, vec![n as f32,1.]);
            index.add_document(&doc).unwrap();
        }
        index.publish().unwrap();
        let verify = |index: &Index| {
            let req = KnnRequest::new(1, vec![0.,1.], 3);
            let hits = index.knn_search(&req).unwrap();
            assert_eq!(hits.len(), 3);
            let top = Searcher::new(index.leaves()).search(&Query::knn(1, vec![0.,1.], 3),
                &SearchOptions { top_k: 3, ..Default::default() });
            assert_eq!(top.execution_error, None);
            assert_eq!(top.hits.len(), 3);
            assert_eq!(top.hits.iter().map(|h| (h.leaf,h.doc)).collect::<Vec<_>>(),
                hits.iter().map(|h| (h.leaf,h.doc)).collect::<Vec<_>>());
            // Exercise both public dispatch paths with a request override.
            // This must work before seal, after seal, and after reopening.
            let (rescored, stats) = index.knn_search_with(&req,
                &PlannerConfig::default().with_oversample(2.0)).unwrap();
            assert_eq!(rescored.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>(),
                hits.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>());
            // An f32 field ignores the override (ES `needsRescore`): k targets only.
            assert!(stats.vector_ops >= 3);
            let mut query = Query::knn(1, vec![0.,1.], 3);
            if let Query::Knn { oversample, .. } = &mut query { *oversample = Some(2.0); }
            let rescored_query = Searcher::new(index.leaves()).search(&query,
                &SearchOptions { top_k: 3, ..Default::default() });
            assert_eq!(rescored_query.execution_error, None);
            assert_eq!(rescored_query.hits.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>(),
                rescored.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>());
            assert!(index.knn_search_with(&req,
                &PlannerConfig::default().with_oversample(f32::NAN)).is_err());
            let bad = KnnRequest::new(1, vec![f32::NAN,1.], 3);
            assert!(index.knn_search(&bad).is_err());
            let failed = Searcher::new(index.leaves()).search(&Query::knn(1, bad.vector, 3),
                &SearchOptions { top_k: 3, ..Default::default() });
            assert!(failed.execution_error.is_some());
            let expired = Searcher::new(index.leaves()).search(&Query::knn(1, vec![0.,1.], 3),
                &SearchOptions { top_k: 3, timeout: Some(std::time::Duration::ZERO), ..Default::default() });
            assert!(expired.timed_out);
            assert!(expired.execution_error.is_none());
        };
        verify(&index);
        index.refresh().unwrap();
        verify(&index);
        index.commit().unwrap();
        verify(&Index::open_in(dir).unwrap());
    }

    #[test]
    fn mutable_query_counts_only_visible_leaves_while_detached_seal_has_registered_payloads() {
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        use crate::vector::quantize::Quantization;
        for quantization in [Quantization::None, Quantization::Int8, Quantization::Int4, Quantization::Binary] {
            let options = if quantization == Quantization::None { VectorFieldOptions::f32_hnsw() }
                else { VectorFieldOptions::quantized_hnsw(quantization) };
            let fields = vec![Field::new(1, "vector", FieldType::Vector { dim: 2 })
                .with_similarity("l2_norm").with_vector_options(Some(options))];
            let mut index = Index::create_in(Arc::new(RamDirectory::default()), "count", fields, IndexSettings::default()).unwrap();
            index.enable_mutable_vectors().unwrap();
            index.set_nrt(true);
            index.set_auto_seal(false);
            for i in 0..16 {
                let mut doc = Document::new(); doc.add_vector(1, vec![i as f32, 1.]);
                index.add_document(&doc).unwrap();
            }
            index.publish().unwrap();
            let verify = |index: &Index| {
                let result = Searcher::new(index.leaves()).search(&Query::knn(1, vec![0., 1.], 3),
                    &SearchOptions { top_k: 3, ..Default::default() });
                assert_eq!(result.execution_error, None);
                assert_eq!(result.hits.len(), 3);
            };
            verify(&index);
            let output = index.begin_seal(false).unwrap().unwrap().run();
            assert_eq!(index.vector_store().num_alive(1), 16);
            assert!(index.leaves().iter().any(|leaf| matches!(leaf, LeafReader::Memtable(_, ..))));
            verify(&index); // payload and memtable refer to the same 16 points
            index.finish_seal(output).unwrap();
            verify(&index);
        }
    }

    #[test]
    fn quantized_mutable_field_rescores_and_retains_pit_across_sorted_merge_and_failed_commit() {
        use crate::vector::quantize::Quantization;
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        use crate::index_sort::{IndexSort, IndexSortField, IndexSortKind};
        for quantization in [Quantization::Int8, Quantization::Int4, Quantization::Binary] {
            for sorted in [false, true] {
                let dir = Arc::new(MetaFailureDirectory::default());
                let options = VectorFieldOptions::quantized_hnsw(quantization).with_osq(true);
                let fields = vec![Field::new(0, "rank", FieldType::I64),
                    Field::new(1, "vector", FieldType::Vector { dim: 13 }).with_similarity("l2_norm")
                        .with_vector_options(Some(options))];
                let mut index = Index::create_in(dir.clone(), "coded", fields, IndexSettings::default()).unwrap();
                if sorted { index.set_index_sort(Some(IndexSort::new(vec![IndexSortField {
                    field: 0, kind: IndexSortKind::Long, reverse: false, max_mode: false, missing_first: false,
                }]))); }
                index.enable_mutable_vectors().unwrap();
                let raw = |i: usize| (0..13).map(|d| ((i * 17 + d * 7) % 101) as f32 / 37.0 - 1.0).collect::<Vec<_>>();
                for i in 0..160 {
                    let mut doc = Document::new(); doc.add_i64(0, (160-i) as i64); doc.add_vector(1, raw(i));
                    index.add_document(&doc).unwrap();
                    if i % 40 == 39 { index.refresh().unwrap(); }
                }
                index.commit().unwrap();
                let verify = |index: &Index| {
                    let req = KnnRequest::new(1, raw(7), 10);
                    let (hits, _) = index.knn_search_with(&req, &PlannerConfig::default().with_oversample(3.0)).unwrap();
                    assert_eq!(hits.len(), 10);
                    for hit in hits {
                        let expected = index.leaves()[hit.leaf].with_vector(1, hit.doc,
                            |row| Similarity::L2.score(&req.vector, row)).unwrap();
                        assert_eq!(hit.score.to_bits(), expected.to_bits());
                    }
                    let (approximate, _) = index.knn_search_with(&req, &PlannerConfig::default().with_oversample(0.0)).unwrap();
                    assert_eq!(approximate.len(), 10);
                    approximate.into_iter().map(|hit| hit.score.to_bits()).collect::<Vec<_>>()
                };
                let before = verify(&index);
                assert_eq!(verify(&Index::open_in(dir.clone()).unwrap()), before, "restore must not refit codes");
                let captured = crate::segment::Segment::capture_read_view(index.segments()).unwrap();
                let retained_query = Query::knn(1, raw(7), 10);
                let old = Searcher::new(captured.iter().map(|segment| LeafReader::Segment(segment)).collect())
                    .search(&retained_query, &SearchOptions { top_k: 10, ..Default::default() });
                assert!(old.execution_error.is_none());
                let hit = index.knn_search(&KnnRequest::new(1, raw(7), 1)).unwrap()[0].clone();
                index.delete_doc(&DocAddress::new(hit.leaf as u32, hit.doc)).unwrap();
                index.refresh().unwrap();
                index.force_merge().unwrap();
                assert!(index.vectors.segments(1).iter().all(|s| !s.has_graph()));
                let retained = Searcher::new(captured.iter().map(|segment| LeafReader::Segment(segment)).collect())
                    .search(&retained_query, &SearchOptions { top_k: 10, ..Default::default() });
                assert_eq!(retained.hits.iter().map(|h| (h.leaf, h.doc, h.score.to_bits())).collect::<Vec<_>>(),
                    old.hits.iter().map(|h| (h.leaf, h.doc, h.score.to_bits())).collect::<Vec<_>>());
                dir.fail_meta.store(true, Ordering::Release);
                assert!(index.commit().is_err());
                dir.fail_meta.store(false, Ordering::Release);
                assert_eq!(verify(&Index::open_in(dir.clone()).unwrap()), before, "failed commit preserves previous codes/graph");
                index.commit().unwrap();
                assert_eq!(verify(&Index::open_in(dir.clone()).unwrap()), verify(&index));
                let mut invalid = index.schema.clone();
                invalid[1].vector_options = Some(VectorFieldOptions::f32_hnsw());
                assert_eq!(index.vector_fields_to_migrate(&invalid), vec![1], "a code format change rebuilds the graph");
            }
        }
    }

    #[test]
    fn mutable_filtered_projection_survives_delete_merge_and_reopen() {
        use crate::query::query::Query;
        for filter in [None, Some(Query::MatchAll)] {
            let dir = Arc::new(RamDirectory::default());
            let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
            index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
            for batch in 0..2 {
                for n in 0..8 {
                    let mut doc = Document::new(); doc.add_vector(1, vec![(batch*8+n) as f32,1.]);
                    index.add_document(&doc).unwrap();
                }
                index.refresh().unwrap();
            }
            let mut req = KnnRequest::new(1, vec![0.,1.], 4);
            req.filter = filter;
            let before = index.knn_search(&req).unwrap();
            assert_eq!(before.len(),4);
            let repeated = index.knn_search(&req).unwrap();
            assert_eq!(before.iter().map(|h| (h.leaf,h.doc)).collect::<Vec<_>>(), repeated.iter().map(|h| (h.leaf,h.doc)).collect::<Vec<_>>());
            let LeafReader::Segment(segment) = index.leaves()[before[0].leaf] else { panic!("sealed fixture") };
            let address = DocAddress::new(segment.id, before[0].doc);
            index.delete_doc(&address).unwrap();
            index.publish().unwrap();
            let after = index.knn_search(&req).unwrap();
            assert_eq!(after.len(),4);
            assert!(after[0].score < before[0].score);
            let expected: Vec<_> = after.iter().map(|h| h.score).collect();
            index.force_merge().unwrap();
            assert_eq!(index.knn_search(&req).unwrap().iter().map(|h| h.score).collect::<Vec<_>>(),expected);
            index.commit().unwrap();
            let reopened = Index::open_in(dir).unwrap();
            assert_eq!(reopened.knn_search(&req).unwrap().iter().map(|h| h.score).collect::<Vec<_>>(),expected);
        }
    }

    #[test]
    fn mutable_deletes_wait_for_refresh_and_preserve_retained_graph() {
        use crate::vector::vespa_policy::*;
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir, "mutable", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        for n in 0..4 {
            let mut doc = Document::new(); doc.add_vector(1, vec![n as f32,1.]);
            index.add_document(&doc).unwrap();
        }
        index.publish().unwrap();
        let retained = index.vectors.capture_read_store().unwrap();
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 4, explore_additional_hits: 40 }, 5, PreparedGlobalFilter::Inactive).unwrap();
        let count = |store: &VectorStore| store.search_mutable_ann(1, &[0.,1.], &plan,
            |_| true, None).unwrap().unwrap().hits.len();
        index.delete_doc(&DocAddress::new(0, 0)).unwrap();
        assert_eq!(count(&index.vectors), 4, "unpublished delete must retain graph node");
        index.publish().unwrap();
        assert_eq!(count(&index.vectors), 3, "refresh physically removes node");
        assert_eq!(count(&retained), 4, "retained topology and source survive removal");
        index.refresh().unwrap();
        index.delete_doc(&DocAddress::new(0, 1)).unwrap();
        assert_eq!(count(&index.vectors), 3);
        index.publish().unwrap();
        assert_eq!(count(&index.vectors), 2, "sealed document deletion uses same graph owner");
        assert_eq!(count(&retained), 4);
    }

    #[test]
    fn mutable_only_optimizer_backlog_survives_delete_and_reopen() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable-maintenance", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let mut doc = Document::new(); doc.add_vector(1, vec![1., 2.]);
        index.add_document(&doc).unwrap();
        index.refresh().unwrap();
        let retained = Segment::capture_read_view(index.segments()).unwrap();
        let job = index.vector_optimizer();
        assert_eq!(job.backlog().mutable_fields, 1);
        assert_eq!(job.index_pending().unwrap(), OptimizeReport::default());
        assert_eq!(job.maintain_once().unwrap(), OptimizeReport::default());
        let query = KnnRequest::new(1, vec![1., 2.], 10);
        let hits = index.knn_search(&query).unwrap();
        assert_eq!(hits.len(), 1);
        index.delete_doc(&DocAddress::new(hits[0].leaf as u32, hits[0].doc)).unwrap();
        index.publish().unwrap();
        // Empty live graphs can still hold retired storage; keep scheduling.
        assert_eq!(job.backlog().mutable_fields, 1);
        for _ in 0..3 { job.maintain_once().unwrap(); }
        assert_eq!(job.backlog().unsealed_segments + job.backlog().sealed_segments, 0);
        assert_eq!(job.backlog().mutable_fields, 1);
        assert!(index.knn_search(&query).unwrap().is_empty());
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        let old = Searcher::new(retained.iter().map(|segment| LeafReader::Segment(segment)).collect())
            .search(&Query::knn(1, vec![1.,2.], 10), &SearchOptions { top_k: 10, ..Default::default() });
        assert!(old.execution_error.is_none());
        assert_eq!(old.hits.len(), 1);
        drop(retained);
        job.maintain_once().unwrap();
        index.commit().unwrap();
        let reopened = Index::open_in(dir).unwrap();
        assert_eq!(reopened.vector_optimizer().backlog().mutable_fields, 1);
        reopened.vector_optimizer().maintain_once().unwrap();
        assert!(reopened.knn_search(&query).unwrap().is_empty());
    }

    #[test]
    fn unchanged_mutable_updates_reuse_topology_across_hidden_flush_and_pit() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true); index.enable_mutable_vectors().unwrap();
        let mut doc = Document::new(); doc.add_vector(1, vec![1.,2.]);
        index.add_document(&doc).unwrap(); index.refresh().unwrap();
        let retained = Segment::capture_read_view(index.segments()).unwrap();
        let mut address = DocAddress::new(0, 0);
        for _ in 0..3 { address = index.replace_document(Some(&address), &doc).unwrap(); }
        assert_eq!(index.vectors.next_point_id(), 1, "unchanged geometry must not allocate graph points");
        assert_eq!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 1);
        index.seal().unwrap();
        assert_eq!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 1);
        index.publish().unwrap();
        let hits = index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap();
        assert_eq!(hits.len(), 1);
        let address = DocAddress::new(hits[0].leaf as u32, hits[0].doc);
        index.delete_doc(&address).unwrap(); index.delete_doc(&address).unwrap();
        index.publish().unwrap();
        assert!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().is_empty());
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        let old = Searcher::new(retained.iter().map(|segment| LeafReader::Segment(segment)).collect())
            .search(&Query::knn(1, vec![1.,2.], 10), &SearchOptions { top_k: 10, ..Default::default() });
        assert!(old.execution_error.is_none()); assert_eq!(old.hits.len(), 1);
        index.commit().unwrap();
        let reopened = Index::open_in(dir).unwrap();
        assert!(reopened.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().is_empty());
    }

    /// Rows stored before the field had a graph get one built from them in
    /// the background (ES indexes existing data in the new format); exact
    /// search serves the field meanwhile.
    #[test]
    fn mutable_backend_activation_builds_the_graph_from_stored_rows() {
        let mut fresh = Index::new("mutable", schema(), IndexSettings::default());
        fresh.enable_mutable_vectors().unwrap();
        let mut doc = Document::new(); doc.add_vector(1, vec![1.,2.]);
        fresh.add_document(&doc).unwrap(); fresh.refresh().unwrap();
        assert_eq!(fresh.knn_search(&KnnRequest::new(1, vec![1.,2.], 1)).unwrap().len(), 1);
        let mut existing = Index::new("segmented", schema(), IndexSettings::default());
        existing.add_document(&doc).unwrap(); existing.refresh().unwrap();
        existing.enable_mutable_vectors().unwrap();
        assert_eq!(existing.vectors.migrations_pending(), 1, "scheduled, not built under the caller");
        assert_eq!(existing.knn_search(&KnnRequest::new(1, vec![1.,2.], 1)).unwrap().len(), 1, "exact search meanwhile");
        assert_eq!(existing.complete_vector_migrations().unwrap(), 1);
        assert_eq!(existing.vectors.mutable_active_nodes(1), Some(1));
        assert_eq!(existing.knn_search(&KnnRequest::new(1, vec![1.,2.], 1)).unwrap().len(), 1);
    }

    /// `flat` → `hnsw` with rows in every state (memtable, published
    /// memtable, hidden sealed, published sealed): the background migration
    /// builds the graph from the stored rows, and rows still in memtables
    /// enter it when their seal stores them; later writes join the memtable
    /// that holds them.
    #[test]
    fn mapping_upgrade_to_a_graph_builds_it_from_rows_in_every_state() {
        for phase in 0..4 {
            let mut index = Index::new("legacy-upgrade", schema(), IndexSettings::default());
            index.set_nrt(true);
            index.set_auto_seal(false);
            index.schema.iter_mut().find(|f| f.id == 1).unwrap().vector_options.as_mut().unwrap().graph = false;
            let mut doc = Document::new(); doc.add_vector(1, vec![1., 2.]);
            index.add_document(&doc).unwrap();
            match phase {
                0 => {} // writing memtable
                1 => { index.publish().unwrap(); } // published/frozen memtable
                2 => { index.seal().unwrap(); } // hidden sealed rows
                _ => { index.refresh().unwrap(); } // published sealed rows
            }
            let mut upgraded = index.schema.clone();
            upgraded.iter_mut().find(|f| f.id == 1).unwrap().vector_options.as_mut().unwrap().graph = true;
            assert_eq!(index.vector_fields_to_migrate(&upgraded), vec![1], "phase {phase}");
            index.schema = upgraded;
            index.enable_new_mutable_vector_fields().unwrap();
            assert_eq!(index.vectors.migrations_pending(), 1, "phase {phase}: scheduled by activation");
            assert_eq!(index.complete_vector_migrations().unwrap(), 1, "phase {phase}");
            assert!(index.vectors.registered_mutable_field(1).is_some(), "phase {phase}");
            assert_eq!(index.vectors.mutable_active_nodes(1), Some(if phase < 2 { 0 } else { 1 }), "phase {phase}: stored rows");
            index.add_document(&doc).unwrap();
            index.refresh().unwrap();
            assert_eq!(index.vectors.mutable_active_nodes(1), Some(2), "phase {phase}: the sealed memtable rows joined");
            assert_eq!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 2, "phase {phase}");
        }
    }

    #[test]
    fn mutable_mapping_upgrade_rebuilds_the_graph_and_preserves_hidden_replacements_pit_and_restart() {
        use crate::vector::quantize::Quantization;
        use crate::vector::vespa_policy::*;
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        for options in [VectorFieldOptions::quantized_hnsw(Quantization::Int8),
            VectorFieldOptions::f32_hnsw().with_hnsw(24, 200)] {
            let dir = Arc::new(RamDirectory::default());
            let mut index = Index::create_in(dir.clone(), "upgrade", schema(), IndexSettings::default()).unwrap();
            index.set_nrt(true); index.enable_mutable_vectors().unwrap();
            let mut doc = Document::new(); doc.add_vector(1, vec![1., 2.]);
            index.add_document(&doc).unwrap(); index.refresh().unwrap(); index.commit().unwrap();
            let retained = Segment::capture_read_view(index.segments()).unwrap();
            let in_flight = index.vectors.mutable_query_field(1).unwrap();
            let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
                estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                    HitBudget { target_hits: 10, explore_additional_hits: 90 }, in_flight.document_limit(), PreparedGlobalFilter::Inactive).unwrap();
            index.replace_document(Some(&DocAddress::new(0, 0)), &doc).unwrap();
            let mut updated = index.schema.clone();
            updated.iter_mut().find(|f| f.id == 1).unwrap().vector_options = Some(options);
            index.prepare_vector_schema_change(&updated).unwrap();
            let migrate = index.vector_fields_to_migrate(&updated);
            assert_eq!(migrate, vec![1]);
            index.schema = updated;
            index.sync_all_vector_field_options();
            index.request_vector_migrations(&migrate).unwrap();
            // Writes keep going to the current graph until the swap.
            assert!(index.vectors.registered_mutable_field(1).is_some());
            assert_eq!(index.complete_vector_migrations().unwrap(), 1);
            assert!(index.vectors.registered_mutable_field(1).unwrap().matches_index_options(options), "rebuilt with the new options");
            assert_eq!(in_flight.search(&[1.,2.], &plan, |_| true, None).unwrap().hits.len(), 1,
                "an already selected live backend survives retirement");
            assert_eq!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 1);
            index.add_document(&doc).unwrap(); index.refresh().unwrap();
            assert_eq!(index.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 2);
            let old = Searcher::new(retained.iter().map(|s| LeafReader::Segment(s)).collect())
                .search(&Query::knn(1, vec![1.,2.], 10), &SearchOptions { top_k: 10, ..Default::default() });
            assert!(old.execution_error.is_none()); assert_eq!(old.hits.len(), 1);
            index.commit().unwrap();
            let reopened = Index::open_in(dir).unwrap();
            assert_eq!(reopened.knn_search(&KnnRequest::new(1, vec![1.,2.], 10)).unwrap().len(), 2);
        }
    }

    #[test]
    fn parallel_mutable_document_writers_publish_all_reserved_points() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        let documents: Vec<_> = (0..64).map(|n| {
            let mut doc = Document::new(); doc.add_vector(1, vec![n as f32,1.]); doc
        }).collect();
        assert_eq!(index.add_documents_parallel(&documents, 4).unwrap().len(), 64);
        let mut points: Vec<_> = index.writers.iter().flat_map(|writer| writer.reserved_vector_points(1))
            .map(|row| row.1).collect();
        points.sort_unstable();
        assert_eq!(points, (0..64).collect::<Vec<_>>());
        index.refresh().unwrap();
        index.commit().unwrap();
        assert_eq!(hits(&Index::open_in(dir).unwrap()).len(), 10);
    }

    fn background_vector_index(hidden: bool) -> Index {
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "mutable", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        for batch in 0..4 {
            for n in 0..3 {
                let mut doc = Document::new(); doc.add_vector(1, vec![(batch * 3 + n) as f32, 1.]);
                index.add_document(&doc).unwrap();
            }
            if !hidden { index.publish().unwrap(); }
            index.seal().unwrap();
        }
        index
    }

    fn small_merge_policy() -> crate::merge::MergePolicy {
        crate::merge::MergePolicy { max_segments_per_tier: 3, floor_segment_docs: 1, ..Default::default() }
    }

    #[test]
    fn default_policy_compacts_tiny_mutable_segments() {
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "tiny", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.set_auto_seal(false);
        index.enable_mutable_vectors().unwrap();
        for batch in 0..40 {
            for n in 0..8 {
                let mut doc = Document::new(); doc.add_vector(1, vec![(batch * 8 + n) as f32, 1.]);
                index.add_document(&doc).unwrap();
            }
            index.publish().unwrap();
            index.seal().unwrap();
        }
        assert_eq!(index.segments.len(), 40);
        let before = hits(&index);
        let policy = crate::merge::MergePolicy::default();
        let mut count = 0;
        while let Some(job) = index.begin_merge(&policy) {
            let merged = job.run().unwrap();
            assert!(index.finish_merge(job, merged).unwrap());
            count += 1;
            assert!(count < 40, "merge policy must converge");
        }
        assert!(index.segments.len() <= policy.max_segments_per_tier);
        assert_eq!(hits(&index), before);
        assert_eq!(index.segments.iter().map(|s| s.num_live_docs()).sum::<u32>(), 320);
        index.commit().unwrap();
        assert_eq!(hits(&Index::open_in(index.directory.clone()).unwrap()), before);
    }

    #[test]
    fn mutable_background_text_merge_defers_ownership_and_preserves_pit_deletes() {
        use crate::query::{query::Query, searcher::{Searcher, SearchOptions}};
        let mut index = background_vector_index(false);
        let retained = Segment::capture_read_view(index.segments()).unwrap();
        let job = index.begin_merge(&small_merge_policy()).unwrap();
        let owners: Vec<_> = (0..12).map(|p| index.vectors.owner(p)).collect();
        let merged = job.run().unwrap();
        assert!(merged.vector_store().is_none());
        assert_eq!((0..12).map(|p| index.vectors.owner(p)).collect::<Vec<_>>(), owners);
        let leaf = index.segments.iter().position(|s| Arc::ptr_eq(s, &job.chosen[0])).unwrap();
        index.delete_doc(&DocAddress::new(leaf as u32, 0)).unwrap();
        index.publish().unwrap();
        index.delete_doc(&DocAddress::new(leaf as u32, 1)).unwrap();
        assert!(index.finish_merge(job, merged).unwrap());
        assert_eq!(index.segments.len(), 2);
        let count = |index: &Index| index.knn_search(&KnnRequest::new(1, vec![0.,1.], 20)).unwrap().len();
        assert_eq!(count(&index), 11, "unpublished deletion remains searchable");
        index.publish().unwrap();
        assert_eq!(count(&index), 10);
        let old = Searcher::new(retained.iter().map(|s| LeafReader::Segment(s)).collect())
            .search(&Query::knn(1, vec![0.,1.], 20), &SearchOptions { top_k: 20, ..Default::default() });
        assert!(old.execution_error.is_none()); assert_eq!(old.hits.len(), 12);
        index.commit().unwrap();
        assert_eq!(count(&Index::open_in(index.directory.clone()).unwrap()), 10);
    }

    #[test]
    fn mutable_background_text_merge_rejected_job_does_not_move_owners() {
        let mut index = background_vector_index(false);
        let job = index.begin_merge(&small_merge_policy()).unwrap();
        let merged = job.run().unwrap();
        index.force_merge().unwrap();
        let owners: Vec<_> = (0..12).map(|p| index.vectors.owner(p)).collect();
        assert!(!index.finish_merge(job, merged).unwrap());
        assert_eq!((0..12).map(|p| index.vectors.owner(p)).collect::<Vec<_>>(), owners);
        assert_eq!(index.knn_search(&KnnRequest::new(1, vec![0.,1.], 20)).unwrap().len(), 12);
    }

    #[test]
    fn mutable_background_text_merge_keeps_hidden_points_hidden() {
        for publish_during_merge in [false, true] {
            let mut index = background_vector_index(true);
            let job = index.begin_merge(&small_merge_policy()).unwrap();
            let merged = job.run().unwrap();
            if publish_during_merge { index.publish().unwrap(); }
            assert!(index.finish_merge(job, merged).unwrap());
            assert_eq!(index.knn_search(&KnnRequest::new(1, vec![0.,1.], 20)).unwrap().len(),
                if publish_during_merge { 12 } else { 0 });
            index.publish().unwrap();
            assert_eq!(index.knn_search(&KnnRequest::new(1, vec![0.,1.], 20)).unwrap().len(), 12);
        }
    }

    #[test]
    fn mutable_source_force_merge_does_not_build_segment_graphs() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        for value in 0..12 { append(&mut index, &field, value as f32); }
        let before = hits(&index);
        let retained = index.vectors.capture_read_store().unwrap();
        index.force_merge_vectors(1, false).unwrap();
        let segments = index.vectors.segments(1);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].is_sealed());
        assert!(!segments[0].has_graph(), "source compaction must never rebuild field topology");
        assert_eq!(hits(&index), before);
        index.commit().unwrap();
        assert_eq!(hits(&Index::open_in(dir).unwrap()), before);
        // An older retained view still owns its sources after source compaction.
        use crate::vector::vespa_policy::*;
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 10, explore_additional_hits: 90 }, 12, PreparedGlobalFilter::Inactive).unwrap();
        assert_eq!(retained.search_mutable_ann(1, &[0.,1.], &plan,
            |key| retained.owner(key).is_some(), None).unwrap().unwrap().hits, before);
    }

    /// Mutable-graph memtables coalesce: published and parked memtables of
    /// a mutable HNSW field seal into one segment per visibility class;
    /// their points move from the memtables (writer id, doc) to the segment
    /// (no stale memtable owner), and kNN is identical to sealing each
    /// memtable on its own, also after a reopen.
    #[test]
    fn coalesced_mutable_graph_memtables_move_their_owners() {
        let build = |coalesce: bool| -> (Arc<RamDirectory>, Index, Vec<u32>) {
            let dir = Arc::new(RamDirectory::default());
            let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
            index.set_nrt(true);
            index.set_auto_seal(false);
            let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
            let mut writers = Vec::new();
            for m in 0..5u32 {
                let id = index.next_segment_id;
                index.next_segment_id += 1;
                let mut w = SegmentWriter::new(id);
                for k in 0..3u32 {
                    let mut d = Document::new();
                    d.add_vector(1, vec![(m * 3 + k) as f32, 1.]);
                    w.add_document(&d, &index.schema);
                }
                w.stage_mutable_field(1, &field).unwrap();
                writers.push(id);
                if m < 3 { index.frozen.push(Arc::new(w)); } else { index.parked.push(Arc::new(w)); }
            }
            index.publish_view().unwrap();
            if coalesce {
                let published = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("published");
                assert_eq!(published.memtable_ids().len(), 3);
                let parked = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("parked");
                assert_eq!(parked.memtable_ids().len(), 2);
                index.finish_seal(published.run()).unwrap();
                index.finish_seal(parked.run()).unwrap();
                assert_eq!((index.segments().len(), index.hidden_segments().len()), (1, 1));
            } else {
                while let Some(job) = index.begin_seal_part(false, 1).unwrap() {
                    index.finish_seal(job.run()).unwrap();
                }
            }
            index.publish().unwrap();
            (dir, index, writers)
        };
        let (_, single, _) = build(false);
        let (dir, coalesced, writers) = build(true);
        assert_eq!(hits(&coalesced), hits(&single), "kNN changed by the coalesced seal");
        for p in 0..coalesced.vectors.next_point_id() {
            let owner = coalesced.vectors.owner(p);
            assert!(owner.is_some_and(|(seg, _)| !writers.contains(&seg)), "point {p} kept a memtable owner {owner:?}");
        }
        coalesced.commit().unwrap();
        let expected = hits(&coalesced);
        drop(coalesced);
        assert_eq!(hits(&Index::open_in(dir).unwrap()), expected, "kNN changed by the reopen");
    }

    fn store_hits(store: &VectorStore) -> Vec<(u64, f32)> {
        use crate::vector::vespa_policy::*;
        let count = store.next_point_id() as u32;
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. }, PolicyConfig::default()).unwrap().finish(
                HitBudget { target_hits: 10, explore_additional_hits: 90 }, count, PreparedGlobalFilter::Inactive).unwrap();
        store.search_mutable_ann(1, &[0.,1.], &plan, |key| store.owner(key).is_some(), None).unwrap().unwrap().hits
    }

    /// Coalescing (b): kNN queries running while a coalesced seal moves the
    /// points' owners (attach_coalesced) see the same hits throughout: no
    /// missing or duplicate point, no panic; afterwards the alive points
    /// are exactly the live documents.
    #[test]
    fn knn_during_a_coalesced_install_sees_every_point_once() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir, "mutable", schema(), IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.set_auto_seal(false);
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        for m in 0..5u32 {
            let id = index.next_segment_id;
            index.next_segment_id += 1;
            let mut w = SegmentWriter::new(id);
            for k in 0..3u32 {
                let mut d = Document::new();
                d.add_vector(1, vec![(m * 3 + k) as f32, 1.]);
                w.add_document(&d, &index.schema);
            }
            w.stage_mutable_field(1, &field).unwrap();
            index.frozen.push(Arc::new(w));
        }
        index.publish_view().unwrap();
        let store = index.vectors.clone();
        // Encoded: the memtables' points are in the store, owned by their
        // memtables (before the encode a memtable's rows have no owner yet).
        let job = index.begin_seal_bytes(false, Index::COALESCE_MAX_BYTES).unwrap().expect("a job");
        let sealed = job.run();
        let baseline = store_hits(&store);
        assert_eq!(baseline.len(), 10);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (store, stop, baseline) = (store.clone(), stop.clone(), baseline.clone());
            std::thread::spawn(move || {
                let mut rounds = 0;
                while !stop.load(Ordering::Acquire) || rounds == 0 {
                    let hits = store_hits(&store);
                    let mut points: Vec<u64> = hits.iter().map(|h| h.0).collect();
                    points.sort();
                    points.dedup();
                    assert_eq!(points.len(), hits.len(), "a duplicate point: {hits:?}");
                    assert_eq!(hits, baseline, "hits changed during the install");
                    rounds += 1;
                }
                rounds
            })
        };
        // The install moves every owner while the reader searches.
        index.finish_seal(sealed).unwrap();
        stop.store(true, Ordering::Release);
        assert!(reader.join().expect("a concurrent kNN panicked") > 0);
        assert_eq!(index.segments().len(), 1);
        assert_eq!(store.num_alive(1), 15, "alive points differ from the live documents");
        assert_eq!(store_hits(&store), baseline);
    }

    #[test]
    fn recovery_prunes_uncommitted_graph_points_without_exposing_documents() {
        let dir = Arc::new(RamDirectory::default());
        let mut index = Index::create_in(dir.clone(), "mutable", schema(), IndexSettings::default()).unwrap();
        let field = index.vectors.mutable_field(1, 2, Similarity::L2, HnswParams::default()).unwrap();
        append(&mut index, &field, 1.);
        // Graph ready, but this writer is absent from the committed text manifest.
        let mut pending = SegmentWriter::new(99);
        let mut document = Document::new(); document.add_vector(1, vec![0.,1.]);
        pending.add_document(&document, &index.schema);
        pending.stage_mutable_field(1, &field).unwrap();
        index.commit().unwrap();
        let reopened = Index::open_in(dir).unwrap();
        assert_eq!(hits(&reopened).iter().map(|h| h.0).collect::<Vec<_>>(), vec![0]);
        assert_eq!(reopened.vectors.next_point_id(), 2, "uncommitted identities are never reused");
    }
}

/// A seal chunk of at least this many memtables averaging at most
/// [`TINY_MEMTABLE_DOCS`] documents is coalesced into one segment even when
/// the whole capture is over the coalescing bound.
pub(crate) const MANY_MEMTABLES: usize = 64;
/// See [`MANY_MEMTABLES`].
pub(crate) const TINY_MEMTABLE_DOCS: u64 = 32;

/// The coalescing bound [`Index::claim_seal`] gets for one seal chunk:
/// `max_bytes` (one segment) when the whole capture coalesces, or when the
/// chunk is a pile of tiny memtables; else 0 (a segment per memtable,
/// encoded in parallel, sparing them a second pass inside the flush).
///
/// A capture over the coalescing bound used to be written memtable by
/// memtable whatever the memtables. After a write backlog (a paused copy
/// whose background seals fell behind a long flush), thousands of
/// memtables of a few documents each, one per concurrent writer each time
/// the open count passed its cap, made thousands of 1-10 document segments
/// in one flush (chaos seed 3: 11,241 segments), which the
/// merges and a replica recovery then could not catch up with. Ordinary
/// bulk memtables (hundreds of documents) keep the parallel per-memtable
/// encoding a flush beside concurrent bulks relies on.
fn chunk_coalesce_bound(coalesce: bool, chunk: &[(Arc<SegmentWriter>, bool)], max_bytes: u64) -> u64 {
    let docs: u64 = chunk.iter().map(|(w, _)| w.num_docs() as u64).fold(0, u64::saturating_add);
    if coalesce || pile_of_tiny_memtables(docs, chunk.len()) { max_bytes } else { 0 }
}

/// The chunk size a foreground capture of `capture` memtable bytes is cut
/// into: at most `max_bytes`, and small enough that the capture makes
/// [`Index::MAX_PARALLEL_SEALS`] chunks, which encode at once (a refresh
/// or flush waits for them; one coalesced chunk is one encode thread), but
/// at least [`MIN_PARALLEL_CHUNK_BYTES`] (no tiny segments from a small
/// capture).
fn parallel_chunk_bytes(capture: u64, max_bytes: u64) -> u64 {
    let max_bytes = max_bytes.max(1);
    capture.div_ceil(Index::MAX_PARALLEL_SEALS as u64).clamp(MIN_PARALLEL_CHUNK_BYTES.min(max_bytes), max_bytes)
}

/// See [`parallel_chunk_bytes`].
const MIN_PARALLEL_CHUNK_BYTES: u64 = 16 << 20;

/// Whether `count` memtables holding `docs` documents in all are many and
/// tiny ([`MANY_MEMTABLES`], [`TINY_MEMTABLE_DOCS`]).
fn pile_of_tiny_memtables(docs: u64, count: usize) -> bool {
    count >= MANY_MEMTABLES && docs <= TINY_MEMTABLE_DOCS.saturating_mul(count as u64)
}

/// The memtable bytes one seal job coalesces with `backlog` bytes unclaimed:
/// `max_bytes`, or a quarter of a large backlog, up to
/// [`Index::COALESCE_BACKLOG_MAX_BYTES`] (a backlog makes a few large
/// segments, not one per `max_bytes`).
fn coalesce_target(max_bytes: u64, backlog: u64) -> u64 {
    max_bytes.max((backlog / 4).min(Index::COALESCE_BACKLOG_MAX_BYTES))
}

#[cfg(test)]
mod coalesce_target_tests {
    use super::*;

    #[test]
    fn a_large_backlog_seals_into_a_few_large_segments() {
        let m = Index::COALESCE_MAX_BYTES;
        assert_eq!(coalesce_target(m, 64 << 20), m, "a refresh's worth: unchanged");
        assert_eq!(coalesce_target(m, 2 << 30), 512 << 20, "2 GiB: four jobs");
        assert_eq!(coalesce_target(m, 40 << 30), Index::COALESCE_BACKLOG_MAX_BYTES, "capped");
        let jobs = |backlog: u64| backlog.div_ceil(coalesce_target(m, backlog));
        assert!(jobs(10 << 30) <= 20, "10 GiB backlog: {} segments", jobs(10 << 30));
    }
}

/// so_vector's `exists: acceptedAnswerId` kNN filter: a keyword with a
/// distinct value on a share of the documents. Its filter bits were built
/// on every query by listing every doc and decoding its term, after
/// force-merge over one big segment; now they are the column's presence
/// words, cached on the segment.
#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod exists_filter_knn_tests {
    use super::*;
    use crate::query::{Bound, Query};
    use crate::vector::store::VectorFieldOptions;

    fn vector(i: usize, dim: usize) -> Vec<f32> {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..dim).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }).collect()
    }

    /// `docs` docs of `dim` dims; doc `i` has keyword field 2 (a distinct
    /// value) when `i % 4 == 0`, and numeric field 3 = `i % 10`. One segment.
    fn build(docs: usize, dim: usize) -> Index {
        let schema = vec![
            Field::new(1, "v", FieldType::Vector { dim: dim as u16 }).with_similarity("dot_product")
                .with_vector_options(Some(VectorFieldOptions::f32_hnsw())),
            Field::new(2, "accepted", FieldType::Keyword),
            Field::new(3, "rnd", FieldType::I64),
        ];
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "exists-knn", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        for i in 0..docs {
            let mut document = Document::new();
            let mut v = vector(i, dim);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v { *x /= norm; }
            document.add_vector(1, v);
            if i % 4 == 0 { document.add_keyword(2, &format!("a{i}")); }
            document.add_i64(3, (i % 10) as i64);
            index.add_document(&document).unwrap();
        }
        index.publish().unwrap();
        index.seal().unwrap();
        index
    }

    fn knn(index: &Index, query: &[f32], filter: Option<&Query>) -> Vec<(f32, usize, DocId)> {
        let view = index.reader_view();
        let leaves = view.leaves();
        let ctx = crate::query::query::SearchContext::for_query(filter.unwrap_or(&Query::MatchAll), &leaves, Bm25Params::default());
        crate::query::mutable_knn::search(&leaves, 1, query, 10, 50, filter, &ctx, false, None).unwrap().unwrap().0
    }

    #[test]
    fn exists_filter_bits_are_the_keyword_columns_presence() {
        let index = build(2_000, 8);
        let exists = Query::Exists { field: 2 };
        let view = index.reader_view();
        let leaves = view.leaves();
        let ctx = crate::query::query::SearchContext::for_query(&exists, &leaves, Bm25Params::default());
        for leaf in leaves.iter() {
            let bits = exists.filter_bits(leaf, &ctx).unwrap();
            for doc in 0..leaf.num_docs() {
                let has = leaf.value_presence(2).unwrap().contains(&doc);
                assert_eq!(bits[doc as usize / 64] >> (doc % 64) & 1 == 1, has, "doc {doc}");
            }
            assert_eq!(bits.iter().map(|w| w.count_ones()).sum::<u32>(), 500);
        }
        // Every filtered hit has the field; the filter keeps a quarter.
        let hits = knn(&index, &vector(3, 8), Some(&exists));
        assert_eq!(hits.len(), 10);
        let has = |leaf: usize, doc: DocId| leaves[leaf].value_presence(2).unwrap().contains(&doc);
        assert!(hits.iter().all(|&(_, leaf, doc)| has(leaf, doc)));
    }

    /// Microbench (ignored; run with `--ignored --nocapture`, release):
    /// so_vector-like 768-d dot-product vectors, one segment, filtered kNN
    /// k=10/nc=50 p50 for match-all, `exists` on a keyword present on 25%
    /// of docs, and a 10% numeric range; plus the `exists` filter's bits as
    /// built before (one decoded term per doc) and now (presence words).
    /// `RUSTIC_BENCH_DOCS` sets the doc count (default 50,000).
    #[test]
    #[ignore]
    fn bench_filtered_knn_exists_vs_range_768d() {
        let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
        let dim = 768;
        // The default shared budget (256 MiB) is a test default: size it for
        // the f32 rows (graph-resident until sealed and on the store) plus
        // graph and working memory. Raise only, never lower another limit.
        let needed = (docs as u64) * (dim as u64) * 4 * 3 + (1 << 30);
        let budget = crate::admission::shared();
        if budget.limit() < needed { crate::admission::configure_shared(needed); }
        let started = std::time::Instant::now();
        let index = build(docs, dim);
        eprintln!("built {docs} x {dim}d in {:?}", started.elapsed());
        let p50 = |mut times: Vec<std::time::Duration>| { times.sort(); times[times.len() / 2] };
        let run = |name: &str, filter: Option<&Query>| {
            let mut times = Vec::new();
            for q in 0..60 {
                let query = vector(docs + q, dim);
                let t = std::time::Instant::now();
                let hits = knn(&index, &query, filter);
                times.push(t.elapsed());
                assert_eq!(hits.len(), 10, "{name}");
            }
            eprintln!("{name:>24}: p50 {:?}", p50(times));
        };
        run("match_all", None);
        run("exists(accepted) 25%", Some(&Query::Exists { field: 2 }));
        run("range(rnd = 0) 10%", Some(&Query::RangeI64 { field: 3, lower: Bound::Included(0), upper: Bound::Included(0) }));
        // The filter bits alone, old way vs new, on the segment.
        let segment = &index.segments[0];
        let column = segment.fast_fields().get_keyword(2).unwrap();
        let n = segment.num_docs;
        let mut old_times = Vec::new();
        let mut new_times = Vec::new();
        for _ in 0..20 {
            let t = std::time::Instant::now();
            let docs: Vec<DocId> = (0..n).filter(|&d| column.get(d).is_some()).collect();
            let mut words = vec![0u64; (n as usize).div_ceil(64)];
            for d in docs { words[d as usize / 64] |= 1 << (d % 64); }
            old_times.push(t.elapsed());
            std::hint::black_box(&words);
            let t = std::time::Instant::now();
            let fresh = column.presence_words();
            new_times.push(t.elapsed());
            assert_eq!(fresh, words);
        }
        eprintln!("exists bits per query, before (term per doc): p50 {:?}", p50(old_times));
        eprintln!("exists bits, now (presence words, built once per segment): p50 {:?}", p50(new_times));
    }
}

/// Unfiltered mutable kNN with a memtable published (indexing): hits in the
/// memtable resolve through its reserved point ranges, the visible-doc count
/// comes from the segments' point runs and masks, and the result equals the
/// exact search's over the same view (deleted segment docs excluded).
#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod knn_during_indexing_tests {
    use super::*;
    use crate::query::Query;

    #[test]
    fn unfiltered_knn_with_a_memtable_matches_exact_and_resolves_memtable_hits() {
        // The mutable graph needs the field's explicit vector mapping.
        let schema = vec![Field::new(1, "v", FieldType::Vector { dim: 2 }).with_similarity("l2_norm")
            .with_vector_options(Some(crate::vector::store::VectorFieldOptions::f32_hnsw()))];
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "knn-indexing", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let add = |index: &mut Index, rows: std::ops::Range<usize>| for i in rows {
            let mut document = Document::new();
            document.add_vector(1, vec![i as f32, (i % 7) as f32]);
            index.add_document(&document).unwrap();
        };
        add(&mut index, 0..300);
        index.publish().unwrap();
        index.seal().unwrap();
        for doc in (0..300).step_by(5) { index.delete_doc_shared(&DocAddress::new(0, doc)).unwrap(); }
        add(&mut index, 300..360);
        index.publish().unwrap();
        let view = index.reader_view();
        let leaves = view.leaves();
        assert!(leaves.iter().any(|l| matches!(l, LeafReader::Memtable(..))), "a memtable is published");
        let ctx = crate::query::query::SearchContext::for_query(&Query::MatchAll, &leaves, Bm25Params::default());
        for query in [[310.0f32, 2.0], [5.0, 5.0], [299.0, 1.0]] {
            let set = |exact: bool| {
                let mut hits: Vec<(usize, DocId)> = crate::query::mutable_knn::search(&leaves, 1, &query, 20, 100, None, &ctx, exact, None)
                    .unwrap().unwrap().0.into_iter().map(|(_, leaf, doc)| (leaf, doc)).collect();
                hits.sort_unstable();
                hits
            };
            let (graph, exact) = (set(false), set(true));
            assert_eq!(graph, exact, "query {query:?}");
            assert_eq!(graph.len(), 20);
            if query[0] >= 300.0 {
                assert!(graph.iter().any(|&(leaf, _)| matches!(leaves[leaf], LeafReader::Memtable(..))), "memtable hits resolve");
            }
            for &(leaf, doc) in &graph {
                if let LeafReader::Segment(_) = leaves[leaf] { assert!(doc % 5 != 0, "deleted doc {doc} returned"); }
            }
        }
    }

    #[test]
    fn count_bits_counts_a_doc_range() {
        let words = [u64::MAX, 0b1011, u64::MAX];
        assert_eq!(crate::query::mutable_knn::count_bits_for_test(&words, 0, 64), 64);
        assert_eq!(crate::query::mutable_knn::count_bits_for_test(&words, 62, 6), 2 + 3);
        assert_eq!(crate::query::mutable_knn::count_bits_for_test(&words, 64, 4), 3);
        assert_eq!(crate::query::mutable_knn::count_bits_for_test(&words, 10, 150), 54 + 3 + 32);
        assert_eq!(crate::query::mutable_knn::count_bits_for_test(&words, 190, 10), 2);
    }
}

/// A selective filter (under the policy's 2% exact threshold) on a quantized
/// field: the exact route scores codes, gathers k * oversample and rescores
/// them by f32, so returned scores are exact f32 scores, every hit matches
/// the filter, and the top k equals brute force on this data.
#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod selective_filter_exact_tests {
    use super::*;
    use crate::query::Query;
    use crate::vector::store::VectorFieldOptions;

    #[test]
    fn selective_filter_exact_route_rescored_from_codes_matches_brute_force() {
        let dim = 16;
        let row = |i: usize| -> Vec<f32> { (0..dim).map(|d| (((i * 31 + d * 7) % 97) as f32 / 97.0) - 0.5).collect() };
        let schema = vec![
            Field::new(1, "v", FieldType::Vector { dim: dim as u16 }).with_similarity("l2_norm")
                .with_vector_options(Some(VectorFieldOptions { oversample: 3.0, ..VectorFieldOptions::quantized_hnsw(crate::vector::Quantization::Int8) })),
            Field::new(2, "tag", FieldType::Keyword),
        ];
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "selective", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        // 1% match: under the 2% exact threshold, above num_candidates (50),
        // so the policy (not ES's prefilter rule) picks the exact route.
        let docs = 10_000;
        for i in 0..docs {
            let mut document = Document::new();
            document.add_vector(1, row(i));
            if i % 100 == 7 { document.add_keyword(2, "rare"); }
            index.add_document(&document).unwrap();
        }
        index.publish().unwrap();
        index.seal().unwrap();
        let filter = Query::Term(Term::new_text(2, "rare"));
        let view = index.reader_view();
        let leaves = view.leaves();
        let ctx = crate::query::query::SearchContext::for_query(&filter, &leaves, Bm25Params::default());
        for q in [3usize, 1_234, 4_321] {
            let query = row(q + 13);
            let hits = crate::query::mutable_knn::search(&leaves, 1, &query, 10, 50, Some(&filter), &ctx, false, None).unwrap().unwrap().0;
            assert_eq!(hits.len(), 10);
            let mut want: Vec<(f32, usize)> = (0..docs).filter(|i| i % 100 == 7)
                .map(|i| (crate::vector::Similarity::L2.score(&query, &row(i)), i)).collect();
            want.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (rank, &(score, leaf, doc)) in hits.iter().enumerate() {
                let stored = leaves[leaf].with_vector(1, doc, <[f32]>::to_vec).unwrap();
                assert!(leaves[leaf].value_presence(2).unwrap().contains(&doc), "hit outside the filter");
                assert_eq!(score, crate::vector::Similarity::L2.score(&query, &stored), "rank {rank}: an f32 score, not a code estimate");
                assert_eq!(score, want[rank].0, "rank {rank}");
            }
        }
    }
}

/// Filtered-kNN microbench (ignored) shaped after so_vector's filtered tasks:
/// `KNN_BENCH_ONLY=<substring> cargo test --release -p rustic-core --lib
/// filtered_knn_bench -- --ignored --nocapture`. Runs the engine's mutable kNN
/// path (`mutable_knn::search`, as `query/knn` does) on an index whose
/// vector field carries the options ES mappings translate to: `int8_hnsw`
/// (m 16, ef_construction 100, no rescore) and `bbq_hnsw` (m 16,
/// ef_construction 100, `rescore_vector.oversample: 3`).
///
/// Knobs: `KNN_BENCH_DOCS` (200,000), `KNN_BENCH_DIMS` (96; so_vector is
/// 768), `KNN_BENCH_QUERIES` (30), `KNN_BENCH_ONLY` (run only rows whose
/// label contains it, e.g. `bbq`, `merged`, `random`, `k100/nc1000`).
///
/// Rows: quantization x shape (`segments`: one segment per 25k-doc batch,
/// as after bulk load; `merged`: one force-merged segment; `updating`: the
/// segments shape while a writer thread appends and deletes docs with a
/// refresh per 500-doc batch, dense_vector's "knn while indexing") x filter
/// (random term, random range, vector-correlated term at 0.5/2/5/10/20/50%,
/// an acceptedAnswerId-like `exists` on a correlated 25%, and unfiltered) x
/// k/num_candidates (10/50, 10/100, 100/300, 100/1000). Each row prints the
/// route the searches took (recorded by the mutable kNN), p50/p99, and
/// asserts every hit matches the filter; static shapes add vector ops,
/// nodes visited and neighbours rejected by the filter per query (test-only
/// counters) and mean/min recall@k against brute force under the same
/// filter (f32 dot product); the updating shape adds how many queries took
/// at least RUSTIC_KNN_SLOW_MS and their mean phase breakdown. Rows with a
/// so_vector counterpart name it (compare with that task's ES p50).
#[cfg(all(test, feature = "mutable-hnsw-poc"))]
mod filtered_knn_bench {
    use super::*;
    use crate::query::{Bound, Query};
    use crate::vector::store::VectorFieldOptions;
    use crate::vector::Quantization;

    const SELECTIVITIES: [f64; 6] = [0.005, 0.02, 0.05, 0.10, 0.20, 0.50];
    const F_VEC: u16 = 1;
    const F_ID: u16 = 2;
    const F_RANGE: u16 = 3;
    /// Keyword fields 10.. (random, one per selectivity) and 20.. (correlated).
    const F_RANDOM: u16 = 10;
    const F_CORRELATED: u16 = 20;
    /// acceptedAnswerId-like keyword: a distinct value on the ~25% of docs
    /// whose first raw component is above 0.5 (correlated with the vectors).
    const F_EXISTS: u16 = 30;
    fn exists_in(first: f32) -> bool { first > 0.5 }

    fn env(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }
    fn mix(i: u64, salt: u64) -> u64 {
        let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 31; x = x.wrapping_mul(0x94D0_49BB_1331_11EB); x ^= x >> 29;
        x
    }
    /// Unnormalized row (its first component drives the correlated filter)
    /// and the unit vector indexed.
    fn raw(i: usize, dims: usize) -> Vec<f32> {
        (0..dims).map(|d| (mix(i as u64, d as u64 + 1) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0).collect()
    }
    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v { *x /= n; }
        v
    }
    fn random_in(i: usize, s: usize) -> bool { (mix(i as u64, 1_000 + s as u64) % 10_000) < (SELECTIVITIES[s] * 10_000.0) as u64 }
    fn range_value(i: usize) -> i64 { (mix(i as u64, 7_777) % 10_000) as i64 }
    /// The first raw component is uniform in [-1, 1): its top share s.
    fn correlated_in(first: f32, s: usize) -> bool { f64::from(first) > 1.0 - 2.0 * SELECTIVITIES[s] }

    #[derive(Clone, Copy, PartialEq)]
    enum Filter { None, RandomTerm(usize), RandomRange(usize), CorrelatedTerm(usize), ExistsCorrelated }
    impl Filter {
        fn label(self) -> String {
            match self {
                Filter::None => "unfiltered".into(),
                Filter::RandomTerm(s) => format!("random-term {:.1}%", SELECTIVITIES[s] * 100.0),
                Filter::RandomRange(s) => format!("random-range {:.1}%", SELECTIVITIES[s] * 100.0),
                Filter::CorrelatedTerm(s) => format!("correlated-term {:.1}%", SELECTIVITIES[s] * 100.0),
                Filter::ExistsCorrelated => "exists-correlated 25%".into(),
            }
        }
        fn query(self) -> Option<Query> {
            match self {
                Filter::None => None,
                Filter::RandomTerm(s) => Some(Query::Term(Term::new_text(F_RANDOM + s as u16, "y"))),
                Filter::RandomRange(s) => Some(Query::RangeI64 { field: F_RANGE, lower: Bound::Included(0), upper: Bound::Excluded((SELECTIVITIES[s] * 10_000.0) as i64) }),
                Filter::CorrelatedTerm(s) => Some(Query::Term(Term::new_text(F_CORRELATED + s as u16, "y"))),
                Filter::ExistsCorrelated => Some(Query::Exists { field: F_EXISTS }),
            }
        }
        fn matches(self, i: usize, first: f32) -> bool {
            match self {
                Filter::None => true,
                Filter::RandomTerm(s) => random_in(i, s),
                Filter::RandomRange(s) => range_value(i) < (SELECTIVITIES[s] * 10_000.0) as i64,
                Filter::CorrelatedTerm(s) => correlated_in(first, s),
                Filter::ExistsCorrelated => exists_in(first),
            }
        }
        /// The so_vector task this row stands in for, if any.
        fn so_vector(self) -> &'static str {
            match self {
                Filter::RandomRange(3) | Filter::RandomTerm(3) => " [so_vector random 10%]",
                Filter::RandomRange(4) | Filter::RandomTerm(4) => " [so_vector random 20%]",
                Filter::CorrelatedTerm(2) | Filter::RandomTerm(2) => " [so_vector java ~5%]",
                Filter::ExistsCorrelated => " [so_vector acceptedAnswerId]",
                _ => "",
            }
        }
    }

    /// Doc `i` (its vector and first raw component too): every filter's
    /// membership is a function of `i` and that component.
    fn document(i: usize, dims: usize) -> (Document, Vec<f32>, f32) {
        let r = raw(i, dims);
        let first = r[0];
        let v = unit(r);
        let mut document = Document::new();
        document.add_vector(F_VEC, v.clone());
        document.add_i64(F_ID, i as i64);
        document.add_i64(F_RANGE, range_value(i));
        for s in 0..SELECTIVITIES.len() {
            if random_in(i, s) { document.add_keyword(F_RANDOM + s as u16, "y"); }
            if correlated_in(first, s) { document.add_keyword(F_CORRELATED + s as u16, "y"); }
        }
        if exists_in(first) { document.add_keyword(F_EXISTS, &format!("a{i}")); }
        (document, v, first)
    }

    fn build(quantization: Quantization, docs: usize, dims: usize, merged: bool) -> (Index, Vec<Vec<f32>>, Vec<f32>) {
        let options = match quantization {
            Quantization::Binary => VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_hnsw(16, 100).with_oversample(3.0),
            _ => VectorFieldOptions::quantized_hnsw(Quantization::Int8).with_hnsw(16, 100).with_oversample(0.0),
        };
        let mut schema = vec![
            Field::new(F_VEC, "v", FieldType::Vector { dim: dims as u16 }).with_similarity("dot_product").with_vector_options(Some(options)),
            Field::new(F_ID, "id", FieldType::I64),
            Field::new(F_RANGE, "rnd", FieldType::I64),
        ];
        for s in 0..SELECTIVITIES.len() {
            schema.push(Field::new(F_RANDOM + s as u16, &format!("random_{s}"), FieldType::Keyword));
            schema.push(Field::new(F_CORRELATED + s as u16, &format!("correlated_{s}"), FieldType::Keyword));
        }
        schema.push(Field::new(F_EXISTS, "accepted", FieldType::Keyword));
        let mut index = Index::create_in(Arc::new(RamDirectory::default()), "filtered-knn-bench", schema, IndexSettings::default()).unwrap();
        index.set_nrt(true);
        index.enable_mutable_vectors().unwrap();
        let mut vectors = Vec::with_capacity(docs);
        let mut firsts = Vec::with_capacity(docs);
        for i in 0..docs {
            let (document, v, first) = document(i, dims);
            index.add_document(&document).unwrap();
            vectors.push(v);
            firsts.push(first);
            // Bulk-load shape: a segment per 25k docs.
            if (i + 1) % 25_000 == 0 { index.publish().unwrap(); index.seal().unwrap(); }
        }
        index.publish().unwrap();
        index.seal().unwrap();
        if merged {
            index.force_merge().unwrap();
            index.force_merge_vectors(1, false).unwrap();
            index.publish().unwrap();
        }
        (index, vectors, firsts)
    }

    /// The policy's exact route (filter below `exact_lower_limit`, more
    /// matches than `num_candidates`) on an int8 field without rescore
    /// scans the code plane, not every filtered node's f32 row (vec1b: 4-25 s
    /// a shard of cold f32 reads at 1%), and ranks the gathered
    /// `num_candidates` by f32: the f32 top k of the filter.
    #[test]
    fn selective_filter_on_unrescored_int8_scans_codes_for_the_f32_top_k() {
        let (docs, dims, k, nc) = (30_000, 32, 10, 50);
        let (index, vectors, firsts) = build(Quantization::Int8, docs, dims, true);
        let view = index.reader_view();
        let leaves = view.leaves();
        let id_of = |leaf: usize, doc: DocId| leaves[leaf].i64_value(F_ID, doc).expect("id column") as usize;
        let filter = Filter::RandomRange(0);
        let matching: Vec<usize> = (0..docs).filter(|&i| filter.matches(i, firsts[i])).collect();
        assert!(matching.len() > nc && (matching.len() as f64) < 0.02 * docs as f64, "{} matches", matching.len());
        let query = filter.query();
        let ctx = crate::query::query::SearchContext::for_query(query.as_ref().unwrap(), &leaves, Bm25Params::default());
        let mut recall = 0.0;
        for q in 0..10 {
            let qv = unit(raw(60_000_000 + q, dims));
            let mut truth: Vec<(f32, usize)> = matching.iter().map(|&i| (crate::vector::dot(&qv, &vectors[i]), i)).collect();
            truth.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let _ = crate::vector::mutable_ingest::take_exact_phases();
            let (hits, _) = crate::query::mutable_knn::search(&leaves, F_VEC, &qv, k, nc, query.as_ref(), &ctx, false, None)
                .unwrap().expect("a mutable graph field");
            assert_eq!(crate::vector::hnsw::last_route(), "exact");
            assert!(crate::vector::mutable_ingest::take_exact_phases()[3] > 0.0, "the code plane was not scanned");
            assert_eq!(hits.len(), k);
            let found: Vec<usize> = hits.iter().map(|&(_, leaf, doc)| id_of(leaf, doc)).collect();
            for &i in &found { assert!(filter.matches(i, firsts[i]), "hit {i} outside the filter"); }
            // f32 scores, best first.
            for pair in hits.windows(2) { assert!(pair[0].0 >= pair[1].0); }
            recall += truth[..k].iter().filter(|(_, i)| found.contains(i)).count() as f64 / k as f64;
        }
        assert!(recall / 10.0 >= 0.98, "recall {}", recall / 10.0);
    }

    #[test]
    #[ignore]
    fn filtered_knn_bench() {
        let docs = env("KNN_BENCH_DOCS", 200_000);
        let dims = env("KNN_BENCH_DIMS", 96);
        let queries = env("KNN_BENCH_QUERIES", 30);
        let only = std::env::var("KNN_BENCH_ONLY").ok();
        let wanted = |label: &str| only.as_deref().is_none_or(|o| label.contains(o));
        // f32 rows, graph and codes for the shard: lift the test-default budget.
        let needed = (docs as u64) * (dims as u64) * 4 * 4 + (1 << 30);
        if crate::admission::shared().limit() < needed { crate::admission::configure_shared(needed); }
        let mut filters = vec![Filter::None, Filter::ExistsCorrelated];
        for s in 0..SELECTIVITIES.len() { filters.extend([Filter::RandomTerm(s), Filter::RandomRange(s), Filter::CorrelatedTerm(s)]); }
        let configs = [(10usize, 50usize), (10, 100), (100, 300), (100, 1000)];
        let qvecs: Vec<Vec<f32>> = (0..queries).map(|q| unit(raw(50_000_000 + q, dims))).collect();
        for (qname, quantization) in [("int8_hnsw", Quantization::Int8), ("bbq_hnsw os3", Quantization::Binary)] {
            for (shape, merged) in [("segments", false), ("merged", true), ("updating", false)] {
                let prefix = format!("{qname} {shape}");
                if !filters.iter().any(|f| configs.iter().any(|(k, nc)| wanted(&format!("{prefix} {} k{k}/nc{nc}", f.label())))) { continue; }
                let started = std::time::Instant::now();
                let (mut index, vectors, firsts) = build(quantization, docs, dims, merged);
                eprintln!("== {prefix}: {docs} x {dims}d built in {:?}", started.elapsed());
                if shape == "updating" {
                    run_updating(&prefix, &mut index, docs, dims, &qvecs, &filters, &configs, &wanted);
                    continue;
                }
                let view = index.reader_view();
                let leaves = view.leaves();
                let id_of = |leaf: usize, doc: DocId| leaves[leaf].i64_value(F_ID, doc).expect("id column") as usize;
                for &filter in &filters {
                    let matching: Vec<usize> = (0..docs).filter(|&i| filter.matches(i, firsts[i])).collect();
                    let query = filter.query();
                    let ctx = crate::query::query::SearchContext::for_query(query.as_ref().unwrap_or(&Query::MatchAll), &leaves, Bm25Params::default());
                    // Brute force under the same filter, once per query.
                    let truth: Vec<Vec<usize>> = qvecs.iter().map(|q| {
                        let mut scored: Vec<(f32, usize)> = matching.iter().map(|&i| (crate::vector::dot(q, &vectors[i]), i)).collect();
                        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                        scored.truncate(1000);
                        scored.into_iter().map(|(_, i)| i).collect()
                    }).collect();
                    for &(k, nc) in &configs {
                        let label = format!("{prefix} {} k{k}/nc{nc}", filter.label());
                        if !wanted(&label) { continue; }
                        let mut times = Vec::with_capacity(queries);
                        let (mut ops, mut recall_sum, mut recall_min) = (0u64, 0.0f64, 1.0f64);
                        let counts_before = crate::vector::hnsw::search_counts();
                        let mut routes: Vec<&'static str> = Vec::new();
                        for (q, qv) in qvecs.iter().enumerate() {
                            let t = std::time::Instant::now();
                            let (hits, stats) = crate::query::mutable_knn::search(&leaves, F_VEC, qv, k, nc, query.as_ref(), &ctx, false, None)
                                .unwrap().expect("a mutable graph field");
                            times.push(t.elapsed());
                            let route = crate::vector::hnsw::last_route();
                            if !routes.contains(&route) { routes.push(route); }
                            ops += stats.vector_ops;
                            let found: std::collections::HashSet<usize> = hits.iter().map(|&(_, leaf, doc)| id_of(leaf, doc)).collect();
                            for &i in &found { assert!(filter.matches(i, firsts[i]), "{label}: hit {i} outside the filter"); }
                            let want = &truth[q][..k.min(truth[q].len())];
                            let recall = if want.is_empty() { 1.0 } else { want.iter().filter(|i| found.contains(i)).count() as f64 / want.len() as f64 };
                            recall_sum += recall;
                            recall_min = recall_min.min(recall);
                        }
                        let counts_after = crate::vector::hnsw::search_counts();
                        let n = queries.max(1) as u64;
                        let (p50, p99) = percentiles(times);
                        eprintln!("{label:<60} route {:<14} p50 {:>8.3} ms  p99 {:>8.3} ms  ops/q {:>7}  visited/q {:>7}  rejected/q {:>7}  recall {:.3} (min {:.2}){}",
                            routes.join("+"), ms(p50), ms(p99), ops / n,
                            (counts_after.0 - counts_before.0) / n, (counts_after.1 - counts_before.1) / n,
                            recall_sum / n as f64, recall_min, filter.so_vector());
                    }
                }
            }
        }
    }

    /// so_vector's filtered tasks on the track's own data (ignored):
    /// `RUSTIC_SO_DIR` holds `vectors.f32` (n x 768, little-endian),
    /// `qid.i64` (questionId), `flags.u8` (bit 0: tagged `java`, bit 1: has
    /// an acceptedAnswerId) and `queries.f32` (the track's query vectors),
    /// converted from documents_base64.json (every other document: one
    /// shard's worth of the 2-shard index). The field is the track's
    /// `bbq_hnsw` (m 16, ef_construction 100, oversample 3). Shapes as in
    /// `filtered_knn_bench` (segments of 25k docs, then force-merged); rows
    /// are the track's filters (match-all, questionId <= 650000 / 1500000,
    /// tags: java, exists acceptedAnswerId) at its k/num_candidates, with
    /// route, p50/p99, ops, visited/rejected and recall@k against f32 brute
    /// force under the same filter. `RUSTIC_SO_DOCS` caps the documents,
    /// `KNN_BENCH_ONLY` selects rows.
    #[test]
    #[ignore]
    fn so_vector_bench() {
        use rayon::prelude::*;
        const DIMS: usize = 768;
        const F_QID: u16 = 4;
        const F_JAVA: u16 = 5;
        // Needs the converted track data: skip (a filter matching both
        // benches, e.g. `filtered_knn_bench`, runs this one too).
        let Ok(dir) = std::env::var("RUSTIC_SO_DIR") else {
            eprintln!("so_vector_bench: RUSTIC_SO_DIR unset, skipped");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
        let floats = |bytes: Vec<u8>| -> Vec<f32> { bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let all = floats(read("vectors.f32"));
        let docs = (all.len() / DIMS).min(env("RUSTIC_SO_DOCS", usize::MAX));
        let vectors: Vec<&[f32]> = all.chunks_exact(DIMS).take(docs).collect();
        let qids: Vec<i64> = read("qid.i64").chunks_exact(8).take(docs).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        let flags = read("flags.u8");
        let queries: Vec<Vec<f32>> = floats(read("queries.f32")).chunks_exact(DIMS).map(<[f32]>::to_vec).collect();
        let only = std::env::var("KNN_BENCH_ONLY").ok();
        let wanted = |label: &str| only.as_deref().is_none_or(|o| label.contains(o));
        // Rows, codes, graph and a force-merge's copy of the vectors.
        let needed = ((docs as u64) * (DIMS as u64) * 4 * 8 + (2 << 30)).max(20 << 30);
        if crate::admission::shared().limit() < needed { crate::admission::configure_shared(needed); }
        let filters: [(&str, Option<Query>, &(dyn Fn(usize) -> bool + Sync)); 5] = [
            ("match-all", None, &|_| true),
            ("random-10-percent (questionId <= 650000)", Some(Query::RangeI64 { field: F_QID, lower: Bound::Unbounded, upper: Bound::Included(650_000) }), &|i| qids[i] <= 650_000),
            ("random-20-percent (questionId <= 1500000)", Some(Query::RangeI64 { field: F_QID, lower: Bound::Unbounded, upper: Bound::Included(1_500_000) }), &|i| qids[i] <= 1_500_000),
            ("java (tags: java)", Some(Query::Term(Term::new_text(F_JAVA, "java"))), &|i| flags[i] & 1 != 0),
            ("acceptedAnswerId (exists)", Some(Query::Exists { field: F_EXISTS }), &|i| flags[i] & 2 != 0),
        ];
        for (name, _, matches) in &filters {
            let n = (0..docs).filter(|&i| matches(i)).count();
            eprintln!("filter {name}: {n} of {docs} docs ({:.1}%)", n as f64 * 100.0 / docs as f64);
        }
        let configs = [(10usize, 50usize), (100, 300)];
        // Brute force under each filter (f32 dot product), once.
        let truth: Vec<Vec<Vec<usize>>> = filters.iter().map(|(_, _, matches)| queries.par_iter().map(|q| {
            let mut scored: Vec<(f32, usize)> = (0..docs).filter(|&i| matches(i)).map(|i| (crate::vector::dot(q, vectors[i]), i)).collect();
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            scored.truncate(100);
            scored.into_iter().map(|(_, i)| i).collect()
        }).collect()).collect();
        // The ground truth for other engines' runs (ES on the same data):
        // per filter, per query, the top 100 document ordinals (u32); only
        // for the whole data set (a capped run would overwrite it).
        for (fi, per_query) in truth.iter().enumerate().filter(|_| std::env::var("RUSTIC_SO_DOCS").is_err()) {
            let bytes: Vec<u8> = per_query.iter().flat_map(|ids| (0..100).map(move |r| ids.get(r).map_or(u32::MAX, |&i| i as u32))).flat_map(u32::to_le_bytes).collect();
            std::fs::write(dir.join(format!("truth_{fi}.u32")), bytes).unwrap();
        }
        for (shape, merged) in [("segments", false), ("force-merged", true)] {
            if !filters.iter().any(|(name, _, _)| configs.iter().any(|(k, nc)| wanted(&format!("{shape} {name} k{k}/nc{nc}")))) { continue; }
            let started = std::time::Instant::now();
            let options = VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_hnsw(16, 100).with_oversample(3.0);
            let schema = vec![
                Field::new(F_VEC, "titleVector", FieldType::Vector { dim: DIMS as u16 }).with_similarity("dot_product").with_vector_options(Some(options)),
                Field::new(F_ID, "id", FieldType::I64),
                Field::new(F_QID, "questionId", FieldType::I64),
                Field::new(F_JAVA, "tags", FieldType::Keyword),
                Field::new(F_EXISTS, "acceptedAnswerId", FieldType::Keyword),
            ];
            let mut index = Index::create_in(Arc::new(RamDirectory::default()), "so-vector-bench", schema, IndexSettings::default()).unwrap();
            index.set_nrt(true);
            index.enable_mutable_vectors().unwrap();
            for i in 0..docs {
                let mut document = Document::new();
                document.add_vector(F_VEC, vectors[i].to_vec());
                document.add_i64(F_ID, i as i64);
                document.add_i64(F_QID, qids[i]);
                if flags[i] & 1 != 0 { document.add_keyword(F_JAVA, "java"); }
                if flags[i] & 2 != 0 { document.add_keyword(F_EXISTS, &format!("a{i}")); }
                index.add_document(&document).unwrap();
                if (i + 1) % 25_000 == 0 { index.publish().unwrap(); index.seal().unwrap(); }
            }
            index.publish().unwrap();
            index.seal().unwrap();
            if merged {
                index.force_merge().unwrap();
                index.force_merge_vectors(1, false).unwrap();
                index.publish().unwrap();
            }
            eprintln!("== so_vector {shape}: {docs} x {DIMS}d bbq_hnsw built in {:?}", started.elapsed());
            let view = index.reader_view();
            let leaves = view.leaves();
            let id_of = |leaf: usize, doc: DocId| leaves[leaf].i64_value(F_ID, doc).expect("id column") as usize;
            for (fi, (name, query, matches)) in filters.iter().enumerate() {
                let ctx = crate::query::query::SearchContext::for_query(query.as_ref().unwrap_or(&Query::MatchAll), &leaves, Bm25Params::default());
                for &(k, nc) in &configs {
                    let label = format!("{shape} {name} k{k}/nc{nc}");
                    if !wanted(&label) { continue; }
                    // One warm-up pass (projection caches, page cache), then measured.
                    for qv in &queries { let _ = crate::query::mutable_knn::search(&leaves, F_VEC, qv, k, nc, query.as_ref(), &ctx, false, None).unwrap(); }
                    let mut times = Vec::with_capacity(queries.len());
                    let (mut ops, mut recall_sum, mut recall_min) = (0u64, 0.0f64, 1.0f64);
                    let counts_before = crate::vector::hnsw::search_counts();
                    let mut routes: Vec<&'static str> = Vec::new();
                    for (q, qv) in queries.iter().enumerate() {
                        let t = std::time::Instant::now();
                        let (hits, stats) = crate::query::mutable_knn::search(&leaves, F_VEC, qv, k, nc, query.as_ref(), &ctx, false, None)
                            .unwrap().expect("a mutable graph field");
                        times.push(t.elapsed());
                        let route = crate::vector::hnsw::last_route();
                        if !routes.contains(&route) { routes.push(route); }
                        ops += stats.vector_ops;
                        let found: std::collections::HashSet<usize> = hits.iter().map(|&(_, leaf, doc)| id_of(leaf, doc)).collect();
                        for &i in &found { assert!(matches(i), "{label}: hit {i} outside the filter"); }
                        let want = &truth[fi][q][..k.min(truth[fi][q].len())];
                        let recall = if want.is_empty() { 1.0 } else { want.iter().filter(|i| found.contains(i)).count() as f64 / want.len() as f64 };
                        recall_sum += recall;
                        recall_min = recall_min.min(recall);
                    }
                    let counts_after = crate::vector::hnsw::search_counts();
                    let n = queries.len().max(1) as u64;
                    let (p50, p99) = percentiles(times);
                    eprintln!("{label:<62} route {:<14} p50 {:>8.3} ms  p99 {:>8.3} ms  ops/q {:>7}  visited/q {:>7}  rejected/q {:>7}  recall {:.3} (min {:.2})",
                        routes.join("+"), ms(p50), ms(p99), ops / n,
                        (counts_after.0 - counts_before.0) / n, (counts_after.1 - counts_before.1) / n,
                        recall_sum / n as f64, recall_min);
                }
            }
        }
    }

    fn ms(d: std::time::Duration) -> f64 { d.as_secs_f64() * 1e3 }
    fn percentiles(mut times: Vec<std::time::Duration>) -> (std::time::Duration, std::time::Duration) {
        times.sort();
        (times[times.len() / 2], times[(times.len() * 99 / 100).min(times.len() - 1)])
    }

    /// Shape (c), dense_vector's "knn while indexing": a writer thread keeps
    /// appending docs (and deleting old ones, so delete repair runs), with a
    /// refresh after every 500-doc batch and a seal every fourth, while the
    /// rows search the latest published view. The data changes under the
    /// searches, so rows report latency (p50/p99), routes and the filter
    /// check, not recall; plus how many queries took at least
    /// `RUSTIC_KNN_SLOW_MS` and their mean phase breakdown.
    #[allow(clippy::too_many_arguments)]
    fn run_updating(prefix: &str, index: &mut Index, docs: usize, dims: usize, qvecs: &[Vec<f32>],
        filters: &[Filter], configs: &[(usize, usize)], wanted: &dyn Fn(&str) -> bool) {
        let slot = index.view_slot().clone();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                let mut next = docs;
                let mut batches = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    for i in next..next + 500 {
                        let (document, _, _) = document(i, dims);
                        index.add_document(&document).unwrap();
                    }
                    // Delete as many old docs from the first segment (updates).
                    if !index.segments.is_empty() {
                        let n = index.segments[0].num_docs;
                        for j in 0..500u32 { let _ = index.delete_doc_shared(&DocAddress::new(0, (mix(next as u64, u64::from(j)) % u64::from(n.max(1))) as u32)); }
                    }
                    next += 500;
                    batches += 1;
                    index.publish().unwrap();
                    if batches % 4 == 0 { index.seal().unwrap(); }
                }
                next - docs
            });
            *crate::query::mutable_knn::SLOW_PHASES.lock() = (0, [0.0; 5]);
            for &filter in filters {
                let query = filter.query();
                for &(k, nc) in configs {
                    let label = format!("{prefix} {} k{k}/nc{nc}", filter.label());
                    if !wanted(&label) { continue; }
                    let mut times = Vec::with_capacity(qvecs.len());
                    let mut routes: Vec<&'static str> = Vec::new();
                    for qv in qvecs {
                        let view = slot.load();
                        let leaves = view.leaves();
                        let ctx = crate::query::query::SearchContext::for_query(query.as_ref().unwrap_or(&Query::MatchAll), &leaves, Bm25Params::default());
                        let t = std::time::Instant::now();
                        let (hits, _) = crate::query::mutable_knn::search(&leaves, F_VEC, qv, k, nc, query.as_ref(), &ctx, false, None)
                            .unwrap().expect("a mutable graph field");
                        times.push(t.elapsed());
                        let route = crate::vector::hnsw::last_route();
                        if !routes.contains(&route) { routes.push(route); }
                        for &(_, leaf, doc) in &hits {
                            let i = leaves[leaf].i64_value(F_ID, doc).expect("id column") as usize;
                            let first = raw(i, dims)[0];
                            assert!(filter.matches(i, first), "{label}: hit {i} outside the filter");
                        }
                    }
                    let (p50, p99) = percentiles(times);
                    eprintln!("{label:<60} route {:<14} p50 {:>8.3} ms  p99 {:>8.3} ms{}", routes.join("+"), ms(p50), ms(p99), filter.so_vector());
                }
            }
            stop.store(true, std::sync::atomic::Ordering::Release);
            let written = writer.join().unwrap();
            let (slow, sums) = *crate::query::mutable_knn::SLOW_PHASES.lock();
            let mean = |i: usize| if slow == 0 { 0.0 } else { sums[i] / slow as f64 };
            eprintln!("{prefix}: {written} docs written meanwhile; {slow} queries >= RUSTIC_KNN_SLOW_MS, mean ms: masks {:.2} projection {:.2} owner-lock {:.2} graph {:.2} finish {:.2}",
                mean(0), mean(1), mean(2), mean(3), mean(4));
        });
    }
}
